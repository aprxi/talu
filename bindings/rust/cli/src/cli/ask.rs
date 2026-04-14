use std::env;
use std::io::{self, Read, Write};
use std::path::PathBuf;

use anyhow::{anyhow, bail, Result};
use indicatif::{ProgressBar, ProgressStyle};
use serde_json;
use talu::responses::ResponsesView;

use talu::error::last_error_message;
use talu::{ChatHandle, InferenceBackend, StorageError, StorageHandle};

use crate::provider::{get_provider, parse_model_target, ModelTarget};

use super::repo::{resolve_model_for_inference, UnifiedProgressCtx};
use super::sessions::{print_sessions_with_stats, resolve_session_target, show_session_transcript};
use super::util::{truncate_str, DEFAULT_MAX_TOKENS};
use super::{AskArgs, AskOutputFormat};

use super::models::list_provider_models;

pub(super) const DEFAULT_SYSTEM_MESSAGE: &str = "You are a helpful assistant.";
const DEFAULT_STDIN_IMAGE_PROMPT: &str = "Describe this image.";
const STREAM_FLUSH_BYTES: usize = 512;

fn has_visible_text(text: &str) -> bool {
    text.chars().any(|ch| {
        if ch.is_ascii() {
            !ch.is_ascii_whitespace() && !ch.is_ascii_control()
        } else {
            !ch.is_whitespace()
        }
    })
}

fn should_flush_stream_output(
    token_text: &str,
    pending_flush_bytes: usize,
    flush_each_token: bool,
) -> bool {
    flush_each_token || token_text.contains('\n') || pending_flush_bytes >= STREAM_FLUSH_BYTES
}

struct StreamCtx {
    raw_output: bool,
    hide_thinking: bool,
    flush_each_token: bool,
    in_reasoning: bool,
    emitted_visible: bool,
    prefill_spinner: Option<indicatif::ProgressBar>,
    pending_flush_bytes: usize,
}

impl StreamCtx {
    fn new(raw_output: bool, hide_thinking: bool, flush_each_token: bool) -> Self {
        Self {
            raw_output,
            hide_thinking,
            flush_each_token,
            in_reasoning: false,
            emitted_visible: false,
            prefill_spinner: None,
            pending_flush_bytes: 0,
        }
    }

    fn on_token(&mut self, token: &talu::router::StreamToken) {
        // Complete prefill progress on first token
        if let Some(spinner) = self.prefill_spinner.take() {
            spinner.finish_with_message("done");
        }

        if self.hide_thinking && token.item_type == talu::responses::ItemType::Reasoning {
            return;
        }

        if !self.raw_output {
            let is_reasoning = token.item_type == talu::responses::ItemType::Reasoning;

            // Emit ANSI transitions on reasoning boundary changes
            if is_reasoning && !self.in_reasoning {
                // Entering reasoning: dim + italic
                let _ = io::stdout().write_all(b"\x1b[2;3m");
                self.pending_flush_bytes += 6;
            } else if !is_reasoning && self.in_reasoning {
                // Leaving reasoning: reset
                let _ = io::stdout().write_all(b"\x1b[0m");
                self.pending_flush_bytes += 4;
            }
            self.in_reasoning = is_reasoning;
        }

        let _ = io::stdout().write_all(token.text.as_bytes());
        self.pending_flush_bytes += token.text.len();
        if should_flush_stream_output(&token.text, self.pending_flush_bytes, self.flush_each_token)
        {
            let _ = io::stdout().flush();
            self.pending_flush_bytes = 0;
        }
        if has_visible_text(token.text) {
            self.emitted_visible = true;
        }
    }

    fn flush(&mut self) {
        // Complete prefill progress if generation produced no tokens
        if let Some(spinner) = self.prefill_spinner.take() {
            spinner.finish_with_message("done");
        }
        if !self.raw_output && self.in_reasoning {
            // Reset ANSI if still in reasoning at end
            let _ = io::stdout().write_all(b"\x1b[0m");
            self.pending_flush_bytes += 4;
        }
        let _ = io::stdout().write_all(b"\n");
        self.pending_flush_bytes += 1;
        let _ = io::stdout().flush();
        self.pending_flush_bytes = 0;
    }
}

pub(super) fn reasoning_text_for_item(conv: &impl ResponsesView, index: usize) -> Result<String> {
    let reasoning = conv.get_reasoning(index)?;
    let mut out = String::new();
    // Content parts (populated by direct generation).
    for part_index in 0..reasoning.content_count {
        let part = conv.get_reasoning_content(index, part_index)?;
        out.push_str(&part.data_utf8_lossy());
    }
    // Fall back to summary parts (populated by load_responses_json).
    if out.is_empty() {
        for part_index in 0..reasoning.summary_count {
            let part = conv.get_reasoning_summary(index, part_index)?;
            out.push_str(&part.data_utf8_lossy());
        }
    }
    Ok(out)
}

pub(super) fn latest_visible_text(
    chat: &ChatHandle,
    include_reasoning: bool,
) -> Result<Option<String>> {
    let conv = chat.responses();
    let count = conv.item_count();
    if count == 0 {
        return Ok(None);
    }

    for index in (0..count).rev() {
        if conv.item_type(index) == talu::responses::ItemType::Message {
            let msg = conv.get_message(index)?;
            if msg.role != talu::responses::MessageRole::Assistant {
                continue;
            }

            let message_text = conv.message_text(index)?;
            if !include_reasoning {
                return Ok(Some(message_text));
            }

            let mut reasoning_segments = Vec::new();
            let mut cursor = index;
            while cursor > 0 {
                let prev = cursor - 1;
                if conv.item_type(prev) != talu::responses::ItemType::Reasoning {
                    break;
                }
                reasoning_segments.push(reasoning_text_for_item(&conv, prev)?);
                cursor = prev;
            }

            reasoning_segments.reverse();
            let mut out = String::new();
            for segment in reasoning_segments {
                out.push_str(&segment);
            }
            out.push_str(&message_text);
            return Ok(Some(out));
        }
    }

    if include_reasoning {
        let mut reasoning_segments = Vec::new();
        let mut cursor = count;
        while cursor > 0 {
            let idx = cursor - 1;
            if conv.item_type(idx) != talu::responses::ItemType::Reasoning {
                break;
            }
            reasoning_segments.push(reasoning_text_for_item(&conv, idx)?);
            cursor = idx;
        }

        if !reasoning_segments.is_empty() {
            reasoning_segments.reverse();
            let mut out = String::new();
            for segment in reasoning_segments {
                out.push_str(&segment);
            }
            return Ok(Some(out));
        }
    }

    Ok(None)
}

#[derive(Debug)]
struct ParsedStdin {
    text: Option<String>,
    images: Vec<talu::router::ContentPart>,
}

fn map_stdin_image_mime(info: &talu::file::FileInfo) -> String {
    if !info.mime.is_empty() && info.mime.starts_with("image/") {
        return info.mime.clone();
    }
    match info.image.map(|image| image.format) {
        Some(talu::file::ImageFormat::Jpeg) => "image/jpeg".to_string(),
        Some(talu::file::ImageFormat::Png) => "image/png".to_string(),
        Some(talu::file::ImageFormat::Webp) => "image/webp".to_string(),
        _ => "application/octet-stream".to_string(),
    }
}

fn trim_trailing_stdin_whitespace(bytes: &mut Vec<u8>) {
    while matches!(bytes.last(), Some(b'\n' | b'\r' | b' ')) {
        bytes.pop();
    }
}

fn encode_base64(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    let mut i = 0usize;
    while i + 3 <= bytes.len() {
        let b0 = bytes[i];
        let b1 = bytes[i + 1];
        let b2 = bytes[i + 2];
        out.push(TABLE[(b0 >> 2) as usize] as char);
        out.push(TABLE[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
        out.push(TABLE[(((b1 & 0x0f) << 2) | (b2 >> 6)) as usize] as char);
        out.push(TABLE[(b2 & 0x3f) as usize] as char);
        i += 3;
    }

    let rem = bytes.len() - i;
    if rem == 1 {
        let b0 = bytes[i];
        out.push(TABLE[(b0 >> 2) as usize] as char);
        out.push(TABLE[((b0 & 0x03) << 4) as usize] as char);
        out.push('=');
        out.push('=');
    } else if rem == 2 {
        let b0 = bytes[i];
        let b1 = bytes[i + 1];
        out.push(TABLE[(b0 >> 2) as usize] as char);
        out.push(TABLE[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
        out.push(TABLE[((b1 & 0x0f) << 2) as usize] as char);
        out.push('=');
    }

    out
}

fn parse_stdin_content(mut stdin_buf: Vec<u8>) -> Result<ParsedStdin> {
    if stdin_buf.is_empty() {
        return Ok(ParsedStdin {
            text: None,
            images: vec![],
        });
    }

    if let Ok(info) = talu::file::inspect_bytes(&stdin_buf) {
        if info.kind == talu::file::FileKind::Image {
            let mime = map_stdin_image_mime(&info);
            let data_url = format!("data:{};base64,{}", mime, encode_base64(&stdin_buf));
            return Ok(ParsedStdin {
                text: None,
                images: vec![talu::router::ContentPart::ImageUrl {
                    url: data_url,
                    mime: Some(mime),
                }],
            });
        }

        // Auto-convert PDF pages to images for VLM consumption.
        if info.kind == talu::file::FileKind::Document && info.mime == "application/pdf" {
            let count = talu::file::pdf_page_count(&stdin_buf)
                .map_err(|e| anyhow!("Failed to get PDF page count: {e}"))?;
            let mut images = Vec::with_capacity(count as usize);
            for i in 0..count {
                let opts = talu::file::TransformOptions {
                    output_format: Some(talu::file::OutputFormat::Png),
                    ..Default::default()
                };
                let rendered = talu::file::pdf_transform_page(&stdin_buf, i, 150, opts)
                    .map_err(|e| anyhow!("Failed to render PDF page {}: {e}", i + 1))?;
                let data_url = format!("data:image/png;base64,{}", encode_base64(&rendered.bytes));
                images.push(talu::router::ContentPart::ImageUrl {
                    url: data_url,
                    mime: Some("image/png".into()),
                });
            }
            return Ok(ParsedStdin { text: None, images });
        }

        if !info.mime.starts_with("text/") && stdin_buf.contains(&0) {
            bail!(
                "Error: stdin contains binary data ({}) and cannot be used as text prompt.",
                if info.mime.is_empty() {
                    "unknown MIME"
                } else {
                    info.mime.as_str()
                }
            );
        }
    }

    trim_trailing_stdin_whitespace(&mut stdin_buf);
    if stdin_buf.is_empty() {
        return Ok(ParsedStdin {
            text: None,
            images: vec![],
        });
    }
    let text = String::from_utf8_lossy(&stdin_buf).to_string();
    if text.contains('\0') {
        bail!("Error: stdin contains NUL bytes and cannot be used as text prompt.");
    }
    Ok(ParsedStdin {
        text: Some(text),
        images: vec![],
    })
}

/// Chat: new session by default; append via --session.
pub(super) fn cmd_ask(args: AskArgs, stdin_is_pipe: bool, verbose: u8) -> Result<()> {
    let no_chat = args.no_chat;
    let use_json = args.format == Some(AskOutputFormat::Json);
    let session_id_only = args.session_id_only;
    let quiet = (args.quiet && !use_json) || session_id_only;
    let silent = args.silent;
    let raw_output = args.raw;
    let hide_thinking = args.hide_thinking;
    let mut system_msg = args.system.clone();
    let mut prompt_parts = args.prompt.clone();
    let mut stdin_image_parts: Vec<talu::router::ContentPart> = Vec::new();
    let endpoint_url_override = args.endpoint_url.clone();
    let seed = args.seed.unwrap_or(0);
    let db_path =
        crate::config::resolve_bucket(args.no_bucket, args.bucket.clone(), &args.profile)?;
    let session_env = env::var("SESSION_ID").ok().filter(|s| !s.is_empty());

    if silent && args.quiet {
        bail!("Error: cannot specify both --quiet and --silent.");
    }

    if args.new {
        if args.no_bucket {
            bail!("Error: --new cannot be used with --no-bucket.");
        }
        if args.delete {
            bail!("Error: --new cannot be used with --delete.");
        }
        if session_id_only {
            bail!("Error: --new cannot be used with --session-id.");
        }
        if args.session.is_some() || session_env.is_some() {
            bail!("Error: --new cannot be used with --session or SESSION_ID.");
        }
    }
    if session_id_only {
        if args.no_bucket {
            bail!("Error: --session-id cannot be used with --no-bucket.");
        }
        if args.delete {
            bail!("Error: --session-id cannot be used with --delete.");
        }
        if args.new {
            bail!("Error: --session-id cannot be used with --new.");
        }
        if args.output.is_some() {
            bail!("Error: --session-id cannot be used with --output.");
        }
        if use_json {
            bail!("Error: --session-id cannot be used with --format json.");
        }
        if args.silent {
            bail!("Error: --session-id cannot be used with --silent.");
        }
        if args.session.is_some() {
            bail!("Error: --session-id cannot be used with --session.");
        }
        // --session-id takes precedence over SESSION_ID.
    }

    let output_path = args.output.clone();
    let no_stream =
        args.no_stream || use_json || silent || output_path.is_some() || session_id_only;
    let emit_output = |text: &str| -> Result<()> {
        if session_id_only {
            return Ok(());
        }
        if let Some(path) = &output_path {
            std::fs::write(path, format!("{text}\n"))?;
            return Ok(());
        }
        if silent {
            return Ok(());
        }
        println!("{text}");
        Ok(())
    };

    // --each-line: collect stdin lines as separate prompts for batched decode.
    let mut each_line_prompts: Vec<String> = Vec::new();

    if stdin_is_pipe {
        let mut stdin_buf = Vec::new();
        if io::stdin().read_to_end(&mut stdin_buf).is_ok() && !stdin_buf.is_empty() {
            if args.each_line {
                // Each non-empty line becomes a separate prompt.
                let text = String::from_utf8_lossy(&stdin_buf);
                let suffix = prompt_parts.join(" ");
                for line in text.lines() {
                    let trimmed = line.trim();
                    if !trimmed.is_empty() {
                        if suffix.is_empty() {
                            each_line_prompts.push(trimmed.to_string());
                        } else {
                            each_line_prompts.push(format!("{}\n{}", trimmed, suffix));
                        }
                    }
                }
            } else {
                let parsed = parse_stdin_content(stdin_buf)?;
                if let Some(text) = parsed.text {
                    prompt_parts.push(text);
                }
                stdin_image_parts = parsed.images;
            }
        }
    }

    if !stdin_image_parts.is_empty() && prompt_parts.is_empty() {
        prompt_parts.push(DEFAULT_STDIN_IMAGE_PROMPT.to_string());
    }

    let prompt = prompt_parts.join(" ");

    if args.new {
        if !prompt.is_empty() {
            bail!("Error: --new does not accept a prompt.");
        }
        let db = match db_path {
            Some(ref p) => p,
            None => bail!("Error: Storage is disabled. Nothing to create."),
        };
        let session_id = talu::responses::new_session_id()?;
        let chat = ChatHandle::new(None)?;
        let db_str = db.to_string_lossy();
        chat.set_storage_db(&db_str, &session_id)?;

        let model_opt = args.model.clone().or_else(crate::config::get_default_model);
        let _ = chat.notify_session_update(model_opt.as_deref(), None, Some("active"));

        emit_output(&session_id)?;
        return Ok(());
    }

    let session_target = if session_id_only {
        None
    } else {
        args.session.as_deref().or(session_env.as_deref())
    };

    if session_id_only && prompt.is_empty() {
        let db = match db_path {
            Some(ref p) => p,
            None => bail!("Error: Storage is disabled. Nothing to create."),
        };
        let session_id = talu::responses::new_session_id()?;
        let chat = ChatHandle::new(None)?;
        let db_str = db.to_string_lossy();
        chat.set_storage_db(&db_str, &session_id)?;

        let model_opt = args.model.clone().or_else(crate::config::get_default_model);
        let _ = chat.notify_session_update(model_opt.as_deref(), None, Some("active"));

        println!("{session_id}");
        return Ok(());
    }

    // No prompt → list sessions or show/delete by ID
    if prompt.is_empty() {
        if args.delete && session_target.is_none() {
            bail!("Error: --delete requires --session <id>.");
        }

        let db = match db_path {
            Some(ref p) => p,
            None => bail!("Error: Storage is disabled. Nothing to show."),
        };
        let handle = match StorageHandle::open(db) {
            Ok(h) => h,
            Err(StorageError::StorageNotFound(p)) => {
                bail!("Error: No sessions yet. Use 'talu ask \"prompt\"' to start one.\n  Storage: {}", p.display());
            }
            Err(e) => return Err(e.into()),
        };

        if let Some(target) = session_target {
            let session_id = resolve_session_target(&handle, target)?;
            if args.delete {
                let title = handle
                    .get_session(&session_id)
                    .ok()
                    .and_then(|s| s.title.clone())
                    .unwrap_or_default();
                handle.delete_session(&session_id)?;
                let display_title = if title.is_empty() {
                    "(untitled)".to_string()
                } else {
                    title
                };
                eprintln!(
                    "Deleted session {} ({})",
                    truncate_str(&session_id, 8),
                    display_title
                );
                return Ok(());
            }
            if use_json {
                let conv = handle
                    .load_session(&session_id)
                    .map_err(|e| anyhow!("Error: Failed to load session: {}", e))?;
                let json = conv.to_responses_json(1)?;
                emit_output(&json)?;
                return Ok(());
            }
            return show_session_transcript(&handle, &session_id, verbose);
        }

        let sessions = handle.list_sessions(Some(50))?;
        let total_count = handle.session_count()?;
        if use_json {
            let list: Vec<_> = sessions
                .iter()
                .map(|s| {
                    serde_json::json!({
                        "session_id": s.session_id,
                        "title": s.title,
                        "model": s.model,
                        "marker": s.marker,
                        "created_at": s.created_at,
                        "updated_at": s.updated_at,
                    })
                })
                .collect();
            let json = serde_json::to_string(&list)?;
            emit_output(&json)?;
        } else {
            print_sessions_with_stats(&handle, &sessions);
        }
        if total_count > 50 {
            eprintln!("Showing 50 of {} sessions.", total_count);
        }
        return Ok(());
    }

    let model_arg = if let Some(path) = args.model.clone() {
        path
    } else if let Some(default) = crate::config::get_default_model() {
        default
    } else {
        eprintln!("Error: No model specified.\n");
        eprintln!("  talu set                                Set a default model");
        eprintln!("  MODEL_URI=Org/Model talu ask \"prompt\"  Use env var");
        return Ok(());
    };

    if args.delete {
        bail!("Error: --delete can only be used without a prompt.");
    }

    // Determine whether to continue an existing session or start a new one.
    let (is_continue, session_id) = if session_id_only {
        let sid = if let Some(target) = args.session.as_deref() {
            target.to_string()
        } else {
            talu::responses::new_session_id()?
        };
        (false, sid)
    } else if let Some(target) = session_target {
        let db = match db_path {
            Some(ref p) => p,
            None => {
                bail!("Error: Storage is disabled. Use --session only with persistent storage.")
            }
        };
        let handle = match StorageHandle::open(db) {
            Ok(h) => h,
            Err(StorageError::StorageNotFound(p)) => {
                bail!("Error: No sessions yet. Use 'talu ask \"prompt\"' to start one.\n  Storage: {}", p.display());
            }
            Err(e) => return Err(e.into()),
        };
        let session_id = resolve_session_target(&handle, target)?;
        (true, session_id)
    } else {
        (false, talu::responses::new_session_id()?)
    };

    // Parse provider prefix
    let target = parse_model_target(&model_arg)?;

    // Handle remote providers
    if let ModelTarget::Remote { provider, model } = target {
        if !stdin_image_parts.is_empty() {
            bail!("Error: piped image input is currently supported only for local models.");
        }
        return cmd_ask_remote(
            &provider,
            model,
            &prompt,
            &system_msg,
            no_stream,
            raw_output,
            hide_thinking,
            endpoint_url_override,
            seed,
            is_continue,
            &session_id,
            &db_path,
            &output_path,
            quiet,
            silent,
            use_json,
            session_id_only,
        );
    }

    // Local inference path
    let resolved_model = resolve_model_for_inference(&model_arg)?;

    // Parse environment overrides
    let max_tokens = env::var("TOKENS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MAX_TOKENS);
    let temperature_from_env = env::var("TEMPERATURE")
        .ok()
        .and_then(|v| v.parse::<f32>().ok());
    let top_k_from_env = env::var("TOP_K").ok().and_then(|v| v.parse::<usize>().ok());
    let top_p_from_env = env::var("TOP_P").ok().and_then(|v| v.parse::<f32>().ok());
    let presence_penalty_from_env = env::var("PRESENCE_PENALTY")
        .ok()
        .and_then(|v| v.parse::<f32>().ok());
    let frequency_penalty_from_env = env::var("FREQUENCY_PENALTY")
        .ok()
        .and_then(|v| v.parse::<f32>().ok());
    let repetition_penalty_from_env = env::var("REPETITION_PENALTY")
        .ok()
        .and_then(|v| v.parse::<f32>().ok());
    let min_p_from_env = env::var("MIN_P").ok().and_then(|v| v.parse::<f32>().ok());
    // Use core-owned policy to resolve effective generation config
    let effective = talu::model::resolve_effective_generation_config(
        &resolved_model,
        &talu::EffectiveGenConfigRequest {
            temperature: temperature_from_env,
            top_k: top_k_from_env,
            top_p: top_p_from_env,
            presence_penalty: presence_penalty_from_env,
            frequency_penalty: frequency_penalty_from_env,
            repetition_penalty: repetition_penalty_from_env,
            min_p: min_p_from_env,
            seed,
            max_tokens,
        },
    )?;

    // Validate top_p range
    if !(0.0..=1.0).contains(&effective.top_p) {
        bail!(
            "Error: TOP_P must be in range [0.0, 1.0], got {}",
            effective.top_p
        );
    }

    if no_chat {
        system_msg.clear();
    }

    // When continuing from DB, create chat WITHOUT system message — the
    // persisted session already contains the original system message.
    let chat = if is_continue && db_path.is_some() {
        ChatHandle::new(None)?
    } else {
        ChatHandle::new(if system_msg.is_empty() {
            None
        } else {
            Some(system_msg.as_str())
        })?
    };

    if let Some(ref db) = db_path {
        let db_str = db.to_string_lossy();
        chat.set_storage_db(&db_str, &session_id)?;
    }

    let mut content = Vec::new();
    for image_part in stdin_image_parts {
        content.push(image_part);
    }
    if !prompt.is_empty() {
        content.push(talu::router::ContentPart::Text(prompt.clone()));
    }

    let think = args.think;
    let mut cfg = talu::router::GenerateConfig {
        max_tokens: effective.max_tokens,
        temperature: effective.temperature,
        top_k: effective.top_k,
        top_p: effective.top_p,
        min_p: effective.min_p,
        repetition_penalty: effective.repetition_penalty,
        presence_penalty: effective.presence_penalty,
        frequency_penalty: effective.frequency_penalty,
        seed: effective.seed,
        raw_output,
        max_reasoning_tokens: if think { None } else { Some(0) },
        ..Default::default()
    };

    if no_chat {
        cfg.template_override = Some("{{ messages[-1].content }}".to_string());
    }

    // Create inference backend with progress reporting
    let progress_ctx = std::sync::Arc::new(std::sync::Mutex::new(UnifiedProgressCtx::new()));
    let callback: Option<talu::LoadProgressCallback> =
        if quiet || use_json || silent || output_path.is_some() {
            None
        } else {
            let progress_arc = progress_ctx.clone();
            Some(Box::new(move |update| {
                use talu::ConvertProgressAction;
                let action = match update.action {
                    ConvertProgressAction::Add => talu::RepoProgressAction::Add,
                    ConvertProgressAction::Update => talu::RepoProgressAction::Update,
                    ConvertProgressAction::Complete => talu::RepoProgressAction::Complete,
                };
                if let Ok(mut ctx) = progress_arc.lock() {
                    ctx.on_update(
                        action,
                        update.line_id,
                        &update.label,
                        &update.message,
                        update.current,
                        update.total,
                        false,
                        true,
                    );
                }
            }))
        };
    if let Some(ctx) = args.ctx_size {
        env::set_var("TALU_CUDA_MAX_SEQ_LEN", ctx.to_string());
    }
    let backend = InferenceBackend::new_with_progress(&resolved_model, callback)?;
    if let Ok(mut ctx) = progress_ctx.lock() {
        ctx.finish();
    }

    // Batched completions: submit requests to the batch scheduler.
    // --each-line: each stdin line is a unique prompt, batched up to max_batch_size.
    // -n N:        N completions of the same prompt (different seeds).
    let n_completions = args.completions;
    let use_batch = n_completions > 1 || !each_line_prompts.is_empty();
    if use_batch {
        if each_line_prompts.is_empty() && n_completions < 2 {
            bail!("Error: --each-line requires piped stdin with at least one non-empty line.");
        }
        // Build list of (prompt, label) pairs.
        let prompts: Vec<(String, String)> = if each_line_prompts.is_empty() {
            // -n only: same prompt, N completions.
            (0..n_completions)
                .map(|i| (prompt.clone(), format!("Completion {}", i + 1)))
                .collect()
        } else {
            // --each-line: each line is a unique prompt.
            each_line_prompts
                .iter()
                .enumerate()
                .map(|(i, p)| (p.clone(), format!("Line {}", i + 1)))
                .collect()
        };

        // Scheduler auto-clamps to backend's max_batch_size (default 8,
        // override via TALU_MAX_BATCH_SIZE). No explicit config needed.
        let batch = talu::batch::BatchHandle::new(&backend, None)?;
        let mut request_ids: Vec<(u64, ChatHandle, String)> = Vec::new();
        for (i, (prompt_text, label)) in prompts.iter().enumerate() {
            let chat_i = ChatHandle::new(if system_msg.is_empty() {
                None
            } else {
                Some(system_msg.as_str())
            })?;
            chat_i.append_user_message(prompt_text)?;
            let mut cfg_i = talu::router::GenerateConfig {
                max_tokens: cfg.max_tokens,
                temperature: cfg.temperature,
                top_k: cfg.top_k,
                top_p: cfg.top_p,
                min_p: cfg.min_p,
                repetition_penalty: cfg.repetition_penalty,
                presence_penalty: cfg.presence_penalty,
                frequency_penalty: cfg.frequency_penalty,
                seed: cfg.seed.wrapping_add(i as u64),
                raw_output: cfg.raw_output,
                ..Default::default()
            };
            if no_chat {
                cfg_i.template_override = Some("{{ messages[-1].content }}".to_string());
            }
            let rid = batch.submit(&chat_i, &cfg_i)?;
            request_ids.push((rid, chat_i, label.clone()));
        }

        // Run decode loop, collect events as segments per request (by item_type/content_type).
        // This mirrors the server's batch path — segments are later converted to responses JSON
        // and loaded into each ChatHandle so the responses API handles reasoning/message display.
        struct Segment {
            item_type: u8,
            content_type: u8,
            text: String,
        }
        let total_requests = request_ids.len();
        let mut per_request_segments: Vec<Vec<Segment>> =
            (0..total_requests).map(|_| Vec::new()).collect();
        let pending = std::sync::atomic::AtomicBool::new(false);
        batch.run_loop(&pending, |event| {
            if let Some(idx) = request_ids
                .iter()
                .position(|(id, _, _)| *id == event.request_id)
            {
                if !event.text.is_empty() {
                    let segs = &mut per_request_segments[idx];
                    let needs_new = segs.last().map_or(true, |s| {
                        s.item_type != event.item_type || s.content_type != event.content_type
                    });
                    if needs_new {
                        segs.push(Segment {
                            item_type: event.item_type,
                            content_type: event.content_type,
                            text: event.text.clone(),
                        });
                    } else if let Some(last) = segs.last_mut() {
                        last.text.push_str(&event.text);
                    }
                }
            }
        })?;

        // Display results. Format segments with ANSI like the streaming path.
        for (i, (rid, chat_i, label)) in request_ids.iter().enumerate() {
            let segments = &per_request_segments[i];

            eprintln!("\n--- {} ---", label);

            if use_json {
                // JSON mode: build responses JSON from segments, load into chat, serialize.
                let mut items: Vec<serde_json::Value> = Vec::new();
                let mut si = 0;
                while si < segments.len() {
                    match segments[si].item_type {
                        0 => {
                            let mut parts = Vec::new();
                            while si < segments.len() && segments[si].item_type == 0 {
                                parts.push(serde_json::json!({
                                    "type": "output_text", "text": segments[si].text
                                }));
                                si += 1;
                            }
                            items.push(serde_json::json!({
                                "type": "message", "role": "assistant", "content": parts
                            }));
                        }
                        3 => {
                            let mut text = String::new();
                            while si < segments.len() && segments[si].item_type == 3 {
                                text.push_str(&segments[si].text);
                                si += 1;
                            }
                            items.push(serde_json::json!({
                                "type": "reasoning",
                                "summary": [{"type": "summary_text", "text": text}]
                            }));
                        }
                        _ => {
                            si += 1;
                        }
                    }
                }
                if !items.is_empty() {
                    let output_json = serde_json::to_string(&items)?;
                    chat_i.load_responses_json(&output_json)?;
                }
                let json = chat_i.to_responses_json(1)?;
                println!("{}", json);
            } else {
                // Text mode: format segments with ANSI like the streaming path.
                let mut in_reasoning = false;
                for seg in segments {
                    let is_reasoning = seg.item_type == 3;
                    if hide_thinking && is_reasoning {
                        continue;
                    }
                    if !raw_output {
                        if is_reasoning && !in_reasoning {
                            let _ = io::stdout().write_all(b"\x1b[2;3m");
                        } else if !is_reasoning && in_reasoning {
                            let _ = io::stdout().write_all(b"\x1b[0m");
                        }
                        in_reasoning = is_reasoning;
                    }
                    let _ = io::stdout().write_all(seg.text.as_bytes());
                }
                if !raw_output && in_reasoning {
                    let _ = io::stdout().write_all(b"\x1b[0m");
                }
                let _ = io::stdout().write_all(b"\n");
                let _ = io::stdout().flush();
            }

            if !quiet && !silent {
                if let Some(result) = batch.take_result(*rid) {
                    let output_tok_per_sec = if result.generation_ns > 0 {
                        (result.completion_tokens as f64)
                            / (result.generation_ns as f64 / 1_000_000_000.0)
                    } else {
                        0.0
                    };
                    let input_tok_per_sec = if result.prefill_ns > 0 {
                        (result.prompt_tokens as f64) / (result.prefill_ns as f64 / 1_000_000_000.0)
                    } else {
                        0.0
                    };
                    eprintln!(
                        "[input: {} tok @ {:.1} t/s | output: {} tok @ {:.1} t/s | prefill {:.2}s | decode {:.2}s]",
                        result.prompt_tokens,
                        input_tok_per_sec,
                        result.completion_tokens,
                        output_tok_per_sec,
                        result.prefill_ns as f64 / 1_000_000_000.0,
                        result.generation_ns as f64 / 1_000_000_000.0
                    );
                }
            }
        }
        return Ok(());
    }

    if no_stream {
        // Set up prefill progress bar for non-streaming path
        let spinner = if !quiet && !silent && !use_json && output_path.is_none() {
            let bar = ProgressBar::new_spinner();
            bar.set_style(
                ProgressStyle::default_spinner()
                    .template("{spinner:.cyan} {msg}")
                    .unwrap(),
            );
            bar.enable_steady_tick(std::time::Duration::from_millis(100));
            let bar_for_cb = bar.clone();
            cfg.prefill_progress = Some(Box::new(move |completed, total| {
                if bar_for_cb.length() != Some(total as u64) {
                    bar_for_cb.set_length(total as u64);
                    bar_for_cb.set_style(
                        ProgressStyle::default_bar()
                            .template(
                                "{spinner:.cyan} Prefill [{bar:30.cyan/bright.black}] {pos}/{len}",
                            )
                            .unwrap()
                            .progress_chars("#-"),
                    );
                }
                bar_for_cb.set_position(completed as u64);
            }));
            Some(bar)
        } else {
            None
        };

        let result = talu::router::generate(&chat, &content, &backend, &cfg)?;
        if let Some(s) = spinner {
            s.finish_with_message("done");
        }

        if result.error_code() != 0 {
            let code = result.error_code();
            let message = last_error_message().unwrap_or_else(|| "generation failed".to_string());
            return Err(anyhow!("Error: {} (code {})", message, code));
        }

        if use_json {
            let json = chat.to_responses_json(1)?;
            emit_output(&json)?;
        } else if raw_output {
            if let Some(text) = result.text() {
                emit_output(&text)?;
            }
        } else if let Some(text) = latest_visible_text(&chat, !hide_thinking)? {
            emit_output(&text)?;
        } else if let Some(text) = result.text() {
            // Fallback to raw generation text if item reconstruction is unavailable.
            emit_output(&text)?;
        }

        if !quiet && !silent && !use_json && !session_id_only && output_path.is_none() {
            let output_tok_per_sec = if result.generation_ns() > 0 {
                (result.token_count() as f64) / (result.generation_ns() as f64 / 1_000_000_000.0)
            } else {
                0.0
            };
            eprintln!(
                "\n[{:.1} tok/s | {} tokens | {:.2}s]",
                output_tok_per_sec,
                result.token_count(),
                result.generation_ns() as f64 / 1_000_000_000.0
            );
        }
    } else {
        let flush_each_token = true;
        let ctx = std::sync::Arc::new(std::sync::Mutex::new(StreamCtx::new(
            raw_output,
            hide_thinking,
            flush_each_token,
        )));

        // Set up prefill progress bar (cleared on first token)
        if !quiet && !silent && !use_json && output_path.is_none() {
            let bar = ProgressBar::new_spinner();
            bar.set_style(
                ProgressStyle::default_spinner()
                    .template("{spinner:.cyan} {msg}")
                    .unwrap(),
            );
            bar.enable_steady_tick(std::time::Duration::from_millis(100));
            let bar_for_cb = bar.clone();
            cfg.prefill_progress = Some(Box::new(move |completed, total| {
                if bar_for_cb.length() != Some(total as u64) {
                    // First call: switch from spinner to determinate bar
                    bar_for_cb.set_length(total as u64);
                    bar_for_cb.set_style(
                        ProgressStyle::default_bar()
                            .template(
                                "{spinner:.cyan} Prefill [{bar:30.cyan/bright.black}] {pos}/{len}",
                            )
                            .unwrap()
                            .progress_chars("#-"),
                    );
                }
                bar_for_cb.set_position(completed as u64);
            }));
            if let Ok(mut guard) = ctx.lock() {
                guard.prefill_spinner = Some(bar);
            }
        }

        let ctx_clone = ctx.clone();
        let callback: talu::router::StreamCallback = Box::new(move |token| {
            let mut guard = ctx_clone.lock().unwrap();
            guard.on_token(token);
            true
        });
        let stream_result =
            talu::router::generate_stream(&chat, &content, &backend, &cfg, callback)?;
        // Flush after streaming
        let mut emitted_visible = false;
        if let Ok(mut guard) = ctx.lock() {
            guard.flush();
            emitted_visible = guard.emitted_visible;
        }

        if !emitted_visible && !silent && !use_json && output_path.is_none() && !raw_output {
            if let Some(text) = latest_visible_text(&chat, !hide_thinking)? {
                emit_output(&text)?;
            }
        }

        // Print stats (cyan color)
        if !quiet && !silent && !use_json && !session_id_only && output_path.is_none() {
            let input_tok_per_sec = stream_result.prefill_tokens_per_second();
            let output_tok_per_sec = stream_result.tokens_per_second();
            eprintln!(
                "\n\x1b[36minput: {} tok @ {:.1} t/s | output: {} tok @ {:.1} t/s\x1b[0m",
                stream_result.prompt_tokens,
                input_tok_per_sec,
                stream_result.completion_tokens,
                output_tok_per_sec
            );
        }
    }

    // Persist session metadata
    if db_path.is_some() {
        // Get model name (last path component)
        let model_name = std::path::Path::new(&resolved_model)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(&resolved_model);

        if is_continue {
            let _ = chat.notify_session_update(
                Some(model_name),
                None, // Don't update title on continue
                Some("active"),
            );
        } else {
            // Derive title from first ~50 chars of prompt
            let title = if prompt.len() > 50 {
                format!("{}...", &prompt[..47])
            } else {
                prompt.clone()
            };

            let _ = chat.notify_session_update(Some(model_name), Some(&title), Some("active"));
        }
    }

    if session_id_only {
        println!("{session_id}");
    }

    Ok(())
}

/// Generate text using a remote OpenAI-compatible provider via the C API
#[allow(clippy::too_many_arguments)]
fn cmd_ask_remote(
    provider_name: &str,
    model_id: &str,
    prompt: &str,
    system_msg: &str,
    no_stream: bool,
    raw_output: bool,
    hide_thinking: bool,
    endpoint_url_override: Option<String>,
    seed: u64,
    is_continue: bool,
    session_id: &str,
    db_path: &Option<PathBuf>,
    output_path: &Option<PathBuf>,
    quiet: bool,
    silent: bool,
    use_json: bool,
    session_id_only: bool,
) -> Result<()> {
    use std::time::Instant;

    let provider = get_provider(provider_name)
        .ok_or_else(|| anyhow!("Unknown provider: {}", provider_name))?;

    // Determine endpoint URL
    let base_url = endpoint_url_override
        .clone()
        .or_else(|| env::var(format!("{}_ENDPOINT", provider_name.to_uppercase())).ok())
        .unwrap_or_else(|| provider.default_endpoint.to_string());

    // If no model specified, list available models and exit
    if model_id.is_empty() {
        eprintln!("Error: No model specified for {}.\n", provider_name);

        // Try to list models
        let api_key = provider
            .api_key_env
            .as_ref()
            .and_then(|env_var| env::var(env_var).ok());

        match list_provider_models(provider_name, &base_url, api_key.as_deref()) {
            Ok(models) if !models.is_empty() => {
                eprintln!("Available models on {} ({}):", provider_name, base_url);
                for model in &models {
                    eprintln!("  {}", model);
                }
            }
            Ok(_) => {
                eprintln!("No models found on {} ({})", provider_name, base_url);
            }
            Err(e) => {
                eprintln!("Could not list models on {}: {}", provider_name, e);
            }
        }

        eprintln!("\nUsage: talu ask -m {}::<model> \"prompt\"", provider_name);
        return Ok(());
    }

    if prompt.is_empty() {
        bail!("Error: ask requires a prompt.\nUse 'talu ask' to list sessions.");
    }

    // Get API key if needed
    let api_key = provider
        .api_key_env
        .as_ref()
        .and_then(|env_var| env::var(env_var).ok());

    // Get generation parameters
    let max_tokens = env::var("TOKENS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MAX_TOKENS);
    let temperature = env::var("TEMPERATURE")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.7);

    // When continuing from DB, create chat WITHOUT system message — the
    // persisted session already contains the original system message.
    let chat = if is_continue && db_path.is_some() {
        ChatHandle::new(None)?
    } else {
        ChatHandle::new(if system_msg.is_empty() {
            None
        } else {
            Some(system_msg)
        })?
    };

    if let Some(ref db) = db_path {
        let db_str = db.to_string_lossy();
        chat.set_storage_db(&db_str, session_id)?;
    }

    // Create remote backend via C API
    let backend = InferenceBackend::new_openai_compatible(
        model_id,
        &base_url,
        api_key.as_deref(),
        120_000, // 2 minute timeout
    )?;

    // Build content part for the prompt
    let content = vec![talu::router::ContentPart::Text(prompt.to_string())];

    // Build generation config
    let cfg = talu::router::GenerateConfig {
        max_tokens,
        temperature,
        seed,
        raw_output,
        ..Default::default()
    };

    let start_time = Instant::now();

    // Generate via the router safe API
    let emit_output = |text: &str| -> Result<()> {
        if session_id_only {
            return Ok(());
        }
        if let Some(path) = output_path {
            std::fs::write(path, format!("{text}\n"))?;
            return Ok(());
        }
        if silent {
            return Ok(());
        }
        println!("{text}");
        Ok(())
    };

    if no_stream {
        let result = talu::router::generate(&chat, &content, &backend, &cfg)?;

        if result.error_code() != 0 {
            let code = result.error_code();
            let message =
                last_error_message().unwrap_or_else(|| "remote generation failed".to_string());
            return Err(anyhow!("Error: {} (code {})", message, code));
        }

        if use_json {
            let json = chat.to_responses_json(1)?;
            emit_output(&json)?;
        } else if raw_output {
            if let Some(text) = result.text() {
                emit_output(&text)?;
            }
        } else if let Some(text) = latest_visible_text(&chat, !hide_thinking)? {
            emit_output(&text)?;
        } else if let Some(text) = result.text() {
            // Fallback to raw generation text if item reconstruction is unavailable.
            emit_output(&text)?;
        }

        if !quiet && !silent && !use_json && !session_id_only && output_path.is_none() {
            let elapsed = start_time.elapsed();
            eprintln!(
                "\n\x1b[36m{}::{} | {:.1}s | {} tokens\x1b[0m",
                provider_name,
                model_id,
                elapsed.as_secs_f64(),
                result.token_count()
            );
        }
    } else {
        // Streaming - use a simple callback that prints content
        let flush_each_token = true;
        let pending_flush_bytes = std::sync::Arc::new(std::sync::Mutex::new(0usize));
        let pending_flush_bytes_cb = pending_flush_bytes.clone();
        let callback: talu::router::StreamCallback = Box::new(move |token| {
            if hide_thinking && token.item_type == talu::responses::ItemType::Reasoning {
                return true;
            }
            let _ = io::stdout().write_all(token.text.as_bytes());
            if let Ok(mut pending) = pending_flush_bytes_cb.lock() {
                *pending += token.text.len();
                if should_flush_stream_output(&token.text, *pending, flush_each_token) {
                    let _ = io::stdout().flush();
                    *pending = 0;
                }
            }
            true
        });
        talu::router::generate_stream(&chat, &content, &backend, &cfg, callback)?;
        let _ = io::stdout().flush();
        println!(); // Newline after streaming

        if !quiet && !silent && !use_json && !session_id_only && output_path.is_none() {
            let elapsed = start_time.elapsed();
            eprintln!(
                "\n\x1b[36m{}::{} | {:.1}s\x1b[0m",
                provider_name,
                model_id,
                elapsed.as_secs_f64()
            );
        }
    }

    // Persist session metadata
    if db_path.is_some() {
        if is_continue {
            let _ = chat.notify_session_update(
                Some(model_id),
                None, // Don't update title on continue
                Some("active"),
            );
        } else {
            let title = if prompt.len() > 50 {
                format!("{}...", &prompt[..47])
            } else {
                prompt.to_string()
            };

            let _ = chat.notify_session_update(Some(model_id), Some(&title), Some("active"));
        }
    }

    if session_id_only {
        println!("{session_id}");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_base64_known_vectors() {
        assert_eq!(encode_base64(b""), "");
        assert_eq!(encode_base64(b"f"), "Zg==");
        assert_eq!(encode_base64(b"fo"), "Zm8=");
        assert_eq!(encode_base64(b"foo"), "Zm9v");
        assert_eq!(encode_base64(b"hello"), "aGVsbG8=");
    }

    #[test]
    fn parse_stdin_content_text_trims_trailing_whitespace() {
        let parsed = parse_stdin_content(b"hello world\n".to_vec()).expect("parse text stdin");
        assert_eq!(parsed.text.as_deref(), Some("hello world"));
        assert!(parsed.images.is_empty());
    }

    #[test]
    fn parse_stdin_content_image_builds_data_url() {
        let jpeg = include_bytes!("../../../../../core/tests/image/corpus/1x1_red.jpg");
        let parsed = parse_stdin_content(jpeg.to_vec()).expect("parse jpeg stdin");
        assert!(parsed.text.is_none());
        assert_eq!(parsed.images.len(), 1);
        match &parsed.images[0] {
            talu::router::ContentPart::ImageUrl { url, mime } => {
                assert_eq!(mime.as_deref(), Some("image/jpeg"));
                assert!(url.starts_with("data:image/jpeg;base64,"));
            }
            _ => panic!("expected image url content part"),
        }
    }

    #[test]
    fn parse_stdin_content_pdf_converts_to_images() {
        let pdf = include_bytes!("../../../../../deps/pdfium/testing/resources/hello_world.pdf");
        let parsed = parse_stdin_content(pdf.to_vec()).expect("parse pdf stdin");
        assert!(parsed.text.is_none());
        assert_eq!(parsed.images.len(), 1); // hello_world.pdf is a single page
        match &parsed.images[0] {
            talu::router::ContentPart::ImageUrl { url, mime } => {
                assert_eq!(mime.as_deref(), Some("image/png"));
                assert!(url.starts_with("data:image/png;base64,"));
            }
            _ => panic!("expected image url content part"),
        }
    }

    #[test]
    fn parse_stdin_content_rejects_binary_nul_data() {
        let err = parse_stdin_content(vec![0, 1, 2, 3]).expect_err("binary stdin should fail");
        let msg = err.to_string();
        assert!(msg.contains("binary data"));
    }

    #[test]
    fn stream_flush_policy_flushes_each_token_for_interactive_output() {
        assert!(should_flush_stream_output("x", 1, true));
    }

    #[test]
    fn stream_flush_policy_buffers_when_not_interactive() {
        assert!(should_flush_stream_output("\n", 1, false));
        assert!(should_flush_stream_output(
            "token",
            STREAM_FLUSH_BYTES,
            false
        ));
        assert!(!should_flush_stream_output(
            "token",
            STREAM_FLUSH_BYTES - 1,
            false
        ));
    }
}
