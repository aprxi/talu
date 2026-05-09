use std::env;
use std::io::{self, Read, Write};

use anyhow::{anyhow, bail, Context, Result};
use base64::Engine;
use indicatif::{ProgressBar, ProgressStyle};
use serde_json::json;
use talu::responses::ResponsesView;

use talu::error::last_error_message;
use talu::{ChatHandle, InferenceBackend};

use crate::provider::ensure_local_model_target;
use crate::server::vision;

use super::repo::{resolve_model_for_inference, UnifiedProgressCtx};
use super::talupi_sessions::{self, SessionRecord, UpsertSessionRequest};
use super::util::DEFAULT_MAX_TOKENS;
use super::{AskArgs, AskOutputFormat};

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
            // Clear transient progress UI before printing model output.
            spinner.finish_and_clear();
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
            // Clear transient progress UI before flushing the final line.
            spinner.finish_and_clear();
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
    prepared_stdin_json: Option<String>,
}

fn trim_trailing_stdin_whitespace(bytes: &mut Vec<u8>) {
    while matches!(bytes.last(), Some(b'\n' | b'\r' | b' ')) {
        bytes.pop();
    }
}

fn sniff_stdin_mime(bytes: &[u8]) -> &'static str {
    if bytes.starts_with(&[0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n']) {
        return "image/png";
    }
    if bytes.len() >= 3 && bytes[0] == 0xff && bytes[1] == 0xd8 && bytes[2] == 0xff {
        return "image/jpeg";
    }
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return "image/webp";
    }
    if bytes.starts_with(b"%PDF-") {
        return "application/pdf";
    }
    "application/octet-stream"
}

fn parse_stdin_content(mut stdin_buf: Vec<u8>) -> Result<ParsedStdin> {
    if stdin_buf.is_empty() {
        return Ok(ParsedStdin {
            text: None,
            images: vec![],
            prepared_stdin_json: None,
        });
    }

    let is_binary = stdin_buf.contains(&0) || std::str::from_utf8(&stdin_buf).is_err();
    if is_binary {
        if vision::file_host_from_env().is_none() {
            bail!(
                "Error: binary stdin input is not supported in this build without TALU_FILE_HOST."
            );
        }
        let mime = sniff_stdin_mime(&stdin_buf);
        let data_url = format!(
            "data:{};base64,{}",
            mime,
            base64::engine::general_purpose::STANDARD.encode(&stdin_buf)
        );
        return Ok(ParsedStdin {
            text: None,
            images: vec![talu::router::ContentPart::ImageUrl {
                url: data_url,
                mime: Some(mime.to_string()),
            }],
            prepared_stdin_json: None,
        });
    }

    trim_trailing_stdin_whitespace(&mut stdin_buf);
    if stdin_buf.is_empty() {
        return Ok(ParsedStdin {
            text: None,
            images: vec![],
            prepared_stdin_json: None,
        });
    }
    let text = String::from_utf8(stdin_buf)
        .map_err(|_| anyhow!("Error: non-UTF-8 stdin input is not supported in this build."))?;
    if vision::stdin_json_has_prepared_payload(&text) {
        return Ok(ParsedStdin {
            text: None,
            images: vec![],
            prepared_stdin_json: Some(text),
        });
    }
    Ok(ParsedStdin {
        text: Some(text),
        images: vec![],
        prepared_stdin_json: None,
    })
}

#[derive(Debug, Clone)]
struct SessionCtx {
    db_host: String,
    session_id: String,
    existing: Option<SessionRecord>,
}

fn conversation_title_from_prompt(prompt: &str) -> Option<String> {
    let trimmed = prompt.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed.chars().take(80).collect())
}

fn default_metadata_object(metadata: &serde_json::Value) -> serde_json::Value {
    if metadata.is_object() {
        metadata.clone()
    } else {
        json!({})
    }
}

fn load_session_history(chat: &ChatHandle, session: Option<&SessionCtx>) -> Result<()> {
    if let Some(sess) = session {
        if !sess
            .existing
            .as_ref()
            .map_or(true, |s| s.responses_json.trim().is_empty())
        {
            if let Some(existing) = sess.existing.as_ref() {
                chat.load_responses_json(&existing.responses_json)?;
            }
        }
    }
    Ok(())
}

fn persist_session(
    chat: &ChatHandle,
    session: &SessionCtx,
    model_id: &str,
    system_msg: &str,
    prompt: &str,
) -> Result<()> {
    let responses_json = chat.to_responses_json(1)?;
    let existing = session.existing.as_ref();
    let metadata = existing
        .map(|s| default_metadata_object(&s.metadata))
        .unwrap_or_else(|| json!({}));
    let title = existing
        .and_then(|s| s.title.clone())
        .or_else(|| conversation_title_from_prompt(prompt));
    let req = UpsertSessionRequest {
        session_id: session.session_id.clone(),
        responses_json,
        model: existing
            .and_then(|s| s.model.clone())
            .or_else(|| Some(model_id.to_string())),
        title,
        system_prompt: existing.and_then(|s| s.system_prompt.clone()).or_else(|| {
            if system_msg.is_empty() {
                None
            } else {
                Some(system_msg.to_string())
            }
        }),
        metadata,
        project_id: existing.and_then(|s| s.project_id.clone()),
        marker: existing
            .and_then(|s| s.marker.clone())
            .or_else(|| Some("active".to_string())),
        parent_session_id: existing.and_then(|s| s.parent_session_id.clone()),
    };
    talupi_sessions::upsert_session(&session.db_host, &req)
}

fn message_role_label(role: talu::responses::MessageRole) -> &'static str {
    match role {
        talu::responses::MessageRole::System => "system",
        talu::responses::MessageRole::User => "user",
        talu::responses::MessageRole::Assistant => "assistant",
        talu::responses::MessageRole::Developer => "developer",
        _ => "unknown",
    }
}

fn format_session_transcript(session: &SessionRecord, include_reasoning: bool) -> Result<String> {
    let chat = ChatHandle::new(None)?;
    chat.load_responses_json(&session.responses_json)?;
    let conv = chat.responses();
    let mut lines = Vec::new();
    for index in 0..conv.item_count() {
        let item_type = conv.item_type(index);
        if item_type == talu::responses::ItemType::Message {
            let msg = conv.get_message(index)?;
            let text = conv.message_text(index)?;
            if !text.trim().is_empty() {
                lines.push(format!("{}: {}", message_role_label(msg.role), text));
            }
        } else if include_reasoning && item_type == talu::responses::ItemType::Reasoning {
            let text = reasoning_text_for_item(&conv, index)?;
            if !text.trim().is_empty() {
                lines.push(format!("assistant_reasoning: {}", text));
            }
        }
    }
    if lines.is_empty() {
        Ok(String::new())
    } else {
        Ok(lines.join("\n"))
    }
}

/// Chat: one-shot generation (stateless).
pub(super) fn cmd_ask(args: AskArgs, stdin_is_pipe: bool, _verbose: u8) -> Result<()> {
    let no_chat = args.no_chat;
    let use_json = args.format == Some(AskOutputFormat::Json);
    let quiet = args.quiet && !use_json;
    let mut silent = args.silent;
    let raw_output = args.raw;
    let hide_thinking = args.hide_thinking;
    let mut system_msg = args.system.clone();
    let mut prompt_parts = args.prompt.clone();
    let mut stdin_text_part: Option<String> = None;
    let mut stdin_image_parts: Vec<talu::router::ContentPart> = Vec::new();
    let mut stdin_prepared_payload_json: Option<String> = None;
    let seed = args.seed.unwrap_or(0);
    let session_env = env::var("SESSION_ID").ok().filter(|s| !s.is_empty());
    let session_target = args.session.clone().or(session_env);
    let db_host = talupi_sessions::db_host_from_env();
    let mut preloaded_stdin: Option<Vec<u8>> = None;
    let output_path = args.output.clone();

    if silent && args.quiet {
        bail!("Error: cannot specify both --quiet and --silent.");
    }
    if args.session_id_only && session_target.is_some() {
        bail!("Error: --session-id cannot be combined with --session or SESSION_ID.");
    }
    if args.new {
        if !args.prompt.is_empty() {
            bail!("Error: --new does not accept a prompt.");
        }
        let db_host = db_host.ok_or_else(|| {
            anyhow!("Error: --new requires TALU_DB_HOST (example: TALU_DB_HOST=localhost:7258).")
        })?;
        let session_id = format!("sess_{}", uuid::Uuid::new_v4().hyphenated());
        let req = UpsertSessionRequest {
            session_id: session_id.clone(),
            responses_json: "[]".to_string(),
            model: args.model.clone().or_else(crate::config::get_default_model),
            title: None,
            system_prompt: if no_chat {
                None
            } else {
                Some(system_msg.clone())
            },
            metadata: json!({}),
            project_id: None,
            marker: Some("active".to_string()),
            parent_session_id: None,
        };
        talupi_sessions::upsert_session(&db_host, &req)?;
        if let Some(path) = &output_path {
            std::fs::write(path, format!("{session_id}\n"))?;
        } else if !silent {
            println!("{session_id}");
        }
        return Ok(());
    }
    if args.delete {
        let target = session_target
            .as_deref()
            .ok_or_else(|| anyhow!("Error: --delete requires --session or SESSION_ID."))?;
        let db_host = db_host.ok_or_else(|| {
            anyhow!("Error: --delete requires TALU_DB_HOST (example: TALU_DB_HOST=localhost:7258).")
        })?;
        let session_id = talupi_sessions::resolve_session_id(&db_host, target)?;
        talupi_sessions::delete_session(&db_host, &session_id)?;
        return Ok(());
    }
    if let Some(target) = session_target.as_deref() {
        if args.prompt.is_empty() && !args.session_id_only {
            if stdin_is_pipe {
                let mut probe = Vec::new();
                io::stdin()
                    .read_to_end(&mut probe)
                    .context("read stdin for session dispatch")?;
                if !probe.is_empty() {
                    preloaded_stdin = Some(probe);
                } else {
                    let db_host = db_host.ok_or_else(|| {
                        anyhow!(
                            "Error: --session requires TALU_DB_HOST (example: TALU_DB_HOST=localhost:7258)."
                        )
                    })?;
                    let session_id = talupi_sessions::resolve_session_id(&db_host, target)?;
                    let session = talupi_sessions::get_session(&db_host, &session_id)?;
                    let output = if use_json {
                        session.responses_json
                    } else {
                        format_session_transcript(&session, !hide_thinking)?
                    };
                    if let Some(path) = &output_path {
                        std::fs::write(path, format!("{output}\n"))?;
                    } else if !silent {
                        println!("{output}");
                    }
                    return Ok(());
                }
            } else {
                let db_host = db_host.ok_or_else(|| {
                    anyhow!(
                        "Error: --session requires TALU_DB_HOST (example: TALU_DB_HOST=localhost:7258)."
                    )
                })?;
                let session_id = talupi_sessions::resolve_session_id(&db_host, target)?;
                let session = talupi_sessions::get_session(&db_host, &session_id)?;
                let output = if use_json {
                    session.responses_json
                } else {
                    format_session_transcript(&session, !hide_thinking)?
                };
                if let Some(path) = &output_path {
                    std::fs::write(path, format!("{output}\n"))?;
                } else if !silent {
                    println!("{output}");
                }
                return Ok(());
            }
        }
    }
    if args.session_id_only {
        silent = true;
    }

    let no_stream = args.no_stream || use_json || silent || output_path.is_some();
    let emit_output = |text: &str| -> Result<()> {
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
    let emit_session_id = |session_id: &str| -> Result<()> {
        if let Some(path) = &output_path {
            std::fs::write(path, format!("{session_id}\n"))?;
        } else {
            println!("{session_id}");
        }
        Ok(())
    };

    // --each-line: collect stdin lines as separate prompts for batched decode.
    let mut each_line_prompts: Vec<String> = Vec::new();

    if stdin_is_pipe {
        let stdin_buf = if let Some(buf) = preloaded_stdin.take() {
            buf
        } else {
            let mut buf = Vec::new();
            if io::stdin().read_to_end(&mut buf).is_err() {
                Vec::new()
            } else {
                buf
            }
        };
        if !stdin_buf.is_empty() {
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
                    stdin_text_part = Some(text);
                }
                stdin_image_parts = parsed.images;
                stdin_prepared_payload_json = parsed.prepared_stdin_json;
            }
        }
    }

    if !stdin_image_parts.is_empty() && prompt_parts.is_empty() {
        prompt_parts.push(DEFAULT_STDIN_IMAGE_PROMPT.to_string());
    }
    if stdin_prepared_payload_json.is_some() && prompt_parts.is_empty() {
        prompt_parts.push(DEFAULT_STDIN_IMAGE_PROMPT.to_string());
    }
    if let Some(text) = stdin_text_part.take() {
        prompt_parts.push(text);
    }

    let prompt = prompt_parts.join(" ");
    if prompt.is_empty() {
        bail!("Error: ask requires a prompt.");
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

    let mut session_ctx: Option<SessionCtx> = None;
    if let Some(target) = session_target.as_deref() {
        let db_host = db_host.ok_or_else(|| {
            anyhow!(
                "Error: persistent sessions require TALU_DB_HOST (example: TALU_DB_HOST=localhost:7258)."
            )
        })?;
        let session_id = talupi_sessions::resolve_session_id(&db_host, target)?;
        let existing = talupi_sessions::get_session(&db_host, &session_id)?;
        if !no_chat
            && system_msg == DEFAULT_SYSTEM_MESSAGE
            && existing
                .system_prompt
                .as_deref()
                .map(|s| !s.trim().is_empty())
                .unwrap_or(false)
        {
            system_msg = existing.system_prompt.clone().unwrap_or_default();
        }
        session_ctx = Some(SessionCtx {
            db_host,
            session_id,
            existing: Some(existing),
        });
    } else if let Some(db_host) = db_host.clone() {
        session_ctx = Some(SessionCtx {
            db_host,
            session_id: format!("sess_{}", uuid::Uuid::new_v4().hyphenated()),
            existing: None,
        });
    }
    if args.session_id_only && session_ctx.is_none() {
        bail!("Error: --session-id requires TALU_DB_HOST (example: TALU_DB_HOST=localhost:7258).");
    }
    if session_ctx.is_some() && (args.each_line || args.completions > 1) {
        bail!("Error: persistent session mode does not support --each-line or --completions > 1.");
    }

    ensure_local_model_target(&model_arg)?;

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

    let chat = ChatHandle::new(if system_msg.is_empty() {
        None
    } else {
        Some(system_msg.as_str())
    })?;
    load_session_history(&chat, session_ctx.as_ref())?;

    let has_stdin_images = !stdin_image_parts.is_empty();
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

    let stdin_prepared_prefill = if let Some(prepared_json) = stdin_prepared_payload_json.as_deref()
    {
        vision::prepare_vision_prefill_from_stdin_prepared_json(prepared_json, &resolved_model)?
    } else {
        None
    };
    if stdin_prepared_prefill.is_some() && has_stdin_images {
        bail!("Error: cannot mix piped prepared vision JSON with raw piped image input.");
    }
    if let Some(prefill) = stdin_prepared_prefill.as_ref() {
        for index in (0..prefill.images.len()).rev() {
            content.insert(
                0,
                talu::router::ContentPart::ImageUrl {
                    url: format!("prepared://input_{index}"),
                    mime: None,
                },
            );
        }
    }

    cfg.vision_prefill = if stdin_prepared_prefill.is_some() {
        stdin_prepared_prefill
    } else {
        vision::prepare_vision_prefill_from_content(&content, &resolved_model)?
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
    if use_batch && cfg.vision_prefill.is_some() {
        bail!("Error: batched completions with image input are not supported.");
    }
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
            s.finish_and_clear();
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
        if let Some(ref session) = session_ctx {
            persist_session(&chat, session, &model_arg, &system_msg, &prompt)?;
            if args.session_id_only {
                emit_session_id(&session.session_id)?;
                return Ok(());
            }
        }

        if !quiet && !silent && !use_json && output_path.is_none() {
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
            if let Ok(mut guard) = ctx_clone.lock() {
                guard.on_token(token);
                true
            } else {
                false
            }
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
        if let Some(ref session) = session_ctx {
            persist_session(&chat, session, &model_arg, &system_msg, &prompt)?;
            if args.session_id_only {
                emit_session_id(&session.session_id)?;
                return Ok(());
            }
        }

        // Print stats (cyan color)
        if !quiet && !silent && !use_json && output_path.is_none() {
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

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use once_cell::sync::Lazy;
    use std::sync::Mutex;

    const JPEG_MAGIC_AND_BINARY_TAIL: &[u8] = &[0xff, 0xd8, 0xff, 0x00];
    static ENV_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

    #[test]
    fn parse_stdin_content_text_trims_trailing_whitespace() {
        let parsed = parse_stdin_content(b"hello world\n".to_vec()).expect("parse text stdin");
        assert_eq!(parsed.text.as_deref(), Some("hello world"));
        assert!(parsed.images.is_empty());
        assert!(parsed.prepared_stdin_json.is_none());
    }

    #[test]
    fn parse_stdin_content_rejects_binary_image_bytes_without_file_host() {
        let _guard = ENV_LOCK.lock().expect("env lock");
        std::env::remove_var("TALU_FILE_HOST");
        std::env::remove_var("TALUPI_HOST");
        let err = parse_stdin_content(JPEG_MAGIC_AND_BINARY_TAIL.to_vec())
            .expect_err("binary image stdin should fail");
        assert!(err.to_string().contains("without TALU_FILE_HOST"));
    }

    #[test]
    fn parse_stdin_content_rejects_binary_pdf_bytes_without_file_host() {
        let _guard = ENV_LOCK.lock().expect("env lock");
        std::env::remove_var("TALU_FILE_HOST");
        std::env::remove_var("TALUPI_HOST");
        // Include a non-UTF8 byte so stdin is treated as binary.
        let pdf = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n\xff";
        let err = parse_stdin_content(pdf.to_vec()).expect_err("binary pdf stdin should fail");
        assert!(err.to_string().contains("without TALU_FILE_HOST"));
    }

    #[test]
    fn parse_stdin_content_rejects_binary_nul_data_without_file_host() {
        let _guard = ENV_LOCK.lock().expect("env lock");
        std::env::remove_var("TALU_FILE_HOST");
        std::env::remove_var("TALUPI_HOST");
        let err = parse_stdin_content(vec![0, 1, 2, 3]).expect_err("binary stdin should fail");
        assert!(err.to_string().contains("without TALU_FILE_HOST"));
    }

    #[test]
    fn parse_stdin_content_rejects_non_utf8_text_data_without_file_host() {
        let _guard = ENV_LOCK.lock().expect("env lock");
        std::env::remove_var("TALU_FILE_HOST");
        std::env::remove_var("TALUPI_HOST");
        let err = parse_stdin_content(vec![0xf0, 0x28, 0x8c, 0x28])
            .expect_err("non-utf8 stdin should fail");
        assert!(err.to_string().contains("without TALU_FILE_HOST"));
    }

    #[test]
    fn parse_stdin_content_binary_image_with_file_host_builds_data_url() {
        let _guard = ENV_LOCK.lock().expect("env lock");
        std::env::set_var("TALU_FILE_HOST", "http://localhost:7258");
        std::env::remove_var("TALUPI_HOST");
        let parsed =
            parse_stdin_content(JPEG_MAGIC_AND_BINARY_TAIL.to_vec()).expect("parse jpeg stdin");
        assert!(parsed.text.is_none());
        assert_eq!(parsed.images.len(), 1);
        assert!(parsed.prepared_stdin_json.is_none());
        match &parsed.images[0] {
            talu::router::ContentPart::ImageUrl { url, mime } => {
                assert_eq!(mime.as_deref(), Some("image/jpeg"));
                assert!(url.starts_with("data:image/jpeg;base64,"));
            }
            _ => panic!("expected image url content part"),
        }
        std::env::remove_var("TALU_FILE_HOST");
    }

    #[test]
    fn parse_stdin_content_detects_prepared_vision_wrapper_json() {
        let payload = r#"{
            "talu_prepared_vision": {
                "model_profile": {
                    "version": "2026-04-17",
                    "normalize": "minus_one_to_one",
                    "temporal_frames": 1,
                    "patch_size": 16,
                    "temporal_patch_size": 1,
                    "spatial_merge_size": 1
                },
                "item": {
                    "dtype": "f32",
                    "layout": "cthw",
                    "channels": 3,
                    "temporal_frames": 1,
                    "width": 16,
                    "height": 16,
                    "grid": { "temporal": 1, "height": 1, "width": 1 },
                    "token_count": 1,
                    "normalize": "minus_one_to_one",
                    "tensor_b64": "AAAAAA=="
                }
            }
        }"#;
        let parsed =
            parse_stdin_content(payload.as_bytes().to_vec()).expect("parse prepared vision JSON");
        assert!(parsed.text.is_none());
        assert!(parsed.images.is_empty());
        assert!(parsed.prepared_stdin_json.is_some());
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
