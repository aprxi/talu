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

use super::repo::{generation_config, resolve_model_for_inference, UnifiedProgressCtx};
use super::sessions::{print_sessions_with_stats, resolve_session_target, show_session_transcript};
use super::util::{truncate_str, DEFAULT_MAX_TOKENS};
use super::{AskArgs, AskOutputFormat};

use super::models::list_provider_models;

struct StreamCtx {
    raw_output: bool,
    hide_thinking: bool,
    in_reasoning: bool,
    prefill_spinner: Option<indicatif::ProgressBar>,
}

impl StreamCtx {
    fn new(raw_output: bool, hide_thinking: bool) -> Self {
        Self {
            raw_output,
            hide_thinking,
            in_reasoning: false,
            prefill_spinner: None,
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
            } else if !is_reasoning && self.in_reasoning {
                // Leaving reasoning: reset
                let _ = io::stdout().write_all(b"\x1b[0m");
            }
            self.in_reasoning = is_reasoning;
        }

        let _ = io::stdout().write_all(token.text.as_bytes());
        let _ = io::stdout().flush();
    }

    fn flush(&mut self) {
        // Complete prefill progress if generation produced no tokens
        if let Some(spinner) = self.prefill_spinner.take() {
            spinner.finish_with_message("done");
        }
        if !self.raw_output && self.in_reasoning {
            // Reset ANSI if still in reasoning at end
            let _ = io::stdout().write_all(b"\x1b[0m");
        }
        let _ = io::stdout().write_all(b"\n");
        let _ = io::stdout().flush();
    }
}

fn reasoning_text_for_item(conv: &impl ResponsesView, index: usize) -> Result<String> {
    let reasoning = conv.get_reasoning(index)?;
    let mut out = String::new();
    for part_index in 0..reasoning.content_count {
        let part = conv.get_reasoning_content(index, part_index)?;
        out.push_str(&part.data_utf8_lossy());
    }
    Ok(out)
}

fn latest_visible_text(chat: &ChatHandle, include_reasoning: bool) -> Result<Option<String>> {
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

    if stdin_is_pipe {
        let mut stdin_buf = Vec::new();
        if io::stdin().read_to_end(&mut stdin_buf).is_ok() && !stdin_buf.is_empty() {
            while matches!(stdin_buf.last(), Some(b'\n' | b'\r' | b' ')) {
                stdin_buf.pop();
            }
            if !stdin_buf.is_empty() {
                prompt_parts.push(String::from_utf8_lossy(&stdin_buf).to_string());
            }
        }
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
                    .load_conversation(&session_id)
                    .map_err(|e| anyhow!("Error: Failed to load conversation: {}", e))?;
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

    let gen_cfg = generation_config(&resolved_model)?;

    let max_tokens = env::var("TOKENS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MAX_TOKENS);
    let temperature_from_env = env::var("TEMPERATURE")
        .ok()
        .and_then(|v| v.parse::<f32>().ok());
    let temperature = temperature_from_env.unwrap_or(gen_cfg.temperature);
    let top_k = env::var("TOP_K")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(gen_cfg.top_k);
    let top_p = env::var("TOP_P")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(gen_cfg.top_p);

    if temperature < 0.0 {
        bail!("Error: TEMPERATURE must be >= 0, got {}", temperature);
    }
    if !(0.0..=1.0).contains(&top_p) {
        bail!("Error: TOP_P must be in range [0.0, 1.0], got {}", top_p);
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

    let content = vec![talu::router::ContentPart::Text(prompt.clone())];

    let mut cfg = talu::router::GenerateConfig {
        max_tokens,
        seed,
        raw_output,
        ..Default::default()
    };
    if (gen_cfg.do_sample || temperature_from_env.is_some()) && temperature > 0.0 {
        cfg.temperature = temperature;
        cfg.top_k = top_k;
        cfg.top_p = top_p;
    } else {
        cfg.temperature = 0.0;
    }

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
                    );
                }
            }))
        };
    let backend = InferenceBackend::new_with_progress(&resolved_model, callback)?;
    if let Ok(mut ctx) = progress_ctx.lock() {
        ctx.finish();
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
        let ctx = std::sync::Arc::new(std::sync::Mutex::new(StreamCtx::new(
            raw_output,
            hide_thinking,
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
        if let Ok(mut guard) = ctx.lock() {
            guard.flush();
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
        let callback: talu::router::StreamCallback = Box::new(move |token| {
            if hide_thinking && token.item_type == talu::responses::ItemType::Reasoning {
                return true;
            }
            let _ = io::stdout().write_all(token.text.as_bytes());
            let _ = io::stdout().flush();
            true
        });
        talu::router::generate_stream(&chat, &content, &backend, &cfg, callback)?;
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
