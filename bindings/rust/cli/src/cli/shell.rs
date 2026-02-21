use std::env;
use std::io::{self, Read};

use anyhow::{Result, anyhow, bail};

use talu::{ChatHandle, InferenceBackend};

use crate::provider::{ModelTarget, get_provider, parse_model_target};

use super::AgentArgs;
use super::repo::{generation_config, resolve_model_for_inference};
use super::util::DEFAULT_MAX_TOKENS;

/// Interactive agent mode (`talu agent`)
///
/// Enters an agent loop with the built-in `execute_command` tool.
/// The model can request shell commands; execution requires user confirmation.
pub(super) fn cmd_agent(args: AgentArgs, stdin_is_pipe: bool) -> Result<()> {
    let mut prompt_parts = args.prompt.clone();
    let seed = args.seed.unwrap_or(0);
    let db_path =
        crate::config::resolve_bucket(args.no_bucket, args.bucket.clone(), &args.profile)?;

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

    let model_arg = if let Some(path) = args.model.clone() {
        path
    } else if let Ok(env_path) = env::var("MODEL_URI") {
        env_path
    } else {
        bail!("Error: No model specified. Use -m/--model-uri or set MODEL_URI.");
    };

    let target = parse_model_target(&model_arg)?;

    let max_tokens = env::var("TOKENS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MAX_TOKENS);

    if prompt.is_empty() {
        bail!("Error: agent requires a prompt");
    }

    // Shell system prompt with CWD context
    let cwd = env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| ".".to_string());
    if args.tool_timeout == 0 {
        bail!("Error: --tool-timeout must be > 0");
    }

    let tool_registry = crate::agent::tools::load_tools(&args.tools)?;
    let tool_names = tool_registry.names();
    let tools_hint = if tool_names.is_empty() {
        "You have access to a Linux shell via the execute_command tool.".to_string()
    } else {
        format!(
            "You have access to a Linux shell via the execute_command tool and custom tools: {}.",
            tool_names.join(", "),
        )
    };
    let system_msg = format!(
        "You are a helpful shell assistant. {} Current directory: {}",
        tools_hint, cwd,
    );

    let chat = ChatHandle::new(Some(&system_msg))?;

    let session_id = uuid::Uuid::new_v4().hyphenated().to_string();

    if let Some(ref db) = db_path {
        let db_str = db.to_string_lossy();
        chat.set_storage_db(&db_str, &session_id)?;
    }

    let content = vec![talu::router::ContentPart::Text(prompt.clone())];

    // Build backend and config based on local vs remote target
    let mut cfg = talu::router::GenerateConfig {
        max_tokens,
        seed,
        ..Default::default()
    };

    // model_display_name: used for session metadata at the end
    let model_display_name: String;

    let backend = match target {
        ModelTarget::Remote {
            provider: ref provider_name,
            model: model_id,
        } => {
            let provider_info = get_provider(provider_name)
                .ok_or_else(|| anyhow!("Unknown provider: {}", provider_name))?;

            let base_url = env::var(format!("{}_ENDPOINT", provider_name.to_uppercase()))
                .ok()
                .unwrap_or_else(|| provider_info.default_endpoint.to_string());

            let api_key = provider_info
                .api_key_env
                .as_ref()
                .and_then(|env_var| env::var(env_var).ok());

            let temperature = env::var("TEMPERATURE")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(0.7);
            cfg.temperature = temperature;

            model_display_name = format!("{}::{}", provider_name, model_id);

            InferenceBackend::new_openai_compatible(
                model_id,
                &base_url,
                api_key.as_deref(),
                120_000,
            )?
        }
        ModelTarget::Local(_) => {
            let resolved_model = resolve_model_for_inference(&model_arg)?;
            let gen_cfg = generation_config(&resolved_model)?;

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

            if (gen_cfg.do_sample || temperature_from_env.is_some()) && temperature > 0.0 {
                cfg.temperature = temperature;
                cfg.top_k = top_k;
                cfg.top_p = top_p;
            } else {
                cfg.temperature = 0.0;
            }

            model_display_name = std::path::Path::new(&resolved_model)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or(&resolved_model)
                .to_string();

            InferenceBackend::new(&resolved_model)?
        }
    };

    let tools_json = crate::agent::tools::shell_tool_schema(&tool_registry).to_string();
    crate::agent::run_shell(
        &chat,
        &content,
        &backend,
        &mut cfg,
        &tools_json,
        args.policy.as_deref(),
        &tool_registry,
        std::time::Duration::from_secs(args.tool_timeout),
    )?;

    if db_path.is_some() {
        let title = if prompt.len() > 50 {
            format!("{}...", &prompt[..47])
        } else {
            prompt.clone()
        };

        let _ = chat.notify_session_update(Some(&model_display_name), Some(&title), Some("active"));
    }

    Ok(())
}
