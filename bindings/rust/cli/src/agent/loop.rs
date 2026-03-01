//! Shell agent: single-turn structured tool call.
//!
//! `talu agent` generates exactly one tool call via grammar-constrained
//! sampling. The built-in `execute_command` tool is executed with a
//! conservative shell whitelist; custom tools are executed via their
//! manifests. All tool calls require user confirmation.
//!
//! Safety model: whitelist, not blocklist. Only commands whose leading
//! executable appears in `ALLOWED_COMMANDS` are permitted. Everything
//! else is blocked by default. Pipe chains (`|`, `&&`, `||`, `;`) are
//! split and every segment must pass independently.

use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};

use anyhow::{anyhow, Result};
use serde_json::Value;

use talu::policy::Policy;
use talu::responses::ItemStatus;
use talu::responses::ResponsesView;
use talu::{ChatHandle, FinishReason, InferenceBackend};

use super::tools;
use super::tools::ToolRegistry;

/// Generates a single tool call, validates safety, asks permission, executes, exits.
///
/// `policy_path`: optional path to a JSON policy file. If `None`, uses the
/// built-in default-deny shell policy.
pub fn run_shell(
    chat: &ChatHandle,
    initial_content: &[talu::router::ContentPart],
    backend: &InferenceBackend,
    cfg: &mut talu::router::GenerateConfig,
    tools_json: &str,
    policy_path: Option<&str>,
    tool_registry: &ToolRegistry,
    tool_timeout: std::time::Duration,
) -> Result<()> {
    cfg.tools_json = Some(tools_json.to_string());

    // Load policy: user-provided file or built-in default.
    let policy = match policy_path {
        Some(path) => {
            let json = std::fs::read_to_string(path)
                .map_err(|e| anyhow!("Failed to read policy file '{}': {}", path, e))?;
            let json = add_custom_tool_allows(&json, tool_registry)?;
            let p = Policy::from_json(&json)
                .map_err(|e| anyhow!("Failed to parse policy file '{}': {}", path, e))?;
            eprintln!("\x1b[2m[policy] Loaded from {}\x1b[0m", path);
            p
        }
        None => {
            eprintln!("\x1b[2m[policy] Using built-in default-deny shell policy\x1b[0m");
            default_shell_policy(tool_registry)?
        }
    };
    chat.set_policy(Some(&policy))
        .map_err(|e| anyhow!("Failed to attach policy: {}", e))?;

    let has_custom_policy = policy_path.is_some();

    let result = talu::router::generate(chat, initial_content, backend, cfg)?;

    if result.error_code() != 0 {
        let msg =
            talu::error::last_error_message().unwrap_or_else(|| "generation failed".to_string());
        return Err(anyhow!(
            "Generation error: {} (code {})",
            msg,
            result.error_code()
        ));
    }

    // Check if the core policy denied the tool call.
    // The tool call item is the last item in the conversation.
    if result.finish_reason() == FinishReason::ToolCalls {
        let conv = chat.responses();
        let count = conv.item_count();
        if count > 0 {
            if let Ok(item) = conv.get_item(count - 1) {
                if item.status == ItemStatus::Failed {
                    if let Some(call) = extract_tool_call(&result, chat) {
                        let display = tool_call_display(&call);
                        eprintln!("\x1b[1;31m[POLICY DENIED]\x1b[0m \x1b[1m{}\x1b[0m", display,);
                    } else {
                        eprintln!("\x1b[1;31m[POLICY DENIED]\x1b[0m \x1b[1m<unknown>\x1b[0m",);
                    }
                    return Ok(());
                }
            }
        }
    }

    // Try structured tool calls first
    let tool_call = if result.finish_reason() == FinishReason::ToolCalls {
        extract_tool_call(&result, chat)
    } else {
        None
    };

    // Fallback: try to extract "command" from text output (model may emit JSON as text)
    let fallback_command = if tool_call.is_none() {
        result.text().and_then(|t| extract_command_from_json(&t))
    } else {
        None
    };

    match (tool_call, fallback_command) {
        (Some(call), _) => run_tool_call(
            &call,
            tool_registry,
            &policy,
            has_custom_policy,
            tool_timeout,
        ),
        (None, Some(command)) => run_shell_command(&command, &policy, has_custom_policy),
        _ => {
            eprintln!("No valid tool call generated.");
            Ok(())
        }
    }
}

fn run_shell_command(command: &str, policy: &Policy, has_custom_policy: bool) -> Result<()> {
    if command.is_empty() {
        eprintln!("No valid command generated.");
        return Ok(());
    }

    // Policy evaluation on the Rust side.
    //
    // The core policy fires inside commitToolCallAndReturn() but only when
    // the grammar sampler completes. Small models often emit tool-call JSON
    // as plain text (grammar incomplete), so the core never evaluates.
    // We evaluate here as well to cover that fallback path.
    use talu::policy::Effect;
    let action = talu::shell::normalize_command(command)
        .unwrap_or_else(|_| command.to_string());
    if policy.evaluate(&action) == Effect::Deny {
        eprintln!("\x1b[1;31m[POLICY DENIED]\x1b[0m \x1b[1m{}\x1b[0m", command,);
        return Ok(());
    }

    // Whitelist check via Zig core: defense-in-depth for the built-in default
    // policy. When a custom --policy is provided, the core policy is the
    // authority and the whitelist check is skipped.
    if !has_custom_policy {
        match talu::shell::check_command(command) {
            Ok(check) if !check.allowed => {
                let reason = check.reason.as_deref().unwrap_or("blocked by safety policy");
                eprintln!(
                    "\x1b[1;31m[BLOCKED]\x1b[0m \x1b[1m{}\x1b[0m\n  Reason: {}",
                    command, reason,
                );
                return Ok(());
            }
            _ => {}
        }
    }

    if !prompt_allow(&format!("\x1b[1;34m$\x1b[0m {}", command))? {
        eprintln!("\x1b[33m[denied]\x1b[0m");
        return Ok(());
    }

    let output = tools::run_shell_command(command);
    print!("{}", output);
    Ok(())
}

fn run_tool_call(
    call: &ToolCallRequest,
    tool_registry: &ToolRegistry,
    policy: &Policy,
    has_custom_policy: bool,
    tool_timeout: std::time::Duration,
) -> Result<()> {
    if call.name == tools::RESERVED_TOOL_NAME {
        if let Some(cmd) = extract_command_from_json(&call.arguments) {
            return run_shell_command(&cmd, policy, has_custom_policy);
        }
        eprintln!("No valid command generated.");
        return Ok(());
    }

    let manifest = tool_registry
        .get(&call.name)
        .ok_or_else(|| anyhow!("unknown tool '{}'", call.name))?;

    use talu::policy::Effect;
    let action = format!("tool:{}", call.name);
    if policy.evaluate(&action) == Effect::Deny {
        eprintln!(
            "\x1b[1;31m[POLICY DENIED]\x1b[0m \x1b[1m{}\x1b[0m",
            tool_call_display(call),
        );
        return Ok(());
    }

    let display = tool_call_display(call);
    if !prompt_allow(&display)? {
        eprintln!("\x1b[33m[denied]\x1b[0m");
        return Ok(());
    }

    let exec = tools::run_custom_tool(manifest, &call.arguments, tool_timeout)?;
    if !exec.stdout.is_empty() {
        print!("{}", exec.stdout);
    }
    if !exec.stderr.is_empty() {
        eprint!("{}", exec.stderr);
    }

    if exec.timed_out {
        eprintln!("\x1b[1;31m[TOOL TIMED OUT]\x1b[0m {}", call.name);
        return Ok(());
    }
    if let Some(code) = exec.status {
        if code != 0 {
            let stderr = if exec.stderr.trim().is_empty() {
                "<no stderr>"
            } else {
                exec.stderr.trim()
            };
            eprintln!(
                "\x1b[1;31m[TOOL ERROR]\x1b[0m tool returned error (exit {}): {}",
                code, stderr,
            );
        }
    }

    Ok(())
}

fn prompt_allow(display: &str) -> Result<bool> {
    // Always ask permission.
    // Read from /dev/tty so piped stdin doesn't skip the prompt.
    eprintln!("{}", display);
    eprint!("  Allow? [y/N] ");
    io::stderr().flush()?;

    let answer = match File::open("/dev/tty") {
        Ok(tty) => {
            let mut line = String::new();
            BufReader::new(tty).read_line(&mut line)?;
            line.trim().to_lowercase()
        }
        Err(_) => {
            // No TTY (e.g. CI, cron) — refuse by default
            String::new()
        }
    };

    Ok(answer == "y" || answer == "yes")
}

#[derive(Debug)]
struct ToolCallRequest {
    name: String,
    arguments: String,
}

fn extract_tool_call(result: &talu::GenerateResult, chat: &ChatHandle) -> Option<ToolCallRequest> {
    let calls = result.tool_calls();
    let tc = calls.first()?;
    let arguments = if tc.arguments.is_empty() && tc.item_index > 0 {
        let conv = chat.responses();
        conv.get_function_call(tc.item_index)
            .map(|fc| fc.arguments)
            .ok()?
    } else {
        tc.arguments.clone()
    };
    Some(ToolCallRequest {
        name: tc.name.clone(),
        arguments,
    })
}

fn tool_call_display(call: &ToolCallRequest) -> String {
    if call.name == tools::RESERVED_TOOL_NAME {
        if let Some(cmd) = extract_command_from_json(&call.arguments) {
            return cmd;
        }
    }
    let args = if call.arguments.trim().is_empty() {
        "{}"
    } else {
        call.arguments.as_str()
    };
    format!("{}({})", call.name, args)
}

/// Extracts the first `"command"` string value found in JSON text.
/// Ignores nested objects, repeated keys, and other garbage from small models.
fn extract_command_from_json(text: &str) -> Option<String> {
    let needle = "\"command\"";
    let idx = text.find(needle)?;
    let after = &text[idx + needle.len()..];

    let after = after.trim_start();
    let after = after.strip_prefix(':')?;
    let after = after.trim_start();

    let after = after.strip_prefix('"')?;
    let end = after.find('"')?;
    let cmd = &after[..end];

    if cmd.is_empty() {
        return None;
    }
    Some(cmd.to_string())
}

/// Builds the default shell policy (IAM-style, default-deny).
///
/// Delegates to the Zig core for the whitelist + deny rules, then
/// injects custom tool allows on top.
pub fn default_shell_policy(tool_registry: &ToolRegistry) -> Result<Policy> {
    let json = talu::shell::default_policy_json()
        .map_err(|e| anyhow!("Failed to get default policy JSON: {}", e))?;
    let json = add_custom_tool_allows(&json, tool_registry)?;
    Policy::from_json(&json).map_err(|e| anyhow!("Failed to create shell policy: {}", e))
}

fn add_custom_tool_allows(json: &str, tool_registry: &ToolRegistry) -> Result<String> {
    if tool_registry.is_empty() {
        return Ok(json.to_string());
    }
    let mut value: Value =
        serde_json::from_str(json).map_err(|e| anyhow!("Failed to parse policy JSON: {}", e))?;
    let obj = value
        .as_object_mut()
        .ok_or_else(|| anyhow!("policy JSON must be an object"))?;
    let statements = obj
        .entry("statements")
        .or_insert_with(|| Value::Array(Vec::new()));
    let arr = statements
        .as_array_mut()
        .ok_or_else(|| anyhow!("policy JSON 'statements' must be an array"))?;
    for name in tool_registry.names() {
        arr.push(serde_json::json!({
            "effect": "allow",
            "action": format!("tool:{}", name),
        }));
    }
    serde_json::to_string(&value).map_err(|e| anyhow!("Failed to serialize policy JSON: {}", e))
}
