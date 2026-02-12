//! Shell tool definition and executor for the agent loop.

use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use anyhow::{anyhow, bail, Result};

/// OpenAI-style tool schema for the built-in `execute_command` tool.
fn execute_command_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": "Execute a shell command on the host and return its stdout and stderr output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute, e.g. 'ls -la' or 'find . -size +1G'"
                    }
                },
                "required": ["command"],
                "additionalProperties": false
            }
        }
    })
}

/// Runs a shell command and returns the combined stdout + stderr output.
///
/// Returns an error description string on spawn failure; never panics.
pub fn run_shell_command(cmd: &str) -> String {
    match Command::new("sh").arg("-c").arg(cmd).output() {
        Ok(output) => {
            let mut result = String::new();
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            if !stdout.is_empty() {
                result.push_str(&stdout);
            }
            if !stderr.is_empty() {
                if !result.is_empty() {
                    result.push('\n');
                }
                result.push_str("[stderr] ");
                result.push_str(&stderr);
            }
            if result.is_empty() {
                result.push_str("(no output)");
            }
            result
        }
        Err(e) => format!("Failed to execute command: {}", e),
    }
}

/// Reserved tool name for the built-in shell tool.
pub const RESERVED_TOOL_NAME: &str = "execute_command";

/// In-memory representation of a custom tool manifest.
#[derive(Debug, Clone)]
pub struct ToolManifest {
    pub name: String,
    pub function: serde_json::Value,
    pub runtime_command: Vec<String>,
}

/// Registry of custom tools keyed by tool name.
#[derive(Debug, Default)]
pub struct ToolRegistry {
    tools: HashMap<String, ToolManifest>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    pub fn get(&self, name: &str) -> Option<&ToolManifest> {
        self.tools.get(name)
    }

    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tools.keys().cloned().collect();
        names.sort();
        names
    }

    fn insert(&mut self, manifest: ToolManifest) -> Result<()> {
        if self.tools.contains_key(&manifest.name) {
            bail!("duplicate tool name '{}'", manifest.name);
        }
        self.tools.insert(manifest.name.clone(), manifest);
        Ok(())
    }
}

/// Build the OpenAI-style tools array with `execute_command` + custom tools.
pub fn shell_tool_schema(registry: &ToolRegistry) -> serde_json::Value {
    let mut tools = vec![execute_command_schema()];
    for name in registry.names() {
        if let Some(manifest) = registry.get(&name) {
            tools.push(serde_json::json!({
                "type": "function",
                "function": manifest.function.clone(),
            }));
        }
    }
    serde_json::Value::Array(tools)
}

/// Load custom tool manifests from paths (files or directories).
pub fn load_tools(paths: &[String]) -> Result<ToolRegistry> {
    let mut registry = ToolRegistry::new();
    if paths.is_empty() {
        return Ok(registry);
    }

    for raw_path in paths {
        let path = Path::new(raw_path);
        if path.is_dir() {
            let entries = fs::read_dir(path)
                .map_err(|e| anyhow!("Failed to read tools dir '{}': {}", raw_path, e))?;
            let mut files = Vec::new();
            for entry in entries {
                let entry = entry.map_err(|e| anyhow!("Failed to read tools dir entry: {}", e))?;
                let entry_path = entry.path();
                if entry_path.extension().and_then(|s| s.to_str()) == Some("json") {
                    files.push(entry_path);
                }
            }
            files.sort();
            for entry_path in files {
                let manifest = load_manifest(&entry_path)?;
                registry.insert(manifest)?;
            }
        } else {
            let manifest = load_manifest(path)?;
            registry.insert(manifest)?;
        }
    }

    Ok(registry)
}

fn load_manifest(path: &Path) -> Result<ToolManifest> {
    let text = fs::read_to_string(path)
        .map_err(|e| anyhow!("Failed to read manifest '{}': {}", path.display(), e))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| anyhow!("Failed to parse manifest '{}': {}", path.display(), e))?;

    let obj = value
        .as_object()
        .ok_or_else(|| anyhow!("manifest '{}' is not a JSON object", path.display()))?;

    let function = obj
        .get("function")
        .ok_or_else(|| anyhow!("manifest '{}' missing 'function' key", path.display()))?;
    let function_obj = function
        .as_object()
        .ok_or_else(|| anyhow!("manifest '{}' 'function' must be an object", path.display()))?;

    let name = function_obj
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("manifest '{}' function missing 'name'", path.display()))?;
    if name == RESERVED_TOOL_NAME {
        bail!("tool name '{}' is reserved", RESERVED_TOOL_NAME);
    }

    let parameters = function_obj.get("parameters").ok_or_else(|| {
        anyhow!(
            "manifest '{}' function missing 'parameters'",
            path.display()
        )
    })?;
    if !parameters.is_object() {
        bail!(
            "manifest '{}' function.parameters must be an object",
            path.display()
        );
    }

    let runtime = obj
        .get("runtime")
        .ok_or_else(|| anyhow!("manifest '{}' missing 'runtime' key", path.display()))?;
    let runtime_obj = runtime
        .as_object()
        .ok_or_else(|| anyhow!("manifest '{}' runtime must be an object", path.display()))?;
    let command_value = runtime_obj
        .get("command")
        .ok_or_else(|| anyhow!("manifest '{}' runtime missing 'command'", path.display()))?;
    let command_array = command_value.as_array().ok_or_else(|| {
        anyhow!(
            "manifest '{}' runtime.command must be an array",
            path.display()
        )
    })?;
    if command_array.is_empty() {
        bail!(
            "manifest '{}' runtime.command must be non-empty",
            path.display()
        );
    }

    let mut raw_command = Vec::with_capacity(command_array.len());
    for entry in command_array {
        let arg = entry.as_str().ok_or_else(|| {
            anyhow!(
                "manifest '{}' runtime.command entries must be strings",
                path.display()
            )
        })?;
        raw_command.push(arg.to_string());
    }

    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
    let resolved_command = resolve_command_paths(base_dir, &raw_command);

    warn_if_missing_command(&resolved_command);

    Ok(ToolManifest {
        name: name.to_string(),
        function: function.clone(),
        runtime_command: resolved_command,
    })
}

fn resolve_command_paths(base_dir: &Path, command: &[String]) -> Vec<String> {
    command
        .iter()
        .map(|arg| resolve_arg_path(base_dir, arg))
        .collect()
}

fn resolve_arg_path(base_dir: &Path, arg: &str) -> String {
    let candidate = base_dir.join(arg);
    if candidate.exists() {
        return candidate.to_string_lossy().to_string();
    }
    if arg.starts_with("./") || arg.starts_with("../") || arg.contains('/') {
        return candidate.to_string_lossy().to_string();
    }
    arg.to_string()
}

fn warn_if_missing_command(command: &[String]) {
    if command.is_empty() {
        return;
    }
    let exe = &command[0];
    if exe.starts_with('/') || exe.starts_with("./") || exe.starts_with("../") || exe.contains('/')
    {
        if !Path::new(exe).exists() {
            eprintln!(
                "\x1b[33m[tools] Warning:\x1b[0m command not found at '{}'",
                exe,
            );
        }
    }
}

/// Result of executing a custom tool.
pub struct ToolExecution {
    pub stdout: String,
    pub stderr: String,
    pub status: Option<i32>,
    pub timed_out: bool,
}

/// Runs a custom tool by piping JSON arguments to stdin.
pub fn run_custom_tool(
    manifest: &ToolManifest,
    arguments: &str,
    timeout: Duration,
) -> Result<ToolExecution> {
    let payload = validate_tool_arguments(manifest, arguments)?;

    let mut cmd = Command::new(&manifest.runtime_command[0]);
    if manifest.runtime_command.len() > 1 {
        cmd.args(&manifest.runtime_command[1..]);
    }
    let mut child = cmd
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| anyhow!("tool execution failed: {}", e))?;

    if let Some(stdin) = child.stdin.as_mut() {
        stdin
            .write_all(payload.as_bytes())
            .map_err(|e| anyhow!("tool stdin write failed: {}", e))?;
    }
    drop(child.stdin.take());

    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("tool stdout unavailable"))?;
    let mut stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("tool stderr unavailable"))?;

    let (stdout_tx, stdout_rx) = mpsc::channel();
    let (stderr_tx, stderr_rx) = mpsc::channel();
    let stdout_thread = thread::spawn(move || {
        let mut buf = Vec::new();
        let res = stdout.read_to_end(&mut buf);
        let _ = stdout_tx.send((res, buf));
    });
    let stderr_thread = thread::spawn(move || {
        let mut buf = Vec::new();
        let res = stderr.read_to_end(&mut buf);
        let _ = stderr_tx.send((res, buf));
    });

    let start = std::time::Instant::now();
    let mut timed_out = false;
    let status = loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|e| anyhow!("tool wait failed: {}", e))?
        {
            break status;
        }
        if start.elapsed() >= timeout {
            timed_out = true;
            let _ = child.kill();
            break child
                .wait()
                .map_err(|e| anyhow!("tool wait failed: {}", e))?;
        }
        thread::sleep(Duration::from_millis(10));
    };

    stdout_thread
        .join()
        .map_err(|_| anyhow!("tool stdout thread panicked"))?;
    stderr_thread
        .join()
        .map_err(|_| anyhow!("tool stderr thread panicked"))?;

    let (stdout_result, stdout_buf) = stdout_rx
        .recv()
        .map_err(|_| anyhow!("tool stdout channel closed"))?;
    let (stderr_result, stderr_buf) = stderr_rx
        .recv()
        .map_err(|_| anyhow!("tool stderr channel closed"))?;
    stdout_result.map_err(|e| anyhow!("tool stdout read failed: {}", e))?;
    stderr_result.map_err(|e| anyhow!("tool stderr read failed: {}", e))?;

    Ok(ToolExecution {
        stdout: String::from_utf8_lossy(&stdout_buf).to_string(),
        stderr: String::from_utf8_lossy(&stderr_buf).to_string(),
        status: status.code(),
        timed_out,
    })
}

fn validate_tool_arguments(manifest: &ToolManifest, arguments: &str) -> Result<String> {
    let function = manifest
        .function
        .as_object()
        .ok_or_else(|| anyhow!("tool '{}' has invalid function schema", manifest.name))?;
    let parameters = function
        .get("parameters")
        .ok_or_else(|| anyhow!("tool '{}' missing parameters", manifest.name))?;
    let params_obj = parameters
        .as_object()
        .ok_or_else(|| anyhow!("tool '{}' parameters must be an object", manifest.name))?;

    let mut value = if arguments.trim().is_empty() {
        serde_json::json!({})
    } else {
        serde_json::from_str::<serde_json::Value>(arguments)
            .map_err(|e| anyhow!("tool '{}' arguments invalid JSON: {}", manifest.name, e))?
    };

    let args_obj = value
        .as_object_mut()
        .ok_or_else(|| anyhow!("tool '{}' arguments must be a JSON object", manifest.name))?;

    let required = params_obj.get("required").and_then(|v| v.as_array());
    if let Some(required) = required {
        let mut missing = Vec::new();
        for item in required {
            if let Some(name) = item.as_str() {
                if !args_obj.contains_key(name) {
                    missing.push(name.to_string());
                }
            }
        }
        if !missing.is_empty() {
            bail!(
                "tool '{}' missing required arguments: {}",
                manifest.name,
                missing.join(", "),
            );
        }
    }

    let properties = params_obj.get("properties").and_then(|v| v.as_object());

    let additional = params_obj
        .get("additionalProperties")
        .and_then(|v| v.as_bool());
    if additional == Some(false) {
        if let Some(props) = properties {
            let extras: Vec<String> = args_obj
                .keys()
                .filter(|k| !props.contains_key(*k))
                .cloned()
                .collect();
            if !extras.is_empty() {
                bail!(
                    "tool '{}' has unexpected arguments: {}",
                    manifest.name,
                    extras.join(", "),
                );
            }
        }
    }

    if let Some(props) = properties {
        for (key, value) in args_obj.iter() {
            if let Some(schema) = props.get(key).and_then(|v| v.as_object()) {
                if let Some(expected) = schema.get("type") {
                    if !value_matches_type(value, expected) {
                        bail!(
                            "tool '{}' argument '{}' has invalid type",
                            manifest.name,
                            key,
                        );
                    }
                }
            }
        }
    }

    serde_json::to_string(&value).map_err(|e| {
        anyhow!(
            "tool '{}' arguments serialization failed: {}",
            manifest.name,
            e
        )
    })
}

fn value_matches_type(value: &serde_json::Value, expected: &serde_json::Value) -> bool {
    let types = match expected {
        serde_json::Value::String(t) => vec![t.as_str()],
        serde_json::Value::Array(arr) => arr.iter().filter_map(|v| v.as_str()).collect(),
        _ => Vec::new(),
    };

    if types.is_empty() {
        return true;
    }

    types.into_iter().any(|t| match t {
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "boolean" => value.is_boolean(),
        "object" => value.is_object(),
        "array" => value.is_array(),
        "null" => value.is_null(),
        _ => true,
    })
}
