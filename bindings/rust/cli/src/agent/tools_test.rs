use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use super::tools::{load_tools, run_custom_tool, ToolExecution, ToolManifest};

fn temp_dir(prefix: &str) -> PathBuf {
    let mut dir = std::env::temp_dir();
    let id = uuid::Uuid::new_v4().to_string();
    dir.push(format!("talu-cli-{}-{}", prefix, id));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_manifest(dir: &Path, filename: &str, name: &str, command: &[&str]) -> PathBuf {
    let manifest = serde_json::json!({
        "function": {
            "name": name,
            "description": "test",
            "parameters": { "type": "object", "properties": {}, "required": [] }
        },
        "runtime": {
            "command": command
        }
    });
    let path = dir.join(filename);
    fs::write(&path, serde_json::to_string(&manifest).unwrap()).unwrap();
    path
}

fn first_tool(manifest: &ToolManifest) -> &str {
    manifest
        .runtime_command
        .first()
        .map(String::as_str)
        .unwrap_or("")
}

fn manifest_with_params(command: &[&str], required: &[&str], arg_type: &str) -> ToolManifest {
    ToolManifest {
        name: "tool_test".to_string(),
        function: serde_json::json!({
            "name": "tool_test",
            "description": "test",
            "parameters": {
                "type": "object",
                "properties": { "arg": { "type": arg_type } },
                "required": required,
                "additionalProperties": false
            }
        }),
        runtime_command: command.iter().map(|s| s.to_string()).collect(),
    }
}

fn assert_exec(exec: ToolExecution, code: i32, stderr_contains: &str) {
    assert_eq!(exec.status, Some(code));
    assert!(exec.stderr.contains(stderr_contains));
}

#[test]
fn load_tools_from_directory() {
    let dir = temp_dir("tools");
    write_manifest(&dir, "tool_a.json", "tool_a", &["./script_a.sh"]);
    write_manifest(&dir, "tool_b.json", "tool_b", &["python3", "./tool_b.py"]);

    let registry = load_tools(&[dir.to_string_lossy().to_string()]).unwrap();
    assert!(registry.get("tool_a").is_some());
    assert!(registry.get("tool_b").is_some());
}

#[test]
fn resolves_relative_paths() {
    let dir = temp_dir("resolve");
    write_manifest(&dir, "tool_rel.json", "tool_rel", &["./tool.py"]);

    let registry = load_tools(&[dir.to_string_lossy().to_string()]).unwrap();
    let manifest = registry.get("tool_rel").unwrap();
    let cmd = first_tool(manifest);
    assert!(cmd.contains("tool.py"));
    assert!(cmd.contains(dir.to_string_lossy().as_ref()));
}

#[test]
fn rejects_duplicate_names() {
    let dir = temp_dir("dupe");
    let first = write_manifest(&dir, "tool_dup_a.json", "tool_dup", &["./a"]);
    let second = write_manifest(&dir, "tool_dup_b.json", "tool_dup", &["./b"]);

    let result = load_tools(&[
        first.to_string_lossy().to_string(),
        second.to_string_lossy().to_string(),
    ]);
    assert!(result.is_err());
}

#[test]
fn run_custom_tool_nonzero_exit() {
    let manifest = manifest_with_params(&["sh", "-c", "echo fail 1>&2; exit 2"], &[], "string");
    let exec = run_custom_tool(&manifest, "{}", Duration::from_secs(2)).unwrap();
    assert_exec(exec, 2, "fail");
}

#[test]
fn run_custom_tool_invalid_json_args() {
    let manifest = manifest_with_params(&["sh", "-c", "exit 0"], &[], "string");
    let result = run_custom_tool(&manifest, "{bad", Duration::from_secs(2));
    assert!(result.is_err());
}

#[test]
fn run_custom_tool_timeout() {
    let manifest = manifest_with_params(&["sh", "-c", "sleep 2"], &[], "string");
    let exec = run_custom_tool(&manifest, "{}", Duration::from_millis(50)).unwrap();
    assert!(exec.timed_out);
}

#[test]
fn run_custom_tool_type_mismatch() {
    let manifest = manifest_with_params(&["sh", "-c", "exit 0"], &[], "string");
    let result = run_custom_tool(&manifest, r#"{"arg": 123}"#, Duration::from_secs(2));
    assert!(result.is_err());
}
