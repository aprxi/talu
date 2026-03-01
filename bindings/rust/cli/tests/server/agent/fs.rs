use crate::server::common::{delete_json, get, post_json, ServerConfig, ServerTestContext};
use tempfile::TempDir;

fn config_with_workspace(workspace: &TempDir) -> ServerConfig {
    let mut cfg = ServerConfig::new();
    cfg.env_vars.push((
        "TALU_WORKSPACE_DIR".to_string(),
        workspace.path().to_string_lossy().to_string(),
    ));
    cfg
}

/// Helper: create a server with a workspace, write a file, return (ctx, workspace).
fn setup_with_file(
    rel_path: &str,
    content: &str,
) -> (ServerTestContext, TempDir) {
    let workspace = TempDir::new().expect("workspace");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));
    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/write",
        &serde_json::json!({
            "path": rel_path,
            "content": content,
            "mkdir": true
        }),
    );
    assert_eq!(resp.status, 200, "setup write failed: {}", resp.body);
    (ctx, workspace)
}

#[test]
fn agent_fs_write_read_edit_roundtrip() {
    let workspace = TempDir::new().expect("workspace");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let write_resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/write",
        &serde_json::json!({
            "path": "notes/main.txt",
            "content": "hello",
            "encoding": "utf-8",
            "mkdir": true
        }),
    );
    assert_eq!(write_resp.status, 200, "body: {}", write_resp.body);
    assert_eq!(write_resp.json()["bytes_written"], 5);

    let read_resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/read",
        &serde_json::json!({
            "path": "notes/main.txt"
        }),
    );
    assert_eq!(read_resp.status, 200, "body: {}", read_resp.body);
    assert_eq!(read_resp.json()["content"], "hello");

    let edit_resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/edit",
        &serde_json::json!({
            "path": "notes/main.txt",
            "old_text": "hello",
            "new_text": "world"
        }),
    );
    assert_eq!(edit_resp.status, 200, "body: {}", edit_resp.body);
    assert_eq!(edit_resp.json()["replacements"], 1);

    let read_after = post_json(
        ctx.addr(),
        "/v1/agent/fs/read",
        &serde_json::json!({
            "path": "notes/main.txt"
        }),
    );
    assert_eq!(read_after.status, 200, "body: {}", read_after.body);
    assert_eq!(read_after.json()["content"], "world");
}

#[test]
fn agent_fs_edit_rejects_empty_old_text_via_core_validation() {
    let workspace = TempDir::new().expect("workspace");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let write_resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/write",
        &serde_json::json!({
            "path": "notes/main.txt",
            "content": "hello",
            "encoding": "utf-8",
            "mkdir": true
        }),
    );
    assert_eq!(write_resp.status, 200, "body: {}", write_resp.body);

    let edit_resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/edit",
        &serde_json::json!({
            "path": "notes/main.txt",
            "old_text": "",
            "new_text": "world"
        }),
    );
    assert_eq!(edit_resp.status, 400, "body: {}", edit_resp.body);
    assert_eq!(edit_resp.json()["error"]["code"], "invalid_request");
}

#[test]
fn agent_fs_stat_list_rename_and_remove() {
    let workspace = TempDir::new().expect("workspace");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let mkdir = post_json(
        ctx.addr(),
        "/v1/agent/fs/mkdir",
        &serde_json::json!({
            "path": "src/new_module",
            "recursive": true
        }),
    );
    assert_eq!(mkdir.status, 200, "body: {}", mkdir.body);

    let write = post_json(
        ctx.addr(),
        "/v1/agent/fs/write",
        &serde_json::json!({
            "path": "src/new_module/file.txt",
            "content": "abc",
            "mkdir": true
        }),
    );
    assert_eq!(write.status, 200, "body: {}", write.body);

    let stat = post_json(
        ctx.addr(),
        "/v1/agent/fs/stat",
        &serde_json::json!({
            "path": "src/new_module/file.txt"
        }),
    );
    assert_eq!(stat.status, 200, "body: {}", stat.body);
    assert_eq!(stat.json()["exists"], true);
    assert_eq!(stat.json()["is_file"], true);

    let ls = post_json(
        ctx.addr(),
        "/v1/agent/fs/ls",
        &serde_json::json!({
            "path": "src",
            "glob": "*.txt",
            "recursive": true
        }),
    );
    assert_eq!(ls.status, 200, "body: {}", ls.body);
    let ls_json = ls.json();
    let entries = ls_json["entries"].as_array().expect("entries array");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0]["name"], "file.txt");

    let rename = post_json(
        ctx.addr(),
        "/v1/agent/fs/rename",
        &serde_json::json!({
            "from": "src/new_module/file.txt",
            "to": "src/new_module/file_renamed.txt"
        }),
    );
    assert_eq!(rename.status, 200, "body: {}", rename.body);

    let rm = crate::server::common::delete_json(
        ctx.addr(),
        "/v1/agent/fs/rm",
        &serde_json::json!({
            "path": "src",
            "recursive": true
        }),
    );
    assert_eq!(rm.status, 200, "body: {}", rm.body);
}

#[test]
fn agent_fs_blocks_outside_workspace_access() {
    let workspace = TempDir::new().expect("workspace");
    let outside = TempDir::new().expect("outside");
    let outside_file = outside.path().join("outside.txt");
    std::fs::write(&outside_file, "blocked").expect("write outside file");

    let ctx = ServerTestContext::new(config_with_workspace(&workspace));
    let read = post_json(
        ctx.addr(),
        "/v1/agent/fs/read",
        &serde_json::json!({
            "path": outside_file.to_string_lossy().to_string()
        }),
    );
    assert_eq!(read.status, 403, "body: {}", read.body);
    assert_eq!(read.json()["error"]["code"], "permission_denied");
}

#[test]
fn agent_fs_read_missing_file_returns_not_found() {
    let workspace = TempDir::new().expect("workspace");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let read = post_json(
        ctx.addr(),
        "/v1/agent/fs/read",
        &serde_json::json!({
            "path": "missing.txt"
        }),
    );
    assert_eq!(read.status, 404, "body: {}", read.body);
    assert_eq!(read.json()["error"]["code"], "not_found");
}

#[test]
fn agent_fs_read_directory_returns_is_directory() {
    let workspace = TempDir::new().expect("workspace");
    std::fs::create_dir_all(workspace.path().join("somedir")).expect("create dir");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let read = post_json(
        ctx.addr(),
        "/v1/agent/fs/read",
        &serde_json::json!({
            "path": "somedir"
        }),
    );
    assert_eq!(read.status, 400, "body: {}", read.body);
    assert_eq!(read.json()["error"]["code"], "is_directory");
}

#[test]
fn agent_fs_remove_non_empty_dir_without_recursive_returns_conflict() {
    let workspace = TempDir::new().expect("workspace");
    std::fs::create_dir_all(workspace.path().join("dir")).expect("create dir");
    std::fs::write(workspace.path().join("dir/file.txt"), "x").expect("write file");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let rm = crate::server::common::delete_json(
        ctx.addr(),
        "/v1/agent/fs/rm",
        &serde_json::json!({
            "path": "dir",
            "recursive": false
        }),
    );
    assert_eq!(rm.status, 409, "body: {}", rm.body);
    assert_eq!(rm.json()["error"]["code"], "not_empty");
}

#[test]
fn agent_fs_read_max_bytes_exceeded_returns_too_large() {
    let workspace = TempDir::new().expect("workspace");
    std::fs::write(workspace.path().join("big.txt"), "0123456789").expect("write file");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let read = post_json(
        ctx.addr(),
        "/v1/agent/fs/read",
        &serde_json::json!({
            "path": "big.txt",
            "max_bytes": 4
        }),
    );
    assert_eq!(read.status, 413, "body: {}", read.body);
    assert_eq!(read.json()["error"]["code"], "too_large");
}

#[test]
fn agent_fs_write_and_read_base64_roundtrip() {
    let workspace = TempDir::new().expect("workspace");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let write_resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/write",
        &serde_json::json!({
            "path": "bin/payload.bin",
            "content": "aGVsbG8Ad29ybGQ=",
            "encoding": "base64",
            "mkdir": true
        }),
    );
    assert_eq!(write_resp.status, 200, "body: {}", write_resp.body);

    let read_resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/read",
        &serde_json::json!({
            "path": "bin/payload.bin",
            "encoding": "base64"
        }),
    );
    assert_eq!(read_resp.status, 200, "body: {}", read_resp.body);
    assert_eq!(read_resp.json()["content"], "aGVsbG8Ad29ybGQ=");
    assert_eq!(read_resp.json()["size"], 11);
}

#[test]
fn agent_fs_edit_replace_all_replaces_every_match() {
    let workspace = TempDir::new().expect("workspace");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let write_resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/write",
        &serde_json::json!({
            "path": "notes/multi.txt",
            "content": "foo foo foo",
            "mkdir": true
        }),
    );
    assert_eq!(write_resp.status, 200, "body: {}", write_resp.body);

    let edit_resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/edit",
        &serde_json::json!({
            "path": "notes/multi.txt",
            "old_text": "foo",
            "new_text": "bar",
            "replace_all": true
        }),
    );
    assert_eq!(edit_resp.status, 200, "body: {}", edit_resp.body);
    assert_eq!(edit_resp.json()["replacements"], 3);

    let read_resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/read",
        &serde_json::json!({
            "path": "notes/multi.txt"
        }),
    );
    assert_eq!(read_resp.status, 200, "body: {}", read_resp.body);
    assert_eq!(read_resp.json()["content"], "bar bar bar");
}

#[test]
fn agent_fs_read_non_utf8_file_returns_io_error() {
    let workspace = TempDir::new().expect("workspace");
    std::fs::write(
        workspace.path().join("binary.bin"),
        [0xff_u8, 0xfe_u8, 0xfd_u8],
    )
    .expect("write non-utf8");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let read = post_json(
        ctx.addr(),
        "/v1/agent/fs/read",
        &serde_json::json!({
            "path": "binary.bin"
        }),
    );
    assert_eq!(read.status, 500, "body: {}", read.body);
    assert_eq!(read.json()["error"]["code"], "io_error");
}

// ---------------------------------------------------------------------------
// Read: response shape and defaults
// ---------------------------------------------------------------------------

#[test]
fn agent_fs_read_response_has_correct_shape() {
    let (ctx, _ws) = setup_with_file("src/lib.rs", "fn main() {}");

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/read",
        &serde_json::json!({ "path": "src/lib.rs" }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let j = resp.json();
    assert_eq!(j["path"], "src/lib.rs", "path should be workspace-relative");
    assert_eq!(j["content"], "fn main() {}");
    assert_eq!(j["encoding"], "utf-8", "default encoding");
    assert_eq!(j["size"], 12);
    assert_eq!(j["truncated"], false);
}

#[test]
fn agent_fs_read_empty_file() {
    let (ctx, _ws) = setup_with_file("empty.txt", "");

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/read",
        &serde_json::json!({ "path": "empty.txt" }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(resp.json()["content"], "");
    assert_eq!(resp.json()["size"], 0);
}

#[test]
fn agent_fs_read_with_sufficient_max_bytes() {
    let (ctx, _ws) = setup_with_file("small.txt", "hi");

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/read",
        &serde_json::json!({ "path": "small.txt", "max_bytes": 1024 }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(resp.json()["content"], "hi");
}

// ---------------------------------------------------------------------------
// Read: sandbox enforcement
// ---------------------------------------------------------------------------

#[test]
fn agent_fs_read_dot_dot_traversal_blocked() {
    let workspace = TempDir::new().expect("workspace");
    let outside = TempDir::new().expect("outside");
    std::fs::write(outside.path().join("secret.txt"), "leaked").expect("write");

    let ctx = ServerTestContext::new(config_with_workspace(&workspace));
    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/read",
        &serde_json::json!({ "path": "../secret.txt" }),
    );
    // The `../` path resolves to nothing inside the workspace → 404.
    // The key assertion: it never returns 200 with content from outside.
    assert_eq!(resp.status, 404, "path traversal must be blocked: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "not_found");
}

// ---------------------------------------------------------------------------
// Write: edge cases
// ---------------------------------------------------------------------------

#[test]
fn agent_fs_write_without_mkdir_fails_when_parent_missing() {
    let workspace = TempDir::new().expect("workspace");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/write",
        &serde_json::json!({
            "path": "deep/nested/file.txt",
            "content": "hello"
            // mkdir defaults to false
        }),
    );
    assert_eq!(resp.status, 404, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "not_found");
}

#[test]
fn agent_fs_write_overwrites_existing_file() {
    let (ctx, _ws) = setup_with_file("data.txt", "old");

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/write",
        &serde_json::json!({ "path": "data.txt", "content": "new" }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(resp.json()["bytes_written"], 3);

    let read = post_json(
        ctx.addr(),
        "/v1/agent/fs/read",
        &serde_json::json!({ "path": "data.txt" }),
    );
    assert_eq!(read.json()["content"], "new", "content should be replaced");
}

#[test]
fn agent_fs_write_outside_workspace_blocked() {
    let workspace = TempDir::new().expect("workspace");
    let outside = TempDir::new().expect("outside");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/write",
        &serde_json::json!({
            "path": outside.path().join("hack.txt").to_string_lossy().to_string(),
            "content": "pwned"
        }),
    );
    assert_eq!(resp.status, 403, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "permission_denied");
}

#[test]
fn agent_fs_write_invalid_base64_returns_400() {
    let workspace = TempDir::new().expect("workspace");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/write",
        &serde_json::json!({
            "path": "file.bin",
            "content": "not valid base64 !!!",
            "encoding": "base64",
            "mkdir": true
        }),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
}

#[test]
fn agent_fs_write_invalid_encoding_returns_400() {
    let workspace = TempDir::new().expect("workspace");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/write",
        &serde_json::json!({
            "path": "file.txt",
            "content": "hello",
            "encoding": "latin-1",
            "mkdir": true
        }),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
}

#[test]
fn agent_fs_write_response_path_is_workspace_relative() {
    let workspace = TempDir::new().expect("workspace");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    // Write using absolute path
    let abs_path = workspace.path().join("abs.txt");
    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/write",
        &serde_json::json!({
            "path": abs_path.to_string_lossy().to_string(),
            "content": "test"
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(
        resp.json()["path"], "abs.txt",
        "response path should be workspace-relative even when request used absolute"
    );
}

// ---------------------------------------------------------------------------
// Edit: edge cases and error paths
// ---------------------------------------------------------------------------

#[test]
fn agent_fs_edit_missing_file_returns_not_found() {
    let workspace = TempDir::new().expect("workspace");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/edit",
        &serde_json::json!({
            "path": "ghost.txt",
            "old_text": "a",
            "new_text": "b"
        }),
    );
    assert_eq!(resp.status, 404, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "not_found");
}

#[test]
fn agent_fs_edit_no_match_returns_error() {
    let (ctx, _ws) = setup_with_file("file.txt", "hello world");

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/edit",
        &serde_json::json!({
            "path": "file.txt",
            "old_text": "MISSING_TEXT",
            "new_text": "replacement"
        }),
    );
    assert_eq!(resp.status, 500, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "io_error");
}

#[test]
fn agent_fs_edit_multiple_matches_without_replace_all_returns_error() {
    let (ctx, _ws) = setup_with_file("multi.txt", "aaa bbb aaa");

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/edit",
        &serde_json::json!({
            "path": "multi.txt",
            "old_text": "aaa",
            "new_text": "ccc"
            // replace_all defaults to false → single-match mode
        }),
    );
    assert_eq!(resp.status, 500, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "io_error");
}

#[test]
fn agent_fs_edit_preserves_surrounding_content() {
    let (ctx, _ws) = setup_with_file("code.py", "def foo():\n    pass\n\ndef bar():\n    pass\n");

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/edit",
        &serde_json::json!({
            "path": "code.py",
            "old_text": "def foo():\n    pass",
            "new_text": "def foo():\n    return 42"
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(resp.json()["replacements"], 1);

    let read = post_json(
        ctx.addr(),
        "/v1/agent/fs/read",
        &serde_json::json!({ "path": "code.py" }),
    );
    let content = read.json()["content"].as_str().unwrap().to_string();
    assert!(content.contains("return 42"), "edit should apply");
    assert!(content.contains("def bar():\n    pass"), "surrounding content must be preserved");
}

#[test]
fn agent_fs_edit_outside_workspace_blocked() {
    let workspace = TempDir::new().expect("workspace");
    let outside = TempDir::new().expect("outside");
    std::fs::write(outside.path().join("target.txt"), "data").expect("write");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/edit",
        &serde_json::json!({
            "path": outside.path().join("target.txt").to_string_lossy().to_string(),
            "old_text": "data",
            "new_text": "hacked"
        }),
    );
    assert_eq!(resp.status, 403, "body: {}", resp.body);
}

// ---------------------------------------------------------------------------
// Stat: comprehensive field validation
// ---------------------------------------------------------------------------

#[test]
fn agent_fs_stat_existing_file_returns_full_metadata() {
    let (ctx, _ws) = setup_with_file("info.txt", "twelve chars");

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/stat",
        &serde_json::json!({ "path": "info.txt" }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let j = resp.json();
    assert_eq!(j["exists"], true);
    assert_eq!(j["is_file"], true);
    assert_eq!(j["is_dir"], false);
    assert_eq!(j["is_symlink"], false);
    assert_eq!(j["size"], 12);
    assert!(j["mode"].is_string(), "mode should be octal string");
    assert!(j["modified_at"].as_i64().unwrap() > 0, "modified_at should be set");
}

#[test]
fn agent_fs_stat_directory() {
    let workspace = TempDir::new().expect("workspace");
    std::fs::create_dir_all(workspace.path().join("mydir")).expect("mkdir");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/stat",
        &serde_json::json!({ "path": "mydir" }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let j = resp.json();
    assert_eq!(j["exists"], true);
    assert_eq!(j["is_file"], false);
    assert_eq!(j["is_dir"], true);
}

#[test]
fn agent_fs_stat_missing_path_returns_exists_false() {
    let workspace = TempDir::new().expect("workspace");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/stat",
        &serde_json::json!({ "path": "nonexistent.txt" }),
    );
    assert_eq!(resp.status, 200, "stat on missing should be 200 with exists=false: {}", resp.body);
    let j = resp.json();
    assert_eq!(j["exists"], false);
    assert_eq!(j["is_file"], false);
    assert_eq!(j["is_dir"], false);
    assert_eq!(j["size"], 0);
}

// ---------------------------------------------------------------------------
// List: variations
// ---------------------------------------------------------------------------

#[test]
fn agent_fs_list_without_glob_returns_all_entries() {
    let workspace = TempDir::new().expect("workspace");
    std::fs::write(workspace.path().join("a.txt"), "a").expect("write");
    std::fs::write(workspace.path().join("b.rs"), "b").expect("write");
    std::fs::create_dir(workspace.path().join("sub")).expect("mkdir");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/ls",
        &serde_json::json!({ "path": "." }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let j = resp.json();
    let entries = j["entries"].as_array().expect("entries");
    let names: Vec<&str> = entries.iter().map(|e| e["name"].as_str().unwrap()).collect();
    assert!(names.contains(&"a.txt"), "missing a.txt in {names:?}");
    assert!(names.contains(&"b.rs"), "missing b.rs in {names:?}");
    assert!(names.contains(&"sub"), "missing sub in {names:?}");
}

#[test]
fn agent_fs_list_non_recursive_does_not_descend() {
    let workspace = TempDir::new().expect("workspace");
    std::fs::create_dir_all(workspace.path().join("parent/child")).expect("mkdirs");
    std::fs::write(workspace.path().join("parent/top.txt"), "t").expect("write");
    std::fs::write(workspace.path().join("parent/child/deep.txt"), "d").expect("write");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/ls",
        &serde_json::json!({
            "path": "parent",
            "glob": "*.txt",
            "recursive": false
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let j = resp.json();
    let entries = j["entries"].as_array().expect("entries");
    assert_eq!(entries.len(), 1, "should only find top.txt");
    assert_eq!(entries[0]["name"], "top.txt");
}

#[test]
fn agent_fs_list_recursive_finds_nested_files() {
    let workspace = TempDir::new().expect("workspace");
    std::fs::create_dir_all(workspace.path().join("a/b")).expect("mkdirs");
    std::fs::write(workspace.path().join("root.txt"), "r").expect("write");
    std::fs::write(workspace.path().join("a/mid.txt"), "m").expect("write");
    std::fs::write(workspace.path().join("a/b/deep.txt"), "d").expect("write");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/ls",
        &serde_json::json!({
            "path": ".",
            "glob": "*.txt",
            "recursive": true
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let j = resp.json();
    let entries = j["entries"].as_array().expect("entries");
    assert_eq!(entries.len(), 3, "should find all 3 txt files recursively");
}

#[test]
fn agent_fs_list_with_limit_truncates() {
    let workspace = TempDir::new().expect("workspace");
    for i in 0..5u32 {
        std::fs::write(workspace.path().join(format!("f{i}.txt")), "x").expect("write");
    }
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/ls",
        &serde_json::json!({
            "path": ".",
            "glob": "*.txt",
            "limit": 3
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let j = resp.json();
    let entries = j["entries"].as_array().expect("entries");
    assert_eq!(entries.len(), 3);
    assert_eq!(j["truncated"], true, "should indicate truncation");
}

#[test]
fn agent_fs_list_empty_directory() {
    let workspace = TempDir::new().expect("workspace");
    std::fs::create_dir(workspace.path().join("empty")).expect("mkdir");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/ls",
        &serde_json::json!({ "path": "empty" }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let j = resp.json();
    let entries = j["entries"].as_array().expect("entries");
    assert_eq!(entries.len(), 0);
    assert_eq!(j["truncated"], false);
}

#[test]
fn agent_fs_list_entry_has_expected_fields() {
    let (ctx, _ws) = setup_with_file("doc.md", "# Title");

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/ls",
        &serde_json::json!({ "path": "." }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let j = resp.json();
    let entries = j["entries"].as_array().expect("entries");
    let entry = &entries[0];
    assert!(entry["name"].is_string(), "name");
    assert!(entry["path"].is_string(), "path");
    assert!(entry.get("is_dir").is_some(), "is_dir");
    assert!(entry.get("is_symlink").is_some(), "is_symlink");
    assert!(entry.get("size").is_some(), "size");
    assert!(entry.get("modified_at").is_some(), "modified_at");
}

// ---------------------------------------------------------------------------
// Mkdir: edge cases
// ---------------------------------------------------------------------------

#[test]
fn agent_fs_mkdir_non_recursive_fails_when_parent_missing() {
    let workspace = TempDir::new().expect("workspace");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/mkdir",
        &serde_json::json!({
            "path": "a/b/c",
            "recursive": false
        }),
    );
    assert_eq!(resp.status, 404, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "not_found");
}

#[test]
fn agent_fs_mkdir_recursive_creates_full_tree() {
    let workspace = TempDir::new().expect("workspace");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/mkdir",
        &serde_json::json!({
            "path": "x/y/z",
            "recursive": true
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    // Verify with stat
    let stat = post_json(
        ctx.addr(),
        "/v1/agent/fs/stat",
        &serde_json::json!({ "path": "x/y/z" }),
    );
    assert_eq!(stat.json()["exists"], true);
    assert_eq!(stat.json()["is_dir"], true);
}

// ---------------------------------------------------------------------------
// Rename: edge cases
// ---------------------------------------------------------------------------

#[test]
fn agent_fs_rename_missing_source_returns_not_found() {
    let workspace = TempDir::new().expect("workspace");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/rename",
        &serde_json::json!({
            "from": "does_not_exist.txt",
            "to": "target.txt"
        }),
    );
    assert_eq!(resp.status, 404, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "not_found");
}

#[test]
fn agent_fs_rename_outside_workspace_blocked() {
    let workspace = TempDir::new().expect("workspace");
    let outside = TempDir::new().expect("outside");
    std::fs::write(workspace.path().join("local.txt"), "data").expect("write");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/rename",
        &serde_json::json!({
            "from": "local.txt",
            "to": outside.path().join("escaped.txt").to_string_lossy().to_string()
        }),
    );
    assert_eq!(resp.status, 403, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "permission_denied");
}

#[test]
fn agent_fs_rename_response_paths_are_workspace_relative() {
    let (ctx, ws) = setup_with_file("old.txt", "content");

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/rename",
        &serde_json::json!({
            "from": ws.path().join("old.txt").to_string_lossy().to_string(),
            "to": ws.path().join("new.txt").to_string_lossy().to_string()
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(resp.json()["from"], "old.txt");
    assert_eq!(resp.json()["to"], "new.txt");
}

// ---------------------------------------------------------------------------
// Remove: edge cases
// ---------------------------------------------------------------------------

#[test]
fn agent_fs_remove_single_file() {
    let (ctx, _ws) = setup_with_file("deleteme.txt", "bye");

    let resp = delete_json(
        ctx.addr(),
        "/v1/agent/fs/rm",
        &serde_json::json!({ "path": "deleteme.txt" }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(resp.json()["removed"], true);

    // Verify it's gone
    let stat = post_json(
        ctx.addr(),
        "/v1/agent/fs/stat",
        &serde_json::json!({ "path": "deleteme.txt" }),
    );
    assert_eq!(stat.json()["exists"], false);
}

#[test]
fn agent_fs_remove_missing_path_returns_not_found() {
    let workspace = TempDir::new().expect("workspace");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let resp = delete_json(
        ctx.addr(),
        "/v1/agent/fs/rm",
        &serde_json::json!({ "path": "ghost.txt" }),
    );
    assert_eq!(resp.status, 404, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "not_found");
}

#[test]
fn agent_fs_remove_outside_workspace_blocked() {
    let workspace = TempDir::new().expect("workspace");
    let outside = TempDir::new().expect("outside");
    std::fs::write(outside.path().join("keep.txt"), "safe").expect("write");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let resp = delete_json(
        ctx.addr(),
        "/v1/agent/fs/rm",
        &serde_json::json!({
            "path": outside.path().join("keep.txt").to_string_lossy().to_string()
        }),
    );
    assert_eq!(resp.status, 403, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "permission_denied");
}

// ---------------------------------------------------------------------------
// Cross-cutting: invalid JSON
// ---------------------------------------------------------------------------

#[test]
fn agent_fs_invalid_json_body_returns_400() {
    let workspace = TempDir::new().expect("workspace");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let resp = crate::server::common::send_request(
        ctx.addr(),
        "POST",
        "/v1/agent/fs/read",
        &[("content-type", "application/json")],
        Some("{not valid json"),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_json");
}

#[test]
fn agent_fs_missing_required_field_returns_400() {
    let workspace = TempDir::new().expect("workspace");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    // Read without path field
    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/read",
        &serde_json::json!({}),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_json");
}

// ---------------------------------------------------------------------------
// Cross-cutting: wrong HTTP method
// ---------------------------------------------------------------------------

#[test]
fn agent_fs_wrong_method_returns_501() {
    let workspace = TempDir::new().expect("workspace");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    // GET on a POST-only endpoint
    let resp = get(ctx.addr(), "/v1/agent/fs/read");
    assert_eq!(resp.status, 501, "body: {}", resp.body);
}

// ===========================================================================
// Security: symlink sandbox escape
// ===========================================================================

#[test]
fn agent_fs_read_symlink_outside_workspace_blocked() {
    let workspace = TempDir::new().expect("workspace");
    let outside = TempDir::new().expect("outside");
    std::fs::write(outside.path().join("secret.txt"), "leaked").expect("write");

    // Create symlink inside workspace pointing outside
    #[cfg(unix)]
    std::os::unix::fs::symlink(
        outside.path().join("secret.txt"),
        workspace.path().join("escape.txt"),
    )
    .expect("symlink");

    let ctx = ServerTestContext::new(config_with_workspace(&workspace));
    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/read",
        &serde_json::json!({ "path": "escape.txt" }),
    );
    assert_eq!(resp.status, 403, "symlink escape must be blocked: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "permission_denied");
}

#[test]
fn agent_fs_edit_symlink_outside_workspace_blocked() {
    let workspace = TempDir::new().expect("workspace");
    let outside = TempDir::new().expect("outside");
    std::fs::write(outside.path().join("target.txt"), "original").expect("write");

    #[cfg(unix)]
    std::os::unix::fs::symlink(
        outside.path().join("target.txt"),
        workspace.path().join("link.txt"),
    )
    .expect("symlink");

    let ctx = ServerTestContext::new(config_with_workspace(&workspace));
    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/edit",
        &serde_json::json!({
            "path": "link.txt",
            "old_text": "original",
            "new_text": "hacked"
        }),
    );
    assert_eq!(resp.status, 403, "symlink edit escape must be blocked: {}", resp.body);

    // Verify the outside file was not modified
    let content = std::fs::read_to_string(outside.path().join("target.txt")).expect("read");
    assert_eq!(content, "original", "outside file must not be modified");
}

#[test]
fn agent_fs_remove_symlink_outside_workspace_blocked() {
    let workspace = TempDir::new().expect("workspace");
    let outside = TempDir::new().expect("outside");
    std::fs::write(outside.path().join("keep.txt"), "safe").expect("write");

    #[cfg(unix)]
    std::os::unix::fs::symlink(
        outside.path().join("keep.txt"),
        workspace.path().join("link.txt"),
    )
    .expect("symlink");

    let ctx = ServerTestContext::new(config_with_workspace(&workspace));
    let resp = delete_json(
        ctx.addr(),
        "/v1/agent/fs/rm",
        &serde_json::json!({ "path": "link.txt" }),
    );
    assert_eq!(resp.status, 403, "symlink remove escape must be blocked: {}", resp.body);

    // Verify the outside file still exists
    assert!(outside.path().join("keep.txt").exists(), "outside file must survive");
}

#[test]
fn agent_fs_stat_symlink_outside_workspace_blocked() {
    let workspace = TempDir::new().expect("workspace");
    let outside = TempDir::new().expect("outside");
    std::fs::write(outside.path().join("probe.txt"), "data").expect("write");

    #[cfg(unix)]
    std::os::unix::fs::symlink(
        outside.path().join("probe.txt"),
        workspace.path().join("link.txt"),
    )
    .expect("symlink");

    let ctx = ServerTestContext::new(config_with_workspace(&workspace));
    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/stat",
        &serde_json::json!({ "path": "link.txt" }),
    );
    assert_eq!(resp.status, 403, "stat on symlink escape must be blocked: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "permission_denied");
}

#[test]
fn agent_fs_write_through_symlink_dir_outside_workspace_blocked() {
    let workspace = TempDir::new().expect("workspace");
    let outside = TempDir::new().expect("outside");
    std::fs::create_dir(outside.path().join("target_dir")).expect("mkdir");

    // Symlink a directory
    #[cfg(unix)]
    std::os::unix::fs::symlink(
        outside.path().join("target_dir"),
        workspace.path().join("link_dir"),
    )
    .expect("symlink");

    let ctx = ServerTestContext::new(config_with_workspace(&workspace));
    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/write",
        &serde_json::json!({
            "path": "link_dir/pwned.txt",
            "content": "escaped"
        }),
    );
    assert_eq!(resp.status, 403, "write through symlinked dir must be blocked: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "permission_denied");

    // Verify nothing was written outside
    assert!(
        !outside.path().join("target_dir/pwned.txt").exists(),
        "file must not appear outside workspace"
    );
}

// ===========================================================================
// Sandbox: stat, list, mkdir outside workspace
// ===========================================================================

#[test]
fn agent_fs_stat_outside_workspace_blocked() {
    let workspace = TempDir::new().expect("workspace");
    let outside = TempDir::new().expect("outside");
    std::fs::write(outside.path().join("probe.txt"), "x").expect("write");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/stat",
        &serde_json::json!({
            "path": outside.path().join("probe.txt").to_string_lossy().to_string()
        }),
    );
    assert_eq!(resp.status, 403, "stat outside workspace must be blocked: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "permission_denied");
}

#[test]
fn agent_fs_list_outside_workspace_blocked() {
    let workspace = TempDir::new().expect("workspace");
    let outside = TempDir::new().expect("outside");
    std::fs::write(outside.path().join("secret.txt"), "x").expect("write");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/ls",
        &serde_json::json!({
            "path": outside.path().to_string_lossy().to_string()
        }),
    );
    assert_eq!(resp.status, 403, "list outside workspace must be blocked: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "permission_denied");
}

#[test]
fn agent_fs_mkdir_outside_workspace_blocked() {
    let workspace = TempDir::new().expect("workspace");
    let outside = TempDir::new().expect("outside");
    let target = outside.path().join("should_not_exist");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/mkdir",
        &serde_json::json!({
            "path": target.to_string_lossy().to_string(),
            "recursive": true
        }),
    );
    assert_eq!(resp.status, 403, "body: {}", resp.body);
    assert!(!target.exists(), "directory must not be created outside workspace");
}

// ===========================================================================
// Edge cases
// ===========================================================================

#[test]
fn agent_fs_write_empty_content_creates_zero_byte_file() {
    let workspace = TempDir::new().expect("workspace");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/write",
        &serde_json::json!({ "path": "empty.txt", "content": "" }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(resp.json()["bytes_written"], 0);

    let stat = post_json(
        ctx.addr(),
        "/v1/agent/fs/stat",
        &serde_json::json!({ "path": "empty.txt" }),
    );
    assert_eq!(stat.json()["exists"], true);
    assert_eq!(stat.json()["size"], 0);
}

#[test]
fn agent_fs_edit_new_text_empty_deletes_matched_text() {
    let (ctx, _ws) = setup_with_file("strip.txt", "keep DELETE_ME keep");

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/edit",
        &serde_json::json!({
            "path": "strip.txt",
            "old_text": "DELETE_ME ",
            "new_text": ""
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(resp.json()["replacements"], 1);

    let read = post_json(
        ctx.addr(),
        "/v1/agent/fs/read",
        &serde_json::json!({ "path": "strip.txt" }),
    );
    assert_eq!(read.json()["content"], "keep keep");
}

#[test]
fn agent_fs_stat_workspace_root() {
    let workspace = TempDir::new().expect("workspace");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/stat",
        &serde_json::json!({ "path": "." }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let j = resp.json();
    assert_eq!(j["exists"], true);
    assert_eq!(j["is_dir"], true);
    assert_eq!(j["is_file"], false);
}

#[test]
fn agent_fs_list_file_path_returns_error() {
    let (ctx, _ws) = setup_with_file("just_a_file.txt", "content");

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/ls",
        &serde_json::json!({ "path": "just_a_file.txt" }),
    );
    assert_ne!(resp.status, 200, "listing a file should fail: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "not_directory");
}

#[test]
fn agent_fs_mkdir_already_exists_returns_error() {
    let workspace = TempDir::new().expect("workspace");
    std::fs::create_dir(workspace.path().join("existing")).expect("mkdir");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/mkdir",
        &serde_json::json!({ "path": "existing" }),
    );
    // mkdir is not idempotent — creating an already-existing dir is an error
    assert_eq!(resp.status, 500, "mkdir on existing dir should fail: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "io_error");
    // The directory should still exist unchanged
    assert!(workspace.path().join("existing").is_dir());
}

#[test]
fn agent_fs_unicode_filename() {
    let workspace = TempDir::new().expect("workspace");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/write",
        &serde_json::json!({
            "path": "日本語/données.txt",
            "content": "café ñ 中文",
            "mkdir": true
        }),
    );
    assert_eq!(resp.status, 200, "unicode write failed: {}", resp.body);

    let read = post_json(
        ctx.addr(),
        "/v1/agent/fs/read",
        &serde_json::json!({ "path": "日本語/données.txt" }),
    );
    assert_eq!(read.status, 200, "unicode read failed: {}", read.body);
    assert_eq!(read.json()["content"], "café ñ 中文");
}

// ===========================================================================
// Cross-encoding
// ===========================================================================

#[test]
fn agent_fs_write_utf8_read_base64() {
    let (ctx, _ws) = setup_with_file("text.txt", "hello");

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/read",
        &serde_json::json!({ "path": "text.txt", "encoding": "base64" }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(resp.json()["encoding"], "base64");
    // "hello" in base64 = "aGVsbG8="
    assert_eq!(resp.json()["content"], "aGVsbG8=");
}

#[test]
fn agent_fs_write_base64_read_utf8() {
    let workspace = TempDir::new().expect("workspace");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    // Write "world" as base64 (d29ybGQ=)
    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/write",
        &serde_json::json!({
            "path": "encoded.txt",
            "content": "d29ybGQ=",
            "encoding": "base64"
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let read = post_json(
        ctx.addr(),
        "/v1/agent/fs/read",
        &serde_json::json!({ "path": "encoded.txt" }),
    );
    assert_eq!(read.status, 200, "body: {}", read.body);
    assert_eq!(read.json()["encoding"], "utf-8");
    assert_eq!(read.json()["content"], "world");
}

// ===========================================================================
// Gap tests: additional coverage
// ===========================================================================

/// Write to a symlink file (not through a symlinked directory) pointing outside.
#[test]
fn agent_fs_write_to_symlink_file_outside_workspace_blocked() {
    let workspace = TempDir::new().expect("workspace");
    let outside = TempDir::new().expect("outside");
    std::fs::write(outside.path().join("target.txt"), "original").expect("write");

    #[cfg(unix)]
    std::os::unix::fs::symlink(
        outside.path().join("target.txt"),
        workspace.path().join("link.txt"),
    )
    .expect("symlink");

    let ctx = ServerTestContext::new(config_with_workspace(&workspace));
    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/write",
        &serde_json::json!({
            "path": "link.txt",
            "content": "overwritten"
        }),
    );
    assert_eq!(resp.status, 403, "write to symlink file escape must be blocked: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "permission_denied");

    // Verify the outside file was not modified
    let content = std::fs::read_to_string(outside.path().join("target.txt")).expect("read");
    assert_eq!(content, "original", "outside file must not be modified");
}

/// Rename when the destination file already exists — should overwrite on POSIX.
#[test]
fn agent_fs_rename_overwrites_existing_destination() {
    let workspace = TempDir::new().expect("workspace");
    std::fs::write(workspace.path().join("src.txt"), "new content").expect("write");
    std::fs::write(workspace.path().join("dst.txt"), "old content").expect("write");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/rename",
        &serde_json::json!({
            "from": "src.txt",
            "to": "dst.txt"
        }),
    );
    assert_eq!(resp.status, 200, "rename to existing should succeed: {}", resp.body);

    // Source is gone
    let stat_src = post_json(
        ctx.addr(),
        "/v1/agent/fs/stat",
        &serde_json::json!({ "path": "src.txt" }),
    );
    assert_eq!(stat_src.json()["exists"], false);

    // Destination has the new content
    let read = post_json(
        ctx.addr(),
        "/v1/agent/fs/read",
        &serde_json::json!({ "path": "dst.txt" }),
    );
    assert_eq!(read.json()["content"], "new content");
}

/// Remove an empty directory without recursive flag — should succeed.
#[test]
fn agent_fs_remove_empty_dir_without_recursive_succeeds() {
    let workspace = TempDir::new().expect("workspace");
    std::fs::create_dir(workspace.path().join("empty_dir")).expect("mkdir");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let resp = delete_json(
        ctx.addr(),
        "/v1/agent/fs/rm",
        &serde_json::json!({
            "path": "empty_dir",
            "recursive": false
        }),
    );
    assert_eq!(resp.status, 200, "removing empty dir should work: {}", resp.body);
    assert_eq!(resp.json()["removed"], true);

    // Verify it's gone
    assert!(!workspace.path().join("empty_dir").exists());
}

/// List recursive entries include relative path with directory prefix.
#[test]
fn agent_fs_list_recursive_entry_paths_include_relative_dir() {
    let workspace = TempDir::new().expect("workspace");
    std::fs::create_dir_all(workspace.path().join("a/b")).expect("mkdirs");
    std::fs::write(workspace.path().join("a/b/deep.txt"), "d").expect("write");
    let ctx = ServerTestContext::new(config_with_workspace(&workspace));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/fs/ls",
        &serde_json::json!({
            "path": "a",
            "glob": "*.txt",
            "recursive": true
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let j = resp.json();
    let entries = j["entries"].as_array().expect("entries");
    assert_eq!(entries.len(), 1, "should find deep.txt");

    // The `path` field should include the relative directory from the listed root
    let entry_path = entries[0]["path"].as_str().expect("entry path");
    assert!(
        entry_path.contains("b/deep.txt") || entry_path.contains("b\\deep.txt"),
        "entry path should include relative dir, got: {entry_path}"
    );
}
