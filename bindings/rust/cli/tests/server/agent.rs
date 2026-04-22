use std::path::PathBuf;

use tempfile::TempDir;

use crate::server::common::*;

fn workdir_config(workdir: PathBuf) -> ServerConfig {
    let mut config = ServerConfig::new();
    config.workdir = Some(workdir);
    config
}

#[test]
fn agent_docs_and_openapi_are_served() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let docs = get(ctx.addr(), "/docs");
    assert_eq!(docs.status, 200, "body: {}", docs.body);
    for link in [
        "/docs/agent/fs",
        "/docs/agent/exec",
        "/docs/agent/shell",
        "/docs/agent/process",
    ] {
        assert!(docs.body.contains(link), "missing docs link: {link}");
    }

    for (path, expected) in [
        ("/openapi/agent/fs.json", "/v1/agent/fs/read"),
        ("/openapi/agent/exec.json", "/v1/agent/exec"),
        ("/openapi/agent/shell.json", "/v1/agent/shells"),
        ("/openapi/agent/process.json", "/v1/agent/processes/spawn"),
    ] {
        let resp = get(ctx.addr(), path);
        assert_eq!(resp.status, 200, "path={path} body={}", resp.body);
        assert!(
            resp.body.contains(expected),
            "path={path} missing endpoint {expected}"
        );
    }
}

#[test]
fn agent_fs_roundtrip_works() {
    let workdir = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(workdir_config(workdir.path().to_path_buf()));

    let write = post_json(
        ctx.addr(),
        "/v1/agent/fs/write",
        &serde_json::json!({
            "path": "notes.txt",
            "content": "hello world"
        }),
    );
    assert_eq!(write.status, 200, "body: {}", write.body);

    let read = post_json(
        ctx.addr(),
        "/v1/agent/fs/read",
        &serde_json::json!({
            "path": "notes.txt"
        }),
    );
    assert_eq!(read.status, 200, "body: {}", read.body);
    let read_json = read.json();
    assert_eq!(read_json["content"], "hello world");

    let ls = post_json(
        ctx.addr(),
        "/v1/agent/fs/ls",
        &serde_json::json!({
            "path": "."
        }),
    );
    assert_eq!(ls.status, 200, "body: {}", ls.body);
    assert!(ls.body.contains("notes.txt"));

    let rename = post_json(
        ctx.addr(),
        "/v1/agent/fs/rename",
        &serde_json::json!({
            "from": "notes.txt",
            "to": "renamed.txt"
        }),
    );
    assert_eq!(rename.status, 200, "body: {}", rename.body);

    let rm = delete_json(
        ctx.addr(),
        "/v1/agent/fs/rm",
        &serde_json::json!({
            "path": "renamed.txt"
        }),
    );
    assert_eq!(rm.status, 200, "body: {}", rm.body);
}

#[test]
fn agent_fs_roundtrip_works_with_collab_disabled() {
    let workdir = TempDir::new().expect("temp dir");
    let mut config = workdir_config(workdir.path().to_path_buf());
    config
        .env_vars
        .push(("TALU_DISABLE_COLLAB".to_string(), "true".to_string()));
    let ctx = ServerTestContext::new(config);

    let write = post_json(
        ctx.addr(),
        "/v1/agent/fs/write",
        &serde_json::json!({
            "path": "notes.txt",
            "content": "hello world"
        }),
    );
    assert_eq!(write.status, 200, "body: {}", write.body);

    let read = post_json(
        ctx.addr(),
        "/v1/agent/fs/read",
        &serde_json::json!({
            "path": "notes.txt"
        }),
    );
    assert_eq!(read.status, 200, "body: {}", read.body);
    let read_json = read.json();
    assert_eq!(read_json["content"], "hello world");
    assert_eq!(read_json["path"], "notes.txt");
}

#[test]
fn agent_exec_streams_output_and_exit() {
    let workdir = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(workdir_config(workdir.path().to_path_buf()));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/exec",
        &serde_json::json!({
            "command": "echo hello"
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(
        resp.header("content-type"),
        Some("text/event-stream; charset=utf-8")
    );
    assert!(
        resp.body.contains(r#""type":"stdout""#),
        "body: {}",
        resp.body
    );
    assert!(resp.body.contains("hello"), "body: {}", resp.body);
    assert!(
        resp.body.contains(r#""type":"exit""#),
        "body: {}",
        resp.body
    );
}

#[test]
fn agent_shell_lifecycle_works() {
    let workdir = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(workdir_config(workdir.path().to_path_buf()));

    let create = post_json(
        ctx.addr(),
        "/v1/agent/shells",
        &serde_json::json!({
            "cols": 80,
            "rows": 24
        }),
    );
    assert_eq!(create.status, 200, "body: {}", create.body);
    let shell_id = create.json()["shell_id"]
        .as_str()
        .expect("shell id")
        .to_string();

    let list = get(ctx.addr(), "/v1/agent/shells");
    assert_eq!(list.status, 200, "body: {}", list.body);
    assert!(list.body.contains(&shell_id), "body: {}", list.body);

    let get_one = get(ctx.addr(), &format!("/v1/agent/shells/{shell_id}"));
    assert_eq!(get_one.status, 200, "body: {}", get_one.body);

    let delete = delete(ctx.addr(), &format!("/v1/agent/shells/{shell_id}"));
    assert_eq!(delete.status, 200, "body: {}", delete.body);
}

#[test]
fn agent_process_spawn_stream_and_delete() {
    let workdir = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(workdir_config(workdir.path().to_path_buf()));

    let spawn = post_json(
        ctx.addr(),
        "/v1/agent/processes/spawn",
        &serde_json::json!({
            "command": "echo ping"
        }),
    );
    assert_eq!(spawn.status, 200, "body: {}", spawn.body);
    let process_id = spawn.json()["process_id"]
        .as_str()
        .expect("process id")
        .to_string();

    let list = get(ctx.addr(), "/v1/agent/processes");
    assert_eq!(list.status, 200, "body: {}", list.body);
    assert!(list.body.contains(&process_id), "body: {}", list.body);

    let stream = get(
        ctx.addr(),
        &format!("/v1/agent/processes/{process_id}/stream"),
    );
    assert_eq!(stream.status, 200, "body: {}", stream.body);
    assert!(stream.body.contains("ping"), "body: {}", stream.body);
    assert!(
        stream.body.contains(r#""type":"exit""#),
        "body: {}",
        stream.body
    );

    let delete = delete(ctx.addr(), &format!("/v1/agent/processes/{process_id}"));
    assert_eq!(delete.status, 200, "body: {}", delete.body);
}
