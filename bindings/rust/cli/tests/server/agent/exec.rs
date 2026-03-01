use crate::server::common::{post_json, ServerConfig, ServerTestContext};

#[test]
fn agent_exec_streams_stdout_and_exit() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/exec",
        &serde_json::json!({
            "command": "echo hello",
            "timeout_ms": 30_000
        }),
    );

    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let content_type = resp.header("content-type").unwrap_or("");
    assert!(
        content_type.contains("text/event-stream"),
        "content-type={content_type}"
    );
    assert!(
        resp.body.contains("\"type\":\"stdout\""),
        "missing stdout event: {}",
        resp.body
    );
    assert!(
        resp.body.contains("\"type\":\"exit\""),
        "missing exit event: {}",
        resp.body
    );
}

#[test]
fn agent_exec_denies_disallowed_command() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/exec",
        &serde_json::json!({
            "command": "rm -rf /"
        }),
    );

    assert_eq!(resp.status, 403, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "command_denied");
}
