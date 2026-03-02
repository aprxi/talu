use crate::server::common::{post_json, send_request, ServerConfig, ServerTestContext};

/// Parse SSE `data: {...}\n\n` lines into JSON values.
fn parse_sse_events(body: &str) -> Vec<serde_json::Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter_map(|data| serde_json::from_str(data).ok())
        .collect()
}

// ---------------------------------------------------------------------------
// Input validation
// ---------------------------------------------------------------------------

#[test]
fn agent_exec_malformed_json_returns_400() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/agent/exec",
        &[("content-type", "application/json")],
        Some("{not valid json"),
    );

    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_json");
}

#[test]
fn agent_exec_missing_command_returns_400() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let resp = post_json(ctx.addr(), "/v1/agent/exec", &serde_json::json!({}));

    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_json");
}

#[test]
fn agent_exec_empty_command_returns_400() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/exec",
        &serde_json::json!({ "command": "" }),
    );

    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_request");
}

#[test]
fn agent_exec_whitespace_command_returns_400() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/exec",
        &serde_json::json!({ "command": "   " }),
    );

    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_request");
}

// ---------------------------------------------------------------------------
// Command policy
// ---------------------------------------------------------------------------

#[test]
fn agent_exec_denies_disallowed_command() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/exec",
        &serde_json::json!({ "command": "rm -rf /" }),
    );

    assert_eq!(resp.status, 403, "body: {}", resp.body);
    let json = resp.json();
    assert_eq!(json["error"]["code"], "command_denied");
    assert!(
        json["error"]["message"]
            .as_str()
            .is_some_and(|m| !m.is_empty()),
        "expected non-empty denial message: {json}"
    );
}

#[test]
fn agent_exec_denies_chained_dangerous_command() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    // `echo` is whitelisted, but `rm` is not — the chain should be denied.
    let resp = post_json(
        ctx.addr(),
        "/v1/agent/exec",
        &serde_json::json!({ "command": "echo safe && rm -rf /" }),
    );

    assert_eq!(resp.status, 403, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "command_denied");
}


// ---------------------------------------------------------------------------
// SSE streaming — happy path
// ---------------------------------------------------------------------------

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

    let events = parse_sse_events(&resp.body);
    let stdout_events: Vec<_> = events.iter().filter(|e| e["type"] == "stdout").collect();
    let exit_events: Vec<_> = events.iter().filter(|e| e["type"] == "exit").collect();

    assert!(!stdout_events.is_empty(), "no stdout events: {events:?}");
    assert_eq!(
        exit_events.len(),
        1,
        "expected exactly one exit event: {events:?}"
    );

    let all_stdout: String = stdout_events
        .iter()
        .filter_map(|e| e["data"].as_str())
        .collect();
    assert!(
        all_stdout.contains("hello"),
        "stdout missing 'hello': {all_stdout}"
    );

    assert_eq!(exit_events[0]["code"], 0, "exit: {}", exit_events[0]);
}

#[test]
fn agent_exec_exit_code_nonzero() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    // `grep` is whitelisted; a nonexistent flag produces exit code 2.
    let resp = post_json(
        ctx.addr(),
        "/v1/agent/exec",
        &serde_json::json!({
            "command": "grep --this-flag-does-not-exist /dev/null",
            "timeout_ms": 30_000
        }),
    );

    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let events = parse_sse_events(&resp.body);
    let exit_event = events
        .iter()
        .find(|e| e["type"] == "exit")
        .expect("no exit event");
    let code = exit_event["code"].as_i64().expect("code not an integer");
    assert_ne!(code, 0, "expected nonzero exit: {exit_event}");
}

#[test]
fn agent_exec_stderr_event() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    // `grep` with bad flag writes to stderr.
    let resp = post_json(
        ctx.addr(),
        "/v1/agent/exec",
        &serde_json::json!({
            "command": "grep --this-flag-does-not-exist /dev/null",
            "timeout_ms": 30_000
        }),
    );

    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let events = parse_sse_events(&resp.body);
    let stderr_events: Vec<_> = events.iter().filter(|e| e["type"] == "stderr").collect();
    assert!(!stderr_events.is_empty(), "no stderr events: {events:?}");
}

#[test]
fn agent_exec_with_custom_cwd() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    // `readlink -f .` prints the resolved working directory.
    let resp = post_json(
        ctx.addr(),
        "/v1/agent/exec",
        &serde_json::json!({
            "command": "readlink -f .",
            "cwd": "/tmp",
            "timeout_ms": 30_000
        }),
    );

    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let events = parse_sse_events(&resp.body);
    let all_stdout: String = events
        .iter()
        .filter(|e| e["type"] == "stdout")
        .filter_map(|e| e["data"].as_str())
        .collect();
    assert!(
        all_stdout.contains("/tmp"),
        "expected cwd /tmp in stdout: {all_stdout}"
    );
}

#[test]
fn agent_exec_invalid_cwd_streams_error_event() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/exec",
        &serde_json::json!({
            "command": "echo hello",
            "cwd": "/nonexistent/path/xyz",
            "timeout_ms": 30_000
        }),
    );

    // Handler returns 200 and streams an error event — exec failure is in-band.
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let events = parse_sse_events(&resp.body);
    let has_error = events.iter().any(|e| e["type"] == "error");
    assert!(has_error, "expected SSE error event for bad cwd: {events:?}");

    // Should NOT have a normal exit event.
    let has_exit = events.iter().any(|e| e["type"] == "exit");
    assert!(
        !has_exit,
        "should not have exit event on spawn failure: {events:?}"
    );
}
