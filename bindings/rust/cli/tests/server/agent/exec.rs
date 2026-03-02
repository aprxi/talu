use crate::server::common::{
    assert_server_startup_fails, post_json, send_request, ServerConfig, ServerTestContext,
};

/// Parse SSE `data: {...}\n\n` lines into JSON values.
fn parse_sse_events(body: &str) -> Vec<serde_json::Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter_map(|data| serde_json::from_str(data).ok())
        .collect()
}

fn config_with_policy(policy_json: &str) -> ServerConfig {
    let mut cfg = ServerConfig::new();
    cfg.env_vars.push((
        "TALU_AGENT_POLICY_JSON".to_string(),
        policy_json.to_string(),
    ));
    cfg
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

#[test]
fn agent_exec_policy_denied_surfaces_error_event() {
    let policy = r#"{
        "default":"deny",
        "statements":[
            {"effect":"allow","action":"tool.fs.read","resource":"**"}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/exec",
        &serde_json::json!({ "command": "echo hello" }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let events = parse_sse_events(&resp.body);
    assert!(
        events.iter().any(|e| e["type"] == "error"),
        "expected error event: {events:?}"
    );
}

#[test]
fn agent_exec_invalid_policy_json_fails_startup() {
    let mut cfg = ServerConfig::new();
    cfg.env_vars.push((
        "TALU_AGENT_POLICY_JSON".to_string(),
        "{not-valid-json".to_string(),
    ));
    assert_server_startup_fails(cfg, "parse agent runtime policy JSON");
}

#[test]
fn agent_exec_invalid_policy_schema_fails_startup() {
    let policy = r#"{
        "default":"maybe",
        "statements":[]
    }"#;
    assert_server_startup_fails(
        config_with_policy(policy),
        "parse agent runtime policy JSON",
    );
}

#[test]
fn agent_exec_invalid_policy_schema_missing_statements_fails_startup() {
    let policy = r#"{
        "default":"deny"
    }"#;
    assert_server_startup_fails(
        config_with_policy(policy),
        "parse agent runtime policy JSON",
    );
}

#[test]
fn agent_exec_policy_cwd_denied_surfaces_error_event() {
    let policy = r#"{
        "default":"deny",
        "statements":[
            {"effect":"allow","action":"tool.exec","command":"echo *"},
            {"effect":"deny","action":"tool.exec","command":"echo *","cwd":"."}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/exec",
        &serde_json::json!({ "command": "echo hello", "cwd": "." }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let events = parse_sse_events(&resp.body);
    let error_event = events
        .iter()
        .find(|e| e["type"] == "error")
        .expect("expected error event");
    let message = error_event["message"].as_str().unwrap_or("");
    assert!(
        message.contains("policy denied cwd"),
        "unexpected error event: {error_event}"
    );
}

#[test]
fn agent_exec_policy_deny_wins_even_with_later_allow() {
    let policy = r#"{
        "default":"deny",
        "statements":[
            {"effect":"deny","action":"tool.exec","command":"echo secret*"},
            {"effect":"allow","action":"tool.exec","command":"echo *"}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/exec",
        &serde_json::json!({ "command": "echo secret-value" }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let events = parse_sse_events(&resp.body);
    let error_event = events
        .iter()
        .find(|e| e["type"] == "error")
        .expect("expected error event");
    let message = error_event["message"].as_str().unwrap_or("");
    assert!(
        message.contains("policy denied exec"),
        "unexpected error event: {error_event}"
    );
}

#[test]
fn agent_exec_policy_deny_wins_regardless_of_statement_order() {
    let policies = [
        r#"{
            "default":"allow",
            "statements":[
                {"effect":"allow","action":"tool.exec","command":"echo *"},
                {"effect":"deny","action":"tool.exec","command":"echo secret*"}
            ]
        }"#,
        r#"{
            "default":"allow",
            "statements":[
                {"effect":"deny","action":"tool.exec","command":"echo secret*"},
                {"effect":"allow","action":"tool.exec","command":"echo *"}
            ]
        }"#,
    ];

    for policy in policies {
        let ctx = ServerTestContext::new(config_with_policy(policy));
        let resp = post_json(
            ctx.addr(),
            "/v1/agent/exec",
            &serde_json::json!({ "command": "echo secret-value" }),
        );
        assert_eq!(resp.status, 200, "body: {}", resp.body);
        let events = parse_sse_events(&resp.body);
        let error_event = events
            .iter()
            .find(|e| e["type"] == "error")
            .expect("expected error event");
        let message = error_event["message"].as_str().unwrap_or("");
        assert!(
            message.contains("policy denied exec"),
            "unexpected error event: {error_event}"
        );
    }
}

#[test]
fn agent_exec_policy_tool_process_does_not_grant_tool_exec() {
    let policy = r#"{
        "default":"deny",
        "statements":[
            {"effect":"allow","action":"tool.process","command":"echo *"}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/exec",
        &serde_json::json!({ "command": "echo hello" }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let events = parse_sse_events(&resp.body);
    let error_event = events
        .iter()
        .find(|e| e["type"] == "error")
        .expect("expected error event");
    let message = error_event["message"].as_str().unwrap_or("");
    assert!(
        message.contains("policy denied exec"),
        "unexpected error event: {error_event}"
    );
}

#[test]
fn agent_exec_policy_tool_shell_does_not_grant_tool_exec() {
    let policy = r#"{
        "default":"deny",
        "statements":[
            {"effect":"allow","action":"tool.shell"}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/exec",
        &serde_json::json!({ "command": "echo hello" }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let events = parse_sse_events(&resp.body);
    let error_event = events
        .iter()
        .find(|e| e["type"] == "error")
        .expect("expected error event");
    let message = error_event["message"].as_str().unwrap_or("");
    assert!(
        message.contains("policy denied exec"),
        "unexpected error event: {error_event}"
    );
}

#[test]
fn agent_exec_policy_allows_absolute_executable_via_normalization() {
    let policy = r#"{
        "default":"deny",
        "statements":[
            {"effect":"allow","action":"tool.exec","command":"echo *"}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/exec",
        &serde_json::json!({ "command": "/bin/echo hello" }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let events = parse_sse_events(&resp.body);
    assert!(
        !events.iter().any(|e| e["type"] == "error"),
        "did not expect policy error: {events:?}"
    );
    assert!(
        events.iter().any(|e| e["type"] == "exit"),
        "expected exit event: {events:?}"
    );
}

#[test]
fn agent_exec_policy_denies_absolute_executable_when_pattern_mismatches() {
    let policy = r#"{
        "default":"deny",
        "statements":[
            {"effect":"allow","action":"tool.exec","command":"printf *"}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/exec",
        &serde_json::json!({ "command": "/bin/echo hello" }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let events = parse_sse_events(&resp.body);
    let error_event = events
        .iter()
        .find(|e| e["type"] == "error")
        .expect("expected error event");
    let message = error_event["message"].as_str().unwrap_or("");
    assert!(
        message.contains("policy denied exec"),
        "unexpected error event: {error_event}"
    );
}

#[test]
fn agent_exec_policy_exact_command_without_wildcard_does_not_allow_arguments() {
    let policy = r#"{
        "default":"deny",
        "statements":[
            {"effect":"allow","action":"tool.exec","command":"echo"}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/exec",
        &serde_json::json!({ "command": "echo hello" }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let events = parse_sse_events(&resp.body);
    let error_event = events
        .iter()
        .find(|e| e["type"] == "error")
        .expect("expected error event");
    let message = error_event["message"].as_str().unwrap_or("");
    assert!(
        message.contains("policy denied exec"),
        "unexpected error event: {error_event}"
    );
}

#[test]
fn agent_exec_policy_exact_command_without_wildcard_allows_bare_command() {
    let policy = r#"{
        "default":"deny",
        "statements":[
            {"effect":"allow","action":"tool.exec","command":"echo"}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/exec",
        &serde_json::json!({ "command": "echo" }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let events = parse_sse_events(&resp.body);
    assert!(
        !events.iter().any(|e| e["type"] == "error"),
        "did not expect policy error: {events:?}"
    );
    assert!(
        events.iter().any(|e| e["type"] == "exit"),
        "expected exit event: {events:?}"
    );
}

#[test]
fn agent_exec_policy_cwd_scope_allows_tmp_and_denies_non_matching_cwd() {
    let policy = r#"{
        "default":"deny",
        "statements":[
            {"effect":"allow","action":"tool.exec","command":"echo *","cwd":"/tmp"},
            {"effect":"deny","action":"tool.exec","command":"echo *","cwd":"."}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let allowed = post_json(
        ctx.addr(),
        "/v1/agent/exec",
        &serde_json::json!({ "command": "echo ok", "cwd": "/tmp" }),
    );
    assert_eq!(allowed.status, 200, "body: {}", allowed.body);
    let allowed_events = parse_sse_events(&allowed.body);
    assert!(
        !allowed_events.iter().any(|e| e["type"] == "error"),
        "did not expect policy error in /tmp: {allowed_events:?}"
    );
    assert!(
        allowed_events.iter().any(|e| e["type"] == "exit"),
        "expected exit event: {allowed_events:?}"
    );

    let denied = post_json(
        ctx.addr(),
        "/v1/agent/exec",
        &serde_json::json!({ "command": "echo nope", "cwd": "." }),
    );
    assert_eq!(denied.status, 200, "body: {}", denied.body);
    let denied_events = parse_sse_events(&denied.body);
    let error_event = denied_events
        .iter()
        .find(|e| e["type"] == "error")
        .expect("expected policy error in workspace cwd");
    let message = error_event["message"].as_str().unwrap_or("");
    assert!(
        message.contains("policy denied exec"),
        "unexpected error event: {error_event}"
    );
}

#[test]
fn agent_exec_policy_denies_andand_segment_not_explicitly_allowed() {
    let policy = r#"{
        "default":"deny",
        "statements":[
            {"effect":"allow","action":"tool.exec","command":"echo *"}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/exec",
        &serde_json::json!({ "command": "echo hello && ls" }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let events = parse_sse_events(&resp.body);
    let error_event = events
        .iter()
        .find(|e| e["type"] == "error")
        .expect("expected error event");
    let message = error_event["message"].as_str().unwrap_or("");
    assert!(
        message.contains("policy denied exec"),
        "unexpected error event: {error_event}"
    );
}

#[test]
fn agent_exec_policy_denies_or_segment_not_explicitly_allowed() {
    let policy = r#"{
        "default":"deny",
        "statements":[
            {"effect":"allow","action":"tool.exec","command":"echo *"}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/exec",
        &serde_json::json!({ "command": "echo hello || ls" }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let events = parse_sse_events(&resp.body);
    let error_event = events
        .iter()
        .find(|e| e["type"] == "error")
        .expect("expected error event");
    let message = error_event["message"].as_str().unwrap_or("");
    assert!(
        message.contains("policy denied exec"),
        "unexpected error event: {error_event}"
    );
}

#[test]
fn agent_exec_policy_denies_pipe_segment_not_explicitly_allowed() {
    let policy = r#"{
        "default":"deny",
        "statements":[
            {"effect":"allow","action":"tool.exec","command":"echo *"}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/exec",
        &serde_json::json!({ "command": "echo hello | grep hello" }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let events = parse_sse_events(&resp.body);
    let error_event = events
        .iter()
        .find(|e| e["type"] == "error")
        .expect("expected error event");
    let message = error_event["message"].as_str().unwrap_or("");
    assert!(
        message.contains("policy denied exec"),
        "unexpected error event: {error_event}"
    );
}

#[test]
fn agent_exec_policy_denies_semicolon_segment_not_explicitly_allowed() {
    let policy = r#"{
        "default":"deny",
        "statements":[
            {"effect":"allow","action":"tool.exec","command":"echo *"}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/exec",
        &serde_json::json!({ "command": "echo hello ; grep hello /dev/null" }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let events = parse_sse_events(&resp.body);
    let error_event = events
        .iter()
        .find(|e| e["type"] == "error")
        .expect("expected error event");
    let message = error_event["message"].as_str().unwrap_or("");
    assert!(
        message.contains("policy denied exec"),
        "unexpected error event: {error_event}"
    );
}

#[test]
fn agent_exec_policy_explicit_cwd_deny_reports_cwd_reason() {
    let policy = r#"{
        "default":"deny",
        "statements":[
            {"effect":"allow","action":"tool.exec","command":"echo *"},
            {"effect":"deny","action":"tool.exec","command":"echo *","cwd":"/tmp"}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/exec",
        &serde_json::json!({ "command": "echo hello", "cwd": "/tmp" }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let events = parse_sse_events(&resp.body);
    let error_event = events
        .iter()
        .find(|e| e["type"] == "error")
        .expect("expected error event");
    let message = error_event["message"].as_str().unwrap_or("");
    assert!(
        message.contains("policy denied cwd"),
        "unexpected error event: {error_event}"
    );
}

#[test]
fn agent_exec_policy_default_allow_with_explicit_deny_blocks_matching_command() {
    let policy = r#"{
        "default":"allow",
        "statements":[
            {"effect":"deny","action":"tool.exec","command":"echo secret*"}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/exec",
        &serde_json::json!({ "command": "echo secret-value" }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let events = parse_sse_events(&resp.body);
    let error_event = events
        .iter()
        .find(|e| e["type"] == "error")
        .expect("expected error event");
    let message = error_event["message"].as_str().unwrap_or("");
    assert!(
        message.contains("policy denied exec"),
        "unexpected error event: {error_event}"
    );
}

#[test]
fn agent_exec_policy_allow_command_pattern_matches_bare_command() {
    let policy = r#"{
        "default":"deny",
        "statements":[
            {"effect":"allow","action":"tool.exec","command":"echo *"}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/exec",
        &serde_json::json!({ "command": "echo" }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let events = parse_sse_events(&resp.body);
    assert!(
        !events.iter().any(|e| e["type"] == "error"),
        "did not expect policy error: {events:?}"
    );
    assert!(
        events.iter().any(|e| e["type"] == "exit"),
        "expected exit event: {events:?}"
    );
}

#[test]
fn agent_exec_policy_cwd_deny_does_not_apply_when_cwd_not_provided() {
    let policy = r#"{
        "default":"deny",
        "statements":[
            {"effect":"allow","action":"tool.exec","command":"echo *"},
            {"effect":"deny","action":"tool.exec","command":"echo *","cwd":"."}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/exec",
        &serde_json::json!({ "command": "echo hello" }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let events = parse_sse_events(&resp.body);
    assert!(
        !events.iter().any(|e| e["type"] == "error"),
        "did not expect policy error when cwd is omitted: {events:?}"
    );
    assert!(
        events.iter().any(|e| e["type"] == "exit"),
        "expected exit event: {events:?}"
    );
}

#[test]
fn agent_exec_allow_all_policy_still_enforces_baseline_safety() {
    let policy = r#"{
        "default":"allow",
        "statements":[]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/exec",
        &serde_json::json!({ "command": "rm -rf /" }),
    );

    assert_eq!(resp.status, 403, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "command_denied");
}

#[test]
fn agent_exec_policy_max_timeout_clamps_request_timeout() {
    let policy = r#"{
        "default":"allow",
        "max_timeout_ms": 20,
        "statements":[]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/exec",
        &serde_json::json!({
            "command": "tail -f /dev/null",
            "timeout_ms": 30_000
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let events = parse_sse_events(&resp.body);
    let has_error = events.iter().any(|e| e["type"] == "error");
    let has_done_stdout = events
        .iter()
        .filter(|e| e["type"] == "stdout")
        .filter_map(|e| e["data"].as_str())
        .any(|s| s.contains("done"));
    let has_exit = events.iter().any(|e| e["type"] == "exit");

    assert!(has_error, "expected timeout error event: {events:?}");
    assert!(
        !has_done_stdout,
        "command should not complete when timeout is clamped: {events:?}"
    );
    assert!(
        !has_exit,
        "timed-out execution should not emit normal exit: {events:?}"
    );
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
    assert!(
        has_error,
        "expected SSE error event for bad cwd: {events:?}"
    );

    // Should NOT have a normal exit event.
    let has_exit = events.iter().any(|e| e["type"] == "exit");
    assert!(
        !has_exit,
        "should not have exit event on spawn failure: {events:?}"
    );
}
