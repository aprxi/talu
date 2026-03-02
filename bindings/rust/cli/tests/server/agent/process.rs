use crate::server::common::{
    delete, get, post_json, send_request, ServerConfig, ServerTestContext, TenantSpec,
};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

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

fn gateway_config_with_policy(policy_json: Option<&str>) -> ServerConfig {
    let mut cfg = ServerConfig::new();
    cfg.gateway_secret = Some("secret".to_string());
    cfg.tenants = vec![TenantSpec {
        id: "tenant-a".to_string(),
        storage_prefix: "tenant-a".to_string(),
        allowed_models: vec![],
    }];
    if let Some(policy) = policy_json {
        cfg.env_vars
            .push(("TALU_AGENT_POLICY_JSON".to_string(), policy.to_string()));
    }
    cfg
}

fn spawn_process(ctx: &ServerTestContext, command: &str) -> String {
    let resp = post_json(
        ctx.addr(),
        "/v1/agent/processes/spawn",
        &serde_json::json!({ "command": command }),
    );
    assert_eq!(resp.status, 200, "spawn failed: {}", resp.body);
    resp.json()["process_id"]
        .as_str()
        .expect("process_id")
        .to_string()
}

fn spawn_process_as_user(ctx: &ServerTestContext, command: &str, user_id: &str) -> String {
    let body = serde_json::json!({ "command": command }).to_string();
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/agent/processes/spawn",
        &[
            ("Content-Type", "application/json"),
            ("X-Talu-Gateway-Secret", "secret"),
            ("X-Talu-Tenant-Id", "tenant-a"),
            ("X-Talu-User-Id", user_id),
        ],
        Some(&body),
    );
    assert_eq!(resp.status, 200, "spawn failed: {}", resp.body);
    resp.json()["process_id"]
        .as_str()
        .expect("process_id")
        .to_string()
}

fn open_stream_connection(ctx: &ServerTestContext, path: &str) -> (TcpStream, String) {
    let addr = ctx.addr();
    let mut stream =
        TcpStream::connect_timeout(&addr.into(), Duration::from_secs(5)).expect("connect");
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .expect("set read timeout");
    stream
        .set_write_timeout(Some(Duration::from_secs(5)))
        .expect("set write timeout");

    let request = format!(
        "GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: keep-alive\r\nContent-Length: 0\r\n\r\n",
        host = addr
    );
    stream.write_all(request.as_bytes()).expect("write request");
    stream.flush().expect("flush");

    let mut raw = Vec::new();
    loop {
        let mut buf = [0u8; 4096];
        let n = stream.read(&mut buf).expect("read response");
        assert!(n > 0, "stream closed before headers");
        raw.extend_from_slice(&buf[..n]);
        if let Some(pos) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&raw[..pos]).to_string();
            return (stream, head);
        }
    }
}

#[test]
fn agent_process_spawn_list_delete_roundtrip() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let process_id = spawn_process(&ctx, "echo ready");

    let list = get(ctx.addr(), "/v1/agent/processes");
    assert_eq!(list.status, 200, "list body: {}", list.body);
    let list_json = list.json();
    let data = list_json["data"].as_array().expect("data array");
    assert!(
        data.iter()
            .any(|entry| entry["process_id"].as_str() == Some(process_id.as_str())),
        "spawned process should be listed"
    );

    let del = delete(ctx.addr(), &format!("/v1/agent/processes/{process_id}"));
    assert_eq!(del.status, 200, "delete body: {}", del.body);
    assert_eq!(del.json()["terminated"], true);

    let missing = delete(ctx.addr(), &format!("/v1/agent/processes/{process_id}"));
    assert_eq!(missing.status, 404, "body: {}", missing.body);
}

#[test]
fn agent_process_send_then_stream_output_and_exit() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let process_id = spawn_process(
        &ctx,
        "while IFS= read -r line; do echo \"$line\"; [ \"$line\" = \"quit\" ] && break; done",
    );

    let send_hello = post_json(
        ctx.addr(),
        &format!("/v1/agent/processes/{process_id}/send"),
        &serde_json::json!({ "data": "hello\n" }),
    );
    assert_eq!(send_hello.status, 200, "body: {}", send_hello.body);

    let send_quit = post_json(
        ctx.addr(),
        &format!("/v1/agent/processes/{process_id}/send"),
        &serde_json::json!({ "data": "quit\n" }),
    );
    assert_eq!(send_quit.status, 200, "body: {}", send_quit.body);

    let stream = get(
        ctx.addr(),
        &format!("/v1/agent/processes/{process_id}/stream"),
    );
    assert_eq!(stream.status, 200, "stream body: {}", stream.body);
    let content_type = stream.header("content-type").unwrap_or("");
    assert!(
        content_type.contains("text/event-stream"),
        "content-type={content_type}"
    );

    let events = parse_sse_events(&stream.body);
    let output_events: Vec<_> = events.iter().filter(|e| e["type"] == "output").collect();
    let exit_event = events.iter().find(|e| e["type"] == "exit");

    assert!(
        !output_events.is_empty(),
        "expected at least one output event: {events:?}"
    );
    let all_output: String = output_events
        .iter()
        .filter_map(|e| e["data"].as_str())
        .collect();
    assert!(
        all_output.contains("hello"),
        "output should contain hello, got: {all_output}"
    );
    assert!(exit_event.is_some(), "expected exit event: {events:?}");
}

#[test]
fn agent_process_spawn_requires_command() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/processes/spawn",
        &serde_json::json!({ "command": "   " }),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_request");
}

#[test]
fn agent_process_spawn_denied_by_policy() {
    let policy = r#"{
        "default":"deny",
        "statements":[
            {"effect":"allow","action":"tool.fs.read","resource":"**"}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/processes/spawn",
        &serde_json::json!({ "command": "echo hi" }),
    );
    assert_eq!(resp.status, 403, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "policy_denied_exec");
}

#[test]
fn agent_process_spawn_with_invalid_policy_json_returns_500() {
    let mut cfg = ServerConfig::new();
    cfg.env_vars.push((
        "TALU_AGENT_POLICY_JSON".to_string(),
        "{not-valid-json".to_string(),
    ));
    let ctx = ServerTestContext::new(cfg);

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/processes/spawn",
        &serde_json::json!({ "command": "echo hi" }),
    );
    assert_eq!(resp.status, 500, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "policy_invalid");
}

#[test]
fn agent_process_spawn_with_invalid_policy_schema_returns_500() {
    let policy = r#"{
        "default":"maybe",
        "statements":[]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/processes/spawn",
        &serde_json::json!({ "command": "echo hi" }),
    );
    assert_eq!(resp.status, 500, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "policy_invalid");
}

#[test]
fn agent_process_spawn_with_invalid_policy_schema_missing_statements_returns_500() {
    let policy = r#"{
        "default":"deny"
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/processes/spawn",
        &serde_json::json!({ "command": "echo hi" }),
    );
    assert_eq!(resp.status, 500, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "policy_invalid");
}

#[test]
fn agent_process_spawn_allowed_by_policy_command_filter() {
    let policy = r#"{
        "default":"deny",
        "statements":[
            {"effect":"allow","action":"tool.process","command":"echo *"}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let process_id = spawn_process(&ctx, "echo hello");
    let del = delete(ctx.addr(), &format!("/v1/agent/processes/{process_id}"));
    assert_eq!(del.status, 200, "body: {}", del.body);
}

#[test]
fn agent_process_spawn_denied_by_policy_command_filter() {
    let policy = r#"{
        "default":"deny",
        "statements":[
            {"effect":"allow","action":"tool.process","command":"echo *"}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/processes/spawn",
        &serde_json::json!({ "command": "ls -la" }),
    );
    assert_eq!(resp.status, 403, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "policy_denied_exec");
}

#[test]
fn agent_process_spawn_denied_by_policy_cwd_filter() {
    let policy = r#"{
        "default":"deny",
        "statements":[
            {"effect":"allow","action":"tool.process","command":"echo *"},
            {"effect":"deny","action":"tool.process","command":"echo *","cwd":"."}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/processes/spawn",
        &serde_json::json!({ "command": "echo hi", "cwd": "." }),
    );
    assert_eq!(resp.status, 403, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "policy_denied_cwd");
}

#[test]
fn agent_process_spawn_denied_when_chain_contains_disallowed_segment() {
    let policy = r#"{
        "default":"deny",
        "statements":[
            {"effect":"allow","action":"tool.process","command":"echo *"}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/processes/spawn",
        &serde_json::json!({ "command": "echo safe && ls" }),
    );
    assert_eq!(resp.status, 403, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "policy_denied_exec");
}

#[test]
fn agent_process_spawn_denied_when_or_contains_disallowed_segment() {
    let policy = r#"{
        "default":"deny",
        "statements":[
            {"effect":"allow","action":"tool.process","command":"echo *"}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/processes/spawn",
        &serde_json::json!({ "command": "echo hello || ls" }),
    );
    assert_eq!(resp.status, 403, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "policy_denied_exec");
}

#[test]
fn agent_process_spawn_denied_when_pipe_contains_disallowed_segment() {
    let policy = r#"{
        "default":"deny",
        "statements":[
            {"effect":"allow","action":"tool.process","command":"echo *"}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/processes/spawn",
        &serde_json::json!({ "command": "echo hello | grep hello" }),
    );
    assert_eq!(resp.status, 403, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "policy_denied_exec");
}

#[test]
fn agent_process_spawn_denied_when_semicolon_contains_disallowed_segment() {
    let policy = r#"{
        "default":"deny",
        "statements":[
            {"effect":"allow","action":"tool.process","command":"echo *"}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/processes/spawn",
        &serde_json::json!({ "command": "echo hello ; grep hello /dev/null" }),
    );
    assert_eq!(resp.status, 403, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "policy_denied_exec");
}

#[test]
fn agent_process_spawn_default_allow_with_explicit_deny_blocks_matching_command() {
    let policy = r#"{
        "default":"allow",
        "statements":[
            {"effect":"deny","action":"tool.process","command":"echo secret*"}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/processes/spawn",
        &serde_json::json!({ "command": "echo secret-value" }),
    );
    assert_eq!(resp.status, 403, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "policy_denied_exec");
}

#[test]
fn agent_process_spawn_allow_command_pattern_matches_bare_command() {
    let policy = r#"{
        "default":"deny",
        "statements":[
            {"effect":"allow","action":"tool.process","command":"echo *"}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let process_id = spawn_process(&ctx, "echo");
    let del = delete(ctx.addr(), &format!("/v1/agent/processes/{process_id}"));
    assert_eq!(del.status, 200, "body: {}", del.body);
}

#[test]
fn agent_process_spawn_cwd_deny_does_not_apply_when_cwd_not_provided() {
    let policy = r#"{
        "default":"deny",
        "statements":[
            {"effect":"allow","action":"tool.process","command":"echo *"},
            {"effect":"deny","action":"tool.process","command":"echo *","cwd":"."}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let process_id = spawn_process(&ctx, "echo hello");
    let del = delete(ctx.addr(), &format!("/v1/agent/processes/{process_id}"));
    assert_eq!(del.status, 200, "body: {}", del.body);
}

#[test]
fn agent_process_spawn_policy_deny_wins_even_with_later_allow() {
    let policy = r#"{
        "default":"deny",
        "statements":[
            {"effect":"deny","action":"tool.process","command":"echo secret*"},
            {"effect":"allow","action":"tool.process","command":"echo *"}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/processes/spawn",
        &serde_json::json!({ "command": "echo secret-value" }),
    );
    assert_eq!(resp.status, 403, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "policy_denied_exec");
}

#[test]
fn agent_process_spawn_policy_deny_wins_regardless_of_statement_order() {
    let policies = [
        r#"{
            "default":"allow",
            "statements":[
                {"effect":"allow","action":"tool.process","command":"echo *"},
                {"effect":"deny","action":"tool.process","command":"echo secret*"}
            ]
        }"#,
        r#"{
            "default":"allow",
            "statements":[
                {"effect":"deny","action":"tool.process","command":"echo secret*"},
                {"effect":"allow","action":"tool.process","command":"echo *"}
            ]
        }"#,
    ];

    for policy in policies {
        let ctx = ServerTestContext::new(config_with_policy(policy));
        let resp = post_json(
            ctx.addr(),
            "/v1/agent/processes/spawn",
            &serde_json::json!({ "command": "echo secret-value" }),
        );
        assert_eq!(resp.status, 403, "body: {}", resp.body);
        assert_eq!(resp.json()["error"]["code"], "policy_denied_exec");
    }
}

#[test]
fn agent_process_spawn_policy_tool_exec_does_not_grant_tool_process() {
    let policy = r#"{
        "default":"deny",
        "statements":[
            {"effect":"allow","action":"tool.exec","command":"echo *"}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/processes/spawn",
        &serde_json::json!({ "command": "echo hello" }),
    );
    assert_eq!(resp.status, 403, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "policy_denied_exec");
}

#[test]
fn agent_process_spawn_policy_tool_shell_does_not_grant_tool_process() {
    let policy = r#"{
        "default":"deny",
        "statements":[
            {"effect":"allow","action":"tool.shell"}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/processes/spawn",
        &serde_json::json!({ "command": "echo hello" }),
    );
    assert_eq!(resp.status, 403, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "policy_denied_exec");
}

#[test]
fn agent_process_spawn_policy_allows_absolute_executable_via_normalization() {
    let policy = r#"{
        "default":"deny",
        "statements":[
            {"effect":"allow","action":"tool.process","command":"echo *"}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let process_id = spawn_process(&ctx, "/bin/echo hello");
    let del = delete(ctx.addr(), &format!("/v1/agent/processes/{process_id}"));
    assert_eq!(del.status, 200, "body: {}", del.body);
}

#[test]
fn agent_process_spawn_policy_denies_absolute_executable_when_pattern_mismatches() {
    let policy = r#"{
        "default":"deny",
        "statements":[
            {"effect":"allow","action":"tool.process","command":"printf *"}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/processes/spawn",
        &serde_json::json!({ "command": "/bin/echo hello" }),
    );
    assert_eq!(resp.status, 403, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "policy_denied_exec");
}

#[test]
fn agent_process_spawn_policy_exact_command_without_wildcard_does_not_allow_arguments() {
    let policy = r#"{
        "default":"deny",
        "statements":[
            {"effect":"allow","action":"tool.process","command":"echo"}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/processes/spawn",
        &serde_json::json!({ "command": "echo hello" }),
    );
    assert_eq!(resp.status, 403, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "policy_denied_exec");
}

#[test]
fn agent_process_spawn_policy_exact_command_without_wildcard_allows_bare_command() {
    let policy = r#"{
        "default":"deny",
        "statements":[
            {"effect":"allow","action":"tool.process","command":"echo"}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let process_id = spawn_process(&ctx, "echo");
    let del = delete(ctx.addr(), &format!("/v1/agent/processes/{process_id}"));
    assert_eq!(del.status, 200, "body: {}", del.body);
}

#[test]
fn agent_process_spawn_policy_cwd_scope_allows_tmp_and_denies_non_matching_cwd() {
    let policy = r#"{
        "default":"deny",
        "statements":[
            {"effect":"allow","action":"tool.process","command":"echo *","cwd":"/tmp"}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let allowed = post_json(
        ctx.addr(),
        "/v1/agent/processes/spawn",
        &serde_json::json!({ "command": "echo ok", "cwd": "/tmp" }),
    );
    assert_eq!(allowed.status, 200, "body: {}", allowed.body);
    let process_id = allowed.json()["process_id"]
        .as_str()
        .expect("process_id")
        .to_string();
    let del = delete(ctx.addr(), &format!("/v1/agent/processes/{process_id}"));
    assert_eq!(del.status, 200, "body: {}", del.body);

    let denied = post_json(
        ctx.addr(),
        "/v1/agent/processes/spawn",
        &serde_json::json!({ "command": "echo nope", "cwd": "." }),
    );
    assert_eq!(denied.status, 403, "body: {}", denied.body);
    assert_eq!(denied.json()["error"]["code"], "policy_denied_exec");
}

#[test]
fn agent_process_spawn_policy_explicit_cwd_deny_reports_cwd_error() {
    let policy = r#"{
        "default":"deny",
        "statements":[
            {"effect":"allow","action":"tool.process","command":"echo *"},
            {"effect":"deny","action":"tool.process","command":"echo *","cwd":"/tmp"}
        ]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let denied = post_json(
        ctx.addr(),
        "/v1/agent/processes/spawn",
        &serde_json::json!({ "command": "echo nope", "cwd": "/tmp" }),
    );
    assert_eq!(denied.status, 403, "body: {}", denied.body);
    assert_eq!(denied.json()["error"]["code"], "policy_denied_cwd");
}

#[test]
fn agent_process_send_nonexistent_returns_404() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/processes/00000000-0000-0000-0000-000000000000/send",
        &serde_json::json!({ "data": "hello\n" }),
    );
    assert_eq!(resp.status, 404, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "not_found");
}

#[test]
fn agent_process_stream_nonexistent_returns_404() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let resp = get(
        ctx.addr(),
        "/v1/agent/processes/00000000-0000-0000-0000-000000000000/stream",
    );
    assert_eq!(resp.status, 404, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "not_found");
}

#[test]
fn agent_process_send_invalid_path_returns_400() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/processes/has/slash/send",
        &serde_json::json!({ "data": "hello\n" }),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_request");
}

#[test]
fn agent_process_stream_invalid_path_returns_400() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let resp = get(ctx.addr(), "/v1/agent/processes/has/slash/stream");
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_request");
}

#[test]
fn agent_process_send_malformed_json_returns_400() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let process_id = spawn_process(&ctx, "cat");

    let resp = crate::server::common::send_request(
        ctx.addr(),
        "POST",
        &format!("/v1/agent/processes/{process_id}/send"),
        &[("Content-Type", "application/json")],
        Some("{not valid json"),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_json");

    let _ = delete(ctx.addr(), &format!("/v1/agent/processes/{process_id}"));
}

#[test]
fn agent_process_spawn_disallowed_command_returns_403_command_denied() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/processes/spawn",
        &serde_json::json!({ "command": "rm -rf /" }),
    );
    assert_eq!(resp.status, 403, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "command_denied");
}

#[test]
fn agent_process_spawn_policy_allow_all_still_enforces_baseline_safety() {
    let policy = r#"{
        "default":"allow",
        "statements":[]
    }"#;
    let ctx = ServerTestContext::new(config_with_policy(policy));

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/processes/spawn",
        &serde_json::json!({ "command": "rm -rf /" }),
    );
    assert_eq!(resp.status, 403, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "command_denied");
}

#[test]
fn agent_process_delete_invalid_path_returns_400() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let resp = delete(ctx.addr(), "/v1/agent/processes/has/slash");
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_request");
}

#[test]
fn agent_process_send_missing_data_returns_400() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let process_id = spawn_process(&ctx, "cat");

    let resp = post_json(
        ctx.addr(),
        &format!("/v1/agent/processes/{process_id}/send"),
        &serde_json::json!({}),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_json");

    let _ = delete(ctx.addr(), &format!("/v1/agent/processes/{process_id}"));
}

#[test]
fn agent_process_send_after_delete_returns_404() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let process_id = spawn_process(&ctx, "cat");

    let del = delete(ctx.addr(), &format!("/v1/agent/processes/{process_id}"));
    assert_eq!(del.status, 200, "body: {}", del.body);

    let resp = post_json(
        ctx.addr(),
        &format!("/v1/agent/processes/{process_id}/send"),
        &serde_json::json!({ "data": "hello\n" }),
    );
    assert_eq!(resp.status, 404, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "not_found");
}

#[test]
fn agent_process_spawn_enforces_per_owner_limit() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let mut process_ids = Vec::new();
    for _ in 0..10 {
        process_ids.push(spawn_process(&ctx, "cat"));
    }

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/processes/spawn",
        &serde_json::json!({ "command": "echo overflow" }),
    );
    assert_eq!(resp.status, 429, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "resource_exhausted");

    for process_id in process_ids {
        let del = delete(ctx.addr(), &format!("/v1/agent/processes/{process_id}"));
        assert_eq!(del.status, 200, "delete body: {}", del.body);
    }
}

#[test]
fn agent_process_owner_isolation_with_gateway_auth() {
    let ctx = ServerTestContext::new(gateway_config_with_policy(None));
    let process_id = spawn_process_as_user(&ctx, "cat", "user-a");

    let list_user_a = send_request(
        ctx.addr(),
        "GET",
        "/v1/agent/processes",
        &[
            ("X-Talu-Gateway-Secret", "secret"),
            ("X-Talu-Tenant-Id", "tenant-a"),
            ("X-Talu-User-Id", "user-a"),
        ],
        None,
    );
    assert_eq!(list_user_a.status, 200, "body: {}", list_user_a.body);
    let list_user_a_json = list_user_a.json();
    let data_a = list_user_a_json["data"].as_array().expect("data array");
    assert!(
        data_a
            .iter()
            .any(|entry| entry["process_id"].as_str() == Some(process_id.as_str())),
        "owner should see spawned process"
    );

    let list_user_b = send_request(
        ctx.addr(),
        "GET",
        "/v1/agent/processes",
        &[
            ("X-Talu-Gateway-Secret", "secret"),
            ("X-Talu-Tenant-Id", "tenant-a"),
            ("X-Talu-User-Id", "user-b"),
        ],
        None,
    );
    assert_eq!(list_user_b.status, 200, "body: {}", list_user_b.body);
    let list_user_b_json = list_user_b.json();
    let data_b = list_user_b_json["data"].as_array().expect("data array");
    assert!(
        data_b
            .iter()
            .all(|entry| entry["process_id"].as_str() != Some(process_id.as_str())),
        "non-owner should not see owner process"
    );

    let send_user_b = send_request(
        ctx.addr(),
        "POST",
        &format!("/v1/agent/processes/{process_id}/send"),
        &[
            ("Content-Type", "application/json"),
            ("X-Talu-Gateway-Secret", "secret"),
            ("X-Talu-Tenant-Id", "tenant-a"),
            ("X-Talu-User-Id", "user-b"),
        ],
        Some(r#"{"data":"hello\n"}"#),
    );
    assert_eq!(send_user_b.status, 403, "body: {}", send_user_b.body);
    assert_eq!(send_user_b.json()["error"]["code"], "forbidden");

    let stream_user_b = send_request(
        ctx.addr(),
        "GET",
        &format!("/v1/agent/processes/{process_id}/stream"),
        &[
            ("X-Talu-Gateway-Secret", "secret"),
            ("X-Talu-Tenant-Id", "tenant-a"),
            ("X-Talu-User-Id", "user-b"),
        ],
        None,
    );
    assert_eq!(stream_user_b.status, 403, "body: {}", stream_user_b.body);
    assert_eq!(stream_user_b.json()["error"]["code"], "forbidden");

    let delete_user_b = send_request(
        ctx.addr(),
        "DELETE",
        &format!("/v1/agent/processes/{process_id}"),
        &[
            ("X-Talu-Gateway-Secret", "secret"),
            ("X-Talu-Tenant-Id", "tenant-a"),
            ("X-Talu-User-Id", "user-b"),
        ],
        None,
    );
    assert_eq!(delete_user_b.status, 403, "body: {}", delete_user_b.body);
    assert_eq!(delete_user_b.json()["error"]["code"], "forbidden");

    let delete_user_a = send_request(
        ctx.addr(),
        "DELETE",
        &format!("/v1/agent/processes/{process_id}"),
        &[
            ("X-Talu-Gateway-Secret", "secret"),
            ("X-Talu-Tenant-Id", "tenant-a"),
            ("X-Talu-User-Id", "user-a"),
        ],
        None,
    );
    assert_eq!(delete_user_a.status, 200, "body: {}", delete_user_a.body);
}

#[test]
fn agent_process_spawn_limit_is_scoped_per_owner_under_gateway_auth() {
    let ctx = ServerTestContext::new(gateway_config_with_policy(None));
    let mut owner_a_ids = Vec::new();
    for _ in 0..10 {
        owner_a_ids.push(spawn_process_as_user(&ctx, "cat", "user-a"));
    }

    let overflow_user_a = send_request(
        ctx.addr(),
        "POST",
        "/v1/agent/processes/spawn",
        &[
            ("Content-Type", "application/json"),
            ("X-Talu-Gateway-Secret", "secret"),
            ("X-Talu-Tenant-Id", "tenant-a"),
            ("X-Talu-User-Id", "user-a"),
        ],
        Some(r#"{"command":"echo overflow"}"#),
    );
    assert_eq!(
        overflow_user_a.status, 429,
        "body: {}",
        overflow_user_a.body
    );
    assert_eq!(
        overflow_user_a.json()["error"]["code"],
        "resource_exhausted"
    );

    let user_b_first = send_request(
        ctx.addr(),
        "POST",
        "/v1/agent/processes/spawn",
        &[
            ("Content-Type", "application/json"),
            ("X-Talu-Gateway-Secret", "secret"),
            ("X-Talu-Tenant-Id", "tenant-a"),
            ("X-Talu-User-Id", "user-b"),
        ],
        Some(r#"{"command":"echo ok"}"#),
    );
    assert_eq!(user_b_first.status, 200, "body: {}", user_b_first.body);
    let user_b_id = user_b_first.json()["process_id"]
        .as_str()
        .expect("process_id")
        .to_string();

    for process_id in owner_a_ids {
        let del = send_request(
            ctx.addr(),
            "DELETE",
            &format!("/v1/agent/processes/{process_id}"),
            &[
                ("X-Talu-Gateway-Secret", "secret"),
                ("X-Talu-Tenant-Id", "tenant-a"),
                ("X-Talu-User-Id", "user-a"),
            ],
            None,
        );
        assert_eq!(del.status, 200, "delete body: {}", del.body);
    }
    let del_b = send_request(
        ctx.addr(),
        "DELETE",
        &format!("/v1/agent/processes/{user_b_id}"),
        &[
            ("X-Talu-Gateway-Secret", "secret"),
            ("X-Talu-Tenant-Id", "tenant-a"),
            ("X-Talu-User-Id", "user-b"),
        ],
        None,
    );
    assert_eq!(del_b.status, 200, "delete body: {}", del_b.body);
}

#[test]
fn agent_process_stream_rejects_second_attached_stream() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let process_id = spawn_process(&ctx, "cat");
    let path = format!("/v1/agent/processes/{process_id}/stream");

    let (stream_conn, headers) = open_stream_connection(&ctx, &path);
    assert!(
        headers
            .lines()
            .next()
            .is_some_and(|line| line.contains("200")),
        "expected 200 stream response, got headers: {headers}"
    );

    let second = get(ctx.addr(), &path);
    assert_eq!(second.status, 409, "body: {}", second.body);
    assert_eq!(second.json()["error"]["code"], "already_streaming");

    drop(stream_conn);
    let del = delete(ctx.addr(), &format!("/v1/agent/processes/{process_id}"));
    assert_eq!(del.status, 200, "delete body: {}", del.body);
}
