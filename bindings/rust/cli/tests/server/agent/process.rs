use crate::server::common::{delete, get, post_json, ServerConfig, ServerTestContext};

fn parse_sse_events(body: &str) -> Vec<serde_json::Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter_map(|data| serde_json::from_str(data).ok())
        .collect()
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
