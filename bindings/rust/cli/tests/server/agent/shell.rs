use crate::server::common::{
    delete, get, post_json, send_request, ServerConfig, ServerTestContext,
};

#[test]
fn agent_shell_create_list_get_delete_roundtrip() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let create = post_json(
        ctx.addr(),
        "/v1/agent/shells",
        &serde_json::json!({
            "cols": 100,
            "rows": 30
        }),
    );
    assert_eq!(create.status, 200, "body: {}", create.body);
    let shell_id = create.json()["shell_id"]
        .as_str()
        .expect("shell_id in create response")
        .to_string();

    let list = get(ctx.addr(), "/v1/agent/shells");
    assert_eq!(list.status, 200, "body: {}", list.body);
    let list_json = list.json();
    let list_data = list_json["data"].as_array().expect("data array");
    assert!(
        list_data
            .iter()
            .any(|entry| entry["shell_id"].as_str() == Some(shell_id.as_str())),
        "shell id should be listed"
    );

    let get_resp = get(ctx.addr(), &format!("/v1/agent/shells/{shell_id}"));
    assert_eq!(get_resp.status, 200, "body: {}", get_resp.body);
    assert_eq!(get_resp.json()["shell_id"], shell_id);

    let delete_resp = delete(ctx.addr(), &format!("/v1/agent/shells/{shell_id}"));
    assert_eq!(delete_resp.status, 200, "body: {}", delete_resp.body);
    assert_eq!(delete_resp.json()["terminated"], true);

    let missing = get(ctx.addr(), &format!("/v1/agent/shells/{shell_id}"));
    assert_eq!(missing.status, 404, "body: {}", missing.body);
}

#[test]
fn agent_shell_ws_upgrade_returns_switching_protocols() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let create = post_json(ctx.addr(), "/v1/agent/shells", &serde_json::json!({}));
    assert_eq!(create.status, 200, "body: {}", create.body);
    let shell_id = create.json()["shell_id"]
        .as_str()
        .expect("shell_id in create response")
        .to_string();

    let ws_path = format!("/v1/agent/shells/{shell_id}/ws");
    let resp = send_request(
        ctx.addr(),
        "GET",
        &ws_path,
        &[
            ("Connection", "Upgrade"),
            ("Upgrade", "websocket"),
            ("Sec-WebSocket-Version", "13"),
            ("Sec-WebSocket-Key", "dGhlIHNhbXBsZSBub25jZQ=="),
        ],
        None,
    );

    assert_eq!(
        resp.status, 101,
        "headers: {} body: {}",
        resp.headers, resp.body
    );
}
