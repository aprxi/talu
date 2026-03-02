use crate::server::common::{
    delete, get, post_json, send_request, ServerConfig, ServerTestContext,
};
use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message;

type WsStream =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// Helper: create a shell and return its id.
fn create_shell(ctx: &ServerTestContext) -> String {
    let resp = post_json(ctx.addr(), "/v1/agent/shells", &serde_json::json!({}));
    assert_eq!(resp.status, 200, "create failed: {}", resp.body);
    resp.json()["shell_id"]
        .as_str()
        .expect("shell_id")
        .to_string()
}

/// Open a real WebSocket connection to an existing shell session.
async fn ws_connect(ctx: &ServerTestContext, shell_id: &str) -> WsStream {
    let url = format!(
        "ws://127.0.0.1:{}/v1/agent/shells/{}/ws",
        ctx.addr().port(),
        shell_id
    );
    let (ws, _) = tokio_tungstenite::connect_async(&url)
        .await
        .unwrap_or_else(|e| panic!("WS connect to {url} failed: {e}"));
    ws
}

/// Read WS frames until we get a text message (JSON), skipping binary frames.
/// Returns the parsed JSON. Panics after 5000 iterations (deadlock guard).
async fn ws_next_text(ws: &mut WsStream) -> serde_json::Value {
    for _ in 0..5000 {
        match ws.next().await {
            Some(Ok(Message::Text(text))) => {
                return serde_json::from_str(&text)
                    .unwrap_or_else(|e| panic!("invalid JSON in WS text: {e}\nraw: {text}"));
            }
            Some(Ok(Message::Binary(_))) => continue,
            Some(Ok(Message::Ping(_))) => continue,
            Some(Ok(Message::Pong(_))) => continue,
            Some(Ok(Message::Close(_))) => panic!("WS closed before text message received"),
            Some(Err(e)) => panic!("WS receive error: {e}"),
            None => panic!("WS stream ended before text message received"),
            _ => continue,
        }
    }
    panic!("ws_next_text: guard limit reached (5000 frames without text)");
}

/// Drain all binary output from the WS until the shell exits (text exit event)
/// or we hit the guard limit. Returns (collected_output, exit_event_if_any).
async fn ws_drain_output(ws: &mut WsStream) -> (Vec<u8>, Option<serde_json::Value>) {
    let mut output = Vec::new();
    for _ in 0..50_000 {
        match tokio::time::timeout(std::time::Duration::from_secs(10), ws.next()).await {
            Ok(Some(Ok(Message::Binary(data)))) => output.extend_from_slice(&data),
            Ok(Some(Ok(Message::Text(text)))) => {
                let json: serde_json::Value = serde_json::from_str(&text).unwrap_or_default();
                if json["type"] == "exit" || json["type"] == "error" {
                    return (output, Some(json));
                }
            }
            Ok(Some(Ok(_))) => continue,
            Ok(Some(Err(e))) => panic!("WS receive error during drain: {e}"),
            Ok(None) => return (output, None),
            Err(_) => panic!("ws_drain_output: timeout (10s without data)"),
        }
    }
    panic!("ws_drain_output: guard limit reached (50000 frames)");
}

// ---------------------------------------------------------------------------
// CREATE (POST /v1/agent/shells)
// ---------------------------------------------------------------------------

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
fn agent_shell_create_with_empty_body_uses_defaults() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let resp = post_json(ctx.addr(), "/v1/agent/shells", &serde_json::json!({}));
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(json["cols"], 120, "default cols");
    assert_eq!(json["rows"], 40, "default rows");
}

#[test]
fn agent_shell_create_response_shape() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/shells",
        &serde_json::json!({"cols": 80, "rows": 24}),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert!(json["shell_id"].is_string(), "shell_id should be string");
    assert!(json["cols"].is_number(), "cols should be number");
    assert!(json["rows"].is_number(), "rows should be number");
    assert!(
        json.get("attached_clients").is_some(),
        "attached_clients should be present"
    );
    assert_eq!(json["attached_clients"], 0);
}

#[test]
fn agent_shell_create_with_custom_cols_rows() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/shells",
        &serde_json::json!({"cols": 200, "rows": 50}),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(json["cols"], 200);
    assert_eq!(json["rows"], 50);
}

#[test]
fn agent_shell_create_with_cols_zero_returns_400() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/shells",
        &serde_json::json!({"cols": 0, "rows": 24}),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_request");
}

#[test]
fn agent_shell_create_with_rows_zero_returns_400() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/shells",
        &serde_json::json!({"cols": 80, "rows": 0}),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_request");
}

#[test]
fn agent_shell_create_with_malformed_json_returns_400() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/agent/shells",
        &[("Content-Type", "application/json")],
        Some("{not valid json"),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_json");
}

#[test]
fn agent_shell_create_with_cwd_returns_it_in_response() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/shells",
        &serde_json::json!({"cwd": "/tmp"}),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(resp.json()["cwd"], "/tmp");
}

// ---------------------------------------------------------------------------
// LIST (GET /v1/agent/shells)
// ---------------------------------------------------------------------------

#[test]
fn agent_shell_list_empty_returns_empty_array() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let resp = get(ctx.addr(), "/v1/agent/shells");
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert!(data.is_empty(), "expected empty list, got: {data:?}");
}

#[test]
fn agent_shell_list_multiple_shells() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let id1 = create_shell(&ctx);
    let id2 = create_shell(&ctx);
    let id3 = create_shell(&ctx);

    let resp = get(ctx.addr(), "/v1/agent/shells");
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 3, "expected 3 shells, got: {}", data.len());

    let ids: Vec<&str> = data
        .iter()
        .filter_map(|e| e["shell_id"].as_str())
        .collect();
    assert!(ids.contains(&id1.as_str()), "missing id1");
    assert!(ids.contains(&id2.as_str()), "missing id2");
    assert!(ids.contains(&id3.as_str()), "missing id3");
}

// ---------------------------------------------------------------------------
// GET (GET /v1/agent/shells/{id})
// ---------------------------------------------------------------------------

#[test]
fn agent_shell_get_nonexistent_returns_404() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let resp = get(
        ctx.addr(),
        "/v1/agent/shells/00000000-0000-0000-0000-000000000000",
    );
    assert_eq!(resp.status, 404, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "not_found");
}

#[test]
fn agent_shell_get_response_shape() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let create = post_json(
        ctx.addr(),
        "/v1/agent/shells",
        &serde_json::json!({"cols": 132, "rows": 43}),
    );
    assert_eq!(create.status, 200, "body: {}", create.body);
    let shell_id = create.json()["shell_id"]
        .as_str()
        .expect("shell_id")
        .to_string();

    let resp = get(ctx.addr(), &format!("/v1/agent/shells/{shell_id}"));
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(json["shell_id"], shell_id);
    assert_eq!(json["cols"], 132);
    assert_eq!(json["rows"], 43);
    assert_eq!(json["attached_clients"], 0);
}

#[test]
fn agent_shell_get_invalid_path_returns_400() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let resp = get(ctx.addr(), "/v1/agent/shells/has/slash");
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_request");
}

// ---------------------------------------------------------------------------
// DELETE (DELETE /v1/agent/shells/{id})
// ---------------------------------------------------------------------------

#[test]
fn agent_shell_delete_nonexistent_returns_404() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let resp = delete(
        ctx.addr(),
        "/v1/agent/shells/00000000-0000-0000-0000-000000000000",
    );
    assert_eq!(resp.status, 404, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "not_found");
}

#[test]
fn agent_shell_double_delete_returns_404() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let shell_id = create_shell(&ctx);

    let first = delete(ctx.addr(), &format!("/v1/agent/shells/{shell_id}"));
    assert_eq!(first.status, 200, "body: {}", first.body);

    let second = delete(ctx.addr(), &format!("/v1/agent/shells/{shell_id}"));
    assert_eq!(second.status, 404, "body: {}", second.body);
    assert_eq!(second.json()["error"]["code"], "not_found");
}

#[test]
fn agent_shell_delete_removes_from_list() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let id1 = create_shell(&ctx);
    let id2 = create_shell(&ctx);

    let del = delete(ctx.addr(), &format!("/v1/agent/shells/{id1}"));
    assert_eq!(del.status, 200, "body: {}", del.body);

    let list = get(ctx.addr(), "/v1/agent/shells");
    assert_eq!(list.status, 200, "body: {}", list.body);

    let list_json = list.json();
    let data = list_json["data"].as_array().expect("data array");
    let ids: Vec<&str> = data
        .iter()
        .filter_map(|e| e["shell_id"].as_str())
        .collect();
    assert!(!ids.contains(&id1.as_str()), "deleted shell still listed");
    assert!(ids.contains(&id2.as_str()), "surviving shell missing");
}

// ---------------------------------------------------------------------------
// WS UPGRADE EDGE CASES
// ---------------------------------------------------------------------------

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

#[test]
fn agent_shell_ws_upgrade_nonexistent_returns_404() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let resp = send_request(
        ctx.addr(),
        "GET",
        "/v1/agent/shells/00000000-0000-0000-0000-000000000000/ws",
        &[
            ("Connection", "Upgrade"),
            ("Upgrade", "websocket"),
            ("Sec-WebSocket-Version", "13"),
            ("Sec-WebSocket-Key", "dGhlIHNhbXBsZSBub25jZQ=="),
        ],
        None,
    );
    assert_eq!(resp.status, 404, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "not_found");
}

#[test]
fn agent_shell_ws_upgrade_missing_key_returns_400() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let shell_id = create_shell(&ctx);

    let resp = send_request(
        ctx.addr(),
        "GET",
        &format!("/v1/agent/shells/{shell_id}/ws"),
        &[
            ("Connection", "Upgrade"),
            ("Upgrade", "websocket"),
            ("Sec-WebSocket-Version", "13"),
        ],
        None,
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_request");
}

#[test]
fn agent_shell_ws_upgrade_invalid_path_returns_400() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let resp = send_request(
        ctx.addr(),
        "GET",
        "/v1/agent/shells/has/slash/ws",
        &[
            ("Connection", "Upgrade"),
            ("Upgrade", "websocket"),
            ("Sec-WebSocket-Version", "13"),
            ("Sec-WebSocket-Key", "dGhlIHNhbXBsZSBub25jZQ=="),
        ],
        None,
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_request");
}

#[test]
fn agent_shell_delete_invalid_path_returns_400() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let resp = delete(ctx.addr(), "/v1/agent/shells/has/slash");
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_request");
}

#[test]
fn agent_shell_create_with_invalid_cwd_returns_500() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let resp = post_json(
        ctx.addr(),
        "/v1/agent/shells",
        &serde_json::json!({"cwd": "/nonexistent/path/xyz_9182736"}),
    );
    assert_eq!(resp.status, 500, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "shell_error");
}

// ---------------------------------------------------------------------------
// WEBSOCKET I/O
// ---------------------------------------------------------------------------

#[tokio::test]
async fn agent_shell_ws_echo_command_output() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let shell_id = create_shell(&ctx);
    let mut ws = ws_connect(&ctx, &shell_id).await;

    // Write a command and exit to the PTY.
    ws.send(Message::Binary(b"echo hello\nexit\n".to_vec().into()))
        .await
        .expect("ws send");

    let (output, exit_event) = ws_drain_output(&mut ws).await;
    let text = String::from_utf8_lossy(&output);
    assert!(
        text.contains("hello"),
        "expected 'hello' in PTY output, got: {text}"
    );

    let exit = exit_event.expect("should receive exit event");
    assert_eq!(exit["type"], "exit");
}

#[tokio::test]
async fn agent_shell_ws_exit_event_on_shell_exit() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let shell_id = create_shell(&ctx);
    let mut ws = ws_connect(&ctx, &shell_id).await;

    ws.send(Message::Binary(b"exit 0\n".to_vec().into()))
        .await
        .expect("ws send");

    let (_, exit_event) = ws_drain_output(&mut ws).await;
    let exit = exit_event.expect("should receive exit event");
    assert_eq!(exit["type"], "exit");
}

#[tokio::test]
async fn agent_shell_ws_invalid_json_returns_error() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let shell_id = create_shell(&ctx);
    let mut ws = ws_connect(&ctx, &shell_id).await;

    ws.send(Message::Text("{not valid json".into()))
        .await
        .expect("ws send");

    let resp = ws_next_text(&mut ws).await;
    assert_eq!(resp["type"], "error");
    assert_eq!(resp["code"], "invalid_json");

    // Clean up: exit the shell so the server task terminates.
    let _ = ws.send(Message::Binary(b"exit\n".to_vec().into())).await;
}

#[tokio::test]
async fn agent_shell_ws_unsupported_message_type() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let shell_id = create_shell(&ctx);
    let mut ws = ws_connect(&ctx, &shell_id).await;

    ws.send(Message::Text(
        serde_json::json!({"type": "unknown_type"}).to_string().into(),
    ))
    .await
    .expect("ws send");

    let resp = ws_next_text(&mut ws).await;
    assert_eq!(resp["type"], "error");
    assert_eq!(resp["code"], "unsupported_message");

    let _ = ws.send(Message::Binary(b"exit\n".to_vec().into())).await;
}

#[tokio::test]
async fn agent_shell_ws_resize_zero_returns_error() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let shell_id = create_shell(&ctx);
    let mut ws = ws_connect(&ctx, &shell_id).await;

    ws.send(Message::Text(
        serde_json::json!({"type": "resize", "cols": 0, "rows": 0})
            .to_string()
            .into(),
    ))
    .await
    .expect("ws send");

    let resp = ws_next_text(&mut ws).await;
    assert_eq!(resp["type"], "error");
    assert_eq!(resp["code"], "invalid_resize");

    let _ = ws.send(Message::Binary(b"exit\n".to_vec().into())).await;
}

#[tokio::test]
async fn agent_shell_ws_signal_kill_terminates_shell() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let shell_id = create_shell(&ctx);
    let mut ws = ws_connect(&ctx, &shell_id).await;

    // Wait for shell to produce initial output (prompt/banner).
    // This confirms the PTY child is running and the WS loops are active.
    // Timeout is a deadlock guard only — the shell always produces output.
    match tokio::time::timeout(std::time::Duration::from_secs(5), ws.next()).await {
        Ok(Some(Ok(_))) => {}
        other => panic!("shell did not produce initial output: {other:?}"),
    }

    // Use SIGKILL: interactive shells ignore SIGTERM (POSIX behaviour),
    // so only SIGKILL (which cannot be caught/ignored) reliably terminates
    // an idle interactive shell.
    ws.send(Message::Text(
        serde_json::json!({"type": "signal", "signal": "KILL"})
            .to_string()
            .into(),
    ))
    .await
    .expect("ws send");

    let (_, exit_event) = ws_drain_output(&mut ws).await;
    let exit = exit_event.expect("should receive exit event after SIGKILL");
    assert_eq!(exit["type"], "exit");
}

#[tokio::test]
async fn agent_shell_ws_signal_invalid_returns_error() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let shell_id = create_shell(&ctx);
    let mut ws = ws_connect(&ctx, &shell_id).await;

    ws.send(Message::Text(
        serde_json::json!({"type": "signal", "signal": "BOGUS"})
            .to_string()
            .into(),
    ))
    .await
    .expect("ws send");

    let resp = ws_next_text(&mut ws).await;
    assert_eq!(resp["type"], "error");
    assert_eq!(resp["code"], "invalid_signal");

    let _ = ws.send(Message::Binary(b"exit\n".to_vec().into())).await;
}
