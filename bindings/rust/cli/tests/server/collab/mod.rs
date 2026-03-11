//! Integration tests for `/v1/collab/resources/*`.

use base64::Engine as _;
use futures_util::{SinkExt, StreamExt};
use tempfile::TempDir;
use tokio::time::{timeout, Duration};
use tokio_tungstenite::tungstenite::Message;

use crate::server::common::*;
use crate::server::db::db_config;

type WsStream =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

async fn ws_connect(ctx: &ServerTestContext, path: &str) -> WsStream {
    let url = format!("ws://127.0.0.1:{}{}", ctx.addr().port(), path);
    let (ws, _) = tokio_tungstenite::connect_async(&url)
        .await
        .unwrap_or_else(|e| panic!("WS connect to {url} failed: {e}"));
    ws
}

async fn ws_next_json(ws: &mut WsStream) -> serde_json::Value {
    timeout(Duration::from_secs(5), async {
        loop {
            match ws.next().await {
                Some(Ok(Message::Text(text))) => {
                    return serde_json::from_str(&text)
                        .unwrap_or_else(|e| panic!("invalid JSON in WS text: {e}\nraw: {text}"));
                }
                Some(Ok(Message::Ping(_))) | Some(Ok(Message::Pong(_))) => continue,
                Some(Ok(Message::Close(_))) => panic!("WS closed before JSON message"),
                Some(Err(e)) => panic!("WS receive error: {e}"),
                None => panic!("WS stream ended before JSON message"),
                _ => continue,
            }
        }
    })
    .await
    .expect("timed out waiting for WS JSON message")
}

#[test]
fn collab_session_ops_history_snapshot_roundtrip() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));
    let payload = base64::engine::general_purpose::STANDARD.encode(br#"{"insert":"h"}"#);
    let snapshot = base64::engine::general_purpose::STANDARD.encode("hello".as_bytes());

    let open = post_json(
        ctx.addr(),
        "/v1/collab/resources/text_document/doc-1/sessions",
        &serde_json::json!({
            "participant_id": "human:1",
            "participant_kind": "human",
            "role": "editor",
        }),
    );
    assert_eq!(open.status, 200, "body: {}", open.body);
    let open_json = open.json();
    assert_eq!(open_json["resource_kind"], "text_document");
    assert_eq!(open_json["resource_id"], "doc-1");
    assert_eq!(open_json["participant_id"], "human:1");
    assert!(open_json["namespace"]
        .as_str()
        .unwrap_or_default()
        .starts_with("collab-text_document-"));

    let submit = post_json(
        ctx.addr(),
        "/v1/collab/resources/text_document/doc-1/ops",
        &serde_json::json!({
            "actor_id": "agent:alpha",
            "actor_seq": 1,
            "op_id": "op-0001",
            "payload_base64": payload,
            "snapshot_base64": snapshot,
        }),
    );
    assert_eq!(submit.status, 200, "body: {}", submit.body);
    let submit_json = submit.json();
    assert_eq!(submit_json["accepted"], true);
    assert_eq!(submit_json["resource_kind"], "text_document");
    assert_eq!(submit_json["resource_id"], "doc-1");
    assert_eq!(submit_json["op_key"], "ops/agent:alpha:1:op-0001");

    let history = get(
        ctx.addr(),
        "/v1/collab/resources/text_document/doc-1/history?limit=10",
    );
    assert_eq!(history.status, 200, "body: {}", history.body);
    let history_json = history.json();
    assert_eq!(history_json["count"], 1);
    assert_eq!(history_json["data"][0]["actor_id"], "agent:alpha");
    assert_eq!(history_json["data"][0]["actor_seq"], 1);
    assert_eq!(history_json["data"][0]["op_id"], "op-0001");

    let snap = get(
        ctx.addr(),
        "/v1/collab/resources/text_document/doc-1/snapshot",
    );
    assert_eq!(snap.status, 200, "body: {}", snap.body);
    let snap_json = snap.json();
    assert_eq!(
        snap_json["snapshot_base64"].as_str().unwrap_or_default(),
        base64::engine::general_purpose::STANDARD.encode("hello".as_bytes())
    );
}

#[test]
fn collab_history_supports_cursor_pagination() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    for seq in 1..=3_u64 {
        let payload = base64::engine::general_purpose::STANDARD
            .encode(format!(r#"{{"seq":{seq}}}"#).as_bytes());
        let submit = post_json(
            ctx.addr(),
            "/v1/collab/resources/text_document/doc-page/ops",
            &serde_json::json!({
                "actor_id": "agent:alpha",
                "actor_seq": seq,
                "op_id": format!("op-{seq}"),
                "payload_base64": payload,
            }),
        );
        assert_eq!(submit.status, 200, "body: {}", submit.body);
    }

    let page_one = get(
        ctx.addr(),
        "/v1/collab/resources/text_document/doc-page/history?limit=2",
    );
    assert_eq!(page_one.status, 200, "body: {}", page_one.body);
    let page_one_json = page_one.json();
    assert_eq!(page_one_json["count"], 2);
    assert_eq!(page_one_json["data"][0]["op_id"], "op-1");
    assert_eq!(page_one_json["data"][1]["op_id"], "op-2");
    let cursor = page_one_json["next_cursor"]
        .as_str()
        .expect("next cursor")
        .to_string();

    let page_two = get(
        ctx.addr(),
        &format!("/v1/collab/resources/text_document/doc-page/history?limit=2&cursor={cursor}"),
    );
    assert_eq!(page_two.status, 200, "body: {}", page_two.body);
    let page_two_json = page_two.json();
    assert_eq!(page_two_json["count"], 1);
    assert_eq!(page_two_json["data"][0]["op_id"], "op-3");
    assert_eq!(page_two_json["next_cursor"], serde_json::Value::Null);
}

#[test]
fn collab_presence_put_and_get_roundtrip() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let put = put_json(
        ctx.addr(),
        "/v1/collab/resources/file_buffer/main.zig/presence/human%3A1",
        &serde_json::json!({
            "presence": {
                "cursor": 42,
                "selection": [40, 45],
                "focused": true
            }
        }),
    );
    assert_eq!(put.status, 200, "body: {}", put.body);
    let put_json = put.json();
    assert_eq!(put_json["participant_id"], "human:1");
    assert_eq!(put_json["ttl_ms"], 5000);

    let get_resp = get(
        ctx.addr(),
        "/v1/collab/resources/file_buffer/main.zig/presence/human%3A1",
    );
    assert_eq!(get_resp.status, 200, "body: {}", get_resp.body);
    let get_json = get_resp.json();
    assert_eq!(get_json["found"], true);
    assert_eq!(get_json["participant_id"], "human:1");
    assert_eq!(get_json["presence"]["cursor"], 42);
    assert_eq!(get_json["presence"]["selection"][0], 40);
    assert_eq!(get_json["presence"]["selection"][1], 45);
}

#[test]
fn collab_resource_ws_upgrade_returns_switching_protocols() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let resp = send_request(
        ctx.addr(),
        "GET",
        "/v1/collab/resources/text_document/doc-1/ws",
        &[
            ("Connection", "Upgrade"),
            ("Upgrade", "websocket"),
            ("Sec-WebSocket-Version", "13"),
            ("Sec-WebSocket-Key", "dGhlIHNhbXBsZSBub25jZQ=="),
        ],
        None,
    );

    assert_eq!(resp.status, 101, "headers: {} body: {}", resp.headers, resp.body);
}

#[tokio::test]
async fn collab_resource_ws_submit_op_broadcasts_snapshot() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));
    let resource_path = "/v1/collab/resources/workdir_file/notes.txt/ws";
    let payload = base64::engine::general_purpose::STANDARD
        .encode(br#"{"type":"ui_live","source":"test"}"#);
    let snapshot =
        base64::engine::general_purpose::STANDARD.encode("hello from websocket".as_bytes());

    let mut ws_a = ws_connect(&ctx, resource_path).await;
    let mut ws_b = ws_connect(&ctx, resource_path).await;

    ws_a.send(Message::Text(
        serde_json::json!({
            "type": "open",
            "participant_id": "human:test:a",
            "participant_kind": "human",
            "role": "editor",
        })
        .to_string()
        .into(),
    ))
    .await
    .expect("send open A");
    ws_b.send(Message::Text(
        serde_json::json!({
            "type": "open",
            "participant_id": "human:test:b",
            "participant_kind": "human",
            "role": "editor",
        })
        .to_string()
        .into(),
    ))
    .await
    .expect("send open B");

    let ready_a = ws_next_json(&mut ws_a).await;
    let ready_b = ws_next_json(&mut ws_b).await;
    assert_eq!(ready_a["type"], "ready");
    assert_eq!(ready_b["type"], "ready");

    ws_a.send(Message::Text(
        serde_json::json!({
            "type": "submit_op",
            "actor_seq": 1,
            "op_id": "ws-op-1",
            "payload_base64": payload,
            "snapshot_base64": snapshot,
        })
        .to_string()
        .into(),
    ))
    .await
    .expect("send submit_op");

    let ack = ws_next_json(&mut ws_a).await;
    assert_eq!(ack["type"], "ack");
    assert_eq!(ack["op_id"], "ws-op-1");

    let snapshot_msg = ws_next_json(&mut ws_b).await;
    assert_eq!(snapshot_msg["type"], "snapshot");
    assert_eq!(
        snapshot_msg["snapshot_base64"].as_str().unwrap_or_default(),
        base64::engine::general_purpose::STANDARD.encode("hello from websocket".as_bytes())
    );

    let snap = get(
        ctx.addr(),
        "/v1/collab/resources/workdir_file/notes.txt/snapshot",
    );
    assert_eq!(snap.status, 200, "body: {}", snap.body);
    assert_eq!(
        snap.json()["snapshot_base64"].as_str().unwrap_or_default(),
        base64::engine::general_purpose::STANDARD.encode("hello from websocket".as_bytes())
    );
}
