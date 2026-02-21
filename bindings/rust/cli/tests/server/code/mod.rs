//! Integration tests for the `/v1/code` tree-sitter endpoints.
//!
//! Uses in-process hyper connections (no TCP socket) for speed and determinism.
//! No model loading or disk storage required.
//!
//! Run: `cargo test -p talu-cli --test server server::code`

mod graph;
mod highlight;
mod languages;
mod parse;
mod query;
mod reaper;
mod session;
mod ws;

use std::collections::HashMap;
use std::sync::Arc;

use bytes::Bytes;
use futures_util::{SinkExt, StreamExt};
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::client::conn::http1 as client_http1;
use hyper::server::conn::http1 as server_http1;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use hyper_util::service::TowerToHyperService;
use serde_json::Value;
use tokio::sync::Mutex;
use tokio_tungstenite::tungstenite::Message;

use talu_cli::server::http::Router;
use talu_cli::server::state::{AppState, BackendState};

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------

fn build_app() -> Router {
    let state = AppState {
        backend: Arc::new(Mutex::new(BackendState {
            backend: None,
            current_model: None,
        })),
        configured_model: None,
        response_store: Mutex::new(HashMap::new()),
        gateway_secret: None,
        tenant_registry: None,
        bucket_path: None,
        html_dir: None,
        plugin_tokens: Mutex::new(HashMap::new()),
        max_file_upload_bytes: 100 * 1024 * 1024,
        max_file_inspect_bytes: 50 * 1024 * 1024,
        code_sessions: Mutex::new(HashMap::new()),
        code_session_ttl: std::time::Duration::from_secs(15 * 60),
    };
    Router::new(Arc::new(state))
}

async fn send_request(router: &Router, req: Request<Full<Bytes>>) -> Response<Incoming> {
    let (client_io, server_io) = tokio::io::duplex(1024 * 1024);

    let service = TowerToHyperService::new(router.clone());
    tokio::spawn(async move {
        let _ = server_http1::Builder::new()
            .serve_connection(TokioIo::new(server_io), service)
            .with_upgrades()
            .await;
    });

    let (mut sender, conn) = client_http1::handshake(TokioIo::new(client_io))
        .await
        .expect("client handshake failed");

    tokio::spawn(async move {
        let _ = conn.await;
    });

    sender.send_request(req).await.expect("send_request failed")
}

type WsStream = tokio_tungstenite::WebSocketStream<tokio::io::DuplexStream>;

/// Open a WebSocket connection to `/v1/code/ws` via in-process duplex IO.
async fn ws_connect(router: &Router) -> WsStream {
    let (client_io, server_io) = tokio::io::duplex(1024 * 1024);

    let service = TowerToHyperService::new(router.clone());
    tokio::spawn(async move {
        let _ = server_http1::Builder::new()
            .serve_connection(TokioIo::new(server_io), service)
            .with_upgrades()
            .await;
    });

    let (ws, _) = tokio_tungstenite::client_async("ws://localhost/v1/code/ws", client_io)
        .await
        .expect("WebSocket handshake failed");

    ws
}

/// Send a JSON message and receive the response.
async fn ws_roundtrip(ws: &mut WsStream, msg: &Value) -> Value {
    ws.send(Message::Text(msg.to_string()))
        .await
        .expect("ws send failed");
    let reply = ws.next().await.expect("ws stream ended").expect("ws recv failed");
    match reply {
        Message::Text(text) => serde_json::from_str(&text).expect("invalid JSON in ws reply"),
        other => panic!("expected text message, got: {other:?}"),
    }
}

async fn body_json(resp: Response<Incoming>) -> (StatusCode, Value) {
    let status = resp.status();
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap_or_else(|e| {
        panic!(
            "invalid JSON in response body: {e}\nbody: {}",
            String::from_utf8_lossy(&body)
        )
    });
    (status, json)
}

fn get(uri: &str) -> Request<Full<Bytes>> {
    Request::builder()
        .method("GET")
        .uri(uri)
        .body(Full::new(Bytes::new()))
        .unwrap()
}

fn post_json(uri: &str, body: &Value) -> Request<Full<Bytes>> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body.to_string())))
        .unwrap()
}

fn delete(uri: &str) -> Request<Full<Bytes>> {
    Request::builder()
        .method("DELETE")
        .uri(uri)
        .body(Full::new(Bytes::new()))
        .unwrap()
}
