use std::collections::HashMap;
use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::client::conn::http1 as client_http1;
use hyper::server::conn::http1 as server_http1;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use hyper_util::service::TowerToHyperService;
use tokio::sync::Mutex;

use talu_cli::server::http::Router;
use talu_cli::server::state::{AppState, BackendState};
use talu_cli::server::{AgentRuntimeMode, SandboxBackend};

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
        workdir: Some(std::env::current_dir().expect("cwd")),
        agent_policy_json: None,
        agent_policy: None,
        html_dir: None,
        plugin_tokens: Mutex::new(HashMap::new()),
        max_file_upload_bytes: 100 * 1024 * 1024,
        max_file_inspect_bytes: 50 * 1024 * 1024,
        code_sessions: Mutex::new(HashMap::new()),
        code_session_ttl: std::time::Duration::from_secs(15 * 60),
        shell_sessions: Mutex::new(HashMap::new()),
        shell_session_ttl: std::time::Duration::from_secs(15 * 60),
        process_sessions: Mutex::new(HashMap::new()),
        process_session_ttl: std::time::Duration::from_secs(15 * 60),
        kv_handles: Mutex::new(HashMap::new()),
        collab_handles: Mutex::new(HashMap::new()),
        agent_runtime_mode: AgentRuntimeMode::Host,
        sandbox_backend: SandboxBackend::LinuxLocal,
        pubsub: Mutex::new(talu_cli::server::pubsub::PubSubState::new()),
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

fn ws_request(path: &str) -> Request<Full<Bytes>> {
    Request::builder()
        .method("GET")
        .uri(path)
        .header("connection", "Upgrade")
        .header("upgrade", "websocket")
        .header("sec-websocket-key", "dGhlIHNhbXBsZSBub25jZQ==")
        .header("sec-websocket-version", "13")
        .body(Full::new(Bytes::new()))
        .unwrap()
}

#[tokio::test]
async fn collab_pubsub_websocket_is_exposed_under_collab_namespace() {
    let app = build_app();
    let resp = send_request(&app, ws_request("/v1/collab/pubsub/ws")).await;
    assert_eq!(resp.status(), StatusCode::SWITCHING_PROTOCOLS);
}

#[tokio::test]
async fn legacy_pubsub_route_is_not_listed_anymore() {
    let app = build_app();
    let resp = send_request(&app, ws_request("/v1/pubsub/ws")).await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&body);
    assert!(text.contains("not found"), "body: {text}");
}
