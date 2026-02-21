//! Integration tests for the stateless `/v1/file` endpoints.

mod inspect;
mod transform;

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
use serde_json::Value;
use tokio::sync::Mutex;

use talu_cli::server::http::Router;
use talu_cli::server::state::{AppState, BackendState};

// Test images compiled into the test binary.
const RED_PNG: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../core/tests/image/corpus/1x1_red.png"
));
const RED_JPEG: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../core/tests/image/corpus/1x1_red.jpg"
));
const RED_WEBP: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../core/tests/image/corpus/1x1_red.webp"
));
const BLUE_JPEG_2X3: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../core/tests/image/corpus/2x3_blue.jpg"
));
const EXIF_ROTATE90: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../core/tests/image/corpus/exif_rotate90.jpg"
));

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

fn build_app_with_inspect_limit(limit: u64) -> Router {
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
        max_file_inspect_bytes: limit,
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

async fn body_bytes(resp: Response<Incoming>) -> (StatusCode, Bytes) {
    let status = resp.status();
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    (status, body)
}

/// Build a multipart/form-data request body from a file part and optional text fields.
fn multipart_request(
    method: &str,
    uri: &str,
    boundary: &str,
    file_bytes: &[u8],
    file_name: &str,
    file_content_type: &str,
    fields: &[(&str, &str)],
) -> Request<Full<Bytes>> {
    let mut body = Vec::new();

    for (name, value) in fields {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
        );
        body.extend_from_slice(value.as_bytes());
        body.extend_from_slice(b"\r\n");
    }

    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        format!(
            "Content-Disposition: form-data; name=\"file\"; filename=\"{file_name}\"\r\n\
             Content-Type: {file_content_type}\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(file_bytes);
    body.extend_from_slice(b"\r\n");
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());

    let content_type = format!("multipart/form-data; boundary={boundary}");
    Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", content_type)
        .body(Full::new(Bytes::from(body)))
        .unwrap()
}

/// Build a multipart/form-data body with only text fields (no file part).
fn multipart_no_file(uri: &str, boundary: &str, fields: &[(&str, &str)]) -> Request<Full<Bytes>> {
    let mut body = Vec::new();
    for (name, value) in fields {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
        );
        body.extend_from_slice(value.as_bytes());
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());

    let content_type = format!("multipart/form-data; boundary={boundary}");
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", content_type)
        .body(Full::new(Bytes::from(body)))
        .unwrap()
}

/// Extract width and height from raw PNG bytes by reading the IHDR chunk.
/// PNG layout: 8-byte signature, then IHDR chunk (length[4] + "IHDR"[4] + width[4] + height[4]).
fn png_dimensions(data: &[u8]) -> (u32, u32) {
    assert!(data.len() >= 24, "PNG too small for IHDR");
    assert_eq!(&data[0..8], b"\x89PNG\r\n\x1a\n", "not a valid PNG");
    let width = u32::from_be_bytes([data[16], data[17], data[18], data[19]]);
    let height = u32::from_be_bytes([data[20], data[21], data[22], data[23]]);
    (width, height)
}
