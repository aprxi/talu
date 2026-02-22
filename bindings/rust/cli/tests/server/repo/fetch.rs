//! Integration tests for `POST /v1/repo/models` (download).

use crate::server::common::*;
use serde_json::json;
use std::io::Write;

fn repo_config() -> ServerConfig {
    let mut config = ServerConfig::new();
    config.no_bucket = true;
    config
}

// ---------------------------------------------------------------------------
// Validation errors
// ---------------------------------------------------------------------------

/// Invalid JSON body returns 400 with standard error envelope.
#[test]
fn fetch_invalid_json_returns_400() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/repo/models",
        &[("Content-Type", "application/json")],
        Some("not json {{{"),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);

    let json = resp.json();
    let error = json["error"].as_object().expect("should have error object");
    assert_eq!(error["code"], "invalid_request");
    assert!(
        error["message"].as_str().is_some_and(|m| !m.is_empty()),
        "error.message should be a non-empty string"
    );
}

/// Missing model_id field (required by serde) returns 400 with error envelope.
#[test]
fn fetch_missing_model_id_returns_400() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = post_json(ctx.addr(), "/v1/repo/models", &json!({}));
    assert_eq!(resp.status, 400, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(
        json["error"]["code"], "invalid_request",
        "serde deserialization failure should produce invalid_request"
    );
}

/// Empty model_id returns 400.
#[test]
fn fetch_empty_model_id_returns_400() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = post_json(ctx.addr(), "/v1/repo/models", &json!({"model_id": ""}));
    assert_eq!(resp.status, 400, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(json["error"]["code"], "invalid_request");
    assert!(
        json["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("model_id")),
        "error message should mention model_id"
    );
}

/// Error responses have application/json content type.
#[test]
fn fetch_error_has_json_content_type() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = post_json(ctx.addr(), "/v1/repo/models", &json!({"model_id": ""}));
    assert_eq!(resp.status, 400);
    assert_eq!(resp.header("content-type"), Some("application/json"));
}

// ---------------------------------------------------------------------------
// SSE streaming: validation errors bypass streaming
// ---------------------------------------------------------------------------

/// With Accept: text/event-stream, invalid JSON still returns 400 JSON
/// (not an SSE stream). Validation runs before the stream branch.
#[test]
fn fetch_stream_accept_invalid_json_returns_400_json() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/repo/models",
        &[
            ("Content-Type", "application/json"),
            ("Accept", "text/event-stream"),
        ],
        Some("not json"),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(
        resp.header("content-type"),
        Some("application/json"),
        "validation errors should return JSON, not SSE"
    );
    assert_eq!(resp.json()["error"]["code"], "invalid_request");
}

/// With Accept: text/event-stream, empty model_id still returns 400 JSON.
#[test]
fn fetch_stream_accept_empty_model_id_returns_400_json() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/repo/models",
        &[
            ("Content-Type", "application/json"),
            ("Accept", "text/event-stream"),
        ],
        Some(r#"{"model_id": ""}"#),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(
        resp.header("content-type"),
        Some("application/json"),
        "validation errors should return JSON even with SSE Accept header"
    );
    assert_eq!(resp.json()["error"]["code"], "invalid_request");
}

// ---------------------------------------------------------------------------
// endpoint_url and skip_weights fields
// ---------------------------------------------------------------------------

/// Request with endpoint_url is accepted (validation passes).
#[test]
fn fetch_accepts_endpoint_url() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = post_json(
        ctx.addr(),
        "/v1/repo/models",
        &json!({
            "model_id": "nonexistent-org/nonexistent-model",
            "endpoint_url": "https://custom-mirror.example.com"
        }),
    );
    // Will fail to actually download but should not return 400.
    // We just verify it's not rejected as invalid_request.
    assert_ne!(
        resp.json()["error"]["code"], "invalid_request",
        "endpoint_url field should be accepted"
    );
}

/// Request with skip_weights is accepted (validation passes).
#[test]
fn fetch_accepts_skip_weights() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = post_json(
        ctx.addr(),
        "/v1/repo/models",
        &json!({
            "model_id": "nonexistent-org/nonexistent-model",
            "skip_weights": true
        }),
    );
    // Will fail to actually download but should not return 400.
    assert_ne!(
        resp.json()["error"]["code"], "invalid_request",
        "skip_weights field should be accepted"
    );
}

// ---------------------------------------------------------------------------
// SSE streaming: error event for failed downloads
// ---------------------------------------------------------------------------

/// SSE streaming request for nonexistent model returns text/event-stream
/// with an error event (download fails, but response format is correct).
#[test]
fn fetch_stream_returns_event_stream_content_type() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/repo/models",
        &[
            ("Content-Type", "application/json"),
            ("Accept", "text/event-stream"),
        ],
        Some(r#"{"model_id": "nonexistent-org/nonexistent-model"}"#),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(
        resp.header("content-type"),
        Some("text/event-stream"),
        "SSE Accept should produce text/event-stream response"
    );
}

/// SSE stream for nonexistent model contains an error event with message.
#[test]
fn fetch_stream_error_event_for_nonexistent_model() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/repo/models",
        &[
            ("Content-Type", "application/json"),
            ("Accept", "text/event-stream"),
        ],
        Some(r#"{"model_id": "nonexistent-org/nonexistent-model"}"#),
    );
    assert_eq!(resp.status, 200);
    assert!(
        resp.body.contains("data: "),
        "SSE body should contain 'data: ' lines"
    );
    assert!(
        resp.body.contains("\"event\":\"error\"") || resp.body.contains("\"event\": \"error\""),
        "SSE body should contain error event for nonexistent model, got: {}",
        resp.body
    );
}

/// Client disconnect during SSE stream does not crash the server.
/// We open a streaming request, close the connection immediately, then verify
/// the server is still responsive by making another request.
#[test]
fn fetch_stream_client_disconnect_does_not_crash_server() {
    let ctx = ServerTestContext::new(repo_config());

    // Open a streaming request and close the connection without reading the response.
    {
        let body = r#"{"model_id": "nonexistent-org/some-model"}"#;
        let request = format!(
            "POST /v1/repo/models HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nAccept: text/event-stream\r\nContent-Length: {}\r\n\r\n{}",
            ctx.addr(),
            body.len(),
            body
        );
        let mut stream = std::net::TcpStream::connect_timeout(
            &ctx.addr().into(),
            std::time::Duration::from_secs(5),
        )
        .expect("connect");
        stream
            .set_write_timeout(Some(std::time::Duration::from_secs(5)))
            .expect("set write timeout");
        stream
            .write_all(request.as_bytes())
            .expect("write request");
        stream.flush().expect("flush");
        // Drop the stream immediately — simulates client disconnect.
    }

    // Small delay to let the server process the disconnect.
    std::thread::sleep(std::time::Duration::from_millis(200));

    // Verify the server is still alive by making a normal request.
    let resp = post_json(
        ctx.addr(),
        "/v1/repo/models",
        &json!({"model_id": "another-org/another-model"}),
    );
    // Should get an error (model not found) but NOT a connection failure.
    assert!(
        resp.status > 0,
        "Server should still be responsive after client disconnect"
    );
}

// ---------------------------------------------------------------------------
// Bare path variant
// ---------------------------------------------------------------------------

/// POST /repo/models with invalid body returns 400 (bare path routing works).
#[test]
fn fetch_bare_path_validation() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = post_json(ctx.addr(), "/repo/models", &json!({"model_id": ""}));
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_request");
}
