//! Integration tests for `POST /v1/repo/models` (download).

use crate::server::common::*;
use serde_json::json;

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
