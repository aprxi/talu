//! Integration tests for pin management endpoints.

use crate::server::common::*;
use serde_json::json;
use std::path::PathBuf;

/// Config with a bucket (required for pin operations).
fn pin_config() -> (ServerConfig, PathBuf) {
    let dir = tempfile::tempdir().expect("tempdir");
    let bucket = dir.keep();
    let mut config = ServerConfig::new();
    config.bucket = Some(bucket.clone());
    (config, bucket)
}

/// Config without a bucket (pin operations should fail).
fn no_bucket_config() -> ServerConfig {
    let mut config = ServerConfig::new();
    config.no_bucket = true;
    config
}

// ---------------------------------------------------------------------------
// Pin / Unpin basics
// ---------------------------------------------------------------------------

/// POST /v1/repo/pins returns pinned=true.
#[test]
fn pin_returns_pinned_true() {
    let (config, _bucket) = pin_config();
    let ctx = ServerTestContext::new(config);
    let resp = post_json(
        ctx.addr(),
        "/v1/repo/pins",
        &json!({"model_id": "test-org/test-model"}),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(json["model_id"], "test-org/test-model");
    assert_eq!(json["pinned"], true);
}

/// Pinning the same model twice succeeds (idempotent).
#[test]
fn pin_idempotent() {
    let (config, _bucket) = pin_config();
    let ctx = ServerTestContext::new(config);

    let resp1 = post_json(
        ctx.addr(),
        "/v1/repo/pins",
        &json!({"model_id": "test-org/model-a"}),
    );
    assert_eq!(resp1.status, 200);
    assert_eq!(resp1.json()["pinned"], true);

    let resp2 = post_json(
        ctx.addr(),
        "/v1/repo/pins",
        &json!({"model_id": "test-org/model-a"}),
    );
    assert_eq!(resp2.status, 200);
    assert_eq!(resp2.json()["pinned"], true);
}

/// Pin with empty model_id returns 400.
#[test]
fn pin_empty_model_id_returns_400() {
    let (config, _bucket) = pin_config();
    let ctx = ServerTestContext::new(config);
    let resp = post_json(
        ctx.addr(),
        "/v1/repo/pins",
        &json!({"model_id": ""}),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_request");
}

/// Pin with invalid JSON returns 400.
#[test]
fn pin_invalid_json_returns_400() {
    let (config, _bucket) = pin_config();
    let ctx = ServerTestContext::new(config);
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/repo/pins",
        &[("Content-Type", "application/json")],
        Some("not json"),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_request");
}

// ---------------------------------------------------------------------------
// List pins
// ---------------------------------------------------------------------------

/// GET /v1/repo/pins on a fresh bucket returns empty list.
#[test]
fn list_pins_empty() {
    let (config, _bucket) = pin_config();
    let ctx = ServerTestContext::new(config);
    let resp = get(ctx.addr(), "/v1/repo/pins");
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let pins = json["pins"].as_array().expect("pins should be array");
    assert!(pins.is_empty(), "fresh bucket should have no pins");
}

/// List pins after pinning a model shows it.
#[test]
fn list_pins_after_pin() {
    let (config, _bucket) = pin_config();
    let ctx = ServerTestContext::new(config);

    // Pin a model
    let resp = post_json(
        ctx.addr(),
        "/v1/repo/pins",
        &json!({"model_id": "test-org/test-model"}),
    );
    assert_eq!(resp.status, 200);

    // List pins
    let resp = get(ctx.addr(), "/v1/repo/pins");
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let pins = json["pins"].as_array().expect("pins should be array");
    assert_eq!(pins.len(), 1);
    assert_eq!(pins[0]["model_uri"], "test-org/test-model");
    assert!(pins[0]["pinned_at_ms"].is_number(), "pinned_at_ms should be number");
}

/// Pin list has application/json content type.
#[test]
fn list_pins_has_json_content_type() {
    let (config, _bucket) = pin_config();
    let ctx = ServerTestContext::new(config);
    let resp = get(ctx.addr(), "/v1/repo/pins");
    assert_eq!(resp.status, 200);
    assert_eq!(resp.header("content-type"), Some("application/json"));
}

// ---------------------------------------------------------------------------
// Unpin
// ---------------------------------------------------------------------------

/// DELETE /v1/repo/pins/{model_id} returns pinned=false.
#[test]
fn unpin_returns_pinned_false() {
    let (config, _bucket) = pin_config();
    let ctx = ServerTestContext::new(config);

    // Pin first
    let resp = post_json(
        ctx.addr(),
        "/v1/repo/pins",
        &json!({"model_id": "test-org/test-model"}),
    );
    assert_eq!(resp.status, 200);

    // Unpin
    let resp = delete(ctx.addr(), "/v1/repo/pins/test-org/test-model");
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(json["model_id"], "test-org/test-model");
    assert_eq!(json["pinned"], false);
}

/// Unpin a model that was never pinned still returns 200.
#[test]
fn unpin_nonexistent() {
    let (config, _bucket) = pin_config();
    let ctx = ServerTestContext::new(config);
    let resp = delete(ctx.addr(), "/v1/repo/pins/nonexistent-org/nonexistent-model");
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(json["pinned"], false);
}

/// Unpin with percent-encoded model ID decodes correctly.
#[test]
fn unpin_percent_encoded_model_id() {
    let (config, _bucket) = pin_config();
    let ctx = ServerTestContext::new(config);

    // Pin
    let resp = post_json(
        ctx.addr(),
        "/v1/repo/pins",
        &json!({"model_id": "meta-llama/Llama-3.2-1B"}),
    );
    assert_eq!(resp.status, 200);

    // Unpin with %2F encoding
    let resp = delete(ctx.addr(), "/v1/repo/pins/meta-llama%2FLlama-3.2-1B");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(
        resp.json()["model_id"], "meta-llama/Llama-3.2-1B",
        "percent-encoded slash should be decoded"
    );

    // Verify it's actually unpinned
    let resp = get(ctx.addr(), "/v1/repo/pins");
    let json = resp.json();
    let pins = json["pins"].as_array().expect("pins array");
    assert!(pins.is_empty(), "model should be unpinned");
}

/// Unpinning then listing confirms the model is removed.
#[test]
fn unpin_removes_from_list() {
    let (config, _bucket) = pin_config();
    let ctx = ServerTestContext::new(config);

    // Pin two models
    post_json(
        ctx.addr(),
        "/v1/repo/pins",
        &json!({"model_id": "org/model-a"}),
    );
    post_json(
        ctx.addr(),
        "/v1/repo/pins",
        &json!({"model_id": "org/model-b"}),
    );

    // Unpin one
    delete(ctx.addr(), "/v1/repo/pins/org/model-a");

    // List should have only model-b
    let resp = get(ctx.addr(), "/v1/repo/pins");
    let json = resp.json();
    let pins = json["pins"].as_array().expect("pins array");
    assert_eq!(pins.len(), 1);
    assert_eq!(pins[0]["model_uri"], "org/model-b");
}

// ---------------------------------------------------------------------------
// No-bucket errors
// ---------------------------------------------------------------------------

/// Pin operations return 400 when no bucket is configured.
#[test]
fn pin_no_bucket_returns_400() {
    let ctx = ServerTestContext::new(no_bucket_config());

    let resp = get(ctx.addr(), "/v1/repo/pins");
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "pin_store_unavailable");

    let resp = post_json(
        ctx.addr(),
        "/v1/repo/pins",
        &json!({"model_id": "org/model"}),
    );
    assert_eq!(resp.status, 400);
    assert_eq!(resp.json()["error"]["code"], "pin_store_unavailable");

    let resp = delete(ctx.addr(), "/v1/repo/pins/org/model");
    assert_eq!(resp.status, 400);
    assert_eq!(resp.json()["error"]["code"], "pin_store_unavailable");
}

// ---------------------------------------------------------------------------
// Sync pins
// ---------------------------------------------------------------------------

/// Sync pins with dry_run returns correct structure.
#[test]
fn sync_pins_dry_run() {
    let (config, _bucket) = pin_config();
    let ctx = ServerTestContext::new(config);

    // Pin a model (it won't be cached)
    post_json(
        ctx.addr(),
        "/v1/repo/pins",
        &json!({"model_id": "nonexistent-org/nonexistent-model"}),
    );

    // Sync with dry_run (default)
    let resp = post_json(
        ctx.addr(),
        "/v1/repo/sync-pins",
        &json!({}),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert!(json["total"].is_number(), "total should be number");
    assert!(json["cached"].is_number(), "cached should be number");
    assert!(json["missing"].is_number(), "missing should be number");
    assert!(json["downloaded"].is_number(), "downloaded should be number");
    assert!(json["errors"].is_array(), "errors should be array");

    // dry_run should not download anything
    assert_eq!(json["downloaded"], 0, "dry_run should not download");
    assert_eq!(json["total"], 1);
    assert_eq!(json["missing"], 1);

    // missing_size_bytes may be present (null if HF API unreachable) or absent.
    // When present, it should be a number.
    if !json["missing_size_bytes"].is_null() {
        if let Some(msb) = json.get("missing_size_bytes") {
            assert!(msb.is_number(), "missing_size_bytes should be a number when present");
        }
    }
}

/// Sync pins with SSE Accept header returns text/event-stream.
#[test]
fn sync_pins_sse_returns_event_stream() {
    let (config, _bucket) = pin_config();
    let ctx = ServerTestContext::new(config);

    // Pin a model
    post_json(
        ctx.addr(),
        "/v1/repo/pins",
        &json!({"model_id": "nonexistent-org/nonexistent-model"}),
    );

    // Sync with dry_run=false and SSE accept.
    // The download will fail for a nonexistent model, but we should still
    // get SSE events (scan, download_start, download_error, done).
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/repo/sync-pins",
        &[
            ("Content-Type", "application/json"),
            ("Accept", "text/event-stream"),
        ],
        Some(r#"{"dry_run": false}"#),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(
        resp.header("content-type"),
        Some("text/event-stream"),
        "SSE Accept should produce text/event-stream response"
    );

    // Body should contain SSE data lines.
    assert!(
        resp.body.contains("data: "),
        "SSE body should contain 'data: ' lines"
    );
    // Should contain scan event.
    assert!(
        resp.body.contains("\"event\":\"scan\"") || resp.body.contains("\"event\": \"scan\""),
        "SSE body should contain scan event"
    );
    // Should contain done event.
    assert!(
        resp.body.contains("\"event\":\"done\"") || resp.body.contains("\"event\": \"done\""),
        "SSE body should contain done event"
    );
}

/// Sync pins with dry_run=true ignores SSE Accept header (returns JSON).
#[test]
fn sync_pins_dry_run_ignores_sse_accept() {
    let (config, _bucket) = pin_config();
    let ctx = ServerTestContext::new(config);

    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/repo/sync-pins",
        &[
            ("Content-Type", "application/json"),
            ("Accept", "text/event-stream"),
        ],
        Some(r#"{}"#), // dry_run defaults to true
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(
        resp.header("content-type"),
        Some("application/json"),
        "dry_run should always return JSON regardless of Accept header"
    );
}

/// Sync pins with no bucket returns 400.
#[test]
fn sync_pins_no_bucket_returns_400() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let resp = post_json(
        ctx.addr(),
        "/v1/repo/sync-pins",
        &json!({}),
    );
    assert_eq!(resp.status, 400);
    assert_eq!(resp.json()["error"]["code"], "pin_store_unavailable");
}

// ---------------------------------------------------------------------------
// Bare path variants
// ---------------------------------------------------------------------------

/// Bare paths work for pin endpoints.
#[test]
fn pin_bare_path() {
    let (config, _bucket) = pin_config();
    let ctx = ServerTestContext::new(config);

    // List via bare path
    let resp = get(ctx.addr(), "/repo/pins");
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    // Pin via bare path
    let resp = post_json(
        ctx.addr(),
        "/repo/pins",
        &json!({"model_id": "test-org/test-model"}),
    );
    assert_eq!(resp.status, 200);

    // Unpin via bare path
    let resp = delete(ctx.addr(), "/repo/pins/test-org/test-model");
    assert_eq!(resp.status, 200);

    // Sync via bare path
    let resp = post_json(ctx.addr(), "/repo/sync-pins", &json!({}));
    assert_eq!(resp.status, 200);
}

// ---------------------------------------------------------------------------
// OpenAPI registration
// ---------------------------------------------------------------------------

/// Pin endpoints are registered in the OpenAPI spec.
#[test]
fn pin_endpoints_in_openapi() {
    let (config, _bucket) = pin_config();
    let ctx = ServerTestContext::new(config);
    let resp = get(ctx.addr(), "/openapi.json");
    assert_eq!(resp.status, 200);

    let json = resp.json();
    let paths = json["paths"].as_object().expect("should have paths");

    assert!(
        paths.contains_key("/v1/repo/pins"),
        "OpenAPI spec should contain /v1/repo/pins"
    );
    assert!(
        paths.contains_key("/v1/repo/sync-pins"),
        "OpenAPI spec should contain /v1/repo/sync-pins"
    );
    assert!(
        paths.contains_key("/v1/repo/pins/{model_id}"),
        "OpenAPI spec should contain /v1/repo/pins/{{model_id}}"
    );

    // Verify HTTP methods
    let pins_path = &paths["/v1/repo/pins"];
    assert!(pins_path.get("get").is_some(), "/v1/repo/pins should have GET");
    assert!(pins_path.get("post").is_some(), "/v1/repo/pins should have POST");

    let unpin_path = &paths["/v1/repo/pins/{model_id}"];
    assert!(
        unpin_path.get("delete").is_some(),
        "/v1/repo/pins/{{model_id}} should have DELETE"
    );

    let sync_path = &paths["/v1/repo/sync-pins"];
    assert!(
        sync_path.get("post").is_some(),
        "/v1/repo/sync-pins should have POST"
    );
}
