//! Integration tests for `DELETE /v1/repo/models/{model_id}`.

use crate::server::common::*;

fn repo_config() -> ServerConfig {
    let mut config = ServerConfig::new();
    config.no_bucket = true;
    config
}

// ---------------------------------------------------------------------------
// Basic behavior
// ---------------------------------------------------------------------------

/// DELETE for a nonexistent model returns 200 with deleted=false.
#[test]
fn delete_nonexistent_returns_ok_with_false() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = delete(
        ctx.addr(),
        "/v1/repo/models/nonexistent-org/nonexistent-model",
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(json["deleted"], false);
    assert_eq!(json["model_id"], "nonexistent-org/nonexistent-model");
}

/// Response has application/json content type.
#[test]
fn delete_has_json_content_type() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = delete(
        ctx.addr(),
        "/v1/repo/models/nonexistent-org/nonexistent-model",
    );
    assert_eq!(resp.status, 200);
    assert_eq!(resp.header("content-type"), Some("application/json"));
}

/// Response contains exactly the expected fields.
#[test]
fn delete_response_has_expected_fields() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = delete(ctx.addr(), "/v1/repo/models/org/model");
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let obj = json.as_object().expect("response should be an object");
    assert!(obj.contains_key("deleted"), "should have 'deleted' field");
    assert!(obj.contains_key("model_id"), "should have 'model_id' field");
    assert!(json["deleted"].is_boolean(), "deleted should be a boolean");
    assert!(json["model_id"].is_string(), "model_id should be a string");
}

/// Model ID with slashes is preserved correctly in the response.
#[test]
fn delete_preserves_model_id_with_slashes() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = delete(ctx.addr(), "/v1/repo/models/meta-llama/Llama-3.2-1B");
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(
        json["model_id"], "meta-llama/Llama-3.2-1B",
        "model ID with slash should round-trip correctly"
    );
}

// ---------------------------------------------------------------------------
// URL-encoded model IDs
// ---------------------------------------------------------------------------

/// Model ID with percent-encoded slashes is decoded correctly.
#[test]
fn delete_percent_encoded_model_id() {
    let ctx = ServerTestContext::new(repo_config());
    // "meta-llama%2FLlama-3.2-1B" → "meta-llama/Llama-3.2-1B"
    let resp = delete(ctx.addr(), "/v1/repo/models/meta-llama%2FLlama-3.2-1B");
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(
        json["model_id"], "meta-llama/Llama-3.2-1B",
        "percent-encoded slash should be decoded"
    );
}

// ---------------------------------------------------------------------------
// Bare path variant
// ---------------------------------------------------------------------------

/// DELETE /repo/models/{id} works (without /v1 prefix).
#[test]
fn delete_bare_path() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = delete(ctx.addr(), "/repo/models/some-org/some-model");
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(json["deleted"], false);
    assert_eq!(json["model_id"], "some-org/some-model");
}
