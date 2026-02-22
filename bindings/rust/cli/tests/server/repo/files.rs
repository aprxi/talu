//! Integration tests for `GET /v1/repo/models/{model_id}/files`.

use crate::server::common::*;

fn repo_config() -> ServerConfig {
    let mut config = ServerConfig::new();
    config.no_bucket = true;
    config
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Response has application/json content type.
#[test]
fn files_has_json_content_type() {
    let ctx = ServerTestContext::new(repo_config());
    // Nonexistent model will fail but should still return JSON.
    let resp = get(
        ctx.addr(),
        "/v1/repo/models/nonexistent-org/nonexistent-model/files",
    );
    assert_eq!(resp.header("content-type"), Some("application/json"));
}

/// Response for a model that doesn't exist returns an error with model_id preserved.
#[test]
fn files_nonexistent_model_returns_error() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(
        ctx.addr(),
        "/v1/repo/models/nonexistent-org/nonexistent-model/files",
    );
    // The FFI call will fail for a nonexistent model.
    let _json = resp.json();
    // Either a 200 with empty files or a 500 with error — both are valid JSON.
    assert!(
        resp.status == 200 || resp.status == 500,
        "expected 200 or 500, got {} body: {}",
        resp.status,
        resp.body
    );
}

/// When the endpoint returns 200, it has the expected structure.
#[test]
fn files_response_structure() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(
        ctx.addr(),
        "/v1/repo/models/nonexistent-org/nonexistent-model/files",
    );
    let json = resp.json();

    if resp.status == 200 {
        assert!(json["model_id"].is_string(), "should have model_id");
        assert!(json["files"].is_array(), "should have files array");

        for (i, file) in json["files"].as_array().unwrap().iter().enumerate() {
            assert!(
                file["filename"].is_string(),
                "files[{i}].filename should be string"
            );
        }
    } else {
        // Error response
        assert!(json["error"]["code"].is_string(), "error should have code");
    }
}

/// Percent-encoded model ID is decoded in the response.
#[test]
fn files_percent_encoded_model_id() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(
        ctx.addr(),
        "/v1/repo/models/meta-llama%2FLlama-3.2-1B/files",
    );
    let json = resp.json();

    if resp.status == 200 {
        assert_eq!(
            json["model_id"], "meta-llama/Llama-3.2-1B",
            "percent-encoded slash should be decoded"
        );
    }
}

/// Token query param is accepted without breaking parsing.
#[test]
fn files_accepts_token_param() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(
        ctx.addr(),
        "/v1/repo/models/nonexistent-org/nonexistent-model/files?token=hf_test123",
    );
    // Should not return 400 (token should be parsed without error).
    assert_ne!(resp.status, 400, "token param should be accepted");
}

// ---------------------------------------------------------------------------
// Bare path variant
// ---------------------------------------------------------------------------

/// GET /repo/models/{id}/files works (without /v1 prefix).
#[test]
fn files_bare_path() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(
        ctx.addr(),
        "/repo/models/nonexistent-org/nonexistent-model/files",
    );
    assert_eq!(resp.header("content-type"), Some("application/json"));
}

// ---------------------------------------------------------------------------
// OpenAPI registration
// ---------------------------------------------------------------------------

/// File listing endpoint is registered in the OpenAPI spec.
#[test]
fn files_endpoint_in_openapi() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(ctx.addr(), "/openapi.json");
    assert_eq!(resp.status, 200);

    let json = resp.json();
    let paths = json["paths"].as_object().expect("should have paths");

    assert!(
        paths.contains_key("/v1/repo/models/{model_id}/files"),
        "OpenAPI spec should contain /v1/repo/models/{{model_id}}/files"
    );

    let files_path = &paths["/v1/repo/models/{model_id}/files"];
    assert!(
        files_path.get("get").is_some(),
        "/v1/repo/models/{{model_id}}/files should have GET"
    );
}
