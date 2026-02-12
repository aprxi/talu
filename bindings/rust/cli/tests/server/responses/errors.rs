//! Error response tests over real HTTP.
//!
//! Basic error handling (invalid JSON, missing input, unknown path, unimplemented
//! method, error shape, error content-type) is covered by the in-process
//! `api_compliance` test suite. This module tests error scenarios that require
//! a loaded model or subprocess-specific behavior.

use crate::server::common::{
    model_config, post_json, require_model, send_request, ServerConfig, ServerTestContext,
};

/// Requesting an unavailable model returns 500 with inference_error.
#[test]
fn model_not_available_500() {
    let _ = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": "nonexistent-model-that-does-not-exist",
        "input": "Hello",
        "max_output_tokens": 5,
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 500, "body: {}", resp.body);
    let json = resp.json();
    assert_eq!(json["error"]["code"].as_str(), Some("inference_error"));
    assert!(json["error"]["message"]
        .as_str()
        .unwrap_or("")
        .contains("not available"));
}

/// Streaming with unavailable model returns error (not SSE stream).
#[test]
fn streaming_model_not_available() {
    let _ = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": "nonexistent-model-that-does-not-exist",
        "input": "Hello",
        "stream": true,
        "max_output_tokens": 5,
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 500, "body: {}", resp.body);
    let json = resp.json();
    assert!(
        json["error"]["code"].as_str().is_some(),
        "should have error code"
    );
}

/// Empty body returns 400.
#[test]
fn empty_body_400() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/responses",
        &[("Content-Type", "application/json")],
        Some(""),
    );
    assert_eq!(resp.status, 400);
}

/// Input present with no model (when server has configured model) still works.
#[test]
fn no_model_field_uses_default() {
    let _ = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "input": "Hello",
        "max_output_tokens": 5,
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(
        resp.status, 200,
        "should use configured model: {}",
        resp.body
    );
    let json = resp.json();
    assert!(
        json["model"].as_str().is_some(),
        "should have model in response"
    );
}

/// PUT on /v1/responses returns appropriate error.
#[test]
fn put_method_not_supported() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let resp = send_request(ctx.addr(), "PUT", "/v1/responses", &[], None);
    assert!(
        resp.status == 501 || resp.status == 405,
        "PUT should be rejected: {}",
        resp.status
    );
}

/// DELETE on /v1/responses returns appropriate error.
#[test]
fn delete_method_not_supported() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let resp = send_request(ctx.addr(), "DELETE", "/v1/responses", &[], None);
    assert!(
        resp.status == 501 || resp.status == 405,
        "DELETE should be rejected: {}",
        resp.status
    );
}

/// POST on /v1/models returns appropriate error (only GET).
#[test]
fn post_models_not_supported() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let resp = send_request(ctx.addr(), "POST", "/v1/models", &[], None);
    assert!(
        resp.status == 501 || resp.status == 404,
        "POST /v1/models should be rejected: {}",
        resp.status
    );
}

/// Request with previous_response_id but no input succeeds
/// (previous_response_id satisfies input requirement).
#[test]
fn previous_response_id_without_input() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    // First: create a response to chain from.
    let body1 = serde_json::json!({
        "model": &model,
        "input": "Remember: the answer is 42.",
        "max_output_tokens": 20,
    });
    let resp1 = post_json(ctx.addr(), "/v1/responses", &body1);
    assert_eq!(resp1.status, 200);
    let response_id = resp1.json()["id"].as_str().expect("id").to_string();

    // Second: chain with previous_response_id but no input field at all.
    let body2 = serde_json::json!({
        "model": &model,
        "previous_response_id": response_id,
        "max_output_tokens": 20,
    });
    let resp2 = post_json(ctx.addr(), "/v1/responses", &body2);
    assert_eq!(
        resp2.status, 200,
        "should succeed with just previous_response_id: {}",
        resp2.body
    );
}
