//! Error response tests over real HTTP.

use crate::server::common::{
    model_config, model_path, post_json, send_request, ServerConfig, ServerTestContext,
};

/// Invalid JSON body returns 400 with error object.
#[test]
fn invalid_json_400() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/responses",
        &[("Content-Type", "application/json")],
        Some("{not json"),
    );
    assert_eq!(resp.status, 400);
    let json = resp.json();
    let error = &json["error"];
    assert_eq!(error["code"].as_str(), Some("invalid_request"));
    assert!(error["message"].as_str().is_some());
}

/// Error responses have application/json Content-Type.
#[test]
fn error_content_type() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/responses",
        &[("Content-Type", "application/json")],
        Some("invalid"),
    );
    assert!(resp.status >= 400);
    let ct = resp
        .header("content-type")
        .expect("should have Content-Type");
    assert!(ct.contains("application/json"), "error Content-Type: {ct}");
}

/// Missing input field returns 400.
#[test]
fn missing_input_400() {
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({"model": "test"});
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 400, "body: {}", resp.body);
}

/// Unknown path returns 404.
#[test]
fn unknown_path_404() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let resp = send_request(ctx.addr(), "GET", "/v1/nonexistent", &[], None);
    assert_eq!(resp.status, 404);
}

/// GET on /v1/responses (only POST implemented) returns 501.
#[test]
fn unimplemented_method_501() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let resp = send_request(ctx.addr(), "GET", "/v1/responses", &[], None);
    assert_eq!(resp.status, 501);
    let json = resp.json();
    assert_eq!(json["error"]["code"].as_str(), Some("not_implemented"));
}

/// Requesting an unavailable model returns 500 with inference_error.
#[test]
fn model_not_available_500() {
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

/// Error response always has the standard error object shape.
#[test]
fn error_response_shape() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/responses",
        &[("Content-Type", "application/json")],
        Some("{}"),
    );
    assert!(resp.status >= 400);
    let json = resp.json();
    assert!(json.get("error").is_some(), "should have error key");
    let error = &json["error"];
    assert!(error["code"].is_string(), "error.code should be string");
    assert!(
        error["message"].is_string(),
        "error.message should be string"
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
    let ctx = ServerTestContext::new(model_config());
    // First: create a response to chain from.
    let body1 = serde_json::json!({
        "model": model_path(),
        "input": "Remember: the answer is 42.",
        "max_output_tokens": 20,
    });
    let resp1 = post_json(ctx.addr(), "/v1/responses", &body1);
    assert_eq!(resp1.status, 200);
    let response_id = resp1.json()["id"].as_str().expect("id").to_string();

    // Second: chain with previous_response_id but no input field at all.
    let body2 = serde_json::json!({
        "model": model_path(),
        "previous_response_id": response_id,
        "max_output_tokens": 20,
    });
    let resp2 = post_json(ctx.addr(), "/v1/responses", &body2);
    // The handler validates: input_value.is_none() && previous_response_id.is_none()
    // Since previous_response_id is present, this should not return 400.
    assert_eq!(
        resp2.status, 200,
        "should succeed with just previous_response_id: {}",
        resp2.body
    );
}
