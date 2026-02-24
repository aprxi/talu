//! `/v1/responses` error-path tests.

use crate::server::common::{
    model_config, post_json, require_model, send_request, ServerConfig, ServerTestContext,
};

#[test]
fn responses_model_not_available_returns_500_server_error() {
    let _ = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": "nonexistent-model-that-does-not-exist",
        "input": "Hello",
        "max_output_tokens": 5
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 500, "body: {}", resp.body);
    let json = resp.json();
    assert_eq!(json["error"]["type"].as_str(), Some("server_error"));
    assert_eq!(json["error"]["code"].as_str(), Some("inference_error"));
}

#[test]
fn responses_streaming_model_not_available_returns_json_error() {
    let _ = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": "nonexistent-model-that-does-not-exist",
        "input": "Hello",
        "stream": true,
        "max_output_tokens": 5
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 500, "body: {}", resp.body);
    let ct = resp.header("content-type").unwrap_or_default();
    assert!(
        ct.contains("application/json"),
        "strict responses errors should be JSON, got Content-Type {ct}"
    );
    let json = resp.json();
    assert!(json["error"]["code"].is_string());
}

#[test]
fn responses_empty_body_returns_400() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/responses",
        &[("Content-Type", "application/json")],
        Some(""),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    let json = resp.json();
    assert_eq!(
        json["error"]["type"].as_str(),
        Some("invalid_request_error")
    );
}

#[test]
fn responses_no_model_uses_default_configured_model() {
    let _ = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "input": "Hello",
        "max_output_tokens": 5
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(
        resp.status, 200,
        "request without model field should use configured default model: {}",
        resp.body
    );
    assert!(resp.json()["model"].as_str().is_some());
}

#[test]
fn responses_put_and_delete_not_supported() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let put = send_request(ctx.addr(), "PUT", "/v1/responses", &[], None);
    assert!(
        put.status == 404 || put.status == 405 || put.status == 501,
        "PUT /v1/responses should be rejected, got {}",
        put.status
    );

    let del = send_request(ctx.addr(), "DELETE", "/v1/responses", &[], None);
    assert!(
        del.status == 404 || del.status == 405 || del.status == 501,
        "DELETE /v1/responses should be rejected, got {}",
        del.status
    );
}
