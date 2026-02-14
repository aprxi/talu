//! Request parameter passthrough and enforcement tests.

use crate::server::common::{get, model_config, post_json, require_model, ServerTestContext};

/// Helper: generate with custom parameters.
fn generate_with(
    ctx: &ServerTestContext,
    model: &str,
    extra: serde_json::Value,
) -> serde_json::Value {
    let mut body = serde_json::json!({
        "model": model,
        "input": "Hello",
        "max_output_tokens": 10,
    });
    if let Some(obj) = extra.as_object() {
        for (k, v) in obj {
            body[k.clone()] = v.clone();
        }
    }
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    resp.json()
}

/// Temperature is reflected in the response resource.
#[test]
fn temperature_passthrough() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let json = generate_with(&ctx, &model, serde_json::json!({"temperature": 0.7}));
    let temp = json["temperature"].as_f64().expect("temperature field");
    assert!(
        (temp - 0.7).abs() < 0.01,
        "temperature should be 0.7, got {temp}"
    );
}

/// top_p is reflected in the response resource.
#[test]
fn top_p_passthrough() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let json = generate_with(&ctx, &model, serde_json::json!({"top_p": 0.9}));
    let top_p = json["top_p"].as_f64().expect("top_p field");
    assert!(
        (top_p - 0.9).abs() < 0.01,
        "top_p should be 0.9, got {top_p}"
    );
}

/// max_output_tokens is echoed back in the response resource.
#[test]
fn max_output_tokens_echo() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let json = generate_with(&ctx, &model, serde_json::json!({"max_output_tokens": 5}));
    assert_eq!(
        json["max_output_tokens"].as_i64(),
        Some(5),
        "max_output_tokens should be echoed: {:?}",
        json["max_output_tokens"]
    );
}

/// max_output_tokens limits actual output length.
#[test]
fn max_output_tokens_limits_output() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    // Request with very few tokens.
    let json = generate_with(
        &ctx,
        &model,
        serde_json::json!({
            "max_output_tokens": 3,
            "input": "Write a very long story about dragons"
        }),
    );
    let output_tokens = json["usage"]["output_tokens"]
        .as_u64()
        .expect("output_tokens");
    // The model should generate at most max_output_tokens tokens (some slack for stop token).
    assert!(
        output_tokens <= 5,
        "output_tokens ({output_tokens}) should be limited to ~3"
    );
}

/// Default temperature (when not specified) is 0.0.
#[test]
fn default_temperature() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let json = generate_with(&ctx, &model, serde_json::json!({}));
    assert_eq!(
        json["temperature"].as_f64(),
        Some(0.0),
        "default temperature should be 0.0",
    );
}

/// Default top_p (when not specified) is 1.0.
#[test]
fn default_top_p() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let json = generate_with(&ctx, &model, serde_json::json!({}));
    assert_eq!(
        json["top_p"].as_f64(),
        Some(1.0),
        "default top_p should be 1.0",
    );
}

/// Bare /responses path (without /v1 prefix) works.
#[test]
fn bare_responses_path() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": model,
        "input": "Hello",
        "max_output_tokens": 5,
    });
    let resp = post_json(ctx.addr(), "/responses", &body);
    assert_eq!(
        resp.status, 200,
        "bare /responses should work: {}",
        resp.body
    );
    let json = resp.json();
    assert_eq!(json["object"].as_str(), Some("response"));
}

/// /models path (without /v1 prefix) works like /v1/models.
#[test]
fn bare_models_path() {
    let _ = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let v1 = get(ctx.addr(), "/v1/models").json();
    let bare = get(ctx.addr(), "/models").json();
    assert_eq!(v1["object"], bare["object"], "both should be list objects");
    let v1_data = v1["data"].as_array().expect("/v1/models data");
    let bare_data = bare["data"].as_array().expect("/models data");
    assert_eq!(v1_data.len(), bare_data.len(), "same number of models");
}

/// Streaming with bare /responses path works.
#[test]
fn bare_responses_streaming() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": model,
        "input": "Hi",
        "stream": true,
        "max_output_tokens": 10,
    });
    let resp = post_json(ctx.addr(), "/responses", &body);
    assert_eq!(resp.status, 200);
    let ct = resp.header("content-type").expect("Content-Type");
    assert!(ct.contains("text/event-stream"), "should be SSE: {ct}");
    assert!(resp.body.contains("data: {"), "should contain SSE events");
}
