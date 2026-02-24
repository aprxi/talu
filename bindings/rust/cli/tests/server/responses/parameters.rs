//! `/v1/responses` request parameter passthrough and defaults.

use crate::server::common::{model_config, post_json, require_model, ServerTestContext};

fn generate_with(
    ctx: &ServerTestContext,
    model: &str,
    extra: serde_json::Value,
) -> serde_json::Value {
    let mut body = serde_json::json!({
        "model": model,
        "input": "Hello",
        "max_output_tokens": 10
    });
    if let Some(obj) = extra.as_object() {
        for (k, v) in obj {
            body[k.as_str()] = v.clone();
        }
    }
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    resp.json()
}

#[test]
fn responses_temperature_passthrough() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let json = generate_with(&ctx, &model, serde_json::json!({"temperature": 0.7}));
    let temp = json["temperature"].as_f64().expect("temperature field");
    assert!((temp - 0.7).abs() < 0.01, "expected 0.7, got {temp}");
}

#[test]
fn responses_top_p_passthrough() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let json = generate_with(&ctx, &model, serde_json::json!({"top_p": 0.9}));
    let top_p = json["top_p"].as_f64().expect("top_p field");
    assert!((top_p - 0.9).abs() < 0.01, "expected 0.9, got {top_p}");
}

#[test]
fn responses_max_output_tokens_echo() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let json = generate_with(&ctx, &model, serde_json::json!({"max_output_tokens": 5}));
    assert_eq!(json["max_output_tokens"].as_i64(), Some(5));
}

#[test]
fn responses_max_output_tokens_limits_output() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
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
    assert!(output_tokens <= 5, "expected <=5, got {output_tokens}");
}

#[test]
fn responses_default_sampling_and_penalty_fields() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let json = generate_with(&ctx, &model, serde_json::json!({}));
    assert_eq!(json["temperature"].as_f64(), Some(0.0));
    assert_eq!(json["top_p"].as_f64(), Some(1.0));
    assert_eq!(json["presence_penalty"].as_f64(), Some(0.0));
    assert_eq!(json["frequency_penalty"].as_f64(), Some(0.0));
    assert_eq!(json["top_logprobs"].as_i64(), Some(0));
}

#[test]
fn responses_accepts_unsupported_spec_fields_without_400() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": model,
        "input": "hello",
        "max_output_tokens": 8,
        "parallel_tool_calls": true,
        "background": true,
        "service_tier": "default",
        "prompt_cache_key": "cache-key-1"
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_ne!(resp.status, 400, "body: {}", resp.body);
}
