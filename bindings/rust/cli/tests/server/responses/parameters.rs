//! `/v1/responses` request parameter passthrough and defaults.

use crate::server::common::{
    model_config, post_json, require_model, ServerConfig, ServerTestContext,
};

fn generate_with(
    ctx: &ServerTestContext,
    model: &str,
    extra: serde_json::Value,
) -> serde_json::Value {
    let mut body = serde_json::json!({
        "model": model,
        "input": "Hello",
        "max_output_tokens": 16
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
fn responses_sampling_bounds_round_trip() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let json = generate_with(
        &ctx,
        &model,
        serde_json::json!({
            "temperature": 2.0,
            "top_p": 1.0
        }),
    );
    assert_eq!(json["temperature"].as_f64(), Some(2.0));
    assert_eq!(json["top_p"].as_f64(), Some(1.0));
}

#[test]
fn responses_rejects_presence_penalty_until_core_supports_it() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": model,
        "input": "Hello",
        "max_output_tokens": 16,
        "presence_penalty": 0.4
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert!(
        resp.body.contains("presence_penalty"),
        "body: {}",
        resp.body
    );
}

#[test]
fn responses_rejects_frequency_penalty_until_core_supports_it() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": model,
        "input": "Hello",
        "max_output_tokens": 16,
        "frequency_penalty": 0.6
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert!(
        resp.body.contains("frequency_penalty"),
        "body: {}",
        resp.body
    );
}

#[test]
fn responses_max_output_tokens_echo() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let json = generate_with(&ctx, &model, serde_json::json!({"max_output_tokens": 20}));
    assert_eq!(json["max_output_tokens"].as_i64(), Some(20));
}

#[test]
fn responses_max_output_tokens_limits_output() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let json = generate_with(
        &ctx,
        &model,
        serde_json::json!({
            "max_output_tokens": 16,
            "input": "Write a very long story about dragons"
        }),
    );
    let output_tokens = json["usage"]["output_tokens"]
        .as_u64()
        .expect("output_tokens");
    assert!(output_tokens <= 18, "expected <=18, got {output_tokens}");
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
fn responses_truncation_auto_round_trips() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let json = generate_with(&ctx, &model, serde_json::json!({"truncation": "auto"}));
    assert_eq!(json["truncation"].as_str(), Some("auto"));
}

#[test]
fn responses_rejects_truncation_disabled_until_supported() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": model,
        "input": "Hello",
        "max_output_tokens": 16,
        "truncation": "disabled"
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert!(
        resp.body.contains("truncation.disabled"),
        "body: {}",
        resp.body
    );
}

#[test]
fn responses_rejects_top_logprobs_until_core_supports_it() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": model,
        "input": "Hello",
        "max_output_tokens": 16,
        "top_logprobs": 3
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert!(resp.body.contains("top_logprobs"), "body: {}", resp.body);
}

#[test]
fn responses_rejects_reasoning_configuration_until_core_supports_it() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": model,
        "input": "Hello",
        "max_output_tokens": 16,
        "reasoning": {
            "effort": "high",
            "summary": "concise"
        }
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert!(resp.body.contains("reasoning"), "body: {}", resp.body);
}

#[test]
fn responses_rejects_json_schema_text_format_until_core_supports_it() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": model,
        "input": "Return whether you can comply.",
        "max_output_tokens": 16,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "compliance_result",
                "schema": { "type": "object" }
            }
        }
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert!(
        resp.body.contains("text.format.json_schema"),
        "body: {}",
        resp.body
    );
}

#[test]
fn responses_rejects_json_object_text_format_per_openapi_request_schema() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let body = serde_json::json!({
        "input": "Hello",
        "max_output_tokens": 16,
        "text": {
            "format": {
                "type": "json_object"
            }
        }
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 400, "body: {}", resp.body);
}

#[test]
fn responses_accepts_text_format_type_text_shape() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let body = serde_json::json!({
        "input": "Hello",
        "max_output_tokens": 16,
        "text": {
            "format": {
                "type": "text"
            }
        }
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_ne!(resp.status, 400, "body: {}", resp.body);
}

#[test]
fn responses_accepts_null_text_format_shape() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let body = serde_json::json!({
        "input": "Hello",
        "max_output_tokens": 16,
        "text": {
            "format": null
        }
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_ne!(resp.status, 400, "body: {}", resp.body);
}

#[test]
fn responses_accepts_safety_identifier_shape() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let _json = generate_with(
        &ctx,
        &model,
        serde_json::json!({
            "safety_identifier": "req-123"
        }),
    );
}

#[test]
fn responses_rejects_unimplemented_spec_fields() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": model,
        "input": "hello",
        "max_output_tokens": 16,
        "parallel_tool_calls": true,
        "background": true,
        "service_tier": "default",
        "prompt_cache_key": "cache-key-1"
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 400, "body: {}", resp.body);
}
