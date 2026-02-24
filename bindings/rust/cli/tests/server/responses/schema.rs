//! `/v1/responses` response schema completeness tests.

use crate::server::common::{model_config, post_json, require_model, ServerTestContext};

fn generate(ctx: &ServerTestContext, model: &str) -> serde_json::Value {
    let body = serde_json::json!({
        "model": model,
        "input": "Hello",
        "max_output_tokens": 10
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    resp.json()
}

#[test]
fn responses_non_streaming_has_all_required_fields() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let json = generate(&ctx, &model);

    let required_fields = [
        "id",
        "object",
        "created_at",
        "completed_at",
        "status",
        "incomplete_details",
        "model",
        "previous_response_id",
        "instructions",
        "output",
        "error",
        "tools",
        "tool_choice",
        "truncation",
        "parallel_tool_calls",
        "text",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "top_logprobs",
        "temperature",
        "reasoning",
        "usage",
        "max_output_tokens",
        "max_tool_calls",
        "store",
        "background",
        "service_tier",
        "metadata",
        "safety_identifier",
        "prompt_cache_key",
    ];

    for field in &required_fields {
        assert!(
            json.get(field).is_some(),
            "response missing required field: {field}"
        );
    }
}

#[test]
fn responses_field_types_match_contract() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let json = generate(&ctx, &model);

    assert!(json["id"].is_string());
    assert!(json["object"].is_string());
    assert!(json["status"].is_string());
    assert!(json["model"].is_string());
    assert!(json["truncation"].is_string());
    assert!(json["service_tier"].is_string());

    assert!(json["created_at"].is_number());
    assert!(json["temperature"].is_number());
    assert!(json["top_p"].is_number());
    assert!(json["presence_penalty"].is_number());
    assert!(json["frequency_penalty"].is_number());
    assert!(json["top_logprobs"].is_number());

    assert!(json["parallel_tool_calls"].is_boolean());
    assert!(json["store"].is_boolean());
    assert!(json["background"].is_boolean());

    assert!(json["output"].is_array());
    assert!(json["tools"].is_array());

    assert!(json["usage"].is_object());
    assert!(json["text"].is_object());
    assert!(json["reasoning"].is_object());
    assert!(json["metadata"].is_object());
}

#[test]
fn responses_usage_and_default_field_values() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let json = generate(&ctx, &model);
    let usage = &json["usage"];

    assert!(usage["input_tokens"].is_number());
    assert!(usage["output_tokens"].is_number());
    assert!(usage["total_tokens"].is_number());
    assert!(usage["input_tokens_details"]["cached_tokens"].is_number());
    assert!(usage["output_tokens_details"]["reasoning_tokens"].is_number());

    assert_eq!(json["object"].as_str(), Some("response"));
    assert_eq!(json["truncation"].as_str(), Some("auto"));
    assert_eq!(json["parallel_tool_calls"].as_bool(), Some(false));
    assert_eq!(json["presence_penalty"].as_f64(), Some(0.0));
    assert_eq!(json["frequency_penalty"].as_f64(), Some(0.0));
    assert_eq!(json["top_logprobs"].as_u64(), Some(0));
    assert_eq!(json["store"].as_bool(), Some(false));
    assert_eq!(json["background"].as_bool(), Some(false));
    assert_eq!(json["service_tier"].as_str(), Some("default"));
    assert_eq!(json["text"]["format"]["type"].as_str(), Some("text"));
    assert!(json["reasoning"].get("effort").is_some());
    assert!(json["reasoning"].get("summary").is_some());
}

#[test]
fn responses_default_nullable_fields_are_null() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let json = generate(&ctx, &model);

    assert!(json["incomplete_details"].is_null());
    assert!(json["error"].is_null());
    assert!(json["instructions"].is_null());
    assert!(json["max_tool_calls"].is_null());
    assert!(json["safety_identifier"].is_null());
    assert!(json["prompt_cache_key"].is_null());
    assert!(json["previous_response_id"].is_null());
}

#[test]
fn responses_completed_at_is_after_created_at() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let json = generate(&ctx, &model);
    let created = json["created_at"].as_f64().expect("created_at");
    let completed = json["completed_at"].as_f64().expect("completed_at");
    assert!(
        completed >= created,
        "completed_at ({completed}) should be >= created_at ({created})"
    );
}
