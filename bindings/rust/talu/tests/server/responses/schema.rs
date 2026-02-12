//! Response schema completeness tests.
//!
//! Validates that the response resource contains all fields required by the
//! OpenAPI spec, and that their types are correct.

use crate::server::common::{get, model_config, model_path, post_json, ServerTestContext};

/// Helper: make a non-streaming request and return the parsed JSON response.
fn generate(ctx: &ServerTestContext) -> serde_json::Value {
    let body = serde_json::json!({
        "model": model_path(),
        "input": "Hello",
        "max_output_tokens": 10,
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    resp.json()
}

/// All 31 required response resource fields are present.
#[test]
fn response_has_all_required_fields() {
    let ctx = ServerTestContext::new(model_config());
    let json = generate(&ctx);

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
            "response missing required field: {field}",
        );
    }
}

/// Field types match the OpenAPI schema.
#[test]
fn response_field_types() {
    let ctx = ServerTestContext::new(model_config());
    let json = generate(&ctx);

    // String fields
    assert!(json["id"].is_string(), "id should be string");
    assert!(json["object"].is_string(), "object should be string");
    assert!(json["status"].is_string(), "status should be string");
    assert!(json["model"].is_string(), "model should be string");
    assert!(
        json["truncation"].is_string(),
        "truncation should be string"
    );
    assert!(
        json["service_tier"].is_string(),
        "service_tier should be string"
    );

    // Number fields
    assert!(
        json["created_at"].is_number(),
        "created_at should be number"
    );
    assert!(
        json["temperature"].is_number(),
        "temperature should be number"
    );
    assert!(json["top_p"].is_number(), "top_p should be number");
    assert!(
        json["presence_penalty"].is_number(),
        "presence_penalty should be number"
    );
    assert!(
        json["frequency_penalty"].is_number(),
        "frequency_penalty should be number"
    );
    assert!(
        json["top_logprobs"].is_number(),
        "top_logprobs should be number"
    );

    // Boolean fields
    assert!(
        json["parallel_tool_calls"].is_boolean(),
        "parallel_tool_calls should be boolean"
    );
    assert!(json["store"].is_boolean(), "store should be boolean");
    assert!(
        json["background"].is_boolean(),
        "background should be boolean"
    );

    // Array fields
    assert!(json["output"].is_array(), "output should be array");
    assert!(json["tools"].is_array(), "tools should be array");

    // Object fields
    assert!(json["usage"].is_object(), "usage should be object");
    assert!(json["text"].is_object(), "text should be object");
    assert!(json["reasoning"].is_object(), "reasoning should be object");
    assert!(json["metadata"].is_object(), "metadata should be object");
}

/// Usage object has all required sub-fields including details.
#[test]
fn usage_sub_fields() {
    let ctx = ServerTestContext::new(model_config());
    let json = generate(&ctx);
    let usage = &json["usage"];

    assert!(usage["input_tokens"].is_number(), "usage.input_tokens");
    assert!(usage["output_tokens"].is_number(), "usage.output_tokens");
    assert!(usage["total_tokens"].is_number(), "usage.total_tokens");

    // Details sub-objects
    let input_details = &usage["input_tokens_details"];
    assert!(
        input_details.is_object(),
        "usage.input_tokens_details should be object"
    );
    assert!(input_details["cached_tokens"].is_number(), "cached_tokens");

    let output_details = &usage["output_tokens_details"];
    assert!(
        output_details.is_object(),
        "usage.output_tokens_details should be object"
    );
    assert!(
        output_details["reasoning_tokens"].is_number(),
        "reasoning_tokens"
    );
}

/// The text field has the expected format sub-object.
#[test]
fn text_format_field() {
    let ctx = ServerTestContext::new(model_config());
    let json = generate(&ctx);
    let text = &json["text"];
    assert!(text["format"].is_object(), "text.format should be object");
    assert_eq!(text["format"]["type"].as_str(), Some("text"));
}

/// The reasoning field has effort and summary keys.
#[test]
fn reasoning_field() {
    let ctx = ServerTestContext::new(model_config());
    let json = generate(&ctx);
    let reasoning = &json["reasoning"];
    assert!(
        reasoning.get("effort").is_some(),
        "reasoning should have effort"
    );
    assert!(
        reasoning.get("summary").is_some(),
        "reasoning should have summary"
    );
}

/// Default values for hardcoded fields.
#[test]
fn default_field_values() {
    let ctx = ServerTestContext::new(model_config());
    let json = generate(&ctx);

    assert_eq!(json["object"].as_str(), Some("response"));
    assert_eq!(json["truncation"].as_str(), Some("auto"));
    assert_eq!(json["parallel_tool_calls"].as_bool(), Some(false));
    assert_eq!(json["presence_penalty"].as_f64(), Some(0.0));
    assert_eq!(json["frequency_penalty"].as_f64(), Some(0.0));
    assert_eq!(json["top_logprobs"].as_u64(), Some(0));
    assert_eq!(json["store"].as_bool(), Some(false));
    assert_eq!(json["background"].as_bool(), Some(false));
    assert_eq!(json["service_tier"].as_str(), Some("default"));

    // Null fields
    assert!(
        json["incomplete_details"].is_null(),
        "incomplete_details should be null"
    );
    assert!(json["error"].is_null(), "error should be null");
    assert!(
        json["instructions"].is_null(),
        "instructions should be null"
    );
    assert!(
        json["max_tool_calls"].is_null(),
        "max_tool_calls should be null"
    );
    assert!(
        json["safety_identifier"].is_null(),
        "safety_identifier should be null"
    );
    assert!(
        json["prompt_cache_key"].is_null(),
        "prompt_cache_key should be null"
    );
}

/// completed_at is present and >= created_at.
#[test]
fn completed_at_after_created_at() {
    let ctx = ServerTestContext::new(model_config());
    let json = generate(&ctx);
    let created = json["created_at"].as_f64().expect("created_at");
    let completed = json["completed_at"].as_f64().expect("completed_at");
    assert!(
        completed >= created,
        "completed_at ({completed}) should be >= created_at ({created})"
    );
}

/// Models list items have all required fields.
#[test]
fn model_object_schema() {
    let ctx = ServerTestContext::new(model_config());
    let resp = get(ctx.addr(), "/v1/models");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert!(!data.is_empty());
    let model = &data[0];
    assert!(model["id"].is_string(), "model.id");
    assert!(model["object"].is_string(), "model.object");
    assert!(model["created"].is_number(), "model.created");
    assert!(model["owned_by"].is_string(), "model.owned_by");
}

/// GET /openapi.json returns a valid OpenAPI spec.
#[test]
fn openapi_json_endpoint() {
    let ctx = ServerTestContext::new(model_config());
    let resp = get(ctx.addr(), "/openapi.json");
    assert_eq!(resp.status, 200);
    let ct = resp.header("content-type").expect("Content-Type");
    assert!(ct.contains("application/json"), "Content-Type: {ct}");
    let json = resp.json();
    assert!(json.is_object(), "should be JSON object");
    assert!(json.get("openapi").is_some(), "should have openapi version");
    assert!(json.get("paths").is_some(), "should have paths");
}
