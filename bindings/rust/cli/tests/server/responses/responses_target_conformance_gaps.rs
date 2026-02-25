//! Granular target-state conformance gaps for `/v1/responses`.
//!
//! Each test encodes desired OpenResponses behavior and currently fails
//! when the server returns `400 invalid_request` for spec-surface fields
//! that are accepted but not implemented yet.

use crate::server::common::{post_json, ServerConfig, ServerTestContext};

fn with_base_request(extra: serde_json::Value) -> serde_json::Value {
    let mut body = serde_json::json!({
        "input": "hello",
        "max_output_tokens": 16
    });
    if let Some(obj) = extra.as_object() {
        for (k, v) in obj {
            body[k.as_str()] = v.clone();
        }
    }
    body
}

fn assert_target_accepts(extra: serde_json::Value) {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let body = with_base_request(extra);
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_ne!(
        resp.status, 400,
        "target-state expected non-400. body: {}",
        resp.body
    );
}

macro_rules! target_case {
    ($name:ident, $extra:expr) => {
        #[test]
        fn $name() {
            assert_target_accepts($extra);
        }
    };
}

// include[] and top_logprobs

target_case!(
    include_message_output_text_logprobs,
    serde_json::json!({ "include": ["message.output_text.logprobs"] })
);
target_case!(
    include_reasoning_encrypted_content,
    serde_json::json!({ "include": ["reasoning.encrypted_content"] })
);
target_case!(
    include_both_logprobs_and_encrypted_reasoning,
    serde_json::json!({
        "include": [
            "message.output_text.logprobs",
            "reasoning.encrypted_content"
        ]
    })
);
target_case!(
    include_logprobs_with_duplicates,
    serde_json::json!({
        "include": [
            "message.output_text.logprobs",
            "message.output_text.logprobs"
        ]
    })
);

target_case!(
    top_logprobs_min_zero,
    serde_json::json!({ "top_logprobs": 0 })
);
target_case!(
    top_logprobs_mid_ten,
    serde_json::json!({ "top_logprobs": 10 })
);
target_case!(
    top_logprobs_max_twenty,
    serde_json::json!({ "top_logprobs": 20 })
);
target_case!(
    top_logprobs_max_with_include,
    serde_json::json!({
        "include": ["message.output_text.logprobs"],
        "top_logprobs": 20
    })
);

// sampling penalties

target_case!(
    presence_penalty_min,
    serde_json::json!({ "presence_penalty": -2.0 })
);
target_case!(
    presence_penalty_negative_fraction,
    serde_json::json!({ "presence_penalty": -0.25 })
);
target_case!(
    presence_penalty_zero,
    serde_json::json!({ "presence_penalty": 0.0 })
);
target_case!(
    presence_penalty_positive_fraction,
    serde_json::json!({ "presence_penalty": 0.25 })
);
target_case!(
    presence_penalty_max,
    serde_json::json!({ "presence_penalty": 2.0 })
);

target_case!(
    frequency_penalty_min,
    serde_json::json!({ "frequency_penalty": -2.0 })
);
target_case!(
    frequency_penalty_negative_fraction,
    serde_json::json!({ "frequency_penalty": -0.25 })
);
target_case!(
    frequency_penalty_zero,
    serde_json::json!({ "frequency_penalty": 0.0 })
);
target_case!(
    frequency_penalty_positive_fraction,
    serde_json::json!({ "frequency_penalty": 0.25 })
);
target_case!(
    frequency_penalty_max,
    serde_json::json!({ "frequency_penalty": 2.0 })
);

target_case!(
    penalties_opposite_extremes,
    serde_json::json!({
        "presence_penalty": -2.0,
        "frequency_penalty": 2.0
    })
);
target_case!(
    penalties_centered_nonzero,
    serde_json::json!({
        "presence_penalty": 1.0,
        "frequency_penalty": -1.0
    })
);
target_case!(
    penalties_both_max,
    serde_json::json!({
        "presence_penalty": 2.0,
        "frequency_penalty": 2.0
    })
);

// reasoning config

target_case!(
    reasoning_effort_none,
    serde_json::json!({ "reasoning": { "effort": "none" } })
);
target_case!(
    reasoning_effort_low,
    serde_json::json!({ "reasoning": { "effort": "low" } })
);
target_case!(
    reasoning_effort_medium,
    serde_json::json!({ "reasoning": { "effort": "medium" } })
);
target_case!(
    reasoning_effort_high,
    serde_json::json!({ "reasoning": { "effort": "high" } })
);
target_case!(
    reasoning_effort_xhigh,
    serde_json::json!({ "reasoning": { "effort": "xhigh" } })
);

target_case!(
    reasoning_summary_auto,
    serde_json::json!({ "reasoning": { "summary": "auto" } })
);
target_case!(
    reasoning_summary_concise,
    serde_json::json!({ "reasoning": { "summary": "concise" } })
);
target_case!(
    reasoning_summary_detailed,
    serde_json::json!({ "reasoning": { "summary": "detailed" } })
);

target_case!(
    reasoning_high_concise,
    serde_json::json!({ "reasoning": { "effort": "high", "summary": "concise" } })
);
target_case!(
    reasoning_none_auto,
    serde_json::json!({ "reasoning": { "effort": "none", "summary": "auto" } })
);
target_case!(
    reasoning_xhigh_detailed,
    serde_json::json!({ "reasoning": { "effort": "xhigh", "summary": "detailed" } })
);

// structured outputs

target_case!(
    text_format_json_schema_minimal,
    serde_json::json!({
        "text": {
            "format": {
                "type": "json_schema",
                "name": "reply",
                "schema": { "type": "object" }
            }
        }
    })
);
target_case!(
    text_format_json_schema_strict_true,
    serde_json::json!({
        "text": {
            "format": {
                "type": "json_schema",
                "name": "reply",
                "schema": { "type": "object" },
                "strict": true
            }
        }
    })
);
target_case!(
    text_format_json_schema_strict_false,
    serde_json::json!({
        "text": {
            "format": {
                "type": "json_schema",
                "name": "reply",
                "schema": { "type": "object" },
                "strict": false
            }
        }
    })
);
target_case!(
    text_format_json_schema_with_description,
    serde_json::json!({
        "text": {
            "format": {
                "type": "json_schema",
                "name": "reply",
                "description": "response format",
                "schema": { "type": "object" }
            }
        }
    })
);
target_case!(
    text_format_json_schema_array_type,
    serde_json::json!({
        "text": {
            "format": {
                "type": "json_schema",
                "name": "arr",
                "schema": { "type": "array", "items": { "type": "string" } }
            }
        }
    })
);
target_case!(
    text_format_json_schema_nested_object,
    serde_json::json!({
        "text": {
            "format": {
                "type": "json_schema",
                "name": "nested",
                "schema": {
                    "type": "object",
                    "properties": {
                        "meta": {
                            "type": "object",
                            "properties": { "ok": { "type": "boolean" } }
                        }
                    }
                }
            }
        }
    })
);
target_case!(
    text_format_json_schema_enum_field,
    serde_json::json!({
        "text": {
            "format": {
                "type": "json_schema",
                "name": "enumish",
                "schema": {
                    "type": "object",
                    "properties": {
                        "status": { "type": "string", "enum": ["ok", "error"] }
                    }
                }
            }
        }
    })
);
target_case!(
    text_format_text,
    serde_json::json!({
        "text": {
            "format": {
                "type": "text"
            }
        }
    })
);
target_case!(
    text_format_null,
    serde_json::json!({
        "text": {
            "format": null
        }
    })
);

target_case!(
    text_verbosity_low,
    serde_json::json!({ "text": { "verbosity": "low" } })
);
target_case!(
    text_verbosity_medium,
    serde_json::json!({ "text": { "verbosity": "medium" } })
);
target_case!(
    text_verbosity_high,
    serde_json::json!({ "text": { "verbosity": "high" } })
);

target_case!(
    text_schema_with_high_verbosity,
    serde_json::json!({
        "text": {
            "format": {
                "type": "json_schema",
                "name": "reply",
                "schema": { "type": "object" }
            },
            "verbosity": "high"
        }
    })
);

// accepted-but-unimplemented request fields

target_case!(
    stream_options_include_obfuscation_true,
    serde_json::json!({ "stream_options": { "include_obfuscation": true } })
);
target_case!(
    stream_options_include_obfuscation_false,
    serde_json::json!({ "stream_options": { "include_obfuscation": false } })
);

target_case!(truncation_auto, serde_json::json!({ "truncation": "auto" }));
target_case!(
    truncation_disabled,
    serde_json::json!({ "truncation": "disabled" })
);

target_case!(
    safety_identifier_req_123,
    serde_json::json!({ "safety_identifier": "req-123" })
);
target_case!(
    safety_identifier_64_chars,
    serde_json::json!({ "safety_identifier": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" })
);

target_case!(max_tool_calls_1, serde_json::json!({ "max_tool_calls": 1 }));
target_case!(max_tool_calls_8, serde_json::json!({ "max_tool_calls": 8 }));

target_case!(
    parallel_tool_calls_false,
    serde_json::json!({ "parallel_tool_calls": false })
);
target_case!(
    parallel_tool_calls_true,
    serde_json::json!({ "parallel_tool_calls": true })
);

target_case!(background_false, serde_json::json!({ "background": false }));
target_case!(background_true, serde_json::json!({ "background": true }));

target_case!(
    service_tier_default,
    serde_json::json!({ "service_tier": "default" })
);
target_case!(
    service_tier_auto,
    serde_json::json!({ "service_tier": "auto" })
);
target_case!(
    service_tier_flex,
    serde_json::json!({ "service_tier": "flex" })
);
target_case!(
    service_tier_priority,
    serde_json::json!({ "service_tier": "priority" })
);

target_case!(
    prompt_cache_key_short,
    serde_json::json!({ "prompt_cache_key": "k1" })
);
target_case!(
    prompt_cache_key_long,
    serde_json::json!({
        "prompt_cache_key": "responses-target-cache-key-abcdefghijklmnopqrstuvwxyz"
    })
);

// multi-feature interaction scenarios

target_case!(
    combined_logprobs_reasoning,
    serde_json::json!({
        "include": ["message.output_text.logprobs", "reasoning.encrypted_content"],
        "top_logprobs": 5,
        "reasoning": { "effort": "high", "summary": "concise" }
    })
);
target_case!(
    combined_penalties_with_reasoning,
    serde_json::json!({
        "presence_penalty": 1.0,
        "frequency_penalty": -0.5,
        "reasoning": { "effort": "medium" }
    })
);
target_case!(
    combined_json_schema_with_penalties,
    serde_json::json!({
        "text": {
            "format": {
                "type": "json_schema",
                "name": "reply",
                "schema": { "type": "object" }
            }
        },
        "presence_penalty": 0.5,
        "frequency_penalty": 0.5
    })
);
target_case!(
    combined_service_background_prompt_cache,
    serde_json::json!({
        "service_tier": "auto",
        "background": true,
        "prompt_cache_key": "combo-key"
    })
);
target_case!(
    combined_parallel_and_max_tool_calls,
    serde_json::json!({
        "parallel_tool_calls": true,
        "max_tool_calls": 4
    })
);
target_case!(
    combined_stream_options_and_truncation,
    serde_json::json!({
        "stream_options": { "include_obfuscation": true },
        "truncation": "disabled"
    })
);
target_case!(
    combined_top_logprobs_and_text_format_text,
    serde_json::json!({
        "top_logprobs": 3,
        "include": ["message.output_text.logprobs"],
        "text": { "format": { "type": "text" } }
    })
);
target_case!(
    combined_reasoning_and_verbosity,
    serde_json::json!({
        "reasoning": { "effort": "xhigh", "summary": "detailed" },
        "text": { "verbosity": "high" }
    })
);
target_case!(
    combined_everything_supported_shape,
    serde_json::json!({
        "include": ["message.output_text.logprobs", "reasoning.encrypted_content"],
        "top_logprobs": 20,
        "presence_penalty": -2.0,
        "frequency_penalty": 2.0,
        "reasoning": { "effort": "none", "summary": "auto" },
        "text": { "format": { "type": "text" }, "verbosity": "medium" },
        "stream_options": { "include_obfuscation": false },
        "truncation": "auto",
        "safety_identifier": "req-combo",
        "max_tool_calls": 1,
        "parallel_tool_calls": false,
        "background": false,
        "service_tier": "default",
        "prompt_cache_key": "cache-combo"
    })
);
