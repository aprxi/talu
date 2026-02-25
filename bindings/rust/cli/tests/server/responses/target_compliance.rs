//! Target-state OpenResponses conformance matrix.
//!
//! This suite encodes the *desired* `/v1/responses` behavior for features
//! present in the OpenAPI surface. It should fail until all gaps are
//! implemented end-to-end.

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

#[test]
fn responses_target_conformance_matrix_has_no_gaps() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let mut failures: Vec<String> = Vec::new();

    let mut expect_success = |name: String, extra: serde_json::Value| {
        let body = with_base_request(extra);
        let resp = post_json(ctx.addr(), "/v1/responses", &body);
        if resp.status == 400 {
            let payload = resp.json();
            let message = payload
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .unwrap_or("");
            failures.push(format!(
                "{name}: status=400 invalid_request message={message}"
            ));
        }
    };

    // Sampling penalties.
    for value in [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0] {
        expect_success(
            format!("presence_penalty_{value}"),
            serde_json::json!({ "presence_penalty": value }),
        );
        expect_success(
            format!("frequency_penalty_{value}"),
            serde_json::json!({ "frequency_penalty": value }),
        );
    }

    // Logprobs includes and top_logprobs values.
    expect_success(
        "include_message_output_text_logprobs".to_string(),
        serde_json::json!({
            "include": ["message.output_text.logprobs"]
        }),
    );
    expect_success(
        "include_reasoning_encrypted_content".to_string(),
        serde_json::json!({
            "include": ["reasoning.encrypted_content"]
        }),
    );
    for top in 1..=20 {
        expect_success(
            format!("top_logprobs_{top}"),
            serde_json::json!({
                "top_logprobs": top
            }),
        );
    }

    // Reasoning config.
    for effort in ["none", "low", "medium", "high", "xhigh"] {
        expect_success(
            format!("reasoning_effort_{effort}"),
            serde_json::json!({
                "reasoning": { "effort": effort }
            }),
        );
    }
    for summary in ["auto", "concise", "detailed"] {
        expect_success(
            format!("reasoning_summary_{summary}"),
            serde_json::json!({
                "reasoning": { "summary": summary }
            }),
        );
    }
    expect_success(
        "reasoning_effort_high_summary_concise".to_string(),
        serde_json::json!({
            "reasoning": { "effort": "high", "summary": "concise" }
        }),
    );
    expect_success(
        "reasoning_effort_medium_summary_auto".to_string(),
        serde_json::json!({
            "reasoning": { "effort": "medium", "summary": "auto" }
        }),
    );
    expect_success(
        "reasoning_effort_low_summary_detailed".to_string(),
        serde_json::json!({
            "reasoning": { "effort": "low", "summary": "detailed" }
        }),
    );

    // Structured outputs / JSON schema.
    expect_success(
        "text_format_json_schema_strict_true".to_string(),
        serde_json::json!({
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "reply",
                    "schema": { "type": "object" },
                    "strict": true
                }
            }
        }),
    );
    expect_success(
        "text_format_json_schema_strict_false".to_string(),
        serde_json::json!({
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "reply",
                    "schema": { "type": "object" },
                    "strict": false
                }
            }
        }),
    );
    expect_success(
        "text_format_text".to_string(),
        serde_json::json!({
            "text": {
                "format": {
                    "type": "text"
                }
            }
        }),
    );
    expect_success(
        "text_format_null".to_string(),
        serde_json::json!({
            "text": {
                "format": null
            }
        }),
    );

    // Request fields currently treated as unimplemented in server validation.
    expect_success(
        "stream_options_include_obfuscation".to_string(),
        serde_json::json!({
            "stream_options": { "include_obfuscation": true }
        }),
    );
    expect_success(
        "truncation_auto".to_string(),
        serde_json::json!({ "truncation": "auto" }),
    );
    expect_success(
        "truncation_disabled".to_string(),
        serde_json::json!({ "truncation": "disabled" }),
    );
    expect_success(
        "safety_identifier_req_123".to_string(),
        serde_json::json!({ "safety_identifier": "req-123" }),
    );
    for max_tool_calls in [1, 2, 4] {
        expect_success(
            format!("max_tool_calls_{max_tool_calls}"),
            serde_json::json!({ "max_tool_calls": max_tool_calls }),
        );
    }
    for flag in [false, true] {
        expect_success(
            format!("parallel_tool_calls_{flag}"),
            serde_json::json!({ "parallel_tool_calls": flag }),
        );
        expect_success(
            format!("background_{flag}"),
            serde_json::json!({ "background": flag }),
        );
    }
    for tier in ["default", "auto", "flex", "priority"] {
        expect_success(
            format!("service_tier_{tier}"),
            serde_json::json!({ "service_tier": tier }),
        );
    }
    expect_success(
        "prompt_cache_key_nonempty".to_string(),
        serde_json::json!({
            "prompt_cache_key": "responses-target-cache-key"
        }),
    );

    assert!(
        failures.is_empty(),
        "OpenResponses target conformance gaps ({}):\n{}",
        failures.len(),
        failures.join("\n")
    );
}
