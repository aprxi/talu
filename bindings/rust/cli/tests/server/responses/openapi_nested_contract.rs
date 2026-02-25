//! Nested OpenAPI contract checks for `/v1/responses`.
//!
//! These cases focus on nested polymorphic request shapes (tools, tool_choice,
//! and ItemParam/content variants) rather than top-level scalar fields.

use super::openapi_schema::OpenApiSchemaValidator;
use crate::server::common::{post_json, ServerConfig, ServerTestContext};

#[derive(Clone, Copy)]
enum Expectation {
    MustBe400,
    MustNotBe400,
}

struct NestedCase {
    name: &'static str,
    extra: serde_json::Value,
    schema_valid: bool,
    expect: Expectation,
}

fn with_base_request(extra: &serde_json::Value) -> serde_json::Value {
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

fn nested_cases() -> Vec<NestedCase> {
    let long_name_65 = "a".repeat(65);
    let mut allowed_tools_129 = Vec::new();
    for i in 0..129 {
        allowed_tools_129.push(serde_json::json!({
            "type": "function",
            "name": format!("tool_{i}")
        }));
    }

    vec![
        NestedCase {
            name: "tools_valid_minimal_function",
            extra: serde_json::json!({
                "tools": [
                    { "type": "function", "name": "get_weather" }
                ]
            }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
        },
        NestedCase {
            name: "tools_valid_strict_with_parameters_object",
            extra: serde_json::json!({
                "tools": [
                    {
                        "type": "function",
                        "name": "search_docs",
                        "strict": true,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": { "type": "string" }
                            }
                        }
                    }
                ]
            }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
        },
        NestedCase {
            name: "tools_invalid_empty_name",
            extra: serde_json::json!({
                "tools": [
                    { "type": "function", "name": "" }
                ]
            }),
            schema_valid: false,
            expect: Expectation::MustBe400,
        },
        NestedCase {
            name: "tools_invalid_name_too_long",
            extra: serde_json::json!({
                "tools": [
                    { "type": "function", "name": long_name_65 }
                ]
            }),
            schema_valid: false,
            expect: Expectation::MustBe400,
        },
        NestedCase {
            name: "tools_invalid_name_pattern",
            extra: serde_json::json!({
                "tools": [
                    { "type": "function", "name": "bad name" }
                ]
            }),
            schema_valid: false,
            expect: Expectation::MustBe400,
        },
        NestedCase {
            name: "tools_invalid_type",
            extra: serde_json::json!({
                "tools": [
                    { "type": "web_search", "name": "search" }
                ]
            }),
            schema_valid: false,
            expect: Expectation::MustBe400,
        },
        NestedCase {
            name: "tools_invalid_parameters_non_object",
            extra: serde_json::json!({
                "tools": [
                    { "type": "function", "name": "lookup", "parameters": 7 }
                ]
            }),
            schema_valid: false,
            expect: Expectation::MustBe400,
        },
        NestedCase {
            name: "tool_choice_specific_function_valid",
            extra: serde_json::json!({
                "tools": [{ "type": "function", "name": "lookup" }],
                "tool_choice": { "type": "function", "name": "lookup" }
            }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
        },
        NestedCase {
            name: "tool_choice_specific_function_missing_name",
            extra: serde_json::json!({
                "tool_choice": { "type": "function" }
            }),
            schema_valid: false,
            expect: Expectation::MustBe400,
        },
        NestedCase {
            name: "tool_choice_specific_function_invalid_name_pattern",
            extra: serde_json::json!({
                "tool_choice": { "type": "function", "name": "bad name" }
            }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
        },
        NestedCase {
            name: "tool_choice_allowed_tools_valid_mode_required",
            extra: serde_json::json!({
                "tool_choice": {
                    "type": "allowed_tools",
                    "tools": [{ "type": "function", "name": "lookup" }],
                    "mode": "required"
                }
            }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
        },
        NestedCase {
            name: "tool_choice_allowed_tools_invalid_empty_tools",
            extra: serde_json::json!({
                "tool_choice": {
                    "type": "allowed_tools",
                    "tools": []
                }
            }),
            schema_valid: false,
            expect: Expectation::MustBe400,
        },
        NestedCase {
            name: "tool_choice_allowed_tools_invalid_too_many_tools",
            extra: serde_json::json!({
                "tool_choice": {
                    "type": "allowed_tools",
                    "tools": allowed_tools_129
                }
            }),
            schema_valid: false,
            expect: Expectation::MustBe400,
        },
        NestedCase {
            name: "tool_choice_allowed_tools_invalid_mode",
            extra: serde_json::json!({
                "tool_choice": {
                    "type": "allowed_tools",
                    "tools": [{ "type": "function", "name": "lookup" }],
                    "mode": "sometimes"
                }
            }),
            schema_valid: false,
            expect: Expectation::MustBe400,
        },
        NestedCase {
            name: "input_function_call_output_with_video_valid",
            extra: serde_json::json!({
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_123",
                        "output": [
                            { "type": "input_video", "video_url": "https://example.com/v.mp4" }
                        ]
                    }
                ]
            }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
        },
        NestedCase {
            name: "input_function_call_empty_call_id_invalid",
            extra: serde_json::json!({
                "input": [
                    {
                        "type": "function_call",
                        "call_id": "",
                        "name": "lookup",
                        "arguments": "{}"
                    }
                ]
            }),
            schema_valid: false,
            expect: Expectation::MustBe400,
        },
        NestedCase {
            name: "input_function_call_name_pattern_invalid",
            extra: serde_json::json!({
                "input": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "bad name",
                        "arguments": "{}"
                    }
                ]
            }),
            schema_valid: false,
            expect: Expectation::MustBe400,
        },
        NestedCase {
            name: "input_function_call_output_missing_call_id_invalid",
            extra: serde_json::json!({
                "input": [
                    { "type": "function_call_output", "output": "{}" }
                ]
            }),
            schema_valid: false,
            expect: Expectation::MustBe400,
        },
        NestedCase {
            name: "input_item_reference_missing_id_invalid",
            extra: serde_json::json!({
                "input": [
                    { "type": "item_reference" }
                ]
            }),
            schema_valid: false,
            expect: Expectation::MustBe400,
        },
        NestedCase {
            name: "input_user_image_null_url_valid",
            extra: serde_json::json!({
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            { "type": "input_image", "image_url": null }
                        ]
                    }
                ]
            }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
        },
        NestedCase {
            name: "input_user_file_null_file_fields_valid",
            extra: serde_json::json!({
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_file",
                                "file_data": null,
                                "file_url": null,
                                "filename": "doc.txt"
                            }
                        ]
                    }
                ]
            }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
        },
        NestedCase {
            name: "input_user_image_invalid_detail_enum",
            extra: serde_json::json!({
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": "https://example.com/x.png",
                                "detail": "ultra"
                            }
                        ]
                    }
                ]
            }),
            schema_valid: false,
            expect: Expectation::MustBe400,
        },
        NestedCase {
            name: "input_assistant_refusal_missing_refusal_field",
            extra: serde_json::json!({
                "input": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            { "type": "refusal" }
                        ]
                    }
                ]
            }),
            schema_valid: false,
            expect: Expectation::MustBe400,
        },
        NestedCase {
            name: "input_reasoning_summary_empty_array_valid",
            extra: serde_json::json!({
                "input": [
                    { "type": "reasoning", "summary": [] }
                ]
            }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
        },
        NestedCase {
            name: "input_reasoning_summary_invalid_item_type",
            extra: serde_json::json!({
                "input": [
                    {
                        "type": "reasoning",
                        "summary": [
                            { "type": "output_text", "text": "wrong summary part" }
                        ]
                    }
                ]
            }),
            schema_valid: false,
            expect: Expectation::MustBe400,
        },
    ]
}

#[test]
fn responses_openapi_nested_cases_match_schema_classification() {
    let validator = OpenApiSchemaValidator::load_responses_spec();
    let mut failures = Vec::new();
    for case in nested_cases() {
        let body = with_base_request(&case.extra);
        let schema_ok = validator
            .validate_named_schema("CreateResponseBody", &body)
            .is_ok();
        if schema_ok != case.schema_valid {
            failures.push(format!(
                "{}: schema classification mismatch. expected schema_valid={} got={} payload={}",
                case.name, case.schema_valid, schema_ok, body
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "OpenResponses nested-case schema classification mismatches ({}):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn responses_openapi_nested_runtime_contract_matrix() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let mut failures = Vec::new();
    for case in nested_cases() {
        let body = with_base_request(&case.extra);
        let resp = post_json(ctx.addr(), "/v1/responses", &body);
        match case.expect {
            Expectation::MustBe400 => {
                if resp.status != 400 {
                    failures.push(format!(
                        "{}: expected 400 for invalid nested shape, got {} body={}",
                        case.name, resp.status, resp.body
                    ));
                }
            }
            Expectation::MustNotBe400 => {
                if resp.status == 400 {
                    failures.push(format!(
                        "{}: expected non-400 for schema-valid nested shape, got 400 body={}",
                        case.name, resp.body
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "OpenResponses nested runtime contract failures ({}):\n{}",
        failures.len(),
        failures.join("\n")
    );
}
