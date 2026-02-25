//! Target-state conformance for polymorphic `/v1/responses` request shapes.
//!
//! This suite focuses on diverse, schema-valid payloads across ItemParam,
//! content-part unions, tools, and tool_choice. It is expected to fail while
//! valid OpenResponses fields remain unimplemented in the server.

use super::openapi_schema::OpenApiSchemaValidator;
use crate::server::common::{post_json, ServerConfig, ServerTestContext};
use std::collections::BTreeSet;

struct PolyCase {
    name: &'static str,
    extra: serde_json::Value,
    tags: &'static [&'static str],
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

fn poly_cases() -> Vec<PolyCase> {
    vec![
        PolyCase {
            name: "user_text_part_with_presence_penalty",
            extra: serde_json::json!({
                "input": [
                    { "type": "message", "role": "user", "content": [
                        { "type": "input_text", "text": "hello" }
                    ]}
                ],
                "presence_penalty": 0.5
            }),
            tags: &["item:user", "content:input_text", "field:presence_penalty"],
        },
        PolyCase {
            name: "user_image_part_with_frequency_penalty",
            extra: serde_json::json!({
                "input": [
                    { "type": "message", "role": "user", "content": [
                        {
                            "type": "input_image",
                            "image_url": "file_123",
                            "detail": "auto"
                        }
                    ]}
                ],
                "frequency_penalty": -0.5
            }),
            tags: &[
                "item:user",
                "content:input_image",
                "field:frequency_penalty",
            ],
        },
        PolyCase {
            name: "user_file_part_with_parallel_tool_calls",
            extra: serde_json::json!({
                "input": [
                    { "type": "message", "role": "user", "content": [
                        {
                            "type": "input_file",
                            "file_url": "file_123",
                            "filename": "ctx.txt"
                        }
                    ]}
                ],
                "parallel_tool_calls": true
            }),
            tags: &[
                "item:user",
                "content:input_file",
                "field:parallel_tool_calls",
            ],
        },
        PolyCase {
            name: "assistant_output_text_with_include_logprobs",
            extra: serde_json::json!({
                "input": [
                    { "type": "message", "role": "assistant", "content": [
                        { "type": "output_text", "text": "prior", "annotations": [] }
                    ]}
                ],
                "include": ["message.output_text.logprobs"]
            }),
            tags: &[
                "item:assistant",
                "content:output_text",
                "field:include_logprobs",
            ],
        },
        PolyCase {
            name: "assistant_refusal_with_reasoning_effort",
            extra: serde_json::json!({
                "input": [
                    { "type": "message", "role": "assistant", "content": [
                        { "type": "refusal", "refusal": "cannot do that" }
                    ]}
                ],
                "reasoning": { "effort": "high" }
            }),
            tags: &["item:assistant", "content:refusal", "field:reasoning"],
        },
        PolyCase {
            name: "system_message_with_reasoning_summary",
            extra: serde_json::json!({
                "input": [
                    { "type": "message", "role": "system", "content": "follow policy" }
                ],
                "reasoning": { "summary": "detailed" }
            }),
            tags: &["item:system", "field:reasoning"],
        },
        PolyCase {
            name: "developer_message_with_service_tier",
            extra: serde_json::json!({
                "input": [
                    { "type": "message", "role": "developer", "content": "prefer concise code" }
                ],
                "service_tier": "priority"
            }),
            tags: &["item:developer", "field:service_tier"],
        },
        PolyCase {
            name: "reasoning_item_with_encrypted_content_include",
            extra: serde_json::json!({
                "input": [
                    {
                        "type": "reasoning",
                        "summary": [{ "type": "summary_text", "text": "think briefly" }]
                    }
                ],
                "include": ["reasoning.encrypted_content"]
            }),
            tags: &[
                "item:reasoning",
                "content:summary_text",
                "field:include_reasoning",
            ],
        },
        PolyCase {
            name: "function_call_item_with_background",
            extra: serde_json::json!({
                "input": [
                    {
                        "type": "function_call",
                        "call_id": "call_123",
                        "name": "lookup",
                        "arguments": "{}"
                    }
                ],
                "background": false
            }),
            tags: &["item:function_call", "field:background"],
        },
        PolyCase {
            name: "function_call_output_string_with_prompt_cache_key",
            extra: serde_json::json!({
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_123",
                        "output": "{\"ok\":true}"
                    }
                ],
                "prompt_cache_key": "cache-key"
            }),
            tags: &[
                "item:function_call_output",
                "output:string",
                "field:prompt_cache_key",
            ],
        },
        PolyCase {
            name: "function_call_output_video_array_with_top_logprobs",
            extra: serde_json::json!({
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_123",
                        "output": [
                            { "type": "input_video", "video_url": "https://example.com/v.mp4" }
                        ]
                    }
                ],
                "top_logprobs": 5
            }),
            tags: &[
                "item:function_call_output",
                "output:array",
                "content:input_video",
                "field:top_logprobs",
            ],
        },
        PolyCase {
            name: "item_reference_with_stream_options",
            extra: serde_json::json!({
                "input": [
                    { "type": "item_reference", "id": "msg_123" }
                ],
                "stream_options": { "include_obfuscation": false }
            }),
            tags: &["item:item_reference", "field:stream_options"],
        },
        PolyCase {
            name: "tools_function_with_max_tool_calls",
            extra: serde_json::json!({
                "tools": [
                    { "type": "function", "name": "get_weather" }
                ],
                "max_tool_calls": 2
            }),
            tags: &["tools:function", "field:max_tool_calls"],
        },
        PolyCase {
            name: "tool_choice_specific_function",
            extra: serde_json::json!({
                "tools": [{ "type": "function", "name": "get_weather" }],
                "tool_choice": { "type": "function", "name": "get_weather" }
            }),
            tags: &["tools:function", "tool_choice:function"],
        },
        PolyCase {
            name: "tool_choice_allowed_tools_mode_required",
            extra: serde_json::json!({
                "tools": [{ "type": "function", "name": "get_weather" }],
                "tool_choice": {
                    "type": "allowed_tools",
                    "tools": [{ "type": "function", "name": "get_weather" }],
                    "mode": "required"
                }
            }),
            tags: &["tools:function", "tool_choice:allowed_tools"],
        },
        PolyCase {
            name: "truncation_disabled_with_json_schema_format",
            extra: serde_json::json!({
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "reply",
                        "description": "typed answer",
                        "schema": { "type": "object" },
                        "strict": true
                    }
                },
                "truncation": "disabled"
            }),
            tags: &["field:truncation_disabled", "field:text_json_schema"],
        },
        PolyCase {
            name: "text_verbosity_with_reasoning_and_penalties",
            extra: serde_json::json!({
                "text": { "verbosity": "high" },
                "reasoning": { "effort": "medium", "summary": "concise" },
                "presence_penalty": 0.5,
                "frequency_penalty": 0.5
            }),
            tags: &[
                "field:text_verbosity",
                "field:reasoning",
                "field:presence_penalty",
                "field:frequency_penalty",
            ],
        },
        PolyCase {
            name: "multimodal_user_content_with_service_and_background",
            extra: serde_json::json!({
                "input": [
                    { "type": "message", "role": "user", "content": [
                        { "type": "input_text", "text": "what is this?" },
                        { "type": "input_image", "image_url": "file_img_1" },
                        { "type": "input_file", "file_url": "file_doc_1", "filename": "doc.txt" }
                    ]}
                ],
                "service_tier": "auto",
                "background": true
            }),
            tags: &[
                "item:user",
                "content:input_text",
                "content:input_image",
                "content:input_file",
                "field:service_tier",
                "field:background",
            ],
        },
        PolyCase {
            name: "previous_response_chain_with_null_input_and_logprobs",
            extra: serde_json::json!({
                "previous_response_id": "resp_123",
                "input": null,
                "include": ["message.output_text.logprobs"],
                "top_logprobs": 1
            }),
            tags: &[
                "field:previous_response_id",
                "field:include_logprobs",
                "field:top_logprobs",
            ],
        },
        PolyCase {
            name: "streaming_with_stream_options_and_reasoning_summary",
            extra: serde_json::json!({
                "stream": true,
                "stream_options": { "include_obfuscation": true },
                "reasoning": { "summary": "auto" }
            }),
            tags: &["field:stream", "field:stream_options", "field:reasoning"],
        },
        PolyCase {
            name: "text_format_null",
            extra: serde_json::json!({
                "text": { "format": null }
            }),
            tags: &["field:text_format_null"],
        },
    ]
}

#[test]
fn responses_target_polymorphic_matrix_has_no_gaps() {
    let validator = OpenApiSchemaValidator::load_responses_spec();
    let ctx = ServerTestContext::new(ServerConfig::new());
    let cases = poly_cases();

    let mut failures = Vec::new();
    let mut covered = BTreeSet::new();

    for case in &cases {
        for tag in case.tags {
            covered.insert((*tag).to_string());
        }

        let body = with_base_request(&case.extra);
        if let Err(errors) = validator.validate_named_schema("CreateResponseBody", &body) {
            failures.push(format!(
                "{}: payload should be schema-valid but validator rejected:\n{}\npayload={}",
                case.name,
                errors.join("\n"),
                body
            ));
            continue;
        }

        let resp = post_json(ctx.addr(), "/v1/responses", &body);
        if resp.status == 400 {
            failures.push(format!(
                "{}: target-state expected non-400 for schema-valid payload, got 400 body={}",
                case.name, resp.body
            ));
        }
    }

    let required_tags = [
        "item:user",
        "item:assistant",
        "item:system",
        "item:developer",
        "item:reasoning",
        "item:function_call",
        "item:function_call_output",
        "item:item_reference",
        "content:input_text",
        "content:input_image",
        "content:input_file",
        "content:input_video",
        "content:output_text",
        "content:refusal",
        "content:summary_text",
        "tools:function",
        "tool_choice:function",
        "tool_choice:allowed_tools",
        "field:previous_response_id",
        "field:presence_penalty",
        "field:frequency_penalty",
        "field:top_logprobs",
        "field:reasoning",
        "field:text_json_schema",
        "field:text_verbosity",
        "field:stream_options",
        "field:truncation_disabled",
    ];
    for tag in required_tags {
        if !covered.contains(tag) {
            failures.push(format!(
                "coverage tag missing from polymorphic matrix: {tag}"
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "OpenResponses polymorphic target gaps ({}):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn responses_target_polymorphic_cases_are_unique() {
    let cases = poly_cases();
    let mut names = BTreeSet::new();
    let mut payloads = BTreeSet::new();
    let mut failures = Vec::new();

    for case in cases {
        if !names.insert(case.name) {
            failures.push(format!("duplicate polymorphic case name: {}", case.name));
        }
        let key = serde_json::to_string(&case.extra).expect("serialize polymorphic case payload");
        if !payloads.insert(key) {
            failures.push(format!(
                "duplicate polymorphic payload shape (different name): {}",
                case.name
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "OpenResponses polymorphic case quality failures ({}):\n{}",
        failures.len(),
        failures.join("\n")
    );
}
