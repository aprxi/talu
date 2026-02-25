//! Item/content union OpenAPI request variants for `/v1/responses`.
//!
//! This suite stresses `ItemParam` polymorphism and nested content parts with
//! both valid and invalid shapes.

use super::openapi_schema::OpenApiSchemaValidator;
use crate::server::common::{post_json, ServerConfig, ServerTestContext};
use std::collections::BTreeSet;

#[derive(Clone, Copy)]
enum Expectation {
    MustBe400,
    MustNotBe400,
}

struct Case {
    name: &'static str,
    input: serde_json::Value,
    schema_valid: bool,
    expect: Expectation,
    tags: &'static [&'static str],
}

fn with_input(input: &serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "input": input,
        "max_output_tokens": 16
    })
}

fn item_cases() -> Vec<Case> {
    vec![
        Case {
            name: "item_reference_minimal",
            input: serde_json::json!([{ "id": "msg_123" }]),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["item:item_reference"],
        },
        Case {
            name: "item_reference_type_null",
            input: serde_json::json!([{ "type": null, "id": "msg_123" }]),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["item:item_reference", "item:item_reference:type_null"],
        },
        Case {
            name: "item_reference_type_string",
            input: serde_json::json!([{ "type": "item_reference", "id": "msg_123" }]),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["item:item_reference", "item:item_reference:type_value"],
        },
        Case {
            name: "reasoning_item_summary_text",
            input: serde_json::json!([{
                "type": "reasoning",
                "summary": [{ "type": "summary_text", "text": "brief summary" }]
            }]),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["item:reasoning", "content:summary_text"],
        },
        Case {
            name: "user_message_string",
            input: serde_json::json!([{
                "type": "message",
                "role": "user",
                "content": "hello from user"
            }]),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["item:user"],
        },
        Case {
            name: "user_message_input_text_part",
            input: serde_json::json!([{
                "type": "message",
                "role": "user",
                "content": [{ "type": "input_text", "text": "hello" }]
            }]),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["item:user", "content:input_text"],
        },
        Case {
            name: "user_message_input_image_part",
            input: serde_json::json!([{
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_image",
                    "image_url": "https://example.com/a.png",
                    "detail": "low"
                }]
            }]),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["item:user", "content:input_image"],
        },
        Case {
            name: "user_message_input_file_part",
            input: serde_json::json!([{
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_file",
                    "filename": "ctx.txt",
                    "file_data": "dGVzdA=="
                }]
            }]),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["item:user", "content:input_file"],
        },
        Case {
            name: "system_message_string",
            input: serde_json::json!([{
                "type": "message",
                "role": "system",
                "content": "system guidance"
            }]),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["item:system"],
        },
        Case {
            name: "system_message_input_text_part",
            input: serde_json::json!([{
                "type": "message",
                "role": "system",
                "content": [{ "type": "input_text", "text": "system note" }]
            }]),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["item:system", "content:input_text"],
        },
        Case {
            name: "developer_message_string",
            input: serde_json::json!([{
                "type": "message",
                "role": "developer",
                "content": "developer guidance"
            }]),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["item:developer"],
        },
        Case {
            name: "developer_message_input_text_part",
            input: serde_json::json!([{
                "type": "message",
                "role": "developer",
                "content": [{ "type": "input_text", "text": "dev note" }]
            }]),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["item:developer", "content:input_text"],
        },
        Case {
            name: "assistant_message_string",
            input: serde_json::json!([{
                "type": "message",
                "role": "assistant",
                "content": "assistant prior"
            }]),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["item:assistant"],
        },
        Case {
            name: "assistant_output_text_with_citation",
            input: serde_json::json!([{
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": "See docs",
                    "annotations": [{
                        "type": "url_citation",
                        "start_index": 0,
                        "end_index": 3,
                        "url": "https://example.com",
                        "title": "Example"
                    }]
                }]
            }]),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &[
                "item:assistant",
                "content:output_text",
                "content:url_citation",
            ],
        },
        Case {
            name: "assistant_refusal_part",
            input: serde_json::json!([{
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "refusal", "refusal": "cannot comply" }]
            }]),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["item:assistant", "content:refusal"],
        },
        Case {
            name: "function_call_status_in_progress",
            input: serde_json::json!([{
                "type": "function_call",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": "{}",
                "status": "in_progress"
            }]),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["item:function_call", "status:in_progress"],
        },
        Case {
            name: "function_call_status_completed",
            input: serde_json::json!([{
                "type": "function_call",
                "call_id": "call_2",
                "name": "lookup",
                "arguments": "{}",
                "status": "completed"
            }]),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["item:function_call", "status:completed"],
        },
        Case {
            name: "function_call_output_string_status_incomplete",
            input: serde_json::json!([{
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "{\"ok\":true}",
                "status": "incomplete"
            }]),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &[
                "item:function_call_output",
                "output:string",
                "status:incomplete",
            ],
        },
        Case {
            name: "function_call_output_array_input_text",
            input: serde_json::json!([{
                "type": "function_call_output",
                "call_id": "call_1",
                "output": [{ "type": "input_text", "text": "result text" }]
            }]),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &[
                "item:function_call_output",
                "output:array",
                "content:input_text",
            ],
        },
        Case {
            name: "function_call_output_array_input_image",
            input: serde_json::json!([{
                "type": "function_call_output",
                "call_id": "call_1",
                "output": [{
                    "type": "input_image",
                    "image_url": "https://example.com/img.png",
                    "detail": "auto"
                }]
            }]),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &[
                "item:function_call_output",
                "output:array",
                "content:input_image",
            ],
        },
        Case {
            name: "function_call_output_array_input_file",
            input: serde_json::json!([{
                "type": "function_call_output",
                "call_id": "call_1",
                "output": [{
                    "type": "input_file",
                    "filename": "out.json",
                    "file_url": "https://example.com/out.json"
                }]
            }]),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &[
                "item:function_call_output",
                "output:array",
                "content:input_file",
            ],
        },
        Case {
            name: "function_call_output_array_input_video",
            input: serde_json::json!([{
                "type": "function_call_output",
                "call_id": "call_1",
                "output": [{
                    "type": "input_video",
                    "video_url": "https://example.com/v.mp4"
                }]
            }]),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &[
                "item:function_call_output",
                "output:array",
                "content:input_video",
            ],
        },
        Case {
            name: "item_reference_missing_id",
            input: serde_json::json!([{ "type": "item_reference" }]),
            schema_valid: false,
            expect: Expectation::MustBe400,
            tags: &["invalid:item_reference:required_id"],
        },
        Case {
            name: "reasoning_summary_wrong_item_type",
            input: serde_json::json!([{
                "type": "reasoning",
                "summary": [{ "type": "output_text", "text": "wrong" }]
            }]),
            schema_valid: false,
            expect: Expectation::MustBe400,
            tags: &["invalid:reasoning:summary_item"],
        },
        Case {
            name: "user_message_output_text_is_invalid",
            input: serde_json::json!([{
                "type": "message",
                "role": "user",
                "content": [{ "type": "output_text", "text": "wrong for user input" }]
            }]),
            schema_valid: false,
            expect: Expectation::MustBe400,
            tags: &["invalid:user:content_type"],
        },
        Case {
            name: "system_message_input_image_is_invalid",
            input: serde_json::json!([{
                "type": "message",
                "role": "system",
                "content": [{ "type": "input_image", "image_url": "https://example.com/img.png" }]
            }]),
            schema_valid: false,
            expect: Expectation::MustBe400,
            tags: &["invalid:system:content_type"],
        },
        Case {
            name: "developer_message_input_file_is_invalid",
            input: serde_json::json!([{
                "type": "message",
                "role": "developer",
                "content": [{ "type": "input_file", "filename": "x.txt" }]
            }]),
            schema_valid: false,
            expect: Expectation::MustBe400,
            tags: &["invalid:developer:content_type"],
        },
        Case {
            name: "assistant_message_input_text_is_invalid",
            input: serde_json::json!([{
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "input_text", "text": "invalid for assistant input item" }]
            }]),
            schema_valid: false,
            expect: Expectation::MustBe400,
            tags: &["invalid:assistant:content_type"],
        },
        Case {
            name: "assistant_output_text_annotation_missing_title",
            input: serde_json::json!([{
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": "See docs",
                    "annotations": [{
                        "type": "url_citation",
                        "start_index": 0,
                        "end_index": 3,
                        "url": "https://example.com"
                    }]
                }]
            }]),
            schema_valid: false,
            expect: Expectation::MustBe400,
            tags: &["invalid:assistant:url_citation_required"],
        },
        Case {
            name: "function_call_missing_arguments",
            input: serde_json::json!([{
                "type": "function_call",
                "call_id": "call_1",
                "name": "lookup"
            }]),
            schema_valid: false,
            expect: Expectation::MustBe400,
            tags: &["invalid:function_call:required_arguments"],
        },
        Case {
            name: "function_call_bad_name_pattern",
            input: serde_json::json!([{
                "type": "function_call",
                "call_id": "call_1",
                "name": "bad name",
                "arguments": "{}"
            }]),
            schema_valid: false,
            expect: Expectation::MustBe400,
            tags: &["invalid:function_call:name_pattern"],
        },
        Case {
            name: "function_call_call_id_empty",
            input: serde_json::json!([{
                "type": "function_call",
                "call_id": "",
                "name": "lookup",
                "arguments": "{}"
            }]),
            schema_valid: false,
            expect: Expectation::MustBe400,
            tags: &["invalid:function_call:call_id_min_length"],
        },
        Case {
            name: "function_call_status_invalid_enum",
            input: serde_json::json!([{
                "type": "function_call",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": "{}",
                "status": "done"
            }]),
            schema_valid: false,
            expect: Expectation::MustBe400,
            tags: &["invalid:function_call:status_enum"],
        },
        Case {
            name: "function_call_output_missing_output",
            input: serde_json::json!([{
                "type": "function_call_output",
                "call_id": "call_1"
            }]),
            schema_valid: false,
            expect: Expectation::MustBe400,
            tags: &["invalid:function_call_output:required_output"],
        },
        Case {
            name: "function_call_output_call_id_empty",
            input: serde_json::json!([{
                "type": "function_call_output",
                "call_id": "",
                "output": "{}"
            }]),
            schema_valid: false,
            expect: Expectation::MustBe400,
            tags: &["invalid:function_call_output:call_id_min_length"],
        },
        Case {
            name: "function_call_output_status_invalid_enum",
            input: serde_json::json!([{
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "{}",
                "status": "done"
            }]),
            schema_valid: false,
            expect: Expectation::MustBe400,
            tags: &["invalid:function_call_output:status_enum"],
        },
        Case {
            name: "function_call_output_array_bad_content_type",
            input: serde_json::json!([{
                "type": "function_call_output",
                "call_id": "call_1",
                "output": [{ "type": "output_text", "text": "invalid content branch" }]
            }]),
            schema_valid: false,
            expect: Expectation::MustBe400,
            tags: &["invalid:function_call_output:content_type"],
        },
    ]
}

#[test]
fn responses_openapi_item_cases_match_schema_classification() {
    let validator = OpenApiSchemaValidator::load_responses_spec();
    let mut failures = Vec::new();

    for case in item_cases() {
        let body = with_input(&case.input);
        let schema_ok = validator
            .validate_named_schema("CreateResponseBody", &body)
            .is_ok();
        if schema_ok != case.schema_valid {
            failures.push(format!(
                "{}: schema classification mismatch expected={} got={} payload={}",
                case.name, case.schema_valid, schema_ok, body
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "OpenResponses item schema classification mismatches ({}):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn responses_openapi_item_runtime_contract_matrix() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let mut failures = Vec::new();
    let mut covered_tags = BTreeSet::new();

    for case in item_cases() {
        for tag in case.tags {
            covered_tags.insert((*tag).to_string());
        }

        let body = with_input(&case.input);
        let resp = post_json(ctx.addr(), "/v1/responses", &body);
        match case.expect {
            Expectation::MustBe400 => {
                if resp.status != 400 {
                    failures.push(format!(
                        "{}: expected 400 for invalid item/content shape, got {} body={}",
                        case.name, resp.status, resp.body
                    ));
                }
            }
            Expectation::MustNotBe400 => {
                if resp.status == 400 {
                    failures.push(format!(
                        "{}: expected non-400 for schema-valid item/content shape, got 400 body={}",
                        case.name, resp.body
                    ));
                }
            }
        }
    }

    let required_tags = [
        "item:item_reference",
        "item:reasoning",
        "item:user",
        "item:system",
        "item:developer",
        "item:assistant",
        "item:function_call",
        "item:function_call_output",
        "content:input_text",
        "content:input_image",
        "content:input_file",
        "content:input_video",
        "content:output_text",
        "content:refusal",
        "content:summary_text",
        "content:url_citation",
        "status:in_progress",
        "status:completed",
        "status:incomplete",
        "invalid:user:content_type",
        "invalid:system:content_type",
        "invalid:developer:content_type",
        "invalid:assistant:content_type",
        "invalid:function_call:required_arguments",
        "invalid:function_call_output:required_output",
    ];
    for tag in required_tags {
        if !covered_tags.contains(tag) {
            failures.push(format!("item/content coverage tag missing: {tag}"));
        }
    }

    assert!(
        failures.is_empty(),
        "OpenResponses item/content runtime contract failures ({}):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn responses_openapi_item_cases_are_unique() {
    let mut names = BTreeSet::new();
    let mut payloads = BTreeSet::new();
    let mut failures = Vec::new();

    for case in item_cases() {
        if !names.insert(case.name) {
            failures.push(format!("duplicate item case name: {}", case.name));
        }
        let key = serde_json::to_string(&case.input).expect("serialize item case payload");
        if !payloads.insert(key) {
            failures.push(format!(
                "duplicate item payload shape (different name): {}",
                case.name
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "OpenResponses item case quality failures ({}):\n{}",
        failures.len(),
        failures.join("\n")
    );
}
