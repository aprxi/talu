//! Top-level OpenAPI request variants for `/v1/responses`.
//!
//! These cases focus on nullable fields, enum domains, and scalar bounds in
//! `CreateResponseBody` with explicit schema-vs-runtime assertions.

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
    extra: serde_json::Value,
    schema_valid: bool,
    expect: Expectation,
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

fn top_level_cases() -> Vec<Case> {
    vec![
        Case {
            name: "model_null",
            extra: serde_json::json!({ "model": null }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["field:model", "field:model:null"],
        },
        Case {
            name: "input_null_with_previous_response_id",
            extra: serde_json::json!({
                "input": null,
                "previous_response_id": "resp_123"
            }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &[
                "field:input",
                "field:input:null",
                "field:previous_response_id",
            ],
        },
        Case {
            name: "previous_response_id_null",
            extra: serde_json::json!({ "previous_response_id": null }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &[
                "field:previous_response_id",
                "field:previous_response_id:null",
            ],
        },
        Case {
            name: "include_empty",
            extra: serde_json::json!({ "include": [] }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["field:include", "field:include:empty"],
        },
        Case {
            name: "include_all_supported_values",
            extra: serde_json::json!({
                "include": ["reasoning.encrypted_content", "message.output_text.logprobs"]
            }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["field:include", "field:include:all_values"],
        },
        Case {
            name: "tools_null",
            extra: serde_json::json!({ "tools": null }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["field:tools", "field:tools:null"],
        },
        Case {
            name: "tool_choice_null",
            extra: serde_json::json!({ "tool_choice": null }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["field:tool_choice", "field:tool_choice:null"],
        },
        Case {
            name: "tool_choice_none",
            extra: serde_json::json!({ "tool_choice": "none" }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["field:tool_choice", "field:tool_choice:none"],
        },
        Case {
            name: "tool_choice_required",
            extra: serde_json::json!({ "tool_choice": "required" }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["field:tool_choice", "field:tool_choice:required"],
        },
        Case {
            name: "metadata_null",
            extra: serde_json::json!({ "metadata": null }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["field:metadata", "field:metadata:null"],
        },
        Case {
            name: "text_null",
            extra: serde_json::json!({ "text": null }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["field:text", "field:text:null"],
        },
        Case {
            name: "temperature_null",
            extra: serde_json::json!({ "temperature": null }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["field:temperature", "field:temperature:null"],
        },
        Case {
            name: "top_p_null",
            extra: serde_json::json!({ "top_p": null }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["field:top_p", "field:top_p:null"],
        },
        Case {
            name: "presence_penalty_null",
            extra: serde_json::json!({ "presence_penalty": null }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["field:presence_penalty", "field:presence_penalty:null"],
        },
        Case {
            name: "frequency_penalty_null",
            extra: serde_json::json!({ "frequency_penalty": null }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["field:frequency_penalty", "field:frequency_penalty:null"],
        },
        Case {
            name: "parallel_tool_calls_null",
            extra: serde_json::json!({ "parallel_tool_calls": null }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &[
                "field:parallel_tool_calls",
                "field:parallel_tool_calls:null",
            ],
        },
        Case {
            name: "stream_true",
            extra: serde_json::json!({ "stream": true }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["field:stream", "field:stream:true"],
        },
        Case {
            name: "stream_false",
            extra: serde_json::json!({ "stream": false }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["field:stream", "field:stream:false"],
        },
        Case {
            name: "stream_options_null",
            extra: serde_json::json!({ "stream_options": null }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["field:stream_options", "field:stream_options:null"],
        },
        Case {
            name: "background_false",
            extra: serde_json::json!({ "background": false }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["field:background", "field:background:false"],
        },
        Case {
            name: "max_output_tokens_null",
            extra: serde_json::json!({ "max_output_tokens": null }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["field:max_output_tokens", "field:max_output_tokens:null"],
        },
        Case {
            name: "max_tool_calls_null",
            extra: serde_json::json!({ "max_tool_calls": null }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["field:max_tool_calls", "field:max_tool_calls:null"],
        },
        Case {
            name: "reasoning_null",
            extra: serde_json::json!({ "reasoning": null }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["field:reasoning", "field:reasoning:null"],
        },
        Case {
            name: "safety_identifier_null",
            extra: serde_json::json!({ "safety_identifier": null }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["field:safety_identifier", "field:safety_identifier:null"],
        },
        Case {
            name: "prompt_cache_key_null",
            extra: serde_json::json!({ "prompt_cache_key": null }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["field:prompt_cache_key", "field:prompt_cache_key:null"],
        },
        Case {
            name: "truncation_auto",
            extra: serde_json::json!({ "truncation": "auto" }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["field:truncation", "field:truncation:auto"],
        },
        Case {
            name: "truncation_disabled",
            extra: serde_json::json!({ "truncation": "disabled" }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["field:truncation", "field:truncation:disabled"],
        },
        Case {
            name: "instructions_null",
            extra: serde_json::json!({ "instructions": null }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["field:instructions", "field:instructions:null"],
        },
        Case {
            name: "store_true",
            extra: serde_json::json!({ "store": true }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["field:store", "field:store:true"],
        },
        Case {
            name: "service_tier_auto",
            extra: serde_json::json!({ "service_tier": "auto" }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["field:service_tier", "field:service_tier:auto"],
        },
        Case {
            name: "service_tier_default",
            extra: serde_json::json!({ "service_tier": "default" }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["field:service_tier", "field:service_tier:default"],
        },
        Case {
            name: "service_tier_flex",
            extra: serde_json::json!({ "service_tier": "flex" }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["field:service_tier", "field:service_tier:flex"],
        },
        Case {
            name: "service_tier_priority",
            extra: serde_json::json!({ "service_tier": "priority" }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["field:service_tier", "field:service_tier:priority"],
        },
        Case {
            name: "top_logprobs_null",
            extra: serde_json::json!({ "top_logprobs": null }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["field:top_logprobs", "field:top_logprobs:null"],
        },
        Case {
            name: "top_logprobs_zero",
            extra: serde_json::json!({ "top_logprobs": 0 }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["field:top_logprobs", "field:top_logprobs:0"],
        },
        Case {
            name: "top_logprobs_twenty",
            extra: serde_json::json!({ "top_logprobs": 20 }),
            schema_valid: true,
            expect: Expectation::MustNotBe400,
            tags: &["field:top_logprobs", "field:top_logprobs:20"],
        },
        Case {
            name: "include_wrong_type",
            extra: serde_json::json!({ "include": "reasoning.encrypted_content" }),
            schema_valid: false,
            expect: Expectation::MustBe400,
            tags: &["invalid:include:type"],
        },
        Case {
            name: "stream_options_wrong_type",
            extra: serde_json::json!({ "stream_options": true }),
            schema_valid: false,
            expect: Expectation::MustBe400,
            tags: &["invalid:stream_options:type"],
        },
        Case {
            name: "model_wrong_type",
            extra: serde_json::json!({ "model": 42 }),
            schema_valid: false,
            expect: Expectation::MustBe400,
            tags: &["invalid:model:type"],
        },
        Case {
            name: "service_tier_invalid_enum",
            extra: serde_json::json!({ "service_tier": "standard" }),
            schema_valid: false,
            expect: Expectation::MustBe400,
            tags: &["invalid:service_tier:enum"],
        },
        Case {
            name: "truncation_invalid_enum",
            extra: serde_json::json!({ "truncation": "clip" }),
            schema_valid: false,
            expect: Expectation::MustBe400,
            tags: &["invalid:truncation:enum"],
        },
        Case {
            name: "max_tool_calls_zero",
            extra: serde_json::json!({ "max_tool_calls": 0 }),
            schema_valid: false,
            expect: Expectation::MustBe400,
            tags: &["invalid:max_tool_calls:min"],
        },
        Case {
            name: "top_logprobs_above_max",
            extra: serde_json::json!({ "top_logprobs": 21 }),
            schema_valid: false,
            expect: Expectation::MustBe400,
            tags: &["invalid:top_logprobs:max"],
        },
        Case {
            name: "top_logprobs_below_min",
            extra: serde_json::json!({ "top_logprobs": -1 }),
            schema_valid: false,
            expect: Expectation::MustBe400,
            tags: &["invalid:top_logprobs:min"],
        },
        Case {
            name: "max_output_tokens_below_min",
            extra: serde_json::json!({ "max_output_tokens": 15 }),
            schema_valid: false,
            expect: Expectation::MustBe400,
            tags: &["invalid:max_output_tokens:min"],
        },
        Case {
            name: "safety_identifier_too_long",
            extra: serde_json::json!({ "safety_identifier": "a".repeat(65) }),
            schema_valid: false,
            expect: Expectation::MustBe400,
            tags: &["invalid:safety_identifier:maxLength"],
        },
        Case {
            name: "prompt_cache_key_too_long",
            extra: serde_json::json!({ "prompt_cache_key": "a".repeat(65) }),
            schema_valid: false,
            expect: Expectation::MustBe400,
            tags: &["invalid:prompt_cache_key:maxLength"],
        },
        Case {
            name: "metadata_wrong_type",
            extra: serde_json::json!({ "metadata": "kv" }),
            schema_valid: false,
            expect: Expectation::MustBe400,
            tags: &["invalid:metadata:type"],
        },
        Case {
            name: "stream_wrong_type",
            extra: serde_json::json!({ "stream": "true" }),
            schema_valid: false,
            expect: Expectation::MustBe400,
            tags: &["invalid:stream:type"],
        },
        Case {
            name: "background_wrong_type",
            extra: serde_json::json!({ "background": "false" }),
            schema_valid: false,
            expect: Expectation::MustBe400,
            tags: &["invalid:background:type"],
        },
        Case {
            name: "store_wrong_type",
            extra: serde_json::json!({ "store": "true" }),
            schema_valid: false,
            expect: Expectation::MustBe400,
            tags: &["invalid:store:type"],
        },
        Case {
            name: "parallel_tool_calls_wrong_type",
            extra: serde_json::json!({ "parallel_tool_calls": "false" }),
            schema_valid: false,
            expect: Expectation::MustBe400,
            tags: &["invalid:parallel_tool_calls:type"],
        },
        Case {
            name: "reasoning_wrong_type",
            extra: serde_json::json!({ "reasoning": true }),
            schema_valid: false,
            expect: Expectation::MustBe400,
            tags: &["invalid:reasoning:type"],
        },
        Case {
            name: "text_wrong_type",
            extra: serde_json::json!({ "text": "text-config" }),
            schema_valid: false,
            expect: Expectation::MustBe400,
            tags: &["invalid:text:type"],
        },
    ]
}

#[test]
fn responses_openapi_top_level_cases_match_schema_classification() {
    let validator = OpenApiSchemaValidator::load_responses_spec();
    let mut failures = Vec::new();

    for case in top_level_cases() {
        let body = with_base_request(&case.extra);
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
        "OpenResponses top-level schema classification mismatches ({}):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn responses_openapi_top_level_runtime_contract_matrix() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let mut failures = Vec::new();
    let mut covered_tags = BTreeSet::new();

    for case in top_level_cases() {
        for tag in case.tags {
            covered_tags.insert((*tag).to_string());
        }

        let body = with_base_request(&case.extra);
        let resp = post_json(ctx.addr(), "/v1/responses", &body);
        match case.expect {
            Expectation::MustBe400 => {
                if resp.status != 400 {
                    failures.push(format!(
                        "{}: expected 400 for invalid shape, got {} body={}",
                        case.name, resp.status, resp.body
                    ));
                }
            }
            Expectation::MustNotBe400 => {
                if resp.status == 400 {
                    failures.push(format!(
                        "{}: expected non-400 for schema-valid shape, got 400 body={}",
                        case.name, resp.body
                    ));
                }
            }
        }
    }

    let required_tags = [
        "field:model:null",
        "field:input:null",
        "field:previous_response_id:null",
        "field:include:all_values",
        "field:tools:null",
        "field:tool_choice:null",
        "field:tool_choice:none",
        "field:tool_choice:required",
        "field:metadata:null",
        "field:text:null",
        "field:temperature:null",
        "field:top_p:null",
        "field:presence_penalty:null",
        "field:frequency_penalty:null",
        "field:parallel_tool_calls:null",
        "field:stream:true",
        "field:stream:false",
        "field:stream_options:null",
        "field:background:false",
        "field:max_output_tokens:null",
        "field:max_tool_calls:null",
        "field:reasoning:null",
        "field:safety_identifier:null",
        "field:prompt_cache_key:null",
        "field:truncation:auto",
        "field:truncation:disabled",
        "field:instructions:null",
        "field:store:true",
        "field:service_tier:auto",
        "field:service_tier:default",
        "field:service_tier:flex",
        "field:service_tier:priority",
        "field:top_logprobs:null",
        "field:top_logprobs:0",
        "field:top_logprobs:20",
        "invalid:max_output_tokens:min",
        "invalid:max_tool_calls:min",
        "invalid:top_logprobs:min",
        "invalid:top_logprobs:max",
        "invalid:service_tier:enum",
        "invalid:truncation:enum",
    ];
    for tag in required_tags {
        if !covered_tags.contains(tag) {
            failures.push(format!("top-level coverage tag missing: {tag}"));
        }
    }

    assert!(
        failures.is_empty(),
        "OpenResponses top-level runtime contract failures ({}):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn responses_openapi_top_level_cases_are_unique() {
    let mut names = BTreeSet::new();
    let mut payloads = BTreeSet::new();
    let mut failures = Vec::new();

    for case in top_level_cases() {
        if !names.insert(case.name) {
            failures.push(format!("duplicate top-level case name: {}", case.name));
        }
        let key = serde_json::to_string(&case.extra).expect("serialize top-level payload");
        if !payloads.insert(key) {
            failures.push(format!(
                "duplicate top-level payload shape (different name): {}",
                case.name
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "OpenResponses top-level case quality failures ({}):\n{}",
        failures.len(),
        failures.join("\n")
    );
}
