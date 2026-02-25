//! Spec-driven `/v1/responses` request coverage and contract checks.
//!
//! This suite is intentionally exhaustive for `CreateResponseBody` fields from
//! `issues/responses-openapi.json`. It validates that every request field has
//! explicit test coverage and that shape/boundary constraints are enforced.

use super::openapi_schema::OpenApiSchemaValidator;
use crate::server::common::{post_json, ServerConfig, ServerTestContext};
use std::collections::BTreeSet;
use std::path::PathBuf;

const COVERED_CREATE_RESPONSE_FIELDS: &[&str] = &[
    "background",
    "frequency_penalty",
    "include",
    "input",
    "instructions",
    "max_output_tokens",
    "max_tool_calls",
    "metadata",
    "model",
    "parallel_tool_calls",
    "presence_penalty",
    "previous_response_id",
    "prompt_cache_key",
    "reasoning",
    "safety_identifier",
    "service_tier",
    "store",
    "stream",
    "stream_options",
    "temperature",
    "text",
    "tool_choice",
    "tools",
    "top_logprobs",
    "top_p",
    "truncation",
];

#[derive(Clone, Copy)]
enum Expectation {
    Status(u16),
    StatusContains(u16, &'static str),
    NotStatus(u16),
}

struct Case {
    name: &'static str,
    field: &'static str,
    extra: serde_json::Value,
    expect: Expectation,
}

fn spec_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../issues/responses-openapi.json")
        .canonicalize()
        .expect("canonicalize responses-openapi.json path")
}

fn openapi_create_response_fields() -> BTreeSet<String> {
    let raw = std::fs::read_to_string(spec_path()).expect("read responses-openapi.json");
    let json: serde_json::Value = serde_json::from_str(&raw).expect("parse responses-openapi.json");
    json["components"]["schemas"]["CreateResponseBody"]["properties"]
        .as_object()
        .expect("CreateResponseBody.properties object")
        .keys()
        .cloned()
        .collect()
}

fn request_with_base(extra: &serde_json::Value) -> serde_json::Value {
    let mut body = serde_json::json!({
        "input": "hello",
        "max_output_tokens": 16
    });
    if let Some(obj) = extra.as_object() {
        for (k, v) in obj {
            body[k] = v.clone();
        }
    }
    body
}

fn run_case(ctx: &ServerTestContext, case: &Case) -> Result<(), String> {
    let body = request_with_base(&case.extra);
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    match case.expect {
        Expectation::Status(code) => {
            if resp.status != code {
                return Err(format!(
                    "[{}:{}] expected status {} got {} body={}",
                    case.field, case.name, code, resp.status, resp.body
                ));
            }
        }
        Expectation::StatusContains(code, snippet) => {
            if resp.status != code {
                return Err(format!(
                    "[{}:{}] expected status {} got {} body={}",
                    case.field, case.name, code, resp.status, resp.body
                ));
            }
            if !resp.body.contains(snippet) {
                return Err(format!(
                    "[{}:{}] expected body to contain {:?}, body={}",
                    case.field, case.name, snippet, resp.body
                ));
            }
        }
        Expectation::NotStatus(code) => {
            if resp.status == code {
                return Err(format!(
                    "[{}:{}] expected status != {} got {} body={}",
                    case.field, case.name, code, resp.status, resp.body
                ));
            }
        }
    }
    Ok(())
}

fn contract_cases() -> Vec<Case> {
    let long_input = "a".repeat(10_485_761);
    let long_metadata_key = "k".repeat(65);
    let long_metadata_value = "v".repeat(513);
    let long_prompt_cache_key = "p".repeat(65);

    let mut too_many_metadata_entries = serde_json::Map::new();
    for i in 0..17 {
        too_many_metadata_entries.insert(format!("k{i}"), serde_json::json!("v"));
    }

    vec![
        // background
        Case {
            name: "background_valid_bool_unimplemented",
            field: "background",
            extra: serde_json::json!({ "background": true }),
            expect: Expectation::StatusContains(400, "background"),
        },
        Case {
            name: "background_invalid_type",
            field: "background",
            extra: serde_json::json!({ "background": "true" }),
            expect: Expectation::Status(400),
        },
        // frequency_penalty
        Case {
            name: "frequency_penalty_valid_in_range_unimplemented",
            field: "frequency_penalty",
            extra: serde_json::json!({ "frequency_penalty": 0.5 }),
            expect: Expectation::StatusContains(400, "frequency_penalty"),
        },
        Case {
            name: "frequency_penalty_invalid_out_of_range",
            field: "frequency_penalty",
            extra: serde_json::json!({ "frequency_penalty": 2.1 }),
            expect: Expectation::StatusContains(400, "between -2 and 2"),
        },
        // include
        Case {
            name: "include_valid_logprobs_unimplemented",
            field: "include",
            extra: serde_json::json!({ "include": ["message.output_text.logprobs"] }),
            expect: Expectation::StatusContains(400, "include.message.output_text.logprobs"),
        },
        Case {
            name: "include_invalid_value",
            field: "include",
            extra: serde_json::json!({ "include": ["message.output_text.unknown"] }),
            expect: Expectation::StatusContains(400, "unsupported value"),
        },
        Case {
            name: "include_invalid_type",
            field: "include",
            extra: serde_json::json!({ "include": "message.output_text.logprobs" }),
            expect: Expectation::StatusContains(400, "`include` must be an array"),
        },
        // input
        Case {
            name: "input_valid_string",
            field: "input",
            extra: serde_json::json!({ "input": "hello world" }),
            expect: Expectation::NotStatus(400),
        },
        Case {
            name: "input_invalid_object",
            field: "input",
            extra: serde_json::json!({ "input": { "bad": "shape" } }),
            expect: Expectation::StatusContains(400, "`input` must be"),
        },
        Case {
            name: "input_invalid_too_large",
            field: "input",
            extra: serde_json::json!({ "input": long_input }),
            expect: Expectation::StatusContains(400, "input"),
        },
        // instructions
        Case {
            name: "instructions_valid_string",
            field: "instructions",
            extra: serde_json::json!({ "instructions": "be concise" }),
            expect: Expectation::NotStatus(400),
        },
        Case {
            name: "instructions_invalid_type",
            field: "instructions",
            extra: serde_json::json!({ "instructions": { "bad": true } }),
            expect: Expectation::Status(400),
        },
        // max_output_tokens
        Case {
            name: "max_output_tokens_valid_min",
            field: "max_output_tokens",
            extra: serde_json::json!({ "max_output_tokens": 16 }),
            expect: Expectation::NotStatus(400),
        },
        Case {
            name: "max_output_tokens_invalid_below_min",
            field: "max_output_tokens",
            extra: serde_json::json!({ "max_output_tokens": 15 }),
            expect: Expectation::StatusContains(400, "at least 16"),
        },
        // max_tool_calls
        Case {
            name: "max_tool_calls_valid_min_unimplemented",
            field: "max_tool_calls",
            extra: serde_json::json!({ "max_tool_calls": 1 }),
            expect: Expectation::StatusContains(400, "max_tool_calls"),
        },
        Case {
            name: "max_tool_calls_invalid_below_min",
            field: "max_tool_calls",
            extra: serde_json::json!({ "max_tool_calls": 0 }),
            expect: Expectation::StatusContains(400, "at least 1"),
        },
        // metadata
        Case {
            name: "metadata_valid_object",
            field: "metadata",
            extra: serde_json::json!({ "metadata": { "k": "v" } }),
            expect: Expectation::NotStatus(400),
        },
        Case {
            name: "metadata_invalid_non_object",
            field: "metadata",
            extra: serde_json::json!({ "metadata": "nope" }),
            expect: Expectation::StatusContains(400, "`metadata` must be an object"),
        },
        Case {
            name: "metadata_invalid_too_many_entries",
            field: "metadata",
            extra: serde_json::json!({ "metadata": serde_json::Value::Object(too_many_metadata_entries) }),
            expect: Expectation::StatusContains(400, "metadata"),
        },
        Case {
            name: "metadata_invalid_key_too_long",
            field: "metadata",
            extra: serde_json::json!({ "metadata": { long_metadata_key: "v" } }),
            expect: Expectation::StatusContains(400, "metadata"),
        },
        Case {
            name: "metadata_invalid_value_too_long",
            field: "metadata",
            extra: serde_json::json!({ "metadata": { "k": long_metadata_value } }),
            expect: Expectation::StatusContains(400, "metadata"),
        },
        // model
        Case {
            name: "model_valid_string",
            field: "model",
            extra: serde_json::json!({ "model": "unknown/model" }),
            expect: Expectation::NotStatus(400),
        },
        Case {
            name: "model_invalid_type",
            field: "model",
            extra: serde_json::json!({ "model": 123 }),
            expect: Expectation::Status(400),
        },
        // parallel_tool_calls
        Case {
            name: "parallel_tool_calls_valid_bool_unimplemented",
            field: "parallel_tool_calls",
            extra: serde_json::json!({ "parallel_tool_calls": false }),
            expect: Expectation::StatusContains(400, "parallel_tool_calls"),
        },
        Case {
            name: "parallel_tool_calls_invalid_type",
            field: "parallel_tool_calls",
            extra: serde_json::json!({ "parallel_tool_calls": "false" }),
            expect: Expectation::Status(400),
        },
        // presence_penalty
        Case {
            name: "presence_penalty_valid_in_range_unimplemented",
            field: "presence_penalty",
            extra: serde_json::json!({ "presence_penalty": -0.5 }),
            expect: Expectation::StatusContains(400, "presence_penalty"),
        },
        Case {
            name: "presence_penalty_invalid_out_of_range",
            field: "presence_penalty",
            extra: serde_json::json!({ "presence_penalty": -2.1 }),
            expect: Expectation::StatusContains(400, "between -2 and 2"),
        },
        // previous_response_id
        Case {
            name: "previous_response_id_valid_string",
            field: "previous_response_id",
            extra: serde_json::json!({ "previous_response_id": "resp_123" }),
            expect: Expectation::NotStatus(400),
        },
        Case {
            name: "previous_response_id_invalid_type",
            field: "previous_response_id",
            extra: serde_json::json!({ "previous_response_id": 123 }),
            expect: Expectation::Status(400),
        },
        // prompt_cache_key
        Case {
            name: "prompt_cache_key_valid_unimplemented",
            field: "prompt_cache_key",
            extra: serde_json::json!({ "prompt_cache_key": "cache-key" }),
            expect: Expectation::StatusContains(400, "prompt_cache_key"),
        },
        Case {
            name: "prompt_cache_key_invalid_too_long",
            field: "prompt_cache_key",
            extra: serde_json::json!({ "prompt_cache_key": long_prompt_cache_key }),
            expect: Expectation::StatusContains(400, "at most 64"),
        },
        Case {
            name: "prompt_cache_key_invalid_type",
            field: "prompt_cache_key",
            extra: serde_json::json!({ "prompt_cache_key": 1 }),
            expect: Expectation::Status(400),
        },
        // reasoning
        Case {
            name: "reasoning_valid_effort_unimplemented",
            field: "reasoning",
            extra: serde_json::json!({ "reasoning": { "effort": "high" } }),
            expect: Expectation::StatusContains(400, "reasoning"),
        },
        Case {
            name: "reasoning_invalid_effort_enum",
            field: "reasoning",
            extra: serde_json::json!({ "reasoning": { "effort": "minimal" } }),
            expect: Expectation::StatusContains(400, "reasoning.effort"),
        },
        // safety_identifier
        Case {
            name: "safety_identifier_valid",
            field: "safety_identifier",
            extra: serde_json::json!({ "safety_identifier": "req-123" }),
            expect: Expectation::NotStatus(400),
        },
        Case {
            name: "safety_identifier_invalid_too_long",
            field: "safety_identifier",
            extra: serde_json::json!({ "safety_identifier": "a".repeat(65) }),
            expect: Expectation::StatusContains(400, "at most 64"),
        },
        // service_tier
        Case {
            name: "service_tier_valid_priority_unimplemented",
            field: "service_tier",
            extra: serde_json::json!({ "service_tier": "priority" }),
            expect: Expectation::StatusContains(400, "service_tier"),
        },
        Case {
            name: "service_tier_invalid_enum",
            field: "service_tier",
            extra: serde_json::json!({ "service_tier": "standard" }),
            expect: Expectation::StatusContains(400, "must be one of"),
        },
        // store
        Case {
            name: "store_valid_bool",
            field: "store",
            extra: serde_json::json!({ "store": false }),
            expect: Expectation::NotStatus(400),
        },
        Case {
            name: "store_invalid_type",
            field: "store",
            extra: serde_json::json!({ "store": "false" }),
            expect: Expectation::Status(400),
        },
        // stream
        Case {
            name: "stream_valid_bool",
            field: "stream",
            extra: serde_json::json!({ "stream": true }),
            expect: Expectation::NotStatus(400),
        },
        Case {
            name: "stream_invalid_type",
            field: "stream",
            extra: serde_json::json!({ "stream": "true" }),
            expect: Expectation::Status(400),
        },
        // stream_options
        Case {
            name: "stream_options_valid_include_obfuscation_unimplemented",
            field: "stream_options",
            extra: serde_json::json!({ "stream_options": { "include_obfuscation": true } }),
            expect: Expectation::StatusContains(400, "stream_options.include_obfuscation"),
        },
        Case {
            name: "stream_options_invalid_include_obfuscation_type",
            field: "stream_options",
            extra: serde_json::json!({ "stream_options": { "include_obfuscation": "yes" } }),
            expect: Expectation::StatusContains(400, "must be a boolean"),
        },
        Case {
            name: "stream_options_invalid_unknown_key",
            field: "stream_options",
            extra: serde_json::json!({ "stream_options": { "other": 1 } }),
            expect: Expectation::StatusContains(400, "not supported"),
        },
        // temperature
        Case {
            name: "temperature_valid_max",
            field: "temperature",
            extra: serde_json::json!({ "temperature": 2.0 }),
            expect: Expectation::NotStatus(400),
        },
        Case {
            name: "temperature_invalid_out_of_range",
            field: "temperature",
            extra: serde_json::json!({ "temperature": 5.0 }),
            expect: Expectation::StatusContains(400, "between 0 and 2"),
        },
        // text
        Case {
            name: "text_valid_text_format",
            field: "text",
            extra: serde_json::json!({ "text": { "format": { "type": "text" } } }),
            expect: Expectation::NotStatus(400),
        },
        Case {
            name: "text_valid_null_format",
            field: "text",
            extra: serde_json::json!({ "text": { "format": null } }),
            expect: Expectation::NotStatus(400),
        },
        Case {
            name: "text_invalid_json_object_not_in_request_schema",
            field: "text",
            extra: serde_json::json!({ "text": { "format": { "type": "json_object" } } }),
            expect: Expectation::Status(400),
        },
        Case {
            name: "text_valid_json_schema_unimplemented",
            field: "text",
            extra: serde_json::json!({
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "reply",
                        "schema": { "type": "object" }
                    }
                }
            }),
            expect: Expectation::StatusContains(400, "text.format.json_schema"),
        },
        Case {
            name: "text_invalid_format_type",
            field: "text",
            extra: serde_json::json!({ "text": { "format": { "type": "xml" } } }),
            expect: Expectation::StatusContains(400, "text.format.type"),
        },
        // tool_choice
        Case {
            name: "tool_choice_valid_string",
            field: "tool_choice",
            extra: serde_json::json!({ "tool_choice": "auto" }),
            expect: Expectation::NotStatus(400),
        },
        Case {
            name: "tool_choice_invalid_string",
            field: "tool_choice",
            extra: serde_json::json!({ "tool_choice": "sometimes" }),
            expect: Expectation::StatusContains(400, "tool_choice"),
        },
        // tools
        Case {
            name: "tools_valid_array",
            field: "tools",
            extra: serde_json::json!({ "tools": [] }),
            expect: Expectation::NotStatus(400),
        },
        Case {
            name: "tools_invalid_non_array",
            field: "tools",
            extra: serde_json::json!({ "tools": {} }),
            expect: Expectation::StatusContains(400, "`tools` must be an array"),
        },
        // top_logprobs
        Case {
            name: "top_logprobs_valid_zero",
            field: "top_logprobs",
            extra: serde_json::json!({ "top_logprobs": 0 }),
            expect: Expectation::NotStatus(400),
        },
        Case {
            name: "top_logprobs_valid_positive_unimplemented",
            field: "top_logprobs",
            extra: serde_json::json!({ "top_logprobs": 3 }),
            expect: Expectation::StatusContains(400, "top_logprobs"),
        },
        Case {
            name: "top_logprobs_invalid_above_max",
            field: "top_logprobs",
            extra: serde_json::json!({ "top_logprobs": 21 }),
            expect: Expectation::StatusContains(400, "between 0 and 20"),
        },
        Case {
            name: "top_logprobs_invalid_below_min",
            field: "top_logprobs",
            extra: serde_json::json!({ "top_logprobs": -1 }),
            expect: Expectation::StatusContains(400, "between 0 and 20"),
        },
        // top_p
        Case {
            name: "top_p_valid_max",
            field: "top_p",
            extra: serde_json::json!({ "top_p": 1.0 }),
            expect: Expectation::NotStatus(400),
        },
        Case {
            name: "top_p_invalid_out_of_range",
            field: "top_p",
            extra: serde_json::json!({ "top_p": -0.1 }),
            expect: Expectation::StatusContains(400, "between 0 and 1"),
        },
        // truncation
        Case {
            name: "truncation_valid_auto",
            field: "truncation",
            extra: serde_json::json!({ "truncation": "auto" }),
            expect: Expectation::NotStatus(400),
        },
        Case {
            name: "truncation_valid_disabled_unimplemented",
            field: "truncation",
            extra: serde_json::json!({ "truncation": "disabled" }),
            expect: Expectation::StatusContains(400, "truncation.disabled"),
        },
        Case {
            name: "truncation_invalid_enum",
            field: "truncation",
            extra: serde_json::json!({ "truncation": "clip" }),
            expect: Expectation::StatusContains(400, "must be one of"),
        },
    ]
}

#[test]
fn responses_openapi_create_response_body_fields_are_exhaustively_covered() {
    let expected = openapi_create_response_fields();
    let covered: BTreeSet<String> = COVERED_CREATE_RESPONSE_FIELDS
        .iter()
        .map(|s| s.to_string())
        .collect();

    let missing: Vec<String> = expected.difference(&covered).cloned().collect();
    let extra: Vec<String> = covered.difference(&expected).cloned().collect();

    assert!(
        missing.is_empty() && extra.is_empty(),
        "CreateResponseBody coverage mismatch. missing={missing:?} extra={extra:?}"
    );
}

#[test]
fn responses_openapi_create_response_body_contract_matrix() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let cases = contract_cases();

    let covered_from_cases: BTreeSet<String> = cases.iter().map(|c| c.field.to_string()).collect();
    let expected: BTreeSet<String> = COVERED_CREATE_RESPONSE_FIELDS
        .iter()
        .map(|s| s.to_string())
        .collect();
    let missing_case_fields: Vec<String> =
        expected.difference(&covered_from_cases).cloned().collect();
    assert!(
        missing_case_fields.is_empty(),
        "missing field cases in contract matrix: {missing_case_fields:?}"
    );

    let mut failures = Vec::new();
    for case in &cases {
        if let Err(failure) = run_case(&ctx, case) {
            failures.push(failure);
        }
    }

    assert!(
        failures.is_empty(),
        "OpenResponses CreateResponseBody contract failures ({}):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn responses_openapi_contract_cases_are_unique_and_varied() {
    use std::collections::{BTreeMap, BTreeSet};

    let validator = OpenApiSchemaValidator::load_responses_spec();
    let cases = contract_cases();
    let mut name_seen = BTreeSet::new();
    let mut payload_seen = BTreeSet::new();
    let mut by_field: BTreeMap<&str, (usize, usize, usize)> = BTreeMap::new();

    let mut duplicate_names = Vec::new();
    let mut duplicate_payloads = Vec::new();

    for case in &cases {
        if !name_seen.insert(case.name) {
            duplicate_names.push(case.name.to_string());
        }

        let key = format!(
            "{}::{}",
            case.field,
            serde_json::to_string(&case.extra).expect("serialize case payload")
        );
        if !payload_seen.insert(key) {
            duplicate_payloads.push(case.name.to_string());
        }

        let body = request_with_base(&case.extra);
        let schema_ok = validator
            .validate_named_schema("CreateResponseBody", &body)
            .is_ok();
        let entry = by_field.entry(case.field).or_insert((0, 0, 0));
        if schema_ok {
            entry.1 += 1;
        } else {
            entry.0 += 1;
        }
        if matches!(
            case.expect,
            Expectation::Status(400) | Expectation::StatusContains(400, _)
        ) {
            entry.2 += 1;
        }
    }

    let mut failures = Vec::new();
    if !duplicate_names.is_empty() {
        failures.push(format!("duplicate case names: {duplicate_names:?}"));
    }
    if !duplicate_payloads.is_empty() {
        failures.push(format!(
            "duplicate field+payload cases: {duplicate_payloads:?}"
        ));
    }

    for (field, (schema_bad, schema_ok, runtime_bad)) in by_field {
        if schema_bad == 0 && runtime_bad == 0 {
            failures.push(format!(
                "field `{field}` has no negative contract case (schema-invalid or runtime-invalid)"
            ));
        }
        if schema_ok == 0 {
            failures.push(format!("field `{field}` has no schema-valid contract case"));
        }
    }

    assert!(
        failures.is_empty(),
        "OpenResponses contract case quality failures ({}):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn responses_openapi_contract_case_shapes_match_create_response_schema() {
    let validator = OpenApiSchemaValidator::load_responses_spec();
    let schema_invalid: std::collections::BTreeSet<&str> = [
        "background_invalid_type",
        "include_invalid_value",
        "include_invalid_type",
        "input_invalid_object",
        "input_invalid_too_large",
        "instructions_invalid_type",
        "max_output_tokens_invalid_below_min",
        "max_tool_calls_invalid_below_min",
        "metadata_invalid_non_object",
        "metadata_invalid_too_many_entries",
        "metadata_invalid_value_too_long",
        "model_invalid_type",
        "parallel_tool_calls_invalid_type",
        "previous_response_id_invalid_type",
        "prompt_cache_key_invalid_too_long",
        "prompt_cache_key_invalid_type",
        "reasoning_invalid_effort_enum",
        "safety_identifier_invalid_too_long",
        "service_tier_invalid_enum",
        "store_invalid_type",
        "stream_invalid_type",
        "stream_options_invalid_include_obfuscation_type",
        "text_invalid_json_object_not_in_request_schema",
        "text_invalid_format_type",
        "tool_choice_invalid_string",
        "tools_invalid_non_array",
        "top_logprobs_invalid_above_max",
        "top_logprobs_invalid_below_min",
        "truncation_invalid_enum",
    ]
    .into_iter()
    .collect();
    let schema_valid: std::collections::BTreeSet<&str> = [
        "background_valid_bool_unimplemented",
        "frequency_penalty_valid_in_range_unimplemented",
        "frequency_penalty_invalid_out_of_range",
        "include_valid_logprobs_unimplemented",
        "input_valid_string",
        "instructions_valid_string",
        "max_output_tokens_valid_min",
        "max_tool_calls_valid_min_unimplemented",
        "metadata_valid_object",
        "metadata_invalid_key_too_long",
        "model_valid_string",
        "parallel_tool_calls_valid_bool_unimplemented",
        "presence_penalty_valid_in_range_unimplemented",
        "presence_penalty_invalid_out_of_range",
        "previous_response_id_valid_string",
        "prompt_cache_key_valid_unimplemented",
        "reasoning_valid_effort_unimplemented",
        "safety_identifier_valid",
        "service_tier_valid_priority_unimplemented",
        "store_valid_bool",
        "stream_valid_bool",
        "stream_options_valid_include_obfuscation_unimplemented",
        "stream_options_invalid_unknown_key",
        "temperature_valid_max",
        "temperature_invalid_out_of_range",
        "text_valid_text_format",
        "text_valid_null_format",
        "text_valid_json_schema_unimplemented",
        "tool_choice_valid_string",
        "tools_valid_array",
        "top_logprobs_valid_zero",
        "top_logprobs_valid_positive_unimplemented",
        "top_p_valid_max",
        "top_p_invalid_out_of_range",
        "truncation_valid_auto",
        "truncation_valid_disabled_unimplemented",
    ]
    .into_iter()
    .collect();

    let mut failures = Vec::new();
    let mut unclassified = Vec::new();

    for case in contract_cases() {
        let body = request_with_base(&case.extra);
        let schema_ok = validator
            .validate_named_schema("CreateResponseBody", &body)
            .is_ok();
        let is_invalid = schema_invalid.contains(case.name);
        let is_valid = schema_valid.contains(case.name);

        if !is_invalid && !is_valid {
            unclassified.push(case.name.to_string());
        }

        if is_invalid && schema_ok {
            failures.push(format!(
                "[{}:{}] expected schema INVALID but validator accepted body={}",
                case.field, case.name, body
            ));
        }
        if is_valid && !schema_ok {
            failures.push(format!(
                "[{}:{}] expected schema VALID but validator rejected body={}",
                case.field, case.name, body
            ));
        }
    }

    if !unclassified.is_empty() {
        failures.push(format!("unclassified cases: {:?}", unclassified));
    }

    assert!(
        failures.is_empty(),
        "OpenResponses case/schema mismatch ({}):\n{}",
        failures.len(),
        failures.join("\n")
    );
}
