//! Schema-driven payload checks for `/v1/responses`.
//!
//! These tests validate emitted JSON payloads against OpenResponses schemas
//! from `issues/responses-openapi.json`.

use super::openapi_schema::OpenApiSchemaValidator;
use crate::server::common::{model_config, post_json, require_model, ServerTestContext};
use std::collections::BTreeSet;

fn parse_sse_events(body: &str) -> Vec<serde_json::Value> {
    let mut events = Vec::new();
    for line in body.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            continue;
        }
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
            events.push(json);
        }
    }
    events
}

#[test]
fn responses_openapi_validator_supports_all_reachable_keywords() {
    let validator = OpenApiSchemaValidator::load_responses_spec();
    let unsupported = validator.unsupported_keywords_for_responses_surface();
    assert!(
        unsupported.is_empty(),
        "OpenAPI validator does not support all keywords used by /responses surface: {:?}",
        unsupported
    );
}

#[test]
fn responses_openapi_stream_event_registry_matches_path_schema() {
    let validator = OpenApiSchemaValidator::load_responses_spec();
    let spec = validator.spec();

    let refs = spec["paths"]["/responses"]["post"]["responses"]["200"]["content"]
        ["text/event-stream"]["schema"]["oneOf"]
        .as_array()
        .expect("event stream oneOf array");
    let mut from_path = BTreeSet::new();
    for entry in refs {
        let reference = entry["$ref"]
            .as_str()
            .expect("oneOf entry has $ref to event schema");
        let schema_name = reference
            .strip_prefix("#/components/schemas/")
            .expect("components schema ref");
        let schema = validator.schema_by_name(schema_name);
        let event_type = schema["properties"]["type"]["enum"]
            .as_array()
            .and_then(|a| a.first())
            .and_then(|v| v.as_str())
            .expect("event schema type enum");
        from_path.insert(event_type.to_string());
    }

    let mut from_components = BTreeSet::new();
    let schemas = spec["components"]["schemas"]
        .as_object()
        .expect("components.schemas object");
    for (name, schema) in schemas {
        if !(name.ends_with("StreamingEvent") || name == "ErrorStreamingEvent") {
            continue;
        }
        let Some(event_type) = schema["properties"]["type"]["enum"]
            .as_array()
            .and_then(|a| a.first())
            .and_then(|v| v.as_str())
        else {
            continue;
        };
        from_components.insert(event_type.to_string());
    }

    assert_eq!(
        from_path, from_components,
        "event types in /responses path oneOf and component streaming schemas diverged"
    );
}

#[test]
fn responses_openapi_non_stream_payload_matches_response_resource_contract() {
    let model = require_model!();
    let validator = OpenApiSchemaValidator::load_responses_spec();
    let schema = validator
        .spec()
        .pointer("/paths/~1responses/post/responses/200/content/application~1json/schema")
        .expect("responses 200 application/json schema");

    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": model,
        "input": "hello",
        "max_output_tokens": 16
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let payload = resp.json();

    if let Err(errors) = validator.validate_schema(schema, &payload) {
        panic!(
            "Response payload does not satisfy OpenResponses ResponseResource schema:\n{}\npayload={}",
            errors.join("\n"),
            payload
        );
    }
}

#[test]
fn responses_openapi_stream_payloads_match_event_contracts() {
    let model = require_model!();
    let validator = OpenApiSchemaValidator::load_responses_spec();

    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": model,
        "input": "hello",
        "stream": true,
        "max_output_tokens": 16
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let events = parse_sse_events(&resp.body);
    assert!(!events.is_empty(), "stream emitted no JSON events");

    let mut failures = Vec::new();
    for event in &events {
        if let Err(errors) = validator.validate_responses_stream_event_schema(event) {
            failures.push(format!(
                "event failed stream schema validation:\n{}\nevent={}",
                errors.join("\n"),
                event
            ));
        }

        if let Some(response) = event.get("response") {
            if let Err(errors) = validator.validate_named_schema("ResponseResource", response) {
                failures.push(format!(
                    "event.response failed ResponseResource validation:\n{}\nresponse={}",
                    errors.join("\n"),
                    response
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "OpenResponses stream payload contract failures ({}):\n{}",
        failures.len(),
        failures.join("\n")
    );
}
