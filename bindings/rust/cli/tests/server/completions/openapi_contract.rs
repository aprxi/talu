//! OpenAPI contract checks for `/v1/chat/completions`.

use crate::server::common::{get, ServerConfig, ServerTestContext};
use std::collections::BTreeSet;
use std::sync::OnceLock;

fn load_chat_spec() -> serde_json::Value {
    static SPEC: OnceLock<serde_json::Value> = OnceLock::new();
    SPEC.get_or_init(|| {
        let ctx = ServerTestContext::new(ServerConfig::new());
        let resp = get(ctx.addr(), "/openapi/chat.json");
        assert_eq!(
            resp.status, 200,
            "failed to fetch /openapi/chat.json: {}",
            resp.body
        );
        serde_json::from_str(&resp.body).expect("parse /openapi/chat.json")
    })
    .clone()
}

fn chat_post(spec: &serde_json::Value) -> &serde_json::Value {
    spec.pointer("/paths/~1v1~1chat~1completions/post")
        .expect("missing /v1/chat/completions POST in chat spec")
}

#[test]
fn chat_openapi_is_chat_scoped_only() {
    let spec = load_chat_spec();
    let paths = spec["paths"].as_object().expect("paths object");
    assert!(!paths.is_empty(), "chat paths must not be empty");
    assert!(
        paths.keys().all(|k| k.starts_with("/v1/chat/")),
        "chat spec must only expose /v1/chat/* paths, got: {:?}",
        paths.keys().collect::<Vec<_>>()
    );
}

#[test]
fn chat_openapi_has_post_operation_and_standard_responses() {
    let spec = load_chat_spec();
    let post = chat_post(&spec);
    assert!(
        post.get("requestBody").is_some(),
        "chat completions must define requestBody"
    );
    let responses = post["responses"].as_object().expect("responses object");
    for code in ["200", "400", "500"] {
        assert!(
            responses.contains_key(code),
            "chat completions missing response {code}"
        );
    }
}

#[test]
fn chat_openapi_create_request_fields_are_exhaustively_covered() {
    let spec = load_chat_spec();
    let props = spec["components"]["schemas"]["CreateChatCompletionBody"]["properties"]
        .as_object()
        .expect("CreateChatCompletionBody.properties object");
    let actual: BTreeSet<String> = props.keys().cloned().collect();
    let expected: BTreeSet<String> = [
        "model",
        "messages",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "stream",
        "seed",
        "presence_penalty",
        "frequency_penalty",
        "tools",
        "tool_choice",
        "max_completion_tokens",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();

    assert_eq!(
        actual, expected,
        "CreateChatCompletionBody field drift; update tests and runtime together"
    );
}

#[test]
fn chat_openapi_create_request_requires_messages() {
    let spec = load_chat_spec();
    let required = spec["components"]["schemas"]["CreateChatCompletionBody"]["required"]
        .as_array()
        .expect("CreateChatCompletionBody.required array");
    let required: BTreeSet<String> = required
        .iter()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();
    assert!(
        required.contains("messages"),
        "messages must be required in CreateChatCompletionBody"
    );
}

#[test]
fn chat_openapi_create_request_disallows_unknown_fields() {
    let spec = load_chat_spec();
    let schema = &spec["components"]["schemas"]["CreateChatCompletionBody"];
    assert_eq!(
        schema.get("additionalProperties"),
        Some(&serde_json::Value::Bool(false)),
        "CreateChatCompletionBody must set additionalProperties=false"
    );
}

#[test]
fn chat_openapi_response_schema_includes_core_fields() {
    let spec = load_chat_spec();
    let chat_completion = &spec["components"]["schemas"]["ChatCompletion"]["properties"];
    let props = chat_completion
        .as_object()
        .expect("ChatCompletion.properties object");
    for key in ["id", "object", "created", "model", "choices", "usage"] {
        assert!(
            props.contains_key(key),
            "ChatCompletion schema missing {key}"
        );
    }

    let chunk_props = spec["components"]["schemas"]["ChatCompletionChunk"]["properties"]
        .as_object()
        .expect("ChatCompletionChunk.properties object");
    for key in ["id", "object", "created", "model", "choices"] {
        assert!(
            chunk_props.contains_key(key),
            "ChatCompletionChunk schema missing {key}"
        );
    }
}
