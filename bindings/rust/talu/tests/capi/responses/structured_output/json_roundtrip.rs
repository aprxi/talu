//! Structured output serialization roundtrip tests.

use crate::capi::responses::common::ResponsesTestContext;
use talu::responses::{MessageRole, ResponsesView};

#[test]
fn tool_call_args_as_json() {
    let args = r#"{"temperature":72,"unit":"F"}"#;
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_function_call("c1", "get_weather", args).unwrap();

    let json = h.to_responses_json(1).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    let items = parsed.as_array().unwrap();
    let fc_item = &items[0];

    // The arguments field in the JSON should contain valid JSON
    let arguments_str = fc_item.get("arguments").and_then(|v| v.as_str());
    assert!(
        arguments_str.is_some(),
        "arguments should be present in JSON"
    );
    let parsed_args: Result<serde_json::Value, _> = serde_json::from_str(arguments_str.unwrap());
    assert!(
        parsed_args.is_ok(),
        "arguments should be valid JSON in output"
    );
}

#[test]
fn structured_response_completions_roundtrip() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_message(MessageRole::User, "Generate JSON")
        .unwrap();
    h.append_message(MessageRole::Assistant, r#"{"name":"Alice","age":30}"#)
        .unwrap();

    // Serialize to completions JSON
    let completions_json = h.to_completions_json().unwrap();

    // Reload into a fresh handle
    let mut h2 = talu::responses::ResponsesHandle::new().unwrap();
    h2.load_completions_json(&completions_json).unwrap();

    assert_eq!(h2.item_count(), 2, "reloaded should have 2 items");
    let text = h2.message_text(1).unwrap();
    assert_eq!(
        text, r#"{"name":"Alice","age":30}"#,
        "structured content should survive completions roundtrip"
    );
}
