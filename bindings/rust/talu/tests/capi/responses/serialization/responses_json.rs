//! Open Responses JSON format serialization tests.
//!
//! `to_responses_json` returns a JSON array of items (not a wrapped object).

use crate::capi::responses::common::{
    build_simple_conversation, build_with_function_call, ResponsesTestContext,
};
use talu::responses::{MessageRole, ResponsesView};

#[test]
fn serialize_request_direction() {
    let mut ctx = ResponsesTestContext::new();
    build_simple_conversation(ctx.handle_mut());

    let json = ctx.handle().to_responses_json(0).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("should be valid JSON");
    assert!(
        parsed.is_array(),
        "request direction JSON should be an array"
    );
}

#[test]
fn serialize_response_direction() {
    let mut ctx = ResponsesTestContext::new();
    build_simple_conversation(ctx.handle_mut());

    let json = ctx.handle().to_responses_json(1).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("should be valid JSON");
    assert!(
        parsed.is_array(),
        "response direction JSON should be an array"
    );
}

#[test]
fn serialize_empty_conversation() {
    let ctx = ResponsesTestContext::new();
    let json = ctx.handle().to_responses_json(1).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("should be valid JSON");
    let arr = parsed.as_array().expect("should be an array");
    assert_eq!(
        arr.len(),
        0,
        "empty conversation should serialize to empty array"
    );
}

#[test]
fn serialize_multi_item_types() {
    let mut ctx = ResponsesTestContext::new();
    build_with_function_call(ctx.handle_mut());

    let json = ctx.handle().to_responses_json(1).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    let items = parsed.as_array().unwrap();

    let types: Vec<_> = items
        .iter()
        .filter_map(|item| item.get("type").and_then(|t| t.as_str()))
        .collect();
    assert!(types.contains(&"message"), "should have message items");
    assert!(
        types.contains(&"function_call"),
        "should have function_call items"
    );
    assert!(
        types.contains(&"function_call_output"),
        "should have function_call_output items"
    );
}

#[test]
fn serialize_preserves_message_text() {
    let mut ctx = ResponsesTestContext::new();
    ctx.handle_mut()
        .append_message(MessageRole::User, "Preserved text content")
        .unwrap();

    let json = ctx.handle().to_responses_json(1).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    let items = parsed.as_array().unwrap();
    let msg = &items[0];
    let content = msg.get("content").and_then(|v| v.as_array()).unwrap();
    let text = content[0].get("text").and_then(|t| t.as_str());
    assert_eq!(
        text,
        Some("Preserved text content"),
        "text content should be preserved in JSON"
    );
}

#[test]
fn serialize_function_call_fields() {
    let mut ctx = ResponsesTestContext::new();
    ctx.handle_mut()
        .append_function_call("call_x", "my_func", r#"{"key":"val"}"#)
        .unwrap();

    let json = ctx.handle().to_responses_json(1).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    let items = parsed.as_array().unwrap();
    let fc = &items[0];

    assert_eq!(
        fc.get("name").and_then(|v| v.as_str()),
        Some("my_func"),
        "name should be in JSON"
    );
    assert_eq!(
        fc.get("call_id").and_then(|v| v.as_str()),
        Some("call_x"),
        "call_id should be in JSON"
    );
    assert!(fc.get("arguments").is_some(), "arguments should be in JSON");
}

#[test]
fn response_direction_keeps_all() {
    let mut ctx = ResponsesTestContext::new();
    build_simple_conversation(ctx.handle_mut());

    let json = ctx.handle().to_responses_json(1).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    let items = parsed.as_array().unwrap();
    assert_eq!(items.len(), 3, "response direction should keep all items");
}
