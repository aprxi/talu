//! Parallel tool call tests.

use crate::capi::responses::common::{
    assert_fc_fields, assert_item_type, build_with_parallel_calls, ResponsesTestContext,
};
use talu::responses::{ItemType, MessageRole, ResponsesView};

#[test]
fn parallel_two_calls() {
    let mut ctx = ResponsesTestContext::new();
    build_with_parallel_calls(ctx.handle_mut(), 2);
    let h = ctx.handle();

    // user + FC_0 + FC_1 + FCO_0 + FCO_1 + assistant = 6
    assert_eq!(h.item_count(), 6);

    assert_item_type(h, 0, ItemType::Message);
    assert_item_type(h, 1, ItemType::FunctionCall);
    assert_item_type(h, 2, ItemType::FunctionCall);
    assert_item_type(h, 3, ItemType::FunctionCallOutput);
    assert_item_type(h, 4, ItemType::FunctionCallOutput);
    assert_item_type(h, 5, ItemType::Message);

    let fc0 = h.get_function_call(1).unwrap();
    let fco0 = h.get_function_call_output(3).unwrap();
    assert_eq!(fc0.call_id, "call_0");
    assert_eq!(fco0.call_id, "call_0");

    let fc1 = h.get_function_call(2).unwrap();
    let fco1 = h.get_function_call_output(4).unwrap();
    assert_eq!(fc1.call_id, "call_1");
    assert_eq!(fco1.call_id, "call_1");
}

#[test]
fn parallel_three_calls() {
    let mut ctx = ResponsesTestContext::new();
    build_with_parallel_calls(ctx.handle_mut(), 3);
    let h = ctx.handle();

    // user + 3*FC + 3*FCO + assistant = 8
    assert_eq!(h.item_count(), 8);

    for i in 0..3 {
        let fc = h.get_function_call(1 + i).unwrap();
        let fco = h.get_function_call_output(4 + i).unwrap();
        let expected_id = format!("call_{}", i);
        assert_eq!(fc.call_id, expected_id, "FC_{i} call_id mismatch");
        assert_eq!(fco.call_id, expected_id, "FCO_{i} call_id mismatch");
    }
}

#[test]
fn parallel_calls_different_functions() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_message(MessageRole::User, "weather and time")
        .unwrap();
    h.append_function_call("c_w", "get_weather", r#"{"loc":"NYC"}"#)
        .unwrap();
    h.append_function_call("c_t", "get_time", r#"{"tz":"EST"}"#)
        .unwrap();
    h.append_function_call_output("c_w", "Sunny").unwrap();
    h.append_function_call_output("c_t", "3pm").unwrap();

    assert_fc_fields(h, 1, "get_weather", "c_w");
    assert_fc_fields(h, 2, "get_time", "c_t");
}

#[test]
fn parallel_calls_serialization() {
    let mut ctx = ResponsesTestContext::new();
    build_with_parallel_calls(ctx.handle_mut(), 2);

    let json = ctx.handle().to_responses_json(1).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    let items = parsed.as_array().expect("JSON should be an array");

    // Count function_call and function_call_output items in JSON
    let fc_count = items
        .iter()
        .filter(|item| item.get("type").and_then(|t| t.as_str()) == Some("function_call"))
        .count();
    let fco_count = items
        .iter()
        .filter(|item| item.get("type").and_then(|t| t.as_str()) == Some("function_call_output"))
        .count();
    assert_eq!(fc_count, 2, "should have 2 FCs in JSON");
    assert_eq!(fco_count, 2, "should have 2 FCOs in JSON");
}
