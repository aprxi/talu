//! Multi-turn tool loop tests.

use crate::capi::responses::common::{
    assert_fc_fields, assert_item_type, assert_message_text, build_tool_loop, ResponsesTestContext,
};
use talu::responses::{ItemType, MessageRole, ResponsesView};

#[test]
fn tool_loop_conversation_shape() {
    let mut ctx = ResponsesTestContext::new();
    build_tool_loop(ctx.handle_mut());
    let h = ctx.handle();
    assert_eq!(h.item_count(), 5, "tool loop should have 5 items");

    assert_item_type(h, 0, ItemType::Message);
    assert_item_type(h, 1, ItemType::Message);
    assert_item_type(h, 2, ItemType::FunctionCall);
    assert_item_type(h, 3, ItemType::FunctionCallOutput);
    assert_item_type(h, 4, ItemType::Message);
}

#[test]
fn tool_loop_item_order() {
    let mut ctx = ResponsesTestContext::new();
    build_tool_loop(ctx.handle_mut());

    let types: Vec<_> = ctx.handle().items().map(|i| i.unwrap().item_type).collect();
    assert_eq!(
        types,
        vec![
            ItemType::Message,            // system
            ItemType::Message,            // user
            ItemType::FunctionCall,       // FC
            ItemType::FunctionCallOutput, // FCO
            ItemType::Message,            // assistant
        ]
    );
}

#[test]
fn tool_loop_fc_then_fco_content() {
    let mut ctx = ResponsesTestContext::new();
    build_tool_loop(ctx.handle_mut());
    let h = ctx.handle();

    let fc = h.get_function_call(2).unwrap();
    assert_eq!(fc.name, "get_weather");
    assert_eq!(fc.arguments, r#"{"location":"SF"}"#);

    let fco = h.get_function_call_output(3).unwrap();
    assert_eq!(fco.call_id, "call_abc");
    assert_eq!(fco.output_text.as_deref(), Some("Foggy, 58F"));
}

#[test]
fn tool_loop_final_assistant_text() {
    let mut ctx = ResponsesTestContext::new();
    build_tool_loop(ctx.handle_mut());

    let text = ctx.handle().last_assistant_message_text().unwrap();
    assert_eq!(text.as_deref(), Some("It's foggy and 58F in SF."));
}

#[test]
fn tool_loop_serialization() {
    let mut ctx = ResponsesTestContext::new();
    build_tool_loop(ctx.handle_mut());

    let json = ctx.handle().to_responses_json(1).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    let items = parsed.as_array().expect("JSON should be an array");
    assert_eq!(items.len(), 5, "array should contain all 5 items");
}

#[test]
fn tool_loop_clear_keeping_system() {
    let mut ctx = ResponsesTestContext::new();
    build_tool_loop(ctx.handle_mut());
    assert_eq!(ctx.handle().item_count(), 5);

    ctx.handle_mut().clear_keeping_system();
    assert_eq!(
        ctx.handle().item_count(),
        1,
        "only system message should remain"
    );
    assert_message_text(ctx.handle(), 0, "You are a helpful assistant with tools.");
}

#[test]
fn two_tool_turns() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_message(MessageRole::User, "Weather and time?")
        .unwrap();
    // First tool turn
    h.append_function_call("c1", "get_weather", r#"{"loc":"NYC"}"#)
        .unwrap();
    h.append_function_call_output("c1", "Sunny").unwrap();
    // Second tool turn
    h.append_function_call("c2", "get_time", r#"{"tz":"EST"}"#)
        .unwrap();
    h.append_function_call_output("c2", "3:00 PM").unwrap();
    h.append_message(MessageRole::Assistant, "NYC is sunny at 3:00 PM.")
        .unwrap();

    assert_eq!(h.item_count(), 6);

    assert_fc_fields(h, 1, "get_weather", "c1");
    assert_fc_fields(h, 3, "get_time", "c2");
}
