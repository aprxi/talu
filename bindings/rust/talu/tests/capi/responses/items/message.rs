//! Message item variant tests.

use crate::capi::responses::common::{
    assert_item_type, build_simple_conversation, ResponsesTestContext,
};
use talu::responses::{ItemType, MessageRole, ResponsesView};

#[test]
fn message_role_roundtrip() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_message(MessageRole::User, "hello").unwrap();
    let msg = h.get_message(0).unwrap();
    assert_eq!(msg.role, MessageRole::User, "role should round-trip");
}

#[test]
fn message_content_count() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_message(MessageRole::User, "initial text").unwrap();
    assert_eq!(
        h.message_content_count(0),
        1,
        "should have 1 content part after append"
    );

    h.append_text_content(0, " more text").unwrap();
    // append_text_content appends to the existing text part, count stays 1
    let count = h.message_content_count(0);
    assert!(
        count >= 1,
        "content count should be at least 1 after append_text_content"
    );
}

#[test]
fn message_text_concatenates_parts() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_message(MessageRole::Assistant, "Hello").unwrap();
    h.append_text_content(0, " World").unwrap();

    let text = h.message_text(0).unwrap();
    assert_eq!(
        text, "Hello World",
        "message_text should concatenate all text parts"
    );
}

#[test]
fn get_message_on_non_message_errors() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_function_call("call_1", "func", "{}").unwrap();
    let result = h.get_message(0);
    assert!(result.is_err(), "get_message on function_call should error");
}

#[test]
fn get_item_returns_header() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_message(MessageRole::User, "test").unwrap();
    let item = h.get_item(0).unwrap();
    assert_eq!(item.item_type, ItemType::Message);
    // id is assigned by core; just verify it's set
    // created_at_ms should be a reasonable timestamp or zero
}

#[test]
fn item_usage_stats() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_message(MessageRole::User, "test").unwrap();
    let item = h.get_item(0).unwrap();
    // For manually appended items, token counts default to 0
    assert_eq!(
        item.input_tokens, 0,
        "input_tokens should be 0 for manual items"
    );
    assert_eq!(
        item.output_tokens, 0,
        "output_tokens should be 0 for manual items"
    );
}

#[test]
fn item_performance_metrics() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_message(MessageRole::Assistant, "response")
        .unwrap();
    let item = h.get_item(0).unwrap();
    // For manually appended items, timing metrics default to 0
    assert_eq!(
        item.prefill_ns, 0,
        "prefill_ns should be 0 for manual items"
    );
    assert_eq!(
        item.generation_ns, 0,
        "generation_ns should be 0 for manual items"
    );
}

#[test]
fn item_finish_reason() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_message(MessageRole::Assistant, "done").unwrap();
    let item = h.get_item(0).unwrap();
    // For manually appended items, finish_reason is None
    assert!(
        item.finish_reason.is_none(),
        "finish_reason should be None for manual items"
    );
}

#[test]
fn item_type_discriminator() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_message(MessageRole::User, "msg").unwrap();
    h.append_function_call("c1", "fn1", "{}").unwrap();
    h.append_function_call_output("c1", "out").unwrap();

    assert_item_type(h, 0, ItemType::Message);
    assert_item_type(h, 1, ItemType::FunctionCall);
    assert_item_type(h, 2, ItemType::FunctionCallOutput);
}

#[test]
fn last_assistant_message_text() {
    let mut ctx = ResponsesTestContext::new();
    build_simple_conversation(ctx.handle_mut());

    let text = ctx.handle().last_assistant_message_text().unwrap();
    assert_eq!(
        text.as_deref(),
        Some("Hi there!"),
        "should find last assistant message"
    );
}

#[test]
fn last_assistant_message_text_none() {
    let mut ctx = ResponsesTestContext::new();
    ctx.handle_mut()
        .append_message(MessageRole::User, "only user")
        .unwrap();
    let text = ctx.handle().last_assistant_message_text().unwrap();
    assert!(
        text.is_none(),
        "should return None when no assistant message"
    );
}
