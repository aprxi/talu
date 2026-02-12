//! Item append/insert/remove/clear tests.

use crate::capi::responses::common::{
    assert_item_type, assert_message_text, build_simple_conversation, ResponsesTestContext,
};
use talu::responses::{ItemType, MessageRole, ResponsesView};

#[test]
fn append_message_increments_count() {
    let mut ctx = ResponsesTestContext::new();
    assert_eq!(ctx.handle().item_count(), 0);
    ctx.handle_mut()
        .append_message(MessageRole::User, "Hello")
        .unwrap();
    assert_eq!(
        ctx.handle().item_count(),
        1,
        "count should be 1 after append"
    );
}

#[test]
fn append_all_roles() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_message(MessageRole::System, "sys").unwrap();
    h.append_message(MessageRole::User, "usr").unwrap();
    h.append_message(MessageRole::Assistant, "asst").unwrap();
    h.append_message(MessageRole::Developer, "dev").unwrap();
    assert_eq!(h.item_count(), 4);

    let roles: Vec<_> = (0..4).map(|i| h.get_message(i).unwrap().role).collect();
    assert_eq!(
        roles,
        vec![
            MessageRole::System,
            MessageRole::User,
            MessageRole::Assistant,
            MessageRole::Developer,
        ]
    );
}

#[test]
fn insert_message_at_index() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_message(MessageRole::User, "first").unwrap();
    h.append_message(MessageRole::Assistant, "third").unwrap();
    h.insert_message(1, MessageRole::User, "second").unwrap();

    assert_eq!(h.item_count(), 3);
    assert_message_text(h, 0, "first");
    assert_message_text(h, 1, "second");
    assert_message_text(h, 2, "third");
}

#[test]
fn append_function_call() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_function_call("call_1", "get_weather", r#"{"loc":"NYC"}"#)
        .unwrap();
    assert_eq!(h.item_count(), 1);
    assert_item_type(h, 0, ItemType::FunctionCall);

    let fc = h.get_function_call(0).unwrap();
    assert_eq!(fc.name, "get_weather");
    assert_eq!(fc.call_id, "call_1");
    assert_eq!(fc.arguments, r#"{"loc":"NYC"}"#);
}

#[test]
fn append_function_call_output() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_function_call_output("call_1", "Sunny, 72F")
        .unwrap();
    assert_eq!(h.item_count(), 1);
    assert_item_type(h, 0, ItemType::FunctionCallOutput);

    let fco = h.get_function_call_output(0).unwrap();
    assert_eq!(fco.call_id, "call_1");
    assert_eq!(fco.output_text.as_deref(), Some("Sunny, 72F"));
}

#[test]
fn pop_removes_last() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_message(MessageRole::User, "first").unwrap();
    h.append_message(MessageRole::Assistant, "second").unwrap();
    assert_eq!(h.item_count(), 2);

    h.pop().unwrap();
    assert_eq!(h.item_count(), 1);
    assert_message_text(h, 0, "first");
}

#[test]
fn remove_at_index() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_message(MessageRole::User, "A").unwrap();
    h.append_message(MessageRole::User, "B").unwrap();
    h.append_message(MessageRole::User, "C").unwrap();

    h.remove(1).unwrap();
    assert_eq!(h.item_count(), 2);
    assert_message_text(h, 0, "A");
    assert_message_text(h, 1, "C");
}

#[test]
fn clear_empties_conversation() {
    let mut ctx = ResponsesTestContext::new();
    build_simple_conversation(ctx.handle_mut());
    assert_eq!(ctx.handle().item_count(), 3);

    ctx.handle_mut().clear();
    assert_eq!(
        ctx.handle().item_count(),
        0,
        "clear should remove all items"
    );
}

#[test]
fn clear_keeping_system_preserves_first() {
    let mut ctx = ResponsesTestContext::new();
    build_simple_conversation(ctx.handle_mut());
    assert_eq!(ctx.handle().item_count(), 3);

    ctx.handle_mut().clear_keeping_system();
    assert_eq!(
        ctx.handle().item_count(),
        1,
        "should keep only the system message"
    );
    assert_message_text(ctx.handle(), 0, "You are a test assistant.");
}

#[test]
fn clear_keeping_system_no_system() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_message(MessageRole::User, "user msg").unwrap();
    h.append_message(MessageRole::Assistant, "asst msg")
        .unwrap();

    h.clear_keeping_system();
    assert_eq!(
        h.item_count(),
        0,
        "all items cleared when first is not system"
    );
}

#[test]
fn append_to_empty_after_clear() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_message(MessageRole::User, "before").unwrap();
    h.clear();
    assert_eq!(h.item_count(), 0);

    h.append_message(MessageRole::User, "after").unwrap();
    assert_eq!(h.item_count(), 1);
    assert_message_text(h, 0, "after");
}

#[test]
fn pop_empty_is_error() {
    let mut ctx = ResponsesTestContext::new();
    let result = ctx.handle_mut().pop();
    assert!(result.is_err(), "pop on empty should return error");
}

#[test]
fn remove_out_of_bounds() {
    let mut ctx = ResponsesTestContext::new();
    ctx.handle_mut()
        .append_message(MessageRole::User, "only item")
        .unwrap();
    let result = ctx.handle_mut().remove(5);
    assert!(result.is_err(), "remove out of bounds should return error");
}
