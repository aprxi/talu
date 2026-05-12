//! Item append, insert, remove, and pop tests.

use crate::capi::responses::common::{
    assert_item_type, assert_message_text, RawResponsesHandle, ResponsesTestContext,
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

#[test]
fn append_message_rejects_invalid_raw_role() {
    let h = RawResponsesHandle::new();
    let before = h.item_count();
    let content = b"invalid role";

    let result = unsafe {
        talu_sys::talu_responses_append_message(h.as_ptr(), 99, content.as_ptr(), content.len())
    };

    assert!(result < 0, "invalid role must return a negative error code");
    assert_eq!(
        h.item_count(),
        before,
        "failed append must not mutate the conversation"
    );
}

#[test]
fn insert_message_rejects_invalid_raw_role() {
    let h = RawResponsesHandle::new();
    let content = b"invalid role";

    let result = unsafe {
        talu_sys::talu_responses_insert_message(h.as_ptr(), 0, 99, content.as_ptr(), content.len())
    };

    assert!(result < 0, "invalid role must return a negative error code");
    assert_eq!(
        h.item_count(),
        0,
        "failed insert must not mutate the conversation"
    );
}
