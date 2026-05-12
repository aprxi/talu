//! Conversation clearing tests.

use crate::capi::responses::common::{
    assert_message_text, build_simple_conversation, ResponsesTestContext,
};
use talu::responses::{MessageRole, ResponsesView};

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
