//! Item ordering, truncate_after, clone, and clone_prefix tests.

use crate::capi::responses::common::{
    assert_message_text, RawResponsesHandle, ResponsesTestContext,
};
use talu::responses::MessageRole;

#[test]
fn items_maintain_insertion_order() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    let texts = ["alpha", "beta", "gamma", "delta"];
    for t in &texts {
        h.append_message(MessageRole::User, t).unwrap();
    }
    for (i, expected) in texts.iter().enumerate() {
        assert_message_text(h, i, expected);
    }
}

#[test]
fn truncate_after_keeps_prefix() {
    let mut h = RawResponsesHandle::new();
    for i in 0..5 {
        h.append_message(talu_sys::MessageRole::User, &format!("msg_{}", i));
    }
    assert_eq!(h.item_count(), 5);

    let rc = unsafe { talu_sys::talu_responses_truncate_after(h.as_ptr(), 2) };
    assert_eq!(rc, 0, "truncate_after should succeed");
    assert_eq!(h.item_count(), 3, "should keep items 0..=2");
    assert_eq!(h.message_text(0), "msg_0");
    assert_eq!(h.message_text(1), "msg_1");
    assert_eq!(h.message_text(2), "msg_2");
}

#[test]
fn clone_copies_all_items() {
    let mut source = RawResponsesHandle::new();
    for i in 0..3 {
        source.append_message(talu_sys::MessageRole::User, &format!("User turn {}", i));
        source.append_message(
            talu_sys::MessageRole::Assistant,
            &format!("Assistant turn {}", i),
        );
    }
    assert_eq!(source.item_count(), 6);

    let dest = RawResponsesHandle::new();
    let rc = unsafe { talu_sys::talu_responses_clone(dest.as_ptr(), source.as_ptr(), 0) };
    assert_eq!(rc, 0, "clone should succeed");
    assert_eq!(dest.item_count(), 6, "clone should copy all items");

    // Verify content is independent
    for i in 0..6 {
        let orig = source.message_text(i);
        let cloned = dest.message_text(i);
        assert_eq!(orig, cloned, "cloned item {} text should match", i);
    }
}

#[test]
fn clone_prefix_partial() {
    let mut source = RawResponsesHandle::new();
    for i in 0..3 {
        source.append_message(talu_sys::MessageRole::User, &format!("User turn {}", i));
        source.append_message(
            talu_sys::MessageRole::Assistant,
            &format!("Assistant turn {}", i),
        );
    }
    assert_eq!(source.item_count(), 6);

    let dest = RawResponsesHandle::new();
    // Clone only items 0..=2 (first 3)
    let rc = unsafe { talu_sys::talu_responses_clone_prefix(dest.as_ptr(), source.as_ptr(), 2, 0) };
    assert_eq!(rc, 0, "clone_prefix should succeed");
    assert_eq!(dest.item_count(), 3, "clone_prefix should copy only prefix");
}
