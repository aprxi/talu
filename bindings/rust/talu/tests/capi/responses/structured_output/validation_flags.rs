//! Validation flag get/set tests.
//!
//! These tests exercise `set_item_validation_flags` directly through talu_sys
//! because the safe Rust API does not yet expose validation flag operations.

use crate::capi::responses::common::ResponsesTestContext;
use talu::responses::{MessageRole, ResponsesView};

/// Helper: set validation flags on an item.
fn set_flags(
    handle: *mut talu_sys::ResponsesHandle,
    index: usize,
    json_valid: bool,
    schema_valid: bool,
    repaired: bool,
) -> i32 {
    unsafe {
        talu_sys::talu_responses_set_item_validation_flags(
            handle,
            index,
            json_valid,
            schema_valid,
            repaired,
        )
    }
}

#[test]
fn set_and_get_validation_flags() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_message(MessageRole::Assistant, "output").unwrap();

    let rc = set_flags(h.as_ptr(), 0, true, true, false);
    assert_eq!(rc, 0, "set_item_validation_flags should succeed");
}

#[test]
fn json_valid_only() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_message(MessageRole::Assistant, "output").unwrap();

    let rc = set_flags(h.as_ptr(), 0, true, false, false);
    assert_eq!(rc, 0, "setting json_valid alone should succeed");
}

#[test]
fn schema_valid_implies_json_valid() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_message(MessageRole::Assistant, "output").unwrap();

    let rc = set_flags(h.as_ptr(), 0, true, true, false);
    assert_eq!(rc, 0, "both json_valid and schema_valid should succeed");
}

#[test]
fn repaired_flag() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_message(MessageRole::Assistant, "output").unwrap();

    let rc = set_flags(h.as_ptr(), 0, true, true, true);
    assert_eq!(rc, 0, "setting repaired flag should succeed");
}

#[test]
fn flags_on_assistant_message() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_message(MessageRole::Assistant, "structured output")
        .unwrap();
    let rc = set_flags(h.as_ptr(), 0, true, false, false);
    assert_eq!(rc, 0, "flags should work on assistant messages");
}

#[test]
fn flags_on_function_call() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_function_call("c1", "fn1", r#"{"key":"value"}"#)
        .unwrap();
    let rc = set_flags(h.as_ptr(), 0, true, true, false);
    assert_eq!(rc, 0, "flags should work on function_call items");
}

#[test]
fn flags_out_of_bounds() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_message(MessageRole::User, "only item").unwrap();
    let rc = set_flags(h.as_ptr(), 99, true, false, false);
    assert_ne!(rc, 0, "out-of-bounds should return error");
}
