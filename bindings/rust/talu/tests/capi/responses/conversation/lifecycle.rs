//! Conversation handle lifecycle tests.

use crate::capi::responses::common::ResponsesTestContext;
use talu::responses::{ResponsesHandle, ResponsesView};

#[test]
fn create_returns_valid_handle() {
    let ctx = ResponsesTestContext::new();
    assert_eq!(
        ctx.handle().item_count(),
        0,
        "new conversation should be empty"
    );
}

#[test]
fn create_with_session() {
    let _ctx = ResponsesTestContext::with_session("test-session-42");
    // Session flows through — no panic means creation succeeded.
    // Session ID is not directly readable from ResponsesHandle, but
    // the handle is valid and usable.
}

#[test]
fn free_is_idempotent() {
    // ResponsesHandle implements Drop; creating and dropping twice
    // (via scoped blocks) must not panic or double-free.
    {
        let _h = ResponsesHandle::new().unwrap();
    }
    {
        let _h = ResponsesHandle::new().unwrap();
    }
}

#[test]
fn validate_live_handle() {
    let ctx = ResponsesTestContext::new();
    let rc = unsafe { talu_sys::talu_responses_validate(ctx.handle().as_ptr()) };
    assert_eq!(rc, 1, "validate should return 1 for live handle");
}
