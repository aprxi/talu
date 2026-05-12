//! Item status transition tests.

use crate::capi::responses::common::{RawResponsesHandle, ResponsesTestContext};
use talu::responses::{ItemStatus, MessageRole, ResponsesView};

#[test]
fn set_item_status_roundtrips_all_statuses() {
    let mut ctx = ResponsesTestContext::new();
    ctx.handle_mut()
        .append_message(MessageRole::Assistant, "pending")
        .unwrap();

    for status in [
        ItemStatus::InProgress,
        ItemStatus::Waiting,
        ItemStatus::Completed,
        ItemStatus::Incomplete,
        ItemStatus::Failed,
    ] {
        ctx.handle_mut().set_item_status(0, status).unwrap();
        let item = ctx.handle().get_item(0).unwrap();
        assert_eq!(item.status, status, "status should roundtrip");
    }
}

#[test]
fn set_item_status_out_of_bounds_errors() {
    let mut ctx = ResponsesTestContext::new();
    let result = ctx.handle_mut().set_item_status(0, ItemStatus::Completed);
    assert!(result.is_err(), "status update must reject missing item");
}

#[test]
fn set_item_status_rejects_invalid_raw_status() {
    let mut h = RawResponsesHandle::new();
    h.append_message(talu_sys::MessageRole::User, "hello");

    let mut before = talu_sys::CItem::default();
    let rc = unsafe { talu_sys::talu_responses_get_item(h.as_ptr(), 0, &mut before) };
    assert_eq!(rc, 0, "item should be readable before invalid update");

    let rc = unsafe { talu_sys::talu_responses_set_item_status(h.as_ptr(), 0, 99) };
    assert_ne!(rc, 0, "invalid raw status must fail");

    let mut item = talu_sys::CItem::default();
    let rc = unsafe { talu_sys::talu_responses_get_item(h.as_ptr(), 0, &mut item) };
    assert_eq!(rc, 0, "item should remain readable");
    assert_eq!(
        ItemStatus::from(item.status),
        ItemStatus::from(before.status),
        "failed status update must not mutate item"
    );
}

#[test]
fn set_item_status_rejects_null_handle() {
    let rc = unsafe {
        talu_sys::talu_responses_set_item_status(
            std::ptr::null_mut(),
            0,
            ItemStatus::Completed as u8,
        )
    };
    assert_ne!(rc, 0, "null handle must fail");
}
