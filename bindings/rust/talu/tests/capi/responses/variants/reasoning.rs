//! Reasoning item variant tests.
//!
//! Reasoning items are created by the core during generation (no direct
//! `append_*` API). Tests use `talu_responses_load_storage_records` to
//! construct conversations containing reasoning items, then validate
//! the read accessors.

use crate::capi::responses::common::ResponsesTestContext;
use std::ffi::CString;
use talu::responses::{ContentType, ItemType, MessageRole, ResponsesView};

/// Helper: load a single reasoning storage record into the context.
///
/// Constructs a `CStorageRecord` with `item_type=3` (reasoning) and
/// the given `content_json`, then calls the C API load function.
fn load_reasoning_record(ctx: &mut ResponsesTestContext, content_json: &str) {
    let c_json = CString::new(content_json).unwrap();
    let record = talu_sys::CStorageRecord {
        item_type: 3, // reasoning
        status: 2,    // completed
        content_json: c_json.as_ptr(),
        ..Default::default()
    };
    let rc = unsafe {
        talu_sys::talu_responses_load_storage_records(ctx.handle().as_ptr(), &record as *const _, 1)
    };
    assert_eq!(rc, 0, "load_storage_records failed with rc={rc}");
}

#[test]
fn reasoning_content_parts() {
    let mut ctx = ResponsesTestContext::new();
    load_reasoning_record(
        &mut ctx,
        r#"{"type":"reasoning","summary":[{"type":"summary_text","text":"sum"}],"content":[{"type":"reasoning_text","text":"Let me think..."}]}"#,
    );

    assert_eq!(ctx.handle().item_count(), 1);
    assert_eq!(ctx.handle().item_type(0), ItemType::Reasoning);

    let r = ctx.handle().get_reasoning(0).unwrap();
    assert_eq!(r.content_count, 1, "should have 1 content part");

    let part = ctx.handle().get_reasoning_content(0, 0).unwrap();
    assert_eq!(part.content_type, ContentType::ReasoningText);
    assert_eq!(part.data, b"Let me think...");
}

#[test]
fn reasoning_summary_parts() {
    let mut ctx = ResponsesTestContext::new();
    load_reasoning_record(
        &mut ctx,
        r#"{"type":"reasoning","summary":[{"type":"summary_text","text":"First summary"},{"type":"summary_text","text":"Second summary"}],"content":[]}"#,
    );

    let r = ctx.handle().get_reasoning(0).unwrap();
    assert_eq!(r.summary_count, 2, "should have 2 summary parts");

    let s0 = ctx.handle().get_reasoning_summary(0, 0).unwrap();
    assert_eq!(s0.content_type, ContentType::SummaryText);
    assert_eq!(s0.data, b"First summary");

    let s1 = ctx.handle().get_reasoning_summary(0, 1).unwrap();
    assert_eq!(s1.data, b"Second summary");
}

#[test]
fn reasoning_summary_text() {
    let mut ctx = ResponsesTestContext::new();
    load_reasoning_record(
        &mut ctx,
        r#"{"type":"reasoning","summary":[{"type":"summary_text","text":"Part A"},{"type":"summary_text","text":"Part B"}],"content":[]}"#,
    );

    let text = ctx.handle().reasoning_summary_text(0).unwrap();
    assert_eq!(
        text, "Part APart B",
        "summary text should concatenate all parts"
    );
}

#[test]
fn get_reasoning_on_message_errors() {
    let mut ctx = ResponsesTestContext::new();
    ctx.handle_mut()
        .append_message(MessageRole::User, "not reasoning")
        .unwrap();
    let result = ctx.handle().get_reasoning(0);
    assert!(
        result.is_err(),
        "get_reasoning on message should return error"
    );
}
