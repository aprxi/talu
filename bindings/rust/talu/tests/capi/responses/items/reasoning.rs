//! Reasoning item variant tests.
//!
//! Reasoning items are created via `load_responses_json` (no direct append API).

use crate::capi::responses::common::ResponsesTestContext;
use talu::responses::{ContentType, MessageRole, ResponsesView};

#[test]
fn reasoning_summary_parts() {
    let mut ctx = ResponsesTestContext::new();
    ctx.handle_mut()
        .load_responses_json(
            r#"[{"type":"reasoning","summary":[{"type":"summary_text","text":"First summary"},{"type":"summary_text","text":"Second summary"}],"content":[]}]"#,
        )
        .unwrap();

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
    ctx.handle_mut()
        .load_responses_json(
            r#"[{"type":"reasoning","summary":[{"type":"summary_text","text":"Part A"},{"type":"summary_text","text":"Part B"}],"content":[]}]"#,
        )
        .unwrap();

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
