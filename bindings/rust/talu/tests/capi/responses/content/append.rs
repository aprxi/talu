//! Streaming text accumulation (append_text_content) tests.

use crate::capi::responses::common::ResponsesTestContext;
use talu::responses::{MessageRole, ResponsesView};

#[test]
fn append_text_builds_content() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_message(MessageRole::Assistant, "Hello").unwrap();
    h.append_text_content(0, " World").unwrap();

    let text = h.message_text(0).unwrap();
    assert_eq!(
        text, "Hello World",
        "append_text_content should build content incrementally"
    );
}

#[test]
fn append_text_multiple_calls() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_message(MessageRole::Assistant, "A").unwrap();
    h.append_text_content(0, "B").unwrap();
    h.append_text_content(0, "C").unwrap();

    let text = h.message_text(0).unwrap();
    assert_eq!(text, "ABC", "multiple appends should concatenate");
    // Content count stays at 1 — appending extends the existing part
    let count = h.message_content_count(0);
    assert_eq!(
        count, 1,
        "append_text_content should extend, not add new parts"
    );
}

#[test]
fn append_text_to_non_message_errors() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_function_call("c1", "f1", "{}").unwrap();
    let result = h.append_text_content(0, "oops");
    assert!(
        result.is_err(),
        "append_text_content on function_call should error"
    );
}
