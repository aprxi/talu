//! Text content part tests.

use crate::capi::responses::common::{is_text_type, ResponsesTestContext};
use talu::responses::{ContentType, MessageRole, ResponsesView};

#[test]
fn text_content_data() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_message(MessageRole::User, "Hello World").unwrap();

    let part = h.get_message_content(0, 0).unwrap();
    assert!(
        is_text_type(part.content_type),
        "content_type should be a text type, got {:?}",
        part.content_type
    );
    assert_eq!(
        part.data_str(),
        Some("Hello World"),
        "data should match appended text"
    );
}

#[test]
fn text_content_ref_zero_copy() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_message(MessageRole::User, "borrowed view")
        .unwrap();

    let part_ref = h.get_message_content_ref(0, 0).unwrap();
    assert!(is_text_type(part_ref.content_type));
    assert_eq!(
        part_ref.data_str(),
        Some("borrowed view"),
        "ref should provide same data"
    );
}

#[test]
fn content_type_matches_role() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_message(MessageRole::User, "user text").unwrap();
    h.append_message(MessageRole::Assistant, "assistant text")
        .unwrap();

    let user_part = h.get_message_content(0, 0).unwrap();
    let asst_part = h.get_message_content(1, 0).unwrap();

    assert_eq!(
        user_part.content_type,
        ContentType::InputText,
        "user content should be InputText"
    );
    assert_eq!(
        asst_part.content_type,
        ContentType::OutputText,
        "assistant content should be OutputText"
    );
}
