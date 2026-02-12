//! Item reference variant tests.
//!
//! Item references are created via `load_responses_json` (no direct append API).

use crate::capi::responses::common::ResponsesTestContext;
use talu::responses::{ItemType, MessageRole, ResponsesView};

#[test]
fn item_reference_id() {
    let mut ctx = ResponsesTestContext::new();
    let json = r#"[{"type": "item_reference", "id": "msg_abc123"}]"#;
    ctx.handle_mut().load_responses_json(json).unwrap();

    assert_eq!(ctx.handle().item_count(), 1);
    assert_eq!(ctx.handle().item_type(0), ItemType::ItemReference);

    let item_ref = ctx.handle().get_item_reference(0).unwrap();
    assert_eq!(item_ref.id, "msg_abc123");
}

#[test]
fn get_item_reference_on_wrong_type() {
    let mut ctx = ResponsesTestContext::new();
    ctx.handle_mut()
        .append_message(MessageRole::User, "not a reference")
        .unwrap();
    let result = ctx.handle().get_item_reference(0);
    assert!(
        result.is_err(),
        "get_item_reference on message should return error"
    );
}
