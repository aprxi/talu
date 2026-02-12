//! Legacy Completions JSON format tests.

use crate::capi::responses::common::{build_simple_conversation, ResponsesTestContext};
use talu::responses::{MessageRole, ResponsesHandle, ResponsesView};

#[test]
fn completions_json_roundtrip() {
    let mut ctx = ResponsesTestContext::new();
    build_simple_conversation(ctx.handle_mut());

    let json = ctx.handle().to_completions_json().unwrap();

    let mut dest = ResponsesHandle::new().unwrap();
    dest.load_completions_json(&json).unwrap();

    assert_eq!(
        dest.item_count(),
        ctx.handle().item_count(),
        "roundtrip should preserve item count"
    );
    for i in 0..dest.item_count() {
        let orig = ctx.handle().message_text(i).unwrap();
        let loaded = dest.message_text(i).unwrap();
        assert_eq!(
            orig, loaded,
            "text at index {} should match after roundtrip",
            i
        );
    }
}

#[test]
fn completions_json_message_roles() {
    let mut ctx = ResponsesTestContext::new();
    build_simple_conversation(ctx.handle_mut());

    let json = ctx.handle().to_completions_json().unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    let messages = parsed.as_array().expect("completions JSON should be array");

    let roles: Vec<_> = messages
        .iter()
        .filter_map(|m| m.get("role").and_then(|r| r.as_str()))
        .collect();
    assert_eq!(roles, vec!["system", "user", "assistant"]);
}

#[test]
fn load_completions_json_clears_first() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_message(MessageRole::User, "existing").unwrap();
    assert_eq!(h.item_count(), 1);

    let json = r#"[{"role":"user","content":"replacement"}]"#;
    h.load_completions_json(json).unwrap();

    assert_eq!(h.item_count(), 1, "load should replace existing items");
    let text = h.message_text(0).unwrap();
    assert_eq!(text, "replacement");
}

#[test]
fn load_completions_json_invalid() {
    let mut ctx = ResponsesTestContext::new();
    let result = ctx.handle_mut().load_completions_json("{not valid json");
    assert!(result.is_err(), "invalid JSON should return error");
}
