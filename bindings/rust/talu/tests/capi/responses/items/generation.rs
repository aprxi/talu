//! Tests for generation metadata on items.
//!
//! The `generation` field on Item stores model and sampling parameters
//! used to produce assistant responses. It's only populated by actual
//! generation (via LocalEngine), not manual append operations.

use crate::capi::responses::common::ResponsesTestContext;
use talu::responses::{MessageRole, ResponsesView};

#[test]
fn manually_added_message_has_no_generation() {
    let mut ctx = ResponsesTestContext::new();
    ctx.handle_mut()
        .append_message(MessageRole::Assistant, "Hello!")
        .unwrap();

    let item = ctx.handle().get_item(0).unwrap();
    assert!(
        item.generation.is_none(),
        "manually added message should have generation=None"
    );
}

#[test]
fn all_manual_roles_have_no_generation() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_message(MessageRole::System, "sys").unwrap();
    h.append_message(MessageRole::User, "usr").unwrap();
    h.append_message(MessageRole::Assistant, "asst").unwrap();
    h.append_message(MessageRole::Developer, "dev").unwrap();

    for i in 0..4 {
        let item = ctx.handle().get_item(i).unwrap();
        assert!(
            item.generation.is_none(),
            "item {} should have generation=None",
            i
        );
    }
}

#[test]
fn function_call_has_no_generation() {
    let mut ctx = ResponsesTestContext::new();
    ctx.handle_mut()
        .append_function_call("call_1", "get_weather", r#"{"loc":"NYC"}"#)
        .unwrap();

    let item = ctx.handle().get_item(0).unwrap();
    assert!(
        item.generation.is_none(),
        "function call should have generation=None"
    );
}

#[test]
fn function_call_output_has_no_generation() {
    let mut ctx = ResponsesTestContext::new();
    ctx.handle_mut()
        .append_function_call_output("call_1", "Sunny")
        .unwrap();

    let item = ctx.handle().get_item(0).unwrap();
    assert!(
        item.generation.is_none(),
        "function call output should have generation=None"
    );
}

#[test]
fn item_iterator_returns_generation_none_for_manual() {
    let mut ctx = ResponsesTestContext::new();
    ctx.handle_mut()
        .append_message(MessageRole::User, "Hello")
        .unwrap();
    ctx.handle_mut()
        .append_message(MessageRole::Assistant, "Hi there")
        .unwrap();

    // Use iterator to access items
    let items: Vec<_> = ctx.handle().items().collect();
    assert_eq!(items.len(), 2);

    for (i, result) in items.iter().enumerate() {
        let item = result.as_ref().expect("should get item");
        assert!(
            item.generation.is_none(),
            "iterator item {} should have generation=None",
            i
        );
    }
}
