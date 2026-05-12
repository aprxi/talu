//! ItemIterator: count, types, exhaustion, empty tests.

use crate::capi::responses::common::{build_simple_conversation, ResponsesTestContext};
use talu::responses::{ItemType, MessageRole, ResponsesView};

#[test]
fn iterator_yields_all_items() {
    let mut ctx = ResponsesTestContext::new();
    build_simple_conversation(ctx.handle_mut());

    let count = ctx.handle().items().count();
    assert_eq!(
        count,
        ctx.handle().item_count(),
        "iterator should yield item_count items"
    );
}

#[test]
fn iterator_exact_size() {
    let mut ctx = ResponsesTestContext::new();
    build_simple_conversation(ctx.handle_mut());

    let iter = ctx.handle().items();
    assert_eq!(
        iter.len(),
        3,
        "ExactSizeIterator::len should match item_count"
    );
}

#[test]
fn iterator_empty_conversation() {
    let ctx = ResponsesTestContext::new();
    let mut iter = ctx.handle().items();
    assert!(
        iter.next().is_none(),
        "empty conversation iterator should yield None"
    );
    assert_eq!(iter.len(), 0);
}

#[test]
fn iterator_item_types_match() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_message(MessageRole::User, "msg").unwrap();
    h.append_function_call("c1", "fn1", "{}").unwrap();
    h.append_function_call_output("c1", "out").unwrap();

    let types: Vec<_> = h.items().map(|item| item.unwrap().item_type).collect();
    assert_eq!(
        types,
        vec![
            ItemType::Message,
            ItemType::FunctionCall,
            ItemType::FunctionCallOutput,
        ],
        "iterator item types should match item_type(i)"
    );
}

#[test]
fn iterator_multiple_passes() {
    let mut ctx = ResponsesTestContext::new();
    build_simple_conversation(ctx.handle_mut());

    let pass1: Vec<_> = ctx.handle().items().map(|i| i.unwrap().item_type).collect();
    let pass2: Vec<_> = ctx.handle().items().map(|i| i.unwrap().item_type).collect();
    assert_eq!(pass1, pass2, "two iteration passes should be identical");
}
