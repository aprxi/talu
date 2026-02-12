//! Single tool call round tests.

use crate::capi::responses::common::ResponsesTestContext;
use talu::responses::{ItemType, MessageRole, ResponsesView};

#[test]
fn fc_fco_pair_roundtrip() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_message(MessageRole::User, "What is the weather?")
        .unwrap();
    h.append_function_call("call_1", "get_weather", r#"{"location":"NYC"}"#)
        .unwrap();
    h.append_function_call_output("call_1", "Sunny, 72F")
        .unwrap();

    let fc = h.get_function_call(1).unwrap();
    assert_eq!(fc.name, "get_weather");
    assert_eq!(fc.call_id, "call_1");
    assert_eq!(fc.arguments, r#"{"location":"NYC"}"#);

    let fco = h.get_function_call_output(2).unwrap();
    assert_eq!(fco.call_id, "call_1");
    assert_eq!(fco.output_text.as_deref(), Some("Sunny, 72F"));
}

#[test]
fn fc_item_type_is_function_call() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_function_call("c1", "fn1", "{}").unwrap();
    assert_eq!(h.item_type(0), ItemType::FunctionCall);
}

#[test]
fn fco_item_type_is_function_call_output() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_function_call_output("c1", "result").unwrap();
    assert_eq!(h.item_type(0), ItemType::FunctionCallOutput);
}

#[test]
fn fc_arguments_json_preserved() {
    let complex_args = r#"{"query":"SELECT * FROM users WHERE id > 100","format":"json","options":{"limit":50,"offset":0}}"#;
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_function_call("call_complex", "run_query", complex_args)
        .unwrap();
    let fc = h.get_function_call(0).unwrap();
    // arguments use length-based access and are always exact
    assert_eq!(
        fc.arguments, complex_args,
        "complex JSON args should roundtrip exactly"
    );
}

#[test]
fn fco_text_output_preserved() {
    let multiline = "Line 1\nLine 2\nLine 3\n\ttabbed line";
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_function_call_output("c1", multiline).unwrap();
    let fco = h.get_function_call_output(0).unwrap();
    assert_eq!(
        fco.output_text.as_deref(),
        Some(multiline),
        "multi-line text should roundtrip exactly"
    );
}

#[test]
fn fc_fco_call_id_matches() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_function_call("shared_id", "func", "{}").unwrap();
    h.append_function_call_output("shared_id", "result")
        .unwrap();

    let fc = h.get_function_call(0).unwrap();
    let fco = h.get_function_call_output(1).unwrap();
    assert_eq!(fc.call_id, "shared_id");
    assert_eq!(fco.call_id, "shared_id");
}
