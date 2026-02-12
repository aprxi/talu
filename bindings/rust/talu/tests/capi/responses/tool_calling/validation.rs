//! Tool call edge cases and error path tests.

use crate::capi::responses::common::ResponsesTestContext;
use talu::responses::{MessageRole, ResponsesView};

#[test]
fn fc_single_char_call_id() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_function_call("x", "func", "{}").unwrap();
    let fc = h.get_function_call(0).unwrap();
    assert_eq!(fc.call_id, "x");
}

#[test]
fn fc_single_char_name() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_function_call("c1", "f", "{}").unwrap();
    let fc = h.get_function_call(0).unwrap();
    assert_eq!(fc.name, "f");
}

#[test]
fn fc_empty_arguments() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_function_call("c1", "f1", "").unwrap();
    let fc = h.get_function_call(0).unwrap();
    assert_eq!(fc.arguments, "", "empty arguments should be preserved");
}

#[test]
fn fco_empty_output() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_function_call_output("c1", "").unwrap();
    let fco = h.get_function_call_output(0).unwrap();
    // Empty output may come back as Some("") or None depending on the C API behavior
    let text = fco.output_text.as_deref().unwrap_or("");
    assert_eq!(text, "", "empty output should be preserved");
}

#[test]
fn fc_large_arguments() {
    // 100KB JSON payload
    let big_json = format!(r#"{{"data":"{}"}}"#, "x".repeat(100_000));
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_function_call("c_big", "big_func", &big_json)
        .unwrap();
    let fc = h.get_function_call(0).unwrap();
    assert_eq!(fc.arguments, big_json, "large arguments should roundtrip");
}

#[test]
fn fco_large_output() {
    let big_output = "Y".repeat(100_000);
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_function_call_output("c_big", &big_output).unwrap();
    let fco = h.get_function_call_output(0).unwrap();
    assert_eq!(
        fco.output_text.as_deref(),
        Some(big_output.as_str()),
        "large output should roundtrip"
    );
}

#[test]
fn get_fc_on_message_item() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_message(MessageRole::User, "not a FC").unwrap();
    let result = h.get_function_call(0);
    assert!(result.is_err(), "get_function_call on message should error");
}

#[test]
fn get_fco_on_fc_item() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_function_call("c1", "fn1", "{}").unwrap();
    let result = h.get_function_call_output(0);
    assert!(
        result.is_err(),
        "get_function_call_output on FC should error"
    );
}

#[test]
fn fc_special_chars_in_call_id() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_function_call("call-123_abc.xyz", "func", "{}")
        .unwrap();
    let fc = h.get_function_call(0).unwrap();
    assert_eq!(fc.call_id, "call-123_abc.xyz");
}

#[test]
fn fc_unicode_in_arguments() {
    let args = r#"{"greeting":"Hello \u4e16\u754c"}"#;
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_function_call("c1", "greet", args).unwrap();
    let fc = h.get_function_call(0).unwrap();
    assert_eq!(
        fc.arguments, args,
        "unicode in arguments should be preserved"
    );
}
