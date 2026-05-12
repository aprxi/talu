//! Function call item variant tests.

use crate::capi::responses::common::ResponsesTestContext;
use talu::responses::{MessageRole, ResponsesView};

#[test]
fn function_call_fields() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_function_call("call_abc", "get_weather", r#"{"location":"NYC"}"#)
        .unwrap();
    let fc = h.get_function_call(0).unwrap();
    assert_eq!(fc.name, "get_weather");
    assert_eq!(fc.call_id, "call_abc");
    assert_eq!(fc.arguments, r#"{"location":"NYC"}"#);
}

#[test]
fn function_call_empty_arguments() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_function_call("call_1", "no_args", "").unwrap();
    let fc = h.get_function_call(0).unwrap();
    assert_eq!(fc.arguments, "", "empty arguments should be preserved");
}

#[test]
fn get_function_call_on_message_errors() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_message(MessageRole::User, "not a function call")
        .unwrap();
    let result = h.get_function_call(0);
    assert!(
        result.is_err(),
        "get_function_call on message should return error"
    );
}
