//! Function call output item variant tests.

use crate::capi::responses::common::ResponsesTestContext;
use talu::responses::{MessageRole, ResponsesView};

#[test]
fn fco_text_output() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_function_call_output("call_42", "Temperature is 72F")
        .unwrap();
    let fco = h.get_function_call_output(0).unwrap();
    assert_eq!(fco.call_id, "call_42");
    assert_eq!(fco.output_text.as_deref(), Some("Temperature is 72F"));
    assert!(fco.is_text_output, "should be text output");
}

#[test]
fn get_fco_on_wrong_type_errors() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();
    h.append_message(MessageRole::User, "not a FCO").unwrap();
    let result = h.get_function_call_output(0);
    assert!(
        result.is_err(),
        "get_function_call_output on message should return error"
    );
}
