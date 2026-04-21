//! Core CAPI contract tests for chat/completions request validation.

fn validate(
    messages_json: &str,
    max_tokens: Option<i64>,
    max_completion_tokens: Option<i64>,
    temperature: f64,
    top_p: f64,
    presence_penalty: f64,
    frequency_penalty: f64,
    tools_json: Option<&str>,
    tool_choice_json: Option<&str>,
) -> i32 {
    let (tools_ptr, tools_len) = if let Some(v) = tools_json {
        (v.as_ptr(), v.len())
    } else {
        (std::ptr::null(), 0)
    };
    let (tool_choice_ptr, tool_choice_len) = if let Some(v) = tool_choice_json {
        (v.as_ptr(), v.len())
    } else {
        (std::ptr::null(), 0)
    };

    unsafe {
        talu_sys::talu_completions_validate_request(
            messages_json.as_ptr(),
            messages_json.len(),
            usize::from(max_tokens.is_some()),
            max_tokens.unwrap_or_default(),
            usize::from(max_completion_tokens.is_some()),
            max_completion_tokens.unwrap_or_default(),
            temperature,
            top_p,
            presence_penalty,
            frequency_penalty,
            tools_ptr,
            tools_len,
            tool_choice_ptr,
            tool_choice_len,
        )
    }
}

#[test]
fn completions_validate_accepts_minimal_valid_request() {
    let rc = validate(
        r#"[{"role":"user","content":"hello"}]"#,
        None,
        None,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        None,
        None,
    );
    assert_eq!(
        rc,
        0,
        "expected success, core error: {:?}",
        talu::error::last_error_message()
    );
}

#[test]
fn completions_validate_rejects_invalid_message_role() {
    let rc = validate(
        r#"[{"role":"bogus","content":"hello"}]"#,
        None,
        None,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        None,
        None,
    );
    assert_ne!(rc, 0, "invalid message role must fail");
}

#[test]
fn completions_validate_rejects_negative_max_tokens() {
    let rc = validate(
        r#"[{"role":"user","content":"hello"}]"#,
        Some(-2),
        None,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        None,
        None,
    );
    assert_ne!(rc, 0, "negative max_tokens must fail");
}

#[test]
fn completions_validate_rejects_invalid_tools_shape() {
    let rc = validate(
        r#"[{"role":"user","content":"hello"}]"#,
        None,
        None,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        Some(r#"{"bad":"shape"}"#),
        None,
    );
    assert_ne!(rc, 0, "invalid tools shape must fail");
}

#[test]
fn completions_validate_accepts_valid_tools_and_tool_choice() {
    let rc = validate(
        r#"[{"role":"user","content":"hello"}]"#,
        Some(16),
        None,
        0.1,
        0.9,
        0.0,
        0.0,
        Some(r#"[{"type":"function","function":{"name":"echo","parameters":{"type":"object"}}}]"#),
        Some(r#"{"type":"function","function":{"name":"echo"}}"#),
    );
    assert_eq!(
        rc,
        0,
        "valid tools/tool_choice should pass, core error: {:?}",
        talu::error::last_error_message()
    );
}
