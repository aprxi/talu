//! Backend capability query tests.
//!
//! Exercises `talu_backend_get_capabilities` for local and OpenAI-compatible
//! backends. No model file required.

#[test]
fn local_capabilities_reports_streaming() {
    let mut caps = talu_sys::TaluCapabilities::default();
    let config: talu_sys::BackendUnion = unsafe { std::mem::zeroed() };
    let rc = unsafe { talu_sys::talu_backend_get_capabilities(0, &config, &mut caps) };
    assert_eq!(rc, 0, "get_capabilities for local should succeed");
    assert_eq!(caps.streaming, 1, "local backend should support streaming");
}

#[test]
fn local_capabilities_reports_embeddings() {
    let mut caps = talu_sys::TaluCapabilities::default();
    let config: talu_sys::BackendUnion = unsafe { std::mem::zeroed() };
    let rc = unsafe { talu_sys::talu_backend_get_capabilities(0, &config, &mut caps) };
    assert_eq!(rc, 0);
    assert_eq!(caps.embeddings, 1, "local backend should support embeddings");
}

#[test]
fn openai_capabilities_reports_logprobs() {
    let mut caps = talu_sys::TaluCapabilities::default();
    let config: talu_sys::BackendUnion = unsafe { std::mem::zeroed() };
    let rc = unsafe { talu_sys::talu_backend_get_capabilities(1, &config, &mut caps) };
    assert_eq!(rc, 0, "get_capabilities for OpenAI should succeed");
    assert_eq!(caps.logprobs, 1, "OpenAI backend should support logprobs");
}

#[test]
fn capabilities_all_fields_are_boolean() {
    let mut caps = talu_sys::TaluCapabilities::default();
    let config: talu_sys::BackendUnion = unsafe { std::mem::zeroed() };
    let rc = unsafe { talu_sys::talu_backend_get_capabilities(0, &config, &mut caps) };
    assert_eq!(rc, 0);

    assert!(caps.streaming <= 1, "streaming should be boolean");
    assert!(caps.tool_calling <= 1, "tool_calling should be boolean");
    assert!(caps.logprobs <= 1, "logprobs should be boolean");
    assert!(caps.embeddings <= 1, "embeddings should be boolean");
    assert!(caps.json_schema <= 1, "json_schema should be boolean");
}

#[test]
fn openai_capabilities_all_fields_are_boolean() {
    let mut caps = talu_sys::TaluCapabilities::default();
    let config: talu_sys::BackendUnion = unsafe { std::mem::zeroed() };
    let rc = unsafe { talu_sys::talu_backend_get_capabilities(1, &config, &mut caps) };
    assert_eq!(rc, 0);

    assert!(caps.streaming <= 1);
    assert!(caps.tool_calling <= 1);
    assert!(caps.logprobs <= 1);
    assert!(caps.embeddings <= 1);
    assert!(caps.json_schema <= 1);
}

#[test]
fn null_config_returns_error() {
    let mut caps = talu_sys::TaluCapabilities::default();
    let rc =
        unsafe { talu_sys::talu_backend_get_capabilities(0, std::ptr::null(), &mut caps) };
    assert_ne!(rc, 0, "null backend_config should return error");
}

#[test]
fn null_out_caps_returns_error() {
    let config: talu_sys::BackendUnion = unsafe { std::mem::zeroed() };
    let rc = unsafe {
        talu_sys::talu_backend_get_capabilities(0, &config, std::ptr::null_mut())
    };
    assert_ne!(rc, 0, "null out_caps should return error");
}
