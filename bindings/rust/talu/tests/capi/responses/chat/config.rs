//! Chat metadata and configuration C API tests.

use crate::capi::responses::common::RawChatHandle;
use std::ffi::{CStr, CString};
use talu::responses::new_session_id;

#[test]
fn session_id_new_returns_uuid_v4_shape() {
    let session_id = new_session_id().expect("session id generation should succeed");

    assert_eq!(session_id.len(), 36, "UUID text length");
    assert_eq!(session_id.as_bytes()[8], b'-');
    assert_eq!(session_id.as_bytes()[13], b'-');
    assert_eq!(session_id.as_bytes()[18], b'-');
    assert_eq!(session_id.as_bytes()[23], b'-');
    assert_eq!(session_id.as_bytes()[14], b'4', "UUID version should be 4");
    assert!(
        matches!(session_id.as_bytes()[19], b'8' | b'9' | b'a' | b'b'),
        "UUID variant should be RFC 4122"
    );
    assert!(
        session_id
            .bytes()
            .all(|b| b == b'-' || b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "session id must be lowercase UUID-compatible hex"
    );
}

#[test]
fn tools_roundtrip_uses_explicit_length() {
    let chat = RawChatHandle::new();
    let expected = r#"[{"type":"function","function":{"name":"weather"}}]"#;
    let with_suffix = CString::new(format!("{expected}THIS_SUFFIX_MUST_NOT_APPEAR")).unwrap();

    let rc = unsafe {
        talu_sys::talu_chat_set_tools(chat.as_ptr(), with_suffix.as_ptr(), expected.len())
    };
    assert_eq!(rc, 0, "set_tools should succeed");

    let got = unsafe { talu_sys::talu_chat_get_tools(chat.as_ptr()) };
    assert!(!got.is_null(), "get_tools should return stored JSON");
    let text = unsafe { CStr::from_ptr(got) }
        .to_string_lossy()
        .into_owned();
    unsafe { talu_sys::talu_text_free(got) };
    assert_eq!(text, expected, "tools storage must honor json_len");
}

#[test]
fn tools_null_clears_value() {
    let chat = RawChatHandle::new();
    let tools = CString::new(r#"[{"type":"function"}]"#).unwrap();

    let rc = unsafe {
        talu_sys::talu_chat_set_tools(chat.as_ptr(), tools.as_ptr(), tools.as_bytes().len())
    };
    assert_eq!(rc, 0);
    let got = unsafe { talu_sys::talu_chat_get_tools(chat.as_ptr()) };
    assert!(!got.is_null());
    unsafe { talu_sys::talu_text_free(got) };

    let rc = unsafe { talu_sys::talu_chat_set_tools(chat.as_ptr(), std::ptr::null(), 0) };
    assert_eq!(rc, 0, "null tools should clear");
    assert!(
        unsafe { talu_sys::talu_chat_get_tools(chat.as_ptr()) }.is_null(),
        "tools should be absent after clearing"
    );
}

#[test]
fn tool_choice_roundtrip_uses_explicit_length() {
    let chat = RawChatHandle::new();
    let expected = r#"{"type":"function","function":{"name":"weather"}}"#;
    let with_suffix = CString::new(format!("{expected}THIS_SUFFIX_MUST_NOT_APPEAR")).unwrap();

    let rc = unsafe {
        talu_sys::talu_chat_set_tool_choice(chat.as_ptr(), with_suffix.as_ptr(), expected.len())
    };
    assert_eq!(rc, 0, "set_tool_choice should succeed");

    let got = unsafe { talu_sys::talu_chat_get_tool_choice(chat.as_ptr()) };
    assert!(!got.is_null(), "get_tool_choice should return stored JSON");
    let text = unsafe { CStr::from_ptr(got) }
        .to_string_lossy()
        .into_owned();
    unsafe { talu_sys::talu_text_free(got) };
    assert_eq!(text, expected, "tool_choice storage must honor json_len");
}

#[test]
fn tool_choice_null_clears_value() {
    let chat = RawChatHandle::new();
    let choice = CString::new(r#"{"type":"auto"}"#).unwrap();

    let rc = unsafe {
        talu_sys::talu_chat_set_tool_choice(chat.as_ptr(), choice.as_ptr(), choice.as_bytes().len())
    };
    assert_eq!(rc, 0);
    let got = unsafe { talu_sys::talu_chat_get_tool_choice(chat.as_ptr()) };
    assert!(!got.is_null());
    unsafe { talu_sys::talu_text_free(got) };

    let rc = unsafe { talu_sys::talu_chat_set_tool_choice(chat.as_ptr(), std::ptr::null(), 0) };
    assert_eq!(rc, 0, "null tool_choice should clear");
    assert!(
        unsafe { talu_sys::talu_chat_get_tool_choice(chat.as_ptr()) }.is_null(),
        "tool_choice should be absent after clearing"
    );
}

#[test]
fn tools_and_tool_choice_reject_null_chat() {
    let json = CString::new("{}").unwrap();
    let rc = unsafe { talu_sys::talu_chat_set_tools(std::ptr::null_mut(), json.as_ptr(), 2) };
    assert_ne!(rc, 0, "set_tools must reject null chat");

    let rc = unsafe { talu_sys::talu_chat_set_tool_choice(std::ptr::null_mut(), json.as_ptr(), 2) };
    assert_ne!(rc, 0, "set_tool_choice must reject null chat");

    assert!(unsafe { talu_sys::talu_chat_get_tools(std::ptr::null_mut()) }.is_null());
    assert!(unsafe { talu_sys::talu_chat_get_tool_choice(std::ptr::null_mut()) }.is_null());
}
