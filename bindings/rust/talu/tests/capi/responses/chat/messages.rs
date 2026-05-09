//! Chat message management tests.

use crate::capi::responses::common::RawChatHandle;
use std::ffi::CString;

#[test]
fn chat_to_json_roundtrip() {
    let chat = RawChatHandle::with_system("system msg");

    // Append a user message
    {
        let conv_ptr = unsafe { talu_sys::talu_chat_get_conversation(chat.as_ptr()) };
        let content = b"hello";
        unsafe {
            talu_sys::talu_responses_append_message(
                conv_ptr,
                talu_sys::MessageRole::User as u8,
                content.as_ptr(),
                content.len(),
            )
        };
    }

    let json_ptr = unsafe { talu_sys::talu_chat_to_json(chat.as_ptr()) };
    assert!(!json_ptr.is_null(), "to_json should return non-null");
    let json = unsafe { std::ffi::CStr::from_ptr(json_ptr) }
        .to_string_lossy()
        .to_string();
    unsafe { talu_sys::talu_text_free(json_ptr as *mut _) };

    // Load into another chat
    let chat2 = RawChatHandle::new();
    let c_json = CString::new(json).unwrap();
    let rc = unsafe { talu_sys::talu_chat_set_messages(chat2.as_ptr(), c_json.as_ptr()) };
    assert_eq!(rc, 0, "set_messages should succeed");
}

#[test]
fn set_system_prompt() {
    let chat = RawChatHandle::new();
    let sys = CString::new("New system").unwrap();
    let rc = unsafe { talu_sys::talu_chat_set_system(chat.as_ptr(), sys.as_ptr()) };
    assert_eq!(rc, 0, "set_system should succeed");

    let got = unsafe { talu_sys::talu_chat_get_system(chat.as_ptr()) };
    assert!(!got.is_null());
    let text = unsafe { std::ffi::CStr::from_ptr(got) }
        .to_string_lossy()
        .to_string();
    assert_eq!(text, "New system");
}

#[test]
fn get_system_returns_null_when_unset() {
    let chat = RawChatHandle::new();
    let system = unsafe { talu_sys::talu_chat_get_system(chat.as_ptr()) };
    assert!(
        system.is_null(),
        "get_system should return null when no system prompt set"
    );
}
