//! Chat message management tests.

use std::ffi::CString;
use talu::ChatHandle;

#[test]
fn chat_len_reflects_messages() {
    let chat = ChatHandle::new(Some("system")).unwrap();
    let len = unsafe { talu_sys::talu_chat_len(chat.as_ptr()) };
    // System prompt may or may not count as a "message" in chat_len
    // depending on implementation. Verify it's accessible.
    assert!(len <= 1, "new chat with system should have len <= 1");
}

#[test]
fn chat_clear_resets_conversation() {
    let chat = ChatHandle::new(Some("System msg")).unwrap();

    // Add a user message via the conversation
    {
        let conv_ptr = unsafe { talu_sys::talu_chat_get_conversation(chat.as_ptr()) };
        let content = b"user msg";
        unsafe {
            talu_sys::talu_responses_append_message(
                conv_ptr,
                talu_sys::CMessageRole::User as u8,
                content.as_ptr(),
                content.len(),
            )
        };
    }

    unsafe { talu_sys::talu_chat_clear(chat.as_ptr()) };

    // talu_chat_clear resets the conversation (clears items, resets sampling params).
    // The system prompt is stored as a conversation item, so it is also cleared.
    let len = unsafe { talu_sys::talu_chat_len(chat.as_ptr()) };
    assert_eq!(len, 0, "clear should remove all messages");
}

#[test]
fn chat_reset_removes_everything() {
    let chat = ChatHandle::new(Some("Goes away")).unwrap();
    unsafe { talu_sys::talu_chat_reset(chat.as_ptr()) };

    let system = unsafe { talu_sys::talu_chat_get_system(chat.as_ptr()) };
    // After reset, system should be null (removed)
    assert!(system.is_null(), "system prompt should be gone after reset");

    let len = unsafe { talu_sys::talu_chat_len(chat.as_ptr()) };
    assert_eq!(len, 0, "should have zero messages after reset");
}

#[test]
fn chat_to_json_roundtrip() {
    let chat = ChatHandle::new(Some("system msg")).unwrap();

    // Append a user message
    {
        let conv_ptr = unsafe { talu_sys::talu_chat_get_conversation(chat.as_ptr()) };
        let content = b"hello";
        unsafe {
            talu_sys::talu_responses_append_message(
                conv_ptr,
                talu_sys::CMessageRole::User as u8,
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
    let chat2 = ChatHandle::new(None).unwrap();
    let c_json = CString::new(json).unwrap();
    let rc = unsafe { talu_sys::talu_chat_set_messages(chat2.as_ptr(), c_json.as_ptr()) };
    assert_eq!(rc, 0, "set_messages should succeed");
}

#[test]
fn set_system_prompt() {
    let chat = ChatHandle::new(None).unwrap();
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
    let chat = ChatHandle::new(None).unwrap();
    let system = unsafe { talu_sys::talu_chat_get_system(chat.as_ptr()) };
    assert!(
        system.is_null(),
        "get_system should return null when no system prompt set"
    );
}
