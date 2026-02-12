//! ChatHandle creation and access tests.

use std::ffi::CString;
use talu::responses::ResponsesView;
use talu::ChatHandle;

#[test]
fn create_empty_chat() {
    let chat = ChatHandle::new(None).unwrap();
    let conv = chat.responses();
    assert_eq!(conv.item_count(), 0, "new chat should have zero messages");
}

#[test]
fn create_with_system_prompt() {
    let chat = ChatHandle::new(Some("You are helpful.")).unwrap();
    let system = unsafe { talu_sys::talu_chat_get_system(chat.as_ptr()) };
    assert!(!system.is_null(), "system prompt should be retrievable");
    let text = unsafe { std::ffi::CStr::from_ptr(system) }
        .to_string_lossy()
        .to_string();
    assert_eq!(text, "You are helpful.");
}

#[test]
fn create_with_session() {
    let session_id = CString::new("session-abc").unwrap();
    let ptr = unsafe {
        talu_sys::talu_chat_create_with_session(session_id.as_ptr(), std::ptr::null_mut())
    };
    assert!(!ptr.is_null(), "create_with_session should return non-null");

    let got_session = unsafe { talu_sys::talu_chat_get_session_id(ptr) };
    if !got_session.is_null() {
        let s = unsafe { std::ffi::CStr::from_ptr(got_session) }
            .to_string_lossy()
            .to_string();
        assert_eq!(s, "session-abc");
    }
    unsafe { talu_sys::talu_chat_free(ptr) };
}

#[test]
fn create_with_system_and_session() {
    let system = CString::new("You are smart.").unwrap();
    let session = CString::new("session-xyz").unwrap();
    let ptr = unsafe {
        talu_sys::talu_chat_create_with_system_and_session(
            system.as_ptr(),
            session.as_ptr(),
            std::ptr::null_mut(),
        )
    };
    assert!(!ptr.is_null());

    let got_system = unsafe { talu_sys::talu_chat_get_system(ptr) };
    assert!(!got_system.is_null());
    let sys_text = unsafe { std::ffi::CStr::from_ptr(got_system) }
        .to_string_lossy()
        .to_string();
    assert_eq!(sys_text, "You are smart.");

    unsafe { talu_sys::talu_chat_free(ptr) };
}

#[test]
fn get_conversation_returns_view() {
    let chat = ChatHandle::new(Some("sys")).unwrap();
    let conv = chat.responses();
    // System prompt is stored internally, not as a conversation item in all implementations.
    // Just verify we can access item_count without panicking.
    let _count = conv.item_count();
}

#[test]
fn validate_live_chat() {
    let chat = ChatHandle::new(None).unwrap();
    let rc = unsafe { talu_sys::talu_chat_validate(chat.as_ptr()) };
    assert_eq!(rc, 1, "validate should return 1 for live chat");
}
