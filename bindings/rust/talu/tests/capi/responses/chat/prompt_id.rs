//! C API tests for talu_chat_get_prompt_id and talu_chat_set_prompt_id.
//!
//! These functions allow getting/setting the prompt document ID for lineage tracking.

use std::ffi::CString;
use talu::ChatHandle;

/// Setting and getting prompt_id on a chat handle.
#[test]
fn set_and_get_prompt_id() {
    let chat = ChatHandle::new(None).expect("ChatHandle::new failed");
    let prompt_id = CString::new("doc_abc123").expect("CString");

    // Initially null
    let initial = unsafe { talu_sys::talu_chat_get_prompt_id(chat.as_ptr()) };
    assert!(initial.is_null(), "initial prompt_id should be null");

    // Set prompt_id
    let rc = unsafe { talu_sys::talu_chat_set_prompt_id(chat.as_ptr(), prompt_id.as_ptr()) };
    assert_eq!(rc, 0, "set_prompt_id should succeed");

    // Get prompt_id
    let result = unsafe { talu_sys::talu_chat_get_prompt_id(chat.as_ptr()) };
    assert!(
        !result.is_null(),
        "get_prompt_id should return non-null after set"
    );

    let retrieved = unsafe { std::ffi::CStr::from_ptr(result) }
        .to_string_lossy()
        .to_string();
    assert_eq!(
        retrieved, "doc_abc123",
        "prompt_id should match what was set"
    );
}

/// Setting prompt_id to null clears it.
#[test]
fn set_prompt_id_null_clears() {
    let chat = ChatHandle::new(None).expect("ChatHandle::new failed");
    let prompt_id = CString::new("doc_xyz").expect("CString");

    // Set prompt_id
    let rc = unsafe { talu_sys::talu_chat_set_prompt_id(chat.as_ptr(), prompt_id.as_ptr()) };
    assert_eq!(rc, 0);

    // Verify it's set
    let result = unsafe { talu_sys::talu_chat_get_prompt_id(chat.as_ptr()) };
    assert!(!result.is_null());

    // Clear by setting to null
    let rc = unsafe { talu_sys::talu_chat_set_prompt_id(chat.as_ptr(), std::ptr::null()) };
    assert_eq!(rc, 0, "setting null should succeed");

    // Verify it's cleared
    let result = unsafe { talu_sys::talu_chat_get_prompt_id(chat.as_ptr()) };
    assert!(result.is_null(), "prompt_id should be null after clearing");
}

/// Get prompt_id on null handle returns null.
#[test]
fn get_prompt_id_null_handle() {
    let result = unsafe { talu_sys::talu_chat_get_prompt_id(std::ptr::null_mut()) };
    assert!(
        result.is_null(),
        "get_prompt_id on null handle should return null"
    );
}

/// Set prompt_id on null handle returns error.
#[test]
fn set_prompt_id_null_handle() {
    let prompt_id = CString::new("doc_test").expect("CString");
    let rc = unsafe { talu_sys::talu_chat_set_prompt_id(std::ptr::null_mut(), prompt_id.as_ptr()) };
    assert_ne!(rc, 0, "set_prompt_id on null handle should return error");
}

/// Prompt_id with special characters.
#[test]
fn prompt_id_special_characters() {
    let chat = ChatHandle::new(None).expect("ChatHandle::new failed");
    let prompt_id = CString::new("doc-with_special.chars:123").expect("CString");

    let rc = unsafe { talu_sys::talu_chat_set_prompt_id(chat.as_ptr(), prompt_id.as_ptr()) };
    assert_eq!(rc, 0);

    let result = unsafe { talu_sys::talu_chat_get_prompt_id(chat.as_ptr()) };
    assert!(!result.is_null());

    let retrieved = unsafe { std::ffi::CStr::from_ptr(result) }
        .to_string_lossy()
        .to_string();
    assert_eq!(retrieved, "doc-with_special.chars:123");
}

/// Setting empty string is equivalent to clearing.
#[test]
fn set_prompt_id_empty_string() {
    let chat = ChatHandle::new(None).expect("ChatHandle::new failed");

    // Set a value first
    let prompt_id = CString::new("doc_initial").expect("CString");
    let rc = unsafe { talu_sys::talu_chat_set_prompt_id(chat.as_ptr(), prompt_id.as_ptr()) };
    assert_eq!(rc, 0);

    // Set to empty string
    let empty = CString::new("").expect("CString");
    let rc = unsafe { talu_sys::talu_chat_set_prompt_id(chat.as_ptr(), empty.as_ptr()) };
    assert_eq!(rc, 0);

    // Should be null/cleared
    let result = unsafe { talu_sys::talu_chat_get_prompt_id(chat.as_ptr()) };
    assert!(result.is_null(), "empty string should clear prompt_id");
}

/// Multiple set/get cycles work correctly.
#[test]
fn prompt_id_multiple_set_get() {
    let chat = ChatHandle::new(None).expect("ChatHandle::new failed");

    for i in 0..10 {
        let prompt_id = CString::new(format!("doc_iteration_{}", i)).expect("CString");
        let rc = unsafe { talu_sys::talu_chat_set_prompt_id(chat.as_ptr(), prompt_id.as_ptr()) };
        assert_eq!(rc, 0, "iteration {} set failed", i);

        let result = unsafe { talu_sys::talu_chat_get_prompt_id(chat.as_ptr()) };
        assert!(!result.is_null(), "iteration {} get returned null", i);

        let retrieved = unsafe { std::ffi::CStr::from_ptr(result) }
            .to_string_lossy()
            .to_string();
        assert_eq!(retrieved, format!("doc_iteration_{}", i));
    }

    // Final clear
    let rc = unsafe { talu_sys::talu_chat_set_prompt_id(chat.as_ptr(), std::ptr::null()) };
    assert_eq!(rc, 0);
}
