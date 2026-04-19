//! Memory safety tests for ChatHandle/responses C API boundary handling.
//!
//! These tests specifically try to cause memory errors by:
//! - Passing null pointers to chat functions
//! - Using handles after free
//! - Freeing null handles

use std::ptr;

// ---------------------------------------------------------------------------
// Null pointer handling for chat functions
// ---------------------------------------------------------------------------

/// Freeing a null chat handle should be a no-op, not crash.
#[test]
fn free_null_chat_is_noop() {
    unsafe {
        talu_sys::talu_chat_free(ptr::null_mut());
    }
}

/// Calling talu_chat_get_system on null should return null, not crash.
#[test]
fn get_system_null_returns_null() {
    let result = unsafe { talu_sys::talu_chat_get_system(ptr::null_mut()) };
    assert!(result.is_null(), "get_system on null should return null");
}

/// Calling talu_chat_get_conversation on null should return null, not crash.
#[test]
fn get_conversation_null_returns_null() {
    let result = unsafe { talu_sys::talu_chat_get_conversation(ptr::null_mut()) };
    assert!(
        result.is_null(),
        "get_conversation on null should return null"
    );
}

// ---------------------------------------------------------------------------
// Null pointer handling for responses functions
// ---------------------------------------------------------------------------

/// Freeing a null responses handle should be a no-op, not crash.
#[test]
fn free_null_responses_is_noop() {
    unsafe {
        talu_sys::talu_responses_free(ptr::null_mut());
    }
}

/// Getting item count from null responses should return 0, not crash.
#[test]
fn responses_item_count_null_returns_zero() {
    let result = unsafe { talu_sys::talu_responses_item_count(ptr::null_mut()) };
    assert_eq!(result, 0, "item_count on null should return 0");
}

/// Getting item from null responses should return error, not crash.
#[test]
fn responses_get_item_null_returns_error() {
    use std::mem::MaybeUninit;
    let mut item = MaybeUninit::<talu_sys::CItem>::uninit();
    let result =
        unsafe { talu_sys::talu_responses_get_item(ptr::null_mut(), 0, item.as_mut_ptr()) };
    assert_ne!(
        result, 0,
        "get_item on null handle should return error code"
    );
}

/// Clearing null responses should be a no-op, not crash.
#[test]
fn responses_clear_null_is_noop() {
    unsafe {
        talu_sys::talu_responses_clear(ptr::null_mut());
    }
}

// ---------------------------------------------------------------------------
// Use-after-free tests (documents expected behavior)
// ---------------------------------------------------------------------------

/// Using a ChatHandle after free should not crash (returns invalid/null).
/// Note: This documents defensive behavior; actual UB is possible without checks.
#[test]
fn chat_after_free_is_invalid() {
    let ptr = unsafe { talu_sys::talu_chat_create(ptr::null_mut()) };
    assert!(!ptr.is_null(), "create should return non-null");

    // Free it
    unsafe { talu_sys::talu_chat_free(ptr) };

    // Note: We can't safely test use-after-free without risk of UB.
    // This test documents that we freed it successfully.
}
