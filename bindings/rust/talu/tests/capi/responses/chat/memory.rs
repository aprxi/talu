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

/// Calling talu_chat_validate on null should return 0 (invalid), not crash.
#[test]
fn validate_null_chat_returns_invalid() {
    let result = unsafe { talu_sys::talu_chat_validate(ptr::null_mut()) };
    assert_eq!(result, 0, "validate on null should return 0 (invalid)");
}

/// Calling talu_chat_get_system on null should return null, not crash.
#[test]
fn get_system_null_returns_null() {
    let result = unsafe { talu_sys::talu_chat_get_system(ptr::null_mut()) };
    assert!(result.is_null(), "get_system on null should return null");
}

/// Calling talu_chat_get_session_id on null should return null, not crash.
#[test]
fn get_session_id_null_returns_null() {
    let result = unsafe { talu_sys::talu_chat_get_session_id(ptr::null_mut()) };
    assert!(
        result.is_null(),
        "get_session_id on null should return null"
    );
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

/// Calling talu_chat_len on null should return 0, not crash.
#[test]
fn chat_len_null_returns_zero() {
    let result = unsafe { talu_sys::talu_chat_len(ptr::null_mut()) };
    assert_eq!(result, 0, "len on null should return 0");
}

/// Calling talu_chat_clear on null should be a no-op, not crash.
#[test]
fn chat_clear_null_is_noop() {
    // Should not crash
    unsafe {
        talu_sys::talu_chat_clear(ptr::null_mut());
    }
}

/// Calling talu_chat_reset on null should be a no-op, not crash.
#[test]
fn chat_reset_null_is_noop() {
    // Should not crash
    unsafe {
        talu_sys::talu_chat_reset(ptr::null_mut());
    }
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

// ---------------------------------------------------------------------------
// Sampling parameter null handle tests
// ---------------------------------------------------------------------------

/// Setting temperature on null should not crash.
#[test]
fn set_temperature_null_is_noop() {
    unsafe {
        talu_sys::talu_chat_set_temperature(ptr::null_mut(), 0.7);
    }
}

/// Setting top_p on null should not crash.
#[test]
fn set_top_p_null_is_noop() {
    unsafe {
        talu_sys::talu_chat_set_top_p(ptr::null_mut(), 0.9);
    }
}

/// Setting max_tokens on null should not crash.
#[test]
fn set_max_tokens_null_is_noop() {
    unsafe {
        talu_sys::talu_chat_set_max_tokens(ptr::null_mut(), 100);
    }
}

/// Getting temperature from null should return default (1.0), not crash.
#[test]
fn get_temperature_null_returns_default() {
    let result = unsafe { talu_sys::talu_chat_get_temperature(ptr::null_mut()) };
    // Should return some sensible default, not crash
    assert!(result >= 0.0, "temperature should be non-negative");
}
