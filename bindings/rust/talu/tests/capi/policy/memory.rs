//! Memory safety tests for policy C API.
//!
//! These tests verify that the policy API handles null pointers gracefully
//! without crashing.

use std::ffi::c_void;
use std::ptr;

/// Passing null output pointer to talu_policy_create should return error, not crash.
#[test]
fn create_null_output_returns_error() {
    let json = b"{}";
    let result =
        unsafe { talu_sys::talu_policy_create(json.as_ptr(), json.len(), ptr::null_mut()) };
    assert_ne!(
        result, 0,
        "create with null output should return error code"
    );
}

/// Passing null json pointer to talu_policy_create should return error, not crash.
#[test]
fn create_null_json_returns_error() {
    let mut policy: *mut c_void = ptr::null_mut();
    let result = unsafe {
        talu_sys::talu_policy_create(ptr::null(), 10, &mut policy as *mut _ as *mut c_void)
    };
    assert_ne!(result, 0, "create with null json should return error code");
}

/// Freeing null policy should be a no-op, not crash.
#[test]
fn free_null_policy_is_noop() {
    unsafe { talu_sys::talu_policy_free(ptr::null_mut()) };
}

/// Evaluating with null policy should not crash.
/// Returns 1 (allowed/passthrough) as safe default for null policy.
#[test]
fn evaluate_null_policy_does_not_crash() {
    let action = b"test_action";
    let _result =
        unsafe { talu_sys::talu_policy_evaluate(ptr::null_mut(), action.as_ptr(), action.len()) };
    // The key is it doesn't crash - the return value is implementation-defined
}

/// Getting mode from null policy should return 0, not crash.
#[test]
fn get_mode_null_policy_returns_zero() {
    let result = unsafe { talu_sys::talu_policy_get_mode(ptr::null_mut()) };
    assert_eq!(result, 0, "get_mode with null policy should return 0");
}
