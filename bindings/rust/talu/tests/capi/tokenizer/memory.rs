//! Memory safety tests for tokenizer C API.
//!
//! These tests verify that the tokenizer API handles null pointers gracefully
//! without crashing.

use std::ffi::c_void;
use std::ptr;

/// Passing null output pointer to talu_tokenizer_create should return error, not crash.
#[test]
fn create_null_output_returns_error() {
    let model_path = c"test_model";
    let result = unsafe { talu_sys::talu_tokenizer_create(model_path.as_ptr(), ptr::null_mut()) };
    assert_ne!(
        result, 0,
        "create with null output should return error code"
    );
}

/// Passing null output pointer to talu_tokenizer_create_from_json should return error.
#[test]
fn create_from_json_null_output_returns_error() {
    let json = b"{}";
    let result = unsafe {
        talu_sys::talu_tokenizer_create_from_json(json.as_ptr(), json.len(), ptr::null_mut())
    };
    assert_ne!(
        result, 0,
        "create_from_json with null output should return error code"
    );
}

/// Freeing null tokenizer should be a no-op, not crash.
#[test]
fn free_null_tokenizer_is_noop() {
    unsafe { talu_sys::talu_tokenizer_free(ptr::null_mut()) };
}

/// Getting vocab size from null tokenizer should return 0, not crash.
#[test]
fn vocab_size_null_tokenizer_returns_zero() {
    let result = unsafe { talu_sys::talu_tokenizer_get_vocab_size(ptr::null_mut()) };
    assert_eq!(result, 0, "vocab_size with null tokenizer should return 0");
}

/// Getting model max length from null tokenizer should return 0, not crash.
#[test]
fn model_max_length_null_tokenizer_returns_zero() {
    let result = unsafe { talu_sys::talu_tokenizer_get_model_max_length(ptr::null_mut()) };
    assert_eq!(
        result, 0,
        "model_max_length with null tokenizer should return 0"
    );
}
