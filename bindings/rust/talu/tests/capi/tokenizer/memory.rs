//! Memory safety tests for tokenizer C API.
//!
//! These tests verify that the tokenizer API handles null pointers gracefully
//! without crashing.

use std::ffi::c_void;
use std::ptr;

// ---- Creation / destruction ----

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

// ---- Metadata queries ----

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

/// Getting model dir from null tokenizer should return error, not crash.
#[test]
fn get_model_dir_null_tokenizer_returns_error() {
    let mut out_path: *mut i8 = ptr::null_mut();
    let rc = unsafe {
        talu_sys::talu_tokenizer_get_model_dir(
            ptr::null_mut(),
            &mut out_path as *mut _ as *mut c_void,
        )
    };
    assert_ne!(rc, 0, "get_model_dir with null handle should return error");
    assert!(out_path.is_null(), "output should remain null on error");
}

// ---- Encode / decode with null handle ----

/// Encoding with null handle returns error result, not crash.
#[test]
fn encode_null_handle_returns_error() {
    let text = b"Hello";
    let result = unsafe {
        talu_sys::talu_tokenizer_encode(
            ptr::null_mut(),
            text.as_ptr(),
            text.len(),
            talu_sys::EncodeOptions::default(),
        )
    };
    assert!(
        !result.error_msg.is_null(),
        "encode with null handle should set error_msg"
    );
    assert!(result.ids.is_null(), "ids should be null on error");
    assert_eq!(result.num_tokens, 0);
}

/// Decoding with null handle returns error result, not crash.
#[test]
fn decode_null_handle_returns_error() {
    let tokens = [1u32, 2, 3];
    let result = unsafe {
        talu_sys::talu_tokenizer_decode(
            ptr::null_mut(),
            tokens.as_ptr(),
            tokens.len(),
            talu_sys::DecodeOptionsC::default(),
        )
    };
    assert!(
        !result.error_msg.is_null(),
        "decode with null handle should set error_msg"
    );
    assert!(result.text.is_null(), "text should be null on error");
    assert_eq!(result.text_len, 0);
}

/// Batch encoding with null handle returns error result, not crash.
#[test]
fn encode_batch_null_handle_returns_error() {
    let text = b"Hello";
    let ptrs = [text.as_ptr()];
    let lengths = [text.len()];
    let result = unsafe {
        talu_sys::talu_tokenizer_encode_batch(
            ptr::null_mut(),
            ptrs.as_ptr() as *const u8,
            lengths.as_ptr(),
            1,
            talu_sys::EncodeOptions::default(),
        )
    };
    assert!(
        !result.error_msg.is_null(),
        "encode_batch with null handle should set error_msg"
    );
    assert_eq!(result.total_tokens, 0);
}

// ---- Tokenize with null handle ----

/// Tokenize with null handle returns error result, not crash.
#[test]
fn tokenize_null_handle_returns_error() {
    let text = b"Hello";
    let result = unsafe {
        talu_sys::talu_tokenizer_tokenize(ptr::null_mut(), text.as_ptr(), text.len())
    };
    assert!(
        !result.error_msg.is_null(),
        "tokenize with null handle should set error_msg"
    );
    assert_eq!(result.num_tokens, 0);
}

/// Tokenize bytes with null handle returns error result, not crash.
#[test]
fn tokenize_bytes_null_handle_returns_error() {
    let text = b"Hello";
    let result = unsafe {
        talu_sys::talu_tokenizer_tokenize_bytes(ptr::null_mut(), text.as_ptr(), text.len())
    };
    assert!(
        !result.error_msg.is_null(),
        "tokenize_bytes with null handle should set error_msg"
    );
    assert_eq!(result.num_tokens, 0);
}

// ---- Vocabulary with null handle ----

/// get_vocab with null handle returns error result, not crash.
#[test]
fn get_vocab_null_handle_returns_error() {
    let result = unsafe { talu_sys::talu_tokenizer_get_vocab(ptr::null_mut()) };
    assert!(
        !result.error_msg.is_null(),
        "get_vocab with null handle should set error_msg"
    );
    assert_eq!(result.num_entries, 0);
}

/// get_special_tokens with null handle returns -1 sentinel values, not crash.
#[test]
fn get_special_tokens_null_handle_returns_defaults() {
    let result = unsafe { talu_sys::talu_tokenizer_get_special_tokens(ptr::null_mut()) };
    assert_eq!(result.bos_token_id, -1, "bos should be -1 for null handle");
    assert_eq!(result.unk_token_id, -1, "unk should be -1 for null handle");
    assert_eq!(result.pad_token_id, -1, "pad should be -1 for null handle");
}

/// get_eos_tokens with null handle returns empty result, not crash.
#[test]
fn get_eos_tokens_null_handle_returns_empty() {
    let result = unsafe { talu_sys::talu_tokenizer_get_eos_tokens(ptr::null_mut()) };
    assert_eq!(result.num_tokens, 0);
}

/// id_to_token with null handle returns error, not crash.
#[test]
fn id_to_token_null_handle_returns_error() {
    let mut out: *mut i8 = ptr::null_mut();
    let rc = unsafe {
        talu_sys::talu_tokenizer_id_to_token(
            ptr::null_mut(),
            0,
            &mut out as *mut _ as *mut c_void,
        )
    };
    assert_ne!(rc, 0, "id_to_token with null handle should return error");
    assert!(out.is_null(), "output should remain null on error");
}

/// token_to_id with null handle returns error code, not crash.
#[test]
fn token_to_id_null_handle_returns_error() {
    let token = b"<s>";
    let rc = unsafe {
        talu_sys::talu_tokenizer_token_to_id(ptr::null_mut(), token.as_ptr(), token.len())
    };
    // The error code should be some non-trivial value (not a valid token ID).
    // The C API returns the error code from errorToCode(InvalidHandle).
    assert!(
        rc < 0 || rc > 100,
        "token_to_id with null handle should return error code, got {rc}"
    );
}

// ---- Free functions with null ----

/// Freeing null tokens is a no-op.
#[test]
fn tokens_free_null_is_noop() {
    unsafe { talu_sys::talu_tokens_free(ptr::null(), 0) };
}

/// Freeing null decode result is a no-op.
#[test]
fn decode_result_free_null_is_noop() {
    unsafe { talu_sys::talu_decode_result_free(ptr::null(), 0) };
}

/// Freeing null tokenize result is a no-op.
#[test]
fn tokenize_result_free_null_is_noop() {
    unsafe { talu_sys::talu_tokenize_result_free(ptr::null(), 0) };
}

/// Freeing null tokenize bytes result is a no-op.
#[test]
fn tokenize_bytes_result_free_null_is_noop() {
    unsafe {
        talu_sys::talu_tokenize_bytes_result_free(ptr::null(), 0, ptr::null(), 0)
    };
}

/// Freeing null batch encode result is a no-op.
#[test]
fn batch_encode_result_free_null_is_noop() {
    unsafe {
        talu_sys::talu_batch_encode_result_free(ptr::null(), ptr::null(), 0, 0)
    };
}

/// Freeing zeroed encode result is a no-op.
#[test]
fn encode_result_free_zeroed_is_noop() {
    let result = talu_sys::EncodeResult::default();
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Freeing null padded tensor result is a no-op.
#[test]
fn padded_tensor_result_free_null_is_noop() {
    unsafe {
        talu_sys::talu_padded_tensor_result_free(ptr::null(), ptr::null(), 0, 0)
    };
}

/// Freeing null vocab result is a no-op.
#[test]
fn vocab_result_free_null_is_noop() {
    unsafe {
        talu_sys::talu_vocab_result_free(ptr::null(), ptr::null(), ptr::null(), 0)
    };
}
