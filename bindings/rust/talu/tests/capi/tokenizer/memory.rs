//! Memory safety tests for tokenizer C API.
//!
//! These tests verify that the tokenizer API handles null pointers gracefully
//! without crashing.

use crate::capi::tokenizer::common;
use crate::capi::tokenizer::common::TokenizerTestContext;
use std::ffi::c_void;
use std::ptr;

// ---- Creation / destruction ----

/// Passing null output pointer to talu_tokenizer_create should return error, not crash.
#[test]
fn create_null_output_returns_error() {
    let model_path = c"test_model";
    let result = unsafe { talu_sys::talu_tokenizer_create(model_path.as_ptr(), ptr::null_mut()) };
    assert_eq!(
        result,
        talu_sys::ErrorCode::InvalidArgument as i32,
        "create with null output should return InvalidArgument"
    );
}

/// Passing null output pointer to talu_tokenizer_create_from_json should return error.
#[test]
fn create_from_json_null_output_returns_error() {
    let json = b"{}";
    let result = unsafe {
        talu_sys::talu_tokenizer_create_from_json(json.as_ptr(), json.len(), ptr::null_mut())
    };
    assert_eq!(
        result,
        talu_sys::ErrorCode::InvalidArgument as i32,
        "create_from_json with null output should return InvalidArgument"
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
    assert_eq!(
        rc,
        talu_sys::ErrorCode::InvalidHandle as i32,
        "get_model_dir with null handle should return InvalidHandle"
    );
    assert!(out_path.is_null(), "output should remain null on error");
}

/// get_model_dir must clear a stale output pointer before reporting an error.
#[test]
fn get_model_dir_null_tokenizer_clears_stale_output() {
    let mut out_path = std::ptr::dangling_mut::<i8>();
    let rc = unsafe {
        talu_sys::talu_tokenizer_get_model_dir(
            ptr::null_mut(),
            &mut out_path as *mut _ as *mut c_void,
        )
    };
    assert_eq!(
        rc,
        talu_sys::ErrorCode::InvalidHandle as i32,
        "get_model_dir with null handle should return InvalidHandle"
    );
    assert!(
        out_path.is_null(),
        "get_model_dir must null out a stale output pointer on error"
    );
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
            &talu_sys::EncodeOptions::default(),
        )
    };
    assert!(
        !result.error_msg.is_null(),
        "encode with null handle should set error_msg"
    );
    let code = unsafe { talu_sys::talu_last_error_code() };
    assert_eq!(
        code,
        talu_sys::ErrorCode::InvalidHandle as i32,
        "encode null-handle path must set InvalidHandle code"
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
            &talu_sys::DecodeOptionsC::default(),
        )
    };
    assert!(
        !result.error_msg.is_null(),
        "decode with null handle should set error_msg"
    );
    let code = unsafe { talu_sys::talu_last_error_code() };
    assert_eq!(
        code,
        talu_sys::ErrorCode::InvalidHandle as i32,
        "decode null-handle path must set InvalidHandle code"
    );
    assert!(result.text.is_null(), "text should be null on error");
    assert_eq!(result.text_len, 0);
}

/// Decoding with null token pointer and zero length should be empty success.
#[test]
fn decode_null_tokens_zero_len_is_empty_success() {
    let json = br#"{
  "version": "1.0",
  "model": {"type":"BPE","vocab":{"a":0},"merges":[]},
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    let mut handle: *mut c_void = ptr::null_mut();
    let rc = unsafe {
        talu_sys::talu_tokenizer_create_from_json(
            json.as_ptr(),
            json.len(),
            &mut handle as *mut _ as *mut c_void,
        )
    };
    assert_eq!(rc, 0);
    assert!(!handle.is_null());

    let result = unsafe {
        talu_sys::talu_tokenizer_decode(
            handle,
            ptr::null(),
            0,
            &talu_sys::DecodeOptionsC::default(),
        )
    };
    assert!(result.error_msg.is_null(), "zero-length decode should succeed");
    assert!(result.text.is_null(), "empty decode should have null text pointer");
    assert_eq!(result.text_len, 0);
    unsafe { talu_sys::talu_tokenizer_free(handle) };
}

/// Decoding with a null token pointer and non-zero length must return
/// InvalidArgument rather than reading through NULL.
#[test]
fn decode_null_tokens_nonzero_len_returns_error() {
    let ctx = TokenizerTestContext::new();
    let result = unsafe {
        talu_sys::talu_tokenizer_decode(
            ctx.handle(),
            ptr::null(),
            3,
            &talu_sys::DecodeOptionsC::default(),
        )
    };
    assert!(
        !result.error_msg.is_null(),
        "decode with null tokens and non-zero length must fail"
    );
    assert!(result.text.is_null(), "text should remain null on error");
    assert_eq!(result.text_len, 0);
    assert_eq!(
        unsafe { talu_sys::talu_last_error_code() },
        talu_sys::ErrorCode::InvalidArgument as i32,
        "decode null tokens + non-zero len must set InvalidArgument"
    );
}

/// Encoding with null text pointer and zero length should be empty success.
#[test]
fn encode_null_text_zero_len_is_empty_success() {
    let ctx = TokenizerTestContext::new();
    let result = unsafe { common::encode_raw_null_options(ctx.handle(), &[]) };
    assert!(result.error_msg.is_null(), "zero-length encode should succeed");
    assert!(
        !result.ids.is_null(),
        "empty encode should expose sliceable sentinel buffers"
    );
    assert!(
        !result.offsets.is_null(),
        "empty encode should expose sliceable sentinel offsets"
    );
    assert!(
        !result.attention_mask.is_null(),
        "empty encode should expose sliceable sentinel attention mask"
    );
    assert!(
        !result.special_tokens_mask.is_null(),
        "empty encode should expose sliceable sentinel special-token mask"
    );
    assert_eq!(result.num_tokens, 0);
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Encoding with a null text pointer and non-zero length must return
/// InvalidArgument rather than reading through NULL.
#[test]
fn encode_null_text_nonzero_len_returns_error() {
    let ctx = TokenizerTestContext::new();
    let result = unsafe {
        talu_sys::talu_tokenizer_encode(
            ctx.handle(),
            ptr::null(),
            5,
            &talu_sys::EncodeOptions::default(),
        )
    };
    assert!(
        !result.error_msg.is_null(),
        "encode with null text and non-zero length must fail"
    );
    assert!(result.ids.is_null(), "ids should remain null on error");
    assert_eq!(result.num_tokens, 0);
    assert_eq!(
        unsafe { talu_sys::talu_last_error_code() },
        talu_sys::ErrorCode::InvalidArgument as i32,
        "encode null text + non-zero len must set InvalidArgument"
    );
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
            ptrs.as_ptr(),
            lengths.as_ptr(),
            1,
            &talu_sys::EncodeOptions::default(),
        )
    };
    assert!(
        !result.error_msg.is_null(),
        "encode_batch with null handle should set error_msg"
    );
    assert_eq!(result.total_tokens, 0);
}

/// Batch encoding with null text and length arrays and zero texts should be empty success.
#[test]
fn encode_batch_null_arrays_zero_count_is_empty_success() {
    let ctx = TokenizerTestContext::new();
    let result = unsafe {
        talu_sys::talu_tokenizer_encode_batch(
            ctx.handle(),
            ptr::null(),
            ptr::null(),
            0,
            &talu_sys::EncodeOptions::default(),
        )
    };
    assert!(result.error_msg.is_null(), "zero-count batch encode should succeed");
    assert!(result.ids.is_null(), "empty batch encode should have null ids");
    assert!(result.offsets.is_null(), "empty batch encode should have null offsets");
    assert_eq!(result.total_tokens, 0);
    assert_eq!(result.num_sequences, 0);
}

/// Batch encoding with null text array and non-zero count must return
/// InvalidArgument rather than dereferencing a null pointer.
#[test]
fn encode_batch_null_texts_nonzero_count_returns_error() {
    let ctx = TokenizerTestContext::new();
    let lengths = [5usize];
    let result = unsafe {
        talu_sys::talu_tokenizer_encode_batch(
            ctx.handle(),
            ptr::null(),
            lengths.as_ptr(),
            1,
            &talu_sys::EncodeOptions::default(),
        )
    };
    assert!(
        !result.error_msg.is_null(),
        "batch encode with null texts and non-zero count must fail"
    );
    assert!(result.ids.is_null(), "ids should remain null on error");
    assert!(result.offsets.is_null(), "offsets should remain null on error");
    assert_eq!(result.total_tokens, 0);
    assert_eq!(result.num_sequences, 0);
    assert_eq!(
        unsafe { talu_sys::talu_last_error_code() },
        talu_sys::ErrorCode::InvalidArgument as i32,
        "batch encode null texts + non-zero count must set InvalidArgument"
    );
}

/// Batch encoding with null lengths array and non-zero count must return
/// InvalidArgument rather than dereferencing a null pointer.
#[test]
fn encode_batch_null_lengths_nonzero_count_returns_error() {
    let ctx = TokenizerTestContext::new();
    let text = b"Hello";
    let ptrs = [text.as_ptr()];
    let result = unsafe {
        talu_sys::talu_tokenizer_encode_batch(
            ctx.handle(),
            ptrs.as_ptr(),
            ptr::null(),
            1,
            &talu_sys::EncodeOptions::default(),
        )
    };
    assert!(
        !result.error_msg.is_null(),
        "batch encode with null lengths and non-zero count must fail"
    );
    assert!(result.ids.is_null(), "ids should remain null on error");
    assert!(result.offsets.is_null(), "offsets should remain null on error");
    assert_eq!(result.total_tokens, 0);
    assert_eq!(result.num_sequences, 0);
    assert_eq!(
        unsafe { talu_sys::talu_last_error_code() },
        talu_sys::ErrorCode::InvalidArgument as i32,
        "batch encode null lengths + non-zero count must set InvalidArgument"
    );
}

// ---- Tokenize with null handle ----

/// Tokenize with null handle returns error result, not crash.
#[test]
fn tokenize_null_handle_returns_error() {
    let text = b"Hello";
    let result =
        unsafe { talu_sys::talu_tokenizer_tokenize(ptr::null_mut(), text.as_ptr(), text.len()) };
    assert!(
        !result.error_msg.is_null(),
        "tokenize with null handle should set error_msg"
    );
    let code = unsafe { talu_sys::talu_last_error_code() };
    assert_eq!(
        code,
        talu_sys::ErrorCode::InvalidHandle as i32,
        "tokenize null-handle path must set InvalidHandle code"
    );
    assert_eq!(result.num_tokens, 0);
}

/// tokenize with null text pointer and zero length should succeed with zero tokens.
#[test]
fn tokenize_null_text_zero_len_is_empty_success() {
    let json = br#"{
  "version": "1.0",
  "model": {"type":"BPE","vocab":{"a":0},"merges":[]},
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    let mut handle: *mut c_void = ptr::null_mut();
    let rc = unsafe {
        talu_sys::talu_tokenizer_create_from_json(
            json.as_ptr(),
            json.len(),
            &mut handle as *mut _ as *mut c_void,
        )
    };
    assert_eq!(rc, 0);
    assert!(!handle.is_null());

    let result = unsafe { talu_sys::talu_tokenizer_tokenize(handle, ptr::null(), 0) };
    assert!(result.error_msg.is_null(), "zero-length tokenize should succeed");
    assert_eq!(result.num_tokens, 0);
    unsafe {
        talu_sys::talu_tokenize_result_free(result.tokens, result.num_tokens);
        talu_sys::talu_tokenizer_free(handle);
    }
}

/// Tokenize with a null text pointer and non-zero length must return
/// InvalidArgument rather than reading through NULL.
#[test]
fn tokenize_null_text_nonzero_len_returns_error() {
    let ctx = TokenizerTestContext::new();
    let result = unsafe { talu_sys::talu_tokenizer_tokenize(ctx.handle(), ptr::null(), 5) };
    assert!(
        !result.error_msg.is_null(),
        "tokenize with null text and non-zero length must fail"
    );
    assert_eq!(result.num_tokens, 0);
    assert_eq!(
        unsafe { talu_sys::talu_last_error_code() },
        talu_sys::ErrorCode::InvalidArgument as i32,
        "tokenize null text + non-zero len must set InvalidArgument"
    );
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
    let code = unsafe { talu_sys::talu_last_error_code() };
    assert_eq!(
        code,
        talu_sys::ErrorCode::InvalidHandle as i32,
        "tokenize_bytes null-handle path must set InvalidHandle code"
    );
    assert_eq!(result.num_tokens, 0);
}

/// tokenize_bytes with null text pointer and zero length should succeed with zero tokens.
#[test]
fn tokenize_bytes_null_text_zero_len_is_empty_success() {
    let ctx = TokenizerTestContext::new();
    let result = unsafe { talu_sys::talu_tokenizer_tokenize_bytes(ctx.handle(), ptr::null(), 0) };
    assert!(
        result.error_msg.is_null(),
        "zero-length tokenize_bytes should succeed"
    );
    assert!(
        !result.data.is_null(),
        "empty tokenize_bytes should expose sliceable sentinel data"
    );
    assert_eq!(result.data_len, 0);
    assert_eq!(result.num_tokens, 0);
    assert!(
        !result.offsets.is_null(),
        "empty tokenize_bytes should expose a sliceable sentinel offset buffer"
    );
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens + 1) };
    assert_eq!(
        offsets,
        &[0],
        "empty tokenize_bytes should expose a single sentinel offset"
    );
    unsafe {
        talu_sys::talu_tokenize_bytes_result_free(
            result.data,
            result.data_len,
            result.offsets,
            result.num_tokens,
        )
    };
}

/// tokenize_bytes with a null text pointer and non-zero length must return
/// InvalidArgument rather than reading through NULL.
#[test]
fn tokenize_bytes_null_text_nonzero_len_returns_error() {
    let ctx = TokenizerTestContext::new();
    let result = unsafe { talu_sys::talu_tokenizer_tokenize_bytes(ctx.handle(), ptr::null(), 5) };
    assert!(
        !result.error_msg.is_null(),
        "tokenize_bytes with null text and non-zero length must fail"
    );
    assert!(result.data.is_null(), "data should remain null on error");
    assert_eq!(result.data_len, 0);
    assert_eq!(result.num_tokens, 0);
    assert_eq!(
        unsafe { talu_sys::talu_last_error_code() },
        talu_sys::ErrorCode::InvalidArgument as i32,
        "tokenize_bytes null text + non-zero len must set InvalidArgument"
    );
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
        talu_sys::talu_tokenizer_id_to_token(ptr::null_mut(), 0, &mut out as *mut _ as *mut c_void)
    };
    assert_eq!(
        rc,
        talu_sys::ErrorCode::InvalidHandle as i32,
        "id_to_token with null handle should return InvalidHandle"
    );
    assert!(out.is_null(), "output should remain null on error");
}

/// token_to_id with null handle returns error code, not crash.
#[test]
fn token_to_id_null_handle_returns_error() {
    let token = b"<s>";
    let rc = unsafe {
        talu_sys::talu_tokenizer_token_to_id(ptr::null_mut(), token.as_ptr(), token.len())
    };
    assert!(
        rc == talu_sys::ErrorCode::InvalidHandle as i32,
        "token_to_id with null handle should return InvalidHandle, got {rc}"
    );
}

// ---- Free functions with null ----

/// Freeing null tokens is a no-op.
#[test]
fn tokens_free_null_is_noop() {
    unsafe { talu_sys::talu_tokens_free(ptr::null_mut(), 0) };
}

/// Freeing null decode result is a no-op.
#[test]
fn decode_result_free_null_is_noop() {
    unsafe { talu_sys::talu_decode_result_free(ptr::null_mut(), 0) };
}

/// Freeing null tokenize result is a no-op.
#[test]
fn tokenize_result_free_null_is_noop() {
    unsafe { talu_sys::talu_tokenize_result_free(ptr::null_mut(), 0) };
}

/// Freeing null tokenize bytes result is a no-op.
#[test]
fn tokenize_bytes_result_free_null_is_noop() {
    unsafe { talu_sys::talu_tokenize_bytes_result_free(ptr::null_mut(), 0, ptr::null_mut(), 0) };
}

/// Freeing null batch encode result is a no-op.
#[test]
fn batch_encode_result_free_null_is_noop() {
    unsafe { talu_sys::talu_batch_encode_result_free(ptr::null_mut(), ptr::null_mut(), 0, 0) };
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
    unsafe { talu_sys::talu_padded_tensor_result_free(ptr::null_mut(), ptr::null_mut(), 0, 0) };
}

/// batch_to_padded_tensor with null ids and non-zero sequence count returns error.
#[test]
fn batch_to_padded_tensor_null_ids_returns_error() {
    let offsets = [0usize, 0usize];
    let result = unsafe {
        talu_sys::talu_batch_to_padded_tensor(
            ptr::null(),
            offsets.as_ptr(),
            1,
            &talu_sys::PaddedTensorOptions::default(),
        )
    };
    assert!(
        !result.error_msg.is_null(),
        "null ids with non-zero sequences must return an error"
    );
    assert!(result.input_ids.is_null(), "input_ids must be null on error");
}

/// batch_to_padded_tensor with null offsets and non-zero sequence count returns error.
#[test]
fn batch_to_padded_tensor_null_offsets_returns_error() {
    let ids = [44u32, 77u32];
    let result = unsafe {
        talu_sys::talu_batch_to_padded_tensor(
            ids.as_ptr(),
            ptr::null(),
            1,
            &talu_sys::PaddedTensorOptions::default(),
        )
    };
    assert!(
        !result.error_msg.is_null(),
        "null offsets with non-zero sequences must return an error"
    );
    assert!(result.input_ids.is_null(), "input_ids must be null on error");
}

/// batch_to_padded_tensor with zero sequences should return an empty success result.
#[test]
fn batch_to_padded_tensor_zero_sequences_is_empty_success() {
    let result = unsafe {
        talu_sys::talu_batch_to_padded_tensor(
            ptr::null(),
            ptr::null(),
            0,
            &talu_sys::PaddedTensorOptions::default(),
        )
    };
    assert!(result.error_msg.is_null(), "zero-sequence call should succeed");
    assert!(result.input_ids.is_null(), "input_ids should be null for empty result");
    assert_eq!(result.num_sequences, 0);
    assert_eq!(result.padded_length, 0);
}

/// Freeing null vocab result is a no-op.
#[test]
fn vocab_result_free_null_is_noop() {
    unsafe { talu_sys::talu_vocab_result_free(ptr::null_mut(), ptr::null_mut(), ptr::null_mut(), 0) };
}

/// talu_take_last_error must report, then consume and clear tokenizer errors.
#[test]
fn take_last_error_query_then_consume_for_tokenizer_error() {
    let text = b"Hello";
    let _ = unsafe {
        talu_sys::talu_tokenizer_encode(
            ptr::null_mut(),
            text.as_ptr(),
            text.len(),
            &talu_sys::EncodeOptions::default(),
        )
    };

    // Query mode: returns required size and code, but does not clear.
    let mut query_code: i32 = 0;
    let required = unsafe {
        talu_sys::talu_take_last_error(ptr::null_mut(), 0, &mut query_code as *mut _ as *mut c_void)
    };
    assert_eq!(
        query_code,
        talu_sys::ErrorCode::InvalidHandle as i32,
        "query mode must return InvalidHandle for null tokenizer handle"
    );
    assert!(required > 1, "required size should include at least one byte plus NUL");
    assert_eq!(
        unsafe { talu_sys::talu_last_error_code() },
        talu_sys::ErrorCode::InvalidHandle as i32,
        "query mode must not clear error state"
    );

    // Consume mode: copies message and clears error state.
    let mut buf = vec![0u8; required];
    let mut consume_code: i32 = 0;
    let copied = unsafe {
        talu_sys::talu_take_last_error(
            buf.as_mut_ptr(),
            buf.len(),
            &mut consume_code as *mut _ as *mut c_void,
        )
    };
    assert_eq!(
        consume_code,
        talu_sys::ErrorCode::InvalidHandle as i32,
        "consume mode must return InvalidHandle for null tokenizer handle"
    );
    assert!(copied > 0, "consume mode must copy a non-empty error message");
    assert_eq!(
        unsafe { talu_sys::talu_last_error_code() },
        talu_sys::ErrorCode::Ok as i32,
        "consume mode must clear error state"
    );
}
