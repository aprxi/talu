//! Offset computation and tokenize (string representation) tests.
//!
//! Validates compute_offsets, tokenize, and tokenize_bytes with exact assertions.

use crate::capi::tokenizer::common::TokenizerTestContext;

/// compute_offsets for "Hello" returns per-character byte spans [0,1)...[4,5).
#[test]
fn compute_offsets_hello() {
    let ctx = TokenizerTestContext::new();
    let text = "Hello";
    let result = unsafe {
        talu_sys::talu_tokenizer_compute_offsets(
            ctx.handle(),
            text.as_bytes().as_ptr(),
            text.len(),
        )
    };
    assert!(result.error_msg.is_null());
    assert_eq!(result.len, 5);

    let offsets =
        unsafe { std::slice::from_raw_parts(result.offsets, result.len) };

    // Each character occupies exactly one byte.
    for (i, off) in offsets.iter().enumerate() {
        assert_eq!(off.start as usize, i, "offset[{i}].start");
        assert_eq!(off.end as usize, i + 1, "offset[{i}].end");
    }

    unsafe { talu_sys::talu_offsets_free(result) };
}

/// compute_offsets on empty string returns zero offsets.
#[test]
fn compute_offsets_empty() {
    let ctx = TokenizerTestContext::new();
    let result = unsafe {
        talu_sys::talu_tokenizer_compute_offsets(ctx.handle(), [].as_ptr(), 0)
    };
    assert!(result.error_msg.is_null());
    assert_eq!(result.len, 0);
    unsafe { talu_sys::talu_offsets_free(result) };
}

/// compute_offsets produces non-overlapping, contiguous spans.
#[test]
fn compute_offsets_contiguous() {
    let ctx = TokenizerTestContext::new();
    let text = "abc";
    let result = unsafe {
        talu_sys::talu_tokenizer_compute_offsets(
            ctx.handle(),
            text.as_bytes().as_ptr(),
            text.len(),
        )
    };
    assert!(result.error_msg.is_null());
    let offsets =
        unsafe { std::slice::from_raw_parts(result.offsets, result.len) };

    // Verify contiguous: each offset.start == previous offset.end.
    assert_eq!(offsets[0].start, 0);
    for w in offsets.windows(2) {
        assert_eq!(w[0].end, w[1].start, "offsets should be contiguous");
    }
    assert_eq!(offsets.last().unwrap().end as usize, text.len());

    unsafe { talu_sys::talu_offsets_free(result) };
}

/// tokenize "Hi" returns string representations ["H", "i"].
#[test]
fn tokenize_hi_strings() {
    let ctx = TokenizerTestContext::new();
    let text = "Hi";
    let result = unsafe {
        talu_sys::talu_tokenizer_tokenize(
            ctx.handle(),
            text.as_bytes().as_ptr(),
            text.len(),
        )
    };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 2);

    let token_ptrs = unsafe {
        std::slice::from_raw_parts(result.tokens as *const *const i8, result.num_tokens)
    };
    let t0 = unsafe { std::ffi::CStr::from_ptr(token_ptrs[0]) }
        .to_string_lossy()
        .to_string();
    let t1 = unsafe { std::ffi::CStr::from_ptr(token_ptrs[1]) }
        .to_string_lossy()
        .to_string();
    assert_eq!(t0, "H");
    assert_eq!(t1, "i");

    unsafe { talu_sys::talu_tokenize_result_free(result.tokens, result.num_tokens) };
}

/// tokenize empty string returns zero tokens.
#[test]
fn tokenize_empty() {
    let ctx = TokenizerTestContext::new();
    let result = unsafe {
        talu_sys::talu_tokenizer_tokenize(ctx.handle(), [].as_ptr(), 0)
    };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 0);
    unsafe { talu_sys::talu_tokenize_result_free(result.tokens, result.num_tokens) };
}

/// tokenize_bytes for "Hi" returns contiguous buffer with 2 tokens.
#[test]
fn tokenize_bytes_hi() {
    let ctx = TokenizerTestContext::new();
    let text = "Hi";
    let result = unsafe {
        talu_sys::talu_tokenizer_tokenize_bytes(
            ctx.handle(),
            text.as_bytes().as_ptr(),
            text.len(),
        )
    };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 2);

    // Offsets array has num_tokens+1 entries.
    let offsets = unsafe {
        std::slice::from_raw_parts(result.offsets, result.num_tokens + 1)
    };
    assert_eq!(offsets[0], 0);
    assert_eq!(*offsets.last().unwrap(), result.data_len);

    // Each token's bytes should be "H" and "i".
    let data = unsafe { std::slice::from_raw_parts(result.data, result.data_len) };
    let t0 = std::str::from_utf8(&data[offsets[0]..offsets[1]]).unwrap();
    let t1 = std::str::from_utf8(&data[offsets[1]..offsets[2]]).unwrap();
    assert_eq!(t0, "H");
    assert_eq!(t1, "i");

    unsafe {
        talu_sys::talu_tokenize_bytes_result_free(
            result.data,
            result.data_len,
            result.offsets,
            result.num_tokens,
        )
    };
}

/// compute_offsets for single character "A" returns one offset [0, 1).
#[test]
fn compute_offsets_single_char() {
    let ctx = TokenizerTestContext::new();
    let text = "A";
    let result = unsafe {
        talu_sys::talu_tokenizer_compute_offsets(
            ctx.handle(),
            text.as_bytes().as_ptr(),
            text.len(),
        )
    };
    assert!(result.error_msg.is_null());
    assert_eq!(result.len, 1);

    let offsets =
        unsafe { std::slice::from_raw_parts(result.offsets, result.len) };
    assert_eq!(offsets[0].start, 0);
    assert_eq!(offsets[0].end, 1);

    unsafe { talu_sys::talu_offsets_free(result) };
}

/// tokenize_bytes on empty input returns zero tokens.
#[test]
fn tokenize_bytes_empty() {
    let ctx = TokenizerTestContext::new();
    let result = unsafe {
        talu_sys::talu_tokenizer_tokenize_bytes(ctx.handle(), [].as_ptr(), 0)
    };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 0);
    unsafe {
        talu_sys::talu_tokenize_bytes_result_free(
            result.data,
            result.data_len,
            result.offsets,
            result.num_tokens,
        )
    };
}

/// tokenize_bytes token count matches encode token count for same input.
#[test]
fn tokenize_bytes_count_matches_encode() {
    let ctx = TokenizerTestContext::new();
    let text = "Hello";
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    let encode_tokens = ctx.encode_with(text, &opts);

    let result = unsafe {
        talu_sys::talu_tokenizer_tokenize_bytes(
            ctx.handle(),
            text.as_bytes().as_ptr(),
            text.len(),
        )
    };
    assert!(result.error_msg.is_null());
    assert_eq!(
        result.num_tokens,
        encode_tokens.len(),
        "tokenize_bytes and encode should agree on token count"
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
