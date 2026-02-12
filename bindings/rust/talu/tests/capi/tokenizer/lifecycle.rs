//! Tokenizer creation, metadata access, and destruction tests.
//!
//! Validates: create_from_json -> query metadata -> free.
//! Token IDs for this fixture: <pad>=0, <s>=1, </s>=2, <unk>=3,
//! then ASCII 0x20..0x7E at IDs 4..98 (99 total).

use crate::capi::tokenizer::common::TokenizerTestContext;
use std::ffi::c_void;
use std::ptr;

/// Creating a tokenizer from the minimal JSON fixture succeeds and produces
/// a working handle that can encode text.
#[test]
fn create_from_json_and_encode() {
    let ctx = TokenizerTestContext::new();
    // Verify the handle works by encoding a known input.
    let tokens = ctx.encode("Hi");
    assert_eq!(tokens, [44, 77], "H=44, i=77");
}

/// Vocab size matches the fixture: 4 special + 95 ASCII = 99.
#[test]
fn vocab_size_matches_fixture() {
    let ctx = TokenizerTestContext::new();
    let size = unsafe { talu_sys::talu_tokenizer_get_vocab_size(ctx.handle()) };
    assert_eq!(size, 99);
}

/// model_max_length returns 0 for JSON-created tokenizer (no config file).
#[test]
fn model_max_length_zero_without_config() {
    let ctx = TokenizerTestContext::new();
    let max_len = unsafe { talu_sys::talu_tokenizer_get_model_max_length(ctx.handle()) };
    assert_eq!(max_len, 0);
}

/// model_dir returns empty string for JSON-created tokenizer.
#[test]
fn model_dir_empty_for_json_tokenizer() {
    let ctx = TokenizerTestContext::new();
    let mut out_path: *mut i8 = ptr::null_mut();
    let rc = unsafe {
        talu_sys::talu_tokenizer_get_model_dir(
            ctx.handle(),
            &mut out_path as *mut _ as *mut c_void,
        )
    };
    assert_eq!(rc, 0);
    assert!(!out_path.is_null());

    let path = unsafe { std::ffi::CStr::from_ptr(out_path) }
        .to_string_lossy()
        .to_string();
    assert_eq!(path, "");
    unsafe { talu_sys::talu_text_free(out_path) };
}

/// Creating from invalid JSON returns error and leaves handle null.
#[test]
fn create_from_invalid_json_returns_error() {
    let json = b"not valid json";
    let mut handle: *mut c_void = ptr::null_mut();
    let rc = unsafe {
        talu_sys::talu_tokenizer_create_from_json(
            json.as_ptr(),
            json.len(),
            &mut handle as *mut _ as *mut c_void,
        )
    };
    assert_ne!(rc, 0);
    assert!(handle.is_null(), "handle must remain null on error");
}

/// Creating from empty JSON returns error and leaves handle null.
#[test]
fn create_from_empty_json_returns_error() {
    let json = b"";
    let mut handle: *mut c_void = ptr::null_mut();
    let rc = unsafe {
        talu_sys::talu_tokenizer_create_from_json(
            json.as_ptr(),
            0,
            &mut handle as *mut _ as *mut c_void,
        )
    };
    assert_ne!(rc, 0);
    assert!(handle.is_null(), "handle must remain null on error");
}

/// Two tokenizers from the same JSON operate independently.
#[test]
fn multiple_tokenizers_independent() {
    let a = TokenizerTestContext::new();
    let b = TokenizerTestContext::new();

    // Same input, same output — both are functional.
    let tokens_a = a.encode("Hi");
    let tokens_b = b.encode("Hi");
    assert_eq!(tokens_a, [44, 77]);
    assert_eq!(tokens_b, [44, 77]);

    // Handles are distinct allocations.
    assert_ne!(a.handle(), b.handle());
}

/// special_tokens returns correct IDs for the minimal fixture.
#[test]
fn special_tokens_match_fixture() {
    let ctx = TokenizerTestContext::new();
    let result = unsafe { talu_sys::talu_tokenizer_get_special_tokens(ctx.handle()) };
    assert_eq!(result.bos_token_id, 1, "BOS should be <s>=1");
    assert_eq!(result.unk_token_id, 3, "UNK should be <unk>=3");
    assert_eq!(result.pad_token_id, 0, "PAD should be <pad>=0");
}

/// EOS tokens are empty for JSON-created tokenizer (no generation config).
#[test]
fn eos_tokens_empty_without_gen_config() {
    let ctx = TokenizerTestContext::new();
    let result = unsafe { talu_sys::talu_tokenizer_get_eos_tokens(ctx.handle()) };
    assert_eq!(result.num_tokens, 0);
    assert!(result.tokens.is_null());
}
