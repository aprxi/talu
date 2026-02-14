//! Vocabulary access tests.
//!
//! Validates vocab enumeration, special token queries, and bidirectional ID↔token mapping.

use crate::capi::tokenizer::common::TokenizerTestContext;
use std::ffi::c_void;
use std::ptr;

/// get_vocab returns all 99 entries with IDs 0..99.
#[test]
fn get_vocab_all_entries() {
    let ctx = TokenizerTestContext::new();
    let result = unsafe { talu_sys::talu_tokenizer_get_vocab(ctx.handle()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_entries, 99);

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_entries) };
    let mut sorted_ids = ids.to_vec();
    sorted_ids.sort();
    let expected: Vec<u32> = (0..99).collect();
    assert_eq!(sorted_ids, expected);

    unsafe {
        talu_sys::talu_vocab_result_free(
            result.tokens,
            result.lengths,
            result.ids,
            result.num_entries,
        )
    };
}

/// get_vocab_size matches get_vocab num_entries.
#[test]
fn vocab_size_matches_vocab() {
    let ctx = TokenizerTestContext::new();
    let size = unsafe { talu_sys::talu_tokenizer_get_vocab_size(ctx.handle()) };
    let result = unsafe { talu_sys::talu_tokenizer_get_vocab(ctx.handle()) };
    assert_eq!(size, result.num_entries);
    unsafe {
        talu_sys::talu_vocab_result_free(
            result.tokens,
            result.lengths,
            result.ids,
            result.num_entries,
        )
    };
}

/// id_to_token for all four special tokens.
#[test]
fn id_to_token_special_tokens() {
    let ctx = TokenizerTestContext::new();
    let expected = [(0, "<pad>"), (1, "<s>"), (2, "</s>"), (3, "<unk>")];

    for (id, text) in expected {
        let mut out: *mut i8 = ptr::null_mut();
        let rc = unsafe {
            talu_sys::talu_tokenizer_id_to_token(
                ctx.handle(),
                id,
                &mut out as *mut _ as *mut c_void,
            )
        };
        assert_eq!(rc, 0, "id_to_token({id}) failed");
        let actual = unsafe { std::ffi::CStr::from_ptr(out) }
            .to_string_lossy()
            .to_string();
        assert_eq!(actual, text, "token {id} mismatch");
        unsafe { talu_sys::talu_text_free(out) };
    }
}

/// id_to_token for out-of-range ID returns error and null output.
#[test]
fn id_to_token_out_of_range() {
    let ctx = TokenizerTestContext::new();
    let mut out: *mut i8 = ptr::null_mut();
    let rc = unsafe {
        talu_sys::talu_tokenizer_id_to_token(ctx.handle(), 9999, &mut out as *mut _ as *mut c_void)
    };
    assert_ne!(rc, 0);
    assert!(out.is_null());
}

/// token_to_id for special tokens returns exact IDs.
#[test]
fn token_to_id_special_tokens() {
    let ctx = TokenizerTestContext::new();
    let expected = [("<pad>", 0), ("<s>", 1), ("</s>", 2), ("<unk>", 3)];

    for (token, id) in expected {
        let actual = unsafe {
            talu_sys::talu_tokenizer_token_to_id(
                ctx.handle(),
                token.as_bytes().as_ptr(),
                token.len(),
            )
        };
        assert_eq!(actual, id, "token_to_id({token:?}) mismatch");
    }
}

/// token_to_id for unknown token returns -1.
#[test]
fn token_to_id_unknown() {
    let ctx = TokenizerTestContext::new();
    let id = unsafe {
        talu_sys::talu_tokenizer_token_to_id(ctx.handle(), b"nonexistent_xyz".as_ptr(), 15)
    };
    assert_eq!(id, -1);
}

/// token_to_id for known regular ASCII tokens returns correct IDs.
#[test]
fn token_to_id_regular_ascii_tokens() {
    let ctx = TokenizerTestContext::new();
    let expected = [("a", 69), ("b", 70), ("H", 44), ("0", 20), ("!", 5)];

    for (token, id) in expected {
        let actual = unsafe {
            talu_sys::talu_tokenizer_token_to_id(
                ctx.handle(),
                token.as_bytes().as_ptr(),
                token.len(),
            )
        };
        assert_eq!(actual, id, "token_to_id({token:?}) mismatch");
    }
}

/// id_to_token and token_to_id are inverses for all 99 vocab entries.
#[test]
fn id_token_roundtrip_all() {
    let ctx = TokenizerTestContext::new();

    for token_id in 0..99i32 {
        // ID → token string
        let mut out: *mut i8 = ptr::null_mut();
        let rc = unsafe {
            talu_sys::talu_tokenizer_id_to_token(
                ctx.handle(),
                token_id,
                &mut out as *mut _ as *mut c_void,
            )
        };
        assert_eq!(rc, 0, "id_to_token({token_id}) failed");

        // token string → ID
        let text = unsafe { std::ffi::CStr::from_ptr(out) };
        let bytes = text.to_bytes();
        let roundtrip_id = unsafe {
            talu_sys::talu_tokenizer_token_to_id(ctx.handle(), bytes.as_ptr(), bytes.len())
        };
        assert_eq!(roundtrip_id, token_id, "roundtrip failed for ID {token_id}");

        unsafe { talu_sys::talu_text_free(out) };
    }
}

// ===========================================================================
// Byte-level fixture vocab (260 tokens)
// ===========================================================================

/// Byte-level fixture has 260 tokens (4 special + 256 byte-level).
#[test]
fn byte_level_vocab_entries() {
    let ctx = TokenizerTestContext::with_byte_level();
    let result = unsafe { talu_sys::talu_tokenizer_get_vocab(ctx.handle()) };
    assert!(result.error_msg.is_null());
    assert_eq!(
        result.num_entries, 260,
        "byte-level vocab should have 260 entries"
    );

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_entries) };
    let mut sorted_ids = ids.to_vec();
    sorted_ids.sort();
    let expected: Vec<u32> = (0..260).collect();
    assert_eq!(sorted_ids, expected, "IDs should be contiguous 0..260");

    unsafe {
        talu_sys::talu_vocab_result_free(
            result.tokens,
            result.lengths,
            result.ids,
            result.num_entries,
        )
    };
}

/// id↔token roundtrip for all 260 entries in byte-level fixture.
#[test]
fn byte_level_id_token_roundtrip() {
    let ctx = TokenizerTestContext::with_byte_level();

    for token_id in 0..260i32 {
        let mut out: *mut i8 = ptr::null_mut();
        let rc = unsafe {
            talu_sys::talu_tokenizer_id_to_token(
                ctx.handle(),
                token_id,
                &mut out as *mut _ as *mut c_void,
            )
        };
        assert_eq!(rc, 0, "id_to_token({token_id}) failed");

        let text = unsafe { std::ffi::CStr::from_ptr(out) };
        let bytes = text.to_bytes();
        let roundtrip_id = unsafe {
            talu_sys::talu_tokenizer_token_to_id(ctx.handle(), bytes.as_ptr(), bytes.len())
        };
        assert_eq!(roundtrip_id, token_id, "roundtrip failed for ID {token_id}");

        unsafe { talu_sys::talu_text_free(out) };
    }
}

// ===========================================================================
// Merges fixture vocab (multi-char tokens)
// ===========================================================================

/// Merged tokens are accessible via id_to_token.
#[test]
fn merged_token_id_to_token() {
    let ctx = TokenizerTestContext::with_merges();
    let expected = [
        (99, "he"),
        (100, "ll"),
        (101, "lo"),
        (102, "hel"),
        (103, "hell"),
        (104, "hello"),
    ];

    for (id, text) in expected {
        let mut out: *mut i8 = ptr::null_mut();
        let rc = unsafe {
            talu_sys::talu_tokenizer_id_to_token(
                ctx.handle(),
                id,
                &mut out as *mut _ as *mut c_void,
            )
        };
        assert_eq!(rc, 0, "id_to_token({id}) failed");
        let actual = unsafe { std::ffi::CStr::from_ptr(out) }
            .to_string_lossy()
            .to_string();
        assert_eq!(actual, text, "merged token {id} mismatch");
        unsafe { talu_sys::talu_text_free(out) };
    }
}

/// token_to_id for multi-char merged tokens returns correct IDs.
#[test]
fn merged_token_to_id() {
    let ctx = TokenizerTestContext::with_merges();
    let expected = [("he", 99), ("ll", 100), ("lo", 101), ("hello", 104)];

    for (token, id) in expected {
        let actual = unsafe {
            talu_sys::talu_tokenizer_token_to_id(
                ctx.handle(),
                token.as_bytes().as_ptr(),
                token.len(),
            )
        };
        assert_eq!(actual, id, "token_to_id({token:?}) mismatch");
    }
}

/// Merges fixture has 105 total tokens (99 base + 6 merged).
#[test]
fn merges_vocab_size() {
    let ctx = TokenizerTestContext::with_merges();
    let size = unsafe { talu_sys::talu_tokenizer_get_vocab_size(ctx.handle()) };
    assert_eq!(size, 105);
}

// ===========================================================================
// Boundary and edge cases
// ===========================================================================

/// token_to_id for empty string returns -1 (not found).
#[test]
fn token_to_id_empty_string() {
    let ctx = TokenizerTestContext::new();
    let id = unsafe { talu_sys::talu_tokenizer_token_to_id(ctx.handle(), b"".as_ptr(), 0) };
    assert_eq!(id, -1, "empty string should not match any token");
}

/// id_to_token for ID 0 (first valid ID) succeeds.
#[test]
fn id_to_token_boundary_zero() {
    let ctx = TokenizerTestContext::new();
    let mut out: *mut i8 = ptr::null_mut();
    let rc = unsafe {
        talu_sys::talu_tokenizer_id_to_token(ctx.handle(), 0, &mut out as *mut _ as *mut c_void)
    };
    assert_eq!(rc, 0, "ID 0 should be valid");
    let text = unsafe { std::ffi::CStr::from_ptr(out) }
        .to_string_lossy()
        .to_string();
    assert_eq!(text, "<pad>");
    unsafe { talu_sys::talu_text_free(out) };
}

/// id_to_token for last valid ID succeeds; ID+1 fails.
#[test]
fn id_to_token_boundary_last() {
    let ctx = TokenizerTestContext::new();

    // Last valid ID = 98
    let mut out: *mut i8 = ptr::null_mut();
    let rc = unsafe {
        talu_sys::talu_tokenizer_id_to_token(ctx.handle(), 98, &mut out as *mut _ as *mut c_void)
    };
    assert_eq!(rc, 0, "last ID (98) should be valid");
    unsafe { talu_sys::talu_text_free(out) };

    // ID 99 is out of range for base fixture.
    out = ptr::null_mut();
    let rc2 = unsafe {
        talu_sys::talu_tokenizer_id_to_token(ctx.handle(), 99, &mut out as *mut _ as *mut c_void)
    };
    assert_ne!(rc2, 0, "ID 99 should be out of range");
}

/// Special-tokens fixture: added-only special tokens are NOT in model vocab.
///
/// When special tokens exist only in `added_tokens` (not `model.vocab`),
/// `token_to_id` returns -1 for them. This is the trade-off that enables
/// observable `skip_special_tokens` behavior during decode.
#[test]
fn special_tokens_fixture_added_only_not_in_vocab() {
    let ctx = TokenizerTestContext::with_special_tokens();
    let size = unsafe { talu_sys::talu_tokenizer_get_vocab_size(ctx.handle()) };
    // 95 regular tokens + 4 added special tokens = 99 total.
    assert_eq!(size, 99);

    // Special tokens are only in added_tokens, so token_to_id returns -1.
    for token in ["<pad>", "<s>", "</s>", "<unk>"] {
        let id = unsafe {
            talu_sys::talu_tokenizer_token_to_id(
                ctx.handle(),
                token.as_bytes().as_ptr(),
                token.len(),
            )
        };
        assert_eq!(
            id, -1,
            "added-only token {token:?} should not be found by token_to_id"
        );
    }

    // But regular tokens ARE found.
    let id = unsafe { talu_sys::talu_tokenizer_token_to_id(ctx.handle(), b"H".as_ptr(), 1) };
    assert_eq!(id, 44, "regular token 'H' should be in vocab");
}

/// Vocab across different fixtures: each fixture has the expected size.
#[test]
fn vocab_size_across_fixtures() {
    let sizes = [
        ("base", TokenizerTestContext::new(), 99),
        ("merges", TokenizerTestContext::with_merges(), 105),
        ("byte_level", TokenizerTestContext::with_byte_level(), 260),
        (
            "special_tokens",
            TokenizerTestContext::with_special_tokens(),
            99,
        ),
    ];
    for (name, ctx, expected) in sizes {
        let size = unsafe { talu_sys::talu_tokenizer_get_vocab_size(ctx.handle()) };
        assert_eq!(size, expected, "{name} fixture vocab size mismatch");
    }
}
