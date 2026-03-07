//! Offset, attention_mask, and special_tokens_mask tests.
//!
//! Validates encode offsets, tokenize, tokenize_bytes, and the new EncodeResult
//! fields (attention_mask, special_tokens_mask) with exact assertions.

use crate::capi::tokenizer::common::{build_byte_level_tokenizer_json, TokenizerTestContext};

fn no_bos() -> talu_sys::EncodeOptions {
    talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    }
}

/// Encoding "Hello" returns per-character byte spans [0,1)...[4,5).
#[test]
fn encode_offsets_hello() {
    let ctx = TokenizerTestContext::new();
    let text = "Hello";
    let result = unsafe { super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 5);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };

    // Each character occupies exactly one byte.
    for (i, off) in offsets.iter().enumerate() {
        assert_eq!(off.start as usize, i, "offset[{i}].start");
        assert_eq!(off.end as usize, i + 1, "offset[{i}].end");
    }

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Encoding empty string returns zero tokens.
#[test]
fn encode_offsets_empty() {
    let ctx = TokenizerTestContext::new();
    let result = unsafe { super::common::encode_raw(ctx.handle(), &[], &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 0);
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Encoding produces non-overlapping, contiguous offset spans.
#[test]
fn encode_offsets_contiguous() {
    let ctx = TokenizerTestContext::new();
    let text = "abc";
    let result = unsafe { super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
    assert!(result.error_msg.is_null());
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };

    // Verify contiguous: each offset.start == previous offset.end.
    assert_eq!(offsets[0].start, 0);
    for w in offsets.windows(2) {
        assert_eq!(w[0].end, w[1].start, "offsets should be contiguous");
    }
    assert_eq!(offsets.last().unwrap().end as usize, text.len());

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// tokenize "Hi" returns string representations ["H", "i"].
#[test]
fn tokenize_hi_strings() {
    let ctx = TokenizerTestContext::new();
    let text = "Hi";
    let result = unsafe {
        talu_sys::talu_tokenizer_tokenize(ctx.handle(), text.as_bytes().as_ptr(), text.len())
    };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 2);

    let token_ptrs =
        unsafe { std::slice::from_raw_parts(result.tokens as *const *const i8, result.num_tokens) };
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
    let result = unsafe { talu_sys::talu_tokenizer_tokenize(ctx.handle(), [].as_ptr(), 0) };
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
        talu_sys::talu_tokenizer_tokenize_bytes(ctx.handle(), text.as_bytes().as_ptr(), text.len())
    };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 2);

    // Offsets array has num_tokens+1 entries.
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens + 1) };
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

/// Encoding single character "A" returns one offset [0, 1).
#[test]
fn encode_offsets_single_char() {
    let ctx = TokenizerTestContext::new();
    let text = "A";
    let result = unsafe { super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 1);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!(offsets[0].start, 0);
    assert_eq!(offsets[0].end, 1);

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// tokenize_bytes on empty input returns zero tokens.
#[test]
fn tokenize_bytes_empty() {
    let ctx = TokenizerTestContext::new();
    let result = unsafe { talu_sys::talu_tokenizer_tokenize_bytes(ctx.handle(), [].as_ptr(), 0) };
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
        talu_sys::talu_tokenizer_tokenize_bytes(ctx.handle(), text.as_bytes().as_ptr(), text.len())
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

// ===========================================================================
// attention_mask: always 1 for all tokens in single encode
// ===========================================================================

/// attention_mask for "Hello" is all 1s (5 tokens, all real).
#[test]
fn attention_mask_all_ones_hello() {
    let ctx = TokenizerTestContext::new();
    let text = "Hello";
    let result = unsafe { super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 5);

    let mask = unsafe { std::slice::from_raw_parts(result.attention_mask, result.num_tokens) };
    for (i, &m) in mask.iter().enumerate() {
        assert_eq!(m, 1, "attention_mask[{i}] should be 1");
    }

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// attention_mask is a sliceable empty sentinel for empty input.
#[test]
fn attention_mask_empty_input() {
    let ctx = TokenizerTestContext::new();
    let result = unsafe { super::common::encode_raw(ctx.handle(), &[], &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 0);
    assert!(!result.attention_mask.is_null());
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// attention_mask for byte-level fixture with emoji is all 1s.
#[test]
fn attention_mask_byte_level_emoji() {
    let ctx = TokenizerTestContext::with_byte_level();
    let text = "Hi😊";
    let result = unsafe { super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, text.len()); // 2 + 4 = 6 bytes

    let mask = unsafe { std::slice::from_raw_parts(result.attention_mask, result.num_tokens) };
    assert!(
        mask.iter().all(|&m| m == 1),
        "all attention_mask values should be 1"
    );

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// attention_mask for single token is [1].
#[test]
fn attention_mask_single_token() {
    let ctx = TokenizerTestContext::new();
    let result = unsafe { super::common::encode_raw(ctx.handle(), b"A", &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 1);

    let mask = unsafe { std::slice::from_raw_parts(result.attention_mask, result.num_tokens) };
    assert_eq!(mask, &[1]);

    unsafe { talu_sys::talu_encode_result_free(result) };
}

// ===========================================================================
// special_tokens_mask: 0 for normal tokens without post-processor
// ===========================================================================

/// special_tokens_mask for "Hello" is all 0s (no special tokens, no post-processor).
#[test]
fn special_tokens_mask_all_zeros_hello() {
    let ctx = TokenizerTestContext::new();
    let text = "Hello";
    let result = unsafe { super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 5);

    let mask = unsafe { std::slice::from_raw_parts(result.special_tokens_mask, result.num_tokens) };
    for (i, &m) in mask.iter().enumerate() {
        assert_eq!(
            m, 0,
            "special_tokens_mask[{i}] should be 0 (no post-processor)"
        );
    }

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// special_tokens_mask is a sliceable empty sentinel for empty input.
#[test]
fn special_tokens_mask_empty_input() {
    let ctx = TokenizerTestContext::new();
    let result = unsafe { super::common::encode_raw(ctx.handle(), &[], &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 0);
    assert!(!result.special_tokens_mask.is_null());
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// special_tokens_mask for byte-level fixture is all 0s.
#[test]
fn special_tokens_mask_byte_level_all_zeros() {
    let ctx = TokenizerTestContext::with_byte_level();
    let text = "café";
    let result = unsafe { super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, text.len()); // 5 bytes

    let mask = unsafe { std::slice::from_raw_parts(result.special_tokens_mask, result.num_tokens) };
    assert!(
        mask.iter().all(|&m| m == 0),
        "all special_tokens_mask values should be 0 (no post-processor)"
    );

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// special_tokens_mask for merges fixture is all 0s (merged tokens are not special).
#[test]
fn special_tokens_mask_merges_all_zeros() {
    let ctx = TokenizerTestContext::with_merges();
    let text = "hello";
    let result = unsafe { super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
    assert!(result.error_msg.is_null());
    // "hello" → [104] (fully merged)
    assert_eq!(result.num_tokens, 1);

    let mask = unsafe { std::slice::from_raw_parts(result.special_tokens_mask, result.num_tokens) };
    assert_eq!(mask, &[0], "merged token should not be special");

    unsafe { talu_sys::talu_encode_result_free(result) };
}

// ===========================================================================
// attention_mask and special_tokens_mask array lengths match num_tokens
// ===========================================================================

/// All encode result arrays have consistent lengths.
#[test]
fn encode_result_arrays_consistent_lengths() {
    let ctx = TokenizerTestContext::new();
    for text in ["Hello", "A", "abc123", "!@#"] {
        let result = unsafe { super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
        assert!(result.error_msg.is_null());
        let n = result.num_tokens;
        assert_eq!(n, text.len(), "num_tokens for {text:?}");

        // Verify all arrays are non-null and have n elements by reading them.
        assert!(!result.ids.is_null());
        assert!(!result.offsets.is_null());
        assert!(!result.attention_mask.is_null());
        assert!(!result.special_tokens_mask.is_null());

        // Read all arrays to verify they're accessible (no segfault).
        let ids = unsafe { std::slice::from_raw_parts(result.ids, n) };
        let offsets = unsafe { std::slice::from_raw_parts(result.offsets, n) };
        let attn = unsafe { std::slice::from_raw_parts(result.attention_mask, n) };
        let special = unsafe { std::slice::from_raw_parts(result.special_tokens_mask, n) };

        assert_eq!(ids.len(), n);
        assert_eq!(offsets.len(), n);
        assert_eq!(attn.len(), n);
        assert_eq!(special.len(), n);

        unsafe { talu_sys::talu_encode_result_free(result) };
    }
}

// ===========================================================================
// Offsets with merged BPE tokens
// ===========================================================================

/// "hello" with merges → 1 token spanning [0, 5).
#[test]
fn offsets_merged_hello_single_token() {
    let ctx = TokenizerTestContext::with_merges();
    let text = "hello";
    let result = unsafe { super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
    assert!(result.error_msg.is_null());
    // "hello" → [104] (fully merged)
    assert_eq!(result.num_tokens, 1);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!(offsets[0].start, 0, "merged token should start at 0");
    assert_eq!(offsets[0].end, 5, "merged token should end at 5");

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// "hell" with merges → 1 token spanning [0, 4).
#[test]
fn offsets_merged_hell() {
    let ctx = TokenizerTestContext::with_merges();
    let text = "hell";
    let result = unsafe { super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
    assert!(result.error_msg.is_null());
    // "hell" → [103]
    assert_eq!(result.num_tokens, 1);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!(offsets[0].start, 0);
    assert_eq!(offsets[0].end, 4);

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// "helloabc" with merges → "hello" merged [0,5) + a[5,6) + b[6,7) + c[7,8).
#[test]
fn offsets_merged_followed_by_unmerged() {
    let ctx = TokenizerTestContext::with_merges();
    let text = "helloabc";
    let result = unsafe { super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
    assert!(result.error_msg.is_null());
    // "hello" → [104], a → [69], b → [70], c → [71]
    assert_eq!(result.num_tokens, 4);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 5), "hello span");
    assert_eq!((offsets[1].start, offsets[1].end), (5, 6), "a span");
    assert_eq!((offsets[2].start, offsets[2].end), (6, 7), "b span");
    assert_eq!((offsets[3].start, offsets[3].end), (7, 8), "c span");

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// "helo" with merges → "hel"[0,3) + "o"[3,4).
#[test]
fn offsets_merged_partial() {
    let ctx = TokenizerTestContext::with_merges();
    let text = "helo";
    let result = unsafe { super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
    assert!(result.error_msg.is_null());
    // "helo" → [hel=102, o=83]
    assert_eq!(result.num_tokens, 2);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 3), "hel span");
    assert_eq!((offsets[1].start, offsets[1].end), (3, 4), "o span");

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Offsets with merges are contiguous and cover full input.
#[test]
fn offsets_merged_contiguous_and_complete() {
    let ctx = TokenizerTestContext::with_merges();
    for text in ["hello", "hell", "helo", "lo", "abc"] {
        let result = unsafe { super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
        assert!(result.error_msg.is_null());
        let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };

        if !offsets.is_empty() {
            assert_eq!(offsets[0].start, 0, "first offset.start for {text:?}");
            assert_eq!(
                offsets.last().unwrap().end as usize,
                text.len(),
                "last offset.end for {text:?}"
            );
            for w in offsets.windows(2) {
                assert_eq!(w[0].end, w[1].start, "contiguous offsets for {text:?}");
            }
        }
        unsafe { talu_sys::talu_encode_result_free(result) };
    }
}

// ===========================================================================
// Offsets with multi-byte UTF-8 (byte-level fixture)
// ===========================================================================

/// Offsets for "café" (5 bytes) on byte-level fixture: each byte → [i, i+1).
#[test]
fn offsets_byte_level_cafe() {
    let ctx = TokenizerTestContext::with_byte_level();
    let text = "café";
    let result = unsafe { super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 5); // c(1) + a(1) + f(1) + é(2) = 5 bytes

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    for (i, off) in offsets.iter().enumerate() {
        assert_eq!(off.start as usize, i, "byte-level offset[{i}].start");
        assert_eq!(off.end as usize, i + 1, "byte-level offset[{i}].end");
    }

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Offsets for "日" (3 UTF-8 bytes) on byte-level fixture span [0,1), [1,2), [2,3).
#[test]
fn offsets_byte_level_cjk() {
    let ctx = TokenizerTestContext::with_byte_level();
    let text = "日";
    assert_eq!(text.len(), 3);
    let result = unsafe { super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 3);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 1));
    assert_eq!((offsets[1].start, offsets[1].end), (1, 2));
    assert_eq!((offsets[2].start, offsets[2].end), (2, 3));

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Offsets for mixed ASCII+emoji span all bytes contiguously.
#[test]
fn offsets_byte_level_mixed_ascii_emoji() {
    let ctx = TokenizerTestContext::with_byte_level();
    let text = "A😊B";
    // A(1 byte) + 😊(4 bytes) + B(1 byte) = 6 bytes
    assert_eq!(text.len(), 6);
    let result = unsafe { super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 6);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    // Each byte → one token → one single-byte offset span
    for (i, off) in offsets.iter().enumerate() {
        assert_eq!(off.start as usize, i, "offset[{i}].start");
        assert_eq!(off.end as usize, i + 1, "offset[{i}].end");
    }

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Offsets for a regional-indicator flag emoji must still map one token per
/// UTF-8 byte on the byte-level fixture.
#[test]
fn offsets_byte_level_flag_emoji() {
    let ctx = TokenizerTestContext::with_byte_level();
    let text = "🇺🇸";
    assert_eq!(text.len(), 8);
    let result = unsafe { super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 8);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    for (i, off) in offsets.iter().enumerate() {
        assert_eq!(off.start as usize, i, "flag offset[{i}].start");
        assert_eq!(off.end as usize, i + 1, "flag offset[{i}].end");
    }

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Offsets for a variation-selector emoji sequence must remain single-byte
/// spans on the byte-level fixture.
#[test]
fn offsets_byte_level_variation_selector_emoji() {
    let ctx = TokenizerTestContext::with_byte_level();
    let text = "❤️";
    assert_eq!(text.len(), 6);
    let result = unsafe { super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 6);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    for (i, off) in offsets.iter().enumerate() {
        assert_eq!(
            off.start as usize, i,
            "variation-selector offset[{i}].start"
        );
        assert_eq!(
            off.end as usize,
            i + 1,
            "variation-selector offset[{i}].end"
        );
    }

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Offsets for a ZWJ family emoji sequence must remain one-byte spans on the
/// byte-level fixture even though the visible glyph is a single grapheme.
#[test]
fn offsets_byte_level_zwj_family_sequence() {
    let ctx = TokenizerTestContext::with_byte_level();
    let text = "👨\u{200D}👩\u{200D}👧\u{200D}👦";
    assert_eq!(text.len(), 25);
    let result = unsafe { super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 25);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    for (i, off) in offsets.iter().enumerate() {
        assert_eq!(off.start as usize, i, "zwj offset[{i}].start");
        assert_eq!(off.end as usize, i + 1, "zwj offset[{i}].end");
    }

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Offsets for decomposed combining-mark text must also remain one-byte spans
/// on the byte-level fixture.
#[test]
fn offsets_byte_level_decomposed_combining_marks() {
    let ctx = TokenizerTestContext::with_byte_level();
    let text = "a\u{0301}\u{0308}";
    assert_eq!(text.len(), 5);
    let result = unsafe { super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 5);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    for (i, off) in offsets.iter().enumerate() {
        assert_eq!(off.start as usize, i, "combining offset[{i}].start");
        assert_eq!(off.end as usize, i + 1, "combining offset[{i}].end");
    }

    unsafe { talu_sys::talu_encode_result_free(result) };
}

// ===========================================================================
// Truncation preserves all encode result fields
// ===========================================================================

/// Right truncation keeps first 2 tokens with correct offsets.
#[test]
fn truncation_right_preserves_offsets() {
    let ctx = TokenizerTestContext::new();
    let text = "Hello"; // 5 tokens: H=44, e=73, l=80, l=80, o=83
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        truncation: 1,
        truncation_side: 0, // right: keep first
        max_length: 2,
        ..Default::default()
    };
    let result = unsafe { super::common::encode_raw(ctx.handle(), text.as_bytes(), &opts) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 2);

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    assert_eq!(ids, &[44, 73]); // H, e

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 1), "H offset");
    assert_eq!((offsets[1].start, offsets[1].end), (1, 2), "e offset");

    let attn = unsafe { std::slice::from_raw_parts(result.attention_mask, result.num_tokens) };
    assert_eq!(attn, &[1, 1], "attention_mask after right truncation");

    let special =
        unsafe { std::slice::from_raw_parts(result.special_tokens_mask, result.num_tokens) };
    assert_eq!(
        special,
        &[0, 0],
        "special_tokens_mask after right truncation"
    );

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Left truncation keeps last 2 tokens with correct offsets.
#[test]
fn truncation_left_preserves_offsets() {
    let ctx = TokenizerTestContext::new();
    let text = "Hello"; // 5 tokens
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        truncation: 1,
        truncation_side: 1, // left: keep last
        max_length: 2,
        ..Default::default()
    };
    let result = unsafe { super::common::encode_raw(ctx.handle(), text.as_bytes(), &opts) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 2);

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    assert_eq!(ids, &[80, 83]); // l, o

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    // These are the original offsets for tokens 3 and 4 (0-indexed).
    assert_eq!((offsets[0].start, offsets[0].end), (3, 4), "l offset");
    assert_eq!((offsets[1].start, offsets[1].end), (4, 5), "o offset");

    let attn = unsafe { std::slice::from_raw_parts(result.attention_mask, result.num_tokens) };
    assert_eq!(attn, &[1, 1], "attention_mask after left truncation");

    let special =
        unsafe { std::slice::from_raw_parts(result.special_tokens_mask, result.num_tokens) };
    assert_eq!(
        special,
        &[0, 0],
        "special_tokens_mask after left truncation"
    );

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Right truncation after ByteLevel `add_prefix_space` must keep the
/// zero-width synthetic prefix span followed by the first retained source spans.
#[test]
fn truncation_right_with_add_prefix_space_preserves_prefix_offset_contract() {
    let json = build_byte_level_tokenizer_json()
        .replace("\"add_prefix_space\": false", "\"add_prefix_space\": true");
    let ctx = TokenizerTestContext::from_json(&json);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        truncation: 1,
        truncation_side: 0,
        max_length: 3,
        ..Default::default()
    };
    let result = unsafe { super::common::encode_raw(ctx.handle(), b"Hello", &opts) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 3);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 0));
    assert_eq!((offsets[1].start, offsets[1].end), (0, 1));
    assert_eq!((offsets[2].start, offsets[2].end), (1, 2));

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Left truncation after ByteLevel `add_prefix_space` must drop the synthetic
/// prefix and preserve the original tail byte spans.
#[test]
fn truncation_left_with_add_prefix_space_preserves_tail_offsets() {
    let json = build_byte_level_tokenizer_json()
        .replace("\"add_prefix_space\": false", "\"add_prefix_space\": true");
    let ctx = TokenizerTestContext::from_json(&json);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        truncation: 1,
        truncation_side: 1,
        max_length: 3,
        ..Default::default()
    };
    let result = unsafe { super::common::encode_raw(ctx.handle(), b"Hello", &opts) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 3);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (2, 3));
    assert_eq!((offsets[1].start, offsets[1].end), (3, 4));
    assert_eq!((offsets[2].start, offsets[2].end), (4, 5));

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Truncation to 1 token preserves all fields.
#[test]
fn truncation_to_one_preserves_all_fields() {
    let ctx = TokenizerTestContext::new();
    let text = "abc";
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        truncation: 1,
        truncation_side: 0,
        max_length: 1,
        ..Default::default()
    };
    let result = unsafe { super::common::encode_raw(ctx.handle(), text.as_bytes(), &opts) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 1);

    let ids = unsafe { std::slice::from_raw_parts(result.ids, 1) };
    assert_eq!(ids, &[69]); // a

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, 1) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 1));

    let attn = unsafe { std::slice::from_raw_parts(result.attention_mask, 1) };
    assert_eq!(attn, &[1]);

    let special = unsafe { std::slice::from_raw_parts(result.special_tokens_mask, 1) };
    assert_eq!(special, &[0]);

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Truncation of merged tokens preserves offset spans.
#[test]
fn truncation_merged_preserves_offsets() {
    let ctx = TokenizerTestContext::with_merges();
    let text = "helloabc"; // → [hello=104, a=69, b=70, c=71] (4 tokens)
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        truncation: 1,
        truncation_side: 0,
        max_length: 2,
        ..Default::default()
    };
    let result = unsafe { super::common::encode_raw(ctx.handle(), text.as_bytes(), &opts) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 2);

    let ids = unsafe { std::slice::from_raw_parts(result.ids, 2) };
    assert_eq!(ids, &[104, 69]); // hello, a

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, 2) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 5), "hello span");
    assert_eq!((offsets[1].start, offsets[1].end), (5, 6), "a span");

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Truncation with max_length >= token count is a no-op for all fields.
#[test]
fn truncation_noop_all_fields_intact() {
    let ctx = TokenizerTestContext::new();
    let text = "Hi"; // 2 tokens
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        truncation: 1,
        max_length: 100,
        ..Default::default()
    };
    let result = unsafe { super::common::encode_raw(ctx.handle(), text.as_bytes(), &opts) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 2);

    let ids = unsafe { std::slice::from_raw_parts(result.ids, 2) };
    assert_eq!(ids, &[44, 77]);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, 2) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 1));
    assert_eq!((offsets[1].start, offsets[1].end), (1, 2));

    let attn = unsafe { std::slice::from_raw_parts(result.attention_mask, 2) };
    assert_eq!(attn, &[1, 1]);

    let special = unsafe { std::slice::from_raw_parts(result.special_tokens_mask, 2) };
    assert_eq!(special, &[0, 0]);

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Offsets must map normalized lowercase+accent-stripped output back to original bytes.
#[test]
fn offsets_with_lowercase_and_strip_accents_map_to_original_bytes() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "c": 1, "a": 2, "f": 3, "e": 4
    },
    "merges": []
  },
  "added_tokens": [{"id": 0, "content": "<unk>", "special": true}],
  "normalizer": {"type": "Sequence", "normalizers": [{"type": "Lowercase"}, {"type": "StripAccents"}]},
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    let result = unsafe { super::common::encode_raw(ctx.handle(), "CAFÉ".as_bytes(), &opts) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 4, "CAFÉ should normalize to c a f e");

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    assert_eq!(
        ids,
        &[1, 2, 3, 4],
        "expected normalized token IDs for c a f e"
    );

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 1));
    assert_eq!((offsets[1].start, offsets[1].end), (1, 2));
    assert_eq!((offsets[2].start, offsets[2].end), (2, 3));
    assert_eq!(
        (offsets[3].start, offsets[3].end),
        (3, 5),
        "accent-stripped e must map to original É byte span"
    );

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Lowercase-only normalization must map tokens back to original uppercase source bytes.
#[test]
fn offsets_with_lowercase_map_to_original_bytes() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"<unk>": 0, "a": 1, "b": 2, "c": 3},
    "merges": []
  },
  "added_tokens": [{"id": 0, "content": "<unk>", "special": true}],
  "normalizer": {"type": "Lowercase"},
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    let result = unsafe { super::common::encode_raw(ctx.handle(), b"ABC", &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 3);

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    assert_eq!(ids, &[1, 2, 3]);
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 1));
    assert_eq!((offsets[1].start, offsets[1].end), (1, 2));
    assert_eq!((offsets[2].start, offsets[2].end), (2, 3));
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// StripAccents-only normalization must map stripped chars to original accented source bytes.
#[test]
fn offsets_with_strip_accents_map_to_original_bytes() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"<unk>": 0, "e": 1},
    "merges": []
  },
  "added_tokens": [{"id": 0, "content": "<unk>", "special": true}],
  "normalizer": {"type": "StripAccents"},
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    let result = unsafe { super::common::encode_raw(ctx.handle(), "é".as_bytes(), &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 1);
    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    assert_eq!(ids, &[1]);
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!(
        (offsets[0].start, offsets[0].end),
        (0, 2),
        "stripped e must map to original composed-é byte span"
    );
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// NFC normalization composing e + combining-acute into é must preserve source span.
#[test]
fn offsets_with_nfc_composition_map_to_decomposed_source_span() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"<unk>": 0, "\u00E9": 1},
    "merges": []
  },
  "added_tokens": [{"id": 0, "content": "<unk>", "special": true}],
  "normalizer": {"type": "NFC"},
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    let input = "e\u{0301}";
    assert_eq!(input.len(), 3, "decomposed e+acute must be 3 UTF-8 bytes");

    let result = unsafe { super::common::encode_raw(ctx.handle(), input.as_bytes(), &opts) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 1, "NFC should compose to one token");
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!(
        (offsets[0].start, offsets[0].end),
        (0, 3),
        "composed token must map to full decomposed source span"
    );
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// NFKC normalization (fullwidth -> ASCII) must map back to original source byte spans.
#[test]
fn offsets_with_nfkc_fullwidth_map_to_original_bytes() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"<unk>": 0, "A": 1, "B": 2},
    "merges": []
  },
  "added_tokens": [{"id": 0, "content": "<unk>", "special": true}],
  "normalizer": {"type": "NFKC"},
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    let input = "\u{FF21}\u{FF22}"; // fullwidth A, B (3 bytes each)
    assert_eq!(input.len(), 6);

    let result = unsafe { super::common::encode_raw(ctx.handle(), input.as_bytes(), &opts) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 2, "NFKC should normalize to ASCII A,B");
    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    assert_eq!(ids, &[1, 2]);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 3));
    assert_eq!((offsets[1].start, offsets[1].end), (3, 6));
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// WordPiece offsets must survive BertNormalizer mutations that lowercase and
/// strip accents, mapping the normalized token back to the original bytes.
#[test]
fn offsets_wordpiece_lowercase_strip_accents_map_to_original_bytes() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 100,
    "vocab": {
      "[UNK]": 0, "cafe": 1
    }
  },
  "added_tokens": [{"id": 0, "content": "[UNK]", "special": true}],
  "normalizer": {
    "type": "BertNormalizer",
    "clean_text": true,
    "handle_chinese_chars": false,
    "strip_accents": true,
    "lowercase": true
  },
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let input = "CAFÉ";
    assert_eq!(input.len(), 5, "É must occupy two UTF-8 bytes here");

    let result = unsafe { super::common::encode_raw(ctx.handle(), input.as_bytes(), &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(
        result.num_tokens, 1,
        "normalized whole word should stay one token"
    );

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    assert_eq!(ids, &[1]);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 5));

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// BertNormalizer `clean_text=true` rewrites CR/LF controls to spaces. Offsets
/// for surviving WordPiece tokens must still map to the original source bytes.
#[test]
fn offsets_wordpiece_clean_text_crlf_maps_to_original_bytes() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 100,
    "vocab": {
      "[UNK]": 0, "a": 1, "b": 2
    }
  },
  "added_tokens": [{"id": 0, "content": "[UNK]", "special": true}],
  "normalizer": {
    "type": "BertNormalizer",
    "clean_text": true,
    "handle_chinese_chars": false,
    "strip_accents": false,
    "lowercase": true
  },
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let input = "A\r\nB";
    assert_eq!(input.len(), 4);

    let result = unsafe { super::common::encode_raw(ctx.handle(), input.as_bytes(), &no_bos()) };
    assert!(result.error_msg.is_null(), "encode failed");
    assert_eq!(result.num_tokens, 2, "CR/LF cleanup should still yield A and B");

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    assert_eq!(ids, &[1, 2]);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 1));
    assert_eq!(
        (offsets[1].start, offsets[1].end),
        (3, 4),
        "token 'b' must map to the original 'B' byte after CR/LF cleanup"
    );
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// If pretokenization removes all user content, template post-processing must
/// still return a valid specials-only sequence with synthetic zero offsets.
#[test]
fn offsets_whitespace_removed_then_template_specials_stay_synthetic() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"H": 4, "i": 5},
    "merges": []
  },
  "added_tokens": [
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {
    "type": "Split",
    "pattern": {"Regex": "\\s+"},
    "behavior": "Removed",
    "invert": false
  },
  "post_processor": {
    "type": "TemplateProcessing",
    "single": [
      {"SpecialToken": {"id": "<s>", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}},
      {"SpecialToken": {"id": "</s>", "type_id": 0}}
    ],
    "pair": [
      {"SpecialToken": {"id": "<s>", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}},
      {"SpecialToken": {"id": "</s>", "type_id": 0}},
      {"Sequence": {"id": "B", "type_id": 1}},
      {"SpecialToken": {"id": "</s>", "type_id": 0}}
    ],
    "special_tokens": {
      "<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]},
      "</s>": {"id": "</s>", "ids": [2], "tokens": ["</s>"]}
    }
  },
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let opts = talu_sys::EncodeOptions {
        add_bos: 1,
        add_eos: 1,
        ..Default::default()
    };
    let input = "\r\n \t";
    let result = unsafe { super::common::encode_raw(ctx.handle(), input.as_bytes(), &opts) };
    assert!(result.error_msg.is_null(), "encode failed");
    assert_eq!(result.num_tokens, 2, "only BOS/EOS should remain");

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    assert_eq!(ids, &[1, 2]);
    let special =
        unsafe { std::slice::from_raw_parts(result.special_tokens_mask, result.num_tokens) };
    assert_eq!(special, &[1, 1], "both output tokens must be marked special");
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 0));
    assert_eq!((offsets[1].start, offsets[1].end), (0, 0));
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Template post-processing must set masks and offsets consistently.
#[test]
fn postprocessor_masks_and_offsets_are_exact() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"H": 4, "i": 5},
    "merges": []
  },
  "added_tokens": [
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": {
    "type": "TemplateProcessing",
    "single": [
      {"SpecialToken": {"id": "<s>", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}},
      {"SpecialToken": {"id": "</s>", "type_id": 0}}
    ],
    "pair": [
      {"SpecialToken": {"id": "<s>", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}},
      {"SpecialToken": {"id": "</s>", "type_id": 0}},
      {"Sequence": {"id": "B", "type_id": 1}},
      {"SpecialToken": {"id": "</s>", "type_id": 0}}
    ],
    "special_tokens": {
      "<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]},
      "</s>": {"id": "</s>", "ids": [2], "tokens": ["</s>"]}
    }
  },
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    // Rust zeroes FFI structs for Default, so add_eos must be explicit when
    // the test expects the template post-processor to append SEP/EOS.
    let opts = talu_sys::EncodeOptions {
        add_bos: 1,
        add_eos: 1,
        ..Default::default()
    };
    let result = unsafe { super::common::encode_raw(ctx.handle(), b"Hi", &opts) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 4);

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    assert_eq!(ids, &[1, 4, 5, 2]);
    let attn = unsafe { std::slice::from_raw_parts(result.attention_mask, result.num_tokens) };
    assert_eq!(attn, &[1, 1, 1, 1]);
    let special =
        unsafe { std::slice::from_raw_parts(result.special_tokens_mask, result.num_tokens) };
    assert_eq!(special, &[1, 0, 0, 1]);
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 0));
    assert_eq!((offsets[1].start, offsets[1].end), (0, 1));
    assert_eq!((offsets[2].start, offsets[2].end), (1, 2));
    assert_eq!((offsets[3].start, offsets[3].end), (0, 0));

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Empty input with template post-processing should still produce exact masks and offsets.
#[test]
fn postprocessor_masks_and_offsets_empty_input() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"H": 4, "i": 5},
    "merges": []
  },
  "added_tokens": [
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": {
    "type": "TemplateProcessing",
    "single": [
      {"SpecialToken": {"id": "<s>", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}},
      {"SpecialToken": {"id": "</s>", "type_id": 0}}
    ],
    "pair": [
      {"SpecialToken": {"id": "<s>", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}},
      {"SpecialToken": {"id": "</s>", "type_id": 0}},
      {"Sequence": {"id": "B", "type_id": 1}},
      {"SpecialToken": {"id": "</s>", "type_id": 0}}
    ],
    "special_tokens": {
      "<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]},
      "</s>": {"id": "</s>", "ids": [2], "tokens": ["</s>"]}
    }
  },
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let opts = talu_sys::EncodeOptions {
        add_bos: 1,
        add_eos: 1,
        ..Default::default()
    };
    let result = unsafe { super::common::encode_raw(ctx.handle(), b"", &opts) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 2);
    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    assert_eq!(ids, &[1, 2]);
    let special =
        unsafe { std::slice::from_raw_parts(result.special_tokens_mask, result.num_tokens) };
    assert_eq!(special, &[1, 1]);
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 0));
    assert_eq!((offsets[1].start, offsets[1].end), (0, 0));
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Replace normalizer expansion (& -> and) should map all expanded tokens to source span.
#[test]
fn offsets_replace_expansion_maps_to_single_source_span() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"<unk>": 0, "a": 1, "n": 2, "d": 3},
    "merges": []
  },
  "added_tokens": [{"id": 0, "content": "<unk>", "special": true}],
  "normalizer": {"type": "Replace", "pattern": {"String": "&"}, "content": "and"},
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    let result = unsafe { super::common::encode_raw(ctx.handle(), b"&", &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 3);
    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    assert_eq!(ids, &[1, 2, 3]);
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    for off in offsets {
        assert_eq!(
            (off.start, off.end),
            (0, 1),
            "all expanded chars should map to original '&' source byte span"
        );
    }
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Replace expansion in middle of text must map expanded chars to the original source byte.
#[test]
fn offsets_replace_expansion_in_middle_preserves_neighbor_spans() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"<unk>": 0, "A": 1, "a": 2, "n": 3, "d": 4, "B": 5},
    "merges": []
  },
  "added_tokens": [{"id": 0, "content": "<unk>", "special": true}],
  "normalizer": {"type": "Replace", "pattern": {"String": "&"}, "content": "and"},
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    let result = unsafe { super::common::encode_raw(ctx.handle(), b"A&B", &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 5);
    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    assert_eq!(ids, &[1, 2, 3, 4, 5]);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 1), "A span");
    assert_eq!(
        (offsets[1].start, offsets[1].end),
        (1, 2),
        "expanded a span"
    );
    assert_eq!(
        (offsets[2].start, offsets[2].end),
        (1, 2),
        "expanded n span"
    );
    assert_eq!(
        (offsets[3].start, offsets[3].end),
        (1, 2),
        "expanded d span"
    );
    assert_eq!((offsets[4].start, offsets[4].end), (2, 3), "B span");
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Large NFKC expansions must keep every emitted token mapped to the original
/// source scalar span, not drift or walk past the source buffer.
#[test]
fn offsets_with_nfkc_large_expansion_map_to_single_source_scalar() {
    let json = build_byte_level_tokenizer_json().replace(
        "\"normalizer\": null,",
        "\"normalizer\": {\"type\": \"NFKC\"},",
    );
    let ctx = TokenizerTestContext::from_json(&json);
    let input = "\u{FDFA}";
    assert_eq!(input.len(), 3, "U+FDFA must be a 3-byte UTF-8 scalar");

    let result = unsafe { super::common::encode_raw(ctx.handle(), input.as_bytes(), &no_bos()) };
    assert!(result.error_msg.is_null());
    assert!(
        result.num_tokens > 10,
        "U+FDFA should expand substantially under NFKC, got only {} tokens",
        result.num_tokens
    );

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    for (idx, off) in offsets.iter().enumerate() {
        assert_eq!(
            (off.start, off.end),
            (0, 3),
            "expanded token {idx} must map back to the original U+FDFA byte span"
        );
    }

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Repeating a large-expansion scalar many times must not overflow offset
/// reconstruction. Every expanded token still belongs to exactly one original
/// 3-byte scalar span.
#[test]
fn offsets_with_repeated_nfkc_large_expansion_stay_within_source_bounds() {
    let json = build_byte_level_tokenizer_json().replace(
        "\"normalizer\": null,",
        "\"normalizer\": {\"type\": \"NFKC\"},",
    );
    let ctx = TokenizerTestContext::from_json(&json);
    let input = "\u{FDFA}".repeat(64);
    assert_eq!(
        input.len(),
        64 * 3,
        "repeated U+FDFA must stay 3 bytes per scalar"
    );

    let result = unsafe { super::common::encode_raw(ctx.handle(), input.as_bytes(), &no_bos()) };
    assert!(result.error_msg.is_null());
    assert!(
        result.num_tokens > 64 * 10,
        "repeated U+FDFA should expand substantially under NFKC, got only {} tokens",
        result.num_tokens
    );

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    for (idx, off) in offsets.iter().enumerate() {
        let start = off.start as usize;
        let end = off.end as usize;
        assert!(
            start < end,
            "expanded token {idx} must have a non-empty source span"
        );
        assert!(
            end <= input.len(),
            "expanded token {idx} must stay within the source bounds"
        );
        assert_eq!(
            end - start,
            3,
            "expanded token {idx} must map to exactly one source scalar"
        );
        assert_eq!(
            start % 3,
            0,
            "expanded token {idx} start must align to the repeated-scalar boundary"
        );
    }

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Replace normalizer shrinking text (remove apostrophe) must preserve original byte mapping.
#[test]
fn offsets_replace_shrink_preserves_original_positions() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"<unk>": 0, "c": 1, "a": 2, "n": 3, "t": 4},
    "merges": []
  },
  "added_tokens": [{"id": 0, "content": "<unk>", "special": true}],
  "normalizer": {"type": "Replace", "pattern": {"String": "'"}, "content": ""},
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    let result = unsafe { super::common::encode_raw(ctx.handle(), b"can't", &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 4);
    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    assert_eq!(ids, &[1, 2, 3, 4]);
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 1));
    assert_eq!((offsets[1].start, offsets[1].end), (1, 2));
    assert_eq!((offsets[2].start, offsets[2].end), (2, 3));
    assert_eq!(
        (offsets[3].start, offsets[3].end),
        (4, 5),
        "token 't' should map after removed apostrophe"
    );
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Byte fallback emits multiple output tokens for one unknown UTF-8 symbol; all
/// emitted tokens must map back to that symbol's full source span.
#[test]
fn offsets_byte_fallback_multibyte_unknown_map_to_original_span() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "unk_token": "<unk>",
    "vocab": {"<unk>": 0, "<0xC3>": 1, "<0xA9>": 2},
    "merges": []
  },
  "added_tokens": [{"id": 0, "content": "<unk>", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    let result = unsafe { super::common::encode_raw(ctx.handle(), "é".as_bytes(), &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 2);

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!(ids, &[1, 2]);
    assert_eq!(
        (offsets[0].start, offsets[0].end),
        (0, 2),
        "first fallback byte must map to the full composed-é span"
    );
    assert_eq!(
        (offsets[1].start, offsets[1].end),
        (0, 2),
        "second fallback byte must map to the full composed-é span"
    );
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Without byte fallback, per-byte `<unk>` emission for an unknown UTF-8 symbol
/// must still map every emitted token to the original source span.
#[test]
fn offsets_per_byte_unk_multibyte_unknown_map_to_original_span() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "unk_token": "<unk>",
    "vocab": {"<unk>": 0},
    "merges": []
  },
  "added_tokens": [{"id": 0, "content": "<unk>", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    let result = unsafe { super::common::encode_raw(ctx.handle(), "é".as_bytes(), &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 2);

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!(ids, &[0, 0]);
    assert_eq!(
        (offsets[0].start, offsets[0].end),
        (0, 2),
        "first per-byte <unk> must map to the full composed-é span"
    );
    assert_eq!(
        (offsets[1].start, offsets[1].end),
        (0, 2),
        "second per-byte <unk> must map to the full composed-é span"
    );
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Repeated merged subsequences should never produce zero offsets.
#[test]
fn offsets_repeated_merged_subsequences_no_zero_spans() {
    let ctx = TokenizerTestContext::with_merges();
    let result = unsafe { super::common::encode_raw(ctx.handle(), b"hellohello", &no_bos()) };
    assert!(result.error_msg.is_null());
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!(
        result.num_tokens, 2,
        "hellohello should merge into two hello tokens"
    );
    assert_eq!((offsets[0].start, offsets[0].end), (0, 5));
    assert_eq!((offsets[1].start, offsets[1].end), (5, 10));
}

/// Right truncation after postprocessing must preserve mask/offset semantics.
#[test]
fn postprocessor_truncation_right_preserves_masks_and_offsets() {
    let json = r####"{
  "version": "1.0",
  "model": { "type": "BPE", "vocab": {"H": 4, "i": 5}, "merges": [] },
  "added_tokens": [
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": {
    "type": "TemplateProcessing",
    "single": [
      {"SpecialToken": {"id": "<s>", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}},
      {"SpecialToken": {"id": "</s>", "type_id": 0}}
    ],
    "pair": [
      {"SpecialToken": {"id": "<s>", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}},
      {"SpecialToken": {"id": "</s>", "type_id": 0}},
      {"Sequence": {"id": "B", "type_id": 1}},
      {"SpecialToken": {"id": "</s>", "type_id": 0}}
    ],
    "special_tokens": {
      "<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]},
      "</s>": {"id": "</s>", "ids": [2], "tokens": ["</s>"]}
    }
  },
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let opts = talu_sys::EncodeOptions {
        add_bos: 1,
        add_eos: 1,
        truncation: 1,
        truncation_side: 0,
        max_length: 3,
        ..Default::default()
    };
    let result = unsafe { super::common::encode_raw(ctx.handle(), b"Hi", &opts) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 3);
    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    let special =
        unsafe { std::slice::from_raw_parts(result.special_tokens_mask, result.num_tokens) };
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!(ids, &[1, 4, 5]);
    assert_eq!(special, &[1, 0, 0]);
    assert_eq!((offsets[0].start, offsets[0].end), (0, 0));
    assert_eq!((offsets[1].start, offsets[1].end), (0, 1));
    assert_eq!((offsets[2].start, offsets[2].end), (1, 2));
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Left truncation after postprocessing must preserve mask/offset semantics.
#[test]
fn postprocessor_truncation_left_preserves_masks_and_offsets() {
    let json = r####"{
  "version": "1.0",
  "model": { "type": "BPE", "vocab": {"H": 4, "i": 5}, "merges": [] },
  "added_tokens": [
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": {
    "type": "TemplateProcessing",
    "single": [
      {"SpecialToken": {"id": "<s>", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}},
      {"SpecialToken": {"id": "</s>", "type_id": 0}}
    ],
    "pair": [
      {"SpecialToken": {"id": "<s>", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}},
      {"SpecialToken": {"id": "</s>", "type_id": 0}},
      {"Sequence": {"id": "B", "type_id": 1}},
      {"SpecialToken": {"id": "</s>", "type_id": 0}}
    ],
    "special_tokens": {
      "<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]},
      "</s>": {"id": "</s>", "ids": [2], "tokens": ["</s>"]}
    }
  },
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let opts = talu_sys::EncodeOptions {
        add_bos: 1,
        add_eos: 1,
        truncation: 1,
        truncation_side: 1,
        max_length: 3,
        ..Default::default()
    };
    let result = unsafe { super::common::encode_raw(ctx.handle(), b"Hi", &opts) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 3);
    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    let special =
        unsafe { std::slice::from_raw_parts(result.special_tokens_mask, result.num_tokens) };
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!(ids, &[4, 5, 2]);
    assert_eq!(special, &[0, 0, 1]);
    assert_eq!((offsets[0].start, offsets[0].end), (0, 1));
    assert_eq!((offsets[1].start, offsets[1].end), (1, 2));
    assert_eq!((offsets[2].start, offsets[2].end), (0, 0));
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// ByteLevel `add_prefix_space` inserts a synthetic leading token that must not
/// claim any real source span.
#[test]
fn byte_level_add_prefix_space_synthetic_prefix_has_zero_width_offset() {
    let json = build_byte_level_tokenizer_json()
        .replace("\"add_prefix_space\": false", "\"add_prefix_space\": true");
    let ctx = TokenizerTestContext::from_json(&json);

    let result = unsafe { super::common::encode_raw(ctx.handle(), b"Hello", &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 6);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!(
        (offsets[0].start, offsets[0].end),
        (0, 0),
        "synthetic prefix token must have zero-width source span"
    );
    assert_eq!((offsets[1].start, offsets[1].end), (0, 1));
    assert_eq!((offsets[2].start, offsets[2].end), (1, 2));
    assert_eq!((offsets[3].start, offsets[3].end), (2, 3));
    assert_eq!((offsets[4].start, offsets[4].end), (3, 4));
    assert_eq!((offsets[5].start, offsets[5].end), (4, 5));

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// When the source text already begins with a real space, ByteLevel
/// `add_prefix_space` must preserve that byte as a normal non-zero-width span.
#[test]
fn byte_level_add_prefix_space_real_leading_space_keeps_real_offset() {
    let json = build_byte_level_tokenizer_json()
        .replace("\"add_prefix_space\": false", "\"add_prefix_space\": true");
    let ctx = TokenizerTestContext::from_json(&json);

    let result = unsafe { super::common::encode_raw(ctx.handle(), b" Hello", &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 6);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!(
        (offsets[0].start, offsets[0].end),
        (0, 1),
        "real leading space must keep a real source span"
    );
    assert_eq!((offsets[1].start, offsets[1].end), (1, 2));
    assert_eq!((offsets[2].start, offsets[2].end), (2, 3));
    assert_eq!((offsets[3].start, offsets[3].end), (3, 4));
    assert_eq!((offsets[4].start, offsets[4].end), (4, 5));
    assert_eq!((offsets[5].start, offsets[5].end), (5, 6));

    unsafe { talu_sys::talu_encode_result_free(result) };
}
