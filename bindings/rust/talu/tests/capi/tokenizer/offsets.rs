//! Offset, attention_mask, and special_tokens_mask tests.
//!
//! Validates encode offsets, tokenize, tokenize_bytes, and the new EncodeResult
//! fields (attention_mask, special_tokens_mask) with exact assertions.

use crate::capi::tokenizer::common::TokenizerTestContext;

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
    let result = unsafe {
        super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos())
    };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 5);

    let offsets =
        unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };

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
    let result = unsafe {
        super::common::encode_raw(ctx.handle(), &[], &no_bos())
    };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 0);
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Encoding produces non-overlapping, contiguous offset spans.
#[test]
fn encode_offsets_contiguous() {
    let ctx = TokenizerTestContext::new();
    let text = "abc";
    let result = unsafe {
        super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos())
    };
    assert!(result.error_msg.is_null());
    let offsets =
        unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };

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

/// Encoding single character "A" returns one offset [0, 1).
#[test]
fn encode_offsets_single_char() {
    let ctx = TokenizerTestContext::new();
    let text = "A";
    let result = unsafe {
        super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos())
    };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 1);

    let offsets =
        unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!(offsets[0].start, 0);
    assert_eq!(offsets[0].end, 1);

    unsafe { talu_sys::talu_encode_result_free(result) };
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

// ===========================================================================
// attention_mask: always 1 for all tokens in single encode
// ===========================================================================

/// attention_mask for "Hello" is all 1s (5 tokens, all real).
#[test]
fn attention_mask_all_ones_hello() {
    let ctx = TokenizerTestContext::new();
    let text = "Hello";
    let result = unsafe {
        super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos())
    };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 5);

    let mask = unsafe { std::slice::from_raw_parts(result.attention_mask, result.num_tokens) };
    for (i, &m) in mask.iter().enumerate() {
        assert_eq!(m, 1, "attention_mask[{i}] should be 1");
    }

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// attention_mask is empty (null pointer) for empty input.
#[test]
fn attention_mask_empty_input() {
    let ctx = TokenizerTestContext::new();
    let result = unsafe {
        super::common::encode_raw(ctx.handle(), &[], &no_bos())
    };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 0);
    assert!(result.attention_mask.is_null());
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// attention_mask for byte-level fixture with emoji is all 1s.
#[test]
fn attention_mask_byte_level_emoji() {
    let ctx = TokenizerTestContext::with_byte_level();
    let text = "Hi😊";
    let result = unsafe {
        super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos())
    };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, text.len()); // 2 + 4 = 6 bytes

    let mask = unsafe { std::slice::from_raw_parts(result.attention_mask, result.num_tokens) };
    assert!(mask.iter().all(|&m| m == 1), "all attention_mask values should be 1");

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// attention_mask for single token is [1].
#[test]
fn attention_mask_single_token() {
    let ctx = TokenizerTestContext::new();
    let result = unsafe {
        super::common::encode_raw(ctx.handle(), b"A", &no_bos())
    };
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
    let result = unsafe {
        super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos())
    };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 5);

    let mask = unsafe {
        std::slice::from_raw_parts(result.special_tokens_mask, result.num_tokens)
    };
    for (i, &m) in mask.iter().enumerate() {
        assert_eq!(m, 0, "special_tokens_mask[{i}] should be 0 (no post-processor)");
    }

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// special_tokens_mask is null for empty input.
#[test]
fn special_tokens_mask_empty_input() {
    let ctx = TokenizerTestContext::new();
    let result = unsafe {
        super::common::encode_raw(ctx.handle(), &[], &no_bos())
    };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 0);
    assert!(result.special_tokens_mask.is_null());
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// special_tokens_mask for byte-level fixture is all 0s.
#[test]
fn special_tokens_mask_byte_level_all_zeros() {
    let ctx = TokenizerTestContext::with_byte_level();
    let text = "café";
    let result = unsafe {
        super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos())
    };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, text.len()); // 5 bytes

    let mask = unsafe {
        std::slice::from_raw_parts(result.special_tokens_mask, result.num_tokens)
    };
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
    let result = unsafe {
        super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos())
    };
    assert!(result.error_msg.is_null());
    // "hello" → [104] (fully merged)
    assert_eq!(result.num_tokens, 1);

    let mask = unsafe {
        std::slice::from_raw_parts(result.special_tokens_mask, result.num_tokens)
    };
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
        let result = unsafe {
            super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos())
        };
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
    let result = unsafe {
        super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos())
    };
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
    let result = unsafe {
        super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos())
    };
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
    let result = unsafe {
        super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos())
    };
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
    let result = unsafe {
        super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos())
    };
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
        let result = unsafe {
            super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos())
        };
        assert!(result.error_msg.is_null());
        let offsets =
            unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };

        if !offsets.is_empty() {
            assert_eq!(offsets[0].start, 0, "first offset.start for {text:?}");
            assert_eq!(
                offsets.last().unwrap().end as usize,
                text.len(),
                "last offset.end for {text:?}"
            );
            for w in offsets.windows(2) {
                assert_eq!(
                    w[0].end, w[1].start,
                    "contiguous offsets for {text:?}"
                );
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
    let result = unsafe {
        super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos())
    };
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
    let result = unsafe {
        super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos())
    };
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
    let result = unsafe {
        super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos())
    };
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
    let result = unsafe {
        super::common::encode_raw(ctx.handle(), text.as_bytes(), &opts)
    };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 2);

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    assert_eq!(ids, &[44, 73]); // H, e

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 1), "H offset");
    assert_eq!((offsets[1].start, offsets[1].end), (1, 2), "e offset");

    let attn = unsafe { std::slice::from_raw_parts(result.attention_mask, result.num_tokens) };
    assert_eq!(attn, &[1, 1], "attention_mask after right truncation");

    let special = unsafe {
        std::slice::from_raw_parts(result.special_tokens_mask, result.num_tokens)
    };
    assert_eq!(special, &[0, 0], "special_tokens_mask after right truncation");

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
    let result = unsafe {
        super::common::encode_raw(ctx.handle(), text.as_bytes(), &opts)
    };
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

    let special = unsafe {
        std::slice::from_raw_parts(result.special_tokens_mask, result.num_tokens)
    };
    assert_eq!(special, &[0, 0], "special_tokens_mask after left truncation");

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
    let result = unsafe {
        super::common::encode_raw(ctx.handle(), text.as_bytes(), &opts)
    };
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
    let result = unsafe {
        super::common::encode_raw(ctx.handle(), text.as_bytes(), &opts)
    };
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
    let result = unsafe {
        super::common::encode_raw(ctx.handle(), text.as_bytes(), &opts)
    };
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
