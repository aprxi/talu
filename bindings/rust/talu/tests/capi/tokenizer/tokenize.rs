//! BPE merge and tokenization tests.
//!
//! Uses the merges fixture (105 tokens, 6 merge rules) to verify that the
//! tokenizer performs actual subword merging — not just character splitting.
//!
//! Merge rules (rank order):
//!   0: h+e→he(99), 1: he+l→hel(102), 2: hel+l→hell(103),
//!   3: hell+o→hello(104), 4: l+l→ll(100), 5: l+o→lo(101)

use crate::capi::tokenizer::common::TokenizerTestContext;

/// Default encode options (no BOS).
fn no_bos() -> talu_sys::EncodeOptions {
    talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    }
}

// ===========================================================================
// BPE merge behavior
// ===========================================================================

/// "hello" fully merges into a single token (all 4 merge rules applied).
#[test]
fn merge_hello_single_token() {
    let ctx = TokenizerTestContext::with_merges();
    assert_eq!(ctx.encode_with("hello", &no_bos()), [104]);
}

/// "hell" merges via h+e→he, he+l→hel, hel+l→hell.
#[test]
fn merge_hell_single_token() {
    let ctx = TokenizerTestContext::with_merges();
    assert_eq!(ctx.encode_with("hell", &no_bos()), [103]);
}

/// "hel" merges via h+e→he, he+l→hel.
#[test]
fn merge_hel_single_token() {
    let ctx = TokenizerTestContext::with_merges();
    assert_eq!(ctx.encode_with("hel", &no_bos()), [102]);
}

/// "he" merges via h+e→he.
#[test]
fn merge_he_single_token() {
    let ctx = TokenizerTestContext::with_merges();
    assert_eq!(ctx.encode_with("he", &no_bos()), [99]);
}

/// "helo" → [hel, o]: "hel" merges but "hel o" is not a merge rule.
#[test]
fn merge_helo_partial() {
    let ctx = TokenizerTestContext::with_merges();
    assert_eq!(ctx.encode_with("helo", &no_bos()), [102, 83]);
}

/// "llo" → [ll, o]: l+l merges (rank 4) but "ll o" is not a merge rule.
#[test]
fn merge_llo_partial() {
    let ctx = TokenizerTestContext::with_merges();
    assert_eq!(ctx.encode_with("llo", &no_bos()), [100, 83]);
}

/// "lo" → [lo]: single merge via l+o (rank 5).
#[test]
fn merge_lo_single_token() {
    let ctx = TokenizerTestContext::with_merges();
    assert_eq!(ctx.encode_with("lo", &no_bos()), [101]);
}

/// "ll" → [ll]: single merge via l+l (rank 4).
#[test]
fn merge_ll_single_token() {
    let ctx = TokenizerTestContext::with_merges();
    assert_eq!(ctx.encode_with("ll", &no_bos()), [100]);
}

/// "hi" has no merge rule for h+i → stays as 2 character tokens.
#[test]
fn no_merge_hi_stays_chars() {
    let ctx = TokenizerTestContext::with_merges();
    // h=76, i=77
    assert_eq!(ctx.encode_with("hi", &no_bos()), [76, 77]);
}

/// "abc" has no merge rules → stays as 3 character tokens.
#[test]
fn no_merge_abc_stays_chars() {
    let ctx = TokenizerTestContext::with_merges();
    assert_eq!(ctx.encode_with("abc", &no_bos()), [69, 70, 71]);
}

/// Merged tokens decode back to original text.
#[test]
fn merge_decode_roundtrip() {
    let ctx = TokenizerTestContext::with_merges();
    let opts = no_bos();

    for input in ["hello", "hell", "hel", "he", "helo", "llo", "lo", "ll"] {
        let tokens = ctx.encode_with(input, &opts);
        let decoded = ctx.decode(&tokens);
        assert_eq!(decoded, input, "roundtrip failed for {input:?}");
    }
}

/// Mixed text with merges: "helloabc" → [hello, a, b, c].
#[test]
fn merge_mixed_text() {
    let ctx = TokenizerTestContext::with_merges();
    // "hello"=104, a=69, b=70, c=71
    assert_eq!(ctx.encode_with("helloabc", &no_bos()), [104, 69, 70, 71]);
}

/// Repeated merge pattern: "hellohello" → [hello, hello].
#[test]
fn merge_repeated_pattern() {
    let ctx = TokenizerTestContext::with_merges();
    assert_eq!(ctx.encode_with("hellohello", &no_bos()), [104, 104]);
}

// ===========================================================================
// talu_tokenizer_tokenize (string representations)
// ===========================================================================

/// tokenize returns string representations of merged tokens.
#[test]
fn tokenize_strings_with_merges() {
    let ctx = TokenizerTestContext::with_merges();
    let text = "hello";
    let result = unsafe {
        talu_sys::talu_tokenizer_tokenize(
            ctx.handle(),
            text.as_bytes().as_ptr(),
            text.len(),
        )
    };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 1);

    let ptrs = unsafe {
        std::slice::from_raw_parts(result.tokens as *const *const i8, result.num_tokens)
    };
    let t0 = unsafe { std::ffi::CStr::from_ptr(ptrs[0]) }
        .to_string_lossy()
        .to_string();
    assert_eq!(t0, "hello");

    unsafe { talu_sys::talu_tokenize_result_free(result.tokens, result.num_tokens) };
}

/// tokenize "helo" returns ["hel", "o"].
#[test]
fn tokenize_strings_partial_merge() {
    let ctx = TokenizerTestContext::with_merges();
    let text = "helo";
    let result = unsafe {
        talu_sys::talu_tokenizer_tokenize(
            ctx.handle(),
            text.as_bytes().as_ptr(),
            text.len(),
        )
    };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 2);

    let ptrs = unsafe {
        std::slice::from_raw_parts(result.tokens as *const *const i8, result.num_tokens)
    };
    let tokens: Vec<String> = (0..result.num_tokens)
        .map(|i| {
            unsafe { std::ffi::CStr::from_ptr(ptrs[i]) }
                .to_string_lossy()
                .to_string()
        })
        .collect();
    assert_eq!(tokens, ["hel", "o"]);

    unsafe { talu_sys::talu_tokenize_result_free(result.tokens, result.num_tokens) };
}

/// tokenize "abc" with no merge rules returns ["a", "b", "c"].
#[test]
fn tokenize_strings_no_merges() {
    let ctx = TokenizerTestContext::with_merges();
    let text = "abc";
    let result = unsafe {
        talu_sys::talu_tokenizer_tokenize(
            ctx.handle(),
            text.as_bytes().as_ptr(),
            text.len(),
        )
    };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 3);

    let ptrs = unsafe {
        std::slice::from_raw_parts(result.tokens as *const *const i8, result.num_tokens)
    };
    let tokens: Vec<String> = (0..result.num_tokens)
        .map(|i| {
            unsafe { std::ffi::CStr::from_ptr(ptrs[i]) }
                .to_string_lossy()
                .to_string()
        })
        .collect();
    assert_eq!(tokens, ["a", "b", "c"]);

    unsafe { talu_sys::talu_tokenize_result_free(result.tokens, result.num_tokens) };
}

// ===========================================================================
// talu_tokenizer_tokenize_bytes (byte representations)
// ===========================================================================

/// tokenize_bytes "hello" returns one token spanning all 5 bytes.
#[test]
fn tokenize_bytes_merged_hello() {
    let ctx = TokenizerTestContext::with_merges();
    let text = "hello";
    let result = unsafe {
        talu_sys::talu_tokenizer_tokenize_bytes(
            ctx.handle(),
            text.as_bytes().as_ptr(),
            text.len(),
        )
    };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 1);

    let offsets = unsafe {
        std::slice::from_raw_parts(result.offsets, result.num_tokens + 1)
    };
    assert_eq!(offsets, [0, 5]);

    let data = unsafe { std::slice::from_raw_parts(result.data, result.data_len) };
    assert_eq!(std::str::from_utf8(data).unwrap(), "hello");

    unsafe {
        talu_sys::talu_tokenize_bytes_result_free(
            result.data, result.data_len, result.offsets, result.num_tokens,
        )
    };
}

/// tokenize_bytes "helo" returns two tokens: "hel" and "o".
#[test]
fn tokenize_bytes_partial_merge() {
    let ctx = TokenizerTestContext::with_merges();
    let text = "helo";
    let result = unsafe {
        talu_sys::talu_tokenizer_tokenize_bytes(
            ctx.handle(),
            text.as_bytes().as_ptr(),
            text.len(),
        )
    };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 2);

    let offsets = unsafe {
        std::slice::from_raw_parts(result.offsets, result.num_tokens + 1)
    };
    let data = unsafe { std::slice::from_raw_parts(result.data, result.data_len) };

    let t0 = std::str::from_utf8(&data[offsets[0]..offsets[1]]).unwrap();
    let t1 = std::str::from_utf8(&data[offsets[1]..offsets[2]]).unwrap();
    assert_eq!(t0, "hel");
    assert_eq!(t1, "o");

    unsafe {
        talu_sys::talu_tokenize_bytes_result_free(
            result.data, result.data_len, result.offsets, result.num_tokens,
        )
    };
}

// ===========================================================================
// compute_offsets with merged tokens
// ===========================================================================

/// compute_offsets for "hello" with merges: 1 token spanning [0, 5).
#[test]
fn compute_offsets_merged_hello() {
    let ctx = TokenizerTestContext::with_merges();
    let text = "hello";
    let result = unsafe {
        talu_sys::talu_tokenizer_compute_offsets(
            ctx.handle(),
            text.as_bytes().as_ptr(),
            text.len(),
        )
    };
    assert!(result.error_msg.is_null());
    assert_eq!(result.len, 1);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.len) };
    assert_eq!(offsets[0].start, 0);
    assert_eq!(offsets[0].end, 5);

    unsafe { talu_sys::talu_offsets_free(result) };
}

/// compute_offsets for "helo" with merges: 2 tokens at [0,3) and [3,4).
#[test]
fn compute_offsets_partial_merge() {
    let ctx = TokenizerTestContext::with_merges();
    let text = "helo";
    let result = unsafe {
        talu_sys::talu_tokenizer_compute_offsets(
            ctx.handle(),
            text.as_bytes().as_ptr(),
            text.len(),
        )
    };
    assert!(result.error_msg.is_null());
    assert_eq!(result.len, 2);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.len) };
    assert_eq!(offsets[0].start, 0);
    assert_eq!(offsets[0].end, 3);
    assert_eq!(offsets[1].start, 3);
    assert_eq!(offsets[1].end, 4);

    unsafe { talu_sys::talu_offsets_free(result) };
}

/// compute_offsets for "hellohello" with merges: 2 tokens each spanning 5 bytes.
#[test]
fn compute_offsets_repeated_merge() {
    let ctx = TokenizerTestContext::with_merges();
    let text = "hellohello";
    let result = unsafe {
        talu_sys::talu_tokenizer_compute_offsets(
            ctx.handle(),
            text.as_bytes().as_ptr(),
            text.len(),
        )
    };
    assert!(result.error_msg.is_null());
    assert_eq!(result.len, 2);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.len) };
    assert_eq!(offsets[0].start, 0);
    assert_eq!(offsets[0].end, 5);
    assert_eq!(offsets[1].start, 5);
    assert_eq!(offsets[1].end, 10);

    unsafe { talu_sys::talu_offsets_free(result) };
}

// ===========================================================================
// Vocab size reflects merged tokens
// ===========================================================================

/// Merges fixture has 105 vocab entries (99 base + 6 merged).
#[test]
fn merges_vocab_size() {
    let ctx = TokenizerTestContext::with_merges();
    let size = unsafe { talu_sys::talu_tokenizer_get_vocab_size(ctx.handle()) };
    assert_eq!(size, 105);
}

// ===========================================================================
// Batch encode with merges
// ===========================================================================

/// Batch encode with merges: "hello" (1 token) + "abc" (3 tokens).
#[test]
fn batch_encode_with_merges() {
    let ctx = TokenizerTestContext::with_merges();
    let batch = ctx.encode_batch(&["hello", "abc"], &no_bos());

    assert_eq!(batch.num_sequences, 2);
    assert_eq!(batch.offsets, [0, 1, 4]);
    assert_eq!(batch.ids, [104, 69, 70, 71]);
}

/// Padded tensor with merges: "hello" (1) padded to match "abc" (3).
#[test]
fn padded_tensor_with_merges() {
    let ctx = TokenizerTestContext::with_merges();
    let pad_opts = talu_sys::PaddedTensorOptions {
        pad_id: 0,
        padding_side: 0, // right
        return_attention_mask: true,
        ..Default::default()
    };

    let result = ctx.batch_to_padded_tensor(&["hello", "abc"], &no_bos(), &pad_opts);

    assert_eq!(result.num_sequences, 2);
    assert_eq!(result.padded_length, 3);

    // Row 0: "hello" = [104, 0, 0] (1 real + 2 pad).
    assert_eq!(&result.input_ids[0..3], &[104, 0, 0]);
    assert_eq!(&result.attention_mask[0..3], &[1, 0, 0]);

    // Row 1: "abc" = [69, 70, 71] (no padding).
    assert_eq!(&result.input_ids[3..6], &[69, 70, 71]);
    assert_eq!(&result.attention_mask[3..6], &[1, 1, 1]);
}
