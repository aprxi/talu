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
// Encode offsets with merged tokens
// ===========================================================================

/// Encoding "hello" with merges: 1 token spanning [0, 5).
#[test]
fn encode_offsets_merged_hello() {
    let ctx = TokenizerTestContext::with_merges();
    let text = "hello";
    let result = unsafe {
        super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos())
    };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 1);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!(offsets[0].start, 0);
    assert_eq!(offsets[0].end, 5);

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Encoding "helo" with merges: 2 tokens at [0,3) and [3,4).
#[test]
fn encode_offsets_partial_merge() {
    let ctx = TokenizerTestContext::with_merges();
    let text = "helo";
    let result = unsafe {
        super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos())
    };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 2);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!(offsets[0].start, 0);
    assert_eq!(offsets[0].end, 3);
    assert_eq!(offsets[1].start, 3);
    assert_eq!(offsets[1].end, 4);

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Encoding "hellohello" with merges: 2 tokens each spanning 5 bytes.
#[test]
fn encode_offsets_repeated_merge() {
    let ctx = TokenizerTestContext::with_merges();
    let text = "hellohello";
    let result = unsafe {
        super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos())
    };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 2);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!(offsets[0].start, 0);
    assert_eq!(offsets[0].end, 5);
    assert_eq!(offsets[1].start, 5);
    assert_eq!(offsets[1].end, 10);

    unsafe { talu_sys::talu_encode_result_free(result) };
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

// ===========================================================================
// tokenize/tokenize_bytes must NOT include special tokens
// ===========================================================================
//
// tokenizeToBytes used the full encode pipeline (with post_processor),
// so a tokenizer with BOS would include <|begin_of_text|> in the output.
// tokenize/tokenize_bytes is a pre-encode step that should only run
// pretokenization + model (BPE/WP/Unigram), never the post_processor.

/// Minimal BPE tokenizer with Sequence-wrapped TemplateProcessing.
/// When encoded with add_bos=1, BOS (ID 1) is prepended.
/// tokenize/tokenize_bytes must NOT include BOS.
const TOKENIZE_BOS_JSON: &str = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "H": 4, "i": 5, "e": 6, "l": 7, "o": 8
    },
    "merges": ["H i"]
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<|begin_of_text|>", "special": true},
    {"id": 2, "content": "<|end_of_text|>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": {
    "type": "Sequence",
    "processors": [
      {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": false},
      {
        "type": "TemplateProcessing",
        "single": [
          {"SpecialToken": {"id": "<|begin_of_text|>", "type_id": 0}},
          {"Sequence": {"id": "A", "type_id": 0}}
        ],
        "pair": [
          {"SpecialToken": {"id": "<|begin_of_text|>", "type_id": 0}},
          {"Sequence": {"id": "A", "type_id": 0}},
          {"Sequence": {"id": "B", "type_id": 0}}
        ],
        "special_tokens": {
          "<|begin_of_text|>": {"id": "<|begin_of_text|>", "ids": [1], "tokens": ["<|begin_of_text|>"]}
        }
      }
    ]
  },
  "decoder": {"type": "ByteLevel"}
}"####;

/// tokenize_bytes with a post_processor must NOT include BOS token bytes.
#[test]
fn tokenize_bytes_excludes_special_tokens() {
    let ctx = TokenizerTestContext::from_json(TOKENIZE_BOS_JSON);
    let text = "Hi";
    let result = unsafe {
        talu_sys::talu_tokenizer_tokenize_bytes(
            ctx.handle(),
            text.as_bytes().as_ptr(),
            text.len(),
        )
    };
    assert!(result.error_msg.is_null());

    // "Hi" should tokenize without BOS — only the raw subword tokens.
    // With the "H i" merge rule: "Hi" → ["Hi"] (1 token).
    // Bug would produce ["<|begin_of_text|>", "Hi"] (2 tokens).
    assert_eq!(
        result.num_tokens, 1,
        "tokenize_bytes must not include BOS, expected 1 token, got {}",
        result.num_tokens
    );

    let offsets = unsafe {
        std::slice::from_raw_parts(result.offsets, result.num_tokens + 1)
    };
    let data = unsafe { std::slice::from_raw_parts(result.data, result.data_len) };
    let t0 = std::str::from_utf8(&data[offsets[0]..offsets[1]]).unwrap();
    assert_eq!(t0, "Hi");

    unsafe {
        talu_sys::talu_tokenize_bytes_result_free(
            result.data, result.data_len, result.offsets, result.num_tokens,
        )
    };
}

/// tokenize (strings) with a post_processor must NOT include BOS token.
#[test]
fn tokenize_strings_excludes_special_tokens() {
    let ctx = TokenizerTestContext::from_json(TOKENIZE_BOS_JSON);
    let text = "Hi";
    let result = unsafe {
        talu_sys::talu_tokenizer_tokenize(
            ctx.handle(),
            text.as_bytes().as_ptr(),
            text.len(),
        )
    };
    assert!(result.error_msg.is_null());

    // Must be 1 token ("Hi"), not 2 (BOS + "Hi").
    assert_eq!(
        result.num_tokens, 1,
        "tokenize must not include BOS, expected 1 token, got {}",
        result.num_tokens
    );

    let ptrs = unsafe {
        std::slice::from_raw_parts(result.tokens as *const *const i8, result.num_tokens)
    };
    let t0 = unsafe { std::ffi::CStr::from_ptr(ptrs[0]) }
        .to_string_lossy()
        .to_string();
    assert_eq!(t0, "Hi");

    unsafe { talu_sys::talu_tokenize_result_free(result.tokens, result.num_tokens) };
}

// ===========================================================================
// Batch encode with merges
// ===========================================================================

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
