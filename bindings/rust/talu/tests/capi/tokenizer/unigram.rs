//! Unigram (SentencePiece) model tests.
//!
//! Tests specific to the Unigram tokenization model: array-style vocab loading,
//! Viterbi encoding, Metaspace decode with ▁→space replacement, and
//! skip_special_tokens support.

use crate::capi::tokenizer::common::TokenizerTestContext;

fn no_bos() -> talu_sys::EncodeOptions {
    talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    }
}

fn tokenize_strings(ctx: &TokenizerTestContext, text: &str) -> Vec<String> {
    let result =
        unsafe { talu_sys::talu_tokenizer_tokenize(ctx.handle(), text.as_bytes().as_ptr(), text.len()) };
    assert!(result.error_msg.is_null(), "tokenize failed");
    let tokens = if result.tokens.is_null() || result.num_tokens == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(result.tokens, result.num_tokens) }
            .iter()
            .map(|ptr| unsafe { std::ffi::CStr::from_ptr(*ptr) }.to_str().unwrap().to_owned())
            .collect()
    };
    unsafe { talu_sys::talu_tokenize_result_free(result.tokens, result.num_tokens) };
    tokens
}

fn tokenize_bytes_strings(ctx: &TokenizerTestContext, text: &str) -> Vec<String> {
    let result = unsafe {
        talu_sys::talu_tokenizer_tokenize_bytes(ctx.handle(), text.as_bytes().as_ptr(), text.len())
    };
    assert!(result.error_msg.is_null(), "tokenize_bytes failed");
    let data = unsafe { std::slice::from_raw_parts(result.data, result.data_len) };
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens + 1) };
    let tokens: Vec<String> = offsets
        .windows(2)
        .map(|w| std::str::from_utf8(&data[w[0]..w[1]]).unwrap().to_owned())
        .collect();
    unsafe {
        talu_sys::talu_tokenize_bytes_result_free(
            result.data,
            result.data_len,
            result.offsets,
            result.num_tokens,
        )
    };
    tokens
}

// ---------------------------------------------------------------------------
// Loading: Unigram JSON fails to load
// ---------------------------------------------------------------------------

/// Unigram tokenizer JSON with array-style vocab must load successfully.
///
/// Unigram uses `[["token", score], ...]` array format instead of the
/// object-style `{"token": id}` format. The loader must parse both.
#[test]
fn load_from_json() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "unk_token": "<unk>",
    "vocab": [["<unk>", 0.0], ["hello", -2.0], ["world", -3.0]]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    let bytes = json.as_bytes();
    let mut handle: *mut std::ffi::c_void = std::ptr::null_mut();
    let rc = unsafe {
        talu_sys::talu_tokenizer_create_from_json(
            bytes.as_ptr(),
            bytes.len(),
            &mut handle as *mut _ as *mut std::ffi::c_void,
        )
    };
    assert_eq!(
        rc, 0,
        "Unigram tokenizer must load from JSON (got error {rc})"
    );
    assert!(!handle.is_null());
    unsafe { talu_sys::talu_tokenizer_free(handle) };
}

/// Metaspace decoder with `add_prefix_space: true` must strip the leading space
/// during decode. SentencePiece tokens use ▁ (U+2581) for word boundaries;
/// the first ▁ was added by the pretokenizer and must be removed on decode.
#[test]
fn metaspace_decode_strips_prefix_space() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "unk_id": 0,
    "vocab": [["<unk>", 0.0], ["\u2581Clark", -2.0], ["\u2581grant", -3.0], ["s", -1.5]]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true },
  "decoder": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true },
  "post_processor": null
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    // Decode [1, 3, 2] → "▁Clark" + "s" + "▁grant" → " Clark" + "s" + " grant"
    // With add_prefix_space strip → "Clarks grant"
    let decoded = ctx.decode(&[1, 3, 2]);
    assert_eq!(
        decoded, "Clarks grant",
        "Metaspace add_prefix_space must strip leading space, got: {decoded:?}"
    );
}

/// With explicit `skip_special_tokens=1`, Unigram decode must remove special
/// tokens before applying metaspace cleanup.
#[test]
fn unigram_decode_skips_special_tokens_when_requested() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "unk_id": 0,
    "vocab": [
      ["<unk>", 0.0],
      ["\u2581hello", -2.0],
      ["\u2581world", -3.0]
    ]
  },
  "added_tokens": [
    {"id": 3, "content": "<s>", "special": true},
    {"id": 4, "content": "</s>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true },
  "decoder": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true },
  "post_processor": null
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    let skip = talu_sys::DecodeOptionsC {
        skip_special_tokens: 1,
    };
    assert_eq!(ctx.decode_with(&[3, 1, 2, 4], &skip), "hello world");
}

/// Unigram special-token skipping should match the oracle of manually removing
/// special IDs before decode.
#[test]
fn unigram_decode_skip_special_matches_manual_filtering() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "unk_id": 0,
    "vocab": [
      ["<unk>", 0.0],
      ["\u2581hello", -2.0],
      ["\u2581world", -3.0]
    ]
  },
  "added_tokens": [
    {"id": 3, "content": "<s>", "special": true},
    {"id": 4, "content": "</s>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true },
  "decoder": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true },
  "post_processor": null
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    let skip = talu_sys::DecodeOptionsC {
        skip_special_tokens: 1,
    };
    let manual = ctx.decode(&[1, 2]);
    let skipped = ctx.decode_with(&[3, 1, 2, 4], &skip);
    assert_eq!(skipped, manual);
}

/// With special-token skipping disabled, Unigram decode must retain the
/// special token text in order around metaspace-decoded tokens.
#[test]
fn unigram_decode_retains_special_tokens_when_requested() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "unk_id": 0,
    "vocab": [
      ["<unk>", 0.0],
      ["\u2581hello", -2.0],
      ["\u2581world", -3.0]
    ]
  },
  "added_tokens": [
    {"id": 3, "content": "<s>", "special": true},
    {"id": 4, "content": "</s>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true },
  "decoder": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true },
  "post_processor": null
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    let keep = talu_sys::DecodeOptionsC {
        skip_special_tokens: 0,
    };
    assert_eq!(ctx.decode_with(&[3, 1, 2, 4], &keep), "<s> hello world</s>");
}

/// A single added special token should decode to its literal content in
/// retain mode rather than producing a decoder error.
#[test]
fn unigram_decode_single_special_token_retained() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "unk_id": 0,
    "vocab": [
      ["<unk>", 0.0],
      ["\u2581hello", -2.0]
    ]
  },
  "added_tokens": [
    {"id": 2, "content": "<s>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true },
  "decoder": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true },
  "post_processor": null
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    let keep = talu_sys::DecodeOptionsC {
        skip_special_tokens: 0,
    };
    assert_eq!(ctx.decode_with(&[2], &keep), "<s>");
}

/// A single added special token should decode to empty output in skip mode
/// rather than producing a decoder error.
#[test]
fn unigram_decode_single_special_token_skipped() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "unk_id": 0,
    "vocab": [
      ["<unk>", 0.0],
      ["\u2581hello", -2.0]
    ]
  },
  "added_tokens": [
    {"id": 2, "content": "<s>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true },
  "decoder": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true },
  "post_processor": null
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    let skip = talu_sys::DecodeOptionsC {
        skip_special_tokens: 1,
    };
    assert_eq!(ctx.decode_with(&[2], &skip), "");
}

/// Null decode options must use the C-API default `skip_special_tokens=true`
/// for Unigram too, not error out on added special IDs.
#[test]
fn unigram_decode_null_options_skip_special_defaults_true() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "unk_id": 0,
    "vocab": [
      ["<unk>", 0.0],
      ["\u2581hello", -2.0]
    ]
  },
  "added_tokens": [
    {"id": 2, "content": "<s>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true },
  "decoder": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true },
  "post_processor": null
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    let result = unsafe { super::common::decode_raw_null_options(ctx.handle(), &[2, 1]) };
    assert!(
        result.error_msg.is_null(),
        "null decode options should succeed for unigram special tokens"
    );
    let text = unsafe {
        let slice = std::slice::from_raw_parts(result.text, result.text_len);
        std::str::from_utf8(slice).expect("decode must return valid UTF-8")
    };
    assert_eq!(text, "hello");
    unsafe { talu_sys::talu_decode_result_free(result.text, result.text_len) };
}

// ---------------------------------------------------------------------------
// Unigram encode: basic encoding must produce correct tokens
// ---------------------------------------------------------------------------
//
// The Unigram model uses the Viterbi algorithm to find the most likely
// segmentation based on token scores (log probabilities). Tokens with higher
// scores are preferred.

/// Unigram encoder must use Viterbi to find optimal segmentation.
///
/// With a vocab where "▁hello" has a high score, encoding "hello" with
/// Metaspace pretokenizer (which prepends ▁) must produce ["▁hello"],
/// not character-level fallback.
#[test]
fn unigram_encode_produces_correct_tokens() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "unk_id": 0,
    "vocab": [
      ["<unk>", 0.0],
      ["\u2581", -1.0],
      ["\u2581hello", -5.0],
      ["\u2581world", -5.0],
      ["h", -10.0],
      ["e", -10.0],
      ["l", -10.0],
      ["o", -10.0],
      ["w", -10.0],
      ["r", -10.0],
      ["d", -10.0]
    ]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true },
  "post_processor": null,
  "decoder": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true }
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    // "hello" → Metaspace prepend → "▁hello" → Unigram Viterbi → [2] (▁hello)
    let tokens = ctx.encode_with("hello", &opts);
    assert_eq!(
        tokens,
        vec![2],
        "Unigram must use Viterbi to match '▁hello' (score -5.0) over char fallback, got: {tokens:?}"
    );
}

/// The tokenization APIs must expose the raw SentencePiece token string with
/// the leading metaspace marker intact.
#[test]
fn unigram_tokenize_surfaces_expose_metaspace_prefixed_token() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "unk_id": 0,
    "vocab": [
      ["<unk>", 0.0],
      ["\u2581", -1.0],
      ["\u2581hello", -5.0]
    ]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true },
  "post_processor": null,
  "decoder": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true }
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    assert_eq!(tokenize_strings(&ctx, "hello"), vec!["▁hello"]);
    assert_eq!(tokenize_bytes_strings(&ctx, "hello"), vec!["▁hello"]);
}

/// A metaspace-prefixed whole-word token should map back to the real source
/// word span, not claim a synthetic leading byte.
#[test]
fn unigram_offsets_whole_word_ignore_synthetic_prefix_space() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "unk_id": 0,
    "vocab": [
      ["<unk>", 0.0],
      ["\u2581", -1.0],
      ["\u2581hello", -5.0]
    ]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true },
  "post_processor": null,
  "decoder": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true }
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    let result = unsafe { super::common::encode_raw(ctx.handle(), b"hello", &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 1);
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 5));
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Multiword metaspace whole-word encoding must preserve the real source spans
/// of each word rather than collapsing both tokens to zero-width.
#[test]
fn unigram_offsets_multiword_whole_tokens_map_to_each_word() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "unk_id": 0,
    "vocab": [
      ["<unk>", 0.0],
      ["\u2581", -1.0],
      ["\u2581hello", -5.0],
      ["\u2581world", -5.0]
    ]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true },
  "post_processor": null,
  "decoder": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true }
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    let result = unsafe { super::common::encode_raw(ctx.handle(), b"hello world", &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 2);

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    assert_eq!(ids, &[2, 3]);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 5));
    assert_eq!((offsets[1].start, offsets[1].end), (6, 11));

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Unigram encoder handles multi-word input with Metaspace.
///
/// "hello world" → Metaspace → "▁hello▁world" → Unigram → [▁hello, ▁world].
#[test]
fn unigram_encode_multiword() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "unk_id": 0,
    "vocab": [
      ["<unk>", 0.0],
      ["\u2581", -1.0],
      ["\u2581hello", -5.0],
      ["\u2581world", -5.0],
      ["h", -10.0],
      ["e", -10.0],
      ["l", -10.0],
      ["o", -10.0],
      ["w", -10.0],
      ["r", -10.0],
      ["d", -10.0]
    ]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true },
  "post_processor": null,
  "decoder": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true }
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    // "hello world" → "▁hello ▁world" → Unigram → [▁hello=2, ▁world=3]
    let tokens = ctx.encode_with("hello world", &opts);
    assert_eq!(
        tokens,
        vec![2, 3],
        "Unigram multi-word encoding with Metaspace must produce [▁hello, ▁world], got: {tokens:?}"
    );
}

// ---------------------------------------------------------------------------
// Character-level fallback
// ---------------------------------------------------------------------------

/// Shared Unigram fixture with character-level fallback tokens.
///
/// Vocab has whole-word tokens with high scores and single-char tokens
/// with low scores. Words not in vocab fall back to character-level.
const UNIGRAM_FALLBACK_JSON: &str = r#"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "unk_id": 0,
    "vocab": [
      ["<unk>", 0.0],
      ["\u2581", -1.0],
      ["\u2581hello", -5.0],
      ["\u2581world", -5.0],
      ["h", -10.0],
      ["e", -10.0],
      ["l", -10.0],
      ["o", -10.0],
      ["w", -10.0],
      ["r", -10.0],
      ["d", -10.0],
      ["a", -10.0],
      ["b", -10.0],
      ["c", -10.0],
      ["\u2581he", -6.0],
      ["\u2581hel", -7.0]
    ]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true },
  "post_processor": null,
  "decoder": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true }
}"#;

/// Word not in vocab falls back to character-level tokens.
///
/// "abc" is not a whole-word token. Viterbi segments it as [a, b, c]
/// using single-character fallback tokens.
#[test]
fn unigram_char_fallback_for_unknown_word() {
    let ctx = TokenizerTestContext::from_json(UNIGRAM_FALLBACK_JSON);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    // "abc" → Metaspace → "▁abc"
    // No whole-word "▁abc", falls back to char-level: [▁, a, b, c]
    let tokens = ctx.encode_with("abc", &opts);
    assert_eq!(
        tokens,
        vec![1, 11, 12, 13],
        "unknown word must fall back to exact char sequence [▁,a,b,c]"
    );
}

/// Character fallback must remain visible on both tokenization APIs, including
/// the standalone metaspace marker inserted for the word boundary.
#[test]
fn unigram_tokenize_surfaces_expose_char_fallback_sequence() {
    let ctx = TokenizerTestContext::from_json(UNIGRAM_FALLBACK_JSON);
    assert_eq!(tokenize_strings(&ctx, "abc"), vec!["▁", "a", "b", "c"]);
    assert_eq!(tokenize_bytes_strings(&ctx, "abc"), vec!["▁", "a", "b", "c"]);
}

/// The tokenization surfaces must show distinct metaspace-prefixed tokens for
/// multiword whole-word matches, not collapse the word boundary.
#[test]
fn unigram_tokenize_surfaces_expose_multiword_whole_tokens() {
    let ctx = TokenizerTestContext::from_json(UNIGRAM_FALLBACK_JSON);
    assert_eq!(tokenize_strings(&ctx, "hello world"), vec!["▁hello", "▁world"]);
    assert_eq!(
        tokenize_bytes_strings(&ctx, "hello world"),
        vec!["▁hello", "▁world"]
    );
}

/// For a single-word char fallback path, the synthetic metaspace token must be
/// zero-width and the remaining characters must map exactly to source bytes.
#[test]
fn unigram_offsets_char_fallback_keep_synthetic_prefix_zero_width() {
    let ctx = TokenizerTestContext::from_json(UNIGRAM_FALLBACK_JSON);
    let result = unsafe { super::common::encode_raw(ctx.handle(), b"abc", &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 4);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 0));
    assert_eq!((offsets[1].start, offsets[1].end), (0, 1));
    assert_eq!((offsets[2].start, offsets[2].end), (1, 2));
    assert_eq!((offsets[3].start, offsets[3].end), (2, 3));

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Viterbi prefers longer tokens when they have better total score.
///
/// "hello" → "▁hello" (score -5.0) is better than "▁he"+"l"+"l"+"o"
/// (score -6.0 + -10.0 + -10.0 + -10.0 = -36.0). Viterbi must choose
/// the whole word.
#[test]
fn unigram_viterbi_prefers_whole_word() {
    let ctx = TokenizerTestContext::from_json(UNIGRAM_FALLBACK_JSON);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    let tokens = ctx.encode_with("hello", &opts);
    // Should be a single token [▁hello=2], not multiple subwords
    assert_eq!(
        tokens,
        vec![2],
        "Viterbi must prefer whole-word '▁hello' over subword split, got: {tokens:?}"
    );
}

/// Empty input produces empty output.
#[test]
fn unigram_encode_empty() {
    let ctx = TokenizerTestContext::from_json(UNIGRAM_FALLBACK_JSON);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    let tokens = ctx.encode_with("", &opts);
    assert_eq!(tokens, Vec::<u32>::new(), "empty input must produce empty output");
}

/// Encode→decode roundtrip preserves text.
#[test]
fn unigram_roundtrip() {
    let ctx = TokenizerTestContext::from_json(UNIGRAM_FALLBACK_JSON);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    for text in ["hello", "hello world", "abc"] {
        let tokens = ctx.encode_with(text, &opts);
        let decoded = ctx.decode(&tokens);
        assert_eq!(decoded, text, "Unigram roundtrip failed for {text:?}");
    }
}

/// Single character input encodes to character token.
#[test]
fn unigram_single_char() {
    let ctx = TokenizerTestContext::from_json(UNIGRAM_FALLBACK_JSON);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    // "a" → Metaspace → "▁a" → Unigram → [▁, a] or similar subword split
    let tokens = ctx.encode_with("a", &opts);
    assert_eq!(tokens, vec![1, 11], "single char should split as [▁, a]");
}

/// Viterbi selects subword split when no whole-word match exists.
///
/// "hel" → Metaspace → "▁hel" → "▁hel" (score -7.0) is better than
/// "▁he"+"l" (score -6.0 + -10.0 = -16.0). Viterbi should pick "▁hel".
#[test]
fn unigram_viterbi_subword_selection() {
    let ctx = TokenizerTestContext::from_json(UNIGRAM_FALLBACK_JSON);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    let tokens = ctx.encode_with("hel", &opts);
    assert_eq!(
        tokens,
        vec![15],
        "Viterbi should prefer exact token ▁hel (id=15), got: {tokens:?}"
    );
}

/// Equal-score paths must remain deterministic: ties keep the first path
/// encountered in vocab order because the DP only updates on strictly better
/// scores.
#[test]
fn unigram_equal_score_tie_prefers_earlier_whole_token() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "vocab": [
      ["ab", -2.0],
      ["a", -1.0],
      ["b", -1.0]
    ]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    let tokens = ctx.encode_with("ab", &opts);
    assert_eq!(
        tokens,
        vec![0],
        "equal-score tie must keep the earlier whole-token path, got: {tokens:?}"
    );
}

/// The same equal-score tie must flip when vocab order changes, proving the
/// contract is vocab-order determinism rather than an accidental preference for
/// whole-word matches.
#[test]
fn unigram_equal_score_tie_tracks_vocab_order() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "vocab": [
      ["a", -1.0],
      ["ab", -2.0],
      ["b", -1.0]
    ]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    let tokens = ctx.encode_with("ab", &opts);
    assert_eq!(
        tokens,
        vec![0, 2],
        "equal-score tie must follow first-seen vocab order, got: {tokens:?}"
    );
}

/// Multi-branch equal-score ties must still follow first-seen vocab order at
/// each DP position, not an accidental preference for fewer tokens or longer
/// matches.
#[test]
fn unigram_equal_score_multibranch_tie_prefers_first_seen_path() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "vocab": [
      ["a", -1.0],
      ["ab", -2.0],
      ["abc", -3.0],
      ["bc", -2.0],
      ["b", -1.0],
      ["c", -1.0]
    ]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    let tokens = ctx.encode_with("abc", &opts);
    assert_eq!(
        tokens,
        vec![0, 3],
        "multi-branch equal-score tie must keep the first-seen path per DP state, got: {tokens:?}"
    );
}

/// Reordering the same equal-score multi-branch vocab must flip the chosen
/// path, proving the contract is stable vocab-order determinism rather than an
/// incidental longest-match heuristic.
#[test]
fn unigram_equal_score_multibranch_tie_tracks_vocab_order() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "vocab": [
      ["abc", -3.0],
      ["ab", -2.0],
      ["a", -1.0],
      ["bc", -2.0],
      ["b", -1.0],
      ["c", -1.0]
    ]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    let tokens = ctx.encode_with("abc", &opts);
    assert_eq!(
        tokens,
        vec![0],
        "reordered multi-branch equal-score tie must follow first-seen vocab order, got: {tokens:?}"
    );
}

/// Near-epsilon score advantages must still be honored deterministically. A
/// whole token that is slightly better than the split path must win.
#[test]
fn unigram_near_epsilon_score_advantage_prefers_whole_token() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "vocab": [
      ["ab", -2.0000000],
      ["a", -1.0000000],
      ["b", -1.0000010]
    ]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode_with("ab", &no_bos());
    assert_eq!(
        tokens,
        vec![0],
        "slightly better whole-token score must beat the split path, got: {tokens:?}"
    );
}

/// The converse near-epsilon case must also be stable: if the split path is
/// slightly better than the whole token, Viterbi must choose the split.
#[test]
fn unigram_near_epsilon_score_advantage_prefers_split_path() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "vocab": [
      ["ab", -2.0000020],
      ["a", -1.0000000],
      ["b", -1.0000010]
    ]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode_with("ab", &no_bos());
    assert_eq!(
        tokens,
        vec![1, 2],
        "slightly better split path must beat the whole token, got: {tokens:?}"
    );
}

// ---------------------------------------------------------------------------
// Unigram with unk_id: unknown chars produce unk token
// ---------------------------------------------------------------------------

/// Characters not in vocab produce the unk token.
#[test]
fn unigram_unknown_char_produces_unk() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "unk_id": 0,
    "vocab": [
      ["<unk>", 0.0],
      ["a", -1.0],
      ["b", -1.0]
    ]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    // "z" is not in vocab → should produce unk token
    let tokens = ctx.encode_with("z", &opts);
    assert_eq!(tokens, vec![0], "unknown char 'z' must produce [unk]");
}

/// Long unbroken input should complete deterministically without panics.
#[test]
fn unigram_long_unbroken_input_deterministic() {
    let ctx = TokenizerTestContext::from_json(UNIGRAM_FALLBACK_JSON);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    let input = "a".repeat(20_000);
    let first = ctx.encode_with(&input, &opts);
    let second = ctx.encode_with(&input, &opts);
    assert_eq!(first, second, "long unigram input must be deterministic");
    assert!(
        !first.is_empty(),
        "long unigram input should produce at least one token"
    );
}

/// Long repeated input must not drift to a different local segmentation based
/// on absolute position in the stream.
#[test]
fn unigram_long_repeated_words_keep_same_local_segmentation() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "unk_id": 0,
    "vocab": [
      ["<unk>", 0.0],
      ["\u2581hello", -0.1],
      ["\u2581he", -0.2],
      ["llo", -0.2]
    ]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true },
  "decoder": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true },
  "post_processor": null
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    let input = std::iter::repeat_n("hello", 2000).collect::<Vec<_>>().join(" ");
    let tokens = ctx.encode_with(&input, &no_bos());

    assert_eq!(tokens.len(), 2000, "each repeated word should stay one token");
    assert!(
        tokens.iter().all(|&id| id == 1),
        "all repeated words should keep the same winning token regardless of absolute position"
    );
}

/// Mixed known and unknown chars: known chars encode, unknown → unk.
#[test]
fn unigram_mixed_known_unknown() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "unk_id": 0,
    "vocab": [
      ["<unk>", 0.0],
      ["a", -1.0],
      ["b", -1.0]
    ]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    // "azb" → [a=1, unk=0, b=2]
    let tokens = ctx.encode_with("azb", &opts);
    assert_eq!(
        tokens,
        vec![1, 0, 2],
        "known chars encode, unknown → unk: 'azb' → [a, unk, b], got: {tokens:?}"
    );
}

/// Without a pre-tokenizer, unknown characters should still be visible as
/// `<unk>` on both tokenization surfaces rather than disappearing.
#[test]
fn unigram_tokenize_surfaces_expose_unknown_char_without_pretokenizer() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "unk_id": 0,
    "vocab": [
      ["<unk>", 0.0],
      ["a", -1.0],
      ["b", -1.0]
    ]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    assert_eq!(tokenize_strings(&ctx, "z"), vec!["<unk>"]);
    assert_eq!(tokenize_bytes_strings(&ctx, "z"), vec!["<unk>"]);
}

/// Unknown characters without a pre-tokenizer must keep ownership of their
/// original byte span in encode offsets.
#[test]
fn unigram_offsets_unknown_char_without_pretokenizer_cover_source_span() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "unk_id": 0,
    "vocab": [
      ["<unk>", 0.0],
      ["a", -1.0],
      ["b", -1.0]
    ]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    let result = unsafe { super::common::encode_raw(ctx.handle(), b"z", &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 1);

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    assert_eq!(ids, &[0]);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 1));

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Batch Unigram encoding must slice into the same per-sequence IDs as
/// individual encoding for both whole-word and char-fallback cases.
#[test]
fn unigram_batch_matches_individual_wholeword_and_charfallback() {
    let ctx = TokenizerTestContext::from_json(UNIGRAM_FALLBACK_JSON);
    let batch = ctx.encode_batch(&["hello", "abc", ""], &no_bos());
    assert_eq!(batch.num_sequences, 3);
    assert_eq!(batch.offsets, vec![0, 1, 5, 5]);
    assert_eq!(batch.ids[batch.offsets[0]..batch.offsets[1]], ctx.encode_with("hello", &no_bos()));
    assert_eq!(batch.ids[batch.offsets[1]..batch.offsets[2]], ctx.encode_with("abc", &no_bos()));
    assert_eq!(batch.ids[batch.offsets[2]..batch.offsets[3]], ctx.encode_with("", &no_bos()));
}
