//! Unigram (SentencePiece) model tests.
//!
//! Tests specific to the Unigram tokenization model: array-style vocab loading,
//! Viterbi encoding, Metaspace decode with ▁→space replacement, and
//! skip_special_tokens support.

use crate::capi::tokenizer::common::TokenizerTestContext;

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
