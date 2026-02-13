//! WordPiece model tests.
//!
//! Tests specific to the WordPiece tokenization algorithm: subword splitting,
//! `##` continuing-subword prefix handling in encode and decode, unknown word
//! fallback, and model type detection.

use crate::capi::tokenizer::common::TokenizerTestContext;

/// Minimal WordPiece tokenizer with BertPreTokenizer and `##` prefix.
const WORDPIECE_JSON: &str = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 100,
    "vocab": {
      "[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3,
      "hello": 4, "world": 5, "go": 6, "good": 7, "morning": 8,
      "##ing": 9, "##ed": 10, "##lab": 11, "##s": 12,
      ",": 13, "!": 14, ".": 15
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true},
    {"id": 1, "content": "[CLS]", "special": true},
    {"id": 2, "content": "[SEP]", "special": true},
    {"id": 3, "content": "[PAD]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}
}"####;

// ---------------------------------------------------------------------------
// Encode: subword splitting
// ---------------------------------------------------------------------------

/// Known whole-word encodes to a single token.
#[test]
fn encode_whole_word() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    assert_eq!(ctx.encode("hello"), vec![4]);
}

/// Word + subword suffix: "going" → ["go", "##ing"].
#[test]
fn encode_subword_split() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    assert_eq!(ctx.encode("going"), vec![6, 9]);
}

/// Multiple words: "hello world" → ["hello", "world"].
#[test]
fn encode_multiple_words() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    assert_eq!(ctx.encode("hello world"), vec![4, 5]);
}

/// Unknown word falls back to [UNK].
#[test]
fn encode_unknown_word() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    // "xyz" is not in vocab and can't be split into known subwords
    let ids = ctx.encode("xyz");
    assert_eq!(ids, vec![0], "unknown word should produce [UNK] token");
}

/// Punctuation is split by BertPreTokenizer.
#[test]
fn encode_punctuation_split() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    // "hello, world!" → ["hello", ",", "world", "!"]
    assert_eq!(ctx.encode("hello, world!"), vec![4, 13, 5, 14]);
}

/// Whitespace-only input produces empty output (no segfault).
#[test]
fn encode_whitespace_only() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    assert_eq!(ctx.encode("   "), Vec::<u32>::new());
}

/// Empty input produces empty output.
#[test]
fn encode_empty() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    assert_eq!(ctx.encode(""), Vec::<u32>::new());
}

// ---------------------------------------------------------------------------
// Decode: ## prefix handling
// ---------------------------------------------------------------------------

/// Single-token decode of a `##` subword strips the prefix.
/// WordPiece decoder always strips `##` — the prefix is a tokenization
/// artifact, not content.
#[test]
fn decode_single_subword_strips_prefix() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    assert_eq!(ctx.decode(&[11]), "lab");
}

/// Single-token decode of `##ing` strips the prefix.
#[test]
fn decode_single_subword_ing_strips_prefix() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    assert_eq!(ctx.decode(&[9]), "ing");
}

/// In multi-token context, ## prefix is stripped and tokens are joined.
/// "go" + "##ing" → "going"
#[test]
fn decode_subword_in_context_strips_prefix() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    assert_eq!(ctx.decode(&[6, 9]), "going");
}

/// Multiple words with subwords: "go" + "##ing" + "good" → "going good"
#[test]
fn decode_mixed_words_and_subwords() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    assert_eq!(ctx.decode(&[6, 9, 7]), "going good");
}

/// Whole-word token decode (no prefix).
#[test]
fn decode_whole_word() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    assert_eq!(ctx.decode(&[4]), "hello");
}

/// Roundtrip: encode then decode.
#[test]
fn roundtrip_going() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    let ids = ctx.encode("going");
    assert_eq!(ctx.decode(&ids), "going");
}

/// Roundtrip: multi-word sentence.
#[test]
fn roundtrip_sentence() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    let ids = ctx.encode("good morning, world!");
    assert_eq!(ctx.decode(&ids), "good morning, world!");
}

// ---------------------------------------------------------------------------
// skip_special_tokens (ignored for WordPiece — BUG)
// ---------------------------------------------------------------------------

/// skip_special_tokens=1 should strip [CLS] and [SEP] from output.
///
/// Bug: `decodeWithOptions()` ignores `skip_special_tokens` for WordPiece
/// models (only BPE honors decode options). This test will fail until fixed.
#[test]
fn skip_special_strips_cls_and_sep() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    let skip = talu_sys::DecodeOptionsC { skip_special_tokens: 1 };
    // [CLS]=1, hello=4, world=5, [SEP]=2
    assert_eq!(
        ctx.decode_with(&[1, 4, 5, 2], &skip), "hello world",
        "skip_special_tokens should strip [CLS] and [SEP]"
    );
}

/// skip_special_tokens=1 on only-special-tokens produces empty string.
#[test]
fn skip_special_all_special_produces_empty() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    let skip = talu_sys::DecodeOptionsC { skip_special_tokens: 1 };
    assert_eq!(ctx.decode_with(&[1, 2], &skip), "");
}
