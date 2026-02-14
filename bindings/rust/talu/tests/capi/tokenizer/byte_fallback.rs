//! SentencePiece byte-fallback decoder tests.
//!
//! Tests `<0xXX>` byte-fallback tokens used by SentencePiece-based models
//! for representing raw bytes when subword tokenization cannot find a match
//! in the vocabulary.
//!
//! The decoder pipeline for these models is typically:
//!   Sequence(Replace(▁→space), ByteFallback, Fuse, Strip(start=1))
//!
//! ByteFallback converts `<0xXX>` tokens back to raw bytes, which must be
//! correctly reassembled into multi-byte UTF-8 characters.

use crate::capi::tokenizer::common::TokenizerTestContext;

/// Minimal SentencePiece-style BPE tokenizer with byte-fallback tokens.
///
/// Vocab layout:
///   0: `<unk>`, 1: `<s>`, 2: `</s>`
///   3: `▁` (standalone word boundary)
///   4: `▁hello`, 5: `▁world`, 6: `cat`
///   7–16: selected `<0xXX>` byte-fallback tokens for testing
///
/// Decoder: Sequence(Replace(▁→space), ByteFallback, Fuse, Strip(start=1))
const BYTE_FALLBACK_JSON: &str = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "unk_token": "<unk>",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "▁": 3,
      "▁hello": 4, "▁world": 5, "cat": 6,
      "<0x48>": 7, "<0x69>": 8,
      "<0xC3>": 9, "<0xA9>": 10,
      "<0xE4>": 11, "<0xB8>": 12, "<0x96>": 13,
      "<0xF0>": 14, "<0x9F>": 15, "<0x98>": 16, "<0x8A>": 17
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": {
    "type": "Sequence",
    "decoders": [
      {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
      {"type": "ByteFallback"},
      {"type": "Fuse"},
      {"type": "Strip", "content": " ", "start": 1, "stop": 0}
    ]
  }
}
"####;

// ---------------------------------------------------------------------------
// Single-byte fallback
// ---------------------------------------------------------------------------

/// `<0x48>` (H) decodes to "H".
#[test]
fn decode_single_byte_ascii_h() {
    let ctx = TokenizerTestContext::from_json(BYTE_FALLBACK_JSON);
    assert_eq!(ctx.decode(&[7]), "H");
}

/// `<0x69>` (i) decodes to "i".
#[test]
fn decode_single_byte_ascii_i() {
    let ctx = TokenizerTestContext::from_json(BYTE_FALLBACK_JSON);
    assert_eq!(ctx.decode(&[8]), "i");
}

/// `<0x48>` + `<0x69>` → "Hi".
#[test]
fn decode_two_ascii_bytes() {
    let ctx = TokenizerTestContext::from_json(BYTE_FALLBACK_JSON);
    assert_eq!(ctx.decode(&[7, 8]), "Hi");
}

// ---------------------------------------------------------------------------
// Multi-byte UTF-8 reconstruction
// ---------------------------------------------------------------------------

/// `<0xC3>` + `<0xA9>` → "é" (2-byte UTF-8: U+00E9).
#[test]
fn decode_two_byte_utf8_e_acute() {
    let ctx = TokenizerTestContext::from_json(BYTE_FALLBACK_JSON);
    assert_eq!(ctx.decode(&[9, 10]), "é");
}

/// `<0xE4>` + `<0xB8>` + `<0x96>` → "世" (3-byte UTF-8: U+4E16).
#[test]
fn decode_three_byte_utf8_cjk() {
    let ctx = TokenizerTestContext::from_json(BYTE_FALLBACK_JSON);
    assert_eq!(ctx.decode(&[11, 12, 13]), "世");
}

/// `<0xF0>` + `<0x9F>` + `<0x98>` + `<0x8A>` → "😊" (4-byte UTF-8: U+1F60A).
#[test]
fn decode_four_byte_utf8_emoji() {
    let ctx = TokenizerTestContext::from_json(BYTE_FALLBACK_JSON);
    assert_eq!(ctx.decode(&[14, 15, 16, 17]), "😊");
}

// ---------------------------------------------------------------------------
// Regular tokens with ▁ → space replacement and strip
// ---------------------------------------------------------------------------

/// `▁hello` decodes to "hello" (▁ → space, strip leading space).
#[test]
fn decode_regular_token_with_word_boundary() {
    let ctx = TokenizerTestContext::from_json(BYTE_FALLBACK_JSON);
    assert_eq!(ctx.decode(&[4]), "hello");
}

/// `▁hello` + `▁world` → "hello world".
#[test]
fn decode_two_regular_tokens() {
    let ctx = TokenizerTestContext::from_json(BYTE_FALLBACK_JSON);
    assert_eq!(ctx.decode(&[4, 5]), "hello world");
}

// ---------------------------------------------------------------------------
// Mixed regular tokens + byte-fallback
// ---------------------------------------------------------------------------

/// `▁hello` + `▁` + `<0xC3>` + `<0xA9>` → "hello é".
///
/// The standalone `▁` (ID 3) becomes a space separator between "hello" and
/// the byte-fallback é.
#[test]
fn decode_regular_then_byte_fallback_utf8() {
    let ctx = TokenizerTestContext::from_json(BYTE_FALLBACK_JSON);
    assert_eq!(ctx.decode(&[4, 3, 9, 10]), "hello é");
}

/// `▁hello` + `<0xF0>` + `<0x9F>` + `<0x98>` + `<0x8A>` → "hello😊".
///
/// No separator between regular token and byte-fallback emoji.
#[test]
fn decode_regular_then_byte_fallback_emoji() {
    let ctx = TokenizerTestContext::from_json(BYTE_FALLBACK_JSON);
    assert_eq!(ctx.decode(&[4, 14, 15, 16, 17]), "hello😊");
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

/// Empty token array produces empty string.
#[test]
fn decode_empty() {
    let ctx = TokenizerTestContext::from_json(BYTE_FALLBACK_JSON);
    assert_eq!(ctx.decode(&[]), "");
}
