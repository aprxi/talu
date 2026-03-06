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

/// A standalone SentencePiece word-boundary token decodes to a leading space,
/// then the Strip decoder removes that one leading space.
#[test]
fn decode_standalone_word_boundary_strips_to_empty() {
    let ctx = TokenizerTestContext::from_json(BYTE_FALLBACK_JSON);
    assert_eq!(ctx.decode(&[3]), "");
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

/// A leading standalone word-boundary before byte-fallback bytes should insert
/// one leading space that is then removed by Strip(start=1).
#[test]
fn decode_leading_word_boundary_then_byte_fallback_utf8() {
    let ctx = TokenizerTestContext::from_json(BYTE_FALLBACK_JSON);
    assert_eq!(ctx.decode(&[3, 9, 10]), "é");
}

/// The same Strip behavior must apply to a leading word-boundary before a
/// 4-byte emoji reconstructed from byte-fallback tokens.
#[test]
fn decode_leading_word_boundary_then_byte_fallback_emoji() {
    let ctx = TokenizerTestContext::from_json(BYTE_FALLBACK_JSON);
    assert_eq!(ctx.decode(&[3, 14, 15, 16, 17]), "😊");
}

/// Consecutive standalone word-boundary tokens are not collapsed by the Fuse
/// decoder; only the single leading space is stripped.
#[test]
fn decode_multiple_word_boundaries_preserve_internal_spaces() {
    let ctx = TokenizerTestContext::from_json(BYTE_FALLBACK_JSON);
    assert_eq!(ctx.decode(&[4, 3, 3, 5]), "hello   world");
}

/// A standalone word-boundary between two byte-fallback UTF-8 sequences must
/// decode to a visible space separator between the reconstructed characters.
#[test]
fn decode_word_boundary_between_two_byte_fallback_sequences() {
    let ctx = TokenizerTestContext::from_json(BYTE_FALLBACK_JSON);
    assert_eq!(ctx.decode(&[9, 10, 3, 14, 15, 16, 17]), "é 😊");
}

/// With explicit `skip_special_tokens=1`, special tokens must be removed even
/// when byte-fallback tokens appear between them.
#[test]
fn decode_byte_fallback_skips_special_tokens_when_requested() {
    let ctx = TokenizerTestContext::from_json(BYTE_FALLBACK_JSON);
    let opts = talu_sys::DecodeOptionsC {
        skip_special_tokens: 1,
        ..Default::default()
    };
    assert_eq!(ctx.decode_with(&[1, 9, 10, 2], &opts), "é");
}

/// When special-token skipping is disabled, byte-fallback output must remain
/// correctly interleaved with the retained special token text.
#[test]
fn decode_byte_fallback_retains_special_tokens_when_requested() {
    let ctx = TokenizerTestContext::from_json(BYTE_FALLBACK_JSON);
    let opts = talu_sys::DecodeOptionsC {
        skip_special_tokens: 0,
        ..Default::default()
    };
    assert_eq!(ctx.decode_with(&[1, 9, 10, 2], &opts), "<s>é</s>");
}

/// Skipping special tokens in the byte-fallback decoder should be equivalent
/// to manually filtering those IDs out before decode, even in mixed sequences.
#[test]
fn decode_byte_fallback_skip_special_matches_manual_filtering() {
    let ctx = TokenizerTestContext::from_json(BYTE_FALLBACK_JSON);
    let keep = talu_sys::DecodeOptionsC {
        skip_special_tokens: 0,
        ..Default::default()
    };
    let skip = talu_sys::DecodeOptionsC {
        skip_special_tokens: 1,
        ..Default::default()
    };

    let full = [1, 4, 3, 9, 10, 2, 5];
    let filtered = [4, 3, 9, 10, 5];
    assert_eq!(
        ctx.decode_with(&full, &skip),
        ctx.decode_with(&filtered, &keep),
        "skip_special_tokens must match manual filtering for byte-fallback decode"
    );
}

/// Null decode options must use the C-API default `skip_special_tokens=true`
/// even for byte-fallback decoder pipelines.
#[test]
fn decode_byte_fallback_null_options_defaults_to_skip_special_tokens() {
    let ctx = TokenizerTestContext::from_json(BYTE_FALLBACK_JSON);
    let result = unsafe { super::common::decode_raw_null_options(ctx.handle(), &[1, 4, 3, 9, 10, 2, 5]) };
    assert!(result.error_msg.is_null(), "decode with null options should succeed");
    let text = unsafe {
        let slice = std::slice::from_raw_parts(result.text, result.text_len);
        std::str::from_utf8(slice).expect("decode must return valid UTF-8")
    };
    assert_eq!(text, "hello é world");
    unsafe { talu_sys::talu_decode_result_free(result.text, result.text_len) };
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

/// A truncated 2-byte UTF-8 sequence from byte-fallback must sanitize to a
/// single replacement character rather than returning invalid UTF-8.
#[test]
fn decode_truncated_two_byte_fallback_sequence_sanitizes_to_replacement() {
    let ctx = TokenizerTestContext::from_json(BYTE_FALLBACK_JSON);
    assert_eq!(ctx.decode(&[9]), "\u{FFFD}");
}

/// A truncated 4-byte UTF-8 sequence from byte-fallback must also sanitize to
/// a single replacement character.
#[test]
fn decode_truncated_four_byte_fallback_sequence_sanitizes_to_replacement() {
    let ctx = TokenizerTestContext::from_json(BYTE_FALLBACK_JSON);
    assert_eq!(ctx.decode(&[14, 15, 16]), "\u{FFFD}");
}

/// If a byte-fallback leading byte is followed by a non-continuation byte, the
/// invalid byte must sanitize to U+FFFD and the next ASCII byte must survive.
#[test]
fn decode_invalid_continuation_after_byte_fallback_preserves_following_ascii() {
    let ctx = TokenizerTestContext::from_json(BYTE_FALLBACK_JSON);
    assert_eq!(ctx.decode(&[9, 7]), "\u{FFFD}H");
}

/// Sanitization must also apply in mixed decoder output after normal word
/// decoding and word-boundary replacement.
#[test]
fn decode_regular_then_truncated_byte_fallback_sanitizes_tail() {
    let ctx = TokenizerTestContext::from_json(BYTE_FALLBACK_JSON);
    assert_eq!(ctx.decode(&[4, 3, 9]), "hello \u{FFFD}");
}

/// A standalone invalid continuation byte represented via byte-fallback must
/// sanitize to a single replacement character.
#[test]
fn decode_invalid_continuation_byte_fallback_sanitizes_to_replacement() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "unk_token": "<unk>",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "<0x80>": 3, "<0x48>": 4
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
      {"type": "ByteFallback"},
      {"type": "Fuse"}
    ]
  }
}
"####;
    let ctx = TokenizerTestContext::from_json(json);
    assert_eq!(ctx.decode(&[3]), "\u{FFFD}");
    assert_eq!(ctx.decode(&[4, 3, 4]), "H\u{FFFD}H");
}

/// Empty token array produces empty string.
#[test]
fn decode_empty() {
    let ctx = TokenizerTestContext::from_json(BYTE_FALLBACK_JSON);
    assert_eq!(ctx.decode(&[]), "");
}
