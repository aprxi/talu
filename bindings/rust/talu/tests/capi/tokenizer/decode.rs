//! Decode functional tests.
//!
//! Validates token-to-text decoding with exact assertions.
//! Uses hardcoded token IDs from the fixture vocab.

use crate::capi::tokenizer::common::TokenizerTestContext;

/// Decoding [44, 77] produces "Hi" (H=44, i=77).
#[test]
fn decode_known_ids() {
    let ctx = TokenizerTestContext::new();
    assert_eq!(ctx.decode(&[44, 77]), "Hi");
}

/// Decoding [44, 73, 80, 80, 83] produces "Hello".
#[test]
fn decode_hello() {
    let ctx = TokenizerTestContext::new();
    assert_eq!(ctx.decode(&[44, 73, 80, 80, 83]), "Hello");
}

/// Decoding a single token: A=37.
#[test]
fn decode_single_token() {
    let ctx = TokenizerTestContext::new();
    assert_eq!(ctx.decode(&[37]), "A");
}

/// Decoding an empty token array produces an empty string.
#[test]
fn decode_empty() {
    let ctx = TokenizerTestContext::new();
    assert_eq!(ctx.decode(&[]), "");
}

/// In the base fixture, special tokens are in both `model.vocab` and
/// `added_tokens`. The BPE decoder checks `id_to_token` first (which
/// always sets `is_special=false`), so `skip_special_tokens` has no
/// observable effect. This locks down that behavior.
#[test]
fn decode_bos_token_base_fixture() {
    let ctx = TokenizerTestContext::new();
    let skip = talu_sys::DecodeOptionsC { skip_special_tokens: 1 };
    let retain = talu_sys::DecodeOptionsC { skip_special_tokens: 0 };

    assert_eq!(ctx.decode_with(&[1], &skip), "<s>");
    assert_eq!(ctx.decode_with(&[1], &retain), "<s>");
}

/// Roundtrip for multiple known strings.
#[test]
fn decode_roundtrip() {
    let ctx = TokenizerTestContext::new();
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };

    for input in ["Hello", "abc123", "A", "!@#$%"] {
        let decoded = ctx.decode(&ctx.encode_with(input, &opts));
        assert_eq!(decoded, input, "roundtrip failed for {input:?}");
    }
}

// ===========================================================================
// skip_special_tokens with special-tokens-only fixture
// ===========================================================================
//
// Uses `with_special_tokens()`: special tokens (IDs 0–3) are ONLY in
// `added_tokens`, not in `model.vocab`. This makes `skip_special_tokens`
// observable — the BPE decoder falls through to `added_tokens` and reads
// the `special` flag.

/// skip_special_tokens=1 strips BOS from decode output.
#[test]
fn skip_special_strips_bos() {
    let ctx = TokenizerTestContext::with_special_tokens();
    let skip = talu_sys::DecodeOptionsC { skip_special_tokens: 1 };
    // [BOS=1, H=44, i=77]
    assert_eq!(ctx.decode_with(&[1, 44, 77], &skip), "Hi");
}

/// skip_special_tokens=0 retains BOS in decode output.
#[test]
fn retain_special_keeps_bos() {
    let ctx = TokenizerTestContext::with_special_tokens();
    let retain = talu_sys::DecodeOptionsC { skip_special_tokens: 0 };
    assert_eq!(ctx.decode_with(&[1, 44, 77], &retain), "<s>Hi");
}

/// skip_special_tokens=1 strips both BOS and EOS.
#[test]
fn skip_special_strips_bos_and_eos() {
    let ctx = TokenizerTestContext::with_special_tokens();
    let skip = talu_sys::DecodeOptionsC { skip_special_tokens: 1 };
    // [BOS=1, H=44, i=77, EOS=2]
    assert_eq!(ctx.decode_with(&[1, 44, 77, 2], &skip), "Hi");
}

/// skip_special_tokens=0 retains both BOS and EOS.
#[test]
fn retain_special_keeps_bos_and_eos() {
    let ctx = TokenizerTestContext::with_special_tokens();
    let retain = talu_sys::DecodeOptionsC { skip_special_tokens: 0 };
    assert_eq!(ctx.decode_with(&[1, 44, 77, 2], &retain), "<s>Hi</s>");
}

/// Decoding only special tokens with skip=1 produces empty string.
#[test]
fn skip_special_all_special_produces_empty() {
    let ctx = TokenizerTestContext::with_special_tokens();
    let skip = talu_sys::DecodeOptionsC { skip_special_tokens: 1 };
    assert_eq!(ctx.decode_with(&[1, 2], &skip), "");
}

/// BOS sandwiched in content is also stripped when skip=1.
#[test]
fn skip_special_strips_sandwiched_bos() {
    let ctx = TokenizerTestContext::with_special_tokens();
    let skip = talu_sys::DecodeOptionsC { skip_special_tokens: 1 };
    // [H=44, BOS=1, i=77]
    assert_eq!(ctx.decode_with(&[44, 1, 77], &skip), "Hi");
}

// ===========================================================================
// Backslash escape handling in ByteLevel BPE decoder
// ===========================================================================
//
// BPE merged tokens can contain literal backslash sequences (e.g. `\n`, `\"`,
// `\\`). The decoder must return these as literal characters — NOT interpret
// them as C-style escape sequences.
//
// Bug: our decoder turns `\n` → newline, `\"` → `"`, `\\` → `\`.
// Affects: Qwen, Llama-3, OpenAI, Mistral, GLM (~70 models).

// ===========================================================================
// SentencePiece BPE: unicode_to_byte must not corrupt non-byte-level tokens
// ===========================================================================
//
// SentencePiece models (Llama, Mistral, Phi, Gemma, TinyLlama) use BPE with
// raw UTF-8 token strings and ▁ for word boundaries. They do NOT use GPT-2's
// byte-to-unicode mapping. The decoder's `unicode_to_byte` map must not be
// applied to these tokens, or non-ASCII chars like é, ü, ñ get mangled.
//
// Bug: the BPE decoder applies `unicode_to_byte` to ALL codepoints, converting
// é (U+00E9) to single byte 0xE9 which is invalid standalone UTF-8 → "�".
// Affects: TinyLlama, Gemma, Phi-3, Mistral, e5-mistral (~100K token failures).

/// Minimal SentencePiece-style BPE with ▁ word boundaries and non-ASCII tokens.
const SENTENCEPIECE_BPE_JSON: &str = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "\u2581": 3,
      "\u2581caf": 4, "é": 5,
      "\u2581r": 6, "\u2581és": 7, "um": 8,
      "\u2581na": 9, "ï": 10, "ve": 11,
      "\u2581Hello": 12, ",": 13, "\u2581world": 14, "!": 15,
      "\u2581": 16
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true},
  "post_processor": null,
  "decoder": {"type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true}
}"####;

/// Decoding tokens with non-ASCII UTF-8 chars (é, ï) must preserve them.
///
/// Bug: unicode_to_byte maps codepoint U+00E9 → byte 0xE9, producing
/// invalid UTF-8 → "�" instead of "é".
#[test]
fn sentencepiece_decode_preserves_accented_chars() {
    let ctx = TokenizerTestContext::from_json(SENTENCEPIECE_BPE_JSON);
    // "▁café" = tokens [4, 5] → "café" (with leading space stripped by Metaspace)
    let decoded = ctx.decode(&[4, 5]);
    assert_eq!(decoded, "café", "SentencePiece BPE must preserve é, got: {decoded:?}");
}

/// Decoding ï (U+00EF) must not be corrupted by byte-level mapping.
#[test]
fn sentencepiece_decode_preserves_diaeresis() {
    let ctx = TokenizerTestContext::from_json(SENTENCEPIECE_BPE_JSON);
    // "▁na" + "ï" + "ve" = tokens [9, 10, 11] → "naïve"
    let decoded = ctx.decode(&[9, 10, 11]);
    assert_eq!(decoded, "naïve", "SentencePiece BPE must preserve ï, got: {decoded:?}");
}

/// Full sentence roundtrip with SentencePiece tokens.
#[test]
fn sentencepiece_decode_full_sentence() {
    let ctx = TokenizerTestContext::from_json(SENTENCEPIECE_BPE_JSON);
    // "▁Hello" + "," + "▁world" + "!" → "Hello, world!"
    let decoded = ctx.decode(&[12, 13, 14, 15]);
    assert_eq!(decoded, "Hello, world!", "SentencePiece decode full sentence, got: {decoded:?}");
}

/// Minimal ByteLevel BPE tokenizer with merged tokens containing backslashes.
///
/// Token 7 = `\n` (literal backslash + n)
/// Token 8 = `\"` (literal backslash + quote)
/// Token 9 = `\\` (two literal backslashes)
/// Token 10 = `\t` (literal backslash + t)
const BACKSLASH_TOKENIZER_JSON: &str = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3,
      "\\": 4, "n": 5, "\"": 6,
      "\\n": 7, "\\\"": 8, "\\\\": 9,
      "t": 10, "\\t": 11
    },
    "merges": ["\\ n", "\\ \"", "\\ \\", "\\ t"]
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<pad>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;

#[test]
fn decode_backslash_n_literal() {
    let ctx = TokenizerTestContext::from_json(BACKSLASH_TOKENIZER_JSON);
    let decoded = ctx.decode(&[7]);
    assert_eq!(decoded, "\\n", "\\n must decode as two literal chars, not newline");
}

#[test]
fn decode_backslash_quote_literal() {
    let ctx = TokenizerTestContext::from_json(BACKSLASH_TOKENIZER_JSON);
    let decoded = ctx.decode(&[8]);
    assert_eq!(decoded, "\\\"", "\\\" must decode as backslash + quote");
}

#[test]
fn decode_double_backslash_literal() {
    let ctx = TokenizerTestContext::from_json(BACKSLASH_TOKENIZER_JSON);
    let decoded = ctx.decode(&[9]);
    assert_eq!(decoded, "\\\\", "\\\\ must decode as two backslashes");
}

#[test]
fn decode_backslash_t_literal() {
    let ctx = TokenizerTestContext::from_json(BACKSLASH_TOKENIZER_JSON);
    let decoded = ctx.decode(&[11]);
    assert_eq!(decoded, "\\t", "\\t must decode as two literal chars, not tab");
}

#[test]
fn decode_backslash_sequence_in_context() {
    let ctx = TokenizerTestContext::from_json(BACKSLASH_TOKENIZER_JSON);
    // "n" + "\n" + "n" — the middle token is a merged backslash-n
    let decoded = ctx.decode(&[5, 7, 5]);
    assert_eq!(decoded, "n\\nn", "backslash tokens must stay literal in sequence");
}
