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

// ===========================================================================
// Sequence decoder (Replace + Strip): leading space count
// ===========================================================================
//
// SentencePiece models (Mistral, Llama, etc.) use a Sequence decoder:
//   Replace(▁→space) + ByteFallback + Fuse + Strip(start=1, stop=0)
//
// The Strip decoder removes exactly 1 leading space. The Metaspace
// pretokenizer's add_prefix_space (prepend_scheme="first") applies only
// during encode — it must NOT cause an additional strip during decode.
//
// For a token containing N ▁ characters, the decode result must have N-1
// spaces (N replaced to spaces, then Strip removes 1).
// Affects: mistralai/Mistral-7B-v0.1, v0.3, Instruct variants
// (~56 vocab_decode, 4 roundtrip failures).

/// Minimal SentencePiece BPE with Sequence decoder matching Mistral's config.
const SEQUENCE_STRIP_DECODER_JSON: &str = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "\u2581": 3,
      "\u2581\u2581\u2581\u2581": 4,
      "\u2581Hello": 5, "\u2581world": 6
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Metaspace", "replacement": "\u2581", "prepend_scheme": "first"},
  "post_processor": null,
  "decoder": {
    "type": "Sequence",
    "decoders": [
      {"type": "Replace", "pattern": {"String": "\u2581"}, "content": " "},
      {"type": "ByteFallback"},
      {"type": "Fuse"},
      {"type": "Strip", "content": " ", "start": 1, "stop": 0}
    ]
  }
}"####;

/// ▁▁▁▁ (4 metaspace chars) must decode to "   " (3 spaces).
///
/// Replace converts each ▁ to a space (4 spaces), then Strip removes
/// exactly 1 leading space → 3 spaces. The pretokenizer's add_prefix_space
/// must not cause a second strip.
#[test]
fn sequence_decode_four_metaspace_to_three_spaces() {
    let ctx = TokenizerTestContext::from_json(SEQUENCE_STRIP_DECODER_JSON);
    // Token 4 = ▁▁▁▁ → "    " (4 spaces) → Strip(1) → "   " (3 spaces)
    let decoded = ctx.decode(&[4]);
    assert_eq!(
        decoded, "   ",
        "▁▁▁▁ must decode to 3 spaces (4 replaced, Strip removes 1), got {:?} ({} chars)",
        decoded, decoded.len()
    );
}

/// ▁Hello + ▁world must decode to "Hello world".
///
/// Replace: " Hello" + " world" → Fuse: " Hello world" → Strip(1): "Hello world".
/// The pretokenizer's add_prefix_space must not remove a second space.
#[test]
fn sequence_decode_words_strip_one_leading() {
    let ctx = TokenizerTestContext::from_json(SEQUENCE_STRIP_DECODER_JSON);
    let decoded = ctx.decode(&[5, 6]);
    assert_eq!(
        decoded, "Hello world",
        "▁Hello + ▁world must decode to 'Hello world', got: {decoded:?}"
    );
}

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

// ===========================================================================
// Bug D: JSON \b and \f escapes not decoded to control characters
// ===========================================================================
//
// Tokens in tokenizer.json vocab can contain JSON escape sequences \b
// (backspace U+0008) and \f (form feed U+000C). The JSON parser must
// unescape these to their control character values during loading.
//
// Bug: json_utils.unescapeJsonString doesn't handle \b and \f — they
// fall through to the else branch which keeps the literal backslash.
// Result: token "\\b" in JSON → stored as two chars (\, b) → decoded
// as literal "\b" instead of backspace character.
// Affects: Gemma, Mistral, e5-mistral, phi-1.5, phi-2 (~1530 failures).

/// Minimal SentencePiece BPE with tokens containing \b and \f JSON escapes.
///
/// Token 4 = backspace character (U+0008, from JSON \b escape)
/// Token 5 = form feed character (U+000C, from JSON \f escape)
/// Token 6 = tab character (U+0009, from JSON \t escape — should already work)
const CONTROL_CHAR_TOKENIZER_JSON: &str = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "\u2581": 3,
      "\b": 4, "\f": 5, "\t": 6,
      "\u2581Hello": 7, "H": 8, "e": 9, "l": 10, "o": 11
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

/// JSON \b escape in vocab must decode to backspace (U+0008).
///
/// Bug: unescapeJsonString keeps \b as literal backslash + b.
#[test]
fn decode_json_backspace_escape() {
    let ctx = TokenizerTestContext::from_json(CONTROL_CHAR_TOKENIZER_JSON);
    let decoded = ctx.decode(&[4]);
    assert_eq!(
        decoded, "\u{8}",
        "JSON \\b in vocab must decode to backspace (U+0008), got: {decoded:?}"
    );
}

/// JSON \f escape in vocab must decode to form feed (U+000C).
///
/// Bug: unescapeJsonString keeps \f as literal backslash + f.
#[test]
fn decode_json_formfeed_escape() {
    let ctx = TokenizerTestContext::from_json(CONTROL_CHAR_TOKENIZER_JSON);
    let decoded = ctx.decode(&[5]);
    assert_eq!(
        decoded, "\u{c}",
        "JSON \\f in vocab must decode to form feed (U+000C), got: {decoded:?}"
    );
}

// ===========================================================================
// Added token ID mapping: explicit IDs must be honored
// ===========================================================================
//
// Added tokens in tokenizer.json have explicit "id" fields that may be
// non-sequential or have gaps. The loader must use these explicit IDs, not
// the array position, when building the id→string mapping.
//
// Bug: added tokens are mapped by position rather than explicit ID, so
// decode returns the wrong string for tokens with non-contiguous IDs.
// Affects: microsoft/Phi-4-mini-flash-reasoning (8 vocab_decode failures).

/// Minimal BPE tokenizer with non-contiguous added token IDs.
const ADDED_TOKEN_IDS_JSON: &str = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"hello": 0, "world": 1},
    "merges": []
  },
  "added_tokens": [
    {"id": 5, "content": "<|first|>", "special": true},
    {"id": 10, "content": "<|second|>", "special": true},
    {"id": 15, "content": "<|third|>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;

/// Added token with explicit ID must decode to its own content string.
///
/// Token ID 5 is "<|first|>", ID 10 is "<|second|>", ID 15 is "<|third|>".
/// The decoder must use the explicit IDs, not array position.
#[test]
fn added_token_decode_uses_explicit_id() {
    let ctx = TokenizerTestContext::from_json(ADDED_TOKEN_IDS_JSON);
    assert_eq!(
        ctx.decode(&[5]), "<|first|>",
        "token ID 5 must decode to '<|first|>', got: {:?}",
        ctx.decode(&[5])
    );
    assert_eq!(
        ctx.decode(&[10]), "<|second|>",
        "token ID 10 must decode to '<|second|>', got: {:?}",
        ctx.decode(&[10])
    );
    assert_eq!(
        ctx.decode(&[15]), "<|third|>",
        "token ID 15 must decode to '<|third|>', got: {:?}",
        ctx.decode(&[15])
    );
}
