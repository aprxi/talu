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

/// Decoding an out-of-range token ID must return a typed C-API error.
#[test]
fn decode_invalid_token_id_out_of_vocab_errors() {
    let ctx = TokenizerTestContext::new();
    let opts = talu_sys::DecodeOptionsC::default();
    let result = unsafe { super::common::decode_raw(ctx.handle(), &[999_999], &opts) };
    assert!(
        !result.error_msg.is_null(),
        "out-of-range token ID must return non-null error_msg"
    );
    assert!(
        result.text.is_null(),
        "text pointer must be null on decode error"
    );
    assert_eq!(result.text_len, 0, "text_len must be 0 on decode error");
    let code = unsafe { talu_sys::talu_last_error_code() };
    assert_eq!(
        code,
        talu_sys::ErrorCode::TokenizerInvalidTokenId as i32,
        "invalid token ID decode must set TokenizerInvalidTokenId"
    );
}

/// Decoding u32::MAX must be rejected as invalid token ID.
#[test]
fn decode_invalid_token_id_u32_max_errors() {
    let ctx = TokenizerTestContext::new();
    let opts = talu_sys::DecodeOptionsC::default();
    let result = unsafe { super::common::decode_raw(ctx.handle(), &[u32::MAX], &opts) };
    assert!(
        !result.error_msg.is_null(),
        "u32::MAX token ID must return non-null error_msg"
    );
    assert!(
        result.text.is_null(),
        "text pointer must be null on decode error"
    );
    assert_eq!(result.text_len, 0, "text_len must be 0 on decode error");
    let code = unsafe { talu_sys::talu_last_error_code() };
    assert_eq!(
        code,
        talu_sys::ErrorCode::TokenizerInvalidTokenId as i32,
        "u32::MAX decode must set TokenizerInvalidTokenId"
    );
}

/// decode with null options pointer must use C-API default skip_special_tokens=true.
#[test]
fn decode_null_options_defaults_to_skip_special_tokens() {
    let ctx = TokenizerTestContext::with_special_tokens();
    let result = unsafe { super::common::decode_raw_null_options(ctx.handle(), &[1, 44, 77, 2]) };
    assert!(
        result.error_msg.is_null(),
        "decode with null options should succeed"
    );
    let text = unsafe {
        let slice = std::slice::from_raw_parts(result.text, result.text_len);
        std::str::from_utf8(slice).expect("decode must return valid UTF-8")
    };
    assert_eq!(
        text, "Hi",
        "null decode options must default to skip_special_tokens=true"
    );
    unsafe { talu_sys::talu_decode_result_free(result.text, result.text_len) };
}

/// encode with null options pointer must use C-API default add_special_tokens=true.
#[test]
fn encode_null_options_defaults_to_add_special_tokens() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"H": 4, "i": 5},
    "merges": []
  },
  "added_tokens": [
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": {
    "type": "TemplateProcessing",
    "single": [
      {"SpecialToken": {"id": "<s>", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}},
      {"SpecialToken": {"id": "</s>", "type_id": 0}}
    ],
    "pair": [
      {"SpecialToken": {"id": "<s>", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}},
      {"SpecialToken": {"id": "</s>", "type_id": 0}},
      {"Sequence": {"id": "B", "type_id": 1}},
      {"SpecialToken": {"id": "</s>", "type_id": 0}}
    ],
    "special_tokens": {
      "<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]},
      "</s>": {"id": "</s>", "ids": [2], "tokens": ["</s>"]}
    }
  },
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let result = unsafe { super::common::encode_raw_null_options(ctx.handle(), b"Hi") };
    assert!(
        result.error_msg.is_null(),
        "encode with null options should succeed"
    );
    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    assert_eq!(
        ids,
        &[1, 4, 5, 2],
        "null encode options must default to add_special_tokens=true"
    );
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// In the base fixture, special tokens are in both `model.vocab` and
/// `added_tokens`. `skip_special_tokens=1` must still strip them.
#[test]
fn decode_skip_special_strips_vocab_shadowed_specials() {
    let ctx = TokenizerTestContext::new();
    let skip = talu_sys::DecodeOptionsC {
        skip_special_tokens: 1,
    };
    let retain = talu_sys::DecodeOptionsC {
        skip_special_tokens: 0,
    };

    assert_eq!(
        ctx.decode_with(&[1], &skip),
        "",
        "skip_special_tokens must strip BOS even when ID exists in vocab"
    );
    assert_eq!(
        ctx.decode_with(&[1], &retain),
        "<s>",
        "retain mode must keep BOS token"
    );
}

/// Null decode options must default to skip_special_tokens=true even for vocab-shadowed specials.
#[test]
fn decode_null_options_strip_vocab_shadowed_specials() {
    let ctx = TokenizerTestContext::new();
    let result = unsafe { super::common::decode_raw_null_options(ctx.handle(), &[1, 44, 2]) };
    assert!(
        result.error_msg.is_null(),
        "decode with null options should succeed"
    );
    let text = unsafe {
        let slice = std::slice::from_raw_parts(result.text, result.text_len);
        std::str::from_utf8(slice).expect("decode must return valid UTF-8")
    };
    assert_eq!(
        text, "H",
        "null decode options must strip BOS/EOS by default"
    );
    unsafe { talu_sys::talu_decode_result_free(result.text, result.text_len) };
}

/// Cleanup must apply punctuation-space contractions sequentially across a
/// cascade of punctuation tokens, not stop after the first replacement.
#[test]
fn cleanup_handles_punctuation_cascade_sequence() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 200,
    "vocab": {
      "[UNK]": 0, "hello": 1, ",": 2, ".": 3, "?": 4, "!": 5
    }
  },
  "added_tokens": [{"id": 0, "content": "[UNK]", "special": true}],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let decoded = ctx.decode(&[1, 2, 3, 4, 5]);
    assert_eq!(
        decoded, "hello,.?!",
        "cleanup must handle punctuation cascades in sequence"
    );
}

fn hf_cleanup_reference(text: &str) -> String {
    text.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
}

/// Cleanup behavior must match the exact sequential HF-style replacement chain
/// over dense mixed punctuation/contraction token permutations.
#[test]
fn cleanup_matches_reference_replace_chain_on_dense_token_permutations() {
    let json_cleanup_false = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 200,
    "vocab": {
      "[UNK]": 0,
      "foo": 1, "bar": 2,
      ".": 3, ",": 4, "?": 5, "!": 6,
      "'": 7, "n": 8, "t": 9, "m": 10, "s": 11, "ve": 12, "re": 13
    }
  },
  "added_tokens": [{"id": 0, "content": "[UNK]", "special": true}],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false}
}"####;
    let json_cleanup_true = json_cleanup_false.replace("\"cleanup\": false", "\"cleanup\": true");
    let raw_ctx = TokenizerTestContext::from_json(json_cleanup_false);
    let clean_ctx = TokenizerTestContext::from_json(&json_cleanup_true);

    let pool = [1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
    let mut seed: u64 = 0xC0FFEE_F00DBABE;
    for case_idx in 0..256usize {
        let len = 4 + ((seed as usize) % 10);
        let mut ids = Vec::with_capacity(len);
        for _ in 0..len {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let idx = ((seed >> 32) as usize) % pool.len();
            ids.push(pool[idx]);
        }

        let raw = raw_ctx.decode(&ids);
        let cleaned = clean_ctx.decode(&ids);
        let expected = hf_cleanup_reference(&raw);
        assert_eq!(
            cleaned, expected,
            "cleanup chain mismatch at permutation case {case_idx}: ids={ids:?} raw={raw:?}"
        );
    }
}

/// `skip_special_tokens` must filter by special-token IDs, not by token text.
/// A regular vocab token with the same text as a special token must be kept.
#[test]
fn skip_special_filters_by_id_not_token_string() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"<unk>": 0, "<s>": 1, "H": 2},
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 99, "content": "<s>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let skip = talu_sys::DecodeOptionsC {
        skip_special_tokens: 1,
    };
    let keep = talu_sys::DecodeOptionsC {
        skip_special_tokens: 0,
    };

    assert_eq!(
        ctx.decode_with(&[1, 2, 99, 2], &keep),
        "<s>H<s>H",
        "retain mode must decode both regular and special <s> token IDs"
    );
    assert_eq!(
        ctx.decode_with(&[1, 2, 99, 2], &skip),
        "<s>HH",
        "skip mode must strip only special ID 99 while retaining vocab ID 1"
    );
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
    let skip = talu_sys::DecodeOptionsC {
        skip_special_tokens: 1,
    };
    // [BOS=1, H=44, i=77]
    assert_eq!(ctx.decode_with(&[1, 44, 77], &skip), "Hi");
}

/// skip_special_tokens=0 retains BOS in decode output.
#[test]
fn retain_special_keeps_bos() {
    let ctx = TokenizerTestContext::with_special_tokens();
    let retain = talu_sys::DecodeOptionsC {
        skip_special_tokens: 0,
    };
    assert_eq!(ctx.decode_with(&[1, 44, 77], &retain), "<s>Hi");
}

/// skip_special_tokens=1 strips both BOS and EOS.
#[test]
fn skip_special_strips_bos_and_eos() {
    let ctx = TokenizerTestContext::with_special_tokens();
    let skip = talu_sys::DecodeOptionsC {
        skip_special_tokens: 1,
    };
    // [BOS=1, H=44, i=77, EOS=2]
    assert_eq!(ctx.decode_with(&[1, 44, 77, 2], &skip), "Hi");
}

/// skip_special_tokens=0 retains both BOS and EOS.
#[test]
fn retain_special_keeps_bos_and_eos() {
    let ctx = TokenizerTestContext::with_special_tokens();
    let retain = talu_sys::DecodeOptionsC {
        skip_special_tokens: 0,
    };
    assert_eq!(ctx.decode_with(&[1, 44, 77, 2], &retain), "<s>Hi</s>");
}

/// Decoding only special tokens with skip=1 produces empty string.
#[test]
fn skip_special_all_special_produces_empty() {
    let ctx = TokenizerTestContext::with_special_tokens();
    let skip = talu_sys::DecodeOptionsC {
        skip_special_tokens: 1,
    };
    assert_eq!(ctx.decode_with(&[1, 2], &skip), "");
}

/// BOS sandwiched in content is also stripped when skip=1.
#[test]
fn skip_special_strips_sandwiched_bos() {
    let ctx = TokenizerTestContext::with_special_tokens();
    let skip = talu_sys::DecodeOptionsC {
        skip_special_tokens: 1,
    };
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
// The decoder must preserve literal backslash sequences in token content.

// ===========================================================================
// SentencePiece BPE: unicode_to_byte must not corrupt non-byte-level tokens
// ===========================================================================
//
// SentencePiece BPE models use raw UTF-8 token strings and ▁ for word
// boundaries. They do NOT use GPT-2's byte-to-unicode mapping. The decoder's
// `unicode_to_byte` map must not be applied to these tokens, or non-ASCII
// chars like é, ü, ñ get mangled into invalid UTF-8.

// ===========================================================================
// Sequence decoder (Replace + Strip): leading space count
// ===========================================================================
//
// SentencePiece Sequence decoder: Replace(▁→space) + ByteFallback + Fuse +
// Strip(start=1, stop=0). Strip removes exactly 1 leading space. For a token
// containing N ▁ characters, the decode result must have N-1 spaces.

/// Minimal SentencePiece BPE with Sequence decoder (Replace + ByteFallback + Fuse + Strip).
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

fn nested_sequence_strip_decoder_json() -> String {
    SEQUENCE_STRIP_DECODER_JSON.replace(
        r#""decoder": {
    "type": "Sequence",
    "decoders": [
      {"type": "Replace", "pattern": {"String": "\u2581"}, "content": " "},
      {"type": "ByteFallback"},
      {"type": "Fuse"},
      {"type": "Strip", "content": " ", "start": 1, "stop": 0}
    ]
  }"#,
        r#""decoder": {
    "type": "Sequence",
    "decoders": [
      {
        "type": "Sequence",
        "decoders": [
          {"type": "Replace", "pattern": {"String": "\u2581"}, "content": " "},
          {"type": "ByteFallback"},
          {"type": "Fuse"},
          {"type": "Strip", "content": " ", "start": 1, "stop": 0}
        ]
      }
    ]
  }"#,
    )
}

fn nested_metaspace_decoder_json() -> String {
    SENTENCEPIECE_BPE_JSON.replace(
        r#""decoder": {"type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true}"#,
        r#""decoder": {
    "type": "Sequence",
    "decoders": [
      {"type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true}
    ]
  }"#,
    )
}

fn doubly_nested_metaspace_decoder_json() -> String {
    SENTENCEPIECE_BPE_JSON.replace(
        r#""decoder": {"type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true}"#,
        r#""decoder": {
    "type": "Sequence",
    "decoders": [
      {
        "type": "Sequence",
        "decoders": [
          {"type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true}
        ]
      }
    ]
  }"#,
    )
}

fn doubly_nested_metaspace_special_decoder_json() -> String {
    METASPACE_SPECIAL_TOKEN_JSON.replace(
        r#""decoder": {"type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true}"#,
        r#""decoder": {
    "type": "Sequence",
    "decoders": [
      {
        "type": "Sequence",
        "decoders": [
          {"type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true}
        ]
      }
    ]
  }"#,
    )
}

fn flat_metaspace_special_no_prefix_json() -> &'static str {
    r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "\u2581": 3,
      "\u2581and": 4, "\u2581are": 5, "\u2581special": 6, "\u2581tokens": 7
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
  "decoder": {"type": "Metaspace", "replacement": "\u2581", "add_prefix_space": false}
}"####
}

fn doubly_nested_metaspace_special_no_prefix_json() -> String {
    flat_metaspace_special_no_prefix_json().replace(
        r#""decoder": {"type": "Metaspace", "replacement": "\u2581", "add_prefix_space": false}"#,
        r#""decoder": {
    "type": "Sequence",
    "decoders": [
      {
        "type": "Sequence",
        "decoders": [
          {"type": "Metaspace", "replacement": "\u2581", "add_prefix_space": false}
        ]
      }
    ]
  }"#,
    )
}

fn outer_sequence_with_nested_replace_json() -> String {
    SEQUENCE_STRIP_DECODER_JSON.replace(
        r#"{"type": "Replace", "pattern": {"String": "\u2581"}, "content": " "}"#,
        r#"{
        "type": "Sequence",
        "decoders": [
          {"type": "Replace", "pattern": {"String": "\u2581"}, "content": " "}
        ]
      }"#,
    )
}

fn outer_sequence_with_nested_strip_json() -> String {
    SEQUENCE_STRIP_DECODER_JSON.replace(
        r#"{"type": "Strip", "content": " ", "start": 1, "stop": 0}"#,
        r#"{
        "type": "Sequence",
        "decoders": [
          {"type": "Strip", "content": " ", "start": 1, "stop": 0}
        ]
      }"#,
    )
}

const REPLACE_ONLY_DECODER_JSON: &str = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0,
      "\u2581Hello": 1
    },
    "merges": []
  },
  "added_tokens": [{"id": 0, "content": "<unk>", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": {
    "type": "Sequence",
    "decoders": [
      {"type": "Replace", "pattern": {"String": "\u2581"}, "content": " "}
    ]
  }
}"####;

fn nested_replace_only_decoder_json() -> String {
    REPLACE_ONLY_DECODER_JSON.replace(
        r#""decoder": {
    "type": "Sequence",
    "decoders": [
      {"type": "Replace", "pattern": {"String": "\u2581"}, "content": " "}
    ]
  }"#,
        r#""decoder": {
    "type": "Sequence",
    "decoders": [
      {
        "type": "Sequence",
        "decoders": [
          {"type": "Replace", "pattern": {"String": "\u2581"}, "content": " "}
        ]
      }
    ]
  }"#,
    )
}

const STRIP_ONLY_DECODER_JSON: &str = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0,
      " hello": 1
    },
    "merges": []
  },
  "added_tokens": [{"id": 0, "content": "<unk>", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": {"type": "Strip", "content": " ", "start": 1, "stop": 0}
}"####;

fn nested_strip_only_decoder_json() -> String {
    STRIP_ONLY_DECODER_JSON.replace(
        r#""decoder": {"type": "Strip", "content": " ", "start": 1, "stop": 0}"#,
        r#""decoder": {
    "type": "Sequence",
    "decoders": [
      {
        "type": "Sequence",
        "decoders": [
          {"type": "Strip", "content": " ", "start": 1, "stop": 0}
        ]
      }
    ]
  }"#,
    )
}

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
        decoded,
        "   ",
        "▁▁▁▁ must decode to 3 spaces (4 replaced, Strip removes 1), got {:?} ({} chars)",
        decoded,
        decoded.len()
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

/// A nested Sequence decoder wrapper must preserve the exact runtime behavior
/// of the equivalent flat Replace+ByteFallback+Fuse+Strip chain.
#[test]
fn nested_sequence_decoder_matches_flat_sequence_behavior() {
    let flat = TokenizerTestContext::from_json(SEQUENCE_STRIP_DECODER_JSON);
    let nested = TokenizerTestContext::from_json(&nested_sequence_strip_decoder_json());

    for ids in [&[4][..], &[5, 6][..]] {
        assert_eq!(
            nested.decode(ids),
            flat.decode(ids),
            "nested Sequence decoder must behave exactly like the equivalent flat chain for ids {ids:?}"
        );
    }
}

/// Nested Replace-only decoders must behave exactly like the flat Replace
/// chain. This isolates whether nested decoder traversal applies Replace at
/// all, independent of ByteFallback/Fuse/Strip.
#[test]
fn nested_replace_only_decoder_matches_flat_behavior() {
    let flat = TokenizerTestContext::from_json(REPLACE_ONLY_DECODER_JSON);
    let nested = TokenizerTestContext::from_json(&nested_replace_only_decoder_json());
    assert_eq!(
        nested.decode(&[1]),
        flat.decode(&[1]),
        "nested Replace-only decoder must match flat behavior"
    );
}

/// Nested Strip-only decoders must behave exactly like the flat Strip decoder.
#[test]
fn nested_strip_only_decoder_matches_flat_behavior() {
    let flat = TokenizerTestContext::from_json(STRIP_ONLY_DECODER_JSON);
    let nested = TokenizerTestContext::from_json(&nested_strip_only_decoder_json());
    assert_eq!(
        nested.decode(&[1]),
        flat.decode(&[1]),
        "nested Strip-only decoder must match flat behavior"
    );
}

/// Strip(start=1, content=" ") must not remove unrelated leading characters.
#[test]
fn decoder_strip_start_does_not_remove_non_matching_characters() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0,
      " HelloWorld": 1,
      "HelloWorld": 2
    },
    "merges": []
  },
  "added_tokens": [{"id": 0, "content": "<unk>", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": {"type": "Strip", "content": " ", "start": 1, "stop": 0}
}"####;
    let ctx = TokenizerTestContext::from_json(json);

    assert_eq!(
        ctx.decode(&[1]),
        "HelloWorld",
        "matching leading strip content must be removed exactly once"
    );
    assert_eq!(
        ctx.decode(&[2]),
        "HelloWorld",
        "non-matching leading character must not be removed by Strip(start=1)"
    );
}

fn run_decoder_strip_out_of_bounds_start_stop_inner() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0,
      "   ": 1,
      "Hello": 2
    },
    "merges": []
  },
  "added_tokens": [{"id": 0, "content": "<unk>", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": {"type": "Strip", "content": " ", "start": 100, "stop": 100}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let result =
        unsafe { super::common::decode_raw(ctx.handle(), &[1], &talu_sys::DecodeOptionsC::default()) };
    assert!(
        result.error_msg.is_null(),
        "strip decoder with out-of-bounds bounds must not error"
    );
    let text = if result.text.is_null() || result.text_len == 0 {
        ""
    } else {
        let slice = unsafe { std::slice::from_raw_parts(result.text, result.text_len) };
        std::str::from_utf8(slice).expect("decode must return valid UTF-8")
    };
    assert_eq!(
        text, "",
        "Strip(start=100, stop=100) over short matching content must clamp to empty"
    );
    unsafe { talu_sys::talu_decode_result_free(result.text, result.text_len) };

    let non_match =
        unsafe { super::common::decode_raw(ctx.handle(), &[2], &talu_sys::DecodeOptionsC::default()) };
    assert!(
        non_match.error_msg.is_null(),
        "strip decoder must not error on non-matching leading content"
    );
    let non_match_text = if non_match.text.is_null() || non_match.text_len == 0 {
        ""
    } else {
        let slice = unsafe { std::slice::from_raw_parts(non_match.text, non_match.text_len) };
        std::str::from_utf8(slice).expect("decode must return valid UTF-8")
    };
    assert_eq!(
        non_match_text, "Hello",
        "non-matching content must remain unchanged even with oversized start/stop"
    );
    unsafe { talu_sys::talu_decode_result_free(non_match.text, non_match.text_len) };
}

/// Strip decoder bounds far outside the decoded string must clamp safely and
/// never crash (subprocess-isolated to treat native aborts as test failures).
#[test]
fn decoder_strip_out_of_bounds_start_stop_clamps_safely() {
    const INNER_ENV: &str = "TALU_INNER_STRIP_OOB_CLAMP";
    if std::env::var_os(INNER_ENV).is_some() {
        run_decoder_strip_out_of_bounds_start_stop_inner();
        return;
    }

    let exe = std::env::current_exe().expect("current test executable path must resolve");
    let output = std::process::Command::new(exe)
        .arg("--exact")
        .arg("capi::tokenizer::decode::decoder_strip_out_of_bounds_start_stop_clamps_safely")
        .arg("--nocapture")
        .env(INNER_ENV, "1")
        .output()
        .expect("subprocess launch for strip out-of-bounds clamp test must succeed");

    assert!(
        output.status.success(),
        "strip out-of-bounds clamp subprocess failed (status: {:?})\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

/// A nested Replace stage inside an otherwise flat outer Sequence must behave
/// exactly like the flat chain.
#[test]
fn outer_sequence_with_nested_replace_matches_flat_behavior() {
    let flat = TokenizerTestContext::from_json(SEQUENCE_STRIP_DECODER_JSON);
    let nested = TokenizerTestContext::from_json(&outer_sequence_with_nested_replace_json());

    for ids in [&[4][..], &[5, 6][..]] {
        assert_eq!(
            nested.decode(ids),
            flat.decode(ids),
            "outer Sequence with nested Replace stage must match flat behavior for ids {ids:?}"
        );
    }
}

/// A nested Strip stage inside an otherwise flat outer Sequence must behave
/// exactly like the flat chain.
#[test]
fn outer_sequence_with_nested_strip_matches_flat_behavior() {
    let flat = TokenizerTestContext::from_json(SEQUENCE_STRIP_DECODER_JSON);
    let nested = TokenizerTestContext::from_json(&outer_sequence_with_nested_strip_json());

    for ids in [&[4][..], &[5, 6][..]] {
        assert_eq!(
            nested.decode(ids),
            flat.decode(ids),
            "outer Sequence with nested Strip stage must match flat behavior for ids {ids:?}"
        );
    }
}

/// Minimal SentencePiece-style BPE with unique vocab keys.
///
/// JSON object keys must be unique. This fixture intentionally keeps only one
/// standalone `▁` entry so loader tests are checking tokenizer behavior rather
/// than ambiguous duplicate-key parsing.
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
      "\u2581Hello": 12, ",": 13, "\u2581world": 14, "!": 15
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
/// SentencePiece BPE tokens contain raw UTF-8 (not byte-level mapped).
/// The unicode_to_byte map must not be applied, or multi-byte chars get
/// corrupted into invalid UTF-8.
#[test]
fn sentencepiece_decode_preserves_accented_chars() {
    let ctx = TokenizerTestContext::from_json(SENTENCEPIECE_BPE_JSON);
    // "▁café" = tokens [4, 5] → "café" (with leading space stripped by Metaspace)
    let decoded = ctx.decode(&[4, 5]);
    assert_eq!(
        decoded, "café",
        "SentencePiece BPE must preserve é, got: {decoded:?}"
    );
}

/// Wrapping a Metaspace decoder in a Sequence must not change runtime decode
/// behavior for the same token stream.
#[test]
fn nested_metaspace_decoder_matches_flat_behavior() {
    let flat = TokenizerTestContext::from_json(SENTENCEPIECE_BPE_JSON);
    let nested = TokenizerTestContext::from_json(&nested_metaspace_decoder_json());

    for ids in [&[4, 5][..], &[12, 13, 14, 15][..]] {
        assert_eq!(
            nested.decode(ids),
            flat.decode(ids),
            "nested Metaspace decoder must behave exactly like the flat decoder for ids {ids:?}"
        );
    }
}

/// A root nested Metaspace subtree must also preserve the exact runtime
/// behavior of the flat decoder.
#[test]
fn doubly_nested_metaspace_decoder_matches_flat_behavior() {
    let flat = TokenizerTestContext::from_json(SENTENCEPIECE_BPE_JSON);
    let nested = TokenizerTestContext::from_json(&doubly_nested_metaspace_decoder_json());

    for ids in [&[4, 5][..], &[12, 13, 14, 15][..]] {
        assert_eq!(
            nested.decode(ids),
            flat.decode(ids),
            "doubly nested Metaspace decoder must behave exactly like the flat decoder for ids {ids:?}"
        );
    }
}

/// Root nested Metaspace must also preserve the non-default
/// `add_prefix_space=false` behavior of the flat decoder.
#[test]
fn doubly_nested_metaspace_no_prefix_matches_flat_behavior() {
    let flat_json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0,
      "\u2581hello": 1
    },
    "merges": []
  },
  "added_tokens": [{"id": 0, "content": "<unk>", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": {"type": "Metaspace", "replacement": "\u2581", "add_prefix_space": false}
}"####;
    let nested_json = flat_json.replace(
        r#""decoder": {"type": "Metaspace", "replacement": "\u2581", "add_prefix_space": false}"#,
        r#""decoder": {
    "type": "Sequence",
    "decoders": [
      {
        "type": "Sequence",
        "decoders": [
          {"type": "Metaspace", "replacement": "\u2581", "add_prefix_space": false}
        ]
      }
    ]
  }"#,
    );

    let flat = TokenizerTestContext::from_json(flat_json);
    let nested = TokenizerTestContext::from_json(&nested_json);
    assert_eq!(
        nested.decode(&[1]),
        flat.decode(&[1]),
        "doubly nested Metaspace with add_prefix_space=false must match flat behavior"
    );
}

/// A flat Metaspace decoder followed by a nested Strip stage must behave like
/// the equivalent flat Sequence(Metaspace, Strip) chain.
#[test]
fn metaspace_then_nested_strip_matches_flat_behavior() {
    let flat_json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0,
      "\u2581caf": 1, "é": 2
    },
    "merges": []
  },
  "added_tokens": [{"id": 0, "content": "<unk>", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": {
    "type": "Sequence",
    "decoders": [
      {"type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true},
      {"type": "Strip", "content": " ", "start": 1, "stop": 0}
    ]
  }
}"####;
    let nested_json = flat_json.replace(
        r#"{"type": "Strip", "content": " ", "start": 1, "stop": 0}"#,
        r#"{
        "type": "Sequence",
        "decoders": [
          {"type": "Strip", "content": " ", "start": 1, "stop": 0}
        ]
      }"#,
    );

    let flat = TokenizerTestContext::from_json(flat_json);
    let nested = TokenizerTestContext::from_json(&nested_json);
    assert_eq!(
        nested.decode(&[1, 2]),
        flat.decode(&[1, 2]),
        "Metaspace followed by nested Strip must match the flat decoder chain"
    );
}

/// A flat Metaspace decoder followed by a nested Replace stage must behave
/// like the equivalent flat Sequence(Metaspace, Replace) chain.
#[test]
fn metaspace_then_nested_replace_matches_flat_behavior() {
    let flat_json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0,
      "\u2581caf": 1, "é": 2
    },
    "merges": []
  },
  "added_tokens": [{"id": 0, "content": "<unk>", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": {
    "type": "Sequence",
    "decoders": [
      {"type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true},
      {"type": "Replace", "pattern": {"String": " "}, "content": "_"}
    ]
  }
}"####;
    let nested_json = flat_json.replace(
        r#"{"type": "Replace", "pattern": {"String": " "}, "content": "_"}"#,
        r#"{
        "type": "Sequence",
        "decoders": [
          {"type": "Replace", "pattern": {"String": " "}, "content": "_"}
        ]
      }"#,
    );

    let flat = TokenizerTestContext::from_json(flat_json);
    let nested = TokenizerTestContext::from_json(&nested_json);
    assert_eq!(
        nested.decode(&[1, 2]),
        flat.decode(&[1, 2]),
        "Metaspace followed by nested Replace must match the flat decoder chain"
    );
}

/// A flat WordPiece decoder with `cleanup=false` followed by a nested Strip
/// stage must preserve the same runtime behavior as the flat chain.
#[test]
fn wordpiece_then_nested_strip_matches_flat_behavior() {
    let flat_json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "vocab": {
      "[UNK]": 0,
      " hello": 1
    }
  },
  "added_tokens": [{"id": 0, "content": "[UNK]", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": {
    "type": "Sequence",
    "decoders": [
      {"type": "WordPiece", "prefix": "##", "cleanup": false},
      {"type": "Strip", "content": " ", "start": 1, "stop": 0}
    ]
  }
}"####;
    let nested_json = flat_json.replace(
        r#"{"type": "Strip", "content": " ", "start": 1, "stop": 0}"#,
        r#"{
        "type": "Sequence",
        "decoders": [
          {"type": "Strip", "content": " ", "start": 1, "stop": 0}
        ]
      }"#,
    );

    let flat = TokenizerTestContext::from_json(flat_json);
    let nested = TokenizerTestContext::from_json(&nested_json);
    assert_eq!(
        nested.decode(&[1]),
        flat.decode(&[1]),
        "WordPiece followed by nested Strip must match the flat decoder chain"
    );
}

/// A flat WordPiece decoder with `cleanup=false` followed by a nested Replace
/// stage must preserve the same runtime behavior as the flat chain.
#[test]
fn wordpiece_then_nested_replace_matches_flat_behavior() {
    let flat_json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "vocab": {
      "[UNK]": 0,
      "i": 1, "'m": 2
    }
  },
  "added_tokens": [{"id": 0, "content": "[UNK]", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": {
    "type": "Sequence",
    "decoders": [
      {"type": "WordPiece", "prefix": "##", "cleanup": false},
      {"type": "Replace", "pattern": {"String": " "}, "content": "_"}
    ]
  }
}"####;
    let nested_json = flat_json.replace(
        r#"{"type": "Replace", "pattern": {"String": " "}, "content": "_"}"#,
        r#"{
        "type": "Sequence",
        "decoders": [
          {"type": "Replace", "pattern": {"String": " "}, "content": "_"}
        ]
      }"#,
    );

    let flat = TokenizerTestContext::from_json(flat_json);
    let nested = TokenizerTestContext::from_json(&nested_json);
    assert_eq!(
        nested.decode(&[1, 2]),
        flat.decode(&[1, 2]),
        "WordPiece followed by nested Replace must match the flat decoder chain"
    );
}

/// Decoding ï (U+00EF) must not be corrupted by byte-level mapping.
#[test]
fn sentencepiece_decode_preserves_diaeresis() {
    let ctx = TokenizerTestContext::from_json(SENTENCEPIECE_BPE_JSON);
    // "▁na" + "ï" + "ve" = tokens [9, 10, 11] → "naïve"
    let decoded = ctx.decode(&[9, 10, 11]);
    assert_eq!(
        decoded, "naïve",
        "SentencePiece BPE must preserve ï, got: {decoded:?}"
    );
}

/// Full sentence roundtrip with SentencePiece tokens.
#[test]
fn sentencepiece_decode_full_sentence() {
    let ctx = TokenizerTestContext::from_json(SENTENCEPIECE_BPE_JSON);
    // "▁Hello" + "," + "▁world" + "!" → "Hello, world!"
    let decoded = ctx.decode(&[12, 13, 14, 15]);
    assert_eq!(
        decoded, "Hello, world!",
        "SentencePiece decode full sentence, got: {decoded:?}"
    );
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
    assert_eq!(
        decoded, "\\n",
        "\\n must decode as two literal chars, not newline"
    );
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
    assert_eq!(
        decoded, "\\t",
        "\\t must decode as two literal chars, not tab"
    );
}

#[test]
fn decode_backslash_sequence_in_context() {
    let ctx = TokenizerTestContext::from_json(BACKSLASH_TOKENIZER_JSON);
    // "n" + "\n" + "n" — the middle token is a merged backslash-n
    let decoded = ctx.decode(&[5, 7, 5]);
    assert_eq!(
        decoded, "n\\nn",
        "backslash tokens must stay literal in sequence"
    );
}

// ===========================================================================
// JSON \b and \f escapes must be decoded to control characters
// ===========================================================================
//
// Tokens in tokenizer.json vocab can contain JSON escape sequences \b
// (backspace U+0008) and \f (form feed U+000C). The JSON parser must
// unescape these to their control character values during loading.

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
        ctx.decode(&[5]),
        "<|first|>",
        "token ID 5 must decode to '<|first|>', got: {:?}",
        ctx.decode(&[5])
    );
    assert_eq!(
        ctx.decode(&[10]),
        "<|second|>",
        "token ID 10 must decode to '<|second|>', got: {:?}",
        ctx.decode(&[10])
    );
    assert_eq!(
        ctx.decode(&[15]),
        "<|third|>",
        "token ID 15 must decode to '<|third|>', got: {:?}",
        ctx.decode(&[15])
    );
}

// ===========================================================================
// Metaspace BPE: special token matching preserves word boundaries
// ===========================================================================
//
// When special tokens appear in the input, the text around them is segmented
// and each segment goes through Metaspace pretokenization independently.
// Trailing spaces before special tokens produce a standalone ▁ token (correct).

/// Minimal SentencePiece BPE with <s>/<eos> special tokens and Metaspace.
const METASPACE_SPECIAL_TOKEN_JSON: &str = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "\u2581": 3,
      "\u2581and": 4, "\u2581are": 5, "\u2581special": 6, "\u2581tokens": 7
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Metaspace", "replacement": "\u2581", "prepend_scheme": "first", "add_prefix_space": true},
  "post_processor": null,
  "decoder": {"type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true}
}"####;

/// Metaspace encodes trailing space before special token as standalone ▁.
///
/// Input "<s> and </s> are special tokens" splits into segments:
/// - <s> matched as special token (ID 1)
/// - " and " → Metaspace → [▁and, ▁] (trailing space becomes standalone ▁)
/// - </s> matched as special token (ID 2)
/// - " are special tokens" → Metaspace → [▁are, ▁special, ▁tokens]
#[test]
fn metaspace_encode_with_special_tokens() {
    let ctx = TokenizerTestContext::from_json(METASPACE_SPECIAL_TOKEN_JSON);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    let tokens = ctx.encode_with("<s> and </s> are special tokens", &opts);
    assert_eq!(
        tokens,
        vec![1, 4, 3, 2, 5, 6, 7],
        "trailing space before </s> produces standalone ▁ token, got: {tokens:?}"
    );
}

/// The Metaspace decoder must roundtrip the same special-token boundary shape
/// that the encoder emits: special literals retained, internal spaces restored,
/// and the standalone ▁ before </s> becoming a real trailing space.
#[test]
fn metaspace_decode_with_special_tokens_roundtrips_exact_text() {
    let ctx = TokenizerTestContext::from_json(METASPACE_SPECIAL_TOKEN_JSON);
    let retain = talu_sys::DecodeOptionsC {
        skip_special_tokens: 0,
    };
    let decoded = ctx.decode_with(&[1, 4, 3, 2, 5, 6, 7], &retain);
    assert_eq!(
        decoded, "<s> and </s> are special tokens",
        "Metaspace decode must restore spaces exactly across retained special-token boundaries, got: {decoded:?}"
    );
}

/// Root nested Metaspace must preserve the same retained-special-token decode
/// behavior as the flat decoder on special-token boundary text.
#[test]
fn doubly_nested_metaspace_with_special_tokens_matches_flat_behavior() {
    let flat = TokenizerTestContext::from_json(METASPACE_SPECIAL_TOKEN_JSON);
    let nested = TokenizerTestContext::from_json(&doubly_nested_metaspace_special_decoder_json());
    let retain = talu_sys::DecodeOptionsC {
        skip_special_tokens: 0,
    };
    assert_eq!(
        nested.decode_with(&[1, 4, 3, 2, 5, 6, 7], &retain),
        flat.decode_with(&[1, 4, 3, 2, 5, 6, 7], &retain),
        "doubly nested Metaspace must match flat retained-special behavior across special-token boundaries"
    );
}

/// Root nested Metaspace must also preserve special-token boundary behavior on
/// the non-default `add_prefix_space=false` branch.
#[test]
fn doubly_nested_metaspace_no_prefix_with_special_tokens_matches_flat_behavior() {
    let flat = TokenizerTestContext::from_json(flat_metaspace_special_no_prefix_json());
    let nested = TokenizerTestContext::from_json(&doubly_nested_metaspace_special_no_prefix_json());
    let retain = talu_sys::DecodeOptionsC {
        skip_special_tokens: 0,
    };
    assert_eq!(
        nested.decode_with(&[1, 4, 3, 2, 5, 6, 7], &retain),
        flat.decode_with(&[1, 4, 3, 2, 5, 6, 7], &retain),
        "doubly nested Metaspace add_prefix_space=false must match flat retained-special behavior"
    );
}

/// Root nested Metaspace must also preserve skip-special behavior on the
/// non-default `add_prefix_space=false` branch.
#[test]
fn doubly_nested_metaspace_no_prefix_with_special_tokens_skip_matches_flat_behavior() {
    let flat = TokenizerTestContext::from_json(flat_metaspace_special_no_prefix_json());
    let nested = TokenizerTestContext::from_json(&doubly_nested_metaspace_special_no_prefix_json());
    let skip = talu_sys::DecodeOptionsC {
        skip_special_tokens: 1,
    };
    assert_eq!(
        nested.decode_with(&[1, 4, 3, 2, 5, 6, 7], &skip),
        flat.decode_with(&[1, 4, 3, 2, 5, 6, 7], &skip),
        "doubly nested Metaspace add_prefix_space=false must match flat skip-special behavior"
    );
}

/// Null decode options must preserve the same default skip-special behavior on
/// the root nested Metaspace `add_prefix_space=false` branch as on the flat decoder.
#[test]
fn doubly_nested_metaspace_no_prefix_with_special_tokens_null_options_matches_flat_behavior() {
    let flat = TokenizerTestContext::from_json(flat_metaspace_special_no_prefix_json());
    let nested = TokenizerTestContext::from_json(&doubly_nested_metaspace_special_no_prefix_json());
    let ids = [1, 4, 3, 2, 5, 6, 7];

    let flat_result = unsafe { super::common::decode_raw_null_options(flat.handle(), &ids) };
    assert!(
        flat_result.error_msg.is_null(),
        "flat decode with null options should succeed"
    );
    let flat_text = unsafe {
        let slice = std::slice::from_raw_parts(flat_result.text, flat_result.text_len);
        std::str::from_utf8(slice)
            .expect("flat decode must return valid UTF-8")
            .to_owned()
    };
    unsafe { talu_sys::talu_decode_result_free(flat_result.text, flat_result.text_len) };

    let nested_result = unsafe { super::common::decode_raw_null_options(nested.handle(), &ids) };
    assert!(
        nested_result.error_msg.is_null(),
        "nested decode with null options should succeed"
    );
    let nested_text = unsafe {
        let slice = std::slice::from_raw_parts(nested_result.text, nested_result.text_len);
        std::str::from_utf8(slice)
            .expect("nested decode must return valid UTF-8")
            .to_owned()
    };
    unsafe { talu_sys::talu_decode_result_free(nested_result.text, nested_result.text_len) };

    assert_eq!(
        nested_text, flat_text,
        "doubly nested Metaspace add_prefix_space=false must match flat null-options decode behavior"
    );
}

/// Root nested Metaspace must also preserve skip-special behavior across
/// special-token boundaries on the `add_prefix_space=true` branch.
#[test]
fn doubly_nested_metaspace_with_special_tokens_skip_matches_flat_behavior() {
    let flat = TokenizerTestContext::from_json(METASPACE_SPECIAL_TOKEN_JSON);
    let nested = TokenizerTestContext::from_json(&doubly_nested_metaspace_special_decoder_json());
    let skip = talu_sys::DecodeOptionsC {
        skip_special_tokens: 1,
    };
    assert_eq!(
        nested.decode_with(&[1, 4, 3, 2, 5, 6, 7], &skip),
        flat.decode_with(&[1, 4, 3, 2, 5, 6, 7], &skip),
        "doubly nested Metaspace must match flat skip-special behavior across special-token boundaries"
    );
}

/// Null decode options must preserve the same default skip-special behavior on
/// a root nested Metaspace decoder subtree as on the flat decoder.
#[test]
fn doubly_nested_metaspace_with_special_tokens_null_options_matches_flat_behavior() {
    let flat = TokenizerTestContext::from_json(METASPACE_SPECIAL_TOKEN_JSON);
    let nested = TokenizerTestContext::from_json(&doubly_nested_metaspace_special_decoder_json());
    let ids = [1, 4, 3, 2, 5, 6, 7];

    let flat_result = unsafe { super::common::decode_raw_null_options(flat.handle(), &ids) };
    assert!(
        flat_result.error_msg.is_null(),
        "flat decode with null options should succeed"
    );
    let flat_text = unsafe {
        let slice = std::slice::from_raw_parts(flat_result.text, flat_result.text_len);
        std::str::from_utf8(slice)
            .expect("flat decode must return valid UTF-8")
            .to_owned()
    };
    unsafe { talu_sys::talu_decode_result_free(flat_result.text, flat_result.text_len) };

    let nested_result = unsafe { super::common::decode_raw_null_options(nested.handle(), &ids) };
    assert!(
        nested_result.error_msg.is_null(),
        "nested decode with null options should succeed"
    );
    let nested_text = unsafe {
        let slice = std::slice::from_raw_parts(nested_result.text, nested_result.text_len);
        std::str::from_utf8(slice)
            .expect("nested decode must return valid UTF-8")
            .to_owned()
    };
    unsafe { talu_sys::talu_decode_result_free(nested_result.text, nested_result.text_len) };

    assert_eq!(
        nested_text, flat_text,
        "doubly nested Metaspace must match flat null-options decode behavior across special-token boundaries"
    );
}

// ===========================================================================
// ByteLevel BPE: added token with tab character decodes correctly
// ===========================================================================
//
// Added tokens can contain tab characters (JSON escape \t → U+0009). The
// loader must unescape the JSON, and the decoder must output the literal
// tab character, not the two-character sequence "\t".

/// Added token with tab character content must decode to literal tabs.
#[test]
fn added_token_tab_content_decodes_to_tab() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"hello": 0, "world": 1},
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "hello", "special": false},
    {"id": 1, "content": "world", "special": false},
    {"id": 5, "content": "\t\t", "special": false},
    {"id": 6, "content": "\t\t\t", "special": false}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    // Token 5 = two tab characters, Token 6 = three tab characters
    let decoded5 = ctx.decode(&[5]);
    assert_eq!(
        decoded5, "\t\t",
        "added token with \\t\\t content must decode to 2 tab chars, got: {decoded5:?}"
    );
    let decoded6 = ctx.decode(&[6]);
    assert_eq!(
        decoded6, "\t\t\t",
        "added token with \\t\\t\\t content must decode to 3 tab chars, got: {decoded6:?}"
    );
}

// ===========================================================================
// SentencePiece BPE: no extra ▁ after second special token
// ===========================================================================
//
// Models with a Prepend("▁") + Replace(" " → "▁") normalizer (SentencePiece
// style) normalize the full input once. The Prepend adds ▁ at the start of
// the text. When the text begins with a special token, the initial ▁ is
// skipped and re-attached to the first non-special segment. However, no
// extra ▁ should be added to segments after subsequent special tokens —
// those segments already have ▁ from the Replace normalizer.

const PREPEND_REPLACE_BPE_JSON: &str = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "\u2581": 3,
      "h": 4, "e": 5, "l": 6, "o": 7,
      "w": 8, "r": 9, "d": 10,
      "\u2581h": 11, "\u2581he": 12, "\u2581hel": 13, "\u2581hell": 14, "\u2581hello": 15,
      "\u2581w": 16, "\u2581wo": 17, "\u2581wor": 18, "\u2581worl": 19, "\u2581world": 20
    },
    "merges": [
      "\u2581 h", "\u2581h e", "\u2581he l", "\u2581hel l", "\u2581hell o",
      "\u2581 w", "\u2581w o", "\u2581wo r", "\u2581wor l", "\u2581worl d"
    ]
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true}
  ],
  "normalizer": {
    "type": "Sequence",
    "normalizers": [
      {"type": "Prepend", "prepend": "\u2581"},
      {"type": "Replace", "pattern": {"String": " "}, "content": "\u2581"}
    ]
  },
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;

/// SentencePiece Prepend must re-arm after each special token.
///
/// Input "<s> hello </s> world" normalizes to "▁<s>▁hello▁</s>▁world".
/// The initial ▁ before <s> is re-attached to the first segment ("▁hello▁").
/// After </s>, the Prepend ▁ must ALSO re-attach to "▁world", producing an
/// extra standalone ▁ token.
#[test]
fn sentencepiece_prepend_rearms_after_second_special_token() {
    let ctx = TokenizerTestContext::from_json(PREPEND_REPLACE_BPE_JSON);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    let tokens = ctx.encode_with("<s> hello </s> world", &opts);
    // Expected: <s>(1), ▁(3), ▁hello(15), ▁(3), </s>(2), ▁(3), ▁world(20)
    // The ▁(3) after <s> is the re-attached initial Prepend ▁.
    // The ▁(3) after </s> is the re-armed Prepend ▁ for the next segment.
    assert_eq!(
        tokens,
        vec![1, 3, 15, 3, 2, 3, 20],
        "prepend ▁ must re-arm after every special token, got: {tokens:?}"
    );
}

/// This fixture has `decoder: null`, so decode must preserve the raw
/// SentencePiece marker tokens exactly as stored in vocab. Retained specials
/// must not cause the raw `▁` markers around them to be dropped or fused.
#[test]
fn sentencepiece_prepend_decode_without_decoder_preserves_raw_markers() {
    let ctx = TokenizerTestContext::from_json(PREPEND_REPLACE_BPE_JSON);
    let retain = talu_sys::DecodeOptionsC {
        skip_special_tokens: 0,
    };
    let decoded = ctx.decode_with(&[1, 3, 15, 3, 2, 3, 20], &retain);
    assert_eq!(
        decoded, "<s>▁▁hello▁</s>▁▁world",
        "without an explicit decoder, decode must preserve raw SentencePiece marker tokens across retained specials, got: {decoded:?}"
    );
}

// ===========================================================================
// BPE decode: contraction spacing in clean_up_tokenization_spaces
// ===========================================================================
//
// HuggingFace's clean_up_tokenization_spaces removes space before apostrophe
// in contractions: " n't" → "n't", " 'm" → "'m", " 's" → "'s", " 've" → "'ve",
// " 're" → "'re". It also removes space before standalone apostrophe when
// surrounded by spaces: " ' " → "'".

/// Cleanup removes space before question mark.
///
/// WordPiece decoder with cleanup must remove space before "?".
#[test]
fn cleanup_removes_space_before_question() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 200,
    "vocab": {
      "[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
      "hello": 3, "?": 4, "!": 5, ".": 6
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true},
    {"id": 1, "content": "[CLS]", "special": true},
    {"id": 2, "content": "[SEP]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}
}"####;
    let ctx = TokenizerTestContext::from_json(json);

    // "hello" + "?" → cleanup removes space before ? → "hello?"
    let decoded = ctx.decode(&[3, 4]);
    assert_eq!(
        decoded, "hello?",
        "cleanup must remove space before ?, got: {decoded:?}"
    );
}

/// Cleanup removes space before exclamation mark.
#[test]
fn cleanup_removes_space_before_exclamation() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 200,
    "vocab": {
      "[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
      "hello": 3, "!": 4
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true},
    {"id": 1, "content": "[CLS]", "special": true},
    {"id": 2, "content": "[SEP]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}
}"####;
    let ctx = TokenizerTestContext::from_json(json);

    let decoded = ctx.decode(&[3, 4]);
    assert_eq!(
        decoded, "hello!",
        "cleanup must remove space before !, got: {decoded:?}"
    );
}

/// Cleanup removes space before period.
#[test]
fn cleanup_removes_space_before_period() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 200,
    "vocab": {
      "[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
      "hello": 3, ".": 4
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true},
    {"id": 1, "content": "[CLS]", "special": true},
    {"id": 2, "content": "[SEP]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}
}"####;
    let ctx = TokenizerTestContext::from_json(json);

    let decoded = ctx.decode(&[3, 4]);
    assert_eq!(
        decoded, "hello.",
        "cleanup must remove space before period, got: {decoded:?}"
    );
}

/// Multiple consecutive special tokens all stripped with skip=1.
#[test]
fn skip_special_multiple_consecutive() {
    let ctx = TokenizerTestContext::with_special_tokens();
    let skip = talu_sys::DecodeOptionsC {
        skip_special_tokens: 1,
    };
    // [BOS=1, BOS=1, PAD=0, H=44, i=77, EOS=2, EOS=2]
    let decoded = ctx.decode_with(&[1, 1, 0, 44, 77, 2, 2], &skip);
    assert_eq!(
        decoded, "Hi",
        "multiple consecutive special tokens must all be stripped, got: {decoded:?}"
    );
}

/// Interleaved special and regular tokens with skip=1: only specials removed.
#[test]
fn skip_special_interleaved() {
    let ctx = TokenizerTestContext::with_special_tokens();
    let skip = talu_sys::DecodeOptionsC {
        skip_special_tokens: 1,
    };
    // [H=44, BOS=1, i=77, EOS=2, 37=A]
    let decoded = ctx.decode_with(&[44, 1, 77, 2, 37], &skip);
    assert_eq!(
        decoded, "HiA",
        "interleaved specials must be removed, regular preserved, got: {decoded:?}"
    );
}

/// Invariant: skip_special output equals retained output with special literals removed.
#[test]
fn skip_special_invariant_matches_manual_filtering() {
    let ctx = TokenizerTestContext::with_special_tokens();
    let keep = talu_sys::DecodeOptionsC {
        skip_special_tokens: 0,
    };
    let skip = talu_sys::DecodeOptionsC {
        skip_special_tokens: 1,
    };
    let ids = [1, 44, 77, 2, 44, 1, 77, 2, 0, 44];
    let retained = ctx.decode_with(&ids, &keep);
    let skipped = ctx.decode_with(&ids, &skip);
    let manual = retained
        .replace("<s>", "")
        .replace("</s>", "")
        .replace("<pad>", "")
        .replace("<unk>", "");
    assert_eq!(
        skipped, manual,
        "skip_special output must equal retained output with special tokens removed"
    );
}

/// Cleanup removes space before apostrophe contractions.
///
/// Contractions n't, 's, 'm must have space before apostrophe removed.
#[test]
fn cleanup_removes_contraction_space() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 200,
    "vocab": {
      "[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
      "i": 3, "do": 4, "n": 5, "'": 6, "t": 7,
      "he": 8, "##'": 9, "##s": 10
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true},
    {"id": 1, "content": "[CLS]", "special": true},
    {"id": 2, "content": "[SEP]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}
}"####;
    let ctx = TokenizerTestContext::from_json(json);

    // "i" + "do" + "n" + "'" + "t" → "i don 't" before cleanup
    // cleanup: " n't" → "n't" → "i don't"
    let decoded = ctx.decode(&[3, 4, 5, 6, 7]);
    assert_eq!(
        decoded, "i don't",
        "cleanup must remove space before n't contraction, got: {decoded:?}"
    );
}

/// Cleanup must apply multiple contraction rules in one output string.
#[test]
fn cleanup_applies_multiple_contractions_in_sequence() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 200,
    "vocab": {
      "[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
      "i": 3, "'m": 4, "we": 5, "'re": 6, "they": 7, "'ve": 8, "?": 9
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true},
    {"id": 1, "content": "[CLS]", "special": true},
    {"id": 2, "content": "[SEP]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}
}"####;
    let ctx = TokenizerTestContext::from_json(json);

    // Before cleanup: "i 'm we 're they 've ?"
    // After cleanup:  "i'm we're they've?"
    let decoded = ctx.decode(&[3, 4, 5, 6, 7, 8, 9]);
    assert_eq!(
        decoded, "i'm we're they've?",
        "cleanup must handle overlapping contraction + punctuation rules, got: {decoded:?}"
    );
}

/// Cleanup collapses standalone apostrophe spacing: "a ' b" -> "a'b".
#[test]
fn cleanup_collapses_standalone_apostrophe_spacing() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 200,
    "vocab": {
      "[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
      "a": 3, "'": 4, "b": 5
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true},
    {"id": 1, "content": "[CLS]", "special": true},
    {"id": 2, "content": "[SEP]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let decoded = ctx.decode(&[3, 4, 5]);
    assert_eq!(
        decoded, "a'b",
        "cleanup must collapse standalone apostrophe spacing, got: {decoded:?}"
    );
}

/// With cleanup disabled, standalone apostrophe spacing must be preserved.
#[test]
fn cleanup_disabled_preserves_standalone_apostrophe_spacing() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 200,
    "vocab": {
      "[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
      "a": 3, "'": 4, "b": 5
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true},
    {"id": 1, "content": "[CLS]", "special": true},
    {"id": 2, "content": "[SEP]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let decoded = ctx.decode(&[3, 4, 5]);
    assert_eq!(
        decoded, "a ' b",
        "cleanup=false must preserve apostrophe spacing, got: {decoded:?}"
    );
}

/// With cleanup disabled, spacing before punctuation/contractions must remain.
#[test]
fn cleanup_disabled_preserves_spaces() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 200,
    "vocab": {
      "[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
      "i": 3, "'m": 4, "sure": 5, "?": 6
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true},
    {"id": 1, "content": "[CLS]", "special": true},
    {"id": 2, "content": "[SEP]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let decoded = ctx.decode(&[3, 4, 5, 6]);
    assert_eq!(
        decoded, "i 'm sure ?",
        "cleanup=false must preserve intermediate spaces, got: {decoded:?}"
    );
}

/// Cleanup removes only the space immediately before punctuation, matching the
/// sequential HuggingFace replacement contract even when literal space tokens
/// create a wider run of spaces.
#[test]
fn cleanup_multiple_literal_spaces_before_question_mark() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 200,
    "vocab": {
      "[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
      "hello": 3, " ": 4, "?": 5
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true},
    {"id": 1, "content": "[CLS]", "special": true},
    {"id": 2, "content": "[SEP]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let decoded = ctx.decode(&[3, 4, 5]);
    assert_eq!(
        decoded, "hello  ?",
        "cleanup must remove only the final punctuation-adjacent space in a literal space run, got: {decoded:?}"
    );
}

/// Large cleanup workloads with many literal-space tokens before punctuation
/// must remain deterministic and preserve the same sequential replacement
/// contract as small cases.
#[test]
fn cleanup_large_literal_space_run_before_question_mark_is_deterministic() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 200,
    "vocab": {
      "[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
      " ": 3, "?": 4
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true},
    {"id": 1, "content": "[CLS]", "special": true},
    {"id": 2, "content": "[SEP]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let n = 20_000usize;
    let mut ids = vec![3u32; n];
    ids.push(4);

    let first = ctx.decode(&ids);
    let second = ctx.decode(&ids);
    assert_eq!(first, second, "large cleanup decode must be deterministic");

    let expected = format!("{}?", " ".repeat((2 * n) - 1));
    assert_eq!(
        first, expected,
        "large literal-space cleanup must preserve the same one-space removal contract"
    );
}

/// Cleanup should not strip newline content itself. If a literal newline token
/// sits before punctuation, only the ordinary space inserted before the
/// punctuation should be removed.
#[test]
fn cleanup_preserves_newline_before_question_mark() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 200,
    "vocab": {
      "[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
      "hello": 3, "\n": 4, "?": 5
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true},
    {"id": 1, "content": "[CLS]", "special": true},
    {"id": 2, "content": "[SEP]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let decoded = ctx.decode(&[3, 4, 5]);
    assert_eq!(
        decoded, "hello \n?",
        "cleanup must preserve literal newline content while stripping only the final space before punctuation, got: {decoded:?}"
    );
}

/// Cleanup with French guillemets should preserve interior quote spacing while
/// still applying punctuation cleanup before a trailing question mark.
#[test]
fn cleanup_french_guillemets_preserve_inner_spacing_and_strip_space_before_question() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 200,
    "vocab": {
      "[UNK]": 0, "hello": 1, "«": 2, "»": 3, "?": 4
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let decoded = ctx.decode(&[2, 1, 3, 4]);
    assert_eq!(
        decoded, "« hello »?",
        "cleanup should preserve guillemet inner spacing while stripping final punctuation-adjacent space"
    );
}

/// Invalid token IDs must error even when mixed with valid IDs.
#[test]
fn decode_mixed_valid_and_invalid_token_ids_errors() {
    let ctx = TokenizerTestContext::new();
    let opts = talu_sys::DecodeOptionsC::default();
    let result = unsafe { super::common::decode_raw(ctx.handle(), &[44, 999_999, 77], &opts) };
    assert!(
        !result.error_msg.is_null(),
        "decode must fail when any token ID is invalid"
    );
    assert!(
        result.text.is_null(),
        "text pointer must be null on decode error"
    );
    assert_eq!(result.text_len, 0, "text_len must be 0 on decode error");
}

/// WordPiece decode must reject invalid token IDs consistently with BPE decode.
#[test]
fn wordpiece_decode_invalid_token_id_errors() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 200,
    "vocab": {
      "[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
      "hello": 3, "world": 4
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true},
    {"id": 1, "content": "[CLS]", "special": true},
    {"id": 2, "content": "[SEP]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let opts = talu_sys::DecodeOptionsC::default();
    let result = unsafe { super::common::decode_raw(ctx.handle(), &[3, 999_999, 4], &opts) };
    assert!(
        !result.error_msg.is_null(),
        "WordPiece decode should reject out-of-range token IDs"
    );
    assert!(
        result.text.is_null(),
        "text pointer must be null for invalid WordPiece decode"
    );
    assert_eq!(
        result.text_len, 0,
        "text_len must be 0 for invalid WordPiece decode"
    );
}

/// Invalid ID before a valid WordPiece continuation token must fail cleanly and
/// not enter a partial subword decode state.
#[test]
fn wordpiece_decode_invalid_then_subword_errors_cleanly() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 200,
    "vocab": {
      "[UNK]": 0, "go": 1, "##ing": 2
    }
  },
  "added_tokens": [{"id": 0, "content": "[UNK]", "special": true}],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let result = unsafe {
        super::common::decode_raw(
            ctx.handle(),
            &[999_999, 2],
            &talu_sys::DecodeOptionsC::default(),
        )
    };
    assert!(
        !result.error_msg.is_null(),
        "invalid+subword decode must fail with a typed error"
    );
    assert!(result.text.is_null());
    assert_eq!(result.text_len, 0);
    unsafe { talu_sys::talu_decode_result_free(result.text, result.text_len) };

    // Erroring decode must not poison subsequent decodes on the same handle.
    let ok = unsafe {
        super::common::decode_raw(ctx.handle(), &[1, 2], &talu_sys::DecodeOptionsC::default())
    };
    assert!(
        ok.error_msg.is_null(),
        "decoder state must recover after invalid+subword failure"
    );
    let text = unsafe {
        let bytes = std::slice::from_raw_parts(ok.text, ok.text_len);
        std::str::from_utf8(bytes).expect("wordpiece decode output must be UTF-8")
    };
    assert_eq!(text, "going");
    unsafe { talu_sys::talu_decode_result_free(ok.text, ok.text_len) };
}

/// Valid WordPiece continuation token followed by invalid ID must also fail
/// cleanly and avoid partial output emission.
#[test]
fn wordpiece_decode_subword_then_invalid_errors_cleanly() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 200,
    "vocab": {
      "[UNK]": 0, "go": 1, "##ing": 2
    }
  },
  "added_tokens": [{"id": 0, "content": "[UNK]", "special": true}],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let result = unsafe {
        super::common::decode_raw(
            ctx.handle(),
            &[2, 999_999],
            &talu_sys::DecodeOptionsC::default(),
        )
    };
    assert!(
        !result.error_msg.is_null(),
        "subword+invalid decode must fail with a typed error"
    );
    assert!(result.text.is_null());
    assert_eq!(result.text_len, 0);
    unsafe { talu_sys::talu_decode_result_free(result.text, result.text_len) };

    // Erroring decode must not poison subsequent decodes on the same handle.
    let ok = unsafe {
        super::common::decode_raw(ctx.handle(), &[1, 2], &talu_sys::DecodeOptionsC::default())
    };
    assert!(
        ok.error_msg.is_null(),
        "decoder state must recover after subword+invalid failure"
    );
    let text = unsafe {
        let bytes = std::slice::from_raw_parts(ok.text, ok.text_len);
        std::str::from_utf8(bytes).expect("wordpiece decode output must be UTF-8")
    };
    assert_eq!(text, "going");
    unsafe { talu_sys::talu_decode_result_free(ok.text, ok.text_len) };
}

/// Special-token wrappers must not hide invalid IDs in the middle.
#[test]
fn decode_bos_invalid_eos_errors_cleanly() {
    let ctx = TokenizerTestContext::with_special_tokens();
    let result = unsafe {
        super::common::decode_raw(
            ctx.handle(),
            &[1, 999_999, 2],
            &talu_sys::DecodeOptionsC {
                skip_special_tokens: 0,
            },
        )
    };
    assert!(
        !result.error_msg.is_null(),
        "BOS/invalid/EOS decode must fail instead of partially decoding"
    );
    assert!(result.text.is_null());
    assert_eq!(result.text_len, 0);

    // Special-token decode error must not poison subsequent decodes.
    let ok = unsafe {
        super::common::decode_raw(
            ctx.handle(),
            &[1, 44, 77, 2],
            &talu_sys::DecodeOptionsC {
                skip_special_tokens: 0,
            },
        )
    };
    assert!(
        ok.error_msg.is_null(),
        "decoder state must recover after BOS/invalid/EOS failure"
    );
    let text = unsafe {
        let bytes = std::slice::from_raw_parts(ok.text, ok.text_len);
        std::str::from_utf8(bytes).expect("decode output must be UTF-8")
    };
    assert_eq!(text, "<s>Hi</s>");
    unsafe { talu_sys::talu_decode_result_free(ok.text, ok.text_len) };
}
