//! Encode functional tests.
//!
//! Validates token encoding with exact token IDs, EncodeOptions flags,
//! truncation, and roundtrip fidelity.
//!
//! Fixture vocab: each ASCII char maps to a single token (no merges).
//!   H=44, e=73, l=80, o=83, i=77, A=37, a=69, b=70, c=71,
//!   0=20, 1=21, 2=22. Space → <unk>=3 (ByteLevel remaps 0x20).

use crate::capi::tokenizer::common::TokenizerTestContext;

/// Encoding "Hi" without BOS produces exactly [H=44, i=77].
#[test]
fn encode_exact_token_ids() {
    let ctx = TokenizerTestContext::new();
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    assert_eq!(ctx.encode_with("Hi", &opts), [44, 77]);
}

/// Encoding "Hello" without BOS produces [H=44, e=73, l=80, l=80, o=83].
#[test]
fn encode_hello_per_character() {
    let ctx = TokenizerTestContext::new();
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    assert_eq!(ctx.encode_with("Hello", &opts), [44, 73, 80, 80, 83]);
}

/// Digits 0-9 map to IDs 20-29.
#[test]
fn encode_digits() {
    let ctx = TokenizerTestContext::new();
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    assert_eq!(ctx.encode_with("012", &opts), [20, 21, 22]);
}

/// Lowercase a-c map to IDs 69-71.
#[test]
fn encode_lowercase() {
    let ctx = TokenizerTestContext::new();
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    assert_eq!(ctx.encode_with("abc", &opts), [69, 70, 71]);
}

/// Space is remapped by ByteLevel pre-tokenizer; falls back to <unk>=3.
#[test]
fn encode_space_becomes_unk() {
    let ctx = TokenizerTestContext::new();
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    let tokens = ctx.encode_with("ab cd", &opts);
    assert_eq!(tokens, [69, 70, 3, 71, 72]);
}

/// Empty string with add_bos=0 produces zero tokens.
#[test]
fn encode_empty_string() {
    let ctx = TokenizerTestContext::new();
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    assert_eq!(ctx.encode_with("", &opts), Vec::<u32>::new());
}

/// The add_bos flag has no effect with this tokenizer (no post-processor).
/// Both add_bos=0 and add_bos=1 produce identical output.
#[test]
fn encode_bos_flag_no_effect_without_post_processor() {
    let ctx = TokenizerTestContext::new();
    let with_bos = talu_sys::EncodeOptions {
        add_bos: 1,
        ..Default::default()
    };
    let without_bos = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    assert_eq!(
        ctx.encode_with("Hi", &with_bos),
        ctx.encode_with("Hi", &without_bos),
    );
}

/// Right truncation keeps the first max_length tokens.
#[test]
fn encode_truncation_right() {
    let ctx = TokenizerTestContext::new();
    // "Hello" => [44, 73, 80, 80, 83] (5 tokens), truncate to 2.
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        truncation: 1,
        truncation_side: 0, // right: keep first
        max_length: 2,
        ..Default::default()
    };
    assert_eq!(ctx.encode_with("Hello", &opts), [44, 73]);
}

/// Left truncation keeps the last max_length tokens.
#[test]
fn encode_truncation_left() {
    let ctx = TokenizerTestContext::new();
    // "Hello" => [44, 73, 80, 80, 83] (5 tokens), truncate to 2.
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        truncation: 1,
        truncation_side: 1, // left: keep last
        max_length: 2,
        ..Default::default()
    };
    assert_eq!(ctx.encode_with("Hello", &opts), [80, 83]);
}

/// Truncation with max_length >= token count is a no-op.
#[test]
fn encode_truncation_noop_when_under_limit() {
    let ctx = TokenizerTestContext::new();
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        truncation: 1,
        max_length: 1000,
        ..Default::default()
    };
    assert_eq!(ctx.encode_with("Hi", &opts), [44, 77]);
}

/// Repeated pattern with spaces: "abc123 " × 50.
/// Spaces become unk (3), ASCII chars encode normally.
#[test]
fn encode_repeated_pattern_with_spaces() {
    let ctx = TokenizerTestContext::new();
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    let pattern = "abc123 ";
    let input = pattern.repeat(50);
    let tokens = ctx.encode_with(&input, &opts);

    // "abc123 " = 7 bytes → 7 tokens per repetition.
    assert_eq!(tokens.len(), 350);

    // Each repetition: a=69, b=70, c=71, 1=21, 2=22, 3=23, space=3(unk).
    let expected_chunk = [69u32, 70, 71, 21, 22, 23, 3];
    for (i, chunk) in tokens.chunks(7).enumerate() {
        assert_eq!(chunk, expected_chunk, "repetition {i} mismatch");
    }
}

/// count_tokens (encode length) is consistent across multiple methods.
#[test]
fn count_tokens_matches_encode_length() {
    let ctx = TokenizerTestContext::new();
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };

    for text in ["Hello", "abc", "012", "café", "🎉", ""] {
        let encode_len = ctx.encode_with(text, &opts).len();

        // tokenize_bytes should agree on token count.
        let result = unsafe {
            talu_sys::talu_tokenizer_tokenize_bytes(
                ctx.handle(),
                text.as_bytes().as_ptr(),
                text.len(),
            )
        };
        assert!(result.error_msg.is_null());
        assert_eq!(
            result.num_tokens, encode_len,
            "token count mismatch for {text:?}"
        );
        unsafe {
            talu_sys::talu_tokenize_bytes_result_free(
                result.data,
                result.data_len,
                result.offsets,
                result.num_tokens,
            )
        };

        // Encode result offsets count should also agree.
        let encode_result = unsafe {
            super::common::encode_raw(ctx.handle(), text.as_bytes(), &opts)
        };
        assert!(encode_result.error_msg.is_null());
        assert_eq!(
            encode_result.num_tokens, encode_len,
            "encode offsets count mismatch for {text:?}"
        );
        unsafe { talu_sys::talu_encode_result_free(encode_result) };
    }
}

/// Encode then decode recovers the original text (non-space ASCII).
#[test]
fn encode_decode_roundtrip() {
    let ctx = TokenizerTestContext::new();
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    for input in ["Hello", "abc", "012", "A", "!@#$%"] {
        let tokens = ctx.encode_with(input, &opts);
        let decoded = ctx.decode(&tokens);
        assert_eq!(decoded, input, "roundtrip failed for {input:?}");
    }
}

// ===========================================================================
// TemplateProcessing post_processor: BOS/EOS insertion during encode
// ===========================================================================
//
// When a tokenizer has a TemplateProcessing post_processor, encode with
// add_bos=1 must prepend the BOS token and append the EOS token.
//
// Bug: `build_tokenizer_from_root` copies the cls_token/sep_token strings
// from the post_processor spec but never resolves their IDs from the vocab.
// cls_id/sep_id stay at -1, so the wrong token ID (-1 / 4294967295) is
// inserted, or no token is inserted at all.
// Affects: Llama-3, Gemma, LiquidAI, TinyLlama, Mistral (~52 models).

/// Minimal BPE tokenizer with TemplateProcessing post_processor.
///
/// Special tokens (only in added_tokens, not model.vocab):
///   1: `<s>` (BOS/CLS), 2: `</s>` (EOS/SEP)
///
/// The TemplateProcessing post_processor should:
/// - Prepend `<s>` (ID 1) to the encoded output
/// - Append `</s>` (ID 2) to the encoded output
const TEMPLATE_POSTPROC_JSON: &str = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "H": 4, "i": 5, "e": 6, "l": 7, "o": 8
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
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

/// TemplateProcessing post_processor must prepend BOS and append EOS.
///
/// Bug: cls_id/sep_id never resolved from vocab → wrong or missing BOS/EOS.
#[test]
fn encode_template_postproc_adds_bos_and_eos() {
    let ctx = TokenizerTestContext::from_json(TEMPLATE_POSTPROC_JSON);
    let opts = talu_sys::EncodeOptions {
        add_bos: 1,
        ..Default::default()
    };
    // "Hi" → [H=4, i=5], with BOS/EOS → [<s>=1, H=4, i=5, </s>=2]
    let tokens = ctx.encode_with("Hi", &opts);
    assert_eq!(
        tokens, vec![1, 4, 5, 2],
        "TemplateProcessing must add BOS=1 and EOS=2, got: {tokens:?}"
    );
}

/// Empty string with BOS/EOS should produce [BOS, EOS].
#[test]
fn encode_template_postproc_empty_string() {
    let ctx = TokenizerTestContext::from_json(TEMPLATE_POSTPROC_JSON);
    let opts = talu_sys::EncodeOptions {
        add_bos: 1,
        ..Default::default()
    };
    let tokens = ctx.encode_with("", &opts);
    assert_eq!(
        tokens, vec![1, 2],
        "empty string with TemplateProcessing should produce [BOS, EOS], got: {tokens:?}"
    );
}

/// TemplateProcessing with add_bos=0 must NOT add BOS/EOS.
///
/// Regression: tri-state add_bos mapped 0 to "model default" instead of
/// "skip post_processor", causing BOS/EOS insertion even when disabled.
#[test]
fn encode_template_postproc_skip_when_add_bos_zero() {
    let ctx = TokenizerTestContext::from_json(TEMPLATE_POSTPROC_JSON);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    // "Hi" → [H=4, i=5] only, no BOS/EOS
    let tokens = ctx.encode_with("Hi", &opts);
    assert_eq!(
        tokens, vec![4, 5],
        "add_bos=0 must skip post_processor, got: {tokens:?}"
    );
}

/// Default encode (EncodeOptions::default() has add_bos=0) must skip
/// post_processor — consistent with HF encode(text, add_special_tokens=False).
#[test]
fn encode_template_postproc_default_opts_no_special() {
    let ctx = TokenizerTestContext::from_json(TEMPLATE_POSTPROC_JSON);
    let tokens = ctx.encode("Hi");
    assert_eq!(
        tokens, vec![4, 5],
        "default encode must not add BOS/EOS, got: {tokens:?}"
    );
}

// ===========================================================================
// Metaspace pretokenizer: encode must add ▁ prefix and replace spaces
// ===========================================================================
//
// SentencePiece BPE models (Mistral, TinyLlama, Phi-3, etc.) use a Metaspace
// pretokenizer that:
// 1. Prepends ▁ to the input when add_prefix_space is true
// 2. Replaces internal spaces with ▁
//
// Bug: spaces are tokenized as individual characters instead of being
// converted to ▁ and merged with adjacent text.
// Affects: Mistral-7B (all versions), ~80 encode failures.

/// Minimal SentencePiece BPE with Metaspace pretokenizer.
///
/// Vocab includes intermediate merge tokens so BPE can combine characters
/// into whole-word tokens (e.g. ▁ + H + e + l + l + o → ▁Hello).
const METASPACE_ENCODE_JSON: &str = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "\u2581Hello": 3, ",": 4, "\u2581world": 5, "!": 6,
      "\u2581": 7, "H": 8, "e": 9, "l": 10, "o": 11,
      "w": 12, "r": 13, "d": 14,
      "He": 15, "Hel": 16, "Hell": 17, "Hello": 18,
      "wo": 19, "wor": 20, "worl": 21, "world": 22
    },
    "merges": [
      "H e", "He l", "Hel l", "Hell o", "\u2581 Hello",
      "w o", "wo r", "wor l", "worl d", "\u2581 world"
    ]
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

/// Metaspace pretokenizer: encode must prepend ▁ and tokenize correctly.
///
/// "Hello" with add_prefix_space=true → "▁Hello" → token [3] (▁Hello)
///
/// Bug: Metaspace pretokenizer may not prepend ▁ during encode, causing
/// "Hello" to be tokenized character-by-character.
#[test]
fn encode_metaspace_single_word() {
    let ctx = TokenizerTestContext::from_json(METASPACE_ENCODE_JSON);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    let tokens = ctx.encode_with("Hello", &opts);
    assert_eq!(
        tokens, vec![3],
        "Metaspace encode 'Hello' → [▁Hello=3], got: {tokens:?}"
    );
}

/// Metaspace pretokenizer: encode multi-word replaces spaces with ▁.
///
/// "Hello, world!" → "▁Hello" + "," + "▁world" + "!" → [3, 4, 5, 6]
#[test]
fn encode_metaspace_multi_word() {
    let ctx = TokenizerTestContext::from_json(METASPACE_ENCODE_JSON);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    let tokens = ctx.encode_with("Hello, world!", &opts);
    assert_eq!(
        tokens, vec![3, 4, 5, 6],
        "Metaspace encode 'Hello, world!' → [▁Hello, comma, ▁world, !], got: {tokens:?}"
    );
}
