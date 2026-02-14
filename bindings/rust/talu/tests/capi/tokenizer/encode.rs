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
        let encode_result =
            unsafe { super::common::encode_raw(ctx.handle(), text.as_bytes(), &opts) };
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
// Encode with add_bos=1 must resolve CLS/SEP token IDs from the vocab and
// prepend BOS / append EOS according to the template.

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
/// Verifies cls_id/sep_id are resolved from vocab → correct BOS/EOS inserted.
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
        tokens,
        vec![1, 4, 5, 2],
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
        tokens,
        vec![1, 2],
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
        tokens,
        vec![4, 5],
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
        tokens,
        vec![4, 5],
        "default encode must not add BOS/EOS, got: {tokens:?}"
    );
}

// ===========================================================================
// Metaspace pretokenizer: encode must add ▁ prefix and replace spaces
// ===========================================================================
//
// Metaspace pretokenizer must prepend ▁ when add_prefix_space is true
// and replace internal spaces with ▁.

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
/// Verifies Metaspace pretokenizer prepends ▁ during encode. Without it,
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
        tokens,
        vec![3],
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
        tokens,
        vec![3, 4, 5, 6],
        "Metaspace encode 'Hello, world!' → [▁Hello, comma, ▁world, !], got: {tokens:?}"
    );
}

// ===========================================================================
// Sequence-wrapped TemplateProcessing
// ===========================================================================
//
// When TemplateProcessing is nested inside a Sequence post_processor,
// the loader must recurse into the processors array to find and apply it.

/// Minimal BPE tokenizer with Sequence-wrapped TemplateProcessing.
///
/// Post_processor is Sequence containing ByteLevel + TemplateProcessing.
/// BOS token is `<|begin_of_text|>` (ID 1).
const SEQUENCE_POSTPROC_JSON: &str = r####"{
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

/// Sequence-wrapped TemplateProcessing must prepend BOS.
///
/// Verifies Sequence-wrapped TemplateProcessing is found and applied.
#[test]
fn encode_sequence_wrapped_template_adds_bos() {
    let ctx = TokenizerTestContext::from_json(SEQUENCE_POSTPROC_JSON);
    let opts = talu_sys::EncodeOptions {
        add_bos: 1,
        ..Default::default()
    };
    // "Hi" → [H=4, i=5], with BOS → [<|begin_of_text|>=1, H=4, i=5]
    let tokens = ctx.encode_with("Hi", &opts);
    assert_eq!(
        tokens,
        vec![1, 4, 5],
        "Sequence-wrapped TemplateProcessing must add BOS=1, got: {tokens:?}"
    );
}

/// Sequence-wrapped TemplateProcessing: BOS-only template must NOT add EOS.
///
/// The single template is [BOS, A] — no EOS entry. Must not append any token.
#[test]
fn encode_sequence_wrapped_template_no_eos() {
    let ctx = TokenizerTestContext::from_json(SEQUENCE_POSTPROC_JSON);
    let opts = talu_sys::EncodeOptions {
        add_bos: 1,
        ..Default::default()
    };
    // Empty string with BOS-only template → [BOS] only, no EOS
    let tokens = ctx.encode_with("", &opts);
    assert_eq!(
        tokens,
        vec![1],
        "BOS-only template must produce [BOS] for empty string, got: {tokens:?}"
    );
}

// ===========================================================================
// BOS-only TemplateProcessing must not append an unwanted SEP
// ===========================================================================
//
// When the single template is [<s>, A] with no EOS/SEP, the encoder must
// prepend BOS only. It must not default SEP to the CLS token.

/// Minimal BPE with BOS-only TemplateProcessing (no EOS in template).
///
/// The special_tokens map has ONLY `<s>` — no `</s>`.
/// The single template is `[<s>, A]` — BOS prepended, nothing appended.
const BOS_ONLY_TEMPLATE_JSON: &str = r####"{
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
      {"Sequence": {"id": "A", "type_id": 0}}
    ],
    "pair": [
      {"SpecialToken": {"id": "<s>", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}},
      {"SpecialToken": {"id": "<s>", "type_id": 1}},
      {"Sequence": {"id": "B", "type_id": 1}}
    ],
    "special_tokens": {
      "<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]}
    }
  },
  "decoder": {"type": "ByteLevel"}
}"####;

/// BOS-only template must NOT append any token at end.
///
/// Verifies SEP is not added when the template only specifies BOS.
#[test]
fn encode_bos_only_template_no_trailing_token() {
    let ctx = TokenizerTestContext::from_json(BOS_ONLY_TEMPLATE_JSON);
    let opts = talu_sys::EncodeOptions {
        add_bos: 1,
        ..Default::default()
    };
    // "Hi" → [H=4, i=5], with BOS only → [<s>=1, H=4, i=5]
    let tokens = ctx.encode_with("Hi", &opts);
    assert_eq!(
        tokens,
        vec![1, 4, 5],
        "BOS-only template must not append extra token, got: {tokens:?}"
    );
}

/// BOS-only template: empty string produces [BOS] only.
#[test]
fn encode_bos_only_template_empty_string() {
    let ctx = TokenizerTestContext::from_json(BOS_ONLY_TEMPLATE_JSON);
    let opts = talu_sys::EncodeOptions {
        add_bos: 1,
        ..Default::default()
    };
    let tokens = ctx.encode_with("", &opts);
    assert_eq!(
        tokens,
        vec![1],
        "BOS-only template empty string → [BOS] only, got: {tokens:?}"
    );
}

// ===========================================================================
// Metaspace prepend_scheme:"first" must be handled as add_prefix_space
// ===========================================================================
//
// The newer HuggingFace config uses `prepend_scheme: "first"` instead of
// `add_prefix_space: true`. Both must produce the same ▁ prefix behavior.

/// Minimal SentencePiece BPE with Metaspace using prepend_scheme:"first".
///
/// Uses the newer HuggingFace Metaspace config with prepend_scheme
/// instead of add_prefix_space.
const METASPACE_PREPEND_SCHEME_JSON: &str = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "\u2581Hello": 3, ",": 4, "\u2581world": 5, "!": 6,
      "\u2581": 7, "H": 8, "e": 9, "l": 10, "o": 11,
      "w": 12, "r": 13, "d": 14,
      "He": 15, "Hel": 16, "Hell": 17, "Hello": 18,
      "wo": 19, "wor": 20, "worl": 21, "world": 22,
      "\u2581\u2581\u2581": 23, "\u2581\u2581": 24
    },
    "merges": [
      "H e", "He l", "Hel l", "Hell o", "\u2581 Hello",
      "w o", "wo r", "wor l", "worl d", "\u2581 world",
      "\u2581 \u2581", "\u2581\u2581 \u2581"
    ]
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Metaspace", "replacement": "\u2581", "prepend_scheme": "first"},
  "post_processor": null,
  "decoder": {"type": "Metaspace", "replacement": "\u2581", "prepend_scheme": "first"}
}"####;

/// Metaspace with prepend_scheme:"first" must prepend ▁ to first word.
///
/// Verifies prepend_scheme:"first" is recognized as add_prefix_space.
#[test]
fn encode_metaspace_prepend_scheme_first() {
    let ctx = TokenizerTestContext::from_json(METASPACE_PREPEND_SCHEME_JSON);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    let tokens = ctx.encode_with("Hello", &opts);
    assert_eq!(
        tokens,
        vec![3],
        "Metaspace prepend_scheme:'first' must prepend ▁: 'Hello' → [▁Hello=3], got: {tokens:?}"
    );
}

/// Metaspace with prepend_scheme:"first": whitespace-only must not be empty.
///
/// Verifies whitespace-only input "   " produces space tokens, not empty.
#[test]
fn encode_metaspace_whitespace_only() {
    let ctx = TokenizerTestContext::from_json(METASPACE_PREPEND_SCHEME_JSON);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    let tokens = ctx.encode_with("   ", &opts);
    assert_eq!(
        tokens,
        vec![23],
        "Metaspace whitespace '   ' → [▁▁▁=23], got: {tokens:?}"
    );
}

// ===========================================================================
// EOS-only TemplateProcessing must append EOS without prepending BOS
// ===========================================================================
//
// When the single template is [A, <|endoftext|>], the encoder must append
// EOS at the end without prepending any BOS token.

/// Minimal BPE with EOS-only TemplateProcessing.
///
/// The single template is [A, EOS] — content first, then EOS appended.
/// No BOS is prepended. The special_tokens map has only `<|endoftext|>`.
const EOS_ONLY_TEMPLATE_JSON: &str = r####"{
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
    {"id": 1, "content": "<|endoftext|>", "special": true},
    {"id": 2, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": {
    "type": "TemplateProcessing",
    "single": [
      {"Sequence": {"id": "A", "type_id": 0}},
      {"SpecialToken": {"id": "<|endoftext|>", "type_id": 0}}
    ],
    "pair": [
      {"Sequence": {"id": "A", "type_id": 0}},
      {"Sequence": {"id": "B", "type_id": 0}},
      {"SpecialToken": {"id": "<|endoftext|>", "type_id": 0}}
    ],
    "special_tokens": {
      "<|endoftext|>": {"id": "<|endoftext|>", "ids": [1], "tokens": ["<|endoftext|>"]}
    }
  },
  "decoder": {"type": "ByteLevel"}
}"####;

/// EOS-only template must append EOS at end, not prepend BOS at start.
///
/// "Hi" → [H=4, i=5], with EOS → [H=4, i=5, <|endoftext|>=1]
#[test]
fn encode_eos_only_template_appends_eos() {
    let ctx = TokenizerTestContext::from_json(EOS_ONLY_TEMPLATE_JSON);
    let opts = talu_sys::EncodeOptions {
        add_bos: 1,
        ..Default::default()
    };
    let tokens = ctx.encode_with("Hi", &opts);
    assert_eq!(
        tokens,
        vec![4, 5, 1],
        "EOS-only template must produce [tokens..., EOS=1], got: {tokens:?}"
    );
}

/// EOS-only template: empty string produces [EOS] only.
#[test]
fn encode_eos_only_template_empty_string() {
    let ctx = TokenizerTestContext::from_json(EOS_ONLY_TEMPLATE_JSON);
    let opts = talu_sys::EncodeOptions {
        add_bos: 1,
        ..Default::default()
    };
    let tokens = ctx.encode_with("", &opts);
    assert_eq!(
        tokens,
        vec![1],
        "EOS-only template empty string → [EOS=1], got: {tokens:?}"
    );
}

/// EOS-only template with add_bos=0 must skip post_processor entirely.
#[test]
fn encode_eos_only_template_no_special() {
    let ctx = TokenizerTestContext::from_json(EOS_ONLY_TEMPLATE_JSON);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    // With add_bos=0, post_processor is skipped → just raw tokens, no EOS.
    let tokens = ctx.encode_with("Hi", &opts);
    assert_eq!(
        tokens,
        vec![4, 5],
        "add_bos=0 must skip post_processor (no EOS), got: {tokens:?}"
    );
}

// ===========================================================================
// SentencePiece BPE: space token after special token must be preserved
// ===========================================================================
//
// When encoding text that contains special tokens as literals (e.g.
// "<s> and </s> are"), the text is split around special token spans.
// After each special token, the remaining text starts with a space that
// must be tokenized as a separate ▁ (metaspace) token.
//
// The space after a special token must appear as a separate ▁ token,
// not be consumed or merged into the following word.

/// Space after a special token in the middle of text must be preserved.
///
/// "<s> and </s> are" should tokenize the space after </s> as a separate
/// token, not merge it into the next word.
#[test]
fn space_after_special_token_preserved() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "\u2581": 3, "and": 4, "are": 5,
      "\u2581and": 6, "\u2581are": 7,
      "a": 8, "n": 9, "d": 10, "r": 11, "e": 12
    },
    "merges": ["a n", "an d", "a r", "ar e"]
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true },
  "post_processor": null,
  "decoder": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true }
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    // "<s> and </s> are" with <s> and </s> as special tokens:
    //   <s> → id=1 (special)
    //   " and " → "▁and▁" → [▁=3, and=4, ▁=3] or [▁and=6, ▁=3]
    //   </s> → id=2 (special)
    //   " are" → "▁are" → [▁=3, are=5] or [▁are=7]
    //
    // The key assertion: there must be a ▁ token (id=3) after </s>,
    // not merged into the next word.
    let tokens = ctx.encode_with("<s> and </s> are", &opts);

    // Find </s> (id=2) in the output and check that a ▁ follows it
    let sep_pos = tokens.iter().position(|&t| t == 2);
    assert!(
        sep_pos.is_some(),
        "Output must contain </s> token (id=2), got: {tokens:?}"
    );
    let after_sep = &tokens[sep_pos.unwrap() + 1..];
    assert!(
        !after_sep.is_empty(),
        "There must be tokens after </s>, got: {tokens:?}"
    );
    // The first token after </s> should be ▁ (id=3) — the space must not
    // be merged into the next word.
    assert_eq!(
        after_sep[0], 3,
        "First token after </s> must be ▁ (id=3), not merged into next word. Full output: {tokens:?}"
    );
}

// ===========================================================================
// SentencePiece BPE: normalizer prepend ▁ must re-attach after every special
// token, not just the first one.
// ===========================================================================
//
// When the normalizer has a `prepend: "▁"` (SentencePiece convention), the
// encoder skips the leading ▁ when the input starts with a special token and
// re-attaches it to the first non-special segment. But this is a one-shot
// flag: after the first segment consumes it, subsequent segments after
// special tokens mid-text do NOT get ▁ re-attached.
//
// The normalizer prepend ▁ must re-attach after every special token span,
// not just the first non-special segment.

/// SentencePiece BPE: ▁ must be re-prepended after each mid-text special token.
///
/// Uses only Prepend normalizer (no Replace) and spaceless input so the
/// prepend is the ONLY source of ▁. Without the fix, "world" after </s>
/// starts with 'w'; with the fix, it starts with '▁'.
#[test]
fn sentencepiece_prepend_after_every_special_token() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "\u2581": 3,
      "h": 4, "e": 5, "l": 6, "o": 7,
      "w": 8, "r": 9, "d": 10
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true}
  ],
  "normalizer": {
    "type": "Prepend",
    "prepend": "\u2581"
  },
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode("<s>hello</s>world");

    // Normalized: "▁<s>hello</s>world"
    // Segments: <s>(1) | "hello" | </s>(2) | "world"
    // ▁ prefix skipped at start (starts with <s>), re-attached to "hello":
    //   "hello" + prepend → "▁hello" → [▁=3, h=4, e=5, l=6, l=6, o=7]
    // After </s>, ▁ must be re-attached to "world":
    //   "world" + prepend → "▁world" → [▁=3, w=8, o=7, r=9, l=6, d=10]
    // Without the fix, "world" has no prepend:
    //   "world" (no prepend) → [w=8, o=7, r=9, l=6, d=10]
    let sep_pos = tokens.iter().position(|&t| t == 2);
    assert!(
        sep_pos.is_some(),
        "Output must contain </s> token (id=2), got: {tokens:?}"
    );
    let after_sep = &tokens[sep_pos.unwrap() + 1..];
    assert!(
        !after_sep.is_empty(),
        "There must be tokens after </s>, got: {tokens:?}"
    );
    assert_eq!(
        after_sep[0], 3,
        "First token after </s> must be ▁ (id=3) — prepend must re-attach after every special token. Got: {tokens:?}"
    );
}

// ===========================================================================
// TemplateProcessing: CLS/SEP token selection must use template ordering,
// not JSON map iteration order.
// ===========================================================================
//
// The TemplateProcessing post-processor has a `special_tokens` map that
// maps token names to IDs. The loader picks the first key as CLS and the
// second key as SEP. But JSON object iteration order is not guaranteed,
// so a map with [CLS], [MASK], [PAD], [SEP], [UNK] may pick [MASK] as
// SEP instead of [SEP].
//
// CLS/SEP must be selected by parsing the template array, not by iterating
// special_tokens map keys (which have no guaranteed order).

/// TemplateProcessing must select CLS/SEP tokens from template, not map order.
///
/// When the special_tokens map has multiple entries, the loader must parse
/// the "single" template array to determine which token is CLS (before
/// content) and which is SEP (after content).
#[test]
fn template_processing_correct_sep_token() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "hello": 0, "world": 1,
      "[UNK]": 10, "[CLS]": 11, "[SEP]": 12, "[PAD]": 13, "[MASK]": 14
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 10, "content": "[UNK]", "special": true},
    {"id": 11, "content": "[CLS]", "special": true},
    {"id": 12, "content": "[SEP]", "special": true},
    {"id": 13, "content": "[PAD]", "special": true},
    {"id": 14, "content": "[MASK]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
  "post_processor": {
    "type": "TemplateProcessing",
    "single": [
      {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}},
      {"SpecialToken": {"id": "[SEP]", "type_id": 0}}
    ],
    "pair": [
      {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}},
      {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
      {"Sequence": {"id": "B", "type_id": 1}},
      {"SpecialToken": {"id": "[SEP]", "type_id": 1}}
    ],
    "special_tokens": {
      "[CLS]": {"id": "[CLS]", "ids": [11], "tokens": ["[CLS]"]},
      "[MASK]": {"id": "[MASK]", "ids": [14], "tokens": ["[MASK]"]},
      "[PAD]": {"id": "[PAD]", "ids": [13], "tokens": ["[PAD]"]},
      "[SEP]": {"id": "[SEP]", "ids": [12], "tokens": ["[SEP]"]},
      "[UNK]": {"id": "[UNK]", "ids": [10], "tokens": ["[UNK]"]}
    }
  },
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let opts = talu_sys::EncodeOptions {
        add_bos: 1,
        ..Default::default()
    };
    // encode_with adds CLS/SEP from post-processor when add_bos=1.
    let tokens = ctx.encode_with("hello", &opts);

    // Expected: [CLS]=11, <content>, [SEP]=12
    // Without template parsing, [SEP] may be selected as [MASK]=14
    let last = *tokens.last().unwrap();
    assert_eq!(
        last, 12,
        "Last token must be [SEP]=12, not [MASK]=14 or any other token. Got: {tokens:?}"
    );
    assert_eq!(
        tokens[0], 11,
        "First token must be [CLS]=11. Got: {tokens:?}"
    );
}

/// TemplateProcessing: empty input must still get CLS + SEP tokens.
///
/// Empty input with CLS/SEP template must still produce [CLS, SEP].
#[test]
fn template_processing_empty_input_gets_cls_sep() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "[UNK]": 10, "[CLS]": 11, "[SEP]": 12, "[PAD]": 13, "[MASK]": 14
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 10, "content": "[UNK]", "special": true},
    {"id": 11, "content": "[CLS]", "special": true},
    {"id": 12, "content": "[SEP]", "special": true},
    {"id": 13, "content": "[PAD]", "special": true},
    {"id": 14, "content": "[MASK]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
  "post_processor": {
    "type": "TemplateProcessing",
    "single": [
      {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}},
      {"SpecialToken": {"id": "[SEP]", "type_id": 0}}
    ],
    "special_tokens": {
      "[CLS]": {"id": "[CLS]", "ids": [11], "tokens": ["[CLS]"]},
      "[MASK]": {"id": "[MASK]", "ids": [14], "tokens": ["[MASK]"]},
      "[PAD]": {"id": "[PAD]", "ids": [13], "tokens": ["[PAD]"]},
      "[SEP]": {"id": "[SEP]", "ids": [12], "tokens": ["[SEP]"]},
      "[UNK]": {"id": "[UNK]", "ids": [10], "tokens": ["[UNK]"]}
    }
  },
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let opts = talu_sys::EncodeOptions {
        add_bos: 1,
        ..Default::default()
    };
    // Empty input should produce just [CLS, SEP]
    let tokens = ctx.encode_with("", &opts);
    assert_eq!(
        tokens,
        vec![11, 12],
        "Empty input must produce [CLS=11, SEP=12], got: {tokens:?}"
    );
}

// ===========================================================================
// Added token rstrip: trailing whitespace consumed by the token
// ===========================================================================
//
// When an added token has rstrip=true, the space (or SentencePiece ▁)
// immediately after it should be consumed into the token's span, so
// the next segment does NOT start with a space/▁.
//
// Added tokens with rstrip=true must consume trailing whitespace
// (including ▁ after normalization) into the token span.

/// Added token with rstrip=true must consume trailing whitespace.
///
/// Uses Prepend+Replace normalizer (SentencePiece-style) so spaces become ▁.
/// The </s> token has rstrip=true, so the ▁ after it should be consumed
/// and NOT appear as a standalone token.
#[test]
fn rstrip_added_token_consumes_trailing_space() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "\u2581": 3,
      "h": 4, "e": 5, "l": 6, "o": 7,
      "w": 8, "r": 9, "d": 10
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": false, "rstrip": true}
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
    let ctx = TokenizerTestContext::from_json(json);
    // Input: </s> followed by space then "world"
    let tokens = ctx.encode("</s> world");

    // Normalized: "▁</s>▁world"
    // </s> has rstrip=true → consumes the trailing ▁ (originally a space)
    // So the segment after </s> is "world" (no leading ▁), with prepend → "▁world"
    //   [</s>=2, ▁=3, w=8, o=7, r=9, l=6, d=10]
    // Without rstrip, the ▁ after </s> is NOT consumed, so:
    //   [</s>=2, ▁=3, ▁=3, w=8, o=7, r=9, l=6, d=10]
    //   (two ▁ tokens — one from rstrip space, one from prepend)

    let sep_pos = tokens.iter().position(|&t| t == 2);
    assert!(
        sep_pos.is_some(),
        "Output must contain </s> token (id=2), got: {tokens:?}"
    );
    let after_sep = &tokens[sep_pos.unwrap() + 1..];
    assert!(
        !after_sep.is_empty(),
        "There must be tokens after </s>, got: {tokens:?}"
    );
    // With rstrip working: exactly one ▁ (from prepend), not two
    assert_eq!(
        after_sep[0], 3,
        "First token after </s> must be ▁ (id=3) from prepend. Got: {tokens:?}"
    );
    // Verify no double ▁
    assert_ne!(
        after_sep.get(1).copied(),
        Some(3),
        "Must not have double ▁ after </s> — rstrip should consume the space. Got: {tokens:?}"
    );
}

// ===========================================================================
// Split(String pattern) must be treated as literal, not regex.
// ===========================================================================
//
// When a Split pretokenizer has pattern type {"String": "..."}, the value
// is a literal string match, not a regex. Some models store a regex-like
// string as a String pattern — which never matches literally, making the
// Split a no-op. The ByteLevel pretokenizer then handles the actual
// splitting via use_regex=true.
//
// Three bugs combined:
// 1. String pattern compiled as regex (interpreted regex metacharacters)
// 2. Isolated behavior only emitted matches, not matches+gaps
// 3. ByteLevel use_regex=true didn't set the GPT-2 regex

/// Split with String pattern must be a literal match, not regex.
///
/// Uses Sequence[Split(String, Isolated), ByteLevel(use_regex=true)].
/// The String pattern never matches literally, so the Split is a no-op.
/// ByteLevel with use_regex=true provides the actual GPT-2 regex splitting.
#[test]
fn split_string_pattern_is_literal_not_regex() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "Hello": 0, "Ġworld": 1, "ĠĠĠĠ": 2, "Ġ": 3, "world": 4,
      "ĠĠĠ": 5, "Ġhello": 6, "hello": 7
    },
    "merges": ["Ġ world"]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": {
    "type": "Sequence",
    "pretokenizers": [
      {
        "type": "Split",
        "pattern": {"String": "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}"},
        "behavior": "Isolated",
        "invert": false
      },
      {
        "type": "ByteLevel",
        "add_prefix_space": false,
        "trim_offsets": true,
        "use_regex": true
      }
    ]
  },
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode("Hello world");

    // The String pattern "[^\r\n\p{L}..." never matches literally in "Hello world".
    // Split(Isolated) passes the entire input through as a gap.
    // ByteLevel(use_regex=true) splits " world" → "Ġworld" (merged).
    // Without the fix, String is compiled as regex → wrong splits.
    assert_eq!(
        tokens,
        vec![0, 1],
        "Expected [Hello=0, Ġworld=1] — Split(String) is no-op, ByteLevel splits. Got: {tokens:?}"
    );
}
