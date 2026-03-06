//! JSON tokenizer loading edge-case tests.
//!
//! Tests that the JSON parser correctly handles escape sequences in vocab
//! keys, null/missing fields, invalid JSON, and various model type configs.

use std::ffi::c_void;
use std::ptr;

use crate::capi::tokenizer::common::TokenizerTestContext;

fn no_bos() -> talu_sys::EncodeOptions {
    talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    }
}

/// Try to load a tokenizer from JSON; return Ok(handle) or Err(error_code).
fn try_load(json_str: &str) -> Result<*mut c_void, i32> {
    let json = json_str.as_bytes();
    let mut handle: *mut c_void = ptr::null_mut();
    let rc = unsafe {
        talu_sys::talu_tokenizer_create_from_json(
            json.as_ptr(),
            json.len(),
            &mut handle as *mut _ as *mut c_void,
        )
    };
    if rc == 0 && !handle.is_null() {
        Ok(handle)
    } else {
        Err(rc)
    }
}

fn nested_sequence_normalizer(depth: usize, leaf: &str) -> String {
    let mut current = leaf.to_owned();
    for _ in 0..depth {
        current = format!(
            r#"{{"type":"Sequence","normalizers":[{}]}}"#,
            current
        );
    }
    current
}

// ===========================================================================
// Basic loading validation
// ===========================================================================

/// Valid minimal BPE JSON loads without error.
#[test]
fn valid_minimal_bpe_loads() {
    let json = r#"{
  "version": "1.0",
  "model": { "type": "BPE", "vocab": {"a": 0}, "merges": [] },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    let handle = try_load(json).expect("valid BPE JSON must load");
    unsafe { talu_sys::talu_tokenizer_free(handle) };
}

/// HuggingFace BPE tokenizers often emit unsupported-option fields with
/// neutral defaults (`null` / `false`). Those inert values must load; only
/// non-default values should be rejected.
#[test]
fn bpe_hf_default_optional_fields_load() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": "<UNK>",
    "continuing_subword_prefix": null,
    "end_of_word_suffix": null,
    "fuse_unk": false,
    "byte_fallback": false,
    "ignore_merges": false,
    "vocab": {"<UNK>": 0, "h": 1, "e": 2, "he": 3},
    "merges": ["h e"]
  },
  "added_tokens": [{"id": 0, "content": "<UNK>", "special": true}],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
  "post_processor": null,
  "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true}
}"#;
    let handle = try_load(json).expect("neutral HF-emitted BPE option fields must load");
    unsafe { talu_sys::talu_tokenizer_free(handle) };
}

/// Valid minimal WordPiece JSON loads without error.
#[test]
fn valid_minimal_wordpiece_loads() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "vocab": {"[UNK]": 0, "hello": 1}
  },
  "added_tokens": [{"id": 0, "content": "[UNK]", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    let handle = try_load(json).expect("valid WordPiece JSON must load");
    unsafe { talu_sys::talu_tokenizer_free(handle) };
}

/// Valid minimal Unigram JSON loads without error.
#[test]
fn valid_minimal_unigram_loads() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "unk_id": 0,
    "vocab": [["<unk>", 0.0], ["a", -1.0]]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    let handle = try_load(json).expect("valid Unigram JSON must load");
    unsafe { talu_sys::talu_tokenizer_free(handle) };
}

/// Malformed JSON (missing closing brace) returns error.
#[test]
fn invalid_json_returns_error() {
    let json = r#"{ "version": "1.0", "model": { "type": "BPE""#;
    assert!(
        try_load(json).is_err(),
        "malformed JSON must return error"
    );
}

/// Empty string returns error.
#[test]
fn empty_string_returns_error() {
    assert!(try_load("").is_err(), "empty string must return error");
}

/// Missing `model` section must return error.
#[test]
fn missing_model_section_returns_error() {
    let json = r#"{
  "version": "1.0",
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(try_load(json).is_err(), "missing model must be rejected");
}

/// Unknown model type must return error.
#[test]
fn unknown_model_type_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": { "type": "NotAModel", "vocab": {"a": 0}, "merges": [] },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(try_load(json).is_err(), "unknown model type must be rejected");
}

/// BPE configs with duplicate vocab IDs must be rejected.
#[test]
fn bpe_duplicate_vocab_ids_return_error() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"<unk>": 0, "a": 1, "b": 1},
    "merges": []
  },
  "added_tokens": [{"id": 0, "content": "<unk>", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "duplicate vocab IDs must not load (ambiguous id->token mapping)"
    );
}

/// Invalid unpaired-surrogate escape in vocab key must be rejected.
#[test]
fn invalid_unicode_surrogate_escape_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"<unk>": 0, "\uD800": 1},
    "merges": []
  },
  "added_tokens": [{"id": 0, "content": "<unk>", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "unpaired surrogate escape must be rejected"
    );
}

/// Unknown normalizer type must be rejected.
#[test]
fn unknown_normalizer_type_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": { "type": "BPE", "vocab": {"a": 0}, "merges": [] },
  "added_tokens": [],
  "normalizer": { "type": "DoesNotExist" },
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "unknown normalizer type must be rejected"
    );
}

/// Unknown pre_tokenizer type must be rejected.
#[test]
fn unknown_pretokenizer_type_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": { "type": "BPE", "vocab": {"a": 0}, "merges": [] },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": { "type": "DoesNotExist" },
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "unknown pre_tokenizer type must be rejected"
    );
}

/// A deeply nested but valid normalizer tree must load and preserve semantics
/// rather than failing or crashing due to parser/loader recursion.
#[test]
fn deeply_nested_sequence_normalizer_loads_and_normalizes() {
    let normalizer = nested_sequence_normalizer(64, r#"{"type":"Lowercase"}"#);
    let json = format!(
        r#"{{
  "version": "1.0",
  "model": {{
    "type": "BPE",
    "vocab": {{"<unk>": 0, "a": 1, "b": 2, "c": 3}},
    "merges": []
  }},
  "added_tokens": [{{"id": 0, "content": "<unk>", "special": true}}],
  "normalizer": {},
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}}"#,
        normalizer
    );

    let ctx = TokenizerTestContext::from_json(&json);
    let tokens = ctx.encode_with("ABC", &no_bos());
    assert_eq!(
        tokens,
        vec![1, 2, 3],
        "deeply nested lowercase normalizer must still apply semantics correctly"
    );
}

/// Unknown decoder type must be rejected.
#[test]
fn unknown_decoder_type_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": { "type": "BPE", "vocab": {"a": 0}, "merges": [] },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": { "type": "DoesNotExist" }
}"#;
    assert!(try_load(json).is_err(), "unknown decoder type must be rejected");
}

/// TemplateProcessing that references undefined special tokens must be rejected.
#[test]
fn template_processing_missing_special_token_definition_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"H": 4, "i": 5},
    "merges": []
  },
  "added_tokens": [],
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
      {"Sequence": {"id": "B", "type_id": 1}}
    ],
    "special_tokens": {}
  },
  "decoder": {"type": "ByteLevel"}
}"#;
    assert!(
        try_load(json).is_err(),
        "template processor with undefined special token mapping must be rejected"
    );
}

/// Missing model.type must be rejected (no implicit type default).
#[test]
fn missing_model_type_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": { "vocab": {"a": 0}, "merges": [] },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(try_load(json).is_err(), "missing model.type must be rejected");
}

/// BPE model missing vocab must be rejected.
#[test]
fn bpe_missing_vocab_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": { "type": "BPE", "merges": [] },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(try_load(json).is_err(), "BPE without vocab must be rejected");
}

/// Unknown post_processor type must be rejected.
#[test]
fn unknown_postprocessor_type_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": { "type": "BPE", "vocab": {"a": 0}, "merges": [] },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": { "type": "DoesNotExist" },
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "unknown post_processor type must be rejected"
    );
}

/// Split pre_tokenizer without a pattern must be rejected.
#[test]
fn split_pretokenizer_missing_pattern_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": { "type": "BPE", "vocab": {"a": 0}, "merges": [] },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": { "type": "Split", "behavior": "Removed" },
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "Split pre_tokenizer without pattern must be rejected"
    );
}

/// Replace normalizer missing pattern must be rejected.
#[test]
fn replace_normalizer_missing_pattern_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": { "type": "BPE", "vocab": {"a": 0}, "merges": [] },
  "added_tokens": [],
  "normalizer": { "type": "Replace", "content": "x" },
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "Replace normalizer without pattern must be rejected"
    );
}

/// Prepend normalizer missing prepend text must be rejected.
#[test]
fn prepend_normalizer_missing_text_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": { "type": "BPE", "vocab": {"a": 0}, "merges": [] },
  "added_tokens": [],
  "normalizer": { "type": "Prepend" },
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "Prepend normalizer without prepend text must be rejected"
    );
}

/// added_tokens entries missing content must be rejected.
#[test]
fn added_token_missing_content_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": { "type": "BPE", "vocab": {"a": 0}, "merges": [] },
  "added_tokens": [{"id": 10, "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "added token missing content must be rejected"
    );
}

/// added_tokens entries with non-numeric id must be rejected.
#[test]
fn added_token_non_numeric_id_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": { "type": "BPE", "vocab": {"a": 0}, "merges": [] },
  "added_tokens": [{"id": "x", "content": "<x>", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "added token non-numeric id must be rejected"
    );
}

/// Duplicate added token IDs with different content must be rejected.
#[test]
fn added_tokens_duplicate_ids_return_error() {
    let json = r#"{
  "version": "1.0",
  "model": { "type": "BPE", "vocab": {"a": 0}, "merges": [] },
  "added_tokens": [
    {"id": 10, "content": "<a>", "special": true},
    {"id": 10, "content": "<b>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "duplicate added token IDs must be rejected"
    );
}

/// Duplicate added token content with different IDs must be rejected.
#[test]
fn added_tokens_duplicate_content_return_error() {
    let json = r#"{
  "version": "1.0",
  "model": { "type": "BPE", "vocab": {"a": 0}, "merges": [] },
  "added_tokens": [
    {"id": 10, "content": "<a>", "special": true},
    {"id": 11, "content": "<a>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "duplicate added token content with different IDs must be rejected"
    );
}

/// ID collision between model vocab and added_tokens with different content must be rejected.
#[test]
fn vocab_added_token_id_collision_with_different_content_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"a": 0, "b": 1},
    "merges": []
  },
  "added_tokens": [{"id": 1, "content": "<special-b>", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "vocab/added-token ID collision with different content must be rejected"
    );
}

/// BPE vocab IDs must be integers; float IDs are invalid.
#[test]
fn bpe_vocab_float_id_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"<unk>": 0, "a": 1.5},
    "merges": []
  },
  "added_tokens": [{"id": 0, "content": "<unk>", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(try_load(json).is_err(), "float vocab IDs must be rejected");
}

/// BPE vocab IDs must be non-negative.
#[test]
fn bpe_vocab_negative_id_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"<unk>": 0, "a": -1},
    "merges": []
  },
  "added_tokens": [{"id": 0, "content": "<unk>", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(try_load(json).is_err(), "negative vocab IDs must be rejected");
}

/// BPE merges must be an array.
#[test]
fn bpe_merges_non_array_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"a": 0, "b": 1},
    "merges": {"a b": 0}
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(try_load(json).is_err(), "non-array merges must be rejected");
}

/// Array-format BPE merges must contain exactly two string tokens.
#[test]
fn bpe_merge_pair_with_three_items_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"a": 0, "b": 1, "c": 2},
    "merges": [["a", "b", "c"]]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "merge entries with arity != 2 must be rejected"
    );
}

/// Array-format BPE merges must contain only strings.
#[test]
fn bpe_merge_pair_non_string_element_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"a": 0, "b": 1},
    "merges": [["a", 1]]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "merge entries with non-string elements must be rejected"
    );
}

/// String-format BPE merge entries must contain exactly one space separator.
#[test]
fn bpe_merge_string_without_separator_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"a": 0, "b": 1, "ab": 2},
    "merges": ["ab"]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "string merge entries without a pair separator must be rejected"
    );
}

/// Duplicate merge entries are ambiguous and must be rejected.
#[test]
fn bpe_duplicate_merge_entries_return_error() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"a": 0, "b": 1, "ab": 2},
    "merges": ["a b", "a b"]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "duplicate merge pairs must be rejected to avoid rank ambiguity"
    );
}

/// String-format merge entries with extra separators are malformed.
#[test]
fn bpe_merge_string_with_double_space_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"a": 0, "b": 1, "ab": 2},
    "merges": ["a  b"]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "merge strings must have exactly one separator space"
    );
}

/// String-format merge entries must not have leading spaces.
#[test]
fn bpe_merge_string_with_leading_space_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"a": 0, "b": 1, "ab": 2},
    "merges": [" a b"]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "merge strings with leading spaces must be rejected"
    );
}

/// String-format merge entries must not have trailing spaces.
#[test]
fn bpe_merge_string_with_trailing_space_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"a": 0, "b": 1, "ab": 2},
    "merges": ["a b "]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "merge strings with trailing spaces must be rejected"
    );
}

/// Merge rules must reference existing vocab symbols on both sides.
#[test]
fn bpe_merge_referencing_unknown_symbol_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"a": 0, "b": 1, "ab": 2},
    "merges": ["a z"]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "merge entries with unknown symbols must be rejected"
    );
}

/// Unsupported BPE dropout must be rejected rather than silently ignored,
/// because it changes tokenization semantics.
#[test]
fn bpe_unsupported_dropout_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"<unk>": 0, "a": 1},
    "merges": [],
    "dropout": 0.1
  },
  "added_tokens": [{"id": 0, "content": "<unk>", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "unsupported BPE dropout must be rejected, not ignored"
    );
}

/// Unsupported BPE fuse_unk must also be rejected rather than silently
/// accepted, because it materially changes unknown-token behavior.
#[test]
fn bpe_unsupported_fuse_unk_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"<unk>": 0, "a": 1},
    "merges": [],
    "fuse_unk": true
  },
  "added_tokens": [{"id": 0, "content": "<unk>", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "unsupported BPE fuse_unk must be rejected, not ignored"
    );
}

/// Unsupported explicit BPE byte_fallback configuration must also be rejected
/// rather than silently ignored.
#[test]
fn bpe_unsupported_byte_fallback_flag_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"<unk>": 0, "a": 1},
    "merges": [],
    "byte_fallback": true
  },
  "added_tokens": [{"id": 0, "content": "<unk>", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "unsupported explicit BPE byte_fallback flag must be rejected, not ignored"
    );
}

/// BPE must reject unsupported WordPiece-style continuing_subword_prefix rather
/// than silently accepting a field whose semantics are not implemented.
#[test]
fn bpe_unsupported_continuing_subword_prefix_returns_error() {
    let json = r###"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"<unk>": 0, "a": 1},
    "merges": [],
    "continuing_subword_prefix": "##"
  },
  "added_tokens": [{"id": 0, "content": "<unk>", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"###;
    assert!(
        try_load(json).is_err(),
        "unsupported BPE continuing_subword_prefix must be rejected, not ignored"
    );
}

/// BPE must reject unsupported end_of_word_suffix rather than silently
/// accepting segmentation semantics it does not implement.
#[test]
fn bpe_unsupported_end_of_word_suffix_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"<unk>": 0, "a": 1},
    "merges": [],
    "end_of_word_suffix": "</w>"
  },
  "added_tokens": [{"id": 0, "content": "<unk>", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "unsupported BPE end_of_word_suffix must be rejected, not ignored"
    );
}

/// BPE must reject unsupported ignore_merges rather than silently ignoring a
/// flag that changes the core tokenization algorithm.
#[test]
fn bpe_unsupported_ignore_merges_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"<unk>": 0, "a": 1},
    "merges": [],
    "ignore_merges": true
  },
  "added_tokens": [{"id": 0, "content": "<unk>", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "unsupported BPE ignore_merges must be rejected, not ignored"
    );
}

/// added_tokens entries missing id must be rejected.
#[test]
fn added_token_missing_id_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": { "type": "BPE", "vocab": {"a": 0}, "merges": [] },
  "added_tokens": [{"content": "<x>", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "added token missing id must be rejected"
    );
}

/// added_tokens IDs must be non-negative integers.
#[test]
fn added_token_negative_id_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": { "type": "BPE", "vocab": {"a": 0}, "merges": [] },
  "added_tokens": [{"id": -7, "content": "<x>", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "negative added token IDs must be rejected"
    );
}

/// added_tokens IDs must be integer values, not floats.
#[test]
fn added_token_float_id_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": { "type": "BPE", "vocab": {"a": 0}, "merges": [] },
  "added_tokens": [{"id": 3.14, "content": "<x>", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "float added token IDs must be rejected"
    );
}

/// added_tokens entries with empty content are invalid.
#[test]
fn added_token_empty_content_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": { "type": "BPE", "vocab": {"a": 0}, "merges": [] },
  "added_tokens": [{"id": 10, "content": "", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "added token empty content must be rejected"
    );
}

/// Duplicate token keys in vocab are ambiguous and must be rejected.
#[test]
fn duplicate_vocab_token_keys_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"a": 0, "a": 1},
    "merges": []
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "duplicate vocab token keys must be rejected"
    );
}

/// Tokens containing embedded NUL bytes are invalid in C-string based tokenizer internals.
#[test]
fn vocab_token_with_embedded_nul_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"<unk>": 0, "a\u0000b": 1},
    "merges": []
  },
  "added_tokens": [{"id": 0, "content": "<unk>", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "vocab keys with embedded NUL must be rejected"
    );
}

/// added_tokens content containing embedded NUL bytes must be rejected.
#[test]
fn added_token_content_with_embedded_nul_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": { "type": "BPE", "vocab": {"a": 0}, "merges": [] },
  "added_tokens": [{"id": 7, "content": "x\u0000y", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "added token content with embedded NUL must be rejected"
    );
}

/// added_tokens must be an array when present.
#[test]
fn added_tokens_non_array_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": { "type": "BPE", "vocab": {"a": 0}, "merges": [] },
  "added_tokens": {},
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(try_load(json).is_err(), "non-array added_tokens must be rejected");
}

/// normalizer must be object or null.
#[test]
fn normalizer_non_object_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": { "type": "BPE", "vocab": {"a": 0}, "merges": [] },
  "added_tokens": [],
  "normalizer": "Lowercase",
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "non-object normalizer must be rejected"
    );
}

/// pre_tokenizer must be object or null.
#[test]
fn pretokenizer_non_object_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": { "type": "BPE", "vocab": {"a": 0}, "merges": [] },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": "ByteLevel",
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "non-object pre_tokenizer must be rejected"
    );
}

/// post_processor must be object or null.
#[test]
fn postprocessor_non_object_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": { "type": "BPE", "vocab": {"a": 0}, "merges": [] },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": "TemplateProcessing",
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "non-object post_processor must be rejected"
    );
}

/// decoder must be object or null.
#[test]
fn decoder_non_object_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": { "type": "BPE", "vocab": {"a": 0}, "merges": [] },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": "ByteLevel"
}"#;
    assert!(try_load(json).is_err(), "non-object decoder must be rejected");
}

/// TemplateProcessing single field must be an array of template elements.
#[test]
fn template_processing_single_non_array_returns_error() {
    let json = r#"{
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
    "single": "not-an-array",
    "pair": [],
    "special_tokens": {
      "<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]},
      "</s>": {"id": "</s>", "ids": [2], "tokens": ["</s>"]}
    }
  },
  "decoder": {"type": "ByteLevel"}
}"#;
    assert!(
        try_load(json).is_err(),
        "TemplateProcessing.single must be an array"
    );
}

/// TemplateProcessing special_tokens entries must have ids as an integer array.
#[test]
fn template_processing_special_tokens_ids_non_array_returns_error() {
    let json = r#"{
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
      {"Sequence": {"id": "A", "type_id": 0}}
    ],
    "pair": [],
    "special_tokens": {
      "<s>": {"id": "<s>", "ids": 1, "tokens": ["<s>"]},
      "</s>": {"id": "</s>", "ids": [2], "tokens": ["</s>"]}
    }
  },
  "decoder": {"type": "ByteLevel"}
}"#;
    assert!(
        try_load(json).is_err(),
        "TemplateProcessing special_tokens ids must be an array"
    );
}

/// WordPiece vocab must use object form `{token: id}`, not Unigram array form.
#[test]
fn wordpiece_array_vocab_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "vocab": [["[UNK]", 0.0], ["hello", -1.0]]
  },
  "added_tokens": [{"id": 0, "content": "[UNK]", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "WordPiece must reject Unigram-style array vocab"
    );
}

/// Unigram vocab must use array form `[[token, score], ...]`, not object form.
#[test]
fn unigram_object_vocab_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "unk_id": 0,
    "vocab": {"<unk>": 0, "a": 1}
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "Unigram must reject object-form vocab"
    );
}

/// Each Unigram vocab entry must contain exactly `[token, score]`.
#[test]
fn unigram_vocab_entry_missing_score_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "unk_id": 0,
    "vocab": [["<unk>"], ["a", -1.0]]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "Unigram entries missing score must be rejected"
    );
}

/// Unigram scores must be numeric.
#[test]
fn unigram_vocab_entry_non_numeric_score_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "unk_id": 0,
    "vocab": [["<unk>", 0.0], ["a", "bad-score"]]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "Unigram entries with non-numeric score must be rejected"
    );
}

/// Unigram entries with extra fields are malformed.
#[test]
fn unigram_vocab_entry_extra_field_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "unk_id": 0,
    "vocab": [["<unk>", 0.0, 7], ["a", -1.0]]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert!(
        try_load(json).is_err(),
        "Unigram entries with arity != 2 must be rejected"
    );
}

// ===========================================================================
// JSON escape sequences in vocab keys
// ===========================================================================

/// JSON \\uXXXX escape in vocab key decodes to the correct unicode char.
///
/// Vocab key "\\u00E9" is é (U+00E9). The token must decode to "é".
#[test]
fn json_unicode_escape_in_vocab() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"<unk>": 0, "\u00E9": 1, "c": 2, "a": 3, "f": 4},
    "merges": []
  },
  "added_tokens": [{"id": 0, "content": "<unk>", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    let decoded = ctx.decode(&[1]);
    assert_eq!(
        decoded, "é",
        "JSON \\u00E9 must decode to é, got: {decoded:?}"
    );
}

/// JSON \\t escape in vocab key creates a tab character token.
#[test]
fn json_tab_escape_in_vocab() {
    let json = "{\n  \"version\": \"1.0\",\n  \"model\": {\n    \"type\": \"BPE\",\n    \"vocab\": {\"<unk>\": 0, \"\\t\": 1},\n    \"merges\": []\n  },\n  \"added_tokens\": [{\"id\": 0, \"content\": \"<unk>\", \"special\": true}],\n  \"normalizer\": null,\n  \"pre_tokenizer\": null,\n  \"post_processor\": null,\n  \"decoder\": null\n}";
    let ctx = TokenizerTestContext::from_json(json);
    let decoded = ctx.decode(&[1]);
    assert_eq!(
        decoded, "\t",
        "JSON \\t in vocab must decode to tab, got: {decoded:?}"
    );
}

/// JSON \\n escape in vocab key creates a newline character token.
#[test]
fn json_newline_escape_in_vocab() {
    let json = "{\n  \"version\": \"1.0\",\n  \"model\": {\n    \"type\": \"BPE\",\n    \"vocab\": {\"<unk>\": 0, \"\\n\": 1},\n    \"merges\": []\n  },\n  \"added_tokens\": [{\"id\": 0, \"content\": \"<unk>\", \"special\": true}],\n  \"normalizer\": null,\n  \"pre_tokenizer\": null,\n  \"post_processor\": null,\n  \"decoder\": null\n}";
    let ctx = TokenizerTestContext::from_json(json);
    let decoded = ctx.decode(&[1]);
    assert_eq!(
        decoded, "\n",
        "JSON \\n in vocab must decode to newline, got: {decoded:?}"
    );
}

/// JSON \\\\ escape in vocab key creates a single backslash token.
#[test]
fn json_backslash_escape_in_vocab() {
    let json = "{\n  \"version\": \"1.0\",\n  \"model\": {\n    \"type\": \"BPE\",\n    \"vocab\": {\"<unk>\": 0, \"\\\\\": 1},\n    \"merges\": []\n  },\n  \"added_tokens\": [{\"id\": 0, \"content\": \"<unk>\", \"special\": true}],\n  \"normalizer\": null,\n  \"pre_tokenizer\": null,\n  \"post_processor\": null,\n  \"decoder\": null\n}";
    let ctx = TokenizerTestContext::from_json(json);
    let decoded = ctx.decode(&[1]);
    assert_eq!(
        decoded, "\\",
        "JSON \\\\ in vocab must decode to single backslash, got: {decoded:?}"
    );
}

/// JSON \\u0120 escape (GPT-2 space token Ġ) in vocab key.
#[test]
fn json_gpt2_space_unicode_escape() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"<unk>": 0, "\u0120": 1},
    "merges": []
  },
  "added_tokens": [{"id": 0, "content": "<unk>", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    // Ġ (U+0120) is the GPT-2 byte-to-unicode mapping for space (0x20).
    let decoded = ctx.decode(&[1]);
    assert_eq!(decoded.len(), 2, "Ġ is 2 UTF-8 bytes, got: {decoded:?}");
}

// ===========================================================================
// Null/empty field handling
// ===========================================================================

/// BPE with empty merges list is valid (no merge rules).
#[test]
fn empty_merges_is_valid() {
    let json = r#"{
  "version": "1.0",
  "model": { "type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": [] },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    // Each char stays as individual token
    let tokens = ctx.encode_with("ab", &no_bos());
    assert_eq!(tokens, vec![0, 1], "empty merges: each char is its own token");
}

/// All optional pipeline stages set to null.
#[test]
fn all_null_pipeline_stages() {
    let json = r#"{
  "version": "1.0",
  "model": { "type": "BPE", "vocab": {"a": 0}, "merges": [] },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    let handle = try_load(json).expect("all-null pipeline must load");
    unsafe { talu_sys::talu_tokenizer_free(handle) };
}

/// Empty added_tokens array is valid.
#[test]
fn empty_added_tokens() {
    let json = r#"{
  "version": "1.0",
  "model": { "type": "BPE", "vocab": {"hello": 0, "world": 1}, "merges": [] },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    let handle = try_load(json).expect("empty added_tokens must load");
    unsafe { talu_sys::talu_tokenizer_free(handle) };
}

// ===========================================================================
// Non-contiguous and large token IDs
// ===========================================================================

/// Vocab with non-contiguous IDs loads and encodes correctly.
#[test]
fn non_contiguous_vocab_ids() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"<unk>": 0, "hello": 100, "world": 200},
    "merges": []
  },
  "added_tokens": [{"id": 0, "content": "<unk>", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    let decoded = ctx.decode(&[100]);
    assert_eq!(decoded, "hello", "non-contiguous ID 100 must map to 'hello'");
    let decoded = ctx.decode(&[200]);
    assert_eq!(decoded, "world", "non-contiguous ID 200 must map to 'world'");
}

// ===========================================================================
// Multi-char vocab keys with JSON escapes
// ===========================================================================

/// Vocab key with embedded JSON escapes produces correct multi-char token.
///
/// Key "a\\tb" (with JSON \\t) becomes "a<tab>b" — 3 characters.
#[test]
fn json_escape_in_multichar_vocab_key() {
    let json = "{\n  \"version\": \"1.0\",\n  \"model\": {\n    \"type\": \"BPE\",\n    \"vocab\": {\"<unk>\": 0, \"a\\tb\": 1},\n    \"merges\": []\n  },\n  \"added_tokens\": [{\"id\": 0, \"content\": \"<unk>\", \"special\": true}],\n  \"normalizer\": null,\n  \"pre_tokenizer\": null,\n  \"post_processor\": null,\n  \"decoder\": null\n}";
    let ctx = TokenizerTestContext::from_json(json);
    let decoded = ctx.decode(&[1]);
    assert_eq!(
        decoded, "a\tb",
        "vocab key 'a\\\\tb' must decode to 'a<tab>b', got: {decoded:?}"
    );
}

/// Vocab key with \\u2581 (SentencePiece metaspace ▁) loads and decodes.
///
/// With decoder=null, the raw token text should be returned as-is.
/// ▁ (U+2581) should NOT be converted to space without a Metaspace decoder.
#[test]
fn json_metaspace_unicode_escape() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"<unk>": 0, "\u2581hello": 1},
    "merges": []
  },
  "added_tokens": [{"id": 0, "content": "<unk>", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    let decoded = ctx.decode(&[1]);
    // decoder=null: raw token text must be returned, ▁ preserved as-is
    assert_eq!(
        decoded, "\u{2581}hello",
        "with null decoder, \\u2581 must be preserved as ▁, got: {decoded:?}"
    );
}
