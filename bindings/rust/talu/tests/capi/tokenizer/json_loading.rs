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

// Keep in sync with core/src/tokenizer/loader.zig.
const MAX_JSON_PIPELINE_DEPTH: usize = 128;

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

fn assert_rejected(json_str: &str, expected: talu_sys::ErrorCode, context: &str) {
    let rc = try_load(json_str).expect_err(context);
    assert_eq!(
        rc, expected as i32,
        "{context}: expected {:?} ({}) but got rc={}",
        expected, expected as i32, rc
    );
}

fn assert_rejected_valid_json(json_str: &str, expected: talu_sys::ErrorCode, context: &str) {
    serde_json::from_str::<serde_json::Value>(json_str)
        .expect("test fixture must be syntactically valid JSON");
    assert_rejected(json_str, expected, context);
}

fn max_object_depth(json: &str) -> usize {
    let bytes = json.as_bytes();
    let mut i = 0usize;
    let mut depth = 0usize;
    let mut max_depth = 0usize;

    while i < bytes.len() {
        match bytes[i] {
            b'"' => {
                i += 1;
                while i < bytes.len() {
                    if bytes[i] == b'\\' && i + 1 < bytes.len() {
                        i += 2;
                        continue;
                    }
                    if bytes[i] == b'"' {
                        i += 1;
                        break;
                    }
                    i += 1;
                }
            }
            b'{' => {
                depth += 1;
                max_depth = max_depth.max(depth);
                i += 1;
            }
            b'}' => {
                depth = depth.saturating_sub(1);
                i += 1;
            }
            _ => i += 1,
        }
    }
    max_depth
}

fn nested_sequence_normalizer(depth: usize, leaf: &str) -> String {
    let mut current = leaf.to_owned();
    for _ in 0..depth {
        current = format!(r#"{{"type":"Sequence","normalizers":[{}]}}"#, current);
    }
    current
}

fn nested_sequence_pretokenizer(depth: usize, leaf: &str) -> String {
    let mut current = leaf.to_owned();
    for _ in 0..depth {
        current = format!(r#"{{"type":"Sequence","pretokenizers":[{}]}}"#, current);
    }
    current
}

fn nested_sequence_postprocessor(depth: usize, leaf: &str) -> String {
    let mut current = leaf.to_owned();
    for _ in 0..depth {
        current = format!(r#"{{"type":"Sequence","processors":[{}]}}"#, current);
    }
    current
}

fn nested_sequence_decoder(depth: usize, leaf: &str) -> String {
    let mut current = leaf.to_owned();
    for _ in 0..depth {
        current = format!(r#"{{"type":"Sequence","decoders":[{}]}}"#, current);
    }
    current
}

fn nested_sequence_wordpiece_decoder(depth: usize, cleanup: bool) -> String {
    let mut current = format!(
        r###"{{"type":"WordPiece","prefix":"##","cleanup":{}}}"###,
        if cleanup { "true" } else { "false" }
    );
    for _ in 0..depth {
        current = format!(r#"{{"type":"Sequence","decoders":[{}]}}"#, current);
    }
    current
}

fn nested_sequence_metaspace_decoder(depth: usize, add_prefix_space: bool) -> String {
    let mut current = format!(
        r#"{{"type":"Metaspace","replacement":"▁","add_prefix_space":{}}}"#,
        if add_prefix_space { "true" } else { "false" }
    );
    for _ in 0..depth {
        current = format!(r#"{{"type":"Sequence","decoders":[{}]}}"#, current);
    }
    current
}

#[test]
fn nested_sequence_helpers_object_depth_matches_requested_depth() {
    let depth = 7usize;
    let normalizer = nested_sequence_normalizer(depth, r#"{"type":"Lowercase"}"#);
    let pretokenizer = nested_sequence_pretokenizer(depth, r#"{"type":"Whitespace"}"#);
    let postprocessor = nested_sequence_postprocessor(depth, r#"{"type":"ByteLevel"}"#);
    let decoder = nested_sequence_decoder(depth, r#"{"type":"ByteLevel"}"#);

    // One object for each Sequence wrapper plus one object for the leaf node.
    let expected = depth + 1;
    assert_eq!(max_object_depth(&normalizer), expected);
    assert_eq!(max_object_depth(&pretokenizer), expected);
    assert_eq!(max_object_depth(&postprocessor), expected);
    assert_eq!(max_object_depth(&decoder), expected);
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

/// Extremely sparse BPE vocab IDs must be rejected up front rather than
/// attempting to allocate an `id_to_token` table near the attacker-controlled
/// max ID.
#[test]
fn bpe_astronomically_sparse_vocab_id_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"<unk>": 0, "boom": 2147483647},
    "merges": []
  },
  "added_tokens": [{"id": 0, "content": "<unk>", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "astronomically sparse BPE vocab IDs must be rejected deterministically",
    );
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

/// Extremely sparse WordPiece vocab IDs must also be rejected before building
/// the dense ID lookup table.
#[test]
fn wordpiece_astronomically_sparse_vocab_id_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "vocab": {"[UNK]": 0, "boom": 2147483647}
  },
  "added_tokens": [{"id": 0, "content": "[UNK]", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "astronomically sparse WordPiece vocab IDs must be rejected deterministically",
    );
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
    assert_rejected(
        json,
        talu_sys::ErrorCode::InternalError,
        "malformed JSON must return error",
    );
}

/// Empty string returns error.
#[test]
fn empty_string_returns_error() {
    assert_rejected(
        "",
        talu_sys::ErrorCode::InvalidArgument,
        "empty string must return error",
    );
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "missing model must be rejected",
    );
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "unknown model type must be rejected",
    );
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "duplicate vocab IDs must not load (ambiguous id->token mapping)",
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
    assert_rejected(
        json,
        talu_sys::ErrorCode::InternalError,
        "unpaired surrogate escape must be rejected",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "unknown normalizer type must be rejected",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "unknown pre_tokenizer type must be rejected",
    );
}

/// A deeply nested but valid normalizer tree must load and preserve semantics
/// rather than failing or crashing due to parser/loader recursion.
#[test]
fn deeply_nested_sequence_normalizer_loads_and_normalizes() {
    let normalizer = nested_sequence_normalizer(MAX_JSON_PIPELINE_DEPTH, r#"{"type":"Lowercase"}"#);
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

/// Extremely deep Sequence normalizers must be rejected deterministically
/// instead of recursing until stack exhaustion.
#[test]
fn deeply_nested_sequence_normalizer_over_limit_returns_error() {
    let normalizer =
        nested_sequence_normalizer(MAX_JSON_PIPELINE_DEPTH + 1, r#"{"type":"Lowercase"}"#);
    let json = format!(
        r#"{{
  "version": "1.0",
  "model": {{
    "type": "BPE",
    "vocab": {{"<unk>": 0, "a": 1}},
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
    assert_rejected(
        &json,
        talu_sys::ErrorCode::InternalError,
        "over-limit normalizer nesting must return an error, not overflow",
    );
}

/// Sequence normalizers exactly at the supported nesting budget must still load.
#[test]
fn deeply_nested_sequence_normalizer_at_limit_loads() {
    let normalizer = nested_sequence_normalizer(MAX_JSON_PIPELINE_DEPTH, r#"{"type":"Lowercase"}"#);
    let json = format!(
        r#"{{
  "version": "1.0",
  "model": {{
    "type": "BPE",
    "vocab": {{"<unk>": 0, "a": 1}},
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
    let handle = try_load(&json).expect("at-limit normalizer nesting must load");
    unsafe { talu_sys::talu_tokenizer_free(handle) };
}

/// Extremely deep Sequence pre-tokenizers must be rejected deterministically
/// instead of recursing until stack exhaustion.
#[test]
fn deeply_nested_sequence_pretokenizer_over_limit_returns_error() {
    let pre_tokenizer =
        nested_sequence_pretokenizer(MAX_JSON_PIPELINE_DEPTH + 1, r#"{"type":"Whitespace"}"#);
    let json = format!(
        r#"{{
  "version": "1.0",
  "model": {{
    "type": "BPE",
    "vocab": {{"<unk>": 0, "a": 1}},
    "merges": []
  }},
  "added_tokens": [{{"id": 0, "content": "<unk>", "special": true}}],
  "normalizer": null,
  "pre_tokenizer": {},
  "post_processor": null,
  "decoder": null
}}"#,
        pre_tokenizer
    );
    assert_rejected(
        &json,
        talu_sys::ErrorCode::InternalError,
        "over-limit pretokenizer nesting must return an error, not overflow",
    );
}

/// Sequence pre-tokenizers exactly at the supported nesting budget must still load.
#[test]
fn deeply_nested_sequence_pretokenizer_at_limit_loads() {
    let pre_tokenizer =
        nested_sequence_pretokenizer(MAX_JSON_PIPELINE_DEPTH, r#"{"type":"Whitespace"}"#);
    let json = format!(
        r#"{{
  "version": "1.0",
  "model": {{
    "type": "BPE",
    "vocab": {{"<unk>": 0, "a": 1}},
    "merges": []
  }},
  "added_tokens": [{{"id": 0, "content": "<unk>", "special": true}}],
  "normalizer": null,
  "pre_tokenizer": {},
  "post_processor": null,
  "decoder": null
}}"#,
        pre_tokenizer
    );
    let handle = try_load(&json).expect("at-limit pretokenizer nesting must load");
    unsafe { talu_sys::talu_tokenizer_free(handle) };
}

/// Extremely deep Sequence post-processors must be rejected deterministically
/// instead of recursing until stack exhaustion.
#[test]
fn deeply_nested_sequence_postprocessor_over_limit_returns_error() {
    let post_processor =
        nested_sequence_postprocessor(MAX_JSON_PIPELINE_DEPTH + 1, r#"{"type":"ByteLevel"}"#);
    let json = format!(
        r#"{{
  "version": "1.0",
  "model": {{
    "type": "BPE",
    "vocab": {{"<unk>": 0, "a": 1}},
    "merges": []
  }},
  "added_tokens": [{{"id": 0, "content": "<unk>", "special": true}}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": {},
  "decoder": null
}}"#,
        post_processor
    );
    assert_rejected(
        &json,
        talu_sys::ErrorCode::InternalError,
        "over-limit postprocessor nesting must return an error, not overflow",
    );
}

/// Sequence post-processors exactly at the supported nesting budget must still load.
#[test]
fn deeply_nested_sequence_postprocessor_at_limit_loads() {
    let post_processor =
        nested_sequence_postprocessor(MAX_JSON_PIPELINE_DEPTH, r#"{"type":"ByteLevel"}"#);
    let json = format!(
        r#"{{
  "version": "1.0",
  "model": {{
    "type": "BPE",
    "vocab": {{"<unk>": 0, "a": 1}},
    "merges": []
  }},
  "added_tokens": [{{"id": 0, "content": "<unk>", "special": true}}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": {},
  "decoder": null
}}"#,
        post_processor
    );
    let handle = try_load(&json).expect("at-limit postprocessor nesting must load");
    unsafe { talu_sys::talu_tokenizer_free(handle) };
}

/// Sequence decoders above the supported nesting budget must return a typed
/// loader error instead of recursing indefinitely.
#[test]
fn deeply_nested_sequence_decoder_over_limit_returns_error() {
    let decoder = nested_sequence_decoder(MAX_JSON_PIPELINE_DEPTH + 1, r#"{"type":"ByteLevel"}"#);
    let json = format!(
        r#"{{
  "version": "1.0",
  "model": {{
    "type": "BPE",
    "vocab": {{"<unk>": 0, "a": 1}},
    "merges": []
  }},
  "added_tokens": [{{"id": 0, "content": "<unk>", "special": true}}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": {}
}}"#,
        decoder
    );
    assert_rejected(
        &json,
        talu_sys::ErrorCode::InternalError,
        "over-limit decoder nesting must return an error, not overflow",
    );
}

/// Sequence decoders exactly at the supported nesting budget must still load.
#[test]
fn deeply_nested_sequence_decoder_at_limit_loads() {
    let decoder = nested_sequence_decoder(MAX_JSON_PIPELINE_DEPTH, r#"{"type":"ByteLevel"}"#);
    let json = format!(
        r#"{{
  "version": "1.0",
  "model": {{
    "type": "BPE",
    "vocab": {{"<unk>": 0, "a": 1}},
    "merges": []
  }},
  "added_tokens": [{{"id": 0, "content": "<unk>", "special": true}}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": {}
}}"#,
        decoder
    );
    let handle = try_load(&json).expect("at-limit decoder nesting must load");
    unsafe { talu_sys::talu_tokenizer_free(handle) };
}

/// Over-limit nested Metaspace decoder trees must also be rejected, not just
/// generic ByteLevel decoder leaves.
#[test]
fn deeply_nested_sequence_metaspace_decoder_over_limit_returns_error() {
    let decoder = nested_sequence_metaspace_decoder(MAX_JSON_PIPELINE_DEPTH + 1, true);
    let json = format!(
        r#"{{
  "version": "1.0",
  "model": {{
    "type": "BPE",
    "vocab": {{"<unk>": 0, "a": 1}},
    "merges": []
  }},
  "added_tokens": [{{"id": 0, "content": "<unk>", "special": true}}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": {}
}}"#,
        decoder
    );
    assert_rejected(
        &json,
        talu_sys::ErrorCode::InternalError,
        "over-limit nested Metaspace decoder must return an error, not overflow or load",
    );
}

/// Over-limit nested WordPiece decoder trees must also be rejected.
#[test]
fn deeply_nested_sequence_wordpiece_decoder_over_limit_returns_error() {
    let decoder = nested_sequence_wordpiece_decoder(MAX_JSON_PIPELINE_DEPTH + 1, false);
    let json = format!(
        r#"{{
  "version": "1.0",
  "model": {{
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "vocab": {{"[UNK]": 0, "a": 1}}
  }},
  "added_tokens": [{{"id": 0, "content": "[UNK]", "special": true}}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": {}
}}"#,
        decoder
    );
    assert_rejected(
        &json,
        talu_sys::ErrorCode::InternalError,
        "over-limit nested WordPiece decoder must return an error, not overflow or load",
    );
}

/// At-limit nested Metaspace decoder trees must still load.
#[test]
fn deeply_nested_sequence_metaspace_decoder_at_limit_loads() {
    let decoder = nested_sequence_metaspace_decoder(MAX_JSON_PIPELINE_DEPTH, true);
    let json = format!(
        r#"{{
  "version": "1.0",
  "model": {{
    "type": "BPE",
    "vocab": {{"<unk>": 0, "a": 1}},
    "merges": []
  }},
  "added_tokens": [{{"id": 0, "content": "<unk>", "special": true}}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": {}
}}"#,
        decoder
    );
    let handle = try_load(&json).expect("at-limit nested Metaspace decoder must load");
    unsafe { talu_sys::talu_tokenizer_free(handle) };
}

/// At-limit nested WordPiece decoder trees must still load.
#[test]
fn deeply_nested_sequence_wordpiece_decoder_at_limit_loads() {
    let decoder = nested_sequence_wordpiece_decoder(MAX_JSON_PIPELINE_DEPTH, false);
    let json = format!(
        r#"{{
  "version": "1.0",
  "model": {{
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "vocab": {{"[UNK]": 0, "a": 1}}
  }},
  "added_tokens": [{{"id": 0, "content": "[UNK]", "special": true}}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": {}
}}"#,
        decoder
    );
    let handle = try_load(&json).expect("at-limit nested WordPiece decoder must load");
    unsafe { talu_sys::talu_tokenizer_free(handle) };
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "unknown decoder type must be rejected",
    );
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "template processor with undefined special token mapping must be rejected",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "missing model.type must be rejected",
    );
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "BPE without vocab must be rejected",
    );
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "unknown post_processor type must be rejected",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "Split pre_tokenizer without pattern must be rejected",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "Replace normalizer without pattern must be rejected",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "Prepend normalizer without prepend text must be rejected",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "added token missing content must be rejected",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "added token non-numeric id must be rejected",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "duplicate added token IDs must be rejected",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "duplicate added token content with different IDs must be rejected",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "vocab/added-token ID collision with different content must be rejected",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "float vocab IDs must be rejected",
    );
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "negative vocab IDs must be rejected",
    );
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "non-array merges must be rejected",
    );
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "merge entries with arity != 2 must be rejected",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "merge entries with non-string elements must be rejected",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "string merge entries without a pair separator must be rejected",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "duplicate merge pairs must be rejected to avoid rank ambiguity",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "merge strings must have exactly one separator space",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "merge strings with leading spaces must be rejected",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "merge strings with trailing spaces must be rejected",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "merge entries with unknown symbols must be rejected",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "unsupported BPE dropout must be rejected, not ignored",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "unsupported BPE fuse_unk must be rejected, not ignored",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "unsupported explicit BPE byte_fallback flag must be rejected, not ignored",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "unsupported BPE continuing_subword_prefix must be rejected, not ignored",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "unsupported BPE end_of_word_suffix must be rejected, not ignored",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "unsupported BPE ignore_merges must be rejected, not ignored",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "added token missing id must be rejected",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "negative added token IDs must be rejected",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "float added token IDs must be rejected",
    );
}

/// Empty added-token content is ignored rather than rejected.
///
/// This matches the runtime contract exercised by the added-tokens suite:
/// loading must succeed, and the empty entry must not interfere with normal
/// encoding behavior.
#[test]
fn added_token_empty_content_is_ignored() {
    let json = r#"{
  "version": "1.0",
  "model": { "type": "BPE", "vocab": {"a": 0}, "merges": [] },
  "added_tokens": [{"id": 10, "content": "", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    let handle = try_load(json).expect("empty added-token content must be ignored");
    unsafe { talu_sys::talu_tokenizer_free(handle) };
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "duplicate vocab token keys must be rejected",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "vocab keys with embedded NUL must be rejected",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "added token content with embedded NUL must be rejected",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "non-array added_tokens must be rejected",
    );
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "non-object normalizer must be rejected",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "non-object pre_tokenizer must be rejected",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "non-object post_processor must be rejected",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "non-object decoder must be rejected",
    );
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "TemplateProcessing.single must be an array",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "TemplateProcessing special_tokens ids must be an array",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "WordPiece must reject Unigram-style array vocab",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "Unigram must reject object-form vocab",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "Unigram entries missing score must be rejected",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "Unigram entries with non-numeric score must be rejected",
    );
}

/// Unigram must reject non-finite scores (NaN) instead of accepting poisoned
/// DP weights.
#[test]
fn unigram_vocab_entry_nan_score_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "unk_id": 0,
    "vocab": [["<unk>", 0.0], ["a", NaN]]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert_rejected(
        json,
        talu_sys::ErrorCode::InternalError,
        "Unigram NaN scores must be rejected",
    );
}

/// Unigram must reject non-finite scores (+Infinity) to avoid unstable
/// behavior from malformed model files.
#[test]
fn unigram_vocab_entry_infinity_score_returns_error() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "unk_id": 0,
    "vocab": [["<unk>", 0.0], ["a", Infinity]]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    assert_rejected(
        json,
        talu_sys::ErrorCode::InternalError,
        "Unigram +Infinity scores must be rejected",
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
    assert_rejected_valid_json(
        json,
        talu_sys::ErrorCode::InternalError,
        "Unigram entries with arity != 2 must be rejected",
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
    assert_eq!(
        tokens,
        vec![0, 1],
        "empty merges: each char is its own token"
    );
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
    assert_eq!(
        decoded, "hello",
        "non-contiguous ID 100 must map to 'hello'"
    );
    let decoded = ctx.decode(&[200]);
    assert_eq!(
        decoded, "world",
        "non-contiguous ID 200 must map to 'world'"
    );
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
