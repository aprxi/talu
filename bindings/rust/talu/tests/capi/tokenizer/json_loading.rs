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
