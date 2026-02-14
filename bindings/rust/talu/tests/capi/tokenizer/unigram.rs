//! Unigram (SentencePiece) model tests.
//!
//! Tests specific to the Unigram tokenization model used by SentencePiece-based
//! models (T5, Llama, ALBERT, XLNet, etc.).
//!
//! ## Known bugs (two layers)
//!
//! **Loading:** `load_from_slice_streaming` uses a fast-path vocab parser that
//! only handles object-style vocabs `{"token": id}`. Unigram uses array-style
//! `[["token", score], ...]` which is silently skipped → empty vocab → load
//! fails with error 999. All decode tests below are blocked by this.
//!
//! **Decode (once loading is fixed):** the Unigram decode path uses a naive
//! space-join that doesn't apply ▁→space replacement, strip behavior, or
//! `skip_special_tokens`.

use crate::capi::tokenizer::common::TokenizerTestContext;

// ---------------------------------------------------------------------------
// Loading: Unigram JSON fails to load
// ---------------------------------------------------------------------------

/// Unigram tokenizer JSON must load successfully.
///
/// Bug: `load_from_slice_streaming` fast-path vocab parser checks for `{`
/// (object), but Unigram vocab is `[` (array). The array is silently skipped,
/// producing an empty vocab, causing `tokenizer_unigram_create_from_spec`
/// to return null (vocab_len == 0 guard).
#[test]
fn load_from_json() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "unk_token": "<unk>",
    "vocab": [["<unk>", 0.0], ["hello", -2.0], ["world", -3.0]]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"#;
    let bytes = json.as_bytes();
    let mut handle: *mut std::ffi::c_void = std::ptr::null_mut();
    let rc = unsafe {
        talu_sys::talu_tokenizer_create_from_json(
            bytes.as_ptr(),
            bytes.len(),
            &mut handle as *mut _ as *mut std::ffi::c_void,
        )
    };
    assert_eq!(
        rc, 0,
        "Unigram tokenizer must load from JSON (got error {rc})"
    );
    assert!(!handle.is_null());
    unsafe { talu_sys::talu_tokenizer_free(handle) };
}

/// Metaspace decoder with `add_prefix_space: true` must strip the leading space
/// during decode. SentencePiece tokens use ▁ (U+2581) for word boundaries;
/// the first ▁ was added by the pretokenizer and must be removed on decode.
///
/// Bug: `unigram_decode` converts ▁→space but does not check the Metaspace
/// decoder's `add_prefix_space` flag, so the leading space persists.
#[test]
fn metaspace_decode_strips_prefix_space() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "unk_id": 0,
    "vocab": [["<unk>", 0.0], ["\u2581Clark", -2.0], ["\u2581grant", -3.0], ["s", -1.5]]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true },
  "decoder": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true },
  "post_processor": null
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    // Decode [1, 3, 2] → "▁Clark" + "s" + "▁grant" → " Clark" + "s" + " grant"
    // With add_prefix_space strip → "Clarks grant"
    let decoded = ctx.decode(&[1, 3, 2]);
    assert_eq!(
        decoded, "Clarks grant",
        "Metaspace add_prefix_space must strip leading space, got: {decoded:?}"
    );
}

// ---------------------------------------------------------------------------
// Unigram encode: basic encoding must produce correct tokens
// ---------------------------------------------------------------------------
//
// The Unigram model uses the Viterbi algorithm to find the most likely
// segmentation based on token scores (log probabilities). Tokens with higher
// scores are preferred.
//
// Bug: flan-t5-large (Unigram model) produces completely wrong tokens —
// individual byte/character IDs instead of proper Unigram subwords.
// encode("Hello, world!") expected [8774, 6, 296, 55] got [566, 15, ...].
// This suggests the Unigram encoding algorithm is broken or the vocab
// isn't being used correctly during encoding.
//
// Affects: google/flan-t5-large (20/21 encode failures), potentially all
// Unigram-based models (T5, ALBERT, XLNet).

/// Unigram encoder must use Viterbi to find optimal segmentation.
///
/// With a vocab where "▁hello" has a high score, encoding "hello" with
/// Metaspace pretokenizer (which prepends ▁) must produce ["▁hello"],
/// not character-level fallback.
#[test]
fn unigram_encode_produces_correct_tokens() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "unk_id": 0,
    "vocab": [
      ["<unk>", 0.0],
      ["\u2581", -1.0],
      ["\u2581hello", -5.0],
      ["\u2581world", -5.0],
      ["h", -10.0],
      ["e", -10.0],
      ["l", -10.0],
      ["o", -10.0],
      ["w", -10.0],
      ["r", -10.0],
      ["d", -10.0]
    ]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true },
  "post_processor": null,
  "decoder": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true }
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    // "hello" → Metaspace prepend → "▁hello" → Unigram Viterbi → [2] (▁hello)
    let tokens = ctx.encode_with("hello", &opts);
    assert_eq!(
        tokens,
        vec![2],
        "Unigram must use Viterbi to match '▁hello' (score -5.0) over char fallback, got: {tokens:?}"
    );
}

/// Unigram encoder handles multi-word input with Metaspace.
///
/// "hello world" → Metaspace → "▁hello▁world" → Unigram → [▁hello, ▁world].
#[test]
fn unigram_encode_multiword() {
    let json = r#"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "unk_id": 0,
    "vocab": [
      ["<unk>", 0.0],
      ["\u2581", -1.0],
      ["\u2581hello", -5.0],
      ["\u2581world", -5.0],
      ["h", -10.0],
      ["e", -10.0],
      ["l", -10.0],
      ["o", -10.0],
      ["w", -10.0],
      ["r", -10.0],
      ["d", -10.0]
    ]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true },
  "post_processor": null,
  "decoder": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true }
}"#;
    let ctx = TokenizerTestContext::from_json(json);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    // "hello world" → "▁hello ▁world" → Unigram → [▁hello=2, ▁world=3]
    let tokens = ctx.encode_with("hello world", &opts);
    assert_eq!(
        tokens,
        vec![2, 3],
        "Unigram multi-word encoding with Metaspace must produce [▁hello, ▁world], got: {tokens:?}"
    );
}
