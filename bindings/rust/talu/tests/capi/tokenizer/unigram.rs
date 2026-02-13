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
    assert_eq!(rc, 0, "Unigram tokenizer must load from JSON (got error {rc})");
    assert!(!handle.is_null());
    unsafe { talu_sys::talu_tokenizer_free(handle) };
}
