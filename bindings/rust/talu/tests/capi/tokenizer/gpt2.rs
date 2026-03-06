//! GPT-2 pretokenizer (ByteLevel with use_regex) tests.
//!
//! Tests the ByteLevel pretokenizer with `use_regex=true`, which applies the
//! GPT-2 regex pattern to split text before byte-level encoding:
//! - Contractions: 's, 't, 're, 've, 'm, 'll, 'd
//! - Letter groups, digit groups, punctuation groups
//! - Space-prefixed word grouping
//!
//! The critical behavior: BPE merges cannot cross regex split boundaries.

use crate::capi::tokenizer::common::TokenizerTestContext;

fn no_bos() -> talu_sys::EncodeOptions {
    talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    }
}

// ===========================================================================
// Cross-boundary merge prevention (contraction boundary)
// ===========================================================================

/// use_regex=true prevents a merge from firing across a contraction boundary.
///
/// Merge: n+' → n'. With regex, "don't" splits as ["don", "'t"] —
/// n and ' are in different chunks, so the merge cannot fire.
#[test]
fn use_regex_prevents_cross_contraction_merge() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "d": 4, "o": 5, "n": 6, "'": 7, "t": 8,
      "n'": 9
    },
    "merges": ["n '"]
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode_with("don't", &no_bos());
    // GPT-2 regex: "don't" → ["don", "'t"]
    // Merge n+' can't fire across boundary → 5 tokens
    assert_eq!(
        tokens,
        vec![4, 5, 6, 7, 8],
        "use_regex=true must prevent n+' merge across contraction boundary, got: {tokens:?}"
    );
}

/// With use_regex=false, no GPT-2 regex splitting occurs. The entire
/// word is one chunk, so the n+' merge CAN fire.
///
/// "don't" without regex → single chunk → n+' merge fires → [d, o, n', t]
#[test]
fn use_regex_false_allows_cross_contraction_merge() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "d": 4, "o": 5, "n": 6, "'": 7, "t": 8,
      "n'": 9
    },
    "merges": ["n '"]
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode_with("don't", &no_bos());
    // use_regex=false: no contraction splitting → one chunk → n+' merge fires
    // → [d=4, o=5, n'=9, t=8] = 4 tokens
    assert_eq!(
        tokens,
        vec![4, 5, 9, 8],
        "use_regex=false must allow n+' merge (no contraction boundary), got: {tokens:?}"
    );
}

// ===========================================================================
// Cross-boundary merge prevention (digit/letter boundary)
// ===========================================================================

/// use_regex=true prevents a merge from crossing a digit/letter boundary.
///
/// GPT-2 regex splits "abc123" into ["abc", "123"]. A merge c+1 cannot
/// fire because c and 1 are in different chunks.
#[test]
fn use_regex_prevents_cross_digit_letter_merge() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "a": 4, "b": 5, "c": 6, "1": 7, "2": 8, "3": 9,
      "c1": 10
    },
    "merges": ["c 1"]
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode_with("abc123", &no_bos());
    // GPT-2 regex: "abc123" → ["abc", "123"]
    // c+1 merge can't fire → 6 tokens
    assert_eq!(
        tokens,
        vec![4, 5, 6, 7, 8, 9],
        "use_regex must prevent c+1 merge across digit/letter boundary, got: {tokens:?}"
    );
}

/// With use_regex=false, no GPT-2 regex splitting occurs. The entire
/// word is one chunk, so the c+1 merge CAN fire across digit/letter boundary.
///
/// "abc123" without regex → single chunk → c+1 merge fires → [a, b, c1, 2, 3]
#[test]
fn use_regex_false_allows_cross_digit_letter_merge() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "a": 4, "b": 5, "c": 6, "1": 7, "2": 8, "3": 9,
      "c1": 10
    },
    "merges": ["c 1"]
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode_with("abc123", &no_bos());
    // use_regex=false: no digit/letter splitting → one chunk → c+1 merge fires
    // → [a=4, b=5, c1=10, 2=8, 3=9] = 5 tokens
    assert_eq!(
        tokens,
        vec![4, 5, 10, 8, 9],
        "use_regex=false must allow c+1 merge (no digit/letter boundary), got: {tokens:?}"
    );
}

// ===========================================================================
// Cross-boundary merge prevention (space/word boundary)
// ===========================================================================

/// use_regex=true prevents a merge from crossing a space/word boundary.
///
/// GPT-2 regex splits "hello world" into ["hello", " world"]. After
/// byte-to-unicode: ["hello", "Ġworld"]. A merge o+Ġ cannot fire because
/// o and Ġ are in different chunks.
#[test]
fn use_regex_prevents_cross_space_merge() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "h": 4, "e": 5, "l": 6, "o": 7, "w": 8, "r": 9, "d": 10,
      "\u0120": 11,
      "o\u0120": 12
    },
    "merges": ["o \u0120"]
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode_with("hello world", &no_bos());
    // GPT-2 regex: "hello world" → ["hello", " world"]
    // byte_to_unicode: ["hello", "Ġworld"]
    // o+Ġ merge can't fire → 11 tokens
    assert_eq!(
        tokens.len(),
        11,
        "use_regex must prevent o+Ġ merge: 'hello'(5) + 'Ġworld'(6) = 11, got: {tokens:?}"
    );
    // Verify no oĠ merge token appears
    assert!(
        !tokens.contains(&12),
        "oĠ merge token (12) must not appear, got: {tokens:?}"
    );
}

// ===========================================================================
// Contraction splitting: each contraction merged within its chunk
// ===========================================================================

/// Shared fixture for all contraction tests.
///
/// ByteLevel with use_regex=true. Merges produce all 7 contraction tokens.
/// Vocab has individual byte-level chars plus merged contraction tokens.
const CONTRACTION_JSON: &str = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "'": 4, "s": 5, "t": 6, "r": 7, "e": 8, "v": 9, "m": 10, "l": 11, "d": 12,
      "I": 13, "i": 14, "n": 15, "o": 16, "h": 17, "a": 18, "y": 19,
      "'t": 20, "'s": 21, "'d": 22, "'m": 23,
      "'r": 24, "'re": 25,
      "'v": 26, "'ve": 27,
      "'l": 28, "'ll": 29
    },
    "merges": [
      "' t", "' s", "' d", "' m",
      "' r", "'r e",
      "' v", "'v e",
      "' l", "'l l"
    ]
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;

/// 't contraction: "don't" → ["don", "'t"] → [d, o, n, 't].
#[test]
fn contraction_t() {
    let ctx = TokenizerTestContext::from_json(CONTRACTION_JSON);
    let tokens = ctx.encode_with("don't", &no_bos());
    // "don" → [d=12, o=16, n=15], "'t" → merge → ['t=20]
    assert_eq!(
        tokens,
        vec![12, 16, 15, 20],
        "'t contraction: 'don\\'t' → [d, o, n, 't], got: {tokens:?}"
    );
}

/// 's contraction: "it's" → ["it", "'s"] → [i, t, 's].
#[test]
fn contraction_s() {
    let ctx = TokenizerTestContext::from_json(CONTRACTION_JSON);
    let tokens = ctx.encode_with("it's", &no_bos());
    assert_eq!(
        tokens,
        vec![14, 6, 21],
        "'s contraction: 'it\\'s' → [i, t, 's], got: {tokens:?}"
    );
}

/// 're contraction: "they're" → ["they", "'re"] → [t, h, e, y, 're].
///
/// Two-step merge: '+r → 'r, then 'r+e → 're.
#[test]
fn contraction_re() {
    let ctx = TokenizerTestContext::from_json(CONTRACTION_JSON);
    let tokens = ctx.encode_with("they're", &no_bos());
    assert_eq!(
        tokens,
        vec![6, 17, 8, 19, 25],
        "'re contraction: 'they\\'re' → [t, h, e, y, 're], got: {tokens:?}"
    );
}

/// 've contraction: "I've" → ["I", "'ve"] → [I, 've].
///
/// Two-step merge: '+v → 'v, then 'v+e → 've.
#[test]
fn contraction_ve() {
    let ctx = TokenizerTestContext::from_json(CONTRACTION_JSON);
    let tokens = ctx.encode_with("I've", &no_bos());
    assert_eq!(
        tokens,
        vec![13, 27],
        "'ve contraction: 'I\\'ve' → [I, 've], got: {tokens:?}"
    );
}

/// 'm contraction: "I'm" → ["I", "'m"] → [I, 'm].
#[test]
fn contraction_m() {
    let ctx = TokenizerTestContext::from_json(CONTRACTION_JSON);
    let tokens = ctx.encode_with("I'm", &no_bos());
    assert_eq!(
        tokens,
        vec![13, 23],
        "'m contraction: 'I\\'m' → [I, 'm], got: {tokens:?}"
    );
}

/// 'll contraction: "they'll" → ["they", "'ll"] → [t, h, e, y, 'll].
///
/// Two-step merge: '+l → 'l, then 'l+l → 'll.
#[test]
fn contraction_ll() {
    let ctx = TokenizerTestContext::from_json(CONTRACTION_JSON);
    let tokens = ctx.encode_with("they'll", &no_bos());
    assert_eq!(
        tokens,
        vec![6, 17, 8, 19, 29],
        "'ll contraction: 'they\\'ll' → [t, h, e, y, 'll], got: {tokens:?}"
    );
}

/// 'd contraction: "he'd" → ["he", "'d"] → [h, e, 'd].
#[test]
fn contraction_d() {
    let ctx = TokenizerTestContext::from_json(CONTRACTION_JSON);
    let tokens = ctx.encode_with("he'd", &no_bos());
    assert_eq!(
        tokens,
        vec![17, 8, 22],
        "'d contraction: 'he\\'d' → [h, e, 'd], got: {tokens:?}"
    );
}

// ===========================================================================
// Contraction roundtrip
// ===========================================================================

/// Encode→decode roundtrip for all contraction patterns.
#[test]
fn contraction_roundtrip_decode() {
    let ctx = TokenizerTestContext::from_json(CONTRACTION_JSON);
    let opts = no_bos();

    for text in ["don't", "it's", "they're", "I've", "I'm", "they'll", "he'd"] {
        let tokens = ctx.encode_with(text, &opts);
        let decoded = ctx.decode(&tokens);
        assert_eq!(decoded, text, "contraction roundtrip failed for {text:?}");
    }
}
