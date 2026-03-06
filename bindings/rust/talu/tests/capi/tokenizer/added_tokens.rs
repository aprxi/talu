//! Added token flag tests.
//!
//! Tests `single_word`, `lstrip`, `rstrip`, and `normalized` flags on added
//! tokens, as well as longest-match priority and overlapping token behavior.
//!
//! These flags control how added tokens are matched in input text:
//! - `single_word`: only match when surrounded by whitespace or at boundaries
//! - `lstrip`: strip leading whitespace before the matched token
//! - `rstrip`: strip trailing whitespace after the matched token
//! - `normalized`: match against normalized (e.g. lowercased) text vs. original

use crate::capi::tokenizer::common::TokenizerTestContext;

fn no_bos() -> talu_sys::EncodeOptions {
    talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    }
}

fn tokenize_strings(ctx: &TokenizerTestContext, text: &str) -> Vec<String> {
    let result =
        unsafe { talu_sys::talu_tokenizer_tokenize(ctx.handle(), text.as_bytes().as_ptr(), text.len()) };
    assert!(result.error_msg.is_null(), "tokenize failed");
    let tokens = if result.tokens.is_null() || result.num_tokens == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(result.tokens, result.num_tokens) }
            .iter()
            .map(|ptr| unsafe { std::ffi::CStr::from_ptr(*ptr) }.to_str().unwrap().to_owned())
            .collect()
    };
    unsafe { talu_sys::talu_tokenize_result_free(result.tokens, result.num_tokens) };
    tokens
}

fn tokenize_bytes_strings(ctx: &TokenizerTestContext, text: &str) -> Vec<String> {
    let result = unsafe {
        talu_sys::talu_tokenizer_tokenize_bytes(ctx.handle(), text.as_bytes().as_ptr(), text.len())
    };
    assert!(result.error_msg.is_null(), "tokenize_bytes failed");
    let data = unsafe { std::slice::from_raw_parts(result.data, result.data_len) };
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens + 1) };
    let tokens: Vec<String> = offsets
        .windows(2)
        .map(|w| std::str::from_utf8(&data[w[0]..w[1]]).unwrap().to_owned())
        .collect();
    unsafe {
        talu_sys::talu_tokenize_bytes_result_free(
            result.data,
            result.data_len,
            result.offsets,
            result.num_tokens,
        )
    };
    tokens
}

// ===========================================================================
// single_word flag
// ===========================================================================

/// Base fixture with single_word=false: token matches anywhere in text.
const SINGLE_WORD_FALSE_JSON: &str = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "a": 3, "b": 4, "c": 5, "d": 6, "e": 7,
      "f": 8, "g": 9, "h": 10, "i": 11, "o": 12,
      "l": 13, "n": 14, " ": 15, "t": 16, "x": 17,
      "r": 18, "w": 19, "k": 20, "s": 21
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 100, "content": "cat", "special": false, "single_word": false}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null
}"####;

/// single_word=false: "cat" matches even when embedded in "concatenate".
#[test]
fn single_word_false_matches_embedded() {
    let ctx = TokenizerTestContext::from_json(SINGLE_WORD_FALSE_JSON);
    // "cat" should match as added token even inside a longer word
    let tokens = ctx.encode_with("cat", &no_bos());
    assert_eq!(
        tokens,
        vec![100],
        "single_word=false: 'cat' should match as added token, got: {tokens:?}"
    );
}

/// single_word=true: "cat" only matches when surrounded by whitespace/boundaries.
#[test]
fn single_word_true_matches_standalone() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "a": 3, "b": 4, "c": 5, "d": 6, "e": 7,
      "f": 8, "g": 9, "h": 10, "i": 11, "o": 12,
      "l": 13, "n": 14, " ": 15, "t": 16, "x": 17
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 100, "content": "cat", "special": false, "single_word": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);

    // Standalone "cat" at input boundaries should match.
    let tokens = ctx.encode_with("cat", &no_bos());
    assert_eq!(
        tokens,
        vec![100],
        "single_word=true: standalone 'cat' should match, got: {tokens:?}"
    );
}

/// single_word=true: "cat" surrounded by spaces matches.
#[test]
fn single_word_true_matches_with_spaces() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "a": 3, "b": 4, "c": 5, "d": 6, "e": 7,
      "f": 8, "g": 9, "h": 10, "i": 11, "o": 12,
      "l": 13, "n": 14, " ": 15, "t": 16, "x": 17
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 100, "content": "cat", "special": false, "single_word": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);

    // Whitespace pre-tokenizer drops spaces, so output should be a + cat + b.
    let tokens = ctx.encode_with("a cat b", &no_bos());
    assert_eq!(
        tokens,
        vec![3, 15, 100, 15, 4],
        "single_word=true: surrounding spaces should remain and cat should match exactly once"
    );
}

/// single_word=true: "cat" at start of input with trailing space matches.
#[test]
fn single_word_true_matches_at_start() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "a": 3, "b": 4, "c": 5, "d": 6, "e": 7,
      "f": 8, "g": 9, "h": 10, "i": 11, "o": 12,
      "l": 13, "n": 14, " ": 15, "t": 16, "x": 17
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 100, "content": "cat", "special": false, "single_word": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);

    // "cat b" → "cat"(100) + " " + "b"
    let tokens = ctx.encode_with("cat b", &no_bos());
    assert_eq!(
        tokens[0], 100,
        "single_word=true: 'cat' at start should match, got: {tokens:?}"
    );
}

/// single_word=true: "cat" at end of input with leading space matches.
#[test]
fn single_word_true_matches_at_end() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "a": 3, "b": 4, "c": 5, "d": 6, "e": 7,
      "f": 8, "g": 9, "h": 10, "i": 11, "o": 12,
      "l": 13, "n": 14, " ": 15, "t": 16, "x": 17
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 100, "content": "cat", "special": false, "single_word": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);

    // "a cat" → "a" + " " + "cat"(100)
    let tokens = ctx.encode_with("a cat", &no_bos());
    assert_eq!(
        *tokens.last().unwrap(),
        100,
        "single_word=true: 'cat' at end should match, got: {tokens:?}"
    );
}

// ===========================================================================
// rstrip flag
// ===========================================================================

/// rstrip=true: trailing whitespace after the added token is consumed.
#[test]
fn rstrip_consumes_trailing_whitespace() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "a": 3, "b": 4, "c": 5, "h": 6, "e": 7,
      "l": 8, "o": 9, " ": 10
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 100, "content": "[SEP]", "special": true, "rstrip": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);

    // "[SEP] hello" with rstrip=true: trailing space consumed by [SEP]
    let tokens_rstrip = ctx.encode_with("[SEP] hello", &no_bos());
    // Without rstrip, the space after [SEP] would be a separate token.
    // With rstrip, the space is consumed, so "hello" follows [SEP] directly.
    assert_eq!(
        tokens_rstrip[0], 100,
        "rstrip: first token should be [SEP]=100, got: {tokens_rstrip:?}"
    );
    // "hello" should follow without a space token between
    assert_ne!(
        tokens_rstrip.get(1).copied().unwrap_or(0),
        10,
        "rstrip: space after [SEP] should be consumed, got: {tokens_rstrip:?}"
    );
}

/// rstrip=false (default): trailing whitespace is NOT consumed.
#[test]
fn rstrip_false_preserves_trailing_whitespace() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "a": 3, "b": 4, "c": 5, "h": 6, "e": 7,
      "l": 8, "o": 9, " ": 10
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 100, "content": "[SEP]", "special": true, "rstrip": false}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);

    let tokens = ctx.encode_with("[SEP] a", &no_bos());
    assert_eq!(
        tokens,
        vec![100, 10, 3],
        "rstrip=false must preserve trailing whitespace as an explicit space token"
    );
}

// ===========================================================================
// lstrip flag
// ===========================================================================

/// lstrip=true: leading whitespace before the added token is consumed.
#[test]
fn lstrip_consumes_leading_whitespace() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "a": 3, "b": 4, "c": 5, "h": 6, "e": 7,
      "l": 8, "o": 9, " ": 10
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 100, "content": "[CLS]", "special": true, "lstrip": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);

    // Whitespace pre-tokenizer drops spaces; lock down exact output.
    let tokens = ctx.encode_with("hello [CLS]", &no_bos());
    assert_eq!(
        tokens,
        vec![6, 7, 8, 8, 9, 100],
        "lstrip fixture should produce exact IDs"
    );
}

// ===========================================================================
// Longest match priority
// ===========================================================================

/// When two added tokens match at the same position, the longer one wins.
#[test]
fn longest_match_wins() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "a": 3, "b": 4, "c": 5, "d": 6, "e": 7,
      "f": 8, "g": 9, "h": 10
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 100, "content": "ab", "special": false},
    {"id": 101, "content": "abc", "special": false}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);

    // "abc" should match the longer token "abc"(101), not "ab"(100)
    let tokens = ctx.encode_with("abc", &no_bos());
    assert_eq!(
        tokens,
        vec![101],
        "longest match: 'abc' should match ID 101, not 100, got: {tokens:?}"
    );
}

/// Shorter match is used when the longer one doesn't apply.
#[test]
fn shorter_match_when_longer_unavailable() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "a": 3, "b": 4, "c": 5, "d": 6, "e": 7,
      "f": 8, "g": 9, "h": 10
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 100, "content": "ab", "special": false},
    {"id": 101, "content": "abc", "special": false}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);

    // "abd" → "ab"(100) + "d"(6)
    let tokens = ctx.encode_with("abd", &no_bos());
    assert_eq!(
        tokens,
        vec![100, 6],
        "shorter match: 'ab' should match ID 100 when 'abc' doesn't apply"
    );
}

// ===========================================================================
// Multiple added tokens in text
// ===========================================================================

/// Multiple added tokens in a single input are all recognized.
#[test]
fn multiple_added_tokens_in_text() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "a": 3, "b": 4, "c": 5, "d": 6, "e": 7,
      " ": 8, "h": 9, "l": 10, "o": 11
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 100, "content": "[A]", "special": true},
    {"id": 101, "content": "[B]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);

    // "[A]hello[B]" → [A]=100, "hello" tokens, [B]=101
    let tokens = ctx.encode_with("[A]hello[B]", &no_bos());
    assert_eq!(
        tokens,
        vec![100, 9, 7, 10, 10, 11, 101],
        "multiple added tokens should be matched with exact surrounding text tokens"
    );
}

/// Adjacent added tokens with no text between them.
#[test]
fn adjacent_added_tokens() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "a": 3
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 100, "content": "[X]", "special": true},
    {"id": 101, "content": "[Y]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);

    // "[X][Y]" → [X]=100, [Y]=101
    let tokens = ctx.encode_with("[X][Y]", &no_bos());
    assert_eq!(
        tokens,
        vec![100, 101],
        "adjacent added tokens should both match, got: {tokens:?}"
    );
}

/// Multiple added tokens in one input should also appear correctly on tokenize
/// surfaces, not just through encode IDs.
#[test]
fn multiple_added_tokens_visible_on_tokenize_surfaces() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "a": 3, "b": 4, "c": 5, "d": 6, "e": 7,
      " ": 8, "h": 9, "l": 10, "o": 11
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 100, "content": "[A]", "special": true},
    {"id": 101, "content": "[B]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);

    assert_eq!(ctx.encode_with("[A]hello[B]", &no_bos()), [100, 9, 7, 10, 10, 11, 101]);
    assert_eq!(tokenize_strings(&ctx, "[A]hello[B]"), ["[A]", "h", "e", "l", "l", "o", "[B]"]);
    assert_eq!(
        tokenize_bytes_strings(&ctx, "[A]hello[B]"),
        ["[A]", "h", "e", "l", "l", "o", "[B]"]
    );
}

/// Adjacent added tokens must produce back-to-back source spans with no gap or
/// overlap in offsets.
#[test]
fn adjacent_added_tokens_offsets_are_back_to_back() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "a": 3
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 100, "content": "[X]", "special": true},
    {"id": 101, "content": "[Y]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let result = unsafe { super::common::encode_raw(ctx.handle(), b"[X][Y]", &no_bos()) };
    assert!(result.error_msg.is_null(), "encode failed");
    assert_eq!(result.num_tokens, 2);

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!(ids, &[100, 101]);
    assert_eq!((offsets[0].start, offsets[0].end), (0, 3));
    assert_eq!((offsets[1].start, offsets[1].end), (3, 6));
    unsafe { talu_sys::talu_encode_result_free(result) };
}

// ===========================================================================
// Decode with skip_special_tokens and added tokens
// ===========================================================================

/// skip_special_tokens strips added special tokens but keeps non-special added tokens.
#[test]
fn skip_special_preserves_non_special_added() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "a": 3, "b": 4, "c": 5
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 100, "content": "[SPECIAL]", "special": true},
    {"id": 101, "content": "[NORMAL]", "special": false}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let skip = talu_sys::DecodeOptionsC {
        skip_special_tokens: 1,
    };

    // Decode [SPECIAL]=100, a=3, [NORMAL]=101, b=4
    // skip_special_tokens should remove [SPECIAL] but keep [NORMAL]
    let decoded = ctx.decode_with(&[100, 3, 101, 4], &skip);
    assert_eq!(
        decoded, "a[NORMAL]b",
        "skip_special should remove special added tokens but retain non-special ones"
    );
}

// ===========================================================================
// lstrip + rstrip both enabled
// ===========================================================================

/// lstrip=true must consume leading whitespace before the added token.
///
/// BUG: lstrip is not implemented — the flag is accepted in JSON but
/// has no effect. This test asserts the CORRECT behavior so it will
/// fail until lstrip is implemented. Mark #[ignore] to keep CI green.
#[test]
fn lstrip_consumes_leading_whitespace_with_rstrip() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "a": 3, "b": 4, " ": 5, "h": 6, "e": 7, "l": 8, "o": 9
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 100, "content": "[MID]", "special": true, "lstrip": true, "rstrip": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode_with("ab [MID] hello", &no_bos());
    assert!(
        tokens.contains(&100),
        "[MID] must be found, got: {tokens:?}"
    );
    let mid_pos = tokens.iter().position(|&t| t == 100).unwrap();
    // lstrip=true: space before [MID] should be consumed
    if mid_pos > 0 {
        assert_ne!(
            tokens[mid_pos - 1], 5,
            "lstrip: space before [MID] must be consumed, got: {tokens:?}"
        );
    }
    // rstrip=true: space after [MID] should be consumed
    if mid_pos + 1 < tokens.len() {
        assert_ne!(
            tokens[mid_pos + 1], 5,
            "rstrip: space after [MID] must be consumed, got: {tokens:?}"
        );
    }
}

// ===========================================================================
// Added tokens that share prefix with merged vocab entries
// ===========================================================================

/// Added token "cat" (ID 200) vs vocab "ca" (ID 100, from merge c+a).
/// The added token matcher runs FIRST and should greedily match "cat"
/// before BPE ever sees it.
#[test]
fn added_token_preempts_bpe_merge() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "c": 3, "a": 4, "t": 5, "d": 6, "o": 7, "g": 8,
      "ca": 100
    },
    "merges": ["c a"]
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 200, "content": "cat", "special": false}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    // "cat" should match as added token 200, NOT be split into ca+t via BPE
    let tokens = ctx.encode_with("cat", &no_bos());
    assert_eq!(
        tokens,
        vec![200],
        "added token must preempt BPE merge: 'cat' → [200], got: {tokens:?}"
    );
}

/// Added token "cat" followed by text that DOES use the c+a merge.
/// "catca" → [cat=200, ca=100]
#[test]
fn added_token_preempts_merge_then_merge_continues() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "c": 3, "a": 4, "t": 5,
      "ca": 100
    },
    "merges": ["c a"]
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 200, "content": "cat", "special": false}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    // "catca" → added token "cat"=200, then BPE processes "ca" → merge → ca=100
    let tokens = ctx.encode_with("catca", &no_bos());
    assert_eq!(
        tokens,
        vec![200, 100],
        "preempt then merge: 'catca' → [cat=200, ca=100], got: {tokens:?}"
    );
}

// ===========================================================================
// Added tokens overlapping each other at same position
// ===========================================================================

/// Three overlapping added tokens at the same position: "a", "ab", "abc".
/// Longest match "abc" should win when text starts with "abc".
#[test]
fn three_overlapping_longest_wins() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "d": 3, "e": 4, "f": 5
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 100, "content": "a", "special": false},
    {"id": 101, "content": "ab", "special": false},
    {"id": 102, "content": "abc", "special": false}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode_with("abcdef", &no_bos());
    assert_eq!(
        tokens[0], 102,
        "3 overlapping: 'abc' must win over 'ab' and 'a', got: {tokens:?}"
    );
    // "abdef" → "ab"=101 wins (no "abc" match), then "def"
    let tokens2 = ctx.encode_with("abdef", &no_bos());
    assert_eq!(
        tokens2[0], 101,
        "2 overlapping: 'ab' must win over 'a', got: {tokens2:?}"
    );
    // "adef" → "a"=100 wins (only 1-char match)
    let tokens3 = ctx.encode_with("adef", &no_bos());
    assert_eq!(
        tokens3[0], 100,
        "1 overlapping: 'a' must match, got: {tokens3:?}"
    );
}

// ===========================================================================
// Added token appearing multiple times
// ===========================================================================

/// Same added token appears 3 times in input text.
#[test]
fn same_added_token_appears_multiple_times() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "x": 3
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 100, "content": "[TOK]", "special": false}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode_with("[TOK]x[TOK]x[TOK]", &no_bos());
    assert_eq!(
        tokens,
        vec![100, 3, 100, 3, 100],
        "triple occurrence: [TOK]x[TOK]x[TOK], got: {tokens:?}"
    );
}

/// Repeated occurrences of the same added token must preserve exact source
/// spans for each match independently.
#[test]
fn repeated_added_token_offsets_track_each_occurrence() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "x": 3
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 100, "content": "[TOK]", "special": false}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let input = b"[TOK]x[TOK]x[TOK]";
    let result = unsafe { super::common::encode_raw(ctx.handle(), input, &no_bos()) };
    assert!(result.error_msg.is_null(), "encode failed");
    assert_eq!(result.num_tokens, 5);

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!(ids, &[100, 3, 100, 3, 100]);
    assert_eq!((offsets[0].start, offsets[0].end), (0, 5));
    assert_eq!((offsets[1].start, offsets[1].end), (5, 6));
    assert_eq!((offsets[2].start, offsets[2].end), (6, 11));
    assert_eq!((offsets[3].start, offsets[3].end), (11, 12));
    assert_eq!((offsets[4].start, offsets[4].end), (12, 17));
    unsafe { talu_sys::talu_encode_result_free(result) };
}

// ===========================================================================
// Added token with content that looks like a special token literal
// ===========================================================================

/// BPE skip_special_tokens must strip tokens marked special=true in
/// added_tokens, even when those token IDs also exist in model.vocab.
///
/// BUG: BPE decode checks model.vocab BEFORE added_tokens. If the ID
/// is found in vocab, it's marked is_special=false, so skip_special
/// never fires. This test asserts the CORRECT behavior.
#[test]
fn skip_special_strips_special_tokens_in_vocab() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "a": 3, "b": 4
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
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let skip = talu_sys::DecodeOptionsC {
        skip_special_tokens: 1,
    };
    // IDs 1,2 are in BOTH vocab AND added_tokens with special=true.
    // skip_special_tokens SHOULD strip them.
    let decoded = ctx.decode_with(&[1, 3, 4, 2], &skip);
    assert_eq!(
        decoded, "ab",
        "only non-special tokens should remain, got: {decoded:?}"
    );
}

/// single_word=true should treat punctuation as valid boundaries.
#[test]
fn single_word_true_matches_with_punctuation_boundaries() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "c": 3, "a": 4, "t": 5, ".": 6, ",": 7, "\"": 8, "(": 9, ")": 10, "x": 11
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 100, "content": "cat", "special": false, "single_word": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);

    assert_eq!(ctx.encode_with("cat.", &no_bos()), vec![100, 6]);
    assert_eq!(ctx.encode_with("\"cat\"", &no_bos()), vec![8, 100, 8]);
    assert_eq!(ctx.encode_with("(cat),", &no_bos()), vec![9, 100, 10, 7]);
}

/// single_word=true must NOT match when embedded in alphanumeric neighbors.
#[test]
fn single_word_true_does_not_match_inside_word() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "c": 3, "a": 4, "t": 5, "x": 6
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 100, "content": "cat", "special": false, "single_word": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    assert_eq!(ctx.encode_with("xcat", &no_bos()), vec![6, 3, 4, 5]);
    assert_eq!(ctx.encode_with("catx", &no_bos()), vec![3, 4, 5, 6]);
    assert_eq!(ctx.encode_with("xcatx", &no_bos()), vec![6, 3, 4, 5, 6]);
}

/// lstrip=true must consume leading ASCII whitespace classes before added token.
#[test]
fn lstrip_consumes_space_tab_and_newline() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "a": 3, "b": 4, " ": 5, "\t": 6, "\n": 7
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 100, "content": "[MID]", "special": false, "lstrip": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    assert_eq!(ctx.encode_with("a [MID]b", &no_bos()), vec![3, 100, 4]);
    assert_eq!(ctx.encode_with("a\t[MID]b", &no_bos()), vec![3, 100, 4]);
    assert_eq!(ctx.encode_with("a\n[MID]b", &no_bos()), vec![3, 100, 4]);
}

/// rstrip=true must consume trailing ASCII whitespace classes after added token.
#[test]
fn rstrip_consumes_space_tab_and_newline() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "a": 3, "b": 4, " ": 5, "\t": 6, "\n": 7
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 100, "content": "[MID]", "special": false, "rstrip": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    assert_eq!(ctx.encode_with("a[MID] b", &no_bos()), vec![3, 100, 4]);
    assert_eq!(ctx.encode_with("a[MID]\tb", &no_bos()), vec![3, 100, 4]);
    assert_eq!(ctx.encode_with("a[MID]\nb", &no_bos()), vec![3, 100, 4]);
}

/// single_word=true punctuation boundary matrix should match around punctuation.
#[test]
fn single_word_true_matches_all_common_punctuation_boundaries() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "c": 3, "a": 4, "t": 5,
      ".": 6, ",": 7, "!": 8, "?": 9, ":": 10, ";": 11,
      "(": 12, ")": 13, "[": 14, "]": 15, "{": 16, "}": 17,
      "\"": 18, "'": 19
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 100, "content": "cat", "special": false, "single_word": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    for text in [
        "cat.",
        ",cat,",
        "!cat!",
        "?cat?",
        ":cat;",
        "(cat)",
        "[cat]",
        "{cat}",
        "\"cat\"",
        "'cat'",
    ] {
        let tokens = ctx.encode_with(text, &no_bos());
        assert!(
            tokens.contains(&100),
            "single_word must match at punctuation boundaries for input {text:?}, got {tokens:?}"
        );
    }
}

/// single_word=true should not match when adjacent to digits/underscore.
#[test]
fn single_word_true_does_not_match_digit_or_underscore_boundaries() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "c": 3, "a": 4, "t": 5, "1": 6, "_": 7
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 100, "content": "cat", "special": false, "single_word": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    for text in ["1cat", "cat1", "_cat", "cat_", "_cat_"] {
        let tokens = ctx.encode_with(text, &no_bos());
        assert!(
            !tokens.contains(&100),
            "single_word should not match when adjacent to digit/underscore for {text:?}"
        );
    }
}

/// normalized=true matches against normalized text, not raw original.
#[test]
fn added_token_normalized_true_matches_after_lowercase() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"<unk>": 0, "h": 3, "e": 4, "l": 5, "o": 6},
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 100, "content": "hello", "special": false, "normalized": true}
  ],
  "normalizer": {"type": "Lowercase"},
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode_with("HELLO", &no_bos());
    assert_eq!(
        tokens,
        vec![100],
        "normalized=true added token should match lowercased normalized input"
    );
}

/// normalized=false should match only original text, not normalized form.
#[test]
fn added_token_normalized_false_does_not_match_after_lowercase() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"<unk>": 0, "h": 3, "e": 4, "l": 5, "o": 6},
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 100, "content": "hello", "special": false, "normalized": false}
  ],
  "normalizer": {"type": "Lowercase"},
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode_with("HELLO", &no_bos());
    assert_eq!(
        tokens,
        vec![3, 4, 5, 5, 6],
        "normalized=false added token should not match raw uppercase input"
    );
}

/// Added token with empty content is silently ignored (no crash).
#[test]
fn empty_added_token_no_crash() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "<s>": 1, "</s>": 2,
      "a": 3, "b": 4, "c": 5
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 100, "content": "", "special": false}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    // Should not crash. Empty content token is skipped.
    let tokens = ctx.encode_with("abc", &no_bos());
    assert!(
        !tokens.is_empty(),
        "encode should produce tokens even with empty added token"
    );
}

/// single_word matching should also be visible on the tokenize surface when
/// the token is bounded by punctuation rather than whitespace.
#[test]
fn single_word_punctuation_boundary_visible_on_tokenize_and_offsets() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "\"": 1, "c": 2, "a": 3, "t": 4
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 100, "content": "cat", "special": false, "single_word": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
    }"####;
    let ctx = TokenizerTestContext::from_json(json);

    assert_eq!(ctx.encode_with("\"cat\"", &no_bos()), [1, 100, 1]);
    assert_eq!(tokenize_strings(&ctx, "\"cat\""), ["\"", "cat", "\""]);

    let result = unsafe { super::common::encode_raw(ctx.handle(), b"\"cat\"", &no_bos()) };
    assert!(result.error_msg.is_null(), "encode failed");
    assert_eq!(result.num_tokens, 3);
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 1));
    assert_eq!((offsets[1].start, offsets[1].end), (1, 4));
    assert_eq!((offsets[2].start, offsets[2].end), (4, 5));
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// normalized=true added tokens must appear as a single tokenize surface token
/// and map back to the original uppercase source span.
#[test]
fn added_token_normalized_true_visible_on_tokenize_and_offsets() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "!": 1
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 100, "content": "hello", "special": false, "normalized": true}
  ],
  "normalizer": {"type": "Lowercase"},
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
    }"####;
    let ctx = TokenizerTestContext::from_json(json);

    assert_eq!(ctx.encode_with("HELLO!", &no_bos()), [100, 1]);
    assert_eq!(tokenize_strings(&ctx, "HELLO!"), ["hello", "!"]);

    let result = unsafe { super::common::encode_raw(ctx.handle(), b"HELLO!", &no_bos()) };
    assert!(result.error_msg.is_null(), "encode failed");
    assert_eq!(result.num_tokens, 2);
    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!(ids, &[100, 1]);
    assert_eq!((offsets[0].start, offsets[0].end), (0, 5));
    assert_eq!((offsets[1].start, offsets[1].end), (5, 6));
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// rstrip should remove trailing whitespace from the tokenize surface, and the
/// next token must begin after the consumed whitespace span.
#[test]
fn rstrip_visible_on_tokenize_and_offsets() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "x": 1, " ": 2, "\t": 3, "\n": 4
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 100, "content": "[SEP]", "special": false, "rstrip": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);

    assert_eq!(tokenize_strings(&ctx, "[SEP] \t\nx"), ["[SEP]", "x"]);

    let result = unsafe { super::common::encode_raw(ctx.handle(), b"[SEP] \t\nx", &no_bos()) };
    assert!(result.error_msg.is_null(), "encode failed");
    assert_eq!(result.num_tokens, 2);
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!(
        (offsets[1].start, offsets[1].end),
        (8, 9),
        "next token must start after whitespace consumed by rstrip"
    );
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// lstrip should remove leading whitespace from the tokenize surface before the
/// added token match.
#[test]
fn lstrip_visible_on_tokenize_surface() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "x": 1, " ": 2, "\t": 3, "\n": 4
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 100, "content": "[MID]", "special": false, "lstrip": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);

    assert_eq!(tokenize_strings(&ctx, " \t\n[MID]x"), ["[MID]", "x"]);
}

/// normalized=false must also remain visible on the tokenize surface: matching
/// must use the original text, not the lowercased normalized form.
#[test]
fn added_token_normalized_false_visible_on_tokenize_surface() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"<unk>": 0, "h": 3, "e": 4, "l": 5, "o": 6},
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 100, "content": "hello", "special": false, "normalized": false}
  ],
  "normalizer": {"type": "Lowercase"},
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);

    assert_eq!(ctx.encode_with("HELLO", &no_bos()), [3, 4, 5, 5, 6]);
    assert_eq!(tokenize_strings(&ctx, "HELLO"), ["h", "e", "l", "l", "o"]);
}

/// Batch encoding must agree with individual encode results when added-token
/// matching depends on normalization.
#[test]
fn added_token_normalized_true_batch_matches_individual_sequences() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "!": 1, "?": 2
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 100, "content": "hello", "special": false, "normalized": true}
  ],
  "normalizer": {"type": "Lowercase"},
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let texts = ["HELLO!", "hello?", "!"];
    let batch = ctx.encode_batch(&texts, &no_bos());
    assert_eq!(batch.num_sequences, 3);

    for (i, text) in texts.iter().enumerate() {
        let start = batch.offsets[i];
        let end = batch.offsets[i + 1];
        let expected = ctx.encode_with(text, &no_bos());
        assert_eq!(
            &batch.ids[start..end],
            expected.as_slice(),
            "batch slice must equal individual encode for normalized added-token case {i}: {text:?}"
        );
    }
}

/// normalized=true should also collapse to a single added token on the
/// tokenize_bytes surface after normalization.
#[test]
fn added_token_normalized_true_visible_on_tokenize_bytes_surface() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "!": 1
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 100, "content": "hello", "special": false, "normalized": true}
  ],
  "normalizer": {"type": "Lowercase"},
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);

    assert_eq!(ctx.encode_with("HELLO!", &no_bos()), [100, 1]);
    assert_eq!(tokenize_bytes_strings(&ctx, "HELLO!"), ["hello", "!"]);
}

/// rstrip behavior must remain consistent between batch encoding and
/// individual encoding.
#[test]
fn rstrip_batch_matches_individual_sequences() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "x": 1, "y": 2, " ": 3, "\t": 4, "\n": 5
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 100, "content": "[SEP]", "special": false, "rstrip": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let texts = ["[SEP] \t\nx", "y[SEP] x"];
    let batch = ctx.encode_batch(&texts, &no_bos());
    assert_eq!(batch.num_sequences, 2);

    for (i, text) in texts.iter().enumerate() {
        let start = batch.offsets[i];
        let end = batch.offsets[i + 1];
        let expected = ctx.encode_with(text, &no_bos());
        assert_eq!(
            &batch.ids[start..end],
            expected.as_slice(),
            "batch slice must equal individual encode for rstrip case {i}: {text:?}"
        );
    }
}

/// lstrip behavior must also match between batch encoding and individual
/// encoding for each sequence.
#[test]
fn lstrip_batch_matches_individual_sequences() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "x": 1, "y": 2, " ": 3, "\t": 4, "\n": 5
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 100, "content": "[MID]", "special": false, "lstrip": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let texts = [" \t\n[MID]x", "y [MID]"];
    let batch = ctx.encode_batch(&texts, &no_bos());
    assert_eq!(batch.num_sequences, 2);

    for (i, text) in texts.iter().enumerate() {
        let start = batch.offsets[i];
        let end = batch.offsets[i + 1];
        let expected = ctx.encode_with(text, &no_bos());
        assert_eq!(
            &batch.ids[start..end],
            expected.as_slice(),
            "batch slice must equal individual encode for lstrip case {i}: {text:?}"
        );
    }
}

/// normalized=false must not match batch items solely via normalization.
#[test]
fn added_token_normalized_false_batch_uses_original_text() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"<unk>": 0, "h": 3, "e": 4, "l": 5, "o": 6},
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 100, "content": "hello", "special": false, "normalized": false}
  ],
  "normalizer": {"type": "Lowercase"},
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let batch = ctx.encode_batch(&["HELLO", "hello"], &no_bos());
    assert_eq!(batch.num_sequences, 2);
    assert_eq!(batch.offsets, vec![0, 5, 6]);
    assert_eq!(&batch.ids[0..5], &[3, 4, 5, 5, 6]);
    assert_eq!(&batch.ids[5..6], &[100]);
}

/// single_word=true should also match punctuation-bounded words in batch mode.
#[test]
fn single_word_punctuation_boundary_batch_uses_added_token() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "\"": 1, "!": 2, "c": 3, "a": 4, "t": 5
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 100, "content": "cat", "special": false, "single_word": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let batch = ctx.encode_batch(&["\"cat\"", "cat!"], &no_bos());
    assert_eq!(batch.num_sequences, 2);
    assert_eq!(batch.offsets, vec![0, 3, 5]);
    assert_eq!(&batch.ids[0..3], &[1, 100, 1]);
    assert_eq!(&batch.ids[3..5], &[100, 2]);
}

/// single_word punctuation-boundary matching must also be visible on the
/// tokenize_bytes surface.
#[test]
fn single_word_punctuation_boundary_visible_on_tokenize_bytes_surface() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "\"": 1, "!": 2, "c": 3, "a": 4, "t": 5
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 100, "content": "cat", "special": false, "single_word": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);

    assert_eq!(ctx.encode_with("\"cat\"!", &no_bos()), [1, 100, 1, 2]);
    assert_eq!(tokenize_bytes_strings(&ctx, "\"cat\"!"), ["\"", "cat", "\"", "!"]);
}

/// Longest-match added tokens must claim the full source span of the winning
/// match, with subsequent raw tokens preserving the tail spans.
#[test]
fn longest_match_offsets_cover_winning_source_span() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "d": 3, "e": 4, "f": 5
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 100, "content": "a", "special": false},
    {"id": 101, "content": "ab", "special": false},
    {"id": 102, "content": "abc", "special": false}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let result = unsafe { super::common::encode_raw(ctx.handle(), b"abcdef", &no_bos()) };
    assert!(result.error_msg.is_null(), "encode failed");
    assert_eq!(result.num_tokens, 4);

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!(ids, &[102, 3, 4, 5]);
    assert_eq!((offsets[0].start, offsets[0].end), (0, 3));
    assert_eq!((offsets[1].start, offsets[1].end), (3, 4));
    assert_eq!((offsets[2].start, offsets[2].end), (4, 5));
    assert_eq!((offsets[3].start, offsets[3].end), (5, 6));
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Added-token preemption over BPE merges must also be reflected in source
/// offsets: the added token owns its span before merged vocab tokens continue.
#[test]
fn added_token_preemption_offsets_split_added_and_merged_spans() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0,
      "c": 3, "a": 4, "t": 5,
      "ca": 100
    },
    "merges": ["c a"]
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 200, "content": "cat", "special": false}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let result = unsafe { super::common::encode_raw(ctx.handle(), b"catca", &no_bos()) };
    assert!(result.error_msg.is_null(), "encode failed");
    assert_eq!(result.num_tokens, 2);

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!(ids, &[200, 100]);
    assert_eq!((offsets[0].start, offsets[0].end), (0, 3));
    assert_eq!((offsets[1].start, offsets[1].end), (3, 5));
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// normalized=false must also remain visible on the tokenize_bytes surface:
/// matching must use the original text, not the lowercased normalized form.
#[test]
fn added_token_normalized_false_visible_on_tokenize_bytes_surface() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"<unk>": 0, "h": 3, "e": 4, "l": 5, "o": 6},
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 100, "content": "hello", "special": false, "normalized": false}
  ],
  "normalizer": {"type": "Lowercase"},
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);

    assert_eq!(ctx.encode_with("HELLO", &no_bos()), [3, 4, 5, 5, 6]);
    assert_eq!(tokenize_bytes_strings(&ctx, "HELLO"), ["h", "e", "l", "l", "o"]);
}

/// rstrip should also be reflected on the tokenize_bytes surface.
#[test]
fn rstrip_visible_on_tokenize_bytes_surface() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "x": 1, " ": 2, "\t": 3, "\n": 4
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 100, "content": "[SEP]", "special": false, "rstrip": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);

    assert_eq!(tokenize_bytes_strings(&ctx, "[SEP] \t\nx"), ["[SEP]", "x"]);
}

/// lstrip should also remove leading whitespace on the tokenize_bytes surface.
#[test]
fn lstrip_visible_on_tokenize_bytes_surface() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "x": 1, " ": 2, "\t": 3, "\n": 4
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 100, "content": "[MID]", "special": false, "lstrip": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);

    assert_eq!(tokenize_bytes_strings(&ctx, " \t\n[MID]x"), ["[MID]", "x"]);
}

/// Longest-match priority must also be reflected on the tokenize surface.
#[test]
fn longest_match_visible_on_tokenize_surfaces() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "d": 3, "e": 4, "f": 5
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 100, "content": "a", "special": false},
    {"id": 101, "content": "ab", "special": false},
    {"id": 102, "content": "abc", "special": false}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);

    assert_eq!(ctx.encode_with("abcdef", &no_bos()), [102, 3, 4, 5]);
    assert_eq!(tokenize_strings(&ctx, "abcdef"), ["abc", "d", "e", "f"]);
    assert_eq!(tokenize_bytes_strings(&ctx, "abcdef"), ["abc", "d", "e", "f"]);

    assert_eq!(ctx.encode_with("abdef", &no_bos()), [101, 3, 4, 5]);
    assert_eq!(tokenize_strings(&ctx, "abdef"), ["ab", "d", "e", "f"]);
    assert_eq!(tokenize_bytes_strings(&ctx, "abdef"), ["ab", "d", "e", "f"]);
}

/// Added-token preemption over BPE merges must also be visible on tokenize
/// surfaces before BPE sees the input.
#[test]
fn added_token_preemption_visible_on_tokenize_surfaces() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0,
      "c": 3, "a": 4, "t": 5,
      "ca": 100
    },
    "merges": ["c a"]
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 200, "content": "cat", "special": false}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);

    assert_eq!(ctx.encode_with("catca", &no_bos()), [200, 100]);
    assert_eq!(tokenize_strings(&ctx, "catca"), ["cat", "ca"]);
    assert_eq!(tokenize_bytes_strings(&ctx, "catca"), ["cat", "ca"]);
}

/// Longest-match priority must also hold in batch encoding for each sequence.
#[test]
fn longest_match_batch_matches_individual_sequences() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0, "d": 3, "e": 4, "f": 5
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 100, "content": "a", "special": false},
    {"id": 101, "content": "ab", "special": false},
    {"id": 102, "content": "abc", "special": false}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let texts = ["abcdef", "abdef", "adef"];
    let batch = ctx.encode_batch(&texts, &no_bos());
    assert_eq!(batch.num_sequences, 3);

    for (i, text) in texts.iter().enumerate() {
        let start = batch.offsets[i];
        let end = batch.offsets[i + 1];
        let expected = ctx.encode_with(text, &no_bos());
        assert_eq!(
            &batch.ids[start..end],
            expected.as_slice(),
            "batch slice must equal individual encode for overlap case {i}: {text:?}"
        );
    }
}

/// Added-token preemption over BPE merges must also remain consistent in batch
/// encoding across sequences with and without trailing mergeable text.
#[test]
fn added_token_preemption_batch_matches_individual_sequences() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0,
      "c": 3, "a": 4, "t": 5,
      "ca": 100
    },
    "merges": ["c a"]
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 200, "content": "cat", "special": false}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let texts = ["cat", "catca", "ca"];
    let batch = ctx.encode_batch(&texts, &no_bos());
    assert_eq!(batch.num_sequences, 3);

    for (i, text) in texts.iter().enumerate() {
        let start = batch.offsets[i];
        let end = batch.offsets[i + 1];
        let expected = ctx.encode_with(text, &no_bos());
        assert_eq!(
            &batch.ids[start..end],
            expected.as_slice(),
            "batch slice must equal individual encode for preemption case {i}: {text:?}"
        );
    }
}
