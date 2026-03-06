//! GPT-2 pretokenizer (ByteLevel with use_regex) tests.
//!
//! Tests the ByteLevel pretokenizer with `use_regex=true`, which applies the
//! GPT-2 regex pattern to split text before byte-level encoding:
//! - Contractions: 's, 't, 're, 've, 'm, 'll, 'd
//! - Letter groups, digit groups, punctuation groups
//! - Space-prefixed word grouping
//!
//! The critical behavior: BPE merges cannot cross regex split boundaries.

use crate::capi::tokenizer::common::{encode_raw, TokenizerTestContext};

fn no_bos() -> talu_sys::EncodeOptions {
    talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    }
}

fn tokenize_strings(ctx: &TokenizerTestContext, text: &str) -> Vec<String> {
    let result = unsafe {
        talu_sys::talu_tokenizer_tokenize(ctx.handle(), text.as_bytes().as_ptr(), text.len())
    };
    assert!(result.error_msg.is_null(), "tokenize failed");

    let ptrs =
        unsafe { std::slice::from_raw_parts(result.tokens as *const *const i8, result.num_tokens) };
    let tokens: Vec<String> = (0..result.num_tokens)
        .map(|i| {
            unsafe { std::ffi::CStr::from_ptr(ptrs[i]) }
                .to_string_lossy()
                .into_owned()
        })
        .collect();

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

#[test]
fn use_regex_true_contraction_boundary_consistent_across_tokenize_surfaces() {
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

    assert_eq!(tokenize_strings(&ctx, "don't"), ["d", "o", "n", "'", "t"]);
    assert_eq!(tokenize_bytes_strings(&ctx, "don't"), ["d", "o", "n", "'", "t"]);

    let batch = ctx.encode_batch(&["don't", "don't"], &no_bos());
    assert_eq!(batch.num_sequences, 2);
    assert_eq!(batch.offsets, vec![0, 5, 10]);
    assert_eq!(&batch.ids[0..5], &[4, 5, 6, 7, 8]);
    assert_eq!(&batch.ids[5..10], &[4, 5, 6, 7, 8]);
}

#[test]
fn use_regex_false_contraction_merge_consistent_across_offsets_and_tokenize_surfaces() {
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

    assert_eq!(tokenize_strings(&ctx, "don't"), ["d", "o", "n'", "t"]);
    assert_eq!(tokenize_bytes_strings(&ctx, "don't"), ["d", "o", "n'", "t"]);

    let result = unsafe { encode_raw(ctx.handle(), b"don't", &no_bos()) };
    assert!(result.error_msg.is_null(), "encode failed");
    assert_eq!(result.num_tokens, 4);

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    assert_eq!(ids, &[4, 5, 9, 8]);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 1));
    assert_eq!((offsets[1].start, offsets[1].end), (1, 2));
    assert_eq!(
        (offsets[2].start, offsets[2].end),
        (2, 4),
        "merged n' token must span both source bytes when use_regex=false"
    );
    assert_eq!((offsets[3].start, offsets[3].end), (4, 5));

    unsafe { talu_sys::talu_encode_result_free(result) };
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

#[test]
fn use_regex_digit_letter_boundary_is_consistent_for_batch_and_offsets() {
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

    assert_eq!(tokenize_strings(&ctx, "abc123"), ["a", "b", "c1", "2", "3"]);
    assert_eq!(tokenize_bytes_strings(&ctx, "abc123"), ["a", "b", "c1", "2", "3"]);

    let result = unsafe { encode_raw(ctx.handle(), b"abc123", &no_bos()) };
    assert!(result.error_msg.is_null(), "encode failed");
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 1));
    assert_eq!((offsets[1].start, offsets[1].end), (1, 2));
    assert_eq!(
        (offsets[2].start, offsets[2].end),
        (2, 4),
        "merged c1 token must span the letter-digit boundary when regex splitting is disabled"
    );
    assert_eq!((offsets[3].start, offsets[3].end), (4, 5));
    assert_eq!((offsets[4].start, offsets[4].end), (5, 6));
    unsafe { talu_sys::talu_encode_result_free(result) };

    let batch = ctx.encode_batch(&["abc123", "abc123"], &no_bos());
    assert_eq!(batch.num_sequences, 2);
    assert_eq!(batch.offsets, vec![0, 5, 10]);
    assert_eq!(&batch.ids[0..5], &[4, 5, 10, 8, 9]);
    assert_eq!(&batch.ids[5..10], &[4, 5, 10, 8, 9]);
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
    assert_eq!(
        tokens,
        vec![4, 5, 6, 6, 7, 11, 8, 7, 9, 6, 10],
        "use_regex=true must keep o and Ġ separated across the regex boundary"
    );
}

/// With use_regex=false, cross-space merge o+Ġ should be allowed.
#[test]
fn use_regex_false_allows_cross_space_merge() {
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
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode_with("hello world", &no_bos());
    assert_eq!(
        tokens,
        vec![4, 5, 6, 6, 12, 8, 7, 9, 6, 10],
        "use_regex=false should allow o+Ġ merge within one chunk"
    );
}

#[test]
fn use_regex_true_space_boundary_consistent_across_tokenize_bytes_and_decode() {
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
    let ids = ctx.encode_with("hello world", &no_bos());
    assert_eq!(ids, vec![4, 5, 6, 6, 7, 11, 8, 7, 9, 6, 10]);
    assert_eq!(
        tokenize_strings(&ctx, "hello world"),
        ["h", "e", "l", "l", "o", "\u{0120}", "w", "o", "r", "l", "d"]
    );
    assert_eq!(
        tokenize_bytes_strings(&ctx, "hello world"),
        ["h", "e", "l", "l", "o", "\u{0120}", "w", "o", "r", "l", "d"]
    );
    assert_eq!(ctx.decode(&ids), "hello world");
}

#[test]
fn use_regex_false_space_merge_consistent_across_tokenize_bytes_and_offsets() {
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
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let ids = ctx.encode_with("hello world", &no_bos());
    assert_eq!(ids, vec![4, 5, 6, 6, 12, 8, 7, 9, 6, 10]);
    assert_eq!(
        tokenize_strings(&ctx, "hello world"),
        ["h", "e", "l", "l", "o\u{0120}", "w", "o", "r", "l", "d"]
    );
    assert_eq!(
        tokenize_bytes_strings(&ctx, "hello world"),
        ["h", "e", "l", "l", "o\u{0120}", "w", "o", "r", "l", "d"]
    );

    let result = unsafe { encode_raw(ctx.handle(), b"hello world", &no_bos()) };
    assert!(result.error_msg.is_null(), "encode failed");
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!(
        (offsets[4].start, offsets[4].end),
        (4, 6),
        "merged o+space token must cover both source bytes when regex splitting is disabled"
    );
    unsafe { talu_sys::talu_encode_result_free(result) };
}

// ===========================================================================
// Punctuation grouping and letter/punctuation boundaries
// ===========================================================================

/// GPT-2 regex keeps adjacent punctuation in one chunk, so merges within
/// punctuation should still fire with use_regex=true.
#[test]
fn use_regex_allows_merge_within_punctuation_chunk() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "!": 4, "?": 5, "!?": 6
    },
    "merges": ["! ?"]
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
    assert_eq!(ctx.encode_with("!?", &no_bos()), vec![6]);
    assert_eq!(tokenize_strings(&ctx, "!?"), ["!?"]);
    assert_eq!(tokenize_bytes_strings(&ctx, "!?"), ["!?"]);
}

/// GPT-2 regex splits letters and punctuation into separate chunks, so a merge
/// across that boundary must not fire with use_regex=true.
#[test]
fn use_regex_prevents_cross_letter_punctuation_merge() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "h": 4, "i": 5, "!": 6, "i!": 7
    },
    "merges": ["i !"]
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
    assert_eq!(ctx.encode_with("hi!", &no_bos()), vec![4, 5, 6]);
    assert_eq!(tokenize_strings(&ctx, "hi!"), ["h", "i", "!"]);
}

/// With regex disabled, the entire string is one chunk, so a merge across the
/// letter/punctuation boundary must fire.
#[test]
fn use_regex_false_allows_cross_letter_punctuation_merge() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "h": 4, "i": 5, "!": 6, "i!": 7
    },
    "merges": ["i !"]
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
    assert_eq!(ctx.encode_with("hi!", &no_bos()), vec![4, 7]);
    assert_eq!(tokenize_strings(&ctx, "hi!"), ["h", "i!"]);

    let result = unsafe { encode_raw(ctx.handle(), b"hi!", &no_bos()) };
    assert!(result.error_msg.is_null(), "encode failed");
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 1));
    assert_eq!(
        (offsets[1].start, offsets[1].end),
        (1, 3),
        "merged i! token must span the source punctuation boundary when regex splitting is disabled"
    );
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Letter groups remain a single GPT-2 regex chunk, so merges inside the
/// letter run must still fire when use_regex=true.
#[test]
fn use_regex_allows_merge_within_letter_chunk() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "a": 4, "b": 5, "c": 6, "ab": 7
    },
    "merges": ["a b"]
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
    assert_eq!(ctx.encode_with("abc", &no_bos()), vec![7, 6]);
    assert_eq!(tokenize_strings(&ctx, "abc"), ["ab", "c"]);

    let result = unsafe { encode_raw(ctx.handle(), b"abc", &no_bos()) };
    assert!(result.error_msg.is_null(), "encode failed");
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 2));
    assert_eq!((offsets[1].start, offsets[1].end), (2, 3));
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Digit groups remain a single GPT-2 regex chunk, so merges inside the
/// digit run must still fire when use_regex=true.
#[test]
fn use_regex_allows_merge_within_digit_chunk() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "1": 4, "2": 5, "3": 6, "12": 7
    },
    "merges": ["1 2"]
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
    assert_eq!(ctx.encode_with("123", &no_bos()), vec![7, 6]);
    assert_eq!(tokenize_strings(&ctx, "123"), ["12", "3"]);

    let batch = ctx.encode_batch(&["123", "123"], &no_bos());
    assert_eq!(batch.num_sequences, 2);
    assert_eq!(batch.offsets, vec![0, 2, 4]);
    assert_eq!(&batch.ids[0..2], &[7, 6]);
    assert_eq!(&batch.ids[2..4], &[7, 6]);
}

/// A space-prefixed word is one GPT-2 regex chunk, so merges between Ġ and
/// the following letter must still fire when use_regex=true.
#[test]
fn use_regex_allows_merge_within_space_prefixed_word_chunk() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "\u0120": 4, "h": 5, "i": 6, "\u0120h": 7
    },
    "merges": ["\u0120 h"]
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
    assert_eq!(ctx.encode_with(" hi", &no_bos()), vec![7, 6]);
    assert_eq!(tokenize_strings(&ctx, " hi"), ["\u{0120}h", "i"]);
    assert_eq!(tokenize_bytes_strings(&ctx, " hi"), ["\u{0120}h", "i"]);

    let result = unsafe { encode_raw(ctx.handle(), b" hi", &no_bos()) };
    assert!(result.error_msg.is_null(), "encode failed");
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!(
        (offsets[0].start, offsets[0].end),
        (0, 2),
        "merged Ġh token must cover both the source space and following letter"
    );
    assert_eq!((offsets[1].start, offsets[1].end), (2, 3));
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// With regex enabled, mixed class input must allow merges inside each class
/// chunk while still blocking earlier-ranked cross-class merges.
#[test]
fn use_regex_true_mixed_classes_merge_only_within_chunks() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "a": 4, "b": 5, "1": 6, "2": 7, "!": 8, "?": 9,
      "ab": 10, "12": 11, "!?": 12,
      "b1": 13, "2!": 14
    },
    "merges": ["b 1", "2 !", "a b", "1 2", "! ?"]
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

    assert_eq!(ctx.encode_with("ab12!?", &no_bos()), vec![10, 11, 12]);
    assert_eq!(tokenize_strings(&ctx, "ab12!?"), ["ab", "12", "!?"]);
    assert_eq!(tokenize_bytes_strings(&ctx, "ab12!?"), ["ab", "12", "!?"]);

    let result = unsafe { encode_raw(ctx.handle(), b"ab12!?", &no_bos()) };
    assert!(result.error_msg.is_null(), "encode failed");
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 2));
    assert_eq!((offsets[1].start, offsets[1].end), (2, 4));
    assert_eq!((offsets[2].start, offsets[2].end), (4, 6));
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// With regex disabled, the same mixed-class string becomes one chunk and
/// earlier-ranked cross-class merges must be allowed to fire.
#[test]
fn use_regex_false_mixed_classes_allow_cross_chunk_merges() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "a": 4, "b": 5, "1": 6, "2": 7, "!": 8, "?": 9,
      "ab": 10, "12": 11, "!?": 12,
      "b1": 13, "2!": 14
    },
    "merges": ["b 1", "2 !", "a b", "1 2", "! ?"]
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

    assert_eq!(ctx.encode_with("ab12!?", &no_bos()), vec![4, 13, 14, 9]);
    assert_eq!(tokenize_strings(&ctx, "ab12!?"), ["a", "b1", "2!", "?"]);
    assert_eq!(tokenize_bytes_strings(&ctx, "ab12!?"), ["a", "b1", "2!", "?"]);

    let result = unsafe { encode_raw(ctx.handle(), b"ab12!?", &no_bos()) };
    assert!(result.error_msg.is_null(), "encode failed");
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 1));
    assert_eq!((offsets[1].start, offsets[1].end), (1, 3));
    assert_eq!((offsets[2].start, offsets[2].end), (3, 5));
    assert_eq!((offsets[3].start, offsets[3].end), (5, 6));
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// With regex enabled, a space-prefixed word and following punctuation are
/// separate chunks: merges may occur inside each chunk, but not across them.
#[test]
fn use_regex_true_space_prefixed_word_then_punctuation_merges_per_chunk() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "\u0120": 4, "h": 5, "i": 6, "!": 7, "?": 8,
      "\u0120h": 9, "i!": 10, "!?": 11
    },
    "merges": ["i !", "\u0120 h", "! ?"]
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

    assert_eq!(ctx.encode_with(" hi!?", &no_bos()), vec![9, 6, 11]);
    assert_eq!(tokenize_strings(&ctx, " hi!?"), ["\u{0120}h", "i", "!?"]);
    assert_eq!(tokenize_bytes_strings(&ctx, " hi!?"), ["\u{0120}h", "i", "!?"]);

    let result = unsafe { encode_raw(ctx.handle(), b" hi!?", &no_bos()) };
    assert!(result.error_msg.is_null(), "encode failed");
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 2));
    assert_eq!((offsets[1].start, offsets[1].end), (2, 3));
    assert_eq!((offsets[2].start, offsets[2].end), (3, 5));
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// With regex disabled, the same input is one chunk, so the cross-boundary
/// i+! merge must be allowed to fire before the punctuation-pair merge.
#[test]
fn use_regex_false_space_prefixed_word_then_punctuation_allows_cross_boundary_merge() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "\u0120": 4, "h": 5, "i": 6, "!": 7, "?": 8,
      "\u0120h": 9, "i!": 10, "!?": 11
    },
    "merges": ["i !", "\u0120 h", "! ?"]
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

    assert_eq!(ctx.encode_with(" hi!?", &no_bos()), vec![9, 10, 8]);
    assert_eq!(tokenize_strings(&ctx, " hi!?"), ["\u{0120}h", "i!", "?"]);
    assert_eq!(tokenize_bytes_strings(&ctx, " hi!?"), ["\u{0120}h", "i!", "?"]);

    let result = unsafe { encode_raw(ctx.handle(), b" hi!?", &no_bos()) };
    assert!(result.error_msg.is_null(), "encode failed");
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 2));
    assert_eq!((offsets[1].start, offsets[1].end), (2, 4));
    assert_eq!((offsets[2].start, offsets[2].end), (4, 5));
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Batch encoding with regex enabled must preserve the per-sequence merged
/// chunk structure for mixed-class inputs.
#[test]
fn use_regex_true_mixed_batch_offsets_match_individual_chunk_merges() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "a": 4, "b": 5, "1": 6, "2": 7, "!": 8, "?": 9,
      "ab": 10, "12": 11, "!?": 12,
      "\u0120": 13, "h": 14, "i": 15, "\u0120h": 16
    },
    "merges": ["a b", "1 2", "! ?", "\u0120 h"]
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

    let batch = ctx.encode_batch(&["ab12!?", " hi!?"], &no_bos());
    assert_eq!(batch.num_sequences, 2);
    assert_eq!(batch.offsets, vec![0, 3, 6]);
    assert_eq!(&batch.ids[0..3], &[10, 11, 12]);
    assert_eq!(&batch.ids[3..6], &[16, 15, 12]);
}

/// With regex disabled, the same mixed batch must allow cross-class merges in
/// each sequence rather than preserving regex chunk boundaries.
#[test]
fn use_regex_false_mixed_batch_allows_cross_class_merges_per_sequence() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "a": 4, "b": 5, "1": 6, "2": 7, "!": 8, "?": 9,
      "ab": 10, "12": 11, "!?": 12, "b1": 13, "2!": 14,
      "\u0120": 15, "h": 16, "i": 17, "\u0120h": 18, "i!": 19
    },
    "merges": ["b 1", "2 !", "i !", "a b", "1 2", "! ?", "\u0120 h"]
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

    let batch = ctx.encode_batch(&["ab12!?", " hi!?"], &no_bos());
    assert_eq!(batch.num_sequences, 2);
    assert_eq!(batch.offsets, vec![0, 4, 7]);
    assert_eq!(&batch.ids[0..4], &[4, 13, 14, 9]);
    assert_eq!(&batch.ids[4..7], &[18, 19, 9]);
}

/// With regex enabled, a digit chunk and following space-prefixed word chunk
/// must stay separate, so a 2+Ġ merge cannot fire across that boundary.
#[test]
fn use_regex_prevents_cross_digit_space_merge() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "1": 4, "2": 5, "\u0120": 6, "h": 7, "i": 8,
      "2\u0120": 9, "\u0120h": 10
    },
    "merges": ["2 \u0120", "\u0120 h"]
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

    assert_eq!(ctx.encode_with("12 hi", &no_bos()), vec![4, 5, 10, 8]);
    assert_eq!(tokenize_strings(&ctx, "12 hi"), ["1", "2", "\u{0120}h", "i"]);
    assert_eq!(tokenize_bytes_strings(&ctx, "12 hi"), ["1", "2", "\u{0120}h", "i"]);
}

/// With regex disabled, the entire string is one chunk, so the 2+Ġ merge must
/// be allowed to fire across the digit/space boundary.
#[test]
fn use_regex_false_allows_cross_digit_space_merge() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "1": 4, "2": 5, "\u0120": 6, "h": 7, "i": 8,
      "2\u0120": 9, "\u0120h": 10
    },
    "merges": ["2 \u0120", "\u0120 h"]
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

    assert_eq!(ctx.encode_with("12 hi", &no_bos()), vec![4, 9, 7, 8]);
    assert_eq!(tokenize_strings(&ctx, "12 hi"), ["1", "2\u{0120}", "h", "i"]);
    assert_eq!(tokenize_bytes_strings(&ctx, "12 hi"), ["1", "2\u{0120}", "h", "i"]);

    let result = unsafe { encode_raw(ctx.handle(), b"12 hi", &no_bos()) };
    assert!(result.error_msg.is_null(), "encode failed");
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 1));
    assert_eq!((offsets[1].start, offsets[1].end), (1, 3));
    assert_eq!((offsets[2].start, offsets[2].end), (3, 4));
    assert_eq!((offsets[3].start, offsets[3].end), (4, 5));
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Batch encoding with regex enabled must preserve the separated digit and
/// space-prefixed-word chunks for each sequence.
#[test]
fn use_regex_true_digit_space_batch_matches_expected_chunks() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "1": 4, "2": 5, "\u0120": 6, "h": 7, "i": 8,
      "2\u0120": 9, "\u0120h": 10
    },
    "merges": ["2 \u0120", "\u0120 h"]
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
    let batch = ctx.encode_batch(&["12 hi", "12 hi"], &no_bos());
    assert_eq!(batch.num_sequences, 2);
    assert_eq!(batch.offsets, vec![0, 4, 8]);
    assert_eq!(&batch.ids[0..4], &[4, 5, 10, 8]);
    assert_eq!(&batch.ids[4..8], &[4, 5, 10, 8]);
}

/// With regex disabled, batch mode must allow the digit/space cross-boundary
/// merge in every sequence rather than preserving regex chunking.
#[test]
fn use_regex_false_digit_space_batch_allows_cross_boundary_merge() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "1": 4, "2": 5, "\u0120": 6, "h": 7, "i": 8,
      "2\u0120": 9, "\u0120h": 10
    },
    "merges": ["2 \u0120", "\u0120 h"]
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
    let batch = ctx.encode_batch(&["12 hi", "12 hi"], &no_bos());
    assert_eq!(batch.num_sequences, 2);
    assert_eq!(batch.offsets, vec![0, 4, 8]);
    assert_eq!(&batch.ids[0..4], &[4, 9, 7, 8]);
    assert_eq!(&batch.ids[4..8], &[4, 9, 7, 8]);
}

/// With regex enabled, punctuation and the following space-prefixed word are
/// separate chunks, so a ?+Ġ merge must not fire across that boundary.
#[test]
fn use_regex_prevents_cross_punctuation_space_merge() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "!": 4, "?": 5, "\u0120": 6, "h": 7, "i": 8,
      "?\u0120": 9, "\u0120h": 10
    },
    "merges": ["? \u0120", "\u0120 h"]
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

    assert_eq!(ctx.encode_with("!? hi", &no_bos()), vec![4, 5, 10, 8]);
    assert_eq!(tokenize_strings(&ctx, "!? hi"), ["!", "?", "\u{0120}h", "i"]);
    assert_eq!(tokenize_bytes_strings(&ctx, "!? hi"), ["!", "?", "\u{0120}h", "i"]);
}

/// With regex disabled, the same input is one chunk, so the ?+Ġ merge must
/// be allowed to fire across the punctuation/space boundary.
#[test]
fn use_regex_false_allows_cross_punctuation_space_merge() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "!": 4, "?": 5, "\u0120": 6, "h": 7, "i": 8,
      "?\u0120": 9, "\u0120h": 10
    },
    "merges": ["? \u0120", "\u0120 h"]
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

    assert_eq!(ctx.encode_with("!? hi", &no_bos()), vec![4, 9, 7, 8]);
    assert_eq!(tokenize_strings(&ctx, "!? hi"), ["!", "?\u{0120}", "h", "i"]);
    assert_eq!(tokenize_bytes_strings(&ctx, "!? hi"), ["!", "?\u{0120}", "h", "i"]);

    let result = unsafe { encode_raw(ctx.handle(), b"!? hi", &no_bos()) };
    assert!(result.error_msg.is_null(), "encode failed");
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 1));
    assert_eq!((offsets[1].start, offsets[1].end), (1, 3));
    assert_eq!((offsets[2].start, offsets[2].end), (3, 4));
    assert_eq!((offsets[3].start, offsets[3].end), (4, 5));
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// With regex enabled, digits and punctuation are separate chunks, so a 2+!
/// merge must not fire across that boundary.
#[test]
fn use_regex_prevents_cross_digit_punctuation_merge() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "1": 4, "2": 5, "!": 6, "?": 7,
      "2!": 8, "!?": 9
    },
    "merges": ["2 !", "! ?"]
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

    assert_eq!(ctx.encode_with("12!?", &no_bos()), vec![4, 5, 9]);
    assert_eq!(tokenize_strings(&ctx, "12!?"), ["1", "2", "!?"]);
    assert_eq!(tokenize_bytes_strings(&ctx, "12!?"), ["1", "2", "!?"]);
}

/// With regex disabled, the same input is one chunk, so the 2+! merge must
/// be allowed to fire across the digit/punctuation boundary.
#[test]
fn use_regex_false_allows_cross_digit_punctuation_merge() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "1": 4, "2": 5, "!": 6, "?": 7,
      "2!": 8, "!?": 9
    },
    "merges": ["2 !", "! ?"]
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

    assert_eq!(ctx.encode_with("12!?", &no_bos()), vec![4, 8, 7]);
    assert_eq!(tokenize_strings(&ctx, "12!?"), ["1", "2!", "?"]);
    assert_eq!(tokenize_bytes_strings(&ctx, "12!?"), ["1", "2!", "?"]);

    let result = unsafe { encode_raw(ctx.handle(), b"12!?", &no_bos()) };
    assert!(result.error_msg.is_null(), "encode failed");
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 1));
    assert_eq!((offsets[1].start, offsets[1].end), (1, 3));
    assert_eq!((offsets[2].start, offsets[2].end), (3, 4));
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Batch encoding with regex enabled must preserve separated digit and
/// punctuation chunks for each sequence.
#[test]
fn use_regex_true_digit_punctuation_batch_matches_expected_chunks() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "1": 4, "2": 5, "!": 6, "?": 7,
      "2!": 8, "!?": 9
    },
    "merges": ["2 !", "! ?"]
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
    let batch = ctx.encode_batch(&["12!?", "12!?"], &no_bos());
    assert_eq!(batch.num_sequences, 2);
    assert_eq!(batch.offsets, vec![0, 3, 6]);
    assert_eq!(&batch.ids[0..3], &[4, 5, 9]);
    assert_eq!(&batch.ids[3..6], &[4, 5, 9]);
}

/// With regex enabled, batch encoding must agree exactly with individual
/// encodes for mixed digit/punctuation inputs of different lengths.
#[test]
fn use_regex_true_digit_punctuation_batch_matches_individual_sequences() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "1": 4, "2": 5, "!": 6, "?": 7,
      "2!": 8, "!?": 9
    },
    "merges": ["2 !", "! ?"]
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
    let texts = ["12!?", "1!?", "12?"];
    let batch = ctx.encode_batch(&texts, &no_bos());
    assert_eq!(batch.num_sequences, 3);

    for (i, text) in texts.iter().enumerate() {
        let start = batch.offsets[i];
        let end = batch.offsets[i + 1];
        let expected = ctx.encode_with(text, &no_bos());
        assert_eq!(
            &batch.ids[start..end],
            expected.as_slice(),
            "regex-enabled batch slice must equal individual encode for sequence {i}: {text:?}"
        );
    }
}

/// With regex disabled, batch mode must allow the digit/punctuation
/// cross-boundary merge in every sequence.
#[test]
fn use_regex_false_digit_punctuation_batch_allows_cross_boundary_merge() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "1": 4, "2": 5, "!": 6, "?": 7,
      "2!": 8, "!?": 9
    },
    "merges": ["2 !", "! ?"]
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
    let batch = ctx.encode_batch(&["12!?", "12!?"], &no_bos());
    assert_eq!(batch.num_sequences, 2);
    assert_eq!(batch.offsets, vec![0, 3, 6]);
    assert_eq!(&batch.ids[0..3], &[4, 8, 7]);
    assert_eq!(&batch.ids[3..6], &[4, 8, 7]);
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

/// The merged `'re` contraction token must cover the full source substring.
#[test]
fn contraction_re_offset_covers_apostrophe_and_suffix() {
    let ctx = TokenizerTestContext::from_json(CONTRACTION_JSON);
    let result = unsafe { encode_raw(ctx.handle(), b"they're", &no_bos()) };
    assert!(result.error_msg.is_null(), "encode failed");
    assert_eq!(result.num_tokens, 5);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 1));
    assert_eq!((offsets[1].start, offsets[1].end), (1, 2));
    assert_eq!((offsets[2].start, offsets[2].end), (2, 3));
    assert_eq!((offsets[3].start, offsets[3].end), (3, 4));
    assert_eq!(
        (offsets[4].start, offsets[4].end),
        (4, 7),
        "merged 're token must span apostrophe+r+e source bytes"
    );

    unsafe { talu_sys::talu_encode_result_free(result) };
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

/// Batch encoding preserves contraction merges independently per sequence.
#[test]
fn contraction_batch_sequences_preserve_merged_suffixes() {
    let ctx = TokenizerTestContext::from_json(CONTRACTION_JSON);
    let batch = ctx.encode_batch(&["I'm", "they'll"], &no_bos());

    assert_eq!(batch.num_sequences, 2);
    assert_eq!(batch.offsets, vec![0, 2, 7]);
    assert_eq!(&batch.ids[0..2], &[13, 23]);
    assert_eq!(&batch.ids[2..7], &[6, 17, 8, 19, 29]);
}
