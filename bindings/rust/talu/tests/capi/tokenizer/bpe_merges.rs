//! BPE merge algorithm edge-case tests.
//!
//! Tests merge rank ordering, tie-breaking (leftmost pair wins), cascade
//! merges, repeated characters, small-word vs. large-word merge paths,
//! and single/two-symbol fast paths.

use crate::capi::tokenizer::common::TokenizerTestContext;

fn no_bos() -> talu_sys::EncodeOptions {
    talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    }
}

// ===========================================================================
// Leftmost-wins tie-breaking
// ===========================================================================

/// When the same merge pair appears at multiple positions, the leftmost
/// occurrence is merged first.
///
/// Vocab: a=3, b=4, ab=5; Merge: a+b→ab.
/// Input "aba" has pair (a,b) at position 0 → merge → "ab a" = [5, 3].
/// NOT "a ba" because leftmost wins.
#[test]
fn leftmost_pair_wins() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "a": 4, "b": 5, "ab": 6
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
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    // "aba" → symbols [a, b, a] → merge leftmost (a,b) → [ab, a] = [6, 4]
    let tokens = ctx.encode_with("aba", &no_bos());
    assert_eq!(
        tokens,
        vec![6, 4],
        "leftmost pair must be merged first: 'aba' → [ab=6, a=4], got: {tokens:?}"
    );
}

// ===========================================================================
// Repeated characters with merge
// ===========================================================================

/// "aaa" with merge a+a→aa: leftmost pair merged first → "aa" + "a".
#[test]
fn repeated_chars_leftmost_merge() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "a": 4, "aa": 5
    },
    "merges": ["a a"]
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    // "aaa" → [a, a, a] → merge leftmost (a,a) → [aa, a] = [5, 4]
    let tokens = ctx.encode_with("aaa", &no_bos());
    assert_eq!(
        tokens,
        vec![5, 4],
        "'aaa' with a+a merge: leftmost pair → [aa=5, a=4], got: {tokens:?}"
    );
}

/// "aaaa" with merge a+a→aa: two merges → "aa" + "aa".
#[test]
fn four_repeated_chars_double_merge() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "a": 4, "aa": 5
    },
    "merges": ["a a"]
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    // "aaaa" → [a, a, a, a] → merge left (a,a) → [aa, a, a]
    // → merge left (a,a) → [aa, aa] = [5, 5]
    let tokens = ctx.encode_with("aaaa", &no_bos());
    assert_eq!(tokens, vec![5, 5], "'aaaa' → [aa=5, aa=5], got: {tokens:?}");
}

/// "aaaaa" (5 a's) with merge a+a→aa: → [aa, aa, a].
#[test]
fn five_repeated_chars() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "a": 4, "aa": 5
    },
    "merges": ["a a"]
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    // "aaaaa" → [a,a,a,a,a] → merge leftmost → [aa,a,a,a] → [aa,aa,a] = [5,5,4]
    let tokens = ctx.encode_with("aaaaa", &no_bos());
    assert_eq!(
        tokens,
        vec![5, 5, 4],
        "'aaaaa' → [aa=5, aa=5, a=4], got: {tokens:?}"
    );
}

// ===========================================================================
// Cascade merges (merge creates new mergeable pair)
// ===========================================================================

/// Merging "a"+"b"→"ab" then "ab"+"c"→"abc" in a single word.
#[test]
fn cascade_merge_two_steps() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "a": 4, "b": 5, "c": 6, "ab": 7, "abc": 8
    },
    "merges": ["a b", "ab c"]
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    // "abc" → [a, b, c] → merge rank 0: a+b→ab → [ab, c]
    // → merge rank 1: ab+c→abc → [abc] = [8]
    let tokens = ctx.encode_with("abc", &no_bos());
    assert_eq!(
        tokens,
        vec![8],
        "cascade merge: 'abc' → [abc=8], got: {tokens:?}"
    );
}

/// Three-level cascade: a+b→ab, c+d→cd, ab+cd→abcd.
#[test]
fn cascade_merge_three_levels() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "a": 4, "b": 5, "c": 6, "d": 7,
      "ab": 8, "cd": 9, "abcd": 10
    },
    "merges": ["a b", "c d", "ab cd"]
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    // "abcd" → [a,b,c,d] → merge rank 0: a+b→ab → [ab,c,d]
    // → merge rank 1: c+d→cd → [ab,cd]
    // → merge rank 2: ab+cd→abcd → [abcd] = [10]
    let tokens = ctx.encode_with("abcd", &no_bos());
    assert_eq!(
        tokens,
        vec![10],
        "three-level cascade: 'abcd' → [abcd=10], got: {tokens:?}"
    );
}

// ===========================================================================
// Merge rank ordering
// ===========================================================================

/// Lower-ranked merge is applied before higher-ranked even if both are possible.
///
/// Merges: rank 0 = b+c, rank 1 = a+b.
/// "abc" → [a,b,c] → rank 0 first: b+c→bc → [a,bc].
/// NOT a+b first.
#[test]
fn lower_rank_merge_applied_first() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "a": 4, "b": 5, "c": 6, "bc": 7, "ab": 8
    },
    "merges": ["b c", "a b"]
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    // "abc" → [a,b,c] → rank 0 (b+c) wins over rank 1 (a+b)
    // → [a, bc] = [4, 7]
    let tokens = ctx.encode_with("abc", &no_bos());
    assert_eq!(
        tokens,
        vec![4, 7],
        "lower rank b+c should apply first: 'abc' → [a=4, bc=7], got: {tokens:?}"
    );
}

// ===========================================================================
// Single-symbol and two-symbol fast paths
// ===========================================================================

/// Single character: no merge possible, skip merge loop.
#[test]
fn single_char_no_merge() {
    let ctx = TokenizerTestContext::with_merges();
    let tokens = ctx.encode_with("a", &no_bos());
    assert_eq!(tokens, vec![69], "single char should skip merge loop");
}

/// Two characters with no merge rule: each stays separate.
#[test]
fn two_chars_no_merge_rule() {
    let ctx = TokenizerTestContext::with_merges();
    // "ab" — no merge for a+b in the merges fixture
    let tokens = ctx.encode_with("ab", &no_bos());
    assert_eq!(
        tokens,
        vec![69, 70],
        "two chars with no merge rule: 'ab' → [a=69, b=70]"
    );
}

/// Two characters with matching merge rule: single token output.
#[test]
fn two_chars_with_merge() {
    let ctx = TokenizerTestContext::with_merges();
    // "he" → merge h+e → he(99)
    let tokens = ctx.encode_with("he", &no_bos());
    assert_eq!(tokens, vec![99], "'he' should merge to single token");
}

// ===========================================================================
// Words crossing the small/large merge threshold
// ===========================================================================
//
// The BPE implementation uses a simple re-scan for 3-5 symbols and a
// cached-pair approach for 6+ symbols. These tests exercise the boundary.

/// 5-symbol word (small path): "helox" with partial merges.
#[test]
fn five_symbol_word_small_path() {
    let ctx = TokenizerTestContext::with_merges();
    // "helox" → [h,e,l,o,x] → merge h+e→he → [he,l,o,x]
    // → merge he+l→hel → [hel,o,x] — 3 tokens
    let tokens = ctx.encode_with("helox", &no_bos());
    assert_eq!(
        tokens,
        vec![102, 83, 92],
        "5-symbol small path: 'helox' → [hel=102, o=83, x=92], got: {tokens:?}"
    );
}

/// 6-symbol word (large path): "hellox" with partial merges.
#[test]
fn six_symbol_word_large_path() {
    let ctx = TokenizerTestContext::with_merges();
    // "hellox" → [h,e,l,l,o,x] → merge chain for "hello" + x
    // h+e→he, he+l→hel, hel+l→hell, hell+o→hello
    // → [hello=104, x=92]
    let tokens = ctx.encode_with("hellox", &no_bos());
    assert_eq!(
        tokens,
        vec![104, 92],
        "6-symbol large path: 'hellox' → [hello=104, x=92], got: {tokens:?}"
    );
}

/// 7-symbol word: "helloab" — verify large-path correctness.
#[test]
fn seven_symbol_word() {
    let ctx = TokenizerTestContext::with_merges();
    // "helloab" → full merge chain for "hello" + a + b
    // → [hello=104, a=69, b=70]
    let tokens = ctx.encode_with("helloab", &no_bos());
    assert_eq!(
        tokens,
        vec![104, 69, 70],
        "7-symbol word: 'helloab' → [hello=104, a=69, b=70], got: {tokens:?}"
    );
}

// ===========================================================================
// Encode→decode roundtrip with merges
// ===========================================================================

/// Roundtrip through encode→decode preserves text for merged words.
#[test]
fn merge_encode_decode_roundtrip() {
    let ctx = TokenizerTestContext::with_merges();
    let opts = no_bos();

    for text in ["hello", "hell", "he", "lo", "ll"] {
        let tokens = ctx.encode_with(text, &opts);
        let decoded = ctx.decode(&tokens);
        assert_eq!(decoded, text, "roundtrip failed for {text:?}");
    }
}

/// No merges exist for this word — still roundtrips correctly.
#[test]
fn no_merge_roundtrip() {
    let ctx = TokenizerTestContext::with_merges();
    let opts = no_bos();

    for text in ["abc", "xyz", "Hi"] {
        let tokens = ctx.encode_with(text, &opts);
        let decoded = ctx.decode(&tokens);
        assert_eq!(decoded, text, "no-merge roundtrip failed for {text:?}");
    }
}

// ===========================================================================
// Deep cascade merges (5+ levels, exercises cached-pair path)
// ===========================================================================

/// 5-level cascade on 8 initial symbols (cached-pair path, 6+ symbols).
///
/// Each merge produces a token that participates in the NEXT merge.
/// This stresses the cache invalidation/update logic: after each merge,
/// the newly created symbol must form a valid pair with its neighbor
/// and that pair must be found in the cache or re-inserted.
///
/// Merge chain: a+b→ab, ab+c→abc, abc+d→abcd, abcd+e→abcde, abcde+f→abcdef
/// Input "abcdefgh" → [abcdef, g, h]
#[test]
fn deep_cascade_five_levels_cached_path() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "a": 4, "b": 5, "c": 6, "d": 7, "e": 8, "f": 9, "g": 10, "h": 11,
      "ab": 12, "abc": 13, "abcd": 14, "abcde": 15, "abcdef": 16
    },
    "merges": ["a b", "ab c", "abc d", "abcd e", "abcde f"]
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode_with("abcdefgh", &no_bos());
    assert_eq!(
        tokens,
        vec![16, 10, 11],
        "5-level cascade: 'abcdefgh' → [abcdef=16, g=10, h=11], got: {tokens:?}"
    );
}

/// Deep power-of-two cascade at the MAX_WORD_SYMBOLS boundary (512 symbols)
/// must reduce deterministically to one token without cache corruption.
#[test]
fn deep_power_of_two_cascade_512_symbols_reduces_to_one_token() {
    let levels = 9usize; // 2^9 = 512 symbols.
    let mut vocab_entries = vec![
        "\"<pad>\": 0".to_string(),
        "\"<s>\": 1".to_string(),
        "\"</s>\": 2".to_string(),
        "\"<unk>\": 3".to_string(),
        "\"a\": 4".to_string(),
    ];
    let mut merges = Vec::with_capacity(levels);

    let mut prev = "a".to_string();
    for i in 0..levels {
        let merged = format!("{prev}{prev}");
        let id = 5 + (i as u32);
        vocab_entries.push(format!("\"{}\": {}", merged, id));
        merges.push(format!("\"{} {}\"", prev, prev));
        prev = merged;
    }

    let json = format!(
        r####"{{
  "version": "1.0",
  "model": {{
    "type": "BPE",
    "vocab": {{
      {}
    }},
    "merges": [{}]
  }},
  "added_tokens": [
    {{"id": 0, "content": "<pad>", "special": true}},
    {{"id": 1, "content": "<s>", "special": true}},
    {{"id": 2, "content": "</s>", "special": true}},
    {{"id": 3, "content": "<unk>", "special": true}}
  ],
  "normalizer": null,
  "pre_tokenizer": {{"type": "ByteLevel", "add_prefix_space": false}},
  "post_processor": null,
  "decoder": {{"type": "ByteLevel"}}
}}"####,
        vocab_entries.join(",\n      "),
        merges.join(", ")
    );

    let ctx = TokenizerTestContext::from_json(&json);
    let input = "a".repeat(1usize << levels);
    let first = ctx.encode_with(&input, &no_bos());
    let second = ctx.encode_with(&input, &no_bos());
    assert_eq!(first, second, "deep cascade must be deterministic");
    assert_eq!(
        first.len(),
        1,
        "2^9 repeated-symbol cascade should fully reduce to one token"
    );
    assert_eq!(
        first,
        vec![5 + ((levels as u32) - 1)],
        "final reduced token ID must match the deepest configured cascade level"
    );
}

/// An unbalanced deep cascade (left-growing chain) at the 512-symbol boundary
/// must remain deterministic and fully reducible.
///
/// Merge chain:
///   a b -> ab
///   ab b -> abb
///   abb b -> abbb
///   ...
/// Input: "a" followed by 511 "b" symbols (512 total symbols).
#[test]
fn deep_unbalanced_cascade_512_symbols_reduces_to_one_token() {
    let levels = 511usize;
    let mut vocab_entries = vec![
        "\"<pad>\": 0".to_string(),
        "\"<s>\": 1".to_string(),
        "\"</s>\": 2".to_string(),
        "\"<unk>\": 3".to_string(),
        "\"a\": 4".to_string(),
        "\"b\": 5".to_string(),
    ];
    let mut merges = Vec::with_capacity(levels);

    let mut prev = "a".to_string();
    for i in 0..levels {
        let merged = format!("{prev}b");
        let id = 6 + (i as u32);
        vocab_entries.push(format!("\"{}\": {}", merged, id));
        merges.push(format!("\"{} b\"", prev));
        prev = merged;
    }

    let json = format!(
        r####"{{
  "version": "1.0",
  "model": {{
    "type": "BPE",
    "vocab": {{
      {}
    }},
    "merges": [{}]
  }},
  "added_tokens": [
    {{"id": 0, "content": "<pad>", "special": true}},
    {{"id": 1, "content": "<s>", "special": true}},
    {{"id": 2, "content": "</s>", "special": true}},
    {{"id": 3, "content": "<unk>", "special": true}}
  ],
  "normalizer": null,
  "pre_tokenizer": {{"type": "ByteLevel", "add_prefix_space": false}},
  "post_processor": null,
  "decoder": {{"type": "ByteLevel"}}
}}"####,
        vocab_entries.join(",\n      "),
        merges.join(", ")
    );

    let ctx = TokenizerTestContext::from_json(&json);
    let input = format!("a{}", "b".repeat(levels));
    let first = ctx.encode_with(&input, &no_bos());
    let second = ctx.encode_with(&input, &no_bos());
    assert_eq!(first, second, "unbalanced cascade must be deterministic");
    assert_eq!(
        first.len(),
        1,
        "unbalanced 512-symbol cascade should fully reduce to one token"
    );
    assert_eq!(
        first,
        vec![6 + ((levels as u32) - 1)],
        "final token ID must match the deepest unbalanced chain level"
    );
}

// ===========================================================================
// Merge creates pair at distant position (cache re-scan)
// ===========================================================================

/// Merge at position 4-5 creates a new mergeable pair at position 3-4.
///
/// This tests whether the merge algorithm correctly discovers that
/// a newly-created symbol forms a valid pair with its LEFT neighbor,
/// not just the right neighbor. In the cached-pair path, after merging
/// symbols at positions i and i+1, the algorithm must check if
/// (i-1, new_symbol) is a valid merge and add it to the cache.
///
/// Setup (7 initial symbols → cached path):
///   Input: "xyzabcd"
///   Merges rank 0: a+b→ab, rank 1: z+ab→zab
///   After rank 0: [x, y, z, ab, c, d]
///   The pair (z, ab) now exists → rank 1 fires → [x, y, zab, c, d]
#[test]
fn merge_creates_left_neighbor_pair() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "x": 4, "y": 5, "z": 6, "a": 7, "b": 8, "c": 9, "d": 10,
      "ab": 11, "zab": 12
    },
    "merges": ["a b", "z ab"]
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode_with("xyzabcd", &no_bos());
    assert_eq!(
        tokens,
        vec![4, 5, 12, 9, 10],
        "left-neighbor pair: 'xyzabcd' → [x, y, zab=12, c, d], got: {tokens:?}"
    );
}

/// Merge at position 0-1 creates a pair with the RIGHT neighbor at position 2.
///
/// After merging positions 0+1 into a new symbol, the algorithm must
/// check if (new_symbol, symbol_at_2) forms a valid merge pair.
///
/// Input: "abcdefg" (7 symbols → cached path)
/// rank 0: a+b→ab; rank 1: ab+c→abc
/// After rank 0: [ab, c, d, e, f, g] → pair (ab,c) exists → rank 1 fires
#[test]
fn merge_creates_right_neighbor_pair() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "a": 4, "b": 5, "c": 6, "d": 7, "e": 8, "f": 9, "g": 10,
      "ab": 11, "abc": 12
    },
    "merges": ["a b", "ab c"]
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode_with("abcdefg", &no_bos());
    assert_eq!(
        tokens,
        vec![12, 7, 8, 9, 10],
        "right-neighbor pair: 'abcdefg' → [abc=12, d, e, f, g], got: {tokens:?}"
    );
}

// ===========================================================================
// Competing merges: same rank consumed by earlier merge
// ===========================================================================

/// Two instances of the same pair compete. Leftmost wins, consuming the
/// shared symbol and invalidating the rightmost instance.
///
/// Input: "abab" (4 symbols → small path, but let's test with 8)
/// We pad to 8 symbols for cached path: "xxababyy"
/// Merge rank 0: a+b→ab
/// "xxababyy" → [x, x, ab, ab, y, y] — both a+b pairs merge (no conflict).
///
/// Now the tricky case: "xabaxaby" (8 symbols)
/// Merge rank 0: a+b→ab; rank 1: ab+a→aba
/// → [x, ab, a, x, ab, y] → rank 1: leftmost (ab,a) at pos 1-2 fires
/// → [x, aba, x, ab, y]
/// The second ab+a would need the "a" at pos 2, but it was consumed.
#[test]
fn competing_merges_leftmost_consumes_shared_symbol() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "x": 4, "a": 5, "b": 6, "y": 7,
      "ab": 8, "aba": 9
    },
    "merges": ["a b", "ab a"]
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    // "xabaxaby" → [x, a, b, a, x, a, b, y]
    // rank 0 (a+b): leftmost at pos 1-2 → [x, ab, a, x, a, b, y]
    //   then next a+b at pos 4-5 → [x, ab, a, x, ab, y]
    // rank 1 (ab+a): leftmost at pos 1-2 → [x, aba, x, ab, y]
    //   second ab+a at pos 3-4 would need "a" but none follows ab at pos 3
    let tokens = ctx.encode_with("xabaxaby", &no_bos());
    assert_eq!(
        tokens,
        vec![4, 9, 4, 8, 7],
        "competing: 'xabaxaby' → [x, aba=9, x, ab=8, y], got: {tokens:?}"
    );
}

// ===========================================================================
// Merge that straddles the small/cached boundary differently
// ===========================================================================

/// Exactly 6 initial symbols with merges that reduce to 2 symbols.
///
/// Tests the cached-pair path with aggressive merging that quickly
/// shrinks the symbol list. After each merge the cache shrinks and
/// position indices shift (swap-remove in cache).
///
/// Input: "abcdef" (6 symbols → cached path)
/// Merges: a+b→ab, c+d→cd, e+f→ef, ab+cd→abcd, abcd+ef→abcdef
/// → final: [abcdef]
#[test]
fn cached_path_aggressive_reduction() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "a": 4, "b": 5, "c": 6, "d": 7, "e": 8, "f": 9,
      "ab": 10, "cd": 11, "ef": 12, "abcd": 13, "abcdef": 14
    },
    "merges": ["a b", "c d", "e f", "ab cd", "abcd ef"]
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode_with("abcdef", &no_bos());
    assert_eq!(
        tokens,
        vec![14],
        "aggressive reduction: 'abcdef' → [abcdef=14], got: {tokens:?}"
    );
}

// ===========================================================================
// Merge-implied token creation and boundary behavior
// ===========================================================================

/// BPE must support merge rules whose concatenated token is not present in the
/// original vocab by auto-creating that token internally.
#[test]
fn merge_implied_token_auto_created_when_missing_from_vocab() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "a": 4, "b": 5
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
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode_with("ab", &no_bos());
    assert_eq!(
        tokens.len(),
        1,
        "merge-implied token should produce one token, got: {tokens:?}"
    );
    let decoded = ctx.decode(&tokens);
    assert_eq!(
        decoded, "ab",
        "auto-created merged token must decode back to 'ab'"
    );
}

/// Cascaded merges should also work when intermediate merge products are not
/// explicitly present in the original vocab.
#[test]
fn cascade_merges_auto_create_missing_intermediate_tokens() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "a": 4, "b": 5, "c": 6, "ab": 7
    },
    "merges": ["a b", "ab c"]
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode_with("abc", &no_bos());
    assert_eq!(
        tokens.len(),
        1,
        "cascade should reduce to one token, got: {tokens:?}"
    );
    let decoded = ctx.decode(&tokens);
    assert_eq!(
        decoded, "abc",
        "cascade token auto-created from 'ab c' must decode back to 'abc'"
    );
}

/// With a whitespace pre-tokenizer, merge rules must not cross tokenized word
/// boundaries (e.g., "a b" should never merge to "ab").
#[test]
fn merges_do_not_cross_whitespace_pretokenizer_boundaries() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "a": 4, "b": 5, "ab": 6
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
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);

    let split_tokens = ctx.encode_with("a b", &no_bos());
    assert_eq!(
        split_tokens,
        vec![4, 5],
        "Whitespace pre-tokenizer must block the cross-word merge and drop the separator itself"
    );
    assert!(
        !split_tokens.contains(&6),
        "cross-word merge token 'ab' must not appear for input with whitespace boundary"
    );

    let merged_tokens = ctx.encode_with("ab", &no_bos());
    assert_eq!(
        merged_tokens,
        vec![6],
        "same merge rule should still apply within one pretokenized word"
    );
}

/// String-form and array-form merge encodings in tokenizer JSON must be
/// behaviorally equivalent.
#[test]
fn merge_string_and_array_formats_are_behaviorally_equivalent() {
    let json_string_merges = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "a": 4, "b": 5, "c": 6, "ab": 7, "abc": 8
    },
    "merges": ["a b", "ab c"]
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let json_array_merges = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "a": 4, "b": 5, "c": 6, "ab": 7, "abc": 8
    },
    "merges": [["a", "b"], ["ab", "c"]]
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let a = TokenizerTestContext::from_json(json_string_merges);
    let b = TokenizerTestContext::from_json(json_array_merges);
    let cases = ["abc", "ab", "a", "cab", "abcabc"];

    for text in cases {
        let ids_a = a.encode_with(text, &no_bos());
        let ids_b = b.encode_with(text, &no_bos());
        assert_eq!(
            ids_a, ids_b,
            "merge format mismatch for input {text:?}: string={ids_a:?} array={ids_b:?}"
        );
    }
}

/// Auto-created merged tokens must be consistent across encode, tokenize, and
/// tokenize_bytes surfaces.
#[test]
fn auto_created_merge_token_consistent_across_surfaces() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "a": 4, "b": 5
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
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let text = "ab";
    let ids = ctx.encode_with(text, &no_bos());
    assert_eq!(ids.len(), 1, "encode must produce one merged token");

    let tok = unsafe {
        talu_sys::talu_tokenizer_tokenize(ctx.handle(), text.as_bytes().as_ptr(), text.len())
    };
    assert!(tok.error_msg.is_null(), "tokenize should succeed");
    assert_eq!(tok.num_tokens, 1, "tokenize must produce one token");
    let ptrs =
        unsafe { std::slice::from_raw_parts(tok.tokens as *const *const i8, tok.num_tokens) };
    let t0 = unsafe { std::ffi::CStr::from_ptr(ptrs[0]) }
        .to_string_lossy()
        .to_string();
    assert_eq!(t0, "ab", "tokenize token text must be merged token");
    unsafe { talu_sys::talu_tokenize_result_free(tok.tokens, tok.num_tokens) };

    let bytes = unsafe {
        talu_sys::talu_tokenizer_tokenize_bytes(ctx.handle(), text.as_bytes().as_ptr(), text.len())
    };
    assert!(bytes.error_msg.is_null(), "tokenize_bytes should succeed");
    assert_eq!(bytes.num_tokens, 1, "tokenize_bytes must produce one token");
    let offsets = unsafe { std::slice::from_raw_parts(bytes.offsets, bytes.num_tokens + 1) };
    let data = unsafe { std::slice::from_raw_parts(bytes.data, bytes.data_len) };
    assert_eq!(
        std::str::from_utf8(&data[offsets[0]..offsets[1]]).unwrap(),
        "ab"
    );
    unsafe {
        talu_sys::talu_tokenize_bytes_result_free(
            bytes.data,
            bytes.data_len,
            bytes.offsets,
            bytes.num_tokens,
        )
    };
}

/// Auto-created merged token behavior must be identical between individual and
/// batch encode paths.
#[test]
fn batch_and_individual_agree_for_auto_created_merged_tokens() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "a": 4, "b": 5
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
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let a = ctx.encode_with("ab", &no_bos());
    let b = ctx.encode_with("ba", &no_bos());
    let batch = ctx.encode_batch(&["ab", "ba"], &no_bos());
    assert_eq!(batch.num_sequences, 2);
    assert_eq!(batch.offsets, vec![0, a.len(), a.len() + b.len()]);
    assert_eq!(&batch.ids[0..a.len()], a.as_slice(), "batch seq0 mismatch");
    assert_eq!(
        &batch.ids[a.len()..a.len() + b.len()],
        b.as_slice(),
        "batch seq1 mismatch"
    );
}

/// Offsets for an auto-created merged token must span the full source word.
#[test]
fn offsets_for_auto_created_merged_token_cover_full_span() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "a": 4, "b": 5
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
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let result = unsafe { super::common::encode_raw(ctx.handle(), b"ab", &no_bos()) };
    assert!(result.error_msg.is_null(), "encode should succeed");
    assert_eq!(result.num_tokens, 1, "expected one merged token");
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!(offsets[0].start, 0, "merged token start offset");
    assert_eq!(offsets[0].end, 2, "merged token end offset");
    unsafe { talu_sys::talu_encode_result_free(result) };
}

// ===========================================================================
// Unknown-symbol fallback and long-word boundaries
// ===========================================================================

/// If per-symbol lookup fails but the full word exists in vocab, BPE should
/// fall back to the full-word token instead of emitting per-symbol UNKs.
#[test]
fn unknown_symbols_fallback_to_full_word_vocab_token() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0,
      "foobar": 1
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode_with("foobar", &no_bos());
    assert_eq!(
        tokens,
        vec![1],
        "full-word vocab fallback should return token id 1 for 'foobar'"
    );
}

/// SentencePiece-style byte fallback should map unknown single-byte input to
/// its `<0xNN>` token when available.
#[test]
fn byte_fallback_encodes_unknown_ascii_byte() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "unk_token": "<unk>",
    "vocab": {
      "<unk>": 0,
      "<0x61>": 1
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode_with("a", &no_bos());
    assert_eq!(
        tokens,
        vec![1],
        "unknown 'a' should use <0x61> byte fallback"
    );
}

/// Multi-byte UTF-8 unknown chars should emit one fallback token per byte when
/// all `<0xNN>` tokens exist.
#[test]
fn byte_fallback_encodes_multibyte_utf8_unknown_char() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "unk_token": "<unk>",
    "vocab": {
      "<unk>": 0,
      "<0xC3>": 1,
      "<0xA9>": 2
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode_with("é", &no_bos());
    assert_eq!(
        tokens,
        vec![1, 2],
        "unknown 'é' should emit byte fallback tokens [<0xC3>, <0xA9>]"
    );
}

/// If byte fallback is only partially available, missing bytes must degrade to
/// `<unk>` while available bytes still use `<0xNN>` IDs.
#[test]
fn byte_fallback_partial_table_uses_unk_for_missing_bytes() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "unk_token": "<unk>",
    "vocab": {
      "<unk>": 0,
      "<0xC3>": 1
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode_with("é", &no_bos());
    assert_eq!(
        tokens,
        vec![1, 0],
        "partial byte fallback should emit [<0xC3>, <unk>] for 'é'"
    );
}

/// An unknown multi-byte symbol in the middle of a word must not block valid
/// BPE merges on the known left and right neighbors when byte fallback is used.
#[test]
fn byte_fallback_unknown_middle_preserves_neighbor_merges() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "unk_token": "<unk>",
    "vocab": {
      "<unk>": 0,
      "a": 1,
      "b": 2,
      "c": 3,
      "d": 4,
      "ab": 5,
      "cd": 6,
      "<0xC3>": 7,
      "<0xA9>": 8
    },
    "merges": ["a b", "c d"]
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode_with("abécd", &no_bos());
    assert_eq!(
        tokens,
        vec![5, 7, 8, 6],
        "unknown middle symbol should use byte fallback while preserving ab/cd merges"
    );
}

/// Without byte fallback, the same mixed known/unknown path must still preserve
/// valid merges on both sides and degrade the unknown symbol to per-byte <unk>.
#[test]
fn unknown_middle_without_byte_fallback_preserves_neighbor_merges() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "unk_token": "<unk>",
    "vocab": {
      "<unk>": 0,
      "a": 1,
      "b": 2,
      "c": 3,
      "d": 4,
      "ab": 5,
      "cd": 6
    },
    "merges": ["a b", "c d"]
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode_with("abécd", &no_bos());
    assert_eq!(
        tokens,
        vec![5, 0, 0, 6],
        "unknown middle symbol should degrade to per-byte <unk> while preserving ab/cd merges"
    );
}

/// When byte fallback emits multiple tokens for one unknown source symbol, the
/// surrounding merged tokens and fallback bytes must keep exact source spans.
#[test]
fn byte_fallback_unknown_middle_offsets_cover_merged_and_fallback_spans() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "unk_token": "<unk>",
    "vocab": {
      "<unk>": 0,
      "a": 1,
      "b": 2,
      "c": 3,
      "d": 4,
      "ab": 5,
      "cd": 6,
      "<0xC3>": 7,
      "<0xA9>": 8
    },
    "merges": ["a b", "c d"]
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let result = unsafe { super::common::encode_raw(ctx.handle(), "abécd".as_bytes(), &no_bos()) };
    assert!(result.error_msg.is_null(), "encode failed");
    assert_eq!(result.num_tokens, 4);

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!(ids, &[5, 7, 8, 6]);
    assert_eq!(
        (offsets[0].start, offsets[0].end),
        (0, 2),
        "merged ab must span the first two source bytes"
    );
    assert_eq!(
        (offsets[1].start, offsets[1].end),
        (2, 4),
        "first fallback byte must map to the full é source span"
    );
    assert_eq!(
        (offsets[2].start, offsets[2].end),
        (2, 4),
        "second fallback byte must map to the full é source span"
    );
    assert_eq!(
        (offsets[3].start, offsets[3].end),
        (4, 6),
        "merged cd must span the final two source bytes"
    );

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Words longer than MAX_WORD_SYMBOLS (512) must still be encoded correctly.
/// With only merge `a+a->aa`, 513 'a' should reduce to 256 'aa' + 1 'a'.
#[test]
fn long_word_over_max_word_symbols_encodes_deterministically() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "a": 4, "aa": 5
    },
    "merges": ["a a"]
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let input = "a".repeat(513);
    let tokens = ctx.encode_with(&input, &no_bos());
    assert_eq!(
        tokens.len(),
        257,
        "513 'a' with a+a merge must produce 257 tokens (256 'aa' + 1 'a')"
    );
    assert!(
        tokens[..256].iter().all(|&id| id == 5),
        "first 256 tokens should be 'aa' (id 5)"
    );
    assert_eq!(tokens[256], 4, "final odd symbol should be 'a' (id 4)");
}

/// Very long words must remain deterministic across repeated encodes.
#[test]
fn long_word_over_max_word_symbols_is_deterministic() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "a": 4, "aa": 5
    },
    "merges": ["a a"]
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let input = "a".repeat(1025);
    let first = ctx.encode_with(&input, &no_bos());
    let second = ctx.encode_with(&input, &no_bos());
    assert_eq!(
        first, second,
        "long-word BPE encoding must be deterministic"
    );
}

// ===========================================================================
// Interleaved merge ranks with non-adjacent application
// ===========================================================================

/// Merge rank ordering where lower rank fires in the MIDDLE of the word,
/// and higher rank fires at the EDGES — testing that the algorithm doesn't
/// just scan left-to-right but truly uses rank priority.
///
/// Input: "abcba" (5 symbols → small path)
/// Merges: rank 0: c+b→cb (fires at pos 2-3, middle)
///         rank 1: a+b→ab (fires at pos 0-1, left edge)
/// After rank 0: [a, b, cb, a] → rank 1: a+b at pos 0-1 → [ab, cb, a]
#[test]
fn rank_priority_middle_before_edge() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "a": 4, "b": 5, "c": 6,
      "cb": 7, "ab": 8
    },
    "merges": ["c b", "a b"]
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    // rank 0 (c+b) at pos 2-3 fires first → [a, b, cb, a]
    // rank 1 (a+b) at pos 0-1 → [ab, cb, a]
    let tokens = ctx.encode_with("abcba", &no_bos());
    assert_eq!(
        tokens,
        vec![8, 7, 4],
        "rank priority: 'abcba' → [ab=8, cb=7, a=4], got: {tokens:?}"
    );
}

/// Same scenario on 8 symbols (cached path).
///
/// Input: "xxabcbay" (8 symbols)
/// Merges: rank 0: c+b→cb (fires at pos 4-5, middle)
///         rank 1: a+b→ab (fires at pos 2-3)
/// After rank 0: [x, x, a, b, cb, a, y] → rank 1: [x, x, ab, cb, a, y]
#[test]
fn rank_priority_middle_before_edge_cached() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "x": 4, "a": 5, "b": 6, "c": 7, "y": 8,
      "cb": 9, "ab": 10
    },
    "merges": ["c b", "a b"]
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode_with("xxabcbay", &no_bos());
    assert_eq!(
        tokens,
        vec![4, 4, 10, 9, 5, 8],
        "cached rank priority: 'xxabcbay' → [x, x, ab=10, cb=9, a, y], got: {tokens:?}"
    );
}

// ===========================================================================
// Merge invalidates a pending pair (stale cache entry)
// ===========================================================================

/// Two different merge pairs share a symbol. The first merge consumes
/// the shared symbol, making the second pair stale.
///
/// Input: "abcdefgh" (8 symbols → cached path)
/// Merges: rank 0: b+c→bc, rank 1: c+d→cd
/// The "c" is shared. rank 0 fires first (b+c at pos 1-2),
/// consuming "c". rank 1 (c+d) becomes stale — "c" no longer exists.
/// Result: [a, bc, d, e, f, g, h]
#[test]
fn stale_pair_after_shared_symbol_consumed() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "a": 4, "b": 5, "c": 6, "d": 7, "e": 8, "f": 9, "g": 10, "h": 11,
      "bc": 12, "cd": 13
    },
    "merges": ["b c", "c d"]
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode_with("abcdefgh", &no_bos());
    assert_eq!(
        tokens,
        vec![4, 12, 7, 8, 9, 10, 11],
        "stale pair: 'abcdefgh' → [a, bc=12, d, e, f, g, h], got: {tokens:?}"
    );
}

/// Same-rank pair at two positions with shared middle symbol.
///
/// Input: "xabcbax" (7 symbols → cached path)
/// Merge rank 0: b+c→bc (appears at pos 1-2)
/// Merge rank 1: c+b→cb (appears at pos 2-3, but c consumed by rank 0)
///
/// After rank 0: [x, a, bc, b, a, x] — c is gone, c+b can't fire.
/// Then rank 1 would look for c+b but c is consumed. Stale.
#[test]
fn same_symbol_consumed_prevents_adjacent_merge() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "x": 4, "a": 5, "b": 6, "c": 7,
      "bc": 8, "cb": 9
    },
    "merges": ["b c", "c b"]
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    // "xabcbax" → [x, a, b, c, b, a, x]
    // rank 0 (b+c at pos 2-3) fires → [x, a, bc, b, a, x]
    // rank 1 (c+b) — no "c" left. Stale.
    let tokens = ctx.encode_with("xabcbax", &no_bos());
    assert_eq!(
        tokens,
        vec![4, 5, 8, 6, 5, 4],
        "consumed prevents adjacent: 'xabcbax' → [x, a, bc=8, b, a, x], got: {tokens:?}"
    );
}

// ===========================================================================
// Parallel independent merges (non-overlapping pairs)
// ===========================================================================

/// Multiple independent merges fire at the same rank level.
///
/// Input: "abcdabcd" (8 symbols → cached path)
/// Merges: rank 0: a+b→ab, rank 1: c+d→cd
///
/// Both pairs are independent (non-overlapping). After rank 0:
/// [ab, c, d, ab, c, d]. After rank 1: [ab, cd, ab, cd].
#[test]
fn parallel_independent_merges() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "a": 4, "b": 5, "c": 6, "d": 7,
      "ab": 8, "cd": 9
    },
    "merges": ["a b", "c d"]
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode_with("abcdabcd", &no_bos());
    assert_eq!(
        tokens,
        vec![8, 9, 8, 9],
        "parallel merges: 'abcdabcd' → [ab, cd, ab, cd], got: {tokens:?}"
    );
}

// ===========================================================================
// Merge chain across the entire word (complete reduction)
// ===========================================================================

/// 7-level cascade that reduces 8 symbols to 1 token.
///
/// Stresses cache management: every merge reduces symbol count by 1,
/// requiring 7 cache updates with index shifts.
#[test]
fn full_reduction_eight_to_one() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "a": 4, "b": 5, "c": 6, "d": 7, "e": 8, "f": 9, "g": 10, "h": 11,
      "ab": 12, "abc": 13, "abcd": 14, "abcde": 15,
      "abcdef": 16, "abcdefg": 17, "abcdefgh": 18
    },
    "merges": ["a b", "ab c", "abc d", "abcd e", "abcde f", "abcdef g", "abcdefg h"]
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode_with("abcdefgh", &no_bos());
    assert_eq!(
        tokens,
        vec![18],
        "full 8→1 reduction: 'abcdefgh' → [abcdefgh=18], got: {tokens:?}"
    );
}

// ===========================================================================
// Merge with repeated pair where second instance is invalidated
// ===========================================================================

/// Input has the same pair THREE times. After leftmost merge, the
/// symbols shift and we verify the remaining instances still merge.
///
/// Input: "ababab" (6 symbols → cached path)
/// Merge rank 0: a+b→ab
/// All three (a,b) pairs are independent → all merge.
/// → [ab, ab, ab]
#[test]
fn triple_same_pair_all_merge() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "a": 4, "b": 5, "ab": 6
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
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode_with("ababab", &no_bos());
    assert_eq!(
        tokens,
        vec![6, 6, 6],
        "triple pair: 'ababab' → [ab, ab, ab], got: {tokens:?}"
    );
}

/// Overlapping repeated pairs: "aaa" with a+a→aa, then aa+a→aaa.
///
/// "aaa" → [a, a, a] → rank 0 leftmost (a,a) at pos 0-1 → [aa, a]
/// → rank 1 (aa,a) → [aaa]
#[test]
fn overlapping_repeated_with_cascade() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "a": 4, "aa": 5, "aaa": 6
    },
    "merges": ["a a", "aa a"]
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode_with("aaa", &no_bos());
    assert_eq!(
        tokens,
        vec![6],
        "overlapping cascade: 'aaa' → [aaa=6], got: {tokens:?}"
    );
}

/// 6 repeated chars with two cascade levels on cached path.
///
/// "aaaaaa" with a+a→aa, aa+a→aaa
/// → [a,a,a,a,a,a] → rank 0 leftmost → [aa,a,a,a,a] → [aa,aa,a,a]
/// → [aa,aa,aa] → rank 1 leftmost (aa,a) — wait, all are "aa" now.
/// Actually: pairs at rank 0: (a,a). After all rank-0 merges: [aa,aa,aa].
/// rank 1: (aa,a) — no "a" left, only "aa". So rank 1 doesn't fire.
/// Result: [aa, aa, aa]
#[test]
fn six_repeated_chars_two_merge_levels() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "a": 4, "aa": 5, "aaa": 6
    },
    "merges": ["a a", "aa a"]
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode_with("aaaaaa", &no_bos());
    assert_eq!(
        tokens,
        vec![5, 5, 5],
        "6 repeated: 'aaaaaa' → [aa, aa, aa], got: {tokens:?}"
    );
}

/// 7 repeated chars: tests odd count with cascade.
///
/// "aaaaaaa" with a+a→aa, aa+a→aaa
/// → rank 0: [aa,aa,aa,a] (3 pairs merged, 1 leftover)
/// → rank 1: (aa,a) at pos 2-3 fires → [aa, aa, aaa]
#[test]
fn seven_repeated_chars_cascade() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "a": 4, "aa": 5, "aaa": 6
    },
    "merges": ["a a", "aa a"]
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode_with("aaaaaaa", &no_bos());
    assert_eq!(
        tokens,
        vec![5, 5, 6],
        "7 repeated cascade: 'aaaaaaa' → [aa, aa, aaa=6], got: {tokens:?}"
    );
}

/// The 5-symbol re-scan path and the 6-symbol cached path must agree on the
/// same cascade family at the threshold where the implementation switches
/// algorithms.
#[test]
fn threshold_boundary_cascade_family_matches_small_and_cached_paths() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      "a": 4, "aa": 5, "aaaa": 6, "aaaaa": 7, "aaaaaa": 8
    },
    "merges": ["a a", "aa aa", "aaaa a", "aaaa aa"]
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);

    let five = ctx.encode_with("aaaaa", &no_bos());
    assert_eq!(
        five,
        vec![7],
        "5-symbol small path must collapse to [aaaaa=7], got: {five:?}"
    );

    let six = ctx.encode_with("aaaaaa", &no_bos());
    assert_eq!(
        six,
        vec![8],
        "6-symbol cached path must collapse to [aaaaaa=8], got: {six:?}"
    );
}

// ===========================================================================
// Unknown token handling fast paths (single/two symbol)
// ===========================================================================

/// Single unknown multi-byte symbol in non-byte-level mode should emit one
/// fallback/unk token per source byte.
#[test]
fn single_unknown_multibyte_non_bytelevel_emits_per_byte_unk() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "unk_token": "<unk>",
    "vocab": { "<unk>": 0 },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode_with("é", &no_bos()); // UTF-8: C3 A9
    assert_eq!(
        tokens,
        vec![0, 0],
        "single unknown multibyte symbol should map per-byte to <unk>"
    );
}

/// Two-symbol fast path: known + unknown multi-byte with byte-fallback should
/// emit known ID followed by fallback byte IDs.
#[test]
fn two_symbol_mixed_known_unknown_with_byte_fallback() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "unk_token": "<unk>",
    "vocab": {
      "<unk>": 0,
      "a": 1,
      "<0xC3>": 2,
      "<0xA9>": 3
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode_with("aé", &no_bos());
    assert_eq!(
        tokens,
        vec![1, 2, 3],
        "known+unknown two-symbol path should emit known token then byte fallback IDs"
    );
}

/// Two-symbol fast path without byte-fallback should emit known ID and then
/// one <unk> per byte of the unknown symbol.
#[test]
fn two_symbol_mixed_known_unknown_without_byte_fallback_uses_unk_per_byte() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "unk_token": "<unk>",
    "vocab": {
      "<unk>": 0,
      "a": 1
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode_with("aé", &no_bos());
    assert_eq!(
        tokens,
        vec![1, 0, 0],
        "without byte-fallback, unknown multi-byte symbol should emit per-byte <unk>"
    );
}

/// Byte-level mode with incomplete raw-byte mapping should emit a single <unk>
/// for unknown single-byte symbols on the single-symbol fast path.
#[test]
fn single_unknown_bytelevel_symbol_emits_single_unk() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "unk_token": "<unk>",
    "vocab": { "<unk>": 0 },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let tokens = ctx.encode_with("a", &no_bos());
    assert_eq!(
        tokens,
        vec![0],
        "byte-level unknown single symbol should emit one <unk>, not per-byte expansion"
    );
}

// ===========================================================================
// Non-byte-level large-word boundary (> MAX_WORD_SYMBOLS by char count)
// ===========================================================================

/// Non-byte-level path with multi-byte UTF-8 chars must handle >512 symbols
/// without crashing and with correct pair-merge cardinality.
#[test]
fn non_bytelevel_multibyte_long_word_over_symbol_limit() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "unk_token": "<unk>",
    "vocab": {
      "<unk>": 0,
      "é": 1,
      "éé": 2
    },
    "merges": ["é é"]
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let input = "é".repeat(513);
    let tokens = ctx.encode_with(&input, &no_bos());
    assert_eq!(
        tokens.len(),
        257,
        "513 UTF-8 symbols with é+é merge should produce 257 tokens"
    );
    assert!(
        tokens[..256].iter().all(|&id| id == 2),
        "first 256 tokens should be merged 'éé' (id 2)"
    );
    assert_eq!(tokens[256], 1, "final odd symbol should be 'é' (id 1)");
}

/// Exactly MAX_WORD_SYMBOLS (512) non-byte-level UTF-8 symbols should remain
/// stable and follow expected merge cardinality without heap-overflow issues.
#[test]
fn non_bytelevel_multibyte_exact_symbol_limit() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "unk_token": "<unk>",
    "vocab": {
      "<unk>": 0,
      "é": 1,
      "éé": 2
    },
    "merges": ["é é"]
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let input = "é".repeat(512);
    let tokens = ctx.encode_with(&input, &no_bos());
    assert_eq!(
        tokens.len(),
        256,
        "512 symbols should reduce to 256 merged tokens"
    );
    assert!(
        tokens.iter().all(|&id| id == 2),
        "all tokens at 512 boundary should be merged 'éé' (id 2)"
    );
}

/// Repeated encoding of long non-byte-level UTF-8 words should be deterministic.
#[test]
fn non_bytelevel_multibyte_long_word_repeated_is_deterministic() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "unk_token": "<unk>",
    "vocab": {
      "<unk>": 0,
      "é": 1,
      "éé": 2
    },
    "merges": ["é é"]
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let input = "é".repeat(513);
    let first = ctx.encode_with(&input, &no_bos());
    let second = ctx.encode_with(&input, &no_bos());
    assert_eq!(
        first, second,
        "repeated long non-byte-level UTF-8 encoding must be deterministic"
    );
}
