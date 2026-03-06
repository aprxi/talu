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
    assert_eq!(
        tokens,
        vec![5, 5],
        "'aaaa' → [aa=5, aa=5], got: {tokens:?}"
    );
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
