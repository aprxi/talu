//! WordPiece model tests.
//!
//! Tests specific to the WordPiece tokenization algorithm: subword splitting,
//! `##` continuing-subword prefix handling in encode and decode, unknown word
//! fallback, and model type detection.

use crate::capi::tokenizer::common::TokenizerTestContext;

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
    let tokens = if result.tokens.is_null() || result.num_tokens == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(result.tokens, result.num_tokens) }
            .iter()
            .map(|ptr| {
                unsafe { std::ffi::CStr::from_ptr(*ptr) }
                    .to_str()
                    .unwrap()
                    .to_owned()
            })
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

/// Minimal WordPiece tokenizer with BertPreTokenizer and `##` prefix.
const WORDPIECE_JSON: &str = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 100,
    "vocab": {
      "[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3,
      "hello": 4, "world": 5, "go": 6, "good": 7, "morning": 8,
      "##ing": 9, "##ed": 10, "##lab": 11, "##s": 12,
      ",": 13, "!": 14, ".": 15
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true},
    {"id": 1, "content": "[CLS]", "special": true},
    {"id": 2, "content": "[SEP]", "special": true},
    {"id": 3, "content": "[PAD]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}
}"####;

// ---------------------------------------------------------------------------
// Encode: subword splitting
// ---------------------------------------------------------------------------

/// Known whole-word encodes to a single token.
#[test]
fn encode_whole_word() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    assert_eq!(ctx.encode("hello"), vec![4]);
}

/// Word + subword suffix: "going" → ["go", "##ing"].
#[test]
fn encode_subword_split() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    assert_eq!(ctx.encode("going"), vec![6, 9]);
}

/// Multiple words: "hello world" → ["hello", "world"].
#[test]
fn encode_multiple_words() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    assert_eq!(ctx.encode("hello world"), vec![4, 5]);
}

/// Unknown word falls back to [UNK].
#[test]
fn encode_unknown_word() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    // "xyz" is not in vocab and can't be split into known subwords
    let ids = ctx.encode("xyz");
    assert_eq!(ids, vec![0], "unknown word should produce [UNK] token");
}

/// Punctuation is split by BertPreTokenizer.
#[test]
fn encode_punctuation_split() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    // "hello, world!" → ["hello", ",", "world", "!"]
    assert_eq!(ctx.encode("hello, world!"), vec![4, 13, 5, 14]);
}

/// Whitespace-only input produces empty output (no segfault).
#[test]
fn encode_whitespace_only() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    assert_eq!(ctx.encode("   "), Vec::<u32>::new());
}

/// Empty input produces empty output.
#[test]
fn encode_empty() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    assert_eq!(ctx.encode(""), Vec::<u32>::new());
}

/// Tokenization surfaces must expose the `##` continuation prefix for a
/// subword split exactly as the model vocabulary stores it.
#[test]
fn tokenize_surfaces_expose_continuing_subword_prefix() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    assert_eq!(tokenize_strings(&ctx, "going"), vec!["go", "##ing"]);
    assert_eq!(tokenize_bytes_strings(&ctx, "going"), vec!["go", "##ing"]);
}

/// Raw encode offsets for a WordPiece split must map each subword back to its
/// exact byte range within the original word.
#[test]
fn offsets_for_subword_split_follow_original_word_boundaries() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    let result = unsafe { super::common::encode_raw(ctx.handle(), b"going", &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 2);

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    assert_eq!(ids, &[6, 9]);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 2));
    assert_eq!((offsets[1].start, offsets[1].end), (2, 5));

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// BertPreTokenizer punctuation splitting must be visible on both tokenization
/// surfaces, not only on the ID encode path.
#[test]
fn tokenize_surfaces_show_consecutive_punctuation_split() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    assert_eq!(
        tokenize_strings(&ctx, "hello..."),
        vec!["hello", ".", ".", "."]
    );
    assert_eq!(
        tokenize_bytes_strings(&ctx, "hello..."),
        vec!["hello", ".", ".", "."]
    );
}

/// Consecutive punctuation offsets must remain exact after BertPreTokenizer
/// splitting, with no dropped or merged punctuation spans.
#[test]
fn offsets_for_consecutive_punctuation_are_exact() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    let result = unsafe { super::common::encode_raw(ctx.handle(), b"hello...", &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 4);

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    assert_eq!(ids, &[4, 15, 15, 15]);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 5));
    assert_eq!((offsets[1].start, offsets[1].end), (5, 6));
    assert_eq!((offsets[2].start, offsets[2].end), (6, 7));
    assert_eq!((offsets[3].start, offsets[3].end), (7, 8));

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Batch WordPiece encoding must slice into the same per-sequence IDs as
/// individual encoding for mixed subword and punctuation-heavy inputs.
#[test]
fn batch_matches_individual_for_subword_and_punctuation_cases() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    let batch = ctx.encode_batch(&["going", "hello...", "xyz"], &no_bos());
    assert_eq!(batch.num_sequences, 3);
    assert_eq!(batch.offsets, vec![0, 2, 6, 7]);

    assert_eq!(
        batch.ids[batch.offsets[0]..batch.offsets[1]],
        ctx.encode_with("going", &no_bos())
    );
    assert_eq!(
        batch.ids[batch.offsets[1]..batch.offsets[2]],
        ctx.encode_with("hello...", &no_bos())
    );
    assert_eq!(
        batch.ids[batch.offsets[2]..batch.offsets[3]],
        ctx.encode_with("xyz", &no_bos())
    );
}

/// Unknown WordPiece words must surface as a single `[UNK]` token on both
/// tokenization APIs, not as dropped text or per-character fallbacks.
#[test]
fn tokenize_surfaces_emit_single_unk_for_unknown_word() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    assert_eq!(tokenize_strings(&ctx, "xyz"), vec!["[UNK]"]);
    assert_eq!(tokenize_bytes_strings(&ctx, "xyz"), vec!["[UNK]"]);
}

/// A whole unknown word must keep ownership of its entire source span in the
/// encode offsets, even though it collapses to one `[UNK]` token.
#[test]
fn offsets_unknown_word_cover_full_word_span() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    let result = unsafe { super::common::encode_raw(ctx.handle(), b"xyz", &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 1);

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    assert_eq!(ids, &[0]);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 3));

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// An unknown word in the middle of a sentence must own only its middle source
/// span, not collapse to zero or steal neighboring word spans.
#[test]
fn offsets_unknown_middle_word_keep_neighbor_boundaries() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 100,
    "vocab": {
      "[UNK]": 0, "hello": 1, "world": 2
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let result = unsafe { super::common::encode_raw(ctx.handle(), b"hello xyz world", &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 3);

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    assert_eq!(ids, &[1, 0, 2]);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 5));
    assert_eq!((offsets[1].start, offsets[1].end), (6, 9));
    assert_eq!((offsets[2].start, offsets[2].end), (10, 15));

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// A literal continuing-subword prefix in raw input must not be treated as a
/// position-0 WordPiece token. BertPreTokenizer splits the punctuation, so the
/// raw text "##ing" becomes "#" "#" "ing", not the vocab token "##ing".
#[test]
fn encode_literal_continuing_prefix_at_word_start_is_not_promoted_to_subword_id() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    assert_eq!(ctx.encode("##ing"), vec![0, 0, 0]);
    assert_eq!(
        tokenize_strings(&ctx, "##ing"),
        vec!["[UNK]", "[UNK]", "[UNK]"]
    );
    assert_eq!(
        tokenize_bytes_strings(&ctx, "##ing"),
        vec!["[UNK]", "[UNK]", "[UNK]"]
    );

    let result = unsafe { super::common::encode_raw(ctx.handle(), b"##ing", &no_bos()) };
    assert!(result.error_msg.is_null());
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 1));
    assert_eq!((offsets[1].start, offsets[1].end), (1, 2));
    assert_eq!((offsets[2].start, offsets[2].end), (2, 5));
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// A literal "##" prefix before a known whole word must remain punctuation in
/// the raw encode path rather than being stripped into a fake subword match.
#[test]
fn encode_literal_continuing_prefix_before_known_word_preserves_punctuation_split() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    assert_eq!(ctx.encode("##hello"), vec![0, 0, 4]);
    assert_eq!(
        tokenize_strings(&ctx, "##hello"),
        vec!["[UNK]", "[UNK]", "hello"]
    );
    assert_eq!(
        tokenize_bytes_strings(&ctx, "##hello"),
        vec!["[UNK]", "[UNK]", "hello"]
    );
}

/// When skip_special_tokens removes a leading special token, the first
/// remaining WordPiece subword must still decode as position 0 and preserve its
/// `##` prefix.
#[test]
fn skip_special_then_leading_subword_preserves_prefix() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    let skip = talu_sys::DecodeOptionsC {
        skip_special_tokens: 1,
    };
    assert_eq!(ctx.decode_with(&[1, 9, 2], &skip), "##ing");
}

/// The max-input-chars guard is a separate path from ordinary unknown lookup;
/// it must still surface as a single `[UNK]` token on both tokenization APIs.
#[test]
fn tokenize_surfaces_emit_single_unk_for_over_limit_word() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 5,
    "vocab": {
      "[UNK]": 0, "hello": 1, "a": 2, "##a": 3
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    assert_eq!(tokenize_strings(&ctx, "aaaaaa"), vec!["[UNK]"]);
    assert_eq!(tokenize_bytes_strings(&ctx, "aaaaaa"), vec!["[UNK]"]);
}

/// The max-input-chars `[UNK]` path must also preserve the original word span
/// in encode offsets.
#[test]
fn offsets_over_limit_word_cover_full_word_span() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 5,
    "vocab": {
      "[UNK]": 0, "hello": 1, "a": 2, "##a": 3
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let result = unsafe { super::common::encode_raw(ctx.handle(), b"aaaaaa", &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 1);

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    assert_eq!(ids, &[0]);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 6));

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// A multibyte unknown codepoint in the middle of a sentence must still own its
/// exact UTF-8 byte span when collapsed to `[UNK]`.
#[test]
fn offsets_multibyte_unknown_middle_word_keep_exact_span() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 100,
    "vocab": {
      "[UNK]": 0, "hello": 1, "world": 2
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let input = "hello 🌍 world";
    let result = unsafe { super::common::encode_raw(ctx.handle(), input.as_bytes(), &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 3);

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    assert_eq!(ids, &[1, 0, 2]);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 5));
    assert_eq!((offsets[1].start, offsets[1].end), (6, 10));
    assert_eq!((offsets[2].start, offsets[2].end), (11, 16));

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// The max-input-chars guard must preserve neighboring word spans when the
/// over-limit word appears in the middle of a sentence.
#[test]
fn offsets_over_limit_middle_word_keep_neighbor_boundaries() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 5,
    "vocab": {
      "[UNK]": 0, "hello": 1, "world": 2, "a": 3, "##a": 4
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let result =
        unsafe { super::common::encode_raw(ctx.handle(), b"hello aaaaaa world", &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 3);

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    assert_eq!(ids, &[1, 0, 2]);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 5));
    assert_eq!((offsets[1].start, offsets[1].end), (6, 12));
    assert_eq!((offsets[2].start, offsets[2].end), (13, 18));

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// The tokenization APIs must still expose a single `[UNK]` token for a
/// multibyte unknown word, not drop it or split its UTF-8 bytes.
#[test]
fn tokenize_surfaces_emit_single_unk_for_multibyte_unknown_word() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 100,
    "vocab": {
      "[UNK]": 0, "hello": 1, "world": 2
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    assert_eq!(tokenize_strings(&ctx, "🌍"), vec!["[UNK]"]);
    assert_eq!(tokenize_bytes_strings(&ctx, "🌍"), vec!["[UNK]"]);
}

// ---------------------------------------------------------------------------
// Decode: ## prefix handling
// ---------------------------------------------------------------------------

/// Single `##` subword at position 0 preserves the prefix.
///
/// HuggingFace's WordPiece decoder only strips `##` from non-first tokens
/// (`i != 0`). A `##` token decoded alone (position 0) keeps its prefix.
#[test]
fn decode_single_subword_preserves_prefix() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    assert_eq!(ctx.decode(&[11]), "##lab");
}

/// Single `##ing` at position 0 preserves the prefix.
#[test]
fn decode_single_subword_ing_preserves_prefix() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    assert_eq!(ctx.decode(&[9]), "##ing");
}

/// In multi-token context, ## prefix is stripped and tokens are joined.
/// "go" + "##ing" → "going"
#[test]
fn decode_subword_in_context_strips_prefix() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    assert_eq!(ctx.decode(&[6, 9]), "going");
}

/// Multiple words with subwords: "go" + "##ing" + "good" → "going good"
#[test]
fn decode_mixed_words_and_subwords() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    assert_eq!(ctx.decode(&[6, 9, 7]), "going good");
}

/// Whole-word token decode (no prefix).
#[test]
fn decode_whole_word() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    assert_eq!(ctx.decode(&[4]), "hello");
}

/// Roundtrip: encode then decode.
#[test]
fn roundtrip_going() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    let ids = ctx.encode("going");
    assert_eq!(ctx.decode(&ids), "going");
}

/// Roundtrip: multi-word sentence.
#[test]
fn roundtrip_sentence() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    let ids = ctx.encode("good morning, world!");
    assert_eq!(ctx.decode(&ids), "good morning, world!");
}

// ---------------------------------------------------------------------------
// Decode cleanup: only specific punctuation removes leading space
// ---------------------------------------------------------------------------

/// Cleanup must NOT remove space before colon, closing-paren, or percent.
///
/// `clean_up_tokenization_spaces` only removes space before `.` `?` `!` `,`
/// `'` and `-`. Space before `)`, `:`, and `%` must be preserved.
#[test]
fn cleanup_preserves_space_before_closing_paren_and_colon() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "vocab": {
      "[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
      "hello": 3, "world": 4,
      "(": 5, ")": 6, ":": 7, "%": 8, "100": 9
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true},
    {"id": 1, "content": "[CLS]", "special": true},
    {"id": 2, "content": "[SEP]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    // "(" + "100" + "%" + ")" → "( 100 % )"
    // Cleanup should NOT remove space before ) or %
    let decoded = ctx.decode(&[5, 9, 8, 6]);
    assert_eq!(
        decoded, "( 100 % )",
        "cleanup must not remove space before ) or %, got: {decoded:?}"
    );

    // "hello" + ":" → "hello :" (space before colon preserved)
    let decoded = ctx.decode(&[3, 7]);
    assert_eq!(
        decoded, "hello :",
        "cleanup must not remove space before :, got: {decoded:?}"
    );
}

/// cleanup=false must preserve spaces before punctuation that cleanup=true would strip.
#[test]
fn cleanup_false_preserves_space_before_comma_and_question() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "vocab": {
      "[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
      "hello": 3, ",": 4, "world": 5, "?": 6
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true},
    {"id": 1, "content": "[CLS]", "special": true},
    {"id": 2, "content": "[SEP]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let decoded = ctx.decode(&[3, 4, 5, 6]);
    assert_eq!(
        decoded, "hello , world ?",
        "cleanup=false must preserve spaces before comma/question, got: {decoded:?}"
    );
}

/// cleanup=false must preserve contraction spacing.
#[test]
fn cleanup_false_preserves_contraction_spacing() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "vocab": {
      "[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
      "i": 3, "'m": 4
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true},
    {"id": 1, "content": "[CLS]", "special": true},
    {"id": 2, "content": "[SEP]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let decoded = ctx.decode(&[3, 4]);
    assert_eq!(
        decoded, "i 'm",
        "cleanup=false must preserve space before apostrophe contraction, got: {decoded:?}"
    );
}

/// Wrapping a WordPiece decoder in a Sequence must preserve the same runtime
/// behavior, including `cleanup=false`.
#[test]
fn nested_wordpiece_decoder_matches_flat_behavior() {
    let flat_json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "vocab": {
      "[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
      "i": 3, "'m": 4
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true},
    {"id": 1, "content": "[CLS]", "special": true},
    {"id": 2, "content": "[SEP]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
    "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false}
}"####;
    let nested_json = flat_json.replace(
        r###""decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false}"###,
        r###""decoder": {
    "type": "Sequence",
    "decoders": [
      {"type": "WordPiece", "prefix": "##", "cleanup": false}
    ]
  }"###,
    );

    let flat = TokenizerTestContext::from_json(flat_json);
    let nested = TokenizerTestContext::from_json(&nested_json);
    assert_eq!(
        nested.decode(&[3, 4]),
        flat.decode(&[3, 4]),
        "nested WordPiece decoder must match flat behavior"
    );
}

/// A root nested WordPiece subtree must also preserve the exact runtime
/// behavior of the flat decoder.
#[test]
fn doubly_nested_wordpiece_decoder_matches_flat_behavior() {
    let flat_json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "vocab": {
      "[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
      "i": 3, "'m": 4
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true},
    {"id": 1, "content": "[CLS]", "special": true},
    {"id": 2, "content": "[SEP]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false}
}"####;
    let nested_json = flat_json.replace(
        r###""decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false}"###,
        r###""decoder": {
    "type": "Sequence",
    "decoders": [
      {
        "type": "Sequence",
        "decoders": [
          {"type": "WordPiece", "prefix": "##", "cleanup": false}
        ]
      }
    ]
  }"###,
    );

    let flat = TokenizerTestContext::from_json(flat_json);
    let nested = TokenizerTestContext::from_json(&nested_json);
    assert_eq!(
        nested.decode(&[3, 4]),
        flat.decode(&[3, 4]),
        "doubly nested WordPiece decoder must match flat behavior"
    );
}

/// Root nested WordPiece with `cleanup=true` must preserve the exact cleanup
/// behavior of the flat decoder.
#[test]
fn doubly_nested_wordpiece_decoder_cleanup_true_matches_flat_behavior() {
    let flat_json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "vocab": {
      "[UNK]": 0,
      "hello": 1, ",": 2, "world": 3, "?": 4
    }
  },
  "added_tokens": [{"id": 0, "content": "[UNK]", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}
}"####;
    let nested_json = flat_json.replace(
        r###""decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}"###,
        r###""decoder": {
    "type": "Sequence",
    "decoders": [
      {
        "type": "Sequence",
        "decoders": [
          {"type": "WordPiece", "prefix": "##", "cleanup": true}
        ]
      }
    ]
  }"###,
    );

    let flat = TokenizerTestContext::from_json(flat_json);
    let nested = TokenizerTestContext::from_json(&nested_json);
    assert_eq!(
        nested.decode(&[1, 2, 3, 4]),
        flat.decode(&[1, 2, 3, 4]),
        "doubly nested WordPiece decoder with cleanup=true must match flat behavior"
    );
}

/// Root nested WordPiece with `cleanup=true` must also preserve contraction
/// cleanup behavior, not just punctuation cleanup.
#[test]
fn doubly_nested_wordpiece_cleanup_true_contraction_matches_flat_behavior() {
    let flat_json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "vocab": {
      "[UNK]": 0,
      "i": 1, "'m": 2
    }
  },
  "added_tokens": [{"id": 0, "content": "[UNK]", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}
}"####;
    let nested_json = flat_json.replace(
        r###""decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}"###,
        r###""decoder": {
    "type": "Sequence",
    "decoders": [
      {
        "type": "Sequence",
        "decoders": [
          {"type": "WordPiece", "prefix": "##", "cleanup": true}
        ]
      }
    ]
  }"###,
    );

    let flat = TokenizerTestContext::from_json(flat_json);
    let nested = TokenizerTestContext::from_json(&nested_json);
    assert_eq!(
        nested.decode(&[1, 2]),
        flat.decode(&[1, 2]),
        "doubly nested WordPiece cleanup=true contraction behavior must match flat"
    );
}

/// Root nested WordPiece with `cleanup=true` must also preserve punctuation
/// cleanup behavior when special tokens are retained or skipped.
#[test]
fn doubly_nested_wordpiece_cleanup_true_punctuation_with_specials_matches_flat_behavior() {
    let flat_json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "vocab": {
      "[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
      "hello": 3, ",": 4, "world": 5, "?": 6
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true},
    {"id": 1, "content": "[CLS]", "special": true},
    {"id": 2, "content": "[SEP]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}
}"####;
    let nested_json = flat_json.replace(
        r###""decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}"###,
        r###""decoder": {
    "type": "Sequence",
    "decoders": [
      {
        "type": "Sequence",
        "decoders": [
          {"type": "WordPiece", "prefix": "##", "cleanup": true}
        ]
      }
    ]
  }"###,
    );

    let flat = TokenizerTestContext::from_json(flat_json);
    let nested = TokenizerTestContext::from_json(&nested_json);
    let keep = talu_sys::DecodeOptionsC {
        skip_special_tokens: 0,
    };
    let skip = talu_sys::DecodeOptionsC {
        skip_special_tokens: 1,
    };
    let ids = [1, 3, 4, 5, 6, 2];

    assert_eq!(
        nested.decode_with(&ids, &keep),
        flat.decode_with(&ids, &keep),
        "doubly nested WordPiece cleanup=true punctuation retain-special behavior must match flat"
    );
    assert_eq!(
        nested.decode_with(&ids, &skip),
        flat.decode_with(&ids, &skip),
        "doubly nested WordPiece cleanup=true punctuation skip-special behavior must match flat"
    );
}

/// Root nested WordPiece with `cleanup=false` must preserve the same
/// skip-special behavior as the flat decoder.
#[test]
fn doubly_nested_wordpiece_cleanup_false_skip_special_matches_flat_behavior() {
    let flat_json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "vocab": {
      "[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
      "i": 3, "'m": 4
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true},
    {"id": 1, "content": "[CLS]", "special": true},
    {"id": 2, "content": "[SEP]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false}
}"####;
    let nested_json = flat_json.replace(
        r###""decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false}"###,
        r###""decoder": {
    "type": "Sequence",
    "decoders": [
      {
        "type": "Sequence",
        "decoders": [
          {"type": "WordPiece", "prefix": "##", "cleanup": false}
        ]
      }
    ]
  }"###,
    );

    let flat = TokenizerTestContext::from_json(flat_json);
    let nested = TokenizerTestContext::from_json(&nested_json);
    let keep = talu_sys::DecodeOptionsC {
        skip_special_tokens: 0,
    };
    let skip = talu_sys::DecodeOptionsC {
        skip_special_tokens: 1,
    };

    assert_eq!(
        nested.decode_with(&[1, 3, 4, 2], &keep),
        flat.decode_with(&[1, 3, 4, 2], &keep),
        "doubly nested WordPiece cleanup=false retain-special behavior must match flat"
    );
    assert_eq!(
        nested.decode_with(&[1, 3, 4, 2], &skip),
        flat.decode_with(&[1, 3, 4, 2], &skip),
        "doubly nested WordPiece cleanup=false skip-special behavior must match flat"
    );
}

/// Even when special tokens are stripped, a doubly nested WordPiece decoder
/// with `cleanup=false` must preserve the same non-cleanup spacing as the flat
/// decoder.
#[test]
fn doubly_nested_wordpiece_cleanup_false_skip_only_matches_flat_behavior() {
    let flat_json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "vocab": {
      "[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
      "i": 3, "'m": 4
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true},
    {"id": 1, "content": "[CLS]", "special": true},
    {"id": 2, "content": "[SEP]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false}
}"####;
    let nested_json = flat_json.replace(
        r###""decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false}"###,
        r###""decoder": {
    "type": "Sequence",
    "decoders": [
      {
        "type": "Sequence",
        "decoders": [
          {"type": "WordPiece", "prefix": "##", "cleanup": false}
        ]
      }
    ]
  }"###,
    );

    let flat = TokenizerTestContext::from_json(flat_json);
    let nested = TokenizerTestContext::from_json(&nested_json);
    let skip = talu_sys::DecodeOptionsC {
        skip_special_tokens: 1,
    };

    assert_eq!(
        nested.decode_with(&[1, 3, 4, 2], &skip),
        flat.decode_with(&[1, 3, 4, 2], &skip),
        "doubly nested WordPiece cleanup=false skip-only behavior must match flat"
    );
}

/// Null decode options must preserve the same default skip-special behavior on
/// a root nested WordPiece cleanup=false decoder as on the flat decoder.
#[test]
fn doubly_nested_wordpiece_cleanup_false_null_options_matches_flat_behavior() {
    let flat_json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "vocab": {
      "[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
      "i": 3, "'m": 4
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true},
    {"id": 1, "content": "[CLS]", "special": true},
    {"id": 2, "content": "[SEP]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false}
}"####;
    let nested_json = flat_json.replace(
        r###""decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false}"###,
        r###""decoder": {
    "type": "Sequence",
    "decoders": [
      {
        "type": "Sequence",
        "decoders": [
          {"type": "WordPiece", "prefix": "##", "cleanup": false}
        ]
      }
    ]
  }"###,
    );

    let flat = TokenizerTestContext::from_json(flat_json);
    let nested = TokenizerTestContext::from_json(&nested_json);
    let ids = [1, 3, 4, 2];

    let flat_result = unsafe { super::common::decode_raw_null_options(flat.handle(), &ids) };
    assert!(
        flat_result.error_msg.is_null(),
        "flat decode with null options should succeed"
    );
    let flat_text = unsafe {
        let slice = std::slice::from_raw_parts(flat_result.text, flat_result.text_len);
        std::str::from_utf8(slice)
            .expect("flat decode must return valid UTF-8")
            .to_owned()
    };
    unsafe { talu_sys::talu_decode_result_free(flat_result.text, flat_result.text_len) };

    let nested_result = unsafe { super::common::decode_raw_null_options(nested.handle(), &ids) };
    assert!(
        nested_result.error_msg.is_null(),
        "nested decode with null options should succeed"
    );
    let nested_text = unsafe {
        let slice = std::slice::from_raw_parts(nested_result.text, nested_result.text_len);
        std::str::from_utf8(slice)
            .expect("nested decode must return valid UTF-8")
            .to_owned()
    };
    unsafe { talu_sys::talu_decode_result_free(nested_result.text, nested_result.text_len) };

    assert_eq!(
        nested_text, flat_text,
        "doubly nested WordPiece cleanup=false must match flat null-options decode behavior"
    );
}

/// Root nested WordPiece with `cleanup=false` must also preserve punctuation
/// spacing behavior, not just apostrophe-contraction spacing.
#[test]
fn doubly_nested_wordpiece_cleanup_false_punctuation_matches_flat_behavior() {
    let flat_json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "vocab": {
      "[UNK]": 0,
      "hello": 1, ",": 2, "world": 3, "?": 4
    }
  },
  "added_tokens": [{"id": 0, "content": "[UNK]", "special": true}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false}
}"####;
    let nested_json = flat_json.replace(
        r###""decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false}"###,
        r###""decoder": {
    "type": "Sequence",
    "decoders": [
      {
        "type": "Sequence",
        "decoders": [
          {"type": "WordPiece", "prefix": "##", "cleanup": false}
        ]
      }
    ]
  }"###,
    );

    let flat = TokenizerTestContext::from_json(flat_json);
    let nested = TokenizerTestContext::from_json(&nested_json);
    assert_eq!(
        nested.decode(&[1, 2, 3, 4]),
        flat.decode(&[1, 2, 3, 4]),
        "doubly nested WordPiece cleanup=false punctuation behavior must match flat"
    );
}

/// Root nested WordPiece with `cleanup=false` must preserve punctuation
/// spacing behavior even when special tokens are retained or skipped.
#[test]
fn doubly_nested_wordpiece_cleanup_false_punctuation_with_specials_matches_flat_behavior() {
    let flat_json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "vocab": {
      "[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
      "hello": 3, ",": 4, "world": 5, "?": 6
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true},
    {"id": 1, "content": "[CLS]", "special": true},
    {"id": 2, "content": "[SEP]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false}
}"####;
    let nested_json = flat_json.replace(
        r###""decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false}"###,
        r###""decoder": {
    "type": "Sequence",
    "decoders": [
      {
        "type": "Sequence",
        "decoders": [
          {"type": "WordPiece", "prefix": "##", "cleanup": false}
        ]
      }
    ]
  }"###,
    );

    let flat = TokenizerTestContext::from_json(flat_json);
    let nested = TokenizerTestContext::from_json(&nested_json);
    let keep = talu_sys::DecodeOptionsC {
        skip_special_tokens: 0,
    };
    let skip = talu_sys::DecodeOptionsC {
        skip_special_tokens: 1,
    };
    let ids = [1, 3, 4, 5, 6, 2];

    assert_eq!(
        nested.decode_with(&ids, &keep),
        flat.decode_with(&ids, &keep),
        "doubly nested WordPiece cleanup=false punctuation retain-special behavior must match flat"
    );
    assert_eq!(
        nested.decode_with(&ids, &skip),
        flat.decode_with(&ids, &skip),
        "doubly nested WordPiece cleanup=false punctuation skip-special behavior must match flat"
    );
}

/// Even when special tokens are stripped, a doubly nested WordPiece decoder
/// with `cleanup=false` must preserve flat punctuation spacing behavior.
#[test]
fn doubly_nested_wordpiece_cleanup_false_punctuation_skip_only_matches_flat_behavior() {
    let flat_json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "vocab": {
      "[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
      "hello": 3, ",": 4, "world": 5, "?": 6
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true},
    {"id": 1, "content": "[CLS]", "special": true},
    {"id": 2, "content": "[SEP]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false}
}"####;
    let nested_json = flat_json.replace(
        r###""decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false}"###,
        r###""decoder": {
    "type": "Sequence",
    "decoders": [
      {
        "type": "Sequence",
        "decoders": [
          {"type": "WordPiece", "prefix": "##", "cleanup": false}
        ]
      }
    ]
  }"###,
    );

    let flat = TokenizerTestContext::from_json(flat_json);
    let nested = TokenizerTestContext::from_json(&nested_json);
    let skip = talu_sys::DecodeOptionsC {
        skip_special_tokens: 1,
    };
    let ids = [1, 3, 4, 5, 6, 2];

    assert_eq!(
        nested.decode_with(&ids, &skip),
        flat.decode_with(&ids, &skip),
        "doubly nested WordPiece cleanup=false punctuation skip-only behavior must match flat"
    );
}

// ---------------------------------------------------------------------------
// skip_special_tokens for WordPiece
// ---------------------------------------------------------------------------

/// skip_special_tokens=1 must strip [CLS] and [SEP] from decode output.
#[test]
fn skip_special_strips_cls_and_sep() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    let skip = talu_sys::DecodeOptionsC {
        skip_special_tokens: 1,
    };
    // [CLS]=1, hello=4, world=5, [SEP]=2
    assert_eq!(
        ctx.decode_with(&[1, 4, 5, 2], &skip),
        "hello world",
        "skip_special_tokens should strip [CLS] and [SEP]"
    );
}

/// skip_special_tokens=1 on only-special-tokens produces empty string.
#[test]
fn skip_special_all_special_produces_empty() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    let skip = talu_sys::DecodeOptionsC {
        skip_special_tokens: 1,
    };
    assert_eq!(ctx.decode_with(&[1, 2], &skip), "");
}

// ---------------------------------------------------------------------------
// Encode: WordPiece tokenizer with BertProcessing post_processor
// ---------------------------------------------------------------------------
//
// BERT-family models use a BertProcessing post_processor that adds [CLS] at
// the start and [SEP] at the end of encoded output. The post_processor must
// resolve the token strings "[CLS]" and "[SEP]" to their vocab IDs.

/// WordPiece tokenizer with BertProcessing post_processor.
const BERT_POSTPROC_JSON: &str = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 100,
    "vocab": {
      "[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3,
      "hello": 4, "world": 5, ",": 6, "!": 7
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true},
    {"id": 1, "content": "[CLS]", "special": true},
    {"id": 2, "content": "[SEP]", "special": true},
    {"id": 3, "content": "[PAD]", "special": true}
  ],
  "normalizer": {"type": "BertNormalizer", "lowercase": true},
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": {
    "type": "BertProcessing",
    "cls": ["[CLS]", 1],
    "sep": ["[SEP]", 2]
  },
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}
}"####;

// ---------------------------------------------------------------------------
// BertNormalizer: lowercasing before WordPiece lookup
// ---------------------------------------------------------------------------
//
// BERT-uncased models use a BertNormalizer with lowercase=true. The normalizer
// must convert input to lowercase before WordPiece lookup so that "Hello"
// matches the lowercase vocab entry "hello". Without lowercasing, WordPiece
// can't find "Hello" and falls back to character-by-character tokenization.

/// Mixed-case "Hello" must be normalized to "hello" before WordPiece lookup.
#[test]
fn bert_normalizer_lowercases_before_wordpiece() {
    let ctx = TokenizerTestContext::from_json(BERT_POSTPROC_JSON);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    // "Hello" → lowercased "hello" → single token ID 4.
    // Without normalizer: "Hello" → [H, e, l, l, o] character fallback.
    let tokens = ctx.encode_with("Hello", &opts);
    assert_eq!(
        tokens,
        vec![4],
        "BertNormalizer must lowercase 'Hello' → 'hello' (ID 4), got: {tokens:?}"
    );
}

/// Multi-word mixed-case input must be lowercased before WordPiece.
#[test]
fn bert_normalizer_lowercases_multiword() {
    let ctx = TokenizerTestContext::from_json(BERT_POSTPROC_JSON);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    // "Hello, World!" → "hello, world!" → [hello=4, ,=6, world=5, !=7]
    let tokens = ctx.encode_with("Hello, World!", &opts);
    assert_eq!(
        tokens,
        vec![4, 6, 5, 7],
        "BertNormalizer must lowercase before WordPiece, got: {tokens:?}"
    );
}

// ---------------------------------------------------------------------------
// Encode: WordPiece tokenizer with BertProcessing post_processor
// ---------------------------------------------------------------------------

/// BertProcessing adds [CLS] and [SEP] to encode output.
#[test]
fn encode_bert_postproc_adds_cls_sep() {
    let ctx = TokenizerTestContext::from_json(BERT_POSTPROC_JSON);
    // Rust Default zeroes the FFI struct, so add_eos must be explicit for
    // BertProcessing tests that expect [SEP].
    let opts = talu_sys::EncodeOptions {
        add_bos: 1,
        add_eos: 1,
        ..Default::default()
    };
    // "hello" → [hello=4], with CLS/SEP → [CLS=1, hello=4, SEP=2]
    assert_eq!(
        ctx.encode_with("hello", &opts),
        vec![1, 4, 2],
        "BertProcessing must add [CLS]=1 and [SEP]=2"
    );
}

/// BertProcessing: empty input produces [CLS, SEP].
#[test]
fn encode_bert_postproc_empty_produces_cls_sep() {
    let ctx = TokenizerTestContext::from_json(BERT_POSTPROC_JSON);
    let opts = talu_sys::EncodeOptions {
        add_bos: 1,
        add_eos: 1,
        ..Default::default()
    };
    assert_eq!(
        ctx.encode_with("", &opts),
        vec![1, 2],
        "BertProcessing: empty input must produce [CLS, SEP]"
    );
}

// ---------------------------------------------------------------------------
// BertPreTokenizer: consecutive punctuation splitting
// ---------------------------------------------------------------------------
//
// BertPreTokenizer must split every punctuation character individually, even
// when multiple punctuation characters appear consecutively. "hello..." must
// produce ["hello", ".", ".", "."], not ["hello", "..."].

/// Consecutive dots "..." must be split into individual "." tokens.
#[test]
fn bert_pretokenizer_splits_consecutive_dots() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    // "hello..." → BertPreTokenizer → ["hello", ".", ".", "."]
    // WordPiece: "hello"=4, "."=15, "."=15, "."=15
    let tokens = ctx.encode("hello...");
    assert_eq!(
        tokens,
        vec![4, 15, 15, 15],
        "BertPreTokenizer must split '...' into three '.' tokens, got: {tokens:?}"
    );
}

/// Mixed consecutive punctuation "?!" must produce separate tokens.
#[test]
fn bert_pretokenizer_splits_mixed_consecutive_punct() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 100,
    "vocab": {
      "[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
      "hello": 3, "?": 4, "!": 5, ".": 6, ",": 7, ":": 8
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true},
    {"id": 1, "content": "[CLS]", "special": true},
    {"id": 2, "content": "[SEP]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    // "hello?!" → ["hello", "?", "!"]
    let tokens = ctx.encode("hello?!");
    assert_eq!(
        tokens,
        vec![3, 4, 5],
        "BertPreTokenizer must split '?!' into '?' and '!', got: {tokens:?}"
    );

    // "hello:..." → ["hello", ":", ".", ".", "."]
    let tokens = ctx.encode("hello:...");
    assert_eq!(
        tokens,
        vec![3, 8, 6, 6, 6],
        "BertPreTokenizer must split ':...' into individual chars, got: {tokens:?}"
    );
}

// ---------------------------------------------------------------------------
// BertNormalizer: handle_chinese_chars must not split non-CJK characters
// ---------------------------------------------------------------------------
//
// BertNormalizer with handle_chinese_chars=true adds spaces around CJK
// characters (Unicode ranges U+4E00-U+9FFF, etc.) to split them for
// WordPiece. However, non-CJK characters must NOT be affected.

/// BertNormalizer with all options enabled must still produce whole-word tokens.
///
/// When lowercase + clean_text + handle_chinese_chars + strip_accents are all
/// active, ASCII words must remain intact (not split char-by-char).
#[test]
fn bert_normalizer_full_options_whole_word() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 100,
    "vocab": {
      "[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3,
      "hello": 4, "world": 5, ",": 6, "!": 7,
      "h": 8, "e": 9, "l": 10, "o": 11, "w": 12, "r": 13, "d": 14
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true},
    {"id": 1, "content": "[CLS]", "special": true},
    {"id": 2, "content": "[SEP]", "special": true},
    {"id": 3, "content": "[PAD]", "special": true}
  ],
  "normalizer": {
    "type": "BertNormalizer",
    "clean_text": true,
    "handle_chinese_chars": true,
    "strip_accents": true,
    "lowercase": true
  },
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    // "Hello" must normalize to "hello" and match as whole word (ID 4),
    // NOT split into individual characters [h, e, l, l, o] = [8, 9, 10, 10, 11].
    let tokens = ctx.encode_with("Hello", &opts);
    assert_eq!(
        tokens, vec![4],
        "BertNormalizer full options: 'Hello' must match whole word 'hello' (ID 4), got: {tokens:?}"
    );
}

/// BertNormalizer with clean_text: whitespace-only input produces empty output.
///
/// clean_text normalizes whitespace to space, and BertPreTokenizer strips it.
/// The result must be empty, not [UNK] tokens for each space.
#[test]
fn bert_normalizer_clean_text_whitespace_empty() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 100,
    "vocab": {
      "[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
      "hello": 3
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true},
    {"id": 1, "content": "[CLS]", "special": true},
    {"id": 2, "content": "[SEP]", "special": true}
  ],
  "normalizer": {
    "type": "BertNormalizer",
    "clean_text": true,
    "handle_chinese_chars": true,
    "strip_accents": true,
    "lowercase": true
  },
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    // "   " (three spaces) → after clean_text → "   " → BertPreTokenizer strips → []
    let tokens = ctx.encode_with("   ", &opts);
    assert_eq!(
        tokens,
        Vec::<u32>::new(),
        "BertNormalizer + BertPreTokenizer: whitespace-only must produce empty, got: {tokens:?}"
    );
}

// ---------------------------------------------------------------------------
// WordPiece: max_input_chars_per_word enforcement
// ---------------------------------------------------------------------------
//
// When a single word exceeds `max_input_chars_per_word`, WordPiece should
// emit a single [UNK] token for that word instead of attempting subword
// decomposition. This prevents quadratic blowup on adversarial inputs.

/// Words longer than max_input_chars_per_word produce [UNK].
#[test]
fn wordpiece_max_input_chars_per_word_produces_unk() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 10,
    "vocab": {
      "[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
      "hello": 3, "a": 4, "##a": 5
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true},
    {"id": 1, "content": "[CLS]", "special": true},
    {"id": 2, "content": "[SEP]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    // "hello" (5 chars) is within the limit → normal encoding
    let short = ctx.encode_with("hello", &opts);
    assert_eq!(
        short,
        vec![3],
        "word within limit encodes normally, got: {short:?}"
    );
    // "aaaaaaaaaaa" (11 chars) exceeds max_input_chars_per_word=10 → [UNK]
    let long = ctx.encode_with("aaaaaaaaaaa", &opts);
    assert_eq!(
        long,
        vec![0],
        "word exceeding max_input_chars_per_word must produce [UNK], got: {long:?}"
    );
}

// ---------------------------------------------------------------------------
// BertNormalizer: clean_text replaces newlines with spaces
// ---------------------------------------------------------------------------
//
// BertNormalizer's `clean_text` flag should replace control characters
// (including \n, \t, \r) with spaces, not drop them entirely. Dropping
// newlines causes words on adjacent lines to merge into a single token.

/// Newlines between words are replaced with spaces, not dropped.
#[test]
fn bert_normalizer_clean_text_replaces_newline_with_space() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 100,
    "vocab": {
      "[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
      "line": 3, "##1": 4, "##2": 5, "##3": 6
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true},
    {"id": 1, "content": "[CLS]", "special": true},
    {"id": 2, "content": "[SEP]", "special": true}
  ],
  "normalizer": {
    "type": "BertNormalizer",
    "clean_text": true,
    "handle_chinese_chars": false,
    "strip_accents": false,
    "lowercase": false
  },
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    // "line1\nline2" — \n should be replaced with space, giving two words:
    // "line1" → [line, ##1] and "line2" → [line, ##2]
    let tokens = ctx.encode_with("line1\nline2", &opts);
    assert_eq!(
        tokens,
        vec![3, 4, 3, 5],
        "newline must be replaced with space (producing two words), got: {tokens:?}"
    );
}

// ---------------------------------------------------------------------------
// BertProcessing: custom CLS/SEP token names from config
// ---------------------------------------------------------------------------
//
// Post-processor CLS/SEP token IDs must come from the post_processor JSON
// configuration ("cls": ["<s>", 3], "sep": ["</s>", 4]), not from
// hardcoded token names like [CLS]/[SEP]. Models that use <s> and </s> as
// CLS/SEP must have the post-processor honor those configured values.

/// BertProcessing with <s>/<\/s> tokens uses IDs from config, not [CLS]/[SEP].
#[test]
fn bert_postproc_uses_configured_cls_sep_not_defaults() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 200,
    "vocab": {
      "[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
      "<s>": 3, "</s>": 4,
      "hello": 5
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true},
    {"id": 1, "content": "[CLS]", "special": true},
    {"id": 2, "content": "[SEP]", "special": true},
    {"id": 3, "content": "<s>", "special": true},
    {"id": 4, "content": "</s>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": {
    "type": "BertProcessing",
    "cls": ["<s>", 3],
    "sep": ["</s>", 4]
  },
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let opts = talu_sys::EncodeOptions {
        add_bos: 1,
        add_eos: 1,
        ..Default::default()
    };
    // With BertProcessing cls=<s>(3) sep=</s>(4):
    // encode "hello" → [<s>=3, hello=5, </s>=4]
    // NOT [CLS]=1, hello=5, [SEP]=2
    let tokens = ctx.encode_with("hello", &opts);
    assert_eq!(
        tokens,
        vec![3, 5, 4],
        "BertProcessing must use configured <s>=3 and </s>=4, not [CLS]=1/[SEP]=2, got: {tokens:?}"
    );
}

// ---------------------------------------------------------------------------
// WordPiece decode cleanup: space before dash must be preserved
// ---------------------------------------------------------------------------
//
// `clean_up_tokenization_spaces` removes space before a small set of
// punctuation: . ? ! , ' (and contractions like n't, 'm, etc.).
// Space before dash (-) must NOT be removed.

/// Cleanup removes space before comma but preserves space before dash.
#[test]
fn cleanup_removes_comma_space_but_preserves_dash_space() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 200,
    "vocab": {
      "[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
      "hello": 3, "world": 4, ",": 5, "-": 6, "item": 7
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true},
    {"id": 1, "content": "[CLS]", "special": true},
    {"id": 2, "content": "[SEP]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}
}"####;
    let ctx = TokenizerTestContext::from_json(json);

    // "hello" + "," + "world" → cleanup removes space before comma
    let decoded = ctx.decode(&[3, 5, 4]);
    assert_eq!(
        decoded, "hello, world",
        "cleanup must remove space before comma, got: {decoded:?}"
    );

    // "hello" + "-" + "item" → cleanup must NOT remove space before dash
    let decoded = ctx.decode(&[3, 6, 7]);
    assert_eq!(
        decoded, "hello - item",
        "cleanup must preserve space before dash, got: {decoded:?}"
    );
}

// ---------------------------------------------------------------------------
// BertNormalizer: handle_chinese_chars must detect extended CJK ranges
// ---------------------------------------------------------------------------
//
// handle_chinese_chars must check all CJK Unicode ranges, not just the
// basic CJK Unified Ideographs (U+4E00-U+9FFF). Extended ranges include
// CJK Extension A (U+3400-U+4DBF), Extension B (U+20000-U+2A6DF),
// Compatibility Ideographs (U+F900-U+FAFF), and more.

/// CJK Extension A character (U+3400) must be isolated by handle_chinese_chars.
///
/// Input "hello㐀world" without CJK detection: "hello㐀world" is one word →
/// WordPiece character fallback → many tokens. With proper detection:
/// "hello 㐀 world" → ["hello", "㐀", "world"] → [hello, [UNK], world].
#[test]
fn handle_chinese_chars_detects_cjk_extension_a() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 200,
    "vocab": {
      "[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
      "hello": 3, "world": 4
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true},
    {"id": 1, "content": "[CLS]", "special": true},
    {"id": 2, "content": "[SEP]", "special": true}
  ],
  "normalizer": {
    "type": "BertNormalizer",
    "clean_text": true,
    "handle_chinese_chars": true,
    "strip_accents": true,
    "lowercase": true
  },
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    // U+3400 (㐀) is CJK Extension A — handle_chinese_chars must add spaces
    // around it so BertPreTokenizer treats it as a separate word.
    // "hello㐀world" → "hello 㐀 world" → ["hello", "㐀", "world"]
    // → [hello=3, [UNK]=0, world=4]
    let tokens = ctx.encode_with("hello㐀world", &opts);
    assert_eq!(
        tokens,
        vec![3, 0, 4],
        "CJK Extension A (U+3400) must be isolated by handle_chinese_chars, got: {tokens:?}"
    );
}

// ---------------------------------------------------------------------------
// WordPiece encode: emoji (4-byte UTF-8) must produce [UNK], not be dropped
// ---------------------------------------------------------------------------
//
// WordPiece encoder produces [UNK] for any word that can't be decomposed
// into known subwords. Emoji characters (4-byte UTF-8) not in the vocab
// must produce a single [UNK] token, not be silently dropped.

/// Multiple subword splits: "goings" → ["go", "##ing", "##s"].
#[test]
fn encode_multiple_subword_splits() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    // "goings" → "go" + "##ing" + "##s"
    assert_eq!(
        ctx.encode("goings"),
        vec![6, 9, 12],
        "'goings' must split into [go, ##ing, ##s]"
    );
}

/// Greedy longest match: "good" matched as whole word, not "go"+"##od".
#[test]
fn encode_greedy_longest_match() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    // "good" is in vocab as whole word → single token
    assert_eq!(
        ctx.encode("good"),
        vec![7],
        "'good' must match as whole word, not 'go'+'##od'"
    );
}

/// Consecutive unknown words each produce [UNK].
#[test]
fn encode_consecutive_unknown_words() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    // "xyz abc" → ["xyz", "abc"] → each unknown → [UNK, UNK]
    let tokens = ctx.encode("xyz abc");
    assert_eq!(
        tokens,
        vec![0, 0],
        "each unknown word should produce [UNK], got: {tokens:?}"
    );
}

/// Roundtrip encode→decode with subword tokens.
#[test]
fn roundtrip_subword() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    // "going" → [go, ##ing] → decode → "going"
    let ids = ctx.encode("going");
    let decoded = ctx.decode(&ids);
    assert_eq!(decoded, "going", "subword roundtrip failed");
}

/// Roundtrip for sentence with mixed known, subword, and punctuation.
#[test]
fn roundtrip_mixed_sentence() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    let text = "good morning!";
    let ids = ctx.encode(text);
    let decoded = ctx.decode(&ids);
    assert_eq!(decoded, text, "mixed sentence roundtrip failed");
}

/// Consecutive subword tokens decode correctly.
///
/// "go" + "##ing" + "##ed" → "goinged" (artificial but tests decoder).
#[test]
fn decode_consecutive_subwords() {
    let ctx = TokenizerTestContext::from_json(WORDPIECE_JSON);
    let decoded = ctx.decode(&[6, 9, 10]);
    assert_eq!(
        decoded, "goinged",
        "consecutive subwords: go+##ing+##ed → 'goinged', got: {decoded:?}"
    );
}

/// Exactly max_input_chars_per_word chars encodes normally (boundary).
#[test]
fn wordpiece_exactly_at_max_input_chars() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 5,
    "vocab": {
      "[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
      "hello": 3, "a": 4, "##a": 5
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true},
    {"id": 1, "content": "[CLS]", "special": true},
    {"id": 2, "content": "[SEP]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    // "hello" (5 chars) == max_input_chars_per_word → should encode normally
    let tokens = ctx.encode_with("hello", &opts);
    assert_eq!(
        tokens,
        vec![3],
        "word at exactly max_input_chars_per_word must encode normally, got: {tokens:?}"
    );
    // "aaaaaa" (6 chars) > max_input_chars_per_word → [UNK]
    let tokens = ctx.encode_with("aaaaaa", &opts);
    assert_eq!(
        tokens,
        vec![0],
        "word exceeding max_input_chars_per_word must produce [UNK], got: {tokens:?}"
    );
}

/// Emoji character (🌍 U+1F30D) must produce [UNK], not be silently dropped.
#[test]
fn encode_emoji_produces_unk_not_dropped() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 100,
    "vocab": {
      "[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
      "hello": 3, "world": 4
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true},
    {"id": 1, "content": "[CLS]", "special": true},
    {"id": 2, "content": "[SEP]", "special": true}
  ],
  "normalizer": {
    "type": "BertNormalizer",
    "clean_text": true,
    "handle_chinese_chars": true,
    "strip_accents": true,
    "lowercase": true
  },
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    // "hello 🌍 world" → ["hello", "🌍", "world"]
    // "hello" → 3, "🌍" → [UNK]=0, "world" → 4
    let tokens = ctx.encode_with("hello 🌍 world", &opts);
    assert_eq!(
        tokens,
        vec![3, 0, 4],
        "Emoji 🌍 (U+1F30D, 4-byte UTF-8) must produce [UNK], not be dropped, got: {tokens:?}"
    );
}

/// max_input_chars_per_word must be measured in Unicode scalar count, not
/// UTF-8 byte count.
#[test]
fn wordpiece_max_input_chars_uses_unicode_scalars_not_bytes() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 5,
    "vocab": {
      "[UNK]": 0,
      "é": 1,
      "##é": 2
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let opts = no_bos();

    // 5 Unicode scalars, 10 UTF-8 bytes. This should pass the max=5 boundary.
    let within_limit = "ééééé";
    let tokens = ctx.encode_with(within_limit, &opts);
    assert_eq!(
        tokens,
        vec![1, 2, 2, 2, 2],
        "max_input_chars_per_word must use scalar count; 5-char input should not become [UNK]"
    );

    // 6 Unicode scalars should exceed the max and collapse to [UNK].
    let over_limit = "éééééé";
    let tokens = ctx.encode_with(over_limit, &opts);
    assert_eq!(
        tokens,
        vec![0],
        "input above max_input_chars_per_word must produce [UNK]"
    );
}
