//! WordPiece model tests.
//!
//! Tests specific to the WordPiece tokenization algorithm: subword splitting,
//! `##` continuing-subword prefix handling in encode and decode, unknown word
//! fallback, and model type detection.

use crate::capi::tokenizer::common::TokenizerTestContext;

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
/// HuggingFace `clean_up_tokenization_spaces` preserves space before these.
///
/// Bug: cleanup removes space before `)` and `:`, but HF only removes
/// space before `.` `?` `!` `,` `'` and `-`.
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

// ---------------------------------------------------------------------------
// skip_special_tokens (ignored for WordPiece — BUG)
// ---------------------------------------------------------------------------

/// skip_special_tokens=1 should strip [CLS] and [SEP] from output.
///
/// Bug: `decodeWithOptions()` ignores `skip_special_tokens` for WordPiece
/// models (only BPE honors decode options). This test will fail until fixed.
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
//
// Bug: `encode_special("")` should produce [101, 102] ([CLS] + [SEP]) for
// BERT models, but produces [] (empty). The post_processor's cls_id/sep_id
// are not resolved, or the BertProcessing path is not activated.
// Affects: BAAI/bge, sentence-transformers, google-bert (~10 models).

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
// Affects: google-bert/bert-base-uncased (~19 encode, 19 roundtrip failures).

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
///
/// Bug: WordPiece encode with BertProcessing post_processor should
/// wrap output with [CLS]=1 and [SEP]=2.
#[test]
fn encode_bert_postproc_adds_cls_sep() {
    let ctx = TokenizerTestContext::from_json(BERT_POSTPROC_JSON);
    let opts = talu_sys::EncodeOptions {
        add_bos: 1,
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
// Affects: BAAI/bge, sentence-transformers (~9 encode failures per model).

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
//
// Bug: the CJK detection may incorrectly classify non-CJK codepoints,
// causing "hello" to be split into individual characters ("h e l l o").
// Affects: google-bert/bert-base-uncased (19 encode failures).

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
// decomposition. This prevents quadratic blowup on adversarial inputs and
// matches HuggingFace behavior.
//
// Affects: BAAI/bge-large-en-v1.5, sentence-transformers/all-MiniLM-L6-v2,
// sentence-transformers/all-mpnet-base-v2 (long "aaa..." inputs).

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
//
// Affects: BAAI/bge-large-en-v1.5, sentence-transformers/all-MiniLM-L6-v2
// (encode "line1\nline2\nline3" produces wrong token IDs).

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
// hardcoded token names like [CLS]/[SEP]. RoBERTa-family models use
// <s> and </s> as CLS/SEP, and the post-processor must honor those.
//
// Affects: sentence-transformers/all-mpnet-base-v2 (encode_special uses
// [CLS]=105 and [SEP]=106 instead of the configured <s>=0 and </s>=2).

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
// HuggingFace's `clean_up_tokenization_spaces` removes space before a small
// set of punctuation: . ? ! , ' (and contractions like n't, 'm, etc.).
// Space before dash (-) must NOT be removed. Markdown-style lists like
// "# title - item 1 - item 2" must decode with spaces around dashes.
//
// Affects: BAAI/bge-large-en-v1.5, sentence-transformers/all-MiniLM-L6-v2,
// sentence-transformers/all-mpnet-base-v2 (roundtrip "# Title\n\n- item 1"
// decodes as "# title- item 1" instead of "# title - item 1").

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
// HuggingFace's BertNormalizer._is_chinese_char checks 8 Unicode ranges:
//   0x4E00-0x9FFF   (CJK Unified Ideographs)
//   0x3400-0x4DBF   (CJK Extension A)
//   0x20000-0x2A6DF (CJK Extension B)
//   0xF900-0xFAFF   (CJK Compatibility Ideographs)
//   0x2F800-0x2FA1F (CJK Compatibility Supplement)
//   ...and more
//
// Bug: our handle_chinese_chars only checks 0x4E00-0x9FFF. Characters in
// CJK Extension A (U+3400-U+4DBF) or Extension B (U+20000-U+2A6DF) are
// not detected, so they don't get spaces added around them. When adjacent
// to other text, they merge into one long word that WordPiece can't match.
//
// Affects: google-bert/bert-base-uncased, BAAI/bge, sentence-transformers
// (CJK inputs produce wrong tokenization — characters not isolated).

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
// HuggingFace's WordPiece encoder produces [UNK] for any word that can't be
// decomposed into known subwords. Emoji characters like 🌍 (U+1F30D, 4-byte
// UTF-8: F0 9F 8C 8D) are not in typical BERT vocabs and should produce a
// single [UNK] token.
//
// Bug: emoji characters are silently dropped during encoding — the token
// count is one less than expected. The emoji produces no output instead of
// [UNK].
//
// Affects: BAAI/bge-large-en-v1.5, sentence-transformers/all-MiniLM-L6-v2,
// sentence-transformers/all-mpnet-base-v2 (encode "Hello 你好 مرحبا 🌍"
// missing final [UNK] for 🌍).

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
