//! Tests for normalizer and pre-tokenizer pipeline stages.
//!
//! Creates synthetic tokenizer fixtures with different normalizer and
//! pre-tokenizer configurations to verify that each pipeline stage
//! transforms input text correctly before BPE tokenization.

use crate::capi::tokenizer::common::{
    build_byte_level_tokenizer_json, byte_token_id, encode_raw, TokenizerTestContext,
};
use std::sync::mpsc;
use std::time::Duration;

fn no_bos() -> talu_sys::EncodeOptions {
    talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    }
}

fn encode_with_deadlock_guard(json: &str, text: &str, timeout_ms: u64) -> Vec<u32> {
    let (tx, rx) = mpsc::channel();
    let json_owned = json.to_owned();
    let text_owned = text.to_owned();

    std::thread::spawn(move || {
        let ctx = TokenizerTestContext::from_json(&json_owned);
        let tokens = ctx.encode_with(&text_owned, &no_bos());
        tx.send(tokens)
            .expect("zero-width regex worker must send encode result");
    });

    rx.recv_timeout(Duration::from_millis(timeout_ms))
        .expect("encode likely hung (zero-width regex loop did not make progress)")
}

// ---------------------------------------------------------------------------
// Fixture builders
// ---------------------------------------------------------------------------

/// Build a byte-level fixture with a custom normalizer JSON fragment.
///
/// Uses the full 256 byte-level vocab (IDs 4–259) so every byte produces
/// a real token. The normalizer transforms text before tokenization.
fn byte_level_with_normalizer(normalizer_json: &str) -> String {
    let table = gpt2_byte_to_unicode();
    let vocab = build_byte_level_vocab(&table);
    format!(
        r#"{{
  "version": "1.0",
  "model": {{
    "type": "BPE",
    "vocab": {{
      {vocab}
    }},
    "merges": []
  }},
  "added_tokens": [
    {{"id": 0, "content": "<pad>", "special": true}},
    {{"id": 1, "content": "<s>", "special": true}},
    {{"id": 2, "content": "</s>", "special": true}},
    {{"id": 3, "content": "<unk>", "special": true}}
  ],
  "normalizer": {normalizer_json},
  "pre_tokenizer": {{"type": "ByteLevel", "add_prefix_space": false}},
  "post_processor": null,
  "decoder": {{"type": "ByteLevel"}}
}}"#
    )
}

/// Build a fixture with a custom pre-tokenizer and no normalizer.
///
/// Uses a simple ASCII vocab (IDs 4–98, space through tilde) without
/// ByteLevel remapping. Decoder is null since we're not using ByteLevel.
fn ascii_with_pretokenizer(pre_tokenizer_json: &str) -> String {
    format!(
        r##"{{
  "version": "1.0",
  "model": {{
    "type": "BPE",
    "vocab": {{
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      " ": 4, "!": 5, "\"": 6, "#": 7, "$": 8, "%": 9,
      "&": 10, "'": 11, "(": 12, ")": 13, "*": 14, "+": 15,
      ",": 16, "-": 17, ".": 18, "/": 19,
      "0": 20, "1": 21, "2": 22, "3": 23, "4": 24,
      "5": 25, "6": 26, "7": 27, "8": 28, "9": 29,
      ":": 30, ";": 31, "<": 32, "=": 33, ">": 34, "?": 35, "@": 36,
      "A": 37, "B": 38, "C": 39, "D": 40, "E": 41, "F": 42,
      "G": 43, "H": 44, "I": 45, "J": 46, "K": 47, "L": 48,
      "M": 49, "N": 50, "O": 51, "P": 52, "Q": 53, "R": 54,
      "S": 55, "T": 56, "U": 57, "V": 58, "W": 59, "X": 60,
      "Y": 61, "Z": 62, "[": 63, "\\": 64, "]": 65, "^": 66,
      "_": 67, "`": 68,
      "a": 69, "b": 70, "c": 71, "d": 72, "e": 73, "f": 74,
      "g": 75, "h": 76, "i": 77, "j": 78, "k": 79, "l": 80,
      "m": 81, "n": 82, "o": 83, "p": 84, "q": 85, "r": 86,
      "s": 87, "t": 88, "u": 89, "v": 90, "w": 91, "x": 92,
      "y": 93, "z": 94, "{{": 95, "|": 96, "}}": 97, "~": 98
    }},
    "merges": []
  }},
  "added_tokens": [
    {{"id": 0, "content": "<pad>", "special": true}},
    {{"id": 1, "content": "<s>", "special": true}},
    {{"id": 2, "content": "</s>", "special": true}},
    {{"id": 3, "content": "<unk>", "special": true}}
  ],
  "normalizer": null,
  "pre_tokenizer": {pre_tokenizer_json},
  "post_processor": null,
  "decoder": null
}}"##
    )
}

// Reuse the GPT-2 byte-to-unicode mapping from the byte-level fixture.
fn gpt2_byte_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut shift = 256u32;
    for b in 0u16..=255 {
        let is_direct =
            (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        if is_direct {
            table[b as usize] = char::from_u32(b as u32).unwrap();
        } else {
            table[b as usize] = char::from_u32(shift).unwrap();
            shift += 1;
        }
    }
    table
}

fn build_byte_level_vocab(table: &[char; 256]) -> String {
    let mut entries = Vec::with_capacity(260);
    entries.push(r#""<pad>": 0"#.to_string());
    entries.push(r#""<s>": 1"#.to_string());
    entries.push(r#""</s>": 2"#.to_string());
    entries.push(r#""<unk>": 3"#.to_string());
    for b in 0u16..=255 {
        let ch = table[b as usize];
        let id = b + 4;
        let key = match ch {
            '"' => r#"\""#.to_string(),
            '\\' => r#"\\"#.to_string(),
            c if c.is_ascii_graphic() || c == ' ' => c.to_string(),
            c => format!("\\u{:04X}", c as u32),
        };
        entries.push(format!("\"{key}\": {id}"));
    }
    entries.join(",\n      ")
}

// ===========================================================================
// Lowercase normalizer
// ===========================================================================

/// Lowercase normalizer: "HELLO" produces same byte tokens as "hello".
#[test]
fn normalizer_lowercase_basic() {
    let json = byte_level_with_normalizer(r#"{"lowercase": true}"#);
    let ctx = TokenizerTestContext::from_json(&json);
    let opts = no_bos();

    let upper = ctx.encode_with("HELLO", &opts);
    let lower = ctx.encode_with("hello", &opts);
    assert_eq!(upper, lower, "uppercase should normalize to lowercase");
}

/// Lowercase normalizer: mixed case normalizes to lowercase bytes.
#[test]
fn normalizer_lowercase_mixed_case() {
    let json = byte_level_with_normalizer(r#"{"lowercase": true}"#);
    let ctx = TokenizerTestContext::from_json(&json);
    let opts = no_bos();

    let mixed = ctx.encode_with("HeLLo WoRLd", &opts);
    let lower = ctx.encode_with("hello world", &opts);
    assert_eq!(mixed, lower);
}

/// Lowercase normalizer: already-lowercase text is unchanged.
#[test]
fn normalizer_lowercase_noop_on_lowercase() {
    let json = byte_level_with_normalizer(r#"{"lowercase": true}"#);
    let ctx = TokenizerTestContext::from_json(&json);
    let opts = no_bos();

    // Without normalizer
    let ctx_base = TokenizerTestContext::with_byte_level();
    let normalized = ctx.encode_with("hello", &opts);
    let raw = ctx_base.encode_with("hello", &opts);
    assert_eq!(normalized, raw, "lowercase text should be unaffected");
}

/// Lowercase normalizer: roundtrip decode produces lowercase.
#[test]
fn normalizer_lowercase_decode_roundtrip() {
    let json = byte_level_with_normalizer(r#"{"lowercase": true}"#);
    let ctx = TokenizerTestContext::from_json(&json);
    let opts = no_bos();

    let tokens = ctx.encode_with("HELLO", &opts);
    let decoded = ctx.decode(&tokens);
    assert_eq!(
        decoded, "hello",
        "decode after lowercase should be lowercase"
    );
}

// ===========================================================================
// NFC normalizer
// ===========================================================================

/// NFC normalizer: decomposed é (e + combining accent, 3 bytes) normalizes
/// to composed é (2 bytes), producing fewer byte tokens.
#[test]
fn normalizer_nfc_composes_accent() {
    let json_nfc = byte_level_with_normalizer(r#"{"type": "NFC"}"#);
    let ctx_nfc = TokenizerTestContext::from_json(&json_nfc);
    let ctx_raw = TokenizerTestContext::with_byte_level();
    let opts = no_bos();

    // Decomposed: "e" + U+0301 (combining acute) = 3 bytes (65, CC, 81)
    let decomposed = "e\u{0301}";
    assert_eq!(decomposed.len(), 3);

    // Composed: "é" = 2 bytes (C3, A9)
    let composed = "é";
    assert_eq!(composed.len(), 2);

    // Without NFC: decomposed form produces 3 tokens (one per byte).
    let raw_tokens = ctx_raw.encode_with(decomposed, &opts);
    assert_eq!(raw_tokens.len(), 3, "raw decomposed should be 3 tokens");

    // With NFC: decomposed form normalizes to composed → 2 tokens.
    let nfc_tokens = ctx_nfc.encode_with(decomposed, &opts);
    assert_eq!(nfc_tokens.len(), 2, "NFC should compose to 2 tokens");

    // The NFC result should match direct encoding of composed form.
    let composed_tokens = ctx_raw.encode_with(composed, &opts);
    assert_eq!(nfc_tokens, composed_tokens, "NFC(decomposed) == composed");
}

/// NFC normalizer: already-composed text is unchanged.
#[test]
fn normalizer_nfc_noop_on_composed() {
    let json = byte_level_with_normalizer(r#"{"type": "NFC"}"#);
    let ctx_nfc = TokenizerTestContext::from_json(&json);
    let ctx_raw = TokenizerTestContext::with_byte_level();
    let opts = no_bos();

    let text = "café";
    let nfc_tokens = ctx_nfc.encode_with(text, &opts);
    let raw_tokens = ctx_raw.encode_with(text, &opts);
    assert_eq!(
        nfc_tokens, raw_tokens,
        "already-composed should be unchanged"
    );
}

// ===========================================================================
// StripAccents normalizer
// ===========================================================================

/// StripAccents normalizer: "café" → "cafe" (accent removed).
///
/// The composed é (C3 A9) is stripped to plain e (65) by the normalizer,
/// reducing byte count and changing token IDs.
#[test]
fn normalizer_strip_accents() {
    let json = byte_level_with_normalizer(r#"{"strip_accents": true}"#);
    let ctx = TokenizerTestContext::from_json(&json);
    let ctx_raw = TokenizerTestContext::with_byte_level();
    let opts = no_bos();

    let stripped = ctx.encode_with("café", &opts);
    let plain = ctx_raw.encode_with("cafe", &opts);
    assert_eq!(stripped, plain, "strip_accents should remove accent from é");
    assert_eq!(stripped.len(), 4, "café stripped → 4 byte tokens (c,a,f,e)");
}

/// StripAccents: plain ASCII is unaffected.
#[test]
fn normalizer_strip_accents_noop_on_ascii() {
    let json = byte_level_with_normalizer(r#"{"strip_accents": true}"#);
    let ctx = TokenizerTestContext::from_json(&json);
    let ctx_raw = TokenizerTestContext::with_byte_level();
    let opts = no_bos();

    let stripped = ctx.encode_with("hello", &opts);
    let raw = ctx_raw.encode_with("hello", &opts);
    assert_eq!(stripped, raw, "ASCII should be unaffected by strip_accents");
}

// ===========================================================================
// Sequence normalizer (chaining)
// ===========================================================================

/// Sequence normalizer: Lowercase then NFC applied in order.
#[test]
fn normalizer_sequence_lowercase_then_nfc() {
    let json = byte_level_with_normalizer(
        r#"{"type": "Sequence", "normalizers": [{"lowercase": true}, {"type": "NFC"}]}"#,
    );
    let ctx = TokenizerTestContext::from_json(&json);
    let ctx_raw = TokenizerTestContext::with_byte_level();
    let opts = no_bos();

    // "CAFÉ" with decomposed É → lowercase + NFC → "café"
    let input = "CAFE\u{0301}"; // CAFÉ with decomposed accent
    let tokens = ctx.encode_with(input, &opts);
    let expected_tokens = ctx_raw.encode_with("café", &opts);
    assert_eq!(
        tokens, expected_tokens,
        "Sequence(Lowercase,NFC) should produce 'café'"
    );
}

// ===========================================================================
// Whitespace pre-tokenizer
// ===========================================================================

/// Whitespace pre-tokenizer: "hello world" splits into two words.
///
/// Without ByteLevel remapping, space stays as space (ID 4 in ASCII vocab).
/// Each word is tokenized independently by BPE.
#[test]
fn pretokenizer_whitespace_splits_words() {
    let json = ascii_with_pretokenizer(r#"{"type": "Whitespace"}"#);
    let ctx = TokenizerTestContext::from_json(&json);
    let opts = no_bos();

    let tokens = ctx.encode_with("hello world", &opts);
    assert_eq!(
        tokens,
        vec![76, 73, 80, 80, 83, 91, 83, 86, 80, 72],
        "Whitespace pre-tokenizer should drop spaces and emit only word chars"
    );
}

/// Whitespace pre-tokenizer: single word has no split.
#[test]
fn pretokenizer_whitespace_single_word() {
    let json = ascii_with_pretokenizer(r#"{"type": "Whitespace"}"#);
    let ctx = TokenizerTestContext::from_json(&json);
    let opts = no_bos();

    let tokens = ctx.encode_with("hello", &opts);
    assert_eq!(tokens.len(), 5, "5 chars should produce 5 tokens");
}

/// Whitespace pre-tokenizer should drop tab/newline separators as boundaries.
#[test]
fn pretokenizer_whitespace_drops_tab_and_newline() {
    let json = ascii_with_pretokenizer(r#"{"type": "Whitespace"}"#);
    let ctx = TokenizerTestContext::from_json(&json);
    let tokens = ctx.encode_with("a\tb\nc", &no_bos());
    assert_eq!(
        tokens,
        vec![69, 70, 71],
        "Whitespace pre-tokenizer should emit only word chars across tab/newline boundaries"
    );
}

/// Whitespace pre-tokenizer should drop leading/trailing spaces around words.
#[test]
fn pretokenizer_whitespace_drops_leading_and_trailing_spaces() {
    let json = ascii_with_pretokenizer(r#"{"type": "Whitespace"}"#);
    let ctx = TokenizerTestContext::from_json(&json);
    let tokens = ctx.encode_with("  a  ", &no_bos());
    assert_eq!(
        tokens,
        vec![69],
        "Whitespace pre-tokenizer should not emit separator space tokens"
    );
}

// ===========================================================================
// Punctuation pre-tokenizer
// ===========================================================================

/// Punctuation pre-tokenizer: "hello,world" splits around the comma.
#[test]
fn pretokenizer_punctuation_splits() {
    let json = ascii_with_pretokenizer(r#"{"type": "Punctuation"}"#);
    let ctx = TokenizerTestContext::from_json(&json);
    let opts = no_bos();

    let tokens = ctx.encode_with("hello,world", &opts);
    assert_eq!(
        tokens,
        vec![76, 73, 80, 80, 83, 16, 91, 83, 86, 80, 72],
        "Punctuation pre-tokenizer should isolate comma as its own token"
    );
}

// ===========================================================================
// BertPreTokenizer (whitespace + punctuation)
// ===========================================================================

/// BertPreTokenizer: splits on both whitespace and punctuation.
#[test]
fn pretokenizer_bert_splits_both() {
    let json = ascii_with_pretokenizer(r#"{"type": "BertPreTokenizer"}"#);
    let ctx = TokenizerTestContext::from_json(&json);
    let opts = no_bos();

    let tokens = ctx.encode_with("hello, world!", &opts);
    assert_eq!(
        tokens,
        vec![76, 73, 80, 80, 83, 16, 91, 83, 86, 80, 72, 5],
        "BertPreTokenizer should drop whitespace and isolate punctuation"
    );
}

// ===========================================================================
// Normalizer + pre-tokenizer combination
// ===========================================================================

/// Lowercase normalizer + Whitespace pre-tokenizer: "HELLO WORLD" → lowercase tokens.
#[test]
fn normalizer_plus_pretokenizer() {
    // Combine lowercase normalizer with whitespace pre-tokenizer.
    let table = gpt2_byte_to_unicode();
    let vocab = build_byte_level_vocab(&table);
    let json = format!(
        r#"{{
  "version": "1.0",
  "model": {{
    "type": "BPE",
    "vocab": {{
      {vocab}
    }},
    "merges": []
  }},
  "added_tokens": [
    {{"id": 0, "content": "<pad>", "special": true}},
    {{"id": 1, "content": "<s>", "special": true}},
    {{"id": 2, "content": "</s>", "special": true}},
    {{"id": 3, "content": "<unk>", "special": true}}
  ],
  "normalizer": {{"lowercase": true}},
  "pre_tokenizer": {{"type": "ByteLevel", "add_prefix_space": false}},
  "post_processor": null,
  "decoder": {{"type": "ByteLevel"}}
}}"#
    );
    let ctx = TokenizerTestContext::from_json(&json);
    let opts = no_bos();

    let upper_tokens = ctx.encode_with("HELLO WORLD", &opts);
    let lower_tokens = ctx.encode_with("hello world", &opts);
    assert_eq!(
        upper_tokens, lower_tokens,
        "Lowercase should normalize before ByteLevel"
    );

    // Verify decode produces lowercase.
    let decoded = ctx.decode(&upper_tokens);
    assert_eq!(decoded, "hello world");
}

// ===========================================================================
// Type-based normalizer dispatch (check for fast-path missing type branches)
// ===========================================================================

/// {"type": "Lowercase"} normalizes uppercase to lowercase.
///
/// The HuggingFace format uses only the "type" field with no explicit boolean
/// flags. This tests the fast-path type dispatch in applyNormalizerFromJson.
#[test]
fn normalizer_type_lowercase() {
    let json = byte_level_with_normalizer(r#"{"type": "Lowercase"}"#);
    let ctx = TokenizerTestContext::from_json(&json);
    let opts = no_bos();

    let upper = ctx.encode_with("HELLO", &opts);
    let lower = ctx.encode_with("hello", &opts);
    assert_eq!(
        upper, lower,
        "type Lowercase should normalize uppercase to lowercase"
    );
}

/// {"type": "Lowercase"} roundtrip: encode("HELLO") decodes to "hello".
#[test]
fn normalizer_type_lowercase_roundtrip() {
    let json = byte_level_with_normalizer(r#"{"type": "Lowercase"}"#);
    let ctx = TokenizerTestContext::from_json(&json);
    let opts = no_bos();

    let tokens = ctx.encode_with("HELLO", &opts);
    let decoded = ctx.decode(&tokens);
    assert_eq!(decoded, "hello");
}

/// {"type": "StripAccents"} removes accents from composed characters.
#[test]
fn normalizer_type_strip_accents() {
    let json = byte_level_with_normalizer(r#"{"type": "StripAccents"}"#);
    let ctx = TokenizerTestContext::from_json(&json);
    let ctx_raw = TokenizerTestContext::with_byte_level();
    let opts = no_bos();

    // Composed é (C3 A9) stripped to e (65)
    let stripped = ctx.encode_with("café", &opts);
    let plain = ctx_raw.encode_with("cafe", &opts);
    assert_eq!(
        stripped, plain,
        "type StripAccents should remove accent from é"
    );
}

/// {"type": "BertNormalizer"} lowercases AND strips accents.
#[test]
fn normalizer_type_bert_normalizer() {
    let json = byte_level_with_normalizer(r#"{"type": "BertNormalizer"}"#);
    let ctx = TokenizerTestContext::from_json(&json);
    let ctx_raw = TokenizerTestContext::with_byte_level();
    let opts = no_bos();

    // Should lowercase "CAFÉ" and strip the accent on É
    let tokens = ctx.encode_with("CAFÉ", &opts);
    let expected = ctx_raw.encode_with("cafe", &opts);
    assert_eq!(
        tokens, expected,
        "BertNormalizer should lowercase + strip accents"
    );
}

/// {"type": "BertNormalizer"} adds spaces around CJK characters.
#[test]
fn normalizer_type_bert_cjk_spacing() {
    let json = byte_level_with_normalizer(r#"{"type": "BertNormalizer"}"#);
    let ctx = TokenizerTestContext::from_json(&json);
    let ctx_raw = TokenizerTestContext::with_byte_level();
    let opts = no_bos();

    // "日" is 3 UTF-8 bytes → 3 tokens without normalizer
    let raw_tokens = ctx_raw.encode_with("日", &opts);
    assert_eq!(raw_tokens.len(), 3);

    // With BertNormalizer, CJK gets surrounding spaces: " 日 " = space + 3 bytes + space
    let bert_tokens = ctx.encode_with("日", &opts);
    assert_eq!(
        bert_tokens.len(),
        raw_tokens.len() + 2,
        "BertNormalizer should add exactly one ASCII space on each side of a CJK char"
    );
}

/// Split pre-tokenizer regex that can match empty strings must not loop forever.
#[test]
fn pretokenizer_split_regex_empty_match_completes() {
    let json = ascii_with_pretokenizer(
        r#"{"type":"Split","pattern":{"Regex":"^|\\b"},"behavior":"Removed","invert":false}"#,
    );
    let ctx = TokenizerTestContext::from_json(&json);
    let opts = no_bos();
    let text = "hello world";
    let first = ctx.encode_with(text, &opts);
    let second = ctx.encode_with(text, &opts);
    assert_eq!(
        first, second,
        "empty-match regex pretokenization must be deterministic"
    );
}

/// Split with a zero-width start-anchor regex must terminate and preserve
/// tokenization parity with a null pre-tokenizer (no chars consumed).
#[test]
fn pretokenizer_split_regex_caret_empty_match_matches_null_pretokenizer() {
    let split_json = ascii_with_pretokenizer(
        r#"{"type":"Split","pattern":{"Regex":"^"},"behavior":"Removed","invert":false}"#,
    );
    let baseline_json = ascii_with_pretokenizer("null");

    let text = "hello world";

    // Deadlock guard: zero-width regex engines can loop if the cursor is not
    // advanced on empty matches.
    let split_tokens = encode_with_deadlock_guard(&split_json, text, 500);
    let baseline_tokens = encode_with_deadlock_guard(&baseline_json, text, 500);

    assert_eq!(
        split_tokens, baseline_tokens,
        "zero-width '^' split must not alter tokenization vs null pre-tokenizer"
    );
}

/// Split with a zero-width word-boundary regex must terminate and preserve
/// tokenization parity with a null pre-tokenizer (boundary matches consume
/// zero bytes and must not cause looping or token loss).
#[test]
fn pretokenizer_split_regex_word_boundary_empty_match_matches_null_pretokenizer() {
    let split_json = ascii_with_pretokenizer(
        r#"{"type":"Split","pattern":{"Regex":"\\b"},"behavior":"Removed","invert":false}"#,
    );
    let baseline_json = ascii_with_pretokenizer("null");

    let text = "hello world";

    // Deadlock guard: zero-width regex engines can loop if the cursor is not
    // advanced on empty matches.
    let split_tokens = encode_with_deadlock_guard(&split_json, text, 500);
    let baseline_tokens = encode_with_deadlock_guard(&baseline_json, text, 500);

    assert_eq!(
        split_tokens, baseline_tokens,
        "zero-width '\\\\b' split must not alter tokenization vs null pre-tokenizer"
    );
}

/// Split behavior Removed should drop regex matches and keep non-matching gaps.
#[test]
fn pretokenizer_split_removed_drops_matches() {
    let json = ascii_with_pretokenizer(
        r#"{"type":"Split","pattern":{"Regex":"\\s+"},"behavior":"Removed","invert":false}"#,
    );
    let ctx = TokenizerTestContext::from_json(&json);
    let tokens = ctx.encode_with("a b", &no_bos());
    assert_eq!(
        tokens,
        vec![69, 70],
        "Split(Removed) should remove whitespace matches entirely"
    );
}

/// Split(Removed) should remove punctuation matches and keep only non-matching text.
#[test]
fn pretokenizer_split_removed_drops_punctuation_matches() {
    let json = ascii_with_pretokenizer(
        r#"{"type":"Split","pattern":{"Regex":"[,!.]+"},"behavior":"Removed","invert":false}"#,
    );
    let ctx = TokenizerTestContext::from_json(&json);
    let tokens = ctx.encode_with("a,b!c.", &no_bos());
    assert_eq!(
        tokens,
        vec![69, 70, 71],
        "Split(Removed) should drop punctuation matches"
    );
}

/// Split behavior Isolated should keep both matches and non-matching gaps.
#[test]
fn pretokenizer_split_isolated_keeps_matches_and_gaps() {
    let json = ascii_with_pretokenizer(
        r#"{"type":"Split","pattern":{"Regex":"\\s+"},"behavior":"Isolated","invert":false}"#,
    );
    let ctx = TokenizerTestContext::from_json(&json);
    let tokens = ctx.encode_with("a b", &no_bos());
    assert_eq!(
        tokens,
        vec![69, 4, 70],
        "Split(Isolated) should preserve space token as isolated match"
    );
}

/// Split invert=true should emit only regex matches.
#[test]
fn pretokenizer_split_invert_true_keeps_only_matches() {
    let json = ascii_with_pretokenizer(
        r#"{"type":"Split","pattern":{"Regex":"\\d+"},"behavior":"Removed","invert":true}"#,
    );
    let ctx = TokenizerTestContext::from_json(&json);
    let tokens = ctx.encode_with("a12b34", &no_bos());
    assert_eq!(
        tokens,
        vec![21, 22, 23, 24],
        "Split(invert=true) should keep only digit matches"
    );
}

/// Split(invert=true) with punctuation regex should keep only punctuation matches.
#[test]
fn pretokenizer_split_invert_true_keeps_only_punctuation() {
    let json = ascii_with_pretokenizer(
        r#"{"type":"Split","pattern":{"Regex":"[,!.]+"},"behavior":"Removed","invert":true}"#,
    );
    let ctx = TokenizerTestContext::from_json(&json);
    let tokens = ctx.encode_with("a,b!c.", &no_bos());
    assert_eq!(
        tokens,
        vec![16, 5, 18],
        "Split(invert=true) should keep only punctuation matches"
    );
}

/// Pathological regex patterns must complete deterministically.
#[test]
fn pretokenizer_split_pathological_regex_deterministic() {
    let json = ascii_with_pretokenizer(
        r#"{"type":"Split","pattern":{"Regex":"(a+)+b"},"behavior":"Isolated","invert":false}"#,
    );
    let ctx = TokenizerTestContext::from_json(&json);
    let text = format!("{}b", "a".repeat(160));
    let first = ctx.encode_with(&text, &no_bos());
    let second = ctx.encode_with(&text, &no_bos());
    assert_eq!(first, second, "pathological regex must be deterministic");
}

/// A larger non-matching catastrophic-backtracking shape must also complete
/// and remain deterministic. This guards against regex-limit regressions.
#[test]
fn pretokenizer_split_pathological_regex_large_non_match_deterministic() {
    let json = ascii_with_pretokenizer(
        r#"{"type":"Split","pattern":{"Regex":"(a+)+b"},"behavior":"Isolated","invert":false}"#,
    );
    let ctx = TokenizerTestContext::from_json(&json);
    let text = format!("{}c", "a".repeat(4096));
    let first = ctx.encode_with(&text, &no_bos());
    let second = ctx.encode_with(&text, &no_bos());
    assert_eq!(
        first, second,
        "large non-matching pathological regex input must be deterministic"
    );
    assert!(
        !first.is_empty(),
        "large non-matching pathological regex input must still produce output tokens"
    );
}

/// A byte-unit regex (`\\C`) can intentionally bisect a valid multi-byte UTF-8
/// emoji when used in Split pretokenization. The pipeline must not crash and
/// must preserve roundtrip decode correctness.
#[test]
fn pretokenizer_split_byte_unit_regex_bisected_emoji_is_safe() {
    let json = build_byte_level_tokenizer_json().replace(
        "\"pre_tokenizer\": {\"type\": \"ByteLevel\", \"add_prefix_space\": false},",
        r#""pre_tokenizer": {"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"\\C\\C"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false}]}, "#,
    );
    let ctx = TokenizerTestContext::from_json(&json);
    let input = "A😊B";

    let result = unsafe { encode_raw(ctx.handle(), input.as_bytes(), &no_bos()) };
    assert!(result.error_msg.is_null(), "encode must succeed on bisected emoji");
    assert_eq!(
        result.num_tokens,
        input.len(),
        "byte-level path must still emit one token per input byte"
    );

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) }.to_vec();
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    for (idx, off) in offsets.iter().enumerate() {
        let start = off.start as usize;
        let end = off.end as usize;
        assert!(
            start <= end && end <= input.len(),
            "offset[{idx}] out of bounds for bisected emoji path: ({start},{end})"
        );
    }
    assert_eq!(
        offsets.iter().map(|o| o.end as usize).max().unwrap_or(0),
        input.len(),
        "offsets must still cover the full input byte range"
    );
    unsafe { talu_sys::talu_encode_result_free(result) };

    let decoded = ctx.decode(&ids);
    assert_eq!(
        decoded, input,
        "decode must preserve text even when Split bisects UTF-8 bytes"
    );
}

#[cfg(target_os = "linux")]
unsafe fn run_pcre2_invalid_utf8_page_boundary_inner() {
    use std::ffi::c_void;
    use std::ptr;

    unsafe extern "C" {
        fn getpagesize() -> i32;
        fn mmap(
            addr: *mut c_void,
            length: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: isize,
        ) -> *mut c_void;
        fn mprotect(addr: *mut c_void, len: usize, prot: i32) -> i32;
        fn munmap(addr: *mut c_void, len: usize) -> i32;
    }

    const PROT_NONE: i32 = 0;
    const PROT_READ: i32 = 0x1;
    const PROT_WRITE: i32 = 0x2;
    const MAP_PRIVATE: i32 = 0x02;
    const MAP_ANONYMOUS: i32 = 0x20;

    let page_size = unsafe { getpagesize() };
    assert!(page_size > 0, "getpagesize must return a positive value");
    let page = page_size as usize;
    let len = page * 2;

    let mapping = unsafe {
        mmap(
            ptr::null_mut(),
            len,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    assert_ne!(
        mapping as isize, -1,
        "mmap for guarded boundary test must succeed"
    );

    let base = mapping.cast::<u8>();
    let guard = unsafe { base.add(page) };
    let protect_rc = unsafe { mprotect(guard.cast(), page, PROT_NONE) };
    assert_eq!(protect_rc, 0, "mprotect(PROT_NONE) for guard page failed");

    let tail = unsafe { base.add(page - 1) };
    unsafe { *tail = 0xF0 };

    let json = ascii_with_pretokenizer(
        r#"{"type":"Split","pattern":{"Regex":"\\p{L}+"},"behavior":"Isolated","invert":false}"#,
    );
    let ctx = TokenizerTestContext::from_json(&json);
    let opts = no_bos();
    let input = unsafe { std::slice::from_raw_parts(tail, 1) };
    let result = unsafe { encode_raw(ctx.handle(), input, &opts) };
    unsafe { talu_sys::talu_encode_result_free(result) };

    let unmap_rc = unsafe { munmap(mapping, len) };
    assert_eq!(unmap_rc, 0, "munmap for guarded boundary mapping failed");
}

/// Invalid UTF-8 fed to PCRE2-backed Split pretokenization must never trigger
/// an out-of-bounds read across an unreadable page boundary.
///
/// This runs in a subprocess so a native crash is reported as a test failure
/// instead of terminating the parent harness.
#[cfg(target_os = "linux")]
#[test]
fn pretokenizer_pcre2_invalid_utf8_page_boundary_overread() {
    const INNER_ENV: &str = "TALU_INNER_PCRE2_PAGE_GUARD";
    if std::env::var_os(INNER_ENV).is_some() {
        unsafe { run_pcre2_invalid_utf8_page_boundary_inner() };
        return;
    }

    let exe = std::env::current_exe().expect("current test executable path must resolve");
    let output = std::process::Command::new(exe)
        .arg("--exact")
        .arg("capi::tokenizer::pipeline::pretokenizer_pcre2_invalid_utf8_page_boundary_overread")
        .arg("--nocapture")
        .env(INNER_ENV, "1")
        .output()
        .expect("subprocess launch for guarded invalid-UTF8 PCRE2 test must succeed");

    assert!(
        output.status.success(),
        "guarded invalid-UTF8 PCRE2 path crashed or failed (status: {:?})\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

/// Pathological backtracking patterns on very large non-matching input must
/// return a typed error when PCRE2 exhausts internal match limits.
#[test]
#[ignore = "expensive regex-engine stress; run manually when validating PCRE2 failure propagation"]
fn pretokenizer_pcre2_match_limit_exhaustion_returns_error() {
    let json = ascii_with_pretokenizer(
        r#"{"type":"Split","pattern":{"Regex":"(a+)+b"},"behavior":"Isolated","invert":false}"#,
    );
    let ctx = TokenizerTestContext::from_json(&json);
    let text = vec![b'a'; 5 * 1024 * 1024];
    let result = unsafe { encode_raw(ctx.handle(), &text, &no_bos()) };
    assert!(
        !result.error_msg.is_null(),
        "PCRE2 match-limit exhaustion must propagate as a hard tokenizer error"
    );
    if result.error_msg.is_null() {
        unsafe { talu_sys::talu_encode_result_free(result) };
    }
}

/// Exponential Sequence(Replace) normalizer bombs must fail safely through the
/// C API (typed error), not hang or crash the process.
#[test]
#[ignore = "expensive OOM-path stress; run manually for malicious-normalizer validation"]
fn normalizer_exponential_expansion_bomb_is_safely_caught() {
    let stage =
        r#"{"type":"Replace","pattern":{"String":"a"},"content":"aaaaa"}"#.to_string();
    let chain = vec![stage; 20].join(",");
    let normalizer = format!(r#"{{"type":"Sequence","normalizers":[{chain}]}}"#);
    let json = byte_level_with_normalizer(&normalizer);
    let ctx = TokenizerTestContext::from_json(&json);
    let result = unsafe { encode_raw(ctx.handle(), b"a", &no_bos()) };
    assert!(
        !result.error_msg.is_null(),
        "exponential normalizer expansion must return a typed error instead of crashing"
    );
    if result.error_msg.is_null() {
        unsafe { talu_sys::talu_encode_result_free(result) };
    }
}

/// Large Replace-normalizer expansions must remain deterministic and bounded,
/// not hang or crash on repeated expansion-heavy input.
#[test]
fn normalizer_replace_large_expansion_deterministic() {
    let json = byte_level_with_normalizer(
        r#"{"type":"Replace","pattern":{"String":"a"},"content":"aaaaa"}"#,
    );
    let ctx = TokenizerTestContext::from_json(&json);
    let text = "a".repeat(20_000);

    let first = ctx.encode_with(&text, &no_bos());
    let second = ctx.encode_with(&text, &no_bos());

    assert_eq!(
        first, second,
        "large Replace expansion must be deterministic across runs"
    );
    assert_eq!(
        first.len(),
        100_000,
        "Replace(a->aaaaa) must expand token count 5x on pure 'a' input"
    );
}

/// Sequence containing {"type": "Lowercase"} works correctly.
#[test]
fn normalizer_sequence_with_type_lowercase() {
    let json = byte_level_with_normalizer(
        r#"{"type": "Sequence", "normalizers": [{"type": "Lowercase"}, {"type": "NFC"}]}"#,
    );
    let ctx = TokenizerTestContext::from_json(&json);
    let ctx_raw = TokenizerTestContext::with_byte_level();
    let opts = no_bos();

    // "CAFÉ" with decomposed É → Lowercase("cafe\u{0301}") → NFC("café")
    let input = "CAFE\u{0301}";
    let tokens = ctx.encode_with(input, &opts);
    let expected = ctx_raw.encode_with("café", &opts);
    assert_eq!(
        tokens, expected,
        "Sequence(type:Lowercase, type:NFC) should produce 'café'"
    );
}

// ===========================================================================
// Unicode normalization flags (check if NFD/NFKC/NFKD is parsed and applied)
// ===========================================================================

/// NFKC normalizes fullwidth Ａ (U+FF21, 3 bytes) to ASCII A (1 byte).
#[test]
fn normalizer_nfkc_fullwidth_to_ascii() {
    let json = byte_level_with_normalizer(r#"{"type": "NFKC"}"#);
    let ctx = TokenizerTestContext::from_json(&json);
    let ctx_raw = TokenizerTestContext::with_byte_level();
    let opts = no_bos();

    // Fullwidth Ａ is 3 UTF-8 bytes (EF BC A1) → 3 tokens raw
    let fullwidth = "\u{FF21}";
    assert_eq!(fullwidth.len(), 3);
    let raw_tokens = ctx_raw.encode_with(fullwidth, &opts);
    assert_eq!(raw_tokens.len(), 3);

    // NFKC normalizes to ASCII A (1 byte) → 1 token
    let nfkc_tokens = ctx.encode_with(fullwidth, &opts);
    let ascii_tokens = ctx_raw.encode_with("A", &opts);
    assert_eq!(
        nfkc_tokens, ascii_tokens,
        "NFKC should normalize fullwidth A to ASCII A"
    );
}

/// NFKC decomposes ﬁ ligature (U+FB01, 3 bytes) to fi (2 bytes).
#[test]
fn normalizer_nfkc_ligature() {
    let json = byte_level_with_normalizer(r#"{"type": "NFKC"}"#);
    let ctx = TokenizerTestContext::from_json(&json);
    let ctx_raw = TokenizerTestContext::with_byte_level();
    let opts = no_bos();

    // ﬁ ligature is 3 UTF-8 bytes (EF AC 81) → 3 tokens raw
    let ligature = "\u{FB01}";
    assert_eq!(ligature.len(), 3);
    let raw_tokens = ctx_raw.encode_with(ligature, &opts);
    assert_eq!(raw_tokens.len(), 3);

    // NFKC decomposes to "fi" (2 bytes) → 2 tokens
    let nfkc_tokens = ctx.encode_with(ligature, &opts);
    let fi_tokens = ctx_raw.encode_with("fi", &opts);
    assert_eq!(
        nfkc_tokens, fi_tokens,
        "NFKC should decompose fi ligature to 'fi'"
    );
}

/// NFD decomposes composed é (2 bytes) to e + combining accent (3 bytes).
#[test]
fn normalizer_nfd_decomposes() {
    let json = byte_level_with_normalizer(r#"{"type": "NFD"}"#);
    let ctx = TokenizerTestContext::from_json(&json);
    let ctx_raw = TokenizerTestContext::with_byte_level();
    let opts = no_bos();

    // Composed é is 2 UTF-8 bytes (C3 A9) → 2 tokens raw
    let composed = "é";
    assert_eq!(composed.len(), 2);
    let raw_tokens = ctx_raw.encode_with(composed, &opts);
    assert_eq!(raw_tokens.len(), 2);

    // NFD decomposes to e (1 byte) + combining acute (2 bytes) = 3 bytes → 3 tokens
    let nfd_tokens = ctx.encode_with(composed, &opts);
    assert_eq!(
        nfd_tokens.len(),
        3,
        "NFD should decompose é to 3 bytes (e + combining accent)"
    );
}

/// NFKD normalizes fullwidth Ａ (U+FF21, 3 bytes) to ASCII A (1 byte).
#[test]
fn normalizer_nfkd_fullwidth() {
    let json = byte_level_with_normalizer(r#"{"type": "NFKD"}"#);
    let ctx = TokenizerTestContext::from_json(&json);
    let ctx_raw = TokenizerTestContext::with_byte_level();
    let opts = no_bos();

    let fullwidth = "\u{FF21}";
    assert_eq!(fullwidth.len(), 3);

    // NFKD decomposes fullwidth to ASCII
    let nfkd_tokens = ctx.encode_with(fullwidth, &opts);
    let ascii_tokens = ctx_raw.encode_with("A", &opts);
    assert_eq!(
        nfkd_tokens, ascii_tokens,
        "NFKD should decompose fullwidth A to ASCII A"
    );
}

// ===========================================================================
// Replace normalizer
// ===========================================================================

/// Replace normalizer with String pattern replaces literal matches.
#[test]
fn normalizer_replace_string_pattern() {
    let json = byte_level_with_normalizer(
        r#"{"type": "Replace", "pattern": {"String": "hello"}, "content": "world"}"#,
    );
    let ctx = TokenizerTestContext::from_json(&json);
    let ctx_raw = TokenizerTestContext::with_byte_level();
    let opts = no_bos();

    let replaced = ctx.encode_with("hello", &opts);
    let expected = ctx_raw.encode_with("world", &opts);
    assert_eq!(
        replaced, expected,
        "Replace normalizer must replace 'hello' → 'world'"
    );
}

/// Replace normalizer: non-matching text is unchanged.
#[test]
fn normalizer_replace_no_match() {
    let json = byte_level_with_normalizer(
        r#"{"type": "Replace", "pattern": {"String": "xyz"}, "content": "abc"}"#,
    );
    let ctx = TokenizerTestContext::from_json(&json);
    let ctx_raw = TokenizerTestContext::with_byte_level();
    let opts = no_bos();

    let result = ctx.encode_with("hello", &opts);
    let expected = ctx_raw.encode_with("hello", &opts);
    assert_eq!(result, expected, "non-matching text must be unchanged");
}

// ===========================================================================
// Prepend normalizer
// ===========================================================================

/// Prepend normalizer adds a prefix string to the input.
#[test]
fn normalizer_prepend() {
    let json = byte_level_with_normalizer(r#"{"type": "Prepend", "prepend": "X"}"#);
    let ctx = TokenizerTestContext::from_json(&json);
    let ctx_raw = TokenizerTestContext::with_byte_level();
    let opts = no_bos();

    let prepended = ctx.encode_with("hello", &opts);
    let expected = ctx_raw.encode_with("Xhello", &opts);
    assert_eq!(
        prepended, expected,
        "Prepend normalizer must add 'X' prefix"
    );
}

/// Prepend normalizer on empty input must still produce the prefix.
///
/// Prepend("X") on "" should produce "X" → tokens for "X".
#[test]
fn normalizer_prepend_empty_input() {
    let json = byte_level_with_normalizer(r#"{"type": "Prepend", "prepend": "X"}"#);
    let ctx = TokenizerTestContext::from_json(&json);
    let ctx_raw = TokenizerTestContext::with_byte_level();
    let opts = no_bos();

    let prepended = ctx.encode_with("", &opts);
    let expected = ctx_raw.encode_with("X", &opts);
    assert_eq!(
        prepended, expected,
        "Prepend on empty input must produce just the prefix, got: {prepended:?}"
    );
}

// ===========================================================================
// BertNormalizer with explicit flags
// ===========================================================================

/// BertNormalizer with clean_text=true replaces tab characters with spaces.
#[test]
fn bert_normalizer_clean_text_replaces_tab() {
    let json = byte_level_with_normalizer(
        r#"{"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": false, "strip_accents": false, "lowercase": false}"#,
    );
    let ctx = TokenizerTestContext::from_json(&json);
    let ctx_raw = TokenizerTestContext::with_byte_level();
    let opts = no_bos();

    // "a\tb" → clean_text → "a b"
    let cleaned = ctx.encode_with("a\tb", &opts);
    let expected = ctx_raw.encode_with("a b", &opts);
    assert_eq!(
        cleaned, expected,
        "BertNormalizer clean_text must replace tab with space"
    );
}

/// BertNormalizer with handle_chinese_chars=false does NOT add CJK spaces.
#[test]
fn bert_normalizer_no_cjk_spacing_when_disabled() {
    let json = byte_level_with_normalizer(
        r#"{"type": "BertNormalizer", "clean_text": false, "handle_chinese_chars": false, "strip_accents": false, "lowercase": false}"#,
    );
    let ctx = TokenizerTestContext::from_json(&json);
    let ctx_raw = TokenizerTestContext::with_byte_level();
    let opts = no_bos();

    // "日" with handle_chinese_chars=false → no added spaces
    let bert_tokens = ctx.encode_with("日", &opts);
    let raw_tokens = ctx_raw.encode_with("日", &opts);
    assert_eq!(
        bert_tokens, raw_tokens,
        "handle_chinese_chars=false must not add spaces around CJK"
    );
}

/// BertNormalizer with strip_accents=false does NOT strip accents.
#[test]
fn bert_normalizer_no_strip_when_disabled() {
    let json = byte_level_with_normalizer(
        r#"{"type": "BertNormalizer", "clean_text": false, "handle_chinese_chars": false, "strip_accents": false, "lowercase": false}"#,
    );
    let ctx = TokenizerTestContext::from_json(&json);
    let ctx_raw = TokenizerTestContext::with_byte_level();
    let opts = no_bos();

    // "café" with strip_accents=false → accent preserved
    let bert_tokens = ctx.encode_with("café", &opts);
    let raw_tokens = ctx_raw.encode_with("café", &opts);
    assert_eq!(
        bert_tokens, raw_tokens,
        "strip_accents=false must preserve accents"
    );
}

/// BertNormalizer with lowercase=false does NOT lowercase.
#[test]
fn bert_normalizer_no_lowercase_when_disabled() {
    let json = byte_level_with_normalizer(
        r#"{"type": "BertNormalizer", "clean_text": false, "handle_chinese_chars": false, "strip_accents": false, "lowercase": false}"#,
    );
    let ctx = TokenizerTestContext::from_json(&json);
    let ctx_raw = TokenizerTestContext::with_byte_level();
    let opts = no_bos();

    // "HELLO" with lowercase=false → not lowercased
    let bert_tokens = ctx.encode_with("HELLO", &opts);
    let raw_tokens = ctx_raw.encode_with("HELLO", &opts);
    assert_eq!(
        bert_tokens, raw_tokens,
        "lowercase=false must preserve case"
    );
}

// ===========================================================================
// Metaspace pretokenizer edge cases
// ===========================================================================

/// Metaspace with add_prefix_space=false does not prepend ▁.
#[test]
fn pretokenizer_metaspace_no_prefix_space() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0,
      "\u2581": 1, "\u2581hello": 2, "hello": 3,
      "h": 4, "e": 5, "l": 6, "o": 7
    },
    "merges": ["h e", "he l", "hel l", "hell o"]
  },
  "added_tokens": [{"id": 0, "content": "<unk>", "special": true}],
  "normalizer": null,
  "pre_tokenizer": {"type": "Metaspace", "replacement": "\u2581", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "Metaspace", "replacement": "\u2581", "add_prefix_space": false}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let opts = no_bos();
    // add_prefix_space=false must not inject metaspace marker token.
    let tokens = ctx.encode_with("hello", &opts);
    assert!(
        !tokens.contains(&2),
        "add_prefix_space=false must not prepend ▁hello token"
    );
    assert_eq!(
        ctx.decode(&tokens),
        "hello",
        "add_prefix_space=false must preserve plain decode text"
    );
}

// ===========================================================================
// Sequence normalizer with three stages
// ===========================================================================

/// Three-stage sequence normalizer: Lowercase + StripAccents + NFC.
#[test]
fn normalizer_sequence_three_stages() {
    let json = byte_level_with_normalizer(
        r#"{"type": "Sequence", "normalizers": [{"type": "Lowercase"}, {"type": "StripAccents"}, {"type": "NFC"}]}"#,
    );
    let ctx = TokenizerTestContext::from_json(&json);
    let ctx_raw = TokenizerTestContext::with_byte_level();
    let opts = no_bos();

    // "CAFÉ" → Lowercase → "café" → StripAccents → "cafe" → NFC → "cafe"
    let tokens = ctx.encode_with("CAFÉ", &opts);
    let expected = ctx_raw.encode_with("cafe", &opts);
    assert_eq!(
        tokens, expected,
        "Sequence(Lowercase, StripAccents, NFC): 'CAFÉ' → 'cafe'"
    );
}

/// Sequence[Split(Removed), ByteLevel] with repetitive input must preserve
/// per-occurrence source offsets for repeated identical tokens.
#[test]
fn pretokenizer_split_removed_then_bytelevel_repetitive_offsets_stay_aligned() {
    let json = build_byte_level_tokenizer_json().replace(
        r#""pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false}"#,
        r#""pre_tokenizer": {"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"\\s*,\\s*"},"behavior":"Removed","invert":false},{"type":"ByteLevel","add_prefix_space":false}]}"#,
    );
    let ctx = TokenizerTestContext::from_json(&json);
    let input = "a , a , a , a";

    let result = unsafe { encode_raw(ctx.handle(), input.as_bytes(), &no_bos()) };
    assert!(result.error_msg.is_null(), "encode failed");
    assert_eq!(
        result.num_tokens, 4,
        "split/removed should keep only four a's"
    );

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    assert_eq!(
        ids,
        &[
            byte_token_id(b'a'),
            byte_token_id(b'a'),
            byte_token_id(b'a'),
            byte_token_id(b'a')
        ]
    );

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 1));
    assert_eq!((offsets[1].start, offsets[1].end), (4, 5));
    assert_eq!((offsets[2].start, offsets[2].end), (8, 9));
    assert_eq!((offsets[3].start, offsets[3].end), (12, 13));

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Split pretokenizer on malformed UTF-8 must stay deterministic and must not
/// crash across repeated calls.
#[test]
fn pretokenizer_split_invalid_utf8_is_stable() {
    let json = ascii_with_pretokenizer(
        r#"{"type":"Split","pattern":{"Regex":"\\w+"},"behavior":"Isolated","invert":false}"#,
    );
    let ctx = TokenizerTestContext::from_json(&json);
    let bytes = [0xFF, b'a', 0xC3, b'b', 0x80];

    let first = unsafe { encode_raw(ctx.handle(), &bytes, &no_bos()) };
    let second = unsafe { encode_raw(ctx.handle(), &bytes, &no_bos()) };

    assert_eq!(
        first.error_msg.is_null(),
        second.error_msg.is_null(),
        "split pretokenizer invalid-UTF8 behavior must be deterministic"
    );

    if first.error_msg.is_null() {
        assert!(second.error_msg.is_null());
        assert_eq!(first.num_tokens, second.num_tokens);

        let first_ids = if first.num_tokens == 0 || first.ids.is_null() {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(first.ids, first.num_tokens) }
        };
        let second_ids = if second.num_tokens == 0 || second.ids.is_null() {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(second.ids, second.num_tokens) }
        };
        assert_eq!(first_ids, second_ids);
    }

    unsafe {
        talu_sys::talu_encode_result_free(first);
        talu_sys::talu_encode_result_free(second);
    }
}

fn run_normalizer_nfc_with_invalid_utf8_fails_gracefully_inner() {
    let json = byte_level_with_normalizer(r#"{"type": "NFC"}"#);
    let ctx = TokenizerTestContext::from_json(&json);
    let bytes = [0xFFu8, 0xC0u8, 0x80u8];

    let first = unsafe { encode_raw(ctx.handle(), &bytes, &no_bos()) };
    let second = unsafe { encode_raw(ctx.handle(), &bytes, &no_bos()) };

    assert_eq!(
        first.error_msg.is_null(),
        second.error_msg.is_null(),
        "NFC invalid-UTF8 behavior must be deterministic across calls"
    );

    if first.error_msg.is_null() {
        assert!(second.error_msg.is_null());
        assert_eq!(
            first.num_tokens, second.num_tokens,
            "successful NFC invalid-UTF8 path must be deterministic in token count"
        );

        let first_ids = if first.ids.is_null() || first.num_tokens == 0 {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(first.ids, first.num_tokens) }
        };
        let second_ids = if second.ids.is_null() || second.num_tokens == 0 {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(second.ids, second.num_tokens) }
        };
        assert_eq!(
            first_ids, second_ids,
            "successful NFC invalid-UTF8 path must be deterministic in IDs"
        );
    }

    unsafe {
        talu_sys::talu_encode_result_free(first);
        talu_sys::talu_encode_result_free(second);
    }
}

/// NFC normalization on invalid UTF-8 must fail gracefully or produce a stable
/// fallback path; it must never crash inside normalization.
#[test]
fn normalizer_nfc_with_invalid_utf8_fails_gracefully() {
    const INNER_ENV: &str = "TALU_INNER_NFC_INVALID_UTF8";
    if std::env::var_os(INNER_ENV).is_some() {
        run_normalizer_nfc_with_invalid_utf8_fails_gracefully_inner();
        return;
    }

    let exe = std::env::current_exe().expect("current test executable path must resolve");
    let output = std::process::Command::new(exe)
        .arg("--exact")
        .arg("capi::tokenizer::pipeline::normalizer_nfc_with_invalid_utf8_fails_gracefully")
        .arg("--nocapture")
        .env(INNER_ENV, "1")
        .output()
        .expect("subprocess launch for NFC invalid-UTF8 test must succeed");

    assert!(
        output.status.success(),
        "NFC invalid-UTF8 subprocess failed (status: {:?})\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}
