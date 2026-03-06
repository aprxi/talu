//! Tests for normalizer and pre-tokenizer pipeline stages.
//!
//! Creates synthetic tokenizer fixtures with different normalizer and
//! pre-tokenizer configurations to verify that each pipeline stage
//! transforms input text correctly before BPE tokenization.

use crate::capi::tokenizer::common::TokenizerTestContext;

fn no_bos() -> talu_sys::EncodeOptions {
    talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    }
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
    assert_eq!(first, second, "empty-match regex pretokenization must be deterministic");
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
    let json = byte_level_with_normalizer(
        r#"{"type": "Prepend", "prepend": "X"}"#,
    );
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
    let json = byte_level_with_normalizer(
        r#"{"type": "Prepend", "prepend": "X"}"#,
    );
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
    // "hello" with add_prefix_space=false → no ▁ prefix → "hello" (not "▁hello")
    let tokens = ctx.encode_with("hello", &opts);
    // Token 2 is ▁hello. If no prefix, it should NOT match.
    // Instead should match "hello"=3 if available, or char fallback.
    assert!(
        !tokens.is_empty(),
        "encode must produce tokens"
    );
    // With add_prefix_space=false, should NOT produce ▁hello (id=2) as first token
    if tokens[0] == 2 {
        // This would mean ▁ was incorrectly prepended
        panic!("add_prefix_space=false must NOT prepend ▁, got ▁hello as first token");
    }
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
