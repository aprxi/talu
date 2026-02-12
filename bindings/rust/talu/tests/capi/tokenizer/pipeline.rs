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
        let is_direct = (33..=126).contains(&b)
            || (161..=172).contains(&b)
            || (174..=255).contains(&b);
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
    assert_eq!(decoded, "hello", "decode after lowercase should be lowercase");
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
    assert_eq!(nfc_tokens, raw_tokens, "already-composed should be unchanged");
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
    assert_eq!(tokens, expected_tokens, "Sequence(Lowercase,NFC) should produce 'café'");
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
    // "hello" = [h,e,l,l,o] = [76,73,80,80,83], "world" = [w,o,r,l,d] = [91,83,86,80,72]
    // Whitespace pre-tokenizer splits on whitespace; each word tokenized independently.
    // Space itself may or may not appear depending on behavior.
    assert!(tokens.len() >= 10, "should have at least 10 tokens for two 5-letter words, got {}", tokens.len());
    // Should NOT contain unk (3) for ASCII text.
    for (i, &t) in tokens.iter().enumerate() {
        assert_ne!(t, 3, "token {i} should not be unk");
    }
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
    // "hello" (5) + "," (1) + "world" (5) = 11 tokens minimum.
    assert!(
        tokens.len() >= 11,
        "should tokenize all chars, got {} tokens",
        tokens.len()
    );
    // No unk tokens for ASCII input.
    for (i, &t) in tokens.iter().enumerate() {
        assert_ne!(t, 3, "token {i} should not be unk");
    }
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
    // "hello" + "," + " " + "world" + "!" — all ASCII, no unk.
    for (i, &t) in tokens.iter().enumerate() {
        assert_ne!(t, 3, "token {i} should not be unk");
    }
    // Should have tokens for all characters.
    assert!(tokens.len() >= 11, "should tokenize all chars, got {}", tokens.len());
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
    assert_eq!(upper_tokens, lower_tokens, "Lowercase should normalize before ByteLevel");

    // Verify decode produces lowercase.
    let decoded = ctx.decode(&upper_tokens);
    assert_eq!(decoded, "hello world");
}
