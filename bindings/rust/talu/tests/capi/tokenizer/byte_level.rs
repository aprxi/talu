//! Byte-level BPE tests with full 256-byte vocabulary.
//!
//! Uses a fixture with all 256 GPT-2 byte-level tokens so that every input
//! byte maps to a real token (never `<unk>`). This validates the semantic
//! correctness of byte-level BPE tokenization — not just crash safety.
//!
//! Token ID layout: byte b → ID b + 4.

use crate::capi::tokenizer::common::{build_byte_level_tokenizer_json, byte_token_id, TokenizerTestContext};

fn no_bos() -> talu_sys::EncodeOptions {
    talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    }
}

// ===========================================================================
// Byte-level encoding: non-ASCII produces real tokens, not unk
// ===========================================================================

/// "😊" (4 UTF-8 bytes: F0 9F 98 8A) produces 4 real tokens, none are unk(3).
#[test]
fn emoji_tokenizes_to_bytes_not_unk() {
    let ctx = TokenizerTestContext::with_byte_level();
    let tokens = ctx.encode_with("😊", &no_bos());
    assert_eq!(tokens.len(), 4);
    for (i, &t) in tokens.iter().enumerate() {
        assert_ne!(t, 3, "byte {i} should not be unk");
    }
    // Verify exact byte-to-ID mapping.
    let bytes = "😊".as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        assert_eq!(tokens[i], byte_token_id(b), "byte {i} (0x{b:02X}) ID mismatch");
    }
}

/// "café" (5 bytes: c a f C3 A9) → 5 tokens, é bytes are real tokens not unk.
#[test]
fn cafe_tokenizes_to_byte_tokens() {
    let ctx = TokenizerTestContext::with_byte_level();
    let tokens = ctx.encode_with("café", &no_bos());
    assert_eq!(tokens.len(), 5);
    let bytes = "café".as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        assert_eq!(tokens[i], byte_token_id(b), "byte {i} (0x{b:02X}) ID mismatch");
    }
}

/// "日" (3 UTF-8 bytes: E6 97 A5) → 3 real byte tokens, not unk.
#[test]
fn cjk_tokenizes_to_byte_tokens() {
    let ctx = TokenizerTestContext::with_byte_level();
    let tokens = ctx.encode_with("日", &no_bos());
    assert_eq!(tokens.len(), 3);
    let bytes = "日".as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        assert_eq!(tokens[i], byte_token_id(b), "byte {i} (0x{b:02X}) ID mismatch");
    }
}

/// Every ASCII byte (0x00–0x7F) produces the correct byte token.
/// Non-ASCII bytes (0x80–0xFF) are tested via multi-byte UTF-8 strings
/// in the emoji, CJK, and accented roundtrip tests above.
#[test]
fn all_ascii_bytes_produce_correct_token() {
    let ctx = TokenizerTestContext::with_byte_level();
    let opts = no_bos();

    for b in 0u8..=127 {
        // Each ASCII byte is valid single-byte UTF-8.
        let s = unsafe { std::str::from_utf8_unchecked(std::slice::from_ref(&b)) };
        let tokens = ctx.encode_with(s, &opts);
        assert!(
            !tokens.is_empty(),
            "byte 0x{b:02X}: expected at least 1 token"
        );
        assert_ne!(tokens[0], 3, "byte 0x{b:02X} should not be unk");
        assert_eq!(
            tokens[0],
            byte_token_id(b),
            "byte 0x{b:02X} ID mismatch"
        );
    }
}

// ===========================================================================
// Encode→decode roundtrip for non-ASCII text
// ===========================================================================

/// "Hello 😊!" encode→decode roundtrips exactly.
#[test]
fn encode_decode_roundtrip_emoji() {
    let ctx = TokenizerTestContext::with_byte_level();
    let input = "Hello😊!";
    let tokens = ctx.encode_with(input, &no_bos());
    let decoded = ctx.decode(&tokens);
    assert_eq!(decoded, input);
}

/// "日本語" encode→decode roundtrips exactly.
#[test]
fn encode_decode_roundtrip_cjk() {
    let ctx = TokenizerTestContext::with_byte_level();
    let input = "日本語";
    let tokens = ctx.encode_with(input, &no_bos());
    let decoded = ctx.decode(&tokens);
    assert_eq!(decoded, input);
}

/// Accented text encode→decode roundtrips exactly.
#[test]
fn encode_decode_roundtrip_accented() {
    let ctx = TokenizerTestContext::with_byte_level();
    for input in ["café", "résumé", "naïve"] {
        let tokens = ctx.encode_with(input, &no_bos());
        let decoded = ctx.decode(&tokens);
        assert_eq!(decoded, input, "roundtrip failed for {input:?}");
    }
}

/// Space maps to a byte token (not unk) in the byte-level fixture.
#[test]
fn space_maps_to_byte_token() {
    let ctx = TokenizerTestContext::with_byte_level();
    let tokens = ctx.encode_with(" ", &no_bos());
    assert_eq!(tokens.len(), 1);
    assert_ne!(tokens[0], 3, "space should not be unk");
    assert_eq!(tokens[0], byte_token_id(0x20));
}

/// Token count equals byte count for any input (no merges in fixture).
#[test]
fn byte_count_equals_token_count() {
    let ctx = TokenizerTestContext::with_byte_level();
    let opts = no_bos();

    let cases = [
        "Hello",
        "café",
        "日本語",
        "🎉🚀",
        "Hi 😊 bye",
        "Привет",
        "مرحبا",
    ];
    for text in cases {
        let tokens = ctx.encode_with(text, &opts);
        assert_eq!(
            tokens.len(),
            text.len(),
            "token count should equal byte count for {text:?}"
        );
    }
}

// ===========================================================================
// Offsets with byte-level fixture
// ===========================================================================

/// Offsets for "Hi😊bye" span entire byte range contiguously.
#[test]
fn offsets_cover_full_emoji_span() {
    let ctx = TokenizerTestContext::with_byte_level();
    let text = "Hi😊bye";
    let result = unsafe {
        super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos())
    };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, text.len()); // 2 + 4 + 3 = 9 bytes

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!(offsets[0].start, 0);
    assert_eq!(offsets.last().unwrap().end as usize, text.len());

    for w in offsets.windows(2) {
        assert_eq!(w[0].end, w[1].start, "offsets should be contiguous");
    }

    unsafe { talu_sys::talu_encode_result_free(result) };
}

// ===========================================================================
// Batch and roundtrip
// ===========================================================================

/// Batch encode ["Hello", "日本語", "🎉"] → decode each → exact roundtrip.
#[test]
fn batch_encode_unicode_roundtrip() {
    let ctx = TokenizerTestContext::with_byte_level();
    let texts = ["Hello", "日本語", "🎉"];
    let batch = ctx.encode_batch(&texts, &no_bos());

    assert_eq!(batch.num_sequences, 3);

    // Decode each sequence and verify roundtrip.
    for (i, text) in texts.iter().enumerate() {
        let start = batch.offsets[i];
        let end = batch.offsets[i + 1];
        let seq_tokens = &batch.ids[start..end];
        let decoded = ctx.decode(seq_tokens);
        assert_eq!(decoded, *text, "roundtrip failed for sequence {i}: {text:?}");
    }
}

/// Mixed script text roundtrips: "Testing 123 日本語".
#[test]
fn mixed_script_roundtrip() {
    let ctx = TokenizerTestContext::with_byte_level();
    let input = "Testing123日本語";
    let tokens = ctx.encode_with(input, &no_bos());
    let decoded = ctx.decode(&tokens);
    assert_eq!(decoded, input);
}

/// ZWJ emoji sequence "👨‍👩‍👧‍👦" (25 bytes) roundtrips.
#[test]
fn zwj_emoji_roundtrip() {
    let ctx = TokenizerTestContext::with_byte_level();
    let input = "👨\u{200D}👩\u{200D}👧\u{200D}👦";
    assert_eq!(input.len(), 25);
    let tokens = ctx.encode_with(input, &no_bos());
    assert_eq!(tokens.len(), 25);
    let decoded = ctx.decode(&tokens);
    assert_eq!(decoded, input);
}

/// Vocab size is 260 (4 special + 256 byte tokens).
#[test]
fn byte_level_vocab_size() {
    let ctx = TokenizerTestContext::with_byte_level();
    let size = unsafe { talu_sys::talu_tokenizer_get_vocab_size(ctx.handle()) };
    assert_eq!(size, 260);
}

// ===========================================================================
// add_prefix_space roundtrip
// ===========================================================================

/// Encode→decode with `add_prefix_space: true` should roundtrip exactly.
///
/// The ByteLevel pretokenizer adds a space (→ Ġ) before the first token.
/// The decoder must strip this added prefix so "Hello" roundtrips to "Hello",
/// not " Hello".
///
/// Bug: the decode path does not track or reverse `add_prefix_space`, so the
/// leading space leaks into the output.
#[test]
fn add_prefix_space_roundtrip() {
    let json = build_byte_level_tokenizer_json()
        .replace("\"add_prefix_space\": false", "\"add_prefix_space\": true");
    let ctx = TokenizerTestContext::from_json(&json);
    let tokens = ctx.encode_with("Hello", &no_bos());
    let decoded = ctx.decode(&tokens);
    assert_eq!(decoded, "Hello", "add_prefix_space roundtrip must not leak leading space");
}
