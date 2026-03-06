//! Byte-level BPE tests with full 256-byte vocabulary.
//!
//! Uses a fixture with all 256 GPT-2 byte-level tokens so that every input
//! byte maps to a real token (never `<unk>`). This validates the semantic
//! correctness of byte-level BPE tokenization — not just crash safety.
//!
//! Token ID layout: byte b → ID b + 4.

use crate::capi::tokenizer::common::{
    build_byte_level_tokenizer_json, byte_token_id, encode_raw, TokenizerTestContext,
};

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
        assert_eq!(
            tokens[i],
            byte_token_id(b),
            "byte {i} (0x{b:02X}) ID mismatch"
        );
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
        assert_eq!(
            tokens[i],
            byte_token_id(b),
            "byte {i} (0x{b:02X}) ID mismatch"
        );
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
        assert_eq!(
            tokens[i],
            byte_token_id(b),
            "byte {i} (0x{b:02X}) ID mismatch"
        );
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
        assert_eq!(tokens[0], byte_token_id(b), "byte 0x{b:02X} ID mismatch");
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
    let result = unsafe { super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
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
        assert_eq!(
            decoded, *text,
            "roundtrip failed for sequence {i}: {text:?}"
        );
    }
}

/// Text with spaces roundtrips: "Pachelbel's Canon in D".
#[test]
fn text_with_spaces_roundtrip() {
    let ctx = TokenizerTestContext::with_byte_level();
    let input = "Pachelbel's Canon in D";
    let tokens = ctx.encode_with(input, &no_bos());
    let decoded = ctx.decode(&tokens);
    assert_eq!(decoded, input);
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
#[test]
fn add_prefix_space_roundtrip() {
    let json = build_byte_level_tokenizer_json()
        .replace("\"add_prefix_space\": false", "\"add_prefix_space\": true");
    let ctx = TokenizerTestContext::from_json(&json);
    let tokens = ctx.encode_with("Hello", &no_bos());
    let decoded = ctx.decode(&tokens);
    assert_eq!(
        decoded, "Hello",
        "add_prefix_space roundtrip must not leak leading space"
    );
}

/// With `add_prefix_space: true`, a leading synthetic space byte is inserted
/// before inputs that do not already start with a space.
#[test]
fn add_prefix_space_inserts_single_leading_space_token() {
    let json = build_byte_level_tokenizer_json()
        .replace("\"add_prefix_space\": false", "\"add_prefix_space\": true");
    let ctx = TokenizerTestContext::from_json(&json);

    let tokens = ctx.encode_with("Hello", &no_bos());
    let expected = vec![
        byte_token_id(b' '),
        byte_token_id(b'H'),
        byte_token_id(b'e'),
        byte_token_id(b'l'),
        byte_token_id(b'l'),
        byte_token_id(b'o'),
    ];
    assert_eq!(tokens, expected);
}

/// If the input already starts with a space, ByteLevel must not inject a
/// second synthetic prefix space.
#[test]
fn add_prefix_space_does_not_duplicate_existing_leading_space() {
    let json = build_byte_level_tokenizer_json()
        .replace("\"add_prefix_space\": false", "\"add_prefix_space\": true");
    let ctx = TokenizerTestContext::from_json(&json);

    let tokens = ctx.encode_with(" Hello", &no_bos());
    let expected: Vec<u32> = " Hello".as_bytes().iter().map(|&b| byte_token_id(b)).collect();
    assert_eq!(tokens, expected);
    // Decode cannot distinguish a real leading space from a synthetic
    // add_prefix_space byte once they collapse to the same ID stream. The
    // meaningful contract here is encode-time: existing-leading-space input
    // must produce exactly one leading space token, not two.
}

/// Batch encoding applies `add_prefix_space` independently to each sequence.
#[test]
fn add_prefix_space_applies_per_batch_sequence() {
    let json = build_byte_level_tokenizer_json()
        .replace("\"add_prefix_space\": false", "\"add_prefix_space\": true");
    let ctx = TokenizerTestContext::from_json(&json);

    let batch = ctx.encode_batch(&["Hello", "World"], &no_bos());
    assert_eq!(batch.num_sequences, 2);
    assert_eq!(batch.offsets, vec![0, 6, 12]);

    let seq0 = &batch.ids[batch.offsets[0]..batch.offsets[1]];
    let seq1 = &batch.ids[batch.offsets[1]..batch.offsets[2]];
    assert_eq!(
        seq0,
        &[
            byte_token_id(b' '),
            byte_token_id(b'H'),
            byte_token_id(b'e'),
            byte_token_id(b'l'),
            byte_token_id(b'l'),
            byte_token_id(b'o')
        ]
    );
    assert_eq!(
        seq1,
        &[
            byte_token_id(b' '),
            byte_token_id(b'W'),
            byte_token_id(b'o'),
            byte_token_id(b'r'),
            byte_token_id(b'l'),
            byte_token_id(b'd')
        ]
    );
}

/// The synthetic prefix space must appear on token surfaces as a leading Ġ
/// token, but it must not decode into a persistent leading space.
#[test]
fn add_prefix_space_token_surfaces_show_single_prefix_marker() {
    let json = build_byte_level_tokenizer_json()
        .replace("\"add_prefix_space\": false", "\"add_prefix_space\": true");
    let ctx = TokenizerTestContext::from_json(&json);

    assert_eq!(tokenize_strings(&ctx, "Hello"), ["Ġ", "H", "e", "l", "l", "o"]);
    assert_eq!(
        tokenize_bytes_strings(&ctx, "Hello"),
        ["Ġ", "H", "e", "l", "l", "o"]
    );
}

/// The synthetic prefix token must not claim a real source-byte span.
#[test]
fn add_prefix_space_synthetic_token_has_zero_width_offset() {
    let json = build_byte_level_tokenizer_json()
        .replace("\"add_prefix_space\": false", "\"add_prefix_space\": true");
    let ctx = TokenizerTestContext::from_json(&json);

    let result = unsafe { encode_raw(ctx.handle(), b"Hello", &no_bos()) };
    assert!(result.error_msg.is_null(), "encode failed");
    assert_eq!(result.num_tokens, 6);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!(
        (offsets[0].start, offsets[0].end),
        (0, 0),
        "synthetic prefix space must have zero-width source offset"
    );
    assert_eq!((offsets[1].start, offsets[1].end), (0, 1));
    assert_eq!((offsets[2].start, offsets[2].end), (1, 2));
    assert_eq!((offsets[3].start, offsets[3].end), (2, 3));
    assert_eq!((offsets[4].start, offsets[4].end), (3, 4));
    assert_eq!((offsets[5].start, offsets[5].end), (4, 5));

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// A real leading space in the source must retain its own non-zero span.
#[test]
fn add_prefix_space_real_leading_space_keeps_real_offset() {
    let json = build_byte_level_tokenizer_json()
        .replace("\"add_prefix_space\": false", "\"add_prefix_space\": true");
    let ctx = TokenizerTestContext::from_json(&json);

    let result = unsafe { encode_raw(ctx.handle(), b" Hello", &no_bos()) };
    assert!(result.error_msg.is_null(), "encode failed");
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!(result.num_tokens, 6);
    assert_eq!(
        (offsets[0].start, offsets[0].end),
        (0, 1),
        "real leading space must keep its source-byte span"
    );
    assert_eq!((offsets[1].start, offsets[1].end), (1, 2));
    assert_eq!((offsets[5].start, offsets[5].end), (5, 6));

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// An existing leading space should surface as exactly one ByteLevel marker,
/// not a duplicated synthetic+real pair.
#[test]
fn add_prefix_space_existing_leading_space_has_single_marker_on_token_surfaces() {
    let json = build_byte_level_tokenizer_json()
        .replace("\"add_prefix_space\": false", "\"add_prefix_space\": true");
    let ctx = TokenizerTestContext::from_json(&json);

    assert_eq!(tokenize_strings(&ctx, " Hello"), ["Ġ", "H", "e", "l", "l", "o"]);
    assert_eq!(
        tokenize_bytes_strings(&ctx, " Hello"),
        ["Ġ", "H", "e", "l", "l", "o"]
    );
}

/// In batch mode, each sequence must decide prefix insertion from its own
/// first byte: existing-space inputs keep one real marker, non-space inputs
/// get one synthetic marker.
#[test]
fn add_prefix_space_mixed_batch_keeps_per_sequence_prefix_contract() {
    let json = build_byte_level_tokenizer_json()
        .replace("\"add_prefix_space\": false", "\"add_prefix_space\": true");
    let ctx = TokenizerTestContext::from_json(&json);

    let batch = ctx.encode_batch(&[" Hello", "World"], &no_bos());
    assert_eq!(batch.num_sequences, 2);
    assert_eq!(batch.offsets, vec![0, 6, 12]);

    let seq0 = &batch.ids[0..6];
    let seq1 = &batch.ids[6..12];
    let expected0: Vec<u32> = " Hello".as_bytes().iter().map(|&b| byte_token_id(b)).collect();
    let expected1 = vec![
        byte_token_id(b' '),
        byte_token_id(b'W'),
        byte_token_id(b'o'),
        byte_token_id(b'r'),
        byte_token_id(b'l'),
        byte_token_id(b'd'),
    ];
    assert_eq!(seq0, expected0);
    assert_eq!(seq1, expected1);
}

/// Right truncation must apply after the synthetic prefix token has been
/// inserted, so the prefix remains part of the kept prefix window.
#[test]
fn add_prefix_space_right_truncation_keeps_prefix_token() {
    let json = build_byte_level_tokenizer_json()
        .replace("\"add_prefix_space\": false", "\"add_prefix_space\": true");
    let ctx = TokenizerTestContext::from_json(&json);
    let opts = talu_sys::EncodeOptions {
        truncation: 1,
        truncation_side: 0,
        max_length: 3,
        ..Default::default()
    };

    let tokens = ctx.encode_with("Hello", &opts);
    assert_eq!(
        tokens,
        vec![byte_token_id(b' '), byte_token_id(b'H'), byte_token_id(b'e')]
    );
    assert_eq!(ctx.decode(&tokens), "He");
}

/// Left truncation must keep the last N tokens after prefix insertion, which
/// means the synthetic prefix token is dropped when the retained window is at
/// the tail of the sequence.
#[test]
fn add_prefix_space_left_truncation_drops_prefix_token() {
    let json = build_byte_level_tokenizer_json()
        .replace("\"add_prefix_space\": false", "\"add_prefix_space\": true");
    let ctx = TokenizerTestContext::from_json(&json);
    let opts = talu_sys::EncodeOptions {
        truncation: 1,
        truncation_side: 1,
        max_length: 3,
        ..Default::default()
    };

    let tokens = ctx.encode_with("Hello", &opts);
    assert_eq!(
        tokens,
        vec![byte_token_id(b'l'), byte_token_id(b'l'), byte_token_id(b'o')]
    );
    assert_eq!(ctx.decode(&tokens), "llo");
}

/// Right truncation with `add_prefix_space` must keep the zero-width synthetic
/// prefix offset followed by the first retained source-byte spans.
#[test]
fn add_prefix_space_right_truncation_preserves_prefix_offset_contract() {
    let json = build_byte_level_tokenizer_json()
        .replace("\"add_prefix_space\": false", "\"add_prefix_space\": true");
    let ctx = TokenizerTestContext::from_json(&json);
    let opts = talu_sys::EncodeOptions {
        truncation: 1,
        truncation_side: 0,
        max_length: 3,
        ..Default::default()
    };

    let result = unsafe { encode_raw(ctx.handle(), b"Hello", &opts) };
    assert!(result.error_msg.is_null(), "encode failed");
    assert_eq!(result.num_tokens, 3);
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (0, 0));
    assert_eq!((offsets[1].start, offsets[1].end), (0, 1));
    assert_eq!((offsets[2].start, offsets[2].end), (1, 2));
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Left truncation with `add_prefix_space` must drop the synthetic prefix and
/// preserve the original tail byte spans.
#[test]
fn add_prefix_space_left_truncation_preserves_tail_offsets() {
    let json = build_byte_level_tokenizer_json()
        .replace("\"add_prefix_space\": false", "\"add_prefix_space\": true");
    let ctx = TokenizerTestContext::from_json(&json);
    let opts = talu_sys::EncodeOptions {
        truncation: 1,
        truncation_side: 1,
        max_length: 3,
        ..Default::default()
    };

    let result = unsafe { encode_raw(ctx.handle(), b"Hello", &opts) };
    assert!(result.error_msg.is_null(), "encode failed");
    assert_eq!(result.num_tokens, 3);
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!((offsets[0].start, offsets[0].end), (2, 3));
    assert_eq!((offsets[1].start, offsets[1].end), (3, 4));
    assert_eq!((offsets[2].start, offsets[2].end), (4, 5));
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Empty input must stay empty even when `add_prefix_space` is enabled.
#[test]
fn add_prefix_space_empty_input_stays_empty() {
    let json = build_byte_level_tokenizer_json()
        .replace("\"add_prefix_space\": false", "\"add_prefix_space\": true");
    let ctx = TokenizerTestContext::from_json(&json);

    let tokens = ctx.encode_with("", &no_bos());
    assert!(tokens.is_empty(), "empty input must not gain a synthetic prefix token");
    assert_eq!(ctx.decode(&tokens), "");
    assert!(tokenize_strings(&ctx, "").is_empty());
    assert!(tokenize_bytes_strings(&ctx, "").is_empty());
}
