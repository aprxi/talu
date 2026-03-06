//! Multi-byte UTF-8 and non-ASCII input tests.
//!
//! The minimal fixture has a ByteLevel pre-tokenizer and a 99-token ASCII vocab.
//! Non-ASCII bytes map to characters outside the vocab → `<unk>=3`.
//! Each input byte produces exactly one token (no merges in base fixture).

use crate::capi::tokenizer::common::TokenizerTestContext;

fn no_bos() -> talu_sys::EncodeOptions {
    talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    }
}

// ===========================================================================
// Encode: multi-byte UTF-8
// ===========================================================================

/// Accented Latin: "café" → 5 bytes (é = 2 UTF-8 bytes), both non-ASCII → unk.
#[test]
fn encode_accented_latin() {
    let ctx = TokenizerTestContext::new();
    // c=71, a=69, f=74, 0xC3→unk, 0xA9→unk
    assert_eq!(ctx.encode_with("café", &no_bos()), [71, 69, 74, 3, 3]);
}

/// Single CJK character: "日" = 3 UTF-8 bytes, all → unk.
#[test]
fn encode_cjk_single() {
    let ctx = TokenizerTestContext::new();
    assert_eq!(ctx.encode_with("日", &no_bos()), [3, 3, 3]);
}

/// Two CJK characters: "日本" = 6 UTF-8 bytes, all → unk.
#[test]
fn encode_cjk_two_chars() {
    let ctx = TokenizerTestContext::new();
    assert_eq!(ctx.encode_with("日本", &no_bos()), [3, 3, 3, 3, 3, 3]);
}

/// Single emoji: "🎉" = 4 UTF-8 bytes, all → unk.
#[test]
fn encode_emoji_4byte() {
    let ctx = TokenizerTestContext::new();
    assert_eq!(ctx.encode_with("🎉", &no_bos()), [3, 3, 3, 3]);
}

/// Mixed ASCII + emoji: "Hi🎉" → ASCII tokens + 4 unk.
#[test]
fn encode_mixed_ascii_emoji() {
    let ctx = TokenizerTestContext::new();
    // H=44, i=77, then 4 emoji bytes → unk
    assert_eq!(ctx.encode_with("Hi🎉", &no_bos()), [44, 77, 3, 3, 3, 3]);
}

/// Korean: "한" = 3 UTF-8 bytes → [3, 3, 3].
#[test]
fn encode_korean() {
    let ctx = TokenizerTestContext::new();
    assert_eq!(ctx.encode_with("한", &no_bos()), [3, 3, 3]);
}

/// Russian: "Ж" = 2 UTF-8 bytes → [3, 3].
#[test]
fn encode_cyrillic() {
    let ctx = TokenizerTestContext::new();
    assert_eq!(ctx.encode_with("Ж", &no_bos()), [3, 3]);
}

/// Arabic: "م" = 2 UTF-8 bytes → [3, 3].
#[test]
fn encode_arabic() {
    let ctx = TokenizerTestContext::new();
    assert_eq!(ctx.encode_with("م", &no_bos()), [3, 3]);
}

/// ZWJ emoji sequence: "👨‍👩‍👧‍👦" = 25 UTF-8 bytes, all → unk.
#[test]
fn encode_emoji_zwj_sequence() {
    let ctx = TokenizerTestContext::new();
    let text = "👨\u{200D}👩\u{200D}👧\u{200D}👦"; // family ZWJ sequence
    assert_eq!(text.len(), 25);
    let tokens = ctx.encode_with(text, &no_bos());
    assert_eq!(tokens.len(), 25);
    assert!(tokens.iter().all(|&t| t == 3), "all bytes should be unk");
}

/// Flag emoji: "🇺🇸" = 8 UTF-8 bytes, all → unk.
#[test]
fn encode_flag_emoji() {
    let ctx = TokenizerTestContext::new();
    let text = "🇺🇸";
    assert_eq!(text.len(), 8);
    assert_eq!(ctx.encode_with(text, &no_bos()), [3, 3, 3, 3, 3, 3, 3, 3]);
}

/// Skin-tone emoji: "👍🏽" = 8 UTF-8 bytes, all → unk.
#[test]
fn encode_skin_tone_emoji() {
    let ctx = TokenizerTestContext::new();
    let text = "👍🏽";
    assert_eq!(text.len(), 8);
    assert_eq!(ctx.encode_with(text, &no_bos()), [3, 3, 3, 3, 3, 3, 3, 3]);
}

/// Mixed scripts: ASCII + CJK + Cyrillic in one string.
#[test]
fn encode_mixed_scripts() {
    let ctx = TokenizerTestContext::new();
    // "Hi日" = H(44), i(77), then 3 CJK bytes → unk
    assert_eq!(ctx.encode_with("Hi日", &no_bos()), [44, 77, 3, 3, 3]);

    // "aЖb" = a(69), then 2 Cyrillic bytes → unk, b(70)
    assert_eq!(ctx.encode_with("aЖb", &no_bos()), [69, 3, 3, 70]);
}

/// Token count equals byte count for any input (ByteLevel, no merges).
#[test]
fn token_count_equals_byte_count() {
    let ctx = TokenizerTestContext::new();
    let opts = no_bos();

    let cases = [
        "Hello",     // 5 bytes
        "café",      // 5 bytes
        "日本語",    // 9 bytes
        "🎉🚀",      // 8 bytes
        "Hi 🎉 bye", // 10 bytes
    ];
    for text in cases {
        let tokens = ctx.encode_with(text, &opts);
        assert_eq!(
            tokens.len(),
            text.len(), // &str.len() is byte count in Rust
            "token count should equal byte count for {:?}",
            text
        );
    }
}

// ===========================================================================
// Decode roundtrip
// ===========================================================================

/// ASCII-only text roundtrips exactly through encode→decode.
#[test]
fn decode_roundtrip_ascii() {
    let ctx = TokenizerTestContext::new();
    let opts = no_bos();

    for text in ["Hello", "abc", "012", "!@#"] {
        let decoded = ctx.decode(&ctx.encode_with(text, &opts));
        assert_eq!(decoded, text, "roundtrip failed for {text:?}");
    }
}

/// Mixed text with unk tokens: decode produces unk substitution for non-ASCII.
/// The <unk> token string replaces each non-ASCII byte.
#[test]
fn decode_mixed_ascii_unk() {
    let ctx = TokenizerTestContext::new();
    let tokens = ctx.encode_with("café", &no_bos());
    let decoded = ctx.decode(&tokens);
    assert_eq!(
        decoded, "caf<unk><unk>",
        "non-ASCII bytes should decode as <unk> tokens in base fixture"
    );
}

// ===========================================================================
// Tokenize (string representation) with multi-byte input
// ===========================================================================

/// tokenize "café" returns 5 string tokens (one per byte).
#[test]
fn tokenize_strings_multibyte() {
    let ctx = TokenizerTestContext::new();
    let text = "café"; // 5 UTF-8 bytes
    let result = unsafe {
        talu_sys::talu_tokenizer_tokenize(ctx.handle(), text.as_bytes().as_ptr(), text.len())
    };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 5);

    let ptrs =
        unsafe { std::slice::from_raw_parts(result.tokens as *const *const i8, result.num_tokens) };
    let tokens: Vec<String> = (0..result.num_tokens)
        .map(|i| {
            unsafe { std::ffi::CStr::from_ptr(ptrs[i]) }
                .to_string_lossy()
                .to_string()
        })
        .collect();
    // First 3 are ASCII chars, last 2 are unk tokens.
    assert_eq!(&tokens[0], "c");
    assert_eq!(&tokens[1], "a");
    assert_eq!(&tokens[2], "f");
    assert_eq!(&tokens[3], "<unk>");
    assert_eq!(&tokens[4], "<unk>");

    unsafe { talu_sys::talu_tokenize_result_free(result.tokens, result.num_tokens) };
}

/// tokenize "🎉" returns 4 string tokens (one per byte).
#[test]
fn tokenize_strings_emoji() {
    let ctx = TokenizerTestContext::new();
    let text = "🎉";
    let result = unsafe {
        talu_sys::talu_tokenizer_tokenize(ctx.handle(), text.as_bytes().as_ptr(), text.len())
    };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 4);
    unsafe { talu_sys::talu_tokenize_result_free(result.tokens, result.num_tokens) };
}

/// Stress Unicode normalization/tokenization with dense combining marks.
#[test]
fn encode_zalgo_combining_marks_is_deterministic() {
    let ctx = TokenizerTestContext::with_byte_level();
    let opts = no_bos();
    let combining = "\u{0301}".repeat(200);
    let input = format!("a{combining}");
    let first = ctx.encode_with(&input, &opts);
    let second = ctx.encode_with(&input, &opts);
    assert_eq!(first, second, "combining-mark heavy input must be deterministic");
    assert_eq!(
        first.len(),
        input.len(),
        "byte-level fixture should produce one token per byte"
    );
    let decoded = ctx.decode(&first);
    assert_eq!(decoded, input, "combining-mark heavy input must roundtrip");
}

// ===========================================================================
// Encode offsets with multi-byte input
// ===========================================================================

/// Offsets for "café" (5 bytes): 5 contiguous single-byte spans.
#[test]
fn encode_offsets_multibyte() {
    let ctx = TokenizerTestContext::new();
    let text = "café";
    let result = unsafe { super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 5); // 5 bytes = 5 tokens = 5 offsets

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!(offsets[0].start, 0);
    assert_eq!(offsets.last().unwrap().end as usize, text.len());

    // All spans are contiguous.
    for w in offsets.windows(2) {
        assert_eq!(w[0].end, w[1].start, "offsets should be contiguous");
    }

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Offsets for "🎉" (4 bytes): 4 spans covering [0,4).
#[test]
fn encode_offsets_emoji() {
    let ctx = TokenizerTestContext::new();
    let text = "🎉";
    let result = unsafe { super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 4);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!(offsets[0].start, 0);
    assert_eq!(offsets[3].end as usize, 4);

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Offsets for "Hi🎉bye" (9 bytes): 9 contiguous spans covering [0,9).
#[test]
fn encode_offsets_mixed_ascii_emoji() {
    let ctx = TokenizerTestContext::new();
    let text = "Hi🎉bye";
    let result = unsafe { super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, text.len());

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!(offsets[0].start, 0);
    assert_eq!(offsets.last().unwrap().end as usize, text.len());

    for w in offsets.windows(2) {
        assert_eq!(w[0].end, w[1].start);
    }

    unsafe { talu_sys::talu_encode_result_free(result) };
}

// ===========================================================================
// BPE merges + multi-byte input
// ===========================================================================

/// Offsets for CJK text "日本" (6 UTF-8 bytes): 6 contiguous single-byte spans.
#[test]
fn encode_offsets_cjk() {
    let ctx = TokenizerTestContext::new();
    let text = "日本";
    let result = unsafe { super::common::encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 6); // 6 bytes = 6 tokens = 6 offsets

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!(offsets[0].start, 0);
    assert_eq!(offsets.last().unwrap().end as usize, 6);

    for w in offsets.windows(2) {
        assert_eq!(w[0].end, w[1].start, "offsets should be contiguous");
    }

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// "hello🎉": "hello" merges to one token, emoji bytes → unk.
#[test]
fn merges_with_emoji() {
    let ctx = TokenizerTestContext::with_merges();
    // hello=104, then 4 emoji bytes → 3,3,3,3
    assert_eq!(ctx.encode_with("hello🎉", &no_bos()), [104, 3, 3, 3, 3]);
}

/// "hello world": "hello" merges, space→unk, "world" stays chars.
#[test]
fn merges_with_space() {
    let ctx = TokenizerTestContext::with_merges();
    // hello=104, space=3, w=91, o=83, r=86, l=80, d=72
    assert_eq!(
        ctx.encode_with("hello world", &no_bos()),
        [104, 3, 91, 83, 86, 80, 72]
    );
}
