//! Encode functional tests.
//!
//! Validates token encoding with exact token IDs, EncodeOptions flags,
//! truncation, and roundtrip fidelity.
//!
//! Fixture vocab: each ASCII char maps to a single token (no merges).
//!   H=44, e=73, l=80, o=83, i=77, A=37, a=69, b=70, c=71,
//!   0=20, 1=21, 2=22. Space → <unk>=3 (ByteLevel remaps 0x20).

use crate::capi::tokenizer::common::TokenizerTestContext;

/// Encoding "Hi" without BOS produces exactly [H=44, i=77].
#[test]
fn encode_exact_token_ids() {
    let ctx = TokenizerTestContext::new();
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    assert_eq!(ctx.encode_with("Hi", &opts), [44, 77]);
}

/// Encoding "Hello" without BOS produces [H=44, e=73, l=80, l=80, o=83].
#[test]
fn encode_hello_per_character() {
    let ctx = TokenizerTestContext::new();
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    assert_eq!(ctx.encode_with("Hello", &opts), [44, 73, 80, 80, 83]);
}

/// Digits 0-9 map to IDs 20-29.
#[test]
fn encode_digits() {
    let ctx = TokenizerTestContext::new();
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    assert_eq!(ctx.encode_with("012", &opts), [20, 21, 22]);
}

/// Lowercase a-c map to IDs 69-71.
#[test]
fn encode_lowercase() {
    let ctx = TokenizerTestContext::new();
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    assert_eq!(ctx.encode_with("abc", &opts), [69, 70, 71]);
}

/// Space is remapped by ByteLevel pre-tokenizer; falls back to <unk>=3.
#[test]
fn encode_space_becomes_unk() {
    let ctx = TokenizerTestContext::new();
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    let tokens = ctx.encode_with("ab cd", &opts);
    assert_eq!(tokens, [69, 70, 3, 71, 72]);
}

/// Empty string with add_bos=0 produces zero tokens.
#[test]
fn encode_empty_string() {
    let ctx = TokenizerTestContext::new();
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    assert_eq!(ctx.encode_with("", &opts), Vec::<u32>::new());
}

/// The add_bos flag has no effect with this tokenizer (no post-processor).
/// Both add_bos=0 and add_bos=1 produce identical output.
#[test]
fn encode_bos_flag_no_effect_without_post_processor() {
    let ctx = TokenizerTestContext::new();
    let with_bos = talu_sys::EncodeOptions {
        add_bos: 1,
        ..Default::default()
    };
    let without_bos = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    assert_eq!(
        ctx.encode_with("Hi", &with_bos),
        ctx.encode_with("Hi", &without_bos),
    );
}

/// Right truncation keeps the first max_length tokens.
#[test]
fn encode_truncation_right() {
    let ctx = TokenizerTestContext::new();
    // "Hello" => [44, 73, 80, 80, 83] (5 tokens), truncate to 2.
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        truncation: 1,
        truncation_side: 0, // right: keep first
        max_length: 2,
        ..Default::default()
    };
    assert_eq!(ctx.encode_with("Hello", &opts), [44, 73]);
}

/// Left truncation keeps the last max_length tokens.
#[test]
fn encode_truncation_left() {
    let ctx = TokenizerTestContext::new();
    // "Hello" => [44, 73, 80, 80, 83] (5 tokens), truncate to 2.
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        truncation: 1,
        truncation_side: 1, // left: keep last
        max_length: 2,
        ..Default::default()
    };
    assert_eq!(ctx.encode_with("Hello", &opts), [80, 83]);
}

/// Truncation with max_length >= token count is a no-op.
#[test]
fn encode_truncation_noop_when_under_limit() {
    let ctx = TokenizerTestContext::new();
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        truncation: 1,
        max_length: 1000,
        ..Default::default()
    };
    assert_eq!(ctx.encode_with("Hi", &opts), [44, 77]);
}

/// Repeated pattern with spaces: "abc123 " × 50.
/// Spaces become unk (3), ASCII chars encode normally.
#[test]
fn encode_repeated_pattern_with_spaces() {
    let ctx = TokenizerTestContext::new();
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    let pattern = "abc123 ";
    let input = pattern.repeat(50);
    let tokens = ctx.encode_with(&input, &opts);

    // "abc123 " = 7 bytes → 7 tokens per repetition.
    assert_eq!(tokens.len(), 350);

    // Each repetition: a=69, b=70, c=71, 1=21, 2=22, 3=23, space=3(unk).
    let expected_chunk = [69u32, 70, 71, 21, 22, 23, 3];
    for (i, chunk) in tokens.chunks(7).enumerate() {
        assert_eq!(chunk, expected_chunk, "repetition {i} mismatch");
    }
}

/// count_tokens (encode length) is consistent across multiple methods.
#[test]
fn count_tokens_matches_encode_length() {
    let ctx = TokenizerTestContext::new();
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };

    for text in ["Hello", "abc", "012", "café", "🎉", ""] {
        let encode_len = ctx.encode_with(text, &opts).len();

        // tokenize_bytes should agree on token count.
        let result = unsafe {
            talu_sys::talu_tokenizer_tokenize_bytes(
                ctx.handle(),
                text.as_bytes().as_ptr(),
                text.len(),
            )
        };
        assert!(result.error_msg.is_null());
        assert_eq!(
            result.num_tokens, encode_len,
            "token count mismatch for {text:?}"
        );
        unsafe {
            talu_sys::talu_tokenize_bytes_result_free(
                result.data,
                result.data_len,
                result.offsets,
                result.num_tokens,
            )
        };

        // compute_offsets should also agree.
        let offsets_result = unsafe {
            talu_sys::talu_tokenizer_compute_offsets(
                ctx.handle(),
                text.as_bytes().as_ptr(),
                text.len(),
            )
        };
        assert!(offsets_result.error_msg.is_null());
        assert_eq!(
            offsets_result.len, encode_len,
            "offsets count mismatch for {text:?}"
        );
        unsafe { talu_sys::talu_offsets_free(offsets_result) };
    }
}

/// Encode then decode recovers the original text (non-space ASCII).
#[test]
fn encode_decode_roundtrip() {
    let ctx = TokenizerTestContext::new();
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };
    for input in ["Hello", "abc", "012", "A", "!@#$%"] {
        let tokens = ctx.encode_with(input, &opts);
        let decoded = ctx.decode(&tokens);
        assert_eq!(decoded, input, "roundtrip failed for {input:?}");
    }
}
