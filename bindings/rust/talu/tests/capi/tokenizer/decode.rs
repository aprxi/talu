//! Decode functional tests.
//!
//! Validates token-to-text decoding with exact assertions.
//! Uses hardcoded token IDs from the fixture vocab.

use crate::capi::tokenizer::common::TokenizerTestContext;

/// Decoding [44, 77] produces "Hi" (H=44, i=77).
#[test]
fn decode_known_ids() {
    let ctx = TokenizerTestContext::new();
    assert_eq!(ctx.decode(&[44, 77]), "Hi");
}

/// Decoding [44, 73, 80, 80, 83] produces "Hello".
#[test]
fn decode_hello() {
    let ctx = TokenizerTestContext::new();
    assert_eq!(ctx.decode(&[44, 73, 80, 80, 83]), "Hello");
}

/// Decoding a single token: A=37.
#[test]
fn decode_single_token() {
    let ctx = TokenizerTestContext::new();
    assert_eq!(ctx.decode(&[37]), "A");
}

/// Decoding an empty token array produces an empty string.
#[test]
fn decode_empty() {
    let ctx = TokenizerTestContext::new();
    assert_eq!(ctx.decode(&[]), "");
}

/// In the base fixture, special tokens are in both `model.vocab` and
/// `added_tokens`. The BPE decoder checks `id_to_token` first (which
/// always sets `is_special=false`), so `skip_special_tokens` has no
/// observable effect. This locks down that behavior.
#[test]
fn decode_bos_token_base_fixture() {
    let ctx = TokenizerTestContext::new();
    let skip = talu_sys::DecodeOptionsC { skip_special_tokens: 1 };
    let retain = talu_sys::DecodeOptionsC { skip_special_tokens: 0 };

    assert_eq!(ctx.decode_with(&[1], &skip), "<s>");
    assert_eq!(ctx.decode_with(&[1], &retain), "<s>");
}

/// Roundtrip for multiple known strings.
#[test]
fn decode_roundtrip() {
    let ctx = TokenizerTestContext::new();
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    };

    for input in ["Hello", "abc123", "A", "!@#$%"] {
        let decoded = ctx.decode(&ctx.encode_with(input, &opts));
        assert_eq!(decoded, input, "roundtrip failed for {input:?}");
    }
}

// ===========================================================================
// skip_special_tokens with special-tokens-only fixture
// ===========================================================================
//
// Uses `with_special_tokens()`: special tokens (IDs 0–3) are ONLY in
// `added_tokens`, not in `model.vocab`. This makes `skip_special_tokens`
// observable — the BPE decoder falls through to `added_tokens` and reads
// the `special` flag.

/// skip_special_tokens=1 strips BOS from decode output.
#[test]
fn skip_special_strips_bos() {
    let ctx = TokenizerTestContext::with_special_tokens();
    let skip = talu_sys::DecodeOptionsC { skip_special_tokens: 1 };
    // [BOS=1, H=44, i=77]
    assert_eq!(ctx.decode_with(&[1, 44, 77], &skip), "Hi");
}

/// skip_special_tokens=0 retains BOS in decode output.
#[test]
fn retain_special_keeps_bos() {
    let ctx = TokenizerTestContext::with_special_tokens();
    let retain = talu_sys::DecodeOptionsC { skip_special_tokens: 0 };
    assert_eq!(ctx.decode_with(&[1, 44, 77], &retain), "<s>Hi");
}

/// skip_special_tokens=1 strips both BOS and EOS.
#[test]
fn skip_special_strips_bos_and_eos() {
    let ctx = TokenizerTestContext::with_special_tokens();
    let skip = talu_sys::DecodeOptionsC { skip_special_tokens: 1 };
    // [BOS=1, H=44, i=77, EOS=2]
    assert_eq!(ctx.decode_with(&[1, 44, 77, 2], &skip), "Hi");
}

/// skip_special_tokens=0 retains both BOS and EOS.
#[test]
fn retain_special_keeps_bos_and_eos() {
    let ctx = TokenizerTestContext::with_special_tokens();
    let retain = talu_sys::DecodeOptionsC { skip_special_tokens: 0 };
    assert_eq!(ctx.decode_with(&[1, 44, 77, 2], &retain), "<s>Hi</s>");
}

/// Decoding only special tokens with skip=1 produces empty string.
#[test]
fn skip_special_all_special_produces_empty() {
    let ctx = TokenizerTestContext::with_special_tokens();
    let skip = talu_sys::DecodeOptionsC { skip_special_tokens: 1 };
    assert_eq!(ctx.decode_with(&[1, 2], &skip), "");
}

/// BOS sandwiched in content is also stripped when skip=1.
#[test]
fn skip_special_strips_sandwiched_bos() {
    let ctx = TokenizerTestContext::with_special_tokens();
    let skip = talu_sys::DecodeOptionsC { skip_special_tokens: 1 };
    // [H=44, BOS=1, i=77]
    assert_eq!(ctx.decode_with(&[44, 1, 77], &skip), "Hi");
}
