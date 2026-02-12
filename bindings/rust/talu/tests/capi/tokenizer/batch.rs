//! Batch encode and padded tensor tests.
//!
//! Validates encode_batch offset arithmetic and padded tensor layout.

use crate::capi::tokenizer::common::TokenizerTestContext;

/// Default encode options (no BOS).
fn no_bos() -> talu_sys::EncodeOptions {
    talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    }
}

/// Batch encoding a single text matches individual encode.
#[test]
fn batch_single_matches_individual() {
    let ctx = TokenizerTestContext::new();
    let batch = ctx.encode_batch(&["Hello"], &no_bos());

    assert_eq!(batch.num_sequences, 1);
    assert_eq!(batch.offsets, [0, 5]);
    assert_eq!(batch.ids, [44, 73, 80, 80, 83]);
}

/// Batch encoding multiple texts: offsets partition the flat ID array correctly.
#[test]
fn batch_three_texts_exact_offsets() {
    let ctx = TokenizerTestContext::new();
    // "Hi"=2 tokens, "A"=1 token, "abc"=3 tokens → offsets [0,2,3,6].
    let batch = ctx.encode_batch(&["Hi", "A", "abc"], &no_bos());

    assert_eq!(batch.num_sequences, 3);
    assert_eq!(batch.offsets, [0, 2, 3, 6]);
    assert_eq!(&batch.ids[0..2], &[44, 77]);    // "Hi"
    assert_eq!(&batch.ids[2..3], &[37]);         // "A"
    assert_eq!(&batch.ids[3..6], &[69, 70, 71]); // "abc"
}

/// Batch encoding zero texts produces empty result.
#[test]
fn batch_zero_texts() {
    let ctx = TokenizerTestContext::new();
    let batch = ctx.encode_batch(&[], &no_bos());
    assert_eq!(batch.num_sequences, 0);
    assert!(batch.ids.is_empty());
    assert!(batch.offsets.is_empty());
}

/// Padded tensor: right-padding aligns "A"(1 token) and "Hello"(5 tokens)
/// to padded_length=5 with pad_id=0 filling the short sequence.
#[test]
fn padded_tensor_right_padding() {
    let ctx = TokenizerTestContext::new();
    let pad_opts = talu_sys::PaddedTensorOptions {
        pad_id: 0,
        padding_side: 0, // right
        return_attention_mask: true,
        ..Default::default()
    };

    let result = ctx.batch_to_padded_tensor(&["A", "Hello"], &no_bos(), &pad_opts);

    assert_eq!(result.num_sequences, 2);
    assert_eq!(result.padded_length, 5);

    // Row 0: "A" = [37, 0, 0, 0, 0] (right-padded).
    assert_eq!(&result.input_ids[0..5], &[37, 0, 0, 0, 0]);
    assert_eq!(&result.attention_mask[0..5], &[1, 0, 0, 0, 0]);

    // Row 1: "Hello" = [44, 73, 80, 80, 83] (no padding needed).
    assert_eq!(&result.input_ids[5..10], &[44, 73, 80, 80, 83]);
    assert_eq!(&result.attention_mask[5..10], &[1, 1, 1, 1, 1]);
}

/// Padded tensor: left-padding puts pad tokens at the start of short sequences.
#[test]
fn padded_tensor_left_padding() {
    let ctx = TokenizerTestContext::new();
    let pad_opts = talu_sys::PaddedTensorOptions {
        pad_id: 0,
        padding_side: 1, // left
        return_attention_mask: true,
        ..Default::default()
    };

    let result = ctx.batch_to_padded_tensor(&["A", "Hello"], &no_bos(), &pad_opts);

    assert_eq!(result.padded_length, 5);

    // Row 0: "A" = [0, 0, 0, 0, 37] (left-padded).
    assert_eq!(&result.input_ids[0..5], &[0, 0, 0, 0, 37]);
    assert_eq!(&result.attention_mask[0..5], &[0, 0, 0, 0, 1]);

    // Row 1: "Hello" = [44, 73, 80, 80, 83] (no padding needed).
    assert_eq!(&result.input_ids[5..10], &[44, 73, 80, 80, 83]);
    assert_eq!(&result.attention_mask[5..10], &[1, 1, 1, 1, 1]);
}

/// Padded tensor with return_attention_mask=false produces empty mask.
#[test]
fn padded_tensor_no_mask() {
    let ctx = TokenizerTestContext::new();
    let pad_opts = talu_sys::PaddedTensorOptions {
        return_attention_mask: false,
        ..Default::default()
    };

    let result = ctx.batch_to_padded_tensor(&["Hi"], &no_bos(), &pad_opts);
    assert_eq!(result.num_sequences, 1);
    assert!(result.attention_mask.is_empty());
    assert_eq!(result.input_ids, [44, 77]);
}

/// Equal-length sequences produce no padding.
#[test]
fn padded_tensor_equal_lengths_no_padding() {
    let ctx = TokenizerTestContext::new();
    let pad_opts = talu_sys::PaddedTensorOptions {
        pad_id: 99,
        padding_side: 0,
        return_attention_mask: true,
        ..Default::default()
    };

    // "Hi"=2 tokens, "ab"=2 tokens → padded_length=2, no padding.
    let result = ctx.batch_to_padded_tensor(&["Hi", "ab"], &no_bos(), &pad_opts);

    assert_eq!(result.padded_length, 2);
    assert_eq!(&result.input_ids[0..2], &[44, 77]);
    assert_eq!(&result.input_ids[2..4], &[69, 70]);
    // All mask values are 1 (no padding injected).
    assert_eq!(result.attention_mask, [1, 1, 1, 1]);
}

/// Batch encode with unicode edge cases: null byte, ZWS, emoji, CJK, Arabic.
#[test]
fn batch_encode_unicode_edge_cases() {
    let ctx = TokenizerTestContext::new();
    let batch = ctx.encode_batch(
        &["\x00", "\u{200b}", "🎉", "日", "م"],
        &no_bos(),
    );

    assert_eq!(batch.num_sequences, 5);
    // "\x00"=1 byte→[3], "\u{200b}"=3 bytes→[3,3,3],
    // "🎉"=4 bytes→[3,3,3,3], "日"=3 bytes→[3,3,3], "م"=2 bytes→[3,3]
    assert_eq!(batch.offsets, [0, 1, 4, 8, 11, 13]);
    assert!(batch.ids.iter().all(|&id| id == 3), "all should be unk");
}

// ===========================================================================
// PaddedTensorOptions: max_length and truncate
// ===========================================================================

/// max_length + truncate=true: sequences longer than max_length are truncated.
#[test]
fn padded_tensor_max_length_truncates() {
    let ctx = TokenizerTestContext::new();
    let pad_opts = talu_sys::PaddedTensorOptions {
        pad_id: 0,
        padding_side: 0,
        max_length: 3,
        truncate: true,
        return_attention_mask: true,
    };

    // "Hello"=5 tokens, "Hi"=2 tokens.
    let result = ctx.batch_to_padded_tensor(&["Hello", "Hi"], &no_bos(), &pad_opts);

    assert_eq!(result.num_sequences, 2);
    assert_eq!(result.padded_length, 3);

    // Row 0: "Hello" truncated to first 3 tokens: [44, 73, 80].
    assert_eq!(&result.input_ids[0..3], &[44, 73, 80]);
    assert_eq!(&result.attention_mask[0..3], &[1, 1, 1]);

    // Row 1: "Hi" = [44, 77, 0] (right-padded to length 3).
    assert_eq!(&result.input_ids[3..6], &[44, 77, 0]);
    assert_eq!(&result.attention_mask[3..6], &[1, 1, 0]);
}

/// max_length without truncate: sequences longer than max_length are NOT truncated.
#[test]
fn padded_tensor_max_length_no_truncate_pads() {
    let ctx = TokenizerTestContext::new();
    let pad_opts = talu_sys::PaddedTensorOptions {
        pad_id: 0,
        padding_side: 0,
        max_length: 10,
        truncate: false,
        return_attention_mask: true,
    };

    // "Hi"=2 tokens → padded to max_length=10.
    let result = ctx.batch_to_padded_tensor(&["Hi"], &no_bos(), &pad_opts);

    assert_eq!(result.padded_length, 10);
    // First 2 are content, rest are padding.
    assert_eq!(&result.input_ids[0..2], &[44, 77]);
    assert_eq!(&result.input_ids[2..10], &[0, 0, 0, 0, 0, 0, 0, 0]);
    // Mask: content=1, padding=0.
    assert_eq!(&result.attention_mask[0..2], &[1, 1]);
    assert_eq!(&result.attention_mask[2..10], &[0, 0, 0, 0, 0, 0, 0, 0]);
}

/// max_length exactly equals token count: no padding, no truncation.
#[test]
fn padded_tensor_max_length_at_boundary() {
    let ctx = TokenizerTestContext::new();
    let pad_opts = talu_sys::PaddedTensorOptions {
        pad_id: 0,
        padding_side: 0,
        max_length: 2,
        truncate: true,
        return_attention_mask: true,
    };

    // "Hi"=2 tokens, max_length=2: exact fit.
    let result = ctx.batch_to_padded_tensor(&["Hi"], &no_bos(), &pad_opts);

    assert_eq!(result.padded_length, 2);
    assert_eq!(result.input_ids, [44, 77]);
    assert_eq!(result.attention_mask, [1, 1]);
}

/// Left padding with truncation: combined behavior.
#[test]
fn padded_tensor_left_pad_with_truncation() {
    let ctx = TokenizerTestContext::new();
    let pad_opts = talu_sys::PaddedTensorOptions {
        pad_id: 0,
        padding_side: 1, // left
        max_length: 3,
        truncate: true,
        return_attention_mask: true,
    };

    // "Hello"=5 tokens (truncated to 3), "A"=1 token (left-padded to 3).
    let result = ctx.batch_to_padded_tensor(&["Hello", "A"], &no_bos(), &pad_opts);

    assert_eq!(result.padded_length, 3);

    // Row 0: "Hello" truncated: [44, 73, 80].
    assert_eq!(&result.input_ids[0..3], &[44, 73, 80]);
    assert_eq!(&result.attention_mask[0..3], &[1, 1, 1]);

    // Row 1: "A" left-padded: [0, 0, 37].
    assert_eq!(&result.input_ids[3..6], &[0, 0, 37]);
    assert_eq!(&result.attention_mask[3..6], &[0, 0, 1]);
}

/// Five sequences of varied lengths: verify every mask cell.
#[test]
fn padded_tensor_complex_batch_mask() {
    let ctx = TokenizerTestContext::new();
    let pad_opts = talu_sys::PaddedTensorOptions {
        pad_id: 99,
        padding_side: 0, // right
        return_attention_mask: true,
        ..Default::default()
    };

    // Lengths: "A"=1, "Hi"=2, "abc"=3, "Hello"=5, ""=0.
    let result =
        ctx.batch_to_padded_tensor(&["A", "Hi", "abc", "Hello", ""], &no_bos(), &pad_opts);

    assert_eq!(result.num_sequences, 5);
    assert_eq!(result.padded_length, 5); // longest is "Hello"=5

    let pl = result.padded_length;

    // Row 0: "A" = [37, 99, 99, 99, 99], mask [1, 0, 0, 0, 0].
    assert_eq!(result.input_ids[0], 37);
    assert_eq!(result.attention_mask[0], 1);
    for j in 1..pl {
        assert_eq!(result.input_ids[j], 99, "row 0 col {j} should be pad");
        assert_eq!(result.attention_mask[j], 0, "row 0 col {j} mask should be 0");
    }

    // Row 3: "Hello" = [44,73,80,80,83], all mask=1.
    let start = 3 * pl;
    assert_eq!(&result.input_ids[start..start + 5], &[44, 73, 80, 80, 83]);
    for j in 0..pl {
        assert_eq!(result.attention_mask[start + j], 1, "row 3 col {j} mask should be 1");
    }

    // Row 4: empty = all pad, all mask=0.
    let start = 4 * pl;
    for j in 0..pl {
        assert_eq!(result.input_ids[start + j], 99, "row 4 col {j} should be pad");
        assert_eq!(result.attention_mask[start + j], 0, "row 4 col {j} mask should be 0");
    }
}

// ===========================================================================
// Token utilities
// ===========================================================================

/// tokens_concat joins two token arrays.
#[test]
fn tokens_concat() {
    let a = [44u32, 77]; // "Hi"
    let b = [69u32, 70, 71]; // "abc"

    let result = unsafe {
        talu_sys::talu_tokens_concat(a.as_ptr(), a.len(), b.as_ptr(), b.len())
    };
    assert!(!result.is_null());

    let combined = unsafe { std::slice::from_raw_parts(result, 5) };
    assert_eq!(combined, [44, 77, 69, 70, 71]);

    unsafe { talu_sys::talu_tokens_free(result, 5) };
}
