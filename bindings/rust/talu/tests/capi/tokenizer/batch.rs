//! Batch encode and padded tensor tests.
//!
//! Validates encode_batch offset arithmetic and padded tensor layout.

use crate::capi::tokenizer::common::{
    build_byte_level_tokenizer_json, byte_token_id, TokenizerTestContext,
};

/// Default encode options (no BOS).
fn no_bos() -> talu_sys::EncodeOptions {
    talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    }
}

/// Minimal BPE + TemplateProcessing fixture used to test BOS/EOS option semantics.
const TEMPLATE_BATCH_JSON: &str = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"H": 4, "i": 5},
    "merges": []
  },
  "added_tokens": [
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": {
    "type": "TemplateProcessing",
    "single": [
      {"SpecialToken": {"id": "<s>", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}},
      {"SpecialToken": {"id": "</s>", "type_id": 0}}
    ],
    "pair": [
      {"SpecialToken": {"id": "<s>", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}},
      {"SpecialToken": {"id": "</s>", "type_id": 0}},
      {"Sequence": {"id": "B", "type_id": 1}},
      {"SpecialToken": {"id": "</s>", "type_id": 0}}
    ],
    "special_tokens": {
      "<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]},
      "</s>": {"id": "</s>", "ids": [2], "tokens": ["</s>"]}
    }
  },
  "decoder": {"type": "ByteLevel"}
}"####;

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
    assert_eq!(&batch.ids[0..2], &[44, 77]); // "Hi"
    assert_eq!(&batch.ids[2..3], &[37]); // "A"
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

/// ByteLevel `add_prefix_space` increases each non-space-prefixed sequence by
/// one token in batch mode.
#[test]
fn batch_encode_byte_level_add_prefix_space_shifts_sequence_lengths() {
    let json = build_byte_level_tokenizer_json()
        .replace("\"add_prefix_space\": false", "\"add_prefix_space\": true");
    let ctx = TokenizerTestContext::from_json(&json);

    let batch = ctx.encode_batch(&["Hello", "Hi"], &no_bos());
    assert_eq!(batch.num_sequences, 2);
    assert_eq!(batch.offsets, vec![0, 6, 9]);
    assert_eq!(
        &batch.ids[0..6],
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
        &batch.ids[6..9],
        &[byte_token_id(b' '), byte_token_id(b'H'), byte_token_id(b'i')]
    );
}

/// Empty sequences must remain empty in batch mode even when ByteLevel
/// `add_prefix_space` is enabled.
#[test]
fn batch_encode_byte_level_add_prefix_space_empty_sequence_stays_empty() {
    let json = build_byte_level_tokenizer_json()
        .replace("\"add_prefix_space\": false", "\"add_prefix_space\": true");
    let ctx = TokenizerTestContext::from_json(&json);

    let batch = ctx.encode_batch(&["", "Hi"], &no_bos());
    assert_eq!(batch.num_sequences, 2);
    assert_eq!(batch.offsets, vec![0, 0, 3]);
    assert_eq!(
        batch.ids,
        vec![byte_token_id(b' '), byte_token_id(b'H'), byte_token_id(b'i')]
    );
}

/// Batch slicing must preserve the original ID stream for sequences that
/// already begin with a real space, even when `add_prefix_space` is enabled.
#[test]
fn batch_encode_byte_level_add_prefix_space_roundtrip_preserves_real_leading_space() {
    let json = build_byte_level_tokenizer_json()
        .replace("\"add_prefix_space\": false", "\"add_prefix_space\": true");
    let ctx = TokenizerTestContext::from_json(&json);

    let batch = ctx.encode_batch(&[" Hello", "Hi"], &no_bos());
    assert_eq!(batch.num_sequences, 2);

    let seq0 = &batch.ids[batch.offsets[0]..batch.offsets[1]];
    let seq1 = &batch.ids[batch.offsets[1]..batch.offsets[2]];
    let expected0: Vec<u32> = " Hello".as_bytes().iter().map(|&b| byte_token_id(b)).collect();
    let expected1 = vec![
        byte_token_id(b' '),
        byte_token_id(b'H'),
        byte_token_id(b'i'),
    ];
    assert_eq!(seq0, expected0);
    assert_eq!(seq1, expected1);
}

/// Batch slices must match individual encodes even when the batch mixes empty,
/// real-leading-space, and synthetic-prefix-space cases.
#[test]
fn batch_encode_byte_level_add_prefix_space_matches_individual_mixed_sequences() {
    let json = build_byte_level_tokenizer_json()
        .replace("\"add_prefix_space\": false", "\"add_prefix_space\": true");
    let ctx = TokenizerTestContext::from_json(&json);

    let texts = ["", " Hello", "Hi"];
    let batch = ctx.encode_batch(&texts, &no_bos());
    assert_eq!(batch.num_sequences, 3);

    for (i, text) in texts.iter().enumerate() {
        let start = batch.offsets[i];
        let end = batch.offsets[i + 1];
        let expected = ctx.encode_with(text, &no_bos());
        assert_eq!(
            &batch.ids[start..end],
            expected.as_slice(),
            "batch slice must equal individual encode for sequence {i}: {text:?}"
        );
    }
}

/// Under right truncation, each batch slice must still equal the individual
/// encode result for that same sequence and option set.
#[test]
fn batch_encode_byte_level_add_prefix_space_truncation_right_matches_individual_sequences() {
    let json = build_byte_level_tokenizer_json()
        .replace("\"add_prefix_space\": false", "\"add_prefix_space\": true");
    let ctx = TokenizerTestContext::from_json(&json);
    let texts = ["", " Hello", "World"];
    let opts = talu_sys::EncodeOptions {
        truncation: 1,
        truncation_side: 0,
        max_length: 3,
        ..Default::default()
    };

    let batch = ctx.encode_batch(&texts, &opts);
    assert_eq!(batch.num_sequences, 3);

    for (i, text) in texts.iter().enumerate() {
        let start = batch.offsets[i];
        let end = batch.offsets[i + 1];
        let expected = ctx.encode_with(text, &opts);
        assert_eq!(
            &batch.ids[start..end],
            expected.as_slice(),
            "right-truncated batch slice must equal individual encode for sequence {i}: {text:?}"
        );
    }
}

/// Under left truncation, each batch slice must still equal the individual
/// encode result for that same sequence and option set.
#[test]
fn batch_encode_byte_level_add_prefix_space_truncation_left_matches_individual_sequences() {
    let json = build_byte_level_tokenizer_json()
        .replace("\"add_prefix_space\": false", "\"add_prefix_space\": true");
    let ctx = TokenizerTestContext::from_json(&json);
    let texts = ["", " Hello", "World"];
    let opts = talu_sys::EncodeOptions {
        truncation: 1,
        truncation_side: 1,
        max_length: 3,
        ..Default::default()
    };

    let batch = ctx.encode_batch(&texts, &opts);
    assert_eq!(batch.num_sequences, 3);

    for (i, text) in texts.iter().enumerate() {
        let start = batch.offsets[i];
        let end = batch.offsets[i + 1];
        let expected = ctx.encode_with(text, &opts);
        assert_eq!(
            &batch.ids[start..end],
            expected.as_slice(),
            "left-truncated batch slice must equal individual encode for sequence {i}: {text:?}"
        );
    }
}

/// Right truncation must apply after the synthetic prefix-space token has been
/// inserted for each non-space-prefixed sequence.
#[test]
fn batch_encode_byte_level_add_prefix_space_truncation_right_preserves_prefix_token() {
    let json = build_byte_level_tokenizer_json()
        .replace("\"add_prefix_space\": false", "\"add_prefix_space\": true");
    let ctx = TokenizerTestContext::from_json(&json);
    let t0 = b"Hello";
    let t1 = b" Hi";
    let ptrs = [t0.as_ptr(), t1.as_ptr()];
    let lengths = [t0.len(), t1.len()];
    let opts = talu_sys::EncodeOptions {
        truncation: 1,
        truncation_side: 0,
        max_length: 3,
        ..Default::default()
    };
    let result = unsafe { super::common::encode_batch_raw(ctx.handle(), &ptrs, &lengths, &opts) };
    assert!(result.error_msg.is_null(), "batch truncation should succeed");

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.total_tokens) };
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_sequences + 1) };
    assert_eq!(
        ids,
        &[
            byte_token_id(b' '),
            byte_token_id(b'H'),
            byte_token_id(b'e'),
            byte_token_id(b' '),
            byte_token_id(b'H'),
            byte_token_id(b'i')
        ]
    );
    assert_eq!(offsets, &[0, 3, 6]);

    unsafe {
        talu_sys::talu_batch_encode_result_free(
            result.ids,
            result.offsets,
            result.total_tokens,
            result.num_sequences,
        )
    };
}

/// After right truncation, decoding the batch slice should reflect the kept
/// prefix window, not the original untruncated input.
#[test]
fn batch_encode_byte_level_add_prefix_space_truncation_right_decodes_kept_window() {
    let json = build_byte_level_tokenizer_json()
        .replace("\"add_prefix_space\": false", "\"add_prefix_space\": true");
    let ctx = TokenizerTestContext::from_json(&json);
    let t0 = b"Hello";
    let ptrs = [t0.as_ptr()];
    let lengths = [t0.len()];
    let opts = talu_sys::EncodeOptions {
        truncation: 1,
        truncation_side: 0,
        max_length: 3,
        ..Default::default()
    };
    let result = unsafe { super::common::encode_batch_raw(ctx.handle(), &ptrs, &lengths, &opts) };
    assert!(result.error_msg.is_null(), "batch truncation should succeed");

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.total_tokens) };
    assert_eq!(ctx.decode(ids), "He");

    unsafe {
        talu_sys::talu_batch_encode_result_free(
            result.ids,
            result.offsets,
            result.total_tokens,
            result.num_sequences,
        )
    };
}

/// Left truncation must keep the tail of each sequence after prefix insertion,
/// so the synthetic prefix token is dropped for longer non-space-prefixed rows.
#[test]
fn batch_encode_byte_level_add_prefix_space_truncation_left_drops_prefix_token() {
    let json = build_byte_level_tokenizer_json()
        .replace("\"add_prefix_space\": false", "\"add_prefix_space\": true");
    let ctx = TokenizerTestContext::from_json(&json);
    let t0 = b"Hello";
    let t1 = b" Hi";
    let ptrs = [t0.as_ptr(), t1.as_ptr()];
    let lengths = [t0.len(), t1.len()];
    let opts = talu_sys::EncodeOptions {
        truncation: 1,
        truncation_side: 1,
        max_length: 3,
        ..Default::default()
    };
    let result = unsafe { super::common::encode_batch_raw(ctx.handle(), &ptrs, &lengths, &opts) };
    assert!(result.error_msg.is_null(), "batch truncation should succeed");

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.total_tokens) };
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_sequences + 1) };
    assert_eq!(
        ids,
        &[
            byte_token_id(b'l'),
            byte_token_id(b'l'),
            byte_token_id(b'o'),
            byte_token_id(b' '),
            byte_token_id(b'H'),
            byte_token_id(b'i')
        ]
    );
    assert_eq!(offsets, &[0, 3, 6]);

    unsafe {
        talu_sys::talu_batch_encode_result_free(
            result.ids,
            result.offsets,
            result.total_tokens,
            result.num_sequences,
        )
    };
}

/// After left truncation, decoding the batch slice should reflect only the
/// retained tail window.
#[test]
fn batch_encode_byte_level_add_prefix_space_truncation_left_decodes_tail_window() {
    let json = build_byte_level_tokenizer_json()
        .replace("\"add_prefix_space\": false", "\"add_prefix_space\": true");
    let ctx = TokenizerTestContext::from_json(&json);
    let t0 = b"Hello";
    let ptrs = [t0.as_ptr()];
    let lengths = [t0.len()];
    let opts = talu_sys::EncodeOptions {
        truncation: 1,
        truncation_side: 1,
        max_length: 3,
        ..Default::default()
    };
    let result = unsafe { super::common::encode_batch_raw(ctx.handle(), &ptrs, &lengths, &opts) };
    assert!(result.error_msg.is_null(), "batch truncation should succeed");

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.total_tokens) };
    assert_eq!(ctx.decode(ids), "llo");

    unsafe {
        talu_sys::talu_batch_encode_result_free(
            result.ids,
            result.offsets,
            result.total_tokens,
            result.num_sequences,
        )
    };
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
    let batch = ctx.encode_batch(&["\x00", "\u{200b}", "🎉", "日", "م"], &no_bos());

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
    let result = ctx.batch_to_padded_tensor(&["A", "Hi", "abc", "Hello", ""], &no_bos(), &pad_opts);

    assert_eq!(result.num_sequences, 5);
    assert_eq!(result.padded_length, 5); // longest is "Hello"=5

    let pl = result.padded_length;

    // Row 0: "A" = [37, 99, 99, 99, 99], mask [1, 0, 0, 0, 0].
    assert_eq!(result.input_ids[0], 37);
    assert_eq!(result.attention_mask[0], 1);
    for j in 1..pl {
        assert_eq!(result.input_ids[j], 99, "row 0 col {j} should be pad");
        assert_eq!(
            result.attention_mask[j], 0,
            "row 0 col {j} mask should be 0"
        );
    }

    // Row 3: "Hello" = [44,73,80,80,83], all mask=1.
    let start = 3 * pl;
    assert_eq!(&result.input_ids[start..start + 5], &[44, 73, 80, 80, 83]);
    for j in 0..pl {
        assert_eq!(
            result.attention_mask[start + j],
            1,
            "row 3 col {j} mask should be 1"
        );
    }

    // Row 4: empty = all pad, all mask=0.
    let start = 4 * pl;
    for j in 0..pl {
        assert_eq!(
            result.input_ids[start + j],
            99,
            "row 4 col {j} should be pad"
        );
        assert_eq!(
            result.attention_mask[start + j],
            0,
            "row 4 col {j} mask should be 0"
        );
    }
}

/// A highly skewed batch shape must still produce correct flat row strides:
/// one long sequence followed by many empty rows. This targets tail-row stride
/// math rather than ordinary padding semantics.
#[test]
fn padded_tensor_skewed_long_plus_many_empty_sequences() {
    let ctx = TokenizerTestContext::new();
    let long = "a".repeat(2048);
    let mut texts = Vec::with_capacity(257);
    texts.push(long.as_str());
    for _ in 0..256 {
        texts.push("");
    }

    let pad_opts = talu_sys::PaddedTensorOptions {
        pad_id: 99,
        padding_side: 0,
        return_attention_mask: true,
        ..Default::default()
    };

    let result = ctx.batch_to_padded_tensor(&texts, &no_bos(), &pad_opts);
    assert_eq!(result.num_sequences, 257);
    assert_eq!(result.padded_length, 2048);

    let row = result.padded_length;
    assert_eq!(result.input_ids.len(), result.num_sequences * row);
    assert_eq!(result.attention_mask.len(), result.num_sequences * row);

    assert!(
        result.input_ids[0..row].iter().all(|&id| id == 69),
        "first row should contain the long 'a' sequence only"
    );
    assert!(
        result.attention_mask[0..row].iter().all(|&m| m == 1),
        "first row mask should be entirely active"
    );

    for seq in 1..result.num_sequences {
        let start = seq * row;
        let end = start + row;
        assert!(
            result.input_ids[start..end].iter().all(|&id| id == 99),
            "empty row {seq} must be entirely pad_id"
        );
        assert!(
            result.attention_mask[start..end].iter().all(|&m| m == 0),
            "empty row {seq} mask must be entirely zero"
        );
    }
}

/// Padded tensor layout must include the synthetic prefix-space token in the
/// encoded lengths before padding is applied.
#[test]
fn padded_tensor_byte_level_add_prefix_space_uses_prefixed_lengths() {
    let json = build_byte_level_tokenizer_json()
        .replace("\"add_prefix_space\": false", "\"add_prefix_space\": true");
    let ctx = TokenizerTestContext::from_json(&json);
    let pad_opts = talu_sys::PaddedTensorOptions {
        pad_id: 0,
        padding_side: 0,
        return_attention_mask: true,
        ..Default::default()
    };

    let result = ctx.batch_to_padded_tensor(&["Hi", "Hello"], &no_bos(), &pad_opts);
    assert_eq!(result.num_sequences, 2);
    assert_eq!(result.padded_length, 6);

    assert_eq!(
        &result.input_ids[0..6],
        &[
            byte_token_id(b' '),
            byte_token_id(b'H'),
            byte_token_id(b'i'),
            0,
            0,
            0
        ]
    );
    assert_eq!(&result.attention_mask[0..6], &[1, 1, 1, 0, 0, 0]);

    assert_eq!(
        &result.input_ids[6..12],
        &[
            byte_token_id(b' '),
            byte_token_id(b'H'),
            byte_token_id(b'e'),
            byte_token_id(b'l'),
            byte_token_id(b'l'),
            byte_token_id(b'o')
        ]
    );
    assert_eq!(&result.attention_mask[6..12], &[1, 1, 1, 1, 1, 1]);
}

/// In padded-tensor conversion, an empty sequence under `add_prefix_space`
/// must still become an all-padding row rather than a synthetic-prefix row.
#[test]
fn padded_tensor_byte_level_add_prefix_space_empty_sequence_is_all_padding() {
    let json = build_byte_level_tokenizer_json()
        .replace("\"add_prefix_space\": false", "\"add_prefix_space\": true");
    let ctx = TokenizerTestContext::from_json(&json);
    let pad_opts = talu_sys::PaddedTensorOptions {
        pad_id: 0,
        padding_side: 0,
        return_attention_mask: true,
        ..Default::default()
    };

    let result = ctx.batch_to_padded_tensor(&["", "Hi"], &no_bos(), &pad_opts);
    assert_eq!(result.num_sequences, 2);
    assert_eq!(result.padded_length, 3);

    assert_eq!(&result.input_ids[0..3], &[0, 0, 0]);
    assert_eq!(&result.attention_mask[0..3], &[0, 0, 0]);

    assert_eq!(
        &result.input_ids[3..6],
        &[byte_token_id(b' '), byte_token_id(b'H'), byte_token_id(b'i')]
    );
    assert_eq!(&result.attention_mask[3..6], &[1, 1, 1]);
}

/// Left padding must treat real-leading-space and synthetic-prefix-space rows
/// according to their encoded lengths after ByteLevel prefix handling.
#[test]
fn padded_tensor_byte_level_add_prefix_space_left_padding_mixed_real_and_synthetic() {
    let json = build_byte_level_tokenizer_json()
        .replace("\"add_prefix_space\": false", "\"add_prefix_space\": true");
    let ctx = TokenizerTestContext::from_json(&json);
    let pad_opts = talu_sys::PaddedTensorOptions {
        pad_id: 0,
        padding_side: 1,
        return_attention_mask: true,
        ..Default::default()
    };

    let result = ctx.batch_to_padded_tensor(&[" Hi", "A"], &no_bos(), &pad_opts);
    assert_eq!(result.num_sequences, 2);
    assert_eq!(result.padded_length, 3);

    assert_eq!(
        &result.input_ids[0..3],
        &[byte_token_id(b' '), byte_token_id(b'H'), byte_token_id(b'i')]
    );
    assert_eq!(&result.attention_mask[0..3], &[1, 1, 1]);

    assert_eq!(
        &result.input_ids[3..6],
        &[0, byte_token_id(b' '), byte_token_id(b'A')]
    );
    assert_eq!(&result.attention_mask[3..6], &[0, 1, 1]);
}

/// Right truncation in padded-tensor conversion must keep the prefix token
/// because truncation is applied after ByteLevel prefix insertion.
#[test]
fn padded_tensor_byte_level_add_prefix_space_right_truncation_keeps_prefix() {
    let json = build_byte_level_tokenizer_json()
        .replace("\"add_prefix_space\": false", "\"add_prefix_space\": true");
    let ctx = TokenizerTestContext::from_json(&json);
    let pad_opts = talu_sys::PaddedTensorOptions {
        pad_id: 0,
        padding_side: 0,
        max_length: 3,
        truncate: true,
        return_attention_mask: true,
    };

    let result = ctx.batch_to_padded_tensor(&["Hello", " Hi"], &no_bos(), &pad_opts);
    assert_eq!(result.num_sequences, 2);
    assert_eq!(result.padded_length, 3);

    assert_eq!(
        &result.input_ids[0..3],
        &[byte_token_id(b' '), byte_token_id(b'H'), byte_token_id(b'e')]
    );
    assert_eq!(&result.attention_mask[0..3], &[1, 1, 1]);

    assert_eq!(
        &result.input_ids[3..6],
        &[byte_token_id(b' '), byte_token_id(b'H'), byte_token_id(b'i')]
    );
    assert_eq!(&result.attention_mask[3..6], &[1, 1, 1]);
}


// ===========================================================================
// Token utilities
// ===========================================================================

/// tokens_concat joins two token arrays.
#[test]
fn tokens_concat() {
    let a = [44u32, 77]; // "Hi"
    let b = [69u32, 70, 71]; // "abc"

    let result = unsafe { talu_sys::talu_tokens_concat(a.as_ptr(), a.len(), b.as_ptr(), b.len()) };
    assert!(!result.is_null());

    let combined = unsafe { std::slice::from_raw_parts(result, 5) };
    assert_eq!(combined, [44, 77, 69, 70, 71]);

    unsafe { talu_sys::talu_tokens_free(result, 5) };
}

/// tokens_concat must reject null pointer paired with non-zero length.
#[test]
fn tokens_concat_rejects_null_with_nonzero_length() {
    let b = [69u32, 70, 71];
    let result = unsafe { talu_sys::talu_tokens_concat(std::ptr::null(), 1, b.as_ptr(), b.len()) };
    assert!(
        result.is_null(),
        "null tokens pointer with non-zero length must be rejected"
    );
    if !result.is_null() {
        unsafe { talu_sys::talu_tokens_free(result, 4) };
    }
}

/// tokens_concat must reject null second pointer paired with non-zero length.
#[test]
fn tokens_concat_rejects_null_second_with_nonzero_length() {
    let a = [44u32, 77];
    let result = unsafe { talu_sys::talu_tokens_concat(a.as_ptr(), a.len(), std::ptr::null(), 1) };
    assert!(
        result.is_null(),
        "null second tokens pointer with non-zero length must be rejected"
    );
    if !result.is_null() {
        unsafe { talu_sys::talu_tokens_free(result, 3) };
    }
}

/// encode_batch with null options pointer must use C-API default add_special_tokens=true.
#[test]
fn batch_encode_null_options_defaults_to_add_special_tokens() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"H": 4, "i": 5},
    "merges": []
  },
  "added_tokens": [
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": {
    "type": "TemplateProcessing",
    "single": [
      {"SpecialToken": {"id": "<s>", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}},
      {"SpecialToken": {"id": "</s>", "type_id": 0}}
    ],
    "pair": [
      {"SpecialToken": {"id": "<s>", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}},
      {"SpecialToken": {"id": "</s>", "type_id": 0}},
      {"Sequence": {"id": "B", "type_id": 1}},
      {"SpecialToken": {"id": "</s>", "type_id": 0}}
    ],
    "special_tokens": {
      "<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]},
      "</s>": {"id": "</s>", "ids": [2], "tokens": ["</s>"]}
    }
  },
  "decoder": {"type": "ByteLevel"}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let t0 = b"Hi";
    let t1 = b"H";
    let ptrs = [t0.as_ptr(), t1.as_ptr()];
    let lengths = [t0.len(), t1.len()];
    let result = unsafe { super::common::encode_batch_raw_null_options(ctx.handle(), &ptrs, &lengths) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_sequences, 2);
    assert_eq!(result.total_tokens, 7);

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.total_tokens) };
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_sequences + 1) };
    assert_eq!(ids, &[1, 4, 5, 2, 1, 4, 2]);
    assert_eq!(offsets, &[0, 4, 7]);

    unsafe {
        talu_sys::talu_batch_encode_result_free(
            result.ids,
            result.offsets,
            result.total_tokens,
            result.num_sequences,
        )
    };
}

/// batch_to_padded_tensor with null options pointer must use C-API defaults.
#[test]
fn padded_tensor_null_options_uses_defaults() {
    let ctx = TokenizerTestContext::new();
    let t0 = b"A";
    let t1 = b"Hi";
    let ptrs = [t0.as_ptr(), t1.as_ptr()];
    let lengths = [t0.len(), t1.len()];
    let batch = unsafe { super::common::encode_batch_raw(ctx.handle(), &ptrs, &lengths, &no_bos()) };
    assert!(batch.error_msg.is_null());
    assert_eq!(batch.num_sequences, 2);

    let tensor = unsafe {
        super::common::batch_to_padded_tensor_raw_null_options(
            batch.ids,
            batch.offsets,
            batch.num_sequences,
        )
    };
    assert!(tensor.error_msg.is_null());
    assert_eq!(tensor.num_sequences, 2);
    assert_eq!(tensor.padded_length, 2);
    assert!(
        !tensor.attention_mask.is_null(),
        "null options must default return_attention_mask=true"
    );

    let total = tensor.num_sequences * tensor.padded_length;
    let ids = unsafe { std::slice::from_raw_parts(tensor.input_ids, total) };
    let mask = unsafe { std::slice::from_raw_parts(tensor.attention_mask, total) };
    assert_eq!(ids, &[37, 0, 44, 77], "default right padding with pad_id=0");
    assert_eq!(mask, &[1, 0, 1, 1], "default attention mask values");

    unsafe {
        talu_sys::talu_padded_tensor_result_free(
            tensor.input_ids,
            tensor.attention_mask,
            tensor.num_sequences,
            tensor.padded_length,
        );
        talu_sys::talu_batch_encode_result_free(
            batch.ids,
            batch.offsets,
            batch.total_tokens,
            batch.num_sequences,
        );
    }
}

/// Batch encode: add_bos=1 + add_eos=0 must add BOS only.
#[test]
fn batch_encode_add_bos_without_eos() {
    let ctx = TokenizerTestContext::from_json(TEMPLATE_BATCH_JSON);
    let t0 = b"Hi";
    let t1 = b"H";
    let ptrs = [t0.as_ptr(), t1.as_ptr()];
    let lengths = [t0.len(), t1.len()];
    let opts = talu_sys::EncodeOptions {
        add_bos: 1,
        add_eos: 0,
        ..Default::default()
    };
    let result = unsafe { super::common::encode_batch_raw(ctx.handle(), &ptrs, &lengths, &opts) };
    assert!(result.error_msg.is_null());

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.total_tokens) };
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_sequences + 1) };
    assert_eq!(ids, &[1, 4, 5, 1, 4], "add_bos=1/add_eos=0 in batch");
    assert_eq!(offsets, &[0, 3, 5], "offsets for BOS-only batch");

    unsafe {
        talu_sys::talu_batch_encode_result_free(
            result.ids,
            result.offsets,
            result.total_tokens,
            result.num_sequences,
        )
    };
}

/// Batch encode: add_bos=0 + add_eos=1 must add EOS only.
#[test]
fn batch_encode_add_eos_without_bos() {
    let ctx = TokenizerTestContext::from_json(TEMPLATE_BATCH_JSON);
    let t0 = b"Hi";
    let t1 = b"H";
    let ptrs = [t0.as_ptr(), t1.as_ptr()];
    let lengths = [t0.len(), t1.len()];
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        add_eos: 1,
        ..Default::default()
    };
    let result = unsafe { super::common::encode_batch_raw(ctx.handle(), &ptrs, &lengths, &opts) };
    assert!(result.error_msg.is_null());

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.total_tokens) };
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_sequences + 1) };
    assert_eq!(ids, &[4, 5, 2, 4, 2], "add_bos=0/add_eos=1 in batch");
    assert_eq!(offsets, &[0, 3, 5], "offsets for EOS-only batch");

    unsafe {
        talu_sys::talu_batch_encode_result_free(
            result.ids,
            result.offsets,
            result.total_tokens,
            result.num_sequences,
        )
    };
}

/// Batch encode: empty input with add_bos=1/add_eos=0 should produce BOS only.
#[test]
fn batch_encode_empty_add_bos_without_eos() {
    let ctx = TokenizerTestContext::from_json(TEMPLATE_BATCH_JSON);
    let t0 = b"";
    let ptrs = [t0.as_ptr()];
    let lengths = [t0.len()];
    let opts = talu_sys::EncodeOptions {
        add_bos: 1,
        add_eos: 0,
        ..Default::default()
    };
    let result = unsafe { super::common::encode_batch_raw(ctx.handle(), &ptrs, &lengths, &opts) };
    assert!(result.error_msg.is_null());
    let ids = if result.ids.is_null() || result.total_tokens == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(result.ids, result.total_tokens) }.to_vec()
    };
    let offsets = if result.offsets.is_null() || result.num_sequences == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(result.offsets, result.num_sequences + 1) }.to_vec()
    };
    assert_eq!(ids, &[1], "empty batch item should produce BOS only");
    assert_eq!(offsets, &[0, 1], "offsets for single BOS-only sequence");
    unsafe {
        talu_sys::talu_batch_encode_result_free(
            result.ids,
            result.offsets,
            result.total_tokens,
            result.num_sequences,
        )
    };
}

/// Batch encode: empty input with add_bos=0/add_eos=1 should produce EOS only.
#[test]
fn batch_encode_empty_add_eos_without_bos() {
    let ctx = TokenizerTestContext::from_json(TEMPLATE_BATCH_JSON);
    let t0 = b"";
    let ptrs = [t0.as_ptr()];
    let lengths = [t0.len()];
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        add_eos: 1,
        ..Default::default()
    };
    let result = unsafe { super::common::encode_batch_raw(ctx.handle(), &ptrs, &lengths, &opts) };
    assert!(result.error_msg.is_null());
    let ids = if result.ids.is_null() || result.total_tokens == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(result.ids, result.total_tokens) }.to_vec()
    };
    let offsets = if result.offsets.is_null() || result.num_sequences == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(result.offsets, result.num_sequences + 1) }.to_vec()
    };
    assert_eq!(ids, &[2], "empty batch item should produce EOS only");
    assert_eq!(offsets, &[0, 1], "offsets for single EOS-only sequence");
    unsafe {
        talu_sys::talu_batch_encode_result_free(
            result.ids,
            result.offsets,
            result.total_tokens,
            result.num_sequences,
        )
    };
}

/// Padded tensor must reject invalid padding_side values outside {0,1}.
#[test]
fn padded_tensor_rejects_invalid_padding_side() {
    let ids = [44u32, 77];
    let offsets = [0usize, 2usize];
    let opts = talu_sys::PaddedTensorOptions {
        padding_side: 2,
        ..Default::default()
    };
    let result =
        unsafe { super::common::batch_to_padded_tensor_raw(ids.as_ptr(), offsets.as_ptr(), 1, &opts) };
    assert!(
        !result.error_msg.is_null(),
        "invalid padding_side must return an error"
    );
    if result.error_msg.is_null() {
        unsafe {
            talu_sys::talu_padded_tensor_result_free(
                result.input_ids,
                result.attention_mask,
                result.num_sequences,
                result.padded_length,
            )
        };
    }
}

/// Padded tensor must reject offset arrays whose first element is not zero.
#[test]
fn padded_tensor_rejects_offsets_not_starting_at_zero() {
    let ids = [44u32, 77];
    let offsets = [1usize, 2usize];
    let result = unsafe {
        super::common::batch_to_padded_tensor_raw(
            ids.as_ptr(),
            offsets.as_ptr(),
            1,
            &talu_sys::PaddedTensorOptions::default(),
        )
    };
    assert!(
        !result.error_msg.is_null(),
        "offsets must start at 0 for a valid batch encoding"
    );
    if result.error_msg.is_null() {
        unsafe {
            talu_sys::talu_padded_tensor_result_free(
                result.input_ids,
                result.attention_mask,
                result.num_sequences,
                result.padded_length,
            )
        };
    }
}

/// Padded tensor consumes exactly the prefix described by `offsets[num_sequences]`.
///
/// The C ABI does not include a separate `ids_len`, so memory beyond the last
/// declared offset is outside the contract and must be ignored by the callee.
#[test]
fn padded_tensor_ignores_memory_beyond_last_declared_offset() {
    let ids = [44u32, 77u32, 69u32];
    let offsets = [0usize, 1usize, 1usize];
    let result = unsafe {
        super::common::batch_to_padded_tensor_raw_null_options(ids.as_ptr(), offsets.as_ptr(), 2)
    };
    assert!(
        result.error_msg.is_null(),
        "ids beyond offsets[num_sequences] are not part of the ABI contract"
    );

    let total = result.num_sequences * result.padded_length;
    let input_ids = unsafe { std::slice::from_raw_parts(result.input_ids, total) };
    let attention_mask = unsafe { std::slice::from_raw_parts(result.attention_mask, total) };
    assert_eq!(result.num_sequences, 2);
    assert_eq!(result.padded_length, 1);
    assert_eq!(input_ids, &[44, 0], "only the declared prefix should be consumed");
    assert_eq!(attention_mask, &[1, 0]);

    unsafe {
        talu_sys::talu_padded_tensor_result_free(
            result.input_ids,
            result.attention_mask,
            result.num_sequences,
            result.padded_length,
        )
    };
}

/// Left and right padding with truncation must produce exact matrices on mixed empty/content batch.
#[test]
fn padded_tensor_mixed_empty_with_truncation_both_sides() {
    let ctx = TokenizerTestContext::new();
    let encode_opts = no_bos();

    let right_opts = talu_sys::PaddedTensorOptions {
        pad_id: 0,
        padding_side: 0,
        max_length: 3,
        truncate: true,
        return_attention_mask: true,
    };
    let right = ctx.batch_to_padded_tensor(&["Hello", "", "A"], &encode_opts, &right_opts);
    assert_eq!(right.num_sequences, 3);
    assert_eq!(right.padded_length, 3);
    assert_eq!(&right.input_ids[0..3], &[44, 73, 80]); // "Hello" truncated
    assert_eq!(&right.input_ids[3..6], &[0, 0, 0]); // empty
    assert_eq!(&right.input_ids[6..9], &[37, 0, 0]); // "A" right pad
    assert_eq!(&right.attention_mask[0..3], &[1, 1, 1]);
    assert_eq!(&right.attention_mask[3..6], &[0, 0, 0]);
    assert_eq!(&right.attention_mask[6..9], &[1, 0, 0]);

    let left_opts = talu_sys::PaddedTensorOptions {
        pad_id: 0,
        padding_side: 1,
        max_length: 3,
        truncate: true,
        return_attention_mask: true,
    };
    let left = ctx.batch_to_padded_tensor(&["Hello", "", "A"], &encode_opts, &left_opts);
    assert_eq!(left.num_sequences, 3);
    assert_eq!(left.padded_length, 3);
    assert_eq!(&left.input_ids[0..3], &[44, 73, 80]); // truncated sequence unchanged
    assert_eq!(&left.input_ids[3..6], &[0, 0, 0]); // empty
    assert_eq!(&left.input_ids[6..9], &[0, 0, 37]); // "A" left pad
    assert_eq!(&left.attention_mask[0..3], &[1, 1, 1]);
    assert_eq!(&left.attention_mask[3..6], &[0, 0, 0]);
    assert_eq!(&left.attention_mask[6..9], &[0, 0, 1]);
}

/// Large batch encoding should be deterministic across repeated runs.
#[test]
fn batch_encode_huge_determinism() {
    let ctx = TokenizerTestContext::new();
    let mut texts = Vec::new();
    for i in 0..1200usize {
        let len = i % 11;
        let s: String = (0..len)
            .map(|j| (b'a' + ((i + j) % 26) as u8) as char)
            .collect();
        texts.push(s);
    }
    let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    let first = ctx.encode_batch(&refs, &no_bos());
    let second = ctx.encode_batch(&refs, &no_bos());
    assert_eq!(first.num_sequences, second.num_sequences);
    assert_eq!(first.offsets, second.offsets);
    assert_eq!(first.ids, second.ids);
}

fn lcg_next(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    *state
}

/// tokenize_bytes result offsets must be monotonic and terminate at data_len.
#[test]
fn tokenize_bytes_offsets_seeded_invariants() {
    let ctx = TokenizerTestContext::new();
    let mut seed = 0xA11CE5EEDu64;
    for case_idx in 0..220usize {
        let len = (lcg_next(&mut seed) % 80) as usize;
        let mut s = String::with_capacity(len);
        for _ in 0..len {
            let b = 0x20u8 + (lcg_next(&mut seed) % 95) as u8; // printable ascii incl space
            s.push(b as char);
        }
        let result = unsafe {
            talu_sys::talu_tokenizer_tokenize_bytes(ctx.handle(), s.as_bytes().as_ptr(), s.len())
        };
        assert!(result.error_msg.is_null(), "case {case_idx}: tokenize_bytes failed");
        let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens + 1) };
        assert_eq!(offsets[0], 0, "case {case_idx}: offsets[0]");
        assert_eq!(
            *offsets.last().unwrap(),
            result.data_len,
            "case {case_idx}: last offset must equal data_len"
        );
        for w in offsets.windows(2) {
            assert!(w[0] <= w[1], "case {case_idx}: offsets must be monotonic");
        }
        unsafe {
            talu_sys::talu_tokenize_bytes_result_free(
                result.data,
                result.data_len,
                result.offsets,
                result.num_tokens,
            )
        };
    }
}

/// Batch encode with right truncation must truncate each sequence independently.
#[test]
fn batch_encode_truncation_right_applies_per_sequence() {
    let ctx = TokenizerTestContext::new();
    let t0 = b"Hello";
    let t1 = b"abc";
    let ptrs = [t0.as_ptr(), t1.as_ptr()];
    let lengths = [t0.len(), t1.len()];
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        truncation: 1,
        truncation_side: 0, // right
        max_length: 2,
        ..Default::default()
    };
    let result = unsafe { super::common::encode_batch_raw(ctx.handle(), &ptrs, &lengths, &opts) };
    assert!(result.error_msg.is_null(), "batch right truncation should succeed");

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.total_tokens) };
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_sequences + 1) };
    assert_eq!(
        ids,
        &[44, 73, 69, 70],
        "right truncation must keep first 2 tokens of each sequence"
    );
    assert_eq!(offsets, &[0, 2, 4], "offsets must reflect truncated lengths");

    unsafe {
        talu_sys::talu_batch_encode_result_free(
            result.ids,
            result.offsets,
            result.total_tokens,
            result.num_sequences,
        )
    };
}

/// Batch encode with left truncation must keep the last N tokens per sequence.
#[test]
fn batch_encode_truncation_left_applies_per_sequence() {
    let ctx = TokenizerTestContext::new();
    let t0 = b"Hello";
    let t1 = b"abc";
    let ptrs = [t0.as_ptr(), t1.as_ptr()];
    let lengths = [t0.len(), t1.len()];
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        truncation: 1,
        truncation_side: 1, // left
        max_length: 2,
        ..Default::default()
    };
    let result = unsafe { super::common::encode_batch_raw(ctx.handle(), &ptrs, &lengths, &opts) };
    assert!(result.error_msg.is_null(), "batch left truncation should succeed");

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.total_tokens) };
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_sequences + 1) };
    assert_eq!(
        ids,
        &[80, 83, 70, 71],
        "left truncation must keep last 2 tokens of each sequence"
    );
    assert_eq!(offsets, &[0, 2, 4], "offsets must reflect truncated lengths");

    unsafe {
        talu_sys::talu_batch_encode_result_free(
            result.ids,
            result.offsets,
            result.total_tokens,
            result.num_sequences,
        )
    };
}

/// Batch encode with template postprocessor + truncation must cap total sequence length.
#[test]
fn batch_encode_template_truncation_caps_total_length() {
    let ctx = TokenizerTestContext::from_json(TEMPLATE_BATCH_JSON);
    let t0 = b"Hi";
    let t1 = b"H";
    let ptrs = [t0.as_ptr(), t1.as_ptr()];
    let lengths = [t0.len(), t1.len()];
    let opts = talu_sys::EncodeOptions {
        add_bos: 1,
        add_eos: 1,
        truncation: 1,
        truncation_side: 0,
        max_length: 3,
        ..Default::default()
    };
    let result = unsafe { super::common::encode_batch_raw(ctx.handle(), &ptrs, &lengths, &opts) };
    assert!(
        result.error_msg.is_null(),
        "batch template truncation should succeed"
    );

    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.total_tokens) };
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_sequences + 1) };
    assert_eq!(
        ids,
        &[1, 4, 5, 1, 4, 2],
        "max_length=3 must cap each sequence after post-processing"
    );
    assert_eq!(offsets, &[0, 3, 6], "offsets must reflect capped sequence lengths");

    unsafe {
        talu_sys::talu_batch_encode_result_free(
            result.ids,
            result.offsets,
            result.total_tokens,
            result.num_sequences,
        )
    };
}

/// Batch encode must reject truncation_side values outside {0,1}.
#[test]
fn batch_encode_rejects_invalid_truncation_side() {
    let ctx = TokenizerTestContext::new();
    let t0 = b"Hello";
    let ptrs = [t0.as_ptr()];
    let lengths = [t0.len()];
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        truncation: 1,
        truncation_side: 2,
        max_length: 2,
        ..Default::default()
    };
    let result = unsafe { super::common::encode_batch_raw(ctx.handle(), &ptrs, &lengths, &opts) };
    assert!(
        !result.error_msg.is_null(),
        "invalid truncation_side must return an error in batch encode"
    );
    if result.error_msg.is_null() {
        unsafe {
            talu_sys::talu_batch_encode_result_free(
                result.ids,
                result.offsets,
                result.total_tokens,
                result.num_sequences,
            )
        };
    }
}
