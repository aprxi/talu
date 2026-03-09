//! Edge-case input and option combination tests.
//!
//! Tests control characters, special token strings as text, whitespace edge
//! cases, option interactions, batch edge cases, and encode determinism.

use crate::capi::tokenizer::common::{
    build_byte_level_tokenizer_json, byte_token_surface, TokenizerTestContext,
};

fn no_bos() -> talu_sys::EncodeOptions {
    talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    }
}

fn assert_invalid_utf8_vector_maps_to_unk_per_byte(
    ctx: &TokenizerTestContext,
    case_name: &str,
    bytes: &[u8],
) {
    let opts = no_bos();
    let expected_ids = vec![3u32; bytes.len()];

    let encoded = unsafe { super::common::encode_raw(ctx.handle(), bytes, &opts) };
    assert!(
        encoded.error_msg.is_null(),
        "{case_name}: encode must not error for malformed UTF-8"
    );
    let ids = unsafe { std::slice::from_raw_parts(encoded.ids, encoded.num_tokens) };
    let offsets = unsafe { std::slice::from_raw_parts(encoded.offsets, encoded.num_tokens) };
    assert_eq!(
        ids,
        expected_ids.as_slice(),
        "{case_name}: each malformed byte must map to <unk>"
    );
    assert_eq!(
        encoded.num_tokens,
        bytes.len(),
        "{case_name}: malformed UTF-8 must produce one token per byte"
    );
    for (idx, off) in offsets.iter().enumerate() {
        assert_eq!(off.start as usize, idx, "{case_name}: offset[{idx}].start");
        assert_eq!(off.end as usize, idx + 1, "{case_name}: offset[{idx}].end");
    }
    let decoded = ctx.decode(ids);
    assert_eq!(
        decoded,
        "<unk>".repeat(bytes.len()),
        "{case_name}: decode(ids) must retain exact malformed-byte cardinality"
    );
    unsafe { talu_sys::talu_encode_result_free(encoded) };

    let tokenized =
        unsafe { talu_sys::talu_tokenizer_tokenize(ctx.handle(), bytes.as_ptr(), bytes.len()) };
    assert!(
        tokenized.error_msg.is_null(),
        "{case_name}: tokenize must not error for malformed UTF-8"
    );
    assert_eq!(
        tokenized.num_tokens,
        bytes.len(),
        "{case_name}: tokenize must emit one token per malformed byte"
    );
    let ptrs = unsafe {
        std::slice::from_raw_parts(tokenized.tokens as *const *const i8, tokenized.num_tokens)
    };
    let tokens: Vec<String> = ptrs
        .iter()
        .map(|&ptr| {
            unsafe { std::ffi::CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    let expected_token_strings: Vec<String> =
        (0..bytes.len()).map(|_| "<unk>".to_string()).collect();
    assert_eq!(
        tokens, expected_token_strings,
        "{case_name}: tokenize token strings must be one <unk> per malformed byte"
    );
    unsafe { talu_sys::talu_tokenize_result_free(tokenized.tokens, tokenized.num_tokens) };

    let tokenized_bytes = unsafe {
        talu_sys::talu_tokenizer_tokenize_bytes(ctx.handle(), bytes.as_ptr(), bytes.len())
    };
    assert!(
        tokenized_bytes.error_msg.is_null(),
        "{case_name}: tokenize_bytes must not error for malformed UTF-8"
    );
    assert_eq!(
        tokenized_bytes.num_tokens,
        bytes.len(),
        "{case_name}: tokenize_bytes must emit one token per malformed byte"
    );
    let data =
        unsafe { std::slice::from_raw_parts(tokenized_bytes.data, tokenized_bytes.data_len) };
    let byte_offsets = unsafe {
        std::slice::from_raw_parts(tokenized_bytes.offsets, tokenized_bytes.num_tokens + 1)
    };
    let token_slices: Vec<&str> = byte_offsets
        .windows(2)
        .map(|w| std::str::from_utf8(&data[w[0]..w[1]]).unwrap())
        .collect();
    assert_eq!(
        token_slices,
        vec!["<unk>"; bytes.len()],
        "{case_name}: tokenize_bytes slices must be one <unk> per malformed byte"
    );
    unsafe {
        talu_sys::talu_tokenize_bytes_result_free(
            tokenized_bytes.data,
            tokenized_bytes.data_len,
            tokenized_bytes.offsets,
            tokenized_bytes.num_tokens,
        )
    };
}

// ===========================================================================
// Control characters
// ===========================================================================

/// Null byte encodes to unk, no crash.
#[test]
fn encode_null_byte() {
    let ctx = TokenizerTestContext::new();
    assert_eq!(ctx.encode_with("\x00", &no_bos()), [3]);
}

/// Embedded null: "Hi\0Bye" → H, i, unk, B, y, e.
#[test]
fn encode_embedded_null() {
    let ctx = TokenizerTestContext::new();
    // H=44, i=77, \0=unk(3), B=38, y=93, e=73
    assert_eq!(
        ctx.encode_with("Hi\x00Bye", &no_bos()),
        [44, 77, 3, 38, 93, 73]
    );
}

/// Tab encodes to unk (ByteLevel maps 0x09 outside vocab).
#[test]
fn encode_tab() {
    let ctx = TokenizerTestContext::new();
    assert_eq!(ctx.encode_with("\t", &no_bos()), [3]);
}

/// Newline encodes to unk.
#[test]
fn encode_newline() {
    let ctx = TokenizerTestContext::new();
    assert_eq!(ctx.encode_with("\n", &no_bos()), [3]);
}

/// SOH control char (0x01) encodes to unk.
#[test]
fn encode_soh_control_char() {
    let ctx = TokenizerTestContext::new();
    assert_eq!(ctx.encode_with("\x01", &no_bos()), [3]);
}

/// ESC control char (0x1b) encodes to unk.
#[test]
fn encode_esc_control_char() {
    let ctx = TokenizerTestContext::new();
    assert_eq!(ctx.encode_with("\x1b", &no_bos()), [3]);
}

/// Mixed whitespace: "a\tb\nc" → a, unk, b, unk, c.
#[test]
fn encode_mixed_whitespace() {
    let ctx = TokenizerTestContext::new();
    // a=69, \t=3, b=70, \n=3, c=71
    assert_eq!(ctx.encode_with("a\tb\nc", &no_bos()), [69, 3, 70, 3, 71]);
}

/// Malformed UTF-8 must be treated as raw bytes: each byte becomes one `<unk>`
/// token, offsets stay bytewise, and decoding those IDs must surface the exact
/// `<unk>` sequence.
#[test]
fn encode_invalid_utf8_bytes_map_to_unk_per_byte_with_exact_offsets() {
    let ctx = TokenizerTestContext::new();
    let opts = no_bos();
    let bytes: &[u8] = &[0xF0, 0x9F, 0x8E]; // truncated 4-byte UTF-8 sequence

    let result = unsafe { super::common::encode_raw(ctx.handle(), bytes, &opts) };
    assert!(
        result.error_msg.is_null(),
        "invalid UTF-8 must not fail encode in the base fixture"
    );
    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!(ids, &[3, 3, 3], "each malformed byte must map to <unk>");
    assert_eq!(
        result.num_tokens,
        bytes.len(),
        "one token per malformed byte"
    );
    assert_eq!((offsets[0].start, offsets[0].end), (0, 1));
    assert_eq!((offsets[1].start, offsets[1].end), (1, 2));
    assert_eq!((offsets[2].start, offsets[2].end), (2, 3));
    assert_eq!(
        ctx.decode(ids),
        "<unk><unk><unk>",
        "decode must surface the exact number of malformed input bytes"
    );

    unsafe {
        talu_sys::talu_encode_result_free(result);
    }
}

/// `tokenize` must surface one `<unk>` token string per malformed byte.
#[test]
fn tokenize_invalid_utf8_bytes_surface_unk_per_byte() {
    let ctx = TokenizerTestContext::new();
    let bytes: &[u8] = &[0xED, 0xA0, 0x80]; // surrogate-like invalid UTF-8 sequence

    let result =
        unsafe { talu_sys::talu_tokenizer_tokenize(ctx.handle(), bytes.as_ptr(), bytes.len()) };
    assert!(
        result.error_msg.is_null(),
        "invalid UTF-8 must not fail tokenize in the base fixture"
    );
    assert_eq!(
        result.num_tokens,
        bytes.len(),
        "one token per malformed byte"
    );
    let ptrs =
        unsafe { std::slice::from_raw_parts(result.tokens as *const *const i8, result.num_tokens) };
    let tokens: Vec<String> = ptrs
        .iter()
        .map(|&ptr| {
            unsafe { std::ffi::CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    assert_eq!(tokens, vec!["<unk>", "<unk>", "<unk>"]);
    unsafe { talu_sys::talu_tokenize_result_free(result.tokens, result.num_tokens) };
}

/// `tokenize_bytes` must expose one `<unk>` slice per malformed byte, with
/// offsets delimiting the returned token-byte buffer exactly.
#[test]
fn tokenize_bytes_invalid_utf8_surface_unk_per_byte() {
    let ctx = TokenizerTestContext::new();
    let bytes: &[u8] = &[0xE2, 0x82]; // truncated 3-byte UTF-8 sequence

    let result = unsafe {
        talu_sys::talu_tokenizer_tokenize_bytes(ctx.handle(), bytes.as_ptr(), bytes.len())
    };
    assert!(
        result.error_msg.is_null(),
        "invalid UTF-8 must not fail tokenize_bytes in the base fixture"
    );
    assert_eq!(
        result.num_tokens,
        bytes.len(),
        "one token per malformed byte"
    );
    let data = unsafe { std::slice::from_raw_parts(result.data, result.data_len) };
    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens + 1) };
    let tokens: Vec<&str> = offsets
        .windows(2)
        .map(|w| std::str::from_utf8(&data[w[0]..w[1]]).unwrap())
        .collect();
    assert_eq!(tokens, vec!["<unk>", "<unk>"]);
    assert_eq!(
        offsets,
        &[0, 5, 10],
        "offsets must delimit '<unk>' slices exactly"
    );
    unsafe {
        talu_sys::talu_tokenize_bytes_result_free(
            result.data,
            result.data_len,
            result.offsets,
            result.num_tokens,
        );
    }
}

/// Unicode-consortium-style malformed UTF-8 vectors (overlong encodings,
/// surrogate ranges, invalid leading bytes, and out-of-range scalars) must be
/// handled deterministically as one `<unk>` per byte without panics.
#[test]
fn malformed_utf8_matrix_maps_to_unk_per_byte_across_all_surfaces() {
    let ctx = TokenizerTestContext::new();
    let cases: &[(&str, &[u8])] = &[
        ("overlong_2byte_ascii_slash", &[0xC0, 0xAF]),
        ("overlong_3byte_nul", &[0xE0, 0x80, 0x80]),
        ("overlong_4byte_nul", &[0xF0, 0x80, 0x80, 0x80]),
        ("surrogate_high_d800", &[0xED, 0xA0, 0x80]),
        ("surrogate_low_dc00", &[0xED, 0xB0, 0x80]),
        ("out_of_range_u110000", &[0xF4, 0x90, 0x80, 0x80]),
        ("lone_continuation", &[0x80]),
        ("invalid_start_fe", &[0xFE]),
        ("invalid_start_ff", &[0xFF]),
        (
            "mixed_invalid_sequence",
            &[0xF0, 0x9F, 0x8E, 0xC0, 0xAF, 0x80, 0xFF],
        ),
    ];

    for (case_name, bytes) in cases {
        assert_invalid_utf8_vector_maps_to_unk_per_byte(&ctx, case_name, bytes);
    }
}

/// WordPiece with BertNormalizer must not silently drop malformed middle bytes.
/// If encode succeeds, the invalid byte must still be represented by a token
/// span and surfaced as unknown content rather than disappearing.
#[test]
fn wordpiece_bertnormalizer_invalid_utf8_middle_byte_not_silently_dropped() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 200,
    "vocab": {
      "[UNK]": 0,
      "hello": 1,
      "world": 2
    }
  },
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true}
  ],
  "normalizer": {
    "type": "BertNormalizer",
    "clean_text": true,
    "handle_chinese_chars": true,
    "strip_accents": false,
    "lowercase": true
  },
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let bytes = b"Hello \xFF World";
    const INVALID_IDX: usize = 6;

    let result = unsafe { super::common::encode_raw(ctx.handle(), bytes, &no_bos()) };
    if result.error_msg.is_null() {
        let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
        let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
        assert!(
            ids.contains(&0),
            "successful WordPiece/BertNormalizer malformed-UTF8 path must surface unknown content"
        );
        assert!(
            offsets.iter().any(|off| {
                let start = off.start as usize;
                let end = off.end as usize;
                start <= INVALID_IDX && INVALID_IDX < end
            }),
            "invalid middle byte must remain owned by at least one token span (no silent byte drop)"
        );
    }
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// Unigram + Metaspace must also avoid silently dropping malformed middle
/// bytes. If encode succeeds, the invalid byte must remain represented by at
/// least one token span (typically as unknown content) rather than vanishing.
#[test]
fn unigram_invalid_utf8_middle_byte_not_silently_dropped() {
    let json = r####"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "unk_id": 0,
    "vocab": [
      ["<unk>", 0.0],
      ["\u2581hello", -2.0],
      ["\u2581world", -3.0]
    ]
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true},
  "post_processor": null,
  "decoder": {"type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true}
}"####;
    let ctx = TokenizerTestContext::from_json(json);
    let bytes = b"Hello \xFF World";
    const INVALID_IDX: usize = 6;

    let result = unsafe { super::common::encode_raw(ctx.handle(), bytes, &no_bos()) };
    if result.error_msg.is_null() {
        let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
        let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
        let offset_pairs: Vec<(u32, u32)> = offsets.iter().map(|o| (o.start, o.end)).collect();
        eprintln!("DEBUG unigram invalid utf8 ids={ids:?} offsets={offset_pairs:?}");
        assert!(
            ids.contains(&0),
            "successful Unigram malformed-UTF8 path must surface unknown content"
        );
        assert!(
            offsets.iter().any(|off| {
                let start = off.start as usize;
                let end = off.end as usize;
                start <= INVALID_IDX && INVALID_IDX < end
            }),
            "invalid middle byte must remain owned by at least one token span (no silent byte drop)"
        );
    }
    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// In a byte-level BPE model, adjacent malformed UTF-8 bytes must still be
/// eligible for BPE merges when an explicit merge rule exists for their
/// byte-token surfaces.
#[test]
fn byte_level_invalid_utf8_adjacent_bytes_can_merge_when_rule_exists() {
    let mut value: serde_json::Value =
        serde_json::from_str(&build_byte_level_tokenizer_json()).expect("valid fixture json");

    let left = byte_token_surface(0xFF);
    let right = byte_token_surface(0xFE);
    let merged = format!("{left}{right}");
    let merged_id = 4096u32;

    let model = value
        .get_mut("model")
        .and_then(serde_json::Value::as_object_mut)
        .expect("model object must exist");
    let vocab = model
        .get_mut("vocab")
        .and_then(serde_json::Value::as_object_mut)
        .expect("vocab object must exist");
    vocab.insert(merged.clone(), serde_json::json!(merged_id));
    model.insert(
        "merges".to_string(),
        serde_json::json!([format!("{left} {right}")]),
    );

    let json = serde_json::to_string(&value).expect("json serialization must succeed");
    let ctx = TokenizerTestContext::from_json(&json);
    let bytes = [0xFFu8, 0xFEu8];
    let result = unsafe { super::common::encode_raw(ctx.handle(), &bytes, &no_bos()) };
    assert!(result.error_msg.is_null(), "encode must succeed");

    assert_eq!(
        result.num_tokens, 1,
        "adjacent malformed bytes must merge into one token when merge rule exists"
    );
    let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
    assert_eq!(ids, &[merged_id], "merged invalid-byte token ID mismatch");

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!(
        (offsets[0].start, offsets[0].end),
        (0, 2),
        "merged invalid-byte token must span both source bytes"
    );
    unsafe { talu_sys::talu_encode_result_free(result) };
}

// ===========================================================================
// Special token strings as literal text
// ===========================================================================

/// Encoding "<s>" as text produces the BOS special token ID (1),
/// because the added_tokens matcher recognizes it in the input.
#[test]
fn encode_bos_string_as_text() {
    let ctx = TokenizerTestContext::new();
    assert_eq!(ctx.encode_with("<s>", &no_bos()), [1]);
}

/// Encoding "</s>" produces the EOS token ID (2).
#[test]
fn encode_eos_string_as_text() {
    let ctx = TokenizerTestContext::new();
    assert_eq!(ctx.encode_with("</s>", &no_bos()), [2]);
}

/// Encoding "<pad>" produces the PAD token ID (0).
#[test]
fn encode_pad_string_as_text() {
    let ctx = TokenizerTestContext::new();
    assert_eq!(ctx.encode_with("<pad>", &no_bos()), [0]);
}

/// Encoding "<unk>" produces the UNK token ID (3).
#[test]
fn encode_unk_string_as_text() {
    let ctx = TokenizerTestContext::new();
    assert_eq!(ctx.encode_with("<unk>", &no_bos()), [3]);
}

/// Special token embedded in text: "Hi<s>Bye" → H, i, <s>, B, y, e.
#[test]
fn encode_special_token_embedded() {
    let ctx = TokenizerTestContext::new();
    // H=44, i=77, <s>=1, B=38, y=93, e=73
    assert_eq!(
        ctx.encode_with("Hi<s>Bye", &no_bos()),
        [44, 77, 1, 38, 93, 73]
    );
}

// ===========================================================================
// Whitespace edge cases
// ===========================================================================

/// Leading and trailing spaces: " Hi " → unk, H, i, unk.
#[test]
fn encode_leading_trailing_spaces() {
    let ctx = TokenizerTestContext::new();
    // space=3(unk), H=44, i=77, space=3(unk)
    assert_eq!(ctx.encode_with(" Hi ", &no_bos()), [3, 44, 77, 3]);
}

/// Double space: "a  b" → a, unk, unk, b.
#[test]
fn encode_double_space() {
    let ctx = TokenizerTestContext::new();
    assert_eq!(ctx.encode_with("a  b", &no_bos()), [69, 3, 3, 70]);
}

/// Spaces only: "   " → [3, 3, 3].
#[test]
fn encode_spaces_only() {
    let ctx = TokenizerTestContext::new();
    assert_eq!(ctx.encode_with("   ", &no_bos()), [3, 3, 3]);
}

// ===========================================================================
// Option combinations
// ===========================================================================

/// Right truncation combined with add_bos=0 on multi-byte input.
#[test]
fn truncation_right_on_multibyte() {
    let ctx = TokenizerTestContext::new();
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        truncation: 1,
        truncation_side: 0, // right: keep first
        max_length: 3,
        ..Default::default()
    };
    // "café" = [71, 69, 74, 3, 3] → truncate to first 3: [71, 69, 74]
    assert_eq!(ctx.encode_with("café", &opts), [71, 69, 74]);
}

/// Left truncation on multi-byte input.
#[test]
fn truncation_left_on_multibyte() {
    let ctx = TokenizerTestContext::new();
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        truncation: 1,
        truncation_side: 1, // left: keep last
        max_length: 3,
        ..Default::default()
    };
    // "café" = [71, 69, 74, 3, 3] → keep last 3: [74, 3, 3]
    assert_eq!(ctx.encode_with("café", &opts), [74, 3, 3]);
}

// ===========================================================================
// Batch edge cases
// ===========================================================================

/// Batch with mixed empty and non-empty strings.
#[test]
fn batch_mixed_empty_and_content() {
    let ctx = TokenizerTestContext::new();
    let batch = ctx.encode_batch(&["Hi", "", "abc"], &no_bos());

    assert_eq!(batch.num_sequences, 3);
    // "Hi"=2 tokens, ""=0, "abc"=3 → offsets [0, 2, 2, 5]
    assert_eq!(batch.offsets, [0, 2, 2, 5]);
    assert_eq!(batch.ids, [44, 77, 69, 70, 71]);
}

/// Batch with all empty strings.
#[test]
fn batch_all_empty() {
    let ctx = TokenizerTestContext::new();
    let batch = ctx.encode_batch(&["", "", ""], &no_bos());

    assert_eq!(batch.num_sequences, 3);
    assert_eq!(batch.offsets, [0, 0, 0, 0]);
    assert!(batch.ids.is_empty());
}

/// Batch with whitespace-only string (all become unk).
#[test]
fn batch_whitespace_only() {
    let ctx = TokenizerTestContext::new();
    let batch = ctx.encode_batch(&["  ", "Hi"], &no_bos());

    assert_eq!(batch.num_sequences, 2);
    // "  "=2 unk tokens, "Hi"=2 tokens → offsets [0, 2, 4]
    assert_eq!(batch.offsets, [0, 2, 4]);
    assert_eq!(batch.ids, [3, 3, 44, 77]);
}

/// Padded tensor where one sequence is empty.
#[test]
fn padded_tensor_with_empty_sequence() {
    let ctx = TokenizerTestContext::new();
    let pad_opts = talu_sys::PaddedTensorOptions {
        pad_id: 0,
        padding_side: 0, // right
        return_attention_mask: true,
        ..Default::default()
    };

    let result = ctx.batch_to_padded_tensor(&["Hi", ""], &no_bos(), &pad_opts);

    assert_eq!(result.num_sequences, 2);
    assert_eq!(result.padded_length, 2); // longest is 2 tokens

    // Row 0: "Hi" = [44, 77], mask [1, 1].
    assert_eq!(&result.input_ids[0..2], &[44, 77]);
    assert_eq!(&result.attention_mask[0..2], &[1, 1]);

    // Row 1: empty = [0, 0] (all pad), mask [0, 0].
    assert_eq!(&result.input_ids[2..4], &[0, 0]);
    assert_eq!(&result.attention_mask[2..4], &[0, 0]);
}

/// Batch encode with multi-byte text.
#[test]
fn batch_encode_multibyte() {
    let ctx = TokenizerTestContext::new();
    let batch = ctx.encode_batch(&["Hi", "日"], &no_bos());

    assert_eq!(batch.num_sequences, 2);
    // "Hi"=2 tokens, "日"=3 bytes=3 tokens → offsets [0, 2, 5]
    assert_eq!(batch.offsets, [0, 2, 5]);
    assert_eq!(batch.ids, [44, 77, 3, 3, 3]);
}

// ===========================================================================
// Determinism
// ===========================================================================

/// Same text encoded 50 times produces identical results each time.
#[test]
fn encode_determinism_50_repetitions() {
    let ctx = TokenizerTestContext::new();
    let opts = no_bos();
    let expected = ctx.encode_with("Hello world!", &opts);

    for i in 0..50 {
        let actual = ctx.encode_with("Hello world!", &opts);
        assert_eq!(actual, expected, "iteration {i}: non-deterministic encode");
    }
}

/// Batch encode truncation must apply per sequence (parity with single encode).
#[test]
fn batch_encode_applies_truncation_per_sequence() {
    let ctx = TokenizerTestContext::new();
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        truncation: 1,
        truncation_side: 0,
        max_length: 2,
        ..Default::default()
    };
    // "Hello" -> [44,73], "ab" -> [69,70] with max_length=2.
    let batch = ctx.encode_batch(&["Hello", "ab"], &opts);

    assert_eq!(batch.num_sequences, 2);
    assert_eq!(batch.offsets, [0, 2, 4]);
    assert_eq!(batch.ids, [44, 73, 69, 70]);
}

/// Batch encode with invalid UTF-8 byte slices must be deterministic.
#[test]
fn batch_encode_invalid_utf8_bytes_is_deterministic() {
    let ctx = TokenizerTestContext::new();
    let opts = no_bos();

    let bad0: &[u8] = &[0xF0, 0x9F, 0x8E]; // truncated 4-byte sequence
    let bad1: &[u8] = &[0xED, 0xA0, 0x80]; // surrogate-like sequence
    let ptrs = [bad0.as_ptr(), bad1.as_ptr()];
    let lengths = [bad0.len(), bad1.len()];

    let first = unsafe { super::common::encode_batch_raw(ctx.handle(), &ptrs, &lengths, &opts) };
    let second = unsafe { super::common::encode_batch_raw(ctx.handle(), &ptrs, &lengths, &opts) };

    assert_eq!(
        first.error_msg.is_null(),
        second.error_msg.is_null(),
        "invalid UTF-8 batch encode should have deterministic success/failure status"
    );

    if first.error_msg.is_null() {
        let ids_a = if first.ids.is_null() || first.total_tokens == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(first.ids, first.total_tokens) }.to_vec()
        };
        let ids_b = if second.ids.is_null() || second.total_tokens == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(second.ids, second.total_tokens) }.to_vec()
        };
        let off_a = if first.offsets.is_null() || first.num_sequences == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(first.offsets, first.num_sequences + 1) }.to_vec()
        };
        let off_b = if second.offsets.is_null() || second.num_sequences == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(second.offsets, second.num_sequences + 1) }.to_vec()
        };
        assert_eq!(
            ids_a, ids_b,
            "invalid UTF-8 batch IDs must be deterministic"
        );
        assert_eq!(
            off_a, off_b,
            "invalid UTF-8 batch offsets must be deterministic"
        );
    }

    unsafe {
        talu_sys::talu_batch_encode_result_free(
            first.ids,
            first.offsets,
            first.total_tokens,
            first.num_sequences,
        );
        talu_sys::talu_batch_encode_result_free(
            second.ids,
            second.offsets,
            second.total_tokens,
            second.num_sequences,
        );
    }
}

/// tokenize invalid UTF-8 should produce deterministic token strings when successful.
#[test]
fn tokenize_invalid_utf8_token_strings_are_deterministic() {
    let ctx = TokenizerTestContext::new();
    let bytes: &[u8] = &[0xF0, 0x9F, 0x8E]; // truncated sequence

    let first =
        unsafe { talu_sys::talu_tokenizer_tokenize(ctx.handle(), bytes.as_ptr(), bytes.len()) };
    let second =
        unsafe { talu_sys::talu_tokenizer_tokenize(ctx.handle(), bytes.as_ptr(), bytes.len()) };
    assert_eq!(
        first.error_msg.is_null(),
        second.error_msg.is_null(),
        "invalid UTF-8 tokenize should have deterministic success/failure status"
    );

    if first.error_msg.is_null() {
        let ptrs_a = unsafe {
            std::slice::from_raw_parts(first.tokens as *const *const i8, first.num_tokens)
        };
        let ptrs_b = unsafe {
            std::slice::from_raw_parts(second.tokens as *const *const i8, second.num_tokens)
        };
        let toks_a: Vec<String> = ptrs_a
            .iter()
            .map(|p| {
                unsafe { std::ffi::CStr::from_ptr(*p) }
                    .to_string_lossy()
                    .to_string()
            })
            .collect();
        let toks_b: Vec<String> = ptrs_b
            .iter()
            .map(|p| {
                unsafe { std::ffi::CStr::from_ptr(*p) }
                    .to_string_lossy()
                    .to_string()
            })
            .collect();
        assert_eq!(
            toks_a, toks_b,
            "invalid UTF-8 tokenize token strings must be deterministic"
        );
    }

    unsafe {
        talu_sys::talu_tokenize_result_free(first.tokens, first.num_tokens);
        talu_sys::talu_tokenize_result_free(second.tokens, second.num_tokens);
    }
}

/// Decode is deterministic over repeated calls.
#[test]
fn decode_determinism_50_repetitions() {
    let ctx = TokenizerTestContext::new();
    let tokens = [44u32, 73, 80, 80, 83]; // "Hello"
    let expected = ctx.decode(&tokens);

    for i in 0..50 {
        let actual = ctx.decode(&tokens);
        assert_eq!(actual, expected, "iteration {i}: non-deterministic decode");
    }
}
