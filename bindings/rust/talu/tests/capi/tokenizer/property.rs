//! Property-based tokenizer invariants.
//!
//! These tests complement example fixtures by probing random byte inputs and
//! asserting core safety/consistency properties across the C API boundary.

use crate::capi::tokenizer::common::{
    build_byte_level_tokenizer_json, byte_token_id, decode_raw, encode_batch_raw, encode_raw,
    TokenizerTestContext,
};
use proptest::prelude::*;
use std::process::Command;

fn no_bos() -> talu_sys::EncodeOptions {
    talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    }
}

fn byte_level_variant_json_internal(
    lowercase: bool,
    clean_text: bool,
    add_prefix_space: bool,
    use_regex: bool,
    decoder_json: &str,
) -> String {
    let normalizer = format!(
        r#"{{"type":"BertNormalizer","clean_text":{},"handle_chinese_chars":true,"strip_accents":false,"lowercase":{}}}"#,
        if clean_text { "true" } else { "false" },
        if lowercase { "true" } else { "false" }
    );
    let pre = format!(
        r#"{{"type":"ByteLevel","add_prefix_space":{},"trim_offsets":true,"use_regex":{}}}"#,
        if add_prefix_space { "true" } else { "false" },
        if use_regex { "true" } else { "false" }
    );
    build_byte_level_tokenizer_json()
        .replace("\"normalizer\": null,", &format!("\"normalizer\": {normalizer},"))
        .replace(
            "\"pre_tokenizer\": {\"type\": \"ByteLevel\", \"add_prefix_space\": false},",
            &format!("\"pre_tokenizer\": {pre},"),
        )
        .replace(
            "\"decoder\": {\"type\": \"ByteLevel\"}",
            &format!("\"decoder\": {decoder_json}"),
        )
}

fn byte_level_variant_json_with_decoder_mode(
    lowercase: bool,
    clean_text: bool,
    add_prefix_space: bool,
    use_regex: bool,
    decoder_mode: u8,
) -> String {
    let byte_level = format!(
        r#"{{"type":"ByteLevel","add_prefix_space":{},"trim_offsets":true,"use_regex":{}}}"#,
        if add_prefix_space { "true" } else { "false" },
        if use_regex { "true" } else { "false" }
    );
    let decoder = match decoder_mode {
        0 => byte_level.clone(),
        1 => format!(
            r#"{{"type":"Sequence","decoders":[{},{{"type":"Strip","content":" ","start":1,"stop":0}}]}}"#,
            byte_level
        ),
        _ => format!(
            r#"{{"type":"Sequence","decoders":[{{"type":"Strip","content":" ","start":1,"stop":0}},{}]}}"#,
            byte_level
        ),
    };
    byte_level_variant_json_internal(
        lowercase,
        clean_text,
        add_prefix_space,
        use_regex,
        &decoder,
    )
}

fn assert_subprocess_test_ok(test_name: &str, env_key: &str, env_val: &str) {
    if std::env::var_os(env_key).is_some() {
        return;
    }
    let exe = std::env::current_exe().expect("current test executable path must resolve");
    let output = Command::new(exe)
        .arg("--exact")
        .arg(test_name)
        .arg("--nocapture")
        .env(env_key, env_val)
        .output()
        .expect("subprocess launch for property test must succeed");
    assert!(
        output.status.success(),
        "subprocess test {test_name} failed (status: {:?})\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// For byte-level tokenizers, arbitrary raw bytes must produce one token
    /// per byte with exact offsets and stable byte-token IDs.
    #[test]
    fn byte_level_random_bytes_roundtrip_and_offsets(bytes in proptest::collection::vec(any::<u8>(), 0..2048)) {
        let ctx = TokenizerTestContext::with_byte_level();
        let enc = unsafe { encode_raw(ctx.handle(), &bytes, &no_bos()) };
        prop_assert!(enc.error_msg.is_null(), "encode failed on random byte input");

        let ids: Vec<u32> = if enc.ids.is_null() || enc.num_tokens == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(enc.ids, enc.num_tokens) }.to_vec()
        };
        let offsets: Vec<talu_sys::TokenOffset> = if enc.offsets.is_null() || enc.num_tokens == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(enc.offsets, enc.num_tokens) }.to_vec()
        };
        unsafe { talu_sys::talu_encode_result_free(enc) };

        prop_assert_eq!(ids.len(), bytes.len(), "byte-level encode must return one token per byte");
        prop_assert_eq!(offsets.len(), bytes.len(), "offset count must match token count");
        for (idx, (&id, &byte)) in ids.iter().zip(bytes.iter()).enumerate() {
            prop_assert_eq!(id, byte_token_id(byte), "token ID mismatch at byte index {}", idx);
            prop_assert_eq!(
                offsets[idx].start as usize,
                idx,
                "offset start mismatch at index {}",
                idx
            );
            prop_assert_eq!(
                offsets[idx].end as usize,
                idx + 1,
                "offset end mismatch at index {}",
                idx
            );
        }
    }

    /// For valid Unicode text, byte-level encode/decode should roundtrip
    /// exactly.
    #[test]
    #[ignore = "executed via subprocess wrapper to isolate native crashes while preserving failure signal"]
    fn byte_level_valid_unicode_roundtrips_exactly(
        // Must cross the 512-symbol threshold so property runs exercise both
        // BPE word paths (small iterative and large cached-pair).
        text in proptest::string::string_regex("(?s).{0,1024}").expect("regex must compile")
    ) {
        let ctx = TokenizerTestContext::with_byte_level();
        let enc = unsafe { encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
        prop_assert!(enc.error_msg.is_null(), "encode failed on valid unicode text");
        let ids: Vec<u32> = if enc.ids.is_null() || enc.num_tokens == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(enc.ids, enc.num_tokens) }.to_vec()
        };
        unsafe { talu_sys::talu_encode_result_free(enc) };

        let dec = unsafe {
            decode_raw(
                ctx.handle(),
                &ids,
                &talu_sys::DecodeOptionsC {
                    skip_special_tokens: 0,
                },
            )
        };
        prop_assert!(dec.error_msg.is_null(), "decode failed on byte-level IDs from valid text");
        let decoded = if dec.text.is_null() || dec.text_len == 0 {
            String::new()
        } else {
            let bytes = unsafe { std::slice::from_raw_parts(dec.text, dec.text_len) };
            std::str::from_utf8(bytes).expect("decode output must be utf8").to_owned()
        };
        unsafe { talu_sys::talu_decode_result_free(dec.text, dec.text_len) };
        prop_assert_eq!(decoded, text, "decode(encode(text)) must roundtrip valid unicode text");
    }

    /// Byte-level batch encoding must be exactly slice-equivalent to encoding
    /// each sequence independently, even for random heterogeneous byte payloads.
    #[test]
    fn byte_level_batch_random_sequences_match_individual_encodes(
        sequences in proptest::collection::vec(proptest::collection::vec(any::<u8>(), 0..256), 0..10)
    ) {
        let ctx = TokenizerTestContext::with_byte_level();
        let ptrs: Vec<*const u8> = sequences.iter().map(|bytes| bytes.as_ptr()).collect();
        let lengths: Vec<usize> = sequences.iter().map(Vec::len).collect();
        let batch = unsafe { encode_batch_raw(ctx.handle(), &ptrs, &lengths, &no_bos()) };
        prop_assert!(batch.error_msg.is_null(), "batch encode failed on random sequences");
        prop_assert_eq!(batch.num_sequences, sequences.len(), "sequence count mismatch");

        let ids: Vec<u32> = if batch.ids.is_null() || batch.total_tokens == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(batch.ids, batch.total_tokens) }.to_vec()
        };
        let offsets: Vec<usize> = if batch.offsets.is_null() || batch.num_sequences == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(batch.offsets, batch.num_sequences + 1) }.to_vec()
        };

        if !batch.ids.is_null() && batch.total_tokens > 0 {
            unsafe {
                talu_sys::talu_batch_encode_result_free(
                    batch.ids,
                    batch.offsets,
                    batch.total_tokens,
                    batch.num_sequences,
                )
            };
        }

        if sequences.is_empty() {
            prop_assert!(offsets.is_empty(), "empty batch should report no offsets");
            prop_assert!(ids.is_empty(), "empty batch should report no IDs");
        } else {
            prop_assert_eq!(offsets[0], 0, "offsets[0] must be zero");
            prop_assert_eq!(
                *offsets.last().expect("non-empty offsets"),
                ids.len(),
                "last offset must equal total token count"
            );
        }

        for (idx, bytes) in sequences.iter().enumerate() {
            let expected: Vec<u32> = bytes.iter().map(|&b| byte_token_id(b)).collect();
            let actual = if sequences.is_empty() {
                &[][..]
            } else {
                &ids[offsets[idx]..offsets[idx + 1]]
            };
            prop_assert_eq!(
                actual,
                expected.as_slice(),
                "batch slice mismatch for sequence {}",
                idx
            );
        }
    }

    /// Decoding arbitrary token-ID arrays must never crash; it must either
    /// return a valid UTF-8 string or a typed error with empty text output.
    #[test]
    fn decode_random_id_arrays_returns_coherent_result(
        ids in proptest::collection::vec(any::<u32>(), 0..128)
    ) {
        let ctx = TokenizerTestContext::new();
        let dec = unsafe {
            decode_raw(
                ctx.handle(),
                &ids,
                &talu_sys::DecodeOptionsC {
                    skip_special_tokens: 0,
                },
            )
        };

        if dec.error_msg.is_null() {
            if dec.text_len > 0 {
                let bytes = unsafe { std::slice::from_raw_parts(dec.text, dec.text_len) };
                prop_assert!(
                    std::str::from_utf8(bytes).is_ok(),
                    "successful decode must return valid UTF-8 bytes"
                );
            }
        } else {
            prop_assert!(dec.text.is_null(), "error decode must not return text pointer");
            prop_assert_eq!(dec.text_len, 0, "error decode must return zero text length");
        }

        unsafe { talu_sys::talu_decode_result_free(dec.text, dec.text_len) };
    }

    /// WordPiece with Bert normalizer/pretokenizer must keep every offset
    /// directly sliceable on UTF-8 boundaries for arbitrary Unicode input.
    #[test]
    fn wordpiece_random_unicode_offsets_are_sliceable(
        text in proptest::string::string_regex("(?s).{0,128}").expect("regex must compile")
    ) {
        let json = r####"{
  "version": "1.0",
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 64,
    "vocab": {
      "[UNK]": 0,
      "a": 1,
      "b": 2,
      "ab": 3,
      "##b": 4
    }
  },
  "added_tokens": [{"id": 0, "content": "[UNK]", "special": true}],
  "normalizer": {
    "type": "BertNormalizer",
    "clean_text": true,
    "handle_chinese_chars": true,
    "strip_accents": true,
    "lowercase": true
  },
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}
}"####;
        let ctx = TokenizerTestContext::from_json(json);
        let enc = unsafe { encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
        prop_assert!(enc.error_msg.is_null(), "wordpiece encode failed");
        let offsets: Vec<talu_sys::TokenOffset> = if enc.offsets.is_null() || enc.num_tokens == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(enc.offsets, enc.num_tokens) }.to_vec()
        };
        unsafe { talu_sys::talu_encode_result_free(enc) };

        for (idx, off) in offsets.iter().enumerate() {
            let start = off.start as usize;
            let end = off.end as usize;
            prop_assert!(
                start <= end && end <= text.len(),
                "wordpiece offset[{}] out of bounds: ({},{}) vs len {}",
                idx,
                start,
                end,
                text.len()
            );
            prop_assert!(
                text.get(start..end).is_some(),
                "wordpiece offset[{}] must be UTF-8 sliceable: ({},{})",
                idx,
                start,
                end
            );
        }
    }

    /// Unigram + Metaspace must also preserve sliceable UTF-8 offset spans for
    /// arbitrary Unicode input.
    #[test]
    fn unigram_random_unicode_offsets_are_sliceable(
        text in proptest::string::string_regex("(?s).{0,128}").expect("regex must compile")
    ) {
        let json = r####"{
  "version": "1.0",
  "model": {
    "type": "Unigram",
    "unk_id": 0,
    "vocab": [
      ["<unk>", 0.0],
      ["\u2581", -0.5],
      ["a", -1.0],
      ["b", -1.0],
      ["\u2581ab", -0.1]
    ]
  },
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true },
  "post_processor": null,
  "decoder": { "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true }
}"####;
        let ctx = TokenizerTestContext::from_json(json);
        let enc = unsafe { encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
        prop_assert!(enc.error_msg.is_null(), "unigram encode failed");
        let offsets: Vec<talu_sys::TokenOffset> = if enc.offsets.is_null() || enc.num_tokens == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(enc.offsets, enc.num_tokens) }.to_vec()
        };
        unsafe { talu_sys::talu_encode_result_free(enc) };

        for (idx, off) in offsets.iter().enumerate() {
            let start = off.start as usize;
            let end = off.end as usize;
            prop_assert!(
                start <= end && end <= text.len(),
                "unigram offset[{}] out of bounds: ({},{}) vs len {}",
                idx,
                start,
                end,
                text.len()
            );
            prop_assert!(
                text.get(start..end).is_some(),
                "unigram offset[{}] must be UTF-8 sliceable: ({},{})",
                idx,
                start,
                end
            );
        }
    }

    /// Under a compound normalizer chain (Prepend + NFKC + Lowercase +
    /// StripAccents), all offsets must remain within source bounds and on
    /// UTF-8 scalar boundaries for arbitrary valid Unicode inputs.
    #[test]
    fn compound_normalizer_random_unicode_offsets_stay_in_bounds(
        text in proptest::string::string_regex("(?s).{0,128}").expect("regex must compile")
    ) {
        let json = build_byte_level_tokenizer_json().replace(
            "\"normalizer\": null,",
            "\"normalizer\": {\"type\": \"Sequence\", \"normalizers\": [{\"type\": \"Prepend\", \"prepend\": \"X\"}, {\"type\": \"NFKC\"}, {\"type\": \"Lowercase\"}, {\"type\": \"StripAccents\"}]},",
        );
        let ctx = TokenizerTestContext::from_json(&json);
        let enc = unsafe { encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
        prop_assert!(
            enc.error_msg.is_null(),
            "encode failed for compound-normalizer random unicode input"
        );

        let offsets: Vec<talu_sys::TokenOffset> = if enc.offsets.is_null() || enc.num_tokens == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(enc.offsets, enc.num_tokens) }.to_vec()
        };
        unsafe { talu_sys::talu_encode_result_free(enc) };

        for (idx, off) in offsets.iter().enumerate() {
            let start = off.start as usize;
            let end = off.end as usize;
            prop_assert!(
                start <= end && end <= text.len(),
                "offset[{}] out of bounds: ({},{}) vs len {}",
                idx,
                start,
                end,
                text.len()
            );
            prop_assert!(
                text.is_char_boundary(start) && text.is_char_boundary(end),
                "offset[{}] not on UTF-8 boundaries: ({},{})",
                idx,
                start,
                end
            );
        }
    }

}

/// Runs the long-range byte-level roundtrip proptest in a subprocess so native
/// faults (SIGBUS/SIGSEGV) are reported as clean failures without aborting the
/// parent test process.
#[test]
fn byte_level_valid_unicode_roundtrips_exactly_subprocess() {
    let exe = std::env::current_exe().expect("current test executable path must resolve");
    let output = Command::new(exe)
        .arg("--exact")
        .arg("capi::tokenizer::property::byte_level_valid_unicode_roundtrips_exactly")
        .arg("--ignored")
        .arg("--nocapture")
        .output()
        .expect("subprocess launch for long-range roundtrip proptest must succeed");
    assert!(
        output.status.success(),
        "subprocess test capi::tokenizer::property::byte_level_valid_unicode_roundtrips_exactly failed (status: {:?})\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

fn run_byte_level_pipeline_variant_configs_are_deterministic_inner() {
    let texts: &[&[u8]] = &[
        b"",
        b"Hello world",
        "A\u{0301}B".as_bytes(),
        "emoji: \u{1F469}\u{200D}\u{1F4BB}".as_bytes(),
    ];
    for &lowercase in &[false, true] {
        for &clean_text in &[false, true] {
            for &add_prefix_space in &[false, true] {
                for &use_regex in &[false, true] {
                    for decoder_mode in 0..=2u8 {
                        let json = byte_level_variant_json_with_decoder_mode(
                            lowercase,
                            clean_text,
                            add_prefix_space,
                            use_regex,
                            decoder_mode,
                        );
                        let ctx = TokenizerTestContext::from_json(&json);
                        for text in texts {
                            let first = unsafe { encode_raw(ctx.handle(), text, &no_bos()) };
                            let second = unsafe { encode_raw(ctx.handle(), text, &no_bos()) };

                            assert_eq!(
                                first.error_msg.is_null(),
                                second.error_msg.is_null(),
                                "variant encode success/failure must be deterministic for flags lc={lowercase} clean={clean_text} prefix={add_prefix_space} regex={use_regex} dec_mode={decoder_mode}"
                            );

                            if first.error_msg.is_null() {
                                let first_ids: Vec<u32> = if first.ids.is_null() || first.num_tokens == 0 {
                                    Vec::new()
                                } else {
                                    unsafe { std::slice::from_raw_parts(first.ids, first.num_tokens) }.to_vec()
                                };
                                let second_ids: Vec<u32> = if second.ids.is_null() || second.num_tokens == 0 {
                                    Vec::new()
                                } else {
                                    unsafe { std::slice::from_raw_parts(second.ids, second.num_tokens) }.to_vec()
                                };
                                assert_eq!(first_ids, second_ids, "variant IDs must match for deterministic encode");

                                let first_offsets: Vec<(u32, u32)> =
                                    if first.offsets.is_null() || first.num_tokens == 0 {
                                        Vec::new()
                                    } else {
                                        unsafe { std::slice::from_raw_parts(first.offsets, first.num_tokens) }
                                            .iter()
                                            .map(|off| (off.start, off.end))
                                            .collect()
                                    };
                                let second_offsets: Vec<(u32, u32)> =
                                    if second.offsets.is_null() || second.num_tokens == 0 {
                                        Vec::new()
                                    } else {
                                        unsafe { std::slice::from_raw_parts(second.offsets, second.num_tokens) }
                                            .iter()
                                            .map(|off| (off.start, off.end))
                                            .collect()
                                    };
                                assert_eq!(first_offsets, second_offsets, "variant offsets must match for deterministic encode");
                            }

                            unsafe {
                                talu_sys::talu_encode_result_free(first);
                                talu_sys::talu_encode_result_free(second);
                            };
                        }
                    }
                }
            }
        }
    }
}

/// Deterministic matrix over byte-level pipeline variants, including decoder
/// sequence order, must never crash and must produce stable results. Run in a
/// subprocess so native crashes are reported as clean test failures.
#[test]
fn byte_level_pipeline_variant_configs_are_deterministic() {
    const ENV_KEY: &str = "TALU_INNER_BYTE_LEVEL_PIPELINE_VARIANTS";
    if std::env::var_os(ENV_KEY).is_some() {
        run_byte_level_pipeline_variant_configs_are_deterministic_inner();
        return;
    }
    assert_subprocess_test_ok(
        "capi::tokenizer::property::byte_level_pipeline_variant_configs_are_deterministic",
        ENV_KEY,
        "1",
    );
}
