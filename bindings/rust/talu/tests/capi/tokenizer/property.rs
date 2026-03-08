//! Property-based tokenizer invariants.
//!
//! These tests complement example fixtures by probing random byte inputs and
//! asserting core safety/consistency properties across the C API boundary.

use crate::capi::tokenizer::common::{
    build_byte_level_tokenizer_json, byte_token_id, decode_raw, encode_batch_raw, encode_raw,
    TokenizerTestContext,
};
use proptest::prelude::*;
use std::collections::HashMap;
use std::process::Command;

fn no_bos() -> talu_sys::EncodeOptions {
    talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    }
}

fn build_byte_level_tokenizer_json_without_added_tokens() -> String {
    let json = build_byte_level_tokenizer_json();
    let start = json
        .find("\"added_tokens\": [")
        .expect("byte-level fixture must include added_tokens array");
    let tail = &json[start..];
    let end_rel = tail
        .find("],\n  \"normalizer\"")
        .expect("byte-level fixture added_tokens section must precede normalizer");
    let end = start + end_rel + 1; // include trailing ']'

    let mut out = String::with_capacity(json.len());
    out.push_str(&json[..start]);
    out.push_str("\"added_tokens\": []");
    out.push_str(&json[end..]);
    out
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

fn build_wordpiece_overlap_json(
    include_ab: bool,
    include_ba: bool,
    include_aba: bool,
    include_bab: bool,
    max_input_chars_per_word: u32,
) -> String {
    let mut entries: Vec<(&str, u32)> = Vec::new();
    entries.push(("[UNK]", 0));
    entries.push(("a", 1));
    entries.push(("b", 2));
    entries.push(("##a", 3));
    entries.push(("##b", 4));
    let mut next = 5u32;

    if include_ab {
        entries.push(("ab", next));
        next += 1;
        entries.push(("##ab", next));
        next += 1;
    }
    if include_ba {
        entries.push(("ba", next));
        next += 1;
        entries.push(("##ba", next));
        next += 1;
    }
    if include_aba {
        entries.push(("aba", next));
        next += 1;
    }
    if include_bab {
        entries.push(("bab", next));
    }

    let vocab = entries
        .into_iter()
        .map(|(tok, id)| format!(r#""{tok}": {id}"#))
        .collect::<Vec<_>>()
        .join(", ");

    format!(
        r####"{{
  "version": "1.0",
  "model": {{
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": {max_input_chars_per_word},
    "vocab": {{ {vocab} }}
  }},
  "added_tokens": [{{"id": 0, "content": "[UNK]", "special": true}}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": {{"type": "WordPiece", "prefix": "##", "cleanup": false}}
}}"####
    )
}

fn build_unigram_score_variant_json(
    score_a: i16,
    score_b: i16,
    score_ab: i16,
    score_ba: i16,
    include_aba: bool,
    score_aba: i16,
) -> String {
    let mut vocab = vec![
        format!(r#"["<unk>", 0.0]"#),
        format!(r#"["a", {}]"#, (score_a as f32) / 10.0),
        format!(r#"["b", {}]"#, (score_b as f32) / 10.0),
        format!(r#"["ab", {}]"#, (score_ab as f32) / 10.0),
        format!(r#"["ba", {}]"#, (score_ba as f32) / 10.0),
    ];
    if include_aba {
        vocab.push(format!(r#"["aba", {}]"#, (score_aba as f32) / 10.0));
    }
    let vocab_json = vocab.join(", ");

    format!(
        r####"{{
  "version": "1.0",
  "model": {{
    "type": "Unigram",
    "unk_id": 0,
    "vocab": [{vocab_json}]
  }},
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}}"####
    )
}

fn build_bpe_overlap_json(
    include_ab: bool,
    include_ba: bool,
    include_aba: bool,
    include_bab: bool,
    ab_first: bool,
) -> String {
    let mut entries: Vec<(&str, u32)> = Vec::new();
    entries.push(("<unk>", 0));
    entries.push(("a", 1));
    entries.push(("b", 2));
    let mut next = 3u32;

    if include_ab {
        entries.push(("ab", next));
        next += 1;
    }
    if include_ba {
        entries.push(("ba", next));
        next += 1;
    }
    if include_aba && include_ab {
        entries.push(("aba", next));
        next += 1;
    }
    if include_bab && include_ba {
        entries.push(("bab", next));
    }

    let mut merges: Vec<&str> = Vec::new();
    if ab_first {
        if include_ab {
            merges.push("a b");
        }
        if include_ba {
            merges.push("b a");
        }
    } else {
        if include_ba {
            merges.push("b a");
        }
        if include_ab {
            merges.push("a b");
        }
    }
    if include_aba && include_ab {
        merges.push("ab a");
    }
    if include_bab && include_ba {
        merges.push("ba b");
    }

    let vocab = entries
        .into_iter()
        .map(|(tok, id)| format!(r#""{tok}": {id}"#))
        .collect::<Vec<_>>()
        .join(", ");
    let merges_json = merges
        .into_iter()
        .map(|m| format!(r#""{m}""#))
        .collect::<Vec<_>>()
        .join(", ");

    format!(
        r####"{{
  "version": "1.0",
  "model": {{
    "type": "BPE",
    "unk_token": "<unk>",
    "vocab": {{ {vocab} }},
    "merges": [{merges_json}]
  }},
  "added_tokens": [{{"id": 0, "content": "<unk>", "special": true}}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}}"####
    )
}

#[derive(Debug, Clone)]
struct BpeReferenceModel {
    vocab: HashMap<String, u32>,
    merge_ranks: HashMap<(String, String), usize>,
}

fn build_bpe_deep_variant_json(mask: u16, rotate: u8) -> (String, BpeReferenceModel) {
    let mut entries: Vec<(String, u32)> = Vec::new();
    let mut vocab: HashMap<String, u32> = HashMap::new();
    let mut next = 0u32;
    let mut ensure_vocab = |token: &str| {
        if vocab.contains_key(token) {
            return;
        }
        let id = next;
        next += 1;
        entries.push((token.to_owned(), id));
        vocab.insert(token.to_owned(), id);
    };

    for tok in [
        "<unk>", "a", "b", "c", "d", "ab", "bc", "cd", "abc", "bcd", "abcd", "aa", "aab",
    ] {
        ensure_vocab(tok);
    }

    let candidates: [(&str, &str); 11] = [
        ("a", "b"),
        ("b", "c"),
        ("c", "d"),
        ("ab", "c"),
        ("bc", "d"),
        ("abc", "d"),
        ("a", "bc"),
        ("ab", "cd"),
        ("b", "cd"),
        ("a", "a"),
        ("aa", "b"),
    ];

    // Merge products are not guaranteed to be present in vocab in real BPE
    // JSONs, but this parity model expects stable token IDs for all products.
    for (lhs, rhs) in candidates {
        let merged = format!("{lhs}{rhs}");
        ensure_vocab(&merged);
    }

    let mut selected: Vec<usize> = (0..candidates.len())
        .filter(|&idx| ((mask >> idx) & 1) == 1)
        .collect();
    if selected.is_empty() {
        selected.push(0);
    }
    let rot = (rotate as usize) % selected.len();
    selected.rotate_left(rot);

    let mut merges: Vec<(String, String)> = Vec::with_capacity(selected.len());
    let mut merge_ranks: HashMap<(String, String), usize> = HashMap::new();
    for (rank, idx) in selected.into_iter().enumerate() {
        let (lhs, rhs) = candidates[idx];
        merges.push((lhs.to_owned(), rhs.to_owned()));
        merge_ranks.insert((lhs.to_owned(), rhs.to_owned()), rank);
    }

    let vocab_json = entries
        .iter()
        .map(|(tok, id)| format!(r#""{tok}": {id}"#))
        .collect::<Vec<_>>()
        .join(", ");
    let merges_json = merges
        .iter()
        .map(|(lhs, rhs)| format!(r#""{lhs} {rhs}""#))
        .collect::<Vec<_>>()
        .join(", ");
    let unk_id = *vocab
        .get("<unk>")
        .expect("deep bpe model must include <unk> token");

    let json = format!(
        r####"{{
  "version": "1.0",
  "model": {{
    "type": "BPE",
    "unk_token": "<unk>",
    "vocab": {{ {vocab_json} }},
    "merges": [{merges_json}]
  }},
  "added_tokens": [{{"id": {unk_id}, "content": "<unk>", "special": true}}],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}}"####
    );

    (json, BpeReferenceModel { vocab, merge_ranks })
}

fn reference_bpe_encode_ascii(
    text: &str,
    model: &BpeReferenceModel,
) -> (Vec<u32>, Vec<(u32, u32)>) {
    let mut symbols: Vec<String> = text.chars().map(|c| c.to_string()).collect();
    let mut starts: Vec<usize> = Vec::with_capacity(symbols.len());
    let mut ends: Vec<usize> = Vec::with_capacity(symbols.len());

    let mut cursor = 0usize;
    for ch in text.chars() {
        starts.push(cursor);
        cursor += ch.len_utf8();
        ends.push(cursor);
    }

    while symbols.len() >= 2 {
        let mut best_idx: Option<usize> = None;
        let mut best_rank = usize::MAX;
        for i in 0..(symbols.len() - 1) {
            if let Some(&rank) = model
                .merge_ranks
                .get(&(symbols[i].clone(), symbols[i + 1].clone()))
            {
                if best_idx.is_none() || rank < best_rank {
                    best_idx = Some(i);
                    best_rank = rank;
                }
            }
        }

        let Some(i) = best_idx else { break };
        let rhs = symbols.remove(i + 1);
        symbols[i].push_str(&rhs);
        let merged_end = ends.remove(i + 1);
        starts.remove(i + 1);
        ends[i] = merged_end;
    }

    let unk_id = *model
        .vocab
        .get("<unk>")
        .expect("reference model must include <unk>");
    let ids: Vec<u32> = symbols
        .iter()
        .map(|sym| model.vocab.get(sym).copied().unwrap_or(unk_id))
        .collect();
    let offsets: Vec<(u32, u32)> = starts
        .iter()
        .zip(ends.iter())
        .map(|(s, e)| (*s as u32, *e as u32))
        .collect();
    (ids, offsets)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// For byte-level tokenizers, arbitrary raw bytes must produce one token
    /// per byte with exact offsets and stable byte-token IDs.
    #[test]
    fn byte_level_random_bytes_roundtrip_and_offsets(bytes in proptest::collection::vec(any::<u8>(), 0..2048)) {
        // Remove added_tokens so special-token content (e.g. "<s>") in random
        // raw bytes cannot preempt byte-level IDs and violate 1-byte->1-token.
        let ctx = TokenizerTestContext::from_json(
            &build_byte_level_tokenizer_json_without_added_tokens()
        );
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

    /// WordPiece greedy matching must stay deterministic and offset-safe across
    /// randomized overlapping-vocab variants, not just a single static model.
    #[test]
    #[ignore = "executed via subprocess wrapper to isolate native crashes while preserving failure signal"]
    fn wordpiece_random_model_variants_are_deterministic_and_offset_safe(
        include_ab in any::<bool>(),
        include_ba in any::<bool>(),
        include_aba in any::<bool>(),
        include_bab in any::<bool>(),
        max_chars in 1u32..64,
        text in proptest::string::string_regex("[ab]{0,192}").expect("regex must compile")
    ) {
        let json = build_wordpiece_overlap_json(
            include_ab,
            include_ba,
            include_aba,
            include_bab,
            max_chars,
        );
        let ctx = TokenizerTestContext::from_json(&json);

        let first = unsafe { encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
        let second = unsafe { encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
        prop_assert!(first.error_msg.is_null(), "wordpiece first encode failed");
        prop_assert!(second.error_msg.is_null(), "wordpiece second encode failed");

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
        let first_offsets: Vec<(u32, u32)> = if first.offsets.is_null() || first.num_tokens == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(first.offsets, first.num_tokens) }
                .iter()
                .map(|off| (off.start, off.end))
                .collect()
        };
        let second_offsets: Vec<(u32, u32)> = if second.offsets.is_null() || second.num_tokens == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(second.offsets, second.num_tokens) }
                .iter()
                .map(|off| (off.start, off.end))
                .collect()
        };
        unsafe {
            talu_sys::talu_encode_result_free(first);
            talu_sys::talu_encode_result_free(second);
        };

        prop_assert_eq!(
            first_ids.as_slice(),
            second_ids.as_slice(),
            "wordpiece IDs must be deterministic"
        );
        prop_assert_eq!(
            first_offsets.as_slice(),
            second_offsets.as_slice(),
            "wordpiece offsets must be deterministic"
        );

        let dec = unsafe {
            decode_raw(
                ctx.handle(),
                &first_ids,
                &talu_sys::DecodeOptionsC {
                    skip_special_tokens: 0,
                },
            )
        };
        prop_assert!(
            dec.error_msg.is_null(),
            "wordpiece decode must succeed for encode-produced IDs"
        );
        let decoded = if dec.text.is_null() || dec.text_len == 0 {
            String::new()
        } else {
            let bytes = unsafe { std::slice::from_raw_parts(dec.text, dec.text_len) };
            std::str::from_utf8(bytes)
                .expect("wordpiece decode output must be UTF-8")
                .to_owned()
        };
        unsafe { talu_sys::talu_decode_result_free(dec.text, dec.text_len) };
        if text.chars().count() <= max_chars as usize {
            prop_assert_eq!(
                decoded.as_str(),
                text.as_str(),
                "wordpiece decode(encode(text)) must roundtrip when word length is within max_input_chars_per_word"
            );
        }

        let mut prev_end = 0usize;
        for (idx, &(start_u32, end_u32)) in first_offsets.iter().enumerate() {
            let start = start_u32 as usize;
            let end = end_u32 as usize;
            prop_assert!(
                start <= end && end <= text.len(),
                "wordpiece random-model offset[{}] out of bounds: ({},{}) vs len {}",
                idx, start, end, text.len()
            );
            prop_assert_eq!(
                start, prev_end,
                "wordpiece random-model offsets must be contiguous"
            );
            prop_assert!(
                text.get(start..end).is_some(),
                "wordpiece random-model offset[{}] must be UTF-8 sliceable",
                idx
            );
            prev_end = end;
        }
        prop_assert_eq!(
            prev_end,
            text.len(),
            "wordpiece random-model offsets must fully cover source"
        );

        let mut decoded_piece_by_id: HashMap<u32, String> = HashMap::new();
        for (idx, (&id, &(start_u32, end_u32))) in first_ids.iter().zip(first_offsets.iter()).enumerate() {
            let start = start_u32 as usize;
            let end = end_u32 as usize;
            if start == end {
                continue;
            }
            let span = &text[start..end];
            let piece = if let Some(existing) = decoded_piece_by_id.get(&id) {
                existing.clone()
            } else {
                let one = unsafe {
                    decode_raw(
                        ctx.handle(),
                        &[id],
                        &talu_sys::DecodeOptionsC {
                            skip_special_tokens: 0,
                        },
                    )
                };
                prop_assert!(one.error_msg.is_null(), "single-token decode must succeed for id {id}");
                let s = if one.text.is_null() || one.text_len == 0 {
                    String::new()
                } else {
                    let bytes = unsafe { std::slice::from_raw_parts(one.text, one.text_len) };
                    std::str::from_utf8(bytes)
                        .expect("single-token decode output must be UTF-8")
                        .to_owned()
                };
                unsafe { talu_sys::talu_decode_result_free(one.text, one.text_len) };
                decoded_piece_by_id.insert(id, s.clone());
                s
            };

            if piece != "[UNK]" {
                let semantic_piece = piece.strip_prefix("##").unwrap_or(&piece);
                prop_assert_eq!(
                    semantic_piece,
                    span,
                    "wordpiece semantic offset mismatch at token {}: id={}, span={:?}, decoded_piece={:?}",
                    idx,
                    id,
                    span,
                    piece
                );
            }
        }
    }

    /// Unigram Viterbi segmentation must stay deterministic and offset-safe
    /// across randomized score-model variants.
    #[test]
    #[ignore = "executed via subprocess wrapper to isolate native crashes while preserving failure signal"]
    fn unigram_random_score_models_are_deterministic_and_offset_safe(
        score_a in -90i16..=0,
        score_b in -90i16..=0,
        score_ab in -90i16..=0,
        score_ba in -90i16..=0,
        include_aba in any::<bool>(),
        score_aba in -90i16..=0,
        text in proptest::string::string_regex("[ab]{0,192}").expect("regex must compile")
    ) {
        let json = build_unigram_score_variant_json(
            score_a,
            score_b,
            score_ab,
            score_ba,
            include_aba,
            score_aba,
        );
        let ctx = TokenizerTestContext::from_json(&json);

        let first = unsafe { encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
        let second = unsafe { encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
        prop_assert!(first.error_msg.is_null(), "unigram first encode failed");
        prop_assert!(second.error_msg.is_null(), "unigram second encode failed");

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
        let first_offsets: Vec<(u32, u32)> = if first.offsets.is_null() || first.num_tokens == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(first.offsets, first.num_tokens) }
                .iter()
                .map(|off| (off.start, off.end))
                .collect()
        };
        let second_offsets: Vec<(u32, u32)> = if second.offsets.is_null() || second.num_tokens == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(second.offsets, second.num_tokens) }
                .iter()
                .map(|off| (off.start, off.end))
                .collect()
        };
        unsafe {
            talu_sys::talu_encode_result_free(first);
            talu_sys::talu_encode_result_free(second);
        };

        prop_assert_eq!(
            first_ids.as_slice(),
            second_ids.as_slice(),
            "unigram IDs must be deterministic"
        );
        prop_assert_eq!(
            first_offsets.as_slice(),
            second_offsets.as_slice(),
            "unigram offsets must be deterministic"
        );

        let dec = unsafe {
            decode_raw(
                ctx.handle(),
                &first_ids,
                &talu_sys::DecodeOptionsC {
                    skip_special_tokens: 0,
                },
            )
        };
        prop_assert!(
            dec.error_msg.is_null(),
            "unigram decode must succeed for encode-produced IDs"
        );
        let decoded = if dec.text.is_null() || dec.text_len == 0 {
            String::new()
        } else {
            let bytes = unsafe { std::slice::from_raw_parts(dec.text, dec.text_len) };
            std::str::from_utf8(bytes)
                .expect("unigram decode output must be UTF-8")
                .to_owned()
        };
        unsafe { talu_sys::talu_decode_result_free(dec.text, dec.text_len) };
        prop_assert_eq!(
            decoded.as_str(),
            text.as_str(),
            "unigram decode(encode(text)) must roundtrip for generated score models"
        );

        let mut prev_end = 0usize;
        for (idx, &(start_u32, end_u32)) in first_offsets.iter().enumerate() {
            let start = start_u32 as usize;
            let end = end_u32 as usize;
            prop_assert!(
                start <= end && end <= text.len(),
                "unigram random-model offset[{}] out of bounds: ({},{}) vs len {}",
                idx, start, end, text.len()
            );
            prop_assert_eq!(
                start, prev_end,
                "unigram random-model offsets must be contiguous"
            );
            prop_assert!(
                text.get(start..end).is_some(),
                "unigram random-model offset[{}] must be UTF-8 sliceable",
                idx
            );
            prev_end = end;
        }
        prop_assert_eq!(
            prev_end,
            text.len(),
            "unigram random-model offsets must fully cover source"
        );

        let mut decoded_piece_by_id: HashMap<u32, String> = HashMap::new();
        for (idx, (&id, &(start_u32, end_u32))) in first_ids.iter().zip(first_offsets.iter()).enumerate() {
            let start = start_u32 as usize;
            let end = end_u32 as usize;
            if start == end {
                continue;
            }
            let span = &text[start..end];
            let piece = if let Some(existing) = decoded_piece_by_id.get(&id) {
                existing.clone()
            } else {
                let one = unsafe {
                    decode_raw(
                        ctx.handle(),
                        &[id],
                        &talu_sys::DecodeOptionsC {
                            skip_special_tokens: 0,
                        },
                    )
                };
                prop_assert!(one.error_msg.is_null(), "single-token decode must succeed for id {id}");
                let s = if one.text.is_null() || one.text_len == 0 {
                    String::new()
                } else {
                    let bytes = unsafe { std::slice::from_raw_parts(one.text, one.text_len) };
                    std::str::from_utf8(bytes)
                        .expect("single-token decode output must be UTF-8")
                        .to_owned()
                };
                unsafe { talu_sys::talu_decode_result_free(one.text, one.text_len) };
                decoded_piece_by_id.insert(id, s.clone());
                s
            };

            if piece != "<unk>" {
                prop_assert_eq!(
                    piece.as_str(),
                    span,
                    "unigram semantic offset mismatch at token {}: id={}, span={:?}, decoded_piece={:?}",
                    idx,
                    id,
                    span,
                    piece
                );
            }
        }
    }

    /// BPE merge behavior must remain deterministic and offset-semantically
    /// correct across randomized overlapping merge-graph variants.
    #[test]
    #[ignore = "executed via subprocess wrapper to isolate native crashes while preserving failure signal"]
    fn bpe_random_model_variants_are_deterministic_and_offset_safe(
        include_ab in any::<bool>(),
        include_ba in any::<bool>(),
        include_aba in any::<bool>(),
        include_bab in any::<bool>(),
        ab_first in any::<bool>(),
        text in proptest::string::string_regex("[ab]{0,256}").expect("regex must compile")
    ) {
        let json = build_bpe_overlap_json(include_ab, include_ba, include_aba, include_bab, ab_first);
        let ctx = TokenizerTestContext::from_json(&json);

        let first = unsafe { encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
        let second = unsafe { encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
        prop_assert!(first.error_msg.is_null(), "bpe first encode failed");
        prop_assert!(second.error_msg.is_null(), "bpe second encode failed");

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
        let first_offsets: Vec<(u32, u32)> = if first.offsets.is_null() || first.num_tokens == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(first.offsets, first.num_tokens) }
                .iter()
                .map(|off| (off.start, off.end))
                .collect()
        };
        let second_offsets: Vec<(u32, u32)> = if second.offsets.is_null() || second.num_tokens == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(second.offsets, second.num_tokens) }
                .iter()
                .map(|off| (off.start, off.end))
                .collect()
        };
        unsafe {
            talu_sys::talu_encode_result_free(first);
            talu_sys::talu_encode_result_free(second);
        };

        prop_assert_eq!(first_ids.as_slice(), second_ids.as_slice(), "bpe IDs must be deterministic");
        prop_assert_eq!(
            first_offsets.as_slice(),
            second_offsets.as_slice(),
            "bpe offsets must be deterministic"
        );

        let dec = unsafe {
            decode_raw(
                ctx.handle(),
                &first_ids,
                &talu_sys::DecodeOptionsC {
                    skip_special_tokens: 0,
                },
            )
        };
        prop_assert!(dec.error_msg.is_null(), "bpe decode must succeed for encode-produced IDs");
        let decoded = if dec.text.is_null() || dec.text_len == 0 {
            String::new()
        } else {
            let bytes = unsafe { std::slice::from_raw_parts(dec.text, dec.text_len) };
            std::str::from_utf8(bytes)
                .expect("bpe decode output must be UTF-8")
                .to_owned()
        };
        unsafe { talu_sys::talu_decode_result_free(dec.text, dec.text_len) };
        prop_assert_eq!(decoded.as_str(), text.as_str(), "bpe decode(encode(text)) must roundtrip");

        let mut prev_end = 0usize;
        for (idx, &(start_u32, end_u32)) in first_offsets.iter().enumerate() {
            let start = start_u32 as usize;
            let end = end_u32 as usize;
            prop_assert!(
                start <= end && end <= text.len(),
                "bpe random-model offset[{}] out of bounds: ({},{}) vs len {}",
                idx, start, end, text.len()
            );
            prop_assert_eq!(start, prev_end, "bpe random-model offsets must be contiguous");
            prop_assert!(
                text.get(start..end).is_some(),
                "bpe random-model offset[{}] must be UTF-8 sliceable",
                idx
            );
            prev_end = end;
        }
        prop_assert_eq!(prev_end, text.len(), "bpe random-model offsets must fully cover source");

        let mut decoded_piece_by_id: HashMap<u32, String> = HashMap::new();
        for (idx, (&id, &(start_u32, end_u32))) in first_ids.iter().zip(first_offsets.iter()).enumerate() {
            let start = start_u32 as usize;
            let end = end_u32 as usize;
            if start == end {
                continue;
            }
            let span = &text[start..end];
            let piece = if let Some(existing) = decoded_piece_by_id.get(&id) {
                existing.clone()
            } else {
                let one = unsafe {
                    decode_raw(
                        ctx.handle(),
                        &[id],
                        &talu_sys::DecodeOptionsC {
                            skip_special_tokens: 0,
                        },
                    )
                };
                prop_assert!(one.error_msg.is_null(), "single-token decode must succeed for id {id}");
                let s = if one.text.is_null() || one.text_len == 0 {
                    String::new()
                } else {
                    let bytes = unsafe { std::slice::from_raw_parts(one.text, one.text_len) };
                    std::str::from_utf8(bytes)
                        .expect("single-token decode output must be UTF-8")
                        .to_owned()
                };
                unsafe { talu_sys::talu_decode_result_free(one.text, one.text_len) };
                decoded_piece_by_id.insert(id, s.clone());
                s
            };

            if piece != "<unk>" {
                prop_assert_eq!(
                    piece.as_str(),
                    span,
                    "bpe semantic offset mismatch at token {}: id={}, span={:?}, decoded_piece={:?}",
                    idx,
                    id,
                    span,
                    piece
                );
            }
        }
    }

    /// Deep BPE merge-graph variants must exactly match an independent
    /// leftmost-min-rank reference implementation for IDs and offsets.
    #[test]
    #[ignore = "executed via subprocess wrapper to isolate native crashes while preserving failure signal"]
    fn bpe_deep_merge_graph_matches_reference(
        mask in 1u16..2048u16,
        rotate in any::<u8>(),
        text in proptest::string::string_regex("[abcd]{0,128}").expect("regex must compile")
    ) {
        let (json, reference_model) = build_bpe_deep_variant_json(mask, rotate);
        let ctx = TokenizerTestContext::from_json(&json);
        let enc = unsafe { encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
        prop_assert!(enc.error_msg.is_null(), "deep bpe encode failed");

        let ids: Vec<u32> = if enc.ids.is_null() || enc.num_tokens == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(enc.ids, enc.num_tokens) }.to_vec()
        };
        let offsets: Vec<(u32, u32)> = if enc.offsets.is_null() || enc.num_tokens == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(enc.offsets, enc.num_tokens) }
                .iter()
                .map(|off| (off.start, off.end))
                .collect()
        };
        unsafe { talu_sys::talu_encode_result_free(enc) };

        let (expected_ids, expected_offsets) = reference_bpe_encode_ascii(&text, &reference_model);
        prop_assert_eq!(
            ids.as_slice(),
            expected_ids.as_slice(),
            "deep bpe reference ID mismatch"
        );
        prop_assert_eq!(
            offsets.as_slice(),
            expected_offsets.as_slice(),
            "deep bpe reference offset mismatch"
        );

        let dec = unsafe {
            decode_raw(
                ctx.handle(),
                &ids,
                &talu_sys::DecodeOptionsC {
                    skip_special_tokens: 0,
                },
            )
        };
        prop_assert!(dec.error_msg.is_null(), "deep bpe decode failed");
        let decoded = if dec.text.is_null() || dec.text_len == 0 {
            String::new()
        } else {
            let bytes = unsafe { std::slice::from_raw_parts(dec.text, dec.text_len) };
            std::str::from_utf8(bytes)
                .expect("deep bpe decode output must be UTF-8")
                .to_owned()
        };
        unsafe { talu_sys::talu_decode_result_free(dec.text, dec.text_len) };
        prop_assert_eq!(
            decoded.as_str(),
            text.as_str(),
            "deep bpe decode(encode(text)) must roundtrip"
        );
    }

}

fn run_bpe_heap_path_reference_parity_inner() {
    // Enable all candidate merges with a non-zero rotation so rank ordering is
    // not fixed to builder insertion order.
    let (json, reference_model) = build_bpe_deep_variant_json(0x07FF, 3);
    let ctx = TokenizerTestContext::from_json(&json);

    // 2048 symbols (>512) forces heap-symbol path, while mixed pattern
    // exercises non-uniform merge/cache updates.
    let pattern = "abcdabca";
    let repeats = 2048 / pattern.len();
    let mut text = pattern.repeat(repeats);
    while text.len() < 2048 {
        text.push('d');
    }

    let enc = unsafe { encode_raw(ctx.handle(), text.as_bytes(), &no_bos()) };
    assert!(
        enc.error_msg.is_null(),
        "heap-path reference encode must succeed"
    );

    let ids: Vec<u32> = if enc.ids.is_null() || enc.num_tokens == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(enc.ids, enc.num_tokens) }.to_vec()
    };
    let offsets: Vec<(u32, u32)> = if enc.offsets.is_null() || enc.num_tokens == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(enc.offsets, enc.num_tokens) }
            .iter()
            .map(|off| (off.start, off.end))
            .collect()
    };
    unsafe { talu_sys::talu_encode_result_free(enc) };

    let (expected_ids, expected_offsets) = reference_bpe_encode_ascii(&text, &reference_model);
    assert_eq!(
        ids.as_slice(),
        expected_ids.as_slice(),
        "heap-path BPE IDs must match reference implementation"
    );
    assert_eq!(
        offsets.as_slice(),
        expected_offsets.as_slice(),
        "heap-path BPE offsets must match reference implementation"
    );
}

/// Heap-symbol BPE path (>512 symbols) must remain semantically identical to a
/// naive reference implementation on a heterogeneous long word.
#[test]
fn bpe_heap_path_reference_parity_2048_symbols_subprocess() {
    const ENV_KEY: &str = "TALU_INNER_BPE_HEAP_REFERENCE";
    if std::env::var_os(ENV_KEY).is_some() {
        run_bpe_heap_path_reference_parity_inner();
        return;
    }
    assert_subprocess_test_ok(
        "capi::tokenizer::property::bpe_heap_path_reference_parity_2048_symbols_subprocess",
        ENV_KEY,
        "1",
    );
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

/// Runs the randomized WordPiece model-variant proptest in a subprocess so
/// native faults are reported as clean failures without aborting the parent.
#[test]
fn wordpiece_random_model_variants_are_deterministic_and_offset_safe_subprocess() {
    let exe = std::env::current_exe().expect("current test executable path must resolve");
    let output = Command::new(exe)
        .arg("--exact")
        .arg("capi::tokenizer::property::wordpiece_random_model_variants_are_deterministic_and_offset_safe")
        .arg("--ignored")
        .arg("--nocapture")
        .env("PROPTEST_RNG_SEED", "2004318071")
        .env("PROPTEST_DISABLE_FAILURE_PERSISTENCE", "1")
        .output()
        .expect("subprocess launch for wordpiece model-variant proptest must succeed");
    assert!(
        output.status.success(),
        "subprocess test capi::tokenizer::property::wordpiece_random_model_variants_are_deterministic_and_offset_safe failed (status: {:?})\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

/// Runs the randomized Unigram score-model proptest in a subprocess so native
/// faults are reported as clean failures without aborting the parent.
#[test]
fn unigram_random_score_models_are_deterministic_and_offset_safe_subprocess() {
    let exe = std::env::current_exe().expect("current test executable path must resolve");
    let output = Command::new(exe)
        .arg("--exact")
        .arg("capi::tokenizer::property::unigram_random_score_models_are_deterministic_and_offset_safe")
        .arg("--ignored")
        .arg("--nocapture")
        .env("PROPTEST_RNG_SEED", "2004318071")
        .env("PROPTEST_DISABLE_FAILURE_PERSISTENCE", "1")
        .output()
        .expect("subprocess launch for unigram score-model proptest must succeed");
    assert!(
        output.status.success(),
        "subprocess test capi::tokenizer::property::unigram_random_score_models_are_deterministic_and_offset_safe failed (status: {:?})\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

/// Runs the randomized BPE merge-model proptest in a subprocess so native
/// faults are reported as clean failures without aborting the parent.
#[test]
fn bpe_random_model_variants_are_deterministic_and_offset_safe_subprocess() {
    let exe = std::env::current_exe().expect("current test executable path must resolve");
    let output = Command::new(exe)
        .arg("--exact")
        .arg("capi::tokenizer::property::bpe_random_model_variants_are_deterministic_and_offset_safe")
        .arg("--ignored")
        .arg("--nocapture")
        .env("PROPTEST_RNG_SEED", "2004318071")
        .env("PROPTEST_DISABLE_FAILURE_PERSISTENCE", "1")
        .output()
        .expect("subprocess launch for bpe merge-model proptest must succeed");
    assert!(
        output.status.success(),
        "subprocess test capi::tokenizer::property::bpe_random_model_variants_are_deterministic_and_offset_safe failed (status: {:?})\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

/// Runs deep BPE merge-graph reference parity in a subprocess so native faults
/// are reported as clean failures without aborting the parent.
#[test]
fn bpe_deep_merge_graph_matches_reference_subprocess() {
    let exe = std::env::current_exe().expect("current test executable path must resolve");
    let output = Command::new(exe)
        .arg("--exact")
        .arg("capi::tokenizer::property::bpe_deep_merge_graph_matches_reference")
        .arg("--ignored")
        .arg("--nocapture")
        .env("PROPTEST_RNG_SEED", "2004318071")
        .env("PROPTEST_DISABLE_FAILURE_PERSISTENCE", "1")
        .output()
        .expect("subprocess launch for deep bpe reference proptest must succeed");
    assert!(
        output.status.success(),
        "subprocess test capi::tokenizer::property::bpe_deep_merge_graph_matches_reference failed (status: {:?})\nstdout:\n{}\nstderr:\n{}",
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
