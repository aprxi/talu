//! Property-based tokenizer invariants.
//!
//! These tests complement example fixtures by probing random byte inputs and
//! asserting core safety/consistency properties across the C API boundary.

use crate::capi::tokenizer::common::{byte_token_id, decode_raw, encode_raw, TokenizerTestContext};
use proptest::prelude::*;

fn no_bos() -> talu_sys::EncodeOptions {
    talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    }
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
    fn byte_level_valid_unicode_roundtrips_exactly(
        text in proptest::string::string_regex("(?s).{0,256}").expect("regex must compile")
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
}
