//! Stress, concurrency, and large-input tests for tokenizer C API.
//!
//! These tests exercise buffer handling, resource lifecycle, and thread safety
//! at scale — matching the depth of the db/chat stress tests.

use crate::capi::tokenizer::common::{
    byte_token_id, byte_token_surface, OwnedDecodeResult, OwnedEncodeResult, TokenizerTestContext,
    TOKENIZER_JSON,
};
use std::ffi::c_void;
use std::ptr;
use std::sync::{Arc, Barrier};
use std::thread;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Thread-safe wrapper around a raw tokenizer handle.
///
/// The Zig tokenizer handle is safe for concurrent reads (encode/decode).
struct SharedHandle(*mut c_void);

unsafe impl Send for SharedHandle {}
unsafe impl Sync for SharedHandle {}

impl SharedHandle {
    fn ptr(&self) -> *mut c_void {
        self.0
    }
}

/// Generate a repeating ASCII string of `n` bytes using a-z (no spaces).
///
/// Each character maps to exactly one token in the fixture tokenizer.
fn ascii_string(n: usize) -> String {
    "abcdefghijklmnopqrstuvwxyz"
        .chars()
        .cycle()
        .take(n)
        .collect()
}

fn lcg_next(state: &mut u64) -> u64 {
    // Numerical Recipes LCG constants; deterministic test data only.
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    *state
}

fn seeded_ascii_non_space(seed: u64, len: usize) -> String {
    let mut state = seed;
    let mut out = String::with_capacity(len);
    for _ in 0..len {
        let v = (lcg_next(&mut state) % 94) as u8; // 0x21..=0x7E
        out.push((0x21 + v) as char);
    }
    out
}

fn seeded_bytes(seed: u64, len: usize) -> Vec<u8> {
    let mut state = seed;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push((lcg_next(&mut state) & 0xFF) as u8);
    }
    out
}

/// Default encode options (no BOS).
fn no_bos() -> talu_sys::EncodeOptions {
    talu_sys::EncodeOptions {
        add_bos: 0,
        ..Default::default()
    }
}

// ===========================================================================
// Large input tests
// ===========================================================================

/// Encoding a 1MB string produces exactly 1M tokens (one per char, no merges).
#[test]
fn encode_1mb_string() {
    let ctx = TokenizerTestContext::new();
    let input = ascii_string(1_000_000);
    let tokens = ctx.encode_with(&input, &no_bos());
    assert_eq!(tokens.len(), 1_000_000);
}

/// Decoding 1M tokens roundtrips to the original 1MB string.
#[test]
fn decode_1mb_roundtrip() {
    let ctx = TokenizerTestContext::new();
    let input = ascii_string(1_000_000);
    let tokens = ctx.encode_with(&input, &no_bos());
    let decoded = ctx.decode(&tokens);
    assert_eq!(decoded, input);
}

/// Encoding 100K chars produces offsets with contiguous single-byte spans.
#[test]
fn encode_offsets_100k_chars() {
    let ctx = TokenizerTestContext::new();
    let input = ascii_string(100_000);
    let result = unsafe { super::common::encode_raw(ctx.handle(), input.as_bytes(), &no_bos()) };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 100_000);

    let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
    assert_eq!(offsets[0].start, 0);
    assert_eq!(offsets[0].end, 1);
    assert_eq!(offsets.last().unwrap().end as usize, 100_000);

    unsafe { talu_sys::talu_encode_result_free(result) };
}

/// tokenize_bytes on 100K chars produces 100K tokens.
#[test]
fn tokenize_bytes_100k_chars() {
    let ctx = TokenizerTestContext::new();
    let input = ascii_string(100_000);
    let result = unsafe {
        talu_sys::talu_tokenizer_tokenize_bytes(
            ctx.handle(),
            input.as_bytes().as_ptr(),
            input.len(),
        )
    };
    assert!(result.error_msg.is_null());
    assert_eq!(result.num_tokens, 100_000);

    unsafe {
        talu_sys::talu_tokenize_bytes_result_free(
            result.data,
            result.data_len,
            result.offsets,
            result.num_tokens,
        )
    };
}

/// Byte-level encoding of deterministic arbitrary bytes must preserve exact
/// byte-token IDs and single-byte offsets, even for embedded NULs and invalid
/// UTF-8 subsequences.
#[test]
fn byte_level_seeded_raw_bytes_preserve_exact_byte_ids_and_offsets() {
    let ctx = TokenizerTestContext::with_byte_level();

    for (seed, len) in [(1_u64, 0_usize), (7, 1), (19, 31), (42, 257)] {
        let bytes = seeded_bytes(seed, len);
        let result = unsafe { super::common::encode_raw(ctx.handle(), &bytes, &no_bos()) };
        assert!(result.error_msg.is_null(), "seed={seed} len={len}: encode failed");
        assert_eq!(
            result.num_tokens,
            bytes.len(),
            "seed={seed} len={len}: byte-level token count must equal raw byte count"
        );

        let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
        let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };

        for (idx, (&byte, &id)) in bytes.iter().zip(ids.iter()).enumerate() {
            assert_eq!(
                id,
                byte_token_id(byte),
                "seed={seed} len={len}: raw byte at index {idx} must map to its exact byte token ID"
            );
            assert_eq!(offsets[idx].start as usize, idx, "seed={seed} len={len}: offset[{idx}].start");
            assert_eq!(offsets[idx].end as usize, idx + 1, "seed={seed} len={len}: offset[{idx}].end");
        }

        unsafe { talu_sys::talu_encode_result_free(result) };
    }
}

/// Batch byte-level encoding of deterministic arbitrary bytes must preserve the
/// exact per-sequence byte-token streams produced by individual encoding.
#[test]
fn byte_level_seeded_raw_bytes_batch_matches_individual_sequences() {
    let ctx = TokenizerTestContext::with_byte_level();

    let cases = [
        seeded_bytes(11, 0),
        seeded_bytes(23, 3),
        seeded_bytes(47, 19),
        seeded_bytes(99, 128),
    ];

    let ptrs: Vec<*const u8> = cases.iter().map(|bytes| bytes.as_ptr()).collect();
    let lengths: Vec<usize> = cases.iter().map(Vec::len).collect();
    let batch = unsafe { super::common::encode_batch_raw(ctx.handle(), &ptrs, &lengths, &no_bos()) };
    assert!(batch.error_msg.is_null(), "batch encode failed");
    assert_eq!(batch.num_sequences, cases.len());

    let ids = unsafe { std::slice::from_raw_parts(batch.ids, batch.total_tokens) };
    let offsets = unsafe { std::slice::from_raw_parts(batch.offsets, batch.num_sequences + 1) };

    for (idx, bytes) in cases.iter().enumerate() {
        let expected: Vec<u32> = bytes.iter().map(|&byte| byte_token_id(byte)).collect();
        assert_eq!(
            &ids[offsets[idx]..offsets[idx + 1]],
            expected.as_slice(),
            "seeded raw-byte batch slice must equal exact byte-token IDs for sequence {idx}"
        );
    }

    unsafe {
        talu_sys::talu_batch_encode_result_free(
            batch.ids,
            batch.offsets,
            batch.total_tokens,
            batch.num_sequences,
        )
    };
}

/// Batch byte-level offsets must equal the cumulative per-sequence token
/// counts from individual encodes, even for empty slices and arbitrary bytes.
#[test]
fn byte_level_seeded_raw_bytes_batch_offsets_match_individual_lengths() {
    let ctx = TokenizerTestContext::with_byte_level();

    let cases = [
        seeded_bytes(3, 0),
        seeded_bytes(17, 1),
        seeded_bytes(41, 11),
        seeded_bytes(73, 64),
    ];

    let ptrs: Vec<*const u8> = cases.iter().map(|bytes| bytes.as_ptr()).collect();
    let lengths: Vec<usize> = cases.iter().map(Vec::len).collect();
    let batch = unsafe { super::common::encode_batch_raw(ctx.handle(), &ptrs, &lengths, &no_bos()) };
    assert!(batch.error_msg.is_null(), "batch encode failed");
    assert_eq!(batch.num_sequences, cases.len());

    let offsets = unsafe { std::slice::from_raw_parts(batch.offsets, batch.num_sequences + 1) };
    let mut expected = Vec::with_capacity(cases.len() + 1);
    expected.push(0usize);
    for bytes in &cases {
        expected.push(expected.last().copied().unwrap() + bytes.len());
    }
    assert_eq!(
        offsets,
        expected.as_slice(),
        "seeded raw-byte batch offsets must equal cumulative individual token counts"
    );

    unsafe {
        talu_sys::talu_batch_encode_result_free(
            batch.ids,
            batch.offsets,
            batch.total_tokens,
            batch.num_sequences,
        )
    };
}

/// For valid UTF-8 inputs, each batch byte-level slice must decode back to its
/// original sequence when decoded independently. This catches slice-boundary
/// bugs that ID-only batch comparisons can miss.
#[test]
fn byte_level_batch_valid_utf8_slices_decode_to_original_sequences() {
    let ctx = TokenizerTestContext::with_byte_level();
    let cases = ["", "Hello", "café", "🇺🇸", "👨\u{200d}👩\u{200d}👧\u{200d}👦", "a\u{0301}"];

    let ptrs: Vec<*const u8> = cases.iter().map(|text| text.as_bytes().as_ptr()).collect();
    let lengths: Vec<usize> = cases.iter().map(|text| text.len()).collect();
    let batch = unsafe { super::common::encode_batch_raw(ctx.handle(), &ptrs, &lengths, &no_bos()) };
    assert!(batch.error_msg.is_null(), "batch encode failed");
    assert_eq!(batch.num_sequences, cases.len());

    let ids = unsafe { std::slice::from_raw_parts(batch.ids, batch.total_tokens) };
    let offsets = unsafe { std::slice::from_raw_parts(batch.offsets, batch.num_sequences + 1) };

    for (idx, text) in cases.iter().enumerate() {
        let decoded = ctx.decode_with(
            &ids[offsets[idx]..offsets[idx + 1]],
            &talu_sys::DecodeOptionsC {
                skip_special_tokens: 0,
            },
        );
        assert_eq!(
            decoded, *text,
            "batch byte-level slice must decode to the original valid UTF-8 sequence at index {idx}"
        );
    }

    unsafe {
        talu_sys::talu_batch_encode_result_free(
            batch.ids,
            batch.offsets,
            batch.total_tokens,
            batch.num_sequences,
        )
    };
}

/// Byte-level tokenize_bytes on deterministic arbitrary bytes must expose the
/// exact UTF-8 token surfaces for the GPT-2 byte-to-Unicode mapping, not raw
/// input bytes.
#[test]
fn byte_level_seeded_raw_bytes_tokenize_bytes_surfaces_exact_slices() {
    let ctx = TokenizerTestContext::with_byte_level();

    for (seed, len) in [(5_u64, 0_usize), (13, 2), (29, 17), (77, 96)] {
        let bytes = seeded_bytes(seed, len);
        let result = unsafe {
            talu_sys::talu_tokenizer_tokenize_bytes(ctx.handle(), bytes.as_ptr(), bytes.len())
        };
        assert!(
            result.error_msg.is_null(),
            "seed={seed} len={len}: tokenize_bytes failed"
        );
        assert_eq!(
            result.num_tokens,
            bytes.len(),
            "seed={seed} len={len}: byte-level tokenize_bytes count must equal raw byte count"
        );

        let data = unsafe { std::slice::from_raw_parts(result.data, result.data_len) };
        let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens + 1) };
        assert_eq!(offsets[0], 0, "seed={seed} len={len}: offsets[0]");
        assert_eq!(
            *offsets.last().unwrap(),
            data.len(),
            "seed={seed} len={len}: last offset must equal data_len"
        );

        for (idx, &byte) in bytes.iter().enumerate() {
            let start = offsets[idx];
            let end = offsets[idx + 1];
            let expected = byte_token_surface(byte);
            assert_eq!(
                &data[start..end],
                expected.as_bytes(),
                "seed={seed} len={len}: token {idx} surface must equal exact byte-level token bytes"
            );
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

// ===========================================================================
// Rapid alloc/free cycles
// ===========================================================================

/// 200 encode→free cycles on the same handle.
#[test]
fn rapid_encode_free_200_cycles() {
    let ctx = TokenizerTestContext::new();
    let opts = no_bos();
    let texts = ["Hello", "abc", "012", "!@#$%", "A"];

    for i in 0..200 {
        let text = texts[i % texts.len()];
        let tokens = ctx.encode_with(text, &opts);
        assert!(!tokens.is_empty(), "cycle {i}: should produce tokens");
    }
}

/// 200 decode→free cycles.
#[test]
fn rapid_decode_free_200_cycles() {
    let ctx = TokenizerTestContext::new();
    let token_sets: &[&[u32]] = &[
        &[44, 77],             // "Hi"
        &[44, 73, 80, 80, 83], // "Hello"
        &[37],                 // "A"
        &[69, 70, 71],         // "abc"
        &[20, 21, 22],         // "012"
    ];

    for i in 0..200 {
        let tokens = token_sets[i % token_sets.len()];
        let text = ctx.decode(tokens);
        assert!(!text.is_empty(), "cycle {i}: should produce text");
    }
}

/// 100 batch_encode→free cycles with 3 sequences each.
#[test]
fn rapid_batch_encode_free_100_cycles() {
    let ctx = TokenizerTestContext::new();
    let opts = no_bos();

    for i in 0..100 {
        let batch = ctx.encode_batch(&["Hi", "Hello", "abc"], &opts);
        assert_eq!(batch.num_sequences, 3, "cycle {i}: 3 sequences");
        assert_eq!(batch.ids.len(), 10, "cycle {i}: 2+5+3 = 10 tokens");
    }
}

/// 100 tokenizer create→free cycles.
#[test]
fn rapid_tokenizer_create_free_100_cycles() {
    let json = TOKENIZER_JSON.as_bytes();

    for i in 0..100 {
        let mut handle: *mut c_void = ptr::null_mut();
        let rc = unsafe {
            talu_sys::talu_tokenizer_create_from_json(
                json.as_ptr(),
                json.len(),
                &mut handle as *mut _ as *mut c_void,
            )
        };
        assert_eq!(rc, 0, "cycle {i}: create should succeed");
        assert!(!handle.is_null());
        unsafe { talu_sys::talu_tokenizer_free(handle) };
    }
}

/// 100 encode→free cycles checking offsets.
#[test]
fn rapid_encode_offsets_free_100_cycles() {
    let ctx = TokenizerTestContext::new();
    let opts = no_bos();
    let texts = ["Hello", "abc", "!@#$%"];

    for i in 0..100 {
        let text = texts[i % texts.len()];
        let result = unsafe { super::common::encode_raw(ctx.handle(), text.as_bytes(), &opts) };
        assert!(result.error_msg.is_null(), "cycle {i}: should succeed");
        assert_eq!(result.num_tokens, text.len());
        unsafe { talu_sys::talu_encode_result_free(result) };
    }
}

// ===========================================================================
// Concurrent multi-threaded tests
// ===========================================================================

/// 8 threads × 50 iterations encoding on a shared tokenizer handle.
///
/// Verifies that concurrent encode calls produce correct, independent results.
#[test]
fn concurrent_encode_8_threads() {
    let ctx = TokenizerTestContext::new();
    let shared = Arc::new(SharedHandle(ctx.handle()));
    let num_threads = 8;
    let iterations = 50;
    let mut handles = Vec::new();

    for thread_id in 0..num_threads {
        let shared = Arc::clone(&shared);
        let handle = thread::spawn(move || {
            let texts: &[&[u8]] = &[b"Hi", b"Hello", b"abc", b"012", b"A"];
            let expected: &[&[u32]] = &[
                &[44, 77],
                &[44, 73, 80, 80, 83],
                &[69, 70, 71],
                &[20, 21, 22],
                &[37],
            ];
            let opts = no_bos();

            for i in 0..iterations {
                let idx = (thread_id + i) % texts.len();
                let result = unsafe { super::common::encode_raw(shared.ptr(), texts[idx], &opts) };
                assert!(
                    result.error_msg.is_null(),
                    "thread {thread_id} iter {i}: encode failed"
                );
                let tokens = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
                assert_eq!(
                    tokens, expected[idx],
                    "thread {thread_id} iter {i}: wrong tokens"
                );
                unsafe { talu_sys::talu_encode_result_free(result) };
            }
        });
        handles.push(handle);
    }

    // Join ALL threads before ctx is dropped (handle must outlive threads).
    let results: Vec<thread::Result<()>> = handles.into_iter().map(|h| h.join()).collect();
    for (i, result) in results.into_iter().enumerate() {
        result.unwrap_or_else(|_| panic!("Thread {i} panicked"));
    }
}

/// 8 threads × 50 iterations decoding on a shared tokenizer handle.
#[test]
fn concurrent_decode_8_threads() {
    let ctx = TokenizerTestContext::new();
    let shared = Arc::new(SharedHandle(ctx.handle()));
    let num_threads = 8;
    let iterations = 50;
    let mut handles = Vec::new();

    for thread_id in 0..num_threads {
        let shared = Arc::clone(&shared);
        let handle = thread::spawn(move || {
            let cases: &[(&[u32], &str)] = &[
                (&[44, 77], "Hi"),
                (&[44, 73, 80, 80, 83], "Hello"),
                (&[69, 70, 71], "abc"),
                (&[20, 21, 22], "012"),
                (&[37], "A"),
            ];
            let opts = talu_sys::DecodeOptionsC::default();

            for i in 0..iterations {
                let idx = (thread_id + i) % cases.len();
                let (tokens, expected) = cases[idx];
                let result = unsafe { super::common::decode_raw(shared.ptr(), tokens, &opts) };
                assert!(
                    result.error_msg.is_null(),
                    "thread {thread_id} iter {i}: decode failed"
                );
                let text = unsafe {
                    let slice = std::slice::from_raw_parts(result.text, result.text_len);
                    std::str::from_utf8(slice).unwrap()
                };
                assert_eq!(text, expected, "thread {thread_id} iter {i}: wrong decode");
                unsafe { talu_sys::talu_decode_result_free(result.text, result.text_len) };
            }
        });
        handles.push(handle);
    }

    let results: Vec<thread::Result<()>> = handles.into_iter().map(|h| h.join()).collect();
    for (i, result) in results.into_iter().enumerate() {
        result.unwrap_or_else(|_| panic!("Thread {i} panicked"));
    }
}

/// 8 threads × 25 iterations of encode→decode roundtrip on shared handle.
///
/// Each thread encodes a string, decodes the result, and verifies the roundtrip.
#[test]
fn concurrent_roundtrip_8_threads() {
    let ctx = TokenizerTestContext::new();
    let shared = Arc::new(SharedHandle(ctx.handle()));
    let num_threads = 8;
    let iterations = 25;
    let mut handles = Vec::new();

    for thread_id in 0..num_threads {
        let shared = Arc::clone(&shared);
        let handle = thread::spawn(move || {
            let texts: &[&[u8]] = &[b"Hi", b"Hello", b"abc", b"012", b"A"];
            let text_strs = ["Hi", "Hello", "abc", "012", "A"];
            let encode_opts = no_bos();
            let decode_opts = talu_sys::DecodeOptionsC::default();

            for i in 0..iterations {
                let idx = (thread_id + i) % texts.len();

                // Encode
                let enc =
                    unsafe { super::common::encode_raw(shared.ptr(), texts[idx], &encode_opts) };
                assert!(enc.error_msg.is_null());

                // Decode
                let dec = unsafe {
                    super::common::decode_raw(
                        shared.ptr(),
                        std::slice::from_raw_parts(enc.ids, enc.num_tokens),
                        &decode_opts,
                    )
                };
                assert!(dec.error_msg.is_null());

                let decoded = unsafe {
                    let slice = std::slice::from_raw_parts(dec.text, dec.text_len);
                    std::str::from_utf8(slice).unwrap()
                };
                assert_eq!(
                    decoded, text_strs[idx],
                    "thread {thread_id} iter {i}: roundtrip mismatch"
                );

                unsafe {
                    talu_sys::talu_encode_result_free(enc);
                    talu_sys::talu_decode_result_free(dec.text, dec.text_len);
                };
            }
        });
        handles.push(handle);
    }

    let results: Vec<thread::Result<()>> = handles.into_iter().map(|h| h.join()).collect();
    for (i, result) in results.into_iter().enumerate() {
        result.unwrap_or_else(|_| panic!("Thread {i} panicked"));
    }
}

/// A barrier-backed high-contention encode workload must remain correct under
/// real scheduling overlap on a single shared handle.
#[test]
fn concurrent_encode_barrier_high_contention() {
    let ctx = TokenizerTestContext::new();
    let shared = Arc::new(SharedHandle(ctx.handle()));
    let barrier = Arc::new(Barrier::new(16));
    let iterations = 1_000;
    let mut handles = Vec::new();

    for thread_id in 0..16 {
        let shared = Arc::clone(&shared);
        let barrier = Arc::clone(&barrier);
        let handle = thread::spawn(move || {
            let texts: &[&[u8]] = &[b"Hi", b"Hello", b"abc", b"012", b"A"];
            let expected: &[&[u32]] = &[
                &[44, 77],
                &[44, 73, 80, 80, 83],
                &[69, 70, 71],
                &[20, 21, 22],
                &[37],
            ];
            let opts = no_bos();

            barrier.wait();
            for i in 0..iterations {
                let idx = (thread_id + i) % texts.len();
                let result =
                    OwnedEncodeResult::new(unsafe { super::common::encode_raw(shared.ptr(), texts[idx], &opts) });
                assert!(result.error_msg().is_null(), "thread {thread_id} iter {i}: encode failed");
                let tokens = result.ids();
                assert_eq!(tokens, expected[idx], "thread {thread_id} iter {i}: wrong tokens");
            }
        });
        handles.push(handle);
    }

    for (i, handle) in handles.into_iter().enumerate() {
        handle.join().unwrap_or_else(|_| panic!("Thread {i} panicked"));
    }
}

/// Error state must be thread-isolated: an error on one thread must not leak
/// into `talu_last_error_code()` reads on another thread using the same handle.
#[test]
fn concurrent_last_error_code_is_thread_isolated() {
    let ctx = TokenizerTestContext::new();
    let shared = Arc::new(SharedHandle(ctx.handle()));
    let barrier = Arc::new(Barrier::new(2));

    let ok_handle = {
        let shared = Arc::clone(&shared);
        let barrier = Arc::clone(&barrier);
        thread::spawn(move || {
            let opts = no_bos();
            barrier.wait();
            for i in 0..500 {
                let result =
                    OwnedEncodeResult::new(unsafe { super::common::encode_raw(shared.ptr(), b"Hello", &opts) });
                assert!(result.error_msg().is_null(), "success thread iter {i}: encode failed");
                assert_eq!(
                    unsafe { talu_sys::talu_last_error_code() },
                    talu_sys::ErrorCode::Ok as i32,
                    "success thread iter {i}: error code leaked from another thread"
                );
            }
        })
    };

    let err_handle = {
        let shared = Arc::clone(&shared);
        let barrier = Arc::clone(&barrier);
        thread::spawn(move || {
            let opts = talu_sys::DecodeOptionsC::default();
            barrier.wait();
            for i in 0..500 {
                let result =
                    OwnedDecodeResult::new(unsafe { super::common::decode_raw(shared.ptr(), &[u32::MAX], &opts) });
                assert!(
                    !result.error_msg().is_null(),
                    "error thread iter {i}: invalid-token decode must fail"
                );
                assert_eq!(
                    unsafe { talu_sys::talu_last_error_code() },
                    talu_sys::ErrorCode::TokenizerInvalidTokenId as i32,
                    "error thread iter {i}: wrong error code"
                );
            }
        })
    };

    ok_handle.join().expect("success thread panicked");
    err_handle.join().expect("error thread panicked");
}

/// A barrier-backed mixed workload must remain correct under real overlap
/// across encode, decode, id lookup, and vocab-size queries.
#[test]
fn concurrent_mixed_workload_barrier_high_contention() {
    let ctx = TokenizerTestContext::new();
    let shared = Arc::new(SharedHandle(ctx.handle()));
    let barrier = Arc::new(Barrier::new(24));
    let encode_opts = no_bos();
    let decode_opts = talu_sys::DecodeOptionsC::default();
    let mut handles = Vec::new();

    for thread_id in 0..24 {
        let shared = Arc::clone(&shared);
        let barrier = Arc::clone(&barrier);
        let h = thread::spawn(move || {
            barrier.wait();
            for i in 0..500 {
                match (thread_id + i) % 4 {
                    0 => {
                        let enc =
                            OwnedEncodeResult::new(unsafe { super::common::encode_raw(shared.ptr(), b"Hello", &encode_opts) });
                        assert!(enc.error_msg().is_null(), "thread {thread_id} iter {i}: encode failed");
                        let ids = enc.ids();
                        assert_eq!(ids, &[44, 73, 80, 80, 83]);
                    }
                    1 => {
                        let dec =
                            OwnedDecodeResult::new(unsafe { super::common::decode_raw(shared.ptr(), &[44, 77], &decode_opts) });
                        assert!(dec.error_msg().is_null(), "thread {thread_id} iter {i}: decode failed");
                        assert_eq!(dec.text(), "Hi");
                    }
                    2 => {
                        let mut token: *mut i8 = std::ptr::null_mut();
                        let rc = unsafe {
                            talu_sys::talu_tokenizer_id_to_token(
                                shared.ptr(),
                                44,
                                &mut token as *mut _ as *mut std::ffi::c_void,
                            )
                        };
                        assert_eq!(rc, 0, "thread {thread_id} iter {i}: id_to_token failed");
                        let text = unsafe { std::ffi::CStr::from_ptr(token) }.to_string_lossy();
                        assert_eq!(text, "H");
                        unsafe { talu_sys::talu_text_free(token) };
                    }
                    _ => {
                        let size = unsafe { talu_sys::talu_tokenizer_get_vocab_size(shared.ptr()) };
                        assert!(size >= 99, "thread {thread_id} iter {i}: bad vocab size");
                    }
                }
            }
        });
        handles.push(h);
    }

    for (i, h) in handles.into_iter().enumerate() {
        h.join().unwrap_or_else(|_| panic!("Thread {i} panicked"));
    }
}

/// `talu_take_last_error` must also be thread-isolated: one thread consuming an
/// error must not observe or clear another thread's clean state.
#[test]
fn concurrent_take_last_error_is_thread_isolated() {
    let ctx = TokenizerTestContext::new();
    let shared = Arc::new(SharedHandle(ctx.handle()));
    let barrier = Arc::new(Barrier::new(2));

    let ok_handle = {
        let shared = Arc::clone(&shared);
        let barrier = Arc::clone(&barrier);
        thread::spawn(move || {
            let opts = no_bos();
            barrier.wait();
            for i in 0..400 {
                let result = unsafe { super::common::encode_raw(shared.ptr(), b"Hello", &opts) };
                assert!(result.error_msg.is_null(), "success thread iter {i}: encode failed");
                let mut code = -1;
                let required = unsafe {
                    talu_sys::talu_take_last_error(
                        std::ptr::null_mut(),
                        0,
                        &mut code as *mut _ as *mut std::ffi::c_void,
                    )
                };
                assert_eq!(required, 0, "success thread iter {i}: take_last_error should be empty");
                assert_eq!(
                    code,
                    talu_sys::ErrorCode::Ok as i32,
                    "success thread iter {i}: take_last_error code leaked from another thread"
                );
                unsafe { talu_sys::talu_encode_result_free(result) };
            }
        })
    };

    let err_handle = {
        let shared = Arc::clone(&shared);
        let barrier = Arc::clone(&barrier);
        thread::spawn(move || {
            let opts = talu_sys::DecodeOptionsC::default();
            barrier.wait();
            for i in 0..400 {
                let result = unsafe { super::common::decode_raw(shared.ptr(), &[u32::MAX], &opts) };
                assert!(
                    !result.error_msg.is_null(),
                    "error thread iter {i}: invalid-token decode must fail"
                );
                let mut code = 0;
                let required = unsafe {
                    talu_sys::talu_take_last_error(
                        std::ptr::null_mut(),
                        0,
                        &mut code as *mut _ as *mut std::ffi::c_void,
                    )
                };
                assert!(
                    required > 0,
                    "error thread iter {i}: take_last_error query must report a message"
                );
                assert_eq!(
                    code,
                    talu_sys::ErrorCode::TokenizerInvalidTokenId as i32,
                    "error thread iter {i}: take_last_error must report the decode error"
                );
            }
        })
    };

    ok_handle.join().expect("success thread panicked");
    err_handle.join().expect("error thread panicked");
}

/// Repeated tokenizer-creation failures on one thread must not poison clean
/// encode calls or error-state queries on another thread. This stresses the
/// create/destroy error path, not just decode failures on an existing handle.
#[test]
fn concurrent_invalid_json_creation_does_not_bleed_into_clean_encode_thread() {
    let ctx = TokenizerTestContext::new();
    let shared = Arc::new(SharedHandle(ctx.handle()));
    let barrier = Arc::new(Barrier::new(2));

    let ok_handle = {
        let shared = Arc::clone(&shared);
        let barrier = Arc::clone(&barrier);
        thread::spawn(move || {
            let opts = no_bos();
            barrier.wait();
            for i in 0..400 {
                let result =
                    OwnedEncodeResult::new(unsafe { super::common::encode_raw(shared.ptr(), b"Hello", &opts) });
                assert!(result.error_msg().is_null(), "success thread iter {i}: encode failed");
                assert_eq!(result.ids(), &[44, 73, 80, 80, 83]);
                assert_eq!(
                    unsafe { talu_sys::talu_last_error_code() },
                    talu_sys::ErrorCode::Ok as i32,
                    "success thread iter {i}: error code leaked from create failure thread"
                );
            }
        })
    };

    let err_handle = {
        let barrier = Arc::clone(&barrier);
        thread::spawn(move || {
            barrier.wait();
            for i in 0..400 {
                let mut handle: *mut c_void = ptr::null_mut();
                let rc = unsafe {
                    talu_sys::talu_tokenizer_create_from_json(
                        ptr::null(),
                        0,
                        &mut handle as *mut _ as *mut c_void,
                    )
                };
                assert_ne!(rc, 0, "error thread iter {i}: null/empty JSON input must fail to load");
                assert!(
                    handle.is_null(),
                    "error thread iter {i}: failed creation must not return a handle"
                );
                assert_ne!(
                    unsafe { talu_sys::talu_last_error_code() },
                    talu_sys::ErrorCode::Ok as i32,
                    "error thread iter {i}: create failure must set a thread-local error"
                );
            }
        })
    };

    ok_handle.join().expect("success thread panicked");
    err_handle.join().expect("error thread panicked");
}

// ===========================================================================
// Batch scaling tests
// ===========================================================================

/// Batch encoding 100 single-character sequences.
#[test]
fn batch_100_sequences() {
    let ctx = TokenizerTestContext::new();
    let opts = no_bos();

    // 100 single-character strings cycling A-Z.
    let texts: Vec<String> = (b'A'..=b'Z')
        .cycle()
        .take(100)
        .map(|c| String::from(c as char))
        .collect();
    let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

    let batch = ctx.encode_batch(&refs, &opts);

    assert_eq!(batch.num_sequences, 100);
    assert_eq!(batch.offsets.len(), 101);
    assert_eq!(batch.ids.len(), 100); // 1 token per char

    // Offsets should be [0, 1, 2, ..., 100].
    let expected_offsets: Vec<usize> = (0..=100).collect();
    assert_eq!(batch.offsets, expected_offsets);
}

/// Padded tensor with 50 sequences of increasing length (1..=50).
#[test]
fn padded_tensor_50_sequences_varying_length() {
    let ctx = TokenizerTestContext::new();
    let opts = no_bos();
    let pad_opts = talu_sys::PaddedTensorOptions {
        pad_id: 0,
        padding_side: 0, // right
        return_attention_mask: true,
        ..Default::default()
    };

    // Sequences of length 1..=50 using "a" repeated.
    let texts: Vec<String> = (1..=50).map(|n| "a".repeat(n)).collect();
    let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

    let result = ctx.batch_to_padded_tensor(&refs, &opts, &pad_opts);

    assert_eq!(result.num_sequences, 50);
    assert_eq!(result.padded_length, 50); // longest = 50 tokens
    assert_eq!(result.input_ids.len(), 50 * 50);
    assert_eq!(result.attention_mask.len(), 50 * 50);

    // Row 0: 1 real token + 49 padding (right-padded).
    assert_eq!(result.attention_mask[0], 1);
    for j in 1..50 {
        assert_eq!(result.attention_mask[j], 0, "row 0, col {j} should be pad");
    }

    // Row 49 (last): 50 real tokens, no padding.
    let last_start = 49 * 50;
    for j in 0..50 {
        assert_eq!(
            result.attention_mask[last_start + j],
            1,
            "last row, col {j} should be real"
        );
    }
}

// ===========================================================================
// Edge cases
// ===========================================================================

/// Truncation to max_length=1 keeps exactly one token.
#[test]
fn encode_truncation_to_one_token() {
    let ctx = TokenizerTestContext::new();
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        truncation: 1,
        truncation_side: 0, // right: keep first
        max_length: 1,
        ..Default::default()
    };
    assert_eq!(ctx.encode_with("Hello", &opts), [44]); // just "H"
}

/// Left truncation to max_length=1 keeps the last token.
#[test]
fn encode_truncation_left_to_one_token() {
    let ctx = TokenizerTestContext::new();
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        truncation: 1,
        truncation_side: 1, // left: keep last
        max_length: 1,
        ..Default::default()
    };
    assert_eq!(ctx.encode_with("Hello", &opts), [83]); // just "o"
}

/// Truncation with max_length=0 is a no-op (0 means "no limit").
#[test]
fn encode_truncation_max_length_zero_is_noop() {
    let ctx = TokenizerTestContext::new();
    let opts = talu_sys::EncodeOptions {
        add_bos: 0,
        truncation: 1,
        truncation_side: 0,
        max_length: 0,
        ..Default::default()
    };
    // max_length=0 means no truncation applied.
    assert_eq!(ctx.encode_with("Hello", &opts), [44, 73, 80, 80, 83]);
}

/// 8 threads concurrently creating tokenizers from JSON.
///
/// Each thread independently creates a tokenizer, encodes a string,
/// verifies the result, and frees the handle.
#[test]
fn concurrent_tokenizer_creation_8_threads() {
    let num_threads = 8;
    let iterations = 50;
    let mut handles = Vec::new();

    for thread_id in 0..num_threads {
        let handle = thread::spawn(move || {
            for i in 0..iterations {
                let json = TOKENIZER_JSON.as_bytes();
                let mut h: *mut c_void = ptr::null_mut();
                let rc = unsafe {
                    talu_sys::talu_tokenizer_create_from_json(
                        json.as_ptr(),
                        json.len(),
                        &mut h as *mut _ as *mut c_void,
                    )
                };
                assert_eq!(rc, 0, "thread {thread_id} iter {i}: create failed");
                assert!(!h.is_null());

                // Encode "Hi" and verify.
                let opts = no_bos();
                let result = unsafe { super::common::encode_raw(h, b"Hi", &opts) };
                assert!(result.error_msg.is_null());
                let tokens = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
                assert_eq!(
                    tokens,
                    &[44, 77],
                    "thread {thread_id} iter {i}: wrong encode"
                );

                unsafe {
                    talu_sys::talu_encode_result_free(result);
                    talu_sys::talu_tokenizer_free(h);
                };
            }
        });
        handles.push(handle);
    }

    let results: Vec<thread::Result<()>> = handles.into_iter().map(|h| h.join()).collect();
    for (i, result) in results.into_iter().enumerate() {
        result.unwrap_or_else(|_| panic!("Thread {i} panicked"));
    }
}

/// Every non-space printable ASCII char (0x21..=0x7E) encodes to exactly
/// one token with ID = byte_value - 0x20 + 4.
#[test]
fn encode_all_printable_ascii_non_space() {
    let ctx = TokenizerTestContext::new();
    let opts = no_bos();

    for byte in 0x21u8..=0x7Eu8 {
        let s = String::from(byte as char);
        let tokens = ctx.encode_with(&s, &opts);
        let expected_id = (byte as u32) - 0x20 + 4;
        assert_eq!(
            tokens,
            [expected_id],
            "char {:?} (0x{:02X}) should map to ID {}",
            byte as char,
            byte,
            expected_id,
        );
    }
}

/// Encoding and decoding a string with all non-space printable ASCII
/// produces an exact roundtrip.
#[test]
fn roundtrip_all_printable_ascii() {
    let ctx = TokenizerTestContext::new();
    let opts = no_bos();

    // All printable ASCII except space (which becomes <unk>).
    let input: String = (0x21u8..=0x7Eu8).map(|b| b as char).collect();
    let tokens = ctx.encode_with(&input, &opts);
    assert_eq!(tokens.len(), 94); // 0x7E - 0x21 + 1 = 94 chars
    let decoded = ctx.decode(&tokens);
    assert_eq!(decoded, input);
}

/// Seeded property test: ASCII non-space roundtrips exactly.
#[test]
fn seeded_property_roundtrip_ascii_non_space() {
    let ctx = TokenizerTestContext::new();
    let opts = no_bos();
    let mut seed = 0xDEADBEEFCAFEBABEu64;
    for case_idx in 0..200 {
        let len = ((lcg_next(&mut seed) % 128) + 1) as usize;
        let input = seeded_ascii_non_space(seed ^ case_idx as u64, len);
        let tokens = ctx.encode_with(&input, &opts);
        let decoded = ctx.decode(&tokens);
        assert_eq!(decoded, input, "seeded roundtrip failed for case {case_idx}");
    }
}

/// Seeded property test: offsets are monotonic, in-bounds, and contiguous for base fixture.
#[test]
fn seeded_property_offsets_invariants() {
    let ctx = TokenizerTestContext::new();
    let opts = no_bos();
    let mut seed = 0x1234_5678_9ABC_DEF0u64;
    for case_idx in 0..120 {
        let len = ((lcg_next(&mut seed) % 96) + 1) as usize;
        let input = seeded_ascii_non_space(seed ^ case_idx as u64, len);
        let result = unsafe { super::common::encode_raw(ctx.handle(), input.as_bytes(), &opts) };
        assert!(result.error_msg.is_null());
        assert_eq!(result.num_tokens, input.len());
        let offsets = unsafe { std::slice::from_raw_parts(result.offsets, result.num_tokens) };
        if !offsets.is_empty() {
            assert_eq!(offsets[0].start, 0);
            assert_eq!(offsets.last().unwrap().end as usize, input.len());
            for window in offsets.windows(2) {
                assert!(
                    window[0].start <= window[0].end && window[1].start <= window[1].end,
                    "offset spans must satisfy start<=end"
                );
                assert_eq!(window[0].end, window[1].start, "offsets must be contiguous");
            }
        }
        unsafe { talu_sys::talu_encode_result_free(result) };
    }
}

/// Seeded property test: left/right truncation must equal suffix/prefix of full encode.
#[test]
fn seeded_property_truncation_invariants() {
    let ctx = TokenizerTestContext::new();
    let mut seed = 0x0F0F_F0F0_A5A5_5A5Au64;
    for case_idx in 0..150 {
        let len = ((lcg_next(&mut seed) % 80) + 4) as usize;
        let input = seeded_ascii_non_space(seed ^ case_idx as u64, len);
        let full = ctx.encode_with(&input, &no_bos());
        let keep = ((lcg_next(&mut seed) % (full.len() as u64 - 1)) + 1) as usize;

        let right_opts = talu_sys::EncodeOptions {
            add_bos: 0,
            truncation: 1,
            truncation_side: 0,
            max_length: keep,
            ..Default::default()
        };
        let left_opts = talu_sys::EncodeOptions {
            add_bos: 0,
            truncation: 1,
            truncation_side: 1,
            max_length: keep,
            ..Default::default()
        };
        let right = ctx.encode_with(&input, &right_opts);
        let left = ctx.encode_with(&input, &left_opts);
        assert_eq!(right, full[..keep], "right truncation mismatch at case {case_idx}");
        assert_eq!(
            left,
            full[full.len() - keep..],
            "left truncation mismatch at case {case_idx}"
        );
    }
}

// ===========================================================================
// Byte-level fixture concurrency
// ===========================================================================

/// 8 threads encode "café" on byte-level fixture, verify exact byte-token IDs.
#[test]
fn concurrent_byte_level_encode_8_threads() {
    let ctx = TokenizerTestContext::with_byte_level();
    let handle = Arc::new(SharedHandle(ctx.handle()));
    let opts = no_bos();

    let expected_ids: Vec<u32> = "café"
        .as_bytes()
        .iter()
        .map(|&b| byte_token_id(b))
        .collect();

    let threads: Vec<_> = (0..8)
        .map(|_| {
            let h = Arc::clone(&handle);
            let expected = expected_ids.clone();
            thread::spawn(move || {
                let text = "café".as_bytes();
                for _ in 0..50 {
                    let result =
                        unsafe { crate::capi::tokenizer::common::encode_raw(h.ptr(), text, &opts) };
                    assert!(result.error_msg.is_null());
                    let tokens =
                        unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
                    assert_eq!(tokens, expected.as_slice(), "byte-level encode mismatch");
                    unsafe { talu_sys::talu_encode_result_free(result) };
                }
            })
        })
        .collect();

    for t in threads {
        t.join().expect("thread panicked");
    }
}

/// 8 threads encode→decode "日本語" on byte-level fixture, verify roundtrip.
#[test]
fn concurrent_byte_level_roundtrip_8_threads() {
    let ctx = TokenizerTestContext::with_byte_level();
    let handle = Arc::new(SharedHandle(ctx.handle()));
    let opts = no_bos();
    let decode_opts = talu_sys::DecodeOptionsC {
        skip_special_tokens: 0,
    };

    let threads: Vec<_> = (0..8)
        .map(|_| {
            let h = Arc::clone(&handle);
            thread::spawn(move || {
                let text = "日本語";
                for _ in 0..50 {
                    let result = unsafe {
                        crate::capi::tokenizer::common::encode_raw(h.ptr(), text.as_bytes(), &opts)
                    };
                    assert!(result.error_msg.is_null());
                    let tokens =
                        unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };

                    let dec_result = unsafe {
                        crate::capi::tokenizer::common::decode_raw(h.ptr(), tokens, &decode_opts)
                    };
                    assert!(dec_result.error_msg.is_null());
                    let decoded = unsafe {
                        std::str::from_utf8(std::slice::from_raw_parts(
                            dec_result.text as *const u8,
                            dec_result.text_len,
                        ))
                        .expect("decode must return valid UTF-8")
                    };
                    assert_eq!(decoded, text, "roundtrip mismatch");

                    unsafe {
                        talu_sys::talu_text_free(dec_result.text as *const i8);
                        talu_sys::talu_encode_result_free(result);
                    };
                }
            })
        })
        .collect();

    for t in threads {
        t.join().expect("thread panicked");
    }
}

/// High-contention mixed workload on one shared tokenizer handle.
#[test]
fn concurrent_mixed_workload_high_contention() {
    let ctx = TokenizerTestContext::new();
    let shared = Arc::new(SharedHandle(ctx.handle()));
    let encode_opts = no_bos();
    let decode_opts = talu_sys::DecodeOptionsC::default();
    let num_threads = 32;
    let iterations = 300;
    let mut handles = Vec::with_capacity(num_threads);

    for thread_id in 0..num_threads {
        let shared = Arc::clone(&shared);
        let h = thread::spawn(move || {
            for i in 0..iterations {
                let case = (thread_id + i) % 4;
                match case {
                    0 => {
                        let enc = unsafe { super::common::encode_raw(shared.ptr(), b"Hello", &encode_opts) };
                        assert!(enc.error_msg.is_null());
                        let ids = unsafe { std::slice::from_raw_parts(enc.ids, enc.num_tokens) };
                        assert_eq!(ids, &[44, 73, 80, 80, 83]);
                        unsafe { talu_sys::talu_encode_result_free(enc) };
                    }
                    1 => {
                        let dec = unsafe { super::common::decode_raw(shared.ptr(), &[44, 77], &decode_opts) };
                        assert!(dec.error_msg.is_null());
                        let text = unsafe {
                            std::str::from_utf8(std::slice::from_raw_parts(dec.text, dec.text_len))
                                .expect("decode must return valid UTF-8")
                        };
                        assert_eq!(text, "Hi");
                        unsafe { talu_sys::talu_decode_result_free(dec.text, dec.text_len) };
                    }
                    2 => {
                        let mut token: *mut i8 = std::ptr::null_mut();
                        let rc = unsafe {
                            talu_sys::talu_tokenizer_id_to_token(
                                shared.ptr(),
                                44,
                                &mut token as *mut _ as *mut std::ffi::c_void,
                            )
                        };
                        assert_eq!(rc, 0);
                        let text = unsafe { std::ffi::CStr::from_ptr(token) }.to_string_lossy();
                        assert_eq!(text, "H");
                        unsafe { talu_sys::talu_text_free(token) };
                    }
                    _ => {
                        let size = unsafe { talu_sys::talu_tokenizer_get_vocab_size(shared.ptr()) };
                        assert!(size >= 99);
                    }
                }
            }
        });
        handles.push(h);
    }

    for h in handles {
        h.join().expect("thread panicked");
    }
}
