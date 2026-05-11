//! Safe Rust tokenizer wrapper tests.
//!
//! The raw C API tests cover native boundary behavior. These tests lock down
//! the public `talu` wrapper contract over that same surface.

use crate::capi::tokenizer::common::{SPECIAL_TOKENS_TOKENIZER_JSON, TOKENIZER_JSON};
use std::ffi::{c_void, CStr, CString};
use std::fs;
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::{Arc, Barrier};
use std::thread;
use talu::{TokenizerEncodeOptions, TokenizerHandle, TokenizerTruncation, TokenizerTruncationSide};

const TEMPLATE_POSTPROCESSOR_TOKENIZER_JSON: &str = r####"{
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

struct TempTokenizerModel {
    dir: tempfile::TempDir,
}

impl TempTokenizerModel {
    fn new(tokenizer_config: Option<&str>, generation_config: Option<&str>) -> Self {
        let dir = tempfile::tempdir().expect("temp tokenizer model dir must be created");
        fs::write(dir.path().join("tokenizer.json"), TOKENIZER_JSON)
            .expect("tokenizer.json must be written");

        if let Some(config) = tokenizer_config {
            fs::write(dir.path().join("tokenizer_config.json"), config)
                .expect("tokenizer_config.json must be written");
        }
        if let Some(config) = generation_config {
            fs::write(dir.path().join("generation_config.json"), config)
                .expect("generation_config.json must be written");
        }

        Self { dir }
    }

    fn path(&self) -> &Path {
        self.dir.path()
    }

    fn path_str(&self) -> &str {
        self.dir
            .path()
            .to_str()
            .expect("temp tokenizer model path must be valid UTF-8")
    }
}

fn create_capi_tokenizer(model_path: &Path) -> *mut c_void {
    let c_path = CString::new(
        model_path
            .to_str()
            .expect("temp tokenizer model path must be valid UTF-8"),
    )
    .expect("temp tokenizer model path must not contain NUL");
    let mut handle: *mut c_void = ptr::null_mut();

    // SAFETY: c_path is a live C string and handle is a valid out pointer.
    let rc = unsafe {
        talu_sys::talu_tokenizer_create(c_path.as_ptr(), &mut handle as *mut _ as *mut c_void)
    };
    assert_eq!(rc, 0, "file-loaded tokenizer creation must succeed");
    assert!(
        !handle.is_null(),
        "file-loaded tokenizer handle must be non-null"
    );
    handle
}

fn c_string_from_talu(out: *mut i8) -> String {
    assert!(!out.is_null(), "C API returned null text pointer");
    // SAFETY: out is non-null and owned by the C API result contract.
    let text = unsafe { CStr::from_ptr(out) }.to_string_lossy().to_string();
    // SAFETY: out was allocated by the C API and is freed exactly once here.
    unsafe { talu_sys::talu_text_free(out) };
    text
}

fn capi_model_dir(handle: *mut c_void) -> String {
    let mut out_path: *mut i8 = ptr::null_mut();
    // SAFETY: handle is a valid tokenizer handle and out_path is a valid out pointer.
    let rc = unsafe {
        talu_sys::talu_tokenizer_get_model_dir(handle, &mut out_path as *mut _ as *mut c_void)
    };
    assert_eq!(
        rc, 0,
        "get_model_dir must succeed for file-loaded tokenizer"
    );
    c_string_from_talu(out_path)
}

fn capi_eos_tokens(handle: *mut c_void) -> Vec<u32> {
    // SAFETY: handle is a valid tokenizer handle.
    let result = unsafe { talu_sys::talu_tokenizer_get_eos_tokens(handle) };
    let tokens = if result.tokens.is_null() || result.num_tokens == 0 {
        Vec::new()
    } else {
        // SAFETY: result.tokens points to result.num_tokens u32 values by C API contract.
        unsafe { std::slice::from_raw_parts(result.tokens, result.num_tokens) }.to_vec()
    };
    // SAFETY: result.tokens was allocated by the C API; null is an allowed no-op.
    unsafe { talu_sys::talu_tokens_free(result.tokens, result.num_tokens) };
    tokens
}

fn non_empty_error_text(error: talu::Error) -> String {
    let text = error.to_string();
    assert!(!text.is_empty(), "wrapper error text must not be empty");
    text
}

fn encode_with_template_options(
    tokenizer: &TokenizerHandle,
    text: &str,
    options: TokenizerEncodeOptions,
) -> Vec<u32> {
    tokenizer
        .encode_with_options(text, options)
        .expect("template encode must succeed")
        .ids
}

#[test]
fn wrapper_from_json_encode_decode_and_vocab_contract() {
    let tokenizer = TokenizerHandle::from_json(TOKENIZER_JSON).expect("JSON tokenizer must load");

    assert_eq!(tokenizer.encode("Hi").unwrap().tokens, vec![44, 77]);
    assert_eq!(tokenizer.decode(&[44, 77], false).unwrap(), "Hi");

    assert_eq!(tokenizer.vocab_size(), 99);
    let vocab = tokenizer.vocab().expect("vocab must be available");
    assert!(vocab
        .iter()
        .any(|entry| entry.token == "H" && entry.id == 44));
    assert!(vocab
        .iter()
        .any(|entry| entry.token == "<s>" && entry.id == 1));

    assert_eq!(tokenizer.token_to_id("<s>").unwrap(), 1);
    assert_eq!(tokenizer.id_to_token(44).unwrap(), "H");

    let special = tokenizer.special_tokens();
    assert_eq!(special.bos_token_id, 1);
    assert_eq!(special.unk_token_id, 3);
    assert_eq!(special.pad_token_id, 0);

    assert_eq!(tokenizer.model_dir(), None);
}

#[test]
fn wrapper_encode_with_options_returns_full_encoding_fields() {
    let tokenizer = TokenizerHandle::from_json(TOKENIZER_JSON).expect("JSON tokenizer must load");

    let right = tokenizer
        .encode_with_options(
            "Hello",
            TokenizerEncodeOptions {
                truncation: Some(TokenizerTruncation {
                    max_length: 2,
                    side: TokenizerTruncationSide::Right,
                }),
                ..Default::default()
            },
        )
        .expect("right truncation encode must succeed");
    assert_eq!(right.ids, vec![44, 73]);
    assert_eq!(right.attention_mask, vec![1, 1]);
    assert_eq!(right.special_tokens_mask, vec![0, 0]);
    assert_eq!(right.offsets, vec![[0, 1], [1, 2]]);

    let left = tokenizer
        .encode_with_options(
            "Hello",
            TokenizerEncodeOptions {
                truncation: Some(TokenizerTruncation {
                    max_length: 2,
                    side: TokenizerTruncationSide::Left,
                }),
                ..Default::default()
            },
        )
        .expect("left truncation encode must succeed");
    assert_eq!(left.ids, vec![80, 83]);
    assert_eq!(left.offsets, vec![[3, 4], [4, 5]]);
}

#[test]
fn wrapper_encode_options_add_bos_eos_are_forwarded() {
    let tokenizer = TokenizerHandle::from_json(TEMPLATE_POSTPROCESSOR_TOKENIZER_JSON)
        .expect("JSON tokenizer must load");

    assert_eq!(
        encode_with_template_options(&tokenizer, "Hi", TokenizerEncodeOptions::default()),
        vec![4, 5],
        "default wrapper options must not add template special tokens"
    );
    assert_eq!(
        encode_with_template_options(
            &tokenizer,
            "Hi",
            TokenizerEncodeOptions {
                add_bos: true,
                ..Default::default()
            },
        ),
        vec![1, 4, 5],
        "add_bos must prepend only BOS"
    );
    assert_eq!(
        encode_with_template_options(
            &tokenizer,
            "Hi",
            TokenizerEncodeOptions {
                add_eos: true,
                ..Default::default()
            },
        ),
        vec![4, 5, 2],
        "add_eos must append only EOS"
    );
    assert_eq!(
        encode_with_template_options(
            &tokenizer,
            "Hi",
            TokenizerEncodeOptions {
                add_bos: true,
                add_eos: true,
                ..Default::default()
            },
        ),
        vec![1, 4, 5, 2],
        "add_bos and add_eos together must apply the full template"
    );
    assert_eq!(
        encode_with_template_options(
            &tokenizer,
            "",
            TokenizerEncodeOptions {
                add_bos: true,
                ..Default::default()
            },
        ),
        vec![1],
        "empty input with add_bos must produce only BOS"
    );
    assert_eq!(
        encode_with_template_options(
            &tokenizer,
            "",
            TokenizerEncodeOptions {
                add_eos: true,
                ..Default::default()
            },
        ),
        vec![2],
        "empty input with add_eos must produce only EOS"
    );
}

#[test]
fn wrapper_batch_options_match_individual_encode() {
    let tokenizer = TokenizerHandle::from_json(TEMPLATE_POSTPROCESSOR_TOKENIZER_JSON)
        .expect("JSON tokenizer must load");
    let texts = vec!["Hi".to_string(), "H".to_string(), String::new()];
    let options = TokenizerEncodeOptions {
        add_bos: true,
        add_eos: true,
        truncation: Some(TokenizerTruncation {
            max_length: 3,
            side: TokenizerTruncationSide::Right,
        }),
    };

    let expected = texts
        .iter()
        .map(|text| encode_with_template_options(&tokenizer, text, options))
        .collect::<Vec<_>>();
    assert_eq!(expected, vec![vec![1, 4, 5], vec![1, 4, 2], vec![1, 2]]);

    let rows = tokenizer
        .encode_batch_ids(&texts, options)
        .expect("batch encode with wrapper options must succeed");
    assert_eq!(
        rows, expected,
        "batch wrapper options must match individual encode_with_options"
    );
}

#[test]
fn wrapper_encoding_fields_include_special_token_masks() {
    let tokenizer = TokenizerHandle::from_json(TEMPLATE_POSTPROCESSOR_TOKENIZER_JSON)
        .expect("JSON tokenizer must load");

    let encoding = tokenizer
        .encode_with_options(
            "Hi",
            TokenizerEncodeOptions {
                add_bos: true,
                add_eos: true,
                ..Default::default()
            },
        )
        .expect("template encode must succeed");

    assert_eq!(encoding.ids, vec![1, 4, 5, 2]);
    assert_eq!(encoding.attention_mask, vec![1, 1, 1, 1]);
    assert_eq!(encoding.special_tokens_mask, vec![1, 0, 0, 1]);
    assert_eq!(encoding.offsets, vec![[0, 0], [0, 1], [1, 2], [0, 0]]);
}

#[test]
fn wrapper_shared_handle_supports_concurrent_queries() {
    let tokenizer =
        Arc::new(TokenizerHandle::from_json(TOKENIZER_JSON).expect("JSON tokenizer must load"));
    let thread_count = 8;
    let barrier = Arc::new(Barrier::new(thread_count));
    let mut handles = Vec::with_capacity(thread_count);

    for thread_id in 0..thread_count {
        let tokenizer = Arc::clone(&tokenizer);
        let barrier = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            barrier.wait();
            for iter in 0..100 {
                assert_eq!(
                    tokenizer.encode("Hi").unwrap().tokens,
                    vec![44, 77],
                    "thread {thread_id} iter {iter}: encode mismatch"
                );
                assert_eq!(
                    tokenizer.decode(&[44, 77], false).unwrap(),
                    "Hi",
                    "thread {thread_id} iter {iter}: decode mismatch"
                );
                assert_eq!(
                    tokenizer.token_to_id("H").unwrap(),
                    44,
                    "thread {thread_id} iter {iter}: token_to_id mismatch"
                );
                assert_eq!(
                    tokenizer.id_to_token(77).unwrap(),
                    "i",
                    "thread {thread_id} iter {iter}: id_to_token mismatch"
                );
            }
        }));
    }

    for handle in handles {
        handle
            .join()
            .expect("concurrent wrapper query thread must not panic");
    }
}

#[test]
fn wrapper_batch_encode_returns_rows_and_handles_empty_inputs() {
    let tokenizer = TokenizerHandle::from_json(TOKENIZER_JSON).expect("JSON tokenizer must load");
    let texts = vec!["Hi".to_string(), "A".to_string(), String::new()];

    let rows = tokenizer
        .encode_batch_ids(&texts, TokenizerEncodeOptions::default())
        .expect("batch encode must succeed");

    assert_eq!(rows, vec![vec![44, 77], vec![37], Vec::<u32>::new()]);
    assert_eq!(
        tokenizer
            .encode_batch_ids(&[], TokenizerEncodeOptions::default())
            .unwrap(),
        Vec::<Vec<u32>>::new()
    );
}

#[test]
fn wrapper_decode_skip_special_tokens_uses_decode_options() {
    let tokenizer = TokenizerHandle::from_json(SPECIAL_TOKENS_TOKENIZER_JSON)
        .expect("JSON tokenizer must load");

    assert_eq!(
        tokenizer.decode(&[1, 44, 77, 2], false).unwrap(),
        "<s>Hi</s>"
    );
    assert_eq!(tokenizer.decode(&[1, 44, 77, 2], true).unwrap(), "Hi");
}

#[test]
fn wrapper_error_paths_return_result_errors_with_context() {
    let invalid_json = TokenizerHandle::from_json("not valid json").unwrap_err();
    assert!(
        non_empty_error_text(invalid_json).contains("Failed to create tokenizer from JSON"),
        "invalid JSON error should preserve C-side context"
    );

    let tokenizer = TokenizerHandle::from_json(TOKENIZER_JSON).expect("JSON tokenizer must load");
    assert!(
        non_empty_error_text(tokenizer.token_to_id("missing-token").unwrap_err())
            .contains("missing-token"),
        "unknown token error should include the requested token"
    );
    assert!(
        non_empty_error_text(tokenizer.id_to_token(9999).unwrap_err()).contains("9999"),
        "unknown id error should include the requested id"
    );
}

#[test]
fn wrapper_new_error_paths_are_diagnosable() {
    let temp = tempfile::tempdir().expect("temp dir must be created");
    let missing_path = temp.path().join("missing-tokenizer");
    let missing_path_text = missing_path
        .to_str()
        .expect("temp missing path must be valid UTF-8");
    let missing_error = non_empty_error_text(TokenizerHandle::new(missing_path_text).unwrap_err());
    assert!(
        missing_error.contains("failed to load tokenizer"),
        "missing path error should name the failed operation: {missing_error}"
    );
    assert!(
        missing_error.contains(missing_path_text),
        "missing path error should include the requested path: {missing_error}"
    );

    let empty_dir = tempfile::tempdir().expect("empty model dir must be created");
    let empty_dir_text = empty_dir
        .path()
        .to_str()
        .expect("temp model dir must be valid UTF-8");
    let empty_dir_error = non_empty_error_text(TokenizerHandle::new(empty_dir_text).unwrap_err());
    assert!(
        empty_dir_error.contains("failed to load tokenizer"),
        "empty model dir error should name the failed operation: {empty_dir_error}"
    );
    assert!(
        empty_dir_error.contains(empty_dir_text),
        "empty model dir error should include the requested path: {empty_dir_error}"
    );

    let nul_error = non_empty_error_text(TokenizerHandle::new("bad\0path").unwrap_err());
    assert!(
        nul_error.contains("interior NUL"),
        "interior-NUL path error should identify the invalid path shape: {nul_error}"
    );
}

#[test]
fn capi_file_loaded_tokenizer_reports_model_dir_max_length_and_special_overrides() {
    let model = TempTokenizerModel::new(
        Some(r#"{"model_max_length": 1234}"#),
        Some(r#"{"eos_token_id": [2, 44], "bos_token_id": 44, "pad_token_id": 4}"#),
    );
    let handle = create_capi_tokenizer(model.path());

    let reported_dir = PathBuf::from(capi_model_dir(handle));
    assert_eq!(
        fs::canonicalize(reported_dir).unwrap(),
        fs::canonicalize(model.path()).unwrap(),
        "get_model_dir must report the resolved model directory"
    );

    // SAFETY: handle is a valid tokenizer handle.
    let max_len = unsafe { talu_sys::talu_tokenizer_get_model_max_length(handle) };
    assert_eq!(max_len, 1234);
    assert_eq!(capi_eos_tokens(handle), vec![2, 44]);

    // SAFETY: handle is a valid tokenizer handle.
    let special = unsafe { talu_sys::talu_tokenizer_get_special_tokens(handle) };
    assert_eq!(special.bos_token_id, 44);
    assert_eq!(special.unk_token_id, 3);
    assert_eq!(special.pad_token_id, 4);

    // SAFETY: handle was returned by talu_tokenizer_create and is freed exactly once here.
    unsafe { talu_sys::talu_tokenizer_free(handle) };
}

#[test]
fn capi_file_loaded_tokenizer_reports_single_eos_id() {
    let model = TempTokenizerModel::new(None, Some(r#"{"eos_token_id": 2}"#));
    let handle = create_capi_tokenizer(model.path());

    assert_eq!(capi_eos_tokens(handle), vec![2]);

    // SAFETY: handle was returned by talu_tokenizer_create and is freed exactly once here.
    unsafe { talu_sys::talu_tokenizer_free(handle) };
}

#[test]
fn wrapper_new_loads_file_tokenizer_metadata_and_encodes() {
    let model = TempTokenizerModel::new(
        Some(r#"{"model_max_length": 512}"#),
        Some(r#"{"eos_token_id": [2], "bos_token_id": 44, "pad_token_id": 4}"#),
    );
    let tokenizer = TokenizerHandle::new(model.path_str()).expect("file tokenizer must load");

    assert_eq!(tokenizer.encode("Hi").unwrap().tokens, vec![44, 77]);
    assert_eq!(tokenizer.decode(&[44, 77], false).unwrap(), "Hi");
    assert_eq!(
        fs::canonicalize(
            tokenizer
                .model_dir()
                .expect("file tokenizer must expose model dir")
        )
        .unwrap(),
        fs::canonicalize(model.path()).unwrap()
    );

    let special = tokenizer.special_tokens();
    assert_eq!(special.bos_token_id, 44);
    assert_eq!(special.unk_token_id, 3);
    assert_eq!(special.pad_token_id, 4);
}
