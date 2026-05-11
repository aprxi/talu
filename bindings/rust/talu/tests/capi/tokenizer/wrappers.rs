//! Safe Rust tokenizer wrapper tests.
//!
//! The raw C API tests cover native boundary behavior. These tests lock down
//! the public `talu` wrapper contract over that same surface.

use crate::capi::tokenizer::common::{SPECIAL_TOKENS_TOKENIZER_JSON, TOKENIZER_JSON};
use std::ffi::{c_void, CStr, CString};
use std::fs;
use std::path::{Path, PathBuf};
use std::ptr;
use talu::{TokenizerEncodeOptions, TokenizerHandle, TokenizerTruncation, TokenizerTruncationSide};

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
