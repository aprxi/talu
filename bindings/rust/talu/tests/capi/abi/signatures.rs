//! Compile-time FFI signature validation.
//!
//! These tests verify that auto-generated FFI bindings have correct signatures
//! for active inference surfaces.

use std::ffi::c_void;
use std::os::raw::{c_char, c_int};
use talu_sys::{CBatchConfig, CBatchEvent, GenerationConfigInfo, ResponsesHandle};

/// Assert that a function has the expected signature at compile time.
macro_rules! assert_fn_signature {
    ($fn_name:path, $expected_type:ty) => {
        const _: $expected_type = $fn_name;
    };
}

// Generation config (model_dir, out_config)
assert_fn_signature!(
    talu_sys::talu_get_generation_config,
    unsafe extern "C" fn(*const c_char, *mut GenerationConfigInfo) -> c_int
);

// Chat APIs
assert_fn_signature!(
    talu_sys::talu_chat_get_conversation,
    unsafe extern "C" fn(*mut c_void) -> *mut ResponsesHandle
);
assert_fn_signature!(
    talu_sys::talu_chat_get_system,
    unsafe extern "C" fn(*mut c_void) -> *const c_char
);
assert_fn_signature!(
    talu_sys::talu_chat_set_system,
    unsafe extern "C" fn(*mut c_void, *const c_char) -> c_int
);
assert_fn_signature!(
    talu_sys::talu_chat_to_json,
    unsafe extern "C" fn(*mut c_void) -> *const c_char
);
assert_fn_signature!(
    talu_sys::talu_chat_set_messages,
    unsafe extern "C" fn(*mut c_void, *const c_char) -> c_int
);

// Responses APIs
assert_fn_signature!(
    talu_sys::talu_responses_append_message,
    unsafe extern "C" fn(*mut ResponsesHandle, u8, *const u8, usize) -> i64
);
assert_fn_signature!(
    talu_sys::talu_batch_create,
    unsafe extern "C" fn(*mut c_void, *const CBatchConfig) -> *mut c_void
);
assert_fn_signature!(
    talu_sys::talu_batch_step,
    unsafe extern "C" fn(*mut c_void, *mut CBatchEvent, usize) -> usize
);
assert_fn_signature!(
    talu_sys::talu_take_last_error,
    unsafe extern "C" fn(*mut u8, usize, *mut c_void) -> usize
);

#[test]
fn active_ffi_symbols_are_present() {
    let chat_to_json_ptr = talu_sys::talu_chat_to_json as *const () as usize;
    let append_message_ptr = talu_sys::talu_responses_append_message as *const () as usize;
    let gen_cfg_ptr = talu_sys::talu_get_generation_config as *const () as usize;
    let batch_create_ptr = talu_sys::talu_batch_create as *const () as usize;

    assert_ne!(
        chat_to_json_ptr, 0,
        "talu_chat_to_json should be a valid function"
    );
    assert_ne!(
        append_message_ptr, 0,
        "talu_responses_append_message should be a valid function"
    );
    assert_ne!(
        gen_cfg_ptr, 0,
        "talu_get_generation_config should be a valid function"
    );
    assert_ne!(
        batch_create_ptr, 0,
        "talu_batch_create should be a valid function"
    );
}

// CGenerateConfig default values must match Zig sentinel conventions.
#[test]
fn generate_config_defaults_match_zig_sentinels() {
    let cfg = talu_sys::CGenerateConfig::default();

    assert_eq!(
        cfg.max_reasoning_tokens,
        usize::MAX,
        "max_reasoning_tokens must default to usize::MAX (unset sentinel)"
    );
    assert_eq!(
        cfg.temperature, -1.0_f32,
        "temperature must default to -1.0 (unset sentinel)"
    );
    assert_eq!(
        cfg.top_p, -1.0_f32,
        "top_p must default to -1.0 (unset sentinel)"
    );
    assert_eq!(
        cfg.min_p, -1.0_f32,
        "min_p must default to -1.0 (unset sentinel)"
    );
    assert_eq!(
        cfg.repetition_penalty, -1.0_f32,
        "repetition_penalty must default to -1.0 (unset sentinel)"
    );
    assert_eq!(
        cfg.presence_penalty, -1.0_f32,
        "presence_penalty must default to -1.0 (unset sentinel)"
    );
    assert_eq!(
        cfg.frequency_penalty, -1.0_f32,
        "frequency_penalty must default to -1.0 (unset sentinel)"
    );
    assert_eq!(
        cfg.completions_mode, 0,
        "completions_mode must default to 0 (off)"
    );
    assert_eq!(cfg.raw_output, 0, "raw_output must default to 0 (off)");
}
