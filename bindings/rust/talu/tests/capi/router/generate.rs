//! Generation tests.
//!
//! Null-safety tests run unconditionally. Tests that perform actual inference
//! require `TALU_TEST_MODEL` to point to a local model file.

use crate::capi::router::common;
use crate::capi::router::common::skip_without_model;
use std::ffi::{c_char, CStr, CString};
use std::ptr;

// ---------------------------------------------------------------------------
// Null safety (no model needed)
// ---------------------------------------------------------------------------

#[test]
fn generate_null_chat_returns_error() {
    let config = talu_sys::CGenerateConfig::default();
    let result = unsafe {
        talu_sys::talu_router_generate_with_backend(
            ptr::null_mut(), // null chat
            ptr::null(),
            0,
            ptr::null_mut(),
            &config,
        )
    };
    assert_ne!(result.error_code, 0, "generate with null chat should fail");
}

#[test]
fn generate_null_backend_returns_error() {
    let chat = common::create_chat(Some("You are helpful."));
    let config = talu_sys::CGenerateConfig::default();
    let result = unsafe {
        talu_sys::talu_router_generate_with_backend(
            chat,
            ptr::null(),
            0,
            ptr::null_mut(), // null backend
            &config,
        )
    };
    assert_ne!(
        result.error_code, 0,
        "generate with null backend should fail"
    );
    unsafe { talu_sys::talu_chat_free(chat) };
}

#[test]
fn result_free_null_is_noop() {
    unsafe { talu_sys::talu_router_result_free(ptr::null_mut()) };
}

// ---------------------------------------------------------------------------
// Model-gated tests
// ---------------------------------------------------------------------------

#[test]
fn generate_text_produces_output() {
    skip_without_model!();
    let model = common::model_path().unwrap();

    let (canon, backend) = common::local_backend(&model);
    let chat = common::create_chat(Some("You are a helpful assistant."));
    common::append_user_message(chat, "Say hello in one word.");

    let mut config = talu_sys::CGenerateConfig::default();
    config.max_tokens = 32;
    config.temperature = 0.0;
    config.seed = 42;

    let text_content = CString::new("Say hello in one word.").unwrap();
    let part = talu_sys::GenerateContentPart {
        content_type: 0, // Text
        data_ptr: text_content.as_ptr() as *const u8,
        data_len: text_content.as_bytes().len(),
        mime_ptr: ptr::null(),
    };

    let mut result = unsafe {
        talu_sys::talu_router_generate_with_backend(chat, &part, 1, backend, &config)
    };
    assert_eq!(result.error_code, 0, "generation should succeed");
    assert!(!result.text.is_null(), "generated text should be non-null");
    assert!(
        result.prompt_tokens > 0,
        "prompt_tokens should be positive"
    );
    assert!(
        result.completion_tokens > 0,
        "completion_tokens should be positive"
    );

    unsafe { talu_sys::talu_router_result_free(&mut result) };
    unsafe { talu_sys::talu_chat_free(chat) };
    unsafe { talu_sys::talu_backend_free(backend) };
    unsafe { talu_sys::talu_config_free(canon) };
}

#[test]
fn generate_with_max_tokens_respects_limit() {
    skip_without_model!();
    let model = common::model_path().unwrap();

    let (canon, backend) = common::local_backend(&model);
    let chat = common::create_chat(None);

    let mut config = talu_sys::CGenerateConfig::default();
    config.max_tokens = 1;
    config.temperature = 0.0;

    let text = CString::new("Count to ten:").unwrap();
    let part = talu_sys::GenerateContentPart {
        content_type: 0,
        data_ptr: text.as_ptr() as *const u8,
        data_len: text.as_bytes().len(),
        mime_ptr: ptr::null(),
    };

    let mut result =
        unsafe { talu_sys::talu_router_generate_with_backend(chat, &part, 1, backend, &config) };

    if result.error_code == 0 {
        // With max_tokens=1, we should get at most a very small number of tokens.
        assert!(
            result.completion_tokens <= 2,
            "completion_tokens ({}) should respect max_tokens=1",
            result.completion_tokens,
        );
    }

    unsafe { talu_sys::talu_router_result_free(&mut result) };
    unsafe { talu_sys::talu_chat_free(chat) };
    unsafe { talu_sys::talu_backend_free(backend) };
    unsafe { talu_sys::talu_config_free(canon) };
}

#[test]
fn generate_with_seed_is_deterministic() {
    skip_without_model!();
    let model = common::model_path().unwrap();

    let mut texts = Vec::new();

    for _ in 0..2 {
        let (canon, backend) = common::local_backend(&model);
        let chat = common::create_chat(None);

        let mut config = talu_sys::CGenerateConfig::default();
        config.max_tokens = 16;
        config.temperature = 0.5;
        config.seed = 12345;

        let prompt = CString::new("The capital of France is").unwrap();
        let part = talu_sys::GenerateContentPart {
            content_type: 0,
            data_ptr: prompt.as_ptr() as *const u8,
            data_len: prompt.as_bytes().len(),
            mime_ptr: ptr::null(),
        };

        let mut result = unsafe {
            talu_sys::talu_router_generate_with_backend(chat, &part, 1, backend, &config)
        };

        if result.error_code == 0 && !result.text.is_null() {
            let text = unsafe { CStr::from_ptr(result.text) }
                .to_string_lossy()
                .to_string();
            texts.push(text);
        }

        unsafe { talu_sys::talu_router_result_free(&mut result) };
        unsafe { talu_sys::talu_chat_free(chat) };
        unsafe { talu_sys::talu_backend_free(backend) };
        unsafe { talu_sys::talu_config_free(canon) };
    }

    assert_eq!(texts.len(), 2, "both generations should succeed");
    assert_eq!(
        texts[0], texts[1],
        "same seed should produce identical output"
    );
}

#[test]
fn generate_with_stop_sequence_halts_output() {
    skip_without_model!();
    let model = common::model_path().unwrap();

    let (canon, backend) = common::local_backend(&model);
    let chat = common::create_chat(None);

    // Use a stop sequence that should appear in a counting task.
    let stop = CString::new("3").unwrap();
    let stop_ptrs: [*const c_char; 1] = [stop.as_ptr()];

    let mut config = talu_sys::CGenerateConfig::default();
    config.max_tokens = 64;
    config.temperature = 0.0;
    config.seed = 42;
    config.stop_sequences = stop_ptrs.as_ptr();
    config.stop_sequence_count = 1;

    let text = CString::new("Count from 1 to 10: 1, 2,").unwrap();
    let part = talu_sys::GenerateContentPart {
        content_type: 0,
        data_ptr: text.as_ptr() as *const u8,
        data_len: text.as_bytes().len(),
        mime_ptr: ptr::null(),
    };

    let mut result =
        unsafe { talu_sys::talu_router_generate_with_backend(chat, &part, 1, backend, &config) };

    assert_eq!(result.error_code, 0, "generation should succeed");
    if !result.text.is_null() {
        let output = unsafe { CStr::from_ptr(result.text) }
            .to_string_lossy()
            .to_string();
        // The output should not contain numbers past the stop sequence.
        // "3" is the stop — output should be short and stop before "4, 5, ..."
        assert!(
            !output.contains("5"),
            "stop sequence should prevent output from reaching '5', got: {:?}",
            output,
        );
    }

    unsafe { talu_sys::talu_router_result_free(&mut result) };
    unsafe { talu_sys::talu_chat_free(chat) };
    unsafe { talu_sys::talu_backend_free(backend) };
    unsafe { talu_sys::talu_config_free(canon) };
}

#[test]
fn generate_with_logit_bias_influences_output() {
    skip_without_model!();
    let model = common::model_path().unwrap();

    let (canon, backend) = common::local_backend(&model);
    let chat = common::create_chat(None);

    // Apply a strong negative bias to suppress common tokens. The exact token
    // IDs are model-specific, but the point is verifying the logit_bias field
    // is correctly marshalled across the FFI boundary without crashing.
    let bias_entries = [
        talu_sys::CLogitBiasEntry {
            token_id: 1,
            bias: -100.0,
        },
        talu_sys::CLogitBiasEntry {
            token_id: 2,
            bias: -100.0,
        },
    ];

    let mut config = talu_sys::CGenerateConfig::default();
    config.max_tokens = 8;
    config.temperature = 0.0;
    config.seed = 42;
    config.logit_bias = bias_entries.as_ptr();
    config.logit_bias_count = bias_entries.len();

    let text = CString::new("Hello").unwrap();
    let part = talu_sys::GenerateContentPart {
        content_type: 0,
        data_ptr: text.as_ptr() as *const u8,
        data_len: text.as_bytes().len(),
        mime_ptr: ptr::null(),
    };

    let mut result =
        unsafe { talu_sys::talu_router_generate_with_backend(chat, &part, 1, backend, &config) };

    // The generation should succeed (logit bias is a soft constraint, not rejection).
    assert_eq!(
        result.error_code, 0,
        "generation with logit_bias should succeed"
    );
    assert!(
        !result.text.is_null(),
        "generation with logit_bias should produce output"
    );

    unsafe { talu_sys::talu_router_result_free(&mut result) };
    unsafe { talu_sys::talu_chat_free(chat) };
    unsafe { talu_sys::talu_backend_free(backend) };
    unsafe { talu_sys::talu_config_free(canon) };
}

#[test]
fn generate_with_multiple_text_parts() {
    skip_without_model!();
    let model = common::model_path().unwrap();

    let (canon, backend) = common::local_backend(&model);
    let chat = common::create_chat(Some("You are a helpful assistant."));
    common::append_user_message(chat, "Combine these.");

    let mut config = talu_sys::CGenerateConfig::default();
    config.max_tokens = 32;
    config.temperature = 0.0;
    config.seed = 42;

    let text1 = CString::new("Hello ").unwrap();
    let text2 = CString::new("world.").unwrap();
    let parts = [
        talu_sys::GenerateContentPart {
            content_type: 0, // Text
            data_ptr: text1.as_ptr() as *const u8,
            data_len: text1.as_bytes().len(),
            mime_ptr: ptr::null(),
        },
        talu_sys::GenerateContentPart {
            content_type: 0, // Text
            data_ptr: text2.as_ptr() as *const u8,
            data_len: text2.as_bytes().len(),
            mime_ptr: ptr::null(),
        },
    ];

    let mut result = unsafe {
        talu_sys::talu_router_generate_with_backend(
            chat,
            parts.as_ptr(),
            parts.len(),
            backend,
            &config,
        )
    };

    assert_eq!(
        result.error_code, 0,
        "generation with multiple text parts should succeed"
    );
    assert!(
        !result.text.is_null(),
        "multi-part generation should produce output"
    );
    assert!(
        result.prompt_tokens > 0,
        "prompt_tokens should be positive for multi-part input"
    );

    unsafe { talu_sys::talu_router_result_free(&mut result) };
    unsafe { talu_sys::talu_chat_free(chat) };
    unsafe { talu_sys::talu_backend_free(backend) };
    unsafe { talu_sys::talu_config_free(canon) };
}

#[test]
fn generate_with_image_part_marshals_correctly() {
    skip_without_model!();
    let model = common::model_path().unwrap();

    let (canon, backend) = common::local_backend(&model);
    let chat = common::create_chat(None);

    let mut config = talu_sys::CGenerateConfig::default();
    config.max_tokens = 8;
    config.temperature = 0.0;

    // Dummy 2x2 RGBA PNG-like bytes. The model likely doesn't support vision,
    // but this verifies that content_type=1 (image) with a non-null mime_ptr
    // and data_ptr is marshalled across the FFI boundary without segfault.
    let dummy_image: [u8; 16] = [0xFF; 16];
    let mime = CString::new("image/png").unwrap();
    let text_prompt = CString::new("Describe the image.").unwrap();

    let parts = [
        talu_sys::GenerateContentPart {
            content_type: 1, // Image
            data_ptr: dummy_image.as_ptr(),
            data_len: dummy_image.len(),
            mime_ptr: mime.as_ptr(),
        },
        // Include a text part so the model has something to work with if it
        // ignores the image gracefully.
        talu_sys::GenerateContentPart {
            content_type: 0, // Text
            data_ptr: text_prompt.as_ptr() as *const u8,
            data_len: text_prompt.as_bytes().len(),
            mime_ptr: ptr::null(),
        },
    ];

    let mut result = unsafe {
        talu_sys::talu_router_generate_with_backend(
            chat,
            parts.as_ptr(),
            parts.len(),
            backend,
            &config,
        )
    };

    // The model may or may not support vision. Either a successful generation
    // or a graceful error is acceptable — the key is no segfault/UB from the
    // image content_type + mime_ptr marshalling.
    if result.error_code == 0 {
        assert!(
            !result.text.is_null(),
            "successful image generation should produce output"
        );
    }

    unsafe { talu_sys::talu_router_result_free(&mut result) };
    unsafe { talu_sys::talu_chat_free(chat) };
    unsafe { talu_sys::talu_backend_free(backend) };
    unsafe { talu_sys::talu_config_free(canon) };
}
