//! Streaming iterator tests.
//!
//! Exercises the pull-based token iterator API. Null-safety tests run
//! unconditionally; actual streaming tests require `TALU_TEST_MODEL`.

use crate::capi::router::common;
use crate::capi::router::common::skip_without_model;
use std::ffi::CString;
use std::ptr;

// ---------------------------------------------------------------------------
// Null safety (no model needed)
// ---------------------------------------------------------------------------

#[test]
fn iterator_null_chat_returns_null() {
    let config = talu_sys::CGenerateConfig::default();
    let iter = unsafe {
        talu_sys::talu_router_create_iterator(
            ptr::null_mut(), // null chat
            ptr::null(),
            0,
            ptr::null_mut(),
            &config,
        )
    };
    assert!(iter.is_null(), "iterator with null chat should be null");
}

#[test]
fn iterator_null_backend_returns_null() {
    let chat = common::create_chat(None);
    let config = talu_sys::CGenerateConfig::default();
    let iter = unsafe {
        talu_sys::talu_router_create_iterator(
            chat,
            ptr::null(),
            0,
            ptr::null_mut(), // null backend
            &config,
        )
    };
    assert!(iter.is_null(), "iterator with null backend should be null");
    unsafe { talu_sys::talu_chat_free(chat) };
}

#[test]
fn iterator_free_null_is_noop() {
    unsafe { talu_sys::talu_router_iterator_free(ptr::null_mut()) };
}

#[test]
fn iterator_error_accessors_safe_on_null() {
    assert!(
        !unsafe { talu_sys::talu_router_iterator_has_error(ptr::null_mut()) },
        "has_error on null should return false"
    );
    let code = unsafe { talu_sys::talu_router_iterator_error_code(ptr::null_mut()) };
    assert_eq!(code, -1, "error_code on null should return -1");
    let msg = unsafe { talu_sys::talu_router_iterator_error_msg(ptr::null_mut()) };
    assert!(msg.is_null(), "error_msg on null should return null");
}

#[test]
fn iterator_stat_accessors_safe_on_null() {
    assert_eq!(
        unsafe { talu_sys::talu_router_iterator_prompt_tokens(ptr::null_mut()) },
        0
    );
    assert_eq!(
        unsafe { talu_sys::talu_router_iterator_completion_tokens(ptr::null_mut()) },
        0
    );
    assert_eq!(
        unsafe { talu_sys::talu_router_iterator_prefill_ns(ptr::null_mut()) },
        0
    );
    assert_eq!(
        unsafe { talu_sys::talu_router_iterator_generation_ns(ptr::null_mut()) },
        0
    );
}

#[test]
fn iterator_cancel_null_is_noop() {
    unsafe { talu_sys::talu_router_iterator_cancel(ptr::null_mut()) };
}

// ---------------------------------------------------------------------------
// Model-gated tests
// ---------------------------------------------------------------------------

#[test]
fn create_iterator_and_drain() {
    skip_without_model!();
    let model = common::model_path().unwrap();

    let (canon, backend) = common::local_backend(&model);
    let chat = common::create_chat(Some("You are a helpful assistant."));
    common::append_user_message(chat, "Say hi.");

    let mut config = talu_sys::CGenerateConfig::default();
    config.max_tokens = 16;
    config.temperature = 0.0;

    let prompt = CString::new("Say hi.").unwrap();
    let part = talu_sys::GenerateContentPart {
        content_type: 0,
        data_ptr: prompt.as_ptr() as *const u8,
        data_len: prompt.as_bytes().len(),
        mime_ptr: ptr::null(),
    };

    let iter =
        unsafe { talu_sys::talu_router_create_iterator(chat, &part, 1, backend, &config) };
    assert!(!iter.is_null(), "iterator creation should succeed");

    let mut token_count = 0usize;
    loop {
        let token_ptr = unsafe { talu_sys::talu_router_iterator_next(iter) };
        if token_ptr.is_null() {
            break;
        }
        token_count += 1;
    }

    assert!(
        !unsafe { talu_sys::talu_router_iterator_has_error(iter) },
        "iterator should not have error after clean drain"
    );
    assert!(token_count > 0, "should have received at least one token");
    assert!(
        unsafe { talu_sys::talu_router_iterator_completion_tokens(iter) } > 0,
        "completion_tokens should be positive after drain"
    );

    unsafe { talu_sys::talu_router_iterator_free(iter) };
    unsafe { talu_sys::talu_chat_free(chat) };
    unsafe { talu_sys::talu_backend_free(backend) };
    unsafe { talu_sys::talu_config_free(canon) };
}

#[test]
fn iterator_stats_populated_after_drain() {
    skip_without_model!();
    let model = common::model_path().unwrap();

    let (canon, backend) = common::local_backend(&model);
    let chat = common::create_chat(None);

    let mut config = talu_sys::CGenerateConfig::default();
    config.max_tokens = 8;
    config.temperature = 0.0;

    let prompt = CString::new("Hello").unwrap();
    let part = talu_sys::GenerateContentPart {
        content_type: 0,
        data_ptr: prompt.as_ptr() as *const u8,
        data_len: prompt.as_bytes().len(),
        mime_ptr: ptr::null(),
    };

    let iter =
        unsafe { talu_sys::talu_router_create_iterator(chat, &part, 1, backend, &config) };
    assert!(!iter.is_null());

    // Drain all tokens.
    while !unsafe { talu_sys::talu_router_iterator_next(iter) }.is_null() {}

    let prompt_tokens = unsafe { talu_sys::talu_router_iterator_prompt_tokens(iter) };
    let completion_tokens = unsafe { talu_sys::talu_router_iterator_completion_tokens(iter) };
    let prefill_ns = unsafe { talu_sys::talu_router_iterator_prefill_ns(iter) };
    let generation_ns = unsafe { talu_sys::talu_router_iterator_generation_ns(iter) };

    assert!(prompt_tokens > 0, "prompt_tokens should be positive");
    assert!(
        completion_tokens > 0,
        "completion_tokens should be positive"
    );
    assert!(prefill_ns > 0, "prefill_ns should be positive");
    assert!(generation_ns > 0, "generation_ns should be positive");

    unsafe { talu_sys::talu_router_iterator_free(iter) };
    unsafe { talu_sys::talu_chat_free(chat) };
    unsafe { talu_sys::talu_backend_free(backend) };
    unsafe { talu_sys::talu_config_free(canon) };
}

#[test]
fn iterator_cancel_stops_generation() {
    skip_without_model!();
    let model = common::model_path().unwrap();

    let (canon, backend) = common::local_backend(&model);
    let chat = common::create_chat(None);

    let mut config = talu_sys::CGenerateConfig::default();
    config.max_tokens = 256; // large limit so we can cancel mid-stream
    config.temperature = 0.0;

    let prompt = CString::new("Write a long essay about the history of computing.").unwrap();
    let part = talu_sys::GenerateContentPart {
        content_type: 0,
        data_ptr: prompt.as_ptr() as *const u8,
        data_len: prompt.as_bytes().len(),
        mime_ptr: ptr::null(),
    };

    let iter =
        unsafe { talu_sys::talu_router_create_iterator(chat, &part, 1, backend, &config) };
    assert!(!iter.is_null());

    // Read a few tokens then cancel.
    let mut got_at_least_one = false;
    for _ in 0..3 {
        let ptr = unsafe { talu_sys::talu_router_iterator_next(iter) };
        if ptr.is_null() {
            break;
        }
        got_at_least_one = true;
    }

    unsafe { talu_sys::talu_router_iterator_cancel(iter) };

    // Drain remaining tokens after cancel.
    while !unsafe { talu_sys::talu_router_iterator_next(iter) }.is_null() {}

    let finish_reason = unsafe { talu_sys::talu_router_iterator_finish_reason(iter) };
    // finish_reason 5 = cancelled (if we managed to cancel), or other if generation
    // finished before cancel took effect. Either is acceptable.
    if got_at_least_one {
        assert!(
            finish_reason == 5 || finish_reason == 0 || finish_reason == 1,
            "finish_reason should be cancelled(5), eos(0), or length(1), got {}",
            finish_reason,
        );
    }

    unsafe { talu_sys::talu_router_iterator_free(iter) };
    unsafe { talu_sys::talu_chat_free(chat) };
    unsafe { talu_sys::talu_backend_free(backend) };
    unsafe { talu_sys::talu_config_free(canon) };
}

#[test]
fn iterator_content_type_is_valid() {
    skip_without_model!();
    let model = common::model_path().unwrap();

    let (canon, backend) = common::local_backend(&model);
    let chat = common::create_chat(None);

    let mut config = talu_sys::CGenerateConfig::default();
    config.max_tokens = 8;
    config.temperature = 0.0;

    let prompt = CString::new("Hello").unwrap();
    let part = talu_sys::GenerateContentPart {
        content_type: 0,
        data_ptr: prompt.as_ptr() as *const u8,
        data_len: prompt.as_bytes().len(),
        mime_ptr: ptr::null(),
    };

    let iter =
        unsafe { talu_sys::talu_router_create_iterator(chat, &part, 1, backend, &config) };
    assert!(!iter.is_null());

    let token_ptr = unsafe { talu_sys::talu_router_iterator_next(iter) };
    if !token_ptr.is_null() {
        let item_type = unsafe { talu_sys::talu_router_iterator_item_type(iter) };
        let content_type = unsafe { talu_sys::talu_router_iterator_content_type(iter) };

        // Valid item types: 0=message, 1=function_call, 3=reasoning, 255=unknown
        assert!(
            matches!(item_type, 0 | 1 | 3 | 255),
            "unexpected item_type: {}",
            item_type
        );
        // Valid content types: 5=output_text, 8=reasoning_text, 255=unknown
        assert!(
            matches!(content_type, 5 | 8 | 255),
            "unexpected content_type: {}",
            content_type
        );
    }

    // Drain remaining.
    while !unsafe { talu_sys::talu_router_iterator_next(iter) }.is_null() {}

    unsafe { talu_sys::talu_router_iterator_free(iter) };
    unsafe { talu_sys::talu_chat_free(chat) };
    unsafe { talu_sys::talu_backend_free(backend) };
    unsafe { talu_sys::talu_config_free(canon) };
}
