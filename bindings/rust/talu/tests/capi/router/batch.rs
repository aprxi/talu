//! Batch C-API tests.
//!
//! Exercises `talu_batch_*` lifecycle, null-safety, and submit/step
//! semantics. Null-safety tests run unconditionally; inference tests
//! require `TALU_TEST_MODEL`.

use crate::capi::router::common;
use crate::capi::router::common::skip_without_model;
use std::ffi::CString;
use std::ptr;

// =============================================================================
// Null-safety (run unconditionally)
// =============================================================================

#[test]
fn create_null_backend_returns_null() {
    let handle = unsafe { talu_sys::talu_batch_create(ptr::null_mut(), ptr::null()) };
    assert!(
        handle.is_null(),
        "create with null backend should return null"
    );
}

#[test]
fn destroy_null_is_noop() {
    unsafe { talu_sys::talu_batch_destroy(ptr::null_mut()) };
}

#[test]
fn submit_null_handle_returns_zero() {
    let id = unsafe { talu_sys::talu_batch_submit(ptr::null_mut(), ptr::null_mut(), ptr::null()) };
    assert_eq!(id, 0, "submit with null handle should return 0");
}

#[test]
fn submit_null_chat_returns_zero() {
    // We can't easily get a valid batch handle without a model,
    // but null chat should be caught before null handle.
    let id = unsafe { talu_sys::talu_batch_submit(ptr::null_mut(), ptr::null_mut(), ptr::null()) };
    assert_eq!(id, 0, "submit with null chat should return 0");
}

#[test]
fn cancel_null_handle_returns_zero() {
    let rc = unsafe { talu_sys::talu_batch_cancel(ptr::null_mut(), 1) };
    assert_eq!(rc, 0, "cancel with null handle should return 0");
}

#[test]
fn step_null_handle_returns_zero() {
    let mut events = [talu_sys::CBatchEvent::default(); 8];
    let n =
        unsafe { talu_sys::talu_batch_step(ptr::null_mut(), events.as_mut_ptr(), events.len()) };
    assert_eq!(n, 0, "step with null handle should return 0");
}

#[test]
fn has_active_null_handle_returns_zero() {
    let rc = unsafe { talu_sys::talu_batch_has_active(ptr::null_mut()) };
    assert_eq!(rc, 0, "has_active with null handle should return 0");
}

#[test]
fn active_count_null_handle_returns_zero() {
    let n = unsafe { talu_sys::talu_batch_active_count(ptr::null_mut()) };
    assert_eq!(n, 0, "active_count with null handle should return 0");
}

#[test]
fn take_result_null_handle_returns_null() {
    let r = unsafe { talu_sys::talu_batch_take_result(ptr::null_mut(), 1) };
    assert!(
        r.is_null(),
        "take_result with null handle should return null"
    );
}

#[test]
fn result_free_null_is_noop() {
    unsafe { talu_sys::talu_batch_result_free(ptr::null_mut()) };
}

#[test]
fn create_non_local_backend_returns_null() {
    let c_model = CString::new("gpt-4").unwrap();
    let c_url = CString::new("https://api.openai.com/v1").unwrap();
    let mut spec = common::make_openai_spec(&c_model, &c_url);
    let canon = common::canonicalize(&mut spec);
    let backend = common::create_backend(canon);

    let handle = unsafe { talu_sys::talu_batch_create(backend, ptr::null()) };
    assert!(
        handle.is_null(),
        "create with non-local backend should return null"
    );

    unsafe { talu_sys::talu_backend_free(backend) };
    unsafe { talu_sys::talu_config_free(canon) };
}

// =============================================================================
// Lifecycle with local backend (requires TALU_TEST_MODEL)
// =============================================================================

#[test]
fn create_and_destroy_with_local_backend() {
    skip_without_model!();
    let model = common::model_path().unwrap();
    let (canon, backend) = common::local_backend(&model);

    let handle = unsafe { talu_sys::talu_batch_create(backend, ptr::null()) };
    assert!(!handle.is_null(), "batch creation should succeed");

    // Fresh batch should have no active requests.
    let active = unsafe { talu_sys::talu_batch_has_active(handle) };
    assert_eq!(active, 0, "new batch should have no active requests");

    let count = unsafe { talu_sys::talu_batch_active_count(handle) };
    assert_eq!(count, 0, "new batch should have active_count == 0");

    unsafe { talu_sys::talu_batch_destroy(handle) };
    unsafe { talu_sys::talu_backend_free(backend) };
    unsafe { talu_sys::talu_config_free(canon) };
}

#[test]
fn create_with_config() {
    skip_without_model!();
    let model = common::model_path().unwrap();
    let (canon, backend) = common::local_backend(&model);

    let config = talu_sys::CBatchConfig { max_concurrent: 4 };
    let handle = unsafe { talu_sys::talu_batch_create(backend, &config) };
    assert!(
        !handle.is_null(),
        "batch creation with config should succeed"
    );

    unsafe { talu_sys::talu_batch_destroy(handle) };
    unsafe { talu_sys::talu_backend_free(backend) };
    unsafe { talu_sys::talu_config_free(canon) };
}

#[test]
fn submit_and_step_to_completion() {
    skip_without_model!();
    let model = common::model_path().unwrap();
    let (canon, backend) = common::local_backend(&model);

    let handle = unsafe { talu_sys::talu_batch_create(backend, ptr::null()) };
    assert!(!handle.is_null());

    // Create a chat with a user message.
    let chat = common::create_chat(None);
    common::append_user_message(chat, "Say hello");

    // Submit with small max_tokens.
    let config = talu_sys::CGenerateConfig {
        max_tokens: 8,
        ..talu_sys::CGenerateConfig::default()
    };
    let request_id = unsafe { talu_sys::talu_batch_submit(handle, chat, &config) };
    assert!(request_id > 0, "submit should return non-zero request id");

    // Should have active requests now.
    let active = unsafe { talu_sys::talu_batch_has_active(handle) };
    assert_eq!(active, 1, "should have active requests after submit");

    // Step until completion.
    let mut total_events = 0usize;
    let mut saw_final = false;
    let mut all_text = String::new();

    for _ in 0..100 {
        let mut events = [talu_sys::CBatchEvent::default(); 16];
        let n = unsafe { talu_sys::talu_batch_step(handle, events.as_mut_ptr(), events.len()) };
        total_events += n;

        for event in &events[..n] {
            assert_eq!(event.request_id, request_id, "event should match request");

            // Collect text deltas.
            if event.event_type == 0 && event.text_len > 0 && !event.text_ptr.is_null() {
                let text = unsafe {
                    std::str::from_utf8(std::slice::from_raw_parts(event.text_ptr, event.text_len))
                        .unwrap_or("<invalid utf8>")
                };
                all_text.push_str(text);
            }

            if event.is_final != 0 {
                saw_final = true;
            }
        }

        if saw_final {
            break;
        }

        // If no more active, we're done even without final event.
        let still_active = unsafe { talu_sys::talu_batch_has_active(handle) };
        if still_active == 0 {
            break;
        }
    }

    assert!(saw_final, "should see a final event");
    assert!(total_events > 0, "should produce at least one event");

    // Take result.
    let result_ptr = unsafe { talu_sys::talu_batch_take_result(handle, request_id) };
    assert!(
        !result_ptr.is_null(),
        "take_result should return non-null for completed request"
    );

    let result = unsafe { &*result_ptr };
    assert!(result.prompt_tokens > 0, "should report prompt tokens");
    assert!(
        result.completion_tokens > 0,
        "should report completion tokens"
    );
    assert!(result.generation_ns > 0, "should report generation time");

    // Text should be non-null.
    if !result.text.is_null() {
        let text = unsafe { std::ffi::CStr::from_ptr(result.text) };
        let text_str = text.to_str().unwrap_or("<invalid>");
        assert!(!text_str.is_empty(), "result text should not be empty");
    }

    unsafe { talu_sys::talu_batch_result_free(result_ptr) };

    // Taking again should return null.
    let again = unsafe { talu_sys::talu_batch_take_result(handle, request_id) };
    assert!(
        again.is_null(),
        "second take_result should return null (already taken)"
    );

    unsafe { talu_sys::talu_chat_free(chat) };
    unsafe { talu_sys::talu_batch_destroy(handle) };
    unsafe { talu_sys::talu_backend_free(backend) };
    unsafe { talu_sys::talu_config_free(canon) };
}

#[test]
fn cancel_active_request() {
    skip_without_model!();
    let model = common::model_path().unwrap();
    let (canon, backend) = common::local_backend(&model);

    let handle = unsafe { talu_sys::talu_batch_create(backend, ptr::null()) };
    assert!(!handle.is_null());

    let chat = common::create_chat(None);
    common::append_user_message(chat, "Tell me a long story");

    let config = talu_sys::CGenerateConfig {
        max_tokens: 100,
        ..talu_sys::CGenerateConfig::default()
    };
    let request_id = unsafe { talu_sys::talu_batch_submit(handle, chat, &config) };
    assert!(request_id > 0);

    // Cancel immediately.
    let cancelled = unsafe { talu_sys::talu_batch_cancel(handle, request_id) };
    assert_eq!(cancelled, 1, "cancel should succeed for active request");

    // Cancel again should return 0.
    let cancelled_again = unsafe { talu_sys::talu_batch_cancel(handle, request_id) };
    assert_eq!(
        cancelled_again, 0,
        "cancel of already-cancelled request should return 0"
    );

    // Cancel non-existent request.
    let bad_cancel = unsafe { talu_sys::talu_batch_cancel(handle, 999999) };
    assert_eq!(
        bad_cancel, 0,
        "cancel of non-existent request should return 0"
    );

    unsafe { talu_sys::talu_chat_free(chat) };
    unsafe { talu_sys::talu_batch_destroy(handle) };
    unsafe { talu_sys::talu_backend_free(backend) };
    unsafe { talu_sys::talu_config_free(canon) };
}

#[test]
fn step_on_idle_returns_zero() {
    skip_without_model!();
    let model = common::model_path().unwrap();
    let (canon, backend) = common::local_backend(&model);

    let handle = unsafe { talu_sys::talu_batch_create(backend, ptr::null()) };
    assert!(!handle.is_null());

    // Step with no active requests should return 0 events.
    let mut events = [talu_sys::CBatchEvent::default(); 8];
    let n = unsafe { talu_sys::talu_batch_step(handle, events.as_mut_ptr(), events.len()) };
    assert_eq!(n, 0, "step on idle batch should return 0 events");

    unsafe { talu_sys::talu_batch_destroy(handle) };
    unsafe { talu_sys::talu_backend_free(backend) };
    unsafe { talu_sys::talu_config_free(canon) };
}

#[test]
fn destroy_with_active_requests() {
    skip_without_model!();
    let model = common::model_path().unwrap();
    let (canon, backend) = common::local_backend(&model);

    let handle = unsafe { talu_sys::talu_batch_create(backend, ptr::null()) };
    assert!(!handle.is_null());

    let chat = common::create_chat(None);
    common::append_user_message(chat, "Hello world");

    let config = talu_sys::CGenerateConfig {
        max_tokens: 50,
        ..talu_sys::CGenerateConfig::default()
    };
    let _request_id = unsafe { talu_sys::talu_batch_submit(handle, chat, &config) };

    // Destroy without draining — should not crash.
    unsafe { talu_sys::talu_batch_destroy(handle) };

    unsafe { talu_sys::talu_chat_free(chat) };
    unsafe { talu_sys::talu_backend_free(backend) };
    unsafe { talu_sys::talu_config_free(canon) };
}
