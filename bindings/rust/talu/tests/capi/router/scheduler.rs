//! Scheduler C-API tests.
//!
//! Exercises `talu_scheduler_*` lifecycle, null-safety, and submit/step
//! semantics. Null-safety tests run unconditionally; inference tests
//! require `TALU_TEST_MODEL`.

use crate::capi::router::common;
use crate::capi::router::common::skip_without_model;
use std::ffi::CString;
use std::ptr;

// =============================================================================
// Helpers
// =============================================================================

/// Encode text into token IDs using the model's tokenizer (with BOS).
/// Returns (token_ids, tokenizer_handle). Caller must free both:
///   - `talu_sys::talu_encode_result_free(result)` for the tokens
///   - `talu_sys::talu_tokenizer_free(tok)` for the tokenizer
unsafe fn encode_prompt(model_path: &str, text: &str) -> (Vec<u32>, *mut std::ffi::c_void) {
    let c_path = CString::new(model_path).unwrap();
    let mut tok: *mut std::ffi::c_void = ptr::null_mut();
    let rc = talu_sys::talu_tokenizer_create(
        c_path.as_ptr(),
        &mut tok as *mut _ as *mut std::ffi::c_void,
    );
    assert_eq!(rc, 0, "tokenizer creation should succeed");
    assert!(!tok.is_null());

    let opts = talu_sys::EncodeOptions {
        add_bos: 1,
        ..talu_sys::EncodeOptions::default()
    };
    let result = talu_sys::talu_tokenizer_encode(tok, text.as_ptr(), text.len(), &opts);
    assert!(result.error_msg.is_null(), "encode should succeed");
    assert!(result.num_tokens > 0, "encode should produce tokens");
    assert!(
        !result.ids.is_null(),
        "encode ids pointer should be non-null"
    );

    let ids = std::slice::from_raw_parts(result.ids, result.num_tokens).to_vec();
    talu_sys::talu_encode_result_free(result);
    (ids, tok)
}

// =============================================================================
// Null-safety (run unconditionally)
// =============================================================================

#[test]
fn create_null_backend_returns_null() {
    let handle = unsafe { talu_sys::talu_scheduler_create(ptr::null_mut(), ptr::null()) };
    assert!(
        handle.is_null(),
        "create with null backend should return null"
    );
}

#[test]
fn destroy_null_is_noop() {
    unsafe { talu_sys::talu_scheduler_destroy(ptr::null_mut()) };
}

#[test]
fn submit_null_handle_returns_zero() {
    let prompt: [u32; 1] = [1];
    let id = unsafe {
        talu_sys::talu_scheduler_submit(ptr::null_mut(), prompt.as_ptr(), 1, 16, ptr::null())
    };
    assert_eq!(id, 0, "submit with null handle should return 0");
}

#[test]
fn cancel_null_handle_returns_zero() {
    let rc = unsafe { talu_sys::talu_scheduler_cancel(ptr::null_mut(), 1) };
    assert_eq!(rc, 0, "cancel with null handle should return 0");
}

#[test]
fn step_null_handle_returns_zero() {
    let mut events = [talu_sys::CTokenEvent::default(); 8];
    let n = unsafe {
        talu_sys::talu_scheduler_step(ptr::null_mut(), events.as_mut_ptr(), events.len())
    };
    assert_eq!(n, 0, "step with null handle should return 0");
}

#[test]
fn has_active_null_handle_returns_zero() {
    let rc = unsafe { talu_sys::talu_scheduler_has_active(ptr::null_mut()) };
    assert_eq!(rc, 0, "has_active with null handle should return 0");
}

#[test]
fn active_count_null_handle_returns_zero() {
    let n = unsafe { talu_sys::talu_scheduler_active_count(ptr::null_mut()) };
    assert_eq!(n, 0, "active_count with null handle should return 0");
}

#[test]
fn create_non_local_backend_returns_null() {
    let c_model = CString::new("gpt-4").unwrap();
    let c_url = CString::new("https://api.openai.com/v1").unwrap();
    let mut spec = common::make_openai_spec(&c_model, &c_url);
    let canon = common::canonicalize(&mut spec);
    let backend = common::create_backend(canon);

    let handle = unsafe { talu_sys::talu_scheduler_create(backend, ptr::null()) };
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

    let handle = unsafe { talu_sys::talu_scheduler_create(backend, ptr::null()) };
    assert!(!handle.is_null(), "scheduler creation should succeed");

    // Fresh scheduler should have no active requests.
    let active = unsafe { talu_sys::talu_scheduler_has_active(handle) };
    assert_eq!(active, 0, "new scheduler should have no active requests");

    let count = unsafe { talu_sys::talu_scheduler_active_count(handle) };
    assert_eq!(count, 0, "new scheduler should have active_count == 0");

    unsafe { talu_sys::talu_scheduler_destroy(handle) };
    unsafe { talu_sys::talu_backend_free(backend) };
    unsafe { talu_sys::talu_config_free(canon) };
}

#[test]
fn create_with_config() {
    skip_without_model!();
    let model = common::model_path().unwrap();
    let (canon, backend) = common::local_backend(&model);

    let config = talu_sys::CSchedulerConfig { max_concurrent: 4 };
    let handle = unsafe { talu_sys::talu_scheduler_create(backend, &config) };
    assert!(
        !handle.is_null(),
        "scheduler creation with config should succeed"
    );

    unsafe { talu_sys::talu_scheduler_destroy(handle) };
    unsafe { talu_sys::talu_backend_free(backend) };
    unsafe { talu_sys::talu_config_free(canon) };
}

#[test]
fn submit_null_tokens_returns_zero() {
    skip_without_model!();
    let model = common::model_path().unwrap();
    let (canon, backend) = common::local_backend(&model);
    let handle = unsafe { talu_sys::talu_scheduler_create(backend, ptr::null()) };
    assert!(!handle.is_null());

    let id = unsafe { talu_sys::talu_scheduler_submit(handle, ptr::null(), 0, 16, ptr::null()) };
    assert_eq!(id, 0, "submit with null tokens should return 0");

    unsafe { talu_sys::talu_scheduler_destroy(handle) };
    unsafe { talu_sys::talu_backend_free(backend) };
    unsafe { talu_sys::talu_config_free(canon) };
}

#[test]
fn submit_and_step_produces_tokens() {
    skip_without_model!();
    let model = common::model_path().unwrap();
    let (canon, backend) = common::local_backend(&model);
    let handle = unsafe { talu_sys::talu_scheduler_create(backend, ptr::null()) };
    assert!(!handle.is_null());

    let (prompt, tok) = unsafe { encode_prompt(&model, "Hello") };
    let max_tokens = 8;

    let request_id = unsafe {
        talu_sys::talu_scheduler_submit(
            handle,
            prompt.as_ptr(),
            prompt.len(),
            max_tokens,
            ptr::null(),
        )
    };
    assert_ne!(request_id, 0, "submit should return non-zero request ID");

    let active = unsafe { talu_sys::talu_scheduler_has_active(handle) };
    assert_eq!(active, 1, "should have active requests after submit");

    // Step until done or max iterations.
    let mut events = [talu_sys::CTokenEvent::default(); 8];
    let mut total_tokens = 0;
    let mut saw_final = false;

    for _ in 0..(max_tokens + 2) {
        let n = unsafe { talu_sys::talu_scheduler_step(handle, events.as_mut_ptr(), events.len()) };
        for event in &events[..n.min(events.len())] {
            assert_eq!(
                event.request_id, request_id,
                "event should match our request"
            );
            total_tokens += 1;
            if event.is_final != 0 {
                saw_final = true;
            }
        }
        if saw_final {
            break;
        }
    }

    assert!(total_tokens > 0, "should have produced at least one token");
    assert!(saw_final, "should have received a final event");

    // After completion, scheduler should be idle.
    let active = unsafe { talu_sys::talu_scheduler_has_active(handle) };
    assert_eq!(active, 0, "scheduler should be idle after completion");

    unsafe { talu_sys::talu_tokenizer_free(tok) };
    unsafe { talu_sys::talu_scheduler_destroy(handle) };
    unsafe { talu_sys::talu_backend_free(backend) };
    unsafe { talu_sys::talu_config_free(canon) };
}

#[test]
fn cancel_active_request() {
    skip_without_model!();
    let model = common::model_path().unwrap();
    let (canon, backend) = common::local_backend(&model);
    let handle = unsafe { talu_sys::talu_scheduler_create(backend, ptr::null()) };
    assert!(!handle.is_null());

    let (prompt, tok) = unsafe { encode_prompt(&model, "Hello") };
    let request_id = unsafe {
        talu_sys::talu_scheduler_submit(handle, prompt.as_ptr(), prompt.len(), 64, ptr::null())
    };
    assert_ne!(request_id, 0);

    let cancelled = unsafe { talu_sys::talu_scheduler_cancel(handle, request_id) };
    assert_eq!(cancelled, 1, "cancel should succeed for pending request");

    // Double-cancel should return 0 (already gone).
    let cancelled2 = unsafe { talu_sys::talu_scheduler_cancel(handle, request_id) };
    assert_eq!(cancelled2, 0, "double-cancel should return 0");

    let active = unsafe { talu_sys::talu_scheduler_has_active(handle) };
    assert_eq!(active, 0, "scheduler should be idle after cancel");

    unsafe { talu_sys::talu_tokenizer_free(tok) };
    unsafe { talu_sys::talu_scheduler_destroy(handle) };
    unsafe { talu_sys::talu_backend_free(backend) };
    unsafe { talu_sys::talu_config_free(canon) };
}

#[test]
fn cancel_nonexistent_request_returns_zero() {
    skip_without_model!();
    let model = common::model_path().unwrap();
    let (canon, backend) = common::local_backend(&model);
    let handle = unsafe { talu_sys::talu_scheduler_create(backend, ptr::null()) };
    assert!(!handle.is_null());

    let cancelled = unsafe { talu_sys::talu_scheduler_cancel(handle, 99999) };
    assert_eq!(
        cancelled, 0,
        "cancel of nonexistent request should return 0"
    );

    unsafe { talu_sys::talu_scheduler_destroy(handle) };
    unsafe { talu_sys::talu_backend_free(backend) };
    unsafe { talu_sys::talu_config_free(canon) };
}

#[test]
fn step_empty_scheduler_returns_zero() {
    skip_without_model!();
    let model = common::model_path().unwrap();
    let (canon, backend) = common::local_backend(&model);
    let handle = unsafe { talu_sys::talu_scheduler_create(backend, ptr::null()) };
    assert!(!handle.is_null());

    let mut events = [talu_sys::CTokenEvent::default(); 8];
    let n = unsafe { talu_sys::talu_scheduler_step(handle, events.as_mut_ptr(), events.len()) };
    assert_eq!(n, 0, "step on empty scheduler should return 0 events");

    unsafe { talu_sys::talu_scheduler_destroy(handle) };
    unsafe { talu_sys::talu_backend_free(backend) };
    unsafe { talu_sys::talu_config_free(canon) };
}

#[test]
fn submit_with_sampling_options() {
    skip_without_model!();
    let model = common::model_path().unwrap();
    let (canon, backend) = common::local_backend(&model);
    let handle = unsafe { talu_sys::talu_scheduler_create(backend, ptr::null()) };
    assert!(!handle.is_null());

    let (prompt, tok) = unsafe { encode_prompt(&model, "Hello") };
    let opts = talu_sys::CSubmitOptions {
        temperature: 0.5,
        top_k: 40,
        seed: 42,
        ..talu_sys::CSubmitOptions::default()
    };

    let request_id =
        unsafe { talu_sys::talu_scheduler_submit(handle, prompt.as_ptr(), prompt.len(), 4, &opts) };
    assert_ne!(request_id, 0, "submit with sampling options should succeed");

    // Run to completion.
    let mut events = [talu_sys::CTokenEvent::default(); 8];
    for _ in 0..6 {
        let n = unsafe { talu_sys::talu_scheduler_step(handle, events.as_mut_ptr(), events.len()) };
        let done = events[..n.min(events.len())]
            .iter()
            .any(|e| e.is_final != 0);
        if done {
            break;
        }
    }

    unsafe { talu_sys::talu_tokenizer_free(tok) };
    unsafe { talu_sys::talu_scheduler_destroy(handle) };
    unsafe { talu_sys::talu_backend_free(backend) };
    unsafe { talu_sys::talu_config_free(canon) };
}
