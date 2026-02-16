//! Backend lifecycle tests.
//!
//! Exercises `talu_backend_create_from_canonical`, `talu_backend_free`,
//! `talu_backend_list_models`, and `talu_backend_list_models_free`.

use crate::capi::router::common;
use std::ffi::{c_void, CString};
use std::ptr;

#[test]
fn create_from_canonical_null_returns_error() {
    let mut backend_ptr: *mut c_void = ptr::null_mut();
    let rc = unsafe {
        talu_sys::talu_backend_create_from_canonical(
            ptr::null_mut(),
            talu_sys::BackendCreateOptions::default(),
            &mut backend_ptr as *mut _ as *mut c_void,
        )
    };
    assert_ne!(rc, 0, "create with null canonical should fail");
    assert!(backend_ptr.is_null(), "backend should remain null on error");
}

#[test]
fn canonicalize_nonexistent_local_model_fails() {
    let c_ref = CString::new("/nonexistent/model.gguf").unwrap();
    let mut spec = common::make_local_spec(&c_ref);

    let mut canon_ptr: *mut c_void = ptr::null_mut();
    let rc = unsafe {
        talu_sys::talu_config_canonicalize(&mut spec, &mut canon_ptr as *mut _ as *mut c_void)
    };
    // Canonicalization performs filesystem existence checks for local paths.
    // A nonexistent path must fail at this stage.
    assert_ne!(rc, 0, "canonicalize for nonexistent local model should fail");
    assert!(
        canon_ptr.is_null(),
        "canonical handle should remain null on error"
    );
}

#[test]
fn backend_free_null_is_noop() {
    unsafe { talu_sys::talu_backend_free(ptr::null_mut()) };
}

#[test]
fn create_openai_backend_succeeds() {
    let c_model = CString::new("gpt-4").unwrap();
    let c_url = CString::new("https://api.openai.com/v1").unwrap();
    let mut spec = common::make_openai_spec(&c_model, &c_url);
    let canon = common::canonicalize(&mut spec);
    let mut backend_ptr: *mut c_void = ptr::null_mut();
    let rc = unsafe {
        talu_sys::talu_backend_create_from_canonical(
            canon,
            talu_sys::BackendCreateOptions::default(),
            &mut backend_ptr as *mut _ as *mut c_void,
        )
    };
    assert_eq!(rc, 0, "OpenAI backend creation should succeed");
    assert!(!backend_ptr.is_null(), "backend handle should be non-null");

    unsafe { talu_sys::talu_backend_free(backend_ptr) };
    unsafe { talu_sys::talu_config_free(canon) };
}

#[test]
fn list_models_null_backend_does_not_crash() {
    // Calling list_models with null may return an error result or crash.
    // The key is it doesn't segfault.
    let result = unsafe { talu_sys::talu_backend_list_models(ptr::null_mut()) };
    // Error code should be non-zero for null backend.
    assert_ne!(
        result.error_code, 0,
        "list_models with null backend should return error"
    );
}

#[test]
fn list_models_free_null_is_noop() {
    unsafe { talu_sys::talu_backend_list_models_free(ptr::null_mut()) };
}

#[test]
fn create_backend_with_progress_callback_succeeds() {
    unsafe extern "C" fn noop_progress(
        _update: *const talu_sys::ProgressUpdate,
        _user_data: *mut c_void,
    ) {
    }

    let c_model = CString::new("gpt-4").unwrap();
    let c_url = CString::new("https://api.openai.com/v1").unwrap();
    let mut spec = common::make_openai_spec(&c_model, &c_url);
    let canon = common::canonicalize(&mut spec);

    let options = talu_sys::BackendCreateOptions {
        progress_callback: noop_progress as *mut c_void,
        progress_user_data: ptr::null_mut(),
    };

    let mut backend_ptr: *mut c_void = ptr::null_mut();
    let rc = unsafe {
        talu_sys::talu_backend_create_from_canonical(
            canon,
            options,
            &mut backend_ptr as *mut _ as *mut c_void,
        )
    };
    assert_eq!(rc, 0, "backend creation with progress callback should succeed");
    assert!(
        !backend_ptr.is_null(),
        "backend handle should be non-null with progress callback"
    );

    unsafe { talu_sys::talu_backend_free(backend_ptr) };
    unsafe { talu_sys::talu_config_free(canon) };
}

#[test]
fn list_models_on_openai_backend_returns_result() {
    let c_model = CString::new("gpt-4").unwrap();
    let c_url = CString::new("https://api.openai.com/v1").unwrap();
    let mut spec = common::make_openai_spec(&c_model, &c_url);
    let canon = common::canonicalize(&mut spec);
    let backend = common::create_backend(canon);

    let mut result = unsafe { talu_sys::talu_backend_list_models(backend) };

    // Without valid API credentials, this will return an error.
    // The important thing is the CRemoteModelListResult struct marshalling
    // works correctly — fields are populated, no crash, no UB.
    if result.error_code == 0 {
        // If somehow the API call succeeded (unlikely without credentials),
        // verify struct consistency.
        if result.count > 0 {
            assert!(
                !result.models.is_null(),
                "non-zero count should have non-null models"
            );
        }
    } else {
        // Error path: models should be null, count should be 0.
        assert!(
            result.models.is_null(),
            "error result should have null models"
        );
        assert_eq!(result.count, 0, "error result should have count 0");
    }

    // Free the result (safe even on error — models is null, freeModelListResult handles it).
    unsafe {
        talu_sys::talu_backend_list_models_free(
            &mut result as *mut _ as *mut c_void,
        )
    };

    unsafe { talu_sys::talu_backend_free(backend) };
    unsafe { talu_sys::talu_config_free(canon) };
}
