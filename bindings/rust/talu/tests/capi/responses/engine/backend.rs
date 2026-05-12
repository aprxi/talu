//! Backend lifecycle tests.
//!
//! Exercises `talu_backend_create_from_canonical` and `talu_backend_free`.

use crate::capi::responses::common::engine as common;
use crate::capi::responses::common::engine::skip_without_model;
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
    assert_ne!(
        rc, 0,
        "canonicalize for nonexistent local model should fail"
    );
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
fn create_backend_with_progress_callback_succeeds() {
    skip_without_model!();
    unsafe extern "C" fn noop_progress(
        _update: *const talu_sys::ProgressUpdate,
        _user_data: *mut c_void,
    ) {
    }

    let model = common::model_path().unwrap();
    let c_model = CString::new(model).unwrap();
    let mut spec = common::make_local_spec(&c_model);
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
    assert_eq!(
        rc, 0,
        "backend creation with progress callback should succeed"
    );
    assert!(
        !backend_ptr.is_null(),
        "backend handle should be non-null with progress callback"
    );

    unsafe { talu_sys::talu_backend_free(backend_ptr) };
    unsafe { talu_sys::talu_config_free(canon) };
}
