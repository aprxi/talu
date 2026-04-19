//! Config canonicalization tests.
//!
//! Exercises `talu_config_canonicalize`, `talu_config_get_view`, and
//! `talu_config_free` with both valid and invalid inputs.

use crate::capi::router::common;
use std::ffi::{c_void, CString};
use std::ptr;
use std::time::{SystemTime, UNIX_EPOCH};

fn create_temp_model_file() -> String {
    let mut path = std::env::temp_dir();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    path.push(format!("talu-test-model-{nanos}.gguf"));
    std::fs::write(&path, b"ok").unwrap();
    path.to_string_lossy().to_string()
}

#[test]
fn canonicalize_null_spec_returns_error() {
    let mut out: *mut c_void = ptr::null_mut();
    let rc = unsafe {
        talu_sys::talu_config_canonicalize(ptr::null_mut(), &mut out as *mut _ as *mut c_void)
    };
    assert_ne!(rc, 0, "canonicalize with null spec should fail");
    assert!(out.is_null(), "output handle should remain null on error");
}

#[test]
fn canonicalize_null_output_returns_error() {
    let c_ref = CString::new("some-model").unwrap();
    let mut spec = talu_sys::TaluModelSpec {
        abi_version: 1,
        struct_size: std::mem::size_of::<talu_sys::TaluModelSpec>() as u32,
        ref_: c_ref.as_ptr(),
        backend_type_raw: 0,
        backend_config: unsafe { std::mem::zeroed() },
    };
    let rc = unsafe { talu_sys::talu_config_canonicalize(&mut spec, ptr::null_mut()) };
    assert_ne!(rc, 0, "canonicalize with null output should fail");
}

#[test]
fn canonicalize_nonexistent_local_model_returns_error() {
    let c_ref = CString::new("/nonexistent/path/to/model.gguf").unwrap();
    let mut spec = common::make_local_spec(&c_ref);
    let mut out: *mut c_void = ptr::null_mut();
    let rc =
        unsafe { talu_sys::talu_config_canonicalize(&mut spec, &mut out as *mut _ as *mut c_void) };
    assert_ne!(
        rc, 0,
        "nonexistent local model should fail canonicalization"
    );
    assert!(out.is_null(), "output handle should remain null on error");
}

#[test]
fn canonicalize_valid_local_spec_succeeds() {
    let model_path = create_temp_model_file();
    let c_model = CString::new(model_path.clone()).unwrap();
    let mut spec = common::make_local_spec(&c_model);
    let mut out: *mut c_void = ptr::null_mut();
    let rc =
        unsafe { talu_sys::talu_config_canonicalize(&mut spec, &mut out as *mut _ as *mut c_void) };
    assert_eq!(rc, 0, "valid local spec should canonicalize successfully");
    assert!(!out.is_null(), "canonical handle should be non-null");
    unsafe { talu_sys::talu_config_free(out) };
    let _ = std::fs::remove_file(model_path);
}

// ---------------------------------------------------------------------------
// talu_config_free
// ---------------------------------------------------------------------------

#[test]
fn config_free_null_is_noop() {
    unsafe { talu_sys::talu_config_free(ptr::null_mut()) };
}

// ---------------------------------------------------------------------------
// talu_config_get_view roundtrip
// ---------------------------------------------------------------------------

#[test]
fn config_get_view_roundtrips() {
    let model_path = create_temp_model_file();
    let c_model = CString::new(model_path.clone()).unwrap();
    let mut spec = common::make_local_spec(&c_model);

    let canon = common::canonicalize(&mut spec);

    let mut view = talu_sys::TaluModelSpec::default();
    let rc = unsafe { talu_sys::talu_config_get_view(canon, &mut view) };
    assert_eq!(rc, 0, "get_view should succeed");

    // Verify the model ref roundtripped
    assert!(!view.ref_.is_null());
    let ref_str = unsafe { std::ffi::CStr::from_ptr(view.ref_) }
        .to_string_lossy()
        .to_string();
    assert_eq!(ref_str, model_path, "model ref should roundtrip");

    // Backend type should be Local (0)
    assert_eq!(view.backend_type_raw, 0, "backend type should be Local");

    unsafe { talu_sys::talu_config_free(canon) };
    let _ = std::fs::remove_file(model_path);
}
