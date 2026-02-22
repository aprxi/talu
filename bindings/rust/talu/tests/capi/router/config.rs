//! Config validation and canonicalization tests.
//!
//! Exercises `talu_config_validate`, `talu_config_canonicalize`,
//! `talu_config_get_view`, and `talu_config_free` with both valid and
//! invalid inputs. No model file required.

use crate::capi::router::common;
use std::ffi::{c_void, CString};
use std::ptr;

// ---------------------------------------------------------------------------
// talu_config_validate
// ---------------------------------------------------------------------------

#[test]
fn validate_null_spec_returns_error() {
    let rc = unsafe { talu_sys::talu_config_validate(ptr::null_mut()) };
    assert_ne!(rc, 0, "validate with null spec should fail");
}

#[test]
fn validate_bad_abi_version_returns_error() {
    let c_ref = CString::new("some-model").unwrap();
    let mut spec = talu_sys::TaluModelSpec {
        abi_version: 99,
        struct_size: std::mem::size_of::<talu_sys::TaluModelSpec>() as u32,
        ref_: c_ref.as_ptr(),
        backend_type_raw: 0,
        backend_config: unsafe { std::mem::zeroed() },
    };
    let rc = unsafe { talu_sys::talu_config_validate(&mut spec) };
    assert_ne!(rc, 0, "abi_version=99 should be rejected");
}

#[test]
fn validate_null_ref_returns_error() {
    let mut spec = talu_sys::TaluModelSpec {
        abi_version: 1,
        struct_size: std::mem::size_of::<talu_sys::TaluModelSpec>() as u32,
        ref_: ptr::null(),
        backend_type_raw: 0,
        backend_config: unsafe { std::mem::zeroed() },
    };
    let rc = unsafe { talu_sys::talu_config_validate(&mut spec) };
    assert_ne!(rc, 0, "null ref should be rejected");
}

#[test]
fn validate_empty_ref_returns_error() {
    let c_ref = CString::new("").unwrap();
    let mut spec = talu_sys::TaluModelSpec {
        abi_version: 1,
        struct_size: std::mem::size_of::<talu_sys::TaluModelSpec>() as u32,
        ref_: c_ref.as_ptr(),
        backend_type_raw: 0,
        backend_config: unsafe { std::mem::zeroed() },
    };
    let rc = unsafe { talu_sys::talu_config_validate(&mut spec) };
    assert_ne!(rc, 0, "empty ref should be rejected");
}

#[test]
fn validate_bad_struct_size_returns_error() {
    let c_ref = CString::new("some-model").unwrap();
    let mut spec = talu_sys::TaluModelSpec {
        abi_version: 1,
        struct_size: 0,
        ref_: c_ref.as_ptr(),
        backend_type_raw: 0,
        backend_config: unsafe { std::mem::zeroed() },
    };
    let rc = unsafe { talu_sys::talu_config_validate(&mut spec) };
    assert_ne!(rc, 0, "struct_size=0 should be rejected");
}

// ---------------------------------------------------------------------------
// talu_config_canonicalize
// ---------------------------------------------------------------------------

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
        backend_type_raw: 1, // OpenAI (avoids file check)
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
fn canonicalize_valid_openai_spec_succeeds() {
    let c_model = CString::new("gpt-4").unwrap();
    let c_url = CString::new("https://api.openai.com/v1").unwrap();
    let mut spec = common::make_openai_spec(&c_model, &c_url);
    let mut out: *mut c_void = ptr::null_mut();
    let rc =
        unsafe { talu_sys::talu_config_canonicalize(&mut spec, &mut out as *mut _ as *mut c_void) };
    assert_eq!(rc, 0, "valid OpenAI spec should canonicalize successfully");
    assert!(!out.is_null(), "canonical handle should be non-null");
    unsafe { talu_sys::talu_config_free(out) };
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
    let c_model = CString::new("gpt-4o").unwrap();
    let c_url = CString::new("https://custom.api.example.com/v1").unwrap();
    let mut spec = common::make_openai_spec(&c_model, &c_url);

    let canon = common::canonicalize(&mut spec);

    let mut view = talu_sys::TaluModelSpec::default();
    let rc = unsafe { talu_sys::talu_config_get_view(canon, &mut view) };
    assert_eq!(rc, 0, "get_view should succeed");

    // Verify the model ref roundtripped
    assert!(!view.ref_.is_null());
    let ref_str = unsafe { std::ffi::CStr::from_ptr(view.ref_) }
        .to_string_lossy()
        .to_string();
    assert_eq!(ref_str, "gpt-4o", "model ref should roundtrip");

    // Backend type should be OpenAI (1)
    assert_eq!(view.backend_type_raw, 1, "backend type should be OpenAI");

    unsafe { talu_sys::talu_config_free(canon) };
}
