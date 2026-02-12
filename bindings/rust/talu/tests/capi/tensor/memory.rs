//! Memory safety tests for tensor C API.
//!
//! These tests verify that the tensor API handles null pointers gracefully
//! without crashing.

use std::ffi::c_void;
use std::ptr;

/// Passing null output pointer to talu_tensor_create should return error, not crash.
#[test]
fn create_null_output_returns_error() {
    let shape = [2i64, 3];
    let result = unsafe {
        talu_sys::talu_tensor_create(
            shape.as_ptr() as *mut c_void,
            shape.len(),
            0, // f32
            1, // CPU
            0, // device_id
            ptr::null_mut(),
        )
    };
    assert_ne!(
        result, 0,
        "create with null output should return error code"
    );
}

/// Passing null output pointer to talu_tensor_test_embeddings should return error.
#[test]
fn test_embeddings_null_output_returns_error() {
    let result = unsafe { talu_sys::talu_tensor_test_embeddings(ptr::null_mut()) };
    assert_ne!(
        result, 0,
        "test_embeddings with null output should return error code"
    );
}

/// Freeing null tensor should be a no-op, not crash.
#[test]
fn free_null_tensor_is_noop() {
    unsafe { talu_sys::talu_tensor_free(ptr::null_mut()) };
}

/// Getting data pointer from null tensor should return error, not crash.
#[test]
fn data_ptr_null_tensor_returns_error() {
    let mut out_ptr: *mut c_void = ptr::null_mut();
    let result = unsafe {
        talu_sys::talu_tensor_data_ptr(ptr::null_mut(), &mut out_ptr as *mut _ as *mut c_void)
    };
    assert_ne!(result, 0, "data_ptr with null tensor should return error");
}

/// Getting data pointer with null output should return error, not crash.
#[test]
fn data_ptr_null_output_returns_error() {
    // Create a valid tensor first
    let shape = [2i64, 3];
    let mut tensor: *mut c_void = ptr::null_mut();
    let create_result = unsafe {
        talu_sys::talu_tensor_create(
            shape.as_ptr() as *mut c_void,
            shape.len(),
            0, // f32
            1, // CPU
            0,
            &mut tensor as *mut _ as *mut c_void,
        )
    };
    if create_result == 0 && !tensor.is_null() {
        let result = unsafe { talu_sys::talu_tensor_data_ptr(tensor, ptr::null_mut()) };
        assert_ne!(result, 0, "data_ptr with null output should return error");
        unsafe { talu_sys::talu_tensor_free(tensor) };
    }
}

/// Getting ndim from null tensor should return 0, not crash.
#[test]
fn ndim_null_tensor_returns_zero() {
    let result = unsafe { talu_sys::talu_tensor_ndim(ptr::null_mut()) };
    assert_eq!(result, 0, "ndim with null tensor should return 0");
}

/// Getting numel from null tensor should return 0, not crash.
#[test]
fn numel_null_tensor_returns_zero() {
    let result = unsafe { talu_sys::talu_tensor_numel(ptr::null_mut()) };
    assert_eq!(result, 0, "numel with null tensor should return 0");
}

/// DLPack export with null tensor should return error, not crash.
#[test]
fn to_dlpack_null_tensor_returns_error() {
    let mut dlpack: *mut c_void = ptr::null_mut();
    let result = unsafe {
        talu_sys::talu_tensor_to_dlpack(ptr::null_mut(), &mut dlpack as *mut _ as *mut c_void)
    };
    assert_ne!(result, 0, "to_dlpack with null tensor should return error");
}

/// DLPack export with null output should return error, not crash.
#[test]
fn to_dlpack_null_output_returns_error() {
    // Create a valid tensor first
    let shape = [2i64, 3];
    let mut tensor: *mut c_void = ptr::null_mut();
    let create_result = unsafe {
        talu_sys::talu_tensor_create(
            shape.as_ptr() as *mut c_void,
            shape.len(),
            0, // f32
            1, // CPU
            0,
            &mut tensor as *mut _ as *mut c_void,
        )
    };
    if create_result == 0 && !tensor.is_null() {
        let result = unsafe { talu_sys::talu_tensor_to_dlpack(tensor, ptr::null_mut()) };
        assert_ne!(result, 0, "to_dlpack with null output should return error");
        unsafe { talu_sys::talu_tensor_free(tensor) };
    }
}
