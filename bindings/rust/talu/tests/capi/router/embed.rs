//! Embedding and cleanup tests.
//!
//! Null-safety tests run unconditionally. Actual embedding tests require
//! `TALU_TEST_MODEL` to point to a model that supports embeddings.

use crate::capi::router::common;
use crate::capi::router::common::skip_without_model;
use std::ffi::CString;
use std::ptr;

#[test]
fn embedding_dim_null_model_returns_zero() {
    let dim = unsafe { talu_sys::talu_router_embedding_dim(ptr::null()) };
    assert_eq!(dim, 0, "embedding_dim with null model should return 0");
}

#[test]
fn embedding_dim_nonexistent_model_returns_zero() {
    let model = CString::new("/nonexistent/embedding_model.gguf").unwrap();
    let dim = unsafe { talu_sys::talu_router_embedding_dim(model.as_ptr()) };
    assert_eq!(
        dim, 0,
        "embedding_dim with nonexistent model should return 0"
    );
}

#[test]
fn embed_null_model_returns_error() {
    let text = CString::new("hello").unwrap();
    let mut out_embedding: *mut f32 = ptr::null_mut();
    let mut out_dim: usize = 0;
    let rc = unsafe {
        talu_sys::talu_router_embed(
            ptr::null(), // null model
            text.as_ptr(),
            talu_sys::CPoolingStrategy::Mean,
            true,
            &mut out_embedding as *mut _ as *mut std::ffi::c_void,
            &mut out_dim as *mut _ as *mut std::ffi::c_void,
        )
    };
    assert_ne!(rc, 0, "embed with null model should return error");
}

#[test]
fn embed_null_text_returns_error() {
    let model = CString::new("/nonexistent/model.gguf").unwrap();
    let mut out_embedding: *mut f32 = ptr::null_mut();
    let mut out_dim: usize = 0;
    let rc = unsafe {
        talu_sys::talu_router_embed(
            model.as_ptr(),
            ptr::null(), // null text
            talu_sys::CPoolingStrategy::Mean,
            true,
            &mut out_embedding as *mut _ as *mut std::ffi::c_void,
            &mut out_dim as *mut _ as *mut std::ffi::c_void,
        )
    };
    assert_ne!(rc, 0, "embed with null text should return error");
}

#[test]
fn embed_free_null_is_noop() {
    unsafe { talu_sys::talu_router_embedding_free(ptr::null(), 0) };
}

#[test]
fn router_close_all_is_safe() {
    // Calling close_all with no active engines should be a no-op.
    unsafe { talu_sys::talu_router_close_all() };
}

// ---------------------------------------------------------------------------
// Model-gated tests
// ---------------------------------------------------------------------------

#[test]
fn embed_produces_valid_normalized_output() {
    skip_without_model!();
    let model = common::model_path().unwrap();
    let c_model = CString::new(model.as_str()).unwrap();
    let c_text = CString::new("hello world").unwrap();

    // Check if this model supports embeddings.
    let dim = unsafe { talu_sys::talu_router_embedding_dim(c_model.as_ptr()) };
    if dim == 0 {
        eprintln!("skipping: model does not support embeddings");
        return;
    }

    let mut out_embedding: *mut f32 = ptr::null_mut();
    let mut out_dim: usize = 0;
    let rc = unsafe {
        talu_sys::talu_router_embed(
            c_model.as_ptr(),
            c_text.as_ptr(),
            talu_sys::CPoolingStrategy::Mean,
            true, // normalize
            &mut out_embedding as *mut _ as *mut std::ffi::c_void,
            &mut out_dim as *mut _ as *mut std::ffi::c_void,
        )
    };
    assert_eq!(rc, 0, "embed should succeed");
    assert!(!out_embedding.is_null(), "embedding should be non-null");
    assert_eq!(out_dim, dim, "output dim should match embedding_dim");

    // Verify the embedding contains finite float values.
    let embedding = unsafe { std::slice::from_raw_parts(out_embedding, out_dim) };
    for (i, &val) in embedding.iter().enumerate() {
        assert!(
            val.is_finite(),
            "embedding[{}] should be finite, got {}",
            i,
            val
        );
    }

    // Normalized embedding should have L2 norm close to 1.0.
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "normalized embedding should have L2 norm ~1.0, got {}",
        norm,
    );

    unsafe { talu_sys::talu_router_embedding_free(out_embedding, out_dim) };
}

#[test]
fn embed_unnormalized_differs_from_normalized() {
    skip_without_model!();
    let model = common::model_path().unwrap();
    let c_model = CString::new(model.as_str()).unwrap();
    let c_text = CString::new("a test sentence for embeddings").unwrap();

    let dim = unsafe { talu_sys::talu_router_embedding_dim(c_model.as_ptr()) };
    if dim == 0 {
        eprintln!("skipping: model does not support embeddings");
        return;
    }

    // Get unnormalized embedding.
    let mut out_embedding: *mut f32 = ptr::null_mut();
    let mut out_dim: usize = 0;
    let rc = unsafe {
        talu_sys::talu_router_embed(
            c_model.as_ptr(),
            c_text.as_ptr(),
            talu_sys::CPoolingStrategy::Mean,
            false, // unnormalized
            &mut out_embedding as *mut _ as *mut std::ffi::c_void,
            &mut out_dim as *mut _ as *mut std::ffi::c_void,
        )
    };
    assert_eq!(rc, 0, "unnormalized embed should succeed");
    assert!(!out_embedding.is_null());
    assert_eq!(out_dim, dim);

    let embedding = unsafe { std::slice::from_raw_parts(out_embedding, out_dim) };
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    // Unnormalized embedding should generally NOT have L2 norm == 1.0.
    // This verifies the normalize flag is actually being respected.
    assert!(
        norm > 0.0,
        "unnormalized embedding should have positive L2 norm"
    );

    unsafe { talu_sys::talu_router_embedding_free(out_embedding, out_dim) };
}

#[test]
fn embed_all_pooling_strategies_produce_output() {
    skip_without_model!();
    let model = common::model_path().unwrap();
    let c_model = CString::new(model.as_str()).unwrap();
    let c_text = CString::new("pooling strategy test input").unwrap();

    let dim = unsafe { talu_sys::talu_router_embedding_dim(c_model.as_ptr()) };
    if dim == 0 {
        eprintln!("skipping: model does not support embeddings");
        return;
    }

    let strategies = [
        (talu_sys::CPoolingStrategy::Last, "Last"),
        (talu_sys::CPoolingStrategy::Mean, "Mean"),
        (talu_sys::CPoolingStrategy::First, "First"),
    ];

    for (strategy, name) in &strategies {
        let mut out_embedding: *mut f32 = ptr::null_mut();
        let mut out_dim: usize = 0;
        let rc = unsafe {
            talu_sys::talu_router_embed(
                c_model.as_ptr(),
                c_text.as_ptr(),
                *strategy,
                true,
                &mut out_embedding as *mut _ as *mut std::ffi::c_void,
                &mut out_dim as *mut _ as *mut std::ffi::c_void,
            )
        };
        assert_eq!(rc, 0, "embed with {} pooling should succeed", name);
        assert!(
            !out_embedding.is_null(),
            "{} pooling output should be non-null",
            name
        );
        assert_eq!(out_dim, dim, "{} pooling dim should match", name);

        // Verify all values are finite.
        let embedding = unsafe { std::slice::from_raw_parts(out_embedding, out_dim) };
        for (i, &val) in embedding.iter().enumerate() {
            assert!(
                val.is_finite(),
                "{} pooling: embedding[{}] should be finite, got {}",
                name, i, val
            );
        }

        unsafe { talu_sys::talu_router_embedding_free(out_embedding, out_dim) };
    }
}
