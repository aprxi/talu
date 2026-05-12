//! Scheduler scoring C API tests.

use crate::capi::responses::common::engine::{self as common, skip_without_model};
use std::ffi::{c_void, CString};
use std::ptr;

struct LocalBackend {
    canon: *mut c_void,
    backend: *mut c_void,
}

impl LocalBackend {
    fn new(model_path: &str) -> Self {
        let (canon, backend) = common::local_backend(model_path);
        Self { canon, backend }
    }

    fn as_ptr(&self) -> *mut c_void {
        self.backend
    }
}

impl Drop for LocalBackend {
    fn drop(&mut self) {
        unsafe {
            talu_sys::talu_backend_free(self.backend);
            talu_sys::talu_config_free(self.canon);
        }
    }
}

struct RawTokenizer {
    ptr: *mut c_void,
}

impl RawTokenizer {
    fn new(model_path: &str) -> Self {
        let c_path = CString::new(model_path).expect("model path should not contain NUL");
        let mut ptr: *mut c_void = ptr::null_mut();
        let rc = unsafe {
            talu_sys::talu_tokenizer_create(c_path.as_ptr(), &mut ptr as *mut _ as *mut c_void)
        };
        assert_eq!(rc, 0, "tokenizer creation should succeed");
        assert!(!ptr.is_null(), "tokenizer creation returned null");
        Self { ptr }
    }

    fn encode_with_bos(&self, text: &str) -> Vec<u32> {
        let opts = talu_sys::EncodeOptions {
            add_bos: 1,
            ..talu_sys::EncodeOptions::default()
        };
        let result =
            unsafe { talu_sys::talu_tokenizer_encode(self.ptr, text.as_ptr(), text.len(), &opts) };
        assert!(result.error_msg.is_null(), "tokenization should succeed");
        assert!(
            result.num_tokens > 1,
            "scoring needs context and target tokens"
        );
        assert!(!result.ids.is_null(), "tokenization returned null ids");

        let ids = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) }.to_vec();
        unsafe { talu_sys::talu_encode_result_free(result) };
        ids
    }
}

impl Drop for RawTokenizer {
    fn drop(&mut self) {
        unsafe { talu_sys::talu_tokenizer_free(self.ptr) };
    }
}

#[test]
fn score_tokens_nll_null_backend_errors_and_clears_outputs() {
    let context = [1_u32];
    let target = [2_u32];
    let mut nll = 123.0;
    let mut scored = 99_usize;

    let rc = unsafe {
        talu_sys::talu_scheduler_score_tokens_nll(
            ptr::null_mut(),
            context.as_ptr(),
            context.len(),
            target.as_ptr(),
            target.len(),
            0,
            &mut nll as *mut f64 as *mut c_void,
            &mut scored as *mut usize as *mut c_void,
        )
    };

    assert_ne!(rc, 0, "null backend must fail");
    assert_eq!(nll, 0.0, "out_nll should be reset before validation");
    assert_eq!(
        scored, 0,
        "out_scored_tokens should be reset before validation"
    );
}

#[test]
fn score_tokens_nll_rejects_null_out_nll() {
    let mut scored = 99_usize;

    let rc = unsafe {
        talu_sys::talu_scheduler_score_tokens_nll(
            ptr::null_mut(),
            ptr::null(),
            0,
            ptr::null(),
            0,
            0,
            ptr::null_mut(),
            &mut scored as *mut usize as *mut c_void,
        )
    };

    assert_ne!(rc, 0, "null out_nll must fail before backend validation");
    assert_eq!(
        scored, 99,
        "out_scored_tokens should not be touched when out_nll is invalid"
    );
}

#[test]
fn score_tokens_nll_rejects_null_out_scored_tokens() {
    let mut nll = 123.0;

    let rc = unsafe {
        talu_sys::talu_scheduler_score_tokens_nll(
            ptr::null_mut(),
            ptr::null(),
            0,
            ptr::null(),
            0,
            0,
            &mut nll as *mut f64 as *mut c_void,
            ptr::null_mut(),
        )
    };

    assert_ne!(
        rc, 0,
        "null out_scored_tokens must fail before backend validation"
    );
    assert_eq!(
        nll, 123.0,
        "out_nll should not be touched when out_scored_tokens is invalid"
    );
}

#[test]
fn score_tokens_nll_scores_real_backend_when_model_available() {
    skip_without_model!();
    let model = common::model_path().unwrap();
    let backend = LocalBackend::new(&model);
    let tokenizer = RawTokenizer::new(&model);
    let tokens = tokenizer.encode_with_bos("Hello world");
    let (context, target) = tokens.split_at(1);
    let mut nll = 0.0_f64;
    let mut scored = 0_usize;

    let rc = unsafe {
        talu_sys::talu_scheduler_score_tokens_nll(
            backend.as_ptr(),
            context.as_ptr(),
            context.len(),
            target.as_ptr(),
            target.len(),
            0,
            &mut nll as *mut f64 as *mut c_void,
            &mut scored as *mut usize as *mut c_void,
        )
    };

    assert_eq!(rc, 0, "real-backend NLL scoring should succeed");
    assert_eq!(scored, target.len(), "all target tokens should be scored");
    assert!(nll.is_finite(), "NLL should be finite");
    assert!(nll >= 0.0, "negative log-likelihood should be non-negative");
}

#[test]
fn score_tokens_joint_null_reference_backend_errors_and_clears_outputs() {
    let context = [1_u32];
    let target = [2_u32];
    let mut reference_nll = 1.0;
    let mut model_nll = 2.0;
    let mut kld = 3.0;
    let mut scored = 4_usize;

    let rc = unsafe {
        talu_sys::talu_scheduler_score_tokens_joint(
            ptr::null_mut(),
            ptr::null_mut(),
            context.as_ptr(),
            context.len(),
            target.as_ptr(),
            target.len(),
            0,
            &mut reference_nll as *mut f64 as *mut c_void,
            &mut model_nll as *mut f64 as *mut c_void,
            &mut kld as *mut f64 as *mut c_void,
            &mut scored as *mut usize as *mut c_void,
        )
    };

    assert_ne!(rc, 0, "null reference backend must fail");
    assert_eq!(reference_nll, 0.0);
    assert_eq!(model_nll, 0.0);
    assert_eq!(kld, 0.0);
    assert_eq!(scored, 0);
}

#[test]
fn score_tokens_joint_scores_same_real_backend_when_model_available() {
    skip_without_model!();
    let model = common::model_path().unwrap();
    let backend = LocalBackend::new(&model);
    let tokenizer = RawTokenizer::new(&model);
    let tokens = tokenizer.encode_with_bos("Hello world");
    let (context, target) = tokens.split_at(1);
    let mut reference_nll = 0.0_f64;
    let mut model_nll = 0.0_f64;
    let mut kld = 0.0_f64;
    let mut scored = 0_usize;

    let rc = unsafe {
        talu_sys::talu_scheduler_score_tokens_joint(
            backend.as_ptr(),
            backend.as_ptr(),
            context.as_ptr(),
            context.len(),
            target.as_ptr(),
            target.len(),
            0,
            &mut reference_nll as *mut f64 as *mut c_void,
            &mut model_nll as *mut f64 as *mut c_void,
            &mut kld as *mut f64 as *mut c_void,
            &mut scored as *mut usize as *mut c_void,
        )
    };

    assert_eq!(rc, 0, "real-backend joint scoring should succeed");
    assert_eq!(scored, target.len(), "all target tokens should be scored");
    assert!(reference_nll.is_finite(), "reference NLL should be finite");
    assert!(model_nll.is_finite(), "model NLL should be finite");
    assert!(kld.is_finite(), "KLD should be finite");
    assert!(reference_nll >= 0.0, "reference NLL should be non-negative");
    assert!(model_nll >= 0.0, "model NLL should be non-negative");

    let tolerance = 1e-6_f64 * reference_nll.abs().max(model_nll.abs()).max(1.0);
    assert!(
        (reference_nll - model_nll).abs() <= tolerance,
        "same backend should produce matching NLLs"
    );
    assert!(
        kld.abs() <= tolerance,
        "same backend should have near-zero KL divergence"
    );
}

#[test]
fn score_tokens_joint_rejects_each_null_output_pointer() {
    let mut reference_nll = 1.0;
    let mut model_nll = 2.0;
    let mut kld = 3.0;
    let mut scored = 4_usize;

    let rc = unsafe {
        talu_sys::talu_scheduler_score_tokens_joint(
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null(),
            0,
            ptr::null(),
            0,
            0,
            ptr::null_mut(),
            &mut model_nll as *mut f64 as *mut c_void,
            &mut kld as *mut f64 as *mut c_void,
            &mut scored as *mut usize as *mut c_void,
        )
    };
    assert_ne!(rc, 0, "null reference NLL output must fail");
    assert_eq!(model_nll, 2.0);
    assert_eq!(kld, 3.0);
    assert_eq!(scored, 4);

    let rc = unsafe {
        talu_sys::talu_scheduler_score_tokens_joint(
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null(),
            0,
            ptr::null(),
            0,
            0,
            &mut reference_nll as *mut f64 as *mut c_void,
            ptr::null_mut(),
            &mut kld as *mut f64 as *mut c_void,
            &mut scored as *mut usize as *mut c_void,
        )
    };
    assert_ne!(rc, 0, "null model NLL output must fail");
    assert_eq!(reference_nll, 1.0);
    assert_eq!(kld, 3.0);
    assert_eq!(scored, 4);

    let rc = unsafe {
        talu_sys::talu_scheduler_score_tokens_joint(
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null(),
            0,
            ptr::null(),
            0,
            0,
            &mut reference_nll as *mut f64 as *mut c_void,
            &mut model_nll as *mut f64 as *mut c_void,
            ptr::null_mut(),
            &mut scored as *mut usize as *mut c_void,
        )
    };
    assert_ne!(rc, 0, "null KLD output must fail");
    assert_eq!(reference_nll, 1.0);
    assert_eq!(model_nll, 2.0);
    assert_eq!(scored, 4);

    let rc = unsafe {
        talu_sys::talu_scheduler_score_tokens_joint(
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null(),
            0,
            ptr::null(),
            0,
            0,
            &mut reference_nll as *mut f64 as *mut c_void,
            &mut model_nll as *mut f64 as *mut c_void,
            &mut kld as *mut f64 as *mut c_void,
            ptr::null_mut(),
        )
    };
    assert_ne!(rc, 0, "null scored-token output must fail");
    assert_eq!(reference_nll, 1.0);
    assert_eq!(model_nll, 2.0);
    assert_eq!(kld, 3.0);
}
