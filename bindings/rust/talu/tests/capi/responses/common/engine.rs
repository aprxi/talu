//! Shared helpers for response engine C-API tests.

use std::ffi::{c_void, CString};

/// Returns the model path from `TALU_TEST_MODEL`, or `None` if unset.
pub fn model_path() -> Option<String> {
    std::env::var("TALU_TEST_MODEL")
        .ok()
        .filter(|s| !s.is_empty())
}

/// Skips the current test if `TALU_TEST_MODEL` is not set.
macro_rules! skip_without_model {
    () => {
        let Some(ref _model) = crate::capi::responses::common::engine::model_path() else {
            eprintln!("skipping: TALU_TEST_MODEL not set");
            return;
        };
    };
}
pub(crate) use skip_without_model;

/// Build a `TaluModelSpec` for a local backend (backend_type_raw = 0).
///
/// The returned spec borrows `path`'s pointer — caller must keep the CString alive.
pub fn make_local_spec(path: &CString) -> talu_sys::TaluModelSpec {
    talu_sys::TaluModelSpec {
        abi_version: 1,
        struct_size: std::mem::size_of::<talu_sys::TaluModelSpec>() as u32,
        ref_: path.as_ptr(),
        backend_type_raw: 0, // Local
        backend_config: unsafe { std::mem::zeroed() },
    }
}

/// Canonicalize a spec, panicking on failure. Returns the opaque canonical handle.
pub fn canonicalize(spec: &mut talu_sys::TaluModelSpec) -> *mut c_void {
    let mut canon_ptr: *mut c_void = std::ptr::null_mut();
    let rc = unsafe {
        talu_sys::talu_config_canonicalize(spec, &mut canon_ptr as *mut _ as *mut c_void)
    };
    assert_eq!(rc, 0, "canonicalize failed with error code {}", rc);
    assert!(!canon_ptr.is_null(), "canonicalize returned null handle");
    canon_ptr
}

/// Create an inference backend from a canonical handle, panicking on failure.
pub fn create_backend(canon: *mut c_void) -> *mut c_void {
    let mut backend_ptr: *mut c_void = std::ptr::null_mut();
    let rc = unsafe {
        talu_sys::talu_backend_create_from_canonical(
            canon,
            talu_sys::BackendCreateOptions::default(),
            &mut backend_ptr as *mut _ as *mut c_void,
        )
    };
    assert_eq!(rc, 0, "create_backend failed with error code {}", rc);
    assert!(
        !backend_ptr.is_null(),
        "create_backend returned null handle"
    );
    backend_ptr
}

/// Create a ChatHandle with an optional system message. Panics on failure.
pub fn create_chat(system: Option<&str>) -> *mut c_void {
    let ptr = unsafe {
        if let Some(msg) = system {
            let c_msg = CString::new(msg).unwrap();
            talu_sys::talu_chat_create_with_system(c_msg.as_ptr(), std::ptr::null_mut())
        } else {
            talu_sys::talu_chat_create(std::ptr::null_mut())
        }
    };
    assert!(!ptr.is_null(), "chat creation failed");
    ptr
}

/// Append a user message to a chat handle via the conversation.
pub fn append_user_message(chat: *mut c_void, content: &str) {
    let conv = unsafe { talu_sys::talu_chat_get_conversation(chat) };
    assert!(!conv.is_null(), "chat has no conversation");
    let result = unsafe {
        // Role codes: 0=system, 1=user, 2=assistant, 3=developer.
        talu_sys::talu_responses_append_message(conv, 1, content.as_ptr(), content.len())
    };
    assert!(result >= 0, "append_user_message failed");
}

/// Convenience: build a local backend from a model path. Returns (canon, backend).
/// Both must be freed by the caller.
pub fn local_backend(model_path: &str) -> (*mut c_void, *mut c_void) {
    let c_path = CString::new(model_path).unwrap();
    let mut spec = make_local_spec(&c_path);
    let canon = canonicalize(&mut spec);
    let backend = create_backend(canon);
    (canon, backend)
}
