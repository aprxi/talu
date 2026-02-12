//! Memory safety and lifecycle tests for template CAPI.
//!
//! Validates null-pointer handling and proper cleanup of allocated strings.

use std::ffi::{c_char, c_int, c_void, CStr, CString};

// ===========================================================================
// talu_template_render memory safety
// ===========================================================================

/// Render result can be freed cleanly with talu_text_free.
#[test]
fn render_result_freed_cleanly() {
    let template = CString::new("Hello {{ name }}").unwrap();
    let vars = CString::new(r#"{"name":"World"}"#).unwrap();
    let mut out: *mut c_char = std::ptr::null_mut();

    let rc = unsafe {
        talu_sys::talu_template_render(
            template.as_ptr(),
            vars.as_ptr(),
            false,
            &mut out as *mut _ as *mut c_void,
        )
    };

    assert_eq!(rc, 0);
    assert!(!out.is_null());

    // Verify content before freeing.
    let s = unsafe { CStr::from_ptr(out) }.to_string_lossy();
    assert_eq!(s, "Hello World");

    unsafe { talu_sys::talu_text_free(out) };
}

/// Chat template result can be freed cleanly with talu_text_free.
#[test]
fn chat_template_result_freed_cleanly() {
    let template = "{{ bos_token }}{% for msg in messages %}{{ msg.content }}{% endfor %}";
    let msgs = CString::new(r#"[{"role":"user","content":"Hi"}]"#).unwrap();
    let bos = CString::new("<s>").unwrap();
    let eos = CString::new("</s>").unwrap();
    let mut out: *mut c_char = std::ptr::null_mut();

    let rc = unsafe {
        talu_sys::talu_apply_chat_template_string(
            template.as_ptr(),
            template.len(),
            msgs.as_ptr(),
            0 as c_int,
            bos.as_ptr(),
            eos.as_ptr(),
            &mut out as *mut _ as *mut c_void,
        )
    };

    assert_eq!(rc, 0);
    assert!(!out.is_null());

    let s = unsafe { CStr::from_ptr(out) }.to_string_lossy();
    assert!(s.contains("Hi"), "got: {s:?}");

    unsafe { talu_sys::talu_text_free(out) };
}

/// On render error, output pointer remains null (no leak).
#[test]
fn render_error_leaves_output_null() {
    let template = CString::new("{% if %}").unwrap(); // syntax error
    let vars = CString::new("{}").unwrap();
    let mut out: *mut c_char = std::ptr::null_mut();

    let rc = unsafe {
        talu_sys::talu_template_render(
            template.as_ptr(),
            vars.as_ptr(),
            false,
            &mut out as *mut _ as *mut c_void,
        )
    };

    assert_ne!(rc, 0, "expected error");
    assert!(out.is_null(), "output should be null on error");
}

/// On chat template error, output pointer remains null.
#[test]
fn chat_template_error_leaves_output_null() {
    let template = "{% if %}"; // syntax error
    let msgs = CString::new(r#"[{"role":"user","content":"Hi"}]"#).unwrap();
    let bos = CString::new("").unwrap();
    let eos = CString::new("").unwrap();
    let mut out: *mut c_char = std::ptr::null_mut();

    let rc = unsafe {
        talu_sys::talu_apply_chat_template_string(
            template.as_ptr(),
            template.len(),
            msgs.as_ptr(),
            0 as c_int,
            bos.as_ptr(),
            eos.as_ptr(),
            &mut out as *mut _ as *mut c_void,
        )
    };

    assert_ne!(rc, 0, "expected error");
    assert!(out.is_null(), "output should be null on error");
}
