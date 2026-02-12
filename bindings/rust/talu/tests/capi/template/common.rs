//! Shared helpers for template CAPI tests.

use std::ffi::{c_char, c_int, c_void, CStr, CString};

/// Render a Jinja2 template with JSON variables.
///
/// Returns the rendered string on success, or the error code on failure.
pub fn render_template(template: &str, json_vars: &str, strict: bool) -> Result<String, i32> {
    let t = CString::new(template).unwrap();
    let v = CString::new(json_vars).unwrap();
    let mut out: *mut c_char = std::ptr::null_mut();

    let rc = unsafe {
        talu_sys::talu_template_render(
            t.as_ptr(),
            v.as_ptr(),
            strict,
            &mut out as *mut _ as *mut c_void,
        )
    };

    if rc != 0 {
        return Err(rc);
    }

    assert!(!out.is_null(), "render succeeded but output is null");
    let s = unsafe { CStr::from_ptr(out) }
        .to_string_lossy()
        .to_string();
    unsafe { talu_sys::talu_text_free(out) };
    Ok(s)
}

/// Apply a chat template string to messages JSON.
///
/// Returns the rendered prompt on success, or the error code on failure.
pub fn apply_chat_template(
    template: &str,
    messages_json: &str,
    add_generation_prompt: bool,
    bos_token: &str,
    eos_token: &str,
) -> Result<String, i32> {
    let msgs = CString::new(messages_json).unwrap();
    let bos = CString::new(bos_token).unwrap();
    let eos = CString::new(eos_token).unwrap();
    let mut out: *mut c_char = std::ptr::null_mut();

    let rc = unsafe {
        talu_sys::talu_apply_chat_template_string(
            template.as_ptr(),
            template.len(),
            msgs.as_ptr(),
            add_generation_prompt as c_int,
            bos.as_ptr(),
            eos.as_ptr(),
            &mut out as *mut _ as *mut c_void,
        )
    };

    if rc != 0 {
        return Err(rc);
    }

    assert!(!out.is_null(), "apply_chat_template succeeded but output is null");
    let s = unsafe { CStr::from_ptr(out) }
        .to_string_lossy()
        .to_string();
    unsafe { talu_sys::talu_text_free(out) };
    Ok(s)
}

// ---------------------------------------------------------------------------
// Chat template fixtures
// ---------------------------------------------------------------------------

/// Minimal chat template: just concatenates role and content.
pub const SIMPLE_TEMPLATE: &str =
    "{% for msg in messages %}{{ msg.role }}: {{ msg.content }}\n{% endfor %}";

/// Template with generation prompt suffix.
pub const GENERATION_TEMPLATE: &str = concat!(
    "{% for msg in messages %}",
    "<|{{ msg.role }}|>{{ msg.content }}</|{{ msg.role }}|>",
    "{% endfor %}",
    "{% if add_generation_prompt %}<|assistant|>{% endif %}",
);

/// Template using BOS/EOS tokens.
pub const BOS_EOS_TEMPLATE: &str = concat!(
    "{{ bos_token }}",
    "{% for msg in messages %}",
    "{{ msg.role }}: {{ msg.content }}",
    "{% if not loop.last %}{{ eos_token }}\n{% endif %}",
    "{% endfor %}",
);

// ---------------------------------------------------------------------------
// Message JSON fixtures
// ---------------------------------------------------------------------------

pub const SINGLE_USER_MSG: &str = r#"[{"role":"user","content":"Hello"}]"#;

pub const MULTITURN_MSGS: &str = r#"[
    {"role":"system","content":"You are helpful."},
    {"role":"user","content":"Hi"},
    {"role":"assistant","content":"Hello!"},
    {"role":"user","content":"How are you?"}
]"#;

