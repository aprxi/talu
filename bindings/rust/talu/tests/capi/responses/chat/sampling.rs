//! Sampling parameter get/set tests.

use talu::ChatHandle;

fn chat() -> ChatHandle {
    ChatHandle::new(None).unwrap()
}

#[test]
fn default_temperature() {
    let c = chat();
    let temp = unsafe { talu_sys::talu_chat_get_temperature(c.as_ptr()) };
    assert!(
        (temp - 0.7).abs() < 1e-5,
        "default temperature should be 0.7, got {}",
        temp
    );
}

#[test]
fn default_max_tokens() {
    let c = chat();
    let max = unsafe { talu_sys::talu_chat_get_max_tokens(c.as_ptr()) };
    assert_eq!(max, 256, "default max_tokens should be 256");
}

#[test]
fn default_top_k() {
    let c = chat();
    let k = unsafe { talu_sys::talu_chat_get_top_k(c.as_ptr()) };
    assert_eq!(k, 50, "default top_k should be 50");
}

#[test]
fn default_top_p() {
    let c = chat();
    let p = unsafe { talu_sys::talu_chat_get_top_p(c.as_ptr()) };
    assert!(
        (p - 0.9).abs() < 1e-5,
        "default top_p should be 0.9, got {}",
        p
    );
}

#[test]
fn default_min_p() {
    let c = chat();
    let p = unsafe { talu_sys::talu_chat_get_min_p(c.as_ptr()) };
    assert!(p.abs() < 1e-5, "default min_p should be 0.0, got {}", p);
}

#[test]
fn default_repetition_penalty() {
    let c = chat();
    let rp = unsafe { talu_sys::talu_chat_get_repetition_penalty(c.as_ptr()) };
    assert!(
        (rp - 1.0).abs() < 1e-5,
        "default repetition_penalty should be 1.0, got {}",
        rp
    );
}

#[test]
fn set_temperature_roundtrip() {
    let c = chat();
    unsafe { talu_sys::talu_chat_set_temperature(c.as_ptr(), 1.5) };
    let temp = unsafe { talu_sys::talu_chat_get_temperature(c.as_ptr()) };
    assert!(
        (temp - 1.5).abs() < 1e-5,
        "temperature should roundtrip, got {}",
        temp
    );
}

#[test]
fn set_max_tokens_roundtrip() {
    let c = chat();
    unsafe { talu_sys::talu_chat_set_max_tokens(c.as_ptr(), 1024) };
    let max = unsafe { talu_sys::talu_chat_get_max_tokens(c.as_ptr()) };
    assert_eq!(max, 1024, "max_tokens should roundtrip");
}

#[test]
fn set_all_params_roundtrip() {
    let c = chat();
    let ptr = c.as_ptr();

    unsafe {
        talu_sys::talu_chat_set_temperature(ptr, 0.3);
        talu_sys::talu_chat_set_max_tokens(ptr, 512);
        talu_sys::talu_chat_set_top_k(ptr, 40);
        talu_sys::talu_chat_set_top_p(ptr, 0.95);
        talu_sys::talu_chat_set_min_p(ptr, 0.05);
        talu_sys::talu_chat_set_repetition_penalty(ptr, 1.1);
    }

    let temp = unsafe { talu_sys::talu_chat_get_temperature(ptr) };
    let max = unsafe { talu_sys::talu_chat_get_max_tokens(ptr) };
    let top_k = unsafe { talu_sys::talu_chat_get_top_k(ptr) };
    let top_p = unsafe { talu_sys::talu_chat_get_top_p(ptr) };
    let min_p = unsafe { talu_sys::talu_chat_get_min_p(ptr) };
    let rp = unsafe { talu_sys::talu_chat_get_repetition_penalty(ptr) };

    assert!((temp - 0.3).abs() < 1e-5, "temperature roundtrip");
    assert_eq!(max, 512, "max_tokens roundtrip");
    assert_eq!(top_k, 40, "top_k roundtrip");
    assert!((top_p - 0.95).abs() < 1e-5, "top_p roundtrip");
    assert!((min_p - 0.05).abs() < 1e-5, "min_p roundtrip");
    assert!((rp - 1.1).abs() < 1e-5, "repetition_penalty roundtrip");
}
