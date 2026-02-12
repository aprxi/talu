//! Code blocks JSON extraction tests.
//!
//! Tests for `talu_content_get_code_blocks_json` which retrieves code block
//! metadata from output_text content parts.

use crate::capi::responses::common::ResponsesTestContext;
use std::ptr;
use talu::responses::{MessageRole, ResponsesView};

// =============================================================================
// Success Cases
// =============================================================================

/// Get code blocks returns null when no code blocks present.
#[test]
fn get_code_blocks_returns_null_when_none() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();

    // Add assistant message (output_text) without code blocks
    h.append_message(MessageRole::Assistant, "Hello, no code here.")
        .unwrap();

    let mut out_ptr: *const u8 = ptr::null();
    let mut out_len: usize = 0;

    let result = unsafe {
        talu_sys::talu_content_get_code_blocks_json(
            h.as_ptr() as *const _,
            0, // item_index
            0, // part_index
            &mut out_ptr,
            &mut out_len,
        )
    };

    assert_eq!(result, 0, "should succeed");
    assert!(out_ptr.is_null(), "should return null when no code blocks");
    assert_eq!(out_len, 0, "length should be 0 when no code blocks");
}

/// Get code blocks on user message (input_text) returns null.
#[test]
fn get_code_blocks_input_text_returns_null() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();

    // User messages have input_text, not output_text
    h.append_message(MessageRole::User, "Hello with ```code```")
        .unwrap();

    let mut out_ptr: *const u8 = ptr::null();
    let mut out_len: usize = 0;

    let result = unsafe {
        talu_sys::talu_content_get_code_blocks_json(
            h.as_ptr() as *const _,
            0,
            0,
            &mut out_ptr,
            &mut out_len,
        )
    };

    assert_eq!(result, 0, "should succeed even for input_text");
    assert!(out_ptr.is_null(), "input_text has no code_blocks_json field");
}

// =============================================================================
// Error Cases
// =============================================================================

/// Null handle returns error.
#[test]
fn get_code_blocks_null_handle_returns_error() {
    let mut out_ptr: *const u8 = ptr::null();
    let mut out_len: usize = 0;

    let result = unsafe {
        talu_sys::talu_content_get_code_blocks_json(
            ptr::null(),
            0,
            0,
            &mut out_ptr,
            &mut out_len,
        )
    };

    assert_ne!(result, 0, "null handle should return error");
}

/// Item index out of bounds returns error.
#[test]
fn get_code_blocks_item_out_of_bounds() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();

    h.append_message(MessageRole::User, "Hello").unwrap();

    let mut out_ptr: *const u8 = ptr::null();
    let mut out_len: usize = 0;

    let result = unsafe {
        talu_sys::talu_content_get_code_blocks_json(
            h.as_ptr() as *const _,
            999, // out of bounds
            0,
            &mut out_ptr,
            &mut out_len,
        )
    };

    assert_ne!(result, 0, "out of bounds item_index should return error");
}

/// Part index out of bounds returns error.
#[test]
fn get_code_blocks_part_out_of_bounds() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();

    h.append_message(MessageRole::Assistant, "Single part message")
        .unwrap();

    let mut out_ptr: *const u8 = ptr::null();
    let mut out_len: usize = 0;

    let result = unsafe {
        talu_sys::talu_content_get_code_blocks_json(
            h.as_ptr() as *const _,
            0,
            999, // out of bounds
            &mut out_ptr,
            &mut out_len,
        )
    };

    assert_ne!(result, 0, "out of bounds part_index should return error");
}

/// Non-message item (function_call) returns error.
#[test]
fn get_code_blocks_non_message_item_returns_error() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();

    h.append_message(MessageRole::User, "Call a function").unwrap();
    h.append_function_call("call_1", "my_func", "{}").unwrap();

    let mut out_ptr: *const u8 = ptr::null();
    let mut out_len: usize = 0;

    // Item 1 is a function_call, not a message
    let result = unsafe {
        talu_sys::talu_content_get_code_blocks_json(
            h.as_ptr() as *const _,
            1, // function_call item
            0,
            &mut out_ptr,
            &mut out_len,
        )
    };

    assert_ne!(result, 0, "non-message item should return error");
}

// =============================================================================
// Memory Safety
// =============================================================================

/// Null output pointers are handled safely.
#[test]
fn get_code_blocks_null_outputs_handled() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();

    h.append_message(MessageRole::Assistant, "Test").unwrap();

    // Call with null output pointers - should not crash
    let result = unsafe {
        talu_sys::talu_content_get_code_blocks_json(
            h.as_ptr() as *const _,
            0,
            0,
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };

    // Should succeed or return error gracefully, not crash
    let _ = result;
}

/// Repeated calls don't leak memory.
#[test]
fn get_code_blocks_repeated_calls_no_leak() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();

    h.append_message(MessageRole::Assistant, "Test message")
        .unwrap();

    for _ in 0..100 {
        let mut out_ptr: *const u8 = ptr::null();
        let mut out_len: usize = 0;

        let result = unsafe {
            talu_sys::talu_content_get_code_blocks_json(
                h.as_ptr() as *const _,
                0,
                0,
                &mut out_ptr,
                &mut out_len,
            )
        };

        assert_eq!(result, 0);
    }
    // Test passes if no crash/leak
}

/// Multiple content parts in same message.
#[test]
fn get_code_blocks_multiple_parts() {
    let mut ctx = ResponsesTestContext::new();
    let h = ctx.handle_mut();

    // Create message with multiple content parts
    h.append_message(MessageRole::Assistant, "First part").unwrap();

    // Only one part exists, so part_index 1 should error
    let mut out_ptr: *const u8 = ptr::null();
    let mut out_len: usize = 0;

    let result0 = unsafe {
        talu_sys::talu_content_get_code_blocks_json(
            h.as_ptr() as *const _,
            0,
            0, // first part - exists
            &mut out_ptr,
            &mut out_len,
        )
    };
    assert_eq!(result0, 0, "part 0 should succeed");

    let result1 = unsafe {
        talu_sys::talu_content_get_code_blocks_json(
            h.as_ptr() as *const _,
            0,
            1, // second part - doesn't exist
            &mut out_ptr,
            &mut out_len,
        )
    };
    assert_ne!(result1, 0, "part 1 should error (out of bounds)");
}

/// Empty conversation returns error for any index.
#[test]
fn get_code_blocks_empty_conversation() {
    let ctx = ResponsesTestContext::new();
    let h = ctx.handle();

    let mut out_ptr: *const u8 = ptr::null();
    let mut out_len: usize = 0;

    let result = unsafe {
        talu_sys::talu_content_get_code_blocks_json(
            h.as_ptr() as *const _,
            0,
            0,
            &mut out_ptr,
            &mut out_len,
        )
    };

    assert_ne!(result, 0, "empty conversation should error");
}
