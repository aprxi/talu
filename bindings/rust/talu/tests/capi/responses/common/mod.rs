//! Shared test fixtures for the Responses integration test suite.

use std::ffi::CString;
use std::os::raw::c_void;
use talu::responses::{ContentType, ItemType, MessageRole, ResponsesHandle, ResponsesView};

/// Test-only owner for raw C chat handles.
pub struct RawChatHandle {
    ptr: *mut c_void,
}

impl RawChatHandle {
    pub fn new() -> Self {
        // SAFETY: null options are accepted by the C API; null return is checked below.
        let ptr = unsafe { talu_sys::talu_chat_create(std::ptr::null_mut()) };
        assert!(!ptr.is_null(), "talu_chat_create returned null");
        Self { ptr }
    }

    pub fn with_system(system: &str) -> Self {
        let system = CString::new(system).expect("system CString");
        // SAFETY: system is a valid C string for the duration of the call.
        let ptr = unsafe {
            talu_sys::talu_chat_create_with_system(system.as_ptr(), std::ptr::null_mut())
        };
        assert!(!ptr.is_null(), "talu_chat_create_with_system returned null");
        Self { ptr }
    }

    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }
}

impl Drop for RawChatHandle {
    fn drop(&mut self) {
        // SAFETY: ptr is either null or owned by this fixture.
        unsafe { talu_sys::talu_chat_free(self.ptr) };
    }
}

/// Test-only owner for raw C responses handles.
pub struct RawResponsesHandle {
    ptr: *mut talu_sys::ResponsesHandle,
}

impl RawResponsesHandle {
    pub fn new() -> Self {
        // SAFETY: no preconditions. Null return is checked below.
        let ptr = unsafe { talu_sys::talu_responses_create() };
        assert!(!ptr.is_null(), "talu_responses_create returned null");
        Self { ptr }
    }

    pub fn as_ptr(&self) -> *mut talu_sys::ResponsesHandle {
        self.ptr
    }

    pub fn append_message(&mut self, role: talu_sys::MessageRole, content: &str) {
        // SAFETY: self.ptr is valid and content bytes remain live for the call.
        let id = unsafe {
            talu_sys::talu_responses_append_message(
                self.ptr,
                role as u8,
                content.as_ptr(),
                content.len(),
            )
        };
        assert!(id >= 0, "append_message failed for {content:?}");
    }

    pub fn item_count(&self) -> usize {
        // SAFETY: self.ptr is valid for the lifetime of this fixture.
        unsafe { talu_sys::talu_responses_item_count(self.ptr) }
    }

    pub fn message_text(&self, index: usize) -> String {
        let mut msg = talu_sys::CMessageItem::default();
        // SAFETY: self.ptr is valid and msg is a valid out-parameter.
        let rc = unsafe { talu_sys::talu_responses_item_as_message(self.ptr, index, &mut msg) };
        assert_eq!(rc, 0, "item {index} should be a message");

        let mut out = String::new();
        for part_index in 0..msg.content_count {
            let mut part = talu_sys::CContentPart::default();
            // SAFETY: self.ptr is valid and part is a valid out-parameter.
            let rc = unsafe {
                talu_sys::talu_responses_item_message_get_content(
                    self.ptr, index, part_index, &mut part,
                )
            };
            assert_eq!(rc, 0, "message content {index}:{part_index} should exist");
            if part.data_ptr.is_null() || part.data_len == 0 {
                continue;
            }
            let content_type = talu_sys::ContentType::from(part.content_type);
            if matches!(
                content_type,
                talu_sys::ContentType::InputText
                    | talu_sys::ContentType::OutputText
                    | talu_sys::ContentType::Text
                    | talu_sys::ContentType::ReasoningText
                    | talu_sys::ContentType::SummaryText
                    | talu_sys::ContentType::Refusal
            ) {
                // SAFETY: C API returns a valid data pointer and length for the handle lifetime.
                let bytes = unsafe { std::slice::from_raw_parts(part.data_ptr, part.data_len) };
                out.push_str(&String::from_utf8_lossy(bytes));
            }
        }
        out
    }
}

impl Drop for RawResponsesHandle {
    fn drop(&mut self) {
        // SAFETY: ptr is either null or owned by this fixture.
        unsafe { talu_sys::talu_responses_free(self.ptr) };
    }
}

/// Ephemeral conversation context for tests.
///
/// Wraps a `ResponsesHandle` so each test starts with a fresh conversation.
pub struct ResponsesTestContext {
    handle: ResponsesHandle,
}

impl ResponsesTestContext {
    /// Create a new context with an empty conversation.
    pub fn new() -> Self {
        let handle = ResponsesHandle::new().expect("failed to create ResponsesHandle");
        Self { handle }
    }

    /// Create a new context with a session identifier.
    pub fn with_session(session_id: &str) -> Self {
        let handle =
            ResponsesHandle::with_session(session_id).expect("failed to create with session");
        Self { handle }
    }

    /// Borrow the underlying handle (read-only via ResponsesView).
    pub fn handle(&self) -> &ResponsesHandle {
        &self.handle
    }

    /// Mutably borrow the underlying handle.
    pub fn handle_mut(&mut self) -> &mut ResponsesHandle {
        &mut self.handle
    }
}

// ---------------------------------------------------------------------------
// Conversation shape builders
// ---------------------------------------------------------------------------

/// system + user + assistant (3 items).
pub fn build_simple_conversation(h: &mut ResponsesHandle) {
    h.append_message(MessageRole::System, "You are a test assistant.")
        .unwrap();
    h.append_message(MessageRole::User, "Hello!").unwrap();
    h.append_message(MessageRole::Assistant, "Hi there!")
        .unwrap();
}

/// user + FC + FCO + assistant (single tool turn).
pub fn build_with_function_call(h: &mut ResponsesHandle) {
    h.append_message(MessageRole::User, "What is the weather?")
        .unwrap();
    h.append_function_call("call_123", "get_weather", r#"{"location":"NYC"}"#)
        .unwrap();
    h.append_function_call_output("call_123", "Sunny, 72F")
        .unwrap();
    h.append_message(MessageRole::Assistant, "The weather in NYC is sunny, 72F.")
        .unwrap();
}

/// user + n FCs + n FCOs + assistant.
pub fn build_with_parallel_calls(h: &mut ResponsesHandle, n: usize) {
    h.append_message(MessageRole::User, "Get weather and time.")
        .unwrap();
    for i in 0..n {
        h.append_function_call(
            &format!("call_{}", i),
            &format!("func_{}", i),
            &format!(r#"{{"arg":{}}}"#, i),
        )
        .unwrap();
    }
    for i in 0..n {
        h.append_function_call_output(&format!("call_{}", i), &format!("result_{}", i))
            .unwrap();
    }
    h.append_message(MessageRole::Assistant, "Done with parallel calls.")
        .unwrap();
}

/// system + user + FC + FCO + assistant (full loop).
pub fn build_tool_loop(h: &mut ResponsesHandle) {
    h.append_message(
        MessageRole::System,
        "You are a helpful assistant with tools.",
    )
    .unwrap();
    h.append_message(MessageRole::User, "What is the weather?")
        .unwrap();
    h.append_function_call("call_abc", "get_weather", r#"{"location":"SF"}"#)
        .unwrap();
    h.append_function_call_output("call_abc", "Foggy, 58F")
        .unwrap();
    h.append_message(MessageRole::Assistant, "It's foggy and 58F in SF.")
        .unwrap();
}

// ---------------------------------------------------------------------------
// Assertion helpers
// ---------------------------------------------------------------------------

/// Assert that the message text at `index` equals `expected`.
pub fn assert_message_text(view: &impl ResponsesView, index: usize, expected: &str) {
    let actual = view.message_text(index).unwrap();
    assert_eq!(actual, expected, "message_text at index {}", index);
}

/// Assert that the item type at `index` matches `expected_type`.
pub fn assert_item_type(view: &impl ResponsesView, index: usize, expected_type: ItemType) {
    let actual = view.item_type(index);
    assert_eq!(actual, expected_type, "item_type at index {}", index);
}

/// Assert function call fields at `index`.
pub fn assert_fc_fields(view: &impl ResponsesView, index: usize, name: &str, call_id: &str) {
    let fc = view.get_function_call(index).unwrap();
    assert_eq!(fc.name, name, "FC name at index {}", index);
    assert_eq!(fc.call_id, call_id, "FC call_id at index {}", index);
}

/// Helper to check if a content type is text.
pub fn is_text_type(ct: ContentType) -> bool {
    matches!(
        ct,
        ContentType::InputText
            | ContentType::OutputText
            | ContentType::Text
            | ContentType::ReasoningText
            | ContentType::SummaryText
            | ContentType::Refusal
    )
}
