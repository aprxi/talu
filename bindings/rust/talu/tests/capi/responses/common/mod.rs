//! Shared test fixtures for the Responses integration test suite.

use talu::responses::{ContentType, ItemType, MessageRole, ResponsesHandle, ResponsesView};

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

/// Alternating user/assistant pairs.
pub fn build_multi_turn(h: &mut ResponsesHandle, turns: usize) {
    for i in 0..turns {
        h.append_message(MessageRole::User, &format!("User turn {}", i))
            .unwrap();
        h.append_message(MessageRole::Assistant, &format!("Assistant turn {}", i))
            .unwrap();
    }
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
