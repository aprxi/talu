//! ChatHandle completions JSON ABI tests.
//!
//! These tests verify that completions-format JSON correctly crosses the
//! Rust→Zig ABI boundary via `talu_chat_load_completions_json`. The key
//! ABI concern is the (ptr, len) string passing — content must survive
//! intact, not just item counts.
//!
//! Run: `cargo test --test capi -- capi::responses::chat::completions`

use talu::ChatHandle;

// =============================================================================
// Content fidelity — the core ABI contract
// =============================================================================

/// Exact text must survive the Rust→Zig→Rust roundtrip for every role.
#[test]
fn content_survives_abi_roundtrip() {
    let chat = ChatHandle::new(None).unwrap();
    let json = r#"[
        {"role":"system","content":"You are a test assistant."},
        {"role":"user","content":"What is 2+2?"},
        {"role":"assistant","content":"The answer is 4."}
    ]"#;
    chat.load_completions_json(json).unwrap();

    let items = read_items(&chat);
    assert_eq!(items.len(), 3);
    assert_item(&items[0], "system", "You are a test assistant.");
    assert_item(&items[1], "user", "What is 2+2?");
    assert_item(&items[2], "assistant", "The answer is 4.");
}

/// Multi-byte UTF-8 (CJK, emoji, accented Latin) must not be truncated
/// at byte boundaries — catches off-by-one in ptr+len passing.
#[test]
fn unicode_content_preserved() {
    let chat = ChatHandle::new(None).unwrap();
    let json = r#"[{"role":"user","content":"こんにちは世界 🌍 café résumé"}]"#;
    chat.load_completions_json(json).unwrap();

    let items = read_items(&chat);
    assert_item(&items[0], "user", "こんにちは世界 🌍 café résumé");
}

/// Characters that stress JSON parsing and C-string boundaries:
/// embedded newlines, tabs, backslashes, quotes, angle brackets.
#[test]
fn special_characters_preserved() {
    let content = "line1\\nline2\\ttab\\\\backslash\\\"quote<tag>html&amp;";
    let json = format!(r#"[{{"role":"user","content":"{}"}}]"#, content);
    let chat = ChatHandle::new(None).unwrap();
    chat.load_completions_json(&json).unwrap();

    let items = read_items(&chat);
    let text = item_text(&items[0]);
    // The JSON parser unescapes \\n to \n, \\t to \t, etc.
    assert!(text.contains("\\"), "backslash must survive");
    assert!(text.contains("<tag>"), "angle brackets must survive");
    assert!(text.contains("&amp;"), "ampersand must survive");
}

/// Actual newlines/tabs in content (JSON-escaped as \n, \t) must survive.
#[test]
fn literal_newlines_and_tabs_preserved() {
    let json = r#"[{"role":"user","content":"line1\nline2\ttabbed"}]"#;
    let chat = ChatHandle::new(None).unwrap();
    chat.load_completions_json(json).unwrap();

    let items = read_items(&chat);
    let text = item_text(&items[0]);
    assert!(text.contains('\n'), "newline must survive: {:?}", text);
    assert!(text.contains('\t'), "tab must survive: {:?}", text);
}

/// Empty string content is valid and must not be confused with null/missing.
#[test]
fn empty_content_string_preserved() {
    let chat = ChatHandle::new(None).unwrap();
    let json = r#"[{"role":"assistant","content":""}]"#;
    chat.load_completions_json(json).unwrap();

    let items = read_items(&chat);
    assert_eq!(items.len(), 1);
    assert_eq!(item_role(&items[0]), "assistant");
    let text = item_text(&items[0]);
    assert!(
        text.is_empty(),
        "empty content should stay empty, got: {:?}",
        text
    );
}

/// Large payload (>4KB) catches buffer sizing bugs at the ABI boundary.
#[test]
fn large_content_preserved() {
    let big_text: String = "A".repeat(8192);
    let json = format!(r#"[{{"role":"user","content":"{}"}}]"#, big_text);
    let chat = ChatHandle::new(None).unwrap();
    chat.load_completions_json(&json).unwrap();

    let items = read_items(&chat);
    let text = item_text(&items[0]);
    assert_eq!(text.len(), 8192, "8KB content must survive intact");
    assert!(
        text.chars().all(|c| c == 'A'),
        "content must not be corrupted"
    );
}

// =============================================================================
// Role handling
// =============================================================================

/// All four completions roles must be accepted and preserved.
#[test]
fn all_roles_preserved() {
    let chat = ChatHandle::new(None).unwrap();
    let json = r#"[
        {"role":"system","content":"sys"},
        {"role":"developer","content":"dev"},
        {"role":"user","content":"usr"},
        {"role":"assistant","content":"ast"}
    ]"#;
    chat.load_completions_json(json).unwrap();

    let items = read_items(&chat);
    assert_eq!(items.len(), 4);

    let roles: Vec<&str> = items.iter().map(|i| item_role(i)).collect();
    assert_eq!(roles, vec!["system", "developer", "user", "assistant"]);

    // Verify content too — role check alone is insufficient
    assert_item(&items[0], "system", "sys");
    assert_item(&items[1], "developer", "dev");
    assert_item(&items[2], "user", "usr");
    assert_item(&items[3], "assistant", "ast");
}

/// Tool role with tool_call_id: verify the tool output content survives.
#[test]
fn tool_role_content_preserved() {
    let chat = ChatHandle::new(None).unwrap();
    let json = r#"[
        {"role":"user","content":"weather?"},
        {"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"NYC\"}"}}]},
        {"role":"tool","tool_call_id":"call_1","content":"72F sunny"}
    ]"#;
    chat.load_completions_json(json).unwrap();

    // Verify via responses JSON that the tool output landed correctly.
    // The exact item structure depends on how tool calls map to conversation
    // items, but the tool output text must appear somewhere.
    let responses_json = chat.to_responses_json(1).unwrap();
    assert!(
        responses_json.contains("72F sunny"),
        "tool output must appear in serialized conversation: {}",
        &responses_json[..responses_json.len().min(500)]
    );
    assert!(
        responses_json.contains("get_weather"),
        "tool name must appear in serialized conversation"
    );
}

/// 10+ turn conversation tests allocator behavior under repeated appends.
#[test]
fn multi_turn_conversation() {
    let mut messages = Vec::new();
    messages.push(r#"{"role":"system","content":"You are a calculator."}"#.to_string());
    for i in 0..10 {
        messages.push(format!(
            r#"{{"role":"user","content":"Turn {} question"}}"#,
            i
        ));
        messages.push(format!(
            r#"{{"role":"assistant","content":"Turn {} answer"}}"#,
            i
        ));
    }
    let json = format!("[{}]", messages.join(","));

    let chat = ChatHandle::new(None).unwrap();
    chat.load_completions_json(&json).unwrap();

    let items = read_items(&chat);
    assert_eq!(items.len(), 21, "system + 10 user/assistant pairs");

    // Spot-check first, middle, and last
    assert_item(&items[0], "system", "You are a calculator.");
    assert_item(&items[1], "user", "Turn 0 question");
    assert_item(&items[2], "assistant", "Turn 0 answer");
    assert_item(&items[19], "user", "Turn 9 question");
    assert_item(&items[20], "assistant", "Turn 9 answer");
}

// =============================================================================
// Mutation semantics
// =============================================================================

/// load_completions_json clears existing conversation state.
#[test]
fn load_clears_prior_state() {
    let chat = ChatHandle::new(Some("old system")).unwrap();
    chat.append_user_message("old message").unwrap();
    assert_eq!(unsafe { talu_sys::talu_chat_len(chat.as_ptr()) }, 2);

    chat.load_completions_json(r#"[{"role":"user","content":"new"}]"#)
        .unwrap();

    let items = read_items(&chat);
    assert_eq!(items.len(), 1, "load must replace, not append");
    assert_item(&items[0], "user", "new");
}

/// Two consecutive loads — second must fully replace first.
#[test]
fn consecutive_loads_replace() {
    let chat = ChatHandle::new(None).unwrap();

    chat.load_completions_json(r#"[{"role":"user","content":"first"}]"#)
        .unwrap();
    let items1 = read_items(&chat);
    assert_eq!(items1.len(), 1);
    assert_item(&items1[0], "user", "first");

    chat.load_completions_json(
        r#"[{"role":"system","content":"second"},{"role":"user","content":"third"}]"#,
    )
    .unwrap();
    let items2 = read_items(&chat);
    assert_eq!(items2.len(), 2);
    assert_item(&items2[0], "system", "second");
    assert_item(&items2[1], "user", "third");
}

/// Empty array clears everything.
#[test]
fn empty_array_clears() {
    let chat = ChatHandle::new(None).unwrap();
    chat.append_user_message("something").unwrap();
    chat.load_completions_json("[]").unwrap();

    assert_eq!(unsafe { talu_sys::talu_chat_len(chat.as_ptr()) }, 0);
}

// =============================================================================
// Error paths
// =============================================================================

#[test]
fn invalid_json_returns_error() {
    let chat = ChatHandle::new(None).unwrap();
    assert!(chat.load_completions_json("{not valid").is_err());
}

#[test]
fn not_an_array_returns_error() {
    let chat = ChatHandle::new(None).unwrap();
    assert!(chat
        .load_completions_json(r#"{"role":"user","content":"hi"}"#)
        .is_err());
}

#[test]
fn missing_role_returns_error() {
    let chat = ChatHandle::new(None).unwrap();
    assert!(chat
        .load_completions_json(r#"[{"content":"no role"}]"#)
        .is_err());
}

#[test]
fn invalid_role_returns_error() {
    let chat = ChatHandle::new(None).unwrap();
    assert!(chat
        .load_completions_json(r#"[{"role":"bogus","content":"hi"}]"#)
        .is_err());
}

/// A failed load clears the original conversation (clear happens before
/// parse). Messages parsed before the error remain — the parser does not
/// roll back on failure.
#[test]
fn failed_load_clears_original_leaves_partial() {
    let chat = ChatHandle::new(None).unwrap();
    chat.append_user_message("original").unwrap();
    assert_eq!(unsafe { talu_sys::talu_chat_len(chat.as_ptr()) }, 1);

    // First message parses OK, second has no role → error
    let bad_json = r#"[{"role":"user","content":"ok"},{"content":"no role"}]"#;
    let result = chat.load_completions_json(bad_json);
    assert!(result.is_err());

    // "original" is gone (cleared). "ok" was appended before the error.
    let items = read_items(&chat);
    assert_eq!(items.len(), 1, "partial: first valid message survives");
    assert_item(&items[0], "user", "ok");
}

// =============================================================================
// CGenerateConfig.completions_mode — field layout and defaults
// =============================================================================

#[test]
fn completions_mode_default_off() {
    let cfg = talu_sys::CGenerateConfig::default();
    assert_eq!(
        cfg.completions_mode, 0,
        "completions_mode must default to off"
    );
}

/// Verify completions_mode and neighboring fields don't alias each other.
/// Sets completions_mode=1 with raw_output=0 and vice versa — if fields
/// overlap due to layout error, both would be nonzero.
#[test]
fn completions_mode_does_not_alias_neighbors() {
    let mut cfg = talu_sys::CGenerateConfig::default();

    cfg.completions_mode = 1;
    assert_eq!(
        cfg.raw_output, 0,
        "raw_output must stay 0 when completions_mode set"
    );
    assert_eq!(cfg.completions_mode, 1);

    cfg.completions_mode = 0;
    cfg.raw_output = 1;
    assert_eq!(
        cfg.completions_mode, 0,
        "completions_mode must stay 0 when raw_output set"
    );
    assert_eq!(cfg.raw_output, 1);
}

// =============================================================================
// Compile-time ABI signature check
// =============================================================================

/// Verify `talu_chat_load_completions_json` takes (handle, ptr, len) → i32.
/// Wrong types (e.g., CString instead of ptr+len) fail to compile.
const _: unsafe extern "C" fn(*mut std::ffi::c_void, *const u8, usize) -> i32 =
    talu_sys::talu_chat_load_completions_json;

// =============================================================================
// Helpers
// =============================================================================

fn read_items(chat: &ChatHandle) -> Vec<serde_json::Value> {
    let json = chat.to_responses_json(1).unwrap();
    serde_json::from_str(&json).unwrap()
}

fn item_role(item: &serde_json::Value) -> &str {
    item.get("role").and_then(|r| r.as_str()).unwrap_or("")
}

fn item_text(item: &serde_json::Value) -> String {
    let content = match item.get("content") {
        Some(c) => c,
        None => return String::new(),
    };
    match content {
        serde_json::Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(""),
        serde_json::Value::String(s) => s.clone(),
        _ => String::new(),
    }
}

fn assert_item(item: &serde_json::Value, expected_role: &str, expected_text: &str) {
    assert_eq!(
        item_role(item),
        expected_role,
        "role mismatch in item: {:?}",
        item
    );
    assert_eq!(
        item_text(item),
        expected_text,
        "text mismatch for role '{}' in item: {:?}",
        expected_role,
        item
    );
}
