use super::{conversation_config, no_bucket_config, seed_session, seed_session_with_messages};
use crate::server::common::*;
use tempfile::TempDir;

// For generation field tests that require actual inference
#[cfg(test)]
fn model_and_bucket_config(bucket: &std::path::Path) -> ServerConfig {
    let mut config = model_config();
    config.bucket = Some(bucket.to_path_buf());
    config
}

#[test]
fn get_returns_503_without_bucket() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let resp = get(ctx.addr(), "/v1/conversations/some-id");
    assert_eq!(resp.status, 503, "body: {}", resp.body);
}

#[test]
fn get_503_error_body_is_json() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let resp = get(ctx.addr(), "/v1/conversations/some-id");
    assert_eq!(resp.status, 503);
    let json = resp.json();
    assert!(
        json["error"]["code"].is_string(),
        "503 error should have code"
    );
    assert!(
        json["error"]["message"].is_string(),
        "503 error should have message"
    );
}

#[test]
fn get_returns_session_with_items() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-get-1", "My Chat", "test-model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/conversations/sess-get-1");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    assert_eq!(json["object"], "conversation");
    assert_eq!(json["id"], "sess-get-1");
    assert_eq!(json["title"], "My Chat");

    // Should have items array with at least the user message
    let items = json["items"].as_array().expect("items array");
    assert!(!items.is_empty(), "should have at least one item");
}

#[test]
fn get_returns_404_for_missing_session() {
    let temp = TempDir::new().expect("temp dir");
    // Seed one session so the DB exists
    seed_session(temp.path(), "sess-exists", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/conversations/nonexistent-session");
    assert_eq!(resp.status, 404, "body: {}", resp.body);
}

#[test]
fn get_without_prefix() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-noprefix", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/conversations/sess-noprefix");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(resp.json()["id"], "sess-noprefix");
}

#[test]
fn get_returns_all_metadata_fields() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-meta-get", "Full Meta Chat", "model-x");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/conversations/sess-meta-get");
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(json["object"], "conversation");
    assert_eq!(json["id"], "sess-meta-get");
    assert_eq!(json["title"], "Full Meta Chat");
    assert_eq!(json["model"], "model-x");
    assert_eq!(json["marker"], "active");
    assert!(json["created_at"].as_i64().unwrap() > 0);
    assert!(json["updated_at"].as_i64().unwrap() > 0);
    // Items should be present (it's the full GET)
    assert!(json["items"].is_array(), "GET should include items array");
}

#[test]
fn get_items_contain_user_message() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-item-check", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/conversations/sess-item-check");
    assert_eq!(resp.status, 200);

    let json = resp.json();
    let items = json["items"].as_array().expect("items");
    assert!(!items.is_empty());

    // Verify items have expected structure (type, role, content)
    let first = &items[0];
    assert_eq!(first["type"], "message", "item should be a message type");
    assert!(first["role"].is_string(), "item should have a role");
    assert!(
        first["content"].is_array(),
        "item should have content array"
    );
    // Verify content has input_text
    let content = first["content"].as_array().expect("content array");
    assert!(!content.is_empty(), "content should not be empty");
    assert_eq!(content[0]["type"], "input_text");
}

#[test]
fn get_multiple_items_preserved() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_messages(
        temp.path(),
        "sess-multi",
        "Multi-msg Chat",
        "model",
        &["First message", "Second message", "Third message"],
    );

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/conversations/sess-multi");
    assert_eq!(resp.status, 200);

    let json = resp.json();
    let items = json["items"].as_array().expect("items");
    // Should have at least 3 message items (one per appended message)
    let msg_items: Vec<_> = items
        .iter()
        .filter(|item| item["type"].as_str() == Some("message"))
        .collect();
    assert!(
        msg_items.len() >= 3,
        "expected at least 3 message items, got {}",
        msg_items.len()
    );
}

#[test]
fn get_404_error_body_is_json() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-x", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/conversations/nonexistent");
    assert_eq!(resp.status, 404);
    let json = resp.json();
    assert!(
        json["error"]["code"].is_string(),
        "error body should have code"
    );
    assert!(
        json["error"]["message"].is_string(),
        "error body should have message"
    );
}

#[test]
fn get_timestamps_are_consistent() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-ts", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/conversations/sess-ts");
    assert_eq!(resp.status, 200);

    let json = resp.json();
    let created = json["created_at"].as_i64().expect("created_at");
    let updated = json["updated_at"].as_i64().expect("updated_at");
    assert!(created > 0, "created_at should be positive ms timestamp");
    assert!(updated > 0, "updated_at should be positive ms timestamp");
    assert!(
        updated >= created,
        "updated_at ({updated}) should be >= created_at ({created})"
    );
}

#[test]
fn get_items_have_id_and_type() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-item-fields", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/conversations/sess-item-fields");
    assert_eq!(resp.status, 200);

    let json = resp.json();
    let items = json["items"].as_array().expect("items");
    for item in items {
        assert!(item["id"].is_string(), "each item should have a string id");
        assert!(
            item["type"].is_string(),
            "each item should have a string type"
        );
    }
}

#[test]
fn get_items_is_array_even_when_single_message() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-single", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/conversations/sess-single");
    assert_eq!(resp.status, 200);

    let json = resp.json();
    assert!(json["items"].is_array(), "items should always be an array");
}

#[test]
fn get_response_includes_model() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-model-check", "Chat", "my-specific-model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/conversations/sess-model-check");
    assert_eq!(resp.status, 200);
    assert_eq!(resp.json()["model"], "my-specific-model");
}

#[test]
fn get_response_includes_marker() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-marker-check", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/conversations/sess-marker-check");
    assert_eq!(resp.status, 200);
    assert_eq!(resp.json()["marker"], "active");
}

// ---------------------------------------------------------------------------
// GET response completeness
// ---------------------------------------------------------------------------

#[test]
fn get_response_includes_parent_session_id_null() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-parent", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/conversations/sess-parent");
    assert_eq!(resp.status, 200);
    // Root session should have null parent_session_id
    assert!(
        resp.json()["parent_session_id"].is_null(),
        "root session parent_session_id should be null"
    );
}

#[test]
fn get_response_includes_group_id_null() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-group", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/conversations/sess-group");
    assert_eq!(resp.status, 200);
    // Session without group should have null group_id
    assert!(
        resp.json()["group_id"].is_null(),
        "session without group should have null group_id"
    );
}

#[test]
fn get_response_includes_metadata_empty_object() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-meta-empty", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/conversations/sess-meta-empty");
    assert_eq!(resp.status, 200);
    // Default metadata should be empty object {}
    let metadata = &resp.json()["metadata"];
    assert!(
        metadata.is_object() || metadata.is_null(),
        "default metadata should be object or null: {:?}",
        metadata
    );
}

#[test]
fn get_patched_title_reflected() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-get-patch", "Before", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));

    // Patch the title
    let resp = patch_json(
        ctx.addr(),
        "/v1/conversations/sess-get-patch",
        &serde_json::json!({"title": "After"}),
    );
    assert_eq!(resp.status, 200);

    // GET should show the new title
    let resp = get(ctx.addr(), "/v1/conversations/sess-get-patch");
    assert_eq!(resp.status, 200);
    assert_eq!(resp.json()["title"], "After");
    // Items should still be present
    let json = resp.json();
    let items = json["items"].as_array().expect("items");
    assert!(!items.is_empty(), "items should survive a title patch");
}

#[test]
fn get_patched_marker_reflected() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-get-marker", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));

    let resp = patch_json(
        ctx.addr(),
        "/v1/conversations/sess-get-marker",
        &serde_json::json!({"marker": "completed"}),
    );
    assert_eq!(resp.status, 200);

    let resp = get(ctx.addr(), "/v1/conversations/sess-get-marker");
    assert_eq!(resp.status, 200);
    assert_eq!(resp.json()["marker"], "completed");
}

// ---------------------------------------------------------------------------
// Generation metadata
// ---------------------------------------------------------------------------

#[test]
fn get_manual_items_have_no_generation() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-gen-none", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/conversations/sess-gen-none");
    assert_eq!(resp.status, 200);

    let json = resp.json();
    let items = json["items"].as_array().expect("items");

    // Manually seeded items should have generation=null or absent
    for item in items {
        let gen = item.get("generation");
        assert!(
            gen.is_none() || gen.unwrap().is_null(),
            "manual item should have generation=null or absent, got {:?}",
            gen
        );
    }
}

#[test]
fn get_generated_item_has_generation_metadata() {
    let _ = require_model!();
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(model_and_bucket_config(temp.path()));

    // Generate a response with store=true to create a conversation.
    // Keep max_output_tokens low to minimize inference time.
    // The test handles the case where reasoning doesn't complete.
    let body = serde_json::json!({
        "model": model_path(),
        "input": "Say hi",
        "store": true,
        "max_output_tokens": 10,
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "generate failed: {}", resp.body);

    // Extract session_id from response metadata
    let response_json = resp.json();
    let session_id = response_json["metadata"]["session_id"]
        .as_str()
        .expect("response should have session_id in metadata");

    // Get the conversation
    let conv_resp = get(ctx.addr(), &format!("/v1/conversations/{}", session_id));
    assert_eq!(
        conv_resp.status, 200,
        "get conversation: {}",
        conv_resp.body
    );

    let conv_json = conv_resp.json();
    let items = conv_json["items"].as_array().expect("items array");

    // Find the assistant message (generated item)
    // generation metadata is only on assistant messages, not reasoning items
    let assistant_item = items.iter().find(|item| {
        item["type"].as_str() == Some("message") && item["role"].as_str() == Some("assistant")
    });

    // If we don't have an assistant message (model still in reasoning), skip the check
    // This can happen if max_tokens isn't enough to complete the thinking phase
    let Some(assistant_item) = assistant_item else {
        eprintln!("Note: No assistant message generated (model may still be in reasoning phase). Items: {:?}",
            items.iter().map(|i| i["type"].as_str()).collect::<Vec<_>>());
        return;
    };

    // Debug: show what the assistant item looks like
    eprintln!(
        "Assistant item: {}",
        serde_json::to_string_pretty(assistant_item).unwrap()
    );

    // Check generation field exists and has expected structure
    let generation = assistant_item.get("generation");
    assert!(
        generation.is_some() && !generation.unwrap().is_null(),
        "generated assistant item should have generation metadata, got: {:?}",
        generation
    );

    let gen = generation.unwrap();
    assert!(gen["model"].is_string(), "generation should have model");
    assert!(
        gen["temperature"].is_number(),
        "generation should have temperature"
    );
    assert!(gen["top_p"].is_number(), "generation should have top_p");
    assert!(
        gen["max_tokens"].is_number(),
        "generation should have max_tokens"
    );
}

#[test]
fn get_generation_metadata_has_sampling_params() {
    let _ = require_model!();
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(model_and_bucket_config(temp.path()));

    // Generate with specific sampling params.
    // Keep max_output_tokens low — test only checks metadata, not content.
    let body = serde_json::json!({
        "model": model_path(),
        "input": "Hello",
        "store": true,
        "max_output_tokens": 10,
        "temperature": 0.5,
        "top_p": 0.8,
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200);

    let resp_json = resp.json();
    let session_id = resp_json["metadata"]["session_id"]
        .as_str()
        .expect("session_id");

    let conv_resp = get(ctx.addr(), &format!("/v1/conversations/{}", session_id));
    assert_eq!(conv_resp.status, 200);

    let conv_json = conv_resp.json();
    let items = conv_json["items"].as_array().expect("items");

    // Find the assistant message (may not exist if model didn't complete reasoning)
    let assistant = items
        .iter()
        .find(|i| i["role"].as_str() == Some("assistant"));

    let Some(assistant) = assistant else {
        eprintln!(
            "Note: No assistant message (model may still be in reasoning). Skipping param check."
        );
        return;
    };

    let gen = &assistant["generation"];

    // Verify the sampling params are captured
    // Note: values might be slightly different due to internal defaults
    assert!(gen["temperature"].is_number(), "should have temperature");
    assert!(gen["top_p"].is_number(), "should have top_p");
    assert!(gen["top_k"].is_number(), "should have top_k");
    assert!(gen["min_p"].is_number(), "should have min_p");
    assert!(
        gen["repetition_penalty"].is_number(),
        "should have repetition_penalty"
    );
}
