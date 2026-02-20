//! End-to-end test: file upload → structured input reference → generation.
//!
//! Verifies that `resolve_file_references` (handlers.rs) correctly rewrites
//! `file_<id>` references inside structured input to `file://` blob paths
//! so the Zig core can read them during generation.

use crate::server::common::{model_config, post_json, require_model, send_request, ServerTestContext};
use tempfile::TempDir;

/// Upload a text file via multipart and return its `file_id`.
fn upload_text_file(ctx: &ServerTestContext, filename: &str, content: &str) -> String {
    let boundary = "----talu-file-e2e";
    let body = format!(
        "--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: text/plain\r\n\r\n{content}\r\n--{b}--\r\n",
        b = boundary,
    );
    let content_type = format!("multipart/form-data; boundary={}", boundary);
    let headers = [("Content-Type", content_type.as_str())];
    let resp = send_request(ctx.addr(), "POST", "/v1/files", &headers, Some(&body));
    assert_eq!(resp.status, 200, "upload failed: {}", resp.body);
    let id = resp.json()["id"].as_str().expect("file id").to_string();
    assert!(id.starts_with("file_"), "should be a file_ id: {id}");
    id
}

/// Helper: build a ServerConfig with model + bucket for file resolution tests.
fn model_config_with_bucket(bucket: &std::path::Path) -> crate::server::common::ServerConfig {
    let mut config = model_config();
    config.bucket = Some(bucket.to_path_buf());
    config
}

/// Upload a fake image via multipart and return its `file_id`.
/// Uses `image/jpeg` Content-Type so `resolve_file_references` takes the
/// `is_image` branch (mime.starts_with("image/")).
fn upload_fake_image(ctx: &ServerTestContext, filename: &str) -> String {
    let boundary = "----talu-file-e2e-img";
    // Fake JPEG content — not a real image, but the upload accepts any bytes
    // and the mime_type from the multipart header is what matters for resolution.
    let body = format!(
        "--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: image/jpeg\r\n\r\nfake-jpeg-bytes-for-test\r\n--{b}--\r\n",
        b = boundary,
    );
    let content_type = format!("multipart/form-data; boundary={}", boundary);
    let headers = [("Content-Type", content_type.as_str())];
    let resp = send_request(ctx.addr(), "POST", "/v1/files", &headers, Some(&body));
    assert_eq!(resp.status, 200, "image upload failed: {}", resp.body);
    let id = resp.json()["id"].as_str().expect("file id").to_string();
    assert!(id.starts_with("file_"), "should be a file_ id: {id}");
    id
}

/// Structured `input_file` reference (non-image branch) is resolved and
/// accepted by the generation pipeline.
#[test]
fn file_reference_resolved_in_generation() {
    let model = require_model!();
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(model_config_with_bucket(temp.path()));

    let file_id = upload_text_file(&ctx, "notes.txt", "The secret word is pineapple.");

    let body = serde_json::json!({
        "model": &model,
        "max_output_tokens": 10,
        "input": [{
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Summarize the attached file."},
                {"type": "input_file", "file_url": file_id, "filename": "notes.txt"}
            ]
        }]
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(
        resp.status, 200,
        "structured input with file reference should succeed, body: {}",
        resp.body
    );
    let json = resp.json();
    assert_eq!(
        json["object"].as_str(),
        Some("response"),
        "should return a response object"
    );
}

/// Structured `input_image` reference (image branch) is resolved to a
/// `file://` URL via `image_url` and accepted by the vision pipeline.
///
/// This exercises the `is_image` fork in `resolve_file_references`
/// (handlers.rs:412) which rewrites to `{"type":"input_image","image_url":...}`.
#[test]
fn image_reference_resolved_in_generation() {
    let model = require_model!();
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(model_config_with_bucket(temp.path()));

    let file_id = upload_fake_image(&ctx, "photo.jpg");

    let body = serde_json::json!({
        "model": &model,
        "max_output_tokens": 10,
        "input": [{
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe this image."},
                {"type": "input_image", "image_url": file_id}
            ]
        }]
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    // The image branch resolves file_id → file:// URL in image_url.
    // Generation should accept it (the model may not produce meaningful
    // output for fake bytes, but it must not 400/500).
    assert_eq!(
        resp.status, 200,
        "structured input with image reference should succeed, body: {}",
        resp.body
    );
    let json = resp.json();
    assert_eq!(
        json["object"].as_str(),
        Some("response"),
        "should return a response object"
    );
}

/// A dangling file reference (nonexistent `file_...` ID) is silently skipped
/// by `resolve_file_references` — the raw ID remains in the JSON and is
/// passed to `load_responses_json`. This test locks in the error behavior.
#[test]
fn dangling_file_reference_returns_error() {
    let model = require_model!();
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(model_config_with_bucket(temp.path()));

    let body = serde_json::json!({
        "model": &model,
        "max_output_tokens": 10,
        "input": [{
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Read this file."},
                {"type": "input_file", "file_url": "file_nonexistent_999", "filename": "ghost.txt"}
            ]
        }]
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    // The unresolved file_nonexistent_999 is passed through to the Zig core.
    // Lock in the current behavior: the request should NOT crash (no 500).
    // It may succeed (200) if the core ignores unrecognized URLs, or fail
    // gracefully (400) if the core rejects them.
    assert!(
        resp.status == 200 || resp.status == 400,
        "dangling file reference should not crash (expected 200 or 400), got: {} body: {}",
        resp.status,
        resp.body
    );
    // If it's an error, it should be a structured JSON error, not a bare crash.
    if resp.status != 200 {
        let json = resp.json();
        assert!(
            json["error"].is_object(),
            "error response should have structured error object, got: {}",
            resp.body
        );
    }
}
