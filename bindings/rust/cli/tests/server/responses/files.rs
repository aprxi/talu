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

/// Structured input with an `input_file` reference is resolved and accepted
/// by the generation pipeline without returning a 400 or 500.
#[test]
fn file_reference_resolved_in_generation() {
    let model = require_model!();
    let temp = TempDir::new().expect("temp dir");

    // Need a config with both model and bucket (file storage).
    let mut config = model_config();
    config.bucket = Some(temp.path().to_path_buf());
    let ctx = ServerTestContext::new(config);

    // Upload a text file.
    let file_id = upload_text_file(&ctx, "notes.txt", "The secret word is pineapple.");

    // Submit a structured input that references the uploaded file.
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
    // The key assertion: file resolution + loading into the chat handle
    // succeeded. The generation may produce any output.
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
