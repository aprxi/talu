//! `/v1/responses` file reference integration tests.

use crate::server::common::{
    model_config, post_json, require_model, send_request, ServerConfig, ServerTestContext,
};
use tempfile::TempDir;

fn upload_text_file(ctx: &ServerTestContext, filename: &str, content: &str) -> String {
    let boundary = "----talu-file-responses";
    let body = format!(
        "--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: text/plain\r\n\r\n{content}\r\n--{b}--\r\n",
        b = boundary
    );
    let content_type = format!("multipart/form-data; boundary={}", boundary);
    let headers = [("Content-Type", content_type.as_str())];
    let resp = send_request(ctx.addr(), "POST", "/v1/files", &headers, Some(&body));
    assert_eq!(resp.status, 200, "upload failed: {}", resp.body);
    let id = resp.json()["id"].as_str().expect("file id").to_string();
    assert!(id.starts_with("file_"), "expected file_ id, got {id}");
    id
}

fn upload_fake_image(ctx: &ServerTestContext, filename: &str) -> String {
    let boundary = "----talu-file-responses-img";
    let body = format!(
        "--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: image/jpeg\r\n\r\nfake-jpeg-bytes-for-test\r\n--{b}--\r\n",
        b = boundary
    );
    let content_type = format!("multipart/form-data; boundary={}", boundary);
    let headers = [("Content-Type", content_type.as_str())];
    let resp = send_request(ctx.addr(), "POST", "/v1/files", &headers, Some(&body));
    assert_eq!(resp.status, 200, "image upload failed: {}", resp.body);
    let id = resp.json()["id"].as_str().expect("file id").to_string();
    assert!(id.starts_with("file_"), "expected file_ id, got {id}");
    id
}

fn model_config_with_bucket(bucket: &std::path::Path) -> ServerConfig {
    let mut cfg = model_config();
    cfg.bucket = Some(bucket.to_path_buf());
    cfg
}

#[test]
fn responses_file_reference_resolved_in_generation() {
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
                { "type": "input_text", "text": "Summarize the attached file." },
                { "type": "input_file", "file_url": file_id, "filename": "notes.txt" }
            ]
        }]
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(
        resp.status, 200,
        "structured input with file reference should succeed: {}",
        resp.body
    );
    assert_eq!(resp.json()["object"].as_str(), Some("response"));
}

#[test]
fn responses_image_reference_resolved_in_generation() {
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
                { "type": "input_text", "text": "Describe this image." },
                { "type": "input_image", "image_url": file_id }
            ]
        }]
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(
        resp.status, 200,
        "structured input with image reference should succeed: {}",
        resp.body
    );
    assert_eq!(resp.json()["object"].as_str(), Some("response"));
}

#[test]
fn responses_dangling_file_reference_does_not_crash_server() {
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
                { "type": "input_text", "text": "Read this file." },
                { "type": "input_file", "file_url": "file_nonexistent_999", "filename": "ghost.txt" }
            ]
        }]
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert!(
        resp.status == 200 || resp.status == 400,
        "dangling reference should return 200 or structured 400, got {} body: {}",
        resp.status,
        resp.body
    );
    if resp.status == 400 {
        let json = resp.json();
        assert!(json["error"].is_object(), "error should be structured JSON");
    }
}
