//! `/v1/files` retrieval/list/error-path tests.

use super::files_config;
use crate::server::common::{delete, get, post_json, send_request, ServerTestContext};
use tempfile::TempDir;

fn upload_text_file(ctx: &ServerTestContext, filename: &str, mime: &str, payload: &str) -> String {
    let boundary = "----talu-file-read-upload";
    let body = format!(
        "--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: {mime}\r\n\r\n{payload}\r\n--{b}--\r\n",
        b = boundary,
        filename = filename,
        mime = mime,
        payload = payload,
    );
    let content_type = format!("multipart/form-data; boundary={}", boundary);
    let headers = [("Content-Type", content_type.as_str())];

    let resp = send_request(ctx.addr(), "POST", "/v1/files", &headers, Some(&body));
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    resp.json()["id"].as_str().expect("file id").to_string()
}

#[test]
fn get_content_and_delete_return_404_for_unknown_file() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    let id = "file_missing_123";

    let get_resp = get(ctx.addr(), &format!("/v1/files/{}", id));
    assert_eq!(get_resp.status, 404, "body: {}", get_resp.body);

    let content_resp = get(ctx.addr(), &format!("/v1/files/{}/content", id));
    assert_eq!(content_resp.status, 404, "body: {}", content_resp.body);

    let delete_resp = delete(ctx.addr(), &format!("/v1/files/{}", id));
    assert_eq!(delete_resp.status, 404, "body: {}", delete_resp.body);
}

#[test]
fn files_alias_routes_work_for_list_get_content_and_delete() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    let payload = "alias route payload";
    let file_id = upload_text_file(&ctx, "alias.txt", "text/plain", payload);

    let list_resp = get(ctx.addr(), "/files");
    assert_eq!(list_resp.status, 200, "body: {}", list_resp.body);

    let get_resp = get(ctx.addr(), &format!("/files/{}", file_id));
    assert_eq!(get_resp.status, 200, "body: {}", get_resp.body);

    let content_resp = get(ctx.addr(), &format!("/files/{}/content", file_id));
    assert_eq!(content_resp.status, 200, "body: {}", content_resp.body);
    assert_eq!(content_resp.body, payload);

    let delete_resp = delete(ctx.addr(), &format!("/files/{}", file_id));
    assert_eq!(delete_resp.status, 200, "body: {}", delete_resp.body);
}

#[test]
fn content_response_uses_uploaded_mime_type() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    let file_id = upload_text_file(&ctx, "image-name.bin", "image/png", "not-a-real-png");

    let content_resp = get(ctx.addr(), &format!("/v1/files/{}/content", file_id));
    assert_eq!(content_resp.status, 200, "body: {}", content_resp.body);

    let ct = content_resp.header("content-type").unwrap_or("");
    assert!(ct.contains("image/png"), "unexpected content-type: {ct}");
}

#[test]
fn list_respects_limit_and_sets_has_more() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    upload_text_file(&ctx, "a.txt", "text/plain", "aaa");
    upload_text_file(&ctx, "b.txt", "text/plain", "bbb");

    let list_resp = get(ctx.addr(), "/v1/files?limit=1");
    assert_eq!(list_resp.status, 200, "body: {}", list_resp.body);

    let json = list_resp.json();
    assert_eq!(json["object"], "list");
    assert_eq!(json["has_more"], true);
    assert_eq!(json["data"].as_array().map(|a| a.len()), Some(1));
}

#[test]
fn list_only_includes_file_documents() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    let file_id = upload_text_file(&ctx, "only-file.txt", "text/plain", "file-payload");

    let doc_body = serde_json::json!({
        "type": "note",
        "title": "plain note",
        "content": { "hello": "world" }
    });
    let doc_resp = post_json(ctx.addr(), "/v1/documents", &doc_body);
    assert_eq!(doc_resp.status, 201, "body: {}", doc_resp.body);
    let note_id = doc_resp.json()["id"].as_str().expect("note id").to_string();

    let list_resp = get(ctx.addr(), "/v1/files");
    assert_eq!(list_resp.status, 200, "body: {}", list_resp.body);
    let data = list_resp.json()["data"]
        .as_array()
        .expect("data should be an array")
        .clone();

    assert!(data.iter().any(|v| v["id"] == file_id));
    assert!(!data.iter().any(|v| v["id"] == note_id));
}

#[test]
fn get_file_returns_500_when_metadata_missing_blob_ref() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    let create_resp = post_json(
        ctx.addr(),
        "/v1/documents",
        &serde_json::json!({
            "type": "file",
            "title": "broken-file-metadata",
            "content": { "no_blob_ref_here": true }
        }),
    );
    assert_eq!(create_resp.status, 201, "body: {}", create_resp.body);
    let file_like_id = create_resp.json()["id"].as_str().expect("id").to_string();

    let get_resp = get(ctx.addr(), &format!("/v1/files/{}", file_like_id));
    assert_eq!(get_resp.status, 500, "body: {}", get_resp.body);
    let json = get_resp.json();
    assert_eq!(json["error"]["code"], "invalid_file_metadata");
}

#[test]
fn upload_requires_multipart_content_type_header() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    let no_header_resp = send_request(ctx.addr(), "POST", "/v1/files", &[], Some(""));
    assert_eq!(no_header_resp.status, 400, "body: {}", no_header_resp.body);
    assert_eq!(
        no_header_resp.json()["error"]["code"],
        "invalid_content_type"
    );

    let bad_header_resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/files",
        &[("Content-Type", "application/json")],
        Some("{}"),
    );
    assert_eq!(
        bad_header_resp.status, 400,
        "body: {}",
        bad_header_resp.body
    );
    assert_eq!(
        bad_header_resp.json()["error"]["code"],
        "invalid_content_type"
    );
}
