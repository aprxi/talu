//! Integration tests for `POST /v1/files/batch` endpoint.

use super::{files_config, no_bucket_config};
use crate::server::common::*;
use serde_json::json;
use tempfile::TempDir;

fn upload_text_file(ctx: &ServerTestContext, filename: &str, mime: &str, payload: &str) -> String {
    let boundary = "----talu-file-batch-test";
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
    assert_eq!(resp.status, 200, "upload body: {}", resp.body);
    resp.json()["id"].as_str().expect("file id").to_string()
}

// ---------------------------------------------------------------------------
// Batch delete
// ---------------------------------------------------------------------------

/// Batch delete removes multiple files.
#[test]
fn batch_delete_removes_files() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    let id1 = upload_text_file(&ctx, "del-1.txt", "text/plain", "delete-me-1");
    let id2 = upload_text_file(&ctx, "del-2.txt", "text/plain", "delete-me-2");
    let id3 = upload_text_file(&ctx, "keep.txt", "text/plain", "keep-me");

    // Delete id1 and id2
    let resp = post_json(
        ctx.addr(),
        "/v1/files/batch",
        &json!({
            "action": "delete",
            "ids": [id1, id2]
        }),
    );
    assert_eq!(resp.status, 204, "body: {}", resp.body);

    // Verify only id3 remains
    let list_resp = get(ctx.addr(), "/v1/files");
    let list_json = list_resp.json();
    let data = list_json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], id3);
}

/// Batch delete is idempotent (deleting non-existent is OK).
#[test]
fn batch_delete_idempotent() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    let id1 = upload_text_file(&ctx, "idem.txt", "text/plain", "idempotent-content");

    // Delete id1 and non-existent file
    let resp = post_json(
        ctx.addr(),
        "/v1/files/batch",
        &json!({
            "action": "delete",
            "ids": [id1, "file_nonexistent_99"]
        }),
    );
    assert_eq!(resp.status, 204);
}

// ---------------------------------------------------------------------------
// Batch archive/unarchive
// ---------------------------------------------------------------------------

/// Batch archive sets marker to "archived".
#[test]
fn batch_archive_sets_marker() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    let id1 = upload_text_file(&ctx, "arch-1.txt", "text/plain", "archive-content-1");
    let id2 = upload_text_file(&ctx, "arch-2.txt", "text/plain", "archive-content-2");

    // Archive id1
    let resp = post_json(
        ctx.addr(),
        "/v1/files/batch",
        &json!({
            "action": "archive",
            "ids": [id1]
        }),
    );
    assert_eq!(resp.status, 204);

    // Verify id1 has marker "archived"
    let get_resp = get(ctx.addr(), &format!("/v1/files/{id1}"));
    let get_json = get_resp.json();
    assert_eq!(get_json["marker"], "archived");

    // id2 should still have "active" marker
    let get_resp2 = get(ctx.addr(), &format!("/v1/files/{id2}"));
    let get_json2 = get_resp2.json();
    assert_eq!(get_json2["marker"], "active");
}

/// Batch unarchive restores the marker to "active".
#[test]
fn batch_unarchive_clears_marker() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    let id1 = upload_text_file(&ctx, "unarch.txt", "text/plain", "unarchive-content");

    // First archive
    post_json(
        ctx.addr(),
        "/v1/files/batch",
        &json!({
            "action": "archive",
            "ids": [id1]
        }),
    );

    // Then unarchive
    let resp = post_json(
        ctx.addr(),
        "/v1/files/batch",
        &json!({
            "action": "unarchive",
            "ids": [id1]
        }),
    );
    assert_eq!(resp.status, 204);

    // Verify marker is "active"
    let get_resp = get(ctx.addr(), &format!("/v1/files/{id1}"));
    let get_json = get_resp.json();
    assert_eq!(get_json["marker"], "active");
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Invalid action returns 400.
#[test]
fn batch_invalid_action() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    let resp = post_json(
        ctx.addr(),
        "/v1/files/batch",
        &json!({
            "action": "invalid_action",
            "ids": ["file_abc"]
        }),
    );
    assert_eq!(resp.status, 400);
    let json = resp.json();
    assert_eq!(json["error"]["code"], "invalid_action");
}

/// Empty ids returns 400.
#[test]
fn batch_empty_ids() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    let resp = post_json(
        ctx.addr(),
        "/v1/files/batch",
        &json!({
            "action": "delete",
            "ids": []
        }),
    );
    assert_eq!(resp.status, 400);
    let json = resp.json();
    assert_eq!(json["error"]["code"], "missing_ids");
}

/// Batch size limit is enforced.
#[test]
fn batch_size_limit() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    // Create 101 IDs (over the 100 limit)
    let ids: Vec<String> = (0..101).map(|i| format!("file_{i}")).collect();

    let resp = post_json(
        ctx.addr(),
        "/v1/files/batch",
        &json!({
            "action": "delete",
            "ids": ids
        }),
    );
    assert_eq!(resp.status, 400);
    let json = resp.json();
    assert_eq!(json["error"]["code"], "batch_too_large");
}

/// Batch operations require storage (no_bucket returns 503).
#[test]
fn batch_no_bucket_returns_503() {
    let ctx = ServerTestContext::new(no_bucket_config());

    let resp = post_json(
        ctx.addr(),
        "/v1/files/batch",
        &json!({
            "action": "delete",
            "ids": ["file_abc"]
        }),
    );
    assert_eq!(resp.status, 503);
    let json = resp.json();
    assert_eq!(json["error"]["code"], "storage_unavailable");
}

/// Batch archive is idempotent.
#[test]
fn batch_archive_idempotent() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    let id1 = upload_text_file(&ctx, "idem-arch.txt", "text/plain", "idempotent-archive");

    // Archive twice
    let resp1 = post_json(
        ctx.addr(),
        "/v1/files/batch",
        &json!({
            "action": "archive",
            "ids": [id1]
        }),
    );
    assert_eq!(resp1.status, 204);

    let resp2 = post_json(
        ctx.addr(),
        "/v1/files/batch",
        &json!({
            "action": "archive",
            "ids": [id1]
        }),
    );
    assert_eq!(resp2.status, 204);

    // Verify still archived
    let get_resp = get(ctx.addr(), &format!("/v1/files/{id1}"));
    let get_json = get_resp.json();
    assert_eq!(get_json["marker"], "archived");
}

/// Batch endpoint works via alias route (without /v1 prefix).
#[test]
fn batch_alias_route() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    let id1 = upload_text_file(&ctx, "alias.txt", "text/plain", "alias-route-content");

    let resp = post_json(
        ctx.addr(),
        "/files/batch",
        &json!({
            "action": "archive",
            "ids": [id1]
        }),
    );
    assert_eq!(resp.status, 204);
}

// ---------------------------------------------------------------------------
// Cross-type safety
// ---------------------------------------------------------------------------

/// Batch delete via /v1/files/batch must not affect non-file documents.
/// A "prompt" document created via /v1/documents must survive a file batch delete.
#[test]
fn batch_delete_does_not_affect_non_file_documents() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    // Create a non-file document (type = "prompt") via the documents API.
    let create_resp = post_json(
        ctx.addr(),
        "/v1/documents",
        &json!({
            "type": "prompt",
            "title": "My Prompt",
            "content": {"text": "Hello world"}
        }),
    );
    assert_eq!(
        create_resp.status, 201,
        "create doc body: {}",
        create_resp.body
    );
    let prompt_id = create_resp.json()["id"]
        .as_str()
        .expect("doc id")
        .to_string();

    // Upload a real file.
    let file_id = upload_text_file(&ctx, "real-file.txt", "text/plain", "file-content");

    // Attempt to batch-delete both the prompt document and the file.
    let resp = post_json(
        ctx.addr(),
        "/v1/files/batch",
        &json!({
            "action": "delete",
            "ids": [prompt_id, file_id]
        }),
    );
    assert_eq!(resp.status, 204);

    // The file should be deleted.
    let file_resp = get(ctx.addr(), &format!("/v1/files/{file_id}"));
    assert_eq!(file_resp.status, 404, "file should be deleted");

    // The prompt document must still exist.
    let doc_resp = get(ctx.addr(), &format!("/v1/documents/{prompt_id}"));
    assert_eq!(
        doc_resp.status, 200,
        "prompt document must survive file batch delete"
    );
}

/// Batch archive via /v1/files/batch must not affect non-file documents.
#[test]
fn batch_archive_does_not_affect_non_file_documents() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    // Create a non-file document (type = "prompt").
    let create_resp = post_json(
        ctx.addr(),
        "/v1/documents",
        &json!({
            "type": "prompt",
            "title": "Safe Prompt",
            "content": {"text": "untouchable"}
        }),
    );
    assert_eq!(create_resp.status, 201);
    let prompt_id = create_resp.json()["id"]
        .as_str()
        .expect("doc id")
        .to_string();

    // Upload a file.
    let file_id = upload_text_file(&ctx, "to-archive.txt", "text/plain", "archive-me");

    // Batch archive both IDs.
    let resp = post_json(
        ctx.addr(),
        "/v1/files/batch",
        &json!({
            "action": "archive",
            "ids": [prompt_id, file_id]
        }),
    );
    assert_eq!(resp.status, 204);

    // File should be archived.
    let file_resp = get(ctx.addr(), &format!("/v1/files/{file_id}"));
    assert_eq!(file_resp.status, 200);
    assert_eq!(file_resp.json()["marker"], "archived");

    // Prompt should remain unmodified (marker should not be "archived").
    let doc_resp = get(ctx.addr(), &format!("/v1/documents/{prompt_id}"));
    assert_eq!(doc_resp.status, 200);
    let doc_marker = &doc_resp.json()["marker"];
    assert_ne!(
        doc_marker, "archived",
        "non-file document must not be archived by file batch: got {:?}",
        doc_marker
    );
}
