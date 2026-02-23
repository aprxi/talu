//! `/v1/files` retrieval/list/error-path tests.

use super::{files_config, no_bucket_config};
use crate::server::common::{delete, get, patch_json, post_json, send_request, ServerTestContext};
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

fn get_with_headers(
    ctx: &ServerTestContext,
    path: &str,
    headers: &[(&str, &str)],
) -> crate::server::common::HttpResponse {
    send_request(ctx.addr(), "GET", path, headers, None)
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
fn content_response_uses_detected_mime_type() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    // Upload text content with a declared image/png MIME type.
    // The server inspects actual content and overrides the declared type.
    let file_id = upload_text_file(&ctx, "image-name.bin", "image/png", "not-a-real-png");

    let content_resp = get(ctx.addr(), &format!("/v1/files/{}/content", file_id));
    assert_eq!(content_resp.status, 200, "body: {}", content_resp.body);

    let ct = content_resp.header("content-type").unwrap_or("");
    assert!(
        ct.contains("text/plain"),
        "expected detected type text/plain, got: {ct}"
    );
}

#[test]
fn content_range_start_end_returns_206_partial_content() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    let payload = "0123456789";
    let file_id = upload_text_file(&ctx, "range.txt", "text/plain", payload);
    let path = format!("/v1/files/{}/content", file_id);
    let resp = get_with_headers(&ctx, &path, &[("Range", "bytes=2-5")]);

    assert_eq!(resp.status, 206, "body: {}", resp.body);
    assert_eq!(resp.body, "2345");
    assert_eq!(resp.header("content-range"), Some("bytes 2-5/10"));
    assert_eq!(resp.header("content-length"), Some("4"));
    assert_eq!(resp.header("accept-ranges"), Some("bytes"));
}

#[test]
fn content_range_open_ended_and_suffix_forms_are_supported() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    let payload = "abcdefghij";
    let file_id = upload_text_file(&ctx, "range-open.txt", "text/plain", payload);
    let path = format!("/v1/files/{}/content", file_id);

    let open_ended = get_with_headers(&ctx, &path, &[("Range", "bytes=6-")]);
    assert_eq!(open_ended.status, 206, "body: {}", open_ended.body);
    assert_eq!(open_ended.body, "ghij");
    assert_eq!(open_ended.header("content-range"), Some("bytes 6-9/10"));

    let suffix = get_with_headers(&ctx, &path, &[("Range", "bytes=-3")]);
    assert_eq!(suffix.status, 206, "body: {}", suffix.body);
    assert_eq!(suffix.body, "hij");
    assert_eq!(suffix.header("content-range"), Some("bytes 7-9/10"));
}

#[test]
fn content_range_unsatisfiable_returns_416_with_content_range_star() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    let payload = "abcdefghij";
    let file_id = upload_text_file(&ctx, "range-invalid.txt", "text/plain", payload);
    let path = format!("/v1/files/{}/content", file_id);
    let resp = get_with_headers(&ctx, &path, &[("Range", "bytes=99-120")]);

    assert_eq!(resp.status, 416, "body: {}", resp.body);
    assert_eq!(resp.header("content-range"), Some("bytes */10"));
    assert_eq!(resp.json()["error"]["code"], "invalid_range");
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
    let doc_resp = post_json(ctx.addr(), "/v1/db/tables/documents", &doc_body);
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
        "/v1/db/tables/documents",
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

// ---------------------------------------------------------------------------
// GET /v1/db/blobs/:hash — blob serving
// ---------------------------------------------------------------------------

/// Upload a file, extract blob_ref via document API, then GET the blob by hash.
#[test]
fn get_blob_returns_uploaded_content() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    let payload = "blob content for serving test";
    let file_id = upload_text_file(&ctx, "blob-test.txt", "text/plain", payload);

    // blob_ref lives in the underlying document content, not in the files API response.
    let doc_resp = get(ctx.addr(), &format!("/v1/db/tables/documents/{}", file_id));
    assert_eq!(doc_resp.status, 200, "body: {}", doc_resp.body);
    let doc_json = doc_resp.json();
    let blob_ref = doc_json["content"]["blob_ref"]
        .as_str()
        .expect("should have blob_ref in document content");

    // Extract hex hash from "sha256:<64hex>".
    let hash = blob_ref.strip_prefix("sha256:").expect("sha256: prefix");
    assert_eq!(hash.len(), 64, "hash should be 64 chars");

    let blob_resp = get(ctx.addr(), &format!("/v1/db/blobs/{}", hash));
    assert_eq!(blob_resp.status, 200, "body: {}", blob_resp.body);

    let ct = blob_resp.header("content-type").unwrap_or("");
    assert!(
        ct.contains("application/octet-stream"),
        "expected octet-stream, got: {ct}"
    );
    let cc = blob_resp.header("cache-control").unwrap_or("");
    assert!(
        cc.contains("immutable"),
        "should have immutable cache-control"
    );
    assert_eq!(
        blob_resp.body, payload,
        "blob content should match uploaded payload"
    );
}

/// Invalid (too-short) hash returns 400.
#[test]
fn get_blob_invalid_hash_returns_400() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    let resp = get(ctx.addr(), "/v1/db/blobs/tooshort");
    assert_eq!(resp.status, 400, "body: {}", resp.body);
}

/// Non-existent but valid-format hash returns error.
#[test]
fn get_blob_nonexistent_hash_returns_error() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    // 64 valid hex chars, but no such blob exists.
    let fake_hash = "a".repeat(64);
    let resp = get(ctx.addr(), &format!("/v1/db/blobs/{}", fake_hash));
    // Expect 404 or another non-200 error.
    assert_ne!(resp.status, 200, "non-existent blob should not return 200");
}

/// Blob endpoint returns 404 when no bucket is configured.
#[test]
fn get_blob_no_bucket_returns_404() {
    let ctx = ServerTestContext::new(no_bucket_config());

    let hash = "b".repeat(64);
    let resp = get(ctx.addr(), &format!("/v1/db/blobs/{}", hash));
    assert_eq!(resp.status, 404, "body: {}", resp.body);
}

// ---------------------------------------------------------------------------
// PATCH /v1/files/:id — file renaming and metadata update
// ---------------------------------------------------------------------------

/// PATCH with filename renames the file.
#[test]
fn patch_file_renames_filename() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    let file_id = upload_text_file(&ctx, "original.txt", "text/plain", "rename me");

    let resp = patch_json(
        ctx.addr(),
        &format!("/v1/files/{}", file_id),
        &serde_json::json!({"filename": "renamed.txt"}),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    assert_eq!(json["filename"], "renamed.txt");

    // Verify rename persists on subsequent GET.
    let get_resp = get(ctx.addr(), &format!("/v1/files/{}", file_id));
    assert_eq!(get_resp.status, 200, "body: {}", get_resp.body);
    assert_eq!(get_resp.json()["filename"], "renamed.txt");
}

/// PATCH sanitizes path-like filenames.
#[test]
fn patch_file_sanitizes_filename() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    let file_id = upload_text_file(&ctx, "safe.txt", "text/plain", "content");

    let resp = patch_json(
        ctx.addr(),
        &format!("/v1/files/{}", file_id),
        &serde_json::json!({"filename": "../../etc/passwd"}),
    );
    // Should either sanitize to "passwd" or reject — not allow path traversal.
    if resp.status == 200 {
        let json = resp.json();
        let name = json["filename"].as_str().unwrap_or("");
        assert!(
            !name.contains(".."),
            "filename should be sanitized, got: {name}"
        );
        assert_eq!(name, "passwd", "path components should be stripped");
    } else {
        // 400 is also acceptable for an empty/invalid result after sanitization.
        assert_eq!(resp.status, 400);
    }
}

/// PATCH with invalid marker returns 400.
#[test]
fn patch_file_invalid_marker_returns_400() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    let file_id = upload_text_file(&ctx, "marker.txt", "text/plain", "test");

    let resp = patch_json(
        ctx.addr(),
        &format!("/v1/files/{}", file_id),
        &serde_json::json!({"marker": "invalid_status"}),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
}

/// PATCH with empty body returns current state (no-op).
#[test]
fn patch_file_no_changes_returns_current_state() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    let file_id = upload_text_file(&ctx, "noop.txt", "text/plain", "unchanged");

    let resp = patch_json(
        ctx.addr(),
        &format!("/v1/files/{}", file_id),
        &serde_json::json!({}),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    assert_eq!(json["id"], file_id);
    assert_eq!(json["filename"], "noop.txt");
}

// ---------------------------------------------------------------------------
// Delete lifecycle: metadata removed, content inaccessible
// ---------------------------------------------------------------------------

/// After deleting a file, both metadata GET and content GET return 404.
///
/// The storage layer performs metadata-only deletion (CAS blob retained for GC),
/// so subsequent lookups fail at the metadata stage before reaching the blob.
#[test]
fn deleted_file_returns_404_for_metadata_and_content() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    let payload = "content that will be deleted";
    let file_id = upload_text_file(&ctx, "doomed.txt", "text/plain", payload);

    // Content is accessible before deletion.
    let content_resp = get(ctx.addr(), &format!("/v1/files/{}/content", file_id));
    assert_eq!(
        content_resp.status, 200,
        "content should be accessible before delete"
    );
    assert_eq!(content_resp.body, payload);

    // Delete the file.
    let del_resp = delete(ctx.addr(), &format!("/v1/files/{}", file_id));
    assert_eq!(del_resp.status, 200, "body: {}", del_resp.body);

    // Metadata GET → 404.
    let meta_resp = get(ctx.addr(), &format!("/v1/files/{}", file_id));
    assert_eq!(
        meta_resp.status, 404,
        "metadata GET should return 404 after delete, body: {}",
        meta_resp.body
    );

    // Content GET → 404.
    let content_resp = get(ctx.addr(), &format!("/v1/files/{}/content", file_id));
    assert_eq!(
        content_resp.status, 404,
        "content GET should return 404 after delete, body: {}",
        content_resp.body
    );
}

// ---------------------------------------------------------------------------
// CAS blob retention after metadata deletion
// ---------------------------------------------------------------------------

/// After deleting file metadata, the CAS blob is still accessible via
/// `GET /v1/db/blobs/:hash`.
///
/// `DELETE /v1/files/:id` removes the metadata document only. The blob
/// store uses content-addressed storage and retains blobs for GC, not
/// deleting them with metadata.
#[test]
fn deleted_file_blob_still_accessible_via_blob_endpoint() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    let payload = "blob survives deletion";
    let file_id = upload_text_file(&ctx, "cas-test.txt", "text/plain", payload);

    // Extract blob_ref from the underlying document.
    let doc_resp = get(ctx.addr(), &format!("/v1/db/tables/documents/{}", file_id));
    assert_eq!(doc_resp.status, 200, "body: {}", doc_resp.body);
    let doc_json = doc_resp.json();
    let blob_ref = doc_json["content"]["blob_ref"]
        .as_str()
        .expect("should have blob_ref in document content");
    let hash = blob_ref.strip_prefix("sha256:").expect("sha256: prefix");

    // Blob is accessible before deletion.
    let blob_before = get(ctx.addr(), &format!("/v1/db/blobs/{}", hash));
    assert_eq!(blob_before.status, 200, "blob should exist before delete");
    assert_eq!(blob_before.body, payload);

    // Delete the file metadata.
    let del_resp = delete(ctx.addr(), &format!("/v1/files/{}", file_id));
    assert_eq!(del_resp.status, 200, "body: {}", del_resp.body);

    // File metadata is gone.
    let meta_resp = get(ctx.addr(), &format!("/v1/files/{}", file_id));
    assert_eq!(meta_resp.status, 404, "metadata should be 404 after delete");

    // Blob is still accessible (CAS retention).
    let blob_after = get(ctx.addr(), &format!("/v1/db/blobs/{}", hash));
    assert_eq!(
        blob_after.status, 200,
        "CAS blob should survive file deletion, body: {}",
        blob_after.body
    );
    assert_eq!(
        blob_after.body, payload,
        "blob content should match original upload"
    );
}

// ---------------------------------------------------------------------------
// Range: inverted start > end
// ---------------------------------------------------------------------------

/// Inverted byte range (start > end) returns 416 Range Not Satisfiable.
#[test]
fn content_range_inverted_start_end_returns_416() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    let payload = "0123456789";
    let file_id = upload_text_file(&ctx, "range-invert.txt", "text/plain", payload);
    let path = format!("/v1/files/{}/content", file_id);

    // bytes=5-2 is inverted (start=5 > end=2).
    let resp = get_with_headers(&ctx, &path, &[("Range", "bytes=5-2")]);
    assert_eq!(resp.status, 416, "body: {}", resp.body);
    assert_eq!(
        resp.header("content-range"),
        Some("bytes */10"),
        "should include Content-Range with total size"
    );
}
