//! `/v1/files` upload endpoint tests.

use super::{files_config, no_bucket_config};
use crate::server::common::{delete, get, send_request, ServerTestContext};
use tempfile::TempDir;

#[test]
fn upload_returns_503_without_bucket() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let boundary = "----talu-upload-test";
    let body = format!(
        "--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"sample.txt\"\r\nContent-Type: text/plain\r\n\r\nhello\r\n--{b}--\r\n",
        b = boundary
    );
    let content_type = format!("multipart/form-data; boundary={}", boundary);
    let headers = [("Content-Type", content_type.as_str())];
    let resp = send_request(ctx.addr(), "POST", "/v1/files", &headers, Some(&body));
    assert_eq!(resp.status, 503, "body: {}", resp.body);
}

#[test]
fn upload_multipart_creates_file_document() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    let boundary = "----talu-upload-ok";
    let file_payload = "hello file upload";
    let body = format!(
        "--{b}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nassistants\r\n--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"note.txt\"\r\nContent-Type: text/plain\r\n\r\n{payload}\r\n--{b}--\r\n",
        b = boundary,
        payload = file_payload
    );
    let content_type = format!("multipart/form-data; boundary={}", boundary);
    let headers = [("Content-Type", content_type.as_str())];

    let upload_resp = send_request(ctx.addr(), "POST", "/v1/files", &headers, Some(&body));
    assert_eq!(upload_resp.status, 200, "body: {}", upload_resp.body);
    let upload_json = upload_resp.json();
    let file_id = upload_json["id"].as_str().expect("id").to_string();

    assert_eq!(upload_json["object"], "file");
    assert_eq!(upload_json["filename"], "note.txt");
    assert_eq!(upload_json["purpose"], "assistants");
    assert_eq!(upload_json["status"], "processed");
    assert!(upload_json["status_details"].is_null());
    assert_eq!(
        upload_json["bytes"].as_u64().expect("bytes"),
        file_payload.len() as u64
    );
    assert!(
        file_id.starts_with("file_"),
        "expected file_ prefix, got {}",
        file_id
    );

    // Verify metadata document was created and includes blob_ref.
    let doc_resp = get(ctx.addr(), &format!("/v1/documents/{}", file_id));
    assert_eq!(doc_resp.status, 200, "body: {}", doc_resp.body);
    let doc_json = doc_resp.json();
    assert_eq!(doc_json["type"], "file");
    assert_eq!(doc_json["title"], "note.txt");
    assert!(doc_json["content"]["blob_ref"].is_string());
    assert_eq!(
        doc_json["content"]["size"].as_u64().expect("size"),
        file_payload.len() as u64
    );
}

#[test]
fn upload_requires_file_part() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    let boundary = "----talu-upload-missing-file";
    let body = format!(
        "--{b}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nassistants\r\n--{b}--\r\n",
        b = boundary
    );
    let content_type = format!("multipart/form-data; boundary={}", boundary);
    let headers = [("Content-Type", content_type.as_str())];

    let resp = send_request(ctx.addr(), "POST", "/v1/files", &headers, Some(&body));
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    let json = resp.json();
    assert_eq!(json["error"]["code"], "invalid_multipart");
}

#[test]
fn upload_get_list_content_and_delete_lifecycle() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    let boundary = "----talu-upload-lifecycle";
    let file_payload = "hello lifecycle content";
    let body = format!(
        "--{b}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nassistants\r\n--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"life.txt\"\r\nContent-Type: text/plain\r\n\r\n{payload}\r\n--{b}--\r\n",
        b = boundary,
        payload = file_payload
    );
    let content_type = format!("multipart/form-data; boundary={}", boundary);
    let headers = [("Content-Type", content_type.as_str())];

    let upload_resp = send_request(ctx.addr(), "POST", "/v1/files", &headers, Some(&body));
    assert_eq!(upload_resp.status, 200, "body: {}", upload_resp.body);
    let upload_json = upload_resp.json();
    let file_id = upload_json["id"].as_str().expect("id").to_string();

    let get_resp = get(ctx.addr(), &format!("/v1/files/{}", file_id));
    assert_eq!(get_resp.status, 200, "body: {}", get_resp.body);
    let get_json = get_resp.json();
    assert_eq!(get_json["id"], file_id);
    assert_eq!(get_json["object"], "file");
    assert_eq!(get_json["filename"], "life.txt");
    assert_eq!(get_json["purpose"], "assistants");
    assert_eq!(get_json["status"], "processed");
    assert!(get_json["status_details"].is_null());
    assert_eq!(get_json["bytes"].as_u64(), Some(file_payload.len() as u64));

    let list_resp = get(ctx.addr(), "/v1/files");
    assert_eq!(list_resp.status, 200, "body: {}", list_resp.body);
    let list_json = list_resp.json();
    assert_eq!(list_json["object"], "list");
    let data = list_json["data"].as_array().expect("data array");
    assert!(
        data.iter().any(|entry| entry["id"] == file_id),
        "list should contain uploaded file id"
    );

    let content_resp = get(ctx.addr(), &format!("/v1/files/{}/content", file_id));
    assert_eq!(content_resp.status, 200, "body: {}", content_resp.body);
    let ct = content_resp.header("content-type").unwrap_or("");
    assert!(
        ct.contains("text/plain"),
        "content-type should be text/plain, got {ct}",
    );
    assert_eq!(content_resp.body, file_payload);

    let delete_resp = delete(ctx.addr(), &format!("/v1/files/{}", file_id));
    assert_eq!(delete_resp.status, 200, "body: {}", delete_resp.body);
    let delete_json = delete_resp.json();
    assert_eq!(delete_json["id"], file_id);
    assert_eq!(delete_json["object"], "file");
    assert_eq!(delete_json["deleted"], true);

    let get_after_delete = get(ctx.addr(), &format!("/v1/files/{}", file_id));
    assert_eq!(
        get_after_delete.status, 404,
        "body: {}",
        get_after_delete.body
    );
}

#[test]
fn upload_rejects_conflicting_filename_sources() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    let boundary = "----talu-upload-conflict";
    let body = format!(
        "--{b}\r\nContent-Disposition: form-data; name=\"filename\"\r\n\r\nfrom-field.txt\r\n--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"from-file.txt\"\r\nContent-Type: text/plain\r\n\r\nhello\r\n--{b}--\r\n",
        b = boundary,
    );
    let content_type = format!("multipart/form-data; boundary={}", boundary);
    let headers = [("Content-Type", content_type.as_str())];

    let resp = send_request(ctx.addr(), "POST", "/v1/files", &headers, Some(&body));
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    let json = resp.json();
    assert_eq!(json["error"]["code"], "invalid_multipart");
    let msg = json["error"]["message"].as_str().unwrap_or_default();
    assert!(msg.contains("conflicting filename values"));
}
