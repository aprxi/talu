//! Tests for the file reference resolution path.
//!
//! Verifies the contract between file upload and structured input resolution:
//! when the UI uploads a file and sends its `file_id` in structured input,
//! the server must be able to resolve that ID to a `file://` path pointing
//! at the actual blob on disk.
//!
//! These tests exercise the storage invariants that `resolve_file_references`
//! depends on, without requiring a model or actual inference.

use super::files_config;
use crate::server::common::{get, send_request, ServerTestContext};
use tempfile::TempDir;

/// Upload a file and return (file_id, blob_ref).
fn upload_file(
    ctx: &ServerTestContext,
    filename: &str,
    mime: &str,
    payload: &str,
) -> (String, String) {
    let boundary = "----talu-resolve-upload";
    let body = format!(
        "--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: {mime}\r\n\r\n{payload}\r\n--{b}--\r\n",
        b = boundary,
    );
    let content_type = format!("multipart/form-data; boundary={}", boundary);
    let headers = [("Content-Type", content_type.as_str())];

    let resp = send_request(ctx.addr(), "POST", "/v1/files", &headers, Some(&body));
    assert_eq!(resp.status, 200, "upload failed: {}", resp.body);
    let file_id = resp.json()["id"].as_str().expect("file id").to_string();

    let doc_resp = get(ctx.addr(), &format!("/v1/documents/{}", file_id));
    assert_eq!(doc_resp.status, 200, "doc lookup failed: {}", doc_resp.body);
    let blob_ref = doc_resp.json()["content"]["blob_ref"]
        .as_str()
        .expect("blob_ref")
        .to_string();

    (file_id, blob_ref)
}

/// Compute the blob file path from a sha256 blob_ref (mirrors blob_ref_to_file_url in handlers.rs).
fn blob_path_from_ref(bucket: &std::path::Path, blob_ref: &str) -> std::path::PathBuf {
    let hex = blob_ref
        .strip_prefix("sha256:")
        .unwrap_or_else(|| panic!("expected sha256: ref, got: {blob_ref}"));
    assert_eq!(hex.len(), 64, "sha256 digest should be 64 hex chars");
    bucket.join("blobs").join(&hex[..2]).join(hex)
}

// =========================================================================
// Core invariant: uploaded blobs are resolvable to file:// paths
// =========================================================================

/// Mirrors the UI flow: upload an image → the blob must exist at the path
/// that resolve_file_references would compute as a file:// URL.
#[test]
fn upload_blob_is_resolvable_to_file_url() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    let (file_id, blob_ref) = upload_file(&ctx, "photo.jpg", "image/jpeg", "fake-jpeg-bytes");

    // file_id is what the UI puts into structured input (input_image.image_url).
    assert!(
        file_id.starts_with("file_"),
        "UI sends file_ IDs: {file_id}"
    );

    // blob_ref must be sha256:<64hex> — the only format blob_ref_to_file_url handles.
    assert!(
        blob_ref.starts_with("sha256:"),
        "expected sha256: ref, got: {blob_ref}"
    );

    // The blob file must exist at {bucket}/blobs/{hex[0..2]}/{hex}.
    // This is the path resolve_file_references turns into file://{path}.
    let blob_path = blob_path_from_ref(temp.path(), &blob_ref);
    assert!(
        blob_path.exists(),
        "blob file must exist at {} for file:// URL to work",
        blob_path.display()
    );

    // Content on disk matches what was uploaded.
    let on_disk = std::fs::read(&blob_path).expect("read blob");
    assert_eq!(on_disk, b"fake-jpeg-bytes");
}

/// Same invariant for non-image files (UI sends input_file with file_url).
#[test]
fn non_image_upload_blob_is_resolvable_to_file_url() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    let (file_id, blob_ref) = upload_file(&ctx, "notes.txt", "text/plain", "some text content");

    assert!(file_id.starts_with("file_"));
    assert!(blob_ref.starts_with("sha256:"));

    let blob_path = blob_path_from_ref(temp.path(), &blob_ref);
    assert!(blob_path.exists());

    let on_disk = std::fs::read(&blob_path).expect("read blob");
    assert_eq!(on_disk, b"some text content");
}

/// When the user re-attaches the same image, the upload is idempotent:
/// same content = same file_id, same blob on disk.
#[test]
fn duplicate_uploads_resolve_to_same_blob_path() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    let (id_a, ref_a) = upload_file(&ctx, "a.jpg", "image/jpeg", "same-image-data");
    let (id_b, ref_b) = upload_file(&ctx, "b.jpg", "image/jpeg", "same-image-data");

    // Same content produces the same file_id (content-addressed).
    assert_eq!(id_a, id_b, "same content should produce same file_id");

    // And the same blob.
    assert_eq!(ref_a, ref_b, "same content should produce same blob_ref");

    // Resolves to a single blob path.
    let path_a = blob_path_from_ref(temp.path(), &ref_a);
    assert!(path_a.exists());
}
