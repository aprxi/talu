//! File upload endpoint backed by blob storage + document metadata.
//!
//! Implements `POST /v1/files` (`/files`) using:
//! - `talu::blobs::BlobsHandle` for raw bytes
//! - `talu::documents::DocumentsHandle` for metadata (`doc_type = "file"`)

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use http_body_util::{BodyExt, Full, StreamBody};
use hyper::body::{Frame, Incoming};
use hyper::{Request, Response, StatusCode};
use serde::Deserialize;
use serde::Serialize;
use talu::blobs::{BlobError, BlobsHandle};
use talu::documents::{DocumentError, DocumentRecord, DocumentsHandle};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;

use crate::server::auth_gateway::AuthContext;
use crate::server::state::AppState;

type BoxBody = http_body_util::combinators::BoxBody<Bytes, std::convert::Infallible>;

#[derive(Debug, Serialize)]
struct FileObjectResponse {
    id: String,
    object: String,
    bytes: u64,
    created_at: u64,
    filename: String,
    purpose: String,
    status: String,
    status_details: Option<String>,
}

#[derive(Debug)]
struct ParsedMultipartUpload {
    filename: String,
    mime_type: Option<String>,
    purpose: Option<String>,
    bytes: u64,
    blob_ref: String,
}

#[derive(Debug)]
enum UploadError {
    InvalidMultipart(String),
    PayloadTooLarge(String),
    Blob(BlobError),
}

#[derive(Debug, Serialize)]
struct FileDeleteResponse {
    id: String,
    object: String,
    deleted: bool,
}

#[derive(Debug, Serialize)]
struct FileListResponse {
    object: String,
    data: Vec<FileObjectResponse>,
    has_more: bool,
}

#[derive(Debug, Deserialize)]
struct FileDocumentContent {
    #[serde(default)]
    blob_ref: Option<String>,
    #[serde(default)]
    original_name: Option<String>,
    #[serde(default)]
    mime_type: Option<String>,
    #[serde(default)]
    size: Option<u64>,
    #[serde(default)]
    purpose: Option<String>,
}

#[derive(Debug)]
struct FileDescriptor {
    id: String,
    filename: String,
    purpose: String,
    bytes: u64,
    created_at: u64,
    blob_ref: String,
    mime_type: Option<String>,
}

/// POST /v1/files - Upload a file as blob + metadata document.
pub async fn handle_upload(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let bucket = match state.bucket_path.as_ref() {
        Some(p) => p,
        None => {
            return json_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "no_storage",
                "Storage not configured",
            );
        }
    };

    let content_type = match req
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
    {
        Some(v) => v,
        None => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_content_type",
                "Missing Content-Type header",
            );
        }
    };

    let boundary = match extract_boundary(content_type) {
        Some(v) => v,
        None => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_content_type",
                "Expected multipart/form-data with boundary",
            );
        }
    };
    let max_upload_bytes = state.max_file_upload_bytes;
    if let Some(content_length) = req
        .headers()
        .get(hyper::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
    {
        if content_length > max_upload_bytes {
            return json_error(
                StatusCode::PAYLOAD_TOO_LARGE,
                "payload_too_large",
                &format!(
                    "Upload exceeds configured limit ({} > {} bytes)",
                    content_length, max_upload_bytes
                ),
            );
        }
    }

    let storage_path = match auth.as_ref() {
        Some(ctx) => bucket.join(&ctx.storage_prefix),
        None => bucket.to_path_buf(),
    };

    let blobs = match BlobsHandle::open(&storage_path) {
        Ok(h) => h,
        Err(e) => return blob_error_response(e),
    };

    let upload = match stream_multipart_upload(req, &boundary, &blobs, max_upload_bytes).await {
        Ok(u) => u,
        Err(UploadError::InvalidMultipart(e)) => {
            return json_error(StatusCode::BAD_REQUEST, "invalid_multipart", &e);
        }
        Err(UploadError::PayloadTooLarge(e)) => {
            return json_error(StatusCode::PAYLOAD_TOO_LARGE, "payload_too_large", &e);
        }
        Err(UploadError::Blob(e)) => return blob_error_response(e),
    };

    let docs = match DocumentsHandle::open(&storage_path) {
        Ok(h) => h,
        Err(e) => return document_error_response(e),
    };

    let file_id = format!("file_{}", uuid::Uuid::new_v4().simple());
    let purpose = upload.purpose.unwrap_or_else(|| "assistants".to_string());
    let created_at = now_unix_seconds();
    let filename = sanitize_filename(&upload.filename).unwrap_or_else(|| "upload.bin".to_string());
    let metadata = serde_json::json!({
        "blob_ref": upload.blob_ref,
        "original_name": filename,
        "mime_type": upload.mime_type,
        "size": upload.bytes,
        "purpose": purpose,
    });

    if let Err(e) = docs.create(
        &file_id,
        "file",
        metadata["original_name"].as_str().unwrap_or_default(),
        &metadata.to_string(),
        None,
        None,
        Some("active"),
        auth.as_ref().and_then(|a| a.group_id.as_deref()),
        auth.as_ref().and_then(|a| a.user_id.as_deref()),
    ) {
        return document_error_response(e);
    }

    let response = FileObjectResponse {
        id: file_id,
        object: "file".to_string(),
        bytes: upload.bytes,
        created_at,
        filename: metadata["original_name"]
            .as_str()
            .unwrap_or_default()
            .to_string(),
        purpose: metadata["purpose"]
            .as_str()
            .unwrap_or("assistants")
            .to_string(),
        status: "processed".to_string(),
        status_details: None,
    };

    json_response(StatusCode::OK, &response)
}

/// GET /v1/files - List file metadata entries.
pub async fn handle_list(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let bucket = match state.bucket_path.as_ref() {
        Some(p) => p,
        None => {
            return json_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "no_storage",
                "Storage not configured",
            );
        }
    };

    let limit = parse_query_param(req.uri().query().unwrap_or(""), "limit")
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(100);

    let storage_path = match auth.as_ref() {
        Some(ctx) => bucket.join(&ctx.storage_prefix),
        None => bucket.to_path_buf(),
    };

    let docs = match DocumentsHandle::open(&storage_path) {
        Ok(h) => h,
        Err(e) => return document_error_response(e),
    };

    let rows = match docs.list(Some("file"), None, None, Some("active"), limit) {
        Ok(v) => v,
        Err(e) => return document_error_response(e),
    };

    let has_more = rows.len() >= limit as usize;
    let mut data = Vec::with_capacity(rows.len());
    for row in rows {
        let doc = match docs.get(&row.doc_id) {
            Ok(Some(doc)) => doc,
            Ok(None) => continue,
            Err(e) => return document_error_response(e),
        };
        let descriptor = match descriptor_from_doc(doc) {
            Ok(d) => d,
            Err(resp) => return resp,
        };
        data.push(to_file_object_response(descriptor));
    }

    json_response(
        StatusCode::OK,
        &FileListResponse {
            object: "list".to_string(),
            data,
            has_more,
        },
    )
}

/// GET /v1/files/:id - Return file metadata.
pub async fn handle_get(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let file_id = match extract_file_id(req.uri().path()) {
        Some(id) => id,
        None => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_argument",
                "Invalid file id path",
            );
        }
    };

    let (doc, _) = match load_file_doc(state, auth, &file_id) {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let descriptor = match descriptor_from_doc(doc) {
        Ok(d) => d,
        Err(resp) => return resp,
    };

    let response = to_file_object_response(descriptor);
    json_response(StatusCode::OK, &response)
}

/// GET /v1/files/:id/content - Return raw file bytes.
pub async fn handle_get_content(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let file_id = match extract_file_id(req.uri().path()) {
        Some(id) => id,
        None => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_argument",
                "Invalid file id path",
            );
        }
    };

    let (doc, storage_path) = match load_file_doc(state, auth, &file_id) {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let descriptor = match descriptor_from_doc(doc) {
        Ok(d) => d,
        Err(resp) => return resp,
    };

    let blobs = match BlobsHandle::open(&storage_path) {
        Ok(h) => h,
        Err(e) => return blob_error_response(e),
    };
    let mut blob_stream = match blobs.open_stream(&descriptor.blob_ref) {
        Ok(s) => s,
        Err(e) => return blob_error_response(e),
    };

    let (tx, rx) = mpsc::unbounded_channel::<Bytes>();
    tokio::task::spawn_blocking(move || {
        const CHUNK_SIZE: usize = 64 * 1024;
        let mut buf = [0u8; CHUNK_SIZE];

        loop {
            match blob_stream.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    if tx.send(Bytes::copy_from_slice(&buf[..n])).is_err() {
                        break;
                    }
                }
                Err(err) => {
                    log::warn!("blob stream read failed for file content: {err}");
                    break;
                }
            }
        }
    });

    let stream = UnboundedReceiverStream::new(rx)
        .map(|chunk| Ok::<_, std::convert::Infallible>(Frame::data(chunk)));
    let body = StreamBody::new(stream).boxed();

    Response::builder()
        .status(StatusCode::OK)
        .header(
            "content-type",
            descriptor
                .mime_type
                .as_deref()
                .unwrap_or("application/octet-stream"),
        )
        .header(
            "content-disposition",
            format!("attachment; filename=\"{}\"", descriptor.filename),
        )
        .header("content-length", descriptor.bytes.to_string())
        .body(body)
        .unwrap()
}

/// DELETE /v1/files/:id - Delete file metadata (blob retained for CAS/GC lifecycle).
pub async fn handle_delete(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let file_id = match extract_file_id(req.uri().path()) {
        Some(id) => id,
        None => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_argument",
                "Invalid file id path",
            );
        }
    };

    let (doc, storage_path) = match load_file_doc(state, auth, &file_id) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    if doc.doc_type != "file" {
        return json_error(
            StatusCode::NOT_FOUND,
            "not_found",
            &format!("File not found: {}", file_id),
        );
    }

    let docs = match DocumentsHandle::open(&storage_path) {
        Ok(h) => h,
        Err(e) => return document_error_response(e),
    };
    if let Err(e) = docs.delete(&file_id) {
        return document_error_response(e);
    }

    json_response(
        StatusCode::OK,
        &FileDeleteResponse {
            id: file_id,
            object: "file".to_string(),
            deleted: true,
        },
    )
}

fn load_file_doc(
    state: Arc<AppState>,
    auth: Option<AuthContext>,
    file_id: &str,
) -> Result<(DocumentRecord, std::path::PathBuf), Response<BoxBody>> {
    let bucket = match state.bucket_path.as_ref() {
        Some(p) => p,
        None => {
            return Err(json_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "no_storage",
                "Storage not configured",
            ));
        }
    };

    let storage_path = match auth.as_ref() {
        Some(ctx) => bucket.join(&ctx.storage_prefix),
        None => bucket.to_path_buf(),
    };

    let docs = match DocumentsHandle::open(&storage_path) {
        Ok(h) => h,
        Err(e) => return Err(document_error_response(e)),
    };

    let doc = match docs.get(file_id) {
        Ok(Some(d)) => d,
        Ok(None) => {
            return Err(json_error(
                StatusCode::NOT_FOUND,
                "not_found",
                &format!("File not found: {}", file_id),
            ));
        }
        Err(e) => return Err(document_error_response(e)),
    };

    if doc.doc_type != "file" {
        return Err(json_error(
            StatusCode::NOT_FOUND,
            "not_found",
            &format!("File not found: {}", file_id),
        ));
    }

    Ok((doc, storage_path))
}

fn descriptor_from_doc(doc: DocumentRecord) -> Result<FileDescriptor, Response<BoxBody>> {
    let parsed: FileDocumentContent =
        serde_json::from_str(&doc.doc_json).unwrap_or(FileDocumentContent {
            blob_ref: None,
            original_name: None,
            mime_type: None,
            size: None,
            purpose: None,
        });

    let blob_ref = match parsed.blob_ref {
        Some(v) if !v.is_empty() => v,
        _ => {
            return Err(json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "invalid_file_metadata",
                "File metadata missing blob_ref",
            ));
        }
    };

    Ok(FileDescriptor {
        id: doc.doc_id,
        filename: parsed
            .original_name
            .filter(|s| !s.trim().is_empty())
            .unwrap_or(doc.title),
        purpose: parsed
            .purpose
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| "assistants".to_string()),
        bytes: parsed.size.unwrap_or(0),
        created_at: (doc.created_at_ms.max(0) as u64) / 1000,
        blob_ref,
        mime_type: parsed.mime_type,
    })
}

fn to_file_object_response(descriptor: FileDescriptor) -> FileObjectResponse {
    FileObjectResponse {
        id: descriptor.id,
        object: "file".to_string(),
        bytes: descriptor.bytes,
        created_at: descriptor.created_at,
        filename: descriptor.filename,
        purpose: descriptor.purpose,
        status: "processed".to_string(),
        status_details: None,
    }
}

fn parse_query_param<'a>(query: &'a str, name: &str) -> Option<&'a str> {
    for pair in query.split('&') {
        let mut it = pair.splitn(2, '=');
        let key = it.next()?;
        let value = it.next().unwrap_or("");
        if key == name {
            return Some(value);
        }
    }
    None
}

fn extract_file_id(path: &str) -> Option<String> {
    let parts: Vec<&str> = path.trim_start_matches('/').split('/').collect();
    match parts.as_slice() {
        ["v1", "files", id, ..] if !id.is_empty() => Some((*id).to_string()),
        ["files", id, ..] if !id.is_empty() => Some((*id).to_string()),
        _ => None,
    }
}

fn extract_boundary(content_type: &str) -> Option<String> {
    multer::parse_boundary(content_type).ok()
}

async fn stream_multipart_upload(
    req: Request<Incoming>,
    boundary: &str,
    blobs: &BlobsHandle,
    max_upload_bytes: u64,
) -> Result<ParsedMultipartUpload, UploadError> {
    let body_stream = req.into_body().into_data_stream();
    let mut multipart = multer::Multipart::new(body_stream, boundary);

    let mut blob_ref: Option<String> = None;
    let mut file_name: Option<String> = None;
    let mut file_mime: Option<String> = None;
    let mut field_filename: Option<String> = None;
    let mut purpose: Option<String> = None;
    let mut file_size: u64 = 0;

    while let Some(mut field) = multipart.next_field().await.map_err(|e| {
        UploadError::InvalidMultipart(format!("Failed to parse multipart field: {}", e))
    })? {
        let field_name = field.name().map(|s| s.to_string());
        match field_name.as_deref() {
            Some("file") => {
                if blob_ref.is_some() {
                    return Err(UploadError::InvalidMultipart(
                        "multiple file parts are not supported".to_string(),
                    ));
                }
                let mut writer = blobs.open_write_stream().map_err(UploadError::Blob)?;
                file_name = field.file_name().map(str::to_string);
                file_mime = field.content_type().map(|m| m.to_string());
                while let Some(chunk) = field.chunk().await.map_err(|e| {
                    UploadError::InvalidMultipart(format!("Failed reading file chunk: {}", e))
                })? {
                    file_size = file_size.checked_add(chunk.len() as u64).ok_or_else(|| {
                        UploadError::PayloadTooLarge(
                            "Upload size overflowed supported range".to_string(),
                        )
                    })?;
                    if file_size > max_upload_bytes {
                        return Err(UploadError::PayloadTooLarge(format!(
                            "Upload exceeds configured limit ({} > {} bytes)",
                            file_size, max_upload_bytes
                        )));
                    }
                    writer.write(&chunk).map_err(UploadError::Blob)?;
                }
                blob_ref = Some(writer.finish().map_err(UploadError::Blob)?);
            }
            Some("purpose") => {
                let value = field.text().await.map_err(|e| {
                    UploadError::InvalidMultipart(format!("Invalid purpose field: {}", e))
                })?;
                let trimmed = value.trim();
                if !trimmed.is_empty() {
                    purpose = Some(trimmed.to_string());
                }
            }
            Some("filename") => {
                let value = field.text().await.map_err(|e| {
                    UploadError::InvalidMultipart(format!("Invalid filename field: {}", e))
                })?;
                let trimmed = value.trim();
                if !trimmed.is_empty() {
                    field_filename = Some(trimmed.to_string());
                }
            }
            _ => {
                while field
                    .chunk()
                    .await
                    .map_err(|e| {
                        UploadError::InvalidMultipart(format!(
                            "Failed draining multipart field: {}",
                            e
                        ))
                    })?
                    .is_some()
                {}
            }
        }
    }

    let blob_ref = blob_ref.ok_or_else(|| {
        UploadError::InvalidMultipart("missing multipart file field 'file'".to_string())
    })?;
    let filename =
        resolve_filename(file_name, field_filename).map_err(UploadError::InvalidMultipart)?;

    Ok(ParsedMultipartUpload {
        filename,
        mime_type: file_mime,
        purpose,
        bytes: file_size,
        blob_ref,
    })
}

fn resolve_filename(
    file_part_filename: Option<String>,
    filename_field: Option<String>,
) -> Result<String, String> {
    let file_part = file_part_filename.and_then(|name| sanitize_filename(&name));
    let field = filename_field.and_then(|name| sanitize_filename(&name));

    if let (Some(a), Some(b)) = (&file_part, &field) {
        if a != b {
            return Err(
                "conflicting filename values between multipart file part and filename field"
                    .to_string(),
            );
        }
    }

    Ok(file_part
        .or(field)
        .unwrap_or_else(|| "upload.bin".to_string()))
}

fn sanitize_filename(name: &str) -> Option<String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return None;
    }

    let basename = trimmed
        .rsplit(|ch| ch == '/' || ch == '\\')
        .next()
        .unwrap_or("")
        .trim();
    if basename.is_empty() || basename == "." || basename == ".." {
        return None;
    }

    Some(basename.to_string())
}

fn now_unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn json_response<T: Serialize>(status: StatusCode, data: &T) -> Response<BoxBody> {
    let body = serde_json::to_vec(data).unwrap_or_default();
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body)).boxed())
        .unwrap()
}

fn json_error(status: StatusCode, code: &str, message: &str) -> Response<BoxBody> {
    let body = serde_json::json!({
        "error": {
            "code": code,
            "message": message
        }
    });
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body.to_string())).boxed())
        .unwrap()
}

fn document_error_response(err: DocumentError) -> Response<BoxBody> {
    match err {
        DocumentError::DocumentNotFound(id) => json_error(
            StatusCode::NOT_FOUND,
            "not_found",
            &format!("Document not found: {}", id),
        ),
        DocumentError::InvalidArgument(msg) => {
            json_error(StatusCode::BAD_REQUEST, "invalid_argument", &msg)
        }
        DocumentError::StorageNotFound(p) => json_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "storage_not_found",
            &format!("Storage not found: {}", p.display()),
        ),
        DocumentError::StorageError(msg) => {
            json_error(StatusCode::INTERNAL_SERVER_ERROR, "storage_error", &msg)
        }
    }
}

fn blob_error_response(err: BlobError) -> Response<BoxBody> {
    match err {
        BlobError::InvalidArgument(msg) => {
            json_error(StatusCode::BAD_REQUEST, "invalid_argument", &msg)
        }
        BlobError::NotFound(msg) => json_error(StatusCode::NOT_FOUND, "not_found", &msg),
        BlobError::PermissionDenied(msg) => {
            json_error(StatusCode::FORBIDDEN, "permission_denied", &msg)
        }
        BlobError::ResourceExhausted(msg) => {
            json_error(StatusCode::INSUFFICIENT_STORAGE, "resource_exhausted", &msg)
        }
        BlobError::StorageError(msg) => {
            json_error(StatusCode::INTERNAL_SERVER_ERROR, "storage_error", &msg)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{extract_boundary, extract_file_id, resolve_filename};
    use talu::blobs::BlobError;

    #[test]
    fn extract_boundary_parses_content_type() {
        let ct = "multipart/form-data; boundary=----abc123";
        assert_eq!(extract_boundary(ct).as_deref(), Some("----abc123"));
    }

    #[test]
    fn extract_boundary_rejects_non_multipart() {
        let ct = "application/json";
        assert!(extract_boundary(ct).is_none());
    }

    #[test]
    fn resolve_filename_rejects_conflict() {
        let err = resolve_filename(Some("a.txt".to_string()), Some("b.txt".to_string()))
            .expect_err("expected conflict");
        assert!(err.contains("conflicting filename values"));
    }

    #[test]
    fn resolve_filename_accepts_equal_values() {
        let resolved = resolve_filename(Some("same.txt".to_string()), Some("same.txt".to_string()))
            .expect("expected resolved filename");
        assert_eq!(resolved, "same.txt");
    }

    #[test]
    fn resolve_filename_sanitizes_path_traversal() {
        let resolved = resolve_filename(Some("../../etc/passwd".to_string()), None)
            .expect("expected sanitized filename");
        assert_eq!(resolved, "passwd");
    }

    #[test]
    fn resolve_filename_uses_default_for_empty_name() {
        let resolved =
            resolve_filename(Some("   ".to_string()), None).expect("expected default filename");
        assert_eq!(resolved, "upload.bin");
    }

    #[test]
    fn extract_file_id_supports_content_paths() {
        assert_eq!(
            extract_file_id("/v1/files/file_abc123/content").as_deref(),
            Some("file_abc123")
        );
        assert_eq!(
            extract_file_id("/files/file_abc123/content").as_deref(),
            Some("file_abc123")
        );
    }

    #[test]
    fn blob_error_response_maps_resource_exhausted_to_507() {
        let resp =
            super::blob_error_response(BlobError::ResourceExhausted("disk full".to_string()));
        assert_eq!(resp.status(), super::StatusCode::INSUFFICIENT_STORAGE);
    }
}
