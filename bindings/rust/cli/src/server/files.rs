//! File upload endpoint backed by blob storage + document metadata.
//!
//! Implements `POST /v1/files` (`/files`) using:
//! - `talu::blobs::BlobsHandle` for raw bytes
//! - `talu::documents::DocumentsHandle` for metadata (`doc_type = "file"`)
//!
//! On upload the handler runs `talu::file::inspect_bytes` to detect MIME type,
//! file kind, and image dimensions.  Inspection is fail-safe: timeouts, panics,
//! and errors are logged and the upload succeeds with metadata fields absent.

use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use http_body_util::{BodyExt, Full, StreamBody};
use hyper::body::{Frame, Incoming};
use hyper::{Request, Response, StatusCode};
use serde::Deserialize;
use serde::Serialize;
use utoipa::ToSchema;
use talu::blobs::{BlobError, BlobsHandle};
use talu::documents::{DocumentError, DocumentRecord, DocumentsHandle};
use talu::file;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;

use crate::server::auth_gateway::AuthContext;
use crate::server::state::AppState;

type BoxBody = http_body_util::combinators::BoxBody<Bytes, std::convert::Infallible>;

#[derive(Debug, Clone, Serialize, ToSchema)]
pub(crate) struct FileObjectResponse {
    id: String,
    object: String,
    bytes: u64,
    created_at: u64,
    filename: String,
    purpose: String,
    status: String,
    status_details: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    kind: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image: Option<FileImageMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    marker: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub(crate) struct FileImageMetadata {
    format: String,
    width: u32,
    height: u32,
    exif_orientation: u8,
    aspect_ratio: f64,
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

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct FileDeleteResponse {
    id: String,
    object: String,
    deleted: bool,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct FileListResponse {
    object: String,
    data: Vec<FileObjectResponse>,
    has_more: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    cursor: Option<String>,
    total: usize,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct FileBatchRequest {
    action: String,
    #[serde(default)]
    ids: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize)]
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    image: Option<FileImageMetadata>,
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
    kind: Option<String>,
    image: Option<FileImageMetadata>,
    marker: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContentRangeRequest {
    Full,
    Partial { start: u64, end: u64 },
}

/// Documentation-only schema for multipart file uploads.
#[derive(ToSchema)]
#[allow(dead_code)]
pub(crate) struct FileUploadForm {
    /// The file binary data.
    #[schema(format = Binary)]
    file: Vec<u8>,
    /// Upload purpose (defaults to "assistants").
    purpose: Option<String>,
    /// Override filename (defaults to multipart filename).
    filename: Option<String>,
}

/// POST /v1/files - Upload a file as blob + metadata document.
#[utoipa::path(post, path = "/v1/files", tag = "Files",
    request_body(content_type = "multipart/form-data", content = inline(FileUploadForm)),
    responses((status = 200, body = FileObjectResponse)))]
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

    log::trace!(target: "server::files", "BlobsHandle::open({:?})", storage_path);
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

    log::trace!(target: "server::files", "DocumentsHandle::open({:?})", storage_path);
    let docs = match DocumentsHandle::open(&storage_path) {
        Ok(h) => h,
        Err(e) => return document_error_response(e),
    };

    let purpose = upload.purpose.unwrap_or_else(|| "assistants".to_string());
    let filename = sanitize_filename(&upload.filename).unwrap_or_else(|| "upload.bin".to_string());

    log::debug!(target: "server::files", "upload: filename={} size={} mime={:?} blob_ref={}",
        filename, upload.bytes, upload.mime_type, upload.blob_ref);

    // Deterministic file ID from content hash — the blob_ref sha256 hex is
    // already computed during streaming, so same content = same document ID.
    // This makes uploads idempotent: docs.get() is an O(1) existence check.
    let blob_hex = upload.blob_ref.strip_prefix("sha256:").unwrap_or(&upload.blob_ref);
    let file_id = format!("file_{}", &blob_hex[..blob_hex.len().min(32)]);

    // O(1) existence check: if this content already exists, return it.
    if let Ok(Some(doc)) = docs.get(&file_id) {
        log::debug!(target: "server::files", "idempotent: {} already exists", file_id);
        if let Ok(descriptor) = descriptor_from_doc(doc) {
            return json_response(StatusCode::OK, &to_file_object_response(descriptor));
        }
    }

    let created_at = now_unix_seconds();

    // Fail-safe inspection: read blob back, inspect, store metadata.
    // Timeouts, panics, and errors are logged — upload always succeeds.
    let inspection = inspect_blob_failsafe(
        &blobs,
        &upload.blob_ref,
        upload.bytes,
        state.max_file_inspect_bytes,
    )
    .await;

    let detected_mime = inspection
        .as_ref()
        .map(|i| i.mime.clone())
        .or(upload.mime_type.clone());

    let mut doc_content = FileDocumentContent {
        blob_ref: Some(upload.blob_ref.clone()),
        original_name: Some(filename.clone()),
        mime_type: detected_mime.clone(),
        size: Some(upload.bytes),
        purpose: Some(purpose.clone()),
        kind: None,
        description: None,
        image: None,
    };

    let mut resp_kind: Option<String> = None;
    let mut resp_image: Option<FileImageMetadata> = None;

    if let Some(info) = inspection {
        let kind_str = map_file_kind(&info.kind);
        log::debug!(target: "server::files", "inspection: kind={} mime={}", kind_str, info.mime);
        doc_content.kind = Some(kind_str.to_string());
        doc_content.description = Some(info.description);

        resp_kind = Some(kind_str.to_string());

        if let Some(img) = info.image {
            let aspect = if img.height > 0 {
                img.width as f64 / img.height as f64
            } else {
                0.0
            };
            let meta = FileImageMetadata {
                format: image_format_str(img.format),
                width: img.width,
                height: img.height,
                exif_orientation: img.exif_orientation,
                aspect_ratio: aspect,
            };
            doc_content.image = Some(meta.clone());
            resp_image = Some(meta);
        }
    }

    let metadata_json = serde_json::to_string(&doc_content).unwrap_or_else(|_| "{}".to_string());

    if let Err(e) = docs.create(
        &file_id,
        "file",
        &filename,
        &metadata_json,
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
        filename,
        purpose,
        status: "processed".to_string(),
        status_details: None,
        mime_type: detected_mime,
        kind: resp_kind,
        image: resp_image,
        marker: Some("active".to_string()),
    };

    json_response(StatusCode::OK, &response)
}

/// GET /v1/files - List file metadata entries.
#[utoipa::path(get, path = "/v1/files", tag = "Files",
    params(
        ("limit" = Option<usize>, Query, description = "Max items to return (default 100, max 1000)"),
        ("offset" = Option<usize>, Query, description = "Number of items to skip"),
        ("cursor" = Option<String>, Query, description = "Pagination cursor (overrides offset)"),
        ("marker" = Option<String>, Query, description = "Filter by marker (default \"active\")"),
        ("sort" = Option<String>, Query, description = "Sort field: date | name (default \"date\")"),
        ("order" = Option<String>, Query, description = "Sort order: asc | desc (default \"desc\")"),
        ("search" = Option<String>, Query, description = "Filter by filename substring"),
    ),
    responses((status = 200, body = FileListResponse)))]
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

    let query_str = req.uri().query().unwrap_or("");
    let limit = parse_query_param(query_str, "limit")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(100)
        .clamp(1, 1000);
    let mut offset = parse_query_param(query_str, "offset")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    let cursor_param = parse_query_param(query_str, "cursor");
    let marker_filter = parse_query_param(query_str, "marker").unwrap_or("active");
    let sort = parse_query_param(query_str, "sort").unwrap_or("date");
    let order = parse_query_param(query_str, "order").unwrap_or("desc");
    let search = parse_query_param(query_str, "search");

    let storage_path = match auth.as_ref() {
        Some(ctx) => bucket.join(&ctx.storage_prefix),
        None => bucket.to_path_buf(),
    };

    let docs = match DocumentsHandle::open(&storage_path) {
        Ok(h) => h,
        Err(e) => return document_error_response(e),
    };

    let rows = match docs.list(Some("file"), None, None, Some(marker_filter), u32::MAX) {
        Ok(v) => v,
        Err(e) => return document_error_response(e),
    };

    let mut filtered: Vec<_> = rows;

    // Apply search filter on title (filename).
    if let Some(ref q) = search {
        let q_lower = q.to_lowercase();
        filtered.retain(|r| r.title.to_lowercase().contains(&q_lower));
    }

    // Sort on summaries (avoids fetching full documents).
    let desc = order == "desc";
    filtered.sort_by(|a, b| {
        let cmp = match sort {
            "name" => a.title.to_lowercase().cmp(&b.title.to_lowercase()),
            "size" => std::cmp::Ordering::Equal, // size not in summary, fall back to date
            "date" | _ => a.created_at_ms.cmp(&b.created_at_ms),
        };
        let cmp = if desc { cmp.reverse() } else { cmp };
        cmp.then_with(|| b.doc_id.cmp(&a.doc_id))
    });

    // If a cursor was provided, find the cursor item in the sorted list
    // and start from the next item (cursor takes precedence over offset).
    if let Some(cursor_str) = cursor_param {
        if let Some((cursor_ts, cursor_id)) = decode_file_cursor(cursor_str) {
            if let Some(pos) = filtered
                .iter()
                .position(|r| r.created_at_ms == cursor_ts && r.doc_id == cursor_id)
            {
                offset = pos + 1;
            }
        }
    }

    let total = filtered.len();

    // Apply offset + limit to get the page of summaries.
    let start = offset.min(total);
    let end = (offset + limit).min(total);
    let page_summaries = &filtered[start..end];
    let has_more = end < total;

    // Only fetch full documents for the current page (avoids N+1 for all files).
    let mut page: Vec<FileObjectResponse> = Vec::with_capacity(page_summaries.len());
    for summary in page_summaries {
        let doc = match docs.get(&summary.doc_id) {
            Ok(Some(doc)) => doc,
            Ok(None) => continue,
            Err(e) => return document_error_response(e),
        };
        let descriptor = match descriptor_from_doc(doc) {
            Ok(d) => d,
            Err(resp) => return resp,
        };
        page.push(to_file_object_response(descriptor));
    }

    // Cursor from last summary (uses created_at_ms so decode matches the
    // lookup against summaries on the next request).
    let next_cursor = if has_more {
        page_summaries
            .last()
            .map(|s| encode_file_cursor(s.created_at_ms, &s.doc_id))
    } else {
        None
    };

    json_response(
        StatusCode::OK,
        &FileListResponse {
            object: "list".to_string(),
            data: page,
            has_more,
            cursor: next_cursor,
            total,
        },
    )
}

/// GET /v1/files/:id - Return file metadata.
#[utoipa::path(get, path = "/v1/files/{file_id}", tag = "Files",
    params(("file_id" = String, Path, description = "File ID")),
    responses((status = 200, body = FileObjectResponse)))]
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
#[utoipa::path(get, path = "/v1/files/{file_id}/content", tag = "Files",
    params(("file_id" = String, Path, description = "File ID")),
    responses((status = 200, description = "File content bytes")))]
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
    let total_size = match blob_stream.total_size() {
        Ok(size) => size,
        Err(e) => return blob_error_response(e),
    };
    let range_header = match req.headers().get("range") {
        Some(value) => match value.to_str() {
            Ok(s) => Some(s),
            Err(_) => return range_not_satisfiable(total_size),
        },
        None => None,
    };
    let requested_range = match parse_range_header(range_header, total_size) {
        Ok(r) => r,
        Err(()) => return range_not_satisfiable(total_size),
    };

    let (status, content_length, content_range_header) = match requested_range {
        ContentRangeRequest::Full => (StatusCode::OK, total_size, None),
        ContentRangeRequest::Partial { start, end } => {
            if let Err(e) = blob_stream.seek(start) {
                return blob_error_response(e);
            }
            (
                StatusCode::PARTIAL_CONTENT,
                end - start + 1,
                Some(format!("bytes {}-{}/{}", start, end, total_size)),
            )
        }
    };

    let (tx, rx) = mpsc::unbounded_channel::<Bytes>();
    tokio::task::spawn_blocking(move || {
        const CHUNK_SIZE: usize = 64 * 1024;
        let mut buf = [0u8; CHUNK_SIZE];
        let mut remaining = Some(content_length);

        loop {
            let next_read_len = match remaining {
                Some(0) => break,
                Some(rem) => CHUNK_SIZE.min(usize::try_from(rem).unwrap_or(usize::MAX)),
                None => CHUNK_SIZE,
            };

            match blob_stream.read(&mut buf[..next_read_len]) {
                Ok(0) => break,
                Ok(n) => {
                    if tx.send(Bytes::copy_from_slice(&buf[..n])).is_err() {
                        break;
                    }
                    if let Some(rem) = remaining.as_mut() {
                        *rem = rem.saturating_sub(n as u64);
                    }
                }
                Err(err) => {
                    log::warn!(target: "server::files", "blob stream read failed for file content: {err}");
                    break;
                }
            }
        }
    });

    let stream = UnboundedReceiverStream::new(rx)
        .map(|chunk| Ok::<_, std::convert::Infallible>(Frame::data(chunk)));
    let body = StreamBody::new(stream).boxed();

    let mut response_builder = Response::builder();
    response_builder = response_builder
        .status(status)
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
        .header("accept-ranges", "bytes")
        .header("content-length", content_length.to_string());
    if let Some(header_value) = content_range_header {
        response_builder = response_builder.header("content-range", header_value);
    }
    response_builder.body(body).unwrap()
}

/// GET /v1/blobs/:hash - Serve raw blob content by sha256 hex digest.
///
/// Used by the UI to display images from stored conversations where the
/// `image_url` field contains a `file://` blob path.  The UI extracts the
/// 64-char hex hash and fetches `/v1/blobs/{hash}`.
#[utoipa::path(get, path = "/v1/blobs/{blob_ref}", tag = "Files",
    params(("blob_ref" = String, Path, description = "Blob reference")),
    responses((status = 200, description = "Raw blob bytes")))]
pub async fn handle_get_blob(
    state: Arc<AppState>,
    _req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = _req.uri().path();
    let hash = path
        .strip_prefix("/v1/blobs/")
        .or_else(|| path.strip_prefix("/blobs/"))
        .unwrap_or("");

    if hash.len() != 64 || !hash.chars().all(|c| c.is_ascii_hexdigit()) {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "Expected 64-char hex sha256 digest",
        );
    }

    let storage_path = match state.bucket_path.as_ref() {
        Some(p) => p.clone(),
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "not_found",
                "No file storage configured",
            );
        }
    };

    let blob_ref = format!("sha256:{hash}");
    let blobs = match BlobsHandle::open(&storage_path) {
        Ok(h) => h,
        Err(e) => return blob_error_response(e),
    };
    let mut blob_stream = match blobs.open_stream(&blob_ref) {
        Ok(s) => s,
        Err(e) => return blob_error_response(e),
    };
    let total_size = match blob_stream.total_size() {
        Ok(size) => size,
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
                    log::warn!(target: "server::files", "blob stream read failed for blob content: {err}");
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
        .header("content-type", "application/octet-stream")
        .header("content-length", total_size.to_string())
        .header("cache-control", "public, max-age=31536000, immutable")
        .body(body)
        .unwrap()
}

/// DELETE /v1/files/:id - Delete file metadata (blob retained for CAS/GC lifecycle).
#[utoipa::path(delete, path = "/v1/files/{file_id}", tag = "Files",
    params(("file_id" = String, Path, description = "File ID")),
    responses((status = 200, body = FileDeleteResponse)))]
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
            kind: None,
            description: None,
            image: None,
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
        kind: parsed.kind,
        image: parsed.image,
        marker: doc.marker,
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
        mime_type: descriptor.mime_type,
        kind: descriptor.kind,
        image: descriptor.image,
        marker: descriptor.marker,
    }
}

// ---------------------------------------------------------------------------
// Fail-safe file inspection
// ---------------------------------------------------------------------------

/// Read the blob back and run `file::inspect_bytes` with timeout + panic guard.
/// Returns `None` on any failure — the upload must never be blocked by inspection.
async fn inspect_blob_failsafe(
    blobs: &BlobsHandle,
    blob_ref: &str,
    file_bytes: u64,
    max_inspect_bytes: u64,
) -> Option<file::FileInfo> {
    if file_bytes == 0 || file_bytes > max_inspect_bytes {
        return None;
    }

    let mut stream = blobs.open_stream(blob_ref).ok()?;
    let total = stream.total_size().ok()?;
    let cap = total.min(max_inspect_bytes) as usize;
    let mut buf = vec![0u8; cap];
    let mut pos = 0usize;
    while pos < cap {
        match stream.read(&mut buf[pos..]) {
            Ok(0) => break,
            Ok(n) => pos += n,
            Err(_) => return None,
        }
    }
    buf.truncate(pos);

    let result = tokio::time::timeout(
        Duration::from_secs(2),
        tokio::task::spawn_blocking(move || {
            std::panic::catch_unwind(AssertUnwindSafe(|| file::inspect_bytes(&buf)))
        }),
    )
    .await;

    match result {
        Ok(Ok(Ok(Ok(info)))) => Some(info),
        Ok(Ok(Ok(Err(e)))) => {
            log::warn!(target: "server::files", "file inspection error (non-fatal): {e}");
            None
        }
        Ok(Ok(Err(_panic))) => {
            log::warn!(target: "server::files", "file inspection panicked (non-fatal)");
            None
        }
        Ok(Err(e)) => {
            log::warn!(target: "server::files", "file inspection task join error (non-fatal): {e}");
            None
        }
        Err(_timeout) => {
            log::warn!(target: "server::files", "file inspection timed out (non-fatal)");
            None
        }
    }
}

fn map_file_kind(kind: &file::FileKind) -> &'static str {
    match kind {
        file::FileKind::Binary => "binary",
        file::FileKind::Image => "image",
        file::FileKind::Document => "document",
        file::FileKind::Audio => "audio",
        file::FileKind::Video => "video",
        file::FileKind::Text => "text",
    }
}

fn image_format_str(f: file::ImageFormat) -> String {
    match f {
        file::ImageFormat::Jpeg => "jpeg".to_string(),
        file::ImageFormat::Png => "png".to_string(),
        file::ImageFormat::Webp => "webp".to_string(),
        file::ImageFormat::Unknown => "unknown".to_string(),
    }
}

// ---------------------------------------------------------------------------
// PATCH handler
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct FilePatchRequest {
    #[serde(default)]
    filename: Option<String>,
    #[serde(default)]
    marker: Option<String>,
}

/// PATCH /v1/files/:id - Rename a file (update metadata).
#[utoipa::path(patch, path = "/v1/files/{file_id}", tag = "Files",
    params(("file_id" = String, Path, description = "File ID")),
    request_body = FilePatchRequest,
    responses((status = 200, body = FileObjectResponse)))]
pub async fn handle_patch(
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

    let body_bytes = match req.collect().await {
        Ok(b) => b.to_bytes(),
        Err(e) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_body",
                &format!("Failed to read body: {e}"),
            );
        }
    };

    let patch: FilePatchRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_json",
                &format!("Invalid JSON: {e}"),
            );
        }
    };

    let (doc, storage_path) = match load_file_doc(state, auth, &file_id) {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    // Validate marker if provided.
    let new_marker = match &patch.marker {
        Some(m) => match m.as_str() {
            "active" | "archived" => Some(m.as_str()),
            _ => {
                return json_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_argument",
                    "marker must be 'active' or 'archived'",
                );
            }
        },
        None => None,
    };

    // Determine new filename.
    let new_filename = match patch.filename {
        Some(raw) => match sanitize_filename(&raw) {
            Some(name) => Some(name),
            None => {
                return json_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_argument",
                    "Invalid filename",
                );
            }
        },
        None => None,
    };

    // Nothing to update — return current state.
    if new_filename.is_none() && new_marker.is_none() {
        let descriptor = match descriptor_from_doc(doc) {
            Ok(d) => d,
            Err(resp) => return resp,
        };
        return json_response(StatusCode::OK, &to_file_object_response(descriptor));
    }

    // Update doc_json if filename changed.
    let new_json = if let Some(ref name) = new_filename {
        let mut content: FileDocumentContent =
            serde_json::from_str(&doc.doc_json).unwrap_or(FileDocumentContent {
                blob_ref: None,
                original_name: None,
                mime_type: None,
                size: None,
                purpose: None,
                kind: None,
                description: None,
                image: None,
            });
        content.original_name = Some(name.clone());
        Some(serde_json::to_string(&content).unwrap_or_else(|_| doc.doc_json.clone()))
    } else {
        None
    };

    let docs = match DocumentsHandle::open(&storage_path) {
        Ok(h) => h,
        Err(e) => return document_error_response(e),
    };

    if let Err(e) = docs.update(
        &file_id,
        new_filename.as_deref(),
        new_json.as_deref(),
        None,
        new_marker,
    ) {
        return document_error_response(e);
    }

    // Re-fetch updated document.
    let updated_doc = match docs.get(&file_id) {
        Ok(Some(d)) => d,
        Ok(None) => {
            return json_error(
                StatusCode::NOT_FOUND,
                "not_found",
                &format!("File not found: {file_id}"),
            );
        }
        Err(e) => return document_error_response(e),
    };

    let descriptor = match descriptor_from_doc(updated_doc) {
        Ok(d) => d,
        Err(resp) => return resp,
    };

    json_response(StatusCode::OK, &to_file_object_response(descriptor))
}

// ---------------------------------------------------------------------------
// Cursor helpers (same pattern as conversations.rs)
// ---------------------------------------------------------------------------

/// Decode a base64 file cursor back to `(created_at_ms, file_id)`.
fn decode_file_cursor(encoded: &str) -> Option<(i64, String)> {
    let decoded = base64_decode(encoded)?;
    let s = std::str::from_utf8(&decoded).ok()?;
    let (ts_str, file_id) = s.split_once(':')?;
    let ts = ts_str.parse::<i64>().ok()?;
    Some((ts, file_id.to_string()))
}

/// Simple base64 decode (standard alphabet, with padding).
fn base64_decode(input: &str) -> Option<Vec<u8>> {
    fn val(c: u8) -> Option<u32> {
        match c {
            b'A'..=b'Z' => Some((c - b'A') as u32),
            b'a'..=b'z' => Some((c - b'a' + 26) as u32),
            b'0'..=b'9' => Some((c - b'0' + 52) as u32),
            b'+' => Some(62),
            b'/' => Some(63),
            b'=' => Some(0),
            _ => None,
        }
    }
    let bytes = input.as_bytes();
    if bytes.len() % 4 != 0 {
        return None;
    }
    let mut result = Vec::with_capacity(bytes.len() / 4 * 3);
    for chunk in bytes.chunks(4) {
        let a = val(chunk[0])?;
        let b = val(chunk[1])?;
        let c = val(chunk[2])?;
        let d = val(chunk[3])?;
        let n = (a << 18) | (b << 12) | (c << 6) | d;
        result.push((n >> 16) as u8);
        if chunk[2] != b'=' {
            result.push((n >> 8 & 0xFF) as u8);
        }
        if chunk[3] != b'=' {
            result.push((n & 0xFF) as u8);
        }
    }
    Some(result)
}

fn encode_file_cursor(created_at_ms: i64, file_id: &str) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let raw = format!("{}:{}", created_at_ms, file_id);
    let input = raw.as_bytes();
    let mut result = String::with_capacity((input.len() + 2) / 3 * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        result.push(ALPHABET[(n >> 18 & 0x3F) as usize] as char);
        result.push(ALPHABET[(n >> 12 & 0x3F) as usize] as char);
        if chunk.len() > 1 { result.push(ALPHABET[(n >> 6 & 0x3F) as usize] as char); } else { result.push('='); }
        if chunk.len() > 2 { result.push(ALPHABET[(n & 0x3F) as usize] as char); } else { result.push('='); }
    }
    result
}

// ---------------------------------------------------------------------------
// Batch handler
// ---------------------------------------------------------------------------

/// POST /v1/files/batch - Batch operations on files (delete, archive, unarchive).
#[utoipa::path(post, path = "/v1/files/batch", tag = "Files",
    request_body = FileBatchRequest,
    responses((status = 200)))]
pub async fn handle_batch(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let bucket = match state.bucket_path.as_ref() {
        Some(p) => p,
        None => {
            return json_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "storage_unavailable",
                "Storage not configured",
            );
        }
    };

    let body_bytes = match req.collect().await {
        Ok(b) => b.to_bytes(),
        Err(e) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_body",
                &format!("Failed to read body: {e}"),
            );
        }
    };

    let batch_req: FileBatchRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_json",
                &format!("Invalid JSON: {e}"),
            );
        }
    };

    if batch_req.ids.is_empty() {
        return json_error(StatusCode::BAD_REQUEST, "missing_ids", "ids must not be empty");
    }
    if batch_req.ids.len() > 100 {
        return json_error(
            StatusCode::BAD_REQUEST,
            "batch_too_large",
            &format!("Batch size {} exceeds limit of 100", batch_req.ids.len()),
        );
    }

    let storage_path = match auth.as_ref() {
        Some(ctx) => bucket.join(&ctx.storage_prefix),
        None => bucket.to_path_buf(),
    };

    let docs = match DocumentsHandle::open(&storage_path) {
        Ok(h) => h,
        Err(e) => return document_error_response(e),
    };

    let action = batch_req.action.clone();
    match action.as_str() {
        "delete" | "archive" | "unarchive" => {}
        _ => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_action",
                &format!("Unknown action: {action}"),
            );
        }
    }

    let ids = batch_req.ids;
    let result = tokio::task::spawn_blocking(move || {
        let id_refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
        match action.as_str() {
            "delete" => docs.delete_batch(&id_refs, Some("file")).map(|_| ()),
            "archive" => docs.set_marker_batch(&id_refs, "archived", Some("file")).map(|_| ()),
            "unarchive" => docs.set_marker_batch(&id_refs, "active", Some("file")).map(|_| ()),
            _ => unreachable!(),
        }
    })
    .await;

    match result {
        Ok(Ok(())) => Response::builder()
            .status(StatusCode::NO_CONTENT)
            .body(Full::new(Bytes::new()).boxed())
            .unwrap(),
        Ok(Err(e)) => document_error_response(e),
        Err(e) => json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            &format!("Task failed: {e}"),
        ),
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

fn parse_range_header(
    range_header: Option<&str>,
    total_size: u64,
) -> Result<ContentRangeRequest, ()> {
    let Some(raw_range_header) = range_header else {
        return Ok(ContentRangeRequest::Full);
    };
    let range_header = raw_range_header.trim();
    if !range_header.starts_with("bytes=") {
        return Err(());
    }
    let spec = &range_header["bytes=".len()..];
    if spec.is_empty() || spec.contains(',') {
        return Err(());
    }
    let (start_raw, end_raw) = spec.split_once('-').ok_or(())?;

    if start_raw.is_empty() {
        if end_raw.is_empty() || total_size == 0 {
            return Err(());
        }
        let suffix_len = end_raw.parse::<u64>().map_err(|_| ())?;
        if suffix_len == 0 {
            return Err(());
        }
        let start = total_size.saturating_sub(suffix_len);
        return Ok(ContentRangeRequest::Partial {
            start,
            end: total_size - 1,
        });
    }

    if total_size == 0 {
        return Err(());
    }

    let start = start_raw.parse::<u64>().map_err(|_| ())?;
    if start >= total_size {
        return Err(());
    }

    let end = if end_raw.is_empty() {
        total_size - 1
    } else {
        let parsed_end = end_raw.parse::<u64>().map_err(|_| ())?;
        if parsed_end < start {
            return Err(());
        }
        parsed_end.min(total_size - 1)
    };

    Ok(ContentRangeRequest::Partial { start, end })
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

fn range_not_satisfiable(total_size: u64) -> Response<BoxBody> {
    let body = serde_json::json!({
        "error": {
            "code": "invalid_range",
            "message": "Requested range not satisfiable"
        }
    });
    Response::builder()
        .status(StatusCode::RANGE_NOT_SATISFIABLE)
        .header("content-type", "application/json")
        .header("content-range", format!("bytes */{}", total_size))
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
    use super::{extract_boundary, extract_file_id, resolve_filename, FileDocumentContent};
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

    #[test]
    fn file_doc_content_deserializes_old_format() {
        let json = r#"{"blob_ref":"sha256:abc","original_name":"test.txt","mime_type":"text/plain","size":100,"purpose":"assistants"}"#;
        let parsed: FileDocumentContent = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.blob_ref.as_deref(), Some("sha256:abc"));
        assert_eq!(parsed.original_name.as_deref(), Some("test.txt"));
        assert!(parsed.kind.is_none());
        assert!(parsed.description.is_none());
        assert!(parsed.image.is_none());
    }

    #[test]
    fn file_doc_content_deserializes_enriched_format() {
        let json = r#"{
            "blob_ref": "sha256:abc",
            "original_name": "photo.jpg",
            "mime_type": "image/jpeg",
            "size": 1024,
            "purpose": "assistants",
            "kind": "image",
            "description": "JPEG image data",
            "image": {
                "format": "jpeg",
                "width": 1920,
                "height": 1080,
                "exif_orientation": 1,
                "aspect_ratio": 1.777
            }
        }"#;
        let parsed: FileDocumentContent = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.kind.as_deref(), Some("image"));
        assert_eq!(parsed.description.as_deref(), Some("JPEG image data"));
        let img = parsed.image.unwrap();
        assert_eq!(img.width, 1920);
        assert_eq!(img.height, 1080);
        assert_eq!(img.format, "jpeg");
    }

    #[test]
    fn file_doc_content_roundtrip_serialization() {
        let content = FileDocumentContent {
            blob_ref: Some("sha256:abc".to_string()),
            original_name: Some("test.txt".to_string()),
            mime_type: Some("text/plain".to_string()),
            size: Some(100),
            purpose: Some("assistants".to_string()),
            kind: Some("text".to_string()),
            description: Some("ASCII text".to_string()),
            image: None,
        };
        let json = serde_json::to_string(&content).unwrap();
        let parsed: FileDocumentContent = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.kind.as_deref(), Some("text"));
        assert!(parsed.image.is_none());
        // image field should be absent in JSON (skip_serializing_if)
        assert!(!json.contains("\"image\""));
    }
}
