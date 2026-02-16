//! Stateless file inspect/transform endpoints.
//!
//! Implements `POST /v1/file/inspect` and `POST /v1/file/transform` using
//! `talu::file` safe wrappers over the Zig core. These endpoints do not
//! persist data — they return metadata or transformed bytes in-memory.

use std::collections::HashMap;
use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::{Request, Response, StatusCode};
use serde::Serialize;
use talu::file::{
    self, FileKind, FitMode, ImageFormat, OutputFormat, ResizeOptions, TransformOptions,
};

use crate::server::auth_gateway::AuthContext;
use crate::server::state::AppState;

type BoxBody = http_body_util::combinators::BoxBody<Bytes, std::convert::Infallible>;

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct FileInspectResponse {
    kind: String,
    mime: String,
    description: String,
    size: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    image: Option<ImageMetadata>,
}

#[derive(Debug, Serialize)]
struct ImageMetadata {
    format: String,
    width: u32,
    height: u32,
    exif_orientation: u8,
    aspect_ratio: f64,
}

// ---------------------------------------------------------------------------
// Buffered multipart reader
// ---------------------------------------------------------------------------

struct BufferedMultipart {
    file_bytes: Vec<u8>,
    fields: HashMap<String, String>,
}

async fn read_multipart_buffered(
    req: Request<Incoming>,
    boundary: &str,
    max_bytes: u64,
) -> Result<BufferedMultipart, Response<BoxBody>> {
    let body_stream = req.into_body().into_data_stream();
    let mut multipart = multer::Multipart::new(body_stream, boundary);

    let mut file_bytes: Option<Vec<u8>> = None;
    let mut fields = HashMap::new();

    while let Some(mut field) = multipart.next_field().await.map_err(|e| {
        json_error(
            StatusCode::BAD_REQUEST,
            "invalid_multipart",
            &format!("Failed to parse multipart field: {}", e),
        )
    })? {
        let field_name = field.name().map(|s| s.to_string());
        match field_name.as_deref() {
            Some("file") => {
                if file_bytes.is_some() {
                    return Err(json_error(
                        StatusCode::BAD_REQUEST,
                        "invalid_multipart",
                        "Multiple file parts are not supported",
                    ));
                }
                let mut buf = Vec::new();
                while let Some(chunk) = field.chunk().await.map_err(|e| {
                    json_error(
                        StatusCode::BAD_REQUEST,
                        "invalid_multipart",
                        &format!("Failed reading file chunk: {}", e),
                    )
                })? {
                    let new_len = buf.len() + chunk.len();
                    if new_len as u64 > max_bytes {
                        return Err(json_error(
                            StatusCode::PAYLOAD_TOO_LARGE,
                            "payload_too_large",
                            &format!(
                                "File exceeds configured limit ({} > {} bytes)",
                                new_len, max_bytes
                            ),
                        ));
                    }
                    buf.extend_from_slice(&chunk);
                }
                file_bytes = Some(buf);
            }
            Some(name) => {
                let value = field.text().await.map_err(|e| {
                    json_error(
                        StatusCode::BAD_REQUEST,
                        "invalid_multipart",
                        &format!("Invalid field '{}': {}", name, e),
                    )
                })?;
                fields.insert(name.to_string(), value.trim().to_string());
            }
            None => {
                // Drain unnamed fields.
                while field
                    .chunk()
                    .await
                    .map_err(|e| {
                        json_error(
                            StatusCode::BAD_REQUEST,
                            "invalid_multipart",
                            &format!("Failed draining multipart field: {}", e),
                        )
                    })?
                    .is_some()
                {}
            }
        }
    }

    let file_bytes = file_bytes.ok_or_else(|| {
        json_error(
            StatusCode::BAD_REQUEST,
            "invalid_multipart",
            "Missing required multipart field 'file'",
        )
    })?;

    if file_bytes.is_empty() {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_multipart",
            "File part is empty",
        ));
    }

    Ok(BufferedMultipart { file_bytes, fields })
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// POST /v1/file/inspect - Detect file type, MIME, and image metadata.
pub async fn handle_inspect(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let boundary = match extract_boundary(&req) {
        Some(b) => b,
        None => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_content_type",
                "Expected multipart/form-data with boundary",
            )
        }
    };

    let max_bytes = state.max_file_inspect_bytes;
    let parsed = match read_multipart_buffered(req, &boundary, max_bytes).await {
        Ok(p) => p,
        Err(resp) => return resp,
    };

    let bytes = parsed.file_bytes;
    let size = bytes.len() as u64;

    let info = match tokio::task::spawn_blocking(move || file::inspect_bytes(&bytes)).await {
        Ok(Ok(info)) => info,
        Ok(Err(e)) => {
            return json_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "inspect_failed",
                &format!("File inspection failed: {}", e),
            )
        }
        Err(e) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                &format!("Inspection task failed: {}", e),
            )
        }
    };

    let kind = map_kind(&info.kind, &info.mime);
    let image = info.image.map(|img| ImageMetadata {
        format: image_format_to_string(img.format),
        width: img.width,
        height: img.height,
        exif_orientation: img.exif_orientation,
        aspect_ratio: if img.height > 0 {
            img.width as f64 / img.height as f64
        } else {
            0.0
        },
    });

    let response = FileInspectResponse {
        kind: kind.to_string(),
        mime: info.mime,
        description: info.description,
        size,
        image,
    };

    json_response(StatusCode::OK, &response)
}

/// POST /v1/file/transform - Resize/re-encode an image.
pub async fn handle_transform(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let boundary = match extract_boundary(&req) {
        Some(b) => b,
        None => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_content_type",
                "Expected multipart/form-data with boundary",
            )
        }
    };

    let max_bytes = state.max_file_inspect_bytes;
    let parsed = match read_multipart_buffered(req, &boundary, max_bytes).await {
        Ok(p) => p,
        Err(resp) => return resp,
    };

    let original_size = parsed.file_bytes.len() as u64;
    let opts = match build_transform_options(&parsed.fields) {
        Ok(opts) => opts,
        Err(resp) => return resp,
    };

    let bytes = parsed.file_bytes;
    let result = match tokio::task::spawn_blocking(move || {
        file::transform_image_bytes(&bytes, opts)
    })
    .await
    {
        Ok(Ok(result)) => result,
        Ok(Err(e)) => {
            return json_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "transform_failed",
                &format!("Image transformation failed: {}", e),
            )
        }
        Err(e) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                &format!("Transform task failed: {}", e),
            )
        }
    };

    let content_type = result
        .image
        .as_ref()
        .map(|img| image_format_to_content_type(img.format))
        .unwrap_or("application/octet-stream");

    let processed_size = result.bytes.len() as u64;

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", content_type)
        .header("x-talu-original-size", original_size.to_string())
        .header("x-talu-processed-size", processed_size.to_string())
        .body(Full::new(Bytes::from(result.bytes)).boxed())
        .unwrap()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn extract_boundary(req: &Request<Incoming>) -> Option<String> {
    let ct = req.headers().get("content-type")?.to_str().ok()?;
    multer::parse_boundary(ct).ok()
}

fn map_kind(kind: &FileKind, mime: &str) -> &'static str {
    match kind {
        FileKind::Image => "image",
        FileKind::Unknown => {
            if mime.starts_with("text/") {
                "text"
            } else {
                "binary"
            }
        }
    }
}

fn image_format_to_string(f: ImageFormat) -> String {
    match f {
        ImageFormat::Jpeg => "jpeg".to_string(),
        ImageFormat::Png => "png".to_string(),
        ImageFormat::Webp => "webp".to_string(),
        ImageFormat::Unknown => "unknown".to_string(),
    }
}

fn image_format_to_content_type(f: ImageFormat) -> &'static str {
    match f {
        ImageFormat::Jpeg => "image/jpeg",
        ImageFormat::Png => "image/png",
        ImageFormat::Webp => "image/webp",
        ImageFormat::Unknown => "application/octet-stream",
    }
}

fn build_transform_options(
    fields: &HashMap<String, String>,
) -> Result<TransformOptions, Response<BoxBody>> {
    let mut opts = TransformOptions::default();

    if let Some(resize_str) = fields.get("resize") {
        let (w, h) = parse_resize(resize_str).ok_or_else(|| {
            json_error(
                StatusCode::BAD_REQUEST,
                "invalid_argument",
                &format!(
                    "Invalid resize value '{}': expected 'WIDTHxHEIGHT' (e.g. '1024x1024')",
                    resize_str
                ),
            )
        })?;
        let fit = fields
            .get("fit")
            .map(|s| parse_fit_mode(s))
            .transpose()
            .map_err(|msg| json_error(StatusCode::BAD_REQUEST, "invalid_argument", &msg))?
            .unwrap_or(FitMode::Contain);
        opts.resize = Some(ResizeOptions {
            width: w,
            height: h,
            fit,
            filter: talu::file::ResizeFilter::Bicubic,
        });
    }

    if let Some(format_str) = fields.get("format") {
        opts.output_format = Some(
            parse_output_format(format_str)
                .map_err(|msg| json_error(StatusCode::BAD_REQUEST, "invalid_argument", &msg))?,
        );
    }

    if let Some(quality_str) = fields.get("quality") {
        let q: u8 = quality_str.parse().map_err(|_| {
            json_error(
                StatusCode::BAD_REQUEST,
                "invalid_argument",
                &format!(
                    "Invalid quality value '{}': expected integer 1-100",
                    quality_str
                ),
            )
        })?;
        if q == 0 || q > 100 {
            return Err(json_error(
                StatusCode::BAD_REQUEST,
                "invalid_argument",
                &format!("Quality must be 1-100, got {}", q),
            ));
        }
        opts.jpeg_quality = q;
    }

    Ok(opts)
}

fn parse_resize(s: &str) -> Option<(u32, u32)> {
    let (w_str, h_str) = s.split_once('x')?;
    let w: u32 = w_str.trim().parse().ok()?;
    let h: u32 = h_str.trim().parse().ok()?;
    if w == 0 || h == 0 {
        return None;
    }
    Some((w, h))
}

fn parse_fit_mode(s: &str) -> Result<FitMode, String> {
    match s {
        "stretch" => Ok(FitMode::Stretch),
        "contain" => Ok(FitMode::Contain),
        "cover" => Ok(FitMode::Cover),
        _ => Err(format!(
            "Invalid fit mode '{}': expected 'stretch', 'contain', or 'cover'",
            s
        )),
    }
}

fn parse_output_format(s: &str) -> Result<OutputFormat, String> {
    match s {
        "jpeg" | "jpg" => Ok(OutputFormat::Jpeg),
        "png" => Ok(OutputFormat::Png),
        _ => Err(format!(
            "Unsupported output format '{}': expected 'jpeg' or 'png'",
            s
        )),
    }
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_resize_valid() {
        assert_eq!(parse_resize("1024x768"), Some((1024, 768)));
        assert_eq!(parse_resize("1x1"), Some((1, 1)));
        assert_eq!(parse_resize(" 800 x 600 "), Some((800, 600)));
    }

    #[test]
    fn parse_resize_invalid() {
        assert_eq!(parse_resize("abc"), None);
        assert_eq!(parse_resize("100"), None);
        assert_eq!(parse_resize("0x100"), None);
        assert_eq!(parse_resize("100x0"), None);
        assert_eq!(parse_resize("x100"), None);
        assert_eq!(parse_resize(""), None);
    }

    #[test]
    fn parse_fit_mode_variants() {
        assert_eq!(parse_fit_mode("stretch").unwrap(), FitMode::Stretch);
        assert_eq!(parse_fit_mode("contain").unwrap(), FitMode::Contain);
        assert_eq!(parse_fit_mode("cover").unwrap(), FitMode::Cover);
        assert!(parse_fit_mode("fill").is_err());
        assert!(parse_fit_mode("").is_err());
    }

    #[test]
    fn parse_output_format_variants() {
        assert_eq!(parse_output_format("jpeg").unwrap(), OutputFormat::Jpeg);
        assert_eq!(parse_output_format("jpg").unwrap(), OutputFormat::Jpeg);
        assert_eq!(parse_output_format("png").unwrap(), OutputFormat::Png);
        assert!(parse_output_format("webp").is_err());
        assert!(parse_output_format("gif").is_err());
    }

    #[test]
    fn map_kind_image() {
        assert_eq!(map_kind(&FileKind::Image, "image/jpeg"), "image");
        assert_eq!(map_kind(&FileKind::Image, "image/png"), "image");
    }

    #[test]
    fn map_kind_text() {
        assert_eq!(map_kind(&FileKind::Unknown, "text/plain"), "text");
        assert_eq!(map_kind(&FileKind::Unknown, "text/html"), "text");
    }

    #[test]
    fn map_kind_binary() {
        assert_eq!(map_kind(&FileKind::Unknown, "application/pdf"), "binary");
        assert_eq!(
            map_kind(&FileKind::Unknown, "application/octet-stream"),
            "binary"
        );
        assert_eq!(map_kind(&FileKind::Unknown, ""), "binary");
    }

    #[test]
    fn image_format_strings() {
        assert_eq!(image_format_to_string(ImageFormat::Jpeg), "jpeg");
        assert_eq!(image_format_to_string(ImageFormat::Png), "png");
        assert_eq!(image_format_to_string(ImageFormat::Webp), "webp");
        assert_eq!(image_format_to_string(ImageFormat::Unknown), "unknown");
    }

    #[test]
    fn image_format_content_types() {
        assert_eq!(
            image_format_to_content_type(ImageFormat::Jpeg),
            "image/jpeg"
        );
        assert_eq!(image_format_to_content_type(ImageFormat::Png), "image/png");
        assert_eq!(
            image_format_to_content_type(ImageFormat::Webp),
            "image/webp"
        );
        assert_eq!(
            image_format_to_content_type(ImageFormat::Unknown),
            "application/octet-stream"
        );
    }
}
