//! `POST /v1/file/inspect` endpoint tests.

use hyper::StatusCode;

use super::*;

// ---------------------------------------------------------------------------
// Happy-path: verify full JSON contract for each file kind
// ---------------------------------------------------------------------------

#[tokio::test]
async fn inspect_text_file_returns_complete_response() {
    let app = build_app();
    let req = multipart_request(
        "POST",
        "/v1/file/inspect",
        "----inspect-text",
        b"Hello, world!",
        "hello.txt",
        "text/plain",
        &[],
    );

    let (status, json) = body_json(send_request(&app, req).await).await;

    assert_eq!(status, StatusCode::OK);
    // Full contract: kind, mime, description, size, image (null for non-image).
    assert_eq!(json["kind"], "text");
    assert_eq!(json["mime"], "text/plain");
    assert_eq!(json["size"], 13);
    assert!(json["image"].is_null());
    // libmagic produces a non-empty description for ASCII text.
    let desc = json["description"]
        .as_str()
        .expect("description should be a string");
    assert!(
        desc.contains("text") || desc.contains("ASCII"),
        "expected text-related description, got: {desc}"
    );
}

#[tokio::test]
async fn inspect_png_returns_complete_image_metadata() {
    let app = build_app();
    let req = multipart_request(
        "POST",
        "/v1/file/inspect",
        "----inspect-png",
        RED_PNG,
        "red.png",
        "image/png",
        &[],
    );

    let (status, json) = body_json(send_request(&app, req).await).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["kind"], "image");
    assert_eq!(json["mime"], "image/png");
    assert_eq!(json["size"], RED_PNG.len() as u64);
    let desc = json["description"].as_str().expect("description");
    assert!(
        desc.contains("PNG"),
        "expected PNG in description, got: {desc}"
    );

    // Verify full image metadata structure.
    let image = &json["image"];
    assert_eq!(image["format"], "png");
    assert_eq!(image["width"], 1);
    assert_eq!(image["height"], 1);
    // PNG has no EXIF; Zig core returns 1 (normal) as the default orientation.
    assert_eq!(image["exif_orientation"], 1);
    assert_eq!(image["aspect_ratio"], 1.0);
}

#[tokio::test]
async fn inspect_jpeg_returns_image_metadata_with_dimensions() {
    let app = build_app();
    let req = multipart_request(
        "POST",
        "/v1/file/inspect",
        "----inspect-jpeg",
        RED_JPEG,
        "red.jpg",
        "image/jpeg",
        &[],
    );

    let (status, json) = body_json(send_request(&app, req).await).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["kind"], "image");
    assert_eq!(json["mime"], "image/jpeg");

    let image = &json["image"];
    assert_eq!(image["format"], "jpeg");
    assert_eq!(image["width"], 1);
    assert_eq!(image["height"], 1);
    assert_eq!(image["aspect_ratio"], 1.0);
}

#[tokio::test]
async fn inspect_webp_returns_image_metadata() {
    let app = build_app();
    let req = multipart_request(
        "POST",
        "/v1/file/inspect",
        "----inspect-webp",
        RED_WEBP,
        "red.webp",
        "image/webp",
        &[],
    );

    let (status, json) = body_json(send_request(&app, req).await).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["kind"], "image");
    assert_eq!(json["mime"], "image/webp");

    let image = &json["image"];
    assert_eq!(image["format"], "webp");
    assert_eq!(image["width"], 1);
    assert_eq!(image["height"], 1);
}

#[tokio::test]
async fn inspect_jpeg_with_exif_orientation_reports_value() {
    let app = build_app();
    let req = multipart_request(
        "POST",
        "/v1/file/inspect",
        "----inspect-exif",
        EXIF_ROTATE90,
        "rotated.jpg",
        "image/jpeg",
        &[],
    );

    let (status, json) = body_json(send_request(&app, req).await).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["kind"], "image");
    assert_eq!(json["image"]["format"], "jpeg");

    // EXIF orientation 6 = 90 degrees clockwise.
    let orientation = json["image"]["exif_orientation"]
        .as_u64()
        .expect("exif_orientation should be an integer");
    assert_eq!(orientation, 6, "expected EXIF orientation 6 (rotate 90 CW)");
}

#[tokio::test]
async fn inspect_non_square_image_reports_correct_aspect_ratio() {
    let app = build_app();
    let req = multipart_request(
        "POST",
        "/v1/file/inspect",
        "----inspect-aspect",
        BLUE_JPEG_2X3,
        "blue.jpg",
        "image/jpeg",
        &[],
    );

    let (status, json) = body_json(send_request(&app, req).await).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["image"]["width"], 2);
    assert_eq!(json["image"]["height"], 3);
    let ratio = json["image"]["aspect_ratio"]
        .as_f64()
        .expect("aspect_ratio");
    let expected = 2.0 / 3.0;
    assert!(
        (ratio - expected).abs() < 1e-6,
        "expected aspect_ratio ~{expected}, got {ratio}"
    );
}

#[tokio::test]
async fn inspect_binary_file_returns_binary_kind() {
    let app = build_app();
    // Bytes 0x00..0x7F include many non-printable characters — libmagic classifies as binary data.
    let garbage: Vec<u8> = (0..128).collect();
    let req = multipart_request(
        "POST",
        "/v1/file/inspect",
        "----inspect-binary",
        &garbage,
        "data.bin",
        "application/octet-stream",
        &[],
    );

    let (status, json) = body_json(send_request(&app, req).await).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["kind"], "binary");
    assert_eq!(json["size"], 128);
    assert!(json["image"].is_null());
    // MIME should be something non-text for binary data.
    let mime = json["mime"].as_str().expect("mime");
    assert!(
        !mime.starts_with("text/"),
        "binary data should not have text/ MIME, got: {mime}"
    );
}

// ---------------------------------------------------------------------------
// Binary format detection: verify kind mapping for non-image, non-text content
// ---------------------------------------------------------------------------

#[tokio::test]
async fn inspect_zip_bytes_returns_binary_kind() {
    let app = build_app();
    // ZIP local file header (PK\x03\x04) contains non-printable bytes (0x03, 0x04, 0x00),
    // so the text heuristic rejects it → classified as binary.
    #[rustfmt::skip]
    let zip: &[u8] = &[
        0x50, 0x4B, 0x03, 0x04, // PK signature
        0x14, 0x00,             // version needed
        0x00, 0x00,             // flags
        0x00, 0x00,             // compression (stored)
        0x00, 0x00, 0x00, 0x00, // mod time/date
        0x00, 0x00, 0x00, 0x00, // CRC-32
        0x00, 0x00, 0x00, 0x00, // compressed size
        0x00, 0x00, 0x00, 0x00, // uncompressed size
        0x01, 0x00,             // file name length: 1
        0x00, 0x00,             // extra field length: 0
        0x61,                   // file name: "a"
        // End of central directory record
        0x50, 0x4B, 0x05, 0x06, // signature
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00,
        0x00, 0x00,             // comment length
    ];
    let req = multipart_request(
        "POST",
        "/v1/file/inspect",
        "----inspect-zip",
        zip,
        "archive.zip",
        "application/zip",
        &[],
    );

    let (status, json) = body_json(send_request(&app, req).await).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["kind"], "binary");
    assert!(json["image"].is_null());
    assert_eq!(json["size"], zip.len() as u64);
    let mime = json["mime"].as_str().expect("mime");
    assert!(
        !mime.starts_with("text/") && !mime.starts_with("image/"),
        "ZIP should not be text or image, got: {mime}"
    );
}

#[tokio::test]
async fn inspect_elf_bytes_returns_binary_kind() {
    let app = build_app();
    // ELF magic (0x7F) exceeds printable ASCII → classified as binary.
    #[rustfmt::skip]
    let elf: &[u8] = &[
        0x7F, 0x45, 0x4C, 0x46, // \x7fELF magic
        0x02, 0x01, 0x01, 0x00, // 64-bit, LE, current version, SysV ABI
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // padding
        0x02, 0x00,             // ET_EXEC
        0x3E, 0x00,             // x86-64
        0x01, 0x00, 0x00, 0x00, // ELF version 1
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];
    let req = multipart_request(
        "POST",
        "/v1/file/inspect",
        "----inspect-elf",
        elf,
        "program",
        "application/octet-stream",
        &[],
    );

    let (status, json) = body_json(send_request(&app, req).await).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["kind"], "binary");
    assert!(json["image"].is_null());
    let mime = json["mime"].as_str().expect("mime");
    assert!(
        !mime.starts_with("text/") && !mime.starts_with("image/"),
        "ELF should not be text or image, got: {mime}"
    );
}

#[tokio::test]
async fn inspect_pdf_header_detected_as_document() {
    let app = build_app();
    // PDF headers start with %PDF- magic bytes. PDF is a rendered format (kind=document)
    // — no intrinsic pixel dimensions; image metadata is null.
    let pdf = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n";
    let req = multipart_request(
        "POST",
        "/v1/file/inspect",
        "----inspect-pdf",
        pdf,
        "doc.pdf",
        "application/pdf",
        &[],
    );

    let (status, json) = body_json(send_request(&app, req).await).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["kind"], "document");
    assert_eq!(json["size"], pdf.len() as u64);
    assert!(json["image"].is_null(), "PDF should have no image metadata");
}

// ---------------------------------------------------------------------------
// Text format detection: verify kind=text for various text types
// ---------------------------------------------------------------------------

#[tokio::test]
async fn inspect_html_returns_text_kind() {
    let app = build_app();
    let html = b"<!DOCTYPE html>\n<html><head><title>Test</title></head><body>Hello</body></html>";
    let req = multipart_request(
        "POST",
        "/v1/file/inspect",
        "----inspect-html",
        html,
        "page.html",
        "text/html",
        &[],
    );

    let (status, json) = body_json(send_request(&app, req).await).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["kind"], "text");
    assert!(json["image"].is_null());
    let mime = json["mime"].as_str().expect("mime");
    assert!(
        mime.starts_with("text/"),
        "HTML (all ASCII) should be detected as text, got: {mime}"
    );
}

#[tokio::test]
async fn inspect_multibyte_utf8_classified_as_binary() {
    let app = build_app();
    // The text heuristic only recognizes ASCII (0x20-0x7E, tab, newline, CR).
    // Multi-byte UTF-8 characters (bytes > 0x7E) exceed the 5% non-ASCII threshold,
    // so content with significant non-ASCII is classified as binary.
    let utf8 = "Héllo wörld — 日本語テスト\n";
    let req = multipart_request(
        "POST",
        "/v1/file/inspect",
        "----inspect-utf8",
        utf8.as_bytes(),
        "intl.txt",
        "text/plain",
        &[],
    );

    let (status, json) = body_json(send_request(&app, req).await).await;

    assert_eq!(status, StatusCode::OK);
    // Multi-byte UTF-8 exceeds the ASCII-only text threshold → binary.
    assert_eq!(json["kind"], "binary");
    assert_eq!(json["size"], utf8.len() as u64);
    assert!(json["image"].is_null());
}

// ---------------------------------------------------------------------------
// Content-based detection: inspect uses bytes, not declared Content-Type
// ---------------------------------------------------------------------------

#[tokio::test]
async fn inspect_detects_mime_from_content_not_declared_type() {
    let app = build_app();
    // Send PNG bytes but declare Content-Type as text/plain in the multipart.
    // The handler must use libmagic (content-based), not the declared type.
    let req = multipart_request(
        "POST",
        "/v1/file/inspect",
        "----inspect-mislabel",
        RED_PNG,
        "not_really.txt",
        "text/plain",
        &[],
    );

    let (status, json) = body_json(send_request(&app, req).await).await;

    assert_eq!(status, StatusCode::OK);
    // libmagic should detect PNG from magic bytes, not trust the declared text/plain.
    assert_eq!(json["kind"], "image");
    assert_eq!(json["mime"], "image/png");
    let image = &json["image"];
    assert_eq!(image["format"], "png");
    assert_eq!(image["width"], 1);
    assert_eq!(image["height"], 1);
}

// ---------------------------------------------------------------------------
// Error paths
// ---------------------------------------------------------------------------

#[tokio::test]
async fn inspect_missing_file_part_returns_400() {
    let app = build_app();
    let req = multipart_no_file(
        "/v1/file/inspect",
        "----inspect-nofile",
        &[("extra", "value")],
    );

    let (status, json) = body_json(send_request(&app, req).await).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_multipart");
}

#[tokio::test]
async fn inspect_non_multipart_returns_400() {
    let app = build_app();
    let req = hyper::Request::builder()
        .method("POST")
        .uri("/v1/file/inspect")
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(b"{}" as &[u8])))
        .unwrap();

    let (status, json) = body_json(send_request(&app, req).await).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_content_type");
}

#[tokio::test]
async fn inspect_empty_file_returns_400() {
    let app = build_app();
    let req = multipart_request(
        "POST",
        "/v1/file/inspect",
        "----inspect-empty",
        b"",
        "empty.bin",
        "application/octet-stream",
        &[],
    );

    let (status, json) = body_json(send_request(&app, req).await).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_multipart");
}

#[tokio::test]
async fn inspect_rejects_payload_over_limit() {
    let app = build_app_with_inspect_limit(64);
    let payload = vec![0x41u8; 256];
    let req = multipart_request(
        "POST",
        "/v1/file/inspect",
        "----inspect-big",
        &payload,
        "big.txt",
        "text/plain",
        &[],
    );

    let (status, json) = body_json(send_request(&app, req).await).await;

    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(json["error"]["code"], "payload_too_large");
}

// ---------------------------------------------------------------------------
// Route variants
// ---------------------------------------------------------------------------

#[tokio::test]
async fn inspect_legacy_path_returns_same_result_as_v1() {
    let app = build_app();

    let req_v1 = multipart_request(
        "POST",
        "/v1/file/inspect",
        "----inspect-v1",
        b"same content",
        "file.txt",
        "text/plain",
        &[],
    );
    let (status_v1, json_v1) = body_json(send_request(&app, req_v1).await).await;

    let req_legacy = multipart_request(
        "POST",
        "/file/inspect",
        "----inspect-legacy",
        b"same content",
        "file.txt",
        "text/plain",
        &[],
    );
    let (status_legacy, json_legacy) = body_json(send_request(&app, req_legacy).await).await;

    assert_eq!(status_v1, StatusCode::OK);
    assert_eq!(status_legacy, StatusCode::OK);
    assert_eq!(json_v1["kind"], json_legacy["kind"]);
    assert_eq!(json_v1["mime"], json_legacy["mime"]);
    assert_eq!(json_v1["size"], json_legacy["size"]);
}
