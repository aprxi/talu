//! `POST /v1/file/transform` endpoint tests.

use hyper::StatusCode;

use super::*;

// ---------------------------------------------------------------------------
// Identity transform: re-encode without options
// ---------------------------------------------------------------------------

#[tokio::test]
async fn transform_png_identity_returns_valid_png_with_size_headers() {
    let app = build_app();
    let req = multipart_request(
        "POST",
        "/v1/file/transform",
        "----transform-id",
        RED_PNG,
        "red.png",
        "image/png",
        &[],
    );

    let resp = send_request(&app, req).await;

    // Exact content-type — identity re-encode of PNG should produce PNG.
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .expect("content-type header required");
    assert_eq!(ct, "image/png");

    let orig_size: u64 = resp
        .headers()
        .get("x-talu-original-size")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok())
        .expect("x-talu-original-size header required");
    assert_eq!(orig_size, RED_PNG.len() as u64);

    let proc_size: u64 = resp
        .headers()
        .get("x-talu-processed-size")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok())
        .expect("x-talu-processed-size header required");

    let (status, body) = body_bytes(resp).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body.len() as u64, proc_size);

    // Verify output is valid PNG with correct 1x1 dimensions.
    let (w, h) = png_dimensions(&body);
    assert_eq!((w, h), (1, 1));
}

// ---------------------------------------------------------------------------
// Format conversion: verify output magic bytes and content-type
// ---------------------------------------------------------------------------

#[tokio::test]
async fn transform_png_to_jpeg_produces_valid_jpeg() {
    let app = build_app();
    let req = multipart_request(
        "POST",
        "/v1/file/transform",
        "----transform-p2j",
        RED_PNG,
        "red.png",
        "image/png",
        &[("format", "jpeg"), ("quality", "90")],
    );

    let resp = send_request(&app, req).await;
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .expect("content-type header");
    assert_eq!(ct, "image/jpeg");

    let (status, body) = body_bytes(resp).await;
    assert_eq!(status, StatusCode::OK);
    // JPEG magic: FF D8 FF.
    assert!(
        body.len() >= 3 && body[0] == 0xFF && body[1] == 0xD8 && body[2] == 0xFF,
        "output should start with JPEG magic bytes, got {:02X?}",
        &body[..body.len().min(4)]
    );
}

#[tokio::test]
async fn transform_jpeg_to_png_produces_valid_png() {
    let app = build_app();
    let req = multipart_request(
        "POST",
        "/v1/file/transform",
        "----transform-j2p",
        RED_JPEG,
        "red.jpg",
        "image/jpeg",
        &[("format", "png")],
    );

    let resp = send_request(&app, req).await;
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .expect("content-type header");
    assert_eq!(ct, "image/png");

    let (status, body) = body_bytes(resp).await;
    assert_eq!(status, StatusCode::OK);
    // Verify valid PNG with correct dimensions.
    let (w, h) = png_dimensions(&body);
    assert_eq!((w, h), (1, 1));
}

#[tokio::test]
async fn transform_webp_to_png_produces_valid_png() {
    let app = build_app();
    // WebP input should be decoded even though WebP output encoding is rejected.
    let req = multipart_request(
        "POST",
        "/v1/file/transform",
        "----transform-w2p",
        RED_WEBP,
        "red.webp",
        "image/webp",
        &[("format", "png")],
    );

    let resp = send_request(&app, req).await;
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .expect("content-type header");
    assert_eq!(ct, "image/png");

    let (status, body) = body_bytes(resp).await;
    assert_eq!(status, StatusCode::OK);
    let (w, h) = png_dimensions(&body);
    assert_eq!((w, h), (1, 1));
}

// ---------------------------------------------------------------------------
// Resize: verify output dimensions
// ---------------------------------------------------------------------------

#[tokio::test]
async fn transform_resize_cover_produces_exact_target_dimensions() {
    let app = build_app();
    // 2x3 input, resize to 1x1 with cover → crops to 1x1 exactly.
    let req = multipart_request(
        "POST",
        "/v1/file/transform",
        "----transform-cover",
        BLUE_JPEG_2X3,
        "blue.jpg",
        "image/jpeg",
        &[("resize", "1x1"), ("fit", "cover"), ("format", "png")],
    );

    let resp = send_request(&app, req).await;
    assert_eq!(
        resp.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some("image/png")
    );

    let (status, body) = body_bytes(resp).await;
    assert_eq!(status, StatusCode::OK);

    let (w, h) = png_dimensions(&body);
    assert_eq!((w, h), (1, 1), "cover mode should produce exact target dimensions");
}

#[tokio::test]
async fn transform_resize_contain_fits_within_bounding_box() {
    let app = build_app();
    // 2x3 input, contain into 10x10 → should scale proportionally.
    // Aspect = 2/3, so fitting into 10x10: height-limited → 6x10 or width-limited → 10x15.
    // Since 2:3, fitting width to 10 → height = 15, exceeds box.
    // Fitting height to 10 → width = 6. So output should be ~6x10, padded to 10x10.
    let req = multipart_request(
        "POST",
        "/v1/file/transform",
        "----transform-contain",
        BLUE_JPEG_2X3,
        "blue.jpg",
        "image/jpeg",
        &[("resize", "10x10"), ("fit", "contain"), ("format", "png")],
    );

    let resp = send_request(&app, req).await;
    let (status, body) = body_bytes(resp).await;
    assert_eq!(status, StatusCode::OK);

    // Contain mode pads to target dimensions.
    let (w, h) = png_dimensions(&body);
    assert_eq!((w, h), (10, 10), "contain mode should produce target dimensions with padding");
}

#[tokio::test]
async fn transform_resize_stretch_distorts_to_target_dimensions() {
    let app = build_app();
    // 2x3 input, stretch to 10x5 — should produce exactly 10x5 (distorted, no padding).
    let req = multipart_request(
        "POST",
        "/v1/file/transform",
        "----transform-stretch",
        BLUE_JPEG_2X3,
        "blue.jpg",
        "image/jpeg",
        &[("resize", "10x5"), ("fit", "stretch"), ("format", "png")],
    );

    let resp = send_request(&app, req).await;
    let (status, body) = body_bytes(resp).await;
    assert_eq!(status, StatusCode::OK);

    let (w, h) = png_dimensions(&body);
    assert_eq!((w, h), (10, 5), "stretch mode should produce exact target dimensions");
}

// ---------------------------------------------------------------------------
// Size limit for transform
// ---------------------------------------------------------------------------

#[tokio::test]
async fn transform_rejects_payload_over_limit() {
    let app = build_app_with_inspect_limit(64);
    // Send a payload larger than the 64-byte limit.
    let payload = vec![0xFFu8; 256];
    let req = multipart_request(
        "POST",
        "/v1/file/transform",
        "----transform-big",
        &payload,
        "big.bin",
        "application/octet-stream",
        &[],
    );

    let (status, json) = body_json(send_request(&app, req).await).await;

    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(json["error"]["code"], "payload_too_large");
}

// ---------------------------------------------------------------------------
// Error paths: parameter validation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn transform_rejects_unsupported_format() {
    let app = build_app();
    let req = multipart_request(
        "POST",
        "/v1/file/transform",
        "----transform-webp",
        RED_PNG,
        "red.png",
        "image/png",
        &[("format", "webp")],
    );

    let (status, json) = body_json(send_request(&app, req).await).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_argument");
    let msg = json["error"]["message"].as_str().expect("message");
    assert!(
        msg.contains("webp"),
        "error message should mention 'webp', got: {msg}"
    );
}

#[tokio::test]
async fn transform_rejects_invalid_resize_string() {
    let app = build_app();
    let req = multipart_request(
        "POST",
        "/v1/file/transform",
        "----transform-badsize",
        RED_PNG,
        "red.png",
        "image/png",
        &[("resize", "abc")],
    );

    let (status, json) = body_json(send_request(&app, req).await).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_argument");
    let msg = json["error"]["message"].as_str().expect("message");
    assert!(msg.contains("abc"), "error should echo the invalid value: {msg}");
}

#[tokio::test]
async fn transform_rejects_zero_dimension_in_resize() {
    let app = build_app();
    let req = multipart_request(
        "POST",
        "/v1/file/transform",
        "----transform-zero",
        RED_PNG,
        "red.png",
        "image/png",
        &[("resize", "0x100")],
    );

    let (status, json) = body_json(send_request(&app, req).await).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_argument");
}

#[tokio::test]
async fn transform_rejects_quality_out_of_range() {
    let app = build_app();
    let req = multipart_request(
        "POST",
        "/v1/file/transform",
        "----transform-badqual",
        RED_PNG,
        "red.png",
        "image/png",
        &[("quality", "200")],
    );

    let (status, json) = body_json(send_request(&app, req).await).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_argument");
    let msg = json["error"]["message"].as_str().expect("message");
    assert!(msg.contains("1-100"), "error should state valid range: {msg}");
}

#[tokio::test]
async fn transform_rejects_non_numeric_quality() {
    let app = build_app();
    let req = multipart_request(
        "POST",
        "/v1/file/transform",
        "----transform-nqual",
        RED_PNG,
        "red.png",
        "image/png",
        &[("quality", "high")],
    );

    let (status, json) = body_json(send_request(&app, req).await).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_argument");
}

#[tokio::test]
async fn transform_rejects_invalid_fit_mode() {
    let app = build_app();
    let req = multipart_request(
        "POST",
        "/v1/file/transform",
        "----transform-badfit",
        RED_PNG,
        "red.png",
        "image/png",
        &[("resize", "10x10"), ("fit", "fill")],
    );

    let (status, json) = body_json(send_request(&app, req).await).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_argument");
    let msg = json["error"]["message"].as_str().expect("message");
    assert!(
        msg.contains("fill"),
        "error should echo the invalid value: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Error paths: missing or invalid input
// ---------------------------------------------------------------------------

#[tokio::test]
async fn transform_missing_file_returns_400() {
    let app = build_app();
    let req = multipart_no_file(
        "/v1/file/transform",
        "----transform-nofile",
        &[("resize", "100x100")],
    );

    let (status, json) = body_json(send_request(&app, req).await).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_multipart");
}

#[tokio::test]
async fn transform_non_image_returns_422() {
    let app = build_app();
    let req = multipart_request(
        "POST",
        "/v1/file/transform",
        "----transform-txt",
        b"this is not an image",
        "readme.txt",
        "text/plain",
        &[],
    );

    let (status, json) = body_json(send_request(&app, req).await).await;

    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(json["error"]["code"], "transform_failed");
}

#[tokio::test]
async fn transform_non_multipart_returns_400() {
    let app = build_app();
    let req = hyper::Request::builder()
        .method("POST")
        .uri("/v1/file/transform")
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(b"{}" as &[u8])))
        .unwrap();

    let (status, json) = body_json(send_request(&app, req).await).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_content_type");
}

// ---------------------------------------------------------------------------
// Route variants
// ---------------------------------------------------------------------------

#[tokio::test]
async fn transform_legacy_path_produces_valid_output() {
    let app = build_app();
    let req = multipart_request(
        "POST",
        "/file/transform",
        "----transform-legacy",
        RED_PNG,
        "red.png",
        "image/png",
        &[("format", "png")],
    );

    let resp = send_request(&app, req).await;
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .expect("content-type header");
    assert_eq!(ct, "image/png");

    let (status, body) = body_bytes(resp).await;
    assert_eq!(status, StatusCode::OK);

    let (w, h) = png_dimensions(&body);
    assert_eq!((w, h), (1, 1));
}
