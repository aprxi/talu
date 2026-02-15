//! `POST /v1/file/transform` endpoint tests.

use hyper::StatusCode;

use super::*;

#[tokio::test]
async fn transform_png_returns_image_bytes() {
    let app = build_app();
    let req = multipart_request(
        "POST",
        "/v1/file/transform",
        "----transform-basic",
        RED_PNG,
        "red.png",
        "image/png",
        &[],
    );

    let resp = send_request(&app, req).await;
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        ct.starts_with("image/"),
        "expected image content-type, got '{ct}'"
    );

    let orig_size = resp
        .headers()
        .get("x-talu-original-size")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());
    assert_eq!(orig_size, Some(RED_PNG.len() as u64));

    let proc_size_header = resp
        .headers()
        .get("x-talu-processed-size")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());
    assert!(proc_size_header.is_some());

    let (status, body) = body_bytes(resp).await;
    assert_eq!(status, StatusCode::OK);
    assert!(!body.is_empty());
    assert_eq!(Some(body.len() as u64), proc_size_header);
}

#[tokio::test]
async fn transform_with_format_conversion() {
    let app = build_app();
    let req = multipart_request(
        "POST",
        "/v1/file/transform",
        "----transform-fmt",
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
        .unwrap_or("");
    assert_eq!(ct, "image/jpeg");

    let (status, body) = body_bytes(resp).await;
    assert_eq!(status, StatusCode::OK);
    // JPEG starts with FF D8 FF.
    assert!(
        body.len() >= 3 && body[0] == 0xFF && body[1] == 0xD8 && body[2] == 0xFF,
        "expected JPEG magic bytes"
    );
}

#[tokio::test]
async fn transform_jpeg_to_png() {
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
        .unwrap_or("");
    assert_eq!(ct, "image/png");

    let (status, body) = body_bytes(resp).await;
    assert_eq!(status, StatusCode::OK);
    // PNG starts with 89 50 4E 47.
    assert!(
        body.len() >= 4 && body[0] == 0x89 && body[1] == 0x50 && body[2] == 0x4E && body[3] == 0x47,
        "expected PNG magic bytes"
    );
}

#[tokio::test]
async fn transform_with_resize() {
    // Use the 2x3 blue JPEG for a meaningful resize test.
    const BLUE_JPEG: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../core/tests/image/corpus/2x3_blue.jpg"
    ));

    let app = build_app();
    let req = multipart_request(
        "POST",
        "/v1/file/transform",
        "----transform-resize",
        BLUE_JPEG,
        "blue.jpg",
        "image/jpeg",
        &[("resize", "1x1"), ("fit", "cover"), ("format", "png")],
    );

    let resp = send_request(&app, req).await;
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert_eq!(ct, "image/png");

    let (status, body) = body_bytes(resp).await;
    assert_eq!(status, StatusCode::OK);
    assert!(!body.is_empty());
}

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
    let msg = json["error"]["message"].as_str().unwrap_or("");
    assert!(msg.contains("webp"), "error should mention webp: {msg}");
}

#[tokio::test]
async fn transform_rejects_invalid_resize() {
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
}

#[tokio::test]
async fn transform_rejects_zero_resize() {
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
async fn transform_rejects_invalid_quality() {
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
        &[("resize", "10x10"), ("fit", "invalid")],
    );

    let (status, json) = body_json(send_request(&app, req).await).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_argument");
}

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
async fn transform_legacy_path_works() {
    let app = build_app();
    let req = multipart_request(
        "POST",
        "/file/transform",
        "----transform-legacy",
        RED_PNG,
        "red.png",
        "image/png",
        &[],
    );

    let resp = send_request(&app, req).await;
    let (status, body) = body_bytes(resp).await;

    assert_eq!(status, StatusCode::OK);
    assert!(!body.is_empty());
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
