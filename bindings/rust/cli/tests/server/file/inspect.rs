//! `POST /v1/file/inspect` endpoint tests.

use hyper::StatusCode;

use super::*;

#[tokio::test]
async fn inspect_text_file_returns_text_kind() {
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
    assert_eq!(json["kind"], "text");
    assert!(json["mime"].as_str().unwrap_or("").starts_with("text/"));
    assert_eq!(json["size"], 13);
    assert!(json["image"].is_null());
}

#[tokio::test]
async fn inspect_png_returns_image_metadata() {
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

    let image = &json["image"];
    assert_eq!(image["format"], "png");
    assert_eq!(image["width"], 1);
    assert_eq!(image["height"], 1);
    assert_eq!(image["aspect_ratio"], 1.0);
}

#[tokio::test]
async fn inspect_jpeg_returns_image_metadata() {
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
    assert!(image["exif_orientation"].is_number());
}

#[tokio::test]
async fn inspect_binary_file_returns_binary_kind() {
    let app = build_app();
    // Random bytes that aren't valid text or image.
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
}

#[tokio::test]
async fn inspect_missing_file_part_returns_400() {
    let app = build_app();
    let req = multipart_no_file("/v1/file/inspect", "----inspect-nofile", &[("extra", "value")]);

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
    let payload = vec![0x41u8; 256]; // 256 bytes exceeds 64 byte limit
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

#[tokio::test]
async fn inspect_legacy_path_works() {
    let app = build_app();
    let req = multipart_request(
        "POST",
        "/file/inspect",
        "----inspect-legacy",
        b"plain text content",
        "file.txt",
        "text/plain",
        &[],
    );

    let (status, json) = body_json(send_request(&app, req).await).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["kind"], "text");
}

#[tokio::test]
async fn inspect_description_is_nonempty() {
    let app = build_app();
    let req = multipart_request(
        "POST",
        "/v1/file/inspect",
        "----inspect-desc",
        RED_PNG,
        "test.png",
        "image/png",
        &[],
    );

    let (status, json) = body_json(send_request(&app, req).await).await;

    assert_eq!(status, StatusCode::OK);
    let desc = json["description"].as_str().unwrap_or("");
    assert!(!desc.is_empty(), "description should not be empty");
}
