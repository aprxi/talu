use hyper::StatusCode;

use super::*;

#[tokio::test]
async fn languages_returns_list() {
    let app = build_app();
    let (status, json) = body_json(send_request(&app, get("/v1/code/languages")).await).await;

    assert_eq!(status, StatusCode::OK);
    let langs = json
        .get("languages")
        .and_then(|v| v.as_str())
        .expect("response should have 'languages' string");
    assert!(!langs.is_empty(), "languages list should not be empty");
}

#[tokio::test]
async fn languages_includes_python() {
    let app = build_app();
    let (status, json) = body_json(send_request(&app, get("/v1/code/languages")).await).await;

    assert_eq!(status, StatusCode::OK);
    let langs = json
        .get("languages")
        .and_then(|v| v.as_str())
        .expect("response should have 'languages' string");
    assert!(
        langs.contains("python"),
        "languages should include 'python', got: {langs}"
    );
}
