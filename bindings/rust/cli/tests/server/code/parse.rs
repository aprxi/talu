use hyper::StatusCode;

use super::*;

#[tokio::test]
async fn parse_python_returns_ast() {
    let app = build_app();
    let body = serde_json::json!({
        "source": "x = 1",
        "language": "python"
    });
    let (status, json) = body_json(send_request(&app, post_json("/v1/code/parse", &body)).await).await;

    assert_eq!(status, StatusCode::OK);
    assert!(json.get("tree").is_some(), "response should have 'tree' key");
    assert!(json.get("language").is_some(), "response should have 'language' key");
}

#[tokio::test]
async fn parse_root_node_kind() {
    let app = build_app();
    let body = serde_json::json!({
        "source": "def foo(): pass",
        "language": "python"
    });
    let (status, json) = body_json(send_request(&app, post_json("/v1/code/parse", &body)).await).await;

    assert_eq!(status, StatusCode::OK);
    let tree = json.get("tree").expect("should have 'tree'");
    let kind = tree.get("kind").and_then(|v| v.as_str());
    assert_eq!(kind, Some("module"), "Python root node should be 'module'");
}

#[tokio::test]
async fn parse_invalid_language() {
    let app = build_app();
    let body = serde_json::json!({
        "source": "hello",
        "language": "nonexistent_language_xyz"
    });
    let (status, json) = body_json(send_request(&app, post_json("/v1/code/parse", &body)).await).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(json.get("error").is_some(), "should return error object");
}
