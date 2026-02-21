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
    assert_eq!(
        json.get("language").and_then(|v| v.as_str()),
        Some("python"),
        "language should be 'python'"
    );
    let tree = json.get("tree").expect("response should have 'tree' key");
    assert_eq!(
        tree.get("kind").and_then(|v| v.as_str()),
        Some("module"),
        "root node kind should be 'module'"
    );
    let children = tree
        .get("children")
        .and_then(|v| v.as_array())
        .expect("root node should have 'children' array");
    assert!(!children.is_empty(), "AST for 'x = 1' should have child nodes");
}

#[tokio::test]
async fn parse_tree_contains_function_definition() {
    let app = build_app();
    let body = serde_json::json!({
        "source": "def foo(): pass",
        "language": "python"
    });
    let (status, json) = body_json(send_request(&app, post_json("/v1/code/parse", &body)).await).await;

    assert_eq!(status, StatusCode::OK);
    let tree = json.get("tree").expect("should have 'tree'");
    let children = tree
        .get("children")
        .and_then(|v| v.as_array())
        .expect("root should have children");

    let has_function_def = children
        .iter()
        .any(|node| node.get("kind").and_then(|v| v.as_str()) == Some("function_definition"));
    assert!(
        has_function_def,
        "AST should contain a function_definition node, got: {:?}",
        children.iter().map(|n| n.get("kind")).collect::<Vec<_>>()
    );
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
