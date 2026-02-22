use hyper::StatusCode;

use super::*;

#[tokio::test]
async fn parse_python_returns_ast() {
    let app = build_app();
    let body = serde_json::json!({
        "source": "x = 1",
        "language": "python"
    });
    let (status, json) =
        body_json(send_request(&app, post_json("/v1/code/parse", &body)).await).await;

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
    assert!(
        !children.is_empty(),
        "AST for 'x = 1' should have child nodes"
    );
}

#[tokio::test]
async fn parse_tree_contains_function_definition() {
    let app = build_app();
    let body = serde_json::json!({
        "source": "def foo(): pass",
        "language": "python"
    });
    let (status, json) =
        body_json(send_request(&app, post_json("/v1/code/parse", &body)).await).await;

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
async fn parse_deeply_nested_produces_truncated_node() {
    use http_body_util::BodyExt;

    let app = build_app();

    // Build deeply nested parentheses: x = ((((...))))
    // 300 open + 300 close parens exceeds the AST depth limit (256).
    let mut source = String::with_capacity(605);
    source.push_str("x=");
    for _ in 0..300 {
        source.push('(');
    }
    source.push('1');
    for _ in 0..300 {
        source.push(')');
    }

    let body = serde_json::json!({
        "source": source,
        "language": "python"
    });
    let resp = send_request(&app, post_json("/v1/code/parse", &body)).await;
    assert_eq!(resp.status(), StatusCode::OK);

    // The JSON is too deeply nested for serde_json's default recursion limit,
    // so verify the raw string contains the expected truncation marker and fields.
    let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let raw = String::from_utf8_lossy(&body_bytes);

    assert!(
        raw.contains("\"_truncated\""),
        "deeply nested AST should contain _truncated node"
    );

    // Verify the truncated node includes positional fields so clients don't crash.
    // Extract a ~500 char window after the _truncated marker to check for required fields.
    let trunc_pos = raw.find("\"_truncated\"").unwrap();
    let window_start = raw[..trunc_pos].rfind('{').unwrap();
    let window_end = (window_start + 500).min(raw.len());
    let window = &raw[window_start..window_end];

    assert!(
        window.contains("\"start_byte\":"),
        "_truncated missing start_byte: {window}"
    );
    assert!(
        window.contains("\"end_byte\":"),
        "_truncated missing end_byte: {window}"
    );
    assert!(
        window.contains("\"start_point\":"),
        "_truncated missing start_point: {window}"
    );
    assert!(
        window.contains("\"child_count\":"),
        "_truncated missing child_count: {window}"
    );
}

#[tokio::test]
async fn parse_invalid_language() {
    let app = build_app();
    let body = serde_json::json!({
        "source": "hello",
        "language": "nonexistent_language_xyz"
    });
    let (status, json) =
        body_json(send_request(&app, post_json("/v1/code/parse", &body)).await).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(json.get("error").is_some(), "should return error object");
}
