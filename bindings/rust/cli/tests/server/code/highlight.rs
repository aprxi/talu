use hyper::StatusCode;

use super::*;

#[tokio::test]
async fn highlight_python_returns_tokens() {
    let app = build_app();
    let body = serde_json::json!({
        "source": "def foo():\n    return 42\n",
        "language": "python"
    });
    let (status, json) =
        body_json(send_request(&app, post_json("/v1/code/highlight", &body)).await).await;

    assert_eq!(status, StatusCode::OK);
    let tokens = json.as_array().expect("response should be a JSON array");
    assert!(!tokens.is_empty(), "should have at least one token");

    // Every token must have s (start byte), e (end byte), t (type/CSS class).
    for token in tokens {
        assert!(token.get("s").is_some(), "token missing 's': {token}");
        assert!(token.get("e").is_some(), "token missing 'e': {token}");
        assert!(token.get("t").is_some(), "token missing 't': {token}");
    }
}

#[tokio::test]
async fn highlight_rich_includes_positions() {
    let app = build_app();
    let body = serde_json::json!({
        "source": "x = 1",
        "language": "python",
        "rich": true
    });
    let (status, json) =
        body_json(send_request(&app, post_json("/v1/code/highlight", &body)).await).await;

    assert_eq!(status, StatusCode::OK);
    let tokens = json.as_array().expect("response should be a JSON array");
    assert!(!tokens.is_empty());

    // Rich tokens include node kind (nk), text (tx), and position fields.
    for token in tokens {
        assert!(
            token.get("nk").is_some(),
            "rich token missing 'nk': {token}"
        );
        assert!(
            token.get("tx").is_some(),
            "rich token missing 'tx': {token}"
        );
        assert!(
            token.get("sr").is_some(),
            "rich token missing 'sr': {token}"
        );
        assert!(
            token.get("sc").is_some(),
            "rich token missing 'sc': {token}"
        );
        assert!(
            token.get("er").is_some(),
            "rich token missing 'er': {token}"
        );
        assert!(
            token.get("ec").is_some(),
            "rich token missing 'ec': {token}"
        );
    }
}

#[tokio::test]
async fn highlight_javascript() {
    let app = build_app();
    let body = serde_json::json!({
        "source": "function bar() { return 1; }\n",
        "language": "javascript"
    });
    let (status, json) =
        body_json(send_request(&app, post_json("/v1/code/highlight", &body)).await).await;

    assert_eq!(status, StatusCode::OK);
    let tokens = json.as_array().expect("response should be a JSON array");
    assert!(
        !tokens.is_empty(),
        "JavaScript source should produce tokens"
    );

    for token in tokens {
        assert!(token.get("s").is_some(), "token missing 's': {token}");
        assert!(token.get("e").is_some(), "token missing 'e': {token}");
        assert!(token.get("t").is_some(), "token missing 't': {token}");
    }
}

#[tokio::test]
async fn highlight_invalid_language() {
    let app = build_app();
    let body = serde_json::json!({
        "source": "hello",
        "language": "nonexistent_language_xyz"
    });
    let (status, json) =
        body_json(send_request(&app, post_json("/v1/code/highlight", &body)).await).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(json.get("error").is_some(), "should return error object");
}

#[tokio::test]
async fn highlight_empty_source() {
    let app = build_app();
    let body = serde_json::json!({
        "source": "",
        "language": "python"
    });
    let (status, json) =
        body_json(send_request(&app, post_json("/v1/code/highlight", &body)).await).await;

    assert_eq!(status, StatusCode::OK);
    let tokens = json.as_array().expect("response should be a JSON array");
    assert!(tokens.is_empty(), "empty source should produce no tokens");
}
