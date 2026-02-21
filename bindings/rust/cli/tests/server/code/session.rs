use hyper::StatusCode;

use super::*;

const PYTHON_SIMPLE: &str = "def foo():\n    return 42\n";

/// Helper: create a session and return (app, session_id).
async fn create_session(app: &Router) -> String {
    let body = serde_json::json!({
        "source": PYTHON_SIMPLE,
        "language": "python"
    });
    let (status, json) =
        body_json(send_request(app, post_json("/v1/code/session/create", &body)).await).await;
    assert_eq!(status, StatusCode::OK, "create failed: {json}");
    json["session_id"]
        .as_str()
        .expect("create should return session_id")
        .to_string()
}

#[tokio::test]
async fn session_create_returns_id_and_tokens() {
    let app = build_app();
    let body = serde_json::json!({
        "source": PYTHON_SIMPLE,
        "language": "python"
    });
    let (status, json) =
        body_json(send_request(&app, post_json("/v1/code/session/create", &body)).await).await;

    assert_eq!(status, StatusCode::OK);
    assert!(
        json.get("session_id").and_then(|v| v.as_str()).is_some(),
        "should have session_id string"
    );
    let tokens = json
        .get("tokens")
        .and_then(|v| v.as_array())
        .expect("should have tokens array");
    assert!(!tokens.is_empty(), "initial parse should produce tokens");
}

#[tokio::test]
async fn session_highlight_cached_tree() {
    let app = build_app();
    let session_id = create_session(&app).await;

    let body = serde_json::json!({
        "session_id": session_id
    });
    let (status, json) =
        body_json(send_request(&app, post_json("/v1/code/session/highlight", &body)).await).await;

    assert_eq!(status, StatusCode::OK);
    let tokens = json
        .get("tokens")
        .and_then(|v| v.as_array())
        .expect("should have tokens array");
    assert!(!tokens.is_empty(), "highlight should return tokens");
}

#[tokio::test]
async fn session_update_full_source() {
    let app = build_app();
    let session_id = create_session(&app).await;

    let body = serde_json::json!({
        "session_id": session_id,
        "source": "x = 1\ny = 2\n"
    });
    let (status, json) =
        body_json(send_request(&app, post_json("/v1/code/session/update", &body)).await).await;

    assert_eq!(status, StatusCode::OK);
    assert!(
        json.get("tokens").and_then(|v| v.as_array()).is_some(),
        "update should return tokens"
    );
}

#[tokio::test]
async fn session_update_delta() {
    let app = build_app();

    // Create with "x = 1" (5 bytes: x, space, =, space, 1)
    let create_body = serde_json::json!({
        "source": "x = 1",
        "language": "python"
    });
    let (status, json) =
        body_json(send_request(&app, post_json("/v1/code/session/create", &create_body)).await).await;
    assert_eq!(status, StatusCode::OK);
    let session_id = json["session_id"].as_str().unwrap().to_string();

    // Delta: replace "1" (byte 4..5) with "42" (byte 4..6)
    // "x = 1" → "x = 42"
    let update_body = serde_json::json!({
        "session_id": session_id,
        "edits": [{
            "start_byte": 4,
            "old_end_byte": 5,
            "new_text": "42",
            "start_row": 0,
            "start_column": 4,
            "old_end_row": 0,
            "old_end_column": 5,
            "new_end_row": 0,
            "new_end_column": 6
        }]
    });
    let (status, json) =
        body_json(send_request(&app, post_json("/v1/code/session/update", &update_body)).await).await;

    assert_eq!(status, StatusCode::OK);
    assert!(
        json.get("tokens").and_then(|v| v.as_array()).is_some(),
        "delta update should return tokens"
    );
}

#[tokio::test]
async fn session_delete_success() {
    let app = build_app();
    let session_id = create_session(&app).await;

    let resp = send_request(&app, delete(&format!("/v1/code/session/{session_id}"))).await;
    assert_eq!(resp.status(), StatusCode::NO_CONTENT);
}

#[tokio::test]
async fn session_delete_nonexistent() {
    let app = build_app();
    let (status, json) =
        body_json(send_request(&app, delete("/v1/code/session/nonexistent_id_xyz")).await).await;

    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(json.get("error").is_some());
}

#[tokio::test]
async fn session_update_nonexistent() {
    let app = build_app();
    let body = serde_json::json!({
        "session_id": "nonexistent_id_xyz",
        "source": "x = 1"
    });
    let (status, json) =
        body_json(send_request(&app, post_json("/v1/code/session/update", &body)).await).await;

    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(
        json["error"]["code"].as_str(),
        Some("session_not_found")
    );
}

#[tokio::test]
async fn session_update_delta_out_of_bounds() {
    let app = build_app();

    // Create with "x = 1" (5 bytes)
    let create_body = serde_json::json!({
        "source": "x = 1",
        "language": "python"
    });
    let (status, json) =
        body_json(send_request(&app, post_json("/v1/code/session/create", &create_body)).await).await;
    assert_eq!(status, StatusCode::OK);
    let session_id = json["session_id"].as_str().unwrap().to_string();

    // Delta with out-of-bounds byte offsets
    let update_body = serde_json::json!({
        "session_id": session_id,
        "edits": [{
            "start_byte": 100,
            "old_end_byte": 200,
            "new_text": "oops",
            "start_row": 0,
            "start_column": 100,
            "old_end_row": 0,
            "old_end_column": 200,
            "new_end_row": 0,
            "new_end_column": 104
        }]
    });
    let (status, json) =
        body_json(send_request(&app, post_json("/v1/code/session/update", &update_body)).await).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        json["error"]["code"].as_str(),
        Some("edit_failed"),
        "should return edit_failed error: {json}"
    );
}
