use hyper::StatusCode;

use super::*;

const PYTHON_MULTI: &str = "import os\n\ndef greet(name):\n    print(name)\n\ndef main():\n    greet('world')\n";

#[tokio::test]
async fn graph_callables_extracts_functions() {
    let app = build_app();
    let body = serde_json::json!({
        "source": PYTHON_MULTI,
        "language": "python",
        "mode": "callables",
        "file_path": "test.py",
        "project_root": "myproject"
    });
    let (status, json) = body_json(send_request(&app, post_json("/v1/code/graph", &body)).await).await;

    assert_eq!(status, StatusCode::OK);
    assert!(
        json.get("callables").is_some(),
        "response should have 'callables' key: {json}"
    );
    let callables = json["callables"].as_array().expect("callables should be an array");
    assert!(
        callables.len() >= 2,
        "should find at least 2 callables (greet, main), got {}",
        callables.len()
    );
}

#[tokio::test]
async fn graph_call_sites() {
    let app = build_app();
    let body = serde_json::json!({
        "source": PYTHON_MULTI,
        "language": "python",
        "mode": "call_sites",
        "definer_fqn": "myproject.test",
        "file_path": "test.py",
        "project_root": "myproject"
    });
    let (status, _json) = body_json(send_request(&app, post_json("/v1/code/graph", &body)).await).await;

    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn graph_invalid_mode() {
    let app = build_app();
    let body = serde_json::json!({
        "source": "x = 1",
        "language": "python",
        "mode": "nonexistent_mode"
    });
    let (status, json) = body_json(send_request(&app, post_json("/v1/code/graph", &body)).await).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(json.get("error").is_some(), "should return error object");
}
