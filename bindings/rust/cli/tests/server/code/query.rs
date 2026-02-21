use hyper::StatusCode;

use super::*;

const PYTHON_MULTI: &str = "import os\n\ndef greet(name):\n    print(name)\n\ndef main():\n    greet('world')\n";

#[tokio::test]
async fn query_finds_function_definitions() {
    let app = build_app();
    let body = serde_json::json!({
        "source": PYTHON_MULTI,
        "language": "python",
        "query": "(function_definition) @fn"
    });
    let (status, json) = body_json(send_request(&app, post_json("/v1/code/query", &body)).await).await;

    assert_eq!(status, StatusCode::OK);
    let matches = json.as_array().expect("response should be a JSON array");
    assert!(
        matches.len() >= 2,
        "should find at least 2 function definitions, got {}",
        matches.len()
    );
}

#[tokio::test]
async fn query_captures_have_fields() {
    let app = build_app();
    let body = serde_json::json!({
        "source": "def foo(): pass",
        "language": "python",
        "query": "(function_definition name: (identifier) @name)"
    });
    let (status, json) = body_json(send_request(&app, post_json("/v1/code/query", &body)).await).await;

    assert_eq!(status, StatusCode::OK);
    let matches = json.as_array().expect("response should be a JSON array");
    assert!(!matches.is_empty(), "should have at least one match");

    let captures = matches[0]
        .get("captures")
        .and_then(|v| v.as_array())
        .expect("match should have 'captures' array");
    assert!(!captures.is_empty());

    let capture = &captures[0];
    assert!(capture.get("name").is_some(), "capture missing 'name'");
    assert!(capture.get("start").is_some(), "capture missing 'start'");
    assert!(capture.get("end").is_some(), "capture missing 'end'");
    assert!(capture.get("text").is_some(), "capture missing 'text'");
    assert_eq!(
        capture.get("text").and_then(|v| v.as_str()),
        Some("foo"),
        "captured text should be the function name"
    );
}

#[tokio::test]
async fn query_no_matches_returns_empty() {
    let app = build_app();
    let body = serde_json::json!({
        "source": "x = 1",
        "language": "python",
        "query": "(function_definition) @fn"
    });
    let (status, json) = body_json(send_request(&app, post_json("/v1/code/query", &body)).await).await;

    assert_eq!(status, StatusCode::OK);
    let matches = json.as_array().expect("response should be a JSON array");
    assert!(matches.is_empty(), "should have no matches for code without functions");
}

#[tokio::test]
async fn query_invalid_pattern() {
    let app = build_app();
    let body = serde_json::json!({
        "source": "x = 1",
        "language": "python",
        "query": "(((invalid_unclosed"
    });
    let (status, json) = body_json(send_request(&app, post_json("/v1/code/query", &body)).await).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(json.get("error").is_some(), "should return error object");
}
