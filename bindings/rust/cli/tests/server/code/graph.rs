use hyper::StatusCode;

use super::*;

const PYTHON_MULTI: &str =
    "import os\n\ndef greet(name):\n    print(name)\n\ndef main():\n    greet('world')\n";

/// Source with a module-level call site for call_sites extraction.
const PYTHON_MODULE_CALLS: &str =
    "import os\n\ndef greet(name):\n    print(name)\n\ngreet('world')\n";

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
    let (status, json) =
        body_json(send_request(&app, post_json("/v1/code/graph", &body)).await).await;

    assert_eq!(status, StatusCode::OK);
    assert!(
        json.get("callables").is_some(),
        "response should have 'callables' key: {json}"
    );
    let callables = json["callables"]
        .as_array()
        .expect("callables should be an array");
    assert!(
        callables.len() >= 2,
        "should find at least 2 callables (greet, main), got {}",
        callables.len()
    );

    // Verify the actual function names are present in the callables.
    let names: Vec<&str> = callables
        .iter()
        .filter_map(|c| c.get("fqn").and_then(|v| v.as_str()))
        .collect();
    assert!(
        names.iter().any(|n| n.contains("greet")),
        "callables should include 'greet', got: {names:?}"
    );
    assert!(
        names.iter().any(|n| n.contains("main")),
        "callables should include 'main', got: {names:?}"
    );
}

#[tokio::test]
async fn graph_call_sites() {
    let app = build_app();

    // Use source with a module-level call site (call_sites extracts top-level calls only).
    let body = serde_json::json!({
        "source": PYTHON_MODULE_CALLS,
        "language": "python",
        "mode": "call_sites",
        "definer_fqn": "::test",
        "file_path": "test.py",
        "project_root": "myproject"
    });
    let (status, json) =
        body_json(send_request(&app, post_json("/v1/code/graph", &body)).await).await;

    assert_eq!(status, StatusCode::OK);
    let call_sites = json
        .as_array()
        .expect("call_sites response should be a JSON array");
    // The source has `greet('world')` at module level.
    assert!(
        !call_sites.is_empty(),
        "should find at least 1 module-level call site (greet)"
    );

    // Verify call site structure: each should have target name and spans.
    for site in call_sites {
        assert!(
            site.get("raw_target_name")
                .and_then(|v| v.as_str())
                .is_some(),
            "call site missing 'raw_target_name': {site}"
        );
        assert!(
            site.get("call_expr_span").is_some(),
            "call site missing 'call_expr_span': {site}"
        );
        assert!(
            site.get("definer_callable_fqn").and_then(|v| v.as_str()) == Some("::test"),
            "call site should carry definer FQN"
        );
    }
}

#[tokio::test]
async fn graph_invalid_mode() {
    let app = build_app();
    let body = serde_json::json!({
        "source": "x = 1",
        "language": "python",
        "mode": "nonexistent_mode"
    });
    let (status, json) =
        body_json(send_request(&app, post_json("/v1/code/graph", &body)).await).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(json.get("error").is_some(), "should return error object");
}
