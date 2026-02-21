use super::*;

#[tokio::test]
async fn ws_create_and_highlight() {
    let app = build_app();
    let mut ws = ws_connect(&app).await;

    // Create session
    let resp = ws_roundtrip(
        &mut ws,
        &serde_json::json!({"type":"create","language":"python","source":"def foo(): pass"}),
    )
    .await;

    assert_eq!(resp["type"], "created");
    assert_eq!(resp["language"], "python");
    let tokens = resp["tokens"].as_array().expect("tokens should be array");
    assert!(!tokens.is_empty(), "create should return tokens");

    // Highlight (no re-parse, uses existing tree)
    let resp = ws_roundtrip(&mut ws, &serde_json::json!({"type":"highlight"})).await;

    assert_eq!(resp["type"], "highlight");
    let tokens = resp["tokens"].as_array().expect("tokens should be array");
    assert!(!tokens.is_empty(), "highlight should return tokens");
}

#[tokio::test]
async fn ws_edit_full_replacement() {
    let app = build_app();
    let mut ws = ws_connect(&app).await;

    // Create
    ws_roundtrip(
        &mut ws,
        &serde_json::json!({"type":"create","language":"python","source":"x = 1"}),
    )
    .await;

    // Edit with full source replacement
    let resp = ws_roundtrip(
        &mut ws,
        &serde_json::json!({"type":"edit","source":"y = 42\nz = 99\n"}),
    )
    .await;

    assert_eq!(resp["type"], "highlight");
    let tokens = resp["tokens"].as_array().expect("tokens should be array");
    assert!(!tokens.is_empty(), "edit should return updated tokens");
}

#[tokio::test]
async fn ws_edit_delta() {
    let app = build_app();
    let mut ws = ws_connect(&app).await;

    // Create with "x = 1" (5 bytes)
    ws_roundtrip(
        &mut ws,
        &serde_json::json!({"type":"create","language":"python","source":"x = 1"}),
    )
    .await;

    // Delta edit: replace "1" (byte 4..5) with "42"
    let resp = ws_roundtrip(
        &mut ws,
        &serde_json::json!({
            "type": "edit",
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
        }),
    )
    .await;

    assert_eq!(resp["type"], "highlight");
    let tokens = resp["tokens"].as_array().expect("tokens should be array");
    assert!(!tokens.is_empty(), "delta edit should return tokens");
}

#[tokio::test]
async fn ws_query() {
    let app = build_app();
    let mut ws = ws_connect(&app).await;

    ws_roundtrip(
        &mut ws,
        &serde_json::json!({"type":"create","language":"python","source":"def hello(): pass"}),
    )
    .await;

    let resp = ws_roundtrip(
        &mut ws,
        &serde_json::json!({"type":"query","query":"(function_definition name: (identifier) @fn)"}),
    )
    .await;

    assert_eq!(resp["type"], "query_result");
    let matches = resp["matches"].as_array().expect("matches should be array");
    assert!(!matches.is_empty(), "should find at least one function");

    // Verify the function name is captured
    let first_match = &matches[0];
    let captures = first_match["captures"]
        .as_array()
        .expect("captures should be array");
    let text = captures[0]["text"]
        .as_str()
        .expect("capture should have text");
    assert_eq!(text, "hello");
}

#[tokio::test]
async fn ws_invalid_message_type() {
    let app = build_app();
    let mut ws = ws_connect(&app).await;

    let resp = ws_roundtrip(
        &mut ws,
        &serde_json::json!({"type":"nonexistent_bogus_type"}),
    )
    .await;

    assert_eq!(resp["type"], "error");
    assert!(
        resp["message"]
            .as_str()
            .is_some_and(|m| m.contains("Unknown message type")),
        "should mention unknown type: {resp}"
    );
}

#[tokio::test]
async fn ws_invalid_json() {
    let app = build_app();
    let mut ws = ws_connect(&app).await;

    // Send invalid JSON
    ws.send(Message::Text("{not valid json".to_string()))
        .await
        .unwrap();

    let reply = ws.next().await.unwrap().unwrap();
    let resp: Value = match reply {
        Message::Text(text) => serde_json::from_str(&text).unwrap(),
        other => panic!("expected text, got: {other:?}"),
    };

    assert_eq!(resp["type"], "error");
    assert_eq!(resp["code"], "invalid_json");
}

#[tokio::test]
async fn ws_highlight_without_create() {
    let app = build_app();
    let mut ws = ws_connect(&app).await;

    // Highlight before creating a session
    let resp = ws_roundtrip(&mut ws, &serde_json::json!({"type":"highlight"})).await;

    assert_eq!(resp["type"], "error");
    assert!(
        resp["message"]
            .as_str()
            .is_some_and(|m| m.contains("No active session")),
        "should mention no active session: {resp}"
    );
}
