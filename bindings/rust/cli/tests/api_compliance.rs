//! HTTP API compliance tests (Phase 6).
//!
//! Full-stack end-to-end tests exercising the real production chain:
//! Router → Handlers → talu → libtalu (Zig) → Model.
//!
//! Uses in-process hyper connections (no TCP socket) for speed and
//! determinism. Everything else is real, including inference on
//! the model set via `TALU_TEST_MODEL`.
//!
//! Run: `LD_LIBRARY_PATH=zig-out/lib cargo test --test api_compliance -- --test-threads=1`

use std::collections::HashMap;
use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::client::conn::http1 as client_http1;
use hyper::server::conn::http1 as server_http1;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use hyper_util::service::TowerToHyperService;
use serde_json::Value;
use tokio::sync::Mutex;

use talu::InferenceBackend;
use talu_cli::server::http::Router;
use talu_cli::server::state::{AppState, BackendState};

// ---------------------------------------------------------------------------
// Test model path (from TALU_TEST_MODEL env or default)
// ---------------------------------------------------------------------------

fn model_path() -> String {
    std::env::var("TALU_TEST_MODEL")
        .unwrap_or_else(|_| "models/LiquidAI/LFM2-350M-GAF4".to_string())
}

// ---------------------------------------------------------------------------
// Test harness: in-process HTTP client ↔ server via duplex stream
// ---------------------------------------------------------------------------

/// Send a request through an in-process hyper connection to the Router.
///
/// Creates a `tokio::io::duplex` pair, spawns hyper's HTTP/1.1 server on
/// one end (serving the Router), and sends the request via hyper's HTTP/1.1
/// client on the other end. Returns the real HTTP response.
async fn send_request(router: &Router, req: Request<Full<Bytes>>) -> Response<Incoming> {
    let (client_io, server_io) = tokio::io::duplex(1024 * 1024);

    // Spawn server side
    let service = TowerToHyperService::new(router.clone());
    tokio::spawn(async move {
        let _ = server_http1::Builder::new()
            .serve_connection(TokioIo::new(server_io), service)
            .await;
    });

    // Client side
    let (mut sender, conn) = client_http1::handshake(TokioIo::new(client_io))
        .await
        .expect("client handshake failed");

    tokio::spawn(async move {
        let _ = conn.await;
    });

    sender.send_request(req).await.expect("send_request failed")
}

/// Collect a response body to bytes.
async fn body_bytes(resp: Response<Incoming>) -> (StatusCode, Bytes) {
    let status = resp.status();
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    (status, body)
}

/// Collect a response body to JSON.
async fn body_json(resp: Response<Incoming>) -> (StatusCode, Value) {
    let (status, bytes) = body_bytes(resp).await;
    let json: Value = serde_json::from_slice(&bytes).unwrap_or_else(|e| {
        panic!(
            "invalid JSON in response body: {e}\nbody: {}",
            String::from_utf8_lossy(&bytes)
        )
    });
    (status, json)
}

/// Build a Router with a real backend loaded (model inference enabled).
fn build_app_with_model() -> Router {
    let path = model_path();
    let backend =
        InferenceBackend::new(&path).unwrap_or_else(|e| panic!("failed to load model {path}: {e}"));

    let state = AppState {
        backend: Arc::new(Mutex::new(BackendState {
            backend: Some(backend),
            current_model: Some(path.clone()),
        })),
        configured_model: Some(path),
        response_store: Mutex::new(HashMap::new()),
        gateway_secret: None,
        tenant_registry: None,
        bucket_path: None,
        html_dir: None,
        plugin_tokens: Mutex::new(HashMap::new()),
    };

    Router::new(Arc::new(state))
}

/// Build a Router with no model (for request validation tests that don't need inference).
fn build_app_no_model() -> Router {
    let state = AppState {
        backend: Arc::new(Mutex::new(BackendState {
            backend: None,
            current_model: None,
        })),
        configured_model: None,
        response_store: Mutex::new(HashMap::new()),
        gateway_secret: None,
        tenant_registry: None,
        bucket_path: None,
        html_dir: None,
        plugin_tokens: Mutex::new(HashMap::new()),
    };

    Router::new(Arc::new(state))
}

/// Helper: build a GET request.
fn get(uri: &str) -> Request<Full<Bytes>> {
    Request::builder()
        .method("GET")
        .uri(uri)
        .body(Full::new(Bytes::new()))
        .unwrap()
}

/// Helper: build a POST request with JSON body.
fn post_json(uri: &str, body: &Value) -> Request<Full<Bytes>> {
    let bytes = serde_json::to_vec(body).unwrap();
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(bytes)))
        .unwrap()
}

/// Helper: build a POST request with raw body bytes.
fn post_raw(uri: &str, body: &[u8]) -> Request<Full<Bytes>> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::copy_from_slice(body)))
        .unwrap()
}

// ===========================================================================
// J1. Protocol envelope & routing
// ===========================================================================

#[tokio::test]
async fn test_health_check() {
    let app = build_app_no_model();
    let resp = send_request(&app, get("/health")).await;
    let (status, body) = body_bytes(resp).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body.as_ref(), b"ok");
}

#[tokio::test]
async fn test_openapi_json() {
    let app = build_app_no_model();
    let resp = send_request(&app, get("/openapi.json")).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap();
    assert_eq!(ct, "application/json");
    let (_, json) = body_json(resp).await;
    assert!(json.is_object(), "OpenAPI spec should be a JSON object");
    assert!(
        json.get("openapi").is_some(),
        "should have openapi version field"
    );
}

#[tokio::test]
async fn test_models_list_format() {
    let app = build_app_with_model();
    let resp = send_request(&app, get("/v1/models")).await;
    let (status, json) = body_json(resp).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json.get("object").and_then(|v| v.as_str()), Some("list"));
    let data = json
        .get("data")
        .and_then(|v| v.as_array())
        .expect("data should be array");
    assert!(!data.is_empty(), "models list should not be empty");
    let model = &data[0];
    assert!(model.get("id").is_some(), "model should have id");
    assert!(model.get("object").is_some(), "model should have object");
    assert!(model.get("created").is_some(), "model should have created");
    assert!(
        model.get("owned_by").is_some(),
        "model should have owned_by"
    );
}

#[tokio::test]
async fn test_models_with_v1_prefix() {
    let app = build_app_with_model();
    let resp_v1 = send_request(&app, get("/v1/models")).await;
    let resp_bare = send_request(&app, get("/models")).await;
    let (s1, j1) = body_json(resp_v1).await;
    let (s2, j2) = body_json(resp_bare).await;
    assert_eq!(s1, StatusCode::OK);
    assert_eq!(s2, StatusCode::OK);
    // Both should have the same structure
    assert_eq!(
        j1.get("object").and_then(|v| v.as_str()),
        j2.get("object").and_then(|v| v.as_str()),
    );
}

#[tokio::test]
async fn test_unknown_path_404() {
    let app = build_app_no_model();
    let resp = send_request(&app, get("/v1/nonexistent")).await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_known_unimplemented_501() {
    let app = build_app_no_model();
    // GET /v1/responses is a known OpenAPI path (POST is implemented) but GET is not
    let resp = send_request(&app, get("/v1/responses")).await;
    let (status, json) = body_json(resp).await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    let error = json.get("error").expect("should have error object");
    assert_eq!(
        error.get("code").and_then(|v| v.as_str()),
        Some("not_implemented"),
    );
}

// ===========================================================================
// J2. POST /v1/responses — request validation
// ===========================================================================

#[tokio::test]
async fn test_invalid_json_400() {
    let app = build_app_no_model();
    let resp = send_request(&app, post_raw("/v1/responses", b"{not json")).await;
    let (status, json) = body_json(resp).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let error = json.get("error").expect("should have error");
    assert_eq!(
        error.get("code").and_then(|v| v.as_str()),
        Some("invalid_request"),
    );
    let msg = error.get("message").and_then(|v| v.as_str()).unwrap_or("");
    assert!(
        msg.contains("Invalid JSON"),
        "message should mention invalid JSON: {msg}"
    );
}

#[tokio::test]
async fn test_missing_input_400() {
    let app = build_app_no_model();
    let body = serde_json::json!({"model": "test"});
    let resp = send_request(&app, post_json("/v1/responses", &body)).await;
    let (status, _json) = body_json(resp).await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "missing input should be 400"
    );
}

#[tokio::test]
async fn test_string_input_accepted() {
    let app = build_app_with_model();
    let body = serde_json::json!({
        "model": model_path(),
        "input": "Hello",
        "max_output_tokens": 5,
    });
    let resp = send_request(&app, post_json("/v1/responses", &body)).await;
    let (status, json) = body_json(resp).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "string input should be accepted. body: {json}"
    );
}

#[tokio::test]
async fn test_message_array_input_accepted() {
    let app = build_app_with_model();
    let body = serde_json::json!({
        "model": model_path(),
        "input": [{"type": "message", "role": "user", "content": "Hi"}],
        "max_output_tokens": 5,
    });
    let resp = send_request(&app, post_json("/v1/responses", &body)).await;
    let (status, _json) = body_json(resp).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "message array input should be accepted"
    );
}

// ===========================================================================
// J3. POST /v1/responses — response envelope (non-streaming)
// ===========================================================================

#[tokio::test]
async fn test_response_envelope() {
    let app = build_app_with_model();
    let body = serde_json::json!({
        "model": model_path(),
        "input": "Say hello",
        "max_output_tokens": 10,
    });
    let resp = send_request(&app, post_json("/v1/responses", &body)).await;
    let (status, json) = body_json(resp).await;
    assert_eq!(status, StatusCode::OK);

    assert_eq!(
        json.get("object").and_then(|v| v.as_str()),
        Some("response")
    );
    let id = json
        .get("id")
        .and_then(|v| v.as_str())
        .expect("should have id");
    assert!(id.starts_with("resp_"), "id should start with resp_: {id}");
    assert!(
        json.get("created_at").and_then(|v| v.as_f64()).is_some(),
        "should have created_at"
    );
    assert_eq!(
        json.get("status").and_then(|v| v.as_str()),
        Some("completed")
    );
}

#[tokio::test]
async fn test_response_model_matches() {
    let app = build_app_with_model();
    let path = model_path();
    let body = serde_json::json!({
        "model": &path,
        "input": "Hi",
        "max_output_tokens": 5,
    });
    let resp = send_request(&app, post_json("/v1/responses", &body)).await;
    let (_, json) = body_json(resp).await;
    let model = json
        .get("model")
        .and_then(|v| v.as_str())
        .expect("should have model");
    assert_eq!(model, path);
}

#[tokio::test]
async fn test_response_output_shape() {
    let app = build_app_with_model();
    let body = serde_json::json!({
        "model": model_path(),
        "input": "Hello",
        "max_output_tokens": 200,
    });
    let resp = send_request(&app, post_json("/v1/responses", &body)).await;
    let (_, json) = body_json(resp).await;

    let output = json
        .get("output")
        .and_then(|v| v.as_array())
        .expect("output should be array");
    assert!(!output.is_empty(), "output should have at least one item");
    // Output may contain reasoning items before the message.
    // Every item must have a valid "type" field.
    for item in output {
        let item_type = item
            .get("type")
            .and_then(|v| v.as_str())
            .expect("item should have type");
        assert!(
            ["message", "reasoning", "function_call"].contains(&item_type),
            "unexpected item type: {item_type}",
        );
    }
    // Should have at least one message item.
    let msg = output
        .iter()
        .find(|item| item.get("type").and_then(|v| v.as_str()) == Some("message"));
    assert!(msg.is_some(), "output should contain a message item");
    let msg = msg.unwrap();
    assert_eq!(msg.get("role").and_then(|v| v.as_str()), Some("assistant"));
}

#[tokio::test]
async fn test_response_content_shape() {
    let app = build_app_with_model();
    let body = serde_json::json!({
        "model": model_path(),
        "input": "Say hi",
        "max_output_tokens": 200,
    });
    let resp = send_request(&app, post_json("/v1/responses", &body)).await;
    let (_, json) = body_json(resp).await;

    let output = json.get("output").and_then(|v| v.as_array()).unwrap();
    // Find the message item (may be preceded by reasoning).
    let msg = output
        .iter()
        .find(|item| item.get("type").and_then(|v| v.as_str()) == Some("message"))
        .expect("should have message item");
    let content = msg
        .get("content")
        .and_then(|v| v.as_array())
        .expect("content should be array");
    assert!(!content.is_empty(), "content should not be empty");
    assert_eq!(
        content[0].get("type").and_then(|v| v.as_str()),
        Some("output_text")
    );
    assert!(
        content[0].get("text").and_then(|v| v.as_str()).is_some(),
        "should have text field"
    );
}

#[tokio::test]
async fn test_response_usage() {
    let app = build_app_with_model();
    let body = serde_json::json!({
        "model": model_path(),
        "input": "Count to three",
        "max_output_tokens": 20,
    });
    let resp = send_request(&app, post_json("/v1/responses", &body)).await;
    let (_, json) = body_json(resp).await;

    let usage = json.get("usage").expect("should have usage");
    let input_tokens = usage
        .get("input_tokens")
        .and_then(|v| v.as_u64())
        .expect("input_tokens");
    let output_tokens = usage
        .get("output_tokens")
        .and_then(|v| v.as_u64())
        .expect("output_tokens");
    let total_tokens = usage
        .get("total_tokens")
        .and_then(|v| v.as_u64())
        .expect("total_tokens");
    assert!(input_tokens > 0, "input_tokens should be > 0");
    assert!(output_tokens > 0, "output_tokens should be > 0");
    assert_eq!(total_tokens, input_tokens + output_tokens);
}

#[tokio::test]
async fn test_response_content_type_header() {
    let app = build_app_with_model();
    let body = serde_json::json!({
        "model": model_path(),
        "input": "Hi",
        "max_output_tokens": 5,
    });
    let resp = send_request(&app, post_json("/v1/responses", &body)).await;
    let ct = resp
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap();
    assert!(
        ct.contains("application/json"),
        "Content-Type should be application/json: {ct}"
    );
}

// ===========================================================================
// J4. POST /v1/responses — streaming (SSE)
// ===========================================================================

/// Collect SSE events from a streaming response.
async fn collect_sse_events(resp: Response<Incoming>) -> Vec<(String, Value)> {
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&body);
    let mut events = Vec::new();

    for line in text.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            if data == "[DONE]" {
                continue;
            }
            if let Ok(json) = serde_json::from_str::<Value>(data) {
                let event_type = json
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                events.push((event_type, json));
            }
        }
    }

    events
}

fn streaming_request(input: &str) -> Value {
    serde_json::json!({
        "model": model_path(),
        "input": input,
        "stream": true,
        "max_output_tokens": 50,
    })
}

#[tokio::test]
async fn test_streaming_content_type() {
    let app = build_app_with_model();
    let resp = send_request(&app, post_json("/v1/responses", &streaming_request("Hi"))).await;
    let ct = resp
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap();
    assert!(
        ct.contains("text/event-stream"),
        "streaming Content-Type: {ct}"
    );
}

#[tokio::test]
async fn test_streaming_cache_headers() {
    let app = build_app_with_model();
    let resp = send_request(&app, post_json("/v1/responses", &streaming_request("Hi"))).await;
    let cc = resp
        .headers()
        .get("cache-control")
        .map(|v| v.to_str().unwrap().to_string());
    assert_eq!(
        cc.as_deref(),
        Some("no-cache"),
        "Cache-Control should be no-cache"
    );
}

#[tokio::test]
async fn test_streaming_sequence() {
    let app = build_app_with_model();
    let resp = send_request(
        &app,
        post_json("/v1/responses", &streaming_request("Hello")),
    )
    .await;
    let events = collect_sse_events(resp).await;

    assert!(!events.is_empty(), "should have events");
    // First event should be response.created
    assert_eq!(events[0].0, "response.created", "first event type");
    // Last event should be a terminal event (completed or incomplete)
    let last_type = &events.last().unwrap().0;
    assert!(
        last_type == "response.completed" || last_type == "response.incomplete",
        "last event should be terminal: {last_type}",
    );
    // Should have at least one delta (output_text or reasoning)
    let has_delta = events.iter().any(|(t, _)| t.ends_with(".delta"));
    assert!(has_delta, "should have at least one delta event");
    // Should have a done event (output_text or reasoning)
    let has_done = events
        .iter()
        .any(|(t, _)| t.ends_with(".done") && t.starts_with("response."));
    assert!(has_done, "should have a done event");
}

#[tokio::test]
async fn test_streaming_created_event() {
    let app = build_app_with_model();
    let resp = send_request(
        &app,
        post_json("/v1/responses", &streaming_request("Hello")),
    )
    .await;
    let events = collect_sse_events(resp).await;

    let (event_type, data) = &events[0];
    assert_eq!(event_type, "response.created");
    let response = data
        .get("response")
        .expect("created event should have response");
    assert_eq!(
        response.get("status").and_then(|v| v.as_str()),
        Some("in_progress")
    );
}

#[tokio::test]
async fn test_streaming_delta_events() {
    let app = build_app_with_model();
    let resp = send_request(
        &app,
        post_json("/v1/responses", &streaming_request("Hello")),
    )
    .await;
    let events = collect_sse_events(resp).await;

    let deltas: Vec<_> = events
        .iter()
        .filter(|(t, _)| t.ends_with(".delta"))
        .collect();
    assert!(!deltas.is_empty(), "should have delta events");
    for (_, data) in &deltas {
        assert!(
            data.get("delta").is_some(),
            "delta event should have delta field"
        );
    }
}

#[tokio::test]
async fn test_streaming_done_event() {
    let app = build_app_with_model();
    let resp = send_request(
        &app,
        post_json("/v1/responses", &streaming_request("Hello")),
    )
    .await;
    let events = collect_sse_events(resp).await;

    let done = events
        .iter()
        .find(|(t, _)| t.ends_with(".done") && t.starts_with("response."));
    assert!(done.is_some(), "should have done event");
    let (_, data) = done.unwrap();
    assert!(
        data.get("text").is_some(),
        "done event should have text field"
    );
}

#[tokio::test]
async fn test_streaming_completed_event() {
    let app = build_app_with_model();
    let resp = send_request(
        &app,
        post_json("/v1/responses", &streaming_request("Hello")),
    )
    .await;
    let events = collect_sse_events(resp).await;

    // Terminal event is either response.completed or response.incomplete
    let terminal = events
        .iter()
        .find(|(t, _)| t == "response.completed" || t == "response.incomplete");
    assert!(terminal.is_some(), "should have a terminal event");
    let (event_type, data) = terminal.unwrap();
    let response = data
        .get("response")
        .expect("terminal event should have response");
    let expected_status = if event_type == "response.completed" {
        "completed"
    } else {
        "incomplete"
    };
    assert_eq!(
        response.get("status").and_then(|v| v.as_str()),
        Some(expected_status)
    );
    let usage = response
        .get("usage")
        .expect("terminal response should have usage");
    assert!(
        usage
            .get("input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
            > 0
    );
    assert!(
        usage
            .get("output_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
            > 0
    );
}

#[tokio::test]
async fn test_streaming_sequence_numbers() {
    let app = build_app_with_model();
    let resp = send_request(
        &app,
        post_json("/v1/responses", &streaming_request("Hello")),
    )
    .await;
    let events = collect_sse_events(resp).await;

    let seq_nums: Vec<u64> = events
        .iter()
        .filter_map(|(_, data)| data.get("sequence_number").and_then(|v| v.as_u64()))
        .collect();
    assert!(!seq_nums.is_empty(), "should have sequence numbers");
    for window in seq_nums.windows(2) {
        assert!(
            window[1] > window[0],
            "sequence numbers should be monotonically increasing: {} -> {}",
            window[0],
            window[1],
        );
    }
}

// ===========================================================================
// J5. Error response format
// ===========================================================================

#[tokio::test]
async fn test_error_json_structure() {
    let app = build_app_no_model();
    let resp = send_request(&app, post_raw("/v1/responses", b"invalid")).await;
    let (status, json) = body_json(resp).await;
    assert!(status.is_client_error(), "should be 4xx error");
    let error = json.get("error").expect("should have error object");
    assert!(
        error.get("code").and_then(|v| v.as_str()).is_some(),
        "error should have code"
    );
    assert!(
        error.get("message").and_then(|v| v.as_str()).is_some(),
        "error should have message"
    );
}

#[tokio::test]
async fn test_error_content_type() {
    let app = build_app_no_model();
    let resp = send_request(&app, post_raw("/v1/responses", b"invalid")).await;
    let ct = resp
        .headers()
        .get("content-type")
        .map(|v| v.to_str().unwrap().to_string());
    assert!(
        ct.as_deref()
            .map_or(false, |c| c.contains("application/json")),
        "error Content-Type should be application/json: {:?}",
        ct,
    );
}

#[tokio::test]
async fn test_400_invalid_body() {
    let app = build_app_no_model();
    let resp = send_request(&app, post_raw("/v1/responses", b"not json at all")).await;
    let (status, json) = body_json(resp).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        json.get("error")
            .and_then(|e| e.get("code"))
            .and_then(|v| v.as_str()),
        Some("invalid_request"),
    );
}

// ===========================================================================
// J6. Structured input round-trip
// ===========================================================================

#[tokio::test]
async fn test_structured_user_message_input() {
    let app = build_app_with_model();
    let body = serde_json::json!({
        "model": model_path(),
        "input": [{"type": "message", "role": "user", "content": "Say hello"}],
        "max_output_tokens": 200,
    });
    let resp = send_request(&app, post_json("/v1/responses", &body)).await;
    let (status, json) = body_json(resp).await;
    assert_eq!(status, StatusCode::OK, "structured input should succeed");
    assert_eq!(
        json.get("object").and_then(|v| v.as_str()),
        Some("response")
    );
    let output = json
        .get("output")
        .and_then(|v| v.as_array())
        .expect("should have output");
    assert!(!output.is_empty(), "output should not be empty");
}

#[tokio::test]
async fn test_system_and_user_message_input() {
    let app = build_app_with_model();
    let body = serde_json::json!({
        "model": model_path(),
        "input": [
            {"type": "message", "role": "system", "content": "You are a helpful assistant."},
            {"type": "message", "role": "user", "content": "Hello"}
        ],
        "max_output_tokens": 200,
    });
    let resp = send_request(&app, post_json("/v1/responses", &body)).await;
    let (status, json) = body_json(resp).await;
    assert_eq!(status, StatusCode::OK, "system+user input should succeed");
    let output = json
        .get("output")
        .and_then(|v| v.as_array())
        .expect("should have output");
    assert!(!output.is_empty(), "should have generated output");
}

#[tokio::test]
async fn test_developer_message_input() {
    let app = build_app_with_model();
    let body = serde_json::json!({
        "model": model_path(),
        "input": [
            {"type": "message", "role": "developer", "content": "Be brief."},
            {"type": "message", "role": "user", "content": "Hi"}
        ],
        "max_output_tokens": 200,
    });
    let resp = send_request(&app, post_json("/v1/responses", &body)).await;
    let (status, _) = body_json(resp).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "developer message should be accepted"
    );
}

// ===========================================================================
// J7. Conversation chaining (previous_response_id)
// ===========================================================================

#[tokio::test]
async fn test_conversation_chaining() {
    let app = build_app_with_model();

    // First request.
    let body1 = serde_json::json!({
        "model": model_path(),
        "input": "Hello, my name is Alice.",
        "max_output_tokens": 50,
    });
    let resp1 = send_request(&app, post_json("/v1/responses", &body1)).await;
    let (status1, json1) = body_json(resp1).await;
    assert_eq!(status1, StatusCode::OK, "first request should succeed");
    let response_id = json1
        .get("id")
        .and_then(|v| v.as_str())
        .expect("should have id");

    // Chained request referencing the first.
    let body2 = serde_json::json!({
        "model": model_path(),
        "input": "What is my name?",
        "previous_response_id": response_id,
        "max_output_tokens": 50,
    });
    let resp2 = send_request(&app, post_json("/v1/responses", &body2)).await;
    let (status2, json2) = body_json(resp2).await;
    assert_eq!(status2, StatusCode::OK, "chained request should succeed");
    assert_eq!(
        json2.get("previous_response_id").and_then(|v| v.as_str()),
        Some(response_id),
        "should reference previous response",
    );
}

// ===========================================================================
// J8. Multi-item output (reasoning + message)
// ===========================================================================

#[tokio::test]
async fn test_multi_item_output() {
    let app = build_app_with_model();
    let body = serde_json::json!({
        "model": model_path(),
        "input": "What is 2+2?",
        "max_output_tokens": 200,
    });
    let resp = send_request(&app, post_json("/v1/responses", &body)).await;
    let (status, json) = body_json(resp).await;
    assert_eq!(status, StatusCode::OK);

    let output = json
        .get("output")
        .and_then(|v| v.as_array())
        .expect("should have output");
    // Qwen3 with think mode typically produces reasoning + message.
    // Verify that each item has a valid type.
    let types: Vec<&str> = output
        .iter()
        .filter_map(|item| item.get("type").and_then(|v| v.as_str()))
        .collect();
    assert!(!types.is_empty(), "output items should have types");
    for t in &types {
        assert!(
            ["reasoning", "message", "function_call"].contains(t),
            "unexpected output item type: {t}",
        );
    }
    // At least one message or reasoning item should be present.
    assert!(
        types.contains(&"message") || types.contains(&"reasoning"),
        "output should contain at least a message or reasoning item",
    );
}
