//! Integration tests for the restored repository HTTP surface.

use crate::server::common::*;
use serde_json::json;
use std::io::Write;

fn repo_config() -> ServerConfig {
    ServerConfig::new()
}

#[test]
fn list_returns_correct_structure() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(ctx.addr(), "/v1/repo/models");
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let obj = json.as_object().expect("response should be a JSON object");
    assert!(obj.contains_key("models"), "missing 'models' field");
    assert!(
        obj.contains_key("total_size_bytes"),
        "missing 'total_size_bytes' field"
    );
    assert!(json["models"].is_array(), "'models' should be an array");
    assert!(
        json["total_size_bytes"].is_number(),
        "'total_size_bytes' should be a number"
    );
}

#[test]
fn list_model_entries_have_expected_fields() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(ctx.addr(), "/v1/repo/models");
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let models = json["models"].as_array().expect("models should be array");

    for (i, model) in models.iter().enumerate() {
        let obj = model
            .as_object()
            .unwrap_or_else(|| panic!("model[{i}] should be an object"));
        assert!(obj.contains_key("id"), "model[{i}] missing 'id'");
        assert!(obj.contains_key("path"), "model[{i}] missing 'path'");
        assert!(obj.contains_key("source"), "model[{i}] missing 'source'");
        assert!(
            obj.contains_key("size_bytes"),
            "model[{i}] missing 'size_bytes'"
        );
        assert!(obj.contains_key("mtime"), "model[{i}] missing 'mtime'");

        assert!(model["id"].is_string(), "model[{i}].id should be string");
        assert!(
            model["path"].is_string(),
            "model[{i}].path should be string"
        );
        assert!(
            model["size_bytes"].is_number(),
            "model[{i}].size_bytes should be number"
        );
        assert!(
            model["mtime"].is_number(),
            "model[{i}].mtime should be number"
        );

        let source = model["source"]
            .as_str()
            .unwrap_or_else(|| panic!("model[{i}].source should be string"));
        assert!(
            source == "hub" || source == "managed",
            "model[{i}].source should be 'hub' or 'managed', got: {source}"
        );

        if let Some(arch) = obj.get("architecture") {
            assert!(arch.is_string(), "model[{i}].architecture should be string");
        }
        if let Some(quant) = obj.get("quant_scheme") {
            assert!(
                quant.is_string(),
                "model[{i}].quant_scheme should be string"
            );
        }
        if let Some(source_model_id) = obj.get("source_model_id") {
            assert!(
                source_model_id.is_string(),
                "model[{i}].source_model_id should be string"
            );
        }
    }
}

#[test]
fn list_has_json_content_type() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(ctx.addr(), "/v1/repo/models");
    assert_eq!(resp.status, 200);
    assert_eq!(resp.header("content-type"), Some("application/json"));
}

#[test]
fn list_accepts_source_filter() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(ctx.addr(), "/v1/repo/models?source=hub");
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    for model in resp.json()["models"].as_array().unwrap() {
        assert_eq!(model["source"], "hub", "source filter should work");
    }
}

#[test]
fn list_source_filter_invalid_ignored() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(ctx.addr(), "/v1/repo/models?source=invalid");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert!(resp.json()["models"].is_array());
}

#[test]
fn list_bare_path() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(ctx.addr(), "/repo/models");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert!(resp.json()["models"].is_array());
}

fn assert_search_accepted(resp: &HttpResponse) {
    assert_ne!(resp.status, 400, "should accept query: {}", resp.body);
    let json = resp.json();
    match resp.status {
        200 => {
            assert!(json["results"].is_array(), "200 should have results array");
        }
        _ => {
            assert!(
                json["error"].is_object(),
                "non-200 response should have error object: {}",
                resp.body
            );
        }
    }
}

#[test]
fn search_missing_query_returns_trending() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(ctx.addr(), "/v1/repo/search");
    assert_search_accepted(&resp);
}

#[test]
fn search_accepts_optional_params() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(
        ctx.addr(),
        "/v1/repo/search?query=test&sort=downloads&direction=ascending&filter=text-generation&library=safetensors&limit=5",
    );
    assert_search_accepted(&resp);
}

#[test]
fn search_bare_path_accepted() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(ctx.addr(), "/repo/search?query=");
    assert_search_accepted(&resp);
}

#[test]
fn fetch_invalid_json_returns_400() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/repo/models",
        &[("Content-Type", "application/json")],
        Some("not json {{{"),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_request");
}

#[test]
fn fetch_empty_model_id_returns_400() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = post_json(ctx.addr(), "/v1/repo/models", &json!({"model_id": ""}));
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_request");
}

#[test]
fn fetch_accepts_endpoint_url_and_skip_weights() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = post_json(
        ctx.addr(),
        "/v1/repo/models",
        &json!({
            "model_id": "nonexistent-org/nonexistent-model",
            "endpoint_url": "https://custom-mirror.example.com",
            "skip_weights": true
        }),
    );
    assert_ne!(
        resp.json()["error"]["code"],
        "invalid_request",
        "request fields should be accepted"
    );
}

#[test]
fn fetch_stream_returns_event_stream_content_type() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/repo/models",
        &[
            ("Content-Type", "application/json"),
            ("Accept", "text/event-stream"),
        ],
        Some(r#"{"model_id":"nonexistent-org/nonexistent-model"}"#),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(resp.header("content-type"), Some("text/event-stream"));
}

#[test]
fn fetch_stream_error_event_for_nonexistent_model() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/repo/models",
        &[
            ("Content-Type", "application/json"),
            ("Accept", "text/event-stream"),
        ],
        Some(r#"{"model_id":"nonexistent-org/nonexistent-model"}"#),
    );
    assert_eq!(resp.status, 200);
    assert!(
        resp.body.contains("data: "),
        "SSE body should contain data lines"
    );
    assert!(
        resp.body.contains("\"event\":\"error\"") || resp.body.contains("\"event\": \"error\""),
        "SSE body should contain error event, got: {}",
        resp.body
    );
}

#[test]
fn fetch_stream_client_disconnect_does_not_crash_server() {
    let ctx = ServerTestContext::new(repo_config());

    {
        let body = r#"{"model_id":"nonexistent-org/some-model"}"#;
        let request = format!(
            "POST /v1/repo/models HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nAccept: text/event-stream\r\nContent-Length: {}\r\n\r\n{}",
            ctx.addr(),
            body.len(),
            body
        );
        let mut stream = std::net::TcpStream::connect_timeout(
            &ctx.addr().into(),
            std::time::Duration::from_secs(5),
        )
        .expect("connect");
        stream
            .set_write_timeout(Some(std::time::Duration::from_secs(5)))
            .expect("set write timeout");
        stream.write_all(request.as_bytes()).expect("write request");
        stream.flush().expect("flush");
    }

    std::thread::sleep(std::time::Duration::from_millis(200));

    let resp = get(ctx.addr(), "/v1/repo/models");
    assert_eq!(resp.status, 200, "server should remain alive");
}

#[test]
fn delete_returns_json_shape() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = delete(
        ctx.addr(),
        "/v1/repo/models/nonexistent-org/nonexistent-model",
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert!(resp.json()["deleted"].is_boolean());
    assert_eq!(resp.json()["model_id"], "nonexistent-org/nonexistent-model");
}

#[test]
fn delete_bare_path() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = delete(ctx.addr(), "/repo/models/nonexistent-org/nonexistent-model");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
}

#[test]
fn files_has_json_content_type() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(
        ctx.addr(),
        "/v1/repo/models/nonexistent-org/nonexistent-model/files",
    );
    assert_eq!(resp.header("content-type"), Some("application/json"));
}

#[test]
fn files_response_structure_or_error() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(
        ctx.addr(),
        "/v1/repo/models/nonexistent-org/nonexistent-model/files?token=hf_test123",
    );
    let json = resp.json();
    if resp.status == 200 {
        assert!(json["model_id"].is_string());
        assert!(json["files"].is_array());
    } else {
        assert_eq!(resp.status, 500, "body: {}", resp.body);
        assert!(json["error"]["code"].is_string());
    }
}

#[test]
fn files_percent_encoded_model_id() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(
        ctx.addr(),
        "/v1/repo/models/meta-llama%2FLlama-3.2-1B/files",
    );
    let json = resp.json();
    if resp.status == 200 {
        assert_eq!(json["model_id"], "meta-llama/Llama-3.2-1B");
    }
}

#[test]
fn files_bare_path() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(
        ctx.addr(),
        "/repo/models/nonexistent-org/nonexistent-model/files",
    );
    assert_eq!(resp.header("content-type"), Some("application/json"));
}

#[test]
fn repo_endpoints_in_root_openapi() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(ctx.addr(), "/openapi.json");
    assert_eq!(resp.status, 200);

    let json = resp.json();
    let paths = json["paths"].as_object().expect("paths object");
    assert!(paths.contains_key("/v1/repo/models"));
    assert!(paths.contains_key("/v1/repo/search"));
    assert!(paths.contains_key("/v1/repo/models/{model_id}"));
    assert!(paths.contains_key("/v1/repo/models/{model_id}/files"));
}

#[test]
fn repo_scoped_openapi_contains_only_repo_paths() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(ctx.addr(), "/openapi/repo.json");
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let paths = json["paths"].as_object().expect("paths object");
    assert!(!paths.is_empty(), "repo scoped spec should not be empty");
    assert!(paths.keys().all(|k| k.starts_with("/v1/repo")));
}

#[test]
fn repo_docs_page_points_to_repo_spec() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(ctx.addr(), "/docs/repo");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert!(
        resp.body.contains("/openapi/repo.json"),
        "docs page should point to repo scoped spec"
    );
}
