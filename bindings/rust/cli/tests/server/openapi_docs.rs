//! Integration tests for the trimmed inference-server docs surface.

use crate::server::common::*;

#[test]
fn health_endpoint_returns_ok() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let resp = get(ctx.addr(), "/health");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(resp.body, "ok");
}

#[test]
fn root_openapi_served_and_contains_paths() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let resp = get(ctx.addr(), "/openapi.json");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let content_type = resp.header("content-type").unwrap_or("");
    assert!(
        content_type.contains("application/json"),
        "content-type={content_type}"
    );

    let json = resp.json();
    let paths = json["paths"].as_object().expect("paths object");
    assert!(
        paths.contains_key("/v1/models"),
        "missing /v1/models in root spec"
    );
    assert!(
        paths.contains_key("/v1/chat/completions"),
        "missing /v1/chat/completions in root spec"
    );
    assert!(
        paths.contains_key("/v1/responses"),
        "missing /v1/responses in root spec"
    );
    assert!(
        paths.keys().any(|k| k.starts_with("/v1/tokenizer")),
        "missing /v1/tokenizer* endpoints in root spec"
    );
    assert!(
        paths.keys().all(|k| k.starts_with("/v1/")),
        "root spec should expose only /v1/* endpoints"
    );
}

#[test]
fn docs_hub_lists_inference_sections() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let resp = get(ctx.addr(), "/docs");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let content_type = resp.header("content-type").unwrap_or("");
    assert!(
        content_type.contains("text/html"),
        "content-type={content_type}"
    );

    for link in [
        "/docs/chat",
        "/docs/responses",
        "/docs/models",
        "/docs/tokenizer",
    ] {
        assert!(resp.body.contains(link), "missing link: {link}");
    }
}

#[test]
fn scoped_docs_pages_point_to_scoped_specs() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    for (path, expected_spec) in [
        ("/docs/chat", "/openapi/chat.json"),
        ("/docs/responses", "/openapi/responses.json"),
        ("/docs/models", "/openapi/models.json"),
        ("/docs/tokenizer", "/openapi/tokenizer.json"),
    ] {
        let resp = get(ctx.addr(), path);
        assert_eq!(resp.status, 200, "path={path} body={}", resp.body);
        let content_type = resp.header("content-type").unwrap_or("");
        assert!(
            content_type.contains("text/html"),
            "path={path} content-type={content_type}"
        );
        assert!(
            resp.body.contains(expected_spec),
            "path={path} expected spec URL {expected_spec}"
        );
        assert!(
            resp.body.contains(r#"href="/docs""#),
            "path={path} expected docs-home link in header"
        );
    }
}

#[test]
fn scoped_openapi_specs_are_prefix_scoped() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    for (path, prefixes) in [
        ("/openapi/chat.json", vec!["/v1/chat/"]),
        ("/openapi/responses.json", vec!["/v1/responses"]),
        ("/openapi/models.json", vec!["/v1/models"]),
        ("/openapi/tokenizer.json", vec!["/v1/tokenizer"]),
    ] {
        let resp = get(ctx.addr(), path);
        assert_eq!(resp.status, 200, "path={path} body={}", resp.body);

        let json = resp.json();
        let paths = json["paths"].as_object().expect("paths object");
        assert!(!paths.is_empty(), "path={path} should not be empty");
        assert!(
            paths
                .keys()
                .all(|k| prefixes.iter().any(|prefix| k.starts_with(prefix))),
            "path={path} contains endpoint outside expected prefixes"
        );
    }
}
