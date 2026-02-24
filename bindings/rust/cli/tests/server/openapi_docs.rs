//! Integration tests for scoped OpenAPI and docs entrypoints outside DB.

use crate::server::common::*;

#[test]
fn docs_hub_lists_non_db_sections() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let resp = get(ctx.addr(), "/docs");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let content_type = resp.header("content-type").unwrap_or("");
    assert!(
        content_type.contains("text/html"),
        "content-type={content_type}"
    );

    for link in [
        "/docs/ai",
        "/docs/chat",
        "/docs/files",
        "/docs/repo",
        "/docs/search",
        "/docs/tags",
        "/docs/settings",
        "/docs/plugins",
        "/docs/code",
        "/docs/db/tables",
        "/docs/db/vectors",
        "/docs/db/kv",
        "/docs/db/blobs",
        "/docs/db/sql",
        "/docs/db/ops",
    ] {
        assert!(resp.body.contains(link), "missing link: {link}");
    }
}

#[test]
fn scoped_docs_pages_point_to_scoped_specs() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    for (path, expected_spec) in [
        ("/docs/ai", "/openapi/ai.json"),
        ("/docs/chat", "/openapi/chat.json"),
        ("/docs/files", "/openapi/files.json"),
        ("/docs/repo", "/openapi/repo.json"),
        ("/docs/search", "/openapi/search.json"),
        ("/docs/tags", "/openapi/tags.json"),
        ("/docs/settings", "/openapi/settings.json"),
        ("/docs/plugins", "/openapi/plugins.json"),
        ("/docs/code", "/openapi/code.json"),
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
        ("/openapi/ai.json", vec!["/v1/models", "/v1/responses"]),
        ("/openapi/chat.json", vec!["/v1/chat/sessions"]),
        ("/openapi/files.json", vec!["/v1/files", "/v1/file"]),
        ("/openapi/repo.json", vec!["/v1/repo"]),
        ("/openapi/search.json", vec!["/v1/search"]),
        ("/openapi/tags.json", vec!["/v1/tags"]),
        ("/openapi/settings.json", vec!["/v1/settings"]),
        ("/openapi/plugins.json", vec!["/v1/plugins", "/v1/proxy"]),
        ("/openapi/code.json", vec!["/v1/code"]),
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
