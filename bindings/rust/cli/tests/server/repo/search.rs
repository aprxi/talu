//! Integration tests for `GET /v1/repo/search`.

use crate::server::common::*;

fn repo_config() -> ServerConfig {
    let mut config = ServerConfig::new();
    config.no_bucket = true;
    config
}

/// Assert the response is a valid search response (200 with results array) or
/// a backend error (500 with error envelope). The key invariant is: query
/// validation never rejects (no 400).
fn assert_search_accepted(resp: &HttpResponse) {
    assert_ne!(resp.status, 400, "should accept query: {}", resp.body);
    let json = resp.json();
    match resp.status {
        200 => {
            assert!(json["results"].is_array(), "200 should have results array");
        }
        _ => {
            // Network/backend error → 500 with error envelope.
            assert!(
                json["error"].is_object(),
                "non-200 response should have error object: {}",
                resp.body
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Empty / missing query → trending browse
// ---------------------------------------------------------------------------

/// Missing 'query' parameter defaults to empty string and searches for
/// trending models (no longer a 400 error).
#[test]
fn search_missing_query_returns_trending() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(ctx.addr(), "/v1/repo/search");
    assert_search_accepted(&resp);
}

/// Empty 'query' parameter returns trending models.
#[test]
fn search_empty_query_returns_trending() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(ctx.addr(), "/v1/repo/search?query=");
    assert_search_accepted(&resp);
}

/// Responses always have application/json content type.
#[test]
fn search_response_has_json_content_type() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(ctx.addr(), "/v1/repo/search?query=");
    assert_eq!(resp.header("content-type"), Some("application/json"));
}

/// Query with only whitespace is accepted (passed through to search backend).
#[test]
fn search_whitespace_query_is_accepted() {
    let ctx = ServerTestContext::new(repo_config());
    // URL-encoded spaces: "query=%20%20"
    let resp = get(ctx.addr(), "/v1/repo/search?query=%20%20");
    assert_search_accepted(&resp);
}

// ---------------------------------------------------------------------------
// Optional params accepted alongside query
// ---------------------------------------------------------------------------

/// Token query param is accepted without error.
#[test]
fn search_accepts_token_param() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(ctx.addr(), "/v1/repo/search?query=test&token=hf_test123");
    assert_search_accepted(&resp);
}

/// Endpoint_url query param is accepted without parsing error.
#[test]
fn search_accepts_endpoint_url_param() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(
        ctx.addr(),
        "/v1/repo/search?query=test&endpoint_url=https://mirror.example.com",
    );
    // endpoint_url points to a non-existent host; expect network error, not
    // a parsing/validation error.
    assert_ne!(resp.status, 400, "should not reject params: {}", resp.body);
    let _json = resp.json(); // Must be valid JSON regardless.
}

/// Sort parameter is accepted.
#[test]
fn search_accepts_sort_param() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(ctx.addr(), "/v1/repo/search?query=test&sort=downloads");
    assert_search_accepted(&resp);
}

/// Direction parameter is accepted.
#[test]
fn search_accepts_direction_param() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(ctx.addr(), "/v1/repo/search?query=test&direction=ascending");
    assert_search_accepted(&resp);
}

/// Filter parameter (HF pipeline tag) is accepted.
#[test]
fn search_accepts_filter_param() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(
        ctx.addr(),
        "/v1/repo/search?query=test&filter=text-generation",
    );
    assert_search_accepted(&resp);
}

/// Library parameter is accepted.
#[test]
fn search_accepts_library_param() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(ctx.addr(), "/v1/repo/search?query=test&library=safetensors");
    assert_search_accepted(&resp);
}

// ---------------------------------------------------------------------------
// Bare path variant
// ---------------------------------------------------------------------------

/// GET /repo/search with empty query works (bare path routing).
#[test]
fn search_bare_path_accepted() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(ctx.addr(), "/repo/search?query=");
    assert_search_accepted(&resp);
}
