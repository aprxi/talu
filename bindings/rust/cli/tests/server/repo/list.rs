//! Integration tests for `GET /v1/repo/models`.

use crate::server::common::*;

fn repo_config() -> ServerConfig {
    let mut config = ServerConfig::new();
    config.no_bucket = true;
    config
}

// ---------------------------------------------------------------------------
// Response shape
// ---------------------------------------------------------------------------

/// GET /v1/repo/models returns 200 with correct top-level JSON structure.
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

/// Each model entry in the list has the expected fields with correct types.
///
/// NOTE: If no models are cached on the test machine, the models array is
/// empty and this loop body never executes. The test still passes because
/// the structure is valid — an empty list is correct. This is inherent to
/// testing a cache-dependent endpoint without fixtures.
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

        // Type checks
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

        // source must be "hub" or "managed"
        let source = model["source"]
            .as_str()
            .unwrap_or_else(|| panic!("model[{i}].source should be string"));
        assert!(
            source == "hub" || source == "managed",
            "model[{i}].source should be 'hub' or 'managed', got: {source}"
        );

        // architecture and quant_scheme are optional but typed when present
        if let Some(arch) = obj.get("architecture") {
            assert!(
                arch.is_string(),
                "model[{i}].architecture should be string"
            );
        }
        if let Some(quant) = obj.get("quant_scheme") {
            assert!(
                quant.is_string(),
                "model[{i}].quant_scheme should be string"
            );
        }
    }
}

/// total_size_bytes is non-negative (u64 serialized).
#[test]
fn list_total_size_is_non_negative() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(ctx.addr(), "/v1/repo/models");
    assert_eq!(resp.status, 200);

    let json = resp.json();
    let total = json["total_size_bytes"]
        .as_u64()
        .expect("total_size_bytes should be a u64");
    // Just verifying it parses as u64 (not negative, not float).
    let _ = total;
}

/// Response has application/json content type.
#[test]
fn list_has_json_content_type() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(ctx.addr(), "/v1/repo/models");
    assert_eq!(resp.status, 200);
    assert_eq!(resp.header("content-type"), Some("application/json"));
}

// ---------------------------------------------------------------------------
// OpenAPI registration
// ---------------------------------------------------------------------------

/// The repo endpoints are registered in the OpenAPI spec.
#[test]
fn repo_endpoints_in_openapi() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(ctx.addr(), "/openapi.json");
    assert_eq!(resp.status, 200);

    let json = resp.json();
    let paths = json["paths"].as_object().expect("should have paths object");

    assert!(
        paths.contains_key("/v1/repo/models"),
        "OpenAPI spec should contain /v1/repo/models"
    );
    assert!(
        paths.contains_key("/v1/repo/search"),
        "OpenAPI spec should contain /v1/repo/search"
    );
    assert!(
        paths.contains_key("/v1/repo/models/{model_id}"),
        "OpenAPI spec should contain /v1/repo/models/{{model_id}}"
    );

    // Verify HTTP methods are registered on the paths
    let models_path = &paths["/v1/repo/models"];
    assert!(
        models_path.get("get").is_some(),
        "/v1/repo/models should have GET"
    );
    assert!(
        models_path.get("post").is_some(),
        "/v1/repo/models should have POST"
    );

    let search_path = &paths["/v1/repo/search"];
    assert!(
        search_path.get("get").is_some(),
        "/v1/repo/search should have GET"
    );

    let delete_path = &paths["/v1/repo/models/{model_id}"];
    assert!(
        delete_path.get("delete").is_some(),
        "/v1/repo/models/{{model_id}} should have DELETE"
    );

    assert!(
        paths.contains_key("/v1/repo/models/{model_id}/files"),
        "OpenAPI spec should contain /v1/repo/models/{{model_id}}/files"
    );
    let files_path = &paths["/v1/repo/models/{model_id}/files"];
    assert!(
        files_path.get("get").is_some(),
        "/v1/repo/models/{{model_id}}/files should have GET"
    );
}

// ---------------------------------------------------------------------------
// Source filter
// ---------------------------------------------------------------------------

/// Source filter with valid value returns valid structure.
#[test]
fn list_accepts_source_filter() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(ctx.addr(), "/v1/repo/models?source=hub");
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert!(json["models"].is_array());
    assert!(json["total_size_bytes"].is_number());

    // All returned models should have source=hub (if any)
    for model in json["models"].as_array().unwrap() {
        assert_eq!(model["source"], "hub", "source filter should work");
    }
}

/// Source filter with managed value also works.
#[test]
fn list_source_filter_managed() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(ctx.addr(), "/v1/repo/models?source=managed");
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    for model in resp.json()["models"].as_array().unwrap() {
        assert_eq!(model["source"], "managed");
    }
}

/// Invalid source filter is ignored (returns all models).
#[test]
fn list_source_filter_invalid_ignored() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(ctx.addr(), "/v1/repo/models?source=invalid");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert!(resp.json()["models"].is_array());
}

// ---------------------------------------------------------------------------
// Pinned field
// ---------------------------------------------------------------------------

/// Each model entry has a boolean `pinned` field.
#[test]
fn list_model_entries_have_pinned_field() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(ctx.addr(), "/v1/repo/models");
    assert_eq!(resp.status, 200);

    for (i, model) in resp.json()["models"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        assert!(
            model["pinned"].is_boolean(),
            "model[{i}].pinned should be boolean"
        );
    }
}

// ---------------------------------------------------------------------------
// Bare path variant (without /v1 prefix)
// ---------------------------------------------------------------------------

/// GET /repo/models works (without /v1 prefix).
#[test]
fn list_bare_path() {
    let ctx = ServerTestContext::new(repo_config());
    let resp = get(ctx.addr(), "/repo/models");
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert!(json["models"].is_array());
    assert!(json["total_size_bytes"].is_number());
}
