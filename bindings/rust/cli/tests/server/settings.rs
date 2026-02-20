//! Integration tests for `/v1/settings` endpoints.

use crate::server::common::*;
use serde_json::json;
use tempfile::TempDir;

/// ServerConfig with bucket set for settings tests.
fn settings_config(bucket: &std::path::Path) -> ServerConfig {
    let mut config = ServerConfig::new();
    config.bucket = Some(bucket.to_path_buf());
    config
}

/// ServerConfig with --no-bucket (storage disabled).
fn no_bucket_config() -> ServerConfig {
    let mut config = ServerConfig::new();
    config.no_bucket = true;
    config
}

// ---------------------------------------------------------------------------
// GET /v1/settings — defaults and error handling
// ---------------------------------------------------------------------------

/// GET with an empty bucket returns default settings (all null/empty).
#[test]
fn get_defaults_with_empty_bucket() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(settings_config(temp.path()));

    let resp = get(ctx.addr(), "/v1/settings");
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert!(json["model"].is_null(), "model should be null by default");
    assert!(
        json["system_prompt"].is_null(),
        "system_prompt should be null by default"
    );
    assert!(
        json["max_output_tokens"].is_null(),
        "max_output_tokens should be null by default"
    );
    assert!(
        json["context_length"].is_null(),
        "context_length should be null by default"
    );
    assert!(
        json["available_models"].is_array(),
        "available_models should be an array"
    );
}

/// GET without a bucket returns 404 with no_storage error.
#[test]
fn get_returns_404_without_bucket() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let resp = get(ctx.addr(), "/v1/settings");
    assert_eq!(resp.status, 404, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(json["error"]["code"], "no_storage");
}

// ---------------------------------------------------------------------------
// PATCH /v1/settings — individual fields
// ---------------------------------------------------------------------------

/// PATCH without a bucket returns 404.
#[test]
fn patch_returns_404_without_bucket() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let resp = patch_json(ctx.addr(), "/v1/settings", &json!({"model": "test"}));
    assert_eq!(resp.status, 404, "body: {}", resp.body);
}

/// PATCH sets the model field.
#[test]
fn patch_sets_model() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(settings_config(temp.path()));

    let resp = patch_json(
        ctx.addr(),
        "/v1/settings",
        &json!({"model": "test-model"}),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(resp.json()["model"], "test-model");
}

/// PATCH sets the system_prompt field.
#[test]
fn patch_sets_system_prompt() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(settings_config(temp.path()));

    let resp = patch_json(
        ctx.addr(),
        "/v1/settings",
        &json!({"system_prompt": "You are a helpful assistant."}),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(resp.json()["system_prompt"], "You are a helpful assistant.");
}

/// PATCH sets max_output_tokens.
#[test]
fn patch_sets_max_output_tokens() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(settings_config(temp.path()));

    let resp = patch_json(
        ctx.addr(),
        "/v1/settings",
        &json!({"max_output_tokens": 4096}),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(resp.json()["max_output_tokens"], 4096);
}

/// PATCH sets context_length.
#[test]
fn patch_sets_context_length() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(settings_config(temp.path()));

    let resp = patch_json(
        ctx.addr(),
        "/v1/settings",
        &json!({"context_length": 8192}),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(resp.json()["context_length"], 8192);
}

// ---------------------------------------------------------------------------
// PATCH /v1/settings — multiple fields and persistence
// ---------------------------------------------------------------------------

/// PATCH multiple fields at once.
#[test]
fn patch_multiple_fields() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(settings_config(temp.path()));

    let resp = patch_json(
        ctx.addr(),
        "/v1/settings",
        &json!({
            "model": "multi-model",
            "system_prompt": "Be concise.",
            "max_output_tokens": 1000,
            "context_length": 2000
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(json["model"], "multi-model");
    assert_eq!(json["system_prompt"], "Be concise.");
    assert_eq!(json["max_output_tokens"], 1000);
    assert_eq!(json["context_length"], 2000);
}

/// PATCH then GET confirms persistence.
#[test]
fn patch_persists_via_get() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(settings_config(temp.path()));

    let resp = patch_json(
        ctx.addr(),
        "/v1/settings",
        &json!({
            "model": "persisted-model",
            "system_prompt": "Persist this."
        }),
    );
    assert_eq!(resp.status, 200);

    let resp = get(ctx.addr(), "/v1/settings");
    assert_eq!(resp.status, 200);

    let json = resp.json();
    assert_eq!(json["model"], "persisted-model");
    assert_eq!(json["system_prompt"], "Persist this.");
}

/// Sequential PATCHes accumulate state (second PATCH does not clear first).
#[test]
fn patch_sequential_accumulates() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(settings_config(temp.path()));

    // First PATCH sets model
    let resp = patch_json(
        ctx.addr(),
        "/v1/settings",
        &json!({"model": "model-a"}),
    );
    assert_eq!(resp.status, 200);

    // Second PATCH sets system_prompt (model should be preserved)
    let resp = patch_json(
        ctx.addr(),
        "/v1/settings",
        &json!({"system_prompt": "Added later."}),
    );
    assert_eq!(resp.status, 200);

    let resp = get(ctx.addr(), "/v1/settings");
    assert_eq!(resp.status, 200);

    let json = resp.json();
    assert_eq!(json["model"], "model-a", "model should be preserved");
    assert_eq!(json["system_prompt"], "Added later.");
}

/// PATCH empty body is a no-op (preserves existing state).
#[test]
fn patch_empty_body_noop() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(settings_config(temp.path()));

    // Set model first
    let resp = patch_json(
        ctx.addr(),
        "/v1/settings",
        &json!({"model": "keep-me"}),
    );
    assert_eq!(resp.status, 200);

    // Empty PATCH
    let resp = patch_json(ctx.addr(), "/v1/settings", &json!({}));
    assert_eq!(resp.status, 200);

    let resp = get(ctx.addr(), "/v1/settings");
    assert_eq!(resp.status, 200);
    assert_eq!(resp.json()["model"], "keep-me");
}

// ---------------------------------------------------------------------------
// PATCH /v1/settings — null values and field clearing
// ---------------------------------------------------------------------------

/// PATCH with null clears a previously set field.
#[test]
fn patch_null_clears_field() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(settings_config(temp.path()));

    // Set model
    let resp = patch_json(
        ctx.addr(),
        "/v1/settings",
        &json!({"model": "to-be-cleared"}),
    );
    assert_eq!(resp.status, 200);
    assert_eq!(resp.json()["model"], "to-be-cleared");

    // Clear model via null
    let resp = patch_json(ctx.addr(), "/v1/settings", &json!({"model": null}));
    assert_eq!(resp.status, 200);

    let resp = get(ctx.addr(), "/v1/settings");
    assert_eq!(resp.status, 200);
    assert!(
        resp.json()["model"].is_null(),
        "model should be null after clearing"
    );
}

// ---------------------------------------------------------------------------
// PATCH /v1/settings — model_overrides
// ---------------------------------------------------------------------------

/// PATCH model_overrides stores per-model generation parameters.
#[test]
fn patch_model_overrides() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(settings_config(temp.path()));

    // Must set active model first (overrides are keyed to active model)
    let resp = patch_json(
        ctx.addr(),
        "/v1/settings",
        &json!({"model": "test-model"}),
    );
    assert_eq!(resp.status, 200);

    // Set overrides
    let resp = patch_json(
        ctx.addr(),
        "/v1/settings",
        &json!({
            "model_overrides": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40
            }
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(resp.json()["model"], "test-model");
}

/// PATCH model_overrides with all null clears the override entry.
#[test]
fn patch_model_overrides_cleanup_on_all_null() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(settings_config(temp.path()));

    // Set model and overrides
    patch_json(
        ctx.addr(),
        "/v1/settings",
        &json!({"model": "test-model"}),
    );
    patch_json(
        ctx.addr(),
        "/v1/settings",
        &json!({"model_overrides": {"temperature": 0.5}}),
    );

    // Clear all override fields
    let resp = patch_json(
        ctx.addr(),
        "/v1/settings",
        &json!({
            "model_overrides": {
                "temperature": null,
                "top_p": null,
                "top_k": null
            }
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
}

// ---------------------------------------------------------------------------
// PATCH /v1/settings — validation
// ---------------------------------------------------------------------------

/// PATCH with invalid JSON returns 400.
#[test]
fn patch_invalid_json_returns_400() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(settings_config(temp.path()));

    let resp = send_request(
        ctx.addr(),
        "PATCH",
        "/v1/settings",
        &[("Content-Type", "application/json")],
        Some("not json {{{"),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(json["error"]["code"], "invalid_request");
}

// ---------------------------------------------------------------------------
// DELETE /v1/settings/models/{model_id}
// ---------------------------------------------------------------------------

/// DELETE without a bucket returns 404.
#[test]
fn delete_model_returns_404_without_bucket() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let resp = delete(ctx.addr(), "/v1/settings/models/test-model");
    assert_eq!(resp.status, 404, "body: {}", resp.body);
}

/// DELETE removes per-model overrides.
#[test]
fn delete_model_overrides() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(settings_config(temp.path()));

    // Set model and overrides
    patch_json(
        ctx.addr(),
        "/v1/settings",
        &json!({"model": "test-model"}),
    );
    patch_json(
        ctx.addr(),
        "/v1/settings",
        &json!({"model_overrides": {"temperature": 0.5, "top_k": 20}}),
    );

    // Delete overrides for that model
    let resp = delete(ctx.addr(), "/v1/settings/models/test-model");
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    // Model field should still be set (DELETE only removes overrides)
    let json = resp.json();
    assert_eq!(json["model"], "test-model");
}

/// DELETE for a nonexistent model is a no-op (returns 200).
#[test]
fn delete_nonexistent_model_noop() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(settings_config(temp.path()));

    let resp = delete(ctx.addr(), "/v1/settings/models/nonexistent");
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    // Response should still have valid settings shape
    let json = resp.json();
    assert!(json["available_models"].is_array());
}

// ---------------------------------------------------------------------------
// Response shape
// ---------------------------------------------------------------------------

/// GET response has application/json content type.
#[test]
fn response_has_json_content_type() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(settings_config(temp.path()));

    let resp = get(ctx.addr(), "/v1/settings");
    assert_eq!(resp.status, 200);
    assert_eq!(resp.header("content-type"), Some("application/json"));
}

/// GET response contains all expected top-level fields.
#[test]
fn response_has_correct_fields() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(settings_config(temp.path()));

    let resp = get(ctx.addr(), "/v1/settings");
    assert_eq!(resp.status, 200);

    let json = resp.json();
    let obj = json.as_object().expect("response should be an object");
    assert!(obj.contains_key("model"), "missing 'model' field");
    assert!(
        obj.contains_key("system_prompt"),
        "missing 'system_prompt' field"
    );
    assert!(
        obj.contains_key("max_output_tokens"),
        "missing 'max_output_tokens' field"
    );
    assert!(
        obj.contains_key("context_length"),
        "missing 'context_length' field"
    );
    assert!(
        obj.contains_key("available_models"),
        "missing 'available_models' field"
    );
}

// ---------------------------------------------------------------------------
// Bare path variants (without /v1 prefix)
// ---------------------------------------------------------------------------

/// GET /settings works (without /v1 prefix).
#[test]
fn bare_path_get() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(settings_config(temp.path()));

    let resp = get(ctx.addr(), "/settings");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert!(resp.json()["available_models"].is_array());
}

/// PATCH /settings works (without /v1 prefix).
#[test]
fn bare_path_patch() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(settings_config(temp.path()));

    let resp = patch_json(
        ctx.addr(),
        "/settings",
        &json!({"model": "bare-model"}),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(resp.json()["model"], "bare-model");
}

/// DELETE /settings/models/{id} works (without /v1 prefix).
#[test]
fn bare_path_delete() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(settings_config(temp.path()));

    let resp = delete(ctx.addr(), "/settings/models/some-model");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
}

// ---------------------------------------------------------------------------
// Multi-tenant settings isolation
// ---------------------------------------------------------------------------

/// Settings are scoped per tenant via storage_prefix.
///
/// Each tenant's settings.toml lives at `<bucket>/<storage_prefix>/settings.toml`,
/// so Tenant A's PATCH should never affect Tenant B's GET.
#[test]
fn settings_tenant_isolation() {
    let temp = TempDir::new().expect("temp dir");
    // Pre-create tenant storage directories so save_bucket_settings can write.
    std::fs::create_dir_all(temp.path().join("acme")).expect("create acme dir");
    std::fs::create_dir_all(temp.path().join("globex")).expect("create globex dir");

    let mut config = ServerConfig::new();
    config.gateway_secret = Some("secret".to_string());
    config.bucket = Some(temp.path().to_path_buf());
    config.tenants = vec![
        TenantSpec {
            id: "acme".to_string(),
            storage_prefix: "acme".to_string(),
            allowed_models: vec![],
        },
        TenantSpec {
            id: "globex".to_string(),
            storage_prefix: "globex".to_string(),
            allowed_models: vec![],
        },
    ];

    let ctx = ServerTestContext::new(config);

    // Acme sets a model.
    let resp = send_request(
        ctx.addr(),
        "PATCH",
        "/v1/settings",
        &[
            ("X-Talu-Gateway-Secret", "secret"),
            ("X-Talu-Tenant-Id", "acme"),
            ("Content-Type", "application/json"),
        ],
        Some(r#"{"model": "acme-model"}"#),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(resp.json()["model"], "acme-model");

    // Globex should still see defaults (null model).
    let resp = send_request(
        ctx.addr(),
        "GET",
        "/v1/settings",
        &[
            ("X-Talu-Gateway-Secret", "secret"),
            ("X-Talu-Tenant-Id", "globex"),
        ],
        None,
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert!(
        resp.json()["model"].is_null(),
        "globex should have null model, got: {}",
        resp.json()["model"]
    );

    // Globex sets its own model.
    let resp = send_request(
        ctx.addr(),
        "PATCH",
        "/v1/settings",
        &[
            ("X-Talu-Gateway-Secret", "secret"),
            ("X-Talu-Tenant-Id", "globex"),
            ("Content-Type", "application/json"),
        ],
        Some(r#"{"model": "globex-model"}"#),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(resp.json()["model"], "globex-model");

    // Acme's model should be unchanged.
    let resp = send_request(
        ctx.addr(),
        "GET",
        "/v1/settings",
        &[
            ("X-Talu-Gateway-Secret", "secret"),
            ("X-Talu-Tenant-Id", "acme"),
        ],
        None,
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(
        resp.json()["model"], "acme-model",
        "acme's model should be unchanged after globex patch"
    );
}

/// PATCH with an empty `model_overrides` object `{}` is a no-op — existing
/// overrides are preserved because no keys are processed.
#[test]
fn patch_empty_model_overrides_is_noop() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(settings_config(temp.path()));

    // Set model and overrides.
    patch_json(
        ctx.addr(),
        "/v1/settings",
        &json!({"model": "test-model"}),
    );
    patch_json(
        ctx.addr(),
        "/v1/settings",
        &json!({"model_overrides": {"temperature": 0.5}}),
    );

    // Send empty overrides object — should be a no-op.
    let resp = patch_json(
        ctx.addr(),
        "/v1/settings",
        &json!({"model_overrides": {}}),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    // Verify overrides survived by reading the TOML file directly.
    // The GET/PATCH response doesn't include raw overrides — only
    // `available_models[].overrides` which requires a real managed model.
    let toml_path = temp.path().join("settings.toml");
    let toml_str = std::fs::read_to_string(&toml_path)
        .expect("settings.toml should exist after PATCH");
    let parsed: toml::Value =
        toml::from_str(&toml_str).expect("settings.toml should be valid TOML");
    let temp_val = parsed
        .get("models")
        .and_then(|m| m.get("test-model"))
        .and_then(|m| m.get("temperature"))
        .and_then(|t| t.as_float())
        .expect("temperature override should still be present");
    assert!(
        (temp_val - 0.5).abs() < 0.001,
        "temperature override should be 0.5 after empty overrides patch, got: {temp_val}"
    );
}
