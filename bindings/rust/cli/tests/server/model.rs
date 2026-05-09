//! Integration tests for `POST /v1/model/config`.

use serde_json::json;

use crate::server::common::*;

#[test]
fn model_config_requires_model_field() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let resp = post_json(ctx.addr(), "/v1/model/config", &json!({}));
    assert_eq!(resp.status, 400, "body: {}", resp.body);

    let body = resp.json();
    assert_eq!(body["error"]["code"], "invalid_request");
}

#[test]
fn model_config_rejects_non_hf_model_id_shape() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let resp = post_json(
        ctx.addr(),
        "/v1/model/config",
        &json!({
            "model": "not-a-valid-model-id"
        }),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);

    let body = resp.json();
    assert_eq!(body["error"]["code"], "model_config_failed");
}

#[test]
fn model_config_route_is_present_in_openapi() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let resp = get(ctx.addr(), "/openapi.json");
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let spec = resp.json();
    let paths = spec["paths"].as_object().expect("paths object");
    assert!(
        paths.contains_key("/v1/model/config"),
        "openapi must include /v1/model/config"
    );
}
