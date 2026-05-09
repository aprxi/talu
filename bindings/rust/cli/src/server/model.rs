use std::convert::Infallible;
use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::{Request, Response, StatusCode};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::server::auth_gateway::AuthContext;
use crate::server::state::AppState;

type BoxBody = http_body_util::combinators::BoxBody<Bytes, Infallible>;

#[derive(Deserialize, ToSchema)]
pub struct ModelConfigRequest {
    /// HuggingFace model id in org/name format.
    pub model: String,
    /// Optional revision (currently only "main" is supported in core).
    pub revision: Option<String>,
    /// Optional custom HuggingFace endpoint base URL.
    pub endpoint_url: Option<String>,
    /// Optional HF token for private repositories.
    pub token: Option<String>,
    /// Force refetch even if config is already cached.
    #[serde(default)]
    pub force_refresh: bool,
    /// Include `size_bytes` best-effort metadata.
    #[serde(default = "default_include_size")]
    pub include_size: bool,
}

fn default_include_size() -> bool {
    true
}

/// Documentation-only response envelope.
#[derive(Serialize, ToSchema)]
#[allow(dead_code)]
pub struct ModelConfigResponse {
    pub model: String,
    pub revision: String,
    pub source: String,
    pub config: serde_json::Value,
    pub minimal: serde_json::Value,
    pub size_bytes: Option<u64>,
}

#[utoipa::path(post, path = "/v1/model/config", tag = "Models",
    request_body = ModelConfigRequest,
    responses((status = 200, body = ModelConfigResponse)))]
pub async fn handle_config(
    _state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let body_bytes = match req.into_body().collect().await {
        Ok(body) => body.to_bytes(),
        Err(_) => return json_error(StatusCode::BAD_REQUEST, "invalid_request", "Invalid body"),
    };

    let payload: ModelConfigRequest = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(e) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                &format!("Invalid JSON: {e}"),
            )
        }
    };

    if payload.model.trim().is_empty() {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "model is required",
        );
    }

    match talu::model::fetch_hf_model_config_json(
        payload.model.trim(),
        payload.revision.as_deref(),
        payload.endpoint_url.as_deref(),
        payload.token.as_deref(),
        payload.force_refresh,
        payload.include_size,
    ) {
        Ok(json) => json_raw(StatusCode::OK, json),
        Err(e) => json_error(
            StatusCode::BAD_REQUEST,
            "model_config_failed",
            &e.to_string(),
        ),
    }
}

fn json_raw(status: StatusCode, json: String) -> Response<BoxBody> {
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(json)).boxed())
        .unwrap()
}

fn json_error(status: StatusCode, code: &str, message: &str) -> Response<BoxBody> {
    let payload = serde_json::json!({
        "error": {
            "code": code,
            "message": message
        }
    });
    let body = serde_json::to_vec(&payload).unwrap_or_else(|_| b"{}".to_vec());
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body)).boxed())
        .unwrap()
}
