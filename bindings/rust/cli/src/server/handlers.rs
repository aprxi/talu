use std::collections::HashSet;
use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use http_body_util::{BodyExt, Full, StreamBody};
use hyper::body::Frame;
use hyper::body::Incoming;
use hyper::{Request, Response, StatusCode};
use serde_json::json;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;

use crate::bucket_settings;
use crate::provider;
use crate::server::auth_gateway::AuthContext;
use crate::server::events;
use crate::server::responses_types;
use crate::server::state::{AppState, ModelLoadResult, StoredResponse};
use talu::documents::{DocumentError, DocumentsHandle};
use talu::responses::{ContentType, ItemType};
use talu::{ChatHandle, FinishReason};

type BoxBody = http_body_util::combinators::BoxBody<Bytes, Infallible>;

/// Guard that sets an `AtomicBool` flag when dropped.
///
/// Used to signal cooperative cancellation to a `spawn_blocking` generation
/// task when the HTTP handler future is dropped (client disconnect).
/// Call [`defuse`](CancelOnDrop::defuse) after the task completes so the
/// flag is not set unnecessarily.
struct CancelOnDrop(Option<Arc<AtomicBool>>);

impl CancelOnDrop {
    fn defuse(mut self) {
        self.0.take();
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if let Some(ref flag) = self.0 {
            flag.store(true, Ordering::Release);
        }
    }
}

#[derive(Debug, Clone)]
struct GenerationRequest {
    model: Option<String>,
    max_output_tokens: Option<i64>,
    max_completion_tokens: Option<i64>,
    max_reasoning_tokens: Option<i64>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u32>,
    presence_penalty: Option<f64>,
    frequency_penalty: Option<f64>,
    seed: Option<u64>,
    logprobs: LogprobConfig,
    reasoning: ReasoningConfig,
    instructions: Option<String>,
    text_format: Option<TextFormatConfig>,
}

#[derive(Debug, Clone, Copy)]
struct LogprobConfig {
    top_logprobs: usize,
}

impl Default for LogprobConfig {
    fn default() -> Self {
        Self { top_logprobs: 0 }
    }
}

#[derive(Debug, Clone)]
enum TextFormatConfig {
    Text,
    JsonSchema,
}

#[derive(Debug, Clone, Default)]
struct ReasoningConfig {
    effort: Option<String>,
    summary: Option<String>,
}

/// Error type for response generation with HTTP status code information.
#[derive(Debug)]
pub enum ResponseError {
    /// Bad request (400) - user error like invalid prompt_id
    BadRequest { code: &'static str, message: String },
    /// Internal server error (500) - system error
    Internal { code: &'static str, message: String },
}

impl ResponseError {
    fn bad_request(code: &'static str, message: impl Into<String>) -> Self {
        Self::BadRequest {
            code,
            message: message.into(),
        }
    }

    fn internal(code: &'static str, message: impl Into<String>) -> Self {
        Self::Internal {
            code,
            message: message.into(),
        }
    }

    fn status_code(&self) -> StatusCode {
        match self {
            Self::BadRequest { .. } => StatusCode::BAD_REQUEST,
            Self::Internal { .. } => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn error_code(&self) -> &'static str {
        match self {
            Self::BadRequest { code, .. } => code,
            Self::Internal { code, .. } => code,
        }
    }

    fn error_message(&self) -> &str {
        match self {
            Self::BadRequest { message, .. } => message,
            Self::Internal { message, .. } => message,
        }
    }
}

impl From<anyhow::Error> for ResponseError {
    fn from(err: anyhow::Error) -> Self {
        Self::Internal {
            code: "inference_error",
            message: err.to_string(),
        }
    }
}

pub async fn handle_responses(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth_ctx: Option<AuthContext>,
) -> Response<BoxBody> {
    handle_generate(state, req, auth_ctx).await
}

async fn handle_generate(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth_ctx: Option<AuthContext>,
) -> Response<BoxBody> {
    let (parts, body) = req.into_parts();
    // `/v1/responses` serves the strict OpenResponses contract only.
    let strict_responses = true;
    if let Some(ctx) = auth_ctx.as_ref() {
        log::info!(
            target: "server::gen",
            "Authenticated tenant: {}, prefix: {}",
            ctx.tenant_id,
            ctx.storage_prefix
        );
    }

    let body_bytes = match body.collect().await {
        Ok(body) => body.to_bytes(),
        Err(_) => {
            return api_error(
                strict_responses,
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "Invalid body",
            )
        }
    };

    // Log reproducible curl command at DEBUG for replay/debugging.
    if log::log_enabled!(target: "server::gen", log::Level::Debug) {
        let host = parts
            .headers
            .get("host")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("localhost:8258");
        let body_str = String::from_utf8_lossy(&body_bytes);
        log::debug!(
            target: "server::gen",
            "curl -s http://{}{} -H 'content-type: application/json' -d '{}'",
            host, parts.uri.path(), body_str.replace('\'', "'\\''")
        );
    }

    let parse_error = |err: serde_json::Error| {
        api_error(
            strict_responses,
            StatusCode::BAD_REQUEST,
            "invalid_request",
            &format!("Invalid JSON: {err}"),
        )
    };
    let parsed = match serde_json::from_slice::<responses_types::CreateResponseBody>(&body_bytes) {
        Ok(val) => val,
        Err(err) => return parse_error(err),
    };
    if let Err(message) = validate_responses_request(&parsed) {
        return api_error(
            strict_responses,
            StatusCode::BAD_REQUEST,
            "invalid_request",
            &message,
        );
    }
    let text_format = match parse_requested_text_format(parsed.text.as_ref()) {
        Ok(value) => value,
        Err(message) => {
            return api_error(
                strict_responses,
                StatusCode::BAD_REQUEST,
                "invalid_request",
                &message,
            )
        }
    };
    let logprobs = match parse_logprob_config(parsed.include.as_ref(), parsed.top_logprobs) {
        Ok(value) => value,
        Err(message) => {
            return api_error(
                strict_responses,
                StatusCode::BAD_REQUEST,
                "invalid_request",
                &message,
            )
        }
    };
    let reasoning = match parse_reasoning_config(parsed.reasoning.as_ref()) {
        Ok(value) => value,
        Err(message) => {
            return api_error(
                strict_responses,
                StatusCode::BAD_REQUEST,
                "invalid_request",
                &message,
            )
        }
    };

    let request = GenerationRequest {
        model: parsed.model.clone(),
        max_output_tokens: parsed.max_output_tokens,
        max_completion_tokens: parsed.max_completion_tokens,
        max_reasoning_tokens: parsed.max_reasoning_tokens,
        temperature: parsed.temperature,
        top_p: parsed.top_p,
        top_k: parsed.top_k,
        presence_penalty: parsed.presence_penalty,
        frequency_penalty: parsed.frequency_penalty,
        seed: parsed.seed,
        logprobs,
        reasoning,
        instructions: parsed.instructions.clone(),
        text_format,
    };

    let stream = parsed.stream.unwrap_or(false);

    let tools_json = parsed.tools.clone();
    let tool_choice_json = parsed.tool_choice.clone();
    let previous_response_id = parsed.previous_response_id.clone();
    let input_value = parsed.input.clone();
    let store = parsed.store.unwrap_or(false);
    // Extract session_id from metadata (allows clients to resume a conversation
    // without previous_response_id, e.g. after a page refresh).
    let request_session_id: Option<String> = parsed
        .metadata
        .as_ref()
        .and_then(|m| m.get("session_id"))
        .and_then(|v| v.as_str())
        .map(String::from);
    let prompt_id = None;

    // Internal project scoping hint (if present in metadata).
    let project_id: Option<String> = parsed
        .metadata
        .as_ref()
        .and_then(|m| m.get("project_id"))
        .and_then(|v| v.as_str())
        .map(String::from);

    log::debug!(
        target: "server::gen",
        "model={} stream={} store={} session_id={:?} prompt_id={:?} prev_response_id={:?}",
        request.model.as_deref().unwrap_or("(default)"),
        stream, store, request_session_id, prompt_id, previous_response_id
    );

    // Validate that input is present (string, array, or null with previous_response_id).
    if input_value.is_none() && previous_response_id.is_none() {
        return api_error(
            strict_responses,
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "Missing input",
        );
    }

    if stream {
        return stream_response(
            state,
            request,
            input_value,
            tools_json,
            tool_choice_json,
            previous_response_id,
            store,
            request_session_id,
            prompt_id,
            project_id,
            strict_responses,
            auth_ctx,
        )
        .await;
    }

    let response_json = match generate_response(
        state,
        request,
        input_value,
        tools_json,
        tool_choice_json,
        previous_response_id,
        store,
        request_session_id,
        prompt_id,
        project_id,
        strict_responses,
        auth_ctx.as_ref(),
    )
    .await
    {
        Ok(val) => val,
        Err(err) => {
            return api_error(
                strict_responses,
                err.status_code(),
                err.error_code(),
                err.error_message(),
            )
        }
    };
    if let Some(ctx) = auth_ctx.as_ref() {
        log_generation_completed(ctx);
    }

    let response_body = match serde_json::to_vec(&response_json) {
        Ok(body) => body,
        Err(_) => {
            return api_error(
                strict_responses,
                StatusCode::INTERNAL_SERVER_ERROR,
                "serialization_error",
                "Failed to serialize response",
            )
        }
    };

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(response_body)).boxed())
        .unwrap()
}

#[utoipa::path(get, path = "/v1/models", tag = "Models",
    responses((status = 200, description = "List of available models")))]
pub async fn handle_models(
    state: Arc<AppState>,
    _req: Request<Incoming>,
    auth_ctx: Option<AuthContext>,
) -> Response<BoxBody> {
    let allowed_models = auth_ctx
        .as_ref()
        .and_then(|ctx| {
            state
                .tenant_registry
                .as_ref()
                .and_then(|registry| registry.get(&ctx.tenant_id))
                .map(|tenant| tenant.allowed_models.clone())
        })
        .unwrap_or_default();

    let mut models = match list_backend_models(state.clone()).await {
        Ok(items) if !items.is_empty() => items,
        _ => {
            let fallback = state
                .backend
                .lock()
                .await
                .current_model
                .clone()
                .or_else(|| state.configured_model.clone());
            if let Some(model_id) = fallback {
                vec![talu::RemoteModelInfo {
                    id: model_id,
                    object: "model".to_string(),
                    created: 0,
                    owned_by: "talu".to_string(),
                }]
            } else {
                Vec::new()
            }
        }
    };

    // Merge models from enabled remote providers (prefixed with "provider::").
    if let Some(ref bucket) = state.bucket_path {
        let db_root = bucket.join("kv").to_string_lossy().into_owned();
        if let Ok(Ok(remote)) =
            tokio::task::spawn_blocking(move || talu::provider_config_list_remote_models(&db_root))
                .await
        {
            models.extend(remote);
        }
    }

    if !allowed_models.is_empty() {
        let allowed: HashSet<String> = allowed_models.into_iter().collect();
        models.retain(|model| allowed.contains(&model.id));
    }

    let models = models
        .into_iter()
        .map(|model| {
            json!({
                "id": model.id,
                "object": model.object,
                "created": model.created,
                "owned_by": model.owned_by
            })
        })
        .collect::<Vec<_>>();

    let payload = json!({
        "object": "list",
        "data": models
    });

    let body = serde_json::to_vec(&payload).unwrap_or_else(|_| b"{}".to_vec());
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body)).boxed())
        .unwrap()
}

/// Compute the absolute blob file path for a sha256 blob ref.
///
/// Returns `Some(path)` for `sha256:<64hex>` refs, `None` for multipart or invalid refs.
fn blob_ref_to_file_url(storage_path: &std::path::Path, blob_ref: &str) -> Option<String> {
    let hex = blob_ref.strip_prefix("sha256:")?;
    if hex.len() != 64 {
        return None;
    }
    let mut path = storage_path.to_path_buf();
    path.push("blobs");
    path.push(&hex[..2]);
    path.push(hex);
    Some(format!("file://{}", path.display()))
}

/// File document metadata (matches the JSON stored by the upload handler).
#[derive(serde::Deserialize)]
struct FileDocContent {
    blob_ref: Option<String>,
    mime_type: Option<String>,
    kind: Option<String>,
}

/// Resolve file references in a structured input JSON array.
///
/// Scans the input for `input_image` or `input_file` items whose URLs look like
/// file IDs (e.g. `file_abc123`). For each, rewrites the URL to a `file://` path
/// pointing directly to the blob on disk, so the engine can read it zero-copy.
///
/// Returns the (possibly modified) JSON string ready for `load_responses_json`.
fn resolve_file_references(input_json: &str, storage_path: &std::path::Path) -> Result<String> {
    let mut items: serde_json::Value =
        serde_json::from_str(input_json).context("failed to parse input JSON")?;

    let arr = match items.as_array_mut() {
        Some(a) => a,
        None => return Ok(input_json.to_string()),
    };

    let mut needs_storage = false;
    // Pre-scan: check if any content parts reference file IDs.
    for item in arr.iter() {
        if let Some(content) = item.get("content").and_then(|c| c.as_array()) {
            for part in content {
                let part_type = part.get("type").and_then(|t| t.as_str()).unwrap_or("");
                match part_type {
                    "input_image" => {
                        if let Some(url) = part.get("image_url").and_then(|u| u.as_str()) {
                            if url.starts_with("file_") {
                                needs_storage = true;
                            }
                        }
                    }
                    "input_file" => {
                        if let Some(url) = part.get("file_url").and_then(|u| u.as_str()) {
                            if url.starts_with("file_") {
                                needs_storage = true;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    if !needs_storage {
        return Ok(input_json.to_string());
    }

    let docs = DocumentsHandle::open(&storage_path.join("tables").join("documents"))
        .map_err(|e| anyhow!("failed to open documents for file resolution: {}", e))?;

    for item in arr.iter_mut() {
        let content = match item.get_mut("content").and_then(|c| c.as_array_mut()) {
            Some(c) => c,
            None => continue,
        };

        let mut replacements: Vec<(usize, serde_json::Value)> = Vec::new();

        for (idx, part) in content.iter().enumerate() {
            let part_type = part.get("type").and_then(|t| t.as_str()).unwrap_or("");

            let file_id = match part_type {
                "input_image" => part.get("image_url").and_then(|u| u.as_str()),
                "input_file" => part.get("file_url").and_then(|u| u.as_str()),
                _ => None,
            };

            let file_id = match file_id {
                Some(id) if id.starts_with("file_") => id,
                _ => continue,
            };

            // Look up document metadata.
            let doc = match docs.get(file_id) {
                Ok(Some(d)) if d.doc_type == "file" => d,
                Ok(_) | Err(_) => continue,
            };

            let meta: FileDocContent =
                serde_json::from_str(&doc.doc_json).unwrap_or(FileDocContent {
                    blob_ref: None,
                    mime_type: None,
                    kind: None,
                });

            let blob_ref = match meta.blob_ref {
                Some(ref r) if !r.is_empty() => r.as_str(),
                _ => continue,
            };

            let mime = meta
                .mime_type
                .as_deref()
                .unwrap_or("application/octet-stream");
            let is_image = meta.kind.as_deref() == Some("image") || mime.starts_with("image/");

            if is_image {
                // Zero-copy: pass file:// URL so the engine reads directly from disk.
                if let Some(file_url) = blob_ref_to_file_url(storage_path, blob_ref) {
                    replacements.push((
                        idx,
                        json!({
                            "type": "input_image",
                            "image_url": file_url
                        }),
                    ));
                }
                // Multipart blobs are unsupported for zero-copy; skip silently.
            } else {
                // Non-image files: pass file:// URL with metadata.
                if let Some(file_url) = blob_ref_to_file_url(storage_path, blob_ref) {
                    let filename = part
                        .get("filename")
                        .and_then(|f| f.as_str())
                        .unwrap_or(&doc.title);
                    replacements.push((
                        idx,
                        json!({
                            "type": "input_file",
                            "file_data": file_url,
                            "filename": filename
                        }),
                    ));
                }
            }
        }

        // Apply replacements in reverse to preserve indices.
        for (idx, new_part) in replacements.into_iter().rev() {
            content[idx] = new_part;
        }
    }

    serde_json::to_string(&items).context("failed to serialize resolved input")
}

async fn generate_response(
    state: Arc<AppState>,
    request: GenerationRequest,
    input_value: Option<serde_json::Value>,
    tools_json: Option<serde_json::Value>,
    tool_choice_json: Option<serde_json::Value>,
    previous_response_id: Option<String>,
    store: bool,
    request_session_id: Option<String>,
    prompt_id: Option<String>,
    project_id: Option<String>,
    strict_responses: bool,
    auth_ctx: Option<&AuthContext>,
) -> Result<serde_json::Value, ResponseError> {
    let request_max_output_tokens = request.max_output_tokens;
    let request_max_completion_tokens = request.max_completion_tokens;
    let request_max_reasoning_tokens = request.max_reasoning_tokens;
    let temperature = request.temperature;
    let top_p = request.top_p;
    let top_k = request.top_k;
    let presence_penalty = request.presence_penalty;
    let frequency_penalty = request.frequency_penalty;
    let seed = request.seed;
    let logprobs = request.logprobs;
    let reasoning = request.reasoning;
    let instructions = request.instructions;
    let text_format = request.text_format;

    let model_id = select_model_id(state.clone(), request.model.clone()).await?;

    // Load previous conversation state if chaining.
    let requester_tenant = auth_ctx.map(|ctx| ctx.tenant_id.as_str());
    let prev_state = if let Some(ref prev_id) = previous_response_id {
        let store = state.response_store.lock().await;
        store.get(prev_id).and_then(|s| {
            if s.tenant_id.as_deref() == requester_tenant {
                Some((
                    s.responses_json.clone(),
                    s.tools_json.clone(),
                    s.tool_choice_json.clone(),
                    s.session_id.clone(),
                ))
            } else {
                None
            }
        })
    } else {
        None
    };

    // Merge: previous conversation's tools/tool_choice are used as defaults
    // if the new request doesn't specify them.
    let effective_tools = tools_json.or_else(|| prev_state.as_ref().and_then(|s| s.1.clone()));
    let effective_tool_choice =
        tool_choice_json.or_else(|| prev_state.as_ref().and_then(|s| s.2.clone()));

    // Resolve session ID: explicit request > previous response chain > new UUID.
    let session_id = request_session_id
        .or_else(|| prev_state.as_ref().and_then(|s| s.3.clone()))
        .unwrap_or_else(|| uuid::Uuid::new_v4().hyphenated().to_string());

    // Resolve bucket path for persistence (only when store=true).
    let bucket_path = if store {
        state.bucket_path.as_ref().map(|base| match auth_ctx {
            Some(ctx) => base.join(&ctx.storage_prefix),
            None => base.to_path_buf(),
        })
    } else {
        None
    };

    // Load auto_title setting from bucket config.
    let auto_title = bucket_path.as_ref().map_or(false, |bp| {
        bucket_settings::load_bucket_settings(bp)
            .map(|s| s.auto_title)
            .unwrap_or(true)
    });
    let is_new_conversation = previous_response_id.is_none();

    // Ensure storage directory exists.
    if let Some(ref bp) = bucket_path {
        std::fs::create_dir_all(bp)
            .map_err(|e| anyhow!("failed to create storage directory: {}", e))?;
    }

    // Load bucket settings for fallback values (max_output_tokens + per-model overrides).
    // Use state.bucket_path (not bucket_path) so fallback works regardless of store flag.
    let fallback_settings = state
        .bucket_path
        .as_ref()
        .map(|base| match auth_ctx {
            Some(ctx) => base.join(&ctx.storage_prefix),
            None => base.to_path_buf(),
        })
        .and_then(|bp| bucket_settings::load_bucket_settings(&bp).ok());

    let max_output_tokens = request_max_output_tokens.or_else(|| {
        fallback_settings
            .as_ref()
            .and_then(|s| s.max_output_tokens.map(|v| v as i64))
    });

    // Apply per-model sampling overrides as fallbacks when the request doesn't specify them.
    let model_overrides = fallback_settings
        .as_ref()
        .and_then(|s| s.models.get(&model_id).cloned());
    let temperature = temperature.or_else(|| model_overrides.as_ref().and_then(|o| o.temperature));
    let top_p = top_p.or_else(|| model_overrides.as_ref().and_then(|o| o.top_p));

    // Resolve effective generation config using model sampling presets.
    // Applies 3-layer precedence: model presets → generation_config.json → API request.
    // For remote/provider models this may fail — fall back to raw request params.
    let effective_gen = talu::model::resolve_effective_generation_config(
        &model_id,
        &talu::EffectiveGenConfigRequest {
            temperature: temperature.map(|t| t as f32),
            top_k: top_k.map(|k| k as usize),
            top_p: top_p.map(|p| p as f32),
            presence_penalty: presence_penalty.map(|p| p as f32),
            frequency_penalty: frequency_penalty.map(|p| p as f32),
            seed: seed.unwrap_or(0),
            max_tokens: max_output_tokens.map(|t| t as usize).unwrap_or(8192),
            ..Default::default()
        },
    );
    if let Err(ref e) = effective_gen {
        log::warn!(target: "server::gen", "resolve_effective_generation_config: {e} (using raw request params)");
    }

    // If prompt_id is provided, fetch the document and extract system prompt.
    let system_prompt_from_doc: Option<String> = if let Some(ref pid) = prompt_id {
        if let Some(ref bp) = bucket_path {
            match DocumentsHandle::open(&bp.join("tables").join("documents")) {
                Ok(docs) => match docs.get(pid) {
                    Ok(Some(doc)) => {
                        // Parse doc_json to extract system prompt
                        // Try direct fields first (current UI format), then nested under "data" (legacy)
                        if let Ok(envelope) =
                            serde_json::from_str::<serde_json::Value>(&doc.doc_json)
                        {
                            envelope
                                .get("system")
                                .or_else(|| envelope.get("system_prompt"))
                                .or_else(|| envelope.get("data").and_then(|d| d.get("system")))
                                .or_else(|| {
                                    envelope.get("data").and_then(|d| d.get("system_prompt"))
                                })
                                .and_then(|s| s.as_str())
                                .map(|s| s.to_string())
                        } else {
                            None
                        }
                    }
                    Ok(None) => {
                        return Err(ResponseError::bad_request(
                            "invalid_request",
                            format!("prompt_id '{}' not found", pid),
                        ));
                    }
                    Err(DocumentError::DocumentNotFound(_)) => {
                        return Err(ResponseError::bad_request(
                            "invalid_request",
                            format!("prompt_id '{}' not found", pid),
                        ));
                    }
                    Err(e) => {
                        return Err(ResponseError::internal(
                            "storage_error",
                            format!("failed to fetch prompt document: {}", e),
                        ));
                    }
                },
                Err(e) => {
                    return Err(ResponseError::internal(
                        "storage_error",
                        format!("failed to open documents store: {}", e),
                    ));
                }
            }
        } else {
            return Err(ResponseError::bad_request(
                "invalid_request",
                "prompt_id requires storage to be configured",
            ));
        }
    } else {
        None
    };
    let effective_system_prompt = if strict_responses {
        instructions.clone()
    } else {
        system_prompt_from_doc.clone()
    };

    let response_id = format!("resp_{}", random_id());
    let created_at = now_unix_seconds();

    let backend = state.backend.clone();
    // Only pass prev_json for in-memory chaining when storage is NOT active.
    // When storage is active, set_storage_db auto-loads existing items.
    let prev_json = if bucket_path.is_none() {
        prev_state.map(|s| s.0)
    } else {
        let _ = prev_state; // consumed
        None
    };

    // Build serialized input for the blocking task.
    let input_json = input_value
        .as_ref()
        .filter(|v| v.is_array())
        .map(|v| serde_json::to_string(v).unwrap_or_default());
    let input_string = input_value
        .as_ref()
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    // File storage path for resolving file references (always available if bucket exists).
    let file_storage_path = state.bucket_path.as_ref().map(|base| match auth_ctx {
        Some(ctx) => base.join(&ctx.storage_prefix),
        None => base.to_path_buf(),
    });

    // Create a stop flag so the generation can be cancelled when the client
    // disconnects (hyper drops the handler future).  The guard sets the flag
    // on drop; we defuse it after a successful result.
    let stop_flag = Arc::new(AtomicBool::new(false));
    if let Ok(mut flags) = state.active_stop_flags.lock() {
        flags.retain(|w| w.strong_count() > 0);
        flags.push(Arc::downgrade(&stop_flag));
    }
    let stop_flag_for_gen = stop_flag.clone();
    let cancel_guard = CancelOnDrop(Some(stop_flag));

    let effective_config = effective_gen.ok();
    let effective_config_for_task = effective_config.clone();

    let session_id_for_task = session_id.clone();
    let bucket_for_task = bucket_path.clone();
    let model_id_for_task = model_id.clone();
    let system_prompt_for_task = effective_system_prompt.clone();
    let prompt_id_for_task = prompt_id.clone();
    let project_id_for_task = project_id.clone();
    let tools_json_for_generation = effective_tools.as_ref().map(|v| v.to_string());
    let tool_choice_for_generation = effective_tool_choice.as_ref().map(|v| v.to_string());
    let reasoning_effort_for_generation = reasoning.effort.clone();
    let batch_scheduler_for_task = state.batch_scheduler.lock().unwrap().clone();
    let (output_items, prompt_tokens, completion_tokens, prefill_ns, generation_ns, result_ttft_ns, responses_json, model_info_json, response_status, incomplete_reason) =
        tokio::task::spawn_blocking(move || {
            // Create ChatHandle with system prompt if prompt_id was provided
            log::trace!(target: "server::gen", "ChatHandle::new(system_prompt={})", system_prompt_for_task.is_some());
            let chat = ChatHandle::new(system_prompt_for_task.as_deref())?;

            // Enable storage persistence if store=true and bucket is configured.
            // set_storage_db auto-loads any existing items for this session.
            let has_storage = if let Some(ref bp) = bucket_for_task {
                if let Some(bp_str) = bp.to_str() {
                    log::trace!(target: "server::gen", "set_storage_db({:?}, {})", bp_str, session_id_for_task);
                    chat.set_storage_db(bp_str, &session_id_for_task)
                        .map_err(|e| anyhow!("failed to set storage: {}", e))?;
                    true
                } else {
                    false
                }
            } else {
                false
            };

            // Restore previous conversation from in-memory JSON only when
            // storage is not active — set_storage_db already loaded persisted items.
            if !has_storage {
                if let Some(ref prev) = prev_json {
                    log::trace!(target: "server::gen", "load_responses_json(prev, {} bytes)", prev.len());
                    chat.load_responses_json(prev)
                        .map_err(|e| anyhow!("failed to load previous conversation: {}", e))?;
                }
            }

            // Load input items into the conversation, resolving file references.
            // The Zig protocol parser stores input_image parts in conversation
            // items; the generate function extracts them for the vision encoder.
            if let Some(ref json) = input_json {
                log::trace!(target: "server::gen", "resolve_file_references(input, {} bytes)", json.len());
                let resolved = match file_storage_path.as_deref() {
                    Some(sp) => resolve_file_references(json, sp)?,
                    None => json.clone(),
                };
                log::trace!(target: "server::gen", "load_responses_json(input, {} bytes)", resolved.len());
                chat.load_responses_json(&resolved)
                    .map_err(|e| anyhow!("failed to load input: {}", e))?;
            } else if let Some(ref text) = input_string {
                log::trace!(target: "server::gen", "append_user_message({} chars)", text.len());
                chat.append_user_message(text)
                    .map_err(|e| anyhow!("failed to append user message: {}", e))?;
            }
            // Record item count before generation to extract only new output items.
            let pre_gen_count = chat.item_count();

            // Persist session metadata BEFORE generation so the conversation
            // appears in list_sessions immediately.
            if bucket_for_task.is_some() && pre_gen_count <= 2 {
                let t = input_string.as_deref().unwrap_or("Untitled");
                let title: String = t.chars().take(47).collect();
                let _ = chat.notify_session_update_ex(
                    Some(&model_id_for_task),
                    Some(&title),
                    Some("active"),
                    prompt_id_for_task.as_deref(),
                    project_id_for_task.as_deref(),
                );
            }

            let mut cfg = talu::router::GenerateConfig::default();
            if let Some(ref eff) = effective_config_for_task {
                cfg.max_tokens = eff.max_tokens;
                cfg.temperature = eff.temperature;
                cfg.top_k = eff.top_k;
                cfg.top_p = eff.top_p;
                cfg.min_p = eff.min_p;
                cfg.repetition_penalty = eff.repetition_penalty;
                cfg.presence_penalty = eff.presence_penalty;
                cfg.frequency_penalty = eff.frequency_penalty;
                cfg.seed = eff.seed;
            } else {
                // Fallback for remote/provider models where presets are unavailable.
                if let Some(max_tokens) = max_output_tokens {
                    cfg.max_tokens = max_tokens as usize;
                }
                if let Some(temp) = temperature {
                    cfg.temperature = temp as f32;
                }
                if let Some(top_p) = top_p {
                    cfg.top_p = top_p as f32;
                }
                if let Some(top_k) = top_k {
                    cfg.top_k = top_k as usize;
                }
                if let Some(seed) = seed {
                    cfg.seed = seed;
                }
                if let Some(pp) = presence_penalty {
                    cfg.presence_penalty = pp as f32;
                }
                if let Some(fp) = frequency_penalty {
                    cfg.frequency_penalty = fp as f32;
                }
            }
            cfg.tools_json = tools_json_for_generation;
            cfg.tool_choice = tool_choice_for_generation;
            cfg.reasoning_effort = reasoning_effort_for_generation;
            if let Some(mct) = request_max_completion_tokens {
                cfg.max_completion_tokens = Some(mct as usize);
            }
            if let Some(mrt) = request_max_reasoning_tokens {
                cfg.max_reasoning_tokens = Some(mrt as usize);
            }
            cfg.stop_flag = Some(stop_flag_for_gen.clone());

            log::debug!(target: "server::gen", "generating: model={} max_tokens={} temp={} top_p={} seed={}",
                model_id_for_task, cfg.max_tokens, cfg.temperature, cfg.top_p, cfg.seed);
            log::trace!(target: "server::gen", "generate(items={}, cfg={{max_tokens={}, temp={}, top_p={}}})",
                pre_gen_count, cfg.max_tokens, cfg.temperature, cfg.top_p);

            if let Some(ref sched) = batch_scheduler_for_task {
                // --- Batch scheduler path (concurrent GPU decode) ---
                let (request_id, event_rx) = sched
                    .submit_final_only(&chat, cfg, stop_flag_for_gen)
                    .map_err(|e| anyhow!("batch submit failed: {}", e))?;

                // Drain events until final and surface scheduler error events.
                let mut final_error: Option<String> = None;
                loop {
                    match event_rx.recv() {
                        Ok(event) => {
                            if matches!(event.event_type, talu::batch::EventType::Error) {
                                final_error = Some(if event.text.is_empty() {
                                    "scheduler run loop failed".to_string()
                                } else {
                                    event.text.clone()
                                });
                            }
                            if event.is_final {
                                break;
                            }
                        }
                        Err(_) => break,
                    }
                }

                if let Some(msg) = final_error {
                    return Err(anyhow!("batch generation failed: {msg}"));
                }

                let batch_result = sched.take_result(request_id);
                // Derive response status from BatchResult. Missing result is a
                // hard scheduler fault (the final event fired but no result was
                // materialized), so surface it as an internal error.
                let (prompt_tokens, completion_tokens, prefill_ns, generation_ns, ttft_ns,
                     batch_status, batch_incomplete_reason) =
                    if let Some(ref r) = batch_result {
                        let (status, reason) = match r.finish_reason {
                            1 => ("incomplete", Some("max_output_tokens")),
                            5 => ("incomplete", Some("cancelled")),
                            _ => ("completed", None),
                        };
                        (r.prompt_tokens, r.completion_tokens, r.prefill_ns,
                         r.generation_ns, r.ttft_ns, status, reason)
                    } else {
                        return Err(anyhow!(
                            "batch request {request_id} completed without a result"
                        ));
                    };

                log::debug!(target: "server::gen", "completed (batch): prompt_tokens={} completion_tokens={}",
                    prompt_tokens, completion_tokens);

                // The Zig batch path's completeRequest already called
                // commitGenerationResult which populated the conversation
                // with correct reasoning + message items. Read from the
                // conversation directly — no segment-based reconstruction.

                let all_json = chat
                    .to_responses_json(1)
                    .map_err(|e| anyhow!("failed to serialize output: {}", e))?;
                let all_items: serde_json::Value =
                    serde_json::from_str(&all_json).unwrap_or_else(|_| json!([]));
                let output_items = if let Some(arr) = all_items.as_array() {
                    serde_json::Value::Array(arr[pre_gen_count..].to_vec())
                } else {
                    json!([])
                };

                let model_info_json = {
                    let guard = backend.blocking_lock();
                    if let Some(ref b) = guard.backend {
                        build_model_info_json(b)
                    } else {
                        serde_json::Value::Null
                    }
                };

                Ok::<_, anyhow::Error>((
                    output_items, prompt_tokens, completion_tokens,
                    prefill_ns, generation_ns, ttft_ns, all_json, model_info_json,
                    batch_status, batch_incomplete_reason,
                ))
            } else {
                // --- Remote provider path (direct generate with backend mutex) ---
                let mut backend = backend.blocking_lock();
                let backend = backend
                    .backend
                    .as_mut()
                    .ok_or_else(|| anyhow!("no backend available"))?;

                let result = talu::router::generate(&chat, &[], backend, &cfg)
                    .map_err(|e| anyhow!("generation failed: {}", e))?;

                let prompt_tokens = result.prompt_tokens();
                let completion_tokens = result.completion_tokens();
                let prefill_ns = result.prefill_ns();
                let generation_ns = result.generation_ns();
                let ttft_ns = result.ttft_ns();
                log::debug!(target: "server::gen", "completed: prompt_tokens={} completion_tokens={}",
                    prompt_tokens, completion_tokens);

                let all_json = chat
                    .to_responses_json(1)
                    .map_err(|e| anyhow!("failed to serialize output: {}", e))?;
                let all_items: serde_json::Value =
                    serde_json::from_str(&all_json).unwrap_or_else(|_| json!([]));
                let output_items = if let Some(arr) = all_items.as_array() {
                    serde_json::Value::Array(arr[pre_gen_count..].to_vec())
                } else {
                    json!([])
                };

                let responses_json = all_json;
                let model_info_json = build_model_info_json(backend);

                Ok::<_, anyhow::Error>((
                    output_items, prompt_tokens, completion_tokens,
                    prefill_ns, generation_ns, ttft_ns, responses_json, model_info_json,
                    "completed", None::<&str>, // remote providers always complete
                ))
            }
        })
        .await
        .context("Generation task failed")??;

    // Generation completed; defuse the cancel guard so it does not signal
    // the (now-finished) blocking task.
    cancel_guard.defuse();

    // Store conversation for future `previous_response_id` lookups.
    let session_id_for_response = session_id.clone();
    let tenant_id_for_store = auth_ctx.map(|ctx| ctx.tenant_id.clone());
    {
        let mut store = state.response_store.lock().await;
        store.insert(
            response_id.clone(),
            StoredResponse {
                responses_json,
                tools_json: effective_tools.clone(),
                tool_choice_json: effective_tool_choice.clone(),
                session_id: Some(session_id),
                tenant_id: tenant_id_for_store,
            },
        );
    }

    let usage = UsageStats {
        input_tokens: prompt_tokens,
        output_tokens: completion_tokens,
        prefill_ns,
        generation_ns,
        ttft_ns: result_ttft_ns,
    };
    let mut response_value = build_response_resource_value(
        &response_id,
        &model_id,
        created_at,
        Some(now_unix_seconds()),
        output_items,
        effective_config
            .as_ref()
            .map(|e| e.max_tokens as i64)
            .or(max_output_tokens),
        effective_config
            .as_ref()
            .map(|e| e.temperature as f64)
            .unwrap_or_else(|| temperature.unwrap_or(0.0)),
        effective_config
            .as_ref()
            .map(|e| e.top_p as f64)
            .unwrap_or_else(|| top_p.unwrap_or(1.0)),
        effective_config
            .as_ref()
            .map(|e| Some(e.top_k as u32))
            .unwrap_or(top_k),
        effective_config
            .as_ref()
            .map(|e| e.presence_penalty as f64)
            .unwrap_or_else(|| presence_penalty.unwrap_or(0.0)),
        effective_config
            .as_ref()
            .map(|e| e.frequency_penalty as f64)
            .unwrap_or_else(|| frequency_penalty.unwrap_or(0.0)),
        logprobs.top_logprobs as i64,
        reasoning.effort.as_deref(),
        reasoning.summary.as_deref(),
        response_status,
        Some(&usage),
        effective_tools.as_ref(),
        effective_tool_choice.as_ref(),
        store,
        instructions.as_deref(),
        text_format.as_ref(),
    );

    // Set previous_response_id on the response resource.
    if let Some(ref prev_id) = previous_response_id {
        response_value["previous_response_id"] = json!(prev_id);
    }
    if !model_info_json.is_null() {
        response_value["model_info"] = model_info_json;
    }
    if let Some(reason) = incomplete_reason {
        response_value["incomplete_details"] = json!({
            "reason": reason
        });
    }

    if !strict_responses {
        // Include session_id (and project_id if set) in metadata so the UI
        // can adopt them for follow-ups.
        let mut meta = json!({ "session_id": session_id_for_response });
        if let Some(ref pid) = project_id {
            meta["project_id"] = json!(pid);
        }
        response_value["metadata"] = meta;
    }

    // Auto-generate a descriptive title for new conversations in the background.
    if !strict_responses && is_new_conversation && auto_title {
        let title_input = input_value
            .as_ref()
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        if let Some(input) = title_input {
            let title_backend = state.backend.clone();
            let title_bucket = bucket_path.clone();
            let title_session_id = session_id_for_response.clone();
            tokio::task::spawn_blocking(move || {
                let _ = generate_title(&title_backend, &title_bucket, &title_session_id, &input);
            });
        }
    }

    Ok(normalize_response_value(response_value))
}

async fn stream_response(
    state: Arc<AppState>,
    request: GenerationRequest,
    input_value: Option<serde_json::Value>,
    tools_json: Option<serde_json::Value>,
    tool_choice_json: Option<serde_json::Value>,
    previous_response_id: Option<String>,
    store: bool,
    request_session_id: Option<String>,
    prompt_id: Option<String>,
    project_id: Option<String>,
    strict_responses: bool,
    auth_ctx: Option<AuthContext>,
) -> Response<BoxBody> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Bytes>();
    let request_id = format!("req_{}", random_id());
    let response_id = format!("resp_{}", random_id());
    let message_id = format!("msg_{}", random_id());
    let created_at = now_unix_seconds();

    let request_max_output_tokens = request.max_output_tokens;
    let request_max_completion_tokens = request.max_completion_tokens;
    let request_max_reasoning_tokens = request.max_reasoning_tokens;
    let temperature = request.temperature;
    let top_p = request.top_p;
    let top_k = request.top_k;
    let presence_penalty = request.presence_penalty;
    let frequency_penalty = request.frequency_penalty;
    let seed = request.seed;
    let logprobs = request.logprobs;
    let reasoning = request.reasoning;
    let instructions = request.instructions;
    let text_format = request.text_format;
    // Resolve model ID without loading — loading happens inside spawn_blocking
    // so that progress events can be streamed through the SSE channel.
    let model_id = match resolve_model_id(state.clone(), request.model.clone()).await {
        Ok(val) => val,
        Err(err) => {
            return api_error(
                strict_responses,
                StatusCode::INTERNAL_SERVER_ERROR,
                "model_error",
                &format!("{err}"),
            )
        }
    };

    // Load previous conversation state if chaining.
    let requester_tenant = auth_ctx.as_ref().map(|ctx| ctx.tenant_id.as_str());
    let prev_state = if let Some(ref prev_id) = previous_response_id {
        let rstore = state.response_store.lock().await;
        rstore.get(prev_id).and_then(|s| {
            if s.tenant_id.as_deref() == requester_tenant {
                Some((
                    s.responses_json.clone(),
                    s.tools_json.clone(),
                    s.tool_choice_json.clone(),
                    s.session_id.clone(),
                ))
            } else {
                None
            }
        })
    } else {
        None
    };

    // Merge tools from previous if not specified in this request.
    let effective_tools = tools_json.or_else(|| prev_state.as_ref().and_then(|s| s.1.clone()));
    let effective_tool_choice =
        tool_choice_json.or_else(|| prev_state.as_ref().and_then(|s| s.2.clone()));

    // Resolve session ID: explicit request > previous response chain > new UUID.
    let session_id = request_session_id
        .or_else(|| prev_state.as_ref().and_then(|s| s.3.clone()))
        .unwrap_or_else(|| uuid::Uuid::new_v4().hyphenated().to_string());

    // Resolve bucket path for persistence (only when store=true).
    let bucket_path = if store {
        state
            .bucket_path
            .as_ref()
            .map(|base| match auth_ctx.as_ref() {
                Some(ctx) => base.join(&ctx.storage_prefix),
                None => base.to_path_buf(),
            })
    } else {
        None
    };

    // Load auto_title setting from bucket config.
    let auto_title = bucket_path.as_ref().map_or(false, |bp| {
        bucket_settings::load_bucket_settings(bp)
            .map(|s| s.auto_title)
            .unwrap_or(true)
    });

    // Ensure storage directory exists.
    if let Some(ref bp) = bucket_path {
        if let Err(e) = std::fs::create_dir_all(bp) {
            log::warn!(target: "server::gen", "failed to create storage directory: {}", e);
        }
    }

    // Load bucket settings for fallback values (max_output_tokens + per-model overrides).
    // Use state.bucket_path (not bucket_path) so fallback works regardless of store flag.
    let fallback_settings = state
        .bucket_path
        .as_ref()
        .map(|base| match auth_ctx.as_ref() {
            Some(ctx) => base.join(&ctx.storage_prefix),
            None => base.to_path_buf(),
        })
        .and_then(|bp| bucket_settings::load_bucket_settings(&bp).ok());

    let max_output_tokens = request_max_output_tokens.or_else(|| {
        fallback_settings
            .as_ref()
            .and_then(|s| s.max_output_tokens.map(|v| v as i64))
    });

    // Apply per-model sampling overrides as fallbacks when the request doesn't specify them.
    let model_overrides = fallback_settings
        .as_ref()
        .and_then(|s| s.models.get(&model_id).cloned());
    let temperature = temperature.or_else(|| model_overrides.as_ref().and_then(|o| o.temperature));
    let top_p = top_p.or_else(|| model_overrides.as_ref().and_then(|o| o.top_p));

    // Resolve effective generation config using model sampling presets.
    let effective_gen = talu::model::resolve_effective_generation_config(
        &model_id,
        &talu::EffectiveGenConfigRequest {
            temperature: temperature.map(|t| t as f32),
            top_k: top_k.map(|k| k as usize),
            top_p: top_p.map(|p| p as f32),
            presence_penalty: presence_penalty.map(|p| p as f32),
            frequency_penalty: frequency_penalty.map(|p| p as f32),
            seed: seed.unwrap_or(0),
            max_tokens: max_output_tokens.map(|t| t as usize).unwrap_or(8192),
            ..Default::default()
        },
    );
    if let Err(ref e) = effective_gen {
        log::warn!(target: "server::gen", "resolve_effective_generation_config: {e} (using raw request params)");
    }
    let effective_config = effective_gen.ok();

    // If prompt_id is provided, fetch the document and extract system prompt.
    let system_prompt_from_doc: Option<String> = if let Some(ref pid) = prompt_id {
        if let Some(ref bp) = bucket_path {
            match DocumentsHandle::open(&bp.join("tables").join("documents")) {
                Ok(docs) => match docs.get(pid) {
                    Ok(Some(doc)) => {
                        // Parse doc_json to extract system prompt
                        // Try direct fields first (current UI format), then nested under "data" (legacy)
                        if let Ok(envelope) =
                            serde_json::from_str::<serde_json::Value>(&doc.doc_json)
                        {
                            envelope
                                .get("system")
                                .or_else(|| envelope.get("system_prompt"))
                                .or_else(|| envelope.get("data").and_then(|d| d.get("system")))
                                .or_else(|| {
                                    envelope.get("data").and_then(|d| d.get("system_prompt"))
                                })
                                .and_then(|s| s.as_str())
                                .map(|s| s.to_string())
                        } else {
                            None
                        }
                    }
                    Ok(None) => {
                        return api_error(
                            strict_responses,
                            StatusCode::BAD_REQUEST,
                            "invalid_request",
                            &format!("prompt_id '{}' not found", pid),
                        );
                    }
                    Err(DocumentError::DocumentNotFound(_)) => {
                        return api_error(
                            strict_responses,
                            StatusCode::BAD_REQUEST,
                            "invalid_request",
                            &format!("prompt_id '{}' not found", pid),
                        );
                    }
                    Err(e) => {
                        return api_error(
                            strict_responses,
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "storage_error",
                            &format!("failed to fetch prompt document: {}", e),
                        );
                    }
                },
                Err(e) => {
                    return api_error(
                        strict_responses,
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "storage_error",
                        &format!("failed to open documents store: {}", e),
                    );
                }
            }
        } else {
            return api_error(
                strict_responses,
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "prompt_id requires storage to be configured",
            );
        }
    } else {
        None
    };
    let effective_system_prompt = if strict_responses {
        instructions.clone()
    } else {
        system_prompt_from_doc.clone()
    };

    // Only pass prev_json for in-memory chaining when storage is NOT active.
    // When storage is active, set_storage_db auto-loads existing items.
    let prev_json = if bucket_path.is_none() {
        prev_state.map(|s| s.0)
    } else {
        let _ = prev_state; // consumed
        None
    };

    // Create a stop flag for cancellation on client disconnect or server shutdown.
    let stop_flag = Arc::new(AtomicBool::new(false));
    let stop_flag_for_gen = stop_flag.clone();
    if let Ok(mut flags) = state.active_stop_flags.lock() {
        flags.retain(|w| w.strong_count() > 0);
        flags.push(Arc::downgrade(&stop_flag));
    }

    // Prepare input for the blocking task.
    let input_json = input_value
        .as_ref()
        .filter(|v| v.is_array())
        .map(|v| serde_json::to_string(v).unwrap_or_default());
    let input_string = input_value
        .as_ref()
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    // File storage path for resolving file references (always available if bucket exists).
    let file_storage_path = state
        .bucket_path
        .as_ref()
        .map(|base| match auth_ctx.as_ref() {
            Some(ctx) => base.join(&ctx.storage_prefix),
            None => base.to_path_buf(),
        });

    // Clones for the blocking task to store response after completion.
    let state_for_store = state.clone();
    let session_id_for_store = session_id.clone();
    let tenant_id_for_store = auth_ctx.as_ref().map(|ctx| ctx.tenant_id.clone());
    let store_tools = effective_tools.clone();
    let store_tool_choice = effective_tool_choice.clone();
    let response_id_for_store = response_id.clone();

    // A brand-new conversation has no previous_response_id and no request_session_id.
    // For these, we persist session metadata BEFORE sending response.created so
    // the session is visible in list_sessions when the client refreshes the sidebar.
    let is_new_conversation = previous_response_id.is_none();

    let backend = state.backend.clone();
    let tools_for_events = effective_tools.clone();
    let tool_choice_for_events = effective_tool_choice.clone();
    let text_format_for_events = text_format.clone();
    let reasoning_for_events = reasoning.clone();

    // Clone project_id for the streaming blocking task.
    let project_id_for_stream = project_id.clone();

    // Clones for post-generation title generation.
    let title_input = input_string.clone();
    let title_bucket = bucket_path.clone();
    let title_session_id = session_id.clone();
    let title_backend = state.backend.clone();
    let events_tenant_id = auth_ctx.as_ref().map(|ctx| ctx.tenant_id.clone());
    let events_request_id = request_id.clone();
    let events_response_id = response_id.clone();
    let events_session_id = session_id.clone();

    let kv_root = state
        .bucket_path
        .as_ref()
        .map(|b| b.join("kv").to_string_lossy().into_owned());

    let previous_response_id_for_ctx = previous_response_id.clone();
    tokio::task::spawn_blocking(move || {
        let mut seq = 0u64;
        if strict_responses {
            let _ = send_event(
                &tx,
                "response.queued",
                json!({
                    "type": "response.queued",
                    "sequence_number": seq,
                    "response": {
                        "id": &response_id,
                        "object": "response",
                        "created_at": created_at,
                        "completed_at": null,
                        "status": "queued",
                        "incomplete_details": null,
                        "model": &model_id,
                        "previous_response_id": previous_response_id_for_ctx,
                        "instructions": instructions,
                        "output": [],
                        "error": null,
                        "tools": tools_for_events.as_ref().cloned().unwrap_or_else(|| json!([])),
                        "tool_choice": tool_choice_for_events.as_ref().cloned().unwrap_or_else(|| json!("none")),
                        "truncation": "auto",
                        "parallel_tool_calls": false,
                        "text": { "format": response_text_format_value(text_format_for_events.as_ref()) },
                        "top_p": top_p.unwrap_or(1.0),
                        "top_k": top_k.unwrap_or(0),
                        "presence_penalty": presence_penalty.unwrap_or(0.0),
                        "frequency_penalty": frequency_penalty.unwrap_or(0.0),
                        "top_logprobs": logprobs.top_logprobs as i64,
                        "temperature": temperature.unwrap_or(0.0),
                        "reasoning": {
                            "effort": reasoning_for_events.effort.as_deref(),
                            "summary": reasoning_for_events.summary.as_deref()
                        },
                        "usage": null,
                        "max_output_tokens": max_output_tokens,
                        "max_tool_calls": null,
                        "store": store,
                        "background": false,
                        "service_tier": "default",
                        "metadata": {},
                        "safety_identifier": null,
                        "prompt_cache_key": null
                    }
                }),
            );
            seq += 1;
        }

        // Load/switch backend if needed, emitting progress events through SSE.
        {
            let mut guard = backend.blocking_lock();
            let needs_load =
                guard.current_model.as_deref() != Some(&model_id) || guard.backend.is_none();
            if needs_load {
                let progress_tenant_id = events_tenant_id.clone();
                let progress_request_id = events_request_id.clone();
                let progress_response_id = events_response_id.clone();
                let progress_session_id = events_session_id.clone();
                let callback: Option<talu::LoadProgressCallback> =
                    Some(Box::new(move |p: talu::LoadProgress| {
                        events::publish_inference_progress(
                            progress_tenant_id.as_deref(),
                            Some(progress_request_id.as_str()),
                            Some(progress_response_id.as_str()),
                            Some(progress_session_id.as_str()),
                            &p.label,
                            p.current,
                            p.total,
                        );
                    }));
                match provider::create_backend_for_model_with_progress(
                    &model_id,
                    kv_root.as_deref(),
                    callback,
                ) {
                    Ok(new_backend) => {
                        // Recreate batch scheduler for new backend.
                        let new_sched =
                            crate::server::batch_scheduler::SchedulerState::new(&new_backend, None)
                                .ok()
                                .map(Arc::new);
                        let old_sched =
                            if let Ok(mut sched_guard) = state_for_store.batch_scheduler.lock() {
                                let old = sched_guard.take();
                                *sched_guard = new_sched;
                                old
                            } else {
                                None
                            };
                        // Take old backend before replacing — it must outlive the
                        // old scheduler's step thread (BatchHandle holds an engine
                        // pointer into it).
                        let old_backend = guard.backend.take();
                        guard.backend = Some(new_backend);
                        guard.current_model = Some(model_id.clone());
                        drop(guard); // Release backend lock before blocking drain.
                                     // Drain old scheduler in background: the step thread
                                     // finishes active requests, then shuts down. The old
                                     // backend is kept alive until after the join.
                        if old_sched.is_some() || old_backend.is_some() {
                            // Join any previous drain thread before spawning
                            // a new one to prevent unbounded accumulation.
                            if let Ok(mut dt) = state_for_store.drain_thread.lock() {
                                if let Some(prev) = dt.take() {
                                    let _ = prev.join();
                                }
                            }
                            let drain_handle = std::thread::Builder::new()
                                .name("sched-drain".into())
                                .spawn(move || {
                                    if let Some(s) = old_sched {
                                        s.shutdown();
                                    }
                                    drop(old_backend);
                                });
                            match drain_handle {
                                Ok(handle) => {
                                    if let Ok(mut dt) = state_for_store.drain_thread.lock() {
                                        *dt = Some(handle);
                                    }
                                }
                                Err(e) => {
                                    log::error!(target: "server::gen",
                                        "failed to spawn drain thread: {e}");
                                }
                            }
                        }
                    }
                    Err(e) => {
                        if strict_responses {
                            let _ = send_event(
                                &tx,
                                "error",
                                json!({
                                    "type": "error",
                                    "sequence_number": seq,
                                    "error": {
                                        "type": "server_error",
                                        "code": "model_error",
                                        "message": format!("Failed to load model: {e}"),
                                        "param": null
                                    }
                                }),
                            );
                            seq += 1;
                        }
                        let _ = send_event(
                            &tx,
                            "response.failed",
                            json!({
                                "type": "response.failed",
                                "sequence_number": seq,
                                "response": {
                                    "error": {
                                        "code": "model_error",
                                        "message": format!("Failed to load model: {e}"),
                                    }
                                }
                            }),
                        );
                        return;
                    }
                }
            }
        }

        // Persist session metadata for new conversations BEFORE sending
        // response.created — the client refreshes the sidebar on receipt and
        // the session must already be in list_sessions at that point.
        if !strict_responses && is_new_conversation && bucket_path.is_some() {
            if let Ok(temp_chat) = ChatHandle::new(effective_system_prompt.as_deref()) {
                if let Some(bp_str) = bucket_path.as_ref().and_then(|p| p.to_str()) {
                    if temp_chat.set_storage_db(bp_str, &session_id).is_ok() {
                        let t = input_string.as_deref().unwrap_or("Untitled");
                        let title: String = t.chars().take(47).collect();
                        let _ = temp_chat.notify_session_update_ex(
                            Some(&model_id),
                            Some(&title),
                            Some("active"),
                            prompt_id.as_deref(),
                            project_id_for_stream.as_deref(),
                        );
                    }
                }
            }
        }

        let mut created_response = normalize_response_value(build_response_resource_value(
            &response_id,
            &model_id,
            created_at,
            None,
            json!([]),
            max_output_tokens,
            temperature.unwrap_or(0.0),
            top_p.unwrap_or(1.0),
            top_k,
            presence_penalty.unwrap_or(0.0),
            frequency_penalty.unwrap_or(0.0),
            logprobs.top_logprobs as i64,
            reasoning_for_events.effort.as_deref(),
            reasoning_for_events.summary.as_deref(),
            "in_progress",
            None,
            tools_for_events.as_ref(),
            tool_choice_for_events.as_ref(),
            store,
            instructions.as_deref(),
            text_format_for_events.as_ref(),
        ));
        if let Some(ref prev_id) = previous_response_id_for_ctx {
            created_response["previous_response_id"] = json!(prev_id);
        }
        if !strict_responses {
            // Include session_id (and project_id if set) in the very first
            // event so the UI can track the conversation immediately.
            let mut meta = json!({ "session_id": session_id });
            if let Some(ref pid) = project_id_for_stream {
                meta["project_id"] = json!(pid);
            }
            created_response["metadata"] = meta;
        }
        let _ = send_event(
            &tx,
            "response.created",
            json!({
                "type": "response.created",
                "sequence_number": seq,
                "response": &created_response
            }),
        );
        seq += 1;

        let _ = send_event(
            &tx,
            "response.in_progress",
            json!({
                "type": "response.in_progress",
                "sequence_number": seq,
                "response": created_response
            }),
        );
        seq += 1;

        let ctx = Arc::new(std::sync::Mutex::new(StreamCtx {
            tx,
            seq,
            response_id,
            stop_flag: stop_flag_for_gen.clone(),
            output_index: 0,
            content_index: 0,
            cur_item_type: None,
            cur_content_type: None,
            accumulated_text: String::new(),
            accumulated_logprobs: Vec::new(),
            accumulated_annotations: Vec::new(),
            annotation_keys: HashSet::new(),
            output_items: Vec::new(),
            cur_item_id: message_id,
            tools_json: tools_for_events,
            tool_choice_json: tool_choice_for_events,
            tenant_id: auth_ctx.as_ref().map(|ctx| ctx.tenant_id.clone()),
            user_id: auth_ctx.and_then(|ctx| ctx.user_id),
            session_id: session_id.clone(),
            strict_responses,
            request_store: store,
            previous_response_id: previous_response_id_for_ctx.clone(),
            instructions: instructions.clone(),
            text_format: text_format_for_events.clone(),
            project_id: project_id.clone(),
            top_logprobs: logprobs.top_logprobs,
            top_k: effective_config
                .as_ref()
                .map(|e| Some(e.top_k as u32))
                .unwrap_or(top_k),
            presence_penalty: effective_config
                .as_ref()
                .map(|e| e.presence_penalty as f64)
                .unwrap_or_else(|| presence_penalty.unwrap_or(0.0)),
            frequency_penalty: effective_config
                .as_ref()
                .map(|e| e.frequency_penalty as f64)
                .unwrap_or_else(|| frequency_penalty.unwrap_or(0.0)),
            reasoning_effort: reasoning_for_events.effort.clone(),
            reasoning_summary: reasoning_for_events.summary.clone(),
            coalesce_buf: String::new(),
            coalesce_event: String::new(),
            coalesce_start: None,
            coalesce_text_start: 0,
            last_tokens_generated: 0,
            last_elapsed_ns: 0,
        }));
        let ctx_for_complete = ctx.clone();

        let batch_sched = state_for_store.batch_scheduler.lock().unwrap().clone();
        let gen_result = if let Some(ref sched) = batch_sched {
            run_batch_streaming_generation(
                sched,
                backend,
                input_json,
                input_string,
                store_tools.clone(),
                store_tool_choice.clone(),
                text_format_for_events.clone(),
                prev_json,
                bucket_path,
                session_id,
                model_id.clone(),
                max_output_tokens,
                request_max_completion_tokens,
                request_max_reasoning_tokens,
                temperature,
                top_p,
                top_k,
                seed,
                presence_penalty,
                frequency_penalty,
                effective_system_prompt,
                prompt_id,
                project_id_for_stream,
                file_storage_path,
                ctx,
                stop_flag_for_gen,
                reasoning_for_events.effort.clone(),
                events_tenant_id.clone(),
                Some(events_request_id.clone()),
                Some(events_response_id.clone()),
                Some(events_session_id.clone()),
                effective_config,
            )
        } else {
            run_streaming_generation(
                backend,
                input_json,
                input_string,
                store_tools.clone(),
                store_tool_choice.clone(),
                text_format_for_events.clone(),
                prev_json,
                bucket_path,
                session_id,
                model_id.clone(),
                max_output_tokens,
                request_max_completion_tokens,
                request_max_reasoning_tokens,
                temperature,
                top_p,
                top_k,
                seed,
                presence_penalty,
                frequency_penalty,
                effective_system_prompt,
                prompt_id,
                project_id_for_stream,
                file_storage_path,
                ctx,
                stop_flag_for_gen,
                reasoning_for_events.effort.clone(),
                events_tenant_id.clone(),
                Some(events_request_id.clone()),
                Some(events_response_id.clone()),
                Some(events_session_id.clone()),
                effective_config,
            )
        };

        // Store conversation for chaining after streaming completes.
        if let Ok(ref r) = gen_result {
            let mut store = state_for_store.response_store.blocking_lock();
            store.insert(
                response_id_for_store.clone(),
                StoredResponse {
                    responses_json: r.responses_json.clone(),
                    tools_json: store_tools,
                    tool_choice_json: store_tool_choice,
                    session_id: Some(session_id_for_store),
                    tenant_id: tenant_id_for_store,
                },
            );
        }

        if let Ok(mut guard) = ctx_for_complete.lock() {
            let _ = guard.flush_completion(
                model_id,
                created_at,
                max_output_tokens,
                temperature.unwrap_or(0.0),
                top_p.unwrap_or(1.0),
                gen_result,
            );
        };

        // Auto-generate a descriptive title for new conversations.
        if !strict_responses && is_new_conversation && auto_title {
            if let Some(ref input) = title_input {
                let _ = generate_title(&title_backend, &title_bucket, &title_session_id, input);
            }
        }
    });

    let stream =
        UnboundedReceiverStream::new(rx).map(|chunk| Ok::<_, Infallible>(Frame::data(chunk)));
    let body = StreamBody::new(stream).boxed();

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .header("connection", "keep-alive")
        .body(body)
        .unwrap()
}

/// Run streaming generation via `talu::router::generate_stream` (callback API).
///
/// Used for remote/provider backends that do not have a batch scheduler.
/// Local backends use [`run_batch_streaming_generation`] instead.
fn run_streaming_generation(
    backend: Arc<tokio::sync::Mutex<crate::server::state::BackendState>>,
    input_json: Option<String>,
    input_string: Option<String>,
    tools_json: Option<serde_json::Value>,
    tool_choice_json: Option<serde_json::Value>,
    _text_format: Option<TextFormatConfig>,
    prev_json: Option<String>,
    bucket_path: Option<std::path::PathBuf>,
    session_id: String,
    model_id: String,
    max_output_tokens: Option<i64>,
    max_completion_tokens: Option<i64>,
    max_reasoning_tokens: Option<i64>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u32>,
    seed: Option<u64>,
    presence_penalty: Option<f64>,
    frequency_penalty: Option<f64>,
    system_prompt: Option<String>,
    prompt_id: Option<String>,
    project_id: Option<String>,
    file_storage_path: Option<std::path::PathBuf>,
    ctx: Arc<std::sync::Mutex<StreamCtx>>,
    stop_flag: Arc<AtomicBool>,
    reasoning_effort: Option<String>,
    event_tenant_id: Option<String>,
    event_request_id: Option<String>,
    event_response_id: Option<String>,
    event_session_id: Option<String>,
    effective_config: Option<talu::model::EffectiveGenConfig>,
) -> Result<StreamGenResult> {
    let mut backend = backend.blocking_lock();
    let backend = backend
        .backend
        .as_mut()
        .ok_or_else(|| anyhow!("no backend available"))?;
    // Create ChatHandle with system prompt if prompt_id was provided
    log::trace!(target: "server::gen", "ChatHandle::new(system_prompt={})", system_prompt.is_some());
    let chat = ChatHandle::new(system_prompt.as_deref())?;

    // Enable storage persistence if store=true and bucket is configured.
    // set_storage_db auto-loads any existing items for this session.
    let has_storage = if let Some(ref bp) = bucket_path {
        if let Some(bp_str) = bp.to_str() {
            log::trace!(target: "server::gen", "set_storage_db({:?}, {})", bp_str, session_id);
            chat.set_storage_db(bp_str, &session_id)
                .map_err(|e| anyhow!("failed to set storage: {}", e))?;
            true
        } else {
            false
        }
    } else {
        false
    };

    // Restore previous conversation from in-memory JSON only when storage
    // is not active — set_storage_db already loaded persisted items.
    if !has_storage {
        if let Some(ref prev) = prev_json {
            log::trace!(target: "server::gen", "load_responses_json(prev, {} bytes)", prev.len());
            chat.load_responses_json(prev)
                .map_err(|e| anyhow!("failed to load previous conversation: {}", e))?;
        }
    }

    // Load input items into the conversation, resolving file references.
    // The Zig protocol parser stores input_image parts in conversation
    // items; the generate function extracts them for the vision encoder.
    if let Some(ref json) = input_json {
        log::trace!(target: "server::gen", "resolve_file_references(input, {} bytes)", json.len());
        let resolved = match file_storage_path.as_deref() {
            Some(sp) => resolve_file_references(json, sp)?,
            None => json.clone(),
        };
        log::trace!(target: "server::gen", "load_responses_json(input, {} bytes)", resolved.len());
        chat.load_responses_json(&resolved)
            .map_err(|e| anyhow!("failed to load input: {}", e))?;
    } else if let Some(ref text) = input_string {
        log::trace!(target: "server::gen", "append_user_message({} chars)", text.len());
        chat.append_user_message(text)
            .map_err(|e| anyhow!("failed to append user message: {}", e))?;
    }
    let mut cfg = talu::router::GenerateConfig::default();
    if let Some(ref eff) = effective_config {
        cfg.max_tokens = eff.max_tokens;
        cfg.temperature = eff.temperature;
        cfg.top_k = eff.top_k;
        cfg.top_p = eff.top_p;
        cfg.min_p = eff.min_p;
        cfg.repetition_penalty = eff.repetition_penalty;
        cfg.presence_penalty = eff.presence_penalty;
        cfg.frequency_penalty = eff.frequency_penalty;
        cfg.seed = eff.seed;
    } else {
        // Fallback for remote/provider models where presets are unavailable.
        if let Some(max_tokens) = max_output_tokens {
            cfg.max_tokens = max_tokens as usize;
        }
        if let Some(temp) = temperature {
            cfg.temperature = temp as f32;
        }
        if let Some(top_p) = top_p {
            cfg.top_p = top_p as f32;
        }
        if let Some(top_k) = top_k {
            cfg.top_k = top_k as usize;
        }
        if let Some(seed) = seed {
            cfg.seed = seed;
        }
        if let Some(pp) = presence_penalty {
            cfg.presence_penalty = pp as f32;
        }
        if let Some(fp) = frequency_penalty {
            cfg.frequency_penalty = fp as f32;
        }
    }
    cfg.tools_json = tools_json.map(|v| v.to_string());
    cfg.tool_choice = tool_choice_json.map(|v| v.to_string());
    cfg.reasoning_effort = reasoning_effort;
    if let Some(mct) = max_completion_tokens {
        cfg.max_completion_tokens = Some(mct as usize);
    }
    if let Some(mrt) = max_reasoning_tokens {
        cfg.max_reasoning_tokens = Some(mrt as usize);
    }

    // Pass the stop flag for cooperative cancellation.
    // Clone before moving into cfg so the callback can also check it.
    let stop_flag_for_cb = stop_flag.clone();
    cfg.stop_flag = Some(stop_flag);

    cfg.prefill_progress = Some(Box::new(move |completed: usize, total: usize| {
        events::publish_inference_progress(
            event_tenant_id.as_deref(),
            event_request_id.as_deref(),
            event_response_id.as_deref(),
            event_session_id.as_deref(),
            "prefill",
            completed as u64,
            total as u64,
        );
    }));

    log::debug!(target: "server::gen", "generating: model={} max_tokens={:?} temp={:?} top_p={:?}",
        model_id, max_output_tokens, temperature, top_p);

    let ctx_clone = ctx.clone();
    let callback: talu::router::StreamCallback = Box::new(move |token| {
        // Check stop flag before sending — stop immediately on Ctrl+C or client disconnect.
        if stop_flag_for_cb.load(Ordering::Acquire) {
            return false;
        }
        if let Ok(mut guard) = ctx_clone.lock() {
            let _ = guard.send_delta(token);
        }
        true
    });

    // Record item count before generation to extract only new output items.
    let pre_gen_count = chat.item_count();

    // Persist session metadata BEFORE generation so the conversation appears
    // in list_sessions immediately — the user can navigate to it while
    // generation is still in progress.
    // A brand-new chat has at most 2 items before generation: an optional
    // system message (from prompt_id) + the user message.  Chained requests
    // have more items and must skip this to avoid overwriting the title.
    if bucket_path.is_some() && pre_gen_count <= 2 {
        let t = input_string.as_deref().unwrap_or("Untitled");
        let title: String = t.chars().take(47).collect();
        let _ = chat.notify_session_update_ex(
            Some(&model_id),
            Some(&title),
            Some("active"),
            prompt_id.as_deref(),
            project_id.as_deref(),
        );
    }

    log::trace!(target: "server::gen", "generate_stream(items={}, cfg={{max_tokens={}, temp={}, top_p={}}})",
        pre_gen_count, cfg.max_tokens, cfg.temperature, cfg.top_p);
    let stream_result = talu::router::generate_stream(&chat, &[], backend, &cfg, callback)
        .map_err(|e| anyhow!("generation failed: {}", e))?;

    // Serialize all items and full conversation for storage/chaining.
    let all_json = chat
        .to_responses_json(1)
        .map_err(|e| anyhow!("failed to serialize output: {}", e))?;
    let output_items = serde_json::from_str::<serde_json::Value>(&all_json)
        .ok()
        .and_then(|v| v.as_array().map(|arr| arr[pre_gen_count..].to_vec()))
        .map(serde_json::Value::Array)
        .unwrap_or_else(|| json!([]));

    log::debug!(target: "server::gen", "completed: prompt_tokens={} completion_tokens={}",
        stream_result.prompt_tokens, stream_result.completion_tokens);

    let model_info_json = build_model_info_json(backend);

    Ok(StreamGenResult {
        output_items,
        usage: UsageStats {
            input_tokens: stream_result.prompt_tokens,
            output_tokens: stream_result.completion_tokens,
            prefill_ns: stream_result.prefill_ns,
            generation_ns: stream_result.generation_ns,
            ttft_ns: stream_result.ttft_ns,
        },
        finish_reason: stream_result.finish_reason,
        responses_json: all_json,
        model_info_json,
    })
}

/// Run streaming generation via the batch scheduler (concurrent GPU decode).
///
/// Used for local backends. Routes through the batch scheduler instead of
/// acquiring the backend mutex. The ChatHandle is borrowed (not consumed)
/// so it remains available for post-generation serialization.
fn run_batch_streaming_generation(
    scheduler: &crate::server::batch_scheduler::SchedulerState,
    backend: Arc<tokio::sync::Mutex<crate::server::state::BackendState>>,
    input_json: Option<String>,
    input_string: Option<String>,
    tools_json: Option<serde_json::Value>,
    tool_choice_json: Option<serde_json::Value>,
    _text_format: Option<TextFormatConfig>,
    prev_json: Option<String>,
    bucket_path: Option<std::path::PathBuf>,
    session_id: String,
    model_id: String,
    max_output_tokens: Option<i64>,
    max_completion_tokens: Option<i64>,
    max_reasoning_tokens: Option<i64>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u32>,
    seed: Option<u64>,
    presence_penalty: Option<f64>,
    frequency_penalty: Option<f64>,
    system_prompt: Option<String>,
    prompt_id: Option<String>,
    project_id: Option<String>,
    file_storage_path: Option<std::path::PathBuf>,
    ctx: Arc<std::sync::Mutex<StreamCtx>>,
    stop_flag: Arc<AtomicBool>,
    reasoning_effort: Option<String>,
    _event_tenant_id: Option<String>,
    _event_request_id: Option<String>,
    _event_response_id: Option<String>,
    _event_session_id: Option<String>,
    effective_config: Option<talu::model::EffectiveGenConfig>,
) -> Result<StreamGenResult> {
    // --- Conversation setup (same as run_streaming_generation) ---
    let chat = ChatHandle::new(system_prompt.as_deref())?;

    let has_storage = if let Some(ref bp) = bucket_path {
        if let Some(bp_str) = bp.to_str() {
            chat.set_storage_db(bp_str, &session_id)
                .map_err(|e| anyhow!("failed to set storage: {}", e))?;
            true
        } else {
            false
        }
    } else {
        false
    };

    // Restore previous conversation from in-memory JSON only when storage
    // is not active — set_storage_db already loaded persisted items.
    if !has_storage {
        if let Some(ref prev) = prev_json {
            chat.load_responses_json(prev)
                .map_err(|e| anyhow!("failed to load previous conversation: {}", e))?;
        }
    }

    log::info!(target: "server::gen",
        "batch_stream setup: has_storage={} items_after_load={} has_prev={}",
        has_storage, chat.item_count(), prev_json.is_some());

    if let Some(ref json) = input_json {
        let resolved = match file_storage_path.as_deref() {
            Some(sp) => resolve_file_references(json, sp)?,
            None => json.clone(),
        };
        chat.load_responses_json(&resolved)
            .map_err(|e| anyhow!("failed to load input: {}", e))?;
    } else if let Some(ref text) = input_string {
        chat.append_user_message(text)
            .map_err(|e| anyhow!("failed to append user message: {}", e))?;
    }

    let pre_gen_count = chat.item_count();
    log::info!(target: "server::gen",
        "batch_stream pre-gen: items={}", pre_gen_count);

    if bucket_path.is_some() && pre_gen_count <= 2 {
        let t = input_string.as_deref().unwrap_or("Untitled");
        let title: String = t.chars().take(47).collect();
        let _ = chat.notify_session_update_ex(
            Some(&model_id),
            Some(&title),
            Some("active"),
            prompt_id.as_deref(),
            project_id.as_deref(),
        );
    }

    // --- Build GenerateConfig ---
    let mut cfg = talu::router::GenerateConfig::default();
    if let Some(ref eff) = effective_config {
        cfg.max_tokens = eff.max_tokens;
        cfg.temperature = eff.temperature;
        cfg.top_k = eff.top_k;
        cfg.top_p = eff.top_p;
        cfg.min_p = eff.min_p;
        cfg.repetition_penalty = eff.repetition_penalty;
        cfg.presence_penalty = eff.presence_penalty;
        cfg.frequency_penalty = eff.frequency_penalty;
        cfg.seed = eff.seed;
    } else {
        if let Some(max_tokens) = max_output_tokens {
            cfg.max_tokens = max_tokens as usize;
        }
        if let Some(temp) = temperature {
            cfg.temperature = temp as f32;
        }
        if let Some(top_p) = top_p {
            cfg.top_p = top_p as f32;
        }
        if let Some(top_k) = top_k {
            cfg.top_k = top_k as usize;
        }
        if let Some(seed) = seed {
            cfg.seed = seed;
        }
        if let Some(pp) = presence_penalty {
            cfg.presence_penalty = pp as f32;
        }
        if let Some(fp) = frequency_penalty {
            cfg.frequency_penalty = fp as f32;
        }
    }
    cfg.tools_json = tools_json.map(|v| v.to_string());
    cfg.tool_choice = tool_choice_json.map(|v| v.to_string());
    cfg.reasoning_effort = reasoning_effort;
    if let Some(mct) = max_completion_tokens {
        cfg.max_completion_tokens = Some(mct as usize);
    }
    if let Some(mrt) = max_reasoning_tokens {
        cfg.max_reasoning_tokens = Some(mrt as usize);
    }
    cfg.stop_flag = Some(stop_flag.clone());

    // Capture generation config before submit moves cfg.
    let gen_temperature = cfg.temperature;
    let gen_top_p = cfg.top_p;
    let gen_top_k = cfg.top_k;
    let gen_min_p = cfg.min_p;
    let gen_max_tokens = cfg.max_tokens;
    let gen_repetition_penalty = cfg.repetition_penalty;
    let gen_seed = cfg.seed;

    // --- Submit to batch scheduler ---
    let (request_id, event_rx) = scheduler
        .submit(&chat, cfg, stop_flag.clone())
        .map_err(|e| anyhow!("batch submit failed: {}", e))?;

    // --- Consume events, feed to StreamCtx ---
    loop {
        if stop_flag.load(Ordering::Acquire) {
            scheduler.cancel(request_id);
            break;
        }
        match event_rx.recv() {
            Ok(event) => {
                let is_final = event.is_final;

                // Map BatchEvent to StreamToken for StreamCtx.
                if !event.text.is_empty() {
                    let item_type = talu_sys::ItemType::from(event.item_type);
                    let content_type = talu_sys::ContentType::from(event.content_type);
                    let token = talu::router::StreamToken {
                        text: &event.text,
                        item_type,
                        content_type,
                        tokens_generated: event.tokens_generated,
                        elapsed_ns: event.timestamp_ns as u64,
                    };
                    if let Ok(mut guard) = ctx.lock() {
                        let _ = guard.send_delta(&token);
                    }
                }

                if is_final {
                    break;
                }
            }
            Err(_) => break, // Channel closed
        }
    }

    // --- Build result from BatchResult ---
    let batch_result = scheduler.take_result(request_id);

    let (prompt_tokens, completion_tokens, prefill_ns, generation_ns, ttft_ns, finish_reason) =
        if let Some(ref r) = batch_result {
            (
                r.prompt_tokens,
                r.completion_tokens,
                r.prefill_ns,
                r.generation_ns,
                r.ttft_ns,
                match r.finish_reason {
                    0 => FinishReason::EosToken,
                    1 => FinishReason::Length,
                    2 => FinishReason::StopSequence,
                    3 => FinishReason::ToolCalls,
                    4 => FinishReason::ContentFilter,
                    5 => FinishReason::Cancelled,
                    _ => FinishReason::EosToken,
                },
            )
        } else {
            // No result means cancellation or scheduler drain — surface as
            // incomplete/cancelled rather than masking under EOS.
            log::warn!(target: "server::gen",
                "batch request {request_id}: take_result returned None \
                 (cancelled or internal fault)");
            (0, 0, 0, 0, 0, FinishReason::Cancelled)
        };

    // Patch function_call items in StreamCtx with metadata from BatchResult.
    // StreamCtx emits function_call items with empty call_id/name (the batch
    // events only carry the arguments text). The real metadata comes from
    // BatchResult.tool_calls which are parsed from the full generated text.
    // Invariant: tool_calls order matches output_items function_call order
    // (both from the same left-to-right parse of generated text).
    if let Some(ref r) = batch_result {
        if !r.tool_calls.is_empty() {
            if let Ok(mut guard) = ctx.lock() {
                let mut tc_idx = 0;
                for item in &mut guard.output_items {
                    if item.get("type").and_then(|v| v.as_str()) == Some("function_call") {
                        if let Some(tc) = r.tool_calls.get(tc_idx) {
                            item["call_id"] = json!(tc.id);
                            item["name"] = json!(tc.name);
                            item["arguments"] = json!(tc.arguments);
                            tc_idx += 1;
                        }
                    }
                }
            }
        }
    }

    // Flush the last content/item so output_items is fully populated before
    // we read it for storage persistence and response building.
    if let Ok(mut guard) = ctx.lock() {
        let _ = guard.flush_coalesce();
        let _ = guard.emit_content_done();
        let _ = guard.emit_item_done();
        log::info!(target: "server::gen",
            "batch_stream post-flush: output_items={}", guard.output_items.len());
    }

    // Build output items from StreamCtx (properly structured with item_type
    // segmentation and now-patched function_call metadata).
    let output_items = if let Ok(guard) = ctx.lock() {
        if !guard.output_items.is_empty() {
            serde_json::Value::Array(guard.output_items.clone())
        } else {
            json!([])
        }
    } else {
        json!([])
    };

    // For conversation chaining, inject generation metadata into the last
    // assistant message so it persists to storage and is available on reload.
    if let Ok(mut guard) = ctx.lock() {
        let mut gen_data = json!({
            "model": model_id,
            "temperature": gen_temperature,
            "top_p": gen_top_p,
            "top_k": gen_top_k,
            "max_output_tokens": gen_max_tokens,
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "prefill_ms": prefill_ns as f64 / 1_000_000.0,
            "generation_ms": generation_ns as f64 / 1_000_000.0
        });
        if gen_min_p > 0.0 {
            gen_data["min_p"] = serde_json::Value::from(gen_min_p);
        }
        if gen_repetition_penalty != 1.0 {
            gen_data["repetition_penalty"] = serde_json::Value::from(gen_repetition_penalty);
        }
        if gen_seed != 0 {
            gen_data["seed"] = serde_json::Value::from(gen_seed);
        }
        for item in guard.output_items.iter_mut().rev() {
            if item.get("type").and_then(|v| v.as_str()) == Some("message")
                && item.get("role").and_then(|v| v.as_str()) == Some("assistant")
            {
                item["generation"] = gen_data;
                break;
            }
        }
    }

    let all_json = chat
        .to_responses_json(1)
        .map_err(|e| anyhow!("failed to serialize output: {}", e))?;

    // Get model info from the backend (quick metadata read, no generation lock).
    let model_info_json = {
        let guard = backend.blocking_lock();
        if let Some(ref b) = guard.backend {
            build_model_info_json(b)
        } else {
            serde_json::Value::Null
        }
    };

    Ok(StreamGenResult {
        output_items,
        usage: UsageStats {
            input_tokens: prompt_tokens,
            output_tokens: completion_tokens,
            prefill_ns,
            generation_ns,
            ttft_ns,
        },
        finish_reason,
        responses_json: all_json,
        model_info_json,
    })
}

/// Generate a short descriptive title for a conversation using the model.
///
/// Called after the main generation completes for new conversations when
/// auto_title is enabled. Uses a short inference (max 20 tokens) with a
/// title-generation system prompt.
fn generate_title(
    backend: &Arc<tokio::sync::Mutex<crate::server::state::BackendState>>,
    bucket_path: &Option<std::path::PathBuf>,
    session_id: &str,
    input_text: &str,
) -> Result<String> {
    let mut guard = backend.blocking_lock();
    let backend = guard
        .backend
        .as_mut()
        .ok_or_else(|| anyhow!("no backend available for title generation"))?;

    let chat = ChatHandle::new(Some(
        "Generate a concise title (under 8 words) for a conversation starting \
         with the user message below. Reply with only the title, no quotes or \
         extra punctuation.",
    ))?;
    chat.append_user_message(input_text)
        .map_err(|e| anyhow!("failed to append message for title gen: {e}"))?;

    let mut cfg = talu::router::GenerateConfig::default();
    cfg.max_tokens = 20;
    cfg.temperature = 0.7;

    let title_buf = Arc::new(std::sync::Mutex::new(String::new()));
    let buf_clone = title_buf.clone();
    let callback: talu::router::StreamCallback = Box::new(move |token| {
        if let Ok(mut buf) = buf_clone.lock() {
            buf.push_str(token.text);
        }
        true
    });

    talu::router::generate_stream(&chat, &[], backend, &cfg, callback)
        .map_err(|e| anyhow!("title generation failed: {e}"))?;

    let title = title_buf.lock().unwrap().clone();
    // Clean up: trim whitespace, strip surrounding quotes.
    let title = title.trim().to_string();
    let title = title
        .trim_matches('"')
        .trim_matches('\'')
        .trim()
        .to_string();
    if title.is_empty() {
        return Err(anyhow!("generated title is empty"));
    }

    // Update the session title in storage.
    if let Some(ref bp) = bucket_path {
        if let Ok(storage) = talu::storage::StorageHandle::open(bp) {
            let update = talu::storage::SessionUpdate {
                title: Some(title.clone()),
                ..Default::default()
            };
            let _ = storage.update_session(session_id, &update);
            log::info!(target: "server::gen", "auto-title for {session_id}: {title:?}");
        }
    }

    Ok(title)
}

/// Result from streaming generation, carrying output items and usage stats.
struct StreamGenResult {
    output_items: serde_json::Value,
    usage: UsageStats,
    finish_reason: FinishReason,
    /// Full conversation JSON for chaining via response_store.
    responses_json: String,
    /// Static model metadata (null for remote backends).
    model_info_json: serde_json::Value,
}

struct StreamCtx {
    tx: tokio::sync::mpsc::UnboundedSender<Bytes>,
    seq: u64,
    response_id: String,
    /// Stop flag for cancellation. Signaled when client disconnects.
    stop_flag: Arc<AtomicBool>,

    // -- Item/content transition tracking for SSE event completeness --
    /// Current output item index (incremented on item_type transitions).
    output_index: u32,
    /// Content part index within the current output item.
    content_index: u32,
    /// Item type of the currently active output item (None before first token).
    cur_item_type: Option<ItemType>,
    /// Content type of the currently active content part.
    cur_content_type: Option<ContentType>,
    /// Accumulated text for the current content part (for .done events).
    accumulated_text: String,
    /// Accumulated sampled logprobs for the current output_text content part.
    accumulated_logprobs: Vec<serde_json::Value>,
    /// Accumulated URL-citation annotations for the current output_text content part.
    accumulated_annotations: Vec<serde_json::Value>,
    /// De-duplication keys for emitted annotations in the current content part.
    annotation_keys: HashSet<String>,
    /// Completed output items emitted during this stream.
    output_items: Vec<serde_json::Value>,
    /// Item ID for the current output item (may differ from message_id for
    /// multi-item responses, e.g. reasoning + message).
    cur_item_id: String,
    /// Tool definitions from the request (for response resource round-tripping).
    tools_json: Option<serde_json::Value>,
    /// Tool choice from the request (for response resource round-tripping).
    tool_choice_json: Option<serde_json::Value>,
    /// Tenant identifier for audit logging.
    tenant_id: Option<String>,
    /// Optional user identifier from the gateway.
    user_id: Option<String>,
    /// TaluDB session ID (returned in metadata so the UI can chain follow-ups).
    session_id: String,
    /// Whether this stream is serving strict `/v1/responses`.
    strict_responses: bool,
    /// Request `store` flag.
    request_store: bool,
    /// Previous response chain id from the request.
    previous_response_id: Option<String>,
    /// Request instructions field for OpenResponses response resources.
    instructions: Option<String>,
    /// Request text format configuration for response resource round-tripping.
    text_format: Option<TextFormatConfig>,
    /// Project ID from request metadata (returned in response metadata).
    project_id: Option<String>,
    /// Requested top_logprobs count from the API request.
    top_logprobs: usize,
    /// Request sampling penalties echoed in response resources.
    top_k: Option<u32>,
    presence_penalty: f64,
    frequency_penalty: f64,
    /// Reasoning config echoed in response resources.
    reasoning_effort: Option<String>,
    reasoning_summary: Option<String>,

    // -- Token coalescing: buffer text deltas and flush periodically --
    /// Accumulated delta text since last flush.
    coalesce_buf: String,
    /// SSE event name for the buffered deltas (e.g. "response.output_text.delta"
    /// or "response.reasoning.delta"). Empty when buffer is empty.
    coalesce_event: String,
    /// Timestamp of the first token in the current coalescing window (handler-level).
    coalesce_start: Option<Instant>,
    /// Byte offset in `accumulated_text` where the current coalesce window started.
    /// Used for URL annotation lookback across flush boundaries.
    coalesce_text_start: usize,
    /// Cumulative token count from the engine (updated per send_delta call).
    last_tokens_generated: usize,
    /// Nanoseconds elapsed since first token from the engine (updated per send_delta call).
    last_elapsed_ns: u64,
}

impl StreamCtx {
    /// Coalescing window for text deltas (ms). Tokens arriving within this
    /// window are batched into a single SSE event to reduce per-token JSON
    /// serialization and network overhead.
    const COALESCE_WINDOW_MS: u64 = 30;

    fn send_delta(&mut self, token: &talu::router::StreamToken) -> Result<()> {
        let item_changed = self.cur_item_type != Some(token.item_type);
        let content_changed = item_changed || self.cur_content_type != Some(token.content_type);

        // Flush any coalesced text deltas before closing content/items on
        // type transition — the buffered deltas belong to the previous type.
        if (content_changed || item_changed) && !self.coalesce_buf.is_empty() {
            self.flush_coalesce()?;
        }

        // Close the previous content part and output item on transition.
        if content_changed && self.cur_content_type.is_some() {
            self.emit_content_done()?;
        }
        if item_changed && self.cur_item_type.is_some() {
            self.emit_item_done()?;
        }

        // Open a new output item and content part on transition.
        if item_changed {
            // Assign a new item_id for each output item beyond the first.
            if self.cur_item_type.is_some() {
                self.output_index += 1;
                self.content_index = 0;
                self.cur_item_id = format!("msg_{}", random_id());
            }
            self.cur_item_type = Some(token.item_type);
            self.emit_item_added(token.item_type)?;
        }
        if content_changed {
            if !item_changed && self.cur_content_type.is_some() {
                self.content_index += 1;
            }
            self.cur_content_type = Some(token.content_type);
            self.accumulated_text.clear();
            self.accumulated_logprobs.clear();
            self.accumulated_annotations.clear();
            self.annotation_keys.clear();
            self.emit_content_part_added(token.content_type)?;
        }

        self.accumulated_text.push_str(token.text);

        // Capture engine-level cumulative timing for tok/s.
        self.last_tokens_generated = token.tokens_generated;
        self.last_elapsed_ns = token.elapsed_ns;

        let event_name =
            talu::responses::stream_delta_event_name(token.item_type, token.content_type);

        // Coalesce output_text and reasoning deltas: buffer and flush periodically.
        let coalesceable =
            event_name == "response.output_text.delta" || event_name == "response.reasoning.delta";
        if coalesceable {
            // If the event type changed (e.g. reasoning → text), flush first.
            if !self.coalesce_buf.is_empty() && self.coalesce_event != event_name {
                self.flush_coalesce()?;
            }
            if self.coalesce_buf.is_empty() {
                self.coalesce_start = Some(Instant::now());
                self.coalesce_event.clear();
                self.coalesce_event.push_str(event_name);
                self.coalesce_text_start =
                    self.accumulated_text.len().saturating_sub(token.text.len());
            }
            self.coalesce_buf.push_str(token.text);
            if self.coalesce_start.map_or(false, |t| {
                t.elapsed().as_millis() as u64 >= Self::COALESCE_WINDOW_MS
            }) {
                self.flush_coalesce()?;
            }
            return Ok(());
        }

        // Non-coalesceable deltas: emit immediately via json! (these are infrequent).
        let payload = match event_name {
            "response.function_call_arguments.delta" => json!({
                "type": event_name,
                "sequence_number": self.seq,
                "item_id": self.cur_item_id,
                "output_index": self.output_index,
                "delta": token.text,
                "obfuscation": ""
            }),
            "response.reasoning_summary_text.delta" => json!({
                "type": event_name,
                "sequence_number": self.seq,
                "item_id": self.cur_item_id,
                "output_index": self.output_index,
                "summary_index": self.content_index,
                "delta": token.text,
                "obfuscation": ""
            }),
            "response.refusal.delta" => json!({
                "type": event_name,
                "sequence_number": self.seq,
                "item_id": self.cur_item_id,
                "output_index": self.output_index,
                "content_index": self.content_index,
                "delta": token.text
            }),
            "response.reasoning.delta" => json!({
                "type": event_name,
                "sequence_number": self.seq,
                "item_id": self.cur_item_id,
                "output_index": self.output_index,
                "content_index": self.content_index,
                "delta": token.text,
                "obfuscation": ""
            }),
            _ => json!({
                "type": event_name,
                "sequence_number": self.seq,
                "item_id": self.cur_item_id,
                "output_index": self.output_index,
                "content_index": self.content_index,
                "delta": token.text,
                "logprobs": [],
                "obfuscation": ""
            }),
        };
        self.seq += 1;

        if send_event(&self.tx, event_name, payload).is_err() {
            self.stop_flag.store(true, Ordering::Release);
            return Err(anyhow!("client disconnected"));
        }
        self.emit_output_text_annotations(token)?;
        Ok(())
    }

    /// Flush the coalesced delta buffer as a single SSE event.
    /// Uses pre-formatted SSE strings to avoid `json!` / `serde_json::Value`
    /// tree allocation on the hot path.
    fn flush_coalesce(&mut self) -> Result<()> {
        if self.coalesce_buf.is_empty() {
            return Ok(());
        }
        // JSON-escape only the delta text; all other fields are numeric or
        // known-safe alphanumeric identifiers.
        let delta_json = serde_json::to_string(&self.coalesce_buf)?;
        let elapsed_ms = self.last_elapsed_ns / 1_000_000;
        let is_output_text = self.coalesce_event == "response.output_text.delta";
        let formatted = if is_output_text {
            format!(
                "event: response.output_text.delta\ndata: {{\"type\":\"response.output_text.delta\",\"sequence_number\":{},\"item_id\":\"{}\",\"output_index\":{},\"content_index\":{},\"delta\":{},\"logprobs\":[],\"obfuscation\":\"\",\"tokens_generated\":{},\"elapsed_ms\":{}}}\n\n",
                self.seq, self.cur_item_id, self.output_index, self.content_index, delta_json, self.last_tokens_generated, elapsed_ms,
            )
        } else {
            format!(
                "event: response.reasoning.delta\ndata: {{\"type\":\"response.reasoning.delta\",\"sequence_number\":{},\"item_id\":\"{}\",\"output_index\":{},\"content_index\":{},\"delta\":{},\"obfuscation\":\"\",\"tokens_generated\":{},\"elapsed_ms\":{}}}\n\n",
                self.seq, self.cur_item_id, self.output_index, self.content_index, delta_json, self.last_tokens_generated, elapsed_ms,
            )
        };
        self.seq += 1;

        if self.tx.send(Bytes::from(formatted)).is_err() {
            self.stop_flag.store(true, Ordering::Release);
            self.coalesce_buf.clear();
            self.coalesce_event.clear();
            self.coalesce_start = None;
            return Err(anyhow!("client disconnected"));
        }

        // Run URL-citation annotation detection on output_text deltas.
        // Scan a lookback window from accumulated_text to catch URLs that
        // straddle the boundary of the previous flush (e.g. "https://exa" |
        // "mple.com/"). The window covers up to 256 bytes before the current
        // coalesce chunk start plus the chunk itself.
        if is_output_text
            && self.cur_item_type == Some(ItemType::Message)
            && self.cur_content_type == Some(ContentType::OutputText)
        {
            const LOOKBACK: usize = 256;
            let window_start = self.coalesce_text_start.saturating_sub(LOOKBACK);
            if let Some(window) = self.accumulated_text.get(window_start..) {
                for (local_start, local_end, url) in extract_urls_with_offsets(window) {
                    let start_index = window_start + local_start;
                    let end_index = window_start + local_end;
                    let dedupe_key = format!("{start_index}:{end_index}:{url}");
                    if !self.annotation_keys.insert(dedupe_key) {
                        continue;
                    }
                    let annotation = json!({
                        "type": "url_citation",
                        "url": url,
                        "start_index": start_index,
                        "end_index": end_index,
                        "title": ""
                    });
                    let annotation_index = self.accumulated_annotations.len();
                    self.accumulated_annotations.push(annotation.clone());
                    let payload = json!({
                        "type": "response.output_text.annotation.added",
                        "sequence_number": self.seq,
                        "item_id": self.cur_item_id,
                        "output_index": self.output_index,
                        "content_index": self.content_index,
                        "annotation_index": annotation_index,
                        "annotation": annotation
                    });
                    self.seq += 1;
                    self.try_send("response.output_text.annotation.added", payload)?;
                }
            }
        }

        self.coalesce_buf.clear();
        self.coalesce_event.clear();
        self.coalesce_start = None;
        Ok(())
    }

    /// Emit `response.output_item.added` for a new output item.
    fn emit_item_added(&mut self, item_type: ItemType) -> Result<()> {
        let item_object = match item_type {
            ItemType::Reasoning => json!({
                "type": "reasoning",
                "id": self.cur_item_id,
                "summary": [],
                "status": "in_progress"
            }),
            ItemType::FunctionCall => json!({
                "type": "function_call",
                "id": self.cur_item_id,
                "call_id": "",
                "name": "",
                "arguments": "",
                "status": "in_progress"
            }),
            _ => json!({
                "type": "message",
                "id": self.cur_item_id,
                "role": "assistant",
                "status": "in_progress",
                "content": []
            }),
        };

        let payload = json!({
            "type": "response.output_item.added",
            "sequence_number": self.seq,
            "output_index": self.output_index,
            "item": item_object
        });
        self.seq += 1;
        self.try_send("response.output_item.added", payload)
    }

    /// Emit `response.content_part.added` for a new content part.
    fn emit_content_part_added(&mut self, content_type: ContentType) -> Result<()> {
        let part_type = talu::responses::stream_content_part_type(content_type);
        let payload = json!({
            "type": "response.content_part.added",
            "sequence_number": self.seq,
            "item_id": self.cur_item_id,
            "output_index": self.output_index,
            "content_index": self.content_index,
            "part": content_part_done_payload(part_type, "", &[], &[])
        });
        self.seq += 1;
        self.try_send("response.content_part.added", payload)?;

        if matches!(content_type, ContentType::SummaryText) {
            let summary_added_payload = json!({
                "type": "response.reasoning_summary_part.added",
                "sequence_number": self.seq,
                "item_id": self.cur_item_id,
                "output_index": self.output_index,
                "summary_index": self.content_index,
                "part": {
                    "type": "summary_text",
                    "text": ""
                }
            });
            self.seq += 1;
            self.try_send(
                "response.reasoning_summary_part.added",
                summary_added_payload,
            )?;
        }
        Ok(())
    }

    /// Emit the type-specific `.done` event and `response.content_part.done`
    /// for the current content part.
    fn emit_content_done(&mut self) -> Result<()> {
        let (item_type, content_type) = match (self.cur_item_type, self.cur_content_type) {
            (Some(it), Some(ct)) => (it, ct),
            _ => return Ok(()),
        };

        // Type-specific done event (e.g. response.output_text.done).
        let done_event = talu::responses::stream_done_event_name(item_type, content_type);
        let done_payload = match done_event {
            "response.output_text.done" => json!({
                "type": done_event,
                "sequence_number": self.seq,
                "item_id": self.cur_item_id,
                "output_index": self.output_index,
                "content_index": self.content_index,
                "text": self.accumulated_text,
                "logprobs": self.accumulated_logprobs.clone()
            }),
            "response.function_call_arguments.done" => json!({
                "type": done_event,
                "sequence_number": self.seq,
                "item_id": self.cur_item_id,
                "output_index": self.output_index,
                "arguments": self.accumulated_text
            }),
            "response.reasoning_summary_text.done" => json!({
                "type": done_event,
                "sequence_number": self.seq,
                "item_id": self.cur_item_id,
                "output_index": self.output_index,
                "summary_index": self.content_index,
                "text": self.accumulated_text
            }),
            "response.refusal.done" => json!({
                "type": done_event,
                "sequence_number": self.seq,
                "item_id": self.cur_item_id,
                "output_index": self.output_index,
                "content_index": self.content_index,
                "refusal": self.accumulated_text
            }),
            "response.reasoning.done" => json!({
                "type": done_event,
                "sequence_number": self.seq,
                "item_id": self.cur_item_id,
                "output_index": self.output_index,
                "content_index": self.content_index,
                "text": self.accumulated_text
            }),
            _ => json!({
                "type": done_event,
                "sequence_number": self.seq,
                "item_id": self.cur_item_id,
                "output_index": self.output_index,
                "content_index": self.content_index,
                "text": self.accumulated_text,
                "logprobs": []
            }),
        };
        self.seq += 1;
        self.try_send(done_event, done_payload)?;

        // content_part.done
        let part_type = talu::responses::stream_content_part_type(content_type);
        let part_payload = json!({
            "type": "response.content_part.done",
            "sequence_number": self.seq,
            "item_id": self.cur_item_id,
            "output_index": self.output_index,
            "content_index": self.content_index,
            "part": content_part_done_payload(
                part_type,
                &self.accumulated_text,
                &self.accumulated_annotations,
                &self.accumulated_logprobs,
            )
        });
        self.seq += 1;
        self.try_send("response.content_part.done", part_payload)?;

        if matches!(content_type, ContentType::SummaryText) {
            let summary_done_payload = json!({
                "type": "response.reasoning_summary_part.done",
                "sequence_number": self.seq,
                "item_id": self.cur_item_id,
                "output_index": self.output_index,
                "summary_index": self.content_index,
                "part": {
                    "type": "summary_text",
                    "text": self.accumulated_text
                }
            });
            self.seq += 1;
            self.try_send("response.reasoning_summary_part.done", summary_done_payload)?;
        }
        Ok(())
    }

    /// Emit `response.output_item.done` for the current output item.
    fn emit_item_done(&mut self) -> Result<()> {
        let item_type = match self.cur_item_type {
            Some(it) => it,
            None => return Ok(()),
        };

        let item_object = match item_type {
            ItemType::Reasoning => {
                let mut summary = Vec::new();
                if !self.accumulated_text.is_empty() {
                    summary.push(json!({
                        "type": "summary_text",
                        "text": self.accumulated_text
                    }));
                }
                json!({
                    "type": "reasoning",
                    "id": self.cur_item_id,
                    "summary": summary,
                    "status": "completed"
                })
            }
            ItemType::FunctionCall => json!({
                "type": "function_call",
                "id": self.cur_item_id,
                "call_id": "",
                "name": "",
                "arguments": self.accumulated_text,
                "status": "completed"
            }),
            _ => json!({
                "type": "message",
                "id": self.cur_item_id,
                "role": "assistant",
                "status": "completed",
                "content": [match self.cur_content_type {
                    Some(ContentType::Refusal) => json!({
                        "type": "refusal",
                        "refusal": self.accumulated_text
                    }),
                    _ => json!({
                        "type": "output_text",
                        "text": self.accumulated_text,
                        "annotations": self.accumulated_annotations.clone(),
                        "logprobs": self.accumulated_logprobs.clone()
                    }),
                }]
            }),
        };
        self.output_items.push(item_object.clone());
        // Reset to prevent duplicate emission on repeated calls.
        self.cur_item_type = None;
        self.cur_content_type = None;

        let payload = json!({
            "type": "response.output_item.done",
            "sequence_number": self.seq,
            "output_index": self.output_index,
            "item": item_object
        });
        self.seq += 1;
        self.try_send("response.output_item.done", payload)
    }

    /// Send an event, signaling the stop flag on client disconnect.
    fn try_send(&self, event: &str, payload: serde_json::Value) -> Result<()> {
        if send_event(&self.tx, event, payload).is_err() {
            self.stop_flag.store(true, Ordering::Release);
            return Err(anyhow!("client disconnected"));
        }
        Ok(())
    }

    fn emit_output_text_annotations(&mut self, token: &talu::router::StreamToken) -> Result<()> {
        if token.item_type != ItemType::Message || token.content_type != ContentType::OutputText {
            return Ok(());
        }
        let chunk_start = self.accumulated_text.len().saturating_sub(token.text.len());
        for (local_start, local_end, url) in extract_urls_with_offsets(token.text) {
            let start_index = chunk_start + local_start;
            let end_index = chunk_start + local_end;
            let dedupe_key = format!("{start_index}:{end_index}:{url}");
            if !self.annotation_keys.insert(dedupe_key) {
                continue;
            }

            let annotation = json!({
                "type": "url_citation",
                "url": url,
                "start_index": start_index,
                "end_index": end_index,
                "title": ""
            });
            let annotation_index = self.accumulated_annotations.len();
            self.accumulated_annotations.push(annotation.clone());

            let payload = json!({
                "type": "response.output_text.annotation.added",
                "sequence_number": self.seq,
                "item_id": self.cur_item_id,
                "output_index": self.output_index,
                "content_index": self.content_index,
                "annotation_index": annotation_index,
                "annotation": annotation
            });
            self.seq += 1;
            self.try_send("response.output_text.annotation.added", payload)?;
        }
        Ok(())
    }

    fn flush_completion(
        &mut self,
        model_id: String,
        created_at: i64,
        max_output_tokens: Option<i64>,
        temperature: f64,
        top_p: f64,
        result: Result<StreamGenResult>,
    ) -> Result<()> {
        // Flush any remaining coalesced text deltas before closing.
        let _ = self.flush_coalesce();
        // Close the last content part and output item (if any tokens were emitted).
        let _ = self.emit_content_done();
        let _ = self.emit_item_done();

        match result {
            Ok(r) => {
                let (status, event_type) = match r.finish_reason {
                    FinishReason::Length => ("incomplete", "response.incomplete"),
                    FinishReason::Cancelled => ("incomplete", "response.incomplete"),
                    _ => ("completed", "response.completed"),
                };
                // Build output items from streaming accumulated state.
                // For streaming, output items are already sent as individual events.
                // The terminal response resource needs the output array for completeness.
                let output_items = if self.output_items.is_empty() {
                    r.output_items
                } else {
                    serde_json::Value::Array(self.output_items.clone())
                };
                let mut response = build_response_resource_value(
                    &self.response_id,
                    &model_id,
                    created_at,
                    Some(now_unix_seconds()),
                    output_items,
                    max_output_tokens,
                    temperature,
                    top_p,
                    self.top_k,
                    self.presence_penalty,
                    self.frequency_penalty,
                    self.top_logprobs as i64,
                    self.reasoning_effort.as_deref(),
                    self.reasoning_summary.as_deref(),
                    status,
                    Some(&r.usage),
                    self.tools_json.as_ref(),
                    self.tool_choice_json.as_ref(),
                    self.request_store,
                    self.instructions.as_deref(),
                    self.text_format.as_ref(),
                );
                if let Some(ref prev_id) = self.previous_response_id {
                    response["previous_response_id"] = json!(prev_id);
                }
                if !r.model_info_json.is_null() {
                    response["model_info"] = r.model_info_json;
                }
                if status == "incomplete" {
                    let reason = match r.finish_reason {
                        FinishReason::Cancelled => "cancelled",
                        _ => "max_output_tokens",
                    };
                    response["incomplete_details"] = json!({
                        "reason": reason
                    });
                }
                if !self.strict_responses {
                    let mut meta = json!({ "session_id": self.session_id });
                    if let Some(ref pid) = self.project_id {
                        meta["project_id"] = json!(pid);
                    }
                    response["metadata"] = meta;
                }
                let response = normalize_response_value(response);
                let payload = json!({
                    "type": event_type,
                    "sequence_number": self.seq,
                    "response": response
                });
                self.seq += 1;
                send_event(&self.tx, event_type, payload)?;
                self.log_generation_completed();
            }
            Err(err) => {
                if self.strict_responses {
                    let payload = json!({
                        "type": "error",
                        "sequence_number": self.seq,
                        "error": {
                            "type": "server_error",
                            "code": "server_error",
                            "message": format!("{err}"),
                            "param": null
                        }
                    });
                    self.seq += 1;
                    send_event(&self.tx, "error", payload)?;
                }
                let mut response = build_response_resource_value(
                    &self.response_id,
                    &model_id,
                    created_at,
                    Some(now_unix_seconds()),
                    json!([]),
                    max_output_tokens,
                    temperature,
                    top_p,
                    self.top_k,
                    self.presence_penalty,
                    self.frequency_penalty,
                    self.top_logprobs as i64,
                    self.reasoning_effort.as_deref(),
                    self.reasoning_summary.as_deref(),
                    "failed",
                    None,
                    self.tools_json.as_ref(),
                    self.tool_choice_json.as_ref(),
                    self.request_store,
                    self.instructions.as_deref(),
                    self.text_format.as_ref(),
                );
                if let Some(ref prev_id) = self.previous_response_id {
                    response["previous_response_id"] = json!(prev_id);
                }
                response["error"] = json!({
                    "code": "server_error",
                    "message": format!("{err}")
                });
                if !self.strict_responses {
                    let mut meta = json!({ "session_id": self.session_id });
                    if let Some(ref pid) = self.project_id {
                        meta["project_id"] = json!(pid);
                    }
                    response["metadata"] = meta;
                }
                let response = normalize_response_value(response);
                let payload = json!({
                    "type": "response.failed",
                    "sequence_number": self.seq,
                    "response": response
                });
                self.seq += 1;
                send_event(&self.tx, "response.failed", payload)?;
            }
        }
        Ok(())
    }

    fn log_generation_completed(&self) {
        if let Some(tenant_id) = self.tenant_id.as_deref() {
            if let Some(user_id) = self.user_id.as_deref() {
                log::info!(
                    target: "server::gen",
                    "[tenant={}] [user={}] Generation completed",
                    tenant_id,
                    user_id
                );
            } else {
                log::info!(target: "server::gen", "[tenant={}] Generation completed", tenant_id);
            }
        }
    }
}

fn send_event(
    tx: &tokio::sync::mpsc::UnboundedSender<Bytes>,
    event: &str,
    payload: serde_json::Value,
) -> Result<()> {
    let data = serde_json::to_string(&payload)?;
    let formatted = format!("event: {}\ndata: {}\n\n", event, data);
    tx.send(Bytes::from(formatted))
        .map_err(|_| anyhow!("stream closed"))?;
    Ok(())
}

fn extract_urls_with_offsets(text: &str) -> Vec<(usize, usize, String)> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        let rest = &bytes[i..];
        let prefix_len = if rest.starts_with(b"http://") {
            7
        } else if rest.starts_with(b"https://") {
            8
        } else {
            i += 1;
            continue;
        };

        let start = i;
        i += prefix_len;
        while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        let mut end = i;
        while end > start
            && matches!(
                bytes[end - 1],
                b'.' | b',' | b';' | b':' | b'!' | b'?' | b')' | b']' | b'}'
            )
        {
            end -= 1;
        }
        if end > start {
            if let Some(url) = text.get(start..end) {
                out.push((start, end, url.to_string()));
            }
        }
    }
    out
}

fn content_part_done_payload(
    part_type: &str,
    text: &str,
    annotations: &[serde_json::Value],
    logprobs: &[serde_json::Value],
) -> serde_json::Value {
    match part_type {
        "refusal" => json!({
            "type": "refusal",
            "refusal": text
        }),
        "summary_text" => json!({
            "type": "summary_text",
            "text": text
        }),
        "reasoning_text" => json!({
            "type": "reasoning_text",
            "text": text
        }),
        _ => json!({
            "type": part_type,
            "text": text,
            "annotations": annotations,
            "logprobs": logprobs
        }),
    }
}

fn response_text_format_value(text_format: Option<&TextFormatConfig>) -> serde_json::Value {
    match text_format {
        Some(TextFormatConfig::JsonSchema) => json!({ "type": "json_schema" }),
        _ => json!({ "type": "text" }),
    }
}

/// Token usage statistics for the response resource.
struct UsageStats {
    input_tokens: usize,
    output_tokens: usize,
    prefill_ns: u64,
    generation_ns: u64,
    ttft_ns: u64,
}

fn dtype_name(val: u8) -> &'static str {
    match val {
        0 => "F32",
        4 => "F16",
        5 => "BF16",
        25 => "GAF4",
        26 => "GAF8",
        27 => "MXFP4",
        28 => "F8E4M3",
        _ => "unknown",
    }
}

fn build_model_info_json(backend: &talu::InferenceBackend) -> serde_json::Value {
    let info = backend.model_info();
    if info.file_size == 0 && info.d_model == 0 {
        return serde_json::Value::Null;
    }
    json!({
        "weight_dtype": dtype_name(info.weight_dtype),
        "file_size_bytes": info.file_size,
        "tensor_count": info.tensor_count,
        "vocab_size": info.vocab_size,
        "d_model": info.d_model,
        "n_layers": info.n_layers,
        "n_heads": info.n_heads,
        "n_kv_groups": info.n_kv_groups,
        "d_ff": info.d_ff,
        "max_seq_len": info.max_seq_len,
        "gaffine_group_size": info.gaffine_group_size,
    })
}

fn build_response_resource_value(
    response_id: &str,
    model_id: &str,
    created_at: i64,
    completed_at: Option<i64>,
    output_items: serde_json::Value,
    max_output_tokens: Option<i64>,
    temperature: f64,
    top_p: f64,
    top_k: Option<u32>,
    presence_penalty: f64,
    frequency_penalty: f64,
    top_logprobs: i64,
    reasoning_effort: Option<&str>,
    reasoning_summary: Option<&str>,
    status: &str,
    usage: Option<&UsageStats>,
    tools: Option<&serde_json::Value>,
    tool_choice: Option<&serde_json::Value>,
    store: bool,
    instructions: Option<&str>,
    text_format: Option<&TextFormatConfig>,
) -> serde_json::Value {
    let usage_value = match usage {
        Some(u) => {
            let mut v = json!({
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "total_tokens": u.input_tokens + u.output_tokens,
                "input_tokens_details": {
                    "cached_tokens": 0
                },
                "output_tokens_details": {
                    "reasoning_tokens": 0
                }
            });
            if u.prefill_ns > 0 {
                v["prefill_ms"] = json!(u.prefill_ns as f64 / 1_000_000.0);
            }
            if u.generation_ns > 0 {
                v["generation_ms"] = json!(u.generation_ns as f64 / 1_000_000.0);
            }
            if u.ttft_ns > 0 {
                v["ttft_ms"] = json!(u.ttft_ns as f64 / 1_000_000.0);
            }
            v
        }
        None => serde_json::Value::Null,
    };

    let tools_value = tools.cloned().unwrap_or_else(|| json!([]));
    let tool_choice_value = tool_choice.cloned().unwrap_or_else(|| json!("none"));

    json!({
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "completed_at": completed_at,
        "status": status,
        "incomplete_details": null,
        "model": model_id,
        "previous_response_id": null,
        "instructions": instructions,
        "output": output_items,
        "error": null,
        "tools": tools_value,
        "tool_choice": tool_choice_value,
        "truncation": "auto",
        "parallel_tool_calls": false,
        "text": { "format": response_text_format_value(text_format) },
        "top_p": top_p,
        "top_k": top_k.unwrap_or(0),
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "top_logprobs": top_logprobs,
        "temperature": temperature,
        "reasoning": { "effort": reasoning_effort, "summary": reasoning_summary },
        "usage": usage_value,
        "max_output_tokens": max_output_tokens,
        "max_tool_calls": null,
        "store": store,
        "background": false,
        "service_tier": "default",
        "metadata": {},
        "safety_identifier": null,
        "prompt_cache_key": null
    })
}

fn normalize_response_value(value: serde_json::Value) -> serde_json::Value {
    value
}

async fn select_model_id(state: Arc<AppState>, request_model: Option<String>) -> Result<String> {
    select_model_id_ex(state, request_model, true).await
}

/// Resolve model ID without loading the backend. Used by the streaming path
/// so that model loading can happen inside the spawn_blocking task where
/// progress events can be emitted through the SSE channel.
async fn resolve_model_id(state: Arc<AppState>, request_model: Option<String>) -> Result<String> {
    select_model_id_ex(state, request_model, false).await
}

async fn select_model_id_ex(
    state: Arc<AppState>,
    request_model: Option<String>,
    load_backend: bool,
) -> Result<String> {
    if let Some(configured) = state.configured_model.clone() {
        if let Some(requested) = request_model {
            // Accept exact match or suffix match (short ID vs full path).
            // The settings endpoint returns short IDs like "Org/Model" while
            // configured_model may be a full filesystem path.
            if requested == configured || configured.ends_with(&format!("/{}", requested)) {
                return Ok(configured);
            }

            // Provider-prefixed model (e.g. "vllm::llama3") — accept directly.
            // Routing to the correct provider backend happens later in
            // create_backend_for_model_with_progress via parse_model_target.
            if requested.contains("::") {
                return Ok(requested);
            }

            // Check remote backend models (e.g. OpenAI-compatible providers).
            let models = list_backend_models(state.clone()).await.unwrap_or_default();
            if models.iter().any(|model| model.id == requested) {
                return Ok(requested);
            }

            // Check managed local models and hot-swap if found.
            let requested_clone = requested.clone();
            let found = tokio::task::spawn_blocking(move || {
                talu::repo::repo_list_models(false)
                    .unwrap_or_default()
                    .into_iter()
                    .find(|m| m.id == requested_clone)
            })
            .await
            .unwrap_or(None);

            if found.is_some() {
                if load_backend {
                    ensure_backend_for_model(state.clone(), &requested).await?;
                }
                return Ok(requested);
            }

            return Err(anyhow!("model not available: {}", requested));
        }

        let models = list_backend_models(state.clone()).await.unwrap_or_default();
        if let Some(first) = models.first() {
            if !first.id.is_empty() {
                return Ok(first.id.clone());
            }
        }

        return Ok(configured);
    }

    let requested = request_model.ok_or_else(|| anyhow!("model is required"))?;
    if load_backend {
        ensure_backend_for_model(state.clone(), &requested).await?;
    }
    Ok(requested)
}

async fn list_backend_models(state: Arc<AppState>) -> Result<Vec<talu::RemoteModelInfo>> {
    let mut guard = state.backend.lock().await;
    let backend = guard
        .backend
        .as_mut()
        .ok_or_else(|| anyhow!("no backend available"))?;
    backend
        .list_models()
        .map_err(|err| anyhow!("model listing failed: {err}"))
}

async fn ensure_backend_for_model(state: Arc<AppState>, model_id: &str) -> Result<()> {
    // 1. Fast path: model already loaded.
    {
        let guard = state.backend.lock().await;
        if guard
            .current_model
            .as_deref()
            .map(|current| current == model_id)
            .unwrap_or(false)
            && guard.backend.is_some()
        {
            return Ok(());
        }
    }

    // 2. Singleflight: check if another task is already loading this model.
    //    The std::sync::Mutex is safe here — critical section is a HashMap
    //    lookup/insert/clone (nanoseconds), never held across .await.
    let maybe_rx = {
        let inflight = state.model_load_inflight.lock().unwrap();
        inflight.get(model_id).cloned()
    };

    if let Some(mut rx) = maybe_rx {
        // Another task is loading this model — wait for its result.
        log::info!(target: "server::gen",
            "waiting for in-flight load of model {}", model_id);
        let _ = rx.changed().await;
        return match &*rx.borrow() {
            ModelLoadResult::Ok => Ok(()),
            ModelLoadResult::Err(msg) => Err(anyhow!("{}", msg)),
            ModelLoadResult::Pending => Err(anyhow!("model load abandoned (loader disappeared)")),
        };
    }

    // 3. No in-flight load — we become the loader.
    //    Atomically check-and-register under the inflight mutex.
    let (tx, rx) = tokio::sync::watch::channel(ModelLoadResult::Pending);
    let lost_race_rx = {
        let mut inflight = state.model_load_inflight.lock().unwrap();
        // Double-check: another task may have registered between our read
        // and this write (two callers passed step 2 concurrently).
        if let Some(existing_rx) = inflight.get(model_id) {
            Some(existing_rx.clone())
        } else {
            inflight.insert(model_id.to_string(), rx);
            None
        }
    };
    if let Some(mut rx) = lost_race_rx {
        // Lost the race — fall back to waiter path.
        let _ = rx.changed().await;
        return match &*rx.borrow() {
            ModelLoadResult::Ok => Ok(()),
            ModelLoadResult::Err(msg) => Err(anyhow!("{}", msg)),
            ModelLoadResult::Pending => Err(anyhow!("model load abandoned (loader disappeared)")),
        };
    }

    // 4. Re-check fast path — the model may have been loaded by a streaming
    //    handler between our first check and registering as the loader.
    {
        let guard = state.backend.lock().await;
        if guard
            .current_model
            .as_deref()
            .map(|current| current == model_id)
            .unwrap_or(false)
            && guard.backend.is_some()
        {
            let _ = tx.send(ModelLoadResult::Ok);
            state.model_load_inflight.lock().unwrap().remove(model_id);
            return Ok(());
        }
    }

    // 5. Load the model.
    log::info!(target: "server::gen", "loading backend for model {}", model_id);
    let model = model_id.to_string();
    let model_key = model_id.to_string();
    let db_root = state
        .bucket_path
        .as_ref()
        .map(|b| b.join("kv").to_string_lossy().into_owned());
    let load_result = tokio::task::spawn_blocking(move || {
        provider::create_backend_for_model(&model, db_root.as_deref())
    })
    .await;

    let backend = match load_result {
        Ok(Ok(b)) => b,
        Ok(Err(e)) => {
            let msg = format!("{e}");
            let _ = tx.send(ModelLoadResult::Err(msg.clone()));
            state.model_load_inflight.lock().unwrap().remove(&model_key);
            return Err(e);
        }
        Err(e) => {
            let msg = format!("model load task panicked: {e}");
            let _ = tx.send(ModelLoadResult::Err(msg.clone()));
            state.model_load_inflight.lock().unwrap().remove(&model_key);
            return Err(anyhow!("{}", msg));
        }
    };

    // 6. Install backend + scheduler, drain old.
    let new_sched = crate::server::batch_scheduler::SchedulerState::new(&backend, None)
        .ok()
        .map(Arc::new);
    let old_sched = if let Ok(mut sched_guard) = state.batch_scheduler.lock() {
        let old = sched_guard.take();
        *sched_guard = new_sched;
        old
    } else {
        None
    };

    let mut guard = state.backend.lock().await;
    // Take old backend before replacing — it must outlive the old
    // scheduler's step thread (BatchHandle holds an engine pointer).
    let old_backend = guard.backend.take();
    guard.backend = Some(backend);
    guard.current_model = Some(model_key.clone());
    drop(guard); // Release backend lock before blocking drain.

    // 7. Signal waiters and clean up inflight entry.
    let _ = tx.send(ModelLoadResult::Ok);
    state.model_load_inflight.lock().unwrap().remove(&model_key);

    // 8. Drain old scheduler in background.
    if old_sched.is_some() || old_backend.is_some() {
        // Join any previous drain thread before spawning a new one.
        if let Ok(mut dt) = state.drain_thread.lock() {
            if let Some(prev) = dt.take() {
                let _ = prev.join();
            }
        }
        let drain_handle = std::thread::Builder::new()
            .name("sched-drain".into())
            .spawn(move || {
                if let Some(s) = old_sched {
                    s.shutdown();
                }
                drop(old_backend);
            });
        match drain_handle {
            Ok(handle) => {
                if let Ok(mut dt) = state.drain_thread.lock() {
                    *dt = Some(handle);
                }
            }
            Err(e) => {
                log::error!(target: "server::gen",
                    "failed to spawn drain thread: {e}");
            }
        }
    }
    Ok(())
}

fn log_generation_completed(ctx: &AuthContext) {
    if let Some(user_id) = ctx.user_id.as_deref() {
        log::info!(
            target: "server::gen",
            "[tenant={}] [user={}] Generation completed",
            ctx.tenant_id,
            user_id
        );
    } else {
        log::info!(target: "server::gen", "[tenant={}] Generation completed", ctx.tenant_id);
    }
}

fn random_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{:x}", nanos)
}

fn now_unix_seconds() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

fn parse_requested_text_format(
    text: Option<&serde_json::Value>,
) -> std::result::Result<Option<TextFormatConfig>, String> {
    let Some(text_value) = text else {
        return Ok(None);
    };
    let text_obj = text_value
        .as_object()
        .ok_or_else(|| "`text` must be an object".to_string())?;

    if let Some(verbosity_value) = text_obj.get("verbosity") {
        match verbosity_value {
            serde_json::Value::Null => {}
            serde_json::Value::String(value)
                if matches!(value.as_str(), "low" | "medium" | "high") => {}
            _ => {
                return Err(
                    "`text.verbosity` must be one of: low, medium, high, or null".to_string(),
                );
            }
        }
    }

    let Some(format_value) = text_obj.get("format") else {
        return Ok(None);
    };
    if format_value.is_null() {
        return Ok(None);
    }
    let format_obj = format_value
        .as_object()
        .ok_or_else(|| "`text.format` must be an object".to_string())?;
    let format_type = format_obj
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "`text.format.type` must be a string".to_string())?;

    match format_type {
        "text" => Ok(Some(TextFormatConfig::Text)),
        "json_schema" => Ok(Some(TextFormatConfig::JsonSchema)),
        "json_object" => Err("`text.format.type` must be `text` or `json_schema`".to_string()),
        _ => Err("`text.format.type` must be `text` or `json_schema`".to_string()),
    }
}

fn parse_include_config(include: Option<&serde_json::Value>) -> std::result::Result<(), String> {
    let Some(include_value) = include else {
        return Ok(());
    };
    let values = include_value
        .as_array()
        .ok_or_else(|| "`include` must be an array of strings".to_string())?;

    for value in values {
        let key = value
            .as_str()
            .ok_or_else(|| "`include` entries must be strings".to_string())?;
        match key {
            "message.output_text.logprobs" | "reasoning.encrypted_content" => {}
            _ => return Err(format!("`include` contains unsupported value `{key}`")),
        }
    }
    Ok(())
}

fn parse_logprob_config(
    include: Option<&serde_json::Value>,
    top_logprobs: Option<i64>,
) -> std::result::Result<LogprobConfig, String> {
    parse_include_config(include)?;
    let top_logprobs_value = match top_logprobs {
        Some(value) if (0..=20).contains(&value) => value as usize,
        Some(_) => return Err("`top_logprobs` must be between 0 and 20".to_string()),
        None => 0,
    };
    Ok(LogprobConfig {
        top_logprobs: top_logprobs_value,
    })
}

fn parse_reasoning_config(
    reasoning: Option<&serde_json::Value>,
) -> std::result::Result<ReasoningConfig, String> {
    let Some(reasoning_value) = reasoning else {
        return Ok(ReasoningConfig::default());
    };
    let obj = reasoning_value
        .as_object()
        .ok_or_else(|| "`reasoning` must be an object".to_string())?;
    for key in obj.keys() {
        if key != "effort" && key != "summary" {
            return Err(format!("`reasoning.{key}` is not supported"));
        }
    }

    let effort = match obj.get("effort") {
        Some(serde_json::Value::Null) | None => None,
        Some(v) => {
            let effort = v
                .as_str()
                .ok_or_else(|| "`reasoning.effort` must be a string or null".to_string())?;
            if !matches!(effort, "none" | "low" | "medium" | "high" | "xhigh") {
                return Err(
                    "`reasoning.effort` must be one of: none, low, medium, high, xhigh".to_string(),
                );
            }
            Some(effort.to_string())
        }
    };

    let summary = match obj.get("summary") {
        Some(serde_json::Value::Null) | None => None,
        Some(v) => {
            let summary = v
                .as_str()
                .ok_or_else(|| "`reasoning.summary` must be a string or null".to_string())?;
            if !matches!(summary, "auto" | "concise" | "detailed") {
                return Err(
                    "`reasoning.summary` must be one of: auto, concise, detailed".to_string(),
                );
            }
            Some(summary.to_string())
        }
    };

    Ok(ReasoningConfig { effort, summary })
}

fn validate_penalty_bounds(field: &str, value: Option<f64>) -> std::result::Result<(), String> {
    let Some(value) = value else {
        return Ok(());
    };
    if !(-2.0..=2.0).contains(&value) {
        return Err(format!("`{field}` must be between -2 and 2"));
    }
    Ok(())
}

fn validate_temperature(temperature: Option<f64>) -> std::result::Result<(), String> {
    let Some(temperature) = temperature else {
        return Ok(());
    };
    if !(0.0..=2.0).contains(&temperature) {
        return Err("`temperature` must be between 0 and 2".to_string());
    }
    Ok(())
}

fn validate_top_p(top_p: Option<f64>) -> std::result::Result<(), String> {
    let Some(top_p) = top_p else {
        return Ok(());
    };
    if !(0.0..=1.0).contains(&top_p) {
        return Err("`top_p` must be between 0 and 1".to_string());
    }
    Ok(())
}

fn validate_max_output_tokens(max_output_tokens: Option<i64>) -> std::result::Result<(), String> {
    let Some(max_output_tokens) = max_output_tokens else {
        return Ok(());
    };
    if max_output_tokens < 1 {
        return Err("`max_output_tokens` must be at least 1".to_string());
    }
    Ok(())
}

fn validate_max_tool_calls(max_tool_calls: Option<i64>) -> std::result::Result<(), String> {
    let Some(max_tool_calls) = max_tool_calls else {
        return Ok(());
    };
    if max_tool_calls < 1 {
        return Err("`max_tool_calls` must be at least 1".to_string());
    }
    Ok(())
}

fn validate_prompt_cache_key(prompt_cache_key: Option<&str>) -> std::result::Result<(), String> {
    let Some(prompt_cache_key) = prompt_cache_key else {
        return Ok(());
    };
    if prompt_cache_key.chars().count() <= 64 {
        return Ok(());
    }
    Err("`prompt_cache_key` must be at most 64 characters".to_string())
}

fn validate_metadata(metadata: Option<&serde_json::Value>) -> std::result::Result<(), String> {
    let Some(metadata) = metadata else {
        return Ok(());
    };
    let obj = metadata
        .as_object()
        .ok_or_else(|| "`metadata` must be an object".to_string())?;
    if obj.len() > 16 {
        return Err("`metadata` must have at most 16 entries".to_string());
    }
    for (key, value) in obj {
        if key.chars().count() > 64 {
            return Err("`metadata` keys must be at most 64 characters".to_string());
        }
        let value = value
            .as_str()
            .ok_or_else(|| "`metadata` values must be strings".to_string())?;
        if value.chars().count() > 512 {
            return Err("`metadata` values must be at most 512 characters".to_string());
        }
    }
    Ok(())
}

fn is_valid_tool_or_function_name(name: &str) -> bool {
    let len = name.chars().count();
    if len == 0 || len > 64 {
        return false;
    }
    name.chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

fn validate_input_items(input: &serde_json::Value) -> std::result::Result<(), String> {
    let items = input
        .as_array()
        .ok_or_else(|| "`input` must be a string, an array, or null".to_string())?;
    for item in items {
        validate_input_item(item)?;
    }
    Ok(())
}

fn validate_input_item(item: &serde_json::Value) -> std::result::Result<(), String> {
    let obj = item
        .as_object()
        .ok_or_else(|| "`input` array entries must be objects".to_string())?;

    match obj.get("type") {
        None => {
            if obj.contains_key("id") {
                validate_item_reference_input(obj)
            } else {
                Err("`input[*].type` must be a string or null".to_string())
            }
        }
        Some(serde_json::Value::Null) => validate_item_reference_input(obj),
        Some(serde_json::Value::String(kind)) => match kind.as_str() {
            "message" => validate_message_input(obj),
            "function_call" => validate_function_call_input(obj),
            "function_call_output" => validate_function_call_output_input(obj),
            "reasoning" => validate_reasoning_input(obj),
            "item_reference" => validate_item_reference_input(obj),
            _ => Err(format!(
                "`input[*].type` contains unsupported value `{kind}`"
            )),
        },
        _ => Err("`input[*].type` must be a string or null".to_string()),
    }
}

fn require_non_empty_string<'a>(
    obj: &'a serde_json::Map<String, serde_json::Value>,
    key: &str,
    context: &str,
) -> std::result::Result<&'a str, String> {
    let value = obj
        .get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("`{context}.{key}` must be a non-empty string"))?;
    if value.is_empty() {
        return Err(format!("`{context}.{key}` must be a non-empty string"));
    }
    Ok(value)
}

fn validate_item_reference_input(
    obj: &serde_json::Map<String, serde_json::Value>,
) -> std::result::Result<(), String> {
    let _ = require_non_empty_string(obj, "id", "input[*]")?;
    Ok(())
}

fn validate_function_call_input(
    obj: &serde_json::Map<String, serde_json::Value>,
) -> std::result::Result<(), String> {
    let _ = require_non_empty_string(obj, "call_id", "input[*]")?;
    let name = require_non_empty_string(obj, "name", "input[*]")?;
    if !is_valid_tool_or_function_name(name) {
        return Err("`input[*].name` must match ^[a-zA-Z0-9_-]{1,64}$".to_string());
    }
    let _ = obj
        .get("arguments")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "`input[*].arguments` must be a string".to_string())?;
    if let Some(status) = obj.get("status") {
        let status = status
            .as_str()
            .ok_or_else(|| "`input[*].status` must be a string".to_string())?;
        if !matches!(status, "in_progress" | "completed") {
            return Err("`input[*].status` must be one of: in_progress, completed".to_string());
        }
    }
    Ok(())
}

fn validate_function_call_output_input(
    obj: &serde_json::Map<String, serde_json::Value>,
) -> std::result::Result<(), String> {
    let _ = require_non_empty_string(obj, "call_id", "input[*]")?;
    if let Some(status) = obj.get("status") {
        let status = status
            .as_str()
            .ok_or_else(|| "`input[*].status` must be a string".to_string())?;
        if !matches!(status, "in_progress" | "completed" | "incomplete") {
            return Err(
                "`input[*].status` must be one of: in_progress, completed, incomplete".to_string(),
            );
        }
    }
    let output = obj
        .get("output")
        .ok_or_else(|| "`input[*].output` is required".to_string())?;
    if output.is_string() {
        return Ok(());
    }
    let parts = output
        .as_array()
        .ok_or_else(|| "`input[*].output` must be a string or an array".to_string())?;
    for part in parts {
        validate_function_call_output_part(part)?;
    }
    Ok(())
}

fn validate_function_call_output_part(part: &serde_json::Value) -> std::result::Result<(), String> {
    let obj = part
        .as_object()
        .ok_or_else(|| "`input[*].output[*]` must be an object".to_string())?;
    let part_type = obj
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "`input[*].output[*].type` must be a string".to_string())?;
    match part_type {
        "input_text" => {
            let _ = obj
                .get("text")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "`input[*].output[*].text` must be a string".to_string())?;
            Ok(())
        }
        "input_image" => validate_image_part(obj, "input[*].output[*]"),
        "input_file" => validate_file_part(obj, "input[*].output[*]"),
        "input_video" => {
            let _ = obj
                .get("video_url")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "`input[*].output[*].video_url` must be a string".to_string())?;
            Ok(())
        }
        _ => Err(format!(
            "`input[*].output[*].type` contains unsupported value `{part_type}`"
        )),
    }
}

fn validate_reasoning_input(
    obj: &serde_json::Map<String, serde_json::Value>,
) -> std::result::Result<(), String> {
    let Some(summary) = obj.get("summary") else {
        return Ok(());
    };
    let items = summary
        .as_array()
        .ok_or_else(|| "`input[*].summary` must be an array".to_string())?;
    for item in items {
        let summary_obj = item
            .as_object()
            .ok_or_else(|| "`input[*].summary[*]` must be an object".to_string())?;
        if summary_obj.get("type").and_then(|v| v.as_str()) != Some("summary_text") {
            return Err("`input[*].summary[*].type` must be `summary_text`".to_string());
        }
        let _ = summary_obj
            .get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "`input[*].summary[*].text` must be a string".to_string())?;
    }
    Ok(())
}

fn validate_message_input(
    obj: &serde_json::Map<String, serde_json::Value>,
) -> std::result::Result<(), String> {
    let role = obj
        .get("role")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "`input[*].role` must be a string".to_string())?;
    if !matches!(role, "user" | "system" | "developer" | "assistant") {
        return Err("`input[*].role` contains unsupported value".to_string());
    }

    let content = obj
        .get("content")
        .ok_or_else(|| "`input[*].content` is required".to_string())?;
    if content.is_string() {
        return Ok(());
    }

    let parts = content
        .as_array()
        .ok_or_else(|| "`input[*].content` must be a string or array".to_string())?;
    for part in parts {
        let part_obj = part
            .as_object()
            .ok_or_else(|| "`input[*].content[*]` must be an object".to_string())?;
        let part_type = part_obj
            .get("type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "`input[*].content[*].type` must be a string".to_string())?;
        match role {
            "user" => match part_type {
                "input_text" => {
                    let _ = part_obj
                        .get("text")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| "`input[*].content[*].text` must be a string".to_string())?;
                }
                "input_image" => validate_image_part(part_obj, "input[*].content[*]")?,
                "input_file" => validate_file_part(part_obj, "input[*].content[*]")?,
                _ => return Err(format!("unsupported user content part type `{part_type}`")),
            },
            "system" | "developer" => {
                if part_type != "input_text" {
                    return Err(format!(
                        "unsupported {role} content part type `{part_type}`"
                    ));
                }
                let _ = part_obj
                    .get("text")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "`input[*].content[*].text` must be a string".to_string())?;
            }
            "assistant" => match part_type {
                "output_text" => validate_output_text_part(part_obj)?,
                "refusal" => {
                    let _ = part_obj
                        .get("refusal")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| {
                            "`input[*].content[*].refusal` must be a string".to_string()
                        })?;
                }
                _ => {
                    return Err(format!(
                        "unsupported assistant content part type `{part_type}`"
                    ))
                }
            },
            _ => unreachable!(),
        }
    }

    Ok(())
}

fn validate_output_text_part(
    part_obj: &serde_json::Map<String, serde_json::Value>,
) -> std::result::Result<(), String> {
    let _ = part_obj
        .get("text")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "`input[*].content[*].text` must be a string".to_string())?;
    if let Some(annotations) = part_obj.get("annotations") {
        let annotations = annotations
            .as_array()
            .ok_or_else(|| "`input[*].content[*].annotations` must be an array".to_string())?;
        for annotation in annotations {
            let annotation_obj = annotation.as_object().ok_or_else(|| {
                "`input[*].content[*].annotations[*]` must be an object".to_string()
            })?;
            if annotation_obj.get("type").and_then(|v| v.as_str()) != Some("url_citation") {
                return Err(
                    "`input[*].content[*].annotations[*].type` must be `url_citation`".to_string(),
                );
            }
            let _ = annotation_obj
                .get("start_index")
                .and_then(|v| v.as_i64())
                .ok_or_else(|| {
                    "`input[*].content[*].annotations[*].start_index` must be an integer"
                        .to_string()
                })?;
            let _ = annotation_obj
                .get("end_index")
                .and_then(|v| v.as_i64())
                .ok_or_else(|| {
                    "`input[*].content[*].annotations[*].end_index` must be an integer".to_string()
                })?;
            let _ = annotation_obj
                .get("url")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    "`input[*].content[*].annotations[*].url` must be a string".to_string()
                })?;
            let _ = annotation_obj
                .get("title")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    "`input[*].content[*].annotations[*].title` must be a string".to_string()
                })?;
        }
    }
    Ok(())
}

fn validate_image_part(
    part_obj: &serde_json::Map<String, serde_json::Value>,
    context: &str,
) -> std::result::Result<(), String> {
    let image_url = part_obj
        .get("image_url")
        .ok_or_else(|| format!("`{context}.image_url` is required"))?;
    if !(image_url.is_string() || image_url.is_null()) {
        return Err(format!("`{context}.image_url` must be a string or null"));
    }
    if let Some(detail_value) = part_obj.get("detail") {
        let detail = detail_value
            .as_str()
            .ok_or_else(|| format!("`{context}.detail` must be a string"))?;
        if !matches!(detail, "auto" | "low" | "high") {
            return Err(format!(
                "`{context}.detail` must be one of: auto, low, high"
            ));
        }
    }
    Ok(())
}

fn validate_file_part(
    part_obj: &serde_json::Map<String, serde_json::Value>,
    context: &str,
) -> std::result::Result<(), String> {
    if let Some(file_data) = part_obj.get("file_data") {
        if !(file_data.is_string() || file_data.is_null()) {
            return Err(format!("`{context}.file_data` must be a string or null"));
        }
    }
    if let Some(file_url) = part_obj.get("file_url") {
        if !(file_url.is_string() || file_url.is_null()) {
            return Err(format!("`{context}.file_url` must be a string or null"));
        }
    }
    if let Some(filename) = part_obj.get("filename") {
        if !filename.is_string() {
            return Err(format!("`{context}.filename` must be a string"));
        }
    }
    Ok(())
}

fn validate_tools_shape(tools: &serde_json::Value) -> std::result::Result<(), String> {
    let tools = tools
        .as_array()
        .ok_or_else(|| "`tools` must be an array".to_string())?;
    for tool in tools {
        let tool_obj = tool
            .as_object()
            .ok_or_else(|| "`tools[*]` must be an object".to_string())?;
        if tool_obj.get("type").and_then(|v| v.as_str()) != Some("function") {
            return Err("`tools[*].type` must be `function`".to_string());
        }
        let name = tool_obj
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "`tools[*].name` must be a string".to_string())?;
        if !is_valid_tool_or_function_name(name) {
            return Err("`tools[*].name` must match ^[a-zA-Z0-9_-]{1,64}$".to_string());
        }
        if let Some(parameters) = tool_obj.get("parameters") {
            if !parameters.is_object() {
                return Err("`tools[*].parameters` must be an object".to_string());
            }
        }
        if let Some(strict) = tool_obj.get("strict") {
            if !strict.is_boolean() {
                return Err("`tools[*].strict` must be a boolean".to_string());
            }
        }
    }
    Ok(())
}

fn parse_stream_options_config(
    stream_options: Option<&serde_json::Value>,
) -> std::result::Result<(), String> {
    let Some(stream_options_value) = stream_options else {
        return Ok(());
    };
    let obj = stream_options_value
        .as_object()
        .ok_or_else(|| "`stream_options` must be an object".to_string())?;

    for (key, value) in obj {
        match key.as_str() {
            "include_obfuscation" => {
                if !value.is_boolean() {
                    return Err(
                        "`stream_options.include_obfuscation` must be a boolean".to_string()
                    );
                }
            }
            _ => return Err(format!("`stream_options.{key}` is not supported")),
        }
    }

    Ok(())
}

fn validate_service_tier(service_tier: Option<&str>) -> std::result::Result<(), String> {
    let Some(service_tier) = service_tier else {
        return Ok(());
    };
    if matches!(service_tier, "auto" | "default" | "flex" | "priority") {
        return Ok(());
    }
    Err("`service_tier` must be one of: auto, default, flex, priority".to_string())
}

fn validate_safety_identifier(safety_identifier: Option<&str>) -> std::result::Result<(), String> {
    let Some(safety_identifier) = safety_identifier else {
        return Ok(());
    };
    if safety_identifier.chars().count() <= 64 {
        return Ok(());
    }
    Err("`safety_identifier` must be at most 64 characters".to_string())
}

fn validate_truncation(truncation: Option<&str>) -> std::result::Result<(), String> {
    let Some(truncation) = truncation else {
        return Ok(());
    };
    match truncation {
        "auto" | "disabled" => Ok(()),
        _ => Err("`truncation` must be one of: auto, disabled".to_string()),
    }
}

fn validate_responses_request(request: &responses_types::CreateResponseBody) -> Result<(), String> {
    if let Some(input) = request.input.as_ref() {
        if !(input.is_string() || input.is_array() || input.is_null()) {
            return Err("`input` must be a string, an array, or null".to_string());
        }
        if let Some(text) = input.as_str() {
            if text.chars().count() > 10_485_760 {
                return Err("`input` string must be at most 10485760 characters".to_string());
            }
        }
        if input.is_array() {
            validate_input_items(input)?;
        }
    }

    validate_metadata(request.metadata.as_ref())?;

    if let Some(tools) = request.tools.as_ref() {
        validate_tools_shape(tools)?;
    }

    if let Some(tool_choice) = request.tool_choice.as_ref() {
        validate_tool_choice(tool_choice)?;
    }

    parse_requested_text_format(request.text.as_ref())?;
    let _ = parse_logprob_config(request.include.as_ref(), request.top_logprobs)?;
    let _ = parse_reasoning_config(request.reasoning.as_ref())?;
    validate_temperature(request.temperature)?;
    validate_top_p(request.top_p)?;
    validate_max_output_tokens(request.max_output_tokens)?;
    validate_max_tool_calls(request.max_tool_calls)?;
    validate_penalty_bounds("presence_penalty", request.presence_penalty)?;
    validate_penalty_bounds("frequency_penalty", request.frequency_penalty)?;
    parse_stream_options_config(request.stream_options.as_ref())?;
    validate_service_tier(request.service_tier.as_deref())?;
    validate_safety_identifier(request.safety_identifier.as_deref())?;
    validate_prompt_cache_key(request.prompt_cache_key.as_deref())?;
    validate_truncation(request.truncation.as_deref())?;
    Ok(())
}

fn validate_tool_choice(tool_choice: &serde_json::Value) -> Result<(), String> {
    if let Some(choice) = tool_choice.as_str() {
        if matches!(choice, "none" | "auto" | "required") {
            return Ok(());
        }
        return Err("`tool_choice` string must be one of: none, auto, required".to_string());
    }

    let obj = tool_choice
        .as_object()
        .ok_or_else(|| "`tool_choice` must be a string or object".to_string())?;
    let choice_type = obj
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "`tool_choice.type` must be a string".to_string())?;

    match choice_type {
        "function" => {
            if obj.get("name").and_then(|v| v.as_str()).is_none() {
                return Err("`tool_choice` of type `function` requires a string `name`".to_string());
            }
            Ok(())
        }
        "allowed_tools" => {
            let tools = obj.get("tools").and_then(|v| v.as_array()).ok_or_else(|| {
                "`tool_choice` of type `allowed_tools` requires `tools` array".to_string()
            })?;
            if tools.is_empty() {
                return Err("`tool_choice.tools` must contain at least one tool".to_string());
            }
            if tools.len() > 128 {
                return Err("`tool_choice.tools` must contain at most 128 tools".to_string());
            }
            for tool in tools {
                let tool_obj = tool
                    .as_object()
                    .ok_or_else(|| "`tool_choice.tools[*]` must be an object".to_string())?;
                if tool_obj.get("type").and_then(|v| v.as_str()) != Some("function")
                    || tool_obj.get("name").and_then(|v| v.as_str()).is_none()
                {
                    return Err(
                        "`tool_choice.tools[*]` must be `{ \"type\": \"function\", \"name\": \"...\" }`"
                            .to_string(),
                    );
                }
            }
            if let Some(mode) = obj.get("mode").and_then(|v| v.as_str()) {
                if !matches!(mode, "none" | "auto" | "required") {
                    return Err(
                        "`tool_choice.mode` must be one of: none, auto, required".to_string()
                    );
                }
            }
            Ok(())
        }
        _ => Err("`tool_choice.type` must be `function` or `allowed_tools`".to_string()),
    }
}

fn json_error(status: StatusCode, code: &str, message: &str) -> Response<BoxBody> {
    let payload = json!({
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

fn api_error(
    strict_responses: bool,
    status: StatusCode,
    code: &str,
    message: &str,
) -> Response<BoxBody> {
    if !strict_responses {
        return json_error(status, code, message);
    }
    let error_type = if status.is_client_error() {
        "invalid_request_error"
    } else {
        "server_error"
    };
    let payload = json!({
        "error": {
            "type": error_type,
            "code": code,
            "message": message,
            "param": null
        }
    });
    let body = serde_json::to_vec(&payload).unwrap_or_else(|_| b"{}".to_vec());
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body)).boxed())
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_request() -> responses_types::CreateResponseBody {
        responses_types::CreateResponseBody {
            background: None,
            frequency_penalty: None,
            include: None,
            input: None,
            instructions: None,
            max_output_tokens: None,
            max_tool_calls: None,
            metadata: None,
            model: None,
            parallel_tool_calls: None,
            presence_penalty: None,
            previous_response_id: None,
            prompt_cache_key: None,
            reasoning: None,
            safety_identifier: None,
            service_tier: None,
            seed: None,
            store: None,
            stream: None,
            stream_options: None,
            temperature: None,
            text: None,
            tool_choice: None,
            tools: None,
            top_logprobs: None,
            top_k: None,
            top_p: None,
            truncation: None,
            max_completion_tokens: None,
            max_reasoning_tokens: None,
        }
    }

    #[test]
    fn validate_responses_request_rejects_object_input() {
        let mut req = empty_request();
        req.input = Some(json!({"bad": "shape"}));
        assert!(validate_responses_request(&req).is_err());
    }

    #[test]
    fn validate_responses_request_rejects_too_long_input_string() {
        let mut req = empty_request();
        req.input = Some(json!("a".repeat(10_485_761)));
        let err = validate_responses_request(&req).expect_err("must reject");
        assert!(err.contains("10485760"), "err={err}");
    }

    #[test]
    fn validate_metadata_rejects_out_of_contract_values() {
        let too_many = (0..17)
            .map(|i| (format!("k{i}"), json!("v")))
            .collect::<serde_json::Map<String, serde_json::Value>>();
        assert!(validate_metadata(Some(&json!({"k": "v"}))).is_ok());
        assert!(validate_metadata(Some(&json!("bad"))).is_err());
        assert!(validate_metadata(Some(&json!(too_many))).is_err());
        assert!(validate_metadata(Some(&json!({ "k".repeat(65): "v" }))).is_err());
        assert!(validate_metadata(Some(&json!({ "k": "v".repeat(513) }))).is_err());
        assert!(validate_metadata(Some(&json!({ "k": 1 }))).is_err());
    }

    #[test]
    fn validate_prompt_cache_key_rejects_too_long_value() {
        assert!(validate_prompt_cache_key(Some(&"p".repeat(64))).is_ok());
        assert!(validate_prompt_cache_key(Some(&"p".repeat(65))).is_err());
    }

    #[test]
    fn validate_max_tool_calls_rejects_values_below_minimum() {
        assert!(validate_max_tool_calls(Some(0)).is_err());
        assert!(validate_max_tool_calls(Some(1)).is_ok());
    }

    #[test]
    fn validate_tool_choice_rejects_invalid_string() {
        assert!(validate_tool_choice(&json!("sometimes")).is_err());
    }

    #[test]
    fn parse_logprob_config_accepts_message_output_text_include() {
        let include = json!(["message.output_text.logprobs"]);
        assert!(parse_logprob_config(Some(&include), None).is_ok());
    }

    #[test]
    fn parse_logprob_config_rejects_unknown_include_value() {
        let include = json!(["message.output_text.unknown"]);
        assert!(parse_logprob_config(Some(&include), None).is_err());
    }

    #[test]
    fn parse_logprob_config_rejects_out_of_range_top_logprobs() {
        assert!(parse_logprob_config(None, Some(21)).is_err());
    }

    #[test]
    fn parse_logprob_config_accepts_positive_top_logprobs() {
        let parsed = parse_logprob_config(None, Some(3)).expect("must parse");
        assert_eq!(parsed.top_logprobs, 3);
    }

    #[test]
    fn parse_reasoning_config_accepts_supported_values() {
        let reasoning = json!({
            "effort": "medium",
            "summary": "concise"
        });
        let parsed = parse_reasoning_config(Some(&reasoning)).expect("must parse");
        assert_eq!(parsed.effort.as_deref(), Some("medium"));
        assert_eq!(parsed.summary.as_deref(), Some("concise"));
    }

    #[test]
    fn parse_reasoning_config_rejects_unknown_effort() {
        let reasoning = json!({ "effort": "max" });
        assert!(parse_reasoning_config(Some(&reasoning)).is_err());
    }

    #[test]
    fn parse_requested_text_format_rejects_json_object() {
        let text = json!({
            "format": {
                "type": "json_object"
            }
        });
        assert!(parse_requested_text_format(Some(&text)).is_err());
    }

    #[test]
    fn parse_requested_text_format_accepts_json_schema_and_null() {
        let schema = json!({
            "format": {
                "type": "json_schema",
                "name": "reply",
                "schema": { "type": "object" }
            }
        });
        let parsed_schema = parse_requested_text_format(Some(&schema)).expect("valid format");
        assert!(matches!(parsed_schema, Some(TextFormatConfig::JsonSchema)));

        let null_format = json!({
            "format": null
        });
        let parsed_null = parse_requested_text_format(Some(&null_format)).expect("valid null");
        assert!(parsed_null.is_none());
    }

    #[test]
    fn parse_stream_options_accepts_include_obfuscation() {
        let stream_options = json!({
            "include_obfuscation": true
        });
        assert!(parse_stream_options_config(Some(&stream_options)).is_ok());
    }

    #[test]
    fn parse_stream_options_rejects_legacy_include_usage_key() {
        let stream_options = json!({
            "include_usage": true
        });
        let err = parse_stream_options_config(Some(&stream_options)).expect_err("must reject");
        assert!(err.contains("stream_options.include_usage"));
    }

    #[test]
    fn validate_truncation_accepts_disabled() {
        assert!(validate_truncation(Some("disabled")).is_ok());
    }

    #[test]
    fn validate_temperature_rejects_out_of_range_values() {
        assert!(validate_temperature(Some(-0.1)).is_err());
        assert!(validate_temperature(Some(2.1)).is_err());
        assert!(validate_temperature(Some(0.0)).is_ok());
        assert!(validate_temperature(Some(2.0)).is_ok());
    }

    #[test]
    fn validate_top_p_rejects_out_of_range_values() {
        assert!(validate_top_p(Some(-0.1)).is_err());
        assert!(validate_top_p(Some(1.1)).is_err());
        assert!(validate_top_p(Some(0.0)).is_ok());
        assert!(validate_top_p(Some(1.0)).is_ok());
    }

    #[test]
    fn validate_max_output_tokens_rejects_values_below_minimum() {
        assert!(validate_max_output_tokens(Some(0)).is_err());
        assert!(validate_max_output_tokens(Some(-1)).is_err());
        assert!(validate_max_output_tokens(Some(1)).is_ok());
    }

    #[test]
    fn extract_urls_with_offsets_detects_urls_and_trims_trailing_punctuation() {
        let text = "See https://example.com/a, and http://foo.test/x.";
        let urls = extract_urls_with_offsets(text);
        assert_eq!(urls.len(), 2);
        assert_eq!(urls[0].2, "https://example.com/a");
        assert_eq!(urls[1].2, "http://foo.test/x");
    }
}
