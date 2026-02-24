use std::collections::HashSet;
use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

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
use crate::server::state::{AppState, StoredResponse};
use talu::documents::{DocumentError, DocumentsHandle};
use talu::responses::{ContentType, ItemType};
use talu::{ChatHandle, FinishReason};

type BoxBody = http_body_util::combinators::BoxBody<Bytes, Infallible>;

#[derive(Debug, Clone)]
struct GenerationRequest {
    model: Option<String>,
    max_output_tokens: Option<i64>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    instructions: Option<String>,
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

    let request_value: serde_json::Value = match serde_json::from_slice(&body_bytes) {
        Ok(val) => val,
        Err(_) => {
            return api_error(
                strict_responses,
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "Invalid JSON",
            )
        }
    };
    let request = GenerationRequest {
        model: request_value
            .get("model")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        max_output_tokens: request_value
            .get("max_output_tokens")
            .and_then(|v| v.as_i64()),
        temperature: request_value.get("temperature").and_then(|v| v.as_f64()),
        top_p: request_value.get("top_p").and_then(|v| v.as_f64()),
        instructions: request_value
            .get("instructions")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
    };

    let stream = request_value
        .get("stream")
        .and_then(|val| val.as_bool())
        .unwrap_or(false);

    let tools_json = request_value.get("tools").cloned();
    let tool_choice_json = request_value.get("tool_choice").cloned();
    let previous_response_id = request_value
        .get("previous_response_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let input_value = request_value.get("input").cloned();
    let store = request_value
        .get("store")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let request_session_id = if strict_responses {
        None
    } else {
        request_value
            .get("session_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    };
    let prompt_id = if strict_responses {
        None
    } else {
        request_value
            .get("prompt_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    };

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

    let docs = DocumentsHandle::open(storage_path)
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
    strict_responses: bool,
    auth_ctx: Option<&AuthContext>,
) -> Result<serde_json::Value, ResponseError> {
    let request_max_output_tokens = request.max_output_tokens;
    let temperature = request.temperature;
    let top_p = request.top_p;
    let instructions = request.instructions;

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

    // If prompt_id is provided, fetch the document and extract system prompt.
    let system_prompt_from_doc: Option<String> = if let Some(ref pid) = prompt_id {
        if let Some(ref bp) = bucket_path {
            match DocumentsHandle::open(bp) {
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

    let session_id_for_task = session_id.clone();
    let bucket_for_task = bucket_path.clone();
    let model_id_for_task = model_id.clone();
    let system_prompt_for_task = effective_system_prompt.clone();
    let prompt_id_for_task = prompt_id.clone();
    let (output_items, prompt_tokens, completion_tokens, responses_json) =
        tokio::task::spawn_blocking(move || {
            let mut backend = backend.blocking_lock();
            let backend = backend
                .backend
                .as_mut()
                .ok_or_else(|| anyhow!("no backend available"))?;
            // Create ChatHandle with system prompt if prompt_id was provided
            log::trace!(target: "server::gen", "ChatHandle::new(system_prompt={})", system_prompt_for_task.is_some());
            let chat = ChatHandle::new(system_prompt_for_task.as_deref())?;

            // Enable storage persistence if store=true and bucket is configured.
            // set_storage_db auto-loads any existing items for this session.
            if let Some(ref bp) = bucket_for_task {
                if let Some(bp_str) = bp.to_str() {
                    log::trace!(target: "server::gen", "set_storage_db({:?}, {})", bp_str, session_id_for_task);
                    chat.set_storage_db(bp_str, &session_id_for_task)
                        .map_err(|e| anyhow!("failed to set storage: {}", e))?;
                }
            }

            // Restore previous conversation from in-memory JSON (only when
            // storage is not active — set_storage_db already loaded items).
            if let Some(ref prev) = prev_json {
                log::trace!(target: "server::gen", "load_responses_json(prev, {} bytes)", prev.len());
                chat.load_responses_json(prev)
                    .map_err(|e| anyhow!("failed to load previous conversation: {}", e))?;
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
                );
            }

            let mut cfg = talu::router::GenerateConfig::default();
            if let Some(max_tokens) = max_output_tokens {
                cfg.max_tokens = max_tokens as usize;
            }
            if let Some(temp) = temperature {
                cfg.temperature = temp as f32;
            }
            if let Some(top_p) = top_p {
                cfg.top_p = top_p as f32;
            }

            log::debug!(target: "server::gen", "generating: model={} max_tokens={:?} temp={:?} top_p={:?}",
                model_id_for_task, max_output_tokens, temperature, top_p);
            log::trace!(target: "server::gen", "generate(items={}, cfg={{max_tokens={}, temp={}, top_p={}}})",
                pre_gen_count, cfg.max_tokens, cfg.temperature, cfg.top_p);
            let result = talu::router::generate(&chat, &[], backend, &cfg)
                .map_err(|e| anyhow!("generation failed: {}", e))?;

            let prompt_tokens = result.prompt_tokens();
            let completion_tokens = result.completion_tokens();
            log::debug!(target: "server::gen", "completed: prompt_tokens={} completion_tokens={}",
                prompt_tokens, completion_tokens);

            // Serialize ALL items (response direction), then slice to output only.
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

            // Store full conversation for chaining.
            let responses_json = all_json;

            Ok::<_, anyhow::Error>((
                output_items,
                prompt_tokens,
                completion_tokens,
                responses_json,
            ))
        })
        .await
        .context("Generation task failed")??;

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
    };
    let mut response_value = build_response_resource_value(
        &response_id,
        &model_id,
        created_at,
        Some(now_unix_seconds()),
        output_items,
        max_output_tokens,
        temperature.unwrap_or(0.0),
        top_p.unwrap_or(1.0),
        "completed",
        Some(&usage),
        effective_tools.as_ref(),
        effective_tool_choice.as_ref(),
        store,
        instructions.as_deref(),
    );

    // Set previous_response_id on the response resource.
    if let Some(ref prev_id) = previous_response_id {
        response_value["previous_response_id"] = json!(prev_id);
    }

    if !strict_responses {
        // Include session_id in metadata so the UI can adopt it for follow-ups.
        response_value["metadata"] = json!({ "session_id": session_id_for_response });
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
    strict_responses: bool,
    auth_ctx: Option<AuthContext>,
) -> Response<BoxBody> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Bytes>();
    let request_id = format!("req_{}", random_id());
    let response_id = format!("resp_{}", random_id());
    let message_id = format!("msg_{}", random_id());
    let created_at = now_unix_seconds();

    let request_max_output_tokens = request.max_output_tokens;
    let temperature = request.temperature;
    let top_p = request.top_p;
    let instructions = request.instructions;
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

    // If prompt_id is provided, fetch the document and extract system prompt.
    let system_prompt_from_doc: Option<String> = if let Some(ref pid) = prompt_id {
        if let Some(ref bp) = bucket_path {
            match DocumentsHandle::open(bp) {
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

    // Create a stop flag for cancellation on client disconnect.
    let stop_flag = Arc::new(AtomicBool::new(false));
    let stop_flag_for_gen = stop_flag.clone();

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

    // Clones for post-generation title generation.
    let title_input = input_string.clone();
    let title_bucket = bucket_path.clone();
    let title_session_id = session_id.clone();
    let title_backend = state.backend.clone();
    let events_tenant_id = auth_ctx.as_ref().map(|ctx| ctx.tenant_id.clone());
    let events_request_id = request_id.clone();
    let events_response_id = response_id.clone();
    let events_session_id = session_id.clone();

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
                        "text": { "format": { "type": "text" } },
                        "top_p": top_p.unwrap_or(1.0),
                        "presence_penalty": 0.0,
                        "frequency_penalty": 0.0,
                        "top_logprobs": 0,
                        "temperature": temperature.unwrap_or(0.0),
                        "reasoning": { "effort": null, "summary": null },
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
                match provider::create_backend_for_model_with_progress(&model_id, callback) {
                    Ok(new_backend) => {
                        guard.backend = Some(new_backend);
                        guard.current_model = Some(model_id.clone());
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
            "in_progress",
            None,
            tools_for_events.as_ref(),
            tool_choice_for_events.as_ref(),
            store,
            instructions.as_deref(),
        ));
        if let Some(ref prev_id) = previous_response_id_for_ctx {
            created_response["previous_response_id"] = json!(prev_id);
        }
        if !strict_responses {
            // Include session_id in the very first event so the UI can track
            // the conversation immediately (before generation completes).
            created_response["metadata"] = json!({ "session_id": session_id });
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
        }));
        let ctx_for_complete = ctx.clone();

        let gen_result = run_streaming_generation(
            backend,
            input_json,
            input_string,
            prev_json,
            bucket_path,
            session_id,
            model_id.clone(),
            max_output_tokens,
            temperature,
            top_p,
            effective_system_prompt,
            prompt_id,
            file_storage_path,
            ctx,
            stop_flag_for_gen,
            events_tenant_id.clone(),
            Some(events_request_id.clone()),
            Some(events_response_id.clone()),
            Some(events_session_id.clone()),
        );

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

fn run_streaming_generation(
    backend: Arc<tokio::sync::Mutex<crate::server::state::BackendState>>,
    input_json: Option<String>,
    input_string: Option<String>,
    prev_json: Option<String>,
    bucket_path: Option<std::path::PathBuf>,
    session_id: String,
    model_id: String,
    max_output_tokens: Option<i64>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    system_prompt: Option<String>,
    prompt_id: Option<String>,
    file_storage_path: Option<std::path::PathBuf>,
    ctx: Arc<std::sync::Mutex<StreamCtx>>,
    stop_flag: Arc<AtomicBool>,
    event_tenant_id: Option<String>,
    event_request_id: Option<String>,
    event_response_id: Option<String>,
    event_session_id: Option<String>,
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
    if let Some(ref bp) = bucket_path {
        if let Some(bp_str) = bp.to_str() {
            log::trace!(target: "server::gen", "set_storage_db({:?}, {})", bp_str, session_id);
            chat.set_storage_db(bp_str, &session_id)
                .map_err(|e| anyhow!("failed to set storage: {}", e))?;
        }
    }

    // Restore previous conversation from in-memory JSON (only when
    // storage is not active — set_storage_db already loaded items).
    if let Some(ref prev) = prev_json {
        log::trace!(target: "server::gen", "load_responses_json(prev, {} bytes)", prev.len());
        chat.load_responses_json(prev)
            .map_err(|e| anyhow!("failed to load previous conversation: {}", e))?;
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
    if let Some(max_tokens) = max_output_tokens {
        cfg.max_tokens = max_tokens as usize;
    }
    if let Some(temp) = temperature {
        cfg.temperature = temp as f32;
    }
    if let Some(top_p) = top_p {
        cfg.top_p = top_p as f32;
    }

    // Pass the stop flag for cooperative cancellation.
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

    Ok(StreamGenResult {
        output_items,
        usage: UsageStats {
            input_tokens: stream_result.prompt_tokens,
            output_tokens: stream_result.completion_tokens,
        },
        finish_reason: stream_result.finish_reason,
        responses_json: all_json,
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
}

impl StreamCtx {
    fn send_delta(&mut self, token: &talu::router::StreamToken) -> Result<()> {
        let item_changed = self.cur_item_type != Some(token.item_type);
        let content_changed = item_changed || self.cur_content_type != Some(token.content_type);

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
            self.emit_content_part_added(token.content_type)?;
        }

        self.accumulated_text.push_str(token.text);

        // Emit the delta event.
        let event_name =
            talu::responses::stream_delta_event_name(token.item_type, token.content_type);
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

        // Detect client disconnect: if send fails, the client has disconnected.
        // Signal the stop flag to halt generation gracefully.
        if send_event(&self.tx, event_name, payload).is_err() {
            self.stop_flag.store(true, Ordering::Release);
            return Err(anyhow!("client disconnected"));
        }
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
            "part": content_part_done_payload(part_type, "")
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
            "part": content_part_done_payload(part_type, &self.accumulated_text)
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
            ItemType::Reasoning => json!({
                "type": "reasoning",
                "id": self.cur_item_id,
                "summary": [],
                "status": "completed"
            }),
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
                        "annotations": [],
                        "logprobs": []
                    }),
                }]
            }),
        };

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

    fn flush_completion(
        &mut self,
        model_id: String,
        created_at: i64,
        max_output_tokens: Option<i64>,
        temperature: f64,
        top_p: f64,
        result: Result<StreamGenResult>,
    ) -> Result<()> {
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
                let output_items = r.output_items;
                let mut response = build_response_resource_value(
                    &self.response_id,
                    &model_id,
                    created_at,
                    Some(now_unix_seconds()),
                    output_items,
                    max_output_tokens,
                    temperature,
                    top_p,
                    status,
                    Some(&r.usage),
                    self.tools_json.as_ref(),
                    self.tool_choice_json.as_ref(),
                    self.request_store,
                    self.instructions.as_deref(),
                );
                if let Some(ref prev_id) = self.previous_response_id {
                    response["previous_response_id"] = json!(prev_id);
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
                    response["metadata"] = json!({ "session_id": self.session_id });
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
                    "failed",
                    None,
                    self.tools_json.as_ref(),
                    self.tool_choice_json.as_ref(),
                    self.request_store,
                    self.instructions.as_deref(),
                );
                if let Some(ref prev_id) = self.previous_response_id {
                    response["previous_response_id"] = json!(prev_id);
                }
                response["error"] = json!({
                    "code": "server_error",
                    "message": format!("{err}")
                });
                if !self.strict_responses {
                    response["metadata"] = json!({ "session_id": self.session_id });
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

fn content_part_done_payload(part_type: &str, text: &str) -> serde_json::Value {
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
            "annotations": [],
            "logprobs": []
        }),
    }
}

/// Token usage statistics for the response resource.
struct UsageStats {
    input_tokens: usize,
    output_tokens: usize,
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
    status: &str,
    usage: Option<&UsageStats>,
    tools: Option<&serde_json::Value>,
    tool_choice: Option<&serde_json::Value>,
    store: bool,
    instructions: Option<&str>,
) -> serde_json::Value {
    let usage_value = match usage {
        Some(u) => json!({
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "total_tokens": u.input_tokens + u.output_tokens,
            "input_tokens_details": {
                "cached_tokens": 0
            },
            "output_tokens_details": {
                "reasoning_tokens": 0
            }
        }),
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
        "text": { "format": { "type": "text" } },
        "top_p": top_p,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "top_logprobs": 0,
        "temperature": temperature,
        "reasoning": { "effort": null, "summary": null },
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

    log::info!(target: "server::gen", "loading backend for model {}", model_id);
    let model = model_id.to_string();
    let backend =
        tokio::task::spawn_blocking(move || provider::create_backend_for_model(&model)).await??;

    let mut guard = state.backend.lock().await;
    guard.backend = Some(backend);
    guard.current_model = Some(model_id.to_string());
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

fn validate_responses_request(request: &responses_types::CreateResponseBody) -> Result<(), String> {
    if let Some(input) = request.input.as_ref() {
        if !(input.is_string() || input.is_array() || input.is_null()) {
            return Err("`input` must be a string, an array, or null".to_string());
        }
        if input.is_array() {
            let serialized = serde_json::to_string(input)
                .map_err(|_| "failed to serialize `input` array".to_string())?;
            let mut conv = talu::responses::ResponsesHandle::new()
                .map_err(|e| format!("failed to validate `input` items: {e}"))?;
            conv.load_responses_json(&serialized).map_err(|e| {
                format!("`input` array contains unsupported item/content shape: {e}")
            })?;
        }
    }

    if let Some(metadata) = request.metadata.as_ref() {
        if !metadata.is_object() {
            return Err("`metadata` must be an object".to_string());
        }
    }

    if let Some(tools) = request.tools.as_ref() {
        if !tools.is_array() {
            return Err("`tools` must be an array".to_string());
        }
    }

    if let Some(tool_choice) = request.tool_choice.as_ref() {
        validate_tool_choice(tool_choice)?;
    }

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

    #[test]
    fn validate_responses_request_rejects_object_input() {
        let req = responses_types::CreateResponseBody {
            background: None,
            frequency_penalty: None,
            include: None,
            input: Some(json!({"bad": "shape"})),
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
            store: None,
            stream: None,
            stream_options: None,
            temperature: None,
            text: None,
            tool_choice: None,
            tools: None,
            top_logprobs: None,
            top_p: None,
            truncation: None,
        };
        assert!(validate_responses_request(&req).is_err());
    }

    #[test]
    fn validate_tool_choice_rejects_invalid_string() {
        assert!(validate_tool_choice(&json!("sometimes")).is_err());
    }
}
