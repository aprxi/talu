//! Search API handlers for POST /v1/search.
//!
//! Provides unified federated search across conversations and documents with multiple modes:
//! - `text`: Case-insensitive substring search
//! - `regex`: Pattern matching (future)
//! - `vector`: Semantic similarity search (future)
//!
//! Supported scopes:
//! - `conversations`: Search conversation metadata and content
//! - `documents`: Search document metadata and content
//! - `items`: Search conversation items
//! - `all`: Federated search across both conversations and documents

use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::{Request, Response, StatusCode};
use serde::{Deserialize, Serialize};

use talu::documents::DocumentsHandle;
use talu::storage::{SearchParams, SessionRecordFull, StorageError, StorageHandle};

use crate::server::auth_gateway::AuthContext;
use crate::server::conversations::{
    decode_cursor, encode_cursor, resolve_tags_for_session, session_to_conversation_json,
};
use crate::server::state::AppState;

// ---------------------------------------------------------------------------
// Request/Response types
// ---------------------------------------------------------------------------

/// Search request body.
#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    /// Scope of search: "conversations" or "items"
    pub scope: String,

    /// Text search (case-insensitive substring)
    #[serde(default)]
    pub text: Option<String>,

    /// Regex pattern search (future)
    #[serde(default)]
    pub regex: Option<String>,

    /// Vector/semantic search (future)
    #[serde(default)]
    pub vector: Option<VectorSearch>,

    /// Structured filters
    #[serde(default)]
    pub filters: Option<SearchFilters>,

    /// Aggregations to compute
    #[serde(default)]
    pub aggregations: Option<Vec<String>>,

    /// Max results to return (default 20, max 100)
    #[serde(default)]
    pub limit: Option<usize>,

    /// Pagination cursor
    #[serde(default)]
    pub cursor: Option<String>,

    /// Include conversation items in response
    #[serde(default)]
    pub include_items: Option<bool>,

    /// Include match snippets/highlights
    #[serde(default)]
    pub highlight: Option<bool>,
}

/// Vector search options.
#[derive(Debug, Deserialize)]
pub struct VectorSearch {
    /// Text to embed and search
    pub text: Option<String>,
    /// Pre-computed embedding vector
    pub embedding: Option<Vec<f32>>,
    /// Find similar to this conversation ID
    pub similar_to: Option<String>,
    /// Minimum similarity score threshold
    pub min_score: Option<f32>,
}

/// Structured filters for search.
#[derive(Debug, Deserialize)]
pub struct SearchFilters {
    /// Tags (AND logic) - must have ALL
    #[serde(default)]
    pub tags: Option<Vec<String>>,

    /// Tags (OR logic) - must have ANY
    #[serde(default)]
    pub tags_any: Option<Vec<String>>,

    /// Model filter (supports wildcards like "qwen*")
    #[serde(default)]
    pub model: Option<String>,

    /// Created after timestamp (ms)
    #[serde(default)]
    pub created_after: Option<i64>,

    /// Created before timestamp (ms)
    #[serde(default)]
    pub created_before: Option<i64>,

    /// Updated after timestamp (ms)
    #[serde(default)]
    pub updated_after: Option<i64>,

    /// Updated before timestamp (ms)
    #[serde(default)]
    pub updated_before: Option<i64>,

    /// Marker exact match
    #[serde(default)]
    pub marker: Option<String>,

    /// Marker any (OR logic)
    #[serde(default)]
    pub marker_any: Option<Vec<String>>,

    /// Has any tags
    #[serde(default)]
    pub has_tags: Option<bool>,

    /// Group ID (multi-tenant filter)
    #[serde(default)]
    pub group_id: Option<String>,
}

/// Search response.
#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub data: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aggregations: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
    pub has_more: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

/// POST /v1/search - Unified search endpoint.
pub async fn handle_search(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let bucket = match state.bucket_path.as_ref() {
        Some(p) => p.clone(),
        None => {
            return json_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "no_storage",
                "Storage not configured",
            )
        }
    };

    // Parse request body
    let body_bytes = match req.collect().await {
        Ok(b) => b.to_bytes(),
        Err(e) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_body",
                &format!("Failed to read body: {e}"),
            )
        }
    };

    let search_req: SearchRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_json",
                &format!("Invalid JSON: {e}"),
            )
        }
    };

    // Validate scope
    let valid_scopes = ["conversations", "documents", "items", "all"];
    if !valid_scopes.contains(&search_req.scope.as_str()) {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_scope",
            "scope must be 'conversations', 'documents', 'items', or 'all'",
        );
    }

    // Handle items scope
    if search_req.scope == "items" {
        return handle_items_search(state, search_req, auth).await;
    }

    // Handle documents scope
    if search_req.scope == "documents" {
        return handle_documents_search(state, bucket, search_req, auth).await;
    }

    // Handle federated (all) scope
    if search_req.scope == "all" {
        return handle_federated_search(state, bucket, search_req, auth).await;
    }

    // Validate search modes (at most one)
    let mode_count = [
        search_req.text.is_some(),
        search_req.regex.is_some(),
        search_req.vector.is_some(),
    ]
    .iter()
    .filter(|&&b| b)
    .count();

    if mode_count > 1 {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_search_mode",
            "specify at most one of: text, regex, vector",
        );
    }

    // Regex and vector modes not yet implemented
    if search_req.regex.is_some() {
        return json_error(
            StatusCode::NOT_IMPLEMENTED,
            "not_implemented",
            "regex search not yet implemented",
        );
    }

    if search_req.vector.is_some() {
        return json_error(
            StatusCode::NOT_IMPLEMENTED,
            "not_implemented",
            "vector search not yet implemented",
        );
    }

    // Extract parameters
    let limit = search_req.limit.unwrap_or(20).clamp(1, 100);
    let cursor = search_req.cursor.as_deref().and_then(decode_cursor);
    let requested_aggregations = search_req.aggregations.clone();

    // Get group_id from filters or auth
    let group_id = search_req
        .filters
        .as_ref()
        .and_then(|f| f.group_id.clone())
        .or_else(|| auth.and_then(|a| a.group_id));

    // Build filter strings for the storage layer
    let text_query = search_req.text.clone();

    // Convert tag IDs/names to space-separated string for AND filter
    // Treat empty array as "no filter" (return None, not empty string)
    let tags_filter = search_req
        .filters
        .as_ref()
        .and_then(|f| f.tags.as_ref())
        .filter(|tags| !tags.is_empty())
        .map(|tags| tags.join(" "));

    // Convert tag IDs/names to space-separated string for OR filter
    // Treat empty array as "no filter" (return None, not empty string)
    let tags_any_filter = search_req
        .filters
        .as_ref()
        .and_then(|f| f.tags_any.as_ref())
        .filter(|tags| !tags.is_empty())
        .map(|tags| tags.join(" "));

    // Extract additional filters
    let filters = &search_req.filters;
    let marker_filter = filters.as_ref().and_then(|f| f.marker.clone());
    // Treat empty array as "no filter" (return None, not empty string)
    let marker_any_filter = filters
        .as_ref()
        .and_then(|f| f.marker_any.as_ref())
        .filter(|m| !m.is_empty())
        .map(|m| m.join(" "));
    let model_filter = filters.as_ref().and_then(|f| f.model.clone());
    let created_after = filters.as_ref().and_then(|f| f.created_after);
    let created_before = filters.as_ref().and_then(|f| f.created_before);
    let updated_after = filters.as_ref().and_then(|f| f.updated_after);
    let updated_before = filters.as_ref().and_then(|f| f.updated_before);
    let has_tags = filters.as_ref().and_then(|f| f.has_tags);

    // Execute search
    let result = tokio::task::spawn_blocking(move || {
        let storage = StorageHandle::open(&bucket)?;

        let search_params = SearchParams {
            query: text_query.as_deref(),
            tags_filter: tags_filter.as_deref(),
            tags_filter_any: tags_any_filter.as_deref(),
            marker_filter: marker_filter.as_deref(),
            marker_filter_any: marker_any_filter.as_deref(),
            model_filter: model_filter.as_deref(),
            created_after_ms: created_after,
            created_before_ms: created_before,
            updated_after_ms: updated_after,
            updated_before_ms: updated_before,
            has_tags,
            source_doc_id: None,
        };

        let list_result = storage.list_sessions_paginated_ex(
            limit,
            cursor.as_ref(),
            group_id.as_deref(),
            &search_params,
        )?;

        // Compute aggregations if requested
        let aggregations = if let Some(ref agg_types) = requested_aggregations {
            if !agg_types.is_empty() {
                // For aggregations, we need all matching sessions (up to a limit)
                // Fetch without pagination cursor to get all results
                let all_sessions = storage.list_sessions_paginated_ex(
                    5000, // Max sessions to aggregate over
                    None, // No cursor - start from beginning
                    group_id.as_deref(),
                    &search_params,
                )?;
                Some(compute_aggregations(
                    &storage,
                    &all_sessions.sessions,
                    agg_types,
                ))
            } else {
                None
            }
        } else {
            None
        };

        // Resolve tags for each session
        let data: Vec<serde_json::Value> = list_result
            .sessions
            .iter()
            .map(|session| {
                let tags = resolve_tags_for_session(&storage, &session.session_id);
                session_to_conversation_json(session, Some(tags))
            })
            .collect();

        Ok::<_, talu::storage::StorageError>((
            data,
            list_result.has_more,
            list_result.next_cursor,
            aggregations,
        ))
    })
    .await;

    let (data, has_more, next_cursor, aggregations) = match result {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => return storage_error_response(e),
        Err(e) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                &format!("{e}"),
            )
        }
    };

    let response = SearchResponse {
        data,
        aggregations,
        cursor: next_cursor.map(|c| encode_cursor(&c)),
        has_more,
        total: None,
    };

    json_response(StatusCode::OK, &response)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

type BoxBody = http_body_util::combinators::BoxBody<Bytes, Infallible>;

fn json_response<T: Serialize>(status: StatusCode, data: &T) -> Response<BoxBody> {
    let body = serde_json::to_vec(data).unwrap_or_default();
    Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(Full::new(Bytes::from(body)).boxed())
        .unwrap()
}

fn json_error(status: StatusCode, code: &str, message: &str) -> Response<BoxBody> {
    let body = serde_json::json!({
        "error": {
            "code": code,
            "message": message
        }
    });
    json_response(status, &body)
}

fn storage_error_response(err: StorageError) -> Response<BoxBody> {
    match err {
        StorageError::SessionNotFound(msg) => json_error(StatusCode::NOT_FOUND, "not_found", &msg),
        StorageError::ItemNotFound(msg) => json_error(StatusCode::NOT_FOUND, "not_found", &msg),
        StorageError::TagNotFound(msg) => json_error(StatusCode::NOT_FOUND, "not_found", &msg),
        StorageError::InvalidArgument(msg) => {
            json_error(StatusCode::BAD_REQUEST, "invalid_argument", &msg)
        }
        _ => json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "storage_error",
            &err.to_string(),
        ),
    }
}

/// Compute aggregations over a set of sessions.
///
/// Supported aggregation types:
/// - "tags": Count sessions per tag
/// - "models": Count sessions per model
/// - "markers": Count sessions per marker
fn compute_aggregations(
    storage: &StorageHandle,
    sessions: &[SessionRecordFull],
    agg_types: &[String],
) -> serde_json::Value {
    let mut result = serde_json::Map::new();

    for agg_type in agg_types {
        match agg_type.as_str() {
            "tags" => {
                result.insert(
                    "tags".to_string(),
                    compute_tags_aggregation(storage, sessions),
                );
            }
            "models" => {
                result.insert("models".to_string(), compute_models_aggregation(sessions));
            }
            "markers" => {
                result.insert("markers".to_string(), compute_markers_aggregation(sessions));
            }
            _ => {
                // Unknown aggregation type - skip silently
            }
        }
    }

    serde_json::Value::Object(result)
}

/// Compute tag aggregation: count sessions per tag.
fn compute_tags_aggregation(
    storage: &StorageHandle,
    sessions: &[SessionRecordFull],
) -> serde_json::Value {
    // Count tag occurrences across all sessions
    let mut tag_counts: HashMap<String, u64> = HashMap::new();

    for session in sessions {
        // Get tags for this session
        if let Ok(tag_ids) = storage.get_conversation_tags(&session.session_id) {
            for tag_id in tag_ids {
                *tag_counts.entry(tag_id).or_insert(0) += 1;
            }
        }
    }

    // Resolve tag details and build response
    let mut tag_aggs: Vec<serde_json::Value> = tag_counts
        .into_iter()
        .filter_map(|(tag_id, count)| {
            // Try to get tag details
            if let Ok(tag) = storage.get_tag(&tag_id) {
                Some(serde_json::json!({
                    "id": tag.tag_id,
                    "name": tag.name,
                    "count": count
                }))
            } else {
                // Tag might have been deleted, still include with ID only
                Some(serde_json::json!({
                    "id": tag_id,
                    "name": tag_id,
                    "count": count
                }))
            }
        })
        .collect();

    // Sort by count descending
    tag_aggs.sort_by(|a, b| {
        let count_a = a["count"].as_u64().unwrap_or(0);
        let count_b = b["count"].as_u64().unwrap_or(0);
        count_b.cmp(&count_a)
    });

    // Limit to top 100
    tag_aggs.truncate(100);

    serde_json::Value::Array(tag_aggs)
}

/// Compute model aggregation: count sessions per model.
fn compute_models_aggregation(sessions: &[SessionRecordFull]) -> serde_json::Value {
    let mut model_counts: HashMap<String, u64> = HashMap::new();

    for session in sessions {
        if let Some(ref model) = session.model {
            if !model.is_empty() {
                *model_counts.entry(model.clone()).or_insert(0) += 1;
            }
        }
    }

    let mut model_aggs: Vec<serde_json::Value> = model_counts
        .into_iter()
        .map(|(value, count)| {
            serde_json::json!({
                "value": value,
                "count": count
            })
        })
        .collect();

    // Sort by count descending
    model_aggs.sort_by(|a, b| {
        let count_a = a["count"].as_u64().unwrap_or(0);
        let count_b = b["count"].as_u64().unwrap_or(0);
        count_b.cmp(&count_a)
    });

    // Limit to top 100
    model_aggs.truncate(100);

    serde_json::Value::Array(model_aggs)
}

/// Compute marker aggregation: count sessions per marker.
fn compute_markers_aggregation(sessions: &[SessionRecordFull]) -> serde_json::Value {
    let mut marker_counts: HashMap<String, u64> = HashMap::new();

    for session in sessions {
        if let Some(ref marker) = session.marker {
            if !marker.is_empty() {
                *marker_counts.entry(marker.clone()).or_insert(0) += 1;
            }
        }
    }

    let mut marker_aggs: Vec<serde_json::Value> = marker_counts
        .into_iter()
        .map(|(value, count)| {
            serde_json::json!({
                "value": value,
                "count": count
            })
        })
        .collect();

    // Sort by count descending
    marker_aggs.sort_by(|a, b| {
        let count_a = a["count"].as_u64().unwrap_or(0);
        let count_b = b["count"].as_u64().unwrap_or(0);
        count_b.cmp(&count_a)
    });

    // Limit to top 100
    marker_aggs.truncate(100);

    serde_json::Value::Array(marker_aggs)
}

// ---------------------------------------------------------------------------
// Item-level search
// ---------------------------------------------------------------------------

use talu::responses::{ItemType, ResponsesView};

/// Search result for an individual item/message.
#[derive(Debug, Serialize)]
struct ItemSearchResult {
    conversation_id: String,
    item_id: u64,
    role: String,
    snippet: String,
    conversation_title: Option<String>,
}

/// Handle item-level search (scope: "items").
///
/// Searches within message content across all conversations, returning
/// matching items with snippets.
async fn handle_items_search(
    state: Arc<AppState>,
    search_req: SearchRequest,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let bucket = match state.bucket_path.as_ref() {
        Some(p) => p.clone(),
        None => {
            return json_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "no_storage",
                "Storage not configured",
            )
        }
    };

    // Text search is required for items scope
    let query = match search_req.text.as_ref() {
        Some(q) if !q.is_empty() => q.clone(),
        _ => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "missing_text",
                "text field is required for items scope",
            )
        }
    };

    let limit = search_req.limit.unwrap_or(20).clamp(1, 100);
    let highlight = search_req.highlight.unwrap_or(false);

    // Get group_id from filters or auth
    let group_id = search_req
        .filters
        .as_ref()
        .and_then(|f| f.group_id.clone())
        .or_else(|| auth.and_then(|a| a.group_id));

    let result = tokio::task::spawn_blocking(move || {
        let storage = StorageHandle::open(&bucket)?;

        // Get all sessions (up to a reasonable limit for item search)
        let sessions_result = storage.list_sessions_paginated_ex(
            500, // Search across up to 500 conversations
            None,
            group_id.as_deref(),
            &SearchParams::default(),
        )?;

        let mut results: Vec<ItemSearchResult> = Vec::new();

        // Search through each session's items
        for session in &sessions_result.sessions {
            if results.len() >= limit {
                break;
            }

            // Load conversation items
            let conv = match storage.load_conversation(&session.session_id) {
                Ok(c) => c,
                Err(_) => continue, // Skip sessions that can't be loaded
            };

            let item_count = conv.items().len();

            for item_idx in 0..item_count {
                if results.len() >= limit {
                    break;
                }

                let item = match conv.items().nth(item_idx) {
                    Some(Ok(item)) => item,
                    _ => continue,
                };

                // Only search message items
                if item.item_type != ItemType::Message {
                    continue;
                }

                // Get message role
                let msg = match conv.get_message(item_idx) {
                    Ok(m) => m,
                    Err(_) => continue,
                };

                // Get text content
                let text = match conv.message_text(item_idx) {
                    Ok(t) => t,
                    Err(_) => continue,
                };

                // Case-insensitive search: find match position in original text
                // We can't use to_lowercase() because it may change byte length for some
                // Unicode characters (e.g., Turkish İ → i̇), making byte positions invalid.
                // Instead, we iterate through the original text to find a case-insensitive match.
                let match_result = find_case_insensitive(&text, &query);

                // Check if query matches
                if let Some((pos, match_len)) = match_result {
                    // Generate snippet
                    let snippet = if highlight {
                        generate_snippet_with_highlight(&text, pos, match_len)
                    } else {
                        generate_snippet(&text, pos)
                    };

                    let role_str = match msg.role {
                        talu::responses::MessageRole::User => "user",
                        talu::responses::MessageRole::Assistant => "assistant",
                        talu::responses::MessageRole::System => "system",
                        talu::responses::MessageRole::Developer => "developer",
                        _ => "unknown",
                    };

                    results.push(ItemSearchResult {
                        conversation_id: session.session_id.clone(),
                        item_id: item.id,
                        role: role_str.to_string(),
                        snippet,
                        conversation_title: session.title.clone(),
                    });
                }
            }
        }

        Ok::<_, StorageError>(results)
    })
    .await;

    let results = match result {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => return storage_error_response(e),
        Err(e) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                &format!("{e}"),
            )
        }
    };

    let has_more = results.len() >= limit;

    let response = SearchResponse {
        data: results.into_iter().map(|r| serde_json::json!(r)).collect(),
        aggregations: None,
        cursor: None, // Item search doesn't support cursor pagination yet
        has_more,
        total: None,
    };

    json_response(StatusCode::OK, &response)
}

/// Generate a snippet around the match position.
fn generate_snippet(text: &str, match_pos: usize) -> String {
    const CONTEXT_CHARS: usize = 50;

    let start = match_pos.saturating_sub(CONTEXT_CHARS);
    let end = (match_pos + CONTEXT_CHARS).min(text.len());

    // Find word boundaries
    let start = text[..start]
        .rfind(char::is_whitespace)
        .map(|p| p + 1)
        .unwrap_or(start);
    let end = text[end..]
        .find(char::is_whitespace)
        .map(|p| end + p)
        .unwrap_or(text.len());

    let mut snippet = String::new();
    if start > 0 {
        snippet.push_str("...");
    }
    snippet.push_str(&text[start..end]);
    if end < text.len() {
        snippet.push_str("...");
    }

    snippet
}

/// Find a case-insensitive match in text, returning (byte_position, match_byte_length).
///
/// This function properly handles Unicode by comparing characters, not bytes.
/// It returns byte positions that are valid for slicing the original text.
fn find_case_insensitive(text: &str, query: &str) -> Option<(usize, usize)> {
    // Early return for empty query
    if query.is_empty() {
        return None;
    }

    let query_lower: String = query.to_lowercase();
    let query_chars: Vec<char> = query_lower.chars().collect();

    // Iterate through text by character, tracking byte positions
    let mut text_chars = text.char_indices().peekable();

    while let Some((start_byte, _)) = text_chars.peek().copied() {
        // Try to match query starting at this position
        let mut query_idx = 0;
        let mut match_end_byte = start_byte;
        let mut chars_clone = text_chars.clone();

        while query_idx < query_chars.len() {
            if let Some((byte_pos, ch)) = chars_clone.next() {
                let ch_lower: String = ch.to_lowercase().collect();
                // Compare lowercased character
                let mut matches = true;
                for qch in ch_lower.chars() {
                    if query_idx >= query_chars.len() || qch != query_chars[query_idx] {
                        matches = false;
                        break;
                    }
                    query_idx += 1;
                }
                if !matches {
                    break;
                }
                match_end_byte = byte_pos + ch.len_utf8();
            } else {
                // End of text before query matched
                break;
            }
        }

        if query_idx == query_chars.len() {
            // Full match found
            return Some((start_byte, match_end_byte - start_byte));
        }

        // Move to next character
        text_chars.next();
    }

    None
}

/// Generate a snippet with **highlighted** match.
fn generate_snippet_with_highlight(text: &str, match_pos: usize, match_len: usize) -> String {
    const CONTEXT_CHARS: usize = 50;

    let start = match_pos.saturating_sub(CONTEXT_CHARS);
    let end = (match_pos + match_len + CONTEXT_CHARS).min(text.len());

    // Find word boundaries
    let start = text[..start]
        .rfind(char::is_whitespace)
        .map(|p| p + 1)
        .unwrap_or(start);
    let end = text[end..]
        .find(char::is_whitespace)
        .map(|p| end + p)
        .unwrap_or(text.len());

    let mut snippet = String::new();
    if start > 0 {
        snippet.push_str("...");
    }

    // Add text before match
    let before_match = &text[start..match_pos];
    snippet.push_str(before_match);

    // Add highlighted match
    snippet.push_str("**");
    snippet.push_str(&text[match_pos..match_pos + match_len]);
    snippet.push_str("**");

    // Add text after match
    let after_match = &text[match_pos + match_len..end];
    snippet.push_str(after_match);

    if end < text.len() {
        snippet.push_str("...");
    }

    snippet
}

// ---------------------------------------------------------------------------
// Document Search (Phase 13: Federated Search)
// ---------------------------------------------------------------------------

/// Handle document-level search (scope: "documents").
///
/// Searches within document content and metadata.
async fn handle_documents_search(
    _state: Arc<AppState>,
    bucket: std::path::PathBuf,
    search_req: SearchRequest,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    // Text search is required for documents scope
    let query = match search_req.text.as_ref() {
        Some(q) if !q.is_empty() => q.clone(),
        _ => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "missing_text",
                "text field is required for documents scope",
            )
        }
    };

    let limit = search_req.limit.unwrap_or(20).clamp(1, 100) as u32;

    // Get group_id from filters or auth
    let group_id = search_req
        .filters
        .as_ref()
        .and_then(|f| f.group_id.clone())
        .or_else(|| auth.and_then(|a| a.group_id));

    // Get type filter from filters
    let doc_type_filter = search_req.filters.as_ref().and_then(|f| f.model.clone()); // Repurpose model as doc_type for documents

    let result = tokio::task::spawn_blocking(move || {
        let _ = group_id; // group_id filtering not yet supported in documents
        let docs = DocumentsHandle::open(&bucket)?;
        let results = docs.search(&query, doc_type_filter.as_deref(), limit)?;

        // Convert to search response format
        let data: Vec<serde_json::Value> = results
            .into_iter()
            .map(|r| {
                serde_json::json!({
                    "id": r.doc_id,
                    "object": "document",
                    "type": r.doc_type,
                    "title": r.title,
                    "snippet": r.snippet,
                })
            })
            .collect();

        Ok::<_, talu::documents::DocumentError>(data)
    })
    .await;

    let data = match result {
        Ok(Ok(d)) => d,
        Ok(Err(e)) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "storage_error",
                &format!("{e:?}"),
            )
        }
        Err(e) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                &format!("{e}"),
            )
        }
    };

    let has_more = data.len() >= limit as usize;

    let response = SearchResponse {
        data,
        aggregations: None,
        cursor: None, // Document search doesn't support cursor pagination yet
        has_more,
        total: None,
    };

    json_response(StatusCode::OK, &response)
}

/// Handle federated search across conversations and documents (scope: "all").
///
/// Searches both storage backends and merges results.
async fn handle_federated_search(
    _state: Arc<AppState>,
    bucket: std::path::PathBuf,
    search_req: SearchRequest,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    // Text search is required for federated scope
    let query = match search_req.text.as_ref() {
        Some(q) if !q.is_empty() => q.clone(),
        _ => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "missing_text",
                "text field is required for federated (all) scope",
            )
        }
    };

    let limit = search_req.limit.unwrap_or(20).clamp(1, 100);
    let cursor = search_req.cursor.as_deref().and_then(decode_cursor);

    // Get group_id from filters or auth
    let group_id = search_req
        .filters
        .as_ref()
        .and_then(|f| f.group_id.clone())
        .or_else(|| auth.as_ref().and_then(|a| a.group_id.clone()));

    // Extract filters
    let filters = &search_req.filters;
    let tags_filter = filters
        .as_ref()
        .and_then(|f| f.tags.as_ref())
        .filter(|tags| !tags.is_empty())
        .map(|tags| tags.join(" "));
    let tags_any_filter = filters
        .as_ref()
        .and_then(|f| f.tags_any.as_ref())
        .filter(|tags| !tags.is_empty())
        .map(|tags| tags.join(" "));
    let marker_filter = filters.as_ref().and_then(|f| f.marker.clone());
    let marker_any_filter = filters
        .as_ref()
        .and_then(|f| f.marker_any.as_ref())
        .filter(|m| !m.is_empty())
        .map(|m| m.join(" "));
    let model_filter = filters.as_ref().and_then(|f| f.model.clone());
    let created_after = filters.as_ref().and_then(|f| f.created_after);
    let created_before = filters.as_ref().and_then(|f| f.created_before);
    let updated_after = filters.as_ref().and_then(|f| f.updated_after);
    let updated_before = filters.as_ref().and_then(|f| f.updated_before);
    let has_tags = filters.as_ref().and_then(|f| f.has_tags);

    let bucket_clone = bucket.clone();
    let group_id_clone = group_id.clone();
    let query_clone = query.clone();

    // Execute conversation search
    let conv_result = tokio::task::spawn_blocking(move || {
        let storage = StorageHandle::open(&bucket)?;

        let search_params = SearchParams {
            query: Some(&query),
            tags_filter: tags_filter.as_deref(),
            tags_filter_any: tags_any_filter.as_deref(),
            marker_filter: marker_filter.as_deref(),
            marker_filter_any: marker_any_filter.as_deref(),
            model_filter: model_filter.as_deref(),
            created_after_ms: created_after,
            created_before_ms: created_before,
            updated_after_ms: updated_after,
            updated_before_ms: updated_before,
            has_tags,
            source_doc_id: None,
        };

        let list_result = storage.list_sessions_paginated_ex(
            limit / 2 + 1, // Split limit between conversations and documents
            cursor.as_ref(),
            group_id.as_deref(),
            &search_params,
        )?;

        // Convert to search results
        let data: Vec<serde_json::Value> = list_result
            .sessions
            .iter()
            .map(|session| {
                let tags = resolve_tags_for_session(&storage, &session.session_id);
                session_to_conversation_json(session, Some(tags))
            })
            .collect();

        Ok::<_, StorageError>((data, list_result.has_more, list_result.next_cursor))
    })
    .await;

    // Execute document search
    let doc_limit = (limit / 2 + 1) as u32;
    let doc_result = tokio::task::spawn_blocking(move || {
        let _ = group_id_clone; // group_id filtering not yet supported in documents
        let docs = DocumentsHandle::open(&bucket_clone)?;
        let results = docs.search(&query_clone, None, doc_limit)?;

        let data: Vec<serde_json::Value> = results
            .into_iter()
            .map(|r| {
                serde_json::json!({
                    "id": r.doc_id,
                    "object": "document",
                    "type": r.doc_type,
                    "title": r.title,
                    "snippet": r.snippet,
                })
            })
            .collect();

        Ok::<_, talu::documents::DocumentError>(data)
    })
    .await;

    // Merge results
    let (conv_data, conv_has_more, conv_cursor) = match conv_result {
        Ok(Ok((data, has_more, cursor))) => (data, has_more, cursor),
        Ok(Err(e)) => return storage_error_response(e),
        Err(e) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                &format!("{e}"),
            )
        }
    };

    let doc_data = match doc_result {
        Ok(Ok(d)) => d,
        Ok(Err(_)) => Vec::new(), // Documents search failed, continue with conversations only
        Err(_) => Vec::new(),
    };

    // Interleave results (conversations first, then documents)
    let mut merged: Vec<serde_json::Value> = Vec::with_capacity(conv_data.len() + doc_data.len());
    merged.extend(conv_data);
    merged.extend(doc_data);

    // Truncate to limit
    let has_more = merged.len() > limit || conv_has_more;
    merged.truncate(limit);

    // Encode cursor for next page (conversations only for now)
    let next_cursor = conv_cursor.map(|c| encode_cursor(&c));

    let response = SearchResponse {
        data: merged,
        aggregations: None,
        cursor: next_cursor,
        has_more,
        total: None,
    };

    json_response(StatusCode::OK, &response)
}
