//! Tree-sitter code analysis endpoints.
//!
//! Stateless endpoints for syntax highlighting, AST parsing, pattern querying,
//! and call graph extraction. Session-based endpoints for incremental parsing.
//! All responses are JSON produced by the Zig core and forwarded without serde
//! round-trips.

use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::{Request, Response, StatusCode};
use std::ffi::CString;

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::server::auth_gateway::AuthContext;
use crate::server::state::{AppState, CodeSession};

type BoxBody = http_body_util::combinators::BoxBody<Bytes, std::convert::Infallible>;

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

#[derive(Deserialize, ToSchema)]
pub struct HighlightRequest {
    /// Source code to highlight.
    pub source: String,
    /// Language identifier (e.g. "python", "javascript").
    pub language: String,
    /// If true, return rich tokens with positions, node kinds, and text.
    #[serde(default)]
    pub rich: bool,
}

#[derive(Deserialize, ToSchema)]
pub struct ParseRequest {
    /// Source code to parse.
    pub source: String,
    /// Language identifier.
    pub language: String,
}

#[derive(Deserialize, ToSchema)]
pub struct QueryRequest {
    /// Source code to query.
    pub source: String,
    /// Language identifier.
    pub language: String,
    /// Tree-sitter S-expression query pattern.
    pub query: String,
}

#[derive(Deserialize, ToSchema)]
pub struct GraphRequest {
    /// Source code to analyze.
    pub source: String,
    /// Language identifier.
    pub language: String,
    #[serde(default)]
    pub file_path: String,
    #[serde(default)]
    pub project_root: String,
    /// Extraction mode: "callables" or "call_sites".
    pub mode: String,
    /// Required when mode is "call_sites".
    #[serde(default)]
    pub definer_fqn: String,
}

/// Response wrapper for the languages endpoint.
#[derive(Serialize, ToSchema)]
pub struct LanguagesResponse {
    /// Comma-separated list of supported language identifiers.
    pub languages: String,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

#[utoipa::path(post, path = "/v1/code/highlight", tag = "Code",
    request_body = HighlightRequest,
    responses((status = 200, description = "Highlight tokens JSON array")))]
/// POST /v1/code/highlight
pub async fn handle_highlight(
    _state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let body_bytes = match req.into_body().collect().await {
        Ok(body) => body.to_bytes(),
        Err(_) => return json_error(StatusCode::BAD_REQUEST, "invalid_request", "Invalid body"),
    };

    let request: HighlightRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                &format!("Invalid JSON: {e}"),
            )
        }
    };

    let result = if request.rich {
        talu::treesitter::highlight_rich(request.source.as_bytes(), &request.language)
    } else {
        talu::treesitter::highlight(request.source.as_bytes(), &request.language)
    };

    match result {
        Ok(json) => json_raw(StatusCode::OK, json),
        Err(e) => json_error(
            StatusCode::BAD_REQUEST,
            "highlight_failed",
            &e.to_string(),
        ),
    }
}

#[utoipa::path(post, path = "/v1/code/parse", tag = "Code",
    request_body = ParseRequest,
    responses((status = 200, description = "JSON AST representation")))]
/// POST /v1/code/parse
pub async fn handle_parse(
    _state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let body_bytes = match req.into_body().collect().await {
        Ok(body) => body.to_bytes(),
        Err(_) => return json_error(StatusCode::BAD_REQUEST, "invalid_request", "Invalid body"),
    };

    let request: ParseRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                &format!("Invalid JSON: {e}"),
            )
        }
    };

    match talu::treesitter::parse_to_json(request.source.as_bytes(), &request.language) {
        Ok(json) => json_raw(StatusCode::OK, json),
        Err(e) => json_error(StatusCode::BAD_REQUEST, "parse_failed", &e.to_string()),
    }
}

#[utoipa::path(post, path = "/v1/code/query", tag = "Code",
    request_body = QueryRequest,
    responses((status = 200, description = "JSON array of query matches")))]
/// POST /v1/code/query
pub async fn handle_query(
    _state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let body_bytes = match req.into_body().collect().await {
        Ok(body) => body.to_bytes(),
        Err(_) => return json_error(StatusCode::BAD_REQUEST, "invalid_request", "Invalid body"),
    };

    let request: QueryRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                &format!("Invalid JSON: {e}"),
            )
        }
    };

    match talu::treesitter::query(request.source.as_bytes(), &request.language, &request.query) {
        Ok(json) => json_raw(StatusCode::OK, json),
        Err(e) => json_error(StatusCode::BAD_REQUEST, "query_failed", &e.to_string()),
    }
}

#[utoipa::path(post, path = "/v1/code/graph", tag = "Code",
    request_body = GraphRequest,
    responses((status = 200, description = "Callable definitions or call sites JSON")))]
/// POST /v1/code/graph
pub async fn handle_graph(
    _state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let body_bytes = match req.into_body().collect().await {
        Ok(body) => body.to_bytes(),
        Err(_) => return json_error(StatusCode::BAD_REQUEST, "invalid_request", "Invalid body"),
    };

    let request: GraphRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                &format!("Invalid JSON: {e}"),
            )
        }
    };

    let result = match request.mode.as_str() {
        "callables" => talu::treesitter::extract_callables(
            request.source.as_bytes(),
            &request.language,
            &request.file_path,
            &request.project_root,
        ),
        "call_sites" => talu::treesitter::extract_call_sites(
            request.source.as_bytes(),
            &request.language,
            &request.definer_fqn,
            &request.file_path,
            &request.project_root,
        ),
        other => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_mode",
                &format!("Unknown mode: {other}. Use \"callables\" or \"call_sites\""),
            )
        }
    };

    match result {
        Ok(json) => json_raw(StatusCode::OK, json),
        Err(e) => json_error(StatusCode::BAD_REQUEST, "graph_failed", &e.to_string()),
    }
}

#[utoipa::path(get, path = "/v1/code/languages", tag = "Code",
    responses((status = 200, body = LanguagesResponse)))]
/// GET /v1/code/languages
pub async fn handle_languages(
    _state: Arc<AppState>,
    _req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    match talu::treesitter::languages() {
        Ok(langs) => {
            // Wrap in JSON object: {"languages": "python,javascript,..."}
            let json = format!("{{\"languages\":{}}}", serde_json::json!(langs));
            json_raw(StatusCode::OK, json)
        }
        Err(e) => json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            &e.to_string(),
        ),
    }
}

// ---------------------------------------------------------------------------
// Session request types
// ---------------------------------------------------------------------------

#[derive(Deserialize, ToSchema)]
pub struct SessionCreateRequest {
    /// Source code to parse.
    pub source: String,
    /// Language identifier.
    pub language: String,
    /// If true, return rich tokens.
    #[serde(default)]
    pub rich: bool,
}

#[derive(Deserialize, ToSchema)]
pub struct SessionUpdateRequest {
    /// Session identifier from create.
    pub session_id: String,
    /// Full source replacement (used when `edits` is empty).
    #[serde(default)]
    pub source: String,
    /// If true, return rich tokens.
    #[serde(default)]
    pub rich: bool,
    /// Delta edits (LSP-style). When non-empty, applied instead of full source.
    #[serde(default)]
    pub edits: Vec<SessionTextEdit>,
}

/// A single text edit delta for HTTP session updates.
///
/// All row/column coordinates are required — the server does not compute them
/// from the source buffer. Editors (Monaco, CodeMirror, VSCode) always know
/// these values at edit time.
#[derive(Deserialize, Default, ToSchema)]
pub struct SessionTextEdit {
    /// Start byte offset of the edit range.
    pub start_byte: u32,
    /// End byte offset of the old text being replaced.
    pub old_end_byte: u32,
    /// New text to insert at the edit range.
    #[serde(default)]
    pub new_text: String,
    /// 0-indexed row of the start position.
    pub start_row: u32,
    /// 0-indexed byte column of the start position.
    pub start_column: u32,
    /// 0-indexed row of the old end position.
    pub old_end_row: u32,
    /// 0-indexed byte column of the old end position.
    pub old_end_column: u32,
    /// 0-indexed row of the new end position.
    pub new_end_row: u32,
    /// 0-indexed byte column of the new end position.
    pub new_end_column: u32,
}

#[derive(Deserialize, ToSchema)]
pub struct SessionHighlightRequest {
    /// Session identifier.
    pub session_id: String,
    /// If true, return rich tokens.
    #[serde(default)]
    pub rich: bool,
}

// ---------------------------------------------------------------------------
// Session handlers
// ---------------------------------------------------------------------------

#[utoipa::path(post, path = "/v1/code/session/create", tag = "Code",
    request_body = SessionCreateRequest,
    responses((status = 200, description = "Session ID + initial highlight tokens")))]
/// POST /v1/code/session/create
///
/// Creates a parser, parses source, returns highlight tokens + session_id.
pub async fn handle_session_create(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let body_bytes = match req.into_body().collect().await {
        Ok(body) => body.to_bytes(),
        Err(_) => return json_error(StatusCode::BAD_REQUEST, "invalid_request", "Invalid body"),
    };

    let request: SessionCreateRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                &format!("Invalid JSON: {e}"),
            )
        }
    };

    let parser = match talu::treesitter::ParserHandle::new(&request.language) {
        Ok(p) => p,
        Err(e) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "parser_failed",
                &format!("Failed to create parser: {e}"),
            )
        }
    };

    let c_language = match CString::new(request.language.as_str()) {
        Ok(c) => c,
        Err(e) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_language",
                &format!("Invalid language string: {e}"),
            )
        }
    };

    let source_bytes = request.source.into_bytes();

    let tree = match parser.parse(&source_bytes, None) {
        Ok(t) => t,
        Err(e) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "parse_failed",
                &format!("Parse failed: {e}"),
            )
        }
    };

    let tokens_json = if request.rich {
        tree.highlight_rich_with_c_lang(&source_bytes, &c_language)
    } else {
        tree.highlight_with_c_lang(&source_bytes, &c_language)
    };

    let tokens_json = match tokens_json {
        Ok(j) => j,
        Err(e) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "highlight_failed",
                &format!("Highlight failed: {e}"),
            )
        }
    };

    let session_id = generate_session_id();

    let session = CodeSession {
        parser,
        tree,
        language: request.language,
        c_language,
        source: source_bytes,
        last_access: std::time::Instant::now(),
    };

    state.code_sessions.lock().await.insert(session_id.clone(), session);

    let response = format!(
        "{{\"session_id\":{},\"tokens\":{}}}",
        serde_json::json!(session_id),
        tokens_json,
    );
    json_raw(StatusCode::OK, response)
}

#[utoipa::path(post, path = "/v1/code/session/update", tag = "Code",
    request_body = SessionUpdateRequest,
    responses((status = 200, description = "Updated highlight tokens")))]
/// POST /v1/code/session/update
///
/// Accepts either full source replacement or delta edits. Applies `tree.edit()`
/// before re-parsing so tree-sitter can reuse unchanged subtrees.
pub async fn handle_session_update(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let body_bytes = match req.into_body().collect().await {
        Ok(body) => body.to_bytes(),
        Err(_) => return json_error(StatusCode::BAD_REQUEST, "invalid_request", "Invalid body"),
    };

    let request: SessionUpdateRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                &format!("Invalid JSON: {e}"),
            )
        }
    };

    let mut sessions = state.code_sessions.lock().await;
    let session = match sessions.get_mut(&request.session_id) {
        Some(s) => s,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "session_not_found",
                &format!("Session not found: {}", request.session_id),
            )
        }
    };

    if !request.edits.is_empty() {
        // Delta mode: apply edits to source buffer + tree, then re-parse.
        if let Err(msg) = apply_session_deltas(session, &request.edits) {
            return json_error(StatusCode::BAD_REQUEST, "edit_failed", &msg);
        }
    } else {
        // Full source replacement: apply a synthetic edit covering the whole buffer.
        let old_source = &session.source;
        let old_len = old_source.len() as u32;
        let new_source = request.source.as_bytes();
        let new_len = new_source.len() as u32;

        let (old_end_row, old_end_col) = byte_to_point(old_source, old_len);
        let (new_end_row, new_end_col) = byte_to_point(new_source, new_len);

        session.tree.edit(talu::treesitter::InputEdit {
            start_byte: 0,
            old_end_byte: old_len,
            new_end_byte: new_len,
            start_row: 0,
            start_column: 0,
            old_end_row,
            old_end_column: old_end_col,
            new_end_row,
            new_end_column: new_end_col,
        });

        session.source = new_source.to_vec();
    }

    let new_tree = match session.parser.parse(&session.source, Some(&session.tree)) {
        Ok(t) => t,
        Err(e) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "parse_failed",
                &format!("Incremental parse failed: {e}"),
            )
        }
    };

    let tokens_json = if request.rich {
        new_tree.highlight_rich_with_c_lang(&session.source, &session.c_language)
    } else {
        new_tree.highlight_with_c_lang(&session.source, &session.c_language)
    };

    let tokens_json = match tokens_json {
        Ok(j) => j,
        Err(e) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "highlight_failed",
                &format!("Highlight failed: {e}"),
            )
        }
    };

    session.tree = new_tree;
    session.last_access = std::time::Instant::now();

    let response = format!(
        "{{\"session_id\":{},\"tokens\":{}}}",
        serde_json::json!(request.session_id),
        tokens_json,
    );
    json_raw(StatusCode::OK, response)
}

#[utoipa::path(post, path = "/v1/code/session/highlight", tag = "Code",
    request_body = SessionHighlightRequest,
    responses((status = 200, description = "Highlight tokens from cached tree")))]
/// POST /v1/code/session/highlight
///
/// Highlights using the current tree (no re-parse).
pub async fn handle_session_highlight(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let body_bytes = match req.into_body().collect().await {
        Ok(body) => body.to_bytes(),
        Err(_) => return json_error(StatusCode::BAD_REQUEST, "invalid_request", "Invalid body"),
    };

    let request: SessionHighlightRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                &format!("Invalid JSON: {e}"),
            )
        }
    };

    let mut sessions = state.code_sessions.lock().await;
    let session = match sessions.get_mut(&request.session_id) {
        Some(s) => s,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "session_not_found",
                &format!("Session not found: {}", request.session_id),
            )
        }
    };

    let tokens_json = if request.rich {
        session.tree.highlight_rich_with_c_lang(&session.source, &session.c_language)
    } else {
        session.tree.highlight_with_c_lang(&session.source, &session.c_language)
    };

    session.last_access = std::time::Instant::now();

    match tokens_json {
        Ok(json) => {
            let response = format!(
                "{{\"session_id\":{},\"tokens\":{}}}",
                serde_json::json!(request.session_id),
                json,
            );
            json_raw(StatusCode::OK, response)
        }
        Err(e) => json_error(
            StatusCode::BAD_REQUEST,
            "highlight_failed",
            &e.to_string(),
        ),
    }
}

#[utoipa::path(delete, path = "/v1/code/session/{id}", tag = "Code",
    params(("id" = String, Path, description = "Session ID to delete")),
    responses((status = 204, description = "Session deleted")))]
/// DELETE /v1/code/session/:id
///
/// Destroys a session, freeing parser and tree resources.
pub async fn handle_session_delete(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let session_id = path
        .strip_prefix("/v1/code/session/")
        .or_else(|| path.strip_prefix("/code/session/"))
        .unwrap_or("");

    if session_id.is_empty() {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "Missing session_id in path",
        );
    }

    let removed = state.code_sessions.lock().await.remove(session_id);

    if removed.is_some() {
        Response::builder()
            .status(StatusCode::NO_CONTENT)
            .body(Full::new(Bytes::new()).boxed())
            .unwrap()
    } else {
        json_error(
            StatusCode::NOT_FOUND,
            "session_not_found",
            &format!("Session not found: {session_id}"),
        )
    }
}

/// Generate a random session ID (16 hex chars).
fn generate_session_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    // Mix timestamp with a simple hash for uniqueness within a single server process.
    let hash = ts.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    format!("{:016x}", hash)
}

// ---------------------------------------------------------------------------
// Delta edit helpers (shared between session handlers)
// ---------------------------------------------------------------------------

/// Apply delta edits to a session's source buffer and tree.
///
/// Edits are applied in reverse byte order so earlier edits don't shift
/// the byte offsets of later edits.
fn apply_session_deltas(session: &mut CodeSession, edits: &[SessionTextEdit]) -> Result<(), String> {
    let mut sorted: Vec<&SessionTextEdit> = edits.iter().collect();
    sorted.sort_by(|a, b| b.start_byte.cmp(&a.start_byte));

    for edit in &sorted {
        let start = edit.start_byte as usize;
        let old_end = edit.old_end_byte as usize;
        let new_bytes = edit.new_text.as_bytes();
        let new_end_byte = start as u32 + new_bytes.len() as u32;

        if start > session.source.len() || old_end > session.source.len() {
            return Err(format!(
                "Edit out of bounds: start_byte={}, old_end_byte={}, source_len={}",
                start,
                old_end,
                session.source.len()
            ));
        }

        // Splice the source buffer.
        session.source.splice(start..old_end, new_bytes.iter().copied());

        // Apply tree edit for correct incremental parsing.
        // All coordinates are client-provided — no server-side O(N) scan.
        session.tree.edit(talu::treesitter::InputEdit {
            start_byte: edit.start_byte,
            old_end_byte: edit.old_end_byte,
            new_end_byte,
            start_row: edit.start_row,
            start_column: edit.start_column,
            old_end_row: edit.old_end_row,
            old_end_column: edit.old_end_column,
            new_end_row: edit.new_end_row,
            new_end_column: edit.new_end_column,
        });
    }

    Ok(())
}

/// Compute (row, column) for a byte offset in a source buffer.
fn byte_to_point(source: &[u8], byte: u32) -> (u32, u32) {
    let byte = byte as usize;
    let slice = if byte <= source.len() {
        &source[..byte]
    } else {
        source
    };
    let mut row: u32 = 0;
    let mut last_newline: usize = 0;
    for (i, &ch) in slice.iter().enumerate() {
        if ch == b'\n' {
            row += 1;
            last_newline = i + 1;
        }
    }
    let col = (slice.len() - last_newline) as u32;
    (row, col)
}

// ---------------------------------------------------------------------------
// Response helpers
// ---------------------------------------------------------------------------

/// Serve a pre-serialized JSON string directly (zero-copy from Zig core).
fn json_raw(status: StatusCode, json: String) -> Response<BoxBody> {
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(json)).boxed())
        .unwrap()
}

fn json_error(status: StatusCode, code: &str, message: &str) -> Response<BoxBody> {
    let body = serde_json::json!({
        "error": {
            "code": code,
            "message": message
        }
    });
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body.to_string())).boxed())
        .unwrap()
}
