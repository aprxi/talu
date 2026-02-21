use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::Mutex;

use talu::InferenceBackend;

use crate::server::tenant::TenantRegistry;

pub struct BackendState {
    pub backend: Option<InferenceBackend>,
    pub current_model: Option<String>,
}

/// Stored conversation state for `previous_response_id` lookups.
pub struct StoredResponse {
    /// Serialized conversation JSON (Open Responses format).
    pub responses_json: String,
    /// Tool definitions (if any).
    pub tools_json: Option<serde_json::Value>,
    /// Tool choice (if any).
    pub tool_choice_json: Option<serde_json::Value>,
    /// Session ID in TaluDB storage (for persistence across chained requests).
    pub session_id: Option<String>,
}

/// Per-plugin capability token entry.
/// Maps a bearer token to the plugin that owns it and its permissions.
pub struct PluginTokenEntry {
    pub plugin_id: String,
    /// Network domain permissions extracted from manifest (e.g., "api.example.com", "*.google.com").
    pub network_permissions: Vec<String>,
}

/// In-memory store for plugin capability tokens.
/// Cleared and regenerated on each `GET /v1/plugins` call.
pub type PluginTokenStore = HashMap<String, PluginTokenEntry>;

/// A code analysis session holding a parser and tree for incremental re-parsing.
///
/// Sessions enable sub-millisecond re-highlighting on edits by reusing the
/// tree-sitter parser and feeding the previous tree for incremental parsing.
pub struct CodeSession {
    pub parser: talu::treesitter::ParserHandle,
    pub tree: talu::treesitter::TreeHandle,
    pub language: String,
    /// Pre-allocated CString for the language identifier.
    /// Avoids per-keystroke CString allocation in highlight/query FFI calls.
    pub c_language: std::ffi::CString,
    /// Source code that produced the current tree. Kept for highlight-without-reparse.
    pub source: Vec<u8>,
    pub last_access: std::time::Instant,
}

pub struct AppState {
    pub backend: Arc<Mutex<BackendState>>,
    pub configured_model: Option<String>,
    /// In-memory response store for `previous_response_id` conversation chaining.
    pub response_store: Mutex<HashMap<String, StoredResponse>>,
    pub gateway_secret: Option<String>,
    pub tenant_registry: Option<TenantRegistry>,
    /// TaluDB storage bucket for `/v1/conversations` endpoints.
    pub bucket_path: Option<PathBuf>,
    /// Serve console UI from this directory instead of bundled assets.
    pub html_dir: Option<PathBuf>,
    /// Plugin capability tokens — maps bearer token → plugin_id + permissions.
    pub plugin_tokens: Mutex<PluginTokenStore>,
    /// Max allowed size (bytes) for `/v1/files` uploads.
    pub max_file_upload_bytes: u64,
    /// Max allowed size (bytes) for `/v1/file/inspect` and `/v1/file/transform` in-memory operations.
    pub max_file_inspect_bytes: u64,
    /// In-memory code session store for incremental parsing.
    pub code_sessions: Mutex<HashMap<String, CodeSession>>,
}
