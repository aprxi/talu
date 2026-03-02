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
    /// Tenant scope for secure `previous_response_id` chaining.
    /// `None` means gateway auth is disabled (single-tenant server mode).
    pub tenant_id: Option<String>,
}

/// Per-plugin capability token entry.
/// Maps a bearer token to the plugin that owns it and its permissions.
#[derive(Clone)]
pub struct PluginTokenEntry {
    pub plugin_id: String,
    /// Network domain permissions extracted from manifest (e.g., "api.example.com", "*.google.com").
    pub network_permissions: Vec<String>,
    /// Whether this token may access `/v1/agent/fs/*`.
    pub allow_filesystem: bool,
    /// Whether this token may access `/v1/agent/exec`, `/v1/agent/shells/*`, and `/v1/agent/processes/*`.
    pub allow_exec: bool,
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

/// An interactive shell session backed by core `talu_shell_*` APIs.
pub struct ShellSession {
    pub shell: Arc<Mutex<talu::shell::ShellSession>>,
    pub owner_key: String,
    pub cwd: Option<String>,
    pub cols: u16,
    pub rows: u16,
    pub created_at: std::time::Instant,
    pub last_access: std::time::Instant,
    /// Number of currently attached WebSocket clients.
    pub attached_clients: usize,
}

/// A long-lived non-PTY process session backed by core `talu_process_*` APIs.
pub struct ProcessSession {
    pub process: Arc<Mutex<talu::process::ProcessSession>>,
    pub owner_key: String,
    pub command: String,
    pub cwd: Option<String>,
    pub created_at: std::time::Instant,
    pub last_access: std::time::Instant,
    /// Number of currently attached SSE stream clients.
    pub attached_streams: usize,
}

pub struct AppState {
    pub backend: Arc<Mutex<BackendState>>,
    pub configured_model: Option<String>,
    /// In-memory response store for `previous_response_id` conversation chaining.
    pub response_store: Mutex<HashMap<String, StoredResponse>>,
    pub gateway_secret: Option<String>,
    pub tenant_registry: Option<TenantRegistry>,
    /// TaluDB storage bucket for `/v1/chat/sessions` endpoints.
    pub bucket_path: Option<PathBuf>,
    /// Canonical workspace root for `/v1/agent/fs/*` endpoints.
    pub workspace_dir: PathBuf,
    /// Optional JSON policy applied to `/v1/agent/*` runtime operations.
    pub agent_policy_json: Option<String>,
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
    /// Max idle time before a code session is evicted by the reaper.
    pub code_session_ttl: std::time::Duration,
    /// In-memory shell session store for `/v1/agent/shells/*`.
    pub shell_sessions: Mutex<HashMap<String, ShellSession>>,
    /// Max idle time before a detached shell session is evicted by the reaper.
    pub shell_session_ttl: std::time::Duration,
    /// In-memory process session store for `/v1/agent/processes/*`.
    pub process_sessions: Mutex<HashMap<String, ProcessSession>>,
    /// Max idle time before a detached process session is evicted by the reaper.
    pub process_session_ttl: std::time::Duration,
}
