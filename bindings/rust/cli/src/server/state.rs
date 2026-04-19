use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Weak};

use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use talu::InferenceBackend;

use crate::server::tenant::TenantRegistry;
use crate::server::tokenizer::TokenizerInstance;

/// Result of an in-flight model load, broadcast to waiters via `watch` channel.
#[derive(Clone, Debug)]
pub enum ModelLoadResult {
    /// Load still in progress.
    Pending,
    /// Load succeeded — backend installed.
    Ok,
    /// Load failed with this error message.
    Err(String),
}

pub struct BackendState {
    pub backend: Option<InferenceBackend>,
    pub current_model: Option<String>,
}

/// Stored conversation state for `previous_response_id` lookups.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoredResponse {
    /// Serialized conversation JSON (Open Responses format).
    pub responses_json: String,
    /// Tool definitions (if any).
    pub tools_json: Option<serde_json::Value>,
    /// Tool choice (if any).
    pub tool_choice_json: Option<serde_json::Value>,
    /// Session ID associated with this in-memory conversation chain.
    pub session_id: Option<String>,
    /// Tenant scope for secure `previous_response_id` chaining.
    /// `None` means gateway auth is disabled (single-tenant server mode).
    pub tenant_id: Option<String>,
}

pub struct AppState {
    pub backend: Arc<Mutex<BackendState>>,
    /// Batch scheduler for concurrent local GPU decode.
    /// Behind a Mutex so it can be replaced when the backend changes (model switch).
    pub batch_scheduler:
        std::sync::Mutex<Option<Arc<crate::server::batch_scheduler::SchedulerState>>>,
    pub configured_model: Option<String>,
    /// In-memory response store for `previous_response_id` conversation chaining.
    pub response_store: Mutex<HashMap<String, StoredResponse>>,
    pub gateway_secret: Option<String>,
    pub tenant_registry: Option<TenantRegistry>,
    /// Canonical workdir root for request-scoped integrations.
    pub workdir: Option<PathBuf>,
    /// Stateful tokenizer handles keyed by `tokenizer_id` for `/v1/tokenizer/*`.
    /// Map access is protected separately from per-instance operations to avoid
    /// global serialization across independent tokenizer instances.
    pub tokenizer_instances: Mutex<HashMap<String, Arc<Mutex<TokenizerInstance>>>>,
    /// Active generation stop flags. Set all on shutdown for immediate cancellation.
    pub active_stop_flags: std::sync::Mutex<Vec<Weak<AtomicBool>>>,
    /// Previous scheduler drain thread. Joined before spawning a new one
    /// on model switch to prevent unbounded drain thread accumulation.
    pub drain_thread: std::sync::Mutex<Option<std::thread::JoinHandle<()>>>,
    /// Singleflight guard for model loading. Maps model_id to an in-progress
    /// load's watch receiver. Waiters clone the receiver and await completion.
    pub model_load_inflight:
        std::sync::Mutex<HashMap<String, tokio::sync::watch::Receiver<ModelLoadResult>>>,
}
