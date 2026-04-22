//! Safe, idiomatic Rust SDK for talu LLM inference.
//!
//! This crate provides safe wrappers around the low-level FFI bindings
//! in `talu-sys`, offering RAII-based resource management and Result-based
//! error handling.
//!
//! # Example
//!
//! ```no_run
//! use talu::{ChatHandle, InferenceBackend};
//!
//! let chat = ChatHandle::new(Some("You are a helpful assistant."))?;
//! let backend = InferenceBackend::new("path/to/model")?;
//! // Use chat and backend for generation...
//! # Ok::<(), talu::Error>(())
//! ```
//!
//! # Responses API (Item-Based Architecture)
//!
//! For advanced conversation inspection using the Open Responses API:
//!
//! ```no_run
//! use talu::responses::{ResponsesHandle, ResponsesView, MessageRole, ItemType};
//!
//! let mut conv = ResponsesHandle::new()?;
//! conv.append_message(MessageRole::User, "Hello!")?;
//!
//! // Iterate over items
//! for item in conv.items() {
//!     let item = item?;
//!     match item.item_type {
//!         ItemType::Message => {
//!             let msg = conv.get_message(0)?;
//!             println!("Role: {:?}", msg.role);
//!         }
//!         _ => {}
//!     }
//! }
//! # Ok::<(), talu::Error>(())
//! ```
//!
//! # Ownership Model
//!
//! The Responses API provides two handle types:
//! - [`responses::ResponsesHandle`]: Owned conversation (freed on drop)
//! - [`responses::ResponsesRef`]: Borrowed reference (does NOT free on drop)
//!
//! Both implement [`responses::ResponsesView`] for read access to items.

pub mod batch;
pub mod collab;
pub mod convert;
pub mod error;
pub mod fs;
pub mod logging;
pub mod model;
pub mod policy;
pub mod process;
pub mod repo;
pub mod responses;
pub mod router;
pub mod shell;
mod wrappers;
pub mod xray;

pub use batch::{BatchConfig, BatchEvent, BatchHandle, BatchResult, EventType};
pub use collab::{
    BinaryValue as CollabBinaryValue, CollabError, CollabHandle,
    HistoryEntry as CollabHistoryEntry, OpSubmitResult as CollabOpSubmitResult,
    ParticipantKind as CollabParticipantKind, ResourceSummary as CollabResourceSummary,
    SessionInfo as CollabSessionInfo, WatchBatch as CollabWatchBatch,
    WatchDurability as CollabWatchDurability, WatchEvent as CollabWatchEvent,
    WatchEventType as CollabWatchEventType, WatchWaitResult as CollabWatchWaitResult,
};
pub use error::Error;
pub use wrappers::{
    CanonicalSpec, ChatHandle, EncodeResult, FinishReason, GenerateResult, InferenceBackend,
    LoadProgress, LoadProgressCallback, TextPtr, TokenizerHandle, ToolCall,
};

// Re-export commonly used types for CLI convenience
pub use convert::{
    ConvertOptions, ConvertProfile, ConvertProgress, ConvertResult,
    ProgressAction as ConvertProgressAction, Scheme,
};
pub use fs::{FsEditResult, FsError, FsHandle, FsReadResult, FsStat, FsWriteResult};
pub use logging::{LogFormat, LogLevel};
pub use model::{
    EffectiveGenConfig, EffectiveGenConfigRequest, GenerationConfigInfo, ModelInfo, QuantMethod,
};
pub use policy::{Policy, ProcessDenyReason, ProcessPolicyDecision, StrictEmulationDecisions};
pub use process::{ProcessError, ProcessSession};
pub use repo::{
    CacheOrigin, CachedModel, DownloadOptions, DownloadProgress, HfSearchResult,
    ProgressAction as RepoProgressAction, SearchDirection, SearchSort,
};
pub use shell::{
    default_policy_json, exec, exec_streaming, exec_streaming_with_policy,
    exec_streaming_with_policy_runtime, exec_with_policy, exec_with_policy_runtime,
    normalize_command, validate_strict_runtime, validate_strict_runtime_ext, AgentRuntimeMode,
    CapabilityReport, ExecOutput, SafetyCheck, SandboxBackend, ShellError, ShellSession,
};
pub use xray::{TraceRecord, TraceStats, XrayCaptureHandle};

pub type Result<T> = std::result::Result<T, Error>;
