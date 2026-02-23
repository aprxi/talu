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

pub mod blobs;
pub mod convert;
pub mod documents;
pub mod error;
pub mod file;
pub mod kv;
pub mod logging;
pub mod model;
pub mod plugins;
pub mod policy;
pub mod provider;
pub mod repo;
pub mod repo_meta;
pub mod responses;
pub mod router;
pub mod sql;
pub mod storage;
pub mod table;
pub mod treesitter;
pub mod vector;
mod wrappers;
pub mod xray;

pub use error::Error;
pub use wrappers::{
    CanonicalSpec, ChatHandle, EncodeResult, FinishReason, GenerateResult, InferenceBackend,
    LoadProgress, LoadProgressCallback, RemoteModelInfo, TextPtr, TokenizerHandle, ToolCall,
};

// Re-export commonly used types for CLI convenience
pub use blobs::{BlobError, BlobGcStats, BlobReadStream, BlobWriteStream, BlobsHandle};
pub use convert::{
    ConvertOptions, ConvertProgress, ConvertResult, ProgressAction as ConvertProgressAction, Scheme,
};
pub use documents::{
    ChangeAction, ChangeRecord, CompactionStats, DocumentError, DocumentRecord, DocumentSummary,
    DocumentsHandle, SearchResult as DocSearchResult,
};
pub use file::{
    FileInfo, FileKind, FitMode, ImageFormat, ImageInfo, Limits as FileLimits, OutputFormat,
    ResizeFilter, ResizeOptions, TransformOptions as FileTransformOptions, TransformResult,
};
pub use kv::{KvEntry, KvError, KvHandle, KvValue};
pub use logging::{LogFormat, LogLevel};
pub use model::{GenerationConfigInfo, ModelInfo, QuantMethod};
pub use repo::{
    CacheOrigin, CachedModel, DownloadOptions, DownloadProgress, HfSearchResult,
    ProgressAction as RepoProgressAction, SearchDirection, SearchSort,
};
pub use repo_meta::{RepoMetaError, RepoMetaStore, RepoPinEntry};
pub use sql::{SqlEngine, SqlError};
pub use storage::{SessionRecord, StorageError, StorageHandle};
pub use table::{TableError, TableHandle, TableRecord, TableSearchResult, TableSummary};
pub use vector::{SearchBatchResult, VectorError, VectorStore};
pub use xray::{TraceRecord, TraceStats, XrayCaptureHandle};

/// Write durability mode for TaluDB storage operations.
///
/// Controls whether writes are fsynced to disk immediately (full durability)
/// or buffered in the OS page cache (async durability).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Durability {
    /// Fsync after every WAL write. Survives OS crash and power loss.
    Full = 0,
    /// Skip fsync, rely on OS page cache. Survives app crashes only.
    AsyncOs = 1,
}

pub type Result<T> = std::result::Result<T, Error>;
