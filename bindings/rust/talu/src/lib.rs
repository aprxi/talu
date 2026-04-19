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
pub mod convert;
pub mod error;
pub mod logging;
pub mod model;
pub mod repo;
pub mod responses;
pub mod router;
mod wrappers;
pub mod xray;

pub use batch::{BatchConfig, BatchEvent, BatchHandle, BatchResult, EventType};
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
pub use logging::{LogFormat, LogLevel};
pub use model::{
    EffectiveGenConfig, EffectiveGenConfigRequest, GenerationConfigInfo, ModelInfo, QuantMethod,
};
pub use repo::{
    CacheOrigin, CachedModel, DownloadOptions, DownloadProgress, HfSearchResult,
    ProgressAction as RepoProgressAction, SearchDirection, SearchSort,
};
pub use xray::{TraceRecord, TraceStats, XrayCaptureHandle};

pub type Result<T> = std::result::Result<T, Error>;
