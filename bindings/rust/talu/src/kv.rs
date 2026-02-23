//! KV-plane aliases over repository metadata storage.
//!
//! This module aligns SDK naming with the unified DB API. It currently maps
//! KV operations to the repository metadata implementation.

pub use crate::repo_meta::{
    RepoMetaError as KvError, RepoMetaStore as KvHandle, RepoPinEntry as KvEntry,
};
