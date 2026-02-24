//! Integration tests for `/v1/db/tables/documents` endpoints.
//!
//! Tests the table plane using the `documents` table.

mod api;
mod crud;
mod search;
mod tags;

use crate::server::common::ServerConfig;
use std::path::Path;

/// Create a ServerConfig with bucket set for table tests.
pub fn documents_config(bucket: &Path) -> ServerConfig {
    let mut config = ServerConfig::new();
    config.bucket = Some(bucket.to_path_buf());
    config
}

/// Create a ServerConfig with --no-bucket (storage disabled).
pub fn no_bucket_config() -> ServerConfig {
    let mut config = ServerConfig::new();
    config.no_bucket = true;
    config
}

