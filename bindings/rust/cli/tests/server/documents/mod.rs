//! Integration tests for `/v1/documents` endpoints.
//!
//! Tests the Documents API endpoints:
//! - GET /v1/documents (list)
//! - GET /v1/documents/:id (get)
//! - POST /v1/documents (create)
//! - PATCH /v1/documents/:id (update)
//! - DELETE /v1/documents/:id (delete)
//! - POST /v1/documents/search (search)
//! - GET /v1/documents/:id/tags (get tags)
//! - POST /v1/documents/:id/tags (add tags)
//! - DELETE /v1/documents/:id/tags (remove tags)

mod api;
mod crud;
mod search;
mod tags;

use crate::server::common::ServerConfig;
use std::path::Path;

/// Create a ServerConfig with bucket set for document tests.
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
