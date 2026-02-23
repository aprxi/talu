//! Integration tests for `/v1/db/tables/documents` endpoints.
//!
//! Tests the Documents API endpoints:
//! - GET /v1/db/tables/documents (list)
//! - GET /v1/db/tables/documents/:id (get)
//! - POST /v1/db/tables/documents (create)
//! - PATCH /v1/db/tables/documents/:id (update)
//! - DELETE /v1/db/tables/documents/:id (delete)
//! - POST /v1/db/tables/documents/search (search)
//! - GET /v1/db/tables/documents/:id/tags (get tags)
//! - POST /v1/db/tables/documents/:id/tags (add tags)
//! - DELETE /v1/db/tables/documents/:id/tags (remove tags)

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
