//! Integration tests for low-level `/v1/db/*` endpoints.

mod auth;
mod blob;
mod contracts;
mod error_shapes;
mod kv;
mod method_matrix;
mod openapi_docs;
mod ops;
mod routing;
mod side_effects;
mod sql;
mod table;
mod tenant_isolation;
mod vector;

use std::path::Path;

use crate::server::common::ServerConfig;

/// Create a ServerConfig with bucket set for DB tests.
pub fn db_config(bucket: &Path) -> ServerConfig {
    let mut config = ServerConfig::new();
    config.bucket = Some(bucket.to_path_buf());
    config
}
