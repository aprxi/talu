//! Integration tests for `/v1/tags` endpoints.

mod crud;
mod session_tags;

use std::path::Path;

use crate::server::common::ServerConfig;

/// Create a ServerConfig with bucket set.
pub fn tags_config(bucket: &Path) -> ServerConfig {
    let mut config = ServerConfig::new();
    config.bucket = Some(bucket.to_path_buf());
    config
}
