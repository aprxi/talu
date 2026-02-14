//! Integration tests for `/v1/files` endpoint.

mod upload;

use crate::server::common::ServerConfig;
use std::path::Path;

pub fn files_config(bucket: &Path) -> ServerConfig {
    let mut config = ServerConfig::new();
    config.bucket = Some(bucket.to_path_buf());
    config
}

pub fn no_bucket_config() -> ServerConfig {
    let mut config = ServerConfig::new();
    config.no_bucket = true;
    config
}
