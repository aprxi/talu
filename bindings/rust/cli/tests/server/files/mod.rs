//! Integration tests for `/v1/files` endpoint.

mod batch;
mod list;
mod read;
mod resolve;
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

pub fn files_config_with_upload_limit(bucket: &Path, max_bytes: u64) -> ServerConfig {
    let mut config = files_config(bucket);
    config.env_vars.push((
        "TALU_MAX_FILE_UPLOAD_BYTES".to_string(),
        max_bytes.to_string(),
    ));
    config
}
