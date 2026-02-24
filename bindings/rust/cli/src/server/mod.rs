use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use clap::Args;
use log::LevelFilter;
use talu::blobs::BlobsHandle;

pub mod auth_gateway;
pub mod code;
pub mod code_ws;
pub mod db;
pub mod file;
pub mod files;
pub mod handlers;
pub mod http;
pub mod listen;
mod logger;
pub mod openapi;
pub mod plugins;
pub mod proxy;
pub mod repo;
pub mod responses;
pub mod responses_types;
pub mod search;
pub mod sessions;
pub mod settings;
pub mod state;
pub mod tags;
pub mod tenant;

const DEFAULT_MAX_FILE_UPLOAD_BYTES: u64 = 100 * 1024 * 1024;
const DEFAULT_MAX_FILE_INSPECT_BYTES: u64 = 50 * 1024 * 1024;
const DEFAULT_STARTUP_BLOB_GC_MIN_AGE_SECONDS: u64 = 15 * 60;

#[derive(Args, Debug)]
pub struct ServerArgs {
    /// Model: local path or provider::model for remote backends
    #[arg(short, long, env = "MODEL_URI")]
    pub model: Option<String>,

    /// Address to bind [default: 127.0.0.1]
    #[arg(long, default_value = "127.0.0.1")]
    pub host: IpAddr,

    /// TCP port to bind
    #[arg(long, default_value_t = 8258)]
    pub port: u16,

    /// Unix domain socket path
    #[arg(long, default_value = "~/.talu/talu.sock")]
    pub socket: String,

    /// Shared secret for trusted gateway authentication
    #[arg(long)]
    pub gateway_secret: Option<String>,

    /// Path to tenants.json configuration for gateway authentication
    #[arg(long)]
    pub tenant_config: Option<PathBuf>,

    /// Storage profile name
    #[arg(long, env = "TALU_PROFILE", default_value = "default")]
    pub profile: String,

    /// Override bucket path (bypasses profile resolution)
    #[arg(long, env = "TALU_BUCKET")]
    pub bucket: Option<PathBuf>,

    /// Disable storage entirely
    #[arg(long)]
    pub no_bucket: bool,

    /// Serve console UI from this directory instead of bundled assets
    #[arg(long)]
    pub html_dir: Option<PathBuf>,

    /// Max request body size for `POST /v1/files` uploads (bytes).
    #[arg(long, env = "TALU_MAX_FILE_UPLOAD_BYTES", default_value_t = DEFAULT_MAX_FILE_UPLOAD_BYTES)]
    pub max_file_upload_bytes: u64,

    /// Max request body size for `POST /v1/file/inspect` and `POST /v1/file/transform` (bytes).
    #[arg(long, env = "TALU_MAX_FILE_INSPECT_BYTES", default_value_t = DEFAULT_MAX_FILE_INSPECT_BYTES)]
    pub max_file_inspect_bytes: u64,
}

pub fn run_server(args: ServerArgs, verbose: u8, log_filter: Option<&str>) -> Result<()> {
    let level = match verbose {
        0 | 1 => LevelFilter::Info,
        2 => LevelFilter::Debug,
        _ => LevelFilter::Trace,
    };
    logger::init(level, log_filter);

    // Pass filter to Zig core logging as well.
    if let Some(filter) = log_filter {
        talu::logging::set_log_filter(filter);
    }
    if args.max_file_upload_bytes == 0 {
        bail!("--max-file-upload-bytes must be greater than zero");
    }

    let model = args.model.clone();
    let profile = args.profile.clone();
    let socket_path = expand_socket_path(&args.socket);

    let backend_state = if let Some(ref model_id) = model {
        let backend = crate::provider::create_backend_for_model(model_id)?;
        state::BackendState {
            backend: Some(backend),
            current_model: Some(model_id.clone()),
        }
    } else {
        state::BackendState {
            backend: None,
            current_model: None,
        }
    };

    let (gateway_secret, tenant_registry) = match (args.gateway_secret, args.tenant_config) {
        (Some(secret), Some(path)) => {
            let registry = tenant::TenantRegistry::load(&path)
                .with_context(|| format!("Failed to load tenant config {}", path.display()))?;
            (Some(secret), Some(registry))
        }
        (Some(_), None) => {
            bail!("--gateway-secret requires --tenant-config");
        }
        (None, Some(_)) => {
            bail!("--tenant-config requires --gateway-secret");
        }
        (None, None) => (None, None),
    };

    let bucket_path = crate::config::resolve_bucket(args.no_bucket, args.bucket, &args.profile)?;

    let state = state::AppState {
        backend: Arc::new(tokio::sync::Mutex::new(backend_state)),
        configured_model: model,
        response_store: tokio::sync::Mutex::new(std::collections::HashMap::new()),
        gateway_secret,
        tenant_registry,
        bucket_path,
        html_dir: args.html_dir,
        plugin_tokens: tokio::sync::Mutex::new(std::collections::HashMap::new()),
        max_file_upload_bytes: args.max_file_upload_bytes,
        max_file_inspect_bytes: args.max_file_inspect_bytes,
        code_sessions: tokio::sync::Mutex::new(std::collections::HashMap::new()),
        code_session_ttl: listen::CODE_SESSION_TTL,
    };

    let addr = SocketAddr::new(args.host, args.port);
    log::info!(target: "server::init", "talu server starting");
    if let Some(ref bucket) = state.bucket_path {
        let bucket_for_gc = bucket.clone();
        std::thread::spawn(move || {
            match run_startup_blob_gc_once(&bucket_for_gc, DEFAULT_STARTUP_BLOB_GC_MIN_AGE_SECONDS)
            {
                Ok(stats) => {
                    log::info!(
                        target: "server::init",
                        "startup blob gc: referenced={}, total={}, deleted={}, reclaimed_bytes={}",
                        stats.referenced_blob_count,
                        stats.total_blob_files,
                        stats.deleted_blob_files,
                        stats.reclaimed_bytes,
                    );
                }
                Err(err) => {
                    log::warn!(target: "server::init", "startup blob gc skipped: {err}");
                }
            }
        });

        log::info!(target: "server::init", "profile: {}", profile);
        log::info!(target: "server::init", "bucket: {}", bucket.display());
    } else {
        log::info!(target: "server::init", "storage disabled (--no-bucket)");
    }
    log::info!(target: "server::init", "console: http://{}/", addr);
    log::info!(target: "server::init", "listening on http://{}", addr);
    log::info!(target: "server::init", "listening on unix://{}", socket_path.display());
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(listen::serve(state, addr, socket_path))?;
    Ok(())
}

pub fn expand_socket_path(path: &str) -> PathBuf {
    if let Some(stripped) = path.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(stripped);
        }
    }
    PathBuf::from(path)
}

fn run_startup_blob_gc_once(
    bucket_path: &Path,
    min_blob_age_seconds: u64,
) -> Result<talu::BlobGcStats> {
    let blobs = BlobsHandle::open(bucket_path).context("open blob storage for startup gc")?;
    blobs
        .gc_with_min_age(min_blob_age_seconds)
        .context("run startup blob gc")
}

#[cfg(test)]
mod tests {
    use super::run_startup_blob_gc_once;
    use talu::blobs::BlobsHandle;

    #[test]
    fn run_startup_blob_gc_once_deletes_unreferenced_blobs_with_zero_grace() {
        let tmp = tempfile::TempDir::new().expect("temp dir");
        let blobs = BlobsHandle::open(tmp.path()).expect("open blobs");

        let orphan_ref = blobs.put(b"startup-gc-orphan").expect("write orphan");
        assert!(blobs.contains(&orphan_ref).expect("contains before gc"));

        let stats = run_startup_blob_gc_once(tmp.path(), 0).expect("run startup gc");
        assert_eq!(stats.total_blob_files, 1);
        assert_eq!(stats.deleted_blob_files, 1);

        let blobs = BlobsHandle::open(tmp.path()).expect("re-open blobs");
        assert!(!blobs.contains(&orphan_ref).expect("contains after gc"));
    }
}
