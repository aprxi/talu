use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use clap::Args;
use simplelog::{Config, LevelFilter, SimpleLogger};

pub mod auth_gateway;
pub mod conversations;
pub mod documents;
pub mod files;
pub mod generated;
pub mod handlers;
pub mod http;
pub mod listen;
pub mod plugins;
pub mod proxy;
pub mod search;
pub mod settings;
pub mod state;
pub mod tags;
pub mod tenant;

const DEFAULT_MAX_FILE_UPLOAD_BYTES: u64 = 100 * 1024 * 1024;

#[derive(Args, Debug)]
pub struct ServerArgs {
    /// Model: local path or provider::model for remote backends
    #[arg(short, long, env = "MODEL_URI")]
    pub model: Option<String>,

    /// TCP port to bind (127.0.0.1)
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
}

pub fn run_server(args: ServerArgs) -> Result<()> {
    let _ = SimpleLogger::init(LevelFilter::Info, Config::default());
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
    };

    let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), args.port);
    log::info!("talu server starting");
    if let Some(ref bucket) = state.bucket_path {
        log::info!("profile: {}", profile);
        log::info!("bucket: {}", bucket.display());
    } else {
        log::info!("storage disabled (--no-bucket)");
    }
    log::info!("console: http://{}/", addr);
    log::info!("listening on http://{}", addr);
    log::info!("listening on unix://{}", socket_path.display());
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
