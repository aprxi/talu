use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use clap::{Args, ValueEnum};
use log::LevelFilter;
use talu::blobs::BlobsHandle;

pub mod agent;
pub mod auth_gateway;
pub mod code;
pub mod code_ws;
pub mod collab;
pub mod db;
pub mod events;
pub mod file;
pub mod files;
pub mod handlers;
pub mod http;
pub mod listen;
mod logger;
pub mod openapi;
pub mod plugins;
pub mod projects;
pub mod providers;
pub mod proxy;
pub mod pubsub;
pub mod repo;
pub mod responses;
mod responses_openapi;
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum AgentRuntimeMode {
    Host,
    Strict,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum SandboxBackendArg {
    Auto,
    LinuxLocal,
    Oci,
    AppleContainer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SandboxBackend {
    LinuxLocal,
    Oci,
    AppleContainer,
}

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

    /// Path to runtime policy JSON file applied to `/v1/agent/*`.
    ///
    /// This is equivalent to setting `TALU_AGENT_POLICY_JSON` with the file
    /// content, but is safer and easier to manage operationally.
    #[arg(long, env = "TALU_POLICY_FILE")]
    pub policy_file: Option<PathBuf>,

    /// Canonical workdir root for `/v1/agent/fs/*` endpoint sandboxing.
    ///
    /// Relative request paths are resolved against this directory.
    #[arg(long, env = "TALU_WORKDIR")]
    pub workdir: Option<PathBuf>,

    /// Runtime mode for `/v1/agent/exec`, `/v1/agent/shells/*`, and
    /// `/v1/agent/processes/*`.
    ///
    /// `strict` enables kernel/runtime sandbox enforcement guarantees.
    /// `host` keeps passthrough behavior without firewall guarantees.
    #[arg(long, env = "TALU_AGENT_RUNTIME_MODE", value_enum, default_value_t = AgentRuntimeMode::Host)]
    pub agent_runtime_mode: AgentRuntimeMode,

    /// Sandbox backend selection for strict runtime mode.
    #[arg(long, env = "TALU_SANDBOX_BACKEND", value_enum, default_value_t = SandboxBackendArg::Auto)]
    pub sandbox_backend: SandboxBackendArg,

    /// Run conformance probes at startup to verify sandbox enforcement end-to-end.
    ///
    /// Only meaningful when `--agent-runtime-mode strict` is set.
    #[arg(long, env = "TALU_STRICT_PROBES", default_value_t = false)]
    pub strict_probes: bool,
}

pub fn run_server(args: ServerArgs, verbose: u8, log_filter: Option<&str>) -> Result<()> {
    let level = match verbose {
        0 | 1 => LevelFilter::Info,
        2 => LevelFilter::Debug,
        _ => LevelFilter::Trace,
    };
    logger::init(level, log_filter);
    events::maybe_reset_global_for_tests();
    events::install_core_log_bridge();

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
    let sandbox_backend = resolve_sandbox_backend(args.agent_runtime_mode, args.sandbox_backend)?;

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

    if let Some(ref bucket) = bucket_path {
        crate::bucket_settings::migrate_layout_if_needed(bucket)?;
    }

    let backend_state = if let Some(ref model_id) = model {
        let kv_root = bucket_path
            .as_ref()
            .map(|b| b.join("kv").to_string_lossy().into_owned());
        let backend = crate::provider::create_backend_for_model(model_id, kv_root.as_deref())?;
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

    let workdir = args
        .workdir
        .or_else(|| std::env::var_os("TALU_WORKDIR").map(PathBuf::from))
        .map(|path| {
            path.canonicalize()
                .with_context(|| format!("canonicalize agent fs workdir path: {}", path.display()))
        })
        .transpose()?;
    let env_agent_policy_json = std::env::var("TALU_AGENT_POLICY_JSON").ok();
    if args.policy_file.is_some() && env_agent_policy_json.is_some() {
        bail!("--policy-file conflicts with TALU_AGENT_POLICY_JSON; use exactly one source");
    }
    let agent_policy_json = if let Some(path) = args.policy_file.as_ref() {
        Some(
            std::fs::read_to_string(path)
                .with_context(|| format!("read agent policy file {}", path.display()))?,
        )
    } else {
        env_agent_policy_json
    };
    let agent_policy = if let Some(policy_json) = agent_policy_json.as_deref() {
        // Parse once at startup so policy mistakes fail fast and endpoints do
        // not re-parse JSON on each request.
        Some(Arc::new(
            talu::policy::Policy::from_json(policy_json)
                .context("parse agent runtime policy JSON")?,
        ))
    } else {
        None
    };
    if args.agent_runtime_mode == AgentRuntimeMode::Strict && agent_policy.is_none() {
        bail!("strict agent runtime mode requires a policy via --policy-file or TALU_AGENT_POLICY_JSON");
    }
    if args.agent_runtime_mode == AgentRuntimeMode::Strict {
        let strict_policy = agent_policy
            .as_deref()
            .expect("strict runtime mode requires policy by prior check");
        talu::shell::validate_strict_runtime(
            Some(strict_policy),
            workdir
                .as_ref()
                .map(|path| path.to_string_lossy())
                .as_deref(),
            sandbox_backend_for_talu(sandbox_backend),
        )
        .context("validate strict runtime policy and sandbox support")?;

        // Capability detection + optional conformance probes.
        let report = talu::shell::validate_strict_runtime_ext(
            sandbox_backend_for_talu(sandbox_backend),
            true, // strict_required
            args.strict_probes,
            workdir
                .as_ref()
                .map(|path| path.to_string_lossy())
                .as_deref(),
        )
        .context("validate strict runtime capabilities")?;
        log::info!(
            target: "server::init",
            "sandbox capabilities: kernel={}.{} landlock_abi={} user_ns={} seccomp={} cgroupv2={}/{}",
            report.kernel_version_major,
            report.kernel_version_minor,
            report.landlock_abi_version,
            report.user_ns_available,
            report.seccomp_available,
            report.cgroupv2_available,
            report.cgroupv2_writable,
        );
        if args.strict_probes {
            log::info!(
                target: "server::init",
                "conformance probes: {}",
                if report.probes_passed { "passed" } else { "FAILED" }
            );
        }
    }

    let state = state::AppState {
        backend: Arc::new(tokio::sync::Mutex::new(backend_state)),
        configured_model: model,
        response_store: tokio::sync::Mutex::new(std::collections::HashMap::new()),
        gateway_secret,
        tenant_registry,
        bucket_path,
        workdir,
        agent_policy_json,
        agent_policy,
        html_dir: args.html_dir,
        plugin_tokens: tokio::sync::Mutex::new(std::collections::HashMap::new()),
        max_file_upload_bytes: args.max_file_upload_bytes,
        max_file_inspect_bytes: args.max_file_inspect_bytes,
        code_sessions: tokio::sync::Mutex::new(std::collections::HashMap::new()),
        code_session_ttl: listen::CODE_SESSION_TTL,
        shell_sessions: tokio::sync::Mutex::new(std::collections::HashMap::new()),
        shell_session_ttl: listen::SHELL_SESSION_TTL,
        process_sessions: tokio::sync::Mutex::new(std::collections::HashMap::new()),
        process_session_ttl: listen::PROCESS_SESSION_TTL,
        kv_handles: tokio::sync::Mutex::new(std::collections::HashMap::new()),
        collab_handles: tokio::sync::Mutex::new(std::collections::HashMap::new()),
        agent_runtime_mode: args.agent_runtime_mode,
        sandbox_backend,
        pubsub: tokio::sync::Mutex::new(pubsub::PubSubState::new()),
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
    if let Some(path) = args.policy_file.as_ref() {
        log::info!(
            target: "server::init",
            "agent runtime policy file: {}",
            path.display()
        );
    } else if state.agent_policy_json.is_some() {
        log::info!(
            target: "server::init",
            "agent runtime policy loaded from TALU_AGENT_POLICY_JSON"
        );
    } else {
        log::info!(target: "server::init", "agent runtime policy: none");
    }
    log::info!(
        target: "server::init",
        "agent runtime mode: {:?}, sandbox backend: {:?}",
        state.agent_runtime_mode,
        state.sandbox_backend
    );
    if state.agent_runtime_mode == AgentRuntimeMode::Host {
        log::warn!(
            target: "server::init",
            "agent runtime mode host: terminal/process firewall guarantees are disabled"
        );
    } else {
        log::info!(
            target: "server::init",
            "agent runtime mode strict: kernel/runtime sandbox enforcement enabled"
        );
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

fn resolve_sandbox_backend(
    mode: AgentRuntimeMode,
    backend: SandboxBackendArg,
) -> Result<SandboxBackend> {
    match backend {
        SandboxBackendArg::Auto => {
            if cfg!(target_os = "linux") {
                Ok(SandboxBackend::LinuxLocal)
            } else if cfg!(target_os = "macos") {
                Ok(SandboxBackend::AppleContainer)
            } else {
                bail!("no default sandbox backend for this platform");
            }
        }
        SandboxBackendArg::LinuxLocal => {
            if mode == AgentRuntimeMode::Strict && !cfg!(target_os = "linux") {
                if cfg!(target_os = "macos") {
                    log::warn!(
                        target: "server::init",
                        "strict runtime requested with linux-local backend on macOS; using apple-container backend"
                    );
                    return Ok(SandboxBackend::AppleContainer);
                }
                bail!("strict runtime with linux-local backend is only supported on Linux");
            }
            Ok(SandboxBackend::LinuxLocal)
        }
        SandboxBackendArg::Oci => {
            if mode == AgentRuntimeMode::Strict {
                bail!("strict runtime with oci backend is not implemented yet");
            }
            Ok(SandboxBackend::Oci)
        }
        SandboxBackendArg::AppleContainer => {
            if mode == AgentRuntimeMode::Strict && !cfg!(target_os = "macos") {
                bail!("strict runtime with apple-container backend is only supported on macOS");
            }
            Ok(SandboxBackend::AppleContainer)
        }
    }
}

fn sandbox_backend_for_talu(backend: SandboxBackend) -> talu::shell::SandboxBackend {
    match backend {
        SandboxBackend::LinuxLocal => talu::shell::SandboxBackend::LinuxLocal,
        SandboxBackend::Oci => talu::shell::SandboxBackend::Oci,
        SandboxBackend::AppleContainer => talu::shell::SandboxBackend::AppleContainer,
    }
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
