use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use clap::Args;
use log::LevelFilter;
use talu::ChatHandle;

pub mod auth_gateway;
pub mod batch_scheduler;
pub mod completions;
pub mod completions_types;
pub mod events;
pub mod handlers;
pub mod http;
pub mod listen;
mod logger;
pub mod openapi;
pub mod repo;
pub mod responses;
mod responses_openapi;
pub mod responses_types;
pub mod state;
pub mod tenant;
pub mod tokenizer;
pub mod vision;

#[derive(Args, Debug)]
pub struct ServerArgs {
    /// Model: local path or HuggingFace model ID
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

    /// Canonical workdir root for request-scoped integrations.
    #[arg(long, env = "TALU_WORKDIR")]
    pub workdir: Option<PathBuf>,
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
    tokenizer::validate_path_source_policy_env()
        .map_err(|e| anyhow::anyhow!("invalid tokenizer path-source policy configuration: {e}"))?;

    let model = args.model.clone();
    let socket_path = expand_socket_path(&args.socket);
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

    let workdir = args
        .workdir
        .or_else(|| std::env::var_os("TALU_WORKDIR").map(PathBuf::from))
        .map(|path| {
            path.canonicalize()
                .with_context(|| format!("canonicalize workdir path: {}", path.display()))
        })
        .transpose()?;

    // Create batch scheduler for the local backend (enables concurrent GPU decode).
    let batch_scheduler = backend_state.backend.as_ref().and_then(|b| {
        match batch_scheduler::SchedulerState::new(b, None) {
            Ok(s) => {
                log::info!(target: "server::init", "batch scheduler created for local backend");
                Some(Arc::new(s))
            }
            Err(e) => {
                log::debug!(target: "server::init", "batch scheduler not available: {e}");
                None
            }
        }
    });

    // Warm up scheduler/model hot path once at startup so the first user
    // request does not pay one-time initialization latency.
    if let Some(ref sched) = batch_scheduler {
        warmup_batch_scheduler(sched, backend_state.current_model.as_deref());
    }

    let state = state::AppState {
        backend: Arc::new(tokio::sync::Mutex::new(backend_state)),
        batch_scheduler: std::sync::Mutex::new(batch_scheduler),
        configured_model: model,
        response_store: tokio::sync::Mutex::new(std::collections::HashMap::new()),
        gateway_secret,
        tenant_registry,
        workdir,
        tokenizer_instances: tokio::sync::Mutex::new(std::collections::HashMap::new()),
        active_stop_flags: std::sync::Mutex::new(Vec::new()),
        drain_thread: std::sync::Mutex::new(None),
        model_load_inflight: std::sync::Mutex::new(std::collections::HashMap::new()),
    };

    let addr = SocketAddr::new(args.host, args.port);
    log::info!(target: "server::init", "talu server starting");
    log::info!(target: "server::init", "stateless runtime (local persistence disabled)");
    log::info!(target: "server::init", "listening on http://{}", addr);
    log::info!(target: "server::init", "listening on unix://{}", socket_path.display());
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(listen::serve(state, addr, socket_path))?;
    // Give spawn_blocking tasks up to 3 seconds to respond to stop flags,
    // then forcefully shut down so the process exits promptly on Ctrl+C.
    runtime.shutdown_timeout(std::time::Duration::from_secs(3));
    Ok(())
}

pub(crate) fn warmup_batch_scheduler(
    scheduler: &Arc<batch_scheduler::SchedulerState>,
    model_id: Option<&str>,
) {
    use std::sync::mpsc::RecvTimeoutError;

    let started = std::time::Instant::now();
    let chat = match ChatHandle::new(None) {
        Ok(c) => c,
        Err(e) => {
            log::warn!(target: "server::init", "warmup skipped (chat init failed): {e}");
            return;
        }
    };
    // Keep warmup prompt tiny; goal is priming request path, not benchmarking.
    if let Err(e) = chat.load_completions_json(r#"[{"role":"user","content":"warmup"}]"#) {
        log::warn!(target: "server::init", "warmup skipped (message load failed): {e}");
        return;
    }

    let mut cfg = talu::router::GenerateConfig::default();
    cfg.completions_mode = true;
    cfg.max_tokens = 1;
    cfg.temperature = 0.0;
    let stop_flag = Arc::new(AtomicBool::new(false));

    let submit = scheduler.submit_final_only(&chat, cfg, stop_flag);
    let (request_id, event_rx) = match submit {
        Ok(v) => v,
        Err(e) => {
            log::warn!(target: "server::init", "warmup submit failed: {e}");
            return;
        }
    };

    let warmup_timeout_ms = std::env::var("TALU_SCHEDULER_WARMUP_TIMEOUT_MS")
        .ok()
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .filter(|ms| *ms > 0)
        .unwrap_or(1_500);
    let timeout = std::time::Duration::from_millis(warmup_timeout_ms);

    let mut saw_final = false;
    let mut final_error: Option<String> = None;

    loop {
        match event_rx.recv_timeout(timeout) {
            Ok(event) => {
                if matches!(event.event_type, talu::batch::EventType::Error) {
                    final_error = Some(if event.text.is_empty() {
                        "scheduler run loop failed during warmup".to_string()
                    } else {
                        event.text.clone()
                    });
                }
                if event.is_final {
                    saw_final = true;
                    break;
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                log::warn!(
                    target: "server::init",
                    "warmup timed out: model={} timeout_ms={}",
                    model_id.unwrap_or("(unknown)"),
                    warmup_timeout_ms
                );
                break;
            }
            Err(RecvTimeoutError::Disconnected) => {
                log::warn!(
                    target: "server::init",
                    "warmup event channel closed early: model={}",
                    model_id.unwrap_or("(unknown)")
                );
                break;
            }
        }
    }

    if !saw_final {
        scheduler.cancel(request_id);
        let _ = scheduler.take_result(request_id);
        return;
    }

    if let Some(msg) = final_error {
        let _ = scheduler.take_result(request_id);
        log::warn!(
            target: "server::init",
            "warmup failed: model={} error={}",
            model_id.unwrap_or("(unknown)"),
            msg
        );
        return;
    }

    if scheduler.take_result(request_id).is_none() {
        log::warn!(
            target: "server::init",
            "warmup finished without result: model={}",
            model_id.unwrap_or("(unknown)")
        );
        return;
    }
    log::info!(
        target: "server::init",
        "warmup complete: model={} elapsed_ms={:.2}",
        model_id.unwrap_or("(unknown)"),
        started.elapsed().as_secs_f64() * 1000.0
    );
}

pub fn expand_socket_path(path: &str) -> PathBuf {
    if let Some(stripped) = path.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(stripped);
        }
    }
    PathBuf::from(path)
}
