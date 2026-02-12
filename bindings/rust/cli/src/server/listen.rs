use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use hyper::server::conn::http1;
use hyper_util::rt::TokioIo;
use hyper_util::service::TowerToHyperService;
use tokio::net::{TcpListener, TcpStream};

#[cfg(unix)]
use tokio::net::{UnixListener, UnixStream};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

use crate::server::http::Router;
use crate::server::state::AppState;

pub async fn serve(state: AppState, addr: SocketAddr, socket: PathBuf) -> Result<()> {
    let state = Arc::new(state);
    let router = Router::new(state.clone());

    let tcp_listener = TcpListener::bind(addr)
        .await
        .with_context(|| format!("Failed to bind TCP listener at {}", addr))?;
    log::info!("TCP listener bound at {}", addr);
    let tcp_task = tokio::spawn(accept_tcp(tcp_listener, router.clone()));

    #[cfg(unix)]
    let socket_guard = prepare_unix_socket(&socket)?;
    #[cfg(unix)]
    let uds_task = {
        let listener = UnixListener::bind(&socket)
            .with_context(|| format!("Failed to bind UDS listener at {}", socket.display()))?;
        let _ = std::fs::set_permissions(&socket, std::fs::Permissions::from_mode(0o600));
        log::info!("UDS listener bound at {}", socket.display());
        tokio::spawn(accept_unix(listener, router))
    };

    #[cfg(not(unix))]
    let _socket_guard = socket;

    #[cfg(unix)]
    {
        tokio::select! {
            _ = tcp_task => {},
            _ = uds_task => {},
            _ = tokio::signal::ctrl_c() => {},
        }
    }

    #[cfg(not(unix))]
    {
        tokio::select! {
            _ = tcp_task => {},
            _ = tokio::signal::ctrl_c() => {},
        }
    }

    #[cfg(unix)]
    {
        drop(socket_guard);
    }

    Ok(())
}

async fn accept_tcp(listener: TcpListener, router: Router) {
    loop {
        let (stream, _) = match listener.accept().await {
            Ok(conn) => conn,
            Err(_) => continue,
        };
        let router = router.clone();
        tokio::spawn(async move {
            if let Err(_err) = serve_tcp_connection(stream, router).await {
                // ignore per-connection errors
            }
        });
    }
}

async fn serve_tcp_connection(stream: TcpStream, router: Router) -> Result<()> {
    let io = TokioIo::new(stream);
    let service: TowerToHyperService<Router> = TowerToHyperService::new(router);
    http1::Builder::new()
        .serve_connection(io, service)
        .await
        .context("TCP connection failed")?;
    Ok(())
}

#[cfg(unix)]
async fn accept_unix(listener: UnixListener, router: Router) {
    loop {
        let (stream, _) = match listener.accept().await {
            Ok(conn) => conn,
            Err(_) => continue,
        };
        let router = router.clone();
        tokio::spawn(async move {
            if let Err(_err) = serve_unix_connection(stream, router).await {
                // ignore per-connection errors
            }
        });
    }
}

#[cfg(unix)]
async fn serve_unix_connection(stream: UnixStream, router: Router) -> Result<()> {
    let io = TokioIo::new(stream);
    let service: TowerToHyperService<Router> = TowerToHyperService::new(router);
    http1::Builder::new()
        .serve_connection(io, service)
        .await
        .context("UDS connection failed")?;
    Ok(())
}

#[cfg(unix)]
fn prepare_unix_socket(path: &Path) -> Result<SocketGuard> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create socket dir {}", parent.display()))?;
    }
    if path.exists() {
        std::fs::remove_file(path)
            .with_context(|| format!("Failed to remove stale socket {}", path.display()))?;
    }
    Ok(SocketGuard {
        path: path.to_path_buf(),
    })
}

#[cfg(unix)]
struct SocketGuard {
    path: PathBuf,
}

#[cfg(unix)]
impl Drop for SocketGuard {
    fn drop(&mut self) {
        if self.path.exists() {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}
