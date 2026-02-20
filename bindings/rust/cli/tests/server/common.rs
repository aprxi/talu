//! Shared test fixtures for the server integration tests.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

use serde::Serialize;
use tempfile::TempDir;

#[derive(Debug, Clone, Serialize)]
pub struct TenantSpec {
    pub id: String,
    pub storage_prefix: String,
    #[serde(default)]
    pub allowed_models: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub model: Option<String>,
    pub gateway_secret: Option<String>,
    pub tenants: Vec<TenantSpec>,
    pub bucket: Option<PathBuf>,
    pub no_bucket: bool,
    /// Serve console UI from this directory instead of bundled assets.
    pub html_dir: Option<PathBuf>,
    /// Extra environment variables to set on the server process.
    pub env_vars: Vec<(String, String)>,
}

impl ServerConfig {
    pub fn new() -> Self {
        Self {
            model: None,
            gateway_secret: None,
            tenants: Vec::new(),
            bucket: None,
            no_bucket: false,
            html_dir: None,
            env_vars: Vec::new(),
        }
    }
}

pub struct ServerTestContext {
    _temp_dir: TempDir,
    addr: SocketAddr,
    child: Child,
}

impl ServerTestContext {
    pub fn new(config: ServerConfig) -> Self {
        let temp_dir = TempDir::new().expect("temp dir");
        let port = pick_free_port();
        let addr = SocketAddr::from(([127, 0, 0, 1], port));
        let socket_path = temp_dir.path().join("talu.sock");

        let mut command = Command::new(talu_bin_path());
        command
            .arg("serve")
            .arg("--port")
            .arg(port.to_string())
            .arg("--socket")
            .arg(&socket_path)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            // Clear environment variables that might cause conflicts
            .env_remove("LD_LIBRARY_PATH")
            .env_remove("LD_PRELOAD")
            .env_remove("MALLOC_CHECK_")
            .env_remove("MALLOC_PERTURB_")
            // Ensure clean environment for subprocess
            .env("TALU_LOG_LEVEL", "info");

        if let Some(model) = config.model.as_ref() {
            command.arg("--model").arg(model);
            if model.starts_with("openai::") {
                command.env("OPENAI_ENDPOINT", "http://127.0.0.1:1");
            }
        }

        if let Some(secret) = config.gateway_secret.as_ref() {
            let tenant_path = write_tenant_config(temp_dir.path(), &config.tenants);
            command
                .arg("--gateway-secret")
                .arg(secret)
                .arg("--tenant-config")
                .arg(&tenant_path);
        }

        if config.no_bucket {
            command.arg("--no-bucket");
        } else if let Some(ref bucket) = config.bucket {
            command.arg("--bucket").arg(bucket);
        }

        if let Some(ref html_dir) = config.html_dir {
            command.arg("--html-dir").arg(html_dir);
        }

        for (key, value) in &config.env_vars {
            command.env(key, value);
        }

        let mut child = command.spawn().expect("spawn server");
        let (tx, rx) = mpsc::channel();

        if let Some(stdout) = child.stdout.take() {
            spawn_reader(stdout, tx.clone());
        }
        if let Some(stderr) = child.stderr.take() {
            spawn_reader(stderr, tx.clone());
        }

        wait_for_ready(&rx, &mut child, addr);

        Self {
            _temp_dir: temp_dir,
            addr,
            child,
        }
    }

    pub fn addr(&self) -> SocketAddr {
        self.addr
    }
}

impl Drop for ServerTestContext {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub struct HttpResponse {
    pub status: u16,
    pub headers: String,
    pub body: String,
}

impl HttpResponse {
    /// Parse a header value by name (case-insensitive).
    pub fn header(&self, name: &str) -> Option<&str> {
        let lower = name.to_ascii_lowercase();
        for line in self.headers.lines() {
            if let Some((key, value)) = line.split_once(':') {
                if key.trim().to_ascii_lowercase() == lower {
                    return Some(value.trim());
                }
            }
        }
        None
    }

    /// Parse response body as JSON.
    pub fn json(&self) -> serde_json::Value {
        serde_json::from_str(&self.body).unwrap_or_else(|e| {
            panic!(
                "invalid JSON: {e}\nbody: {}",
                &self.body[..self.body.len().min(500)]
            )
        })
    }
}

/// Return the model path if TALU_TEST_MODEL is set, or None.
pub fn try_model_path() -> Option<String> {
    std::env::var("TALU_TEST_MODEL").ok()
}

/// Return the model path from TALU_TEST_MODEL env var.
/// Panics if not set — prefer `require_model!()` in tests.
pub fn model_path() -> String {
    try_model_path().expect("TALU_TEST_MODEL must be set to an absolute model path")
}

/// Build a ServerConfig with the test model pre-loaded.
/// Panics if TALU_TEST_MODEL is not set — prefer `require_model!()` first.
pub fn model_config() -> ServerConfig {
    let mut config = ServerConfig::new();
    config.model = Some(model_path());
    config
}

/// Skip the current test if TALU_TEST_MODEL is not set.
/// Returns the model path string if available.
macro_rules! require_model {
    () => {
        match $crate::server::common::try_model_path() {
            Some(path) => path,
            None => {
                eprintln!("Skipped: TALU_TEST_MODEL not set");
                return;
            }
        }
    };
}
pub(crate) use require_model;

/// POST JSON to a path on the server. Convenience wrapper around send_request.
pub fn post_json(addr: SocketAddr, path: &str, body: &serde_json::Value) -> HttpResponse {
    let json = serde_json::to_string(body).expect("serialize json");
    send_request(
        addr,
        "POST",
        path,
        &[("Content-Type", "application/json")],
        Some(&json),
    )
}

/// GET a path on the server.
pub fn get(addr: SocketAddr, path: &str) -> HttpResponse {
    send_request(addr, "GET", path, &[], None)
}

/// DELETE a path on the server.
pub fn delete(addr: SocketAddr, path: &str) -> HttpResponse {
    send_request(addr, "DELETE", path, &[], None)
}

/// PATCH JSON to a path on the server.
pub fn patch_json(addr: SocketAddr, path: &str, body: &serde_json::Value) -> HttpResponse {
    let json = serde_json::to_string(body).expect("serialize json");
    send_request(
        addr,
        "PATCH",
        path,
        &[("Content-Type", "application/json")],
        Some(&json),
    )
}

/// PUT JSON to a path on the server.
pub fn put_json(addr: SocketAddr, path: &str, body: &serde_json::Value) -> HttpResponse {
    let json = serde_json::to_string(body).expect("serialize json");
    send_request(
        addr,
        "PUT",
        path,
        &[("Content-Type", "application/json")],
        Some(&json),
    )
}

/// DELETE with JSON body to a path on the server.
pub fn delete_json(addr: SocketAddr, path: &str, body: &serde_json::Value) -> HttpResponse {
    let json = serde_json::to_string(body).expect("serialize json");
    send_request(
        addr,
        "DELETE",
        path,
        &[("Content-Type", "application/json")],
        Some(&json),
    )
}

pub fn send_request(
    addr: SocketAddr,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: Option<&str>,
) -> HttpResponse {
    let body = body.unwrap_or("");
    let mut request = format!(
        "{method} {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n",
        host = addr
    );
    for (name, value) in headers {
        request.push_str(&format!("{name}: {value}\r\n"));
    }
    request.push_str(&format!("Content-Length: {}\r\n\r\n", body.len()));
    request.push_str(body);

    eprintln!("[DEBUG] Connecting to {addr}...");
    let mut stream =
        TcpStream::connect_timeout(&addr.into(), Duration::from_secs(5)).expect("connect");
    eprintln!("[DEBUG] Connected to {addr}");
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .expect("set read timeout");
    stream
        .set_write_timeout(Some(Duration::from_secs(5)))
        .expect("set write timeout");
    eprintln!("[DEBUG] Request:\n{request}");
    stream.write_all(request.as_bytes()).expect("write request");
    stream.flush().expect("flush");
    eprintln!("[DEBUG] Request sent to {addr}");

    let mut raw = Vec::new();
    loop {
        let mut buf = [0u8; 8192];
        match stream.read(&mut buf) {
            Ok(0) => {
                eprintln!("[DEBUG] Read returned 0 bytes (EOF)");
                break;
            }
            Ok(n) => {
                eprintln!("[DEBUG] Read {n} bytes");
                raw.extend_from_slice(&buf[..n]);
            }
            Err(e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
            {
                if raw.is_empty() {
                    panic!(
                        "Read timeout waiting for response to {method} {path} \
                         (no data received within 30s)"
                    );
                }
                // Partial response received, server stopped sending — done.
                eprintln!("[DEBUG] Read timeout after {} bytes", raw.len());
                break;
            }
            Err(e) => panic!("read error: {e}"),
        }
    }
    if raw.is_empty() {
        eprintln!("[DEBUG] Empty response from server for {method} {path}");
    }

    let response = String::from_utf8_lossy(&raw);
    let (head, body_raw) = response.split_once("\r\n\r\n").unwrap_or((&response, ""));

    let status = head
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|code| code.parse::<u16>().ok())
        .unwrap_or(0);

    let header_section = head.to_string();

    // Decode chunked transfer-encoding if present.
    let is_chunked = head.lines().any(|l| {
        l.to_ascii_lowercase().starts_with("transfer-encoding:")
            && l.to_ascii_lowercase().contains("chunked")
    });

    let decoded_body = if is_chunked {
        decode_chunked(body_raw)
    } else {
        body_raw.to_string()
    };

    HttpResponse {
        status,
        headers: header_section,
        body: decoded_body,
    }
}

/// Decode a chunked transfer-encoded body.
fn decode_chunked(raw: &str) -> String {
    let mut result = String::new();
    let mut remaining = raw;

    loop {
        // Skip leading \r\n
        remaining = remaining.trim_start_matches("\r\n");
        if remaining.is_empty() {
            break;
        }

        // Read chunk size (hex)
        let size_end = remaining.find("\r\n").unwrap_or(remaining.len());
        let size_str = &remaining[..size_end];
        // Chunk size may have extensions after a semicolon
        let hex = size_str.split(';').next().unwrap_or("0").trim();
        let chunk_size = usize::from_str_radix(hex, 16).unwrap_or(0);

        if chunk_size == 0 {
            break;
        }

        remaining = &remaining[size_end + 2..]; // skip size line + \r\n
        let end = chunk_size.min(remaining.len());
        result.push_str(&remaining[..end]);
        remaining = &remaining[end..];
    }

    result
}

fn write_tenant_config(dir: &Path, tenants: &[TenantSpec]) -> PathBuf {
    let path = dir.join("tenants.json");
    let payload = serde_json::to_string_pretty(tenants).expect("serialize tenants");
    std::fs::write(&path, payload).expect("write tenants.json");
    path
}

fn pick_free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    drop(listener);
    port
}

fn spawn_reader<R>(reader: R, tx: mpsc::Sender<String>)
where
    R: Read + Send + 'static,
{
    std::thread::spawn(move || {
        let buf = BufReader::new(reader);
        for line in buf.lines().flatten() {
            eprintln!("[SERVER LOG] {}", &line);
            let _ = tx.send(line);
        }
    });
}

fn wait_for_ready(rx: &Receiver<String>, child: &mut Child, addr: SocketAddr) {
    let marker = format!("TCP listener bound at {addr}");
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut logs = Vec::new();

    loop {
        if let Ok(Some(status)) = child.try_wait() {
            panic!("server exited early: {status}\nlogs: {logs:?}");
        }

        let timeout = deadline.saturating_duration_since(Instant::now());
        if timeout.is_zero() {
            panic!("server did not become ready\nlogs: {logs:?}");
        }

        match rx.recv_timeout(timeout) {
            Ok(line) => {
                if line.contains(&marker) {
                    eprintln!("[DEBUG] Server ready at {addr}");
                    // Small delay to ensure server is fully ready to accept connections
                    std::thread::sleep(Duration::from_millis(50));
                    return;
                }
                logs.push(line);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                panic!("server did not become ready\nlogs: {logs:?}");
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                panic!("server output closed\nlogs: {logs:?}");
            }
        }
    }
}

fn talu_bin_path() -> PathBuf {
    if let Ok(path) = std::env::var("TALU_CLI_BIN") {
        let path = PathBuf::from(path);
        if path.exists() {
            return path;
        }
        panic!("TALU_CLI_BIN set but not found: {}", path.display());
    }

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..");
    let candidate = root.join("zig-out").join("bin").join("talu");
    if candidate.exists() {
        return candidate;
    }

    panic!(
        "talu CLI binary not found. Build with `zig build release -Drelease` or set TALU_CLI_BIN."
    );
}
