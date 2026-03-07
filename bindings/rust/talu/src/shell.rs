//! Safe wrappers for shell execution and command safety C APIs (`talu_shell_*`).

use std::ffi::{c_void, CString};
use std::os::raw::{c_char, c_int};

const ERROR_CODE_OK: i32 = 0;
const ERROR_CODE_SHELL_COMMAND_DENIED: i32 = 800;
const ERROR_CODE_SHELL_EXEC_FAILED: i32 = 801;
const ERROR_CODE_SHELL_SESSION_CLOSED: i32 = 802;
const ERROR_CODE_POLICY_DENIED_EXEC: i32 = 808;
const ERROR_CODE_POLICY_DENIED_CWD: i32 = 809;
const ERROR_CODE_POLICY_STRICT_UNAVAILABLE: i32 = 811;
const ERROR_CODE_POLICY_STRICT_SETUP_FAILED: i32 = 812;
const ERROR_CODE_SANDBOX_DETECT_FAILED: i32 = 813;
const ERROR_CODE_SANDBOX_PROBE_FAILED: i32 = 814;
const ERROR_CODE_SANDBOX_CGROUP_UNAVAILABLE: i32 = 815;
const ERROR_CODE_INVALID_ARGUMENT: i32 = 901;
const ERROR_CODE_INVALID_HANDLE: i32 = 902;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum AgentRuntimeMode {
    Host = 0,
    Strict = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum SandboxBackend {
    LinuxLocal = 0,
    Oci = 1,
    AppleContainer = 2,
}

type StreamCallback = unsafe extern "C" fn(*mut c_void, *const u8, usize) -> bool;

unsafe extern "C" {
    #[link_name = "talu_shell_exec"]
    fn talu_shell_exec_raw(
        command: *const c_char,
        cwd: *const c_char,
        policy: *mut c_void,
        runtime_mode: c_int,
        sandbox_backend: c_int,
        out_stdout: *mut *const u8,
        out_stdout_len: *mut usize,
        out_stderr: *mut *const u8,
        out_stderr_len: *mut usize,
        out_exit_code: *mut i32,
    ) -> c_int;

    #[link_name = "talu_shell_exec_streaming"]
    fn talu_shell_exec_streaming_raw(
        command: *const c_char,
        cwd: *const c_char,
        policy: *mut c_void,
        runtime_mode: c_int,
        sandbox_backend: c_int,
        timeout_ms: u64,
        on_stdout: Option<StreamCallback>,
        on_stdout_ctx: *mut c_void,
        on_stderr: Option<StreamCallback>,
        on_stderr_ctx: *mut c_void,
        out_exit_code: *mut i32,
    ) -> c_int;

    #[link_name = "talu_shell_check_command"]
    fn talu_shell_check_command_raw(
        command: *const c_char,
        out_allowed: *mut bool,
        out_reason: *mut *const u8,
        out_reason_len: *mut usize,
    ) -> c_int;

    #[link_name = "talu_shell_default_policy_json"]
    fn talu_shell_default_policy_json_raw(
        out_json: *mut *const u8,
        out_json_len: *mut usize,
    ) -> c_int;

    #[link_name = "talu_shell_normalize_command"]
    fn talu_shell_normalize_command_raw(
        command: *const c_char,
        out_normalized: *mut *const u8,
        out_normalized_len: *mut usize,
    ) -> c_int;

    #[link_name = "talu_shell_open"]
    fn talu_shell_open_raw(
        cols: u16,
        rows: u16,
        cwd: *const c_char,
        policy: *mut c_void,
        runtime_mode: c_int,
        sandbox_backend: c_int,
        out_shell: *mut *mut c_void,
    ) -> c_int;

    #[link_name = "talu_shell_close"]
    fn talu_shell_close_raw(shell_handle: *mut c_void);

    #[link_name = "talu_shell_write"]
    fn talu_shell_write_raw(shell_handle: *mut c_void, data: *const u8, len: usize) -> c_int;

    #[link_name = "talu_shell_read"]
    fn talu_shell_read_raw(
        shell_handle: *mut c_void,
        buf: *mut u8,
        buf_len: usize,
        out_read: *mut usize,
    ) -> c_int;

    #[link_name = "talu_shell_resize"]
    fn talu_shell_resize_raw(shell_handle: *mut c_void, cols: u16, rows: u16) -> c_int;

    #[link_name = "talu_shell_signal"]
    fn talu_shell_signal_raw(shell_handle: *mut c_void, signal: u8) -> c_int;

    #[link_name = "talu_shell_alive"]
    fn talu_shell_alive_raw(shell_handle: *mut c_void) -> bool;

    #[link_name = "talu_shell_scrollback"]
    fn talu_shell_scrollback_raw(
        shell_handle: *mut c_void,
        out_data: *mut *const u8,
        out_len: *mut usize,
    ) -> c_int;

    #[link_name = "talu_shell_free_string"]
    fn talu_shell_free_string_raw(ptr: *const u8, len: usize);

    #[link_name = "talu_agent_runtime_validate_strict"]
    fn talu_agent_runtime_validate_strict_raw(
        policy: *mut c_void,
        cwd: *const c_char,
        sandbox_backend: c_int,
    ) -> c_int;

    #[link_name = "talu_agent_runtime_validate_strict_ext"]
    fn talu_agent_runtime_validate_strict_ext_raw(
        sandbox_backend: c_int,
        strict_required: bool,
        run_probes: bool,
        cwd: *const c_char,
        out_report: *mut CapabilityReport,
    ) -> c_int;
}

/// Output from executing a shell command.
#[derive(Debug)]
pub struct ExecOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
}

/// Result of a safety check on a command.
#[derive(Debug)]
pub struct SafetyCheck {
    pub allowed: bool,
    pub reason: Option<String>,
}

/// Errors from shell operations.
#[derive(Debug)]
pub enum ShellError {
    CommandDenied(String),
    PolicyDeniedExec(String),
    PolicyDeniedCwd(String),
    StrictUnavailable(String),
    StrictSetupFailed(String),
    SandboxDetectFailed(String),
    SandboxProbeFailed(String),
    SandboxCgroupUnavailable(String),
    ExecFailed(String),
    SessionClosed(String),
    InvalidArgument(String),
    InvalidHandle(String),
    Internal(i32),
}

impl ShellError {
    fn from_code(code: i32, fallback: &str) -> Self {
        let detail = crate::error::last_error_message().unwrap_or_else(|| fallback.to_string());
        match code {
            ERROR_CODE_SHELL_COMMAND_DENIED => Self::CommandDenied(detail),
            ERROR_CODE_POLICY_DENIED_EXEC => Self::PolicyDeniedExec(detail),
            ERROR_CODE_POLICY_DENIED_CWD => Self::PolicyDeniedCwd(detail),
            ERROR_CODE_POLICY_STRICT_UNAVAILABLE => Self::StrictUnavailable(detail),
            ERROR_CODE_POLICY_STRICT_SETUP_FAILED => Self::StrictSetupFailed(detail),
            ERROR_CODE_SANDBOX_DETECT_FAILED => Self::SandboxDetectFailed(detail),
            ERROR_CODE_SANDBOX_PROBE_FAILED => Self::SandboxProbeFailed(detail),
            ERROR_CODE_SANDBOX_CGROUP_UNAVAILABLE => Self::SandboxCgroupUnavailable(detail),
            ERROR_CODE_SHELL_EXEC_FAILED => Self::ExecFailed(detail),
            ERROR_CODE_SHELL_SESSION_CLOSED => Self::SessionClosed(detail),
            ERROR_CODE_INVALID_ARGUMENT => Self::InvalidArgument(detail),
            ERROR_CODE_INVALID_HANDLE => Self::InvalidHandle(detail),
            _ => Self::Internal(code),
        }
    }
}

impl std::fmt::Display for ShellError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShellError::CommandDenied(s) => write!(f, "command denied: {}", s),
            ShellError::PolicyDeniedExec(s) => write!(f, "policy denied exec: {}", s),
            ShellError::PolicyDeniedCwd(s) => write!(f, "policy denied cwd: {}", s),
            ShellError::StrictUnavailable(s) => write!(f, "strict runtime unavailable: {}", s),
            ShellError::StrictSetupFailed(s) => write!(f, "strict runtime setup failed: {}", s),
            ShellError::SandboxDetectFailed(s) => write!(f, "sandbox detection failed: {}", s),
            ShellError::SandboxProbeFailed(s) => write!(f, "sandbox probe failed: {}", s),
            ShellError::SandboxCgroupUnavailable(s) => {
                write!(f, "sandbox cgroup unavailable: {}", s)
            }
            ShellError::ExecFailed(s) => write!(f, "execution failed: {}", s),
            ShellError::SessionClosed(s) => write!(f, "session closed: {}", s),
            ShellError::InvalidArgument(s) => write!(f, "invalid argument: {}", s),
            ShellError::InvalidHandle(s) => write!(f, "invalid shell handle: {}", s),
            ShellError::Internal(code) => write!(f, "internal error (code {})", code),
        }
    }
}

impl std::error::Error for ShellError {}

/// RAII shell session for interactive PTY access.
#[derive(Debug)]
pub struct ShellSession {
    handle: *mut c_void,
}

// SAFETY: `ShellSession` contains only an opaque native handle. All mutation
// and I/O are funneled through methods taking `&mut self`; in server usage it
// is additionally guarded by `tokio::sync::Mutex`, so operations are serialized.
// The underlying core API treats the handle as owned state with explicit close.
unsafe impl Send for ShellSession {}

impl ShellSession {
    /// Open a new interactive shell session.
    pub fn open(cols: u16, rows: u16, cwd: Option<&str>) -> Result<Self, ShellError> {
        Self::open_with_policy(cols, rows, cwd, None)
    }

    /// Open a new interactive shell session with an optional agent policy.
    pub fn open_with_policy(
        cols: u16,
        rows: u16,
        cwd: Option<&str>,
        policy: Option<&crate::policy::Policy>,
    ) -> Result<Self, ShellError> {
        Self::open_with_policy_runtime(
            cols,
            rows,
            cwd,
            policy,
            AgentRuntimeMode::Host,
            SandboxBackend::LinuxLocal,
        )
    }

    /// Open a new interactive shell session with policy and explicit runtime configuration.
    pub fn open_with_policy_runtime(
        cols: u16,
        rows: u16,
        cwd: Option<&str>,
        policy: Option<&crate::policy::Policy>,
        runtime_mode: AgentRuntimeMode,
        sandbox_backend: SandboxBackend,
    ) -> Result<Self, ShellError> {
        let c_cwd = if let Some(value) = cwd {
            Some(
                CString::new(value)
                    .map_err(|_| ShellError::InvalidArgument("cwd contains null byte".into()))?,
            )
        } else {
            None
        };

        let mut handle: *mut c_void = std::ptr::null_mut();
        let policy_ptr = policy.map(|p| p.as_ptr()).unwrap_or(std::ptr::null_mut());
        // SAFETY: pointers are valid for the duration of this call.
        let rc = unsafe {
            talu_shell_open_raw(
                cols,
                rows,
                c_cwd.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
                policy_ptr,
                runtime_mode as c_int,
                sandbox_backend as c_int,
                &mut handle,
            )
        };
        if rc != ERROR_CODE_OK {
            return Err(ShellError::from_code(rc, "failed to open shell session"));
        }
        if handle.is_null() {
            return Err(ShellError::Internal(ERROR_CODE_SHELL_EXEC_FAILED));
        }

        Ok(Self { handle })
    }

    /// Write bytes to shell stdin.
    pub fn write(&mut self, data: &[u8]) -> Result<(), ShellError> {
        let ptr = if data.is_empty() {
            std::ptr::null()
        } else {
            data.as_ptr()
        };
        // SAFETY: handle and data pointer are valid for the duration of this call.
        let rc = unsafe { talu_shell_write_raw(self.handle, ptr, data.len()) };
        if rc != ERROR_CODE_OK {
            return Err(ShellError::from_code(rc, "failed to write shell session"));
        }
        Ok(())
    }

    /// Read bytes from shell stdout/stderr into `buf`.
    pub fn read(&mut self, buf: &mut [u8]) -> Result<usize, ShellError> {
        let mut read = 0usize;
        let ptr = if buf.is_empty() {
            std::ptr::null_mut()
        } else {
            buf.as_mut_ptr()
        };
        // SAFETY: handle and output pointers are valid for the duration of this call.
        let rc = unsafe { talu_shell_read_raw(self.handle, ptr, buf.len(), &mut read) };
        if rc != ERROR_CODE_OK {
            return Err(ShellError::from_code(rc, "failed to read shell session"));
        }
        Ok(read)
    }

    /// Resize PTY window.
    pub fn resize(&mut self, cols: u16, rows: u16) -> Result<(), ShellError> {
        // SAFETY: handle is valid for the duration of this call.
        let rc = unsafe { talu_shell_resize_raw(self.handle, cols, rows) };
        if rc != ERROR_CODE_OK {
            return Err(ShellError::from_code(rc, "failed to resize shell session"));
        }
        Ok(())
    }

    /// Send POSIX signal to shell process.
    pub fn signal(&mut self, signal: u8) -> Result<(), ShellError> {
        // SAFETY: handle is valid for the duration of this call.
        let rc = unsafe { talu_shell_signal_raw(self.handle, signal) };
        if rc != ERROR_CODE_OK {
            return Err(ShellError::from_code(rc, "failed to signal shell session"));
        }
        Ok(())
    }

    /// Return whether the shell process is alive.
    pub fn is_alive(&mut self) -> Result<bool, ShellError> {
        // SAFETY: clear thread-local C error before probing boolean API.
        unsafe { talu_sys::talu_clear_error() };
        // SAFETY: handle is valid for the duration of this call.
        let alive = unsafe { talu_shell_alive_raw(self.handle) };
        if !alive {
            if let Some(msg) = crate::error::last_error_message() {
                // Distinguish session-closed from invalid-handle errors.
                if msg.contains("session closed") {
                    return Err(ShellError::SessionClosed(msg));
                }
                return Err(ShellError::InvalidHandle(msg));
            }
        }
        Ok(alive)
    }

    /// Copy shell scrollback buffer.
    pub fn scrollback(&mut self) -> Result<Vec<u8>, ShellError> {
        let mut ptr: *const u8 = std::ptr::null();
        let mut len: usize = 0;
        // SAFETY: handle and output pointers are valid for this call.
        let rc = unsafe { talu_shell_scrollback_raw(self.handle, &mut ptr, &mut len) };
        if rc != ERROR_CODE_OK {
            return Err(ShellError::from_code(rc, "failed to read shell scrollback"));
        }

        let bytes = if ptr.is_null() || len == 0 {
            Vec::new()
        } else {
            // SAFETY: C API returns a valid buffer of `len` bytes on success.
            unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
        };

        // SAFETY: pointer+len originate from shell C API and are freed exactly once here.
        unsafe { talu_shell_free_string_raw(ptr, len) };
        Ok(bytes)
    }

    /// Close shell session. Safe to call multiple times.
    pub fn close(&mut self) {
        if self.handle.is_null() {
            return;
        }
        // SAFETY: handle is owned by this ShellSession and is closed at most once.
        unsafe { talu_shell_close_raw(self.handle) };
        self.handle = std::ptr::null_mut();
    }
}

impl Drop for ShellSession {
    fn drop(&mut self) {
        self.close();
    }
}

/// Validate strict runtime support and precompile runtime policy profiles.
///
/// This is intended to be called once at server startup when strict mode is enabled.
pub fn validate_strict_runtime(
    policy: Option<&crate::policy::Policy>,
    cwd: Option<&str>,
    sandbox_backend: SandboxBackend,
) -> Result<(), ShellError> {
    let c_cwd = cwd
        .map(|value| {
            CString::new(value)
                .map_err(|_| ShellError::InvalidArgument("cwd contains null byte".into()))
        })
        .transpose()?;
    let policy_ptr = policy.map(|p| p.as_ptr()).unwrap_or(std::ptr::null_mut());
    let rc = unsafe {
        talu_agent_runtime_validate_strict_raw(
            policy_ptr,
            c_cwd.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
            sandbox_backend as c_int,
        )
    };
    if rc != ERROR_CODE_OK {
        return Err(ShellError::from_code(
            rc,
            "strict runtime validation failed",
        ));
    }
    Ok(())
}

/// Capability detection report from strict runtime validation.
#[derive(Debug, Clone, Default)]
#[repr(C)]
pub struct CapabilityReport {
    pub kernel_version_ok: bool,
    pub kernel_version_major: u32,
    pub kernel_version_minor: u32,
    pub landlock_available: bool,
    pub landlock_abi_version: u8,
    pub user_ns_available: bool,
    pub seccomp_available: bool,
    pub cgroupv2_available: bool,
    pub cgroupv2_writable: bool,
    pub probes_passed: bool,
}

/// Validate strict runtime with capability detection and optional conformance probes.
///
/// Returns a `CapabilityReport` describing host capabilities. When `strict_required`
/// is true, returns an error if required capabilities are missing or probes fail.
pub fn validate_strict_runtime_ext(
    sandbox_backend: SandboxBackend,
    strict_required: bool,
    run_probes: bool,
    cwd: Option<&str>,
) -> Result<CapabilityReport, ShellError> {
    let c_cwd = cwd
        .map(|value| {
            CString::new(value)
                .map_err(|_| ShellError::InvalidArgument("cwd contains null byte".into()))
        })
        .transpose()?;
    let mut report = CapabilityReport::default();
    let rc = unsafe {
        talu_agent_runtime_validate_strict_ext_raw(
            sandbox_backend as c_int,
            strict_required,
            run_probes,
            c_cwd.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
            &mut report,
        )
    };
    if rc != ERROR_CODE_OK {
        return Err(ShellError::from_code(
            rc,
            "strict runtime ext validation failed",
        ));
    }
    Ok(report)
}

/// Execute a shell command and capture its output.
///
/// This does NOT perform safety checks — call `check_command` first
/// if policy enforcement is needed.
pub fn exec(command: &str) -> Result<ExecOutput, ShellError> {
    exec_with_policy(command, None, None)
}

/// Execute a shell command with optional cwd/policy and capture its output.
pub fn exec_with_policy(
    command: &str,
    cwd: Option<&str>,
    policy: Option<&crate::policy::Policy>,
) -> Result<ExecOutput, ShellError> {
    exec_with_policy_runtime(
        command,
        cwd,
        policy,
        AgentRuntimeMode::Host,
        SandboxBackend::LinuxLocal,
    )
}

/// Execute a shell command with optional policy and explicit runtime configuration.
pub fn exec_with_policy_runtime(
    command: &str,
    cwd: Option<&str>,
    policy: Option<&crate::policy::Policy>,
    runtime_mode: AgentRuntimeMode,
    sandbox_backend: SandboxBackend,
) -> Result<ExecOutput, ShellError> {
    let c_cmd = CString::new(command)
        .map_err(|_| ShellError::InvalidArgument("command contains null byte".to_string()))?;
    let c_cwd = if let Some(value) = cwd {
        Some(
            CString::new(value)
                .map_err(|_| ShellError::InvalidArgument("cwd contains null byte".to_string()))?,
        )
    } else {
        None
    };
    let policy_ptr = policy.map(|p| p.as_ptr()).unwrap_or(std::ptr::null_mut());

    let mut stdout_ptr: *const u8 = std::ptr::null();
    let mut stdout_len: usize = 0;
    let mut stderr_ptr: *const u8 = std::ptr::null();
    let mut stderr_len: usize = 0;
    let mut exit_code: i32 = -1;

    // SAFETY: pointers are valid for the duration of this call.
    let rc = unsafe {
        talu_shell_exec_raw(
            c_cmd.as_ptr(),
            c_cwd.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
            policy_ptr,
            runtime_mode as c_int,
            sandbox_backend as c_int,
            &mut stdout_ptr,
            &mut stdout_len,
            &mut stderr_ptr,
            &mut stderr_len,
            &mut exit_code,
        )
    };

    if rc != ERROR_CODE_OK {
        return Err(ShellError::from_code(rc, "shell exec failed"));
    }

    let stdout = if stdout_ptr.is_null() || stdout_len == 0 {
        String::new()
    } else {
        // SAFETY: C API returns a valid buffer of `stdout_len` bytes on success.
        let bytes = unsafe { std::slice::from_raw_parts(stdout_ptr, stdout_len) };
        let s = String::from_utf8_lossy(bytes).to_string();
        // SAFETY: pointer+len originate from shell C API and are freed exactly once here.
        unsafe { talu_shell_free_string_raw(stdout_ptr, stdout_len) };
        s
    };

    let stderr = if stderr_ptr.is_null() || stderr_len == 0 {
        String::new()
    } else {
        // SAFETY: C API returns a valid buffer of `stderr_len` bytes on success.
        let bytes = unsafe { std::slice::from_raw_parts(stderr_ptr, stderr_len) };
        let s = String::from_utf8_lossy(bytes).to_string();
        // SAFETY: pointer+len originate from shell C API and are freed exactly once here.
        unsafe { talu_shell_free_string_raw(stderr_ptr, stderr_len) };
        s
    };

    Ok(ExecOutput {
        stdout,
        stderr,
        exit_code: Some(exit_code),
    })
}

unsafe extern "C" fn stream_callback_bridge<F>(
    ctx: *mut c_void,
    data: *const u8,
    len: usize,
) -> bool
where
    F: FnMut(&[u8]) -> bool,
{
    if ctx.is_null() {
        return true;
    }
    // SAFETY: caller ensures `ctx` points to a valid `F` for the duration of callback execution.
    let callback: &mut F = unsafe { &mut *(ctx as *mut F) };
    let chunk = if data.is_null() || len == 0 {
        &[][..]
    } else {
        // SAFETY: C API passes a valid buffer for callback payload.
        unsafe { std::slice::from_raw_parts(data, len) }
    };
    callback(chunk)
}

/// Execute a shell command and stream output chunks through callbacks.
pub fn exec_streaming<FStdout, FStderr>(
    command: &str,
    cwd: Option<&str>,
    timeout_ms: u64,
    on_stdout: FStdout,
    on_stderr: FStderr,
) -> Result<Option<i32>, ShellError>
where
    FStdout: FnMut(&[u8]) -> bool,
    FStderr: FnMut(&[u8]) -> bool,
{
    exec_streaming_with_policy(command, cwd, timeout_ms, None, on_stdout, on_stderr)
}

/// Execute a shell command with optional policy and stream output chunks.
pub fn exec_streaming_with_policy<FStdout, FStderr>(
    command: &str,
    cwd: Option<&str>,
    timeout_ms: u64,
    policy: Option<&crate::policy::Policy>,
    on_stdout: FStdout,
    on_stderr: FStderr,
) -> Result<Option<i32>, ShellError>
where
    FStdout: FnMut(&[u8]) -> bool,
    FStderr: FnMut(&[u8]) -> bool,
{
    exec_streaming_with_policy_runtime(
        command,
        cwd,
        timeout_ms,
        policy,
        AgentRuntimeMode::Host,
        SandboxBackend::LinuxLocal,
        on_stdout,
        on_stderr,
    )
}

/// Execute a shell command with optional policy/runtime and stream output chunks.
pub fn exec_streaming_with_policy_runtime<FStdout, FStderr>(
    command: &str,
    cwd: Option<&str>,
    timeout_ms: u64,
    policy: Option<&crate::policy::Policy>,
    runtime_mode: AgentRuntimeMode,
    sandbox_backend: SandboxBackend,
    mut on_stdout: FStdout,
    mut on_stderr: FStderr,
) -> Result<Option<i32>, ShellError>
where
    FStdout: FnMut(&[u8]) -> bool,
    FStderr: FnMut(&[u8]) -> bool,
{
    let c_cmd = CString::new(command)
        .map_err(|_| ShellError::InvalidArgument("command contains null byte".to_string()))?;

    let c_cwd = if let Some(value) = cwd {
        Some(
            CString::new(value)
                .map_err(|_| ShellError::InvalidArgument("cwd contains null byte".to_string()))?,
        )
    } else {
        None
    };
    let policy_ptr = policy.map(|p| p.as_ptr()).unwrap_or(std::ptr::null_mut());

    let mut exit_code = -1i32;

    // SAFETY: command/cwd pointers are valid for this call. Callback contexts point
    // to stack closures and remain valid until the FFI call returns.
    let rc = unsafe {
        talu_shell_exec_streaming_raw(
            c_cmd.as_ptr(),
            c_cwd
                .as_ref()
                .map_or(std::ptr::null(), |value| value.as_ptr()),
            policy_ptr,
            runtime_mode as c_int,
            sandbox_backend as c_int,
            timeout_ms,
            Some(stream_callback_bridge::<FStdout>),
            (&mut on_stdout as *mut FStdout).cast::<c_void>(),
            Some(stream_callback_bridge::<FStderr>),
            (&mut on_stderr as *mut FStderr).cast::<c_void>(),
            &mut exit_code,
        )
    };

    if rc != ERROR_CODE_OK {
        return Err(ShellError::from_code(rc, "shell streaming exec failed"));
    }

    Ok(Some(exit_code))
}

/// Check whether a command is allowed by the built-in whitelist.
pub fn check_command(command: &str) -> Result<SafetyCheck, ShellError> {
    let c_cmd = CString::new(command)
        .map_err(|_| ShellError::InvalidArgument("command contains null byte".to_string()))?;

    let mut allowed = false;
    let mut reason_ptr: *const u8 = std::ptr::null();
    let mut reason_len: usize = 0;

    // SAFETY: pointers are valid for the duration of this call.
    let rc = unsafe {
        talu_shell_check_command_raw(
            c_cmd.as_ptr(),
            &mut allowed,
            &mut reason_ptr,
            &mut reason_len,
        )
    };

    if rc != ERROR_CODE_OK {
        return Err(ShellError::from_code(rc, "shell safety check failed"));
    }

    let reason = if reason_ptr.is_null() || reason_len == 0 {
        None
    } else {
        // Reason points to static memory — do NOT free.
        // SAFETY: reason pointer is valid for reason_len bytes.
        let bytes = unsafe { std::slice::from_raw_parts(reason_ptr, reason_len) };
        Some(String::from_utf8_lossy(bytes).to_string())
    };

    Ok(SafetyCheck { allowed, reason })
}

/// Get the default IAM-style policy JSON containing the whitelist.
pub fn default_policy_json() -> Result<String, ShellError> {
    let mut json_ptr: *const u8 = std::ptr::null();
    let mut json_len: usize = 0;

    // SAFETY: output pointers are valid for this call.
    let rc = unsafe { talu_shell_default_policy_json_raw(&mut json_ptr, &mut json_len) };

    if rc != ERROR_CODE_OK {
        return Err(ShellError::from_code(
            rc,
            "failed to build default shell policy",
        ));
    }

    if json_ptr.is_null() || json_len == 0 {
        return Ok(String::new());
    }

    // SAFETY: C API returns a valid buffer of `json_len` bytes on success.
    let bytes = unsafe { std::slice::from_raw_parts(json_ptr, json_len) };
    let s = String::from_utf8_lossy(bytes).to_string();
    // SAFETY: pointer+len originate from shell C API and are freed exactly once here.
    unsafe { talu_shell_free_string_raw(json_ptr, json_len) };
    Ok(s)
}

/// Normalize a command for policy evaluation (absolute path -> basename).
pub fn normalize_command(command: &str) -> Result<String, ShellError> {
    let c_cmd = CString::new(command)
        .map_err(|_| ShellError::InvalidArgument("command contains null byte".to_string()))?;

    let mut out_ptr: *const u8 = std::ptr::null();
    let mut out_len: usize = 0;

    // SAFETY: pointers are valid for the duration of this call.
    let rc =
        unsafe { talu_shell_normalize_command_raw(c_cmd.as_ptr(), &mut out_ptr, &mut out_len) };

    if rc != ERROR_CODE_OK {
        return Err(ShellError::from_code(rc, "failed to normalize command"));
    }

    if out_ptr.is_null() || out_len == 0 {
        return Ok(String::new());
    }

    // SAFETY: C API returns a valid buffer of `out_len` bytes on success.
    let bytes = unsafe { std::slice::from_raw_parts(out_ptr, out_len) };
    let s = String::from_utf8_lossy(bytes).to_string();
    // SAFETY: pointer+len originate from shell C API and are freed exactly once here.
    unsafe { talu_shell_free_string_raw(out_ptr, out_len) };
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_command_allows_whitelisted() {
        let check = check_command("ls -la").unwrap();
        assert!(check.allowed);
        assert!(check.reason.is_none());
    }

    #[test]
    fn test_check_command_denies_non_whitelisted() {
        let check = check_command("rm -rf /").unwrap();
        assert!(!check.allowed);
        assert!(check.reason.is_some());
    }

    #[test]
    fn test_exec_runs_command() {
        let output = exec("echo hello").unwrap();
        assert_eq!(output.stdout, "hello\n");
        assert_eq!(output.exit_code, Some(0));
    }

    #[test]
    fn test_exec_captures_stderr() {
        let output = exec("echo oops >&2").unwrap();
        assert_eq!(output.stderr, "oops\n");
        assert_eq!(output.exit_code, Some(0));
    }

    #[test]
    fn test_exec_streaming_callbacks_receive_output() {
        let mut saw_stdout = false;
        let mut saw_stderr = false;

        let exit = exec_streaming(
            "echo out && echo err >&2",
            None,
            30_000,
            |chunk| {
                if chunk.windows(3).any(|w| w == b"out") {
                    saw_stdout = true;
                }
                true
            },
            |chunk| {
                if chunk.windows(3).any(|w| w == b"err") {
                    saw_stderr = true;
                }
                true
            },
        )
        .unwrap();

        assert_eq!(exit, Some(0));
        assert!(saw_stdout);
        assert!(saw_stderr);
    }

    #[test]
    fn test_default_policy_json() {
        let json = default_policy_json().unwrap();
        assert!(json.starts_with("{\"default\":\"deny\""));
        assert!(json.contains("\"effect\":\"allow\",\"action\":\"ls\""));
    }

    #[test]
    fn test_normalize_command_strips_path() {
        let normalized = normalize_command("/usr/bin/git status").unwrap();
        assert_eq!(normalized, "git status");
    }

    #[test]
    fn test_normalize_command_passes_through() {
        let normalized = normalize_command("git status").unwrap();
        assert_eq!(normalized, "git status");
    }

    #[test]
    fn test_shell_session_roundtrip() {
        let mut shell = ShellSession::open(80, 24, None).unwrap();

        shell.write(b"echo hello\n").unwrap();
        shell.write(b"exit\n").unwrap();

        let mut buf = [0u8; 4096];
        let mut output = Vec::new();
        let mut guard = 0usize;

        while shell.is_alive().unwrap_or(false) && guard < 50_000 {
            let n = shell.read(&mut buf).unwrap();
            if n > 0 {
                output.extend_from_slice(&buf[..n]);
            }
            guard += 1;
        }

        loop {
            let n = shell.read(&mut buf).unwrap();
            if n == 0 {
                break;
            }
            output.extend_from_slice(&buf[..n]);
        }

        let text = String::from_utf8_lossy(&output);
        assert!(text.contains("hello"));

        let scrollback = shell.scrollback().unwrap();
        let scrollback_text = String::from_utf8_lossy(&scrollback);
        assert!(scrollback_text.contains("hello"));
    }

    #[test]
    fn test_validate_strict_runtime_rejects_invalid_runtime_policy() {
        if !cfg!(target_os = "linux") {
            return;
        }
        let policy = crate::policy::Policy::from_json(
            r#"{
                "default":"deny",
                "statements":[
                    {"effect":"allow","action":"tool.exec","command":"echo *"},
                    {"effect":"allow","action":"tool.fs.write","resource":"src/**"},
                    {"effect":"deny","action":"tool.fs.write","resource":"src/private/**"}
                ]
            }"#,
        )
        .unwrap();

        let err = validate_strict_runtime(Some(&policy), None, SandboxBackend::LinuxLocal)
            .expect_err("expected strict setup failure");
        assert!(matches!(err, ShellError::StrictSetupFailed(_)));
    }
}
