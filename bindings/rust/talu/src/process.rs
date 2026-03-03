//! Safe wrappers for process session C APIs (`talu_process_*`).

use std::ffi::{c_void, CString};
use std::os::raw::{c_char, c_int};

const ERROR_CODE_OK: i32 = 0;
const ERROR_CODE_SHELL_COMMAND_DENIED: i32 = 800;
const ERROR_CODE_SHELL_EXEC_FAILED: i32 = 801;
const ERROR_CODE_SHELL_SESSION_CLOSED: i32 = 802;
const ERROR_CODE_SHELL_PROCESS_EXITED: i32 = 804;
const ERROR_CODE_POLICY_DENIED_EXEC: i32 = 808;
const ERROR_CODE_POLICY_DENIED_CWD: i32 = 809;
const ERROR_CODE_POLICY_STRICT_UNAVAILABLE: i32 = 811;
const ERROR_CODE_POLICY_STRICT_SETUP_FAILED: i32 = 812;
const ERROR_CODE_INVALID_ARGUMENT: i32 = 901;
const ERROR_CODE_INVALID_HANDLE: i32 = 902;

unsafe extern "C" {
    #[link_name = "talu_process_open"]
    fn talu_process_open_raw(
        command: *const c_char,
        cwd: *const c_char,
        policy: *mut c_void,
        runtime_mode: c_int,
        sandbox_backend: c_int,
        out_process: *mut *mut c_void,
    ) -> c_int;
    #[link_name = "talu_process_close"]
    fn talu_process_close_raw(process_handle: *mut c_void);
    #[link_name = "talu_process_write"]
    fn talu_process_write_raw(process_handle: *mut c_void, data: *const u8, len: usize) -> c_int;
    #[link_name = "talu_process_read"]
    fn talu_process_read_raw(
        process_handle: *mut c_void,
        buf: *mut u8,
        buf_len: usize,
        out_read: *mut usize,
    ) -> c_int;
    #[link_name = "talu_process_signal"]
    fn talu_process_signal_raw(process_handle: *mut c_void, sig: u8) -> c_int;
    #[link_name = "talu_process_alive"]
    fn talu_process_alive_raw(process_handle: *mut c_void) -> bool;
    #[link_name = "talu_process_exit_code"]
    fn talu_process_exit_code_raw(
        process_handle: *mut c_void,
        out_code: *mut i32,
        out_has_code: *mut bool,
    ) -> c_int;
}

/// Errors from process session operations.
#[derive(Debug)]
pub enum ProcessError {
    CommandDenied(String),
    PolicyDeniedExec(String),
    PolicyDeniedCwd(String),
    StrictUnavailable(String),
    StrictSetupFailed(String),
    ExecFailed(String),
    SessionClosed(String),
    ProcessExited(String),
    InvalidArgument(String),
    InvalidHandle(String),
    Internal(i32),
}

impl ProcessError {
    fn from_code(code: i32, fallback: &str) -> Self {
        let detail = crate::error::last_error_message().unwrap_or_else(|| fallback.to_string());
        match code {
            ERROR_CODE_SHELL_COMMAND_DENIED => Self::CommandDenied(detail),
            ERROR_CODE_POLICY_DENIED_EXEC => Self::PolicyDeniedExec(detail),
            ERROR_CODE_POLICY_DENIED_CWD => Self::PolicyDeniedCwd(detail),
            ERROR_CODE_POLICY_STRICT_UNAVAILABLE => Self::StrictUnavailable(detail),
            ERROR_CODE_POLICY_STRICT_SETUP_FAILED => Self::StrictSetupFailed(detail),
            ERROR_CODE_SHELL_EXEC_FAILED => Self::ExecFailed(detail),
            ERROR_CODE_SHELL_SESSION_CLOSED => Self::SessionClosed(detail),
            ERROR_CODE_SHELL_PROCESS_EXITED => Self::ProcessExited(detail),
            ERROR_CODE_INVALID_ARGUMENT => Self::InvalidArgument(detail),
            ERROR_CODE_INVALID_HANDLE => Self::InvalidHandle(detail),
            _ => Self::Internal(code),
        }
    }
}

impl std::fmt::Display for ProcessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProcessError::CommandDenied(s) => write!(f, "command denied: {}", s),
            ProcessError::PolicyDeniedExec(s) => write!(f, "policy denied process exec: {}", s),
            ProcessError::PolicyDeniedCwd(s) => write!(f, "policy denied cwd: {}", s),
            ProcessError::StrictUnavailable(s) => write!(f, "strict runtime unavailable: {}", s),
            ProcessError::StrictSetupFailed(s) => write!(f, "strict runtime setup failed: {}", s),
            ProcessError::ExecFailed(s) => write!(f, "process spawn failed: {}", s),
            ProcessError::SessionClosed(s) => write!(f, "session closed: {}", s),
            ProcessError::ProcessExited(s) => write!(f, "process exited: {}", s),
            ProcessError::InvalidArgument(s) => write!(f, "invalid argument: {}", s),
            ProcessError::InvalidHandle(s) => write!(f, "invalid process handle: {}", s),
            ProcessError::Internal(code) => write!(f, "internal error (code {})", code),
        }
    }
}

impl std::error::Error for ProcessError {}

/// RAII process session with piped stdin/stdout/stderr.
#[derive(Debug)]
pub struct ProcessSession {
    handle: *mut c_void,
}

// SAFETY: `ProcessSession` contains only an opaque native handle. Access is
// serialized by requiring `&mut self` for all mutating operations.
unsafe impl Send for ProcessSession {}

impl ProcessSession {
    /// Open a new process session with `/bin/sh -c <command>`.
    pub fn open(command: &str, cwd: Option<&str>) -> Result<Self, ProcessError> {
        Self::open_with_policy(command, cwd, None)
    }

    /// Open a new process session with optional agent policy.
    pub fn open_with_policy(
        command: &str,
        cwd: Option<&str>,
        policy: Option<&crate::policy::Policy>,
    ) -> Result<Self, ProcessError> {
        Self::open_with_policy_runtime(
            command,
            cwd,
            policy,
            crate::shell::AgentRuntimeMode::Host,
            crate::shell::SandboxBackend::LinuxLocal,
        )
    }

    /// Open a new process session with optional policy and explicit runtime configuration.
    pub fn open_with_policy_runtime(
        command: &str,
        cwd: Option<&str>,
        policy: Option<&crate::policy::Policy>,
        runtime_mode: crate::shell::AgentRuntimeMode,
        sandbox_backend: crate::shell::SandboxBackend,
    ) -> Result<Self, ProcessError> {
        let c_command = CString::new(command)
            .map_err(|_| ProcessError::InvalidArgument("command contains null byte".into()))?;
        let c_cwd = if let Some(value) = cwd {
            Some(
                CString::new(value)
                    .map_err(|_| ProcessError::InvalidArgument("cwd contains null byte".into()))?,
            )
        } else {
            None
        };
        let policy_ptr = policy.map(|p| p.as_ptr()).unwrap_or(std::ptr::null_mut());

        let mut handle: *mut c_void = std::ptr::null_mut();
        // SAFETY: pointers are valid for the duration of this call.
        let rc = unsafe {
            talu_process_open_raw(
                c_command.as_ptr(),
                c_cwd.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
                policy_ptr,
                runtime_mode as c_int,
                sandbox_backend as c_int,
                &mut handle,
            )
        };
        if rc != ERROR_CODE_OK {
            return Err(ProcessError::from_code(
                rc,
                "failed to open process session",
            ));
        }
        if handle.is_null() {
            return Err(ProcessError::Internal(ERROR_CODE_SHELL_EXEC_FAILED));
        }

        Ok(Self { handle })
    }

    /// Write bytes to process stdin.
    pub fn write(&mut self, data: &[u8]) -> Result<(), ProcessError> {
        let ptr = if data.is_empty() {
            std::ptr::null()
        } else {
            data.as_ptr()
        };
        // SAFETY: handle and data pointer are valid for this call.
        let rc = unsafe { talu_process_write_raw(self.handle, ptr, data.len()) };
        if rc != ERROR_CODE_OK {
            return Err(ProcessError::from_code(rc, "failed to write process stdin"));
        }
        Ok(())
    }

    /// Read bytes from process stdout/stderr into `buf`.
    pub fn read(&mut self, buf: &mut [u8]) -> Result<usize, ProcessError> {
        let mut read = 0usize;
        let ptr = if buf.is_empty() {
            std::ptr::null_mut()
        } else {
            buf.as_mut_ptr()
        };
        // SAFETY: handle and output pointers are valid for this call.
        let rc = unsafe { talu_process_read_raw(self.handle, ptr, buf.len(), &mut read) };
        if rc != ERROR_CODE_OK {
            return Err(ProcessError::from_code(rc, "failed to read process output"));
        }
        Ok(read)
    }

    /// Send POSIX signal to process.
    pub fn signal(&mut self, sig: u8) -> Result<(), ProcessError> {
        // SAFETY: handle is valid for this call.
        let rc = unsafe { talu_process_signal_raw(self.handle, sig) };
        if rc != ERROR_CODE_OK {
            return Err(ProcessError::from_code(rc, "failed to signal process"));
        }
        Ok(())
    }

    /// Return whether the process is alive.
    pub fn is_alive(&mut self) -> Result<bool, ProcessError> {
        // SAFETY: clear thread-local C error before probing boolean API.
        unsafe { talu_sys::talu_clear_error() };
        // SAFETY: handle is valid for this call.
        let alive = unsafe { talu_process_alive_raw(self.handle) };
        if !alive {
            if let Some(msg) = crate::error::last_error_message() {
                if msg.contains("session closed") {
                    return Err(ProcessError::SessionClosed(msg));
                }
                return Err(ProcessError::InvalidHandle(msg));
            }
        }
        Ok(alive)
    }

    /// Return process exit code when available.
    pub fn exit_code(&mut self) -> Result<Option<i32>, ProcessError> {
        let mut code = 0i32;
        let mut has_code = false;
        // SAFETY: handle and output pointers are valid for this call.
        let rc = unsafe { talu_process_exit_code_raw(self.handle, &mut code, &mut has_code) };
        if rc != ERROR_CODE_OK {
            return Err(ProcessError::from_code(
                rc,
                "failed to read process exit code",
            ));
        }
        Ok(if has_code { Some(code) } else { None })
    }

    /// Close process session. Safe to call multiple times.
    pub fn close(&mut self) {
        if self.handle.is_null() {
            return;
        }
        // SAFETY: handle is owned by this ProcessSession and is closed at most once.
        unsafe { talu_process_close_raw(self.handle) };
        self.handle = std::ptr::null_mut();
    }
}

impl Drop for ProcessSession {
    fn drop(&mut self) {
        self.close();
    }
}
