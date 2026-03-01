//! Safe wrappers for shell execution and command safety C APIs (`talu_shell_*`).

use std::ffi::CString;
use std::os::raw::{c_char, c_int};

const ERROR_CODE_OK: i32 = 0;

unsafe extern "C" {
    #[link_name = "talu_shell_exec"]
    fn talu_shell_exec_raw(
        command: *const c_char,
        out_stdout: *mut *const u8,
        out_stdout_len: *mut usize,
        out_stderr: *mut *const u8,
        out_stderr_len: *mut usize,
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

    #[link_name = "talu_shell_free_string"]
    fn talu_shell_free_string_raw(ptr: *const u8, len: usize);
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
    ExecFailed(String),
    InvalidArgument(String),
    Internal(i32),
}

impl std::fmt::Display for ShellError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShellError::CommandDenied(s) => write!(f, "command denied: {}", s),
            ShellError::ExecFailed(s) => write!(f, "execution failed: {}", s),
            ShellError::InvalidArgument(s) => write!(f, "invalid argument: {}", s),
            ShellError::Internal(code) => write!(f, "internal error (code {})", code),
        }
    }
}

impl std::error::Error for ShellError {}

/// Execute a shell command and capture its output.
///
/// This does NOT perform safety checks — call `check_command` first
/// if policy enforcement is needed.
pub fn exec(command: &str) -> Result<ExecOutput, ShellError> {
    let c_cmd = CString::new(command).map_err(|_| {
        ShellError::InvalidArgument("command contains null byte".to_string())
    })?;

    let mut stdout_ptr: *const u8 = std::ptr::null();
    let mut stdout_len: usize = 0;
    let mut stderr_ptr: *const u8 = std::ptr::null();
    let mut stderr_len: usize = 0;
    let mut exit_code: i32 = -1;

    let rc = unsafe {
        talu_shell_exec_raw(
            c_cmd.as_ptr(),
            &mut stdout_ptr,
            &mut stdout_len,
            &mut stderr_ptr,
            &mut stderr_len,
            &mut exit_code,
        )
    };

    if rc != ERROR_CODE_OK {
        return Err(ShellError::ExecFailed(format!("error code {}", rc)));
    }

    let stdout = if stdout_ptr.is_null() || stdout_len == 0 {
        String::new()
    } else {
        let bytes = unsafe { std::slice::from_raw_parts(stdout_ptr, stdout_len) };
        let s = String::from_utf8_lossy(bytes).to_string();
        unsafe { talu_shell_free_string_raw(stdout_ptr, stdout_len) };
        s
    };

    let stderr = if stderr_ptr.is_null() || stderr_len == 0 {
        String::new()
    } else {
        let bytes = unsafe { std::slice::from_raw_parts(stderr_ptr, stderr_len) };
        let s = String::from_utf8_lossy(bytes).to_string();
        unsafe { talu_shell_free_string_raw(stderr_ptr, stderr_len) };
        s
    };

    Ok(ExecOutput {
        stdout,
        stderr,
        exit_code: Some(exit_code),
    })
}

/// Check whether a command is allowed by the built-in whitelist.
pub fn check_command(command: &str) -> Result<SafetyCheck, ShellError> {
    let c_cmd = CString::new(command).map_err(|_| {
        ShellError::InvalidArgument("command contains null byte".to_string())
    })?;

    let mut allowed = false;
    let mut reason_ptr: *const u8 = std::ptr::null();
    let mut reason_len: usize = 0;

    let rc = unsafe {
        talu_shell_check_command_raw(
            c_cmd.as_ptr(),
            &mut allowed,
            &mut reason_ptr,
            &mut reason_len,
        )
    };

    if rc != ERROR_CODE_OK {
        return Err(ShellError::Internal(rc));
    }

    let reason = if reason_ptr.is_null() || reason_len == 0 {
        None
    } else {
        // Reason points to static memory — do NOT free.
        let bytes = unsafe { std::slice::from_raw_parts(reason_ptr, reason_len) };
        Some(String::from_utf8_lossy(bytes).to_string())
    };

    Ok(SafetyCheck { allowed, reason })
}

/// Get the default IAM-style policy JSON containing the whitelist.
pub fn default_policy_json() -> Result<String, ShellError> {
    let mut json_ptr: *const u8 = std::ptr::null();
    let mut json_len: usize = 0;

    let rc = unsafe {
        talu_shell_default_policy_json_raw(&mut json_ptr, &mut json_len)
    };

    if rc != ERROR_CODE_OK {
        return Err(ShellError::Internal(rc));
    }

    if json_ptr.is_null() || json_len == 0 {
        return Ok(String::new());
    }

    let bytes = unsafe { std::slice::from_raw_parts(json_ptr, json_len) };
    let s = String::from_utf8_lossy(bytes).to_string();
    unsafe { talu_shell_free_string_raw(json_ptr, json_len) };
    Ok(s)
}

/// Normalize a command for policy evaluation (absolute path → basename).
pub fn normalize_command(command: &str) -> Result<String, ShellError> {
    let c_cmd = CString::new(command).map_err(|_| {
        ShellError::InvalidArgument("command contains null byte".to_string())
    })?;

    let mut out_ptr: *const u8 = std::ptr::null();
    let mut out_len: usize = 0;

    let rc = unsafe {
        talu_shell_normalize_command_raw(c_cmd.as_ptr(), &mut out_ptr, &mut out_len)
    };

    if rc != ERROR_CODE_OK {
        return Err(ShellError::Internal(rc));
    }

    if out_ptr.is_null() || out_len == 0 {
        return Ok(String::new());
    }

    let bytes = unsafe { std::slice::from_raw_parts(out_ptr, out_len) };
    let s = String::from_utf8_lossy(bytes).to_string();
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
}
