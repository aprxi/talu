//! IAM-style policy engine for tool call filtering.
//!
//! Provides a safe wrapper around the core policy C API. Policies are parsed
//! from JSON and attached to a [`ChatHandle`](crate::ChatHandle) to filter
//! tool calls before execution.
//!
//! # Example
//!
//! ```no_run
//! use talu::policy::{Policy, Effect, Mode};
//!
//! let json = r#"{"default":"deny","statements":[{"effect":"allow","action":"ls *"}]}"#;
//! let policy = Policy::from_json(json)?;
//!
//! assert_eq!(policy.evaluate("ls -la"), Effect::Allow);
//! assert_eq!(policy.evaluate("rm -rf /"), Effect::Deny);
//! assert_eq!(policy.mode(), Mode::Enforce);
//! # Ok::<(), talu::Error>(())
//! ```

use crate::error::error_from_last_or;
use crate::Result;
use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};

unsafe extern "C" {
    #[link_name = "talu_agent_policy_check_process"]
    fn talu_agent_policy_check_process_raw(
        policy: *mut c_void,
        action: *const c_char,
        command: *const c_char,
        cwd: *const c_char,
        out_allowed: *mut c_void,
    ) -> c_int;

    #[link_name = "talu_agent_policy_prepare_runtime"]
    fn talu_agent_policy_prepare_runtime_raw(policy: *mut c_void, cwd: *const c_char) -> c_int;
}

/// IAM-style evaluation result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Effect {
    /// Action is allowed.
    Allow = 0,
    /// Action is denied.
    Deny = 1,
}

/// Policy enforcement mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Mode {
    /// Denied actions are blocked.
    Enforce = 0,
    /// Denied actions are logged but allowed through.
    Audit = 1,
}

/// RAII wrapper for a policy handle.
///
/// Policies are immutable after creation. To change the active policy on a
/// chat session, create a new `Policy` and call
/// [`ChatHandle::set_policy`](crate::ChatHandle::set_policy).
pub struct Policy {
    ptr: *mut c_void,
}

// SAFETY: Policy points to immutable core-owned policy data after creation.
// It can be shared across threads as read-only state.
unsafe impl Send for Policy {}
unsafe impl Sync for Policy {}

impl Policy {
    /// Parse a policy from a JSON string.
    ///
    /// The JSON must follow the IAM-style schema:
    /// ```json
    /// {
    ///   "default": "deny",
    ///   "mode": "enforce",
    ///   "statements": [
    ///     {"effect": "allow", "action": "ls *"},
    ///     {"effect": "deny",  "action": "rm *"}
    ///   ]
    /// }
    /// ```
    ///
    /// Fields:
    /// - `default`: `"allow"` or `"deny"` (required)
    /// - `mode`: `"enforce"` or `"audit"` (optional, defaults to `"enforce"`)
    /// - `statements`: array of `{effect, action}` rules (required)
    ///
    /// Evaluation follows AWS IAM semantics: explicit deny always wins.
    pub fn from_json(json: &str) -> Result<Self> {
        let mut out: *mut c_void = std::ptr::null_mut();
        let rc = unsafe {
            talu_sys::talu_agent_policy_create(
                json.as_ptr(),
                json.len(),
                &mut out as *mut _ as *mut c_void,
            )
        };
        if rc != 0 || out.is_null() {
            return Err(error_from_last_or("Failed to parse policy JSON"));
        }
        Ok(Self { ptr: out })
    }

    /// Evaluate an action string against this policy.
    ///
    /// Returns [`Effect::Allow`] or [`Effect::Deny`] based on IAM-style
    /// evaluation: explicit deny wins, then explicit allow, then the default.
    pub fn evaluate(&self, action: &str) -> Effect {
        let result =
            unsafe { talu_sys::talu_policy_evaluate(self.ptr, action.as_ptr(), action.len()) };
        match result {
            0 => Effect::Allow,
            _ => Effect::Deny,
        }
    }

    /// Returns the enforcement mode of this policy.
    pub fn mode(&self) -> Mode {
        let result = unsafe { talu_sys::talu_policy_get_mode(self.ptr) };
        match result {
            0 => Mode::Enforce,
            _ => Mode::Audit,
        }
    }

    /// Returns the raw pointer for FFI use.
    pub(crate) fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Evaluate a process-style action (`tool.exec` / `tool.shell` /
    /// `tool.process`) with optional command and cwd filters.
    pub fn check_process(
        &self,
        action: &str,
        command: Option<&str>,
        cwd: Option<&str>,
    ) -> Result<bool> {
        let c_action = CString::new(action)?;
        let c_command = command.map(CString::new).transpose()?;
        let c_cwd = cwd.map(CString::new).transpose()?;

        let mut allowed = false;
        let rc = unsafe {
            talu_agent_policy_check_process_raw(
                self.ptr,
                c_action.as_ptr(),
                c_command.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
                c_cwd.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
                (&mut allowed as *mut bool).cast(),
            )
        };
        if rc != 0 {
            return Err(error_from_last_or("Failed to evaluate process policy"));
        }
        Ok(allowed)
    }

    /// Pre-compile strict runtime profiles for this policy and optional cwd.
    ///
    /// This is intended for server startup preparation so strict mode can fail
    /// fast before handling requests.
    pub fn prepare_runtime(&self, cwd: Option<&str>) -> Result<()> {
        let c_cwd = cwd.map(CString::new).transpose()?;
        let rc = unsafe {
            talu_agent_policy_prepare_runtime_raw(
                self.ptr,
                c_cwd.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
            )
        };
        if rc != 0 {
            return Err(error_from_last_or(
                "Failed to precompile strict runtime policy",
            ));
        }
        Ok(())
    }
}

impl Drop for Policy {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: ptr was obtained from talu_agent_policy_create and not yet freed.
            unsafe { talu_sys::talu_agent_policy_free(self.ptr) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_and_evaluate() {
        let json = r#"{"default":"deny","statements":[{"effect":"allow","action":"ls *"}]}"#;
        let policy = Policy::from_json(json).unwrap();
        assert_eq!(policy.evaluate("ls -la"), Effect::Allow);
        assert_eq!(policy.evaluate("rm -rf /"), Effect::Deny);
    }

    #[test]
    fn test_explicit_deny_wins() {
        let json = r#"{
            "default": "allow",
            "statements": [
                {"effect": "allow", "action": "git *"},
                {"effect": "deny",  "action": "git push *"}
            ]
        }"#;
        let policy = Policy::from_json(json).unwrap();
        assert_eq!(policy.evaluate("git show HEAD"), Effect::Allow);
        assert_eq!(policy.evaluate("git push origin main"), Effect::Deny);
    }

    #[test]
    fn test_mode_default_enforce() {
        let json = r#"{"default":"deny","statements":[]}"#;
        let policy = Policy::from_json(json).unwrap();
        assert_eq!(policy.mode(), Mode::Enforce);
    }

    #[test]
    fn test_mode_audit() {
        let json = r#"{"default":"deny","mode":"audit","statements":[]}"#;
        let policy = Policy::from_json(json).unwrap();
        assert_eq!(policy.mode(), Mode::Audit);
    }

    #[test]
    fn test_invalid_json_returns_error() {
        let result = Policy::from_json("not valid json");
        assert!(result.is_err());
    }

    #[test]
    fn test_drop_is_safe() {
        let json = r#"{"default":"deny","statements":[]}"#;
        let _policy = Policy::from_json(json).unwrap();
        // policy dropped here — no double-free, no leak
    }

    #[test]
    fn test_check_process_applies_command_pattern() {
        let json = r#"{
            "default":"allow",
            "statements":[
                {"effect":"deny","action":"tool.exec","command":"git *"}
            ]
        }"#;
        let policy = Policy::from_json(json).unwrap();
        assert!(!policy
            .check_process("tool.exec", Some("git status"), None)
            .unwrap());
        assert!(policy.check_process("tool.exec", Some("ls"), None).unwrap());
    }
}
