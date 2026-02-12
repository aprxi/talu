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
use std::os::raw::c_void;

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
            talu_sys::talu_policy_create(
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
}

impl Drop for Policy {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: ptr was obtained from talu_policy_create and not yet freed.
            unsafe { talu_sys::talu_policy_free(self.ptr) };
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
}
