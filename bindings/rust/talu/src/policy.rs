//! Agent runtime policy wrapper backed by the `talu_agent_policy_*` C API.

use crate::error::error_from_last_or;
use crate::Result;
use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};

unsafe extern "C" {
    #[link_name = "talu_agent_policy_create"]
    fn talu_agent_policy_create_raw(
        json: *const u8,
        len: usize,
        out_policy: *mut *mut c_void,
    ) -> c_int;

    #[link_name = "talu_agent_policy_free"]
    fn talu_agent_policy_free_raw(policy: *mut c_void);

    #[link_name = "talu_agent_policy_prepare_runtime"]
    fn talu_agent_policy_prepare_runtime_raw(policy: *mut c_void, cwd: *const c_char) -> c_int;

    #[link_name = "talu_agent_policy_check_action"]
    fn talu_agent_policy_check_action_raw(
        policy: *mut c_void,
        action: *const c_char,
        command: *const c_char,
        cwd: *const c_char,
        resource: *const c_char,
        timeout_ms: u64,
        out_allowed: *mut bool,
        out_reason: *mut *const u8,
        out_reason_len: *mut usize,
    ) -> c_int;

    #[link_name = "talu_agent_policy_check_file"]
    fn talu_agent_policy_check_file_raw(
        policy: *mut c_void,
        action: *const c_char,
        resource: *const c_char,
        is_dir: bool,
        out_allowed: *mut bool,
    ) -> c_int;

    #[link_name = "talu_agent_policy_check_process_detailed"]
    fn talu_agent_policy_check_process_detailed_raw(
        policy: *mut c_void,
        action: *const c_char,
        command: *const c_char,
        cwd: *const c_char,
        out_allowed: *mut bool,
        out_deny_reason: *mut c_int,
    ) -> c_int;

    #[link_name = "talu_agent_policy_validate_strict_emulation"]
    fn talu_agent_policy_validate_strict_emulation_raw(policy: *mut c_void) -> c_int;

    #[link_name = "talu_agent_policy_strict_emulation_decisions"]
    fn talu_agent_policy_strict_emulation_decisions_raw(
        policy: *mut c_void,
        cwd: *const c_char,
        out_deny_descendant_exec: *mut bool,
        out_deny_write: *mut bool,
        out_allow_python_exec: *mut bool,
    ) -> c_int;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ProcessDenyReason {
    Action = 1,
    Cwd = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessPolicyDecision {
    pub allowed: bool,
    pub deny_reason: Option<ProcessDenyReason>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StrictEmulationDecisions {
    pub deny_descendant_exec: bool,
    pub deny_write: bool,
    pub allow_python_exec: bool,
}

pub struct Policy {
    ptr: *mut c_void,
}

unsafe impl Send for Policy {}
unsafe impl Sync for Policy {}

impl Policy {
    pub fn from_json(json: &str) -> Result<Self> {
        let mut out: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { talu_agent_policy_create_raw(json.as_ptr(), json.len(), &mut out) };
        if rc != 0 || out.is_null() {
            return Err(error_from_last_or("Failed to parse agent policy JSON"));
        }
        Ok(Self { ptr: out })
    }

    pub(crate) fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    pub fn check_action(
        &self,
        action: &str,
        command: Option<&str>,
        cwd: Option<&str>,
        resource: Option<&str>,
        timeout_ms: u64,
    ) -> Result<bool> {
        let c_action = CString::new(action)?;
        let c_command = command.map(CString::new).transpose()?;
        let c_cwd = cwd.map(CString::new).transpose()?;
        let c_resource = resource.map(CString::new).transpose()?;

        let mut allowed = false;
        let mut reason_ptr: *const u8 = std::ptr::null();
        let mut reason_len = 0usize;
        let rc = unsafe {
            talu_agent_policy_check_action_raw(
                self.ptr,
                c_action.as_ptr(),
                c_command.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
                c_cwd.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
                c_resource.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
                timeout_ms,
                &mut allowed,
                &mut reason_ptr,
                &mut reason_len,
            )
        };
        if rc != 0 {
            return Err(error_from_last_or("Failed to evaluate agent policy action"));
        }
        Ok(allowed)
    }

    pub fn check_process(
        &self,
        action: &str,
        command: Option<&str>,
        cwd: Option<&str>,
    ) -> Result<bool> {
        Ok(self.check_process_detailed(action, command, cwd)?.allowed)
    }

    pub fn check_process_detailed(
        &self,
        action: &str,
        command: Option<&str>,
        cwd: Option<&str>,
    ) -> Result<ProcessPolicyDecision> {
        let c_action = CString::new(action)?;
        let c_command = command.map(CString::new).transpose()?;
        let c_cwd = cwd.map(CString::new).transpose()?;

        let mut allowed = false;
        let mut deny_reason_code = 0i32;
        let rc = unsafe {
            talu_agent_policy_check_process_detailed_raw(
                self.ptr,
                c_action.as_ptr(),
                c_command.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
                c_cwd.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
                &mut allowed,
                &mut deny_reason_code,
            )
        };
        if rc != 0 {
            return Err(error_from_last_or("Failed to evaluate process policy"));
        }

        let deny_reason = match deny_reason_code {
            0 => None,
            1 => Some(ProcessDenyReason::Action),
            2 => Some(ProcessDenyReason::Cwd),
            _ => return Err(error_from_last_or("Failed to decode process deny reason")),
        };

        Ok(ProcessPolicyDecision {
            allowed,
            deny_reason: if allowed { None } else { deny_reason },
        })
    }

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

    pub fn check_file(&self, action: &str, resource: &str, is_dir: bool) -> Result<bool> {
        let c_action = CString::new(action)?;
        let c_resource = CString::new(resource)?;
        let mut allowed = false;
        let rc = unsafe {
            talu_agent_policy_check_file_raw(
                self.ptr,
                c_action.as_ptr(),
                c_resource.as_ptr(),
                is_dir,
                &mut allowed,
            )
        };
        if rc != 0 {
            return Err(error_from_last_or("Failed to evaluate file policy"));
        }
        Ok(allowed)
    }

    pub fn validate_strict_emulation(&self) -> Result<()> {
        let rc = unsafe { talu_agent_policy_validate_strict_emulation_raw(self.ptr) };
        if rc != 0 {
            return Err(error_from_last_or(
                "Strict runtime emulation is not representable by this policy",
            ));
        }
        Ok(())
    }

    pub fn strict_emulation_decisions(
        &self,
        cwd: Option<&str>,
    ) -> Result<StrictEmulationDecisions> {
        let c_cwd = cwd.map(CString::new).transpose()?;
        let mut deny_descendant_exec = false;
        let mut deny_write = false;
        let mut allow_python_exec = true;
        let rc = unsafe {
            talu_agent_policy_strict_emulation_decisions_raw(
                self.ptr,
                c_cwd.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
                &mut deny_descendant_exec,
                &mut deny_write,
                &mut allow_python_exec,
            )
        };
        if rc != 0 {
            return Err(error_from_last_or(
                "Failed to derive strict runtime policy decisions",
            ));
        }
        Ok(StrictEmulationDecisions {
            deny_descendant_exec,
            deny_write,
            allow_python_exec,
        })
    }
}

impl Drop for Policy {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { talu_agent_policy_free_raw(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Policy, ProcessDenyReason};

    #[test]
    fn test_parse_and_drop_policy() {
        let json = r#"{"default":"deny","statements":[]}"#;
        let _policy = Policy::from_json(json).unwrap();
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

    #[test]
    fn test_check_process_detailed_reports_cwd_reason() {
        let json = r#"{
            "default":"deny",
            "statements":[
                {"effect":"allow","action":"tool.exec","command":"rg *"},
                {"effect":"deny","action":"tool.exec","command":"rg *","cwd":"tmp/**"}
            ]
        }"#;
        let policy = Policy::from_json(json).unwrap();
        let decision = policy
            .check_process_detailed("tool.exec", Some("rg foo"), Some("tmp"))
            .unwrap();
        assert!(!decision.allowed);
        assert_eq!(decision.deny_reason, Some(ProcessDenyReason::Cwd));
    }

    #[test]
    fn test_validate_strict_emulation_rejects_mixed_file_rules() {
        let json = r#"{
            "default":"deny",
            "statements":[
                {"effect":"allow","action":"tool.fs.write","resource":"src/**"},
                {"effect":"deny","action":"tool.fs.write","resource":"src/private/**"}
            ]
        }"#;
        let policy = Policy::from_json(json).unwrap();
        assert!(policy.validate_strict_emulation().is_err());
    }

    #[test]
    fn test_strict_emulation_decisions_default_deny_is_conservative() {
        let json = r#"{"default":"deny","statements":[]}"#;
        let policy = Policy::from_json(json).unwrap();
        let decisions = policy.strict_emulation_decisions(None).unwrap();
        assert!(decisions.deny_descendant_exec);
        assert!(decisions.deny_write);
        assert!(!decisions.allow_python_exec);
    }
}
