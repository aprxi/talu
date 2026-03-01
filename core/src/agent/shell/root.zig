//! Shell execution and command safety.
//!
//! Provides:
//! - `exec`: One-shot `sh -c` execution with captured stdout/stderr.
//! - `safety`: Command whitelist validation, chain splitting, policy JSON.

pub const exec = @import("exec.zig");
pub const safety = @import("safety.zig");

// Re-export primary types
pub const ExecResult = exec.ExecResult;
pub const CheckResult = safety.CheckResult;

test {
    @import("std").testing.refAllDecls(@This());
}
