//! Shell execution and command safety.
//!
//! Provides:
//! - `exec`: One-shot `sh -c` execution with captured stdout/stderr.
//! - `safety`: Command whitelist validation, chain splitting, policy JSON.
//! - `pty`: PTY lifecycle helpers for interactive shells.
//! - `session`: Persistent shell session abstraction.
//! - `signal`: Signal forwarding helpers.

pub const exec = @import("exec.zig");
pub const safety = @import("safety.zig");
pub const pty = @import("pty.zig");
pub const session = @import("session.zig");
pub const signal = @import("signal.zig");

// Re-export primary types
pub const ExecResult = exec.ExecResult;
pub const CheckResult = safety.CheckResult;

test {
    @import("std").testing.refAllDecls(@This());
}
