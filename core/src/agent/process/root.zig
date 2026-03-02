//! Long-lived non-PTY process sessions for `/v1/agent/processes/*`.
//!
//! This module provides bidirectional stdin/stdout/stderr pipes for
//! line-oriented RPC-style processes (e.g. `pi --mode rpc`).

pub const session = @import("session.zig");
pub const ProcessSession = session.ProcessSession;

test {
    @import("std").testing.refAllDecls(@This());
}
