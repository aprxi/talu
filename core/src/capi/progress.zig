//! C-API progress aliases.
//!
//! This boundary re-exports core progress contracts so C-facing modules keep
//! stable names while core modules avoid depending on capi internals.

const std = @import("std");
const core_progress = @import("../progress.zig");

pub const ProgressAction = core_progress.ProgressAction;
pub const ProgressUpdate = core_progress.ProgressUpdate;
pub const CProgressCallback = core_progress.Callback;
pub const ProgressContext = core_progress.Context;

test "ProgressContext.NONE is a safe no-op" {
    const ctx = ProgressContext.NONE;
    try std.testing.expect(!ctx.isActive());
    ctx.emit(.{ .action = .update, .current = 1 });
}
