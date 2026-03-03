//! Integration tests for `agent.sandbox.probe`.

const std = @import("std");
const builtin = @import("builtin");
const main = @import("main");
const probe = main.agent.sandbox.probe;

test "runAll returns valid ProbeReport" {
    if (builtin.os.tag != .linux) return;

    const report = probe.runAll(std.testing.allocator, "/tmp") catch |err| {
        // On hosts without sandbox support, runAll may fail during
        // policy parse or profile build — that's expected, not a test failure.
        std.log.warn("probe runAll failed (expected on some hosts): {}", .{err});
        return;
    };

    // Each probe must be pass or skip (never fail on a working host).
    const c = report.counts();
    try std.testing.expectEqual(@as(u8, 0), c.fail);
    try std.testing.expect(c.pass + c.skip == 5);
}
