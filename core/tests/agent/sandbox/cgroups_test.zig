//! Integration tests for `agent.sandbox.cgroups`.

const std = @import("std");
const builtin = @import("builtin");
const main = @import("main");
const cgroups = main.agent.sandbox.cgroups;

test "createForPid returns typed error on most dev hosts" {
    if (builtin.os.tag != .linux) return;
    // Most dev/CI hosts lack cgroup delegation, so this exercises the
    // error path. Hosts with delegation exercise the success path.
    const result = cgroups.createForPid(77777, .{
        .pids_max = 64,
        .memory_max = 128 * 1024 * 1024,
    });
    _ = result catch |err| switch (err) {
        error.SandboxCgroupUnavailable => return,
        error.StrictSetupFailed => return,
        else => return err,
    };
    cgroups.cleanupForPid(77777);
}

test "cleanupForPid is idempotent" {
    if (builtin.os.tag != .linux) return;
    cgroups.cleanupForPid(66666);
    cgroups.cleanupForPid(66666);
}
