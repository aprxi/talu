//! Cgroups v2 per-sandbox resource limits.
//!
//! Creates a child cgroup under `/sys/fs/cgroup/talu-sandbox/` for each
//! sandboxed session. Limits are applied via controller files (pids.max,
//! memory.max, cpu.max). Cleanup kills remaining processes and removes the
//! cgroup directory.
//!
//! Cgroup creation and cleanup happen in the **parent** process — the child
//! cannot access `/sys/fs/cgroup` after mount namespace + pivot_root.

const std = @import("std");
const builtin = @import("builtin");

const CGROUP_ROOT = "/sys/fs/cgroup";
const PARENT_GROUP = CGROUP_ROOT ++ "/talu-sandbox";

pub const CgroupConfig = struct {
    /// Maximum number of PIDs in this cgroup (-> pids.max).
    pids_max: ?u64 = null,
    /// Memory limit in bytes (-> memory.max).
    memory_max: ?u64 = null,
    /// CPU quota as percentage of one core (100 = 1 full core).
    /// Maps to cpu.max as "quota period" where period = 100000us.
    cpu_percent: ?u32 = null,
};

/// Create a cgroup for child_pid, configure limits, and move the child into it.
///
/// Called from the parent process after fork(), before waitForExecBoundary().
/// Returns `error.SandboxCgroupUnavailable` if the cgroup hierarchy is not
/// writable (expected on hosts without cgroup delegation).
pub fn createForPid(pid: std.posix.pid_t, config: CgroupConfig) !void {
    if (builtin.os.tag != .linux) return error.SandboxCgroupUnavailable;

    ensureParentGroup() catch return error.SandboxCgroupUnavailable;

    var session_path_buf: [128]u8 = undefined;
    const session_path = sessionPath(pid, &session_path_buf) catch
        return error.SandboxCgroupUnavailable;

    std.posix.mkdir(session_path, 0o755) catch
        return error.SandboxCgroupUnavailable;
    errdefer std.posix.rmdir(session_path) catch {};

    if (config.pids_max) |v| try writeController(session_path, "pids.max", v);
    if (config.memory_max) |v| try writeController(session_path, "memory.max", v);
    if (config.cpu_percent) |pct| {
        // cpu.max format: "quota period" where period is 100000us (100ms)
        const period: u64 = 100_000;
        const quota: u64 = @as(u64, pct) * 1_000; // pct% of 100ms = pct * 1000us
        var buf: [64]u8 = undefined;
        const val = std.fmt.bufPrint(&buf, "{d} {d}", .{ quota, period }) catch
            return error.StrictSetupFailed;
        writeControllerStr(session_path, "cpu.max", val) catch
            return error.StrictSetupFailed;
    }

    // Move child into the cgroup.
    var pid_buf: [32]u8 = undefined;
    const pid_str = std.fmt.bufPrint(&pid_buf, "{d}", .{pid}) catch
        return error.StrictSetupFailed;
    writeControllerStr(session_path, "cgroup.procs", pid_str) catch
        return error.StrictSetupFailed;
}

/// Best-effort cleanup: kill remaining processes and remove the cgroup.
///
/// Called from the parent process after the child exits (in close()).
/// Ignores all errors — same pattern as mounts.cleanupSessionRootForPid.
pub fn cleanupForPid(pid: std.posix.pid_t) void {
    if (builtin.os.tag != .linux) return;

    var session_path_buf: [128]u8 = undefined;
    const session_path = sessionPath(pid, &session_path_buf) catch return;

    // Try cgroup.kill first (kernel >= 5.14), fall back to manual SIGKILL.
    writeControllerStr(session_path, "cgroup.kill", "1") catch {
        killAllInCgroup(session_path);
    };

    // Wait briefly for processes to exit, then remove directory.
    var attempts: u32 = 0;
    while (attempts < 5) : (attempts += 1) {
        std.posix.rmdir(session_path) catch {
            std.Thread.sleep(10 * std.time.ns_per_ms);
            continue;
        };
        return;
    }
    std.posix.rmdir(session_path) catch {};
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn ensureParentGroup() !void {
    std.posix.mkdir(PARENT_GROUP, 0o755) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };

    // Enable controllers in the parent group's subtree.
    writeControllerStr(CGROUP_ROOT, "cgroup.subtree_control", "+pids +memory +cpu") catch {};
    writeControllerStr(PARENT_GROUP, "cgroup.subtree_control", "+pids +memory +cpu") catch {};
}

fn sessionPath(pid: std.posix.pid_t, buf: *[128]u8) ![]const u8 {
    return std.fmt.bufPrint(buf, PARENT_GROUP ++ "/session-{d}", .{pid}) catch
        return error.StrictSetupFailed;
}

fn writeController(dir: []const u8, file: []const u8, value: u64) !void {
    var buf: [32]u8 = undefined;
    const str = std.fmt.bufPrint(&buf, "{d}", .{value}) catch
        return error.StrictSetupFailed;
    try writeControllerStr(dir, file, str);
}

fn writeControllerStr(dir: []const u8, file: []const u8, value: []const u8) !void {
    var path_buf: [256]u8 = undefined;
    const path = joinPath(dir, file, &path_buf) catch
        return error.StrictSetupFailed;

    const fd = std.posix.open(path, .{ .ACCMODE = .WRONLY }, 0) catch
        return error.StrictSetupFailed;
    defer std.posix.close(fd);
    _ = std.posix.write(fd, value) catch return error.StrictSetupFailed;
}

fn joinPath(dir: []const u8, file: []const u8, buf: *[256]u8) ![]const u8 {
    if (dir.len + 1 + file.len > buf.len)
        return error.StrictSetupFailed;
    @memcpy(buf[0..dir.len], dir);
    buf[dir.len] = '/';
    @memcpy(buf[dir.len + 1 ..][0..file.len], file);
    return buf[0 .. dir.len + 1 + file.len];
}

fn killAllInCgroup(session_path: []const u8) void {
    var path_buf: [256]u8 = undefined;
    const procs_path = joinPath(session_path, "cgroup.procs", &path_buf) catch return;

    const fd = std.posix.open(procs_path, .{ .ACCMODE = .RDONLY }, 0) catch return;
    defer std.posix.close(fd);

    var read_buf: [1024]u8 = undefined;
    const n = std.posix.read(fd, &read_buf) catch return;
    if (n == 0) return;

    var iter = std.mem.splitScalar(u8, read_buf[0..n], '\n');
    while (iter.next()) |line| {
        if (line.len == 0) continue;
        const pid_val = std.fmt.parseInt(std.posix.pid_t, line, 10) catch continue;
        std.posix.kill(pid_val, 9) catch {};
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "CgroupConfig defaults are all null" {
    const config = CgroupConfig{};
    try std.testing.expect(config.pids_max == null);
    try std.testing.expect(config.memory_max == null);
    try std.testing.expect(config.cpu_percent == null);
}

test "createForPid returns SandboxCgroupUnavailable on non-Linux" {
    if (builtin.os.tag == .linux) return;
    try std.testing.expectError(error.SandboxCgroupUnavailable, createForPid(99999, .{ .pids_max = 10 }));
}

test "createForPid returns typed error when cgroupfs not writable" {
    if (builtin.os.tag != .linux) return;
    // Use a PID unlikely to collide. On most dev/CI hosts without cgroup
    // delegation, this returns SandboxCgroupUnavailable.
    const result = createForPid(99999, .{ .pids_max = 10 });
    _ = result catch |err| switch (err) {
        error.SandboxCgroupUnavailable => return,
        error.StrictSetupFailed => return,
        else => return err,
    };
    // If it succeeded (host has cgroup delegation), clean up.
    cleanupForPid(99999);
}

test "cleanupForPid is no-op for nonexistent cgroup" {
    if (builtin.os.tag != .linux) return;
    // Must not crash or error.
    cleanupForPid(88888);
}

test "joinPath produces correct path" {
    var buf: [256]u8 = undefined;
    const path = try joinPath("/sys/fs/cgroup/test", "pids.max", &buf);
    try std.testing.expectEqualStrings("/sys/fs/cgroup/test/pids.max", path);
}
