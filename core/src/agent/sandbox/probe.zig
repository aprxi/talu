//! Conformance probe runner for strict sandbox mode.
//!
//! Spawns short-lived sandbox sessions to verify enforcement end-to-end.
//! Each probe tests a specific guarantee (exec deny, file deny, etc.) and
//! returns a typed result (pass/fail/skip). Probes are run at server startup
//! when `--strict-probes` is enabled.
//!
//! Probes reuse the existing ProcessSession spawn path, so they exercise the
//! real sandbox code — not a simulation.

const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const process_session = @import("../process/session.zig");
const policy_mod = @import("../policy/root.zig");
const profile_mod = @import("profile.zig");

pub const ProbeStatus = enum(u8) {
    pass = 0,
    fail = 1,
    skip = 2,
};

pub const ProbeResult = struct {
    name: []const u8,
    status: ProbeStatus,
    detail_buf: [128]u8 = std.mem.zeroes([128]u8),
    detail_len: u8 = 0,

    fn passed(name: []const u8) ProbeResult {
        return .{ .name = name, .status = .pass };
    }

    fn failed(name: []const u8, msg: []const u8) ProbeResult {
        var r = ProbeResult{ .name = name, .status = .fail };
        const len: u8 = @intCast(@min(msg.len, r.detail_buf.len));
        @memcpy(r.detail_buf[0..len], msg[0..len]);
        r.detail_len = len;
        return r;
    }

    fn skipped(name: []const u8) ProbeResult {
        return .{ .name = name, .status = .skip };
    }

    pub fn detail(self: *const ProbeResult) []const u8 {
        return self.detail_buf[0..self.detail_len];
    }
};

pub const ProbeReport = struct {
    exec_deny: ProbeResult,
    exec_allow: ProbeResult,
    file_deny: ProbeResult,
    file_allow: ProbeResult,
    descendant_inherit: ProbeResult,

    /// True when no probe returned .fail.
    pub fn allPassed(self: ProbeReport) bool {
        const results = [_]ProbeResult{
            self.exec_deny,
            self.exec_allow,
            self.file_deny,
            self.file_allow,
            self.descendant_inherit,
        };
        for (results) |r| {
            if (r.status == .fail) return false;
        }
        return true;
    }

    /// Count probes with each status.
    pub fn counts(self: ProbeReport) struct { pass: u8, fail: u8, skip: u8 } {
        var p: u8 = 0;
        var f: u8 = 0;
        var s: u8 = 0;
        const results = [_]ProbeResult{
            self.exec_deny,
            self.exec_allow,
            self.file_deny,
            self.file_allow,
            self.descendant_inherit,
        };
        for (results) |r| switch (r.status) {
            .pass => p += 1,
            .fail => f += 1,
            .skip => s += 1,
        };
        return .{ .pass = p, .fail = f, .skip = s };
    }
};

/// Run all conformance probes and return the aggregate report.
///
/// Creates a minimal policy (allow only echo + sh), builds an ExecProfile,
/// and spawns short-lived sandbox sessions for each probe.
pub fn runAll(allocator: Allocator, workspace: []const u8) !ProbeReport {
    if (builtin.os.tag != .linux) {
        return ProbeReport{
            .exec_deny = ProbeResult.skipped("exec_deny"),
            .exec_allow = ProbeResult.skipped("exec_allow"),
            .file_deny = ProbeResult.skipped("file_deny"),
            .file_allow = ProbeResult.skipped("file_allow"),
            .descendant_inherit = ProbeResult.skipped("descendant_inherit"),
        };
    }

    // Build a minimal probe policy: allow only echo and sh.
    const probe_policy_json =
        \\{"default":"deny","statements":[
        \\  {"effect":"allow","action":"tool.process","command":"echo *"},
        \\  {"effect":"allow","action":"tool.process","command":"sh *"},
        \\  {"effect":"allow","action":"tool.process","command":"touch *"}
        \\]}
    ;
    var pol = try policy_mod.parsePolicy(allocator, probe_policy_json);
    defer pol.deinit();

    var exec_profile = try profile_mod.buildExecProfile(allocator, &pol, .{
        .action = "tool.process",
        .cwd = workspace,
        .include_shell_paths = true,
    });
    defer exec_profile.deinit();

    return .{
        .exec_deny = probeExecDeny(allocator, workspace, &exec_profile),
        .exec_allow = probeExecAllow(allocator, workspace, &exec_profile),
        .file_deny = probeFileDeny(allocator, workspace, &exec_profile),
        .file_allow = probeFileAllow(allocator, workspace, &exec_profile),
        .descendant_inherit = probeDescendantInherit(allocator, workspace, &exec_profile),
    };
}

// ---------------------------------------------------------------------------
// Individual probes
// ---------------------------------------------------------------------------

fn probeExecDeny(allocator: Allocator, cwd: []const u8, exec_profile: *profile_mod.ExecProfile) ProbeResult {
    // /usr/bin/id is NOT in the probe allowlist. If sandbox enforcement
    // works, exec fails and $? is nonzero. Skip if the binary is absent
    // on the host to avoid a false pass (binary-not-found also gives
    // nonzero RC, indistinguishable from sandbox denial).
    std.fs.accessAbsolute("/usr/bin/id", .{}) catch {
        return ProbeResult.skipped("exec_deny");
    };
    return runCommandProbe(allocator, "exec_deny", "/usr/bin/id 2>/dev/null; echo RC:$?", cwd, exec_profile, .{
        .expect_contains = "RC:",
        .expect_not_contains = "RC:0",
    });
}

fn probeExecAllow(allocator: Allocator, cwd: []const u8, exec_profile: *profile_mod.ExecProfile) ProbeResult {
    // touch is an external binary (not a shell builtin) and is in the
    // probe allowlist. Using it proves the Landlock exec allowlist
    // permits execve() for allowed paths, not just shell builtins.
    return runCommandProbe(allocator, "exec_allow", "touch talu-probe-exec-test && echo EXEC_OK && rm -f talu-probe-exec-test", cwd, exec_profile, .{
        .expect_contains = "EXEC_OK",
    });
}

fn probeFileDeny(allocator: Allocator, cwd: []const u8, exec_profile: *profile_mod.ExecProfile) ProbeResult {
    // Writing outside workspace should fail.
    return runCommandProbe(allocator, "file_deny", "touch /tmp/talu-probe-deny-test 2>/dev/null; echo RC:$?", cwd, exec_profile, .{
        .expect_contains = "RC:",
        .expect_not_contains = "RC:0",
    });
}

fn probeFileAllow(allocator: Allocator, cwd: []const u8, exec_profile: *profile_mod.ExecProfile) ProbeResult {
    // Writing inside workspace should succeed (Landlock allows workspace).
    return runCommandProbe(allocator, "file_allow", "touch talu-probe-allow-test && echo WRITE_OK && rm -f talu-probe-allow-test", cwd, exec_profile, .{
        .expect_contains = "WRITE_OK",
    });
}

fn probeDescendantInherit(allocator: Allocator, cwd: []const u8, exec_profile: *profile_mod.ExecProfile) ProbeResult {
    // Shell fork: echo (allowed) succeeds, /usr/bin/id (denied) fails.
    // Skip if /usr/bin/id is absent (same rationale as exec_deny).
    std.fs.accessAbsolute("/usr/bin/id", .{}) catch {
        return ProbeResult.skipped("descendant_inherit");
    };
    return runCommandProbe(allocator, "descendant_inherit", "sh -c 'echo INHERIT_OK' && sh -c '/usr/bin/id 2>/dev/null; echo CHILD:$?'", cwd, exec_profile, .{
        .expect_contains = "INHERIT_OK",
        .expect_not_contains = "CHILD:0",
    });
}

// ---------------------------------------------------------------------------
// Probe infrastructure
// ---------------------------------------------------------------------------

const ProbeExpectation = struct {
    expect_contains: ?[]const u8 = null,
    expect_not_contains: ?[]const u8 = null,
};

fn runCommandProbe(
    allocator: Allocator,
    name: []const u8,
    command: []const u8,
    cwd: []const u8,
    exec_profile: *profile_mod.ExecProfile,
    expect: ProbeExpectation,
) ProbeResult {
    var session = process_session.ProcessSession.open(
        allocator,
        command,
        cwd,
        .{
            .mode = .strict,
            .backend = .linux_local,
            .exec_profile = exec_profile,
        },
    ) catch |err| switch (err) {
        error.StrictUnavailable => return ProbeResult.skipped(name),
        else => return ProbeResult.failed(name, "session open failed"),
    };
    defer session.close();

    const output = collectOutput(session, allocator) catch
        return ProbeResult.failed(name, "output collection failed");
    defer allocator.free(output);

    if (expect.expect_contains) |needle| {
        if (std.mem.indexOf(u8, output, needle) == null) {
            return ProbeResult.failed(name, "expected output not found");
        }
    }

    if (expect.expect_not_contains) |needle| {
        if (std.mem.indexOf(u8, output, needle) != null) {
            return ProbeResult.failed(name, "unexpected output found");
        }
    }

    return ProbeResult.passed(name);
}

fn collectOutput(session: *process_session.ProcessSession, allocator: Allocator) ![]u8 {
    var read_buf: [2048]u8 = undefined;
    var output = std.ArrayList(u8).empty;
    errdefer output.deinit(allocator);

    // Guard count is a deadlock guard, not timing synchronization.
    // Yield when no data is available to avoid busy-waiting.
    var guard: usize = 0;
    while (try session.isAlive() and guard < 500_000) : (guard += 1) {
        const n = try session.read(&read_buf);
        if (n > 0) {
            try output.appendSlice(allocator, read_buf[0..n]);
        } else {
            std.Thread.yield() catch {};
        }
    }
    // Drain remaining output after process exits.
    while (true) {
        const n = try session.read(&read_buf);
        if (n == 0) break;
        try output.appendSlice(allocator, read_buf[0..n]);
    }
    return try output.toOwnedSlice(allocator);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "ProbeReport allPassed returns true when all pass" {
    const report = ProbeReport{
        .exec_deny = ProbeResult.passed("exec_deny"),
        .exec_allow = ProbeResult.passed("exec_allow"),
        .file_deny = ProbeResult.passed("file_deny"),
        .file_allow = ProbeResult.passed("file_allow"),
        .descendant_inherit = ProbeResult.passed("descendant_inherit"),
    };
    try std.testing.expect(report.allPassed());
}

test "ProbeReport allPassed returns false when one fails" {
    const report = ProbeReport{
        .exec_deny = ProbeResult.passed("exec_deny"),
        .exec_allow = ProbeResult.failed("exec_allow", "test"),
        .file_deny = ProbeResult.passed("file_deny"),
        .file_allow = ProbeResult.passed("file_allow"),
        .descendant_inherit = ProbeResult.passed("descendant_inherit"),
    };
    try std.testing.expect(!report.allPassed());
}

test "ProbeReport allPassed returns true when some skip" {
    const report = ProbeReport{
        .exec_deny = ProbeResult.skipped("exec_deny"),
        .exec_allow = ProbeResult.skipped("exec_allow"),
        .file_deny = ProbeResult.skipped("file_deny"),
        .file_allow = ProbeResult.skipped("file_allow"),
        .descendant_inherit = ProbeResult.skipped("descendant_inherit"),
    };
    try std.testing.expect(report.allPassed());
}

test "ProbeReport counts returns correct counts" {
    const report = ProbeReport{
        .exec_deny = ProbeResult.passed("exec_deny"),
        .exec_allow = ProbeResult.failed("exec_allow", "x"),
        .file_deny = ProbeResult.skipped("file_deny"),
        .file_allow = ProbeResult.passed("file_allow"),
        .descendant_inherit = ProbeResult.skipped("descendant_inherit"),
    };
    const c = report.counts();
    try std.testing.expectEqual(@as(u8, 2), c.pass);
    try std.testing.expectEqual(@as(u8, 1), c.fail);
    try std.testing.expectEqual(@as(u8, 2), c.skip);
}

test "ProbeResult detail returns stored message" {
    const r = ProbeResult.failed("test", "something went wrong");
    try std.testing.expect(std.mem.indexOf(u8, r.detail(), "something") != null);
}
