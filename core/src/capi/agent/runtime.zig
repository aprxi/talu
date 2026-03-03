//! Runtime mode/backend parsing for agent C API entrypoints.

const std = @import("std");
const builtin = @import("builtin");
const sandbox = @import("../../agent/sandbox/root.zig");
const policy_api = @import("policy.zig");
const capi_error = @import("../error.zig");
const error_codes = @import("../error_codes.zig");

pub const TaluAgentRuntimeMode = enum(c_int) {
    host = 0,
    strict = 1,
};

pub const TaluSandboxBackend = enum(c_int) {
    linux_local = 0,
    oci = 1,
    apple_container = 2,
};

pub fn parseRuntimeMode(value: c_int) !sandbox.RuntimeMode {
    return switch (value) {
        0 => .host,
        1 => .strict,
        else => error.InvalidArgument,
    };
}

pub fn parseBackend(value: c_int) !sandbox.Backend {
    return switch (value) {
        0 => .linux_local,
        1 => .oci,
        2 => .apple_container,
        else => error.InvalidArgument,
    };
}

pub fn validate(mode: sandbox.RuntimeMode, backend: sandbox.Backend) !void {
    return sandbox.validate(mode, backend);
}

pub fn talu_agent_runtime_validate_strict(
    policy: ?*policy_api.TaluAgentPolicy,
    cwd: ?[*:0]const u8,
    sandbox_backend: c_int,
) callconv(.c) i32 {
    capi_error.clearError();

    const backend = parseBackend(sandbox_backend) catch |err| {
        capi_error.setError(err, "invalid sandbox backend", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    validate(.strict, backend) catch |err| {
        capi_error.setError(err, "strict runtime unavailable", .{});
        return @intFromEnum(error_codes.ErrorCode.policy_strict_unavailable);
    };

    const cwd_slice: ?[]const u8 = if (cwd) |value| blk: {
        const slice = std.mem.sliceTo(value, 0);
        if (slice.len == 0) {
            capi_error.setErrorWithCode(.invalid_argument, "cwd is empty", .{});
            return @intFromEnum(error_codes.ErrorCode.invalid_argument);
        }
        break :blk slice;
    } else null;

    policy_api.prepareRuntimeProfiles(policy, cwd_slice) catch |err| switch (err) {
        // Startup validation is workspace-scoped; cwd-dependent allowlist paths
        // may be absent at startup and are compiled on request with effective cwd.
        error.StrictDeferred => {},
        else => {
            capi_error.setError(err, "strict runtime profile preparation failed", .{});
            return @intFromEnum(error_codes.ErrorCode.policy_strict_setup_failed);
        },
    };

    sandbox.landlock.validateAvailability() catch |err| {
        capi_error.setError(err, "strict runtime support check failed", .{});
        return @intFromEnum(error_codes.ErrorCode.policy_strict_unavailable);
    };

    return 0;
}

pub const TaluCapabilityReport = extern struct {
    kernel_version_ok: bool = false,
    kernel_version_major: u32 = 0,
    kernel_version_minor: u32 = 0,
    landlock_available: bool = false,
    landlock_abi_version: u8 = 0,
    user_ns_available: bool = false,
    seccomp_available: bool = false,
    cgroupv2_available: bool = false,
    cgroupv2_writable: bool = false,
    probes_passed: bool = false,
};

pub fn talu_agent_runtime_validate_strict_ext(
    sandbox_backend: c_int,
    strict_required: bool,
    run_probes: bool,
    cwd: ?[*:0]const u8,
    out_report: ?*TaluCapabilityReport,
) callconv(.c) i32 {
    capi_error.clearError();

    const backend = parseBackend(sandbox_backend) catch |err| {
        capi_error.setError(err, "invalid sandbox backend", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    // Non-linux-local backends are not implemented for strict mode yet.
    if (backend != .linux_local) {
        if (strict_required) {
            capi_error.setErrorWithCode(.sandbox_detect_failed, "strict mode not implemented for this backend", .{});
            return @intFromEnum(error_codes.ErrorCode.sandbox_detect_failed);
        }
        if (out_report) |r| r.* = std.mem.zeroes(TaluCapabilityReport);
        return 0;
    }

    const cap = sandbox.detect.detect();
    if (out_report) |r| {
        r.* = std.mem.zeroes(TaluCapabilityReport);
        r.kernel_version_ok = cap.kernel_version_ok;
        r.kernel_version_major = cap.kernel_version[0];
        r.kernel_version_minor = cap.kernel_version[1];
        r.landlock_available = cap.landlock_available;
        r.landlock_abi_version = cap.landlock_abi_version;
        r.user_ns_available = cap.user_ns_available;
        r.seccomp_available = cap.seccomp_available;
        r.cgroupv2_available = cap.cgroupv2_available;
        r.cgroupv2_writable = cap.cgroupv2_writable;
    }

    if (strict_required and !cap.allRequired()) {
        capi_error.setErrorWithCode(.sandbox_detect_failed, "missing required capabilities", .{});
        return @intFromEnum(error_codes.ErrorCode.sandbox_detect_failed);
    }

    if (run_probes) {
        const ws = if (cwd) |c| std.mem.sliceTo(c, 0) else "/tmp";
        const report = sandbox.probe.runAll(std.heap.c_allocator, ws) catch {
            if (strict_required) {
                capi_error.setErrorWithCode(.sandbox_probe_failed, "probe execution failed", .{});
                return @intFromEnum(error_codes.ErrorCode.sandbox_probe_failed);
            }
            return 0;
        };
        if (out_report) |r| r.probes_passed = report.allPassed();
        if (strict_required and !report.allPassed()) {
            capi_error.setErrorWithCode(.sandbox_probe_failed, "conformance probes failed", .{});
            return @intFromEnum(error_codes.ErrorCode.sandbox_probe_failed);
        }
    }

    return 0;
}

test "parseRuntimeMode accepts host and strict" {
    try std.testing.expectEqual(sandbox.RuntimeMode.host, try parseRuntimeMode(0));
    try std.testing.expectEqual(sandbox.RuntimeMode.strict, try parseRuntimeMode(1));
}

test "parseBackend accepts known values" {
    try std.testing.expectEqual(sandbox.Backend.linux_local, try parseBackend(0));
    try std.testing.expectEqual(sandbox.Backend.oci, try parseBackend(1));
    try std.testing.expectEqual(sandbox.Backend.apple_container, try parseBackend(2));
}

test "strict validate rejects unsupported backend" {
    var policy: ?*policy_api.TaluAgentPolicy = null;
    const policy_json = "{\"default\":\"deny\",\"statements\":[]}";
    const rc = policy_api.talu_agent_policy_create(policy_json.ptr, policy_json.len, &policy);
    defer policy_api.talu_agent_policy_free(policy);
    try std.testing.expectEqual(@as(i32, 0), rc);

    const validate_rc = talu_agent_runtime_validate_strict(
        policy,
        null,
        @intFromEnum(TaluSandboxBackend.oci),
    );
    try std.testing.expect(validate_rc != 0);
}

test "strict validate fails for deny-plus-allow fs write policy" {
    if (builtin.os.tag != .linux) return;

    var policy: ?*policy_api.TaluAgentPolicy = null;
    const policy_json =
        \\{
        \\  "default":"deny",
        \\  "statements":[
        \\    {"effect":"allow","action":"tool.exec","command":"echo *"},
        \\    {"effect":"allow","action":"tool.fs.write","resource":"src/**"},
        \\    {"effect":"deny","action":"tool.fs.write","resource":"src/private/**"}
        \\  ]
        \\}
    ;
    const rc = policy_api.talu_agent_policy_create(policy_json.ptr, policy_json.len, &policy);
    defer policy_api.talu_agent_policy_free(policy);
    try std.testing.expectEqual(@as(i32, 0), rc);

    const validate_rc = talu_agent_runtime_validate_strict(
        policy,
        null,
        @intFromEnum(TaluSandboxBackend.linux_local),
    );
    try std.testing.expectEqual(
        @as(i32, @intFromEnum(error_codes.ErrorCode.policy_strict_setup_failed)),
        validate_rc,
    );
}

test "strict validate allows deferred cwd-dependent fs rules" {
    if (builtin.os.tag != .linux) return;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const cwd = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(cwd);
    const cwd_z = try std.testing.allocator.dupeZ(u8, cwd);
    defer std.testing.allocator.free(cwd_z);

    var policy: ?*policy_api.TaluAgentPolicy = null;
    const policy_json =
        \\{
        \\  "default":"deny",
        \\  "statements":[
        \\    {"effect":"allow","action":"tool.exec","command":"echo *"},
        \\    {"effect":"allow","action":"tool.fs.write","resource":"allowed/**"}
        \\  ]
        \\}
    ;
    const rc = policy_api.talu_agent_policy_create(policy_json.ptr, policy_json.len, &policy);
    defer policy_api.talu_agent_policy_free(policy);
    try std.testing.expectEqual(@as(i32, 0), rc);

    const validate_rc = talu_agent_runtime_validate_strict(
        policy,
        cwd_z.ptr,
        @intFromEnum(TaluSandboxBackend.linux_local),
    );
    // Deferred compile must not be reported as strict setup failure.
    try std.testing.expect(validate_rc != @intFromEnum(error_codes.ErrorCode.policy_strict_setup_failed));
}

test "validate_strict_ext populates capability report" {
    var report = std.mem.zeroes(TaluCapabilityReport);
    const rc = talu_agent_runtime_validate_strict_ext(
        @intFromEnum(TaluSandboxBackend.linux_local),
        false, // strict_required
        false, // run_probes
        null,
        &report,
    );
    try std.testing.expectEqual(@as(i32, 0), rc);
    // On Linux, kernel version should be detected.
    if (builtin.os.tag == .linux) {
        try std.testing.expect(report.kernel_version_major > 0);
    }
}

test "validate_strict_ext rejects invalid backend" {
    const rc = talu_agent_runtime_validate_strict_ext(
        99, // invalid
        false,
        false,
        null,
        null,
    );
    try std.testing.expectEqual(
        @as(i32, @intFromEnum(error_codes.ErrorCode.invalid_argument)),
        rc,
    );
}

test "validate_strict_ext oci backend fails when strict_required" {
    var report = std.mem.zeroes(TaluCapabilityReport);
    const rc = talu_agent_runtime_validate_strict_ext(
        @intFromEnum(TaluSandboxBackend.oci),
        true, // strict_required
        false,
        null,
        &report,
    );
    try std.testing.expectEqual(
        @as(i32, @intFromEnum(error_codes.ErrorCode.sandbox_detect_failed)),
        rc,
    );
}

test "validate_strict_ext oci backend succeeds when not strict_required" {
    var report = std.mem.zeroes(TaluCapabilityReport);
    const rc = talu_agent_runtime_validate_strict_ext(
        @intFromEnum(TaluSandboxBackend.oci),
        false, // strict_required
        false,
        null,
        &report,
    );
    try std.testing.expectEqual(@as(i32, 0), rc);
    // Non-linux-local backend returns zeroed report.
    try std.testing.expect(!report.kernel_version_ok);
    try std.testing.expectEqual(@as(u32, 0), report.kernel_version_major);
}

test "validate_strict_ext apple_container backend fails when strict_required" {
    var report = std.mem.zeroes(TaluCapabilityReport);
    const rc = talu_agent_runtime_validate_strict_ext(
        @intFromEnum(TaluSandboxBackend.apple_container),
        true,
        false,
        null,
        &report,
    );
    try std.testing.expectEqual(
        @as(i32, @intFromEnum(error_codes.ErrorCode.sandbox_detect_failed)),
        rc,
    );
}

test "validate_strict_ext apple_container backend succeeds when not strict_required" {
    var report = std.mem.zeroes(TaluCapabilityReport);
    const rc = talu_agent_runtime_validate_strict_ext(
        @intFromEnum(TaluSandboxBackend.apple_container),
        false,
        false,
        null,
        &report,
    );
    try std.testing.expectEqual(@as(i32, 0), rc);
    // Non-linux-local backend returns zeroed report.
    try std.testing.expect(!report.kernel_version_ok);
    try std.testing.expectEqual(@as(u32, 0), report.kernel_version_major);
}
