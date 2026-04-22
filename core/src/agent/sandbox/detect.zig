//! Kernel capability detection for strict sandbox mode.
//!
//! Probes are fast, idempotent, and side-effect-free. Each checks whether a
//! specific kernel mechanism is available on the running host. The aggregate
//! CapabilityReport determines whether linux-local strict mode can be enabled.

const std = @import("std");
const builtin = @import("builtin");

const c = if (builtin.os.tag == .linux) @cImport({
    @cInclude("sys/prctl.h");
    @cInclude("sys/syscall.h");
    @cInclude("unistd.h");
    @cInclude("errno.h");
}) else struct {};

pub const CapabilityReport = struct {
    kernel_version_ok: bool = false,
    kernel_version: [2]u32 = .{ 0, 0 },
    landlock_available: bool = false,
    landlock_abi_version: u8 = 0,
    user_ns_available: bool = false,
    seccomp_available: bool = false,
    cgroupv2_available: bool = false,
    cgroupv2_writable: bool = false,

    /// True when all capabilities required for linux_local strict mode are present.
    pub fn allRequired(self: CapabilityReport) bool {
        return self.kernel_version_ok and
            self.landlock_available and
            self.user_ns_available and
            self.seccomp_available;
    }

    /// Write a comma-separated list of missing required capabilities into buf.
    pub fn describeMissing(self: CapabilityReport, buf: []u8) []const u8 {
        var stream = std.io.fixedBufferStream(buf);
        const writer = stream.writer();
        var count: u32 = 0;

        const fields = .{
            .{ self.kernel_version_ok, "kernel>=5.13" },
            .{ self.landlock_available, "landlock" },
            .{ self.user_ns_available, "user_namespaces" },
            .{ self.seccomp_available, "seccomp" },
        };
        inline for (fields) |entry| {
            if (!entry[0]) {
                if (count > 0) writer.writeAll(", ") catch return stream.getWritten();
                writer.writeAll(entry[1]) catch return stream.getWritten();
                count += 1;
            }
        }
        return stream.getWritten();
    }
};

/// Run all capability probes and return the aggregate report.
///
/// Never errors — each probe returns a boolean. On non-Linux platforms
/// every field is false.
pub fn detect() CapabilityReport {
    if (builtin.os.tag != .linux) return .{};

    var report = CapabilityReport{};

    const kv = detectKernelVersion();
    report.kernel_version = kv.version;
    report.kernel_version_ok = kv.ok;

    const ll = detectLandlock();
    report.landlock_available = ll.available;
    report.landlock_abi_version = ll.abi_version;

    report.user_ns_available = detectUserNamespace();
    report.seccomp_available = detectSeccomp();

    const cg = detectCgroupV2();
    report.cgroupv2_available = cg.available;
    report.cgroupv2_writable = cg.writable;

    return report;
}

// ---------------------------------------------------------------------------
// Sub-probes (private, no allocation, no side effects)
// ---------------------------------------------------------------------------

const KernelVersionResult = struct { ok: bool, version: [2]u32 };

fn detectKernelVersion() KernelVersionResult {
    if (builtin.os.tag != .linux) return .{ .ok = false, .version = .{ 0, 0 } };

    var uts: std.posix.utsname = undefined;
    if (std.os.linux.uname(&uts) != 0) return .{ .ok = false, .version = .{ 0, 0 } };

    const release: []const u8 = std.mem.sliceTo(&uts.release, 0);
    const parsed = parseKernelVersion(release);
    return .{
        .ok = parsed[0] > 5 or (parsed[0] == 5 and parsed[1] >= 13),
        .version = parsed,
    };
}

/// Parse "X.Y..." from a kernel release string. Returns {major, minor}.
fn parseKernelVersion(release: []const u8) [2]u32 {
    var major: u32 = 0;
    var minor: u32 = 0;
    var i: usize = 0;

    // Parse major
    while (i < release.len and release[i] >= '0' and release[i] <= '9') : (i += 1) {
        major = major *| 10 +| (release[i] - '0');
    }
    if (i < release.len and release[i] == '.') i += 1;

    // Parse minor
    while (i < release.len and release[i] >= '0' and release[i] <= '9') : (i += 1) {
        minor = minor *| 10 +| (release[i] - '0');
    }

    return .{ major, minor };
}

const LANDLOCK_CREATE_RULESET_VERSION: c_ulong = 1 << 0;

const LandlockResult = struct { available: bool, abi_version: u8 };

fn detectLandlock() LandlockResult {
    if (builtin.os.tag != .linux) return .{ .available = false, .abi_version = 0 };

    // SYS_landlock_create_ruleset with NULL attr, 0 size, and VERSION flag
    // returns the ABI version on success.
    const ret = c.syscall(
        c.SYS_landlock_create_ruleset,
        @as(c_ulong, 0), // NULL attr
        @as(c_ulong, 0), // 0 size
        LANDLOCK_CREATE_RULESET_VERSION,
    );

    if (ret < 0) return .{ .available = false, .abi_version = 0 };
    return .{ .available = true, .abi_version = @intCast(@as(u64, @bitCast(ret))) };
}

fn detectUserNamespace() bool {
    if (builtin.os.tag != .linux) return false;

    // Heuristic: verify user namespaces are compiled into the kernel and
    // not disabled by sysctl or AppArmor. A direct unshare(CLONE_NEWUSER)
    // probe would be authoritative but has side effects (creates a
    // namespace, risks interaction with the calling process). The
    // heuristic covers the common cases: sysctl toggle, AppArmor
    // restriction (Ubuntu 24.04+), and kernel compile-time absence.
    const ns_fd = std.posix.open("/proc/self/ns/user", .{ .ACCMODE = .RDONLY }, 0) catch {
        return false;
    };
    std.posix.close(ns_fd);

    // Check /proc/sys/kernel/unprivileged_userns_clone if it exists.
    // Value "1" = enabled, "0" = disabled by sysctl.
    // File absent = sysctl not present; check apparmor/LSM restrictions
    // via /proc/sys/kernel/apparmor_restrict_unprivileged_userns instead.
    const fd = std.posix.open("/proc/sys/kernel/unprivileged_userns_clone", .{ .ACCMODE = .RDONLY }, 0) catch {
        // Sysctl absent — check AppArmor restriction (Ubuntu 24.04+).
        return !isAppArmorUsernsRestricted();
    };
    defer std.posix.close(fd);

    var buf: [8]u8 = undefined;
    const n = std.posix.read(fd, &buf) catch return false;
    if (n == 0) return false;
    return buf[0] == '1';
}

/// Check if AppArmor restricts unprivileged user namespaces.
/// Returns true when the restriction is active (userns blocked).
fn isAppArmorUsernsRestricted() bool {
    const fd = std.posix.open(
        "/proc/sys/kernel/apparmor_restrict_unprivileged_userns",
        .{ .ACCMODE = .RDONLY },
        0,
    ) catch return false; // File absent = no AppArmor restriction
    defer std.posix.close(fd);

    var buf: [8]u8 = undefined;
    const n = std.posix.read(fd, &buf) catch return false;
    if (n == 0) return false;
    return buf[0] == '1';
}

const PR_GET_SECCOMP: c_int = 21;

fn detectSeccomp() bool {
    if (builtin.os.tag != .linux) return false;

    // prctl(PR_GET_SECCOMP) returns 0 if seccomp is compiled in but no
    // filter is applied (our expected state). Returns -1 with EINVAL if
    // seccomp is not compiled into the kernel.
    const ret = c.prctl(PR_GET_SECCOMP, @as(c_ulong, 0), @as(c_ulong, 0), @as(c_ulong, 0), @as(c_ulong, 0));
    return ret >= 0;
}

const CgroupV2Result = struct { available: bool, writable: bool };

fn detectCgroupV2() CgroupV2Result {
    if (builtin.os.tag != .linux) return .{ .available = false, .writable = false };

    // Check if cgroup v2 hierarchy is mounted.
    const ctrl_fd = std.posix.open("/sys/fs/cgroup/cgroup.controllers", .{ .ACCMODE = .RDONLY }, 0) catch {
        return .{ .available = false, .writable = false };
    };
    std.posix.close(ctrl_fd);

    // Check if we can create child cgroups (write access to cgroup tree).
    // Try to mkdir + rmdir a probe directory under talu-sandbox parent.
    const probe_path = "/sys/fs/cgroup/talu-sandbox-probe";
    std.posix.mkdir(probe_path, 0o755) catch {
        return .{ .available = true, .writable = false };
    };
    std.posix.rmdir(probe_path) catch {};
    return .{ .available = true, .writable = true };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "detect returns populated CapabilityReport on Linux" {
    if (builtin.os.tag != .linux) return;
    const report = detect();
    try std.testing.expect(report.kernel_version[0] > 0);
}

test "detect returns all-false CapabilityReport on non-Linux" {
    if (builtin.os.tag == .linux) return;
    const report = detect();
    try std.testing.expect(!report.allRequired());
    try std.testing.expectEqual(@as(u32, 0), report.kernel_version[0]);
}

test "parseKernelVersion parses standard release string" {
    const v1 = parseKernelVersion("5.13.0-generic");
    try std.testing.expectEqual(@as(u32, 5), v1[0]);
    try std.testing.expectEqual(@as(u32, 13), v1[1]);

    const v2 = parseKernelVersion("6.17.0-14-generic");
    try std.testing.expectEqual(@as(u32, 6), v2[0]);
    try std.testing.expectEqual(@as(u32, 17), v2[1]);
}

test "parseKernelVersion handles edge cases" {
    const empty = parseKernelVersion("");
    try std.testing.expectEqual(@as(u32, 0), empty[0]);
    try std.testing.expectEqual(@as(u32, 0), empty[1]);

    const major_only = parseKernelVersion("6");
    try std.testing.expectEqual(@as(u32, 6), major_only[0]);
    try std.testing.expectEqual(@as(u32, 0), major_only[1]);
}

test "CapabilityReport allRequired returns true when all required present" {
    const report = CapabilityReport{
        .kernel_version_ok = true,
        .kernel_version = .{ 6, 1 },
        .landlock_available = true,
        .landlock_abi_version = 3,
        .user_ns_available = true,
        .seccomp_available = true,
        .cgroupv2_available = false,
        .cgroupv2_writable = false,
    };
    try std.testing.expect(report.allRequired());
}

test "CapabilityReport allRequired returns false when landlock missing" {
    const report = CapabilityReport{
        .kernel_version_ok = true,
        .kernel_version = .{ 6, 1 },
        .landlock_available = false,
        .user_ns_available = true,
        .seccomp_available = true,
    };
    try std.testing.expect(!report.allRequired());
}

test "describeMissing lists missing required capabilities" {
    var buf: [256]u8 = undefined;
    const report = CapabilityReport{
        .kernel_version_ok = true,
        .landlock_available = false,
        .user_ns_available = true,
        .seccomp_available = false,
    };
    const desc = report.describeMissing(&buf);
    try std.testing.expect(std.mem.indexOf(u8, desc, "landlock") != null);
    try std.testing.expect(std.mem.indexOf(u8, desc, "seccomp") != null);
    try std.testing.expect(std.mem.indexOf(u8, desc, "kernel") == null);
}

test "describeMissing returns empty for full report" {
    var buf: [256]u8 = undefined;
    const report = CapabilityReport{
        .kernel_version_ok = true,
        .landlock_available = true,
        .user_ns_available = true,
        .seccomp_available = true,
    };
    const desc = report.describeMissing(&buf);
    try std.testing.expectEqual(@as(usize, 0), desc.len);
}
