//! Integration tests for `agent.sandbox.detect`.

const std = @import("std");
const builtin = @import("builtin");
const main = @import("main");
const detect = main.agent.sandbox.detect;

test "detect returns CapabilityReport with valid kernel version" {
    if (builtin.os.tag != .linux) return;
    const report = detect.detect();
    try std.testing.expect(report.kernel_version[0] >= 5);
}

test "detect allRequired reflects host capabilities" {
    if (builtin.os.tag != .linux) return;
    const report = detect.detect();
    // On a modern Linux host, at minimum kernel version and seccomp
    // should be available. Landlock and user_ns depend on distro config.
    try std.testing.expect(report.kernel_version_ok);
    try std.testing.expect(report.seccomp_available);
}

test "detect describeMissing produces valid output" {
    var buf: [256]u8 = undefined;
    const report = detect.CapabilityReport{};
    const desc = report.describeMissing(&buf);
    // All-false report should list all required capabilities.
    try std.testing.expect(desc.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, desc, "kernel") != null);
    try std.testing.expect(std.mem.indexOf(u8, desc, "landlock") != null);
}
