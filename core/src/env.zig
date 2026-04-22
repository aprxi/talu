const std = @import("std");
const builtin = @import("builtin");

threadlocal var env_cache: std.StringHashMapUnmanaged(?[]const u8) = .{};

pub fn getenv(name: []const u8) ?[]const u8 {
    const allocator = std.heap.page_allocator;
    const current_value = std.process.getEnvVarOwned(allocator, name) catch null;

    if (env_cache.getPtr(name)) |cached_ptr| {
        if (current_value) |value| {
            if (cached_ptr.*) |cached| {
                if (std.mem.eql(u8, cached, value)) {
                    allocator.free(value);
                    return cached;
                }
            }
            // Keep prior allocations alive so previously returned slices never dangle.
            cached_ptr.* = value;
            return value;
        }
        cached_ptr.* = null;
        return null;
    }

    const key = allocator.dupe(u8, name) catch {
        if (current_value) |value| allocator.free(value);
        return current_value;
    };
    errdefer allocator.free(key);

    env_cache.put(allocator, key, current_value) catch return current_value;
    return current_value;
}

fn setEnvVarTest(allocator: std.mem.Allocator, key: []const u8, value: []const u8) !void {
    if (builtin.os.tag == .windows) {
        const key_w = try std.unicode.utf8ToUtf16LeAllocZ(allocator, key);
        defer allocator.free(key_w);
        const value_w = try std.unicode.wtf8ToWtf16LeAllocZ(allocator, value);
        defer allocator.free(value_w);
        try std.os.windows.SetEnvironmentVariable(key_w, value_w);
        return;
    }

    const EnvFns = struct {
        extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
    };

    const key_z = try allocator.allocSentinel(u8, key.len, 0);
    defer allocator.free(key_z);
    @memcpy(key_z[0..key.len], key);

    const value_z = try allocator.allocSentinel(u8, value.len, 0);
    defer allocator.free(value_z);
    @memcpy(value_z[0..value.len], value);

    if (EnvFns.setenv(key_z.ptr, value_z.ptr, 1) != 0) return error.Unexpected;
}

fn unsetEnvVarTest(allocator: std.mem.Allocator, key: []const u8) !void {
    if (builtin.os.tag == .windows) {
        const key_w = try std.unicode.utf8ToUtf16LeAllocZ(allocator, key);
        defer allocator.free(key_w);
        try std.os.windows.SetEnvironmentVariable(key_w, null);
        return;
    }

    const EnvFns = struct {
        extern "c" fn unsetenv(name: [*:0]const u8) c_int;
    };

    const key_z = try allocator.allocSentinel(u8, key.len, 0);
    defer allocator.free(key_z);
    @memcpy(key_z[0..key.len], key);

    if (EnvFns.unsetenv(key_z.ptr) != 0) return error.Unexpected;
}

test "getenv reflects updated value after setenv" {
    const allocator = std.testing.allocator;
    const key = "TALU_ENV_TEST_UPDATE";

    try setEnvVarTest(allocator, key, "abc");
    try std.testing.expectEqualStrings("abc", getenv(key).?);

    try setEnvVarTest(allocator, key, "xyz");
    try std.testing.expectEqualStrings("xyz", getenv(key).?);
}

test "getenv reflects unset after prior successful read" {
    const allocator = std.testing.allocator;
    const key = "TALU_ENV_TEST_UNSET";

    try setEnvVarTest(allocator, key, "abc");
    try std.testing.expectEqualStrings("abc", getenv(key).?);

    try unsetEnvVarTest(allocator, key);
    try std.testing.expect(getenv(key) == null);
}

test "getenv reflects newly set value after prior miss" {
    const allocator = std.testing.allocator;
    const key = "TALU_ENV_TEST_MISS_THEN_SET";

    try unsetEnvVarTest(allocator, key);
    try std.testing.expect(getenv(key) == null);

    try setEnvVarTest(allocator, key, "late");
    try std.testing.expectEqualStrings("late", getenv(key).?);
}
