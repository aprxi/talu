const std = @import("std");

threadlocal var env_cache: std.StringHashMapUnmanaged([]const u8) = .{};

pub fn getenv(name: []const u8) ?[]const u8 {
    if (env_cache.get(name)) |cached| return cached;

    const allocator = std.heap.page_allocator;
    const value = std.process.getEnvVarOwned(allocator, name) catch return null;
    errdefer allocator.free(value);

    const key = allocator.dupe(u8, name) catch return null;
    errdefer allocator.free(key);

    env_cache.put(allocator, key, value) catch return null;
    return value;
}
