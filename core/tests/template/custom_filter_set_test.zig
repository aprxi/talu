//! Integration tests for text.template_engine.CustomFilterSet
//!
//! CustomFilterSet is a container for custom filters that can be passed to
//! the template evaluator. Custom filters allow extending the template engine
//! with Python callbacks or other external filter implementations.

const std = @import("std");
const main = @import("main");

const CustomFilterSet = main.template.CustomFilterSet;
const CustomFilter = main.template.CustomFilter;
const CustomFilterCallback = main.template.CustomFilterCallback;

// Dummy callback function for testing
fn dummyCallback(
    _: [*:0]const u8,
    _: [*:0]const u8,
    _: ?*anyopaque,
) callconv(.c) ?[*:0]u8 {
    return null;
}

// =============================================================================
// Type Verification Tests
// =============================================================================

test "CustomFilterSet type is accessible" {
    const T = CustomFilterSet;
    _ = T;
}

test "CustomFilterSet is a struct" {
    const info = @typeInfo(CustomFilterSet);
    try std.testing.expect(info == .@"struct");
}

test "CustomFilterSet has expected fields" {
    const info = @typeInfo(CustomFilterSet);
    const fields = info.@"struct".fields;

    var has_filters = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "filters")) has_filters = true;
    }

    try std.testing.expect(has_filters);
}

// =============================================================================
// Lifecycle Tests
// =============================================================================

test "CustomFilterSet.init creates empty set" {
    var set = CustomFilterSet.init();
    defer set.deinit(std.testing.allocator);

    try std.testing.expect(set.get("nonexistent") == null);
}

test "CustomFilterSet.deinit frees resources" {
    var set = CustomFilterSet.init();
    // Adding a filter to ensure deinit handles non-empty state
    try set.put(std.testing.allocator, "test", .{
        .callback = &dummyCallback,
        .user_data = null,
    });
    set.deinit(std.testing.allocator);
    // No leak = success
}

// =============================================================================
// Filter Storage Tests
// =============================================================================

test "CustomFilterSet.put stores filter by name" {
    var set = CustomFilterSet.init();
    defer set.deinit(std.testing.allocator);

    const filter = CustomFilter{
        .callback = &dummyCallback,
        .user_data = null,
    };

    try set.put(std.testing.allocator, "my_filter", filter);

    const retrieved = set.get("my_filter");
    try std.testing.expect(retrieved != null);
}

test "CustomFilterSet.put allows multiple filters" {
    var set = CustomFilterSet.init();
    defer set.deinit(std.testing.allocator);

    try set.put(std.testing.allocator, "filter1", .{ .callback = &dummyCallback, .user_data = null });
    try set.put(std.testing.allocator, "filter2", .{ .callback = &dummyCallback, .user_data = null });
    try set.put(std.testing.allocator, "filter3", .{ .callback = &dummyCallback, .user_data = null });

    try std.testing.expect(set.get("filter1") != null);
    try std.testing.expect(set.get("filter2") != null);
    try std.testing.expect(set.get("filter3") != null);
}

test "CustomFilterSet.put overwrites existing filter" {
    var set = CustomFilterSet.init();
    defer set.deinit(std.testing.allocator);

    // Create a sentinel value to distinguish filters
    var data1: u8 = 1;
    var data2: u8 = 2;

    try set.put(std.testing.allocator, "my_filter", .{
        .callback = &dummyCallback,
        .user_data = &data1,
    });

    try set.put(std.testing.allocator, "my_filter", .{
        .callback = &dummyCallback,
        .user_data = &data2,
    });

    const retrieved = set.get("my_filter").?;
    const ptr: *u8 = @ptrCast(@alignCast(retrieved.user_data.?));
    try std.testing.expectEqual(@as(u8, 2), ptr.*);
}

// =============================================================================
// Filter Lookup Tests
// =============================================================================

test "CustomFilterSet.get returns null for unknown filter" {
    var set = CustomFilterSet.init();
    defer set.deinit(std.testing.allocator);

    try std.testing.expect(set.get("unknown") == null);
}

test "CustomFilterSet.get retrieves stored filter" {
    var set = CustomFilterSet.init();
    defer set.deinit(std.testing.allocator);

    var user_data: u32 = 42;
    try set.put(std.testing.allocator, "my_filter", .{
        .callback = &dummyCallback,
        .user_data = &user_data,
    });

    const filter = set.get("my_filter").?;
    const ptr: *u32 = @ptrCast(@alignCast(filter.user_data.?));
    try std.testing.expectEqual(@as(u32, 42), ptr.*);
}

test "CustomFilterSet.get is case sensitive" {
    var set = CustomFilterSet.init();
    defer set.deinit(std.testing.allocator);

    try set.put(std.testing.allocator, "MyFilter", .{ .callback = &dummyCallback, .user_data = null });

    try std.testing.expect(set.get("MyFilter") != null);
    try std.testing.expect(set.get("myfilter") == null);
    try std.testing.expect(set.get("MYFILTER") == null);
}

// =============================================================================
// CustomFilter Type Tests
// =============================================================================

test "CustomFilter can store callback and user_data" {
    const filter = CustomFilter{
        .callback = &dummyCallback,
        .user_data = null,
    };

    try std.testing.expect(filter.callback == &dummyCallback);
    try std.testing.expect(filter.user_data == null);
}

test "CustomFilter user_data can be any pointer" {
    var my_context: u64 = 12345;
    const filter = CustomFilter{
        .callback = &dummyCallback,
        .user_data = &my_context,
    };

    const ptr: *u64 = @ptrCast(@alignCast(filter.user_data.?));
    try std.testing.expectEqual(@as(u64, 12345), ptr.*);
}
