//! Integration tests for ItemRecord
//!
//! ItemRecord is the self-contained item format for storage.
//! It provides a portable snapshot of an Item for persistence.

const std = @import("std");
const main = @import("main");
const ItemRecord = main.responses.ItemRecord;
const ItemType = main.responses.ItemType;
const MessageRole = main.responses.MessageRole;
const ItemStatus = main.responses.ItemStatus;

// =============================================================================
// Basic Construction Tests
// =============================================================================

test "ItemRecord has expected fields" {
    const info = @typeInfo(ItemRecord);
    try std.testing.expect(info == .@"struct");

    const fields = info.@"struct".fields;
    var has_item_id = false;
    var has_item_type = false;
    var has_created_at_ms = false;
    var has_variant = false;
    var has_metadata = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "item_id")) has_item_id = true;
        if (comptime std.mem.eql(u8, field.name, "item_type")) has_item_type = true;
        if (comptime std.mem.eql(u8, field.name, "created_at_ms")) has_created_at_ms = true;
        if (comptime std.mem.eql(u8, field.name, "variant")) has_variant = true;
        if (comptime std.mem.eql(u8, field.name, "metadata")) has_metadata = true;
    }

    try std.testing.expect(has_item_id);
    try std.testing.expect(has_item_type);
    try std.testing.expect(has_created_at_ms);
    try std.testing.expect(has_variant);
    try std.testing.expect(has_metadata);
}

// =============================================================================
// ItemType Tests
// =============================================================================

test "ItemType has expected values" {
    try std.testing.expectEqual(@as(u8, 0), @intFromEnum(ItemType.message));
    try std.testing.expectEqual(@as(u8, 1), @intFromEnum(ItemType.function_call));
    try std.testing.expectEqual(@as(u8, 2), @intFromEnum(ItemType.function_call_output));
    try std.testing.expectEqual(@as(u8, 3), @intFromEnum(ItemType.reasoning));
    try std.testing.expectEqual(@as(u8, 4), @intFromEnum(ItemType.item_reference));
}

test "ItemType values are distinct" {
    try std.testing.expect(ItemType.message != ItemType.function_call);
    try std.testing.expect(ItemType.function_call != ItemType.function_call_output);
    try std.testing.expect(ItemType.function_call_output != ItemType.reasoning);
    try std.testing.expect(ItemType.reasoning != ItemType.item_reference);
}

// =============================================================================
// ItemStatus Tests
// =============================================================================

test "ItemStatus has expected values" {
    const info = @typeInfo(ItemStatus);
    try std.testing.expect(info == .@"enum");

    const fields = info.@"enum".fields;
    var has_in_progress = false;
    var has_completed = false;
    var has_incomplete = false;
    var has_failed = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "in_progress")) has_in_progress = true;
        if (comptime std.mem.eql(u8, field.name, "completed")) has_completed = true;
        if (comptime std.mem.eql(u8, field.name, "incomplete")) has_incomplete = true;
        if (comptime std.mem.eql(u8, field.name, "failed")) has_failed = true;
    }

    try std.testing.expect(has_in_progress);
    try std.testing.expect(has_completed);
    try std.testing.expect(has_incomplete);
    try std.testing.expect(has_failed);
}

test "ItemStatus values are distinct" {
    try std.testing.expect(ItemStatus.in_progress != ItemStatus.completed);
    try std.testing.expect(ItemStatus.completed != ItemStatus.incomplete);
    try std.testing.expect(ItemStatus.incomplete != ItemStatus.failed);
}

// =============================================================================
// Usage Pattern Tests
// =============================================================================

test "ItemType works in switch statements" {
    const types = [_]ItemType{ .message, .function_call, .function_call_output, .reasoning, .item_reference };
    var counts = [_]u32{ 0, 0, 0, 0, 0 };

    for (types) |item_type| {
        switch (item_type) {
            .message => counts[0] += 1,
            .function_call => counts[1] += 1,
            .function_call_output => counts[2] += 1,
            .reasoning => counts[3] += 1,
            .item_reference => counts[4] += 1,
            .unknown => {},
        }
    }

    try std.testing.expectEqual(@as(u32, 1), counts[0]);
    try std.testing.expectEqual(@as(u32, 1), counts[1]);
    try std.testing.expectEqual(@as(u32, 1), counts[2]);
    try std.testing.expectEqual(@as(u32, 1), counts[3]);
    try std.testing.expectEqual(@as(u32, 1), counts[4]);
}

test "ItemType can be compared" {
    try std.testing.expect(ItemType.message == ItemType.message);
    try std.testing.expect(ItemType.message != ItemType.function_call);
    try std.testing.expect(ItemType.function_call != ItemType.reasoning);
}
