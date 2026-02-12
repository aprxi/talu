//! Integration tests for responses.ItemStatus
//!
//! ItemStatus tracks the lifecycle state of an Item.

const std = @import("std");
const main = @import("main");
const ItemStatus = main.responses.ItemStatus;

test "ItemStatus enum has expected variants" {
    const info = @typeInfo(ItemStatus);
    try std.testing.expect(info == .@"enum");

    var has_in_progress = false;
    var has_completed = false;
    var has_incomplete = false;
    var has_failed = false;

    inline for (info.@"enum".fields) |field| {
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
