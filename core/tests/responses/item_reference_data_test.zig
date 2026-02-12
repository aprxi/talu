//! Integration tests for responses.ItemReferenceData
//!
//! ItemReferenceData is the payload for item_reference Items.

const std = @import("std");
const main = @import("main");
const ItemReferenceData = main.responses.ItemReferenceData;

test "ItemReferenceData is a struct" {
    const info = @typeInfo(ItemReferenceData);
    try std.testing.expect(info == .@"struct");
}

test "ItemReferenceData has expected fields" {
    const info = @typeInfo(ItemReferenceData);
    const fields = info.@"struct".fields;

    var has_id = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "id")) has_id = true;
    }

    try std.testing.expect(has_id);
}
