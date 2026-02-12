//! Integration tests for responses.StoredItemEnvelope
//!
//! StoredItemEnvelope is a wrapper for ItemRecord storage with session_id.
//! Used for persistence and session restore.

const std = @import("std");
const main = @import("main");
const StoredItemEnvelope = main.responses.StoredItemEnvelope;
const ItemRecord = main.responses.ItemRecord;

test "StoredItemEnvelope is a struct" {
    const info = @typeInfo(StoredItemEnvelope);
    try std.testing.expect(info == .@"struct");
}

test "StoredItemEnvelope has expected fields" {
    const info = @typeInfo(StoredItemEnvelope);
    const fields = info.@"struct".fields;

    var has_session_id = false;
    var has_record = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "session_id")) has_session_id = true;
        if (comptime std.mem.eql(u8, field.name, "record")) has_record = true;
    }

    try std.testing.expect(has_session_id);
    try std.testing.expect(has_record);
}
