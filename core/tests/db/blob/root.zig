//! Integration tests for the db.blob module.

const std = @import("std");

pub const blob_store = @import("blob_store_test.zig");
pub const blob_ref = @import("blob_ref_test.zig");
pub const blob_read_stream = @import("blob_read_stream_test.zig");

test {
    std.testing.refAllDecls(@This());
}
