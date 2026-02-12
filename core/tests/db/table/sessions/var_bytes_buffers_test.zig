//! Integration tests for db.table.sessions.VarBytesBuffers
//!
//! VarBytesBuffers is a variable-length byte buffer reader for columnar data.
//! Tests verify sliceForRow returns correct slices and deinit frees memory.

const std = @import("std");
const main = @import("main");
const db = main.db;

const VarBytesBuffers = db.table.sessions.VarBytesBuffers;

// ===== sliceForRow =====

test "VarBytesBuffers: sliceForRow returns correct slice for each row" {
    const alloc = std.testing.allocator;

    // Simulate 3 rows packed into a contiguous buffer:
    // row 0: "hello" (offset=0, length=5)
    // row 1: "world!" (offset=5, length=6)
    // row 2: "zig" (offset=11, length=3)
    const data = try alloc.dupe(u8, "helloworld!zig");
    const offsets = try alloc.dupe(u32, &[_]u32{ 0, 5, 11 });
    const lengths = try alloc.dupe(u32, &[_]u32{ 5, 6, 3 });

    var bufs = VarBytesBuffers{
        .data = data,
        .offsets = offsets,
        .lengths = lengths,
    };
    defer bufs.deinit(alloc);

    try std.testing.expectEqualStrings("hello", try bufs.sliceForRow(0));
    try std.testing.expectEqualStrings("world!", try bufs.sliceForRow(1));
    try std.testing.expectEqualStrings("zig", try bufs.sliceForRow(2));
}

test "VarBytesBuffers: sliceForRow returns error for out-of-bounds row" {
    const alloc = std.testing.allocator;

    const data = try alloc.dupe(u8, "abc");
    const offsets = try alloc.dupe(u32, &[_]u32{0});
    const lengths = try alloc.dupe(u32, &[_]u32{3});

    var bufs = VarBytesBuffers{
        .data = data,
        .offsets = offsets,
        .lengths = lengths,
    };
    defer bufs.deinit(alloc);

    try std.testing.expectError(error.InvalidColumnData, bufs.sliceForRow(1));
}

test "VarBytesBuffers: sliceForRow returns error when slice exceeds data bounds" {
    const alloc = std.testing.allocator;

    // offset + length would exceed data.len
    const data = try alloc.dupe(u8, "short");
    const offsets = try alloc.dupe(u32, &[_]u32{0});
    const lengths = try alloc.dupe(u32, &[_]u32{100}); // 100 > data.len

    var bufs = VarBytesBuffers{
        .data = data,
        .offsets = offsets,
        .lengths = lengths,
    };
    defer bufs.deinit(alloc);

    try std.testing.expectError(error.InvalidColumnData, bufs.sliceForRow(0));
}

test "VarBytesBuffers: sliceForRow handles zero-length row" {
    const alloc = std.testing.allocator;

    const data = try alloc.dupe(u8, "abc");
    const offsets = try alloc.dupe(u32, &[_]u32{ 0, 0 });
    const lengths = try alloc.dupe(u32, &[_]u32{ 3, 0 });

    var bufs = VarBytesBuffers{
        .data = data,
        .offsets = offsets,
        .lengths = lengths,
    };
    defer bufs.deinit(alloc);

    try std.testing.expectEqualStrings("abc", try bufs.sliceForRow(0));
    try std.testing.expectEqualStrings("", try bufs.sliceForRow(1));
}

// ===== deinit =====

test "VarBytesBuffers: deinit frees all allocations" {
    const alloc = std.testing.allocator;

    const data = try alloc.dupe(u8, "test-data");
    const offsets = try alloc.dupe(u32, &[_]u32{ 0, 4 });
    const lengths = try alloc.dupe(u32, &[_]u32{ 4, 5 });

    var bufs = VarBytesBuffers{
        .data = data,
        .offsets = offsets,
        .lengths = lengths,
    };

    // deinit should free all 3 slices — std.testing.allocator will
    // detect leaks if any allocation is missed.
    bufs.deinit(alloc);
}
