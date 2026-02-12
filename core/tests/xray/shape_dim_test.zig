//! Integration tests for xray.ShapeDim
//!
//! ShapeDim represents a dimension that can be either static (compile-time known)
//! or dynamic (.seq for sequence length). Used in KernelOp for flexible shape specs.

const std = @import("std");
const main = @import("main");
const xray = main.xray;

const ShapeDim = xray.ShapeDim;

test "ShapeDim: static formatting" {
    const dim: ShapeDim = .{ .static = 42 };
    var buf: [16]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);
    try dim.formatTo(stream.writer());
    try std.testing.expectEqualStrings("42", stream.getWritten());
}

test "ShapeDim: seq formatting" {
    const dim: ShapeDim = .seq;
    var buf: [16]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);
    try dim.formatTo(stream.writer());
    try std.testing.expectEqualStrings("seq", stream.getWritten());
}

test "ShapeDim: static value access" {
    const dim: ShapeDim = .{ .static = 1024 };

    // Can match and extract static value
    switch (dim) {
        .static => |val| try std.testing.expectEqual(@as(usize, 1024), val),
        .seq => unreachable,
    }
}

test "ShapeDim: seq is distinguishable from static" {
    const seq_dim: ShapeDim = .seq;
    const static_dim: ShapeDim = .{ .static = 1 };

    switch (seq_dim) {
        .seq => {}, // expected
        .static => unreachable,
    }

    switch (static_dim) {
        .static => {}, // expected
        .seq => unreachable,
    }
}

test "ShapeDim: can be used in arrays" {
    const dims = [_]ShapeDim{
        .seq,
        .{ .static = 128 },
        .{ .static = 256 },
    };

    try std.testing.expect(dims[0] == .seq);
    try std.testing.expectEqual(@as(usize, 128), dims[1].static);
    try std.testing.expectEqual(@as(usize, 256), dims[2].static);
}
