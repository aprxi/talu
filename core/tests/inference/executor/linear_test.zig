//! Integration tests for inference.executor.Linear
//!
//! Tests the Linear type from the executor module.

const std = @import("std");
const main = @import("main");

const Linear = main.inference.executor.Linear;

test "Linear type is accessible" {
    const T = Linear;
    _ = T;
}

test "Linear has expected structure" {
    const info = @typeInfo(Linear);
    try std.testing.expect(info == .@"struct");

    const fields = info.@"struct".fields;
    var has_weight = false;
    var has_in_features = false;
    var has_out_features = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "weight")) has_weight = true;
        if (comptime std.mem.eql(u8, field.name, "in_features")) has_in_features = true;
        if (comptime std.mem.eql(u8, field.name, "out_features")) has_out_features = true;
    }

    try std.testing.expect(has_weight);
    try std.testing.expect(has_in_features);
    try std.testing.expect(has_out_features);
}
