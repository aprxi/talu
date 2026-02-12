//! Integration tests for responses.ReasoningData
//!
//! ReasoningData is the payload for reasoning Items (chain-of-thought).

const std = @import("std");
const main = @import("main");
const ReasoningData = main.responses.ReasoningData;

test "ReasoningData is a struct" {
    const info = @typeInfo(ReasoningData);
    try std.testing.expect(info == .@"struct");
}

test "ReasoningData has expected fields" {
    const info = @typeInfo(ReasoningData);
    const fields = info.@"struct".fields;

    var has_content = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "content")) has_content = true;
    }

    try std.testing.expect(has_content);
}
