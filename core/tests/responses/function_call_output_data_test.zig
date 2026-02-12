//! Integration tests for responses.FunctionCallOutputData
//!
//! FunctionCallOutputData is the payload for function_call_output Items.

const std = @import("std");
const main = @import("main");
const FunctionCallOutputData = main.responses.FunctionCallOutputData;
const Conversation = main.responses.Conversation;

test "FunctionCallOutputData is a struct" {
    const info = @typeInfo(FunctionCallOutputData);
    try std.testing.expect(info == .@"struct");
}

test "FunctionCallOutputData has expected fields" {
    const info = @typeInfo(FunctionCallOutputData);
    const fields = info.@"struct".fields;

    var has_call_id = false;
    var has_output = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "call_id")) has_call_id = true;
        if (comptime std.mem.eql(u8, field.name, "output")) has_output = true;
    }

    try std.testing.expect(has_call_id);
    try std.testing.expect(has_output);
}

test "FunctionCallOutputData created via Conversation" {
    const conv = try Conversation.init(std.testing.allocator);
    defer conv.deinit();

    const item = try conv.appendFunctionCallOutput("call_123", "Sunny, 72F");
    const fco = item.asFunctionCallOutput().?;

    try std.testing.expectEqualStrings("call_123", fco.call_id);
}
