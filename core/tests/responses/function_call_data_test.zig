//! Integration tests for responses.FunctionCallData
//!
//! FunctionCallData is the payload for function_call Items.

const std = @import("std");
const main = @import("main");
const FunctionCallData = main.responses.FunctionCallData;
const Conversation = main.responses.Conversation;

test "FunctionCallData is a struct" {
    const info = @typeInfo(FunctionCallData);
    try std.testing.expect(info == .@"struct");
}

test "FunctionCallData has expected fields" {
    const info = @typeInfo(FunctionCallData);
    const fields = info.@"struct".fields;

    var has_name = false;
    var has_call_id = false;
    var has_arguments = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "name")) has_name = true;
        if (comptime std.mem.eql(u8, field.name, "call_id")) has_call_id = true;
        if (comptime std.mem.eql(u8, field.name, "arguments")) has_arguments = true;
    }

    try std.testing.expect(has_name);
    try std.testing.expect(has_call_id);
    try std.testing.expect(has_arguments);
}

test "FunctionCallData created via Conversation" {
    const conv = try Conversation.init(std.testing.allocator);
    defer conv.deinit();

    const item = try conv.appendFunctionCall("get_weather", "call_123", "{\"city\":\"NYC\"}");
    const fc = item.asFunctionCall().?;

    try std.testing.expectEqualStrings("get_weather", fc.name);
    try std.testing.expectEqualStrings("call_123", fc.call_id);
}
