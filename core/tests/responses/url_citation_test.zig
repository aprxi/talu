//! Integration tests for responses.UrlCitation
//!
//! UrlCitation represents a web citation annotation in output text.

const std = @import("std");
const main = @import("main");
const UrlCitation = main.responses.UrlCitation;

test "UrlCitation is a struct" {
    const info = @typeInfo(UrlCitation);
    try std.testing.expect(info == .@"struct");
}

test "UrlCitation has expected fields" {
    const info = @typeInfo(UrlCitation);
    const fields = info.@"struct".fields;

    var has_url = false;
    var has_title = false;
    var has_start_index = false;
    var has_end_index = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "url")) has_url = true;
        if (comptime std.mem.eql(u8, field.name, "title")) has_title = true;
        if (comptime std.mem.eql(u8, field.name, "start_index")) has_start_index = true;
        if (comptime std.mem.eql(u8, field.name, "end_index")) has_end_index = true;
    }

    try std.testing.expect(has_url);
    try std.testing.expect(has_title);
    try std.testing.expect(has_start_index);
    try std.testing.expect(has_end_index);
}
