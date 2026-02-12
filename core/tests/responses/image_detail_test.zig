//! Integration tests for responses.ImageDetail
//!
//! ImageDetail specifies the resolution for image processing.

const std = @import("std");
const main = @import("main");
const ImageDetail = main.responses.ImageDetail;

test "ImageDetail is an enum" {
    const info = @typeInfo(ImageDetail);
    try std.testing.expect(info == .@"enum");
}

test "ImageDetail has expected variants" {
    const info = @typeInfo(ImageDetail);
    const fields = info.@"enum".fields;

    var has_low = false;
    var has_high = false;
    var has_auto = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "low")) has_low = true;
        if (comptime std.mem.eql(u8, field.name, "high")) has_high = true;
        if (comptime std.mem.eql(u8, field.name, "auto")) has_auto = true;
    }

    try std.testing.expect(has_low);
    try std.testing.expect(has_high);
    try std.testing.expect(has_auto);
}

test "ImageDetail values are distinct" {
    try std.testing.expect(ImageDetail.low != ImageDetail.high);
    try std.testing.expect(ImageDetail.high != ImageDetail.auto);
    try std.testing.expect(ImageDetail.low != ImageDetail.auto);
}
