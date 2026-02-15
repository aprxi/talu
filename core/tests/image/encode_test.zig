const std = @import("std");
const main = @import("main");
const image = main.core.image;

test "image.encode writes JPEG that decodes with same geometry" {
    const src_data = try std.testing.allocator.alloc(u8, 2 * 1 * 3);
    defer std.testing.allocator.free(src_data);

    src_data[0] = 255;
    src_data[1] = 0;
    src_data[2] = 0;
    src_data[3] = 0;
    src_data[4] = 0;
    src_data[5] = 255;

    const src: image.Image = .{
        .width = 2,
        .height = 1,
        .stride = 6,
        .format = .rgb8,
        .data = src_data,
    };

    const encoded = try image.encode(std.testing.allocator, src, .{
        .format = .jpeg,
        .jpeg_quality = 90,
    });
    defer std.testing.allocator.free(encoded);

    try std.testing.expectEqual(image.Format.jpeg, image.detectFormat(encoded).?);

    var decoded = try image.decode(std.testing.allocator, encoded, .{});
    defer decoded.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 2), decoded.width);
    try std.testing.expectEqual(@as(u32, 1), decoded.height);
}

test "image.encode writes PNG that decodes with same geometry" {
    const src_data = try std.testing.allocator.alloc(u8, 2 * 2 * 4);
    defer std.testing.allocator.free(src_data);
    @memset(src_data, 255);

    const src: image.Image = .{
        .width = 2,
        .height = 2,
        .stride = 8,
        .format = .rgba8,
        .data = src_data,
    };

    const encoded = try image.encode(std.testing.allocator, src, .{
        .format = .png,
    });
    defer std.testing.allocator.free(encoded);

    try std.testing.expectEqual(image.Format.png, image.detectFormat(encoded).?);

    var decoded = try image.decode(std.testing.allocator, encoded, .{});
    defer decoded.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 2), decoded.width);
    try std.testing.expectEqual(@as(u32, 2), decoded.height);
}

test "image.encode PNG rejects unsupported stride" {
    const src_data = try std.testing.allocator.alloc(u8, 8);
    defer std.testing.allocator.free(src_data);
    @memset(src_data, 0);

    const src: image.Image = .{
        .width = 2,
        .height = 1,
        .stride = 8,
        .format = .rgb8,
        .data = src_data,
    };

    try std.testing.expectError(
        error.UnsupportedStride,
        image.encode(std.testing.allocator, src, .{
            .format = .png,
        }),
    );
}
