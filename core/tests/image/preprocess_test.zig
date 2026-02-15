const std = @import("std");
const main = @import("main");
const image = main.core.image;

test "image.smartResize aligns and enforces max pixels" {
    const out = try image.smartResize(10_000, 10_000, .{
        .factor = 32,
        .max_pixels = 1_000_000,
    });
    try std.testing.expect(out.width % 32 == 0);
    try std.testing.expect(out.height % 32 == 0);
    try std.testing.expect(@as(u64, out.width) * @as(u64, out.height) <= 1_000_000);
}

test "image.toPlanarF32 duplicates temporal frames with minus_one_to_one normalization" {
    const src = try std.testing.allocator.alloc(u8, 3);
    defer std.testing.allocator.free(src);
    src[0] = 255;
    src[1] = 0;
    src[2] = 128;

    const img: image.Image = .{
        .width = 1,
        .height = 1,
        .stride = 3,
        .format = .rgb8,
        .data = src,
    };

    const out = try image.toPlanarF32(std.testing.allocator, img, .{
        .temporal_frames = 2,
        .normalize = .minus_one_to_one,
    });
    defer std.testing.allocator.free(out);

    try std.testing.expectEqual(@as(usize, 6), out.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), out[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), out[3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.003921628), out[4], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.003921628), out[5], 1e-6);
}

test "image.calculateVisionGrid and image.calculateMergedTokenCount match expected tokens" {
    const grid = try image.calculateVisionGrid(
        224,
        224,
        16,
        2,
        2,
    );
    try std.testing.expectEqual(@as(u32, 1), grid.temporal);
    try std.testing.expectEqual(@as(u32, 14), grid.height);
    try std.testing.expectEqual(@as(u32, 14), grid.width);

    const tokens = try image.calculateMergedTokenCount(grid, 2);
    try std.testing.expectEqual(@as(u32, 49), tokens);
}

test "image.calculateTokenCountForImage supports smart resize path" {
    const tokens = try image.calculateTokenCountForImage(3840, 2160, .{
        .patch_size = 16,
        .spatial_merge_size = 2,
        .temporal_frames = 2,
        .temporal_patch_size = 2,
        .smart_resize = .{
            .factor = 32,
            .min_pixels = 4 * 1024,
            .max_pixels = 2 * 1024 * 1024,
        },
    });
    try std.testing.expect(tokens > 0);
}

test "image.preprocessImage returns tensor, grid and token count" {
    const src_buf = try std.testing.allocator.alloc(u8, 3);
    defer std.testing.allocator.free(src_buf);
    src_buf[0] = 255;
    src_buf[1] = 0;
    src_buf[2] = 0;

    const src: image.Image = .{
        .width = 1,
        .height = 1,
        .stride = 3,
        .format = .rgb8,
        .data = src_buf,
    };

    var result = try image.preprocessImage(std.testing.allocator, src, .{
        .normalize = .minus_one_to_one,
        .temporal_frames = 2,
        .patch_size = 16,
        .temporal_patch_size = 2,
        .spatial_merge_size = 2,
        .explicit_resize = .{
            .width = 32,
            .height = 32,
            .fit = .stretch,
            .filter = .nearest,
        },
    });
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u32, 32), result.width);
    try std.testing.expectEqual(@as(u32, 32), result.height);
    try std.testing.expectEqual(@as(u32, 1), result.grid.temporal);
    try std.testing.expectEqual(@as(u32, 2), result.grid.height);
    try std.testing.expectEqual(@as(u32, 2), result.grid.width);
    try std.testing.expectEqual(@as(u32, 1), result.token_count);
    try std.testing.expectEqual(@as(usize, 3 * 2 * 32 * 32), result.pixels.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result.pixels[0], 1e-6);
}
