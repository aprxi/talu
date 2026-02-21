//! Shared multimodal RoPE helpers for inference backends.

const std = @import("std");
const tensor = @import("../tensor.zig");
const vision_common = @import("vision_types.zig");

pub const PrefillVisionImage = vision_common.PrefillVisionImage;

pub fn applyPositionDelta(base_pos: usize, delta: isize) !usize {
    if (delta == 0) return base_pos;
    const shifted: i64 = @as(i64, @intCast(base_pos)) + @as(i64, delta);
    if (shifted < 0) return error.InvalidShape;
    return @as(usize, @intCast(shifted));
}

pub fn resolveMropeSection(config: *const tensor.ModelConfig, head_dim: usize) [3]usize {
    const half_dim = head_dim / 2;
    const parsed = config.rope_scaling.mrope_section;
    const parsed_total = @as(usize, parsed[0]) + @as(usize, parsed[1]) + @as(usize, parsed[2]);
    if (parsed_total == half_dim and parsed[1] > 0 and parsed[2] > 0) {
        return .{ parsed[0], parsed[1], parsed[2] };
    }
    return .{ 0, 0, 0 };
}

pub fn computePositionDelta(pos_t: []const u32, pos_h: []const u32, pos_w: []const u32) !isize {
    if (pos_t.len != pos_h.len or pos_t.len != pos_w.len) return error.InvalidShape;
    if (pos_t.len == 0) return 0;

    var max_pos: u32 = 0;
    for (0..pos_t.len) |idx| {
        max_pos = @max(max_pos, pos_t[idx]);
        max_pos = @max(max_pos, pos_h[idx]);
        max_pos = @max(max_pos, pos_w[idx]);
    }
    const delta_i64 = (@as(i64, max_pos) + 1) - @as(i64, @intCast(pos_t.len));
    return std.math.cast(isize, delta_i64) orelse error.InvalidShape;
}

pub fn buildMultimodalMropePositions(
    tokens: []const u32,
    images: []const PrefillVisionImage,
    image_token_id: u32,
    spatial_merge_size: usize,
    pos_t: []u32,
    pos_h: []u32,
    pos_w: []u32,
) !void {
    if (pos_t.len != tokens.len or pos_h.len != tokens.len or pos_w.len != tokens.len) return error.InvalidShape;
    if (spatial_merge_size == 0) return error.InvalidShape;

    var image_idx: usize = 0;
    var scan_pos: usize = 0;
    var write_pos: usize = 0;
    var next_base: u32 = 0;

    while (image_idx < images.len) {
        const image_start = std.mem.indexOfScalarPos(u32, tokens, scan_pos, image_token_id) orelse return error.InvalidPromptImageTokens;
        const text_len = image_start - scan_pos;

        for (0..text_len) |offset| {
            const step: u32 = @intCast(offset);
            const pos = next_base + step;
            pos_t[write_pos] = pos;
            pos_h[write_pos] = pos;
            pos_w[write_pos] = pos;
            write_pos += 1;
        }
        next_base += @intCast(text_len);

        const img = images[image_idx];
        const grid_t: usize = @intCast(img.grid.temporal);
        const grid_h: usize = @intCast(img.grid.height);
        const grid_w: usize = @intCast(img.grid.width);
        if ((grid_h % spatial_merge_size) != 0 or (grid_w % spatial_merge_size) != 0) return error.InvalidPromptImageTokens;

        const llm_h = grid_h / spatial_merge_size;
        const llm_w = grid_w / spatial_merge_size;
        const image_tokens = grid_t * llm_h * llm_w;
        if (image_tokens != img.token_count) return error.InvalidPromptImageTokens;

        for (0..grid_t) |t_idx| {
            for (0..llm_h) |h_idx| {
                for (0..llm_w) |w_idx| {
                    pos_t[write_pos] = next_base + @as(u32, @intCast(t_idx));
                    pos_h[write_pos] = next_base + @as(u32, @intCast(h_idx));
                    pos_w[write_pos] = next_base + @as(u32, @intCast(w_idx));
                    write_pos += 1;
                }
            }
        }

        const advance = @max(grid_t, @max(llm_h, llm_w));
        next_base += @as(u32, @intCast(advance));
        scan_pos = image_start + image_tokens;
        image_idx += 1;
    }

    if (std.mem.indexOfScalarPos(u32, tokens, scan_pos, image_token_id) != null) return error.InvalidPromptImageTokens;

    const trailing_len = tokens.len - scan_pos;
    for (0..trailing_len) |offset| {
        const step: u32 = @intCast(offset);
        const pos = next_base + step;
        pos_t[write_pos] = pos;
        pos_h[write_pos] = pos;
        pos_w[write_pos] = pos;
        write_pos += 1;
    }

    if (write_pos != tokens.len) return error.InvalidPromptImageTokens;
}

test "applyPositionDelta shifts base position and rejects negative result" {
    try std.testing.expectEqual(@as(usize, 7), try applyPositionDelta(5, 2));
    try std.testing.expectEqual(@as(usize, 3), try applyPositionDelta(5, -2));
    try std.testing.expectError(error.InvalidShape, applyPositionDelta(1, -3));
}

test "resolveMropeSection returns configured section only when valid" {
    var cfg = tensor.ModelConfig{
        .vocab_size = 10,
        .d_model = 8,
        .n_layers = 1,
        .n_heads = 1,
        .n_kv_groups = 1,
        .d_ff = 16,
        .max_seq_len = 16,
        .head_dim = 8,
        .rope_theta = 10000.0,
        .norm_eps = 1e-5,
        .gaffine_group_size = 128,
    };
    cfg.rope_scaling.mrope_section = .{ 1, 1, 2 }; // sum=4 == head_dim/2
    try std.testing.expectEqual([3]usize{ 1, 1, 2 }, resolveMropeSection(&cfg, 8));

    cfg.rope_scaling.mrope_section = .{ 1, 0, 3 }; // invalid (middle section zero)
    try std.testing.expectEqual([3]usize{ 0, 0, 0 }, resolveMropeSection(&cfg, 8));
}

test "computePositionDelta computes max-position offset" {
    const pos_t = [_]u32{ 0, 1, 1 };
    const pos_h = [_]u32{ 0, 2, 1 };
    const pos_w = [_]u32{ 0, 1, 3 };
    const delta = try computePositionDelta(&pos_t, &pos_h, &pos_w);
    // max component = 3, len=3 => (3+1)-3 = 1
    try std.testing.expectEqual(@as(isize, 1), delta);
}

test "buildMultimodalMropePositions writes text and image coordinates" {
    const tokens = [_]u32{ 101, 999, 999, 102 };
    const images = [_]PrefillVisionImage{
        .{
            .pixels = &.{},
            .width = 1,
            .height = 1,
            .grid = .{ .temporal = 1, .height = 1, .width = 2 },
            .token_count = 2,
        },
    };
    var pos_t = [_]u32{0} ** tokens.len;
    var pos_h = [_]u32{0} ** tokens.len;
    var pos_w = [_]u32{0} ** tokens.len;

    try buildMultimodalMropePositions(&tokens, &images, 999, 1, &pos_t, &pos_h, &pos_w);
    try std.testing.expectEqual(@as(u32, 0), pos_t[0]); // leading text
    try std.testing.expectEqual(@as(u32, 1), pos_t[1]); // image token 0
    try std.testing.expectEqual(@as(u32, 1), pos_h[1]);
    try std.testing.expectEqual(@as(u32, 2), pos_w[2]); // image token 1 increments W
    try std.testing.expectEqual(@as(u32, 3), pos_t[3]); // trailing text
}
