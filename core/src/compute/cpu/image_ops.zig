//! Spatial tensor preprocessing/layout primitives for CPU compute path.

const std = @import("std");

/// Extract fixed-size spatial blocks in row-major traversal order.
pub fn extractGridBlocksRowMajor(
    pixels: []const f32,
    height: usize,
    width: usize,
    grid_h: usize,
    grid_w: usize,
    grid_t: usize,
    patch_size: usize,
    temporal_patch_size: usize,
    out: []f32,
) !void {
    const total_frames = pixels.len / (3 * height * width);
    const patch_dim = 3 * temporal_patch_size * patch_size * patch_size;
    var patch_idx: usize = 0;

    for (0..grid_t) |t_block| {
        const frame_base = t_block * temporal_patch_size;
        for (0..grid_h) |patch_h| {
            for (0..grid_w) |patch_w| {
                const y0 = patch_h * patch_size;
                const x0 = patch_w * patch_size;

                var dst = patch_idx * patch_dim;
                for (0..temporal_patch_size) |tp| {
                    const frame_idx = frame_base + tp;
                    if (frame_idx >= total_frames) return error.InvalidImageDimensions;
                    for (0..patch_size) |py| {
                        for (0..patch_size) |px| {
                            const src_y = y0 + py;
                            const src_x = x0 + px;
                            for (0..3) |c| {
                                const src_idx = (((c * total_frames + frame_idx) * height + src_y) * width + src_x);
                                out[dst] = pixels[src_idx];
                                dst += 1;
                            }
                        }
                    }
                }
                patch_idx += 1;
            }
        }
    }
}

/// Extract fixed-size spatial blocks in merged traversal order.
pub fn extractGridBlocksMerged(
    pixels: []const f32,
    height: usize,
    width: usize,
    grid_h: usize,
    grid_w: usize,
    grid_t: usize,
    patch_size: usize,
    temporal_patch_size: usize,
    spatial_merge_size: usize,
    out: []f32,
) !void {
    const merged_h = grid_h / spatial_merge_size;
    const merged_w = grid_w / spatial_merge_size;
    const total_frames = pixels.len / (3 * height * width);
    const patch_dim = 3 * temporal_patch_size * patch_size * patch_size;
    var patch_idx: usize = 0;

    for (0..grid_t) |t_block| {
        const frame_base = t_block * temporal_patch_size;
        for (0..merged_h) |bh| {
            for (0..merged_w) |bw| {
                for (0..spatial_merge_size) |ih| {
                    for (0..spatial_merge_size) |iw| {
                        const patch_h = bh * spatial_merge_size + ih;
                        const patch_w = bw * spatial_merge_size + iw;
                        const y0 = patch_h * patch_size;
                        const x0 = patch_w * patch_size;

                        var dst = patch_idx * patch_dim;
                        for (0..3) |c| {
                            for (0..temporal_patch_size) |tp| {
                                const frame_idx = frame_base + tp;
                                if (frame_idx >= total_frames) return error.InvalidImageDimensions;
                                for (0..patch_size) |py| {
                                    for (0..patch_size) |px| {
                                        const src_y = y0 + py;
                                        const src_x = x0 + px;
                                        const src_idx = (((c * total_frames + frame_idx) * height + src_y) * width + src_x);
                                        out[dst] = pixels[src_idx];
                                        dst += 1;
                                    }
                                }
                            }
                        }
                        patch_idx += 1;
                    }
                }
            }
        }
    }
}

/// Bilinear interpolate one feature row from a 2D reference grid.
pub fn bilinearGridRow(
    pos_embed_f32: []const f32,
    num_grid_side: usize,
    feature_width: usize,
    h_idx: usize,
    w_idx: usize,
    grid_h: usize,
    grid_w: usize,
    row_major_mapping: bool,
    out_row: []f32,
) !void {
    const side_minus_1 = @as(f32, @floatFromInt(num_grid_side - 1));
    const hf = if (row_major_mapping) blk: {
        if (grid_h <= 1) break :blk 0.0;
        const scale = @as(f32, @floatFromInt(num_grid_side)) / @as(f32, @floatFromInt(grid_h));
        const mapped = (@as(f32, @floatFromInt(h_idx)) + 0.5) * scale - 0.5;
        break :blk std.math.clamp(mapped, 0.0, side_minus_1);
    } else if (grid_h <= 1)
        0.0
    else
        @as(f32, @floatFromInt(h_idx)) * side_minus_1 / @as(f32, @floatFromInt(grid_h - 1));

    const wf = if (row_major_mapping) blk: {
        if (grid_w <= 1) break :blk 0.0;
        const scale = @as(f32, @floatFromInt(num_grid_side)) / @as(f32, @floatFromInt(grid_w));
        const mapped = (@as(f32, @floatFromInt(w_idx)) + 0.5) * scale - 0.5;
        break :blk std.math.clamp(mapped, 0.0, side_minus_1);
    } else if (grid_w <= 1)
        0.0
    else
        @as(f32, @floatFromInt(w_idx)) * side_minus_1 / @as(f32, @floatFromInt(grid_w - 1));

    const h0 = @as(usize, @intFromFloat(@floor(hf)));
    const w0 = @as(usize, @intFromFloat(@floor(wf)));
    const h1 = @min(num_grid_side - 1, h0 + 1);
    const w1 = @min(num_grid_side - 1, w0 + 1);

    const dh = hf - @as(f32, @floatFromInt(h0));
    const dw = wf - @as(f32, @floatFromInt(w0));

    const w00 = (1.0 - dh) * (1.0 - dw);
    const w01 = (1.0 - dh) * dw;
    const w10 = dh * (1.0 - dw);
    const w11 = dh * dw;

    const idx00 = (h0 * num_grid_side + w0) * feature_width;
    const idx01 = (h0 * num_grid_side + w1) * feature_width;
    const idx10 = (h1 * num_grid_side + w0) * feature_width;
    const idx11 = (h1 * num_grid_side + w1) * feature_width;

    for (0..feature_width) |d| {
        out_row[d] = pos_embed_f32[idx00 + d] * w00 +
            pos_embed_f32[idx01 + d] * w01 +
            pos_embed_f32[idx10 + d] * w10 +
            pos_embed_f32[idx11 + d] * w11;
    }
}

/// Interpolate position features over traversal order.
pub fn interpolateGridEmbeddings(
    pos_embed_f32: []const f32,
    num_grid_side: usize,
    feature_width: usize,
    grid_h: usize,
    grid_w: usize,
    grid_t: usize,
    spatial_merge_size: usize,
    row_major_order: bool,
    row_major_mapping: bool,
    out: []f32,
) !void {
    const merged_h = grid_h / spatial_merge_size;
    const merged_w = grid_w / spatial_merge_size;
    const expected_tokens = grid_t * grid_h * grid_w;
    if (out.len != expected_tokens * feature_width) return error.InvalidShape;

    var dst_token: usize = 0;
    if (row_major_order) {
        for (0..grid_t) |_| {
            for (0..grid_h) |h_idx| {
                for (0..grid_w) |w_idx| {
                    try bilinearGridRow(
                        pos_embed_f32,
                        num_grid_side,
                        feature_width,
                        h_idx,
                        w_idx,
                        grid_h,
                        grid_w,
                        row_major_mapping,
                        out[dst_token * feature_width ..][0..feature_width],
                    );
                    dst_token += 1;
                }
            }
        }
        return;
    }

    for (0..grid_t) |_| {
        for (0..merged_h) |bh| {
            for (0..merged_w) |bw| {
                for (0..spatial_merge_size) |ih| {
                    for (0..spatial_merge_size) |iw| {
                        const h_idx = bh * spatial_merge_size + ih;
                        const w_idx = bw * spatial_merge_size + iw;
                        try bilinearGridRow(
                            pos_embed_f32,
                            num_grid_side,
                            feature_width,
                            h_idx,
                            w_idx,
                            grid_h,
                            grid_w,
                            row_major_mapping,
                            out[dst_token * feature_width ..][0..feature_width],
                        );
                        dst_token += 1;
                    }
                }
            }
        }
    }
}

/// Pack `[patch_tokens, hidden]` into `[merged_tokens, merge_units * hidden]`.
///
/// When `row_major_input` is false, source is expected in merge-block order.
/// When `row_major_input` is true, source is expected in row-major order per frame.
pub fn packMergedGridTokens(
    hidden: []const f32,
    grid_h: usize,
    grid_w: usize,
    grid_t: usize,
    feature_width: usize,
    spatial_merge_size: usize,
    row_major_input: bool,
    out: []f32,
) !void {
    if (spatial_merge_size == 0) return error.InvalidShape;
    if ((grid_h % spatial_merge_size) != 0 or (grid_w % spatial_merge_size) != 0) return error.InvalidShape;

    const merge_units = spatial_merge_size * spatial_merge_size;
    const patch_count = grid_t * grid_h * grid_w;
    const merged_h = grid_h / spatial_merge_size;
    const merged_w = grid_w / spatial_merge_size;
    const merged_tokens = grid_t * merged_h * merged_w;
    const merged_width = feature_width * merge_units;

    if (hidden.len != patch_count * feature_width) return error.InvalidShape;
    if (out.len != merged_tokens * merged_width) return error.InvalidShape;

    if (!row_major_input) {
        for (0..merged_tokens) |dst_token| {
            const dst_base = dst_token * merged_width;
            for (0..merge_units) |unit_idx| {
                const src_token = dst_token * merge_units + unit_idx;
                const src_base = src_token * feature_width;
                const dst_offset = dst_base + unit_idx * feature_width;
                @memcpy(
                    out[dst_offset..][0..feature_width],
                    hidden[src_base..][0..feature_width],
                );
            }
        }
        return;
    }

    const patches_per_frame = grid_h * grid_w;
    const merged_per_frame = merged_h * merged_w;

    for (0..grid_t) |t_idx| {
        const src_frame_base = t_idx * patches_per_frame;
        const dst_frame_base = t_idx * merged_per_frame;
        for (0..merged_h) |bh| {
            for (0..merged_w) |bw| {
                const dst_token = dst_frame_base + bh * merged_w + bw;
                const dst_base = dst_token * merged_width;
                var unit_idx: usize = 0;
                for (0..spatial_merge_size) |ih| {
                    for (0..spatial_merge_size) |iw| {
                        const row = bh * spatial_merge_size + ih;
                        const col = bw * spatial_merge_size + iw;
                        const src_token = src_frame_base + row * grid_w + col;
                        const src_base = src_token * feature_width;
                        const dst_offset = dst_base + unit_idx * feature_width;
                        @memcpy(
                            out[dst_offset..][0..feature_width],
                            hidden[src_base..][0..feature_width],
                        );
                        unit_idx += 1;
                    }
                }
            }
        }
    }
}

test "extractGridBlocksRowMajor extracts single patch in RGB order per pixel" {
    const pixels = [_]f32{
        1, 2, 3, 4, // C0
        5, 6, 7, 8, // C1
        9, 10, 11, 12, // C2
    };
    var out = [_]f32{0} ** 12;
    try extractGridBlocksRowMajor(&pixels, 2, 2, 1, 1, 1, 2, 1, &out);
    try std.testing.expectEqualSlices(f32, &[_]f32{
        1, 5, 9,
        2, 6, 10,
        3, 7, 11,
        4, 8, 12,
    }, &out);
}

test "extractGridBlocksMerged extracts merged traversal patches" {
    const pixels = [_]f32{
        1, 2, 3, 4, // C0
        5, 6, 7, 8, // C1
        9, 10, 11, 12, // C2
    };
    var out = [_]f32{0} ** (4 * 3);
    try extractGridBlocksMerged(&pixels, 2, 2, 2, 2, 1, 1, 1, 2, &out);
    try std.testing.expectEqualSlices(f32, &[_]f32{
        1, 5, 9,
        2, 6, 10,
        3, 7, 11,
        4, 8, 12,
    }, &out);
}

test "bilinearGridRow returns exact corner value on aligned grid" {
    const pos = [_]f32{
        1, 2,
        3, 4,
    };
    var out = [_]f32{0};
    try bilinearGridRow(&pos, 2, 1, 1, 0, 2, 2, false, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), out[0], 1e-6);
}

test "interpolateGridEmbeddings writes row-major sequence" {
    const pos = [_]f32{
        1, 2,
        3, 4,
    };
    var out = [_]f32{0} ** 4;
    try interpolateGridEmbeddings(&pos, 2, 1, 2, 2, 1, 1, true, false, &out);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1, 2, 3, 4 }, &out);
}

test "packMergedGridTokens packs row-major tokens into merged width" {
    const hidden = [_]f32{
        1, 10, // row0 col0
        2, 20, // row0 col1
        3, 30, // row1 col0
        4, 40, // row1 col1
    };
    var out = [_]f32{0} ** (1 * 8);
    try packMergedGridTokens(&hidden, 2, 2, 1, 2, 2, true, &out);
    try std.testing.expectEqualSlices(f32, &[_]f32{
        1, 10, 2, 20, 3, 30, 4, 40,
    }, &out);
}
