//! NVFP4 Converter
//!
//! Converts grouped-affine q4 tensors into modelopt-compatible NVFP4 tensor
//! layout and rewrites config metadata accordingly.

const std = @import("std");
const grouped_affine = @import("grouped_affine.zig");
const json = @import("io_pkg").json;
const safetensors = @import("io_pkg").safetensors.root;
const tensor = @import("tensor_pkg");
const dtype = @import("dtype_pkg");

const nvfp4_group_size: usize = 16;
const fp4_codebook = [_]f32{
    0.0,  0.5,  1.0,  1.5,
    2.0,  3.0,  4.0,  6.0,
    -0.0, -0.5, -1.0, -1.5,
    -2.0, -3.0, -4.0, -6.0,
};

pub const modelIdFromOutputPath = grouped_affine.modelIdFromOutputPath;

pub const ConvertOptions = struct {
    output_dir: []const u8 = "models",
    destination: ?[]const u8 = null,
    output_suffix: ?[]const u8 = null,
    force: bool = false,
    max_shard_size: u64 = 0,
    progress: grouped_affine.ProgressContext = grouped_affine.ProgressContext.NONE,
    profile: @TypeOf((grouped_affine.ConvertOptions{}).profile) = .good,
    calib_iters: u32 = 300,
    calib_nsamples: u32 = 128,
    calib_seqlen: u32 = 2048,
    calib_batch_size: u32 = 1,
    calib_nblocks: u32 = 1,
    calib_seed: u64 = 42,
};

pub fn convertToNvfp4(
    allocator: std.mem.Allocator,
    input_path: []const u8,
    options: ConvertOptions,
) ![]const u8 {
    const output_path = try grouped_affine.convertToGroupedAffine(allocator, input_path, .{
        .quant = .{
            .bits = 4,
            .group_size = 32,
        },
        .output_dir = options.output_dir,
        .destination = options.destination,
        .output_suffix = options.output_suffix orelse "NVFP4",
        .force = options.force,
        .max_shard_size = options.max_shard_size,
        .progress = options.progress,
        .profile = options.profile,
        .calib_iters = options.calib_iters,
        .calib_nsamples = options.calib_nsamples,
        .calib_seqlen = options.calib_seqlen,
        .calib_batch_size = options.calib_batch_size,
        .calib_nblocks = options.calib_nblocks,
        .calib_seed = options.calib_seed,
    });
    errdefer allocator.free(output_path);
    try augmentWithPackedNvfp4Companions(allocator, output_path);
    try rewriteConfigToCanonical(allocator, output_path);
    return output_path;
}

fn augmentWithPackedNvfp4Companions(allocator: std.mem.Allocator, output_path: []const u8) !void {
    const weights_path = try std.fs.path.join(allocator, &.{ output_path, "model.safetensors" });
    defer allocator.free(weights_path);

    var st = try safetensors.UnifiedSafeTensors.load(allocator, weights_path);
    defer st.deinit();

    const names = try st.tensorNames(allocator);
    defer allocator.free(names);

    var builder = safetensors.Builder.init(allocator);
    defer builder.deinit();

    for (names) |name| {
        const weight = st.getTensor(name, null) catch continue;
        if (weight.dtype != .grouped_affine_u4) continue;
        if (!std.mem.endsWith(u8, name, ".weight")) continue;
        try addPackedNvfp4CompanionForWeight(allocator, &st, &builder, name, weight);
    }

    for (names) |name| {
        if (shouldOmitGroupedAffineTensor(&st, name)) continue;

        const t = try st.getTensor(name, null);
        const shape_array = t.shapeAsUsize();
        const shape = shape_array[0..@intCast(t.n_dims)];
        try builder.addTensor(name, t.dtype, shape, t.data()[0..t.data_size]);
    }

    var out_file = try std.fs.cwd().createFile(weights_path, .{ .truncate = true });
    out_file.close();
    try builder.save(output_path, "model.safetensors");
}

fn shouldOmitGroupedAffineTensor(st: *safetensors.UnifiedSafeTensors, name: []const u8) bool {
    if (std.mem.endsWith(u8, name, ".weight")) {
        const weight = st.getTensor(name, null) catch return false;
        return weight.dtype == .grouped_affine_u4;
    }

    if (std.mem.endsWith(u8, name, ".scales")) {
        const base = name[0 .. name.len - ".scales".len];
        var weight_name_buf: [512]u8 = undefined;
        const weight_name = std.fmt.bufPrint(&weight_name_buf, "{s}.weight", .{base}) catch return false;
        const weight = st.getTensor(weight_name, null) catch return false;
        return weight.dtype == .grouped_affine_u4;
    }

    if (std.mem.endsWith(u8, name, ".biases")) {
        const base = name[0 .. name.len - ".biases".len];
        var weight_name_buf: [512]u8 = undefined;
        const weight_name = std.fmt.bufPrint(&weight_name_buf, "{s}.weight", .{base}) catch return false;
        const weight = st.getTensor(weight_name, null) catch return false;
        return weight.dtype == .grouped_affine_u4;
    }

    return false;
}

fn fp4E2m1NibbleToF32(nibble: u8) f32 {
    return fp4_codebook[nibble & 0x0F];
}

const SourceLayout = struct {
    values_per_u32: usize,
    unpacked_cols: usize,
    source_group_size: usize,
};

fn selectSourceLayout(packed_word_cols: usize, source_group_count: usize, prefer_u8: bool) ?SourceLayout {
    if (packed_word_cols == 0 or source_group_count == 0) return null;

    var u4_layout: ?SourceLayout = null;
    var u8_layout: ?SourceLayout = null;
    const candidates = [_]usize{ 8, 4 };
    for (candidates) |values_per_u32| {
        const cols = std.math.mul(usize, packed_word_cols, values_per_u32) catch continue;
        if (cols % source_group_count != 0) continue;
        const source_group_size = cols / source_group_count;
        if (source_group_size < 8 or source_group_size > 256) continue;
        if ((source_group_size % 8) != 0) continue;
        const layout = SourceLayout{
            .values_per_u32 = values_per_u32,
            .unpacked_cols = cols,
            .source_group_size = source_group_size,
        };
        if (values_per_u32 == 8)
            u4_layout = layout
        else
            u8_layout = layout;
    }

    if (prefer_u8) {
        if (u8_layout) |layout| return layout;
        if (u4_layout) |layout| return layout;
        return null;
    }

    if (u4_layout) |layout| return layout;
    if (u8_layout) |layout| return layout;
    return null;
}

fn isEmbeddingWeight(weight_name: []const u8) bool {
    return std.mem.endsWith(u8, weight_name, "embed_tokens.weight") or
        std.mem.endsWith(u8, weight_name, "embedding.weight");
}

fn nearestFp4E2m1Nibble(value: f32) u8 {
    var best: u8 = 0;
    var best_err = std.math.inf(f32);
    var nibble: u8 = 0;
    while (nibble < 16) : (nibble += 1) {
        const err = @abs(value - fp4E2m1NibbleToF32(nibble));
        if (err < best_err) {
            best_err = err;
            best = nibble;
        }
    }
    return best;
}

fn addPackedNvfp4CompanionForWeight(
    allocator: std.mem.Allocator,
    st: *safetensors.UnifiedSafeTensors,
    builder: *safetensors.Builder,
    weight_name: []const u8,
    weight: tensor.Tensor,
) !void {
    const base = weight_name[0 .. weight_name.len - ".weight".len];

    const scales_name = try std.fmt.allocPrint(allocator, "{s}.scales", .{base});
    defer allocator.free(scales_name);
    const biases_name = try std.fmt.allocPrint(allocator, "{s}.biases", .{base});
    defer allocator.free(biases_name);

    const scales = try st.getTensor(scales_name, null);
    const biases = try st.getTensor(biases_name, null);
    if (weight.n_dims != 2 or scales.n_dims != 2 or biases.n_dims != 2) return;

    const rows: usize = @intCast(weight.shape[0]);
    const packed_word_cols: usize = @intCast(weight.shape[1]);
    const source_group_count: usize = @intCast(scales.shape[1]);
    if (source_group_count == 0) return;
    // grouped-affine U32 payload can represent either:
    // - u4 packing: 8 values per u32
    // - u8 packing: 4 values per u32
    // Infer source layout from packed width + scale columns and prefer the
    // embedding/u8 interpretation for embedding tables.
    const source_layout = selectSourceLayout(
        packed_word_cols,
        source_group_count,
        isEmbeddingWeight(weight_name),
    ) orelse return;
    const cols = source_layout.unpacked_cols;
    const source_values_per_u32 = source_layout.values_per_u32;
    const source_group_size = source_layout.source_group_size;
    if ((cols % nvfp4_group_size) != 0) return;
    const groups = cols / nvfp4_group_size;

    const src_words = weight.asSliceUnaligned(u32);
    const src_scales = scales.asSliceUnaligned(u16);
    const src_biases = biases.asSliceUnaligned(u16);

    const packed_bytes = try allocator.alloc(u8, rows * (cols / 2));
    defer allocator.free(packed_bytes);
    const packed_scales = try allocator.alloc(u8, rows * groups);
    defer allocator.free(packed_scales);
    const weight_scale_2 = [_]f32{1.0};
    const input_scale = [_]f32{1.0};

    var block_vals: [nvfp4_group_size]f32 = undefined;
    for (0..rows) |r| {
        for (0..groups) |g| {
            const grouped_affine_group = (g * nvfp4_group_size) / source_group_size;
            const scale = dtype.bf16ToF32(src_scales[r * source_group_count + grouped_affine_group]);
            const bias = dtype.bf16ToF32(src_biases[r * source_group_count + grouped_affine_group]);
            const group_start = g * nvfp4_group_size;

            var max_abs: f32 = 0.0;
            for (0..nvfp4_group_size) |i| {
                const col = group_start + i;
                const word = src_words[r * packed_word_cols + (col / source_values_per_u32)];
                const quant: u8 = if (source_values_per_u32 == 8)
                    @intCast((word >> @as(u5, @intCast((col % 8) * 4))) & 0xF)
                else
                    @intCast((word >> @as(u5, @intCast((col % 4) * 8))) & 0xFF);
                const value = @as(f32, @floatFromInt(quant)) * scale + bias;
                block_vals[i] = value;
                max_abs = @max(max_abs, @abs(value));
            }

            const block_scale = chooseNvfp4BlockScale(block_vals[0..], max_abs);
            packed_scales[r * groups + g] = dtype.f32ToFp8E4M3(block_scale);
            const scale_f32 = dtype.fp8e4m3ToF32(packed_scales[r * groups + g]);

            for (0..(nvfp4_group_size / 2)) |pair_idx| {
                const lo_val = if (scale_f32 > 0.0) block_vals[pair_idx * 2] / scale_f32 else 0.0;
                const hi_val = if (scale_f32 > 0.0) block_vals[pair_idx * 2 + 1] / scale_f32 else 0.0;
                const lo = nearestFp4E2m1Nibble(lo_val);
                const hi = nearestFp4E2m1Nibble(hi_val);
                const dst_idx = r * (cols / 2) + g * (nvfp4_group_size / 2) + pair_idx;
                packed_bytes[dst_idx] = lo | (hi << 4);
            }
        }
    }

    // modelopt/NVFP4 expects packed fp4 payload under "<base>.weight".
    try builder.addTensor(weight_name, .u8, &[_]usize{ rows, cols / 2 }, packed_bytes);

    const scale_name = try std.fmt.allocPrint(allocator, "{s}.weight_scale", .{base});
    defer allocator.free(scale_name);
    try builder.addTensor(scale_name, .f8_e4m3, &[_]usize{ rows, groups }, packed_scales);

    const scale2_name = try std.fmt.allocPrint(allocator, "{s}.weight_scale_2", .{base});
    defer allocator.free(scale2_name);
    try builder.addTensor(scale2_name, .f32, &[_]usize{}, std.mem.sliceAsBytes(&weight_scale_2));

    const input_scale_name = try std.fmt.allocPrint(allocator, "{s}.input_scale", .{base});
    defer allocator.free(input_scale_name);
    try builder.addTensor(input_scale_name, .f32, &[_]usize{}, std.mem.sliceAsBytes(&input_scale));
}

test "selectSourceLayout picks u8 for embedding when ambiguous" {
    const layout_opt = selectSourceLayout(640, 160, true);
    try std.testing.expect(layout_opt != null);
    const layout = layout_opt.?;
    try std.testing.expectEqual(@as(usize, 4), layout.values_per_u32);
    try std.testing.expectEqual(@as(usize, 2560), layout.unpacked_cols);
    try std.testing.expectEqual(@as(usize, 16), layout.source_group_size);
}

test "selectSourceLayout picks u4 for non-embedding when ambiguous" {
    const layout_opt = selectSourceLayout(320, 80, false);
    try std.testing.expect(layout_opt != null);
    const layout = layout_opt.?;
    try std.testing.expectEqual(@as(usize, 8), layout.values_per_u32);
    try std.testing.expectEqual(@as(usize, 2560), layout.unpacked_cols);
    try std.testing.expectEqual(@as(usize, 32), layout.source_group_size);
}

test "selectSourceLayout rejects invalid ratios" {
    try std.testing.expectEqual(@as(?SourceLayout, null), selectSourceLayout(640, 81, false));
}

fn chooseNvfp4BlockScale(block_vals: []const f32, max_abs: f32) f32 {
    if (max_abs <= 0.0) return 0.0;

    const candidate_scales = [_]f32{
        max_abs / 6.0,
        max_abs / 4.0,
    };

    var best_scale = candidate_scales[0];
    var best_mse = std.math.inf(f32);
    for (candidate_scales) |candidate_scale| {
        if (candidate_scale <= 0.0 or !std.math.isFinite(candidate_scale)) continue;
        var mse: f32 = 0.0;
        for (block_vals) |value| {
            const scaled = value / candidate_scale;
            const nibble = nearestFp4E2m1Nibble(scaled);
            const decoded = fp4E2m1NibbleToF32(nibble) * candidate_scale;
            const err = value - decoded;
            mse += err * err;
        }
        mse /= @as(f32, @floatFromInt(block_vals.len));
        if (mse < best_mse) {
            best_mse = mse;
            best_scale = candidate_scale;
        }
    }
    return best_scale;
}

pub fn rewriteConfigToCanonical(allocator: std.mem.Allocator, output_path: []const u8) !void {
    const config_path = try std.fs.path.join(allocator, &.{ output_path, "config.json" });
    defer allocator.free(config_path);

    const config_bytes = try std.fs.cwd().readFileAlloc(allocator, config_path, 1024 * 1024);
    defer allocator.free(config_bytes);

    var parsed = json.parseValue(allocator, config_bytes, .{
        .max_size_bytes = 1024 * 1024,
        .max_value_bytes = 1024 * 1024,
        .max_string_bytes = 256 * 1024,
    }) catch |err| {
        return switch (err) {
            error.InputTooLarge, error.InputTooDeep, error.StringTooLong, error.InvalidJson => error.InvalidConfig,
            else => err,
        };
    };
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidConfig;

    var output_buf = std.ArrayListUnmanaged(u8){};
    defer output_buf.deinit(allocator);
    try output_buf.append(allocator, '{');

    var first_field = true;
    var iter = parsed.value.object.iterator();
    while (iter.next()) |kv| {
        if (std.mem.eql(u8, kv.key_ptr.*, "quantization")) continue;
        if (std.mem.eql(u8, kv.key_ptr.*, "quantization_config")) continue;

        if (!first_field) try output_buf.append(allocator, ',');
        first_field = false;

        try output_buf.append(allocator, '"');
        try output_buf.appendSlice(allocator, kv.key_ptr.*);
        try output_buf.appendSlice(allocator, "\":");

        const value_json = try std.json.Stringify.valueAlloc(allocator, kv.value_ptr.*, .{});
        defer allocator.free(value_json);
        try output_buf.appendSlice(allocator, value_json);
    }

    if (!first_field) try output_buf.append(allocator, ',');
    try output_buf.appendSlice(
        allocator,
        "\"quantization_config\":{\"config_groups\":{\"group_0\":{\"input_activations\":{\"dynamic\":false,\"num_bits\":4,\"type\":\"float\",\"group_size\":16},\"weights\":{\"dynamic\":false,\"num_bits\":4,\"type\":\"float\",\"group_size\":16}}},\"bits\":4,\"quant_algo\":\"NVFP4\",\"kv_cache_scheme\":{\"dynamic\":false,\"num_bits\":8,\"type\":\"float\"},\"producer\":{\"name\":\"modelopt\",\"version\":\"0.37.0\"},\"quant_method\":\"modelopt\"}",
    );
    try output_buf.append(allocator, '}');

    var out_file = try std.fs.cwd().createFile(config_path, .{});
    defer out_file.close();
    try out_file.writeAll(output_buf.items);
}
