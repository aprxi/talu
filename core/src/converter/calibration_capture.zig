//! Inference-backed calibration activation capture.
//!
//! Builds deterministic per-layer activation matrices from a real scheduler
//! run and exposes layer/width keyed sampling for converter calibration.

const std = @import("std");
const build_options = @import("build_options");
const dtype_mod = @import("dtype_pkg");
const log = @import("log_pkg");
const xray = @import("xray_pkg");
const router_local = @import("../router/local.zig");
const backend_root = @import("inference_pkg").backend;
const progress_mod = @import("progress_pkg");
const xray_bridge_enabled: bool = if (@hasDecl(build_options, "xray_bridge")) build_options.xray_bridge else true;

pub const CaptureOptions = struct {
    seed: u64 = 42,
    max_prompt_tokens: usize = 128,
    max_rows_per_key: usize = 512,
    max_records: usize = 0,
    memory_limit_bytes: usize = 512 * 1024 * 1024,
    backend_selection: backend_root.Selection = .cpu,
};

pub const SampledActivations = struct {
    values: []f32,
    sample_count: usize,
    cols: usize,

    pub fn deinit(self: SampledActivations, allocator: std.mem.Allocator) void {
        allocator.free(self.values);
    }
};

pub const SampledActivationPair = struct {
    inputs: []f32,
    targets: []f32,
    sample_count: usize,
    input_cols: usize,
    output_cols: usize,

    pub fn deinit(self: SampledActivationPair, allocator: std.mem.Allocator) void {
        allocator.free(self.inputs);
        allocator.free(self.targets);
    }
};

pub const ActivationRole = enum(u8) {
    generic,
    attn_input,
    attn_output,
    ffn_input,
    ffn_output,
};

const ActivationKey = struct {
    layer: u16,
    cols: usize,
    point: xray.trace.TracePoint,
};

const ActivationMatrix = struct {
    values: []f32,
    rows: usize,
    cols: usize,

    fn deinit(self: ActivationMatrix, allocator: std.mem.Allocator) void {
        allocator.free(self.values);
    }
};

pub const LayerActivationCache = struct {
    allocator: std.mem.Allocator,
    map: std.AutoHashMap(ActivationKey, ActivationMatrix),

    pub fn init(allocator: std.mem.Allocator) LayerActivationCache {
        return .{
            .allocator = allocator,
            .map = std.AutoHashMap(ActivationKey, ActivationMatrix).init(allocator),
        };
    }

    pub fn deinit(self: *LayerActivationCache) void {
        var it = self.map.valueIterator();
        while (it.next()) |entry| entry.deinit(self.allocator);
        self.map.deinit();
    }

    pub fn count(self: *const LayerActivationCache) usize {
        return self.map.count();
    }

    pub fn has(self: *const LayerActivationCache, layer: u32, cols: usize) bool {
        if (layer > std.math.maxInt(u16)) return false;
        const layer_u16: u16 = @intCast(layer);
        var it = self.map.keyIterator();
        while (it.next()) |key| {
            if (key.layer == layer_u16 and key.cols == cols) return true;
        }
        return false;
    }

    fn appendRecordRows(
        self: *LayerActivationCache,
        key: ActivationKey,
        record: *const xray.capture.CapturedTensor,
        rows: usize,
        cols: usize,
        max_rows_per_key: usize,
    ) !void {
        if (rows == 0 or cols == 0 or max_rows_per_key == 0) return;
        const data = record.data orelse return;

        const take_rows = if (self.map.getPtr(key)) |existing|
            @min(rows, max_rows_per_key -| existing.rows)
        else
            @min(rows, max_rows_per_key);
        if (take_rows == 0) return;

        var matrix = if (self.map.fetchRemove(key)) |removed|
            removed.value
        else
            ActivationMatrix{
                .values = &.{},
                .rows = 0,
                .cols = cols,
            };
        errdefer matrix.deinit(self.allocator);

        if (matrix.cols != cols) return;
        const old_len = matrix.values.len;
        const new_rows = matrix.rows + take_rows;
        const new_len = new_rows * cols;
        matrix.values = if (old_len == 0)
            try self.allocator.alloc(f32, new_len)
        else
            try self.allocator.realloc(matrix.values, new_len);

        for (0..take_rows) |row_idx| {
            for (0..cols) |col_idx| {
                const src_elem = row_idx * cols + col_idx;
                const dst_elem = (matrix.rows + row_idx) * cols + col_idx;
                matrix.values[dst_elem] = readRecordElementF32(record.dtype, data, src_elem) orelse 0.0;
            }
        }
        matrix.rows = new_rows;
        try self.map.put(key, matrix);
    }

    pub fn ingestCapture(
        self: *LayerActivationCache,
        cap: *const xray.capture.TraceCapture,
        max_rows_per_key: usize,
        max_records: usize,
    ) !void {
        const total = if (max_records == 0)
            cap.records.items.len
        else
            @min(cap.records.items.len, max_records);

        for (cap.records.items[0..total]) |*record| {
            if (record.layer == xray.trace.TraceEmission.NO_LAYER) continue;
            if (record.data == null) continue;
            const shape = shapeRowsCols(record.ndim, record.shape) orelse continue;
            const rows = shape.rows;
            const cols = shape.cols;
            if (rows == 0 or cols == 0) continue;
            switch (record.dtype) {
                .f32, .f16, .bf16 => {},
                else => continue,
            }
            try self.appendRecordRows(.{
                .layer = record.layer,
                .cols = cols,
                .point = record.point,
            }, record, rows, cols, max_rows_per_key);
        }
    }
};

pub fn isAvailable() bool {
    return xray_bridge_enabled;
}

const RowsCols = struct {
    rows: usize,
    cols: usize,
};

fn shapeRowsCols(ndim: u8, shape: [4]u32) ?RowsCols {
    if (ndim == 0 or ndim > shape.len) return null;
    const cols = @as(usize, @intCast(shape[ndim - 1]));
    if (cols == 0) return null;

    var rows: usize = 1;
    var i: usize = 0;
    while (i + 1 < ndim) : (i += 1) {
        rows *= @as(usize, @intCast(shape[i]));
    }
    if (rows == 0) return null;
    return .{ .rows = rows, .cols = cols };
}

fn readRecordElementF32(dtype: xray.trace.DType, data: []const u8, elem_idx: usize) ?f32 {
    return switch (dtype) {
        .f32 => blk: {
            const start = elem_idx * 4;
            if (start + 4 > data.len) break :blk null;
            const bits = std.mem.readInt(u32, data[start..][0..4], .little);
            break :blk @bitCast(bits);
        },
        .f16 => blk: {
            const start = elem_idx * 2;
            if (start + 2 > data.len) break :blk null;
            const raw = std.mem.readInt(u16, data[start..][0..2], .little);
            break :blk dtype_mod.fp16ToF32(raw);
        },
        .bf16 => blk: {
            const start = elem_idx * 2;
            if (start + 2 > data.len) break :blk null;
            const raw = std.mem.readInt(u16, data[start..][0..2], .little);
            break :blk dtype_mod.bf16ToF32(raw);
        },
        else => null,
    };
}

inline fn mix64(v: u64) u64 {
    var z = v +% 0x9e3779b97f4a7c15;
    z = (z ^ (z >> 30)) *% 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) *% 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

pub fn sampleLayerActivations(
    allocator: std.mem.Allocator,
    cache: *const LayerActivationCache,
    layer: u32,
    cols: usize,
    sample_count: usize,
    seed: u64,
) !?SampledActivations {
    return sampleLayerActivationsForRole(
        allocator,
        cache,
        layer,
        cols,
        sample_count,
        seed,
        .generic,
    );
}

pub fn sampleLayerActivationsForRole(
    allocator: std.mem.Allocator,
    cache: *const LayerActivationCache,
    layer: u32,
    cols: usize,
    sample_count: usize,
    seed: u64,
    role: ActivationRole,
) !?SampledActivations {
    const attn_input_points = [_]xray.trace.TracePoint{ .layer_attn_norm, .layer_input };
    const attn_output_points = [_]xray.trace.TracePoint{ .attn_out, .block_out, .layer_input };
    const ffn_input_points = [_]xray.trace.TracePoint{ .layer_ffn_norm, .layer_input };
    const ffn_output_points = [_]xray.trace.TracePoint{ .ffn_act_mix, .ffn_act, .block_out };
    const generic_points = [_]xray.trace.TracePoint{ .layer_input, .layer_ffn_norm, .layer_attn_norm, .block_out };
    const preferred_points: []const xray.trace.TracePoint = switch (role) {
        .attn_input => attn_input_points[0..],
        .attn_output => attn_output_points[0..],
        .ffn_input => ffn_input_points[0..],
        .ffn_output => ffn_output_points[0..],
        .generic => generic_points[0..],
    };

    for (preferred_points) |point| {
        if (try sampleLayerActivationsForPoint(allocator, cache, layer, cols, sample_count, seed, point)) |sampled| {
            return sampled;
        }
    }
    return null;
}

pub fn sampleLayerActivationPairForPoints(
    allocator: std.mem.Allocator,
    cache: *const LayerActivationCache,
    layer: u32,
    input_cols: usize,
    output_cols: usize,
    sample_count: usize,
    seed: u64,
    input_point: xray.trace.TracePoint,
    output_point: xray.trace.TracePoint,
) !?SampledActivationPair {
    if (sample_count == 0 or input_cols == 0 or output_cols == 0) return null;
    if (layer > std.math.maxInt(u16)) return null;

    const layer_u16: u16 = @intCast(layer);
    const input_matrix = cache.map.get(.{
        .layer = layer_u16,
        .cols = input_cols,
        .point = input_point,
    }) orelse return null;
    const output_matrix = cache.map.get(.{
        .layer = layer_u16,
        .cols = output_cols,
        .point = output_point,
    }) orelse return null;

    const available_rows = @min(input_matrix.rows, output_matrix.rows);
    if (available_rows == 0) return null;
    const take = @min(sample_count, available_rows);
    if (take == 0) return null;

    const total_input = take * input_cols;
    const total_output = take * output_cols;
    const input_values = try allocator.alloc(f32, total_input);
    errdefer allocator.free(input_values);
    const target_values = try allocator.alloc(f32, total_output);
    errdefer allocator.free(target_values);

    const start = @as(
        usize,
        @intCast(
            mix64(
                seed ^
                    (@as(u64, layer_u16) *% 0x9e3779b97f4a7c15) ^
                    (@as(u64, @intFromEnum(input_point)) *% 0xbf58476d1ce4e5b9) ^
                    (@as(u64, @intFromEnum(output_point)) *% 0x94d049bb133111eb),
            ) % available_rows,
        ),
    );

    for (0..take) |row_idx| {
        const src_row = (start + row_idx) % available_rows;
        const in_src = input_matrix.values[src_row * input_cols .. (src_row + 1) * input_cols];
        const out_src = output_matrix.values[src_row * output_cols .. (src_row + 1) * output_cols];
        const in_dst = input_values[row_idx * input_cols .. (row_idx + 1) * input_cols];
        const out_dst = target_values[row_idx * output_cols .. (row_idx + 1) * output_cols];
        @memcpy(in_dst, in_src);
        @memcpy(out_dst, out_src);
    }

    return .{
        .inputs = input_values,
        .targets = target_values,
        .sample_count = take,
        .input_cols = input_cols,
        .output_cols = output_cols,
    };
}

fn sampleLayerActivationsForPoint(
    allocator: std.mem.Allocator,
    cache: *const LayerActivationCache,
    layer: u32,
    cols: usize,
    sample_count: usize,
    seed: u64,
    point: xray.trace.TracePoint,
) !?SampledActivations {
    if (sample_count == 0 or cols == 0) return null;
    if (layer > std.math.maxInt(u16)) return null;

    const matrix = cache.map.get(.{
        .layer = @intCast(layer),
        .cols = cols,
        .point = point,
    }) orelse return null;
    if (matrix.rows == 0 or matrix.cols != cols) return null;

    const total = sample_count * cols;
    const values = try allocator.alloc(f32, total);
    errdefer allocator.free(values);

    const start = @as(usize, @intCast(mix64(seed ^ (@as(u64, layer) *% 0x9e3779b97f4a7c15)) % matrix.rows));
    for (0..sample_count) |i| {
        const src_row = (start + i) % matrix.rows;
        const src = matrix.values[src_row * cols .. (src_row + 1) * cols];
        const dst = values[i * cols .. (i + 1) * cols];
        @memcpy(dst, src);
    }

    return .{
        .values = values,
        .sample_count = sample_count,
        .cols = cols,
    };
}

pub fn captureFromInference(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    prompt_tokens: []const u32,
    options: CaptureOptions,
) !LayerActivationCache {
    var cache = LayerActivationCache.init(allocator);
    errdefer cache.deinit();

    if (!xray_bridge_enabled) return cache;
    if (prompt_tokens.len == 0 or options.max_prompt_tokens == 0) return cache;

    var engine = try router_local.LocalEngine.initWithSeedAndResolutionConfig(
        allocator,
        model_path,
        options.seed,
        .{},
        .{ .selection = options.backend_selection },
        progress_mod.Context.NONE,
    );
    defer engine.deinit();

    var points = xray.TracePointSet.none();
    points.layer_input = true;
    points.layer_attn_norm = true;
    points.attn_q = true;
    points.attn_k = true;
    points.attn_v = true;
    points.attn_out = true;
    points.layer_ffn_norm = true;
    points.ffn_gate = true;
    points.ffn_up = true;
    points.ffn_down = true;
    points.ffn_act = true;
    points.ffn_act_mix = true;
    points.block_out = true;
    points.conv_in_proj = true;
    points.conv_conv = true;
    points.conv_out_proj = true;
    points.mamba_out = true;
    points.gdelta_in_proj = true;
    points.gdelta_norm = true;
    points.gdelta_out = true;

    var capture = xray.TraceCapture.init(allocator, .{
        .points = points,
        .mode = .full,
        .memory_limit = options.memory_limit_bytes,
    });
    defer capture.deinit();

    xray.enableCapture(&capture);
    defer xray.disableCapture();

    const prompt_len = @min(prompt_tokens.len, options.max_prompt_tokens);
    const prompt_slice = prompt_tokens[0..prompt_len];
    var scheduler = try engine.createScheduler(.{});
    defer scheduler.deinit();

    var result = try scheduler.generateSync(prompt_slice, 1, .{
        .sampling = .{
            .strategy = .greedy,
            .temperature = 0.0,
            .top_k = 1,
            .top_p = 1.0,
            .min_p = 0.0,
            .repetition_penalty = 1.0,
            .seed = options.seed,
        },
        .return_final_logits = false,
    });
    defer result.deinit(allocator);

    try cache.ingestCapture(&capture, options.max_rows_per_key, options.max_records);
    log.info("convert", "Inference-backed activation capture complete", .{
        .records = capture.records.items.len,
        .entries = cache.count(),
        .overflow = capture.overflow,
    });

    return cache;
}

test "shapeRowsCols flattens leading dimensions" {
    const rc = shapeRowsCols(3, .{ 2, 3, 16, 0 }).?;
    try std.testing.expectEqual(@as(usize, 6), rc.rows);
    try std.testing.expectEqual(@as(usize, 16), rc.cols);
}

test "sampleLayerActivationsForRole falls back across preferred points" {
    var cache = LayerActivationCache.init(std.testing.allocator);
    defer cache.deinit();

    const values = try std.testing.allocator.alloc(f32, 4);
    values[0] = 1.0;
    values[1] = 2.0;
    values[2] = 3.0;
    values[3] = 4.0;
    try cache.map.put(.{
        .layer = 0,
        .cols = 2,
        .point = .layer_input,
    }, .{
        .values = values,
        .rows = 2,
        .cols = 2,
    });

    const sampled = (try sampleLayerActivationsForRole(
        std.testing.allocator,
        &cache,
        0,
        2,
        2,
        42,
        .attn_input,
    )).?;
    defer sampled.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 2), sampled.sample_count);
    try std.testing.expectEqual(@as(usize, 2), sampled.cols);
}

test "sampleLayerActivationsForRole prefers role-specific point" {
    var cache = LayerActivationCache.init(std.testing.allocator);
    defer cache.deinit();

    const layer_input_vals = try std.testing.allocator.alloc(f32, 2);
    layer_input_vals[0] = 1.0;
    layer_input_vals[1] = 1.0;
    try cache.map.put(.{
        .layer = 0,
        .cols = 2,
        .point = .layer_input,
    }, .{
        .values = layer_input_vals,
        .rows = 1,
        .cols = 2,
    });

    const attn_out_vals = try std.testing.allocator.alloc(f32, 2);
    attn_out_vals[0] = 9.0;
    attn_out_vals[1] = 9.0;
    try cache.map.put(.{
        .layer = 0,
        .cols = 2,
        .point = .attn_out,
    }, .{
        .values = attn_out_vals,
        .rows = 1,
        .cols = 2,
    });

    const sampled = (try sampleLayerActivationsForRole(
        std.testing.allocator,
        &cache,
        0,
        2,
        1,
        1,
        .attn_output,
    )).?;
    defer sampled.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(f32, 9.0), sampled.values[0]);
    try std.testing.expectEqual(@as(f32, 9.0), sampled.values[1]);
}

test "sampleLayerActivationPairForPoints returns aligned rows" {
    var cache = LayerActivationCache.init(std.testing.allocator);
    defer cache.deinit();

    const in_values = try std.testing.allocator.alloc(f32, 6);
    in_values[0] = 1.0;
    in_values[1] = 2.0;
    in_values[2] = 3.0;
    in_values[3] = 4.0;
    in_values[4] = 5.0;
    in_values[5] = 6.0;
    try cache.map.put(.{
        .layer = 0,
        .cols = 2,
        .point = .layer_attn_norm,
    }, .{
        .values = in_values,
        .rows = 3,
        .cols = 2,
    });

    const out_values = try std.testing.allocator.alloc(f32, 9);
    out_values[0] = 10.0;
    out_values[1] = 11.0;
    out_values[2] = 12.0;
    out_values[3] = 13.0;
    out_values[4] = 14.0;
    out_values[5] = 15.0;
    out_values[6] = 16.0;
    out_values[7] = 17.0;
    out_values[8] = 18.0;
    try cache.map.put(.{
        .layer = 0,
        .cols = 3,
        .point = .attn_q,
    }, .{
        .values = out_values,
        .rows = 3,
        .cols = 3,
    });

    const pair = (try sampleLayerActivationPairForPoints(
        std.testing.allocator,
        &cache,
        0,
        2,
        3,
        2,
        42,
        .layer_attn_norm,
        .attn_q,
    )).?;
    defer pair.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 2), pair.sample_count);
    try std.testing.expectEqual(@as(usize, 2), pair.input_cols);
    try std.testing.expectEqual(@as(usize, 3), pair.output_cols);
}
