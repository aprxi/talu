//! NVFP4 Converter
//!
//! Converts dense weight tensors into modelopt-compatible NVFP4 tensor
//! layout and rewrites config metadata accordingly.

const std = @import("std");
const log = @import("log_pkg");
const grouped_affine = @import("grouped_affine.zig");
const calibration_capture = @import("calibration_capture.zig");
const json = @import("io_pkg").json;
const repository = @import("io_pkg").repository.root;
const safetensors = @import("io_pkg").safetensors.root;
const tensor = @import("tensor_pkg");
const dtype = @import("dtype_pkg");

const nvfp4_group_size: usize = 16;
const global_scale_sample_limit: usize = 8192;
const activation_sample_cap: usize = 64;
const activation_importance_min_weight: f32 = 1e-6;
const small_model_preserve_threshold_params: u64 = 8_000_000_000;
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
    // Build canonical converted artifacts first (fusions/transforms/config paths),
    // then repack all eligible dense weights into NVFP4 directly to avoid
    // grouped-affine->NVFP4 transcoding loss.
    const output_path = try grouped_affine.convertToGroupedAffine(allocator, input_path, .{
        .quant = null,
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

    try augmentWithPackedNvfp4Companions(allocator, output_path, options.progress, options);
    try rewriteConfigToCanonical(allocator, output_path);
    return output_path;
}

fn augmentWithPackedNvfp4Companions(
    allocator: std.mem.Allocator,
    output_path: []const u8,
    progress: grouped_affine.ProgressContext,
    options: ConvertOptions,
) !void {
    var model_bundle = try repository.resolve(allocator, output_path, .{});
    defer model_bundle.deinit();

    const input_weights_path = try resolveInputWeightsPath(allocator, output_path);
    defer allocator.free(input_weights_path);
    const output_weights_path = try std.fs.path.join(allocator, &.{ output_path, "model.safetensors" });
    defer allocator.free(output_weights_path);

    var st = try safetensors.UnifiedSafeTensors.load(allocator, input_weights_path);
    defer st.deinit();

    const names = try st.tensorNames(allocator);
    defer allocator.free(names);

    var converted_weight_bases = std.StringHashMap(void).init(allocator);
    defer {
        var key_iter = converted_weight_bases.keyIterator();
        while (key_iter.next()) |key| allocator.free(key.*);
        converted_weight_bases.deinit();
    }

    var target_weights: usize = 0;
    var dense_weight_params_total: u64 = 0;
    var max_layer_index: ?u32 = null;
    for (names) |name| {
        const weight = st.getTensor(name, null) catch |err| {
            log.warn("convert", "NVFP4 scan tensor lookup failed", .{
                .tensor = name,
                .err = @errorName(err),
            });
            return err;
        };
        if (weight.n_dims == 2 and weight.shape[0] > 0 and weight.shape[1] > 0 and
            (weight.dtype == .f32 or weight.dtype == .f16 or weight.dtype == .bf16))
        {
            const rows: u64 = @intCast(weight.shape[0]);
            const cols: u64 = @intCast(weight.shape[1]);
            const tensor_params = std.math.mul(u64, rows, cols) catch std.math.maxInt(u64);
            dense_weight_params_total = std.math.add(u64, dense_weight_params_total, tensor_params) catch std.math.maxInt(u64);
        }
        if (extractLayerIndexFromTensorName(name)) |layer| {
            if (max_layer_index == null or layer > max_layer_index.?) max_layer_index = layer;
        }
    }

    const small_model_policy = makeSmallModelPreservePolicy(
        options.profile,
        dense_weight_params_total,
        max_layer_index,
    );
    if (small_model_policy.enabled) {
        log.info("convert", "NVFP4 small-model preserve policy enabled", .{
            .total_dense_params = dense_weight_params_total,
            .threshold = small_model_preserve_threshold_params,
            .last_layer = small_model_policy.last_layer_index orelse 0,
        });
    }

    for (names) |name| {
        const weight = st.getTensor(name, null) catch |err| {
            log.warn("convert", "NVFP4 scan tensor lookup failed", .{
                .tensor = name,
                .err = @errorName(err),
            });
            return err;
        };
        if (!shouldConvertDenseWeight(name, weight, options.profile)) continue;
        if (shouldPreserveWeightBySmallModelPolicy(name, small_model_policy)) continue;
        target_weights += 1;

        const base = name[0 .. name.len - ".weight".len];
        const owned_base = try allocator.dupe(u8, base);
        errdefer allocator.free(owned_base);
        try converted_weight_bases.put(owned_base, {});
    }
    if (target_weights == 0) return;

    var activation_cache = try captureActivationCacheBestEffort(
        allocator,
        output_path,
        model_bundle.tokenizer_path(),
        options,
    );
    defer if (activation_cache) |*cache| cache.deinit();
    const activation_sample_count = resolveActivationSampleCount(options);

    progress.addLine(1, "Packing NVFP4", target_weights, null, "weights");

    var output_specs = std.ArrayListUnmanaged(OutputTensorSpec){};
    defer deinitOutputSpecs(allocator, &output_specs);

    var packed_done: usize = 0;
    for (names) |name| {
        const weight = st.getTensor(name, null) catch |err| {
            log.warn("convert", "NVFP4 output-spec tensor lookup failed", .{
                .tensor = name,
                .err = @errorName(err),
            });
            return err;
        };
        if (shouldConvertDenseWeight(name, weight, options.profile) and !shouldPreserveWeightBySmallModelPolicy(name, small_model_policy)) {
            const base = name[0 .. name.len - ".weight".len];
            const scale_name = try std.fmt.allocPrint(allocator, "{s}.weight_scale", .{base});
            errdefer allocator.free(scale_name);
            const scale2_name = try std.fmt.allocPrint(allocator, "{s}.weight_scale_2", .{base});
            errdefer allocator.free(scale2_name);
            const input_scale_name = try std.fmt.allocPrint(allocator, "{s}.input_scale", .{base});
            errdefer allocator.free(input_scale_name);

            try output_specs.append(allocator, .{
                .kind = .converted_weight,
                .name = name,
                .source_weight_name = name,
                .owns_name = false,
            });
            try output_specs.append(allocator, .{
                .kind = .converted_scale,
                .name = scale_name,
                .source_weight_name = name,
                .owns_name = true,
            });
            try output_specs.append(allocator, .{
                .kind = .converted_scale_2,
                .name = scale2_name,
                .source_weight_name = name,
                .owns_name = true,
            });
            try output_specs.append(allocator, .{
                .kind = .converted_input_scale,
                .name = input_scale_name,
                .source_weight_name = name,
                .owns_name = true,
            });

            packed_done += 1;
            var msg_buf: [256]u8 = undefined;
            const copy_len = @min(name.len, msg_buf.len - 1);
            @memcpy(msg_buf[0..copy_len], name[0..copy_len]);
            msg_buf[copy_len] = 0;
            progress.updateLine(1, packed_done, @ptrCast(&msg_buf));
            continue;
        }
        if (shouldOmitGroupedAffineTensorForConvertedBase(&converted_weight_bases, name)) continue;

        try output_specs.append(allocator, .{
            .kind = .passthrough,
            .name = name,
            .source_weight_name = null,
            .owns_name = false,
        });
    }

    const tmp_weights_path = try std.fs.path.join(allocator, &.{ output_path, "model.safetensors.nvfp4.tmp" });
    defer allocator.free(tmp_weights_path);
    std.fs.cwd().deleteFile(tmp_weights_path) catch |err| switch (err) {
        error.FileNotFound => {},
        else => return err,
    };

    var out_file = try std.fs.cwd().createFile(tmp_weights_path, .{ .truncate = true });
    errdefer out_file.close();
    errdefer std.fs.cwd().deleteFile(tmp_weights_path) catch {};
    writeStreamedSafetensorsHeader(allocator, out_file, &st, output_specs.items) catch |err| {
        log.warn("convert", "NVFP4 streamed header write failed", .{ .err = @errorName(err) });
        return err;
    };
    writeStreamedSafetensorsData(
        allocator,
        out_file,
        &st,
        output_specs.items,
        options.profile,
        if (activation_cache) |*cache| cache else null,
        activation_sample_count,
        options.calib_seed,
    ) catch |err| {
        log.warn("convert", "NVFP4 streamed data write failed", .{ .err = @errorName(err) });
        return err;
    };
    try out_file.sync();
    out_file.close();

    std.fs.cwd().rename(tmp_weights_path, output_weights_path) catch |err| switch (err) {
        error.PathAlreadyExists => {
            try std.fs.cwd().deleteFile(output_weights_path);
            try std.fs.cwd().rename(tmp_weights_path, output_weights_path);
        },
        else => return err,
    };

    progress.completeLine(1);
    auditPackedNvfp4Output(allocator, output_weights_path, &converted_weight_bases) catch |err| {
        log.warn("convert", "NVFP4 pack audit failed", .{ .err = @errorName(err) });
        return err;
    };
}

const OutputTensorKind = enum {
    passthrough,
    converted_weight,
    converted_scale,
    converted_scale_2,
    converted_input_scale,
};

const OutputTensorSpec = struct {
    kind: OutputTensorKind,
    name: []const u8,
    source_weight_name: ?[]const u8,
    owns_name: bool,
};

fn deinitOutputSpecs(allocator: std.mem.Allocator, specs: *std.ArrayListUnmanaged(OutputTensorSpec)) void {
    for (specs.items) |spec| {
        if (spec.owns_name) allocator.free(spec.name);
    }
    specs.deinit(allocator);
}

fn resolveInputWeightsPath(allocator: std.mem.Allocator, output_path: []const u8) ![]u8 {
    const single_path = try std.fs.path.join(allocator, &.{ output_path, "model.safetensors" });
    errdefer allocator.free(single_path);
    if (std.fs.cwd().access(single_path, .{})) |_| {
        if (isValidSafetensorsSingleFile(single_path)) return single_path;
    } else |err| switch (err) {
        error.FileNotFound => {},
        else => return err,
    }

    allocator.free(single_path);
    const index_path = try std.fs.path.join(allocator, &.{ output_path, "model.safetensors.index.json" });
    errdefer allocator.free(index_path);
    if (std.fs.cwd().access(index_path, .{})) |_| return index_path else |err| switch (err) {
        error.FileNotFound => return error.WeightsNotFound,
        else => return err,
    }
}

fn isValidSafetensorsSingleFile(path: []const u8) bool {
    var file = std.fs.cwd().openFile(path, .{}) catch return false;
    defer file.close();

    const file_size = file.getEndPos() catch return false;
    if (file_size <= 8) return false;

    var len_buf: [8]u8 = undefined;
    const read_len = file.readAll(&len_buf) catch return false;
    if (read_len != len_buf.len) return false;
    const header_len = std.mem.readInt(u64, &len_buf, .little);
    if (header_len == 0) return false;
    return header_len <= (file_size - len_buf.len);
}

inline fn mix64(v: u64) u64 {
    var z = v +% 0x9e3779b97f4a7c15;
    z = (z ^ (z >> 30)) *% 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) *% 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

fn resolveActivationSampleCount(options: ConvertOptions) usize {
    const nsamples: usize = @intCast(@max(options.calib_nsamples, 1));
    const batch_size: usize = @intCast(@max(options.calib_batch_size, 1));
    const nblocks: usize = @intCast(@max(options.calib_nblocks, 1));
    const requested = std.math.mul(usize, nsamples, batch_size) catch activation_sample_cap;
    const expanded = std.math.mul(usize, requested, nblocks) catch activation_sample_cap;
    return @max(@as(usize, 8), @min(activation_sample_cap, expanded));
}

fn captureActivationCacheBestEffort(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    tokenizer_path: []const u8,
    options: ConvertOptions,
) !?calibration_capture.LayerActivationCache {
    if (options.calib_iters == 0) return null;
    if (!calibration_capture.isAvailable()) return null;

    const grouped_options: grouped_affine.ConvertOptions = .{
        .calib_iters = options.calib_iters,
        .calib_nsamples = options.calib_nsamples,
        .calib_seqlen = options.calib_seqlen,
        .calib_batch_size = options.calib_batch_size,
        .calib_nblocks = options.calib_nblocks,
        .calib_seed = options.calib_seed,
    };
    const prompt_tokens = try grouped_affine.loadCalibrationTokenPoolForConvert(
        allocator,
        tokenizer_path,
        grouped_options,
    ) orelse return null;
    defer allocator.free(prompt_tokens);

    const sample_count = resolveActivationSampleCount(options);
    const max_rows_per_key = @max(@as(usize, 64), sample_count * 2);
    var cache = calibration_capture.captureFromInference(
        allocator,
        model_path,
        prompt_tokens,
        .{
            .seed = options.calib_seed,
            .max_prompt_tokens = @max(
                @as(usize, 1),
                @min(@as(usize, @intCast(@max(options.calib_seqlen, 1))), @as(usize, 1024)),
            ),
            .max_rows_per_key = max_rows_per_key,
            .backend_selection = .auto,
        },
    ) catch |err| {
        log.warn("convert", "NVFP4 activation capture unavailable; using weight-only scaling", .{
            .err = @errorName(err),
        });
        return null;
    };
    if (cache.count() == 0) {
        cache.deinit();
        log.warn("convert", "NVFP4 activation capture empty; using weight-only scaling", .{});
        return null;
    }
    return cache;
}

fn extractLayerIndexFromTensorName(name: []const u8) ?u32 {
    const marker = "layers.";
    var search_start: usize = 0;
    while (std.mem.indexOfPos(u8, name, search_start, marker)) |pos| {
        const digits_start = pos + marker.len;
        if (digits_start >= name.len or !std.ascii.isDigit(name[digits_start])) {
            search_start = pos + 1;
            continue;
        }
        var digits_end = digits_start;
        while (digits_end < name.len and std.ascii.isDigit(name[digits_end])) : (digits_end += 1) {}
        if (digits_end >= name.len or name[digits_end] != '.') {
            search_start = pos + 1;
            continue;
        }
        return std.fmt.parseInt(u32, name[digits_start..digits_end], 10) catch null;
    }
    return null;
}

const SmallModelPreservePolicy = struct {
    enabled: bool = false,
    last_layer_index: ?u32 = null,
};

fn makeSmallModelPreservePolicy(
    profile: @TypeOf((grouped_affine.ConvertOptions{}).profile),
    dense_weight_params_total: u64,
    max_layer_index: ?u32,
) SmallModelPreservePolicy {
    switch (profile) {
        .good, .best => {},
        else => return .{},
    }
    if (dense_weight_params_total > small_model_preserve_threshold_params) return .{};
    return .{
        .enabled = true,
        .last_layer_index = max_layer_index,
    };
}

fn shouldPreserveWeightBySmallModelPolicy(
    weight_name: []const u8,
    policy: SmallModelPreservePolicy,
) bool {
    if (!policy.enabled) return false;
    if (std.mem.endsWith(u8, weight_name, "lm_head.weight")) return true;

    const layer_index = extractLayerIndexFromTensorName(weight_name) orelse return false;
    const last_layer = policy.last_layer_index orelse return false;
    if (!(layer_index == 0 or layer_index == last_layer)) return false;

    return std.mem.endsWith(u8, weight_name, ".self_attn.o_proj.weight") or
        std.mem.endsWith(u8, weight_name, ".linear_attn.out_proj.weight") or
        std.mem.endsWith(u8, weight_name, ".mlp.down_proj.weight") or
        std.mem.endsWith(u8, weight_name, ".mlp.fc2.weight");
}

fn activationRoleForTensorName(name: []const u8) calibration_capture.ActivationRole {
    if (std.mem.indexOf(u8, name, ".self_attn.q_proj.weight") != null or
        std.mem.indexOf(u8, name, ".self_attn.k_proj.weight") != null or
        std.mem.indexOf(u8, name, ".self_attn.v_proj.weight") != null or
        std.mem.indexOf(u8, name, ".linear_attn.in_proj_qkv.weight") != null or
        std.mem.indexOf(u8, name, ".linear_attn.in_proj_a.weight") != null)
    {
        return .attn_input;
    }
    if (std.mem.indexOf(u8, name, ".self_attn.o_proj.weight") != null or
        std.mem.indexOf(u8, name, ".linear_attn.out_proj.weight") != null)
    {
        return .attn_output;
    }
    if (std.mem.indexOf(u8, name, ".mlp.gate_proj.weight") != null or
        std.mem.indexOf(u8, name, ".mlp.up_proj.weight") != null or
        std.mem.indexOf(u8, name, ".mlp.fc1.weight") != null)
    {
        return .ffn_input;
    }
    if (std.mem.indexOf(u8, name, ".mlp.down_proj.weight") != null or
        std.mem.indexOf(u8, name, ".mlp.fc2.weight") != null)
    {
        return .ffn_output;
    }
    return .generic;
}

fn buildActivationGroupImportance(
    allocator: std.mem.Allocator,
    source_weight_name: []const u8,
    cols: usize,
    groups: usize,
    activation_cache: ?*const calibration_capture.LayerActivationCache,
    activation_sample_count: usize,
    calib_seed: u64,
) !?[]f32 {
    const cache = activation_cache orelse return null;
    const layer_index = extractLayerIndexFromTensorName(source_weight_name) orelse return null;
    if (activation_sample_count == 0 or groups == 0) return null;
    const role = activationRoleForTensorName(source_weight_name);
    const name_seed = mix64(calib_seed ^ std.hash.Wyhash.hash(0, source_weight_name));

    const sampled = calibration_capture.sampleLayerActivationsForRole(
        allocator,
        cache,
        layer_index,
        cols,
        activation_sample_count,
        name_seed,
        role,
    ) catch |err| {
        log.warn("convert", "NVFP4 activation sampling failed; using weight-only scaling", .{
            .tensor = source_weight_name,
            .err = @errorName(err),
        });
        return null;
    } orelse return null;
    defer sampled.deinit(allocator);

    const importance = try allocator.alloc(f32, groups);
    errdefer allocator.free(importance);
    @memset(importance, 0.0);

    for (0..sampled.sample_count) |sample_idx| {
        const row = sampled.values[sample_idx * cols .. (sample_idx + 1) * cols];
        for (0..groups) |g| {
            const offset = g * nvfp4_group_size;
            var sum_sq: f32 = 0.0;
            for (0..nvfp4_group_size) |i| {
                const value = row[offset + i];
                sum_sq += value * value;
            }
            importance[g] += sum_sq / @as(f32, @floatFromInt(nvfp4_group_size));
        }
    }

    const inv_samples = 1.0 / @as(f32, @floatFromInt(sampled.sample_count));
    var avg_importance: f32 = 0.0;
    for (importance) |*weight| {
        weight.* = @max(weight.* * inv_samples, activation_importance_min_weight);
        avg_importance += weight.*;
    }
    avg_importance /= @as(f32, @floatFromInt(groups));
    if (avg_importance > 0.0 and std.math.isFinite(avg_importance)) {
        for (importance) |*weight| {
            weight.* /= avg_importance;
        }
    }
    return importance;
}

fn shouldOmitGroupedAffineTensorForConvertedBase(
    converted_weight_bases: *const std.StringHashMap(void),
    name: []const u8,
) bool {
    if (std.mem.endsWith(u8, name, ".weight")) {
        const base = name[0 .. name.len - ".weight".len];
        return converted_weight_bases.contains(base);
    }
    if (std.mem.endsWith(u8, name, ".weight_scale")) {
        const base = name[0 .. name.len - ".weight_scale".len];
        return converted_weight_bases.contains(base);
    }
    if (std.mem.endsWith(u8, name, ".weight_scale_2")) {
        const base = name[0 .. name.len - ".weight_scale_2".len];
        return converted_weight_bases.contains(base);
    }
    if (std.mem.endsWith(u8, name, ".input_scale")) {
        const base = name[0 .. name.len - ".input_scale".len];
        return converted_weight_bases.contains(base);
    }
    return false;
}

fn shouldExcludeWeightByProfile(weight_name: []const u8, profile: @TypeOf((grouped_affine.ConvertOptions{}).profile)) bool {
    return switch (profile) {
        // `lm_head` is usually high-sensitivity; keep higher precision in
        // quality-oriented presets.
        .good, .best => std.mem.endsWith(u8, weight_name, "lm_head.weight"),
        else => false,
    };
}

fn shouldConvertDenseWeight(
    weight_name: []const u8,
    weight: tensor.Tensor,
    profile: @TypeOf((grouped_affine.ConvertOptions{}).profile),
) bool {
    if (!std.mem.endsWith(u8, weight_name, ".weight")) return false;
    if (weight.n_dims != 2) return false;
    if (weight.shape[0] <= 0 or weight.shape[1] <= 0) return false;
    if (shouldExcludeWeightByProfile(weight_name, profile)) return false;

    switch (weight.dtype) {
        .f32, .f16, .bf16 => {},
        else => return false,
    }

    const cols: usize = @intCast(weight.shape[1]);
    if ((cols % nvfp4_group_size) != 0) return false;
    return true;
}

fn fp4E2m1NibbleToF32(nibble: u8) f32 {
    return fp4_codebook[nibble & 0x0F];
}

fn maxFp8E4m3Positive() f32 {
    var max_val: f32 = 0.0;
    var code: u16 = 0;
    while (code <= 255) : (code += 1) {
        const value = dtype.fp8e4m3ToF32(@intCast(code));
        if (std.math.isFinite(value) and value > max_val) max_val = value;
    }
    return max_val;
}

fn scaledBlockMse(samples: []const f32, sample_weights: []const f32, global_scale: f32) f32 {
    if (samples.len == 0 or global_scale <= 0.0 or !std.math.isFinite(global_scale)) return std.math.inf(f32);
    if (sample_weights.len != samples.len) return std.math.inf(f32);

    var mse: f32 = 0.0;
    var total_weight: f32 = 0.0;
    for (samples, sample_weights) |ideal_scale, raw_weight| {
        if (ideal_scale <= 0.0 or !std.math.isFinite(ideal_scale)) continue;
        const weight = if (raw_weight > 0.0 and std.math.isFinite(raw_weight)) raw_weight else activation_importance_min_weight;
        const encoded = dtype.f32ToFp8E4M3(ideal_scale / global_scale);
        const decoded = dtype.fp8e4m3ToF32(encoded) * global_scale;
        const err = ideal_scale - decoded;
        mse += weight * (err * err);
        total_weight += weight;
    }
    if (!(total_weight > 0.0) or !std.math.isFinite(total_weight)) return std.math.inf(f32);
    return mse / total_weight;
}

fn chooseNvfp4GlobalScale(samples: []const f32, sample_weights: []const f32, observed_max_scale: f32) f32 {
    if (samples.len == 0 or observed_max_scale <= 0.0 or !std.math.isFinite(observed_max_scale)) return 1.0;
    if (sample_weights.len != samples.len) return 1.0;

    const fp8_max = maxFp8E4m3Positive();
    if (!(fp8_max > 0.0)) return 1.0;
    const base = @max(observed_max_scale / (fp8_max * 0.95), 1e-12);

    const candidates = [_]f32{
        1.0,
        base * 0.5,
        base * 0.66,
        base * 0.75,
        base * 0.9,
        base,
        base * 1.1,
        base * 1.25,
        base * 1.5,
        base * 2.0,
    };

    var best = candidates[0];
    var best_mse = std.math.inf(f32);
    for (candidates) |candidate| {
        if (!(candidate > 0.0) or !std.math.isFinite(candidate)) continue;
        const mse = scaledBlockMse(samples, sample_weights, candidate);
        if (mse < best_mse) {
            best_mse = mse;
            best = candidate;
        }
    }
    return best;
}

inline fn useAdvancedNvfp4Search(profile: @TypeOf((grouped_affine.ConvertOptions{}).profile)) bool {
    return profile == .best;
}

fn clipMultipliersForProfile(profile: @TypeOf((grouped_affine.ConvertOptions{}).profile)) []const f32 {
    return switch (profile) {
        .best => &[_]f32{ 0.9, 0.95, 1.0, 1.05, 1.1 },
        else => &[_]f32{1.0},
    };
}

fn collectSampledBlockScales(
    source: DenseWeightView,
    rows: usize,
    cols: usize,
    groups: usize,
    clip_multiplier: f32,
    group_importance: ?[]const f32,
    sampled_scales: *[global_scale_sample_limit]f32,
    sampled_importance: *[global_scale_sample_limit]f32,
    sampled_count: *usize,
    sampled_seen: *usize,
    observed_max_scale: *f32,
) void {
    sampled_count.* = 0;
    sampled_seen.* = 0;
    observed_max_scale.* = 0.0;

    var block_vals: [nvfp4_group_size]f32 = undefined;
    for (0..rows) |r| {
        for (0..groups) |g| {
            const group_start = g * nvfp4_group_size;
            var max_abs: f32 = 0.0;
            for (0..nvfp4_group_size) |i| {
                const col = group_start + i;
                const value = source.valueAt(r * cols + col);
                block_vals[i] = value;
                max_abs = @max(max_abs, @abs(value));
            }

            const block_scale = chooseNvfp4BlockScale(block_vals[0..], max_abs, clip_multiplier);
            const importance = if (group_importance) |weights| weights[g] else 1.0;
            observed_max_scale.* = @max(observed_max_scale.*, block_scale);
            if (sampled_count.* < global_scale_sample_limit) {
                sampled_scales[sampled_count.*] = block_scale;
                sampled_importance[sampled_count.*] = importance;
                sampled_count.* += 1;
            } else {
                const idx = sampled_seen.* % global_scale_sample_limit;
                sampled_scales[idx] = block_scale;
                sampled_importance[idx] = importance;
            }
            sampled_seen.* += 1;
        }
    }
}

fn estimateNvfp4ForwardProxyMse(
    source: DenseWeightView,
    rows: usize,
    cols: usize,
    groups: usize,
    global_scale: f32,
    clip_multiplier: f32,
    group_importance: ?[]const f32,
) f64 {
    if (!(global_scale > 0.0) or !std.math.isFinite(global_scale)) return std.math.inf(f64);

    var total_err: f64 = 0.0;
    var total_weight: f64 = 0.0;
    var block_vals: [nvfp4_group_size]f32 = undefined;

    for (0..rows) |r| {
        for (0..groups) |g| {
            const group_start = g * nvfp4_group_size;
            var max_abs: f32 = 0.0;
            for (0..nvfp4_group_size) |i| {
                const col = group_start + i;
                const value = source.valueAt(r * cols + col);
                block_vals[i] = value;
                max_abs = @max(max_abs, @abs(value));
            }

            const block_scale = chooseNvfp4BlockScale(block_vals[0..], max_abs, clip_multiplier);
            const packed_scale = dtype.f32ToFp8E4M3(block_scale / global_scale);
            const scale_f32 = dtype.fp8e4m3ToF32(packed_scale) * global_scale;
            const weight = if (group_importance) |weights| weights[g] else 1.0;
            const stable_weight = if (weight > 0.0 and std.math.isFinite(weight)) weight else activation_importance_min_weight;

            for (0..nvfp4_group_size) |i| {
                const value = block_vals[i];
                const scaled = if (scale_f32 > 0.0) value / scale_f32 else 0.0;
                const q = nearestFp4E2m1Nibble(scaled);
                const dq = fp4E2m1NibbleToF32(q) * scale_f32;
                const err = @as(f64, @floatCast(value - dq));
                total_err += err * err * @as(f64, stable_weight);
                total_weight += @as(f64, stable_weight);
            }
        }
    }
    if (!(total_weight > 0.0) or !std.math.isFinite(total_weight)) return std.math.inf(f64);
    return total_err / total_weight;
}

const DenseWeightView = union(enum) {
    f32: []align(1) const f32,
    f16: []align(1) const u16,
    bf16: []align(1) const u16,

    fn init(weight: tensor.Tensor) !DenseWeightView {
        const values = weight.numel;
        const bytes = weight.data();
        return switch (weight.dtype) {
            .f32 => blk: {
                const needed = std.math.mul(usize, values, @sizeOf(f32)) catch return error.InvalidShape;
                if (bytes.len < needed) return error.InvalidShape;
                break :blk .{ .f32 = @as([*]align(1) const f32, @ptrCast(bytes.ptr))[0..values] };
            },
            .f16 => blk: {
                const slice = weight.asSliceUnaligned(u16);
                if (slice.len < values) return error.InvalidShape;
                break :blk .{ .f16 = slice[0..values] };
            },
            .bf16 => blk: {
                const slice = weight.asSliceUnaligned(u16);
                if (slice.len < values) return error.InvalidShape;
                break :blk .{ .bf16 = slice[0..values] };
            },
            else => error.UnsupportedDType,
        };
    }

    fn valueAt(self: DenseWeightView, idx: usize) f32 {
        return switch (self) {
            .f32 => |slice| slice[idx],
            .f16 => |slice| dtype.fp16ToF32(slice[idx]),
            .bf16 => |slice| dtype.bf16ToF32(slice[idx]),
        };
    }
};

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

fn safetensorsDTypeString(dt: dtype.DType) []const u8 {
    return switch (dt) {
        .f32 => "F32",
        .f64 => "F64",
        .f16 => "F16",
        .bf16 => "BF16",
        .i8 => "I8",
        .i16 => "I16",
        .i32 => "I32",
        .i64 => "I64",
        .u8 => "U8",
        .u16 => "U16",
        .u32 => "U32",
        .u64 => "U64",
        .grouped_affine_u4, .grouped_affine_u8 => "U32",
        .mxfp4 => "U8",
        .f8_e4m3 => "F8_E4M3",
    };
}

fn safetensorsDTypeByteWidth(dt: dtype.DType) ?usize {
    return switch (dt) {
        .f64, .i64, .u64 => 8,
        .f32, .i32, .u32, .grouped_affine_u4, .grouped_affine_u8 => 4,
        .f16, .bf16, .i16, .u16 => 2,
        .f8_e4m3, .i8, .u8, .mxfp4 => 1,
    };
}

fn normalizePassthroughShape(t: tensor.Tensor, dims_out: *[tensor.MAX_NDIM]usize) !usize {
    if (t.n_dims < 0 or t.n_dims > tensor.MAX_NDIM) return error.InvalidConfig;
    const n_dims: usize = @intCast(t.n_dims);
    if (n_dims == 0) return 0;

    for (0..n_dims) |i| {
        const raw = t.shape[i];
        if (raw <= 0) return error.InvalidConfig;
        dims_out[i] = @intCast(raw);
    }

    const elem_bytes = safetensorsDTypeByteWidth(t.dtype) orelse return error.InvalidConfig;
    if (elem_bytes == 0 or t.data_size % elem_bytes != 0) return error.InvalidConfig;
    const expected_numel = t.data_size / elem_bytes;

    var shape_numel: usize = 1;
    for (0..n_dims) |i| {
        shape_numel = std.math.mul(usize, shape_numel, dims_out[i]) catch return error.InvalidConfig;
    }
    if (shape_numel == expected_numel) return n_dims;

    if (n_dims == 0) return error.InvalidConfig;
    var prefix_numel: usize = 1;
    for (0..(n_dims - 1)) |i| {
        prefix_numel = std.math.mul(usize, prefix_numel, dims_out[i]) catch return error.InvalidConfig;
    }
    if (prefix_numel == 0 or expected_numel % prefix_numel != 0) return error.InvalidConfig;

    const recovered_last = expected_numel / prefix_numel;
    if (recovered_last == 0) return error.InvalidConfig;
    dims_out[n_dims - 1] = recovered_last;
    return n_dims;
}

fn writeStreamedSafetensorsHeader(
    allocator: std.mem.Allocator,
    file: std.fs.File,
    st: *safetensors.UnifiedSafeTensors,
    specs: []const OutputTensorSpec,
) !void {
    var header_buf = std.ArrayListUnmanaged(u8){};
    defer header_buf.deinit(allocator);
    try header_buf.append(allocator, '{');

    var data_offset: usize = 0;
    for (specs, 0..) |spec, idx| {
        if (idx > 0) try header_buf.append(allocator, ',');
        switch (spec.kind) {
            .passthrough => {
                const t = try st.getTensor(spec.name, null);
                var dims_buf: [tensor.MAX_NDIM]usize = undefined;
                const n_dims = normalizePassthroughShape(t, &dims_buf) catch |err| {
                    log.warn("convert", "NVFP4 passthrough tensor shape invalid", .{
                        .tensor = spec.name,
                        .dtype = @tagName(t.dtype),
                        .n_dims = t.n_dims,
                        .data_size = t.data_size,
                        .err = @errorName(err),
                    });
                    return err;
                };
                try header_buf.writer(allocator).print("\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{
                    spec.name,
                    safetensorsDTypeString(t.dtype),
                });
                for (0..n_dims) |dim_idx| {
                    if (dim_idx > 0) try header_buf.append(allocator, ',');
                    try header_buf.writer(allocator).print("{d}", .{dims_buf[dim_idx]});
                }
                try header_buf.writer(allocator).print("],\"data_offsets\":[{d},{d}]}}", .{
                    data_offset,
                    data_offset + t.data_size,
                });
                data_offset += t.data_size;
            },
            .converted_weight, .converted_scale, .converted_scale_2, .converted_input_scale => {
                const source_name = spec.source_weight_name orelse return error.InvalidConfig;
                const source = try st.getTensor(source_name, null);
                if (source.n_dims != 2) return error.InvalidConfig;
                const rows: usize = @intCast(source.shape[0]);
                const cols: usize = @intCast(source.shape[1]);
                const groups = cols / nvfp4_group_size;
                switch (spec.kind) {
                    .converted_weight => {
                        const packed_cols = cols / 2;
                        const data_size = rows * packed_cols;
                        try header_buf.writer(allocator).print(
                            "\"{s}\":{{\"dtype\":\"U8\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}",
                            .{ spec.name, rows, packed_cols, data_offset, data_offset + data_size },
                        );
                        data_offset += data_size;
                    },
                    .converted_scale => {
                        const data_size = rows * groups;
                        try header_buf.writer(allocator).print(
                            "\"{s}\":{{\"dtype\":\"F8_E4M3\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}",
                            .{ spec.name, rows, groups, data_offset, data_offset + data_size },
                        );
                        data_offset += data_size;
                    },
                    .converted_scale_2, .converted_input_scale => {
                        const data_size = @sizeOf(f32);
                        try header_buf.writer(allocator).print(
                            "\"{s}\":{{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[{d},{d}]}}",
                            .{ spec.name, data_offset, data_offset + data_size },
                        );
                        data_offset += data_size;
                    },
                    else => unreachable,
                }
            },
        }
    }

    if (specs.len > 0) try header_buf.append(allocator, ',');
    try header_buf.appendSlice(allocator, "\"__metadata__\":{}");
    try header_buf.append(allocator, '}');

    const header_len = header_buf.items.len;
    const padded_len = (header_len + 7) & ~@as(usize, 7);
    const padding = padded_len - header_len;
    for (0..padding) |_| try header_buf.append(allocator, ' ');

    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, @intCast(padded_len), .little);
    try file.writeAll(&len_buf);
    try file.writeAll(header_buf.items);
}

const PackedNvfp4Data = struct {
    source_weight_name: []const u8,
    packed_weight: []u8,
    packed_scale: []u8,
    weight_scale_2: f32,
    input_scale: f32,

    fn deinit(self: *PackedNvfp4Data, allocator: std.mem.Allocator) void {
        allocator.free(self.packed_weight);
        allocator.free(self.packed_scale);
    }
};

fn writeStreamedSafetensorsData(
    allocator: std.mem.Allocator,
    file: std.fs.File,
    st: *safetensors.UnifiedSafeTensors,
    specs: []const OutputTensorSpec,
    profile: @TypeOf((grouped_affine.ConvertOptions{}).profile),
    activation_cache: ?*const calibration_capture.LayerActivationCache,
    activation_sample_count: usize,
    calib_seed: u64,
) !void {
    var packed_cache: ?PackedNvfp4Data = null;
    defer if (packed_cache) |*cache| cache.deinit(allocator);

    for (specs) |spec| {
        switch (spec.kind) {
            .passthrough => {
                const t = st.getTensor(spec.name, null) catch |err| {
                    log.warn("convert", "NVFP4 passthrough tensor lookup failed", .{
                        .tensor = spec.name,
                        .err = @errorName(err),
                    });
                    return err;
                };
                try file.writeAll(t.data()[0..t.data_size]);
            },
            .converted_weight, .converted_scale, .converted_scale_2, .converted_input_scale => {
                const source_name = spec.source_weight_name orelse return error.InvalidConfig;
                if (packed_cache == null or !std.mem.eql(u8, packed_cache.?.source_weight_name, source_name)) {
                    if (packed_cache != null) return error.InvalidConfig;
                    const source = st.getTensor(source_name, null) catch |err| {
                        log.warn("convert", "NVFP4 source tensor lookup failed", .{
                            .tensor = source_name,
                            .err = @errorName(err),
                        });
                        return err;
                    };
                    packed_cache = packDenseWeightToNvfp4(
                        allocator,
                        source_name,
                        source,
                        profile,
                        activation_cache,
                        activation_sample_count,
                        calib_seed,
                    ) catch |err| {
                        log.warn("convert", "NVFP4 dense-to-packed conversion failed", .{
                            .tensor = source_name,
                            .err = @errorName(err),
                        });
                        return err;
                    };
                }
                const cache = &packed_cache.?;
                switch (spec.kind) {
                    .converted_weight => try file.writeAll(cache.packed_weight),
                    .converted_scale => try file.writeAll(cache.packed_scale),
                    .converted_scale_2 => {
                        var scale2 = cache.weight_scale_2;
                        try file.writeAll(std.mem.asBytes(&scale2));
                    },
                    .converted_input_scale => {
                        var input_scale = cache.input_scale;
                        try file.writeAll(std.mem.asBytes(&input_scale));
                        cache.deinit(allocator);
                        packed_cache = null;
                    },
                    else => unreachable,
                }
            },
        }
    }

    if (packed_cache != null) return error.InvalidConfig;
}

fn packDenseWeightToNvfp4(
    allocator: std.mem.Allocator,
    source_weight_name: []const u8,
    weight: tensor.Tensor,
    profile: @TypeOf((grouped_affine.ConvertOptions{}).profile),
    activation_cache: ?*const calibration_capture.LayerActivationCache,
    activation_sample_count: usize,
    calib_seed: u64,
) !PackedNvfp4Data {
    if (!shouldConvertDenseWeight(source_weight_name, weight, profile)) return error.InvalidConfig;
    const rows: usize = @intCast(weight.shape[0]);
    const cols: usize = @intCast(weight.shape[1]);
    const groups = cols / nvfp4_group_size;
    const source = try DenseWeightView.init(weight);

    const packed_cols = cols / 2;
    const packed_len = std.math.mul(usize, rows, packed_cols) catch return error.InvalidShape;
    const scale_len = std.math.mul(usize, rows, groups) catch return error.InvalidShape;

    const packed_bytes = try allocator.alloc(u8, packed_len);
    errdefer allocator.free(packed_bytes);
    const packed_scales = try allocator.alloc(u8, scale_len);
    errdefer allocator.free(packed_scales);
    const input_scale: f32 = 1.0;

    const group_importance = try buildActivationGroupImportance(
        allocator,
        source_weight_name,
        cols,
        groups,
        activation_cache,
        activation_sample_count,
        calib_seed,
    );
    defer if (group_importance) |weights| allocator.free(weights);

    var sampled_scales: [global_scale_sample_limit]f32 = undefined;
    var sampled_importance: [global_scale_sample_limit]f32 = undefined;
    var global_scale: f32 = 1.0;
    var clip_multiplier: f32 = 1.0;
    if (useAdvancedNvfp4Search(profile)) {
        var best_global_scale: f32 = 1.0;
        var best_clip_multiplier: f32 = 1.0;
        var best_proxy_mse: f64 = std.math.inf(f64);
        var fallback_global_scale: f32 = 1.0;
        var have_fallback = false;

        const clip_candidates = clipMultipliersForProfile(profile);
        for (clip_candidates) |candidate_clip_multiplier| {
            var sampled_count: usize = 0;
            var sampled_seen: usize = 0;
            var observed_max_scale: f32 = 0.0;
            collectSampledBlockScales(
                source,
                rows,
                cols,
                groups,
                candidate_clip_multiplier,
                if (group_importance) |weights| weights else null,
                &sampled_scales,
                &sampled_importance,
                &sampled_count,
                &sampled_seen,
                &observed_max_scale,
            );
            if (sampled_count == 0) continue;

            const candidate_global_scale = chooseNvfp4GlobalScale(
                sampled_scales[0..sampled_count],
                sampled_importance[0..sampled_count],
                observed_max_scale,
            );
            if (!have_fallback) {
                fallback_global_scale = candidate_global_scale;
                have_fallback = true;
            }
            const candidate_proxy_mse = estimateNvfp4ForwardProxyMse(
                source,
                rows,
                cols,
                groups,
                candidate_global_scale,
                candidate_clip_multiplier,
                if (group_importance) |weights| weights else null,
            );
            if (candidate_proxy_mse < best_proxy_mse) {
                best_proxy_mse = candidate_proxy_mse;
                best_global_scale = candidate_global_scale;
                best_clip_multiplier = candidate_clip_multiplier;
            }
        }
        global_scale = if (std.math.isFinite(best_proxy_mse)) best_global_scale else fallback_global_scale;
        clip_multiplier = if (std.math.isFinite(best_proxy_mse)) best_clip_multiplier else @as(f32, 1.0);
    } else {
        var sampled_count: usize = 0;
        var sampled_seen: usize = 0;
        var observed_max_scale: f32 = 0.0;
        collectSampledBlockScales(
            source,
            rows,
            cols,
            groups,
            1.0,
            if (group_importance) |weights| weights else null,
            &sampled_scales,
            &sampled_importance,
            &sampled_count,
            &sampled_seen,
            &observed_max_scale,
        );
        if (sampled_count > 0) {
            global_scale = chooseNvfp4GlobalScale(
                sampled_scales[0..sampled_count],
                sampled_importance[0..sampled_count],
                observed_max_scale,
            );
        }
        clip_multiplier = 1.0;
    }

    var block_vals: [nvfp4_group_size]f32 = undefined;
    for (0..rows) |r| {
        for (0..groups) |g| {
            const group_start = g * nvfp4_group_size;

            var max_abs: f32 = 0.0;
            for (0..nvfp4_group_size) |i| {
                const col = group_start + i;
                const value = source.valueAt(r * cols + col);
                block_vals[i] = value;
                max_abs = @max(max_abs, @abs(value));
            }

            const block_scale = chooseNvfp4BlockScale(block_vals[0..], max_abs, clip_multiplier);
            packed_scales[r * groups + g] = dtype.f32ToFp8E4M3(block_scale / global_scale);
            const scale_f32 = dtype.fp8e4m3ToF32(packed_scales[r * groups + g]) * global_scale;

            for (0..(nvfp4_group_size / 2)) |pair_idx| {
                const lo_val = if (scale_f32 > 0.0) block_vals[pair_idx * 2] / scale_f32 else 0.0;
                const hi_val = if (scale_f32 > 0.0) block_vals[pair_idx * 2 + 1] / scale_f32 else 0.0;
                const lo = nearestFp4E2m1Nibble(lo_val);
                const hi = nearestFp4E2m1Nibble(hi_val);
                const dst_idx = r * packed_cols + g * (nvfp4_group_size / 2) + pair_idx;
                packed_bytes[dst_idx] = lo | (hi << 4);
            }
        }
    }

    return .{
        .source_weight_name = source_weight_name,
        .packed_weight = packed_bytes,
        .packed_scale = packed_scales,
        .weight_scale_2 = global_scale,
        .input_scale = input_scale,
    };
}

test "shouldConvertDenseWeight accepts 2D float weights with NVFP4-compatible columns" {
    var t = std.mem.zeroes(tensor.Tensor);
    t.dtype = .bf16;
    t.n_dims = 2;
    t.shape[0] = 64;
    t.shape[1] = 32;
    t.numel = 64 * 32;
    try std.testing.expect(shouldConvertDenseWeight("model.layers.0.self_attn.q_proj.weight", t, .good));
}

test "shouldConvertDenseWeight rejects unsupported shapes and dtypes" {
    var t = std.mem.zeroes(tensor.Tensor);
    t.dtype = .u8;
    t.n_dims = 2;
    t.shape[0] = 64;
    t.shape[1] = 32;
    t.numel = 64 * 32;
    try std.testing.expect(!shouldConvertDenseWeight("model.layers.0.self_attn.q_proj.weight", t, .good));

    t.dtype = .f16;
    t.n_dims = 1;
    t.shape[0] = 2048;
    t.shape[1] = 0;
    t.numel = 2048;
    try std.testing.expect(!shouldConvertDenseWeight("model.layers.0.self_attn.q_proj.weight", t, .good));

    t.n_dims = 2;
    t.shape[0] = 64;
    t.shape[1] = 30;
    t.numel = 64 * 30;
    try std.testing.expect(!shouldConvertDenseWeight("model.layers.0.self_attn.q_proj.weight", t, .good));

    t.shape[1] = 16;
    t.numel = 16 * 16;
    try std.testing.expect(shouldConvertDenseWeight("model.layers.0.self_attn.q_proj.weight", t, .good));
}

test "shouldConvertDenseWeight excludes lm_head for good profile" {
    var t = std.mem.zeroes(tensor.Tensor);
    t.dtype = .bf16;
    t.n_dims = 2;
    t.shape[0] = 64;
    t.shape[1] = 32;
    t.numel = 64 * 32;

    try std.testing.expect(!shouldConvertDenseWeight("lm_head.weight", t, .good));
    try std.testing.expect(shouldConvertDenseWeight("lm_head.weight", t, .custom));
}

test "DenseWeightView decodes f16 and bf16 values" {
    var f16_vals = [_]u16{ dtype.f32ToFp16(1.0), dtype.f32ToFp16(-2.0) };
    var bf16_vals = [_]u16{ dtype.f32ToBf16(0.5), dtype.f32ToBf16(-1.5) };

    var f16_tensor = std.mem.zeroes(tensor.Tensor);
    f16_tensor.dtype = .f16;
    f16_tensor.n_dims = 2;
    f16_tensor.shape[0] = 1;
    f16_tensor.shape[1] = 2;
    f16_tensor.numel = 2;
    f16_tensor.data_ptr = @ptrCast(std.mem.asBytes(&f16_vals).ptr);
    f16_tensor.data_size = std.mem.asBytes(&f16_vals).len;

    var bf16_tensor = std.mem.zeroes(tensor.Tensor);
    bf16_tensor.dtype = .bf16;
    bf16_tensor.n_dims = 2;
    bf16_tensor.shape[0] = 1;
    bf16_tensor.shape[1] = 2;
    bf16_tensor.numel = 2;
    bf16_tensor.data_ptr = @ptrCast(std.mem.asBytes(&bf16_vals).ptr);
    bf16_tensor.data_size = std.mem.asBytes(&bf16_vals).len;

    const f16_view = try DenseWeightView.init(f16_tensor);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), f16_view.valueAt(0), 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), f16_view.valueAt(1), 0.02);

    const bf16_view = try DenseWeightView.init(bf16_tensor);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), bf16_view.valueAt(0), 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, -1.5), bf16_view.valueAt(1), 0.02);
}

test "maxFp8E4m3Positive returns finite positive max" {
    const max_scale = maxFp8E4m3Positive();
    try std.testing.expect(max_scale > 0.0);
    try std.testing.expect(std.math.isFinite(max_scale));
}

test "chooseNvfp4GlobalScale improves block-scale reconstruction vs unit scale" {
    const samples = [_]f32{ 0.0012, 0.0048, 0.025, 0.17, 1.9, 6.7, 11.3 };
    const weights = [_]f32{1.0} ** samples.len;
    const chosen = chooseNvfp4GlobalScale(&samples, &weights, 11.3);
    try std.testing.expect(chosen > 0.0);
    try std.testing.expect(std.math.isFinite(chosen));

    const mse_unit = scaledBlockMse(&samples, &weights, 1.0);
    const mse_chosen = scaledBlockMse(&samples, &weights, chosen);
    try std.testing.expect(mse_chosen <= mse_unit);
}

test "chooseNvfp4GlobalScale returns fallback for invalid sample weights" {
    const samples = [_]f32{ 0.5, 1.0, 2.0 };
    const invalid_weights = [_]f32{ 1.0, 1.0 };
    try std.testing.expectEqual(@as(f32, 1.0), chooseNvfp4GlobalScale(&samples, &invalid_weights, 2.0));
}

test "clipMultipliersForProfile returns search window for best profile" {
    const clips = clipMultipliersForProfile(.best);
    try std.testing.expectEqual(@as(usize, 5), clips.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0.9), clips[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), clips[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.1), clips[4], 1e-6);
}

test "clipMultipliersForProfile returns fast path for good profile" {
    const clips = clipMultipliersForProfile(.good);
    try std.testing.expectEqual(@as(usize, 1), clips.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), clips[0], 1e-6);
}

test "useAdvancedNvfp4Search only enables best profile" {
    try std.testing.expect(useAdvancedNvfp4Search(.best));
    try std.testing.expect(!useAdvancedNvfp4Search(.good));
    try std.testing.expect(!useAdvancedNvfp4Search(.custom));
}

test "chooseNvfp4BlockScale responds to clip multiplier" {
    const block = [_]f32{6.0} ++ ([_]f32{0.0} ** (nvfp4_group_size - 1));
    const no_clip = chooseNvfp4BlockScale(&block, 6.0, 1.0);
    const clipped = chooseNvfp4BlockScale(&block, 6.0, 0.8);
    try std.testing.expect(clipped <= no_clip);
    try std.testing.expect(clipped > 0.0);
}

test "estimateNvfp4ForwardProxyMse is finite for simple tensor" {
    const values = [_]f32{
        1.0, -0.5, 0.25, 0.0,
        -1.0, 0.75, 0.5, -0.25,
        0.33, -0.66, 1.25, -1.5,
        0.1, 0.2, -0.3, 0.4,
    };
    const view: DenseWeightView = .{ .f32 = values[0..] };
    const mse = estimateNvfp4ForwardProxyMse(view, 1, 16, 1, 1.0, 1.0, null);
    try std.testing.expect(std.math.isFinite(mse));
    try std.testing.expect(mse >= 0.0);
}

test "extractLayerIndexFromTensorName parses layer identifiers" {
    try std.testing.expectEqual(
        @as(?u32, 17),
        extractLayerIndexFromTensorName("model.language_model.layers.17.self_attn.q_proj.weight"),
    );
    try std.testing.expectEqual(
        @as(?u32, null),
        extractLayerIndexFromTensorName("model.language_model.layer_norm.weight"),
    );
}

test "makeSmallModelPreservePolicy enables only for good/best small models" {
    const good_small = makeSmallModelPreservePolicy(.good, 4_000_000_000, 27);
    try std.testing.expect(good_small.enabled);
    try std.testing.expectEqual(@as(?u32, 27), good_small.last_layer_index);

    const best_large = makeSmallModelPreservePolicy(.best, 12_000_000_000, 63);
    try std.testing.expect(!best_large.enabled);

    const custom_small = makeSmallModelPreservePolicy(.custom, 4_000_000_000, 27);
    try std.testing.expect(!custom_small.enabled);
}

test "shouldPreserveWeightBySmallModelPolicy preserves lm_head and boundary outputs only" {
    const policy = SmallModelPreservePolicy{
        .enabled = true,
        .last_layer_index = 27,
    };
    try std.testing.expect(!shouldPreserveWeightBySmallModelPolicy("model.embed_tokens.weight", policy));
    try std.testing.expect(shouldPreserveWeightBySmallModelPolicy("lm_head.weight", policy));
    try std.testing.expect(shouldPreserveWeightBySmallModelPolicy("model.language_model.layers.0.self_attn.o_proj.weight", policy));
    try std.testing.expect(shouldPreserveWeightBySmallModelPolicy("model.language_model.layers.27.mlp.down_proj.weight", policy));
    try std.testing.expect(!shouldPreserveWeightBySmallModelPolicy("model.language_model.layers.13.mlp.down_proj.weight", policy));
    try std.testing.expect(!shouldPreserveWeightBySmallModelPolicy("model.language_model.layers.27.self_attn.q_proj.weight", policy));
}

test "activationRoleForTensorName maps linear attention and mlp tensors" {
    try std.testing.expectEqual(
        calibration_capture.ActivationRole.attn_input,
        activationRoleForTensorName("model.language_model.layers.0.linear_attn.in_proj_qkv.weight"),
    );
    try std.testing.expectEqual(
        calibration_capture.ActivationRole.attn_output,
        activationRoleForTensorName("model.language_model.layers.0.linear_attn.out_proj.weight"),
    );
    try std.testing.expectEqual(
        calibration_capture.ActivationRole.ffn_input,
        activationRoleForTensorName("model.language_model.layers.0.mlp.gate_proj.weight"),
    );
    try std.testing.expectEqual(
        calibration_capture.ActivationRole.ffn_output,
        activationRoleForTensorName("model.language_model.layers.0.mlp.down_proj.weight"),
    );
}

test "normalizePassthroughShape recovers trailing dim from data size" {
    var t = std.mem.zeroes(tensor.Tensor);
    t.dtype = .bf16;
    t.n_dims = 5;
    t.shape = .{ 1152, 3, 2, 16, 6128432240, 0, 0, 0 };
    t.data_size = 3_538_944; // 1152*3*2*16*16*2

    var dims: [tensor.MAX_NDIM]usize = undefined;
    const n_dims = try normalizePassthroughShape(t, &dims);
    try std.testing.expectEqual(@as(usize, 5), n_dims);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 1152, 3, 2, 16, 16 }, dims[0..n_dims]);
}

test "normalizePassthroughShape rejects incompatible size" {
    var t = std.mem.zeroes(tensor.Tensor);
    t.dtype = .bf16;
    t.n_dims = 5;
    t.shape = .{ 1152, 3, 2, 16, 6128432240, 0, 0, 0 };
    t.data_size = 3_538_943;

    var dims: [tensor.MAX_NDIM]usize = undefined;
    try std.testing.expectError(error.InvalidConfig, normalizePassthroughShape(t, &dims));
}

fn chooseNvfp4BlockScale(block_vals: []const f32, max_abs: f32, clip_multiplier: f32) f32 {
    if (max_abs <= 0.0) return 0.0;
    const clipped_abs = if (clip_multiplier > 0.0 and std.math.isFinite(clip_multiplier))
        max_abs * clip_multiplier
    else
        max_abs;
    const effective_abs = if (clipped_abs > 0.0 and std.math.isFinite(clipped_abs)) clipped_abs else max_abs;

    const candidate_scales = [_]f32{
        effective_abs / 6.0,
        effective_abs / 5.0,
        effective_abs / 4.5,
        effective_abs / 4.0,
        effective_abs / 3.5,
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

fn ensureScalarF32Tensor(t: tensor.Tensor) !void {
    if (t.dtype != .f32) return error.InvalidConfig;
    if (!(t.n_dims == 0 or (t.n_dims == 1 and t.shape[0] == 1))) return error.InvalidConfig;
}

fn auditPackedNvfp4Output(
    allocator: std.mem.Allocator,
    weights_path: []const u8,
    converted_weight_bases: *const std.StringHashMap(void),
) !void {
    var st = try safetensors.UnifiedSafeTensors.load(allocator, weights_path);
    defer st.deinit();

    var key_iter = converted_weight_bases.keyIterator();
    while (key_iter.next()) |base_ptr| {
        const base = base_ptr.*;

        const weight_name = try std.fmt.allocPrint(allocator, "{s}.weight", .{base});
        defer allocator.free(weight_name);
        const scale_name = try std.fmt.allocPrint(allocator, "{s}.weight_scale", .{base});
        defer allocator.free(scale_name);
        const scale2_name = try std.fmt.allocPrint(allocator, "{s}.weight_scale_2", .{base});
        defer allocator.free(scale2_name);
        const input_scale_name = try std.fmt.allocPrint(allocator, "{s}.input_scale", .{base});
        defer allocator.free(input_scale_name);
        const grouped_scales_name = try std.fmt.allocPrint(allocator, "{s}.scales", .{base});
        defer allocator.free(grouped_scales_name);
        const grouped_biases_name = try std.fmt.allocPrint(allocator, "{s}.biases", .{base});
        defer allocator.free(grouped_biases_name);
        const grouped_weight_bias_name = try std.fmt.allocPrint(allocator, "{s}.weight_bias", .{base});
        defer allocator.free(grouped_weight_bias_name);

        if (!st.hasTensor(weight_name)) return error.InvalidConfig;
        if (!st.hasTensor(scale_name)) return error.InvalidConfig;
        if (!st.hasTensor(scale2_name)) return error.InvalidConfig;
        if (!st.hasTensor(input_scale_name)) return error.InvalidConfig;
        if (st.hasTensor(grouped_scales_name)) return error.InvalidConfig;
        if (st.hasTensor(grouped_biases_name)) return error.InvalidConfig;
        if (st.hasTensor(grouped_weight_bias_name)) return error.InvalidConfig;

        const weight_tensor = try st.getTensor(weight_name, null);
        const scale_tensor = try st.getTensor(scale_name, null);
        const scale2_tensor = try st.getTensor(scale2_name, null);
        const input_scale_tensor = try st.getTensor(input_scale_name, null);

        if (weight_tensor.dtype != .u8 and weight_tensor.dtype != .i8) return error.InvalidConfig;
        if (weight_tensor.n_dims != 2) return error.InvalidConfig;
        if (weight_tensor.shape[0] <= 0 or weight_tensor.shape[1] <= 0) return error.InvalidConfig;

        if (scale_tensor.dtype != .f8_e4m3) return error.InvalidConfig;
        if (scale_tensor.n_dims != 2) return error.InvalidConfig;
        if (scale_tensor.shape[0] != weight_tensor.shape[0]) return error.InvalidConfig;
        if (scale_tensor.shape[1] <= 0) return error.InvalidConfig;

        try ensureScalarF32Tensor(scale2_tensor);
        try ensureScalarF32Tensor(input_scale_tensor);

        const packed_cols: usize = @intCast(weight_tensor.shape[1]);
        const unpacked_cols = std.math.mul(usize, packed_cols, 2) catch return error.InvalidConfig;
        if ((unpacked_cols % nvfp4_group_size) != 0) return error.InvalidConfig;
        const expected_scale_cols = unpacked_cols / nvfp4_group_size;
        if (scale_tensor.shape[1] != @as(i64, @intCast(expected_scale_cols))) return error.InvalidConfig;
    }
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
