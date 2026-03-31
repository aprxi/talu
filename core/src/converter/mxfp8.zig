//! MXFP8 Model Conversion
//!
//! Converts transformer models to MXFP8 quantization: E4M3 data + UE8M0
//! 1×32 block scales. This format is native to Blackwell (sm_120) tensor
//! cores via cuBLASLt block-scaled GEMM.

const std = @import("std");
const log = @import("../log.zig");
const tensor = @import("../tensor.zig");
const dtype_mod = @import("../dtype.zig");
const safetensors = @import("../io/safetensors/root.zig");
const repository = @import("../io/repository/root.zig");
const gaf_paths = @import("gaf_paths.zig");
const config_loader = @import("../models/config/root.zig");
const op_types = @import("../models/op_types.zig");
const parallel = @import("../system/parallel.zig");
const convert = @import("root.zig");
const models_registry = @import("../models/registry.zig");
const load_transforms = @import("../models/load/transforms.zig");
const json = @import("../io/json/root.zig");

const Tensor = tensor.Tensor;
const DType = dtype_mod.DType;

/// MXFP8 block size: 32 elements per scale group.
const group_size: usize = 32;

/// Max representable E4M3 value.
const fp8_e4m3_max: f32 = 448.0;

/// Progress types (re-exported from scheme).
pub const CProgressCallback = convert.scheme.CProgressCallback;
pub const ProgressContext = convert.scheme.ProgressContext;

/// Conversion options for MXFP8 export.
pub const ConvertOptions = struct {
    output_dir: []const u8 = "models",
    destination: ?[]const u8 = null,
    output_suffix: ?[]const u8 = null,
    force: bool = false,
    max_shard_size: u64 = 0,
    progress: ProgressContext = ProgressContext.NONE,
};

pub const modelIdFromOutputPath = gaf_paths.modelIdFromOutputPath;

/// Convert a transformer model to MXFP8 (E4M3 + UE8M0 block-32 scales).
/// Returns the output path (caller owns the memory).
pub fn convertToMxfp8(
    allocator: std.mem.Allocator,
    input_path: []const u8,
    options: ConvertOptions,
) ![]const u8 {
    // 1. Resolve input model files
    var model_bundle = try repository.resolve(allocator, input_path, .{});
    defer model_bundle.deinit();

    // 2. Load source model config
    const model_config = try config_loader.loadConfig(allocator, model_bundle.config_path());

    // 3. Determine output path
    const output_dir_path = if (options.destination) |dest|
        try allocator.dupe(u8, dest)
    else blk: {
        const suffix = options.output_suffix orelse "MXFP8";
        break :blk try gaf_paths.generateOutputName(
            allocator,
            input_path,
            suffix,
            options.output_dir,
        );
    };
    errdefer allocator.free(output_dir_path);

    // 4. Check if output exists
    if (options.force) {
        std.fs.cwd().deleteTree(output_dir_path) catch {};
    } else {
        std.fs.cwd().access(output_dir_path, .{}) catch |err| switch (err) {
            error.FileNotFound => {},
            else => return err,
        };
        if (std.fs.cwd().openDir(output_dir_path, .{})) |d| {
            var dir = d;
            if (isCompleteConversionOutput(dir)) {
                dir.close();
                log.info("convert", "Output already exists, skipping", .{ .path = output_dir_path });
                return output_dir_path;
            }
            dir.close();
            log.warn("convert", "Removing incomplete output directory before conversion", .{ .path = output_dir_path });
            try std.fs.cwd().deleteTree(output_dir_path);
        } else |_| {}
    }

    // 5. Load source weights
    var source_tensors = try safetensors.UnifiedSafeTensors.load(allocator, model_bundle.weights_path() orelse return error.WeightsNotFound);
    defer source_tensors.deinit();

    // 6. Build weight layout map from architecture metadata
    const model_type = try config_loader.readModelType(allocator, model_bundle.config_path());
    defer if (model_type) |mt| allocator.free(mt);

    var layout_map: ?convert.WeightLayoutMap = null;
    defer if (layout_map) |*lm| lm.deinit();
    var fusion_map: ?convert.ConversionFusionMap = null;
    defer if (fusion_map) |*fm| fm.deinit();

    var runtime_arch: ?*const op_types.Architecture = null;
    if (model_type) |mt| {
        runtime_arch = models_registry.runtimeArchitectureByModelType(mt);
        if (runtime_arch == null) {
            if (models_registry.detectByModelType(mt)) |entry| {
                runtime_arch = models_registry.runtimeArchitectureById(entry.id);
            }
        }
    }

    var layer_types_override: ?[]const u8 = null;
    defer if (layer_types_override) |lt| allocator.free(lt);

    if (runtime_arch) |arch| {
        if (arch.isHeterogeneous()) {
            if (arch.block_variants) |variants| {
                const variant_names = try allocator.alloc([]const u8, variants.len);
                defer allocator.free(variant_names);
                for (variants, 0..) |variant, i| {
                    variant_names[i] = variant.name;
                }
                layer_types_override = config_loader.parseLayerTypes(allocator, model_bundle.config_path(), variant_names, arch.variant_aliases) catch null;
            }
        }
        layout_map = convert.buildWeightLayoutMapWithOverride(allocator, arch, @intCast(model_config.n_layers), layer_types_override) catch |err| blk: {
            log.warn("converter", "Failed to build layout map", .{ .err = @errorName(err), .arch = arch.name });
            break :blk null;
        };
        fusion_map = convert.buildConversionFusionMap(allocator, arch, @intCast(model_config.n_layers)) catch |err| blk: {
            log.warn("converter", "Failed to build fusion map", .{ .err = @errorName(err), .arch = arch.name });
            break :blk null;
        };
    } else if (model_type) |mt| {
        log.warn("converter", "No runtime architecture metadata for model_type", .{ .model_type = mt });
    }

    if (layout_map == null) {
        return error.MissingArchitectureLayout;
    }

    // 7. Create output directory
    var keep_output = false;
    errdefer if (!keep_output) std.fs.cwd().deleteTree(output_dir_path) catch {};
    var output_dir = try gaf_paths.GAFModelDir.init(allocator, output_dir_path);
    defer output_dir.deinit();

    // 8. Process and write weights
    try writeMxfp8Weights(
        allocator,
        &source_tensors,
        model_config.tie_word_embeddings,
        options.max_shard_size,
        output_dir_path,
        options.progress,
        if (layout_map) |*lm| lm else null,
        if (fusion_map) |*fm| fm else null,
    );

    // 9. Copy config.json with MXFP8 quantization info
    try copyConfigWithMxfp8Quantization(allocator, model_bundle.config_path(), output_dir_path, model_config.tie_word_embeddings);

    // 10. Copy all model assets
    try convert.copyModelAssets(allocator, model_bundle.dir, output_dir_path);

    // 11. Generate model card
    const model_name = convert.model_card.extractModelName(input_path);
    const base_model_id = convert.model_card.extractBaseModelId(input_path);
    convert.model_card.writeModelCard(allocator, output_dir_path, model_name, base_model_id, .mxfp8) catch |err| {
        log.warn("converter", "Failed to generate model card", .{ .err = @errorName(err) });
    };

    keep_output = true;
    return output_dir_path;
}

fn isCompleteConversionOutput(dir: std.fs.Dir) bool {
    dir.access("config.json", .{}) catch return false;
    dir.access("model.safetensors", .{}) catch {
        dir.access("model.safetensors.index.json", .{}) catch return false;
    };
    return true;
}

// =============================================================================
// Weight Processing
// =============================================================================

fn writeMxfp8Weights(
    allocator: std.mem.Allocator,
    source_tensors: *safetensors.UnifiedSafeTensors,
    tie_embeddings: bool,
    max_shard_size: u64,
    output_dir: []const u8,
    progress: ProgressContext,
    layout_map: ?*const convert.WeightLayoutMap,
    fusion_map: ?*const convert.ConversionFusionMap,
) !void {
    var tensor_builder = safetensors.Builder.init(allocator);
    defer tensor_builder.deinit();
    tensor_builder.max_shard_size = max_shard_size;

    const tensor_names = try source_tensors.tensorNames(allocator);
    defer allocator.free(tensor_names);

    progress.addLine(0, "Converting", tensor_names.len, null, "tensors");
    var last_log_ms: i64 = 0;
    var last_percent: i64 = -1;

    for (tensor_names, 0..) |tensor_name, tensor_index| {
        var msg_buf: [256]u8 = undefined;
        const copy_len = @min(tensor_name.len, msg_buf.len - 1);
        @memcpy(msg_buf[0..copy_len], tensor_name[0..copy_len]);
        msg_buf[copy_len] = 0;
        progress.updateLine(0, tensor_index + 1, @ptrCast(&msg_buf));
        if (tensor_names.len > 0) {
            const now_ms = std.time.milliTimestamp();
            const current = tensor_index + 1;
            const total = tensor_names.len;
            const percent: i64 = @intCast((current * 100) / total);
            const should_log = (now_ms - last_log_ms) >= 3000 or percent == 100;
            if (should_log and percent != last_percent) {
                last_log_ms = now_ms;
                last_percent = percent;
                var log_buf: [128]u8 = undefined;
                const msg = std.fmt.bufPrint(&log_buf, "Converting model ({d} / {d} tensors) {d}%", .{
                    current,
                    total,
                    percent,
                }) catch continue;
                log.info("convert", msg, .{ .tensor = tensor_name });
            }
        }

        if (convert.shouldSkipForTiedEmbeddingsByName(layout_map, tensor_name, tie_embeddings)) {
            continue;
        }

        if (fusion_map) |map| {
            if (map.isConsumedNonTrigger(tensor_name)) continue;
            if (map.planForTrigger(tensor_name)) |plan| {
                if (try maybeWriteFusedTensorForPlan(allocator, source_tensors, &tensor_builder, plan)) {
                    continue;
                }
            }
        }

        {
            const source_tensor = source_tensors.getTensor(tensor_name, null) catch |err| {
                log.warn("convert", "MXFP8 source tensor missing", .{
                    .tensor = tensor_name,
                    .err = @errorName(err),
                });
                return err;
            };

            if (shouldMxfp8Quantize(layout_map, tensor_name, source_tensor)) {
                try quantizeMxfp8Tensor(allocator, source_tensors, &tensor_builder, tensor_name, source_tensor);
                continue;
            }
            try copyTensorUnchanged(allocator, &tensor_builder, tensor_name, source_tensor);
        }
    }

    // Synthesize MXFP8 lm_head from embedding when embeddings are tied.
    if (tie_embeddings) {
        if (findEmbeddingTensorName(layout_map, source_tensors)) |embed_name| {
            const embed_tensor = try source_tensors.getTensor(embed_name, null);
            try quantizeMxfp8Tensor(allocator, source_tensors, &tensor_builder, "lm_head.weight", embed_tensor);
        } else {
            log.warn("convert", "MXFP8 tied-embedding synth failed to resolve embedding tensor", .{});
            return error.NotFound;
        }
    }

    progress.completeLine(0);
    try tensor_builder.save(output_dir, "model.safetensors");
}

fn findEmbeddingTensorName(layout_map: ?*const convert.WeightLayoutMap, source_tensors: *safetensors.UnifiedSafeTensors) ?[]const u8 {
    const map = layout_map orelse return null;
    const preferred_names = [_][]const u8{
        "model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
        "embed_tokens.weight",
        "transformer.wte.weight",
        "backbone.embedding.weight",
        "language_model.model.embed_tokens.weight",
    };

    for (preferred_names) |name| {
        if (map.layouts.get(name) == .embedding) {
            _ = source_tensors.getTensor(name, null) catch continue;
            return name;
        }
    }

    var iter = map.layouts.iterator();
    while (iter.next()) |kv| {
        if (kv.value_ptr.* != .embedding) continue;
        _ = source_tensors.getTensor(kv.key_ptr.*, null) catch continue;
        return kv.key_ptr.*;
    }
    return null;
}

fn maybeWriteFusedTensorForPlan(
    allocator: std.mem.Allocator,
    source_tensors: *safetensors.UnifiedSafeTensors,
    builder: *safetensors.Builder,
    plan: *const convert.ConversionFusionPlan,
) !bool {
    switch (plan.kind) {
        .gated_delta_split_in_proj => {
            if (plan.required_inputs.len != 4) return error.InvalidWeightTransform;
            const qkv = try source_tensors.getTensor(plan.required_inputs[0], null);
            const z = try source_tensors.getTensor(plan.required_inputs[1], null);
            const b = try source_tensors.getTensor(plan.required_inputs[2], null);
            const a = try source_tensors.getTensor(plan.required_inputs[3], null);

            const fused = try load_transforms.buildGatedDeltaSplitInProj(
                allocator,
                &qkv,
                &z,
                &b,
                &a,
            );
            defer @constCast(fused).deinit(allocator);

            try quantizeMxfp8Tensor(allocator, source_tensors, builder, plan.output_name, fused.*);
            return true;
        },
    }
}

// =============================================================================
// Quantization Decision
// =============================================================================

fn shouldMxfp8Quantize(
    layout_map: ?*const convert.WeightLayoutMap,
    tensor_name: []const u8,
    src_tensor: tensor.Tensor,
) bool {
    const map = layout_map orelse return false;
    const layout = map.layouts.get(tensor_name) orelse return false;
    if (layout != .linear) return false;
    if (src_tensor.n_dims != 2) return false;
    switch (src_tensor.dtype) {
        .f32, .f16, .bf16, .f8_e4m3 => {},
        else => return false,
    }
    if (src_tensor.numel < 1024) return false;
    return true;
}

// =============================================================================
// MXFP8 Quantization
// =============================================================================

/// Compute per-32-element group absmax → UE8M0 scale → quantize to E4M3.
///
/// UE8M0 scale encoding: value = 2^(e8m0 - 127).
/// To quantize: find absmax in group, compute e8m0 = ceil_log2(absmax / fp8_max) + 127,
/// then scale = 2^(e8m0 - 127), fp8 = f32ToFp8E4M3(value / scale).

/// Compute the UE8M0 exponent for a group absmax.
/// Returns the exponent byte such that 2^(e8m0-127) >= absmax/448.
fn absMaxToE8M0(absmax: f32) u8 {
    if (absmax == 0) return 0; // 2^(-127) ≈ 0, effectively zero
    // We need: scale >= absmax / fp8_e4m3_max
    // scale = 2^(e8m0 - 127)
    // e8m0 = ceil(log2(absmax / 448)) + 127
    const ratio = absmax / fp8_e4m3_max;
    // Extract the IEEE754 exponent of ratio, then ceil by checking if there's a fractional part
    const bits: u32 = @bitCast(ratio);
    const biased_exp: i32 = @intCast((bits >> 23) & 0xFF);
    const mantissa = bits & 0x7FFFFF;
    // ceil_log2: if mantissa != 0, we need one more exponent
    const ceil_exp: i32 = if (mantissa != 0) biased_exp + 1 else biased_exp;
    // Convert from IEEE754 biased exponent (bias=127) to E8M0 (also bias=127)
    // IEEE754: value = 2^(biased_exp - 127) * 1.mantissa
    // E8M0: value = 2^(e8m0 - 127)
    // So e8m0 = ceil_exp (the IEEE754 biased exponent is already in E8M0 encoding)
    return @intCast(std.math.clamp(ceil_exp, 0, 255));
}

/// Convert E8M0 byte to f32 scale value: 2^(e8m0 - 127).
inline fn e8m0ToScale(e8m0: u8) f32 {
    const exp_bits = @as(u32, e8m0) << 23;
    return @bitCast(exp_bits);
}

/// Pass 1 context: compute per-group absmax across rows.
/// For a [rows × cols] weight, scale grid is [rows × ceil(cols/32)].
const GroupAbsmaxContext = struct {
    source_data: []align(1) const f32,
    e8m0_scales: []u8,
    rows: usize,
    cols: usize,
    scale_cols: usize,
};

fn computeGroupAbsmax(row_start: usize, row_end: usize, ctx: *GroupAbsmaxContext) void {
    const cols = ctx.cols;
    const scale_cols = ctx.scale_cols;

    for (row_start..row_end) |row| {
        const row_vals = ctx.source_data[row * cols .. (row + 1) * cols];
        var col: usize = 0;
        var scale_idx: usize = 0;
        while (col < cols) : ({
            col += group_size;
            scale_idx += 1;
        }) {
            const col_end = @min(col + group_size, cols);
            var absmax: f32 = 0;
            for (row_vals[col..col_end]) |v| {
                const a = @abs(v);
                if (a > absmax) absmax = a;
            }
            ctx.e8m0_scales[row * scale_cols + scale_idx] = absMaxToE8M0(absmax);
        }
    }
}

/// Pass 2 context: quantize using precomputed E8M0 scales.
const Mxfp8QuantizeContext = struct {
    source_data: []align(1) const f32,
    fp8_data: []u8,
    e8m0_scales: []const u8,
    cols: usize,
    scale_cols: usize,
};

fn quantizeRowSlice(row_start: usize, row_end: usize, ctx: *Mxfp8QuantizeContext) void {
    const cols = ctx.cols;
    const scale_cols = ctx.scale_cols;

    for (row_start..row_end) |row| {
        const row_values = ctx.source_data[row * cols .. (row + 1) * cols];
        const row_fp8 = ctx.fp8_data[row * cols .. (row + 1) * cols];

        var col: usize = 0;
        var scale_idx: usize = 0;
        while (col < cols) : ({
            col += group_size;
            scale_idx += 1;
        }) {
            const col_end = @min(col + group_size, cols);
            const e8m0 = ctx.e8m0_scales[row * scale_cols + scale_idx];
            const scale = e8m0ToScale(e8m0);
            const inv_scale: f32 = if (scale > 0) 1.0 / scale else 0;

            for (row_values[col..col_end], row_fp8[col..col_end]) |v, *out| {
                out.* = dtype_mod.f32ToFp8E4M3(v * inv_scale);
            }
        }
    }
}

/// Quantize a single tensor to MXFP8: E4M3 data + UE8M0 block-32 scales.
fn quantizeMxfp8Tensor(
    allocator: std.mem.Allocator,
    source_tensors: *safetensors.UnifiedSafeTensors,
    builder: *safetensors.Builder,
    tensor_name: []const u8,
    source_tensor: Tensor,
) !void {
    const rows: usize = @intCast(source_tensor.shape[0]);
    const cols: usize = @intCast(source_tensor.shape[1]);

    // Convert source to F32
    const f32_source = try tensorToF32ForQuantization(allocator, source_tensors, tensor_name, source_tensor);
    defer f32_source.deinit(allocator);
    const source_values = f32_source.asF32Slice();

    // Scale grid: one E8M0 byte per 32 elements in each row
    const scale_cols = (cols + group_size - 1) / group_size;

    // Allocate output buffers
    const fp8_values = try allocator.alloc(u8, rows * cols);
    defer allocator.free(fp8_values);
    const e8m0_scales = try allocator.alloc(u8, rows * scale_cols);
    defer allocator.free(e8m0_scales);

    const pool = parallel.global();

    // Pass 1: compute per-group absmax → E8M0 scales (parallel over rows)
    var absmax_ctx = GroupAbsmaxContext{
        .source_data = source_values,
        .e8m0_scales = e8m0_scales,
        .rows = rows,
        .cols = cols,
        .scale_cols = scale_cols,
    };
    pool.parallelFor(rows, computeGroupAbsmax, &absmax_ctx);

    // Pass 2: quantize to E4M3 using precomputed scales (parallel over rows)
    var quant_ctx = Mxfp8QuantizeContext{
        .source_data = source_values,
        .fp8_data = fp8_values,
        .e8m0_scales = e8m0_scales,
        .cols = cols,
        .scale_cols = scale_cols,
    };
    pool.parallelFor(rows, quantizeRowSlice, &quant_ctx);

    // Write E4M3 weight tensor
    try builder.addTensor(
        tensor_name,
        .f8_e4m3,
        &[_]usize{ rows, cols },
        fp8_values,
    );

    // Write UE8M0 block scale tensor: "{base}.weight_block_scale"
    const tensor_base_name = if (std.mem.endsWith(u8, tensor_name, ".weight"))
        tensor_name[0 .. tensor_name.len - ".weight".len]
    else
        tensor_name;
    var scale_name_buf: [256]u8 = undefined;
    const scale_tensor_name = try std.fmt.bufPrint(&scale_name_buf, "{s}.weight_block_scale", .{tensor_base_name});
    try builder.addTensor(
        scale_tensor_name,
        .u8,
        &[_]usize{ rows, scale_cols },
        e8m0_scales,
    );
}

/// Convert source tensor to F32, handling FP8 sources with scale-aware dequantization.
fn tensorToF32ForQuantization(
    allocator: std.mem.Allocator,
    source_tensors: *safetensors.UnifiedSafeTensors,
    tensor_name: []const u8,
    source_tensor: Tensor,
) !convert.F32Result {
    if (source_tensor.dtype != .f8_e4m3) return convert.tensorToF32(allocator, source_tensor);

    // FP8 tensors need inverse scales for correct dequantization
    const base = if (std.mem.endsWith(u8, tensor_name, ".weight"))
        tensor_name[0 .. tensor_name.len - ".weight".len]
    else
        tensor_name;

    // Try per-block BF16 scale first (standard FP8 format)
    var scale_name_buf: [256]u8 = undefined;
    const scale_name = std.fmt.bufPrint(&scale_name_buf, "{s}.weight_scale_inv", .{base}) catch {
        return convert.tensorToF32(allocator, source_tensor);
    };
    const scale_tensor = source_tensors.getTensor(scale_name, null) catch {
        return convert.tensorToF32(allocator, source_tensor);
    };

    return convert.fp8.dequantizeFp8WithScale(allocator, source_tensor, scale_tensor) catch |err| {
        log.warn("convert", "FP8 dequantization with scales failed; falling back to raw FP8 conversion", .{
            .tensor = tensor_name,
            .err = @errorName(err),
        });
        return convert.tensorToF32(allocator, source_tensor);
    };
}

fn copyTensorUnchanged(
    allocator: std.mem.Allocator,
    builder: *safetensors.Builder,
    tensor_name: []const u8,
    source_tensor: Tensor,
) !void {
    _ = allocator;
    const shape_array = source_tensor.shapeAsUsize();
    const shape = shape_array[0..@intCast(source_tensor.n_dims)];
    try builder.addTensor(tensor_name, source_tensor.dtype, shape, source_tensor.data()[0..source_tensor.data_size]);
}

// =============================================================================
// Config.json
// =============================================================================

fn copyConfigWithMxfp8Quantization(
    allocator: std.mem.Allocator,
    source_config_path: []const u8,
    output_dir: []const u8,
    untie_embeddings: bool,
) !void {
    const source_file = std.fs.cwd().openFile(source_config_path, .{}) catch |err| {
        if (err == error.FileNotFound) return;
        return err;
    };
    defer source_file.close();

    const source_content = try source_file.readToEndAlloc(allocator, 1024 * 1024);
    defer allocator.free(source_content);

    var parsed = json.parseValue(allocator, source_content, .{
        .max_size_bytes = 10 * 1024 * 1024,
        .max_value_bytes = 10 * 1024 * 1024,
        .max_string_bytes = 1 * 1024 * 1024,
    }) catch |err| {
        return switch (err) {
            error.InputTooLarge => error.InvalidConfig,
            error.InputTooDeep => error.InvalidConfig,
            error.StringTooLong => error.InvalidConfig,
            error.InvalidJson => error.InvalidConfig,
            error.OutOfMemory => error.OutOfMemory,
        };
    };
    defer parsed.deinit();

    if (parsed.value != .object) {
        return error.InvalidConfig;
    }

    var output_buf = std.ArrayListUnmanaged(u8){};
    defer output_buf.deinit(allocator);

    try output_buf.append(allocator, '{');

    var first_field = true;
    var iter = parsed.value.object.iterator();
    while (iter.next()) |kv| {
        if (std.mem.eql(u8, kv.key_ptr.*, "quantization_config")) continue;
        if (std.mem.eql(u8, kv.key_ptr.*, "quantization")) continue;

        if (!first_field) try output_buf.append(allocator, ',');
        first_field = false;

        if (untie_embeddings and std.mem.eql(u8, kv.key_ptr.*, "tie_word_embeddings")) {
            try output_buf.appendSlice(allocator, "\"tie_word_embeddings\":false");
            continue;
        }

        try output_buf.append(allocator, '"');
        try output_buf.appendSlice(allocator, kv.key_ptr.*);
        try output_buf.appendSlice(allocator, "\":");

        if (untie_embeddings and std.mem.eql(u8, kv.key_ptr.*, "text_config")) {
            var text_config_json = try std.json.Stringify.valueAlloc(allocator, kv.value_ptr.*, .{});
            defer allocator.free(text_config_json);
            if (std.mem.indexOf(u8, text_config_json, "\"tie_word_embeddings\":true")) |pos| {
                try output_buf.appendSlice(allocator, text_config_json[0..pos]);
                try output_buf.appendSlice(allocator, "\"tie_word_embeddings\":false");
                try output_buf.appendSlice(allocator, text_config_json[pos + "\"tie_word_embeddings\":true".len ..]);
            } else {
                try output_buf.appendSlice(allocator, text_config_json);
            }
            continue;
        }

        const value_json = try std.json.Stringify.valueAlloc(allocator, kv.value_ptr.*, .{});
        defer allocator.free(value_json);
        try output_buf.appendSlice(allocator, value_json);
    }

    // Add MXFP8 quantization_config
    if (!first_field) try output_buf.append(allocator, ',');
    try output_buf.appendSlice(allocator,
        "\"quantization_config\":{\"quant_method\":\"mxfp8\",\"fmt\":\"e4m3\",\"scale_fmt\":\"e8m0\",\"block_size\":32}",
    );

    try output_buf.append(allocator, '}');

    const dst_config_path = try std.fs.path.join(allocator, &.{ output_dir, "config.json" });
    defer allocator.free(dst_config_path);

    var dst_file = try std.fs.cwd().createFile(dst_config_path, .{});
    defer dst_file.close();
    try dst_file.writeAll(output_buf.items);
}

// =============================================================================
// Tests
// =============================================================================

test "absMaxToE8M0 basic values" {
    // absmax=0 → e8m0=0
    try std.testing.expectEqual(@as(u8, 0), absMaxToE8M0(0.0));

    // absmax=448 → ratio=1.0 → e8m0=127 (2^0 = 1.0)
    try std.testing.expectEqual(@as(u8, 127), absMaxToE8M0(448.0));

    // absmax=896 → ratio=2.0 → e8m0=128 (2^1 = 2.0)
    try std.testing.expectEqual(@as(u8, 128), absMaxToE8M0(896.0));

    // absmax=224 → ratio=0.5 → e8m0=126 (2^-1 = 0.5)
    try std.testing.expectEqual(@as(u8, 126), absMaxToE8M0(224.0));

    // absmax=1.0 → ratio=1/448 ≈ 0.00223 → ceil_log2 ≈ -9 → e8m0=118
    const e = absMaxToE8M0(1.0);
    const scale = e8m0ToScale(e);
    // scale should be >= 1/448 and a power of 2
    try std.testing.expect(scale >= 1.0 / 448.0);
    try std.testing.expect(scale <= 2.0 / 448.0);
}

test "absMaxToE8M0 roundtrip covers range" {
    // For any absmax, scale * 448 >= absmax (no overflow)
    const test_values = [_]f32{ 0.001, 0.1, 1.0, 10.0, 100.0, 448.0, 1000.0 };
    for (test_values) |absmax| {
        const e8m0 = absMaxToE8M0(absmax);
        const scale = e8m0ToScale(e8m0);
        // scale * fp8_max must cover the absmax
        try std.testing.expect(scale * fp8_e4m3_max >= absmax);
        // scale should not be more than 2× too large (efficiency check)
        if (absmax > 0.01) {
            try std.testing.expect(scale * fp8_e4m3_max < absmax * 2.01);
        }
    }
}

test "computeGroupAbsmax computes correct per-row-group absmax" {
    const cols = 64;
    const scale_cols = 2; // ceil(64/32)

    var source = [_]f32{0} ** cols;
    // Group 0 (cols 0-31): max at col 5
    source[5] = 10.0;
    // Group 1 (cols 32-63): max at col 50
    source[50] = -25.0;

    var e8m0_out = [_]u8{0} ** scale_cols;

    var ctx = GroupAbsmaxContext{
        .source_data = &source,
        .e8m0_scales = &e8m0_out,
        .rows = 1,
        .cols = cols,
        .scale_cols = scale_cols,
    };

    computeGroupAbsmax(0, 1, &ctx);

    // Verify scales cover the absmax values
    const scale0 = e8m0ToScale(e8m0_out[0]);
    const scale1 = e8m0ToScale(e8m0_out[1]);
    try std.testing.expect(scale0 * fp8_e4m3_max >= 10.0);
    try std.testing.expect(scale1 * fp8_e4m3_max >= 25.0);
}

test "quantizeRowSlice produces valid MXFP8 output" {
    const cols = 32;
    const scale_cols = 1;

    var source = [_]f32{0} ** cols;
    source[0] = 1.0;
    source[1] = -1.0;

    // Compute scale for absmax=1.0
    const e8m0 = absMaxToE8M0(1.0);
    var e8m0_scales = [_]u8{e8m0};

    var fp8_out = [_]u8{0} ** cols;

    var ctx = Mxfp8QuantizeContext{
        .source_data = &source,
        .fp8_data = &fp8_out,
        .e8m0_scales = &e8m0_scales,
        .cols = cols,
        .scale_cols = scale_cols,
    };

    quantizeRowSlice(0, 1, &ctx);

    // Non-zero values should produce non-zero FP8
    try std.testing.expect(fp8_out[0] != 0);
    try std.testing.expect(fp8_out[1] != 0);
    // Zero values should stay zero
    try std.testing.expectEqual(@as(u8, 0), fp8_out[2]);
}

test "MXFP8 roundtrip accuracy" {
    const cols = 32;
    const scale_cols = 1;

    var source: [cols]f32 = undefined;
    for (&source, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i)) - 16.0;
    }

    // Pass 1: absmax → E8M0
    var e8m0_scales: [scale_cols]u8 = undefined;
    var ctx1 = GroupAbsmaxContext{
        .source_data = &source,
        .e8m0_scales = &e8m0_scales,
        .rows = 1,
        .cols = cols,
        .scale_cols = scale_cols,
    };
    computeGroupAbsmax(0, 1, &ctx1);

    // Pass 2: quantize
    var fp8_out = [_]u8{0} ** cols;
    var ctx2 = Mxfp8QuantizeContext{
        .source_data = &source,
        .fp8_data = &fp8_out,
        .e8m0_scales = &e8m0_scales,
        .cols = cols,
        .scale_cols = scale_cols,
    };
    quantizeRowSlice(0, 1, &ctx2);

    // Dequantize and check error
    const scale = e8m0ToScale(e8m0_scales[0]);
    var max_error: f32 = 0;
    for (&source, fp8_out) |original, fp8_val| {
        const dequantized = dtype_mod.fp8e4m3ToF32(fp8_val) * scale;
        const err = @abs(original - dequantized);
        if (err > max_error) max_error = err;
    }

    // Range [-16, 15], E4M3 has 3-bit mantissa, scale is power-of-2.
    // Max error should be small relative to range.
    try std.testing.expect(max_error < 2.0);
}
