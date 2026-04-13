//! FP8 E4M3 Per-Block Model Conversion
//!
//! Converts transformer models to FP8 E4M3 quantization with per-block [128,128]
//! scaling and exports in HuggingFace-compatible SafeTensors format.

const std = @import("std");
const log = @import("log_pkg");
const tensor = @import("tensor_pkg");
const dtype_mod = @import("dtype_pkg");
const safetensors = @import("io_pkg").safetensors.root;
const repository = @import("io_pkg").repository.root;
const gaf_paths = @import("gaf_paths.zig");
const config_loader = @import("models_pkg").config;
const op_types = @import("models_pkg").op_types;
const parallel = @import("compute_pkg").parallel;
const convert = @import("root.zig");
const models_registry = @import("models_pkg").registry;
const load_transforms = @import("models_pkg").load.transforms;
const json = @import("io_pkg").json;

const Tensor = tensor.Tensor;
const DType = dtype_mod.DType;

/// FP8 block size for per-block scaling (rows and cols).
const block_size: usize = 128;

/// Max representable E4M3 value.
const fp8_e4m3_max: f32 = 448.0;

/// Progress types (re-exported from scheme).
pub const CProgressCallback = convert.scheme.CProgressCallback;
pub const ProgressContext = convert.scheme.ProgressContext;

/// Conversion options for FP8 export.
pub const ConvertOptions = struct {
    output_dir: []const u8 = "models",
    destination: ?[]const u8 = null,
    output_suffix: ?[]const u8 = null,
    force: bool = false,
    max_shard_size: u64 = 0,
    progress: ProgressContext = ProgressContext.NONE,
};

pub const modelIdFromOutputPath = gaf_paths.modelIdFromOutputPath;

/// Convert a transformer model to FP8 E4M3 with per-block [128,128] scaling.
/// Returns the output path (caller owns the memory).
pub fn convertToFp8(
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
        const suffix = options.output_suffix orelse "FP8";
        break :blk try gaf_paths.generateOutputName(
            allocator,
            input_path,
            suffix,
            options.output_dir,
        );
    };
    errdefer allocator.free(output_dir_path);
    const output_tmp_path = try std.fmt.allocPrint(allocator, "{s}.tmp", .{output_dir_path});
    defer allocator.free(output_tmp_path);

    // 4. Check if output exists
    // Always clean stale temporary output from a prior interrupted run.
    std.fs.cwd().deleteTree(output_tmp_path) catch {};
    if (!options.force) {
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

    // For heterogeneous models, parse layer_types from config.json so we use
    // the correct variant for each layer (avoids falling off a short static layer_map).
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

    // Architecture metadata required for quantization decisions
    if (layout_map == null) {
        return error.MissingArchitectureLayout;
    }

    // 7. Create output directory
    var keep_output = false;
    errdefer if (!keep_output) std.fs.cwd().deleteTree(output_tmp_path) catch {};
    var output_dir = try gaf_paths.GAFModelDir.init(allocator, output_tmp_path);
    defer output_dir.deinit();

    // 8. Process and write weights
    try writeFp8Weights(
        allocator,
        &source_tensors,
        model_config.tie_word_embeddings,
        options.max_shard_size,
        output_tmp_path,
        options.progress,
        if (layout_map) |*lm| lm else null,
        if (fusion_map) |*fm| fm else null,
    );

    // 9. Copy config.json with FP8 quantization info
    try copyConfigWithFp8Quantization(allocator, model_bundle.config_path(), output_tmp_path, model_config.tie_word_embeddings);

    // 10. Copy all model assets
    try convert.copyModelAssets(allocator, model_bundle.dir, output_tmp_path);

    // 11. Generate model card
    const model_name = convert.model_card.extractModelName(input_path);
    const base_model_id = convert.model_card.extractBaseModelId(input_path);
    convert.model_card.writeModelCard(allocator, output_tmp_path, model_name, base_model_id, .fp8_e4m3) catch |err| {
        log.warn("converter", "Failed to generate model card", .{ .err = @errorName(err) });
    };

    if (options.force) {
        std.fs.cwd().access(output_dir_path, .{}) catch |err| switch (err) {
            error.FileNotFound => {},
            else => return err,
        };
        if (std.fs.cwd().openDir(output_dir_path, .{})) |d| {
            var dir = d;
            dir.close();
            try std.fs.cwd().deleteTree(output_dir_path);
        } else |_| {}
    }
    try std.fs.cwd().rename(output_tmp_path, output_dir_path);

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

fn writeFp8Weights(
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
        // Progress reporting
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

        // Skip lm_head when embeddings are tied
        if (convert.shouldSkipForTiedEmbeddingsByName(layout_map, tensor_name, tie_embeddings)) {
            continue;
        }

        // Handle fusion plans
        if (fusion_map) |map| {
            if (map.isConsumedNonTrigger(tensor_name)) continue;
            if (map.planForTrigger(tensor_name)) |plan| {
                if (try maybeWriteFusedTensorForPlan(allocator, source_tensors, &tensor_builder, plan)) {
                    continue;
                }
            }
        }

        {
            const source_tensor = try source_tensors.getTensor(tensor_name, null);

            if (shouldFp8Quantize(layout_map, tensor_name, source_tensor)) {
                try quantizeFp8Tensor(allocator, source_tensors, &tensor_builder, tensor_name, source_tensor);
                continue;
            }
            try copyTensorUnchanged(allocator, &tensor_builder, tensor_name, source_tensor);
        }
    }

    // Synthesize FP8 lm_head from embedding when embeddings are tied.
    // Embedding stays BF16 (lookup table); lm_head gets a separate FP8 copy (matmul).
    if (tie_embeddings) {
        if (findEmbeddingTensorName(layout_map, source_tensors)) |embed_name| {
            const embed_tensor = try source_tensors.getTensor(embed_name, null);
            try quantizeFp8Tensor(allocator, source_tensors, &tensor_builder, "lm_head.weight", embed_tensor);
        } else {
            return error.NotFound;
        }
    }

    progress.completeLine(0);
    try tensor_builder.save(output_dir, "model.safetensors");
}

/// Find the embedding tensor name from the layout map (first entry with .embedding layout).
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

            try quantizeFp8Tensor(allocator, source_tensors, builder, plan.output_name, fused.*);
            return true;
        },
        .dense_mlp_gate_up, .attention_qkv => {
            // FP8 converter: structural fusions not yet implemented.
            return false;
        },
    }
}

// =============================================================================
// Quantization Decision
// =============================================================================

/// FP8-specific quantization decision: only quantize linear (matmul) weights.
/// Embeddings use integer lookup, not matmul — FP8 doesn't apply.
fn shouldFp8Quantize(
    layout_map: ?*const convert.WeightLayoutMap,
    tensor_name: []const u8,
    src_tensor: tensor.Tensor,
) bool {
    const map = layout_map orelse return false;

    // Only quantize tensors with .linear layout (not .embedding)
    const layout = map.layouts.get(tensor_name) orelse return false;
    if (layout != .linear) return false;

    // Standard validation: 2D, float, not tiny
    if (src_tensor.n_dims != 2) return false;
    switch (src_tensor.dtype) {
        .f32, .f16, .bf16, .f8_e4m3 => {},
        else => return false,
    }
    if (src_tensor.numel < 1024) return false;

    return true;
}

// =============================================================================
// FP8 Quantization
// =============================================================================

/// Pass 1 context: compute per-block absmax.
///
/// Scale convention: `weight_scale_inv = absmax / 448`. To dequantize: `float = fp8 * scale_inv`.
/// To quantize: `fp8 = f32ToFp8E4M3(float / scale_inv)` = `f32ToFp8E4M3(float * inv_scale)`.
const BlockAbsmaxContext = struct {
    source_data: []align(1) const f32,
    absmax_data: []f32,
    rows: usize,
    cols: usize,
    scale_cols: usize,
};

/// Compute absmax for a range of block-rows (called by each thread).
/// Each block-row covers up to 128 rows of the weight matrix.
fn computeBlockAbsmax(block_row_start: usize, block_row_end: usize, ctx: *BlockAbsmaxContext) void {
    const cols = ctx.cols;
    const scale_cols = ctx.scale_cols;
    const total_rows = ctx.rows;

    for (block_row_start..block_row_end) |block_row| {
        const row_start = block_row * block_size;
        const row_end = @min(row_start + block_size, total_rows);

        var scale_col: usize = 0;
        var col: usize = 0;
        while (col < cols) : ({
            col += block_size;
            scale_col += 1;
        }) {
            const col_end = @min(col + block_size, cols);

            // Find absmax over the entire block (all rows × cols in this block)
            var absmax: f32 = 0;
            for (row_start..row_end) |r| {
                const row_vals = ctx.source_data[r * cols + col .. r * cols + col_end];
                for (row_vals) |v| {
                    const a = @abs(v);
                    if (a > absmax) absmax = a;
                }
            }
            ctx.absmax_data[block_row * scale_cols + scale_col] = absmax;
        }
    }
}

/// Pass 2 context: quantize using precomputed scales.
const Fp8QuantizeContext = struct {
    source_data: []align(1) const f32,
    fp8_data: []u8,
    scale_data: []u16,
    cols: usize,
    scale_cols: usize,
};

/// Quantize a range of rows using precomputed per-block scales (called by each thread).
/// scale_data contains scale_inv = absmax/448. To quantize: fp8 = f32ToFp8E4M3(val / scale_inv).
fn quantizeRowSlice(row_start: usize, row_end: usize, ctx: *Fp8QuantizeContext) void {
    const cols = ctx.cols;
    const scale_cols = ctx.scale_cols;

    for (row_start..row_end) |row| {
        const row_values = ctx.source_data[row * cols .. (row + 1) * cols];
        const row_fp8 = ctx.fp8_data[row * cols .. (row + 1) * cols];
        const scale_row = row / block_size;

        var col: usize = 0;
        var scale_col_idx: usize = 0;
        while (col < cols) : ({
            col += block_size;
            scale_col_idx += 1;
        }) {
            const block_end = @min(col + block_size, cols);
            const scale_inv_bf16 = ctx.scale_data[scale_row * scale_cols + scale_col_idx];
            const scale_inv = dtype_mod.bf16ToF32(scale_inv_bf16);
            // inv_scale = 1/scale_inv = 448/absmax — maps max value to FP8 max
            const inv_scale: f32 = if (scale_inv > 0) 1.0 / scale_inv else 0;

            for (row_values[col..block_end], row_fp8[col..block_end]) |v, *out| {
                out.* = dtype_mod.f32ToFp8E4M3(v * inv_scale);
            }
        }
    }
}

/// Quantize a single tensor to FP8 E4M3 with per-block [128,128] scales.
fn quantizeFp8Tensor(
    allocator: std.mem.Allocator,
    source_tensors: *safetensors.UnifiedSafeTensors,
    builder: *safetensors.Builder,
    tensor_name: []const u8,
    source_tensor: Tensor,
) !void {
    const rows: usize = @intCast(source_tensor.shape[0]);
    const cols: usize = @intCast(source_tensor.shape[1]);

    // Convert source to F32 (handles BF16/F16/F32/FP8 with scale-aware dequant)
    const f32_source = try tensorToF32ForQuantization(allocator, source_tensors, tensor_name, source_tensor);
    defer f32_source.deinit(allocator);
    const source_values = f32_source.asF32Slice();

    // Scale grid dimensions
    const scale_rows = (rows + block_size - 1) / block_size;
    const scale_cols = (cols + block_size - 1) / block_size;

    // Allocate output buffers
    const fp8_values = try allocator.alloc(u8, rows * cols);
    defer allocator.free(fp8_values);
    const scale_values = try allocator.alloc(u16, scale_rows * scale_cols);
    defer allocator.free(scale_values);
    const absmax_values = try allocator.alloc(f32, scale_rows * scale_cols);
    defer allocator.free(absmax_values);

    const pool = parallel.global();

    // Pass 1: compute per-block absmax (parallel over block-rows)
    var absmax_ctx = BlockAbsmaxContext{
        .source_data = source_values,
        .absmax_data = absmax_values,
        .rows = rows,
        .cols = cols,
        .scale_cols = scale_cols,
    };
    pool.parallelFor(scale_rows, computeBlockAbsmax, &absmax_ctx);

    // Convert absmax to BF16 scale_inv (dequant multiplier: float = fp8 * scale_inv)
    for (absmax_values, scale_values) |absmax, *scale_out| {
        const scale_inv: f32 = if (absmax > 0) absmax / fp8_e4m3_max else 0;
        scale_out.* = convert.f32ToBf16(scale_inv);
    }

    // Pass 2: quantize elements using precomputed scales (parallel over rows)
    var quant_ctx = Fp8QuantizeContext{
        .source_data = source_values,
        .fp8_data = fp8_values,
        .scale_data = scale_values,
        .cols = cols,
        .scale_cols = scale_cols,
    };
    pool.parallelFor(rows, quantizeRowSlice, &quant_ctx);

    // Write FP8 weight tensor
    try builder.addTensor(
        tensor_name,
        .f8_e4m3,
        &[_]usize{ rows, cols },
        fp8_values,
    );

    // Write scale_inv tensor: "{base}.weight_scale_inv"
    const tensor_base_name = if (std.mem.endsWith(u8, tensor_name, ".weight"))
        tensor_name[0 .. tensor_name.len - ".weight".len]
    else
        tensor_name;
    var scale_name_buf: [256]u8 = undefined;
    const scale_tensor_name = try std.fmt.bufPrint(&scale_name_buf, "{s}.weight_scale_inv", .{tensor_base_name});
    try builder.addTensor(
        scale_tensor_name,
        .bf16,
        &[_]usize{ scale_rows, scale_cols },
        std.mem.sliceAsBytes(scale_values),
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

    var scale_name_buf: [256]u8 = undefined;
    const scale_name = std.fmt.bufPrint(&scale_name_buf, "{s}.weight_scale_inv", .{base}) catch {
        return convert.tensorToF32(allocator, source_tensor);
    };
    const scale_tensor = source_tensors.getTensor(scale_name, null) catch {
        return convert.tensorToF32(allocator, source_tensor);
    };

    return dequantizeFp8WithScale(allocator, source_tensor, scale_tensor) catch |err| {
        log.warn("convert", "FP8 dequantization with scales failed; falling back to raw FP8 conversion", .{
            .tensor = tensor_name,
            .err = @errorName(err),
        });
        return convert.tensorToF32(allocator, source_tensor);
    };
}

fn scaleToF32(scale_tensor: Tensor, idx: usize) !f32 {
    if (scale_tensor.dtype != .bf16 and scale_tensor.dtype != .f16) return error.UnsupportedDType;
    const scale_u16 = scale_tensor.asSliceUnaligned(u16);
    if (idx >= scale_u16.len) return error.InvalidShape;
    return if (scale_tensor.dtype == .bf16)
        dtype_mod.bf16ToF32(scale_u16[idx])
    else
        dtype_mod.fp16ToF32(scale_u16[idx]);
}

pub fn dequantizeFp8WithScale(
    allocator: std.mem.Allocator,
    weight_tensor: Tensor,
    scale_tensor: Tensor,
) !convert.F32Result {
    if (weight_tensor.n_dims != 2) return error.InvalidShape;

    const rows: usize = @intCast(weight_tensor.shape[0]);
    const cols: usize = @intCast(weight_tensor.shape[1]);
    const src_u8 = weight_tensor.asSliceUnaligned(u8);
    if (src_u8.len != rows * cols) return error.InvalidShape;

    const f32_values = try allocator.alloc(f32, rows * cols);

    if (scale_tensor.n_dims == 2) {
        const scale_rows: usize = @intCast(scale_tensor.shape[0]);
        const scale_cols: usize = @intCast(scale_tensor.shape[1]);
        if (scale_rows == 0 or scale_cols == 0) return error.InvalidShape;
        if (rows % scale_rows != 0 or cols % scale_cols != 0) return error.InvalidShape;

        const block_row_size = rows / scale_rows;
        const block_col_size = cols / scale_cols;

        for (0..rows) |r| {
            const scale_row = r / block_row_size;
            for (0..cols) |c| {
                const scale_col = c / block_col_size;
                const scale_idx = scale_row * scale_cols + scale_col;
                const scale_inv = try scaleToF32(scale_tensor, scale_idx);
                const src_idx = r * cols + c;
                f32_values[src_idx] = dtype_mod.fp8e4m3ToF32(src_u8[src_idx]) * scale_inv;
            }
        }
        return .{ .data = std.mem.sliceAsBytes(f32_values), .owned = f32_values };
    }

    // Scalar scale fallback
    const scale_inv = try scaleToF32(scale_tensor, 0);
    for (src_u8, f32_values) |src_val, *dst_ptr| {
        dst_ptr.* = dtype_mod.fp8e4m3ToF32(src_val) * scale_inv;
    }
    return .{ .data = std.mem.sliceAsBytes(f32_values), .owned = f32_values };
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

/// Copy config.json with FP8 quantization_config added.
fn copyConfigWithFp8Quantization(
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
        // Strip existing quantization fields — we add our own
        if (std.mem.eql(u8, kv.key_ptr.*, "quantization_config")) continue;
        if (std.mem.eql(u8, kv.key_ptr.*, "quantization")) continue;

        if (!first_field) try output_buf.append(allocator, ',');
        first_field = false;

        // Override tie_word_embeddings when we synthesized a separate FP8 lm_head
        if (untie_embeddings and std.mem.eql(u8, kv.key_ptr.*, "tie_word_embeddings")) {
            try output_buf.appendSlice(allocator, "\"tie_word_embeddings\":false");
            continue;
        }

        try output_buf.append(allocator, '"');
        try output_buf.appendSlice(allocator, kv.key_ptr.*);
        try output_buf.appendSlice(allocator, "\":");

        // For text_config, override tie_word_embeddings inside the nested object
        if (untie_embeddings and std.mem.eql(u8, kv.key_ptr.*, "text_config")) {
            var text_config_json = try std.json.Stringify.valueAlloc(allocator, kv.value_ptr.*, .{});
            defer allocator.free(text_config_json);
            // Replace "tie_word_embeddings":true with false inside text_config
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

    // Add FP8 quantization_config
    if (!first_field) try output_buf.append(allocator, ',');
    try output_buf.appendSlice(
        allocator,
        "\"quantization_config\":{\"quant_method\":\"fp8\",\"fmt\":\"e4m3\",\"weight_block_size\":[128,128],\"activation_scheme\":\"dynamic\"}",
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

test "computeBlockAbsmax computes correct per-block absmax" {
    const rows = 2;
    const cols = 256;
    const scale_rows = 1; // ceil(2/128)
    const scale_cols = 2; // ceil(256/128)

    var source = [_]f32{0} ** (rows * cols);
    // Block [0,0]: rows 0-1, cols 0-127. Set max at row 0 col 0.
    source[0] = 10.0;
    // Block [0,1]: rows 0-1, cols 128-255. Set max at row 1 col 200.
    source[cols + 200] = -25.0;

    var absmax_out = [_]f32{0} ** (scale_rows * scale_cols);

    var ctx = BlockAbsmaxContext{
        .source_data = &source,
        .absmax_data = &absmax_out,
        .rows = rows,
        .cols = cols,
        .scale_cols = scale_cols,
    };

    computeBlockAbsmax(0, scale_rows, &ctx);

    try std.testing.expectApproxEqAbs(@as(f32, 10.0), absmax_out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 25.0), absmax_out[1], 1e-6);
}

test "quantizeRowSlice produces valid FP8 output" {
    const cols = 128;
    const scale_cols = 1;

    // scale_inv = absmax/448 = 1.0/448.0 (absmax = 1.0)
    // inv_scale = 1/scale_inv = 448.0
    var source = [_]f32{0} ** cols;
    source[0] = 1.0;
    source[1] = -1.0;

    var fp8_out = [_]u8{0} ** cols;
    var scale_out = [_]u16{0} ** scale_cols;
    scale_out[0] = convert.f32ToBf16(1.0 / 448.0);

    var ctx = Fp8QuantizeContext{
        .source_data = &source,
        .fp8_data = &fp8_out,
        .scale_data = &scale_out,
        .cols = cols,
        .scale_cols = scale_cols,
    };

    quantizeRowSlice(0, 1, &ctx);

    // 1.0 * 448 = 448 → FP8 max (0x7E), -1.0 * 448 = -448 → FP8 min (0xFE)
    try std.testing.expectEqual(@as(u8, 0x7E), fp8_out[0]);
    try std.testing.expectEqual(@as(u8, 0xFE), fp8_out[1]);
    // Rest should be 0
    try std.testing.expectEqual(@as(u8, 0), fp8_out[2]);
}

test "quantizeRowSlice handles all-zero scale" {
    const cols = 128;
    const scale_cols = 1;

    var source = [_]f32{0} ** cols;
    var fp8_out = [_]u8{0xFF} ** cols;
    var scale_out = [_]u16{0} ** scale_cols; // scale_inv = 0 (all zeros)

    var ctx = Fp8QuantizeContext{
        .source_data = &source,
        .fp8_data = &fp8_out,
        .scale_data = &scale_out,
        .cols = cols,
        .scale_cols = scale_cols,
    };

    quantizeRowSlice(0, 1, &ctx);

    // All zeros → all FP8 outputs should be 0
    for (fp8_out) |v| {
        try std.testing.expectEqual(@as(u8, 0), v);
    }
}

test "quantizeRowSlice handles non-128-aligned cols" {
    const cols = 200;
    const scale_cols = 2; // ceil(200/128)

    var source = [_]f32{1.0} ** cols;
    var fp8_out = [_]u8{0} ** cols;
    // scale_inv = absmax/448 = 1.0/448.0
    var scale_out: [scale_cols]u16 = undefined;
    scale_out[0] = convert.f32ToBf16(1.0 / 448.0);
    scale_out[1] = convert.f32ToBf16(1.0 / 448.0);

    var ctx = Fp8QuantizeContext{
        .source_data = &source,
        .fp8_data = &fp8_out,
        .scale_data = &scale_out,
        .cols = cols,
        .scale_cols = scale_cols,
    };

    quantizeRowSlice(0, 1, &ctx);

    // All outputs should be the same (uniform input, same scale)
    const first_val = fp8_out[0];
    try std.testing.expect(first_val != 0);
    for (fp8_out) |v| {
        try std.testing.expectEqual(first_val, v);
    }
}

test "computeBlockAbsmax and quantizeRowSlice roundtrip accuracy" {
    const cols = 128;
    const scale_cols = 1;

    var source: [cols]f32 = undefined;
    for (&source, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i)) - 64.0;
    }

    // Pass 1: compute absmax
    var absmax_out = [_]f32{0} ** scale_cols;
    var absmax_ctx = BlockAbsmaxContext{
        .source_data = &source,
        .absmax_data = &absmax_out,
        .rows = 1,
        .cols = cols,
        .scale_cols = scale_cols,
    };
    computeBlockAbsmax(0, 1, &absmax_ctx);

    // Convert to BF16 scale_inv (absmax / 448)
    var scale_out: [scale_cols]u16 = undefined;
    for (absmax_out, &scale_out) |absmax, *s| {
        const scale_inv: f32 = if (absmax > 0) absmax / fp8_e4m3_max else 0;
        s.* = convert.f32ToBf16(scale_inv);
    }

    // Pass 2: quantize
    var fp8_out = [_]u8{0} ** cols;
    var quant_ctx = Fp8QuantizeContext{
        .source_data = &source,
        .fp8_data = &fp8_out,
        .scale_data = &scale_out,
        .cols = cols,
        .scale_cols = scale_cols,
    };
    quantizeRowSlice(0, 1, &quant_ctx);

    // Dequantize: float = fp8 * scale_inv
    const scale_inv = dtype_mod.bf16ToF32(scale_out[0]);

    var max_error: f32 = 0;
    for (&source, fp8_out) |original, fp8_val| {
        const dequantized = dtype_mod.fp8e4m3ToF32(fp8_val) * scale_inv;
        const err = @abs(original - dequantized);
        if (err > max_error) max_error = err;
    }

    // Range [-64, 63], absmax=64, scale_inv=7.0. E4M3 has 3-bit mantissa;
    // near the max representable value, step size is 32 in FP8 space = ~4.6 in source space.
    try std.testing.expect(max_error < 5.0);
}
