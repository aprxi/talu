//! Grouped-affine Model Conversion (MLX-compatible export)
//!
//! Converts transformer models to grouped-affine quantization and exports
//! them in MLX-compatible SafeTensors layout.

const std = @import("std");
const log = @import("../log.zig");
const tensor = @import("../tensor.zig");
const dtype_mod = @import("../dtype.zig");
const safetensors = @import("../io/safetensors/root.zig");
const repository = @import("../io/repository/root.zig");
const gaf_paths = @import("gaf_paths.zig");
const config_loader = @import("../graph/config/root.zig");
const compute = @import("../compute/root.zig");
const parallel = compute.parallel;
const convert = @import("root.zig");
const models_registry = @import("../models/registry.zig");

const Tensor = tensor.Tensor;
const DType = dtype_mod.DType;

/// Quantization configuration
pub const QuantConfig = struct {
    bits: u8,
    group_size: u32,
};

/// Progress types (re-exported from scheme).
pub const CProgressCallback = convert.scheme.CProgressCallback;
pub const ProgressContext = convert.scheme.ProgressContext;

/// Conversion options
pub const ConvertOptions = struct {
    quant: ?QuantConfig = null, // If null, preserve source precision
    output_dir: []const u8 = "models",
    /// Explicit output path. If set, output_dir is ignored.
    destination: ?[]const u8 = null,
    /// Output suffix for auto-generated path (e.g., "GAF4", "GAF8-G128").
    /// If null, defaults to "F16" (no quantization).
    output_suffix: ?[]const u8 = null,
    force: bool = false,
    /// Maximum shard size in bytes. 0 = no limit (single file).
    max_shard_size: u64 = 0,
    /// Progress context for emitting progress updates.
    progress: ProgressContext = ProgressContext.NONE,

    /// Derive the Scheme from quant config for model card generation.
    pub fn toScheme(self: ConvertOptions) convert.scheme.Scheme {
        if (self.quant) |q| {
            // Map bits + group_size to scheme
            return switch (q.bits) {
                4 => switch (q.group_size) {
                    32 => .gaf4_32,
                    64 => .gaf4_64,
                    128 => .gaf4_128,
                    else => .gaf4_64,
                },
                8 => switch (q.group_size) {
                    32 => .gaf8_32,
                    64 => .gaf8_64,
                    128 => .gaf8_128,
                    else => .gaf8_64,
                },
                else => .f16,
            };
        }
        return .f16;
    }
};

pub const modelIdFromOutputPath = gaf_paths.modelIdFromOutputPath;

/// Convert a transformer model to grouped-affine weights in MLX format (optionally quantized).
/// Returns the output path (caller owns the memory).
pub fn convertToGroupedAffine(
    allocator: std.mem.Allocator,
    input_path: []const u8,
    options: ConvertOptions,
) ![]const u8 {
    // 1. Resolve input model files
    var model_bundle = try repository.resolve(allocator, input_path, .{});
    defer model_bundle.deinit();

    // 2. Load source model config
    const model_config = try config_loader.loadConfig(allocator, model_bundle.config_path());

    // 3. Determine output path (explicit destination or auto-generated)
    const output_dir_path = if (options.destination) |dest|
        try allocator.dupe(u8, dest)
    else blk: {
        // Use provided suffix, or default to "F16" if no quantization
        const suffix = options.output_suffix orelse "F16";
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
                // Idempotent: already converted, return the existing path
                log.info("convert", "Output already exists, skipping", .{ .path = output_dir_path });
                return output_dir_path;
            }
            dir.close();
            // Incomplete output (e.g. failed prior conversion): clean and retry.
            log.warn("convert", "Removing incomplete output directory before conversion", .{ .path = output_dir_path });
            try std.fs.cwd().deleteTree(output_dir_path);
        } else |_| {}
    }

    // 5. Load source weights and validate (supports both single and sharded models)
    var source_tensors = try safetensors.UnifiedSafeTensors.load(allocator, model_bundle.weights_path() orelse return error.WeightsNotFound);
    defer source_tensors.deinit();

    // 6. Check if model is already quantized
    if (options.quant != null and convert.isAlreadyQuantized(&source_tensors)) {
        return error.AlreadyQuantized;
    }

    // 7. Validate quantization config
    if (options.quant) |q| {
        if (q.bits != 4 and q.bits != 8) {
            return error.UnsupportedBits;
        }
    }

    // 8. Create output directory structure
    var keep_output = false;
    errdefer if (!keep_output) std.fs.cwd().deleteTree(output_dir_path) catch {};
    var output_dir = try gaf_paths.GAFModelDir.init(allocator, output_dir_path);
    defer output_dir.deinit();

    // 9. Build weight layout map from static model architecture metadata.

    // Read model_type and look up architecture
    const model_type = try config_loader.readModelType(allocator, model_bundle.config_path());
    defer if (model_type) |mt| allocator.free(mt);

    var layout_map: ?convert.WeightLayoutMap = null;
    defer if (layout_map) |*lm| lm.deinit();

    if (model_type) |mt| {
        if (models_registry.detectByModelType(mt)) |entry| {
            if (models_registry.runtimeArchitectureById(entry.id)) |arch| {
                layout_map = convert.buildWeightLayoutMap(allocator, arch, @intCast(model_config.n_layers)) catch |err| blk: {
                    log.warn("converter", "Failed to build layout map", .{ .err = @errorName(err) });
                    break :blk null;
                };
            }
        }
    }

    // Quantized conversion requires architecture-driven layout metadata.
    // This keeps name semantics in static model graph metadata, not in converter internals.
    if (options.quant != null and layout_map == null) {
        return error.MissingArchitectureLayout;
    }

    // 10. Process and write weights
    if (options.quant) |quant_config| {
        try writeQuantizedWeights(
            allocator,
            &source_tensors,
            quant_config,
            model_config.tie_word_embeddings,
            options.max_shard_size,
            output_dir_path,
            options.progress,
            if (layout_map) |*lm| lm else null,
        );
    } else {
        try writeUnquantizedWeights(
            allocator,
            &source_tensors,
            model_config.tie_word_embeddings,
            options.max_shard_size,
            output_dir_path,
            options.progress,
            if (layout_map) |*lm| lm else null,
        );
    }

    // 10. Copy original config.json with quantization info added
    // This preserves all architecture-specific fields (e.g., full_attn_idxs, conv_dim for LFM2)
    const quant_config: ?convert.GAFQuantizationConfig = if (options.quant) |q| .{
        .group_size = q.group_size,
        .bits = q.bits,
    } else null;
    try convert.copyConfigWithGAFQuantization(allocator, model_bundle.config_path(), output_dir_path, quant_config);

    // 11. Copy all model assets (tokenizer, chat template, vocab, etc.)
    try convert.copyModelAssets(allocator, model_bundle.dir, output_dir_path);

    // 12. Generate Model Card (README.md)
    const model_name = convert.model_card.extractModelName(input_path);
    const base_model_id = convert.model_card.extractBaseModelId(input_path);
    const scheme = options.toScheme();
    convert.model_card.writeModelCard(allocator, output_dir_path, model_name, base_model_id, scheme) catch |err| {
        // Log but don't fail conversion if model card generation fails
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

/// Quantize all tensors and write to SafeTensors file.
fn writeQuantizedWeights(
    allocator: std.mem.Allocator,
    source_tensors: *safetensors.UnifiedSafeTensors,
    quant_config: QuantConfig,
    tie_embeddings: bool,
    max_shard_size: u64,
    output_dir: []const u8,
    progress: ProgressContext,
    layout_map: ?*const convert.WeightLayoutMap,
) !void {
    var tensor_builder = safetensors.Builder.init(allocator);
    defer tensor_builder.deinit();

    // Set sharding configuration
    tensor_builder.max_shard_size = max_shard_size;

    const tensor_names = try source_tensors.tensorNames(allocator);
    defer allocator.free(tensor_names);

    // Emit progress: start converting
    progress.addLine(0, "Converting", tensor_names.len, null, "tensors");
    var last_log_ms: i64 = 0;
    var last_percent: i64 = -1;

    for (tensor_names, 0..) |tensor_name, tensor_index| {
        // Report progress via unified callback
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

        const source_tensor = try source_tensors.getTensor(tensor_name, null);

        // Use graph-driven layout to determine if tensor should be quantized.
        // Unknown tensors are kept in source precision.
        if (convert.shouldQuantizeTensorByLayout(layout_map, tensor_name, source_tensor)) {
            try quantizeGroupedAffineTensor(allocator, source_tensors, &tensor_builder, tensor_name, source_tensor, quant_config);
        } else {
            try copyTensorUnchanged(allocator, &tensor_builder, tensor_name, source_tensor);
        }
    }

    // Emit progress: conversion complete
    progress.completeLine(0);

    // Write weights (single file or sharded based on max_shard_size)
    // Builder.save() handles sharding decision internally based on max_shard_size
    try tensor_builder.save(output_dir, "model.safetensors");
}

/// Copy all tensors without quantization (preserve original format).
fn writeUnquantizedWeights(
    allocator: std.mem.Allocator,
    source_tensors: *safetensors.UnifiedSafeTensors,
    tie_embeddings: bool,
    max_shard_size: u64,
    output_dir: []const u8,
    progress: ProgressContext,
    layout_map: ?*const convert.WeightLayoutMap,
) !void {
    var tensor_builder = safetensors.Builder.init(allocator);
    defer tensor_builder.deinit();

    // Set sharding configuration
    tensor_builder.max_shard_size = max_shard_size;

    const tensor_names = try source_tensors.tensorNames(allocator);
    defer allocator.free(tensor_names);

    // Emit progress: start converting
    progress.addLine(0, "Converting", tensor_names.len, null, "tensors");
    var last_log_ms: i64 = 0;
    var last_percent: i64 = -1;

    for (tensor_names, 0..) |tensor_name, tensor_index| {
        // Report progress via unified callback
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

        const source_tensor = try source_tensors.getTensor(tensor_name, null);
        try copyTensorUnchanged(allocator, &tensor_builder, tensor_name, source_tensor);
    }

    // Emit progress: conversion complete
    progress.completeLine(0);

    // Write weights (single file or sharded based on max_shard_size)
    // Builder.save() handles sharding decision internally based on max_shard_size
    try tensor_builder.save(output_dir, "model.safetensors");
}

/// Context for parallel quantization.
const RowQuantizeContext = struct {
    source_data: []align(1) const f32,
    packed_row_data: []u32,
    scale_data: []u16,
    bias_data: []u16,
    col_count: usize,
    packed_col_count: usize,
    group_count: usize,
    group_len: usize,
    quant_bits: u8,
};

/// Quantize a range of rows (called by each thread).
fn quantizeRowSlice(row_start: usize, row_end: usize, ctx: *RowQuantizeContext) void {
    const col_count = ctx.col_count;
    const packed_col_count = ctx.packed_col_count;
    const group_count = ctx.group_count;
    const group_len = ctx.group_len;
    const quant_bits = ctx.quant_bits;

    const values_per_u32: usize = if (quant_bits == 4) 8 else 4;
    const max_quant_value: f32 = if (quant_bits == 4) 15.0 else 255.0;

    for (row_start..row_end) |row_idx| {
        const row_values = ctx.source_data[row_idx * col_count .. (row_idx + 1) * col_count];
        const row_packed_words = ctx.packed_row_data[row_idx * packed_col_count .. (row_idx + 1) * packed_col_count];
        const row_scales = ctx.scale_data[row_idx * group_count .. (row_idx + 1) * group_count];
        const row_biases = ctx.bias_data[row_idx * group_count .. (row_idx + 1) * group_count];

        for (0..group_count) |group_idx| {
            const group_start = group_idx * group_len;
            const group_values = row_values[group_start .. group_start + group_len];

            var min_val: f32 = group_values[0];
            var max_val: f32 = group_values[0];
            for (group_values) |value| {
                if (value < min_val) min_val = value;
                if (value > max_val) max_val = value;
            }

            const value_range = max_val - min_val;
            const group_scale: f32 = if (value_range > 0) value_range / max_quant_value else 0;
            const group_bias: f32 = min_val;

            row_scales[group_idx] = convert.f32ToBf16(group_scale);
            row_biases[group_idx] = convert.f32ToBf16(group_bias);

            const words_per_group = group_len / values_per_u32;
            for (0..words_per_group) |pack_word_idx| {
                const value_base = group_start + pack_word_idx * values_per_u32;
                var packed_word: u32 = 0;

                for (0..values_per_u32) |value_idx| {
                    const value = row_values[value_base + value_idx];
                    var quantized: u32 = 0;
                    if (group_scale > 0) {
                        const normalized = (value - group_bias) / group_scale;
                        quantized = @intFromFloat(@max(0, @min(max_quant_value, @round(normalized))));
                    }
                    packed_word |= quantized << @intCast(value_idx * quant_bits);
                }

                row_packed_words[(group_start / values_per_u32) + pack_word_idx] = packed_word;
            }
        }
    }
}

/// Quantize a tensor to grouped-affine weights (4-bit or 8-bit).
fn quantizeGroupedAffineTensor(
    allocator: std.mem.Allocator,
    source_tensors: *safetensors.UnifiedSafeTensors,
    builder: *safetensors.Builder,
    tensor_name: []const u8,
    source_tensor: Tensor,
    quant_config: QuantConfig,
) !void {
    const quant_bits = quant_config.bits;
    if (quant_bits != 4 and quant_bits != 8) return error.UnsupportedBits;

    const row_count: usize = @intCast(source_tensor.shape[0]);
    const col_count: usize = @intCast(source_tensor.shape[1]);
    const group_len = quant_config.group_size;

    const values_per_u32: usize = if (quant_bits == 4) 8 else 4;

    // Ensure cols is divisible by group_size and values_per_word
    if (col_count % group_len != 0 or col_count % values_per_u32 != 0) {
        return copyTensorUnchanged(allocator, builder, tensor_name, source_tensor);
    }

    // Convert source to F32.
    // FP8 sources must be dequantized with weight_scale_inv before requantization.
    const f32_source = try tensorToF32ForQuantization(allocator, source_tensors, tensor_name, source_tensor);
    defer f32_source.deinit(allocator);

    const source_values = f32_source.asF32Slice();

    // Calculate output sizes
    const packed_col_count = col_count / values_per_u32;
    const group_count = col_count / group_len;

    // Allocate output buffers
    const packed_row_words = try allocator.alloc(u32, row_count * packed_col_count);
    defer allocator.free(packed_row_words);

    const scale_values = try allocator.alloc(u16, row_count * group_count);
    defer allocator.free(scale_values);

    const bias_values = try allocator.alloc(u16, row_count * group_count);
    defer allocator.free(bias_values);

    // Quantize rows in parallel
    var quant_ctx = RowQuantizeContext{
        .source_data = source_values,
        .packed_row_data = packed_row_words,
        .scale_data = scale_values,
        .bias_data = bias_values,
        .col_count = col_count,
        .packed_col_count = packed_col_count,
        .group_count = group_count,
        .group_len = group_len,
        .quant_bits = quant_bits,
    };

    const pool = parallel.global();
    pool.parallelFor(row_count, quantizeRowSlice, &quant_ctx);

    // Add tensors to builder
    const quant_dtype: DType = if (quant_bits == 4) .grouped_affine_u4 else .grouped_affine_u8;
    try builder.addTensor(
        tensor_name,
        quant_dtype,
        &[_]usize{ row_count, packed_col_count },
        std.mem.sliceAsBytes(packed_row_words),
    );

    // Scales and biases names
    var scales_name_buf: [256]u8 = undefined;
    const tensor_base_name = if (std.mem.endsWith(u8, tensor_name, ".weight"))
        tensor_name[0 .. tensor_name.len - ".weight".len]
    else
        tensor_name;
    const scales_tensor_name = try std.fmt.bufPrint(&scales_name_buf, "{s}.scales", .{tensor_base_name});
    try builder.addTensor(
        scales_tensor_name,
        .bf16,
        &[_]usize{ row_count, group_count },
        std.mem.sliceAsBytes(scale_values),
    );

    var biases_name_buf: [256]u8 = undefined;
    const biases_tensor_name = try std.fmt.bufPrint(&biases_name_buf, "{s}.biases", .{tensor_base_name});
    try builder.addTensor(
        biases_tensor_name,
        .bf16,
        &[_]usize{ row_count, group_count },
        std.mem.sliceAsBytes(bias_values),
    );
}

fn tensorToF32ForQuantization(
    allocator: std.mem.Allocator,
    source_tensors: *safetensors.UnifiedSafeTensors,
    tensor_name: []const u8,
    source_tensor: Tensor,
) !convert.F32Result {
    if (source_tensor.dtype != .f8_e4m3) return convert.tensorToF32(allocator, source_tensor);

    // FP8 tensors are quantized and require inverse scales for correct values.
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

fn dequantizeFp8WithScale(
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

    // Scalar scale fallback (first element).
    const scale_inv = try scaleToF32(scale_tensor, 0);
    for (src_u8, f32_values) |src_val, *dst_ptr| {
        dst_ptr.* = dtype_mod.fp8e4m3ToF32(src_val) * scale_inv;
    }
    return .{ .data = std.mem.sliceAsBytes(f32_values), .owned = f32_values };
}

/// Copy a tensor without quantization - preserves original dtype.
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
// Tests
// =============================================================================

test "QuantConfig: defaults" {
    const config = QuantConfig{ .bits = 4, .group_size = 64 };
    try std.testing.expectEqual(@as(u8, 4), config.bits);
    try std.testing.expectEqual(@as(u32, 64), config.group_size);
}

// =============================================================================
// QuantConfig Tests
// =============================================================================

test "QuantConfig: various bit depths" {
    {
        const config = QuantConfig{ .bits = 4, .group_size = 64 };
        try std.testing.expectEqual(@as(u8, 4), config.bits);
    }
    {
        const config = QuantConfig{ .bits = 8, .group_size = 128 };
        try std.testing.expectEqual(@as(u8, 8), config.bits);
    }
}

test "QuantConfig: various group sizes" {
    {
        const config = QuantConfig{ .bits = 4, .group_size = 32 };
        try std.testing.expectEqual(@as(u32, 32), config.group_size);
    }
    {
        const config = QuantConfig{ .bits = 4, .group_size = 64 };
        try std.testing.expectEqual(@as(u32, 64), config.group_size);
    }
    {
        const config = QuantConfig{ .bits = 4, .group_size = 128 };
        try std.testing.expectEqual(@as(u32, 128), config.group_size);
    }
    {
        const config = QuantConfig{ .bits = 4, .group_size = 256 };
        try std.testing.expectEqual(@as(u32, 256), config.group_size);
    }
}

// =============================================================================
// ConvertOptions Tests
// =============================================================================

test "ConvertOptions: default values" {
    const options = ConvertOptions{};
    try std.testing.expect(options.quant == null);
    try std.testing.expectEqualStrings("models", options.output_dir);
    try std.testing.expect(!options.force);
}

test "ConvertOptions: with quantization" {
    const options = ConvertOptions{
        .quant = .{ .bits = 4, .group_size = 64 },
        .output_dir = "custom_output",
        .force = true,
    };
    try std.testing.expect(options.quant != null);
    try std.testing.expectEqual(@as(u8, 4), options.quant.?.bits);
    try std.testing.expectEqual(@as(u32, 64), options.quant.?.group_size);
    try std.testing.expectEqualStrings("custom_output", options.output_dir);
    try std.testing.expect(options.force);
}

test "ConvertOptions: without quantization (preserve precision)" {
    const options = ConvertOptions{
        .quant = null,
        .output_dir = "output",
    };
    try std.testing.expect(options.quant == null);
}

test "isCompleteConversionOutput requires config and weights" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{ .sub_path = "config.json", .data = "{}" });
    {
        var dir = try tmp.dir.openDir(".", .{});
        defer dir.close();
        try std.testing.expect(!isCompleteConversionOutput(dir));
    }

    try tmp.dir.writeFile(.{ .sub_path = "model.safetensors", .data = "x" });
    {
        var dir = try tmp.dir.openDir(".", .{});
        defer dir.close();
        try std.testing.expect(isCompleteConversionOutput(dir));
    }
}

// =============================================================================
// quantizeRowSlice Tests
// =============================================================================

test "convertToGroupedAffine/quantizeRowSlice: single row 4-bit quantization" {
    const allocator = std.testing.allocator;

    // Create test data: 64 values (2 groups of 32)
    const col_count: usize = 64;
    const group_len: usize = 32;
    const group_count: usize = col_count / group_len;

    const source_data = try allocator.alloc(f32, col_count);
    defer allocator.free(source_data);

    // Fill with test values
    for (source_data, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }

    // Allocate output buffers
    const packed_col_count: usize = col_count / 8; // 8 values per u32 for 4-bit
    const packed_row_data = try allocator.alloc(u32, packed_col_count);
    defer allocator.free(packed_row_data);

    const scale_data = try allocator.alloc(u16, group_count);
    defer allocator.free(scale_data);

    const bias_data = try allocator.alloc(u16, group_count);
    defer allocator.free(bias_data);

    var ctx = RowQuantizeContext{
        .source_data = source_data,
        .packed_row_data = packed_row_data,
        .scale_data = scale_data,
        .bias_data = bias_data,
        .col_count = col_count,
        .packed_col_count = packed_col_count,
        .group_count = group_count,
        .group_len = group_len,
        .quant_bits = 4,
    };

    quantizeRowSlice(0, 1, &ctx);

    // Verify scales and biases were computed
    try std.testing.expect(scale_data[0] != 0);
    try std.testing.expect(scale_data[1] != 0);
}

test "convertToGroupedAffine/quantizeRowSlice: 8-bit quantization" {
    const allocator = std.testing.allocator;

    const col_count: usize = 32;
    const group_len: usize = 32;
    const group_count: usize = 1;

    const source_data = try allocator.alloc(f32, col_count);
    defer allocator.free(source_data);

    // Fill with test pattern
    for (source_data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i)) * 0.1;
    }

    const packed_col_count: usize = col_count / 4; // 4 values per u32 for 8-bit
    const packed_row_data = try allocator.alloc(u32, packed_col_count);
    defer allocator.free(packed_row_data);

    const scale_data = try allocator.alloc(u16, group_count);
    defer allocator.free(scale_data);

    const bias_data = try allocator.alloc(u16, group_count);
    defer allocator.free(bias_data);

    var ctx = RowQuantizeContext{
        .source_data = source_data,
        .packed_row_data = packed_row_data,
        .scale_data = scale_data,
        .bias_data = bias_data,
        .col_count = col_count,
        .packed_col_count = packed_col_count,
        .group_count = group_count,
        .group_len = group_len,
        .quant_bits = 8,
    };

    quantizeRowSlice(0, 1, &ctx);

    // Verify outputs are non-zero
    try std.testing.expect(scale_data[0] != 0);
    try std.testing.expect(packed_row_data[0] != 0);
}

test "convertToGroupedAffine/quantizeRowSlice: multiple rows" {
    const allocator = std.testing.allocator;

    const row_count: usize = 4;
    const col_count: usize = 64;
    const group_len: usize = 64;
    const group_count: usize = 1;

    const source_data = try allocator.alloc(f32, row_count * col_count);
    defer allocator.free(source_data);

    // Fill each row with different value ranges to ensure different scales
    for (0..row_count) |row| {
        const row_multiplier: f32 = @as(f32, @floatFromInt((row + 1) * 100));
        for (0..col_count) |col| {
            source_data[row * col_count + col] = row_multiplier + @as(f32, @floatFromInt(col));
        }
    }

    const packed_col_count: usize = col_count / 8;
    const packed_row_data = try allocator.alloc(u32, row_count * packed_col_count);
    defer allocator.free(packed_row_data);

    const scale_data = try allocator.alloc(u16, row_count * group_count);
    defer allocator.free(scale_data);

    const bias_data = try allocator.alloc(u16, row_count * group_count);
    defer allocator.free(bias_data);

    var ctx = RowQuantizeContext{
        .source_data = source_data,
        .packed_row_data = packed_row_data,
        .scale_data = scale_data,
        .bias_data = bias_data,
        .col_count = col_count,
        .packed_col_count = packed_col_count,
        .group_count = group_count,
        .group_len = group_len,
        .quant_bits = 4,
    };

    // Quantize all rows
    quantizeRowSlice(0, row_count, &ctx);

    // Verify each row has different bias (due to different value ranges)
    const bias0 = dtype_mod.bf16ToF32(bias_data[0]);
    const bias1 = dtype_mod.bf16ToF32(bias_data[1]);
    const bias2 = dtype_mod.bf16ToF32(bias_data[2]);
    const bias3 = dtype_mod.bf16ToF32(bias_data[3]);

    // Each row starts at a different offset, so biases should be clearly different
    try std.testing.expect(@abs(bias1 - bias0) > 50.0);
    try std.testing.expect(@abs(bias2 - bias1) > 50.0);
    try std.testing.expect(@abs(bias3 - bias2) > 50.0);
}

test "convertToGroupedAffine/quantizeRowSlice: zero values" {
    const allocator = std.testing.allocator;

    const col_count: usize = 32;
    const group_len: usize = 32;
    const group_count: usize = 1;

    const source_data = try allocator.alloc(f32, col_count);
    defer allocator.free(source_data);
    @memset(source_data, 0.0);

    const packed_col_count: usize = col_count / 8;
    const packed_row_data = try allocator.alloc(u32, packed_col_count);
    defer allocator.free(packed_row_data);

    const scale_data = try allocator.alloc(u16, group_count);
    defer allocator.free(scale_data);

    const bias_data = try allocator.alloc(u16, group_count);
    defer allocator.free(bias_data);

    var ctx = RowQuantizeContext{
        .source_data = source_data,
        .packed_row_data = packed_row_data,
        .scale_data = scale_data,
        .bias_data = bias_data,
        .col_count = col_count,
        .packed_col_count = packed_col_count,
        .group_count = group_count,
        .group_len = group_len,
        .quant_bits = 4,
    };

    quantizeRowSlice(0, 1, &ctx);

    // Scale should be zero for all-zero data
    const scale_f32 = dtype_mod.bf16ToF32(scale_data[0]);
    try std.testing.expectEqual(@as(f32, 0.0), scale_f32);
}

test "convertToGroupedAffine/quantizeRowSlice: negative values" {
    const allocator = std.testing.allocator;

    const col_count: usize = 32;
    const group_len: usize = 32;
    const group_count: usize = 1;

    const source_data = try allocator.alloc(f32, col_count);
    defer allocator.free(source_data);

    // Fill with negative values
    for (source_data, 0..) |*val, i| {
        val.* = -@as(f32, @floatFromInt(i + 1));
    }

    const packed_col_count: usize = col_count / 8;
    const packed_row_data = try allocator.alloc(u32, packed_col_count);
    defer allocator.free(packed_row_data);

    const scale_data = try allocator.alloc(u16, group_count);
    defer allocator.free(scale_data);

    const bias_data = try allocator.alloc(u16, group_count);
    defer allocator.free(bias_data);

    var ctx = RowQuantizeContext{
        .source_data = source_data,
        .packed_row_data = packed_row_data,
        .scale_data = scale_data,
        .bias_data = bias_data,
        .col_count = col_count,
        .packed_col_count = packed_col_count,
        .group_count = group_count,
        .group_len = group_len,
        .quant_bits = 4,
    };

    quantizeRowSlice(0, 1, &ctx);

    // Should handle negative values correctly
    const bias_f32 = dtype_mod.bf16ToF32(bias_data[0]);
    try std.testing.expect(bias_f32 < 0.0);
}

test "convertToGroupedAffine/quantizeRowSlice: mixed positive and negative" {
    const allocator = std.testing.allocator;

    const col_count: usize = 64;
    const group_len: usize = 32;
    const group_count: usize = 2;

    const source_data = try allocator.alloc(f32, col_count);
    defer allocator.free(source_data);

    // First group: positive values
    for (0..32) |i| {
        source_data[i] = @floatFromInt(i);
    }
    // Second group: negative values
    for (32..64) |i| {
        source_data[i] = -@as(f32, @floatFromInt(i - 32));
    }

    const packed_col_count: usize = col_count / 8;
    const packed_row_data = try allocator.alloc(u32, packed_col_count);
    defer allocator.free(packed_row_data);

    const scale_data = try allocator.alloc(u16, group_count);
    defer allocator.free(scale_data);

    const bias_data = try allocator.alloc(u16, group_count);
    defer allocator.free(bias_data);

    var ctx = RowQuantizeContext{
        .source_data = source_data,
        .packed_row_data = packed_row_data,
        .scale_data = scale_data,
        .bias_data = bias_data,
        .col_count = col_count,
        .packed_col_count = packed_col_count,
        .group_count = group_count,
        .group_len = group_len,
        .quant_bits = 4,
    };

    quantizeRowSlice(0, 1, &ctx);

    // First group bias should be non-negative (min is 0)
    const bias0 = dtype_mod.bf16ToF32(bias_data[0]);
    try std.testing.expect(bias0 >= 0.0);

    // Second group bias should be negative (min is negative)
    const bias1 = dtype_mod.bf16ToF32(bias_data[1]);
    try std.testing.expect(bias1 < 0.0);
}

test "dequantizeFp8WithScale applies scalar scale" {
    var weight_bytes = [_]u8{ 0x48, 0x48, 0x48, 0x48 }; // FP8 value 4.0
    var scale_vals = [_]u16{dtype_mod.f32ToBf16(0.5)};

    const weight = Tensor.view(weight_bytes[0..].ptr, &.{ 2, 2 }, .f8_e4m3, weight_bytes.len);
    const scale = Tensor.view(std.mem.sliceAsBytes(scale_vals[0..]).ptr, &.{1}, .bf16, scale_vals.len * @sizeOf(u16));

    const res = try dequantizeFp8WithScale(std.testing.allocator, weight, scale);
    defer res.deinit(std.testing.allocator);

    const out = res.asF32Slice();
    for (out) |v| try std.testing.expectApproxEqAbs(@as(f32, 2.0), v, 1e-6);
}

test "dequantizeFp8WithScale applies per-block scales" {
    var weight_bytes = [_]u8{ 0x48, 0x48, 0x48, 0x48 }; // FP8 value 4.0
    var scale_vals = [_]u16{
        dtype_mod.f32ToBf16(0.5), // block (0,0)
        dtype_mod.f32ToBf16(1.0), // block (0,1)
        dtype_mod.f32ToBf16(2.0), // block (1,0)
        dtype_mod.f32ToBf16(4.0), // block (1,1)
    };

    const weight = Tensor.view(weight_bytes[0..].ptr, &.{ 2, 2 }, .f8_e4m3, weight_bytes.len);
    const scale = Tensor.view(std.mem.sliceAsBytes(scale_vals[0..]).ptr, &.{ 2, 2 }, .bf16, scale_vals.len * @sizeOf(u16));

    const res = try dequantizeFp8WithScale(std.testing.allocator, weight, scale);
    defer res.deinit(std.testing.allocator);

    const out = res.asF32Slice();
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), out[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), out[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 16.0), out[3], 1e-6);
}
