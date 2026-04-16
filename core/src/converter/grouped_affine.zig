//! Grouped-affine Model Conversion (MLX-compatible export)
//!
//! Converts transformer models to grouped-affine quantization and exports
//! them in MLX-compatible SafeTensors layout.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const log = @import("log_pkg");
const tensor = @import("tensor_pkg");
const dtype_mod = @import("dtype_pkg");
const safetensors = @import("io_pkg").safetensors.root;
const repository = @import("io_pkg").repository.root;
const gaf_paths = @import("gaf_paths.zig");
const config_loader = @import("models_pkg").config;
const op_types = @import("models_pkg").op_types;
const parallel = @import("compute_pkg").parallel;
const compute = @import("compute_pkg");
const convert = @import("root.zig");
const models_registry = @import("models_pkg").registry;
const load_transforms = @import("models_pkg").load.transforms;
const json = @import("io_pkg").json;
const http = @import("io_pkg").transport.http;
const tokenizer_mod = @import("../tokenizer/root.zig");
const calibration_capture = @import("calibration_capture.zig");
const xray = @import("xray_pkg");
const has_metal_gpu_calib = build_options.enable_metal and builtin.os.tag == .macos;
const has_cuda_gpu_calib = build_options.enable_cuda and
    (builtin.os.tag == .linux or builtin.os.tag == .windows);
extern fn mlx_runtime_binary_dir() ?[*:0]const u8;

const Tensor = tensor.Tensor;
const DType = dtype_mod.DType;
const max_calibration_input_samples: usize = 4096;
const calibration_rows_max_bytes: usize = 32 * 1024 * 1024;
const calibration_eval_max_bytes: usize = 32 * 1024 * 1024;
const cuda_grouped_quant_u4_symbol: [:0]const u8 = "talu_gaffine_quantize_u4_f32";
const cuda_grouped_quant_u8_symbol: [:0]const u8 = "talu_gaffine_quantize_u8_f32";
const cuda_grouped_build_dq_symbol: [:0]const u8 = "talu_gaffine_build_dq_weights_f32";
const cuda_reduce_mse_symbol: [:0]const u8 = "talu_reduce_mse_f32";
const cuda_grouped_quant_block_x: u32 = 128;
const cuda_grouped_build_dq_block_x: u32 = 256;
const cuda_calibration_default_buffer_mib: usize = 128;
const cuda_calibration_min_buffer_bytes: usize = 64 * 1024 * 1024;
const cuda_calibration_max_buffer_bytes: usize = 256 * 1024 * 1024;
const cuda_grouped_quant_default_tile_mib: usize = 1024;
const cuda_grouped_quant_min_tile_bytes: usize = 64 * 1024 * 1024;
const cuda_grouped_quant_max_tile_bytes: usize = 2 * 1024 * 1024 * 1024;
const cuda_grouped_quant_max_grid_y_rows: usize = 65535;
const cuda_reduce_mse_block_x: u32 = 256;
const cuda_reduce_mse_max_blocks: u32 = 4096;

/// Quantization configuration
pub const QuantConfig = struct {
    bits: u8,
    group_size: u32,
};

const CalibrationOptimizer = enum {
    search,
    clip,
    clip_search,
};

const CalibrationProgressMode = enum {
    block,
    layer,
};

const CalibrationBlockMap = struct {
    arch: ?*const op_types.Architecture = null,
    layer_types_override: ?[]const u8 = null,

    fn variantName(self: CalibrationBlockMap, layer: u32) ?[]const u8 {
        const arch = self.arch orelse return null;
        if (!arch.isHeterogeneous()) return null;
        const variant = arch.getVariantWithOverride(@intCast(layer), self.layer_types_override) orelse return null;
        return variant.name;
    }
};

fn calibrationOptimizerOverrideName(optimizer: CalibrationOptimizer) []const u8 {
    return switch (optimizer) {
        .search => "search",
        .clip => "clip",
        .clip_search => "clip+search",
    };
}

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
    profile: convert.scheme.QualityProfile = .good,
    calib_iters: u32 = 1,
    calib_nsamples: u32 = 16,
    calib_seqlen: u32 = 256,
    calib_batch_size: u32 = 1,
    calib_nblocks: u32 = 1,
    calib_seed: u64 = 42,

    /// Derive the Scheme from quant config for model card generation.
    pub fn toScheme(self: ConvertOptions) convert.scheme.Scheme {
        if (self.quant) |q| {
            // Map bits + group_size to scheme
            return switch (q.bits) {
                4 => switch (q.group_size) {
                    32 => .tq4_32,
                    64 => .tq4_64,
                    128 => .tq4_128,
                    else => .tq4_32,
                },
                8 => switch (q.group_size) {
                    32 => .tq8_32,
                    64 => .tq8_64,
                    128 => .tq8_128,
                    else => .tq8_32,
                },
                else => .f16,
            };
        }
        return .f16;
    }
};

pub const modelIdFromOutputPath = gaf_paths.modelIdFromOutputPath;

/// Concatenate multiple rank-2 tensors along axis 0 (rows).
/// All inputs must have the same number of columns and dtype.
/// Caller owns the returned tensor's data buffer.
fn concatWeightsAxis0(allocator: std.mem.Allocator, tensors: []const *const Tensor) !Tensor {
    if (tensors.len == 0) return error.InvalidWeightTransform;
    const cols: usize = @intCast(tensors[0].shape[1]);
    const dt = tensors[0].dtype;
    var total_rows: usize = 0;
    var total_bytes: usize = 0;
    for (tensors) |t| {
        if (t.n_dims != 2) return error.InvalidWeightTransform;
        if (@as(usize, @intCast(t.shape[1])) != cols) return error.InvalidWeightTransform;
        if (t.dtype != dt) return error.InvalidWeightTransform;
        total_rows += @intCast(t.shape[0]);
        total_bytes += t.data_size;
    }
    const buf = try allocator.alloc(u8, total_bytes);
    errdefer allocator.free(buf);
    var offset: usize = 0;
    for (tensors) |t| {
        const ptr = t.data_ptr orelse return error.InvalidWeightTransform;
        @memcpy(buf[offset .. offset + t.data_size], ptr[0..t.data_size]);
        offset += t.data_size;
    }
    var fused = std.mem.zeroes(Tensor);
    fused.dtype = dt;
    fused.n_dims = 2;
    fused.shape[0] = @intCast(total_rows);
    fused.shape[1] = @intCast(cols);
    fused.numel = total_rows * cols;
    fused.strides[0] = @intCast(cols);
    fused.strides[1] = 1;
    fused.data_ptr = buf.ptr;
    fused.data_size = total_bytes;
    fused.owns_data = false;
    return fused;
}

fn maybeWriteFusedTensorForPlan(
    allocator: std.mem.Allocator,
    cuda_ctx: ?*CudaCalibContext,
    source_tensors: *safetensors.UnifiedSafeTensors,
    builder: *safetensors.Builder,
    plan: *const convert.ConversionFusionPlan,
    quant_config: QuantConfig,
    options: ConvertOptions,
    token_pool: ?[]const u32,
    block_input_cache: *GroupedBlockInputCache,
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

            _ = try quantizeGroupedAffineTensor(allocator, cuda_ctx, source_tensors, builder, plan.output_name, fused.*, quant_config, options, token_pool, block_input_cache);
            return true;
        },
        .dense_mlp_gate_up => {
            if (plan.required_inputs.len != 2) return error.InvalidWeightTransform;
            const gate = try source_tensors.getTensor(plan.required_inputs[0], null);
            const up = try source_tensors.getTensor(plan.required_inputs[1], null);

            const fused_tensor = try concatWeightsAxis0(allocator, &.{ &gate, &up });
            defer allocator.free(fused_tensor.data_ptr.?[0..fused_tensor.data_size]);

            _ = try quantizeGroupedAffineTensor(allocator, cuda_ctx, source_tensors, builder, plan.output_name, fused_tensor, quant_config, options, token_pool, block_input_cache);
            return true;
        },
        .attention_qkv => {
            if (plan.required_inputs.len != 3) return error.InvalidWeightTransform;
            const q = try source_tensors.getTensor(plan.required_inputs[0], null);
            const k = try source_tensors.getTensor(plan.required_inputs[1], null);
            const v = try source_tensors.getTensor(plan.required_inputs[2], null);

            const fused_tensor = try concatWeightsAxis0(allocator, &.{ &q, &k, &v });
            defer allocator.free(fused_tensor.data_ptr.?[0..fused_tensor.data_size]);

            _ = try quantizeGroupedAffineTensor(allocator, cuda_ctx, source_tensors, builder, plan.output_name, fused_tensor, quant_config, options, token_pool, block_input_cache);
            return true;
        },
    }
}

fn calibrationDatasetSnapshotDir(allocator: std.mem.Allocator) ![]u8 {
    const hf_home = try repository.cache.getHfHome(allocator);
    defer allocator.free(hf_home);
    return std.fs.path.join(allocator, &.{ hf_home, "hub", "datasets--NeelNanda--pile-10k", "snapshots", "main" });
}

fn calibrationRowsCachePath(allocator: std.mem.Allocator, offset: usize, length: usize) ![]u8 {
    const snapshot_dir = try calibrationDatasetSnapshotDir(allocator);
    defer allocator.free(snapshot_dir);
    const file_name = try std.fmt.allocPrint(allocator, "rows-offset-{d}-length-{d}.json", .{ offset, length });
    defer allocator.free(file_name);
    return std.fs.path.join(allocator, &.{ snapshot_dir, file_name });
}

fn fetchDatasetRowsJson(allocator: std.mem.Allocator, offset: usize, length: usize) ![]u8 {
    const cache_path = try calibrationRowsCachePath(allocator, offset, length);
    defer allocator.free(cache_path);

    if (std.fs.cwd().openFile(cache_path, .{})) |cache_file| {
        defer cache_file.close();
        return cache_file.readToEndAlloc(allocator, calibration_rows_max_bytes);
    } else |err| switch (err) {
        error.FileNotFound => {},
        else => return err,
    }

    const url = try std.fmt.allocPrint(
        allocator,
        "https://datasets-server.huggingface.co/rows?dataset=NeelNanda/pile-10k&config=default&split=train&offset={d}&length={d}",
        .{ offset, length },
    );
    defer allocator.free(url);
    const payload = try http.fetch(allocator, url, .{
        .user_agent = "talu-convert/1.0",
        .max_response_bytes = calibration_rows_max_bytes,
    });
    errdefer allocator.free(payload);

    const parent = std.fs.path.dirname(cache_path) orelse return error.NotFound;
    try std.fs.cwd().makePath(parent);
    var out_file = try std.fs.cwd().createFile(cache_path, .{ .truncate = true });
    defer out_file.close();
    try out_file.writeAll(payload);

    return payload;
}

fn fetchDatasetRowsJsonWithRetry(
    allocator: std.mem.Allocator,
    offset: usize,
    length: usize,
) ![]u8 {
    const max_attempts: usize = 6;
    var attempt: usize = 0;
    while (attempt < max_attempts) : (attempt += 1) {
        const payload = fetchDatasetRowsJson(allocator, offset, length) catch |err| {
            switch (err) {
                error.NotFound, error.Unauthorized => return err,
                else => {},
            }
            if (attempt + 1 == max_attempts) return err;
            std.Thread.sleep(@as(u64, @intCast(120 + attempt * 120)) * std.time.ns_per_ms);
            continue;
        };
        return payload;
    }
    return error.HttpError;
}

fn appendTokenizedRows(
    allocator: std.mem.Allocator,
    tokenizer: *tokenizer_mod.Tokenizer,
    json_bytes: []const u8,
    out_tokens: *std.ArrayListUnmanaged(u32),
    target_tokens: usize,
) !usize {
    var parsed = json.parseValue(allocator, json_bytes, .{
        .max_size_bytes = calibration_rows_max_bytes,
        .max_value_bytes = calibration_rows_max_bytes,
        .max_string_bytes = 8 * 1024 * 1024,
    }) catch |err| {
        return switch (err) {
            error.InputTooLarge,
            error.InputTooDeep,
            error.StringTooLong,
            error.InvalidJson,
            => error.InvalidConfig,
            error.OutOfMemory => error.OutOfMemory,
        };
    };
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidConfig;

    const rows_ptr = parsed.value.object.get("rows") orelse return error.NotFound;
    if (rows_ptr != .array) return error.InvalidConfig;

    const before = out_tokens.items.len;
    for (rows_ptr.array.items) |row_entry| {
        if (out_tokens.items.len >= target_tokens) break;
        if (row_entry != .object) continue;
        const row_obj_ptr = row_entry.object.get("row") orelse continue;
        if (row_obj_ptr != .object) continue;
        const text_ptr = row_obj_ptr.object.get("text") orelse continue;
        if (text_ptr != .string) continue;

        const token_ids = tokenizer.encodeSliceWithOptions(text_ptr.string, .{ .add_special_tokens = false }) catch continue;
        defer allocator.free(token_ids);
        if (token_ids.len == 0) continue;

        const remaining = target_tokens - out_tokens.items.len;
        const take = @min(remaining, token_ids.len);
        try out_tokens.appendSlice(allocator, token_ids[0..take]);
    }
    return out_tokens.items.len - before;
}

fn loadCalibrationTokenPool(allocator: std.mem.Allocator, tokenizer_path: []const u8, options: ConvertOptions) !?[]u32 {
    if (tokenizer_path.len == 0) return null;
    if (options.calib_iters == 0) return null;

    var tokenizer = try tokenizer_mod.Tokenizer.initFromPath(allocator, tokenizer_path);
    defer tokenizer.deinit();

    var tokens = std.ArrayListUnmanaged(u32){};
    errdefer tokens.deinit(allocator);

    const requested_u64 = @as(u64, @max(options.calib_nsamples, 1)) * @as(u64, @max(options.calib_seqlen, 1));
    const requested = std.math.cast(usize, requested_u64) orelse std.math.maxInt(usize);
    const target_tokens = @max(@as(usize, 1024), requested);
    const rows_per_page: usize = 64;
    const total_rows: usize = 10_000;
    const max_pages_from_target = @max(@as(usize, 24), @divTrunc(target_tokens + 2047, 2048));
    const max_pages: usize = @min(@as(usize, 256), max_pages_from_target);
    const seed_offset: usize = @intCast(options.calib_seed % (total_rows - rows_per_page));

    var page: usize = 0;
    while (page < max_pages and tokens.items.len < target_tokens) : (page += 1) {
        const offset = (seed_offset + page * rows_per_page) % (total_rows - rows_per_page);
        const rows_json = fetchDatasetRowsJsonWithRetry(
            allocator,
            offset,
            rows_per_page,
        ) catch |err| {
            log.warn("convert", "Calibration rows fetch failed", .{
                .dataset = "NeelNanda/pile-10k",
                .offset = offset,
                .rows = rows_per_page,
                .err = @errorName(err),
            });
            return error.CalibrationDataUnavailable;
        };
        defer allocator.free(rows_json);
        const appended = appendTokenizedRows(allocator, &tokenizer, rows_json, &tokens, target_tokens) catch |err| {
            log.warn("convert", "Calibration rows parse/tokenize failed", .{
                .dataset = "NeelNanda/pile-10k",
                .offset = offset,
                .rows = rows_per_page,
                .err = @errorName(err),
            });
            return error.CalibrationDataUnavailable;
        };
        if (appended == 0) {
            log.warn("convert", "Calibration rows yielded no tokens", .{
                .dataset = "NeelNanda/pile-10k",
                .offset = offset,
                .rows = rows_per_page,
            });
            return error.CalibrationDataUnavailable;
        }
    }

    if (tokens.items.len < requested) {
        log.warn("convert", "Calibration token pool coverage insufficient", .{
            .required_tokens = requested,
            .loaded_tokens = tokens.items.len,
            .dataset = "NeelNanda/pile-10k",
        });
        return error.CalibrationDataUnavailable;
    }
    if (tokens.items.len == 0) return error.CalibrationDataUnavailable;
    const owned = try tokens.toOwnedSlice(allocator);
    return owned;
}

pub fn loadCalibrationTokenPoolForConvert(
    allocator: std.mem.Allocator,
    tokenizer_path: []const u8,
    options: ConvertOptions,
) !?[]u32 {
    return loadCalibrationTokenPool(allocator, tokenizer_path, options);
}

inline fn calibrationDatasetRequired(options: ConvertOptions) bool {
    return options.calib_iters > 0;
}

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
    const output_tmp_path = try std.fmt.allocPrint(allocator, "{s}.tmp", .{output_dir_path});
    defer allocator.free(output_tmp_path);

    // 4. Check if output exists
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
    errdefer if (!keep_output) std.fs.cwd().deleteTree(output_tmp_path) catch {};
    var output_dir = try gaf_paths.GAFModelDir.init(allocator, output_tmp_path);
    defer output_dir.deinit();

    // 9. Build weight layout map from static model architecture metadata.

    // Read model_type and look up architecture
    const model_type = try config_loader.readModelType(allocator, model_bundle.config_path());
    defer if (model_type) |mt| allocator.free(mt);

    var layout_map: ?convert.WeightLayoutMap = null;
    defer if (layout_map) |*lm| lm.deinit();
    var fusion_map: ?convert.ConversionFusionMap = null;
    defer if (fusion_map) |*fm| fm.deinit();

    var runtime_arch: ?*const op_types.Architecture = null;
    if (model_type) |mt| {
        // Primary path: resolve static runtime architecture directly by model_type.
        runtime_arch = models_registry.runtimeArchitectureByModelType(mt);
        // Fallback path: route via registry entry id for compatibility.
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

    // Quantized conversion requires architecture-driven layout metadata.
    // This keeps name semantics in static model metadata, not in converter internals.
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
            output_tmp_path,
            model_bundle.tokenizer_path(),
            options.progress,
            options,
            if (layout_map) |*lm| lm else null,
            if (fusion_map) |*fm| fm else null,
            runtime_arch,
            layer_types_override,
        );
    } else {
        try writeUnquantizedWeights(
            allocator,
            &source_tensors,
            model_config.tie_word_embeddings,
            options.max_shard_size,
            output_tmp_path,
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
    try convert.copyConfigWithGAFQuantization(allocator, model_bundle.config_path(), output_tmp_path, quant_config);

    // 11. Copy all model assets (tokenizer, chat template, vocab, etc.)
    try convert.copyModelAssets(allocator, model_bundle.dir, output_tmp_path);

    // 12. Generate Model Card (README.md)
    const model_name = convert.model_card.extractModelName(input_path);
    const base_model_id = convert.model_card.extractBaseModelId(input_path);
    const scheme = options.toScheme();
    convert.model_card.writeModelCard(allocator, output_tmp_path, model_name, base_model_id, scheme) catch |err| {
        // Log but don't fail conversion if model card generation fails
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

fn hasNonEmptyFile(dir: std.fs.Dir, sub_path: []const u8) bool {
    var file = dir.openFile(sub_path, .{}) catch return false;
    defer file.close();
    const size = file.getEndPos() catch return false;
    return size > 0;
}

fn hasValidSafetensorsFile(dir: std.fs.Dir, sub_path: []const u8) bool {
    var file = dir.openFile(sub_path, .{}) catch return false;
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

fn isCompleteConversionOutput(dir: std.fs.Dir) bool {
    dir.access("config.json", .{}) catch return false;
    if (hasValidSafetensorsFile(dir, "model.safetensors")) return true;
    return hasNonEmptyFile(dir, "model.safetensors.index.json");
}

/// Quantize all tensors and write to SafeTensors file.
fn writeQuantizedWeights(
    allocator: std.mem.Allocator,
    source_tensors: *safetensors.UnifiedSafeTensors,
    quant_config: QuantConfig,
    tie_embeddings: bool,
    max_shard_size: u64,
    output_dir: []const u8,
    tokenizer_path: []const u8,
    progress: ProgressContext,
    options: ConvertOptions,
    layout_map: ?*const convert.WeightLayoutMap,
    fusion_map: ?*const convert.ConversionFusionMap,
    runtime_arch: ?*const op_types.Architecture,
    layer_types_override: ?[]const u8,
) !void {
    const calib_start_ms = std.time.milliTimestamp();
    var tensor_builder = safetensors.Builder.init(allocator);
    defer tensor_builder.deinit();

    // Set sharding configuration
    tensor_builder.max_shard_size = max_shard_size;

    const tensor_names = try source_tensors.tensorNames(allocator);
    defer allocator.free(tensor_names);
    convert.sortTensorNames(tensor_names);

    // Initialize CUDA calibration context if BACKEND=cuda
    var cuda_calib_ctx: ?CudaCalibContext = if (comptime has_cuda_gpu_calib) blk: {
        if (!isCudaCalibrationEnabled()) break :blk null;
        break :blk CudaCalibContext.init() catch |err| {
            log.warn("convert", "CUDA calibration init failed; falling back to CPU", .{
                .err = @errorName(err),
            });
            break :blk null;
        };
    } else null;
    defer if (cuda_calib_ctx) |*ctx| ctx.deinit();
    if (cuda_calib_ctx) |*ctx| {
        if (isCudaGroupedQuantizationEnabled() and !ctx.cuda_quantization_available) {
            log.warn("convert", "CUDA grouped-affine quantization kernels unavailable; using CPU quantization", .{});
        }
    }
    // Keep converter Metal behavior consistent with inference: place mlx.metallib
    // at the MLX runtime loader path before backend selection.
    ensureMlxMetallibColocatedForCalibration();

    const show_calib_progress = options.calib_iters > 0 and
        envFlagEnabledDefault("TALU_CONVERT_CALIB_PROGRESS", true);
    const show_per_tensor_calib = show_calib_progress and envFlagEnabled("TALU_CONVERT_CALIB_PER_TENSOR");
    const calib_probe_only = show_calib_progress and envFlagEnabled("TALU_CONVERT_CALIB_PROBE_ONLY");
    const calib_layer_window = CalibrationLayerWindow.fromEnv();
    const optimizer_mode = calibrationOptimizerFromEnv(options.profile);
    const progress_mode = calibrationProgressModeFromEnv();
    const block_map: CalibrationBlockMap = .{
        .arch = runtime_arch,
        .layer_types_override = layer_types_override,
    };
    if (!show_calib_progress) {
        progress.addLine(0, "Converting", tensor_names.len, null, "tensors");
    } else {
        // The CLI may already have an active progress line for conversion. Close it so
        // per-layer calibration logs remain visible and newline-stable.
        progress.completeLine(0);
        std.debug.print("Calib {s} loss minimization\n", .{@tagName(progress_mode)});
        const scorer_backend = if (isMetalCalibrationEnabled()) "metal" else if (cuda_calib_ctx != null) "cuda" else "cpu";
        std.debug.print("Calib scorer backend: {s}\n", .{scorer_backend});
        std.debug.print("Calib optimizer: {s}\n", .{@tagName(optimizer_mode)});
        std.debug.print(
            "Calib tuning args: --profile custom optimizer={s},iters={d},samples={d},seqlen={d},batch_size={d},nblocks={d} --seed {d}\n",
            .{
                calibrationOptimizerOverrideName(optimizer_mode),
                options.calib_iters,
                options.calib_nsamples,
                options.calib_seqlen,
                options.calib_batch_size,
                options.calib_nblocks,
                options.calib_seed,
            },
        );
        if (optimizer_mode == .clip and options.calib_iters > 1) {
            std.debug.print("Calib note: optimizer=clip is clip-only; iterative search is disabled\n", .{});
        }
        if (progress_mode == .block and runtime_arch != null and runtime_arch.?.isHeterogeneous()) {
            std.debug.print("Calib block map: architecture variants enabled\n", .{});
        }
        if (calib_layer_window.enabled()) {
            std.debug.print(
                "Calib layer focus: [{d}..{d}] (outside range uses baseline quantization)\n",
                .{
                    calib_layer_window.min orelse 0,
                    calib_layer_window.max orelse std.math.maxInt(u32),
                },
            );
        }
        if (calib_probe_only) {
            std.debug.print("Calib probe-only mode: skipping artifact write and non-calibrated tensors\n", .{});
        }
    }
    var last_log_ms: i64 = 0;
    var last_percent: i64 = -1;
    var quantized_layers: usize = 0;
    var mse_sum: f64 = 0.0;
    var baseline_sum: f64 = 0.0;
    var best_running_normalized_ratio: f64 = 1.0;
    var active_layer: ?u32 = null;
    var active_layer_tensor_count: usize = 0;
    var active_layer_first_sum: f64 = 0.0;
    var active_layer_best_sum: f64 = 0.0;
    var active_layer_best_iter: usize = 0;
    var emitted_unit_count: usize = 0;
    var tuned_tensor_count: usize = 0;
    var baseline_tensor_count: usize = 0;
    const target_quantized_tensors = estimateGroupedTargetQuantizedTensorCount(
        tensor_names,
        layout_map,
        source_tensors,
        tie_embeddings,
        fusion_map,
        calib_probe_only,
        calib_layer_window,
    );
    const token_pool = blk: {
        const loaded = loadCalibrationTokenPool(allocator, tokenizer_path, options) catch |err| {
            if (calibrationDatasetRequired(options)) return err;
            log.warn("convert", "Failed to load calibration token pool; using deterministic block-input fallback activations", .{
                .err = @errorName(err),
                .dataset = "NeelNanda/pile-10k",
            });
            break :blk null;
        };
        if (loaded == null and calibrationDatasetRequired(options)) {
            return error.CalibrationDataUnavailable;
        }
        break :blk loaded;
    };
    defer if (token_pool) |pool| allocator.free(pool);
    const require_embedding_lookup = options.calib_iters > 0;
    var activation_capture_cache: ?calibration_capture.LayerActivationCache = null;
    defer if (activation_capture_cache) |*cache| cache.deinit();
    if (token_pool) |pool| {
        if (require_embedding_lookup and calibration_capture.isAvailable()) {
            if (std.fs.path.dirname(tokenizer_path)) |model_dir| {
                activation_capture_cache = calibration_capture.captureFromInference(
                    allocator,
                    model_dir,
                    pool,
                    .{
                        .seed = options.calib_seed,
                        .max_prompt_tokens = @max(@as(usize, 1), @min(@as(usize, @intCast(@max(options.calib_seqlen, 1))), @as(usize, 1024))),
                        .max_rows_per_key = @max(
                            @as(usize, 64),
                            @as(usize, @intCast(@max(options.calib_nsamples, 1))) *
                                @as(usize, @intCast(@max(options.calib_batch_size, 1))) *
                                @as(usize, @intCast(@max(options.calib_nblocks, 1))),
                        ),
                        .backend_selection = .auto,
                    },
                ) catch |err| blk: {
                    log.warn("convert", "Inference-backed activation capture unavailable; using embedding-derived calibration activations", .{
                        .err = @errorName(err),
                    });
                    break :blk null;
                };
                if (activation_capture_cache) |*cache| {
                    if (cache.count() == 0) {
                        cache.deinit();
                        activation_capture_cache = null;
                        log.warn("convert", "Inference-backed activation capture produced no layer activations; using embedding-derived calibration activations", .{});
                    }
                }
            } else {
                log.warn("convert", "Tokenizer path has no parent directory; using embedding-derived calibration activations", .{
                    .tokenizer_path = tokenizer_path,
                });
            }
        }
    }
    var block_input_cache = GroupedBlockInputCache.init(
        allocator,
        source_tensors,
        token_pool,
        options.calib_seed,
        require_embedding_lookup,
        if (activation_capture_cache) |*cache| cache else null,
    );
    defer block_input_cache.deinit();

    for (tensor_names, 0..) |tensor_name, tensor_index| {
        // Keep custom calibration output newline-stable by avoiding spinner rewrites.
        if (!show_calib_progress) {
            var msg_buf: [256]u8 = undefined;
            const copy_len = @min(tensor_name.len, msg_buf.len - 1);
            @memcpy(msg_buf[0..copy_len], tensor_name[0..copy_len]);
            msg_buf[copy_len] = 0;
            progress.updateLine(0, tensor_index + 1, @ptrCast(&msg_buf));
        }
        if (!show_calib_progress and tensor_names.len > 0) {
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

        if (fusion_map) |map| {
            if (map.isConsumedNonTrigger(tensor_name)) continue;
            if (map.planForTrigger(tensor_name)) |plan| {
                if (try maybeWriteFusedTensorForPlan(allocator, if (cuda_calib_ctx) |*ctx| ctx else null, source_tensors, &tensor_builder, plan, quant_config, options, token_pool, &block_input_cache)) {
                    continue;
                }
            }
        }

        {
            const source_tensor = try source_tensors.getTensor(tensor_name, null);

            // Use architecture-driven layout to determine if tensor should be quantized.
            // Unknown tensors are kept in source precision.
            if (convert.shouldQuantizeTensorByLayout(layout_map, tensor_name, source_tensor)) {
                const layer_index = extractLayerIndexFromTensorName(tensor_name);
                if (show_calib_progress) {
                    if (active_layer) |layer| {
                        if (layer_index == null or layer_index.? != layer) {
                            const running_avg_mse = mse_sum / @as(f64, @floatFromInt(@max(quantized_layers, 1)));
                            const running_avg_base = baseline_sum / @as(f64, @floatFromInt(@max(quantized_layers, 1)));
                            emitGroupedLayerCalibrationProgress(
                                progress_mode,
                                block_map,
                                layer,
                                active_layer_tensor_count,
                                active_layer_first_sum,
                                active_layer_best_sum,
                                active_layer_best_iter,
                                running_avg_mse,
                                running_avg_base,
                                best_running_normalized_ratio,
                                quantized_layers,
                                target_quantized_tensors,
                            );
                            const emitted_running_ratio = if (running_avg_base > 0) running_avg_mse / running_avg_base else 1.0;
                            if (emitted_running_ratio < best_running_normalized_ratio) {
                                best_running_normalized_ratio = emitted_running_ratio;
                            }
                            emitted_unit_count += 1;
                            active_layer = null;
                            active_layer_tensor_count = 0;
                            active_layer_first_sum = 0.0;
                            active_layer_best_sum = 0.0;
                            active_layer_best_iter = 0;
                        }
                    }
                }
                const tensor_options = blk: {
                    if (show_calib_progress and calib_layer_window.enabled() and !calib_layer_window.allows(layer_index)) {
                        var narrowed = options;
                        narrowed.calib_iters = 1;
                        break :blk narrowed;
                    }
                    break :blk options;
                };
                if (calib_probe_only and calib_layer_window.enabled() and !calib_layer_window.allows(layer_index)) {
                    continue;
                }
                if (show_calib_progress and calib_layer_window.enabled() and tensor_options.calib_iters == 1 and options.calib_iters > 1) {
                    baseline_tensor_count += 1;
                } else {
                    tuned_tensor_count += 1;
                }
                // TALU_CONVERT_EMBED_BITS controls embedding quantization:
                //   4  = grouped-affine U4 (default, same as linear weights)
                //   8  = grouped-affine U8 (higher quality, calibrated)
                //   16 = BF16 (lossless, largest)
                const embed_bits: ?u8 = blk: {
                    if (layout_map) |map| {
                        if (map.layouts.get(tensor_name)) |layout| {
                            if (layout == .embedding) {
                                const default_embed_bits: u8 = 8;
                                const bits: u8 = if (std.posix.getenv("TALU_CONVERT_EMBED_BITS")) |raw|
                                    std.fmt.parseInt(u8, raw, 10) catch default_embed_bits
                                else
                                    default_embed_bits;
                                break :blk bits;
                            }
                        }
                    }
                    break :blk null;
                };
                // BF16 embedding: skip quantization entirely.
                if (embed_bits) |eb| {
                    if (eb == 16) {
                        try copyTensorAsBf16(allocator, &tensor_builder, tensor_name, source_tensor);
                        quantized_layers += 1;
                        continue;
                    }
                }
                // Override quant bits for embedding (U8 vs U4).
                const effective_quant_config = blk: {
                    if (embed_bits) |eb| {
                        if ((eb == 4 or eb == 8) and eb != quant_config.bits) {
                            var embed_cfg = quant_config;
                            embed_cfg.bits = eb;
                            break :blk embed_cfg;
                        }
                    }
                    break :blk quant_config;
                };
                const calib_summary = try quantizeGroupedAffineTensor(allocator, if (cuda_calib_ctx) |*ctx| ctx else null, source_tensors, &tensor_builder, tensor_name, source_tensor, effective_quant_config, tensor_options, token_pool, &block_input_cache);
                quantized_layers += 1;
                mse_sum += calib_summary.best_mse;
                baseline_sum += calib_summary.baseline_mse;
                const running_avg_mse = mse_sum / @as(f64, @floatFromInt(@max(quantized_layers, 1)));
                const running_avg_base = baseline_sum / @as(f64, @floatFromInt(@max(quantized_layers, 1)));
                if (show_calib_progress) {
                    if (show_per_tensor_calib) {
                        emitGroupedTensorCalibrationProgress(tensor_name, calib_summary);
                    }
                    if (layer_index) |layer| {
                        if (active_layer == null) active_layer = layer;
                        active_layer_tensor_count += 1;
                        active_layer_first_sum += calib_summary.first_mse;
                        active_layer_best_sum += calib_summary.best_mse;
                        if (calib_summary.best_iter > active_layer_best_iter) {
                            active_layer_best_iter = calib_summary.best_iter;
                        }
                    } else if (!show_per_tensor_calib) {
                        emitGroupedNonLayerCalibrationProgress(
                            tensor_name,
                            running_avg_mse,
                            running_avg_base,
                            best_running_normalized_ratio,
                            quantized_layers,
                            target_quantized_tensors,
                            calib_summary,
                        );
                        const emitted_running_ratio = if (running_avg_base > 0) running_avg_mse / running_avg_base else 1.0;
                        if (emitted_running_ratio < best_running_normalized_ratio) {
                            best_running_normalized_ratio = emitted_running_ratio;
                        }
                    }
                }
                continue;
            }
            try copyTensorUnchanged(allocator, &tensor_builder, tensor_name, source_tensor);
        }
    }

    if (show_calib_progress) {
        if (active_layer) |layer| {
            const running_avg_mse = mse_sum / @as(f64, @floatFromInt(@max(quantized_layers, 1)));
            const running_avg_base = baseline_sum / @as(f64, @floatFromInt(@max(quantized_layers, 1)));
            emitGroupedLayerCalibrationProgress(
                progress_mode,
                block_map,
                layer,
                active_layer_tensor_count,
                active_layer_first_sum,
                active_layer_best_sum,
                active_layer_best_iter,
                running_avg_mse,
                running_avg_base,
                best_running_normalized_ratio,
                quantized_layers,
                target_quantized_tensors,
            );
            const emitted_running_ratio = if (running_avg_base > 0) running_avg_mse / running_avg_base else 1.0;
            if (emitted_running_ratio < best_running_normalized_ratio) {
                best_running_normalized_ratio = emitted_running_ratio;
            }
            emitted_unit_count += 1;
            active_layer = null;
            active_layer_tensor_count = 0;
            active_layer_first_sum = 0.0;
            active_layer_best_sum = 0.0;
            active_layer_best_iter = 0;
        }
    }

    if (show_calib_progress and quantized_layers > 0) {
        const avg_mse = mse_sum / @as(f64, @floatFromInt(quantized_layers));
        const avg_baseline = baseline_sum / @as(f64, @floatFromInt(quantized_layers));
        const avg_improvement_pct = if (avg_baseline > 0 and avg_mse <= avg_baseline)
            ((avg_baseline - avg_mse) / avg_baseline) * 100.0
        else
            0.0;
        std.debug.print(
            "Calib done tensors={d} units={d} mode={s} avg_improve={d:.2}%\n",
            .{ quantized_layers, emitted_unit_count, @tagName(progress_mode), avg_improvement_pct },
        );
        const normalized_ratio = if (avg_baseline > 0) avg_mse / avg_baseline else 1.0;
        std.debug.print(
            "Calib a2a: normalized_mse_ratio={d:.6}, relative_mse_reduction_pct={d:.2}%\n",
            .{ normalized_ratio, avg_improvement_pct },
        );
        if (calib_layer_window.enabled()) {
            std.debug.print("Calib focus stats: tuned_tensors={d} baseline_tensors={d}\n", .{ tuned_tensor_count, baseline_tensor_count });
        }
    }

    const calib_elapsed_ms = @max(@as(i64, 1), std.time.milliTimestamp() - calib_start_ms);
    const calib_tokens_per_layer = @as(u64, @max(options.calib_nsamples, 1)) * @as(u64, @max(options.calib_seqlen, 1));
    const total_calib_tokens = calib_tokens_per_layer * @as(u64, @intCast(@max(quantized_layers, 1)));
    const calib_tok_per_s = @as(f64, @floatFromInt(total_calib_tokens)) / (@as(f64, @floatFromInt(calib_elapsed_ms)) / 1000.0);
    log.info("convert", "GAF calibration summary", .{
        .layers = quantized_layers,
        .elapsed_ms = calib_elapsed_ms,
        .tokens = total_calib_tokens,
        .tok_per_s = calib_tok_per_s,
        .iters = options.calib_iters,
        .nsamples = options.calib_nsamples,
        .seqlen = options.calib_seqlen,
        .batch_size = options.calib_batch_size,
        .nblocks = options.calib_nblocks,
        .bits = quant_config.bits,
        .group_size = quant_config.group_size,
    });

    if (!show_calib_progress) {
        // Emit progress: conversion complete
        progress.completeLine(0);
    }

    // Write weights (single file or sharded based on max_shard_size)
    // Builder.save() handles sharding decision internally based on max_shard_size
    try tensor_builder.save(output_dir, "model.safetensors");
}

fn envFlagEnabledDefault(name: []const u8, default_value: bool) bool {
    const raw = std.posix.getenv(name) orelse return default_value;
    if (raw.len == 0) return true;
    if (std.mem.eql(u8, raw, "0")) return false;
    if (std.mem.eql(u8, raw, "false")) return false;
    if (std.mem.eql(u8, raw, "FALSE")) return false;
    if (std.mem.eql(u8, raw, "off")) return false;
    if (std.mem.eql(u8, raw, "OFF")) return false;
    return true;
}

fn envFlagEnabled(name: []const u8) bool {
    return envFlagEnabledDefault(name, false);
}

fn parseCalibrationOptimizer(raw: []const u8) ?CalibrationOptimizer {
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    if (trimmed.len == 0) return null;
    if (std.ascii.eqlIgnoreCase(trimmed, "search")) return .search;
    if (std.ascii.eqlIgnoreCase(trimmed, "clip")) return .clip;
    if (std.ascii.eqlIgnoreCase(trimmed, "clip_search") or
        std.ascii.eqlIgnoreCase(trimmed, "clip+search") or
        std.ascii.eqlIgnoreCase(trimmed, "clip-search"))
    {
        return .clip_search;
    }
    return null;
}

fn calibrationOptimizerFromEnv(profile: convert.scheme.QualityProfile) CalibrationOptimizer {
    if (std.posix.getenv("TALU_CONVERT_CALIB_OPTIMIZER")) |raw| {
        return parseCalibrationOptimizer(raw) orelse .search;
    }
    return switch (profile) {
        .good, .best => .clip_search,
        .custom => .search,
    };
}

fn parseCalibrationProgressMode(raw: []const u8) ?CalibrationProgressMode {
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    if (trimmed.len == 0) return null;
    if (std.ascii.eqlIgnoreCase(trimmed, "block") or std.ascii.eqlIgnoreCase(trimmed, "blocks")) return .block;
    if (std.ascii.eqlIgnoreCase(trimmed, "layer") or std.ascii.eqlIgnoreCase(trimmed, "layers")) return .layer;
    return null;
}

fn calibrationProgressModeFromEnv() CalibrationProgressMode {
    const raw = std.posix.getenv("TALU_CONVERT_CALIB_PROGRESS_UNIT") orelse return .block;
    return parseCalibrationProgressMode(raw) orelse .block;
}

fn estimateGroupedTargetQuantizedTensorCount(
    tensor_names: []const []const u8,
    layout_map: ?*const convert.WeightLayoutMap,
    source_tensors: *safetensors.UnifiedSafeTensors,
    tie_embeddings: bool,
    fusion_map: ?*const convert.ConversionFusionMap,
    calib_probe_only: bool,
    layer_window: CalibrationLayerWindow,
) usize {
    var total: usize = 0;
    for (tensor_names) |tensor_name| {
        if (convert.shouldSkipForTiedEmbeddingsByName(layout_map, tensor_name, tie_embeddings)) continue;
        if (fusion_map) |map| {
            if (map.isConsumedNonTrigger(tensor_name)) continue;
            if (map.planForTrigger(tensor_name) != null) continue;
        }
        const source_tensor = source_tensors.getTensor(tensor_name, null) catch continue;
        if (!convert.shouldQuantizeTensorByLayout(layout_map, tensor_name, source_tensor)) continue;
        if (calib_probe_only and layer_window.enabled()) {
            if (!layer_window.allows(extractLayerIndexFromTensorName(tensor_name))) continue;
        }
        total += 1;
    }
    return total;
}

fn parseLayerBoundEnv(name: []const u8) ?u32 {
    const raw = std.posix.getenv(name) orelse return null;
    if (raw.len == 0) return null;
    return std.fmt.parseInt(u32, raw, 10) catch null;
}

const CalibrationLayerWindow = struct {
    min: ?u32 = null,
    max: ?u32 = null,

    fn fromEnv() CalibrationLayerWindow {
        return .{
            .min = parseLayerBoundEnv("TALU_CONVERT_CALIB_LAYER_MIN"),
            .max = parseLayerBoundEnv("TALU_CONVERT_CALIB_LAYER_MAX"),
        };
    }

    fn enabled(self: CalibrationLayerWindow) bool {
        return self.min != null or self.max != null;
    }

    fn allows(self: CalibrationLayerWindow, layer_index: ?u32) bool {
        if (!self.enabled()) return true;
        const layer = layer_index orelse return false;
        if (self.min) |min_layer| {
            if (layer < min_layer) return false;
        }
        if (self.max) |max_layer| {
            if (layer > max_layer) return false;
        }
        return true;
    }
};

fn isMetalCalibrationEnabled() bool {
    if (comptime !has_metal_gpu_calib) return false;
    if (std.posix.getenv("BACKEND")) |raw| {
        const token = std.mem.trim(u8, raw, " \t\r\n");
        if (!std.ascii.eqlIgnoreCase(token, "metal") and
            !std.ascii.eqlIgnoreCase(token, "auto") and
            token.len > 0) return false;
    }
    if (!compute.metal.isAvailable()) return false;
    return isMlxMetallibAvailableForCalibration();
}

fn pathExistsAbsoluteOrCwd(path: []const u8) bool {
    if (path.len == 0) return false;
    if (std.fs.path.isAbsolute(path)) {
        std.fs.accessAbsolute(path, .{}) catch return false;
        return true;
    }
    std.fs.cwd().access(path, .{}) catch return false;
    return true;
}

fn mlxRuntimeBinaryDir() ?[]const u8 {
    if (comptime !has_metal_gpu_calib) return null;
    const raw = mlx_runtime_binary_dir() orelse return null;
    const dir = std.mem.sliceTo(raw, 0);
    if (dir.len == 0) return null;
    return dir;
}

fn isMlxMetallibAvailableForCalibration() bool {
    if (mlxRuntimeBinaryDir()) |runtime_dir| {
        var runtime_buf: [std.fs.max_path_bytes]u8 = undefined;
        const runtime_path = std.fmt.bufPrint(&runtime_buf, "{s}/mlx.metallib", .{runtime_dir}) catch "";
        if (runtime_path.len > 0 and pathExistsAbsoluteOrCwd(runtime_path)) return true;
    }

    if (std.posix.getenv("MLX_METALLIB")) |env_path| {
        const trimmed = std.mem.trim(u8, env_path, " \t\r\n");
        if (trimmed.len > 0) return pathExistsAbsoluteOrCwd(trimmed);
    }

    const candidates = [_][]const u8{
        "mlx.metallib",
        "zig-out/bin/mlx.metallib",
        "zig-out/lib/mlx.metallib",
        "deps/mlx/lib/mlx.metallib",
        "deps/mlx-src/build/mlx/backend/metal/kernels/mlx.metallib",
        "/opt/homebrew/bin/mlx.metallib",
        "/usr/local/bin/mlx.metallib",
    };
    for (candidates) |candidate| {
        if (pathExistsAbsoluteOrCwd(candidate)) return true;
    }
    return false;
}

fn ensureMlxMetallibColocatedForCalibration() void {
    if (comptime !has_metal_gpu_calib) return;
    if (!compute.metal.isAvailable()) return;
    const runtime_dir = mlxRuntimeBinaryDir() orelse return;

    var dst_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dst = std.fmt.bufPrint(&dst_buf, "{s}/mlx.metallib", .{runtime_dir}) catch return;
    if (pathExistsAbsoluteOrCwd(dst)) return;

    var exe_dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const exe_dir = std.fs.selfExeDirPath(&exe_dir_buf) catch "";
    var exe_candidate_buf: [std.fs.max_path_bytes]u8 = undefined;
    var exe_lib_candidate_buf: [std.fs.max_path_bytes]u8 = undefined;
    const candidate_exe = if (exe_dir.len > 0)
        (std.fmt.bufPrint(&exe_candidate_buf, "{s}/mlx.metallib", .{exe_dir}) catch "")
    else
        "";
    const candidate_exe_lib = if (exe_dir.len > 0)
        (std.fmt.bufPrint(&exe_lib_candidate_buf, "{s}/../lib/mlx.metallib", .{exe_dir}) catch "")
    else
        "";
    const candidate_env = std.posix.getenv("MLX_METALLIB") orelse "";

    const candidates = [_][]const u8{
        candidate_env,
        candidate_exe,
        candidate_exe_lib,
        "mlx.metallib",
        "zig-out/bin/mlx.metallib",
        "zig-out/lib/mlx.metallib",
        "deps/mlx/lib/mlx.metallib",
        "deps/mlx-src/build/mlx/backend/metal/kernels/mlx.metallib",
        "/opt/homebrew/bin/mlx.metallib",
        "/usr/local/bin/mlx.metallib",
    };
    for (candidates) |candidate| {
        if (!pathExistsAbsoluteOrCwd(candidate)) continue;
        std.fs.cwd().copyFile(candidate, std.fs.cwd(), dst, .{}) catch continue;
        log.info("convert", "metal colocated mlx.metallib for converter calibration", .{
            .src = candidate,
            .dst = dst,
        });
        return;
    }

    log.warn("convert", "metal calibration mlx.metallib missing at runtime lookup path", .{
        .runtime_dir = runtime_dir,
        .runtime_path = dst,
    });
}

fn isCudaCalibrationEnabled() bool {
    if (comptime !has_cuda_gpu_calib) return false;
    const raw = std.posix.getenv("BACKEND") orelse return false;
    const token = std.mem.trim(u8, raw, " \t\r\n");
    return std.ascii.eqlIgnoreCase(token, "cuda");
}

fn isCudaGroupedQuantizationEnabled() bool {
    if (!isCudaCalibrationEnabled()) return false;
    return envFlagEnabledDefault("TALU_CONVERT_CUDA_QUANT", true);
}

const CudaCalibContext = struct {
    device: compute.cuda.device.Device,
    blas: compute.cuda.matmul.Blas,
    quant_module: ?compute.cuda.module.Module = null,
    quantize_u4_function: ?compute.cuda.module.Function = null,
    quantize_u8_function: ?compute.cuda.module.Function = null,
    build_dq_weights_function: ?compute.cuda.module.Function = null,
    reduce_mse_function: ?compute.cuda.module.Function = null,
    cuda_quantization_available: bool = false,
    cuda_quantization_failed: bool = false,
    cuda_mse_available: bool = false,
    cuda_dq_weight_build_available: bool = false,
    x_dev: compute.cuda.device.Buffer,
    w_dev: compute.cuda.device.Buffer,
    out_dev: compute.cuda.device.Buffer,
    ref_w_dev: compute.cuda.device.Buffer,
    ref_out_dev: compute.cuda.device.Buffer,
    mse_sum_dev: compute.cuda.device.Buffer,
    // Track cached x_dev upload: block_inputs.values pointer + dimensions.
    // Avoids re-uploading the same input matrix 91+ times per weight tensor.
    cached_x_ptr: ?[*]const f32 = null,
    cached_x_len: usize = 0,
    // Track cached reference output upload used by CUDA-side MSE reduction.
    cached_ref_ptr: ?[*]const f32 = null,
    cached_ref_len: usize = 0,

    fn init() !CudaCalibContext {
        if (comptime !has_cuda_gpu_calib) return error.CudaNotEnabled;
        if (compute.cuda.device.probeRuntime() != .available) return error.CudaNotEnabled;

        var device = try compute.cuda.device.Device.init();
        errdefer device.deinit();

        var blas = try compute.cuda.matmul.Blas.init(&device);
        errdefer blas.deinit(&device);

        // Size calibration buffers from available VRAM so large tensors stay on the
        // GPU scorer path instead of dropping to CPU on the first oversized block.
        const max_buf = cudaCalibrationBufferBudgetBytes(&device);

        var x_dev = try device.allocBuffer(max_buf);
        errdefer x_dev.deinit(&device);
        var w_dev = try device.allocBuffer(max_buf);
        errdefer w_dev.deinit(&device);
        var out_dev = try device.allocBuffer(max_buf);
        errdefer out_dev.deinit(&device);
        var ref_w_dev = try device.allocBuffer(max_buf);
        errdefer ref_w_dev.deinit(&device);
        var ref_out_dev = try device.allocBuffer(max_buf);
        errdefer ref_out_dev.deinit(&device);
        var mse_sum_dev = try device.allocBuffer(@sizeOf(f32));
        errdefer mse_sum_dev.deinit(&device);

        var quant_module: ?compute.cuda.module.Module = null;
        var quantize_u4_function: ?compute.cuda.module.Function = null;
        var quantize_u8_function: ?compute.cuda.module.Function = null;
        var build_dq_weights_function: ?compute.cuda.module.Function = null;
        var reduce_mse_function: ?compute.cuda.module.Function = null;
        var cuda_quantization_available = false;
        var cuda_mse_available = false;
        var cuda_dq_weight_build_available = false;
        if (isCudaGroupedQuantizationEnabled()) {
            if (compute.cuda.module.Module.load(&device, compute.cuda.gaffine_u4_matvec.embedded_module)) |module| {
                var loaded_module = module;
                quantize_u4_function = loaded_module.getFunction(&device, cuda_grouped_quant_u4_symbol) catch null;
                quantize_u8_function = loaded_module.getFunction(&device, cuda_grouped_quant_u8_symbol) catch null;
                build_dq_weights_function = loaded_module.getFunction(&device, cuda_grouped_build_dq_symbol) catch null;
                reduce_mse_function = loaded_module.getFunction(&device, cuda_reduce_mse_symbol) catch null;
                if (quantize_u4_function != null and quantize_u8_function != null) {
                    quant_module = loaded_module;
                    cuda_quantization_available = true;
                    // GPU MSE reduction uses F32 accumulation with non-deterministic
                    // atomicAdd ordering, which can mislead the calibration optimizer
                    // for certain weight distributions (e.g., up_proj in 27B models).
                    // Force CPU F64 MSE for deterministic calibration while keeping
                    // GPU matmul and weight-build for speed.
                    cuda_mse_available = false;
                    cuda_dq_weight_build_available = build_dq_weights_function != null;
                } else {
                    loaded_module.deinit(&device);
                }
            } else |_| {}
        }

        return .{
            .device = device,
            .blas = blas,
            .quant_module = quant_module,
            .quantize_u4_function = quantize_u4_function,
            .quantize_u8_function = quantize_u8_function,
            .build_dq_weights_function = build_dq_weights_function,
            .reduce_mse_function = reduce_mse_function,
            .cuda_quantization_available = cuda_quantization_available,
            .cuda_quantization_failed = false,
            .cuda_mse_available = cuda_mse_available,
            .cuda_dq_weight_build_available = cuda_dq_weight_build_available,
            .x_dev = x_dev,
            .w_dev = w_dev,
            .out_dev = out_dev,
            .ref_w_dev = ref_w_dev,
            .ref_out_dev = ref_out_dev,
            .mse_sum_dev = mse_sum_dev,
            .cached_x_ptr = null,
            .cached_x_len = 0,
            .cached_ref_ptr = null,
            .cached_ref_len = 0,
        };
    }

    fn deinit(self: *CudaCalibContext) void {
        if (self.quant_module) |*module| {
            module.deinit(&self.device);
        }
        self.mse_sum_dev.deinit(&self.device);
        self.ref_out_dev.deinit(&self.device);
        self.ref_w_dev.deinit(&self.device);
        self.out_dev.deinit(&self.device);
        self.w_dev.deinit(&self.device);
        self.x_dev.deinit(&self.device);
        self.blas.deinit(&self.device);
        self.device.deinit();
    }
};

fn cudaCalibrationBufferBudgetBytes(device: *compute.cuda.device.Device) usize {
    const mib: usize = 1024 * 1024;
    if (std.posix.getenv("TALU_CONVERT_CUDA_CALIB_BUF_MIB")) |raw| {
        if (std.fmt.parseInt(usize, raw, 10)) |buffer_mib| {
            if (buffer_mib > 0) {
                const requested = std.math.mul(usize, buffer_mib, mib) catch cuda_calibration_max_buffer_bytes;
                return std.math.clamp(requested, cuda_calibration_min_buffer_bytes, cuda_calibration_max_buffer_bytes);
            }
        } else |_| {}
    }

    if (device.memoryInfo()) |info| {
        const adaptive = info.free / 20;
        return std.math.clamp(adaptive, cuda_calibration_min_buffer_bytes, cuda_calibration_max_buffer_bytes);
    } else |_| {
        const fallback = std.math.mul(usize, cuda_calibration_default_buffer_mib, mib) catch cuda_calibration_min_buffer_bytes;
        return std.math.clamp(fallback, cuda_calibration_min_buffer_bytes, cuda_calibration_max_buffer_bytes);
    }
}

fn cudaGroupedQuantTileBudgetBytes(device: *compute.cuda.device.Device) usize {
    const mib: usize = 1024 * 1024;
    if (std.posix.getenv("TALU_CONVERT_CUDA_QUANT_TILE_MIB")) |raw| {
        if (std.fmt.parseInt(usize, raw, 10)) |tile_mib| {
            if (tile_mib > 0) {
                const requested = std.math.mul(usize, tile_mib, mib) catch cuda_grouped_quant_max_tile_bytes;
                return std.math.clamp(requested, cuda_grouped_quant_min_tile_bytes, cuda_grouped_quant_max_tile_bytes);
            }
        } else |_| {}
    }

    if (device.memoryInfo()) |info| {
        const adaptive = info.free / 2;
        return std.math.clamp(adaptive, cuda_grouped_quant_min_tile_bytes, cuda_grouped_quant_max_tile_bytes);
    } else |_| {
        const fallback = std.math.mul(usize, cuda_grouped_quant_default_tile_mib, mib) catch cuda_grouped_quant_min_tile_bytes;
        return std.math.clamp(fallback, cuda_grouped_quant_min_tile_bytes, cuda_grouped_quant_max_tile_bytes);
    }
}

fn tryQuantizeGroupedAffineRowsCuda(
    allocator: std.mem.Allocator,
    cuda_ctx: *CudaCalibContext,
    source_values: []align(1) const f32,
    row_count: usize,
    col_count: usize,
    group_len: usize,
    quant_bits: u8,
    group_scale_factors: []const f32,
    group_bias_shifts: []const f32,
    group_round_shifts: []const f32,
    packed_row_words: []u32,
    scale_values: []u16,
    bias_values: []u16,
) !bool {
    if (!isCudaGroupedQuantizationEnabled()) return false;
    if (!cuda_ctx.cuda_quantization_available or cuda_ctx.cuda_quantization_failed) return false;
    if (row_count == 0 or col_count == 0 or group_len == 0) return false;
    if (quant_bits != 4 and quant_bits != 8) return false;
    if (col_count % group_len != 0) return false;

    const values_per_u32: usize = if (quant_bits == 4) 8 else 4;
    if (col_count % values_per_u32 != 0) return false;

    if (std.math.cast(u32, row_count) == null) return false;
    const col_count_u32 = std.math.cast(u32, col_count) orelse return false;
    const group_len_u32 = std.math.cast(u32, group_len) orelse return false;
    const packed_col_count = col_count / values_per_u32;
    const packed_col_count_u32 = std.math.cast(u32, packed_col_count) orelse return false;
    const group_count = col_count / group_len;
    const group_count_u32 = std.math.cast(u32, group_count) orelse return false;

    const total_packed_words = std.math.mul(usize, row_count, packed_col_count) catch return false;
    const total_group_values = std.math.mul(usize, row_count, group_count) catch return false;
    if (packed_row_words.len < total_packed_words) return false;
    if (scale_values.len < total_group_values or bias_values.len < total_group_values) return false;

    const quant_function = if (quant_bits == 4)
        cuda_ctx.quantize_u4_function orelse return false
    else
        cuda_ctx.quantize_u8_function orelse return false;

    const group_scale_factors_full = try allocator.alloc(f32, group_count);
    defer allocator.free(group_scale_factors_full);
    @memset(group_scale_factors_full, 1.0);
    const sf_copy_len = @min(group_count, group_scale_factors.len);
    @memcpy(group_scale_factors_full[0..sf_copy_len], group_scale_factors[0..sf_copy_len]);

    const group_bias_shifts_full = try allocator.alloc(f32, group_count);
    defer allocator.free(group_bias_shifts_full);
    @memset(group_bias_shifts_full, 0.0);
    const bs_copy_len = @min(group_count, group_bias_shifts.len);
    @memcpy(group_bias_shifts_full[0..bs_copy_len], group_bias_shifts[0..bs_copy_len]);

    const group_round_shifts_full = try allocator.alloc(f32, group_count);
    defer allocator.free(group_round_shifts_full);
    @memset(group_round_shifts_full, 0.0);
    const rs_copy_len = @min(group_count, group_round_shifts.len);
    @memcpy(group_round_shifts_full[0..rs_copy_len], group_round_shifts[0..rs_copy_len]);

    const input_row_bytes = std.math.mul(usize, col_count, @sizeOf(f32)) catch return false;
    const packed_row_bytes = std.math.mul(usize, packed_col_count, @sizeOf(u32)) catch return false;
    const scale_row_bytes = std.math.mul(usize, group_count, @sizeOf(u16)) catch return false;
    const bias_row_bytes = std.math.mul(usize, group_count, @sizeOf(u16)) catch return false;
    const row_bytes_a = std.math.add(usize, input_row_bytes, packed_row_bytes) catch return false;
    const row_bytes_b = std.math.add(usize, scale_row_bytes, bias_row_bytes) catch return false;
    const row_total_bytes = std.math.add(usize, row_bytes_a, row_bytes_b) catch return false;
    if (row_total_bytes == 0) return false;

    const group_factor_bytes = std.math.mul(usize, group_count, @sizeOf(f32)) catch return false;
    const fixed_bytes = std.math.mul(usize, group_factor_bytes, 3) catch return false;
    var tile_budget_bytes = cudaGroupedQuantTileBudgetBytes(&cuda_ctx.device);
    if (tile_budget_bytes <= fixed_bytes) {
        tile_budget_bytes = (std.math.add(usize, fixed_bytes, row_total_bytes) catch return false);
    }
    const row_budget_bytes = tile_budget_bytes - fixed_bytes;
    var rows_per_tile = row_budget_bytes / row_total_bytes;
    if (rows_per_tile == 0) rows_per_tile = 1;
    rows_per_tile = @min(rows_per_tile, row_count);
    rows_per_tile = @min(rows_per_tile, cuda_grouped_quant_max_grid_y_rows);
    rows_per_tile = @min(rows_per_tile, @as(usize, std.math.maxInt(u32)));

    const max_tile_input_bytes = std.math.mul(usize, rows_per_tile, input_row_bytes) catch return false;
    const max_tile_packed_bytes = std.math.mul(usize, rows_per_tile, packed_row_bytes) catch return false;
    const max_tile_scale_bytes = std.math.mul(usize, rows_per_tile, scale_row_bytes) catch return false;
    const max_tile_bias_bytes = std.math.mul(usize, rows_per_tile, bias_row_bytes) catch return false;

    var group_scale_factors_dev = try cuda_ctx.device.allocBuffer(group_factor_bytes);
    defer group_scale_factors_dev.deinit(&cuda_ctx.device);
    var group_bias_shifts_dev = try cuda_ctx.device.allocBuffer(group_factor_bytes);
    defer group_bias_shifts_dev.deinit(&cuda_ctx.device);
    var group_round_shifts_dev = try cuda_ctx.device.allocBuffer(group_factor_bytes);
    defer group_round_shifts_dev.deinit(&cuda_ctx.device);
    try group_scale_factors_dev.upload(&cuda_ctx.device, std.mem.sliceAsBytes(group_scale_factors_full));
    try group_bias_shifts_dev.upload(&cuda_ctx.device, std.mem.sliceAsBytes(group_bias_shifts_full));
    try group_round_shifts_dev.upload(&cuda_ctx.device, std.mem.sliceAsBytes(group_round_shifts_full));

    var input_dev = try cuda_ctx.device.allocBuffer(max_tile_input_bytes);
    defer input_dev.deinit(&cuda_ctx.device);
    var packed_dev = try cuda_ctx.device.allocBuffer(max_tile_packed_bytes);
    defer packed_dev.deinit(&cuda_ctx.device);
    var scales_dev = try cuda_ctx.device.allocBuffer(max_tile_scale_bytes);
    defer scales_dev.deinit(&cuda_ctx.device);
    var biases_dev = try cuda_ctx.device.allocBuffer(max_tile_bias_bytes);
    defer biases_dev.deinit(&cuda_ctx.device);

    var arg_pack = compute.cuda.args.ArgPack.init(allocator);
    defer arg_pack.deinit();

    var row_start: usize = 0;
    while (row_start < row_count) {
        const tile_rows = @min(rows_per_tile, row_count - row_start);
        const tile_rows_u32: u32 = @intCast(tile_rows);

        const input_offset = std.math.mul(usize, row_start, col_count) catch return false;
        const input_len = std.math.mul(usize, tile_rows, col_count) catch return false;
        const input_slice = source_values[input_offset .. input_offset + input_len];
        try input_dev.upload(&cuda_ctx.device, std.mem.sliceAsBytes(input_slice));

        arg_pack.reset();
        try arg_pack.appendBufferPtr(&input_dev);
        try arg_pack.appendBufferPtr(&group_scale_factors_dev);
        try arg_pack.appendBufferPtr(&group_bias_shifts_dev);
        try arg_pack.appendBufferPtr(&group_round_shifts_dev);
        try arg_pack.appendBufferPtr(&packed_dev);
        try arg_pack.appendBufferPtr(&scales_dev);
        try arg_pack.appendBufferPtr(&biases_dev);
        try arg_pack.appendScalar(u32, tile_rows_u32);
        try arg_pack.appendScalar(u32, col_count_u32);
        try arg_pack.appendScalar(u32, group_len_u32);
        try arg_pack.appendScalar(u32, packed_col_count_u32);

        try compute.cuda.launch.launchWithFamily(
            &cuda_ctx.device,
            quant_function,
            .{
                .grid_x = group_count_u32,
                .grid_y = tile_rows_u32,
                .block_x = cuda_grouped_quant_block_x,
            },
            &arg_pack,
            .pointwise,
        );
        try cuda_ctx.device.synchronize();

        const packed_offset = std.math.mul(usize, row_start, packed_col_count) catch return false;
        const packed_len = std.math.mul(usize, tile_rows, packed_col_count) catch return false;
        const packed_slice = packed_row_words[packed_offset .. packed_offset + packed_len];
        try packed_dev.download(&cuda_ctx.device, std.mem.sliceAsBytes(packed_slice));

        const scale_offset = std.math.mul(usize, row_start, group_count) catch return false;
        const scale_len = std.math.mul(usize, tile_rows, group_count) catch return false;
        const scales_slice = scale_values[scale_offset .. scale_offset + scale_len];
        const biases_slice = bias_values[scale_offset .. scale_offset + scale_len];
        try scales_dev.download(&cuda_ctx.device, std.mem.sliceAsBytes(scales_slice));
        try biases_dev.download(&cuda_ctx.device, std.mem.sliceAsBytes(biases_slice));

        row_start += tile_rows;
    }

    return true;
}

fn resetMseAccumulatorCuda(cuda_ctx: *CudaCalibContext) !bool {
    if (!cuda_ctx.cuda_mse_available) return false;
    var zero = [_]f32{0.0};
    try cuda_ctx.mse_sum_dev.upload(&cuda_ctx.device, std.mem.sliceAsBytes(zero[0..]));
    return true;
}

fn accumulateMseCuda(
    allocator: std.mem.Allocator,
    cuda_ctx: *CudaCalibContext,
    ref_dev: *const compute.cuda.device.Buffer,
    dq_dev: *const compute.cuda.device.Buffer,
    element_count: usize,
) !bool {
    if (!cuda_ctx.cuda_mse_available) return false;
    const mse_fn = cuda_ctx.reduce_mse_function orelse return false;
    if (element_count == 0) return false;
    const n_u32 = std.math.cast(u32, element_count) orelse return false;

    var arg_pack = compute.cuda.args.ArgPack.init(allocator);
    defer arg_pack.deinit();
    try arg_pack.appendBufferPtr(ref_dev);
    try arg_pack.appendBufferPtr(dq_dev);
    try arg_pack.appendBufferPtr(&cuda_ctx.mse_sum_dev);
    try arg_pack.appendScalar(u32, n_u32);

    const blocks_unclamped = (n_u32 + (cuda_reduce_mse_block_x - 1)) / cuda_reduce_mse_block_x;
    const blocks = @max(@as(u32, 1), @min(blocks_unclamped, cuda_reduce_mse_max_blocks));
    try compute.cuda.launch.launchWithFamily(
        &cuda_ctx.device,
        mse_fn,
        .{
            .grid_x = blocks,
            .block_x = cuda_reduce_mse_block_x,
        },
        &arg_pack,
        .pointwise,
    );
    return true;
}

fn finishMseCuda(cuda_ctx: *CudaCalibContext, element_count: usize) !?f64 {
    if (!cuda_ctx.cuda_mse_available) return null;
    if (element_count == 0) return null;
    try cuda_ctx.device.synchronize();

    var sum_host = [_]f32{0.0};
    try cuda_ctx.mse_sum_dev.download(&cuda_ctx.device, std.mem.sliceAsBytes(sum_host[0..]));
    return @as(f64, @floatCast(sum_host[0])) / @as(f64, @floatFromInt(element_count));
}

fn computeMseCuda(
    allocator: std.mem.Allocator,
    cuda_ctx: *CudaCalibContext,
    ref_dev: *const compute.cuda.device.Buffer,
    dq_dev: *const compute.cuda.device.Buffer,
    element_count: usize,
) !?f64 {
    if (!(try resetMseAccumulatorCuda(cuda_ctx))) return null;
    if (!(try accumulateMseCuda(allocator, cuda_ctx, ref_dev, dq_dev, element_count))) return null;
    return try finishMseCuda(cuda_ctx, element_count);
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
    group_scale_factors: []const f32,
    group_bias_shifts: []const f32,
    group_round_shifts: []const f32,
};

const GroupedCalibrationSummary = struct {
    scale_factor: f32,
    bias_shift: f32,
    round_shift: f32,
    best_mse: f64,
    baseline_mse: f64,
    first_mse: f64,
    best_step: usize,
    steps: usize,
    best_iter: usize,
};

const GroupedCalibrationParams = struct {
    summary: GroupedCalibrationSummary,
    group_scale_factors: []f32,
    group_bias_shifts: []f32,
    group_round_shifts: []f32,

    fn deinit(self: *GroupedCalibrationParams, allocator: std.mem.Allocator) void {
        allocator.free(self.group_scale_factors);
        allocator.free(self.group_bias_shifts);
        allocator.free(self.group_round_shifts);
        self.* = undefined;
    }
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
            const base_scale: f32 = if (max_val > min_val) (max_val - min_val) / max_quant_value else 0;
            const group_scale_factor = if (group_idx < ctx.group_scale_factors.len)
                ctx.group_scale_factors[group_idx]
            else
                1.0;
            const group_bias_shift = if (group_idx < ctx.group_bias_shifts.len)
                ctx.group_bias_shifts[group_idx]
            else
                0.0;
            const group_round_shift = if (group_idx < ctx.group_round_shifts.len)
                ctx.group_round_shifts[group_idx]
            else
                0.0;
            const group_scale = base_scale * group_scale_factor;
            const group_bias = groupedBiasFromShift(min_val, group_scale, group_bias_shift);

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
                        const normalized = (value - group_bias) / group_scale + group_round_shift;
                        quantized = @intFromFloat(@max(0, @min(max_quant_value, @round(normalized))));
                    }
                    packed_word |= quantized << @intCast(value_idx * quant_bits);
                }

                row_packed_words[(group_start / values_per_u32) + pack_word_idx] = packed_word;
            }
        }
    }
}

const GroupedCalibrationEvalBudget = struct {
    row_samples: usize,
    input_samples: usize,
};

const GroupedBlockInputMatrix = struct {
    values: []f32,
    input_samples: usize,
    cols: usize,

    fn deinit(self: GroupedBlockInputMatrix, allocator: std.mem.Allocator) void {
        allocator.free(self.values);
    }

    inline fn at(self: *const GroupedBlockInputMatrix, sample_idx: usize, col_idx: usize) f32 {
        return self.values[sample_idx * self.cols + col_idx];
    }
};

const GroupedReplayTargetMatrix = struct {
    values: []f32,
    sample_count: usize,
    cols: usize,

    fn deinit(self: GroupedReplayTargetMatrix, allocator: std.mem.Allocator) void {
        allocator.free(self.values);
    }

    inline fn at(self: *const GroupedReplayTargetMatrix, sample_idx: usize, col_idx: usize) f32 {
        return self.values[sample_idx * self.cols + col_idx];
    }
};

const ReplayPointPair = struct {
    input_point: xray.TracePoint,
    output_point: xray.TracePoint,
};

const ReplayPointCandidates = struct {
    pairs: [3]ReplayPointPair,
    count: usize,
};

const GroupedBlockInputCacheKey = struct {
    layer: u32,
    role: calibration_capture.ActivationRole,
    cols: usize,
    input_samples: usize,
};

const GroupedBlockInputCache = struct {
    allocator: std.mem.Allocator,
    source_tensors: ?*safetensors.UnifiedSafeTensors,
    token_pool: ?[]const u32,
    seed: u64,
    require_embedding_lookup: bool,
    activation_capture: ?*const calibration_capture.LayerActivationCache,
    map: std.AutoHashMap(GroupedBlockInputCacheKey, GroupedBlockInputMatrix),

    fn init(
        allocator: std.mem.Allocator,
        source_tensors: ?*safetensors.UnifiedSafeTensors,
        token_pool: ?[]const u32,
        seed: u64,
        require_embedding_lookup: bool,
        activation_capture: ?*const calibration_capture.LayerActivationCache,
    ) GroupedBlockInputCache {
        return .{
            .allocator = allocator,
            .source_tensors = source_tensors,
            .token_pool = token_pool,
            .seed = seed,
            .require_embedding_lookup = require_embedding_lookup,
            .activation_capture = activation_capture,
            .map = std.AutoHashMap(GroupedBlockInputCacheKey, GroupedBlockInputMatrix).init(allocator),
        };
    }

    fn deinit(self: *GroupedBlockInputCache) void {
        var it = self.map.valueIterator();
        while (it.next()) |entry| {
            entry.deinit(self.allocator);
        }
        self.map.deinit();
    }

    fn get(
        self: *GroupedBlockInputCache,
        layer_index: ?u32,
        role: calibration_capture.ActivationRole,
        cols: usize,
        input_samples: usize,
    ) !*const GroupedBlockInputMatrix {
        const key: GroupedBlockInputCacheKey = .{
            .layer = layer_index orelse std.math.maxInt(u32),
            .role = role,
            .cols = cols,
            .input_samples = input_samples,
        };
        if (self.map.getPtr(key)) |existing| return existing;
        if (self.activation_capture) |capture| {
            if (layer_index) |layer| {
                if (try calibration_capture.sampleLayerActivationsForRole(
                    self.allocator,
                    capture,
                    layer,
                    cols,
                    input_samples,
                    self.seed,
                    role,
                )) |sampled| {
                    const built = GroupedBlockInputMatrix{
                        .values = sampled.values,
                        .input_samples = sampled.sample_count,
                        .cols = sampled.cols,
                    };
                    try self.map.put(key, built);
                    return self.map.getPtr(key).?;
                }
            }
        }
        const built = try buildGroupedBlockInputMatrix(
            self.allocator,
            self.source_tensors,
            cols,
            input_samples,
            self.token_pool,
            self.seed,
            self.require_embedding_lookup,
        );
        try self.map.put(key, built);
        return self.map.getPtr(key).?;
    }
};

const GroupedEmbeddingInputLookup = struct {
    tensor: Tensor,
    input_dim: usize,
    vocab_dim: usize,
    transposed: bool,

    inline fn value(self: GroupedEmbeddingInputLookup, token: u32, feature_col: usize) ?f32 {
        if (self.input_dim == 0 or self.vocab_dim == 0) return null;
        const token_idx = @as(usize, token) % self.vocab_dim;
        const feature_idx = feature_col % self.input_dim;
        const flat_idx = if (!self.transposed)
            token_idx * self.input_dim + feature_idx
        else
            feature_idx * self.vocab_dim + token_idx;
        return groupedTensorScalarToF32(self.tensor, flat_idx);
    }
};

fn groupedTensorScalarToF32(t: Tensor, idx: usize) ?f32 {
    return switch (t.dtype) {
        .f32 => blk: {
            const src = t.asSliceUnaligned(f32);
            if (idx >= src.len) break :blk null;
            break :blk src[idx];
        },
        .f16 => blk: {
            const src = t.asSliceUnaligned(u16);
            if (idx >= src.len) break :blk null;
            break :blk dtype_mod.fp16ToF32(src[idx]);
        },
        .bf16 => blk: {
            const src = t.asSliceUnaligned(u16);
            if (idx >= src.len) break :blk null;
            break :blk dtype_mod.bf16ToF32(src[idx]);
        },
        .f8_e4m3 => blk: {
            const src = t.asSliceUnaligned(u8);
            if (idx >= src.len) break :blk null;
            break :blk dtype_mod.fp8e4m3ToF32(src[idx]);
        },
        else => null,
    };
}

fn findGroupedEmbeddingInputLookup(source_tensors: ?*safetensors.UnifiedSafeTensors) ?GroupedEmbeddingInputLookup {
    const src = source_tensors orelse return null;
    const candidates = [_][]const u8{
        "model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
        "embed_tokens.weight",
        "transformer.wte.weight",
        "backbone.embedding.weight",
        "language_model.model.embed_tokens.weight",
    };

    for (candidates) |name| {
        const tensor_value = src.getTensor(name, null) catch continue;
        if (tensor_value.n_dims != 2) continue;
        const dim0 = @as(usize, @intCast(tensor_value.shape[0]));
        const dim1 = @as(usize, @intCast(tensor_value.shape[1]));
        if (dim0 == 0 or dim1 == 0) continue;
        switch (tensor_value.dtype) {
            .f32, .f16, .bf16, .f8_e4m3 => {},
            else => continue,
        }
        if (dim0 >= dim1) {
            return .{
                .tensor = tensor_value,
                .input_dim = dim1,
                .vocab_dim = dim0,
                .transposed = false,
            };
        }
        return .{
            .tensor = tensor_value,
            .input_dim = dim0,
            .vocab_dim = dim1,
            .transposed = true,
        };
    }
    return null;
}

inline fn tokenFromPool(token_pool: ?[]const u32, sample_idx: usize, seed: u64) u32 {
    if (token_pool) |pool| {
        if (pool.len > 0) {
            const idx = deterministicPoolIndex(seed, sample_idx, pool.len);
            return pool[idx];
        }
    }
    return @intCast(groupedMix64(seed ^ (@as(u64, @intCast(sample_idx + 1)) *% 0xbf58476d1ce4e5b9)) & 0xffff);
}

noinline fn deterministicPoolIndex(seed: u64, sample_idx: usize, len: usize) usize {
    const base = @as(usize, @intCast(seed % len));
    return (base + sample_idx) % len;
}

inline fn tokenFallbackActivation(token: u32, sample_idx: usize, col_idx: usize, seed: u64) f32 {
    const mixed = groupedMix64(seed ^
        (@as(u64, token) *% 0x9e3779b97f4a7c15) ^
        (@as(u64, @intCast(sample_idx + 1)) *% 0xbf58476d1ce4e5b9) ^
        (@as(u64, @intCast(col_idx + 1)) *% 0x94d049bb133111eb));
    const mantissa: u32 = @intCast(mixed & 0x007fffff);
    const raw: u32 = 0x3f000000 | mantissa;
    const unit: f32 = @bitCast(raw);
    return (unit - 0.75) * 2.0;
}

noinline fn buildGroupedBlockInputMatrix(
    allocator: std.mem.Allocator,
    source_tensors: ?*safetensors.UnifiedSafeTensors,
    cols: usize,
    input_samples: usize,
    token_pool: ?[]const u32,
    seed: u64,
    require_embedding_lookup: bool,
) !GroupedBlockInputMatrix {
    if (input_samples == 0) return error.CalibrationDataUnavailable;
    const total = cols * input_samples;
    const values = try allocator.alloc(f32, total);
    errdefer allocator.free(values);

    const lookup = findGroupedEmbeddingInputLookup(source_tensors);
    if (lookup == null and require_embedding_lookup) return error.CalibrationDataUnavailable;

    const FillCtx = struct {
        values: []f32,
        cols: usize,
        input_samples: usize,
        token_pool: ?[]const u32,
        seed: u64,
        lookup: ?GroupedEmbeddingInputLookup,
    };
    var fill_ctx = FillCtx{
        .values = values,
        .cols = cols,
        .input_samples = input_samples,
        .token_pool = token_pool,
        .seed = seed,
        .lookup = lookup,
    };
    const FillFn = struct {
        fn run(start: usize, end: usize, ctx: *FillCtx) void {
            for (start..end) |p| {
                if (p >= ctx.input_samples) break;
                const token = tokenFromPool(ctx.token_pool, p, ctx.seed);
                for (0..ctx.cols) |col| {
                    const idx = p * ctx.cols + col;
                    if (ctx.lookup) |emb| {
                        if (emb.value(token, col)) |v0| {
                            if (col < emb.input_dim) {
                                ctx.values[idx] = v0;
                            } else {
                                const mixed_col = (col + p) % emb.input_dim;
                                const v1 = emb.value(token, mixed_col) orelse v0;
                                ctx.values[idx] = (v0 * 0.75) + (v1 * 0.25);
                            }
                            continue;
                        }
                    }
                    ctx.values[idx] = tokenFallbackActivation(token, p, col, ctx.seed);
                }
            }
        }
    };
    const pool = parallel.global();
    pool.parallelForCompute(input_samples, FillFn.run, &fill_ctx);

    return .{
        .values = values,
        .input_samples = input_samples,
        .cols = cols,
    };
}

fn groupedCalibrationEvalBudget(rows: usize, cols: usize, options: ConvertOptions) GroupedCalibrationEvalBudget {
    const nsamples = @as(usize, @intCast(@max(options.calib_nsamples, 1)));
    const seqlen = @as(usize, @intCast(@max(options.calib_seqlen, 1)));
    const batch_size = @as(usize, @intCast(@max(options.calib_batch_size, 1)));
    const nblocks = @as(usize, @intCast(@max(options.calib_nblocks, 1)));

    const row_target = @max(@as(usize, 512), nsamples * 2);
    const row_samples = @max(@as(usize, 1), @min(rows, @min(row_target, @as(usize, 2048))));

    const requested_inputs_u64 = @as(u64, @intCast(seqlen)) * @as(u64, @intCast(batch_size)) * @as(u64, @intCast(nblocks));
    const requested_inputs = @max(@as(usize, 1), std.math.cast(usize, requested_inputs_u64) orelse std.math.maxInt(usize));
    const max_inputs_by_mem = @max(@as(usize, 1), @divTrunc(calibration_eval_max_bytes, @sizeOf(f32) * @max(cols, 1)));
    // Keep nblocks semantics active, but avoid over-constraining nblocks=1 runs.
    // A 32-token ceiling proved too small for stable quality tuning.
    const max_inputs_by_nblocks = @min(max_calibration_input_samples, 2048 * nblocks);
    const input_samples = @max(@as(usize, 1), @min(@min(requested_inputs, max_inputs_by_mem), max_inputs_by_nblocks));

    return .{
        .row_samples = row_samples,
        .input_samples = input_samples,
    };
}

inline fn groupedMix64(value: u64) u64 {
    var z = value +% 0x9e3779b97f4a7c15;
    z = (z ^ (z >> 30)) *% 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) *% 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

inline fn groupedBiasFromShift(min_val: f32, group_scale: f32, bias_shift: f32) f32 {
    if (group_scale <= 0) return min_val;
    return min_val + bias_shift * group_scale;
}

const GroupedBlockForwardEvalContext = struct {
    source_values: []align(1) const f32,
    col_count: usize,
    group_len: usize,
    max_quant_value: f32,
    row_cache: *const GroupedCalibrationRowCache,
    block_inputs: *const GroupedBlockInputMatrix,
    replay_targets: ?*const GroupedReplayTargetMatrix,
    input_samples: usize,
    group_scale_factors: []const f32,
    group_bias_shifts: []const f32,
    group_round_shifts: []const f32,
    err_sum: f64 = 0.0,
    count: usize = 0,
    mutex: std.Thread.Mutex = .{},
};

const GroupedCalibrationRowCache = struct {
    row_indices: []usize,
    group_min: []f32,
    group_base_scale: []f32,
    sample_rows: usize,
    group_count: usize,

    fn deinit(self: *GroupedCalibrationRowCache, allocator: std.mem.Allocator) void {
        allocator.free(self.row_indices);
        allocator.free(self.group_min);
        allocator.free(self.group_base_scale);
        self.* = undefined;
    }
};

fn buildGroupedCalibrationRowCache(
    allocator: std.mem.Allocator,
    source_values: []align(1) const f32,
    row_count: usize,
    col_count: usize,
    group_len: usize,
    quant_bits: u8,
    sample_rows_budget: usize,
    row_offset: usize,
    row_stride: usize,
) !GroupedCalibrationRowCache {
    const max_quant_value: f32 = if (quant_bits == 4) 15.0 else 255.0;
    const group_count = col_count / group_len;
    const row_indices = try allocator.alloc(usize, sample_rows_budget);
    errdefer allocator.free(row_indices);
    const group_min = try allocator.alloc(f32, sample_rows_budget * group_count);
    errdefer allocator.free(group_min);
    const group_base_scale = try allocator.alloc(f32, sample_rows_budget * group_count);
    errdefer allocator.free(group_base_scale);

    for (0..sample_rows_budget) |sampled_rows| {
        const row = (row_offset + sampled_rows * row_stride) % row_count;
        row_indices[sampled_rows] = row;
        const row_values = source_values[row * col_count .. (row + 1) * col_count];
        for (0..group_count) |g| {
            const group_start = g * group_len;
            const group_values = row_values[group_start .. group_start + group_len];
            var min_val: f32 = group_values[0];
            var max_val: f32 = group_values[0];
            for (group_values) |value| {
                if (value < min_val) min_val = value;
                if (value > max_val) max_val = value;
            }
            const idx = sampled_rows * group_count + g;
            group_min[idx] = min_val;
            group_base_scale[idx] = if (max_val > min_val) (max_val - min_val) / max_quant_value else 0.0;
        }
    }

    return .{
        .row_indices = row_indices,
        .group_min = group_min,
        .group_base_scale = group_base_scale,
        .sample_rows = sample_rows_budget,
        .group_count = group_count,
    };
}

fn buildGroupedBlockForwardReferenceOutputsMetal(
    allocator: std.mem.Allocator,
    source_values: []align(1) const f32,
    col_count: usize,
    row_cache: *const GroupedCalibrationRowCache,
    block_inputs: *const GroupedBlockInputMatrix,
) ?[]f32 {
    if (!compute.metal.isAvailable()) return null;
    if (row_cache.sample_rows == 0 or col_count == 0) return null;
    if (block_inputs.cols != col_count or block_inputs.input_samples == 0) return null;

    const graph = compute.metal.graph;
    const input_samples = block_inputs.input_samples;
    const sample_rows_budget = row_cache.sample_rows;
    const ref_weights = allocator.alloc(f32, col_count * sample_rows_budget) catch return null;
    defer allocator.free(ref_weights);

    var sampled_rows: usize = 0;
    while (sampled_rows < sample_rows_budget) : (sampled_rows += 1) {
        const row = row_cache.row_indices[sampled_rows];
        const row_values = source_values[row * col_count .. (row + 1) * col_count];
        for (0..col_count) |col| {
            const idx = col * sample_rows_budget + sampled_rows;
            ref_weights[idx] = row_values[col];
        }
    }

    const x_shape = [_]i64{ @intCast(input_samples), @intCast(col_count) };
    const w_shape = [_]i64{ @intCast(col_count), @intCast(sample_rows_budget) };
    const out_len = input_samples * sample_rows_budget;
    const ref_host = allocator.alloc(f32, out_len) catch return null;
    errdefer allocator.free(ref_host);

    graph.beginForwardGraphBuild();
    const x_handle = graph.createArrayF32(block_inputs.values, &x_shape);
    if (x_handle == null) {
        allocator.free(ref_host);
        return null;
    }
    defer graph.freeArray(x_handle);
    const ref_w_handle = graph.createArrayF32(ref_weights, &w_shape);
    if (ref_w_handle == null) {
        allocator.free(ref_host);
        return null;
    }
    defer graph.freeArray(ref_w_handle);
    const ref_out_handle = graph.mlx_lazy_matmul(x_handle, ref_w_handle);
    if (ref_out_handle == null) {
        allocator.free(ref_host);
        return null;
    }
    defer graph.freeArray(ref_out_handle);

    var eval_handles = [_]compute.metal.graph.ArrayHandle{ref_out_handle};
    graph.eval(&eval_handles);
    graph.copyToHost(ref_out_handle, ref_host);
    return ref_host;
}

fn buildGroupedBlockForwardReferenceOutputsCuda(
    allocator: std.mem.Allocator,
    cuda_ctx: *CudaCalibContext,
    source_values: []align(1) const f32,
    col_count: usize,
    row_cache: *const GroupedCalibrationRowCache,
    block_inputs: *const GroupedBlockInputMatrix,
) ?[]f32 {
    if (row_cache.sample_rows == 0 or col_count == 0) return null;
    if (block_inputs.cols != col_count or block_inputs.input_samples == 0) return null;

    const input_samples = block_inputs.input_samples;
    const sample_rows_budget = row_cache.sample_rows;

    const out_len = input_samples * sample_rows_budget;
    const x_bytes = input_samples * col_count * @sizeOf(f32);
    if (x_bytes > cuda_ctx.x_dev.size) return null;

    const max_tile_rows = cudaCalibrationMaxSampleRowsPerTile(cuda_ctx, input_samples, col_count);
    if (max_tile_rows == 0) return null;
    const tile_row_capacity = @min(sample_rows_budget, max_tile_rows);
    const tile_weight_len = col_count * tile_row_capacity;
    const tile_out_len = input_samples * tile_row_capacity;
    const ref_weights = allocator.alloc(f32, tile_weight_len) catch return null;
    defer allocator.free(ref_weights);
    const out_tile = allocator.alloc(f32, tile_out_len) catch return null;
    defer allocator.free(out_tile);

    const ref_host = allocator.alloc(f32, out_len) catch return null;
    errdefer allocator.free(ref_host);

    // Skip x_dev upload if already cached (same pointer + length from prior call)
    const x_slice = block_inputs.values[0 .. input_samples * col_count];
    if (cuda_ctx.cached_x_ptr != x_slice.ptr or cuda_ctx.cached_x_len != x_slice.len) {
        cuda_ctx.x_dev.upload(&cuda_ctx.device, std.mem.sliceAsBytes(x_slice)) catch return null;
        cuda_ctx.cached_x_ptr = x_slice.ptr;
        cuda_ctx.cached_x_len = x_slice.len;
    }

    var tile_start: usize = 0;
    while (tile_start < sample_rows_budget) : (tile_start += tile_row_capacity) {
        const tile_rows = @min(tile_row_capacity, sample_rows_budget - tile_start);
        const ref_tile = ref_weights[0 .. col_count * tile_rows];
        buildGroupedRefWeightsTile(source_values, col_count, row_cache, tile_start, tile_rows, ref_tile);
        cuda_ctx.w_dev.upload(&cuda_ctx.device, std.mem.sliceAsBytes(ref_tile)) catch return null;
        cuda_ctx.blas.matmulF32(&cuda_ctx.device, &cuda_ctx.x_dev, input_samples, col_count, &cuda_ctx.w_dev, tile_rows, &cuda_ctx.out_dev) catch return null;
        cuda_ctx.device.synchronize() catch return null;

        const out_tile_slice = out_tile[0 .. input_samples * tile_rows];
        cuda_ctx.out_dev.download(&cuda_ctx.device, std.mem.sliceAsBytes(out_tile_slice)) catch return null;
        var p: usize = 0;
        while (p < input_samples) : (p += 1) {
            const src = out_tile_slice[p * tile_rows .. (p + 1) * tile_rows];
            const dst = ref_host[p * sample_rows_budget + tile_start .. p * sample_rows_budget + tile_start + tile_rows];
            @memcpy(dst, src);
        }
    }

    return ref_host;
}

fn evaluateGroupedBlockForwardRows(sample_start: usize, sample_end: usize, ctx: *GroupedBlockForwardEvalContext) void {
    var local_err_sum: f64 = 0.0;
    var local_count: usize = 0;

    var sampled_rows: usize = sample_start;
    while (sampled_rows < sample_end) : (sampled_rows += 1) {
        const row = ctx.row_cache.row_indices[sampled_rows];
        const row_values = ctx.source_values[row * ctx.col_count .. (row + 1) * ctx.col_count];
        const use_replay_targets = ctx.replay_targets != null;
        var ref_out = std.mem.zeroes([max_calibration_input_samples]f64);
        var dq_out = std.mem.zeroes([max_calibration_input_samples]f64);

        const group_count = ctx.row_cache.group_count;
        for (0..group_count) |g| {
            const group_start = g * ctx.group_len;
            const group_values = row_values[group_start .. group_start + ctx.group_len];
            const row_group_idx = sampled_rows * group_count + g;
            const min_val = ctx.row_cache.group_min[row_group_idx];
            const base_scale = ctx.row_cache.group_base_scale[row_group_idx];
            const group_scale_factor = if (g < ctx.group_scale_factors.len)
                ctx.group_scale_factors[g]
            else
                1.0;
            const group_bias_shift = if (g < ctx.group_bias_shifts.len)
                ctx.group_bias_shifts[g]
            else
                0.0;
            const group_round_shift = if (g < ctx.group_round_shifts.len)
                ctx.group_round_shifts[g]
            else
                0.0;
            const group_scale = base_scale * group_scale_factor;
            const group_bias = groupedBiasFromShift(min_val, group_scale, group_bias_shift);

            for (group_values, 0..) |value, value_idx| {
                const col = group_start + value_idx;
                var quantized: f32 = 0.0;
                if (group_scale > 0) {
                    const normalized = (value - group_bias) / group_scale + group_round_shift;
                    quantized = @max(0.0, @min(ctx.max_quant_value, @round(normalized)));
                }
                const dequant = quantized * group_scale + group_bias;
                var p: usize = 0;
                while (p < ctx.input_samples) : (p += 1) {
                    const x = ctx.block_inputs.at(p, col);
                    if (!use_replay_targets) {
                        ref_out[p] += @as(f64, x) * @as(f64, value);
                    }
                    dq_out[p] += @as(f64, x) * @as(f64, dequant);
                }
            }
        }

        var p: usize = 0;
        while (p < ctx.input_samples) : (p += 1) {
            const ref_value = if (ctx.replay_targets) |targets|
                @as(f64, targets.at(p, row))
            else
                ref_out[p];
            const diff = ref_value - dq_out[p];
            local_err_sum += diff * diff;
            local_count += 1;
        }
    }

    if (local_count == 0) return;
    ctx.mutex.lock();
    defer ctx.mutex.unlock();
    ctx.err_sum += local_err_sum;
    ctx.count += local_count;
}

const GroupedCudaDeviceWeightScratch = struct {
    sampled_source_dev: compute.cuda.device.Buffer,
    row_group_min_dev: compute.cuda.device.Buffer,
    row_group_base_scale_dev: compute.cuda.device.Buffer,
    group_scale_factors_dev: compute.cuda.device.Buffer,
    group_bias_shifts_dev: compute.cuda.device.Buffer,
    group_round_shifts_dev: compute.cuda.device.Buffer,

    fn deinit(self: *GroupedCudaDeviceWeightScratch, device: *compute.cuda.device.Device) void {
        self.group_round_shifts_dev.deinit(device);
        self.group_bias_shifts_dev.deinit(device);
        self.group_scale_factors_dev.deinit(device);
        self.row_group_base_scale_dev.deinit(device);
        self.row_group_min_dev.deinit(device);
        self.sampled_source_dev.deinit(device);
    }
};

const GroupedCudaWeightScratch = struct {
    dq_weights: []f32,
    ref_weights: ?[]f32 = null,
    device: ?GroupedCudaDeviceWeightScratch = null,

    fn deinit(
        self: *GroupedCudaWeightScratch,
        allocator: std.mem.Allocator,
        device: ?*compute.cuda.device.Device,
    ) void {
        allocator.free(self.dq_weights);
        if (self.ref_weights) |buf| allocator.free(buf);
        if (device) |dev| {
            if (self.device) |*gpu| gpu.deinit(dev);
        }
    }
};

fn evaluateGroupedBlockForwardCalibrationCandidate(
    allocator: std.mem.Allocator,
    metal_calibration_enabled: bool,
    cuda_ctx: ?*CudaCalibContext,
    source_values: []align(1) const f32,
    col_count: usize,
    group_len: usize,
    quant_bits: u8,
    row_cache: *const GroupedCalibrationRowCache,
    block_inputs: *const GroupedBlockInputMatrix,
    group_scale_factors: []const f32,
    group_bias_shifts: []const f32,
    group_round_shifts: []const f32,
    replay_targets: ?*const GroupedReplayTargetMatrix,
    ref_outputs: ?[]const f32,
    cuda_weight_scratch: ?*GroupedCudaWeightScratch,
) f64 {
    const max_quant_value: f32 = if (quant_bits == 4) 15.0 else 255.0;
    const group_count = row_cache.group_count;
    if (group_count == 0) return std.math.inf(f64);
    if (row_cache.sample_rows == 0) return std.math.inf(f64);
    if (block_inputs.cols != col_count or block_inputs.input_samples == 0) return std.math.inf(f64);

    const input_samples = block_inputs.input_samples;
    if (input_samples > max_calibration_input_samples) return std.math.inf(f64);
    if (replay_targets) |targets| {
        if (targets.sample_count != input_samples) return std.math.inf(f64);
        for (row_cache.row_indices) |row| {
            if (row >= targets.cols) return std.math.inf(f64);
        }
    }

    if (metal_calibration_enabled) {
        if (evaluateGroupedBlockForwardCalibrationCandidateMetal(
            allocator,
            source_values,
            col_count,
            group_len,
            quant_bits,
            row_cache,
            block_inputs,
            group_scale_factors,
            group_bias_shifts,
            group_round_shifts,
            replay_targets,
            ref_outputs,
        )) |gpu_err| {
            return gpu_err;
        }
    }
    if (cuda_ctx) |ctx| {
        if (evaluateGroupedBlockForwardCalibrationCandidateCuda(
            allocator,
            ctx,
            source_values,
            col_count,
            group_len,
            quant_bits,
            row_cache,
            block_inputs,
            group_scale_factors,
            group_bias_shifts,
            group_round_shifts,
            replay_targets,
            ref_outputs,
            cuda_weight_scratch,
        )) |gpu_err| {
            return gpu_err;
        }
    }

    var eval_ctx = GroupedBlockForwardEvalContext{
        .source_values = source_values,
        .col_count = col_count,
        .group_len = group_len,
        .max_quant_value = max_quant_value,
        .row_cache = row_cache,
        .block_inputs = block_inputs,
        .replay_targets = replay_targets,
        .input_samples = input_samples,
        .group_scale_factors = group_scale_factors,
        .group_bias_shifts = group_bias_shifts,
        .group_round_shifts = group_round_shifts,
    };
    const pool = parallel.global();
    pool.parallelForCompute(row_cache.sample_rows, evaluateGroupedBlockForwardRows, &eval_ctx);
    if (eval_ctx.count == 0) return std.math.inf(f64);
    return eval_ctx.err_sum / @as(f64, @floatFromInt(eval_ctx.count));
}

fn evaluateGroupedBlockForwardCalibrationCandidateMetal(
    allocator: std.mem.Allocator,
    source_values: []align(1) const f32,
    col_count: usize,
    group_len: usize,
    quant_bits: u8,
    row_cache: *const GroupedCalibrationRowCache,
    block_inputs: *const GroupedBlockInputMatrix,
    group_scale_factors: []const f32,
    group_bias_shifts: []const f32,
    group_round_shifts: []const f32,
    replay_targets: ?*const GroupedReplayTargetMatrix,
    ref_outputs: ?[]const f32,
) ?f64 {
    if (!compute.metal.isAvailable()) return null;
    if (group_len == 0 or col_count % group_len != 0) return null;
    const graph = compute.metal.graph;
    const max_quant_value: f32 = if (quant_bits == 4) 15.0 else 255.0;
    const group_count = row_cache.group_count;
    const input_samples = block_inputs.input_samples;
    if (group_count == 0 or input_samples == 0) return null;
    const sample_rows_budget = row_cache.sample_rows;

    const use_replay_targets = replay_targets != null;
    const use_cached_ref = !use_replay_targets and ref_outputs != null;
    var ref_weights_opt: ?[]f32 = null;
    if (!use_cached_ref and !use_replay_targets) {
        ref_weights_opt = allocator.alloc(f32, col_count * sample_rows_budget) catch return null;
    }
    defer if (ref_weights_opt) |ref_weights| allocator.free(ref_weights);
    const dq_weights = allocator.alloc(f32, col_count * sample_rows_budget) catch return null;
    defer allocator.free(dq_weights);

    var build_ctx = GroupedDqWeightBuildContext{
        .source_values = source_values,
        .col_count = col_count,
        .group_len = group_len,
        .max_quant_value = max_quant_value,
        .row_cache = row_cache,
        .group_scale_factors = group_scale_factors,
        .group_bias_shifts = group_bias_shifts,
        .group_round_shifts = group_round_shifts,
        .ref_weights = ref_weights_opt,
        .dq_weights = dq_weights,
    };
    const pool = parallel.global();
    pool.parallelForCompute(sample_rows_budget, buildGroupedDqWeightsRows, &build_ctx);

    const x_shape = [_]i64{ @intCast(input_samples), @intCast(col_count) };
    const w_shape = [_]i64{ @intCast(col_count), @intCast(sample_rows_budget) };
    const out_len = input_samples * sample_rows_budget;
    var ref_host_owned: ?[]f32 = null;
    const ref_host: []const f32 = if (use_replay_targets)
        &[_]f32{}
    else if (ref_outputs) |cached|
        cached
    else blk: {
        ref_host_owned = allocator.alloc(f32, out_len) catch return null;
        break :blk ref_host_owned.?;
    };
    defer if (ref_host_owned) |owned| allocator.free(owned);
    const dq_host = allocator.alloc(f32, out_len) catch return null;
    defer allocator.free(dq_host);

    graph.beginForwardGraphBuild();
    const x_handle = graph.createArrayF32(block_inputs.values, &x_shape);
    if (x_handle == null) return null;
    defer graph.freeArray(x_handle);
    const ref_w_handle = if (ref_weights_opt) |ref_weights| graph.createArrayF32(ref_weights, &w_shape) else null;
    if (!use_cached_ref and !use_replay_targets and ref_w_handle == null) return null;
    defer if (!use_cached_ref and !use_replay_targets) graph.freeArray(ref_w_handle);
    const dq_w_handle = graph.createArrayF32(dq_weights, &w_shape);
    if (dq_w_handle == null) return null;
    defer graph.freeArray(dq_w_handle);

    const ref_out_handle = if (!use_cached_ref and !use_replay_targets) graph.mlx_lazy_matmul(x_handle, ref_w_handle) else null;
    if (!use_cached_ref and !use_replay_targets and ref_out_handle == null) return null;
    defer if (!use_cached_ref and !use_replay_targets) graph.freeArray(ref_out_handle);
    const dq_out_handle = graph.mlx_lazy_matmul(x_handle, dq_w_handle);
    if (dq_out_handle == null) return null;
    defer graph.freeArray(dq_out_handle);

    if (!use_cached_ref and !use_replay_targets) {
        var eval_handles = [_]compute.metal.graph.ArrayHandle{ ref_out_handle, dq_out_handle };
        graph.eval(&eval_handles);
        graph.copyToHost(ref_out_handle, ref_host_owned.?);
    } else {
        var eval_handles = [_]compute.metal.graph.ArrayHandle{dq_out_handle};
        graph.eval(&eval_handles);
    }
    graph.copyToHost(dq_out_handle, dq_host);

    var err_sum: f64 = 0.0;
    if (use_replay_targets) {
        const targets = replay_targets.?;
        var sampled_rows: usize = 0;
        while (sampled_rows < sample_rows_budget) : (sampled_rows += 1) {
            const row = row_cache.row_indices[sampled_rows];
            var p: usize = 0;
            while (p < input_samples) : (p += 1) {
                const idx = p * sample_rows_budget + sampled_rows;
                const ref_val = targets.at(p, row);
                const diff = @as(f64, @floatCast(ref_val - dq_host[idx]));
                err_sum += diff * diff;
            }
        }
    } else {
        for (ref_host, dq_host) |a, b| {
            const diff = @as(f64, @floatCast(a - b));
            err_sum += diff * diff;
        }
    }
    if (out_len == 0) return null;
    return err_sum / @as(f64, @floatFromInt(out_len));
}

fn evaluateGroupedBlockForwardCalibrationCandidateCuda(
    allocator: std.mem.Allocator,
    cuda_ctx: *CudaCalibContext,
    source_values: []align(1) const f32,
    col_count: usize,
    group_len: usize,
    quant_bits: u8,
    row_cache: *const GroupedCalibrationRowCache,
    block_inputs: *const GroupedBlockInputMatrix,
    group_scale_factors: []const f32,
    group_bias_shifts: []const f32,
    group_round_shifts: []const f32,
    replay_targets: ?*const GroupedReplayTargetMatrix,
    ref_outputs: ?[]const f32,
    cuda_weight_scratch: ?*GroupedCudaWeightScratch,
) ?f64 {
    if (group_len == 0 or col_count % group_len != 0) return null;
    const max_quant_value: f32 = if (quant_bits == 4) 15.0 else 255.0;
    const group_count = row_cache.group_count;
    const input_samples = block_inputs.input_samples;
    if (group_count == 0 or input_samples == 0) return null;
    const sample_rows_budget = row_cache.sample_rows;

    const use_replay_targets = replay_targets != null;
    const use_cached_ref = !use_replay_targets and ref_outputs != null;
    const need_ref_matmul = !use_replay_targets and ref_outputs == null;
    const out_len = input_samples * sample_rows_budget;
    const x_bytes = input_samples * col_count * @sizeOf(f32);
    const w_bytes = col_count * sample_rows_budget * @sizeOf(f32);
    const out_bytes = out_len * @sizeOf(f32);
    if (x_bytes > cuda_ctx.x_dev.size) return null;

    const max_tile_rows = cudaCalibrationMaxSampleRowsPerTile(cuda_ctx, input_samples, col_count);
    if (max_tile_rows == 0) return null;
    const scratch_tile_rows = if (cuda_weight_scratch) |scratch|
        if (col_count > 0) @divTrunc(scratch.dq_weights.len, col_count) else 0
    else
        max_tile_rows;
    const tile_row_capacity = @min(sample_rows_budget, @min(max_tile_rows, scratch_tile_rows));
    if (tile_row_capacity == 0) return null;
    if (sample_rows_budget > tile_row_capacity or
        w_bytes > cuda_ctx.w_dev.size or
        out_bytes > cuda_ctx.out_dev.size or
        w_bytes > cuda_ctx.ref_w_dev.size or
        out_bytes > cuda_ctx.ref_out_dev.size)
    {
        return evaluateGroupedBlockForwardCalibrationCandidateCudaTiled(
            allocator,
            cuda_ctx,
            source_values,
            col_count,
            group_len,
            max_quant_value,
            row_cache,
            block_inputs,
            group_scale_factors,
            group_bias_shifts,
            group_round_shifts,
            replay_targets,
            ref_outputs,
            cuda_weight_scratch,
        );
    }

    const weight_len = col_count * sample_rows_budget;
    const use_gpu_weight_build = !need_ref_matmul and cuda_ctx.cuda_dq_weight_build_available and blk: {
        if (cuda_weight_scratch) |scratch| {
            break :blk scratch.device != null;
        }
        break :blk false;
    };

    // Build dequantized weights either on GPU (preferred) or on CPU fallback.
    var ref_weights_opt: ?[]f32 = null;
    var ref_weights_owned = false;
    if (!use_gpu_weight_build and !use_cached_ref and !use_replay_targets) {
        if (cuda_weight_scratch) |scratch| {
            if (scratch.ref_weights) |buf| {
                if (buf.len == weight_len) {
                    ref_weights_opt = buf;
                }
            }
        }
        if (ref_weights_opt == null) {
            ref_weights_opt = allocator.alloc(f32, weight_len) catch return null;
            ref_weights_owned = true;
        }
    }
    defer if (ref_weights_owned) allocator.free(ref_weights_opt.?);

    if (use_gpu_weight_build) {
        const gpu_scratch = &cuda_weight_scratch.?.device.?;
        const build_fn = cuda_ctx.build_dq_weights_function orelse return null;
        const col_count_u32 = std.math.cast(u32, col_count) orelse return null;
        const sample_rows_u32 = std.math.cast(u32, sample_rows_budget) orelse return null;
        const group_len_u32 = std.math.cast(u32, group_len) orelse return null;
        const grid_x = @max(@as(u32, 1), (col_count_u32 + (cuda_grouped_build_dq_block_x - 1)) / cuda_grouped_build_dq_block_x);

        gpu_scratch.group_scale_factors_dev.upload(&cuda_ctx.device, std.mem.sliceAsBytes(group_scale_factors)) catch return null;
        gpu_scratch.group_bias_shifts_dev.upload(&cuda_ctx.device, std.mem.sliceAsBytes(group_bias_shifts)) catch return null;
        gpu_scratch.group_round_shifts_dev.upload(&cuda_ctx.device, std.mem.sliceAsBytes(group_round_shifts)) catch return null;

        var arg_pack = compute.cuda.args.ArgPack.init(allocator);
        defer arg_pack.deinit();
        arg_pack.appendBufferPtr(&gpu_scratch.sampled_source_dev) catch return null;
        arg_pack.appendBufferPtr(&gpu_scratch.row_group_min_dev) catch return null;
        arg_pack.appendBufferPtr(&gpu_scratch.row_group_base_scale_dev) catch return null;
        arg_pack.appendBufferPtr(&gpu_scratch.group_scale_factors_dev) catch return null;
        arg_pack.appendBufferPtr(&gpu_scratch.group_bias_shifts_dev) catch return null;
        arg_pack.appendBufferPtr(&gpu_scratch.group_round_shifts_dev) catch return null;
        arg_pack.appendBufferPtr(&cuda_ctx.w_dev) catch return null;
        arg_pack.appendScalar(u32, sample_rows_u32) catch return null;
        arg_pack.appendScalar(u32, col_count_u32) catch return null;
        arg_pack.appendScalar(u32, group_len_u32) catch return null;
        arg_pack.appendScalar(f32, max_quant_value) catch return null;
        compute.cuda.launch.launchWithFamily(
            &cuda_ctx.device,
            build_fn,
            .{
                .grid_x = grid_x,
                .grid_y = sample_rows_u32,
                .block_x = cuda_grouped_build_dq_block_x,
            },
            &arg_pack,
            .pointwise,
        ) catch return null;
    } else {
        var dq_weights_owned = false;
        const dq_weights = blk: {
            if (cuda_weight_scratch) |scratch| {
                if (scratch.dq_weights.len == weight_len) break :blk scratch.dq_weights;
            }
            dq_weights_owned = true;
            break :blk allocator.alloc(f32, weight_len) catch return null;
        };
        defer if (dq_weights_owned) allocator.free(dq_weights);

        var build_ctx = GroupedDqWeightBuildContext{
            .source_values = source_values,
            .col_count = col_count,
            .group_len = group_len,
            .max_quant_value = max_quant_value,
            .row_cache = row_cache,
            .group_scale_factors = group_scale_factors,
            .group_bias_shifts = group_bias_shifts,
            .group_round_shifts = group_round_shifts,
            .ref_weights = ref_weights_opt,
            .dq_weights = dq_weights,
        };
        const pool = parallel.global();
        pool.parallelForCompute(sample_rows_budget, buildGroupedDqWeightsRows, &build_ctx);

        cuda_ctx.w_dev.upload(&cuda_ctx.device, std.mem.sliceAsBytes(dq_weights)) catch return null;
    }

    // Skip x_dev upload if already cached (same pointer + length from prior call)
    const x_slice = block_inputs.values[0 .. input_samples * col_count];
    if (cuda_ctx.cached_x_ptr != x_slice.ptr or cuda_ctx.cached_x_len != x_slice.len) {
        cuda_ctx.x_dev.upload(&cuda_ctx.device, std.mem.sliceAsBytes(x_slice)) catch return null;
        cuda_ctx.cached_x_ptr = x_slice.ptr;
        cuda_ctx.cached_x_len = x_slice.len;
    }

    // Issue all GPU matmuls before synchronizing once
    if (need_ref_matmul) {
        if (ref_weights_opt) |ref_weights| {
            cuda_ctx.ref_w_dev.upload(&cuda_ctx.device, std.mem.sliceAsBytes(ref_weights)) catch return null;
            cuda_ctx.blas.matmulF32(&cuda_ctx.device, &cuda_ctx.x_dev, input_samples, col_count, &cuda_ctx.ref_w_dev, sample_rows_budget, &cuda_ctx.ref_out_dev) catch return null;
        } else return null;
    }
    cuda_ctx.blas.matmulF32(&cuda_ctx.device, &cuda_ctx.x_dev, input_samples, col_count, &cuda_ctx.w_dev, sample_rows_budget, &cuda_ctx.out_dev) catch return null;

    if (!use_replay_targets) {
        if (ref_outputs) |cached_ref| {
            if (cached_ref.len != out_len) return null;
            if (cuda_ctx.cached_ref_ptr != cached_ref.ptr or cuda_ctx.cached_ref_len != cached_ref.len) {
                cuda_ctx.ref_out_dev.upload(&cuda_ctx.device, std.mem.sliceAsBytes(cached_ref)) catch return null;
                cuda_ctx.cached_ref_ptr = cached_ref.ptr;
                cuda_ctx.cached_ref_len = cached_ref.len;
            }
        }
        const ref_dev: *const compute.cuda.device.Buffer = &cuda_ctx.ref_out_dev;
        if ((computeMseCuda(allocator, cuda_ctx, ref_dev, &cuda_ctx.out_dev, out_len) catch null)) |gpu_mse| {
            return gpu_mse;
        }
    }

    // Download results
    cuda_ctx.device.synchronize() catch return null;
    var ref_host_owned: ?[]f32 = null;
    if (need_ref_matmul) {
        ref_host_owned = allocator.alloc(f32, out_len) catch return null;
        cuda_ctx.ref_out_dev.download(&cuda_ctx.device, std.mem.sliceAsBytes(ref_host_owned.?)) catch return null;
    }
    defer if (ref_host_owned) |owned| allocator.free(owned);

    const ref_host: []const f32 = if (use_replay_targets)
        &[_]f32{}
    else if (ref_outputs) |cached|
        cached
    else if (ref_host_owned) |owned|
        owned
    else
        return null;

    const dq_host = allocator.alloc(f32, out_len) catch return null;
    defer allocator.free(dq_host);
    cuda_ctx.out_dev.download(&cuda_ctx.device, std.mem.sliceAsBytes(dq_host)) catch return null;

    // Compute MSE on CPU
    var err_sum: f64 = 0.0;
    if (use_replay_targets) {
        const targets = replay_targets.?;
        var sampled_rows: usize = 0;
        while (sampled_rows < sample_rows_budget) : (sampled_rows += 1) {
            const row = row_cache.row_indices[sampled_rows];
            var p: usize = 0;
            while (p < input_samples) : (p += 1) {
                const idx = p * sample_rows_budget + sampled_rows;
                const ref_val = targets.at(p, row);
                const diff = @as(f64, @floatCast(ref_val - dq_host[idx]));
                err_sum += diff * diff;
            }
        }
    } else {
        for (ref_host, dq_host) |a, b| {
            const diff = @as(f64, @floatCast(a - b));
            err_sum += diff * diff;
        }
    }
    if (out_len == 0) return null;
    return err_sum / @as(f64, @floatFromInt(out_len));
}

const GroupedDqWeightBuildContext = struct {
    source_values: []align(1) const f32,
    col_count: usize,
    group_len: usize,
    max_quant_value: f32,
    row_cache: *const GroupedCalibrationRowCache,
    group_scale_factors: []const f32,
    group_bias_shifts: []const f32,
    group_round_shifts: []const f32,
    ref_weights: ?[]f32,
    dq_weights: []f32,
};

fn buildGroupedDqWeightsRows(sample_start: usize, sample_end: usize, ctx: *GroupedDqWeightBuildContext) void {
    const group_count = ctx.col_count / ctx.group_len;
    var sampled_rows: usize = sample_start;
    while (sampled_rows < sample_end) : (sampled_rows += 1) {
        const row = ctx.row_cache.row_indices[sampled_rows];
        const row_values = ctx.source_values[row * ctx.col_count .. (row + 1) * ctx.col_count];

        for (0..group_count) |g| {
            const group_start = g * ctx.group_len;
            const group_values = row_values[group_start .. group_start + ctx.group_len];
            const row_group_idx = sampled_rows * group_count + g;
            const min_val = ctx.row_cache.group_min[row_group_idx];
            const base_scale = ctx.row_cache.group_base_scale[row_group_idx];
            const group_scale_factor = if (g < ctx.group_scale_factors.len)
                ctx.group_scale_factors[g]
            else
                1.0;
            const group_bias_shift = if (g < ctx.group_bias_shifts.len)
                ctx.group_bias_shifts[g]
            else
                0.0;
            const group_round_shift = if (g < ctx.group_round_shifts.len)
                ctx.group_round_shifts[g]
            else
                0.0;
            const group_scale = base_scale * group_scale_factor;
            const group_bias = groupedBiasFromShift(min_val, group_scale, group_bias_shift);

            for (group_values, 0..) |value, value_idx| {
                const col = group_start + value_idx;
                var quantized: f32 = 0.0;
                if (group_scale > 0) {
                    const normalized = (value - group_bias) / group_scale + group_round_shift;
                    quantized = @max(0.0, @min(ctx.max_quant_value, @round(normalized)));
                }
                const dequant = quantized * group_scale + group_bias;
                const idx = col * ctx.row_cache.sample_rows + sampled_rows;
                if (ctx.ref_weights) |ref_weights| ref_weights[idx] = value;
                ctx.dq_weights[idx] = dequant;
            }
        }
    }
}

fn buildGroupedRefWeightsTile(
    source_values: []align(1) const f32,
    col_count: usize,
    row_cache: *const GroupedCalibrationRowCache,
    tile_start: usize,
    tile_rows: usize,
    ref_weights: []f32,
) void {
    var local_row: usize = 0;
    while (local_row < tile_rows) : (local_row += 1) {
        const sampled_row = tile_start + local_row;
        const row = row_cache.row_indices[sampled_row];
        const row_values = source_values[row * col_count .. (row + 1) * col_count];
        for (0..col_count) |col| {
            ref_weights[col * tile_rows + local_row] = row_values[col];
        }
    }
}

fn buildGroupedDqWeightsTile(
    source_values: []align(1) const f32,
    col_count: usize,
    group_len: usize,
    max_quant_value: f32,
    row_cache: *const GroupedCalibrationRowCache,
    group_scale_factors: []const f32,
    group_bias_shifts: []const f32,
    group_round_shifts: []const f32,
    tile_start: usize,
    tile_rows: usize,
    ref_weights: ?[]f32,
    dq_weights: []f32,
) void {
    const group_count = row_cache.group_count;
    var local_row: usize = 0;
    while (local_row < tile_rows) : (local_row += 1) {
        const sampled_row = tile_start + local_row;
        const row = row_cache.row_indices[sampled_row];
        const row_values = source_values[row * col_count .. (row + 1) * col_count];

        for (0..group_count) |g| {
            const group_start = g * group_len;
            const group_values = row_values[group_start .. group_start + group_len];
            const row_group_idx = sampled_row * group_count + g;
            const min_val = row_cache.group_min[row_group_idx];
            const base_scale = row_cache.group_base_scale[row_group_idx];
            const group_scale_factor = if (g < group_scale_factors.len)
                group_scale_factors[g]
            else
                1.0;
            const group_bias_shift = if (g < group_bias_shifts.len)
                group_bias_shifts[g]
            else
                0.0;
            const group_round_shift = if (g < group_round_shifts.len)
                group_round_shifts[g]
            else
                0.0;
            const group_scale = base_scale * group_scale_factor;
            const group_bias = groupedBiasFromShift(min_val, group_scale, group_bias_shift);

            for (group_values, 0..) |value, value_idx| {
                const col = group_start + value_idx;
                var quantized: f32 = 0.0;
                if (group_scale > 0) {
                    const normalized = (value - group_bias) / group_scale + group_round_shift;
                    quantized = @max(0.0, @min(max_quant_value, @round(normalized)));
                }
                const dequant = quantized * group_scale + group_bias;
                const idx = col * tile_rows + local_row;
                if (ref_weights) |ref_buf| ref_buf[idx] = value;
                dq_weights[idx] = dequant;
            }
        }
    }
}

fn cudaCalibrationMaxSampleRowsPerTile(
    cuda_ctx: *const CudaCalibContext,
    input_samples: usize,
    col_count: usize,
) usize {
    if (input_samples == 0 or col_count == 0) return 0;
    const weight_row_bytes = std.math.mul(usize, col_count, @sizeOf(f32)) catch return 0;
    const output_row_bytes = std.math.mul(usize, input_samples, @sizeOf(f32)) catch return 0;
    if (weight_row_bytes == 0 or output_row_bytes == 0) return 0;

    const rows_by_w = cuda_ctx.w_dev.size / weight_row_bytes;
    const rows_by_ref_w = cuda_ctx.ref_w_dev.size / weight_row_bytes;
    const rows_by_out = cuda_ctx.out_dev.size / output_row_bytes;
    const rows_by_ref_out = cuda_ctx.ref_out_dev.size / output_row_bytes;
    const tile_rows = @min(@min(rows_by_w, rows_by_ref_w), @min(rows_by_out, rows_by_ref_out));
    if (tile_rows == 0) return 0;
    return tile_rows;
}

fn evaluateGroupedBlockForwardCalibrationCandidateCudaTiled(
    allocator: std.mem.Allocator,
    cuda_ctx: *CudaCalibContext,
    source_values: []align(1) const f32,
    col_count: usize,
    group_len: usize,
    max_quant_value: f32,
    row_cache: *const GroupedCalibrationRowCache,
    block_inputs: *const GroupedBlockInputMatrix,
    group_scale_factors: []const f32,
    group_bias_shifts: []const f32,
    group_round_shifts: []const f32,
    replay_targets: ?*const GroupedReplayTargetMatrix,
    ref_outputs: ?[]const f32,
    cuda_weight_scratch: ?*GroupedCudaWeightScratch,
) ?f64 {
    const input_samples = block_inputs.input_samples;
    const sample_rows_budget = row_cache.sample_rows;
    const use_replay_targets = replay_targets != null;
    const need_ref_matmul = !use_replay_targets and ref_outputs == null;
    const max_tile_rows = cudaCalibrationMaxSampleRowsPerTile(cuda_ctx, input_samples, col_count);
    if (max_tile_rows == 0) return null;

    const scratch_tile_rows = if (cuda_weight_scratch) |scratch|
        if (col_count > 0) @divTrunc(scratch.dq_weights.len, col_count) else 0
    else
        max_tile_rows;
    const tile_row_capacity = @min(sample_rows_budget, @min(max_tile_rows, scratch_tile_rows));
    if (tile_row_capacity == 0) return null;

    const x_slice = block_inputs.values[0 .. input_samples * col_count];
    if (cuda_ctx.cached_x_ptr != x_slice.ptr or cuda_ctx.cached_x_len != x_slice.len) {
        cuda_ctx.x_dev.upload(&cuda_ctx.device, std.mem.sliceAsBytes(x_slice)) catch return null;
        cuda_ctx.cached_x_ptr = x_slice.ptr;
        cuda_ctx.cached_x_len = x_slice.len;
    }

    const tile_weight_len = col_count * tile_row_capacity;
    var owned_dq_weights: ?[]f32 = null;
    defer if (owned_dq_weights) |buf| allocator.free(buf);
    const dq_weights_buf = if (cuda_weight_scratch) |scratch|
        scratch.dq_weights[0..tile_weight_len]
    else blk: {
        owned_dq_weights = allocator.alloc(f32, tile_weight_len) catch return null;
        break :blk owned_dq_weights.?;
    };

    var owned_ref_weights: ?[]f32 = null;
    defer if (owned_ref_weights) |buf| allocator.free(buf);
    const ref_weights_buf: ?[]f32 = if (need_ref_matmul)
        if (cuda_weight_scratch) |scratch|
            if (scratch.ref_weights) |buf|
                buf[0..tile_weight_len]
            else blk: {
                owned_ref_weights = allocator.alloc(f32, tile_weight_len) catch return null;
                break :blk owned_ref_weights.?;
            }
        else blk: {
            owned_ref_weights = allocator.alloc(f32, tile_weight_len) catch return null;
            break :blk owned_ref_weights.?;
        }
    else
        null;

    const tile_out_capacity = input_samples * tile_row_capacity;
    var owned_cached_ref_tile: ?[]f32 = null;
    defer if (owned_cached_ref_tile) |buf| allocator.free(buf);
    const cached_ref_tile_buf: ?[]f32 = if (!use_replay_targets and ref_outputs != null)
        allocator.alloc(f32, tile_out_capacity) catch return null
    else
        null;
    owned_cached_ref_tile = cached_ref_tile_buf;

    const dq_out_tile_buf = allocator.alloc(f32, tile_out_capacity) catch return null;
    defer allocator.free(dq_out_tile_buf);

    const ref_out_tile_buf: ?[]f32 = if (need_ref_matmul)
        allocator.alloc(f32, tile_out_capacity) catch return null
    else
        null;
    defer if (ref_out_tile_buf) |buf| allocator.free(buf);

    var weighted_err_sum: f64 = 0.0;
    var total_count: usize = 0;
    const use_gpu_mse = !use_replay_targets and (resetMseAccumulatorCuda(cuda_ctx) catch false);
    var tile_start: usize = 0;
    while (tile_start < sample_rows_budget) {
        const tile_rows = @min(tile_row_capacity, sample_rows_budget - tile_start);
        const dq_tile = dq_weights_buf[0 .. col_count * tile_rows];
        const ref_tile = if (ref_weights_buf) |buf| buf[0 .. col_count * tile_rows] else null;
        buildGroupedDqWeightsTile(
            source_values,
            col_count,
            group_len,
            max_quant_value,
            row_cache,
            group_scale_factors,
            group_bias_shifts,
            group_round_shifts,
            tile_start,
            tile_rows,
            ref_tile,
            dq_tile,
        );
        cuda_ctx.w_dev.upload(&cuda_ctx.device, std.mem.sliceAsBytes(dq_tile)) catch return null;

        if (ref_tile) |buf| {
            cuda_ctx.ref_w_dev.upload(&cuda_ctx.device, std.mem.sliceAsBytes(buf)) catch return null;
            cuda_ctx.blas.matmulF32(&cuda_ctx.device, &cuda_ctx.x_dev, input_samples, col_count, &cuda_ctx.ref_w_dev, tile_rows, &cuda_ctx.ref_out_dev) catch return null;
        } else if (!use_replay_targets) {
            const cached = ref_outputs.?;
            const cached_tile = cached_ref_tile_buf.?[0 .. input_samples * tile_rows];
            var p: usize = 0;
            while (p < input_samples) : (p += 1) {
                const src = cached[p * sample_rows_budget + tile_start .. p * sample_rows_budget + tile_start + tile_rows];
                const dst = cached_tile[p * tile_rows .. (p + 1) * tile_rows];
                @memcpy(dst, src);
            }
            cuda_ctx.ref_out_dev.upload(&cuda_ctx.device, std.mem.sliceAsBytes(cached_tile)) catch return null;
        }
        cuda_ctx.blas.matmulF32(&cuda_ctx.device, &cuda_ctx.x_dev, input_samples, col_count, &cuda_ctx.w_dev, tile_rows, &cuda_ctx.out_dev) catch return null;

        const tile_out_len = input_samples * tile_rows;
        if (use_gpu_mse) {
            if (!(accumulateMseCuda(allocator, cuda_ctx, &cuda_ctx.ref_out_dev, &cuda_ctx.out_dev, tile_out_len) catch false)) return null;
            total_count += tile_out_len;
            tile_start += tile_rows;
            continue;
        }

        cuda_ctx.device.synchronize() catch return null;
        if (use_replay_targets) {
            const dq_out_tile = dq_out_tile_buf[0..tile_out_len];
            cuda_ctx.out_dev.download(&cuda_ctx.device, std.mem.sliceAsBytes(dq_out_tile)) catch return null;
            const targets = replay_targets.?;
            var local_row: usize = 0;
            while (local_row < tile_rows) : (local_row += 1) {
                const row = row_cache.row_indices[tile_start + local_row];
                var p: usize = 0;
                while (p < input_samples) : (p += 1) {
                    const diff = @as(f64, @floatCast(targets.at(p, row) - dq_out_tile[p * tile_rows + local_row]));
                    weighted_err_sum += diff * diff;
                    total_count += 1;
                }
            }
        } else {
            const dq_out_tile = dq_out_tile_buf[0..tile_out_len];
            cuda_ctx.out_dev.download(&cuda_ctx.device, std.mem.sliceAsBytes(dq_out_tile)) catch return null;
            if (need_ref_matmul) {
                const ref_out_tile = ref_out_tile_buf.?[0..tile_out_len];
                cuda_ctx.ref_out_dev.download(&cuda_ctx.device, std.mem.sliceAsBytes(ref_out_tile)) catch return null;
                for (ref_out_tile, dq_out_tile) |a, b| {
                    const diff = @as(f64, @floatCast(a - b));
                    weighted_err_sum += diff * diff;
                    total_count += 1;
                }
            } else {
                const cached_tile = cached_ref_tile_buf.?[0..tile_out_len];
                for (cached_tile, dq_out_tile) |a, b| {
                    const diff = @as(f64, @floatCast(a - b));
                    weighted_err_sum += diff * diff;
                    total_count += 1;
                }
            }
        }

        tile_start += tile_rows;
    }

    if (use_gpu_mse) {
        return finishMseCuda(cuda_ctx, total_count) catch null;
    }

    if (total_count == 0) return null;
    return weighted_err_sum / @as(f64, @floatFromInt(total_count));
}

inline fn averageSlice(values: []const f32, fallback: f32) f32 {
    if (values.len == 0) return fallback;
    var sum: f64 = 0.0;
    for (values) |v| sum += @as(f64, @floatCast(v));
    return @as(f32, @floatCast(sum / @as(f64, @floatFromInt(values.len))));
}

fn clipSearchGroupedCalibration(
    allocator: std.mem.Allocator,
    metal_calibration_enabled: bool,
    cuda_ctx: ?*CudaCalibContext,
    source_values: []align(1) const f32,
    col_count: usize,
    group_len: usize,
    quant_bits: u8,
    row_cache: *const GroupedCalibrationRowCache,
    block_inputs: *const GroupedBlockInputMatrix,
    replay_targets: ?*const GroupedReplayTargetMatrix,
    ref_outputs: ?[]const f32,
    cuda_weight_scratch: ?*GroupedCudaWeightScratch,
    current_scale_factors: []f32,
    current_bias_shifts: []f32,
    current_round_shifts: []f32,
    baseline_err: f64,
) struct { err: f64, step: usize } {
    const clip_multipliers = [_]f32{ 1.00, 0.975, 0.95, 0.925, 0.90, 0.875, 0.85, 0.825, 0.80, 0.775, 0.75, 0.70, 0.65, 1.05, 1.10 };
    const round_shifts = [_]f32{ 0.0, -0.125, 0.125 };
    var best_err = baseline_err;
    var best_step: usize = 0;
    var best_mult: f32 = 1.0;
    var best_bias: f32 = 0.0;
    var best_shift: f32 = 0.0;
    var eval_idx: usize = 0;

    for (clip_multipliers) |mult| {
        for (round_shifts) |shift| {
            eval_idx += 1;
            @memset(current_scale_factors, mult);
            @memset(current_bias_shifts, 0.0);
            @memset(current_round_shifts, shift);
            const candidate_err = evaluateGroupedBlockForwardCalibrationCandidate(
                allocator,
                metal_calibration_enabled,
                cuda_ctx,
                source_values,
                col_count,
                group_len,
                quant_bits,
                row_cache,
                block_inputs,
                current_scale_factors,
                current_bias_shifts,
                current_round_shifts,
                replay_targets,
                ref_outputs,
                cuda_weight_scratch,
            );
            if (candidate_err < best_err) {
                best_err = candidate_err;
                best_step = eval_idx;
                best_mult = mult;
                best_bias = 0.0;
                best_shift = shift;
            }
        }
    }

    const refine_mult_offsets = [_]f32{ -0.05, -0.025, 0.0, 0.025, 0.05 };
    const refine_bias_offsets = [_]f32{ -0.0625, 0.0, 0.0625 };
    const refine_shift_offsets = [_]f32{ -0.0625, 0.0, 0.0625 };
    for (refine_mult_offsets) |moff| {
        for (refine_bias_offsets) |boff| {
            for (refine_shift_offsets) |soff| {
                eval_idx += 1;
                const mult = std.math.clamp(best_mult + moff, 0.5, 2.0);
                const bias = std.math.clamp(best_bias + boff, -1.0, 1.0);
                const shift = std.math.clamp(best_shift + soff, -0.5, 0.5);
                @memset(current_scale_factors, mult);
                @memset(current_bias_shifts, bias);
                @memset(current_round_shifts, shift);
                const candidate_err = evaluateGroupedBlockForwardCalibrationCandidate(
                    allocator,
                    metal_calibration_enabled,
                    cuda_ctx,
                    source_values,
                    col_count,
                    group_len,
                    quant_bits,
                    row_cache,
                    block_inputs,
                    current_scale_factors,
                    current_bias_shifts,
                    current_round_shifts,
                    replay_targets,
                    ref_outputs,
                    cuda_weight_scratch,
                );
                if (candidate_err < best_err) {
                    best_err = candidate_err;
                    best_step = eval_idx;
                    best_mult = mult;
                    best_bias = bias;
                    best_shift = shift;
                }
            }
        }
    }

    @memset(current_scale_factors, best_mult);
    @memset(current_bias_shifts, best_bias);
    @memset(current_round_shifts, best_shift);
    return .{ .err = best_err, .step = best_step };
}

fn estimateGroupedBlockForwardCalibrationParameters(
    allocator: std.mem.Allocator,
    cuda_ctx: ?*CudaCalibContext,
    source_tensors: ?*safetensors.UnifiedSafeTensors,
    source_values: []align(1) const f32,
    row_count: usize,
    col_count: usize,
    layer_index: ?u32,
    activation_role: calibration_capture.ActivationRole,
    replay_points: ?ReplayPointCandidates,
    group_len: usize,
    quant_bits: u8,
    options: ConvertOptions,
    token_pool: ?[]const u32,
    block_input_cache: ?*GroupedBlockInputCache,
) !GroupedCalibrationParams {
    if (row_count == 0 or col_count == 0 or group_len == 0 or col_count % group_len != 0) {
        const empty_scale = try allocator.alloc(f32, 1);
        errdefer allocator.free(empty_scale);
        const empty_bias = try allocator.alloc(f32, 1);
        errdefer allocator.free(empty_bias);
        const empty_round = try allocator.alloc(f32, 1);
        errdefer allocator.free(empty_round);
        empty_scale[0] = 1.0;
        empty_bias[0] = 0.0;
        empty_round[0] = 0.0;
        return .{
            .summary = .{
                .scale_factor = 1.0,
                .bias_shift = 0.0,
                .round_shift = 0.0,
                .best_mse = 0.0,
                .baseline_mse = 0.0,
                .first_mse = 0.0,
                .best_step = 0,
                .steps = 0,
                .best_iter = 0,
            },
            .group_scale_factors = empty_scale,
            .group_bias_shifts = empty_bias,
            .group_round_shifts = empty_round,
        };
    }

    const eval_budget = groupedCalibrationEvalBudget(row_count, col_count, options);
    const group_count = col_count / group_len;
    const steps = @max(@as(usize, 1), @as(usize, @intCast(@max(options.calib_iters, 1))));
    const row_offset = if (row_count > 0) @as(usize, @intCast(options.calib_seed % row_count)) else 0;
    const row_stride = if (eval_budget.row_samples > 0)
        @max(@as(usize, 1), row_count / eval_budget.row_samples)
    else
        1;
    var row_cache = try buildGroupedCalibrationRowCache(
        allocator,
        source_values,
        row_count,
        col_count,
        group_len,
        quant_bits,
        eval_budget.row_samples,
        row_offset,
        row_stride,
    );
    defer row_cache.deinit(allocator);
    var owned_block_inputs: ?GroupedBlockInputMatrix = null;
    defer if (owned_block_inputs) |*matrix| matrix.deinit(allocator);
    var replay_targets: ?GroupedReplayTargetMatrix = null;
    defer if (replay_targets) |targets| targets.deinit(allocator);

    const block_inputs = blk: {
        if (block_input_cache) |cache| {
            if (cache.activation_capture) |capture| {
                if (layer_index) |layer| {
                    if (replay_points) |pairs| {
                        var pair_idx: usize = 0;
                        while (pair_idx < pairs.count) : (pair_idx += 1) {
                            const pair = pairs.pairs[pair_idx];
                            if (try calibration_capture.sampleLayerActivationPairForPoints(
                                allocator,
                                capture,
                                layer,
                                col_count,
                                row_count,
                                eval_budget.input_samples,
                                options.calib_seed,
                                pair.input_point,
                                pair.output_point,
                            )) |sampled| {
                                replay_targets = .{
                                    .values = sampled.targets,
                                    .sample_count = sampled.sample_count,
                                    .cols = sampled.output_cols,
                                };
                                owned_block_inputs = .{
                                    .values = sampled.inputs,
                                    .input_samples = sampled.sample_count,
                                    .cols = sampled.input_cols,
                                };
                                break :blk &owned_block_inputs.?;
                            }
                        }
                    }
                }
            }
            break :blk try cache.get(layer_index, activation_role, col_count, eval_budget.input_samples);
        }
        owned_block_inputs = try buildGroupedBlockInputMatrix(
            allocator,
            source_tensors,
            col_count,
            eval_budget.input_samples,
            token_pool,
            options.calib_seed,
            source_tensors != null and options.calib_iters > 0,
        );
        break :blk &owned_block_inputs.?;
    };
    const metal_calibration_enabled = isMetalCalibrationEnabled();
    const ref_outputs = if (replay_targets == null) blk: {
        if (metal_calibration_enabled) {
            if (buildGroupedBlockForwardReferenceOutputsMetal(allocator, source_values, col_count, &row_cache, block_inputs)) |r| break :blk r;
        }
        if (cuda_ctx) |ctx| {
            if (buildGroupedBlockForwardReferenceOutputsCuda(allocator, ctx, source_values, col_count, &row_cache, block_inputs)) |r| break :blk r;
        }
        break :blk @as(?[]f32, null);
    } else @as(?[]f32, null);
    defer if (ref_outputs) |buf| allocator.free(buf);

    var current_scale_factors = try allocator.alloc(f32, group_count);
    errdefer allocator.free(current_scale_factors);
    var current_bias_shifts = try allocator.alloc(f32, group_count);
    errdefer allocator.free(current_bias_shifts);
    var current_round_shifts = try allocator.alloc(f32, group_count);
    errdefer allocator.free(current_round_shifts);
    @memset(current_scale_factors, 1.0);
    @memset(current_bias_shifts, 0.0);
    @memset(current_round_shifts, 0.0);

    var cuda_weight_scratch: ?GroupedCudaWeightScratch = null;
    defer if (cuda_weight_scratch) |*scratch| {
        scratch.deinit(allocator, if (cuda_ctx) |ctx| &ctx.device else null);
    };
    if (cuda_ctx != null) {
        const tile_row_capacity = if (cuda_ctx) |ctx| blk: {
            const max_tile_rows = cudaCalibrationMaxSampleRowsPerTile(ctx, block_inputs.input_samples, col_count);
            if (max_tile_rows == 0) break :blk @as(usize, 0);
            break :blk @min(row_cache.sample_rows, max_tile_rows);
        } else row_cache.sample_rows;
        const scratch_len = col_count * tile_row_capacity;
        const dq_weights = try allocator.alloc(f32, scratch_len);
        const ref_weights = if (replay_targets == null and ref_outputs == null)
            try allocator.alloc(f32, scratch_len)
        else
            @as(?[]f32, null);
        cuda_weight_scratch = GroupedCudaWeightScratch{
            .dq_weights = dq_weights,
            .ref_weights = ref_weights,
            .device = null,
        };

        if (cuda_ctx) |ctx| {
            if (ctx.cuda_dq_weight_build_available and
                tile_row_capacity == row_cache.sample_rows and
                row_cache.sample_rows > 0 and
                col_count > 0 and
                group_count > 0)
            {
                const sampled_source = allocator.alloc(f32, scratch_len) catch null;
                if (sampled_source) |sampled| {
                    defer allocator.free(sampled);
                    for (0..tile_row_capacity) |sampled_row| {
                        const row = row_cache.row_indices[sampled_row];
                        const src_row = source_values[row * col_count .. (row + 1) * col_count];
                        const dst_row = sampled[sampled_row * col_count .. (sampled_row + 1) * col_count];
                        @memcpy(dst_row, src_row);
                    }

                    const sampled_bytes = std.math.mul(usize, scratch_len, @sizeOf(f32)) catch 0;
                    const row_group_len = std.math.mul(usize, tile_row_capacity, group_count) catch 0;
                    const row_group_bytes = std.math.mul(usize, row_group_len, @sizeOf(f32)) catch 0;
                    const group_param_bytes = std.math.mul(usize, group_count, @sizeOf(f32)) catch 0;

                    if (sampled_bytes > 0 and sampled_bytes <= ctx.w_dev.size and row_group_bytes > 0 and group_param_bytes > 0) {
                        var sampled_source_dev = ctx.device.allocBuffer(sampled_bytes) catch null;
                        if (sampled_source_dev) |*sampled_dev| {
                            defer if (cuda_weight_scratch.?.device == null) sampled_dev.deinit(&ctx.device);
                            var row_group_min_dev = ctx.device.allocBuffer(row_group_bytes) catch null;
                            if (row_group_min_dev) |*min_dev| {
                                defer if (cuda_weight_scratch.?.device == null) min_dev.deinit(&ctx.device);
                                var row_group_base_scale_dev = ctx.device.allocBuffer(row_group_bytes) catch null;
                                if (row_group_base_scale_dev) |*base_dev| {
                                    defer if (cuda_weight_scratch.?.device == null) base_dev.deinit(&ctx.device);
                                    var group_scale_factors_dev = ctx.device.allocBuffer(group_param_bytes) catch null;
                                    if (group_scale_factors_dev) |*sf_dev| {
                                        defer if (cuda_weight_scratch.?.device == null) sf_dev.deinit(&ctx.device);
                                        var group_bias_shifts_dev = ctx.device.allocBuffer(group_param_bytes) catch null;
                                        if (group_bias_shifts_dev) |*bs_dev| {
                                            defer if (cuda_weight_scratch.?.device == null) bs_dev.deinit(&ctx.device);
                                            var group_round_shifts_dev = ctx.device.allocBuffer(group_param_bytes) catch null;
                                            if (group_round_shifts_dev) |*rs_dev| {
                                                defer if (cuda_weight_scratch.?.device == null) rs_dev.deinit(&ctx.device);

                                                var uploads_ok = true;
                                                sampled_dev.upload(&ctx.device, std.mem.sliceAsBytes(sampled)) catch {
                                                    uploads_ok = false;
                                                };
                                                min_dev.upload(&ctx.device, std.mem.sliceAsBytes(row_cache.group_min[0..row_group_len])) catch {
                                                    uploads_ok = false;
                                                };
                                                base_dev.upload(&ctx.device, std.mem.sliceAsBytes(row_cache.group_base_scale[0..row_group_len])) catch {
                                                    uploads_ok = false;
                                                };

                                                if (uploads_ok) {
                                                    cuda_weight_scratch.?.device = .{
                                                        .sampled_source_dev = sampled_dev.*,
                                                        .row_group_min_dev = min_dev.*,
                                                        .row_group_base_scale_dev = base_dev.*,
                                                        .group_scale_factors_dev = sf_dev.*,
                                                        .group_bias_shifts_dev = bs_dev.*,
                                                        .group_round_shifts_dev = rs_dev.*,
                                                    };
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    const baseline_err = evaluateGroupedBlockForwardCalibrationCandidate(
        allocator,
        metal_calibration_enabled,
        cuda_ctx,
        source_values,
        col_count,
        group_len,
        quant_bits,
        &row_cache,
        block_inputs,
        current_scale_factors,
        current_bias_shifts,
        current_round_shifts,
        if (replay_targets) |*targets| targets else null,
        ref_outputs,
        if (cuda_weight_scratch) |*scratch| scratch else null,
    );
    const optimizer = calibrationOptimizerFromEnv(options.profile);

    var current_err: f64 = baseline_err;
    var best_step: usize = 0;
    if (optimizer == .clip or optimizer == .clip_search) {
        const clip = clipSearchGroupedCalibration(
            allocator,
            metal_calibration_enabled,
            cuda_ctx,
            source_values,
            col_count,
            group_len,
            quant_bits,
            &row_cache,
            block_inputs,
            if (replay_targets) |*targets| targets else null,
            ref_outputs,
            if (cuda_weight_scratch) |*scratch| scratch else null,
            current_scale_factors,
            current_bias_shifts,
            current_round_shifts,
            baseline_err,
        );
        current_err = clip.err;
        best_step = clip.step;
    }

    if (steps == 1 or optimizer == .clip or group_count == 0) {
        return .{
            .summary = .{
                .scale_factor = averageSlice(current_scale_factors, 1.0),
                .bias_shift = averageSlice(current_bias_shifts, 0.0),
                .round_shift = averageSlice(current_round_shifts, 0.0),
                .best_mse = current_err,
                .baseline_mse = baseline_err,
                .first_mse = baseline_err,
                .best_step = best_step,
                .steps = 1,
                .best_iter = 0,
            },
            .group_scale_factors = current_scale_factors,
            .group_bias_shifts = current_bias_shifts,
            .group_round_shifts = current_round_shifts,
        };
    }

    const best_scale_factors = try allocator.alloc(f32, group_count);
    errdefer allocator.free(best_scale_factors);
    const best_bias_shifts = try allocator.alloc(f32, group_count);
    errdefer allocator.free(best_bias_shifts);
    const best_round_shifts = try allocator.alloc(f32, group_count);
    errdefer allocator.free(best_round_shifts);
    @memcpy(best_scale_factors, current_scale_factors);
    @memcpy(best_bias_shifts, current_bias_shifts);
    @memcpy(best_round_shifts, current_round_shifts);

    var best_err: f64 = current_err;
    const first_err: f64 = baseline_err;
    const nblocks = @as(usize, @intCast(@max(options.calib_nblocks, 1)));
    // Adaptive update coverage:
    // - keep small blocks when many steps are available (acceptance stability)
    // - increase coverage for low-iter runs so more groups are tuned per pass
    const coarse_base_block: usize = @max(@as(usize, 1), @min(group_count, 12 * nblocks));
    const coarse_target_cover_steps = @max(@as(usize, 1), steps / 4);
    const coarse_cover_block = @max(@as(usize, 1), @min(group_count, @divFloor(group_count + coarse_target_cover_steps - 1, coarse_target_cover_steps)));
    const coarse_block_size: usize = @max(coarse_base_block, coarse_cover_block);
    const fine_block_size: usize = @max(@as(usize, 1), @min(group_count, 2 * nblocks));
    const max_update_block_size: usize = @max(coarse_block_size, fine_block_size);
    const block_prev_scale = try allocator.alloc(f32, max_update_block_size);
    defer allocator.free(block_prev_scale);
    const block_prev_bias = try allocator.alloc(f32, max_update_block_size);
    defer allocator.free(block_prev_bias);
    const block_prev_round = try allocator.alloc(f32, max_update_block_size);
    defer allocator.free(block_prev_round);
    const block_acc_scale = try allocator.alloc(f32, max_update_block_size);
    defer allocator.free(block_acc_scale);
    const block_acc_bias = try allocator.alloc(f32, max_update_block_size);
    defer allocator.free(block_acc_bias);
    const block_acc_round = try allocator.alloc(f32, max_update_block_size);
    defer allocator.free(block_acc_round);
    const dir_scale_mem = try allocator.alloc(f32, group_count);
    defer allocator.free(dir_scale_mem);
    const dir_bias_mem = try allocator.alloc(f32, group_count);
    defer allocator.free(dir_bias_mem);
    const dir_round_mem = try allocator.alloc(f32, group_count);
    defer allocator.free(dir_round_mem);
    for (0..group_count) |g| {
        const dseed = groupedMix64(options.calib_seed ^ (@as(u64, @intCast(g + 1)) *% 0x9e3779b97f4a7c15));
        dir_scale_mem[g] = if ((dseed & 1) == 0) -1.0 else 1.0;
        dir_bias_mem[g] = if ((dseed & 2) == 0) -1.0 else 1.0;
        dir_round_mem[g] = if ((dseed & 4) == 0) -1.0 else 1.0;
    }

    var step_idx: usize = 1;
    while (step_idx < steps) : (step_idx += 1) {
        const progress = @as(f32, @floatFromInt(step_idx)) / @as(f32, @floatFromInt(steps));
        const anneal = @max(@as(f32, 0.05), 1.0 - progress);
        const base_lr = @max(@as(f32, 1e-3), @min(@as(f32, 0.25), 1.5 / @as(f32, @floatFromInt(@max(options.calib_iters, 1)))));
        const lr_scale = base_lr * anneal;
        const lr_bias = (base_lr * 0.5) * anneal;
        const lr_round = (base_lr * 0.35) * anneal;
        const row_seed = row_offset;
        const update_block_size = if (progress >= 0.6) fine_block_size else coarse_block_size;
        // Deterministic cyclic coverage gives better low-iter convergence than
        // randomized starts while staying reproducible.
        const block_stride = if (progress >= 0.6) @as(usize, 1) else update_block_size;
        const block_start = (row_seed + step_idx * block_stride) % group_count;

        var block_slot: usize = 0;
        while (block_slot < update_block_size) : (block_slot += 1) {
            const g = (block_start + block_slot) % group_count;
            block_prev_scale[block_slot] = current_scale_factors[g];
            block_prev_bias[block_slot] = current_bias_shifts[g];
            block_prev_round[block_slot] = current_round_shifts[g];
        }

        const backtrack = [_]f32{ 1.0, 0.5, 0.25, 0.125 };
        var accepted_err = current_err;
        var accepted = false;
        var accepted_bt: usize = 0;
        var accepted_mode: usize = 0;
        var bt_idx: usize = 0;
        while (bt_idx < backtrack.len) : (bt_idx += 1) {
            const step_scale = lr_scale * backtrack[bt_idx];
            const step_bias = lr_bias * backtrack[bt_idx];
            const step_round = lr_round * backtrack[bt_idx];
            var mode: usize = 0;
            while (mode < 4) : (mode += 1) {
                block_slot = 0;
                while (block_slot < update_block_size) : (block_slot += 1) {
                    const g = (block_start + block_slot) % group_count;
                    var dir_scale = dir_scale_mem[g];
                    var dir_bias = dir_bias_mem[g];
                    var dir_round = dir_round_mem[g];
                    if (mode == 1) {
                        dir_scale = -dir_scale;
                        dir_bias = -dir_bias;
                        dir_round = -dir_round;
                    } else if (mode >= 2) {
                        const dseed = groupedMix64(options.calib_seed ^
                            (@as(u64, @intCast(step_idx + 1)) *% 0x9e3779b97f4a7c15) ^
                            (@as(u64, @intCast(g + 1)) *% 0xbf58476d1ce4e5b9));
                        dir_scale = if ((dseed & 1) == 0) -1.0 else 1.0;
                        dir_bias = if ((dseed & 2) == 0) -1.0 else 1.0;
                        dir_round = if ((dseed & 4) == 0) -1.0 else 1.0;
                        if (mode == 3) {
                            dir_scale = -dir_scale;
                            dir_bias = -dir_bias;
                            dir_round = -dir_round;
                        }
                    }
                    current_scale_factors[g] = std.math.clamp(block_prev_scale[block_slot] + dir_scale * step_scale, 0.125, 8.0);
                    current_bias_shifts[g] = std.math.clamp(block_prev_bias[block_slot] + dir_bias * step_bias, -4.0, 4.0);
                    current_round_shifts[g] = std.math.clamp(block_prev_round[block_slot] + dir_round * step_round, -1.0, 1.0);
                }

                const candidate_err = evaluateGroupedBlockForwardCalibrationCandidate(
                    allocator,
                    metal_calibration_enabled,
                    cuda_ctx,
                    source_values,
                    col_count,
                    group_len,
                    quant_bits,
                    &row_cache,
                    block_inputs,
                    current_scale_factors,
                    current_bias_shifts,
                    current_round_shifts,
                    if (replay_targets) |*targets| targets else null,
                    ref_outputs,
                    if (cuda_weight_scratch) |*scratch| scratch else null,
                );
                if (candidate_err < accepted_err) {
                    accepted_err = candidate_err;
                    accepted = true;
                    accepted_bt = bt_idx;
                    accepted_mode = mode;
                    block_slot = 0;
                    while (block_slot < update_block_size) : (block_slot += 1) {
                        const g = (block_start + block_slot) % group_count;
                        block_acc_scale[block_slot] = current_scale_factors[g];
                        block_acc_bias[block_slot] = current_bias_shifts[g];
                        block_acc_round[block_slot] = current_round_shifts[g];
                    }
                }
            }
        }

        if (!accepted) {
            block_slot = 0;
            while (block_slot < update_block_size) : (block_slot += 1) {
                const g = (block_start + block_slot) % group_count;
                current_scale_factors[g] = block_prev_scale[block_slot];
                current_bias_shifts[g] = block_prev_bias[block_slot];
                current_round_shifts[g] = block_prev_round[block_slot];
                dir_scale_mem[g] = -dir_scale_mem[g];
                dir_bias_mem[g] = -dir_bias_mem[g];
                dir_round_mem[g] = -dir_round_mem[g];
            }
        } else {
            block_slot = 0;
            while (block_slot < update_block_size) : (block_slot += 1) {
                const g = (block_start + block_slot) % group_count;
                current_scale_factors[g] = block_acc_scale[block_slot];
                current_bias_shifts[g] = block_acc_bias[block_slot];
                current_round_shifts[g] = block_acc_round[block_slot];
                if (accepted_mode == 1) {
                    dir_scale_mem[g] = -dir_scale_mem[g];
                    dir_bias_mem[g] = -dir_bias_mem[g];
                    dir_round_mem[g] = -dir_round_mem[g];
                } else if (accepted_mode >= 2) {
                    const dseed = groupedMix64(options.calib_seed ^
                        (@as(u64, @intCast(step_idx + 1)) *% 0x9e3779b97f4a7c15) ^
                        (@as(u64, @intCast(g + 1)) *% 0xbf58476d1ce4e5b9));
                    var next_scale: f32 = if ((dseed & 1) == 0) -1.0 else 1.0;
                    var next_bias: f32 = if ((dseed & 2) == 0) -1.0 else 1.0;
                    var next_round: f32 = if ((dseed & 4) == 0) -1.0 else 1.0;
                    if (accepted_mode == 3) {
                        next_scale = -next_scale;
                        next_bias = -next_bias;
                        next_round = -next_round;
                    }
                    dir_scale_mem[g] = next_scale;
                    dir_bias_mem[g] = next_bias;
                    dir_round_mem[g] = next_round;
                }
            }

            const accepted_bt_idx = @min(accepted_bt, backtrack.len - 1);
            const refine_scale = (lr_scale * backtrack[accepted_bt_idx]) * 0.5;
            const refine_bias = (lr_bias * backtrack[accepted_bt_idx]) * 0.5;
            const refine_round = (lr_round * backtrack[accepted_bt_idx]) * 0.5;
            block_slot = 0;
            while (block_slot < update_block_size) : (block_slot += 1) {
                const g = (block_start + block_slot) % group_count;
                const dir_scale = dir_scale_mem[g];
                const dir_bias = dir_bias_mem[g];
                const dir_round = dir_round_mem[g];
                current_scale_factors[g] = std.math.clamp(block_acc_scale[block_slot] + dir_scale * refine_scale, 0.125, 8.0);
                current_bias_shifts[g] = std.math.clamp(block_acc_bias[block_slot] + dir_bias * refine_bias, -4.0, 4.0);
                current_round_shifts[g] = std.math.clamp(block_acc_round[block_slot] + dir_round * refine_round, -1.0, 1.0);
            }
            const refine_err = evaluateGroupedBlockForwardCalibrationCandidate(
                allocator,
                metal_calibration_enabled,
                cuda_ctx,
                source_values,
                col_count,
                group_len,
                quant_bits,
                &row_cache,
                block_inputs,
                current_scale_factors,
                current_bias_shifts,
                current_round_shifts,
                if (replay_targets) |*targets| targets else null,
                ref_outputs,
                if (cuda_weight_scratch) |*scratch| scratch else null,
            );
            if (refine_err < accepted_err) {
                accepted_err = refine_err;
            } else {
                block_slot = 0;
                while (block_slot < update_block_size) : (block_slot += 1) {
                    const g = (block_start + block_slot) % group_count;
                    current_scale_factors[g] = block_acc_scale[block_slot];
                    current_bias_shifts[g] = block_acc_bias[block_slot];
                    current_round_shifts[g] = block_acc_round[block_slot];
                }
            }
        }

        if (accepted) {
            current_err = accepted_err;
        }
        if (current_err < best_err) {
            best_err = current_err;
            best_step = step_idx;
            @memcpy(best_scale_factors, current_scale_factors);
            @memcpy(best_bias_shifts, current_bias_shifts);
            @memcpy(best_round_shifts, current_round_shifts);
        }
    }

    const best_iter = if (steps > 1)
        @divTrunc(best_step * @as(usize, @intCast(@max(options.calib_iters, 1) - 1)), steps - 1)
    else
        0;

    allocator.free(current_scale_factors);
    allocator.free(current_bias_shifts);
    allocator.free(current_round_shifts);
    return .{
        .summary = .{
            .scale_factor = averageSlice(best_scale_factors, 1.0),
            .bias_shift = averageSlice(best_bias_shifts, 0.0),
            .round_shift = averageSlice(best_round_shifts, 0.0),
            .best_mse = best_err,
            .baseline_mse = baseline_err,
            .first_mse = first_err,
            .best_step = best_step,
            .steps = steps,
            .best_iter = best_iter,
        },
        .group_scale_factors = best_scale_factors,
        .group_bias_shifts = best_bias_shifts,
        .group_round_shifts = best_round_shifts,
    };
}

fn estimateGroupedBlockForwardCalibration(
    allocator: std.mem.Allocator,
    source_tensors: ?*safetensors.UnifiedSafeTensors,
    source_values: []align(1) const f32,
    row_count: usize,
    col_count: usize,
    group_len: usize,
    quant_bits: u8,
    options: ConvertOptions,
    token_pool: ?[]const u32,
) !GroupedCalibrationSummary {
    var params = try estimateGroupedBlockForwardCalibrationParameters(
        allocator,
        null,
        source_tensors,
        source_values,
        row_count,
        col_count,
        null,
        .generic,
        null,
        group_len,
        quant_bits,
        options,
        token_pool,
        null,
    );
    defer params.deinit(allocator);
    return params.summary;
}

fn emitGroupedLayerCalibrationProgress(
    mode: CalibrationProgressMode,
    block_map: CalibrationBlockMap,
    layer: u32,
    tensor_count: usize,
    first_sum: f64,
    best_sum: f64,
    best_iter: usize,
    running_avg_mse: f64,
    running_avg_base: f64,
    best_running_normalized_ratio: f64,
    quantized_tensors_so_far: usize,
    total_target_tensors: usize,
) void {
    _ = best_iter;
    if (tensor_count == 0) return;
    const avg_first = first_sum / @as(f64, @floatFromInt(tensor_count));
    const avg_best = best_sum / @as(f64, @floatFromInt(tensor_count));
    const improvement_pct = if (avg_first > 0 and avg_best <= avg_first)
        ((avg_first - avg_best) / avg_first) * 100.0
    else
        0.0;
    const normalized_ratio = if (avg_first > 0) avg_best / avg_first else 1.0;
    const running_normalized_ratio = if (running_avg_base > 0) running_avg_mse / running_avg_base else 1.0;
    const running_reduction_pct = if (running_avg_base > 0 and running_avg_mse <= running_avg_base)
        ((running_avg_base - running_avg_mse) / running_avg_base) * 100.0
    else
        0.0;
    const best_row_ratio = @min(best_running_normalized_ratio, running_normalized_ratio);
    const coverage_pct = if (total_target_tensors > 0)
        (@as(f64, @floatFromInt(quantized_tensors_so_far)) / @as(f64, @floatFromInt(total_target_tensors))) * 100.0
    else
        100.0;
    const model_state_ratio = 1.0 - ((coverage_pct / 100.0) * (1.0 - running_normalized_ratio));
    const model_state_reduction_pct = (1.0 - model_state_ratio) * 100.0;
    switch (mode) {
        .layer => std.debug.print(
            "layer {d}: quantized {d} tensors | block_baseline_mse={d:.12} block_mse={d:.12} block_normalized_mse_ratio={d:.6} block_relative_mse_reduction_pct={d:.2}% | running_weighted_baseline_mse={d:.12} running_weighted_mse={d:.12} running_normalized_mse_ratio={d:.6} running_relative_mse_reduction_pct={d:.2}% best_running_normalized_mse_ratio={d:.6} | model_state_coverage_pct={d:.2}% model_state_normalized_mse_ratio={d:.6} model_state_relative_mse_reduction_pct={d:.2}%\n",
            .{
                layer,
                tensor_count,
                avg_first,
                avg_best,
                normalized_ratio,
                improvement_pct,
                running_avg_base,
                running_avg_mse,
                running_normalized_ratio,
                running_reduction_pct,
                best_row_ratio,
                coverage_pct,
                model_state_ratio,
                model_state_reduction_pct,
            },
        ),
        .block => {
            if (block_map.variantName(layer)) |variant_name| {
                std.debug.print(
                    "block {d} ({s}): quantized {d} tensors | block_baseline_mse={d:.12} block_mse={d:.12} block_normalized_mse_ratio={d:.6} block_relative_mse_reduction_pct={d:.2}% | running_weighted_baseline_mse={d:.12} running_weighted_mse={d:.12} running_normalized_mse_ratio={d:.6} running_relative_mse_reduction_pct={d:.2}% best_running_normalized_mse_ratio={d:.6} | model_state_coverage_pct={d:.2}% model_state_normalized_mse_ratio={d:.6} model_state_relative_mse_reduction_pct={d:.2}%\n",
                    .{
                        layer,
                        variant_name,
                        tensor_count,
                        avg_first,
                        avg_best,
                        normalized_ratio,
                        improvement_pct,
                        running_avg_base,
                        running_avg_mse,
                        running_normalized_ratio,
                        running_reduction_pct,
                        best_row_ratio,
                        coverage_pct,
                        model_state_ratio,
                        model_state_reduction_pct,
                    },
                );
            } else {
                std.debug.print(
                    "block {d}: quantized {d} tensors | block_baseline_mse={d:.12} block_mse={d:.12} block_normalized_mse_ratio={d:.6} block_relative_mse_reduction_pct={d:.2}% | running_weighted_baseline_mse={d:.12} running_weighted_mse={d:.12} running_normalized_mse_ratio={d:.6} running_relative_mse_reduction_pct={d:.2}% best_running_normalized_mse_ratio={d:.6} | model_state_coverage_pct={d:.2}% model_state_normalized_mse_ratio={d:.6} model_state_relative_mse_reduction_pct={d:.2}%\n",
                    .{
                        layer,
                        tensor_count,
                        avg_first,
                        avg_best,
                        normalized_ratio,
                        improvement_pct,
                        running_avg_base,
                        running_avg_mse,
                        running_normalized_ratio,
                        running_reduction_pct,
                        best_row_ratio,
                        coverage_pct,
                        model_state_ratio,
                        model_state_reduction_pct,
                    },
                );
            }
        },
    }
}

fn emitGroupedTensorCalibrationProgress(tensor_name: []const u8, calib_summary: GroupedCalibrationSummary) void {
    const improvement_pct = if (calib_summary.baseline_mse > 0 and calib_summary.best_mse <= calib_summary.baseline_mse)
        ((calib_summary.baseline_mse - calib_summary.best_mse) / calib_summary.baseline_mse) * 100.0
    else
        0.0;
    std.debug.print(
        "tensor {s}: loss iter 0: {d:.12} -> iter {d}: {d:.12}, base={d:.12}, improve={d:.2}%, sf={d:.6}, b={d:.6}, v={d:.6}\n",
        .{
            tensor_name,
            calib_summary.first_mse,
            calib_summary.best_iter,
            calib_summary.best_mse,
            calib_summary.baseline_mse,
            improvement_pct,
            calib_summary.scale_factor,
            calib_summary.bias_shift,
            calib_summary.round_shift,
        },
    );
}

fn emitGroupedNonLayerCalibrationProgress(
    tensor_name: []const u8,
    running_avg_mse: f64,
    running_avg_base: f64,
    best_running_normalized_ratio: f64,
    quantized_tensors_so_far: usize,
    total_target_tensors: usize,
    calib_summary: GroupedCalibrationSummary,
) void {
    const improvement_pct = if (calib_summary.baseline_mse > 0 and calib_summary.best_mse <= calib_summary.baseline_mse)
        ((calib_summary.baseline_mse - calib_summary.best_mse) / calib_summary.baseline_mse) * 100.0
    else
        0.0;
    const normalized_ratio = if (calib_summary.baseline_mse > 0) calib_summary.best_mse / calib_summary.baseline_mse else 1.0;
    const running_normalized_ratio = if (running_avg_base > 0) running_avg_mse / running_avg_base else 1.0;
    const running_reduction_pct = if (running_avg_base > 0 and running_avg_mse <= running_avg_base)
        ((running_avg_base - running_avg_mse) / running_avg_base) * 100.0
    else
        0.0;
    const best_row_ratio = @min(best_running_normalized_ratio, running_normalized_ratio);
    const coverage_pct = if (total_target_tensors > 0)
        (@as(f64, @floatFromInt(quantized_tensors_so_far)) / @as(f64, @floatFromInt(total_target_tensors))) * 100.0
    else
        100.0;
    const model_state_ratio = 1.0 - ((coverage_pct / 100.0) * (1.0 - running_normalized_ratio));
    const model_state_reduction_pct = (1.0 - model_state_ratio) * 100.0;
    std.debug.print(
        "quantized non-layer: {s} | block_baseline_mse={d:.12} block_mse={d:.12} block_normalized_mse_ratio={d:.6} block_relative_mse_reduction_pct={d:.2}% | running_weighted_baseline_mse={d:.12} running_weighted_mse={d:.12} running_normalized_mse_ratio={d:.6} running_relative_mse_reduction_pct={d:.2}% best_running_normalized_mse_ratio={d:.6} | model_state_coverage_pct={d:.2}% model_state_normalized_mse_ratio={d:.6} model_state_relative_mse_reduction_pct={d:.2}%, sf={d:.6}, b={d:.6}, v={d:.6}\n",
        .{
            tensor_name,
            calib_summary.baseline_mse,
            calib_summary.best_mse,
            normalized_ratio,
            improvement_pct,
            running_avg_base,
            running_avg_mse,
            running_normalized_ratio,
            running_reduction_pct,
            best_row_ratio,
            coverage_pct,
            model_state_ratio,
            model_state_reduction_pct,
            calib_summary.scale_factor,
            calib_summary.bias_shift,
            calib_summary.round_shift,
        },
    );
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

fn replayPointCandidatesForTensorName(name: []const u8) ?ReplayPointCandidates {
    if (std.mem.indexOf(u8, name, ".self_attn.q_proj.weight") != null) {
        return .{
            .pairs = .{
                .{ .input_point = .layer_attn_norm, .output_point = .attn_q },
                undefined,
                undefined,
            },
            .count = 1,
        };
    }
    if (std.mem.indexOf(u8, name, ".self_attn.k_proj.weight") != null) {
        return .{
            .pairs = .{
                .{ .input_point = .layer_attn_norm, .output_point = .attn_k },
                undefined,
                undefined,
            },
            .count = 1,
        };
    }
    if (std.mem.indexOf(u8, name, ".self_attn.v_proj.weight") != null) {
        return .{
            .pairs = .{
                .{ .input_point = .layer_attn_norm, .output_point = .attn_v },
                undefined,
                undefined,
            },
            .count = 1,
        };
    }
    if (std.mem.indexOf(u8, name, ".mlp.gate_proj.weight") != null) {
        return .{
            .pairs = .{
                .{ .input_point = .layer_ffn_norm, .output_point = .ffn_gate },
                .{ .input_point = .layer_ffn_norm, .output_point = .ffn_act_map },
                undefined,
            },
            .count = 2,
        };
    }
    if (std.mem.indexOf(u8, name, ".mlp.up_proj.weight") != null or
        std.mem.indexOf(u8, name, ".mlp.fc1.weight") != null)
    {
        return .{
            .pairs = .{
                .{ .input_point = .layer_ffn_norm, .output_point = .ffn_up },
                .{ .input_point = .layer_ffn_norm, .output_point = .ffn_gate },
                undefined,
            },
            .count = 2,
        };
    }
    if (std.mem.indexOf(u8, name, ".mlp.down_proj.weight") != null or
        std.mem.indexOf(u8, name, ".mlp.fc2.weight") != null)
    {
        return .{
            .pairs = .{
                .{ .input_point = .ffn_act_mix, .output_point = .ffn_down },
                .{ .input_point = .ffn_act, .output_point = .ffn_down },
                undefined,
            },
            .count = 2,
        };
    }
    if (std.mem.indexOf(u8, name, ".linear_attn.in_proj_qkv.weight") != null) {
        return .{
            .pairs = .{
                .{ .input_point = .layer_attn_norm, .output_point = .conv_in_proj },
                .{ .input_point = .layer_attn_norm, .output_point = .gdelta_in_proj },
                undefined,
            },
            .count = 2,
        };
    }
    if (std.mem.indexOf(u8, name, ".linear_attn.in_proj_a.weight") != null) {
        return .{
            .pairs = .{
                .{ .input_point = .layer_attn_norm, .output_point = .gdelta_in_proj },
                .{ .input_point = .layer_attn_norm, .output_point = .conv_in_proj },
                undefined,
            },
            .count = 2,
        };
    }
    if (std.mem.indexOf(u8, name, ".linear_attn.out_proj.weight") != null) {
        return .{
            .pairs = .{
                .{ .input_point = .gdelta_norm, .output_point = .gdelta_out },
                .{ .input_point = .conv_conv, .output_point = .conv_out_proj },
                undefined,
            },
            .count = 2,
        };
    }
    return null;
}

/// Quantize a tensor to grouped-affine weights (4-bit or 8-bit).
fn quantizeGroupedAffineTensor(
    allocator: std.mem.Allocator,
    cuda_ctx: ?*CudaCalibContext,
    source_tensors: *safetensors.UnifiedSafeTensors,
    builder: *safetensors.Builder,
    tensor_name: []const u8,
    source_tensor: Tensor,
    quant_config: QuantConfig,
    options: ConvertOptions,
    token_pool: ?[]const u32,
    block_input_cache: *GroupedBlockInputCache,
) !GroupedCalibrationSummary {
    const quant_bits = quant_config.bits;
    if (quant_bits != 4 and quant_bits != 8) return error.UnsupportedBits;
    const layer_index = extractLayerIndexFromTensorName(tensor_name);

    // Support both 2D [rows, cols] and 3D [batch, rows, cols] tensors.
    // For 3D (fused expert weights), flatten batch*rows as total_rows for quantization.
    const is_3d = source_tensor.n_dims == 3;
    const batch_dim: usize = if (is_3d) @intCast(source_tensor.shape[0]) else 1;
    const inner_rows: usize = if (is_3d) @intCast(source_tensor.shape[1]) else @intCast(source_tensor.shape[0]);
    const col_count: usize = if (is_3d) @intCast(source_tensor.shape[2]) else @intCast(source_tensor.shape[1]);
    const row_count = batch_dim * inner_rows;
    const group_len = quant_config.group_size;

    const values_per_u32: usize = if (quant_bits == 4) 8 else 4;

    // Ensure cols is divisible by group_size and values_per_word
    if (col_count % group_len != 0 or col_count % values_per_u32 != 0) {
        try copyTensorUnchanged(allocator, builder, tensor_name, source_tensor);
        return .{
            .scale_factor = 1.0,
            .bias_shift = 0.0,
            .round_shift = 0.0,
            .best_mse = 0.0,
            .baseline_mse = 0.0,
            .first_mse = 0.0,
            .best_step = 0,
            .steps = 0,
            .best_iter = 0,
        };
    }

    // Convert source to F32.
    // FP8 sources must be dequantized with weight_scale_inv before requantization.
    const f32_source = try tensorToF32ForQuantization(allocator, source_tensors, tensor_name, source_tensor);
    defer f32_source.deinit(allocator);

    const source_values = f32_source.asF32Slice();
    const activation_role = activationRoleForTensorName(tensor_name);
    const replay_points = replayPointCandidatesForTensorName(tensor_name);
    var calib_params = try estimateGroupedBlockForwardCalibrationParameters(
        allocator,
        cuda_ctx,
        source_tensors,
        source_values,
        row_count,
        col_count,
        layer_index,
        activation_role,
        replay_points,
        group_len,
        quant_bits,
        options,
        token_pool,
        block_input_cache,
    );
    defer calib_params.deinit(allocator);
    const calib_summary = calib_params.summary;

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

    var used_cuda_quant = false;
    if (cuda_ctx) |ctx| {
        if (!ctx.cuda_quantization_failed) {
            used_cuda_quant = tryQuantizeGroupedAffineRowsCuda(
                allocator,
                ctx,
                source_values,
                row_count,
                col_count,
                group_len,
                quant_bits,
                calib_params.group_scale_factors,
                calib_params.group_bias_shifts,
                calib_params.group_round_shifts,
                packed_row_words,
                scale_values,
                bias_values,
            ) catch |err| blk: {
                if (!ctx.cuda_quantization_failed) {
                    log.warn("convert", "CUDA grouped-affine quantization failed; falling back to CPU", .{
                        .tensor = tensor_name,
                        .err = @errorName(err),
                    });
                    ctx.cuda_quantization_failed = true;
                }
                break :blk false;
            };
        }
    }
    if (!used_cuda_quant) {
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
            .group_scale_factors = calib_params.group_scale_factors,
            .group_bias_shifts = calib_params.group_bias_shifts,
            .group_round_shifts = calib_params.group_round_shifts,
        };

        const pool = parallel.global();
        pool.parallelForCompute(row_count, quantizeRowSlice, &quant_ctx);
    }

    // Add tensors to builder — preserve 3D shape for fused expert weights
    const quant_dtype: DType = if (quant_bits == 4) .grouped_affine_u4 else .grouped_affine_u8;
    if (is_3d) {
        try builder.addTensor(
            tensor_name,
            quant_dtype,
            &[_]usize{ batch_dim, inner_rows, packed_col_count },
            std.mem.sliceAsBytes(packed_row_words),
        );
    } else {
        try builder.addTensor(
            tensor_name,
            quant_dtype,
            &[_]usize{ row_count, packed_col_count },
            std.mem.sliceAsBytes(packed_row_words),
        );
    }

    // Scales and biases names
    var scales_name_buf: [256]u8 = undefined;
    const tensor_base_name = if (std.mem.endsWith(u8, tensor_name, ".weight"))
        tensor_name[0 .. tensor_name.len - ".weight".len]
    else
        tensor_name;
    const scales_tensor_name = try std.fmt.bufPrint(&scales_name_buf, "{s}.scales", .{tensor_base_name});
    if (is_3d) {
        try builder.addTensor(
            scales_tensor_name,
            .bf16,
            &[_]usize{ batch_dim, inner_rows, group_count },
            std.mem.sliceAsBytes(scale_values),
        );
    } else {
        try builder.addTensor(
            scales_tensor_name,
            .bf16,
            &[_]usize{ row_count, group_count },
            std.mem.sliceAsBytes(scale_values),
        );
    }

    var biases_name_buf: [256]u8 = undefined;
    const biases_tensor_name = try std.fmt.bufPrint(&biases_name_buf, "{s}.biases", .{tensor_base_name});
    if (is_3d) {
        try builder.addTensor(
            biases_tensor_name,
            .bf16,
            &[_]usize{ batch_dim, inner_rows, group_count },
            std.mem.sliceAsBytes(bias_values),
        );
    } else {
        try builder.addTensor(
            biases_tensor_name,
            .bf16,
            &[_]usize{ row_count, group_count },
            std.mem.sliceAsBytes(bias_values),
        );
    }
    return calib_summary;
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

/// Convert a tensor to BF16 and write it unquantized.
/// Used for embedding tensors when TALU_CONVERT_EMBED_BITS=16.
fn copyTensorAsBf16(
    allocator: std.mem.Allocator,
    builder: *safetensors.Builder,
    tensor_name: []const u8,
    source_tensor: Tensor,
) !void {
    const shape_array = source_tensor.shapeAsUsize();
    const shape = shape_array[0..@intCast(source_tensor.n_dims)];

    if (source_tensor.dtype == .bf16) {
        // Already BF16 — copy as-is.
        try builder.addTensor(tensor_name, .bf16, shape, source_tensor.data()[0..source_tensor.data_size]);
        return;
    }

    // Convert to F32 first, then to BF16.
    const f32_result = try convert.tensorToF32(allocator, source_tensor);
    defer f32_result.deinit(allocator);
    const f32_slice = f32_result.asF32Slice();

    const bf16_values = try allocator.alloc(u16, f32_slice.len);
    defer allocator.free(bf16_values);
    for (f32_slice, bf16_values) |val, *dst| {
        dst.* = convert.f32ToBf16(val);
    }

    try builder.addTensor(tensor_name, .bf16, shape, std.mem.sliceAsBytes(bf16_values));
}

/// Write an embedding tensor in MXFP8 format (E4M3 values + UE8M0 block
/// scales).  MXFP8 is ~2x smaller than BF16 and the runtime only needs to
/// dequantize the rows selected by token IDs (negligible cost).  For tied-
/// weight models the lm_head QMM path can consume the packed data directly.
fn copyTensorAsMxfp8(
    allocator: std.mem.Allocator,
    source_tensors: *safetensors.UnifiedSafeTensors,
    builder: *safetensors.Builder,
    tensor_name: []const u8,
    source_tensor: Tensor,
) !void {
    _ = source_tensors;
    if (source_tensor.n_dims != 2) return error.UnsupportedShape;

    const rows: usize = @intCast(source_tensor.shape[0]);
    const cols: usize = @intCast(source_tensor.shape[1]);
    const scale_cols = (cols + 31) / 32;

    const f32_result = try convert.tensorToF32(allocator, source_tensor);
    defer f32_result.deinit(allocator);
    const f32_slice = f32_result.asF32Slice();

    const fp8_values = try allocator.alloc(u8, rows * cols);
    defer allocator.free(fp8_values);
    const e8m0_scales = try allocator.alloc(u8, rows * scale_cols);
    defer allocator.free(e8m0_scales);

    for (0..rows) |row| {
        const row_src = f32_slice[row * cols ..][0..cols];
        const row_dst = fp8_values[row * cols ..][0..cols];
        const row_scales = e8m0_scales[row * scale_cols ..][0..scale_cols];

        for (0..scale_cols) |g| {
            const start = g * 32;
            const end = @min(start + 32, cols);

            // Compute group absmax.
            var absmax: f32 = 0.0;
            for (row_src[start..end]) |v| {
                const av = @abs(v);
                if (av > absmax) absmax = av;
            }

            // E8M0 encoding: shared_exp = floor(log2(absmax)) - 8, clamped.
            const shared_exp: f32 = blk: {
                if (absmax <= 0.0) break :blk -8.0;
                const exp_floor = @floor(std.math.log2(absmax));
                break :blk std.math.clamp(exp_floor - 8.0, -127.0, 127.0);
            };
            const e8m0: u8 = @intCast(@as(u32, @intFromFloat(std.math.clamp(shared_exp + 127.0, 0.0, 255.0))));
            row_scales[g] = e8m0;

            // Inverse scale for quantizing values into E4M3 range.
            const scale_bits: u32 = @as(u32, e8m0) << 23;
            const scale: f32 = @bitCast(scale_bits);
            const inv_scale: f32 = if (scale > 0.0) 1.0 / scale else 0.0;

            for (row_src[start..end], row_dst[start..end]) |v, *dst| {
                dst.* = dtype_mod.f32ToFp8E4M3(v * inv_scale);
            }
        }
    }

    const shape_2d: [2]usize = .{ rows, cols };
    try builder.addTensor(tensor_name, .f8_e4m3, &shape_2d, fp8_values);

    // Write block scales as a companion tensor.
    const base = if (std.mem.endsWith(u8, tensor_name, ".weight"))
        tensor_name[0 .. tensor_name.len - ".weight".len]
    else
        tensor_name;
    var scale_name_buf: [512]u8 = undefined;
    const scale_name = std.fmt.bufPrint(&scale_name_buf, "{s}.weight_block_scale", .{base}) catch return error.NameTooLong;
    const scale_shape: [2]usize = .{ rows, scale_cols };
    try builder.addTensor(scale_name, .u8, &scale_shape, e8m0_scales);
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
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{ .sub_path = "config.json", .data = "{}" });
    {
        var dir = try tmp.dir.openDir(".", .{});
        defer dir.close();
        try std.testing.expect(!isCompleteConversionOutput(dir));
    }

    {
        var empty = try tmp.dir.createFile("model.safetensors", .{});
        empty.close();
    }
    {
        var dir = try tmp.dir.openDir(".", .{});
        defer dir.close();
        try std.testing.expect(!isCompleteConversionOutput(dir));
    }

    const tmp_real_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_real_path);
    var builder = safetensors.Builder.init(allocator);
    defer builder.deinit();
    const data = [_]u8{1};
    try builder.addTensor("w", .u8, &[_]usize{1}, &data);
    try builder.save(tmp_real_path, "model.safetensors");
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
    const group_scale_factors = [_]f32{1.0} ** group_count;
    const group_bias_shifts = [_]f32{0.0} ** group_count;
    const group_round_shifts = [_]f32{0.0} ** group_count;

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
        .group_scale_factors = group_scale_factors[0..],
        .group_bias_shifts = group_bias_shifts[0..],
        .group_round_shifts = group_round_shifts[0..],
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
    const group_scale_factors = [_]f32{1.0} ** group_count;
    const group_bias_shifts = [_]f32{0.0} ** group_count;
    const group_round_shifts = [_]f32{0.0} ** group_count;

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
        .group_scale_factors = group_scale_factors[0..],
        .group_bias_shifts = group_bias_shifts[0..],
        .group_round_shifts = group_round_shifts[0..],
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
    const group_scale_factors = [_]f32{1.0} ** group_count;
    const group_bias_shifts = [_]f32{0.0} ** group_count;
    const group_round_shifts = [_]f32{0.0} ** group_count;

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
        .group_scale_factors = group_scale_factors[0..],
        .group_bias_shifts = group_bias_shifts[0..],
        .group_round_shifts = group_round_shifts[0..],
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
    const group_scale_factors = [_]f32{1.0} ** group_count;
    const group_bias_shifts = [_]f32{0.0} ** group_count;
    const group_round_shifts = [_]f32{0.0} ** group_count;

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
        .group_scale_factors = group_scale_factors[0..],
        .group_bias_shifts = group_bias_shifts[0..],
        .group_round_shifts = group_round_shifts[0..],
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
    const group_scale_factors = [_]f32{1.0} ** group_count;
    const group_bias_shifts = [_]f32{0.0} ** group_count;
    const group_round_shifts = [_]f32{0.0} ** group_count;

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
        .group_scale_factors = group_scale_factors[0..],
        .group_bias_shifts = group_bias_shifts[0..],
        .group_round_shifts = group_round_shifts[0..],
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
    const group_scale_factors = [_]f32{1.0} ** group_count;
    const group_bias_shifts = [_]f32{0.0} ** group_count;
    const group_round_shifts = [_]f32{0.0} ** group_count;

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
        .group_scale_factors = group_scale_factors[0..],
        .group_bias_shifts = group_bias_shifts[0..],
        .group_round_shifts = group_round_shifts[0..],
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

test "estimateGroupedBlockForwardCalibration is deterministic for same calibration seed" {
    var values: [128]f32 = undefined;
    for (&values, 0..) |*v, i| {
        v.* = (@as(f32, @floatFromInt(i % 13)) - 6.0) * 0.5;
    }

    const options = ConvertOptions{
        .profile = .custom,
        .calib_iters = 23,
        .calib_nsamples = 8,
        .calib_seqlen = 512,
        .calib_batch_size = 2,
        .calib_nblocks = 2,
        .calib_seed = 1234,
    };

    const a = try estimateGroupedBlockForwardCalibration(std.testing.allocator, null, &values, 8, 16, 16, 4, options, null);
    const b = try estimateGroupedBlockForwardCalibration(std.testing.allocator, null, &values, 8, 16, 16, 4, options, null);
    try std.testing.expectEqual(a.steps, b.steps);
    try std.testing.expectEqual(a.best_step, b.best_step);
    try std.testing.expectEqual(a.best_iter, b.best_iter);
    try std.testing.expectApproxEqRel(a.best_mse, b.best_mse, 1e-12);
    try std.testing.expectApproxEqRel(a.baseline_mse, b.baseline_mse, 1e-12);
}

test "estimateGroupedBlockForwardCalibration honors calibration iteration budget" {
    var values: [256]f32 = undefined;
    for (&values, 0..) |*v, i| {
        v.* = @sin(@as(f32, @floatFromInt(i)));
    }

    const options = ConvertOptions{
        .profile = .custom,
        .calib_iters = 41,
        .calib_nsamples = 16,
        .calib_seqlen = 2048,
        .calib_batch_size = 1,
        .calib_nblocks = 1,
        .calib_seed = 42,
    };

    const summary = try estimateGroupedBlockForwardCalibration(std.testing.allocator, null, &values, 16, 16, 16, 8, options, null);
    try std.testing.expectEqual(@as(usize, 41), summary.steps);
    try std.testing.expect(summary.best_iter < 41);
    try std.testing.expect(summary.best_step < summary.steps);
}

test "estimateGroupedBlockForwardCalibration good profile runs clip+search calibration" {
    var values: [64]f32 = undefined;
    for (&values, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i));

    const options = ConvertOptions{
        .profile = .good,
        .calib_iters = 8,
        .calib_nsamples = 16,
        .calib_seqlen = 2048,
        .calib_batch_size = 1,
        .calib_nblocks = 1,
        .calib_seed = 7,
    };

    const summary = try estimateGroupedBlockForwardCalibration(std.testing.allocator, null, &values, 4, 16, 16, 4, options, null);
    try std.testing.expect(summary.steps >= 1);
    try std.testing.expect(summary.best_mse <= summary.baseline_mse);
}

test "parseCalibrationOptimizer accepts aliases and rejects unknown values" {
    try std.testing.expectEqual(@as(?CalibrationOptimizer, .search), parseCalibrationOptimizer("search"));
    try std.testing.expectEqual(@as(?CalibrationOptimizer, .search), parseCalibrationOptimizer("SEARCH"));
    try std.testing.expectEqual(@as(?CalibrationOptimizer, .clip), parseCalibrationOptimizer("clip"));
    try std.testing.expectEqual(@as(?CalibrationOptimizer, .clip_search), parseCalibrationOptimizer("clip+search"));
    try std.testing.expectEqual(@as(?CalibrationOptimizer, .clip_search), parseCalibrationOptimizer("clip_search"));
    try std.testing.expectEqual(@as(?CalibrationOptimizer, null), parseCalibrationOptimizer("foo"));
}

test "calibrationOptimizerFromEnv maps defaults per profile" {
    if (std.posix.getenv("TALU_CONVERT_CALIB_OPTIMIZER") != null) return error.SkipZigTest;
    try std.testing.expectEqual(CalibrationOptimizer.clip_search, calibrationOptimizerFromEnv(.good));
    try std.testing.expectEqual(CalibrationOptimizer.clip_search, calibrationOptimizerFromEnv(.best));
    try std.testing.expectEqual(CalibrationOptimizer.search, calibrationOptimizerFromEnv(.custom));
}

test "parseCalibrationProgressMode accepts aliases and rejects unknown values" {
    try std.testing.expectEqual(@as(?CalibrationProgressMode, .block), parseCalibrationProgressMode("block"));
    try std.testing.expectEqual(@as(?CalibrationProgressMode, .block), parseCalibrationProgressMode("BLOCKS"));
    try std.testing.expectEqual(@as(?CalibrationProgressMode, .layer), parseCalibrationProgressMode("layer"));
    try std.testing.expectEqual(@as(?CalibrationProgressMode, .layer), parseCalibrationProgressMode("  Layers \n"));
    try std.testing.expectEqual(@as(?CalibrationProgressMode, null), parseCalibrationProgressMode("unit"));
    try std.testing.expectEqual(@as(?CalibrationProgressMode, null), parseCalibrationProgressMode(""));
}

test "groupedCalibrationEvalBudget scales input samples with calib_nblocks" {
    const n1 = groupedCalibrationEvalBudget(32, 64, .{
        .profile = .custom,
        .calib_nsamples = 32,
        .calib_seqlen = 4,
        .calib_batch_size = 1,
        .calib_nblocks = 1,
    });
    const n4 = groupedCalibrationEvalBudget(32, 64, .{
        .profile = .custom,
        .calib_nsamples = 32,
        .calib_seqlen = 4,
        .calib_batch_size = 1,
        .calib_nblocks = 4,
    });
    try std.testing.expectEqual(@as(usize, 4), n1.input_samples);
    try std.testing.expectEqual(@as(usize, 16), n4.input_samples);
    try std.testing.expectEqual(n1.row_samples, n4.row_samples);

    const capped = groupedCalibrationEvalBudget(32, 64, .{
        .profile = .custom,
        .calib_nsamples = 32,
        .calib_seqlen = 2048,
        .calib_batch_size = 1,
        .calib_nblocks = 8,
    });
    try std.testing.expectEqual(@as(usize, max_calibration_input_samples), capped.input_samples);
}

test "extractLayerIndexFromTensorName parses canonical layer paths" {
    try std.testing.expectEqual(@as(?u32, 0), extractLayerIndexFromTensorName("model.language_model.layers.0.mlp.down_proj.weight"));
    try std.testing.expectEqual(@as(?u32, 23), extractLayerIndexFromTensorName("model.language_model.layers.23.self_attn.q_proj.weight"));
    try std.testing.expectEqual(@as(?u32, null), extractLayerIndexFromTensorName("lm_head.weight"));
}

test "activationRoleForTensorName maps attention and ffn tensors" {
    try std.testing.expectEqual(
        calibration_capture.ActivationRole.attn_input,
        activationRoleForTensorName("model.language_model.layers.0.self_attn.q_proj.weight"),
    );
    try std.testing.expectEqual(
        calibration_capture.ActivationRole.attn_output,
        activationRoleForTensorName("model.language_model.layers.0.self_attn.o_proj.weight"),
    );
    try std.testing.expectEqual(
        calibration_capture.ActivationRole.ffn_input,
        activationRoleForTensorName("model.language_model.layers.0.mlp.gate_proj.weight"),
    );
    try std.testing.expectEqual(
        calibration_capture.ActivationRole.ffn_output,
        activationRoleForTensorName("model.language_model.layers.0.mlp.down_proj.weight"),
    );
    try std.testing.expectEqual(
        calibration_capture.ActivationRole.generic,
        activationRoleForTensorName("lm_head.weight"),
    );
}

test "replayPointCandidatesForTensorName maps key tensor paths" {
    const q = replayPointCandidatesForTensorName("model.language_model.layers.0.self_attn.q_proj.weight").?;
    try std.testing.expectEqual(@as(usize, 1), q.count);
    try std.testing.expectEqual(xray.TracePoint.layer_attn_norm, q.pairs[0].input_point);
    try std.testing.expectEqual(xray.TracePoint.attn_q, q.pairs[0].output_point);

    const down = replayPointCandidatesForTensorName("model.language_model.layers.4.mlp.down_proj.weight").?;
    try std.testing.expectEqual(@as(usize, 2), down.count);
    try std.testing.expectEqual(xray.TracePoint.ffn_act_mix, down.pairs[0].input_point);
    try std.testing.expectEqual(xray.TracePoint.ffn_down, down.pairs[0].output_point);

    const linear = replayPointCandidatesForTensorName("model.language_model.layers.4.linear_attn.out_proj.weight").?;
    try std.testing.expectEqual(@as(usize, 2), linear.count);
    try std.testing.expectEqual(xray.TracePoint.gdelta_norm, linear.pairs[0].input_point);
    try std.testing.expectEqual(xray.TracePoint.gdelta_out, linear.pairs[0].output_point);

    try std.testing.expect(replayPointCandidatesForTensorName("model.language_model.layers.0.self_attn.o_proj.weight") == null);
}

test "sorted tensor names maintain nondecreasing layer order" {
    var names = [_][]const u8{
        "model.language_model.layers.11.self_attn.v_proj.weight",
        "model.language_model.layers.2.mlp.down_proj.weight",
        "lm_head.weight",
        "model.language_model.layers.0.linear_attn.out_proj.weight",
        "model.language_model.layers.11.mlp.down_proj.weight",
        "model.language_model.layers.2.self_attn.q_proj.weight",
    };
    convert.sortTensorNames(names[0..]);

    var prev_layer: ?u32 = null;
    for (names) |name| {
        const layer = extractLayerIndexFromTensorName(name) orelse continue;
        if (prev_layer) |prev| {
            try std.testing.expect(layer >= prev);
        }
        prev_layer = layer;
    }
}

test "CUDA calibration context disables GPU MSE to ensure F64 precision" {
    // Regression test: GPU MSE reduction uses F32 accumulation with non-deterministic
    // atomicAdd, which can mislead the calibration optimizer for certain weight
    // distributions (e.g., up_proj in 27B models got 46% wrong scale factors).
    // The CPU F64 fallback path must always be used for calibration MSE.
    if (comptime !has_cuda_gpu_calib) return error.SkipZigTest;
    if (compute.cuda.device.probeRuntime() != .available) return error.SkipZigTest;
    if (!isCudaCalibrationEnabled()) return error.SkipZigTest;

    var ctx = CudaCalibContext.init() catch return error.SkipZigTest;
    defer ctx.deinit();

    try std.testing.expect(!ctx.cuda_mse_available);
}

test "calibrationDatasetRequired requires dataset whenever calibration iterations are enabled" {
    try std.testing.expect(calibrationDatasetRequired(.{
        .profile = .best,
        .calib_iters = 500,
    }));
    try std.testing.expect(calibrationDatasetRequired(.{
        .profile = .good,
        .calib_iters = 1,
    }));
    try std.testing.expect(!calibrationDatasetRequired(.{
        .profile = .best,
        .calib_iters = 0,
    }));
}
