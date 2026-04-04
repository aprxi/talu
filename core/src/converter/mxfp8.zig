//! MXFP8 Model Conversion
//!
//! Converts transformer models to MXFP8 quantization: E4M3 data + UE8M0
//! 1×32 block scales. This format is native to Blackwell (sm_120) tensor
//! cores via cuBLASLt block-scaled GEMM.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const log = @import("../log.zig");
const tensor = @import("../tensor.zig");
const dtype_mod = @import("../dtype.zig");
const safetensors = @import("../io/safetensors/root.zig");
const repository = @import("../io/repository/root.zig");
const gaf_paths = @import("gaf_paths.zig");
const config_loader = @import("../models/config/root.zig");
const op_types = @import("../models/op_types.zig");
const parallel = @import("../system/parallel.zig");
const compute = @import("../compute/root.zig");
const convert = @import("root.zig");
const models_registry = @import("../models/registry.zig");
const load_transforms = @import("../models/load/transforms.zig");
const json = @import("../io/json/root.zig");
const http = @import("../io/transport/http.zig");
const tokenizer_mod = @import("../tokenizer/root.zig");
const calibration_capture = @import("calibration_capture.zig");
const xray = @import("../xray/root.zig");
const has_metal_gpu_calib = build_options.enable_metal and builtin.os.tag == .macos;

const Tensor = tensor.Tensor;
const DType = dtype_mod.DType;

/// MXFP8 block size: 32 elements per scale group.
const group_size: usize = 32;
const max_calibration_input_samples: usize = 4096;
const calibration_rows_max_bytes: usize = 32 * 1024 * 1024;
const calibration_eval_max_bytes: usize = 32 * 1024 * 1024;

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
    profile: convert.scheme.QualityProfile = .best,
    calib_iters: u32 = 400,
    calib_nsamples: u32 = 256,
    calib_seqlen: u32 = 2048,
    calib_batch_size: u32 = 1,
    calib_nblocks: u32 = 1,
    calib_seed: u64 = 42,
};

pub const modelIdFromOutputPath = gaf_paths.modelIdFromOutputPath;

const CalibrationProgressMode = enum {
    block,
    layer,
};

const CalibrationOptimizer = enum {
    search,
    clip,
    clip_search,
};

fn calibrationOptimizerOverrideName(optimizer: CalibrationOptimizer) []const u8 {
    return switch (optimizer) {
        .search => "search",
        .clip => "clip",
        .clip_search => "clip+search",
    };
}

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

fn isCalibrationProbeOnly(options: ConvertOptions) bool {
    return options.profile == .custom and options.calib_iters > 0 and envFlagEnabled("TALU_CONVERT_CALIB_PROBE_ONLY");
}

/// Convert a transformer model to MXFP8 (E4M3 + UE8M0 block-32 scales).
/// Returns the output path (caller owns the memory).
pub fn convertToMxfp8(
    allocator: std.mem.Allocator,
    input_path: []const u8,
    options: ConvertOptions,
) ![]const u8 {
    const probe_only = isCalibrationProbeOnly(options);

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
    const output_tmp_path = if (!probe_only)
        try std.fmt.allocPrint(allocator, "{s}.tmp", .{output_dir_path})
    else
        null;
    defer if (output_tmp_path) |path| allocator.free(path);
    const write_output_path = if (output_tmp_path) |path| path else output_dir_path;

    // 4. Check if output exists (skip this in probe-only mode so probe never mutates output artifacts).
    if (!probe_only) {
        // Always clean stale temporary output from a prior interrupted run.
        std.fs.cwd().deleteTree(write_output_path) catch {};
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
    errdefer if (!probe_only and !keep_output) std.fs.cwd().deleteTree(write_output_path) catch {};
    var output_dir: ?gaf_paths.GAFModelDir = null;
    if (!probe_only) {
        output_dir = try gaf_paths.GAFModelDir.init(allocator, write_output_path);
    }
    defer if (output_dir) |*dir| dir.deinit();

    // 8. Process and write weights
    try writeMxfp8Weights(
        allocator,
        &source_tensors,
        model_config.tie_word_embeddings,
        options.max_shard_size,
        write_output_path,
        model_bundle.tokenizer_path(),
        options.progress,
        options,
        if (layout_map) |*lm| lm else null,
        if (fusion_map) |*fm| fm else null,
        runtime_arch,
        layer_types_override,
    );

    if (probe_only) {
        log.info("convert", "MXFP8 calibration probe complete (artifact emission skipped)", .{
            .output_dir = output_dir_path,
        });
        return output_dir_path;
    }

    // 9. Copy config.json with MXFP8 quantization info
    try copyConfigWithMxfp8Quantization(allocator, model_bundle.config_path(), write_output_path, model_config.tie_word_embeddings);

    // 10. Copy all model assets
    try convert.copyModelAssets(allocator, model_bundle.dir, write_output_path);

    // 11. Generate model card
    const model_name = convert.model_card.extractModelName(input_path);
    const base_model_id = convert.model_card.extractBaseModelId(input_path);
    convert.model_card.writeModelCard(allocator, write_output_path, model_name, base_model_id, .mxfp8) catch |err| {
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
    try std.fs.cwd().rename(write_output_path, output_dir_path);

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
    tensor_builder.max_shard_size = max_shard_size;

    const tensor_names = try source_tensors.tensorNames(allocator);
    defer allocator.free(tensor_names);
    convert.sortTensorNames(tensor_names);

    const show_calib_progress = options.calib_iters > 0 and
        envFlagEnabledDefault("TALU_CONVERT_CALIB_PROGRESS", true);
    const show_per_tensor_calib = show_calib_progress and envFlagEnabled("TALU_CONVERT_CALIB_PER_TENSOR");
    const calib_probe_only = show_calib_progress and envFlagEnabled("TALU_CONVERT_CALIB_PROBE_ONLY");
    const calib_layer_window = CalibrationLayerWindow.fromEnv();
    const progress_mode = calibrationProgressModeFromEnv();
    const optimizer_mode = calibrationOptimizerFromEnv(options.profile);
    const experiment_settings = calibrationExperimentSettingsFromEnv();
    const block_map: CalibrationBlockMap = .{
        .arch = runtime_arch,
        .layer_types_override = layer_types_override,
    };
    if (!show_calib_progress) {
        progress.addLine(0, "Converting", tensor_names.len, null, "tensors");
    } else {
        // The CLI may already have an active progress line for conversion. Close it so
        // per-unit calibration logs remain visible and newline-stable.
        progress.completeLine(0);
        std.debug.print("Calib {s} loss minimization\n", .{@tagName(progress_mode)});
        const scorer_backend = if (isGpuCalibrationEnabled()) "metal" else "cpu";
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
        if (!experiment_settings.isDefault()) {
            std.debug.print(
                "Calib experimental args: clip_min={d:.6},clip_max={d:.6},shift_max={d:.6},adaptive_clip_floor={d:.6}\n",
                .{
                    experiment_settings.clip_min,
                    experiment_settings.clip_max,
                    experiment_settings.shift_max,
                    experiment_settings.adaptive_clip_floor,
                },
            );
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
    const token_pool = blk: {
        const loaded = loadCalibrationTokenPool(allocator, tokenizer_path, options) catch |err| {
            if (options.profile != .fast and options.calib_iters > 0) return err;
            log.warn("convert", "Failed to load calibration token pool; using deterministic block-input fallback activations", .{
                .err = @errorName(err),
                .dataset = "NeelNanda/pile-10k",
            });
            break :blk null;
        };
        if (loaded == null and options.profile != .fast and options.calib_iters > 0) {
            return error.CalibrationDataUnavailable;
        }
        break :blk loaded;
    };
    defer if (token_pool) |pool| allocator.free(pool);
    const require_embedding_lookup = options.profile != .fast and options.calib_iters > 0;
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
    var block_input_cache = BlockInputCache.init(
        allocator,
        source_tensors,
        token_pool,
        options.calib_seed,
        require_embedding_lookup,
        if (activation_capture_cache) |*cache| cache else null,
    );
    defer block_input_cache.deinit();
    const target_quantized_tensors = estimateTargetQuantizedTensorCount(
        tensor_names,
        layout_map,
        source_tensors,
        tie_embeddings,
        fusion_map,
        calib_probe_only,
        calib_layer_window,
    ) + blk: {
        if (!tie_embeddings or calib_probe_only) break :blk @as(usize, 0);
        if (findEmbeddingTensorName(layout_map, source_tensors) == null) break :blk @as(usize, 0);
        break :blk @as(usize, 1);
    };
    for (tensor_names, 0..) |tensor_name, tensor_index| {
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

        if (convert.shouldSkipForTiedEmbeddingsByName(layout_map, tensor_name, tie_embeddings)) {
            continue;
        }

        if (fusion_map) |map| {
            if (map.isConsumedNonTrigger(tensor_name)) continue;
            if (map.planForTrigger(tensor_name)) |plan| {
                if (try maybeWriteFusedTensorForPlan(allocator, source_tensors, &tensor_builder, plan, options, token_pool, &block_input_cache)) {
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
                const layer_index = extractLayerIndexFromTensorName(tensor_name);
                if (show_calib_progress) {
                    if (active_layer) |layer| {
                        if (layer_index == null or layer_index.? != layer) {
                            const running_avg_mse = mse_sum / @as(f64, @floatFromInt(@max(quantized_layers, 1)));
                            const running_avg_base = baseline_sum / @as(f64, @floatFromInt(@max(quantized_layers, 1)));
                            emitLayerCalibrationProgress(
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
                if (calib_probe_only and calib_layer_window.enabled() and !calib_layer_window.allows(layer_index)) {
                    continue;
                }
                const tensor_options = blk: {
                    if (show_calib_progress and calib_layer_window.enabled() and !calib_layer_window.allows(layer_index)) {
                        var narrowed = options;
                        narrowed.calib_iters = 1;
                        break :blk narrowed;
                    }
                    break :blk options;
                };
                if (show_calib_progress and calib_layer_window.enabled() and tensor_options.calib_iters == 1 and options.calib_iters > 1) {
                    baseline_tensor_count += 1;
                } else {
                    tuned_tensor_count += 1;
                }
                const calib_summary = try quantizeMxfp8Tensor(allocator, source_tensors, &tensor_builder, tensor_name, source_tensor, tensor_options, token_pool, &block_input_cache);
                quantized_layers += 1;
                mse_sum += calib_summary.best_mse;
                baseline_sum += calib_summary.baseline_mse;
                const running_avg_mse = mse_sum / @as(f64, @floatFromInt(@max(quantized_layers, 1)));
                const running_avg_base = baseline_sum / @as(f64, @floatFromInt(@max(quantized_layers, 1)));
                if (show_calib_progress) {
                    if (show_per_tensor_calib) {
                        emitTensorCalibrationProgress(tensor_name, calib_summary);
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
                        emitNonLayerCalibrationProgress(
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
            if (calib_probe_only) continue;
            try copyTensorUnchanged(allocator, &tensor_builder, tensor_name, source_tensor);
        }
    }

    if (show_calib_progress) {
        if (active_layer) |layer| {
            const running_avg_mse = mse_sum / @as(f64, @floatFromInt(@max(quantized_layers, 1)));
            const running_avg_base = baseline_sum / @as(f64, @floatFromInt(@max(quantized_layers, 1)));
            emitLayerCalibrationProgress(
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

    // Synthesize MXFP8 lm_head from embedding when embeddings are tied.
    if (tie_embeddings and !calib_probe_only) {
        if (findEmbeddingTensorName(layout_map, source_tensors)) |embed_name| {
            const embed_tensor = try source_tensors.getTensor(embed_name, null);
            const calib_summary = try quantizeMxfp8Tensor(allocator, source_tensors, &tensor_builder, "lm_head.weight", embed_tensor, options, token_pool, &block_input_cache);
            quantized_layers += 1;
            mse_sum += calib_summary.best_mse;
            baseline_sum += calib_summary.baseline_mse;
            const running_avg_mse = mse_sum / @as(f64, @floatFromInt(@max(quantized_layers, 1)));
            const running_avg_base = baseline_sum / @as(f64, @floatFromInt(@max(quantized_layers, 1)));
            if (show_calib_progress) {
                if (show_per_tensor_calib) {
                    emitTensorCalibrationProgress("lm_head.weight", calib_summary);
                } else {
                    emitNonLayerCalibrationProgress(
                        "lm_head.weight",
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
        } else {
            log.warn("convert", "MXFP8 tied-embedding synth failed to resolve embedding tensor", .{});
            return error.NotFound;
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
            "Calib done tensors={d} units={d} mode={s} avg_mse={d:.12} avg_base={d:.12} avg_improve={d:.2}%\n",
            .{ quantized_layers, emitted_unit_count, @tagName(progress_mode), avg_mse, avg_baseline, avg_improvement_pct },
        );
        const normalized_ratio = if (avg_baseline > 0) avg_mse / avg_baseline else 1.0;
        std.debug.print(
            "Calib a2a: normalized_mse_ratio={d:.6} (avg_mse/avg_base, lower-better), relative_mse_reduction_pct={d:.2}%\n",
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
    log.info("convert", "MXFP8 calibration summary", .{
        .layers = quantized_layers,
        .elapsed_ms = calib_elapsed_ms,
        .tokens = total_calib_tokens,
        .tok_per_s = calib_tok_per_s,
        .iters = options.calib_iters,
        .nsamples = options.calib_nsamples,
        .seqlen = options.calib_seqlen,
        .batch_size = options.calib_batch_size,
        .nblocks = options.calib_nblocks,
    });

    if (calib_probe_only) return;

    if (!show_calib_progress) {
        progress.completeLine(0);
    }
    try tensor_builder.save(output_dir, "model.safetensors");
}

fn envFlagEnabled(name: []const u8) bool {
    const raw = std.posix.getenv(name) orelse return false;
    if (raw.len == 0) return true;
    if (std.mem.eql(u8, raw, "0")) return false;
    if (std.mem.eql(u8, raw, "false")) return false;
    if (std.mem.eql(u8, raw, "FALSE")) return false;
    if (std.mem.eql(u8, raw, "off")) return false;
    if (std.mem.eql(u8, raw, "OFF")) return false;
    return true;
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

fn isGpuCalibrationEnabled() bool {
    if (!envFlagEnabledDefault("TALU_CONVERT_CALIB_GPU", true)) return false;
    if (comptime has_metal_gpu_calib) {
        return compute.metal.isAvailable();
    }
    return false;
}

fn parseCalibrationProgressMode(raw: []const u8) ?CalibrationProgressMode {
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    if (trimmed.len == 0) return null;
    if (std.ascii.eqlIgnoreCase(trimmed, "block") or std.ascii.eqlIgnoreCase(trimmed, "blocks")) return .block;
    if (std.ascii.eqlIgnoreCase(trimmed, "layer") or std.ascii.eqlIgnoreCase(trimmed, "layers")) return .layer;
    return null;
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
        .fast => .clip,
        .balanced, .good, .best => .clip_search,
        .custom => .search,
    };
}

fn calibrationProgressModeFromEnv() CalibrationProgressMode {
    const raw = std.posix.getenv("TALU_CONVERT_CALIB_PROGRESS_UNIT") orelse return .block;
    return parseCalibrationProgressMode(raw) orelse .block;
}

fn estimateTargetQuantizedTensorCount(
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
        if (!shouldMxfp8Quantize(layout_map, tensor_name, source_tensor)) continue;
        if (calib_probe_only and layer_window.enabled()) {
            if (!layer_window.allows(extractLayerIndexFromTensorName(tensor_name))) continue;
        }
        total += 1;
    }
    return total;
}

fn emitLayerCalibrationProgress(
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
            "layer {d}: quantized {d} tensors | block_baseline_mse={d:.12} block_mse={d:.12} block_normalized_mse_ratio={d:.6} block_relative_mse_reduction_pct={d:.2}% | running_weighted_baseline_mse={d:.12} running_weighted_mse={d:.12} running_normalized_mse_ratio={d:.6} running_relative_mse_reduction_pct={d:.2}% best_running_normalized_mse_ratio={d:.6} | model_state_coverage_pct={d:.2}% model_state_normalized_mse_ratio={d:.6} model_state_relative_mse_reduction_pct={d:.2}% (lower-better)\n",
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
                    "block {d} ({s}): quantized {d} tensors | block_baseline_mse={d:.12} block_mse={d:.12} block_normalized_mse_ratio={d:.6} block_relative_mse_reduction_pct={d:.2}% | running_weighted_baseline_mse={d:.12} running_weighted_mse={d:.12} running_normalized_mse_ratio={d:.6} running_relative_mse_reduction_pct={d:.2}% best_running_normalized_mse_ratio={d:.6} | model_state_coverage_pct={d:.2}% model_state_normalized_mse_ratio={d:.6} model_state_relative_mse_reduction_pct={d:.2}% (lower-better)\n",
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
                    "block {d}: quantized {d} tensors | block_baseline_mse={d:.12} block_mse={d:.12} block_normalized_mse_ratio={d:.6} block_relative_mse_reduction_pct={d:.2}% | running_weighted_baseline_mse={d:.12} running_weighted_mse={d:.12} running_normalized_mse_ratio={d:.6} running_relative_mse_reduction_pct={d:.2}% best_running_normalized_mse_ratio={d:.6} | model_state_coverage_pct={d:.2}% model_state_normalized_mse_ratio={d:.6} model_state_relative_mse_reduction_pct={d:.2}% (lower-better)\n",
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

fn emitTensorCalibrationProgress(tensor_name: []const u8, calib_summary: CalibrationSummary) void {
    const abs_drop = if (calib_summary.best_mse <= calib_summary.baseline_mse)
        (calib_summary.baseline_mse - calib_summary.best_mse)
    else
        0.0;
    const improvement_pct = if (calib_summary.baseline_mse > 0 and calib_summary.best_mse <= calib_summary.baseline_mse)
        ((calib_summary.baseline_mse - calib_summary.best_mse) / calib_summary.baseline_mse) * 100.0
    else
        0.0;
    const normalized_ratio = if (calib_summary.baseline_mse > 0) calib_summary.best_mse / calib_summary.baseline_mse else 1.0;
    std.debug.print(
        "tensor {s}: loss iter 0: {d:.12} -> iter {d}: {d:.12}, base={d:.12}, abs_drop={d:.12}, normalized_mse_ratio={d:.6}, relative_mse_reduction_pct={d:.2}%, tensor_final={d:.12} (lower-better), sf={d:.6}, v={d:.6}\n",
        .{
            tensor_name,
            calib_summary.first_mse,
            calib_summary.best_iter,
            calib_summary.best_mse,
            calib_summary.baseline_mse,
            abs_drop,
            normalized_ratio,
            improvement_pct,
            calib_summary.best_mse,
            calib_summary.scale_factor,
            calib_summary.round_shift,
        },
    );
}

fn emitNonLayerCalibrationProgress(
    tensor_name: []const u8,
    running_avg_mse: f64,
    running_avg_base: f64,
    best_running_normalized_ratio: f64,
    quantized_tensors_so_far: usize,
    total_target_tensors: usize,
    calib_summary: CalibrationSummary,
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
        "quantized non-layer: {s} | block_baseline_mse={d:.12} block_mse={d:.12} block_normalized_mse_ratio={d:.6} block_relative_mse_reduction_pct={d:.2}% | running_weighted_baseline_mse={d:.12} running_weighted_mse={d:.12} running_normalized_mse_ratio={d:.6} running_relative_mse_reduction_pct={d:.2}% best_running_normalized_mse_ratio={d:.6} | model_state_coverage_pct={d:.2}% model_state_normalized_mse_ratio={d:.6} model_state_relative_mse_reduction_pct={d:.2}% (lower-better), sf={d:.6}, v={d:.6}\n",
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
    options: ConvertOptions,
    token_pool: ?[]const u32,
    block_input_cache: *BlockInputCache,
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

            _ = try quantizeMxfp8Tensor(allocator, source_tensors, builder, plan.output_name, fused.*, options, token_pool, block_input_cache);
            return true;
        },
    }
}

fn fetchDatasetRowsJson(allocator: std.mem.Allocator, offset: usize, length: usize) ![]u8 {
    const url = try std.fmt.allocPrint(
        allocator,
        "https://datasets-server.huggingface.co/rows?dataset=NeelNanda/pile-10k&config=default&split=train&offset={d}&length={d}",
        .{ offset, length },
    );
    defer allocator.free(url);
    return http.fetch(allocator, url, .{
        .user_agent = "talu-convert/1.0",
        .max_response_bytes = calibration_rows_max_bytes,
    });
}

fn fetchDatasetRowsJsonWithRetry(
    allocator: std.mem.Allocator,
    base_offset: usize,
    length: usize,
    seed: u64,
) ![]u8 {
    const total_rows: usize = 10_000;
    const max_attempts: usize = 6;
    var attempt: usize = 0;
    while (attempt < max_attempts) : (attempt += 1) {
        const jitter = @as(usize, @intCast((mix64(seed +% @as(u64, @intCast(attempt)) *% 0x9e3779b97f4a7c15) % 97)));
        const offset = (base_offset + jitter) % (total_rows - length);
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

const calibration_cache_magic = "TCALIB01";

fn calibrationCachePath(allocator: std.mem.Allocator, tokenizer_path: []const u8, options: ConvertOptions) ![]u8 {
    const cache_dir = if (std.posix.getenv("TALU_HOME")) |talu_home|
        try std.fs.path.join(allocator, &.{ talu_home, "cache", "convert", "calibration" })
    else blk: {
        const home = std.posix.getenv("HOME") orelse return error.NotFound;
        break :blk try std.fs.path.join(allocator, &.{ home, ".cache", "talu", "cache", "convert", "calibration" });
    };
    defer allocator.free(cache_dir);

    const key = try std.fmt.allocPrint(
        allocator,
        "mxfp8|{s}|{d}|{d}|{d}|{d}",
        .{ tokenizer_path, options.calib_seed, options.calib_nsamples, options.calib_seqlen, options.calib_iters },
    );
    defer allocator.free(key);
    const digest = std.hash.Wyhash.hash(0, key);
    const file_name = try std.fmt.allocPrint(allocator, "pile10k-{x}.tcal", .{digest});
    defer allocator.free(file_name);

    return std.fs.path.join(allocator, &.{ cache_dir, file_name });
}

fn tryLoadCalibrationTokenCache(allocator: std.mem.Allocator, cache_path: []const u8) !?[]u32 {
    var file = std.fs.cwd().openFile(cache_path, .{}) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    defer file.close();

    const bytes = try file.readToEndAlloc(allocator, 16 * 1024 * 1024);
    defer allocator.free(bytes);

    if (bytes.len < 12) return null;
    if (!std.mem.eql(u8, bytes[0..8], calibration_cache_magic)) return null;
    const count = std.mem.readInt(u32, bytes[8..][0..4], .little);
    const expected_len = 12 + @as(usize, @intCast(count)) * 4;
    if (bytes.len != expected_len) return null;

    const tokens = try allocator.alloc(u32, count);
    errdefer allocator.free(tokens);

    var offset: usize = 12;
    for (tokens) |*token| {
        token.* = std.mem.readInt(u32, bytes[offset..][0..4], .little);
        offset += 4;
    }
    return tokens;
}

fn storeCalibrationTokenCache(cache_path: []const u8, tokens: []const u32) void {
    const parent = std.fs.path.dirname(cache_path) orelse return;
    std.fs.cwd().makePath(parent) catch return;

    var file = std.fs.cwd().createFile(cache_path, .{ .truncate = true }) catch return;
    defer file.close();

    var header: [12]u8 = undefined;
    @memcpy(header[0..8], calibration_cache_magic[0..8]);
    const token_count_u32 = std.math.cast(u32, tokens.len) orelse return;
    std.mem.writeInt(u32, header[8..][0..4], token_count_u32, .little);
    file.writeAll(&header) catch return;

    var scratch: [4]u8 = undefined;
    for (tokens) |token| {
        std.mem.writeInt(u32, &scratch, token, .little);
        file.writeAll(&scratch) catch return;
    }
}

fn loadCalibrationTokenPool(allocator: std.mem.Allocator, tokenizer_path: []const u8, options: ConvertOptions) !?[]u32 {
    if (tokenizer_path.len == 0) return null;
    if (options.calib_iters == 0) return null;
    const strict_dataset_mode = options.profile != .fast and options.calib_iters > 0;

    const cache_path = calibrationCachePath(allocator, tokenizer_path, options) catch null;
    defer if (cache_path) |path| allocator.free(path);
    if (cache_path) |path| {
        if (tryLoadCalibrationTokenCache(allocator, path) catch null) |cached| {
            log.info("convert", "Loaded calibration token pool cache", .{
                .path = path,
                .tokens = cached.len,
                .dataset = "NeelNanda/pile-10k",
            });
            return cached;
        }
    }

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

    var consecutive_failures: usize = 0;
    var page: usize = 0;
    while (page < max_pages and tokens.items.len < target_tokens) : (page += 1) {
        const offset = (seed_offset + page * rows_per_page) % (total_rows - rows_per_page);
        const rows_json = fetchDatasetRowsJsonWithRetry(
            allocator,
            offset,
            rows_per_page,
            options.calib_seed +% @as(u64, @intCast(page)),
        ) catch |err| {
            consecutive_failures += 1;
            log.warn("convert", "Calibration rows fetch failed", .{
                .dataset = "NeelNanda/pile-10k",
                .offset = offset,
                .rows = rows_per_page,
                .err = @errorName(err),
            });
            if (strict_dataset_mode and consecutive_failures >= 6) return error.CalibrationDataUnavailable;
            if (consecutive_failures >= 6 and tokens.items.len == 0) return err;
            continue;
        };
        defer allocator.free(rows_json);
        const appended = appendTokenizedRows(allocator, &tokenizer, rows_json, &tokens, target_tokens) catch |err| {
            consecutive_failures += 1;
            log.warn("convert", "Calibration rows parse/tokenize failed", .{
                .dataset = "NeelNanda/pile-10k",
                .offset = offset,
                .rows = rows_per_page,
                .err = @errorName(err),
            });
            if (strict_dataset_mode and consecutive_failures >= 6) return error.CalibrationDataUnavailable;
            if (consecutive_failures >= 6 and tokens.items.len == 0) return err;
            continue;
        };
        if (appended == 0) {
            consecutive_failures += 1;
            if (consecutive_failures >= 6 and tokens.items.len > 0) break;
            continue;
        }
        consecutive_failures = 0;
    }

    if (strict_dataset_mode and tokens.items.len < requested) {
        log.warn("convert", "Calibration token pool coverage insufficient for strict mode", .{
            .required_tokens = requested,
            .loaded_tokens = tokens.items.len,
            .dataset = "NeelNanda/pile-10k",
        });
        return error.CalibrationDataUnavailable;
    }
    if (tokens.items.len == 0) return null;
    const owned = try tokens.toOwnedSlice(allocator);
    if (cache_path) |path| {
        storeCalibrationTokenCache(path, owned);
    }
    return owned;
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
/// Floor-log2 MXFP8 sets:
/// - shared_exp = floor(log2(absmax)) - emax, with emax=8 for E4M3
/// - scale = 2^(shared_exp)
/// - q = fp8(value/scale + v)
/// This helper emits e8m0 such that e8m0-127 == shared_exp.
fn absMaxToE8M0(absmax: f32) u8 {
    const shared_exp = blk: {
        if (absmax <= 0.0) break :blk -8.0;
        // emax for E4M3 is 8.
        const exp_floor = @floor(std.math.log2(absmax));
        break :blk std.math.clamp(exp_floor - 8.0, -127.0, 127.0);
    };
    const e8m0 = @as(i32, @intFromFloat(shared_exp + 127.0));
    return @intCast(std.math.clamp(e8m0, 0, 255));
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
    group_scale_factors: []const f32,
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
            const group_scale_factor = if (scale_idx < ctx.group_scale_factors.len)
                ctx.group_scale_factors[scale_idx]
            else
                1.0;
            const adjusted_absmax = absmax * group_scale_factor;
            ctx.e8m0_scales[row * scale_cols + scale_idx] = absMaxToE8M0(adjusted_absmax);
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
    group_round_shifts: []const f32,
};

const CalibrationSummary = struct {
    scale_factor: f32,
    round_shift: f32,
    best_mse: f64,
    baseline_mse: f64,
    first_mse: f64,
    best_step: usize,
    steps: usize,
    best_iter: usize,
};

const CalibrationParams = struct {
    summary: CalibrationSummary,
    group_scale_factors: []f32,
    group_round_shifts: []f32,

    fn deinit(self: CalibrationParams, allocator: std.mem.Allocator) void {
        allocator.free(self.group_scale_factors);
        allocator.free(self.group_round_shifts);
    }
};

const BlockInputMatrix = struct {
    values: []f32,
    sample_count: usize,
    cols: usize,

    fn deinit(self: BlockInputMatrix, allocator: std.mem.Allocator) void {
        allocator.free(self.values);
    }

    inline fn at(self: *const BlockInputMatrix, sample_idx: usize, col_idx: usize) f32 {
        return self.values[sample_idx * self.cols + col_idx];
    }
};

const ReplayTargetMatrix = struct {
    values: []f32,
    sample_count: usize,
    cols: usize,

    fn deinit(self: ReplayTargetMatrix, allocator: std.mem.Allocator) void {
        allocator.free(self.values);
    }

    inline fn at(self: *const ReplayTargetMatrix, sample_idx: usize, col_idx: usize) f32 {
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

const BlockInputCacheKey = struct {
    layer: u32,
    role: calibration_capture.ActivationRole,
    cols: usize,
    sample_count: usize,
};

const BlockInputCache = struct {
    allocator: std.mem.Allocator,
    source_tensors: ?*safetensors.UnifiedSafeTensors,
    token_pool: ?[]const u32,
    seed: u64,
    require_embedding_lookup: bool,
    activation_capture: ?*const calibration_capture.LayerActivationCache,
    map: std.AutoHashMap(BlockInputCacheKey, BlockInputMatrix),

    fn init(
        allocator: std.mem.Allocator,
        source_tensors: ?*safetensors.UnifiedSafeTensors,
        token_pool: ?[]const u32,
        seed: u64,
        require_embedding_lookup: bool,
        activation_capture: ?*const calibration_capture.LayerActivationCache,
    ) BlockInputCache {
        return .{
            .allocator = allocator,
            .source_tensors = source_tensors,
            .token_pool = token_pool,
            .seed = seed,
            .require_embedding_lookup = require_embedding_lookup,
            .activation_capture = activation_capture,
            .map = std.AutoHashMap(BlockInputCacheKey, BlockInputMatrix).init(allocator),
        };
    }

    fn deinit(self: *BlockInputCache) void {
        var it = self.map.valueIterator();
        while (it.next()) |entry| {
            entry.deinit(self.allocator);
        }
        self.map.deinit();
    }

    fn get(
        self: *BlockInputCache,
        layer_index: ?u32,
        role: calibration_capture.ActivationRole,
        cols: usize,
        sample_count: usize,
    ) !*const BlockInputMatrix {
        const key: BlockInputCacheKey = .{
            .layer = layer_index orelse std.math.maxInt(u32),
            .role = role,
            .cols = cols,
            .sample_count = sample_count,
        };
        if (self.map.getPtr(key)) |existing| return existing;
        if (self.activation_capture) |capture| {
            if (layer_index) |layer| {
                if (try calibration_capture.sampleLayerActivationsForRole(
                    self.allocator,
                    capture,
                    layer,
                    cols,
                    sample_count,
                    self.seed,
                    role,
                )) |sampled| {
                    const built = BlockInputMatrix{
                        .values = sampled.values,
                        .sample_count = sampled.sample_count,
                        .cols = sampled.cols,
                    };
                    try self.map.put(key, built);
                    return self.map.getPtr(key).?;
                }
            }
        }
        const built = try buildBlockInputMatrix(
            self.allocator,
            self.source_tensors,
            cols,
            sample_count,
            self.token_pool,
            self.seed,
            self.require_embedding_lookup,
        );
        try self.map.put(key, built);
        return self.map.getPtr(key).?;
    }
};

const EmbeddingInputLookup = struct {
    tensor: Tensor,
    input_dim: usize,
    vocab_dim: usize,
    transposed: bool,

    inline fn value(self: EmbeddingInputLookup, token: u32, feature_col: usize) ?f32 {
        if (self.input_dim == 0 or self.vocab_dim == 0) return null;

        const token_idx = @as(usize, token) % self.vocab_dim;
        const feature_idx = feature_col % self.input_dim;
        const flat_idx = if (!self.transposed)
            token_idx * self.input_dim + feature_idx
        else
            feature_idx * self.vocab_dim + token_idx;
        return tensorScalarToF32(self.tensor, flat_idx);
    }
};

fn tensorScalarToF32(t: Tensor, idx: usize) ?f32 {
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

fn findEmbeddingInputLookup(source_tensors: ?*safetensors.UnifiedSafeTensors) ?EmbeddingInputLookup {
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
    return @intCast(mix64(seed ^ (@as(u64, @intCast(sample_idx + 1)) *% 0xbf58476d1ce4e5b9)) & 0xffff);
}

noinline fn deterministicPoolIndex(seed: u64, sample_idx: usize, len: usize) usize {
    const base = @as(usize, @intCast(seed % len));
    return (base + sample_idx) % len;
}

inline fn tokenFallbackActivation(token: u32, sample_idx: usize, col_idx: usize, seed: u64) f32 {
    const mixed = mix64(seed ^
        (@as(u64, token) *% 0x9e3779b97f4a7c15) ^
        (@as(u64, @intCast(sample_idx + 1)) *% 0xbf58476d1ce4e5b9) ^
        (@as(u64, @intCast(col_idx + 1)) *% 0x94d049bb133111eb));
    const mantissa: u32 = @intCast(mixed & 0x007fffff);
    const raw: u32 = 0x3f000000 | mantissa;
    const unit: f32 = @bitCast(raw);
    return (unit - 0.75) * 2.0;
}

noinline fn buildBlockInputMatrix(
    allocator: std.mem.Allocator,
    source_tensors: ?*safetensors.UnifiedSafeTensors,
    cols: usize,
    sample_count: usize,
    token_pool: ?[]const u32,
    seed: u64,
    require_embedding_lookup: bool,
) !BlockInputMatrix {
    if (sample_count == 0) return error.CalibrationDataUnavailable;
    const total = cols * sample_count;
    const values = try allocator.alloc(f32, total);
    errdefer allocator.free(values);

    const lookup = findEmbeddingInputLookup(source_tensors);
    if (lookup == null and require_embedding_lookup) return error.CalibrationDataUnavailable;
    for (0..sample_count) |sample_idx| {
        const token = tokenFromPool(token_pool, sample_idx, seed);
        for (0..cols) |col| {
            const idx = sample_idx * cols + col;
            if (lookup) |emb| {
                if (emb.value(token, col)) |v0| {
                    if (col < emb.input_dim) {
                        values[idx] = v0;
                    } else {
                        const mixed_col = (col + sample_idx) % emb.input_dim;
                        const v1 = emb.value(token, mixed_col) orelse v0;
                        values[idx] = (v0 * 0.75) + (v1 * 0.25);
                    }
                    continue;
                }
            }
            values[idx] = tokenFallbackActivation(token, sample_idx, col, seed);
        }
    }

    return .{
        .values = values,
        .sample_count = sample_count,
        .cols = cols,
    };
}

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
            const round_shift = if (scale_idx < ctx.group_round_shifts.len)
                ctx.group_round_shifts[scale_idx]
            else
                0.0;

            for (row_values[col..col_end], row_fp8[col..col_end]) |v, *out| {
                out.* = dtype_mod.f32ToFp8E4M3(v * inv_scale + round_shift);
            }
        }
    }
}

const CalibrationEvalBudget = struct {
    row_samples: usize,
    input_samples: usize,
};

const CalibrationExperimentSettings = struct {
    clip_min: f32 = 0.5,
    clip_max: f32 = 2.0,
    shift_max: f32 = 1.0,
    adaptive_clip_floor: f32 = 0.55,

    fn isDefault(self: CalibrationExperimentSettings) bool {
        return self.clip_min == 0.5 and
            self.clip_max == 2.0 and
            self.shift_max == 1.0 and
            self.adaptive_clip_floor == 0.55;
    }
};

fn parseCalibrationFloatEnv(name: []const u8, fallback: f32, min_value: f32, max_value: f32) f32 {
    const raw = std.posix.getenv(name) orelse return fallback;
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    if (trimmed.len == 0) return fallback;
    const parsed = std.fmt.parseFloat(f32, trimmed) catch return fallback;
    return std.math.clamp(parsed, min_value, max_value);
}

fn calibrationExperimentSettingsFromEnv() CalibrationExperimentSettings {
    var settings: CalibrationExperimentSettings = .{};
    settings.clip_min = parseCalibrationFloatEnv("TALU_CONVERT_CALIB_CLIP_MIN", settings.clip_min, 0.01, 16.0);
    settings.clip_max = parseCalibrationFloatEnv("TALU_CONVERT_CALIB_CLIP_MAX", settings.clip_max, 0.01, 16.0);
    settings.shift_max = parseCalibrationFloatEnv("TALU_CONVERT_CALIB_SHIFT_MAX", settings.shift_max, 0.0, 4.0);
    settings.adaptive_clip_floor = parseCalibrationFloatEnv("TALU_CONVERT_CALIB_ADAPTIVE_CLIP_FLOOR", settings.adaptive_clip_floor, 0.0, 1.0);
    if (settings.clip_min > settings.clip_max) {
        const tmp = settings.clip_min;
        settings.clip_min = settings.clip_max;
        settings.clip_max = tmp;
    }
    return settings;
}

fn calibrationEvalBudget(rows: usize, cols: usize, options: ConvertOptions) CalibrationEvalBudget {
    const nsamples = @as(usize, @intCast(@max(options.calib_nsamples, 1)));
    const seqlen = @as(usize, @intCast(@max(options.calib_seqlen, 1)));
    const batch_size = @as(usize, @intCast(@max(options.calib_batch_size, 1)));
    const nblocks = @as(usize, @intCast(@max(options.calib_nblocks, 1)));
    const fast_budget_mode = blk: {
        if (options.profile == .fast) break :blk true;
        const raw = std.posix.getenv("TALU_CONVERT_CALIB_BUDGET_MODE") orelse break :blk false;
        if (std.ascii.eqlIgnoreCase(raw, "fast")) break :blk true;
        break :blk false;
    };

    const row_target = if (fast_budget_mode)
        nsamples
    else
        @max(@as(usize, 512), nsamples * 2);
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

inline fn mix64(value: u64) u64 {
    var z = value +% 0x9e3779b97f4a7c15;
    z = (z ^ (z >> 30)) *% 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) *% 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

inline fn averageSlice(values: []const f32, fallback: f32) f32 {
    if (values.len == 0) return fallback;
    var sum: f64 = 0.0;
    for (values) |v| sum += @as(f64, @floatCast(v));
    return @as(f32, @floatCast(sum / @as(f64, @floatFromInt(values.len))));
}

fn clipSearchCalibration(
    allocator: std.mem.Allocator,
    source_values: []align(1) const f32,
    cols: usize,
    row_cache: *const Mxfp8CalibrationRowCache,
    block_inputs: *const BlockInputMatrix,
    replay_targets: ?*const ReplayTargetMatrix,
    ref_outputs: ?[]const f32,
    current_scale_factors: []f32,
    current_round_shifts: []f32,
    baseline_err: f64,
    tuning: CalibrationExperimentSettings,
) struct { err: f64, step: usize } {
    // Global clip candidates tuned for fast/high-ROI quality gains.
    const clip_multipliers = [_]f32{ 1.00, 0.975, 0.95, 0.925, 0.90, 0.875, 0.85, 0.825, 0.80, 0.775, 0.75, 0.70, 0.65, 1.05, 1.10 };
    const round_shifts = [_]f32{ 0.0, -0.125, 0.125 };
    var best_err = baseline_err;
    var best_step: usize = 0;
    var best_mult: f32 = 1.0;
    var best_shift: f32 = 0.0;
    var eval_idx: usize = 0;

    for (clip_multipliers) |mult| {
        for (round_shifts) |shift| {
            eval_idx += 1;
            @memset(current_scale_factors, mult);
            @memset(current_round_shifts, shift);
            const candidate_err = evaluateBlockForwardCalibrationCandidate(
                allocator,
                source_values,
                cols,
                row_cache,
                block_inputs,
                current_scale_factors,
                current_round_shifts,
                replay_targets,
                ref_outputs,
            );
            if (candidate_err < best_err) {
                best_err = candidate_err;
                best_step = eval_idx;
                best_mult = mult;
                best_shift = shift;
            }
        }
    }

    // Local refinement around the best global candidate.
    const refine_mult_offsets = [_]f32{ -0.05, -0.025, 0.0, 0.025, 0.05 };
    const refine_shift_offsets = [_]f32{ -0.0625, 0.0, 0.0625 };
    for (refine_mult_offsets) |moff| {
        for (refine_shift_offsets) |soff| {
            eval_idx += 1;
            const mult = std.math.clamp(best_mult + moff, tuning.clip_min, tuning.clip_max);
            const shift = std.math.clamp(best_shift + soff, -tuning.shift_max, tuning.shift_max);
            @memset(current_scale_factors, mult);
            @memset(current_round_shifts, shift);
            const candidate_err = evaluateBlockForwardCalibrationCandidate(
                allocator,
                source_values,
                cols,
                row_cache,
                block_inputs,
                current_scale_factors,
                current_round_shifts,
                replay_targets,
                ref_outputs,
            );
            if (candidate_err < best_err) {
                best_err = candidate_err;
                best_step = eval_idx;
                best_mult = mult;
                best_shift = shift;
            }
        }
    }

    // Adaptive per-group clip heuristic:
    // use row-cache distribution (p90/p75 vs max) to reduce outlier-dominated groups
    // with only one additional objective evaluation.
    const row_samples = row_cache.sample_rows;
    const group_count = row_cache.group_count;
    const scratch = allocator.alloc(f32, row_samples) catch null;
    if (scratch) |tmp| {
        defer allocator.free(tmp);
        var g: usize = 0;
        while (g < group_count) : (g += 1) {
            var r: usize = 0;
            while (r < row_samples) : (r += 1) {
                tmp[r] = row_cache.group_absmax[r * group_count + g];
            }
            std.mem.sort(f32, tmp[0..row_samples], {}, std.sort.asc(f32));
            const max_v = tmp[row_samples - 1];
            const p90 = tmp[@min(row_samples - 1, (row_samples * 9) / 10)];
            const p75 = tmp[@min(row_samples - 1, (row_samples * 3) / 4)];
            const ratio90: f32 = if (max_v > 0) p90 / max_v else 1.0;
            const ratio75: f32 = if (max_v > 0) p75 / max_v else 1.0;
            const adaptive = std.math.clamp((ratio90 * 0.7) + (ratio75 * 0.3), tuning.adaptive_clip_floor, 1.0);
            current_scale_factors[g] = std.math.clamp(best_mult * adaptive, tuning.clip_min, tuning.clip_max);
        }
        @memset(current_round_shifts, best_shift);
        eval_idx += 1;
        const adaptive_err = evaluateBlockForwardCalibrationCandidate(
            allocator,
            source_values,
            cols,
            row_cache,
            block_inputs,
            current_scale_factors,
            current_round_shifts,
            replay_targets,
            ref_outputs,
        );
        if (adaptive_err < best_err) {
            best_err = adaptive_err;
            best_step = eval_idx;
        } else {
            @memset(current_scale_factors, best_mult);
        }
    } else {
        @memset(current_scale_factors, best_mult);
    }
    @memset(current_round_shifts, best_shift);
    return .{ .err = best_err, .step = best_step };
}

fn estimateBlockForwardCalibrationParameters(
    allocator: std.mem.Allocator,
    source_tensors: ?*safetensors.UnifiedSafeTensors,
    source_values: []align(1) const f32,
    rows: usize,
    cols: usize,
    layer_index: ?u32,
    activation_role: calibration_capture.ActivationRole,
    replay_points: ?ReplayPointCandidates,
    options: ConvertOptions,
    token_pool: ?[]const u32,
    block_input_cache: ?*BlockInputCache,
) !CalibrationParams {
    if (rows == 0 or cols == 0) {
        const empty_scale = try allocator.alloc(f32, 1);
        errdefer allocator.free(empty_scale);
        const empty_shift = try allocator.alloc(f32, 1);
        errdefer allocator.free(empty_shift);
        empty_scale[0] = 1.0;
        empty_shift[0] = 0.0;
        return .{
            .summary = .{
                .scale_factor = 1.0,
                .round_shift = 0.0,
                .best_mse = 0.0,
                .baseline_mse = 0.0,
                .first_mse = 0.0,
                .best_step = 0,
                .steps = 0,
                .best_iter = 0,
            },
            .group_scale_factors = empty_scale,
            .group_round_shifts = empty_shift,
        };
    }

    const eval_budget = calibrationEvalBudget(rows, cols, options);
    const tuning = calibrationExperimentSettingsFromEnv();
    const steps = @max(@as(usize, 1), @as(usize, @intCast(@max(options.calib_iters, 1))));
    const row_offset = if (rows > 0) @as(usize, @intCast(options.calib_seed % rows)) else 0;
    const row_stride = if (eval_budget.row_samples > 0)
        @max(@as(usize, 1), rows / eval_budget.row_samples)
    else
        1;
    var row_cache = try buildMxfp8CalibrationRowCache(
        allocator,
        source_values,
        rows,
        cols,
        eval_budget.row_samples,
        row_offset,
        row_stride,
    );
    defer row_cache.deinit(allocator);
    const group_count = (cols + group_size - 1) / group_size;
    var owned_block_inputs: ?BlockInputMatrix = null;
    defer if (owned_block_inputs) |*matrix| matrix.deinit(allocator);
    var replay_targets: ?ReplayTargetMatrix = null;
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
                                cols,
                                rows,
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
                                    .sample_count = sampled.sample_count,
                                    .cols = sampled.input_cols,
                                };
                                break :blk &owned_block_inputs.?;
                            }
                        }
                    }
                }
            }
            break :blk try cache.get(layer_index, activation_role, cols, eval_budget.input_samples);
        }
        owned_block_inputs = try buildBlockInputMatrix(
            allocator,
            source_tensors,
            cols,
            eval_budget.input_samples,
            token_pool,
            options.calib_seed,
            source_tensors != null and options.profile != .fast and options.calib_iters > 0,
        );
        break :blk &owned_block_inputs.?;
    };

    const ref_outputs = if (replay_targets == null and isGpuCalibrationEnabled())
        buildBlockForwardReferenceOutputsMetal(
            allocator,
            source_values,
            cols,
            &row_cache,
            block_inputs,
        )
    else
        null;
    defer if (ref_outputs) |buf| allocator.free(buf);

    var current_scale_factors = try allocator.alloc(f32, group_count);
    errdefer allocator.free(current_scale_factors);
    var current_round_shifts = try allocator.alloc(f32, group_count);
    errdefer allocator.free(current_round_shifts);
    @memset(current_scale_factors, 1.0);
    @memset(current_round_shifts, 0.0);

    const baseline_err = evaluateBlockForwardCalibrationCandidate(
        allocator,
        source_values,
        cols,
        &row_cache,
        block_inputs,
        current_scale_factors,
        current_round_shifts,
        if (replay_targets) |*targets| targets else null,
        ref_outputs,
    );
    const optimizer = calibrationOptimizerFromEnv(options.profile);

    if (group_count == 0) {
        return .{
            .summary = .{
                .scale_factor = averageSlice(current_scale_factors, 1.0),
                .round_shift = averageSlice(current_round_shifts, 0.0),
                .best_mse = baseline_err,
                .baseline_mse = baseline_err,
                .first_mse = baseline_err,
                .best_step = 0,
                .steps = 1,
                .best_iter = 0,
            },
            .group_scale_factors = current_scale_factors,
            .group_round_shifts = current_round_shifts,
        };
    }

    var current_err: f64 = baseline_err;
    var best_step: usize = 0;
    if (optimizer == .clip or optimizer == .clip_search) {
        const clip = clipSearchCalibration(
            allocator,
            source_values,
            cols,
            &row_cache,
            block_inputs,
            if (replay_targets) |*targets| targets else null,
            ref_outputs,
            current_scale_factors,
            current_round_shifts,
            baseline_err,
            tuning,
        );
        current_err = clip.err;
        best_step = clip.step;
    }

    if (steps == 1 or optimizer == .clip) {
        return .{
            .summary = .{
                .scale_factor = averageSlice(current_scale_factors, 1.0),
                .round_shift = averageSlice(current_round_shifts, 0.0),
                .best_mse = current_err,
                .baseline_mse = baseline_err,
                .first_mse = baseline_err,
                .best_step = best_step,
                .steps = 1,
                .best_iter = 0,
            },
            .group_scale_factors = current_scale_factors,
            .group_round_shifts = current_round_shifts,
        };
    }

    const best_scale_factors = try allocator.alloc(f32, group_count);
    errdefer allocator.free(best_scale_factors);
    const best_round_shifts = try allocator.alloc(f32, group_count);
    errdefer allocator.free(best_round_shifts);
    @memcpy(best_scale_factors, current_scale_factors);
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
    const block_prev_shift = try allocator.alloc(f32, max_update_block_size);
    defer allocator.free(block_prev_shift);
    const block_acc_scale = try allocator.alloc(f32, max_update_block_size);
    defer allocator.free(block_acc_scale);
    const block_acc_shift = try allocator.alloc(f32, max_update_block_size);
    defer allocator.free(block_acc_shift);
    const dir_scale_mem = try allocator.alloc(f32, group_count);
    defer allocator.free(dir_scale_mem);
    const dir_shift_mem = try allocator.alloc(f32, group_count);
    defer allocator.free(dir_shift_mem);
    for (0..group_count) |g| {
        const dseed = mix64(options.calib_seed ^ (@as(u64, @intCast(g + 1)) *% 0x9e3779b97f4a7c15));
        dir_scale_mem[g] = if ((dseed & 1) == 0) -1.0 else 1.0;
        dir_shift_mem[g] = if ((dseed & 2) == 0) -1.0 else 1.0;
    }

    var step_idx: usize = 1;
    while (step_idx < steps) : (step_idx += 1) {
        const progress = @as(f32, @floatFromInt(step_idx)) / @as(f32, @floatFromInt(steps));
        const anneal = @max(@as(f32, 0.05), 1.0 - progress);
        const base_lr = @max(@as(f32, 2e-3), @min(@as(f32, 0.3), 2.0 / @as(f32, @floatFromInt(@max(options.calib_iters, 1)))));
        const lr_scale = base_lr * anneal;
        const lr_shift = (base_lr * 0.35) * anneal;
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
            block_prev_shift[block_slot] = current_round_shifts[g];
        }

        const backtrack = [_]f32{ 1.0, 0.5, 0.25, 0.125 };
        var accepted_err = current_err;
        var accepted = false;
        var accepted_bt: usize = 0;
        var accepted_mode: usize = 0;
        var bt_idx: usize = 0;
        while (bt_idx < backtrack.len) : (bt_idx += 1) {
            const step_scale = lr_scale * backtrack[bt_idx];
            const step_shift = lr_shift * backtrack[bt_idx];
            var mode: usize = 0;
            while (mode < 4) : (mode += 1) {
                block_slot = 0;
                while (block_slot < update_block_size) : (block_slot += 1) {
                    const g = (block_start + block_slot) % group_count;
                    var dir_scale = dir_scale_mem[g];
                    var dir_shift = dir_shift_mem[g];
                    if (mode == 1) {
                        dir_scale = -dir_scale;
                        dir_shift = -dir_shift;
                    } else if (mode >= 2) {
                        const dseed = mix64(options.calib_seed ^
                            (@as(u64, @intCast(step_idx + 1)) *% 0x9e3779b97f4a7c15) ^
                            (@as(u64, @intCast(g + 1)) *% 0xbf58476d1ce4e5b9));
                        dir_scale = if ((dseed & 1) == 0) -1.0 else 1.0;
                        dir_shift = if ((dseed & 2) == 0) -1.0 else 1.0;
                        if (mode == 3) {
                            dir_scale = -dir_scale;
                            dir_shift = -dir_shift;
                        }
                    }
                    current_scale_factors[g] = std.math.clamp(block_prev_scale[block_slot] + dir_scale * step_scale, tuning.clip_min, tuning.clip_max);
                    current_round_shifts[g] = std.math.clamp(block_prev_shift[block_slot] + dir_shift * step_shift, -tuning.shift_max, tuning.shift_max);
                }

                const candidate_err = evaluateBlockForwardCalibrationCandidate(
                    allocator,
                    source_values,
                    cols,
                    &row_cache,
                    block_inputs,
                    current_scale_factors,
                    current_round_shifts,
                    if (replay_targets) |*targets| targets else null,
                    ref_outputs,
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
                        block_acc_shift[block_slot] = current_round_shifts[g];
                    }
                }
            }
        }

        if (!accepted) {
            block_slot = 0;
            while (block_slot < update_block_size) : (block_slot += 1) {
                const g = (block_start + block_slot) % group_count;
                current_scale_factors[g] = block_prev_scale[block_slot];
                current_round_shifts[g] = block_prev_shift[block_slot];
                dir_scale_mem[g] = -dir_scale_mem[g];
                dir_shift_mem[g] = -dir_shift_mem[g];
            }
        } else {
            block_slot = 0;
            while (block_slot < update_block_size) : (block_slot += 1) {
                const g = (block_start + block_slot) % group_count;
                current_scale_factors[g] = block_acc_scale[block_slot];
                current_round_shifts[g] = block_acc_shift[block_slot];
                if (accepted_mode == 1) {
                    dir_scale_mem[g] = -dir_scale_mem[g];
                    dir_shift_mem[g] = -dir_shift_mem[g];
                } else if (accepted_mode >= 2) {
                    const dseed = mix64(options.calib_seed ^
                        (@as(u64, @intCast(step_idx + 1)) *% 0x9e3779b97f4a7c15) ^
                        (@as(u64, @intCast(g + 1)) *% 0xbf58476d1ce4e5b9));
                    var next_scale: f32 = if ((dseed & 1) == 0) -1.0 else 1.0;
                    var next_shift: f32 = if ((dseed & 2) == 0) -1.0 else 1.0;
                    if (accepted_mode == 3) {
                        next_scale = -next_scale;
                        next_shift = -next_shift;
                    }
                    dir_scale_mem[g] = next_scale;
                    dir_shift_mem[g] = next_shift;
                }
            }

            const accepted_bt_idx = @min(accepted_bt, backtrack.len - 1);
            const refine_scale = (lr_scale * backtrack[accepted_bt_idx]) * 0.5;
            const refine_shift = (lr_shift * backtrack[accepted_bt_idx]) * 0.5;
            block_slot = 0;
            while (block_slot < update_block_size) : (block_slot += 1) {
                const g = (block_start + block_slot) % group_count;
                const dir_scale = dir_scale_mem[g];
                const dir_shift = dir_shift_mem[g];
                current_scale_factors[g] = std.math.clamp(block_acc_scale[block_slot] + dir_scale * refine_scale, tuning.clip_min, tuning.clip_max);
                current_round_shifts[g] = std.math.clamp(block_acc_shift[block_slot] + dir_shift * refine_shift, -tuning.shift_max, tuning.shift_max);
            }
            const refine_err = evaluateBlockForwardCalibrationCandidate(
                allocator,
                source_values,
                cols,
                &row_cache,
                block_inputs,
                current_scale_factors,
                current_round_shifts,
                if (replay_targets) |*targets| targets else null,
                ref_outputs,
            );
            if (refine_err < accepted_err) {
                accepted_err = refine_err;
            } else {
                block_slot = 0;
                while (block_slot < update_block_size) : (block_slot += 1) {
                    const g = (block_start + block_slot) % group_count;
                    current_scale_factors[g] = block_acc_scale[block_slot];
                    current_round_shifts[g] = block_acc_shift[block_slot];
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
            @memcpy(best_round_shifts, current_round_shifts);
        }
    }
    const best_iter = if (steps > 1)
        @divTrunc(best_step * @as(usize, @intCast(@max(options.calib_iters, 1) - 1)), steps - 1)
    else
        0;
    allocator.free(current_scale_factors);
    allocator.free(current_round_shifts);
    return .{
        .summary = .{
            .scale_factor = averageSlice(best_scale_factors, 1.0),
            .round_shift = averageSlice(best_round_shifts, 0.0),
            .best_mse = best_err,
            .baseline_mse = baseline_err,
            .first_mse = first_err,
            .best_step = best_step,
            .steps = steps,
            .best_iter = best_iter,
        },
        .group_scale_factors = best_scale_factors,
        .group_round_shifts = best_round_shifts,
    };
}

fn estimateBlockForwardCalibration(
    allocator: std.mem.Allocator,
    source_tensors: ?*safetensors.UnifiedSafeTensors,
    source_values: []align(1) const f32,
    rows: usize,
    cols: usize,
    options: ConvertOptions,
    token_pool: ?[]const u32,
) !CalibrationSummary {
    var params = try estimateBlockForwardCalibrationParameters(allocator, source_tensors, source_values, rows, cols, null, .generic, null, options, token_pool, null);
    defer params.deinit(allocator);
    return params.summary;
}

const BlockForwardEvalContext = struct {
    source_values: []align(1) const f32,
    cols: usize,
    row_cache: *const Mxfp8CalibrationRowCache,
    block_inputs: *const BlockInputMatrix,
    replay_targets: ?*const ReplayTargetMatrix,
    input_samples: usize,
    group_scale_factors: []const f32,
    group_round_shifts: []const f32,
    err_sum: f64 = 0.0,
    count: usize = 0,
    mutex: std.Thread.Mutex = .{},
};

const Mxfp8CalibrationRowCache = struct {
    row_indices: []usize,
    group_absmax: []f32,
    sample_rows: usize,
    group_count: usize,

    fn deinit(self: *Mxfp8CalibrationRowCache, allocator: std.mem.Allocator) void {
        allocator.free(self.row_indices);
        allocator.free(self.group_absmax);
        self.* = undefined;
    }
};

fn buildMxfp8CalibrationRowCache(
    allocator: std.mem.Allocator,
    source_values: []align(1) const f32,
    rows: usize,
    cols: usize,
    sample_rows_budget: usize,
    row_offset: usize,
    row_stride: usize,
) !Mxfp8CalibrationRowCache {
    const group_count = (cols + group_size - 1) / group_size;
    const row_indices = try allocator.alloc(usize, sample_rows_budget);
    errdefer allocator.free(row_indices);
    const group_absmax = try allocator.alloc(f32, sample_rows_budget * group_count);
    errdefer allocator.free(group_absmax);

    for (0..sample_rows_budget) |sampled_rows| {
        const row = (row_offset + sampled_rows * row_stride) % rows;
        row_indices[sampled_rows] = row;
        const row_values = source_values[row * cols .. (row + 1) * cols];
        for (0..group_count) |g| {
            const col_start = g * group_size;
            const col_end = @min(col_start + group_size, cols);
            var absmax: f32 = 0.0;
            for (row_values[col_start..col_end]) |v| {
                const a = @abs(v);
                if (a > absmax) absmax = a;
            }
            group_absmax[sampled_rows * group_count + g] = absmax;
        }
    }

    return .{
        .row_indices = row_indices,
        .group_absmax = group_absmax,
        .sample_rows = sample_rows_budget,
        .group_count = group_count,
    };
}

fn buildBlockForwardReferenceOutputsMetal(
    allocator: std.mem.Allocator,
    source_values: []align(1) const f32,
    cols: usize,
    row_cache: *const Mxfp8CalibrationRowCache,
    block_inputs: *const BlockInputMatrix,
) ?[]f32 {
    if (!compute.metal.isAvailable()) return null;
    if (row_cache.sample_rows == 0 or cols == 0) return null;
    if (block_inputs.cols != cols or block_inputs.sample_count == 0) return null;

    const graph = compute.metal.graph;
    const input_samples = block_inputs.sample_count;
    const sample_rows_budget = row_cache.sample_rows;
    const ref_weights = allocator.alloc(f32, cols * sample_rows_budget) catch return null;
    defer allocator.free(ref_weights);

    var sampled_rows: usize = 0;
    while (sampled_rows < sample_rows_budget) : (sampled_rows += 1) {
        const row = row_cache.row_indices[sampled_rows];
        const row_values = source_values[row * cols .. (row + 1) * cols];
        for (0..cols) |col| {
            const idx = col * sample_rows_budget + sampled_rows;
            ref_weights[idx] = row_values[col];
        }
    }

    const x_shape = [_]i64{ @intCast(input_samples), @intCast(cols) };
    const w_shape = [_]i64{ @intCast(cols), @intCast(sample_rows_budget) };
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

fn evaluateBlockForwardRows(sample_start: usize, sample_end: usize, ctx: *BlockForwardEvalContext) void {
    var local_err_sum: f64 = 0.0;
    var local_count: usize = 0;
    const cols = ctx.cols;
    const group_count = ctx.row_cache.group_count;

    var sampled_rows: usize = sample_start;
    while (sampled_rows < sample_end) : (sampled_rows += 1) {
        const row = ctx.row_cache.row_indices[sampled_rows];
        const row_values = ctx.source_values[row * cols .. (row + 1) * cols];
        const use_replay_targets = ctx.replay_targets != null;
        var ref_out = std.mem.zeroes([max_calibration_input_samples]f64);
        var dq_out = std.mem.zeroes([max_calibration_input_samples]f64);

        var g: usize = 0;
        while (g < group_count) : (g += 1) {
            const col_start = g * group_size;
            const col_end = @min(col_start + group_size, cols);
            const absmax = ctx.row_cache.group_absmax[sampled_rows * group_count + g];

            const group_scale_factor = if (g < ctx.group_scale_factors.len)
                ctx.group_scale_factors[g]
            else
                1.0;
            const round_shift = if (g < ctx.group_round_shifts.len)
                ctx.group_round_shifts[g]
            else
                0.0;
            const e8 = absMaxToE8M0(absmax * group_scale_factor);
            const scale = e8m0ToScale(e8);
            const inv_scale: f32 = if (scale > 0) 1.0 / scale else 0.0;

            for (col_start..col_end) |col| {
                const v = row_values[col];
                const q = dtype_mod.f32ToFp8E4M3(v * inv_scale + round_shift);
                const dq = dtype_mod.fp8e4m3ToF32(q) * scale;
                var sample_idx: usize = 0;
                while (sample_idx < ctx.input_samples) : (sample_idx += 1) {
                    const x = ctx.block_inputs.at(sample_idx, col);
                    if (!use_replay_targets) {
                        ref_out[sample_idx] += @as(f64, x) * @as(f64, v);
                    }
                    dq_out[sample_idx] += @as(f64, x) * @as(f64, dq);
                }
            }
        }

        var sample_idx: usize = 0;
        while (sample_idx < ctx.input_samples) : (sample_idx += 1) {
            const ref_value = if (ctx.replay_targets) |targets|
                @as(f64, targets.at(sample_idx, row))
            else
                ref_out[sample_idx];
            const diff = ref_value - dq_out[sample_idx];
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

fn evaluateBlockForwardCalibrationCandidate(
    allocator: std.mem.Allocator,
    source_values: []align(1) const f32,
    cols: usize,
    row_cache: *const Mxfp8CalibrationRowCache,
    block_inputs: *const BlockInputMatrix,
    group_scale_factors: []const f32,
    group_round_shifts: []const f32,
    replay_targets: ?*const ReplayTargetMatrix,
    ref_outputs: ?[]const f32,
) f64 {
    const group_count = row_cache.group_count;
    if (group_count == 0) return std.math.inf(f64);
    if (block_inputs.cols != cols or block_inputs.sample_count == 0) return std.math.inf(f64);
    if (row_cache.sample_rows == 0) return std.math.inf(f64);
    const input_samples = block_inputs.sample_count;
    if (input_samples > max_calibration_input_samples) return std.math.inf(f64);
    if (replay_targets) |targets| {
        if (targets.sample_count != input_samples) return std.math.inf(f64);
        for (row_cache.row_indices) |row| {
            if (row >= targets.cols) return std.math.inf(f64);
        }
    }

    if (isGpuCalibrationEnabled()) {
        if (evaluateBlockForwardCalibrationCandidateMetal(
            allocator,
            source_values,
            cols,
            row_cache,
            block_inputs,
            group_scale_factors,
            group_round_shifts,
            replay_targets,
            ref_outputs,
        )) |gpu_err| {
            return gpu_err;
        }
    }

    var eval_ctx = BlockForwardEvalContext{
        .source_values = source_values,
        .cols = cols,
        .row_cache = row_cache,
        .block_inputs = block_inputs,
        .replay_targets = replay_targets,
        .input_samples = input_samples,
        .group_scale_factors = group_scale_factors,
        .group_round_shifts = group_round_shifts,
    };
    const pool = parallel.global();
    pool.parallelForCompute(row_cache.sample_rows, evaluateBlockForwardRows, &eval_ctx);
    if (eval_ctx.count == 0) return std.math.inf(f64);
    return eval_ctx.err_sum / @as(f64, @floatFromInt(eval_ctx.count));
}

fn evaluateBlockForwardCalibrationCandidateMetal(
    allocator: std.mem.Allocator,
    source_values: []align(1) const f32,
    cols: usize,
    row_cache: *const Mxfp8CalibrationRowCache,
    block_inputs: *const BlockInputMatrix,
    group_scale_factors: []const f32,
    group_round_shifts: []const f32,
    replay_targets: ?*const ReplayTargetMatrix,
    ref_outputs: ?[]const f32,
) ?f64 {
    if (!compute.metal.isAvailable()) return null;
    const graph = compute.metal.graph;
    const input_samples = block_inputs.sample_count;
    const group_count = row_cache.group_count;
    if (group_count == 0 or input_samples == 0) return null;
    const sample_rows_budget = row_cache.sample_rows;

    const use_replay_targets = replay_targets != null;
    const use_cached_ref = !use_replay_targets and ref_outputs != null;
    var ref_weights_opt: ?[]f32 = null;
    if (!use_cached_ref and !use_replay_targets) {
        ref_weights_opt = allocator.alloc(f32, cols * sample_rows_budget) catch return null;
    }
    defer if (ref_weights_opt) |ref_weights| allocator.free(ref_weights);
    const dq_weights = allocator.alloc(f32, cols * sample_rows_budget) catch return null;
    defer allocator.free(dq_weights);

    var build_ctx = Mxfp8DqWeightBuildContext{
        .source_values = source_values,
        .cols = cols,
        .row_cache = row_cache,
        .group_scale_factors = group_scale_factors,
        .group_round_shifts = group_round_shifts,
        .ref_weights = ref_weights_opt,
        .dq_weights = dq_weights,
    };
    const pool = parallel.global();
    pool.parallelForCompute(sample_rows_budget, buildMxfp8DqWeightsRows, &build_ctx);

    const x_shape = [_]i64{ @intCast(input_samples), @intCast(cols) };
    const w_shape = [_]i64{ @intCast(cols), @intCast(sample_rows_budget) };
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
            var sample_idx: usize = 0;
            while (sample_idx < input_samples) : (sample_idx += 1) {
                const idx = sample_idx * sample_rows_budget + sampled_rows;
                const ref_val = targets.at(sample_idx, row);
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

const Mxfp8DqWeightBuildContext = struct {
    source_values: []align(1) const f32,
    cols: usize,
    row_cache: *const Mxfp8CalibrationRowCache,
    group_scale_factors: []const f32,
    group_round_shifts: []const f32,
    ref_weights: ?[]f32,
    dq_weights: []f32,
};

fn buildMxfp8DqWeightsRows(sample_start: usize, sample_end: usize, ctx: *Mxfp8DqWeightBuildContext) void {
    const group_count = (ctx.cols + group_size - 1) / group_size;
    var sampled_rows: usize = sample_start;
    while (sampled_rows < sample_end) : (sampled_rows += 1) {
        const row = ctx.row_cache.row_indices[sampled_rows];
        const row_values = ctx.source_values[row * ctx.cols .. (row + 1) * ctx.cols];

        var g: usize = 0;
        while (g < group_count) : (g += 1) {
            const col_start = g * group_size;
            const col_end = @min(col_start + group_size, ctx.cols);
            const absmax = ctx.row_cache.group_absmax[sampled_rows * group_count + g];

            const group_scale_factor = if (g < ctx.group_scale_factors.len)
                ctx.group_scale_factors[g]
            else
                1.0;
            const round_shift = if (g < ctx.group_round_shifts.len)
                ctx.group_round_shifts[g]
            else
                0.0;
            const e8 = absMaxToE8M0(absmax * group_scale_factor);
            const scale = e8m0ToScale(e8);
            const inv_scale: f32 = if (scale > 0) 1.0 / scale else 0.0;

            for (col_start..col_end) |col| {
                const v = row_values[col];
                const q = dtype_mod.f32ToFp8E4M3(v * inv_scale + round_shift);
                const dq = dtype_mod.fp8e4m3ToF32(q) * scale;
                const idx = col * ctx.row_cache.sample_rows + sampled_rows;
                if (ctx.ref_weights) |ref_weights| ref_weights[idx] = v;
                ctx.dq_weights[idx] = dq;
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
    options: ConvertOptions,
    token_pool: ?[]const u32,
    block_input_cache: *BlockInputCache,
) !CalibrationSummary {
    const rows: usize = @intCast(source_tensor.shape[0]);
    const cols: usize = @intCast(source_tensor.shape[1]);
    const layer_index = extractLayerIndexFromTensorName(tensor_name);

    // Convert source to F32
    const f32_source = try tensorToF32ForQuantization(allocator, source_tensors, tensor_name, source_tensor);
    defer f32_source.deinit(allocator);
    const source_values = f32_source.asF32Slice();
    const activation_role = activationRoleForTensorName(tensor_name);
    const replay_points = replayPointCandidatesForTensorName(tensor_name);
    var calib_params = try estimateBlockForwardCalibrationParameters(
        allocator,
        source_tensors,
        source_values,
        rows,
        cols,
        layer_index,
        activation_role,
        replay_points,
        options,
        token_pool,
        block_input_cache,
    );
    defer calib_params.deinit(allocator);
    const calib_summary = calib_params.summary;

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
        .group_scale_factors = calib_params.group_scale_factors,
    };
    pool.parallelFor(rows, computeGroupAbsmax, &absmax_ctx);

    // Pass 2: quantize to E4M3 using precomputed scales (parallel over rows)
    var quant_ctx = Mxfp8QuantizeContext{
        .source_data = source_values,
        .fp8_data = fp8_values,
        .e8m0_scales = e8m0_scales,
        .cols = cols,
        .scale_cols = scale_cols,
        .group_round_shifts = calib_params.group_round_shifts,
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
    return calib_summary;
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
    try output_buf.appendSlice(
        allocator,
        "\"quantization_config\":{\"quant_method\":\"mxfp8\",\"quant_type\":\"mxfp8\",\"fmt\":\"e4m3\",\"scale_fmt\":\"e8m0\",\"block_size\":32,\"quant_contract_version\":1}",
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
    // absmax=0 follows floor-log2 rule with emax=8:
    // shared_exp = -8, biased exponent = 119.
    try std.testing.expectEqual(@as(u8, 119), absMaxToE8M0(0.0));

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

test "absMaxToE8M0 roundtrip tracks floor-log2 scale band" {
    // Floor-log2 exponent can under-cover the group absmax near bin tops.
    // Validate we stay inside the expected power-of-two band.
    const test_values = [_]f32{ 0.001, 0.1, 1.0, 10.0, 100.0, 448.0, 1000.0 };
    for (test_values) |absmax| {
        const e8m0 = absMaxToE8M0(absmax);
        const scale = e8m0ToScale(e8m0);
        if (absmax <= 0.0) continue;
        const covered = scale * fp8_e4m3_max;
        try std.testing.expect(covered >= absmax * 0.5);
        try std.testing.expect(covered <= absmax * 2.01);
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
    const group_scale_factors = [_]f32{1.0} ** scale_cols;

    var ctx = GroupAbsmaxContext{
        .source_data = &source,
        .e8m0_scales = &e8m0_out,
        .rows = 1,
        .cols = cols,
        .scale_cols = scale_cols,
        .group_scale_factors = &group_scale_factors,
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
    const group_round_shifts = [_]f32{0.0};

    var fp8_out = [_]u8{0} ** cols;

    var ctx = Mxfp8QuantizeContext{
        .source_data = &source,
        .fp8_data = &fp8_out,
        .e8m0_scales = &e8m0_scales,
        .cols = cols,
        .scale_cols = scale_cols,
        .group_round_shifts = &group_round_shifts,
    };

    quantizeRowSlice(0, 1, &ctx);

    // Non-zero values should produce non-zero FP8
    try std.testing.expect(fp8_out[0] != 0);
    try std.testing.expect(fp8_out[1] != 0);
    // Zero values should stay zero
    try std.testing.expectEqual(@as(u8, 0), fp8_out[2]);
}

test "replayPointCandidatesForTensorName maps key tensor paths" {
    const q = replayPointCandidatesForTensorName("model.layers.0.self_attn.q_proj.weight").?;
    try std.testing.expectEqual(@as(usize, 1), q.count);
    try std.testing.expectEqual(xray.TracePoint.layer_attn_norm, q.pairs[0].input_point);
    try std.testing.expectEqual(xray.TracePoint.attn_q, q.pairs[0].output_point);

    const down = replayPointCandidatesForTensorName("model.layers.4.mlp.down_proj.weight").?;
    try std.testing.expectEqual(@as(usize, 2), down.count);
    try std.testing.expectEqual(xray.TracePoint.ffn_act_mix, down.pairs[0].input_point);
    try std.testing.expectEqual(xray.TracePoint.ffn_down, down.pairs[0].output_point);

    const linear = replayPointCandidatesForTensorName("model.layers.4.linear_attn.out_proj.weight").?;
    try std.testing.expectEqual(@as(usize, 2), linear.count);
    try std.testing.expectEqual(xray.TracePoint.gdelta_norm, linear.pairs[0].input_point);
    try std.testing.expectEqual(xray.TracePoint.gdelta_out, linear.pairs[0].output_point);

    try std.testing.expect(replayPointCandidatesForTensorName("model.layers.0.self_attn.o_proj.weight") == null);
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
    const group_scale_factors = [_]f32{1.0} ** scale_cols;
    var ctx1 = GroupAbsmaxContext{
        .source_data = &source,
        .e8m0_scales = &e8m0_scales,
        .rows = 1,
        .cols = cols,
        .scale_cols = scale_cols,
        .group_scale_factors = &group_scale_factors,
    };
    computeGroupAbsmax(0, 1, &ctx1);

    // Pass 2: quantize
    var fp8_out = [_]u8{0} ** cols;
    const group_round_shifts = [_]f32{0.0} ** scale_cols;
    var ctx2 = Mxfp8QuantizeContext{
        .source_data = &source,
        .fp8_data = &fp8_out,
        .e8m0_scales = &e8m0_scales,
        .cols = cols,
        .scale_cols = scale_cols,
        .group_round_shifts = &group_round_shifts,
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

test "estimateBlockForwardCalibration is deterministic for same calibration seed" {
    var values: [96]f32 = undefined;
    for (&values, 0..) |*v, i| {
        v.* = @sin(@as(f32, @floatFromInt(i)));
    }

    const options = ConvertOptions{
        .profile = .custom,
        .calib_iters = 17,
        .calib_nsamples = 8,
        .calib_seqlen = 512,
        .calib_batch_size = 2,
        .calib_nblocks = 2,
        .calib_seed = 1234,
    };

    const a = try estimateBlockForwardCalibration(std.testing.allocator, null, &values, 6, 16, options, null);
    const b = try estimateBlockForwardCalibration(std.testing.allocator, null, &values, 6, 16, options, null);

    try std.testing.expectEqual(a.steps, b.steps);
    try std.testing.expectEqual(a.best_step, b.best_step);
    try std.testing.expectEqual(a.best_iter, b.best_iter);
    try std.testing.expectApproxEqRel(a.best_mse, b.best_mse, 1e-12);
    try std.testing.expectApproxEqRel(a.baseline_mse, b.baseline_mse, 1e-12);
    try std.testing.expectApproxEqRel(a.first_mse, b.first_mse, 1e-12);
}

test "estimateBlockForwardCalibration honors calibration iteration budget" {
    var values: [128]f32 = undefined;
    for (&values, 0..) |*v, i| {
        v.* = (@as(f32, @floatFromInt(i % 11)) - 5.0) * 0.75;
    }

    const options = ConvertOptions{
        .profile = .custom,
        .calib_iters = 33,
        .calib_nsamples = 16,
        .calib_seqlen = 1024,
        .calib_batch_size = 1,
        .calib_nblocks = 1,
        .calib_seed = 42,
    };

    const summary = try estimateBlockForwardCalibration(std.testing.allocator, null, &values, 8, 16, options, null);
    try std.testing.expectEqual(@as(usize, 33), summary.steps);
    try std.testing.expect(summary.best_iter < 33);
    try std.testing.expect(summary.best_step < summary.steps);
}

test "estimateBlockForwardCalibration fast profile runs clip-only calibration" {
    var values: [64]f32 = undefined;
    for (&values, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i));

    const options = ConvertOptions{
        .profile = .fast,
        .calib_iters = 100,
        .calib_nsamples = 16,
        .calib_seqlen = 2048,
        .calib_batch_size = 1,
        .calib_nblocks = 1,
        .calib_seed = 7,
    };
    const summary = try estimateBlockForwardCalibration(std.testing.allocator, null, &values, 4, 16, options, null);
    try std.testing.expectEqual(@as(usize, 1), summary.steps);
    try std.testing.expectEqual(@as(usize, 0), summary.best_iter);
    try std.testing.expect(summary.best_mse <= summary.baseline_mse);
}

test "calibrationEvalBudget scales input samples with calib_nblocks" {
    const n1 = calibrationEvalBudget(32, 64, .{
        .profile = .custom,
        .calib_nsamples = 32,
        .calib_seqlen = 4,
        .calib_batch_size = 1,
        .calib_nblocks = 1,
    });
    const n4 = calibrationEvalBudget(32, 64, .{
        .profile = .custom,
        .calib_nsamples = 32,
        .calib_seqlen = 4,
        .calib_batch_size = 1,
        .calib_nblocks = 4,
    });
    try std.testing.expectEqual(@as(usize, 4), n1.input_samples);
    try std.testing.expectEqual(@as(usize, 16), n4.input_samples);
    try std.testing.expectEqual(n1.row_samples, n4.row_samples);

    const capped = calibrationEvalBudget(32, 64, .{
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

test "parseCalibrationProgressMode accepts aliases and rejects unknown values" {
    try std.testing.expectEqual(@as(?CalibrationProgressMode, .block), parseCalibrationProgressMode("block"));
    try std.testing.expectEqual(@as(?CalibrationProgressMode, .block), parseCalibrationProgressMode("BLOCKS"));
    try std.testing.expectEqual(@as(?CalibrationProgressMode, .layer), parseCalibrationProgressMode("layer"));
    try std.testing.expectEqual(@as(?CalibrationProgressMode, .layer), parseCalibrationProgressMode("  Layers \n"));
    try std.testing.expectEqual(@as(?CalibrationProgressMode, null), parseCalibrationProgressMode("unit"));
    try std.testing.expectEqual(@as(?CalibrationProgressMode, null), parseCalibrationProgressMode(""));
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
    try std.testing.expectEqual(CalibrationOptimizer.clip, calibrationOptimizerFromEnv(.fast));
    try std.testing.expectEqual(CalibrationOptimizer.clip_search, calibrationOptimizerFromEnv(.balanced));
    try std.testing.expectEqual(CalibrationOptimizer.clip_search, calibrationOptimizerFromEnv(.good));
    try std.testing.expectEqual(CalibrationOptimizer.clip_search, calibrationOptimizerFromEnv(.best));
    try std.testing.expectEqual(CalibrationOptimizer.search, calibrationOptimizerFromEnv(.custom));
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
