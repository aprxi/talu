//! NVFP4 Converter
//!
//! Converts dense weight tensors into modelopt-compatible NVFP4 tensor
//! layout and rewrites config metadata accordingly.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const log = @import("log_pkg");
const grouped_affine = @import("grouped_affine.zig");
const gaf_paths = @import("gaf_paths.zig");
const convert = @import("root.zig");
const calibration_capture = @import("calibration_capture.zig");
const router_local = @import("../router/local.zig");
const parallel = @import("compute_pkg").parallel;
const compute = @import("compute_pkg");
const json = @import("io_pkg").json;
const repository = @import("io_pkg").repository.root;
const safetensors = @import("io_pkg").safetensors.root;
const config_loader = @import("models_pkg").config;
const models_registry = @import("models_pkg").registry;
const op_types = @import("models_pkg").op_types;
const tensor = @import("tensor_pkg");
const dtype = @import("dtype_pkg");

const nvfp4_group_size: usize = 16;
const global_scale_sample_limit: usize = 8192;
const activation_sample_cap: usize = 64;
const activation_importance_min_weight: f32 = 1e-6;
const small_model_preserve_threshold_params: u64 = 8_000_000_000;
const small_model_default_threshold_params: u64 = small_model_preserve_threshold_params;
const mixed_preserve_layers_pct_default_good: u32 = 10;
const mixed_preserve_score_sample_blocks_default: usize = 256;
const nvfp4_forward_eval_row_cap: usize = 128;
const nvfp4_forward_eval_max_scales: usize = 8;
const nvfp4_parallel_min_blocks: usize = 32 * 1024;
const nvfp4_block_scale_cache_max_blocks: usize = 16 * 1024 * 1024;
const nvfp4_kl_sample_block_cap: usize = 2048;
const nvfp4_kl_laplace_alpha: f64 = 1e-6;
const nvfp4_update_round_slots: u32 = 96;
const nvfp4_canary_context_tokens: usize = 64;
const nvfp4_quality_tokens_per_sample: usize = 32;
const nvfp4_quality_sample_count: usize = 16;
const nvfp4_canary_target_tokens: usize = nvfp4_quality_tokens_per_sample * nvfp4_quality_sample_count;
const nvfp4_quality_probe_source_dense_bytes_limit: u64 = 24 * 1024 * 1024 * 1024;
const nvfp4_forward_metric_tensor_sample_mod: u64 = 1;
const nvfp4_forward_metric_input_cap: usize = 8;
const nvfp4_forward_metric_row_cap: usize = 16;
const nvfp4_update_accept_nll_tolerance: f64 = 1e-9;
// Metal packing has non-trivial launch/setup overhead and is currently slower
// than CPU packing on our measured NVFP4 workloads. Keep it disabled by
// default and allow explicit opt-in through TALU_NVFP4_METAL_MIN_BLOCKS.
const nvfp4_metal_pack_min_blocks_default: usize = 1_000_000_000;
const has_metal_gpu_eval = build_options.enable_metal and builtin.os.tag == .macos;
const fp4_codebook = [_]f32{
    0.0,  0.5,  1.0,  1.5,
    2.0,  3.0,  4.0,  6.0,
    -0.0, -0.5, -1.0, -1.5,
    -2.0, -3.0, -4.0, -6.0,
};

const Nvfp4ClipScaleCandidate = struct {
    clip_multiplier: f32,
    scale_refine_multiplier: f32,
};

const Nvfp4PreserveFormat = enum {
    bf16,
    mxfp8,
};

extern fn mlx_runtime_binary_dir() ?[*:0]const u8;

pub const modelIdFromOutputPath = grouped_affine.modelIdFromOutputPath;

pub const ConvertOptions = struct {
    output_dir: []const u8 = "models",
    destination: ?[]const u8 = null,
    output_suffix: ?[]const u8 = null,
    force: bool = false,
    update: bool = false,
    update_round: u32 = 0,
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
    if (options.update) return error.InvalidArgument;

    var model_bundle = try repository.resolve(allocator, input_path, .{});
    defer model_bundle.deinit();
    const model_config = try config_loader.loadConfig(allocator, model_bundle.config_path());

    const output_dir_path = if (options.destination) |dest|
        try allocator.dupe(u8, dest)
    else blk: {
        const suffix = options.output_suffix orelse "NVFP4";
        break :blk try gaf_paths.generateOutputName(
            allocator,
            input_path,
            suffix,
            options.output_dir,
        );
    };
    errdefer allocator.free(output_dir_path);
    var source_tensors = try safetensors.UnifiedSafeTensors.load(allocator, model_bundle.weights_path() orelse return error.WeightsNotFound);
    defer source_tensors.deinit();

    const n_layers = std.math.cast(usize, model_config.n_layers) orelse return error.InvalidConfig;
    var layout_map = try resolveNvfp4LayoutMap(allocator, model_bundle.config_path(), n_layers);
    defer if (layout_map) |*map| map.deinit();

    if (options.update) {
        std.fs.cwd().access(output_dir_path, .{}) catch |err| switch (err) {
            error.FileNotFound => return error.FileNotFound,
            else => return err,
        };
        var existing_dir = try std.fs.cwd().openDir(output_dir_path, .{});
        defer existing_dir.close();
        if (!isCompleteConversionOutput(existing_dir)) return error.InvalidConfig;

        const previous_state = try readNvfp4UpdateState(allocator, output_dir_path);
        const next_round = previous_state.round + 1;
        const canary_seed = options.calib_seed;
        var update_options = options;
        update_options.update_round = next_round;
        update_options.calib_seed +%= @as(u64, next_round);

        const output_weights_path = try std.fs.path.join(allocator, &.{ output_dir_path, "model.safetensors" });
        defer allocator.free(output_weights_path);
        const backup_weights_path = try std.fs.path.join(allocator, &.{ output_dir_path, "model.safetensors.nvfp4.update.prev" });
        defer allocator.free(backup_weights_path);
        std.fs.cwd().deleteFile(backup_weights_path) catch |err| switch (err) {
            error.FileNotFound => {},
            else => return err,
        };

        var canary_reference = try captureNvfp4CanaryReference(
            allocator,
            output_dir_path,
            canary_seed,
        );
        defer canary_reference.deinit(allocator);
        const start_canary = canary_reference.evalMetrics();

        std.debug.print(
            "NVFP4 update round: {d} (prev={d}, seed={d}, profile={s})\n",
            .{ next_round, previous_state.round, update_options.calib_seed, nvfp4ProfileName(update_options.profile) },
        );
        std.debug.print(
            "NVFP4 quality start: global={d:.2}% mean={d:.2}% kl={e} | probe_nll={e} ppl={d:.6} tokens={d}\n",
            .{
                previous_state.global_improvement_pct,
                previous_state.mean_improvement_pct,
                if (std.math.isFinite(previous_state.kl_divergence) and previous_state.kl_divergence >= 0.0) previous_state.kl_divergence else 0.0,
                start_canary.mean_nll,
                start_canary.ppl,
                start_canary.scored_tokens,
            },
        );

        try std.fs.cwd().rename(output_weights_path, backup_weights_path);
        var backup_staged = true;
        errdefer if (backup_staged) {
            std.fs.cwd().deleteFile(output_weights_path) catch {};
            std.fs.cwd().rename(backup_weights_path, output_weights_path) catch {};
        };

        const quality_summary = try augmentWithPackedNvfp4Companions(
            allocator,
            &source_tensors,
            output_dir_path,
            model_bundle.dir,
            model_bundle.tokenizer_path(),
            model_config.tie_word_embeddings,
            if (layout_map) |*map| map else null,
            update_options.progress,
            update_options,
            backup_weights_path,
        );
        const end_canary_result = try evaluateNvfp4CanaryCandidate(
            allocator,
            output_dir_path,
            canary_seed,
            &canary_reference,
        );
        const end_canary = end_canary_result.eval;
        const canary_delta_nll = end_canary.mean_nll - start_canary.mean_nll;
        const accepted = canary_delta_nll <= nvfp4_update_accept_nll_tolerance;

        if (!accepted) {
            std.fs.cwd().deleteFile(output_weights_path) catch {};
            try std.fs.cwd().rename(backup_weights_path, output_weights_path);
            backup_staged = false;
            std.debug.print(
                "NVFP4 update rejected: probe_nll delta={s}{e} (ppl {d:.6}->{d:.6}, kld={e})\n",
                .{
                    if (canary_delta_nll >= 0.0) "+" else "",
                    canary_delta_nll,
                    start_canary.ppl,
                    end_canary.ppl,
                    end_canary.mean_kld,
                },
            );
            try appendNvfp4QualityReport(allocator, output_dir_path, .{
                .round = next_round,
                .profile = update_options.profile,
                .accepted = false,
                .internal_global = quality_summary.globalImprovementPct(),
                .internal_mean = quality_summary.meanImprovementPct(),
                .internal_kl = quality_summary.meanKlDivergence(),
                .canary_start_nll = start_canary.mean_nll,
                .canary_end_nll = end_canary.mean_nll,
                .canary_start_ppl = start_canary.ppl,
                .canary_end_ppl = end_canary.ppl,
                .canary_mean_kld = end_canary.mean_kld,
                .scored_tokens = end_canary.scored_tokens,
            });
            try writeNvfp4UpdateState(allocator, output_dir_path, .{
                .round = previous_state.round,
                .global_improvement_pct = previous_state.global_improvement_pct,
                .mean_improvement_pct = previous_state.mean_improvement_pct,
                .kl_divergence = previous_state.kl_divergence,
                .canary_mean_nll = start_canary.mean_nll,
                .canary_ppl = start_canary.ppl,
                .canary_mean_kld = start_canary.mean_kld,
                .canary_scored_tokens = @intCast(start_canary.scored_tokens),
            });
            return output_dir_path;
        }

        try std.fs.cwd().deleteFile(backup_weights_path);
        backup_staged = false;

        const end_global = quality_summary.globalImprovementPct();
        const end_mean = quality_summary.meanImprovementPct();
        const end_kl = quality_summary.meanKlDivergence();
        std.debug.print(
            "NVFP4 quality end: global={d:.2}% (delta={s}{d:.2}%) mean={d:.2}% (delta={s}{d:.2}%) kl={e} | probe_nll={e} (delta={s}{e}) ppl={d:.6} kld={e}\n",
            .{
                end_global,
                if ((end_global - previous_state.global_improvement_pct) >= 0.0) "+" else "",
                end_global - previous_state.global_improvement_pct,
                end_mean,
                if ((end_mean - previous_state.mean_improvement_pct) >= 0.0) "+" else "",
                end_mean - previous_state.mean_improvement_pct,
                end_kl,
                end_canary.mean_nll,
                if (canary_delta_nll >= 0.0) "+" else "",
                canary_delta_nll,
                end_canary.ppl,
                end_canary.mean_kld,
            },
        );
        try rewriteConfigToCanonical(allocator, output_dir_path);
        try writeNvfp4UpdateState(allocator, output_dir_path, .{
            .round = next_round,
            .global_improvement_pct = end_global,
            .mean_improvement_pct = end_mean,
            .kl_divergence = end_kl,
            .canary_mean_nll = end_canary.mean_nll,
            .canary_ppl = end_canary.ppl,
            .canary_mean_kld = end_canary.mean_kld,
            .canary_scored_tokens = @intCast(end_canary.scored_tokens),
        });
        try appendNvfp4QualityReport(allocator, output_dir_path, .{
            .round = next_round,
            .profile = update_options.profile,
            .accepted = true,
            .internal_global = end_global,
            .internal_mean = end_mean,
            .internal_kl = end_kl,
            .canary_start_nll = start_canary.mean_nll,
            .canary_end_nll = end_canary.mean_nll,
            .canary_start_ppl = start_canary.ppl,
            .canary_end_ppl = end_canary.ppl,
            .canary_mean_kld = end_canary.mean_kld,
            .scored_tokens = end_canary.scored_tokens,
        });
        return output_dir_path;
    }

    const output_tmp_path = try std.fmt.allocPrint(allocator, "{s}.tmp", .{output_dir_path});
    defer allocator.free(output_tmp_path);

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

    var keep_output = false;
    errdefer if (!keep_output) std.fs.cwd().deleteTree(output_tmp_path) catch {};
    var output_dir = try gaf_paths.GAFModelDir.init(allocator, output_tmp_path);
    defer output_dir.deinit();

    const quality_summary = try augmentWithPackedNvfp4Companions(
        allocator,
        &source_tensors,
        output_tmp_path,
        model_bundle.dir,
        model_bundle.tokenizer_path(),
        model_config.tie_word_embeddings,
        if (layout_map) |*map| map else null,
        options.progress,
        options,
        null,
    );
    try convert.copyConfigWithGAFQuantization(allocator, model_bundle.config_path(), output_tmp_path, null);
    try rewriteConfigToCanonical(allocator, output_tmp_path);
    try convert.copyModelAssets(allocator, model_bundle.dir, output_tmp_path);

    const model_name = convert.model_card.extractModelName(input_path);
    const base_model_id = convert.model_card.extractBaseModelId(input_path);
    convert.model_card.writeModelCard(allocator, output_tmp_path, model_name, base_model_id, .nvfp4) catch |err| {
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
    switch (resolveNvfp4QualityProbeDecision(quality_summary)) {
        .run => {
            if (evaluateNvfp4CanaryAgainstSource(
                allocator,
                input_path,
                output_dir_path,
                options.calib_seed,
            )) |comparison| {
                printNvfp4CanarySummary(comparison, options, quality_summary);
                writeNvfp4CanaryReport(allocator, output_dir_path, comparison) catch |err| {
                    log.warn("convert", "NVFP4 quality-probe report write failed", .{ .err = @errorName(err) });
                };
                appendNvfp4ProbeHistory(allocator, output_dir_path, options, comparison) catch |err| {
                    log.warn("convert", "NVFP4 probe-history write failed", .{ .err = @errorName(err) });
                };
            } else |err| {
                log.warn("convert", "NVFP4 quality-probe comparison unavailable", .{ .err = @errorName(err) });
            }
        },
        .skip_disabled => {
            log.info("convert", "NVFP4 quality-probe skipped", .{ .reason = "disabled_by_env" });
        },
        .skip_source_too_large => {
            const source_gib: f64 = @as(f64, @floatFromInt(quality_summary.source_dense_weight_bytes)) / @as(f64, 1024.0 * 1024.0 * 1024.0);
            const limit_gib: f64 = @as(f64, @floatFromInt(nvfp4_quality_probe_source_dense_bytes_limit)) / @as(f64, 1024.0 * 1024.0 * 1024.0);
            log.info("convert", "NVFP4 quality-probe skipped", .{
                .reason = "source_dense_too_large",
                .source_gib = source_gib,
                .limit_gib = limit_gib,
            });
        },
    }
    return output_dir_path;
}

fn augmentWithPackedNvfp4Companions(
    allocator: std.mem.Allocator,
    st: *safetensors.UnifiedSafeTensors,
    output_path: []const u8,
    model_path: []const u8,
    tokenizer_path: []const u8,
    tie_embeddings: bool,
    layout_map: ?*const convert.WeightLayoutMap,
    progress: grouped_affine.ProgressContext,
    options: ConvertOptions,
    update_existing_weights_path: ?[]const u8,
) !Nvfp4QualitySummary {
    const overall_start_ns = std.time.nanoTimestamp();
    const output_weights_path = try std.fs.path.join(allocator, &.{ output_path, "model.safetensors" });
    defer allocator.free(output_weights_path);
    var existing_st: ?safetensors.UnifiedSafeTensors = null;
    defer if (existing_st) |*prev| prev.deinit();
    if (options.update and options.update_round > 0) {
        const previous_weights_path = update_existing_weights_path orelse output_weights_path;
        existing_st = try safetensors.UnifiedSafeTensors.load(allocator, previous_weights_path);
    }

    const names = try st.tensorNames(allocator);
    defer allocator.free(names);
    sortNvfp4TensorNames(names);

    var converted_weight_bases = std.StringHashMap(void).init(allocator);
    defer {
        var key_iter = converted_weight_bases.keyIterator();
        while (key_iter.next()) |key| allocator.free(key.*);
        converted_weight_bases.deinit();
    }

    var target_weights: usize = 0;
    var dense_candidate_weights: usize = 0;
    var skipped_small_model_policy: usize = 0;
    var skipped_mixed_preserve: usize = 0;
    var target_blocks_total: u64 = 0;
    var dense_weight_params_total: u64 = 0;
    var language_dense_weight_params_total: u64 = 0;
    var source_dense_weight_bytes_total: u64 = 0;
    var unsupported_moe_tensor_name: ?[]const u8 = null;
    var unsupported_moe_tensor_n_dims: i32 = 0;
    var max_layer_index: ?u32 = null;
    for (names) |name| {
        if (convert.shouldSkipForTiedEmbeddingsByName(layout_map, name, tie_embeddings)) continue;
        const weight = st.getTensor(name, null) catch |err| {
            log.warn("convert", "NVFP4 scan tensor lookup failed", .{
                .tensor = name,
                .err = @errorName(err),
            });
            return err;
        };
        if (unsupported_moe_tensor_name == null and isUnsupportedNvfp4MoeExpertTensor(name, weight)) {
            unsupported_moe_tensor_name = name;
            unsupported_moe_tensor_n_dims = weight.n_dims;
        }
        if (weight.data_size > 0) {
            const tensor_bytes: u64 = @intCast(weight.data_size);
            source_dense_weight_bytes_total = std.math.add(u64, source_dense_weight_bytes_total, tensor_bytes) catch std.math.maxInt(u64);
        }
        if (weight.n_dims == 2 and weight.shape[0] > 0 and weight.shape[1] > 0 and
            (weight.dtype == .f32 or weight.dtype == .f16 or weight.dtype == .bf16))
        {
            const rows: u64 = @intCast(weight.shape[0]);
            const cols: u64 = @intCast(weight.shape[1]);
            const tensor_params = std.math.mul(u64, rows, cols) catch std.math.maxInt(u64);
            dense_weight_params_total = std.math.add(u64, dense_weight_params_total, tensor_params) catch std.math.maxInt(u64);
            if (std.mem.indexOf(u8, name, "language_model.") != null) {
                language_dense_weight_params_total = std.math.add(u64, language_dense_weight_params_total, tensor_params) catch std.math.maxInt(u64);
            }
        }
        if (extractLayerIndexFromTensorName(name)) |layer| {
            if (max_layer_index == null or layer > max_layer_index.?) max_layer_index = layer;
        }
    }
    if (unsupported_moe_tensor_name) |tensor_name| {
        log.warn("convert", "NVFP4 conversion does not support fused MoE expert tensors in native NVFP4 yet", .{
            .tensor = tensor_name,
            .n_dims = unsupported_moe_tensor_n_dims,
            .hint = "use TQ4 for this model family until NVFP4 MoE expert packing/runtime support lands",
        });
        return error.UnsupportedModelArchitecture;
    }

    const small_model_default_scope = isNvfp4SmallModelDefaultScope(
        dense_weight_params_total,
        language_dense_weight_params_total,
    );
    const lm_head_quantized = resolveNvfp4LmHeadQuantized(options.profile, small_model_default_scope);
    const small_model_preserve_enabled = resolveNvfp4SmallModelPreserveEnabled(options.profile, small_model_default_scope);
    const mixed_preserve_default_pct = defaultNvfp4MixedPreserveLayersPct(options.profile, small_model_default_scope);
    const mixed_preserve_blocks_override = resolveNvfp4MixedPreserveBlocksOverride();
    const preserve_format = resolveNvfp4PreserveFormat();

    const small_model_policy = makeSmallModelPreservePolicy(
        small_model_preserve_enabled,
        dense_weight_params_total,
        max_layer_index,
    );
    const mixed_preserve_layers = try buildMixedPreserveLayerList(
        allocator,
        st,
        names,
        options.profile,
        lm_head_quantized,
        mixed_preserve_blocks_override,
        mixed_preserve_default_pct,
    );
    defer allocator.free(mixed_preserve_layers);
    if (small_model_policy.enabled) {
        log.info("convert", "NVFP4 small-model preserve policy enabled", .{
            .total_dense_params = dense_weight_params_total,
            .threshold = small_model_preserve_threshold_params,
            .last_layer = small_model_policy.last_layer_index orelse 0,
        });
    }
    if (mixed_preserve_layers.len > 0) {
        log.info("convert", "NVFP4 mixed preserve layers selected", .{
            .count = mixed_preserve_layers.len,
            .first = mixed_preserve_layers[0],
            .last = mixed_preserve_layers[mixed_preserve_layers.len - 1],
            .default_pct = mixed_preserve_default_pct,
        });
    }
    if (mixed_preserve_blocks_override) |blocks| {
        std.debug.print("NVFP4 mixed preserve blocks: selected={d} (manual={d})\n", .{
            mixed_preserve_layers.len,
            blocks,
        });
    } else {
        std.debug.print("NVFP4 mixed preserve blocks: selected={d} (auto={d}% budget)\n", .{
            mixed_preserve_layers.len,
            mixed_preserve_default_pct,
        });
    }

    for (names) |name| {
        if (convert.shouldSkipForTiedEmbeddingsByName(layout_map, name, tie_embeddings)) continue;
        const weight = st.getTensor(name, null) catch |err| {
            log.warn("convert", "NVFP4 scan tensor lookup failed", .{
                .tensor = name,
                .err = @errorName(err),
            });
            return err;
        };
        const convertible = shouldConvertDenseWeightWithLmHead(name, weight, options.profile, lm_head_quantized);
        if (!convertible) continue;
        dense_candidate_weights += 1;
        const is_dense_2d = weight.n_dims == 2;
        if (is_dense_2d and shouldPreserveWeightBySmallModelPolicy(name, small_model_policy)) {
            skipped_small_model_policy += 1;
            continue;
        }
        if (is_dense_2d and shouldPreserveWeightByLayerList(name, mixed_preserve_layers)) {
            skipped_mixed_preserve += 1;
            continue;
        }
        target_weights += 1;
        const should_retune = shouldRetuneTensorInUpdate(name, options.update_round);
        const use_advanced_search = shouldUseAdvancedSearchForTensor(
            options.profile,
            options.update_round,
        );
        const tensor_work = if (should_retune)
            nvfp4TensorBlockWork(weight, options.profile, options.update_round, use_advanced_search)
        else
            0;
        target_blocks_total = std.math.add(u64, target_blocks_total, tensor_work) catch std.math.maxInt(u64);

        const base = convertedNvfp4BaseName(name);
        const owned_base = try allocator.dupe(u8, base);
        errdefer allocator.free(owned_base);
        try converted_weight_bases.put(owned_base, {});
    }
    std.debug.print(
        "NVFP4 profile behavior: mode={s} lm_head_q={s} small_model_preserve={s} mixed_blocks={d} preserve_format={s}\n",
        .{
            nvfp4ProfileName(options.profile),
            if (lm_head_quantized) "on" else "off",
            if (small_model_policy.enabled) "on" else "off",
            mixed_preserve_layers.len,
            nvfp4PreserveFormatName(preserve_format),
        },
    );
    std.debug.print(
        "NVFP4 packing plan: dense_candidates={d} converted={d} skipped_small={d} skipped_mixed={d}\n",
        .{
            dense_candidate_weights,
            target_weights,
            skipped_small_model_policy,
            skipped_mixed_preserve,
        },
    );
    if (target_weights == 0) return Nvfp4QualitySummary{
        .profile = options.profile,
        .lm_head_quantized = lm_head_quantized,
        .small_model_preserve_enabled = small_model_policy.enabled,
        .preserve_blocks_selected = std.math.cast(u32, mixed_preserve_layers.len) orelse std.math.maxInt(u32),
        .source_dense_weight_bytes = source_dense_weight_bytes_total,
    };

    emitNvfp4CalibHeader(
        options,
        mixed_preserve_layers.len,
        mixed_preserve_blocks_override,
        lm_head_quantized,
        small_model_policy.enabled,
        mixed_preserve_default_pct,
    );
    var activation_cache = try captureActivationCacheBestEffort(
        allocator,
        model_path,
        tokenizer_path,
        options,
    );
    defer if (activation_cache) |*cache| cache.deinit();
    const activation_sample_count = resolveActivationSampleCount(options);
    progress.addLine(1, "Packing NVFP4", target_weights, null, "weights");
    errdefer progress.completeLine(1);
    progress.addLine(2, "Packing blocks", target_blocks_total, null, "blocks");
    errdefer progress.completeLine(2);

    var output_specs = std.ArrayListUnmanaged(OutputTensorSpec){};
    defer deinitOutputSpecs(allocator, &output_specs);

    for (names) |name| {
        if (convert.shouldSkipForTiedEmbeddingsByName(layout_map, name, tie_embeddings)) continue;
        const weight = st.getTensor(name, null) catch |err| {
            log.warn("convert", "NVFP4 output-spec tensor lookup failed", .{
                .tensor = name,
                .err = @errorName(err),
            });
            return err;
        };
        const convertible = shouldConvertDenseWeightWithLmHead(name, weight, options.profile, lm_head_quantized);
        const allow_preserve = weight.n_dims == 2;
        if (convertible and
            !(allow_preserve and shouldPreserveWeightBySmallModelPolicy(name, small_model_policy)) and
            !(allow_preserve and shouldPreserveWeightByLayerList(name, mixed_preserve_layers)))
        {
            const base = convertedNvfp4BaseName(name);
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

            continue;
        }
        if (allow_preserve and convertible and
            (shouldPreserveWeightBySmallModelPolicy(name, small_model_policy) or
                shouldPreserveWeightByLayerList(name, mixed_preserve_layers)))
        {
            switch (preserve_format) {
                .bf16 => {
                    try output_specs.append(allocator, .{
                        .kind = .preserved_bf16_weight,
                        .name = name,
                        .source_weight_name = name,
                        .owns_name = false,
                    });
                },
                .mxfp8 => {
                    const base = name[0 .. name.len - ".weight".len];
                    const block_scale_name = try std.fmt.allocPrint(allocator, "{s}.weight_block_scale", .{base});
                    errdefer allocator.free(block_scale_name);

                    try output_specs.append(allocator, .{
                        .kind = .preserved_mxfp8_weight,
                        .name = name,
                        .source_weight_name = name,
                        .owns_name = false,
                    });
                    try output_specs.append(allocator, .{
                        .kind = .preserved_mxfp8_block_scale,
                        .name = block_scale_name,
                        .source_weight_name = name,
                        .owns_name = true,
                    });
                },
            }
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

    // For untied models with dense lm_head, emit a pre-oriented BF16 RHS
    // companion so strict NVFP4 mmap loads can avoid dense transpose/copy.
    if (!tie_embeddings and !lm_head_quantized) {
        if (findUntiedLmHeadWeightName(names)) |lm_head_weight_name| {
            const rhs_name = try std.fmt.allocPrint(allocator, "{s}_rhs", .{lm_head_weight_name});
            errdefer allocator.free(rhs_name);
            try output_specs.append(allocator, .{
                .kind = .dense_bf16_rhs_weight,
                .name = rhs_name,
                .source_weight_name = lm_head_weight_name,
                .owns_name = true,
            });
        }
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

    const convert_threads = resolveNvfp4ConvertThreads();
    var convert_pool = try parallel.ThreadPool.create(allocator, convert_threads);
    defer convert_pool.deinit();
    const can_use_metal_pack = try shouldUseNvfp4MetalPack(allocator, options.profile);
    const write_start_ns = std.time.nanoTimestamp();

    writeStreamedSafetensorsHeader(allocator, out_file, st, output_specs.items) catch |err| {
        log.warn("convert", "NVFP4 streamed header write failed", .{ .err = @errorName(err) });
        return err;
    };
    var quality_summary = writeStreamedSafetensorsData(
        allocator,
        out_file,
        st,
        if (existing_st) |*prev| prev else null,
        output_specs.items,
        options.profile,
        lm_head_quantized,
        options.update_round,
        can_use_metal_pack,
        if (activation_cache) |*cache| cache else null,
        activation_sample_count,
        options.calib_seed,
        convert_pool,
        progress,
        target_weights,
        target_blocks_total,
    ) catch |err| {
        log.warn("convert", "NVFP4 streamed data write failed", .{ .err = @errorName(err) });
        return err;
    };
    quality_summary.preserve_blocks_selected = std.math.cast(u32, mixed_preserve_layers.len) orelse std.math.maxInt(u32);
    quality_summary.lm_head_quantized = lm_head_quantized;
    quality_summary.small_model_preserve_enabled = small_model_policy.enabled;
    quality_summary.source_dense_weight_bytes = source_dense_weight_bytes_total;
    const write_done_ns = std.time.nanoTimestamp();
    // Keep internal reconstruction summary out of default CLI output.
    // Use canary metrics (nll/ppl/kld/logit_mse) as objective tuning signals.
    progress.completeLine(2);
    const finalize_line_id: u8 = 3;
    progress.addLine(finalize_line_id, "Finalizing NVFP4", 3, null, "steps");
    errdefer progress.completeLine(finalize_line_id);

    progress.updateLine(finalize_line_id, 1, "Flushing safetensors to disk");
    try out_file.sync();
    out_file.close();

    progress.updateLine(finalize_line_id, 2, "Replacing model.safetensors");
    std.fs.cwd().rename(tmp_weights_path, output_weights_path) catch |err| switch (err) {
        error.PathAlreadyExists => {
            try std.fs.cwd().deleteFile(output_weights_path);
            try std.fs.cwd().rename(tmp_weights_path, output_weights_path);
        },
        else => return err,
    };

    progress.updateLine(finalize_line_id, 3, "Auditing packed tensors");
    auditPackedNvfp4Output(allocator, output_weights_path, &converted_weight_bases) catch |err| {
        log.warn("convert", "NVFP4 pack audit failed", .{ .err = @errorName(err) });
        return err;
    };
    progress.completeLine(finalize_line_id);
    progress.completeLine(1);

    if (shouldPrintNvfp4Timing()) {
        const done_ns = std.time.nanoTimestamp();
        const prepare_s: f64 = @floatFromInt(write_start_ns - overall_start_ns);
        const pack_s: f64 = @floatFromInt(write_done_ns - write_start_ns);
        const finalize_s: f64 = @floatFromInt(done_ns - write_done_ns);
        const total_s: f64 = @floatFromInt(done_ns - overall_start_ns);
        std.debug.print(
            "NVFP4 timings: prepare={d:.3}s pack={d:.3}s finalize={d:.3}s total={d:.3}s\n",
            .{
                prepare_s / @as(f64, std.time.ns_per_s),
                pack_s / @as(f64, std.time.ns_per_s),
                finalize_s / @as(f64, std.time.ns_per_s),
                total_s / @as(f64, std.time.ns_per_s),
            },
        );
    }
    return quality_summary;
}

fn resolveNvfp4ConvertThreads() usize {
    if (std.posix.getenv("TALU_CONVERT_THREADS")) |raw| {
        const parsed = std.fmt.parseInt(usize, raw, 10) catch null;
        if (parsed) |value| if (value > 0) return value;
    }
    if (std.posix.getenv("THREADS")) |raw| {
        const parsed = std.fmt.parseInt(usize, raw, 10) catch null;
        if (parsed) |value| if (value > 0) return value;
    }
    return std.Thread.getCpuCount() catch 1;
}

fn resolveNvfp4BlockScaleCacheMaxBlocks() usize {
    if (std.posix.getenv("TALU_NVFP4_BLOCK_SCALE_CACHE_MAX_BLOCKS")) |raw| {
        const parsed = std.fmt.parseInt(usize, raw, 10) catch null;
        if (parsed) |value| if (value > 0) return value;
    }
    return nvfp4_block_scale_cache_max_blocks;
}

fn isNvfp4SmallModelDefaultScope(
    dense_weight_params_total: u64,
    language_dense_weight_params_total: u64,
) bool {
    const params_basis = if (language_dense_weight_params_total > 0)
        language_dense_weight_params_total
    else
        dense_weight_params_total;
    return params_basis > 0 and params_basis <= small_model_default_threshold_params;
}

fn defaultNvfp4MixedPreserveLayersPct(
    profile: @TypeOf((grouped_affine.ConvertOptions{}).profile),
    small_model_default_scope: bool,
) u32 {
    return switch (profile) {
        .good, .best => mixed_preserve_layers_pct_default_good,
        .custom => if (small_model_default_scope) mixed_preserve_layers_pct_default_good else 0,
    };
}

fn mixedPreserveScoreSampleBlocks(profile: @TypeOf((grouped_affine.ConvertOptions{}).profile)) usize {
    _ = profile;
    return mixed_preserve_score_sample_blocks_default;
}

fn resolveNvfp4MixedPreserveBlocksOverride() ?u32 {
    const raw = std.posix.getenv("TALU_NVFP4_MIXED_PRESERVE_BLOCKS") orelse return null;
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    if (trimmed.len == 0) return null;
    const parsed = std.fmt.parseInt(u32, trimmed, 10) catch {
        log.warn("convert", "Invalid TALU_NVFP4_MIXED_PRESERVE_BLOCKS; ignoring override", .{
            .value = trimmed,
        });
        return null;
    };
    return parsed;
}

fn resolveNvfp4PreserveFormat() Nvfp4PreserveFormat {
    const raw = std.posix.getenv("TALU_NVFP4_PRESERVE_FORMAT") orelse return .bf16;
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    if (trimmed.len == 0) return .bf16;
    if (std.ascii.eqlIgnoreCase(trimmed, "bf16")) return .bf16;
    if (std.ascii.eqlIgnoreCase(trimmed, "mxfp8")) return .mxfp8;
    log.warn("convert", "Invalid TALU_NVFP4_PRESERVE_FORMAT; using bf16", .{
        .value = trimmed,
    });
    return .bf16;
}

fn nvfp4PreserveFormatName(format: Nvfp4PreserveFormat) []const u8 {
    return switch (format) {
        .bf16 => "bf16",
        .mxfp8 => "mxfp8",
    };
}

fn parseOptionalBoolEnv(name: []const u8) ?bool {
    const raw = std.posix.getenv(name) orelse return null;
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    if (trimmed.len == 0) return null;
    if (std.ascii.eqlIgnoreCase(trimmed, "1")) return true;
    if (std.ascii.eqlIgnoreCase(trimmed, "true")) return true;
    if (std.ascii.eqlIgnoreCase(trimmed, "yes")) return true;
    if (std.ascii.eqlIgnoreCase(trimmed, "on")) return true;
    if (std.ascii.eqlIgnoreCase(trimmed, "0")) return false;
    if (std.ascii.eqlIgnoreCase(trimmed, "false")) return false;
    if (std.ascii.eqlIgnoreCase(trimmed, "no")) return false;
    if (std.ascii.eqlIgnoreCase(trimmed, "off")) return false;
    return null;
}

const Nvfp4ReplayPolicy = enum {
    proxy_only,
    capture_required,
};

fn parseNvfp4ReplayPolicyEnv(raw: []const u8) ?Nvfp4ReplayPolicy {
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    if (trimmed.len == 0) return null;
    if (std.ascii.eqlIgnoreCase(trimmed, "weighted") or
        std.ascii.eqlIgnoreCase(trimmed, "proxy") or
        std.ascii.eqlIgnoreCase(trimmed, "proxy_only") or
        std.ascii.eqlIgnoreCase(trimmed, "weight_only"))
    {
        return .proxy_only;
    }
    if (std.ascii.eqlIgnoreCase(trimmed, "xray") or
        std.ascii.eqlIgnoreCase(trimmed, "capture") or
        std.ascii.eqlIgnoreCase(trimmed, "capture_required"))
    {
        return .capture_required;
    }
    return null;
}

fn resolveNvfp4ReplayPolicy(profile: @TypeOf((grouped_affine.ConvertOptions{}).profile)) Nvfp4ReplayPolicy {
    const default_policy: Nvfp4ReplayPolicy = switch (profile) {
        .good, .best, .custom => .proxy_only,
    };
    const raw = std.posix.getenv("TALU_NVFP4_REPLAY_POLICY") orelse return default_policy;
    const parsed = parseNvfp4ReplayPolicyEnv(raw) orelse {
        log.warn("convert", "Invalid TALU_NVFP4_REPLAY_POLICY; using default", .{
            .value = raw,
        });
        return default_policy;
    };
    return parsed;
}

fn nvfp4ReplayPolicyLabel(policy: Nvfp4ReplayPolicy) []const u8 {
    return switch (policy) {
        .proxy_only => "weighted",
        .capture_required => "xray",
    };
}

fn resolveNvfp4LmHeadQuantized(
    profile: @TypeOf((grouped_affine.ConvertOptions{}).profile),
    small_model_default_scope: bool,
) bool {
    const default_enabled = switch (profile) {
        .good, .best, .custom => !small_model_default_scope,
    };
    return parseOptionalBoolEnv("TALU_NVFP4_LM_HEAD_Q") orelse default_enabled;
}

fn resolveNvfp4SmallModelPreserveEnabled(
    profile: @TypeOf((grouped_affine.ConvertOptions{}).profile),
    small_model_default_scope: bool,
) bool {
    const default_enabled = switch (profile) {
        .good, .best => true,
        .custom => small_model_default_scope,
    };
    return parseOptionalBoolEnv("TALU_NVFP4_SMALL_MODEL_PRESERVE") orelse default_enabled;
}

fn parsePositiveF32Env(name: []const u8) ?f32 {
    const raw = std.posix.getenv(name) orelse return null;
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    if (trimmed.len == 0) return null;
    const parsed = std.fmt.parseFloat(f32, trimmed) catch return null;
    if (!(parsed > 0.0) or !std.math.isFinite(parsed)) return null;
    return parsed;
}

fn resolveNvfp4CustomClipMultiplier(profile: @TypeOf((grouped_affine.ConvertOptions{}).profile)) f32 {
    if (profile != .custom) return 1.0;
    return parsePositiveF32Env("TALU_NVFP4_CUSTOM_CLIP_MULT") orelse 1.0;
}

fn resolveNvfp4CustomScaleRefineMultiplier(profile: @TypeOf((grouped_affine.ConvertOptions{}).profile)) f32 {
    if (profile != .custom) return 1.0;
    return parsePositiveF32Env("TALU_NVFP4_CUSTOM_SCALE_REFINE_MULT") orelse 1.0;
}

fn resolveNvfp4MetalPackMinBlocks() usize {
    const raw = std.posix.getenv("TALU_NVFP4_METAL_MIN_BLOCKS") orelse return nvfp4_metal_pack_min_blocks_default;
    const parsed = std.fmt.parseInt(usize, raw, 10) catch return nvfp4_metal_pack_min_blocks_default;
    return parsed;
}

fn shouldPrintNvfp4Timing() bool {
    return envFlagEnabled("TALU_NVFP4_TIMING");
}

const Nvfp4QualityProbeDecision = enum {
    run,
    skip_disabled,
    skip_source_too_large,
};

fn shouldAutoSkipNvfp4QualityProbeBySourceSize(source_dense_weight_bytes: u64) bool {
    return source_dense_weight_bytes > nvfp4_quality_probe_source_dense_bytes_limit;
}

fn resolveNvfp4QualityProbeDecision(summary: Nvfp4QualitySummary) Nvfp4QualityProbeDecision {
    if (parseOptionalBoolEnv("TALU_NVFP4_QUALITY_PROBE")) |enabled| {
        return if (enabled) .run else .skip_disabled;
    }
    if (shouldAutoSkipNvfp4QualityProbeBySourceSize(summary.source_dense_weight_bytes)) {
        return .skip_source_too_large;
    }
    return .run;
}

fn emitNvfp4CalibHeader(
    options: ConvertOptions,
    preserve_blocks_selected: usize,
    preserve_blocks_override: ?u32,
    lm_head_quantized: bool,
    small_model_preserve: bool,
    mixed_preserve_default_pct: u32,
) void {
    const uses_search_grid = useAdvancedNvfp4Search(options.profile, options.update_round);
    const clip_candidates = if (uses_search_grid)
        clipMultipliersForProfile(options.profile, options.update_round).len
    else
        @as(usize, 1);
    const scale_refine_candidates = if (uses_search_grid)
        globalScaleRefineMultipliersForProfile(options.profile, options.update_round).len
    else
        @as(usize, 1);
    const replay_policy = resolveNvfp4ReplayPolicy(options.profile);
    const capture_backend = resolveNvfp4CaptureBackendSelection();
    const uses_activation_capture = replay_policy == .capture_required and options.calib_iters > 0;
    const preserve_format = resolveNvfp4PreserveFormat();
    const clip_mult = resolveNvfp4CustomClipMultiplier(options.profile);
    const scale_refine_mult = resolveNvfp4CustomScaleRefineMultiplier(options.profile);

    std.debug.print("Calib block loss minimization\n", .{});
    std.debug.print(
        "Calib mode: optimizer={s}, replay={s}\n",
        .{
            nvfp4OptimizerName(options.profile),
            nvfp4ReplayPolicyLabel(replay_policy),
        },
    );
    std.debug.print(
        "Calib mixed preserve blocks: selected={d} ({s})\n",
        .{
            preserve_blocks_selected,
            if (preserve_blocks_override != null) "manual" else "auto10pct",
        },
    );
    std.debug.print(
        "Calib preserve format: {s}\n",
        .{nvfp4PreserveFormatName(preserve_format)},
    );
    if (preserve_blocks_override) |blocks| {
        std.debug.print(
            "Calib mixed preserve manual override: {d}\n",
            .{blocks},
        );
    } else {
        std.debug.print(
            "Calib mixed preserve auto budget: {d}% layer-equivalent\n",
            .{mixed_preserve_default_pct},
        );
    }
    if (uses_activation_capture) {
        std.debug.print("Calib capture backend: {s}\n", .{switch (capture_backend) {
            .cpu => "cpu",
            .metal => "metal",
            .cuda => "cuda",
            .auto => "auto",
        }});
    }
    std.debug.print(
        "Calib effective path: search_grid={s}, clip_candidates={d}, scale_refine_candidates={d}, activation_capture={s}\n",
        .{
            if (uses_search_grid) "on" else "off",
            clip_candidates,
            scale_refine_candidates,
            if (uses_activation_capture) "on" else "off",
        },
    );
    std.debug.print(
        "Calib opts: --opts replay={s},preserve_blocks={d},preserve_format={s},lm_head_q={d},small_model_preserve={d},clip_mult={d:.3},scale_refine_mult={d:.3}\n",
        .{
            nvfp4ReplayPolicyLabel(replay_policy),
            preserve_blocks_selected,
            nvfp4PreserveFormatName(preserve_format),
            @intFromBool(lm_head_quantized),
            @intFromBool(small_model_preserve),
            clip_mult,
            scale_refine_mult,
        },
    );
}

fn nvfp4OptimizerName(profile: @TypeOf((grouped_affine.ConvertOptions{}).profile)) []const u8 {
    return switch (profile) {
        .good, .best, .custom => "scale_search",
    };
}

fn resolveNvfp4CaptureBackendSelection() @TypeOf((calibration_capture.CaptureOptions{}).backend_selection) {
    // Replay capture must be deterministic and fully instrumented.
    // Use CPU here so replay=xray does not depend on backend-specific trace coverage.
    return .cpu;
}

const OutputTensorKind = enum {
    passthrough,
    converted_weight,
    converted_scale,
    converted_scale_2,
    converted_input_scale,
    preserved_bf16_weight,
    preserved_mxfp8_weight,
    preserved_mxfp8_block_scale,
    dense_bf16_rhs_weight,
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

const Nvfp4UpdateState = struct {
    round: u32 = 0,
    global_improvement_pct: f32 = 0.0,
    mean_improvement_pct: f32 = 0.0,
    kl_divergence: f64 = std.math.nan(f64),
    canary_mean_nll: f64 = std.math.nan(f64),
    canary_ppl: f64 = std.math.nan(f64),
    canary_mean_kld: f64 = std.math.nan(f64),
    canary_scored_tokens: u32 = 0,
};

fn nvfp4UpdateStatePath(allocator: std.mem.Allocator, output_path: []const u8) ![]u8 {
    return std.fs.path.join(allocator, &.{ output_path, "nvfp4_update_state.json" });
}

const Nvfp4CanaryTokens = struct {
    data: []u32,
    context_len: usize,

    fn context(self: *const Nvfp4CanaryTokens) []const u32 {
        return self.data[0..self.context_len];
    }

    fn target(self: *const Nvfp4CanaryTokens) []const u32 {
        return self.data[self.context_len..];
    }

    fn deinit(self: *Nvfp4CanaryTokens, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
        self.* = .{
            .data = &.{},
            .context_len = 0,
        };
    }
};

const Nvfp4CanaryEval = struct {
    scored_tokens: usize,
    mean_nll: f64,
    ppl: f64,
    mean_kld: f64,
    mean_logit_mse: f64,
    mean_logit_nmse: f64,
};

const Nvfp4CanaryReference = struct {
    tokens: Nvfp4CanaryTokens,
    logits: []f32,
    token_nlls: []f64,
    vocab_size: usize,
    nll_sum: f64,
    scored_tokens: usize,

    fn deinit(self: *Nvfp4CanaryReference, allocator: std.mem.Allocator) void {
        self.tokens.deinit(allocator);
        allocator.free(self.logits);
        allocator.free(self.token_nlls);
        self.* = .{
            .tokens = .{ .data = &.{}, .context_len = 0 },
            .logits = &.{},
            .token_nlls = &.{},
            .vocab_size = 0,
            .nll_sum = 0.0,
            .scored_tokens = 0,
        };
    }

    fn evalMetrics(self: *const Nvfp4CanaryReference) Nvfp4CanaryEval {
        if (self.scored_tokens == 0) {
            return .{
                .scored_tokens = 0,
                .mean_nll = std.math.nan(f64),
                .ppl = std.math.nan(f64),
                .mean_kld = 0.0,
                .mean_logit_mse = 0.0,
                .mean_logit_nmse = 0.0,
            };
        }
        const scored_f: f64 = @floatFromInt(self.scored_tokens);
        const mean_nll = self.nll_sum / scored_f;
        return .{
            .scored_tokens = self.scored_tokens,
            .mean_nll = mean_nll,
            .ppl = std.math.exp(mean_nll),
            .mean_kld = 0.0,
            .mean_logit_mse = 0.0,
            .mean_logit_nmse = 0.0,
        };
    }
};

const Nvfp4CanaryComparison = struct {
    source: Nvfp4CanaryEval,
    candidate: Nvfp4CanaryEval,
    delta_nll: f64,
    nll_regression_pct: f64,
    nll_regression_ci95_low_pct: f64,
    nll_regression_ci95_high_pct: f64,
    nll_regression_p95_pct: f64,
    ppl_inflation_pct: f64,
};

const Nvfp4CanaryCandidateResult = struct {
    eval: Nvfp4CanaryEval,
    nll_regression_pct: f64,
    nll_regression_ci95_low_pct: f64,
    nll_regression_ci95_high_pct: f64,
    nll_regression_p95_pct: f64,
};

const Nvfp4QualityReportEntry = struct {
    round: u32,
    profile: @TypeOf((grouped_affine.ConvertOptions{}).profile),
    accepted: bool,
    internal_global: f32,
    internal_mean: f32,
    internal_kl: f64,
    canary_start_nll: f64,
    canary_end_nll: f64,
    canary_start_ppl: f64,
    canary_end_ppl: f64,
    canary_mean_kld: f64,
    scored_tokens: usize,
};

fn nvfp4QualityReportPath(allocator: std.mem.Allocator, output_path: []const u8) ![]u8 {
    return std.fs.path.join(allocator, &.{ output_path, "nvfp4_quality_report.jsonl" });
}

fn appendNvfp4QualityReport(
    allocator: std.mem.Allocator,
    output_path: []const u8,
    entry: Nvfp4QualityReportEntry,
) !void {
    const report_path = try nvfp4QualityReportPath(allocator, output_path);
    defer allocator.free(report_path);

    var report_file = std.fs.cwd().openFile(report_path, .{ .mode = .read_write }) catch |err| switch (err) {
        error.FileNotFound => try std.fs.cwd().createFile(report_path, .{ .truncate = false }),
        else => return err,
    };
    defer report_file.close();

    try report_file.seekFromEnd(0);
    const line = try std.fmt.allocPrint(
        allocator,
        "{{\"ts\":{d},\"round\":{d},\"profile\":\"{s}\",\"accepted\":{s},\"internal_global_pct\":{d:.6},\"internal_mean_pct\":{d:.6},\"internal_kl\":{e},\"canary_start_nll\":{e},\"canary_end_nll\":{e},\"canary_start_ppl\":{e},\"canary_end_ppl\":{e},\"canary_mean_kld\":{e},\"canary_scored_tokens\":{d}}}\n",
        .{
            std.time.timestamp(),
            entry.round,
            nvfp4ProfileName(entry.profile),
            if (entry.accepted) "true" else "false",
            entry.internal_global,
            entry.internal_mean,
            if (std.math.isFinite(entry.internal_kl) and entry.internal_kl >= 0.0) entry.internal_kl else 0.0,
            entry.canary_start_nll,
            entry.canary_end_nll,
            entry.canary_start_ppl,
            entry.canary_end_ppl,
            if (std.math.isFinite(entry.canary_mean_kld) and entry.canary_mean_kld >= 0.0) entry.canary_mean_kld else 0.0,
            entry.scored_tokens,
        },
    );
    defer allocator.free(line);
    try report_file.writeAll(line);
}

fn nvfp4CanaryReportPath(allocator: std.mem.Allocator, output_path: []const u8) ![]u8 {
    return std.fs.path.join(allocator, &.{ output_path, "nvfp4_canary_report.json" });
}

fn nvfp4ProbeHistoryPath(allocator: std.mem.Allocator, output_path: []const u8) ![]u8 {
    return std.fs.path.join(allocator, &.{ output_path, "nvfp4_probe_history.jsonl" });
}

fn appendNvfp4ProbeHistory(
    allocator: std.mem.Allocator,
    output_path: []const u8,
    options: ConvertOptions,
    comparison: Nvfp4CanaryComparison,
) !void {
    const nmse_pct = if (std.math.isFinite(comparison.candidate.mean_logit_nmse))
        comparison.candidate.mean_logit_nmse * 100.0
    else
        std.math.nan(f64);
    const history_path = try nvfp4ProbeHistoryPath(allocator, output_path);
    defer allocator.free(history_path);

    var history_file = std.fs.cwd().openFile(history_path, .{ .mode = .read_write }) catch |err| switch (err) {
        error.FileNotFound => try std.fs.cwd().createFile(history_path, .{ .truncate = false }),
        else => return err,
    };
    defer history_file.close();
    try history_file.seekFromEnd(0);

    const line = try std.fmt.allocPrint(
        allocator,
        "{{\"ts\":{d},\"profile\":\"{s}\",\"tokens\":{d},\"nll_regression_pct\":{e},\"nll_ci95_low_pct\":{e},\"nll_ci95_high_pct\":{e},\"nll_p95_pct\":{e},\"logit_nmse_pct\":{e}}}\n",
        .{
            std.time.timestamp(),
            nvfp4ProfileName(options.profile),
            comparison.candidate.scored_tokens,
            comparison.nll_regression_pct,
            comparison.nll_regression_ci95_low_pct,
            comparison.nll_regression_ci95_high_pct,
            comparison.nll_regression_p95_pct,
            nmse_pct,
        },
    );
    defer allocator.free(line);
    try history_file.writeAll(line);
}

fn writeNvfp4CanaryReport(
    allocator: std.mem.Allocator,
    output_path: []const u8,
    comparison: Nvfp4CanaryComparison,
) !void {
    const report_path = try nvfp4CanaryReportPath(allocator, output_path);
    defer allocator.free(report_path);

    const ppl_ratio = if (comparison.source.ppl > 0.0 and std.math.isFinite(comparison.source.ppl) and std.math.isFinite(comparison.candidate.ppl))
        comparison.candidate.ppl / comparison.source.ppl
    else
        std.math.nan(f64);
    const payload = try std.fmt.allocPrint(
        allocator,
        "{{\"tokens\":{d},\"source\":{{\"mean_nll\":{e},\"ppl\":{e},\"mean_kld\":{e},\"mean_logit_mse\":{e},\"mean_logit_nmse\":{e}}},\"candidate\":{{\"mean_nll\":{e},\"ppl\":{e},\"mean_kld\":{e},\"mean_logit_mse\":{e},\"mean_logit_nmse\":{e}}},\"delta_nll\":{e},\"nll_regression_pct\":{e},\"nll_regression_ci95_low_pct\":{e},\"nll_regression_ci95_high_pct\":{e},\"nll_regression_p95_pct\":{e},\"ppl_ratio\":{e},\"ppl_inflation_pct\":{e}}}\n",
        .{
            comparison.candidate.scored_tokens,
            comparison.source.mean_nll,
            comparison.source.ppl,
            comparison.source.mean_kld,
            comparison.source.mean_logit_mse,
            comparison.source.mean_logit_nmse,
            comparison.candidate.mean_nll,
            comparison.candidate.ppl,
            comparison.candidate.mean_kld,
            comparison.candidate.mean_logit_mse,
            comparison.candidate.mean_logit_nmse,
            comparison.delta_nll,
            comparison.nll_regression_pct,
            comparison.nll_regression_ci95_low_pct,
            comparison.nll_regression_ci95_high_pct,
            comparison.nll_regression_p95_pct,
            ppl_ratio,
            comparison.ppl_inflation_pct,
        },
    );
    defer allocator.free(payload);

    var file = try std.fs.cwd().createFile(report_path, .{ .truncate = true });
    defer file.close();
    try file.writeAll(payload);
}

fn printNvfp4CanarySummary(
    comparison: Nvfp4CanaryComparison,
    options: ConvertOptions,
    quality_summary: Nvfp4QualitySummary,
) void {
    const lm_head_q = quality_summary.lm_head_quantized;
    const small_model_preserve = quality_summary.small_model_preserve_enabled;
    const preserve_format = resolveNvfp4PreserveFormat();
    const replay_policy = resolveNvfp4ReplayPolicy(options.profile);
    const clip_mult = resolveNvfp4CustomClipMultiplier(options.profile);
    const scale_refine_mult = resolveNvfp4CustomScaleRefineMultiplier(options.profile);
    const preserve_blocks = quality_summary.preserve_blocks_selected;
    const pack_weighted_gain_pct = quality_summary.progress_gain_pct;
    const mse_reduction_pct = quality_summary.modelStateRelativeMseReductionPct();
    std.debug.print(
        "\nNVFP4 quality probe: samples={d} tokens_per_sample={d} total_tokens={d}\n",
        .{ nvfp4_quality_sample_count, nvfp4_quality_tokens_per_sample, comparison.candidate.scored_tokens },
    );
    std.debug.print("Settings:\n", .{});
    std.debug.print(
        "  --opts replay={s},preserve_blocks={d},preserve_format={s},lm_head_q={d},small_model_preserve={d},clip_mult={d:.2},scale_refine_mult={d:.2}\n",
        .{
            nvfp4ReplayPolicyLabel(replay_policy),
            preserve_blocks,
            nvfp4PreserveFormatName(preserve_format),
            @intFromBool(lm_head_q),
            @intFromBool(small_model_preserve),
            clip_mult,
            scale_refine_mult,
        },
    );
    std.debug.print("\nResults:\n", .{});
    std.debug.print(
        "  NLL Δ%: {d:.2} | p95: {d:.2} | CI95: {d:.2}..{d:.2}\n",
        .{
            comparison.nll_regression_pct,
            comparison.nll_regression_p95_pct,
            comparison.nll_regression_ci95_low_pct,
            comparison.nll_regression_ci95_high_pct,
        },
    );
    std.debug.print("  pack_weighted_gain_pct: {d:.2}%\n", .{pack_weighted_gain_pct});
    std.debug.print("  mse_reduction_pct: {d:.2}%\n", .{mse_reduction_pct});
}

fn evaluateNvfp4CanaryAgainstSource(
    allocator: std.mem.Allocator,
    source_model_path: []const u8,
    candidate_model_path: []const u8,
    seed: u64,
) !Nvfp4CanaryComparison {
    var reference = try captureNvfp4CanaryReference(allocator, source_model_path, seed);
    defer reference.deinit(allocator);

    const source_eval = reference.evalMetrics();
    const candidate = try evaluateNvfp4CanaryCandidate(
        allocator,
        candidate_model_path,
        seed,
        &reference,
    );
    const candidate_eval = candidate.eval;
    const delta_nll = candidate_eval.mean_nll - source_eval.mean_nll;
    const ppl_inflation_pct = if (source_eval.ppl > 0.0 and std.math.isFinite(source_eval.ppl) and std.math.isFinite(candidate_eval.ppl))
        ((candidate_eval.ppl / source_eval.ppl) - 1.0) * 100.0
    else
        std.math.nan(f64);
    return .{
        .source = source_eval,
        .candidate = candidate_eval,
        .delta_nll = delta_nll,
        .nll_regression_pct = candidate.nll_regression_pct,
        .nll_regression_ci95_low_pct = candidate.nll_regression_ci95_low_pct,
        .nll_regression_ci95_high_pct = candidate.nll_regression_ci95_high_pct,
        .nll_regression_p95_pct = candidate.nll_regression_p95_pct,
        .ppl_inflation_pct = ppl_inflation_pct,
    };
}

fn parseF32FromJsonValue(value: anytype) ?f32 {
    return switch (value) {
        .float => |v| @floatCast(v),
        .integer => |v| @floatFromInt(v),
        else => null,
    };
}

fn parseF64FromJsonValue(value: anytype) ?f64 {
    return switch (value) {
        .float => |v| @floatCast(v),
        .integer => |v| @floatFromInt(v),
        else => null,
    };
}

fn readNvfp4UpdateState(allocator: std.mem.Allocator, output_path: []const u8) !Nvfp4UpdateState {
    const state_path = try nvfp4UpdateStatePath(allocator, output_path);
    defer allocator.free(state_path);

    var state_file = std.fs.cwd().openFile(state_path, .{}) catch |err| switch (err) {
        error.FileNotFound => return .{},
        else => return err,
    };
    defer state_file.close();

    const state_json = try state_file.readToEndAlloc(allocator, 64 * 1024);
    defer allocator.free(state_json);

    var parsed = json.parseValue(allocator, state_json, .{
        .max_size_bytes = 64 * 1024,
        .max_value_bytes = 64 * 1024,
        .max_string_bytes = 16 * 1024,
    }) catch return .{};
    defer parsed.deinit();
    if (parsed.value != .object) return .{};

    var state = Nvfp4UpdateState{};
    if (parsed.value.object.get("round")) |round_value| {
        if (round_value == .integer and round_value.integer >= 0) {
            state.round = std.math.cast(u32, round_value.integer) orelse std.math.maxInt(u32);
        }
    }
    if (parsed.value.object.get("global_improvement_pct")) |global_value| {
        if (parseF32FromJsonValue(global_value)) |global| {
            if (std.math.isFinite(global)) state.global_improvement_pct = global;
        }
    }
    if (parsed.value.object.get("mean_improvement_pct")) |mean_value| {
        if (parseF32FromJsonValue(mean_value)) |mean| {
            if (std.math.isFinite(mean)) state.mean_improvement_pct = mean;
        }
    }
    if (parsed.value.object.get("kl_divergence")) |kl_value| {
        if (parseF64FromJsonValue(kl_value)) |kl| {
            if (std.math.isFinite(kl) and kl >= 0.0) state.kl_divergence = kl;
        }
    }
    if (parsed.value.object.get("canary_mean_nll")) |nll_value| {
        if (parseF64FromJsonValue(nll_value)) |nll| {
            if (std.math.isFinite(nll) and nll >= 0.0) state.canary_mean_nll = nll;
        }
    }
    if (parsed.value.object.get("canary_ppl")) |ppl_value| {
        if (parseF64FromJsonValue(ppl_value)) |ppl| {
            if (std.math.isFinite(ppl) and ppl >= 0.0) state.canary_ppl = ppl;
        }
    }
    if (parsed.value.object.get("canary_mean_kld")) |kld_value| {
        if (parseF64FromJsonValue(kld_value)) |kld| {
            if (std.math.isFinite(kld) and kld >= 0.0) state.canary_mean_kld = kld;
        }
    }
    if (parsed.value.object.get("canary_scored_tokens")) |tokens_value| {
        if (tokens_value == .integer and tokens_value.integer >= 0) {
            state.canary_scored_tokens = std.math.cast(u32, tokens_value.integer) orelse std.math.maxInt(u32);
        }
    }
    return state;
}

fn writeNvfp4UpdateState(allocator: std.mem.Allocator, output_path: []const u8, state: Nvfp4UpdateState) !void {
    const state_path = try nvfp4UpdateStatePath(allocator, output_path);
    defer allocator.free(state_path);

    const state_json = try std.fmt.allocPrint(
        allocator,
        "{{\"version\":2,\"round\":{d},\"global_improvement_pct\":{d:.6},\"mean_improvement_pct\":{d:.6},\"kl_divergence\":{e},\"canary_mean_nll\":{e},\"canary_ppl\":{e},\"canary_mean_kld\":{e},\"canary_scored_tokens\":{d}}}\n",
        .{
            state.round,
            state.global_improvement_pct,
            state.mean_improvement_pct,
            if (std.math.isFinite(state.kl_divergence) and state.kl_divergence >= 0.0) state.kl_divergence else 0.0,
            if (std.math.isFinite(state.canary_mean_nll) and state.canary_mean_nll >= 0.0) state.canary_mean_nll else 0.0,
            if (std.math.isFinite(state.canary_ppl) and state.canary_ppl >= 0.0) state.canary_ppl else 0.0,
            if (std.math.isFinite(state.canary_mean_kld) and state.canary_mean_kld >= 0.0) state.canary_mean_kld else 0.0,
            state.canary_scored_tokens,
        },
    );
    defer allocator.free(state_json);

    var state_file = try std.fs.cwd().createFile(state_path, .{ .truncate = true });
    defer state_file.close();
    try state_file.writeAll(state_json);
}

fn resolveNvfp4LayoutMap(
    allocator: std.mem.Allocator,
    config_path: []const u8,
    n_layers: usize,
) !?convert.WeightLayoutMap {
    const model_type = try config_loader.readModelType(allocator, config_path);
    defer if (model_type) |mt| allocator.free(mt);

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
                layer_types_override = config_loader.parseLayerTypes(allocator, config_path, variant_names, arch.variant_aliases) catch null;
            }
        }
        return convert.buildWeightLayoutMapWithOverride(allocator, arch, @intCast(n_layers), layer_types_override) catch |err| blk: {
            log.warn("converter", "Failed to build layout map", .{ .err = @errorName(err), .arch = arch.name });
            break :blk null;
        };
    }
    if (model_type) |mt| {
        log.warn("converter", "No runtime architecture metadata for model_type", .{ .model_type = mt });
    }
    return null;
}

inline fn mix64(v: u64) u64 {
    var z = v +% 0x9e3779b97f4a7c15;
    z = (z ^ (z >> 30)) *% 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) *% 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

fn negLogProbFromLogits(logits: []const f32, token_id: u32) !f64 {
    const token_index: usize = @intCast(token_id);
    if (token_index >= logits.len) return error.InvalidArgument;

    const target_logit = logits[token_index];
    if (!std.math.isFinite(target_logit)) return error.InvalidArgument;

    var max_logit: f32 = -std.math.inf(f32);
    for (logits) |value| {
        if (std.math.isFinite(value)) {
            max_logit = @max(max_logit, value);
        }
    }
    if (!std.math.isFinite(max_logit)) return error.InvalidArgument;

    var exp_sum: f64 = 0.0;
    for (logits) |value| {
        if (std.math.isFinite(value)) {
            exp_sum += std.math.exp(@as(f64, value - max_logit));
        }
    }
    if (!(exp_sum > 0.0) or !std.math.isFinite(exp_sum)) return error.InvalidArgument;

    const log_denom = @as(f64, max_logit) + std.math.log(f64, std.math.e, exp_sum);
    return log_denom - @as(f64, target_logit);
}

fn klDivergenceFromLogitsWithScratch(
    reference_logits: []const f32,
    model_logits: []const f32,
    ref_exp_scratch: []f64,
) !f64 {
    if (reference_logits.len == 0 or model_logits.len == 0) return error.InvalidArgument;
    if (reference_logits.len != model_logits.len) return error.InvalidArgument;
    if (ref_exp_scratch.len < reference_logits.len) return error.InvalidArgument;

    var max_ref: f32 = -std.math.inf(f32);
    var max_model: f32 = -std.math.inf(f32);
    for (reference_logits, model_logits) |ref_v, model_v| {
        if (!std.math.isFinite(ref_v) or !std.math.isFinite(model_v)) return error.InvalidArgument;
        max_ref = @max(max_ref, ref_v);
        max_model = @max(max_model, model_v);
    }

    var sum_exp_ref: f64 = 0.0;
    var sum_exp_model: f64 = 0.0;
    for (reference_logits, model_logits, 0..) |ref_v, model_v, idx| {
        const ref_exp = std.math.exp(@as(f64, ref_v - max_ref));
        ref_exp_scratch[idx] = ref_exp;
        sum_exp_ref += ref_exp;
        sum_exp_model += std.math.exp(@as(f64, model_v - max_model));
    }
    if (!(sum_exp_ref > 0.0) or !(sum_exp_model > 0.0)) return error.InvalidArgument;

    const ref_log_denom = @as(f64, max_ref) + std.math.log(f64, std.math.e, sum_exp_ref);
    const model_log_denom = @as(f64, max_model) + std.math.log(f64, std.math.e, sum_exp_model);

    var kld: f64 = 0.0;
    const inv_sum_ref = 1.0 / sum_exp_ref;
    for (reference_logits, model_logits, 0..) |ref_v, model_v, idx| {
        const log_p_ref = @as(f64, ref_v) - ref_log_denom;
        const log_p_model = @as(f64, model_v) - model_log_denom;
        const p_ref = ref_exp_scratch[idx] * inv_sum_ref;
        kld += p_ref * (log_p_ref - log_p_model);
    }

    if (!std.math.isFinite(kld) or kld < 0.0) return error.InvalidArgument;
    return kld;
}

fn buildNvfp4CanaryTokens(
    allocator: std.mem.Allocator,
    engine: *router_local.LocalEngine,
) !Nvfp4CanaryTokens {
    const canary_text =
        \\The assistant should preserve model behavior while quantizing efficiently.
        \\Quantization quality must be measured deterministically and reported clearly.
        \\We test summarization, reasoning, and generation quality over fixed prompts.
        \\A compact benchmark should be reproducible across repeated conversion runs.
        \\Numbers: 0 1 2 3 4 5 6 7 8 9.
        \\Code: if (x < y) { return x + y; } else { return y - x; }
    ;
    const encoded = try engine.encode(canary_text);
    defer allocator.free(encoded);
    if (encoded.len == 0) return error.InvalidConfig;

    const total_tokens = nvfp4_canary_context_tokens + nvfp4_canary_target_tokens;
    const data = try allocator.alloc(u32, total_tokens);
    errdefer allocator.free(data);
    for (0..total_tokens) |idx| {
        data[idx] = encoded[idx % encoded.len];
    }
    return .{
        .data = data,
        .context_len = nvfp4_canary_context_tokens,
    };
}

fn captureNvfp4CanaryReference(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    seed: u64,
) !Nvfp4CanaryReference {
    var engine = try router_local.LocalEngine.initWithSeed(allocator, model_path, seed);
    defer engine.deinit();

    var tokens = try buildNvfp4CanaryTokens(allocator, &engine);
    errdefer tokens.deinit(allocator);

    var scheduler = try engine.createScheduler(.{});
    defer scheduler.deinit();

    var cursor = try scheduler.beginTeacherForced(tokens.context());
    defer scheduler.endTeacherForced(&cursor);

    const target = tokens.target();
    if (target.len == 0) return error.InvalidConfig;

    const first_logits = try scheduler.teacherForcedCurrentLogits(&cursor);
    const vocab_size = first_logits.len;
    if (vocab_size == 0) return error.InvalidConfig;
    const total_logits = std.math.mul(usize, target.len, vocab_size) catch return error.InvalidConfig;
    const logits_buf = try allocator.alloc(f32, total_logits);
    errdefer allocator.free(logits_buf);
    const token_nlls = try allocator.alloc(f64, target.len);
    errdefer allocator.free(token_nlls);

    var nll_sum: f64 = 0.0;
    for (target, 0..) |token_id, idx| {
        const logits = if (idx == 0) first_logits else try scheduler.teacherForcedCurrentLogits(&cursor);
        if (logits.len != vocab_size) return error.InvalidConfig;
        const dst = logits_buf[idx * vocab_size .. (idx + 1) * vocab_size];
        @memcpy(dst, logits);
        const token_nll = try negLogProbFromLogits(logits, token_id);
        token_nlls[idx] = token_nll;
        nll_sum += token_nll;
        if (idx + 1 < target.len) {
            try scheduler.advanceTeacherForced(&cursor, token_id);
        }
    }

    return .{
        .tokens = tokens,
        .logits = logits_buf,
        .token_nlls = token_nlls,
        .vocab_size = vocab_size,
        .nll_sum = nll_sum,
        .scored_tokens = target.len,
    };
}

fn evaluateNvfp4CanaryCandidate(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    seed: u64,
    reference: *const Nvfp4CanaryReference,
) !Nvfp4CanaryCandidateResult {
    if (reference.scored_tokens == 0 or reference.vocab_size == 0) return error.InvalidConfig;
    if (reference.logits.len != reference.scored_tokens * reference.vocab_size) return error.InvalidConfig;
    if (reference.token_nlls.len != reference.scored_tokens) return error.InvalidConfig;

    var engine = try router_local.LocalEngine.initWithSeed(allocator, model_path, seed);
    defer engine.deinit();

    var scheduler = try engine.createScheduler(.{});
    defer scheduler.deinit();

    var cursor = try scheduler.beginTeacherForced(reference.tokens.context());
    defer scheduler.endTeacherForced(&cursor);

    const target = reference.tokens.target();
    if (target.len != reference.scored_tokens) return error.InvalidConfig;

    const scratch = try allocator.alloc(f64, reference.vocab_size);
    defer allocator.free(scratch);

    var nll_sum: f64 = 0.0;
    var kld_sum: f64 = 0.0;
    var logit_mse_sum: f64 = 0.0;
    var logit_ref_power_sum: f64 = 0.0;
    var sample_ref_nll_sum: f64 = 0.0;
    var sample_cand_nll_sum: f64 = 0.0;
    var sample_nll_regression_pct = [_]f64{std.math.nan(f64)} ** nvfp4_quality_sample_count;
    var sample_nll_regression_count: usize = 0;
    for (target, 0..) |token_id, idx| {
        const logits = try scheduler.teacherForcedCurrentLogits(&cursor);
        if (logits.len != reference.vocab_size) return error.InvalidConfig;
        const token_nll = try negLogProbFromLogits(logits, token_id);
        nll_sum += token_nll;
        sample_cand_nll_sum += token_nll;
        sample_ref_nll_sum += reference.token_nlls[idx];

        const ref_logits = reference.logits[idx * reference.vocab_size .. (idx + 1) * reference.vocab_size];
        kld_sum += try klDivergenceFromLogitsWithScratch(ref_logits, logits, scratch);
        var token_err_sum: f64 = 0.0;
        var token_ref_pow_sum: f64 = 0.0;
        for (ref_logits, logits) |ref_v, cand_v| {
            const ref_f = @as(f64, ref_v);
            const cand_f = @as(f64, cand_v);
            const diff = ref_f - cand_f;
            token_err_sum += diff * diff;
            token_ref_pow_sum += ref_f * ref_f;
        }
        const vocab_f: f64 = @floatFromInt(reference.vocab_size);
        logit_mse_sum += token_err_sum / vocab_f;
        logit_ref_power_sum += token_ref_pow_sum / vocab_f;

        const is_sample_end = ((idx + 1) % nvfp4_quality_tokens_per_sample) == 0;
        if (is_sample_end) {
            const ref_mean = sample_ref_nll_sum / @as(f64, @floatFromInt(nvfp4_quality_tokens_per_sample));
            const cand_mean = sample_cand_nll_sum / @as(f64, @floatFromInt(nvfp4_quality_tokens_per_sample));
            const reg_pct = if (ref_mean > 0.0 and std.math.isFinite(ref_mean) and std.math.isFinite(cand_mean))
                ((cand_mean / ref_mean) - 1.0) * 100.0
            else
                std.math.nan(f64);
            if (sample_nll_regression_count < sample_nll_regression_pct.len) {
                sample_nll_regression_pct[sample_nll_regression_count] = reg_pct;
                sample_nll_regression_count += 1;
            }
            sample_ref_nll_sum = 0.0;
            sample_cand_nll_sum = 0.0;
        }
        if (idx + 1 < target.len) {
            try scheduler.advanceTeacherForced(&cursor, token_id);
        }
    }

    const scored_f: f64 = @floatFromInt(target.len);
    const mean_nll = nll_sum / scored_f;
    const mean_kld = kld_sum / scored_f;
    const mean_logit_mse = logit_mse_sum / scored_f;
    const mean_logit_nmse = if (logit_ref_power_sum > 0.0 and std.math.isFinite(logit_ref_power_sum))
        logit_mse_sum / logit_ref_power_sum
    else
        std.math.nan(f64);
    var nll_regression_sum: f64 = 0.0;
    var nll_regression_sq_sum: f64 = 0.0;
    for (sample_nll_regression_pct[0..sample_nll_regression_count]) |value| {
        if (!std.math.isFinite(value)) continue;
        nll_regression_sum += value;
        nll_regression_sq_sum += value * value;
    }
    const nll_regression_pct = if (sample_nll_regression_count > 0)
        nll_regression_sum / @as(f64, @floatFromInt(sample_nll_regression_count))
    else
        std.math.nan(f64);
    var nll_regression_ci95_low_pct = std.math.nan(f64);
    var nll_regression_ci95_high_pct = std.math.nan(f64);
    if (sample_nll_regression_count >= 2 and std.math.isFinite(nll_regression_pct)) {
        const n_f: f64 = @floatFromInt(sample_nll_regression_count);
        const var_numer = nll_regression_sq_sum - ((nll_regression_sum * nll_regression_sum) / n_f);
        const sample_var = if (var_numer > 0.0) var_numer / (n_f - 1.0) else 0.0;
        const std_err = std.math.sqrt(sample_var / n_f);
        const margin = 1.96 * std_err;
        nll_regression_ci95_low_pct = nll_regression_pct - margin;
        nll_regression_ci95_high_pct = nll_regression_pct + margin;
    }

    var nll_regression_p95_pct = std.math.nan(f64);
    if (sample_nll_regression_count > 0) {
        var sorted = [_]f64{std.math.nan(f64)} ** nvfp4_quality_sample_count;
        @memcpy(sorted[0..sample_nll_regression_count], sample_nll_regression_pct[0..sample_nll_regression_count]);
        std.mem.sort(f64, sorted[0..sample_nll_regression_count], {}, struct {
            fn lessThan(_: void, a: f64, b: f64) bool {
                return a < b;
            }
        }.lessThan);
        const rank = @max(@as(usize, 1), (sample_nll_regression_count * 95 + 99) / 100);
        nll_regression_p95_pct = sorted[rank - 1];
    }

    return .{
        .eval = .{
            .scored_tokens = target.len,
            .mean_nll = mean_nll,
            .ppl = std.math.exp(mean_nll),
            .mean_kld = mean_kld,
            .mean_logit_mse = mean_logit_mse,
            .mean_logit_nmse = mean_logit_nmse,
        },
        .nll_regression_pct = nll_regression_pct,
        .nll_regression_ci95_low_pct = nll_regression_ci95_low_pct,
        .nll_regression_ci95_high_pct = nll_regression_ci95_high_pct,
        .nll_regression_p95_pct = nll_regression_p95_pct,
    };
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
    if (resolveNvfp4ReplayPolicy(options.profile) != .capture_required) return null;
    if (options.calib_iters == 0) {
        std.debug.print(
            "NVFP4 convert failed: replay=xray requires activation capture (calib iters must be > 0).\n",
            .{},
        );
        return error.Nvfp4ReplayRequiresActivationCapture;
    }
    if (!calibration_capture.isAvailable()) {
        std.debug.print(
            "NVFP4 convert failed: replay=xray requires xray bridge. Rebuild with -Dxray_bridge=true.\n",
            .{},
        );
        return error.Nvfp4ReplayRequiresActivationCapture;
    }

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
    ) orelse {
        std.debug.print(
            "NVFP4 convert failed: replay=xray activation capture prompt token pool is empty.\n",
            .{},
        );
        return error.Nvfp4ReplayRequiresActivationCapture;
    };
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
            .backend_selection = resolveNvfp4CaptureBackendSelection(),
        },
    ) catch |err| {
        std.debug.print(
            "NVFP4 convert failed: replay=xray activation capture runtime error ({s}).\n",
            .{@errorName(err)},
        );
        return error.Nvfp4ReplayRequiresActivationCapture;
    };
    if (cache.count() == 0) {
        cache.deinit();
        std.debug.print(
            "NVFP4 convert failed: replay=xray activation capture produced 0 entries.\n",
            .{},
        );
        return error.Nvfp4ReplayRequiresActivationCapture;
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

fn extractVisualBlockIndexFromTensorName(name: []const u8) ?u32 {
    const marker = "visual.blocks.";
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

fn nvfp4TensorOrderKey(name: []const u8) u32 {
    if (std.mem.indexOf(u8, name, "embed_tokens") != null) return 0;
    if (extractLayerIndexFromTensorName(name)) |layer| return 1000 + layer;
    if (extractVisualBlockIndexFromTensorName(name)) |block| return 2000 + block;
    if (std.mem.indexOf(u8, name, "norm") != null) return 100000;
    if (std.mem.indexOf(u8, name, "lm_head") != null) return 100001;
    return 200000;
}

fn sortNvfp4TensorNames(names: [][]const u8) void {
    std.mem.sort([]const u8, names, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            const ak = nvfp4TensorOrderKey(a);
            const bk = nvfp4TensorOrderKey(b);
            if (ak != bk) return ak < bk;
            return std.mem.lessThan(u8, a, b);
        }
    }.lessThan);
}

const SmallModelPreservePolicy = struct {
    enabled: bool = false,
    last_layer_index: ?u32 = null,
};

fn makeSmallModelPreservePolicy(
    small_model_preserve_enabled: bool,
    dense_weight_params_total: u64,
    max_layer_index: ?u32,
) SmallModelPreservePolicy {
    if (!small_model_preserve_enabled) return .{};
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

const LayerScore = struct {
    layer_index: u32,
    score: f64,
};

fn mixedPreserveLayerCount(total_layers: u32, pct: u32) u32 {
    if (total_layers == 0 or pct == 0) return 0;
    const scaled = std.math.mul(u64, total_layers, pct) catch return total_layers;
    const ceil_count = (scaled + 99) / 100;
    const bounded = @max(@as(u64, 1), @min(@as(u64, total_layers), ceil_count));
    return std.math.cast(u32, bounded) orelse total_layers;
}

fn shouldPreserveWeightByLayerList(weight_name: []const u8, preserved_layers: []const u32) bool {
    if (preserved_layers.len == 0) return false;
    const layer_index = extractLayerIndexFromTensorName(weight_name) orelse return false;
    for (preserved_layers) |preserved| {
        if (layer_index == preserved) return true;
    }
    return false;
}

fn subtractLayerLists(allocator: std.mem.Allocator, lhs: []const u32, rhs: []const u32) ![]u32 {
    if (lhs.len == 0) return allocator.alloc(u32, 0);
    var out = std.ArrayListUnmanaged(u32){};
    errdefer out.deinit(allocator);
    for (lhs) |layer| {
        var found = false;
        for (rhs) |excluded| {
            if (layer == excluded) {
                found = true;
                break;
            }
        }
        if (!found) try out.append(allocator, layer);
    }
    return out.toOwnedSlice(allocator);
}

fn shouldUseAdvancedSearchForTensor(
    profile: @TypeOf((grouped_affine.ConvertOptions{}).profile),
    update_round: u32,
) bool {
    return useAdvancedNvfp4Search(profile, update_round);
}

const Nvfp4BlockScalePolicy = enum {
    legacy_multi,
};

fn nvfp4BlockScalePolicyForProfile(profile: @TypeOf((grouped_affine.ConvertOptions{}).profile)) Nvfp4BlockScalePolicy {
    return switch (profile) {
        .good, .best, .custom => .legacy_multi,
    };
}

fn estimateNvfp4TensorSensitivityQuick(
    source: DenseWeightView,
    rows: usize,
    cols: usize,
    groups: usize,
    sample_blocks_cap: usize,
    sample_seed: u64,
    block_scale_policy: Nvfp4BlockScalePolicy,
) f64 {
    if (rows == 0 or cols == 0 or groups == 0) return std.math.nan(f64);
    const total_blocks = std.math.mul(usize, rows, groups) catch return std.math.nan(f64);
    if (total_blocks == 0) return std.math.nan(f64);

    const sample_blocks = @max(@as(usize, 1), @min(total_blocks, sample_blocks_cap));
    const step = @max(@as(usize, 1), total_blocks / sample_blocks);
    const offset = @as(usize, @intCast(mix64(sample_seed) % @as(u64, @intCast(total_blocks))));

    var sampled_scales = [_]f32{0.0} ** mixed_preserve_score_sample_blocks_default;
    var sampled_weights = [_]f32{1.0} ** mixed_preserve_score_sample_blocks_default;
    var sampled_block_indices = [_]usize{0} ** mixed_preserve_score_sample_blocks_default;
    var observed_max_scale: f32 = 0.0;
    var sampled_count: usize = 0;
    var block_vals: [nvfp4_group_size]f32 = undefined;

    for (0..sample_blocks) |sample_idx| {
        const block_idx = (offset + sample_idx * step) % total_blocks;
        const row = block_idx / groups;
        const group = block_idx % groups;
        const block_base = row * cols + group * nvfp4_group_size;
        const max_abs = loadNvfp4BlockValuesAndMaxAbs(source, block_base, &block_vals);
        const block_scale = chooseNvfp4BlockScaleWithPolicy(block_vals[0..], max_abs, 1.0, block_scale_policy);
        if (!(block_scale > 0.0) or !std.math.isFinite(block_scale)) continue;

        sampled_scales[sampled_count] = block_scale;
        sampled_weights[sampled_count] = 1.0;
        sampled_block_indices[sampled_count] = block_idx;
        observed_max_scale = @max(observed_max_scale, block_scale);
        sampled_count += 1;
    }
    if (sampled_count == 0) return std.math.nan(f64);

    const global_scale = chooseNvfp4GlobalScale(
        sampled_scales[0..sampled_count],
        sampled_weights[0..sampled_count],
        observed_max_scale,
    );
    if (!(global_scale > 0.0) or !std.math.isFinite(global_scale)) return std.math.nan(f64);

    var err_sum: f64 = 0.0;
    var ref_sum: f64 = 0.0;
    for (0..sampled_count) |idx| {
        const block_idx = sampled_block_indices[idx];
        const row = block_idx / groups;
        const group = block_idx % groups;
        const block_base = row * cols + group * nvfp4_group_size;
        const max_abs = loadNvfp4BlockValuesAndMaxAbs(source, block_base, &block_vals);
        const block_scale = chooseNvfp4BlockScaleWithPolicy(block_vals[0..], max_abs, 1.0, block_scale_policy);
        const packed_scale = dtype.f32ToFp8E4M3(block_scale / global_scale);
        const scale_f32 = dtype.fp8e4m3ToF32(packed_scale) * global_scale;
        if (!(scale_f32 > 0.0) or !std.math.isFinite(scale_f32)) continue;

        for (0..nvfp4_group_size) |i| {
            const ref = @as(f64, @floatCast(block_vals[i]));
            const scaled = block_vals[i] / scale_f32;
            const nibble = nearestFp4E2m1Nibble(scaled);
            const dq = @as(f64, @floatCast(fp4E2m1NibbleToF32(nibble) * scale_f32));
            const diff = ref - dq;
            err_sum += diff * diff;
            ref_sum += ref * ref;
        }
    }

    if (!std.math.isFinite(err_sum) or err_sum < 0.0) return std.math.nan(f64);
    if (!(ref_sum > 0.0) or !std.math.isFinite(ref_sum)) return err_sum;
    return err_sum / ref_sum;
}

fn buildMixedPreserveLayerList(
    allocator: std.mem.Allocator,
    st: *safetensors.UnifiedSafeTensors,
    names: [][]const u8,
    profile: @TypeOf((grouped_affine.ConvertOptions{}).profile),
    lm_head_quantized: bool,
    preserve_blocks_override: ?u32,
    default_pct: u32,
) ![]u32 {
    if (default_pct == 0 and preserve_blocks_override == null) return allocator.alloc(u32, 0);
    const sample_blocks_cap = mixedPreserveScoreSampleBlocks(profile);
    const block_scale_policy = nvfp4BlockScalePolicyForProfile(profile);

    var max_layer_index: ?u32 = null;
    var layer_scores = std.AutoHashMap(u32, f64).init(allocator);
    defer layer_scores.deinit();

    for (names) |name| {
        const layer_index = extractLayerIndexFromTensorName(name) orelse continue;
        const weight = st.getTensor(name, null) catch continue;
        if (!shouldConvertDenseWeightWithLmHead(name, weight, profile, lm_head_quantized)) continue;
        if (weight.n_dims != 2 or weight.shape[0] <= 0 or weight.shape[1] <= 0) continue;

        const rows: usize = @intCast(weight.shape[0]);
        const cols: usize = @intCast(weight.shape[1]);
        if (cols % nvfp4_group_size != 0) continue;
        const groups = cols / nvfp4_group_size;
        const source = DenseWeightView.init(weight) catch continue;
        const sample_seed = mix64(std.hash.Wyhash.hash(0, name));
        const nmse = estimateNvfp4TensorSensitivityQuick(source, rows, cols, groups, sample_blocks_cap, sample_seed, block_scale_policy);
        if (!std.math.isFinite(nmse) or nmse < 0.0) continue;

        if (max_layer_index == null or layer_index > max_layer_index.?) {
            max_layer_index = layer_index;
        }

        const params = std.math.mul(usize, rows, cols) catch continue;
        const weighted_score = nmse * @as(f64, @floatFromInt(params));
        if (!std.math.isFinite(weighted_score) or weighted_score <= 0.0) continue;

        const slot = try layer_scores.getOrPut(layer_index);
        if (slot.found_existing) {
            slot.value_ptr.* += weighted_score;
        } else {
            slot.value_ptr.* = weighted_score;
        }
    }

    const last_layer = max_layer_index orelse return allocator.alloc(u32, 0);
    const total_layers = last_layer + 1;
    const target_layers = if (preserve_blocks_override) |override_blocks|
        @min(total_layers, override_blocks)
    else
        mixedPreserveLayerCount(total_layers, default_pct);
    if (target_layers == 0 or layer_scores.count() == 0) return allocator.alloc(u32, 0);

    var scored = try allocator.alloc(LayerScore, layer_scores.count());
    defer allocator.free(scored);
    var idx: usize = 0;
    var it = layer_scores.iterator();
    while (it.next()) |entry| {
        scored[idx] = .{
            .layer_index = entry.key_ptr.*,
            .score = entry.value_ptr.*,
        };
        idx += 1;
    }

    std.mem.sort(LayerScore, scored, {}, struct {
        fn lessThan(_: void, a: LayerScore, b: LayerScore) bool {
            if (a.score != b.score) return a.score > b.score;
            return a.layer_index < b.layer_index;
        }
    }.lessThan);

    const keep = @min(@as(usize, target_layers), scored.len);
    var preserved = try allocator.alloc(u32, keep);
    for (0..keep) |keep_idx| {
        preserved[keep_idx] = scored[keep_idx].layer_index;
    }
    std.mem.sort(u32, preserved, {}, std.sort.asc(u32));
    return preserved;
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

const Nvfp4ForwardReplayInputs = struct {
    sampled_inputs: calibration_capture.SampledActivations,
    row_indices: []usize,
    from_capture: bool,

    fn deinit(self: *Nvfp4ForwardReplayInputs, allocator: std.mem.Allocator) void {
        self.sampled_inputs.deinit(allocator);
        allocator.free(self.row_indices);
    }
};

fn buildNvfp4ForwardReplayInputs(
    allocator: std.mem.Allocator,
    source_weight_name: []const u8,
    source: DenseWeightView,
    rows: usize,
    cols: usize,
    activation_cache: ?*const calibration_capture.LayerActivationCache,
    activation_sample_count: usize,
    calib_seed: u64,
) !?Nvfp4ForwardReplayInputs {
    if (activation_sample_count == 0 or rows == 0 or cols == 0) return null;
    const name_seed = mix64(calib_seed ^ std.hash.Wyhash.hash(0, source_weight_name));
    var from_capture = false;
    const sampled: calibration_capture.SampledActivations = blk: {
        if (activation_cache) |cache| {
            if (extractLayerIndexFromTensorName(source_weight_name)) |layer_index| {
                const role = activationRoleForTensorName(source_weight_name);
                if (try calibration_capture.sampleLayerActivationsForRole(
                    allocator,
                    cache,
                    layer_index,
                    cols,
                    activation_sample_count,
                    name_seed,
                    role,
                )) |captured| {
                    from_capture = true;
                    break :blk captured;
                }
            }
        }

        const synthetic_count = @max(@as(usize, 8), @min(activation_sample_count, @as(usize, 48)));
        const synthetic_values = try allocator.alloc(f32, synthetic_count * cols);
        errdefer allocator.free(synthetic_values);
        const input_offset: usize = @intCast(mix64(name_seed ^ 0x2f98a4c12d947a59) % @as(u64, @intCast(rows)));
        var input_stride: usize = rows / synthetic_count;
        if (input_stride == 0) input_stride = 1;
        for (0..synthetic_count) |sample_idx| {
            const row = (input_offset + sample_idx * input_stride) % rows;
            var l2_sum: f64 = 0.0;
            for (0..cols) |col| {
                const value = source.valueAt(row * cols + col);
                synthetic_values[sample_idx * cols + col] = value;
                const v64 = @as(f64, @floatCast(value));
                l2_sum += v64 * v64;
            }
            const norm = @as(f32, @floatCast(std.math.sqrt(@max(l2_sum / @as(f64, @floatFromInt(cols)), 1e-12))));
            const inv = if (norm > 0.0 and std.math.isFinite(norm)) 1.0 / norm else 1.0;
            for (0..cols) |col| synthetic_values[sample_idx * cols + col] *= inv;
        }
        break :blk .{
            .values = synthetic_values,
            .sample_count = synthetic_count,
            .cols = cols,
        };
    };
    errdefer sampled.deinit(allocator);

    const sample_rows = @max(@as(usize, 1), @min(rows, nvfp4_forward_eval_row_cap));
    const row_indices = try allocator.alloc(usize, sample_rows);
    errdefer allocator.free(row_indices);

    const offset: usize = @intCast(mix64(name_seed ^ 0x7a5b9d12f31e5b7d) % @as(u64, @intCast(rows)));
    var stride: usize = rows / sample_rows;
    if (stride == 0) stride = 1;
    for (0..sample_rows) |idx| {
        row_indices[idx] = (offset + idx * stride) % rows;
    }

    return .{
        .sampled_inputs = sampled,
        .row_indices = row_indices,
        .from_capture = from_capture,
    };
}

fn evaluateNvfp4ForwardReplayMseCpu(
    x_values: []const f32,
    input_samples: usize,
    cols: usize,
    sample_rows: usize,
    ref_weights: []const f32,
    dq_weights: []const f32,
) f64 {
    var err_sum: f64 = 0.0;
    var sample_idx: usize = 0;
    while (sample_idx < input_samples) : (sample_idx += 1) {
        var row_idx: usize = 0;
        while (row_idx < sample_rows) : (row_idx += 1) {
            var ref_out: f64 = 0.0;
            var dq_out: f64 = 0.0;
            var col: usize = 0;
            while (col < cols) : (col += 1) {
                const x = @as(f64, x_values[sample_idx * cols + col]);
                const w_idx = col * sample_rows + row_idx;
                ref_out += x * @as(f64, ref_weights[w_idx]);
                dq_out += x * @as(f64, dq_weights[w_idx]);
            }
            const diff = ref_out - dq_out;
            err_sum += diff * diff;
        }
    }
    const denom = @as(f64, @floatFromInt(input_samples * sample_rows));
    if (!(denom > 0.0)) return std.math.inf(f64);
    return err_sum / denom;
}

fn evaluateNvfp4ForwardReplayMseMetal(
    allocator: std.mem.Allocator,
    x_values: []const f32,
    input_samples: usize,
    cols: usize,
    sample_rows: usize,
    ref_weights: []const f32,
    dq_weights: []const f32,
) ?f64 {
    if (!has_metal_gpu_eval) return null;
    if (!shouldEnableNvfp4MetalReplay()) return null;
    if (!compute.metal.isAvailable()) return null;
    if (!ensureMlxMetallibForNvfp4Replay(allocator)) return null;

    const graph = compute.metal.graph;
    const x_shape = [_]i64{ @intCast(input_samples), @intCast(cols) };
    const w_shape = [_]i64{ @intCast(cols), @intCast(sample_rows) };
    const out_len = input_samples * sample_rows;
    const ref_host = allocator.alloc(f32, out_len) catch return null;
    defer allocator.free(ref_host);
    const dq_host = allocator.alloc(f32, out_len) catch return null;
    defer allocator.free(dq_host);

    graph.beginForwardGraphBuild();
    const x_handle = graph.createArrayF32(x_values, &x_shape);
    if (x_handle == null) return null;
    defer graph.freeArray(x_handle);
    const ref_w_handle = graph.createArrayF32(ref_weights, &w_shape);
    if (ref_w_handle == null) return null;
    defer graph.freeArray(ref_w_handle);
    const dq_w_handle = graph.createArrayF32(dq_weights, &w_shape);
    if (dq_w_handle == null) return null;
    defer graph.freeArray(dq_w_handle);

    const ref_out_handle = graph.mlx_lazy_matmul(x_handle, ref_w_handle);
    if (ref_out_handle == null) return null;
    defer graph.freeArray(ref_out_handle);
    const dq_out_handle = graph.mlx_lazy_matmul(x_handle, dq_w_handle);
    if (dq_out_handle == null) return null;
    defer graph.freeArray(dq_out_handle);

    var eval_handles = [_]compute.metal.graph.ArrayHandle{ ref_out_handle, dq_out_handle };
    graph.eval(&eval_handles);
    graph.copyToHost(ref_out_handle, ref_host);
    graph.copyToHost(dq_out_handle, dq_host);

    var err_sum: f64 = 0.0;
    for (ref_host, dq_host) |a, b| {
        const diff = @as(f64, @floatCast(a - b));
        err_sum += diff * diff;
    }
    return err_sum / @as(f64, @floatFromInt(out_len));
}

fn envFlagEnabled(name: []const u8) bool {
    const raw = std.posix.getenv(name) orelse return false;
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    if (trimmed.len == 0) return false;
    if (std.ascii.eqlIgnoreCase(trimmed, "0")) return false;
    if (std.ascii.eqlIgnoreCase(trimmed, "false")) return false;
    if (std.ascii.eqlIgnoreCase(trimmed, "off")) return false;
    if (std.ascii.eqlIgnoreCase(trimmed, "no")) return false;
    return true;
}

fn shouldEnableNvfp4MetalReplay() bool {
    if (std.posix.getenv("TALU_NVFP4_METAL_REPLAY")) |_| {
        return envFlagEnabled("TALU_NVFP4_METAL_REPLAY");
    }
    const backend_raw = std.posix.getenv("BACKEND") orelse return false;
    return std.ascii.eqlIgnoreCase(backend_raw, "metal");
}

fn shouldEnableNvfp4MetalPack() bool {
    if (!has_metal_gpu_eval) return false;
    if (std.posix.getenv("TALU_NVFP4_METAL_PACK")) |_| {
        if (!envFlagEnabled("TALU_NVFP4_METAL_PACK")) return false;
    }
    const backend_raw = std.posix.getenv("BACKEND") orelse return true;
    if (std.ascii.eqlIgnoreCase(backend_raw, "cpu")) return false;
    if (std.ascii.eqlIgnoreCase(backend_raw, "cuda")) return false;
    if (backend_raw.len == 0) return true;
    if (std.ascii.eqlIgnoreCase(backend_raw, "auto")) return true;
    if (std.ascii.eqlIgnoreCase(backend_raw, "metal")) return true;
    return false;
}

fn shouldPreferMetalReplayEval() bool {
    if (!has_metal_gpu_eval) return false;
    if (!shouldEnableNvfp4MetalReplay()) return false;
    return compute.metal.isAvailable();
}

const MlxMetallibStatus = enum(u8) {
    unknown = 0,
    ready = 1,
    unavailable = 2,
};

var mlx_metallib_status: std.atomic.Value(u8) = .init(@intFromEnum(MlxMetallibStatus.unknown));

const Nvfp4EnvFns = struct {
    extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
};

fn pathExists(path: []const u8) bool {
    if (path.len == 0) return false;
    if (std.fs.path.isAbsolute(path)) {
        std.fs.accessAbsolute(path, .{}) catch return false;
        return true;
    }
    std.fs.cwd().access(path, .{}) catch return false;
    return true;
}

fn mlxRuntimeBinaryDir() ?[]const u8 {
    if (!has_metal_gpu_eval) return null;
    const raw = mlx_runtime_binary_dir() orelse return null;
    const dir = std.mem.sliceTo(raw, 0);
    if (dir.len == 0) return null;
    return dir;
}

fn ensureMlxMetallibColocatedForNvfp4() void {
    if (!has_metal_gpu_eval) return;
    const runtime_dir = mlxRuntimeBinaryDir() orelse return;

    var dst_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dst = std.fmt.bufPrint(&dst_buf, "{s}/mlx.metallib", .{runtime_dir}) catch return;
    if (pathExists(dst)) return;

    var exe_dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const exe_dir = std.fs.selfExeDirPath(&exe_dir_buf) catch "";
    var cand1_buf: [std.fs.max_path_bytes]u8 = undefined;
    var cand2_buf: [std.fs.max_path_bytes]u8 = undefined;
    const candidate_exe = if (exe_dir.len > 0)
        (std.fmt.bufPrint(&cand1_buf, "{s}/mlx.metallib", .{exe_dir}) catch "")
    else
        "";
    const candidate_exe_lib = if (exe_dir.len > 0)
        (std.fmt.bufPrint(&cand2_buf, "{s}/../lib/mlx.metallib", .{exe_dir}) catch "")
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
        if (!pathExists(candidate)) continue;
        std.fs.cwd().copyFile(candidate, std.fs.cwd(), dst, .{}) catch continue;
        return;
    }
}

fn ensureMlxMetallibForNvfp4Replay(allocator: std.mem.Allocator) bool {
    const status = @as(MlxMetallibStatus, @enumFromInt(mlx_metallib_status.load(.acquire)));
    switch (status) {
        .ready => return true,
        .unavailable => return false,
        .unknown => {},
    }

    if (std.posix.getenv("MLX_METALLIB")) |configured| {
        if (pathExists(configured)) {
            mlx_metallib_status.store(@intFromEnum(MlxMetallibStatus.ready), .release);
            return true;
        }
    }

    var exe_dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const exe_dir = std.fs.selfExeDirPath(&exe_dir_buf) catch "";
    var cand1_buf: [std.fs.max_path_bytes]u8 = undefined;
    var cand2_buf: [std.fs.max_path_bytes]u8 = undefined;
    const candidate_exe = if (exe_dir.len > 0)
        (std.fmt.bufPrint(&cand1_buf, "{s}/mlx.metallib", .{exe_dir}) catch "")
    else
        "";
    const candidate_exe_lib = if (exe_dir.len > 0)
        (std.fmt.bufPrint(&cand2_buf, "{s}/../lib/mlx.metallib", .{exe_dir}) catch "")
    else
        "";

    const candidates = [_][]const u8{
        candidate_exe,
        candidate_exe_lib,
        "mlx.metallib",
        "zig-out/lib/mlx.metallib",
        "deps/mlx/lib/mlx.metallib",
    };

    for (candidates) |candidate| {
        if (!pathExists(candidate)) continue;

        const key_z = allocator.dupeZ(u8, "MLX_METALLIB") catch continue;
        defer allocator.free(key_z);
        const value_z = allocator.dupeZ(u8, candidate) catch continue;
        defer allocator.free(value_z);

        if (Nvfp4EnvFns.setenv(key_z.ptr, value_z.ptr, 1) == 0) {
            mlx_metallib_status.store(@intFromEnum(MlxMetallibStatus.ready), .release);
            return true;
        }
    }

    mlx_metallib_status.store(@intFromEnum(MlxMetallibStatus.unavailable), .release);
    return false;
}

fn estimateNvfp4ForwardReplayMse(
    allocator: std.mem.Allocator,
    source: DenseWeightView,
    cols: usize,
    groups: usize,
    global_scale: f32,
    clip_multiplier: f32,
    block_scale_policy: Nvfp4BlockScalePolicy,
    replay: *const Nvfp4ForwardReplayInputs,
) !f64 {
    if (!(global_scale > 0.0) or !std.math.isFinite(global_scale)) return std.math.inf(f64);
    if (replay.sampled_inputs.cols != cols or replay.sampled_inputs.sample_count == 0) return std.math.inf(f64);
    const sample_rows = replay.row_indices.len;
    if (sample_rows == 0) return std.math.inf(f64);

    const ref_weights = try allocator.alloc(f32, cols * sample_rows);
    defer allocator.free(ref_weights);
    const dq_weights = try allocator.alloc(f32, cols * sample_rows);
    defer allocator.free(dq_weights);

    var block_vals: [nvfp4_group_size]f32 = undefined;
    for (replay.row_indices, 0..) |row, sampled_row| {
        for (0..groups) |g| {
            const group_start = g * nvfp4_group_size;
            const block_base = row * cols + group_start;
            const max_abs = loadNvfp4BlockValuesAndMaxAbs(source, block_base, &block_vals);
            for (0..nvfp4_group_size) |i| {
                const col = group_start + i;
                ref_weights[col * sample_rows + sampled_row] = block_vals[i];
            }

            const block_scale = chooseNvfp4BlockScaleWithPolicy(
                block_vals[0..],
                max_abs,
                clip_multiplier,
                block_scale_policy,
            );
            const packed_scale = dtype.f32ToFp8E4M3(block_scale / global_scale);
            const scale_f32 = dtype.fp8e4m3ToF32(packed_scale) * global_scale;
            for (0..nvfp4_group_size) |i| {
                const col = group_start + i;
                const value = block_vals[i];
                const scaled = if (scale_f32 > 0.0) value / scale_f32 else 0.0;
                const q = nearestFp4E2m1Nibble(scaled);
                const dq = fp4E2m1NibbleToF32(q) * scale_f32;
                dq_weights[col * sample_rows + sampled_row] = dq;
            }
        }
    }

    if (evaluateNvfp4ForwardReplayMseMetal(
        allocator,
        replay.sampled_inputs.values,
        replay.sampled_inputs.sample_count,
        cols,
        sample_rows,
        ref_weights,
        dq_weights,
    )) |gpu_mse| {
        return gpu_mse;
    }

    return evaluateNvfp4ForwardReplayMseCpu(
        replay.sampled_inputs.values,
        replay.sampled_inputs.sample_count,
        cols,
        sample_rows,
        ref_weights,
        dq_weights,
    );
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
    if (std.mem.endsWith(u8, name, ".scales")) {
        const base = name[0 .. name.len - ".scales".len];
        return converted_weight_bases.contains(base);
    }
    if (std.mem.endsWith(u8, name, ".biases")) {
        const base = name[0 .. name.len - ".biases".len];
        return converted_weight_bases.contains(base);
    }
    if (std.mem.endsWith(u8, name, ".weight_bias")) {
        const base = name[0 .. name.len - ".weight_bias".len];
        return converted_weight_bases.contains(base);
    }
    if (std.mem.endsWith(u8, name, ".weight_global_scale")) {
        const base = name[0 .. name.len - ".weight_global_scale".len];
        return converted_weight_bases.contains(base);
    }
    if (std.mem.endsWith(u8, name, ".weight_packed")) {
        const base = name[0 .. name.len - ".weight_packed".len];
        return converted_weight_bases.contains(base);
    }
    return false;
}

fn convertedNvfp4BaseName(name: []const u8) []const u8 {
    if (std.mem.endsWith(u8, name, ".weight")) {
        return name[0 .. name.len - ".weight".len];
    }
    return name;
}

const Nvfp4PackedShape = struct {
    dims: [3]usize = .{ 0, 0, 0 },
    ndims: usize = 0,
    rows: usize = 0,
    cols: usize = 0,
};

fn packedShapeForWeight(weight: tensor.Tensor) !Nvfp4PackedShape {
    if (weight.n_dims == 2) {
        if (weight.shape[0] <= 0 or weight.shape[1] <= 0) return error.InvalidShape;
        const rows: usize = @intCast(weight.shape[0]);
        const cols: usize = @intCast(weight.shape[1]);
        return .{
            .dims = .{ rows, cols, 0 },
            .ndims = 2,
            .rows = rows,
            .cols = cols,
        };
    }
    if (weight.n_dims == 3) {
        if (weight.shape[0] <= 0 or weight.shape[1] <= 0 or weight.shape[2] <= 0) return error.InvalidShape;
        const experts: usize = @intCast(weight.shape[0]);
        const out_rows: usize = @intCast(weight.shape[1]);
        const cols: usize = @intCast(weight.shape[2]);
        const rows = std.math.mul(usize, experts, out_rows) catch return error.InvalidShape;
        return .{
            .dims = .{ experts, out_rows, cols },
            .ndims = 3,
            .rows = rows,
            .cols = cols,
        };
    }
    return error.InvalidShape;
}

fn shouldExcludeWeightByProfile(weight_name: []const u8, profile: @TypeOf((grouped_affine.ConvertOptions{}).profile)) bool {
    return shouldExcludeWeightByLmHead(weight_name, resolveNvfp4LmHeadQuantized(profile, false));
}

fn shouldExcludeWeightByLmHead(weight_name: []const u8, lm_head_quantized: bool) bool {
    if (!std.mem.endsWith(u8, weight_name, "lm_head.weight")) return false;
    return !lm_head_quantized;
}

fn shouldExcludeWeightByMoeRouting(weight_name: []const u8) bool {
    return std.mem.endsWith(u8, weight_name, ".mlp.gate.weight") or
        std.mem.endsWith(u8, weight_name, ".mlp.shared_expert_gate.weight");
}

fn shouldConvertDenseWeight(
    weight_name: []const u8,
    weight: tensor.Tensor,
    profile: @TypeOf((grouped_affine.ConvertOptions{}).profile),
) bool {
    return shouldConvertWeightWithLmHead(weight_name, weight, profile, resolveNvfp4LmHeadQuantized(profile, false));
}

fn shouldConvertDenseWeightWithLmHead(
    weight_name: []const u8,
    weight: tensor.Tensor,
    profile: @TypeOf((grouped_affine.ConvertOptions{}).profile),
    lm_head_quantized: bool,
) bool {
    return shouldConvertWeightWithLmHead(weight_name, weight, profile, lm_head_quantized);
}

fn isNvfp4MoeExpertTensorName(name: []const u8) bool {
    return std.mem.endsWith(u8, name, ".mlp.experts.gate_up_proj") or
        std.mem.endsWith(u8, name, ".mlp.experts.down_proj") or
        std.mem.endsWith(u8, name, ".mlp.experts.gate_proj") or
        std.mem.endsWith(u8, name, ".mlp.experts.up_proj");
}

fn shouldConvertNvfp4MoeExpertTensor(weight_name: []const u8, weight: tensor.Tensor) bool {
    if (!isNvfp4MoeExpertTensorName(weight_name)) return false;
    if (weight.n_dims != 3) return false;
    if (weight.shape[0] <= 0 or weight.shape[1] <= 0 or weight.shape[2] <= 0) return false;
    switch (weight.dtype) {
        .f32, .f16, .bf16 => {},
        else => return false,
    }
    const cols: usize = @intCast(weight.shape[2]);
    if ((cols % nvfp4_group_size) != 0) return false;
    return true;
}

fn shouldConvertWeightWithLmHead(
    weight_name: []const u8,
    weight: tensor.Tensor,
    profile: @TypeOf((grouped_affine.ConvertOptions{}).profile),
    lm_head_quantized: bool,
) bool {
    if (shouldExcludeWeightByMoeRouting(weight_name)) return false;
    if (shouldConvertNvfp4MoeExpertTensor(weight_name, weight)) return true;

    _ = profile;
    if (!std.mem.endsWith(u8, weight_name, ".weight")) return false;
    if (weight.n_dims != 2) return false;
    if (weight.shape[0] <= 0 or weight.shape[1] <= 0) return false;
    if (shouldExcludeWeightByLmHead(weight_name, lm_head_quantized)) return false;

    switch (weight.dtype) {
        .f32, .f16, .bf16 => {},
        else => return false,
    }

    const cols: usize = @intCast(weight.shape[1]);
    if ((cols % nvfp4_group_size) != 0) return false;
    return true;
}

fn isUnsupportedNvfp4MoeExpertTensor(name: []const u8, weight: tensor.Tensor) bool {
    if (!isNvfp4MoeExpertTensorName(name)) return false;
    return !shouldConvertNvfp4MoeExpertTensor(name, weight);
}

fn findUntiedLmHeadWeightName(names: [][]const u8) ?[]const u8 {
    for (names) |name| {
        if (std.mem.endsWith(u8, name, "lm_head.weight")) return name;
    }
    return null;
}

fn fp4E2m1NibbleToF32(nibble: u8) f32 {
    return fp4_codebook[nibble & 0x0F];
}

var fp8_e4m3_positive_max_bits: std.atomic.Value(u32) = .init(0);

fn maxFp8E4m3Positive() f32 {
    const cached_bits = fp8_e4m3_positive_max_bits.load(.acquire);
    if (cached_bits != 0) return @bitCast(cached_bits);

    var max_val: f32 = 0.0;
    var code: u16 = 0;
    while (code <= 255) : (code += 1) {
        const value = dtype.fp8e4m3ToF32(@intCast(code));
        if (std.math.isFinite(value) and value > max_val) max_val = value;
    }
    fp8_e4m3_positive_max_bits.store(@bitCast(max_val), .release);
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

    const base = nvfp4InitialGlobalScale(observed_max_scale);

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

fn nvfp4InitialGlobalScale(observed_max_scale: f32) f32 {
    if (!(observed_max_scale > 0.0) or !std.math.isFinite(observed_max_scale)) return 1.0;
    const fp8_max = maxFp8E4m3Positive();
    if (!(fp8_max > 0.0)) return 1.0;
    return @max(observed_max_scale / (fp8_max * 0.95), 1e-12);
}

fn isQualityRegression(
    profile: @TypeOf((grouped_affine.ConvertOptions{}).profile),
    baseline: f64,
    selected: f64,
) bool {
    if (profile != .good and profile != .best) return false;
    if (!std.math.isFinite(baseline) or !std.math.isFinite(selected)) return false;
    if (baseline <= 0.0) return false;
    // Allow tiny floating-point noise but reject any material regression.
    const tolerance = @max(@abs(baseline) * 1e-6, 1e-12);
    return selected > (baseline + tolerance);
}

inline fn useAdvancedNvfp4Search(profile: @TypeOf((grouped_affine.ConvertOptions{}).profile), update_round: u32) bool {
    _ = update_round;
    return profile == .custom and resolveNvfp4ReplayPolicy(profile) == .capture_required;
}

fn clipMultipliersForProfile(profile: @TypeOf((grouped_affine.ConvertOptions{}).profile), update_round: u32) []const f32 {
    _ = update_round;
    return switch (profile) {
        .good, .best => &[_]f32{ 0.97, 1.0, 1.03 },
        .custom => if (resolveNvfp4ReplayPolicy(profile) == .capture_required)
            &[_]f32{ 0.85, 0.95, 1.0, 1.05, 1.15 }
        else
            &[_]f32{1.0},
    };
}

fn globalScaleRefineMultipliersForProfile(profile: @TypeOf((grouped_affine.ConvertOptions{}).profile), update_round: u32) []const f32 {
    _ = update_round;
    return switch (profile) {
        .good, .best => &[_]f32{ 0.99, 1.0, 1.01 },
        .custom => if (resolveNvfp4ReplayPolicy(profile) == .capture_required)
            &[_]f32{ 0.9, 1.0, 1.1 }
        else
            &[_]f32{1.0},
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
    all_block_scales: ?[]f32,
    block_scale_policy: Nvfp4BlockScalePolicy,
    pool: *parallel.ThreadPool,
    block_progress: ?*Nvfp4BlockProgress,
) void {
    sampled_count.* = 0;
    sampled_seen.* = 0;
    observed_max_scale.* = 0.0;

    const total_blocks = std.math.mul(usize, rows, groups) catch 0;
    const kept_samples = @min(total_blocks, global_scale_sample_limit);
    const sample_tail_start = total_blocks - kept_samples;
    sampled_count.* = kept_samples;
    sampled_seen.* = total_blocks;

    const SampleScaleContext = struct {
        source: DenseWeightView,
        cols: usize,
        groups: usize,
        clip_multiplier: f32,
        group_importance: ?[]const f32,
        block_scale_policy: Nvfp4BlockScalePolicy,
        sample_tail_start: usize,
        sampled_scales: *[global_scale_sample_limit]f32,
        sampled_importance: *[global_scale_sample_limit]f32,
        observed_max_scale: *f32,
        all_block_scales: ?[]f32,
        max_mutex: std.Thread.Mutex = .{},
    };
    const SampleScaleFn = struct {
        fn run(start_row: usize, end_row: usize, ctx: *SampleScaleContext) void {
            var block_vals: [nvfp4_group_size]f32 = undefined;
            var local_max: f32 = 0.0;
            for (start_row..end_row) |r| {
                for (0..ctx.groups) |g| {
                    const group_start = g * nvfp4_group_size;
                    const block_base = r * ctx.cols + group_start;
                    const max_abs = loadNvfp4BlockValuesAndMaxAbs(ctx.source, block_base, &block_vals);

                    const block_scale = chooseNvfp4BlockScaleWithPolicy(
                        block_vals[0..],
                        max_abs,
                        ctx.clip_multiplier,
                        ctx.block_scale_policy,
                    );
                    local_max = @max(local_max, block_scale);
                    const block_index = r * ctx.groups + g;
                    if (ctx.all_block_scales) |block_scales| {
                        block_scales[block_index] = block_scale;
                    }
                    if (block_index >= ctx.sample_tail_start) {
                        const sample_idx = block_index - ctx.sample_tail_start;
                        if (sample_idx < global_scale_sample_limit) {
                            ctx.sampled_scales[sample_idx] = block_scale;
                            ctx.sampled_importance[sample_idx] = if (ctx.group_importance) |weights| weights[g] else 1.0;
                        }
                    }
                }
            }

            if (local_max > 0.0) {
                ctx.max_mutex.lock();
                defer ctx.max_mutex.unlock();
                ctx.observed_max_scale.* = @max(ctx.observed_max_scale.*, local_max);
            }
        }
    };

    if (rows > 0 and groups > 0) {
        var ctx = SampleScaleContext{
            .source = source,
            .cols = cols,
            .groups = groups,
            .clip_multiplier = clip_multiplier,
            .group_importance = group_importance,
            .block_scale_policy = block_scale_policy,
            .sample_tail_start = sample_tail_start,
            .sampled_scales = sampled_scales,
            .sampled_importance = sampled_importance,
            .observed_max_scale = observed_max_scale,
            .all_block_scales = all_block_scales,
        };
        const work_blocks = std.math.mul(usize, rows, groups) catch 0;
        if (work_blocks < nvfp4_parallel_min_blocks) {
            SampleScaleFn.run(0, rows, &ctx);
        } else {
            pool.parallelForCompute(rows, SampleScaleFn.run, &ctx);
        }
    }

    if (block_progress) |progress_state| {
        const done_blocks: u64 = @intCast(total_blocks);
        progress_state.bump(done_blocks);
    }
}

fn collectSampledBlockScalesMetal(
    allocator: std.mem.Allocator,
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
    all_block_scales: []f32,
    block_progress: ?*Nvfp4BlockProgress,
) !void {
    const total_blocks = std.math.mul(usize, rows, groups) catch return error.InvalidShape;
    if (all_block_scales.len != total_blocks) return error.InvalidShape;

    sampled_count.* = 0;
    sampled_seen.* = total_blocks;
    observed_max_scale.* = 0.0;

    const graph = compute.metal.graph;
    const weight_shape = [_]i64{ @intCast(rows), @intCast(cols) };

    graph.beginForwardGraphBuild();
    const weight_handle = switch (source) {
        .f32 => |slice| graph.createBorrowedArrayF32Unaligned(slice.ptr, slice.len, &weight_shape),
        .f16 => |slice| graph.createBorrowedArrayF16Unaligned(slice.ptr, slice.len, &weight_shape),
        .bf16 => |slice| graph.createBorrowedArrayBF16Unaligned(slice.ptr, slice.len, &weight_shape),
    };
    if (weight_handle == null) return error.InvalidConfig;
    defer graph.freeArray(weight_handle);

    const block_scales_handle = graph.mlx_lazy_nvfp4_block_scales(weight_handle, clip_multiplier);
    if (block_scales_handle == null) return error.InvalidConfig;
    defer graph.freeArray(block_scales_handle);

    var eval_handles = [_]compute.metal.graph.ArrayHandle{block_scales_handle};
    graph.eval(&eval_handles);
    graph.copyToHost(block_scales_handle, all_block_scales);
    _ = allocator;

    const kept_samples = @min(total_blocks, global_scale_sample_limit);
    const sample_tail_start = total_blocks - kept_samples;
    sampled_count.* = kept_samples;
    for (all_block_scales, 0..) |raw_scale, block_idx| {
        const block_scale = if (raw_scale > 0.0 and std.math.isFinite(raw_scale)) raw_scale else 0.0;
        observed_max_scale.* = @max(observed_max_scale.*, block_scale);
        if (block_idx < sample_tail_start) continue;
        const sample_idx = block_idx - sample_tail_start;
        if (sample_idx >= global_scale_sample_limit) continue;
        sampled_scales[sample_idx] = block_scale;
        const group_idx = if (groups > 0) block_idx % groups else 0;
        sampled_importance[sample_idx] = if (group_importance) |weights| weights[group_idx] else 1.0;
    }

    if (block_progress) |progress_state| {
        const done_blocks: u64 = @intCast(total_blocks);
        progress_state.bump(done_blocks);
    }
}

fn estimateNvfp4ForwardProxyMse(
    source: DenseWeightView,
    rows: usize,
    cols: usize,
    groups: usize,
    global_scale: f32,
    clip_multiplier: f32,
    block_scale_policy: Nvfp4BlockScalePolicy,
    group_importance: ?[]const f32,
    pool: *parallel.ThreadPool,
    block_progress: ?*Nvfp4BlockProgress,
) f64 {
    var out = [_]f64{std.math.inf(f64)};
    const scales = [_]f32{global_scale};
    estimateNvfp4ForwardProxyMseMultiScales(
        source,
        rows,
        cols,
        groups,
        scales[0..],
        clip_multiplier,
        block_scale_policy,
        group_importance,
        out[0..],
        pool,
        block_progress,
    );
    return out[0];
}

fn estimateNvfp4ForwardProxyMseMultiScales(
    source: DenseWeightView,
    rows: usize,
    cols: usize,
    groups: usize,
    global_scales: []const f32,
    clip_multiplier: f32,
    block_scale_policy: Nvfp4BlockScalePolicy,
    group_importance: ?[]const f32,
    out_mse: []f64,
    pool: *parallel.ThreadPool,
    block_progress: ?*Nvfp4BlockProgress,
) void {
    if (global_scales.len == 0 or out_mse.len != global_scales.len or global_scales.len > nvfp4_forward_eval_max_scales) {
        @memset(out_mse, std.math.inf(f64));
        return;
    }
    @memset(out_mse, std.math.inf(f64));

    var valid_scales = [_]bool{false} ** nvfp4_forward_eval_max_scales;
    var scale_count: usize = 0;
    for (global_scales, 0..) |global_scale, idx| {
        if (idx >= nvfp4_forward_eval_max_scales) break;
        const is_valid = (global_scale > 0.0) and std.math.isFinite(global_scale);
        valid_scales[idx] = is_valid;
        if (is_valid) scale_count = idx + 1;
    }
    if (scale_count == 0) return;

    const ProxyEvalContext = struct {
        source: DenseWeightView,
        cols: usize,
        groups: usize,
        global_scales: []const f32,
        valid_scales: []const bool,
        scale_count: usize,
        clip_multiplier: f32,
        block_scale_policy: Nvfp4BlockScalePolicy,
        group_importance: ?[]const f32,
        total_err: [nvfp4_forward_eval_max_scales]f64 = [_]f64{0.0} ** nvfp4_forward_eval_max_scales,
        total_weight: f64 = 0.0,
        reduce_mutex: std.Thread.Mutex = .{},
    };
    const ProxyEvalFn = struct {
        fn run(start_row: usize, end_row: usize, ctx: *ProxyEvalContext) void {
            var local_err = [_]f64{0.0} ** nvfp4_forward_eval_max_scales;
            var local_weight: f64 = 0.0;
            var block_vals: [nvfp4_group_size]f32 = undefined;

            for (start_row..end_row) |r| {
                for (0..ctx.groups) |g| {
                    const group_start = g * nvfp4_group_size;
                    const block_base = r * ctx.cols + group_start;
                    const max_abs = loadNvfp4BlockValuesAndMaxAbs(ctx.source, block_base, &block_vals);

                    const block_scale = chooseNvfp4BlockScaleWithPolicy(
                        block_vals[0..],
                        max_abs,
                        ctx.clip_multiplier,
                        ctx.block_scale_policy,
                    );
                    const weight = if (ctx.group_importance) |weights| weights[g] else 1.0;
                    const stable_weight = if (weight > 0.0 and std.math.isFinite(weight)) weight else activation_importance_min_weight;

                    for (0..ctx.scale_count) |scale_idx| {
                        if (!ctx.valid_scales[scale_idx]) continue;
                        const global_scale = ctx.global_scales[scale_idx];
                        const packed_scale = dtype.f32ToFp8E4M3(block_scale / global_scale);
                        const scale_f32 = dtype.fp8e4m3ToF32(packed_scale) * global_scale;

                        for (0..nvfp4_group_size) |i| {
                            const value = block_vals[i];
                            const scaled = if (scale_f32 > 0.0) value / scale_f32 else 0.0;
                            const q = nearestFp4E2m1Nibble(scaled);
                            const dq = fp4E2m1NibbleToF32(q) * scale_f32;
                            const err = @as(f64, @floatCast(value - dq));
                            local_err[scale_idx] += err * err * @as(f64, stable_weight);
                        }
                    }
                    local_weight += @as(f64, stable_weight) * @as(f64, @floatFromInt(nvfp4_group_size));
                }
            }

            ctx.reduce_mutex.lock();
            defer ctx.reduce_mutex.unlock();
            for (0..ctx.scale_count) |scale_idx| {
                ctx.total_err[scale_idx] += local_err[scale_idx];
            }
            ctx.total_weight += local_weight;
        }
    };

    var ctx = ProxyEvalContext{
        .source = source,
        .cols = cols,
        .groups = groups,
        .global_scales = global_scales,
        .valid_scales = valid_scales[0..],
        .scale_count = scale_count,
        .clip_multiplier = clip_multiplier,
        .block_scale_policy = block_scale_policy,
        .group_importance = group_importance,
    };
    if (rows > 0 and groups > 0) {
        const work_blocks = std.math.mul(usize, rows, groups) catch 0;
        if (work_blocks < nvfp4_parallel_min_blocks) {
            ProxyEvalFn.run(0, rows, &ctx);
        } else {
            pool.parallelForCompute(rows, ProxyEvalFn.run, &ctx);
        }
    }
    const total_weight = ctx.total_weight;
    if (block_progress) |progress_state| {
        const done_blocks = std.math.mul(u64, @as(u64, @intCast(rows)), @as(u64, @intCast(groups))) catch std.math.maxInt(u64);
        progress_state.bump(done_blocks);
    }
    if (!(total_weight > 0.0) or !std.math.isFinite(total_weight)) return;
    for (0..scale_count) |scale_idx| {
        if (!valid_scales[scale_idx]) continue;
        out_mse[scale_idx] = ctx.total_err[scale_idx] / total_weight;
    }
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

fn shouldUseNvfp4MetalPack(
    allocator: std.mem.Allocator,
    profile: @TypeOf((grouped_affine.ConvertOptions{}).profile),
) !bool {
    _ = profile;
    if (!shouldEnableNvfp4MetalPack()) return false;

    ensureMlxMetallibColocatedForNvfp4();
    if (!ensureMlxMetallibForNvfp4Replay(allocator)) {
        if (std.posix.getenv("BACKEND")) |raw| {
            if (std.ascii.eqlIgnoreCase(raw, "metal")) return error.InvalidConfig;
        }
        return false;
    }
    if (!compute.metal.isAvailable()) {
        if (std.posix.getenv("BACKEND")) |raw| {
            if (std.ascii.eqlIgnoreCase(raw, "metal")) return error.InvalidConfig;
        }
        return false;
    }
    return true;
}

fn packNvfp4RowsWithMetal(
    allocator: std.mem.Allocator,
    source: DenseWeightView,
    rows: usize,
    cols: usize,
    groups: usize,
    global_scale: f32,
    block_scale_cache: []const f32,
    packed_scales: []u8,
    packed_bytes: []u8,
    convert_pool: *parallel.ThreadPool,
) !void {
    if (!(global_scale > 0.0) or !std.math.isFinite(global_scale)) return error.InvalidConfig;
    const scale_len = std.math.mul(usize, rows, groups) catch return error.InvalidShape;
    const packed_len = std.math.mul(usize, rows, cols / 2) catch return error.InvalidShape;
    if (packed_scales.len != scale_len or packed_bytes.len != packed_len) return error.InvalidShape;
    if (block_scale_cache.len != scale_len) return error.InvalidShape;

    const decoded_scales = try allocator.alloc(f32, scale_len);
    defer allocator.free(decoded_scales);

    const EncodeScalesContext = struct {
        rows: usize,
        groups: usize,
        global_scale: f32,
        block_scale_cache: []const f32,
        packed_scales: []u8,
        decoded_scales: []f32,
    };
    const EncodeScalesFn = struct {
        fn run(start_row: usize, end_row: usize, ctx: *EncodeScalesContext) void {
            for (start_row..end_row) |r| {
                for (0..ctx.groups) |g| {
                    const block_index = r * ctx.groups + g;
                    const block_scale = ctx.block_scale_cache[block_index];
                    const packed_scale = dtype.f32ToFp8E4M3(block_scale / ctx.global_scale);
                    ctx.packed_scales[block_index] = packed_scale;
                    ctx.decoded_scales[block_index] = dtype.fp8e4m3ToF32(packed_scale) * ctx.global_scale;
                }
            }
        }
    };
    if (rows > 0 and groups > 0) {
        var encode_ctx = EncodeScalesContext{
            .rows = rows,
            .groups = groups,
            .global_scale = global_scale,
            .block_scale_cache = block_scale_cache,
            .packed_scales = packed_scales,
            .decoded_scales = decoded_scales,
        };
        const work_blocks = std.math.mul(usize, rows, groups) catch 0;
        if (work_blocks < nvfp4_parallel_min_blocks) {
            EncodeScalesFn.run(0, rows, &encode_ctx);
        } else {
            convert_pool.parallelForCompute(rows, EncodeScalesFn.run, &encode_ctx);
        }
    }

    const graph = compute.metal.graph;
    const weight_shape = [_]i64{ @intCast(rows), @intCast(cols) };
    const scale_shape = [_]i64{ @intCast(rows), @intCast(groups) };

    graph.beginForwardGraphBuild();
    const weight_handle = switch (source) {
        .f32 => |slice| graph.createBorrowedArrayF32Unaligned(slice.ptr, slice.len, &weight_shape),
        .f16 => |slice| graph.createBorrowedArrayF16Unaligned(slice.ptr, slice.len, &weight_shape),
        .bf16 => |slice| graph.createBorrowedArrayBF16Unaligned(slice.ptr, slice.len, &weight_shape),
    };
    if (weight_handle == null) return error.InvalidConfig;
    defer graph.freeArray(weight_handle);

    const scale_handle = graph.createBorrowedArrayF32Unaligned(decoded_scales.ptr, decoded_scales.len, &scale_shape);
    if (scale_handle == null) return error.InvalidConfig;
    defer graph.freeArray(scale_handle);

    const packed_handle = graph.mlx_lazy_nvfp4_pack(weight_handle, scale_handle);
    if (packed_handle == null) return error.InvalidConfig;
    defer graph.freeArray(packed_handle);

    var eval_handles = [_]compute.metal.graph.ArrayHandle{packed_handle};
    graph.eval(&eval_handles);
    graph.copyU8ToHost(packed_handle, packed_bytes);
}

inline fn loadNvfp4BlockValuesAndMaxAbs(
    source: DenseWeightView,
    block_base: usize,
    out: *[nvfp4_group_size]f32,
) f32 {
    var max_abs: f32 = 0.0;
    switch (source) {
        .f32 => |slice| {
            for (0..nvfp4_group_size) |i| {
                const value = slice[block_base + i];
                out[i] = value;
                max_abs = @max(max_abs, @abs(value));
            }
        },
        .f16 => |slice| {
            for (0..nvfp4_group_size) |i| {
                const value = dtype.fp16ToF32(slice[block_base + i]);
                out[i] = value;
                max_abs = @max(max_abs, @abs(value));
            }
        },
        .bf16 => |slice| {
            for (0..nvfp4_group_size) |i| {
                const value = dtype.bf16ToF32(slice[block_base + i]);
                out[i] = value;
                max_abs = @max(max_abs, @abs(value));
            }
        },
    }
    return max_abs;
}

fn nearestFp4E2m1Nibble(value: f32) u8 {
    if (!std.math.isFinite(value)) return 0;

    // Exact nearest-neighbor thresholds for the E2M1 codebook with tie
    // behavior matching the previous linear scan implementation.
    const mag = @abs(value);
    const level_idx: u8 = if (mag <= 0.25)
        0
    else if (mag <= 0.75)
        1
    else if (mag <= 1.25)
        2
    else if (mag <= 1.75)
        3
    else if (mag <= 2.5)
        4
    else if (mag <= 3.5)
        5
    else if (mag <= 5.0)
        6
    else
        7;

    if (level_idx == 0 or value >= 0.0) return level_idx;
    return switch (level_idx) {
        1 => 9,
        2 => 10,
        3 => 11,
        4 => 12,
        5 => 13,
        6 => 14,
        else => 15,
    };
}

fn nearestFp4E2m1NibbleLinearScan(value: f32) u8 {
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
            .converted_weight, .converted_scale, .converted_scale_2, .converted_input_scale, .preserved_bf16_weight, .preserved_mxfp8_weight, .preserved_mxfp8_block_scale, .dense_bf16_rhs_weight => {
                const source_name = spec.source_weight_name orelse return error.InvalidConfig;
                const source = try st.getTensor(source_name, null);
                const packed_shape = try packedShapeForWeight(source);
                const rows = packed_shape.rows;
                const cols = packed_shape.cols;
                const groups = cols / nvfp4_group_size;
                switch (spec.kind) {
                    .converted_weight => {
                        const packed_cols = cols / 2;
                        const data_size = std.math.mul(usize, rows, packed_cols) catch return error.InvalidConfig;
                        if (packed_shape.ndims == 2) {
                            try header_buf.writer(allocator).print(
                                "\"{s}\":{{\"dtype\":\"U8\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}",
                                .{ spec.name, packed_shape.dims[0], packed_cols, data_offset, data_offset + data_size },
                            );
                        } else if (packed_shape.ndims == 3) {
                            try header_buf.writer(allocator).print(
                                "\"{s}\":{{\"dtype\":\"U8\",\"shape\":[{d},{d},{d}],\"data_offsets\":[{d},{d}]}}",
                                .{ spec.name, packed_shape.dims[0], packed_shape.dims[1], packed_cols, data_offset, data_offset + data_size },
                            );
                        } else return error.InvalidConfig;
                        data_offset += data_size;
                    },
                    .converted_scale => {
                        const data_size = std.math.mul(usize, rows, groups) catch return error.InvalidConfig;
                        if (packed_shape.ndims == 2) {
                            try header_buf.writer(allocator).print(
                                "\"{s}\":{{\"dtype\":\"F8_E4M3\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}",
                                .{ spec.name, packed_shape.dims[0], groups, data_offset, data_offset + data_size },
                            );
                        } else if (packed_shape.ndims == 3) {
                            try header_buf.writer(allocator).print(
                                "\"{s}\":{{\"dtype\":\"F8_E4M3\",\"shape\":[{d},{d},{d}],\"data_offsets\":[{d},{d}]}}",
                                .{ spec.name, packed_shape.dims[0], packed_shape.dims[1], groups, data_offset, data_offset + data_size },
                            );
                        } else return error.InvalidConfig;
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
                    .preserved_bf16_weight => {
                        if (packed_shape.ndims != 2) return error.InvalidConfig;
                        const data_size = std.math.mul(usize, rows, cols) catch return error.InvalidConfig;
                        const byte_size = std.math.mul(usize, data_size, @sizeOf(u16)) catch return error.InvalidConfig;
                        try header_buf.writer(allocator).print(
                            "\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}",
                            .{ spec.name, packed_shape.dims[0], packed_shape.dims[1], data_offset, data_offset + byte_size },
                        );
                        data_offset += byte_size;
                    },
                    .preserved_mxfp8_weight => {
                        if (packed_shape.ndims != 2) return error.InvalidConfig;
                        const data_size = std.math.mul(usize, rows, cols) catch return error.InvalidConfig;
                        try header_buf.writer(allocator).print(
                            "\"{s}\":{{\"dtype\":\"F8_E4M3\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}",
                            .{ spec.name, packed_shape.dims[0], packed_shape.dims[1], data_offset, data_offset + data_size },
                        );
                        data_offset += data_size;
                    },
                    .preserved_mxfp8_block_scale => {
                        if (packed_shape.ndims != 2) return error.InvalidConfig;
                        const scale_cols = (cols + 31) / 32;
                        const data_size = std.math.mul(usize, rows, scale_cols) catch return error.InvalidConfig;
                        try header_buf.writer(allocator).print(
                            "\"{s}\":{{\"dtype\":\"U8\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}",
                            .{ spec.name, packed_shape.dims[0], scale_cols, data_offset, data_offset + data_size },
                        );
                        data_offset += data_size;
                    },
                    .dense_bf16_rhs_weight => {
                        const data_size = std.math.mul(usize, rows, cols) catch return error.InvalidConfig;
                        const byte_size = std.math.mul(usize, data_size, @sizeOf(u16)) catch return error.InvalidConfig;
                        try header_buf.writer(allocator).print(
                            "\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}",
                            .{ spec.name, cols, rows, data_offset, data_offset + byte_size },
                        );
                        data_offset += byte_size;
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
    quality: Nvfp4QualityStats,
    owns_buffers: bool = true,

    fn deinit(self: *PackedNvfp4Data, allocator: std.mem.Allocator) void {
        if (!self.owns_buffers) return;
        allocator.free(self.packed_weight);
        allocator.free(self.packed_scale);
    }
};

const PreservedMxfp8Data = struct {
    source_weight_name: []const u8,
    fp8_weight: []u8,
    block_scales: []u8,
    owns_buffers: bool = true,

    fn deinit(self: *PreservedMxfp8Data, allocator: std.mem.Allocator) void {
        if (!self.owns_buffers) return;
        allocator.free(self.fp8_weight);
        allocator.free(self.block_scales);
    }
};

const ExistingNvfp4Tensor = struct {
    packed_weight: []const u8,
    packed_scale: []const u8,
    weight_scale_2: f32,
    input_scale: f32,
    kl_divergence: f64,
};

const Nvfp4QualityMetric = enum {
    scale_mse,
    proxy_mse,
    forward_mse,
    hybrid_mse,
};

const Nvfp4QualityStats = struct {
    metric: Nvfp4QualityMetric = .scale_mse,
    baseline: f64 = std.math.nan(f64),
    selected: f64 = std.math.nan(f64),
    forward_baseline: f64 = std.math.nan(f64),
    forward_selected: f64 = std.math.nan(f64),
    improvement_pct: f32 = 0.0,
    clip_multiplier: f32 = 1.0,
    global_scale: f32 = 1.0,
    used_activation_importance: bool = false,
    kl_divergence: f64 = std.math.nan(f64),
};

const Nvfp4QualitySummary = struct {
    profile: @TypeOf((grouped_affine.ConvertOptions{}).profile),
    tensors: usize = 0,
    lm_head_quantized: bool = false,
    small_model_preserve_enabled: bool = false,
    preserve_blocks_selected: u32 = 0,
    source_dense_weight_bytes: u64 = 0,
    progress_gain_pct: f32 = 0.0,
    regressions: usize = 0,
    activation_tensors: usize = 0,
    metric_scale: usize = 0,
    metric_proxy: usize = 0,
    metric_forward: usize = 0,
    metric_hybrid: usize = 0,
    weighted_baseline: f64 = 0.0,
    weighted_selected: f64 = 0.0,
    weighted_total: f64 = 0.0,
    forward_weighted_baseline: f64 = 0.0,
    forward_weighted_selected: f64 = 0.0,
    forward_weighted_total: f64 = 0.0,
    forward_tensors: usize = 0,
    kl_weighted_sum: f64 = 0.0,
    kl_weighted_total: f64 = 0.0,
    min_kl_divergence: f64 = std.math.inf(f64),
    max_kl_divergence: f64 = 0.0,
    mean_improvement_sum: f64 = 0.0,
    min_improvement_pct: f32 = std.math.inf(f32),
    max_improvement_pct: f32 = -std.math.inf(f32),

    fn add(self: *Nvfp4QualitySummary, quality: Nvfp4QualityStats, weight: f64) void {
        self.tensors += 1;
        self.mean_improvement_sum += quality.improvement_pct;
        self.min_improvement_pct = @min(self.min_improvement_pct, quality.improvement_pct);
        self.max_improvement_pct = @max(self.max_improvement_pct, quality.improvement_pct);
        if (quality.used_activation_importance) self.activation_tensors += 1;
        switch (quality.metric) {
            .scale_mse => self.metric_scale += 1,
            .proxy_mse => self.metric_proxy += 1,
            .forward_mse => self.metric_forward += 1,
            .hybrid_mse => self.metric_hybrid += 1,
        }
        if (std.math.isFinite(quality.baseline) and
            std.math.isFinite(quality.selected) and
            quality.baseline > 0.0)
        {
            self.weighted_baseline += quality.baseline * weight;
            self.weighted_selected += quality.selected * weight;
            self.weighted_total += weight;
            if (quality.selected > quality.baseline) self.regressions += 1;
        }
        if (std.math.isFinite(quality.forward_baseline) and
            std.math.isFinite(quality.forward_selected) and
            quality.forward_baseline > 0.0)
        {
            self.forward_weighted_baseline += quality.forward_baseline * weight;
            self.forward_weighted_selected += quality.forward_selected * weight;
            self.forward_weighted_total += weight;
            self.forward_tensors += 1;
        }
        if (std.math.isFinite(quality.kl_divergence) and quality.kl_divergence >= 0.0) {
            self.kl_weighted_sum += quality.kl_divergence * weight;
            self.kl_weighted_total += weight;
            self.min_kl_divergence = @min(self.min_kl_divergence, quality.kl_divergence);
            self.max_kl_divergence = @max(self.max_kl_divergence, quality.kl_divergence);
        }
    }

    fn globalImprovementPct(self: Nvfp4QualitySummary) f32 {
        if (!(self.weighted_total > 0.0) or !(self.weighted_baseline > 0.0)) return 0.0;
        return @floatCast(((self.weighted_baseline - self.weighted_selected) / self.weighted_baseline) * 100.0);
    }

    fn meanImprovementPct(self: Nvfp4QualitySummary) f32 {
        if (self.tensors == 0) return 0.0;
        return @floatCast(self.mean_improvement_sum / @as(f64, @floatFromInt(self.tensors)));
    }

    fn meanKlDivergence(self: Nvfp4QualitySummary) f64 {
        if (!(self.kl_weighted_total > 0.0)) return std.math.nan(f64);
        return self.kl_weighted_sum / self.kl_weighted_total;
    }

    fn modelStateNormalizedMseRatio(self: Nvfp4QualitySummary) f32 {
        if (!(self.forward_weighted_total > 0.0) or !(self.forward_weighted_baseline > 0.0)) {
            if (!(self.weighted_total > 0.0) or !(self.weighted_baseline > 0.0)) return 1.0;
            if (!std.math.isFinite(self.weighted_baseline) or !std.math.isFinite(self.weighted_selected)) return 1.0;
            return @floatCast(self.weighted_selected / self.weighted_baseline);
        }
        if (!std.math.isFinite(self.forward_weighted_baseline) or !std.math.isFinite(self.forward_weighted_selected)) return 1.0;
        return @floatCast(self.forward_weighted_selected / self.forward_weighted_baseline);
    }

    fn modelStateRelativeMseReductionPct(self: Nvfp4QualitySummary) f32 {
        const ratio = self.modelStateNormalizedMseRatio();
        return @floatCast((1.0 - ratio) * 100.0);
    }
};

fn shouldSampleNvfp4ForwardMetricTensor(source_weight_name: []const u8) bool {
    if (nvfp4_forward_metric_tensor_sample_mod <= 1) return true;
    const h = mix64(std.hash.Wyhash.hash(0, source_weight_name));
    return (h % nvfp4_forward_metric_tensor_sample_mod) == 0;
}

fn trimNvfp4ForwardReplayInputsForMetric(full: *const Nvfp4ForwardReplayInputs) ?Nvfp4ForwardReplayInputs {
    const sample_count = @min(full.sampled_inputs.sample_count, nvfp4_forward_metric_input_cap);
    const row_count = @min(full.row_indices.len, nvfp4_forward_metric_row_cap);
    if (sample_count == 0 or row_count == 0) return null;
    var sampled = full.sampled_inputs;
    sampled.sample_count = sample_count;
    return .{
        .sampled_inputs = sampled,
        .row_indices = full.row_indices[0..row_count],
        .from_capture = full.from_capture,
    };
}

fn populateNvfp4ForwardMseQuality(
    allocator: std.mem.Allocator,
    source_weight_name: []const u8,
    source: DenseWeightView,
    rows: usize,
    cols: usize,
    groups: usize,
    clip_multiplier: f32,
    block_scale_policy: Nvfp4BlockScalePolicy,
    activation_cache: ?*const calibration_capture.LayerActivationCache,
    activation_sample_count: usize,
    calib_seed: u64,
    baseline_global_scale: f32,
    selected_global_scale: f32,
    quality: *Nvfp4QualityStats,
) void {
    if (!shouldSampleNvfp4ForwardMetricTensor(source_weight_name)) return;
    if (!(baseline_global_scale > 0.0) or !std.math.isFinite(baseline_global_scale)) return;
    if (!(selected_global_scale > 0.0) or !std.math.isFinite(selected_global_scale)) return;
    const replay_full = buildNvfp4ForwardReplayInputs(
        allocator,
        source_weight_name,
        source,
        rows,
        cols,
        activation_cache,
        activation_sample_count,
        calib_seed,
    ) catch null orelse return;
    var replay_owner = replay_full;
    defer replay_owner.deinit(allocator);
    const replay = trimNvfp4ForwardReplayInputsForMetric(&replay_owner) orelse return;
    const baseline_forward = estimateNvfp4ForwardReplayMse(
        allocator,
        source,
        cols,
        groups,
        baseline_global_scale,
        clip_multiplier,
        block_scale_policy,
        &replay,
    ) catch std.math.nan(f64);
    const selected_forward = estimateNvfp4ForwardReplayMse(
        allocator,
        source,
        cols,
        groups,
        selected_global_scale,
        clip_multiplier,
        block_scale_policy,
        &replay,
    ) catch std.math.nan(f64);
    if (!std.math.isFinite(baseline_forward) or !std.math.isFinite(selected_forward) or !(baseline_forward > 0.0)) return;
    quality.forward_baseline = baseline_forward;
    quality.forward_selected = selected_forward;
}

fn nvfp4ProfileName(profile: @TypeOf((grouped_affine.ConvertOptions{}).profile)) []const u8 {
    return switch (profile) {
        .best, .good => "good",
        .custom => "custom",
    };
}

fn shouldRetuneTensorInUpdate(source_weight_name: []const u8, update_round: u32) bool {
    if (update_round == 0) return true;
    const hashed = mix64(std.hash.Wyhash.hash(0, source_weight_name));
    const slot = @as(u32, @intCast(hashed % nvfp4_update_round_slots));
    const round_slot = (update_round - 1) % nvfp4_update_round_slots;
    return slot == round_slot;
}

fn nvfp4TensorBlockWork(
    weight: tensor.Tensor,
    profile: @TypeOf((grouped_affine.ConvertOptions{}).profile),
    update_round: u32,
    use_advanced_search: bool,
) u64 {
    const packed_shape = packedShapeForWeight(weight) catch return 0;
    const rows: u64 = @intCast(packed_shape.rows);
    const cols: u64 = @intCast(packed_shape.cols);
    if (cols == 0 or (cols % nvfp4_group_size) != 0) return 0;
    const groups = cols / nvfp4_group_size;
    const blocks_per_pass = std.math.mul(u64, rows, groups) catch return std.math.maxInt(u64);
    const clip_count: u64 = if (profile == .custom)
        1
    else
        @intCast(clipMultipliersForProfile(profile, update_round).len);
    const pass_count: u64 = if (use_advanced_search)
        (clip_count * 2) + 2
    else
        2;
    return std.math.mul(u64, blocks_per_pass, pass_count) catch std.math.maxInt(u64);
}

fn nvfp4ImprovementPct(baseline: f64, selected: f64) f32 {
    if (!(baseline > 0.0) or !std.math.isFinite(baseline) or !std.math.isFinite(selected)) return 0.0;
    const pct = ((baseline - selected) / baseline) * 100.0;
    return @floatCast(pct);
}

fn nvfp4ForwardMseReductionPct(quality: Nvfp4QualityStats) ?f32 {
    if (!(quality.forward_baseline > 0.0) or
        !std.math.isFinite(quality.forward_baseline) or
        !std.math.isFinite(quality.forward_selected))
    {
        return null;
    }
    return nvfp4ImprovementPct(quality.forward_baseline, quality.forward_selected);
}

fn fp4HistogramBinScaled(value: f32) usize {
    const v = std.math.clamp(value, -6.0, 6.0);
    if (v < -5.0) return 0;
    if (v < -3.5) return 1;
    if (v < -2.5) return 2;
    if (v < -1.75) return 3;
    if (v < -1.25) return 4;
    if (v < -0.75) return 5;
    if (v < -0.25) return 6;
    if (v < 0.25) return 7;
    if (v < 0.75) return 8;
    if (v < 1.25) return 9;
    if (v < 1.75) return 10;
    if (v < 2.5) return 11;
    if (v < 3.5) return 12;
    if (v < 5.0) return 13;
    return 14;
}

fn sampledBlockScaledHistogramKl(
    ref_vals: *const [nvfp4_group_size]f32,
    dq_vals: *const [nvfp4_group_size]f32,
    scale_f32: f32,
) f64 {
    if (!(scale_f32 > 0.0) or !std.math.isFinite(scale_f32)) return std.math.nan(f64);
    const bins = 15;
    var ref_counts = [_]f64{0.0} ** bins;
    var dq_counts = [_]f64{0.0} ** bins;
    for (0..nvfp4_group_size) |i| {
        const ref_u = ref_vals[i] / scale_f32;
        const dq_u = dq_vals[i] / scale_f32;
        ref_counts[fp4HistogramBinScaled(ref_u)] += 1.0;
        dq_counts[fp4HistogramBinScaled(dq_u)] += 1.0;
    }

    const alpha = nvfp4_kl_laplace_alpha;
    const bins_f: f64 = @floatFromInt(bins);
    const denom_ref = @as(f64, @floatFromInt(nvfp4_group_size)) + alpha * bins_f;
    const denom_dq = @as(f64, @floatFromInt(nvfp4_group_size)) + alpha * bins_f;
    if (!(denom_ref > 0.0) or !(denom_dq > 0.0)) return std.math.nan(f64);

    var kl: f64 = 0.0;
    for (0..bins) |idx| {
        const p = (ref_counts[idx] + alpha) / denom_ref;
        const q = (dq_counts[idx] + alpha) / denom_dq;
        kl += p * @log(p / q);
    }
    return if (std.math.isFinite(kl) and kl >= 0.0) kl else std.math.nan(f64);
}

fn nvfp4KlSampleSeedForTensor(source_weight_name: []const u8) u64 {
    // Keep KL sampling deterministic across runs so conversions are comparable.
    return mix64(0x4e564650344b4c31 ^ std.hash.Wyhash.hash(0, source_weight_name));
}

fn estimateNvfp4ReconstructionKl(
    source: DenseWeightView,
    rows: usize,
    cols: usize,
    groups: usize,
    packed_scales: []const u8,
    packed_bytes: []const u8,
    global_scale: f32,
    sample_seed: u64,
) f64 {
    if (rows == 0 or cols == 0 or groups == 0) return std.math.nan(f64);
    if (!(global_scale > 0.0) or !std.math.isFinite(global_scale)) return std.math.nan(f64);
    const total_blocks = std.math.mul(usize, rows, groups) catch return std.math.nan(f64);
    if (total_blocks == 0) return std.math.nan(f64);
    const packed_cols = cols / 2;
    if (packed_cols == 0) return std.math.nan(f64);
    const scale_len = std.math.mul(usize, rows, groups) catch return std.math.nan(f64);
    const packed_len = std.math.mul(usize, rows, packed_cols) catch return std.math.nan(f64);
    if (packed_scales.len != scale_len or packed_bytes.len != packed_len) return std.math.nan(f64);

    const sample_blocks = @max(@as(usize, 1), @min(total_blocks, nvfp4_kl_sample_block_cap));
    const step = @max(@as(usize, 1), total_blocks / sample_blocks);
    const offset = @as(usize, @intCast(mix64(sample_seed) % @as(u64, @intCast(total_blocks))));

    var kl_sum: f64 = 0.0;
    var kl_count: usize = 0;
    var ref_vals: [nvfp4_group_size]f32 = undefined;
    var dq_vals: [nvfp4_group_size]f32 = undefined;
    for (0..sample_blocks) |sample_idx| {
        const raw_block_idx = (offset + sample_idx * step) % total_blocks;
        const row = raw_block_idx / groups;
        const group = raw_block_idx % groups;
        const block_base = row * cols + group * nvfp4_group_size;
        const scale_idx = row * groups + group;
        const packed_scale = packed_scales[scale_idx];
        const scale_f32 = dtype.fp8e4m3ToF32(packed_scale) * global_scale;
        if (!(scale_f32 > 0.0) or !std.math.isFinite(scale_f32)) continue;

        for (0..nvfp4_group_size) |i| {
            ref_vals[i] = source.valueAt(block_base + i);
        }
        for (0..(nvfp4_group_size / 2)) |pair_idx| {
            const packed_idx = row * packed_cols + group * (nvfp4_group_size / 2) + pair_idx;
            const byte = packed_bytes[packed_idx];
            const lo = byte & 0x0F;
            const hi = (byte >> 4) & 0x0F;
            dq_vals[pair_idx * 2] = fp4E2m1NibbleToF32(lo) * scale_f32;
            dq_vals[pair_idx * 2 + 1] = fp4E2m1NibbleToF32(hi) * scale_f32;
        }

        const kl = sampledBlockScaledHistogramKl(&ref_vals, &dq_vals, scale_f32);
        if (!std.math.isFinite(kl) or kl < 0.0) continue;
        kl_sum += kl;
        kl_count += 1;
    }

    if (kl_count == 0) return std.math.nan(f64);
    return kl_sum / @as(f64, @floatFromInt(kl_count));
}

fn scalarF32TensorValue(t: tensor.Tensor) !f32 {
    try ensureScalarF32Tensor(t);
    const data = t.data();
    if (data.len < @sizeOf(f32)) return error.InvalidConfig;
    return @as([*]align(1) const f32, @ptrCast(data.ptr))[0];
}

fn loadExistingNvfp4Tensor(
    allocator: std.mem.Allocator,
    existing_st: *safetensors.UnifiedSafeTensors,
    source_weight_name: []const u8,
    source: tensor.Tensor,
    source_view: DenseWeightView,
    sample_seed: u64,
) !?ExistingNvfp4Tensor {
    if (!existing_st.hasTensor(source_weight_name)) return null;
    const base = convertedNvfp4BaseName(source_weight_name);
    const scale_name = try std.fmt.allocPrint(allocator, "{s}.weight_scale", .{base});
    defer allocator.free(scale_name);
    const scale2_name = try std.fmt.allocPrint(allocator, "{s}.weight_scale_2", .{base});
    defer allocator.free(scale2_name);
    const input_scale_name = try std.fmt.allocPrint(allocator, "{s}.input_scale", .{base});
    defer allocator.free(input_scale_name);
    if (!existing_st.hasTensor(scale_name)) return null;
    if (!existing_st.hasTensor(scale2_name)) return null;
    if (!existing_st.hasTensor(input_scale_name)) return null;

    const weight_tensor = existing_st.getTensor(source_weight_name, null) catch return null;
    const scale_tensor = existing_st.getTensor(scale_name, null) catch return null;
    const scale2_tensor = existing_st.getTensor(scale2_name, null) catch return null;
    const input_scale_tensor = existing_st.getTensor(input_scale_name, null) catch return null;

    const packed_shape = packedShapeForWeight(source) catch return null;
    const rows = packed_shape.rows;
    const cols = packed_shape.cols;
    const groups = cols / nvfp4_group_size;
    const packed_cols = cols / 2;

    if (weight_tensor.n_dims != source.n_dims) return null;
    if (weight_tensor.n_dims == 2) {
        if (weight_tensor.shape[0] != source.shape[0]) return null;
        if (weight_tensor.shape[1] != @as(i64, @intCast(packed_cols))) return null;
    } else if (weight_tensor.n_dims == 3) {
        if (weight_tensor.shape[0] != source.shape[0]) return null;
        if (weight_tensor.shape[1] != source.shape[1]) return null;
        if (weight_tensor.shape[2] != @as(i64, @intCast(packed_cols))) return null;
    } else return null;
    if (weight_tensor.dtype != .u8 and weight_tensor.dtype != .i8) return null;
    if (scale_tensor.dtype != .f8_e4m3 or scale_tensor.n_dims != source.n_dims) return null;
    if (scale_tensor.n_dims == 2) {
        if (scale_tensor.shape[0] != source.shape[0]) return null;
        if (scale_tensor.shape[1] != @as(i64, @intCast(groups))) return null;
    } else if (scale_tensor.n_dims == 3) {
        if (scale_tensor.shape[0] != source.shape[0]) return null;
        if (scale_tensor.shape[1] != source.shape[1]) return null;
        if (scale_tensor.shape[2] != @as(i64, @intCast(groups))) return null;
    } else return null;

    const weight_scale_2 = scalarF32TensorValue(scale2_tensor) catch return null;
    const input_scale = scalarF32TensorValue(input_scale_tensor) catch return null;
    if (!(weight_scale_2 > 0.0) or !std.math.isFinite(weight_scale_2)) return null;

    const packed_weight = weight_tensor.data()[0..weight_tensor.data_size];
    const packed_scale = scale_tensor.data()[0..scale_tensor.data_size];
    const kl = estimateNvfp4ReconstructionKl(
        source_view,
        rows,
        cols,
        groups,
        packed_scale,
        packed_weight,
        weight_scale_2,
        sample_seed,
    );
    if (!std.math.isFinite(kl) or kl < 0.0) return null;

    return .{
        .packed_weight = packed_weight,
        .packed_scale = packed_scale,
        .weight_scale_2 = weight_scale_2,
        .input_scale = input_scale,
        .kl_divergence = kl,
    };
}

const Nvfp4BlockProgress = struct {
    ctx: grouped_affine.ProgressContext,
    line_id: u8,
    total: u64,
    base_done: u64,
    step: u64,
    done: u64 = 0,
    last_emitted: u64 = 0,
    message: ?[*:0]const u8 = null,

    fn init(
        ctx: grouped_affine.ProgressContext,
        line_id: u8,
        total: u64,
        base_done: u64,
        message: ?[*:0]const u8,
    ) Nvfp4BlockProgress {
        // Throttle updates to avoid progress I/O dominating conversion time.
        const step = if (total <= 64) 1 else @max(@as(u64, 1), total / 32);
        return .{
            .ctx = ctx,
            .line_id = line_id,
            .total = total,
            .base_done = base_done,
            .step = step,
            .message = message,
        };
    }

    fn bump(self: *Nvfp4BlockProgress, inc: u64) void {
        if (!self.ctx.isActive() or self.total == 0) return;
        const next = @min(self.total, self.done + inc);
        self.done = next;
        if (self.done == self.total or (self.done - self.last_emitted) >= self.step) {
            self.last_emitted = self.done;
            self.ctx.updateLine(self.line_id, self.base_done + self.done, self.message);
        }
    }

    fn finish(self: *Nvfp4BlockProgress) void {
        if (!self.ctx.isActive() or self.total == 0) return;
        self.done = self.total;
        self.last_emitted = self.total;
        self.ctx.updateLine(self.line_id, self.base_done + self.total, self.message);
    }

    fn setMessage(self: *Nvfp4BlockProgress, message: ?[*:0]const u8) void {
        self.message = message;
        if (!self.ctx.isActive() or self.total == 0) return;
        self.ctx.updateLine(self.line_id, self.base_done + self.done, self.message);
    }
};

fn setNvfp4BlockPhase(
    block_progress: ?*Nvfp4BlockProgress,
    message_buf: *[320]u8,
    source_weight_name: []const u8,
    phase: []const u8,
) void {
    const progress_state = block_progress orelse return;
    const source_len = @min(source_weight_name.len, 220);
    const msg = std.fmt.bufPrintZ(message_buf, "{s} | {s}", .{ source_weight_name[0..source_len], phase }) catch blk: {
        const fallback = std.fmt.bufPrintZ(message_buf, "{s}", .{phase}) catch return;
        break :blk fallback;
    };
    progress_state.setMessage(msg.ptr);
}

fn setNvfp4WritePhaseWithMetrics(
    block_progress: ?*Nvfp4BlockProgress,
    message_buf: *[320]u8,
    source_weight_name: []const u8,
    quality: Nvfp4QualityStats,
) void {
    var phase_buf: [192]u8 = undefined;
    const sign = if (quality.improvement_pct >= 0.0) "+" else "";
    const act = if (quality.used_activation_importance) "on" else "off";
    const phase = switch (quality.metric) {
        .proxy_mse => std.fmt.bufPrint(
            &phase_buf,
            "Write packed nibbles | proxy={e}->{e} ({s}{d:.2}%) clip={d:.2} g={e} kl={e} act={s}",
            .{
                quality.baseline,
                quality.selected,
                sign,
                quality.improvement_pct,
                quality.clip_multiplier,
                quality.global_scale,
                quality.kl_divergence,
                act,
            },
        ) catch "Write packed nibbles",
        .forward_mse => std.fmt.bufPrint(
            &phase_buf,
            "Write packed nibbles | forward={e}->{e} ({s}{d:.2}%) clip={d:.2} g={e} kl={e} act={s}",
            .{
                quality.baseline,
                quality.selected,
                sign,
                quality.improvement_pct,
                quality.clip_multiplier,
                quality.global_scale,
                quality.kl_divergence,
                act,
            },
        ) catch "Write packed nibbles",
        .scale_mse => std.fmt.bufPrint(
            &phase_buf,
            "Write packed nibbles | scale={e}->{e} ({s}{d:.2}%) clip={d:.2} g={e} kl={e} act={s}",
            .{
                quality.baseline,
                quality.selected,
                sign,
                quality.improvement_pct,
                quality.clip_multiplier,
                quality.global_scale,
                quality.kl_divergence,
                act,
            },
        ) catch "Write packed nibbles",
        .hybrid_mse => std.fmt.bufPrint(
            &phase_buf,
            "Write packed nibbles | hybrid={e}->{e} ({s}{d:.2}%) clip={d:.2} g={e} kl={e} act={s}",
            .{
                quality.baseline,
                quality.selected,
                sign,
                quality.improvement_pct,
                quality.clip_multiplier,
                quality.global_scale,
                quality.kl_divergence,
                act,
            },
        ) catch "Write packed nibbles",
    };
    setNvfp4BlockPhase(block_progress, message_buf, source_weight_name, phase);
}

fn formatNvfp4WeightProgressMessage(
    out_buf: *[576]u8,
    tensor_name: []const u8,
    profile: @TypeOf((grouped_affine.ConvertOptions{}).profile),
    quality: Nvfp4QualityStats,
    global_improvement_pct: ?f32,
    global_mse_reduction_pct: ?f32,
) [*:0]const u8 {
    const name_len = @min(tensor_name.len, 160);
    const sign = if (quality.improvement_pct >= 0.0) "+" else "";
    const global_value = global_improvement_pct orelse quality.improvement_pct;
    const global_sign = if (global_value >= 0.0) "+" else "";
    const mse_value = nvfp4ForwardMseReductionPct(quality);
    const mse_global_value = global_mse_reduction_pct;
    var mse_tensor_buf: [32]u8 = undefined;
    var mse_global_buf: [32]u8 = undefined;
    const mse_tensor_text = if (mse_value) |v|
        std.fmt.bufPrint(&mse_tensor_buf, "{s}{d:.2}%", .{ if (v >= 0.0) "+" else "", v }) catch "n/a"
    else
        "n/a";
    const mse_global_text = if (mse_global_value) |v|
        std.fmt.bufPrint(&mse_global_buf, "{s}{d:.2}%", .{ if (v >= 0.0) "+" else "", v }) catch "n/a"
    else
        "n/a";
    const act = if (quality.used_activation_importance) "on" else "off";
    const mode = nvfp4ProfileName(profile);
    const rendered = switch (quality.metric) {
        .proxy_mse => std.fmt.bufPrintZ(
            out_buf,
            "{s} | mode={s} proxy={e}->{e} ({s}{d:.2}%) clip={d:.2} g={e} kl={e} act={s} | pack_gain={s}{d:.2}% | mse={s} | mse_gain={s}",
            .{
                tensor_name[0..name_len],
                mode,
                quality.baseline,
                quality.selected,
                sign,
                quality.improvement_pct,
                quality.clip_multiplier,
                quality.global_scale,
                quality.kl_divergence,
                act,
                global_sign,
                global_value,
                mse_tensor_text,
                mse_global_text,
            },
        ),
        .forward_mse => std.fmt.bufPrintZ(
            out_buf,
            "{s} | mode={s} forward={e}->{e} ({s}{d:.2}%) clip={d:.2} g={e} kl={e} act={s} | pack_gain={s}{d:.2}% | mse={s} | mse_gain={s}",
            .{
                tensor_name[0..name_len],
                mode,
                quality.baseline,
                quality.selected,
                sign,
                quality.improvement_pct,
                quality.clip_multiplier,
                quality.global_scale,
                quality.kl_divergence,
                act,
                global_sign,
                global_value,
                mse_tensor_text,
                mse_global_text,
            },
        ),
        .scale_mse => std.fmt.bufPrintZ(
            out_buf,
            "{s} | mode={s} scale={e}->{e} ({s}{d:.2}%) clip={d:.2} g={e} kl={e} act={s} | pack_gain={s}{d:.2}% | mse={s} | mse_gain={s}",
            .{
                tensor_name[0..name_len],
                mode,
                quality.baseline,
                quality.selected,
                sign,
                quality.improvement_pct,
                quality.clip_multiplier,
                quality.global_scale,
                quality.kl_divergence,
                act,
                global_sign,
                global_value,
                mse_tensor_text,
                mse_global_text,
            },
        ),
        .hybrid_mse => std.fmt.bufPrintZ(
            out_buf,
            "{s} | mode={s} hybrid={e}->{e} ({s}{d:.2}%) clip={d:.2} g={e} kl={e} act={s} | pack_gain={s}{d:.2}% | mse={s} | mse_gain={s}",
            .{
                tensor_name[0..name_len],
                mode,
                quality.baseline,
                quality.selected,
                sign,
                quality.improvement_pct,
                quality.clip_multiplier,
                quality.global_scale,
                quality.kl_divergence,
                act,
                global_sign,
                global_value,
                mse_tensor_text,
                mse_global_text,
            },
        ),
    } catch std.fmt.bufPrintZ(out_buf, "{s}", .{tensor_name[0..name_len]}) catch unreachable;
    return rendered.ptr;
}

fn printNvfp4QualitySummary(summary: Nvfp4QualitySummary) void {
    if (summary.tensors == 0) return;
    const global_pct = summary.globalImprovementPct();
    const mean_pct = summary.meanImprovementPct();
    const mean_kl = summary.meanKlDivergence();
    const min_pct = if (std.math.isFinite(summary.min_improvement_pct)) summary.min_improvement_pct else 0.0;
    const max_pct = if (std.math.isFinite(summary.max_improvement_pct)) summary.max_improvement_pct else 0.0;
    const min_kl = if (std.math.isFinite(summary.min_kl_divergence)) summary.min_kl_divergence else 0.0;
    const max_kl = if (std.math.isFinite(summary.max_kl_divergence)) summary.max_kl_divergence else 0.0;
    const global_sign = if (global_pct >= 0.0) "+" else "";
    const mean_sign = if (mean_pct >= 0.0) "+" else "";
    const min_sign = if (min_pct >= 0.0) "+" else "";
    const max_sign = if (max_pct >= 0.0) "+" else "";
    std.debug.print(
        "NVFP4 quality summary: mode={s} tensors={d} global={s}{d:.2}% mean={s}{d:.2}% min={s}{d:.2}% max={s}{d:.2}% kl_mean={e} kl_min={e} kl_max={e} regressions={d} act={d}/{d} metrics(scale={d},proxy={d},forward={d},hybrid={d})\n",
        .{
            nvfp4ProfileName(summary.profile),
            summary.tensors,
            global_sign,
            global_pct,
            mean_sign,
            mean_pct,
            min_sign,
            min_pct,
            max_sign,
            max_pct,
            mean_kl,
            min_kl,
            max_kl,
            summary.regressions,
            summary.activation_tensors,
            summary.tensors,
            summary.metric_scale,
            summary.metric_proxy,
            summary.metric_forward,
            summary.metric_hybrid,
        },
    );
    std.debug.print(
        "NVFP4 quality note: percentages are internal reconstruction improvement (lower metric is better), and KL is sampled scaled-domain histogram KL(P||Q) with Laplace smoothing; neither is benchmark eval score.\n",
        .{},
    );
}

fn computeNvfp4ProgressTotalQualityWeight(
    st: *safetensors.UnifiedSafeTensors,
    specs: []const OutputTensorSpec,
) !f64 {
    var total: f64 = 0.0;
    for (specs) |spec| {
        if (spec.kind != .converted_input_scale) continue;
        const source_name = spec.source_weight_name orelse continue;
        const source = st.getTensor(source_name, null) catch continue;
        if (source.n_dims != 2 or source.shape[0] <= 0 or source.shape[1] <= 0) continue;
        const rows_u64: u64 = @intCast(source.shape[0]);
        const cols_u64: u64 = @intCast(source.shape[1]);
        if (cols_u64 == 0 or (cols_u64 % nvfp4_group_size) != 0) continue;
        const groups_u64: u64 = cols_u64 / nvfp4_group_size;
        const blocks_u64 = std.math.mul(u64, rows_u64, groups_u64) catch continue;
        const quality_weight_u64: u64 = @max(blocks_u64, 1);
        total += @as(f64, @floatFromInt(quality_weight_u64));
    }
    return total;
}

fn updateNvfp4ProgressGlobalGainPct(
    progress_gain_weighted_sum: *f64,
    progress_global_gain_pct: *f32,
    tensor_improvement_pct: f32,
    tensor_weight: f64,
    total_weight: f64,
) void {
    if (!(tensor_weight > 0.0) or !(total_weight > 0.0)) return;
    const clamped_tensor_gain_pct: f64 = @floatCast(@max(tensor_improvement_pct, 0.0));
    progress_gain_weighted_sum.* += clamped_tensor_gain_pct * tensor_weight;
    const next_gain_pct: f32 = @floatCast(progress_gain_weighted_sum.* / total_weight);
    progress_global_gain_pct.* = @max(progress_global_gain_pct.*, next_gain_pct);
}

fn writeDenseWeightAsBf16(
    allocator: std.mem.Allocator,
    file: std.fs.File,
    source: tensor.Tensor,
) !void {
    if (source.dtype == .bf16) {
        try file.writeAll(source.data()[0..source.data_size]);
        return;
    }
    if (source.n_dims != 2 or source.shape[0] <= 0 or source.shape[1] <= 0) return error.InvalidConfig;

    const rows: usize = @intCast(source.shape[0]);
    const cols: usize = @intCast(source.shape[1]);
    const total = std.math.mul(usize, rows, cols) catch return error.InvalidConfig;
    const view = try DenseWeightView.init(source);
    var bf16_values = try allocator.alloc(u16, total);
    defer allocator.free(bf16_values);
    for (0..total) |idx| {
        bf16_values[idx] = convert.f32ToBf16(view.valueAt(idx));
    }
    try file.writeAll(std.mem.sliceAsBytes(bf16_values));
}

fn writeDenseWeightAsBf16Rhs(
    allocator: std.mem.Allocator,
    file: std.fs.File,
    source: tensor.Tensor,
) !void {
    if (source.n_dims != 2 or source.shape[0] <= 0 or source.shape[1] <= 0) return error.InvalidConfig;

    const rows: usize = @intCast(source.shape[0]);
    const cols: usize = @intCast(source.shape[1]);
    const total = std.math.mul(usize, rows, cols) catch return error.InvalidConfig;
    const view = try DenseWeightView.init(source);
    var bf16_values = try allocator.alloc(u16, total);
    defer allocator.free(bf16_values);

    // Source lm_head is [vocab, hidden]; runtime matmul RHS expects [hidden, vocab].
    for (0..rows) |row| {
        const row_base = row * cols;
        for (0..cols) |col| {
            const dst_idx = col * rows + row;
            bf16_values[dst_idx] = convert.f32ToBf16(view.valueAt(row_base + col));
        }
    }
    try file.writeAll(std.mem.sliceAsBytes(bf16_values));
}

fn packDenseWeightToMxfp8(
    allocator: std.mem.Allocator,
    source_weight_name: []const u8,
    source: tensor.Tensor,
) !PreservedMxfp8Data {
    if (source.n_dims != 2 or source.shape[0] <= 0 or source.shape[1] <= 0) return error.InvalidConfig;
    const rows: usize = @intCast(source.shape[0]);
    const cols: usize = @intCast(source.shape[1]);
    const total = std.math.mul(usize, rows, cols) catch return error.InvalidConfig;
    const scale_cols = (cols + 31) / 32;
    const scales_total = std.math.mul(usize, rows, scale_cols) catch return error.InvalidConfig;
    const view = try DenseWeightView.init(source);

    const fp8_values = try allocator.alloc(u8, total);
    errdefer allocator.free(fp8_values);
    const block_scales = try allocator.alloc(u8, scales_total);
    errdefer allocator.free(block_scales);

    for (0..rows) |row| {
        const row_base = row * cols;
        const row_scale_base = row * scale_cols;
        for (0..scale_cols) |group| {
            const start_col = group * 32;
            const end_col = @min(start_col + 32, cols);

            var absmax: f32 = 0.0;
            for (start_col..end_col) |col| {
                const value = view.valueAt(row_base + col);
                absmax = @max(absmax, @abs(value));
            }

            const shared_exp: f32 = blk: {
                if (absmax <= 0.0) break :blk -8.0;
                const exp_floor = @floor(std.math.log2(absmax));
                break :blk std.math.clamp(exp_floor - 8.0, -127.0, 127.0);
            };
            const e8m0: u8 = @intCast(@as(u32, @intFromFloat(std.math.clamp(shared_exp + 127.0, 0.0, 255.0))));
            block_scales[row_scale_base + group] = e8m0;

            const scale_bits: u32 = @as(u32, e8m0) << 23;
            const scale: f32 = @bitCast(scale_bits);
            const inv_scale: f32 = if (scale > 0.0) 1.0 / scale else 0.0;
            for (start_col..end_col) |col| {
                const value = view.valueAt(row_base + col);
                fp8_values[row_base + col] = dtype.f32ToFp8E4M3(value * inv_scale);
            }
        }
    }

    return .{
        .source_weight_name = source_weight_name,
        .fp8_weight = fp8_values,
        .block_scales = block_scales,
    };
}

fn writeStreamedSafetensorsData(
    allocator: std.mem.Allocator,
    file: std.fs.File,
    st: *safetensors.UnifiedSafeTensors,
    existing_st: ?*safetensors.UnifiedSafeTensors,
    specs: []const OutputTensorSpec,
    profile: @TypeOf((grouped_affine.ConvertOptions{}).profile),
    lm_head_quantized: bool,
    update_round: u32,
    can_use_metal_pack: bool,
    activation_cache: ?*const calibration_capture.LayerActivationCache,
    activation_sample_count: usize,
    calib_seed: u64,
    convert_pool: *parallel.ThreadPool,
    progress: grouped_affine.ProgressContext,
    total_weights: usize,
    total_blocks: u64,
) !Nvfp4QualitySummary {
    var packed_cache: ?PackedNvfp4Data = null;
    defer if (packed_cache) |*cache| cache.deinit(allocator);
    var preserved_mxfp8_cache: ?PreservedMxfp8Data = null;
    defer if (preserved_mxfp8_cache) |*cache| cache.deinit(allocator);
    var packed_done: usize = 0;
    var packed_blocks_done: u64 = 0;
    var current_tensor_blocks_total: u64 = 0;
    var current_tensor_progress_work: u64 = 0;
    var current_source_name: ?[]const u8 = null;
    var quality_summary = Nvfp4QualitySummary{ .profile = profile };
    const progress_total_quality_weight = try computeNvfp4ProgressTotalQualityWeight(st, specs);
    var progress_gain_weighted_sum: f64 = 0.0;
    var progress_global_gain_pct: f32 = 0.0;
    const block_line_id: u8 = 2;

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
            .preserved_bf16_weight => {
                if (packed_cache != null) return error.InvalidConfig;
                if (preserved_mxfp8_cache != null) return error.InvalidConfig;
                const source_name = spec.source_weight_name orelse return error.InvalidConfig;
                const source = st.getTensor(source_name, null) catch |err| {
                    log.warn("convert", "NVFP4 preserved BF16 tensor lookup failed", .{
                        .tensor = source_name,
                        .err = @errorName(err),
                    });
                    return err;
                };
                try writeDenseWeightAsBf16(allocator, file, source);
            },
            .dense_bf16_rhs_weight => {
                if (packed_cache != null) return error.InvalidConfig;
                if (preserved_mxfp8_cache != null) return error.InvalidConfig;
                const source_name = spec.source_weight_name orelse return error.InvalidConfig;
                const source = st.getTensor(source_name, null) catch |err| {
                    log.warn("convert", "NVFP4 dense RHS tensor lookup failed", .{
                        .tensor = source_name,
                        .err = @errorName(err),
                    });
                    return err;
                };
                try writeDenseWeightAsBf16Rhs(allocator, file, source);
            },
            .preserved_mxfp8_weight, .preserved_mxfp8_block_scale => {
                if (packed_cache != null) return error.InvalidConfig;
                const source_name = spec.source_weight_name orelse return error.InvalidConfig;
                if (preserved_mxfp8_cache == null or
                    !std.mem.eql(u8, preserved_mxfp8_cache.?.source_weight_name, source_name))
                {
                    if (preserved_mxfp8_cache != null) return error.InvalidConfig;
                    const source = st.getTensor(source_name, null) catch |err| {
                        log.warn("convert", "NVFP4 preserved MXFP8 tensor lookup failed", .{
                            .tensor = source_name,
                            .err = @errorName(err),
                        });
                        return err;
                    };
                    preserved_mxfp8_cache = packDenseWeightToMxfp8(allocator, source_name, source) catch |err| {
                        log.warn("convert", "NVFP4 preserved MXFP8 conversion failed", .{
                            .tensor = source_name,
                            .err = @errorName(err),
                        });
                        return err;
                    };
                }
                const cache = &preserved_mxfp8_cache.?;
                switch (spec.kind) {
                    .preserved_mxfp8_weight => try file.writeAll(cache.fp8_weight),
                    .preserved_mxfp8_block_scale => {
                        try file.writeAll(cache.block_scales);
                        cache.deinit(allocator);
                        preserved_mxfp8_cache = null;
                    },
                    else => unreachable,
                }
            },
            .converted_weight, .converted_scale, .converted_scale_2, .converted_input_scale => {
                if (preserved_mxfp8_cache != null) return error.InvalidConfig;
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
                    const packed_shape = try packedShapeForWeight(source);
                    current_source_name = source_name;
                    var tensor_msg_buf: [256]u8 = undefined;
                    const msg_copy_len = @min(source_name.len, tensor_msg_buf.len - 1);
                    @memcpy(tensor_msg_buf[0..msg_copy_len], source_name[0..msg_copy_len]);
                    tensor_msg_buf[msg_copy_len] = 0;
                    const tensor_msg: ?[*:0]const u8 = @ptrCast(&tensor_msg_buf);

                    const rows: u64 = @intCast(packed_shape.rows);
                    const cols: u64 = @intCast(packed_shape.cols);
                    const groups = cols / nvfp4_group_size;
                    const tensor_block_count = rows * groups;
                    const should_retune = shouldRetuneTensorInUpdate(source_name, update_round);
                    const use_advanced_search = shouldUseAdvancedSearchForTensor(
                        profile,
                        update_round,
                    );
                    const total_block_work = if (should_retune) blk: {
                        const clip_count: u64 = if (profile == .custom)
                            1
                        else
                            @intCast(clipMultipliersForProfile(profile, update_round).len);
                        const pass_count: u64 = if (use_advanced_search)
                            (clip_count * 2) + 2
                        else
                            2;
                        break :blk tensor_block_count * pass_count;
                    } else 0;
                    current_tensor_blocks_total = tensor_block_count;
                    current_tensor_progress_work = total_block_work;
                    var block_progress = Nvfp4BlockProgress.init(progress, block_line_id, total_block_work, packed_blocks_done, tensor_msg);
                    progress.updateLine(1, packed_done, tensor_msg);
                    const source_view = try DenseWeightView.init(source);
                    const sample_seed = nvfp4KlSampleSeedForTensor(source_name);
                    const existing_tensor = if (existing_st) |prev|
                        try loadExistingNvfp4Tensor(
                            allocator,
                            prev,
                            source_name,
                            source,
                            source_view,
                            sample_seed,
                        )
                    else
                        null;

                    if (should_retune or existing_tensor == null) {
                        packed_cache = packDenseWeightToNvfp4(
                            allocator,
                            source_name,
                            source,
                            profile,
                            lm_head_quantized,
                            can_use_metal_pack,
                            activation_cache,
                            activation_sample_count,
                            calib_seed,
                            update_round,
                            convert_pool,
                            if (total_block_work > 0) &block_progress else null,
                            existing_tensor,
                            use_advanced_search,
                        ) catch |err| {
                            log.warn("convert", "NVFP4 dense-to-packed conversion failed", .{
                                .tensor = source_name,
                                .err = @errorName(err),
                            });
                            return err;
                        };
                    } else {
                        const prev_tensor = existing_tensor.?;
                        packed_cache = .{
                            .source_weight_name = source_name,
                            .packed_weight = @constCast(prev_tensor.packed_weight),
                            .packed_scale = @constCast(prev_tensor.packed_scale),
                            .weight_scale_2 = prev_tensor.weight_scale_2,
                            .input_scale = prev_tensor.input_scale,
                            .quality = .{
                                .metric = .scale_mse,
                                .baseline = prev_tensor.kl_divergence,
                                .selected = prev_tensor.kl_divergence,
                                .improvement_pct = 0.0,
                                .clip_multiplier = 1.0,
                                .global_scale = prev_tensor.weight_scale_2,
                                .used_activation_importance = false,
                                .kl_divergence = prev_tensor.kl_divergence,
                            },
                            .owns_buffers = false,
                        };
                    }
                    block_progress.finish();
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
                        const quality_weight_u64: u64 = @max(current_tensor_blocks_total, 1);
                        const quality_weight: f64 = @floatFromInt(quality_weight_u64);
                        quality_summary.add(cache.quality, quality_weight);
                        updateNvfp4ProgressGlobalGainPct(
                            &progress_gain_weighted_sum,
                            &progress_global_gain_pct,
                            cache.quality.improvement_pct,
                            quality_weight,
                            progress_total_quality_weight,
                        );

                        const progress_mse_reduction_pct: ?f32 = if (quality_summary.forward_weighted_total > 0.0)
                            quality_summary.modelStateRelativeMseReductionPct()
                        else
                            null;
                        var weight_msg_buf: [576]u8 = undefined;
                        const display_name = current_source_name orelse source_name;
                        const progress_msg = formatNvfp4WeightProgressMessage(
                            &weight_msg_buf,
                            display_name,
                            profile,
                            cache.quality,
                            progress_global_gain_pct,
                            progress_mse_reduction_pct,
                        );
                        cache.deinit(allocator);
                        packed_cache = null;
                        packed_blocks_done = std.math.add(u64, packed_blocks_done, current_tensor_progress_work) catch std.math.maxInt(u64);
                        current_tensor_blocks_total = 0;
                        current_tensor_progress_work = 0;
                        packed_done += 1;
                        progress.updateLine(1, packed_done, progress_msg);
                    },
                    else => unreachable,
                }
            },
        }
    }

    if (packed_done != total_weights) {
        log.warn("convert", "NVFP4 packed-weight accounting mismatch", .{
            .expected = total_weights,
            .actual = packed_done,
        });
    }
    if (packed_blocks_done != total_blocks) {
        log.warn("convert", "NVFP4 block-progress accounting mismatch", .{
            .expected = total_blocks,
            .actual = packed_blocks_done,
        });
    }
    if (packed_cache != null) return error.InvalidConfig;
    if (preserved_mxfp8_cache != null) return error.InvalidConfig;
    quality_summary.progress_gain_pct = progress_global_gain_pct;
    return quality_summary;
}

fn packDenseWeightToNvfp4(
    allocator: std.mem.Allocator,
    source_weight_name: []const u8,
    weight: tensor.Tensor,
    profile: @TypeOf((grouped_affine.ConvertOptions{}).profile),
    lm_head_quantized: bool,
    can_use_metal_pack: bool,
    activation_cache: ?*const calibration_capture.LayerActivationCache,
    activation_sample_count: usize,
    calib_seed: u64,
    update_round: u32,
    convert_pool: *parallel.ThreadPool,
    block_progress: ?*Nvfp4BlockProgress,
    existing_tensor: ?ExistingNvfp4Tensor,
    use_advanced_search: bool,
) !PackedNvfp4Data {
    if (!shouldConvertDenseWeightWithLmHead(source_weight_name, weight, profile, lm_head_quantized)) return error.InvalidConfig;
    const packed_shape = try packedShapeForWeight(weight);
    const rows = packed_shape.rows;
    const cols = packed_shape.cols;
    const groups = cols / nvfp4_group_size;
    const source = try DenseWeightView.init(weight);
    const block_count = std.math.mul(usize, rows, groups) catch return error.InvalidShape;

    const packed_cols = cols / 2;
    const packed_len = std.math.mul(usize, rows, packed_cols) catch return error.InvalidShape;
    const scale_len = std.math.mul(usize, rows, groups) catch return error.InvalidShape;

    const packed_bytes = try allocator.alloc(u8, packed_len);
    errdefer allocator.free(packed_bytes);
    const packed_scales = try allocator.alloc(u8, scale_len);
    errdefer allocator.free(packed_scales);
    var input_scale: f32 = 1.0;

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
    const used_activation_importance = group_importance != null;

    var sampled_scales: [global_scale_sample_limit]f32 = undefined;
    var sampled_importance: [global_scale_sample_limit]f32 = undefined;
    var phase_msg_buf: [320]u8 = undefined;
    var quality: Nvfp4QualityStats = .{};
    var global_scale: f32 = 1.0;
    var clip_multiplier: f32 = 1.0;
    var baseline_global_scale_for_forward: f32 = 1.0;
    var have_baseline_global_scale_for_forward = false;
    var block_scale_cache: ?[]f32 = null;
    const block_scale_policy = nvfp4BlockScalePolicyForProfile(profile);
    const use_metal_pack_for_weight = can_use_metal_pack and
        block_count >= resolveNvfp4MetalPackMinBlocks() and
        !std.mem.endsWith(u8, source_weight_name, "lm_head.weight");
    defer if (block_scale_cache) |cache| allocator.free(cache);
    if (use_advanced_search) {
        var forward_replay_inputs: ?Nvfp4ForwardReplayInputs = null;
        if (resolveNvfp4ReplayPolicy(profile) == .capture_required) {
            forward_replay_inputs = try buildNvfp4ForwardReplayInputs(
                allocator,
                source_weight_name,
                source,
                rows,
                cols,
                activation_cache,
                activation_sample_count,
                calib_seed,
            );
        }
        defer if (forward_replay_inputs) |*replay| replay.deinit(allocator);
        const prefer_replay_eval = shouldPreferMetalReplayEval() and (forward_replay_inputs != null);

        var best_global_scale: f32 = 1.0;
        var best_clip_multiplier: f32 = 1.0;
        var best_baseline_global_scale: f32 = 1.0;
        var best_eval_mse: f64 = std.math.inf(f64);
        var baseline_eval_mse: ?f64 = null;
        var best_metric: Nvfp4QualityMetric = .proxy_mse;
        var baseline_metric: Nvfp4QualityMetric = .proxy_mse;
        var fallback_global_scale: f32 = 1.0;
        var have_fallback = false;

        const clip_candidates = clipMultipliersForProfile(profile, update_round);
        for (clip_candidates) |candidate_clip_multiplier| {
            var sample_phase_buf: [64]u8 = undefined;
            const sample_phase = std.fmt.bufPrint(&sample_phase_buf, "Sample scales (clip={d:.2}, metric=pending)", .{candidate_clip_multiplier}) catch "Sample scales";
            setNvfp4BlockPhase(block_progress, &phase_msg_buf, source_weight_name, sample_phase);
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
                null,
                block_scale_policy,
                convert_pool,
                block_progress,
            );
            if (sampled_count == 0) continue;

            const candidate_global_scale = chooseNvfp4GlobalScale(
                sampled_scales[0..sampled_count],
                sampled_importance[0..sampled_count],
                observed_max_scale,
            );
            const candidate_baseline_global_scale = nvfp4InitialGlobalScale(observed_max_scale);
            if (!have_fallback) {
                fallback_global_scale = candidate_global_scale;
                have_fallback = true;
                best_baseline_global_scale = candidate_baseline_global_scale;
            }
            const scale_refine_multipliers = globalScaleRefineMultipliersForProfile(profile, update_round);
            var candidate_metric: Nvfp4QualityMetric = .proxy_mse;
            var candidate_eval_mse: f64 = std.math.inf(f64);
            var candidate_eval_scale = candidate_global_scale;
            var eval_scales = [_]f32{0.0} ** nvfp4_forward_eval_max_scales;
            var eval_count: usize = 0;
            for (scale_refine_multipliers) |scale_mult| {
                if (eval_count >= eval_scales.len) break;
                const eval_scale = candidate_global_scale * scale_mult;
                if (!(eval_scale > 0.0) or !std.math.isFinite(eval_scale)) continue;
                eval_scales[eval_count] = eval_scale;
                eval_count += 1;
            }
            if (eval_count == 0) continue;

            var eval_phase_buf: [96]u8 = undefined;
            const eval_phase = std.fmt.bufPrint(
                &eval_phase_buf,
                "Eval best loss (clip={d:.2}, scales={d})",
                .{ candidate_clip_multiplier, eval_count },
            ) catch "Eval best loss";
            setNvfp4BlockPhase(block_progress, &phase_msg_buf, source_weight_name, eval_phase);

            var proxy_eval = [_]f64{std.math.inf(f64)} ** nvfp4_forward_eval_max_scales;
            if (!prefer_replay_eval) {
                estimateNvfp4ForwardProxyMseMultiScales(
                    source,
                    rows,
                    cols,
                    groups,
                    eval_scales[0..eval_count],
                    candidate_clip_multiplier,
                    block_scale_policy,
                    if (group_importance) |weights| weights else null,
                    proxy_eval[0..eval_count],
                    convert_pool,
                    block_progress,
                );
            }

            for (0..eval_count) |eval_idx| {
                const eval_scale = eval_scales[eval_idx];
                var eval_metric: Nvfp4QualityMetric = .proxy_mse;
                var eval_mse = proxy_eval[eval_idx];
                if (forward_replay_inputs) |*replay| {
                    const replay_mse = estimateNvfp4ForwardReplayMse(
                        allocator,
                        source,
                        cols,
                        groups,
                        eval_scale,
                        candidate_clip_multiplier,
                        block_scale_policy,
                        replay,
                    ) catch |err| {
                        std.debug.print(
                            "NVFP4 convert failed: replay=xray forward replay metric failed for {s} ({s}).\n",
                            .{ source_weight_name, @errorName(err) },
                        );
                        return error.Nvfp4ReplayForwardMetricFailed;
                    };
                    if (std.math.isFinite(replay_mse)) {
                        if (replay.from_capture) {
                            eval_mse = replay_mse;
                            eval_metric = .forward_mse;
                        } else {
                            eval_mse = (eval_mse * 0.8) + (replay_mse * 0.2);
                            eval_metric = .hybrid_mse;
                        }
                    }
                }
                if (!std.math.isFinite(eval_mse) and prefer_replay_eval) {
                    std.debug.print(
                        "NVFP4 convert failed: replay=xray metric returned non-finite value for {s}.\n",
                        .{source_weight_name},
                    );
                    return error.Nvfp4ReplayForwardMetricFailed;
                }
                if (eval_mse < candidate_eval_mse) {
                    candidate_eval_mse = eval_mse;
                    candidate_eval_scale = eval_scale;
                    candidate_metric = eval_metric;
                }
            }
            if (std.math.approxEqAbs(f32, candidate_clip_multiplier, 1.0, 0.0001) and std.math.isFinite(candidate_eval_mse)) {
                baseline_eval_mse = candidate_eval_mse;
                baseline_metric = candidate_metric;
            }
            if (candidate_eval_mse < best_eval_mse) {
                best_eval_mse = candidate_eval_mse;
                best_global_scale = candidate_eval_scale;
                best_clip_multiplier = candidate_clip_multiplier;
                best_metric = candidate_metric;
                best_baseline_global_scale = candidate_baseline_global_scale;
            }
            if (std.math.isFinite(candidate_eval_mse)) {
                const baseline_for_display = baseline_eval_mse orelse candidate_eval_mse;
                const best_for_display = if (std.math.isFinite(best_eval_mse)) best_eval_mse else candidate_eval_mse;
                const improvement_for_display = nvfp4ImprovementPct(baseline_for_display, best_for_display);
                const sign = if (improvement_for_display >= 0.0) "+" else "";
                const metric_label = switch (candidate_metric) {
                    .forward_mse => "forward",
                    .proxy_mse => "proxy",
                    .hybrid_mse => "hybrid",
                    .scale_mse => "scale",
                };
                var metric_phase_buf: [160]u8 = undefined;
                const metric_phase = std.fmt.bufPrint(
                    &metric_phase_buf,
                    "Eval {s} loss (clip={d:.2}) | best={d:.6} ({s}{d:.2}%)",
                    .{ metric_label, candidate_clip_multiplier, best_for_display, sign, improvement_for_display },
                ) catch "Eval best loss";
                setNvfp4BlockPhase(block_progress, &phase_msg_buf, source_weight_name, metric_phase);
            }
        }
        global_scale = if (std.math.isFinite(best_eval_mse)) best_global_scale else fallback_global_scale;
        clip_multiplier = if (std.math.isFinite(best_eval_mse)) best_clip_multiplier else @as(f32, 1.0);
        baseline_global_scale_for_forward = best_baseline_global_scale;
        have_baseline_global_scale_for_forward = true;
        const selected_eval = if (std.math.isFinite(best_eval_mse)) best_eval_mse else (baseline_eval_mse orelse std.math.inf(f64));
        const baseline_eval = baseline_eval_mse orelse selected_eval;
        const selected_metric = if (std.math.isFinite(best_eval_mse))
            best_metric
        else
            baseline_metric;
        quality = .{
            .metric = selected_metric,
            .baseline = baseline_eval,
            .selected = selected_eval,
            .improvement_pct = nvfp4ImprovementPct(baseline_eval, selected_eval),
            .clip_multiplier = clip_multiplier,
            .global_scale = global_scale,
            .used_activation_importance = used_activation_importance,
        };
    } else {
        const simple_clip_multiplier = resolveNvfp4CustomClipMultiplier(profile);
        const simple_scale_refine_multiplier = resolveNvfp4CustomScaleRefineMultiplier(profile);
        const single_custom_candidate = [_]Nvfp4ClipScaleCandidate{
            .{
                .clip_multiplier = simple_clip_multiplier,
                .scale_refine_multiplier = simple_scale_refine_multiplier,
            },
        };
        const clip_scale_candidates = single_custom_candidate[0..];
        var best_candidate_metric = std.math.inf(f64);
        var best_candidate_baseline_metric = std.math.inf(f64);
        var best_candidate_scale: f32 = global_scale;
        var best_candidate_clip: f32 = 1.0;
        var best_candidate_baseline_scale: f32 = 1.0;

        const block_scale_cache_limit = resolveNvfp4BlockScaleCacheMaxBlocks();
        if (block_count > 0 and block_count <= block_scale_cache_limit) {
            block_scale_cache = try allocator.alloc(f32, block_count);
        }

        for (clip_scale_candidates) |candidate| {
            setNvfp4BlockPhase(block_progress, &phase_msg_buf, source_weight_name, "Sample scales (metric=pending)");
            var sampled_count: usize = 0;
            var sampled_seen: usize = 0;
            var observed_max_scale: f32 = 0.0;
            const use_metal_sample = shouldUseMetalBlockScaleCollection() and
                use_metal_pack_for_weight and
                block_scale_cache != null;
            if (use_metal_sample) {
                try collectSampledBlockScalesMetal(
                    allocator,
                    source,
                    rows,
                    cols,
                    groups,
                    candidate.clip_multiplier,
                    if (group_importance) |weights| weights else null,
                    &sampled_scales,
                    &sampled_importance,
                    &sampled_count,
                    &sampled_seen,
                    &observed_max_scale,
                    block_scale_cache.?,
                    block_progress,
                );
            } else {
                collectSampledBlockScales(
                    source,
                    rows,
                    cols,
                    groups,
                    candidate.clip_multiplier,
                    if (group_importance) |weights| weights else null,
                    &sampled_scales,
                    &sampled_importance,
                    &sampled_count,
                    &sampled_seen,
                    &observed_max_scale,
                    block_scale_cache,
                    block_scale_policy,
                    convert_pool,
                    block_progress,
                );
            }
            if (sampled_count == 0) continue;

            var candidate_scale = chooseNvfp4GlobalScale(
                sampled_scales[0..sampled_count],
                sampled_importance[0..sampled_count],
                observed_max_scale,
            );
            const refined_scale = candidate_scale * candidate.scale_refine_multiplier;
            if (refined_scale > 0.0 and std.math.isFinite(refined_scale)) {
                candidate_scale = refined_scale;
            }
            const candidate_metric = @as(f64, scaledBlockMse(
                sampled_scales[0..sampled_count],
                sampled_importance[0..sampled_count],
                candidate_scale,
            ));
            const candidate_baseline_metric = @as(f64, scaledBlockMse(
                sampled_scales[0..sampled_count],
                sampled_importance[0..sampled_count],
                nvfp4InitialGlobalScale(observed_max_scale),
            ));
            const candidate_baseline_scale = nvfp4InitialGlobalScale(observed_max_scale);
            if (!std.math.isFinite(candidate_metric)) continue;
            if (candidate_metric < best_candidate_metric) {
                best_candidate_metric = candidate_metric;
                best_candidate_baseline_metric = candidate_baseline_metric;
                best_candidate_scale = candidate_scale;
                best_candidate_clip = candidate.clip_multiplier;
                best_candidate_baseline_scale = candidate_baseline_scale;
            }
        }

        if (std.math.isFinite(best_candidate_metric)) {
            global_scale = best_candidate_scale;
            clip_multiplier = best_candidate_clip;
            baseline_global_scale_for_forward = best_candidate_baseline_scale;
            have_baseline_global_scale_for_forward = true;
            const baseline_scale = if (std.math.isFinite(best_candidate_baseline_metric))
                best_candidate_baseline_metric
            else
                best_candidate_metric;
            const selected_scale = best_candidate_metric;
            quality = .{
                .metric = .scale_mse,
                .baseline = baseline_scale,
                .selected = selected_scale,
                .improvement_pct = nvfp4ImprovementPct(baseline_scale, selected_scale),
                .clip_multiplier = clip_multiplier,
                .global_scale = global_scale,
                .used_activation_importance = used_activation_importance,
            };
            const sign = if (quality.improvement_pct >= 0.0) "+" else "";
            var metric_ready_buf: [128]u8 = undefined;
            const metric_ready = std.fmt.bufPrint(
                &metric_ready_buf,
                "Scale metric ready | scale={d:.6} ({s}{d:.2}%)",
                .{ quality.selected, sign, quality.improvement_pct },
            ) catch "Scale metric ready";
            setNvfp4BlockPhase(block_progress, &phase_msg_buf, source_weight_name, metric_ready);
        } else {
            clip_multiplier = simple_clip_multiplier;
        }
    }

    quality.clip_multiplier = clip_multiplier;
    quality.global_scale = global_scale;
    quality.used_activation_importance = used_activation_importance;
    if (!have_baseline_global_scale_for_forward) baseline_global_scale_for_forward = global_scale;
    populateNvfp4ForwardMseQuality(
        allocator,
        source_weight_name,
        source,
        rows,
        cols,
        groups,
        clip_multiplier,
        block_scale_policy,
        activation_cache,
        activation_sample_count,
        calib_seed,
        baseline_global_scale_for_forward,
        global_scale,
        &quality,
    );
    const enforce_regression_guard = packed_shape.ndims == 2;
    if (enforce_regression_guard and existing_tensor == null and isQualityRegression(profile, quality.baseline, quality.selected)) {
        // Keep conversion deterministic and complete for large models: if a
        // candidate scale regresses this tensor-level guard, roll back to the
        // baseline scale for this tensor instead of failing the full run.
        global_scale = baseline_global_scale_for_forward;
        clip_multiplier = 1.0;
        quality.selected = quality.baseline;
        quality.improvement_pct = 0.0;
        quality.clip_multiplier = clip_multiplier;
        quality.global_scale = global_scale;
    }
    setNvfp4BlockPhase(block_progress, &phase_msg_buf, source_weight_name, "Write packed nibbles");
    var used_metal_pack = false;
    if (rows > 0 and groups > 0 and use_metal_pack_for_weight and block_scale_cache != null) {
        if (block_count > 0) {
            setNvfp4BlockPhase(block_progress, &phase_msg_buf, source_weight_name, "Write packed nibbles (metal)");
            try packNvfp4RowsWithMetal(
                allocator,
                source,
                rows,
                cols,
                groups,
                global_scale,
                block_scale_cache.?,
                packed_scales,
                packed_bytes,
                convert_pool,
            );
            used_metal_pack = true;
        }
    }

    const PackRowsContext = struct {
        source: DenseWeightView,
        cols: usize,
        groups: usize,
        packed_cols: usize,
        clip_multiplier: f32,
        global_scale: f32,
        block_scale_policy: Nvfp4BlockScalePolicy,
        block_scale_cache: ?[]const f32,
        packed_scales: []u8,
        packed_bytes: []u8,
    };
    const PackRowsFn = struct {
        fn run(start_row: usize, end_row: usize, ctx: *PackRowsContext) void {
            var block_vals: [nvfp4_group_size]f32 = undefined;
            for (start_row..end_row) |r| {
                for (0..ctx.groups) |g| {
                    const group_start = g * nvfp4_group_size;
                    const block_base = r * ctx.cols + group_start;
                    const max_abs = loadNvfp4BlockValuesAndMaxAbs(ctx.source, block_base, &block_vals);
                    const block_index = r * ctx.groups + g;
                    const block_scale = if (ctx.block_scale_cache) |cache|
                        cache[block_index]
                    else
                        chooseNvfp4BlockScaleWithPolicy(
                            block_vals[0..],
                            max_abs,
                            ctx.clip_multiplier,
                            ctx.block_scale_policy,
                        );
                    ctx.packed_scales[r * ctx.groups + g] = dtype.f32ToFp8E4M3(block_scale / ctx.global_scale);
                    const scale_f32 = dtype.fp8e4m3ToF32(ctx.packed_scales[r * ctx.groups + g]) * ctx.global_scale;

                    for (0..(nvfp4_group_size / 2)) |pair_idx| {
                        const lo_val = if (scale_f32 > 0.0) block_vals[pair_idx * 2] / scale_f32 else 0.0;
                        const hi_val = if (scale_f32 > 0.0) block_vals[pair_idx * 2 + 1] / scale_f32 else 0.0;
                        const lo = nearestFp4E2m1Nibble(lo_val);
                        const hi = nearestFp4E2m1Nibble(hi_val);
                        const dst_idx = r * ctx.packed_cols + g * (nvfp4_group_size / 2) + pair_idx;
                        ctx.packed_bytes[dst_idx] = lo | (hi << 4);
                    }
                }
            }
        }
    };
    if (!used_metal_pack and rows > 0 and groups > 0) {
        var pack_ctx = PackRowsContext{
            .source = source,
            .cols = cols,
            .groups = groups,
            .packed_cols = packed_cols,
            .clip_multiplier = clip_multiplier,
            .global_scale = global_scale,
            .block_scale_policy = block_scale_policy,
            .block_scale_cache = if (block_scale_cache) |cache| cache else null,
            .packed_scales = packed_scales,
            .packed_bytes = packed_bytes,
        };
        const work_blocks = std.math.mul(usize, rows, groups) catch 0;
        if (work_blocks < nvfp4_parallel_min_blocks) {
            PackRowsFn.run(0, rows, &pack_ctx);
        } else {
            convert_pool.parallelForCompute(rows, PackRowsFn.run, &pack_ctx);
        }
    }

    const sample_seed = nvfp4KlSampleSeedForTensor(source_weight_name);
    quality.kl_divergence = estimateNvfp4ReconstructionKl(
        source,
        rows,
        cols,
        groups,
        packed_scales,
        packed_bytes,
        global_scale,
        sample_seed,
    );
    if (existing_tensor) |prev| {
        const candidate_kl = quality.kl_divergence;
        const prev_kl = prev.kl_divergence;
        const keep_previous = !std.math.isFinite(candidate_kl) or candidate_kl < 0.0 or candidate_kl + 1e-12 >= prev_kl;
        if (keep_previous) {
            std.mem.copyForwards(u8, packed_bytes, prev.packed_weight);
            std.mem.copyForwards(u8, packed_scales, prev.packed_scale);
            global_scale = prev.weight_scale_2;
            input_scale = prev.input_scale;
            quality = .{
                .metric = .scale_mse,
                .baseline = prev_kl,
                .selected = prev_kl,
                .improvement_pct = 0.0,
                .clip_multiplier = 1.0,
                .global_scale = prev.weight_scale_2,
                .used_activation_importance = false,
                .kl_divergence = prev_kl,
            };
        } else {
            quality.baseline = prev_kl;
            quality.selected = candidate_kl;
            quality.improvement_pct = nvfp4ImprovementPct(prev_kl, candidate_kl);
        }
    }
    setNvfp4WritePhaseWithMetrics(block_progress, &phase_msg_buf, source_weight_name, quality);

    if (block_progress) |progress_state| {
        const done_blocks = std.math.mul(u64, @as(u64, @intCast(rows)), @as(u64, @intCast(groups))) catch std.math.maxInt(u64);
        progress_state.bump(done_blocks);
    }

    return .{
        .source_weight_name = source_weight_name,
        .packed_weight = packed_bytes,
        .packed_scale = packed_scales,
        .weight_scale_2 = global_scale,
        .input_scale = input_scale,
        .quality = quality,
    };
}

fn shouldUseMetalBlockScaleCollection() bool {
    if (!shouldEnableNvfp4MetalPack()) return false;
    if (std.posix.getenv("TALU_NVFP4_METAL_BLOCK_SCALES")) |raw| {
        return !std.mem.eql(u8, raw, "0");
    }
    return true;
}

test "sampledBlockScaledHistogramKl is zero for identical blocks and positive for perturbed blocks" {
    const ref = [_]f32{ 0.0, 1.0, 2.0, -1.0, 3.0, -2.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 4.0, -4.0, 6.0, -6.0 };
    var same = ref;
    var perturbed = ref;
    perturbed[0] += 0.75;
    perturbed[7] -= 0.5;
    perturbed[12] += 1.25;

    const kl_same = sampledBlockScaledHistogramKl(&ref, &same, 1.0);
    const kl_perturbed = sampledBlockScaledHistogramKl(&ref, &perturbed, 1.0);
    try std.testing.expect(std.math.approxEqAbs(f64, kl_same, 0.0, 1e-12));
    try std.testing.expect(std.math.isFinite(kl_perturbed));
    try std.testing.expect(kl_perturbed > 0.0);
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

test "shouldConvertDenseWeight excludes MoE routing and shared gate weights" {
    var t = std.mem.zeroes(tensor.Tensor);
    t.dtype = .bf16;
    t.n_dims = 2;
    t.shape[0] = 64;
    t.shape[1] = 32;
    t.numel = 64 * 32;

    try std.testing.expect(!shouldConvertDenseWeight("model.layers.0.mlp.gate.weight", t, .good));
    try std.testing.expect(!shouldConvertDenseWeight("model.layers.0.mlp.shared_expert_gate.weight", t, .good));
}

test "isUnsupportedNvfp4MoeExpertTensor detects fused 3D MoE experts" {
    var moe = std.mem.zeroes(tensor.Tensor);
    moe.dtype = .bf16;
    moe.n_dims = 3;
    moe.shape[0] = 256;
    moe.shape[1] = 1024;
    moe.shape[2] = 2048;

    try std.testing.expect(!isUnsupportedNvfp4MoeExpertTensor(
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        moe,
    ));
    try std.testing.expect(!isUnsupportedNvfp4MoeExpertTensor(
        "model.language_model.layers.0.mlp.experts.down_proj",
        moe,
    ));
    try std.testing.expect(shouldConvertDenseWeight(
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        moe,
        .good,
    ));

    var invalid_moe = moe;
    invalid_moe.n_dims = 2;
    try std.testing.expect(isUnsupportedNvfp4MoeExpertTensor(
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        invalid_moe,
    ));
    invalid_moe = moe;
    invalid_moe.shape[2] = 15;
    try std.testing.expect(isUnsupportedNvfp4MoeExpertTensor(
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        invalid_moe,
    ));
    try std.testing.expect(!isUnsupportedNvfp4MoeExpertTensor(
        "model.language_model.layers.0.self_attn.q_proj.weight",
        moe,
    ));
}

test "packedShapeForWeight flattens 3D expert weights into row-major matrix geometry" {
    var t = std.mem.zeroes(tensor.Tensor);
    t.dtype = .bf16;
    t.n_dims = 3;
    t.shape[0] = 8;
    t.shape[1] = 1024;
    t.shape[2] = 2048;
    t.numel = 8 * 1024 * 2048;

    const shape = try packedShapeForWeight(t);
    try std.testing.expectEqual(@as(usize, 3), shape.ndims);
    try std.testing.expectEqual(@as(usize, 8 * 1024), shape.rows);
    try std.testing.expectEqual(@as(usize, 2048), shape.cols);
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

test "clipMultipliersForProfile keeps best aligned with good" {
    const clips = clipMultipliersForProfile(.best, 0);
    try std.testing.expectEqual(@as(usize, 3), clips.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0.97), clips[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), clips[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.03), clips[2], 1e-6);
}

test "clipMultipliersForProfile returns bounded search window for good profile" {
    const clips = clipMultipliersForProfile(.good, 0);
    try std.testing.expectEqual(@as(usize, 3), clips.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0.97), clips[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), clips[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.03), clips[2], 1e-6);
}

test "useAdvancedNvfp4Search only activates custom capture-required path" {
    try std.testing.expect(!useAdvancedNvfp4Search(.best, 0));
    try std.testing.expect(!useAdvancedNvfp4Search(.good, 0));
    try std.testing.expect(!useAdvancedNvfp4Search(.good, 1));
    try std.testing.expect(!useAdvancedNvfp4Search(.custom, 0));
}

test "resolveNvfp4CaptureBackendSelection is fixed to cpu for replay capture" {
    try std.testing.expectEqual(
        @as(@TypeOf((calibration_capture.CaptureOptions{}).backend_selection), .cpu),
        resolveNvfp4CaptureBackendSelection(),
    );
}

test "updateNvfp4ProgressGlobalGainPct is monotonic and starts at zero" {
    var gain_sum: f64 = 0.0;
    var global_pct: f32 = 0.0;
    const total_weight: f64 = 100.0;

    updateNvfp4ProgressGlobalGainPct(&gain_sum, &global_pct, 8.0, 10.0, total_weight);
    const first = global_pct;
    try std.testing.expect(first > 0.0);

    // Lower per-tensor gain later should not make global progress go down.
    updateNvfp4ProgressGlobalGainPct(&gain_sum, &global_pct, 1.0, 40.0, total_weight);
    try std.testing.expect(global_pct >= first);

    // Negative per-tensor gain is clamped in progress reporting.
    const before_negative = global_pct;
    updateNvfp4ProgressGlobalGainPct(&gain_sum, &global_pct, -5.0, 50.0, total_weight);
    try std.testing.expect(global_pct >= before_negative);
}

test "shouldRetuneTensorInUpdate is deterministic and always active for round zero" {
    const name = "model.language_model.layers.0.self_attn.q_proj.weight";
    try std.testing.expect(shouldRetuneTensorInUpdate(name, 0));
    const r1 = shouldRetuneTensorInUpdate(name, 1);
    const r1_again = shouldRetuneTensorInUpdate(name, 1);
    try std.testing.expectEqual(r1, r1_again);
}

test "chooseNvfp4BlockScale responds to clip multiplier" {
    const block = [_]f32{6.0} ++ ([_]f32{0.0} ** (nvfp4_group_size - 1));
    const no_clip = chooseNvfp4BlockScale(&block, 6.0, 1.0);
    const clipped = chooseNvfp4BlockScale(&block, 6.0, 0.8);
    try std.testing.expect(clipped <= no_clip);
    try std.testing.expect(clipped > 0.0);
}

test "best profile uses legacy block scale policy" {
    try std.testing.expectEqual(Nvfp4BlockScalePolicy.legacy_multi, nvfp4BlockScalePolicyForProfile(.best));
    try std.testing.expectEqual(Nvfp4BlockScalePolicy.legacy_multi, nvfp4BlockScalePolicyForProfile(.good));
}

test "chooseNvfp4BlockScaleWithPolicy returns finite positive scale" {
    const block = [_]f32{6.0} ++ ([_]f32{0.0} ** (nvfp4_group_size - 1));
    const scale = chooseNvfp4BlockScaleWithPolicy(&block, 6.0, 1.0, .legacy_multi);
    try std.testing.expect(scale > 0.0);
    try std.testing.expect(std.math.isFinite(scale));
}

test "nearestFp4E2m1Nibble matches linear scan across range" {
    const probes = [_]f32{
        std.math.nan(f32),
        std.math.inf(f32),
        -std.math.inf(f32),
        -6.0,
        -5.0,
        -4.0,
        -3.5,
        -3.0,
        -2.5,
        -2.0,
        -1.75,
        -1.5,
        -1.25,
        -1.0,
        -0.75,
        -0.5,
        -0.25,
        -0.0,
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        5.0,
        6.0,
    };
    for (probes) |value| {
        try std.testing.expectEqual(
            nearestFp4E2m1NibbleLinearScan(value),
            nearestFp4E2m1Nibble(value),
        );
    }

    // Dense sweep to guard midpoint tie behavior.
    var i: i32 = -20000;
    while (i <= 20000) : (i += 1) {
        const value: f32 = @as(f32, @floatFromInt(i)) / 2000.0;
        try std.testing.expectEqual(
            nearestFp4E2m1NibbleLinearScan(value),
            nearestFp4E2m1Nibble(value),
        );
    }
}

test "estimateNvfp4ForwardProxyMse is finite for simple tensor" {
    const values = [_]f32{
        1.0,  -0.5,  0.25, 0.0,
        -1.0, 0.75,  0.5,  -0.25,
        0.33, -0.66, 1.25, -1.5,
        0.1,  0.2,   -0.3, 0.4,
    };
    const view: DenseWeightView = .{ .f32 = values[0..] };
    const pool = try parallel.ThreadPool.create(std.testing.allocator, 1);
    defer pool.deinit();
    const mse = estimateNvfp4ForwardProxyMse(view, 1, 16, 1, 1.0, 1.0, .legacy_multi, null, pool, null);
    try std.testing.expect(std.math.isFinite(mse));
    try std.testing.expect(mse >= 0.0);
}

test "estimateNvfp4TensorSensitivityQuick returns finite nmse for simple tensor" {
    const values = [_]f32{
        0.0,  1.0,   -1.0, 2.0,
        0.5,  -0.5,  1.5,  -1.5,
        3.0,  -3.0,  4.0,  -4.0,
        0.25, -0.25, 0.75, -0.75,
    };
    const view: DenseWeightView = .{ .f32 = values[0..] };
    const nmse = estimateNvfp4TensorSensitivityQuick(view, 1, 16, 1, 42, .legacy_multi);
    try std.testing.expect(std.math.isFinite(nmse));
    try std.testing.expect(nmse >= 0.0);
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

test "makeSmallModelPreservePolicy follows enable flag and size threshold" {
    const good_small = makeSmallModelPreservePolicy(true, 4_000_000_000, 27);
    try std.testing.expect(good_small.enabled);
    try std.testing.expectEqual(@as(?u32, 27), good_small.last_layer_index);

    const best_large = makeSmallModelPreservePolicy(true, 12_000_000_000, 63);
    try std.testing.expect(!best_large.enabled);

    const custom_small = makeSmallModelPreservePolicy(false, 4_000_000_000, 27);
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

test "mixedPreserveLayerCount rounds up and clamps to valid range" {
    try std.testing.expectEqual(@as(u32, 0), mixedPreserveLayerCount(0, 10));
    try std.testing.expectEqual(@as(u32, 1), mixedPreserveLayerCount(4, 10));
    try std.testing.expectEqual(@as(u32, 3), mixedPreserveLayerCount(28, 10));
    try std.testing.expectEqual(@as(u32, 28), mixedPreserveLayerCount(28, 100));
}

test "defaultNvfp4MixedPreserveLayersPct keeps good and best aligned" {
    try std.testing.expectEqual(@as(u32, 10), defaultNvfp4MixedPreserveLayersPct(.good, false));
    try std.testing.expectEqual(@as(u32, 0), defaultNvfp4MixedPreserveLayersPct(.custom, false));
    try std.testing.expectEqual(@as(u32, 10), defaultNvfp4MixedPreserveLayersPct(.best, false));
    try std.testing.expectEqual(@as(u32, 10), defaultNvfp4MixedPreserveLayersPct(.custom, true));
}

test "shouldPreserveWeightByLayerList matches configured mixed layers" {
    const preserved = [_]u32{ 3, 7, 12 };
    try std.testing.expect(shouldPreserveWeightByLayerList(
        "model.language_model.layers.7.self_attn.q_proj.weight",
        preserved[0..],
    ));
    try std.testing.expect(!shouldPreserveWeightByLayerList(
        "model.language_model.layers.8.self_attn.q_proj.weight",
        preserved[0..],
    ));
    try std.testing.expect(!shouldPreserveWeightByLayerList(
        "model.visual.blocks.7.attn.qkv.weight",
        preserved[0..],
    ));
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

test "isQualityRegression enforces no-regression contract for good semantics" {
    try std.testing.expect(!isQualityRegression(.custom, 1.0, 1.1));
    try std.testing.expect(!isQualityRegression(.good, 1.0, 1.0));
    try std.testing.expect(!isQualityRegression(.best, 1.0, 1.0));
    try std.testing.expect(isQualityRegression(.good, 1.0, 1.1));
    try std.testing.expect(isQualityRegression(.best, 1.0, 1.1));
}

test "default mixed preserve policy uses unified defaults" {
    try std.testing.expectEqual(@as(u32, mixed_preserve_layers_pct_default_good), defaultNvfp4MixedPreserveLayersPct(.good, false));
    try std.testing.expectEqual(@as(u32, mixed_preserve_layers_pct_default_good), defaultNvfp4MixedPreserveLayersPct(.best, false));
    try std.testing.expectEqual(@as(u32, 0), defaultNvfp4MixedPreserveLayersPct(.custom, false));
    try std.testing.expectEqual(@as(u32, mixed_preserve_layers_pct_default_good), defaultNvfp4MixedPreserveLayersPct(.custom, true));
    try std.testing.expectEqual(@as(usize, mixed_preserve_score_sample_blocks_default), mixedPreserveScoreSampleBlocks(.good));
    try std.testing.expectEqual(@as(usize, mixed_preserve_score_sample_blocks_default), mixedPreserveScoreSampleBlocks(.best));
}

test "small-model default scope enables requested 4B defaults" {
    try std.testing.expect(isNvfp4SmallModelDefaultScope(4_000_000_000, 4_000_000_000));
    try std.testing.expect(!isNvfp4SmallModelDefaultScope(9_000_000_000, 9_000_000_000));
    try std.testing.expect(resolveNvfp4SmallModelPreserveEnabled(.custom, true));
    try std.testing.expect(!resolveNvfp4LmHeadQuantized(.custom, true));
}

test "findUntiedLmHeadWeightName resolves canonical lm_head key" {
    const names = [_][]const u8{
        "model.language_model.embed_tokens.weight",
        "model.language_model.lm_head.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
    };
    try std.testing.expectEqualStrings(
        "model.language_model.lm_head.weight",
        findUntiedLmHeadWeightName(names[0..]).?,
    );
    const names_without_lm_head = [_][]const u8{
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
    };
    try std.testing.expect(findUntiedLmHeadWeightName(names_without_lm_head[0..]) == null);
}

test "quality probe auto-skip threshold uses source dense size" {
    try std.testing.expect(!shouldAutoSkipNvfp4QualityProbeBySourceSize(
        nvfp4_quality_probe_source_dense_bytes_limit,
    ));
    try std.testing.expect(shouldAutoSkipNvfp4QualityProbeBySourceSize(
        nvfp4_quality_probe_source_dense_bytes_limit + 1,
    ));
}

test "negLogProbFromLogits matches uniform distribution expectation" {
    const logits = [_]f32{ 0.0, 0.0 };
    const nll = try negLogProbFromLogits(&logits, 1);
    try std.testing.expectApproxEqAbs(@as(f64, std.math.log(f64, std.math.e, 2.0)), nll, 1e-12);
}

test "klDivergenceFromLogitsWithScratch is zero for identical logits" {
    const reference = [_]f32{ 1.5, -0.25, 0.75, 0.0 };
    var scratch = [_]f64{0.0} ** reference.len;
    const kld = try klDivergenceFromLogitsWithScratch(&reference, &reference, &scratch);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), kld, 1e-12);
}

test "klDivergenceFromLogitsWithScratch matches manual softmax KL and is shift-invariant" {
    // ref probs = softmax([0, 0]) = [0.5, 0.5]
    // model probs = softmax([ln(3), 0]) = [0.75, 0.25]
    // KL(ref||model) = 0.5*ln(0.5/0.75) + 0.5*ln(0.5/0.25) = ln(2/sqrt(3))
    const ln3 = std.math.log(f64, std.math.e, 3.0);
    const expected = std.math.log(f64, std.math.e, 2.0 / std.math.sqrt(3.0));

    const reference = [_]f32{ 0.0, 0.0 };
    const model = [_]f32{ @floatCast(ln3), 0.0 };
    var scratch = [_]f64{0.0} ** reference.len;
    const kld = try klDivergenceFromLogitsWithScratch(&reference, &model, &scratch);
    try std.testing.expectApproxEqAbs(expected, kld, 1e-12);

    // Adding the same constant to both logits in each distribution must not change KL.
    const reference_shifted = [_]f32{ 7.0, 7.0 };
    const model_shifted = [_]f32{ @floatCast(ln3 + 11.0), 11.0 };
    const shifted = try klDivergenceFromLogitsWithScratch(&reference_shifted, &model_shifted, &scratch);
    try std.testing.expectApproxEqAbs(kld, shifted, 1e-12);
}

test "packDenseWeightToNvfp4 good profile is deterministic and non-regressing" {
    const rows: usize = 4;
    const cols: usize = 16;
    var values: [rows * cols]f32 = undefined;
    for (&values, 0..) |*slot, idx| {
        const i: f32 = @floatFromInt(idx);
        slot.* = @sin(i * 0.17) * 1.75 + @cos(i * 0.09) * 0.5;
    }

    var t = std.mem.zeroes(tensor.Tensor);
    t.dtype = .f32;
    t.n_dims = 2;
    t.shape[0] = rows;
    t.shape[1] = cols;
    t.numel = rows * cols;
    t.data_ptr = @ptrCast(std.mem.asBytes(&values).ptr);
    t.data_size = std.mem.asBytes(&values).len;

    var pool = try parallel.ThreadPool.create(std.testing.allocator, 1);
    defer pool.deinit();

    var packed_a = try packDenseWeightToNvfp4(
        std.testing.allocator,
        "model.language_model.layers.0.self_attn.q_proj.weight",
        t,
        .good,
        true,
        false,
        null,
        0,
        42,
        0,
        &pool,
        null,
        null,
        false,
    );
    defer packed_a.deinit(std.testing.allocator);

    var packed_b = try packDenseWeightToNvfp4(
        std.testing.allocator,
        "model.language_model.layers.0.self_attn.q_proj.weight",
        t,
        .good,
        true,
        false,
        null,
        0,
        42,
        0,
        &pool,
        null,
        null,
        false,
    );
    defer packed_b.deinit(std.testing.allocator);

    try std.testing.expectEqualSlices(u8, packed_a.packed_weight, packed_b.packed_weight);
    try std.testing.expectEqualSlices(u8, packed_a.packed_scale, packed_b.packed_scale);
    try std.testing.expectEqual(packed_a.weight_scale_2, packed_b.weight_scale_2);
    try std.testing.expectEqual(packed_a.input_scale, packed_b.input_scale);
    try std.testing.expect(std.math.isFinite(packed_a.quality.baseline));
    try std.testing.expect(std.math.isFinite(packed_a.quality.selected));
    try std.testing.expect(std.math.isFinite(packed_a.quality.kl_divergence));
    try std.testing.expect(packed_a.quality.kl_divergence >= 0.0);
    try std.testing.expect(packed_a.quality.selected <= packed_a.quality.baseline);
    try std.testing.expect(!isQualityRegression(.good, packed_a.quality.baseline, packed_a.quality.selected));
}

fn chooseNvfp4BlockScale(block_vals: []const f32, max_abs: f32, clip_multiplier: f32) f32 {
    return chooseNvfp4BlockScaleWithPolicy(block_vals, max_abs, clip_multiplier, .legacy_multi);
}

fn nvfp4BlockScaleMse(block_vals: []const f32, scale: f32) f32 {
    if (scale <= 0.0 or !std.math.isFinite(scale) or block_vals.len == 0) return std.math.inf(f32);
    var mse: f32 = 0.0;
    for (block_vals) |value| {
        const scaled = value / scale;
        const nibble = nearestFp4E2m1Nibble(scaled);
        const decoded = fp4E2m1NibbleToF32(nibble) * scale;
        const err = value - decoded;
        mse += err * err;
    }
    mse /= @as(f32, @floatFromInt(block_vals.len));
    return mse;
}

fn chooseNvfp4BlockScaleWithPolicy(
    block_vals: []const f32,
    max_abs: f32,
    clip_multiplier: f32,
    policy: Nvfp4BlockScalePolicy,
) f32 {
    _ = policy;
    if (max_abs <= 0.0) return 0.0;
    const clipped_abs = if (clip_multiplier > 0.0 and std.math.isFinite(clip_multiplier))
        max_abs * clip_multiplier
    else
        max_abs;
    const effective_abs = if (clipped_abs > 0.0 and std.math.isFinite(clipped_abs)) clipped_abs else max_abs;

    const legacy_candidate_scales = [_]f32{
        effective_abs / 6.0,
        effective_abs / 5.0,
        effective_abs / 4.5,
        effective_abs / 4.0,
        effective_abs / 3.5,
    };

    var legacy_best_scale = legacy_candidate_scales[0];
    var legacy_best_mse = std.math.inf(f32);
    for (legacy_candidate_scales) |candidate_scale| {
        if (candidate_scale <= 0.0 or !std.math.isFinite(candidate_scale)) continue;
        const mse = nvfp4BlockScaleMse(block_vals, candidate_scale);
        if (mse < legacy_best_mse) {
            legacy_best_mse = mse;
            legacy_best_scale = candidate_scale;
        }
    }
    return legacy_best_scale;
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

        const has_weight_suffix = st.hasTensor(base);
        const weight_name = if (has_weight_suffix)
            try allocator.dupe(u8, base)
        else
            try std.fmt.allocPrint(allocator, "{s}.weight", .{base});
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
        const packed_shape = try packedShapeForWeight(weight_tensor);
        const rows = packed_shape.rows;
        const cols = packed_shape.cols;

        if (weight_tensor.dtype != .u8 and weight_tensor.dtype != .i8) return error.InvalidConfig;
        if (!(weight_tensor.n_dims == 2 or weight_tensor.n_dims == 3)) return error.InvalidConfig;
        if (rows == 0 or cols == 0) return error.InvalidConfig;

        if (scale_tensor.dtype != .f8_e4m3) return error.InvalidConfig;
        if (scale_tensor.n_dims != weight_tensor.n_dims) return error.InvalidConfig;
        if (weight_tensor.n_dims == 2) {
            if (scale_tensor.shape[0] != weight_tensor.shape[0]) return error.InvalidConfig;
            if (scale_tensor.shape[1] <= 0) return error.InvalidConfig;
        } else {
            if (scale_tensor.shape[0] != weight_tensor.shape[0]) return error.InvalidConfig;
            if (scale_tensor.shape[1] != weight_tensor.shape[1]) return error.InvalidConfig;
            if (scale_tensor.shape[2] <= 0) return error.InvalidConfig;
        }

        try ensureScalarF32Tensor(scale2_tensor);
        try ensureScalarF32Tensor(input_scale_tensor);

        const packed_cols: usize = @intCast(if (weight_tensor.n_dims == 2) weight_tensor.shape[1] else weight_tensor.shape[2]);
        const unpacked_cols = std.math.mul(usize, packed_cols, 2) catch return error.InvalidConfig;
        if ((unpacked_cols % nvfp4_group_size) != 0) return error.InvalidConfig;
        const expected_scale_cols = unpacked_cols / nvfp4_group_size;
        const actual_scale_cols: i64 = if (scale_tensor.n_dims == 2) scale_tensor.shape[1] else scale_tensor.shape[2];
        if (actual_scale_cols != @as(i64, @intCast(expected_scale_cols))) return error.InvalidConfig;
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

test "writeNvfp4CanaryReport writes expected json payload" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const output_dir = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(output_dir);

    const comparison: Nvfp4CanaryComparison = .{
        .source = .{
            .scored_tokens = 64,
            .mean_nll = 1.5,
            .ppl = std.math.exp(1.5),
            .mean_kld = 0.0,
            .mean_logit_mse = 0.0,
            .mean_logit_nmse = 0.0,
        },
        .candidate = .{
            .scored_tokens = 64,
            .mean_nll = 1.55,
            .ppl = std.math.exp(1.55),
            .mean_kld = 2.5e-4,
            .mean_logit_mse = 1.2e-3,
            .mean_logit_nmse = 8.0e-4,
        },
        .delta_nll = 0.05,
        .nll_regression_pct = 1.0,
        .nll_regression_ci95_low_pct = 0.7,
        .nll_regression_ci95_high_pct = 1.3,
        .nll_regression_p95_pct = 1.8,
        .ppl_inflation_pct = 1.1,
    };
    try writeNvfp4CanaryReport(allocator, output_dir, comparison);

    const report_path = try nvfp4CanaryReportPath(allocator, output_dir);
    defer allocator.free(report_path);
    const payload = try std.fs.cwd().readFileAlloc(allocator, report_path, 16 * 1024);
    defer allocator.free(payload);

    try std.testing.expect(std.mem.indexOf(u8, payload, "\"tokens\":64") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"source\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"candidate\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"delta_nll\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"ppl_ratio\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"nll_regression_p95_pct\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"mean_logit_mse\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"mean_logit_nmse\"") != null);
}
