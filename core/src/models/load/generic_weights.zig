//! Generic architecture-driven weight loader.
//!
//! Loads tensors using weight specifications from static model metadata and
//! applies transforms/layout handling without model-specific hardcoding.

const std = @import("std");
const tensor = @import("../../tensor.zig");
const st_loader = @import("../../io/safetensors/root.zig");
const model_types = @import("../op_types.zig");
const transforms = @import("transforms.zig");
const log = @import("../../log.zig");

const Tensor = tensor.Tensor;
const Allocator = std.mem.Allocator;
const WeightSpec = model_types.WeightSpec;
const WeightTransform = model_types.WeightTransform;
const WeightMap = std.StringHashMapUnmanaged(*const Tensor);

pub const LoadOptions = struct {
    preserve_native_norm_dtype: bool = false,
    dequantize_mxfp8_to_bf16: bool = false,
    dequantize_nvfp4_to_bf16: bool = true,
};

const NormalizedNameEntry = struct {
    actual_name: []const u8,
    ambiguous: bool = false,
};

/// Resolves tensor candidates with bounded normalization for repeated wrapper
/// segments (for example, duplicated `language_model` namespaces).
pub const NameResolver = struct {
    initialized: bool = false,
    normalized_names: std.StringHashMapUnmanaged(NormalizedNameEntry) = .{},

    pub fn deinit(self: *NameResolver, allocator: Allocator) void {
        var iter = self.normalized_names.iterator();
        while (iter.next()) |kv| allocator.free(kv.key_ptr.*);
        self.normalized_names.deinit(allocator);
        self.* = .{};
    }

    pub fn resolve(
        self: *NameResolver,
        allocator: Allocator,
        safetensors: *st_loader.UnifiedSafeTensors,
        candidate_name: []const u8,
        candidate_name_buf: []u8,
    ) !?[]const u8 {
        if (safetensors.hasTensor(candidate_name)) return candidate_name;

        if (buildQWeightCandidate(candidate_name, candidate_name_buf)) |qweight_name| {
            if (safetensors.hasTensor(qweight_name)) return qweight_name;
        }

        if (buildWeightPackedCandidate(candidate_name, candidate_name_buf)) |packed_name| {
            if (safetensors.hasTensor(packed_name)) return packed_name;
        }

        if (try self.resolveNormalizedCandidate(allocator, safetensors, candidate_name)) |resolved_name| return resolved_name;

        if (buildQWeightCandidate(candidate_name, candidate_name_buf)) |qweight_name| {
            if (try self.resolveNormalizedCandidate(allocator, safetensors, qweight_name)) |resolved_name| return resolved_name;
        }

        if (buildWeightPackedCandidate(candidate_name, candidate_name_buf)) |packed_name| {
            if (try self.resolveNormalizedCandidate(allocator, safetensors, packed_name)) |resolved_name| return resolved_name;
        }

        return null;
    }

    pub fn resolveDType(
        self: *NameResolver,
        allocator: Allocator,
        safetensors: *st_loader.UnifiedSafeTensors,
        candidate_name: []const u8,
        qweight_name_buf: []u8,
    ) !?tensor.DType {
        const resolved_name = try self.resolve(allocator, safetensors, candidate_name, qweight_name_buf) orelse return null;
        if (safetensors.getTensor(resolved_name, null)) |t| return t.dtype else |_| return null;
    }

    fn resolveNormalizedCandidate(
        self: *NameResolver,
        allocator: Allocator,
        safetensors: *st_loader.UnifiedSafeTensors,
        candidate_name: []const u8,
    ) !?[]const u8 {
        try self.ensureInitialized(allocator, safetensors);

        const normalized_buf = try allocator.alloc(u8, candidate_name.len);
        defer allocator.free(normalized_buf);
        const normalized_candidate = try normalizeWrapperSegments(normalized_buf, candidate_name);

        const entry = self.normalized_names.getPtr(normalized_candidate) orelse return null;
        if (entry.ambiguous) return error.AmbiguousWeightName;
        return entry.actual_name;
    }

    fn ensureInitialized(
        self: *NameResolver,
        allocator: Allocator,
        safetensors: *st_loader.UnifiedSafeTensors,
    ) !void {
        if (self.initialized) return;

        const tensor_names = try safetensors.tensorNames(allocator);
        defer allocator.free(tensor_names);

        for (tensor_names) |tensor_name| {
            const normalized_buf = try allocator.alloc(u8, tensor_name.len);
            defer allocator.free(normalized_buf);
            const normalized_name = try normalizeWrapperSegments(normalized_buf, tensor_name);

            if (self.normalized_names.getPtr(normalized_name)) |entry| {
                if (!std.mem.eql(u8, entry.actual_name, tensor_name)) {
                    entry.ambiguous = true;
                }
                continue;
            }

            const normalized_storage = try allocator.dupe(u8, normalized_name);
            try self.normalized_names.put(allocator, normalized_storage, .{
                .actual_name = tensor_name,
                .ambiguous = false,
            });
        }

        self.initialized = true;
    }
};

pub fn loadWeightMap(
    allocator: Allocator,
    safetensors: *st_loader.UnifiedSafeTensors,
    specs: []const WeightSpec,
    weight_prefixes: []const []const u8,
    layer_idx: usize,
    model_config: *const tensor.ModelConfig,
    options: LoadOptions,
) !WeightMap {
    var map = WeightMap{};
    errdefer map.deinit(allocator);
    var name_resolver: NameResolver = .{};
    defer name_resolver.deinit(allocator);

    for (specs) |spec| {
        if (try loadWeightBySpec(
            allocator,
            safetensors,
            spec,
            weight_prefixes,
            layer_idx,
            model_config,
            options,
            &name_resolver,
        )) |weight| {
            try map.put(allocator, spec.id, weight);
        }
    }

    return map;
}

fn loadWeightBySpec(
    allocator: Allocator,
    safetensors: *st_loader.UnifiedSafeTensors,
    spec: WeightSpec,
    weight_prefixes: []const []const u8,
    layer_idx: usize,
    model_config: *const tensor.ModelConfig,
    options: LoadOptions,
    name_resolver: *NameResolver,
) !?*const Tensor {
    var name_buf: [512]u8 = undefined;
    var prefix_buf: [256]u8 = undefined;
    const alias_count = spec.aliases.len;
    var candidate_index: usize = 0;

    while (candidate_index < alias_count + 1) : (candidate_index += 1) {
        const candidate = if (candidate_index == 0) spec.suffix else spec.aliases[candidate_index - 1];

        if (weight_prefixes.len == 0 or std.mem.indexOf(u8, candidate, "{d}") != null) {
            const expanded_name = expandLayerTemplate(name_buf[0..], candidate, layer_idx) catch continue;
            if (try tryLoadCandidate(allocator, safetensors, spec, expanded_name, model_config, options, name_resolver)) |weight| {
                return weight;
            }
            continue;
        }

        for (weight_prefixes) |prefix_template| {
            const expanded_prefix = expandLayerTemplate(prefix_buf[0..], prefix_template, layer_idx) catch continue;
            const total_len = expanded_prefix.len + candidate.len;
            if (total_len > name_buf.len) continue;
            @memcpy(name_buf[0..expanded_prefix.len], expanded_prefix);
            @memcpy(name_buf[expanded_prefix.len..total_len], candidate);
            const expanded_name = name_buf[0..total_len];
            if (try tryLoadCandidate(allocator, safetensors, spec, expanded_name, model_config, options, name_resolver)) |weight| {
                return weight;
            }
        }
    }

    if (spec.required) {
        log.err("load", "Missing required weight", .{
            .id = spec.id,
            .layer = layer_idx,
        }, @src());
        return error.MissingWeight;
    }

    return null;
}

fn tryLoadCandidate(
    allocator: Allocator,
    safetensors: *st_loader.UnifiedSafeTensors,
    spec: WeightSpec,
    name: []const u8,
    model_config: *const tensor.ModelConfig,
    options: LoadOptions,
    name_resolver: *NameResolver,
) !?*const Tensor {
    var qweight_name_buf: [1024]u8 = undefined;
    const effective_name = try name_resolver.resolve(allocator, safetensors, name, qweight_name_buf[0..]) orelse return null;
    const raw_tensor = safetensors.getTensor(effective_name, null) catch return null;

    const transformed = applySpecTransforms(
        allocator,
        safetensors,
        effective_name,
        raw_tensor,
        spec,
        model_config,
        options,
    ) catch |err| {
        log.err("load", "Failed to apply weight transforms", .{
            .id = spec.id,
            .name = effective_name,
            .err = @errorName(err),
        }, @src());
        return err;
    };

    const weight_ptr = try allocator.create(Tensor); // lint:ignore errdefer-alloc - arena freed atomically
    weight_ptr.* = transformed;
    return weight_ptr;
}

fn isRepeatableWrapperSegment(segment: []const u8) bool {
    return std.mem.eql(u8, segment, "model") or
        std.mem.eql(u8, segment, "language_model") or
        std.mem.eql(u8, segment, "text_model") or
        std.mem.eql(u8, segment, "base_model") or
        std.mem.eql(u8, segment, "module");
}

fn normalizeWrapperSegments(buf: []u8, path: []const u8) ![]const u8 {
    var out_idx: usize = 0;
    var token_start: usize = 0;
    var prev_was_wrapper = false;

    while (token_start <= path.len) {
        var token_end = token_start;
        while (token_end < path.len and path[token_end] != '.') : (token_end += 1) {}
        const token = path[token_start..token_end];
        const is_wrapper = isRepeatableWrapperSegment(token);
        const skip_wrapper = prev_was_wrapper and is_wrapper;

        if (!skip_wrapper) {
            if (out_idx > 0) {
                if (out_idx >= buf.len) return error.BufferTooSmall;
                buf[out_idx] = '.';
                out_idx += 1;
            }
            if (out_idx + token.len > buf.len) return error.BufferTooSmall;
            @memcpy(buf[out_idx .. out_idx + token.len], token);
            out_idx += token.len;
        }
        prev_was_wrapper = is_wrapper;

        if (token_end == path.len) break;
        token_start = token_end + 1;
    }

    return buf[0..out_idx];
}

fn buildQWeightCandidate(name: []const u8, qweight_name_buf: []u8) ?[]const u8 {
    return buildWeightSuffixCandidate(name, ".qweight", qweight_name_buf);
}

fn buildWeightPackedCandidate(name: []const u8, packed_name_buf: []u8) ?[]const u8 {
    return buildWeightSuffixCandidate(name, ".weight_packed", packed_name_buf);
}

fn buildWeightSuffixCandidate(name: []const u8, suffix: []const u8, out_name_buf: []u8) ?[]const u8 {
    if (!std.mem.endsWith(u8, name, ".weight")) return null;
    const base_len = name.len - ".weight".len;
    const total = base_len + suffix.len;
    if (total > out_name_buf.len) return null;
    @memcpy(out_name_buf[0..base_len], name[0..base_len]);
    @memcpy(out_name_buf[base_len..total], suffix);
    return out_name_buf[0..total];
}

fn applySpecTransforms(
    allocator: Allocator,
    safetensors: *st_loader.UnifiedSafeTensors,
    name: []const u8,
    raw_tensor: Tensor,
    spec: WeightSpec,
    model_config: *const tensor.ModelConfig,
    options: LoadOptions,
) !Tensor {
    var tensor_view = raw_tensor;
    const expected_in = getExpectedIn(spec, raw_tensor, model_config);

    tensor_view = switch (spec.layout) {
        .linear, .gaffine => try transforms.orientWeight(
            allocator,
            safetensors,
            name,
            expected_in,
            model_config.*,
            options.dequantize_mxfp8_to_bf16,
            options.dequantize_nvfp4_to_bf16,
        ),
        .embedding => try transforms.orientEmbedding(allocator, safetensors, name, model_config.*),
        .conv1d_depthwise => try transforms.ensureF32(allocator, raw_tensor),
        .none => raw_tensor,
    };

    const force_f32 = spec.force_f32 or hasTransform(spec.transforms, .dtype_f32) or spec.layout == .conv1d_depthwise;
    // Skip F32 conversion for 3D+ tensors (fused expert weight blocks) — they stay
    // in native dtype and use the dtype-appropriate matmul kernel directly.
    if (force_f32 or (spec.layout == .none and !isNormModule(spec.module_type) and raw_tensor.n_dims <= 2)) {
        const preserve_norm_dtype = options.preserve_native_norm_dtype and isNormModule(spec.module_type) and !force_f32;
        if (!preserve_norm_dtype) {
            tensor_view = try transforms.ensureF32(allocator, tensor_view);
            if (spec.layout == .linear and tensor_view.dtype == .f32) {
                // Grouped-affine weights are converted to F32 after orientWeight().
                // Re-run F32 orientation so matmulF32 sees [in, out] layout.
                tensor_view = try transforms.orientWeightF32(allocator, tensor_view, expected_in);
            }
        }
    }

    return tensor_view;
}

/// Determine the expected input dimension for weight orientation.
///
/// For Linear weights, we need to know which dimension is "input" to correctly
/// orient them for our matmul. For gaffine quantization disambiguation, we need
/// the UNPACKED input dimension (original float size before quantization packing).
///
/// For quantized weights (gaffine), columns are packed:
/// - GAF4: 8 values per uint32 (in_features / 8 = packed_cols)
/// - GAF8: 4 values per uint32 (in_features / 4 = packed_cols)
///
/// Strategy:
/// 1. If expected_shape is provided, use shape[1] (always unpacked dimensions)
/// 2. Otherwise, infer from tensor shape and d_model, accounting for packing
fn getExpectedIn(spec: WeightSpec, raw_tensor: Tensor, model_config: *const tensor.ModelConfig) usize {
    const d_model: usize = @intCast(model_config.d_model);

    // If expected_shape is provided, always use it - it contains unpacked dimensions
    // This handles both same-size models (exact match) and quantized weights
    if (spec.expected_shape) |shape| {
        if (shape.len >= 2) {
            return shape[1];
        }
    }

    // No expected_shape - infer from tensor dimensions
    if (raw_tensor.n_dims == 2) {
        const rows: usize = @intCast(raw_tensor.shape[0]);
        const cols: usize = @intCast(raw_tensor.shape[1]);

        // Check if this looks like a quantized tensor (dtype indicates packing)
        const is_quantized = raw_tensor.dtype == .grouped_affine_u4 or
            raw_tensor.dtype == .grouped_affine_u8;

        if (is_quantized) {
            // Use config-specified bits for correct unpacking factor.
            // GAF4: 8 values per uint32 (cols * 8), GAF8: 4 values per uint32 (cols * 4).
            // Without this, GAF8 models (e.g., MiniLM) get misidentified as GAF4
            // because inferGaffineParams uses expected_in for disambiguation.
            const values_per_word: usize = if (model_config.gaffine_bits == 8) 4 else 8;
            const unpacked = cols * values_per_word;

            // Determine projection direction by output dimension (rows):
            // - "From d_model": rows != d_model (e.g., q_proj [2048, 128] -> 2048 != 1024)
            // - "To d_model": rows == d_model (e.g., o_proj [1024, 256] -> 1024 == 1024)
            if (rows == d_model) {
                // "To d_model" projection: input is unpacked cols
                return unpacked;
            } else {
                // "From d_model" projection: input is d_model
                return d_model;
            }
        } else {
            // Standard float weights: infer from which dimension matches d_model
            if (cols == d_model) return d_model;
            if (rows == d_model) return cols;
        }
    }

    return d_model;
}

fn validateExpectedShape(tensor_view: Tensor, expected: []const usize) !void {
    if (tensor_view.n_dims != expected.len) return error.InvalidShape;
    for (expected, 0..) |dim, idx| {
        if (@as(usize, @intCast(tensor_view.shape[idx])) != dim) return error.InvalidShape;
    }
}

fn hasTransform(transforms_list: []const WeightTransform, transform: WeightTransform) bool {
    for (transforms_list) |t| {
        if (t == transform) return true;
    }
    return false;
}

fn isNormModule(module_type: []const u8) bool {
    return std.mem.eql(u8, module_type, "RMSNorm") or std.mem.eql(u8, module_type, "LayerNorm");
}

pub fn expandLayerTemplate(buf: []u8, template: []const u8, layer_idx: usize) ![]const u8 {
    var idx_buf: [32]u8 = undefined;
    const idx_str = try std.fmt.bufPrint(&idx_buf, "{d}", .{layer_idx});

    var out_idx: usize = 0;
    var i: usize = 0;
    while (i < template.len) {
        if (i + 2 < template.len and template[i] == '{' and template[i + 1] == 'd' and template[i + 2] == '}') {
            if (out_idx + idx_str.len > buf.len) return error.BufferTooSmall;
            @memcpy(buf[out_idx .. out_idx + idx_str.len], idx_str);
            out_idx += idx_str.len;
            i += 3;
            continue;
        }
        if (out_idx + 1 > buf.len) return error.BufferTooSmall;
        buf[out_idx] = template[i];
        out_idx += 1;
        i += 1;
    }
    return buf[0..out_idx];
}

test "expandLayerTemplate replaces {d} with layer index" {
    var buf: [256]u8 = undefined;

    const result1 = try expandLayerTemplate(&buf, "model.layers.{d}.attn.q_proj.weight", 5);
    try std.testing.expectEqualStrings("model.layers.5.attn.q_proj.weight", result1);

    const result2 = try expandLayerTemplate(&buf, "model.layers.{d}.mlp.{d}.weight", 12);
    try std.testing.expectEqualStrings("model.layers.12.mlp.12.weight", result2);

    const result3 = try expandLayerTemplate(&buf, "no_placeholder", 0);
    try std.testing.expectEqualStrings("no_placeholder", result3);
}

test "expandLayerTemplate handles edge cases" {
    var buf: [256]u8 = undefined;

    // Layer 0
    const result1 = try expandLayerTemplate(&buf, "layer.{d}.weight", 0);
    try std.testing.expectEqualStrings("layer.0.weight", result1);

    // Large layer number
    const result2 = try expandLayerTemplate(&buf, "layer.{d}.weight", 999);
    try std.testing.expectEqualStrings("layer.999.weight", result2);

    // Template at start
    const result3 = try expandLayerTemplate(&buf, "{d}_layer", 42);
    try std.testing.expectEqualStrings("42_layer", result3);

    // Template at end
    const result4 = try expandLayerTemplate(&buf, "layer_{d}", 7);
    try std.testing.expectEqualStrings("layer_7", result4);
}

test "expandLayerTemplate returns error for buffer too small" {
    var small_buf: [5]u8 = undefined;

    const result = expandLayerTemplate(&small_buf, "layer.{d}.weight", 0);
    try std.testing.expectError(error.BufferTooSmall, result);
}

test "normalizeWrapperSegments collapses wrapper runs" {
    var buf: [256]u8 = undefined;
    const input = "model.language_model.language_model.language_model.layers.0.mlp.gate_proj.weight";
    const out = try normalizeWrapperSegments(&buf, input);
    try std.testing.expectEqualStrings("model.layers.0.mlp.gate_proj.weight", out);
}

test "normalizeWrapperSegments resolves mixed wrapper run before visual path" {
    var buf: [256]u8 = undefined;
    const input = "model.language_model.visual.blocks.0.attn.qkv.weight";
    const out = try normalizeWrapperSegments(&buf, input);
    try std.testing.expectEqualStrings("model.visual.blocks.0.attn.qkv.weight", out);
}

test "normalizeWrapperSegments keeps non-wrapper duplicates" {
    var buf: [256]u8 = undefined;
    const input = "layers.layers.0.weight";
    const out = try normalizeWrapperSegments(&buf, input);
    try std.testing.expectEqualStrings("layers.layers.0.weight", out);
}

test "NameResolver resolves .weight to .weight_packed fallback" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);
    const model_path = try std.fs.path.join(allocator, &.{ tmp_path, "weight_packed.safetensors" });
    defer allocator.free(model_path);

    const packed_data = [_]u8{ 0x12, 0x34 };
    const entries = [_]st_loader.TensorEntry{
        .{
            .name = "model.layers.0.mlp.down_proj.weight_packed",
            .dtype = .u8,
            .shape = &.{ 1, 2 },
            .data = packed_data[0..],
        },
    };
    try st_loader.write(allocator, model_path, &entries);

    var safetensors = try st_loader.UnifiedSafeTensors.load(allocator, model_path);
    defer safetensors.deinit();

    var resolver: NameResolver = .{};
    defer resolver.deinit(allocator);
    var candidate_buf: [256]u8 = undefined;

    const resolved = try resolver.resolve(
        allocator,
        &safetensors,
        "model.layers.0.mlp.down_proj.weight",
        candidate_buf[0..],
    );
    try std.testing.expect(resolved != null);
    try std.testing.expectEqualStrings(
        "model.layers.0.mlp.down_proj.weight_packed",
        resolved.?,
    );
}

test "loadWeightMap tested via integration tests" {
    // loadWeightMap requires a valid SafeTensors file handle and ModelConfig
    // which makes it unsuitable for unit testing in isolation.
    // Coverage is achieved through integration tests with actual model files.
    // The function delegates to loadWeightBySpec which is also integration-tested.
}

test "WeightSpec.force_f32 overrides preserve_native_norm_dtype for norm weights" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    var raw_owned = try tensor.OwnedTensor.init(arena_alloc, .bf16, &.{ 2, 2 });
    defer raw_owned.deinit();
    const raw_tensor = raw_owned.view();

    var dummy_safetensors: st_loader.UnifiedSafeTensors = undefined;

    const spec_force = WeightSpec{
        .id = "mixer.norm.weight",
        .suffix = "mixer.norm.weight",
        .module_type = "RMSNorm",
        .layout = .none,
        .dtype = "float32",
        .required = true,
        .force_f32 = true,
    };
    const spec_preserve = WeightSpec{
        .id = "mixer.norm.weight",
        .suffix = "mixer.norm.weight",
        .module_type = "RMSNorm",
        .layout = .none,
        .dtype = "float32",
        .required = true,
        .force_f32 = false,
    };

    const model_config = tensor.ModelConfig{
        .d_model = 2,
        .vocab_size = 16,
        .n_layers = 1,
        .n_heads = 1,
        .n_kv_groups = 1,
        .d_ff = 4,
        .head_dim = 2,
        .max_seq_len = 16,
        .rope_theta = 10_000.0,
        .norm_eps = 1e-5,
        .gaffine_group_size = 32,
        .model_arch = .custom,
    };

    const force_tensor = try applySpecTransforms(
        arena_alloc,
        &dummy_safetensors,
        "model.layers.0.mixer.norm.weight",
        raw_tensor,
        spec_force,
        &model_config,
        .{ .preserve_native_norm_dtype = true },
    );

    const preserve_tensor = try applySpecTransforms(
        arena_alloc,
        &dummy_safetensors,
        "model.layers.0.mixer.norm.weight",
        raw_tensor,
        spec_preserve,
        &model_config,
        .{ .preserve_native_norm_dtype = true },
    );

    try std.testing.expectEqual(tensor.DType.f32, force_tensor.dtype);
    try std.testing.expectEqual(tensor.DType.bf16, preserve_tensor.dtype);
}

test "loadWeightMap resolves repeated wrapper prefix via normalized fallback" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);
    const model_path = try std.fs.path.join(allocator, &.{ tmp_path, "model.safetensors" });
    defer allocator.free(model_path);

    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const entries = [_]st_loader.TensorEntry{
        .{
            .name = "model.language_model.language_model.language_model.layers.0.mlp.gate_proj.weight",
            .dtype = .f32,
            .shape = &.{ 2, 2 },
            .data = std.mem.sliceAsBytes(&data),
        },
    };
    try st_loader.write(allocator, model_path, &entries);

    var safetensors = try st_loader.UnifiedSafeTensors.load(allocator, model_path);
    defer safetensors.deinit();

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    const spec = WeightSpec{
        .id = "mlp.gate_proj.weight",
        .suffix = "mlp.gate_proj.weight",
        .module_type = "Linear",
        .layout = .linear,
        .dtype = "float32",
        .required = true,
    };
    const prefixes = [_][]const u8{"model.language_model.layers.{d}."};
    const config = tensor.ModelConfig{
        .d_model = 2,
        .vocab_size = 16,
        .n_layers = 1,
        .n_heads = 1,
        .n_kv_groups = 1,
        .d_ff = 4,
        .head_dim = 2,
        .max_seq_len = 16,
        .rope_theta = 10000.0,
        .norm_eps = 1e-5,
        .gaffine_group_size = 32,
        .model_arch = .custom,
    };

    var map = try loadWeightMap(
        arena_alloc,
        &safetensors,
        &.{spec},
        &prefixes,
        0,
        &config,
        .{},
    );
    defer map.deinit(arena_alloc);

    try std.testing.expect(map.get("mlp.gate_proj.weight") != null);
}

test "loadWeightMap fails on ambiguous normalized wrapper mapping" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);
    const model_path = try std.fs.path.join(allocator, &.{ tmp_path, "ambiguous.safetensors" });
    defer allocator.free(model_path);

    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const entries = [_]st_loader.TensorEntry{
        .{
            .name = "model.language_model.layers.0.mlp.gate_proj.weight",
            .dtype = .f32,
            .shape = &.{ 2, 2 },
            .data = std.mem.sliceAsBytes(&data),
        },
        .{
            .name = "model.language_model.language_model.layers.0.mlp.gate_proj.weight",
            .dtype = .f32,
            .shape = &.{ 2, 2 },
            .data = std.mem.sliceAsBytes(&data),
        },
    };
    try st_loader.write(allocator, model_path, &entries);

    var safetensors = try st_loader.UnifiedSafeTensors.load(allocator, model_path);
    defer safetensors.deinit();

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    const spec = WeightSpec{
        .id = "mlp.gate_proj.weight",
        .suffix = "mlp.gate_proj.weight",
        .module_type = "Linear",
        .layout = .linear,
        .dtype = "float32",
        .required = true,
    };
    const prefixes = [_][]const u8{"model.language_model.layers.{d}."};
    const config = tensor.ModelConfig{
        .d_model = 2,
        .vocab_size = 16,
        .n_layers = 1,
        .n_heads = 1,
        .n_kv_groups = 1,
        .d_ff = 4,
        .head_dim = 2,
        .max_seq_len = 16,
        .rope_theta = 10000.0,
        .norm_eps = 1e-5,
        .gaffine_group_size = 32,
        .model_arch = .custom,
    };

    try std.testing.expectError(
        error.AmbiguousWeightName,
        loadWeightMap(
            arena_alloc,
            &safetensors,
            &.{spec},
            &prefixes,
            0,
            &config,
            .{},
        ),
    );
}
