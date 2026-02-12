//! Generic graph-driven weight loader.
//!
//! Loads tensors using weight specifications from graph metadata and
//! applies transforms/layout handling without model-specific hardcoding.

const std = @import("std");
const tensor = @import("../../tensor.zig");
const st_loader = @import("../safetensors/root.zig");
const graph_types = @import("../../graph/types.zig");
const transforms = @import("transforms.zig");
const inference_mod = @import("../../inference/root.zig");
const log = @import("../../log.zig");

const Tensor = tensor.Tensor;
const Allocator = std.mem.Allocator;
const WeightSpec = graph_types.WeightSpec;
const WeightTransform = graph_types.WeightTransform;
const WeightMap = inference_mod.backend.block_kernels.WeightMap;

pub const LoadOptions = struct {
    use_metal_norms: bool = false,
    force_mamba_f32: bool = false,
};

pub fn loadWeightMap(
    allocator: Allocator,
    safetensors: *st_loader.UnifiedSafeTensors,
    specs: []const WeightSpec,
    layer_idx: usize,
    model_config: *const tensor.ModelConfig,
    options: LoadOptions,
) !WeightMap {
    var map = WeightMap{};
    errdefer map.deinit(allocator);

    for (specs) |spec| {
        if (try loadWeightBySpec(allocator, safetensors, spec, layer_idx, model_config, options)) |weight| {
            try map.put(allocator, spec.id, weight);
        }
    }

    return map;
}

fn loadWeightBySpec(
    allocator: Allocator,
    safetensors: *st_loader.UnifiedSafeTensors,
    spec: WeightSpec,
    layer_idx: usize,
    model_config: *const tensor.ModelConfig,
    options: LoadOptions,
) !?*const Tensor {
    var name_buf: [256]u8 = undefined;
    for (spec.candidates) |candidate| {
        const name = expandLayerTemplate(name_buf[0..], candidate, layer_idx) catch {
            continue;
        };
        const raw_tensor = safetensors.getTensor(name, null) catch {
            continue;
        };

        const transformed = applySpecTransforms(
            allocator,
            safetensors,
            name,
            raw_tensor,
            spec,
            model_config,
            options,
        ) catch |err| {
            log.err("load", "Failed to apply weight transforms", .{
                .id = spec.id,
                .name = name,
                .err = @errorName(err),
            }, @src());
            return err;
        };

        const weight_ptr = try allocator.create(Tensor); // lint:ignore errdefer-alloc - arena freed atomically
        weight_ptr.* = transformed;
        return weight_ptr;
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
        .linear, .gaffine => try transforms.orientWeight(allocator, safetensors, name, expected_in, model_config.*),
        .embedding => try transforms.orientEmbedding(allocator, safetensors, name, model_config.*),
        .conv1d_depthwise => try transforms.ensureF32(allocator, raw_tensor),
        .none => raw_tensor,
    };

    tensor_view = try maybeForceMambaF32(allocator, tensor_view, spec.id, options);
    if (options.force_mamba_f32 and isMambaKernelWeight(spec.id)) return tensor_view;

    const force_f32 = hasTransform(spec.transforms, .dtype_f32) or spec.layout == .conv1d_depthwise;
    if (force_f32 or (spec.layout == .none and !isNormModule(spec.module_type))) {
        if (!(options.use_metal_norms and isNormModule(spec.module_type))) {
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

fn maybeForceMambaF32(allocator: Allocator, tensor_view: Tensor, spec_id: []const u8, options: LoadOptions) !Tensor {
    if (!options.force_mamba_f32) return tensor_view;
    if (!isMambaKernelWeight(spec_id)) return tensor_view;
    return try transforms.ensureF32(allocator, tensor_view);
}

fn isMambaKernelWeight(spec_id: []const u8) bool {
    return std.mem.eql(u8, spec_id, "mixer.conv1d.weight") or
        std.mem.eql(u8, spec_id, "mixer.conv1d.bias") or
        std.mem.eql(u8, spec_id, "mixer.A_log") or
        std.mem.eql(u8, spec_id, "mixer.D") or
        std.mem.eql(u8, spec_id, "mixer.dt_bias") or
        std.mem.eql(u8, spec_id, "mixer.norm.weight");
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

test "loadWeightMap tested via integration tests" {
    // loadWeightMap requires a valid SafeTensors file handle and ModelConfig
    // which makes it unsuitable for unit testing in isolation.
    // Coverage is achieved through integration tests with actual model files.
    // The function delegates to loadWeightBySpec which is also integration-tested.
}

test "maybeForceMambaF32 converts only mamba kernel weights" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    var owned = try tensor.OwnedTensor.init(arena_alloc, .bf16, &.{ 2, 2 });
    defer owned.deinit();

    // Mamba SSM-specific weights (conv1d, A_log, D, dt_bias, norm) should be converted to F32
    const mamba_forced = try maybeForceMambaF32(arena_alloc, owned.view(), "mixer.A_log", .{ .force_mamba_f32 = true });
    try std.testing.expectEqual(tensor.DType.f32, mamba_forced.dtype);

    const in_proj_unforced = try maybeForceMambaF32(arena_alloc, owned.view(), "mixer.in_proj.weight", .{ .force_mamba_f32 = true });
    try std.testing.expectEqual(tensor.DType.bf16, in_proj_unforced.dtype);

    // MLP weights (even in Mamba blocks) should NOT be forced to F32 by this function
    const mlp_unforced = try maybeForceMambaF32(arena_alloc, owned.view(), "mlp.input_linear.weight", .{ .force_mamba_f32 = true });
    try std.testing.expectEqual(tensor.DType.bf16, mlp_unforced.dtype);
}
