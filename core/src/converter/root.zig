//! Model Conversion Utilities
//!
//! Shared infrastructure for model conversion. NOT a generic walker -
//! just utilities that both grouped-affine (MLX export) and native converters need.
//!
//! Design principle: Keep format-specific logic in grouped_affine.zig and native.zig.
//! This module only contains genuinely shared code.

const std = @import("std");
const json = @import("../io/json/root.zig");
const log = @import("../log.zig");
const tensor_mod = @import("../tensor.zig");
const safetensors = @import("../io/safetensors/root.zig");
const dtype_mod = @import("../dtype.zig");
const op_types = @import("../models/op_types.zig");

pub const mapping = @import("mapping.zig");
pub const grouped_affine = @import("grouped_affine.zig");
pub const gaf_paths = @import("gaf_paths.zig");
pub const scheme = @import("scheme.zig");
pub const model_card = @import("model_card.zig");

// Re-export key types for convenience
pub const Role = mapping.Role;
pub const TensorInfo = mapping.TensorInfo;

// Re-export for backwards compatibility
pub const MLXConfig = gaf_paths.GAFConfig; // Alias for backwards compatibility
pub const MLXModelDir = gaf_paths.GAFModelDir; // Alias for backwards compatibility

// =============================================================================
// Tie Embeddings Logic
// =============================================================================

// =============================================================================
// Quantization Validation
// =============================================================================

/// Check if model contains already-quantized tensors.
/// We don't support re-quantizing quantized models.
pub fn isAlreadyQuantized(src: *safetensors.UnifiedSafeTensors) bool {
    return switch (src.*) {
        .single => |*s| isAlreadyQuantizedSingle(s),
        .sharded => |*s| isAlreadyQuantizedSharded(s),
    };
}

fn isAlreadyQuantizedSingle(src: *safetensors.SafeTensors) bool {
    var entry_iter = src.entries.iterator();
    while (entry_iter.next()) |kv| {
        const entry = kv.value_ptr.*;
        if (isQuantizedDtype(entry.dtype)) return true;
    }
    return false;
}

fn isAlreadyQuantizedSharded(src: *safetensors.ShardedSafeTensors) bool {
    // Check already-loaded shards
    var shard_iter = src.shards.iterator();
    while (shard_iter.next()) |shard_kv| {
        var entry_iter = shard_kv.value_ptr.entries.iterator();
        while (entry_iter.next()) |kv| {
            const entry = kv.value_ptr.*;
            if (isQuantizedDtype(entry.dtype)) return true;
        }
    }
    return false;
}

fn isQuantizedDtype(dtype: dtype_mod.DType) bool {
    return switch (dtype) {
        .grouped_affine_u4, .grouped_affine_u8 => true,
        else => false,
    };
}

/// DEPRECATED: Use shouldQuantizeTensorByLayout for architecture-driven quantization.
/// Check if a tensor should be quantized based on name-derived role.
/// Only used as fallback when architecture metadata is not available.
pub fn shouldQuantizeTensor(info: mapping.TensorInfo, src_tensor: tensor_mod.Tensor) bool {
    // Role-based check first
    if (!mapping.shouldQuantize(info.role)) return false;

    // Only quantize 2D weight matrices
    if (src_tensor.n_dims != 2) return false;

    // Only quantize float tensors (including FP8 source models)
    switch (src_tensor.dtype) {
        .f32, .f16, .bf16, .f8_e4m3 => {},
        else => return false,
    }

    // Skip small tensors (< 1024 elements)
    if (src_tensor.numel < 1024) return false;

    return true;
}

// =============================================================================
// Architecture-Driven Quantization
// =============================================================================

/// Weight layout map for architecture-driven quantization decisions.
/// Maps full tensor names (e.g., "model.layers.0.self_attn.q_proj.weight") to their layout.
pub const WeightLayoutMap = struct {
    allocator: std.mem.Allocator,
    layouts: std.StringHashMap(op_types.WeightLayout),
    lm_head_names: std.StringHashMap(void),

    pub fn init(allocator: std.mem.Allocator) WeightLayoutMap {
        return .{
            .allocator = allocator,
            .layouts = std.StringHashMap(op_types.WeightLayout).init(allocator),
            .lm_head_names = std.StringHashMap(void).init(allocator),
        };
    }

    pub fn deinit(self: *WeightLayoutMap) void {
        // Free all the keys we allocated
        var iter = self.layouts.keyIterator();
        while (iter.next()) |key| {
            self.allocator.free(key.*);
        }
        self.layouts.deinit();

        var lm_iter = self.lm_head_names.keyIterator();
        while (lm_iter.next()) |key| {
            self.allocator.free(key.*);
        }
        self.lm_head_names.deinit();
    }

    /// Check if a tensor should be quantized based on its layout.
    /// Returns null if tensor is not in the map (unknown tensor).
    pub fn shouldQuantize(self: *const WeightLayoutMap, tensor_name: []const u8) ?bool {
        if (self.layouts.get(tensor_name)) |layout| {
            return switch (layout) {
                .linear, .embedding => true,
                .none, .conv1d_depthwise, .gaffine => false,
            };
        }
        return null; // Unknown tensor
    }

    /// Check whether a tensor name is an lm_head candidate from architecture metadata.
    pub fn isLmHead(self: *const WeightLayoutMap, tensor_name: []const u8) bool {
        return self.lm_head_names.contains(tensor_name);
    }
};

/// Build a weight layout map from architecture metadata.
/// This extracts layout information from the architecture's weight specs.
pub fn buildWeightLayoutMap(
    allocator: std.mem.Allocator,
    arch: *const op_types.Architecture,
    num_layers: usize,
) !WeightLayoutMap {
    var layout_map = WeightLayoutMap.init(allocator);
    errdefer layout_map.deinit();

    // Add global weights (embeddings, final norm, lm_head)
    for (arch.global_weights) |weight_spec| {
        // Global weights use candidate names directly
        for (weight_spec.candidates) |candidate| {
            const key = try allocator.dupe(u8, candidate);
            errdefer allocator.free(key);
            try layout_map.layouts.put(key, weight_spec.layout);

            // Track lm_head aliases for tied-embedding skip logic.
            if (std.mem.eql(u8, weight_spec.id, "lm_head")) {
                const lm_key = try allocator.dupe(u8, candidate);
                errdefer allocator.free(lm_key);
                try layout_map.lm_head_names.put(lm_key, {});
            }
        }
    }

    // Add block weights for each layer
    for (0..num_layers) |layer_idx| {
        // Get the weights for this layer (handles heterogeneous models)
        const weights = getWeightsForLayer(arch, layer_idx);

        for (weights) |weight_spec| {
            // Use architecture-provided candidates as the source of truth.
            // Candidates may be generated from weight_prefixes+id or explicitly overridden
            // for architecture-specific naming (e.g., Granite mamba/self_attn aliases).
            for (weight_spec.candidates) |candidate_template| {
                const full_name = try expandLayerPlaceholder(allocator, candidate_template, layer_idx);
                errdefer allocator.free(full_name);
                try layout_map.layouts.put(full_name, weight_spec.layout);
            }
        }
    }

    return layout_map;
}

/// Get the weight specs for a given layer (handles heterogeneous models).
fn getWeightsForLayer(arch: *const op_types.Architecture, layer_idx: usize) []const op_types.WeightSpec {
    if (arch.block_variants) |variants| {
        // Heterogeneous model - get weights from the appropriate variant
        const variant_idx = arch.getVariantIndex(layer_idx);
        if (variant_idx < variants.len) {
            return variants[variant_idx].weights;
        }
    }
    // Homogeneous model or fallback
    return arch.block_weights;
}

/// Expand a template with layer index.
/// E.g., "model.layers.{d}.self_attn.q_proj.weight" + 5 => "model.layers.5.self_attn.q_proj.weight"
fn expandLayerPlaceholder(allocator: std.mem.Allocator, template: []const u8, layer_idx: usize) ![]const u8 {
    var result = std.ArrayListUnmanaged(u8){};
    errdefer result.deinit(allocator);

    var i: usize = 0;
    while (i < template.len) {
        if (i + 2 < template.len and
            template[i] == '{' and
            template[i + 1] == 'd' and
            template[i + 2] == '}')
        {
            // Replace {d} with layer index
            try result.writer(allocator).print("{d}", .{layer_idx});
            i += 3;
        } else {
            try result.append(allocator, template[i]);
            i += 1;
        }
    }

    return try result.toOwnedSlice(allocator);
}

/// Check if a tensor should be quantized using architecture-driven layout information.
/// If the tensor is not present in the layout map, it is not quantized.
pub fn shouldQuantizeTensorByLayout(
    layout_map: ?*const WeightLayoutMap,
    tensor_name: []const u8,
    src_tensor: tensor_mod.Tensor,
) bool {
    const map = layout_map orelse return false;

    // Architecture metadata is the source of truth. Unknown tensors are kept in source precision.
    const should_quantize = map.shouldQuantize(tensor_name) orelse return false;
    if (!should_quantize) return false;

    // Validate tensor properties (applies regardless of layout source)

    // Only quantize 2D weight matrices
    if (src_tensor.n_dims != 2) return false;

    // Only quantize float tensors (including FP8 source models)
    switch (src_tensor.dtype) {
        .f32, .f16, .bf16, .f8_e4m3 => {},
        else => return false,
    }

    // Skip small tensors (< 1024 elements)
    if (src_tensor.numel < 1024) return false;

    return true;
}

/// Check if a tensor should be skipped due to tied embeddings.
/// Uses architecture-derived lm_head aliases when available.
pub fn shouldSkipForTiedEmbeddingsByName(
    layout_map: ?*const WeightLayoutMap,
    tensor_name: []const u8,
    tie_word_embeddings: bool,
) bool {
    if (!tie_word_embeddings) return false;
    if (layout_map) |map| return map.isLmHead(tensor_name);
    return false;
}

// =============================================================================
// Tensor Data Conversion
// =============================================================================

/// Result of tensorToF32 - tracks ownership for proper cleanup.
pub const F32Result = struct {
    data: []const u8,
    owned: ?[]f32, // Non-null if we allocated, null if borrowed

    pub fn deinit(self: F32Result, allocator: std.mem.Allocator) void {
        if (self.owned) |owned| allocator.free(owned);
    }

    pub fn asF32Slice(self: F32Result) []align(1) const f32 {
        // Use explicit align(1) pointer cast instead of bytesAsSlice.
        // bytesAsSlice checks alignment at runtime and panics if unaligned.
        // Source tensor data from mmap'd SafeTensors files may be unaligned.
        const len = self.data.len / @sizeOf(f32);
        return @as([*]align(1) const f32, @ptrCast(self.data.ptr))[0..len];
    }
};

/// Convert tensor data to F32, tracking allocation ownership.
/// Returns borrowed slice for F32 tensors, owned allocation for others.
pub fn tensorToF32(allocator: std.mem.Allocator, t: tensor_mod.Tensor) !F32Result {
    switch (t.dtype) {
        .f32 => return .{ .data = t.data()[0..t.data_size], .owned = null },
        .f16 => {
            const src_u16 = t.asSliceUnaligned(u16);
            const f32_values = try allocator.alloc(f32, src_u16.len);
            for (src_u16, f32_values) |src_val, *dst_ptr| dst_ptr.* = dtype_mod.fp16ToF32(src_val);
            return .{ .data = std.mem.sliceAsBytes(f32_values), .owned = f32_values };
        },
        .bf16 => {
            const src_u16 = t.asSliceUnaligned(u16);
            const f32_values = try allocator.alloc(f32, src_u16.len);
            for (src_u16, f32_values) |src_val, *dst_ptr| dst_ptr.* = dtype_mod.bf16ToF32(src_val);
            return .{ .data = std.mem.sliceAsBytes(f32_values), .owned = f32_values };
        },
        .f8_e4m3 => {
            const src_u8 = t.asSliceUnaligned(u8);
            const f32_values = try allocator.alloc(f32, src_u8.len);
            for (src_u8, f32_values) |src_val, *dst_ptr| dst_ptr.* = dtype_mod.fp8e4m3ToF32(src_val);
            return .{ .data = std.mem.sliceAsBytes(f32_values), .owned = f32_values };
        },
        else => return error.UnsupportedDType,
    }
}

/// Convert tensor data to F32 as owned allocation (always allocates).
fn tensorToF32Alloc(allocator: std.mem.Allocator, t: tensor_mod.Tensor) ![]f32 {
    const element_count: usize = t.numel;
    const f32_values = try allocator.alloc(f32, element_count);
    errdefer allocator.free(f32_values);

    switch (t.dtype) {
        .f32 => {
            const src_f32 = @as([*]align(1) const f32, @ptrCast(t.data().ptr))[0..element_count];
            @memcpy(f32_values, src_f32);
        },
        .f16 => {
            const src_u16 = @as([*]align(1) const u16, @ptrCast(t.data().ptr))[0..element_count];
            for (src_u16, f32_values) |src_val, *dst| dst.* = dtype_mod.fp16ToF32(src_val);
        },
        .bf16 => {
            const src_u16 = @as([*]align(1) const u16, @ptrCast(t.data().ptr))[0..element_count];
            for (src_u16, f32_values) |src_val, *dst| dst.* = dtype_mod.bf16ToF32(src_val);
        },
        .f8_e4m3 => {
            const src_u8 = @as([*]align(1) const u8, @ptrCast(t.data().ptr))[0..element_count];
            for (src_u8, f32_values) |src_val, *dst| dst.* = dtype_mod.fp8e4m3ToF32(src_val);
        },
        else => return error.UnsupportedDType,
    }

    return f32_values;
}

// =============================================================================
// Float Conversion Utilities
// =============================================================================

/// Convert F32 to BF16 (truncate lower 16 bits).
pub fn f32ToBf16(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    return @intCast(bits >> 16);
}

/// Convert F32 to F16 using IEEE 754 conversion.
pub fn f32ToF16(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    const sign: u32 = (bits >> 31) & 1;
    var exp: i32 = @intCast((bits >> 23) & 0xFF);
    var mantissa: u32 = bits & 0x7FFFFF;

    if (exp == 0xFF) {
        // Inf/NaN
        return @intCast((sign << 15) | 0x7C00 | (if (mantissa != 0) @as(u32, 0x200) else 0));
    }

    exp -= 127; // Unbias f32 exponent

    if (exp > 15) {
        // Overflow to inf
        return @intCast((sign << 15) | 0x7C00);
    }

    if (exp < -14) {
        // Underflow to zero or denormal
        if (exp < -24) return @intCast(sign << 15);
        // Denormal
        mantissa |= 0x800000;
        const shift: u5 = @intCast(-exp - 14 + 13);
        mantissa >>= shift;
        return @intCast((sign << 15) | (mantissa & 0x3FF));
    }

    // Normal number
    const f16_exp: u32 = @intCast(exp + 15);
    return @intCast((sign << 15) | (f16_exp << 10) | (mantissa >> 13));
}

/// Convert F16 bits to F32.
fn f16ToF32(bits: u16) f32 {
    return @floatCast(@as(f16, @bitCast(bits)));
}

// =============================================================================
// Tensor Ordering
// =============================================================================

/// Compare tensor names for consistent ordering.
/// Order: token_embd, blk.0.*, blk.1.*, ..., output_norm, output
fn compareTensorNames(a: []const u8, b: []const u8) bool {
    const order_a = getTensorOrderKey(a);
    const order_b = getTensorOrderKey(b);
    if (order_a != order_b) return order_a < order_b;
    // Same category - sort alphabetically
    return std.mem.lessThan(u8, a, b);
}

fn getTensorOrderKey(name: []const u8) u32 {
    // token_embd first (order 0)
    if (std.mem.indexOf(u8, name, "embed_tokens") != null) return 0;

    // Block tensors (order 1000 + layer_num)
    const info = mapping.parseHfName(name);
    if (info.layer) |layer| {
        return 1000 + layer;
    }

    // output_norm near end (order 100000)
    if (std.mem.indexOf(u8, name, "norm") != null) return 100000;

    // output/lm_head last (order 100001)
    if (std.mem.indexOf(u8, name, "lm_head") != null) return 100001;

    // Unknown - put at end
    return 200000;
}

/// Sort tensor names for consistent output ordering.
pub fn sortTensorNames(names: [][]const u8) void {
    std.mem.sort([]const u8, names, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            return compareTensorNames(a, b);
        }
    }.lessThan);
}

// =============================================================================
// Tests
// =============================================================================

test "F32Result.asF32Slice handles unaligned data" {
    // This test ensures that asF32Slice works correctly with unaligned data.
    // SafeTensors files can have tensor data at arbitrary offsets, so we must
    // handle unaligned f32 data without causing Bus errors.

    // Create a buffer with enough space to guarantee we can find an unaligned offset.
    // We need 16 bytes for data + up to 3 bytes offset to get unaligned.
    var buffer: [20]u8 = undefined;

    // Calculate offset needed to make the slice unaligned for f32 (not divisible by 4)
    const base_addr = @intFromPtr(&buffer);
    const unaligned_start: usize = switch (base_addr % 4) {
        0 => 1, // base is aligned, offset by 1
        1 => 0, // base is already unaligned
        2 => 1, // offset by 1 to get alignment 3
        3 => 0, // base is already unaligned
        else => unreachable,
    };

    const data_slice = buffer[unaligned_start .. unaligned_start + 16]; // 4 f32 values

    // Write known f32 values as bytes (little-endian)
    // 1.0f = 0x3F800000
    data_slice[0] = 0x00;
    data_slice[1] = 0x00;
    data_slice[2] = 0x80;
    data_slice[3] = 0x3F;
    // 2.0f = 0x40000000
    data_slice[4] = 0x00;
    data_slice[5] = 0x00;
    data_slice[6] = 0x00;
    data_slice[7] = 0x40;
    // 3.0f = 0x40400000
    data_slice[8] = 0x00;
    data_slice[9] = 0x00;
    data_slice[10] = 0x40;
    data_slice[11] = 0x40;
    // 4.0f = 0x40800000
    data_slice[12] = 0x00;
    data_slice[13] = 0x00;
    data_slice[14] = 0x80;
    data_slice[15] = 0x40;

    // Verify the address is actually unaligned
    const ptr_addr = @intFromPtr(data_slice.ptr);
    try std.testing.expect(ptr_addr % 4 != 0); // Must be unaligned for f32

    // Create F32Result with unaligned data (simulating borrowed F32 tensor data)
    const result = F32Result{ .data = data_slice, .owned = null };

    // This should NOT panic with Bus error - the fix ensures align(1) access
    const f32_slice = result.asF32Slice();

    try std.testing.expectEqual(@as(usize, 4), f32_slice.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), f32_slice[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), f32_slice[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), f32_slice[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), f32_slice[3], 0.001);
}

test "f32ToBf16 round trip" {
    const values = [_]f32{ 0.0, 1.0, -1.0, 0.5, 2.0, 0.001, 100.0 };
    for (values) |v| {
        const bf16 = f32ToBf16(v);
        const back = dtype_mod.bf16ToF32(bf16);
        try std.testing.expectApproxEqAbs(v, back, 0.01);
    }
}

test "compareTensorNames ordering" {
    // Embeddings come first
    try std.testing.expect(compareTensorNames("model.embed_tokens.weight", "model.layers.0.attn.weight"));
    // Lower layers before higher layers
    try std.testing.expect(compareTensorNames("model.layers.0.attn.weight", "model.layers.1.attn.weight"));
    // Layers before final norm
    try std.testing.expect(compareTensorNames("model.layers.5.attn.weight", "model.norm.weight"));
    // Final norm before lm_head
    try std.testing.expect(compareTensorNames("model.norm.weight", "lm_head.weight"));
}

test "f32ToBf16 converts common values" {
    // Zero
    try std.testing.expectEqual(@as(u16, 0x0000), f32ToBf16(0.0));
    // One
    try std.testing.expectEqual(@as(u16, 0x3F80), f32ToBf16(1.0));
    // Negative one
    try std.testing.expectEqual(@as(u16, 0xBF80), f32ToBf16(-1.0));
    // Two
    try std.testing.expectEqual(@as(u16, 0x4000), f32ToBf16(2.0));
}

test "f32ToBf16 preserves sign" {
    const positive = f32ToBf16(3.14);
    const negative = f32ToBf16(-3.14);

    // Sign bit should be different
    try std.testing.expect((positive & 0x8000) == 0); // Positive has sign bit 0
    try std.testing.expect((negative & 0x8000) != 0); // Negative has sign bit 1

    // Magnitude should be same (lower 15 bits)
    try std.testing.expectEqual(positive & 0x7FFF, negative & 0x7FFF);
}

test "f32ToF16 converts common values" {
    // Zero
    try std.testing.expectEqual(@as(u16, 0x0000), f32ToF16(0.0));
    // One
    try std.testing.expectEqual(@as(u16, 0x3C00), f32ToF16(1.0));
    // Negative one
    try std.testing.expectEqual(@as(u16, 0xBC00), f32ToF16(-1.0));
    // Two
    try std.testing.expectEqual(@as(u16, 0x4000), f32ToF16(2.0));
}

test "f32ToF16 handles overflow to infinity" {
    // F16 max is ~65504, larger values should become infinity
    const large_value = f32ToF16(100000.0);
    // F16 infinity is 0x7C00
    try std.testing.expectEqual(@as(u16, 0x7C00), large_value);
}

test "f32ToF16 handles underflow to zero" {
    // Very small values should underflow to zero
    const tiny_value = f32ToF16(1e-10);
    try std.testing.expectEqual(@as(u16, 0x0000), tiny_value);
}

test "f32ToF16 preserves sign" {
    const positive = f32ToF16(0.5);
    const negative = f32ToF16(-0.5);

    try std.testing.expect((positive & 0x8000) == 0);
    try std.testing.expect((negative & 0x8000) != 0);
}

test "shouldSkipForTiedEmbeddingsByName uses architecture lm_head aliases" {
    const allocator = std.testing.allocator;
    var layout_map = WeightLayoutMap.init(allocator);
    defer layout_map.deinit();

    const lm_head_name = "lm_head.weight";
    try layout_map.lm_head_names.put(try allocator.dupe(u8, lm_head_name), {});

    try std.testing.expect(shouldSkipForTiedEmbeddingsByName(&layout_map, lm_head_name, true));
    try std.testing.expect(!shouldSkipForTiedEmbeddingsByName(&layout_map, "model.embed_tokens.weight", true));
    try std.testing.expect(!shouldSkipForTiedEmbeddingsByName(&layout_map, lm_head_name, false));
    try std.testing.expect(!shouldSkipForTiedEmbeddingsByName(null, lm_head_name, true));
}

test "sortTensorNames orders embeddings first" {
    var names = [_][]const u8{
        "model.layers.0.self_attn.q_proj.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
    };

    sortTensorNames(&names);

    try std.testing.expectEqualStrings("model.embed_tokens.weight", names[0]);
}

test "sortTensorNames orders layers numerically" {
    var names = [_][]const u8{
        "model.layers.10.self_attn.weight",
        "model.layers.2.self_attn.weight",
        "model.layers.1.self_attn.weight",
    };

    sortTensorNames(&names);

    try std.testing.expectEqualStrings("model.layers.1.self_attn.weight", names[0]);
    try std.testing.expectEqualStrings("model.layers.2.self_attn.weight", names[1]);
    try std.testing.expectEqualStrings("model.layers.10.self_attn.weight", names[2]);
}

test "sortTensorNames orders lm_head last" {
    var names = [_][]const u8{
        "lm_head.weight",
        "model.norm.weight",
        "model.layers.0.weight",
        "model.embed_tokens.weight",
    };

    sortTensorNames(&names);

    try std.testing.expectEqualStrings("model.embed_tokens.weight", names[0]);
    try std.testing.expectEqualStrings("model.layers.0.weight", names[1]);
    try std.testing.expectEqualStrings("model.norm.weight", names[2]);
    try std.testing.expectEqualStrings("lm_head.weight", names[3]);
}

// =============================================================================
// File Copy Utilities (shared by both converters)
// =============================================================================

/// Copy config.json from source to output directory.
pub fn copyConfigFile(allocator: std.mem.Allocator, source_config_path: []const u8, output_path: []const u8) !void {
    const dst_config_path = try std.fs.path.join(allocator, &.{ output_path, "config.json" });
    defer allocator.free(dst_config_path);

    std.fs.cwd().copyFile(source_config_path, std.fs.cwd(), dst_config_path, .{}) catch |err| {
        if (err != error.FileNotFound) return err;
    };
}

/// Quantization config to inject into config.json
pub const QuantizationConfig = struct {
    quant_method: []const u8,
    quant_type: []const u8,
    bits: u8,
};

/// GAF/MLX quantization config format
pub const GAFQuantizationConfig = struct {
    group_size: usize,
    bits: usize,
};

/// Copy config.json with quantization_config injected.
/// Reads source config, adds/replaces quantization_config object, writes to output.
pub fn copyConfigFileWithQuantization(
    allocator: std.mem.Allocator,
    source_config_path: []const u8,
    output_dir: []const u8,
    quant_config: QuantizationConfig,
) !void {
    // Read source config
    const source_file = std.fs.cwd().openFile(source_config_path, .{}) catch |err| {
        if (err == error.FileNotFound) return;
        return err;
    };
    defer source_file.close();

    const source_content = try source_file.readToEndAlloc(allocator, 1024 * 1024); // 1MB max
    defer allocator.free(source_content);

    // Parse JSON
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

    // Build output JSON by serializing existing fields and appending quantization_config
    var output_buf = std.ArrayListUnmanaged(u8){};
    defer output_buf.deinit(allocator);

    try output_buf.append(allocator, '{');

    // Copy all existing fields except quantization_config
    var first_field = true;
    var iter = parsed.value.object.iterator();
    while (iter.next()) |kv| {
        // Skip existing quantization_config - we'll add our own
        if (std.mem.eql(u8, kv.key_ptr.*, "quantization_config")) continue;

        if (!first_field) try output_buf.append(allocator, ',');
        first_field = false;

        // Write key
        try output_buf.append(allocator, '"');
        try output_buf.appendSlice(allocator, kv.key_ptr.*);
        try output_buf.appendSlice(allocator, "\":");

        // Write value using std.json stringify
        const value_json = try std.json.Stringify.valueAlloc(allocator, kv.value_ptr.*, .{});
        defer allocator.free(value_json);
        try output_buf.appendSlice(allocator, value_json);
    }

    // Add quantization_config
    if (!first_field) try output_buf.append(allocator, ',');
    try output_buf.writer(allocator).print(
        "\"quantization_config\":{{\"quant_method\":\"{s}\",\"quant_type\":\"{s}\",\"bits\":{d}}}",
        .{ quant_config.quant_method, quant_config.quant_type, quant_config.bits },
    );

    try output_buf.append(allocator, '}');

    // Write output
    const dst_config_path = try std.fs.path.join(allocator, &.{ output_dir, "config.json" });
    defer allocator.free(dst_config_path);

    var dst_file = try std.fs.cwd().createFile(dst_config_path, .{});
    defer dst_file.close();
    try dst_file.writeAll(output_buf.items);
}

/// Copy config.json with GAF/MLX quantization info added.
/// Preserves all original fields (architecture-specific like full_attn_idxs, conv_dim, etc.)
/// and adds/replaces the "quantization" block.
pub fn copyConfigWithGAFQuantization(
    allocator: std.mem.Allocator,
    source_config_path: []const u8,
    output_dir: []const u8,
    quant_config: ?GAFQuantizationConfig,
) !void {
    // Read source config
    const source_file = std.fs.cwd().openFile(source_config_path, .{}) catch |err| {
        if (err == error.FileNotFound) return;
        return err;
    };
    defer source_file.close();

    const source_content = try source_file.readToEndAlloc(allocator, 1024 * 1024); // 1MB max
    defer allocator.free(source_content);

    // Parse JSON
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

    // Build output JSON by serializing existing fields and adding/replacing quantization
    var output_buf = std.ArrayListUnmanaged(u8){};
    defer output_buf.deinit(allocator);

    try output_buf.append(allocator, '{');

    // Copy all existing fields except quantization (we'll add our own)
    var first_field = true;
    var iter = parsed.value.object.iterator();
    while (iter.next()) |kv| {
        // Skip existing quantization - we'll add our own if quant_config is provided
        if (std.mem.eql(u8, kv.key_ptr.*, "quantization")) continue;

        if (!first_field) try output_buf.append(allocator, ',');
        first_field = false;

        // Write key
        try output_buf.append(allocator, '"');
        try output_buf.appendSlice(allocator, kv.key_ptr.*);
        try output_buf.appendSlice(allocator, "\":");

        // Write value using std.json stringify
        const value_json = try std.json.Stringify.valueAlloc(allocator, kv.value_ptr.*, .{});
        defer allocator.free(value_json);
        try output_buf.appendSlice(allocator, value_json);
    }

    // Add quantization if provided
    if (quant_config) |qc| {
        if (!first_field) try output_buf.append(allocator, ',');
        try output_buf.writer(allocator).print(
            "\"quantization\":{{\"group_size\":{d},\"bits\":{d}}}",
            .{ qc.group_size, qc.bits },
        );
    }

    try output_buf.append(allocator, '}');

    // Write output
    const dst_config_path = try std.fs.path.join(allocator, &.{ output_dir, "config.json" });
    defer allocator.free(dst_config_path);

    var dst_file = try std.fs.cwd().createFile(dst_config_path, .{});
    defer dst_file.close();
    try dst_file.writeAll(output_buf.items);
}

/// Copy tokenizer files from source directory to output directory.
/// Copies tokenizer.json and tokenizer_config.json if they exist.
///
/// Note: This copies fewer files than MLX conversion (gaf_paths.MLXModelDir.copyTokenizerFiles),
/// which also copies special_tokens_map.json, generation_config.json, merges.txt, vocab.json.
/// Native conversion intentionally copies only the minimal required files.
///
/// DEPRECATED: Use copyModelAssets instead for full asset preservation.
pub fn copyTokenizerFiles(allocator: std.mem.Allocator, source_dir: []const u8, output_path: []const u8) !void {
    const tokenizer_files = [_][]const u8{ "tokenizer.json", "tokenizer_config.json" };

    for (tokenizer_files) |filename| {
        const src_path = try std.fs.path.join(allocator, &.{ source_dir, filename });
        defer allocator.free(src_path);
        const dst_path = try std.fs.path.join(allocator, &.{ output_path, filename });
        defer allocator.free(dst_path);

        std.fs.cwd().copyFile(src_path, std.fs.cwd(), dst_path, .{}) catch |err| {
            if (err != error.FileNotFound) return err;
            // Silently skip missing files
        };
    }
}

/// Copy all model asset files from source directory to output directory.
/// This preserves all files required for inference except weights (which are converted).
///
/// Copies everything EXCEPT:
/// - Weight files: *.safetensors, *.bin, *.pt, *.pth, *.gguf
/// - Config file: config.json (handled separately by copyConfigFileWithQuantization)
/// - Directory artifacts: .git/, .DS_Store
///
/// This ensures chat_template, added_tokens.json, vocab.txt, merges.txt,
/// generation_config.json, LICENSE, and any custom *.py files are preserved.
pub fn copyModelAssets(allocator: std.mem.Allocator, source_dir: []const u8, output_dir: []const u8) !void {
    var dir = std.fs.cwd().openDir(source_dir, .{ .iterate = true }) catch |err| {
        if (err == error.FileNotFound) return;
        return err;
    };
    defer dir.close();

    var iter = dir.iterate();
    while (try iter.next()) |entry| {
        // Skip directories (we don't recurse)
        if (entry.kind == .directory) continue;

        const name = entry.name;

        // Skip weight files
        if (isWeightFile(name)) continue;

        // Skip config.json (handled by copyConfigFileWithQuantization)
        if (std.mem.eql(u8, name, "config.json")) continue;
        if (std.mem.eql(u8, name, "config.json.backup")) continue;

        // Skip hidden files and directory artifacts
        if (name.len > 0 and name[0] == '.') continue;

        // Copy this file
        const src_path = try std.fs.path.join(allocator, &.{ source_dir, name });
        defer allocator.free(src_path);
        const dst_path = try std.fs.path.join(allocator, &.{ output_dir, name });
        defer allocator.free(dst_path);

        std.fs.cwd().copyFile(src_path, std.fs.cwd(), dst_path, .{}) catch |err| {
            // Log but don't fail on copy errors (e.g., permission issues)
            log.warn("converter", "Failed to copy file", .{ .filename = name, .err = @errorName(err) });
        };
    }
}

/// Check if a filename is a weight file that should be skipped during asset copying.
fn isWeightFile(name: []const u8) bool {
    // Weight file extensions to skip
    const weight_extensions = [_][]const u8{
        ".safetensors",
        ".bin",
        ".pt",
        ".pth",
        ".gguf",
    };

    for (weight_extensions) |ext| {
        if (std.mem.endsWith(u8, name, ext)) return true;
    }

    // Also skip weight index files (e.g., model.safetensors.index.json)
    // These reference sharded weight files and shouldn't be copied when
    // converting to a single output file.
    const weight_index_patterns = [_][]const u8{
        ".safetensors.index.json",
        ".bin.index.json",
    };

    for (weight_index_patterns) |pattern| {
        if (std.mem.endsWith(u8, name, pattern)) return true;
    }

    // Skip download artifacts (partial downloads)
    if (std.mem.endsWith(u8, name, ".download")) return true;

    return false;
}

test "copyConfigFileWithQuantization adds quantization_config" {
    const allocator = std.testing.allocator;

    // Create temp directory
    const source_dir = "/tmp/talu_test_config_source";
    const output_dir = "/tmp/talu_test_config_output";
    std.fs.cwd().deleteTree(source_dir) catch {};
    std.fs.cwd().deleteTree(output_dir) catch {};
    try std.fs.cwd().makePath(source_dir);
    try std.fs.cwd().makePath(output_dir);
    defer {
        std.fs.cwd().deleteTree(source_dir) catch {};
        std.fs.cwd().deleteTree(output_dir) catch {};
    }

    // Create source config
    const source_config = "{\"vocab_size\":32000,\"hidden_size\":768}";
    const source_path = try std.fs.path.join(allocator, &.{ source_dir, "config.json" });
    defer allocator.free(source_path);
    {
        var file = try std.fs.cwd().createFile(source_path, .{});
        defer file.close();
        try file.writeAll(source_config);
    }

    // Copy with quantization
    try copyConfigFileWithQuantization(allocator, source_path, output_dir, .{
        .quant_method = "talu",
        .quant_type = "gaf4_64",
        .bits = 4,
    });

    // Read output and verify
    const output_path = try std.fs.path.join(allocator, &.{ output_dir, "config.json" });
    defer allocator.free(output_path);
    var file = try std.fs.cwd().openFile(output_path, .{});
    defer file.close();
    const content = try file.readToEndAlloc(allocator, 4096);
    defer allocator.free(content);

    // Verify quantization_config is present
    try std.testing.expect(std.mem.indexOf(u8, content, "\"quantization_config\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, content, "\"quant_method\":\"talu\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, content, "\"quant_type\":\"gaf4_64\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, content, "\"bits\":4") != null);

    // Verify original fields are preserved
    try std.testing.expect(std.mem.indexOf(u8, content, "\"vocab_size\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, content, "\"hidden_size\"") != null);
}

test "copyConfigFileWithQuantization replaces existing quantization_config" {
    const allocator = std.testing.allocator;

    const source_dir = "/tmp/talu_test_config_replace_source";
    const output_dir = "/tmp/talu_test_config_replace_output";
    std.fs.cwd().deleteTree(source_dir) catch {};
    std.fs.cwd().deleteTree(output_dir) catch {};
    try std.fs.cwd().makePath(source_dir);
    try std.fs.cwd().makePath(output_dir);
    defer {
        std.fs.cwd().deleteTree(source_dir) catch {};
        std.fs.cwd().deleteTree(output_dir) catch {};
    }

    // Create source config with existing quantization_config
    const source_config =
        \\{"vocab_size":32000,"quantization_config":{"old_field":"old_value"},"hidden_size":768}
    ;
    const source_path = try std.fs.path.join(allocator, &.{ source_dir, "config.json" });
    defer allocator.free(source_path);
    {
        var file = try std.fs.cwd().createFile(source_path, .{});
        defer file.close();
        try file.writeAll(source_config);
    }

    try copyConfigFileWithQuantization(allocator, source_path, output_dir, .{
        .quant_method = "talu",
        .quant_type = "gaf8_64",
        .bits = 8,
    });

    const output_path = try std.fs.path.join(allocator, &.{ output_dir, "config.json" });
    defer allocator.free(output_path);
    var file = try std.fs.cwd().openFile(output_path, .{});
    defer file.close();
    const content = try file.readToEndAlloc(allocator, 4096);
    defer allocator.free(content);

    // New values should be present
    try std.testing.expect(std.mem.indexOf(u8, content, "\"quant_type\":\"gaf8_64\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, content, "\"bits\":8") != null);

    // Old values should NOT be present
    try std.testing.expect(std.mem.indexOf(u8, content, "old_field") == null);
    try std.testing.expect(std.mem.indexOf(u8, content, "old_value") == null);
}

test "isWeightFile identifies weight extensions" {
    // Should be identified as weight files
    try std.testing.expect(isWeightFile("model.safetensors"));
    try std.testing.expect(isWeightFile("model-00001-of-00003.safetensors"));
    try std.testing.expect(isWeightFile("pytorch_model.bin"));
    try std.testing.expect(isWeightFile("model.pt"));
    try std.testing.expect(isWeightFile("model.pth"));
    try std.testing.expect(isWeightFile("model.gguf"));

    // Weight index files should also be identified
    try std.testing.expect(isWeightFile("model.safetensors.index.json"));
    try std.testing.expect(isWeightFile("pytorch_model.bin.index.json"));

    // Download artifacts should be identified
    try std.testing.expect(isWeightFile("model-00001-of-00003.safetensors.download"));
    try std.testing.expect(isWeightFile("weights.bin.download"));

    // Should NOT be identified as weight files
    try std.testing.expect(!isWeightFile("tokenizer.json"));
    try std.testing.expect(!isWeightFile("tokenizer_config.json"));
    try std.testing.expect(!isWeightFile("config.json"));
    try std.testing.expect(!isWeightFile("vocab.txt"));
    try std.testing.expect(!isWeightFile("merges.txt"));
    try std.testing.expect(!isWeightFile("README.md"));
    try std.testing.expect(!isWeightFile("LICENSE"));
    try std.testing.expect(!isWeightFile("added_tokens.json"));
    try std.testing.expect(!isWeightFile("special_tokens_map.json"));
    try std.testing.expect(!isWeightFile("generation_config.json"));
    try std.testing.expect(!isWeightFile("modeling_model.py"));
}

test "copyModelAssets copies non-weight files" {
    const allocator = std.testing.allocator;

    const source_dir = "/tmp/talu_test_assets_source";
    const output_dir = "/tmp/talu_test_assets_output";
    std.fs.cwd().deleteTree(source_dir) catch {};
    std.fs.cwd().deleteTree(output_dir) catch {};
    try std.fs.cwd().makePath(source_dir);
    try std.fs.cwd().makePath(output_dir);
    defer {
        std.fs.cwd().deleteTree(source_dir) catch {};
        std.fs.cwd().deleteTree(output_dir) catch {};
    }

    // Create source files
    const files_to_create = [_][]const u8{
        "tokenizer.json",
        "tokenizer_config.json",
        "added_tokens.json",
        "special_tokens_map.json",
        "vocab.txt",
        "merges.txt",
        "generation_config.json",
        "README.md",
        "LICENSE",
        "modeling_custom.py",
        "config.json", // Should NOT be copied
        "model.safetensors", // Should NOT be copied
        "pytorch_model.bin", // Should NOT be copied
        "model.safetensors.index.json", // Should NOT be copied (weight index)
        ".DS_Store", // Should NOT be copied
    };

    for (files_to_create) |filename| {
        const path = try std.fs.path.join(allocator, &.{ source_dir, filename });
        defer allocator.free(path);
        var file = try std.fs.cwd().createFile(path, .{});
        defer file.close();
        try file.writeAll("test content");
    }

    // Copy assets
    try copyModelAssets(allocator, source_dir, output_dir);

    // Verify expected files were copied
    const expected_copied = [_][]const u8{
        "tokenizer.json",
        "tokenizer_config.json",
        "added_tokens.json",
        "special_tokens_map.json",
        "vocab.txt",
        "merges.txt",
        "generation_config.json",
        "README.md",
        "LICENSE",
        "modeling_custom.py",
    };

    for (expected_copied) |filename| {
        const path = try std.fs.path.join(allocator, &.{ output_dir, filename });
        defer allocator.free(path);
        std.fs.cwd().access(path, .{}) catch |err| {
            std.debug.print("Expected file not found: {s}\n", .{filename});
            return err;
        };
    }

    // Verify files that should NOT be copied
    const expected_not_copied = [_][]const u8{
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        ".DS_Store",
    };

    for (expected_not_copied) |filename| {
        const path = try std.fs.path.join(allocator, &.{ output_dir, filename });
        defer allocator.free(path);
        if (std.fs.cwd().access(path, .{})) |_| {
            std.debug.print("File should NOT have been copied: {s}\n", .{filename});
            return error.UnexpectedFileCopied;
        } else |_| {
            // Expected - file should not exist
        }
    }
}

test "copyModelAssets preserves chat_template in tokenizer_config.json" {
    const allocator = std.testing.allocator;

    const source_dir = "/tmp/talu_test_chat_template_source";
    const output_dir = "/tmp/talu_test_chat_template_output";
    std.fs.cwd().deleteTree(source_dir) catch {};
    std.fs.cwd().deleteTree(output_dir) catch {};
    try std.fs.cwd().makePath(source_dir);
    try std.fs.cwd().makePath(output_dir);
    defer {
        std.fs.cwd().deleteTree(source_dir) catch {};
        std.fs.cwd().deleteTree(output_dir) catch {};
    }

    // Create tokenizer_config.json with chat_template
    const tokenizer_config_content =
        \\{"chat_template":"{% for message in messages %}{{message.content}}{% endfor %}","bos_token":"<s>"}
    ;
    const src_path = try std.fs.path.join(allocator, &.{ source_dir, "tokenizer_config.json" });
    defer allocator.free(src_path);
    {
        var file = try std.fs.cwd().createFile(src_path, .{});
        defer file.close();
        try file.writeAll(tokenizer_config_content);
    }

    // Copy assets
    try copyModelAssets(allocator, source_dir, output_dir);

    // Read and verify chat_template is preserved
    const dst_path = try std.fs.path.join(allocator, &.{ output_dir, "tokenizer_config.json" });
    defer allocator.free(dst_path);
    var file = try std.fs.cwd().openFile(dst_path, .{});
    defer file.close();
    const content = try file.readToEndAlloc(allocator, 4096);
    defer allocator.free(content);

    try std.testing.expect(std.mem.indexOf(u8, content, "chat_template") != null);
    try std.testing.expect(std.mem.indexOf(u8, content, "{% for message in messages %}") != null);
}

test "buildWeightLayoutMap uses weight candidates for block weights" {
    const allocator = std.testing.allocator;

    const qproj_candidates = [_][]const u8{
        "model.layers.{d}.self_attn.q_proj.weight",
    };
    const block_weights = [_]op_types.WeightSpec{
        .{
            .id = "mixer.q_proj.weight",
            .candidates = &qproj_candidates,
            .module_type = "Linear",
            .layout = .linear,
            .dtype = "float32",
            .required = true,
        },
    };
    const model_types = [_][]const u8{"test"};

    const arch = op_types.Architecture{
        .name = "test_arch",
        .model_types = &model_types,
        .block_weights = &block_weights,
    };

    var layout_map = try buildWeightLayoutMap(allocator, &arch, 2);
    defer layout_map.deinit();

    try std.testing.expectEqual(@as(?bool, true), layout_map.shouldQuantize("model.layers.0.self_attn.q_proj.weight"));
    try std.testing.expectEqual(@as(?bool, true), layout_map.shouldQuantize("model.layers.1.self_attn.q_proj.weight"));
    try std.testing.expectEqual(@as(?bool, null), layout_map.shouldQuantize("model.layers.0.mixer.q_proj.weight"));
}

test "shouldQuantizeTensorByLayout uses architecture layout for quantization policy" {
    const allocator = std.testing.allocator;

    var layout_map = WeightLayoutMap.init(allocator);
    defer layout_map.deinit();

    const linear_name = "model.layers.0.self_attn.q_proj.weight";
    const norm_name = "model.layers.0.self_attn.q_layernorm.weight";
    try layout_map.layouts.put(try allocator.dupe(u8, linear_name), .linear);
    try layout_map.layouts.put(try allocator.dupe(u8, norm_name), .none);

    const data = try allocator.alloc(f32, 32 * 64);
    defer allocator.free(data);
    @memset(data, 0.0);
    const t = tensor_mod.Tensor.view2DSlice(data, 32, 64);

    try std.testing.expect(shouldQuantizeTensorByLayout(&layout_map, linear_name, t));
    try std.testing.expect(!shouldQuantizeTensorByLayout(&layout_map, norm_name, t));
}

test "shouldQuantizeTensorByLayout keeps unknown tensors in source precision" {
    const allocator = std.testing.allocator;

    const data = try allocator.alloc(f32, 32 * 64);
    defer allocator.free(data);
    @memset(data, 0.0);
    const t = tensor_mod.Tensor.view2DSlice(data, 32, 64);

    try std.testing.expect(!shouldQuantizeTensorByLayout(null, "model.layers.0.feed_forward.w1.weight", t));
    try std.testing.expect(!shouldQuantizeTensorByLayout(null, "model.layers.0.self_attn.out_proj.weight", t));
    try std.testing.expect(!shouldQuantizeTensorByLayout(null, "model.layers.0.conv.in_proj.weight", t));
}
