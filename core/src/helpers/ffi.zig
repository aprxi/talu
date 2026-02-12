//! FFI Conversion Utilities
//!
//! Core logic for converting between Zig types and C-compatible types.
//! The capi/ layer should delegate to these functions rather than implementing
//! conversion logic inline.
//!
//! ## Design Principle
//!
//! The capi/ functions are thin wrappers - they only do argument validation,
//! conversion, and error mapping. Any logic involving loops, complex allocations,
//! or multi-step conversions belongs here in helpers/ffi.zig instead.
//!
//! ## Usage
//!
//! ```zig
//! const ffi = @import("helpers/ffi.zig");
//!
//! // Convert Zig string slice to C string list
//! const list = try ffi.StringList.fromSlices(allocator, strings);
//! defer list.deinit(allocator);
//! ```

const std = @import("std");

/// A list of null-terminated strings for C interop.
/// This is the core type - capi modules can re-export or wrap as needed.
pub const StringList = struct {
    items: [][:0]const u8,

    const Self = @This();

    /// Convert a slice of Zig strings to a C-compatible StringList.
    /// All strings are copied with null terminators.
    /// Caller owns the result and must call deinit().
    pub fn fromSlices(allocator: std.mem.Allocator, strings: []const []const u8) !Self {
        var items = std.ArrayListUnmanaged([:0]const u8){};
        errdefer {
            for (items.items) |item| allocator.free(item);
            items.deinit(allocator);
        }

        for (strings) |s| {
            const cstr = try allocator.allocSentinel(u8, s.len, 0);
            errdefer allocator.free(cstr);
            @memcpy(cstr, s);
            try items.append(allocator, cstr);
        }

        return Self{
            .items = try items.toOwnedSlice(allocator),
        };
    }

    /// Free all strings and the items array.
    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        for (self.items) |item| {
            allocator.free(item);
        }
        allocator.free(self.items);
        self.items = &.{};
    }

    /// Get the number of strings.
    pub fn count(self: *const Self) usize {
        return self.items.len;
    }

    /// Get string at index, or null if out of bounds.
    pub fn get(self: *const Self, idx: usize) ?[:0]const u8 {
        if (idx >= self.items.len) return null;
        return self.items[idx];
    }
};

// =============================================================================
// C Callback Adapters
// =============================================================================

/// C callback type for getting token info by ID.
/// Returns pointer to token bytes and sets out_len to the length.
/// Returns null if token ID is invalid.
pub const TokenInfoCallback = *const fn (
    token_id: u32,
    out_len: *usize,
    ctx: ?*anyopaque,
) callconv(.c) ?[*]const u8;

/// Adapter that wraps a C callback as a tokenizer interface.
/// Used to bridge C callbacks into Zig code that expects a tokenizer with
/// getVocabSize() and idToToken() methods.
pub const CallbackTokenizer = struct {
    callback: TokenInfoCallback,
    ctx: ?*anyopaque,
    vocab_size: usize,

    pub fn getVocabSize(self: @This()) usize {
        return self.vocab_size;
    }

    pub fn idToToken(self: @This(), id: i32) ?[]const u8 {
        if (id < 0) return null;
        var len: usize = 0;
        const ptr = self.callback(@intCast(id), &len, self.ctx);
        if (ptr == null or len == 0) return null;
        return ptr.?[0..len];
    }
};

// =============================================================================
// Tensor Validation Utilities
// =============================================================================

const tensor_mod = @import("../tensor.zig");

/// Check if a dtype is a float type (f32, f16, bf16).
/// Used for validating tensor dtypes in ops.
pub fn isFloatDType(dtype: tensor_mod.DType) bool {
    return dtype == .f32 or dtype == .f16 or dtype == .bf16;
}

/// Check if two tensors have matching shapes.
/// Returns true if both have the same number of dimensions and all dimension sizes match.
pub fn shapesMatch(a: *const tensor_mod.Tensor, b: *const tensor_mod.Tensor) bool {
    if (a.n_dims != b.n_dims) return false;
    const dim_count: usize = @intCast(a.n_dims);
    for (0..dim_count) |dim_idx| {
        if (a.shape[dim_idx] != b.shape[dim_idx]) return false;
    }
    return true;
}

/// Check if two tensors have matching dtype and device.
pub fn tensorDeviceAndDTypeMatch(a: *const tensor_mod.Tensor, b: *const tensor_mod.Tensor) bool {
    return a.dtype == b.dtype and
        a.device.device_type == b.device.device_type and
        a.device.device_id == b.device.device_id;
}

// =============================================================================
// Linear Op Validation
// =============================================================================

/// Error type for linear/matmul validation.
pub const LinearValidationError = enum {
    input_rank_zero,
    weight_not_2d,
    in_features_mismatch,
    dtype_mismatch,
    device_mismatch,
    matmul_rank_too_low,
    matmul_inner_mismatch,
    bias_not_1d,
    bias_size_mismatch,
    bias_dtype_mismatch,
};

/// Result of validating linear parameters.
pub const LinearValidationResult = struct {
    input_rank: usize,
    out_features: usize,
};

/// Validate parameters for talu_linear (input @ weight^T).
/// Returns validated dimensions or null (caller should set capi_error before calling).
pub fn validateLinearParams(
    input: *const tensor_mod.Tensor,
    weight: *const tensor_mod.Tensor,
) ?LinearValidationResult {
    const input_rank = @as(usize, @intCast(input.n_dims));
    const weight_rank = @as(usize, @intCast(weight.n_dims));

    if (input_rank == 0) return null;
    if (weight_rank != 2) return null;
    if (input.shape[input_rank - 1] != weight.shape[1]) return null;
    if (!tensorDeviceAndDTypeMatch(input, weight)) return null;

    return .{
        .input_rank = input_rank,
        .out_features = @intCast(weight.shape[0]),
    };
}

/// Validate parameters for talu_linear including optional bias.
/// Returns error enum for specific error message, or null on success.
pub fn validateLinearBiasParams(
    input: *const tensor_mod.Tensor,
    weight: *const tensor_mod.Tensor,
    bias: ?*const tensor_mod.Tensor,
) ?LinearValidationError {
    const input_rank = @as(usize, @intCast(input.n_dims));
    const weight_rank = @as(usize, @intCast(weight.n_dims));

    if (input_rank == 0) return .input_rank_zero;
    if (weight_rank != 2) return .weight_not_2d;
    if (input.shape[input_rank - 1] != weight.shape[1]) return .in_features_mismatch;
    if (input.dtype != weight.dtype) return .dtype_mismatch;
    if (input.device.device_type != weight.device.device_type or
        input.device.device_id != weight.device.device_id) return .device_mismatch;

    const out_features: usize = @intCast(weight.shape[0]);

    if (bias) |b| {
        if (@as(usize, @intCast(b.n_dims)) != 1) return .bias_not_1d;
        if (@as(usize, @intCast(b.shape[0])) != out_features) return .bias_size_mismatch;
        if (b.dtype != input.dtype) return .bias_dtype_mismatch;
    }

    return null; // No error
}

/// Validate parameters for talu_matmul (a @ b).
/// Returns error enum for specific error message, or null on success.
pub fn validateMatmulParams(
    a: *const tensor_mod.Tensor,
    b: *const tensor_mod.Tensor,
) ?LinearValidationError {
    const a_rank = @as(usize, @intCast(a.n_dims));
    const b_rank = @as(usize, @intCast(b.n_dims));

    if (a_rank < 2 or b_rank < 2) return .matmul_rank_too_low;
    if (a.shape[a_rank - 1] != b.shape[b_rank - 2]) return .matmul_inner_mismatch;
    if (a.dtype != b.dtype) return .dtype_mismatch;
    if (a.device.device_type != b.device.device_type or
        a.device.device_id != b.device.device_id) return .device_mismatch;

    return null; // No error
}

// =============================================================================
// Attention Validation
// =============================================================================

/// Error type for attention validation.
pub const AttentionValidationError = enum {
    q_dtype_not_float,
    k_dtype_mismatch,
    v_dtype_mismatch,
    cos_dtype_mismatch,
    sin_dtype_mismatch,
    k_device_mismatch,
    v_device_mismatch,
    cos_device_mismatch,
    sin_device_mismatch,
    mask_dtype_not_f32,
    mask_device_mismatch,
};

/// Validate Q, K, V tensors have matching float dtypes and devices.
/// Returns error enum or null on success.
pub fn validateQKVParams(
    q: *const tensor_mod.Tensor,
    k: *const tensor_mod.Tensor,
    v: *const tensor_mod.Tensor,
) ?AttentionValidationError {
    if (!isFloatDType(q.dtype)) return .q_dtype_not_float;
    if (k.dtype != q.dtype) return .k_dtype_mismatch;
    if (v.dtype != q.dtype) return .v_dtype_mismatch;
    if (k.device.device_type != q.device.device_type or k.device.device_id != q.device.device_id) return .k_device_mismatch;
    if (v.device.device_type != q.device.device_type or v.device.device_id != q.device.device_id) return .v_device_mismatch;
    return null;
}

/// Validate RoPE tensors (Q, K, cos, sin) have matching dtypes and devices.
/// Returns error enum or null on success.
pub fn validateRopeParams(
    q: *const tensor_mod.Tensor,
    k: *const tensor_mod.Tensor,
    cos_tensor: *const tensor_mod.Tensor,
    sin_tensor: *const tensor_mod.Tensor,
) ?AttentionValidationError {
    if (!isFloatDType(q.dtype)) return .q_dtype_not_float;
    if (k.dtype != q.dtype) return .k_dtype_mismatch;
    if (cos_tensor.dtype != q.dtype) return .cos_dtype_mismatch;
    if (sin_tensor.dtype != q.dtype) return .sin_dtype_mismatch;
    if (k.device.device_type != q.device.device_type or k.device.device_id != q.device.device_id) return .k_device_mismatch;
    if (cos_tensor.device.device_type != q.device.device_type or cos_tensor.device.device_id != q.device.device_id) return .cos_device_mismatch;
    if (sin_tensor.device.device_type != q.device.device_type or sin_tensor.device.device_id != q.device.device_id) return .sin_device_mismatch;
    return null;
}

/// Validate optional mask tensor for SDPA.
/// Returns error enum or null on success.
pub fn validateMaskParam(
    mask: *const tensor_mod.Tensor,
    q: *const tensor_mod.Tensor,
) ?AttentionValidationError {
    if (mask.dtype != .f32) return .mask_dtype_not_f32;
    if (mask.device.device_type != q.device.device_type or mask.device.device_id != q.device.device_id) return .mask_device_mismatch;
    return null;
}

// =============================================================================
// Norm Op Validation
// =============================================================================

/// Error type for layer norm/RMS norm validation.
pub const NormValidationError = enum {
    input_no_dims,
    weight_shape_mismatch,
    input_dtype_not_float,
    weight_dtype_mismatch,
    weight_device_mismatch,
    bias_shape_mismatch,
    bias_dtype_mismatch,
    bias_device_mismatch,
};

/// Validate parameters for layer norm (includes optional bias).
/// Returns error enum for specific error message, or null on success.
pub fn validateLayerNormParams(
    input: *const tensor_mod.Tensor,
    weight: *const tensor_mod.Tensor,
    bias: ?*const tensor_mod.Tensor,
) ?NormValidationError {
    if (input.n_dims == 0) return .input_no_dims;
    const last_dim = input.shape[@as(usize, @intCast(input.n_dims)) - 1];
    if (weight.shape[0] != last_dim) return .weight_shape_mismatch;
    if (!isFloatDType(input.dtype)) return .input_dtype_not_float;
    if (weight.dtype != input.dtype) return .weight_dtype_mismatch;
    if (weight.device.device_type != input.device.device_type or
        weight.device.device_id != input.device.device_id) return .weight_device_mismatch;

    if (bias) |b| {
        if (b.shape[0] != weight.shape[0]) return .bias_shape_mismatch;
        if (b.dtype != input.dtype) return .bias_dtype_mismatch;
        if (b.device.device_type != input.device.device_type or
            b.device.device_id != input.device.device_id) return .bias_device_mismatch;
    }
    return null;
}

/// Validate parameters for RMS norm (no bias).
/// Returns error enum for specific error message, or null on success.
pub fn validateRmsNormParams(
    input: *const tensor_mod.Tensor,
    weight: *const tensor_mod.Tensor,
) ?NormValidationError {
    return validateLayerNormParams(input, weight, null);
}

// =============================================================================
// Embedding Validation
// =============================================================================

/// Error type for embedding validation.
pub const EmbeddingValidationError = enum {
    weight_not_2d,
    indices_no_dims,
    indices_dtype_invalid,
    weight_dtype_not_float,
};

/// Validate parameters for embedding lookup.
/// Returns error enum for specific error message, or null on success.
pub fn validateEmbeddingParams(
    weight: *const tensor_mod.Tensor,
    indices: *const tensor_mod.Tensor,
) ?EmbeddingValidationError {
    const weight_rank = @as(usize, @intCast(weight.n_dims));
    const index_rank = @as(usize, @intCast(indices.n_dims));

    if (weight_rank != 2) return .weight_not_2d;
    if (index_rank == 0) return .indices_no_dims;
    if (indices.dtype != .i32 and indices.dtype != .i64) return .indices_dtype_invalid;
    if (weight.dtype != .f32 and weight.dtype != .f16 and weight.dtype != .bf16) return .weight_dtype_not_float;
    return null;
}

// =============================================================================
// MXFP4 Linear Validation
// =============================================================================

/// Error type for MXFP4 linear validation.
pub const Mxfp4ValidationError = enum {
    input_dtype_invalid,
    input_rank_invalid,
    features_zero,
};

/// Validate parameters for MXFP4 linear layer.
/// Returns error enum for specific error message, or null on success.
pub fn validateMxfp4LinearParams(input: *const tensor_mod.Tensor, out_features: usize) ?Mxfp4ValidationError {
    if (input.dtype != .f32 and input.dtype != .bf16) return .input_dtype_invalid;
    if (input.n_dims < 1 or input.n_dims > 2) return .input_rank_invalid;
    const input_rank = @as(usize, @intCast(input.n_dims));
    const in_features: usize = @intCast(input.shape[input_rank - 1]);
    if (in_features == 0 or out_features == 0) return .features_zero;
    return null;
}

// =============================================================================
// JSON Utilities
// =============================================================================

/// Builds a JSON array of strings from multiple slices.
/// Returns a null-terminated string like `["a","b","c"]`.
/// Caller owns the result.
pub fn buildJsonStringArray(alloc: std.mem.Allocator, slices: []const []const []const u8) ![:0]u8 {
    var buffer = std.ArrayListUnmanaged(u8){};
    errdefer buffer.deinit(alloc);

    try buffer.append(alloc, '[');

    var is_first = true;
    for (slices) |slice| {
        for (slice) |s| {
            if (!is_first) try buffer.append(alloc, ',');
            try buffer.append(alloc, '"');
            try buffer.appendSlice(alloc, s);
            try buffer.append(alloc, '"');
            is_first = false;
        }
    }

    try buffer.append(alloc, ']');

    // Allocate with null terminator
    const result = try alloc.allocSentinel(u8, buffer.items.len, 0);
    @memcpy(result, buffer.items);
    buffer.deinit(alloc);
    return result;
}

// =============================================================================
// Tests
// =============================================================================

test "buildJsonStringArray creates valid JSON" {
    const alloc = std.testing.allocator;
    const slice1 = &[_][]const u8{ "a", "b" };
    const slice2 = &[_][]const u8{"c"};

    const result = try buildJsonStringArray(alloc, &.{ slice1, slice2 });
    defer alloc.free(result);

    try std.testing.expectEqualStrings("[\"a\",\"b\",\"c\"]", result);
}

test "buildJsonStringArray handles empty input" {
    const alloc = std.testing.allocator;
    const empty: []const []const u8 = &.{};

    const result = try buildJsonStringArray(alloc, &.{empty});
    defer alloc.free(result);

    try std.testing.expectEqualStrings("[]", result);
}

test "StringList.fromSlices creates valid list" {
    const allocator = std.testing.allocator;
    const strings = &[_][]const u8{ "hello", "world", "test" };

    var list = try StringList.fromSlices(allocator, strings);
    defer list.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 3), list.count());
    try std.testing.expectEqualStrings("hello", list.get(0).?);
    try std.testing.expectEqualStrings("world", list.get(1).?);
    try std.testing.expectEqualStrings("test", list.get(2).?);
    try std.testing.expect(list.get(3) == null);
}

test "StringList.fromSlices handles empty input" {
    const allocator = std.testing.allocator;
    const strings = &[_][]const u8{};

    var list = try StringList.fromSlices(allocator, strings);
    defer list.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), list.count());
    try std.testing.expect(list.get(0) == null);
}

test "StringList strings are null-terminated" {
    const allocator = std.testing.allocator;
    const strings = &[_][]const u8{"abc"};

    var list = try StringList.fromSlices(allocator, strings);
    defer list.deinit(allocator);

    const str = list.get(0).?;
    // Sentinel-terminated slice should have null at the end
    try std.testing.expectEqual(@as(u8, 0), str[str.len]);
}
