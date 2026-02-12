//! Shape manipulation ops with stride-aware implementations.
//!
//! Key insight for zero-copy:
//! - unsqueeze, squeeze, reshape (when contiguous): just adjust shape/strides, share data
//! - expand: adjust strides (set to 0 for broadcast dims), share data
//! - cat, transpose: may require copy if result must be contiguous

const std = @import("std");
const tv = @import("tensor_view.zig");

const TensorView = tv.TensorView;
const DType = tv.DType;
const MAX_NDIM = tv.MAX_NDIM;

/// Dtype conversion helpers
const dtype_mod = @import("../../dtype.zig");
const f32ToFp16 = dtype_mod.f32ToFp16;
const fp16ToF32 = dtype_mod.fp16ToF32;
const bf16ToF32 = dtype_mod.bf16ToF32;
const f32ToBf16 = dtype_mod.f32ToBf16;

// =============================================================================
// Validation Helpers (used by capi to reduce boilerplate)
// =============================================================================

/// Error type for cat validation.
pub const CatValidationError = error{
    NoTensors,
    NullTensor,
    DimOutOfBounds,
    NdimMismatch,
    DTypeMismatch,
    DeviceMismatch,
    ShapeMismatch,
    TooManyTensors,
};

/// Result of validating cat parameters.
pub const CatValidationResult = struct {
    dim: usize,
    ndim: usize,
    total_dim: i64,
};

/// Normalize negative dimension index.
pub fn normalizeDim(dim_signed: i32, ndim: usize) ?usize {
    const ndim_i32: i32 = @intCast(ndim);
    var normalized = dim_signed;
    if (normalized < 0) normalized += ndim_i32;
    if (normalized < 0 or normalized >= ndim_i32) return null;
    return @intCast(normalized);
}

// =============================================================================
// Split Validation Helpers
// =============================================================================

/// Error type for split validation.
pub const SplitValidationError = enum {
    dim_out_of_bounds,
    num_splits_zero,
    num_splits_too_large,
    not_divisible,
};

/// Result of validating split parameters.
pub const SplitValidationResult = struct {
    input_ndim: usize,
    split_size: usize,
};

/// Validate split parameters.
/// Returns error enum or null on success.
pub fn validateSplitParams(input_ndim: usize, dim_size: i64, dim: usize, num_splits: usize) ?SplitValidationError {
    if (dim >= input_ndim) return .dim_out_of_bounds;
    if (num_splits == 0) return .num_splits_zero;
    if (num_splits > 32) return .num_splits_too_large;
    if (@mod(dim_size, @as(i64, @intCast(num_splits))) != 0) return .not_divisible;
    return null;
}

// =============================================================================
// Expand Validation Helpers
// =============================================================================

/// Error type for expand validation.
pub const ExpandValidationError = enum {
    new_ndim_less_than_input,
    new_ndim_exceeds_max,
    negative_dimension,
    cannot_broadcast,
};

/// Validate expand parameters and check broadcast compatibility.
/// Returns error enum or null on success.
pub fn validateExpandParams(input_ndim: usize, new_shape: []const i64, max_ndim: usize) ?ExpandValidationError {
    const new_ndim = new_shape.len;
    if (new_ndim < input_ndim) return .new_ndim_less_than_input;
    if (new_ndim > max_ndim) return .new_ndim_exceeds_max;
    for (new_shape) |dim_val| {
        if (dim_val < 0) return .negative_dimension;
    }
    return null;
}

/// Check if input shape can broadcast to new shape.
pub fn canBroadcast(input_shape: []const i64, input_ndim: usize, new_shape: []const usize, new_ndim: usize) bool {
    const offset = new_ndim - input_ndim;
    for (0..new_ndim) |dim_idx| {
        if (dim_idx < offset) continue;
        const in_idx = dim_idx - offset;
        const in_dim: usize = @intCast(input_shape[in_idx]);
        const out_dim = new_shape[dim_idx];
        if (in_dim != out_dim and in_dim != 1) return false;
    }
    return true;
}

/// Concatenate tensors along a dimension.
/// Output must be pre-allocated with correct shape.
/// Optimized for contiguous tensors (uses memcpy), falls back to element-wise for strided.
pub fn cat(
    comptime T: type,
    out: TensorView,
    inputs: []const TensorView,
    dim: usize,
) void {
    const out_data = @as([*]T, @ptrCast(@alignCast(out.data)));
    const ndim = out.ndim;

    // Check if all inputs are contiguous for fast path
    var all_contiguous = true;
    for (inputs) |input| {
        if (!input.isContiguous()) {
            all_contiguous = false;
            break;
        }
    }

    if (all_contiguous) {
        // Fast path: use memcpy for contiguous blocks
        // For concat along dim d:
        // - outer_size = product of dims [0, d)
        // - inner_size = product of dims (d, ndim)
        // Memory layout: for each outer index, copy (input.shape[d] * inner_size) elements

        var outer_size: usize = 1;
        for (0..dim) |dim_idx| {
            outer_size *= out.shape[dim_idx];
        }

        var inner_size: usize = 1;
        for ((dim + 1)..ndim) |dim_idx| {
            inner_size *= out.shape[dim_idx];
        }

        // For each outer slice, copy all inputs sequentially
        for (0..outer_size) |outer_idx| {
            var out_offset = outer_idx * out.shape[dim] * inner_size;
            for (inputs) |input| {
                const in_data = @as([*]const T, @ptrCast(@alignCast(input.data)));
                const chunk_size = input.shape[dim] * inner_size;
                const in_offset = outer_idx * chunk_size;
                @memcpy(out_data[out_offset..][0..chunk_size], in_data[in_offset..][0..chunk_size]);
                out_offset += chunk_size;
            }
        }
    } else {
        // Slow path: element-wise copy for strided tensors
        var dim_offset: usize = 0;
        for (inputs) |input| {
            const in_data = @as([*]const T, @ptrCast(@alignCast(input.data)));
            const in_dim_size = input.shape[dim];

            var coords: [MAX_NDIM]usize = undefined;
            for (0..input.numel) |elem_idx| {
                input.indexToCoords(elem_idx, &coords);
                const in_offset = input.coordsToOffset(coords[0..ndim]);
                coords[dim] += dim_offset;
                const out_offset = out.coordsToOffset(coords[0..ndim]);
                out_data[out_offset] = in_data[in_offset];
            }
            dim_offset += in_dim_size;
        }
    }
}

/// Concatenate with dtype dispatch
pub fn catDispatch(out: TensorView, inputs: []const TensorView, dim: usize) void {
    switch (out.dtype) {
        .f32 => cat(f32, out, inputs, dim),
        .f16, .bf16 => cat(u16, out, inputs, dim),
        .i32 => cat(i32, out, inputs, dim),
        .i64 => cat(i64, out, inputs, dim),
    }
}

/// Transpose two dimensions (copy-based for now).
/// Note: Could return a view for certain cases.
pub fn transpose(
    comptime T: type,
    out: TensorView,
    input: TensorView,
    dim0: usize,
    dim1: usize,
) void {
    const out_data = @as([*]T, @ptrCast(@alignCast(out.data)));
    const in_data = @as([*]const T, @ptrCast(@alignCast(input.data)));
    const ndim = input.ndim;

    var coords: [MAX_NDIM]usize = undefined;
    var out_coords: [MAX_NDIM]usize = undefined;

    for (0..input.numel) |elem_idx| {
        // Get input logical coordinates
        input.indexToCoords(elem_idx, &coords);

        // Swap dimensions for output
        for (0..ndim) |dim_idx| {
            out_coords[dim_idx] = coords[dim_idx];
        }
        out_coords[dim0] = coords[dim1];
        out_coords[dim1] = coords[dim0];

        // Get memory offsets
        const in_offset = input.coordsToOffset(coords[0..ndim]);
        const out_offset = out.coordsToOffset(out_coords[0..ndim]);

        out_data[out_offset] = in_data[in_offset];
    }
}

/// Transpose with dtype dispatch
pub fn transposeDispatch(out: TensorView, input: TensorView, dim0: usize, dim1: usize) void {
    switch (out.dtype) {
        .f32 => transpose(f32, out, input, dim0, dim1),
        .f16, .bf16 => transpose(u16, out, input, dim0, dim1),
        .i32 => transpose(i32, out, input, dim0, dim1),
        .i64 => transpose(i64, out, input, dim0, dim1),
    }
}

/// Unsqueeze: insert dimension of size 1.
/// This is a ZERO-COPY view operation - just adjusts shape/strides.
/// Returns new TensorView sharing same data.
pub fn unsqueeze(input: TensorView, dim: usize) TensorView {
    std.debug.assert(dim <= input.ndim);
    std.debug.assert(input.ndim < MAX_NDIM);

    var out = input;
    out.ndim = input.ndim + 1;

    // Shift shape and strides to make room for new dim
    var src_dim_idx: usize = input.ndim;
    while (src_dim_idx > dim) {
        src_dim_idx -= 1;
        out.shape[src_dim_idx + 1] = input.shape[src_dim_idx];
        out.strides[src_dim_idx + 1] = input.strides[src_dim_idx];
    }

    // Insert size-1 dimension
    out.shape[dim] = 1;
    // Stride doesn't matter for size-1 dim (any value works), use next stride
    out.strides[dim] = if (dim < input.ndim) input.strides[dim] else 1;

    return out;
}

/// Squeeze: remove dimensions of size 1.
/// This is a ZERO-COPY view operation.
pub fn squeeze(input: TensorView, dim: ?usize) TensorView {
    var out = input;

    if (dim) |d| {
        // Squeeze specific dimension
        if (input.shape[d] != 1) return input; // Nothing to squeeze

        out.ndim = input.ndim - 1;
        for (d..out.ndim) |dim_idx| {
            out.shape[dim_idx] = input.shape[dim_idx + 1];
            out.strides[dim_idx] = input.strides[dim_idx + 1];
        }
    } else {
        // Squeeze all size-1 dimensions
        var out_dim: usize = 0;
        for (0..input.ndim) |dim_idx| {
            if (input.shape[dim_idx] != 1) {
                out.shape[out_dim] = input.shape[dim_idx];
                out.strides[out_dim] = input.strides[dim_idx];
                out_dim += 1;
            }
        }
        out.ndim = out_dim;
    }

    return out;
}

/// Expand: broadcast to larger shape.
/// This is a ZERO-COPY view operation - uses stride 0 for broadcast dims.
pub fn expand(input: TensorView, new_shape: []const usize) TensorView {
    std.debug.assert(new_shape.len >= input.ndim);

    var out: TensorView = undefined;
    out.data = input.data;
    out.dtype = input.dtype;
    out.ndim = new_shape.len;

    // Align dimensions from the right
    const offset = new_shape.len - input.ndim;
    var numel: usize = 1;

    for (0..new_shape.len) |shape_idx| {
        out.shape[shape_idx] = new_shape[shape_idx];
        numel *= new_shape[shape_idx];

        if (shape_idx < offset) {
            // New dimension (prepended) - broadcast with stride 0
            out.strides[shape_idx] = 0;
        } else {
            const in_idx = shape_idx - offset;
            if (input.shape[in_idx] == new_shape[shape_idx]) {
                // Same size - keep original stride
                out.strides[shape_idx] = input.strides[in_idx];
            } else if (input.shape[in_idx] == 1) {
                // Broadcast from 1 - use stride 0
                out.strides[shape_idx] = 0;
            } else {
                // Invalid broadcast
                unreachable;
            }
        }
    }

    out.numel = numel;
    return out;
}

/// Reshape: change shape while preserving total elements.
/// ZERO-COPY when input is contiguous, otherwise requires copy.
pub fn reshapeView(input: TensorView, new_shape: []const usize) ?TensorView {
    // Verify numel matches
    var new_numel: usize = 1;
    for (new_shape) |dim| {
        new_numel *= dim;
    }
    std.debug.assert(new_numel == input.numel);

    // Can only reshape without copy if contiguous
    if (!input.isContiguous()) {
        return null; // Caller must allocate and copy
    }

    return TensorView.initContiguous(input.data, new_shape, input.dtype);
}

/// Reshape with copy (for non-contiguous inputs)
pub fn reshapeCopy(
    comptime T: type,
    out: TensorView,
    input: TensorView,
) void {
    const out_data = @as([*]T, @ptrCast(@alignCast(out.data)));
    const in_data = @as([*]const T, @ptrCast(@alignCast(input.data)));

    if (input.isContiguous()) {
        // Direct memcpy
        @memcpy(out_data[0..input.numel], in_data[0..input.numel]);
    } else {
        // Strided copy
        var coords: [MAX_NDIM]usize = undefined;
        for (0..input.numel) |elem_idx| {
            input.indexToCoords(elem_idx, &coords);
            const in_offset = input.coordsToOffset(coords[0..input.ndim]);
            out_data[elem_idx] = in_data[in_offset];
        }
    }
}

/// Slice a tensor along multiple dimensions.
/// Output must be pre-allocated.
/// `starts` and `ends` define the slice range [start, end) for each dimension.
pub fn slice(
    comptime T: type,
    out: TensorView,
    input: TensorView,
    starts: []const usize,
    ends: []const usize,
) void {
    _ = ends; // Slice range is implicit in output shape

    const out_data = @as([*]T, @ptrCast(@alignCast(out.data)));
    const in_data = @as([*]const T, @ptrCast(@alignCast(input.data)));
    const ndim = input.ndim;

    var out_coords: [MAX_NDIM]usize = undefined;
    var in_coords: [MAX_NDIM]usize = undefined;

    for (0..out.numel) |elem_idx| {
        out.indexToCoords(elem_idx, &out_coords);

        // Add start offsets to get input coords
        for (0..ndim) |dim_idx| {
            in_coords[dim_idx] = out_coords[dim_idx] + starts[dim_idx];
        }

        const in_offset = input.coordsToOffset(in_coords[0..ndim]);
        const out_offset = out.coordsToOffset(out_coords[0..ndim]);

        out_data[out_offset] = in_data[in_offset];
    }
}

/// Split a tensor along a dimension into equal parts.
/// Returns views (zero-copy) when input is contiguous and split is along last dim.
pub fn split(
    input: TensorView,
    dim: usize,
    num_splits: usize,
    out_views: []TensorView,
) void {
    std.debug.assert(out_views.len == num_splits);
    std.debug.assert(input.shape[dim] % num_splits == 0);

    const split_size = input.shape[dim] / num_splits;
    const elem_size = input.dtype.elementSize();

    for (out_views, 0..) |*out, split_idx| {
        // Create a view for this split
        out.* = input;
        out.shape[dim] = split_size;

        // Calculate data offset for this split
        var offset: usize = 0;
        if (dim == input.ndim - 1 and input.isContiguous()) {
            // Last dim + contiguous: simple offset
            offset = split_idx * split_size * elem_size;
        } else {
            // Need to compute offset using strides
            offset = split_idx * split_size * @as(usize, @intCast(input.strides[dim])) * elem_size;
        }
        out.data = input.data + offset;

        // Recalculate numel
        var numel: usize = 1;
        for (0..out.ndim) |dim_idx| {
            numel *= out.shape[dim_idx];
        }
        out.numel = numel;
    }
}

/// Repeat elements along a dimension.
/// Output must be pre-allocated with correct shape.
/// Optimized for contiguous tensors (uses memcpy), falls back to element-wise for strided.
pub fn repeatInterleave(
    comptime T: type,
    out: TensorView,
    input: TensorView,
    repeats: usize,
    dim: usize,
) void {
    const out_data = @as([*]T, @ptrCast(@alignCast(out.data)));
    const in_data = @as([*]const T, @ptrCast(@alignCast(input.data)));
    const ndim = input.ndim;

    if (input.isContiguous()) {
        // Fast path: copy blocks using memcpy
        // For repeat along dim d:
        // - outer_size = product of dims [0, d)
        // - dim_size = input.shape[d]
        // - inner_size = product of dims (d+1, ndim)
        // Each inner block is copied `repeats` times consecutively

        var outer_size: usize = 1;
        for (0..dim) |dim_idx| {
            outer_size *= input.shape[dim_idx];
        }

        const dim_size = input.shape[dim];

        var inner_size: usize = 1;
        for ((dim + 1)..ndim) |dim_idx| {
            inner_size *= input.shape[dim_idx];
        }

        const in_stride = dim_size * inner_size;
        const out_stride = dim_size * repeats * inner_size;

        for (0..outer_size) |outer_idx| {
            const in_base = outer_idx * in_stride;
            const out_base = outer_idx * out_stride;

            for (0..dim_size) |dim_idx| {
                const src_offset = in_base + dim_idx * inner_size;
                const dst_base = out_base + dim_idx * repeats * inner_size;

                // Copy the same inner block `repeats` times
                for (0..repeats) |repeat_idx| {
                    const dst_offset = dst_base + repeat_idx * inner_size;
                    @memcpy(out_data[dst_offset..][0..inner_size], in_data[src_offset..][0..inner_size]);
                }
            }
        }
    } else {
        // Slow path: element-wise copy for strided tensors
        var in_coords: [MAX_NDIM]usize = undefined; // Safe: indexToCoords fills completely
        var out_coords: [MAX_NDIM]usize = undefined; // Safe: indexToCoords fills completely

        for (0..out.numel) |elem_idx| {
            out.indexToCoords(elem_idx, &out_coords);

            for (0..ndim) |dim_idx| {
                if (dim_idx == dim) {
                    in_coords[dim_idx] = out_coords[dim_idx] / repeats;
                } else {
                    in_coords[dim_idx] = out_coords[dim_idx];
                }
            }

            const in_offset = input.coordsToOffset(in_coords[0..ndim]);
            const out_offset = out.coordsToOffset(out_coords[0..ndim]);

            out_data[out_offset] = in_data[in_offset];
        }
    }
}

test "unsqueeze is zero-copy" {
    var data = [_]f32{ 1, 2, 3, 4 };
    const input = TensorView.initContiguous(@ptrCast(&data), &.{ 2, 2 }, .f32);
    const out = unsqueeze(input, 0);

    try std.testing.expectEqual(@as(usize, 3), out.ndim);
    try std.testing.expectEqual(@as(usize, 1), out.shape[0]);
    try std.testing.expectEqual(@as(usize, 2), out.shape[1]);
    try std.testing.expectEqual(@as(usize, 2), out.shape[2]);
    try std.testing.expectEqual(input.data, out.data); // Same pointer!
}

test "expand uses stride 0" {
    var data = [_]f32{42};
    const input = TensorView.initContiguous(@ptrCast(&data), &.{1}, .f32);
    const out = expand(input, &.{ 3, 4 });

    try std.testing.expectEqual(@as(usize, 2), out.ndim);
    try std.testing.expectEqual(@as(usize, 3), out.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), out.shape[1]);
    try std.testing.expectEqual(@as(usize, 0), out.strides[0]); // Broadcast!
    try std.testing.expectEqual(@as(usize, 0), out.strides[1]); // Broadcast!
    try std.testing.expectEqual(input.data, out.data); // Same pointer!
}

test "squeeze removes size-1 dims" {
    var data = [_]f32{ 1, 2, 3, 4 };
    const input = TensorView.initContiguous(@ptrCast(&data), &.{ 1, 2, 1, 2 }, .f32);
    const out = squeeze(input, null);

    try std.testing.expectEqual(@as(usize, 2), out.ndim);
    try std.testing.expectEqual(@as(usize, 2), out.shape[0]);
    try std.testing.expectEqual(@as(usize, 2), out.shape[1]);
    try std.testing.expectEqual(input.data, out.data); // Same pointer!
}

const parallel = @import("../parallel.zig");

/// Top-k selection along last dimension.
/// Returns top-k values and their indices.
/// Parallelized over outer dimensions, O(n*k) per row (efficient for small k).
pub fn topk(
    comptime T: type,
    values_out: TensorView,
    indices_out: TensorView,
    input: TensorView,
    k: usize,
) void {
    const last_dim = input.shape[input.ndim - 1];
    const outer_size = input.numel / last_dim;

    const TopkCtx = struct {
        in_data: [*]const T,
        val_data: [*]T,
        idx_data: [*]i64,
        last_dim: usize,
        k: usize,

        fn runTopkRows(start: usize, end: usize, self: *@This()) void {
            for (start..end) |outer| {
                const in_offset = outer * self.last_dim;
                const out_offset = outer * self.k;

                // For small k, repeated max-finding is cache-friendly
                // Copy to temp buffer to avoid modifying input
                var topk_buf: [128]T = undefined;
                const copy_len = @min(self.last_dim, 128);
                for (0..copy_len) |elem_idx| {
                    topk_buf[elem_idx] = self.in_data[in_offset + elem_idx];
                }

                for (0..self.k) |ki| {
                    var max_val: T = -std.math.inf(T);
                    var max_idx: usize = 0;

                    // Find max in temp buffer
                    for (0..copy_len) |elem_idx| {
                        if (topk_buf[elem_idx] > max_val) {
                            max_val = topk_buf[elem_idx];
                            max_idx = elem_idx;
                        }
                    }

                    self.val_data[out_offset + ki] = max_val;
                    self.idx_data[out_offset + ki] = @intCast(max_idx);
                    topk_buf[max_idx] = -std.math.inf(T); // Mark as used
                }
            }
        }
    };

    var context = TopkCtx{
        .in_data = @as([*]const T, @ptrCast(@alignCast(input.data))),
        .val_data = @as([*]T, @ptrCast(@alignCast(values_out.data))),
        .idx_data = @as([*]i64, @ptrCast(@alignCast(indices_out.data))),
        .last_dim = last_dim,
        .k = k,
    };

    parallel.global().parallelFor(outer_size, TopkCtx.runTopkRows, &context);
}

// ============================================================================
// Tests for transpose
// ============================================================================

test "transpose 2D matrix" {
    const allocator = std.testing.allocator;

    // Create a 3x2 matrix
    var input_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const input = TensorView.initContiguous(@ptrCast(&input_data), &.{ 3, 2 }, .f32);

    // Allocate output for 2x3 transposed matrix
    const out_data = try allocator.alloc(f32, 6);
    defer allocator.free(out_data);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{ 2, 3 }, .f32);

    // Transpose dimensions 0 and 1
    transpose(f32, out, input, 0, 1);

    // Expected: [[1,3,5], [2,4,6]]
    try std.testing.expectEqual(@as(f32, 1), out_data[0]);
    try std.testing.expectEqual(@as(f32, 3), out_data[1]);
    try std.testing.expectEqual(@as(f32, 5), out_data[2]);
    try std.testing.expectEqual(@as(f32, 2), out_data[3]);
    try std.testing.expectEqual(@as(f32, 4), out_data[4]);
    try std.testing.expectEqual(@as(f32, 6), out_data[5]);
}

test "transpose 3D tensor - swap first two dims" {
    const allocator = std.testing.allocator;

    // Create a 2x3x2 tensor
    var input_data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const input = TensorView.initContiguous(@ptrCast(&input_data), &.{ 2, 3, 2 }, .f32);

    // Allocate output for 3x2x2 transposed tensor
    const out_data = try allocator.alloc(f32, 12);
    defer allocator.free(out_data);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{ 3, 2, 2 }, .f32);

    // Transpose dimensions 0 and 1
    transpose(f32, out, input, 0, 1);

    // Verify shape
    try std.testing.expectEqual(@as(usize, 3), out.shape[0]);
    try std.testing.expectEqual(@as(usize, 2), out.shape[1]);
    try std.testing.expectEqual(@as(usize, 2), out.shape[2]);

    // Verify data: input[0,0,:] = [1,2] should become output[0,0,:]
    try std.testing.expectEqual(@as(f32, 1), out_data[0]);
    try std.testing.expectEqual(@as(f32, 2), out_data[1]);

    // input[1,0,:] = [7,8] should become output[0,1,:]
    try std.testing.expectEqual(@as(f32, 7), out_data[2]);
    try std.testing.expectEqual(@as(f32, 8), out_data[3]);

    // input[0,1,:] = [3,4] should become output[1,0,:]
    try std.testing.expectEqual(@as(f32, 3), out_data[4]);
    try std.testing.expectEqual(@as(f32, 4), out_data[5]);
}

test "transpose 3D tensor - swap last two dims" {
    const allocator = std.testing.allocator;

    // Create a 2x2x3 tensor: shape [2,2,3]
    // input[0,0,:] = [1,2,3], input[0,1,:] = [4,5,6]
    // input[1,0,:] = [7,8,9], input[1,1,:] = [10,11,12]
    var input_data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const input = TensorView.initContiguous(@ptrCast(&input_data), &.{ 2, 2, 3 }, .f32);

    // Allocate output for 2x3x2 transposed tensor
    const out_data = try allocator.alloc(f32, 12);
    defer allocator.free(out_data);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{ 2, 3, 2 }, .f32);

    // Transpose dimensions 1 and 2: output[i,j,k] = input[i,k,j]
    transpose(f32, out, input, 1, 2);

    // For output shape [2,3,2], in row-major order:
    // out[0] = output[0,0,0] = input[0,0,0] = 1
    // out[1] = output[0,0,1] = input[0,1,0] = 4
    // out[2] = output[0,1,0] = input[0,0,1] = 2
    // out[3] = output[0,1,1] = input[0,1,1] = 5
    // out[4] = output[0,2,0] = input[0,0,2] = 3
    // out[5] = output[0,2,1] = input[0,1,2] = 6
    try std.testing.expectEqual(@as(f32, 1), out_data[0]);
    try std.testing.expectEqual(@as(f32, 4), out_data[1]);
    try std.testing.expectEqual(@as(f32, 2), out_data[2]);
    try std.testing.expectEqual(@as(f32, 5), out_data[3]);
    try std.testing.expectEqual(@as(f32, 3), out_data[4]);
    try std.testing.expectEqual(@as(f32, 6), out_data[5]);
}

test "transpose identity - swapping same dimension" {
    const allocator = std.testing.allocator;

    // Create a 3x4 matrix
    var input_data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const input = TensorView.initContiguous(@ptrCast(&input_data), &.{ 3, 4 }, .f32);

    // Allocate output
    const out_data = try allocator.alloc(f32, 12);
    defer allocator.free(out_data);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{ 3, 4 }, .f32);

    // Transpose dimension 0 with itself (should be identity)
    transpose(f32, out, input, 0, 0);

    // Output should be identical to input
    for (input_data, 0..) |val, i| {
        try std.testing.expectEqual(val, out_data[i]);
    }
}

test "transpose with strided input" {
    const allocator = std.testing.allocator;

    // Create a 4x3 matrix
    var input_data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };

    // Create a strided view (every other row)
    var strided_input = TensorView.initContiguous(@ptrCast(&input_data), &.{ 4, 3 }, .f32);
    strided_input.shape[0] = 2; // Only 2 rows
    strided_input.strides[0] = 6; // Skip a row (stride of 2 rows)
    strided_input.numel = 6;

    // Allocate output for 3x2 transposed matrix
    const out_data = try allocator.alloc(f32, 6);
    defer allocator.free(out_data);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{ 3, 2 }, .f32);

    // Transpose
    transpose(f32, out, strided_input, 0, 1);

    // strided_input selects rows 0 and 2: [[1,2,3], [7,8,9]]
    // Transposed: [[1,7], [2,8], [3,9]]
    try std.testing.expectEqual(@as(f32, 1), out_data[0]);
    try std.testing.expectEqual(@as(f32, 7), out_data[1]);
    try std.testing.expectEqual(@as(f32, 2), out_data[2]);
    try std.testing.expectEqual(@as(f32, 8), out_data[3]);
    try std.testing.expectEqual(@as(f32, 3), out_data[4]);
    try std.testing.expectEqual(@as(f32, 9), out_data[5]);
}

test "transpose with different dtypes" {
    const allocator = std.testing.allocator;

    // Test with i32
    var input_i32 = [_]i32{ 10, 20, 30, 40 };
    const input = TensorView.initContiguous(@ptrCast(&input_i32), &.{ 2, 2 }, .i32);

    const out_data = try allocator.alloc(i32, 4);
    defer allocator.free(out_data);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{ 2, 2 }, .i32);

    transpose(i32, out, input, 0, 1);

    try std.testing.expectEqual(@as(i32, 10), out_data[0]);
    try std.testing.expectEqual(@as(i32, 30), out_data[1]);
    try std.testing.expectEqual(@as(i32, 20), out_data[2]);
    try std.testing.expectEqual(@as(i32, 40), out_data[3]);
}

test "transpose 4D tensor" {
    const allocator = std.testing.allocator;

    // Create a 2x2x2x2 tensor (16 elements)
    var input_data: [16]f32 = undefined;
    for (0..16) |i| {
        input_data[i] = @floatFromInt(i);
    }
    const input = TensorView.initContiguous(@ptrCast(&input_data), &.{ 2, 2, 2, 2 }, .f32);

    // Allocate output (same shape)
    const out_data = try allocator.alloc(f32, 16);
    defer allocator.free(out_data);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{ 2, 2, 2, 2 }, .f32);

    // Transpose dimensions 1 and 3
    transpose(f32, out, input, 1, 3);

    // Verify a few elements
    // input[0,0,0,0] = 0 -> output[0,0,0,0] = 0
    try std.testing.expectEqual(@as(f32, 0), out_data[0]);

    // input[0,0,0,1] = 1 -> output[0,1,0,0] = 1
    try std.testing.expectEqual(@as(f32, 1), out_data[4]);

    // input[0,1,0,0] = 4 -> output[0,0,0,1] = 4
    try std.testing.expectEqual(@as(f32, 4), out_data[1]);
}

test "transposeDispatch with f32" {
    const allocator = std.testing.allocator;

    var input_data = [_]f32{ 1, 2, 3, 4 };
    const input = TensorView.initContiguous(@ptrCast(&input_data), &.{ 2, 2 }, .f32);

    const out_data = try allocator.alloc(f32, 4);
    defer allocator.free(out_data);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{ 2, 2 }, .f32);

    transposeDispatch(out, input, 0, 1);

    try std.testing.expectEqual(@as(f32, 1), out_data[0]);
    try std.testing.expectEqual(@as(f32, 3), out_data[1]);
    try std.testing.expectEqual(@as(f32, 2), out_data[2]);
    try std.testing.expectEqual(@as(f32, 4), out_data[3]);
}

// ============================================================================
// Tests for slice
// ============================================================================

test "slice 2D matrix - single row" {
    const allocator = std.testing.allocator;

    // Create a 3x4 matrix
    var input_data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const input = TensorView.initContiguous(@ptrCast(&input_data), &.{ 3, 4 }, .f32);

    // Slice second row: [1:2, :]
    const starts = [_]usize{ 1, 0 };
    const ends = [_]usize{ 2, 4 };

    const out_data = try allocator.alloc(f32, 4);
    defer allocator.free(out_data);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{ 1, 4 }, .f32);

    slice(f32, out, input, &starts, &ends);

    // Should get [5, 6, 7, 8]
    try std.testing.expectEqual(@as(f32, 5), out_data[0]);
    try std.testing.expectEqual(@as(f32, 6), out_data[1]);
    try std.testing.expectEqual(@as(f32, 7), out_data[2]);
    try std.testing.expectEqual(@as(f32, 8), out_data[3]);
}

test "slice 2D matrix - single column" {
    const allocator = std.testing.allocator;

    // Create a 3x4 matrix
    var input_data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const input = TensorView.initContiguous(@ptrCast(&input_data), &.{ 3, 4 }, .f32);

    // Slice third column: [:, 2:3]
    const starts = [_]usize{ 0, 2 };
    const ends = [_]usize{ 3, 3 };

    const out_data = try allocator.alloc(f32, 3);
    defer allocator.free(out_data);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{ 3, 1 }, .f32);

    slice(f32, out, input, &starts, &ends);

    // Should get [3, 7, 11]
    try std.testing.expectEqual(@as(f32, 3), out_data[0]);
    try std.testing.expectEqual(@as(f32, 7), out_data[1]);
    try std.testing.expectEqual(@as(f32, 11), out_data[2]);
}

test "slice 2D matrix - submatrix" {
    const allocator = std.testing.allocator;

    // Create a 4x5 matrix
    var input_data: [20]f32 = undefined;
    for (0..20) |i| {
        input_data[i] = @floatFromInt(i);
    }
    const input = TensorView.initContiguous(@ptrCast(&input_data), &.{ 4, 5 }, .f32);

    // Slice [1:3, 1:4] - middle 2x3 submatrix
    const starts = [_]usize{ 1, 1 };
    const ends = [_]usize{ 3, 4 };

    const out_data = try allocator.alloc(f32, 6);
    defer allocator.free(out_data);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{ 2, 3 }, .f32);

    slice(f32, out, input, &starts, &ends);

    // Row 1: [6, 7, 8], Row 2: [11, 12, 13]
    try std.testing.expectEqual(@as(f32, 6), out_data[0]);
    try std.testing.expectEqual(@as(f32, 7), out_data[1]);
    try std.testing.expectEqual(@as(f32, 8), out_data[2]);
    try std.testing.expectEqual(@as(f32, 11), out_data[3]);
    try std.testing.expectEqual(@as(f32, 12), out_data[4]);
    try std.testing.expectEqual(@as(f32, 13), out_data[5]);
}

test "slice 3D tensor" {
    const allocator = std.testing.allocator;

    // Create a 3x3x3 tensor
    var input_data: [27]f32 = undefined;
    for (0..27) |i| {
        input_data[i] = @floatFromInt(i);
    }
    const input = TensorView.initContiguous(@ptrCast(&input_data), &.{ 3, 3, 3 }, .f32);

    // Slice [0:2, 1:3, 0:2] - 2x2x2 subtensor
    const starts = [_]usize{ 0, 1, 0 };
    const ends = [_]usize{ 2, 3, 2 };

    const out_data = try allocator.alloc(f32, 8);
    defer allocator.free(out_data);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{ 2, 2, 2 }, .f32);

    slice(f32, out, input, &starts, &ends);

    // First slice [0, 1:3, 0:2]: [[3,4], [6,7]]
    try std.testing.expectEqual(@as(f32, 3), out_data[0]);
    try std.testing.expectEqual(@as(f32, 4), out_data[1]);
    try std.testing.expectEqual(@as(f32, 6), out_data[2]);
    try std.testing.expectEqual(@as(f32, 7), out_data[3]);

    // Second slice [1, 1:3, 0:2]: [[12,13], [15,16]]
    try std.testing.expectEqual(@as(f32, 12), out_data[4]);
    try std.testing.expectEqual(@as(f32, 13), out_data[5]);
    try std.testing.expectEqual(@as(f32, 15), out_data[6]);
    try std.testing.expectEqual(@as(f32, 16), out_data[7]);
}

test "slice edge case - single element" {
    const allocator = std.testing.allocator;

    // Create a 3x3 matrix
    var input_data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const input = TensorView.initContiguous(@ptrCast(&input_data), &.{ 3, 3 }, .f32);

    // Slice single element [1:2, 1:2]
    const starts = [_]usize{ 1, 1 };
    const ends = [_]usize{ 2, 2 };

    const out_data = try allocator.alloc(f32, 1);
    defer allocator.free(out_data);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{ 1, 1 }, .f32);

    slice(f32, out, input, &starts, &ends);

    // Should get the center element (5)
    try std.testing.expectEqual(@as(f32, 5), out_data[0]);
}

test "slice with strided input" {
    const allocator = std.testing.allocator;

    // Create a 4x4 matrix
    var input_data: [16]f32 = undefined;
    for (0..16) |i| {
        input_data[i] = @floatFromInt(i);
    }

    // Create a strided view (every other row and column)
    var strided_input = TensorView.initContiguous(@ptrCast(&input_data), &.{ 4, 4 }, .f32);
    strided_input.shape[0] = 2;
    strided_input.shape[1] = 2;
    strided_input.strides[0] = 8; // Skip a row
    strided_input.strides[1] = 2; // Skip a column
    strided_input.numel = 4;

    // Slice [0:2, 0:2] from the strided view
    const starts = [_]usize{ 0, 0 };
    const ends = [_]usize{ 2, 2 };

    const out_data = try allocator.alloc(f32, 4);
    defer allocator.free(out_data);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{ 2, 2 }, .f32);

    slice(f32, out, strided_input, &starts, &ends);

    // Strided input selects: [[0,2], [8,10]]
    try std.testing.expectEqual(@as(f32, 0), out_data[0]);
    try std.testing.expectEqual(@as(f32, 2), out_data[1]);
    try std.testing.expectEqual(@as(f32, 8), out_data[2]);
    try std.testing.expectEqual(@as(f32, 10), out_data[3]);
}

test "slice full tensor - identity slice" {
    const allocator = std.testing.allocator;

    // Create a 2x3 matrix
    var input_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const input = TensorView.initContiguous(@ptrCast(&input_data), &.{ 2, 3 }, .f32);

    // Slice entire tensor [0:2, 0:3]
    const starts = [_]usize{ 0, 0 };
    const ends = [_]usize{ 2, 3 };

    const out_data = try allocator.alloc(f32, 6);
    defer allocator.free(out_data);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{ 2, 3 }, .f32);

    slice(f32, out, input, &starts, &ends);

    // Should get identical copy
    for (input_data, 0..) |val, i| {
        try std.testing.expectEqual(val, out_data[i]);
    }
}

test "slice with different dtypes" {
    const allocator = std.testing.allocator;

    // Test with i64
    var input_i64 = [_]i64{ 100, 200, 300, 400, 500, 600 };
    const input = TensorView.initContiguous(@ptrCast(&input_i64), &.{ 2, 3 }, .i64);

    // Slice [0:1, 1:3]
    const starts = [_]usize{ 0, 1 };
    const ends = [_]usize{ 1, 3 };

    const out_data = try allocator.alloc(i64, 2);
    defer allocator.free(out_data);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{ 1, 2 }, .i64);

    slice(i64, out, input, &starts, &ends);

    try std.testing.expectEqual(@as(i64, 200), out_data[0]);
    try std.testing.expectEqual(@as(i64, 300), out_data[1]);
}

test "slice 4D tensor" {
    const allocator = std.testing.allocator;

    // Create a 2x3x2x2 tensor (24 elements)
    var input_data: [24]f32 = undefined;
    for (0..24) |i| {
        input_data[i] = @floatFromInt(i);
    }
    const input = TensorView.initContiguous(@ptrCast(&input_data), &.{ 2, 3, 2, 2 }, .f32);

    // Slice [0:1, 1:3, 0:2, 0:1] - 1x2x2x1 subtensor
    const starts = [_]usize{ 0, 1, 0, 0 };
    const ends = [_]usize{ 1, 3, 2, 1 };

    const out_data = try allocator.alloc(f32, 4);
    defer allocator.free(out_data);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{ 1, 2, 2, 1 }, .f32);

    slice(f32, out, input, &starts, &ends);

    // Verify shape and some elements
    try std.testing.expectEqual(@as(usize, 4), out.numel);

    // First element should be input[0,1,0,0] = 4
    try std.testing.expectEqual(@as(f32, 4), out_data[0]);

    // input[0,1,1,0] = 6
    try std.testing.expectEqual(@as(f32, 6), out_data[1]);

    // input[0,2,0,0] = 8
    try std.testing.expectEqual(@as(f32, 8), out_data[2]);

    // input[0,2,1,0] = 10
    try std.testing.expectEqual(@as(f32, 10), out_data[3]);
}

test "slice at boundaries" {
    const allocator = std.testing.allocator;

    // Create a 5x5 matrix
    var input_data: [25]f32 = undefined;
    for (0..25) |i| {
        input_data[i] = @floatFromInt(i);
    }
    const input = TensorView.initContiguous(@ptrCast(&input_data), &.{ 5, 5 }, .f32);

    // Slice last row [4:5, :]
    const starts = [_]usize{ 4, 0 };
    const ends = [_]usize{ 5, 5 };

    const out_data = try allocator.alloc(f32, 5);
    defer allocator.free(out_data);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{ 1, 5 }, .f32);

    slice(f32, out, input, &starts, &ends);

    // Should get [20, 21, 22, 23, 24]
    for (0..5) |i| {
        try std.testing.expectEqual(@as(f32, @floatFromInt(20 + i)), out_data[i]);
    }
}

// ============================================================================
// Tests for cat
// ============================================================================

test "cat 2D along dim 0 - contiguous fast path" {
    const allocator = std.testing.allocator;

    // Create two matrices: 2x3 and 3x3
    var a_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var b_data = [_]f32{ 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    const a = TensorView.initContiguous(@ptrCast(&a_data), &.{ 2, 3 }, .f32);
    const b = TensorView.initContiguous(@ptrCast(&b_data), &.{ 3, 3 }, .f32);

    // Output should be 5x3
    const out_data = try allocator.alloc(f32, 15);
    defer allocator.free(out_data);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{ 5, 3 }, .f32);

    const inputs = [_]TensorView{ a, b };
    cat(f32, out, &inputs, 0);

    // Verify concatenation
    try std.testing.expectEqual(@as(f32, 1), out_data[0]);
    try std.testing.expectEqual(@as(f32, 2), out_data[1]);
    try std.testing.expectEqual(@as(f32, 3), out_data[2]);
    try std.testing.expectEqual(@as(f32, 4), out_data[3]);
    try std.testing.expectEqual(@as(f32, 5), out_data[4]);
    try std.testing.expectEqual(@as(f32, 6), out_data[5]);
    try std.testing.expectEqual(@as(f32, 7), out_data[6]);
    try std.testing.expectEqual(@as(f32, 8), out_data[7]);
    try std.testing.expectEqual(@as(f32, 9), out_data[8]);
}

test "cat 2D along dim 1 - contiguous fast path" {
    const allocator = std.testing.allocator;

    // Create two matrices: 2x2 and 2x3
    var a_data = [_]f32{ 1, 2, 3, 4 };
    var b_data = [_]f32{ 5, 6, 7, 8, 9, 10 };
    const a = TensorView.initContiguous(@ptrCast(&a_data), &.{ 2, 2 }, .f32);
    const b = TensorView.initContiguous(@ptrCast(&b_data), &.{ 2, 3 }, .f32);

    // Output should be 2x5
    const out_data = try allocator.alloc(f32, 10);
    defer allocator.free(out_data);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{ 2, 5 }, .f32);

    const inputs = [_]TensorView{ a, b };
    cat(f32, out, &inputs, 1);

    // Expected: [[1,2,5,6,7], [3,4,8,9,10]]
    try std.testing.expectEqual(@as(f32, 1), out_data[0]);
    try std.testing.expectEqual(@as(f32, 2), out_data[1]);
    try std.testing.expectEqual(@as(f32, 5), out_data[2]);
    try std.testing.expectEqual(@as(f32, 6), out_data[3]);
    try std.testing.expectEqual(@as(f32, 7), out_data[4]);
    try std.testing.expectEqual(@as(f32, 3), out_data[5]);
    try std.testing.expectEqual(@as(f32, 4), out_data[6]);
    try std.testing.expectEqual(@as(f32, 8), out_data[7]);
    try std.testing.expectEqual(@as(f32, 9), out_data[8]);
    try std.testing.expectEqual(@as(f32, 10), out_data[9]);
}

test "cat with strided input - slow path" {
    const allocator = std.testing.allocator;

    // Create a 4x3 matrix and make a strided view
    var input_data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var strided = TensorView.initContiguous(@ptrCast(&input_data), &.{ 4, 3 }, .f32);
    strided.shape[0] = 2; // Only 2 rows
    strided.strides[0] = 6; // Skip a row
    strided.numel = 6;

    // Second input
    var b_data = [_]f32{ 13, 14, 15 };
    const b = TensorView.initContiguous(@ptrCast(&b_data), &.{ 1, 3 }, .f32);

    // Output should be 3x3
    const out_data = try allocator.alloc(f32, 9);
    defer allocator.free(out_data);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{ 3, 3 }, .f32);

    const inputs = [_]TensorView{ strided, b };
    cat(f32, out, &inputs, 0);

    // Strided input selects rows 0 and 2: [[1,2,3], [7,8,9]]
    try std.testing.expectEqual(@as(f32, 1), out_data[0]);
    try std.testing.expectEqual(@as(f32, 2), out_data[1]);
    try std.testing.expectEqual(@as(f32, 3), out_data[2]);
    try std.testing.expectEqual(@as(f32, 7), out_data[3]);
    try std.testing.expectEqual(@as(f32, 8), out_data[4]);
    try std.testing.expectEqual(@as(f32, 9), out_data[5]);
    try std.testing.expectEqual(@as(f32, 13), out_data[6]);
    try std.testing.expectEqual(@as(f32, 14), out_data[7]);
    try std.testing.expectEqual(@as(f32, 15), out_data[8]);
}

test "cat 3D along middle dimension" {
    const allocator = std.testing.allocator;

    // Create two 2x2x2 tensors
    var a_data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var b_data = [_]f32{ 9, 10, 11, 12, 13, 14, 15, 16 };
    const a = TensorView.initContiguous(@ptrCast(&a_data), &.{ 2, 2, 2 }, .f32);
    const b = TensorView.initContiguous(@ptrCast(&b_data), &.{ 2, 2, 2 }, .f32);

    // Output should be 2x4x2
    const out_data = try allocator.alloc(f32, 16);
    defer allocator.free(out_data);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{ 2, 4, 2 }, .f32);

    const inputs = [_]TensorView{ a, b };
    cat(f32, out, &inputs, 1);

    // Verify shape and some elements
    try std.testing.expectEqual(@as(usize, 16), out.numel);
    try std.testing.expectEqual(@as(f32, 1), out_data[0]);
    try std.testing.expectEqual(@as(f32, 2), out_data[1]);
    try std.testing.expectEqual(@as(f32, 3), out_data[2]);
    try std.testing.expectEqual(@as(f32, 4), out_data[3]);
}

test "catDispatch with f32" {
    const allocator = std.testing.allocator;

    var a_data = [_]f32{ 1, 2 };
    var b_data = [_]f32{ 3, 4 };
    const a = TensorView.initContiguous(@ptrCast(&a_data), &.{2}, .f32);
    const b = TensorView.initContiguous(@ptrCast(&b_data), &.{2}, .f32);

    const out_data = try allocator.alloc(f32, 4);
    defer allocator.free(out_data);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{4}, .f32);

    const inputs = [_]TensorView{ a, b };
    catDispatch(out, &inputs, 0);

    try std.testing.expectEqual(@as(f32, 1), out_data[0]);
    try std.testing.expectEqual(@as(f32, 2), out_data[1]);
    try std.testing.expectEqual(@as(f32, 3), out_data[2]);
    try std.testing.expectEqual(@as(f32, 4), out_data[3]);
}

test "cat single input - edge case" {
    const allocator = std.testing.allocator;

    var a_data = [_]f32{ 1, 2, 3, 4 };
    const a = TensorView.initContiguous(@ptrCast(&a_data), &.{ 2, 2 }, .f32);

    const out_data = try allocator.alloc(f32, 4);
    defer allocator.free(out_data);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{ 2, 2 }, .f32);

    const inputs = [_]TensorView{a};
    cat(f32, out, &inputs, 0);

    // Should be identical to input
    for (a_data, 0..) |val, i| {
        try std.testing.expectEqual(val, out_data[i]);
    }
}

// ============================================================================
// Tests for reshape
// ============================================================================

test "reshapeView contiguous - zero copy" {
    var data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const input = TensorView.initContiguous(@ptrCast(&data), &.{ 2, 3 }, .f32);

    const reshaped = reshapeView(input, &.{ 3, 2 });
    try std.testing.expect(reshaped != null);
    const out = reshaped.?;

    // Verify shape changed
    try std.testing.expectEqual(@as(usize, 2), out.ndim);
    try std.testing.expectEqual(@as(usize, 3), out.shape[0]);
    try std.testing.expectEqual(@as(usize, 2), out.shape[1]);
    try std.testing.expectEqual(@as(usize, 6), out.numel);

    // Verify data pointer unchanged (zero copy)
    try std.testing.expectEqual(input.data, out.data);
}

test "reshapeView to 1D vector" {
    var data = [_]f32{ 1, 2, 3, 4 };
    const input = TensorView.initContiguous(@ptrCast(&data), &.{ 2, 2 }, .f32);

    const reshaped = reshapeView(input, &.{4});
    try std.testing.expect(reshaped != null);
    const out = reshaped.?;

    try std.testing.expectEqual(@as(usize, 1), out.ndim);
    try std.testing.expectEqual(@as(usize, 4), out.shape[0]);
    try std.testing.expectEqual(input.data, out.data);
}

test "reshapeView to higher dimensions" {
    var data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const input = TensorView.initContiguous(@ptrCast(&data), &.{8}, .f32);

    const reshaped = reshapeView(input, &.{ 2, 2, 2 });
    try std.testing.expect(reshaped != null);
    const out = reshaped.?;

    try std.testing.expectEqual(@as(usize, 3), out.ndim);
    try std.testing.expectEqual(@as(usize, 2), out.shape[0]);
    try std.testing.expectEqual(@as(usize, 2), out.shape[1]);
    try std.testing.expectEqual(@as(usize, 2), out.shape[2]);
    try std.testing.expectEqual(input.data, out.data);
}

test "reshapeView non-contiguous returns null" {
    var data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var strided = TensorView.initContiguous(@ptrCast(&data), &.{ 3, 2 }, .f32);
    strided.strides[0] = 4; // Non-contiguous

    const reshaped = reshapeView(strided, &.{ 2, 3 });
    try std.testing.expect(reshaped == null);
}

test "reshapeCopy contiguous - fast path" {
    const allocator = std.testing.allocator;

    var input_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const input = TensorView.initContiguous(@ptrCast(&input_data), &.{ 2, 3 }, .f32);

    const out_data = try allocator.alloc(f32, 6);
    defer allocator.free(out_data);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{ 3, 2 }, .f32);

    reshapeCopy(f32, out, input);

    // Should have copied all data
    for (input_data, 0..) |val, i| {
        try std.testing.expectEqual(val, out_data[i]);
    }
}

test "reshapeCopy strided - element-wise copy" {
    const allocator = std.testing.allocator;

    var input_data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var strided = TensorView.initContiguous(@ptrCast(&input_data), &.{ 4, 2 }, .f32);
    strided.shape[0] = 2; // Only 2 rows
    strided.strides[0] = 4; // Skip rows
    strided.numel = 4;

    const out_data = try allocator.alloc(f32, 4);
    defer allocator.free(out_data);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{ 2, 2 }, .f32);

    reshapeCopy(f32, out, strided);

    // Should get rows 0 and 2: [1,2,5,6]
    try std.testing.expectEqual(@as(f32, 1), out_data[0]);
    try std.testing.expectEqual(@as(f32, 2), out_data[1]);
    try std.testing.expectEqual(@as(f32, 5), out_data[2]);
    try std.testing.expectEqual(@as(f32, 6), out_data[3]);
}

test "reshapeCopy different output shape" {
    const allocator = std.testing.allocator;

    var input_data = [_]f32{ 1, 2, 3, 4 };
    const input = TensorView.initContiguous(@ptrCast(&input_data), &.{ 2, 2 }, .f32);

    const out_data = try allocator.alloc(f32, 4);
    defer allocator.free(out_data);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{4}, .f32);

    reshapeCopy(f32, out, input);

    for (input_data, 0..) |val, i| {
        try std.testing.expectEqual(val, out_data[i]);
    }
}

// ============================================================================
// Tests for split
// ============================================================================

test "split 1D tensor into equal parts" {
    var data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const input = TensorView.initContiguous(@ptrCast(&data), &.{6}, .f32);

    var out_views: [3]TensorView = undefined;
    split(input, 0, 3, &out_views);

    // Verify first split
    try std.testing.expectEqual(@as(usize, 1), out_views[0].ndim);
    try std.testing.expectEqual(@as(usize, 2), out_views[0].shape[0]);
    try std.testing.expectEqual(@as(usize, 2), out_views[0].numel);

    // Verify data pointers are different (offset into original)
    const data0 = @as([*]const f32, @ptrCast(@alignCast(out_views[0].data)));
    const data1 = @as([*]const f32, @ptrCast(@alignCast(out_views[1].data)));
    const data2 = @as([*]const f32, @ptrCast(@alignCast(out_views[2].data)));

    try std.testing.expectEqual(@as(f32, 1), data0[0]);
    try std.testing.expectEqual(@as(f32, 2), data0[1]);
    try std.testing.expectEqual(@as(f32, 3), data1[0]);
    try std.testing.expectEqual(@as(f32, 4), data1[1]);
    try std.testing.expectEqual(@as(f32, 5), data2[0]);
    try std.testing.expectEqual(@as(f32, 6), data2[1]);
}

test "split 2D along last dimension" {
    var data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const input = TensorView.initContiguous(@ptrCast(&data), &.{ 2, 4 }, .f32);

    var out_views: [2]TensorView = undefined;
    split(input, 1, 2, &out_views);

    // Each split should be 2x2
    try std.testing.expectEqual(@as(usize, 2), out_views[0].shape[0]);
    try std.testing.expectEqual(@as(usize, 2), out_views[0].shape[1]);
    try std.testing.expectEqual(@as(usize, 4), out_views[0].numel);

    const data0 = @as([*]const f32, @ptrCast(@alignCast(out_views[0].data)));
    const data1 = @as([*]const f32, @ptrCast(@alignCast(out_views[1].data)));

    // First split: [[1,2], [5,6]]
    try std.testing.expectEqual(@as(f32, 1), data0[0]);
    try std.testing.expectEqual(@as(f32, 2), data0[1]);

    // Second split: [[3,4], [7,8]]
    try std.testing.expectEqual(@as(f32, 3), data1[0]);
    try std.testing.expectEqual(@as(f32, 4), data1[1]);
}

test "split 2D along first dimension" {
    var data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const input = TensorView.initContiguous(@ptrCast(&data), &.{ 2, 3 }, .f32);

    var out_views: [2]TensorView = undefined;
    split(input, 0, 2, &out_views);

    // Each split should be 1x3
    try std.testing.expectEqual(@as(usize, 1), out_views[0].shape[0]);
    try std.testing.expectEqual(@as(usize, 3), out_views[0].shape[1]);

    const data0 = @as([*]const f32, @ptrCast(@alignCast(out_views[0].data)));
    const data1 = @as([*]const f32, @ptrCast(@alignCast(out_views[1].data)));

    try std.testing.expectEqual(@as(f32, 1), data0[0]);
    try std.testing.expectEqual(@as(f32, 2), data0[1]);
    try std.testing.expectEqual(@as(f32, 3), data0[2]);
    try std.testing.expectEqual(@as(f32, 4), data1[0]);
    try std.testing.expectEqual(@as(f32, 5), data1[1]);
    try std.testing.expectEqual(@as(f32, 6), data1[2]);
}

test "split 3D tensor" {
    var data: [24]f32 = undefined;
    for (0..24) |i| {
        data[i] = @floatFromInt(i);
    }
    const input = TensorView.initContiguous(@ptrCast(&data), &.{ 2, 3, 4 }, .f32);

    var out_views: [3]TensorView = undefined;
    split(input, 1, 3, &out_views);

    // Each split should be 2x1x4
    try std.testing.expectEqual(@as(usize, 2), out_views[0].shape[0]);
    try std.testing.expectEqual(@as(usize, 1), out_views[0].shape[1]);
    try std.testing.expectEqual(@as(usize, 4), out_views[0].shape[2]);
    try std.testing.expectEqual(@as(usize, 8), out_views[0].numel);

    const data0 = @as([*]const f32, @ptrCast(@alignCast(out_views[0].data)));
    try std.testing.expectEqual(@as(f32, 0), data0[0]);
}

test "split single element per split" {
    var data = [_]f32{ 1, 2, 3, 4 };
    const input = TensorView.initContiguous(@ptrCast(&data), &.{4}, .f32);

    var out_views: [4]TensorView = undefined;
    split(input, 0, 4, &out_views);

    for (out_views, 0..) |view, i| {
        try std.testing.expectEqual(@as(usize, 1), view.shape[0]);
        try std.testing.expectEqual(@as(usize, 1), view.numel);
        const view_data = @as([*]const f32, @ptrCast(@alignCast(view.data)));
        try std.testing.expectEqual(@as(f32, @floatFromInt(i + 1)), view_data[0]);
    }
}

// ============================================================================
// Tests for repeatInterleave
// ============================================================================

test "repeatInterleave 1D - contiguous fast path" {
    const allocator = std.testing.allocator;

    var input_data = [_]f32{ 1, 2, 3 };
    const input = TensorView.initContiguous(@ptrCast(&input_data), &.{3}, .f32);

    const out_data = try allocator.alloc(f32, 9);
    defer allocator.free(out_data);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{9}, .f32);

    repeatInterleave(f32, out, input, 3, 0);

    // Each element repeated 3 times
    try std.testing.expectEqual(@as(f32, 1), out_data[0]);
    try std.testing.expectEqual(@as(f32, 1), out_data[1]);
    try std.testing.expectEqual(@as(f32, 1), out_data[2]);
    try std.testing.expectEqual(@as(f32, 2), out_data[3]);
    try std.testing.expectEqual(@as(f32, 2), out_data[4]);
    try std.testing.expectEqual(@as(f32, 2), out_data[5]);
    try std.testing.expectEqual(@as(f32, 3), out_data[6]);
    try std.testing.expectEqual(@as(f32, 3), out_data[7]);
    try std.testing.expectEqual(@as(f32, 3), out_data[8]);
}

test "repeatInterleave 2D along dim 0" {
    const allocator = std.testing.allocator;

    var input_data = [_]f32{ 1, 2, 3, 4 };
    const input = TensorView.initContiguous(@ptrCast(&input_data), &.{ 2, 2 }, .f32);

    const out_data = try allocator.alloc(f32, 8);
    defer allocator.free(out_data);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{ 4, 2 }, .f32);

    repeatInterleave(f32, out, input, 2, 0);

    // Expected: [[1,2], [1,2], [3,4], [3,4]]
    try std.testing.expectEqual(@as(f32, 1), out_data[0]);
    try std.testing.expectEqual(@as(f32, 2), out_data[1]);
    try std.testing.expectEqual(@as(f32, 1), out_data[2]);
    try std.testing.expectEqual(@as(f32, 2), out_data[3]);
    try std.testing.expectEqual(@as(f32, 3), out_data[4]);
    try std.testing.expectEqual(@as(f32, 4), out_data[5]);
    try std.testing.expectEqual(@as(f32, 3), out_data[6]);
    try std.testing.expectEqual(@as(f32, 4), out_data[7]);
}

test "repeatInterleave 2D along dim 1" {
    const allocator = std.testing.allocator;

    var input_data = [_]f32{ 1, 2, 3, 4 };
    const input = TensorView.initContiguous(@ptrCast(&input_data), &.{ 2, 2 }, .f32);

    const out_data = try allocator.alloc(f32, 8);
    defer allocator.free(out_data);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{ 2, 4 }, .f32);

    repeatInterleave(f32, out, input, 2, 1);

    // Expected: [[1,1,2,2], [3,3,4,4]]
    try std.testing.expectEqual(@as(f32, 1), out_data[0]);
    try std.testing.expectEqual(@as(f32, 1), out_data[1]);
    try std.testing.expectEqual(@as(f32, 2), out_data[2]);
    try std.testing.expectEqual(@as(f32, 2), out_data[3]);
    try std.testing.expectEqual(@as(f32, 3), out_data[4]);
    try std.testing.expectEqual(@as(f32, 3), out_data[5]);
    try std.testing.expectEqual(@as(f32, 4), out_data[6]);
    try std.testing.expectEqual(@as(f32, 4), out_data[7]);
}

test "repeatInterleave with strided input - slow path" {
    const allocator = std.testing.allocator;

    var input_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var strided = TensorView.initContiguous(@ptrCast(&input_data), &.{ 3, 2 }, .f32);
    strided.shape[0] = 2; // Only 2 rows
    strided.strides[0] = 4; // Skip rows
    strided.numel = 4;

    const out_data = try allocator.alloc(f32, 8);
    defer allocator.free(out_data);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{ 4, 2 }, .f32);

    repeatInterleave(f32, out, strided, 2, 0);

    // Strided input: [[1,2], [5,6]]
    // Expected: [[1,2], [1,2], [5,6], [5,6]]
    try std.testing.expectEqual(@as(f32, 1), out_data[0]);
    try std.testing.expectEqual(@as(f32, 2), out_data[1]);
    try std.testing.expectEqual(@as(f32, 1), out_data[2]);
    try std.testing.expectEqual(@as(f32, 2), out_data[3]);
    try std.testing.expectEqual(@as(f32, 5), out_data[4]);
    try std.testing.expectEqual(@as(f32, 6), out_data[5]);
    try std.testing.expectEqual(@as(f32, 5), out_data[6]);
    try std.testing.expectEqual(@as(f32, 6), out_data[7]);
}

test "repeatInterleave 3D middle dimension" {
    const allocator = std.testing.allocator;

    var input_data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const input = TensorView.initContiguous(@ptrCast(&input_data), &.{ 2, 2, 2 }, .f32);

    const out_data = try allocator.alloc(f32, 16);
    defer allocator.free(out_data);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{ 2, 4, 2 }, .f32);

    repeatInterleave(f32, out, input, 2, 1);

    // Verify shape and some elements
    try std.testing.expectEqual(@as(usize, 16), out.numel);
    try std.testing.expectEqual(@as(f32, 1), out_data[0]);
    try std.testing.expectEqual(@as(f32, 2), out_data[1]);
    try std.testing.expectEqual(@as(f32, 1), out_data[2]);
    try std.testing.expectEqual(@as(f32, 2), out_data[3]);
}

test "repeatInterleave repeats=1 is identity" {
    const allocator = std.testing.allocator;

    var input_data = [_]f32{ 1, 2, 3, 4 };
    const input = TensorView.initContiguous(@ptrCast(&input_data), &.{ 2, 2 }, .f32);

    const out_data = try allocator.alloc(f32, 4);
    defer allocator.free(out_data);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{ 2, 2 }, .f32);

    repeatInterleave(f32, out, input, 1, 0);

    for (input_data, 0..) |val, i| {
        try std.testing.expectEqual(val, out_data[i]);
    }
}

// ============================================================================
// Tests for topk
// ============================================================================

test "topk 1D - select top 3" {
    const allocator = std.testing.allocator;

    var input_data = [_]f32{ 3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0 };
    const input = TensorView.initContiguous(@ptrCast(&input_data), &.{7}, .f32);

    const val_data = try allocator.alloc(f32, 3);
    defer allocator.free(val_data);
    const values = TensorView.initContiguous(@ptrCast(val_data.ptr), &.{3}, .f32);

    const idx_data = try allocator.alloc(i64, 3);
    defer allocator.free(idx_data);
    const indices = TensorView.initContiguous(@ptrCast(idx_data.ptr), &.{3}, .i64);

    topk(f32, values, indices, input, 3);

    // Top 3 values should be 9.0, 5.0, 4.0
    try std.testing.expectEqual(@as(f32, 9.0), val_data[0]);
    try std.testing.expectEqual(@as(f32, 5.0), val_data[1]);
    try std.testing.expectEqual(@as(f32, 4.0), val_data[2]);

    // Indices should be 5, 4, 2
    try std.testing.expectEqual(@as(i64, 5), idx_data[0]);
    try std.testing.expectEqual(@as(i64, 4), idx_data[1]);
    try std.testing.expectEqual(@as(i64, 2), idx_data[2]);
}

test "topk 2D - select along last dimension" {
    const allocator = std.testing.allocator;

    // 2x5 matrix
    var input_data = [_]f32{ 5.0, 2.0, 8.0, 1.0, 3.0, 7.0, 4.0, 6.0, 9.0, 0.0 };
    const input = TensorView.initContiguous(@ptrCast(&input_data), &.{ 2, 5 }, .f32);

    const val_data = try allocator.alloc(f32, 4);
    defer allocator.free(val_data);
    const values = TensorView.initContiguous(@ptrCast(val_data.ptr), &.{ 2, 2 }, .f32);

    const idx_data = try allocator.alloc(i64, 4);
    defer allocator.free(idx_data);
    const indices = TensorView.initContiguous(@ptrCast(idx_data.ptr), &.{ 2, 2 }, .i64);

    topk(f32, values, indices, input, 2);

    // First row top 2: 8.0, 5.0
    try std.testing.expectEqual(@as(f32, 8.0), val_data[0]);
    try std.testing.expectEqual(@as(f32, 5.0), val_data[1]);

    // First row indices: 2, 0
    try std.testing.expectEqual(@as(i64, 2), idx_data[0]);
    try std.testing.expectEqual(@as(i64, 0), idx_data[1]);

    // Second row top 2: 9.0, 7.0
    try std.testing.expectEqual(@as(f32, 9.0), val_data[2]);
    try std.testing.expectEqual(@as(f32, 7.0), val_data[3]);

    // Second row indices: 3, 0
    try std.testing.expectEqual(@as(i64, 3), idx_data[2]);
    try std.testing.expectEqual(@as(i64, 0), idx_data[3]);
}

test "topk k=1 - select maximum" {
    const allocator = std.testing.allocator;

    var input_data = [_]f32{ 3.0, 7.0, 2.0, 9.0, 1.0 };
    const input = TensorView.initContiguous(@ptrCast(&input_data), &.{5}, .f32);

    const val_data = try allocator.alloc(f32, 1);
    defer allocator.free(val_data);
    const values = TensorView.initContiguous(@ptrCast(val_data.ptr), &.{1}, .f32);

    const idx_data = try allocator.alloc(i64, 1);
    defer allocator.free(idx_data);
    const indices = TensorView.initContiguous(@ptrCast(idx_data.ptr), &.{1}, .i64);

    topk(f32, values, indices, input, 1);

    try std.testing.expectEqual(@as(f32, 9.0), val_data[0]);
    try std.testing.expectEqual(@as(i64, 3), idx_data[0]);
}

test "topk with negative values" {
    const allocator = std.testing.allocator;

    var input_data = [_]f32{ -1.0, -5.0, -2.0, 3.0, 0.0 };
    const input = TensorView.initContiguous(@ptrCast(&input_data), &.{5}, .f32);

    const val_data = try allocator.alloc(f32, 2);
    defer allocator.free(val_data);
    const values = TensorView.initContiguous(@ptrCast(val_data.ptr), &.{2}, .f32);

    const idx_data = try allocator.alloc(i64, 2);
    defer allocator.free(idx_data);
    const indices = TensorView.initContiguous(@ptrCast(idx_data.ptr), &.{2}, .i64);

    topk(f32, values, indices, input, 2);

    // Top 2: 3.0, 0.0
    try std.testing.expectEqual(@as(f32, 3.0), val_data[0]);
    try std.testing.expectEqual(@as(f32, 0.0), val_data[1]);
    try std.testing.expectEqual(@as(i64, 3), idx_data[0]);
    try std.testing.expectEqual(@as(i64, 4), idx_data[1]);
}

test "topk 3D tensor" {
    const allocator = std.testing.allocator;

    // 2x2x3 tensor
    var input_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 };
    const input = TensorView.initContiguous(@ptrCast(&input_data), &.{ 2, 2, 3 }, .f32);

    const val_data = try allocator.alloc(f32, 8);
    defer allocator.free(val_data);
    const values = TensorView.initContiguous(@ptrCast(val_data.ptr), &.{ 2, 2, 2 }, .f32);

    const idx_data = try allocator.alloc(i64, 8);
    defer allocator.free(idx_data);
    const indices = TensorView.initContiguous(@ptrCast(idx_data.ptr), &.{ 2, 2, 2 }, .i64);

    topk(f32, values, indices, input, 2);

    // First row of first slice: top 2 from [1,2,3]
    try std.testing.expectEqual(@as(f32, 3.0), val_data[0]);
    try std.testing.expectEqual(@as(f32, 2.0), val_data[1]);
    try std.testing.expectEqual(@as(i64, 2), idx_data[0]);
    try std.testing.expectEqual(@as(i64, 1), idx_data[1]);

    // Second row of second slice: top 2 from [10,11,12]
    try std.testing.expectEqual(@as(f32, 12.0), val_data[6]);
    try std.testing.expectEqual(@as(f32, 11.0), val_data[7]);
    try std.testing.expectEqual(@as(i64, 2), idx_data[6]);
    try std.testing.expectEqual(@as(i64, 1), idx_data[7]);
}

test "topk with duplicate values" {
    const allocator = std.testing.allocator;

    var input_data = [_]f32{ 5.0, 3.0, 5.0, 1.0, 5.0 };
    const input = TensorView.initContiguous(@ptrCast(&input_data), &.{5}, .f32);

    const val_data = try allocator.alloc(f32, 3);
    defer allocator.free(val_data);
    const values = TensorView.initContiguous(@ptrCast(val_data.ptr), &.{3}, .f32);

    const idx_data = try allocator.alloc(i64, 3);
    defer allocator.free(idx_data);
    const indices = TensorView.initContiguous(@ptrCast(idx_data.ptr), &.{3}, .i64);

    topk(f32, values, indices, input, 3);

    // All top 3 should be 5.0
    try std.testing.expectEqual(@as(f32, 5.0), val_data[0]);
    try std.testing.expectEqual(@as(f32, 5.0), val_data[1]);
    try std.testing.expectEqual(@as(f32, 5.0), val_data[2]);

    // Indices should be 0, 2, 4 (first occurrences)
    try std.testing.expectEqual(@as(i64, 0), idx_data[0]);
    try std.testing.expectEqual(@as(i64, 2), idx_data[1]);
    try std.testing.expectEqual(@as(i64, 4), idx_data[2]);
}

// =============================================================================
// Validation Helper Tests
// =============================================================================

test "normalizeDim positive indices" {
    // Valid positive indices
    try std.testing.expectEqual(@as(?usize, 0), normalizeDim(0, 3));
    try std.testing.expectEqual(@as(?usize, 1), normalizeDim(1, 3));
    try std.testing.expectEqual(@as(?usize, 2), normalizeDim(2, 3));

    // Out of bounds positive
    try std.testing.expectEqual(@as(?usize, null), normalizeDim(3, 3));
    try std.testing.expectEqual(@as(?usize, null), normalizeDim(5, 3));
}

test "normalizeDim negative indices" {
    // Valid negative indices (Python-style)
    try std.testing.expectEqual(@as(?usize, 2), normalizeDim(-1, 3)); // last dim
    try std.testing.expectEqual(@as(?usize, 1), normalizeDim(-2, 3)); // second to last
    try std.testing.expectEqual(@as(?usize, 0), normalizeDim(-3, 3)); // first dim

    // Out of bounds negative
    try std.testing.expectEqual(@as(?usize, null), normalizeDim(-4, 3));
    try std.testing.expectEqual(@as(?usize, null), normalizeDim(-10, 3));
}

test "normalizeDim edge cases" {
    // Single dimension tensor
    try std.testing.expectEqual(@as(?usize, 0), normalizeDim(0, 1));
    try std.testing.expectEqual(@as(?usize, 0), normalizeDim(-1, 1));
    try std.testing.expectEqual(@as(?usize, null), normalizeDim(1, 1));
    try std.testing.expectEqual(@as(?usize, null), normalizeDim(-2, 1));
}

test "validateSplitParams valid cases" {
    // Valid split: 6 elements / 2 = 3 each
    try std.testing.expectEqual(@as(?SplitValidationError, null), validateSplitParams(3, 6, 0, 2));
    // Valid split: 12 elements / 3 = 4 each
    try std.testing.expectEqual(@as(?SplitValidationError, null), validateSplitParams(2, 12, 1, 3));
    // Edge: split into 1 (no split)
    try std.testing.expectEqual(@as(?SplitValidationError, null), validateSplitParams(1, 5, 0, 1));
}

test "validateSplitParams error cases" {
    // dim out of bounds
    try std.testing.expectEqual(@as(?SplitValidationError, .dim_out_of_bounds), validateSplitParams(2, 6, 2, 2));
    try std.testing.expectEqual(@as(?SplitValidationError, .dim_out_of_bounds), validateSplitParams(2, 6, 5, 2));

    // num_splits is zero
    try std.testing.expectEqual(@as(?SplitValidationError, .num_splits_zero), validateSplitParams(2, 6, 0, 0));

    // num_splits too large (>32)
    try std.testing.expectEqual(@as(?SplitValidationError, .num_splits_too_large), validateSplitParams(2, 64, 0, 33));

    // not divisible
    try std.testing.expectEqual(@as(?SplitValidationError, .not_divisible), validateSplitParams(2, 7, 0, 2));
    try std.testing.expectEqual(@as(?SplitValidationError, .not_divisible), validateSplitParams(2, 10, 0, 3));
}

test "validateExpandParams valid cases" {
    // Expand 2D to 3D
    try std.testing.expectEqual(@as(?ExpandValidationError, null), validateExpandParams(2, &.{ 1, 3, 4 }, 8));
    // Same rank
    try std.testing.expectEqual(@as(?ExpandValidationError, null), validateExpandParams(3, &.{ 2, 3, 4 }, 8));
    // Expand 1D to 4D
    try std.testing.expectEqual(@as(?ExpandValidationError, null), validateExpandParams(1, &.{ 2, 3, 4, 5 }, 8));
}

test "validateExpandParams error cases" {
    // new ndim less than input ndim
    try std.testing.expectEqual(@as(?ExpandValidationError, .new_ndim_less_than_input), validateExpandParams(3, &.{ 4, 5 }, 8));

    // new ndim exceeds max
    try std.testing.expectEqual(@as(?ExpandValidationError, .new_ndim_exceeds_max), validateExpandParams(2, &.{ 1, 2, 3, 4, 5 }, 4));

    // negative dimension in new shape
    try std.testing.expectEqual(@as(?ExpandValidationError, .negative_dimension), validateExpandParams(2, &.{ 2, -1, 4 }, 8));
}

test "canBroadcast valid cases" {
    // Same shape
    try std.testing.expect(canBroadcast(&.{ 2, 3, 4 }, 3, &.{ 2, 3, 4 }, 3));

    // Broadcast from 1 in input
    try std.testing.expect(canBroadcast(&.{ 1, 3, 1 }, 3, &.{ 2, 3, 4 }, 3));

    // Broadcast with prepended dimensions
    try std.testing.expect(canBroadcast(&.{ 3, 4 }, 2, &.{ 2, 3, 4 }, 3));

    // Single element broadcast
    try std.testing.expect(canBroadcast(&.{1}, 1, &.{ 2, 3, 4 }, 3));
}

test "canBroadcast invalid cases" {
    // Dimension mismatch (not 1 and not equal)
    try std.testing.expect(!canBroadcast(&.{ 2, 3 }, 2, &.{ 3, 3 }, 2));

    // Incompatible shapes
    try std.testing.expect(!canBroadcast(&.{ 2, 4 }, 2, &.{ 2, 3 }, 2));

    // Can't broadcast 5 to 3
    try std.testing.expect(!canBroadcast(&.{ 5, 3 }, 2, &.{ 2, 3, 3 }, 3));
}
