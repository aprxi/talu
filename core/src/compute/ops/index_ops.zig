//! Index Operations
//!
//! Functions for index manipulation, nonzero finding, and shape inference.
//! Used by capi modules to avoid inline loop logic.

const std = @import("std");
const tensor_mod = @import("../../tensor.zig");
pub const Tensor = tensor_mod.Tensor;
pub const DType = tensor_mod.DType;

/// Result of nonzero operation.
/// Contains indices array and count.
pub const NonzeroResult = struct {
    /// Indices data: [num_nonzero, ndim]
    indices: []i64,
    /// Number of nonzero elements found
    count: usize,
    /// Number of dimensions in the input tensor
    ndim: usize,

    pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
        if (self.indices.len > 0) {
            allocator.free(self.indices);
        }
    }

    pub fn shape(self: *const @This()) [2]i64 {
        return .{ @intCast(self.count), @intCast(self.ndim) };
    }
};

/// Count nonzero elements in a tensor.
pub fn countNonzero(comptime T: type, data: [*]const T, numel: usize) usize {
    var count: usize = 0;
    for (0..numel) |i| {
        if (data[i] != 0) count += 1;
    }
    return count;
}

/// Convert flat index to multi-dimensional coordinates.
/// Stores result in coords array (row-major order).
pub fn flatToCoords(flat_idx: usize, shape: []const i64, coords: []i64) void {
    var remaining = flat_idx;
    var dim_idx: usize = shape.len;
    while (dim_idx > 0) {
        dim_idx -= 1;
        const dim_size: usize = @intCast(shape[dim_idx]);
        coords[dim_idx] = @intCast(remaining % dim_size);
        remaining /= dim_size;
    }
}

/// Find nonzero elements and return their indices.
///
/// Args:
///   allocator: Memory allocator
///   data: Tensor data pointer
///   numel: Number of elements
///   shape: Tensor shape
///   ndim: Number of dimensions
///
/// Returns NonzeroResult on success, error on failure.
pub fn findNonzeroI64(
    allocator: std.mem.Allocator,
    data: [*]const i64,
    numel: usize,
    tensor_shape: []const i64,
) !NonzeroResult {
    const ndim = tensor_shape.len;

    // Count nonzeros
    const count = countNonzero(i64, data, numel);

    if (count == 0) {
        return .{
            .indices = &[_]i64{},
            .count = 0,
            .ndim = ndim,
        };
    }

    // Allocate indices array
    const indices = try allocator.alloc(i64, count * ndim);
    errdefer allocator.free(indices);

    // Fill indices
    var out_idx: usize = 0;
    for (0..numel) |flat_idx| {
        if (data[flat_idx] != 0) {
            flatToCoords(flat_idx, tensor_shape, indices[out_idx * ndim ..][0..ndim]);
            out_idx += 1;
        }
    }

    return .{
        .indices = indices,
        .count = count,
        .ndim = ndim,
    };
}

/// Find nonzero elements for f32 data.
pub fn findNonzeroF32(
    allocator: std.mem.Allocator,
    data: [*]const f32,
    numel: usize,
    tensor_shape: []const i64,
) !NonzeroResult {
    const ndim = tensor_shape.len;

    // Count nonzeros
    const count = countNonzero(f32, data, numel);

    if (count == 0) {
        return .{
            .indices = &[_]i64{},
            .count = 0,
            .ndim = ndim,
        };
    }

    // Allocate indices array
    const indices = try allocator.alloc(i64, count * ndim);
    errdefer allocator.free(indices);

    // Fill indices
    var out_idx: usize = 0;
    for (0..numel) |flat_idx| {
        if (data[flat_idx] != 0) {
            flatToCoords(flat_idx, tensor_shape, indices[out_idx * ndim ..][0..ndim]);
            out_idx += 1;
        }
    }

    return .{
        .indices = indices,
        .count = count,
        .ndim = ndim,
    };
}

/// Error type for nonzero tensor operation.
pub const NonzeroTensorError = error{
    UnsupportedDType,
    OutOfMemory,
};

/// High-level nonzero operation that takes a Tensor and returns a Tensor.
/// Returns 2D tensor [num_nonzero, ndim] with indices of non-zero elements.
/// Supports i64 and f32 dtypes.
pub fn nonzeroTensor(
    allocator: std.mem.Allocator,
    input: *const Tensor,
) NonzeroTensorError!*Tensor {
    const input_ndim: usize = @intCast(input.n_dims);
    const shape_slice = input.shape[0..input_ndim];

    // Find nonzero indices based on dtype
    var result = switch (input.simpleDType()) {
        .i64 => findNonzeroI64(
            allocator,
            @ptrCast(@alignCast(input.data_ptr)),
            input.numel,
            shape_slice,
        ) catch return error.OutOfMemory,
        .f32 => findNonzeroF32(
            allocator,
            @ptrCast(@alignCast(input.data_ptr)),
            input.numel,
            shape_slice,
        ) catch return error.OutOfMemory,
        else => return error.UnsupportedDType,
    };

    // Create output tensor
    const out_shape = result.shape();
    const result_tensor = Tensor.init(allocator, &out_shape, .i64, input.device) catch {
        result.deinit(allocator);
        return error.OutOfMemory;
    };

    // Copy indices to tensor (only if non-empty)
    if (result.count > 0) {
        const out_data: [*]i64 = @ptrCast(@alignCast(result_tensor.data_ptr));
        @memcpy(out_data[0 .. result.count * result.ndim], result.indices);
        result.deinit(allocator);
    }

    return result_tensor;
}

/// Infer shape dimension with -1 placeholder.
/// Returns the computed dimension size, or error if invalid.
pub fn inferShapeDimension(
    current_numel: usize,
    target_shape: []const i64,
) !usize {
    var infer_dim: ?usize = null;
    var known_product: usize = 1;

    for (target_shape, 0..) |dim, i| {
        if (dim == -1) {
            if (infer_dim != null) {
                return error.MultipleInferredDimensions;
            }
            infer_dim = i;
        } else if (dim <= 0) {
            return error.InvalidDimension;
        } else {
            known_product *= @intCast(dim);
        }
    }

    if (infer_dim) |_| {
        if (current_numel % known_product != 0) {
            return error.ShapeMismatch;
        }
        return current_numel / known_product;
    } else {
        // No inference needed, validate total matches
        if (known_product != current_numel) {
            return error.ShapeMismatch;
        }
        return 0; // No dimension to infer
    }
}

/// Resolve shape with -1 placeholder into concrete shape.
/// Writes result to out_shape buffer.
pub fn resolveShape(
    current_numel: usize,
    target_shape: []const i64,
    out_shape: []i64,
) !void {
    const inferred = try inferShapeDimension(current_numel, target_shape);

    for (target_shape, 0..) |dim, i| {
        if (dim == -1) {
            out_shape[i] = @intCast(inferred);
        } else {
            out_shape[i] = dim;
        }
    }
}

/// Index add for f32: scatter-add src values to target at indices along dim.
/// target_shape and target_data are the destination tensor.
/// src_data has shape where dim is replaced by num_indices.
pub fn indexAddF32(
    target_data: [*]f32,
    target_shape: []const i64,
    indices: [*]const i64,
    num_indices: usize,
    src_data: [*]const f32,
    dim: usize,
) void {
    const rank = target_shape.len;

    // Calculate slice size (elements after dim)
    var slice_size: usize = 1;
    for (dim + 1..rank) |d| {
        slice_size *= @intCast(target_shape[d]);
    }

    // Calculate outer size (elements before dim)
    var outer_size: usize = 1;
    for (0..dim) |d| {
        outer_size *= @intCast(target_shape[d]);
    }

    const target_dim_size: usize = @intCast(target_shape[dim]);

    // Scatter-add
    for (0..num_indices) |idx_i| {
        const target_idx: usize = @intCast(indices[idx_i]);
        for (0..outer_size) |outer| {
            const out_base = outer * target_dim_size * slice_size + target_idx * slice_size;
            const src_base = outer * num_indices * slice_size + idx_i * slice_size;
            for (0..slice_size) |s| {
                target_data[out_base + s] += src_data[src_base + s];
            }
        }
    }
}

/// Conditional selection (where) for f32 with i64 condition.
pub fn whereF32CondI64(
    cond: [*]const i64,
    x: [*]const f32,
    y: [*]const f32,
    out: [*]f32,
    numel: usize,
) void {
    for (0..numel) |i| {
        out[i] = if (cond[i] != 0) x[i] else y[i];
    }
}

/// Conditional selection (where) for f32 with f32 condition.
pub fn whereF32CondF32(
    cond: [*]const f32,
    x: [*]const f32,
    y: [*]const f32,
    out: [*]f32,
    numel: usize,
) void {
    for (0..numel) |i| {
        out[i] = if (cond[i] != 0) x[i] else y[i];
    }
}

/// One-hot encoding error type.
pub const OneHotError = error{
    NegativeIndex,
    IndexOutOfBounds,
};

/// Apply one-hot encoding from i64 indices.
/// Output buffer must be pre-zeroed and sized [index_count * num_classes].
pub fn oneHotI64(
    indices: [*]const i64,
    index_count: usize,
    num_classes: usize,
    out: [*]f32,
) OneHotError!void {
    for (0..index_count) |i| {
        const idx_val = indices[i];
        if (idx_val < 0) return error.NegativeIndex;
        const class: usize = @intCast(idx_val);
        if (class >= num_classes) return error.IndexOutOfBounds;
        out[i * num_classes + class] = 1.0;
    }
}

/// Apply one-hot encoding from i32 indices.
pub fn oneHotI32(
    indices: [*]const i32,
    index_count: usize,
    num_classes: usize,
    out: [*]f32,
) OneHotError!void {
    for (0..index_count) |i| {
        const idx_val = indices[i];
        if (idx_val < 0) return error.NegativeIndex;
        const class: usize = @intCast(idx_val);
        if (class >= num_classes) return error.IndexOutOfBounds;
        out[i * num_classes + class] = 1.0;
    }
}

// =============================================================================
// Tests
// =============================================================================

test "oneHotI64 basic" {
    var out: [12]f32 = undefined;
    @memset(&out, 0);

    const indices = [_]i64{ 0, 2, 1 };
    try oneHotI64(&indices, 3, 4, &out);

    // Row 0: [1,0,0,0]
    try std.testing.expectEqual(@as(f32, 1.0), out[0]);
    try std.testing.expectEqual(@as(f32, 0.0), out[1]);

    // Row 1: [0,0,1,0]
    try std.testing.expectEqual(@as(f32, 0.0), out[4]);
    try std.testing.expectEqual(@as(f32, 1.0), out[6]);

    // Row 2: [0,1,0,0]
    try std.testing.expectEqual(@as(f32, 1.0), out[9]);
}

test "oneHotI64 negative index" {
    var out: [4]f32 = undefined;
    @memset(&out, 0);
    const indices = [_]i64{-1};
    try std.testing.expectError(error.NegativeIndex, oneHotI64(&indices, 1, 4, &out));
}

test "oneHotI64 out of bounds" {
    var out: [4]f32 = undefined;
    @memset(&out, 0);
    const indices = [_]i64{5};
    try std.testing.expectError(error.IndexOutOfBounds, oneHotI64(&indices, 1, 4, &out));
}

test "oneHotI32 basic" {
    var out: [6]f32 = undefined;
    @memset(&out, 0);

    const indices = [_]i32{ 1, 0 };
    try oneHotI32(&indices, 2, 3, &out);

    // Row 0: [0,1,0]
    try std.testing.expectEqual(@as(f32, 0.0), out[0]);
    try std.testing.expectEqual(@as(f32, 1.0), out[1]);

    // Row 1: [1,0,0]
    try std.testing.expectEqual(@as(f32, 1.0), out[3]);
}

test "countNonzero i64" {
    const data = [_]i64{ 1, 0, 2, 0, 3 };
    const count = countNonzero(i64, &data, 5);
    try std.testing.expectEqual(@as(usize, 3), count);
}

test "countNonzero f32" {
    const data = [_]f32{ 1.0, 0.0, 0.0, 2.5, 0.0 };
    const count = countNonzero(f32, &data, 5);
    try std.testing.expectEqual(@as(usize, 2), count);
}

test "countNonzero all zeros" {
    const data = [_]i64{ 0, 0, 0 };
    const count = countNonzero(i64, &data, 3);
    try std.testing.expectEqual(@as(usize, 0), count);
}

test "flatToCoords 1D" {
    const shape = [_]i64{5};
    var coords: [1]i64 = undefined;
    flatToCoords(3, &shape, &coords);
    try std.testing.expectEqual(@as(i64, 3), coords[0]);
}

test "flatToCoords 2D" {
    const shape = [_]i64{ 3, 4 };
    var coords: [2]i64 = undefined;

    flatToCoords(0, &shape, &coords);
    try std.testing.expectEqual(@as(i64, 0), coords[0]);
    try std.testing.expectEqual(@as(i64, 0), coords[1]);

    flatToCoords(5, &shape, &coords); // row 1, col 1
    try std.testing.expectEqual(@as(i64, 1), coords[0]);
    try std.testing.expectEqual(@as(i64, 1), coords[1]);

    flatToCoords(11, &shape, &coords); // row 2, col 3
    try std.testing.expectEqual(@as(i64, 2), coords[0]);
    try std.testing.expectEqual(@as(i64, 3), coords[1]);
}

test "flatToCoords 3D" {
    const shape = [_]i64{ 2, 3, 4 };
    var coords: [3]i64 = undefined;

    flatToCoords(13, &shape, &coords); // 13 = 1*12 + 0*4 + 1
    try std.testing.expectEqual(@as(i64, 1), coords[0]);
    try std.testing.expectEqual(@as(i64, 0), coords[1]);
    try std.testing.expectEqual(@as(i64, 1), coords[2]);
}

test "findNonzeroI64 basic" {
    const allocator = std.testing.allocator;
    const data = [_]i64{ 1, 0, 2, 0, 3 };
    const shape = [_]i64{5};

    var result = try findNonzeroI64(allocator, &data, 5, &shape);
    defer result.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 3), result.count);
    try std.testing.expectEqual(@as(usize, 1), result.ndim);

    // Indices should be 0, 2, 4
    try std.testing.expectEqual(@as(i64, 0), result.indices[0]);
    try std.testing.expectEqual(@as(i64, 2), result.indices[1]);
    try std.testing.expectEqual(@as(i64, 4), result.indices[2]);
}

test "findNonzeroI64 2D" {
    const allocator = std.testing.allocator;
    // 2x3 matrix: [[1,0,0], [0,2,3]]
    const data = [_]i64{ 1, 0, 0, 0, 2, 3 };
    const shape = [_]i64{ 2, 3 };

    var result = try findNonzeroI64(allocator, &data, 6, &shape);
    defer result.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 3), result.count);
    try std.testing.expectEqual(@as(usize, 2), result.ndim);

    // First nonzero at (0, 0)
    try std.testing.expectEqual(@as(i64, 0), result.indices[0]);
    try std.testing.expectEqual(@as(i64, 0), result.indices[1]);

    // Second nonzero at (1, 1)
    try std.testing.expectEqual(@as(i64, 1), result.indices[2]);
    try std.testing.expectEqual(@as(i64, 1), result.indices[3]);

    // Third nonzero at (1, 2)
    try std.testing.expectEqual(@as(i64, 1), result.indices[4]);
    try std.testing.expectEqual(@as(i64, 2), result.indices[5]);
}

test "findNonzeroI64 empty result" {
    const allocator = std.testing.allocator;
    const data = [_]i64{ 0, 0, 0 };
    const shape = [_]i64{3};

    var result = try findNonzeroI64(allocator, &data, 3, &shape);
    defer result.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), result.count);
}

test "findNonzeroF32 basic" {
    const allocator = std.testing.allocator;
    const data = [_]f32{ 1.0, 0.0, 2.5 };
    const shape = [_]i64{3};

    var result = try findNonzeroF32(allocator, &data, 3, &shape);
    defer result.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 2), result.count);
    try std.testing.expectEqual(@as(i64, 0), result.indices[0]);
    try std.testing.expectEqual(@as(i64, 2), result.indices[1]);
}

test "inferShapeDimension simple" {
    const shape = [_]i64{ 2, -1, 4 };
    const inferred = try inferShapeDimension(24, &shape);
    try std.testing.expectEqual(@as(usize, 3), inferred);
}

test "inferShapeDimension no inference needed" {
    const shape = [_]i64{ 2, 3, 4 };
    const inferred = try inferShapeDimension(24, &shape);
    try std.testing.expectEqual(@as(usize, 0), inferred);
}

test "inferShapeDimension multiple -1 fails" {
    const shape = [_]i64{ -1, -1, 4 };
    try std.testing.expectError(error.MultipleInferredDimensions, inferShapeDimension(24, &shape));
}

test "inferShapeDimension mismatch fails" {
    const shape = [_]i64{ 2, -1, 5 };
    try std.testing.expectError(error.ShapeMismatch, inferShapeDimension(24, &shape));
}

test "resolveShape with inference" {
    const shape = [_]i64{ 2, -1, 4 };
    var out: [3]i64 = undefined;
    try resolveShape(24, &shape, &out);

    try std.testing.expectEqual(@as(i64, 2), out[0]);
    try std.testing.expectEqual(@as(i64, 3), out[1]);
    try std.testing.expectEqual(@as(i64, 4), out[2]);
}

test "resolveShape no inference" {
    const shape = [_]i64{ 4, 6 };
    var out: [2]i64 = undefined;
    try resolveShape(24, &shape, &out);

    try std.testing.expectEqual(@as(i64, 4), out[0]);
    try std.testing.expectEqual(@as(i64, 6), out[1]);
}
