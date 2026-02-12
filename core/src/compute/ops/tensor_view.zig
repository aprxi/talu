//! Unified tensor view interface for stride-aware operations.
//!
//! This module provides a comptime-checked interface that works with:
//! - `Tensor` (from tensor.zig) - contiguous or strided tensors
//!
//! All compute/ops implementations use this interface, allowing capi/ops/* modules
//! to be a thin FFI bridge that delegates here.

const std = @import("std");
const tensor = @import("../../tensor.zig");

/// Maximum dimensions supported (matches DLPack)
pub const MAX_NDIM: usize = 8;

/// Simplified dtype enum for ops
pub const DType = enum {
    f32,
    f16,
    bf16,
    i32,
    i64,

    pub fn elementSize(self: DType) usize {
        return switch (self) {
            .f32 => 4,
            .f16, .bf16 => 2,
            .i32 => 4,
            .i64 => 8,
        };
    }
};

/// Strided tensor view for compute operations.
/// This is a lightweight struct that can be constructed from any tensor type.
pub const TensorView = struct {
    /// Raw data pointer
    data: [*]u8,
    /// Shape array (first ndim entries valid)
    shape: [MAX_NDIM]usize,
    /// Strides in elements (not bytes)
    strides: [MAX_NDIM]usize,
    /// Number of dimensions
    ndim: usize,
    /// Data type
    dtype: DType,
    /// Total number of elements
    numel: usize,

    const Self = @This();

    /// Create a view from shape and contiguous data (computes row-major strides)
    pub fn initContiguous(data: [*]u8, shape_in: []const usize, dtype: DType) Self {
        var view: Self = undefined;
        view.data = data;
        view.ndim = shape_in.len;
        view.dtype = dtype;

        // Copy shape and compute numel
        var numel: usize = 1;
        for (shape_in, 0..) |dim, dim_idx| {
            view.shape[dim_idx] = dim;
            numel *= dim;
        }
        view.numel = numel;

        // Compute row-major strides
        if (shape_in.len > 0) {
            var dim_idx: usize = shape_in.len;
            var stride: usize = 1;
            while (dim_idx > 0) {
                dim_idx -= 1;
                view.strides[dim_idx] = stride;
                stride *= shape_in[dim_idx];
            }
        }

        // Zero unused slots
        for (shape_in.len..MAX_NDIM) |dim_idx| {
            view.shape[dim_idx] = 0;
            view.strides[dim_idx] = 0;
        }

        return view;
    }

    /// Create a view from explicit shape and strides
    pub fn initStrided(data: [*]u8, shape_in: []const usize, strides_in: []const usize, dtype: DType) Self {
        std.debug.assert(shape_in.len == strides_in.len);

        var view: Self = undefined;
        view.data = data;
        view.ndim = shape_in.len;
        view.dtype = dtype;

        var numel: usize = 1;
        for (shape_in, strides_in, 0..) |dim, stride, dim_idx| {
            view.shape[dim_idx] = dim;
            view.strides[dim_idx] = stride;
            numel *= dim;
        }
        view.numel = numel;

        for (shape_in.len..MAX_NDIM) |dim_idx| {
            view.shape[dim_idx] = 0;
            view.strides[dim_idx] = 0;
        }

        return view;
    }

    /// Check if tensor is contiguous (row-major)
    pub fn isContiguous(self: Self) bool {
        if (self.ndim == 0) return true;

        var expected_stride: usize = 1;
        var dim_idx: usize = self.ndim;
        while (dim_idx > 0) {
            dim_idx -= 1;
            if (self.strides[dim_idx] != expected_stride) return false;
            expected_stride *= self.shape[dim_idx];
        }
        return true;
    }

    /// Get typed data slice (only valid for contiguous tensors)
    pub fn asSlice(self: Self, comptime T: type) []T {
        std.debug.assert(self.isContiguous());
        const aligned: [*]align(@alignOf(T)) u8 = @alignCast(self.data);
        return @as([*]T, @ptrCast(aligned))[0..self.numel];
    }

    /// Get element at logical coordinates using strides
    pub inline fn getElement(self: Self, comptime T: type, coords: []const usize) T {
        std.debug.assert(coords.len == self.ndim);
        var offset: usize = 0;
        for (coords, 0..) |c, dim_idx| {
            offset += c * self.strides[dim_idx];
        }
        const data_ptr = @as([*]const T, @ptrCast(@alignCast(self.data)));
        return data_ptr[offset];
    }

    /// Set element at logical coordinates using strides
    pub inline fn setElement(self: Self, comptime T: type, coords: []const usize, value: T) void {
        std.debug.assert(coords.len == self.ndim);
        var offset: usize = 0;
        for (coords, 0..) |c, dim_idx| {
            offset += c * self.strides[dim_idx];
        }
        const data_ptr = @as([*]T, @ptrCast(@alignCast(self.data)));
        data_ptr[offset] = value;
    }

    /// Convert linear index to coordinates (row-major logical order)
    pub fn indexToCoords(self: Self, index: usize, coords: *[MAX_NDIM]usize) void {
        var remaining = index;
        var divisor: usize = self.numel;
        for (0..self.ndim) |dim_idx| {
            divisor /= self.shape[dim_idx];
            coords[dim_idx] = remaining / divisor;
            remaining %= divisor;
        }
    }

    /// Get memory offset for coordinates
    pub fn coordsToOffset(self: Self, coords: []const usize) usize {
        var offset: usize = 0;
        for (coords, 0..) |c, dim_idx| {
            offset += c * self.strides[dim_idx];
        }
        return offset;
    }
};

fn initView(
    view: *TensorView,
    data: [*]u8,
    ndim: usize,
    dtype: DType,
    numel: usize,
    shape: []const i64,
    strides: []const i64,
) void {
    view.data = data;
    view.ndim = ndim;
    view.dtype = dtype;
    view.numel = numel;

    for (0..ndim) |dim_idx| {
        view.shape[dim_idx] = @intCast(shape[dim_idx]);
        view.strides[dim_idx] = @intCast(strides[dim_idx]);
    }
    for (ndim..MAX_NDIM) |dim_idx| {
        view.shape[dim_idx] = 0;
        view.strides[dim_idx] = 0;
    }
}

/// Convert Tensor to TensorView
pub fn fromTensor(comptime T: type, src: *const T) TensorView {
    var view: TensorView = undefined; // initialized by initView() below
    const dtype: DType = switch (@intFromEnum(src.dtype)) {
        0 => .f32,
        4 => .f16,
        5 => .bf16,
        2 => .i32,
        3 => .i64,
        else => .f32,
    };
    const ndim: usize = @intCast(src.n_dims);
    initView(&view, src.data_ptr orelse unreachable, ndim, dtype, src.numel, src.shape[0..ndim], src.strides[0..ndim]);

    return view;
}

/// Convert Tensor to TensorView with simple dtype mapping.
/// Returns null if dtype is unsupported for TensorView ops.
pub fn fromSimpleTensor(t: *const tensor.Tensor) ?TensorView {
    const dtype: DType = switch (t.simpleDType()) {
        .f32 => .f32,
        .f16 => .f16,
        .bf16 => .bf16,
        .i32 => .i32,
        .i64 => .i64,
        else => return null,
    };

    var view: TensorView = undefined;
    const ndim: usize = @intCast(t.n_dims);
    initView(&view, t.data_ptr orelse return null, ndim, dtype, t.numel, t.shape[0..ndim], t.strides[0..ndim]);

    return view;
}

test "initContiguous TensorView" {
    var data: [12]f32 = .{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const view = TensorView.initContiguous(@ptrCast(&data), &.{ 3, 4 }, .f32);

    try std.testing.expect(view.isContiguous());
    try std.testing.expectEqual(@as(usize, 12), view.numel);
    try std.testing.expectEqual(@as(usize, 4), view.strides[0]); // stride for dim 0
    try std.testing.expectEqual(@as(usize, 1), view.strides[1]); // stride for dim 1
}

test "initStrided TensorView non-contiguous" {
    var data: [12]f32 = .{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    // Transposed view: shape [4, 3] with strides [1, 4]
    const view = TensorView.initStrided(@ptrCast(&data), &.{ 4, 3 }, &.{ 1, 4 }, .f32);

    try std.testing.expect(!view.isContiguous());
    try std.testing.expectEqual(@as(usize, 12), view.numel);

    // Element [1, 2] should be at offset 1*1 + 2*4 = 9
    var coords = [_]usize{ 1, 2, 0, 0, 0, 0, 0, 0 };
    try std.testing.expectEqual(@as(usize, 9), view.coordsToOffset(coords[0..2]));
}

test "DType.elementSize returns correct sizes" {
    try std.testing.expectEqual(@as(usize, 4), DType.f32.elementSize());
    try std.testing.expectEqual(@as(usize, 2), DType.f16.elementSize());
    try std.testing.expectEqual(@as(usize, 2), DType.bf16.elementSize());
    try std.testing.expectEqual(@as(usize, 4), DType.i32.elementSize());
    try std.testing.expectEqual(@as(usize, 8), DType.i64.elementSize());
}

test "TensorView.initContiguous zero dimensions" {
    var data: [1]f32 = .{42.0};
    const view = TensorView.initContiguous(@ptrCast(&data), &.{}, .f32);

    try std.testing.expectEqual(@as(usize, 0), view.ndim);
    try std.testing.expectEqual(@as(usize, 1), view.numel);
    try std.testing.expect(view.isContiguous());
}

test "TensorView.initContiguous 1D" {
    var data: [5]f32 = .{ 1, 2, 3, 4, 5 };
    const view = TensorView.initContiguous(@ptrCast(&data), &.{5}, .f32);

    try std.testing.expectEqual(@as(usize, 1), view.ndim);
    try std.testing.expectEqual(@as(usize, 5), view.numel);
    try std.testing.expectEqual(@as(usize, 5), view.shape[0]);
    try std.testing.expectEqual(@as(usize, 1), view.strides[0]);
    try std.testing.expect(view.isContiguous());
}

test "TensorView.initContiguous 3D" {
    var data: [24]f32 = undefined;
    const view = TensorView.initContiguous(@ptrCast(&data), &.{ 2, 3, 4 }, .f32);

    try std.testing.expectEqual(@as(usize, 3), view.ndim);
    try std.testing.expectEqual(@as(usize, 24), view.numel);
    try std.testing.expectEqual(@as(usize, 2), view.shape[0]);
    try std.testing.expectEqual(@as(usize, 3), view.shape[1]);
    try std.testing.expectEqual(@as(usize, 4), view.shape[2]);
    // Row-major strides: [3*4, 4, 1] = [12, 4, 1]
    try std.testing.expectEqual(@as(usize, 12), view.strides[0]);
    try std.testing.expectEqual(@as(usize, 4), view.strides[1]);
    try std.testing.expectEqual(@as(usize, 1), view.strides[2]);
    try std.testing.expect(view.isContiguous());
}

test "TensorView.initContiguous with different dtypes" {
    var data_f16: [4]u16 = .{ 0, 1, 2, 3 };
    const view_f16 = TensorView.initContiguous(@ptrCast(&data_f16), &.{4}, .f16);
    try std.testing.expectEqual(DType.f16, view_f16.dtype);

    var data_i64: [4]i64 = .{ 0, 1, 2, 3 };
    const view_i64 = TensorView.initContiguous(@ptrCast(&data_i64), &.{4}, .i64);
    try std.testing.expectEqual(DType.i64, view_i64.dtype);
}

test "TensorView.initStrided custom strides" {
    var data: [6]f32 = .{ 1, 2, 3, 4, 5, 6 };
    // Shape [2, 3] with custom strides [3, 1] (row-major)
    const view = TensorView.initStrided(@ptrCast(&data), &.{ 2, 3 }, &.{ 3, 1 }, .f32);

    try std.testing.expectEqual(@as(usize, 2), view.ndim);
    try std.testing.expectEqual(@as(usize, 6), view.numel);
    try std.testing.expectEqual(@as(usize, 3), view.strides[0]);
    try std.testing.expectEqual(@as(usize, 1), view.strides[1]);
    try std.testing.expect(view.isContiguous());
}

test "TensorView.initStrided column-major layout" {
    var data: [6]f32 = .{ 1, 2, 3, 4, 5, 6 };
    // Shape [2, 3] with column-major strides [1, 2]
    const view = TensorView.initStrided(@ptrCast(&data), &.{ 2, 3 }, &.{ 1, 2 }, .f32);

    try std.testing.expectEqual(@as(usize, 2), view.ndim);
    try std.testing.expectEqual(@as(usize, 6), view.numel);
    try std.testing.expectEqual(@as(usize, 1), view.strides[0]);
    try std.testing.expectEqual(@as(usize, 2), view.strides[1]);
    try std.testing.expect(!view.isContiguous());
}

test "TensorView.isContiguous with zero dimensions" {
    var data: [1]f32 = .{0};
    const view = TensorView.initContiguous(@ptrCast(&data), &.{}, .f32);
    try std.testing.expect(view.isContiguous());
}

test "TensorView.isContiguous detects non-contiguous" {
    var data: [12]f32 = undefined;
    // Non-contiguous strides
    const view = TensorView.initStrided(@ptrCast(&data), &.{ 3, 4 }, &.{ 5, 1 }, .f32);
    try std.testing.expect(!view.isContiguous());
}

test "TensorView.asSlice returns correct slice" {
    var data: [6]f32 = .{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const view = TensorView.initContiguous(@ptrCast(&data), &.{ 2, 3 }, .f32);

    const slice = view.asSlice(f32);
    try std.testing.expectEqual(@as(usize, 6), slice.len);
    try std.testing.expectEqual(@as(f32, 1.0), slice[0]);
    try std.testing.expectEqual(@as(f32, 6.0), slice[5]);
}

test "TensorView.asSlice with different types" {
    var data_i32: [4]i32 = .{ 10, 20, 30, 40 };
    const view_i32 = TensorView.initContiguous(@ptrCast(&data_i32), &.{4}, .i32);
    const slice_i32 = view_i32.asSlice(i32);
    try std.testing.expectEqual(@as(usize, 4), slice_i32.len);
    try std.testing.expectEqual(@as(i32, 10), slice_i32[0]);
    try std.testing.expectEqual(@as(i32, 40), slice_i32[3]);

    var data_i64: [3]i64 = .{ 100, 200, 300 };
    const view_i64 = TensorView.initContiguous(@ptrCast(&data_i64), &.{3}, .i64);
    const slice_i64 = view_i64.asSlice(i64);
    try std.testing.expectEqual(@as(usize, 3), slice_i64.len);
    try std.testing.expectEqual(@as(i64, 100), slice_i64[0]);
}

test "TensorView.getElement 2D contiguous" {
    var data: [6]f32 = .{ 1, 2, 3, 4, 5, 6 };
    const view = TensorView.initContiguous(@ptrCast(&data), &.{ 2, 3 }, .f32);

    // [0, 0] -> 1
    try std.testing.expectEqual(@as(f32, 1), view.getElement(f32, &.{ 0, 0 }));
    // [0, 2] -> 3
    try std.testing.expectEqual(@as(f32, 3), view.getElement(f32, &.{ 0, 2 }));
    // [1, 0] -> 4
    try std.testing.expectEqual(@as(f32, 4), view.getElement(f32, &.{ 1, 0 }));
    // [1, 2] -> 6
    try std.testing.expectEqual(@as(f32, 6), view.getElement(f32, &.{ 1, 2 }));
}

test "TensorView.getElement 2D strided" {
    var data: [12]f32 = .{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    // Transposed view: shape [4, 3] with strides [1, 4]
    const view = TensorView.initStrided(@ptrCast(&data), &.{ 4, 3 }, &.{ 1, 4 }, .f32);

    // [0, 0] -> offset 0*1 + 0*4 = 0 -> data[0] = 1
    try std.testing.expectEqual(@as(f32, 1), view.getElement(f32, &.{ 0, 0 }));
    // [1, 0] -> offset 1*1 + 0*4 = 1 -> data[1] = 2
    try std.testing.expectEqual(@as(f32, 2), view.getElement(f32, &.{ 1, 0 }));
    // [0, 1] -> offset 0*1 + 1*4 = 4 -> data[4] = 5
    try std.testing.expectEqual(@as(f32, 5), view.getElement(f32, &.{ 0, 1 }));
    // [1, 2] -> offset 1*1 + 2*4 = 9 -> data[9] = 10
    try std.testing.expectEqual(@as(f32, 10), view.getElement(f32, &.{ 1, 2 }));
}

test "TensorView.setElement 2D contiguous" {
    var data: [6]f32 = .{ 0, 0, 0, 0, 0, 0 };
    const view = TensorView.initContiguous(@ptrCast(&data), &.{ 2, 3 }, .f32);

    view.setElement(f32, &.{ 0, 0 }, 1.0);
    view.setElement(f32, &.{ 0, 2 }, 3.0);
    view.setElement(f32, &.{ 1, 0 }, 4.0);
    view.setElement(f32, &.{ 1, 2 }, 6.0);

    try std.testing.expectEqual(@as(f32, 1.0), data[0]);
    try std.testing.expectEqual(@as(f32, 3.0), data[2]);
    try std.testing.expectEqual(@as(f32, 4.0), data[3]);
    try std.testing.expectEqual(@as(f32, 6.0), data[5]);
}

test "TensorView.setElement 2D strided" {
    var data: [12]f32 = .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    // Transposed view: shape [4, 3] with strides [1, 4]
    const view = TensorView.initStrided(@ptrCast(&data), &.{ 4, 3 }, &.{ 1, 4 }, .f32);

    view.setElement(f32, &.{ 0, 0 }, 1.0);
    view.setElement(f32, &.{ 1, 0 }, 2.0);
    view.setElement(f32, &.{ 0, 1 }, 5.0);
    view.setElement(f32, &.{ 1, 2 }, 10.0);

    // Check physical memory layout
    try std.testing.expectEqual(@as(f32, 1.0), data[0]); // offset 0
    try std.testing.expectEqual(@as(f32, 2.0), data[1]); // offset 1
    try std.testing.expectEqual(@as(f32, 5.0), data[4]); // offset 4
    try std.testing.expectEqual(@as(f32, 10.0), data[9]); // offset 9
}

test "TensorView.setElement with different types" {
    var data_i32: [4]i32 = .{ 0, 0, 0, 0 };
    const view_i32 = TensorView.initContiguous(@ptrCast(&data_i32), &.{4}, .i32);
    view_i32.setElement(i32, &.{2}, 42);
    try std.testing.expectEqual(@as(i32, 42), data_i32[2]);

    var data_i64: [4]i64 = .{ 0, 0, 0, 0 };
    const view_i64 = TensorView.initContiguous(@ptrCast(&data_i64), &.{4}, .i64);
    view_i64.setElement(i64, &.{1}, 12345);
    try std.testing.expectEqual(@as(i64, 12345), data_i64[1]);
}

test "TensorView.indexToCoords 1D" {
    var data: [5]f32 = undefined;
    const view = TensorView.initContiguous(@ptrCast(&data), &.{5}, .f32);

    var coords: [MAX_NDIM]usize = undefined;

    view.indexToCoords(0, &coords);
    try std.testing.expectEqual(@as(usize, 0), coords[0]);

    view.indexToCoords(3, &coords);
    try std.testing.expectEqual(@as(usize, 3), coords[0]);

    view.indexToCoords(4, &coords);
    try std.testing.expectEqual(@as(usize, 4), coords[0]);
}

test "TensorView.indexToCoords 2D" {
    var data: [6]f32 = undefined;
    const view = TensorView.initContiguous(@ptrCast(&data), &.{ 2, 3 }, .f32);

    var coords: [MAX_NDIM]usize = undefined;

    // Index 0 -> [0, 0]
    view.indexToCoords(0, &coords);
    try std.testing.expectEqual(@as(usize, 0), coords[0]);
    try std.testing.expectEqual(@as(usize, 0), coords[1]);

    // Index 2 -> [0, 2]
    view.indexToCoords(2, &coords);
    try std.testing.expectEqual(@as(usize, 0), coords[0]);
    try std.testing.expectEqual(@as(usize, 2), coords[1]);

    // Index 3 -> [1, 0]
    view.indexToCoords(3, &coords);
    try std.testing.expectEqual(@as(usize, 1), coords[0]);
    try std.testing.expectEqual(@as(usize, 0), coords[1]);

    // Index 5 -> [1, 2]
    view.indexToCoords(5, &coords);
    try std.testing.expectEqual(@as(usize, 1), coords[0]);
    try std.testing.expectEqual(@as(usize, 2), coords[1]);
}

test "TensorView.indexToCoords 3D" {
    var data: [24]f32 = undefined;
    const view = TensorView.initContiguous(@ptrCast(&data), &.{ 2, 3, 4 }, .f32);

    var coords: [MAX_NDIM]usize = undefined;

    // Index 0 -> [0, 0, 0]
    view.indexToCoords(0, &coords);
    try std.testing.expectEqual(@as(usize, 0), coords[0]);
    try std.testing.expectEqual(@as(usize, 0), coords[1]);
    try std.testing.expectEqual(@as(usize, 0), coords[2]);

    // Index 5 -> [0, 1, 1] (0*12 + 1*4 + 1)
    view.indexToCoords(5, &coords);
    try std.testing.expectEqual(@as(usize, 0), coords[0]);
    try std.testing.expectEqual(@as(usize, 1), coords[1]);
    try std.testing.expectEqual(@as(usize, 1), coords[2]);

    // Index 12 -> [1, 0, 0] (1*12 + 0*4 + 0)
    view.indexToCoords(12, &coords);
    try std.testing.expectEqual(@as(usize, 1), coords[0]);
    try std.testing.expectEqual(@as(usize, 0), coords[1]);
    try std.testing.expectEqual(@as(usize, 0), coords[2]);

    // Index 23 -> [1, 2, 3] (1*12 + 2*4 + 3 = 23)
    view.indexToCoords(23, &coords);
    try std.testing.expectEqual(@as(usize, 1), coords[0]);
    try std.testing.expectEqual(@as(usize, 2), coords[1]);
    try std.testing.expectEqual(@as(usize, 3), coords[2]);
}

test "TensorView.coordsToOffset 1D" {
    var data: [5]f32 = undefined;
    const view = TensorView.initContiguous(@ptrCast(&data), &.{5}, .f32);

    try std.testing.expectEqual(@as(usize, 0), view.coordsToOffset(&.{0}));
    try std.testing.expectEqual(@as(usize, 3), view.coordsToOffset(&.{3}));
    try std.testing.expectEqual(@as(usize, 4), view.coordsToOffset(&.{4}));
}

test "TensorView.coordsToOffset 2D contiguous" {
    var data: [6]f32 = undefined;
    const view = TensorView.initContiguous(@ptrCast(&data), &.{ 2, 3 }, .f32);

    // [0, 0] -> 0*3 + 0*1 = 0
    try std.testing.expectEqual(@as(usize, 0), view.coordsToOffset(&.{ 0, 0 }));
    // [0, 2] -> 0*3 + 2*1 = 2
    try std.testing.expectEqual(@as(usize, 2), view.coordsToOffset(&.{ 0, 2 }));
    // [1, 0] -> 1*3 + 0*1 = 3
    try std.testing.expectEqual(@as(usize, 3), view.coordsToOffset(&.{ 1, 0 }));
    // [1, 2] -> 1*3 + 2*1 = 5
    try std.testing.expectEqual(@as(usize, 5), view.coordsToOffset(&.{ 1, 2 }));
}

test "TensorView.coordsToOffset 3D" {
    var data: [24]f32 = undefined;
    const view = TensorView.initContiguous(@ptrCast(&data), &.{ 2, 3, 4 }, .f32);

    // [0, 0, 0] -> 0*12 + 0*4 + 0*1 = 0
    try std.testing.expectEqual(@as(usize, 0), view.coordsToOffset(&.{ 0, 0, 0 }));
    // [0, 1, 2] -> 0*12 + 1*4 + 2*1 = 6
    try std.testing.expectEqual(@as(usize, 6), view.coordsToOffset(&.{ 0, 1, 2 }));
    // [1, 0, 0] -> 1*12 + 0*4 + 0*1 = 12
    try std.testing.expectEqual(@as(usize, 12), view.coordsToOffset(&.{ 1, 0, 0 }));
    // [1, 2, 3] -> 1*12 + 2*4 + 3*1 = 23
    try std.testing.expectEqual(@as(usize, 23), view.coordsToOffset(&.{ 1, 2, 3 }));
}

test "TensorView round-trip indexToCoords and coordsToOffset" {
    var data: [12]f32 = undefined;
    const view = TensorView.initContiguous(@ptrCast(&data), &.{ 3, 4 }, .f32);

    var coords: [MAX_NDIM]usize = undefined;

    // Test all indices round-trip correctly
    for (0..12) |idx| {
        view.indexToCoords(idx, &coords);
        const offset = view.coordsToOffset(coords[0..view.ndim]);
        try std.testing.expectEqual(idx, offset);
    }
}

test "initContiguous unused dimensions zeroed" {
    var data: [6]f32 = undefined;
    const view = TensorView.initContiguous(@ptrCast(&data), &.{ 2, 3 }, .f32);

    // Dimensions beyond ndim should be zeroed
    for (view.ndim..MAX_NDIM) |i| {
        try std.testing.expectEqual(@as(usize, 0), view.shape[i]);
        try std.testing.expectEqual(@as(usize, 0), view.strides[i]);
    }
}

test "fromTensor creates view from tensor" {
    // Create a mock Tensor-like struct with the expected fields
    const MockTensor = struct {
        dtype: enum(u8) { f32 = 0, f64 = 1, i32 = 2, i64 = 3, f16 = 4, bf16 = 5 },
        n_dims: i32,
        shape: [MAX_NDIM]i64,
        data_ptr: ?[*]u8,
        strides: [MAX_NDIM]i64,
        numel: usize,
    };

    var data: [12]f32 = .{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const mock_tensor = MockTensor{
        .dtype = .f32,
        .n_dims = 2,
        .shape = .{ 3, 4, 0, 0, 0, 0, 0, 0 },
        .data_ptr = @ptrCast(&data),
        .strides = .{ 4, 1, 0, 0, 0, 0, 0, 0 },
        .numel = 12,
    };

    const view = fromTensor(MockTensor, &mock_tensor);

    try std.testing.expectEqual(DType.f32, view.dtype);
    try std.testing.expectEqual(@as(usize, 2), view.ndim);
    try std.testing.expectEqual(@as(usize, 12), view.numel);
    try std.testing.expectEqual(@as(usize, 3), view.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), view.shape[1]);
    try std.testing.expectEqual(@as(usize, 4), view.strides[0]);
    try std.testing.expectEqual(@as(usize, 1), view.strides[1]);
    try std.testing.expect(view.isContiguous());
}

test "fromSimpleTensor creates view from simple tensor" {
    var data: [24]f32 = undefined;
    const test_tensor = tensor.Tensor.view(@ptrCast(&data), &.{ 2, 3, 4 }, .f32, null);

    const view_opt = fromSimpleTensor(&test_tensor);
    try std.testing.expect(view_opt != null);

    const view = view_opt.?;
    try std.testing.expectEqual(DType.f32, view.dtype);
    try std.testing.expectEqual(@as(usize, 3), view.ndim);
    try std.testing.expectEqual(@as(usize, 24), view.numel);
    try std.testing.expectEqual(@as(usize, 2), view.shape[0]);
    try std.testing.expectEqual(@as(usize, 3), view.shape[1]);
    try std.testing.expectEqual(@as(usize, 4), view.shape[2]);
    // Row-major strides: [3*4, 4, 1] = [12, 4, 1]
    try std.testing.expectEqual(@as(usize, 12), view.strides[0]);
    try std.testing.expectEqual(@as(usize, 4), view.strides[1]);
    try std.testing.expectEqual(@as(usize, 1), view.strides[2]);
    try std.testing.expect(view.isContiguous());
}
