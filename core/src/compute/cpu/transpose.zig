//! TensorView transpose helper for CPU compute execution.

const std = @import("std");
const tv = @import("tensor_view.zig");

const TensorView = tv.TensorView;
const MAX_NDIM = tv.MAX_NDIM;

fn transposeTyped(
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
        input.indexToCoords(elem_idx, &coords);
        for (0..ndim) |dim_idx| out_coords[dim_idx] = coords[dim_idx];
        out_coords[dim0] = coords[dim1];
        out_coords[dim1] = coords[dim0];

        const in_offset = input.coordsToOffset(coords[0..ndim]);
        const out_offset = out.coordsToOffset(out_coords[0..ndim]);
        out_data[out_offset] = in_data[in_offset];
    }
}

pub fn transposeDispatch(out: TensorView, input: TensorView, dim0: usize, dim1: usize) void {
    switch (out.dtype) {
        .f32 => transposeTyped(f32, out, input, dim0, dim1),
        .f16, .bf16 => transposeTyped(u16, out, input, dim0, dim1),
        .i32 => transposeTyped(i32, out, input, dim0, dim1),
        .i64 => transposeTyped(i64, out, input, dim0, dim1),
    }
}

test "transposeDispatch swaps 2D axes for f32 views" {
    var input_data = [_]f32{
        1, 2, 3,
        4, 5, 6,
    };
    var out_data = [_]f32{0} ** 6;

    const input = TensorView.initContiguous(@ptrCast(&input_data), &.{ 2, 3 }, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{ 3, 2 }, .f32);
    transposeDispatch(out, input, 0, 1);

    try std.testing.expectEqualSlices(f32, &[_]f32{
        1, 4,
        2, 5,
        3, 6,
    }, &out_data);
}
