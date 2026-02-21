//! Tensor broadcast primitives for CPU compute path.

const std = @import("std");
const tensor = @import("../../tensor.zig");
const dtype_mod = @import("../../dtype.zig");

const Tensor = tensor.Tensor;

fn readParamValue(param: *const Tensor, elem_idx: usize) f32 {
    return switch (param.dtype) {
        .f32 => param.asSlice(f32)[elem_idx],
        .f16 => dtype_mod.fp16ToF32(param.asSlice(u16)[elem_idx]),
        .bf16 => dtype_mod.bf16ToF32(param.asSlice(u16)[elem_idx]),
        else => 0.0,
    };
}

pub fn applyElementwiseBinaryOp(
    left_tensor: Tensor,
    right_tensor: Tensor,
    output_slice: []f32,
    binary_op: fn (f32, f32) f32,
) !void {
    const left_values = left_tensor.asSlice(f32);
    const right_values = right_tensor.asSlice(f32);
    const left_count = left_tensor.numel;
    const right_count = right_tensor.numel;

    if (left_count == right_count) {
        for (0..left_count) |elem_idx| output_slice[elem_idx] = binary_op(left_values[elem_idx], right_values[elem_idx]);
        return;
    }

    if (left_tensor.n_dims == 4 and right_tensor.n_dims == 4 and left_tensor.shape[1] == right_tensor.shape[1] and left_tensor.shape[2] == right_tensor.shape[2]) {
        const seq_len: usize = @intCast(left_tensor.shape[1]);
        const head_count: usize = @intCast(left_tensor.shape[2]);
        const left_width: usize = @intCast(left_tensor.shape[3]);
        const right_width: usize = @intCast(right_tensor.shape[3]);
        if (left_width == 1 and right_width > 1) {
            for (0..seq_len) |seq_index| {
                for (0..head_count) |head_index| {
                    const base_offset = (seq_index * head_count + head_index) * right_width;
                    const left_value = left_values[seq_index * head_count + head_index];
                    for (0..right_width) |dim_index| {
                        output_slice[base_offset + dim_index] = binary_op(left_value, right_values[base_offset + dim_index]);
                    }
                }
            }
            return;
        }
        if (right_width == 1 and left_width > 1) {
            for (0..seq_len) |seq_index| {
                for (0..head_count) |head_index| {
                    const base_offset = (seq_index * head_count + head_index) * left_width;
                    const right_value = right_values[seq_index * head_count + head_index];
                    for (0..left_width) |dim_index| {
                        output_slice[base_offset + dim_index] = binary_op(left_values[base_offset + dim_index], right_value);
                    }
                }
            }
            return;
        }
    }

    if (left_tensor.n_dims == 3 and right_tensor.n_dims == 3 and left_tensor.shape[1] == right_tensor.shape[1]) {
        const seq_len: usize = @intCast(left_tensor.shape[1]);
        if (left_tensor.shape[2] == 1 and right_tensor.shape[2] > 1) {
            const right_hidden_size: usize = @intCast(right_tensor.shape[2]);
            for (0..seq_len) |seq_index| {
                const left_value = left_values[seq_index];
                for (0..right_hidden_size) |hidden_index| {
                    output_slice[seq_index * right_hidden_size + hidden_index] = binary_op(left_value, right_values[seq_index * right_hidden_size + hidden_index]);
                }
            }
            return;
        }
        if (right_tensor.shape[2] == 1 and left_tensor.shape[2] > 1) {
            const left_hidden_size: usize = @intCast(left_tensor.shape[2]);
            for (0..seq_len) |seq_index| {
                const right_value = right_values[seq_index];
                for (0..left_hidden_size) |hidden_index| {
                    output_slice[seq_index * left_hidden_size + hidden_index] = binary_op(left_values[seq_index * left_hidden_size + hidden_index], right_value);
                }
            }
            return;
        }
    }

    if (left_tensor.n_dims == 1 and right_tensor.n_dims == 3 and left_tensor.shape[0] == right_tensor.shape[2]) {
        const seq_len: usize = @intCast(right_tensor.shape[1]);
        const hidden_size: usize = @intCast(right_tensor.shape[2]);
        for (0..seq_len) |seq_index| {
            const base_offset = seq_index * hidden_size;
            for (0..hidden_size) |hidden_index| {
                output_slice[base_offset + hidden_index] = binary_op(left_values[hidden_index], right_values[base_offset + hidden_index]);
            }
        }
        return;
    }

    if (right_tensor.n_dims == 1 and left_tensor.n_dims == 3 and right_tensor.shape[0] == left_tensor.shape[2]) {
        const seq_len: usize = @intCast(left_tensor.shape[1]);
        const hidden_size: usize = @intCast(left_tensor.shape[2]);
        for (0..seq_len) |seq_index| {
            const base_offset = seq_index * hidden_size;
            for (0..hidden_size) |hidden_index| {
                output_slice[base_offset + hidden_index] = binary_op(left_values[base_offset + hidden_index], right_values[hidden_index]);
            }
        }
        return;
    }

    if (left_tensor.n_dims == 1 and right_tensor.n_dims == 4 and left_tensor.shape[0] == right_tensor.shape[3]) {
        const seq_len: usize = @intCast(right_tensor.shape[1]);
        const head_count: usize = @intCast(right_tensor.shape[2]);
        const hidden_size: usize = @intCast(right_tensor.shape[3]);
        for (0..seq_len) |seq_index| {
            for (0..head_count) |head_index| {
                const base_offset = (seq_index * head_count + head_index) * hidden_size;
                for (0..hidden_size) |dim_index| {
                    output_slice[base_offset + dim_index] = binary_op(left_values[dim_index], right_values[base_offset + dim_index]);
                }
            }
        }
        return;
    }

    if (right_tensor.n_dims == 1 and left_tensor.n_dims == 4 and right_tensor.shape[0] == left_tensor.shape[3]) {
        const seq_len: usize = @intCast(left_tensor.shape[1]);
        const head_count: usize = @intCast(left_tensor.shape[2]);
        const hidden_size: usize = @intCast(left_tensor.shape[3]);
        for (0..seq_len) |seq_index| {
            for (0..head_count) |head_index| {
                const base_offset = (seq_index * head_count + head_index) * hidden_size;
                for (0..hidden_size) |dim_index| {
                    output_slice[base_offset + dim_index] = binary_op(left_values[base_offset + dim_index], right_values[dim_index]);
                }
            }
        }
        return;
    }

    return error.InvalidBroadcast;
}

pub fn addParam(input_tensor: Tensor, param: *const Tensor, output_slice: []f32) !void {
    const input_data = input_tensor.asSlice(f32);
    const output_len = @max(input_tensor.numel, param.numel);

    if (param.n_dims == 1 and input_tensor.n_dims == 4 and param.shape[0] == input_tensor.shape[3]) {
        const add_seq_len_4d: usize = @intCast(input_tensor.shape[1]);
        const head_count: usize = @intCast(input_tensor.shape[2]);
        const hidden_size: usize = @intCast(input_tensor.shape[3]);
        for (0..add_seq_len_4d) |token_idx| {
            for (0..head_count) |head_idx| {
                const row_base = (token_idx * head_count + head_idx) * hidden_size;
                for (0..hidden_size) |hidden_idx| {
                    output_slice[row_base + hidden_idx] = input_data[row_base + hidden_idx] + readParamValue(param, hidden_idx);
                }
            }
        }
    } else if (param.n_dims == 1 and input_tensor.n_dims == 3 and param.shape[0] == input_tensor.shape[2]) {
        const add_seq_len_3d: usize = @intCast(input_tensor.shape[1]);
        const hidden_size: usize = @intCast(input_tensor.shape[2]);
        for (0..add_seq_len_3d) |token_idx| {
            const row_base = token_idx * hidden_size;
            for (0..hidden_size) |hidden_idx| {
                output_slice[row_base + hidden_idx] = input_data[row_base + hidden_idx] + readParamValue(param, hidden_idx);
            }
        }
    } else {
        const p_len = param.numel;
        if (input_tensor.numel != p_len) {
            return error.InvalidBroadcast;
        }
        for (0..p_len) |elem_idx| {
            output_slice[elem_idx] = input_data[elem_idx] + readParamValue(param, elem_idx);
        }
    }

    _ = output_len;
}

pub fn mulParam(input_tensor: Tensor, param: *const Tensor, output_slice: []f32) !void {
    const input_data = input_tensor.asSlice(f32);
    const output_len = @max(input_tensor.numel, param.numel);

    if (param.n_dims == 1 and input_tensor.n_dims == 4 and param.shape[0] == input_tensor.shape[3]) {
        const mul_seq_len_4d: usize = @intCast(input_tensor.shape[1]);
        const head_count: usize = @intCast(input_tensor.shape[2]);
        const hidden_size: usize = @intCast(input_tensor.shape[3]);
        for (0..mul_seq_len_4d) |token_idx| {
            for (0..head_count) |head_idx| {
                const row_base = (token_idx * head_count + head_idx) * hidden_size;
                for (0..hidden_size) |hidden_idx| {
                    output_slice[row_base + hidden_idx] = input_data[row_base + hidden_idx] * readParamValue(param, hidden_idx);
                }
            }
        }
    } else if (param.n_dims == 1 and input_tensor.n_dims == 3 and param.shape[0] == input_tensor.shape[2]) {
        const mul_seq_len_3d: usize = @intCast(input_tensor.shape[1]);
        const hidden_size: usize = @intCast(input_tensor.shape[2]);
        for (0..mul_seq_len_3d) |token_idx| {
            const row_base = token_idx * hidden_size;
            for (0..hidden_size) |hidden_idx| {
                output_slice[row_base + hidden_idx] = input_data[row_base + hidden_idx] * readParamValue(param, hidden_idx);
            }
        }
    } else {
        const p_len = param.numel;
        if (input_tensor.numel != p_len) {
            return error.InvalidBroadcast;
        }
        for (0..p_len) |elem_idx| {
            output_slice[elem_idx] = input_data[elem_idx] * readParamValue(param, elem_idx);
        }
    }

    _ = output_len;
}

pub fn addParamScalar(param: *const Tensor, output_slice: []f32, scalar: f32) void {
    const p_len = param.numel;
    for (0..p_len) |elem_idx| {
        output_slice[elem_idx] = readParamValue(param, elem_idx) + scalar;
    }
}

test "applyElementwiseBinaryOp applies binary op for equal shapes" {
    var left_data = [_]f32{ 1, 2, 3, 4 };
    var right_data = [_]f32{ 10, 20, 30, 40 };
    const left = Tensor.view2DSlice(&left_data, 2, 2);
    const right = Tensor.view2DSlice(&right_data, 2, 2);
    var out = [_]f32{0} ** 4;

    const addOp = struct {
        fn call(a: f32, b: f32) f32 {
            return a + b;
        }
    }.call;

    try applyElementwiseBinaryOp(left, right, &out, addOp);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 11, 22, 33, 44 }, &out);
}

test "addParam and mulParam broadcast row parameter over 3D input" {
    var input_data = [_]f32{
        1, 2, 3,
        4, 5, 6,
    };
    const input = Tensor.view3DSlice(&input_data, 2, 3);
    var param_data = [_]f32{ 10, 20, 30 };
    const param = Tensor{
        .dtype = .f32,
        .n_dims = 1,
        .shape = .{ 3, 0, 0, 0, 0, 0, 0, 0 },
        .data_ptr = @ptrCast(param_data[0..].ptr),
        .data_size = param_data.len * @sizeOf(f32),
        .numel = param_data.len,
        .strides = .{ 1, 0, 0, 0, 0, 0, 0, 0 },
    };

    var add_out = [_]f32{0} ** 6;
    try addParam(input, &param, &add_out);
    try std.testing.expectEqualSlices(f32, &[_]f32{
        11, 22, 33,
        14, 25, 36,
    }, &add_out);

    var mul_out = [_]f32{0} ** 6;
    try mulParam(input, &param, &mul_out);
    try std.testing.expectEqualSlices(f32, &[_]f32{
        10, 40,  90,
        40, 100, 180,
    }, &mul_out);
}

test "addParamScalar adds constant to parameter tensor" {
    var param_data = [_]f32{ 1, 2, 3 };
    const param = Tensor{
        .dtype = .f32,
        .n_dims = 1,
        .shape = .{ 3, 0, 0, 0, 0, 0, 0, 0 },
        .data_ptr = @ptrCast(param_data[0..].ptr),
        .data_size = param_data.len * @sizeOf(f32),
        .numel = param_data.len,
        .strides = .{ 1, 0, 0, 0, 0, 0, 0, 0 },
    };
    var out = [_]f32{ 0, 0, 0 };
    addParamScalar(&param, &out, 0.5);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1.5, 2.5, 3.5 }, &out);
}
