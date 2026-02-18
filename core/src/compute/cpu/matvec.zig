//! Matrix-vector primitives used by CPU kernels.

const std = @import("std");
const tensor = @import("../../tensor.zig");
const dtype_mod = @import("../../dtype.zig");

const Tensor = tensor.Tensor;

/// Compute router logits for MoE routing:
/// `logits = input_vector @ router_weight (+ bias)`.
pub fn denseLogits(
    input_vector: []const f32,
    router_weight: *const Tensor,
    router_bias: ?[]const f32,
    logits_out: []f32,
) void {
    const input_dim = input_vector.len;
    const num_experts = logits_out.len;

    const rows: usize = @intCast(router_weight.shape[0]);
    const cols: usize = @intCast(router_weight.shape[1]);
    const weight_dtype = router_weight.dtype;

    const readWeight = struct {
        fn at(dtype: tensor.DType, data_ptr: [*]const u8, idx: usize) f32 {
            return switch (dtype) {
                .f32 => @as([*]const f32, @ptrCast(@alignCast(data_ptr)))[idx],
                .bf16 => dtype_mod.bf16ToF32(@as([*]align(1) const u16, @ptrCast(data_ptr))[idx]),
                .f16 => @floatCast(@as([*]align(1) const f16, @ptrCast(data_ptr))[idx]),
                else => 0.0,
            };
        }
    };
    const weight_ptr: [*]const u8 = router_weight.data().ptr;

    if (rows == input_dim and cols == num_experts) {
        for (0..num_experts) |expert_index| {
            var sum: f32 = 0.0;
            for (0..input_dim) |input_idx| {
                const w = readWeight.at(weight_dtype, weight_ptr, input_idx * num_experts + expert_index);
                sum += input_vector[input_idx] * w;
            }
            if (router_bias) |bias| sum += bias[expert_index];
            logits_out[expert_index] = sum;
        }
        return;
    }

    if (rows == num_experts and cols == input_dim) {
        for (0..num_experts) |expert_index| {
            var sum: f32 = 0.0;
            for (0..input_dim) |input_idx| {
                const w = readWeight.at(weight_dtype, weight_ptr, expert_index * input_dim + input_idx);
                sum += input_vector[input_idx] * w;
            }
            if (router_bias) |bias| sum += bias[expert_index];
            logits_out[expert_index] = sum;
        }
        return;
    }

    // Preserve deterministic behavior on invalid shape.
    for (0..num_experts) |expert_index| {
        logits_out[expert_index] = if (router_bias) |bias| bias[expert_index] else 0.0;
    }
}

test "denseLogits computes logits for [experts, input_dim] layout" {
    const allocator = std.testing.allocator;
    var weight = try tensor.OwnedTensor.init(allocator, .f32, &.{ 2, 3 });
    defer weight.deinit();
    const ws = weight.asSlice(f32);
    @memcpy(ws, &[_]f32{
        1, 0, 0, // expert 0
        0, 2, 0, // expert 1
    });
    const input = [_]f32{ 3, 4, 5 };
    const bias = [_]f32{ 0.5, -1.0 };
    var logits = [_]f32{ 0, 0 };
    var w_view = weight.view();

    denseLogits(&input, &w_view, &bias, &logits);

    try std.testing.expectApproxEqAbs(@as(f32, 3.5), logits[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), logits[1], 1e-6);
}
