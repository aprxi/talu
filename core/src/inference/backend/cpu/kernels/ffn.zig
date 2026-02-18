//! CPU Feed-Forward Network Kernel
//! SwiGLU (Swish-Gated Linear Unit) implementation
//!
//! This module provides the feed-forward network computation for CPU inference.
//! Uses SwiGLU activation: output = (SiLU(x @ W1) * (x @ W3)) @ W2

const std = @import("std");
const build_options = @import("build_options");
const tensor = @import("../../../../tensor.zig");
const compute = @import("../../../../compute/root.zig");
const matmul = compute.ops.matmul;
const cpu_activation = compute.cpu.activation;
const cpu_common = compute.cpu.common;
const inspect = @import("../../../../xray/root.zig");
const trace = inspect.trace;
const dump = if (build_options.dump_tensors) @import("../../../../xray/dump/capture.zig") else struct {
    pub fn recordGlobal(_: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) void {}
};

const Tensor = tensor.Tensor;
const MatmulFn = matmul.MatmulFn;

pub const GateUpLayout = enum {
    concat,
    interleaved,
};

/// Scratch buffers for FFN computation.
/// Pre-allocated to avoid allocation during inference.
pub const FfnScratch = struct {
    gate: []f32 = &.{},
    gate_act: []f32 = &.{},
    up: []f32 = &.{},
    hidden: []f32 = &.{},

    pub fn deinit(self: *FfnScratch, allocator: std.mem.Allocator) void {
        if (self.gate.len > 0) allocator.free(self.gate);
        if (self.gate_act.len > 0) allocator.free(self.gate_act);
        if (self.up.len > 0) allocator.free(self.up);
        if (self.hidden.len > 0) allocator.free(self.hidden);
        self.* = .{};
    }
};

/// SwiGLU Feed-Forward Network layer.
/// Computes: output = (SiLU(x @ W1) * (x @ W3)) @ W2
pub const SwiGLU = struct {
    /// Canonical kernel-call contract for backend parity checks.
    pub const ForwardParams = struct {
        input_tensor: *const Tensor,
        output_tensor: *Tensor,
        scratch: *FfnScratch,
        matmul_scratch: *matmul.MatmulScratch,
    };

    d_model: usize,
    d_ff: usize,
    use_gelu: bool = false,
    use_swiglu_variant: bool = false,
    layer_idx: u16 = trace.TraceEmission.NO_LAYER,
    w1: ?*const Tensor = null,
    w2: *const Tensor,
    w3: ?*const Tensor = null,
    w1_bias: ?[]const f32 = null,
    w2_bias: ?[]const f32 = null,
    fused_gate_up: ?Tensor = null,
    fused_gate_up_layout: GateUpLayout = .concat,
    allocator: std.mem.Allocator,
    // Baked matmul kernels - resolved at load time, no runtime dispatch
    matmul_gate: MatmulFn, // for w1, w3
    matmul_gate_up: ?MatmulFn = null, // for fused gate+up
    matmul_down: MatmulFn, // for w2
    kernel_name_gate: ?[]const u8 = null,
    kernel_name_gate_up: ?[]const u8 = null,
    kernel_name_down: ?[]const u8 = null,

    pub fn forward(self: *const SwiGLU, input_tensor: *const Tensor, output_tensor: *Tensor, scratch: *FfnScratch, matmul_scratch: *matmul.MatmulScratch) !void {
        // Internal invariants: tensor shapes must match model config
        std.debug.assert(input_tensor.n_dims == 3 and output_tensor.n_dims == 3);
        std.debug.assert(input_tensor.shape[0] == 1 and output_tensor.shape[0] == 1); // Only batch=1 supported
        const sequence_len: usize = @intCast(input_tensor.shape[1]);
        std.debug.assert(input_tensor.shape[2] == self.d_model and output_tensor.shape[2] == self.d_model);

        const use_fused_gate_up = if (self.fused_gate_up) |fg| blk: {
            if (fg.n_dims != 2) break :blk false;
            const matches_d_model_first = fg.shape[0] == self.d_model and fg.shape[1] == self.d_ff * 2;
            const matches_d_model_second = fg.shape[0] == self.d_ff * 2 and fg.shape[1] == self.d_model;
            break :blk matches_d_model_first or matches_d_model_second;
        } else false;

        // Dense-only MLP: w1 (dense_in) + activation + w2 (dense_out), no gate*up multiply.
        // Used by BERT-family models where the FFN is: GELU(x@W1 + b1) @ W2 + b2.
        const is_dense_only = (self.w3 == null and !use_fused_gate_up);

        const gate_buffer_len = if (use_fused_gate_up) sequence_len * (2 * self.d_ff) else sequence_len * self.d_ff;

        try cpu_common.ensureF32Slice(self.allocator, &scratch.gate, gate_buffer_len);
        if (!use_fused_gate_up) {
            try cpu_common.ensureF32Slice(self.allocator, &scratch.gate_act, sequence_len * self.d_ff);
            if (!is_dense_only) {
                try cpu_common.ensureF32Slice(self.allocator, &scratch.hidden, sequence_len * self.d_ff);
                try cpu_common.ensureF32Slice(self.allocator, &scratch.up, sequence_len * self.d_ff);
            }
        } else {
            // For the common decode case (sequence_len=1) and concat layout, we can compute the
            // activation in-place into the gate half and avoid packing into `hidden`.
            if (!(sequence_len == 1 and self.fused_gate_up_layout == .concat)) {
                try cpu_common.ensureF32Slice(self.allocator, &scratch.hidden, sequence_len * self.d_ff);
            }
        }

        const input_view = Tensor.view2D(input_tensor.data(), sequence_len, self.d_model);
        var gate_output: Tensor = undefined; // Safe: both branches assign before use
        var up_output: Tensor = undefined; // Safe: both branches assign before use
        if (use_fused_gate_up) {
            const fused_weight = self.fused_gate_up.?;
            const fused_kernel = self.matmul_gate_up orelse self.matmul_gate;
            var gate_up_output = Tensor.view2DSlice(scratch.gate[0 .. sequence_len * (2 * self.d_ff)], sequence_len, 2 * self.d_ff);
            fused_kernel(&input_view, &fused_weight, &gate_up_output, matmul_scratch);
            gate_output = gate_up_output;
            up_output = gate_up_output;
        } else {
            const gate_weight = self.w1 orelse return error.MissingFFNWeights;
            var gate_workspace = Tensor.view2DSlice(scratch.gate[0 .. sequence_len * self.d_ff], sequence_len, self.d_ff);
            self.matmul_gate(&input_view, gate_weight, &gate_workspace, matmul_scratch);
            if (self.w1_bias) |bias| cpu_common.addBiasRows(gate_workspace.asSlice(f32), bias, sequence_len, self.d_ff);
            gate_output = gate_workspace;

            if (is_dense_only) {
                up_output = gate_workspace; // unused in dense-only path
            } else {
                const up_weight = self.w3 orelse return error.MissingFFNWeights;
                var up_workspace = Tensor.view2DSlice(scratch.up, sequence_len, self.d_ff);
                self.matmul_gate(&input_view, up_weight, &up_workspace, matmul_scratch);
                up_output = up_workspace;
            }
        }

        if (trace.isEnabled()) {
            const gate_dim: usize = if (use_fused_gate_up) 2 * self.d_ff else self.d_ff;
            // Get kernel name for gate/up operation
            const kernel_name: ?[]const u8 = if (use_fused_gate_up)
                self.kernel_name_gate_up
            else
                self.kernel_name_gate;
            trace.emit(
                .ffn_gate,
                self.layer_idx,
                0,
                @intCast(sequence_len),
                gate_output.data().ptr,
                .f32,
                .{ 1, @intCast(sequence_len), @intCast(gate_dim), 0 },
                3,
                kernel_name,
            );
            // Emit ffn_up only for split path (separate up projection)
            // In fused path, gate+up are a single matmul already reported by ffn_gate
            if (!use_fused_gate_up) {
                trace.emit(
                    .ffn_up,
                    self.layer_idx,
                    0,
                    @intCast(sequence_len),
                    up_output.data().ptr,
                    .f32,
                    .{ 1, @intCast(sequence_len), @intCast(self.d_ff), 0 },
                    3,
                    self.kernel_name_gate,
                );
            }
        }

        // Apply activation and elementwise multiply.
        const hidden_element_count = sequence_len * self.d_ff;
        var hidden_output: Tensor = undefined; // Safe: both branches assign before use
        if (use_fused_gate_up) {
            const gate_up_output = gate_output.asSlice(f32);
            if (self.fused_gate_up_layout == .interleaved) {
                const hidden_values = scratch.hidden[0..hidden_element_count];
                for (0..sequence_len) |token_index| {
                    const row = gate_up_output[token_index * (2 * self.d_ff) ..][0 .. 2 * self.d_ff];
                    const out_row = hidden_values[token_index * self.d_ff ..][0..self.d_ff];
                    if (self.use_swiglu_variant) {
                        cpu_activation.swigluVariantInterleaved(row, out_row);
                    } else if (self.use_gelu) {
                        cpu_activation.geluMulInterleaved(row, out_row);
                    } else {
                        cpu_activation.siluMulInterleaved(row, out_row);
                    }
                }
                hidden_output = Tensor.view2DSlice(hidden_values, sequence_len, self.d_ff);
            } else {
                // Concat layout: [gate..., up...] per token.
                // Fast path for decode (sequence_len=1): compute in-place into the gate half to
                // avoid packing into a separate buffer.
                if (sequence_len == 1) {
                    const row = gate_up_output[0 .. 2 * self.d_ff];
                    const gate_row = row[0..self.d_ff];
                    const up_row = row[self.d_ff .. 2 * self.d_ff];
                    if (self.use_swiglu_variant) {
                        cpu_activation.swigluVariantSplit(gate_row, up_row, gate_row);
                    } else if (self.use_gelu) {
                        cpu_activation.geluMulSplit(gate_row, up_row, gate_row);
                    } else {
                        cpu_activation.siluMulSplit(gate_row, up_row, gate_row);
                    }
                    hidden_output = Tensor.view2DSlice(gate_row, 1, self.d_ff);
                } else {
                    const hidden_values = scratch.hidden[0..hidden_element_count];
                    for (0..sequence_len) |token_index| {
                        const row = gate_up_output[token_index * (2 * self.d_ff) ..][0 .. 2 * self.d_ff];
                        const gate_row = row[0..self.d_ff];
                        const up_row = row[self.d_ff .. 2 * self.d_ff];
                        const out_row = hidden_values[token_index * self.d_ff ..][0..self.d_ff];
                        if (self.use_swiglu_variant) {
                            cpu_activation.swigluVariantSplit(gate_row, up_row, out_row);
                        } else if (self.use_gelu) {
                            cpu_activation.geluMulSplit(gate_row, up_row, out_row);
                        } else {
                            cpu_activation.siluMulSplit(gate_row, up_row, out_row);
                        }
                    }
                    hidden_output = Tensor.view2DSlice(hidden_values, sequence_len, self.d_ff);
                }
            }
        } else {
            var gate_activation_view = Tensor.view2DSlice(scratch.gate_act, sequence_len, self.d_ff);
            if (self.use_gelu) {
                const gate_values = gate_output.asSlice(f32);
                const gate_activation_values = gate_activation_view.asSlice(f32);
                cpu_activation.geluMap(gate_values, gate_activation_values);
            } else if (self.use_swiglu_variant) {
                const gate = gate_output.asSlice(f32)[0..hidden_element_count];
                const up = up_output.asSlice(f32)[0..hidden_element_count];
                cpu_activation.swigluVariantSplit(gate, up, scratch.hidden[0..hidden_element_count]);
                hidden_output = Tensor.view2DSlice(scratch.hidden[0..hidden_element_count], sequence_len, self.d_ff);
                if (trace.isEnabled()) {
                    trace.emit(
                        .ffn_act,
                        self.layer_idx,
                        0,
                        @intCast(sequence_len),
                        hidden_output.data().ptr,
                        .f32,
                        .{ 1, @intCast(sequence_len), @intCast(self.d_ff), 0 },
                        3,
                        null,
                    );
                }
                var out_view = Tensor.view2DSlice(output_tensor.asSlice(f32), sequence_len, self.d_model);
                self.matmul_down(&hidden_output, self.w2, &out_view, matmul_scratch);
                if (self.w2_bias) |bias| cpu_common.addBiasRows(output_tensor.asSlice(f32), bias, sequence_len, self.d_model);
                if (trace.isEnabled()) {
                    trace.emit(
                        .ffn_down,
                        self.layer_idx,
                        0,
                        @intCast(sequence_len),
                        output_tensor.data().ptr,
                        .f32,
                        .{ 1, @intCast(sequence_len), @intCast(self.d_model), 0 },
                        3,
                        self.kernel_name_down,
                    );
                }
                // Dump capture (compiled in only for dump binary)
                if (build_options.dump_tensors) {
                    const shape = [4]usize{ 1, sequence_len, self.d_model, 0 };
                    dump.recordGlobal(.ffn_down, self.layer_idx, output_tensor.data().ptr, .f32, shape, 3);
                }
                return;
            } else {
                cpu_activation.siluMap(gate_output.asSlice(f32), gate_activation_view.asSlice(f32));
            }

            if (is_dense_only) {
                // Dense-only MLP: activation(x@W1 + b1) is the hidden state, no gate*up.
                hidden_output = gate_activation_view;
            } else {
                // hidden = gate_act * up (SIMD)
                const gate_activation_values = gate_activation_view.asSlice(f32);
                const up_values = up_output.asSlice(f32);
                const hidden_values = scratch.hidden[0..hidden_element_count];
                cpu_activation.elementwiseMul(gate_activation_values, up_values, hidden_values);
                hidden_output = Tensor.view2DSlice(hidden_values, sequence_len, self.d_ff);
            }
        }

        if (trace.isEnabled()) {
            trace.emit(
                .ffn_act,
                self.layer_idx,
                0,
                @intCast(sequence_len),
                hidden_output.data().ptr,
                .f32,
                .{ 1, @intCast(sequence_len), @intCast(self.d_ff), 0 },
                3,
                null,
            );
        }

        var out_view = Tensor.view2DSlice(output_tensor.asSlice(f32), sequence_len, self.d_model);
        self.matmul_down(&hidden_output, self.w2, &out_view, matmul_scratch);
        if (self.w2_bias) |bias| cpu_common.addBiasRows(output_tensor.asSlice(f32), bias, sequence_len, self.d_model);
        if (trace.isEnabled()) {
            trace.emit(
                .ffn_down,
                self.layer_idx,
                0,
                @intCast(sequence_len),
                output_tensor.data().ptr,
                .f32,
                .{ 1, @intCast(sequence_len), @intCast(self.d_model), 0 },
                3,
                self.kernel_name_down,
            );
        }
        // Dump capture (compiled in only for dump binary)
        if (build_options.dump_tensors) {
            const shape = [4]usize{ 1, sequence_len, self.d_model, 0 };
            dump.recordGlobal(.ffn_down, self.layer_idx, output_tensor.data().ptr, .f32, shape, 3);
        }
    }
};

// =============================================================================
// Unit Tests
// =============================================================================

test "forward SiLU zero" {
    // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    const result = cpu_activation.silu(0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result, 1e-4);
}

test "forward SiLU positive" {
    // silu(1) = 1 * sigmoid(1) ≈ 1 * 0.731 ≈ 0.731
    const result = cpu_activation.silu(1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.731), result, 0.01);
}

test "forward SiLU negative" {
    // silu(-1) = -1 * sigmoid(-1) ≈ -1 * 0.269 ≈ -0.269
    const result = cpu_activation.silu(-1.0);
    try std.testing.expectApproxEqAbs(@as(f32, -0.269), result, 0.01);
}

test "forward SiLU large positive" {
    // silu(5) ≈ 5 * 0.993 ≈ 4.966
    const result = cpu_activation.silu(5.0);
    try std.testing.expectApproxEqAbs(@as(f32, 4.966), result, 0.01);
}

test "forward SiLU large negative" {
    // silu(-5) ≈ -5 * 0.0067 ≈ -0.033
    const result = cpu_activation.silu(-5.0);
    try std.testing.expectApproxEqAbs(@as(f32, -0.033), result, 0.01);
}

test "forward GELU zero" {
    const result = cpu_activation.geluApprox(0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result, 1e-4);
}

test "forward GELU positive" {
    // GELU(1) ≈ 0.841
    const result = cpu_activation.geluApprox(1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.841), result, 0.01);
}

test "forward GELU negative" {
    // GELU(-1) ≈ -0.159
    const result = cpu_activation.geluApprox(-1.0);
    try std.testing.expectApproxEqAbs(@as(f32, -0.159), result, 0.01);
}

test "forward SwiGLU scalar basic" {
    // Test within clip range
    const gate: f32 = 1.0;
    const up: f32 = 2.0;
    const result = cpu_activation.swigluVariantScalar(gate, up);

    // Expected: (1.0 * sigmoid(1.702 * 1.0)) * (2.0 + 1)
    const alpha: f32 = 1.702;
    const sig = 1.0 / (1.0 + @exp(-alpha * gate));
    const expected = (gate * sig) * (up + 1.0);
    try std.testing.expectApproxEqAbs(expected, result, 1e-4);
}

test "forward SwiGLU gate clipping positive" {
    // Test gate clipping at +7
    const gate: f32 = 10.0; // Should clip to 7.0
    const up: f32 = 1.0;
    const result = cpu_activation.swigluVariantScalar(gate, up);

    const alpha: f32 = 1.702;
    const gate_clipped: f32 = 7.0;
    const sig = 1.0 / (1.0 + @exp(-alpha * gate_clipped));
    const expected = (gate_clipped * sig) * (up + 1.0);
    try std.testing.expectApproxEqAbs(expected, result, 1e-4);
}

test "forward SwiGLU gate clipping negative" {
    // Test gate clipping at -7
    const gate: f32 = -10.0; // Should clip to -7.0
    const up: f32 = 1.0;
    const result = cpu_activation.swigluVariantScalar(gate, up);

    const alpha: f32 = 1.702;
    const gate_clipped: f32 = -7.0;
    const sig = 1.0 / (1.0 + @exp(-alpha * gate_clipped));
    const expected = (gate_clipped * sig) * (up + 1.0);
    try std.testing.expectApproxEqAbs(expected, result, 1e-4);
}

test "forward SwiGLU up clipping" {
    // Test up value clipping
    const gate: f32 = 1.0;
    const up: f32 = 10.0; // Should clip to 7.0
    const result = cpu_activation.swigluVariantScalar(gate, up);

    const alpha: f32 = 1.702;
    const up_clipped: f32 = 7.0;
    const sig = 1.0 / (1.0 + @exp(-alpha * gate));
    const expected = (gate * sig) * (up_clipped + 1.0);
    try std.testing.expectApproxEqAbs(expected, result, 1e-4);
}

test "forward SwiGLU interleaved" {
    const allocator = std.testing.allocator;

    // Input: [gate0, up0, gate1, up1]
    const interleaved = [_]f32{ 1.0, 2.0, -1.0, 0.5 };
    const output = try allocator.alloc(f32, 2);
    defer allocator.free(output);

    cpu_activation.swigluVariantInterleaved(&interleaved, output);

    // Expected outputs
    const expected0 = cpu_activation.swigluVariantScalar(1.0, 2.0);
    const expected1 = cpu_activation.swigluVariantScalar(-1.0, 0.5);

    try std.testing.expectApproxEqAbs(expected0, output[0], 1e-4);
    try std.testing.expectApproxEqAbs(expected1, output[1], 1e-4);
}

test "forward SwiGLU split" {
    const allocator = std.testing.allocator;

    const gate = [_]f32{ 1.0, -1.0, 2.0 };
    const up = [_]f32{ 2.0, 0.5, -0.5 };
    const output = try allocator.alloc(f32, 3);
    defer allocator.free(output);

    cpu_activation.swigluVariantSplit(&gate, &up, output);

    for (0..3) |i| {
        const expected = cpu_activation.swigluVariantScalar(gate[i], up[i]);
        try std.testing.expectApproxEqAbs(expected, output[i], 1e-4);
    }
}

test "forward SwiGLU basic split" {
    const allocator = std.testing.allocator;

    const d_model: usize = 4;
    const d_ff: usize = 8;
    const sequence_len: usize = 1;
    const batch: usize = 1;

    // Create input tensor [1, 1, 4] (3D: batch, sequence_len, d_model)
    const input_data = try allocator.alloc(f32, batch * sequence_len * d_model);
    defer allocator.free(input_data);
    @memset(input_data, 1.0);
    var input = Tensor.view3DSlice(input_data, sequence_len, d_model);

    // Create weight tensors with proper shapes for matmul:
    // input [sequence_len, d_model] @ w1 [d_model, d_ff] -> [sequence_len, d_ff]
    const w1_data = try allocator.alloc(f32, d_model * d_ff);
    defer allocator.free(w1_data);
    @memset(w1_data, 0.1);
    var w1 = Tensor.view2DSlice(w1_data, d_model, d_ff);

    // hidden [sequence_len, d_ff] @ w2 [d_ff, d_model] -> [sequence_len, d_model]
    const w2_data = try allocator.alloc(f32, d_ff * d_model);
    defer allocator.free(w2_data);
    @memset(w2_data, 0.1);
    var w2 = Tensor.view2DSlice(w2_data, d_ff, d_model);

    // input [sequence_len, d_model] @ w3 [d_model, d_ff] -> [sequence_len, d_ff]
    const w3_data = try allocator.alloc(f32, d_model * d_ff);
    defer allocator.free(w3_data);
    @memset(w3_data, 0.1);
    var w3 = Tensor.view2DSlice(w3_data, d_model, d_ff);

    // Create output tensor [1, 1, 4] (3D: batch, sequence_len, d_model)
    const output_data = try allocator.alloc(f32, batch * sequence_len * d_model);
    defer allocator.free(output_data);
    @memset(output_data, 0.0);
    var output = Tensor.view3DSlice(output_data, sequence_len, d_model);

    // Create SwiGLU layer
    const swiglu = SwiGLU{
        .d_model = d_model,
        .d_ff = d_ff,
        .use_gelu = false,
        .use_swiglu_variant = false,
        .w1 = &w1,
        .w2 = &w2,
        .w3 = &w3,
        .fused_gate_up = null,
        .allocator = allocator,
        .matmul_gate = matmul.matmulF32,
        .matmul_down = matmul.matmulF32,
    };

    var scratch = FfnScratch{};
    defer scratch.deinit(allocator);

    var matmul_scratch = try matmul.MatmulScratch.init(allocator);
    defer matmul_scratch.deinit();

    try swiglu.forward(&input, &output, &scratch, &matmul_scratch);

    // Output should be non-zero (exact value depends on matmul implementation)
    var has_nonzero = false;
    for (output_data) |val| {
        if (@abs(val) > 1e-6) has_nonzero = true;
    }
    try std.testing.expect(has_nonzero);
}

test "forward SwiGLU GELU" {
    const allocator = std.testing.allocator;

    const d_model: usize = 4;
    const d_ff: usize = 8;
    const sequence_len: usize = 1;

    // 3D tensor [batch=1, sequence_len, d_model]
    const input_data = try allocator.alloc(f32, sequence_len * d_model);
    defer allocator.free(input_data);
    @memset(input_data, 1.0);
    var input = Tensor.view3DSlice(input_data, sequence_len, d_model);

    // Weight shapes for matmul: input [seq, d_model] @ w [d_model, d_ff]
    const w1_data = try allocator.alloc(f32, d_model * d_ff);
    defer allocator.free(w1_data);
    @memset(w1_data, 0.1);
    var w1 = Tensor.view2DSlice(w1_data, d_model, d_ff);

    const w2_data = try allocator.alloc(f32, d_ff * d_model);
    defer allocator.free(w2_data);
    @memset(w2_data, 0.1);
    var w2 = Tensor.view2DSlice(w2_data, d_ff, d_model);

    const w3_data = try allocator.alloc(f32, d_model * d_ff);
    defer allocator.free(w3_data);
    @memset(w3_data, 0.1);
    var w3 = Tensor.view2DSlice(w3_data, d_model, d_ff);

    // 3D tensor [batch=1, sequence_len, d_model]
    const output_data = try allocator.alloc(f32, sequence_len * d_model);
    defer allocator.free(output_data);
    @memset(output_data, 0.0);
    var output = Tensor.view3DSlice(output_data, sequence_len, d_model);

    const swiglu = SwiGLU{
        .d_model = d_model,
        .d_ff = d_ff,
        .use_gelu = true,
        .use_swiglu_variant = false,
        .w1 = &w1,
        .w2 = &w2,
        .w3 = &w3,
        .fused_gate_up = null,
        .allocator = allocator,
        .matmul_gate = matmul.matmulF32,
        .matmul_down = matmul.matmulF32,
    };

    var scratch = FfnScratch{};
    defer scratch.deinit(allocator);

    var matmul_scratch = try matmul.MatmulScratch.init(allocator);
    defer matmul_scratch.deinit();

    try swiglu.forward(&input, &output, &scratch, &matmul_scratch);

    var has_nonzero = false;
    for (output_data) |val| {
        if (@abs(val) > 1e-6) has_nonzero = true;
    }
    try std.testing.expect(has_nonzero);
}

test "forward SwiGLU variant" {
    const allocator = std.testing.allocator;

    const d_model: usize = 4;
    const d_ff: usize = 8;
    const sequence_len: usize = 1;

    // 3D tensor [batch=1, sequence_len, d_model]
    const input_data = try allocator.alloc(f32, sequence_len * d_model);
    defer allocator.free(input_data);
    @memset(input_data, 1.0);
    var input = Tensor.view3DSlice(input_data, sequence_len, d_model);

    // Weight shapes for matmul: input [seq, d_model] @ w [d_model, d_ff]
    const w1_data = try allocator.alloc(f32, d_model * d_ff);
    defer allocator.free(w1_data);
    @memset(w1_data, 0.1);
    var w1 = Tensor.view2DSlice(w1_data, d_model, d_ff);

    const w2_data = try allocator.alloc(f32, d_ff * d_model);
    defer allocator.free(w2_data);
    @memset(w2_data, 0.1);
    var w2 = Tensor.view2DSlice(w2_data, d_ff, d_model);

    const w3_data = try allocator.alloc(f32, d_model * d_ff);
    defer allocator.free(w3_data);
    @memset(w3_data, 0.1);
    var w3 = Tensor.view2DSlice(w3_data, d_model, d_ff);

    // 3D tensor [batch=1, sequence_len, d_model]
    const output_data = try allocator.alloc(f32, sequence_len * d_model);
    defer allocator.free(output_data);
    @memset(output_data, 0.0);
    var output = Tensor.view3DSlice(output_data, sequence_len, d_model);

    const swiglu = SwiGLU{
        .d_model = d_model,
        .d_ff = d_ff,
        .use_gelu = false,
        .use_swiglu_variant = true,
        .w1 = &w1,
        .w2 = &w2,
        .w3 = &w3,
        .fused_gate_up = null,
        .allocator = allocator,
        .matmul_gate = matmul.matmulF32,
        .matmul_down = matmul.matmulF32,
    };

    var scratch = FfnScratch{};
    defer scratch.deinit(allocator);

    var matmul_scratch = try matmul.MatmulScratch.init(allocator);
    defer matmul_scratch.deinit();

    try swiglu.forward(&input, &output, &scratch, &matmul_scratch);

    var has_nonzero = false;
    for (output_data) |val| {
        if (@abs(val) > 1e-6) has_nonzero = true;
    }
    try std.testing.expect(has_nonzero);
}

test "forward SwiGLU fused concat" {
    const allocator = std.testing.allocator;

    const d_model: usize = 4;
    const d_ff: usize = 8;
    const sequence_len: usize = 1;

    // 3D tensor [batch=1, sequence_len, d_model]
    const input_data = try allocator.alloc(f32, sequence_len * d_model);
    defer allocator.free(input_data);
    @memset(input_data, 1.0);
    var input = Tensor.view3DSlice(input_data, sequence_len, d_model);

    // Fused gate_up weight [d_model, 2*d_ff] for matmul
    const fused_data = try allocator.alloc(f32, d_model * 2 * d_ff);
    defer allocator.free(fused_data);
    @memset(fused_data, 0.1);
    const fused = Tensor.view2DSlice(fused_data, d_model, 2 * d_ff);

    const w2_data = try allocator.alloc(f32, d_ff * d_model);
    defer allocator.free(w2_data);
    @memset(w2_data, 0.1);
    var w2 = Tensor.view2DSlice(w2_data, d_ff, d_model);

    // 3D tensor [batch=1, sequence_len, d_model]
    const output_data = try allocator.alloc(f32, sequence_len * d_model);
    defer allocator.free(output_data);
    @memset(output_data, 0.0);
    var output = Tensor.view3DSlice(output_data, sequence_len, d_model);

    const swiglu = SwiGLU{
        .d_model = d_model,
        .d_ff = d_ff,
        .use_gelu = false,
        .use_swiglu_variant = false,
        .w1 = null,
        .w2 = &w2,
        .w3 = null,
        .fused_gate_up = fused,
        .fused_gate_up_layout = .concat,
        .allocator = allocator,
        .matmul_gate = matmul.matmulF32,
        .matmul_down = matmul.matmulF32,
    };

    var scratch = FfnScratch{};
    defer scratch.deinit(allocator);

    var matmul_scratch = try matmul.MatmulScratch.init(allocator);
    defer matmul_scratch.deinit();

    try swiglu.forward(&input, &output, &scratch, &matmul_scratch);

    var has_nonzero = false;
    for (output_data) |val| {
        if (@abs(val) > 1e-6) has_nonzero = true;
    }
    try std.testing.expect(has_nonzero);
}

test "forward SwiGLU fused interleaved" {
    const allocator = std.testing.allocator;

    const d_model: usize = 4;
    const d_ff: usize = 8;
    const sequence_len: usize = 1;

    // 3D tensor [batch=1, sequence_len, d_model]
    const input_data = try allocator.alloc(f32, sequence_len * d_model);
    defer allocator.free(input_data);
    @memset(input_data, 1.0);
    var input = Tensor.view3DSlice(input_data, sequence_len, d_model);

    // Fused gate_up weight [d_model, 2*d_ff] for matmul
    const fused_data = try allocator.alloc(f32, d_model * 2 * d_ff);
    defer allocator.free(fused_data);
    @memset(fused_data, 0.1);
    const fused = Tensor.view2DSlice(fused_data, d_model, 2 * d_ff);

    const w2_data = try allocator.alloc(f32, d_ff * d_model);
    defer allocator.free(w2_data);
    @memset(w2_data, 0.1);
    var w2 = Tensor.view2DSlice(w2_data, d_ff, d_model);

    // 3D tensor [batch=1, sequence_len, d_model]
    const output_data = try allocator.alloc(f32, sequence_len * d_model);
    defer allocator.free(output_data);
    @memset(output_data, 0.0);
    var output = Tensor.view3DSlice(output_data, sequence_len, d_model);

    const swiglu = SwiGLU{
        .d_model = d_model,
        .d_ff = d_ff,
        .use_gelu = false,
        .use_swiglu_variant = false,
        .w1 = null,
        .w2 = &w2,
        .w3 = null,
        .fused_gate_up = fused,
        .fused_gate_up_layout = .interleaved,
        .allocator = allocator,
        .matmul_gate = matmul.matmulF32,
        .matmul_down = matmul.matmulF32,
    };

    var scratch = FfnScratch{};
    defer scratch.deinit(allocator);

    var matmul_scratch = try matmul.MatmulScratch.init(allocator);
    defer matmul_scratch.deinit();

    try swiglu.forward(&input, &output, &scratch, &matmul_scratch);

    var has_nonzero = false;
    for (output_data) |val| {
        if (@abs(val) > 1e-6) has_nonzero = true;
    }
    try std.testing.expect(has_nonzero);
}

test "forward SwiGLU multi-token" {
    const allocator = std.testing.allocator;

    const d_model: usize = 4;
    const d_ff: usize = 8;
    const sequence_len: usize = 3; // Multi-token test

    // 3D tensor [batch=1, sequence_len, d_model]
    const input_data = try allocator.alloc(f32, sequence_len * d_model);
    defer allocator.free(input_data);
    // Different values for each token
    for (0..sequence_len) |i| {
        for (0..d_model) |j| {
            input_data[i * d_model + j] = @as(f32, @floatFromInt(i + 1)) * 0.1;
        }
    }
    var input = Tensor.view3DSlice(input_data, sequence_len, d_model);

    // Weight shapes for matmul: input [seq, d_model] @ w [d_model, d_ff]
    const w1_data = try allocator.alloc(f32, d_model * d_ff);
    defer allocator.free(w1_data);
    @memset(w1_data, 0.1);
    var w1 = Tensor.view2DSlice(w1_data, d_model, d_ff);

    const w2_data = try allocator.alloc(f32, d_ff * d_model);
    defer allocator.free(w2_data);
    @memset(w2_data, 0.1);
    var w2 = Tensor.view2DSlice(w2_data, d_ff, d_model);

    const w3_data = try allocator.alloc(f32, d_model * d_ff);
    defer allocator.free(w3_data);
    @memset(w3_data, 0.1);
    var w3 = Tensor.view2DSlice(w3_data, d_model, d_ff);

    // 3D tensor [batch=1, sequence_len, d_model]
    const output_data = try allocator.alloc(f32, sequence_len * d_model);
    defer allocator.free(output_data);
    @memset(output_data, 0.0);
    var output = Tensor.view3DSlice(output_data, sequence_len, d_model);

    const swiglu = SwiGLU{
        .d_model = d_model,
        .d_ff = d_ff,
        .use_gelu = false,
        .use_swiglu_variant = false,
        .w1 = &w1,
        .w2 = &w2,
        .w3 = &w3,
        .fused_gate_up = null,
        .allocator = allocator,
        .matmul_gate = matmul.matmulF32,
        .matmul_down = matmul.matmulF32,
    };

    var scratch = FfnScratch{};
    defer scratch.deinit(allocator);

    var matmul_scratch = try matmul.MatmulScratch.init(allocator);
    defer matmul_scratch.deinit();

    try swiglu.forward(&input, &output, &scratch, &matmul_scratch);

    // All output tokens should have non-zero values
    for (0..sequence_len) |i| {
        var has_nonzero = false;
        for (0..d_model) |j| {
            if (@abs(output_data[i * d_model + j]) > 1e-6) has_nonzero = true;
        }
        try std.testing.expect(has_nonzero);
    }
}

test "forward SwiGLU scratch reuse" {
    const allocator = std.testing.allocator;

    const d_model: usize = 4;
    const d_ff: usize = 8;
    const sequence_len: usize = 1;

    // 3D tensor [batch=1, sequence_len, d_model]
    const input_data = try allocator.alloc(f32, sequence_len * d_model);
    defer allocator.free(input_data);
    @memset(input_data, 1.0);
    var input = Tensor.view3DSlice(input_data, sequence_len, d_model);

    // Weight shapes for matmul: input [seq, d_model] @ w [d_model, d_ff]
    const w1_data = try allocator.alloc(f32, d_model * d_ff);
    defer allocator.free(w1_data);
    @memset(w1_data, 0.1);
    var w1 = Tensor.view2DSlice(w1_data, d_model, d_ff);

    const w2_data = try allocator.alloc(f32, d_ff * d_model);
    defer allocator.free(w2_data);
    @memset(w2_data, 0.1);
    var w2 = Tensor.view2DSlice(w2_data, d_ff, d_model);

    const w3_data = try allocator.alloc(f32, d_model * d_ff);
    defer allocator.free(w3_data);
    @memset(w3_data, 0.1);
    var w3 = Tensor.view2DSlice(w3_data, d_model, d_ff);

    // 3D tensor [batch=1, sequence_len, d_model]
    const output_data = try allocator.alloc(f32, sequence_len * d_model);
    defer allocator.free(output_data);

    const swiglu = SwiGLU{
        .d_model = d_model,
        .d_ff = d_ff,
        .use_gelu = false,
        .use_swiglu_variant = false,
        .w1 = &w1,
        .w2 = &w2,
        .w3 = &w3,
        .fused_gate_up = null,
        .allocator = allocator,
        .matmul_gate = matmul.matmulF32,
        .matmul_down = matmul.matmulF32,
    };

    var scratch = FfnScratch{};
    defer scratch.deinit(allocator);

    var matmul_scratch = try matmul.MatmulScratch.init(allocator);
    defer matmul_scratch.deinit();

    // Run forward twice to test scratch buffer reuse
    var output1 = Tensor.view3DSlice(output_data, sequence_len, d_model);
    try swiglu.forward(&input, &output1, &scratch, &matmul_scratch);

    const gate_len_first = scratch.gate.len;

    @memset(output_data, 0.0);
    var output2 = Tensor.view3DSlice(output_data, sequence_len, d_model);
    try swiglu.forward(&input, &output2, &scratch, &matmul_scratch);

    // Scratch buffers should be reused (same length)
    try std.testing.expectEqual(gate_len_first, scratch.gate.len);
}

test "FfnScratch deinit cleanup" {
    const allocator = std.testing.allocator;

    var scratch = FfnScratch{};

    // Allocate some buffers
    try cpu_common.ensureF32Slice(allocator, &scratch.gate, 100);
    try cpu_common.ensureF32Slice(allocator, &scratch.gate_act, 100);
    try cpu_common.ensureF32Slice(allocator, &scratch.up, 100);
    try cpu_common.ensureF32Slice(allocator, &scratch.hidden, 100);

    try std.testing.expect(scratch.gate.len == 100);
    try std.testing.expect(scratch.gate_act.len == 100);
    try std.testing.expect(scratch.up.len == 100);
    try std.testing.expect(scratch.hidden.len == 100);

    scratch.deinit(allocator);

    try std.testing.expect(scratch.gate.len == 0);
    try std.testing.expect(scratch.gate_act.len == 0);
    try std.testing.expect(scratch.up.len == 0);
    try std.testing.expect(scratch.hidden.len == 0);
}

test "SwiGLU.forward ensureSlice buffer growth" {
    const allocator = std.testing.allocator;

    var storage: []f32 = &.{};

    try cpu_common.ensureF32Slice(allocator, &storage, 10);
    try std.testing.expectEqual(@as(usize, 10), storage.len);

    // Request larger size - should reallocate
    try cpu_common.ensureF32Slice(allocator, &storage, 20);
    try std.testing.expectEqual(@as(usize, 20), storage.len);

    // Request smaller size - should keep existing buffer
    try cpu_common.ensureF32Slice(allocator, &storage, 15);
    try std.testing.expectEqual(@as(usize, 20), storage.len);

    allocator.free(storage);
}

test "forward numerical stability" {
    const allocator = std.testing.allocator;

    const d_model: usize = 4;
    const d_ff: usize = 8;
    const sequence_len: usize = 1;

    // 3D tensor [batch=1, sequence_len, d_model]
    const input_data = try allocator.alloc(f32, sequence_len * d_model);
    defer allocator.free(input_data);
    // Mix of values including zeros
    input_data[0] = 0.0;
    input_data[1] = 1.0;
    input_data[2] = -1.0;
    input_data[3] = 0.5;
    var input = Tensor.view3DSlice(input_data, sequence_len, d_model);

    // Weight shapes for matmul: input [seq, d_model] @ w [d_model, d_ff]
    const w1_data = try allocator.alloc(f32, d_model * d_ff);
    defer allocator.free(w1_data);
    @memset(w1_data, 0.1);
    var w1 = Tensor.view2DSlice(w1_data, d_model, d_ff);

    const w2_data = try allocator.alloc(f32, d_ff * d_model);
    defer allocator.free(w2_data);
    @memset(w2_data, 0.1);
    var w2 = Tensor.view2DSlice(w2_data, d_ff, d_model);

    const w3_data = try allocator.alloc(f32, d_model * d_ff);
    defer allocator.free(w3_data);
    @memset(w3_data, 0.1);
    var w3 = Tensor.view2DSlice(w3_data, d_model, d_ff);

    // 3D tensor [batch=1, sequence_len, d_model]
    const output_data = try allocator.alloc(f32, sequence_len * d_model);
    defer allocator.free(output_data);
    @memset(output_data, 0.0);
    var output = Tensor.view3DSlice(output_data, sequence_len, d_model);

    const swiglu = SwiGLU{
        .d_model = d_model,
        .d_ff = d_ff,
        .use_gelu = false,
        .use_swiglu_variant = false,
        .w1 = &w1,
        .w2 = &w2,
        .w3 = &w3,
        .fused_gate_up = null,
        .allocator = allocator,
        .matmul_gate = matmul.matmulF32,
        .matmul_down = matmul.matmulF32,
    };

    var scratch = FfnScratch{};
    defer scratch.deinit(allocator);

    var matmul_scratch = try matmul.MatmulScratch.init(allocator);
    defer matmul_scratch.deinit();

    try swiglu.forward(&input, &output, &scratch, &matmul_scratch);

    // Check for NaN or Inf
    for (output_data) |val| {
        try std.testing.expect(!std.math.isNan(val));
        try std.testing.expect(!std.math.isInf(val));
    }
}

test "SwiGLU forward - known weights produce expected output" {
    const allocator = std.testing.allocator;

    const d_model: usize = 2;
    const d_ff: usize = 2;
    const sequence_len: usize = 1;

    // Simple input [1, 1, 2]
    const input_data = try allocator.alloc(f32, sequence_len * d_model);
    defer allocator.free(input_data);
    input_data[0] = 1.0;
    input_data[1] = 0.0;
    var input = Tensor.view3DSlice(input_data, sequence_len, d_model);

    // Identity-like weights for predictable computation
    // w1: [d_model, d_ff] = [[1, 0], [0, 1]]
    const w1_data = try allocator.alloc(f32, d_model * d_ff);
    defer allocator.free(w1_data);
    w1_data[0] = 1.0;
    w1_data[1] = 0.0;
    w1_data[2] = 0.0;
    w1_data[3] = 1.0;
    var w1 = Tensor.view2DSlice(w1_data, d_model, d_ff);

    // w3: [d_model, d_ff] = [[1, 0], [0, 1]]
    const w3_data = try allocator.alloc(f32, d_model * d_ff);
    defer allocator.free(w3_data);
    w3_data[0] = 1.0;
    w3_data[1] = 0.0;
    w3_data[2] = 0.0;
    w3_data[3] = 1.0;
    var w3 = Tensor.view2DSlice(w3_data, d_model, d_ff);

    // w2: [d_ff, d_model] = [[0.5, 0.5], [0.5, 0.5]]
    const w2_data = try allocator.alloc(f32, d_ff * d_model);
    defer allocator.free(w2_data);
    @memset(w2_data, 0.5);
    var w2 = Tensor.view2DSlice(w2_data, d_ff, d_model);

    const output_data = try allocator.alloc(f32, sequence_len * d_model);
    defer allocator.free(output_data);
    @memset(output_data, 0.0);
    var output = Tensor.view3DSlice(output_data, sequence_len, d_model);

    const swiglu = SwiGLU{
        .d_model = d_model,
        .d_ff = d_ff,
        .use_gelu = false,
        .use_swiglu_variant = false,
        .w1 = &w1,
        .w2 = &w2,
        .w3 = &w3,
        .fused_gate_up = null,
        .allocator = allocator,
        .matmul_gate = matmul.matmulF32,
        .matmul_down = matmul.matmulF32,
    };

    var scratch = FfnScratch{};
    defer scratch.deinit(allocator);

    var matmul_scratch = try matmul.MatmulScratch.init(allocator);
    defer matmul_scratch.deinit();

    try swiglu.forward(&input, &output, &scratch, &matmul_scratch);

    // Output should be non-zero and finite
    for (output_data) |val| {
        try std.testing.expect(!std.math.isNan(val));
        try std.testing.expect(!std.math.isInf(val));
    }
}

test "SwiGLU forward - different hidden dimensions" {
    const allocator = std.testing.allocator;
    const sequence_len: usize = 1;

    // Test with various d_ff sizes
    const test_cases = [_]struct { d_model: usize, d_ff: usize }{
        .{ .d_model = 2, .d_ff = 4 },
        .{ .d_model = 4, .d_ff = 16 },
        .{ .d_model = 8, .d_ff = 32 },
        .{ .d_model = 16, .d_ff = 64 },
    };

    for (test_cases) |tc| {
        const input_data = try allocator.alloc(f32, sequence_len * tc.d_model);
        defer allocator.free(input_data);
        @memset(input_data, 0.5);
        var input = Tensor.view3DSlice(input_data, sequence_len, tc.d_model);

        const w1_data = try allocator.alloc(f32, tc.d_model * tc.d_ff);
        defer allocator.free(w1_data);
        @memset(w1_data, 0.1);
        var w1 = Tensor.view2DSlice(w1_data, tc.d_model, tc.d_ff);

        const w2_data = try allocator.alloc(f32, tc.d_ff * tc.d_model);
        defer allocator.free(w2_data);
        @memset(w2_data, 0.1);
        var w2 = Tensor.view2DSlice(w2_data, tc.d_ff, tc.d_model);

        const w3_data = try allocator.alloc(f32, tc.d_model * tc.d_ff);
        defer allocator.free(w3_data);
        @memset(w3_data, 0.1);
        var w3 = Tensor.view2DSlice(w3_data, tc.d_model, tc.d_ff);

        const output_data = try allocator.alloc(f32, sequence_len * tc.d_model);
        defer allocator.free(output_data);
        @memset(output_data, 0.0);
        var output = Tensor.view3DSlice(output_data, sequence_len, tc.d_model);

        const swiglu = SwiGLU{
            .d_model = tc.d_model,
            .d_ff = tc.d_ff,
            .use_gelu = false,
            .use_swiglu_variant = false,
            .w1 = &w1,
            .w2 = &w2,
            .w3 = &w3,
            .fused_gate_up = null,
            .allocator = allocator,
            .matmul_gate = matmul.matmulF32,
            .matmul_down = matmul.matmulF32,
        };

        var scratch = FfnScratch{};
        defer scratch.deinit(allocator);

        var matmul_scratch = try matmul.MatmulScratch.init(allocator);
        defer matmul_scratch.deinit();

        try swiglu.forward(&input, &output, &scratch, &matmul_scratch);

        // Verify output is valid
        var has_nonzero = false;
        for (output_data) |val| {
            try std.testing.expect(!std.math.isNan(val));
            try std.testing.expect(!std.math.isInf(val));
            if (@abs(val) > 1e-6) has_nonzero = true;
        }
        try std.testing.expect(has_nonzero);
    }
}

test "SwiGLU forward - large sequence length" {
    const allocator = std.testing.allocator;

    const d_model: usize = 4;
    const d_ff: usize = 8;
    const sequence_len: usize = 16; // Longer sequence

    const input_data = try allocator.alloc(f32, sequence_len * d_model);
    defer allocator.free(input_data);
    // Varied input across sequence
    for (0..sequence_len) |i| {
        for (0..d_model) |j| {
            input_data[i * d_model + j] = @as(f32, @floatFromInt((i + j) % 3)) * 0.3;
        }
    }
    var input = Tensor.view3DSlice(input_data, sequence_len, d_model);

    const w1_data = try allocator.alloc(f32, d_model * d_ff);
    defer allocator.free(w1_data);
    @memset(w1_data, 0.1);
    var w1 = Tensor.view2DSlice(w1_data, d_model, d_ff);

    const w2_data = try allocator.alloc(f32, d_ff * d_model);
    defer allocator.free(w2_data);
    @memset(w2_data, 0.1);
    var w2 = Tensor.view2DSlice(w2_data, d_ff, d_model);

    const w3_data = try allocator.alloc(f32, d_model * d_ff);
    defer allocator.free(w3_data);
    @memset(w3_data, 0.1);
    var w3 = Tensor.view2DSlice(w3_data, d_model, d_ff);

    const output_data = try allocator.alloc(f32, sequence_len * d_model);
    defer allocator.free(output_data);
    @memset(output_data, 0.0);
    var output = Tensor.view3DSlice(output_data, sequence_len, d_model);

    const swiglu = SwiGLU{
        .d_model = d_model,
        .d_ff = d_ff,
        .use_gelu = false,
        .use_swiglu_variant = false,
        .w1 = &w1,
        .w2 = &w2,
        .w3 = &w3,
        .fused_gate_up = null,
        .allocator = allocator,
        .matmul_gate = matmul.matmulF32,
        .matmul_down = matmul.matmulF32,
    };

    var scratch = FfnScratch{};
    defer scratch.deinit(allocator);

    var matmul_scratch = try matmul.MatmulScratch.init(allocator);
    defer matmul_scratch.deinit();

    try swiglu.forward(&input, &output, &scratch, &matmul_scratch);

    // Check all sequence positions
    for (0..sequence_len) |i| {
        var has_nonzero = false;
        for (0..d_model) |j| {
            const val = output_data[i * d_model + j];
            try std.testing.expect(!std.math.isNan(val));
            try std.testing.expect(!std.math.isInf(val));
            if (@abs(val) > 1e-6) has_nonzero = true;
        }
        try std.testing.expect(has_nonzero);
    }
}

test "SwiGLU forward - missing weights error" {
    const allocator = std.testing.allocator;

    const d_model: usize = 4;
    const d_ff: usize = 8;
    const sequence_len: usize = 1;

    const input_data = try allocator.alloc(f32, sequence_len * d_model);
    defer allocator.free(input_data);
    @memset(input_data, 1.0);
    var input = Tensor.view3DSlice(input_data, sequence_len, d_model);

    const w2_data = try allocator.alloc(f32, d_ff * d_model);
    defer allocator.free(w2_data);
    @memset(w2_data, 0.1);
    var w2 = Tensor.view2DSlice(w2_data, d_ff, d_model);

    const output_data = try allocator.alloc(f32, sequence_len * d_model);
    defer allocator.free(output_data);
    @memset(output_data, 0.0);
    var output = Tensor.view3DSlice(output_data, sequence_len, d_model);

    // Create SwiGLU with missing w1 and w3 (no fused_gate_up)
    const swiglu = SwiGLU{
        .d_model = d_model,
        .d_ff = d_ff,
        .use_gelu = false,
        .use_swiglu_variant = false,
        .w1 = null, // Missing!
        .w2 = &w2,
        .w3 = null, // Missing!
        .fused_gate_up = null,
        .allocator = allocator,
        .matmul_gate = matmul.matmulF32,
        .matmul_down = matmul.matmulF32,
    };

    var scratch = FfnScratch{};
    defer scratch.deinit(allocator);

    var matmul_scratch = try matmul.MatmulScratch.init(allocator);
    defer matmul_scratch.deinit();

    // Should return error.MissingFFNWeights
    const result = swiglu.forward(&input, &output, &scratch, &matmul_scratch);
    try std.testing.expectError(error.MissingFFNWeights, result);
}

test "SwiGLU forward - fused concat with GELU" {
    const allocator = std.testing.allocator;

    const d_model: usize = 4;
    const d_ff: usize = 8;
    const sequence_len: usize = 1;

    const input_data = try allocator.alloc(f32, sequence_len * d_model);
    defer allocator.free(input_data);
    @memset(input_data, 1.0);
    var input = Tensor.view3DSlice(input_data, sequence_len, d_model);

    const fused_data = try allocator.alloc(f32, d_model * 2 * d_ff);
    defer allocator.free(fused_data);
    @memset(fused_data, 0.1);
    const fused = Tensor.view2DSlice(fused_data, d_model, 2 * d_ff);

    const w2_data = try allocator.alloc(f32, d_ff * d_model);
    defer allocator.free(w2_data);
    @memset(w2_data, 0.1);
    var w2 = Tensor.view2DSlice(w2_data, d_ff, d_model);

    const output_data = try allocator.alloc(f32, sequence_len * d_model);
    defer allocator.free(output_data);
    @memset(output_data, 0.0);
    var output = Tensor.view3DSlice(output_data, sequence_len, d_model);

    const swiglu = SwiGLU{
        .d_model = d_model,
        .d_ff = d_ff,
        .use_gelu = true,
        .use_swiglu_variant = false,
        .w1 = null,
        .w2 = &w2,
        .w3 = null,
        .fused_gate_up = fused,
        .fused_gate_up_layout = .concat,
        .allocator = allocator,
        .matmul_gate = matmul.matmulF32,
        .matmul_down = matmul.matmulF32,
    };

    var scratch = FfnScratch{};
    defer scratch.deinit(allocator);

    var matmul_scratch = try matmul.MatmulScratch.init(allocator);
    defer matmul_scratch.deinit();

    try swiglu.forward(&input, &output, &scratch, &matmul_scratch);

    var has_nonzero = false;
    for (output_data) |val| {
        try std.testing.expect(!std.math.isNan(val));
        try std.testing.expect(!std.math.isInf(val));
        if (@abs(val) > 1e-6) has_nonzero = true;
    }
    try std.testing.expect(has_nonzero);
}

test "SwiGLU forward - fused interleaved with variant" {
    const allocator = std.testing.allocator;

    const d_model: usize = 4;
    const d_ff: usize = 8;
    const sequence_len: usize = 1;

    const input_data = try allocator.alloc(f32, sequence_len * d_model);
    defer allocator.free(input_data);
    @memset(input_data, 1.0);
    var input = Tensor.view3DSlice(input_data, sequence_len, d_model);

    const fused_data = try allocator.alloc(f32, d_model * 2 * d_ff);
    defer allocator.free(fused_data);
    @memset(fused_data, 0.1);
    const fused = Tensor.view2DSlice(fused_data, d_model, 2 * d_ff);

    const w2_data = try allocator.alloc(f32, d_ff * d_model);
    defer allocator.free(w2_data);
    @memset(w2_data, 0.1);
    var w2 = Tensor.view2DSlice(w2_data, d_ff, d_model);

    const output_data = try allocator.alloc(f32, sequence_len * d_model);
    defer allocator.free(output_data);
    @memset(output_data, 0.0);
    var output = Tensor.view3DSlice(output_data, sequence_len, d_model);

    const swiglu = SwiGLU{
        .d_model = d_model,
        .d_ff = d_ff,
        .use_gelu = false,
        .use_swiglu_variant = true,
        .w1 = null,
        .w2 = &w2,
        .w3 = null,
        .fused_gate_up = fused,
        .fused_gate_up_layout = .interleaved,
        .allocator = allocator,
        .matmul_gate = matmul.matmulF32,
        .matmul_down = matmul.matmulF32,
    };

    var scratch = FfnScratch{};
    defer scratch.deinit(allocator);

    var matmul_scratch = try matmul.MatmulScratch.init(allocator);
    defer matmul_scratch.deinit();

    try swiglu.forward(&input, &output, &scratch, &matmul_scratch);

    var has_nonzero = false;
    for (output_data) |val| {
        try std.testing.expect(!std.math.isNan(val));
        try std.testing.expect(!std.math.isInf(val));
        if (@abs(val) > 1e-6) has_nonzero = true;
    }
    try std.testing.expect(has_nonzero);
}

test "SwiGLU forward - fused concat multi-token" {
    const allocator = std.testing.allocator;

    const d_model: usize = 4;
    const d_ff: usize = 8;
    const sequence_len: usize = 5; // Multi-token with fused weights

    const input_data = try allocator.alloc(f32, sequence_len * d_model);
    defer allocator.free(input_data);
    for (0..sequence_len) |i| {
        for (0..d_model) |j| {
            input_data[i * d_model + j] = @as(f32, @floatFromInt(i + 1)) * 0.2;
        }
    }
    var input = Tensor.view3DSlice(input_data, sequence_len, d_model);

    const fused_data = try allocator.alloc(f32, d_model * 2 * d_ff);
    defer allocator.free(fused_data);
    @memset(fused_data, 0.1);
    const fused = Tensor.view2DSlice(fused_data, d_model, 2 * d_ff);

    const w2_data = try allocator.alloc(f32, d_ff * d_model);
    defer allocator.free(w2_data);
    @memset(w2_data, 0.1);
    var w2 = Tensor.view2DSlice(w2_data, d_ff, d_model);

    const output_data = try allocator.alloc(f32, sequence_len * d_model);
    defer allocator.free(output_data);
    @memset(output_data, 0.0);
    var output = Tensor.view3DSlice(output_data, sequence_len, d_model);

    const swiglu = SwiGLU{
        .d_model = d_model,
        .d_ff = d_ff,
        .use_gelu = false,
        .use_swiglu_variant = false,
        .w1 = null,
        .w2 = &w2,
        .w3 = null,
        .fused_gate_up = fused,
        .fused_gate_up_layout = .concat,
        .allocator = allocator,
        .matmul_gate = matmul.matmulF32,
        .matmul_down = matmul.matmulF32,
    };

    var scratch = FfnScratch{};
    defer scratch.deinit(allocator);

    var matmul_scratch = try matmul.MatmulScratch.init(allocator);
    defer matmul_scratch.deinit();

    try swiglu.forward(&input, &output, &scratch, &matmul_scratch);

    // Check all tokens have valid outputs
    for (0..sequence_len) |i| {
        var has_nonzero = false;
        for (0..d_model) |j| {
            const val = output_data[i * d_model + j];
            try std.testing.expect(!std.math.isNan(val));
            try std.testing.expect(!std.math.isInf(val));
            if (@abs(val) > 1e-6) has_nonzero = true;
        }
        try std.testing.expect(has_nonzero);
    }
}

test "SwiGLU forward - zero input produces valid output" {
    const allocator = std.testing.allocator;

    const d_model: usize = 4;
    const d_ff: usize = 8;
    const sequence_len: usize = 1;

    // All zeros input
    const input_data = try allocator.alloc(f32, sequence_len * d_model);
    defer allocator.free(input_data);
    @memset(input_data, 0.0);
    var input = Tensor.view3DSlice(input_data, sequence_len, d_model);

    const w1_data = try allocator.alloc(f32, d_model * d_ff);
    defer allocator.free(w1_data);
    @memset(w1_data, 0.1);
    var w1 = Tensor.view2DSlice(w1_data, d_model, d_ff);

    const w2_data = try allocator.alloc(f32, d_ff * d_model);
    defer allocator.free(w2_data);
    @memset(w2_data, 0.1);
    var w2 = Tensor.view2DSlice(w2_data, d_ff, d_model);

    const w3_data = try allocator.alloc(f32, d_model * d_ff);
    defer allocator.free(w3_data);
    @memset(w3_data, 0.1);
    var w3 = Tensor.view2DSlice(w3_data, d_model, d_ff);

    const output_data = try allocator.alloc(f32, sequence_len * d_model);
    defer allocator.free(output_data);
    @memset(output_data, 0.0);
    var output = Tensor.view3DSlice(output_data, sequence_len, d_model);

    const swiglu = SwiGLU{
        .d_model = d_model,
        .d_ff = d_ff,
        .use_gelu = false,
        .use_swiglu_variant = false,
        .w1 = &w1,
        .w2 = &w2,
        .w3 = &w3,
        .fused_gate_up = null,
        .allocator = allocator,
        .matmul_gate = matmul.matmulF32,
        .matmul_down = matmul.matmulF32,
    };

    var scratch = FfnScratch{};
    defer scratch.deinit(allocator);

    var matmul_scratch = try matmul.MatmulScratch.init(allocator);
    defer matmul_scratch.deinit();

    try swiglu.forward(&input, &output, &scratch, &matmul_scratch);

    // Output should be valid (likely zeros, but no NaN/Inf)
    for (output_data) |val| {
        try std.testing.expect(!std.math.isNan(val));
        try std.testing.expect(!std.math.isInf(val));
    }
}

test "SwiGLU forward - extreme weight values" {
    const allocator = std.testing.allocator;

    const d_model: usize = 4;
    const d_ff: usize = 8;
    const sequence_len: usize = 1;

    const input_data = try allocator.alloc(f32, sequence_len * d_model);
    defer allocator.free(input_data);
    @memset(input_data, 0.5);
    var input = Tensor.view3DSlice(input_data, sequence_len, d_model);

    // Very small weights
    const w1_data = try allocator.alloc(f32, d_model * d_ff);
    defer allocator.free(w1_data);
    @memset(w1_data, 0.0001);
    var w1 = Tensor.view2DSlice(w1_data, d_model, d_ff);

    // Very large weights
    const w2_data = try allocator.alloc(f32, d_ff * d_model);
    defer allocator.free(w2_data);
    @memset(w2_data, 10.0);
    var w2 = Tensor.view2DSlice(w2_data, d_ff, d_model);

    const w3_data = try allocator.alloc(f32, d_model * d_ff);
    defer allocator.free(w3_data);
    @memset(w3_data, 0.0001);
    var w3 = Tensor.view2DSlice(w3_data, d_model, d_ff);

    const output_data = try allocator.alloc(f32, sequence_len * d_model);
    defer allocator.free(output_data);
    @memset(output_data, 0.0);
    var output = Tensor.view3DSlice(output_data, sequence_len, d_model);

    const swiglu = SwiGLU{
        .d_model = d_model,
        .d_ff = d_ff,
        .use_gelu = false,
        .use_swiglu_variant = false,
        .w1 = &w1,
        .w2 = &w2,
        .w3 = &w3,
        .fused_gate_up = null,
        .allocator = allocator,
        .matmul_gate = matmul.matmulF32,
        .matmul_down = matmul.matmulF32,
    };

    var scratch = FfnScratch{};
    defer scratch.deinit(allocator);

    var matmul_scratch = try matmul.MatmulScratch.init(allocator);
    defer matmul_scratch.deinit();

    try swiglu.forward(&input, &output, &scratch, &matmul_scratch);

    // Should handle extreme values gracefully
    for (output_data) |val| {
        try std.testing.expect(!std.math.isNan(val));
        try std.testing.expect(!std.math.isInf(val));
    }
}
