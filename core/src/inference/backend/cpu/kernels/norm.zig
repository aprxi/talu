//! CPU Normalization Kernels
//! RMS Normalization implementation
//!
//! This module provides normalization operations for CPU inference.

pub const supported = true;

const std = @import("std");
const build_options = @import("build_options");
const tensor = @import("../../../../tensor.zig");
const compute = @import("../../../../compute/root.zig");
const math = compute.cpu.math;
const inspect = @import("../../../../xray/root.zig");
const trace = inspect.trace;
const dump = if (build_options.dump_tensors) @import("../../../../xray/dump/capture.zig") else struct {
    pub fn recordGlobal(_: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) void {}
};

const Tensor = tensor.Tensor;

/// RMS Normalization configuration.
pub const RMSNorm = struct {
    /// Canonical kernel-call contract for backend parity checks.
    pub const ForwardParams = struct {
        input: *const Tensor,
        output: *Tensor,
    };

    weight: *const Tensor,
    dim: usize,
    eps: f32,
    /// Offset added to weights before scaling (for (1+w) style norms)
    weight_offset: f32 = 0.0,
    /// Layer index for trace emissions.
    layer_idx: u16 = trace.TraceEmission.NO_LAYER,
    /// Which trace point this norm corresponds to (e.g., layer_attn_norm or layer_ffn_norm).
    trace_point: trace.TracePoint = .layer_attn_norm,

    /// Apply RMS normalization: output = x * rsqrt(mean(x²) + eps) * weight
    pub fn forward(self: *const RMSNorm, input: *const Tensor, output: *Tensor) void {
        rmsnormForward(self, input, output);
        const seq_len: u32 = @intCast(output.shape[1]);
        if (trace.isEnabled()) {
            trace.emit(
                self.trace_point,
                self.layer_idx,
                0,
                seq_len,
                output.data().ptr,
                .f32,
                .{ @intCast(output.shape[0]), seq_len, @intCast(output.shape[2]), 0 },
                3,
                "rmsnormForward",
            );
        }
        // Dump capture (compiled in only for dump binary)
        if (build_options.dump_tensors) {
            const shape = [4]usize{ @intCast(output.shape[0]), @intCast(output.shape[1]), @intCast(output.shape[2]), 0 };
            dump.recordGlobal(self.trace_point, self.layer_idx, output.data().ptr, .f32, shape, 3);
        }
    }
};

/// Layer Normalization configuration (with optional bias).
pub const LayerNorm = struct {
    weight: *const Tensor,
    bias: ?*const Tensor = null,
    dim: usize,
    eps: f32,
    /// Layer index for trace emissions.
    layer_idx: u16 = trace.TraceEmission.NO_LAYER,
    /// Which trace point this norm corresponds to (e.g., layer_attn_norm or layer_ffn_norm).
    trace_point: trace.TracePoint = .layer_attn_norm,

    /// Apply LayerNorm: output = (x - mean) / sqrt(var + eps) * weight + bias
    pub fn forward(self: *const LayerNorm, input: *const Tensor, output: *Tensor) void {
        layerNormForward(self, input, output);
        const seq_len: u32 = @intCast(output.shape[1]);
        if (trace.isEnabled()) {
            trace.emit(
                self.trace_point,
                self.layer_idx,
                0,
                seq_len,
                output.data().ptr,
                .f32,
                .{ @intCast(output.shape[0]), seq_len, @intCast(output.shape[2]), 0 },
                3,
                "layerNormForward",
            );
        }
        // Dump capture (compiled in only for dump binary)
        if (build_options.dump_tensors) {
            const shape = [4]usize{ @intCast(output.shape[0]), @intCast(output.shape[1]), @intCast(output.shape[2]), 0 };
            dump.recordGlobal(self.trace_point, self.layer_idx, output.data().ptr, .f32, shape, 3);
        }
    }
};

/// Unified normalization kernel wrapper (RMSNorm or LayerNorm).
pub const NormKernel = union(enum) {
    rms: RMSNorm,
    layer: LayerNorm,

    pub fn forward(self: *const NormKernel, input: *const Tensor, output: *Tensor) void {
        switch (self.*) {
            .rms => |*n| n.forward(input, output),
            .layer => |*n| n.forward(input, output),
        }
    }

    pub fn dim(self: *const NormKernel) usize {
        return switch (self.*) {
            .rms => |n| n.dim,
            .layer => |n| n.dim,
        };
    }
};

/// Apply RMS normalization: output = x * rsqrt(mean(x²) + eps) * weight
pub fn rmsnormForward(norm: *const RMSNorm, input: *const Tensor, output: *Tensor) void {
    // Internal invariants: tensor shapes must match model config
    std.debug.assert(input.dtype == .f32 and output.dtype == .f32);
    std.debug.assert(input.n_dims == 3 and output.n_dims == 3);
    std.debug.assert(input.shape[0] == output.shape[0] and input.shape[1] == output.shape[1] and input.shape[2] == output.shape[2]);
    std.debug.assert(input.shape[2] == norm.dim);

    const feature_dim = norm.dim;
    const token_count: usize = @intCast(input.shape[0] * input.shape[1]);
    const weight_data_type = norm.weight.dtype;
    const weight_f32_values = if (weight_data_type == .f32) norm.weight.asSlice(f32) else null;
    const weight_u16_values = if (weight_data_type == .bf16 or weight_data_type == .f16) norm.weight.asSlice(u16) else null;

    math.rmsnormContiguous(
        output.asSlice(f32),
        input.asSlice(f32),
        weight_f32_values,
        weight_u16_values,
        weight_data_type,
        token_count,
        feature_dim,
        norm.eps,
        norm.weight_offset,
    );
}

/// Apply LayerNorm: output = (x - mean) / sqrt(var + eps) * weight + bias
pub fn layerNormForward(ln: *const LayerNorm, input: *const Tensor, output: *Tensor) void {
    // Internal invariants: tensor shapes must match model config
    std.debug.assert(input.dtype == .f32 and output.dtype == .f32);
    std.debug.assert(input.n_dims == 3 and output.n_dims == 3);
    std.debug.assert(input.shape[0] == output.shape[0] and input.shape[1] == output.shape[1] and input.shape[2] == output.shape[2]);
    std.debug.assert(input.shape[2] == ln.dim);

    const tv = compute.cpu.tensor_view;
    const input_view = tv.fromSimpleTensor(input) orelse unreachable;
    const output_view = tv.fromSimpleTensor(output) orelse unreachable;
    const weight_view = tv.fromSimpleTensor(ln.weight) orelse unreachable;
    const bias_view = if (ln.bias) |b| tv.fromSimpleTensor(b) orelse unreachable else null;

    compute.cpu.norm.layerNorm(output_view, input_view, weight_view, bias_view, ln.eps);
}
