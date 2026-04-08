//! Mixture of Experts (MoE) FFN Layer
//!
//! Implements sparse MoE routing where each token selects top-k experts.

pub const supported = true;

const std = @import("std");
const build_options = @import("build_options");
const tensor = @import("../../../../tensor.zig");
const compute = @import("../../../../compute/root.zig");
const cpu_linalg = compute.cpu.linalg;
const mxfp4 = compute.cpu.mxfp4;
const cpu_activation = compute.cpu.activation;
const cpu_common = compute.cpu.common;
const cpu_matvec = compute.cpu.linalg.matvec;
const cpu_normalization = compute.cpu.normalization;
const cpu_rowwise = compute.cpu.rowwise;
const cpu_topk = compute.cpu.topk;
const dtype_mod = @import("../../../../dtype.zig");
const inspect = @import("../../../../xray/root.zig");
const trace = inspect.trace;
const dump = if (build_options.dump_tensors) @import("../../../../xray/dump/capture.zig") else struct {
    pub fn recordGlobal(_: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) void {}
};

const Tensor = tensor.Tensor;
const MatmulFn = cpu_linalg.MatmulFn;

/// Dispatch matmul for weight in [out_dim, in_dim] layout (standard weight storage).
/// matmulBF16/F16/etc. already expect [n,k] layout, but matmulF32 expects [k,n].
/// Use matmulF32TransB for F32 weights to match the [n,k] convention.
fn matmulWeightTransposed(a: *const Tensor, b: *const Tensor, out: *Tensor, scratch: *cpu_linalg.MatmulScratch) void {
    if (b.dtype == .f32) {
        cpu_linalg.matmulF32TransB(a, b, out, scratch);
    } else {
        const kernel = (cpu_linalg.matmulKernel(b.dtype) catch cpu_linalg.DispatchedKernel{ .func = cpu_linalg.matmulF32TransB, .name = "matmulF32TransB" }).func;
        kernel(a, b, out, scratch);
    }
}

pub const MoEError = error{
    MissingMoEWeights,
    OutOfMemory,
};

/// Scratch buffers for MoE computation
pub const MoEScratch = struct {
    router_logits: []f32 = &.{},
    expert_weights: []f32 = &.{}, // softmax weights for selected experts
    expert_indices: []u32 = &.{}, // indices of selected experts
    expert_outputs: []f32 = &.{}, // outputs from each expert
    gate_up_values: []f32 = &.{}, // intermediate for SwiGLU
    hidden_values: []f32 = &.{}, // intermediate hidden_values state
    // Gemma4 MoE fused kernel scratch
    shared_gate_up_values: []f32 = &.{},
    shared_hidden_values: []f32 = &.{},
    shared_output: []f32 = &.{},
    normed_input: []f32 = &.{},

    pub fn deinit(self: *MoEScratch, allocator: std.mem.Allocator) void {
        if (self.router_logits.len > 0) allocator.free(self.router_logits);
        if (self.expert_weights.len > 0) allocator.free(self.expert_weights);
        if (self.expert_indices.len > 0) allocator.free(self.expert_indices);
        if (self.expert_outputs.len > 0) allocator.free(self.expert_outputs);
        if (self.gate_up_values.len > 0) allocator.free(self.gate_up_values);
        if (self.hidden_values.len > 0) allocator.free(self.hidden_values);
        if (self.shared_gate_up_values.len > 0) allocator.free(self.shared_gate_up_values);
        if (self.shared_hidden_values.len > 0) allocator.free(self.shared_hidden_values);
        if (self.shared_output.len > 0) allocator.free(self.shared_output);
        if (self.normed_input.len > 0) allocator.free(self.normed_input);
        self.* = .{};
    }
};

/// Expert weights for a single expert in MoE layer
pub const ExpertWeights = struct {
    /// Gate projection [d_model, d_ff] - for separate gate/up format
    gate_proj: ?Tensor = null,
    gate_scales: ?[]const u8 = null,
    gate_bias: ?[]const f32 = null,
    /// Up projection [d_model, d_ff] - for separate gate/up format
    up_proj: ?Tensor = null,
    up_scales: ?[]const u8 = null,
    up_bias: ?[]const f32 = null,
    /// Combined gate and up projection [d_model, 2*d_ff] - for fused format
    gate_up_proj: ?Tensor = null, // Can be MXFP4 quantized
    gate_up_scales: ?[]const u8 = null, // E8M0 scales for MXFP4
    gate_up_bias: ?[]const f32 = null,
    /// Down projection [d_ff, d_model]
    down_proj: Tensor,
    down_scales: ?[]const u8 = null,
    down_bias: ?[]const f32 = null,
};

/// Mixture of Experts FFN Layer
/// Implements: output = sum(expert_weight[i] * expert[i](x)) for top-k experts
pub const MoEFFN = struct {
    /// Canonical kernel-call contract for backend parity checks.
    pub const ForwardParams = struct {
        input_tensor: *const Tensor,
        output_tensor: *Tensor,
        scratch: *MoEScratch,
        matmul_scratch: *cpu_linalg.MatmulScratch,
    };

    allocator: std.mem.Allocator,
    d_model: usize,
    d_ff: usize,
    num_experts: usize,
    experts_per_token: usize, // top-k

    /// Router weights [d_model, num_experts]
    router_weight: Tensor,
    router_bias: ?[]const f32 = null,

    /// Expert weights - indexed by expert id
    experts: []ExpertWeights,

    /// Whether experts use MXFP4 quantization
    use_mxfp4: bool = false,
    /// Use SwiGLU variant (alpha=1.702, clipping, (up+1) formulation) inside MoE experts.
    use_swiglu_variant: bool = false,
    /// MXFP4 weights are transposed (input @ weight instead of weight @ input)
    use_transposed_weights: bool = false,

    /// Use GELU activation instead of SiLU in expert FFN
    use_gelu: bool = false,

    /// Shared MLP weights (Gemma4 MoE — coexists with experts)
    shared_gate_proj: ?Tensor = null,
    shared_up_proj: ?Tensor = null,
    shared_down_proj: ?Tensor = null,
    shared_d_ff: usize = 0,

    /// Router scaling (Gemma4 MoE)
    router_input_scale: ?[]const f32 = null,
    router_per_expert_scale: ?[]const f32 = null,
    router_scalar_root_size: f32 = 0.0,

    /// Shared expert gate (Qwen3.5 MoE — sigmoid gate for scaling shared expert output)
    shared_expert_gate: ?Tensor = null,

    /// Internal norms for fused FFN+MoE (Gemma4 MoE)
    pre_ffn_norm_weight: ?Tensor = null,
    post_shared_norm_weight: ?Tensor = null,
    pre_expert_norm_weight: ?Tensor = null,
    post_expert_norm_weight: ?Tensor = null,
    post_combine_norm_weight: ?Tensor = null,
    norm_eps: f32 = 1e-6,
    norm_weight_offset: f32 = 0.0,

    /// Layer index for trace emission (NO_LAYER = 0xFFFF for non-layer points)
    layer_idx: u16 = trace.TraceEmission.NO_LAYER,
    /// Kernel name for trace emission (identifies MoE implementation)
    kernel_name: ?[]const u8 = null,

    pub fn forward(self: *const MoEFFN, input_tensor: *const Tensor, output_tensor: *Tensor, scratch: *MoEScratch, matmul_scratch: *cpu_linalg.MatmulScratch) !void {
        // Dispatch to fused FFN+MoE path when internal norms are present (Gemma4 MoE).
        // Qwen-style shared expert (no internal norms) is handled inline below.
        if (self.pre_ffn_norm_weight != null) {
            return self.forwardFusedMoE(input_tensor, output_tensor, scratch, matmul_scratch);
        }

        std.debug.assert(input_tensor.n_dims == 3 and output_tensor.n_dims == 3);
        std.debug.assert(input_tensor.shape[0] == 1 and output_tensor.shape[0] == 1);
        const seq_len: usize = @intCast(input_tensor.shape[1]);
        std.debug.assert(input_tensor.shape[2] == self.d_model and output_tensor.shape[2] == self.d_model);
        std.debug.assert(self.router_weight.n_dims == 2);

        // Allocate scratch buffers
        const effective_d_ff = @max(self.d_ff, self.shared_d_ff);
        try cpu_common.ensureF32Slice(self.allocator, &scratch.router_logits, seq_len * self.num_experts);
        try cpu_common.ensureF32Slice(self.allocator, &scratch.expert_weights, seq_len * self.experts_per_token);
        try cpu_common.ensureU32Slice(self.allocator, &scratch.expert_indices, seq_len * self.experts_per_token);
        try cpu_common.ensureF32Slice(self.allocator, &scratch.expert_outputs, seq_len * self.d_model * self.experts_per_token);
        try cpu_common.ensureF32Slice(self.allocator, &scratch.gate_up_values, seq_len * 2 * effective_d_ff);
        try cpu_common.ensureF32Slice(self.allocator, &scratch.hidden_values, seq_len * effective_d_ff);

        const input_values = input_tensor.asSlice(f32);
        const output_values = output_tensor.asSlice(f32);

        // Zero output
        @memset(output_values, 0.0);

        // Process each token
        for (0..seq_len) |token_index| {
            const token_input = input_values[token_index * self.d_model ..][0..self.d_model];
            const token_output = output_values[token_index * self.d_model ..][0..self.d_model];

            // 1. Compute router logits: [num_experts]
            const router_logits = scratch.router_logits[0..self.num_experts];
            {
                var input_view = Tensor.view2DSlice(@constCast(token_input), 1, self.d_model);
                var logits_view = Tensor.view2DSlice(router_logits, 1, self.num_experts);
                matmulWeightTransposed(&input_view, &self.router_weight, &logits_view, matmul_scratch);
                if (self.router_bias) |bias| {
                    cpu_common.addBiasRows(router_logits, bias, 1, self.num_experts);
                }
            }

            // 2. Select top-k experts
            const selected_expert_indices = scratch.expert_indices[token_index * self.experts_per_token ..][0..self.experts_per_token];
            const selected_expert_weights = scratch.expert_weights[token_index * self.experts_per_token ..][0..self.experts_per_token];
            try cpu_topk.selectTopKNormalized(router_logits, self.experts_per_token, selected_expert_indices, selected_expert_weights);

            // 3. Run selected experts and combine outputs
            for (0..self.experts_per_token) |expert_selection| {
                const expert_index = selected_expert_indices[expert_selection];
                const weight = selected_expert_weights[expert_selection];

                if (expert_index >= self.num_experts) continue;

                const expert = &self.experts[expert_index];

                // Run expert FFN (SwiGLU)
                const expert_output = scratch.expert_outputs[expert_selection * self.d_model ..][0..self.d_model];
                try self.runExpert(expert, token_input, expert_output, scratch, matmul_scratch);

                // Accumulate weighted output
                cpu_rowwise.addScaledInPlace(token_output, expert_output, weight);
            }

            // Shared expert (Qwen3.5 MoE / Qwen3-Next pattern — no internal norms)
            if (self.shared_gate_proj) |shared_gate| {
                const shared_up = self.shared_up_proj.?;
                const shared_down = self.shared_down_proj.?;
                const s_dff = self.shared_d_ff;

                const s_gate_values = scratch.gate_up_values[0..s_dff];
                const s_up_values = scratch.gate_up_values[s_dff..][0..s_dff];
                const s_hidden = scratch.hidden_values[0..s_dff];
                const s_output = scratch.expert_outputs[0..self.d_model];

                var input_view = Tensor.view2DSlice(@constCast(token_input), 1, self.d_model);
                var gate_out_view = Tensor.view2DSlice(s_gate_values, 1, s_dff);
                var up_out_view = Tensor.view2DSlice(s_up_values, 1, s_dff);
                var hidden_view = Tensor.view2DSlice(s_hidden, 1, s_dff);
                var output_view = Tensor.view2DSlice(s_output, 1, self.d_model);

                matmulWeightTransposed(&input_view, &shared_gate, &gate_out_view, matmul_scratch);
                matmulWeightTransposed(&input_view, &shared_up, &up_out_view, matmul_scratch);

                if (self.use_gelu) {
                    cpu_activation.geluMulSplit(s_gate_values, s_up_values, s_hidden);
                } else {
                    cpu_activation.siluMulSplit(s_gate_values, s_up_values, s_hidden);
                }

                matmulWeightTransposed(&hidden_view, &shared_down, &output_view, matmul_scratch);

                // Optional sigmoid gate scaling
                if (self.shared_expert_gate) |gate_weight| {
                    const gate_data = gate_weight.asSlice(f32);
                    const gate_value = 1.0 / (1.0 + @exp(-cpu_linalg.dot.dotProductF32(token_input, gate_data)));
                    cpu_rowwise.scaleInPlace(s_output, gate_value);
                }

                cpu_rowwise.addScaledInPlace(token_output, s_output, 1.0);
            }
        }

        // Single trace emit at end of forward() - never inside token/expert loops
        if (trace.isEnabled()) {
            trace.emit(
                .ffn_down, // Use ffn_down to match FFN kernel trace point
                self.layer_idx,
                0, // token index
                @intCast(seq_len), // position
                output_tensor.data().ptr,
                .f32,
                .{ 1, @intCast(seq_len), @intCast(self.d_model), 0 },
                3,
                self.kernel_name,
            );
        }
        // Dump capture (compiled in only for dump binary)
        if (build_options.dump_tensors) {
            const shape = [4]usize{ 1, seq_len, self.d_model, 0 };
            dump.recordGlobal(.ffn_down, self.layer_idx, output_tensor.data().ptr, .f32, shape, 3);
        }
    }

    /// Fused FFN+MoE forward pass for Gemma4 MoE.
    /// Handles: shared MLP + custom router (softmax-then-topk) + experts + internal norms.
    fn forwardFusedMoE(self: *const MoEFFN, input_tensor: *const Tensor, output_tensor: *Tensor, scratch: *MoEScratch, matmul_scratch: *cpu_linalg.MatmulScratch) !void {
        std.debug.assert(input_tensor.n_dims == 3 and output_tensor.n_dims == 3);
        std.debug.assert(input_tensor.shape[0] == 1 and output_tensor.shape[0] == 1);
        const seq_len: usize = @intCast(input_tensor.shape[1]);
        std.debug.assert(input_tensor.shape[2] == self.d_model and output_tensor.shape[2] == self.d_model);

        const shared_gate_proj = self.shared_gate_proj.?;
        const shared_up_proj = self.shared_up_proj.?;
        const shared_down_proj = self.shared_down_proj.?;

        // Allocate scratch buffers
        try cpu_common.ensureF32Slice(self.allocator, &scratch.router_logits, self.num_experts);
        try cpu_common.ensureF32Slice(self.allocator, &scratch.expert_weights, self.experts_per_token);
        try cpu_common.ensureU32Slice(self.allocator, &scratch.expert_indices, self.experts_per_token);
        try cpu_common.ensureF32Slice(self.allocator, &scratch.expert_outputs, self.d_model);
        try cpu_common.ensureF32Slice(self.allocator, &scratch.gate_up_values, 2 * self.d_ff);
        try cpu_common.ensureF32Slice(self.allocator, &scratch.hidden_values, self.d_ff);
        try cpu_common.ensureF32Slice(self.allocator, &scratch.shared_gate_up_values, 2 * self.shared_d_ff);
        try cpu_common.ensureF32Slice(self.allocator, &scratch.shared_hidden_values, self.shared_d_ff);
        try cpu_common.ensureF32Slice(self.allocator, &scratch.shared_output, self.d_model);
        try cpu_common.ensureF32Slice(self.allocator, &scratch.normed_input, self.d_model);

        const input_values = input_tensor.asSlice(f32);
        const output_values = output_tensor.asSlice(f32);
        @memset(output_values, 0.0);

        for (0..seq_len) |token_index| {
            const token_input = input_values[token_index * self.d_model ..][0..self.d_model];
            const token_output = output_values[token_index * self.d_model ..][0..self.d_model];
            const normed = scratch.normed_input[0..self.d_model];
            const shared_out = scratch.shared_output[0..self.d_model];

            // === 1. Shared MLP path ===
            // pre_feedforward_layernorm(input) → normed
            @memcpy(normed, token_input);
            if (self.pre_ffn_norm_weight) |*w| {
                cpu_normalization.rmsnormInPlaceWeightTensor(normed, w, self.norm_eps, self.norm_weight_offset);
            }

            // shared_mlp: gate_proj + up_proj → GELU → down_proj
            {
                const sg_values = scratch.shared_gate_up_values[0 .. 2 * self.shared_d_ff];
                const sh_values = scratch.shared_hidden_values[0..self.shared_d_ff];
                const s_gate = sg_values[0..self.shared_d_ff];
                const s_up = sg_values[self.shared_d_ff..][0..self.shared_d_ff];

                var input_view = Tensor.view2DSlice(@constCast(normed), 1, self.d_model);
                var gate_out = Tensor.view2DSlice(s_gate, 1, self.shared_d_ff);
                matmulWeightTransposed(&input_view, &shared_gate_proj, &gate_out, matmul_scratch);

                var up_out = Tensor.view2DSlice(s_up, 1, self.shared_d_ff);
                matmulWeightTransposed(&input_view, &shared_up_proj, &up_out, matmul_scratch);

                cpu_activation.geluMulSplit(s_gate, s_up, sh_values);

                var hidden_view = Tensor.view2DSlice(sh_values, 1, self.shared_d_ff);
                var out_view = Tensor.view2DSlice(shared_out, 1, self.d_model);
                matmulWeightTransposed(&hidden_view, &shared_down_proj, &out_view, matmul_scratch);
            }

            // post_feedforward_layernorm_1(shared_out)
            if (self.post_shared_norm_weight) |*w| {
                cpu_normalization.rmsnormInPlaceWeightTensor(shared_out, w, self.norm_eps, self.norm_weight_offset);
            }

            // === 2. Router (operates on raw residual) ===
            const router_logits = scratch.router_logits[0..self.num_experts];

            // Unscaled RMSNorm + learned scale on the router input
            @memcpy(normed, token_input);
            {
                var sum_sq: f32 = 0.0;
                for (normed) |v| sum_sq += v * v;
                const rms = @sqrt(sum_sq / @as(f32, @floatFromInt(self.d_model)) + self.norm_eps);
                const inv_rms = 1.0 / rms;
                if (self.router_input_scale) |scale| {
                    for (normed, scale) |*x, s| {
                        x.* = x.* * inv_rms * s * self.router_scalar_root_size;
                    }
                } else {
                    for (normed) |*x| x.* *= inv_rms;
                }
            }

            {
                var input_view = Tensor.view2DSlice(@constCast(normed), 1, self.d_model);
                var logits_view = Tensor.view2DSlice(router_logits, 1, self.num_experts);
                matmulWeightTransposed(&input_view, &self.router_weight, &logits_view, matmul_scratch);
                if (self.router_bias) |bias| {
                    cpu_common.addBiasRows(router_logits, bias, 1, self.num_experts);
                }
            }

            // Softmax over ALL experts, then top-k, then renormalize
            {
                var max_logit: f32 = -std.math.inf(f32);
                for (router_logits) |l| {
                    if (l > max_logit) max_logit = l;
                }
                var sum_exp: f32 = 0.0;
                for (router_logits) |*l| {
                    l.* = @exp(l.* - max_logit);
                    sum_exp += l.*;
                }
                if (sum_exp > 0.0) {
                    const inv_sum = 1.0 / sum_exp;
                    for (router_logits) |*l| l.* *= inv_sum;
                }
            }

            const selected_indices = scratch.expert_indices[0..self.experts_per_token];
            const selected_weights = scratch.expert_weights[0..self.experts_per_token];

            // Top-k selection from softmax weights
            for (0..self.experts_per_token) |k| {
                var best_idx: u32 = 0;
                var best_val: f32 = -std.math.inf(f32);
                for (router_logits, 0..) |val, i| {
                    if (val > best_val) {
                        var already = false;
                        for (selected_indices[0..k]) |prev| {
                            if (prev == @as(u32, @intCast(i))) { already = true; break; }
                        }
                        if (!already) {
                            best_val = val;
                            best_idx = @intCast(i);
                        }
                    }
                }
                selected_indices[k] = best_idx;
                selected_weights[k] = best_val;
            }

            // Renormalize selected weights to sum to 1.0
            {
                var weight_sum: f32 = 0.0;
                for (selected_weights) |w| weight_sum += w;
                if (weight_sum > 0.0) {
                    const inv_sum = 1.0 / weight_sum;
                    for (selected_weights) |*w| w.* *= inv_sum;
                }
            }

            // Apply per-expert scale
            if (self.router_per_expert_scale) |per_expert_scale| {
                for (selected_indices, selected_weights) |idx, *w| {
                    if (idx < self.num_experts) {
                        w.* *= per_expert_scale[idx];
                    }
                }
            }

            // === 3. Expert path ===
            // pre_feedforward_layernorm_2(input) → normed
            @memcpy(normed, token_input);
            if (self.pre_expert_norm_weight) |*w| {
                cpu_normalization.rmsnormInPlaceWeightTensor(normed, w, self.norm_eps, self.norm_weight_offset);
            }

            // Run selected experts and accumulate weighted output
            @memset(token_output, 0.0);
            for (0..self.experts_per_token) |expert_sel| {
                const expert_index = selected_indices[expert_sel];
                const weight = selected_weights[expert_sel];
                if (expert_index >= self.num_experts) continue;

                const expert = &self.experts[expert_index];
                const expert_output = scratch.expert_outputs[0..self.d_model];
                try self.runExpert(expert, normed, expert_output, scratch, matmul_scratch);
                cpu_rowwise.addScaledInPlace(token_output, expert_output, weight);
            }

            // post_feedforward_layernorm_2(expert_output)
            if (self.post_expert_norm_weight) |*w| {
                cpu_normalization.rmsnormInPlaceWeightTensor(token_output, w, self.norm_eps, self.norm_weight_offset);
            }

            // === 4. Combine: shared_out + expert_out ===
            for (token_output, shared_out) |*o, s| o.* += s;

            // post_feedforward_layernorm(combined)
            if (self.post_combine_norm_weight) |*w| {
                cpu_normalization.rmsnormInPlaceWeightTensor(token_output, w, self.norm_eps, self.norm_weight_offset);
            }
        }

        // Trace/dump emission
        if (trace.isEnabled()) {
            trace.emit(.ffn_down, self.layer_idx, 0, @intCast(seq_len), output_tensor.data().ptr, .f32, .{ 1, @intCast(seq_len), @intCast(self.d_model), 0 }, 3, self.kernel_name);
        }
        if (build_options.dump_tensors) {
            const shape = [4]usize{ 1, seq_len, self.d_model, 0 };
            dump.recordGlobal(.ffn_down, self.layer_idx, output_tensor.data().ptr, .f32, shape, 3);
        }
    }

    fn runExpert(
        self: *const MoEFFN,
        expert: *const ExpertWeights,
        input_vector: []const f32,
        output_vector: []f32,
        scratch: *MoEScratch,
        matmul_scratch: *cpu_linalg.MatmulScratch,
    ) !void {
        const gate_up_values = scratch.gate_up_values[0 .. 2 * self.d_ff];
        const hidden_values = scratch.hidden_values[0..self.d_ff];
        const gate_values = gate_up_values[0..self.d_ff];
        const up_values = gate_up_values[self.d_ff..][0..self.d_ff];

        // Check if using separate gate_values/up_values projections or fused
        const use_separate_projections = expert.gate_proj != null and expert.up_proj != null;

        if (use_separate_projections) {
            // Separate gate and up projections
            const gate_proj = expert.gate_proj.?;
            const up_proj = expert.up_proj.?;

            if (self.use_mxfp4 and expert.gate_scales != null) {
                // MXFP4 dequantize and matmul for gate_values
                if (self.use_transposed_weights) {
                    // Transposed weights: [in, out], use input @ weight
                    mxfp4.matmulF32Transposed(
                        input_vector,
                        gate_proj.data(),
                        expert.gate_scales.?,
                        gate_values,
                        self.d_model,
                        self.d_ff,
                        null,
                    );
                } else {
                    mxfp4.matmulF32(
                        input_vector,
                        gate_proj.data(),
                        expert.gate_scales.?,
                        gate_values,
                        self.d_model,
                        self.d_ff,
                        null,
                    );
                }
                if (expert.gate_bias) |bias| {
                    cpu_common.addBiasRows(gate_values, bias, 1, self.d_ff);
                }

                // MXFP4 dequantize and matmul for up_values
                if (self.use_transposed_weights) {
                    mxfp4.matmulF32Transposed(
                        input_vector,
                        up_proj.data(),
                        expert.up_scales.?,
                        up_values,
                        self.d_model,
                        self.d_ff,
                        null,
                    );
                } else {
                    mxfp4.matmulF32(
                        input_vector,
                        up_proj.data(),
                        expert.up_scales.?,
                        up_values,
                        self.d_model,
                        self.d_ff,
                        null,
                    );
                }
                if (expert.up_bias) |bias| {
                    cpu_common.addBiasRows(up_values, bias, 1, self.d_ff);
                }
            } else {
                // Standard matmul for gate_values
                var input_view = Tensor.view2DSlice(@constCast(input_vector), 1, self.d_model);
                var gate_output_view = Tensor.view2DSlice(gate_values, 1, self.d_ff);
                matmulWeightTransposed(&input_view, &gate_proj, &gate_output_view, matmul_scratch);
                if (expert.gate_bias) |bias| {
                    cpu_common.addBiasRows(gate_values, bias, 1, self.d_ff);
                }

                // Standard matmul for up_values
                var up_output_view = Tensor.view2DSlice(up_values, 1, self.d_ff);
                matmulWeightTransposed(&input_view, &up_proj, &up_output_view, matmul_scratch);
                if (expert.up_bias) |bias| {
                    cpu_common.addBiasRows(up_values, bias, 1, self.d_ff);
                }
            }
        } else if (expert.gate_up_proj != null) {
            // Fused gate_values+up_values projection
            const gate_up_proj = expert.gate_up_proj.?;
            if (self.use_mxfp4 and expert.gate_up_scales != null) {
                // MXFP4 dequantize and matmul
                // Contiguous format: [gate_values[0:d_ff], up_values[0:d_ff]]
                // The output gate_up_values buffer is already laid out as [gate_values, up_values]
                // so no de-interleaving needed - gate_values and up_values slices point to the right places
                if (self.use_transposed_weights) {
                    mxfp4.matmulF32Transposed(
                        input_vector,
                        gate_up_proj.data(),
                        expert.gate_up_scales.?,
                        gate_up_values,
                        self.d_model,
                        2 * self.d_ff,
                        null,
                    );
                } else {
                    mxfp4.matmulF32(
                        input_vector,
                        gate_up_proj.data(),
                        expert.gate_up_scales.?,
                        gate_up_values,
                        self.d_model,
                        2 * self.d_ff,
                        null,
                    );
                }
                if (expert.gate_up_bias) |bias| {
                    cpu_common.addBiasRows(gate_up_values, bias, 1, 2 * self.d_ff);
                }
                // gate_values and up_values already point to gate_up_values[0..d_ff] and gate_up_values[d_ff..2*d_ff]
                // so no additional work needed
            } else {
                // Standard F32/F16/BF16 matmul
                var input_view = Tensor.view2DSlice(@constCast(input_vector), 1, self.d_model);
                var out_view = Tensor.view2DSlice(gate_up_values, 1, 2 * self.d_ff);
                matmulWeightTransposed(&input_view, &gate_up_proj, &out_view, matmul_scratch);

                if (expert.gate_up_bias) |bias| {
                    cpu_common.addBiasRows(gate_up_values, bias, 1, 2 * self.d_ff);
                }
            }
        } else {
            return error.MissingMoEWeights;
        }

        if (self.use_gelu) {
            cpu_activation.geluMulSplit(gate_values, up_values, hidden_values);
        } else if (self.use_swiglu_variant) {
            // GPT-OSS path uses interleaved gate/up values.
            cpu_activation.swigluVariantInterleaved(gate_up_values, hidden_values);
        } else {
            // Standard SwiGLU: SiLU(gate) * up.
            cpu_activation.siluMulSplit(gate_values, up_values, hidden_values);
        }

        // Down projection
        if (self.use_mxfp4 and expert.down_scales != null) {
            if (self.use_transposed_weights) {
                mxfp4.matmulF32Transposed(
                    hidden_values,
                    expert.down_proj.data(),
                    expert.down_scales.?,
                    output_vector,
                    self.d_ff,
                    self.d_model,
                    null,
                );
            } else {
                mxfp4.matmulF32(
                    hidden_values,
                    expert.down_proj.data(),
                    expert.down_scales.?,
                    output_vector,
                    self.d_ff,
                    self.d_model,
                    null,
                );
            }
            if (expert.down_bias) |bias| {
                cpu_common.addBiasRows(output_vector, bias, 1, self.d_model);
            }
        } else {
            var input_view = Tensor.view2DSlice(hidden_values, 1, self.d_ff);
            var out_view = Tensor.view2DSlice(output_vector, 1, self.d_model);
            matmulWeightTransposed(&input_view, &expert.down_proj, &out_view, matmul_scratch);

            if (expert.down_bias) |bias| {
                cpu_common.addBiasRows(output_vector, bias, 1, self.d_model);
            }
        }
    }
};

// ============================================================================
// Tests
// ============================================================================

test "forward router scoring simple" {
    const alloc = std.testing.allocator;

    const d_model = 4;
    const num_experts = 3;

    // Create router weight [d_model, num_experts]
    var weight_data = [_]f32{
        1.0, 0.0, -1.0, // input dim 0
        0.0, 1.0, 0.0, // input dim 1
        0.5, 0.5, 0.5, // input dim 2
        0.0, 0.0, 1.0, // input dim 3
    };
    var router_weight = Tensor.view2DSlice(&weight_data, d_model, num_experts);

    // Input vector
    const input = [_]f32{ 1.0, 2.0, 1.0, 0.5 };

    // Compute logits
    const logits = try alloc.alloc(f32, num_experts);
    defer alloc.free(logits);

    cpu_matvec.matVecDense(&input, &router_weight, null, logits);

    // Expected: [1.0*1 + 2.0*0 + 1.0*0.5 + 0.5*0,
    //            1.0*0 + 2.0*1 + 1.0*0.5 + 0.5*0,
    //            1.0*-1 + 2.0*0 + 1.0*0.5 + 0.5*1]
    try std.testing.expectApproxEqAbs(1.5, logits[0], 1e-5);
    try std.testing.expectApproxEqAbs(2.5, logits[1], 1e-5);
    try std.testing.expectApproxEqAbs(0.0, logits[2], 1e-5);
}

test "computeRouterLogits supports [num_experts, d_model] router layout" {
    const d_model = 4;
    const num_experts = 3;

    // Router weights stored as [num_experts, d_model] rows.
    var weight_data = [_]f32{
        1.0, 0.0, 0.5, 0.0, // expert 0
        0.0, 1.0, 0.5, 0.0, // expert 1
        -1.0, 0.0, 0.5, 1.0, // expert 2
    };
    var router_weight = Tensor.view2DSlice(&weight_data, num_experts, d_model);

    const input = [_]f32{ 1.0, 2.0, 1.0, 0.5 };
    var logits = [_]f32{ 0.0, 0.0, 0.0 };

    cpu_matvec.matVecDense(&input, &router_weight, null, &logits);

    try std.testing.expectApproxEqAbs(1.5, logits[0], 1e-5);
    try std.testing.expectApproxEqAbs(2.5, logits[1], 1e-5);
    try std.testing.expectApproxEqAbs(0.0, logits[2], 1e-5);
}

test "computeRouterLogits supports BF16 router weights" {
    const d_model = 4;
    const num_experts = 3;

    // Router weights stored as [num_experts, d_model] in BF16.
    const weight_bf16 = [_]u16{
        dtype_mod.f32ToBf16(1.0), dtype_mod.f32ToBf16(0.0), dtype_mod.f32ToBf16(0.5), dtype_mod.f32ToBf16(0.0), // expert 0
        dtype_mod.f32ToBf16(0.0), dtype_mod.f32ToBf16(1.0), dtype_mod.f32ToBf16(0.5), dtype_mod.f32ToBf16(0.0), // expert 1
        dtype_mod.f32ToBf16(-1.0), dtype_mod.f32ToBf16(0.0), dtype_mod.f32ToBf16(0.5), dtype_mod.f32ToBf16(1.0), // expert 2
    };
    const router_weight = Tensor.view(@ptrCast(@constCast(weight_bf16[0..].ptr)), &.{ num_experts, d_model }, .bf16, weight_bf16.len * @sizeOf(u16));

    const input = [_]f32{ 1.0, 2.0, 1.0, 0.5 };
    var logits = [_]f32{ 0.0, 0.0, 0.0 };

    cpu_matvec.matVecDense(&input, &router_weight, null, &logits);

    try std.testing.expectApproxEqAbs(1.5, logits[0], 1e-5);
    try std.testing.expectApproxEqAbs(2.5, logits[1], 1e-5);
    try std.testing.expectApproxEqAbs(0.0, logits[2], 1e-5);
}

test "forward router scoring bias" {
    const alloc = std.testing.allocator;

    const d_model = 2;
    const num_experts = 3;

    var weight_data = [_]f32{
        1.0, 0.0, 0.5,
        0.0, 1.0, 0.5,
    };
    var router_weight = Tensor.view2DSlice(&weight_data, d_model, num_experts);

    const input = [_]f32{ 1.0, 1.0 };
    const bias = [_]f32{ 0.1, 0.2, 0.3 };

    const logits = try alloc.alloc(f32, num_experts);
    defer alloc.free(logits);

    cpu_matvec.matVecDense(&input, &router_weight, &bias, logits);

    // Expected: [1.0 + 0.1, 1.0 + 0.2, 1.0 + 0.3]
    try std.testing.expectApproxEqAbs(1.1, logits[0], 1e-5);
    try std.testing.expectApproxEqAbs(1.2, logits[1], 1e-5);
    try std.testing.expectApproxEqAbs(1.3, logits[2], 1e-5);
}

test "forward top-k highest" {
    const alloc = std.testing.allocator;

    const k = 2;
    const logits = [_]f32{ 1.0, 5.0, 3.0, 2.0, 4.0 };

    const indices = try alloc.alloc(u32, k);
    defer alloc.free(indices);
    const weights = try alloc.alloc(f32, k);
    defer alloc.free(weights);

    try cpu_topk.selectTopKNormalized(&logits, k, indices, weights);

    // Should select experts 1 (5.0) and 4 (4.0)
    try std.testing.expectEqual(@as(u32, 1), indices[0]);
    try std.testing.expectEqual(@as(u32, 4), indices[1]);

    // Weights should sum to 1.0 after softmax
    const weight_sum = weights[0] + weights[1];
    try std.testing.expectApproxEqAbs(1.0, weight_sum, 1e-5);
}

test "forward top-k single expert" {
    const alloc = std.testing.allocator;

    const k = 1;
    const logits = [_]f32{ 1.0, 3.0, 2.0, 0.5 };

    const indices = try alloc.alloc(u32, k);
    defer alloc.free(indices);
    const weights = try alloc.alloc(f32, k);
    defer alloc.free(weights);

    try cpu_topk.selectTopKNormalized(&logits, k, indices, weights);

    // Should select expert 1 (highest score)
    try std.testing.expectEqual(@as(u32, 1), indices[0]);

    // Single weight should be 1.0
    try std.testing.expectApproxEqAbs(1.0, weights[0], 1e-5);
}

test "selectTopKExperts handles non-finite logits deterministically" {
    const alloc = std.testing.allocator;

    const logits = [_]f32{ std.math.nan(f32), std.math.inf(f32), -1.0, 3.0 };
    const k = 2;
    const indices = try alloc.alloc(u32, k);
    defer alloc.free(indices);
    const weights = try alloc.alloc(f32, k);
    defer alloc.free(weights);

    try cpu_topk.selectTopKNormalized(&logits, k, indices, weights);
    try std.testing.expectEqual(@as(u32, 3), indices[0]);
    try std.testing.expectEqual(@as(u32, 2), indices[1]);
    try std.testing.expect(std.math.isFinite(weights[0]));
    try std.testing.expect(std.math.isFinite(weights[1]));
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), weights[0] + weights[1], 1e-6);
}

test "forward top-k all experts" {
    const alloc = std.testing.allocator;

    const k = 4;
    const logits = [_]f32{ 1.0, 4.0, 2.0, 3.0 };

    const indices = try alloc.alloc(u32, k);
    defer alloc.free(indices);
    const weights = try alloc.alloc(f32, k);
    defer alloc.free(weights);

    try cpu_topk.selectTopKNormalized(&logits, k, indices, weights);

    // Should select all in descending order: 1, 3, 2, 0
    try std.testing.expectEqual(@as(u32, 1), indices[0]);
    try std.testing.expectEqual(@as(u32, 3), indices[1]);
    try std.testing.expectEqual(@as(u32, 2), indices[2]);
    try std.testing.expectEqual(@as(u32, 0), indices[3]);

    // Weights should sum to 1.0
    var weight_sum: f32 = 0.0;
    for (weights) |w| weight_sum += w;
    try std.testing.expectApproxEqAbs(1.0, weight_sum, 1e-5);
}

test "forward top-k equal scores" {
    const alloc = std.testing.allocator;

    const k = 2;
    const logits = [_]f32{ 2.0, 2.0, 2.0, 2.0 };

    const indices = try alloc.alloc(u32, k);
    defer alloc.free(indices);
    const weights = try alloc.alloc(f32, k);
    defer alloc.free(weights);

    try cpu_topk.selectTopKNormalized(&logits, k, indices, weights);

    // Should select first k experts due to iteration order
    try std.testing.expectEqual(@as(u32, 0), indices[0]);
    try std.testing.expectEqual(@as(u32, 1), indices[1]);

    // Weights should be equal and sum to 1.0
    try std.testing.expectApproxEqAbs(0.5, weights[0], 1e-5);
    try std.testing.expectApproxEqAbs(0.5, weights[1], 1e-5);
}

test "forward top-k dominant" {
    const alloc = std.testing.allocator;

    const k = 2;
    // Expert 1 has much higher logit
    const logits = [_]f32{ 0.1, 10.0, 0.2, 0.15 };

    const indices = try alloc.alloc(u32, k);
    defer alloc.free(indices);
    const weights = try alloc.alloc(f32, k);
    defer alloc.free(weights);

    try cpu_topk.selectTopKNormalized(&logits, k, indices, weights);

    // Should select expert 1 and 2
    try std.testing.expectEqual(@as(u32, 1), indices[0]);
    try std.testing.expectEqual(@as(u32, 2), indices[1]);

    // Expert 1 should have weight very close to 1.0
    try std.testing.expect(weights[0] > 0.999);
    try std.testing.expect(weights[1] < 0.001);

    // Weights still sum to 1.0
    const weight_sum = weights[0] + weights[1];
    try std.testing.expectApproxEqAbs(1.0, weight_sum, 1e-5);
}

test "forward top-k zero error" {
    const alloc = std.testing.allocator;

    const k = 0;
    const logits = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    const indices = try alloc.alloc(u32, 1);
    defer alloc.free(indices);
    const weights = try alloc.alloc(f32, 1);
    defer alloc.free(weights);

    const result = cpu_topk.selectTopKNormalized(&logits, k, indices, weights);
    try std.testing.expectError(error.InvalidMoEConfig, result);
}

test "forward top-k exceeds error" {
    const alloc = std.testing.allocator;

    const k = 5;
    const logits = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    const indices = try alloc.alloc(u32, k);
    defer alloc.free(indices);
    const weights = try alloc.alloc(f32, k);
    defer alloc.free(weights);

    const result = cpu_topk.selectTopKNormalized(&logits, k, indices, weights);
    try std.testing.expectError(error.InvalidMoEConfig, result);
}

test "forward weight normalization" {
    const alloc = std.testing.allocator;

    const k = 3;
    // Create logits with specific ratios
    const logits = [_]f32{ 1.0, 6.0, 3.0, 2.0, 5.0, 4.0 };

    const indices = try alloc.alloc(u32, k);
    defer alloc.free(indices);
    const weights = try alloc.alloc(f32, k);
    defer alloc.free(weights);

    try cpu_topk.selectTopKNormalized(&logits, k, indices, weights);

    // Top 3: expert 1 (6.0), expert 4 (5.0), expert 5 (4.0)
    try std.testing.expectEqual(@as(u32, 1), indices[0]);
    try std.testing.expectEqual(@as(u32, 4), indices[1]);
    try std.testing.expectEqual(@as(u32, 5), indices[2]);

    // Verify proportions: exp(6)/Z > exp(5)/Z > exp(4)/Z
    try std.testing.expect(weights[0] > weights[1]);
    try std.testing.expect(weights[1] > weights[2]);

    // Verify sum to 1.0
    var weight_sum: f32 = 0.0;
    for (weights) |w| weight_sum += w;
    try std.testing.expectApproxEqAbs(1.0, weight_sum, 1e-5);
}

test "forward determinism" {
    const alloc = std.testing.allocator;

    const k = 2;
    const logits = [_]f32{ 1.5, 3.2, 0.8, 2.1, 4.3 };

    // First run
    const indices1 = try alloc.alloc(u32, k);
    defer alloc.free(indices1);
    const weights1 = try alloc.alloc(f32, k);
    defer alloc.free(weights1);

    try cpu_topk.selectTopKNormalized(&logits, k, indices1, weights1);

    // Second run
    const indices2 = try alloc.alloc(u32, k);
    defer alloc.free(indices2);
    const weights2 = try alloc.alloc(f32, k);
    defer alloc.free(weights2);

    try cpu_topk.selectTopKNormalized(&logits, k, indices2, weights2);

    // Results should be identical
    for (0..k) |i| {
        try std.testing.expectEqual(indices1[i], indices2[i]);
        try std.testing.expectApproxEqAbs(weights1[i], weights2[i], 1e-7);
    }
}

test "forward softmax negative" {
    const alloc = std.testing.allocator;

    const k = 3;
    const logits = [_]f32{ -2.0, -1.0, -3.0, -0.5 };

    const indices = try alloc.alloc(u32, k);
    defer alloc.free(indices);
    const weights = try alloc.alloc(f32, k);
    defer alloc.free(weights);

    try cpu_topk.selectTopKNormalized(&logits, k, indices, weights);

    // Verify sum to 1.0 even with negative logits
    var weight_sum: f32 = 0.0;
    for (weights) |w| weight_sum += w;
    try std.testing.expectApproxEqAbs(1.0, weight_sum, 1e-5);

    // All weights should be positive
    for (weights) |w| {
        try std.testing.expect(w > 0.0);
        try std.testing.expect(w < 1.0);
    }
}

test "forward softmax large diff" {
    const alloc = std.testing.allocator;

    const k = 3;
    const logits = [_]f32{ 0.0, 100.0, 50.0, 1.0, 0.5 };

    const indices = try alloc.alloc(u32, k);
    defer alloc.free(indices);
    const weights = try alloc.alloc(f32, k);
    defer alloc.free(weights);

    try cpu_topk.selectTopKNormalized(&logits, k, indices, weights);

    // Expert 1 (100.0) should dominate
    try std.testing.expectEqual(@as(u32, 1), indices[0]);
    try std.testing.expect(weights[0] > 0.999);

    // Verify numerical stability - no NaN or Inf
    for (weights) |w| {
        try std.testing.expect(!std.math.isNan(w));
        try std.testing.expect(!std.math.isInf(w));
    }

    // Sum should still be 1.0
    var weight_sum: f32 = 0.0;
    for (weights) |w| weight_sum += w;
    try std.testing.expectApproxEqAbs(1.0, weight_sum, 1e-5);
}

test "forward SwiGLU standard" {
    const alloc = std.testing.allocator;

    const d_model = 4;
    const d_ff = 8;

    // Create simple expert with identity-like weights
    var gate_data = try alloc.alloc(f32, d_model * d_ff);
    defer alloc.free(gate_data);
    @memset(gate_data, 0.0);
    for (0..@min(d_model, d_ff)) |i| {
        gate_data[i * d_ff + i] = 1.0;
    }
    const gate_proj = Tensor.view2DSlice(gate_data, d_model, d_ff);

    var up_data = try alloc.alloc(f32, d_model * d_ff);
    defer alloc.free(up_data);
    @memset(up_data, 0.0);
    for (0..@min(d_model, d_ff)) |i| {
        up_data[i * d_ff + i] = 2.0; // Scale by 2
    }
    const up_proj = Tensor.view2DSlice(up_data, d_model, d_ff);

    var down_data = try alloc.alloc(f32, d_ff * d_model);
    defer alloc.free(down_data);
    @memset(down_data, 0.0);
    for (0..@min(d_model, d_ff)) |i| {
        down_data[i * d_model + i] = 1.0;
    }
    const down_proj = Tensor.view2DSlice(down_data, d_ff, d_model);

    const expert = ExpertWeights{
        .gate_proj = gate_proj,
        .up_proj = up_proj,
        .down_proj = down_proj,
    };

    const moe = MoEFFN{
        .allocator = alloc,
        .d_model = d_model,
        .d_ff = d_ff,
        .num_experts = 1,
        .experts_per_token = 1,
        .router_weight = undefined,
        .experts = undefined,
        .use_swiglu_variant = false,
    };

    const input = [_]f32{ 1.0, 0.0, -1.0, 0.5 };
    const output = try alloc.alloc(f32, d_model);
    defer alloc.free(output);

    var scratch = MoEScratch{};
    defer scratch.deinit(alloc);
    var matmul_scratch = try cpu_linalg.MatmulScratch.init(alloc);
    defer matmul_scratch.deinit();

    try cpu_common.ensureF32Slice(alloc, &scratch.gate_up_values, 2 * d_ff);
    try cpu_common.ensureF32Slice(alloc, &scratch.hidden_values, d_ff);

    try moe.runExpert(&expert, &input, output, &scratch, &matmul_scratch);

    // With identity-like weights:
    // gate_proj output: [1.0, 0, -1, 0.5, 0, 0, 0, 0]
    // up_proj output: [2.0, 0, -2, 1.0, 0, 0, 0, 0]
    // For x=1.0: silu(1.0) = 1/(1+e^-1) * 1.0 ≈ 0.731
    // hidden[0] = 0.731 * 2.0 ≈ 1.462

    // Just verify output is non-zero and finite
    try std.testing.expect(!std.math.isNan(output[0]));
    try std.testing.expect(!std.math.isInf(output[0]));
    try std.testing.expect(output[0] != 0.0);
}

test "forward output aggregation" {
    const d_model = 4;
    _ = @as(usize, 2); // k experts

    // Simulate two expert outputs
    const expert1_output = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const expert2_output = [_]f32{ 0.5, 1.0, 1.5, 2.0 };

    // Weights from softmax
    const weight1: f32 = 0.7;
    const weight2: f32 = 0.3;

    // Aggregate
    var final_output = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    for (0..d_model) |i| {
        final_output[i] += weight1 * expert1_output[i];
        final_output[i] += weight2 * expert2_output[i];
    }

    // Expected: [0.7*1.0 + 0.3*0.5, 0.7*2.0 + 0.3*1.0, ...]
    try std.testing.expectApproxEqAbs(0.85, final_output[0], 1e-5);
    try std.testing.expectApproxEqAbs(1.70, final_output[1], 1e-5);
    try std.testing.expectApproxEqAbs(2.55, final_output[2], 1e-5);
    try std.testing.expectApproxEqAbs(3.40, final_output[3], 1e-5);
}

test "forward scratch buffer" {
    const alloc = std.testing.allocator;

    var scratch = MoEScratch{};
    defer scratch.deinit(alloc);

    // Test initial state
    try std.testing.expectEqual(@as(usize, 0), scratch.router_logits.len);
    try std.testing.expectEqual(@as(usize, 0), scratch.expert_weights.len);

    // Test allocation
    try cpu_common.ensureF32Slice(alloc, &scratch.router_logits, 10);
    try std.testing.expectEqual(@as(usize, 10 + cpu_common.GUARD_F32S), scratch.router_logits.len);

    // Test no reallocation when sufficient
    const ptr = scratch.router_logits.ptr;
    try cpu_common.ensureF32Slice(alloc, &scratch.router_logits, 5);
    try std.testing.expectEqual(@as(usize, 10 + cpu_common.GUARD_F32S), scratch.router_logits.len);
    try std.testing.expectEqual(ptr, scratch.router_logits.ptr);

    // Test reallocation when needed
    try cpu_common.ensureF32Slice(alloc, &scratch.router_logits, 20);
    try std.testing.expectEqual(@as(usize, 20 + cpu_common.GUARD_F32S), scratch.router_logits.len);
}

test "forward top-k tie breaking" {
    const alloc = std.testing.allocator;
    const k = 3;
    // Multiple experts with same scores (6 experts)
    const logits = [_]f32{ 2.0, 5.0, 2.0, 5.0, 2.0, 5.0 };

    const indices = try alloc.alloc(u32, k);
    defer alloc.free(indices);
    const weights = try alloc.alloc(f32, k);
    defer alloc.free(weights);

    try cpu_topk.selectTopKNormalized(&logits, k, indices, weights);

    // Should select first 3 experts with score 5.0 (indices 1, 3, 5)
    try std.testing.expectEqual(@as(u32, 1), indices[0]);
    try std.testing.expectEqual(@as(u32, 3), indices[1]);
    try std.testing.expectEqual(@as(u32, 5), indices[2]);

    // All weights should be equal
    try std.testing.expectApproxEqAbs(1.0 / 3.0, weights[0], 1e-5);
    try std.testing.expectApproxEqAbs(1.0 / 3.0, weights[1], 1e-5);
    try std.testing.expectApproxEqAbs(1.0 / 3.0, weights[2], 1e-5);
}

test "forward router zero input" {
    const alloc = std.testing.allocator;

    const d_model = 3;
    const num_experts = 2;

    var weight_data = [_]f32{
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    };
    var router_weight = Tensor.view2DSlice(&weight_data, d_model, num_experts);

    const zero_input = [_]f32{ 0.0, 0.0, 0.0 };
    const bias = [_]f32{ 0.5, -0.5 };

    const logits = try alloc.alloc(f32, num_experts);
    defer alloc.free(logits);

    cpu_matvec.matVecDense(&zero_input, &router_weight, &bias, logits);

    // With zero input, output should be just bias
    try std.testing.expectApproxEqAbs(0.5, logits[0], 1e-5);
    try std.testing.expectApproxEqAbs(-0.5, logits[1], 1e-5);
}

test "forward top-k small diff" {
    const alloc = std.testing.allocator;
    const k = 2;
    // Very close values (4 experts)
    const logits = [_]f32{ 1.0001, 1.0002, 1.0000, 0.9999 };

    const indices = try alloc.alloc(u32, k);
    defer alloc.free(indices);
    const weights = try alloc.alloc(f32, k);
    defer alloc.free(weights);

    try cpu_topk.selectTopKNormalized(&logits, k, indices, weights);

    // Should select experts 1 and 0 (highest values)
    try std.testing.expectEqual(@as(u32, 1), indices[0]);
    try std.testing.expectEqual(@as(u32, 0), indices[1]);

    // Weights should be very close to 0.5 each
    try std.testing.expectApproxEqAbs(0.5, weights[0], 0.01);
    try std.testing.expectApproxEqAbs(0.5, weights[1], 0.01);

    // Sum should be 1.0
    const weight_sum = weights[0] + weights[1];
    try std.testing.expectApproxEqAbs(1.0, weight_sum, 1e-5);
}

test "deinit frees scratch buffers" {
    const alloc = std.testing.allocator;

    var scratch = MoEScratch{};

    // Allocate all scratch buffers
    try cpu_common.ensureF32Slice(alloc, &scratch.router_logits, 100);
    try cpu_common.ensureF32Slice(alloc, &scratch.expert_weights, 50);
    try cpu_common.ensureU32Slice(alloc, &scratch.expert_indices, 50);
    try cpu_common.ensureF32Slice(alloc, &scratch.expert_outputs, 200);
    try cpu_common.ensureF32Slice(alloc, &scratch.gate_up_values, 150);
    try cpu_common.ensureF32Slice(alloc, &scratch.hidden_values, 75);

    // Verify buffers are allocated (f32 buffers include guard zone, u32 does not).
    try std.testing.expectEqual(@as(usize, 100 + cpu_common.GUARD_F32S), scratch.router_logits.len);
    try std.testing.expectEqual(@as(usize, 50 + cpu_common.GUARD_F32S), scratch.expert_weights.len);
    try std.testing.expectEqual(@as(usize, 50), scratch.expert_indices.len);
    try std.testing.expectEqual(@as(usize, 200 + cpu_common.GUARD_F32S), scratch.expert_outputs.len);
    try std.testing.expectEqual(@as(usize, 150 + cpu_common.GUARD_F32S), scratch.gate_up_values.len);
    try std.testing.expectEqual(@as(usize, 75 + cpu_common.GUARD_F32S), scratch.hidden_values.len);

    // Call deinit
    scratch.deinit(alloc);

    // Verify all buffers are reset to empty slices
    try std.testing.expectEqual(@as(usize, 0), scratch.router_logits.len);
    try std.testing.expectEqual(@as(usize, 0), scratch.expert_weights.len);
    try std.testing.expectEqual(@as(usize, 0), scratch.expert_indices.len);
    try std.testing.expectEqual(@as(usize, 0), scratch.expert_outputs.len);
    try std.testing.expectEqual(@as(usize, 0), scratch.gate_up_values.len);
    try std.testing.expectEqual(@as(usize, 0), scratch.hidden_values.len);
}
