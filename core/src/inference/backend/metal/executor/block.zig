//! Metal backend block executor.
//!
//! Centralizes single-layer lazy graph assembly so model-level orchestration
//! can delegate layer work through a stable `TransformerBlock.forward` surface.

const compute = @import("../../../../compute/root.zig");
const attention_kernel = @import("../kernels/attention.zig");
const ffn_kernel = @import("../kernels/ffn.zig");
const moe_kernel = @import("../kernels/moe.zig");
const norm_kernel = @import("../kernels/norm.zig");
const shortconv_kernel = @import("../kernels/shortconv.zig");
const mlx_graph = compute.metal.graph;

pub const Cache = mlx_graph.Cache;
pub const ShortConvCache = mlx_graph.ShortConvCache;

pub const TransformerBlock = struct {
    pub fn forward(
        hidden: mlx_graph.ArrayHandle,
        layer_weights: anytype,
        layer_idx: usize,
        config: anytype,
        weight_handles: anytype,
        cache: ?Cache,
        shortconv_cache: ?ShortConvCache,
        pos_offset: usize,
        runtime_rope_cos_handle: mlx_graph.ArrayHandle,
        runtime_rope_sin_handle: mlx_graph.ArrayHandle,
        runtime_rope_dim: usize,
    ) !mlx_graph.ArrayHandle {
        const norm_eps = config.norm_eps;
        const head_count: usize = @intCast(config.n_heads);
        const kv_head_count: usize = @intCast(config.n_kv_groups);
        const head_dim: usize = @intCast(config.head_dim);
        const lw = layer_weights.*;

        const attn_norm = norm_kernel.RMSNorm{
            .weight = lw.ln1_weight,
            .eps = norm_eps,
        };
        var normed: mlx_graph.ArrayHandle = undefined;
        attn_norm.forward(hidden, &normed);

        const mixer_out = switch (lw.kind) {
            .attention_mlp => blk: {
                const attention = attention_kernel.MultiHeadAttention{
                    .n_heads = head_count,
                    .n_kv_heads = kv_head_count,
                    .head_dim = head_dim,
                    .rope_theta = config.rope_theta,
                    .norm_eps = norm_eps,
                    .query_pre_attn_scalar = config.query_pre_attn_scalar,
                    .attention_multiplier = weight_handles.attention_multiplier,
                    .q_proj = lw.q_proj,
                    .k_proj = lw.k_proj,
                    .v_proj = lw.v_proj,
                    .o_proj = lw.o_proj,
                    .q_proj_bf16 = lw.q_proj_bf16,
                    .k_proj_bf16 = lw.k_proj_bf16,
                    .v_proj_bf16 = lw.v_proj_bf16,
                    .o_proj_bf16 = lw.o_proj_bf16,
                    .q_norm = lw.q_norm,
                    .k_norm = lw.k_norm,
                    .q_bias = lw.q_bias,
                    .k_bias = lw.k_bias,
                    .v_bias = lw.v_bias,
                    .o_bias = lw.o_bias,
                    .attn_sinks = lw.attn_sinks,
                };
                var attn_cache = attention_kernel.AttnCache{
                    .cache = cache,
                    .layer_idx = layer_idx,
                    .pos_offset = pos_offset,
                };
                var attn_scratch = attention_kernel.AttnTemp{
                    .runtime_rope_cos_handle = runtime_rope_cos_handle,
                    .runtime_rope_sin_handle = runtime_rope_sin_handle,
                    .runtime_rope_dim = runtime_rope_dim,
                };
                var attn_matmul_scratch = attention_kernel.MatmulScratch{};
                var attn_out: mlx_graph.ArrayHandle = undefined;
                try attention.forward(
                    normed,
                    &attn_out,
                    &attn_cache,
                    &attn_scratch,
                    &attn_matmul_scratch,
                    cache != null,
                );
                break :blk attn_out;
            },
            .shortconv => blk: {
                const conv_weight = lw.shortconv_conv_weight orelse return error.MissingField;
                const shortconv = shortconv_kernel.ShortConvKernel{
                    .in_proj = if (lw.shortconv_in_proj) |w| w else null,
                    .out_proj = if (lw.shortconv_out_proj) |w| w else null,
                    .in_proj_bf16 = if (lw.shortconv_in_proj_bf16) |h| h else null,
                    .out_proj_bf16 = if (lw.shortconv_out_proj_bf16) |h| h else null,
                    .conv_weight = conv_weight,
                    .conv_bias = if (lw.shortconv_conv_bias) |b| b else null,
                    .d_conv = lw.shortconv_d_conv,
                    .conv_dim = lw.shortconv_conv_dim,
                };
                var shortconv_state = shortconv_kernel.ShortConvState{
                    .cache = shortconv_cache,
                    .layer_idx = layer_idx,
                };
                var shortconv_scratch = shortconv_kernel.ShortConvScratch{};
                var shortconv_matmul_scratch = shortconv_kernel.MatmulScratch{};
                var shortconv_out: mlx_graph.ArrayHandle = undefined;
                try shortconv.forward(
                    normed,
                    &shortconv_out,
                    &shortconv_state,
                    &shortconv_scratch,
                    &shortconv_matmul_scratch,
                );
                break :blk shortconv_out;
            },
        };

        const attn_for_residual = if (weight_handles.use_post_attn_norm) blk: {
            const post_attn_norm = norm_kernel.RMSNorm{
                .weight = lw.ln2_weight,
                .eps = norm_eps,
            };
            var normed_attn: mlx_graph.ArrayHandle = undefined;
            post_attn_norm.forward(mixer_out, &normed_attn);
            break :blk normed_attn;
        } else mixer_out;
        const scaled_attn = if (weight_handles.residual_multiplier != 1.0)
            mlx_graph.mlx_lazy_multiply_scalar(attn_for_residual, weight_handles.residual_multiplier)
        else
            attn_for_residual;
        const hidden_1 = mlx_graph.mlx_lazy_add(hidden, scaled_attn);

        const ffn_norm_weight = if (weight_handles.use_post_attn_norm and lw.pre_ffn_norm != null)
            lw.pre_ffn_norm.?
        else
            lw.ln2_weight;
        const ffn_norm = norm_kernel.RMSNorm{
            .weight = ffn_norm_weight,
            .eps = norm_eps,
        };
        var normed_2: mlx_graph.ArrayHandle = undefined;
        ffn_norm.forward(hidden_1, &normed_2);

        const ffn_out = if (lw.moe) |moe| blk: {
            const ffn_moe = moe_kernel.MoEFFN{ .weights = moe };
            var moe_scratch = moe_kernel.MoEScratch{};
            var moe_matmul_scratch = moe_kernel.MatmulScratch{};
            var moe_out: mlx_graph.ArrayHandle = undefined;
            try ffn_moe.forward(
                normed_2,
                &moe_out,
                &moe_scratch,
                &moe_matmul_scratch,
            );
            break :blk moe_out;
        } else blk: {
            const swiglu = ffn_kernel.SwiGLU{
                .use_gelu = weight_handles.use_gelu,
                .w1 = lw.w1,
                .w2 = lw.w2,
                .w3 = lw.w3,
                .w1_bf16 = lw.w1_bf16,
                .w2_bf16 = lw.w2_bf16,
                .w3_bf16 = lw.w3_bf16,
            };
            var ffn_scratch = ffn_kernel.FfnScratch{};
            var ffn_matmul_scratch = ffn_kernel.MatmulScratch{};
            var ffn_result: mlx_graph.ArrayHandle = undefined;
            try swiglu.forward(
                normed_2,
                &ffn_result,
                &ffn_scratch,
                &ffn_matmul_scratch,
            );
            break :blk ffn_result;
        };

        const ffn_for_residual = if (lw.post_ffn_norm) |post_ffn| blk: {
            const post_norm = norm_kernel.RMSNorm{
                .weight = post_ffn,
                .eps = norm_eps,
            };
            var normed_ffn: mlx_graph.ArrayHandle = undefined;
            post_norm.forward(ffn_out, &normed_ffn);
            break :blk normed_ffn;
        } else ffn_out;
        const scaled_ffn = if (weight_handles.residual_multiplier != 1.0)
            mlx_graph.mlx_lazy_multiply_scalar(ffn_for_residual, weight_handles.residual_multiplier)
        else
            ffn_for_residual;
        return mlx_graph.mlx_lazy_add(hidden_1, scaled_ffn);
    }

    pub fn projectLogits(
        hidden: mlx_graph.ArrayHandle,
        weight_handles: anytype,
        norm_eps: f32,
    ) mlx_graph.ArrayHandle {
        const final_norm = norm_kernel.RMSNorm{
            .weight = weight_handles.ln_final,
            .eps = norm_eps,
        };
        var final_normed: mlx_graph.ArrayHandle = undefined;
        final_norm.forward(hidden, &final_normed);
        const logits = if (weight_handles.lm_head_quantized) |quantized_lm_head| blk: {
            break :blk mlx_graph.mlx_lazy_quantized_matmul(
                final_normed,
                quantized_lm_head.weights,
                quantized_lm_head.scales,
                quantized_lm_head.biases,
                quantized_lm_head.group_size,
                quantized_lm_head.bits,
                true,
            );
        } else blk: {
            if (weight_handles.lm_head_needs_transpose) {
                const transpose_axes = [_]usize{ 1, 0 };
                const lm_head_t = mlx_graph.mlx_lazy_transpose(weight_handles.lm_head.?, &transpose_axes, 2);
                break :blk mlx_graph.mlx_lazy_matmul(final_normed, lm_head_t);
            } else {
                break :blk mlx_graph.mlx_lazy_matmul(final_normed, weight_handles.lm_head.?);
            }
        };
        return if (weight_handles.logits_scaling != 1.0)
            mlx_graph.mlx_lazy_multiply_scalar(logits, 1.0 / weight_handles.logits_scaling)
        else
            logits;
    }
};
