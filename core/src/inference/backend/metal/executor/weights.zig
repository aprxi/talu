//! Metal weight-loading and fused decode model lifecycle.
//!
//! Owns GPU weight materialization (`WeightHandles`) and fused-model planning
//! for the Metal backend. Forward orchestration lives in `model.zig`.

const std = @import("std");
const log = @import("../../../../log.zig");
const tensor_mod = @import("../../../../tensor.zig");
const Tensor = tensor_mod.Tensor;
const graph_loader = @import("../../../../graph/root.zig");
const dtype_mod = @import("../../../../dtype.zig");
const compute = @import("../../../../compute/root.zig");
const mlx_graph = compute.metal.graph;
const mamba_kernel = @import("../kernels/mamba.zig");
const mla_kernel = @import("../kernels/mla_attention.zig");

const builtin = @import("builtin");
const LoadedModel = graph_loader.LoadedModel;

const ArrayHandle = mlx_graph.ArrayHandle;

pub const MLXError = error{
    UnsupportedDType,
    MLXNotAvailable,
    InvalidTensorType,
    NotQuantized,
    InvalidShape,
    InvalidValue,
    MissingField,
    MissingScales,
    NotImplemented,
    FusedModelRequiresQuantizedEmbeddings,
    FusedModelRequiresQuantizedLMHead,
    DenseModelRequiresEmbeddings,
    DenseModelRequiresLMHead,
    OutOfMemory,
};

fn tensorToArray(tensor: *const Tensor) MLXError!ArrayHandle {
    const shape = tensor.shape[0..@as(usize, @intCast(tensor.n_dims))];
    switch (tensor.dtype) {
        .f32 => return mlx_graph.createArrayF32(tensor.asSlice(f32), shape),
        .bf16 => {
            const element_count = tensor.data_size / 2;
            const data_ptr: [*]align(1) const u16 = @ptrCast(tensor.data_ptr);
            return mlx_graph.createArrayBF16Unaligned(data_ptr, element_count, shape);
        },
        .f16 => {
            const element_count = tensor.data_size / 2;
            const data_ptr: [*]align(1) const u16 = @ptrCast(tensor.data_ptr);
            return mlx_graph.createArrayF16Unaligned(data_ptr, element_count, shape);
        },
        else => return error.UnsupportedDType,
    }
}

/// Load a 1D norm weight tensor to MLX array.
/// Handles f32, bf16, and f16 dtypes.
fn loadNormWeight(weight: *const Tensor) MLXError!ArrayHandle {
    switch (weight.dtype) {
        .f32 => {
            // F32 norms - convert element count correctly
            const element_count = weight.data_size / 4;
            const shape = [_]usize{element_count};
            // Use raw pointer to avoid alignment requirements
            return mlx_graph.mlx_array_from_float32(@ptrCast(weight.data_ptr), &shape, 1);
        },
        .f16 => {
            const element_count = weight.data_size / 2;
            const shape = [_]i64{@intCast(element_count)};
            const data_ptr: [*]align(1) const u16 = @ptrCast(weight.data_ptr);
            return mlx_graph.createArrayF16Unaligned(data_ptr, element_count, &shape);
        },
        .bf16 => {
            const element_count = weight.data_size / 2;
            const shape = [_]i64{@intCast(element_count)};
            const data_ptr: [*]align(1) const u16 = @ptrCast(weight.data_ptr);
            return mlx_graph.createArrayBF16Unaligned(data_ptr, element_count, &shape);
        },
        else => return error.UnsupportedDType,
    }
}

fn isGroupedAffineDType(dtype: dtype_mod.DType) bool {
    return dtype == .grouped_affine_u4 or dtype == .grouped_affine_u8;
}

fn isDenseNumericDType(dtype: dtype_mod.DType) bool {
    return switch (dtype) {
        .bf16, .f16, .f32 => true,
        else => false,
    };
}

const AttentionMixerLayout = enum {
    quantized,
    mixed_qkv_quantized_o_dense,
    dense,
};

fn resolveAttentionMixerLayout(
    q_dtype: dtype_mod.DType,
    k_dtype: dtype_mod.DType,
    v_dtype: dtype_mod.DType,
    o_dtype: dtype_mod.DType,
) !AttentionMixerLayout {
    const qkv_quantized = isGroupedAffineDType(q_dtype) and isGroupedAffineDType(k_dtype) and isGroupedAffineDType(v_dtype);
    const qkv_dense = isDenseNumericDType(q_dtype) and isDenseNumericDType(k_dtype) and isDenseNumericDType(v_dtype);
    if (!qkv_quantized and !qkv_dense) return error.InvalidTensorType;

    if (qkv_quantized) {
        if (isGroupedAffineDType(o_dtype)) return .quantized;
        if (isDenseNumericDType(o_dtype)) return .mixed_qkv_quantized_o_dense;
        return error.InvalidTensorType;
    }

    if (!isDenseNumericDType(o_dtype)) return error.InvalidTensorType;
    return .dense;
}

fn quantBitsFor(dtype: dtype_mod.DType) !usize {
    return switch (dtype) {
        .grouped_affine_u8 => 8,
        .grouped_affine_u4 => 4,
        else => error.InvalidTensorType,
    };
}

/// Compute Llama3-style RoPE frequencies with wavelength-dependent scaling.
/// This implements the same formula as mlx_lm's Llama3RoPE class.
fn computeLlama3RopeFreqs(
    allocator: std.mem.Allocator,
    dims: usize,
    base: f32,
    factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    old_context_len: i32,
) ![]f32 {
    const frequency_count = dims / 2;
    const frequencies = try allocator.alloc(f32, frequency_count);
    errdefer allocator.free(frequencies);

    const original_context: f32 = @floatFromInt(old_context_len);
    const low_frequency_wavelength = original_context / low_freq_factor;
    const high_frequency_wavelength = original_context / high_freq_factor;
    const dims_f32: f32 = @floatFromInt(dims);

    for (0..frequency_count) |freq_index| {
        const angle_index: f32 = @floatFromInt(freq_index * 2);
        // Standard RoPE: freq = base^(2i/dims)
        const frequency = std.math.pow(f32, base, angle_index / dims_f32);
        const wavelength = 2.0 * std.math.pi * frequency;

        if (wavelength > low_frequency_wavelength) {
            // Long wavelengths: scale by factor (for extended context)
            frequencies[freq_index] = frequency * factor;
        } else if (wavelength < high_frequency_wavelength) {
            // Short wavelengths: keep original
            frequencies[freq_index] = frequency;
        } else {
            // Medium wavelengths: smooth interpolation
            const interpolation = (original_context / wavelength - low_freq_factor) / (high_freq_factor - low_freq_factor);
            frequencies[freq_index] = frequency / ((1.0 - interpolation) / factor + interpolation);
        }
    }

    return frequencies;
}

/// Load model weights as MLX array handles on GPU.
/// Call once during initialization; reuse weight_handles for the session lifetime.
pub fn loadWeightsToGPU(allocator: std.mem.Allocator, loaded: *LoadedModel) !*WeightHandles {
    if (builtin.os.tag != .macos) {
        return error.MLXNotAvailable;
    }

    const model_blocks = loaded.blocks;

    // MoE models are now supported
    const is_moe_model = loaded.config.num_experts > 0;
    if (is_moe_model) {
        log.info("inference", "MoE model detected", .{ .num_experts = loaded.config.num_experts, .experts_per_token = loaded.config.experts_per_token });
    }

    var weight_handles = try allocator.create(WeightHandles);
    // Initialize all optional fields to null to avoid undefined memory
    weight_handles.fused_model = null;
    weight_handles.dense_model = null;
    weight_handles.compiled_layers = null;
    weight_handles.is_moe = is_moe_model;
    weight_handles.num_experts = if (is_moe_model) @intCast(loaded.config.num_experts) else 0;
    weight_handles.experts_per_token = if (is_moe_model) @intCast(loaded.config.experts_per_token) else 0;

    // Architecture features from graph definition
    weight_handles.has_norm_weight_offset = loaded.runtime.weight_offset != 0.0;
    weight_handles.use_sqrt_embedding_scale = loaded.runtime.weight_offset != 0.0; // architectures with (1+w) norms also use sqrt scaling
    weight_handles.use_gelu = loaded.config.use_gelu;
    weight_handles.use_post_attn_norm = loaded.runtime.weight_offset != 0.0; // architectures with (1+w) norms use post-attn norm before residual
    weight_handles.d_model = @intCast(loaded.config.d_model);

    // Scaling multipliers from config.json (used by Granite and other models)
    weight_handles.embedding_multiplier = loaded.config.embedding_multiplier;
    weight_handles.attention_multiplier = loaded.config.attention_multiplier;
    weight_handles.residual_multiplier = loaded.config.residual_multiplier;
    weight_handles.logits_scaling = loaded.config.logits_scaling;

    // Log if non-default multipliers are in use
    if (weight_handles.embedding_multiplier != 1.0 or weight_handles.residual_multiplier != 1.0 or weight_handles.logits_scaling != 1.0) {
        log.info("inference", "Custom scaling multipliers", .{
            .embedding_multiplier = weight_handles.embedding_multiplier,
            .attention_multiplier = weight_handles.attention_multiplier,
            .residual_multiplier = weight_handles.residual_multiplier,
            .logits_scaling = weight_handles.logits_scaling,
        });
    }

    errdefer allocator.destroy(weight_handles);

    // Mixed-precision models can have quantized attention layers and dense ShortConv layers.
    // Treat the model as quantized if any global projection table is quantized.
    const lm_head_is_quantized = if (loaded.lm_head) |lm| isGroupedAffineDType(lm.dtype) else false;
    const is_quantized_model = isGroupedAffineDType(loaded.original_weight_dtype) or
        isGroupedAffineDType(loaded.token_embeddings.dtype) or
        lm_head_is_quantized;
    weight_handles.is_quantized = is_quantized_model;

    var has_shortconv_layers = false;
    for (model_blocks) |block| {
        switch (block) {
            .shortconv => has_shortconv_layers = true,
            else => {},
        }
    }
    weight_handles.has_shortconv = has_shortconv_layers;

    // Load embedding weights
    // Keep embeddings quantized - will dequantize during lookup
    weight_handles.embed_tokens_quantized = null;
    weight_handles.embed_tokens = null;
    const embedding_dtype = loaded.token_embeddings.dtype;
    if (isGroupedAffineDType(embedding_dtype)) {
        const quant_bits = try quantBitsFor(embedding_dtype);
        const quantized_weight = try loadQuantizedWeight(&loaded.token_embeddings, quant_bits);
        const quantized_weight_ptr = try allocator.create(WeightHandles.QuantizedWeight);
        quantized_weight_ptr.* = quantized_weight;
        weight_handles.embed_tokens_quantized = quantized_weight_ptr;
    } else {
        // Non-quantized embeddings tensor.
        const embed_shape = loaded.token_embeddings.shape[0..@as(usize, @intCast(loaded.token_embeddings.n_dims))];
        switch (embedding_dtype) {
            .bf16 => {
                const embedding_len = loaded.token_embeddings.data_size / 2;
                const embedding_ptr: [*]align(1) const u16 = @ptrCast(loaded.token_embeddings.data_ptr);
                weight_handles.embed_tokens = mlx_graph.createArrayBF16Unaligned(embedding_ptr, embedding_len, embed_shape);
            },
            .f16 => {
                const embedding_len = loaded.token_embeddings.data_size / 2;
                const embedding_ptr: [*]align(1) const u16 = @ptrCast(loaded.token_embeddings.data_ptr);
                weight_handles.embed_tokens = mlx_graph.createArrayF16Unaligned(embedding_ptr, embedding_len, embed_shape);
            },
            .f32 => {
                weight_handles.embed_tokens = mlx_graph.createArrayF32(loaded.token_embeddings.asSlice(f32), embed_shape);
            },
            else => return error.InvalidTensorType,
        }
    }

    // Load per-layer weights
    weight_handles.layers = try allocator.alloc(WeightHandles.LayerWeights, model_blocks.len);
    errdefer allocator.free(weight_handles.layers);

    // Initialize all layer weights to default values (null for optional fields)
    for (weight_handles.layers) |*layer| {
        layer.* = std.mem.zeroes(WeightHandles.LayerWeights);
    }

    for (model_blocks, 0..) |*block, layer_idx| {
        switch (block.*) {
            .attention_mlp => |*attn_block| {
                if (attn_block.isMLA()) {
                    return mla_kernel.unsupported();
                }
                weight_handles.layers[layer_idx].kind = .attention_mlp;

                // ln1_weight - load in native dtype (bf16, f16, or f32)
                var ln1_arr = try loadNormWeight(attn_block.ln1_weight);
                // (1+w) RMSNorm formulation
                if (weight_handles.has_norm_weight_offset) {
                    ln1_arr = mlx_graph.mlx_add_one(ln1_arr);
                }
                weight_handles.layers[layer_idx].ln1_weight = ln1_arr;

                // ln2_weight - load in native dtype (bf16, f16, or f32)
                var ln2_arr = try loadNormWeight(attn_block.ln2_weight);
                // (1+w) RMSNorm formulation
                if (weight_handles.has_norm_weight_offset) {
                    ln2_arr = mlx_graph.mlx_add_one(ln2_arr);
                }
                weight_handles.layers[layer_idx].ln2_weight = ln2_arr;

                const attention_layout = try resolveAttentionMixerLayout(
                    attn_block.q_proj.?.dtype,
                    attn_block.k_proj.?.dtype,
                    attn_block.v_proj.?.dtype,
                    attn_block.o_proj.dtype,
                );

                switch (attention_layout) {
                    .quantized => {
                        // Detect bits from quantized Q projection dtype.
                        const quant_bits = try quantBitsFor(attn_block.q_proj.?.dtype);
                        weight_handles.layers[layer_idx].is_quantized = true;

                        // Q/K/V/O projections (quantized)
                        weight_handles.layers[layer_idx].q_proj = try loadQuantizedWeight(attn_block.q_proj.?, quant_bits);
                        weight_handles.layers[layer_idx].k_proj = try loadQuantizedWeight(attn_block.k_proj.?, quant_bits);
                        weight_handles.layers[layer_idx].v_proj = try loadQuantizedWeight(attn_block.v_proj.?, quant_bits);
                        weight_handles.layers[layer_idx].o_proj = try loadQuantizedWeight(attn_block.o_proj, quant_bits);
                    },
                    .mixed_qkv_quantized_o_dense => {
                        const quant_bits = try quantBitsFor(attn_block.q_proj.?.dtype);
                        weight_handles.layers[layer_idx].is_quantized = false;

                        // Mixed attention path: quantized Q/K/V with dense output projection.
                        weight_handles.layers[layer_idx].q_proj = try loadQuantizedWeight(attn_block.q_proj.?, quant_bits);
                        weight_handles.layers[layer_idx].k_proj = try loadQuantizedWeight(attn_block.k_proj.?, quant_bits);
                        weight_handles.layers[layer_idx].v_proj = try loadQuantizedWeight(attn_block.v_proj.?, quant_bits);
                        weight_handles.layers[layer_idx].o_proj_bf16 = try tensorToArray(attn_block.o_proj);
                    },
                    .dense => {
                        weight_handles.layers[layer_idx].is_quantized = false;

                        // Dense attention path (BF16/F16/F32).
                        weight_handles.layers[layer_idx].q_proj_bf16 = try tensorToArray(attn_block.q_proj.?);
                        weight_handles.layers[layer_idx].k_proj_bf16 = try tensorToArray(attn_block.k_proj.?);
                        weight_handles.layers[layer_idx].v_proj_bf16 = try tensorToArray(attn_block.v_proj.?);
                        weight_handles.layers[layer_idx].o_proj_bf16 = try tensorToArray(attn_block.o_proj);
                    },
                }

                // FFN weights can be quantized or dense independently of attention projections.
                if (attn_block.w1) |w1| {
                    if (isGroupedAffineDType(w1.dtype)) {
                        weight_handles.layers[layer_idx].w1 = try loadQuantizedWeight(w1, try quantBitsFor(w1.dtype));
                    } else {
                        weight_handles.layers[layer_idx].w1_bf16 = try tensorToArray(w1);
                    }
                }
                if (attn_block.w2) |w2| {
                    if (isGroupedAffineDType(w2.dtype)) {
                        weight_handles.layers[layer_idx].w2 = try loadQuantizedWeight(w2, try quantBitsFor(w2.dtype));
                    } else {
                        weight_handles.layers[layer_idx].w2_bf16 = try tensorToArray(w2);
                    }
                }
                if (attn_block.w3) |w3| {
                    if (isGroupedAffineDType(w3.dtype)) {
                        weight_handles.layers[layer_idx].w3 = try loadQuantizedWeight(w3, try quantBitsFor(w3.dtype));
                    } else {
                        weight_handles.layers[layer_idx].w3_bf16 = try tensorToArray(w3);
                    }
                }

                // Optional attention biases/sinks are loaded during weight loading
                // and stored in the loader-stage block weight_handles.
                if (attn_block.q_bias) |b| weight_handles.layers[layer_idx].q_bias = mlx_graph.createArrayF32(b, &[_]i64{@intCast(b.len)});
                if (attn_block.k_bias) |b| weight_handles.layers[layer_idx].k_bias = mlx_graph.createArrayF32(b, &[_]i64{@intCast(b.len)});
                if (attn_block.v_bias) |b| weight_handles.layers[layer_idx].v_bias = mlx_graph.createArrayF32(b, &[_]i64{@intCast(b.len)});
                if (attn_block.o_bias) |b| weight_handles.layers[layer_idx].o_bias = mlx_graph.createArrayF32(b, &[_]i64{@intCast(b.len)});
                if (attn_block.sinks) |s| weight_handles.layers[layer_idx].attn_sinks = mlx_graph.createArrayF32(s, &[_]i64{@intCast(s.len)});

                // QK normalization (optional) - load in native dtype (f32, f16, or bf16)
                if (attn_block.q_norm) |q_norm_tensor| {
                    var q_norm_arr = try loadNormWeight(q_norm_tensor);
                    // (1+w) RMSNorm formulation
                    if (weight_handles.has_norm_weight_offset) {
                        q_norm_arr = mlx_graph.mlx_add_one(q_norm_arr);
                    }
                    weight_handles.layers[layer_idx].q_norm = q_norm_arr;
                } else {
                    weight_handles.layers[layer_idx].q_norm = null;
                }

                if (attn_block.k_norm) |k_norm_tensor| {
                    var k_norm_arr = try loadNormWeight(k_norm_tensor);
                    // (1+w) RMSNorm formulation
                    if (weight_handles.has_norm_weight_offset) {
                        k_norm_arr = mlx_graph.mlx_add_one(k_norm_arr);
                    }
                    weight_handles.layers[layer_idx].k_norm = k_norm_arr;
                } else {
                    weight_handles.layers[layer_idx].k_norm = null;
                }

                // Extra FFN layer norms (4 norms per block) - optional
                if (attn_block.pre_ffn_norm) |pre_ffn_norm_tensor| {
                    var norm_arr = try loadNormWeight(pre_ffn_norm_tensor);
                    // (1+w) RMSNorm formulation
                    if (weight_handles.has_norm_weight_offset) {
                        norm_arr = mlx_graph.mlx_add_one(norm_arr);
                    }
                    weight_handles.layers[layer_idx].pre_ffn_norm = norm_arr;
                } else {
                    weight_handles.layers[layer_idx].pre_ffn_norm = null;
                }

                if (attn_block.post_ffn_norm) |post_ffn_norm_tensor| {
                    var norm_arr = try loadNormWeight(post_ffn_norm_tensor);
                    // (1+w) RMSNorm formulation
                    if (weight_handles.has_norm_weight_offset) {
                        norm_arr = mlx_graph.mlx_add_one(norm_arr);
                    }
                    weight_handles.layers[layer_idx].post_ffn_norm = norm_arr;
                } else {
                    weight_handles.layers[layer_idx].post_ffn_norm = null;
                }

                // Initialize MoE weights to null (will be loaded separately for MoE models)
                weight_handles.layers[layer_idx].moe = null;
            },
            .mamba => {
                // Mamba layers not yet supported on Metal GPU - requires CPU fallback
                return mamba_kernel.unsupported();
            },
            .shortconv => |*shortconv_block| {
                weight_handles.layers[layer_idx].kind = .shortconv;

                // ln1_weight - load in native dtype (bf16, f16, or f32)
                var ln1_arr = try loadNormWeight(shortconv_block.ln1_weight);
                if (weight_handles.has_norm_weight_offset) {
                    ln1_arr = mlx_graph.mlx_add_one(ln1_arr);
                }
                weight_handles.layers[layer_idx].ln1_weight = ln1_arr;

                // ShortConv uses ffn_norm (ln2 equivalent). Require it for consistent FFN path.
                const ln2_tensor = shortconv_block.ln2_weight orelse return error.MissingField;
                var ln2_arr = try loadNormWeight(ln2_tensor);
                if (weight_handles.has_norm_weight_offset) {
                    ln2_arr = mlx_graph.mlx_add_one(ln2_arr);
                }
                weight_handles.layers[layer_idx].ln2_weight = ln2_arr;

                weight_handles.layers[layer_idx].shortconv_d_conv = @intCast(shortconv_block.config.d_conv);
                weight_handles.layers[layer_idx].shortconv_conv_dim = @intCast(shortconv_block.config.conv_dim);
                weight_handles.layers[layer_idx].shortconv_conv_weight = try tensorToArray(shortconv_block.weights.conv1d_weight);
                if (shortconv_block.weights.conv1d_bias) |conv_bias_tensor| {
                    weight_handles.layers[layer_idx].shortconv_conv_bias = try tensorToArray(conv_bias_tensor);
                }

                const mixer_is_quantized = isGroupedAffineDType(shortconv_block.weights.in_proj.dtype);
                weight_handles.layers[layer_idx].is_quantized = mixer_is_quantized;
                if (mixer_is_quantized) {
                    weight_handles.layers[layer_idx].shortconv_in_proj = try loadQuantizedWeight(
                        shortconv_block.weights.in_proj,
                        try quantBitsFor(shortconv_block.weights.in_proj.dtype),
                    );
                    weight_handles.layers[layer_idx].shortconv_out_proj = try loadQuantizedWeight(
                        shortconv_block.weights.out_proj,
                        try quantBitsFor(shortconv_block.weights.out_proj.dtype),
                    );
                } else {
                    weight_handles.layers[layer_idx].shortconv_in_proj_bf16 = try tensorToArray(shortconv_block.weights.in_proj);
                    weight_handles.layers[layer_idx].shortconv_out_proj_bf16 = try tensorToArray(shortconv_block.weights.out_proj);
                }

                // ShortConv FFN can be quantized or dense independently of the mixer projections.
                if (shortconv_block.w1) |w1| {
                    if (isGroupedAffineDType(w1.dtype)) {
                        weight_handles.layers[layer_idx].w1 = try loadQuantizedWeight(w1, try quantBitsFor(w1.dtype));
                    } else {
                        weight_handles.layers[layer_idx].w1_bf16 = try tensorToArray(w1);
                    }
                }
                if (shortconv_block.w2) |w2| {
                    if (isGroupedAffineDType(w2.dtype)) {
                        weight_handles.layers[layer_idx].w2 = try loadQuantizedWeight(w2, try quantBitsFor(w2.dtype));
                    } else {
                        weight_handles.layers[layer_idx].w2_bf16 = try tensorToArray(w2);
                    }
                }
                if (shortconv_block.w3) |w3| {
                    if (isGroupedAffineDType(w3.dtype)) {
                        weight_handles.layers[layer_idx].w3 = try loadQuantizedWeight(w3, try quantBitsFor(w3.dtype));
                    } else {
                        weight_handles.layers[layer_idx].w3_bf16 = try tensorToArray(w3);
                    }
                }

                weight_handles.layers[layer_idx].moe = null;
            },
        }
    }

    // Load MoE weights for each layer (already sanitized by model hooks).
    if (is_moe_model) {
        for (0..model_blocks.len) |layer_idx| {
            const moe_block_weights = switch (model_blocks[layer_idx]) {
                .attention_mlp => |attn| attn.moe_weights orelse continue,
                .mamba, .shortconv => continue, // Mamba/ShortConv layers don't have MoE
            };
            if (!moe_block_weights.use_mxfp4) return error.NotImplemented;
            const num_experts: usize = moe_block_weights.experts.len;
            if (num_experts == 0) return error.InvalidValue;

            const moe_weights = try allocator.create(WeightHandles.MoEWeights);
            errdefer allocator.destroy(moe_weights);

            // Initialize with defaults
            moe_weights.* = .{
                .router_w = undefined,
                .gate_w = undefined,
                .gate_s = undefined,
                .up_w = undefined,
                .up_s = undefined,
                .down_w = undefined,
                .down_s = undefined,
                .router_group_size = 0, // unused for non-quantized router
                .expert_group_size = 32,
                .num_experts = num_experts,
                .experts_per_token = moe_block_weights.experts_per_token,
            };

            // Router weights are stored for CPU as f32 [d_model, num_experts].
            // MLX fused op expects router_w shaped [num_experts, d_model] and transposes internally.
            if (moe_block_weights.router_weight.n_dims != 2 or moe_block_weights.router_weight.dtype != .f32) return error.InvalidShape;
            const router_shape = moe_block_weights.router_weight.shape[0..@intCast(moe_block_weights.router_weight.n_dims)];
            var router_arr = mlx_graph.createArrayF32(moe_block_weights.router_weight.asSlice(f32), router_shape);
            const router_axes = [_]usize{ 1, 0 };
            router_arr = mlx_graph.mlx_lazy_transpose(router_arr, &router_axes, 2);
            moe_weights.router_w = router_arr;
            moe_weights.router_s = null;
            moe_weights.router_b = null;

            if (moe_block_weights.router_bias) |router_bias| {
                moe_weights.router_bias = mlx_graph.createArrayF32(router_bias, &[_]i64{@intCast(router_bias.len)});
            }

            const expert0 = moe_block_weights.experts[0];
            const is_mlx_format = expert0.gate_proj != null;

            // Load expert weights
            if (is_mlx_format) {
                // MLX format: separate gate/up/down projections
                const gate_proj0 = expert0.gate_proj orelse return error.MissingField;
                const up_proj0 = expert0.up_proj orelse return error.MissingField;
                if (gate_proj0.n_dims != 2 or up_proj0.n_dims != 2) return error.InvalidShape;

                const d_ff: usize = @intCast(gate_proj0.shape[0]);
                const gate_packed_dim: usize = @intCast(@divExact(gate_proj0.shape[1], 4));
                const up_packed_dim: usize = @intCast(@divExact(up_proj0.shape[1], 4));

                const gate_shape_i64 = [_]i64{ @intCast(num_experts), @intCast(d_ff), @intCast(gate_packed_dim) };
                moe_weights.gate_w = mlx_graph.createArrayU32Unaligned(
                    @as([*]align(1) const u32, @ptrCast(gate_proj0.data_ptr)),
                    (gate_proj0.data_size * num_experts) / 4,
                    &gate_shape_i64,
                );
                const gate_scales0 = expert0.gate_scales orelse return error.MissingScales;
                const gate_groups: usize = gate_scales0.len / d_ff;
                const gate_scales_shape = [_]usize{ num_experts, d_ff, gate_groups };
                moe_weights.gate_s = mlx_graph.mlx_array_from_uint8(gate_scales0.ptr, &gate_scales_shape, 3);

                const up_shape_i64 = [_]i64{ @intCast(num_experts), @intCast(d_ff), @intCast(up_packed_dim) };
                moe_weights.up_w = mlx_graph.createArrayU32Unaligned(
                    @as([*]align(1) const u32, @ptrCast(up_proj0.data_ptr)),
                    (up_proj0.data_size * num_experts) / 4,
                    &up_shape_i64,
                );
                const up_scales0 = expert0.up_scales orelse return error.MissingScales;
                const up_groups: usize = up_scales0.len / d_ff;
                const up_scales_shape = [_]usize{ num_experts, d_ff, up_groups };
                moe_weights.up_s = mlx_graph.mlx_array_from_uint8(up_scales0.ptr, &up_scales_shape, 3);

                if (expert0.gate_bias) |gate_bias| {
                    const bias_shape_i64 = [_]i64{ @intCast(num_experts), @intCast(d_ff) };
                    const bias_f32_ptr = @as([*]const f32, @ptrCast(gate_bias.ptr));
                    moe_weights.gate_bias = mlx_graph.createArrayF32(bias_f32_ptr[0 .. gate_bias.len * num_experts], &bias_shape_i64);
                }
                if (expert0.up_bias) |up_bias| {
                    const bias_shape_i64 = [_]i64{ @intCast(num_experts), @intCast(d_ff) };
                    const bias_f32_ptr = @as([*]const f32, @ptrCast(up_bias.ptr));
                    moe_weights.up_bias = mlx_graph.createArrayF32(bias_f32_ptr[0 .. up_bias.len * num_experts], &bias_shape_i64);
                }
            } else {
                // HF format: fused gate_up_proj with INTERLEAVED rows (not concatenated!)
                // gate_up_proj_blocks: [num_experts, 2*d_ff, num_groups, block_size] uint8
                // Row 0 = gate[0], Row 1 = up[0], Row 2 = gate[1], Row 3 = up[1], etc.
                // We need to de-interleave: reshape [E, 2*D, P] -> [E, D, 2, P] then slice
                const gate_up0 = expert0.gate_up_proj orelse return error.MissingField;
                if (gate_up0.n_dims != 2) return error.InvalidShape;

                const d_ff_times_2_u: usize = @intCast(gate_up0.shape[0]);
                const d_ff_u: usize = d_ff_times_2_u / 2;
                const packed_dim: usize = @intCast(@divExact(gate_up0.shape[1], 4));
                const n_experts_u: usize = num_experts;

                // Create as [num_experts, 2*d_ff, packed_dim] uint32
                // Note: createArrayU32Unaligned needs i64 shape, extern functions need usize
                const u32_shape_i64 = [_]i64{ @intCast(n_experts_u), @intCast(d_ff_times_2_u), @intCast(packed_dim) };
                const fused_w = mlx_graph.createArrayU32Unaligned(
                    @as([*]align(1) const u32, @ptrCast(gate_up0.data_ptr)),
                    (gate_up0.data_size * num_experts) / 4,
                    &u32_shape_i64,
                );

                // Reshape [E, 2*D, P] -> [E, D, 2, P] to group interleaved rows
                const reshaped_shape = [_]usize{ n_experts_u, d_ff_u, 2, packed_dim };
                const reshaped_w = mlx_graph.mlx_lazy_reshape(fused_w, &reshaped_shape, 4);

                // Slice to de-interleave: gate = [:, :, 0, :], up = [:, :, 1, :]
                const n_experts: c_int = @intCast(n_experts_u);
                const d_ff: c_int = @intCast(d_ff_u);
                const packed_dim_c: c_int = @intCast(packed_dim);

                // gate_w: [E, D, 0:1, P] -> squeeze to [E, D, P]
                const gate_w_start = [_]c_int{ 0, 0, 0, 0 };
                const gate_w_end = [_]c_int{ n_experts, d_ff, 1, packed_dim_c };
                const gate_w_4d = mlx_graph.mlx_lazy_slice(reshaped_w, &gate_w_start, &gate_w_end, 4);
                const gate_w_shape = [_]usize{ n_experts_u, d_ff_u, packed_dim };
                // Use persistent_reshape for final weight - must survive pool resets
                moe_weights.gate_w = mlx_graph.mlx_persistent_reshape(gate_w_4d, &gate_w_shape, 3);

                // up_w: [E, D, 1:2, P] -> squeeze to [E, D, P]
                const up_w_start = [_]c_int{ 0, 0, 1, 0 };
                const up_w_end = [_]c_int{ n_experts, d_ff, 2, packed_dim_c };
                const up_w_4d = mlx_graph.mlx_lazy_slice(reshaped_w, &up_w_start, &up_w_end, 4);
                const up_w_shape = [_]usize{ n_experts_u, d_ff_u, packed_dim };
                moe_weights.up_w = mlx_graph.mlx_persistent_reshape(up_w_4d, &up_w_shape, 3);

                const gate_up_scales0 = expert0.gate_up_scales orelse return error.MissingScales;
                const s_num_groups_u: usize = gate_up_scales0.len / d_ff_times_2_u;
                const fused_s_shape = [_]usize{ n_experts_u, d_ff_times_2_u, s_num_groups_u };
                const fused_s = mlx_graph.mlx_array_from_uint8(gate_up_scales0.ptr, &fused_s_shape, 3);

                // Reshape [E, 2*D, G] -> [E, D, 2, G] then slice
                const s_reshaped_shape = [_]usize{ n_experts_u, d_ff_u, 2, s_num_groups_u };
                const reshaped_s = mlx_graph.mlx_lazy_reshape(fused_s, &s_reshaped_shape, 4);

                const s_num_groups: c_int = @intCast(s_num_groups_u);
                const gate_s_start = [_]c_int{ 0, 0, 0, 0 };
                const gate_s_end = [_]c_int{ n_experts, d_ff, 1, s_num_groups };
                const gate_s_4d = mlx_graph.mlx_lazy_slice(reshaped_s, &gate_s_start, &gate_s_end, 4);
                const gate_s_shape = [_]usize{ n_experts_u, d_ff_u, s_num_groups_u };
                moe_weights.gate_s = mlx_graph.mlx_persistent_reshape(gate_s_4d, &gate_s_shape, 3);

                const up_s_start = [_]c_int{ 0, 0, 1, 0 };
                const up_s_end = [_]c_int{ n_experts, d_ff, 2, s_num_groups };
                const up_s_4d = mlx_graph.mlx_lazy_slice(reshaped_s, &up_s_start, &up_s_end, 4);
                const up_s_shape = [_]usize{ n_experts_u, d_ff_u, s_num_groups_u };
                moe_weights.up_s = mlx_graph.mlx_persistent_reshape(up_s_4d, &up_s_shape, 3);

                if (expert0.gate_up_bias) |gate_up_bias| {
                    const fused_bias_shape_i64 = [_]i64{ @intCast(n_experts_u), @intCast(d_ff_times_2_u) };
                    const bias_f32_ptr = @as([*]const f32, @ptrCast(gate_up_bias.ptr));
                    const fused_bias = mlx_graph.createArrayF32(bias_f32_ptr[0 .. gate_up_bias.len * num_experts], &fused_bias_shape_i64);

                    // Reshape [E, 2*D] -> [E, D, 2] then slice
                    const bias_reshaped_shape = [_]usize{ n_experts_u, d_ff_u, 2 };
                    const reshaped_bias = mlx_graph.mlx_lazy_reshape(fused_bias, &bias_reshaped_shape, 3);

                    const bias_gate_start = [_]c_int{ 0, 0, 0 };
                    const bias_gate_end = [_]c_int{ n_experts, d_ff, 1 };
                    const gate_bias_3d = mlx_graph.mlx_lazy_slice(reshaped_bias, &bias_gate_start, &bias_gate_end, 3);
                    const bias_shape = [_]usize{ n_experts_u, d_ff_u };
                    moe_weights.gate_bias = mlx_graph.mlx_persistent_reshape(gate_bias_3d, &bias_shape, 2);

                    const bias_up_start = [_]c_int{ 0, 0, 1 };
                    const bias_up_end = [_]c_int{ n_experts, d_ff, 2 };
                    const up_bias_3d = mlx_graph.mlx_lazy_slice(reshaped_bias, &bias_up_start, &bias_up_end, 3);
                    moe_weights.up_bias = mlx_graph.mlx_persistent_reshape(up_bias_3d, &bias_shape, 2);
                }
            }

            // Down projection (MXFP4) - uses already-loaded expert weight_handles.
            if (expert0.down_proj.n_dims != 2) return error.InvalidShape;
            const down_rows: usize = @intCast(expert0.down_proj.shape[0]);
            const down_packed_dim: usize = @intCast(@divExact(expert0.down_proj.shape[1], 4));
            const down_shape_i64 = [_]i64{ @intCast(num_experts), @intCast(down_rows), @intCast(down_packed_dim) };
            moe_weights.down_w = mlx_graph.createArrayU32Unaligned(
                @as([*]align(1) const u32, @ptrCast(expert0.down_proj.data_ptr)),
                (expert0.down_proj.data_size * num_experts) / 4,
                &down_shape_i64,
            );
            const down_scales0 = expert0.down_scales orelse return error.MissingScales;
            const down_groups: usize = down_scales0.len / down_rows;
            const down_scales_shape = [_]usize{ num_experts, down_rows, down_groups };
            moe_weights.down_s = mlx_graph.mlx_array_from_uint8(down_scales0.ptr, &down_scales_shape, 3);
            if (expert0.down_bias) |down_bias| {
                const bias_shape_i64 = [_]i64{ @intCast(num_experts), @intCast(down_rows) };
                const bias_f32_ptr = @as([*]const f32, @ptrCast(down_bias.ptr));
                moe_weights.down_bias = mlx_graph.createArrayF32(bias_f32_ptr[0 .. down_bias.len * num_experts], &bias_shape_i64);
            }

            weight_handles.layers[layer_idx].moe = moe_weights;
        }
    }

    // Load final layer norm - in native dtype (bf16, f16, or f32)
    if (loaded.ln_final) |ln_f| {
        var ln_final_arr = try loadNormWeight(&ln_f);
        // (1+w) RMSNorm formulation
        if (weight_handles.has_norm_weight_offset) {
            ln_final_arr = mlx_graph.mlx_add_one(ln_final_arr);
        }
        weight_handles.ln_final = ln_final_arr;
    }

    // Load LM head (optional — embed-only models may not have one)
    weight_handles.lm_head_quantized = null;
    weight_handles.lm_head = null;

    if (loaded.lm_head) |lm_head_tensor| {
        // Detect lm_head orientation from shape:
        // - PyTorch nn.Linear stores weights as (out_features, in_features) = (vocab_size, d_model)
        // - For matmul: hidden (B, S, d_model) @ lm_head needs lm_head to be (d_model, vocab_size)
        // - If shape[0] == vocab_size, we need to transpose; if shape[0] == d_model, already correct
        const lm_shape = lm_head_tensor.shape[0..@as(usize, @intCast(lm_head_tensor.n_dims))];
        const d_model: usize = @intCast(loaded.config.d_model);
        const vocab_size: usize = @intCast(loaded.config.vocab_size);
        weight_handles.lm_head_needs_transpose = if (lm_shape.len >= 2)
            (lm_shape[0] == vocab_size and lm_shape[1] == d_model)
        else
            false;

        if (isGroupedAffineDType(lm_head_tensor.dtype)) {
            // Quantized lm_head
            const quantized_lm_head = try loadQuantizedWeight(&lm_head_tensor, try quantBitsFor(lm_head_tensor.dtype));
            const quantized_lm_head_ptr = try allocator.create(WeightHandles.QuantizedWeight);
            quantized_lm_head_ptr.* = quantized_lm_head;
            weight_handles.lm_head_quantized = quantized_lm_head_ptr;
        } else {
            // Dense lm_head tensor.
            switch (lm_head_tensor.dtype) {
                .bf16 => {
                    const lm_len = lm_head_tensor.data_size / 2;
                    const lm_ptr: [*]align(1) const u16 = @ptrCast(lm_head_tensor.data_ptr);
                    weight_handles.lm_head = mlx_graph.createArrayBF16Unaligned(lm_ptr, lm_len, lm_shape);
                },
                .f16 => {
                    const lm_len = lm_head_tensor.data_size / 2;
                    const lm_ptr: [*]align(1) const u16 = @ptrCast(lm_head_tensor.data_ptr);
                    weight_handles.lm_head = mlx_graph.createArrayF16Unaligned(lm_ptr, lm_len, lm_shape);
                },
                .f32 => {
                    weight_handles.lm_head = mlx_graph.createArrayF32(lm_head_tensor.asSlice(f32), lm_shape);
                },
                else => return error.InvalidTensorType,
            }
        }
    }

    // Initialize compiled layers to null (will be compiled on first use or via compileLayersForFusion)
    weight_handles.compiled_layers = null;

    // Initialize fused model to null (will be created via createFusedModel)
    weight_handles.fused_model = null;

    return weight_handles;
}

/// Create a fully fused decode model (all layers in one C++ call) to
/// reduce per-token FFI overhead.
/// Supports both quantized (4-bit/8-bit) and dense (BF16) models.
const FusedDecodeModelKind = enum {
    none,
    quantized,
    dense,
};

fn classifyFusedDecodeModel(weight_handles: *const WeightHandles) FusedDecodeModelKind {
    // IMPORTANT: keep this selection fully data-driven from traced layer kinds and tensor dtypes.
    // Do not branch on model names here. Naming belongs in tools/archs and traced graphs.
    var all_layers_quantized = true;
    var all_layers_dense = true;

    for (weight_handles.layers) |layer| {
        switch (layer.kind) {
            .attention_mlp => {
                const quantized_attention_complete = layer.q_proj != null and
                    layer.k_proj != null and
                    layer.v_proj != null and
                    layer.o_proj != null;
                const quantized_ffn_complete = layer.w1 != null and layer.w2 != null and layer.w3 != null;
                const dense_attention_complete = layer.q_proj_bf16 != null and
                    layer.k_proj_bf16 != null and
                    layer.v_proj_bf16 != null and
                    layer.o_proj_bf16 != null;
                const dense_ffn_complete = layer.w1_bf16 != null and layer.w2_bf16 != null and layer.w3_bf16 != null;

                const layer_quantized = quantized_attention_complete and quantized_ffn_complete;
                const layer_dense = dense_attention_complete and dense_ffn_complete;

                // Fused kernels only support one consistent representation across all layers.
                if (layer_quantized == layer_dense) return .none;

                if (!layer_quantized) all_layers_quantized = false;
                if (!layer_dense) all_layers_dense = false;
            },
            .shortconv => {
                const layer_quantized = layer.shortconv_in_proj != null and
                    layer.shortconv_out_proj != null and
                    layer.shortconv_conv_weight != null and
                    layer.w1 != null and
                    layer.w2 != null and
                    layer.w3 != null;
                const layer_dense = layer.shortconv_in_proj_bf16 != null and
                    layer.shortconv_out_proj_bf16 != null and
                    layer.shortconv_conv_weight != null and
                    layer.w1_bf16 != null and
                    layer.w2_bf16 != null and
                    layer.w3_bf16 != null;

                // Fused kernels only support one consistent representation across all layers.
                if (layer_quantized == layer_dense) return .none;

                if (!layer_quantized) all_layers_quantized = false;
                if (!layer_dense) all_layers_dense = false;
            },
        }
    }

    if (all_layers_quantized) return .quantized;
    if (all_layers_dense) return .dense;
    return .none;
}

const QuantizedLayout = struct {
    group_size: usize,
    bits: usize,
};

fn quantizedFusedLayout(weight_handles: *const WeightHandles) ?QuantizedLayout {
    var group_size_opt: ?usize = null;
    var quant_bits_opt: ?usize = null;

    for (weight_handles.layers) |layer| {
        const candidate = switch (layer.kind) {
            .attention_mlp => if (layer.q_proj) |q| q else continue,
            .shortconv => if (layer.shortconv_in_proj) |sc| sc else continue,
        };
        if (group_size_opt == null) {
            group_size_opt = candidate.group_size;
            quant_bits_opt = candidate.bits;
        } else if (group_size_opt.? != candidate.group_size or quant_bits_opt.? != candidate.bits) {
            // Fused quantized kernels currently require a single grouped-affine layout.
            return null;
        }
    }

    return .{
        .group_size = group_size_opt orelse return null,
        .bits = quant_bits_opt orelse return null,
    };
}

pub fn createFusedModel(allocator: std.mem.Allocator, weight_handles: *WeightHandles, config: anytype) !void {
    if (weight_handles.fused_model != null) return; // Already created
    if (weight_handles.dense_model != null) return; // Already created dense

    // MoE fused decode is not implemented yet.
    if (weight_handles.is_moe) return;

    const model_kind = classifyFusedDecodeModel(weight_handles);
    if (model_kind == .none) return;

    const layer_count: usize = @intCast(config.n_layers);
    const head_count: usize = @intCast(config.n_heads);
    const kv_head_count: usize = @intCast(config.n_kv_groups);
    const head_dim: usize = @intCast(config.head_dim);
    const model_width: usize = @intCast(config.d_model);

    if (model_kind == .quantized) {
        // QUANTIZED PATH: Use FusedModelWeights with quantized_matmul
        const quant_layout = quantizedFusedLayout(weight_handles) orelse return;

        const fused = mlx_graph.mlx_fused_model_create(
            layer_count,
            head_count,
            kv_head_count,
            head_dim,
            model_width,
            quant_layout.group_size,
            quant_layout.bits,
            config.rope_theta,
            config.norm_eps,
        );

        // Set embeddings (must be quantized for fused model)
        if (weight_handles.embed_tokens_quantized) |quantized_weight| {
            mlx_graph.mlx_fused_model_set_embeddings(fused, quantized_weight.weights, quantized_weight.scales, quantized_weight.biases);
        } else {
            return error.FusedModelRequiresQuantizedEmbeddings;
        }

        // Set final weights
        if (weight_handles.lm_head_quantized) |quantized_weight| {
            mlx_graph.mlx_fused_model_set_final(fused, weight_handles.ln_final, quantized_weight.weights, quantized_weight.scales, quantized_weight.biases);
        } else {
            return error.FusedModelRequiresQuantizedLMHead;
        }

        // Set per-layer weights
        for (weight_handles.layers, 0..) |*layer, layer_idx| {
            switch (layer.kind) {
                .attention_mlp => {
                    mlx_graph.mlx_fused_model_set_layer(
                        fused,
                        layer_idx,
                        layer.ln1_weight,
                        layer.q_proj.?.weights,
                        layer.q_proj.?.scales,
                        layer.q_proj.?.biases,
                        layer.k_proj.?.weights,
                        layer.k_proj.?.scales,
                        layer.k_proj.?.biases,
                        layer.v_proj.?.weights,
                        layer.v_proj.?.scales,
                        layer.v_proj.?.biases,
                        layer.o_proj.?.weights,
                        layer.o_proj.?.scales,
                        layer.o_proj.?.biases,
                        layer.ln2_weight,
                        layer.w1.?.weights, // gate
                        layer.w1.?.scales,
                        layer.w1.?.biases,
                        layer.w3.?.weights, // up
                        layer.w3.?.scales,
                        layer.w3.?.biases,
                        layer.w2.?.weights, // down
                        layer.w2.?.scales,
                        layer.w2.?.biases,
                        if (layer.q_norm) |qn| qn else null,
                        if (layer.k_norm) |kn| kn else null,
                        if (layer.pre_ffn_norm) |n| n else null,
                        if (layer.post_ffn_norm) |n| n else null,
                        0, // attention_mlp
                        0,
                        0,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                    );
                },
                .shortconv => {
                    const sc_in = layer.shortconv_in_proj orelse return;
                    const sc_out = layer.shortconv_out_proj orelse return;
                    mlx_graph.mlx_fused_model_set_layer(
                        fused,
                        layer_idx,
                        layer.ln1_weight,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        layer.ln2_weight,
                        layer.w1.?.weights, // gate
                        layer.w1.?.scales,
                        layer.w1.?.biases,
                        layer.w3.?.weights, // up
                        layer.w3.?.scales,
                        layer.w3.?.biases,
                        layer.w2.?.weights, // down
                        layer.w2.?.scales,
                        layer.w2.?.biases,
                        null,
                        null,
                        if (layer.pre_ffn_norm) |n| n else null,
                        if (layer.post_ffn_norm) |n| n else null,
                        1, // shortconv
                        layer.shortconv_d_conv,
                        layer.shortconv_conv_dim,
                        sc_in.weights,
                        sc_in.scales,
                        sc_in.biases,
                        sc_out.weights,
                        sc_out.scales,
                        sc_out.biases,
                        layer.shortconv_conv_weight.?,
                        if (layer.shortconv_conv_bias) |b| b else null,
                    );
                },
            }
        }

        weight_handles.fused_model = fused;

        // Set architecture-specific config for fused model
        if (weight_handles.has_norm_weight_offset or config.use_gelu or config.query_pre_attn_scalar != 0) {
            mlx_graph.mlx_fused_model_set_arch_config(
                fused,
                weight_handles.has_norm_weight_offset,
                config.use_gelu,
                config.query_pre_attn_scalar,
            );
        }

        // Set custom scaling multipliers if any are non-default
        const has_custom_scaling = config.embedding_multiplier != 1.0 or
            config.attention_multiplier != 0.0 or
            config.residual_multiplier != 1.0 or
            config.logits_scaling != 1.0;
        if (has_custom_scaling) {
            mlx_graph.mlx_fused_model_set_scaling_config(
                fused,
                config.embedding_multiplier,
                config.attention_multiplier,
                config.residual_multiplier,
                config.logits_scaling,
            );
        }

        // Set Llama3 RoPE frequencies if configured
        if (config.rope_scaling.rope_type == .llama3) {
            log.info("inference", "Setting Llama3 RoPE frequencies", .{
                .factor = config.rope_scaling.factor,
                .low_freq_factor = config.rope_scaling.low_freq_factor,
                .high_freq_factor = config.rope_scaling.high_freq_factor,
            });
            const freqs = computeLlama3RopeFreqs(
                allocator,
                @intCast(config.head_dim),
                config.rope_theta,
                config.rope_scaling.factor,
                config.rope_scaling.low_freq_factor,
                config.rope_scaling.high_freq_factor,
                @intCast(config.rope_scaling.original_max_position_embeddings),
            ) catch {
                log.warn("inference", "Failed to compute Llama3 RoPE frequencies, using standard RoPE", .{});
                return;
            };
            defer allocator.free(freqs);
            log.info("inference", "Computed rope frequencies", .{
                .count = freqs.len,
                .first = freqs[0],
                .last = freqs[freqs.len - 1],
            });
            const freqs_array = mlx_graph.createArrayF32(freqs, &[_]i64{@intCast(freqs.len)});
            mlx_graph.mlx_fused_model_set_rope_freqs(fused, freqs_array);
        }

        // Pre-evaluate all weights to ensure GPU transfer happens upfront
        mlx_graph.mlx_fused_model_optimize(fused);
    } else {
        // DENSE PATH: Use FusedDenseModel with dense matmul (BF16)
        const dense = mlx_graph.mlx_dense_model_create(
            layer_count,
            head_count,
            kv_head_count,
            head_dim,
            model_width,
            config.rope_theta,
            config.norm_eps,
        );

        // Set embeddings (BF16)
        if (weight_handles.embed_tokens) |embedding_handle| {
            mlx_graph.mlx_dense_model_set_embeddings(dense, embedding_handle);
        } else {
            return error.DenseModelRequiresEmbeddings;
        }

        // Set final weights (BF16)
        if (weight_handles.lm_head) |lm_head_handle| {
            mlx_graph.mlx_dense_model_set_final(dense, weight_handles.ln_final, lm_head_handle);
        } else {
            return error.DenseModelRequiresLMHead;
        }

        // Set per-layer weights (BF16)
        for (weight_handles.layers, 0..) |*layer, layer_idx| {
            switch (layer.kind) {
                .attention_mlp => mlx_graph.mlx_dense_model_set_layer(
                    dense,
                    layer_idx,
                    layer.ln1_weight,
                    layer.q_proj_bf16.?,
                    layer.k_proj_bf16.?,
                    layer.v_proj_bf16.?,
                    layer.o_proj_bf16.?,
                    layer.ln2_weight,
                    layer.w1_bf16.?, // gate
                    layer.w3_bf16.?, // up
                    layer.w2_bf16.?, // down
                    if (layer.q_norm) |qn| qn else null,
                    if (layer.k_norm) |kn| kn else null,
                    0, // attention_mlp
                    0,
                    0,
                    null,
                    null,
                    null,
                    null,
                ),
                .shortconv => {
                    var in_shape: [8]usize = undefined;
                    var out_shape: [8]usize = undefined;
                    const in_rank = mlx_graph.getShape(layer.shortconv_in_proj_bf16.?, &in_shape);
                    const out_rank = mlx_graph.getShape(layer.shortconv_out_proj_bf16.?, &out_shape);
                    log.debug("inference", "Dense shortconv fused layer", .{
                        .layer_idx = layer_idx,
                        .in_rank = in_rank,
                        .in0 = in_shape[0],
                        .in1 = if (in_rank > 1) in_shape[1] else 0,
                        .out_rank = out_rank,
                        .out0 = out_shape[0],
                        .out1 = if (out_rank > 1) out_shape[1] else 0,
                        .d_conv = layer.shortconv_d_conv,
                        .conv_dim = layer.shortconv_conv_dim,
                    }, @src());
                    mlx_graph.mlx_dense_model_set_layer(
                        dense,
                        layer_idx,
                        layer.ln1_weight,
                        null,
                        null,
                        null,
                        null,
                        layer.ln2_weight,
                        layer.w1_bf16.?, // gate
                        layer.w3_bf16.?, // up
                        layer.w2_bf16.?, // down
                        null,
                        null,
                        1, // shortconv
                        layer.shortconv_d_conv,
                        layer.shortconv_conv_dim,
                        layer.shortconv_in_proj_bf16.?,
                        layer.shortconv_conv_weight.?,
                        if (layer.shortconv_conv_bias) |b| b else null,
                        layer.shortconv_out_proj_bf16.?,
                    );
                },
            }
        }

        weight_handles.dense_model = dense;
    }
}

/// Free all GPU weight handles.
pub fn freeWeights(allocator: std.mem.Allocator, weight_handles: *WeightHandles) void {
    if (weight_handles.embed_tokens) |embedding_handle| {
        mlx_graph.freeArray(embedding_handle);
    }
    if (weight_handles.embed_tokens_quantized) |quantized_weight| {
        freeQuantizedWeight(quantized_weight.*);
        allocator.destroy(quantized_weight);
    }

    for (weight_handles.layers) |*layer| {
        if (layer.ln1_weight) |h| mlx_graph.freeArray(h);
        if (layer.ln2_weight) |h| mlx_graph.freeArray(h);

        // ShortConv mixer fields
        if (layer.shortconv_in_proj) |quantized_weight| freeQuantizedWeight(quantized_weight);
        if (layer.shortconv_out_proj) |quantized_weight| freeQuantizedWeight(quantized_weight);
        if (layer.shortconv_in_proj_bf16) |h| mlx_graph.freeArray(h);
        if (layer.shortconv_out_proj_bf16) |h| mlx_graph.freeArray(h);
        if (layer.shortconv_conv_weight) |h| mlx_graph.freeArray(h);
        if (layer.shortconv_conv_bias) |h| mlx_graph.freeArray(h);

        // Free quantized weights if present
        if (layer.q_proj) |quantized_weight| freeQuantizedWeight(quantized_weight);
        if (layer.k_proj) |quantized_weight| freeQuantizedWeight(quantized_weight);
        if (layer.v_proj) |quantized_weight| freeQuantizedWeight(quantized_weight);
        if (layer.o_proj) |quantized_weight| freeQuantizedWeight(quantized_weight);
        // Free BF16 weights if present
        if (layer.q_proj_bf16) |h| mlx_graph.freeArray(h);
        if (layer.k_proj_bf16) |h| mlx_graph.freeArray(h);
        if (layer.v_proj_bf16) |h| mlx_graph.freeArray(h);
        if (layer.o_proj_bf16) |h| mlx_graph.freeArray(h);
        // FFN quantized
        if (layer.w1) |quantized_weight| freeQuantizedWeight(quantized_weight);
        if (layer.w2) |quantized_weight| freeQuantizedWeight(quantized_weight);
        if (layer.w3) |quantized_weight| freeQuantizedWeight(quantized_weight);
        // FFN BF16
        if (layer.w1_bf16) |h| mlx_graph.freeArray(h);
        if (layer.w2_bf16) |h| mlx_graph.freeArray(h);
        if (layer.w3_bf16) |h| mlx_graph.freeArray(h);
        if (layer.q_bias) |h| mlx_graph.freeArray(h);
        if (layer.k_bias) |h| mlx_graph.freeArray(h);
        if (layer.v_bias) |h| mlx_graph.freeArray(h);
        if (layer.o_bias) |h| mlx_graph.freeArray(h);
        if (layer.attn_sinks) |h| mlx_graph.freeArray(h);
        if (layer.q_norm) |qn| mlx_graph.freeArray(qn);
        if (layer.k_norm) |kn| mlx_graph.freeArray(kn);
        if (layer.pre_ffn_norm) |h| mlx_graph.freeArray(h);
        if (layer.post_ffn_norm) |h| mlx_graph.freeArray(h);
    }
    allocator.free(weight_handles.layers);

    mlx_graph.freeArray(weight_handles.ln_final);
    if (weight_handles.lm_head) |lm_head_handle| mlx_graph.freeArray(lm_head_handle);
    if (weight_handles.lm_head_quantized) |quantized_weight| {
        freeQuantizedWeight(quantized_weight.*);
        allocator.destroy(quantized_weight);
    }

    // Free fused model if created
    if (weight_handles.fused_model) |fm| {
        mlx_graph.mlx_fused_model_free(fm);
    }
    // Free dense model if created
    if (weight_handles.dense_model) |dm| {
        mlx_graph.mlx_dense_model_free(dm);
    }

    allocator.destroy(weight_handles);
}

fn loadQuantizedWeight(tensor: *const Tensor, bits: usize) !WeightHandles.QuantizedWeight {
    const gaffine_meta = tensor.gaffine orelse return error.NotQuantized;

    // Weights are packed uint32
    // Note: tensor.shape has been modified to unpacked dimensions by weights loader
    // We need to reconstruct the packed shape from the actual data
    // Use align(1) pointer cast - data may be unaligned from mmap
    const packed_weights_ptr = @as([*]align(1) const u32, @ptrCast(tensor.data_ptr));
    const packed_word_count = tensor.data_size / @sizeOf(u32);

    // Packed shape: [n, k/8] for 4-bit, [n, k/4] for 8-bit (8 or 4 values per u32)
    const row_count: usize = @intCast(tensor.shape[0]);
    const packed_cols = packed_word_count / row_count; // Calculate from actual data
    const packed_shape = [_]usize{ row_count, packed_cols };

    // Call extern directly with unaligned pointer
    const packed_weights = mlx_graph.mlx_array_from_uint32(
        packed_weights_ptr,
        &packed_shape,
        2,
    );

    // Scales and biases are f16/bf16 (stored as u16)
    // MLX quantized_matmul expects 2D: [n, k_unpacked/group_size]
    // Weight is packed: packed_shape[1] * 32 / bits = unpacked dimension
    const unpacked_cols = packed_cols * 32 / bits;
    const group_count = unpacked_cols / gaffine_meta.group_size;

    // Use align(1) for scales/biases too - mmap data may be unaligned
    const scales_ptr = @as([*]align(1) const u16, @ptrCast(gaffine_meta.scales.ptr));
    const scales_shape = [_]usize{ row_count, group_count };

    // Use correct dtype (F16 or BF16) based on model
    const scales = if (gaffine_meta.scales_dtype == .f16)
        mlx_graph.mlx_array_from_float16(scales_ptr, &scales_shape, 2)
    else
        mlx_graph.mlx_array_from_bfloat16(scales_ptr, &scales_shape, 2);

    const biases_ptr = @as([*]align(1) const u16, @ptrCast(gaffine_meta.biases.ptr));
    const biases = if (gaffine_meta.scales_dtype == .f16)
        mlx_graph.mlx_array_from_float16(biases_ptr, &scales_shape, 2)
    else
        mlx_graph.mlx_array_from_bfloat16(biases_ptr, &scales_shape, 2);

    return .{
        .weights = packed_weights,
        .scales = scales,
        .biases = biases,
        .group_size = gaffine_meta.group_size,
        .bits = bits,
    };
}

fn freeQuantizedWeight(quantized_weight: WeightHandles.QuantizedWeight) void {
    mlx_graph.freeArray(quantized_weight.weights);
    mlx_graph.freeArray(quantized_weight.scales);
    mlx_graph.freeArray(quantized_weight.biases);
}

/// Compile transformer layers for fusion optimization.
/// Call once after loadWeightsToGPU().
/// Note: only works for quantized models; BF16 models skip compilation.
fn compileLayersForFusion(allocator: std.mem.Allocator, weight_handles: *WeightHandles, config: anytype) !void {
    if (weight_handles.compiled_layers != null) return; // Already compiled
    if (!weight_handles.is_quantized) return; // BF16 models don't use compiled layers
    // ShortConv layers require conv-state cache side effects that this compiled layer path
    // does not model. Keep this guard to avoid incorrect decode behavior and silent regressions.
    if (weight_handles.has_shortconv) return;

    const head_count: usize = @intCast(config.n_heads);
    const kv_head_count: usize = @intCast(config.n_kv_groups);
    const head_dim: usize = @intCast(config.head_dim);
    const model_width: usize = @intCast(config.d_model);
    const norm_epsilon = config.norm_eps;
    const rope_theta = config.rope_theta;

    const compiled_layers = try allocator.alloc(mlx_graph.CompiledLayer, weight_handles.layers.len);
    errdefer allocator.free(compiled_layers);

    for (weight_handles.layers, 0..) |*layer, layer_idx| {
        if (layer.kind != .attention_mlp) return;
        const compiled_handle = mlx_graph.mlx_compile_layer(
            layer.q_proj.?.weights,
            layer.q_proj.?.scales,
            layer.q_proj.?.biases,
            layer.k_proj.?.weights,
            layer.k_proj.?.scales,
            layer.k_proj.?.biases,
            layer.v_proj.?.weights,
            layer.v_proj.?.scales,
            layer.v_proj.?.biases,
            layer.o_proj.?.weights,
            layer.o_proj.?.scales,
            layer.o_proj.?.biases,
            layer.w1.?.weights,
            layer.w1.?.scales,
            layer.w1.?.biases, // gate
            layer.w3.?.weights,
            layer.w3.?.scales,
            layer.w3.?.biases, // up
            layer.w2.?.weights,
            layer.w2.?.scales,
            layer.w2.?.biases, // down
            layer.ln1_weight,
            layer.ln2_weight,
            layer.q_norm orelse null,
            layer.k_norm orelse null,
            head_count,
            kv_head_count,
            head_dim,
            model_width,
            layer.q_proj.?.group_size,
            layer.q_proj.?.bits,
            rope_theta,
            norm_epsilon,
        );
        compiled_layers[layer_idx] = .{ .handle = compiled_handle };
    }

    weight_handles.compiled_layers = compiled_layers;
}

/// Weight handles: weights loaded as MLX arrays, kept on GPU.
pub const WeightHandles = struct {
    // Embeddings (either quantized, bf16, or f32)
    embed_tokens: ?ArrayHandle, // f32/bf16 embeddings
    embed_tokens_quantized: ?*QuantizedWeight, // Quantized embeddings

    // Per-layer weights
    layers: []LayerWeights,

    // Compiled layer functions (for fusion optimization)
    compiled_layers: ?[]mlx_graph.CompiledLayer,

    // Fully fused model (all layers in one C++ call - ZERO FFI overhead)
    fused_model: mlx_graph.FusedModelHandle,
    // Dense model for BF16 weights (non-quantized)
    dense_model: mlx_graph.DenseModelHandle = null,

    // Final
    ln_final: ArrayHandle,
    lm_head: ?ArrayHandle, // F32/BF16 lm_head
    lm_head_quantized: ?*QuantizedWeight, // Quantized lm_head
    lm_head_needs_transpose: bool = true, // True if lm_head is (vocab_size, d_model), needs transpose for matmul

    // Track if model is quantized (affects which forward path to use)
    is_quantized: bool = true,

    // MoE configuration
    is_moe: bool = false,
    has_shortconv: bool = false,
    num_experts: usize = 0,
    experts_per_token: usize = 0,

    // Norm weight offset: some architectures use (1+w) formulation for RMSNorm
    has_norm_weight_offset: bool = false,
    // Embedding scaling: some architectures scale embeddings by sqrt(d_model)
    use_sqrt_embedding_scale: bool = false,
    d_model: usize = 0,
    // Activation function for FFN
    use_gelu: bool = false,
    // Post-attention norm applied to attention output before residual (vs pre-FFN norm)
    use_post_attn_norm: bool = false,

    // Scaling multipliers (data-driven from config.json)
    embedding_multiplier: f32 = 1.0,
    attention_multiplier: f32 = 0.0, // 0 means use default 1/sqrt(head_dim)
    residual_multiplier: f32 = 1.0,
    logits_scaling: f32 = 1.0,

    /// MoE weight structure for MXFP4 quantized experts
    /// Supports both MLX format (quantized router, separate gate/up) and
    /// Hub/OpenAI format (BF16 router, fused gate_up)
    pub const MoEWeights = struct {
        // Router weights
        // For MLX format: 8-bit affine quantized (router_w=U32, router_s/router_b=BF16)
        // For HF format: BF16 unquantized (router_w=BF16, router_s/router_b=null)
        router_w: ArrayHandle,
        router_s: ?ArrayHandle = null, // null for BF16 router
        router_b: ?ArrayHandle = null, // null for BF16 router
        router_bias: ?ArrayHandle = null, // Optional linear layer bias

        // Expert weights (MXFP4) - separate gate/up/down
        gate_w: ArrayHandle, // [num_experts, d_ff, packed_dim]
        gate_s: ArrayHandle, // scales
        up_w: ArrayHandle,
        up_s: ArrayHandle,
        down_w: ArrayHandle,
        down_s: ArrayHandle,

        // Expert biases (optional)
        gate_bias: ?ArrayHandle = null, // [num_experts, d_ff]
        up_bias: ?ArrayHandle = null,
        down_bias: ?ArrayHandle = null,

        router_group_size: usize = 64, // 64 for 8-bit router
        expert_group_size: usize = 32, // 32 for MXFP4
        num_experts: usize,
        experts_per_token: usize,
    };

    pub const LayerWeights = struct {
        pub const LayerKind = enum {
            attention_mlp,
            shortconv,
        };

        kind: LayerKind = .attention_mlp,
        ln1_weight: ArrayHandle,
        ln2_weight: ArrayHandle,

        // ShortConv mixer fields (used when kind == .shortconv)
        shortconv_d_conv: usize = 0,
        shortconv_conv_dim: usize = 0,
        shortconv_in_proj: ?QuantizedWeight = null,
        shortconv_out_proj: ?QuantizedWeight = null,
        shortconv_in_proj_bf16: ?ArrayHandle = null,
        shortconv_out_proj_bf16: ?ArrayHandle = null,
        shortconv_conv_weight: ?ArrayHandle = null,
        shortconv_conv_bias: ?ArrayHandle = null,

        // Quantized weights (grouped-affine u4/u8)
        q_proj: ?QuantizedWeight = null,
        k_proj: ?QuantizedWeight = null,
        v_proj: ?QuantizedWeight = null,
        o_proj: ?QuantizedWeight = null,
        // Linear biases for attention (optional)
        q_bias: ?ArrayHandle = null,
        k_bias: ?ArrayHandle = null,
        v_bias: ?ArrayHandle = null,
        o_bias: ?ArrayHandle = null,
        // Attention sinks - per-head scaling for attention
        attn_sinks: ?ArrayHandle = null,
        // BF16 weights (non-quantized)
        q_proj_bf16: ?ArrayHandle = null,
        k_proj_bf16: ?ArrayHandle = null,
        v_proj_bf16: ?ArrayHandle = null,
        o_proj_bf16: ?ArrayHandle = null,
        // FFN - quantized
        w1: ?QuantizedWeight = null,
        w2: ?QuantizedWeight = null,
        w3: ?QuantizedWeight = null,
        // FFN - BF16
        w1_bf16: ?ArrayHandle = null,
        w2_bf16: ?ArrayHandle = null,
        w3_bf16: ?ArrayHandle = null,
        // QK normalization (optional)
        q_norm: ?ArrayHandle = null,
        k_norm: ?ArrayHandle = null,
        // FFN norms (4 norms per block) - optional (4-norm architectures)
        pre_ffn_norm: ?ArrayHandle = null,
        post_ffn_norm: ?ArrayHandle = null,
        // Track if this layer is quantized
        is_quantized: bool = true,
        // MoE weights (for MoE models, replaces w1/w2/w3)
        moe: ?*MoEWeights = null,
    };

    pub const QuantizedWeight = struct {
        weights: ArrayHandle, // Packed uint32
        scales: ArrayHandle, // BF16
        biases: ArrayHandle, // BF16
        group_size: usize,
        bits: usize,
    };
};

// =============================================================================
// Unit Tests
// =============================================================================

const testing = std.testing;
const metal_device = compute.metal.device;
const block_kernels = @import("../../cpu/executor/weights.zig");
const model_executor = @import("model.zig");

test "resolveAttentionMixerLayout returns quantized for fully quantized attention projections" {
    const layout = try resolveAttentionMixerLayout(
        .grouped_affine_u4,
        .grouped_affine_u4,
        .grouped_affine_u4,
        .grouped_affine_u4,
    );
    try testing.expectEqual(AttentionMixerLayout.quantized, layout);
}

test "resolveAttentionMixerLayout returns mixed_qkv_quantized_o_dense for quantized qkv and dense out" {
    const layout = try resolveAttentionMixerLayout(
        .grouped_affine_u4,
        .grouped_affine_u4,
        .grouped_affine_u4,
        .bf16,
    );
    try testing.expectEqual(AttentionMixerLayout.mixed_qkv_quantized_o_dense, layout);
}

test "resolveAttentionMixerLayout returns dense for fully dense attention projections" {
    const layout = try resolveAttentionMixerLayout(.bf16, .bf16, .bf16, .bf16);
    try testing.expectEqual(AttentionMixerLayout.dense, layout);
}

test "resolveAttentionMixerLayout rejects mixed qkv dtypes" {
    try testing.expectError(
        error.InvalidTensorType,
        resolveAttentionMixerLayout(.grouped_affine_u4, .bf16, .grouped_affine_u4, .bf16),
    );
}

fn testHandle(id: usize) ArrayHandle {
    return @as(ArrayHandle, @ptrFromInt(id));
}

fn testQuantizedWeight(id_base: usize, group_size: usize, bits: usize) WeightHandles.QuantizedWeight {
    return .{
        .weights = testHandle(id_base),
        .scales = testHandle(id_base + 1),
        .biases = testHandle(id_base + 2),
        .group_size = group_size,
        .bits = bits,
    };
}

test "classifyFusedDecodeModel returns quantized for mixed attention/shortconv quantized layers" {
    var layers = [_]WeightHandles.LayerWeights{
        .{
            .kind = .attention_mlp,
            .ln1_weight = testHandle(1),
            .ln2_weight = testHandle(2),
            .q_proj = testQuantizedWeight(10, 64, 4),
            .k_proj = testQuantizedWeight(20, 64, 4),
            .v_proj = testQuantizedWeight(30, 64, 4),
            .o_proj = testQuantizedWeight(40, 64, 4),
            .w1 = testQuantizedWeight(50, 64, 4),
            .w2 = testQuantizedWeight(60, 64, 4),
            .w3 = testQuantizedWeight(70, 64, 4),
        },
        .{
            .kind = .shortconv,
            .ln1_weight = testHandle(3),
            .ln2_weight = testHandle(4),
            .shortconv_d_conv = 3,
            .shortconv_conv_dim = 1024,
            .shortconv_in_proj = testQuantizedWeight(80, 64, 4),
            .shortconv_out_proj = testQuantizedWeight(90, 64, 4),
            .shortconv_conv_weight = testHandle(100),
            .w1 = testQuantizedWeight(110, 64, 4),
            .w2 = testQuantizedWeight(120, 64, 4),
            .w3 = testQuantizedWeight(130, 64, 4),
        },
    };
    var weight_handles = WeightHandles{
        .embed_tokens = null,
        .embed_tokens_quantized = null,
        .layers = &layers,
        .compiled_layers = null,
        .fused_model = null,
        .ln_final = testHandle(5),
        .lm_head = null,
        .lm_head_quantized = null,
    };

    try testing.expectEqual(FusedDecodeModelKind.quantized, classifyFusedDecodeModel(&weight_handles));
}

test "classifyFusedDecodeModel returns dense for mixed attention/shortconv dense layers" {
    var layers = [_]WeightHandles.LayerWeights{
        .{
            .kind = .attention_mlp,
            .ln1_weight = testHandle(1),
            .ln2_weight = testHandle(2),
            .q_proj_bf16 = testHandle(10),
            .k_proj_bf16 = testHandle(11),
            .v_proj_bf16 = testHandle(12),
            .o_proj_bf16 = testHandle(13),
            .w1_bf16 = testHandle(14),
            .w2_bf16 = testHandle(15),
            .w3_bf16 = testHandle(16),
        },
        .{
            .kind = .shortconv,
            .ln1_weight = testHandle(20),
            .ln2_weight = testHandle(21),
            .shortconv_d_conv = 3,
            .shortconv_conv_dim = 1024,
            .shortconv_in_proj_bf16 = testHandle(22),
            .shortconv_out_proj_bf16 = testHandle(23),
            .shortconv_conv_weight = testHandle(24),
            .w1_bf16 = testHandle(25),
            .w2_bf16 = testHandle(26),
            .w3_bf16 = testHandle(27),
        },
    };
    var weight_handles = WeightHandles{
        .embed_tokens = null,
        .embed_tokens_quantized = null,
        .layers = &layers,
        .compiled_layers = null,
        .fused_model = null,
        .ln_final = testHandle(30),
        .lm_head = null,
        .lm_head_quantized = null,
    };

    try testing.expectEqual(FusedDecodeModelKind.dense, classifyFusedDecodeModel(&weight_handles));
}

test "quantizedFusedLayout rejects mixed quantization layout" {
    var layers = [_]WeightHandles.LayerWeights{
        .{
            .kind = .attention_mlp,
            .ln1_weight = testHandle(1),
            .ln2_weight = testHandle(2),
            .q_proj = testQuantizedWeight(10, 64, 4),
            .k_proj = testQuantizedWeight(20, 64, 4),
            .v_proj = testQuantizedWeight(30, 64, 4),
            .o_proj = testQuantizedWeight(40, 64, 4),
            .w1 = testQuantizedWeight(50, 64, 4),
            .w2 = testQuantizedWeight(60, 64, 4),
            .w3 = testQuantizedWeight(70, 64, 4),
        },
        .{
            .kind = .shortconv,
            .ln1_weight = testHandle(3),
            .ln2_weight = testHandle(4),
            .shortconv_d_conv = 3,
            .shortconv_conv_dim = 1024,
            .shortconv_in_proj = testQuantizedWeight(80, 128, 4),
            .shortconv_out_proj = testQuantizedWeight(90, 128, 4),
            .shortconv_conv_weight = testHandle(100),
            .w1 = testQuantizedWeight(110, 64, 4),
            .w2 = testQuantizedWeight(120, 64, 4),
            .w3 = testQuantizedWeight(130, 64, 4),
        },
    };
    var weight_handles = WeightHandles{
        .embed_tokens = null,
        .embed_tokens_quantized = null,
        .layers = &layers,
        .compiled_layers = null,
        .fused_model = null,
        .ln_final = testHandle(5),
        .lm_head = null,
        .lm_head_quantized = null,
    };

    try testing.expect(quantizedFusedLayout(&weight_handles) == null);
}

/// Helper to create a minimal test LoadedModel with BF16 weights (non-quantized path)
/// This creates a tiny 1-layer model for testing GPU weight loading
fn createTestLoadedModel(allocator: std.mem.Allocator) !*LoadedModel {
    var arena = std.heap.ArenaAllocator.init(allocator);
    const arena_alloc = arena.allocator();

    // Model dimensions (tiny for testing)
    const vocab_size: usize = 32;
    const d_model: usize = 16;
    const n_heads: usize = 2;
    const head_dim: usize = 8;
    const d_ff: usize = 32;

    // Create embedding tensor [vocab_size, d_model] as f32
    const embed_shape = [_]usize{ vocab_size, d_model };
    var embed_tensor = try tensor_mod.OwnedTensor.init(arena_alloc, .f32, &embed_shape);
    const embed_data = embed_tensor.asSlice(f32);
    for (embed_data, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 10)) * 0.1;
    }

    // Create ln_final tensor [d_model]
    const ln_shape = [_]usize{d_model};
    var ln_final_tensor = try tensor_mod.OwnedTensor.init(arena_alloc, .f32, &ln_shape);
    const ln_data = ln_final_tensor.asSlice(f32);
    for (ln_data) |*v| v.* = 1.0;

    // Create lm_head tensor [d_model, vocab_size] - MLX matmul expects this shape
    // for hidden @ lm_head where hidden is [batch, seq, d_model]
    const lm_head_shape = [_]usize{ d_model, vocab_size };
    var lm_head_tensor = try tensor_mod.OwnedTensor.init(arena_alloc, .f32, &lm_head_shape);
    const lm_head_data = lm_head_tensor.asSlice(f32);
    for (lm_head_data, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 5)) * 0.2;
    }

    // Create single layer block weights
    // ln1_weight [d_model]
    var ln1_tensor = try tensor_mod.OwnedTensor.init(arena_alloc, .f32, &ln_shape);
    for (ln1_tensor.asSlice(f32)) |*v| v.* = 1.0;

    // ln2_weight [d_model]
    var ln2_tensor = try tensor_mod.OwnedTensor.init(arena_alloc, .f32, &ln_shape);
    for (ln2_tensor.asSlice(f32)) |*v| v.* = 1.0;

    // q_proj [d_model, n_heads * head_dim]
    const proj_shape = [_]usize{ d_model, n_heads * head_dim };
    var q_proj = try tensor_mod.OwnedTensor.init(arena_alloc, .f32, &proj_shape);
    for (q_proj.asSlice(f32), 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 3)) * 0.1;

    var k_proj = try tensor_mod.OwnedTensor.init(arena_alloc, .f32, &proj_shape);
    for (k_proj.asSlice(f32), 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 3)) * 0.1;

    var v_proj = try tensor_mod.OwnedTensor.init(arena_alloc, .f32, &proj_shape);
    for (v_proj.asSlice(f32), 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 3)) * 0.1;

    var o_proj = try tensor_mod.OwnedTensor.init(arena_alloc, .f32, &proj_shape);
    for (o_proj.asSlice(f32), 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 3)) * 0.1;

    // FFN weights - shapes follow PyTorch Linear convention: [out_features, in_features]
    // The fused FFN function transposes internally for matmul
    const ffn_gate_shape = [_]usize{ d_ff, d_model }; // gate_proj/w1: [d_ff, d_model]
    var w1 = try tensor_mod.OwnedTensor.init(arena_alloc, .f32, &ffn_gate_shape);
    for (w1.asSlice(f32), 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 4)) * 0.1;

    const ffn_down_shape = [_]usize{ d_model, d_ff }; // down_proj/w2: [d_model, d_ff]
    var w2 = try tensor_mod.OwnedTensor.init(arena_alloc, .f32, &ffn_down_shape);
    for (w2.asSlice(f32), 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 4)) * 0.1;

    var w3 = try tensor_mod.OwnedTensor.init(arena_alloc, .f32, &ffn_gate_shape); // up_proj: same as gate
    for (w3.asSlice(f32), 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 4)) * 0.1;

    // Store tensor views for block weights
    const ln1_view = try arena_alloc.create(tensor_mod.Tensor);
    ln1_view.* = ln1_tensor.toTensor();
    const ln2_view = try arena_alloc.create(tensor_mod.Tensor);
    ln2_view.* = ln2_tensor.toTensor();
    const q_view = try arena_alloc.create(tensor_mod.Tensor);
    q_view.* = q_proj.toTensor();
    const k_view = try arena_alloc.create(tensor_mod.Tensor);
    k_view.* = k_proj.toTensor();
    const v_view = try arena_alloc.create(tensor_mod.Tensor);
    v_view.* = v_proj.toTensor();
    const o_view = try arena_alloc.create(tensor_mod.Tensor);
    o_view.* = o_proj.toTensor();
    const w1_view = try arena_alloc.create(tensor_mod.Tensor);
    w1_view.* = w1.toTensor();
    const w2_view = try arena_alloc.create(tensor_mod.Tensor);
    w2_view.* = w2.toTensor();
    const w3_view = try arena_alloc.create(tensor_mod.Tensor);
    w3_view.* = w3.toTensor();

    // Create block weights array
    const blocks = try arena_alloc.alloc(block_kernels.BlockWeights, 1);
    blocks[0] = .{ .attention_mlp = .{
        .ln1_weight = ln1_view,
        .ln2_weight = ln2_view,
        .q_proj = q_view,
        .k_proj = k_view,
        .v_proj = v_view,
        .o_proj = o_view,
        .w1 = w1_view,
        .w2 = w2_view,
        .w3 = w3_view,
    } };

    // Create LoadedModel
    const loaded = try allocator.create(LoadedModel);
    loaded.* = .{
        .arena = arena,
        .config = .{
            .vocab_size = @intCast(vocab_size),
            .d_model = @intCast(d_model),
            .n_layers = 1,
            .n_heads = @intCast(n_heads),
            .n_kv_groups = @intCast(n_heads),
            .d_ff = @intCast(d_ff),
            .max_seq_len = 128,
            .head_dim = @intCast(head_dim),
            .rope_theta = 10000.0,
            .norm_eps = 1e-5,
            .gaffine_group_size = 64,
        },
        .ln_final = ln_final_tensor.toTensor(),
        .lm_head = lm_head_tensor.toTensor(),
        .token_embeddings = embed_tensor.toTensor(),
        .blocks = blocks,
        .original_weight_dtype = .f32, // Non-quantized
    };

    return loaded;
}

fn destroyTestLoadedModel(allocator: std.mem.Allocator, loaded: *LoadedModel) void {
    loaded.arena.deinit();
    allocator.destroy(loaded);
}

test "loadWeightsToGPU loads f32 model weights to GPU" {
    if (comptime builtin.os.tag != .macos) return;
    if (!metal_device.isAvailable()) return;

    const loaded = try createTestLoadedModel(testing.allocator);
    defer destroyTestLoadedModel(testing.allocator, loaded);

    // Load weights to GPU
    const weight_handles = try loadWeightsToGPU(testing.allocator, loaded);
    defer freeWeights(testing.allocator, weight_handles);

    // Verify weight handles structure
    try testing.expect(weight_handles.embed_tokens != null);
    try testing.expect(!weight_handles.is_quantized);
    try testing.expect(!weight_handles.is_moe);
    try testing.expect(weight_handles.layers.len == 1);
    try testing.expect(weight_handles.ln_final != null);
    try testing.expect(weight_handles.lm_head != null);

    // Verify layer weights loaded
    const layer = weight_handles.layers[0];
    try testing.expect(layer.ln1_weight != null);
    try testing.expect(layer.ln2_weight != null);
    try testing.expect(layer.q_proj_bf16 != null or layer.q_proj != null);
}

test "loadWeightsToGPU sets d_model correctly" {
    if (comptime builtin.os.tag != .macos) return;
    if (!metal_device.isAvailable()) return;

    const loaded = try createTestLoadedModel(testing.allocator);
    defer destroyTestLoadedModel(testing.allocator, loaded);

    const weight_handles = try loadWeightsToGPU(testing.allocator, loaded);
    defer freeWeights(testing.allocator, weight_handles);

    try testing.expectEqual(@as(usize, 16), weight_handles.d_model);
}

test "freeWeights releases all GPU resources" {
    if (comptime builtin.os.tag != .macos) return;
    if (!metal_device.isAvailable()) return;

    const loaded = try createTestLoadedModel(testing.allocator);
    defer destroyTestLoadedModel(testing.allocator, loaded);

    // Load weights
    const weight_handles = try loadWeightsToGPU(testing.allocator, loaded);

    // Free should not crash and should release all resources
    freeWeights(testing.allocator, weight_handles);

    // If we get here without crash, test passes
}

test "createFusedModel skips when fused_model already exists" {
    if (comptime builtin.os.tag != .macos) return;
    if (!metal_device.isAvailable()) return;

    const loaded = try createTestLoadedModel(testing.allocator);
    defer destroyTestLoadedModel(testing.allocator, loaded);

    const weight_handles = try loadWeightsToGPU(testing.allocator, loaded);
    defer freeWeights(testing.allocator, weight_handles);

    // Set a sentinel value for fused_model
    const original_fused = weight_handles.fused_model;
    weight_handles.fused_model = @ptrFromInt(0xDEADBEEF);

    // createFusedModel should return early without changing it
    try createFusedModel(testing.allocator, weight_handles, loaded.config);

    // Verify it wasn't changed
    try testing.expect(weight_handles.fused_model == @as(?*anyopaque, @ptrFromInt(0xDEADBEEF)));

    // Restore for proper cleanup
    weight_handles.fused_model = original_fused;
}

test "createFusedModel skips when dense_model already exists" {
    if (comptime builtin.os.tag != .macos) return;
    if (!metal_device.isAvailable()) return;

    const loaded = try createTestLoadedModel(testing.allocator);
    defer destroyTestLoadedModel(testing.allocator, loaded);

    const weight_handles = try loadWeightsToGPU(testing.allocator, loaded);
    defer freeWeights(testing.allocator, weight_handles);

    // Set a sentinel value for dense_model
    const original_dense = weight_handles.dense_model;
    weight_handles.dense_model = @ptrFromInt(0xDEADBEEF);

    // createFusedModel should return early without error
    try createFusedModel(testing.allocator, weight_handles, loaded.config);

    // Verify dense_model wasn't changed
    try testing.expect(weight_handles.dense_model == @as(?*anyopaque, @ptrFromInt(0xDEADBEEF)));

    // Restore for proper cleanup
    weight_handles.dense_model = original_dense;
}

test "Model.forward produces logits from input tokens" {
    if (comptime builtin.os.tag != .macos) return;
    if (!metal_device.isAvailable()) return;

    const loaded = try createTestLoadedModel(testing.allocator);
    defer destroyTestLoadedModel(testing.allocator, loaded);

    const weight_handles = try loadWeightsToGPU(testing.allocator, loaded);
    defer freeWeights(testing.allocator, weight_handles);

    // Create KV cache
    const cache = mlx_graph.Cache.init(1, true);
    defer cache.deinit();

    // Run forward pass with single token
    const input_ids = [_]u32{1};
    const logits_handle = try model_executor.Model.forward(
        testing.allocator,
        weight_handles,
        &input_ids,
        loaded.config,
        cache,
        null,
        0,
        false,
    );

    // Verify we got a valid logits handle
    try testing.expect(logits_handle != null);

    // Eval and verify shape
    var handles = [_]mlx_graph.ArrayHandle{logits_handle};
    mlx_graph.eval(&handles);

    var shape_buffer: [8]usize = undefined;
    const rank = mlx_graph.getShape(logits_handle, &shape_buffer);
    try testing.expect(rank >= 1);
    // Last dimension should be vocab_size (32)
    try testing.expectEqual(@as(usize, 32), shape_buffer[rank - 1]);

    mlx_graph.freeArray(logits_handle);
}

test "Model.forwardFromGPUToken produces logits from GPU token" {
    if (comptime builtin.os.tag != .macos) return;
    if (!metal_device.isAvailable()) return;

    const loaded = try createTestLoadedModel(testing.allocator);
    defer destroyTestLoadedModel(testing.allocator, loaded);

    const weight_handles = try loadWeightsToGPU(testing.allocator, loaded);
    defer freeWeights(testing.allocator, weight_handles);

    // Create KV cache
    const cache = mlx_graph.Cache.init(1, true);
    defer cache.deinit();

    // First do a prefill to populate cache and get a token
    const prefill_ids = [_]u32{1};
    const prefill_logits = try model_executor.Model.forward(
        testing.allocator,
        weight_handles,
        &prefill_ids,
        loaded.config,
        cache,
        null,
        0,
        false,
    );
    defer mlx_graph.freeArray(prefill_logits);

    // Get last position logits and compute argmax to get a GPU token handle
    const last_logits = mlx_graph.mlx_lazy_slice_last(prefill_logits);
    const token_handle = mlx_graph.mlx_lazy_argmax(last_logits, -1);
    defer mlx_graph.freeArray(token_handle);

    // Eval to materialize the token
    var token_handles = [_]mlx_graph.ArrayHandle{token_handle};
    mlx_graph.eval(&token_handles);

    // Run forward pass with GPU token
    const logits_handle = try model_executor.Model.forwardFromGPUToken(
        testing.allocator,
        weight_handles,
        token_handle,
        loaded.config,
        cache,
        null,
        1, // pos_offset = 1 since we did 1 prefill token
    );

    // Verify we got a valid logits handle
    try testing.expect(logits_handle != null);

    // Eval and verify shape
    var handles = [_]mlx_graph.ArrayHandle{logits_handle};
    mlx_graph.eval(&handles);

    var shape_buffer: [8]usize = undefined;
    const rank = mlx_graph.getShape(logits_handle, &shape_buffer);
    try testing.expect(rank >= 1);
    // Last dimension should be vocab_size (32)
    try testing.expectEqual(@as(usize, 32), shape_buffer[rank - 1]);

    mlx_graph.freeArray(logits_handle);
}
