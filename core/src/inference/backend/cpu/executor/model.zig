//! Transformer model structure and execution.
//!
//! Assembles transformer layers into a complete model with embeddings,
//! blocks, final normalization, and output projection.

const std = @import("std");
const build_options = @import("build_options");
const models = @import("../../../../models/root.zig");
const Block = @import("block.zig").Block;
const layer_ops = @import("../../../../models/layer_ops.zig");
const tensor_mod = @import("../../../../tensor.zig");
const dtype_mod = @import("../../../../dtype.zig");
const compute = @import("../../../../compute/root.zig");
const cpu_linalg = compute.cpu.linalg;
const cpu_common = compute.cpu.common;
const cpu_memory = compute.cpu.memory;
const inspect = @import("../../../../xray/root.zig");
const kernel_info = inspect.kernel_info;
const perf_estimate = inspect.perf_estimate;
const trace = @import("../../../../xray/trace.zig");
const dump = if (build_options.dump_tensors) @import("../../../../xray/dump/capture.zig") else struct {
    pub fn recordGlobal(_: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) void {}
};
const embedding_kernel = @import("../kernels/embedding.zig");
const attn_kernel = @import("../kernels/attention.zig");
const norm_kernel = @import("../kernels/norm.zig");

const Tensor = tensor_mod.Tensor;
const DType = dtype_mod.DType;
const Attention = attn_kernel.MultiHeadAttention;
const FFNLayer = block_kernels.FfnLayer;
const RMSNorm = norm_kernel.RMSNorm;
const ScratchBuffer = @import("runtime.zig").ScratchBuffer;
const PerfEstimate = perf_estimate.PerfEstimate;
const MatmulFn = cpu_linalg.MatmulFn;
const KernelOp = kernel_info.KernelOp;

const block_kernels = @import("weights.zig");
const cpu_blocks = block_kernels;

const kv_cache = @import("../kernels/kv_cache.zig");
const LayeredBatchedKVCache = kv_cache.LayeredBatchedKVCache;

const LoadedModel = models.LoadedModel;
const ModelConfig = tensor_mod.ModelConfig;
const LayerOp = layer_ops.LayerOp;
const BlockKind = block_kernels.BlockType;

pub fn formatLinearLike(
    writer: anytype,
    weight: *const Tensor,
    bias: ?[]const f32,
    in_features: usize,
    out_features: usize,
) !void {
    const weight_dtype = weight.dtype;
    if (weight_dtype == .grouped_affine_u4) {
        const group_size = if (weight.gaffine) |m| m.group_size else 64;
        try writer.print("QuantizedLinear(in={}, out={}, bits=4, group_size={})", .{
            in_features,
            out_features,
            group_size,
        });
    } else if (weight_dtype == .grouped_affine_u8) {
        const group_size = if (weight.gaffine) |m| m.group_size else 64;
        try writer.print("QuantizedLinear(in={}, out={}, bits=8, group_size={})", .{
            in_features,
            out_features,
            group_size,
        });
    } else {
        const dtype_name: []const u8 = switch (weight_dtype) {
            .f32 => "f32",
            .f16 => "f16",
            .bf16 => "bf16",
            else => "unknown",
        };
        try writer.print("Linear(in={}, out={}, bias={}, dtype={s})", .{
            in_features,
            out_features,
            bias != null,
            dtype_name,
        });
    }
}

pub fn formatRmsNormLike(writer: anytype, dim: usize, eps: f32, weight_offset: f32) !void {
    if (weight_offset != 0.0) {
        try writer.print("RMSNorm(dim={}, eps={e}, weight_offset={d:.1})", .{ dim, eps, weight_offset });
    } else {
        try writer.print("RMSNorm(dim={}, eps={e})", .{ dim, eps });
    }
}

/// Linear transformation: y = x @ W + b
/// Owns a pointer to weight tensor (mmap'd) and optional bias.
pub const Linear = struct {
    weight: *const Tensor,
    bias: ?[]const f32 = null,
    in_features: usize,
    out_features: usize,
    matmul_fn: MatmulFn,
    pub fn init(weight: *const Tensor, bias: ?[]const f32) !Linear {
        const in_features: usize = switch (weight.dtype) {
            .f32 => @intCast(weight.shape[0]),
            else => @intCast(weight.shape[1]),
        };
        const out_features: usize = switch (weight.dtype) {
            .f32 => @intCast(weight.shape[1]),
            else => @intCast(weight.shape[0]),
        };
        const dk = try cpu_linalg.matmulKernel(weight.dtype);
        return .{
            .weight = weight,
            .bias = bias,
            .in_features = in_features,
            .out_features = out_features,
            .matmul_fn = dk.func,
        };
    }

    pub fn initWithDims(weight: *const Tensor, bias: ?[]const f32, in_features: usize, out_features: usize) !Linear {
        const dk = try cpu_linalg.matmulKernel(weight.dtype);
        return .{
            .weight = weight,
            .bias = bias,
            .in_features = in_features,
            .out_features = out_features,
            .matmul_fn = dk.func,
        };
    }

    pub inline fn forward(self: *const Linear, input_tensor: *const Tensor, output_tensor: *Tensor, scratch: *cpu_linalg.MatmulScratch) void {
        const row_count: usize = if (input_tensor.n_dims == 3) @intCast(input_tensor.shape[0] * input_tensor.shape[1]) else @intCast(input_tensor.shape[0]);
        const input_view = Tensor.view2D(input_tensor.data(), row_count, self.in_features);
        var output_view = Tensor.view2DSlice(output_tensor.asSlice(f32), row_count, self.out_features);

        self.matmul_fn(&input_view, self.weight, &output_view, scratch);

        if (self.bias) |bias| {
            cpu_common.addBiasRows(output_tensor.asSlice(f32), bias, row_count, self.out_features);
        }
    }

    pub fn formatKernels(self: *const Linear, writer: anytype, indent: usize) !void {
        const matmul_op = KernelOp{ .matmul = .{
            .m = .seq,
            .k = self.in_features,
            .n = self.out_features,
            .dtype = self.weight.dtype,
            .kernel_name = kernel_info.matmulKernelName(self.weight.dtype),
        } };
        try matmul_op.format(writer, indent);

        if (self.bias != null) {
            const bias_op = KernelOp{ .bias_add = .{ .size = self.out_features } };
            try bias_op.format(writer, indent);
        }
    }

    pub fn describe(self: *const Linear, writer: anytype, indent: usize, show_kernels: bool) !void {
        try writer.writeByteNTimes(' ', indent);
        try self.formatTo(writer);
        try writer.writeAll("\n");

        if (show_kernels) {
            try self.formatKernels(writer, indent + 2);
        }
    }

    pub fn formatTo(self: *const Linear, writer: anytype) !void {
        try formatLinearLike(writer, self.weight, self.bias, self.in_features, self.out_features);
    }
};

/// Token embedding lookup table
pub const Embedding = struct {
    weight: *const Tensor,
    vocab_size: usize,
    embed_dim: usize,
    pub fn init(weight: *const Tensor) Embedding {
        return .{
            .weight = weight,
            .vocab_size = @intCast(weight.shape[0]),
            .embed_dim = @intCast(weight.shape[1]),
        };
    }

    pub fn forward(self: *const Embedding, tokens: []const u32, output_tensor: *Tensor) !void {
        try embedding_kernel.gatherEmbeddings(self.weight, tokens, output_tensor);
    }

    pub fn formatKernels(self: *const Embedding, writer: anytype, indent: usize) !void {
        const gather_op = KernelOp{ .gather = .{
            .vocab_size = self.vocab_size,
            .embed_dim = self.embed_dim,
            .dtype = self.weight.dtype,
        } };
        try gather_op.format(writer, indent);
    }

    pub fn describe(self: *const Embedding, writer: anytype, indent: usize, show_kernels: bool) !void {
        try writer.writeByteNTimes(' ', indent);
        try self.formatTo(writer);
        try writer.writeAll("\n");

        if (show_kernels) {
            try self.formatKernels(writer, indent + 2);
        }
    }

    pub fn formatTo(self: *const Embedding, writer: anytype) !void {
        const dtype = self.weight.dtype;
        if (dtype == .grouped_affine_u4 or dtype == .grouped_affine_u8) {
            const bits: u8 = if (dtype == .grouped_affine_u4) 4 else 8;
            try writer.print("Embedding(vocab={}, dim={}, bits={})", .{
                self.vocab_size, self.embed_dim, bits,
            });
        } else {
            try writer.print("Embedding(vocab={}, dim={})", .{
                self.vocab_size, self.embed_dim,
            });
        }
    }
};

/// Complete transformer model for inference
pub const Transformer = struct {
    model_type: []const u8,
    embed_tokens: Embedding,
    layers: []Block,
    norm: ?RMSNorm = null,
    lm_head: ?Linear = null,
    tie_word_embeddings: bool = true,

    // Dimensions
    hidden_size: usize,
    vocab_size: usize,
    num_hidden_layers: usize,

    // Original weight dtype for summary
    weight_dtype: DType,

    // File info for summary (from LoadedModel)
    file_size: usize = 0,

    // Optional prefill progress callback (set transiently by prefillSlot)
    prefill_progress_fn: ?PrefillProgressFn = null,
    prefill_progress_ctx: ?*anyopaque = null,

    tensor_count: usize = 0,

    pub const PrefillProgressFn = *const fn (usize, usize, ?*anyopaque) callconv(.c) void;
    pub const DeepstackAdditions = struct {
        /// Token positions in the prompt where visual placeholders were scattered.
        positions: []const usize,
        /// Per-decoder-layer visual embeddings to add at `positions`.
        /// Layer 0 consumes layer_features[0], layer 1 consumes layer_features[1], etc.
        layer_features: []const []const f32,
    };

    /// Forward pass through transformer layers only (not embedding or final norm).
    /// This is the core transformer body: hidden_states -> layers -> hidden_states
    pub fn forward(
        self: *const Transformer,
        input_tensor: *const Tensor,
        output_tensor: *Tensor,
        scratch: *ScratchBuffer,
        use_cache: bool,
    ) !void {
        if (!use_cache) scratch.resetCaches();
        const seq_len: usize = @intCast(input_tensor.shape[1]);
        try scratch.ensure(seq_len);

        // Use pre-allocated scratch buffer for alternating input/output
        var scratch_tensor_view = Tensor.view3DSlice(scratch.tmp[0], seq_len, self.hidden_size);
        var current_input_tensor: *const Tensor = input_tensor;
        var write_to_scratch_view = false;

        for (self.layers, 0..) |*layer, layer_idx| {
            // Emit trace point for layer input (if handler installed)
            trace.emit(
                .layer_input,
                @intCast(layer_idx),
                0, // token (batch dimension)
                @intCast(seq_len), // position
                current_input_tensor.data().ptr,
                .f32,
                .{ 1, @intCast(seq_len), @intCast(self.hidden_size), 0 },
                3,
                null, // no kernel - this is the input activation
            );

            const attention_cache = &scratch.attn_caches[layer_idx];
            if (write_to_scratch_view) {
                try layer.forward(current_input_tensor, &scratch_tensor_view, scratch, attention_cache, use_cache);
                current_input_tensor = &scratch_tensor_view;
            } else {
                try layer.forward(current_input_tensor, output_tensor, scratch, attention_cache, use_cache);
                current_input_tensor = output_tensor;
            }
            write_to_scratch_view = !write_to_scratch_view;

            // Emit trace point for layer output (if handler installed)
            trace.emit(
                .block_out,
                @intCast(layer_idx),
                0, // token (batch dimension)
                @intCast(seq_len), // position
                current_input_tensor.data().ptr,
                .f32,
                .{ 1, @intCast(seq_len), @intCast(self.hidden_size), 0 },
                3,
                null, // residual is element-wise add, not a matmul kernel
            );
            // Dump capture (compiled in only for dump binary)
            if (build_options.dump_tensors) {
                const shape = [4]usize{ 1, seq_len, self.hidden_size, 0 };
                dump.recordGlobal(.block_out, @intCast(layer_idx), current_input_tensor.data().ptr, .f32, shape, 3);
                // Early stop for partial dump (big runtime win when debugging specific layers)
                if (dump.shouldStopGlobal()) break;
            }
        }

        // Copy final result to out if needed
        if (current_input_tensor != output_tensor) {
            block_kernels.copyTensor(current_input_tensor, output_tensor);
        }
    }

    /// Forward pass through transformer layers using batched KV cache.
    /// This enables continuous batching by using LayeredBatchedKVCache with a slot index.
    pub fn forwardWithBatchedCache(
        self: *const Transformer,
        input_tensor: *const Tensor,
        output_tensor: *Tensor,
        scratch: *ScratchBuffer,
        batched_cache: *LayeredBatchedKVCache,
        slot_index: usize,
        use_cache: bool,
    ) !void {
        return self.forwardWithBatchedCacheWithDeepstack(
            input_tensor,
            output_tensor,
            scratch,
            batched_cache,
            slot_index,
            use_cache,
            null,
        );
    }

    pub fn forwardWithBatchedCacheWithDeepstack(
        self: *const Transformer,
        input_tensor: *const Tensor,
        output_tensor: *Tensor,
        scratch: *ScratchBuffer,
        batched_cache: *LayeredBatchedKVCache,
        slot_index: usize,
        use_cache: bool,
        deepstack: ?*const DeepstackAdditions,
    ) !void {
        if (!use_cache) batched_cache.resetSlot(slot_index);
        const seq_len: usize = @intCast(input_tensor.shape[1]);
        try scratch.ensure(seq_len);

        // Use pre-allocated scratch buffer for alternating input/output
        var scratch_tensor_view = Tensor.view3DSlice(scratch.tmp[0], seq_len, self.hidden_size);
        var current_input_tensor: *const Tensor = input_tensor;
        var write_to_scratch_view = false;

        for (self.layers, 0..) |*layer, layer_idx| {
            // Emit trace point for layer input (if handler installed)
            trace.emit(
                .layer_input,
                @intCast(layer_idx),
                0,
                @intCast(seq_len),
                current_input_tensor.data().ptr,
                .f32,
                .{ 1, @intCast(seq_len), @intCast(self.hidden_size), 0 },
                3,
                null,
            );

            const layer_cache = batched_cache.getLayer(layer_idx);
            if (write_to_scratch_view) {
                try layer.forwardWithBatchedCache(current_input_tensor, &scratch_tensor_view, scratch, layer_cache, slot_index, use_cache);
                current_input_tensor = &scratch_tensor_view;
                if (deepstack) |ctx| {
                    if (layer_idx < ctx.layer_features.len) {
                        try applyDeepstackAdditions(&scratch_tensor_view, seq_len, self.hidden_size, ctx.positions, ctx.layer_features[layer_idx]);
                    }
                }
            } else {
                try layer.forwardWithBatchedCache(current_input_tensor, output_tensor, scratch, layer_cache, slot_index, use_cache);
                current_input_tensor = output_tensor;
                if (deepstack) |ctx| {
                    if (layer_idx < ctx.layer_features.len) {
                        try applyDeepstackAdditions(output_tensor, seq_len, self.hidden_size, ctx.positions, ctx.layer_features[layer_idx]);
                    }
                }
            }
            write_to_scratch_view = !write_to_scratch_view;

            // Emit trace point for layer output
            trace.emit(
                .block_out,
                @intCast(layer_idx),
                0,
                @intCast(seq_len),
                current_input_tensor.data().ptr,
                .f32,
                .{ 1, @intCast(seq_len), @intCast(self.hidden_size), 0 },
                3,
                null,
            );
            // Dump capture (compiled in only for dump binary)
            if (build_options.dump_tensors) {
                const shape = [4]usize{ 1, seq_len, self.hidden_size, 0 };
                dump.recordGlobal(.block_out, @intCast(layer_idx), current_input_tensor.data().ptr, .f32, shape, 3);
                // Early stop for partial dump (big runtime win when debugging specific layers)
                if (dump.shouldStopGlobal()) break;
            }

            // Prefill progress (fires once per layer, outside hot path)
            if (self.prefill_progress_fn) |progress_fn| {
                progress_fn(layer_idx + 1, self.num_hidden_layers, self.prefill_progress_ctx);
            }
        }

        // Copy final result to out if needed
        if (current_input_tensor != output_tensor) {
            block_kernels.copyTensor(current_input_tensor, output_tensor);
        }
    }

    fn applyDeepstackAdditions(
        hidden: *Tensor,
        seq_len: usize,
        hidden_size: usize,
        positions: []const usize,
        features: []const f32,
    ) !void {
        if (positions.len == 0) return;
        try cpu_memory.scatterAddRowsByPositions(
            hidden.asSliceMut(f32),
            seq_len,
            hidden_size,
            positions,
            features,
        );
    }

    pub fn describe(self: *const Transformer, writer: anytype, show_kernels: bool) !void {
        try self.describeCondensed(writer, show_kernels, 3); // Show first 3, last 1
    }

    const DescribeMode = union(enum) {
        condensed: usize,
        all,
    };

    fn describeImpl(self: *const Transformer, writer: anytype, show_kernels: bool, mode: DescribeMode) !void {
        try writer.print("{s}(\n", .{self.model_type});

        try writer.writeAll("  (embed_tokens): ");
        try self.embed_tokens.formatTo(writer);
        try writer.writeAll("\n");

        if (show_kernels) {
            try self.embed_tokens.formatKernels(writer, 4);
        }

        try writer.writeAll("  (layers): [\n");

        const layer_count = self.layers.len;
        switch (mode) {
            .all => {
                for (self.layers) |*layer| {
                    try layer.describe(writer, 4, show_kernels);
                }
            },
            .condensed => |show_first| {
                if (layer_count <= show_first + 2) {
                    for (self.layers) |*layer| {
                        try layer.describe(writer, 4, show_kernels);
                    }
                } else {
                    for (self.layers[0..show_first]) |*layer| {
                        try layer.describe(writer, 4, show_kernels);
                    }
                    const remaining_layers = layer_count - show_first - 1;
                    try writer.print("    ... {d} more identical layers ...\n", .{remaining_layers});
                    try self.layers[layer_count - 1].describe(writer, 4, show_kernels);
                }
            },
        }
        try writer.writeAll("  ]\n");

        if (self.norm) |n| {
            try writer.writeAll("  (norm): ");
            try formatRmsNormLike(writer, n.dim, n.eps, n.weight_offset);
            try writer.writeAll("\n");
        }

        if (self.lm_head) |*head| {
            try writer.writeAll("  (lm_head): ");
            try head.formatTo(writer);
            try writer.writeAll("\n");
            if (show_kernels) {
                try head.formatKernels(writer, 4);
            }
        } else if (self.tie_word_embeddings) {
            try writer.writeAll("  (lm_head): [tied to embed_tokens]\n");
        }

        try writer.writeAll(")\n");
    }

    /// Describe model with condensed layer output.
    /// Shows first `show_first` layers, then "... N more layers ...", then last layer.
    pub fn describeCondensed(self: *const Transformer, writer: anytype, show_kernels: bool, show_first: usize) !void {
        try self.describeImpl(writer, show_kernels, .{ .condensed = show_first });
    }

    /// Describe model showing all layers (no condensing)
    pub fn describeAll(self: *const Transformer, writer: anytype, show_kernels: bool) !void {
        try self.describeImpl(writer, show_kernels, .all);
    }

    pub fn summary(self: *const Transformer, writer: anytype) !void {
        try self.summaryWithSeqLen(writer, 512); // Default seq_len for memory estimates
    }

    pub fn summaryWithSeqLen(self: *const Transformer, writer: anytype, seq_len: usize) !void {
        _ = seq_len; // Reserved for future use

        try writer.print("Model: {s}\n", .{self.model_type});

        // Estimate parameters
        var total_params: usize = 0;
        total_params += self.embed_tokens.vocab_size * self.embed_tokens.embed_dim;

        if (self.firstLayerGeom()) |geom| {
            var layer_params: usize = 0;
            layer_params += geom.total_layer_params;
            total_params += layer_params * self.num_hidden_layers;
        }

        if (self.lm_head) |head| {
            total_params += head.in_features * head.out_features;
        }

        // Parameters
        if (total_params >= 1_000_000_000) {
            const billions = @as(f64, @floatFromInt(total_params)) / 1_000_000_000.0;
            try writer.print("  Parameters: {d:.2}B\n", .{billions});
        } else if (total_params >= 1_000_000) {
            const millions = @as(f64, @floatFromInt(total_params)) / 1_000_000.0;
            try writer.print("  Parameters: {d:.2}M\n", .{millions});
        } else {
            try writer.print("  Parameters: {}\n", .{total_params});
        }

        // Quantization
        const quant_info: []const u8 = switch (self.weight_dtype) {
            .grouped_affine_u4 => "4-bit (grouped affine)",
            .grouped_affine_u8 => "8-bit (grouped affine)",
            .bf16 => "BF16",
            .f16 => "F16",
            .f32 => "F32",
            else => "unknown",
        };
        try writer.print("  Quantization: {s}\n", .{quant_info});

        // Weights (file size)
        if (self.file_size > 0) {
            try writer.writeAll("  Weights: ");
            try perf_estimate.formatBytes(writer, self.file_size);
            try writer.writeAll("\n");
        }

        // Architecture
        try writer.print("  Layers: {}\n", .{self.num_hidden_layers});
        try writer.print("  Hidden size: {}\n", .{self.hidden_size});
        try writer.print("  Vocab size: {}\n", .{self.vocab_size});

        // Help hint
        try writer.writeAll("\nUse -v for module graph, -vv for kernel operations\n");
    }

    /// Estimate weight memory in bytes (accounts for quantization)
    pub fn estimateWeightMemory(self: *const Transformer) u64 {
        var total_params: u64 = 0;

        // Embedding
        total_params += self.embed_tokens.vocab_size * self.embed_tokens.embed_dim;

        // Layers
        if (self.firstLayerGeom()) |geom| {
            var layer_params: u64 = 0;

            // Per-layer weights
            layer_params += @intCast(geom.total_layer_params);

            // Norms (always f32)
            const norm_params: u64 = self.hidden_size * 2; // input + post-attn norm

            total_params += layer_params * self.num_hidden_layers;
            // Norm params are always f32
            total_params += norm_params * self.num_hidden_layers * 4 / self.bytesPerParam();
        }

        // Final norm (f32)
        total_params += self.hidden_size * 4 / self.bytesPerParam();

        // LM head
        if (self.lm_head) |head| {
            total_params += head.in_features * head.out_features;
        }

        return total_params * self.bytesPerParam();
    }

    /// Estimate scratch buffer memory for given sequence length
    pub fn estimateScratchMemory(self: *const Transformer, seq_len: usize) u64 {
        var total: u64 = 0;

        // Activation buffers (f32): tmp[0..2] (layer_tmp, norm_out, branch_out)
        // Each is [seq_len, hidden_size]
        const activation_buf = seq_len * self.hidden_size * 4;
        total += activation_buf * 3;

        if (self.firstLayerKernels()) |layer| {
            const attn = layer.attn;
            const ffn = layer.ffn;

            const kv_per_layer = seq_len * attn.n_kv_heads * attn.head_dim * 4 * 2; // f32, K+V
            total += kv_per_layer * self.num_hidden_layers;

            // Scores: [n_heads, seq_len, seq_len] for prefill
            total += attn.n_heads * seq_len * seq_len * 4;

            const ffn_size = switch (ffn.*) {
                .swiglu => |mlp| seq_len * mlp.d_ff * 4,
                .moe_ffn => |moe| seq_len * moe.d_ff * moe.experts_per_token * 4,
            };
            total += ffn_size;
        }

        return total;
    }

    fn bytesPerParam(self: *const Transformer) u64 {
        return switch (self.weight_dtype) {
            .grouped_affine_u4 => 1, // ~0.5 bytes, round up to 1 for scale overhead
            .grouped_affine_u8 => 1,
            .f16, .bf16 => 2,
            .f32 => 4,
            else => 2,
        };
    }

    fn firstLayerKernels(self: *const Transformer) ?struct { attn: *const Attention, ffn: *const FFNLayer } {
        if (self.layers.len == 0) return null;
        const first = &self.layers[0];
        const attn = first.getAttention() orelse return null;
        const ffn = first.getFFN() orelse return null;
        return .{
            .attn = attn,
            .ffn = ffn,
        };
    }

    fn firstLayerGeom(self: *const Transformer) ?perf_estimate.LayerGeometry {
        const layer = self.firstLayerKernels() orelse return null;
        return perf_estimate.LayerGeometry.init(
            attnToConfig(layer.attn),
            ffnToConfig(layer.ffn),
        );
    }

    /// Convert runtime Attention to primitive AttnConfig for perf estimation.
    fn attnToConfig(attn: *const Attention) perf_estimate.AttnConfig {
        return .{
            .n_heads = attn.n_heads,
            .n_kv_heads = attn.n_kv_heads,
            .head_dim = attn.head_dim,
            .d_model = attn.d_model,
            .has_q_bias = attn.q_bias != null,
            .has_k_bias = attn.k_bias != null,
            .has_v_bias = attn.v_bias != null,
            .has_o_bias = false, // o_bias not present in MultiHeadAttention
        };
    }

    /// Convert runtime FFNLayer to primitive FfnConfig for perf estimation.
    fn ffnToConfig(ffn: *const FFNLayer) perf_estimate.FfnConfig {
        return switch (ffn.*) {
            .swiglu => |mlp| .{ .swiglu = .{
                .d_model = mlp.d_model,
                .d_ff = mlp.d_ff,
            } },
            .moe_ffn => |moe| .{ .moe_ffn = .{
                .d_model = moe.d_model,
                .d_ff = moe.d_ff,
                .num_experts = moe.num_experts,
                .experts_per_token = moe.experts_per_token,
            } },
        };
    }

    /// Estimate FLOPs for a forward pass with given sequence length.
    /// Returns struct with prefill and per-token decode FLOPs.
    pub fn estimateFlops(self: *const Transformer, seq_len: usize) PerfEstimate {
        return self.estimatePerf(seq_len);
    }

    /// Estimate performance characteristics (FLOPs and memory bandwidth) for inference.
    /// Returns struct with prefill and per-token decode estimates.
    pub fn estimatePerf(self: *const Transformer, seq_len: usize) PerfEstimate {
        const layer = self.firstLayerKernels() orelse return .{
            .prefill_flops = 0,
            .per_token_flops = 0,
            .prefill_mem_bytes = 0,
            .per_token_mem_bytes = 0,
            .seq_len = seq_len,
            .weight_dtype = self.weight_dtype,
        };
        return perf_estimate.estimatePerf(.{
            .seq_len = seq_len,
            .weight_dtype = self.weight_dtype,
            .hidden_size = self.hidden_size,
            .vocab_size = self.vocab_size,
            .num_hidden_layers = self.num_hidden_layers,
            .attn = attnToConfig(layer.attn),
            .ffn = ffnToConfig(layer.ffn),
        });
    }

    /// Build a Transformer from LoadedModel weights and CPU kernel blocks.
    pub fn build(
        allocator: std.mem.Allocator,
        loaded: *const LoadedModel,
        blocks: []const cpu_blocks.TransformerBlock,
    ) !Transformer {
        const model_config = loaded.config;
        const layer_count = blocks.len;

        // Build layers as Block (with ops-based execution)
        var block_layers = try allocator.alloc(Block, layer_count);
        errdefer allocator.free(block_layers);

        const static_entry = detectStaticTopologyEntry(loaded);

        // Get block program for each layer (handles heterogeneous models)
        // For heterogeneous models, each layer may have a different program (e.g., Mamba vs Attention)
        // Track Mamba/ShortConv layer indices separately from global layer index
        var mamba_layer_count: usize = 0;
        var shortconv_layer_count: usize = 0;
        for (blocks, 0..) |*block, layer_idx| {
            const block_kind = if (block.isMamba())
                BlockKind.mamba
            else if (block.isShortConv())
                BlockKind.shortconv
            else
                BlockKind.attention_mlp;
            const layer_program = try resolveLayerProgram(static_entry, block_kind);
            const mamba_idx: ?usize = if (block.isMamba()) blk: {
                const idx = mamba_layer_count;
                mamba_layer_count += 1;
                break :blk idx;
            } else null;
            const shortconv_idx: ?usize = if (block.isShortConv()) blk: {
                const idx = shortconv_layer_count;
                shortconv_layer_count += 1;
                break :blk idx;
            } else null;
            block_layers[layer_idx] = buildBlock(block, model_config, layer_idx, layer_program, mamba_idx, shortconv_idx);
            try block_layers[layer_idx].validate();
        }

        // Build embedding
        const embed_tokens = Embedding.init(&loaded.token_embeddings);

        // Build final norm (optional — embed-only models may not have one)
        const final_norm: ?RMSNorm = if (loaded.ln_final) |*ln_f|
            RMSNorm{
                .weight = ln_f,
                .dim = @intCast(model_config.d_model),
                .eps = model_config.norm_eps,
                .weight_offset = loaded.runtime.weight_offset,
                .trace_point = .final_norm,
            }
        else
            null;

        // Build LM head if not tied (optional — embed-only models may not have one)
        const lm_head: ?Linear = if (loaded.lm_head) |*head|
            try Linear.init(head, null)
        else if (model_config.tie_word_embeddings)
            null
        else
            null;

        // Model type for debug output.
        const model_type: []const u8 = if (static_entry) |entry| entry.id else "TransformerModel";

        return .{
            .model_type = model_type,
            .embed_tokens = embed_tokens,
            .layers = block_layers,
            .norm = final_norm,
            .lm_head = lm_head,
            .tie_word_embeddings = model_config.tie_word_embeddings,
            .hidden_size = @intCast(model_config.d_model),
            .vocab_size = @intCast(model_config.vocab_size),
            .num_hidden_layers = layer_count,
            .weight_dtype = loaded.original_weight_dtype,
            .file_size = loaded.file_size,
            .tensor_count = loaded.tensor_count,
        };
    }

    /// Build a Block with ops-based execution from cpu_blocks.TransformerBlock
    fn buildBlock(
        block: *const cpu_blocks.TransformerBlock,
        model_config: ModelConfig,
        layer_idx: usize,
        program: []const LayerOp,
        mamba_layer_idx: ?usize,
        shortconv_layer_idx: ?usize,
    ) Block {
        return .{
            .program = program,
            .block = block,
            .block_idx = layer_idx,
            .hidden_size = @intCast(model_config.d_model),
            .mamba_layer_idx = mamba_layer_idx,
            .shortconv_layer_idx = shortconv_layer_idx,
        };
    }

    fn detectStaticTopologyEntry(loaded: *const LoadedModel) ?models.contract.ModelDescriptor {
        if (loaded.runtime.architecture_id) |arch_id| {
            return models.registry.detectByArchitectureId(arch_id);
        }
        return null;
    }

    fn resolveLayerProgram(
        static_entry: ?models.contract.ModelDescriptor,
        block_kind: BlockKind,
    ) ![]const LayerOp {
        if (static_entry) |entry| {
            return models.registry.blockProgramFor(entry, block_kind) orelse error.UnsupportedModel;
        }
        return error.UnsupportedModel;
    }

    /// Free model allocated by build
    pub fn deinit(self: *Transformer, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
        self.* = undefined;
    }
};

pub const Model = Transformer;

// =============================================================================
// Unit Tests
// =============================================================================

const testing = std.testing;
const expect = testing.expect;

/// Create a minimal mock model for testing inspection/estimation functions only.
/// Does not support forward passes - use for testing describe/summary/estimate* only.
fn createMockTransformer(num_layers: usize, hidden_size: usize, vocab_size: usize) Transformer {
    // Create a transformer with placeholder values - this is only for testing
    // the estimation functions which use the dimensions, not the actual layer data.
    const model = Transformer{
        .model_type = "test_model",
        .embed_tokens = .{
            .weight = undefined, // Not dereferenced in estimation functions
            .vocab_size = vocab_size,
            .embed_dim = hidden_size,
        },
        .layers = &[_]Block{}, // Empty layers slice - firstLayerKernels returns null
        .norm = undefined, // Not dereferenced in estimation
        .lm_head = null,
        .tie_word_embeddings = false,
        .hidden_size = hidden_size,
        .vocab_size = vocab_size,
        .num_hidden_layers = num_layers,
        .weight_dtype = .f32,
        .file_size = 1024 * 1024,
        .tensor_count = 10,
    };
    return model;
}

test "Transformer.summary writes model summary without layers" {
    const model = createMockTransformer(2, 64, 128);

    var buf: std.ArrayList(u8) = .{};
    defer buf.deinit(testing.allocator);

    try model.summary(buf.writer(testing.allocator));
    const output = buf.items;

    // Verify summary contains key information
    try expect(std.mem.indexOf(u8, output, "Model: test_model") != null);
    try expect(std.mem.indexOf(u8, output, "Quantization:") != null);
    try expect(std.mem.indexOf(u8, output, "Layers: 2") != null);
    try expect(std.mem.indexOf(u8, output, "Hidden size: 64") != null);
    try expect(std.mem.indexOf(u8, output, "Vocab size: 128") != null);
}

test "Transformer.summaryWithSeqLen writes model summary with custom seq_len" {
    const model = createMockTransformer(2, 64, 128);

    var buf: std.ArrayList(u8) = .{};
    defer buf.deinit(testing.allocator);

    try model.summaryWithSeqLen(buf.writer(testing.allocator), 1024);
    const output = buf.items;

    try expect(std.mem.indexOf(u8, output, "Model: test_model") != null);
    try expect(std.mem.indexOf(u8, output, "Quantization") != null);
}

test "Transformer.estimateWeightMemory returns reasonable value" {
    const model = createMockTransformer(2, 64, 128);
    const weight_mem = model.estimateWeightMemory();

    // Should be non-zero
    try expect(weight_mem > 0);

    // For f32 model with 128 vocab, 64 hidden, should be reasonable
    // Embedding: 128 * 64 * 4 = 32768 bytes
    // Should be at least embedding size
    try expect(weight_mem >= 32768);
}

test "Transformer.estimateScratchMemory returns reasonable value for seq_len" {
    const model = createMockTransformer(2, 64, 128);

    const scratch_mem_small = model.estimateScratchMemory(16);
    const scratch_mem_large = model.estimateScratchMemory(512);

    // Should be non-zero
    try expect(scratch_mem_small > 0);
    try expect(scratch_mem_large > 0);

    // Larger seq_len should require more scratch memory
    try expect(scratch_mem_large > scratch_mem_small);
}

test "Transformer.estimateFlops returns PerfEstimate struct with no layers" {
    const model = createMockTransformer(2, 64, 128);
    const perf = model.estimateFlops(128);

    // Verify fields are populated
    try expect(perf.seq_len == 128);
    try expect(perf.weight_dtype == .f32);

    // With no layers, FLOPs should be zero
    try expect(perf.prefill_flops == 0);
    try expect(perf.per_token_flops == 0);
}

test "Transformer.estimatePerf returns PerfEstimate with memory estimates" {
    const model = createMockTransformer(2, 64, 128);
    const perf = model.estimatePerf(128);

    try expect(perf.seq_len == 128);
    try expect(perf.weight_dtype == .f32);

    // With no layers (firstLayerKernels returns null), should return zeros
    try expect(perf.prefill_flops == 0);
    try expect(perf.per_token_flops == 0);
    try expect(perf.prefill_mem_bytes == 0);
    try expect(perf.per_token_mem_bytes == 0);
}

test "Transformer.estimateWeightMemory handles quantized dtypes" {
    var model = createMockTransformer(1, 64, 128);
    const weight_mem_f32 = model.estimateWeightMemory();

    // Test with different dtypes
    model.weight_dtype = .f16;
    const weight_mem_f16 = model.estimateWeightMemory();

    model.weight_dtype = .grouped_affine_u4;
    const weight_mem_q4 = model.estimateWeightMemory();

    // f32 should be larger than f16
    try expect(weight_mem_f32 > weight_mem_f16);

    // f16 should be larger than q4
    try expect(weight_mem_f16 > weight_mem_q4);
}

test "Transformer.estimateScratchMemory scales with seq_len and hidden_size" {
    const model_small = createMockTransformer(1, 32, 64);
    const model_large = createMockTransformer(1, 128, 256);

    const scratch_small = model_small.estimateScratchMemory(16);
    const scratch_large = model_large.estimateScratchMemory(16);

    // Larger model should require more scratch memory
    try expect(scratch_large > scratch_small);
}

test "estimateWeightMemory bytesPerParam dtypes" {
    var model = createMockTransformer(1, 64, 128);

    model.weight_dtype = .f32;
    try expect(model.bytesPerParam() == 4);

    model.weight_dtype = .f16;
    try expect(model.bytesPerParam() == 2);

    model.weight_dtype = .bf16;
    try expect(model.bytesPerParam() == 2);

    model.weight_dtype = .grouped_affine_u4;
    try expect(model.bytesPerParam() == 1);

    model.weight_dtype = .grouped_affine_u8;
    try expect(model.bytesPerParam() == 1);
}
