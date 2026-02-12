//! Transformer block execution.
//!
//! Executes transformer blocks using LayerOp bytecode from compute graphs.
//! Handles attention, FFN, and residual connections for each layer.

const std = @import("std");
const builtin = @import("builtin");
const common = @import("common.zig");
const layers = @import("layers.zig");
const types = @import("../../graph/root.zig").layer_ops;
const compute = @import("../../compute/root.zig");
const capi = @import("../../capi/error.zig");
const ops = compute.ops;
const tv = ops.tensor_view;
const dtype_mod = @import("../../dtype.zig");
const kernel_wrapper = @import("../backend/cpu/kernel_wrapper.zig");
const log = @import("../../log.zig");

const Tensor = common.Tensor;
const Attention = common.Attention;
const AttnCache = common.AttnCache;
const ScratchBuffer = common.ScratchBuffer;
const FFNLayer = common.FFNLayer;

const kv_cache = @import("../backend/cpu/kernels/kv_cache.zig");
const BatchedKVCache = kv_cache.BatchedKVCache;

const cpu_forward = common.forward;

const graph = @import("../../graph/root.zig");
const BufferId = types.BufferId;
const ResidualScale = types.ResidualScale;
const LayerOp = types.LayerOp;
const KernelContext = kernel_wrapper.KernelContext;

const addIntoScaled = cpu_forward.addIntoScaled;
const copyTensor = cpu_forward.copyTensor;

/// Unified transformer block using sequential operation execution.
/// The topology (norm count, attention type, etc.) is encoded in the ops slice,
/// not in struct variants. This eliminates duplicate forward() logic.
///
/// Model files (src/models/*.zig) define block_program to create the op sequence.
pub const Block = struct {
    /// The "program" - sequence of operations defining block execution.
    /// Typically points to a static table like `models/llama.zig:block_program`.
    program: []const LayerOp,

    /// CPU kernel container for this layer (single source of truth).
    block: *const cpu_forward.TransformerBlock,

    /// Block index in the model (global layer index)
    block_idx: usize,

    /// Hidden size (d_model)
    hidden_size: usize,

    /// Mamba layer index (for heterogeneous models with Mamba layers).
    /// This is the index into mamba_states[], NOT the global layer index.
    /// For homogeneous attention-only models, this is null.
    /// For Mamba layers: mamba_layer_idx = count of Mamba layers before this one.
    mamba_layer_idx: ?usize = null,

    /// ShortConv layer index (for heterogeneous models with ShortConv layers).
    /// This is the index into shortconv_states[], NOT the global layer index.
    /// For homogeneous attention-only models, this is null.
    /// For ShortConv layers: shortconv_layer_idx = count of ShortConv layers before this one.
    shortconv_layer_idx: ?usize = null,

    fn residualScaleValue(self: *const Block, scale: ResidualScale) f32 {
        return switch (scale) {
            .one => 1.0,
            .residual_multiplier => self.block.residual_multiplier,
            .literal => |v| v,
        };
    }

    fn scratchTempSlice(scratch: *ScratchBuffer, which: BufferId, len: usize) []f32 {
        // BufferId maps directly to scratch.tmp array index for tmp3-tmp63
        // Special buffer_views (residual=0, norm_out=1, branch_out=2) handled by resolveOutputSlice
        const buffer_idx = @intFromEnum(which);
        if (buffer_idx >= 3 and buffer_idx < common.block_kernels.NUM_TMP_BUFFERS) {
            return scratch.tmp[buffer_idx][0..len];
        }
        return &.{};
    }

    /// Create a contiguous f32 tensor with correct strides from shape and data slice.
    fn tensorFromSlice(data: []f32, shape: [8]i64, n_dims: i32) Tensor {
        const byte_data = std.mem.sliceAsBytes(data);
        var strides: [8]i64 = .{ 0, 0, 0, 0, 0, 0, 0, 0 };
        const ndim: usize = @intCast(n_dims);
        if (ndim > 0) {
            var stride: i64 = 1;
            var i: usize = ndim;
            while (i > 0) {
                i -= 1;
                strides[i] = stride;
                stride *= shape[i];
            }
        }
        return Tensor{
            .data_ptr = byte_data.ptr,
            .data_size = byte_data.len,
            .shape = shape,
            .strides = strides,
            .n_dims = n_dims,
            .dtype = .f32,
            .numel = data.len,
        };
    }

    fn resolveOutputSlice(buffer_views: *[64]Tensor, scratch: *ScratchBuffer, buffer_id: BufferId, len: usize) []f32 {
        return switch (buffer_id) {
            .residual, .norm_out, .branch_out => buffer_views[@intFromEnum(buffer_id)].asSlice(f32)[0..len],
            else => scratchTempSlice(scratch, buffer_id, len),
        };
    }

    fn readParamValue(param: *const Tensor, elem_idx: usize) f32 {
        return switch (param.dtype) {
            .f32 => param.asSlice(f32)[elem_idx],
            .f16 => dtype_mod.fp16ToF32(param.asSlice(u16)[elem_idx]),
            .bf16 => dtype_mod.bf16ToF32(param.asSlice(u16)[elem_idx]),
            else => 0.0,
        };
    }

    fn applyElementwiseBinaryOp(
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

    fn writeLayerOpDescription(self: *const Block, layer_op: LayerOp, writer: anytype, indent: usize) !void {
        try writer.writeByteNTimes(' ', indent);
        switch (layer_op) {
            .kernel => |kernel_op| {
                try writer.print("kernel({s} -> {s}, id={}): ", .{ @tagName(kernel_op.in), @tagName(kernel_op.out), kernel_op.id });
                const kernel_id: usize = @intCast(kernel_op.id);
                if (kernel_id >= self.block.kernels.len) {
                    try writer.writeAll("invalid\n");
                    return;
                }
                switch (self.block.kernels[kernel_id]) {
                    .norm => |n| {
                        try layers.formatRmsNormLike(writer, n.dim, n.eps, n.weight_offset);
                        try writer.writeAll("\n");
                    },
                    .attention => |a| {
                        try writer.print("Attention(n_heads={}, head_dim={})\n", .{ a.n_heads, a.head_dim });
                    },
                    .swiglu => |m| try writer.print("MLP(d_ff={})\n", .{m.d_ff}),
                    .moe => |e| try writer.print("MoE(experts={}, per_tok={})\n", .{ e.num_experts, e.experts_per_token }),
                    .mamba => |m| {
                        try writer.print("Mamba(d_model={}, d_state={}, d_conv={})\n", .{ m.config.d_model, m.config.d_state, m.config.d_conv });
                    },
                }
            },
            .add => |add_op| {
                const scale = self.residualScaleValue(add_op.scale);
                if (scale == 1.0) {
                    try writer.print("residual += {s}\n", .{@tagName(add_op.branch)});
                } else {
                    try writer.print("residual += {s} * {d:.2}\n", .{ @tagName(add_op.branch), scale });
                }
            },
        }
    }

    /// Forward pass - executes the operation sequence
    pub fn forward(
        self: *const Block,
        x: *const Tensor,
        out: *Tensor,
        scratch: *ScratchBuffer,
        attn_cache: *AttnCache,
        use_cache: bool,
    ) !void {
        std.debug.assert(x.shape[0] == 1 and out.shape[0] == 1); // Only batch=1 supported
        const seq_len: usize = @intCast(x.shape[1]);
        try scratch.ensure(seq_len);

        // Setup buffer views for current sequence length
        const norm_output_view = Tensor.view3DSlice(scratch.tmp[1], seq_len, self.hidden_size);
        const branch_output_view = Tensor.view3DSlice(scratch.tmp[2], seq_len, self.hidden_size);

        // Buffer lookup table: BufferId -> *Tensor
        // Using array indexing compiles to single pointer offset (effectively free)
        // We support 64 buffer_views for primitive-based execution (residual, norm_out, branch_out, tmp3-tmp63)
        var buffer_views: [64]Tensor = undefined;
        buffer_views[@intFromEnum(BufferId.residual)] = out.*;
        buffer_views[@intFromEnum(BufferId.norm_out)] = norm_output_view;
        buffer_views[@intFromEnum(BufferId.branch_out)] = branch_output_view;
        // tmp3-tmp63 are initialized on-demand during split/primitive ops

        // Initialize residual stream with input
        copyTensor(x, out);

        // Get MLA cache/scratch if this is an MLA block
        const is_mla = self.block.isMLA();
        const mla_cache = if (is_mla) scratch.getMLACache(self.block_idx) else null;
        const mla_scratch = if (is_mla) scratch.getMLAScratch() else null;

        const ctx = KernelContext{
            .scratch = scratch,
            .attn_cache = if (is_mla) null else attn_cache,
            .mla_cache = mla_cache,
            .mla_scratch = mla_scratch,
            .mamba_state = if (self.mamba_layer_idx) |idx| scratch.getMambaState(idx) else null,
            .mamba_scratch = scratch.getMambaScratch(),
            .shortconv_state = if (self.shortconv_layer_idx) |idx| scratch.getShortConvState(idx) else null,
            .shortconv_scratch = scratch.getShortConvScratch(),
            .matmul_scratch = &scratch.matmul_scratch,
            .use_cache = use_cache,
        };

        // Execute the operation sequence
        for (self.program, 0..) |op, op_index| {
            switch (op) {
                .kernel => |kernel_op| {
                    const kernel_id: usize = @intCast(kernel_op.id);
                    if (kernel_id >= self.block.kernels.len) {
                        capi.setContext("block={d}, kernel_id={d}, max={d}", .{ self.block_idx, kernel_op.id, self.block.kernels.len });
                        return error.KernelIndexOutOfBounds;
                    }
                    const kernel = self.block.kernels[kernel_id];
                    if (builtin.mode == .Debug) {
                        const actual_type = kernel.getOpType();
                        if (actual_type != kernel_op.debug_type) {
                            log.err("inference", "Graph/Kernel ordering mismatch", .{
                                .block = self.block_idx,
                                .kernel = kernel_op.id,
                                .expected = @tagName(kernel_op.debug_type),
                                .actual = @tagName(actual_type),
                            }, @src());
                            @panic("Graph/Kernel type mismatch - graph compiler and block init are out of sync");
                        }
                    }
                    const input = &buffer_views[@intFromEnum(kernel_op.in)];
                    const output = &buffer_views[@intFromEnum(kernel_op.out)];
                    try kernel.forward(input, output, ctx);
                },
                .add => |add_op| {
                    addIntoScaled(
                        &buffer_views[@intFromEnum(BufferId.residual)],
                        &buffer_views[@intFromEnum(add_op.branch)],
                        &buffer_views[@intFromEnum(BufferId.residual)],
                        self.residualScaleValue(add_op.scale),
                    );
                },

                // =========================================================================
                // Low-level primitive ops for custom attention/MLP implementations
                // =========================================================================

                .linear => |linear_op| {
                    // Linear projection: output = input @ weight
                    // Look up weight from registry by name
                    const weight = self.block.weight_registry.get(linear_op.weight_name) orelse {
                        capi.setContext("block={d}, op={d}, weight={s}", .{ self.block_idx, op_index, linear_op.weight_name });
                        return error.MissingWeight;
                    };

                    const input_tensor = &buffer_views[@intFromEnum(linear_op.in)];
                    const output_features: usize = if (weight.dtype == .f32)
                        @intCast(weight.shape[1]) // f32 weights are [in, out]
                    else
                        @intCast(weight.shape[0]); // bf16/f16/quantized weights are [out, in]

                    // Create 2D views for matmul
                    // Input: use buffer's current shape
                    const input_view = Tensor.view2D(input_tensor.data(), @intCast(input_tensor.shape[1]), @intCast(input_tensor.shape[2]));

                    // Output buffer selection:
                    // Default to scratch.tmp[2] (branch_out), but if input is also in tmp[2], we must use an alternate buffer
                    // to avoid aliasing (matmul cannot handle overlapping input/output).
                    // IMPORTANT: For odd-indexed layers, the residual buffer (`out`) points to layer_tmp (tmp[0]),
                    // so we can't use layer_tmp as escape hatch - use tmp[1] (norm_out) instead.
                    const output_slice = blk: {
                        // Direct output to specific buffer_views if requested
                        const out_idx = @intFromEnum(linear_op.out);
                        if (out_idx >= 3 and out_idx < common.block_kernels.NUM_TMP_BUFFERS) {
                            break :blk scratch.tmp[out_idx][0 .. seq_len * output_features];
                        }
                        if (linear_op.out == .norm_out) {
                            break :blk scratch.tmp[1][0 .. seq_len * output_features];
                        }

                        const input_ptr = @intFromPtr(input_tensor.data().ptr);
                        const branch_ptr = @intFromPtr(scratch.tmp[2].ptr);
                        const input_aliases_branch = (input_ptr == branch_ptr);

                        // Check if residual uses layer_tmp (odd-indexed layers)
                        const residual_ptr = @intFromPtr(buffer_views[@intFromEnum(BufferId.residual)].data().ptr);
                        const layer_tmp_buf_ptr = @intFromPtr(scratch.tmp[0].ptr); // tmp[0] is layer_tmp
                        const residual_uses_layer_tmp = (residual_ptr == layer_tmp_buf_ptr);

                        break :blk if (input_aliases_branch)
                            // Use tmp[1] (norm_out) if residual is in layer_tmp, otherwise use layer_tmp
                            if (residual_uses_layer_tmp)
                                scratch.tmp[1][0 .. seq_len * output_features]
                            else
                                scratch.tmp[0][0 .. seq_len * output_features]
                        else
                            scratch.tmp[2][0 .. seq_len * output_features];
                    };

                    const out_byte_size = seq_len * output_features * @sizeOf(f32);
                    var output_view = Tensor.view2D(std.mem.sliceAsBytes(output_slice), seq_len, output_features);

                    // Use the appropriate matmul kernel based on weight dtype
                    const dk = common.matmul.matmulKernel(weight.dtype) catch |err| {
                        capi.setContext("block={d}, op={d}, weight={s}, dtype={}", .{ self.block_idx, op_index, linear_op.weight_name, weight.dtype });
                        return err;
                    };
                    dk.func(&input_view, weight, &output_view, &scratch.matmul_scratch);

                    // Update output buffer to point to the result with correct shape
                    const out_bytes = std.mem.sliceAsBytes(output_slice)[0..out_byte_size];
                    buffer_views[@intFromEnum(linear_op.out)] = Tensor.view(out_bytes.ptr, &.{ 1, seq_len, output_features }, .f32, null);
                },

                .split => |split_op| {
                    // Split tensor along last dimension into multiple outputs
                    // Input is [1, seq, total_dim], outputs are [1, seq, split_size_i]

                    // Guard: num_outputs must fit in scratch buffer array starting at out_start.
                    const out_start_idx = @intFromEnum(split_op.out_start);
                    const max_outputs = common.block_kernels.NUM_TMP_BUFFERS - out_start_idx;
                    if (out_start_idx < @intFromEnum(BufferId.tmp3) or split_op.num_outputs > max_outputs) {
                        return error.TooManySplitOutputs;
                    }

                    const input_tensor = &buffer_views[@intFromEnum(split_op.in)];
                    const input_data = input_tensor.asSlice(f32);

                    // For 3D tensor [1, seq, dim], we split along dim (last axis)
                    const total_dim: usize = @intCast(input_tensor.shape[2]); // Last dimension

                    // Calculate actual split sizes
                    // The traced split_sizes may be from dummy config, so compute from model params:
                    // For QKV split: use n_heads, n_kv_heads, head_dim
                    // For gate_up split: use intermediate_size
                    const attn_ptr = self.block.getAttention();

                    var actual_sizes: [3]usize = undefined; // filled based on split type below
                    if (split_op.num_outputs == 3 and attn_ptr != null) {
                        // QKV split: [Q, K, V] = [n_heads*head_dim, n_kv_heads*head_dim, n_kv_heads*head_dim]
                        const attn = attn_ptr.?;
                        actual_sizes[0] = attn.n_heads * attn.head_dim;
                        actual_sizes[1] = attn.n_kv_heads * attn.head_dim;
                        actual_sizes[2] = attn.n_kv_heads * attn.head_dim;
                    } else if (split_op.num_outputs == 2) {
                        // gate_up split: equal halves
                        actual_sizes[0] = total_dim / 2;
                        actual_sizes[1] = total_dim / 2;
                    } else {
                        // Default: equal split
                        for (0..split_op.num_outputs) |out_idx| {
                            actual_sizes[out_idx] = total_dim / split_op.num_outputs;
                        }
                    }

                    // Calculate split sizes and copy data for each output
                    // We need to copy because split creates non-contiguous views
                    var dim_offset: usize = 0;
                    var split_idx: u8 = 0;
                    while (split_idx < split_op.num_outputs) : (split_idx += 1) {
                        const split_size: usize = actual_sizes[split_idx];

                        // Allocate output buffer based on output index
                        // tmp3, tmp4, tmp5, etc. for split outputs
                        const out_idx = @intFromEnum(split_op.out_start) + split_idx;
                        const out_elems = seq_len * split_size;

                        // Use the output index as the scratch buffer selector.
                        const out_slice = scratch.tmp[out_idx][0..out_elems];

                        // Copy data: for each sequence position, copy the slice
                        for (0..seq_len) |seq_idx| {
                            const src_base = seq_idx * total_dim + dim_offset;
                            const dst_base = seq_idx * split_size;
                            @memcpy(out_slice[dst_base..][0..split_size], input_data[src_base..][0..split_size]);
                        }

                        const byte_size = out_elems * @sizeOf(f32);
                        const out_bytes = std.mem.sliceAsBytes(out_slice)[0..byte_size];
                        buffer_views[out_idx] = Tensor.view(out_bytes.ptr, &.{ 1, seq_len, split_size }, .f32, null);

                        dim_offset += split_size;
                    }
                },

                .matmul => |matmul_op| {
                    // Matrix multiplication: out = a @ b
                    const left_input = &buffer_views[@intFromEnum(matmul_op.in_a)];
                    const right_input = &buffer_views[@intFromEnum(matmul_op.in_b)];

                    // Compute output dimensions: [m, k] @ [k, n] = [m, n]
                    // For attention Q@K: [seq, head_dim] @ [seq, head_dim].T = [seq, seq]
                    // Note: matmul uses BF16 convention where B is [n, k] not [k, n]
                    const m_dim: usize = @intCast(left_input.shape[1]); // seq
                    const n_dim: usize = @intCast(right_input.shape[1]); // For Q@K, this would be seq (after reshape)

                    // Allocate output using layer_tmp (tmp[0])
                    const out_size = m_dim * n_dim;
                    const out_slice = scratch.tmp[0][0..out_size];
                    const out_byte_size = out_size * @sizeOf(f32);

                    // Create output tensor view
                    var output_view = Tensor.view2D(std.mem.sliceAsBytes(out_slice), m_dim, n_dim);

                    // Create 2D views for inputs
                    const a_view = Tensor.view2D(left_input.data(), @intCast(left_input.shape[1]), @intCast(left_input.shape[2]));
                    const b_view = Tensor.view2D(right_input.data(), @intCast(right_input.shape[1]), @intCast(right_input.shape[2]));

                    try common.matmul.matmulAuto(&a_view, &b_view, &output_view, &scratch.matmul_scratch);

                    // Store result in buffer
                    const out_bytes = std.mem.sliceAsBytes(out_slice)[0..out_byte_size];
                    buffer_views[@intFromEnum(matmul_op.out)] = Tensor.view(out_bytes.ptr, &.{ 1, m_dim, n_dim }, .f32, null);
                },

                .softmax => |softmax_op| {
                    // Softmax activation
                    const input_tensor = &buffer_views[@intFromEnum(softmax_op.in)];
                    const output_tensor = &buffer_views[@intFromEnum(softmax_op.out)];

                    const input_view = tv.fromTensor(Tensor, input_tensor);
                    const output_view = tv.fromTensor(Tensor, output_tensor);
                    ops.activation.softmax(output_view, input_view);
                },

                .silu => |silu_op| {
                    // SiLU/Swish activation
                    const input_tensor = &buffer_views[@intFromEnum(silu_op.in)];
                    const output_tensor = &buffer_views[@intFromEnum(silu_op.out)];

                    const input_view = tv.fromTensor(Tensor, input_tensor);
                    const output_view = tv.fromTensor(Tensor, output_tensor);
                    ops.activation.silu(output_view, input_view);
                },

                .gelu => |gelu_op| {
                    // GELU activation
                    const input_tensor = &buffer_views[@intFromEnum(gelu_op.in)];
                    const output_tensor = &buffer_views[@intFromEnum(gelu_op.out)];

                    const input_view = tv.fromTensor(Tensor, input_tensor);
                    const output_view = tv.fromTensor(Tensor, output_tensor);
                    ops.activation.gelu(output_view, input_view);
                },

                .mul => |mul_op| {
                    // Element-wise multiply (with broadcasting)
                    const left_tensor = buffer_views[@intFromEnum(mul_op.in)];
                    const right_tensor = buffer_views[@intFromEnum(mul_op.other)];
                    const output_len = @max(left_tensor.numel, right_tensor.numel);

                    const output_slice = resolveOutputSlice(&buffer_views, scratch, mul_op.out, output_len);
                    try applyElementwiseBinaryOp(left_tensor, right_tensor, output_slice, struct {
                        fn multiply(a: f32, b: f32) f32 {
                            return a * b;
                        }
                    }.multiply);

                    const output_shape = if (left_tensor.numel >= right_tensor.numel)
                        left_tensor.shape
                    else
                        right_tensor.shape;
                    const output_dims: i32 = if (left_tensor.numel >= right_tensor.numel)
                        left_tensor.n_dims
                    else
                        right_tensor.n_dims;
                    buffer_views[@intFromEnum(mul_op.out)] = tensorFromSlice(output_slice[0..output_len], output_shape, output_dims);
                },

                .add_tensor => |add_tensor_op| {
                    // Element-wise add (with broadcasting)
                    const left_tensor = buffer_views[@intFromEnum(add_tensor_op.in_a)];
                    const right_tensor = buffer_views[@intFromEnum(add_tensor_op.in_b)];
                    const output_len = @max(left_tensor.numel, right_tensor.numel);

                    const output_slice = resolveOutputSlice(&buffer_views, scratch, add_tensor_op.out, output_len);
                    try applyElementwiseBinaryOp(left_tensor, right_tensor, output_slice, struct {
                        fn addScalar(lhs: f32, rhs: f32) f32 {
                            return lhs + rhs;
                        }
                    }.addScalar);

                    const output_shape = if (left_tensor.numel >= right_tensor.numel)
                        left_tensor.shape
                    else
                        right_tensor.shape;
                    const output_dims: i32 = if (left_tensor.numel >= right_tensor.numel)
                        left_tensor.n_dims
                    else
                        right_tensor.n_dims;
                    buffer_views[@intFromEnum(add_tensor_op.out)] = tensorFromSlice(output_slice[0..output_len], output_shape, output_dims);
                },

                .add_scalar => |add_scalar_op| {
                    const input_tensor = buffer_views[@intFromEnum(add_scalar_op.in)];
                    const input_data = input_tensor.asSlice(f32);
                    const output_len = input_tensor.numel;

                    const output_slice = resolveOutputSlice(&buffer_views, scratch, add_scalar_op.out, output_len);
                    for (0..output_len) |elem_idx| {
                        output_slice[elem_idx] = input_data[elem_idx] + add_scalar_op.scalar;
                    }

                    buffer_views[@intFromEnum(add_scalar_op.out)] = tensorFromSlice(output_slice[0..output_len], input_tensor.shape, input_tensor.n_dims);
                },

                .mul_scalar => |mul_scalar_op| {
                    const input_tensor = buffer_views[@intFromEnum(mul_scalar_op.in)];
                    const input_data = input_tensor.asSlice(f32);
                    const output_len = input_tensor.numel;

                    const output_slice = resolveOutputSlice(&buffer_views, scratch, mul_scalar_op.out, output_len);
                    for (0..output_len) |elem_idx| {
                        output_slice[elem_idx] = input_data[elem_idx] * mul_scalar_op.scalar;
                    }

                    buffer_views[@intFromEnum(mul_scalar_op.out)] = tensorFromSlice(output_slice[0..output_len], input_tensor.shape, input_tensor.n_dims);
                },

                .mean => |mean_op| {
                    const input_tensor = buffer_views[@intFromEnum(mean_op.in)];
                    const input_data = input_tensor.asSlice(f32);

                    if (input_tensor.n_dims == 4) {
                        if (mean_op.dim != -1 and mean_op.dim != 3) return error.UnsupportedMeanDim;

                        const mean_seq_len: usize = @intCast(input_tensor.shape[1]);
                        const head_count: usize = @intCast(input_tensor.shape[2]);
                        const hidden_size: usize = @intCast(input_tensor.shape[3]);
                        const output_len = mean_seq_len * head_count;
                        const output_slice = resolveOutputSlice(&buffer_views, scratch, mean_op.out, output_len);

                        for (0..mean_seq_len) |t| {
                            for (0..head_count) |h| {
                                const base = (t * head_count + h) * hidden_size;
                                var sum: f32 = 0.0;
                                for (0..hidden_size) |d| sum += input_data[base + d];
                                output_slice[t * head_count + h] = sum / @as(f32, @floatFromInt(hidden_size));
                            }
                        }

                        const mean_shape: [8]i64 = if (mean_op.keepdim) .{ 1, @as(i64, @intCast(mean_seq_len)), @as(i64, @intCast(head_count)), 1, 0, 0, 0, 0 } else .{ 1, @as(i64, @intCast(mean_seq_len)), @as(i64, @intCast(head_count)), 0, 0, 0, 0, 0 };
                        const mean_dims: i32 = if (mean_op.keepdim) 4 else 3;
                        buffer_views[@intFromEnum(mean_op.out)] = tensorFromSlice(output_slice[0..output_len], mean_shape, mean_dims);
                    } else {
                        if (mean_op.dim != -1 and mean_op.dim != 2) return error.UnsupportedMeanDim;

                        const mean_seq_len_3d: usize = @intCast(input_tensor.shape[1]);
                        const hidden_size: usize = @intCast(input_tensor.shape[2]);
                        const output_len = mean_seq_len_3d;
                        const output_slice = resolveOutputSlice(&buffer_views, scratch, mean_op.out, output_len);

                        for (0..mean_seq_len_3d) |t| {
                            const base = t * hidden_size;
                            var sum: f32 = 0.0;
                            for (0..hidden_size) |h| sum += input_data[base + h];
                            output_slice[t] = sum / @as(f32, @floatFromInt(hidden_size));
                        }

                        const mean_shape_3d: [8]i64 = if (mean_op.keepdim) .{ 1, @as(i64, @intCast(mean_seq_len_3d)), 1, 0, 0, 0, 0, 0 } else .{ 1, @as(i64, @intCast(mean_seq_len_3d)), 0, 0, 0, 0, 0, 0 };
                        const mean_dims_3d: i32 = if (mean_op.keepdim) 3 else 2;
                        buffer_views[@intFromEnum(mean_op.out)] = tensorFromSlice(output_slice[0..output_len], mean_shape_3d, mean_dims_3d);
                    }
                },

                .pow => |pow_op| {
                    const input_tensor = buffer_views[@intFromEnum(pow_op.in)];
                    const input_data = input_tensor.asSlice(f32);
                    const output_len = input_tensor.numel;
                    const output_slice = resolveOutputSlice(&buffer_views, scratch, pow_op.out, output_len);

                    for (0..output_len) |elem_idx| {
                        output_slice[elem_idx] = std.math.pow(f32, input_data[elem_idx], pow_op.exponent);
                    }

                    buffer_views[@intFromEnum(pow_op.out)] = tensorFromSlice(output_slice[0..output_len], input_tensor.shape, input_tensor.n_dims);
                },

                .rsqrt => |rsqrt_op| {
                    const input_tensor = buffer_views[@intFromEnum(rsqrt_op.in)];
                    const input_data = input_tensor.asSlice(f32);
                    const output_len = input_tensor.numel;
                    const output_slice = resolveOutputSlice(&buffer_views, scratch, rsqrt_op.out, output_len);

                    for (0..output_len) |elem_idx| {
                        output_slice[elem_idx] = 1.0 / std.math.sqrt(input_data[elem_idx]);
                    }

                    buffer_views[@intFromEnum(rsqrt_op.out)] = tensorFromSlice(output_slice[0..output_len], input_tensor.shape, input_tensor.n_dims);
                },

                .add_param => |add_param_op| {
                    const input_tensor = buffer_views[@intFromEnum(add_param_op.in)];
                    const input_data = input_tensor.asSlice(f32);
                    const param = self.block.weight_registry.get(add_param_op.param_name) orelse return error.MissingParam;

                    const output_len = @max(input_tensor.numel, param.numel);
                    const output_slice = resolveOutputSlice(&buffer_views, scratch, add_param_op.out, output_len);

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
                        // Fallback: element-wise add requires matching sizes
                        const p_len = param.numel;
                        if (input_tensor.numel != p_len) {
                            return error.InvalidBroadcast;
                        }
                        for (0..p_len) |elem_idx| {
                            output_slice[elem_idx] = input_data[elem_idx] + readParamValue(param, elem_idx);
                        }
                    }

                    buffer_views[@intFromEnum(add_param_op.out)] = tensorFromSlice(output_slice[0..output_len], input_tensor.shape, input_tensor.n_dims);
                },

                .add_param_scalar => |add_param_scalar_op| {
                    const param = self.block.weight_registry.get(add_param_scalar_op.param_name) orelse return error.MissingParam;
                    const p_len = param.numel;
                    const output_slice = resolveOutputSlice(&buffer_views, scratch, add_param_scalar_op.out, p_len);
                    for (0..p_len) |elem_idx| {
                        output_slice[elem_idx] = readParamValue(param, elem_idx) + add_param_scalar_op.scalar;
                    }

                    buffer_views[@intFromEnum(add_param_scalar_op.out)] = tensorFromSlice(output_slice[0..p_len], param.shape, param.n_dims);
                },

                .mul_param => |mul_param_op| {
                    const input_tensor = buffer_views[@intFromEnum(mul_param_op.in)];
                    const input_data = input_tensor.asSlice(f32);
                    const param = self.block.weight_registry.get(mul_param_op.param_name) orelse return error.MissingParam;

                    const output_len = @max(input_tensor.numel, param.numel);
                    const output_slice = resolveOutputSlice(&buffer_views, scratch, mul_param_op.out, output_len);

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
                        // Fallback: element-wise multiply requires matching sizes
                        const p_len = param.numel;
                        if (input_tensor.numel != p_len) {
                            return error.InvalidBroadcast;
                        }
                        for (0..p_len) |elem_idx| {
                            output_slice[elem_idx] = input_data[elem_idx] * readParamValue(param, elem_idx);
                        }
                    }

                    buffer_views[@intFromEnum(mul_param_op.out)] = tensorFromSlice(output_slice[0..output_len], input_tensor.shape, input_tensor.n_dims);
                },

                .reshape => |reshape_op| {
                    // Reshape is a view operation - update metadata only.
                    // Note: shape inference is limited to common cases; full view tracking is pending.
                    const input_tensor = &buffer_views[@intFromEnum(reshape_op.in)];
                    var output_tensor = input_tensor.*;

                    if (reshape_op.shape.len > 0) {
                        var out_shape: [8]i64 = .{ 0, 0, 0, 0, 0, 0, 0, 0 };
                        var inferred_dim_idx: ?usize = null;
                        var known_product: usize = 1;
                        const total_elems = input_tensor.numel;

                        const n_dims: usize = @min(reshape_op.shape.len, out_shape.len);
                        for (reshape_op.shape[0..n_dims], 0..) |dim, dim_idx| {
                            if (dim == -1) {
                                inferred_dim_idx = dim_idx;
                                continue;
                            }
                            const resolved: i64 = switch (dim) {
                                -2 => input_tensor.shape[0], // B
                                -3 => input_tensor.shape[1], // T
                                else => dim,
                            };
                            out_shape[dim_idx] = resolved;
                            known_product *= @intCast(resolved);
                        }

                        if (inferred_dim_idx) |dim_idx| {
                            if (known_product == 0) return error.InvalidReshape;
                            out_shape[dim_idx] = @intCast(total_elems / known_product);
                        }

                        output_tensor.shape = out_shape;
                        output_tensor.n_dims = @intCast(n_dims);
                    } else if (input_tensor.n_dims == 3) {
                        const reshape_seq_len = input_tensor.shape[1];
                        const hidden = input_tensor.shape[2];
                        const attn_info = self.block.getAttention() orelse return error.AttentionNotAvailable;
                        const heads: i64 = @intCast(attn_info.n_heads);
                        const kv_heads: i64 = @intCast(attn_info.n_kv_heads);
                        const head_dim: i64 = @intCast(attn_info.head_dim);
                        if (hidden == heads * head_dim) {
                            output_tensor.shape = .{ 1, reshape_seq_len, heads, head_dim, 0, 0, 0, 0 };
                            output_tensor.n_dims = 4;
                        } else if (hidden == kv_heads * head_dim) {
                            output_tensor.shape = .{ 1, reshape_seq_len, kv_heads, head_dim, 0, 0, 0, 0 };
                            output_tensor.n_dims = 4;
                        }
                    } else if (input_tensor.n_dims == 4) {
                        const reshape_seq_len_4d = input_tensor.shape[1];
                        const heads = input_tensor.shape[2];
                        const head_dim = input_tensor.shape[3];
                        output_tensor.shape = .{ 1, reshape_seq_len_4d, heads * head_dim, 0, 0, 0, 0, 0 };
                        output_tensor.n_dims = 3;
                    }

                    buffer_views[@intFromEnum(reshape_op.out)] = output_tensor;
                },

                .transpose => |transpose_op| {
                    // Transpose two dimensions of a tensor
                    const in_tensor = &buffer_views[@intFromEnum(transpose_op.in)];

                    // Get output buffer
                    const out_len = in_tensor.numel;
                    const out_slice = resolveOutputSlice(&buffer_views, scratch, transpose_op.out, out_len);

                    // Compute dim indices (handle negative dims)
                    const ndim: usize = @intCast(in_tensor.n_dims);
                    const dim0: usize = if (transpose_op.dim0 < 0)
                        @intCast(@as(i64, @intCast(ndim)) + transpose_op.dim0)
                    else
                        @intCast(transpose_op.dim0);
                    const dim1: usize = if (transpose_op.dim1 < 0)
                        @intCast(@as(i64, @intCast(ndim)) + transpose_op.dim1)
                    else
                        @intCast(transpose_op.dim1);

                    // Convert i64 shape to usize for TensorView
                    var in_shape_dims: [8]usize = undefined;
                    for (0..ndim) |dim_idx| {
                        in_shape_dims[dim_idx] = @intCast(in_tensor.shape[dim_idx]);
                    }
                    for (ndim..8) |dim_idx| {
                        in_shape_dims[dim_idx] = 0;
                    }

                    // Compute transposed shape (usize)
                    var out_shape_dims: [8]usize = in_shape_dims;
                    const tmp_dim = out_shape_dims[dim0];
                    out_shape_dims[dim0] = out_shape_dims[dim1];
                    out_shape_dims[dim1] = tmp_dim;

                    // Create views using initContiguous
                    const in_view = tv.TensorView.initContiguous(
                        in_tensor.data_ptr.?,
                        in_shape_dims[0..ndim],
                        .f32,
                    );
                    const out_view = tv.TensorView.initContiguous(
                        @ptrCast(out_slice.ptr),
                        out_shape_dims[0..ndim],
                        .f32,
                    );

                    ops.shape.transposeDispatch(out_view, in_view, dim0, dim1);

                    // Convert back to i64 shape for output tensor
                    var out_shape_i64: [8]i64 = in_tensor.shape;
                    const tmp_i64 = out_shape_i64[dim0];
                    out_shape_i64[dim0] = out_shape_i64[dim1];
                    out_shape_i64[dim1] = tmp_i64;

                    buffer_views[@intFromEnum(transpose_op.out)] = tensorFromSlice(out_slice[0..out_len], out_shape_i64, in_tensor.n_dims);
                },

                .rope => |rope_op| {
                    // Standalone RoPE for primitive mode
                    // Apply rotary position embedding in-place
                    const in_tensor = &buffer_views[@intFromEnum(rope_op.in)];
                    const input_data = in_tensor.asSlice(f32);

                    // Get RoPE from attention module
                    const attn = self.block.getAttention() orelse {
                        capi.setContext("block={d}, op={d}, type=mamba", .{ self.block_idx, op_index });
                        return error.RopeNotAvailableForMamba;
                    };
                    const rope = attn.rope orelse {
                        capi.setContext("block={d}, op={d}", .{ self.block_idx, op_index });
                        return error.MissingRopeConfig;
                    };

                    // Infer layout from shape
                    const ndim: usize = @intCast(in_tensor.n_dims);

                    // Get position offset from cache
                    const pos_offset = if (use_cache) attn_cache.cache_position else 0;

                    if (ndim == 2) {
                        // Shape [seq, dim] - apply rope to full dimension
                        const rope_seq_len_2d: usize = @intCast(in_tensor.shape[0]);
                        const dim: usize = @intCast(in_tensor.shape[1]);
                        const rope_dim = @min(rope.dim, dim);
                        for (0..rope_seq_len_2d) |t| {
                            const pos = pos_offset + t;
                            const base = t * dim;
                            rope.applyInPlace(input_data[base .. base + rope_dim], pos);
                        }
                    } else if (ndim == 3) {
                        // Shape [seq, n_heads, head_dim] - apply per head
                        const rope_seq_len_3d: usize = @intCast(in_tensor.shape[0]);
                        const n_heads: usize = @intCast(in_tensor.shape[1]);
                        const head_dim: usize = @intCast(in_tensor.shape[2]);
                        const rope_dim = @min(rope.dim, head_dim);
                        const total_dim = n_heads * head_dim;
                        for (0..rope_seq_len_3d) |t| {
                            const pos = pos_offset + t;
                            for (0..n_heads) |h| {
                                const base = t * total_dim + h * head_dim;
                                rope.applyInPlace(input_data[base .. base + rope_dim], pos);
                            }
                        }
                    } else if (ndim == 4) {
                        // Shape [batch, n_heads, seq, head_dim] - after transpose(1,2)
                        // This is the standard layout after Q/K projection and transpose
                        const batch: usize = @intCast(in_tensor.shape[0]);
                        const n_heads: usize = @intCast(in_tensor.shape[1]);
                        const rope_seq_len_4d: usize = @intCast(in_tensor.shape[2]);
                        const head_dim: usize = @intCast(in_tensor.shape[3]);
                        const rope_dim = @min(rope.dim, head_dim);
                        const head_stride = rope_seq_len_4d * head_dim;
                        const batch_stride = n_heads * head_stride;
                        for (0..batch) |b| {
                            for (0..n_heads) |h| {
                                for (0..rope_seq_len_4d) |t| {
                                    const pos = pos_offset + t;
                                    const base = b * batch_stride + h * head_stride + t * head_dim;
                                    rope.applyInPlace(input_data[base .. base + rope_dim], pos);
                                }
                            }
                        }
                    } else {
                        capi.setContext("block={d}, op={d}, ndim={d}", .{ self.block_idx, op_index, ndim });
                        return error.UnsupportedRopeShape;
                    }

                    // Copy to output if different buffer
                    if (rope_op.in != rope_op.out) {
                        const out_slice = resolveOutputSlice(&buffer_views, scratch, rope_op.out, in_tensor.numel);
                        @memcpy(out_slice, input_data);
                        buffer_views[@intFromEnum(rope_op.out)] = tensorFromSlice(out_slice[0..in_tensor.numel], in_tensor.shape, in_tensor.n_dims);
                    }
                },

                .triu => |triu_op| {
                    // Upper triangular mask for causal attention
                    // Set elements below diagonal to -inf
                    const in_tensor = &buffer_views[@intFromEnum(triu_op.in)];
                    const out_buf = &buffer_views[@intFromEnum(triu_op.out)];

                    const data = in_tensor.asSlice(f32);
                    const out_data = out_buf.asSlice(f32);

                    // Assume 2D [seq, seq] or 3D [batch, seq, seq]
                    const n_dims: usize = @intCast(in_tensor.n_dims);
                    const rows: usize = @intCast(in_tensor.shape[n_dims - 2]);
                    const cols: usize = @intCast(in_tensor.shape[n_dims - 1]);
                    const neg_inf = -std.math.inf(f32);

                    for (0..rows) |row| {
                        for (0..cols) |col| {
                            const elem_offset = row * cols + col;
                            const signed_col: i64 = @intCast(col);
                            const signed_row: i64 = @intCast(row);
                            if (signed_col < signed_row + triu_op.diagonal) {
                                out_data[elem_offset] = neg_inf;
                            } else {
                                out_data[elem_offset] = data[elem_offset];
                            }
                        }
                    }
                },

                .sdpa => |sdpa_op| {
                    // Scaled dot-product attention (PyTorch-compatible 4D layout)
                    // Q/K/V: [batch, heads, seq, head_dim]
                    // Output: [batch, heads, seq, head_dim]
                    // Pure attention op - RoPE and KV cache are separate ops in primitive mode

                    const query_buf = &buffer_views[@intFromEnum(sdpa_op.q)];
                    const key_buf = &buffer_views[@intFromEnum(sdpa_op.k)];
                    const value_buf = &buffer_views[@intFromEnum(sdpa_op.v)];

                    // Verify 4D shape
                    if (query_buf.n_dims != 4) {
                        capi.setContext("block={d}, op={d}, got {d}D, need 4D", .{ self.block_idx, op_index, query_buf.n_dims });
                        return error.InvalidShape;
                    }

                    const batch: usize = @intCast(query_buf.shape[0]);
                    const n_heads: usize = @intCast(query_buf.shape[1]);
                    const seq_q: usize = @intCast(query_buf.shape[2]);
                    const head_dim: usize = @intCast(query_buf.shape[3]);
                    const seq_k: usize = @intCast(key_buf.shape[2]);

                    // Compute scale
                    const scale = sdpa_op.scale orelse 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

                    // Allocate output buffer
                    const out_numel = batch * n_heads * seq_q * head_dim;
                    const out_slice = resolveOutputSlice(&buffer_views, scratch, sdpa_op.out, out_numel);

                    // Create TensorViews for the attention op
                    const q_shape = [_]usize{ batch, n_heads, seq_q, head_dim };
                    const k_shape = [_]usize{ batch, n_heads, seq_k, head_dim };
                    const v_shape = [_]usize{ batch, n_heads, seq_k, head_dim };
                    const out_shape = [_]usize{ batch, n_heads, seq_q, head_dim };

                    const q_view = tv.TensorView.initContiguous(query_buf.data_ptr.?, &q_shape, .f32);
                    const k_view = tv.TensorView.initContiguous(key_buf.data_ptr.?, &k_shape, .f32);
                    const v_view = tv.TensorView.initContiguous(value_buf.data_ptr.?, &v_shape, .f32);
                    const out_view = tv.TensorView.initContiguous(@ptrCast(out_slice.ptr), &out_shape, .f32);

                    // Call the attention kernel
                    if (sdpa_op.is_causal) {
                        // Use causal version (no explicit mask needed)
                        ops.attention.sdpaCausal(out_view, q_view, k_view, v_view, scale, 0, scratch.allocator) catch |err| {
                            capi.setContext("block={d}, op={d}, causal=true", .{ self.block_idx, op_index });
                            return err;
                        };
                    } else {
                        // Non-causal (no mask)
                        ops.attention.sdpa(out_view, q_view, k_view, v_view, null, scale, scratch.allocator) catch |err| {
                            capi.setContext("block={d}, op={d}, causal=false", .{ self.block_idx, op_index });
                            return err;
                        };
                    }

                    // Set output tensor metadata
                    const sdpa_shape: [8]i64 = .{
                        @as(i64, @intCast(batch)),
                        @as(i64, @intCast(n_heads)),
                        @as(i64, @intCast(seq_q)),
                        @as(i64, @intCast(head_dim)),
                        0, 0, 0, 0,
                    };
                    buffer_views[@intFromEnum(sdpa_op.out)] = tensorFromSlice(out_slice[0..out_numel], sdpa_shape, 4);
                },

            }
        }

        // Post-norm finalization: if the program's final output is not in the residual
        // buffer (e.g., post-norm architectures like BERT end with a norm → norm_out),
        // copy the result to residual so the caller sees it in `out`.
        const final_buf = graph.compiler.finalOutputBuffer(self.program);
        if (final_buf != .residual) {
            copyTensor(&buffer_views[@intFromEnum(final_buf)], &buffer_views[@intFromEnum(BufferId.residual)]);
        }
    }

    /// Validate the program against the block's weight registry and supported ops.
    /// This is intended for load-time checks to catch invalid graphs early.
    pub fn validate(self: *const Block) !void {
        for (self.program, 0..) |op, op_index| {
            switch (op) {
                .kernel => |kernel_op| {
                    const kernel_id: usize = @intCast(kernel_op.id);
                    if (kernel_id >= self.block.kernels.len) {
                        capi.setContext("block={d}, kernel_id={d}, max={d}", .{ self.block_idx, kernel_op.id, self.block.kernels.len });
                        return error.KernelIndexOutOfBounds;
                    }
                },
                .linear => |linear_op| {
                    if (self.block.weight_registry.get(linear_op.weight_name) == null) {
                        capi.setContext("block={d}, op={d}, weight={s}", .{ self.block_idx, op_index, linear_op.weight_name });
                        return error.MissingWeight;
                    }
                    const weight = self.block.weight_registry.get(linear_op.weight_name).?;
                    _ = common.matmul.matmulKernel(weight.dtype) catch |err| {
                        capi.setContext("block={d}, op={d}, weight={s}, dtype={}", .{ self.block_idx, op_index, linear_op.weight_name, weight.dtype });
                        return err;
                    };
                },
                .add_param => |add_param_op| {
                    if (self.block.weight_registry.get(add_param_op.param_name) == null) {
                        capi.setContext("block={d}, op={d}, param={s}", .{ self.block_idx, op_index, add_param_op.param_name });
                        return error.MissingParam;
                    }
                },
                .add_param_scalar => |add_param_scalar_op| {
                    if (self.block.weight_registry.get(add_param_scalar_op.param_name) == null) {
                        capi.setContext("block={d}, op={d}, param={s}", .{ self.block_idx, op_index, add_param_scalar_op.param_name });
                        return error.MissingParam;
                    }
                },
                .mul_param => |mul_param_op| {
                    if (self.block.weight_registry.get(mul_param_op.param_name) == null) {
                        capi.setContext("block={d}, op={d}, param={s}", .{ self.block_idx, op_index, mul_param_op.param_name });
                        return error.MissingParam;
                    }
                },
                .split => |split_op| {
                    const out_start_idx = @intFromEnum(split_op.out_start);
                    const max_outputs = common.block_kernels.NUM_TMP_BUFFERS - out_start_idx;
                    if (out_start_idx < @intFromEnum(BufferId.tmp3) or split_op.num_outputs == 0 or split_op.num_outputs > max_outputs) {
                        return error.TooManySplitOutputs;
                    }
                },
                .rope, .transpose, .sdpa => {
                    // implemented
                },
                else => {},
            }
        }
    }

    /// Describe block for introspection (hierarchical view by default)
    pub fn describe(self: *const Block, writer: anytype, indent: usize, show_kernels: bool) !void {
        try writer.writeByteNTimes(' ', indent);
        try writer.print("(layers.{}): Block(\n", .{self.block_idx});

        // Hierarchical view: show attention and FFN modules (attention_mlp blocks only)
        if (self.block.getAttention()) |attn_info| {
            try writer.writeByteNTimes(' ', indent + 2);
            try writer.writeAll("(self_attn): ");
            try attn_info.describe(writer, indent + 2, show_kernels);
        }
        if (self.block.getFfnLayer()) |ffn| {
            try writer.writeByteNTimes(' ', indent + 2);
            try writer.writeAll("(ffn): ");
            try ffn.describe(writer, indent + 2, show_kernels);
        }
        if (self.block.getMambaKernel()) |mamba_k| {
            try writer.writeByteNTimes(' ', indent + 2);
            try writer.print("(mixer): Mamba(d_model={}, d_state={}, d_conv={})\n", .{
                mamba_k.config.d_model,
                mamba_k.config.d_state,
                mamba_k.config.d_conv,
            });
        }

        try writer.writeByteNTimes(' ', indent);
        try writer.writeAll(")\n");
    }

    /// Describe block showing operation sequence (topology view)
    pub fn describeTopology(self: *const Block, writer: anytype, indent: usize) !void {
        try writer.writeByteNTimes(' ', indent);
        try writer.print("(layers.{}): Block({} ops)\n", .{ self.block_idx, self.program.len });

        for (self.program, 0..) |op, op_index| {
            try writer.writeByteNTimes(' ', indent + 2);
            try writer.print("[{}] ", .{op_index});
            try self.writeLayerOpDescription(op, writer, 0);
        }
    }

    /// Get hidden size from this block
    pub fn getHiddenSize(self: *const Block) usize {
        return self.hidden_size;
    }

    /// Get block index
    pub fn getBlockIdx(self: *const Block) usize {
        return self.block_idx;
    }

    /// Get attention module (for parameter counting, etc.)
    /// Returns null for Mamba blocks.
    pub fn getAttention(self: *const Block) ?*const Attention {
        return self.block.getAttention();
    }

    /// Get FFN layer (for parameter counting, etc.)
    /// Returns null for Mamba blocks.
    pub fn getFFN(self: *const Block) ?*const FFNLayer {
        return self.block.getFfnLayer();
    }

    /// Forward pass using BatchedKVCache instead of AttnCache.
    /// This enables graph-based execution with batched caching for continuous batching.
    pub fn forwardWithBatchedCache(
        self: *const Block,
        x: *const Tensor,
        out: *Tensor,
        scratch: *ScratchBuffer,
        batched_cache: *BatchedKVCache,
        slot_index: usize,
        use_cache: bool,
    ) !void {
        std.debug.assert(x.shape[0] == 1 and out.shape[0] == 1);
        const seq_len: usize = @intCast(x.shape[1]);
        try scratch.ensure(seq_len);

        // Setup buffer views
        const norm_output_view = Tensor.view3DSlice(scratch.tmp[1], seq_len, self.hidden_size);
        const branch_output_view = Tensor.view3DSlice(scratch.tmp[2], seq_len, self.hidden_size);

        var buffer_views: [64]Tensor = undefined;
        buffer_views[@intFromEnum(BufferId.residual)] = out.*;
        buffer_views[@intFromEnum(BufferId.norm_out)] = norm_output_view;
        buffer_views[@intFromEnum(BufferId.branch_out)] = branch_output_view;

        copyTensor(x, out);

        // Get MLA cache/scratch if this is an MLA block
        const is_mla = self.block.isMLA();
        const mla_cache = if (is_mla) scratch.getMLACache(self.block_idx) else null;
        const mla_scratch = if (is_mla) scratch.getMLAScratch() else null;

        const ctx = KernelContext{
            .scratch = scratch,
            .attn_cache = null,
            .mla_cache = mla_cache,
            .mla_scratch = mla_scratch,
            .mamba_state = if (self.mamba_layer_idx) |idx| scratch.getMambaState(idx) else null,
            .mamba_scratch = scratch.getMambaScratch(),
            .shortconv_state = if (self.shortconv_layer_idx) |idx| scratch.getShortConvState(idx) else null,
            .shortconv_scratch = scratch.getShortConvScratch(),
            .matmul_scratch = &scratch.matmul_scratch,
            .use_cache = use_cache,
        };

        // Execute the operation sequence
        for (self.program) |op| {
            switch (op) {
                .kernel => |kernel_op| {
                    const kernel_id: usize = @intCast(kernel_op.id);
                    if (kernel_id >= self.block.kernels.len) {
                        capi.setContext("block={d}, kernel_id={d}, max={d}", .{ self.block_idx, kernel_op.id, self.block.kernels.len });
                        return error.KernelIndexOutOfBounds;
                    }
                    const kernel = self.block.kernels[kernel_id];
                    if (builtin.mode == .Debug) {
                        const actual_type = kernel.getOpType();
                        if (actual_type != kernel_op.debug_type) {
                            log.err("inference", "Graph/Kernel ordering mismatch", .{
                                .block = self.block_idx,
                                .kernel = kernel_op.id,
                                .expected = @tagName(kernel_op.debug_type),
                                .actual = @tagName(actual_type),
                            }, @src());
                            @panic("Graph/Kernel type mismatch - graph compiler and block init are out of sync");
                        }
                    }
                    const input = &buffer_views[@intFromEnum(kernel_op.in)];
                    const output = &buffer_views[@intFromEnum(kernel_op.out)];
                    try kernel.forwardBatched(input, output, ctx, batched_cache, slot_index);
                },
                .add => |add_op| {
                    addIntoScaled(
                        &buffer_views[@intFromEnum(BufferId.residual)],
                        &buffer_views[@intFromEnum(add_op.branch)],
                        &buffer_views[@intFromEnum(BufferId.residual)],
                        self.residualScaleValue(add_op.scale),
                    );
                },
                .mul_scalar => |mul_scalar_op| {
                    const input_tensor = buffer_views[@intFromEnum(mul_scalar_op.in)];
                    const input_data = input_tensor.asSlice(f32);
                    const output_len = input_tensor.numel;

                    const output_slice = resolveOutputSlice(&buffer_views, scratch, mul_scalar_op.out, output_len);
                    for (0..output_len) |elem_idx| {
                        output_slice[elem_idx] = input_data[elem_idx] * mul_scalar_op.scalar;
                    }

                    buffer_views[@intFromEnum(mul_scalar_op.out)] = tensorFromSlice(output_slice[0..output_len], input_tensor.shape, input_tensor.n_dims);
                },
                .add_tensor => |add_tensor_op| {
                    const left_tensor = buffer_views[@intFromEnum(add_tensor_op.in_a)];
                    const right_tensor = buffer_views[@intFromEnum(add_tensor_op.in_b)];
                    const output_len = @max(left_tensor.numel, right_tensor.numel);

                    const output_slice = resolveOutputSlice(&buffer_views, scratch, add_tensor_op.out, output_len);
                    try applyElementwiseBinaryOp(left_tensor, right_tensor, output_slice, struct {
                        fn addScalar(lhs: f32, rhs: f32) f32 {
                            return lhs + rhs;
                        }
                    }.addScalar);

                    const output_shape = if (left_tensor.numel >= right_tensor.numel)
                        left_tensor.shape
                    else
                        right_tensor.shape;
                    const output_dims: i32 = if (left_tensor.numel >= right_tensor.numel)
                        left_tensor.n_dims
                    else
                        right_tensor.n_dims;
                    buffer_views[@intFromEnum(add_tensor_op.out)] = tensorFromSlice(output_slice[0..output_len], output_shape, output_dims);
                },
                else => |other_op| {
                    // Other ops (linear, rope, etc.) not yet supported in batched mode
                    const op_name = @tagName(other_op);
                    capi.setContext("block={d}, unsupported_op={s}", .{ self.block_idx, op_name });
                    return error.UnsupportedOpInBatchedMode;
                },
            }
        }

        // Post-norm finalization: if the program's final output is not in the residual
        // buffer (e.g., post-norm architectures like BERT end with a norm → norm_out),
        // copy the result to residual so the caller sees it in `out`.
        const final_buf = graph.compiler.finalOutputBuffer(self.program);
        if (final_buf != .residual) {
            copyTensor(&buffer_views[@intFromEnum(final_buf)], &buffer_views[@intFromEnum(BufferId.residual)]);
        }
    }
};

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

// Helper struct to hold weight tensors for testing
const TestWeights = struct {
    q_weight: Tensor,
    k_weight: Tensor,
    v_weight: Tensor,
    o_weight: Tensor,
    w1_weight: Tensor,
    w2_weight: Tensor,
    w3_weight: Tensor,
    ln1_weight: Tensor,
    ln2_weight: Tensor,

    q_data: []f32,
    k_data: []f32,
    v_data: []f32,
    o_data: []f32,
    w1_data: []f32,
    w2_data: []f32,
    w3_data: []f32,
    ln1_data: []f32,
    ln2_data: []f32,

    fn init(allocator: std.mem.Allocator, d_model: usize, d_ff: usize, n_kv_heads: usize, head_dim: usize) !TestWeights {
        var self: TestWeights = undefined;

        self.q_data = try allocator.alloc(f32, d_model * d_model);
        errdefer allocator.free(self.q_data);
        @memset(self.q_data, 0.1);
        self.q_weight = Tensor.view(@ptrCast(self.q_data.ptr), &.{ d_model, d_model }, .f32, null);

        self.k_data = try allocator.alloc(f32, d_model * (n_kv_heads * head_dim));
        errdefer allocator.free(self.k_data);
        @memset(self.k_data, 0.1);
        self.k_weight = Tensor.view(@ptrCast(self.k_data.ptr), &.{ d_model, n_kv_heads * head_dim }, .f32, null);

        self.v_data = try allocator.alloc(f32, d_model * (n_kv_heads * head_dim));
        errdefer allocator.free(self.v_data);
        @memset(self.v_data, 0.1);
        self.v_weight = Tensor.view(@ptrCast(self.v_data.ptr), &.{ d_model, n_kv_heads * head_dim }, .f32, null);

        self.o_data = try allocator.alloc(f32, d_model * d_model);
        errdefer allocator.free(self.o_data);
        @memset(self.o_data, 0.1);
        self.o_weight = Tensor.view(@ptrCast(self.o_data.ptr), &.{ d_model, d_model }, .f32, null);

        self.w1_data = try allocator.alloc(f32, d_model * d_ff);
        errdefer allocator.free(self.w1_data);
        @memset(self.w1_data, 0.1);
        self.w1_weight = Tensor.view(@ptrCast(self.w1_data.ptr), &.{ d_model, d_ff }, .f32, null);

        self.w3_data = try allocator.alloc(f32, d_model * d_ff);
        errdefer allocator.free(self.w3_data);
        @memset(self.w3_data, 0.1);
        self.w3_weight = Tensor.view(@ptrCast(self.w3_data.ptr), &.{ d_model, d_ff }, .f32, null);

        self.w2_data = try allocator.alloc(f32, d_ff * d_model);
        errdefer allocator.free(self.w2_data);
        @memset(self.w2_data, 0.1);
        self.w2_weight = Tensor.view(@ptrCast(self.w2_data.ptr), &.{ d_ff, d_model }, .f32, null);

        self.ln1_data = try allocator.alloc(f32, d_model);
        errdefer allocator.free(self.ln1_data);
        for (self.ln1_data, 0..) |*w, i| w.* = 1.0 + @as(f32, @floatFromInt(i)) * 0.001;
        self.ln1_weight = Tensor.view(@ptrCast(self.ln1_data.ptr), &.{d_model}, .f32, null);

        self.ln2_data = try allocator.alloc(f32, d_model);
        for (self.ln2_data, 0..) |*w, i| w.* = 1.0 + @as(f32, @floatFromInt(i)) * 0.001;
        self.ln2_weight = Tensor.view(@ptrCast(self.ln2_data.ptr), &.{d_model}, .f32, null);

        return self;
    }

    fn deinit(self: *TestWeights, allocator: std.mem.Allocator) void {
        allocator.free(self.q_data);
        allocator.free(self.k_data);
        allocator.free(self.v_data);
        allocator.free(self.o_data);
        allocator.free(self.w1_data);
        allocator.free(self.w2_data);
        allocator.free(self.w3_data);
        allocator.free(self.ln1_data);
        allocator.free(self.ln2_data);
    }
};

// Helper to create a minimal TransformerBlock for testing
fn createTestTransformerBlock(allocator: std.mem.Allocator, weights: *TestWeights) !cpu_forward.TransformerBlock {
    const d_model = 128;
    const d_ff = 512;
    const n_heads = 4;
    const n_kv_heads = 2;
    const head_dim = 32;

    const block_weights: cpu_forward.BlockWeights = .{ .attention_mlp = .{
        .ln1_weight = &weights.ln1_weight,
        .q_proj = &weights.q_weight,
        .k_proj = &weights.k_weight,
        .v_proj = &weights.v_weight,
        .o_proj = &weights.o_weight,
        .ln2_weight = &weights.ln2_weight,
        .w1 = &weights.w1_weight,
        .w3 = &weights.w3_weight,
        .w2 = &weights.w2_weight,
    } };

    return try cpu_forward.TransformerBlock.init(
        allocator,
        d_model,
        d_ff,
        n_heads,
        n_kv_heads,
        head_dim,
        2048,
        block_weights,
        1e-5,
        .{}, // ModelRuntime with default values
        1.0,
        1.0,
        false,
        0,
    );
}

test "Block.getHiddenSize returns correct hidden size" {
    const allocator = testing.allocator;

    var weights = try TestWeights.init(allocator, 128, 512, 2, 32);
    defer weights.deinit(allocator);

    var transformer_block = try createTestTransformerBlock(allocator, &weights);
    defer transformer_block.deinit(allocator);

    const program = [_]LayerOp{
        .{ .kernel = .{ .id = 0, .in = .residual, .out = .norm_out, .debug_type = .norm } },
        .{ .kernel = .{ .id = 1, .in = .norm_out, .out = .branch_out, .debug_type = .multihead_attention } },
        .{ .add = .{ .branch = .branch_out, .scale = .one } },
    };

    const block = Block{
        .program = &program,
        .block = &transformer_block,
        .block_idx = 0,
        .hidden_size = 128,
    };

    try testing.expectEqual(@as(usize, 128), block.getHiddenSize());
}

test "Block.getBlockIdx returns correct block index" {
    const allocator = testing.allocator;

    var weights = try TestWeights.init(allocator, 128, 512, 2, 32);
    defer weights.deinit(allocator);

    var transformer_block = try createTestTransformerBlock(allocator, &weights);
    defer transformer_block.deinit(allocator);

    const program = [_]LayerOp{
        .{ .kernel = .{ .id = 0, .in = .residual, .out = .norm_out, .debug_type = .norm } },
    };

    const block = Block{
        .program = &program,
        .block = &transformer_block,
        .block_idx = 5,
        .hidden_size = 128,
    };

    try testing.expectEqual(@as(usize, 5), block.getBlockIdx());
}

test "Block.getAttention returns valid attention reference" {
    const allocator = testing.allocator;

    var weights = try TestWeights.init(allocator, 128, 512, 2, 32);
    defer weights.deinit(allocator);

    var transformer_block = try createTestTransformerBlock(allocator, &weights);
    defer transformer_block.deinit(allocator);

    const program = [_]LayerOp{
        .{ .kernel = .{ .id = 1, .in = .norm_out, .out = .branch_out, .debug_type = .multihead_attention } },
    };

    const block = Block{
        .program = &program,
        .block = &transformer_block,
        .block_idx = 0,
        .hidden_size = 128,
    };

    const attention = block.getAttention() orelse return error.AttentionNotAvailable;
    try testing.expectEqual(@as(usize, 4), attention.n_heads);
    try testing.expectEqual(@as(usize, 2), attention.n_kv_heads);
    try testing.expectEqual(@as(usize, 32), attention.head_dim);
}

test "Block.getFFN returns valid FFN reference" {
    const allocator = testing.allocator;

    var weights = try TestWeights.init(allocator, 128, 512, 2, 32);
    defer weights.deinit(allocator);

    var transformer_block = try createTestTransformerBlock(allocator, &weights);
    defer transformer_block.deinit(allocator);

    const program = [_]LayerOp{
        .{ .kernel = .{ .id = 3, .in = .norm_out, .out = .branch_out, .debug_type = .mlp } },
    };

    const block = Block{
        .program = &program,
        .block = &transformer_block,
        .block_idx = 0,
        .hidden_size = 128,
    };

    const ffn = block.getFFN() orelse return error.FfnNotAvailable;
    switch (ffn.*) {
        .swiglu => |mlp| {
            try testing.expectEqual(@as(usize, 512), mlp.d_ff);
        },
        .moe_ffn => return error.UnexpectedFFNType,
    }
}

test "Block.validate accepts valid program with all required weights" {
    const allocator = testing.allocator;

    var weights = try TestWeights.init(allocator, 128, 512, 2, 32);
    defer weights.deinit(allocator);

    var transformer_block = try createTestTransformerBlock(allocator, &weights);
    defer transformer_block.deinit(allocator);

    const program = [_]LayerOp{
        .{ .kernel = .{ .id = 0, .in = .residual, .out = .norm_out, .debug_type = .norm } },
        .{ .kernel = .{ .id = 1, .in = .norm_out, .out = .branch_out, .debug_type = .multihead_attention } },
        .{ .add = .{ .branch = .branch_out, .scale = .one } },
        .{ .kernel = .{ .id = 2, .in = .residual, .out = .norm_out, .debug_type = .norm } },
        .{ .kernel = .{ .id = 3, .in = .norm_out, .out = .branch_out, .debug_type = .mlp } },
        .{ .add = .{ .branch = .branch_out, .scale = .one } },
    };

    const block = Block{
        .program = &program,
        .block = &transformer_block,
        .block_idx = 0,
        .hidden_size = 128,
    };

    try block.validate();
}

test "Block.validate detects split with invalid num_outputs" {
    const allocator = testing.allocator;

    var weights = try TestWeights.init(allocator, 128, 512, 2, 32);
    defer weights.deinit(allocator);

    var transformer_block = try createTestTransformerBlock(allocator, &weights);
    defer transformer_block.deinit(allocator);

    // Split with 0 outputs should fail validation
    const program = [_]LayerOp{
        .{ .split = .{ .in = .norm_out, .out_start = .tmp3, .num_outputs = 0, .split_sizes = &.{}, .dim = -1 } },
    };

    const block = Block{
        .program = &program,
        .block = &transformer_block,
        .block_idx = 0,
        .hidden_size = 128,
    };

    try testing.expectError(error.TooManySplitOutputs, block.validate());
}

test "Block.validate detects split with too many outputs" {
    const allocator = testing.allocator;

    var weights = try TestWeights.init(allocator, 128, 512, 2, 32);
    defer weights.deinit(allocator);

    var transformer_block = try createTestTransformerBlock(allocator, &weights);
    defer transformer_block.deinit(allocator);

    // Split starting at tmp3 with 62 outputs (tmp3..tmp64) exceeds NUM_TMP_BUFFERS (64)
    const program = [_]LayerOp{
        .{ .split = .{ .in = .norm_out, .out_start = .tmp3, .num_outputs = 62, .split_sizes = &.{}, .dim = -1 } },
    };

    const block = Block{
        .program = &program,
        .block = &transformer_block,
        .block_idx = 0,
        .hidden_size = 128,
    };

    try testing.expectError(error.TooManySplitOutputs, block.validate());
}

test "Block.forward executes simple norm-attn-add program" {
    const allocator = testing.allocator;

    var weights = try TestWeights.init(allocator, 128, 512, 2, 32);
    defer weights.deinit(allocator);

    var transformer_block = try createTestTransformerBlock(allocator, &weights);
    defer transformer_block.deinit(allocator);

    const program = [_]LayerOp{
        .{ .kernel = .{ .id = 0, .in = .residual, .out = .norm_out, .debug_type = .norm } },
        .{ .kernel = .{ .id = 1, .in = .norm_out, .out = .branch_out, .debug_type = .multihead_attention } },
        .{ .add = .{ .branch = .branch_out, .scale = .one } },
    };

    const block = Block{
        .program = &program,
        .block = &transformer_block,
        .block_idx = 0,
        .hidden_size = 128,
    };

    // Create input tensor
    const input_data = try allocator.alloc(f32, 1 * 4 * 128);
    defer allocator.free(input_data);
    for (input_data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i % 100)) * 0.01;
    }
    const input = Tensor.view(@ptrCast(input_data.ptr), &.{ 1, 4, 128 }, .f32, null);

    // Create output tensor
    const output_data = try allocator.alloc(f32, 1 * 4 * 128);
    defer allocator.free(output_data);
    @memset(output_data, 0.0);
    var output = Tensor.view(@ptrCast(output_data.ptr), &.{ 1, 4, 128 }, .f32, null);

    // Create scratch buffer
    var scratch = try ScratchBuffer.init(allocator, 128, 512, 1);
    defer scratch.deinit();
    try scratch.ensure(4);

    // Create attention cache
    var attn_cache = AttnCache{};
    defer attn_cache.deinit(allocator);

    // Execute forward pass
    try block.forward(&input, &output, &scratch, &attn_cache, false);

    // Verify output is non-zero (computation occurred)
    var has_nonzero = false;
    for (output_data) |val| {
        if (val != 0.0) {
            has_nonzero = true;
            break;
        }
    }
    try testing.expect(has_nonzero);
}

test "Block.forward executes full norm-attn-norm-ffn-add program" {
    const allocator = testing.allocator;

    var weights = try TestWeights.init(allocator, 128, 512, 2, 32);
    defer weights.deinit(allocator);

    var transformer_block = try createTestTransformerBlock(allocator, &weights);
    defer transformer_block.deinit(allocator);

    const program = [_]LayerOp{
        .{ .kernel = .{ .id = 0, .in = .residual, .out = .norm_out, .debug_type = .norm } },
        .{ .kernel = .{ .id = 1, .in = .norm_out, .out = .branch_out, .debug_type = .multihead_attention } },
        .{ .add = .{ .branch = .branch_out, .scale = .one } },
        .{ .kernel = .{ .id = 2, .in = .residual, .out = .norm_out, .debug_type = .norm } },
        .{ .kernel = .{ .id = 3, .in = .norm_out, .out = .branch_out, .debug_type = .mlp } },
        .{ .add = .{ .branch = .branch_out, .scale = .one } },
    };

    const block = Block{
        .program = &program,
        .block = &transformer_block,
        .block_idx = 0,
        .hidden_size = 128,
    };

    // Create input tensor with small seq_len for faster test
    const input_data = try allocator.alloc(f32, 1 * 2 * 128);
    defer allocator.free(input_data);
    for (input_data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i % 50)) * 0.02;
    }
    const input = Tensor.view(@ptrCast(input_data.ptr), &.{ 1, 2, 128 }, .f32, null);

    // Create output tensor
    const output_data = try allocator.alloc(f32, 1 * 2 * 128);
    defer allocator.free(output_data);
    @memset(output_data, 0.0);
    var output = Tensor.view(@ptrCast(output_data.ptr), &.{ 1, 2, 128 }, .f32, null);

    // Create scratch buffer
    var scratch = try ScratchBuffer.init(allocator, 128, 512, 1);
    defer scratch.deinit();
    try scratch.ensure(2);

    // Create attention cache
    var attn_cache = AttnCache{};
    defer attn_cache.deinit(allocator);

    // Execute forward pass
    try block.forward(&input, &output, &scratch, &attn_cache, false);

    // Verify output is non-zero and different from input
    var has_nonzero = false;
    var differs_from_input = false;
    for (output_data, 0..) |val, i| {
        if (val != 0.0) has_nonzero = true;
        if (val != input_data[i]) differs_from_input = true;
    }
    try testing.expect(has_nonzero);
    try testing.expect(differs_from_input);
}

test "Block.forwardWithBatchedCache executes with batched cache" {
    const allocator = testing.allocator;

    var weights = try TestWeights.init(allocator, 128, 512, 2, 32);
    defer weights.deinit(allocator);

    var transformer_block = try createTestTransformerBlock(allocator, &weights);
    defer transformer_block.deinit(allocator);

    const program = [_]LayerOp{
        .{ .kernel = .{ .id = 0, .in = .residual, .out = .norm_out, .debug_type = .norm } },
        .{ .kernel = .{ .id = 1, .in = .norm_out, .out = .branch_out, .debug_type = .multihead_attention } },
        .{ .add = .{ .branch = .branch_out, .scale = .one } },
    };

    const block = Block{
        .program = &program,
        .block = &transformer_block,
        .block_idx = 0,
        .hidden_size = 128,
    };

    // Create input tensor
    const input_data = try allocator.alloc(f32, 1 * 2 * 128);
    defer allocator.free(input_data);
    for (input_data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i % 50)) * 0.02;
    }
    const input = Tensor.view(@ptrCast(input_data.ptr), &.{ 1, 2, 128 }, .f32, null);

    // Create output tensor
    const output_data = try allocator.alloc(f32, 1 * 2 * 128);
    defer allocator.free(output_data);
    @memset(output_data, 0.0);
    var output = Tensor.view(@ptrCast(output_data.ptr), &.{ 1, 2, 128 }, .f32, null);

    // Create scratch buffer
    var scratch = try ScratchBuffer.init(allocator, 128, 512, 1);
    defer scratch.deinit();
    try scratch.ensure(2);

    // Create batched KV cache
    var batched_cache = try BatchedKVCache.init(allocator, 4, 2, 32, 2048);
    defer batched_cache.deinit();

    // Execute forward pass with batched cache
    try block.forwardWithBatchedCache(&input, &output, &scratch, &batched_cache, 0, false);

    // Verify output is non-zero
    var has_nonzero = false;
    for (output_data) |val| {
        if (val != 0.0) {
            has_nonzero = true;
            break;
        }
    }
    try testing.expect(has_nonzero);
}

test "Block.forwardWithBatchedCache handles mul_scalar" {
    const allocator = testing.allocator;

    var weights = try TestWeights.init(allocator, 128, 512, 2, 32);
    defer weights.deinit(allocator);

    var transformer_block = try createTestTransformerBlock(allocator, &weights);
    defer transformer_block.deinit(allocator);

    const program = [_]LayerOp{
        .{ .mul_scalar = .{ .in = .residual, .out = .residual, .scalar = 0.5 } },
    };

    const block = Block{
        .program = &program,
        .block = &transformer_block,
        .block_idx = 0,
        .hidden_size = 128,
    };

    const input_data = try allocator.alloc(f32, 1 * 2 * 128);
    defer allocator.free(input_data);
    for (input_data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i % 23)) * 0.07;
    }
    const input = Tensor.view(@ptrCast(input_data.ptr), &.{ 1, 2, 128 }, .f32, null);

    const output_data = try allocator.alloc(f32, 1 * 2 * 128);
    defer allocator.free(output_data);
    @memset(output_data, 0.0);
    var output = Tensor.view(@ptrCast(output_data.ptr), &.{ 1, 2, 128 }, .f32, null);

    var scratch = try ScratchBuffer.init(allocator, 128, 512, 1);
    defer scratch.deinit();
    try scratch.ensure(2);

    var batched_cache = try BatchedKVCache.init(allocator, 4, 2, 32, 2048);
    defer batched_cache.deinit();

    try block.forwardWithBatchedCache(&input, &output, &scratch, &batched_cache, 0, false);

    for (output_data, 0..) |val, i| {
        try testing.expectApproxEqAbs(input_data[i] * 0.5, val, 1e-6);
    }
}
