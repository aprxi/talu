//! Transformer block execution.
//!
//! Executes transformer blocks using LayerOp bytecode from compute graphs.
//! Handles attention, FFN, and residual connections for each layer.

const std = @import("std");
const builtin = @import("builtin");
const layer_ops = @import("../../../../models/layer_ops.zig");
const plan_compiler = @import("../../../../models/plan/compiler.zig");
const tensor = @import("../../../../tensor.zig");
const compute = @import("../../../../compute/root.zig");
const backend_contract = @import("../../contract.zig");
const error_context = @import("../../../../error_context.zig");
const runtime_contract = @import("../../../runtime_contract/root.zig");
const cpu_linalg = compute.cpu.linalg;
const tv = compute.cpu.tensor_view;
const activation_ops = compute.cpu.activation_view;
const transpose_ops = compute.cpu.layout.transpose;
const attention_ops = cpu_linalg.sdpa;
const cpu_broadcast = compute.cpu.layout.broadcast;
const cpu_elementwise = compute.cpu.elementwise;
const cpu_reduction = compute.cpu.reduction;
const cpu_layout = compute.cpu.layout;
const cpu_masking = compute.cpu.layout.masking;
const cpu_rotary = compute.cpu.rotary;
const runtime = @import("runtime.zig");
const cpu_forward = @import("weights.zig");
const attn_kernel = @import("../kernels/attention.zig");
const log = @import("../../../../log.zig");

const Tensor = tensor.Tensor;
const Attention = attn_kernel.MultiHeadAttention;
const ScratchBuffer = runtime.ScratchBuffer;
const FFNLayer = cpu_forward.FfnLayer;

const kv_cache = @import("../kernels/kv_cache.zig");
const BatchedKVCache = kv_cache.BatchedKVCache;

const BufferId = layer_ops.BufferId;
const ResidualScale = layer_ops.ResidualScale;
const LayerOp = layer_ops.LayerOp;
const SlotContext = runtime.SlotContext;
const SharedPersistentState = runtime.SharedPersistentState;

const addIntoScaled = cpu_forward.addIntoScaled;
const copyTensor = cpu_forward.copyTensor;

/// Return the buffer that holds the final output of the block program.
/// Pre-norm programs end on `.residual`; post-norm programs may end on `.norm_out`.
fn finalOutputBuffer(compiled: *const runtime_contract.CompiledPlan) BufferId {
    const out_reg = runtime_contract.planFinalOutputRegister(&compiled.plan);
    const out_idx = runtime_contract.registerToIndex(out_reg);
    const max_buffer_idx: u16 = @intFromEnum(BufferId.tmp63);
    if (out_idx > max_buffer_idx) return .residual;
    return @enumFromInt(out_idx);
}

fn formatRmsNormLike(writer: anytype, dim: usize, eps: f32, weight_offset: f32) !void {
    if (weight_offset != 0.0) {
        try writer.print("RMSNorm(dim={}, eps={e}, weight_offset={d:.1})", .{ dim, eps, weight_offset });
    } else {
        try writer.print("RMSNorm(dim={}, eps={e})", .{ dim, eps });
    }
}

/// Unified transformer block using sequential operation execution.
/// The topology (norm count, attention type, etc.) is encoded in the ops slice,
/// not in struct variants. This eliminates duplicate forward() logic.
///
/// Model files (src/models/*.zig) define block_program to create the op sequence.
pub const Block = struct {
    /// Compiled execution program for this block.
    compiled_plan: runtime_contract.CompiledPlan,

    /// CPU kernel container for this layer (single source of truth).
    block: *const cpu_forward.TransformerBlock,

    /// Block index in the model (global layer index)
    block_idx: usize,

    /// Hidden size (d_model)
    hidden_size: usize,

    pub fn initWithProgram(
        allocator: std.mem.Allocator,
        block: *const cpu_forward.TransformerBlock,
        block_idx: usize,
        hidden_size: usize,
        program: []const LayerOp,
    ) !Block {
        return .{
            .compiled_plan = try plan_compiler.compileLayerProgram(allocator, program, .decode),
            .block = block,
            .block_idx = block_idx,
            .hidden_size = hidden_size,
        };
    }

    pub fn deinit(self: *Block, allocator: std.mem.Allocator) void {
        plan_compiler.deinitCompiledPlan(allocator, &self.compiled_plan);
        self.* = undefined;
    }

    fn decodeInstructionOp(self: *const Block, op_index: usize) !LayerOp {
        if (op_index >= self.compiled_plan.plan.instructions.len) return error.InvalidInstructionIndex;
        const insn = self.compiled_plan.plan.instructions[op_index];
        const param_block_id = insn.param_block_id orelse return error.MissingParamBlock;
        if (param_block_id >= self.compiled_plan.param_blocks.len) return error.MissingParamBlock;
        const param_block = self.compiled_plan.param_blocks[param_block_id];
        if (param_block.opcode != insn.opcode) return error.ParamBlockOpcodeMismatch;
        if (param_block.data.len != @sizeOf(LayerOp)) return error.InvalidParamBlockSize;
        const op_ptr: *const LayerOp = @ptrCast(@alignCast(param_block.data.ptr));
        return op_ptr.*;
    }

    fn instructionSingleWeightBindingName(self: *const Block, op_index: usize) ![]const u8 {
        return runtime_contract.instructionSingleWeightBindingName(&self.compiled_plan, op_index) catch |err| {
            const weight_ref_count = if (op_index < self.compiled_plan.plan.instructions.len)
                self.compiled_plan.plan.instructions[op_index].weights.len
            else
                @as(usize, 0);
            error_context.setContext(
                "block={d}, op={d}, expected_weight_refs=1, actual_weight_refs={d}, error={s}",
                .{ self.block_idx, op_index, weight_ref_count, @errorName(err) },
            );
            return err;
        };
    }

    const BatchedDispatchMode = enum {
        single_slot,
        slot_batch,
    };

    const BatchedAdapterFn = *const fn (
        self: *const Block,
        op_index: usize,
        op: LayerOp,
        buffer_views: *[64]Tensor,
        scratch: *ScratchBuffer,
        slot_ctx: SlotContext,
        mode: BatchedDispatchMode,
        slot_index: usize,
        slot_indices: []const usize,
    ) anyerror!void;

    const batched_required_opcodes = [_]runtime_contract.Opcode{
        .rmsnorm,
        .multihead_attention,
        .swiglu,
        .moe,
        .mamba_mixer,
        .shortconv,
        .mla_attention,
        .embedding,
        .residual_add,
        .mul_scalar,
        .add_tensor,
    };

    const batched_adapter_table: [256]?BatchedAdapterFn = blk: {
        var table: [256]?BatchedAdapterFn = [_]?BatchedAdapterFn{null} ** 256;

        table[@intFromEnum(runtime_contract.Opcode.rmsnorm)] = batchedKernelAdapter;
        table[@intFromEnum(runtime_contract.Opcode.multihead_attention)] = batchedKernelAdapter;
        table[@intFromEnum(runtime_contract.Opcode.swiglu)] = batchedKernelAdapter;
        table[@intFromEnum(runtime_contract.Opcode.moe)] = batchedKernelAdapter;
        table[@intFromEnum(runtime_contract.Opcode.mamba_mixer)] = batchedKernelAdapter;
        table[@intFromEnum(runtime_contract.Opcode.shortconv)] = batchedKernelAdapter;
        table[@intFromEnum(runtime_contract.Opcode.mla_attention)] = batchedKernelAdapter;
        table[@intFromEnum(runtime_contract.Opcode.embedding)] = batchedKernelAdapter;
        table[@intFromEnum(runtime_contract.Opcode.residual_add)] = batchedResidualAddAdapter;
        table[@intFromEnum(runtime_contract.Opcode.mul_scalar)] = batchedMulScalarAdapter;
        table[@intFromEnum(runtime_contract.Opcode.add_tensor)] = batchedAddTensorAdapter;

        break :blk table;
    };

    comptime {
        backend_contract.assertAdapterTableCoverage(
            batched_adapter_table,
            batched_required_opcodes,
            "cpu.executor.block.batched_adapter_table",
        );
    }

    fn dispatchBatchedInstruction(
        self: *const Block,
        opcode: runtime_contract.Opcode,
        op_index: usize,
        op: LayerOp,
        buffer_views: *[64]Tensor,
        scratch: *ScratchBuffer,
        slot_ctx: SlotContext,
        mode: BatchedDispatchMode,
        slot_index: usize,
        slot_indices: []const usize,
    ) !void {
        const adapter = batched_adapter_table[@intFromEnum(opcode)] orelse {
            const op_name = @tagName(op);
            error_context.setContext("block={d}, unsupported_op={s}", .{ self.block_idx, op_name });
            return error.UnsupportedOpInBatchedMode;
        };
        try adapter(
            self,
            op_index,
            op,
            buffer_views,
            scratch,
            slot_ctx,
            mode,
            slot_index,
            slot_indices,
        );
    }

    fn batchedKernelAdapter(
        self: *const Block,
        op_index: usize,
        op: LayerOp,
        buffer_views: *[64]Tensor,
        _: *ScratchBuffer,
        slot_ctx: SlotContext,
        mode: BatchedDispatchMode,
        slot_index: usize,
        slot_indices: []const usize,
    ) !void {
        const kernel_op = switch (op) {
            .kernel => |kernel| kernel,
            else => return error.InvalidInstructionPayload,
        };

        const kernel_id: usize = @intCast(kernel_op.id);
        if (kernel_id >= self.block.kernels.len) {
            error_context.setContext("block={d}, kernel_id={d}, max={d}", .{
                self.block_idx,
                kernel_op.id,
                self.block.kernels.len,
            });
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
        switch (mode) {
            .single_slot => try kernel.forwardBatched(input, output, slot_ctx, slot_index),
            .slot_batch => try kernel.forwardBatchedSlots(input, output, slot_ctx, slot_indices),
        }

        _ = op_index;
    }

    fn batchedResidualAddAdapter(
        self: *const Block,
        _: usize,
        op: LayerOp,
        buffer_views: *[64]Tensor,
        _: *ScratchBuffer,
        _: SlotContext,
        _: BatchedDispatchMode,
        _: usize,
        _: []const usize,
    ) !void {
        const add_op = switch (op) {
            .add => |add| add,
            else => return error.InvalidInstructionPayload,
        };
        addIntoScaled(
            &buffer_views[@intFromEnum(BufferId.residual)],
            &buffer_views[@intFromEnum(add_op.branch)],
            &buffer_views[@intFromEnum(BufferId.residual)],
            self.residualScaleValue(add_op.scale),
        );
    }

    fn batchedMulScalarAdapter(
        _: *const Block,
        _: usize,
        op: LayerOp,
        buffer_views: *[64]Tensor,
        scratch: *ScratchBuffer,
        _: SlotContext,
        _: BatchedDispatchMode,
        _: usize,
        _: []const usize,
    ) !void {
        const mul_scalar_op = switch (op) {
            .mul_scalar => |mul_op| mul_op,
            else => return error.InvalidInstructionPayload,
        };

        const input_tensor = buffer_views[@intFromEnum(mul_scalar_op.in)];
        const input_data = input_tensor.asSlice(f32);
        const output_len = input_tensor.numel;

        const output_slice = resolveOutputSlice(buffer_views, scratch, mul_scalar_op.out, output_len);
        cpu_elementwise.mulScalar(input_data, output_slice[0..output_len], mul_scalar_op.scalar);

        buffer_views[@intFromEnum(mul_scalar_op.out)] = tensorFromSlice(
            output_slice[0..output_len],
            input_tensor.shape,
            input_tensor.n_dims,
        );
    }

    fn batchedAddTensorAdapter(
        _: *const Block,
        _: usize,
        op: LayerOp,
        buffer_views: *[64]Tensor,
        scratch: *ScratchBuffer,
        _: SlotContext,
        _: BatchedDispatchMode,
        _: usize,
        _: []const usize,
    ) !void {
        const add_tensor_op = switch (op) {
            .add_tensor => |add_op| add_op,
            else => return error.InvalidInstructionPayload,
        };

        const left_tensor = buffer_views[@intFromEnum(add_tensor_op.in_a)];
        const right_tensor = buffer_views[@intFromEnum(add_tensor_op.in_b)];
        const output_len = @max(left_tensor.numel, right_tensor.numel);

        const output_slice = resolveOutputSlice(buffer_views, scratch, add_tensor_op.out, output_len);
        try cpu_broadcast.applyElementwiseBinaryOp(left_tensor, right_tensor, output_slice, struct {
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
        buffer_views[@intFromEnum(add_tensor_op.out)] = tensorFromSlice(
            output_slice[0..output_len],
            output_shape,
            output_dims,
        );
    }

    const SequentialAdapterFn = *const fn (
        self: *const Block,
        op_index: usize,
        op: LayerOp,
        buffer_views: *[64]Tensor,
        scratch: *ScratchBuffer,
        slot_ctx: SlotContext,
    ) anyerror!void;

    const sequential_required_opcodes = [_]runtime_contract.Opcode{
        .rmsnorm,
        .multihead_attention,
        .swiglu,
        .moe,
        .mamba_mixer,
        .shortconv,
        .mla_attention,
        .embedding,
        .residual_add,
        .linear,
        .matmul,
        .split,
        .softmax,
        .silu,
        .gelu,
        .mul,
        .add_tensor,
        .add_scalar,
        .mul_scalar,
        .mean,
        .pow,
        .rsqrt,
        .add_param,
        .add_param_scalar,
        .mul_param,
        .reshape,
        .transpose,
        .rope,
        .triu,
        .scaled_dot_product_attention,
    };

    const sequential_adapter_table: [256]?SequentialAdapterFn = blk: {
        var table: [256]?SequentialAdapterFn = [_]?SequentialAdapterFn{null} ** 256;

        table[@intFromEnum(runtime_contract.Opcode.rmsnorm)] = sequentialKernelAdapter;
        table[@intFromEnum(runtime_contract.Opcode.multihead_attention)] = sequentialKernelAdapter;
        table[@intFromEnum(runtime_contract.Opcode.swiglu)] = sequentialKernelAdapter;
        table[@intFromEnum(runtime_contract.Opcode.moe)] = sequentialKernelAdapter;
        table[@intFromEnum(runtime_contract.Opcode.mamba_mixer)] = sequentialKernelAdapter;
        table[@intFromEnum(runtime_contract.Opcode.shortconv)] = sequentialKernelAdapter;
        table[@intFromEnum(runtime_contract.Opcode.mla_attention)] = sequentialKernelAdapter;
        table[@intFromEnum(runtime_contract.Opcode.embedding)] = sequentialKernelAdapter;
        table[@intFromEnum(runtime_contract.Opcode.residual_add)] = sequentialResidualAddAdapter;
        table[@intFromEnum(runtime_contract.Opcode.linear)] = sequentialPrimitiveAdapter;
        table[@intFromEnum(runtime_contract.Opcode.matmul)] = sequentialPrimitiveAdapter;
        table[@intFromEnum(runtime_contract.Opcode.split)] = sequentialPrimitiveAdapter;
        table[@intFromEnum(runtime_contract.Opcode.softmax)] = sequentialPrimitiveAdapter;
        table[@intFromEnum(runtime_contract.Opcode.silu)] = sequentialPrimitiveAdapter;
        table[@intFromEnum(runtime_contract.Opcode.gelu)] = sequentialPrimitiveAdapter;
        table[@intFromEnum(runtime_contract.Opcode.mul)] = sequentialPrimitiveAdapter;
        table[@intFromEnum(runtime_contract.Opcode.add_tensor)] = sequentialAddTensorAdapter;
        table[@intFromEnum(runtime_contract.Opcode.add_scalar)] = sequentialPrimitiveAdapter;
        table[@intFromEnum(runtime_contract.Opcode.mul_scalar)] = sequentialMulScalarAdapter;
        table[@intFromEnum(runtime_contract.Opcode.mean)] = sequentialPrimitiveAdapter;
        table[@intFromEnum(runtime_contract.Opcode.pow)] = sequentialPrimitiveAdapter;
        table[@intFromEnum(runtime_contract.Opcode.rsqrt)] = sequentialPrimitiveAdapter;
        table[@intFromEnum(runtime_contract.Opcode.add_param)] = sequentialPrimitiveAdapter;
        table[@intFromEnum(runtime_contract.Opcode.add_param_scalar)] = sequentialPrimitiveAdapter;
        table[@intFromEnum(runtime_contract.Opcode.mul_param)] = sequentialPrimitiveAdapter;
        table[@intFromEnum(runtime_contract.Opcode.reshape)] = sequentialPrimitiveAdapter;
        table[@intFromEnum(runtime_contract.Opcode.transpose)] = sequentialPrimitiveAdapter;
        table[@intFromEnum(runtime_contract.Opcode.rope)] = sequentialPrimitiveAdapter;
        table[@intFromEnum(runtime_contract.Opcode.triu)] = sequentialPrimitiveAdapter;
        table[@intFromEnum(runtime_contract.Opcode.scaled_dot_product_attention)] = sequentialPrimitiveAdapter;

        break :blk table;
    };

    comptime {
        backend_contract.assertAdapterTableCoverage(
            sequential_adapter_table,
            sequential_required_opcodes,
            "cpu.executor.block.sequential_adapter_table",
        );
    }

    fn dispatchSequentialInstruction(
        self: *const Block,
        opcode: runtime_contract.Opcode,
        op_index: usize,
        op: LayerOp,
        buffer_views: *[64]Tensor,
        scratch: *ScratchBuffer,
        slot_ctx: SlotContext,
    ) !void {
        const adapter = sequential_adapter_table[@intFromEnum(opcode)] orelse {
            const op_name = @tagName(op);
            error_context.setContext("block={d}, op={d}, unsupported_op={s}", .{
                self.block_idx,
                op_index,
                op_name,
            });
            return error.UnsupportedOpInSequentialMode;
        };
        try adapter(self, op_index, op, buffer_views, scratch, slot_ctx);
    }

    fn sequentialKernelAdapter(
        self: *const Block,
        op_index: usize,
        op: LayerOp,
        buffer_views: *[64]Tensor,
        _: *ScratchBuffer,
        slot_ctx: SlotContext,
    ) !void {
        const kernel_op = switch (op) {
            .kernel => |kernel| kernel,
            else => return error.InvalidInstructionPayload,
        };

        const kernel_id: usize = @intCast(kernel_op.id);
        if (kernel_id >= self.block.kernels.len) {
            error_context.setContext("block={d}, kernel_id={d}, max={d}", .{
                self.block_idx,
                kernel_op.id,
                self.block.kernels.len,
            });
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
        try kernel.forward(input, output, slot_ctx);
        _ = op_index;
    }

    fn sequentialResidualAddAdapter(
        self: *const Block,
        op_index: usize,
        op: LayerOp,
        buffer_views: *[64]Tensor,
        scratch: *ScratchBuffer,
        slot_ctx: SlotContext,
    ) !void {
        try batchedResidualAddAdapter(self, op_index, op, buffer_views, scratch, slot_ctx, .single_slot, 0, &.{});
    }

    fn sequentialMulScalarAdapter(
        self: *const Block,
        op_index: usize,
        op: LayerOp,
        buffer_views: *[64]Tensor,
        scratch: *ScratchBuffer,
        slot_ctx: SlotContext,
    ) !void {
        try batchedMulScalarAdapter(self, op_index, op, buffer_views, scratch, slot_ctx, .single_slot, 0, &.{});
    }

    fn sequentialAddTensorAdapter(
        self: *const Block,
        op_index: usize,
        op: LayerOp,
        buffer_views: *[64]Tensor,
        scratch: *ScratchBuffer,
        slot_ctx: SlotContext,
    ) !void {
        try batchedAddTensorAdapter(self, op_index, op, buffer_views, scratch, slot_ctx, .single_slot, 0, &.{});
    }

    fn sequentialPrimitiveAdapter(
        self: *const Block,
        op_index: usize,
        op: LayerOp,
        buffer_views: *[64]Tensor,
        scratch: *ScratchBuffer,
        slot_ctx: SlotContext,
    ) !void {
        const seq_len: usize = @intCast(buffer_views[@intFromEnum(BufferId.residual)].shape[1]);
        const slot_state = slot_ctx.slotState();
        const use_cache = slot_ctx.use_cache;
        switch (op) {
            .linear => |linear_op| {
                const weight_name = try self.instructionSingleWeightBindingName(op_index);
                const weight = self.block.weight_registry.get(weight_name) orelse {
                    error_context.setContext("block={d}, op={d}, weight={s}", .{
                        self.block_idx,
                        op_index,
                        weight_name,
                    });
                    return error.MissingWeight;
                };

                const input_tensor = &buffer_views[@intFromEnum(linear_op.in)];
                const output_features: usize = if (weight.dtype == .f32)
                    @intCast(weight.shape[1])
                else
                    @intCast(weight.shape[0]);

                const input_view = Tensor.view2D(
                    input_tensor.data(),
                    @intCast(input_tensor.shape[1]),
                    @intCast(input_tensor.shape[2]),
                );

                const output_slice = blk: {
                    const out_idx = @intFromEnum(linear_op.out);
                    if (out_idx >= 3 and out_idx < cpu_forward.NUM_TMP_BUFFERS) {
                        break :blk scratch.tmp[out_idx][0 .. seq_len * output_features];
                    }
                    if (linear_op.out == .norm_out) {
                        break :blk scratch.tmp[1][0 .. seq_len * output_features];
                    }

                    const input_ptr = @intFromPtr(input_tensor.data().ptr);
                    const branch_ptr = @intFromPtr(scratch.tmp[2].ptr);
                    const input_aliases_branch = (input_ptr == branch_ptr);

                    const residual_ptr = @intFromPtr(buffer_views[@intFromEnum(BufferId.residual)].data().ptr);
                    const layer_tmp_buf_ptr = @intFromPtr(scratch.tmp[0].ptr);
                    const residual_uses_layer_tmp = (residual_ptr == layer_tmp_buf_ptr);

                    break :blk if (input_aliases_branch)
                        if (residual_uses_layer_tmp)
                            scratch.tmp[1][0 .. seq_len * output_features]
                        else
                            scratch.tmp[0][0 .. seq_len * output_features]
                    else
                        scratch.tmp[2][0 .. seq_len * output_features];
                };

                const out_byte_size = seq_len * output_features * @sizeOf(f32);
                var output_view = Tensor.view2D(std.mem.sliceAsBytes(output_slice), seq_len, output_features);

                const dk = cpu_linalg.matmulKernel(weight.dtype) catch |err| {
                    error_context.setContext("block={d}, op={d}, weight={s}, dtype={}", .{
                        self.block_idx,
                        op_index,
                        weight_name,
                        weight.dtype,
                    });
                    return err;
                };
                dk.func(&input_view, weight, &output_view, &scratch.matmul_scratch);

                const out_bytes = std.mem.sliceAsBytes(output_slice)[0..out_byte_size];
                buffer_views[@intFromEnum(linear_op.out)] = Tensor.view(
                    out_bytes.ptr,
                    &.{ 1, seq_len, output_features },
                    .f32,
                    null,
                );
            },
            .split => |split_op| {
                const out_start_idx = @intFromEnum(split_op.out_start);
                const max_outputs = cpu_forward.NUM_TMP_BUFFERS - out_start_idx;
                if (out_start_idx < @intFromEnum(BufferId.tmp3) or split_op.num_outputs > max_outputs) {
                    return error.TooManySplitOutputs;
                }

                const input_tensor = &buffer_views[@intFromEnum(split_op.in)];
                const input_data = input_tensor.asSlice(f32);
                const total_dim: usize = @intCast(input_tensor.shape[2]);

                const attn_ptr = self.block.getAttention();
                var actual_sizes: [3]usize = undefined;
                if (split_op.num_outputs == 3 and attn_ptr != null) {
                    const attn = attn_ptr.?;
                    actual_sizes[0] = attn.n_heads * attn.head_dim;
                    actual_sizes[1] = attn.n_kv_heads * attn.head_dim;
                    actual_sizes[2] = attn.n_kv_heads * attn.head_dim;
                } else if (split_op.num_outputs == 2) {
                    actual_sizes[0] = total_dim / 2;
                    actual_sizes[1] = total_dim / 2;
                } else {
                    for (0..split_op.num_outputs) |out_idx| {
                        actual_sizes[out_idx] = total_dim / split_op.num_outputs;
                    }
                }

                var out_slices: [3][]f32 = undefined;
                var split_idx: u8 = 0;
                while (split_idx < split_op.num_outputs) : (split_idx += 1) {
                    const split_size: usize = actual_sizes[split_idx];
                    const out_idx = @intFromEnum(split_op.out_start) + split_idx;
                    const out_elems = seq_len * split_size;
                    out_slices[split_idx] = scratch.tmp[out_idx][0..out_elems];
                }
                try cpu_layout.splitLastDimContiguous(
                    input_data,
                    seq_len,
                    total_dim,
                    actual_sizes[0..split_op.num_outputs],
                    out_slices[0..split_op.num_outputs],
                );

                split_idx = 0;
                while (split_idx < split_op.num_outputs) : (split_idx += 1) {
                    const split_size: usize = actual_sizes[split_idx];
                    const out_idx = @intFromEnum(split_op.out_start) + split_idx;
                    const out_slice = out_slices[split_idx];
                    const byte_size = out_slice.len * @sizeOf(f32);
                    const out_bytes = std.mem.sliceAsBytes(out_slice)[0..byte_size];
                    buffer_views[out_idx] = Tensor.view(
                        out_bytes.ptr,
                        &.{ 1, seq_len, split_size },
                        .f32,
                        null,
                    );
                }
            },
            .matmul => |matmul_op| {
                const left_input = &buffer_views[@intFromEnum(matmul_op.in_a)];
                const right_input = &buffer_views[@intFromEnum(matmul_op.in_b)];

                const m_dim: usize = @intCast(left_input.shape[1]);
                const n_dim: usize = @intCast(right_input.shape[1]);

                const out_size = m_dim * n_dim;
                const out_slice = scratch.tmp[0][0..out_size];
                const out_byte_size = out_size * @sizeOf(f32);
                var output_view = Tensor.view2D(std.mem.sliceAsBytes(out_slice), m_dim, n_dim);

                const a_view = Tensor.view2D(
                    left_input.data(),
                    @intCast(left_input.shape[1]),
                    @intCast(left_input.shape[2]),
                );
                const b_view = Tensor.view2D(
                    right_input.data(),
                    @intCast(right_input.shape[1]),
                    @intCast(right_input.shape[2]),
                );
                try cpu_linalg.matmulAuto(&a_view, &b_view, &output_view, &scratch.matmul_scratch);

                const out_bytes = std.mem.sliceAsBytes(out_slice)[0..out_byte_size];
                buffer_views[@intFromEnum(matmul_op.out)] = Tensor.view(
                    out_bytes.ptr,
                    &.{ 1, m_dim, n_dim },
                    .f32,
                    null,
                );
            },
            .softmax => |softmax_op| {
                const input_tensor = &buffer_views[@intFromEnum(softmax_op.in)];
                const output_tensor = &buffer_views[@intFromEnum(softmax_op.out)];
                const input_view = tv.fromTensor(Tensor, input_tensor);
                const output_view = tv.fromTensor(Tensor, output_tensor);
                activation_ops.softmax(output_view, input_view);
            },
            .silu => |silu_op| {
                const input_tensor = &buffer_views[@intFromEnum(silu_op.in)];
                const output_tensor = &buffer_views[@intFromEnum(silu_op.out)];
                const input_view = tv.fromTensor(Tensor, input_tensor);
                const output_view = tv.fromTensor(Tensor, output_tensor);
                activation_ops.silu(output_view, input_view);
            },
            .gelu => |gelu_op| {
                const input_tensor = &buffer_views[@intFromEnum(gelu_op.in)];
                const output_tensor = &buffer_views[@intFromEnum(gelu_op.out)];
                const input_view = tv.fromTensor(Tensor, input_tensor);
                const output_view = tv.fromTensor(Tensor, output_tensor);
                activation_ops.gelu(output_view, input_view);
            },
            .mul => |mul_op| {
                const left_tensor = buffer_views[@intFromEnum(mul_op.in)];
                const right_tensor = buffer_views[@intFromEnum(mul_op.other)];
                const output_len = @max(left_tensor.numel, right_tensor.numel);

                const output_slice = resolveOutputSlice(buffer_views, scratch, mul_op.out, output_len);
                try cpu_broadcast.applyElementwiseBinaryOp(left_tensor, right_tensor, output_slice, struct {
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
                buffer_views[@intFromEnum(mul_op.out)] = tensorFromSlice(
                    output_slice[0..output_len],
                    output_shape,
                    output_dims,
                );
            },
            .add_scalar => |add_scalar_op| {
                const input_tensor = buffer_views[@intFromEnum(add_scalar_op.in)];
                const input_data = input_tensor.asSlice(f32);
                const output_len = input_tensor.numel;
                const output_slice = resolveOutputSlice(buffer_views, scratch, add_scalar_op.out, output_len);
                cpu_elementwise.addScalar(input_data, output_slice[0..output_len], add_scalar_op.scalar);

                buffer_views[@intFromEnum(add_scalar_op.out)] = tensorFromSlice(
                    output_slice[0..output_len],
                    input_tensor.shape,
                    input_tensor.n_dims,
                );
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
                    const output_slice = resolveOutputSlice(buffer_views, scratch, mean_op.out, output_len);
                    try cpu_reduction.meanLastDim4D(
                        input_data,
                        mean_seq_len,
                        head_count,
                        hidden_size,
                        output_slice[0..output_len],
                    );

                    const mean_shape: [8]i64 = if (mean_op.keepdim)
                        .{
                            1,
                            @as(i64, @intCast(mean_seq_len)),
                            @as(i64, @intCast(head_count)),
                            1,
                            0,
                            0,
                            0,
                            0,
                        }
                    else
                        .{
                            1,
                            @as(i64, @intCast(mean_seq_len)),
                            @as(i64, @intCast(head_count)),
                            0,
                            0,
                            0,
                            0,
                            0,
                        };
                    const mean_dims: i32 = if (mean_op.keepdim) 4 else 3;
                    buffer_views[@intFromEnum(mean_op.out)] = tensorFromSlice(
                        output_slice[0..output_len],
                        mean_shape,
                        mean_dims,
                    );
                } else {
                    if (mean_op.dim != -1 and mean_op.dim != 2) return error.UnsupportedMeanDim;

                    const mean_seq_len_3d: usize = @intCast(input_tensor.shape[1]);
                    const hidden_size: usize = @intCast(input_tensor.shape[2]);
                    const output_len = mean_seq_len_3d;
                    const output_slice = resolveOutputSlice(buffer_views, scratch, mean_op.out, output_len);
                    try cpu_reduction.meanLastDim3D(
                        input_data,
                        mean_seq_len_3d,
                        hidden_size,
                        output_slice[0..output_len],
                    );

                    const mean_shape_3d: [8]i64 = if (mean_op.keepdim)
                        .{
                            1,
                            @as(i64, @intCast(mean_seq_len_3d)),
                            1,
                            0,
                            0,
                            0,
                            0,
                            0,
                        }
                    else
                        .{
                            1,
                            @as(i64, @intCast(mean_seq_len_3d)),
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                        };
                    const mean_dims_3d: i32 = if (mean_op.keepdim) 3 else 2;
                    buffer_views[@intFromEnum(mean_op.out)] = tensorFromSlice(
                        output_slice[0..output_len],
                        mean_shape_3d,
                        mean_dims_3d,
                    );
                }
            },
            .pow => |pow_op| {
                const input_tensor = buffer_views[@intFromEnum(pow_op.in)];
                const input_data = input_tensor.asSlice(f32);
                const output_len = input_tensor.numel;
                const output_slice = resolveOutputSlice(buffer_views, scratch, pow_op.out, output_len);
                cpu_elementwise.powScalar(input_data, output_slice[0..output_len], pow_op.exponent);

                buffer_views[@intFromEnum(pow_op.out)] = tensorFromSlice(
                    output_slice[0..output_len],
                    input_tensor.shape,
                    input_tensor.n_dims,
                );
            },
            .rsqrt => |rsqrt_op| {
                const input_tensor = buffer_views[@intFromEnum(rsqrt_op.in)];
                const input_data = input_tensor.asSlice(f32);
                const output_len = input_tensor.numel;
                const output_slice = resolveOutputSlice(buffer_views, scratch, rsqrt_op.out, output_len);
                cpu_elementwise.rsqrt(input_data, output_slice[0..output_len]);

                buffer_views[@intFromEnum(rsqrt_op.out)] = tensorFromSlice(
                    output_slice[0..output_len],
                    input_tensor.shape,
                    input_tensor.n_dims,
                );
            },
            .add_param => |add_param_op| {
                const input_tensor = buffer_views[@intFromEnum(add_param_op.in)];
                const param_name = try self.instructionSingleWeightBindingName(op_index);
                const param = self.block.weight_registry.get(param_name) orelse {
                    error_context.setContext("block={d}, op={d}, param={s}", .{
                        self.block_idx,
                        op_index,
                        param_name,
                    });
                    return error.MissingParam;
                };

                const output_len = @max(input_tensor.numel, param.numel);
                const output_slice = resolveOutputSlice(buffer_views, scratch, add_param_op.out, output_len);
                try cpu_broadcast.addParam(input_tensor, param, output_slice[0..output_len]);

                buffer_views[@intFromEnum(add_param_op.out)] = tensorFromSlice(
                    output_slice[0..output_len],
                    input_tensor.shape,
                    input_tensor.n_dims,
                );
            },
            .add_param_scalar => |add_param_scalar_op| {
                const param_name = try self.instructionSingleWeightBindingName(op_index);
                const param = self.block.weight_registry.get(param_name) orelse {
                    error_context.setContext("block={d}, op={d}, param={s}", .{
                        self.block_idx,
                        op_index,
                        param_name,
                    });
                    return error.MissingParam;
                };
                const p_len = param.numel;
                const output_slice = resolveOutputSlice(buffer_views, scratch, add_param_scalar_op.out, p_len);
                cpu_broadcast.addParamScalar(param, output_slice[0..p_len], add_param_scalar_op.scalar);

                buffer_views[@intFromEnum(add_param_scalar_op.out)] = tensorFromSlice(
                    output_slice[0..p_len],
                    param.shape,
                    param.n_dims,
                );
            },
            .mul_param => |mul_param_op| {
                const input_tensor = buffer_views[@intFromEnum(mul_param_op.in)];
                const param_name = try self.instructionSingleWeightBindingName(op_index);
                const param = self.block.weight_registry.get(param_name) orelse {
                    error_context.setContext("block={d}, op={d}, param={s}", .{
                        self.block_idx,
                        op_index,
                        param_name,
                    });
                    return error.MissingParam;
                };

                const output_len = @max(input_tensor.numel, param.numel);
                const output_slice = resolveOutputSlice(buffer_views, scratch, mul_param_op.out, output_len);
                try cpu_broadcast.mulParam(input_tensor, param, output_slice[0..output_len]);

                buffer_views[@intFromEnum(mul_param_op.out)] = tensorFromSlice(
                    output_slice[0..output_len],
                    input_tensor.shape,
                    input_tensor.n_dims,
                );
            },
            .reshape => |reshape_op| {
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
                            -2 => input_tensor.shape[0],
                            -3 => input_tensor.shape[1],
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
                const in_tensor = &buffer_views[@intFromEnum(transpose_op.in)];
                const out_len = in_tensor.numel;
                const out_slice = resolveOutputSlice(buffer_views, scratch, transpose_op.out, out_len);

                const ndim: usize = @intCast(in_tensor.n_dims);
                const dim0: usize = if (transpose_op.dim0 < 0)
                    @intCast(@as(i64, @intCast(ndim)) + transpose_op.dim0)
                else
                    @intCast(transpose_op.dim0);
                const dim1: usize = if (transpose_op.dim1 < 0)
                    @intCast(@as(i64, @intCast(ndim)) + transpose_op.dim1)
                else
                    @intCast(transpose_op.dim1);

                var in_shape_dims: [8]usize = undefined;
                for (0..ndim) |dim_idx| in_shape_dims[dim_idx] = @intCast(in_tensor.shape[dim_idx]);
                for (ndim..8) |dim_idx| in_shape_dims[dim_idx] = 0;

                var out_shape_dims: [8]usize = in_shape_dims;
                const tmp_dim = out_shape_dims[dim0];
                out_shape_dims[dim0] = out_shape_dims[dim1];
                out_shape_dims[dim1] = tmp_dim;

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
                transpose_ops.transposeDispatch(out_view, in_view, dim0, dim1);

                var out_shape_i64: [8]i64 = in_tensor.shape;
                const tmp_i64 = out_shape_i64[dim0];
                out_shape_i64[dim0] = out_shape_i64[dim1];
                out_shape_i64[dim1] = tmp_i64;
                buffer_views[@intFromEnum(transpose_op.out)] = tensorFromSlice(
                    out_slice[0..out_len],
                    out_shape_i64,
                    in_tensor.n_dims,
                );
            },
            .rope => |rope_op| {
                const in_tensor = &buffer_views[@intFromEnum(rope_op.in)];
                const input_data = in_tensor.asSlice(f32);

                const attn = self.block.getAttention() orelse {
                    error_context.setContext("block={d}, op={d}, type=mamba", .{ self.block_idx, op_index });
                    return error.RopeNotAvailableForMamba;
                };
                const rope = attn.rope orelse {
                    error_context.setContext("block={d}, op={d}", .{ self.block_idx, op_index });
                    return error.MissingRopeConfig;
                };
                const pos_offset = if (use_cache and slot_state.attn_cache != null)
                    slot_state.attn_cache.?.cache_position
                else
                    0;
                cpu_rotary.applyRopeTensorInPlace(
                    input_data,
                    @intCast(in_tensor.n_dims),
                    in_tensor.shape,
                    rope.dim,
                    pos_offset,
                    rope,
                ) catch |err| {
                    error_context.setContext("block={d}, op={d}, ndim={d}", .{
                        self.block_idx,
                        op_index,
                        in_tensor.n_dims,
                    });
                    return err;
                };

                if (rope_op.in != rope_op.out) {
                    const out_slice = resolveOutputSlice(buffer_views, scratch, rope_op.out, in_tensor.numel);
                    @memcpy(out_slice, input_data);
                    buffer_views[@intFromEnum(rope_op.out)] = tensorFromSlice(
                        out_slice[0..in_tensor.numel],
                        in_tensor.shape,
                        in_tensor.n_dims,
                    );
                }
            },
            .triu => |triu_op| {
                const in_tensor = &buffer_views[@intFromEnum(triu_op.in)];
                const out_buf = &buffer_views[@intFromEnum(triu_op.out)];
                const data = in_tensor.asSlice(f32);
                const out_data = out_buf.asSlice(f32);

                const n_dims: usize = @intCast(in_tensor.n_dims);
                const rows: usize = @intCast(in_tensor.shape[n_dims - 2]);
                const cols: usize = @intCast(in_tensor.shape[n_dims - 1]);
                cpu_masking.triu(data, out_data, rows, cols, triu_op.diagonal);
            },
            .sdpa => |sdpa_op| {
                const query_buf = &buffer_views[@intFromEnum(sdpa_op.q)];
                const key_buf = &buffer_views[@intFromEnum(sdpa_op.k)];
                const value_buf = &buffer_views[@intFromEnum(sdpa_op.v)];

                if (query_buf.n_dims != 4) {
                    error_context.setContext("block={d}, op={d}, got {d}D, need 4D", .{
                        self.block_idx,
                        op_index,
                        query_buf.n_dims,
                    });
                    return error.InvalidShape;
                }

                const batch: usize = @intCast(query_buf.shape[0]);
                const n_heads: usize = @intCast(query_buf.shape[1]);
                const seq_q: usize = @intCast(query_buf.shape[2]);
                const head_dim: usize = @intCast(query_buf.shape[3]);
                const seq_k: usize = @intCast(key_buf.shape[2]);
                const scale = sdpa_op.scale orelse 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

                const out_numel = batch * n_heads * seq_q * head_dim;
                const out_slice = resolveOutputSlice(buffer_views, scratch, sdpa_op.out, out_numel);

                const q_shape = [_]usize{ batch, n_heads, seq_q, head_dim };
                const k_shape = [_]usize{ batch, n_heads, seq_k, head_dim };
                const v_shape = [_]usize{ batch, n_heads, seq_k, head_dim };
                const out_shape = [_]usize{ batch, n_heads, seq_q, head_dim };

                const q_view = tv.TensorView.initContiguous(query_buf.data_ptr.?, &q_shape, .f32);
                const k_view = tv.TensorView.initContiguous(key_buf.data_ptr.?, &k_shape, .f32);
                const v_view = tv.TensorView.initContiguous(value_buf.data_ptr.?, &v_shape, .f32);
                const out_view = tv.TensorView.initContiguous(@ptrCast(out_slice.ptr), &out_shape, .f32);

                if (sdpa_op.is_causal) {
                    attention_ops.sdpaCausal(out_view, q_view, k_view, v_view, scale, 0, scratch.allocator) catch |err| {
                        error_context.setContext("block={d}, op={d}, causal=true", .{ self.block_idx, op_index });
                        return err;
                    };
                } else {
                    attention_ops.sdpa(out_view, q_view, k_view, v_view, null, scale, scratch.allocator) catch |err| {
                        error_context.setContext("block={d}, op={d}, causal=false", .{ self.block_idx, op_index });
                        return err;
                    };
                }

                const sdpa_shape: [8]i64 = .{
                    @as(i64, @intCast(batch)),
                    @as(i64, @intCast(n_heads)),
                    @as(i64, @intCast(seq_q)),
                    @as(i64, @intCast(head_dim)),
                    0,
                    0,
                    0,
                    0,
                };
                buffer_views[@intFromEnum(sdpa_op.out)] = tensorFromSlice(
                    out_slice[0..out_numel],
                    sdpa_shape,
                    4,
                );
            },
            else => return error.InvalidInstructionPayload,
        }
    }

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
        if (buffer_idx >= 3 and buffer_idx < cpu_forward.NUM_TMP_BUFFERS) {
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
                        try formatRmsNormLike(writer, n.dim, n.eps, n.weight_offset);
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
            else => |other_op| {
                try writer.print("{s}\n", .{@tagName(other_op)});
            },
        }
    }

    /// Forward pass - executes the operation sequence
    pub fn forward(
        self: *const Block,
        x: *const Tensor,
        out: *Tensor,
        scratch: *ScratchBuffer,
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

        // Populate shared scratch only for kernels present in this block.
        const is_mla = self.block.getMLAAttention() != null;
        const slot_state = scratch.getSlotState(self.block_idx) orelse return error.InvalidState;
        var shared_state = SharedPersistentState{
            .mla_scratch = if (is_mla) scratch.getMLAScratch() else null,
            .mamba_scratch = scratch.getMambaScratch(),
            .shortconv_scratch = scratch.getShortConvScratch(),
        };
        const ctx = SlotContext{
            .slot_state_ptr = slot_state,
            .shared_state = &shared_state,
            .scratch = scratch,
            .use_cache = use_cache,
        };

        // Execute the operation sequence
        for (self.compiled_plan.plan.instructions, 0..) |insn, op_index| {
            const op = try self.decodeInstructionOp(op_index);
            try self.dispatchSequentialInstruction(
                insn.opcode,
                op_index,
                op,
                &buffer_views,
                scratch,
                ctx,
            );
        }

        // Post-norm finalization: if the program's final output is not in the residual
        // buffer (e.g., post-norm architectures like BERT end with a norm → norm_out),
        // copy the result to residual so the caller sees it in `out`.
        const final_buf = finalOutputBuffer(&self.compiled_plan);
        if (final_buf != .residual) {
            copyTensor(&buffer_views[@intFromEnum(final_buf)], &buffer_views[@intFromEnum(BufferId.residual)]);
        }
    }

    /// Validate the program against the block's weight registry and supported ops.
    /// This is intended for load-time checks to catch invalid graphs early.
    pub fn validate(self: *const Block) !void {
        runtime_contract.validateCompiledPlan(&self.compiled_plan) catch |err| {
            error_context.setContext("block={d}, compiled_plan_validation={s}", .{
                self.block_idx,
                @errorName(err),
            });
            return err;
        };
        runtime_contract.validateExecutionPlanForBlockKind(&self.compiled_plan.plan, self.block.block_type) catch |err| {
            error_context.setContext("block={d}, block_kind_validation={s}, kind={d}", .{
                self.block_idx,
                @errorName(err),
                @intFromEnum(self.block.block_type),
            });
            return err;
        };

        for (self.compiled_plan.plan.instructions, 0..) |insn, op_index| {
            if (sequential_adapter_table[@intFromEnum(insn.opcode)] == null) {
                error_context.setContext("block={d}, op={d}, opcode={d}", .{
                    self.block_idx,
                    op_index,
                    @intFromEnum(insn.opcode),
                });
                return error.UnsupportedOpInSequentialMode;
            }
            const op = try self.decodeInstructionOp(op_index);
            switch (op) {
                .kernel => |kernel_op| {
                    const kernel_id: usize = @intCast(kernel_op.id);
                    if (kernel_id >= self.block.kernels.len) {
                        error_context.setContext("block={d}, kernel_id={d}, max={d}", .{ self.block_idx, kernel_op.id, self.block.kernels.len });
                        return error.KernelIndexOutOfBounds;
                    }
                },
                .linear => {
                    const weight_name = try self.instructionSingleWeightBindingName(op_index);
                    if (self.block.weight_registry.get(weight_name) == null) {
                        error_context.setContext("block={d}, op={d}, weight={s}", .{ self.block_idx, op_index, weight_name });
                        return error.MissingWeight;
                    }
                    const weight = self.block.weight_registry.get(weight_name).?;
                    _ = cpu_linalg.matmulKernel(weight.dtype) catch |err| {
                        error_context.setContext("block={d}, op={d}, weight={s}, dtype={}", .{ self.block_idx, op_index, weight_name, weight.dtype });
                        return err;
                    };
                },
                .add_param => {
                    const param_name = try self.instructionSingleWeightBindingName(op_index);
                    if (self.block.weight_registry.get(param_name) == null) {
                        error_context.setContext("block={d}, op={d}, param={s}", .{ self.block_idx, op_index, param_name });
                        return error.MissingParam;
                    }
                },
                .add_param_scalar => {
                    const param_name = try self.instructionSingleWeightBindingName(op_index);
                    if (self.block.weight_registry.get(param_name) == null) {
                        error_context.setContext("block={d}, op={d}, param={s}", .{ self.block_idx, op_index, param_name });
                        return error.MissingParam;
                    }
                },
                .mul_param => {
                    const param_name = try self.instructionSingleWeightBindingName(op_index);
                    if (self.block.weight_registry.get(param_name) == null) {
                        error_context.setContext("block={d}, op={d}, param={s}", .{ self.block_idx, op_index, param_name });
                        return error.MissingParam;
                    }
                },
                .split => |split_op| {
                    const out_start_idx = @intFromEnum(split_op.out_start);
                    const max_outputs = cpu_forward.NUM_TMP_BUFFERS - out_start_idx;
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
        if (self.block._mamba) |mamba_k| {
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
        const instruction_count = self.compiled_plan.plan.instructions.len;
        try writer.print("(layers.{}): Block({} ops)\n", .{ self.block_idx, instruction_count });

        for (self.compiled_plan.plan.instructions, 0..) |_, op_index| {
            const op = try self.decodeInstructionOp(op_index);
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

    /// Returns true when this block can execute decode in a single batched pass
    /// across multiple scheduler slots.
    pub fn supportsBatchedDecodeSlots(self: *const Block) bool {
        for (self.compiled_plan.plan.instructions, 0..) |_, op_index| {
            const op = self.decodeInstructionOp(op_index) catch return false;
            switch (op) {
                .kernel => |kernel_op| {
                    const kernel_id: usize = @intCast(kernel_op.id);
                    if (kernel_id >= self.block.kernels.len) return false;
                    const kernel = self.block.kernels[kernel_id];
                    switch (kernel) {
                        .attention, .swiglu, .moe, .norm => {},
                        .mla_attention, .mamba, .shortconv => return false,
                    }
                },
                .add, .mul_scalar, .add_tensor => {},
                else => return false,
            }
        }
        return true;
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

        const is_mla = self.block.getMLAAttention() != null;
        const slot_state = scratch.getSlotState(self.block_idx) orelse return error.InvalidState;
        var shared_state = SharedPersistentState{
            .batched_cache = batched_cache,
            .mla_scratch = if (is_mla) scratch.getMLAScratch() else null,
            .mamba_scratch = scratch.getMambaScratch(),
            .shortconv_scratch = scratch.getShortConvScratch(),
        };
        const ctx = SlotContext{
            .slot_state_ptr = slot_state,
            .shared_state = &shared_state,
            .scratch = scratch,
            .use_cache = use_cache,
        };

        // Execute the operation sequence
        for (self.compiled_plan.plan.instructions, 0..) |insn, op_index| {
            const op = try self.decodeInstructionOp(op_index);
            try self.dispatchBatchedInstruction(
                insn.opcode,
                op_index,
                op,
                &buffer_views,
                scratch,
                ctx,
                .single_slot,
                slot_index,
                &.{},
            );
        }

        // Post-norm finalization: if the program's final output is not in the residual
        // buffer (e.g., post-norm architectures like BERT end with a norm → norm_out),
        // copy the result to residual so the caller sees it in `out`.
        const final_buf = finalOutputBuffer(&self.compiled_plan);
        if (final_buf != .residual) {
            copyTensor(&buffer_views[@intFromEnum(final_buf)], &buffer_views[@intFromEnum(BufferId.residual)]);
        }
    }

    /// Forward pass across multiple scheduler slots in a single block execution.
    /// Input/output tensors use decode-batch layout [1, batch_size, d_model].
    pub fn forwardWithBatchedCacheSlots(
        self: *const Block,
        x: *const Tensor,
        out: *Tensor,
        scratch: *ScratchBuffer,
        batched_cache: *BatchedKVCache,
        slot_indices: []const usize,
        use_cache: bool,
    ) !void {
        std.debug.assert(x.shape[0] == 1 and out.shape[0] == 1);
        const batch_size: usize = @intCast(x.shape[1]);
        std.debug.assert(batch_size == slot_indices.len);
        try scratch.ensure(batch_size);

        const norm_output_view = Tensor.view3DSlice(scratch.tmp[1], batch_size, self.hidden_size);
        const branch_output_view = Tensor.view3DSlice(scratch.tmp[2], batch_size, self.hidden_size);

        var buffer_views: [64]Tensor = undefined;
        buffer_views[@intFromEnum(BufferId.residual)] = out.*;
        buffer_views[@intFromEnum(BufferId.norm_out)] = norm_output_view;
        buffer_views[@intFromEnum(BufferId.branch_out)] = branch_output_view;

        copyTensor(x, out);

        const is_mla = self.block.getMLAAttention() != null;
        const slot_state = scratch.getSlotState(self.block_idx) orelse return error.InvalidState;
        var shared_state = SharedPersistentState{
            .batched_cache = batched_cache,
            .mla_scratch = if (is_mla) scratch.getMLAScratch() else null,
            .mamba_scratch = scratch.getMambaScratch(),
            .shortconv_scratch = scratch.getShortConvScratch(),
        };
        const ctx = SlotContext{
            .slot_state_ptr = slot_state,
            .shared_state = &shared_state,
            .scratch = scratch,
            .use_cache = use_cache,
        };

        for (self.compiled_plan.plan.instructions, 0..) |insn, op_index| {
            const op = try self.decodeInstructionOp(op_index);
            try self.dispatchBatchedInstruction(
                insn.opcode,
                op_index,
                op,
                &buffer_views,
                scratch,
                ctx,
                .slot_batch,
                0,
                slot_indices,
            );
        }

        const final_buf = finalOutputBuffer(&self.compiled_plan);
        if (final_buf != .residual) {
            copyTensor(&buffer_views[@intFromEnum(final_buf)], &buffer_views[@intFromEnum(BufferId.residual)]);
        }
    }
};

pub const TransformerBlock = Block;

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

fn createTestBlock(
    allocator: std.mem.Allocator,
    transformer_block: *const cpu_forward.TransformerBlock,
    hidden_size: usize,
    program: []const LayerOp,
) !Block {
    return Block.initWithProgram(allocator, transformer_block, 0, hidden_size, program);
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

    var block = try createTestBlock(allocator, &transformer_block, 128, &program);
    defer block.deinit(allocator);

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

    var block = try createTestBlock(allocator, &transformer_block, 128, &program);
    defer block.deinit(allocator);
    block.block_idx = 5;

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

    var block = try createTestBlock(allocator, &transformer_block, 128, &program);
    defer block.deinit(allocator);

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

    var block = try createTestBlock(allocator, &transformer_block, 128, &program);
    defer block.deinit(allocator);

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

    var block = try createTestBlock(allocator, &transformer_block, 128, &program);
    defer block.deinit(allocator);

    try block.validate();
}

test "Block.validate resolves primitive linear weights via compiled weight bindings" {
    const allocator = testing.allocator;

    var weights = try TestWeights.init(allocator, 128, 512, 2, 32);
    defer weights.deinit(allocator);

    var transformer_block = try createTestTransformerBlock(allocator, &weights);
    defer transformer_block.deinit(allocator);

    const program = [_]LayerOp{
        .{ .linear = .{ .in = .residual, .out = .norm_out, .weight_name = "q_proj" } },
    };

    var block = try createTestBlock(allocator, &transformer_block, 128, &program);
    defer block.deinit(allocator);

    try testing.expectEqual(@as(usize, 1), block.compiled_plan.weight_bindings.len);
    const binding_name = @constCast(block.compiled_plan.weight_bindings[0].name);
    try testing.expect(binding_name.len > 0);
    binding_name[0] = 'x';

    try testing.expectError(error.MissingWeight, block.validate());
}

test "Block.validate rejects invalid compiled weight ref count for primitive opcode" {
    const allocator = testing.allocator;

    var weights = try TestWeights.init(allocator, 128, 512, 2, 32);
    defer weights.deinit(allocator);

    var transformer_block = try createTestTransformerBlock(allocator, &weights);
    defer transformer_block.deinit(allocator);

    const program = [_]LayerOp{
        .{ .linear = .{ .in = .residual, .out = .norm_out, .weight_name = "q_proj" } },
    };

    var block = try createTestBlock(allocator, &transformer_block, 128, &program);
    defer block.deinit(allocator);

    const insn_mut = @constCast(block.compiled_plan.plan.instructions.ptr);
    insn_mut[0].weights = &.{};

    try testing.expectError(error.InvalidWeightRefCount, block.validate());
}

test "Block.validate rejects stateful opcode incompatible with block kind" {
    const allocator = testing.allocator;

    var weights = try TestWeights.init(allocator, 128, 512, 2, 32);
    defer weights.deinit(allocator);

    var transformer_block = try createTestTransformerBlock(allocator, &weights);
    defer transformer_block.deinit(allocator);

    const program = [_]LayerOp{
        .{ .kernel = .{ .id = 0, .in = .residual, .out = .norm_out, .debug_type = .shortconv } },
    };

    var block = try createTestBlock(allocator, &transformer_block, 128, &program);
    defer block.deinit(allocator);

    try testing.expectError(error.InvalidStateDescriptorBinding, block.validate());
}

test "Block.validate rejects unexpected state descriptor binding on stateless opcode" {
    const allocator = testing.allocator;

    var weights = try TestWeights.init(allocator, 128, 512, 2, 32);
    defer weights.deinit(allocator);

    var transformer_block = try createTestTransformerBlock(allocator, &weights);
    defer transformer_block.deinit(allocator);

    const program = [_]LayerOp{
        .{ .add = .{ .branch = .branch_out, .scale = .one } },
    };

    var block = try createTestBlock(allocator, &transformer_block, 128, &program);
    defer block.deinit(allocator);

    const insn_mut = @constCast(block.compiled_plan.plan.instructions.ptr);
    insn_mut[0].state_block_id = @intFromEnum(runtime_contract.StateBlockId.kv_cache);

    try testing.expectError(error.InvalidStateDescriptorBinding, block.validate());
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

    var block = try createTestBlock(allocator, &transformer_block, 128, &program);
    defer block.deinit(allocator);

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

    var block = try createTestBlock(allocator, &transformer_block, 128, &program);
    defer block.deinit(allocator);

    try testing.expectError(error.TooManySplitOutputs, block.validate());
}

test "Block.validate rejects opcode without sequential adapter" {
    const allocator = testing.allocator;

    var weights = try TestWeights.init(allocator, 128, 512, 2, 32);
    defer weights.deinit(allocator);

    var transformer_block = try createTestTransformerBlock(allocator, &weights);
    defer transformer_block.deinit(allocator);

    const program = [_]LayerOp{
        .{ .patch_embed = .{ .in = .residual, .out = .tmp3 } },
    };

    var block = try createTestBlock(allocator, &transformer_block, 128, &program);
    defer block.deinit(allocator);

    try testing.expectError(error.UnsupportedOpInSequentialMode, block.validate());
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

    var block = try createTestBlock(allocator, &transformer_block, 128, &program);
    defer block.deinit(allocator);

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

    // Execute forward pass
    try block.forward(&input, &output, &scratch, false);

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

    var block = try createTestBlock(allocator, &transformer_block, 128, &program);
    defer block.deinit(allocator);

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

    // Execute forward pass
    try block.forward(&input, &output, &scratch, false);

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

test "Block.forward executes mean primitive over last dim" {
    const allocator = testing.allocator;

    var weights = try TestWeights.init(allocator, 128, 512, 2, 32);
    defer weights.deinit(allocator);

    var transformer_block = try createTestTransformerBlock(allocator, &weights);
    defer transformer_block.deinit(allocator);

    const program = [_]LayerOp{
        .{ .mean = .{ .in = .residual, .out = .residual, .dim = -1, .keepdim = false } },
    };

    var block = try createTestBlock(allocator, &transformer_block, 128, &program);
    defer block.deinit(allocator);

    const seq_len = 2;
    const hidden = 128;
    const input_data = try allocator.alloc(f32, 1 * seq_len * hidden);
    defer allocator.free(input_data);
    for (0..hidden) |i| {
        input_data[i] = @as(f32, @floatFromInt(i + 1));
        input_data[hidden + i] = @as(f32, @floatFromInt(201 + i));
    }
    const input = Tensor.view(@ptrCast(input_data.ptr), &.{ 1, seq_len, hidden }, .f32, null);

    const output_data = try allocator.alloc(f32, 1 * seq_len * hidden);
    defer allocator.free(output_data);
    @memset(output_data, 0.0);
    var output = Tensor.view(@ptrCast(output_data.ptr), &.{ 1, seq_len, hidden }, .f32, null);

    var scratch = try ScratchBuffer.init(allocator, hidden, 512, 1);
    defer scratch.deinit();
    try scratch.ensure(seq_len);

    try block.forward(&input, &output, &scratch, false);

    // Row means:
    // 1..128 => 64.5
    // 201..328 => 264.5
    try testing.expectApproxEqAbs(@as(f32, 64.5), output_data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 264.5), output_data[1], 1e-5);
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

    var block = try createTestBlock(allocator, &transformer_block, 128, &program);
    defer block.deinit(allocator);

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

    var block = try createTestBlock(allocator, &transformer_block, 128, &program);
    defer block.deinit(allocator);

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

test "finalOutputBuffer resolves vision ops output buffer" {
    const allocator = testing.allocator;
    const program = [_]LayerOp{
        .{ .patch_embed = .{ .in = .residual, .out = .tmp3 } },
        .{ .spatial_merge = .{ .in = .tmp3, .out = .tmp4, .merge_size = 2 } },
        .{ .deepstack_extract = .{ .in = .tmp4, .out = .tmp5, .layer_index = 3 } },
        .{ .scatter = .{ .text_in = .residual, .vision_in = .tmp5, .out = .branch_out, .image_token_id = 99 } },
    };
    var compiled = try plan_compiler.compileLayerProgram(allocator, &program, .vision_encode);
    defer plan_compiler.deinitCompiledPlan(allocator, &compiled);
    try testing.expectEqual(BufferId.branch_out, finalOutputBuffer(&compiled));
}
