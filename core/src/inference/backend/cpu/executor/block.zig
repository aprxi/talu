//! Transformer block execution.
//!
//! Executes transformer blocks using LayerOp bytecode from compute graphs.
//! Handles attention, FFN, and residual connections for each layer.

const std = @import("std");
const builtin = @import("builtin");
const layer_ops = @import("../../../../models/layer_ops.zig");
const op_types = @import("../../../../models/op_types.zig");
const plan_compiler = @import("../../../../models/plan/compiler.zig");
const tensor = @import("../../../../tensor.zig");
const compute = @import("../../../../compute/root.zig");
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
const vision_adapters = @import("../../../vision_program_adapters.zig");
const log = @import("../../../../log.zig");

const Tensor = tensor.Tensor;
const Attention = attn_kernel.MultiHeadAttention;
const ScratchBuffer = runtime.ScratchBuffer;
const FFNLayer = cpu_forward.FfnLayer;

const kv_cache = @import("../kernels/kv_cache.zig");
const LayeredBatchedKVCache = kv_cache.LayeredBatchedKVCache;

const BufferId = layer_ops.BufferId;
const ResidualScale = layer_ops.ResidualScale;
const LayerOp = layer_ops.LayerOp;
const SlotContext = runtime.SlotContext;
const SharedPersistentState = runtime.SharedPersistentState;
const CpuKernel = cpu_forward.CpuKernel;

const addIntoScaled = cpu_forward.addIntoScaled;
const copyTensor = cpu_forward.copyTensor;
const TMP_BUFFER_MAP_LEN: usize = 64;
const MAX_INSTRUCTION_TENSOR_HANDLES: usize = 16;
var layer_program_dispatch_counters = runtime_contract.DispatchCounters{};

const TmpRegisterLayout = struct {
    map: [TMP_BUFFER_MAP_LEN]u8,
    slot_width_hints: [TMP_BUFFER_MAP_LEN]usize,
    slot_active: [TMP_BUFFER_MAP_LEN]bool,
};

fn identityTmpRegisterMap() [TMP_BUFFER_MAP_LEN]u8 {
    var map: [TMP_BUFFER_MAP_LEN]u8 = undefined;
    for (0..TMP_BUFFER_MAP_LEN) |idx| {
        map[idx] = @intCast(idx);
    }
    return map;
}

fn blockTempWidthHint(block: *const cpu_forward.TransformerBlock, hidden_size: usize) usize {
    var hint = @max(hidden_size, 1);
    if (block.getFfnLayer()) |ffn| {
        switch (ffn.*) {
            .swiglu => |s| hint = @max(hint, s.d_ff * 2),
            .moe_ffn => |m| hint = @max(hint, m.d_ff),
        }
    }
    return hint;
}

fn buildTmpRegisterScratchMap(
    allocator: std.mem.Allocator,
    compiled: *const runtime_contract.CompiledPlan,
    register_width_hint: usize,
) !TmpRegisterLayout {
    var layout = TmpRegisterLayout{
        .map = identityTmpRegisterMap(),
        .slot_width_hints = [_]usize{0} ** TMP_BUFFER_MAP_LEN,
        .slot_active = [_]bool{false} ** TMP_BUFFER_MAP_LEN,
    };
    const register_count: usize = compiled.plan.register_count;
    if (register_count <= 1) return layout;
    if (register_count > TMP_BUFFER_MAP_LEN) return error.UnsupportedModel;

    const specs = try allocator.alloc(runtime_contract.RegisterBufferSpec, register_count);
    defer allocator.free(specs);
    // Register 0 (residual) uses the model output buffer, not scratch.
    // Mark exempt via size=0 so the allocator skips it.
    specs[0] = .{ .size = 0, .@"align" = 0 };
    if (compiled.register_buffer_specs.len != register_count) return error.InvalidRegisterSpecCount;
    // Plan specs carry relative width relationships (e.g., split ratios).
    // The model-dimension floor ensures absolute sizing is at least the
    // backend's required minimum (d_model or d_ff-based).
    const model_dim_floor = @max(register_width_hint, 1);
    for (specs[1..], 1..) |*spec, idx| {
        const plan_spec = compiled.register_buffer_specs[idx];
        spec.* = .{
            .size = @max(plan_spec.size, model_dim_floor),
            .@"align" = @max(plan_spec.@"align", @as(u16, 64)),
        };
    }

    var physical_mapping = try runtime_contract.buildPhysicalMappingLinearScan(allocator, compiled, specs);
    defer runtime_contract.deinitPhysicalMapping(allocator, &physical_mapping);
    if (physical_mapping.physical_count == 0) return layout;

    const physical_to_tmp_slot = try allocator.alloc(u8, physical_mapping.physical_count);
    defer allocator.free(physical_to_tmp_slot);
    @memset(physical_to_tmp_slot, std.math.maxInt(u8));

    var next_tmp_slot: usize = 1;
    for (0..register_count) |reg_idx| {
        const physical_id_u16 = physical_mapping.register_to_physical[reg_idx];
        if (physical_id_u16 == std.math.maxInt(u16)) continue;
        const physical_id: usize = physical_id_u16;
        if (physical_id >= physical_to_tmp_slot.len) return error.InvalidState;
        if (physical_to_tmp_slot[physical_id] == std.math.maxInt(u8)) {
            if (next_tmp_slot >= cpu_forward.NUM_TMP_BUFFERS) return error.TooManySplitOutputs;
            physical_to_tmp_slot[physical_id] = @intCast(next_tmp_slot);
            next_tmp_slot += 1;
        }
        const mapped_slot = physical_to_tmp_slot[physical_id];
        layout.map[reg_idx] = mapped_slot;
        layout.slot_active[mapped_slot] = true;
        layout.slot_width_hints[mapped_slot] = @max(
            layout.slot_width_hints[mapped_slot],
            physical_mapping.physical_specs[physical_id].size,
        );
    }

    return layout;
}

/// Return the buffer that holds the final output of the block program.
/// Pre-norm programs end on `.residual`; post-norm programs may end on `.norm_out`.
fn finalOutputBuffer(compiled: *const runtime_contract.CompiledPlan) BufferId {
    const out_reg = runtime_contract.planFinalOutputRegister(&compiled.plan);
    const out_idx = runtime_contract.registerToIndex(out_reg);
    if (out_idx >= compiled.register_to_buffer_id.len) return .residual;
    return @enumFromInt(compiled.register_to_buffer_id[out_idx]);
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
    /// Per-instruction resolved weight pointer for opcodes with single-weight arity.
    instruction_weight_refs: []?*const Tensor,
    /// Per-instruction resolved kernel pointer for macro `kernel` opcodes.
    instruction_kernel_refs: []?CpuKernel,
    /// Logical register index -> physical scratch tmp slot.
    /// Index 0 (residual) always remains identity (maps to output buffer, not scratch).
    /// Indices 1+ are mapped through liveness analysis.
    tmp_register_to_scratch_idx: [TMP_BUFFER_MAP_LEN]u8 = identityTmpRegisterMap(),
    /// Physical slot width hints derived from compiled-plan liveness allocation.
    tmp_slot_width_hints: [TMP_BUFFER_MAP_LEN]usize = [_]usize{0} ** TMP_BUFFER_MAP_LEN,
    /// Physical scratch slot activity mask.
    tmp_slot_active: [TMP_BUFFER_MAP_LEN]bool = [_]bool{false} ** TMP_BUFFER_MAP_LEN,

    pub fn initWithProgram(
        allocator: std.mem.Allocator,
        block: *const cpu_forward.TransformerBlock,
        block_idx: usize,
        hidden_size: usize,
        program: []const LayerOp,
        mode: runtime_contract.ExecutionMode,
    ) !Block {
        var compiled_plan = try plan_compiler.compileLayerProgram(allocator, program, mode);
        errdefer plan_compiler.deinitCompiledPlan(allocator, &compiled_plan);
        const width_hint = blockTempWidthHint(block, hidden_size);
        var tmp_layout = try buildTmpRegisterScratchMap(allocator, &compiled_plan, width_hint);
        for (&tmp_layout.slot_active, &tmp_layout.slot_width_hints) |active, *slot_width| {
            if (active and slot_width.* < width_hint) slot_width.* = width_hint;
        }
        const weight_refs = try buildInstructionWeightRefs(allocator, block, block_idx, &compiled_plan);
        const kernel_refs = try buildInstructionKernelRefs(allocator, block, block_idx, &compiled_plan);
        errdefer {
            allocator.free(kernel_refs);
            allocator.free(weight_refs);
        }
        return .{
            .compiled_plan = compiled_plan,
            .block = block,
            .block_idx = block_idx,
            .hidden_size = hidden_size,
            .instruction_weight_refs = weight_refs,
            .instruction_kernel_refs = kernel_refs,
            .tmp_register_to_scratch_idx = tmp_layout.map,
            .tmp_slot_width_hints = tmp_layout.slot_width_hints,
            .tmp_slot_active = tmp_layout.slot_active,
        };
    }

    pub fn deinit(self: *Block, allocator: std.mem.Allocator) void {
        allocator.free(self.instruction_kernel_refs);
        allocator.free(self.instruction_weight_refs);
        plan_compiler.deinitCompiledPlan(allocator, &self.compiled_plan);
        self.* = undefined;
    }

    fn buildInstructionWeightRefs(
        allocator: std.mem.Allocator,
        block: *const cpu_forward.TransformerBlock,
        block_idx: usize,
        compiled_plan: *const runtime_contract.CompiledPlan,
    ) ![]?*const Tensor {
        const refs = try allocator.alloc(?*const Tensor, compiled_plan.plan.instructions.len);
        errdefer allocator.free(refs);
        @memset(refs, null);

        for (compiled_plan.plan.instructions, 0..) |insn, op_index| {
            if (insn.weights.len == 0) continue;
            switch (insn.opcode) {
                .rmsnorm,
                .multihead_attention,
                .swiglu,
                .moe,
                .mamba_mixer,
                .shortconv,
                .mla_attention,
                .embedding,
                => {
                    // Macro-op kernel bindings are resolved through instruction_kernel_refs.
                    // They intentionally do not map through tensor weight_registry.
                    continue;
                },
                else => {},
            }
            if (insn.weights.len != 1) return error.InvalidWeightRefCount;
            const weight_name = runtime_contract.instructionSingleWeightBindingName(compiled_plan, op_index) catch |err| {
                error_context.setContext("block={d}, op={d}, bind_error={s}", .{
                    block_idx,
                    op_index,
                    @errorName(err),
                });
                return err;
            };
            const weight = block.weight_registry.get(weight_name) orelse {
                error_context.setContext("block={d}, op={d}, weight={s}", .{
                    block_idx,
                    op_index,
                    weight_name,
                });
                return error.MissingWeight;
            };
            refs[op_index] = weight;
        }
        return refs;
    }

    fn instructionKernelIdFromWeightBindings(
        compiled_plan: *const runtime_contract.CompiledPlan,
        op_index: usize,
        insn: *const runtime_contract.Instruction,
    ) !u32 {
        const expected_slots = runtime_contract.expectedKernelWeightSlots(insn.opcode);
        if (expected_slots.len == 0) return error.InvalidInstructionPayload;
        if (insn.weights.len != expected_slots.len) return error.InvalidWeightRefCount;

        var kernel_id: ?u32 = null;
        for (insn.weights, 0..) |_, slot_idx| {
            const binding_name = try runtime_contract.instructionWeightBindingName(compiled_plan, op_index, slot_idx);
            const parsed = try runtime_contract.parseKernelWeightBindingName(binding_name);
            if (!std.mem.eql(u8, parsed.slot_name, expected_slots[slot_idx])) return error.InvalidWeightBindingName;
            if (kernel_id == null) {
                kernel_id = parsed.kernel_id;
            } else if (kernel_id.? != parsed.kernel_id) {
                return error.InvalidWeightBindingName;
            }
        }
        return kernel_id orelse error.InvalidInstructionPayload;
    }

    fn buildInstructionKernelRefs(
        allocator: std.mem.Allocator,
        block: *const cpu_forward.TransformerBlock,
        block_idx: usize,
        compiled_plan: *const runtime_contract.CompiledPlan,
    ) ![]?CpuKernel {
        const refs = try allocator.alloc(?CpuKernel, compiled_plan.plan.instructions.len);
        errdefer allocator.free(refs);
        @memset(refs, null);

        for (compiled_plan.plan.instructions, 0..) |insn, op_index| {
            switch (insn.opcode) {
                .rmsnorm,
                .multihead_attention,
                .swiglu,
                .moe,
                .mamba_mixer,
                .shortconv,
                .mla_attention,
                .embedding,
                => {},
                else => continue,
            }
            const kernel_id = try instructionKernelIdFromWeightBindings(compiled_plan, op_index, &insn);
            const kernel_idx: usize = @intCast(kernel_id);
            if (kernel_idx >= block.kernels.len) {
                error_context.setContext("block={d}, op={d}, kernel_id={d}, max={d}", .{
                    block_idx,
                    op_index,
                    kernel_id,
                    block.kernels.len,
                });
                return error.KernelIndexOutOfBounds;
            }
            refs[op_index] = block.kernels[kernel_idx];
        }
        return refs;
    }

    fn tensorViewDescFromTensor(value: *const Tensor) runtime_contract.TensorViewDesc {
        var shape: [4]u32 = .{ 0, 0, 0, 0 };
        var strides: [4]u32 = .{ 0, 0, 0, 0 };
        const rank: usize = @intCast(@max(value.n_dims, 0));
        const capped_rank = @min(rank, shape.len);
        var dim: usize = 0;
        while (dim < capped_rank) : (dim += 1) {
            shape[dim] = @intCast(value.shape[dim]);
        }
        if (capped_rank != 0) {
            var stride_acc: usize = 1;
            var rev: usize = capped_rank;
            while (rev > 0) {
                rev -= 1;
                strides[rev] = @intCast(stride_acc);
                stride_acc *= @as(usize, @intCast(@max(shape[rev], 1)));
            }
        }
        return .{
            .dtype = value.dtype,
            .rank = @intCast(capped_rank),
            .shape = shape,
            .stride_elems = strides,
            .layout = .contiguous,
        };
    }

    const BuiltInstructionHandles = struct {
        registers: []runtime_contract.TensorHandle,
        views: []const runtime_contract.TensorViewDesc,
    };

    fn buildInstructionHandles(
        self: *const Block,
        insn: *const runtime_contract.Instruction,
        dispatch_state: *RuntimeDispatchState,
        handle_storage: *[MAX_INSTRUCTION_TENSOR_HANDLES]runtime_contract.TensorHandle,
        view_storage: *[MAX_INSTRUCTION_TENSOR_HANDLES]runtime_contract.TensorViewDesc,
    ) !BuiltInstructionHandles {
        var handle_count: usize = 0;
        var view_count: usize = 0;

        for (insn.inputs) |reg| {
            if (handle_count >= handle_storage.len) return error.InvalidInstructionBinding;
            const reg_idx = runtime_contract.registerToIndex(reg);
            if (reg_idx >= dispatch_state.buffer_views.len) return error.InvalidInstructionBinding;
            const value = &dispatch_state.buffer_views[reg_idx];
            handle_storage[handle_count] = .{
                .register = reg,
                .ptr = @ptrCast(value),
            };
            view_storage[view_count] = tensorViewDescFromTensor(value);
            handle_count += 1;
            view_count += 1;
        }
        for (insn.outputs) |reg| {
            if (handle_count >= handle_storage.len) return error.InvalidInstructionBinding;
            const reg_idx = runtime_contract.registerToIndex(reg);
            if (reg_idx >= dispatch_state.buffer_views.len) return error.InvalidInstructionBinding;
            const value = &dispatch_state.buffer_views[reg_idx];
            handle_storage[handle_count] = .{
                .register = reg,
                .ptr = @ptrCast(value),
            };
            view_storage[view_count] = tensorViewDescFromTensor(value);
            handle_count += 1;
            view_count += 1;
        }
        if (insn.weights.len != 0) switch (insn.opcode) {
            .rmsnorm,
            .multihead_attention,
            .swiglu,
            .moe,
            .mamba_mixer,
            .shortconv,
            .mla_attention,
            .embedding,
            => {
                // Macro-op adapters resolve kernels via instruction_kernel_refs.
                // They do not consume weight tensor handles from registers.
            },
            else => {
                if (insn.weights.len != 1) return error.InvalidWeightRefCount;
                if (handle_count >= handle_storage.len) return error.InvalidInstructionBinding;
                const weight_ref = try self.instructionWeightRef(dispatch_state.op_index);
                handle_storage[handle_count] = .{
                    .register = runtime_contract.registerFromIndex(0),
                    .ptr = @ptrCast(@constCast(weight_ref)),
                };
                handle_count += 1;
            },
        };

        return .{
            .registers = handle_storage[0..handle_count],
            .views = view_storage[0..view_count],
        };
    }

    fn instructionKernelBinding(
        self: *const Block,
        op_index: usize,
        opcode: runtime_contract.Opcode,
    ) !CpuKernel {
        if (op_index >= self.instruction_kernel_refs.len) return error.InvalidInstructionIndex;
        const kernel = self.instruction_kernel_refs[op_index] orelse return error.MissingKernelBinding;
        if (builtin.mode == .Debug) {
            const expected_type: op_types.OpType = switch (opcode) {
                .rmsnorm => .norm,
                .multihead_attention, .mla_attention => .multihead_attention,
                .swiglu => .mlp,
                .moe => .moe,
                .mamba_mixer => .mamba_mixer,
                .shortconv => .shortconv,
                .embedding => .embedding,
                else => return error.InvalidInstructionBinding,
            };
            if (kernel.getOpType() != expected_type) return error.InvalidInstructionBinding;
        }
        return kernel;
    }

    fn kernelWeightRefForSlot(kernel: CpuKernel, opcode: runtime_contract.Opcode, slot_name: []const u8) !*const Tensor {
        return switch (opcode) {
            .rmsnorm => switch (kernel) {
                .norm => |norm_inst| switch (norm_inst.*) {
                    .rms => |rms| if (std.mem.eql(u8, slot_name, "norm_weight")) rms.weight else error.InvalidWeightBindingName,
                    .layer => |layer_norm| if (std.mem.eql(u8, slot_name, "norm_weight")) layer_norm.weight else error.InvalidWeightBindingName,
                },
                else => error.InvalidInstructionBinding,
            },
            .multihead_attention => switch (kernel) {
                .attention => |attn_inst| {
                    if (std.mem.eql(u8, slot_name, "q_proj")) {
                        if (attn_inst.q_proj) |weight| return weight;
                        if (attn_inst.fused_qkv) |*fused| return fused;
                    } else if (std.mem.eql(u8, slot_name, "k_proj")) {
                        if (attn_inst.k_proj) |weight| return weight;
                        if (attn_inst.fused_qkv) |*fused| return fused;
                    } else if (std.mem.eql(u8, slot_name, "v_proj")) {
                        if (attn_inst.v_proj) |weight| return weight;
                        if (attn_inst.fused_qkv) |*fused| return fused;
                    } else if (std.mem.eql(u8, slot_name, "o_proj")) {
                        return attn_inst.o_proj;
                    }
                    return error.MissingWeight;
                },
                else => error.InvalidInstructionBinding,
            },
            .swiglu => switch (kernel) {
                .swiglu => |ffn_inst| {
                    if (std.mem.eql(u8, slot_name, "w1")) {
                        if (ffn_inst.w1) |weight| return weight;
                        if (ffn_inst.fused_gate_up) |*fused| return fused;
                    } else if (std.mem.eql(u8, slot_name, "w3")) {
                        if (ffn_inst.w3) |weight| return weight;
                        if (ffn_inst.fused_gate_up) |*fused| return fused;
                    } else if (std.mem.eql(u8, slot_name, "w2")) {
                        return ffn_inst.w2;
                    }
                    return error.MissingWeight;
                },
                else => error.InvalidInstructionBinding,
            },
            .moe => switch (kernel) {
                .moe => |moe_inst| {
                    if (!std.mem.eql(u8, slot_name, "router")) return error.InvalidWeightBindingName;
                    return &moe_inst.router_weight;
                },
                else => error.InvalidInstructionBinding,
            },
            .mamba_mixer => switch (kernel) {
                .mamba => |mamba_inst| {
                    if (std.mem.eql(u8, slot_name, "in_proj")) return mamba_inst.weights.in_proj;
                    if (std.mem.eql(u8, slot_name, "out_proj")) return mamba_inst.weights.out_proj;
                    return error.InvalidWeightBindingName;
                },
                else => error.InvalidInstructionBinding,
            },
            .shortconv => switch (kernel) {
                .shortconv => |shortconv_inst| {
                    if (std.mem.eql(u8, slot_name, "in_proj")) return shortconv_inst.weights.in_proj;
                    if (std.mem.eql(u8, slot_name, "conv_weight")) return shortconv_inst.weights.conv1d_weight;
                    if (std.mem.eql(u8, slot_name, "out_proj")) return shortconv_inst.weights.out_proj;
                    return error.InvalidWeightBindingName;
                },
                else => error.InvalidInstructionBinding,
            },
            .mla_attention => switch (kernel) {
                .mla_attention => |mla_inst| {
                    if (!std.mem.eql(u8, slot_name, "mla_weights")) return error.InvalidWeightBindingName;
                    return mla_inst.o_proj;
                },
                else => error.InvalidInstructionBinding,
            },
            else => error.InvalidInstructionBinding,
        };
    }

    fn instructionMacroWeightRef(
        self: *const Block,
        op_index: usize,
        insn: *const runtime_contract.Instruction,
        slot_idx: usize,
    ) !*const Tensor {
        const binding_name = try runtime_contract.instructionWeightBindingName(&self.compiled_plan, op_index, slot_idx);
        const parsed = try runtime_contract.parseKernelWeightBindingName(binding_name);
        const kernel = try self.instructionKernelBinding(op_index, insn.opcode);
        return kernelWeightRefForSlot(kernel, insn.opcode, parsed.slot_name);
    }

    fn instructionParams(self: *const Block, insn: *const runtime_contract.Instruction, param_storage: *[1]runtime_contract.ParamBlock) ![]const runtime_contract.ParamBlock {
        const param_id = insn.param_block_id orelse return &.{};
        if (param_id >= self.compiled_plan.param_blocks.len) return error.MissingParamBlock;
        const param_block = self.compiled_plan.param_blocks[param_id];
        param_storage[0] = param_block;
        return param_storage[0..1];
    }

    /// Compute total element count from a TensorViewDesc.
    fn viewNumel(view: runtime_contract.TensorViewDesc) usize {
        var n: usize = 1;
        for (0..view.rank) |i| n *= view.shape[i];
        return n;
    }

    /// Convert TensorViewDesc shape to Tensor shape format ([8]i64).
    fn viewToTensorShape(view: runtime_contract.TensorViewDesc) [8]i64 {
        var shape: [8]i64 = .{ 0, 0, 0, 0, 0, 0, 0, 0 };
        for (0..view.rank) |i| shape[i] = @intCast(view.shape[i]);
        return shape;
    }

    fn instructionRegisterToBufferIndex(reg: runtime_contract.RegisterRef) !usize {
        const idx = runtime_contract.registerToIndex(reg);
        if (idx >= TMP_BUFFER_MAP_LEN) return error.InvalidInstructionBinding;
        return idx;
    }

    fn instructionOutputSlice(
        self: *const Block,
        buffer_views: *[64]Tensor,
        scratch: *ScratchBuffer,
        reg: runtime_contract.RegisterRef,
        len: usize,
    ) ![]f32 {
        const reg_idx = runtime_contract.registerToIndex(reg);
        if (reg_idx >= cpu_forward.NUM_TMP_BUFFERS) return error.InvalidInstructionBinding;
        return self.resolveOutputSlice(buffer_views, scratch, reg_idx, len);
    }

    fn instructionWeightRef(self: *const Block, op_index: usize) !*const Tensor {
        if (op_index >= self.compiled_plan.plan.instructions.len) return error.InvalidInstructionIndex;
        if (op_index >= self.instruction_weight_refs.len) return error.InvalidInstructionIndex;
        const insn = self.compiled_plan.plan.instructions[op_index];
        if (insn.weights.len != 1) return error.InvalidWeightRefCount;
        return self.instruction_weight_refs[op_index] orelse error.MissingWeight;
    }

    fn instructionKernelRef(
        self: *const Block,
        op_index: usize,
        kernel_id: u32,
        expected_type: anytype,
    ) !CpuKernel {
        if (op_index >= self.compiled_plan.plan.instructions.len) return error.InvalidInstructionIndex;
        if (op_index >= self.instruction_kernel_refs.len) return error.InvalidInstructionIndex;
        const kernel = self.instruction_kernel_refs[op_index] orelse return error.MissingKernelBinding;
        if (builtin.mode == .Debug) {
            const actual_type = kernel.getOpType();
            if (actual_type != expected_type) {
                log.err("inference", "Graph/Kernel ordering mismatch", .{
                    .block = self.block_idx,
                    .kernel = kernel_id,
                    .expected = @tagName(expected_type),
                    .actual = @tagName(actual_type),
                }, @src());
                @panic("Graph/Kernel type mismatch - graph compiler and block init are out of sync");
            }
        }
        return kernel;
    }

    fn planUsesOpcode(self: *const Block, opcode: runtime_contract.Opcode) bool {
        for (self.compiled_plan.plan.instructions) |insn| {
            if (insn.opcode == opcode) return true;
        }
        return false;
    }

    const BatchedDispatchMode = enum {
        single_slot,
        slot_batch,
    };

    const RuntimeDispatchState = struct {
        const MaxStateBindings = 256;
        const StateBinding = struct {
            handle: runtime_contract.StateBlockHandle,
        };

        block: *const Block,
        op_index: usize,
        buffer_views: *[64]Tensor,
        scratch: *ScratchBuffer,
        slot_ctx: SlotContext,
        mode: BatchedDispatchMode,
        slot_index: usize,
        slot_indices: []const usize,
        use_batched_dispatch: bool,
        state_bindings: [MaxStateBindings]?StateBinding = [_]?StateBinding{null} ** MaxStateBindings,
        state_binding_count: u8 = 0,

        fn bindState(self: *RuntimeDispatchState, state_block: runtime_contract.StateBlockHandle) !void {
            var idx: usize = 0;
            while (idx < self.state_binding_count) : (idx += 1) {
                const existing = self.state_bindings[idx] orelse continue;
                if (existing.handle.id == state_block.id) {
                    self.state_bindings[idx] = .{ .handle = state_block };
                    return;
                }
            }
            if (self.state_binding_count >= self.state_bindings.len) return error.InvalidStateDescriptorBinding;
            self.state_bindings[self.state_binding_count] = .{ .handle = state_block };
            self.state_binding_count += 1;
        }

        fn stateBinding(self: *const RuntimeDispatchState, state_id: u8) ?StateBinding {
            var idx: usize = 0;
            while (idx < self.state_binding_count) : (idx += 1) {
                const binding = self.state_bindings[idx] orelse continue;
                if (binding.handle.id == state_id) return binding;
            }
            return null;
        }
    };

    const InstructionStateBlocks = struct {
        handles: [1]runtime_contract.StateBlockHandle = undefined,
        len: usize = 0,

        fn slice(self: *InstructionStateBlocks) []runtime_contract.StateBlockHandle {
            return self.handles[0..self.len];
        }
    };

    fn runtimeDispatchState(ctx: *runtime_contract.ExecutionContext) !*RuntimeDispatchState {
        const raw_state = ctx.workspace.any orelse return error.InvalidDispatchState;
        return @ptrCast(@alignCast(raw_state));
    }

    fn bindDispatchStateDescriptors(dispatch_state: *RuntimeDispatchState) !void {
        dispatch_state.state_binding_count = 0;
        dispatch_state.state_bindings = [_]?RuntimeDispatchState.StateBinding{null} ** RuntimeDispatchState.MaxStateBindings;
        _ = try runtime_contract.collectBuiltinStateFlags(&dispatch_state.block.compiled_plan.plan);

        const shared_state = dispatch_state.slot_ctx.sharedState();
        const bound_state_blocks = shared_state.state_blocks;
        const kv_state_id = @intFromEnum(runtime_contract.StateBlockId.kv_cache);
        for (dispatch_state.block.compiled_plan.plan.state_descs) |state_desc| {
            if (state_desc.lifecycle != .slot_persistent) return error.InvalidStateDescriptorBinding;
            const maybe_state_block = runtime_contract.findStateBlock(bound_state_blocks, state_desc.id);
            if (maybe_state_block == null) {
                // Single-slot decode/prefill uses per-slot attn cache and does
                // not require scheduler-supplied KV descriptor bindings.
                if (dispatch_state.mode == .single_slot and state_desc.id == kv_state_id) continue;
                return error.InvalidStateDescriptorBinding;
            }
            const state_block = maybe_state_block.?;
            var normalized_state_block = state_block.*;
            if (normalized_state_block.size < @sizeOf(runtime_contract.OpaqueStateRef)) {
                return error.InvalidStateDescriptorBinding;
            }
            if (normalized_state_block.align_bytes < state_desc.align_bytes) {
                normalized_state_block.align_bytes = state_desc.align_bytes;
            }
            if (state_desc.size_bytes > 0 and normalized_state_block.size < state_desc.size_bytes) {
                return error.InvalidStateDescriptorBinding;
            }
            try dispatch_state.bindState(normalized_state_block);
        }

        if (runtime_contract.findStateDescriptor(&dispatch_state.block.compiled_plan.plan, kv_state_id) != null) {
            if (dispatch_state.mode == .single_slot and runtime_contract.findStateBlock(bound_state_blocks, kv_state_id) == null) {
                shared_state.batched_cache = null;
            } else {
                const layered_cache = runtime_contract.findStateValue(
                    *LayeredBatchedKVCache,
                    bound_state_blocks,
                    kv_state_id,
                ) orelse return error.InvalidStateDescriptorBinding;
                if (dispatch_state.block.block_idx >= layered_cache.layers.len) return error.InvalidStateDescriptorBinding;
                shared_state.batched_cache = layered_cache.getLayer(dispatch_state.block.block_idx);
            }
        } else {
            shared_state.batched_cache = null;
        }

        const mamba_state_id = @intFromEnum(runtime_contract.StateBlockId.mamba);
        if (runtime_contract.findStateDescriptor(&dispatch_state.block.compiled_plan.plan, mamba_state_id) != null) {
            shared_state.mamba_scratch = runtime_contract.findStateValue(
                *runtime.MambaScratch,
                bound_state_blocks,
                mamba_state_id,
            ) orelse return error.InvalidStateDescriptorBinding;
        } else {
            shared_state.mamba_scratch = null;
        }

        const shortconv_state_id = @intFromEnum(runtime_contract.StateBlockId.shortconv);
        if (runtime_contract.findStateDescriptor(&dispatch_state.block.compiled_plan.plan, shortconv_state_id) != null) {
            shared_state.shortconv_scratch = runtime_contract.findStateValue(
                *runtime.ShortConvScratch,
                bound_state_blocks,
                shortconv_state_id,
            ) orelse return error.InvalidStateDescriptorBinding;
        } else {
            shared_state.shortconv_scratch = null;
        }
    }

    fn requireInstructionStateBinding(
        mode: BatchedDispatchMode,
        insn: *const runtime_contract.Instruction,
        plan: *const runtime_contract.ExecutionPlan,
        state_blocks: []const runtime_contract.StateBlockHandle,
    ) !void {
        if (state_blocks.len == 0 and mode == .single_slot) {
            switch (insn.opcode) {
                .multihead_attention, .mla_attention => return,
                else => {},
            }
        }
        _ = try runtime_contract.requireInstructionStateBlockForPlan(insn, plan, state_blocks);
    }

    fn buildInstructionStateBlocks(
        insn: *const runtime_contract.Instruction,
        dispatch_state: *RuntimeDispatchState,
    ) !InstructionStateBlocks {
        var blocks = InstructionStateBlocks{};
        const state_id = insn.state_block_id orelse return blocks;
        const binding = dispatch_state.stateBinding(state_id) orelse {
            if (dispatch_state.mode == .single_slot) switch (insn.opcode) {
                .multihead_attention, .mla_attention => return blocks,
                else => {},
            };
            return error.InvalidStateDescriptorBinding;
        };
        const descriptor = runtime_contract.findStateDescriptor(&dispatch_state.block.compiled_plan.plan, state_id) orelse {
            return error.UnknownStateDescriptorId;
        };
        const state_block = binding.handle;
        if (state_block.align_bytes < descriptor.align_bytes) return error.InvalidStateDescriptorBinding;
        if (descriptor.size_bytes > 0 and state_block.size < descriptor.size_bytes) return error.InvalidStateDescriptorBinding;
        blocks.handles[0] = state_block;
        blocks.len = 1;
        return blocks;
    }

    const required_opcodes = [_]runtime_contract.Opcode{
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
        .linear,
        .matmul,
        .split,
        .softmax,
        .silu,
        .gelu,
        .mul,
        .add_scalar,
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
    } ++ vision_adapters.required_opcodes;

    const adapter_table: runtime_contract.AdapterTable = blk: {
        var table: runtime_contract.AdapterTable = [_]?runtime_contract.KernelAdapterFn{null} ** 256;

        // Unified adapters — branch on use_batched_dispatch internally
        table[@intFromEnum(runtime_contract.Opcode.rmsnorm)] = unifiedKernelRuntimeAdapter;
        table[@intFromEnum(runtime_contract.Opcode.multihead_attention)] = unifiedKernelRuntimeAdapter;
        table[@intFromEnum(runtime_contract.Opcode.swiglu)] = unifiedKernelRuntimeAdapter;
        table[@intFromEnum(runtime_contract.Opcode.moe)] = unifiedKernelRuntimeAdapter;
        table[@intFromEnum(runtime_contract.Opcode.mamba_mixer)] = unifiedKernelRuntimeAdapter;
        table[@intFromEnum(runtime_contract.Opcode.shortconv)] = unifiedKernelRuntimeAdapter;
        table[@intFromEnum(runtime_contract.Opcode.mla_attention)] = unifiedKernelRuntimeAdapter;
        table[@intFromEnum(runtime_contract.Opcode.embedding)] = unifiedKernelRuntimeAdapter;
        table[@intFromEnum(runtime_contract.Opcode.residual_add)] = residualAddRuntimeAdapter;
        table[@intFromEnum(runtime_contract.Opcode.mul_scalar)] = mulScalarRuntimeAdapter;
        table[@intFromEnum(runtime_contract.Opcode.add_tensor)] = addTensorRuntimeAdapter;

        // Sequential-only adapters
        table[@intFromEnum(runtime_contract.Opcode.linear)] = sequentialLinearRuntimeAdapter;
        table[@intFromEnum(runtime_contract.Opcode.matmul)] = sequentialMatmulRuntimeAdapter;
        table[@intFromEnum(runtime_contract.Opcode.split)] = sequentialSplitRuntimeAdapter;
        table[@intFromEnum(runtime_contract.Opcode.softmax)] = sequentialSoftmaxRuntimeAdapter;
        table[@intFromEnum(runtime_contract.Opcode.silu)] = sequentialSiluRuntimeAdapter;
        table[@intFromEnum(runtime_contract.Opcode.gelu)] = sequentialGeluRuntimeAdapter;
        table[@intFromEnum(runtime_contract.Opcode.mul)] = sequentialMulRuntimeAdapter;
        table[@intFromEnum(runtime_contract.Opcode.add_scalar)] = sequentialAddScalarRuntimeAdapter;
        table[@intFromEnum(runtime_contract.Opcode.mean)] = sequentialMeanRuntimeAdapter;
        table[@intFromEnum(runtime_contract.Opcode.pow)] = sequentialPowRuntimeAdapter;
        table[@intFromEnum(runtime_contract.Opcode.rsqrt)] = sequentialRsqrtRuntimeAdapter;
        table[@intFromEnum(runtime_contract.Opcode.add_param)] = sequentialAddParamRuntimeAdapter;
        table[@intFromEnum(runtime_contract.Opcode.add_param_scalar)] = sequentialAddParamScalarRuntimeAdapter;
        table[@intFromEnum(runtime_contract.Opcode.mul_param)] = sequentialMulParamRuntimeAdapter;
        table[@intFromEnum(runtime_contract.Opcode.reshape)] = sequentialReshapeRuntimeAdapter;
        table[@intFromEnum(runtime_contract.Opcode.transpose)] = sequentialTransposeRuntimeAdapter;
        table[@intFromEnum(runtime_contract.Opcode.rope)] = sequentialRopeRuntimeAdapter;
        table[@intFromEnum(runtime_contract.Opcode.triu)] = sequentialTriuRuntimeAdapter;
        table[@intFromEnum(runtime_contract.Opcode.scaled_dot_product_attention)] = sequentialSdpaRuntimeAdapter;

        // Vision opcodes
        for (vision_adapters.required_opcodes) |opcode| {
            table[@intFromEnum(opcode)] = vision_adapters.adapter_table[@intFromEnum(opcode)];
        }

        break :blk table;
    };

    const adapter_capabilities: runtime_contract.AdapterCapabilities = blk: {
        var caps: runtime_contract.AdapterCapabilities = [_]runtime_contract.AdapterCapability{.{
            .supports_batch = false,
            .supports_graph_emit = false,
            .max_batch_size = 1,
        }} ** 256;

        // Unified adapters support slot-batched dispatch.
        caps[@intFromEnum(runtime_contract.Opcode.rmsnorm)] = .{ .supports_batch = true, .supports_graph_emit = false, .max_batch_size = null };
        caps[@intFromEnum(runtime_contract.Opcode.multihead_attention)] = .{ .supports_batch = true, .supports_graph_emit = false, .max_batch_size = null };
        caps[@intFromEnum(runtime_contract.Opcode.swiglu)] = .{ .supports_batch = true, .supports_graph_emit = false, .max_batch_size = null };
        caps[@intFromEnum(runtime_contract.Opcode.moe)] = .{ .supports_batch = true, .supports_graph_emit = false, .max_batch_size = null };
        caps[@intFromEnum(runtime_contract.Opcode.mamba_mixer)] = .{ .supports_batch = true, .supports_graph_emit = false, .max_batch_size = null };
        caps[@intFromEnum(runtime_contract.Opcode.shortconv)] = .{ .supports_batch = true, .supports_graph_emit = false, .max_batch_size = null };
        caps[@intFromEnum(runtime_contract.Opcode.mla_attention)] = .{ .supports_batch = true, .supports_graph_emit = false, .max_batch_size = null };
        caps[@intFromEnum(runtime_contract.Opcode.embedding)] = .{ .supports_batch = true, .supports_graph_emit = false, .max_batch_size = null };
        caps[@intFromEnum(runtime_contract.Opcode.residual_add)] = .{ .supports_batch = true, .supports_graph_emit = false, .max_batch_size = null };
        caps[@intFromEnum(runtime_contract.Opcode.mul_scalar)] = .{ .supports_batch = true, .supports_graph_emit = false, .max_batch_size = null };
        caps[@intFromEnum(runtime_contract.Opcode.add_tensor)] = .{ .supports_batch = true, .supports_graph_emit = false, .max_batch_size = null };

        // Sequential-only adapters remain capped at batch size 1.
        for (vision_adapters.required_opcodes) |opcode| {
            caps[@intFromEnum(opcode)] = .{
                .supports_batch = false,
                .supports_graph_emit = false,
                .max_batch_size = 1,
            };
        }
        break :blk caps;
    };

    comptime {
        runtime_contract.assertAdapterTableCoverage(
            adapter_table,
            required_opcodes,
            "cpu.executor.block.adapter_table",
        );
    }

    fn dispatchInstructionWithState(
        self: *const Block,
        insn: *const runtime_contract.Instruction,
        dispatch_state: *RuntimeDispatchState,
    ) !void {
        const adapter = adapter_table[@intFromEnum(insn.opcode)].?;

        var active_slot: [1]usize = .{dispatch_state.slot_index};
        const no_seq_lengths: [0]u32 = .{};
        const active_slots = switch (dispatch_state.mode) {
            .single_slot => active_slot[0..],
            .slot_batch => dispatch_state.slot_indices,
        };
        var exec_ctx = runtime_contract.ExecutionContext{
            .mode = .decode,
            .active_slots = active_slots,
            .sequence_lengths = no_seq_lengths[0..],
            .batch_size = active_slots.len,
            .dispatch_counters = &layer_program_dispatch_counters,
            .workspace = .{ .any = @ptrCast(dispatch_state) },
        };
        try runtime_contract.validateBatchCapability(
            adapter_capabilities[@intFromEnum(insn.opcode)],
            exec_ctx.batch_size,
        );
        runtime_contract.recordExecutionDispatch(&exec_ctx, insn.opcode);
        var handle_storage: [MAX_INSTRUCTION_TENSOR_HANDLES]runtime_contract.TensorHandle = undefined;
        var view_storage: [MAX_INSTRUCTION_TENSOR_HANDLES]runtime_contract.TensorViewDesc = undefined;
        const built_handles = try self.buildInstructionHandles(insn, dispatch_state, &handle_storage, &view_storage);
        var param_storage: [1]runtime_contract.ParamBlock = undefined;
        const params = try self.instructionParams(insn, &param_storage);
        var state_blocks = try buildInstructionStateBlocks(insn, dispatch_state);
        try requireInstructionStateBinding(dispatch_state.mode, insn, &self.compiled_plan.plan, state_blocks.slice());
        try adapter(
            &exec_ctx,
            insn,
            built_handles.registers,
            built_handles.views,
            state_blocks.slice(),
            params,
        );
    }

    fn unifiedKernelRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        _: []runtime_contract.TensorHandle,
        _: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        _: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try runtimeDispatchState(ctx);
        try requireInstructionStateBinding(state.mode, insn, &state.block.compiled_plan.plan, state_blocks);
        if (insn.inputs.len != 1 or insn.outputs.len != 1) return error.InvalidInstructionPayload;
        const input = &state.buffer_views[runtime_contract.registerToIndex(insn.inputs[0])];
        const output = &state.buffer_views[runtime_contract.registerToIndex(insn.outputs[0])];
        const kernel = try state.block.instructionKernelBinding(state.op_index, insn.opcode);
        if (state.use_batched_dispatch) {
            try bindKernelSharedStateFromInstruction(state, insn, state_blocks);
            switch (state.mode) {
                .single_slot => try kernel.forwardBatched(input, output, state.slot_ctx, state.slot_index),
                .slot_batch => try kernel.forwardBatchedSlots(input, output, state.slot_ctx, state.slot_indices),
            }
        } else {
            try kernel.forward(input, output, state.slot_ctx);
        }
    }

    fn residualAddRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        _: []runtime_contract.TensorHandle,
        _: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        params: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try runtimeDispatchState(ctx);
        try requireInstructionStateBinding(state.mode, insn, &state.block.compiled_plan.plan, state_blocks);
        if (insn.inputs.len < 2) return error.InvalidInstructionBinding;
        const residual = &state.buffer_views[runtime_contract.registerToIndex(insn.inputs[0])];
        const branch = &state.buffer_views[runtime_contract.registerToIndex(insn.inputs[1])];
        const p = try runtime_contract.paramAs(runtime_contract.ResidualAddParam, params, .residual_add);
        const scale = state.block.residualScaleValue(switch (p.scale_tag) {
            0 => .one,
            1 => .residual_multiplier,
            2 => .{ .literal = @bitCast(p.scale_literal) },
            else => return error.InvalidParamBlockABI,
        });
        addIntoScaled(residual, branch, residual, scale);
    }

    fn mulScalarRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        _: []runtime_contract.TensorHandle,
        views: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        params: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try runtimeDispatchState(ctx);
        try requireInstructionStateBinding(state.mode, insn, &state.block.compiled_plan.plan, state_blocks);
        if (insn.inputs.len != 1 or insn.outputs.len != 1) return error.InvalidInstructionBinding;
        const input_view = views[0];
        const output_len = viewNumel(input_view);
        const input_data = state.buffer_views[runtime_contract.registerToIndex(insn.inputs[0])].asSlice(f32);
        const output_slice = try state.block.instructionOutputSlice(
            state.buffer_views,
            state.scratch,
            insn.outputs[0],
            output_len,
        );
        const p = try runtime_contract.paramAs(runtime_contract.ScalarOpParam, params, .mul_scalar);
        cpu_elementwise.mulScalar(input_data[0..output_len], output_slice[0..output_len], @bitCast(p.scalar));
        const out_idx = try instructionRegisterToBufferIndex(insn.outputs[0]);
        state.buffer_views[out_idx] = tensorFromSlice(
            output_slice[0..output_len],
            viewToTensorShape(input_view),
            @intCast(input_view.rank),
        );
    }

    fn addTensorRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        _: []runtime_contract.TensorHandle,
        views: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        _: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try runtimeDispatchState(ctx);
        try requireInstructionStateBinding(state.mode, insn, &state.block.compiled_plan.plan, state_blocks);
        if (insn.inputs.len != 2 or insn.outputs.len != 1) return error.InvalidInstructionBinding;
        const left_numel = viewNumel(views[0]);
        const right_numel = viewNumel(views[1]);
        const left_tensor = state.buffer_views[runtime_contract.registerToIndex(insn.inputs[0])];
        const right_tensor = state.buffer_views[runtime_contract.registerToIndex(insn.inputs[1])];
        const output_len = @max(left_numel, right_numel);
        const output_slice = try state.block.instructionOutputSlice(
            state.buffer_views,
            state.scratch,
            insn.outputs[0],
            output_len,
        );
        try cpu_broadcast.applyElementwiseBinaryOp(left_tensor, right_tensor, output_slice, struct {
            fn addScalar(lhs: f32, rhs: f32) f32 {
                return lhs + rhs;
            }
        }.addScalar);
        const larger_view = if (left_numel >= right_numel) views[0] else views[1];
        const out_idx = try instructionRegisterToBufferIndex(insn.outputs[0]);
        state.buffer_views[out_idx] = tensorFromSlice(
            output_slice[0..output_len],
            viewToTensorShape(larger_view),
            @intCast(larger_view.rank),
        );
    }

    fn sequentialLinearRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        _: []runtime_contract.TensorHandle,
        views: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        _: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try runtimeDispatchState(ctx);
        try requireInstructionStateBinding(state.mode, insn, &state.block.compiled_plan.plan, state_blocks);
        if (insn.inputs.len != 1 or insn.outputs.len != 1) return error.InvalidInstructionBinding;
        const input_view = views[0];
        const input_buf = &state.buffer_views[runtime_contract.registerToIndex(insn.inputs[0])];
        const weight = state.block.instructionWeightRef(state.op_index) catch |err| {
            error_context.setContext("block={d}, op={d}, weight_ref={s}", .{
                state.block.block_idx,
                state.op_index,
                @errorName(err),
            });
            return err;
        };
        const seq_len: usize = input_view.shape[1];
        const output_features: usize = if (weight.dtype == .f32)
            @intCast(weight.shape[1])
        else
            @intCast(weight.shape[0]);

        const out_raw_idx = runtime_contract.registerToIndex(insn.outputs[0]);
        if (out_raw_idx > @intFromEnum(BufferId.tmp63)) return error.InvalidInstructionBinding;
        const output_slice = blk: {
            // Registers 1+ are mapped through liveness analysis.
            if (out_raw_idx >= 1 and out_raw_idx < cpu_forward.NUM_TMP_BUFFERS) {
                const mapped_idx: usize = state.block.tmp_register_to_scratch_idx[out_raw_idx];
                std.debug.assert(mapped_idx >= 1);
                std.debug.assert(mapped_idx < cpu_forward.NUM_TMP_BUFFERS);
                break :blk state.scratch.tmp[mapped_idx][0 .. seq_len * output_features];
            }

            // Fallback for residual output: alias-safe buffer selection.
            const input_ptr = @intFromPtr(input_buf.data().ptr);
            const branch_ptr = @intFromPtr(state.scratch.tmp[2].ptr);
            const input_aliases_branch = input_ptr == branch_ptr;

            const residual_ptr = @intFromPtr(state.buffer_views[@intFromEnum(BufferId.residual)].data().ptr);
            const layer_tmp_buf_ptr = @intFromPtr(state.scratch.tmp[0].ptr);
            const residual_uses_layer_tmp = residual_ptr == layer_tmp_buf_ptr;

            break :blk if (input_aliases_branch)
                if (residual_uses_layer_tmp)
                    state.scratch.tmp[1][0 .. seq_len * output_features]
                else
                    state.scratch.tmp[0][0 .. seq_len * output_features]
            else
                state.scratch.tmp[2][0 .. seq_len * output_features];
        };

        var input_2d = Tensor.view2D(input_buf.data(), seq_len, input_view.shape[2]);
        var output_2d = Tensor.view2D(std.mem.sliceAsBytes(output_slice), seq_len, output_features);
        const dk = cpu_linalg.matmulKernel(weight.dtype) catch |err| {
            error_context.setContext("block={d}, op={d}, dtype={}", .{
                state.block.block_idx,
                state.op_index,
                weight.dtype,
            });
            return err;
        };
        dk.func(&input_2d, weight, &output_2d, &state.scratch.matmul_scratch);

        const out_idx = try instructionRegisterToBufferIndex(insn.outputs[0]);
        const out_byte_size = seq_len * output_features * @sizeOf(f32);
        const out_bytes = std.mem.sliceAsBytes(output_slice)[0..out_byte_size];
        state.buffer_views[out_idx] = Tensor.view(
            out_bytes.ptr,
            &.{ 1, seq_len, output_features },
            .f32,
            null,
        );
    }

    fn sequentialMatmulRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        _: []runtime_contract.TensorHandle,
        views: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        _: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try runtimeDispatchState(ctx);
        try requireInstructionStateBinding(state.mode, insn, &state.block.compiled_plan.plan, state_blocks);
        if (insn.inputs.len != 2 or insn.outputs.len != 1) return error.InvalidInstructionBinding;
        const left_view = views[0];
        const right_view = views[1];
        const left_buf = &state.buffer_views[runtime_contract.registerToIndex(insn.inputs[0])];
        const right_buf = &state.buffer_views[runtime_contract.registerToIndex(insn.inputs[1])];

        const m_dim: usize = left_view.shape[1];
        const n_dim: usize = right_view.shape[1];
        const out_size = m_dim * n_dim;
        const out_slice = try state.block.instructionOutputSlice(
            state.buffer_views,
            state.scratch,
            insn.outputs[0],
            out_size,
        );
        var output_2d = Tensor.view2D(std.mem.sliceAsBytes(out_slice), m_dim, n_dim);
        var a_view = Tensor.view2D(left_buf.data(), m_dim, left_view.shape[2]);
        var b_view = Tensor.view2D(right_buf.data(), n_dim, right_view.shape[2]);
        try cpu_linalg.matmulAuto(&a_view, &b_view, &output_2d, &state.scratch.matmul_scratch);

        const out_idx = try instructionRegisterToBufferIndex(insn.outputs[0]);
        const out_byte_size = out_size * @sizeOf(f32);
        const out_bytes = std.mem.sliceAsBytes(out_slice)[0..out_byte_size];
        state.buffer_views[out_idx] = Tensor.view(
            out_bytes.ptr,
            &.{ 1, m_dim, n_dim },
            .f32,
            null,
        );
    }

    fn sequentialSplitRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        _: []runtime_contract.TensorHandle,
        views: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        _: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try runtimeDispatchState(ctx);
        try requireInstructionStateBinding(state.mode, insn, &state.block.compiled_plan.plan, state_blocks);
        const split_outputs = insn.outputs.len;
        if (insn.inputs.len != 1) return error.InvalidInstructionBinding;
        if (split_outputs == 0 or split_outputs > 3) return error.TooManySplitOutputs;

        const input_view = views[0];
        const input_data = state.buffer_views[runtime_contract.registerToIndex(insn.inputs[0])].asSlice(f32);
        const seq_len: usize = input_view.shape[1];
        const total_dim: usize = input_view.shape[2];

        var actual_sizes: [3]usize = undefined;
        const attn_ptr = state.block.block.getAttention();
        if (split_outputs == 3 and attn_ptr != null) {
            const attn = attn_ptr.?;
            actual_sizes[0] = attn.n_heads * attn.head_dim;
            actual_sizes[1] = attn.n_kv_heads * attn.head_dim;
            actual_sizes[2] = attn.n_kv_heads * attn.head_dim;
        } else if (split_outputs == 2) {
            actual_sizes[0] = total_dim / 2;
            actual_sizes[1] = total_dim / 2;
        } else {
            for (0..split_outputs) |out_idx| {
                actual_sizes[out_idx] = total_dim / split_outputs;
            }
        }

        var out_slices: [3][]f32 = undefined;
        for (0..split_outputs) |out_idx| {
            const split_size = actual_sizes[out_idx];
            const out_elems = seq_len * split_size;
            out_slices[out_idx] = try state.block.instructionOutputSlice(
                state.buffer_views,
                state.scratch,
                insn.outputs[out_idx],
                out_elems,
            );
        }
        try cpu_layout.splitLastDimContiguous(
            input_data,
            seq_len,
            total_dim,
            actual_sizes[0..split_outputs],
            out_slices[0..split_outputs],
        );

        for (0..split_outputs) |out_idx| {
            const split_size = actual_sizes[out_idx];
            const out_slice = out_slices[out_idx];
            const out_buf_idx = try instructionRegisterToBufferIndex(insn.outputs[out_idx]);
            const out_bytes = std.mem.sliceAsBytes(out_slice);
            state.buffer_views[out_buf_idx] = Tensor.view(
                out_bytes.ptr,
                &.{ 1, seq_len, split_size },
                .f32,
                null,
            );
        }
    }

    fn sequentialSoftmaxRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        _: []runtime_contract.TensorHandle,
        _: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        _: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try runtimeDispatchState(ctx);
        try requireInstructionStateBinding(state.mode, insn, &state.block.compiled_plan.plan, state_blocks);
        if (insn.inputs.len != 1 or insn.outputs.len != 1) return error.InvalidInstructionBinding;
        const input_tensor = &state.buffer_views[runtime_contract.registerToIndex(insn.inputs[0])];
        const output_tensor = &state.buffer_views[runtime_contract.registerToIndex(insn.outputs[0])];
        const input_tv = tv.fromTensor(Tensor, input_tensor);
        const output_tv = tv.fromTensor(Tensor, output_tensor);
        activation_ops.softmax(output_tv, input_tv);
    }

    fn sequentialSiluRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        _: []runtime_contract.TensorHandle,
        _: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        _: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try runtimeDispatchState(ctx);
        try requireInstructionStateBinding(state.mode, insn, &state.block.compiled_plan.plan, state_blocks);
        if (insn.inputs.len != 1 or insn.outputs.len != 1) return error.InvalidInstructionBinding;
        const input_tensor = &state.buffer_views[runtime_contract.registerToIndex(insn.inputs[0])];
        const output_tensor = &state.buffer_views[runtime_contract.registerToIndex(insn.outputs[0])];
        const input_tv = tv.fromTensor(Tensor, input_tensor);
        const output_tv = tv.fromTensor(Tensor, output_tensor);
        activation_ops.silu(output_tv, input_tv);
    }

    fn sequentialGeluRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        _: []runtime_contract.TensorHandle,
        _: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        _: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try runtimeDispatchState(ctx);
        try requireInstructionStateBinding(state.mode, insn, &state.block.compiled_plan.plan, state_blocks);
        if (insn.inputs.len != 1 or insn.outputs.len != 1) return error.InvalidInstructionBinding;
        const input_tensor = &state.buffer_views[runtime_contract.registerToIndex(insn.inputs[0])];
        const output_tensor = &state.buffer_views[runtime_contract.registerToIndex(insn.outputs[0])];
        const input_tv = tv.fromTensor(Tensor, input_tensor);
        const output_tv = tv.fromTensor(Tensor, output_tensor);
        activation_ops.gelu(output_tv, input_tv);
    }

    fn sequentialMulRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        _: []runtime_contract.TensorHandle,
        views: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        _: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try runtimeDispatchState(ctx);
        try requireInstructionStateBinding(state.mode, insn, &state.block.compiled_plan.plan, state_blocks);
        if (insn.inputs.len != 2 or insn.outputs.len != 1) return error.InvalidInstructionBinding;
        const left_numel = viewNumel(views[0]);
        const right_numel = viewNumel(views[1]);
        const left_tensor = state.buffer_views[runtime_contract.registerToIndex(insn.inputs[0])];
        const right_tensor = state.buffer_views[runtime_contract.registerToIndex(insn.inputs[1])];
        const output_len = @max(left_numel, right_numel);
        const output_slice = try state.block.instructionOutputSlice(
            state.buffer_views,
            state.scratch,
            insn.outputs[0],
            output_len,
        );
        try cpu_broadcast.applyElementwiseBinaryOp(left_tensor, right_tensor, output_slice, struct {
            fn multiply(a: f32, b: f32) f32 {
                return a * b;
            }
        }.multiply);
        const larger_view = if (left_numel >= right_numel) views[0] else views[1];
        const out_idx = try instructionRegisterToBufferIndex(insn.outputs[0]);
        state.buffer_views[out_idx] = tensorFromSlice(
            output_slice[0..output_len],
            viewToTensorShape(larger_view),
            @intCast(larger_view.rank),
        );
    }

    fn sequentialAddScalarRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        _: []runtime_contract.TensorHandle,
        views: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        params: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try runtimeDispatchState(ctx);
        try requireInstructionStateBinding(state.mode, insn, &state.block.compiled_plan.plan, state_blocks);
        if (insn.inputs.len != 1 or insn.outputs.len != 1) return error.InvalidInstructionBinding;
        const input_view = views[0];
        const output_len = viewNumel(input_view);
        const input_data = state.buffer_views[runtime_contract.registerToIndex(insn.inputs[0])].asSlice(f32);
        const output_slice = try state.block.instructionOutputSlice(
            state.buffer_views,
            state.scratch,
            insn.outputs[0],
            output_len,
        );
        const p = try runtime_contract.paramAs(runtime_contract.ScalarOpParam, params, .add_scalar);
        cpu_elementwise.addScalar(input_data[0..output_len], output_slice[0..output_len], @bitCast(p.scalar));
        const out_idx = try instructionRegisterToBufferIndex(insn.outputs[0]);
        state.buffer_views[out_idx] = tensorFromSlice(
            output_slice[0..output_len],
            viewToTensorShape(input_view),
            @intCast(input_view.rank),
        );
    }

    fn sequentialMeanRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        _: []runtime_contract.TensorHandle,
        views: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        params: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try runtimeDispatchState(ctx);
        try requireInstructionStateBinding(state.mode, insn, &state.block.compiled_plan.plan, state_blocks);
        const p = try runtime_contract.paramAs(runtime_contract.MeanOpParam, params, .mean);
        if (insn.inputs.len != 1 or insn.outputs.len != 1) return error.InvalidInstructionBinding;
        const input_view = views[0];
        const input_data = state.buffer_views[runtime_contract.registerToIndex(insn.inputs[0])].asSlice(f32);
        const keepdim = p.keepdim != 0;

        if (input_view.rank == 4) {
            if (p.dim != -1 and p.dim != 3) return error.UnsupportedMeanDim;
            const mean_seq_len: usize = input_view.shape[1];
            const head_count: usize = input_view.shape[2];
            const hidden_size: usize = input_view.shape[3];
            const output_len = mean_seq_len * head_count;
            const output_slice = try state.block.instructionOutputSlice(
                state.buffer_views,
                state.scratch,
                insn.outputs[0],
                output_len,
            );
            try cpu_reduction.meanLastDim4D(
                input_data,
                mean_seq_len,
                head_count,
                hidden_size,
                output_slice[0..output_len],
            );
            const mean_shape: [8]i64 = if (keepdim)
                .{ 1, @as(i64, @intCast(mean_seq_len)), @as(i64, @intCast(head_count)), 1, 0, 0, 0, 0 }
            else
                .{ 1, @as(i64, @intCast(mean_seq_len)), @as(i64, @intCast(head_count)), 0, 0, 0, 0, 0 };
            const mean_dims: i32 = if (keepdim) 4 else 3;
            const out_idx = try instructionRegisterToBufferIndex(insn.outputs[0]);
            state.buffer_views[out_idx] = tensorFromSlice(
                output_slice[0..output_len],
                mean_shape,
                mean_dims,
            );
            return;
        }

        if (p.dim != -1 and p.dim != 2) return error.UnsupportedMeanDim;
        const mean_seq_len_3d: usize = input_view.shape[1];
        const hidden_size: usize = input_view.shape[2];
        const output_len = mean_seq_len_3d;
        const output_slice = try state.block.instructionOutputSlice(
            state.buffer_views,
            state.scratch,
            insn.outputs[0],
            output_len,
        );
        try cpu_reduction.meanLastDim3D(
            input_data,
            mean_seq_len_3d,
            hidden_size,
            output_slice[0..output_len],
        );
        const mean_shape_3d: [8]i64 = if (keepdim)
            .{ 1, @as(i64, @intCast(mean_seq_len_3d)), 1, 0, 0, 0, 0, 0 }
        else
            .{ 1, @as(i64, @intCast(mean_seq_len_3d)), 0, 0, 0, 0, 0, 0 };
        const mean_dims_3d: i32 = if (keepdim) 3 else 2;
        const out_idx = try instructionRegisterToBufferIndex(insn.outputs[0]);
        state.buffer_views[out_idx] = tensorFromSlice(
            output_slice[0..output_len],
            mean_shape_3d,
            mean_dims_3d,
        );
    }

    fn sequentialPowRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        _: []runtime_contract.TensorHandle,
        views: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        params: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try runtimeDispatchState(ctx);
        try requireInstructionStateBinding(state.mode, insn, &state.block.compiled_plan.plan, state_blocks);
        if (insn.inputs.len != 1 or insn.outputs.len != 1) return error.InvalidInstructionBinding;
        const input_view = views[0];
        const output_len = viewNumel(input_view);
        const input_data = state.buffer_views[runtime_contract.registerToIndex(insn.inputs[0])].asSlice(f32);
        const output_slice = try state.block.instructionOutputSlice(
            state.buffer_views,
            state.scratch,
            insn.outputs[0],
            output_len,
        );
        const p = try runtime_contract.paramAs(runtime_contract.ScalarOpParam, params, .pow);
        cpu_elementwise.powScalar(input_data[0..output_len], output_slice[0..output_len], @bitCast(p.scalar));
        const out_idx = try instructionRegisterToBufferIndex(insn.outputs[0]);
        state.buffer_views[out_idx] = tensorFromSlice(
            output_slice[0..output_len],
            viewToTensorShape(input_view),
            @intCast(input_view.rank),
        );
    }

    fn sequentialRsqrtRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        _: []runtime_contract.TensorHandle,
        views: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        _: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try runtimeDispatchState(ctx);
        try requireInstructionStateBinding(state.mode, insn, &state.block.compiled_plan.plan, state_blocks);
        if (insn.inputs.len != 1 or insn.outputs.len != 1) return error.InvalidInstructionBinding;
        const input_view = views[0];
        const output_len = viewNumel(input_view);
        const input_data = state.buffer_views[runtime_contract.registerToIndex(insn.inputs[0])].asSlice(f32);
        const output_slice = try state.block.instructionOutputSlice(
            state.buffer_views,
            state.scratch,
            insn.outputs[0],
            output_len,
        );
        cpu_elementwise.rsqrt(input_data[0..output_len], output_slice[0..output_len]);
        const out_idx = try instructionRegisterToBufferIndex(insn.outputs[0]);
        state.buffer_views[out_idx] = tensorFromSlice(
            output_slice[0..output_len],
            viewToTensorShape(input_view),
            @intCast(input_view.rank),
        );
    }

    fn sequentialAddParamRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        _: []runtime_contract.TensorHandle,
        views: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        _: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try runtimeDispatchState(ctx);
        try requireInstructionStateBinding(state.mode, insn, &state.block.compiled_plan.plan, state_blocks);
        if (insn.inputs.len != 1 or insn.outputs.len != 1) return error.InvalidInstructionBinding;
        const input_view = views[0];
        const input_numel = viewNumel(input_view);
        const input_tensor = state.buffer_views[runtime_contract.registerToIndex(insn.inputs[0])];
        const param = state.block.instructionWeightRef(state.op_index) catch |err| {
            error_context.setContext("block={d}, op={d}, param_ref={s}", .{
                state.block.block_idx,
                state.op_index,
                @errorName(err),
            });
            return err;
        };
        const output_len = @max(input_numel, param.numel);
        const output_slice = try state.block.instructionOutputSlice(
            state.buffer_views,
            state.scratch,
            insn.outputs[0],
            output_len,
        );
        try cpu_broadcast.addParam(input_tensor, param, output_slice[0..output_len]);
        const out_idx = try instructionRegisterToBufferIndex(insn.outputs[0]);
        state.buffer_views[out_idx] = tensorFromSlice(
            output_slice[0..output_len],
            viewToTensorShape(input_view),
            @intCast(input_view.rank),
        );
    }

    fn sequentialAddParamScalarRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        _: []runtime_contract.TensorHandle,
        _: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        params: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try runtimeDispatchState(ctx);
        try requireInstructionStateBinding(state.mode, insn, &state.block.compiled_plan.plan, state_blocks);
        const param = state.block.instructionWeightRef(state.op_index) catch |err| {
            error_context.setContext("block={d}, op={d}, param_ref={s}", .{
                state.block.block_idx,
                state.op_index,
                @errorName(err),
            });
            return err;
        };
        const p_len = param.numel;
        const output_slice = try state.block.instructionOutputSlice(
            state.buffer_views,
            state.scratch,
            insn.outputs[0],
            p_len,
        );
        const p = try runtime_contract.paramAs(runtime_contract.AddParamScalarParam, params, .add_param_scalar);
        cpu_broadcast.addParamScalar(param, output_slice[0..p_len], @bitCast(p.scalar));
        const out_idx = try instructionRegisterToBufferIndex(insn.outputs[0]);
        state.buffer_views[out_idx] = tensorFromSlice(
            output_slice[0..p_len],
            param.shape,
            param.n_dims,
        );
    }

    fn sequentialMulParamRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        _: []runtime_contract.TensorHandle,
        views: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        _: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try runtimeDispatchState(ctx);
        try requireInstructionStateBinding(state.mode, insn, &state.block.compiled_plan.plan, state_blocks);
        if (insn.inputs.len != 1 or insn.outputs.len != 1) return error.InvalidInstructionBinding;
        const input_view = views[0];
        const input_numel = viewNumel(input_view);
        const input_tensor = state.buffer_views[runtime_contract.registerToIndex(insn.inputs[0])];
        const param = state.block.instructionWeightRef(state.op_index) catch |err| {
            error_context.setContext("block={d}, op={d}, param_ref={s}", .{
                state.block.block_idx,
                state.op_index,
                @errorName(err),
            });
            return err;
        };
        const output_len = @max(input_numel, param.numel);
        const output_slice = try state.block.instructionOutputSlice(
            state.buffer_views,
            state.scratch,
            insn.outputs[0],
            output_len,
        );
        try cpu_broadcast.mulParam(input_tensor, param, output_slice[0..output_len]);
        const out_idx = try instructionRegisterToBufferIndex(insn.outputs[0]);
        state.buffer_views[out_idx] = tensorFromSlice(
            output_slice[0..output_len],
            viewToTensorShape(input_view),
            @intCast(input_view.rank),
        );
    }

    fn sequentialReshapeRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        _: []runtime_contract.TensorHandle,
        views: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        params: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try runtimeDispatchState(ctx);
        try requireInstructionStateBinding(state.mode, insn, &state.block.compiled_plan.plan, state_blocks);
        const p = try runtime_contract.paramAs(runtime_contract.ReshapeOpParam, params, .reshape);
        if (insn.inputs.len != 1 or insn.outputs.len != 1) return error.InvalidInstructionBinding;
        const input_view = views[0];
        var output_tensor = state.buffer_views[runtime_contract.registerToIndex(insn.inputs[0])];

        if (p.count > 0) {
            // Decode variable-length shape from param data after fixed header.
            const data = params[0].data;
            const aligned_offset = std.mem.alignForward(usize, @sizeOf(runtime_contract.ReshapeOpParam), @alignOf(i32));
            const decode_count: usize = @min(@as(usize, p.count), 8);
            const byte_count = decode_count * @sizeOf(i32);
            if (aligned_offset + byte_count > data.len) return error.InvalidParamBlockABI;

            var out_shape: [8]i64 = .{ 0, 0, 0, 0, 0, 0, 0, 0 };
            var inferred_dim_idx: ?usize = null;
            var known_product: usize = 1;
            const total_elems = viewNumel(input_view);
            for (0..decode_count) |idx| {
                const dim = std.mem.readInt(i32, data[aligned_offset + (idx * 4) ..][0..4], .little);
                if (dim == -1) {
                    inferred_dim_idx = idx;
                    continue;
                }
                const resolved: i64 = switch (dim) {
                    -2 => @intCast(input_view.shape[0]),
                    -3 => @intCast(input_view.shape[1]),
                    else => dim,
                };
                out_shape[idx] = resolved;
                known_product *= @intCast(resolved);
            }
            if (inferred_dim_idx) |dim_idx| {
                if (known_product == 0) return error.InvalidReshape;
                out_shape[dim_idx] = @intCast(total_elems / known_product);
            }
            output_tensor.shape = out_shape;
            output_tensor.n_dims = @intCast(decode_count);
        } else if (input_view.rank == 3) {
            const reshape_seq_len: i64 = @intCast(input_view.shape[1]);
            const hidden: i64 = @intCast(input_view.shape[2]);
            const attn_info = state.block.block.getAttention() orelse return error.AttentionNotAvailable;
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
        } else if (input_view.rank == 4) {
            const reshape_seq_len_4d: i64 = @intCast(input_view.shape[1]);
            const heads: i64 = @intCast(input_view.shape[2]);
            const head_dim: i64 = @intCast(input_view.shape[3]);
            output_tensor.shape = .{ 1, reshape_seq_len_4d, heads * head_dim, 0, 0, 0, 0, 0 };
            output_tensor.n_dims = 3;
        }

        const out_idx = try instructionRegisterToBufferIndex(insn.outputs[0]);
        state.buffer_views[out_idx] = output_tensor;
    }

    fn sequentialTransposeRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        _: []runtime_contract.TensorHandle,
        views: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        params: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try runtimeDispatchState(ctx);
        try requireInstructionStateBinding(state.mode, insn, &state.block.compiled_plan.plan, state_blocks);
        const p = try runtime_contract.paramAs(runtime_contract.TransposeOpParam, params, .transpose);
        if (insn.inputs.len != 1 or insn.outputs.len != 1) return error.InvalidInstructionBinding;
        const input_view = views[0];
        const in_buf = &state.buffer_views[runtime_contract.registerToIndex(insn.inputs[0])];
        const out_len = viewNumel(input_view);
        const out_slice = try state.block.instructionOutputSlice(
            state.buffer_views,
            state.scratch,
            insn.outputs[0],
            out_len,
        );

        const ndim: usize = input_view.rank;
        const dim0: usize = if (p.dim0 < 0)
            @intCast(@as(i64, @intCast(ndim)) + p.dim0)
        else
            @intCast(p.dim0);
        const dim1: usize = if (p.dim1 < 0)
            @intCast(@as(i64, @intCast(ndim)) + p.dim1)
        else
            @intCast(p.dim1);

        var in_shape_dims: [8]usize = undefined;
        for (0..ndim) |dim_idx| in_shape_dims[dim_idx] = input_view.shape[dim_idx];
        for (ndim..8) |dim_idx| in_shape_dims[dim_idx] = 0;

        var out_shape_dims: [8]usize = in_shape_dims;
        const tmp_dim = out_shape_dims[dim0];
        out_shape_dims[dim0] = out_shape_dims[dim1];
        out_shape_dims[dim1] = tmp_dim;

        const in_tv = tv.TensorView.initContiguous(in_buf.data_ptr.?, in_shape_dims[0..ndim], .f32);
        const out_tv = tv.TensorView.initContiguous(@ptrCast(out_slice.ptr), out_shape_dims[0..ndim], .f32);
        transpose_ops.transposeDispatch(out_tv, in_tv, dim0, dim1);

        var out_shape_i64 = viewToTensorShape(input_view);
        const tmp_i64 = out_shape_i64[dim0];
        out_shape_i64[dim0] = out_shape_i64[dim1];
        out_shape_i64[dim1] = tmp_i64;
        const out_idx = try instructionRegisterToBufferIndex(insn.outputs[0]);
        state.buffer_views[out_idx] = tensorFromSlice(
            out_slice[0..out_len],
            out_shape_i64,
            @intCast(input_view.rank),
        );
    }

    fn sequentialRopeRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        _: []runtime_contract.TensorHandle,
        views: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        _: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try runtimeDispatchState(ctx);
        try requireInstructionStateBinding(state.mode, insn, &state.block.compiled_plan.plan, state_blocks);
        if (insn.inputs.len != 1 or insn.outputs.len != 1) return error.InvalidInstructionBinding;
        const input_view = views[0];
        const in_buf = &state.buffer_views[runtime_contract.registerToIndex(insn.inputs[0])];
        const input_data = in_buf.asSlice(f32);
        const attn = state.block.block.getAttention() orelse {
            error_context.setContext("block={d}, op={d}, type=mamba", .{ state.block.block_idx, state.op_index });
            return error.RopeNotAvailableForMamba;
        };
        const rope = attn.rope orelse {
            error_context.setContext("block={d}, op={d}", .{ state.block.block_idx, state.op_index });
            return error.MissingRopeConfig;
        };
        const slot_state = state.slot_ctx.slotState();
        const pos_offset = if (state.slot_ctx.use_cache and slot_state.attn_cache != null)
            slot_state.attn_cache.?.cache_position
        else
            0;
        const rope_shape = viewToTensorShape(input_view);
        cpu_rotary.applyRopeTensorInPlace(
            input_data,
            @intCast(input_view.rank),
            rope_shape,
            rope.dim,
            pos_offset,
            rope,
        ) catch |err| {
            error_context.setContext("block={d}, op={d}, ndim={d}", .{
                state.block.block_idx,
                state.op_index,
                input_view.rank,
            });
            return err;
        };
        if (insn.inputs[0] == insn.outputs[0]) return;
        const numel = viewNumel(input_view);
        const out_slice = try state.block.instructionOutputSlice(
            state.buffer_views,
            state.scratch,
            insn.outputs[0],
            numel,
        );
        @memcpy(out_slice, input_data[0..numel]);
        const out_idx = try instructionRegisterToBufferIndex(insn.outputs[0]);
        state.buffer_views[out_idx] = tensorFromSlice(
            out_slice[0..numel],
            rope_shape,
            @intCast(input_view.rank),
        );
    }

    fn sequentialTriuRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        _: []runtime_contract.TensorHandle,
        views: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        params: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try runtimeDispatchState(ctx);
        try requireInstructionStateBinding(state.mode, insn, &state.block.compiled_plan.plan, state_blocks);
        const p = try runtime_contract.paramAs(runtime_contract.TriuOpParam, params, .triu);
        if (insn.inputs.len != 1 or insn.outputs.len != 1) return error.InvalidInstructionBinding;
        const input_view = views[0];
        const in_buf = &state.buffer_views[runtime_contract.registerToIndex(insn.inputs[0])];
        const out_buf = &state.buffer_views[runtime_contract.registerToIndex(insn.outputs[0])];
        const data = in_buf.asSlice(f32);
        const out_data = out_buf.asSlice(f32);
        const ndim: usize = input_view.rank;
        const rows: usize = input_view.shape[ndim - 2];
        const cols: usize = input_view.shape[ndim - 1];
        cpu_masking.triu(data, out_data, rows, cols, p.diagonal);
    }

    fn sequentialSdpaRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        _: []runtime_contract.TensorHandle,
        views: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        params: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try runtimeDispatchState(ctx);
        try requireInstructionStateBinding(state.mode, insn, &state.block.compiled_plan.plan, state_blocks);
        const p = try runtime_contract.paramAs(runtime_contract.SdpaOpParam, params, .scaled_dot_product_attention);
        if (insn.inputs.len != 3 or insn.outputs.len != 1) return error.InvalidInstructionBinding;
        const q_view = views[0];
        const k_view = views[1];

        if (q_view.rank != 4) {
            error_context.setContext("block={d}, op={d}, got {d}D, need 4D", .{
                state.block.block_idx,
                state.op_index,
                q_view.rank,
            });
            return error.InvalidShape;
        }

        const batch: usize = q_view.shape[0];
        const n_heads: usize = q_view.shape[1];
        const seq_q: usize = q_view.shape[2];
        const head_dim: usize = q_view.shape[3];
        const seq_k: usize = k_view.shape[2];
        const is_causal = p.is_causal != 0;
        const sdpa_scale: f32 = if (p.has_scale != 0) blk: {
            const data = params[0].data;
            if (data.len < @sizeOf(runtime_contract.SdpaOpParam) + 4) return error.InvalidParamBlockABI;
            break :blk @bitCast(std.mem.readInt(u32, data[@sizeOf(runtime_contract.SdpaOpParam)..][0..4], .little));
        } else 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
        const out_numel = batch * n_heads * seq_q * head_dim;
        const out_slice = try state.block.instructionOutputSlice(
            state.buffer_views,
            state.scratch,
            insn.outputs[0],
            out_numel,
        );
        const query_buf = &state.buffer_views[runtime_contract.registerToIndex(insn.inputs[0])];
        const key_buf = &state.buffer_views[runtime_contract.registerToIndex(insn.inputs[1])];
        const value_buf = &state.buffer_views[runtime_contract.registerToIndex(insn.inputs[2])];
        const q_shape_arr = [_]usize{ batch, n_heads, seq_q, head_dim };
        const k_shape_arr = [_]usize{ batch, n_heads, seq_k, head_dim };
        const v_shape_arr = [_]usize{ batch, n_heads, seq_k, head_dim };
        const out_shape_arr = [_]usize{ batch, n_heads, seq_q, head_dim };
        const q_tv = tv.TensorView.initContiguous(query_buf.data_ptr.?, &q_shape_arr, .f32);
        const k_tv = tv.TensorView.initContiguous(key_buf.data_ptr.?, &k_shape_arr, .f32);
        const v_tv = tv.TensorView.initContiguous(value_buf.data_ptr.?, &v_shape_arr, .f32);
        const out_tv = tv.TensorView.initContiguous(@ptrCast(out_slice.ptr), &out_shape_arr, .f32);

        if (is_causal) {
            attention_ops.sdpaCausal(out_tv, q_tv, k_tv, v_tv, sdpa_scale, 0, state.scratch.allocator) catch |err| {
                error_context.setContext("block={d}, op={d}, causal=true", .{ state.block.block_idx, state.op_index });
                return err;
            };
        } else {
            attention_ops.sdpa(out_tv, q_tv, k_tv, v_tv, null, sdpa_scale, state.scratch.allocator) catch |err| {
                error_context.setContext("block={d}, op={d}, causal=false", .{ state.block.block_idx, state.op_index });
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
        const out_idx = try instructionRegisterToBufferIndex(insn.outputs[0]);
        state.buffer_views[out_idx] = tensorFromSlice(
            out_slice[0..out_numel],
            sdpa_shape,
            4,
        );
    }

    fn bindKernelSharedStateFromInstruction(
        state: *RuntimeDispatchState,
        insn: *const runtime_contract.Instruction,
        state_blocks: []const runtime_contract.StateBlockHandle,
    ) !void {
        const shared_state = state.slot_ctx.sharedState();
        switch (insn.opcode) {
            .multihead_attention, .mla_attention => {
                if (insn.state_block_id) |state_id| {
                    const layered_cache = runtime_contract.findStateValue(
                        *LayeredBatchedKVCache,
                        state_blocks,
                        state_id,
                    ) orelse return error.InvalidStateDescriptorBinding;
                    if (state.block.block_idx >= layered_cache.layers.len) return error.InvalidStateDescriptorBinding;
                    shared_state.batched_cache = layered_cache.getLayer(state.block.block_idx);
                }
            },
            .shortconv => {
                if (insn.state_block_id) |state_id| {
                    const shortconv_scratch = runtime_contract.findStateValue(
                        *runtime.ShortConvScratch,
                        state_blocks,
                        state_id,
                    ) orelse return error.InvalidStateDescriptorBinding;
                    shared_state.shortconv_scratch = shortconv_scratch;
                }
            },
            .mamba_mixer => {
                if (insn.state_block_id) |state_id| {
                    const mamba_scratch = runtime_contract.findStateValue(
                        *runtime.MambaScratch,
                        state_blocks,
                        state_id,
                    ) orelse return error.InvalidStateDescriptorBinding;
                    shared_state.mamba_scratch = mamba_scratch;
                }
            },
            else => {},
        }
    }

    fn residualScaleValue(self: *const Block, scale: ResidualScale) f32 {
        return switch (scale) {
            .one => 1.0,
            .residual_multiplier => self.block.residual_multiplier,
            .literal => |v| v,
        };
    }

    fn scratchTempSlice(self: *const Block, scratch: *ScratchBuffer, reg_idx: usize, len: usize) []f32 {
        // All non-residual registers (1..63) map through the compiled liveness
        // allocator to physical scratch slots. Register 0 (residual) is handled
        // by resolveOutputSlice directly.
        if (reg_idx >= 1 and reg_idx < cpu_forward.NUM_TMP_BUFFERS) {
            const mapped_idx: usize = self.tmp_register_to_scratch_idx[reg_idx];
            std.debug.assert(mapped_idx >= 1);
            std.debug.assert(mapped_idx < cpu_forward.NUM_TMP_BUFFERS);
            return scratch.tmp[mapped_idx][0..len];
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

    fn resolveOutputSlice(self: *const Block, buffer_views: *[64]Tensor, scratch: *ScratchBuffer, reg_idx: usize, len: usize) []f32 {
        if (reg_idx == 0) return buffer_views[0].asSlice(f32)[0..len];
        return self.scratchTempSlice(scratch, reg_idx, len);
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
        scratch.registerTmpLayout(self.tmp_slot_width_hints, self.tmp_slot_active);
        try scratch.ensureForMode(if (use_cache) .decode else .prefill, seq_len);

        // Buffer lookup table: register index -> Tensor
        // Register 0 (residual) maps to the output buffer; all other registers
        // are backed by scratch slots assigned through liveness analysis.
        var buffer_views: [64]Tensor = undefined;
        buffer_views[@intFromEnum(BufferId.residual)] = out.*;
        for (1..self.compiled_plan.plan.register_count) |reg_idx| {
            const mapped = self.tmp_register_to_scratch_idx[reg_idx];
            buffer_views[reg_idx] = Tensor.view3DSlice(scratch.tmp[mapped], seq_len, self.hidden_size);
        }

        // Initialize residual stream with input
        copyTensor(x, out);

        // Populate shared scratch only for kernels present in this block.
        const is_mla = self.planUsesOpcode(.mla_attention);
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

        var dispatch_state = RuntimeDispatchState{
            .block = self,
            .op_index = 0,
            .buffer_views = &buffer_views,
            .scratch = scratch,
            .slot_ctx = ctx,
            .mode = .single_slot,
            .slot_index = 0,
            .slot_indices = &.{},
            .use_batched_dispatch = false,
        };
        try bindDispatchStateDescriptors(&dispatch_state);

        // Execute the operation sequence
        for (self.compiled_plan.plan.instructions, 0..) |insn, op_index| {
            dispatch_state.op_index = op_index;
            try self.dispatchInstructionWithState(&insn, &dispatch_state);
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

        if (runtime_contract.firstUnsupportedInstructionOpcode(&self.compiled_plan.plan, adapter_table)) |unsupported| {
            error_context.setContext("block={d}, op={d}, opcode={d}", .{
                self.block_idx,
                unsupported.instruction_index,
                @intFromEnum(unsupported.opcode),
            });
            return error.UnsupportedOpInSequentialMode;
        }

        for (self.compiled_plan.plan.instructions, 0..) |_, op_index| {
            const insn = self.compiled_plan.plan.instructions[op_index];
            if (insn.param_block_id == null) {
                error_context.setContext("block={d}, op={d}, param_block=MissingParamBlock", .{
                    self.block_idx,
                    op_index,
                });
                return error.MissingParamBlock;
            }
            switch (insn.opcode) {
                .rmsnorm,
                .multihead_attention,
                .mla_attention,
                .swiglu,
                .moe,
                .mamba_mixer,
                .shortconv,
                .embedding,
                => {
                    if (self.instruction_kernel_refs[op_index] == null) {
                        error_context.setContext("block={d}, op={d}, kernel_ref=MissingKernelBinding", .{
                            self.block_idx,
                            op_index,
                        });
                        return error.KernelIndexOutOfBounds;
                    }
                },
                .linear => {
                    const weight = self.instructionWeightRef(op_index) catch |err| {
                        error_context.setContext("block={d}, op={d}, weight_ref={s}", .{
                            self.block_idx,
                            op_index,
                            @errorName(err),
                        });
                        return err;
                    };
                    _ = cpu_linalg.matmulKernel(weight.dtype) catch |err| {
                        error_context.setContext("block={d}, op={d}, dtype={}", .{ self.block_idx, op_index, weight.dtype });
                        return err;
                    };
                },
                .add_param => {
                    _ = self.instructionWeightRef(op_index) catch |err| {
                        error_context.setContext("block={d}, op={d}, param_ref={s}", .{
                            self.block_idx,
                            op_index,
                            @errorName(err),
                        });
                        return err;
                    };
                },
                .add_param_scalar => {
                    _ = self.instructionWeightRef(op_index) catch |err| {
                        error_context.setContext("block={d}, op={d}, param_ref={s}", .{
                            self.block_idx,
                            op_index,
                            @errorName(err),
                        });
                        return err;
                    };
                },
                .mul_param => {
                    _ = self.instructionWeightRef(op_index) catch |err| {
                        error_context.setContext("block={d}, op={d}, param_ref={s}", .{
                            self.block_idx,
                            op_index,
                            @errorName(err),
                        });
                        return err;
                    };
                },
                .split => {
                    if (insn.outputs.len == 0) return error.TooManySplitOutputs;
                    const out_start_idx = runtime_contract.registerToIndex(insn.outputs[0]);
                    const max_outputs = cpu_forward.NUM_TMP_BUFFERS - out_start_idx;
                    if (out_start_idx < @intFromEnum(BufferId.tmp3) or insn.outputs.len > max_outputs) {
                        return error.TooManySplitOutputs;
                    }
                },
                .rope, .transpose, .scaled_dot_product_attention => {
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

        for (self.compiled_plan.plan.instructions, 0..) |insn, op_index| {
            try writer.writeByteNTimes(' ', indent + 2);
            try writer.print("[{}] {s}", .{ op_index, @tagName(insn.opcode) });
            if (self.instruction_kernel_refs[op_index]) |kernel| {
                try writer.writeAll(": ");
                switch (kernel) {
                    .norm => |n| try formatRmsNormLike(writer, n.dim, n.eps, n.weight_offset),
                    .attention => |a| try writer.print("Attention(n_heads={}, head_dim={})", .{ a.n_heads, a.head_dim }),
                    .swiglu => |m| try writer.print("MLP(d_ff={})", .{m.d_ff}),
                    .moe => |e| try writer.print("MoE(experts={}, per_tok={})", .{ e.num_experts, e.experts_per_token }),
                    .mamba => |m| try writer.print("Mamba(d_model={}, d_state={}, d_conv={})", .{ m.config.d_model, m.config.d_state, m.config.d_conv }),
                    .shortconv => |s| try writer.print("ShortConv(d_model={}, d_conv={})", .{ s.config.d_model, s.config.d_conv }),
                    .mla_attention => |a| try writer.print("MLA(n_heads={}, head_dim={})", .{ a.n_heads, a.head_dim }),
                    .embedding => try writer.writeAll("Embedding"),
                }
            }
            try writer.writeByte('\n');
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
    /// across multiple scheduler slots. Checked at load time; result must be
    /// cached — never call in the hot path.
    pub fn supportsBatchedDecodeSlots(self: *const Block) bool {
        for (self.compiled_plan.plan.instructions, 0..) |insn, op_index| {
            if (adapter_table[@intFromEnum(insn.opcode)] == null) return false;
            switch (insn.opcode) {
                .rmsnorm, .multihead_attention, .swiglu, .moe, .embedding => {
                    if (self.instruction_kernel_refs[op_index] == null) return false;
                },
                // These kernel opcodes are in the batched table but their
                // forwardBatchedSlots returns UnsupportedBatchedDecodeKernel.
                .mamba_mixer, .shortconv, .mla_attention => return false,
                .residual_add, .mul_scalar, .add_tensor => {},
                .vision_patch_embed, .vision_deepstack_extract, .vision_spatial_merge, .vision_scatter => {},
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
        state_blocks: []const runtime_contract.StateBlockHandle,
        slot_index: usize,
        use_cache: bool,
    ) !void {
        std.debug.assert(x.shape[0] == 1 and out.shape[0] == 1);
        const seq_len: usize = @intCast(x.shape[1]);
        scratch.registerTmpLayout(self.tmp_slot_width_hints, self.tmp_slot_active);
        try scratch.ensureForMode(if (use_cache) .decode else .prefill, seq_len);

        var buffer_views: [64]Tensor = undefined;
        buffer_views[@intFromEnum(BufferId.residual)] = out.*;
        for (1..self.compiled_plan.plan.register_count) |reg_idx| {
            const mapped = self.tmp_register_to_scratch_idx[reg_idx];
            buffer_views[reg_idx] = Tensor.view3DSlice(scratch.tmp[mapped], seq_len, self.hidden_size);
        }

        copyTensor(x, out);

        const is_mla = self.planUsesOpcode(.mla_attention);
        const slot_state = scratch.getSlotState(self.block_idx) orelse return error.InvalidState;
        var shared_state = SharedPersistentState{
            .mla_scratch = if (is_mla) scratch.getMLAScratch() else null,
            .mamba_scratch = null,
            .shortconv_scratch = null,
            .state_blocks = state_blocks,
        };
        const ctx = SlotContext{
            .slot_state_ptr = slot_state,
            .shared_state = &shared_state,
            .scratch = scratch,
            .use_cache = use_cache,
        };

        var dispatch_state = RuntimeDispatchState{
            .block = self,
            .op_index = 0,
            .buffer_views = &buffer_views,
            .scratch = scratch,
            .slot_ctx = ctx,
            .mode = .single_slot,
            .slot_index = slot_index,
            .slot_indices = &.{},
            .use_batched_dispatch = true,
        };
        try bindDispatchStateDescriptors(&dispatch_state);

        // Execute the operation sequence
        for (self.compiled_plan.plan.instructions, 0..) |insn, op_index| {
            dispatch_state.op_index = op_index;
            try self.dispatchInstructionWithState(&insn, &dispatch_state);
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
        state_blocks: []const runtime_contract.StateBlockHandle,
        slot_indices: []const usize,
        use_cache: bool,
    ) !void {
        std.debug.assert(x.shape[0] == 1 and out.shape[0] == 1);
        const batch_size: usize = @intCast(x.shape[1]);
        std.debug.assert(batch_size == slot_indices.len);
        scratch.registerTmpLayout(self.tmp_slot_width_hints, self.tmp_slot_active);
        try scratch.ensureForMode(if (use_cache) .decode else .prefill, batch_size);

        var buffer_views: [64]Tensor = undefined;
        buffer_views[@intFromEnum(BufferId.residual)] = out.*;
        for (1..self.compiled_plan.plan.register_count) |reg_idx| {
            const mapped = self.tmp_register_to_scratch_idx[reg_idx];
            buffer_views[reg_idx] = Tensor.view3DSlice(scratch.tmp[mapped], batch_size, self.hidden_size);
        }

        copyTensor(x, out);

        const is_mla = self.planUsesOpcode(.mla_attention);
        const slot_state = scratch.getSlotState(self.block_idx) orelse return error.InvalidState;
        var shared_state = SharedPersistentState{
            .mla_scratch = if (is_mla) scratch.getMLAScratch() else null,
            .mamba_scratch = null,
            .shortconv_scratch = null,
            .state_blocks = state_blocks,
        };
        const ctx = SlotContext{
            .slot_state_ptr = slot_state,
            .shared_state = &shared_state,
            .scratch = scratch,
            .use_cache = use_cache,
        };

        var dispatch_state = RuntimeDispatchState{
            .block = self,
            .op_index = 0,
            .buffer_views = &buffer_views,
            .scratch = scratch,
            .slot_ctx = ctx,
            .mode = .slot_batch,
            .slot_index = 0,
            .slot_indices = slot_indices,
            .use_batched_dispatch = true,
        };
        try bindDispatchStateDescriptors(&dispatch_state);

        for (self.compiled_plan.plan.instructions, 0..) |insn, op_index| {
            dispatch_state.op_index = op_index;
            try self.dispatchInstructionWithState(&insn, &dispatch_state);
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

    var block = try cpu_forward.TransformerBlock.init(
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
    errdefer block.deinit(allocator);
    try block.initWeightRegistry(allocator, block_weights);
    return block;
}

fn createTestBlock(
    allocator: std.mem.Allocator,
    transformer_block: *const cpu_forward.TransformerBlock,
    hidden_size: usize,
    program: []const LayerOp,
) !Block {
    return Block.initWithProgram(allocator, transformer_block, 0, hidden_size, program, .decode);
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

test "Block caches primitive linear weight bindings at init" {
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
    try testing.expectEqual(block.compiled_plan.plan.instructions.len, block.instruction_weight_refs.len);
    try testing.expect(block.instruction_weight_refs[0] != null);
    try block.validate();
}

test "Block caches kernel instruction bindings at init" {
    const allocator = testing.allocator;

    var weights = try TestWeights.init(allocator, 128, 512, 2, 32);
    defer weights.deinit(allocator);

    var transformer_block = try createTestTransformerBlock(allocator, &weights);
    defer transformer_block.deinit(allocator);

    const program = [_]LayerOp{
        .{ .kernel = .{ .id = 0, .in = .residual, .out = .norm_out, .debug_type = .norm } },
        .{ .kernel = .{ .id = 1, .in = .norm_out, .out = .branch_out, .debug_type = .multihead_attention } },
    };

    var block = try createTestBlock(allocator, &transformer_block, 128, &program);
    defer block.deinit(allocator);

    try testing.expectEqual(block.compiled_plan.plan.instructions.len, block.instruction_kernel_refs.len);
    try testing.expect(block.instruction_kernel_refs[0] != null);
    try testing.expect(block.instruction_kernel_refs[1] != null);
    try block.validate();
}

test "Block.validate rejects missing cached kernel binding" {
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

    const refs_mut = @constCast(block.instruction_kernel_refs.ptr);
    refs_mut[0] = null;

    try testing.expectError(error.KernelIndexOutOfBounds, block.validate());
}

test "Block.validate rejects missing cached primitive weight binding" {
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

    const refs_mut = @constCast(block.instruction_weight_refs.ptr);
    refs_mut[0] = null;

    try testing.expectError(error.MissingWeight, block.validate());
}

test "Block.validate rejects instructions without param block payload" {
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
    insn_mut[0].param_block_id = null;

    try testing.expectError(error.MissingParamBlock, block.validate());
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

test "Block rejects split with too many outputs at construction" {
    const allocator = testing.allocator;

    var weights = try TestWeights.init(allocator, 128, 512, 2, 32);
    defer weights.deinit(allocator);

    var transformer_block = try createTestTransformerBlock(allocator, &weights);
    defer transformer_block.deinit(allocator);

    // Split starting at tmp3 with 62 outputs (tmp3..tmp64) exceeds TMP_BUFFER_MAP_LEN (64).
    // register_count > TMP_BUFFER_MAP_LEN → error.UnsupportedModel from buildTmpRegisterScratchMap.
    const program = [_]LayerOp{
        .{ .split = .{ .in = .norm_out, .out_start = .tmp3, .num_outputs = 62, .split_sizes = &.{}, .dim = -1 } },
    };

    try testing.expectError(error.UnsupportedModel, createTestBlock(allocator, &transformer_block, 128, &program));
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
    try scratch.initAttention(&.{0});
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
    try scratch.initAttention(&.{0});
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

    // Create layered KV cache state descriptor binding for block 0.
    var layered_cache = try LayeredBatchedKVCache.init(allocator, 1, 4, 2, 32, 2048);
    defer layered_cache.deinit();
    var state_ref align(64) = runtime_contract.OpaqueStateRef{
        .ptr = @ptrCast(&layered_cache),
    };
    const state_blocks = [_]runtime_contract.StateBlockHandle{
        .{
            .id = @intFromEnum(runtime_contract.StateBlockId.kv_cache),
            .ptr = @ptrCast(&state_ref),
            .size = @sizeOf(runtime_contract.OpaqueStateRef),
            .align_bytes = @alignOf(runtime_contract.OpaqueStateRef),
        },
    };

    // Execute forward pass with batched cache
    try block.forwardWithBatchedCache(&input, &output, &scratch, state_blocks[0..], 0, false);

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

    try block.forwardWithBatchedCache(&input, &output, &scratch, &.{}, 0, false);

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

test "buildTmpRegisterScratchMap reuses physical tmp slots from liveness" {
    const allocator = testing.allocator;
    const reg0 = runtime_contract.registerFromIndex(0);
    const reg3 = runtime_contract.registerFromIndex(3);
    const reg4 = runtime_contract.registerFromIndex(4);
    const reg5 = runtime_contract.registerFromIndex(5);

    const instructions = [_]runtime_contract.Instruction{
        .{
            .opcode = .rmsnorm,
            .inputs = &.{reg0},
            .outputs = &.{reg3},
            .weights = &.{},
            .param_block_id = null,
            .state_block_id = null,
        },
        .{
            .opcode = .silu,
            .inputs = &.{reg3},
            .outputs = &.{reg4},
            .weights = &.{},
            .param_block_id = null,
            .state_block_id = null,
        },
        .{
            .opcode = .gelu,
            .inputs = &.{reg4},
            .outputs = &.{reg5},
            .weights = &.{},
            .param_block_id = null,
            .state_block_id = null,
        },
        .{
            .opcode = .residual_add,
            .inputs = &.{ reg0, reg5 },
            .outputs = &.{reg0},
            .weights = &.{},
            .param_block_id = null,
            .state_block_id = null,
        },
    };

    const kill0 = [_]u64{0};
    const kill1 = [_]u64{1 << 3};
    const kill2 = [_]u64{1 << 4};
    const kill3 = [_]u64{(1 << 0) | (1 << 5)};

    const compiled = runtime_contract.CompiledPlan{
        .plan = .{
            .instructions = &instructions,
            .register_count = 6,
            .state_descs = &.{},
        },
        .param_blocks = &.{},
        .weight_bindings = &.{},
        .register_buffer_specs = &.{
            .{ .size = 1, .@"align" = 64 },
            .{ .size = 1, .@"align" = 64 },
            .{ .size = 1, .@"align" = 64 },
            .{ .size = 1, .@"align" = 64 },
            .{ .size = 1, .@"align" = 64 },
            .{ .size = 1, .@"align" = 64 },
        },
        .liveness = .{
            .register_last_read = &.{ 3, std.math.maxInt(u32), std.math.maxInt(u32), 1, 2, 3 },
            .kill_after_instruction = &.{ kill0[0..], kill1[0..], kill2[0..], kill3[0..] },
        },
        .peak_registers = 2,
        .diagnostics = &.{},
    };

    const tmp_layout = try buildTmpRegisterScratchMap(allocator, &compiled, 256);
    const tmp_map = tmp_layout.map;
    try testing.expectEqual(tmp_map[@intFromEnum(BufferId.tmp3)], tmp_map[@intFromEnum(BufferId.tmp5)]);
    try testing.expect(tmp_map[@intFromEnum(BufferId.tmp4)] != tmp_map[@intFromEnum(BufferId.tmp3)]);
    try testing.expect(tmp_map[@intFromEnum(BufferId.tmp3)] >= 1);
}
