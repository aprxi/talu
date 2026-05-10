//! CUDA decoder block runtime, layer metadata, and batch state.

const std = @import("std");
const models = @import("models_pkg");
const op_types = models.op_types;
const plan_compiler = models.plan.compiler;
const runtime_contract = @import("runtime_contract_pkg");
const compute = @import("compute_pkg");
const tensor = @import("compute_pkg").tensor;
const log = @import("log_pkg");
const cpu_kernels = @import("../../cpu/kernels/root.zig");

const LoadedModel = models.LoadedModel;
const Tensor = tensor.Tensor;

const config = @import("config.zig");
const weights = @import("weights.zig");
const block_init = @import("block_init.zig");

const KvCacheDtype = config.KvCacheDtype;
const DeviceTensor = weights.DeviceTensor;
const missing_device_tensor = weights.missing_device_tensor;
const missing_host_tensor = weights.missing_host_tensor;
const LinearWeight = weights.LinearWeight;

const bufferSlice = @import("../weights/root.zig").bufferSlice;

pub const LayerAttentionRuntime = struct {
    q_dim: usize,
    q_projection_dim: usize,
    kv_dim: usize,
    d_ff: usize,
    sliding_window: usize,
    is_causal: bool,
    query_gate: bool,
    ln1_weight: DeviceTensor,
    ln2_weight: DeviceTensor,
    pre_ffn_norm_weight: ?DeviceTensor = null,
    post_ffn_norm_weight: ?DeviceTensor = null,
    q_norm_weight: ?DeviceTensor = null,
    k_norm_weight: ?DeviceTensor = null,
    q_proj: LinearWeight,
    k_proj: LinearWeight,
    v_proj: LinearWeight,
    o_proj: LinearWeight,
    w1: LinearWeight,
    w2: LinearWeight,
    w3: LinearWeight,
    k_cache: compute.cuda.Buffer,
    v_cache: compute.cuda.Buffer,
    k_scale: compute.cuda.Buffer = .{ .pointer = 0, .size = 0 },
    v_scale: compute.cuda.Buffer = .{ .pointer = 0, .size = 0 },
    kv_capacity: usize,
    qkv_i8_concat: compute.cuda.Buffer = .{ .pointer = 0, .size = 0 },
    qkv_scales_concat: compute.cuda.Buffer = .{ .pointer = 0, .size = 0 },
    qkv_concat_dims: [3]u32 = .{ 0, 0, 0 },
    slot_kv_index: usize,
    kv_shared_source_layer: ?usize = null,
    kv_shared_source_slot_kv_index: ?usize = null,
    use_v_norm: bool = false,
    cpu_kernel: ?cpu_kernels.MultiHeadAttention = null,
    cpu_cache: ?cpu_kernels.AttnCache = null,
    cpu_scratch: ?cpu_kernels.AttnTemp = null,
    cpu_matmul_scratch: ?compute.cpu.linalg.MatmulScratch = null,

    pub fn deinit(self: *LayerAttentionRuntime, device: *compute.cuda.Device) void {
        if (self.cpu_matmul_scratch) |*scratch| scratch.deinit();
        if (self.cpu_scratch) |*scratch| scratch.deinit(self.cpu_kernel.?.allocator);
        if (self.cpu_cache) |*cache| cache.deinit(self.cpu_kernel.?.allocator);
        if (self.qkv_scales_concat.pointer != 0) self.qkv_scales_concat.deinit(device);
        if (self.qkv_i8_concat.pointer != 0) self.qkv_i8_concat.deinit(device);
        if (self.v_scale.pointer != 0) self.v_scale.deinit(device);
        if (self.k_scale.pointer != 0) self.k_scale.deinit(device);
        self.v_cache.deinit(device);
        self.k_cache.deinit(device);
        if (self.post_ffn_norm_weight) |*w| w.deinit(device);
        if (self.pre_ffn_norm_weight) |*w| w.deinit(device);
        if (self.k_norm_weight) |*w| w.deinit(device);
        if (self.q_norm_weight) |*w| w.deinit(device);
        self.w3.deinit(device);
        self.w2.deinit(device);
        self.w1.deinit(device);
        self.o_proj.deinit(device);
        self.v_proj.deinit(device);
        self.k_proj.deinit(device);
        self.q_proj.deinit(device);
        self.ln2_weight.deinit(device);
        self.ln1_weight.deinit(device);
    }
};

pub const LayerAttentionExecConfig = struct {
    q_dim: usize,
    q_projection_dim: usize,
    kv_dim: usize,
    sliding_window: usize,
    is_causal: bool,
    query_gate: bool,
};

pub fn expectedAttentionQProjectionDim(cfg: *const LayerAttentionExecConfig) usize {
    return if (cfg.query_gate) cfg.q_projection_dim else cfg.q_dim;
}

pub fn tensorProjectionOutputDim(weight: *const Tensor, input_dim: usize) !usize {
    if (weight.n_dims != 2) return error.InvalidShape;
    const dim0: usize = @intCast(weight.shape[0]);
    const dim1: usize = @intCast(weight.shape[1]);
    if (dim0 == 0 or dim1 == 0) return error.InvalidShape;
    if (dim0 == input_dim and dim1 != input_dim) return dim1;
    if (dim1 == input_dim and dim0 != input_dim) return dim0;
    if (dim0 == input_dim and dim1 == input_dim) return input_dim;
    return dim0;
}

pub fn bufferF32RowCount(buffer: *const compute.cuda.Buffer, width: usize) !usize {
    if (width == 0) return error.InvalidArgument;
    const row_bytes = std.math.mul(usize, width, @sizeOf(f32)) catch return error.InvalidArgument;
    if (row_bytes == 0) return error.InvalidArgument;
    const rows = std.math.divExact(usize, buffer.size, row_bytes) catch return error.InvalidArgument;
    if (rows == 0) return error.InvalidArgument;
    return rows;
}

pub fn logicalF32RowSlice(
    buffer: *const compute.cuda.Buffer,
    rows: usize,
    row_index: usize,
    logical_width: usize,
) !compute.cuda.Buffer {
    if (rows == 0 or logical_width == 0 or row_index >= rows) return error.InvalidArgument;
    const row_bytes = std.math.mul(usize, logical_width, @sizeOf(f32)) catch return error.InvalidArgument;
    const packed_bytes = std.math.mul(usize, rows, row_bytes) catch return error.InvalidArgument;
    if (buffer.size < packed_bytes) return error.InvalidInstructionBinding;

    const row_stride = if (buffer.size == packed_bytes)
        row_bytes
    else blk: {
        if (buffer.size % rows != 0) return error.InvalidInstructionBinding;
        const stride = buffer.size / rows;
        if (stride < row_bytes) return error.InvalidInstructionBinding;
        break :blk stride;
    };

    const row_offset = std.math.mul(usize, row_index, row_stride) catch return error.InvalidArgument;
    return bufferSlice(buffer, row_offset, row_bytes);
}

pub const QkvI8ConcatRef = struct {
    i8_buf: compute.cuda.Buffer,
    scales_buf: compute.cuda.Buffer,
    dims: [3]u32,
};

pub const AttentionWeightRefs = struct {
    q_proj: ?*const LinearWeight = null,
    k_proj: ?*const LinearWeight = null,
    v_proj: ?*const LinearWeight = null,
    o_proj: ?*const LinearWeight = null,
    q_norm_weight: ?*const DeviceTensor = null,
    k_norm_weight: ?*const DeviceTensor = null,
};

pub const ShortConvBlockRuntime = struct {
    conv_dim: usize,
    d_conv: usize,
    d_ff: usize,
    ln1_weight: DeviceTensor,
    ln2_weight: ?DeviceTensor = null,
    in_proj: LinearWeight,
    out_proj: LinearWeight,
    conv_weight_time_major: DeviceTensor,
    conv_bias: ?DeviceTensor = null,
    conv_state: compute.cuda.Buffer,
    ffn_w1: ?LinearWeight = null,
    ffn_w2: ?LinearWeight = null,
    ffn_w3: ?LinearWeight = null,

    pub fn deinit(self: *ShortConvBlockRuntime, device: *compute.cuda.Device) void {
        if (self.ffn_w3) |*w| w.deinit(device);
        if (self.ffn_w2) |*w| w.deinit(device);
        if (self.ffn_w1) |*w| w.deinit(device);
        self.conv_state.deinit(device);
        if (self.conv_bias) |*w| w.deinit(device);
        self.conv_weight_time_major.deinit(device);
        self.out_proj.deinit(device);
        self.in_proj.deinit(device);
        if (self.ln2_weight) |*w| w.deinit(device);
        self.ln1_weight.deinit(device);
    }
};

pub const GatedDeltaSsmStateFormat = enum(u8) {
    f32,
    i8_per_column_scale,
};

pub const GatedDeltaBlockRuntime = struct {
    d_ff: usize,
    ln1_weight: DeviceTensor,
    ln2_weight: ?DeviceTensor = null,
    ffn_w1: ?LinearWeight = null,
    ffn_w2: ?LinearWeight = null,
    ffn_w3: ?LinearWeight = null,
    in_proj: LinearWeight,
    out_proj: LinearWeight,
    conv_weight_time_major: DeviceTensor,
    conv_bias: ?DeviceTensor = null,
    conv_state_dev: compute.cuda.Buffer,
    conv_ring_head: u32 = 0,
    a_log: DeviceTensor,
    dt_bias: ?DeviceTensor = null,
    norm_weight: DeviceTensor,
    ssm_state_dev: compute.cuda.Buffer,
    ssm_state_format: GatedDeltaSsmStateFormat = .f32,
    ssm_state_scales_offset: u32 = 0,
    kernel: cpu_kernels.GatedDeltaKernel,
    state: cpu_kernels.GatedDeltaState,
    scratch: cpu_kernels.GatedDeltaScratch,
    matmul_scratch: compute.cpu.linalg.MatmulScratch,

    pub fn deinit(self: *GatedDeltaBlockRuntime, allocator: std.mem.Allocator, device: *compute.cuda.Device) void {
        self.conv_state_dev.deinit(device);
        if (self.conv_bias) |*w| w.deinit(device);
        self.conv_weight_time_major.deinit(device);
        self.ssm_state_dev.deinit(device);
        if (self.dt_bias) |*w| w.deinit(device);
        self.norm_weight.deinit(device);
        self.a_log.deinit(device);
        self.out_proj.deinit(device);
        self.in_proj.deinit(device);
        if (self.ffn_w3) |*w| w.deinit(device);
        if (self.ffn_w2) |*w| w.deinit(device);
        if (self.ffn_w1) |*w| w.deinit(device);
        self.scratch.deinit();
        self.state.deinit();
        self.kernel.deinit();
        _ = allocator;
        self.matmul_scratch.deinit();
        if (self.ln2_weight) |*w| w.deinit(device);
        self.ln1_weight.deinit(device);
    }

    pub fn ssmStateDataBytes(self: *const GatedDeltaBlockRuntime) !usize {
        const d_head = @as(usize, self.kernel.config.d_head);
        const d_inner = @as(usize, self.kernel.config.n_heads) * d_head;
        const elems = std.math.mul(usize, d_inner, d_head) catch return error.InvalidArgument;
        return switch (self.ssm_state_format) {
            .f32 => std.math.mul(usize, elems, @sizeOf(f32)) catch return error.InvalidArgument,
            .i8_per_column_scale => elems,
        };
    }

    pub fn ssmStateScalesCount(self: *const GatedDeltaBlockRuntime) usize {
        return switch (self.ssm_state_format) {
            .f32 => 0,
            .i8_per_column_scale => @as(usize, self.kernel.config.n_heads) * @as(usize, self.kernel.config.d_head),
        };
    }
};

pub const ShortConvExecConfig = struct {
    conv_dim: usize,
    d_conv: usize,
};

pub const ShortConvWeightRefs = struct {
    in_proj: ?*const LinearWeight = null,
    conv_weight: ?*const DeviceTensor = null,
    out_proj: ?*const LinearWeight = null,
    conv_bias: ?*const DeviceTensor = null,
};

pub const GatedDeltaWeightRefs = struct {
    in_proj: ?*const Tensor = null,
    conv_weight: ?*const Tensor = null,
    a_log: ?*const Tensor = null,
    out_proj: ?*const Tensor = null,
    conv_bias: ?*const Tensor = null,
    dt_bias: ?*const Tensor = null,
    norm_weight: ?*const Tensor = null,
};

pub const SwiGluWeightRefs = struct {
    w1: ?*const LinearWeight = null,
    w3: ?*const LinearWeight = null,
    w2: ?*const LinearWeight = null,
    w1_bias: ?*const DeviceTensor = null,
    w2_bias: ?*const DeviceTensor = null,
};

pub const MoEWeightRefs = struct {
    expert_gate_up: []LinearWeight,
    expert_down: []LinearWeight,
    shared_gate: LinearWeight,
    shared_up: LinearWeight,
    shared_down: LinearWeight,
    router_proj: LinearWeight,
    // Optional MoE routing/normalization features used by architectures with
    // learned router input scaling and internal FFN/expert norms.
    router_input_scale: ?DeviceTensor = null,
    router_per_expert_scale: ?DeviceTensor = null,
    pre_ffn_norm: ?DeviceTensor = null,
    post_shared_norm: ?DeviceTensor = null,
    pre_expert_norm: ?DeviceTensor = null,
    post_expert_norm: ?DeviceTensor = null,
    post_combine_norm: ?DeviceTensor = null,
    // Optional sigmoid gate for shared expert output scaling.
    shared_expert_gate: ?LinearWeight = null,
    num_experts: u32,
    experts_per_token: u32,
    expert_d_ff: u32,
    shared_d_ff: u32,
    router_scalar: f32,
    use_gelu: bool = true,

    pub fn deinit(self: *MoEWeightRefs, allocator: std.mem.Allocator, device: *compute.cuda.Device) void {
        for (self.expert_gate_up) |*w| w.deinit(device);
        allocator.free(self.expert_gate_up);
        for (self.expert_down) |*w| w.deinit(device);
        allocator.free(self.expert_down);
        self.shared_gate.deinit(device);
        self.shared_up.deinit(device);
        self.shared_down.deinit(device);
        self.router_proj.deinit(device);
        if (self.router_input_scale) |*t| t.deinit(device);
        if (self.router_per_expert_scale) |*t| t.deinit(device);
        if (self.pre_ffn_norm) |*t| t.deinit(device);
        if (self.post_shared_norm) |*t| t.deinit(device);
        if (self.pre_expert_norm) |*t| t.deinit(device);
        if (self.post_expert_norm) |*t| t.deinit(device);
        if (self.post_combine_norm) |*t| t.deinit(device);
        if (self.shared_expert_gate) |*w| w.deinit(device);
    }
};

pub const BlockRuntimeLayer = struct {
    pub const invalid_slot = std.math.maxInt(u8);
    const MaxNormWeights = 4;

    compiled_plan: ?runtime_contract.CompiledPlan = null,
    instruction_norm_weight_slots: []?*const DeviceTensor = &.{},
    instruction_attention_exec_meta: []?LayerAttentionExecConfig = &.{},
    instruction_attention_weight_slots: []?AttentionWeightRefs = &.{},
    instruction_shortconv_exec_meta: []?ShortConvExecConfig = &.{},
    instruction_shortconv_weight_slots: []?ShortConvWeightRefs = &.{},
    instruction_gated_delta_weight_slots: []?GatedDeltaWeightRefs = &.{},
    instruction_swiglu_weight_slots: []?SwiGluWeightRefs = &.{},
    instruction_moe_weight_slots: []?*const MoEWeightRefs = &.{},
    instruction_weight_offsets: []u32 = &.{},
    instruction_weight_ptrs: []?*anyopaque = &.{},
    register_to_slot_map: []const u8 = &.{},
    slot_width_hints: []const usize = &.{},
    attention_runtime: ?LayerAttentionRuntime = null,
    shortconv_runtime: ?ShortConvBlockRuntime = null,
    gated_delta_runtime: ?GatedDeltaBlockRuntime = null,
    attention_binding: ?*LayerAttentionRuntime = null,
    shortconv_binding: ?*ShortConvBlockRuntime = null,
    gated_delta_binding: ?*GatedDeltaBlockRuntime = null,
    moe_runtime: ?MoEWeightRefs = null,
    moe_binding: ?*MoEWeightRefs = null,
    norm_weights: [MaxNormWeights]?*const DeviceTensor = [_]?*const DeviceTensor{null} ** MaxNormWeights,
    norm_weight_count: u8 = 0,

    pub fn instructionKernelIdFromWeightBindings(
        compiled: *const runtime_contract.CompiledPlan,
        op_index: usize,
        opcode: runtime_contract.Opcode,
    ) !u32 {
        return runtime_contract.instructionKernelBindingId(compiled, op_index, opcode);
    }

    const InstructionRefBinderFn = *const fn (
        self: *BlockRuntimeLayer,
        compiled: *const runtime_contract.CompiledPlan,
        op_index: usize,
        insn: *const runtime_contract.Instruction,
        norm_index: *usize,
    ) anyerror!void;

    pub fn bindInstructionNoop(
        _: *BlockRuntimeLayer,
        _: *const runtime_contract.CompiledPlan,
        _: usize,
        _: *const runtime_contract.Instruction,
        _: *usize,
    ) !void {}

    pub fn bindInstructionRmsNorm(
        self: *BlockRuntimeLayer,
        compiled: *const runtime_contract.CompiledPlan,
        op_index: usize,
        insn: *const runtime_contract.Instruction,
        norm_index: *usize,
    ) !void {
        _ = try instructionKernelIdFromWeightBindings(compiled, op_index, insn.opcode);
        if (norm_index.* >= self.norm_weight_count) return error.UnsupportedModel;
        const weight = self.norm_weights[norm_index.*] orelse return error.UnsupportedModel;
        self.instruction_norm_weight_slots[op_index] = weight;
        norm_index.* += 1;
    }

    pub fn bindInstructionAttention(
        self: *BlockRuntimeLayer,
        compiled: *const runtime_contract.CompiledPlan,
        op_index: usize,
        insn: *const runtime_contract.Instruction,
        _: *usize,
    ) !void {
        _ = try instructionKernelIdFromWeightBindings(compiled, op_index, insn.opcode);
        const binding = self.attention_binding orelse return error.UnsupportedModel;
        self.instruction_attention_exec_meta[op_index] = .{
            .q_dim = binding.q_dim,
            .q_projection_dim = binding.q_projection_dim,
            .kv_dim = binding.kv_dim,
            .sliding_window = binding.sliding_window,
            .is_causal = binding.is_causal,
            .query_gate = binding.query_gate,
        };
        self.instruction_attention_weight_slots[op_index] = .{
            .q_proj = &binding.q_proj,
            .k_proj = &binding.k_proj,
            .v_proj = &binding.v_proj,
            .o_proj = &binding.o_proj,
            .q_norm_weight = if (binding.q_norm_weight) |*weight| weight else null,
            .k_norm_weight = if (binding.k_norm_weight) |*weight| weight else null,
        };
    }

    pub fn bindInstructionShortConv(
        self: *BlockRuntimeLayer,
        compiled: *const runtime_contract.CompiledPlan,
        op_index: usize,
        insn: *const runtime_contract.Instruction,
        _: *usize,
    ) !void {
        _ = try instructionKernelIdFromWeightBindings(compiled, op_index, insn.opcode);
        const binding = self.shortconv_binding orelse return error.UnsupportedModel;
        self.instruction_shortconv_exec_meta[op_index] = .{
            .conv_dim = binding.conv_dim,
            .d_conv = binding.d_conv,
        };
        self.instruction_shortconv_weight_slots[op_index] = .{
            .in_proj = &binding.in_proj,
            .conv_weight = &binding.conv_weight_time_major,
            .out_proj = &binding.out_proj,
            .conv_bias = if (binding.conv_bias) |*weight| weight else null,
        };
    }

    pub fn bindInstructionGatedDelta(
        self: *BlockRuntimeLayer,
        compiled: *const runtime_contract.CompiledPlan,
        op_index: usize,
        insn: *const runtime_contract.Instruction,
        _: *usize,
    ) !void {
        _ = try instructionKernelIdFromWeightBindings(compiled, op_index, insn.opcode);
        const binding = self.gated_delta_binding orelse return error.UnsupportedModel;
        self.instruction_gated_delta_weight_slots[op_index] = .{
            .in_proj = binding.kernel.weights.in_proj,
            .conv_weight = binding.kernel.weights.conv1d_weight,
            .a_log = binding.kernel.weights.A_log,
            .out_proj = binding.kernel.weights.out_proj,
            .conv_bias = binding.kernel.weights.conv1d_bias,
            .dt_bias = binding.kernel.weights.dt_bias,
            .norm_weight = binding.kernel.weights.norm_weight,
        };
    }

    pub fn bindInstructionSwiGlu(
        self: *BlockRuntimeLayer,
        compiled: *const runtime_contract.CompiledPlan,
        op_index: usize,
        insn: *const runtime_contract.Instruction,
        _: *usize,
    ) !void {
        _ = try instructionKernelIdFromWeightBindings(compiled, op_index, insn.opcode);
        if (self.attention_binding) |binding| {
            self.instruction_swiglu_weight_slots[op_index] = .{
                .w1 = &binding.w1,
                .w3 = &binding.w3,
                .w2 = &binding.w2,
                .w1_bias = null,
                .w2_bias = null,
            };
            return;
        }
        if (self.shortconv_binding) |binding| {
            self.instruction_swiglu_weight_slots[op_index] = .{
                .w1 = if (binding.ffn_w1) |*w| w else null,
                .w3 = if (binding.ffn_w3) |*w| w else null,
                .w2 = if (binding.ffn_w2) |*w| w else null,
                .w1_bias = null,
                .w2_bias = null,
            };
            return;
        }
        if (self.gated_delta_binding) |binding| {
            self.instruction_swiglu_weight_slots[op_index] = .{
                .w1 = if (binding.ffn_w1) |*w| w else null,
                .w3 = if (binding.ffn_w3) |*w| w else null,
                .w2 = if (binding.ffn_w2) |*w| w else null,
                .w1_bias = null,
                .w2_bias = null,
            };
            return;
        }
        return error.UnsupportedModel;
    }

    pub fn bindInstructionMoE(
        self: *BlockRuntimeLayer,
        compiled: *const runtime_contract.CompiledPlan,
        op_index: usize,
        insn: *const runtime_contract.Instruction,
        _: *usize,
    ) !void {
        _ = try instructionKernelIdFromWeightBindings(compiled, op_index, insn.opcode);
        const binding = self.moe_binding orelse return error.UnsupportedModel;
        self.instruction_moe_weight_slots[op_index] = binding;
    }

    const instruction_rebind_table: [256]?InstructionRefBinderFn = blk: {
        var table: [256]?InstructionRefBinderFn = [_]?InstructionRefBinderFn{bindInstructionNoop} ** 256;
        table[@intFromEnum(runtime_contract.Opcode.rmsnorm)] = bindInstructionRmsNorm;
        table[@intFromEnum(runtime_contract.Opcode.multihead_attention)] = bindInstructionAttention;
        table[@intFromEnum(runtime_contract.Opcode.mla_attention)] = bindInstructionAttention;
        table[@intFromEnum(runtime_contract.Opcode.gated_delta_net)] = bindInstructionGatedDelta;
        table[@intFromEnum(runtime_contract.Opcode.shortconv)] = bindInstructionShortConv;
        table[@intFromEnum(runtime_contract.Opcode.swiglu)] = bindInstructionSwiGlu;
        table[@intFromEnum(runtime_contract.Opcode.moe)] = bindInstructionMoE;
        break :blk table;
    };

    pub fn rebuildInstructionMetadata(self: *BlockRuntimeLayer, allocator: std.mem.Allocator) !void {
        if (self.instruction_norm_weight_slots.len != 0) {
            allocator.free(self.instruction_norm_weight_slots);
            self.instruction_norm_weight_slots = &.{};
        }
        if (self.instruction_attention_exec_meta.len != 0) {
            allocator.free(self.instruction_attention_exec_meta);
            self.instruction_attention_exec_meta = &.{};
        }
        if (self.instruction_attention_weight_slots.len != 0) {
            allocator.free(self.instruction_attention_weight_slots);
            self.instruction_attention_weight_slots = &.{};
        }
        if (self.instruction_shortconv_exec_meta.len != 0) {
            allocator.free(self.instruction_shortconv_exec_meta);
            self.instruction_shortconv_exec_meta = &.{};
        }
        if (self.instruction_shortconv_weight_slots.len != 0) {
            allocator.free(self.instruction_shortconv_weight_slots);
            self.instruction_shortconv_weight_slots = &.{};
        }
        if (self.instruction_gated_delta_weight_slots.len != 0) {
            allocator.free(self.instruction_gated_delta_weight_slots);
            self.instruction_gated_delta_weight_slots = &.{};
        }
        if (self.instruction_swiglu_weight_slots.len != 0) {
            allocator.free(self.instruction_swiglu_weight_slots);
            self.instruction_swiglu_weight_slots = &.{};
        }
        if (self.instruction_moe_weight_slots.len != 0) {
            allocator.free(self.instruction_moe_weight_slots);
            self.instruction_moe_weight_slots = &.{};
        }
        if (self.instruction_weight_offsets.len != 0) {
            allocator.free(self.instruction_weight_offsets);
            self.instruction_weight_offsets = &.{};
        }
        if (self.instruction_weight_ptrs.len != 0) {
            allocator.free(self.instruction_weight_ptrs);
            self.instruction_weight_ptrs = &.{};
        }

        const compiled = self.compiled_plan orelse return;
        const len = compiled.plan.instructions.len;
        self.instruction_norm_weight_slots = try allocator.alloc(?*const DeviceTensor, len);
        self.instruction_attention_exec_meta = try allocator.alloc(?LayerAttentionExecConfig, len);
        self.instruction_attention_weight_slots = try allocator.alloc(?AttentionWeightRefs, len);
        self.instruction_shortconv_exec_meta = try allocator.alloc(?ShortConvExecConfig, len);
        self.instruction_shortconv_weight_slots = try allocator.alloc(?ShortConvWeightRefs, len);
        self.instruction_gated_delta_weight_slots = try allocator.alloc(?GatedDeltaWeightRefs, len);
        self.instruction_swiglu_weight_slots = try allocator.alloc(?SwiGluWeightRefs, len);
        self.instruction_moe_weight_slots = try allocator.alloc(?*const MoEWeightRefs, len);
        @memset(self.instruction_norm_weight_slots, null);
        @memset(self.instruction_attention_exec_meta, null);
        @memset(self.instruction_attention_weight_slots, null);
        @memset(self.instruction_shortconv_exec_meta, null);
        @memset(self.instruction_shortconv_weight_slots, null);
        @memset(self.instruction_gated_delta_weight_slots, null);
        @memset(self.instruction_swiglu_weight_slots, null);
        @memset(self.instruction_moe_weight_slots, null);

        var norm_index: usize = 0;
        for (compiled.plan.instructions, 0..) |insn, op_index| {
            const binder = instruction_rebind_table[@intFromEnum(insn.opcode)] orelse continue;
            try binder(self, &compiled, op_index, &insn, &norm_index);
        }
        try self.buildInstructionWeightTable(allocator, &compiled);
    }

    pub fn resolveInstructionWeightPtrForSlot(
        self: *const BlockRuntimeLayer,
        opcode: runtime_contract.Opcode,
        op_index: usize,
        slot_idx: usize,
    ) !*anyopaque {
        switch (opcode) {
            .rmsnorm => {
                return switch (slot_idx) {
                    0 => blk: {
                        const weight = try self.instructionNormWeightRef(op_index);
                        break :blk @ptrCast(@constCast(weight));
                    },
                    1 => @ptrCast(@constCast(&missing_device_tensor)),
                    else => error.InvalidWeightRefCount,
                };
            },
            .multihead_attention => {
                if (op_index >= self.instruction_attention_weight_slots.len) return error.InvalidInstructionIndex;
                const binding = self.instruction_attention_weight_slots[op_index] orelse return error.UnsupportedModel;
                return switch (slot_idx) {
                    0 => @ptrCast(@constCast(binding.q_proj orelse return error.MissingWeight)),
                    1 => @ptrCast(@constCast(binding.k_proj orelse return error.MissingWeight)),
                    2 => @ptrCast(@constCast(binding.v_proj orelse return error.MissingWeight)),
                    3 => @ptrCast(@constCast(binding.o_proj orelse return error.MissingWeight)),
                    4 => if (binding.q_norm_weight) |q_norm|
                        @ptrCast(@constCast(q_norm))
                    else
                        @ptrCast(@constCast(&missing_device_tensor)),
                    5 => if (binding.k_norm_weight) |k_norm|
                        @ptrCast(@constCast(k_norm))
                    else
                        @ptrCast(@constCast(&missing_device_tensor)),
                    6 => @ptrCast(@constCast(&missing_device_tensor)),
                    7 => @ptrCast(@constCast(&missing_device_tensor)),
                    8 => @ptrCast(@constCast(&missing_device_tensor)),
                    9 => @ptrCast(@constCast(&missing_device_tensor)),
                    10 => @ptrCast(@constCast(&missing_device_tensor)),
                    else => error.InvalidWeightRefCount,
                };
            },
            .mla_attention => {
                return error.UnsupportedModel;
            },
            .mamba_mixer => {
                return error.UnsupportedModel;
            },
            .gated_delta_net => {
                if (op_index >= self.instruction_gated_delta_weight_slots.len) return error.InvalidInstructionIndex;
                const binding = self.instruction_gated_delta_weight_slots[op_index] orelse return error.UnsupportedModel;
                const missing_tensor = &missing_host_tensor;
                return switch (slot_idx) {
                    0 => @ptrCast(@constCast(binding.in_proj orelse missing_tensor)),
                    1 => @ptrCast(@constCast(binding.conv_weight orelse missing_tensor)),
                    2 => @ptrCast(@constCast(binding.a_log orelse missing_tensor)),
                    3 => @ptrCast(@constCast(binding.out_proj orelse missing_tensor)),
                    4 => @ptrCast(@constCast(binding.conv_bias orelse missing_tensor)),
                    5 => @ptrCast(@constCast(binding.dt_bias orelse missing_tensor)),
                    6 => @ptrCast(@constCast(binding.norm_weight orelse missing_tensor)),
                    else => error.InvalidWeightRefCount,
                };
            },
            .swiglu => {
                if (op_index >= self.instruction_swiglu_weight_slots.len) return error.InvalidInstructionIndex;
                const binding = self.instruction_swiglu_weight_slots[op_index] orelse return error.UnsupportedModel;
                return switch (slot_idx) {
                    0 => @ptrCast(@constCast(binding.w1 orelse return error.MissingWeight)),
                    1 => @ptrCast(@constCast(binding.w3 orelse return error.MissingWeight)),
                    2 => @ptrCast(@constCast(binding.w2 orelse return error.MissingWeight)),
                    3 => if (binding.w1_bias) |w1_bias|
                        @ptrCast(@constCast(w1_bias))
                    else
                        @ptrCast(@constCast(&missing_device_tensor)),
                    4 => if (binding.w2_bias) |w2_bias|
                        @ptrCast(@constCast(w2_bias))
                    else
                        @ptrCast(@constCast(&missing_device_tensor)),
                    else => error.InvalidWeightRefCount,
                };
            },
            .moe => {
                // MoE weights are accessed directly via MoEWeightRefs in the adapter,
                // not through the weight handle system. Return placeholders for plan validation.
                return @ptrCast(@constCast(&missing_device_tensor));
            },
            .shortconv => {
                if (op_index >= self.instruction_shortconv_weight_slots.len) return error.InvalidInstructionIndex;
                const binding = self.instruction_shortconv_weight_slots[op_index] orelse return error.UnsupportedModel;
                return switch (slot_idx) {
                    0 => @ptrCast(@constCast(binding.in_proj orelse return error.MissingWeight)),
                    1 => @ptrCast(@constCast(binding.conv_weight orelse return error.MissingWeight)),
                    2 => @ptrCast(@constCast(binding.out_proj orelse return error.MissingWeight)),
                    3 => if (binding.conv_bias) |conv_bias|
                        @ptrCast(@constCast(conv_bias))
                    else
                        @ptrCast(@constCast(&missing_device_tensor)),
                    else => error.InvalidWeightRefCount,
                };
            },
            else => return error.InvalidInstructionBinding,
        }
    }

    pub fn resolveInstructionWeightPtr(
        self: *const BlockRuntimeLayer,
        opcode: runtime_contract.Opcode,
        op_index: usize,
        slot_name: []const u8,
        slot_idx: usize,
    ) !*anyopaque {
        const expected_slots = runtime_contract.expectedKernelWeightSlots(opcode);
        if (slot_idx >= expected_slots.len) return error.InvalidWeightRefCount;
        if (!std.mem.eql(u8, expected_slots[slot_idx], slot_name)) return error.InvalidWeightBindingName;
        return self.resolveInstructionWeightPtrForSlot(opcode, op_index, slot_idx);
    }

    pub fn buildInstructionWeightTable(
        self: *BlockRuntimeLayer,
        allocator: std.mem.Allocator,
        compiled: *const runtime_contract.CompiledPlan,
    ) !void {
        const insn_len = compiled.plan.instructions.len;
        const offsets = try allocator.alloc(u32, insn_len + 1);
        errdefer allocator.free(offsets);

        var total_slots: usize = 0;
        for (compiled.plan.instructions) |insn| total_slots += insn.weights.len;
        const ptrs = try allocator.alloc(?*anyopaque, total_slots);
        errdefer allocator.free(ptrs);
        @memset(ptrs, null);

        var cursor: usize = 0;
        for (compiled.plan.instructions, 0..) |insn, op_index| {
            offsets[op_index] = @intCast(cursor);
            const expected_slots = runtime_contract.expectedKernelWeightSlots(insn.opcode);
            if (insn.weights.len != expected_slots.len) return error.InvalidWeightRefCount;
            for (insn.weights, 0..) |_, slot_idx| {
                const parsed = try runtime_contract.instructionKernelWeightBinding(
                    compiled,
                    op_index,
                    insn.opcode,
                    slot_idx,
                );
                const weight_ptr = try self.resolveInstructionWeightPtr(insn.opcode, op_index, parsed.slot_name, slot_idx);
                ptrs[cursor] = weight_ptr;
                cursor += 1;
            }
        }
        offsets[insn_len] = @intCast(cursor);
        self.instruction_weight_offsets = offsets;
        self.instruction_weight_ptrs = ptrs;
    }

    pub fn instructionNormWeightRef(self: *const BlockRuntimeLayer, op_index: usize) !*const DeviceTensor {
        if (op_index >= self.instruction_norm_weight_slots.len) return error.InvalidInstructionIndex;
        return self.instruction_norm_weight_slots[op_index] orelse return error.UnsupportedModel;
    }

    pub fn instructionAttentionRef(self: *const BlockRuntimeLayer, op_index: usize) !*const LayerAttentionExecConfig {
        if (op_index >= self.instruction_attention_exec_meta.len) return error.InvalidInstructionIndex;
        if (self.instruction_attention_exec_meta[op_index]) |*binding| return binding;
        return error.UnsupportedModel;
    }

    pub fn instructionShortConvRef(self: *const BlockRuntimeLayer, op_index: usize) !*const ShortConvExecConfig {
        if (op_index >= self.instruction_shortconv_exec_meta.len) return error.InvalidInstructionIndex;
        if (self.instruction_shortconv_exec_meta[op_index]) |*binding| return binding;
        return error.UnsupportedModel;
    }

    pub fn appendLayerNormWeight(layer: *BlockRuntimeLayer, weight: ?*const DeviceTensor) void {
        const value = weight orelse return;
        if (layer.norm_weight_count >= layer.norm_weights.len) return;
        layer.norm_weights[layer.norm_weight_count] = value;
        layer.norm_weight_count += 1;
    }

    pub fn bindAttentionNormWeights(layer: *BlockRuntimeLayer, block: *const LayerAttentionRuntime) void {
        appendLayerNormWeight(layer, &block.ln1_weight);
        appendLayerNormWeight(layer, &block.ln2_weight);
        if (block.pre_ffn_norm_weight) |*weight| {
            appendLayerNormWeight(layer, weight);
        } else if (block.post_ffn_norm_weight) |*weight| {
            appendLayerNormWeight(layer, weight);
        }
        if (block.post_ffn_norm_weight) |*weight| {
            appendLayerNormWeight(layer, weight);
        }
    }

    pub fn bindShortConvNormWeights(layer: *BlockRuntimeLayer, block: *const ShortConvBlockRuntime) void {
        appendLayerNormWeight(layer, &block.ln1_weight);
        if (block.ln2_weight) |*weight| {
            appendLayerNormWeight(layer, weight);
        }
    }

    pub fn bindGatedDeltaNormWeights(layer: *BlockRuntimeLayer, block: *const GatedDeltaBlockRuntime) void {
        appendLayerNormWeight(layer, &block.ln1_weight);
        if (block.ln2_weight) |*weight| {
            appendLayerNormWeight(layer, weight);
        }
    }

    pub fn deinit(self: *BlockRuntimeLayer, allocator: std.mem.Allocator, device: *compute.cuda.Device) void {
        if (self.register_to_slot_map.len != 0) allocator.free(self.register_to_slot_map);
        if (self.slot_width_hints.len != 0) allocator.free(self.slot_width_hints);
        if (self.instruction_norm_weight_slots.len != 0) allocator.free(self.instruction_norm_weight_slots);
        if (self.instruction_attention_exec_meta.len != 0) allocator.free(self.instruction_attention_exec_meta);
        if (self.instruction_attention_weight_slots.len != 0) allocator.free(self.instruction_attention_weight_slots);
        if (self.instruction_shortconv_exec_meta.len != 0) allocator.free(self.instruction_shortconv_exec_meta);
        if (self.instruction_shortconv_weight_slots.len != 0) allocator.free(self.instruction_shortconv_weight_slots);
        if (self.instruction_gated_delta_weight_slots.len != 0) allocator.free(self.instruction_gated_delta_weight_slots);
        if (self.instruction_swiglu_weight_slots.len != 0) allocator.free(self.instruction_swiglu_weight_slots);
        if (self.instruction_moe_weight_slots.len != 0) allocator.free(self.instruction_moe_weight_slots);
        if (self.instruction_weight_offsets.len != 0) allocator.free(self.instruction_weight_offsets);
        if (self.instruction_weight_ptrs.len != 0) allocator.free(self.instruction_weight_ptrs);
        if (self.compiled_plan) |*compiled_plan| {
            plan_compiler.deinitCompiledPlan(allocator, compiled_plan);
            self.compiled_plan = null;
        }
        if (self.attention_runtime) |*block| block.deinit(device);
        if (self.shortconv_runtime) |*block| block.deinit(device);
        if (self.gated_delta_runtime) |*block| block.deinit(allocator, device);
        if (self.moe_runtime) |*moe| moe.deinit(allocator, device);
        self.* = .{};
    }
};

pub fn buildCudaLayerProgramRegisterSlotMap(
    allocator: std.mem.Allocator,
    compiled: *const runtime_contract.CompiledPlan,
) ![]u8 {
    const invalid_slot = BlockRuntimeLayer.invalid_slot;
    const register_count = compiled.plan.register_count;
    const register_to_slot = try allocator.alloc(u8, register_count);
    @memset(register_to_slot, invalid_slot);
    errdefer allocator.free(register_to_slot);
    if (register_count <= 1) return register_to_slot;

    const register_specs = try allocator.alloc(runtime_contract.RegisterBufferSpec, register_count);
    defer allocator.free(register_specs);
    // Register 0 (residual) uses runtime_buffers.input_dev, not a slot buffer.
    register_specs[0] = .{ .size = 0, .@"align" = 0 };
    if (compiled.register_buffer_specs.len != register_count) return error.InvalidRegisterSpecCount;
    // Plan specs already contain floors applied at compile time.
    // Backends consume specs exactly.
    for (register_specs[1..], 1..) |*spec, idx| {
        const plan_spec = compiled.register_buffer_specs[idx];
        spec.* = .{
            .size = plan_spec.size,
            .@"align" = plan_spec.@"align",
            .dtype = plan_spec.dtype,
            .layout = plan_spec.layout,
        };
    }

    var physical = try runtime_contract.buildPhysicalMappingLinearScan(allocator, compiled, register_specs);
    defer runtime_contract.deinitPhysicalMapping(allocator, &physical);
    if (physical.physical_count == 0) return register_to_slot;

    const physical_to_slot = try allocator.alloc(u8, physical.physical_count);
    defer allocator.free(physical_to_slot);
    @memset(physical_to_slot, invalid_slot);

    var next_slot: u8 = 0;
    const invalid_physical = std.math.maxInt(u16);
    var register_idx: usize = 0;
    while (register_idx < register_count) : (register_idx += 1) {
        const physical_id_u16 = physical.register_to_physical[register_idx];
        if (physical_id_u16 == invalid_physical) continue;
        const physical_id: usize = physical_id_u16;
        if (physical_id >= physical_to_slot.len) return error.UnsupportedModel;
        if (physical_to_slot[physical_id] == invalid_slot) {
            physical_to_slot[physical_id] = next_slot;
            next_slot += 1;
        }
        register_to_slot[register_idx] = physical_to_slot[physical_id];
    }

    return register_to_slot;
}

pub fn buildCudaLayerProgramSlotWidthHints(
    allocator: std.mem.Allocator,
    compiled: *const runtime_contract.CompiledPlan,
    register_to_slot_map: []const u8,
) ![]usize {
    const invalid_slot = BlockRuntimeLayer.invalid_slot;
    const register_count: usize = compiled.plan.register_count;
    if (register_to_slot_map.len != register_count) return error.InvalidRegisterSpecCount;
    if (compiled.register_buffer_specs.len != register_count) return error.InvalidRegisterSpecCount;

    var required_slots: usize = 0;
    for (register_to_slot_map) |slot_idx| {
        if (slot_idx == invalid_slot) continue;
        const next = @as(usize, slot_idx) + 1;
        if (next > required_slots) required_slots = next;
    }
    if (required_slots == 0) return &.{};

    const slot_width_hints = try allocator.alloc(usize, required_slots);
    @memset(slot_width_hints, 0);
    errdefer allocator.free(slot_width_hints);

    const register_specs = try allocator.alloc(runtime_contract.RegisterBufferSpec, register_count);
    defer allocator.free(register_specs);
    register_specs[0] = .{ .size = 0, .@"align" = 0 };
    for (register_specs[1..], 1..) |*spec, idx| {
        const plan_spec = compiled.register_buffer_specs[idx];
        spec.* = .{
            .size = plan_spec.size,
            .@"align" = plan_spec.@"align",
            .dtype = plan_spec.dtype,
            .layout = plan_spec.layout,
        };
    }
    var physical = try runtime_contract.buildPhysicalMappingLinearScan(allocator, compiled, register_specs);
    defer runtime_contract.deinitPhysicalMapping(allocator, &physical);

    const invalid_physical = std.math.maxInt(u16);
    for (0..register_count) |reg_idx| {
        const slot_idx = register_to_slot_map[reg_idx];
        if (slot_idx == invalid_slot) continue;
        const physical_id_u16 = physical.register_to_physical[reg_idx];
        if (physical_id_u16 == invalid_physical) continue;
        const physical_id: usize = physical_id_u16;
        const width = physical.physical_specs[physical_id].size;
        if (width == 0) return error.InvalidRegisterSpecSize;
        const slot_usize: usize = @intCast(slot_idx);
        if (slot_width_hints[slot_usize] == 0) {
            slot_width_hints[slot_usize] = width;
        } else if (slot_width_hints[slot_usize] != width) {
            return error.InvalidRegisterSpecSize;
        }
    }
    for (slot_width_hints) |width| {
        if (width == 0) return error.InvalidRegisterSpecSize;
    }
    return slot_width_hints;
}

pub fn validateCompiledLayerPlanForCuda(
    compiled: *const runtime_contract.CompiledPlan,
    layer_idx: usize,
    kind: op_types.BlockKind,
    adapter_table: anytype,
) !void {
    runtime_contract.validateExecutionPlanForBlockKind(&compiled.plan, kind) catch |err| {
        log.warn("inference", "CUDA compiled layer plan fails block-kind validation", .{
            .layer = layer_idx,
            .kind = @intFromEnum(kind),
            .reason = @errorName(err),
        });
        return error.UnsupportedModel;
    };
    if (runtime_contract.firstUnsupportedInstructionOpcode(&compiled.plan, adapter_table)) |unsupported| {
        log.warn("inference", "CUDA compiled layer plan contains unsupported opcode", .{
            .layer = layer_idx,
            .kind = @intFromEnum(kind),
            .op_index = unsupported.instruction_index,
            .opcode = @intFromEnum(unsupported.opcode),
        });
        return error.UnsupportedModel;
    }
}

/// Describes a CPU source layer whose KV cache is replicated to a mirror
/// entry on the GPU. Used when cpu_gpu topology places KV-shared source
/// layers on CPU while consumer layers run on GPU.
pub const ReplicatedKvSource = struct {
    /// Global (model-wide) layer index of the CPU source layer.
    global_layer_idx: usize,
    /// KV dimension (= n_kv_heads * head_dim) for this source.
    kv_dim: usize,
    /// Index into per-slot kv[] array for the mirror entry on GPU.
    mirror_kv_index: usize,
};

/// GPU-side mirror buffers for a replicated CPU KV source layer.
pub const MirrorKvBuffers = struct {
    k: compute.cuda.Buffer,
    v: compute.cuda.Buffer,
    k_scale: compute.cuda.Buffer = .{ .pointer = 0, .size = 0 },
    v_scale: compute.cuda.Buffer = .{ .pointer = 0, .size = 0 },
    capacity: usize,
};

pub const BlockRuntime = struct {
    blocks: []BlockRuntimeLayer,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    attention_block_count: usize,
    shortconv_block_count: usize,
    gated_delta_block_count: usize,
    q_norm_blocks: usize,
    k_norm_blocks: usize,
    linear_weight_bytes: usize,
    norm_weight_bytes: usize,
    kv_cache_bytes: usize,
    shortconv_state_bytes: usize,
    gated_delta_state_bytes: usize,
    max_shortconv_dim: usize,
    max_gdelta_proj: usize,
    /// CPU source layers whose KV is replicated to GPU mirror entries.
    replicated_kv_sources: []ReplicatedKvSource = &.{},
    /// GPU-side mirror KV buffers for slot 0. loadKvSlot syncs these from
    /// slot_kv_states for the active slot. Indexed by mirror offset
    /// (mirror_kv_index - attention_block_count).
    mirror_kv: []MirrorKvBuffers = &.{},

    pub fn init(
        allocator: std.mem.Allocator,
        device: *compute.cuda.Device,
        loaded: *const LoadedModel,
        max_seq_len: usize,
        kv_init_tokens: usize,
        gated_delta_ssm_i8_state: bool,
        adapter_table: anytype,
        kv_cache_dtype: KvCacheDtype,
    ) !BlockRuntime {
        return block_init.init(
            allocator,
            device,
            loaded,
            max_seq_len,
            kv_init_tokens,
            gated_delta_ssm_i8_state,
            adapter_table,
            kv_cache_dtype,
        );
    }

    /// Initialize a BlockRuntime for a contiguous range of decoder layers
    /// [layer_start, layer_end). Used by pipeline parallel to split layers
    /// across devices.
    pub fn initRange(
        allocator: std.mem.Allocator,
        device: *compute.cuda.Device,
        loaded: *const LoadedModel,
        max_seq_len: usize,
        kv_init_tokens: usize,
        layer_start: usize,
        layer_end: usize,
        gated_delta_ssm_i8_state: bool,
        adapter_table: anytype,
        kv_cache_dtype: KvCacheDtype,
    ) !BlockRuntime {
        return block_init.initRange(
            allocator,
            device,
            loaded,
            max_seq_len,
            kv_init_tokens,
            layer_start,
            layer_end,
            gated_delta_ssm_i8_state,
            adapter_table,
            kv_cache_dtype,
        );
    }

    pub fn deinit(self: *BlockRuntime, allocator: std.mem.Allocator, device: *compute.cuda.Device) void {
        for (self.mirror_kv) |*mk| {
            if (mk.v_scale.pointer != 0) mk.v_scale.deinit(device);
            if (mk.k_scale.pointer != 0) mk.k_scale.deinit(device);
            mk.v.deinit(device);
            mk.k.deinit(device);
        }
        if (self.mirror_kv.len > 0) allocator.free(self.mirror_kv);
        if (self.replicated_kv_sources.len > 0) allocator.free(self.replicated_kv_sources);
        for (self.blocks) |*block| block.deinit(allocator, device);
        allocator.free(self.blocks);
    }

    pub fn maxDff(self: *const BlockRuntime) usize {
        var max_dff: usize = 0;
        for (self.blocks) |layer| {
            if (layer.attention_binding) |block| {
                if (block.d_ff > max_dff) max_dff = block.d_ff;
            }
            if (layer.shortconv_binding) |block| {
                if (block.d_ff > max_dff) max_dff = block.d_ff;
            }
            if (layer.gated_delta_binding) |block| {
                if (block.d_ff > max_dff) max_dff = block.d_ff;
            }
        }
        return max_dff;
    }

    pub fn maxAttn(self: *const BlockRuntime) usize {
        var max_attn: usize = 0;
        for (self.blocks) |layer| {
            if (layer.attention_binding) |block| {
                if (block.q_projection_dim > max_attn) max_attn = block.q_projection_dim;
            }
        }
        return if (max_attn > 0) max_attn else self.n_heads * self.head_dim;
    }

    pub fn maxKv(self: *const BlockRuntime) usize {
        var max_kv: usize = 0;
        for (self.blocks) |layer| {
            if (layer.attention_binding) |block| {
                if (block.kv_dim > max_kv) max_kv = block.kv_dim;
            }
        }
        return if (max_kv > 0) max_kv else self.n_kv_heads * self.head_dim;
    }

    pub fn maxShortConvDim(self: *const BlockRuntime) usize {
        return self.max_shortconv_dim;
    }

    pub fn maxGatedDeltaProj(self: *const BlockRuntime) usize {
        return if (self.max_gdelta_proj > 0) self.max_gdelta_proj else 1;
    }
};

pub const KvRuntimeState = extern struct {
    runtime_kind: u8,
    _pad: [7]u8 = [_]u8{0} ** 7,
    block_runtime: *BlockRuntime,
    slot_index: usize,
};

pub const RecurrentRuntimeState = extern struct {
    runtime_kind: u8,
    _pad: [7]u8 = [_]u8{0} ** 7,
    block_runtime: *BlockRuntime,
    slot_index: usize,
};

pub const ShortConvRuntimeState = RecurrentRuntimeState;
pub const MambaRuntimeState = RecurrentRuntimeState;
pub const GatedDeltaRuntimeState = RecurrentRuntimeState;

/// Per-row batch info for batched decode (N tokens at different positions/slots).
/// Null for single-token decode and prefill.
pub const BatchDecodeInfo = struct {
    slot_indices: []const usize,
    positions: []const usize,
    seq_lens: []const u32,
    attn_ptrs_row_stride: usize,
    attn_key_cache_ptrs_table_dev: *const compute.cuda.Buffer,
    attn_value_cache_ptrs_table_dev: *const compute.cuda.Buffer,
    gd_ptrs_row_stride: usize,
    gd_conv_state_ptrs_table_dev: *const compute.cuda.Buffer,
    gd_ssm_state_ptrs_table_dev: *const compute.cuda.Buffer,
    gd_conv_ring_heads_table_dev: *const compute.cuda.Buffer,
    attn_k_scale_ptrs_table_dev: *const compute.cuda.Buffer,
    attn_v_scale_ptrs_table_dev: *const compute.cuda.Buffer,
    attn_layer_index: usize,
    gd_layer_index: usize,
    sc_layer_index: usize,
};
