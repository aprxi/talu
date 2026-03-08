//! Transformer block execution.
//!
//! Executes transformer blocks using LayerOp bytecode from compute graphs.
//! Handles attention, FFN, and residual connections for each layer.

const std = @import("std");
const layer_ops = @import("../../../../models/layer_ops.zig");
const plan_compiler = @import("../../../../models/plan/compiler.zig");
const tensor = @import("../../../../tensor.zig");
const compute = @import("../../../../compute/root.zig");
const error_context = @import("../../../../error_context.zig");
const log = @import("../../../../log.zig");
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
const norm_kernel = @import("../kernels/norm.zig");
const mla_kernel = @import("../kernels/mla_attention.zig");
const ffn_kernel = @import("../kernels/ffn.zig");
const moe_kernel = @import("../kernels/moe.zig");
const mamba_kernel = @import("../kernels/mamba.zig");
const gated_delta_kernel = @import("../kernels/gated_delta.zig");
const shortconv_kernel = @import("../kernels/shortconv.zig");
const vision_adapters = @import("../../../vision_program_adapters.zig");
const state_bindings = @import("../state_bindings.zig");
const trace = @import("../../../../xray/root.zig").trace;

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
const NormKernelBinding = *const norm_kernel.NormKernel;
const AttentionKernelBinding = *const attn_kernel.MultiHeadAttention;
const MlaAttentionKernelBinding = *const mla_kernel.MLAttention;
const SwiGLUKernelBinding = *const ffn_kernel.SwiGLU;
const MoeKernelBinding = *const moe_kernel.MoEFFN;
const MambaKernelBinding = *const mamba_kernel.MambaKernel;
const GatedDeltaKernelBinding = *const gated_delta_kernel.GatedDeltaKernel;
const ShortConvKernelBinding = *const shortconv_kernel.ShortConvKernel;
fn zeroTensor() Tensor {
    return .{
        .dtype = .f32,
        .n_dims = 0,
        .shape = [_]i64{0} ** tensor.MAX_NDIM,
        .data_ptr = null,
        .data_size = 0,
    };
}
const missing_weight_tensor: Tensor = zeroTensor();
var missing_optional_bias_value: f32 = 0.0;
var missing_optional_scale_value: u8 = 0;

const addIntoScaled = cpu_forward.addIntoScaled;
const copyTensor = cpu_forward.copyTensor;
// Optional dispatch observability. Keep disabled by default so production
// execution adds zero atomic overhead in the token loop.
const enable_dispatch_observability: bool = false;
var layer_program_dispatch_counters = runtime_contract.DispatchCounters{};

const AttentionRuntimeMetadata = struct {
    d_model: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    scale: f32,
    qk_norm_weight_offset: f32,
    sliding_window: usize,
    is_causal: bool,
    layer_idx: u16,
    rope: @TypeOf(@as(attn_kernel.MultiHeadAttention, undefined).rope),
    runtime_rope: @TypeOf(@as(attn_kernel.MultiHeadAttention, undefined).runtime_rope),
    position_delta: isize,
    rope_interleaved: bool,
    norm_eps: f32,
    allocator: std.mem.Allocator,
    matmul_qkv: @TypeOf(@as(attn_kernel.MultiHeadAttention, undefined).matmul_qkv),
    matmul_k: @TypeOf(@as(attn_kernel.MultiHeadAttention, undefined).matmul_k),
    matmul_v: @TypeOf(@as(attn_kernel.MultiHeadAttention, undefined).matmul_v),
    matmul_qkv_fused: @TypeOf(@as(attn_kernel.MultiHeadAttention, undefined).matmul_qkv_fused),
    matmul_o: @TypeOf(@as(attn_kernel.MultiHeadAttention, undefined).matmul_o),
    kernel_name_qkv: @TypeOf(@as(attn_kernel.MultiHeadAttention, undefined).kernel_name_qkv),
    kernel_name_k: @TypeOf(@as(attn_kernel.MultiHeadAttention, undefined).kernel_name_k),
    kernel_name_v: @TypeOf(@as(attn_kernel.MultiHeadAttention, undefined).kernel_name_v),
    kernel_name_qkv_fused: @TypeOf(@as(attn_kernel.MultiHeadAttention, undefined).kernel_name_qkv_fused),
    kernel_name_o: @TypeOf(@as(attn_kernel.MultiHeadAttention, undefined).kernel_name_o),
    flash_attention_fn: @TypeOf(@as(attn_kernel.MultiHeadAttention, undefined).flash_attention_fn),
};

const NormRuntimeKind = enum {
    rms,
    layer,
};

const NormRuntimeMetadata = struct {
    kind: NormRuntimeKind,
    dim: usize,
    eps: f32,
    weight_offset: f32,
    layer_idx: u16,
    trace_point: @TypeOf(@as(norm_kernel.RMSNorm, undefined).trace_point),
    has_bias: bool,
};

const MlaRuntimeMetadata = struct {
    d_model: usize,
    n_heads: usize,
    max_seq_len: usize,
    config: @TypeOf(@as(mla_kernel.MLAttention, undefined).config),
    allocator: std.mem.Allocator,
    rope: @TypeOf(@as(mla_kernel.MLAttention, undefined).rope),
    norm_eps: f32,
    scale: f32,
    matmul_fn: @TypeOf(@as(mla_kernel.MLAttention, undefined).matmul_fn),
    layer_idx: u16,
};

const SwiGluRuntimeMetadata = struct {
    d_model: usize,
    d_ff: usize,
    use_gelu: bool,
    use_swiglu_variant: bool,
    layer_idx: u16,
    fused_gate_up_layout: @TypeOf(@as(ffn_kernel.SwiGLU, undefined).fused_gate_up_layout),
    allocator: std.mem.Allocator,
    matmul_gate: @TypeOf(@as(ffn_kernel.SwiGLU, undefined).matmul_gate),
    matmul_gate_up: @TypeOf(@as(ffn_kernel.SwiGLU, undefined).matmul_gate_up),
    matmul_down: @TypeOf(@as(ffn_kernel.SwiGLU, undefined).matmul_down),
    kernel_name_gate: @TypeOf(@as(ffn_kernel.SwiGLU, undefined).kernel_name_gate),
    kernel_name_gate_up: @TypeOf(@as(ffn_kernel.SwiGLU, undefined).kernel_name_gate_up),
    kernel_name_down: @TypeOf(@as(ffn_kernel.SwiGLU, undefined).kernel_name_down),
};

const MoeRuntimeMetadata = struct {
    allocator: std.mem.Allocator,
    d_model: usize,
    d_ff: usize,
    num_experts: usize,
    experts_per_token: usize,
    use_mxfp4: bool,
    use_swiglu_variant: bool,
    use_transposed_weights: bool,
    layer_idx: u16,
    kernel_name: ?[]const u8,
    has_gate_proj: bool,
    has_up_proj: bool,
    has_gate_up_proj: bool,
    gate_scales_len: usize,
    up_scales_len: usize,
    gate_up_scales_len: usize,
    down_scales_len: usize,
    gate_bias_len: usize,
    up_bias_len: usize,
    gate_up_bias_len: usize,
    down_bias_len: usize,
};

const MambaRuntimeMetadata = struct {
    mamba_config: @TypeOf(@as(mamba_kernel.MambaKernel, undefined).config),
    matmul_in_proj: @TypeOf(@as(mamba_kernel.MambaKernel, undefined).matmul_in_proj),
    matmul_out_proj: @TypeOf(@as(mamba_kernel.MambaKernel, undefined).matmul_out_proj),
    ssm_scan: ?@TypeOf(@as(mamba_kernel.MambaKernel, undefined).ssm_scan) = null,
    layer_idx: u16,
};

const GatedDeltaRuntimeMetadata = struct {
    config: @TypeOf(@as(gated_delta_kernel.GatedDeltaKernel, undefined).config),
    matmul_in_proj: @TypeOf(@as(gated_delta_kernel.GatedDeltaKernel, undefined).matmul_in_proj),
    matmul_out_proj: @TypeOf(@as(gated_delta_kernel.GatedDeltaKernel, undefined).matmul_out_proj),
    // Flattened runtime handle for pre-transposed conv weights (time-major).
    // When present, the execute path reuses this exact view and preserves the
    // optimized conv1d path during warmup and batched dispatch.
    conv_weight_time_major: ?Tensor = null,
    layer_idx: u16,
};

const ShortConvRuntimeMetadata = struct {
    config: @TypeOf(@as(shortconv_kernel.ShortConvKernel, undefined).config),
    matmul_in_proj: @TypeOf(@as(shortconv_kernel.ShortConvKernel, undefined).matmul_in_proj),
    matmul_out_proj: @TypeOf(@as(shortconv_kernel.ShortConvKernel, undefined).matmul_out_proj),
    matmul_in_proj_name: @TypeOf(@as(shortconv_kernel.ShortConvKernel, undefined).matmul_in_proj_name),
    matmul_out_proj_name: @TypeOf(@as(shortconv_kernel.ShortConvKernel, undefined).matmul_out_proj_name),
    // Flattened runtime handle for pre-transposed conv weights (time-major).
    // When present, adapter slot 1 resolves to this tensor and keeps SIMD path hot.
    conv_weight_time_major: ?Tensor = null,
    layer_idx: u16,
};

const TmpRegisterLayout = struct {
    map: []u8,
    slot_count: usize,
    slot_width_hints: []usize,
    slot_active: []bool,
};

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

fn maxInstructionHandleCapacity(plan: *const runtime_contract.ExecutionPlan) usize {
    var max_handles: usize = 0;
    for (plan.instructions) |insn| {
        const handle_count = insn.inputs.len + insn.outputs.len + insn.weights.len;
        if (handle_count > max_handles) max_handles = handle_count;
    }
    return max_handles;
}

fn buildTmpRegisterScratchMap(
    allocator: std.mem.Allocator,
    compiled: *const runtime_contract.CompiledPlan,
) !TmpRegisterLayout {
    const register_count: usize = compiled.plan.register_count;
    const map = try allocator.alloc(u8, register_count);
    @memset(map, 0);
    errdefer allocator.free(map);

    if (compiled.register_buffer_specs.len != register_count) return error.InvalidRegisterSpecCount;
    const layer_tmp_width: usize = if (compiled.register_buffer_specs.len > 0)
        compiled.register_buffer_specs[0].size
    else
        0;

    if (register_count <= 1) {
        if (register_count == 0) {
            return .{ .map = map, .slot_count = 0, .slot_width_hints = &.{}, .slot_active = &.{} };
        }
        if (layer_tmp_width == 0) return error.InvalidRegisterSpecSize;
        const slot_width_hints = try allocator.alloc(usize, 1);
        errdefer allocator.free(slot_width_hints);
        slot_width_hints[0] = layer_tmp_width;
        const slot_active = try allocator.alloc(bool, 1);
        errdefer allocator.free(slot_active);
        slot_active[0] = true;
        return .{ .map = map, .slot_count = 1, .slot_width_hints = slot_width_hints, .slot_active = slot_active };
    }

    const specs = try allocator.alloc(runtime_contract.RegisterBufferSpec, register_count);
    defer allocator.free(specs);
    // Register 0 (residual) uses the model output buffer, not scratch.
    // Mark exempt via size=0 so the allocator skips it.
    specs[0] = .{ .size = 0, .@"align" = 0 };
    // Plan specs already contain the model-dimension floor applied at compile
    // time (via CompileOptions.size_floor). Backends consume specs exactly.
    for (specs[1..], 1..) |*spec, idx| {
        const plan_spec = compiled.register_buffer_specs[idx];
        spec.* = .{
            .size = plan_spec.size,
            .@"align" = plan_spec.@"align",
            .dtype = plan_spec.dtype,
            .layout = plan_spec.layout,
        };
    }

    var physical_mapping = try runtime_contract.buildPhysicalMappingLinearScan(allocator, compiled, specs);
    defer runtime_contract.deinitPhysicalMapping(allocator, &physical_mapping);
    if (physical_mapping.physical_count == 0) {
        return .{ .map = map, .slot_count = 0, .slot_width_hints = &.{}, .slot_active = &.{} };
    }

    const physical_to_tmp_slot = try allocator.alloc(u8, physical_mapping.physical_count);
    defer allocator.free(physical_to_tmp_slot);
    @memset(physical_to_tmp_slot, std.math.maxInt(u8));

    // Allocate slot arrays sized to max possible slots (1 + physical_count).
    const max_slots = 1 + physical_mapping.physical_count;
    const slot_width_hints = try allocator.alloc(usize, max_slots);
    errdefer allocator.free(slot_width_hints);
    @memset(slot_width_hints, 0);
    const slot_active = try allocator.alloc(bool, max_slots);
    errdefer allocator.free(slot_active);
    @memset(slot_active, false);
    if (layer_tmp_width == 0) return error.InvalidRegisterSpecSize;
    slot_width_hints[0] = layer_tmp_width;
    slot_active[0] = true;

    var next_tmp_slot: usize = 1;
    for (0..register_count) |reg_idx| {
        const physical_id_u16 = physical_mapping.register_to_physical[reg_idx];
        if (physical_id_u16 == std.math.maxInt(u16)) continue;
        const physical_id: usize = physical_id_u16;
        if (physical_id >= physical_to_tmp_slot.len) return error.InvalidState;
        if (physical_to_tmp_slot[physical_id] == std.math.maxInt(u8)) {
            physical_to_tmp_slot[physical_id] = @intCast(next_tmp_slot);
            next_tmp_slot += 1;
        }
        const mapped_slot = physical_to_tmp_slot[physical_id];
        map[reg_idx] = mapped_slot;
        slot_active[mapped_slot] = true;
        if (slot_width_hints[mapped_slot] == 0) {
            slot_width_hints[mapped_slot] = physical_mapping.physical_specs[physical_id].size;
        } else if (slot_width_hints[mapped_slot] != physical_mapping.physical_specs[physical_id].size) {
            return error.InvalidRegisterSpecSize;
        }
    }

    return .{ .map = map, .slot_count = next_tmp_slot, .slot_width_hints = slot_width_hints, .slot_active = slot_active };
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
    /// Per-instruction typed kernel refs for macro adapters (load-time resolved).
    instruction_norm_refs: []?NormKernelBinding = &.{},
    instruction_attention_bindings: []?AttentionKernelBinding = &.{},
    instruction_mla_attention_refs: []?MlaAttentionKernelBinding = &.{},
    instruction_swiglu_bindings: []?SwiGLUKernelBinding = &.{},
    instruction_moe_bindings: []?MoeKernelBinding = &.{},
    instruction_mamba_bindings: []?MambaKernelBinding = &.{},
    instruction_gated_delta_bindings: []?GatedDeltaKernelBinding = &.{},
    instruction_shortconv_bindings: []?ShortConvKernelBinding = &.{},
    /// Execute-path non-weight metadata derived at load time.
    instruction_norm_runtime_metadata: []?NormRuntimeMetadata = &.{},
    instruction_attention_runtime_metadata: []?AttentionRuntimeMetadata = &.{},
    instruction_mla_runtime_metadata: []?MlaRuntimeMetadata = &.{},
    instruction_swiglu_runtime_metadata: []?SwiGluRuntimeMetadata = &.{},
    instruction_moe_runtime_metadata: []?MoeRuntimeMetadata = &.{},
    instruction_mamba_runtime_metadata: []?MambaRuntimeMetadata = &.{},
    instruction_gated_delta_runtime_metadata: []?GatedDeltaRuntimeMetadata = &.{},
    instruction_shortconv_runtime_metadata: []?ShortConvRuntimeMetadata = &.{},
    /// Per-instruction prefix offsets into `instruction_weight_ptrs`.
    instruction_weight_offsets: []u32 = &.{},
    /// Flattened per-instruction weight handles resolved at load time.
    instruction_weight_ptrs: []?*anyopaque = &.{},
    /// Logical register index -> physical scratch tmp slot (dynamically sized to register_count).
    /// Index 0 (residual) always remains identity (maps to output buffer, not scratch).
    /// Indices 1+ are mapped through liveness analysis.
    tmp_register_to_scratch_idx: []const u8 = &.{},
    /// Physical slot width hints derived from compiled-plan liveness allocation.
    tmp_slot_width_hints: []usize = &.{},
    /// Physical scratch slot activity mask.
    tmp_slot_active: []bool = &.{},
    /// Maximum tensor handles required by any instruction in the compiled plan.
    instruction_handle_capacity: usize = 0,

    pub fn initWithProgram(
        allocator: std.mem.Allocator,
        block: *const cpu_forward.TransformerBlock,
        block_idx: usize,
        hidden_size: usize,
        program: []const LayerOp,
        mode: runtime_contract.ExecutionMode,
    ) !Block {
        return initWithProgramOptions(
            allocator,
            block,
            block_idx,
            hidden_size,
            program,
            mode,
            .{},
        );
    }

    pub fn initWithProgramOptions(
        allocator: std.mem.Allocator,
        block: *const cpu_forward.TransformerBlock,
        block_idx: usize,
        hidden_size: usize,
        program: []const LayerOp,
        mode: runtime_contract.ExecutionMode,
        compile_options: plan_compiler.CompileOptions,
    ) !Block {
        const width_hint = blockTempWidthHint(block, hidden_size);
        var resolved_options = compile_options;
        resolved_options.size_floor = @max(width_hint, resolved_options.size_floor);
        var compiled_plan = try plan_compiler.compileLayerProgram(allocator, program, mode, resolved_options);
        errdefer plan_compiler.deinitCompiledPlan(allocator, &compiled_plan);
        const tmp_layout = try buildTmpRegisterScratchMap(allocator, &compiled_plan);
        const typed_kernel_refs = try buildTypedInstructionKernelRefs(
            allocator,
            block,
            block_idx,
            &compiled_plan,
        );
        const runtime_metadata = try buildRuntimeMetadata(
            allocator,
            typed_kernel_refs,
            &compiled_plan.plan,
        );
        const weight_table = try buildInstructionWeightTable(
            allocator,
            block,
            block_idx,
            &compiled_plan,
            typed_kernel_refs,
            &runtime_metadata,
        );
        errdefer {
            allocator.free(weight_table.offsets);
            allocator.free(weight_table.ptrs);
            runtime_metadata.deinit(allocator);
            typed_kernel_refs.deinit(allocator);
        }
        return .{
            .compiled_plan = compiled_plan,
            .block = block,
            .block_idx = block_idx,
            .hidden_size = hidden_size,
            .instruction_norm_refs = typed_kernel_refs.norm,
            .instruction_attention_bindings = typed_kernel_refs.attention,
            .instruction_mla_attention_refs = typed_kernel_refs.mla_attention,
            .instruction_swiglu_bindings = typed_kernel_refs.swiglu,
            .instruction_moe_bindings = typed_kernel_refs.moe,
            .instruction_mamba_bindings = typed_kernel_refs.mamba,
            .instruction_gated_delta_bindings = typed_kernel_refs.gated_delta,
            .instruction_shortconv_bindings = typed_kernel_refs.shortconv,
            .instruction_norm_runtime_metadata = runtime_metadata.norm,
            .instruction_attention_runtime_metadata = runtime_metadata.attention,
            .instruction_mla_runtime_metadata = runtime_metadata.mla,
            .instruction_swiglu_runtime_metadata = runtime_metadata.swiglu,
            .instruction_moe_runtime_metadata = runtime_metadata.moe,
            .instruction_mamba_runtime_metadata = runtime_metadata.mamba,
            .instruction_gated_delta_runtime_metadata = runtime_metadata.gated_delta,
            .instruction_shortconv_runtime_metadata = runtime_metadata.shortconv,
            .instruction_weight_offsets = weight_table.offsets,
            .instruction_weight_ptrs = weight_table.ptrs,
            .tmp_register_to_scratch_idx = tmp_layout.map,
            .tmp_slot_width_hints = tmp_layout.slot_width_hints,
            .tmp_slot_active = tmp_layout.slot_active,
            .instruction_handle_capacity = maxInstructionHandleCapacity(&compiled_plan.plan),
        };
    }

    pub fn deinit(self: *Block, allocator: std.mem.Allocator) void {
        if (self.tmp_register_to_scratch_idx.len > 0) {
            allocator.free(self.tmp_register_to_scratch_idx);
        }
        if (self.tmp_slot_width_hints.len > 0) allocator.free(self.tmp_slot_width_hints);
        if (self.tmp_slot_active.len > 0) allocator.free(self.tmp_slot_active);
        if (self.instruction_weight_offsets.len > 0) allocator.free(self.instruction_weight_offsets);
        if (self.instruction_weight_ptrs.len > 0) allocator.free(self.instruction_weight_ptrs);
        if (self.instruction_norm_refs.len > 0) allocator.free(self.instruction_norm_refs);
        if (self.instruction_attention_bindings.len > 0) allocator.free(self.instruction_attention_bindings);
        if (self.instruction_mla_attention_refs.len > 0) allocator.free(self.instruction_mla_attention_refs);
        if (self.instruction_swiglu_bindings.len > 0) allocator.free(self.instruction_swiglu_bindings);
        if (self.instruction_moe_bindings.len > 0) allocator.free(self.instruction_moe_bindings);
        if (self.instruction_mamba_bindings.len > 0) allocator.free(self.instruction_mamba_bindings);
        if (self.instruction_gated_delta_bindings.len > 0) allocator.free(self.instruction_gated_delta_bindings);
        if (self.instruction_shortconv_bindings.len > 0) allocator.free(self.instruction_shortconv_bindings);
        if (self.instruction_norm_runtime_metadata.len > 0) allocator.free(self.instruction_norm_runtime_metadata);
        if (self.instruction_attention_runtime_metadata.len > 0) allocator.free(self.instruction_attention_runtime_metadata);
        if (self.instruction_mla_runtime_metadata.len > 0) allocator.free(self.instruction_mla_runtime_metadata);
        if (self.instruction_swiglu_runtime_metadata.len > 0) allocator.free(self.instruction_swiglu_runtime_metadata);
        if (self.instruction_moe_runtime_metadata.len > 0) allocator.free(self.instruction_moe_runtime_metadata);
        if (self.instruction_mamba_runtime_metadata.len > 0) allocator.free(self.instruction_mamba_runtime_metadata);
        if (self.instruction_gated_delta_runtime_metadata.len > 0) allocator.free(self.instruction_gated_delta_runtime_metadata);
        if (self.instruction_shortconv_runtime_metadata.len > 0) allocator.free(self.instruction_shortconv_runtime_metadata);
        plan_compiler.deinitCompiledPlan(allocator, &self.compiled_plan);
        self.* = undefined;
    }

    /// Register this block's scratch layout requirements with the shared
    /// scratch allocator. Callers can aggregate all layer layouts up front
    /// before taking long-lived views into `scratch.tmp[0]`.
    pub fn registerScratchLayout(self: *const Block, scratch: *ScratchBuffer) !void {
        try scratch.registerTmpLayout(self.tmp_slot_width_hints, self.tmp_slot_active);
    }

    fn instructionKernelIdFromWeightBindings(
        compiled_plan: *const runtime_contract.CompiledPlan,
        op_index: usize,
        insn: *const runtime_contract.Instruction,
    ) !u32 {
        return runtime_contract.instructionKernelBindingId(compiled_plan, op_index, insn.opcode);
    }

    const InstructionWeightTable = struct {
        offsets: []u32,
        ptrs: []?*anyopaque,
    };

    const TypedInstructionKernelRefs = struct {
        norm: []?NormKernelBinding,
        attention: []?AttentionKernelBinding,
        mla_attention: []?MlaAttentionKernelBinding,
        swiglu: []?SwiGLUKernelBinding,
        moe: []?MoeKernelBinding,
        mamba: []?MambaKernelBinding,
        gated_delta: []?GatedDeltaKernelBinding,
        shortconv: []?ShortConvKernelBinding,

        fn deinit(self: TypedInstructionKernelRefs, allocator: std.mem.Allocator) void {
            allocator.free(self.norm);
            allocator.free(self.attention);
            allocator.free(self.mla_attention);
            allocator.free(self.swiglu);
            allocator.free(self.moe);
            allocator.free(self.mamba);
            allocator.free(self.gated_delta);
            allocator.free(self.shortconv);
        }
    };

    const RuntimeMetadata = struct {
        norm: []?NormRuntimeMetadata,
        attention: []?AttentionRuntimeMetadata,
        mla: []?MlaRuntimeMetadata,
        swiglu: []?SwiGluRuntimeMetadata,
        moe: []?MoeRuntimeMetadata,
        mamba: []?MambaRuntimeMetadata,
        gated_delta: []?GatedDeltaRuntimeMetadata,
        shortconv: []?ShortConvRuntimeMetadata,

        fn deinit(self: RuntimeMetadata, allocator: std.mem.Allocator) void {
            allocator.free(self.norm);
            allocator.free(self.attention);
            allocator.free(self.mla);
            allocator.free(self.swiglu);
            allocator.free(self.moe);
            allocator.free(self.mamba);
            allocator.free(self.gated_delta);
            allocator.free(self.shortconv);
        }
    };

    fn buildRuntimeMetadata(
        allocator: std.mem.Allocator,
        typed_kernel_refs: TypedInstructionKernelRefs,
        plan: *const runtime_contract.ExecutionPlan,
    ) !RuntimeMetadata {
        const len = typed_kernel_refs.norm.len;
        const norm = try allocator.alloc(?NormRuntimeMetadata, len);
        errdefer allocator.free(norm);
        const attention = try allocator.alloc(?AttentionRuntimeMetadata, len);
        errdefer allocator.free(attention);
        const mla = try allocator.alloc(?MlaRuntimeMetadata, len);
        errdefer allocator.free(mla);
        const swiglu = try allocator.alloc(?SwiGluRuntimeMetadata, len);
        errdefer allocator.free(swiglu);
        const moe = try allocator.alloc(?MoeRuntimeMetadata, len);
        errdefer allocator.free(moe);
        const mamba = try allocator.alloc(?MambaRuntimeMetadata, len);
        errdefer allocator.free(mamba);
        const gated_delta = try allocator.alloc(?GatedDeltaRuntimeMetadata, len);
        errdefer allocator.free(gated_delta);
        const shortconv = try allocator.alloc(?ShortConvRuntimeMetadata, len);
        errdefer allocator.free(shortconv);
        @memset(norm, null);
        @memset(attention, null);
        @memset(mla, null);
        @memset(swiglu, null);
        @memset(moe, null);
        @memset(mamba, null);
        @memset(gated_delta, null);
        @memset(shortconv, null);

        for (0..len) |idx| {
            if (typed_kernel_refs.norm[idx]) |binding| {
                norm[idx] = switch (binding.*) {
                    .rms => |rms| .{
                        .kind = .rms,
                        .dim = rms.dim,
                        .eps = rms.eps,
                        .weight_offset = rms.weight_offset,
                        .layer_idx = rms.layer_idx,
                        .trace_point = rms.trace_point,
                        .has_bias = false,
                    },
                    .layer => |layer| .{
                        .kind = .layer,
                        .dim = layer.dim,
                        .eps = layer.eps,
                        .weight_offset = 0.0,
                        .layer_idx = layer.layer_idx,
                        .trace_point = layer.trace_point,
                        .has_bias = layer.bias != null,
                    },
                };
            }
            if (typed_kernel_refs.attention[idx]) |binding| {
                attention[idx] = .{
                    .d_model = binding.d_model,
                    .n_heads = binding.n_heads,
                    .n_kv_heads = binding.n_kv_heads,
                    .head_dim = binding.head_dim,
                    .max_seq_len = binding.max_seq_len,
                    .scale = binding.scale,
                    .qk_norm_weight_offset = binding.qk_norm_weight_offset,
                    .sliding_window = binding.sliding_window,
                    .is_causal = binding.is_causal,
                    .layer_idx = binding.layer_idx,
                    .rope = binding.rope,
                    .runtime_rope = binding.runtime_rope,
                    .position_delta = binding.position_delta,
                    .rope_interleaved = binding.rope_interleaved,
                    .norm_eps = binding.norm_eps,
                    .allocator = binding.allocator,
                    .matmul_qkv = binding.matmul_qkv,
                    .matmul_k = binding.matmul_k,
                    .matmul_v = binding.matmul_v,
                    .matmul_qkv_fused = binding.matmul_qkv_fused,
                    .matmul_o = binding.matmul_o,
                    .kernel_name_qkv = binding.kernel_name_qkv,
                    .kernel_name_k = binding.kernel_name_k,
                    .kernel_name_v = binding.kernel_name_v,
                    .kernel_name_qkv_fused = binding.kernel_name_qkv_fused,
                    .kernel_name_o = binding.kernel_name_o,
                    .flash_attention_fn = binding.flash_attention_fn,
                };
            }
            if (typed_kernel_refs.mla_attention[idx]) |binding| {
                mla[idx] = .{
                    .d_model = binding.d_model,
                    .n_heads = binding.n_heads,
                    .max_seq_len = binding.max_seq_len,
                    .config = binding.config,
                    .allocator = binding.allocator,
                    .rope = binding.rope,
                    .norm_eps = binding.norm_eps,
                    .scale = binding.scale,
                    .matmul_fn = binding.matmul_fn,
                    .layer_idx = binding.layer_idx,
                };
            }
            if (typed_kernel_refs.swiglu[idx]) |binding| {
                swiglu[idx] = .{
                    .d_model = binding.d_model,
                    .d_ff = binding.d_ff,
                    .use_gelu = binding.use_gelu,
                    .use_swiglu_variant = binding.use_swiglu_variant,
                    .layer_idx = binding.layer_idx,
                    .fused_gate_up_layout = binding.fused_gate_up_layout,
                    .allocator = binding.allocator,
                    .matmul_gate = binding.matmul_gate,
                    .matmul_gate_up = binding.matmul_gate_up,
                    .matmul_down = binding.matmul_down,
                    .kernel_name_gate = binding.kernel_name_gate,
                    .kernel_name_gate_up = binding.kernel_name_gate_up,
                    .kernel_name_down = binding.kernel_name_down,
                };
            }
            if (typed_kernel_refs.moe[idx]) |binding| {
                if (binding.experts.len != 1) return error.UnsupportedModel;
                const expert = binding.experts[0];
                const gate_scales_len = if (expert.gate_scales) |gate_scales|
                    gate_scales.len
                else if (expert.gate_up_scales) |gate_up_scales|
                    gate_up_scales.len
                else
                    0;
                const up_scales_len = if (expert.up_scales) |up_scales|
                    up_scales.len
                else if (expert.gate_up_scales) |gate_up_scales|
                    gate_up_scales.len
                else
                    0;
                const gate_bias_len = if (expert.gate_bias) |gate_bias|
                    gate_bias.len
                else if (expert.gate_up_bias) |gate_up_bias|
                    gate_up_bias.len
                else
                    0;
                const up_bias_len = if (expert.up_bias) |up_bias|
                    up_bias.len
                else if (expert.gate_up_bias) |gate_up_bias|
                    gate_up_bias.len
                else
                    0;
                moe[idx] = .{
                    .allocator = binding.allocator,
                    .d_model = binding.d_model,
                    .d_ff = binding.d_ff,
                    .num_experts = binding.num_experts,
                    .experts_per_token = binding.experts_per_token,
                    .use_mxfp4 = binding.use_mxfp4,
                    .use_swiglu_variant = binding.use_swiglu_variant,
                    .use_transposed_weights = binding.use_transposed_weights,
                    .layer_idx = binding.layer_idx,
                    .kernel_name = binding.kernel_name,
                    .has_gate_proj = expert.gate_proj != null,
                    .has_up_proj = expert.up_proj != null,
                    .has_gate_up_proj = expert.gate_up_proj != null,
                    .gate_scales_len = gate_scales_len,
                    .up_scales_len = up_scales_len,
                    .gate_up_scales_len = if (expert.gate_up_scales) |gate_up_scales| gate_up_scales.len else 0,
                    .down_scales_len = if (expert.down_scales) |down_scales| down_scales.len else 0,
                    .gate_bias_len = gate_bias_len,
                    .up_bias_len = up_bias_len,
                    .gate_up_bias_len = if (expert.gate_up_bias) |gate_up_bias| gate_up_bias.len else 0,
                    .down_bias_len = if (expert.down_bias) |down_bias| down_bias.len else 0,
                };
            }
            if (typed_kernel_refs.mamba[idx]) |binding| {
                if (plan.instructions[idx].opcode != .mamba_mixer) return error.InvalidInstructionBinding;
                mamba[idx] = .{
                    .mamba_config = binding.config,
                    .matmul_in_proj = binding.matmul_in_proj,
                    .matmul_out_proj = binding.matmul_out_proj,
                    .ssm_scan = binding.ssm_scan,
                    .layer_idx = binding.layer_idx,
                };
            }
            if (typed_kernel_refs.gated_delta[idx]) |binding| {
                if (plan.instructions[idx].opcode != .gated_delta_net) return error.InvalidInstructionBinding;
                gated_delta[idx] = .{
                    .config = binding.config,
                    .matmul_in_proj = binding.matmul_in_proj,
                    .matmul_out_proj = binding.matmul_out_proj,
                    .conv_weight_time_major = if (binding.conv_weight_transposed) |weight_t| blk: {
                        const d_conv: usize = @intCast(binding.config.d_conv);
                        if (d_conv == 0 or (weight_t.len % d_conv) != 0) break :blk null;
                        break :blk Tensor.view(
                            @ptrCast(std.mem.sliceAsBytes(weight_t).ptr),
                            &.{ d_conv, weight_t.len / d_conv },
                            .f32,
                            null,
                        );
                    } else null,
                    .layer_idx = binding.layer_idx,
                };
            }
            if (typed_kernel_refs.shortconv[idx]) |binding| {
                shortconv[idx] = .{
                    .config = binding.config,
                    .matmul_in_proj = binding.matmul_in_proj,
                    .matmul_out_proj = binding.matmul_out_proj,
                    .matmul_in_proj_name = binding.matmul_in_proj_name,
                    .matmul_out_proj_name = binding.matmul_out_proj_name,
                    .conv_weight_time_major = if (binding.conv_weight_transposed) |weight_t|
                        Tensor.view(
                            @ptrCast(std.mem.sliceAsBytes(weight_t).ptr),
                            &.{
                                @as(usize, @intCast(binding.config.d_conv)),
                                @as(usize, @intCast(binding.config.conv_dim)),
                            },
                            .f32,
                            null,
                        )
                    else
                        null,
                    .layer_idx = binding.layer_idx,
                };
            }
        }

        return .{
            .norm = norm,
            .attention = attention,
            .mla = mla,
            .swiglu = swiglu,
            .moe = moe,
            .mamba = mamba,
            .gated_delta = gated_delta,
            .shortconv = shortconv,
        };
    }

    fn buildTypedInstructionKernelRefs(
        allocator: std.mem.Allocator,
        block: *const cpu_forward.TransformerBlock,
        block_idx: usize,
        compiled_plan: *const runtime_contract.CompiledPlan,
    ) !TypedInstructionKernelRefs {
        const len = compiled_plan.plan.instructions.len;
        const norm = try allocator.alloc(?NormKernelBinding, len);
        errdefer allocator.free(norm);
        const attention = try allocator.alloc(?AttentionKernelBinding, len);
        errdefer allocator.free(attention);
        const mla_attention = try allocator.alloc(?MlaAttentionKernelBinding, len);
        errdefer allocator.free(mla_attention);
        const swiglu = try allocator.alloc(?SwiGLUKernelBinding, len);
        errdefer allocator.free(swiglu);
        const moe = try allocator.alloc(?MoeKernelBinding, len);
        errdefer allocator.free(moe);
        const mamba = try allocator.alloc(?MambaKernelBinding, len);
        errdefer allocator.free(mamba);
        const gated_delta = try allocator.alloc(?GatedDeltaKernelBinding, len);
        errdefer allocator.free(gated_delta);
        const shortconv = try allocator.alloc(?ShortConvKernelBinding, len);
        errdefer allocator.free(shortconv);

        @memset(norm, null);
        @memset(attention, null);
        @memset(mla_attention, null);
        @memset(swiglu, null);
        @memset(moe, null);
        @memset(mamba, null);
        @memset(gated_delta, null);
        @memset(shortconv, null);

        for (compiled_plan.plan.instructions, 0..) |insn, op_index| {
            if (runtime_contract.expectedKernelWeightSlots(insn.opcode).len == 0) continue;
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
            const kernel = block.kernels[kernel_idx];
            const opcode = insn.opcode;
            switch (opcode) {
                .rmsnorm => norm[op_index] = switch (kernel) {
                    .norm => |binding| binding,
                    else => return error.InvalidInstructionBinding,
                },
                .multihead_attention => attention[op_index] = switch (kernel) {
                    .attention => |binding| binding,
                    else => return error.InvalidInstructionBinding,
                },
                .mla_attention => mla_attention[op_index] = switch (kernel) {
                    .mla_attention => |binding| binding,
                    else => return error.InvalidInstructionBinding,
                },
                .swiglu => swiglu[op_index] = switch (kernel) {
                    .swiglu => |binding| binding,
                    else => return error.InvalidInstructionBinding,
                },
                .moe => moe[op_index] = switch (kernel) {
                    .moe => |binding| blk: {
                        if (binding.experts.len != 1) return error.UnsupportedModel;
                        break :blk binding;
                    },
                    else => return error.InvalidInstructionBinding,
                },
                .mamba_mixer => mamba[op_index] = switch (kernel) {
                    .mamba => |binding| binding,
                    else => return error.InvalidInstructionBinding,
                },
                .gated_delta_net => gated_delta[op_index] = switch (kernel) {
                    .gated_delta => |binding| binding,
                    else => return error.InvalidInstructionBinding,
                },
                .shortconv => shortconv[op_index] = switch (kernel) {
                    .shortconv => |binding| binding,
                    else => return error.InvalidInstructionBinding,
                },
                else => {},
            }
        }

        return .{
            .norm = norm,
            .attention = attention,
            .mla_attention = mla_attention,
            .swiglu = swiglu,
            .moe = moe,
            .mamba = mamba,
            .gated_delta = gated_delta,
            .shortconv = shortconv,
        };
    }

    fn resolveKernelWeightPtrForSlot(
        block_idx: usize,
        op_index: usize,
        opcode: runtime_contract.Opcode,
        typed_kernel_refs: TypedInstructionKernelRefs,
        slot_idx: usize,
    ) !*anyopaque {
        switch (opcode) {
            .rmsnorm => {
                const norm_binding = typed_kernel_refs.norm[op_index] orelse return error.InvalidInstructionBinding;
                return switch (slot_idx) {
                    0 => blk: {
                        const weight = switch (norm_binding.*) {
                            .rms => |rms| rms.weight,
                            .layer => |layer| layer.weight,
                        };
                        break :blk @ptrCast(@constCast(weight));
                    },
                    1 => switch (norm_binding.*) {
                        .rms => @ptrCast(@constCast(&missing_weight_tensor)),
                        .layer => |layer| if (layer.bias) |bias|
                            @ptrCast(@constCast(bias))
                        else
                            @ptrCast(@constCast(&missing_weight_tensor)),
                    },
                    else => error.InvalidWeightRefCount,
                };
            },
            .multihead_attention => {
                const attn_binding = typed_kernel_refs.attention[op_index] orelse return error.InvalidInstructionBinding;
                switch (slot_idx) {
                    0 => {
                        if (attn_binding.q_proj) |q_proj| return @ptrCast(@constCast(q_proj));
                        if (attn_binding.fused_qkv) |*fused_qkv| return @ptrCast(@constCast(fused_qkv));
                        return error.MissingWeight;
                    },
                    1 => {
                        if (attn_binding.k_proj) |k_proj| return @ptrCast(@constCast(k_proj));
                        if (attn_binding.fused_qkv != null) return @ptrCast(@constCast(&missing_weight_tensor));
                        return error.MissingWeight;
                    },
                    2 => {
                        if (attn_binding.v_proj) |v_proj| return @ptrCast(@constCast(v_proj));
                        if (attn_binding.fused_qkv != null) return @ptrCast(@constCast(&missing_weight_tensor));
                        return error.MissingWeight;
                    },
                    3 => return @ptrCast(@constCast(attn_binding.o_proj)),
                    4 => if (attn_binding.q_norm) |q_norm|
                        return @ptrCast(@constCast(q_norm))
                    else
                        return @ptrCast(@constCast(&missing_weight_tensor)),
                    5 => if (attn_binding.k_norm) |k_norm|
                        return @ptrCast(@constCast(k_norm))
                    else
                        return @ptrCast(@constCast(&missing_weight_tensor)),
                    6 => if (attn_binding.q_bias) |q_bias|
                        if (q_bias.len != 0) return @ptrCast(@constCast(q_bias.ptr)) else return @ptrCast(&missing_optional_bias_value)
                    else
                        return @ptrCast(&missing_optional_bias_value),
                    7 => if (attn_binding.k_bias) |k_bias|
                        if (k_bias.len != 0) return @ptrCast(@constCast(k_bias.ptr)) else return @ptrCast(&missing_optional_bias_value)
                    else
                        return @ptrCast(&missing_optional_bias_value),
                    8 => if (attn_binding.v_bias) |v_bias|
                        if (v_bias.len != 0) return @ptrCast(@constCast(v_bias.ptr)) else return @ptrCast(&missing_optional_bias_value)
                    else
                        return @ptrCast(&missing_optional_bias_value),
                    9 => if (attn_binding.o_bias) |o_bias|
                        if (o_bias.len != 0) return @ptrCast(@constCast(o_bias.ptr)) else return @ptrCast(&missing_optional_bias_value)
                    else
                        return @ptrCast(&missing_optional_bias_value),
                    10 => if (attn_binding.sinks) |sinks|
                        if (sinks.len != 0) return @ptrCast(@constCast(sinks.ptr)) else return @ptrCast(&missing_optional_bias_value)
                    else
                        return @ptrCast(&missing_optional_bias_value),
                    else => return error.InvalidWeightRefCount,
                }
            },
            .mla_attention => {
                const mla_binding = typed_kernel_refs.mla_attention[op_index] orelse return error.InvalidInstructionBinding;
                return switch (slot_idx) {
                    0 => @ptrCast(@constCast(mla_binding.q_a_proj)),
                    1 => @ptrCast(@constCast(mla_binding.q_a_norm)),
                    2 => @ptrCast(@constCast(mla_binding.q_b_proj)),
                    3 => @ptrCast(@constCast(mla_binding.kv_a_proj)),
                    4 => @ptrCast(@constCast(mla_binding.kv_a_norm)),
                    5 => @ptrCast(@constCast(mla_binding.kv_b_proj)),
                    6 => @ptrCast(@constCast(mla_binding.o_proj)),
                    else => error.InvalidWeightRefCount,
                };
            },
            .swiglu => {
                const ffn_binding = typed_kernel_refs.swiglu[op_index] orelse return error.InvalidInstructionBinding;
                switch (slot_idx) {
                    0 => {
                        if (ffn_binding.w1) |w1| return @ptrCast(@constCast(w1));
                        if (ffn_binding.fused_gate_up) |*fused_gate_up| return @ptrCast(@constCast(fused_gate_up));
                        return error.MissingWeight;
                    },
                    1 => {
                        if (ffn_binding.w3) |w3| return @ptrCast(@constCast(w3));
                        if (ffn_binding.fused_gate_up) |*fused_gate_up| return @ptrCast(@constCast(fused_gate_up));
                        return @ptrCast(@constCast(&missing_weight_tensor));
                    },
                    2 => return @ptrCast(@constCast(ffn_binding.w2)),
                    3 => if (ffn_binding.w1_bias) |w1_bias|
                        if (w1_bias.len != 0) return @ptrCast(@constCast(w1_bias.ptr)) else return @ptrCast(&missing_optional_bias_value)
                    else
                        return @ptrCast(&missing_optional_bias_value),
                    4 => if (ffn_binding.w2_bias) |w2_bias|
                        if (w2_bias.len != 0) return @ptrCast(@constCast(w2_bias.ptr)) else return @ptrCast(&missing_optional_bias_value)
                    else
                        return @ptrCast(&missing_optional_bias_value),
                    else => return error.InvalidWeightRefCount,
                }
            },
            .moe => {
                const moe_binding = typed_kernel_refs.moe[op_index] orelse return error.InvalidInstructionBinding;
                if (moe_binding.experts.len != 1) return error.UnsupportedModel;
                return switch (slot_idx) {
                    0 => @ptrCast(@constCast(&moe_binding.router_weight)),
                    1 => blk: {
                        const expert = &moe_binding.experts[0];
                        if (expert.gate_proj) |*gate_proj| break :blk @ptrCast(@constCast(gate_proj));
                        if (expert.up_proj) |*up_proj| break :blk @ptrCast(@constCast(up_proj));
                        if (expert.gate_up_proj) |*gate_up_proj| break :blk @ptrCast(@constCast(gate_up_proj));
                        return error.MissingWeight;
                    },
                    2 => blk: {
                        const expert = &moe_binding.experts[0];
                        if (expert.up_proj) |*up_proj| break :blk @ptrCast(@constCast(up_proj));
                        if (expert.gate_up_proj) |*gate_up_proj| break :blk @ptrCast(@constCast(gate_up_proj));
                        return error.MissingWeight;
                    },
                    3 => blk: {
                        const expert = &moe_binding.experts[0];
                        break :blk @ptrCast(@constCast(&expert.down_proj));
                    },
                    4 => if (moe_binding.router_bias) |router_bias|
                        if (router_bias.len != 0) @ptrCast(@constCast(router_bias.ptr)) else @ptrCast(&missing_optional_bias_value)
                    else
                        @ptrCast(&missing_optional_bias_value),
                    5 => blk: {
                        const expert = &moe_binding.experts[0];
                        if (expert.gate_scales) |gate_scales| {
                            if (gate_scales.len != 0) break :blk @ptrCast(@constCast(gate_scales.ptr));
                        }
                        if (expert.gate_up_scales) |gate_up_scales| {
                            if (gate_up_scales.len != 0) break :blk @ptrCast(@constCast(gate_up_scales.ptr));
                        }
                        break :blk @ptrCast(&missing_optional_scale_value);
                    },
                    6 => blk: {
                        const expert = &moe_binding.experts[0];
                        if (expert.up_scales) |up_scales| {
                            if (up_scales.len != 0) break :blk @ptrCast(@constCast(up_scales.ptr));
                        }
                        if (expert.gate_up_scales) |gate_up_scales| {
                            if (gate_up_scales.len != 0) break :blk @ptrCast(@constCast(gate_up_scales.ptr));
                        }
                        break :blk @ptrCast(&missing_optional_scale_value);
                    },
                    7 => blk: {
                        const expert = &moe_binding.experts[0];
                        if (expert.down_scales) |down_scales| {
                            if (down_scales.len != 0) break :blk @ptrCast(@constCast(down_scales.ptr));
                        }
                        break :blk @ptrCast(&missing_optional_scale_value);
                    },
                    8 => blk: {
                        const expert = &moe_binding.experts[0];
                        if (expert.gate_bias) |gate_bias| {
                            if (gate_bias.len != 0) break :blk @ptrCast(@constCast(gate_bias.ptr));
                        }
                        if (expert.gate_up_bias) |gate_up_bias| {
                            if (gate_up_bias.len != 0) break :blk @ptrCast(@constCast(gate_up_bias.ptr));
                        }
                        break :blk @ptrCast(&missing_optional_bias_value);
                    },
                    9 => blk: {
                        const expert = &moe_binding.experts[0];
                        if (expert.up_bias) |up_bias| {
                            if (up_bias.len != 0) break :blk @ptrCast(@constCast(up_bias.ptr));
                        }
                        if (expert.gate_up_bias) |gate_up_bias| {
                            if (gate_up_bias.len != 0) break :blk @ptrCast(@constCast(gate_up_bias.ptr));
                        }
                        break :blk @ptrCast(&missing_optional_bias_value);
                    },
                    10 => blk: {
                        const expert = &moe_binding.experts[0];
                        if (expert.down_bias) |down_bias| {
                            if (down_bias.len != 0) break :blk @ptrCast(@constCast(down_bias.ptr));
                        }
                        break :blk @ptrCast(&missing_optional_bias_value);
                    },
                    11 => @ptrCast(@constCast(&missing_weight_tensor)),
                    12 => @ptrCast(@constCast(&missing_weight_tensor)),
                    else => error.InvalidWeightRefCount,
                };
            },
            .mamba_mixer => {
                const mamba_binding = typed_kernel_refs.mamba[op_index] orelse return error.InvalidInstructionBinding;
                const binding = mamba_binding;
                return switch (slot_idx) {
                    0 => @ptrCast(@constCast(binding.weights.in_proj)),
                    1 => @ptrCast(@constCast(binding.weights.conv1d_weight)),
                    2 => @ptrCast(@constCast(binding.weights.A_log)),
                    3 => @ptrCast(@constCast(binding.weights.D)),
                    4 => @ptrCast(@constCast(binding.weights.out_proj)),
                    5 => if (binding.weights.conv1d_bias) |conv_bias|
                        @ptrCast(@constCast(conv_bias))
                    else
                        @ptrCast(@constCast(&missing_weight_tensor)),
                    6 => if (binding.weights.dt_bias) |dt_bias|
                        @ptrCast(@constCast(dt_bias))
                    else
                        @ptrCast(@constCast(&missing_weight_tensor)),
                    7 => if (binding.weights.norm_weight) |norm_weight|
                        @ptrCast(@constCast(norm_weight))
                    else
                        @ptrCast(@constCast(&missing_weight_tensor)),
                    8 => @ptrCast(@constCast(&missing_weight_tensor)),
                    9 => @ptrCast(@constCast(&missing_weight_tensor)),
                    10 => @ptrCast(@constCast(&missing_weight_tensor)),
                    11 => @ptrCast(@constCast(&missing_weight_tensor)),
                    else => error.InvalidWeightRefCount,
                };
            },
            .gated_delta_net => {
                const binding = typed_kernel_refs.gated_delta[op_index] orelse return error.InvalidInstructionBinding;
                return switch (slot_idx) {
                    0 => @ptrCast(@constCast(binding.weights.in_proj)),
                    1 => @ptrCast(@constCast(binding.weights.conv1d_weight)),
                    2 => @ptrCast(@constCast(binding.weights.A_log)),
                    3 => @ptrCast(@constCast(binding.weights.out_proj)),
                    4 => if (binding.weights.conv1d_bias) |conv_bias|
                        @ptrCast(@constCast(conv_bias))
                    else
                        @ptrCast(@constCast(&missing_weight_tensor)),
                    5 => if (binding.weights.dt_bias) |dt_bias|
                        @ptrCast(@constCast(dt_bias))
                    else
                        @ptrCast(@constCast(&missing_weight_tensor)),
                    6 => if (binding.weights.norm_weight) |norm_weight|
                        @ptrCast(@constCast(norm_weight))
                    else
                        @ptrCast(@constCast(&missing_weight_tensor)),
                    else => error.InvalidWeightRefCount,
                };
            },
            .shortconv => {
                const shortconv_binding = typed_kernel_refs.shortconv[op_index] orelse return error.InvalidInstructionBinding;
                return switch (slot_idx) {
                    0 => @ptrCast(@constCast(shortconv_binding.weights.in_proj)),
                    1 => @ptrCast(@constCast(shortconv_binding.weights.conv1d_weight)),
                    2 => @ptrCast(@constCast(shortconv_binding.weights.out_proj)),
                    3 => if (shortconv_binding.weights.conv1d_bias) |conv_bias|
                        @ptrCast(@constCast(conv_bias))
                    else
                        @ptrCast(@constCast(&missing_weight_tensor)),
                    else => error.InvalidWeightRefCount,
                };
            },
            .embedding => return error.InvalidInstructionBinding,
            else => {
                error_context.setContext("block={d}, op={d}, opcode={d}, slot_idx={d}", .{
                    block_idx,
                    op_index,
                    @intFromEnum(opcode),
                    slot_idx,
                });
                return error.InvalidInstructionBinding;
            },
        }
    }

    fn resolveKernelWeightPtr(
        block_idx: usize,
        op_index: usize,
        opcode: runtime_contract.Opcode,
        typed_kernel_refs: TypedInstructionKernelRefs,
        slot_name: []const u8,
        slot_idx: usize,
    ) !*anyopaque {
        // Preserve load-time binding-name validation while routing by typed slot index.
        const expected_slots = runtime_contract.expectedKernelWeightSlots(opcode);
        if (slot_idx >= expected_slots.len) return error.InvalidWeightRefCount;
        if (!std.mem.eql(u8, expected_slots[slot_idx], slot_name)) return error.InvalidWeightBindingName;
        return resolveKernelWeightPtrForSlot(
            block_idx,
            op_index,
            opcode,
            typed_kernel_refs,
            slot_idx,
        );
    }

    fn buildInstructionWeightTable(
        allocator: std.mem.Allocator,
        block: *const cpu_forward.TransformerBlock,
        block_idx: usize,
        compiled_plan: *const runtime_contract.CompiledPlan,
        typed_kernel_refs: TypedInstructionKernelRefs,
        runtime_metadata: *const RuntimeMetadata,
    ) !InstructionWeightTable {
        const insn_len = compiled_plan.plan.instructions.len;
        const offsets = try allocator.alloc(u32, insn_len + 1);
        errdefer allocator.free(offsets);

        var total_slots: usize = 0;
        for (compiled_plan.plan.instructions) |insn| total_slots += insn.weights.len;
        const ptrs = try allocator.alloc(?*anyopaque, total_slots);
        errdefer allocator.free(ptrs);
        @memset(ptrs, null);

        var cursor: usize = 0;
        for (compiled_plan.plan.instructions, 0..) |insn, op_index| {
            offsets[op_index] = @intCast(cursor);
            const expected_slots = runtime_contract.expectedKernelWeightSlots(insn.opcode);
            if (insn.weights.len != expected_slots.len and expected_slots.len != 0) {
                return error.InvalidWeightRefCount;
            }

            if (expected_slots.len == 0) {
                if (insn.weights.len == 0) continue;
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
                ptrs[cursor] = @ptrCast(@constCast(weight));
                cursor += 1;
                continue;
            }

            for (insn.weights, 0..) |_, slot_idx| {
                const parsed = try runtime_contract.instructionKernelWeightBinding(
                    compiled_plan,
                    op_index,
                    insn.opcode,
                    slot_idx,
                );
                if (shortConvTimeMajorWeightPtr(runtime_metadata, insn.opcode, op_index, slot_idx)) |time_major_ptr| {
                    ptrs[cursor] = time_major_ptr;
                    cursor += 1;
                    continue;
                }
                const ptr = try resolveKernelWeightPtr(
                    block_idx,
                    op_index,
                    insn.opcode,
                    typed_kernel_refs,
                    parsed.slot_name,
                    slot_idx,
                );
                ptrs[cursor] = ptr;
                cursor += 1;
            }
        }
        offsets[insn_len] = @intCast(cursor);
        return .{ .offsets = offsets, .ptrs = ptrs };
    }

    fn shortConvTimeMajorWeightPtr(
        runtime_metadata: *const RuntimeMetadata,
        opcode: runtime_contract.Opcode,
        op_index: usize,
        slot_idx: usize,
    ) ?*anyopaque {
        if (opcode != .shortconv or slot_idx != 1) return null;
        if (op_index >= runtime_metadata.shortconv.len) return null;
        if (runtime_metadata.shortconv[op_index]) |*meta| {
            if (meta.conv_weight_time_major) |*tensor_view| {
                return @ptrCast(@constCast(tensor_view));
            }
        }
        return null;
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
        handle_storage: []runtime_contract.TensorHandle,
        view_storage: []runtime_contract.TensorViewDesc,
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
        if (insn.weights.len != 0) {
            if (dispatch_state.op_index + 1 >= self.instruction_weight_offsets.len) return error.InvalidInstructionBinding;
            const start: usize = self.instruction_weight_offsets[dispatch_state.op_index];
            const end: usize = self.instruction_weight_offsets[dispatch_state.op_index + 1];
            if (end < start) return error.InvalidInstructionBinding;
            const slot_count = end - start;
            if (slot_count != insn.weights.len) return error.InvalidWeightRefCount;
            for (0..slot_count) |slot_idx| {
                if (handle_count >= handle_storage.len) return error.InvalidInstructionBinding;
                const ptr_idx = start + slot_idx;
                if (ptr_idx >= self.instruction_weight_ptrs.len) return error.InvalidInstructionBinding;
                const weight_ptr = self.instruction_weight_ptrs[ptr_idx] orelse return error.MissingWeight;
                handle_storage[handle_count] = .{
                    .register = runtime_contract.registerFromIndex(@intCast(slot_idx)),
                    .ptr = weight_ptr,
                };
                handle_count += 1;
            }
        }

        return .{
            .registers = handle_storage[0..handle_count],
            .views = view_storage[0..view_count],
        };
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

    fn instructionRegisterToBufferIndex(reg: runtime_contract.RegisterRef) usize {
        return runtime_contract.registerToIndex(reg);
    }

    fn instructionOutputSlice(
        self: *const Block,
        buffer_views: []Tensor,
        scratch: *ScratchBuffer,
        reg: runtime_contract.RegisterRef,
        len: usize,
    ) []f32 {
        const reg_idx = runtime_contract.registerToIndex(reg);
        return self.resolveOutputSlice(buffer_views, scratch, reg_idx, len);
    }

    fn instructionIoSlices(
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
    ) !struct { inputs: []const runtime_contract.TensorHandle, outputs: []const runtime_contract.TensorHandle } {
        const io_count = insn.inputs.len + insn.outputs.len;
        if (registers.len < io_count) return error.InvalidInstructionBinding;
        return .{
            .inputs = registers[0..insn.inputs.len],
            .outputs = registers[insn.inputs.len..io_count],
        };
    }

    fn instructionWeightSlice(
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
    ) ![]const runtime_contract.TensorHandle {
        const io_count = insn.inputs.len + insn.outputs.len;
        if (registers.len < io_count) return error.InvalidInstructionBinding;
        const weights = registers[io_count..];
        if (weights.len != insn.weights.len) return error.InvalidWeightRefCount;
        return weights;
    }

    fn tensorFromWeightHandle(handle: runtime_contract.TensorHandle) *const Tensor {
        return @ptrCast(@alignCast(handle.ptr));
    }

    fn optionalTensorFromWeightHandle(handle: runtime_contract.TensorHandle) ?*const Tensor {
        const value: *const Tensor = @ptrCast(@alignCast(handle.ptr));
        if (value == &missing_weight_tensor) return null;
        return value;
    }

    fn isMissingWeightTensor(value: *const Tensor) bool {
        return value == &missing_weight_tensor;
    }

    fn isFusedGateUpWeight(weight: *const Tensor, d_model: usize, d_ff: usize) bool {
        if (weight.n_dims != 2) return false;
        const shape0: usize = @intCast(weight.shape[0]);
        const shape1: usize = @intCast(weight.shape[1]);
        return (shape0 == d_model and shape1 == (2 * d_ff)) or
            (shape0 == (2 * d_ff) and shape1 == d_model);
    }

    fn optionalBiasSliceFromWeightHandle(handle: runtime_contract.TensorHandle, len: usize) ?[]const f32 {
        const value: *const f32 = @ptrCast(@alignCast(handle.ptr));
        if (value == &missing_optional_bias_value) return null;
        const many: [*]const f32 = @ptrCast(value);
        return many[0..len];
    }

    fn optionalScaleSliceFromWeightHandle(handle: runtime_contract.TensorHandle, len: usize) ?[]const u8 {
        const value: *const u8 = @ptrCast(@alignCast(handle.ptr));
        if (value == &missing_optional_scale_value) return null;
        const many: [*]const u8 = @ptrCast(value);
        return many[0..len];
    }

    fn hiddenWidthFromTensor(tensor_ptr: *const Tensor) !u32 {
        return switch (tensor_ptr.n_dims) {
            1 => std.math.cast(u32, tensor_ptr.shape[0]) orelse error.InvalidShape,
            2 => std.math.cast(u32, tensor_ptr.shape[1]) orelse error.InvalidShape,
            3 => std.math.cast(u32, tensor_ptr.shape[2]) orelse error.InvalidShape,
            else => error.InvalidShape,
        };
    }

    fn instructionWeightRef(self: *const Block, op_index: usize) !*const Tensor {
        if (op_index >= self.compiled_plan.plan.instructions.len) return error.InvalidInstructionIndex;
        if (op_index + 1 >= self.instruction_weight_offsets.len) return error.InvalidInstructionIndex;
        const insn = self.compiled_plan.plan.instructions[op_index];
        if (insn.weights.len != 1) return error.InvalidWeightRefCount;
        const start: usize = self.instruction_weight_offsets[op_index];
        const end: usize = self.instruction_weight_offsets[op_index + 1];
        if (end < start) return error.InvalidInstructionBinding;
        if (end - start != 1) return error.InvalidWeightRefCount;
        if (start >= self.instruction_weight_ptrs.len) return error.InvalidInstructionBinding;
        const ptr = self.instruction_weight_ptrs[start] orelse return error.MissingWeight;
        return @ptrCast(@alignCast(ptr));
    }

    fn hasTypedKernelBinding(self: *const Block, opcode: runtime_contract.Opcode, op_index: usize) bool {
        return switch (opcode) {
            .rmsnorm => op_index < self.instruction_norm_runtime_metadata.len and self.instruction_norm_runtime_metadata[op_index] != null,
            .multihead_attention => op_index < self.instruction_attention_runtime_metadata.len and self.instruction_attention_runtime_metadata[op_index] != null,
            .mla_attention => op_index < self.instruction_mla_runtime_metadata.len and self.instruction_mla_runtime_metadata[op_index] != null,
            .swiglu => op_index < self.instruction_swiglu_runtime_metadata.len and self.instruction_swiglu_runtime_metadata[op_index] != null,
            .moe => op_index < self.instruction_moe_runtime_metadata.len and self.instruction_moe_runtime_metadata[op_index] != null,
            .mamba_mixer => op_index < self.instruction_mamba_runtime_metadata.len and self.instruction_mamba_runtime_metadata[op_index] != null,
            .gated_delta_net => op_index < self.instruction_gated_delta_runtime_metadata.len and self.instruction_gated_delta_runtime_metadata[op_index] != null,
            .shortconv => op_index < self.instruction_shortconv_runtime_metadata.len and self.instruction_shortconv_runtime_metadata[op_index] != null,
            else => false,
        };
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
        buffer_views: []Tensor,
        scratch: *ScratchBuffer,
        slot_ctx: SlotContext,
        mode: BatchedDispatchMode,
        slot_index: usize,
        slot_indices: []const usize,
        use_batched_dispatch: bool,
        bound_state_blocks: []const runtime_contract.StateBlockHandle,
        instruction_handles: []runtime_contract.TensorHandle,
        instruction_views: []runtime_contract.TensorViewDesc,
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

        const bound_state_blocks = dispatch_state.bound_state_blocks;

        for (dispatch_state.block.compiled_plan.plan.state_descs) |state_desc| {
            const maybe_state_block = runtime_contract.findStateBlock(bound_state_blocks, state_desc.id);
            if (maybe_state_block == null) {
                return error.InvalidStateDescriptorBinding;
            }
            const state_block = maybe_state_block.?;
            const normalized_state_block = state_block.*;
            if (normalized_state_block.align_bytes < state_desc.align_bytes) {
                return error.InvalidStateDescriptorBinding;
            }
            if (state_desc.size_bytes > 0 and normalized_state_block.size < state_desc.size_bytes) {
                return error.InvalidStateDescriptorBinding;
            }
            try dispatch_state.bindState(normalized_state_block);
        }
    }

    fn requireInstructionStateBinding(
        mode: BatchedDispatchMode,
        insn: *const runtime_contract.Instruction,
        plan: *const runtime_contract.ExecutionPlan,
        state_blocks: []const runtime_contract.StateBlockHandle,
    ) !void {
        _ = mode;
        _ = try runtime_contract.requireInstructionStateBlockForPlan(insn, plan, state_blocks);
    }

    fn buildInstructionStateBlocks(
        insn: *const runtime_contract.Instruction,
        dispatch_state: *RuntimeDispatchState,
    ) !InstructionStateBlocks {
        var blocks = InstructionStateBlocks{};
        const state_id = insn.state_block_id orelse return blocks;
        const binding = dispatch_state.stateBinding(state_id) orelse return error.InvalidStateDescriptorBinding;
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
        .mamba_mixer,
        .gated_delta_net,
        .shortconv,
        .mla_attention,
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

        // Slot-batched macro adapters: one opcode-specialized adapter per entry.
        table[@intFromEnum(runtime_contract.Opcode.rmsnorm)] = rmsNormKernelRuntimeAdapter;
        table[@intFromEnum(runtime_contract.Opcode.multihead_attention)] = multiheadAttentionKernelRuntimeAdapter;
        table[@intFromEnum(runtime_contract.Opcode.swiglu)] = swiGluKernelRuntimeAdapter;
        table[@intFromEnum(runtime_contract.Opcode.mamba_mixer)] = mambaMixerKernelRuntimeAdapter;
        table[@intFromEnum(runtime_contract.Opcode.gated_delta_net)] = gatedDeltaNetKernelRuntimeAdapter;
        table[@intFromEnum(runtime_contract.Opcode.shortconv)] = shortConvKernelRuntimeAdapter;
        table[@intFromEnum(runtime_contract.Opcode.mla_attention)] = mlaAttentionKernelRuntimeAdapter;
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
        caps[@intFromEnum(runtime_contract.Opcode.mamba_mixer)] = .{ .supports_batch = true, .supports_graph_emit = false, .max_batch_size = null };
        caps[@intFromEnum(runtime_contract.Opcode.gated_delta_net)] = .{ .supports_batch = true, .supports_graph_emit = false, .max_batch_size = null };
        caps[@intFromEnum(runtime_contract.Opcode.shortconv)] = .{ .supports_batch = true, .supports_graph_emit = false, .max_batch_size = null };
        caps[@intFromEnum(runtime_contract.Opcode.mla_attention)] = .{ .supports_batch = true, .supports_graph_emit = false, .max_batch_size = null };
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

    fn validateRequiresKernelBinding(opcode: runtime_contract.Opcode) bool {
        return validate_requires_kernel_binding[@intFromEnum(opcode)];
    }

    fn validateRequiresLinearWeightCheck(opcode: runtime_contract.Opcode) bool {
        return validate_requires_linear_weight_check[@intFromEnum(opcode)];
    }

    fn validateRequiresParamWeightBinding(opcode: runtime_contract.Opcode) bool {
        return validate_requires_param_weight_binding[@intFromEnum(opcode)];
    }

    fn validateRequiresSplitBoundsCheck(opcode: runtime_contract.Opcode) bool {
        return validate_requires_split_bounds_check[@intFromEnum(opcode)];
    }

    fn batchedDecodeUnsupportedOpcode(opcode: runtime_contract.Opcode) bool {
        return batched_decode_unsupported[@intFromEnum(opcode)];
    }

    fn batchedDecodeSupportedWithoutKernel(opcode: runtime_contract.Opcode) bool {
        return batched_decode_supported_without_kernel[@intFromEnum(opcode)];
    }

    const validate_requires_kernel_binding: [256]bool = blk: {
        var table = [_]bool{false} ** 256;
        table[@intFromEnum(runtime_contract.Opcode.rmsnorm)] = true;
        table[@intFromEnum(runtime_contract.Opcode.multihead_attention)] = true;
        table[@intFromEnum(runtime_contract.Opcode.mla_attention)] = true;
        table[@intFromEnum(runtime_contract.Opcode.swiglu)] = true;
        table[@intFromEnum(runtime_contract.Opcode.mamba_mixer)] = true;
        table[@intFromEnum(runtime_contract.Opcode.gated_delta_net)] = true;
        table[@intFromEnum(runtime_contract.Opcode.shortconv)] = true;
        break :blk table;
    };

    const validate_requires_linear_weight_check: [256]bool = blk: {
        var table = [_]bool{false} ** 256;
        table[@intFromEnum(runtime_contract.Opcode.linear)] = true;
        break :blk table;
    };

    const validate_requires_param_weight_binding: [256]bool = blk: {
        var table = [_]bool{false} ** 256;
        table[@intFromEnum(runtime_contract.Opcode.add_param)] = true;
        table[@intFromEnum(runtime_contract.Opcode.add_param_scalar)] = true;
        table[@intFromEnum(runtime_contract.Opcode.mul_param)] = true;
        break :blk table;
    };

    const validate_requires_split_bounds_check: [256]bool = blk: {
        var table = [_]bool{false} ** 256;
        table[@intFromEnum(runtime_contract.Opcode.split)] = true;
        break :blk table;
    };

    const batched_decode_unsupported: [256]bool = blk: {
        var table = [_]bool{false} ** 256;
        table[@intFromEnum(runtime_contract.Opcode.mamba_mixer)] = true;
        table[@intFromEnum(runtime_contract.Opcode.gated_delta_net)] = true;
        table[@intFromEnum(runtime_contract.Opcode.shortconv)] = true;
        table[@intFromEnum(runtime_contract.Opcode.mla_attention)] = true;
        break :blk table;
    };

    const batched_decode_supported_without_kernel: [256]bool = blk: {
        var table = [_]bool{false} ** 256;
        table[@intFromEnum(runtime_contract.Opcode.residual_add)] = true;
        table[@intFromEnum(runtime_contract.Opcode.mul_scalar)] = true;
        table[@intFromEnum(runtime_contract.Opcode.add_tensor)] = true;
        table[@intFromEnum(runtime_contract.Opcode.vision_patch_embed)] = true;
        table[@intFromEnum(runtime_contract.Opcode.vision_deepstack_extract)] = true;
        table[@intFromEnum(runtime_contract.Opcode.vision_spatial_merge)] = true;
        table[@intFromEnum(runtime_contract.Opcode.vision_scatter)] = true;
        break :blk table;
    };

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
            .dispatch_counters = if (enable_dispatch_observability) &layer_program_dispatch_counters else null,
            .workspace = .{ .any = @ptrCast(dispatch_state) },
        };
        try runtime_contract.validateBatchCapability(
            adapter_capabilities[@intFromEnum(insn.opcode)],
            exec_ctx.batch_size,
        );
        runtime_contract.recordExecutionDispatch(&exec_ctx, insn.opcode);
        const built_handles = try self.buildInstructionHandles(
            insn,
            dispatch_state,
            dispatch_state.instruction_handles,
            dispatch_state.instruction_views,
        );
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

    fn rmsNormKernelRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        views: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        params: []const runtime_contract.ParamBlock,
    ) !void {
        try macroKernelRuntimeAdapter(
            .rmsnorm,
            ctx,
            insn,
            registers,
            views,
            state_blocks,
            params,
        );
    }

    fn multiheadAttentionKernelRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        views: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        params: []const runtime_contract.ParamBlock,
    ) !void {
        try macroKernelRuntimeAdapter(
            .multihead_attention,
            ctx,
            insn,
            registers,
            views,
            state_blocks,
            params,
        );
    }
    fn mlaAttentionKernelRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        views: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        params: []const runtime_contract.ParamBlock,
    ) !void {
        try macroKernelRuntimeAdapter(
            .mla_attention,
            ctx,
            insn,
            registers,
            views,
            state_blocks,
            params,
        );
    }

    fn swiGluKernelRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        views: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        params: []const runtime_contract.ParamBlock,
    ) !void {
        try macroKernelRuntimeAdapter(
            .swiglu,
            ctx,
            insn,
            registers,
            views,
            state_blocks,
            params,
        );
    }

    fn moeKernelRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        views: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        params: []const runtime_contract.ParamBlock,
    ) !void {
        try macroKernelRuntimeAdapter(
            .moe,
            ctx,
            insn,
            registers,
            views,
            state_blocks,
            params,
        );
    }

    fn mambaMixerKernelRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        views: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        params: []const runtime_contract.ParamBlock,
    ) !void {
        try macroKernelRuntimeAdapter(
            .mamba_mixer,
            ctx,
            insn,
            registers,
            views,
            state_blocks,
            params,
        );
    }

    fn gatedDeltaNetKernelRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        views: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        params: []const runtime_contract.ParamBlock,
    ) !void {
        try macroKernelRuntimeAdapter(
            .gated_delta_net,
            ctx,
            insn,
            registers,
            views,
            state_blocks,
            params,
        );
    }

    fn shortConvKernelRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        views: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        params: []const runtime_contract.ParamBlock,
    ) !void {
        try macroKernelRuntimeAdapter(
            .shortconv,
            ctx,
            insn,
            registers,
            views,
            state_blocks,
            params,
        );
    }

    fn macroKernelRuntimeAdapter(
        comptime expected_opcode: runtime_contract.Opcode,
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        _: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        params: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try runtimeDispatchState(ctx);
        try requireInstructionStateBinding(state.mode, insn, &state.block.compiled_plan.plan, state_blocks);
        if (insn.opcode != expected_opcode) return error.UnsupportedOpInSequentialMode;
        if (insn.inputs.len != 1 or insn.outputs.len != 1) return error.InvalidInstructionPayload;
        const io = try instructionIoSlices(insn, registers);
        if (io.inputs.len != 1 or io.outputs.len != 1) return error.InvalidInstructionBinding;
        const weight_handles = try instructionWeightSlice(insn, registers);

        const input = &state.buffer_views[runtime_contract.registerToIndex(insn.inputs[0])];
        const output = &state.buffer_views[runtime_contract.registerToIndex(insn.outputs[0])];

        if (comptime expected_opcode == .rmsnorm) {
            if (weight_handles.len != 2) return error.InvalidWeightRefCount;
            if (state.op_index >= state.block.instruction_norm_runtime_metadata.len) return error.InvalidInstructionIndex;
            const weight = tensorFromWeightHandle(weight_handles[0]);
            const meta = state.block.instruction_norm_runtime_metadata[state.op_index] orelse return error.MissingKernelBinding;
            var norm_local = switch (meta.kind) {
                .rms => norm_kernel.NormKernel{
                    .rms = .{
                        .weight = weight,
                        .dim = meta.dim,
                        .eps = meta.eps,
                        .weight_offset = meta.weight_offset,
                        .layer_idx = meta.layer_idx,
                        .trace_point = meta.trace_point,
                    },
                },
                .layer => norm_kernel.NormKernel{
                    .layer = .{
                        .weight = weight,
                        .bias = if (meta.has_bias) optionalTensorFromWeightHandle(weight_handles[1]) else null,
                        .dim = meta.dim,
                        .eps = meta.eps,
                        .layer_idx = meta.layer_idx,
                        .trace_point = meta.trace_point,
                    },
                },
            };
            try dispatchNormWithMode(input, output, &norm_local);
            return;
        } else if (comptime expected_opcode == .multihead_attention) {
            if (weight_handles.len != 11) return error.InvalidWeightRefCount;
            if (state.op_index >= state.block.instruction_attention_runtime_metadata.len) return error.InvalidInstructionIndex;
            const meta = state.block.instruction_attention_runtime_metadata[state.op_index] orelse return error.MissingKernelBinding;
            const attention_param = try runtime_contract.paramAs(
                runtime_contract.AttentionKernelParam,
                params,
                .multihead_attention,
            );
            const q_proj_or_fused = tensorFromWeightHandle(weight_handles[0]);
            const k_proj_or_missing = tensorFromWeightHandle(weight_handles[1]);
            const v_proj_or_missing = tensorFromWeightHandle(weight_handles[2]);
            var attn_local = attn_kernel.MultiHeadAttention{
                .d_model = meta.d_model,
                .n_heads = meta.n_heads,
                .n_kv_heads = meta.n_kv_heads,
                .head_dim = meta.head_dim,
                .max_seq_len = meta.max_seq_len,
                .scale = meta.scale,
                .qk_norm_weight_offset = meta.qk_norm_weight_offset,
                .sliding_window = meta.sliding_window,
                .is_causal = meta.is_causal,
                .layer_idx = meta.layer_idx,
                .query_gate = attention_param.query_gate != 0,
                .q_proj = null,
                .k_proj = null,
                .v_proj = null,
                .o_proj = tensorFromWeightHandle(weight_handles[3]),
                .fused_qkv = null,
                .rope = meta.rope,
                .runtime_rope = meta.runtime_rope,
                .position_delta = meta.position_delta,
                .rope_interleaved = meta.rope_interleaved,
                .q_norm = null,
                .k_norm = null,
                .norm_eps = meta.norm_eps,
                .allocator = meta.allocator,
                .matmul_qkv = meta.matmul_qkv,
                .matmul_k = meta.matmul_k,
                .matmul_v = meta.matmul_v,
                .matmul_qkv_fused = meta.matmul_qkv_fused,
                .matmul_o = meta.matmul_o,
                .kernel_name_qkv = meta.kernel_name_qkv,
                .kernel_name_k = meta.kernel_name_k,
                .kernel_name_v = meta.kernel_name_v,
                .kernel_name_qkv_fused = meta.kernel_name_qkv_fused,
                .kernel_name_o = meta.kernel_name_o,
                .q_bias = null,
                .k_bias = null,
                .v_bias = null,
                .o_bias = null,
                .sinks = null,
                .flash_attention_fn = meta.flash_attention_fn,
            };
            const fused_qkv = isMissingWeightTensor(k_proj_or_missing) and isMissingWeightTensor(v_proj_or_missing);
            const split_qkv = !isMissingWeightTensor(k_proj_or_missing) and !isMissingWeightTensor(v_proj_or_missing);
            if (fused_qkv) {
                attn_local.fused_qkv = q_proj_or_fused.*;
            } else if (split_qkv) {
                attn_local.q_proj = q_proj_or_fused;
                attn_local.k_proj = k_proj_or_missing;
                attn_local.v_proj = v_proj_or_missing;
            } else {
                return error.InvalidInstructionBinding;
            }
            const query_dim: usize = attn_local.n_heads * attn_local.head_dim;
            const kv_total_dim: usize = attn_local.n_kv_heads * attn_local.head_dim;
            attn_local.q_norm = optionalTensorFromWeightHandle(weight_handles[4]);
            attn_local.k_norm = optionalTensorFromWeightHandle(weight_handles[5]);
            attn_local.q_bias = optionalBiasSliceFromWeightHandle(weight_handles[6], query_dim);
            attn_local.k_bias = optionalBiasSliceFromWeightHandle(weight_handles[7], kv_total_dim);
            attn_local.v_bias = optionalBiasSliceFromWeightHandle(weight_handles[8], kv_total_dim);
            attn_local.o_bias = optionalBiasSliceFromWeightHandle(weight_handles[9], attn_local.d_model);
            attn_local.sinks = optionalBiasSliceFromWeightHandle(weight_handles[10], attn_local.n_heads);
            try dispatchAttentionWithMode(state, insn, state_blocks, input, output, &attn_local);
            return;
        } else if (comptime expected_opcode == .mla_attention) {
            if (weight_handles.len != 7) return error.InvalidWeightRefCount;
            if (state.op_index >= state.block.instruction_mla_runtime_metadata.len) return error.InvalidInstructionIndex;
            const meta = state.block.instruction_mla_runtime_metadata[state.op_index] orelse return error.MissingKernelBinding;
            var mla_local = mla_kernel.MLAttention{
                .d_model = meta.d_model,
                .n_heads = meta.n_heads,
                .max_seq_len = meta.max_seq_len,
                .config = meta.config,
                .allocator = meta.allocator,
                .q_a_proj = tensorFromWeightHandle(weight_handles[0]),
                .q_a_norm = tensorFromWeightHandle(weight_handles[1]),
                .q_b_proj = tensorFromWeightHandle(weight_handles[2]),
                .kv_a_proj = tensorFromWeightHandle(weight_handles[3]),
                .kv_a_norm = tensorFromWeightHandle(weight_handles[4]),
                .kv_b_proj = tensorFromWeightHandle(weight_handles[5]),
                .o_proj = tensorFromWeightHandle(weight_handles[6]),
                .rope = meta.rope,
                .norm_eps = meta.norm_eps,
                .scale = meta.scale,
                .matmul_fn = meta.matmul_fn,
                .layer_idx = meta.layer_idx,
            };
            try dispatchMlaAttentionWithMode(state, insn, state_blocks, input, output, &mla_local);
            return;
        } else if (comptime expected_opcode == .swiglu) {
            if (weight_handles.len != 5) return error.InvalidWeightRefCount;
            if (state.op_index >= state.block.instruction_swiglu_runtime_metadata.len) return error.InvalidInstructionIndex;
            const meta = state.block.instruction_swiglu_runtime_metadata[state.op_index] orelse return error.MissingKernelBinding;
            const gate_or_fused = tensorFromWeightHandle(weight_handles[0]);
            const up_or_missing = tensorFromWeightHandle(weight_handles[1]);
            var ffn_local = ffn_kernel.SwiGLU{
                .d_model = meta.d_model,
                .d_ff = meta.d_ff,
                .use_gelu = meta.use_gelu,
                .use_swiglu_variant = meta.use_swiglu_variant,
                .layer_idx = meta.layer_idx,
                .w1 = null,
                .w2 = tensorFromWeightHandle(weight_handles[2]),
                .w3 = null,
                .w1_bias = optionalBiasSliceFromWeightHandle(weight_handles[3], meta.d_ff),
                .w2_bias = optionalBiasSliceFromWeightHandle(weight_handles[4], meta.d_model),
                .fused_gate_up = null,
                .fused_gate_up_layout = meta.fused_gate_up_layout,
                .allocator = meta.allocator,
                .matmul_gate = meta.matmul_gate,
                .matmul_gate_up = meta.matmul_gate_up,
                .matmul_down = meta.matmul_down,
                .kernel_name_gate = meta.kernel_name_gate,
                .kernel_name_gate_up = meta.kernel_name_gate_up,
                .kernel_name_down = meta.kernel_name_down,
            };
            if (isMissingWeightTensor(up_or_missing)) {
                // Dense-only FFN path: gate + down only.
                ffn_local.w1 = gate_or_fused;
            } else if (up_or_missing == gate_or_fused and isFusedGateUpWeight(gate_or_fused, meta.d_model, meta.d_ff)) {
                // Fused gate/up path (both slots resolve to fused tensor handle).
                ffn_local.fused_gate_up = gate_or_fused.*;
            } else {
                // Split gate/up path.
                ffn_local.w1 = gate_or_fused;
                ffn_local.w3 = up_or_missing;
            }
            try dispatchSwiGluWithMode(state, input, output, &ffn_local);
            return;
        } else if (comptime expected_opcode == .moe) {
            if (weight_handles.len != 13) return error.InvalidWeightRefCount;
            if (state.op_index >= state.block.instruction_moe_runtime_metadata.len) return error.InvalidInstructionIndex;
            const meta = state.block.instruction_moe_runtime_metadata[state.op_index] orelse return error.MissingKernelBinding;
            if (meta.num_experts != 1) return error.UnsupportedModel;
            var expert = moe_kernel.ExpertWeights{
                .gate_proj = null,
                .up_proj = null,
                .gate_up_proj = null,
                .down_proj = tensorFromWeightHandle(weight_handles[3]).*,
                .gate_scales = if (meta.gate_scales_len != 0) optionalScaleSliceFromWeightHandle(weight_handles[5], meta.gate_scales_len) else null,
                .up_scales = if (meta.up_scales_len != 0) optionalScaleSliceFromWeightHandle(weight_handles[6], meta.up_scales_len) else null,
                .gate_up_scales = null,
                .down_scales = if (meta.down_scales_len != 0) optionalScaleSliceFromWeightHandle(weight_handles[7], meta.down_scales_len) else null,
                .gate_bias = if (meta.gate_bias_len != 0) optionalBiasSliceFromWeightHandle(weight_handles[8], meta.gate_bias_len) else null,
                .up_bias = if (meta.up_bias_len != 0) optionalBiasSliceFromWeightHandle(weight_handles[9], meta.up_bias_len) else null,
                .gate_up_bias = null,
                .down_bias = if (meta.down_bias_len != 0) optionalBiasSliceFromWeightHandle(weight_handles[10], meta.down_bias_len) else null,
            };
            const gate_or_fused = tensorFromWeightHandle(weight_handles[1]).*;
            const up_or_fused = tensorFromWeightHandle(weight_handles[2]).*;
            if (meta.has_gate_up_proj) {
                expert.gate_up_proj = gate_or_fused;
                expert.gate_up_scales = if (meta.gate_up_scales_len != 0) optionalScaleSliceFromWeightHandle(weight_handles[5], meta.gate_up_scales_len) else null;
                expert.gate_up_bias = if (meta.gate_up_bias_len != 0) optionalBiasSliceFromWeightHandle(weight_handles[8], meta.gate_up_bias_len) else null;
            } else {
                if (meta.has_gate_proj) expert.gate_proj = gate_or_fused;
                if (meta.has_up_proj) expert.up_proj = up_or_fused;
            }
            var experts = [_]moe_kernel.ExpertWeights{expert};
            var moe_local = moe_kernel.MoEFFN{
                .allocator = meta.allocator,
                .d_model = meta.d_model,
                .d_ff = meta.d_ff,
                .num_experts = meta.num_experts,
                .experts_per_token = meta.experts_per_token,
                .router_weight = tensorFromWeightHandle(weight_handles[0]).*,
                .router_bias = null,
                .experts = experts[0..],
                .use_mxfp4 = meta.use_mxfp4,
                .use_swiglu_variant = meta.use_swiglu_variant,
                .use_transposed_weights = meta.use_transposed_weights,
                .layer_idx = meta.layer_idx,
                .kernel_name = meta.kernel_name,
            };
            moe_local.router_bias = optionalBiasSliceFromWeightHandle(weight_handles[4], moe_local.num_experts);
            // Router quant auxiliaries are part of the flattened slot contract.
            // CPU dense router path does not consume them in this opcode.
            _ = optionalTensorFromWeightHandle(weight_handles[11]);
            _ = optionalTensorFromWeightHandle(weight_handles[12]);
            try dispatchMoeWithMode(state, input, output, &moe_local);
            return;
        } else if (comptime expected_opcode == .mamba_mixer) {
            if (weight_handles.len != 12) return error.InvalidWeightRefCount;
            if (state.op_index >= state.block.instruction_mamba_runtime_metadata.len) return error.InvalidInstructionIndex;
            const meta = state.block.instruction_mamba_runtime_metadata[state.op_index] orelse return error.MissingKernelBinding;
            // Phase-5 slot contract carries these optional/fused follow-on weights.
            // CPU mamba mixer kernel does not consume them directly in this opcode.
            _ = tensorFromWeightHandle(weight_handles[8]);
            _ = optionalTensorFromWeightHandle(weight_handles[9]);
            _ = optionalTensorFromWeightHandle(weight_handles[10]);
            _ = optionalTensorFromWeightHandle(weight_handles[11]);
            var mamba_local = mamba_kernel.MambaKernel{
                .config = meta.mamba_config,
                .weights = .{
                    .in_proj = tensorFromWeightHandle(weight_handles[0]),
                    .conv1d_weight = tensorFromWeightHandle(weight_handles[1]),
                    .conv1d_bias = optionalTensorFromWeightHandle(weight_handles[5]),
                    .A_log = tensorFromWeightHandle(weight_handles[2]),
                    .D = tensorFromWeightHandle(weight_handles[3]),
                    .dt_bias = optionalTensorFromWeightHandle(weight_handles[6]),
                    .norm_weight = optionalTensorFromWeightHandle(weight_handles[7]),
                    .out_proj = tensorFromWeightHandle(weight_handles[4]),
                },
                .matmul_in_proj = meta.matmul_in_proj,
                .matmul_out_proj = meta.matmul_out_proj,
                .ssm_scan = meta.ssm_scan orelse return error.InvalidInstructionBinding,
                .layer_idx = meta.layer_idx,
            };
            try dispatchMambaWithMode(state, insn, state_blocks, input, output, &mamba_local);
            return;
        } else if (comptime expected_opcode == .gated_delta_net) {
            if (weight_handles.len != 7) return error.InvalidWeightRefCount;
            if (state.op_index >= state.block.instruction_gated_delta_runtime_metadata.len) return error.InvalidInstructionIndex;
            const meta = state.block.instruction_gated_delta_runtime_metadata[state.op_index] orelse return error.MissingKernelBinding;
            const d_model = try hiddenWidthFromTensor(input);
            var gated_delta_local = gated_delta_kernel.GatedDeltaKernel{
                .config = .{
                    .d_model = d_model,
                    .d_conv = meta.config.d_conv,
                    .n_heads = meta.config.n_heads,
                    .d_head = meta.config.d_head,
                },
                .weights = .{
                    .in_proj = tensorFromWeightHandle(weight_handles[0]),
                    .conv1d_weight = tensorFromWeightHandle(weight_handles[1]),
                    .conv1d_bias = optionalTensorFromWeightHandle(weight_handles[4]),
                    .A_log = tensorFromWeightHandle(weight_handles[2]),
                    .dt_bias = optionalTensorFromWeightHandle(weight_handles[5]),
                    .norm_weight = optionalTensorFromWeightHandle(weight_handles[6]),
                    .out_proj = tensorFromWeightHandle(weight_handles[3]),
                },
                .matmul_in_proj = meta.matmul_in_proj,
                .matmul_out_proj = meta.matmul_out_proj,
                .layer_idx = meta.layer_idx,
                .conv_weight_transposed = if (meta.conv_weight_time_major) |tensor_view|
                    tensor_view.asSlice(f32)
                else
                    null,
            };
            if (gated_delta_local.conv_weight_transposed == null) {
                log.warn("inference", "CPU gated-delta runtime metadata missing time-major conv view", .{
                    .op_index = state.op_index,
                    .layer_idx = meta.layer_idx,
                    .weight_dtype = @tagName(gated_delta_local.weights.conv1d_weight.dtype),
                    .weight_dims = gated_delta_local.weights.conv1d_weight.n_dims,
                    .weight_shape0 = if (gated_delta_local.weights.conv1d_weight.n_dims > 0) gated_delta_local.weights.conv1d_weight.shape[0] else 0,
                    .weight_shape1 = if (gated_delta_local.weights.conv1d_weight.n_dims > 1) gated_delta_local.weights.conv1d_weight.shape[1] else 0,
                });
            }
            try dispatchGatedDeltaWithMode(state, insn, state_blocks, input, output, &gated_delta_local);
            return;
        } else if (comptime expected_opcode == .shortconv) {
            if (weight_handles.len != 4) return error.InvalidWeightRefCount;
            if (state.op_index >= state.block.instruction_shortconv_runtime_metadata.len) return error.InvalidInstructionIndex;
            const meta = state.block.instruction_shortconv_runtime_metadata[state.op_index] orelse return error.MissingKernelBinding;
            const conv_weight = tensorFromWeightHandle(weight_handles[1]);
            var shortconv_local = shortconv_kernel.ShortConvKernel{
                .config = meta.config,
                .weights = .{
                    .in_proj = tensorFromWeightHandle(weight_handles[0]),
                    .conv1d_weight = conv_weight,
                    .conv1d_bias = optionalTensorFromWeightHandle(weight_handles[3]),
                    .out_proj = tensorFromWeightHandle(weight_handles[2]),
                },
                .matmul_in_proj = meta.matmul_in_proj,
                .matmul_out_proj = meta.matmul_out_proj,
                .matmul_in_proj_name = meta.matmul_in_proj_name,
                .matmul_out_proj_name = meta.matmul_out_proj_name,
                .layer_idx = meta.layer_idx,
                .conv_weight_transposed = if (meta.conv_weight_time_major != null) conv_weight.asSlice(f32) else null,
                .weight_allocator = null,
            };
            try dispatchShortConvWithMode(state, insn, state_blocks, input, output, &shortconv_local);
            return;
        }
        unreachable;
    }

    fn dispatchNormWithMode(
        input: *const Tensor,
        output: *Tensor,
        kernel: *const norm_kernel.NormKernel,
    ) !void {
        kernel.forward(input, output);
    }

    fn dispatchAttentionWithMode(
        state: *RuntimeDispatchState,
        insn: *const runtime_contract.Instruction,
        state_blocks: []const runtime_contract.StateBlockHandle,
        input: *const Tensor,
        output: *Tensor,
        kernel: *const attn_kernel.MultiHeadAttention,
    ) !void {
        const instruction_state_id = insn.state_block_id;
        if (instruction_state_id == null) {
            var stateless_cache = runtime.AttnCache{};
            try kernel.forward(
                input,
                output,
                &stateless_cache,
                &state.scratch.attn_scratch,
                &state.scratch.matmul_scratch,
                false,
            );
            return;
        }
        const batched_cache = try requireLayerBatchedCacheForInstruction(state, insn, state_blocks);
        switch (state.mode) {
            .single_slot => try kernel.forwardWithBatchedCache(
                input,
                output,
                batched_cache,
                state.slot_index,
                &state.scratch.attn_scratch,
                &state.scratch.matmul_scratch,
                state.slot_ctx.use_cache,
            ),
            .slot_batch => try kernel.forwardWithBatchedCacheSlots(
                input,
                output,
                batched_cache,
                state.slot_indices,
                &state.scratch.attn_scratch,
                &state.scratch.matmul_scratch,
                state.slot_ctx.use_cache,
            ),
        }
    }

    fn dispatchMlaAttentionWithMode(
        state: *RuntimeDispatchState,
        insn: *const runtime_contract.Instruction,
        state_blocks: []const runtime_contract.StateBlockHandle,
        input: *const Tensor,
        output: *Tensor,
        kernel: *const mla_kernel.MLAttention,
    ) !void {
        if (!state.use_batched_dispatch) return error.InvalidStateDescriptorBinding;
        if (state.mode == .slot_batch) return runtime.BatchedKernelError.UnsupportedBatchedDecodeKernel;
        const binding = try requireMlaRuntimeBindingForInstruction(
            state,
            insn,
            state_blocks,
            state.block.block_idx,
        );
        try kernel.forward(
            input,
            output,
            binding.cache,
            binding.scratch,
            &state.scratch.matmul_scratch,
            state.slot_ctx.use_cache,
        );
    }

    fn dispatchSwiGluWithMode(
        state: *RuntimeDispatchState,
        input: *const Tensor,
        output: *Tensor,
        kernel: *const ffn_kernel.SwiGLU,
    ) !void {
        if (state.use_batched_dispatch and state.mode == .single_slot) {
            try kernel.forward(input, output, state.scratch.getFfnScratch(state.slot_index), &state.scratch.matmul_scratch);
            return;
        }
        try kernel.forward(input, output, &state.scratch.ffn_scratch, &state.scratch.matmul_scratch);
    }

    fn dispatchMoeWithMode(
        state: *RuntimeDispatchState,
        input: *const Tensor,
        output: *Tensor,
        kernel: *const moe_kernel.MoEFFN,
    ) !void {
        if (state.use_batched_dispatch and state.mode == .single_slot) {
            try kernel.forward(input, output, state.scratch.getMoeScratch(state.slot_index), &state.scratch.matmul_scratch);
            return;
        }
        try kernel.forward(input, output, &state.scratch.moe_scratch, &state.scratch.matmul_scratch);
    }

    fn dispatchMambaWithMode(
        state: *RuntimeDispatchState,
        insn: *const runtime_contract.Instruction,
        state_blocks: []const runtime_contract.StateBlockHandle,
        input: *const Tensor,
        output: *Tensor,
        kernel: *const mamba_kernel.MambaKernel,
    ) !void {
        if (!state.use_batched_dispatch) return error.InvalidStateDescriptorBinding;
        if (state.mode == .slot_batch) return runtime.BatchedKernelError.UnsupportedBatchedDecodeKernel;
        const binding = try requireMambaRuntimeBindingForInstruction(
            state,
            insn,
            state_blocks,
            state.block.block_idx,
        );
        try kernel.forward(
            input,
            output,
            binding.state,
            binding.scratch,
            &state.scratch.matmul_scratch,
        );
    }

    fn dispatchGatedDeltaWithMode(
        state: *RuntimeDispatchState,
        insn: *const runtime_contract.Instruction,
        state_blocks: []const runtime_contract.StateBlockHandle,
        input: *const Tensor,
        output: *Tensor,
        kernel: *const gated_delta_kernel.GatedDeltaKernel,
    ) !void {
        if (!state.use_batched_dispatch) return error.InvalidStateDescriptorBinding;
        if (state.mode == .slot_batch) return runtime.BatchedKernelError.UnsupportedBatchedDecodeKernel;
        const binding = try requireGatedDeltaRuntimeBindingForInstruction(
            state,
            insn,
            state_blocks,
            state.block.block_idx,
        );
        try kernel.forward(
            input,
            output,
            binding.state,
            binding.scratch,
            &state.scratch.matmul_scratch,
        );
    }

    fn dispatchShortConvWithMode(
        state: *RuntimeDispatchState,
        insn: *const runtime_contract.Instruction,
        state_blocks: []const runtime_contract.StateBlockHandle,
        input: *const Tensor,
        output: *Tensor,
        kernel: *const shortconv_kernel.ShortConvKernel,
    ) !void {
        if (!state.use_batched_dispatch) return error.InvalidStateDescriptorBinding;
        if (state.mode == .slot_batch) return runtime.BatchedKernelError.UnsupportedBatchedDecodeKernel;
        const binding = try requireShortConvRuntimeBindingForInstruction(
            state,
            insn,
            state_blocks,
            state.block.block_idx,
        );
        try kernel.forward(
            input,
            output,
            binding.state,
            binding.scratch,
            &state.scratch.matmul_scratch,
        );
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
        const output_slice = state.block.instructionOutputSlice(
            state.buffer_views,
            state.scratch,
            insn.outputs[0],
            output_len,
        );
        const p = try runtime_contract.paramAs(runtime_contract.ScalarOpParam, params, .mul_scalar);
        cpu_elementwise.mulScalar(input_data[0..output_len], output_slice[0..output_len], @bitCast(p.scalar));
        const out_idx = instructionRegisterToBufferIndex(insn.outputs[0]);
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
        const output_slice = state.block.instructionOutputSlice(
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
        const out_idx = instructionRegisterToBufferIndex(insn.outputs[0]);
        state.buffer_views[out_idx] = tensorFromSlice(
            output_slice[0..output_len],
            viewToTensorShape(larger_view),
            @intCast(larger_view.rank),
        );
    }

    fn sequentialLinearRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        views: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        _: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try runtimeDispatchState(ctx);
        try requireInstructionStateBinding(state.mode, insn, &state.block.compiled_plan.plan, state_blocks);
        if (insn.inputs.len != 1 or insn.outputs.len != 1) return error.InvalidInstructionBinding;
        const weight_handles = try instructionWeightSlice(insn, registers);
        if (weight_handles.len != 1) return error.InvalidWeightRefCount;
        const input_view = views[0];
        const input_buf = &state.buffer_views[runtime_contract.registerToIndex(insn.inputs[0])];
        const weight = tensorFromWeightHandle(weight_handles[0]);
        if (isMissingWeightTensor(weight)) {
            error_context.setContext("block={d}, op={d}, weight_ref={s}", .{
                state.block.block_idx,
                state.op_index,
                "missing_weight_tensor",
            });
            return error.MissingWeight;
        }
        const seq_len: usize = input_view.shape[1];
        const output_features: usize = if (weight.dtype == .f32)
            @intCast(weight.shape[1])
        else
            @intCast(weight.shape[0]);

        const out_raw_idx = runtime_contract.registerToIndex(insn.outputs[0]);
        if (out_raw_idx >= state.block.tmp_register_to_scratch_idx.len) return error.InvalidInstructionBinding;
        const output_slice = blk: {
            // Registers 1+ are mapped through liveness analysis.
            if (out_raw_idx >= 1) {
                const mapped_idx: usize = state.block.tmp_register_to_scratch_idx[out_raw_idx];
                std.debug.assert(mapped_idx >= 1);
                std.debug.assert(mapped_idx < state.scratch.tmp.len);
                break :blk state.scratch.tmp[mapped_idx][0 .. seq_len * output_features];
            }

            // Residual output staging uses dedicated layer_tmp (slot 0), whose
            // width is registered from compiled plan specs at block init.
            // Slot 0 is never register-mapped and therefore alias-safe.
            const input_ptr = @intFromPtr(input_buf.data().ptr);
            const layer_tmp_ptr = @intFromPtr(state.scratch.tmp[0].ptr);
            if (input_ptr == layer_tmp_ptr) {
                return error.InvalidScratchLayout;
            }
            break :blk state.scratch.tmp[0][0 .. seq_len * output_features];
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

        const out_idx = instructionRegisterToBufferIndex(insn.outputs[0]);
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
        const out_slice = state.block.instructionOutputSlice(
            state.buffer_views,
            state.scratch,
            insn.outputs[0],
            out_size,
        );
        var output_2d = Tensor.view2D(std.mem.sliceAsBytes(out_slice), m_dim, n_dim);
        var a_view = Tensor.view2D(left_buf.data(), m_dim, left_view.shape[2]);
        var b_view = Tensor.view2D(right_buf.data(), n_dim, right_view.shape[2]);
        try cpu_linalg.matmulAuto(&a_view, &b_view, &output_2d, &state.scratch.matmul_scratch);

        const out_idx = instructionRegisterToBufferIndex(insn.outputs[0]);
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
            out_slices[out_idx] = state.block.instructionOutputSlice(
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
            const out_buf_idx = instructionRegisterToBufferIndex(insn.outputs[out_idx]);
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
        const output_slice = state.block.instructionOutputSlice(
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
        const out_idx = instructionRegisterToBufferIndex(insn.outputs[0]);
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
        const output_slice = state.block.instructionOutputSlice(
            state.buffer_views,
            state.scratch,
            insn.outputs[0],
            output_len,
        );
        const p = try runtime_contract.paramAs(runtime_contract.ScalarOpParam, params, .add_scalar);
        cpu_elementwise.addScalar(input_data[0..output_len], output_slice[0..output_len], @bitCast(p.scalar));
        const out_idx = instructionRegisterToBufferIndex(insn.outputs[0]);
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
            const output_slice = state.block.instructionOutputSlice(
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
            const out_idx = instructionRegisterToBufferIndex(insn.outputs[0]);
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
        const output_slice = state.block.instructionOutputSlice(
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
        const out_idx = instructionRegisterToBufferIndex(insn.outputs[0]);
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
        const output_slice = state.block.instructionOutputSlice(
            state.buffer_views,
            state.scratch,
            insn.outputs[0],
            output_len,
        );
        const p = try runtime_contract.paramAs(runtime_contract.ScalarOpParam, params, .pow);
        cpu_elementwise.powScalar(input_data[0..output_len], output_slice[0..output_len], @bitCast(p.scalar));
        const out_idx = instructionRegisterToBufferIndex(insn.outputs[0]);
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
        const output_slice = state.block.instructionOutputSlice(
            state.buffer_views,
            state.scratch,
            insn.outputs[0],
            output_len,
        );
        cpu_elementwise.rsqrt(input_data[0..output_len], output_slice[0..output_len]);
        const out_idx = instructionRegisterToBufferIndex(insn.outputs[0]);
        state.buffer_views[out_idx] = tensorFromSlice(
            output_slice[0..output_len],
            viewToTensorShape(input_view),
            @intCast(input_view.rank),
        );
    }

    fn sequentialAddParamRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        views: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        _: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try runtimeDispatchState(ctx);
        try requireInstructionStateBinding(state.mode, insn, &state.block.compiled_plan.plan, state_blocks);
        if (insn.inputs.len != 1 or insn.outputs.len != 1) return error.InvalidInstructionBinding;
        const weight_handles = try instructionWeightSlice(insn, registers);
        if (weight_handles.len != 1) return error.InvalidWeightRefCount;
        const input_view = views[0];
        const input_numel = viewNumel(input_view);
        const input_tensor = state.buffer_views[runtime_contract.registerToIndex(insn.inputs[0])];
        const param = tensorFromWeightHandle(weight_handles[0]);
        if (isMissingWeightTensor(param)) {
            error_context.setContext("block={d}, op={d}, param_ref={s}", .{
                state.block.block_idx,
                state.op_index,
                "missing_weight_tensor",
            });
            return error.MissingWeight;
        }
        const output_len = @max(input_numel, param.numel);
        const output_slice = state.block.instructionOutputSlice(
            state.buffer_views,
            state.scratch,
            insn.outputs[0],
            output_len,
        );
        try cpu_broadcast.addParam(input_tensor, param, output_slice[0..output_len]);
        const out_idx = instructionRegisterToBufferIndex(insn.outputs[0]);
        state.buffer_views[out_idx] = tensorFromSlice(
            output_slice[0..output_len],
            viewToTensorShape(input_view),
            @intCast(input_view.rank),
        );
    }

    fn sequentialAddParamScalarRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        _: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        params: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try runtimeDispatchState(ctx);
        try requireInstructionStateBinding(state.mode, insn, &state.block.compiled_plan.plan, state_blocks);
        const weight_handles = try instructionWeightSlice(insn, registers);
        if (weight_handles.len != 1) return error.InvalidWeightRefCount;
        const param = tensorFromWeightHandle(weight_handles[0]);
        if (isMissingWeightTensor(param)) {
            error_context.setContext("block={d}, op={d}, param_ref={s}", .{
                state.block.block_idx,
                state.op_index,
                "missing_weight_tensor",
            });
            return error.MissingWeight;
        }
        const p_len = param.numel;
        const output_slice = state.block.instructionOutputSlice(
            state.buffer_views,
            state.scratch,
            insn.outputs[0],
            p_len,
        );
        const p = try runtime_contract.paramAs(runtime_contract.AddParamScalarParam, params, .add_param_scalar);
        cpu_broadcast.addParamScalar(param, output_slice[0..p_len], @bitCast(p.scalar));
        const out_idx = instructionRegisterToBufferIndex(insn.outputs[0]);
        state.buffer_views[out_idx] = tensorFromSlice(
            output_slice[0..p_len],
            param.shape,
            param.n_dims,
        );
    }

    fn sequentialMulParamRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        views: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        _: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try runtimeDispatchState(ctx);
        try requireInstructionStateBinding(state.mode, insn, &state.block.compiled_plan.plan, state_blocks);
        if (insn.inputs.len != 1 or insn.outputs.len != 1) return error.InvalidInstructionBinding;
        const weight_handles = try instructionWeightSlice(insn, registers);
        if (weight_handles.len != 1) return error.InvalidWeightRefCount;
        const input_view = views[0];
        const input_numel = viewNumel(input_view);
        const input_tensor = state.buffer_views[runtime_contract.registerToIndex(insn.inputs[0])];
        const param = tensorFromWeightHandle(weight_handles[0]);
        if (isMissingWeightTensor(param)) {
            error_context.setContext("block={d}, op={d}, param_ref={s}", .{
                state.block.block_idx,
                state.op_index,
                "missing_weight_tensor",
            });
            return error.MissingWeight;
        }
        const output_len = @max(input_numel, param.numel);
        const output_slice = state.block.instructionOutputSlice(
            state.buffer_views,
            state.scratch,
            insn.outputs[0],
            output_len,
        );
        try cpu_broadcast.mulParam(input_tensor, param, output_slice[0..output_len]);
        const out_idx = instructionRegisterToBufferIndex(insn.outputs[0]);
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

        const out_idx = instructionRegisterToBufferIndex(insn.outputs[0]);
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
        const out_slice = state.block.instructionOutputSlice(
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
        const out_idx = instructionRegisterToBufferIndex(insn.outputs[0]);
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
        const bound_state = if (insn.state_block_id) |state_id| blk: {
            const state_block = runtime_contract.findStateBlock(state_blocks, state_id) orelse {
                return error.InvalidStateDescriptorBinding;
            };
            const state_value = runtime_contract.stateValueFromBlock(*state_bindings.KvRuntimeState, state_block) orelse {
                return error.InvalidStateDescriptorBinding;
            };
            if (state_value.runtime_kind != runtime_contract.state_runtime_kind_kv_cache) {
                return error.InvalidStateDescriptorBinding;
            }
            break :blk state_value;
        } else null;
        const slot_scratch = if (bound_state) |kv_state| kv_state.scratch else state.scratch;
        const slot_index = if (bound_state) |kv_state| kv_state.slot_index else state.slot_index;
        const slot_state = slot_scratch.getSlotLayerState(slot_index, state.block.block_idx) orelse {
            return error.InvalidStateDescriptorBinding;
        };
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
        const out_slice = state.block.instructionOutputSlice(
            state.buffer_views,
            state.scratch,
            insn.outputs[0],
            numel,
        );
        @memcpy(out_slice, input_data[0..numel]);
        const out_idx = instructionRegisterToBufferIndex(insn.outputs[0]);
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
        const out_slice = state.block.instructionOutputSlice(
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
        const out_idx = instructionRegisterToBufferIndex(insn.outputs[0]);
        state.buffer_views[out_idx] = tensorFromSlice(
            out_slice[0..out_numel],
            sdpa_shape,
            4,
        );
    }

    fn requireLayerBatchedCacheForInstruction(
        state: *RuntimeDispatchState,
        insn: *const runtime_contract.Instruction,
        state_blocks: []const runtime_contract.StateBlockHandle,
    ) !*kv_cache.BatchedKVCache {
        const state_id = insn.state_block_id orelse return error.InvalidStateDescriptorBinding;
        const state_block = runtime_contract.findStateBlock(state_blocks, state_id) orelse {
            return error.InvalidStateDescriptorBinding;
        };
        const state_value = runtime_contract.stateValueFromBlock(*state_bindings.KvRuntimeState, state_block) orelse {
            return error.InvalidStateDescriptorBinding;
        };
        if (state_value.runtime_kind != runtime_contract.state_runtime_kind_kv_cache) {
            return error.InvalidStateDescriptorBinding;
        }
        if (state.block.block_idx >= state_value.layered_cache.layers.len) return error.InvalidStateDescriptorBinding;
        return state_value.layered_cache.getLayer(state.block.block_idx);
    }

    const MlaRuntimeBinding = struct {
        cache: *runtime.MLACache,
        scratch: *runtime.MLATemp,
    };

    fn requireMlaRuntimeBindingForInstruction(
        _: *RuntimeDispatchState,
        insn: *const runtime_contract.Instruction,
        state_blocks: []const runtime_contract.StateBlockHandle,
        block_idx: usize,
    ) !MlaRuntimeBinding {
        const state_id = insn.state_block_id orelse return error.InvalidStateDescriptorBinding;
        const state_block = runtime_contract.findStateBlock(state_blocks, state_id) orelse {
            return error.InvalidStateDescriptorBinding;
        };
        const state_value = runtime_contract.stateValueFromBlock(*state_bindings.KvRuntimeState, state_block) orelse {
            return error.InvalidStateDescriptorBinding;
        };
        if (state_value.runtime_kind != runtime_contract.state_runtime_kind_kv_cache) {
            return error.InvalidStateDescriptorBinding;
        }
        const slot_state = state_value.scratch.getSlotLayerState(state_value.slot_index, block_idx) orelse {
            return error.InvalidStateDescriptorBinding;
        };
        const mla_cache = if (slot_state.mla_cache) |*cache| cache else return runtime.SlotContextError.MissingMlaCache;
        const mla_scratch = state_value.scratch.getMLAScratch() orelse return runtime.SlotContextError.MissingMlaScratch;
        return .{
            .cache = mla_cache,
            .scratch = mla_scratch,
        };
    }

    const MambaRuntimeBinding = struct {
        state: *runtime.MambaState,
        scratch: *runtime.MambaScratch,
    };

    const GatedDeltaRuntimeBinding = struct {
        state: *runtime.GatedDeltaState,
        scratch: *runtime.GatedDeltaScratch,
    };

    fn requireMambaRuntimeBindingForInstruction(
        _: *RuntimeDispatchState,
        insn: *const runtime_contract.Instruction,
        state_blocks: []const runtime_contract.StateBlockHandle,
        block_idx: usize,
    ) !MambaRuntimeBinding {
        const state_id = insn.state_block_id orelse return error.InvalidStateDescriptorBinding;
        const state_block = runtime_contract.findStateBlock(state_blocks, state_id) orelse {
            return error.InvalidStateDescriptorBinding;
        };
        const recurrent_state = runtime_contract.stateValueFromBlock(*state_bindings.RecurrentRuntimeState, state_block) orelse {
            return error.InvalidStateDescriptorBinding;
        };
        if (recurrent_state.runtime_kind != runtime_contract.state_runtime_kind_mamba_cache) {
            return error.InvalidStateDescriptorBinding;
        }
        const slot_state = recurrent_state.scratch.getSlotLayerState(recurrent_state.slot_index, block_idx) orelse {
            return error.InvalidStateDescriptorBinding;
        };
        const mamba_state = if (slot_state.mamba_state) |*slot_mamba_state| slot_mamba_state else return error.InvalidStateDescriptorBinding;
        const mamba_scratch = recurrent_state.scratch.getMambaScratch() orelse return error.InvalidStateDescriptorBinding;
        return .{
            .state = mamba_state,
            .scratch = mamba_scratch,
        };
    }

    fn requireGatedDeltaRuntimeBindingForInstruction(
        _: *RuntimeDispatchState,
        insn: *const runtime_contract.Instruction,
        state_blocks: []const runtime_contract.StateBlockHandle,
        block_idx: usize,
    ) !GatedDeltaRuntimeBinding {
        const state_id = insn.state_block_id orelse return error.InvalidStateDescriptorBinding;
        const state_block = runtime_contract.findStateBlock(state_blocks, state_id) orelse {
            return error.InvalidStateDescriptorBinding;
        };
        const recurrent_state = runtime_contract.stateValueFromBlock(*state_bindings.RecurrentRuntimeState, state_block) orelse {
            return error.InvalidStateDescriptorBinding;
        };
        if (recurrent_state.runtime_kind != runtime_contract.state_runtime_kind_gated_delta_cache) {
            return error.InvalidStateDescriptorBinding;
        }
        const slot_state = recurrent_state.scratch.getSlotLayerState(recurrent_state.slot_index, block_idx) orelse {
            return error.InvalidStateDescriptorBinding;
        };
        const gated_delta_state = if (slot_state.gated_delta_state) |*slot_gated_delta_state|
            slot_gated_delta_state
        else {
            return error.InvalidStateDescriptorBinding;
        };
        const gated_delta_scratch = recurrent_state.scratch.getGatedDeltaScratch() orelse {
            return error.InvalidStateDescriptorBinding;
        };
        return .{
            .state = gated_delta_state,
            .scratch = gated_delta_scratch,
        };
    }

    const ShortConvRuntimeBinding = struct {
        state: *runtime.ShortConvState,
        scratch: *runtime.ShortConvScratch,
    };

    fn requireShortConvRuntimeBindingForInstruction(
        _: *RuntimeDispatchState,
        insn: *const runtime_contract.Instruction,
        state_blocks: []const runtime_contract.StateBlockHandle,
        block_idx: usize,
    ) !ShortConvRuntimeBinding {
        const state_id = insn.state_block_id orelse return error.InvalidStateDescriptorBinding;
        const state_block = runtime_contract.findStateBlock(state_blocks, state_id) orelse {
            return error.InvalidStateDescriptorBinding;
        };
        const recurrent_state = runtime_contract.stateValueFromBlock(*state_bindings.RecurrentRuntimeState, state_block) orelse {
            return error.InvalidStateDescriptorBinding;
        };
        if (recurrent_state.runtime_kind != runtime_contract.state_runtime_kind_shortconv_cache) {
            return error.InvalidStateDescriptorBinding;
        }
        const slot_state = recurrent_state.scratch.getSlotLayerState(recurrent_state.slot_index, block_idx) orelse {
            return error.InvalidStateDescriptorBinding;
        };
        const shortconv_state = if (slot_state.shortconv_state) |*slot_shortconv_state| slot_shortconv_state else return error.InvalidStateDescriptorBinding;
        const shortconv_scratch = recurrent_state.scratch.getShortConvScratch() orelse return error.InvalidStateDescriptorBinding;
        return .{
            .state = shortconv_state,
            .scratch = shortconv_scratch,
        };
    }

    fn residualScaleValue(self: *const Block, scale: ResidualScale) f32 {
        return switch (scale) {
            .one => 1.0,
            .residual_multiplier => self.block.residual_multiplier,
            .literal => |v| v,
        };
    }

    fn scratchTempSlice(self: *const Block, scratch: *ScratchBuffer, reg_idx: usize, len: usize) []f32 {
        // All non-residual registers map through the compiled liveness
        // allocator to physical scratch slots. Register 0 (residual) is handled
        // by resolveOutputSlice directly.
        if (reg_idx >= 1 and reg_idx < self.tmp_register_to_scratch_idx.len) {
            const mapped_idx: usize = self.tmp_register_to_scratch_idx[reg_idx];
            std.debug.assert(mapped_idx >= 1);
            std.debug.assert(mapped_idx < scratch.tmp.len);
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

    fn resolveOutputSlice(self: *const Block, buffer_views: []Tensor, scratch: *ScratchBuffer, reg_idx: usize, len: usize) []f32 {
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
        try scratch.registerTmpLayout(self.tmp_slot_width_hints, self.tmp_slot_active);
        try scratch.ensureForMode(if (use_cache) .decode else .prefill, seq_len);

        // Buffer lookup table: register index -> Tensor
        // Register 0 (residual) maps to the output buffer; all other registers
        // are backed by scratch slots assigned through liveness analysis.
        const reg_count = self.compiled_plan.plan.register_count;
        var buffer_views_arr: [runtime_contract.max_register_count]Tensor = undefined;
        const buffer_views = buffer_views_arr[0..reg_count];
        buffer_views[0] = out.*;
        for (1..reg_count) |reg_idx| {
            const mapped = self.tmp_register_to_scratch_idx[reg_idx];
            buffer_views[reg_idx] = Tensor.view3DSlice(scratch.tmp[mapped], seq_len, self.hidden_size);
        }

        // Initialize residual stream with input
        copyTensor(x, out);

        // Populate shared scratch only for kernels present in this block.
        const slot_state = scratch.getSlotLayerState(0, self.block_idx) orelse return error.InvalidState;
        var shared_state = SharedPersistentState{};
        const ctx = SlotContext{
            .slot_state_ptr = slot_state,
            .shared_state = &shared_state,
            .scratch = scratch,
            .use_cache = use_cache,
        };
        const instruction_handles = try scratch.allocator.alloc(
            runtime_contract.TensorHandle,
            self.instruction_handle_capacity,
        );
        defer if (instruction_handles.len > 0) scratch.allocator.free(instruction_handles);
        const instruction_views = try scratch.allocator.alloc(
            runtime_contract.TensorViewDesc,
            self.instruction_handle_capacity,
        );
        defer if (instruction_views.len > 0) scratch.allocator.free(instruction_views);

        var dispatch_state = RuntimeDispatchState{
            .block = self,
            .op_index = 0,
            .buffer_views = buffer_views,
            .scratch = scratch,
            .slot_ctx = ctx,
            .mode = .single_slot,
            .slot_index = 0,
            .slot_indices = &.{},
            .use_batched_dispatch = false,
            .bound_state_blocks = &.{},
            .instruction_handles = instruction_handles,
            .instruction_views = instruction_views,
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
        const final_reg_idx = runtime_contract.registerToIndex(
            runtime_contract.planFinalOutputRegister(&self.compiled_plan.plan),
        );
        if (final_reg_idx != 0) {
            copyTensor(&buffer_views[final_reg_idx], &buffer_views[0]);
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
            if (validateRequiresKernelBinding(insn.opcode)) {
                if (!self.hasTypedKernelBinding(insn.opcode, op_index)) {
                    error_context.setContext("block={d}, op={d}, kernel_ref=MissingKernelBinding", .{
                        self.block_idx,
                        op_index,
                    });
                    return error.KernelIndexOutOfBounds;
                }
            }

            if (validateRequiresLinearWeightCheck(insn.opcode)) {
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
            }

            if (validateRequiresParamWeightBinding(insn.opcode)) {
                _ = self.instructionWeightRef(op_index) catch |err| {
                    error_context.setContext("block={d}, op={d}, param_ref={s}", .{
                        self.block_idx,
                        op_index,
                        @errorName(err),
                    });
                    return err;
                };
            }

            if (validateRequiresSplitBoundsCheck(insn.opcode)) {
                if (insn.outputs.len == 0) return error.TooManySplitOutputs;
                const out_start_idx = runtime_contract.registerToIndex(insn.outputs[0]);
                if (out_start_idx == 0 or insn.outputs.len + out_start_idx > self.compiled_plan.plan.register_count) {
                    return error.TooManySplitOutputs;
                }
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
        if (self.block.getGatedDeltaKernel()) |gated_delta_k| {
            try writer.writeByteNTimes(' ', indent + 2);
            try writer.print("(mixer): GatedDelta(d_model={}, d_head={}, d_conv={})\n", .{
                gated_delta_k.config.d_model,
                gated_delta_k.config.d_head,
                gated_delta_k.config.d_conv,
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
            const opcode = insn.opcode;
            switch (opcode) {
                .rmsnorm => if (self.instruction_norm_refs[op_index]) |binding| {
                    try writer.writeAll(": ");
                    const norm = binding.*;
                    switch (norm) {
                        .rms => |n| try formatRmsNormLike(writer, n.dim, n.eps, n.weight_offset),
                        .layer => |n| try writer.print("LayerNorm(dim={}, eps={e})", .{ n.dim, n.eps }),
                    }
                },
                .multihead_attention => if (self.instruction_attention_bindings[op_index]) |a| {
                    try writer.print(": Attention(n_heads={}, head_dim={})", .{ a.n_heads, a.head_dim });
                },
                .mla_attention => if (self.instruction_mla_attention_refs[op_index]) |a| {
                    try writer.print(": MLA(n_heads={}, head_dim={})", .{ a.n_heads, a.head_dim });
                },
                .swiglu => if (self.instruction_swiglu_bindings[op_index]) |m| {
                    try writer.print(": MLP(d_ff={})", .{m.d_ff});
                },
                .moe => if (self.instruction_moe_bindings[op_index]) |e| {
                    try writer.print(": MoE(experts={}, per_tok={})", .{ e.num_experts, e.experts_per_token });
                },
                .mamba_mixer => if (self.instruction_mamba_bindings[op_index]) |m| {
                    try writer.print(": Mamba(d_model={}, d_state={}, d_conv={})", .{
                        m.config.d_model,
                        m.config.d_state,
                        m.config.d_conv,
                    });
                },
                .gated_delta_net => if (self.instruction_gated_delta_bindings[op_index]) |g| {
                    try writer.print(": GatedDelta(d_model={}, d_head={}, d_conv={})", .{
                        g.config.d_model,
                        g.config.d_head,
                        g.config.d_conv,
                    });
                },
                .shortconv => if (self.instruction_shortconv_bindings[op_index]) |s| {
                    try writer.print(": ShortConv(d_model={}, d_conv={})", .{ s.config.d_model, s.config.d_conv });
                },
                else => {},
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
            if (batchedDecodeUnsupportedOpcode(insn.opcode)) return false;
            if (batchedDecodeSupportedWithoutKernel(insn.opcode)) continue;
            if (runtime_contract.expectedKernelWeightSlots(insn.opcode).len != 0) {
                if (!self.hasTypedKernelBinding(insn.opcode, op_index)) return false;
                continue;
            }
            return false;
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
        try scratch.registerTmpLayout(self.tmp_slot_width_hints, self.tmp_slot_active);
        try scratch.ensureForMode(if (use_cache) .decode else .prefill, seq_len);

        const reg_count = self.compiled_plan.plan.register_count;
        var buffer_views_arr: [runtime_contract.max_register_count]Tensor = undefined;
        const buffer_views = buffer_views_arr[0..reg_count];
        buffer_views[0] = out.*;
        for (1..reg_count) |reg_idx| {
            const mapped = self.tmp_register_to_scratch_idx[reg_idx];
            buffer_views[reg_idx] = Tensor.view3DSlice(scratch.tmp[mapped], seq_len, self.hidden_size);
        }

        copyTensor(x, out);

        const slot_state = scratch.getSlotLayerState(slot_index, self.block_idx) orelse return error.InvalidState;
        var shared_state = SharedPersistentState{};
        const ctx = SlotContext{
            .slot_state_ptr = slot_state,
            .shared_state = &shared_state,
            .scratch = scratch,
            .use_cache = use_cache,
        };
        const instruction_handles = try scratch.allocator.alloc(
            runtime_contract.TensorHandle,
            self.instruction_handle_capacity,
        );
        defer if (instruction_handles.len > 0) scratch.allocator.free(instruction_handles);
        const instruction_views = try scratch.allocator.alloc(
            runtime_contract.TensorViewDesc,
            self.instruction_handle_capacity,
        );
        defer if (instruction_views.len > 0) scratch.allocator.free(instruction_views);

        var dispatch_state = RuntimeDispatchState{
            .block = self,
            .op_index = 0,
            .buffer_views = buffer_views,
            .scratch = scratch,
            .slot_ctx = ctx,
            .mode = .single_slot,
            .slot_index = slot_index,
            .slot_indices = &.{},
            .use_batched_dispatch = true,
            .bound_state_blocks = state_blocks,
            .instruction_handles = instruction_handles,
            .instruction_views = instruction_views,
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
        const final_reg_idx = runtime_contract.registerToIndex(
            runtime_contract.planFinalOutputRegister(&self.compiled_plan.plan),
        );
        if (final_reg_idx != 0) {
            copyTensor(&buffer_views[final_reg_idx], &buffer_views[0]);
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
        try scratch.registerTmpLayout(self.tmp_slot_width_hints, self.tmp_slot_active);
        try scratch.ensureForMode(if (use_cache) .decode else .prefill, batch_size);

        const reg_count = self.compiled_plan.plan.register_count;
        var buffer_views_arr: [runtime_contract.max_register_count]Tensor = undefined;
        const buffer_views = buffer_views_arr[0..reg_count];
        buffer_views[0] = out.*;
        for (1..reg_count) |reg_idx| {
            const mapped = self.tmp_register_to_scratch_idx[reg_idx];
            buffer_views[reg_idx] = Tensor.view3DSlice(scratch.tmp[mapped], batch_size, self.hidden_size);
        }

        copyTensor(x, out);

        const slot_state = scratch.getSlotState(self.block_idx) orelse return error.InvalidState;
        var shared_state = SharedPersistentState{};
        const ctx = SlotContext{
            .slot_state_ptr = slot_state,
            .shared_state = &shared_state,
            .scratch = scratch,
            .use_cache = use_cache,
        };
        const instruction_handles = try scratch.allocator.alloc(
            runtime_contract.TensorHandle,
            self.instruction_handle_capacity,
        );
        defer if (instruction_handles.len > 0) scratch.allocator.free(instruction_handles);
        const instruction_views = try scratch.allocator.alloc(
            runtime_contract.TensorViewDesc,
            self.instruction_handle_capacity,
        );
        defer if (instruction_views.len > 0) scratch.allocator.free(instruction_views);

        var dispatch_state = RuntimeDispatchState{
            .block = self,
            .op_index = 0,
            .buffer_views = buffer_views,
            .scratch = scratch,
            .slot_ctx = ctx,
            .mode = .slot_batch,
            .slot_index = 0,
            .slot_indices = slot_indices,
            .use_batched_dispatch = true,
            .bound_state_blocks = state_blocks,
            .instruction_handles = instruction_handles,
            .instruction_views = instruction_views,
        };
        try bindDispatchStateDescriptors(&dispatch_state);

        for (self.compiled_plan.plan.instructions, 0..) |insn, op_index| {
            dispatch_state.op_index = op_index;
            try self.dispatchInstructionWithState(&insn, &dispatch_state);
        }

        const final_reg_idx = runtime_contract.registerToIndex(
            runtime_contract.planFinalOutputRegister(&self.compiled_plan.plan),
        );
        if (final_reg_idx != 0) {
            copyTensor(&buffer_views[final_reg_idx], &buffer_views[0]);
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
        false,
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
        .{ .kernel = .{ .id = 1, .in = .norm_out, .out = .branch_out, .debug_type = .multihead_attention, .state_block_id = runtime_contract.kv_cache_state_id } },
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
        .{ .kernel = .{ .id = 1, .in = .norm_out, .out = .branch_out, .debug_type = .multihead_attention, .state_block_id = runtime_contract.kv_cache_state_id } },
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
        .{ .kernel = .{ .id = 1, .in = .norm_out, .out = .branch_out, .debug_type = .multihead_attention, .state_block_id = runtime_contract.kv_cache_state_id } },
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
    try testing.expectEqual(block.compiled_plan.plan.instructions.len + 1, block.instruction_weight_offsets.len);
    try testing.expectEqual(@as(usize, 1), block.instruction_weight_ptrs.len);
    try testing.expect(block.instruction_weight_ptrs[0] != null);
    try block.validate();
}

test "Block.forward linear fails when flattened weight table entry is missing" {
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

    // Flattened runtime weight pointers are the single execution source.
    block.instruction_weight_ptrs[0] = null;

    const seq_len = 2;
    const hidden = 128;
    const input_data = try allocator.alloc(f32, 1 * seq_len * hidden);
    defer allocator.free(input_data);
    for (input_data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i % 17)) * 0.05;
    }
    const input = Tensor.view(@ptrCast(input_data.ptr), &.{ 1, seq_len, hidden }, .f32, null);

    const output_data = try allocator.alloc(f32, 1 * seq_len * hidden);
    defer allocator.free(output_data);
    @memset(output_data, 0.0);
    var output = Tensor.view(@ptrCast(output_data.ptr), &.{ 1, seq_len, hidden }, .f32, null);

    var scratch = try ScratchBuffer.init(allocator, hidden, 512, 1);
    defer scratch.deinit();

    try testing.expectError(error.MissingWeight, block.forward(&input, &output, &scratch, false));
}

test "Block caches kernel instruction bindings at init" {
    const allocator = testing.allocator;

    var weights = try TestWeights.init(allocator, 128, 512, 2, 32);
    defer weights.deinit(allocator);

    var transformer_block = try createTestTransformerBlock(allocator, &weights);
    defer transformer_block.deinit(allocator);

    const program = [_]LayerOp{
        .{ .kernel = .{ .id = 0, .in = .residual, .out = .norm_out, .debug_type = .norm } },
        .{ .kernel = .{ .id = 1, .in = .norm_out, .out = .branch_out, .debug_type = .multihead_attention, .state_block_id = runtime_contract.kv_cache_state_id } },
    };

    var block = try createTestBlock(allocator, &transformer_block, 128, &program);
    defer block.deinit(allocator);

    try testing.expectEqual(block.compiled_plan.plan.instructions.len, block.instruction_norm_refs.len);
    try testing.expectEqual(block.compiled_plan.plan.instructions.len, block.instruction_attention_bindings.len);
    try testing.expect(block.instruction_norm_refs[0] != null);
    try testing.expect(block.instruction_attention_bindings[1] != null);
    try testing.expectEqual(block.compiled_plan.plan.instructions.len + 1, block.instruction_weight_offsets.len);
    try testing.expect(block.instruction_weight_ptrs.len != 0);
    try testing.expect(block.instruction_weight_ptrs[0] != null);
    try block.validate();
}

test "resolveKernelWeightPtrForSlot emits missing sentinels for fused attention split slots" {
    const fused_qkv = zeroTensor();
    var attention_binding: attn_kernel.MultiHeadAttention = undefined;
    attention_binding.q_proj = null;
    attention_binding.k_proj = null;
    attention_binding.v_proj = null;
    attention_binding.fused_qkv = fused_qkv;

    var norm = [_]?NormKernelBinding{null};
    var attention = [_]?AttentionKernelBinding{&attention_binding};
    var mla = [_]?MlaAttentionKernelBinding{null};
    var swiglu = [_]?SwiGLUKernelBinding{null};
    var moe = [_]?MoeKernelBinding{null};
    var mamba = [_]?MambaKernelBinding{null};
    var gated_delta = [_]?GatedDeltaKernelBinding{null};
    var shortconv = [_]?ShortConvKernelBinding{null};
    const typed = Block.TypedInstructionKernelRefs{
        .norm = norm[0..],
        .attention = attention[0..],
        .mla_attention = mla[0..],
        .swiglu = swiglu[0..],
        .moe = moe[0..],
        .mamba = mamba[0..],
        .gated_delta = gated_delta[0..],
        .shortconv = shortconv[0..],
    };
    const insn = runtime_contract.Instruction{
        .opcode = .multihead_attention,
        .inputs = &.{},
        .outputs = &.{},
        .weights = &.{},
        .param_block_id = null,
        .state_block_id = null,
    };

    const k_ptr = try Block.resolveKernelWeightPtrForSlot(0, 0, insn.opcode, typed, 1);
    const v_ptr = try Block.resolveKernelWeightPtrForSlot(0, 0, insn.opcode, typed, 2);
    try testing.expectEqual(
        @intFromPtr(&missing_weight_tensor),
        @intFromPtr(@as(*const Tensor, @ptrCast(@alignCast(k_ptr)))),
    );
    try testing.expectEqual(
        @intFromPtr(&missing_weight_tensor),
        @intFromPtr(@as(*const Tensor, @ptrCast(@alignCast(v_ptr)))),
    );
}

test "buildRuntimeMetadata preserves shortconv time-major weight tensor view" {
    const dk = try cpu_linalg.matmulKernel(.f32);
    const in_proj = zeroTensor();
    const conv_weight = zeroTensor();
    const out_proj = zeroTensor();
    var conv_weight_time_major_storage = [_]f32{ 1, 2, 3, 4, 5, 6 };

    var shortconv_binding = shortconv_kernel.ShortConvKernel{
        .config = .{
            .d_model = 4,
            .d_conv = 3,
            .conv_dim = 2,
            .conv_dim_out = 4,
            .has_bias = false,
        },
        .weights = .{
            .in_proj = &in_proj,
            .conv1d_weight = &conv_weight,
            .conv1d_bias = null,
            .out_proj = &out_proj,
        },
        .matmul_in_proj = dk.func,
        .matmul_out_proj = dk.func,
        .matmul_in_proj_name = dk.name,
        .matmul_out_proj_name = dk.name,
        .layer_idx = 0,
        .conv_weight_transposed = conv_weight_time_major_storage[0..],
        .weight_allocator = null,
    };

    var norm = [_]?NormKernelBinding{null};
    var attention = [_]?AttentionKernelBinding{null};
    var mla = [_]?MlaAttentionKernelBinding{null};
    var swiglu = [_]?SwiGLUKernelBinding{null};
    var moe = [_]?MoeKernelBinding{null};
    var mamba = [_]?MambaKernelBinding{null};
    var gated_delta = [_]?GatedDeltaKernelBinding{null};
    var shortconv = [_]?ShortConvKernelBinding{&shortconv_binding};
    const typed = Block.TypedInstructionKernelRefs{
        .norm = norm[0..],
        .attention = attention[0..],
        .mla_attention = mla[0..],
        .swiglu = swiglu[0..],
        .moe = moe[0..],
        .mamba = mamba[0..],
        .gated_delta = gated_delta[0..],
        .shortconv = shortconv[0..],
    };

    const plan = runtime_contract.ExecutionPlan{
        .instructions = &.{
            .{
                .opcode = .shortconv,
                .inputs = &.{},
                .outputs = &.{},
                .weights = &.{},
                .param_block_id = null,
                .state_block_id = null,
            },
        },
        .register_count = 0,
        .state_descs = &.{},
    };
    var runtime_meta = try Block.buildRuntimeMetadata(testing.allocator, typed, &plan);
    defer runtime_meta.deinit(testing.allocator);
    const shortconv_meta = runtime_meta.shortconv[0] orelse return error.TestUnexpectedResult;
    const time_major_tensor = shortconv_meta.conv_weight_time_major orelse return error.TestUnexpectedResult;
    const time_major_slice = time_major_tensor.asSlice(f32);
    try testing.expectEqual(conv_weight_time_major_storage.len, time_major_slice.len);
    try testing.expectEqual(@intFromPtr(conv_weight_time_major_storage[0..].ptr), @intFromPtr(time_major_slice.ptr));
}

test "shortConvTimeMajorWeightPtr returns pointer only for shortconv slot 1" {
    var conv_weight_time_major_storage = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const time_major_tensor = Tensor.view(
        @ptrCast(std.mem.sliceAsBytes(conv_weight_time_major_storage[0..]).ptr),
        &.{ 3, 2 },
        .f32,
        null,
    );

    var shortconv_entry: ShortConvRuntimeMetadata = undefined;
    shortconv_entry.conv_weight_time_major = time_major_tensor;
    var shortconv = [_]?ShortConvRuntimeMetadata{shortconv_entry};
    var norm = [_]?NormRuntimeMetadata{null};
    var attention = [_]?AttentionRuntimeMetadata{null};
    var mla = [_]?MlaRuntimeMetadata{null};
    var swiglu = [_]?SwiGluRuntimeMetadata{null};
    var moe = [_]?MoeRuntimeMetadata{null};
    var mamba = [_]?MambaRuntimeMetadata{null};
    var gated_delta = [_]?GatedDeltaRuntimeMetadata{null};
    const runtime_meta = Block.RuntimeMetadata{
        .norm = norm[0..],
        .attention = attention[0..],
        .mla = mla[0..],
        .swiglu = swiglu[0..],
        .moe = moe[0..],
        .mamba = mamba[0..],
        .gated_delta = gated_delta[0..],
        .shortconv = shortconv[0..],
    };

    const ptr = Block.shortConvTimeMajorWeightPtr(&runtime_meta, .shortconv, 0, 1) orelse {
        return error.TestUnexpectedResult;
    };
    const expected_ptr = blk: {
        if (shortconv[0]) |*meta| {
            if (meta.conv_weight_time_major) |*tensor_view| break :blk @intFromPtr(tensor_view);
        }
        return error.TestUnexpectedResult;
    };
    try testing.expectEqual(expected_ptr, @intFromPtr(@as(*const Tensor, @ptrCast(@alignCast(ptr)))));
    try testing.expect(Block.shortConvTimeMajorWeightPtr(&runtime_meta, .shortconv, 0, 0) == null);
    try testing.expect(Block.shortConvTimeMajorWeightPtr(&runtime_meta, .multihead_attention, 0, 1) == null);
}

test "buildRuntimeMetadata preserves gated-delta time-major conv weight view" {
    const dk = try cpu_linalg.matmulKernel(.f32);
    var conv_weight_time_major_storage = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const gated_delta_binding = gated_delta_kernel.GatedDeltaKernel{
        .config = .{
            .d_model = 4,
            .d_conv = 2,
            .n_heads = 1,
            .d_head = 1,
        },
        .weights = undefined,
        .matmul_in_proj = dk.func,
        .matmul_out_proj = dk.func,
        .layer_idx = 0,
        .conv_weight_transposed = conv_weight_time_major_storage[0..],
        .weight_allocator = null,
    };

    var norm = [_]?NormKernelBinding{null};
    var attention = [_]?AttentionKernelBinding{null};
    var mla = [_]?MlaAttentionKernelBinding{null};
    var swiglu = [_]?SwiGLUKernelBinding{null};
    var moe = [_]?MoeKernelBinding{null};
    var mamba = [_]?MambaKernelBinding{null};
    var gated_delta = [_]?GatedDeltaKernelBinding{&gated_delta_binding};
    var shortconv = [_]?ShortConvKernelBinding{null};
    const typed = Block.TypedInstructionKernelRefs{
        .norm = norm[0..],
        .attention = attention[0..],
        .mla_attention = mla[0..],
        .swiglu = swiglu[0..],
        .moe = moe[0..],
        .mamba = mamba[0..],
        .gated_delta = gated_delta[0..],
        .shortconv = shortconv[0..],
    };

    const plan = runtime_contract.ExecutionPlan{
        .instructions = &.{
            .{
                .opcode = .gated_delta_net,
                .inputs = &.{},
                .outputs = &.{},
                .weights = &.{},
                .param_block_id = null,
                .state_block_id = null,
            },
        },
        .register_count = 0,
        .state_descs = &.{},
    };
    var runtime_meta = try Block.buildRuntimeMetadata(testing.allocator, typed, &plan);
    defer runtime_meta.deinit(testing.allocator);
    const gated_delta_meta = runtime_meta.gated_delta[0] orelse return error.TestUnexpectedResult;
    const time_major_tensor = gated_delta_meta.conv_weight_time_major orelse return error.TestUnexpectedResult;
    const time_major_slice = time_major_tensor.asSlice(f32);
    try testing.expectEqual(conv_weight_time_major_storage.len, time_major_slice.len);
    try testing.expectEqual(@intFromPtr(conv_weight_time_major_storage[0..].ptr), @intFromPtr(time_major_slice.ptr));
}

test "resolveKernelWeightPtrForSlot routes layernorm bias via norm_bias slot" {
    const norm_weight = zeroTensor();
    const norm_bias = zeroTensor();
    var layer_norm = norm_kernel.NormKernel{
        .layer = .{
            .weight = &norm_weight,
            .bias = &norm_bias,
            .dim = 16,
            .eps = 1e-5,
        },
    };

    var norm = [_]?NormKernelBinding{&layer_norm};
    var attention = [_]?AttentionKernelBinding{null};
    var mla = [_]?MlaAttentionKernelBinding{null};
    var swiglu = [_]?SwiGLUKernelBinding{null};
    var moe = [_]?MoeKernelBinding{null};
    var mamba = [_]?MambaKernelBinding{null};
    var gated_delta = [_]?GatedDeltaKernelBinding{null};
    var shortconv = [_]?ShortConvKernelBinding{null};
    const typed = Block.TypedInstructionKernelRefs{
        .norm = norm[0..],
        .attention = attention[0..],
        .mla_attention = mla[0..],
        .swiglu = swiglu[0..],
        .moe = moe[0..],
        .mamba = mamba[0..],
        .gated_delta = gated_delta[0..],
        .shortconv = shortconv[0..],
    };

    const weight_ptr = try Block.resolveKernelWeightPtrForSlot(0, 0, .rmsnorm, typed, 0);
    try testing.expectEqual(@intFromPtr(&norm_weight), @intFromPtr(@as(*const Tensor, @ptrCast(@alignCast(weight_ptr)))));
    const bias_ptr = try Block.resolveKernelWeightPtrForSlot(0, 0, .rmsnorm, typed, 1);
    try testing.expectEqual(@intFromPtr(&norm_bias), @intFromPtr(@as(*const Tensor, @ptrCast(@alignCast(bias_ptr)))));
}

test "resolveKernelWeightPtrForSlot emits missing sentinel for dense swiglu up slot" {
    const gate = zeroTensor();
    const down = zeroTensor();
    var swiglu_binding: ffn_kernel.SwiGLU = undefined;
    swiglu_binding.w1 = &gate;
    swiglu_binding.w2 = &down;
    swiglu_binding.w3 = null;
    swiglu_binding.fused_gate_up = null;

    var norm = [_]?NormKernelBinding{null};
    var attention = [_]?AttentionKernelBinding{null};
    var mla = [_]?MlaAttentionKernelBinding{null};
    var swiglu = [_]?SwiGLUKernelBinding{&swiglu_binding};
    var moe = [_]?MoeKernelBinding{null};
    var mamba = [_]?MambaKernelBinding{null};
    var gated_delta = [_]?GatedDeltaKernelBinding{null};
    var shortconv = [_]?ShortConvKernelBinding{null};
    const typed = Block.TypedInstructionKernelRefs{
        .norm = norm[0..],
        .attention = attention[0..],
        .mla_attention = mla[0..],
        .swiglu = swiglu[0..],
        .moe = moe[0..],
        .mamba = mamba[0..],
        .gated_delta = gated_delta[0..],
        .shortconv = shortconv[0..],
    };
    const insn = runtime_contract.Instruction{
        .opcode = .swiglu,
        .inputs = &.{},
        .outputs = &.{},
        .weights = &.{},
        .param_block_id = null,
        .state_block_id = null,
    };

    const up_ptr = try Block.resolveKernelWeightPtrForSlot(0, 0, insn.opcode, typed, 1);
    try testing.expectEqual(
        @intFromPtr(&missing_weight_tensor),
        @intFromPtr(@as(*const Tensor, @ptrCast(@alignCast(up_ptr)))),
    );
}

test "resolveKernelWeightPtrForSlot rejects multi-expert moe bindings" {
    const router = zeroTensor();
    const gate0 = zeroTensor();
    const up0 = zeroTensor();
    const down0 = zeroTensor();
    const gate1 = zeroTensor();
    const up1 = zeroTensor();
    const down1 = zeroTensor();
    var experts = [_]moe_kernel.ExpertWeights{
        .{ .gate_proj = gate0, .up_proj = up0, .down_proj = down0 },
        .{ .gate_proj = gate1, .up_proj = up1, .down_proj = down1 },
    };
    var moe_binding = moe_kernel.MoEFFN{
        .allocator = testing.allocator,
        .d_model = 16,
        .d_ff = 32,
        .num_experts = experts.len,
        .experts_per_token = 1,
        .router_weight = router,
        .router_bias = null,
        .experts = experts[0..],
    };

    var norm = [_]?NormKernelBinding{null};
    var attention = [_]?AttentionKernelBinding{null};
    var mla = [_]?MlaAttentionKernelBinding{null};
    var swiglu = [_]?SwiGLUKernelBinding{null};
    var moe = [_]?MoeKernelBinding{&moe_binding};
    var mamba = [_]?MambaKernelBinding{null};
    var gated_delta = [_]?GatedDeltaKernelBinding{null};
    var shortconv = [_]?ShortConvKernelBinding{null};
    const typed = Block.TypedInstructionKernelRefs{
        .norm = norm[0..],
        .attention = attention[0..],
        .mla_attention = mla[0..],
        .swiglu = swiglu[0..],
        .moe = moe[0..],
        .mamba = mamba[0..],
        .gated_delta = gated_delta[0..],
        .shortconv = shortconv[0..],
    };

    try testing.expectError(
        error.UnsupportedModel,
        Block.resolveKernelWeightPtrForSlot(0, 0, .moe, typed, 0),
    );
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

    const refs_mut = @constCast(block.instruction_norm_refs.ptr);
    refs_mut[0] = null;
    const metadata_mut = @constCast(block.instruction_norm_runtime_metadata.ptr);
    metadata_mut[0] = null;

    try testing.expectError(error.KernelIndexOutOfBounds, block.validate());
}

test "cpu adapter table rejects moe opcode at load-time validation" {
    const plan = runtime_contract.ExecutionPlan{
        .instructions = &.{
            .{
                .opcode = .moe,
                .inputs = &.{},
                .outputs = &.{},
                .weights = &.{},
                .param_block_id = null,
                .state_block_id = null,
            },
        },
        .register_count = 0,
        .state_descs = &.{},
    };
    const unsupported = runtime_contract.firstUnsupportedInstructionOpcode(&plan, Block.adapter_table);
    try testing.expect(unsupported != null);
    try testing.expectEqual(@as(usize, 0), unsupported.?.instruction_index);
    try testing.expectEqual(runtime_contract.Opcode.moe, unsupported.?.opcode);
}

test "Block.validate rejects missing flattened primitive weight binding" {
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

    const ptrs_mut = @constCast(block.instruction_weight_ptrs.ptr);
    ptrs_mut[0] = null;

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

test "Block.validate rejects stateful opcode for wrong block kind" {
    const allocator = testing.allocator;

    var weights = try TestWeights.init(allocator, 128, 512, 2, 32);
    defer weights.deinit(allocator);

    var transformer_block = try createTestTransformerBlock(allocator, &weights);
    defer transformer_block.deinit(allocator);

    const program = [_]LayerOp{
        .{ .kernel = .{ .id = 0, .in = .residual, .out = .norm_out, .debug_type = .shortconv, .state_block_id = runtime_contract.shortconv_state_id } },
    };

    try testing.expectError(error.InvalidInstructionBinding, createTestBlock(allocator, &transformer_block, 128, &program));
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

    try testing.expectError(error.UnknownStateDescriptorId, block.validate());
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

test "Block handles large split output count dynamically" {
    const allocator = testing.allocator;

    var weights = try TestWeights.init(allocator, 128, 512, 2, 32);
    defer weights.deinit(allocator);

    var transformer_block = try createTestTransformerBlock(allocator, &weights);
    defer transformer_block.deinit(allocator);

    // Split starting at tmp3 with 64 outputs: residual(0) + norm_out(1) + 64 splits = 66 registers.
    // With dynamic slot arrays, this succeeds — no fixed cap.
    const program = [_]LayerOp{
        .{ .split = .{ .in = .norm_out, .out_start = .tmp3, .num_outputs = 64, .split_sizes = &.{}, .dim = -1 } },
    };

    var block = try createTestBlock(allocator, &transformer_block, 128, &program);
    defer block.deinit(allocator);
    try testing.expect(block.tmp_slot_active.len >= 65); // 1 + 64 physical slots
}

test "Block.forward executes simple norm-attn-add program" {
    const allocator = testing.allocator;

    var weights = try TestWeights.init(allocator, 128, 512, 2, 32);
    defer weights.deinit(allocator);

    var transformer_block = try createTestTransformerBlock(allocator, &weights);
    defer transformer_block.deinit(allocator);

    const program = [_]LayerOp{
        .{ .kernel = .{ .id = 0, .in = .residual, .out = .norm_out, .debug_type = .norm } },
        .{ .kernel = .{ .id = 1, .in = .norm_out, .out = .branch_out, .debug_type = .multihead_attention, .state_block_id = runtime_contract.kv_cache_state_id } },
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

    var layered_cache = try LayeredBatchedKVCache.init(allocator, 1, 4, 2, 32, 2048);
    defer layered_cache.deinit();
    var state_storage align(64) = [_]u8{0} ** @intCast(runtime_contract.builtin_state_block_bytes);
    var state_block = runtime_contract.StateBlockHandle{
        .id = @intFromEnum(runtime_contract.StateBlockId.kv_cache),
        .ptr = @ptrCast(&state_storage),
        .size = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
    };
    const state_value0 = runtime_contract.stateValueFromBlock(*state_bindings.KvRuntimeState, &state_block) orelse {
        return error.TestUnexpectedResult;
    };
    state_value0.* = .{
        .runtime_kind = runtime_contract.state_runtime_kind_kv_cache,
        .layered_cache = &layered_cache,
        .scratch = &scratch,
        .slot_index = 0,
    };
    const state_blocks = [_]runtime_contract.StateBlockHandle{state_block};

    // Execute forward pass
    try block.forwardWithBatchedCache(&input, &output, &scratch, state_blocks[0..], 0, false);

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
        .{ .kernel = .{ .id = 1, .in = .norm_out, .out = .branch_out, .debug_type = .multihead_attention, .state_block_id = runtime_contract.kv_cache_state_id } },
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

    var layered_cache = try LayeredBatchedKVCache.init(allocator, 1, 4, 2, 32, 2048);
    defer layered_cache.deinit();
    var state_storage align(64) = [_]u8{0} ** @intCast(runtime_contract.builtin_state_block_bytes);
    var state_block = runtime_contract.StateBlockHandle{
        .id = @intFromEnum(runtime_contract.StateBlockId.kv_cache),
        .ptr = @ptrCast(&state_storage),
        .size = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
    };
    const state_value1 = runtime_contract.stateValueFromBlock(*state_bindings.KvRuntimeState, &state_block) orelse {
        return error.TestUnexpectedResult;
    };
    state_value1.* = .{
        .runtime_kind = runtime_contract.state_runtime_kind_kv_cache,
        .layered_cache = &layered_cache,
        .scratch = &scratch,
        .slot_index = 0,
    };
    const state_blocks = [_]runtime_contract.StateBlockHandle{state_block};

    // Execute forward pass
    try block.forwardWithBatchedCache(&input, &output, &scratch, state_blocks[0..], 0, false);

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
        .{ .kernel = .{ .id = 1, .in = .norm_out, .out = .branch_out, .debug_type = .multihead_attention, .state_block_id = runtime_contract.kv_cache_state_id } },
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

    // Create layered KV cache state descriptor binding for block 0.
    var layered_cache = try LayeredBatchedKVCache.init(allocator, 1, 4, 2, 32, 2048);
    defer layered_cache.deinit();
    var state_storage align(64) = [_]u8{0} ** @intCast(runtime_contract.builtin_state_block_bytes);
    var state_block = runtime_contract.StateBlockHandle{
        .id = @intFromEnum(runtime_contract.StateBlockId.kv_cache),
        .ptr = @ptrCast(&state_storage),
        .size = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
    };
    const state_value2 = runtime_contract.stateValueFromBlock(*state_bindings.KvRuntimeState, &state_block) orelse {
        return error.TestUnexpectedResult;
    };
    state_value2.* = .{
        .runtime_kind = runtime_contract.state_runtime_kind_kv_cache,
        .layered_cache = &layered_cache,
        .scratch = &scratch,
        .slot_index = 0,
    };
    const state_blocks = [_]runtime_contract.StateBlockHandle{state_block};

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

    try block.forwardWithBatchedCache(&input, &output, &scratch, &.{}, 0, false);

    for (output_data, 0..) |val, i| {
        try testing.expectApproxEqAbs(input_data[i] * 0.5, val, 1e-6);
    }
}

test "Block.forwardWithBatchedCache rejects missing descriptor state blocks for attention" {
    const allocator = testing.allocator;

    var weights = try TestWeights.init(allocator, 128, 512, 2, 32);
    defer weights.deinit(allocator);

    var transformer_block = try createTestTransformerBlock(allocator, &weights);
    defer transformer_block.deinit(allocator);

    const program = [_]LayerOp{
        .{ .kernel = .{ .id = 0, .in = .residual, .out = .norm_out, .debug_type = .norm } },
        .{ .kernel = .{ .id = 1, .in = .norm_out, .out = .branch_out, .debug_type = .multihead_attention, .state_block_id = runtime_contract.kv_cache_state_id } },
        .{ .add = .{ .branch = .branch_out, .scale = .one } },
    };

    var block = try createTestBlock(allocator, &transformer_block, 128, &program);
    defer block.deinit(allocator);

    const input_data = try allocator.alloc(f32, 1 * 2 * 128);
    defer allocator.free(input_data);
    @memset(input_data, 1.0);
    const input = Tensor.view(@ptrCast(input_data.ptr), &.{ 1, 2, 128 }, .f32, null);

    const output_data = try allocator.alloc(f32, 1 * 2 * 128);
    defer allocator.free(output_data);
    @memset(output_data, 0.0);
    var output = Tensor.view(@ptrCast(output_data.ptr), &.{ 1, 2, 128 }, .f32, null);

    var scratch = try ScratchBuffer.init(allocator, 128, 512, 1);
    defer scratch.deinit();

    try testing.expectError(
        error.InvalidStateDescriptorBinding,
        block.forwardWithBatchedCache(&input, &output, &scratch, &.{}, 0, true),
    );
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

    const tmp_layout = try buildTmpRegisterScratchMap(allocator, &compiled);
    const tmp_map = tmp_layout.map;
    try testing.expectEqual(tmp_map[@intFromEnum(BufferId.tmp3)], tmp_map[@intFromEnum(BufferId.tmp5)]);
    try testing.expect(tmp_map[@intFromEnum(BufferId.tmp4)] != tmp_map[@intFromEnum(BufferId.tmp3)]);
    try testing.expect(tmp_map[@intFromEnum(BufferId.tmp3)] >= 1);
}

test "hiddenWidthFromTensor returns trailing hidden dimension" {
    const vec_data = [_]f32{ 0, 0, 0, 0 };
    const vec = Tensor.view(@ptrCast(@constCast(&vec_data)), &.{4}, .f32, null);
    try testing.expectEqual(@as(u32, 4), try Block.hiddenWidthFromTensor(&vec));

    const mat_data = [_]f32{0} ** (3 * 5);
    const mat = Tensor.view(@ptrCast(@constCast(&mat_data)), &.{ 3, 5 }, .f32, null);
    try testing.expectEqual(@as(u32, 5), try Block.hiddenWidthFromTensor(&mat));

    const batched_data = [_]f32{0} ** (2 * 3 * 7);
    const batched = Tensor.view(@ptrCast(@constCast(&batched_data)), &.{ 2, 3, 7 }, .f32, null);
    try testing.expectEqual(@as(u32, 7), try Block.hiddenWidthFromTensor(&batched));
}
