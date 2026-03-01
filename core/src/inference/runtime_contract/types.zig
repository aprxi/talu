//! Typed runtime contracts for generic inference execution plans.
//!
//! This module contains backend-agnostic plan/execution ABI types.

const std = @import("std");
const dtype = @import("../../dtype.zig");
const op_types = @import("../../models/op_types.zig");
const layer_ops = @import("../../models/layer_ops.zig");
const opcode_map = @import("../../models/plan/opcode_map.zig");
const opcode_mod = @import("../../models/plan/opcode.zig");

pub const Opcode = opcode_mod.Opcode;

pub const RegisterRef = enum(u16) { _ };

pub fn registerFromIndex(index: u16) RegisterRef {
    return @enumFromInt(index);
}

pub fn registerToIndex(register: RegisterRef) u16 {
    return @intFromEnum(register);
}

pub const WeightRef = struct {
    index: u32,
};

pub const WeightBinding = struct {
    index: u32,
    name: []const u8,
};

pub const StateLifecycle = enum(u8) {
    slot_persistent = 0,
    request_scoped = 1,
    step_scoped = 2,
};

pub const StateLifecycleAction = enum(u8) {
    alloc = 0,
    reset = 1,
    reuse = 2,
    evict = 3,
    clone_for_fork = 4,
    deinit = 5,
};

pub const StateLifecyclePolicy = struct {
    allow_reset: bool,
    allow_reuse: bool,
    allow_evict: bool,
    allow_clone_for_fork: bool,
    zero_on_alloc: bool,
    zero_on_reset: bool,
    zero_on_evict: bool,
};

pub fn stateLifecyclePolicy(lifecycle: StateLifecycle) StateLifecyclePolicy {
    return switch (lifecycle) {
        .slot_persistent => .{
            .allow_reset = true,
            .allow_reuse = true,
            .allow_evict = true,
            .allow_clone_for_fork = false,
            .zero_on_alloc = false,
            .zero_on_reset = false,
            .zero_on_evict = false,
        },
        .request_scoped => .{
            .allow_reset = false,
            .allow_reuse = false,
            .allow_evict = true,
            .allow_clone_for_fork = false,
            .zero_on_alloc = true,
            .zero_on_reset = false,
            .zero_on_evict = true,
        },
        .step_scoped => .{
            .allow_reset = true,
            .allow_reuse = false,
            .allow_evict = true,
            .allow_clone_for_fork = false,
            .zero_on_alloc = true,
            .zero_on_reset = true,
            .zero_on_evict = true,
        },
    };
}

pub fn validateStateLifecycleAction(
    lifecycle: StateLifecycle,
    action: StateLifecycleAction,
) !void {
    const policy = stateLifecyclePolicy(lifecycle);
    switch (action) {
        .alloc, .deinit => return,
        .reset => if (!policy.allow_reset) return error.InvalidStateLifecycleAction,
        .reuse => if (!policy.allow_reuse) return error.InvalidStateLifecycleAction,
        .evict => if (!policy.allow_evict) return error.InvalidStateLifecycleAction,
        .clone_for_fork => if (!policy.allow_clone_for_fork) return error.InvalidStateLifecycleAction,
    }
}

pub fn shouldZeroStateForLifecycleAction(
    descriptor: *const StateDescriptor,
    action: StateLifecycleAction,
) !bool {
    try validateStateLifecycleAction(descriptor.lifecycle, action);
    const policy = stateLifecyclePolicy(descriptor.lifecycle);
    return switch (action) {
        .alloc => descriptor.zero_init or policy.zero_on_alloc,
        .reset => descriptor.zero_init or policy.zero_on_reset,
        .evict => descriptor.zero_init or policy.zero_on_evict,
        .reuse, .clone_for_fork, .deinit => false,
    };
}

pub const StateDescriptor = struct {
    id: u8,
    size_bytes: u64,
    align_bytes: u16,
    zero_init: bool,
    lifecycle: StateLifecycle,
};

pub const StateBlockId = enum(u8) {
    kv_cache = 0,
    shortconv = 1,
    mamba = 2,
};

pub const Instruction = struct {
    opcode: Opcode,
    inputs: []const RegisterRef,
    outputs: []const RegisterRef,
    weights: []const WeightRef,
    param_block_id: ?u16,
    state_block_id: ?u8,
};

pub const ExecutionPlan = struct {
    instructions: []const Instruction,
    register_count: u16,
    state_descs: []const StateDescriptor,
};

pub const LivenessMap = struct {
    /// register -> last instruction index that reads it.
    register_last_read: []const u32,
    /// instruction -> bitset words of registers that die after this instruction.
    kill_after_instruction: []const []const u64,

    pub fn bitsetWordCount(register_count: u16) usize {
        return (@as(usize, register_count) + 63) / 64;
    }
};

pub const PlanDiagnosticLevel = enum(u8) {
    info = 0,
    warn = 1,
};

pub const PlanDiagnostic = struct {
    level: PlanDiagnosticLevel,
    message: []const u8,
};

pub const CompiledPlan = struct {
    plan: ExecutionPlan,
    param_blocks: []const ParamBlock,
    weight_bindings: []const WeightBinding = &.{},
    register_buffer_specs: []const PhysicalBufferSpec = &.{},
    /// Maps register index → legacy BufferId value. Populated by the compiler
    /// during allocation-order register assignment. Backends use this to resolve
    /// which physical buffer a register represents (e.g., residual vs scratch).
    register_to_buffer_id: []const u8 = &.{},
    liveness: LivenessMap,
    peak_registers: u16,
    diagnostics: []const PlanDiagnostic,
};

pub const PhysicalBufferSpec = struct {
    size: usize,
    @"align": u16,
};

pub const PhysicalMapping = struct {
    register_to_physical: []const u16,
    physical_count: u16,
    physical_specs: []const PhysicalBufferSpec,
};

pub const TensorHandle = struct {
    register: RegisterRef,
    ptr: *anyopaque,
};

pub const TensorLayout = enum(u8) {
    contiguous = 0,
    strided = 1,
    backend_native = 2,
};

pub const TensorViewDesc = struct {
    dtype: dtype.DType,
    rank: u8,
    // v1 runtime plan compatibility cap; fail validation when rank exceeds 4.
    shape: [4]u32,
    stride_elems: [4]u32,
    layout: TensorLayout,
};

pub const StateBlockHandle = struct {
    id: u8,
    ptr: [*]align(64) u8,
    size: u64,
    align_bytes: u16,
};

/// Backend-owned opaque state reference carried inside StateBlockHandle.ptr.
/// Backends populate this with pointers to concrete runtime state objects.
pub const OpaqueStateRef = extern struct {
    ptr: ?*anyopaque = null,
};

pub const ParamBlock = struct {
    version: u8,
    opcode: Opcode,
    data: []align(8) const u8,
};

pub const param_block_abi_version_v1: u8 = 1;
pub const max_param_block_data_bytes_v1: usize = 256;
pub const tensor_view_rank_cap_v1: u8 = 4;

pub const ExecutionMode = enum(u8) {
    decode = 0,
    prefill = 1,
    vision_encode = 2,
    scatter = 3,
};

pub const DispatchCounters = struct {
    per_opcode: [256]std.atomic.Value(u64) = [_]std.atomic.Value(u64){std.atomic.Value(u64).init(0)} ** 256,
    total_instructions: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),

    pub fn reset(self: *DispatchCounters) void {
        for (&self.per_opcode) |*counter| {
            counter.store(0, .monotonic);
        }
        self.total_instructions.store(0, .monotonic);
    }

    pub fn record(self: *DispatchCounters, opcode: Opcode) void {
        const idx: usize = @intFromEnum(opcode);
        _ = self.per_opcode[idx].fetchAdd(1, .monotonic);
        _ = self.total_instructions.fetchAdd(1, .monotonic);
    }

    pub fn total(self: *const DispatchCounters) u64 {
        return self.total_instructions.load(.monotonic);
    }

    pub fn countForOpcode(self: *const DispatchCounters, opcode: Opcode) u64 {
        const idx: usize = @intFromEnum(opcode);
        return self.per_opcode[idx].load(.monotonic);
    }
};

pub const Workspace = struct {
    any: ?*anyopaque = null,
    matmul: ?*anyopaque = null,
};

pub const ExecutionContext = struct {
    mode: ExecutionMode,
    active_slots: []const usize,
    sequence_lengths: []const u32,
    batch_size: usize,
    stream_or_queue: ?*anyopaque = null,
    dispatch_counters: ?*DispatchCounters = null,
    workspace: Workspace = .{},
};

pub fn recordExecutionDispatch(ctx: *ExecutionContext, opcode: Opcode) void {
    if (ctx.dispatch_counters) |counters| {
        counters.record(opcode);
    }
}

pub const KernelAdapterFn = *const fn (
    ctx: *ExecutionContext,
    insn: *const Instruction,
    registers: []TensorHandle,
    register_views: []const TensorViewDesc,
    state_blocks: []StateBlockHandle,
    params: []const ParamBlock,
) anyerror!void;

pub const AdapterTable = [256]?KernelAdapterFn;

pub const AdapterCapability = struct {
    supports_batch: bool,
    supports_graph_emit: bool,
    max_batch_size: ?usize,
};

pub const AdapterCapabilities = [256]AdapterCapability;

pub fn validateBatchCapability(capability: AdapterCapability, batch_size: usize) !void {
    if (batch_size == 0) return error.InvalidBatchSize;
    if (!capability.supports_batch and batch_size > 1) return error.UnsupportedBatchSize;
    if (capability.max_batch_size) |max_batch_size| {
        if (batch_size > max_batch_size) return error.UnsupportedBatchSize;
    }
}

pub const UnsupportedLayerProgramOpcode = struct {
    op_index: usize,
    opcode: Opcode,
};

pub const UnsupportedInstructionOpcode = struct {
    instruction_index: usize,
    opcode: Opcode,
};

pub const LayerProgramCompatibilityIssue = union(enum) {
    unsupported_opcode: UnsupportedLayerProgramOpcode,
    state_mismatch: LayerProgramStateMismatch,
};

fn adapterSlotPresent(slot: anytype) bool {
    return switch (@typeInfo(@TypeOf(slot))) {
        .optional => slot != null,
        else => @compileError("adapter table slot type must be optional"),
    };
}

pub fn assertAdapterTableCoverage(
    comptime table: anytype,
    comptime required_opcodes: anytype,
    comptime owner_name: []const u8,
) void {
    const table_type = @TypeOf(table);
    const table_info = @typeInfo(table_type);
    if (table_info != .array) {
        @compileError("Contract for '" ++ owner_name ++ "' requires a fixed array adapter table");
    }
    if (table_info.array.len != 256) {
        @compileError("Contract for '" ++ owner_name ++ "' adapter table must have 256 slots");
    }
    if (@typeInfo(table_info.array.child) != .optional) {
        @compileError("Contract for '" ++ owner_name ++ "' adapter table elements must be optional");
    }

    inline for (required_opcodes) |opcode| {
        const idx: usize = @intCast(@intFromEnum(opcode));
        if (idx >= table_info.array.len) {
            @compileError("Contract for '" ++ owner_name ++ "' opcode index out of bounds");
        }
        if (!adapterSlotPresent(table[idx])) {
            @compileError(
                "Contract for '" ++ owner_name ++ "' missing adapter for opcode '" ++ @tagName(opcode) ++ "'",
            );
        }
    }
}

pub fn firstUnsupportedLayerProgramOpcode(program: []const layer_ops.LayerOp, adapter_table: anytype) ?UnsupportedLayerProgramOpcode {
    for (program, 0..) |op, op_index| {
        const opcode = opcode_map.opcodeForLayerOp(op);
        if (!adapterSlotPresent(adapter_table[@intFromEnum(opcode)])) {
            return .{
                .op_index = op_index,
                .opcode = opcode,
            };
        }
    }
    return null;
}

pub fn firstUnsupportedInstructionOpcode(plan: *const ExecutionPlan, adapter_table: anytype) ?UnsupportedInstructionOpcode {
    for (plan.instructions, 0..) |insn, instruction_index| {
        if (!adapterSlotPresent(adapter_table[@intFromEnum(insn.opcode)])) {
            return .{
                .instruction_index = instruction_index,
                .opcode = insn.opcode,
            };
        }
    }
    return null;
}

pub fn firstLayerProgramCompatibilityIssue(
    program: []const layer_ops.LayerOp,
    kind: op_types.BlockKind,
    adapter_table: anytype,
) ?LayerProgramCompatibilityIssue {
    if (firstUnsupportedLayerProgramOpcode(program, adapter_table)) |unsupported| {
        return .{ .unsupported_opcode = unsupported };
    }
    if (firstLayerProgramStateMismatch(program, kind)) |mismatch| {
        return .{ .state_mismatch = mismatch };
    }
    return null;
}

pub fn stateBlockIdForOpcode(opcode: Opcode) ?u8 {
    return switch (opcode) {
        .multihead_attention, .mla_attention => @intFromEnum(StateBlockId.kv_cache),
        .shortconv => @intFromEnum(StateBlockId.shortconv),
        .mamba_mixer => @intFromEnum(StateBlockId.mamba),
        else => null,
    };
}

pub fn requiredStateBlockIdForOpcode(opcode: Opcode) ?u8 {
    return switch (opcode) {
        .shortconv => @intFromEnum(StateBlockId.shortconv),
        .mamba_mixer => @intFromEnum(StateBlockId.mamba),
        else => null,
    };
}

pub fn expectedKernelWeightSlots(opcode: Opcode) []const []const u8 {
    return switch (opcode) {
        .rmsnorm => &.{"norm_weight"},
        .multihead_attention => &.{ "q_proj", "k_proj", "v_proj", "o_proj" },
        .swiglu => &.{ "w1", "w3", "w2" },
        .moe => &.{"router"},
        .mamba_mixer => &.{ "in_proj", "out_proj" },
        .shortconv => &.{ "in_proj", "conv_weight", "out_proj" },
        .mla_attention => &.{"mla_weights"},
        .embedding => &.{"embedding"},
        else => &.{},
    };
}

pub fn expectedWeightRefCount(opcode: Opcode) usize {
    const kernel_slots = expectedKernelWeightSlots(opcode);
    if (kernel_slots.len != 0) return kernel_slots.len;
    return switch (opcode) {
        .linear, .add_param, .add_param_scalar, .mul_param => 1,
        else => 0,
    };
}

pub const kernel_weight_binding_prefix = "__kernel_weight::";

pub const KernelWeightBindingName = struct {
    kernel_id: u32,
    slot_name: []const u8,
};

pub fn parseKernelWeightBindingName(name: []const u8) !KernelWeightBindingName {
    if (!std.mem.startsWith(u8, name, kernel_weight_binding_prefix)) return error.InvalidWeightBindingName;
    const remainder = name[kernel_weight_binding_prefix.len..];
    const first_sep = std.mem.indexOf(u8, remainder, "::") orelse return error.InvalidWeightBindingName;
    if (first_sep == 0) return error.InvalidWeightBindingName;
    const kernel_id = std.fmt.parseUnsigned(u32, remainder[0..first_sep], 10) catch {
        return error.InvalidWeightBindingName;
    };
    const slot_and_tail = remainder[first_sep + 2 ..];
    const second_sep = std.mem.indexOf(u8, slot_and_tail, "::") orelse return error.InvalidWeightBindingName;
    if (second_sep == 0) return error.InvalidWeightBindingName;
    const slot_name = slot_and_tail[0..second_sep];
    const instruction_suffix = slot_and_tail[second_sep + 2 ..];
    if (instruction_suffix.len == 0) return error.InvalidWeightBindingName;
    _ = std.fmt.parseUnsigned(u32, instruction_suffix, 10) catch return error.InvalidWeightBindingName;
    return .{
        .kernel_id = kernel_id,
        .slot_name = slot_name,
    };
}

pub fn blockKindSupportsState(kind: op_types.BlockKind, state_id: u8) bool {
    return switch (kind) {
        .attention_mlp => state_id == @intFromEnum(StateBlockId.kv_cache),
        .shortconv => state_id == @intFromEnum(StateBlockId.shortconv),
        .mamba => state_id == @intFromEnum(StateBlockId.mamba),
    };
}

pub fn opcodeStateCompatibleWithBlockKind(opcode: Opcode, kind: op_types.BlockKind) bool {
    const state_id = stateBlockIdForOpcode(opcode) orelse return true;
    return blockKindSupportsState(kind, state_id);
}

pub const LayerProgramStateMismatch = struct {
    op_index: usize,
    opcode: Opcode,
    state_id: u8,
};

pub fn firstLayerProgramStateMismatch(
    program: []const layer_ops.LayerOp,
    kind: op_types.BlockKind,
) ?LayerProgramStateMismatch {
    for (program, 0..) |op, op_index| {
        const opcode = opcode_map.opcodeForLayerOp(op);
        if (!opcodeStateCompatibleWithBlockKind(opcode, kind)) {
            return .{
                .op_index = op_index,
                .opcode = opcode,
                .state_id = stateBlockIdForOpcode(opcode).?,
            };
        }
    }
    return null;
}

pub fn defaultStateDescriptor(state_id: StateBlockId) StateDescriptor {
    return switch (state_id) {
        .kv_cache => .{
            .id = @intFromEnum(StateBlockId.kv_cache),
            .size_bytes = 0,
            .align_bytes = 64,
            .zero_init = false,
            .lifecycle = .slot_persistent,
        },
        .shortconv => .{
            .id = @intFromEnum(StateBlockId.shortconv),
            .size_bytes = 0,
            .align_bytes = 64,
            .zero_init = true,
            .lifecycle = .slot_persistent,
        },
        .mamba => .{
            .id = @intFromEnum(StateBlockId.mamba),
            .size_bytes = 0,
            .align_bytes = 64,
            .zero_init = true,
            .lifecycle = .slot_persistent,
        },
    };
}

pub fn planHasStateDescriptor(plan: *const ExecutionPlan, state_id: u8) bool {
    for (plan.state_descs) |desc| {
        if (desc.id == state_id) return true;
    }
    return false;
}

pub fn stateDescriptorIndex(descriptors: []const StateDescriptor, state_id: u8) ?usize {
    for (descriptors, 0..) |descriptor, idx| {
        if (descriptor.id == state_id) return idx;
    }
    return null;
}

pub fn stateDescriptorSlicesEqual(a: StateDescriptor, b: StateDescriptor) bool {
    return a.id == b.id and
        a.size_bytes == b.size_bytes and
        a.align_bytes == b.align_bytes and
        a.zero_init == b.zero_init and
        a.lifecycle == b.lifecycle;
}

pub fn appendUniqueStateDescriptor(
    storage: []StateDescriptor,
    count: *u8,
    descriptor: StateDescriptor,
) !void {
    const used_count: usize = @intCast(count.*);
    const used = storage[0..used_count];
    if (stateDescriptorIndex(used, descriptor.id)) |idx| {
        if (!stateDescriptorSlicesEqual(used[idx], descriptor)) {
            return error.InvalidStateDescriptorBinding;
        }
        return;
    }
    if (used_count >= storage.len) return error.InvalidStateDescriptorBinding;
    storage[used_count] = descriptor;
    count.* += 1;
}

pub fn appendUniquePlanStateDescriptors(
    storage: []StateDescriptor,
    count: *u8,
    plan: *const ExecutionPlan,
) !void {
    for (plan.state_descs) |descriptor| {
        switch (descriptor.id) {
            @intFromEnum(StateBlockId.kv_cache),
            @intFromEnum(StateBlockId.shortconv),
            @intFromEnum(StateBlockId.mamba),
            => {},
            else => return error.UnknownStateDescriptorId,
        }
        if (descriptor.lifecycle != .slot_persistent) return error.InvalidStateDescriptorBinding;
        try appendUniqueStateDescriptor(storage, count, descriptor);
    }
}

pub const BuiltinStateFlags = struct {
    has_kv: bool = false,
    has_shortconv: bool = false,
    has_mamba: bool = false,
};

pub fn collectBuiltinStateFlags(plan: *const ExecutionPlan) !BuiltinStateFlags {
    var flags = BuiltinStateFlags{};
    for (plan.state_descs) |state_desc| {
        switch (state_desc.id) {
            @intFromEnum(StateBlockId.kv_cache) => {
                if (state_desc.lifecycle != .slot_persistent) return error.InvalidStateDescriptorBinding;
                flags.has_kv = true;
            },
            @intFromEnum(StateBlockId.shortconv) => {
                if (state_desc.lifecycle != .slot_persistent) return error.InvalidStateDescriptorBinding;
                flags.has_shortconv = true;
            },
            @intFromEnum(StateBlockId.mamba) => {
                if (state_desc.lifecycle != .slot_persistent) return error.InvalidStateDescriptorBinding;
                flags.has_mamba = true;
            },
            else => return error.UnknownStateDescriptorId,
        }
    }
    return flags;
}

const ParamKind = enum(u8) {
    kernel = 1,
    add = 2,
    linear = 3,
    matmul = 4,
    split = 5,
    softmax = 6,
    silu = 7,
    gelu = 8,
    mul = 9,
    add_tensor = 10,
    add_scalar = 11,
    mul_scalar = 12,
    mean = 13,
    pow = 14,
    rsqrt = 15,
    add_param = 16,
    add_param_scalar = 17,
    mul_param = 18,
    reshape = 19,
    transpose = 20,
    rope = 21,
    triu = 22,
    sdpa = 23,
    patch_embed = 24,
    spatial_merge = 25,
    deepstack_extract = 26,
    scatter = 27,
};

fn paramKindForLayerOp(op: layer_ops.LayerOp) ParamKind {
    return switch (op) {
        .kernel => .kernel,
        .add => .add,
        .linear => .linear,
        .matmul => .matmul,
        .split => .split,
        .softmax => .softmax,
        .silu => .silu,
        .gelu => .gelu,
        .mul => .mul,
        .add_tensor => .add_tensor,
        .add_scalar => .add_scalar,
        .mul_scalar => .mul_scalar,
        .mean => .mean,
        .pow => .pow,
        .rsqrt => .rsqrt,
        .add_param => .add_param,
        .add_param_scalar => .add_param_scalar,
        .mul_param => .mul_param,
        .reshape => .reshape,
        .transpose => .transpose,
        .rope => .rope,
        .triu => .triu,
        .sdpa => .sdpa,
        .patch_embed => .patch_embed,
        .spatial_merge => .spatial_merge,
        .deepstack_extract => .deepstack_extract,
        .scatter => .scatter,
    };
}

fn expectedParamKindForOpcode(opcode: Opcode) !ParamKind {
    return switch (opcode) {
        .rmsnorm,
        .multihead_attention,
        .swiglu,
        .moe,
        .mamba_mixer,
        .shortconv,
        .mla_attention,
        .embedding,
        => .kernel,
        .residual_add => .add,
        .linear => .linear,
        .matmul => .matmul,
        .split => .split,
        .softmax => .softmax,
        .silu => .silu,
        .gelu => .gelu,
        .mul => .mul,
        .add_tensor => .add_tensor,
        .add_scalar => .add_scalar,
        .mul_scalar => .mul_scalar,
        .mean => .mean,
        .pow => .pow,
        .rsqrt => .rsqrt,
        .add_param => .add_param,
        .add_param_scalar => .add_param_scalar,
        .mul_param => .mul_param,
        .reshape => .reshape,
        .transpose => .transpose,
        .rope => .rope,
        .triu => .triu,
        .scaled_dot_product_attention => .sdpa,
        .vision_patch_embed => .patch_embed,
        .vision_spatial_merge => .spatial_merge,
        .vision_deepstack_extract => .deepstack_extract,
        .vision_scatter => .scatter,
        else => error.InvalidParamBlockABI,
    };
}

fn expectedKernelDebugTypeForOpcode(opcode: Opcode) !op_types.OpType {
    return switch (opcode) {
        .rmsnorm => .norm,
        .multihead_attention => .multihead_attention,
        .swiglu => .mlp,
        .moe => .moe,
        .mamba_mixer => .mamba_mixer,
        .shortconv => .shortconv,
        .mla_attention => .multihead_attention,
        .embedding => .embedding,
        else => error.ParamBlockOpcodeMismatch,
    };
}

fn opcodeMatchesLayerOp(opcode: Opcode, op: layer_ops.LayerOp) bool {
    const actual = opcode_map.opcodeForLayerOp(op);
    if (actual == opcode) return true;
    return opcode == .mla_attention and actual == .multihead_attention;
}

fn encodeBufferId(id: layer_ops.BufferId) u8 {
    return @intCast(@intFromEnum(id));
}

fn decodeBufferId(raw: u8) !layer_ops.BufferId {
    if (raw > @intFromEnum(layer_ops.BufferId.tmp63)) return error.InvalidParamBlockABI;
    return @enumFromInt(raw);
}

const ParamEncoder = struct {
    bytes: std.ArrayListUnmanaged(u8) = .{},

    fn deinit(self: *ParamEncoder, allocator: std.mem.Allocator) void {
        self.bytes.deinit(allocator);
    }

    fn append(self: *ParamEncoder, allocator: std.mem.Allocator, raw: []const u8) !void {
        try self.bytes.appendSlice(allocator, raw);
    }

    fn alignTo(self: *ParamEncoder, allocator: std.mem.Allocator, alignment: usize) !void {
        const rem = self.bytes.items.len % alignment;
        if (rem == 0) return;
        try self.bytes.appendNTimes(allocator, 0, alignment - rem);
    }

    fn writeU8(self: *ParamEncoder, allocator: std.mem.Allocator, value: u8) !void {
        try self.bytes.append(allocator, value);
    }

    fn writeI8(self: *ParamEncoder, allocator: std.mem.Allocator, value: i8) !void {
        try self.writeU8(allocator, @bitCast(value));
    }

    fn writeBool(self: *ParamEncoder, allocator: std.mem.Allocator, value: bool) !void {
        try self.writeU8(allocator, @intFromBool(value));
    }

    fn writeU16(self: *ParamEncoder, allocator: std.mem.Allocator, value: u16) !void {
        var buf: [2]u8 = undefined;
        std.mem.writeInt(u16, &buf, value, .little);
        try self.append(allocator, &buf);
    }

    fn writeU32(self: *ParamEncoder, allocator: std.mem.Allocator, value: u32) !void {
        var buf: [4]u8 = undefined;
        std.mem.writeInt(u32, &buf, value, .little);
        try self.append(allocator, &buf);
    }

    fn writeI32(self: *ParamEncoder, allocator: std.mem.Allocator, value: i32) !void {
        var buf: [4]u8 = undefined;
        std.mem.writeInt(i32, &buf, value, .little);
        try self.append(allocator, &buf);
    }

    fn writeUsize(self: *ParamEncoder, allocator: std.mem.Allocator, value: usize) !void {
        var buf: [@sizeOf(usize)]u8 = undefined;
        std.mem.writeInt(usize, &buf, value, .little);
        try self.append(allocator, &buf);
    }

    fn writeF32(self: *ParamEncoder, allocator: std.mem.Allocator, value: f32) !void {
        try self.writeU32(allocator, @bitCast(value));
    }

    fn finish(self: *ParamEncoder, allocator: std.mem.Allocator, opcode: Opcode) !ParamBlock {
        if (self.bytes.items.len > max_param_block_data_bytes_v1) return error.InvalidParamBlockABI;
        const payload = try allocator.alignedAlloc(u8, .@"8", self.bytes.items.len);
        @memcpy(payload, self.bytes.items);
        return .{
            .version = param_block_abi_version_v1,
            .opcode = opcode,
            .data = payload,
        };
    }
};

pub const ParamDecoder = struct {
    data: []const u8,
    offset: usize = 0,

    pub fn init(data: []const u8) ParamDecoder {
        return .{ .data = data, .offset = 0 };
    }

    pub fn alignTo(self: *ParamDecoder, alignment: usize) !void {
        const rem = self.offset % alignment;
        if (rem == 0) return;
        const pad = alignment - rem;
        _ = try self.readBytes(pad);
    }

    pub fn readBytes(self: *ParamDecoder, len: usize) ![]const u8 {
        const end = std.math.add(usize, self.offset, len) catch return error.InvalidParamBlockABI;
        if (end > self.data.len) return error.InvalidParamBlockABI;
        const out = self.data[self.offset..end];
        self.offset = end;
        return out;
    }

    pub fn readU8(self: *ParamDecoder) !u8 {
        return (try self.readBytes(1))[0];
    }

    pub fn readI8(self: *ParamDecoder) !i8 {
        return @bitCast(try self.readU8());
    }

    pub fn readBool(self: *ParamDecoder) !bool {
        return switch (try self.readU8()) {
            0 => false,
            1 => true,
            else => error.InvalidParamBlockABI,
        };
    }

    pub fn readU16(self: *ParamDecoder) !u16 {
        const bytes = try self.readBytes(2);
        var buf: [2]u8 = undefined;
        @memcpy(&buf, bytes);
        return std.mem.readInt(u16, &buf, .little);
    }

    pub fn readU32(self: *ParamDecoder) !u32 {
        const bytes = try self.readBytes(4);
        var buf: [4]u8 = undefined;
        @memcpy(&buf, bytes);
        return std.mem.readInt(u32, &buf, .little);
    }

    pub fn readI32(self: *ParamDecoder) !i32 {
        const bytes = try self.readBytes(4);
        var buf: [4]u8 = undefined;
        @memcpy(&buf, bytes);
        return std.mem.readInt(i32, &buf, .little);
    }

    pub fn readUsize(self: *ParamDecoder) !usize {
        const bytes = try self.readBytes(@sizeOf(usize));
        var buf: [@sizeOf(usize)]u8 = undefined;
        @memcpy(&buf, bytes);
        return std.mem.readInt(usize, &buf, .little);
    }

    pub fn readF32(self: *ParamDecoder) !f32 {
        return @bitCast(try self.readU32());
    }

    pub fn finish(self: *const ParamDecoder) !void {
        if (self.offset != self.data.len) return error.InvalidParamBlockABI;
    }
};

// ---- ABI-stable packed param structs ----
//
// Layouts match the byte encoding produced by `encodeLayerOpParam`.
// Backends cast `ParamBlock.data` to these via `paramAs` — zero parsing,
// zero allocation, zero branching.

pub const ResidualAddParam = packed struct {
    param_kind: u8,
    branch_buffer_id: u8,
    scale_tag: u8,
    scale_literal: u32,
};

pub const ScalarOpParam = packed struct {
    param_kind: u8,
    in_buffer_id: u8,
    out_buffer_id: u8,
    scalar: u32,
};

pub const AddParamScalarParam = packed struct {
    param_kind: u8,
    out_buffer_id: u8,
    scalar: u32,
};

pub const MeanOpParam = packed struct {
    param_kind: u8,
    in_buffer_id: u8,
    out_buffer_id: u8,
    dim: i8,
    keepdim: u8,
};

pub const TransposeOpParam = packed struct {
    param_kind: u8,
    in_buffer_id: u8,
    out_buffer_id: u8,
    dim0: i8,
    dim1: i8,
};

pub const TriuOpParam = packed struct {
    param_kind: u8,
    in_buffer_id: u8,
    out_buffer_id: u8,
    diagonal: i32,
};

pub const SdpaOpParam = packed struct {
    param_kind: u8,
    q_buffer_id: u8,
    k_buffer_id: u8,
    v_buffer_id: u8,
    out_buffer_id: u8,
    is_causal: u8,
    has_scale: u8,
};

pub const ReshapeOpParam = packed struct {
    param_kind: u8,
    in_buffer_id: u8,
    out_buffer_id: u8,
    count: u16,
};

/// Cast raw `ParamBlock.data` to an ABI-stable packed param struct.
///
/// Validates opcode match and minimum size. Returns a pointer into the
/// existing `data` slice — no allocation, no copy.
pub fn paramAs(
    comptime T: type,
    params: []const ParamBlock,
    expected_opcode: Opcode,
) !*const T {
    if (params.len == 0) return error.MissingParamBlock;
    const param_block = params[0];
    if (param_block.opcode != expected_opcode) return error.ParamBlockOpcodeMismatch;
    if (param_block.data.len < @bitSizeOf(T) / 8) return error.InvalidParamBlockABI;
    return @ptrCast(@alignCast(param_block.data.ptr));
}

pub fn encodeLayerOpParam(
    allocator: std.mem.Allocator,
    opcode: Opcode,
    op: layer_ops.LayerOp,
) !ParamBlock {
    if (!opcodeMatchesLayerOp(opcode, op)) return error.ParamBlockOpcodeMismatch;
    const kind = paramKindForLayerOp(op);

    var enc = ParamEncoder{};
    defer enc.deinit(allocator);
    try enc.writeU8(allocator, @intFromEnum(kind));

    switch (op) {
        .kernel => |kernel_op| {
            try enc.writeU32(allocator, kernel_op.id);
            try enc.writeU8(allocator, encodeBufferId(kernel_op.in));
            try enc.writeU8(allocator, encodeBufferId(kernel_op.out));
            try enc.writeU8(allocator, @intFromEnum(kernel_op.debug_type));
        },
        .add => |add_op| {
            try enc.writeU8(allocator, encodeBufferId(add_op.branch));
            switch (add_op.scale) {
                .one => {
                    try enc.writeU8(allocator, 0);
                    try enc.writeF32(allocator, 0.0);
                },
                .residual_multiplier => {
                    try enc.writeU8(allocator, 1);
                    try enc.writeF32(allocator, 0.0);
                },
                .literal => |value| {
                    try enc.writeU8(allocator, 2);
                    try enc.writeF32(allocator, value);
                },
            }
        },
        .linear => |linear_op| {
            try enc.writeU8(allocator, encodeBufferId(linear_op.in));
            try enc.writeU8(allocator, encodeBufferId(linear_op.out));
        },
        .matmul => |matmul_op| {
            try enc.writeU8(allocator, encodeBufferId(matmul_op.in_a));
            try enc.writeU8(allocator, encodeBufferId(matmul_op.in_b));
            try enc.writeU8(allocator, encodeBufferId(matmul_op.out));
        },
        .split => |split_op| {
            const count: u16 = std.math.cast(u16, split_op.split_sizes.len) orelse return error.InvalidParamBlockABI;
            try enc.writeU8(allocator, encodeBufferId(split_op.in));
            try enc.writeU8(allocator, encodeBufferId(split_op.out_start));
            try enc.writeU8(allocator, split_op.num_outputs);
            try enc.writeI8(allocator, split_op.dim);
            try enc.writeU16(allocator, count);
            if (count != 0) {
                try enc.alignTo(allocator, @alignOf(usize));
                for (split_op.split_sizes) |size| {
                    try enc.writeUsize(allocator, size);
                }
            }
        },
        .softmax => |softmax_op| {
            try enc.writeU8(allocator, encodeBufferId(softmax_op.in));
            try enc.writeU8(allocator, encodeBufferId(softmax_op.out));
            try enc.writeI8(allocator, softmax_op.dim);
        },
        .silu => |silu_op| {
            try enc.writeU8(allocator, encodeBufferId(silu_op.in));
            try enc.writeU8(allocator, encodeBufferId(silu_op.out));
        },
        .gelu => |gelu_op| {
            try enc.writeU8(allocator, encodeBufferId(gelu_op.in));
            try enc.writeU8(allocator, encodeBufferId(gelu_op.out));
        },
        .mul => |mul_op| {
            try enc.writeU8(allocator, encodeBufferId(mul_op.in));
            try enc.writeU8(allocator, encodeBufferId(mul_op.other));
            try enc.writeU8(allocator, encodeBufferId(mul_op.out));
        },
        .add_tensor => |add_tensor_op| {
            try enc.writeU8(allocator, encodeBufferId(add_tensor_op.in_a));
            try enc.writeU8(allocator, encodeBufferId(add_tensor_op.in_b));
            try enc.writeU8(allocator, encodeBufferId(add_tensor_op.out));
        },
        .add_scalar => |add_scalar_op| {
            try enc.writeU8(allocator, encodeBufferId(add_scalar_op.in));
            try enc.writeU8(allocator, encodeBufferId(add_scalar_op.out));
            try enc.writeF32(allocator, add_scalar_op.scalar);
        },
        .mul_scalar => |mul_scalar_op| {
            try enc.writeU8(allocator, encodeBufferId(mul_scalar_op.in));
            try enc.writeU8(allocator, encodeBufferId(mul_scalar_op.out));
            try enc.writeF32(allocator, mul_scalar_op.scalar);
        },
        .mean => |mean_op| {
            try enc.writeU8(allocator, encodeBufferId(mean_op.in));
            try enc.writeU8(allocator, encodeBufferId(mean_op.out));
            try enc.writeI8(allocator, mean_op.dim);
            try enc.writeBool(allocator, mean_op.keepdim);
        },
        .pow => |pow_op| {
            try enc.writeU8(allocator, encodeBufferId(pow_op.in));
            try enc.writeU8(allocator, encodeBufferId(pow_op.out));
            try enc.writeF32(allocator, pow_op.exponent);
        },
        .rsqrt => |rsqrt_op| {
            try enc.writeU8(allocator, encodeBufferId(rsqrt_op.in));
            try enc.writeU8(allocator, encodeBufferId(rsqrt_op.out));
        },
        .add_param => |add_param_op| {
            try enc.writeU8(allocator, encodeBufferId(add_param_op.in));
            try enc.writeU8(allocator, encodeBufferId(add_param_op.out));
        },
        .add_param_scalar => |add_param_scalar_op| {
            try enc.writeU8(allocator, encodeBufferId(add_param_scalar_op.out));
            try enc.writeF32(allocator, add_param_scalar_op.scalar);
        },
        .mul_param => |mul_param_op| {
            try enc.writeU8(allocator, encodeBufferId(mul_param_op.in));
            try enc.writeU8(allocator, encodeBufferId(mul_param_op.out));
        },
        .reshape => |reshape_op| {
            const count: u16 = std.math.cast(u16, reshape_op.shape.len) orelse return error.InvalidParamBlockABI;
            try enc.writeU8(allocator, encodeBufferId(reshape_op.in));
            try enc.writeU8(allocator, encodeBufferId(reshape_op.out));
            try enc.writeU16(allocator, count);
            if (count != 0) {
                try enc.alignTo(allocator, @alignOf(i32));
                for (reshape_op.shape) |shape_item| {
                    try enc.writeI32(allocator, shape_item);
                }
            }
        },
        .transpose => |transpose_op| {
            try enc.writeU8(allocator, encodeBufferId(transpose_op.in));
            try enc.writeU8(allocator, encodeBufferId(transpose_op.out));
            try enc.writeI8(allocator, transpose_op.dim0);
            try enc.writeI8(allocator, transpose_op.dim1);
        },
        .rope => |rope_op| {
            try enc.writeU8(allocator, encodeBufferId(rope_op.in));
            try enc.writeU8(allocator, encodeBufferId(rope_op.out));
        },
        .triu => |triu_op| {
            try enc.writeU8(allocator, encodeBufferId(triu_op.in));
            try enc.writeU8(allocator, encodeBufferId(triu_op.out));
            try enc.writeI32(allocator, triu_op.diagonal);
        },
        .sdpa => |sdpa_op| {
            try enc.writeU8(allocator, encodeBufferId(sdpa_op.q));
            try enc.writeU8(allocator, encodeBufferId(sdpa_op.k));
            try enc.writeU8(allocator, encodeBufferId(sdpa_op.v));
            try enc.writeU8(allocator, encodeBufferId(sdpa_op.out));
            try enc.writeBool(allocator, sdpa_op.is_causal);
            try enc.writeBool(allocator, sdpa_op.scale != null);
            if (sdpa_op.scale) |scale| try enc.writeF32(allocator, scale);
        },
        .patch_embed => |patch_op| {
            try enc.writeU8(allocator, encodeBufferId(patch_op.in));
            try enc.writeU8(allocator, encodeBufferId(patch_op.out));
        },
        .spatial_merge => |spatial_op| {
            try enc.writeU8(allocator, encodeBufferId(spatial_op.in));
            try enc.writeU8(allocator, encodeBufferId(spatial_op.out));
            try enc.writeU32(allocator, spatial_op.merge_size);
        },
        .deepstack_extract => |deepstack_op| {
            try enc.writeU8(allocator, encodeBufferId(deepstack_op.in));
            try enc.writeU8(allocator, encodeBufferId(deepstack_op.out));
            try enc.writeU32(allocator, deepstack_op.layer_index);
        },
        .scatter => |scatter_op| {
            try enc.writeU8(allocator, encodeBufferId(scatter_op.text_in));
            try enc.writeU8(allocator, encodeBufferId(scatter_op.vision_in));
            try enc.writeU8(allocator, encodeBufferId(scatter_op.out));
            try enc.writeU32(allocator, scatter_op.image_token_id);
        },
    }

    const param_block = try enc.finish(allocator, opcode);
    try validateParamBlockAbi(&param_block);
    return param_block;
}

fn decodeLayerOpFromParam(opcode: Opcode, data: []const u8) !layer_ops.LayerOp {
    var dec = ParamDecoder.init(data);
    const raw_kind = try dec.readU8();
    const kind: ParamKind = std.meta.intToEnum(ParamKind, raw_kind) catch return error.InvalidParamBlockABI;
    if (kind != try expectedParamKindForOpcode(opcode)) return error.ParamBlockOpcodeMismatch;

    const op: layer_ops.LayerOp = switch (kind) {
        .kernel => blk: {
            const debug_type = try expectedKernelDebugTypeForOpcode(opcode);
            const id = try dec.readU32();
            const in = try decodeBufferId(try dec.readU8());
            const out = try decodeBufferId(try dec.readU8());
            const debug_type_raw = try dec.readU8();
            const stored_debug_type: op_types.OpType = std.meta.intToEnum(op_types.OpType, debug_type_raw) catch return error.InvalidParamBlockABI;
            if (stored_debug_type != debug_type) return error.ParamBlockOpcodeMismatch;
            break :blk .{ .kernel = .{
                .id = id,
                .in = in,
                .out = out,
                .debug_type = debug_type,
            } };
        },
        .add => blk: {
            const branch = try decodeBufferId(try dec.readU8());
            const scale_tag = try dec.readU8();
            const literal = try dec.readF32();
            const scale: layer_ops.ResidualScale = switch (scale_tag) {
                0 => .one,
                1 => .residual_multiplier,
                2 => .{ .literal = literal },
                else => return error.InvalidParamBlockABI,
            };
            break :blk .{ .add = .{
                .branch = branch,
                .scale = scale,
            } };
        },
        .linear => .{ .linear = .{
            .in = try decodeBufferId(try dec.readU8()),
            .out = try decodeBufferId(try dec.readU8()),
            .weight_name = &.{},
        } },
        .matmul => .{ .matmul = .{
            .in_a = try decodeBufferId(try dec.readU8()),
            .in_b = try decodeBufferId(try dec.readU8()),
            .out = try decodeBufferId(try dec.readU8()),
        } },
        .split => blk: {
            const in = try decodeBufferId(try dec.readU8());
            const out_start = try decodeBufferId(try dec.readU8());
            const num_outputs = try dec.readU8();
            const dim = try dec.readI8();
            const count = try dec.readU16();
            var split_sizes: []const usize = &.{};
            if (count != 0) {
                try dec.alignTo(@alignOf(usize));
                const raw = try dec.readBytes(@as(usize, count) * @sizeOf(usize));
                if ((@intFromPtr(raw.ptr) % @alignOf(usize)) != 0) return error.InvalidParamBlockABI;
                const ptr: [*]const usize = @ptrCast(@alignCast(raw.ptr));
                split_sizes = ptr[0..@as(usize, count)];
            }
            break :blk .{ .split = .{
                .in = in,
                .out_start = out_start,
                .num_outputs = num_outputs,
                .dim = dim,
                .split_sizes = split_sizes,
            } };
        },
        .softmax => .{ .softmax = .{
            .in = try decodeBufferId(try dec.readU8()),
            .out = try decodeBufferId(try dec.readU8()),
            .dim = try dec.readI8(),
        } },
        .silu => .{ .silu = .{
            .in = try decodeBufferId(try dec.readU8()),
            .out = try decodeBufferId(try dec.readU8()),
        } },
        .gelu => .{ .gelu = .{
            .in = try decodeBufferId(try dec.readU8()),
            .out = try decodeBufferId(try dec.readU8()),
        } },
        .mul => .{ .mul = .{
            .in = try decodeBufferId(try dec.readU8()),
            .other = try decodeBufferId(try dec.readU8()),
            .out = try decodeBufferId(try dec.readU8()),
        } },
        .add_tensor => .{ .add_tensor = .{
            .in_a = try decodeBufferId(try dec.readU8()),
            .in_b = try decodeBufferId(try dec.readU8()),
            .out = try decodeBufferId(try dec.readU8()),
        } },
        .add_scalar => .{ .add_scalar = .{
            .in = try decodeBufferId(try dec.readU8()),
            .out = try decodeBufferId(try dec.readU8()),
            .scalar = try dec.readF32(),
        } },
        .mul_scalar => .{ .mul_scalar = .{
            .in = try decodeBufferId(try dec.readU8()),
            .out = try decodeBufferId(try dec.readU8()),
            .scalar = try dec.readF32(),
        } },
        .mean => .{ .mean = .{
            .in = try decodeBufferId(try dec.readU8()),
            .out = try decodeBufferId(try dec.readU8()),
            .dim = try dec.readI8(),
            .keepdim = try dec.readBool(),
        } },
        .pow => .{ .pow = .{
            .in = try decodeBufferId(try dec.readU8()),
            .out = try decodeBufferId(try dec.readU8()),
            .exponent = try dec.readF32(),
        } },
        .rsqrt => .{ .rsqrt = .{
            .in = try decodeBufferId(try dec.readU8()),
            .out = try decodeBufferId(try dec.readU8()),
        } },
        .add_param => .{ .add_param = .{
            .in = try decodeBufferId(try dec.readU8()),
            .out = try decodeBufferId(try dec.readU8()),
            .param_name = &.{},
        } },
        .add_param_scalar => .{ .add_param_scalar = .{
            .out = try decodeBufferId(try dec.readU8()),
            .param_name = &.{},
            .scalar = try dec.readF32(),
        } },
        .mul_param => .{ .mul_param = .{
            .in = try decodeBufferId(try dec.readU8()),
            .out = try decodeBufferId(try dec.readU8()),
            .param_name = &.{},
        } },
        .reshape => blk: {
            const in = try decodeBufferId(try dec.readU8());
            const out = try decodeBufferId(try dec.readU8());
            const count = try dec.readU16();
            var shape: []const i32 = &.{};
            if (count != 0) {
                try dec.alignTo(@alignOf(i32));
                const raw = try dec.readBytes(@as(usize, count) * @sizeOf(i32));
                if ((@intFromPtr(raw.ptr) % @alignOf(i32)) != 0) return error.InvalidParamBlockABI;
                const ptr: [*]const i32 = @ptrCast(@alignCast(raw.ptr));
                shape = ptr[0..@as(usize, count)];
            }
            break :blk .{ .reshape = .{
                .in = in,
                .out = out,
                .shape = shape,
            } };
        },
        .transpose => .{ .transpose = .{
            .in = try decodeBufferId(try dec.readU8()),
            .out = try decodeBufferId(try dec.readU8()),
            .dim0 = try dec.readI8(),
            .dim1 = try dec.readI8(),
        } },
        .rope => .{ .rope = .{
            .in = try decodeBufferId(try dec.readU8()),
            .out = try decodeBufferId(try dec.readU8()),
        } },
        .triu => .{ .triu = .{
            .in = try decodeBufferId(try dec.readU8()),
            .out = try decodeBufferId(try dec.readU8()),
            .diagonal = try dec.readI32(),
        } },
        .sdpa => blk: {
            const q = try decodeBufferId(try dec.readU8());
            const k = try decodeBufferId(try dec.readU8());
            const v = try decodeBufferId(try dec.readU8());
            const out = try decodeBufferId(try dec.readU8());
            const is_causal = try dec.readBool();
            const has_scale = try dec.readBool();
            break :blk .{ .sdpa = .{
                .q = q,
                .k = k,
                .v = v,
                .out = out,
                .is_causal = is_causal,
                .scale = if (has_scale) try dec.readF32() else null,
            } };
        },
        .patch_embed => .{ .patch_embed = .{
            .in = try decodeBufferId(try dec.readU8()),
            .out = try decodeBufferId(try dec.readU8()),
        } },
        .spatial_merge => .{ .spatial_merge = .{
            .in = try decodeBufferId(try dec.readU8()),
            .out = try decodeBufferId(try dec.readU8()),
            .merge_size = try dec.readU32(),
        } },
        .deepstack_extract => .{ .deepstack_extract = .{
            .in = try decodeBufferId(try dec.readU8()),
            .out = try decodeBufferId(try dec.readU8()),
            .layer_index = try dec.readU32(),
        } },
        .scatter => .{ .scatter = .{
            .text_in = try decodeBufferId(try dec.readU8()),
            .vision_in = try decodeBufferId(try dec.readU8()),
            .out = try decodeBufferId(try dec.readU8()),
            .image_token_id = try dec.readU32(),
        } },
    };

    try dec.finish();
    if (!opcodeMatchesLayerOp(opcode, op)) return error.ParamBlockOpcodeMismatch;
    return op;
}

pub fn planUsesInstructionWeights(plan: *const ExecutionPlan) bool {
    for (plan.instructions) |insn| {
        if (insn.weights.len != 0) return true;
    }
    return false;
}

pub fn validatePlanWithoutInstructionWeights(compiled: *const CompiledPlan) !void {
    if (planUsesInstructionWeights(&compiled.plan) or compiled.weight_bindings.len != 0) {
        return error.InvalidWeightRefCount;
    }
}

pub fn instructionWeightBindingName(
    compiled: *const CompiledPlan,
    instruction_index: usize,
    weight_slot: usize,
) ![]const u8 {
    if (instruction_index >= compiled.plan.instructions.len) return error.InvalidInstructionIndex;
    const insn = compiled.plan.instructions[instruction_index];
    if (weight_slot >= insn.weights.len) return error.InvalidWeightRefIndex;
    const ref_index = insn.weights[weight_slot].index;
    if (ref_index >= compiled.weight_bindings.len) return error.InvalidWeightRefIndex;
    const binding_name = compiled.weight_bindings[ref_index].name;
    if (binding_name.len == 0) return error.InvalidWeightBindingName;
    return binding_name;
}

pub fn instructionSingleWeightBindingName(compiled: *const CompiledPlan, instruction_index: usize) ![]const u8 {
    if (instruction_index >= compiled.plan.instructions.len) return error.InvalidInstructionIndex;
    const insn = compiled.plan.instructions[instruction_index];
    if (insn.weights.len != 1) return error.InvalidWeightRefCount;
    return instructionWeightBindingName(compiled, instruction_index, 0);
}

pub fn decodeInstructionLayerOp(
    compiled: *const CompiledPlan,
    insn: *const Instruction,
    instruction_index: usize,
) !layer_ops.LayerOp {
    if (instruction_index >= compiled.plan.instructions.len) return error.InvalidInstructionIndex;
    const param_id = insn.param_block_id orelse return error.MissingParamBlock;
    if (param_id >= compiled.param_blocks.len) return error.MissingParamBlock;
    const param_block = compiled.param_blocks[param_id];
    try validateParamBlockAbi(&param_block);
    if (param_block.opcode != insn.opcode) return error.ParamBlockOpcodeMismatch;
    return decodeLayerOpFromParam(insn.opcode, param_block.data);
}

pub fn findStateBlock(
    state_blocks: []const StateBlockHandle,
    state_id: u8,
) ?*const StateBlockHandle {
    for (state_blocks) |*state_block| {
        if (state_block.id == state_id) return state_block;
    }
    return null;
}

pub fn stateValueFromBlock(comptime T: type, state_block: *const StateBlockHandle) ?T {
    comptime {
        if (@typeInfo(T) != .pointer) @compileError("stateValueFromBlock requires a pointer type");
    }
    if (state_block.size < @sizeOf(OpaqueStateRef)) return null;
    if (state_block.align_bytes < @alignOf(OpaqueStateRef)) return null;
    const state_ref: *const OpaqueStateRef = @ptrCast(@alignCast(state_block.ptr));
    const raw = state_ref.ptr orelse return null;
    return @ptrCast(@alignCast(raw));
}

pub fn findStateValue(comptime T: type, state_blocks: []const StateBlockHandle, state_id: u8) ?T {
    const state_block = findStateBlock(state_blocks, state_id) orelse return null;
    return stateValueFromBlock(T, state_block);
}

pub fn findStateDescriptor(
    plan: *const ExecutionPlan,
    state_id: u8,
) ?*const StateDescriptor {
    for (plan.state_descs) |*state_desc| {
        if (state_desc.id == state_id) return state_desc;
    }
    return null;
}

pub fn requireInstructionStateBlock(
    insn: *const Instruction,
    state_blocks: []const StateBlockHandle,
) !?*const StateBlockHandle {
    if (insn.state_block_id) |state_id| {
        const block = findStateBlock(state_blocks, state_id) orelse return error.InvalidStateDescriptorBinding;
        if (block.align_bytes == 0 or block.size == 0) return error.InvalidStateDescriptorBinding;
        return block;
    }
    return null;
}

pub fn requireInstructionStateBlockForPlan(
    insn: *const Instruction,
    plan: *const ExecutionPlan,
    state_blocks: []const StateBlockHandle,
) !?*const StateBlockHandle {
    const state_block = try requireInstructionStateBlock(insn, state_blocks);
    if (insn.state_block_id) |state_id| {
        const descriptor = findStateDescriptor(plan, state_id) orelse return error.UnknownStateDescriptorId;
        if (descriptor.lifecycle != .slot_persistent) return error.InvalidStateDescriptorBinding;
        if (descriptor.align_bytes == 0) return error.InvalidStateDescriptorBinding;
        if (state_block.?.align_bytes < descriptor.align_bytes) return error.InvalidStateDescriptorBinding;
        if (descriptor.size_bytes > 0 and state_block.?.size < descriptor.size_bytes) {
            return error.InvalidStateDescriptorBinding;
        }
    }
    return state_block;
}

pub fn validateStateBlocksForDescriptors(
    descriptors: []const StateDescriptor,
    state_blocks: []const StateBlockHandle,
) !void {
    var desc_seen: [256]bool = [_]bool{false} ** 256;
    var block_seen: [256]bool = [_]bool{false} ** 256;

    for (descriptors) |descriptor| {
        if (desc_seen[descriptor.id]) return error.DuplicateStateDescriptorId;
        desc_seen[descriptor.id] = true;
        if (descriptor.lifecycle != .slot_persistent) return error.InvalidStateDescriptorBinding;
        if (descriptor.align_bytes == 0) return error.InvalidStateAlignment;
    }

    for (state_blocks) |state_block| {
        if (state_block.align_bytes == 0 or state_block.size == 0) return error.InvalidStateDescriptorBinding;
        if (!desc_seen[state_block.id]) return error.UnknownStateDescriptorId;
        if (block_seen[state_block.id]) return error.InvalidStateDescriptorBinding;
        block_seen[state_block.id] = true;
    }

    for (descriptors) |descriptor| {
        const state_block = findStateBlock(state_blocks, descriptor.id) orelse return error.InvalidStateDescriptorBinding;
        if (state_block.align_bytes < descriptor.align_bytes) return error.InvalidStateDescriptorBinding;
        if (descriptor.size_bytes > 0 and state_block.size < descriptor.size_bytes) return error.InvalidStateDescriptorBinding;
    }
}

pub fn validateTensorViewDesc(view: *const TensorViewDesc) !void {
    if (view.rank > tensor_view_rank_cap_v1) return error.UnsupportedTensorRank;
}

pub fn validateExecutionContext(ctx: *const ExecutionContext) !void {
    if (ctx.batch_size != ctx.active_slots.len) return error.InvalidBatchSize;
    if (ctx.sequence_lengths.len != ctx.active_slots.len) return error.InvalidSequenceLengthCount;
}

pub fn validateParamBlockAbi(param_block: *const ParamBlock) !void {
    if (param_block.version != param_block_abi_version_v1) return error.InvalidParamBlockABI;
    if (param_block.data.len > max_param_block_data_bytes_v1) return error.InvalidParamBlockABI;
    if (param_block.data.len > 0 and (@intFromPtr(param_block.data.ptr) % 8) != 0) {
        return error.InvalidParamBlockABI;
    }
}

pub fn validateExecutionPlan(plan: *const ExecutionPlan) !void {
    var state_seen: [256]bool = [_]bool{false} ** 256;
    for (plan.state_descs) |state_desc| {
        if (state_seen[state_desc.id]) return error.DuplicateStateDescriptorId;
        state_seen[state_desc.id] = true;
        if (state_desc.align_bytes == 0) return error.InvalidStateAlignment;
    }

    for (plan.instructions) |insn| {
        const allowed_state_id = stateBlockIdForOpcode(insn.opcode);
        if (insn.state_block_id) |state_id| {
            if (allowed_state_id == null or allowed_state_id.? != state_id) {
                return error.InvalidStateDescriptorBinding;
            }
        } else if (requiredStateBlockIdForOpcode(insn.opcode) != null) {
            return error.InvalidStateDescriptorBinding;
        }

        for (insn.inputs) |register| {
            if (registerToIndex(register) >= plan.register_count) return error.InvalidInstructionRegisterRef;
        }
        for (insn.outputs) |register| {
            if (registerToIndex(register) >= plan.register_count) return error.InvalidInstructionRegisterRef;
        }
        if (insn.state_block_id) |state_id| {
            if (!state_seen[state_id]) return error.UnknownStateDescriptorId;
        }
    }
}

pub fn validateExecutionPlanForBlockKind(plan: *const ExecutionPlan, kind: op_types.BlockKind) !void {
    for (plan.instructions) |insn| {
        if (!opcodeStateCompatibleWithBlockKind(insn.opcode, kind)) {
            return error.InvalidStateDescriptorBinding;
        }
    }
}

fn layerOpInputCount(op: layer_ops.LayerOp) usize {
    return switch (op) {
        .kernel => 1,
        .add => 2,
        .linear => 1,
        .matmul => 2,
        .split => 1,
        .softmax => 1,
        .silu => 1,
        .gelu => 1,
        .mul => 2,
        .add_tensor => 2,
        .add_scalar => 1,
        .mul_scalar => 1,
        .mean => 1,
        .pow => 1,
        .rsqrt => 1,
        .add_param => 1,
        .add_param_scalar => 0,
        .mul_param => 1,
        .reshape => 1,
        .transpose => 1,
        .rope => 1,
        .triu => 1,
        .sdpa => 3,
        .patch_embed => 1,
        .spatial_merge => 1,
        .deepstack_extract => 1,
        .scatter => 2,
    };
}

fn layerOpOutputCount(op: layer_ops.LayerOp) usize {
    return switch (op) {
        .split => |split_op| split_op.num_outputs,
        else => 1,
    };
}

pub fn planFinalOutputRegister(plan: *const ExecutionPlan) RegisterRef {
    if (plan.instructions.len == 0) return registerFromIndex(0);
    const last = plan.instructions[plan.instructions.len - 1];
    if (last.outputs.len == 0) return registerFromIndex(0);
    return last.outputs[last.outputs.len - 1];
}

pub fn validateCompiledPlan(compiled: *const CompiledPlan) !void {
    try validateExecutionPlan(&compiled.plan);

    for (compiled.weight_bindings, 0..) |binding, idx| {
        if (binding.index != idx) return error.InvalidWeightBindingIndex;
        if (binding.name.len == 0) return error.InvalidWeightBindingName;
    }

    if (compiled.liveness.register_last_read.len != compiled.plan.register_count) {
        return error.InvalidLivenessRegisterCount;
    }
    if (compiled.plan.register_count > 0 and compiled.register_buffer_specs.len != compiled.plan.register_count) {
        return error.InvalidRegisterSpecCount;
    }
    for (compiled.register_buffer_specs) |spec| {
        if (spec.size == 0) return error.InvalidRegisterSpecSize;
        if (spec.@"align" == 0) return error.InvalidStateAlignment;
    }
    if (compiled.register_to_buffer_id.len != 0 and compiled.register_to_buffer_id.len != compiled.plan.register_count) {
        return error.InvalidRegisterSpecCount;
    }
    if (compiled.liveness.kill_after_instruction.len != compiled.plan.instructions.len) {
        return error.InvalidLivenessInstructionCount;
    }
    const words = LivenessMap.bitsetWordCount(compiled.plan.register_count);
    for (compiled.liveness.kill_after_instruction) |row| {
        if (row.len != words) return error.InvalidLivenessBitsetWidth;
    }
    if (compiled.peak_registers > compiled.plan.register_count) return error.InvalidPeakRegisters;

    for (compiled.plan.instructions) |insn| {
        if (insn.weights.len != expectedWeightRefCount(insn.opcode)) return error.InvalidWeightRefCount;
        for (insn.weights) |weight_ref| {
            if (weight_ref.index >= compiled.weight_bindings.len) return error.InvalidWeightRefIndex;
        }
        if (insn.param_block_id) |param_id| {
            if (param_id >= compiled.param_blocks.len) return error.UnknownParamBlockId;
            const param_block = compiled.param_blocks[param_id];
            try validateParamBlockAbi(&param_block);
            if (param_block.opcode != insn.opcode) return error.ParamBlockOpcodeMismatch;
            const decoded = try decodeLayerOpFromParam(insn.opcode, param_block.data);
            if (layerOpInputCount(decoded) != insn.inputs.len) return error.InvalidInstructionRegisterRef;
            if (layerOpOutputCount(decoded) != insn.outputs.len) return error.InvalidInstructionRegisterRef;
        }
    }
}

test "register conversion keeps numeric identity" {
    const reg = registerFromIndex(42);
    try std.testing.expectEqual(@as(u16, 42), registerToIndex(reg));
}

test "TensorHandle intentionally has no inline shape fields" {
    try std.testing.expect(@hasField(TensorHandle, "register"));
    try std.testing.expect(@hasField(TensorHandle, "ptr"));
    try std.testing.expect(!@hasField(TensorHandle, "len"));
    try std.testing.expect(!@hasField(TensorHandle, "shape"));
}

test "validateTensorViewDesc rejects rank above v1 cap" {
    var view = TensorViewDesc{
        .dtype = .f32,
        .rank = 5,
        .shape = .{ 1, 1, 1, 1 },
        .stride_elems = .{ 1, 1, 1, 1 },
        .layout = .contiguous,
    };
    try std.testing.expectError(error.UnsupportedTensorRank, validateTensorViewDesc(&view));
    view.rank = tensor_view_rank_cap_v1;
    try validateTensorViewDesc(&view);
}

test "validateExecutionContext checks batch invariants" {
    var slot: [1]usize = .{0};
    var seq_len: [2]u32 = .{ 3, 7 };
    const invalid = ExecutionContext{
        .mode = .decode,
        .active_slots = slot[0..],
        .sequence_lengths = seq_len[0..],
        .batch_size = 1,
    };
    try std.testing.expectError(error.InvalidSequenceLengthCount, validateExecutionContext(&invalid));
}

test "validateBatchCapability enforces backend batch limits" {
    const single_only = AdapterCapability{
        .supports_batch = false,
        .supports_graph_emit = false,
        .max_batch_size = 1,
    };
    try validateBatchCapability(single_only, 1);
    try std.testing.expectError(error.UnsupportedBatchSize, validateBatchCapability(single_only, 2));

    const batched = AdapterCapability{
        .supports_batch = true,
        .supports_graph_emit = false,
        .max_batch_size = 8,
    };
    try validateBatchCapability(batched, 8);
    try std.testing.expectError(error.UnsupportedBatchSize, validateBatchCapability(batched, 9));
}

test "validateParamBlockAbi rejects invalid version, size, and alignment" {
    const good_data: [8]u8 align(8) = .{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const good = ParamBlock{
        .version = param_block_abi_version_v1,
        .opcode = .rmsnorm,
        .data = good_data[0..],
    };
    try validateParamBlockAbi(&good);

    const wrong_version = ParamBlock{
        .version = param_block_abi_version_v1 + 1,
        .opcode = .rmsnorm,
        .data = good_data[0..],
    };
    try std.testing.expectError(error.InvalidParamBlockABI, validateParamBlockAbi(&wrong_version));

    var large_backing: [max_param_block_data_bytes_v1 + 1]u8 align(8) = undefined;
    @memset(large_backing[0..], 0);
    const too_large = ParamBlock{
        .version = param_block_abi_version_v1,
        .opcode = .rmsnorm,
        .data = large_backing[0..],
    };
    try std.testing.expectError(error.InvalidParamBlockABI, validateParamBlockAbi(&too_large));

    // Alignment is enforced by the type system: ParamBlock.data is []align(8) const u8.
    // The runtime check in validateParamBlockAbi is defense-in-depth.
}

test "validateCompiledPlan rejects invalid param block ABI" {
    const insn = Instruction{
        .opcode = .residual_add,
        .inputs = &.{ registerFromIndex(0), registerFromIndex(1) },
        .outputs = &.{registerFromIndex(1)},
        .weights = &.{},
        .param_block_id = 0,
        .state_block_id = null,
    };
    var large_backing: [max_param_block_data_bytes_v1 + 1]u8 align(8) = undefined;
    @memset(large_backing[0..], 0);
    const compiled = CompiledPlan{
        .plan = .{
            .instructions = &.{insn},
            .register_count = 2,
            .state_descs = &.{},
        },
        .param_blocks = &.{.{
            .version = param_block_abi_version_v1,
            .opcode = .residual_add,
            .data = large_backing[0..],
        }},
        .weight_bindings = &.{},
        .register_buffer_specs = &.{
            .{ .size = 1, .@"align" = 64 },
            .{ .size = 1, .@"align" = 64 },
        },
        .liveness = .{
            .register_last_read = &.{ 0, 0 },
            .kill_after_instruction = &.{&.{0}},
        },
        .peak_registers = 2,
        .diagnostics = &.{},
    };
    try std.testing.expectError(error.InvalidParamBlockABI, validateCompiledPlan(&compiled));
}

test "validateExecutionPlan detects invalid register references" {
    const inputs = [_]RegisterRef{registerFromIndex(0)};
    const outputs = [_]RegisterRef{registerFromIndex(4)};
    const insn = Instruction{
        .opcode = .rmsnorm,
        .inputs = inputs[0..],
        .outputs = outputs[0..],
        .weights = &.{},
        .param_block_id = null,
        .state_block_id = null,
    };
    const plan = ExecutionPlan{
        .instructions = &.{insn},
        .register_count = 4,
        .state_descs = &.{},
    };
    try std.testing.expectError(error.InvalidInstructionRegisterRef, validateExecutionPlan(&plan));
}

test "requireInstructionStateBlock enforces descriptor presence for stateful instructions" {
    const stateful_insn = Instruction{
        .opcode = .multihead_attention,
        .inputs = &.{registerFromIndex(0)},
        .outputs = &.{registerFromIndex(1)},
        .weights = &.{},
        .param_block_id = null,
        .state_block_id = @intFromEnum(StateBlockId.kv_cache),
    };
    const stateless_insn = Instruction{
        .opcode = .residual_add,
        .inputs = &.{ registerFromIndex(0), registerFromIndex(1) },
        .outputs = &.{registerFromIndex(0)},
        .weights = &.{},
        .param_block_id = null,
        .state_block_id = null,
    };

    var dummy_bytes align(64) = [_]u8{0} ** 64;
    const matching = [_]StateBlockHandle{.{
        .id = @intFromEnum(StateBlockId.kv_cache),
        .ptr = dummy_bytes[0..].ptr,
        .size = dummy_bytes.len,
        .align_bytes = 64,
    }};

    try std.testing.expect((try requireInstructionStateBlock(&stateful_insn, matching[0..])) != null);
    try std.testing.expect((try requireInstructionStateBlock(&stateless_insn, matching[0..])) == null);
    try std.testing.expectError(error.InvalidStateDescriptorBinding, requireInstructionStateBlock(&stateful_insn, &.{}));

    var bad_size = matching[0];
    bad_size.size = 0;
    try std.testing.expectError(error.InvalidStateDescriptorBinding, requireInstructionStateBlock(&stateful_insn, &.{bad_size}));

    var bad_align = matching[0];
    bad_align.align_bytes = 0;
    try std.testing.expectError(error.InvalidStateDescriptorBinding, requireInstructionStateBlock(&stateful_insn, &.{bad_align}));
}

test "requireInstructionStateBlockForPlan validates descriptor size and alignment requirements" {
    const stateful_insn = Instruction{
        .opcode = .multihead_attention,
        .inputs = &.{registerFromIndex(0)},
        .outputs = &.{registerFromIndex(1)},
        .weights = &.{},
        .param_block_id = null,
        .state_block_id = @intFromEnum(StateBlockId.kv_cache),
    };
    const plan = ExecutionPlan{
        .instructions = &.{stateful_insn},
        .register_count = 2,
        .state_descs = &.{.{
            .id = @intFromEnum(StateBlockId.kv_cache),
            .size_bytes = 128,
            .align_bytes = 64,
            .zero_init = false,
            .lifecycle = .slot_persistent,
        }},
    };

    var bytes align(64) = [_]u8{0} ** 256;
    const ok_blocks = [_]StateBlockHandle{.{
        .id = @intFromEnum(StateBlockId.kv_cache),
        .ptr = bytes[0..].ptr,
        .size = 256,
        .align_bytes = 64,
    }};
    try std.testing.expect((try requireInstructionStateBlockForPlan(&stateful_insn, &plan, ok_blocks[0..])) != null);

    const bad_align_blocks = [_]StateBlockHandle{.{
        .id = @intFromEnum(StateBlockId.kv_cache),
        .ptr = bytes[0..].ptr,
        .size = 256,
        .align_bytes = 32,
    }};
    try std.testing.expectError(
        error.InvalidStateDescriptorBinding,
        requireInstructionStateBlockForPlan(&stateful_insn, &plan, bad_align_blocks[0..]),
    );

    const bad_size_blocks = [_]StateBlockHandle{.{
        .id = @intFromEnum(StateBlockId.kv_cache),
        .ptr = bytes[0..].ptr,
        .size = 64,
        .align_bytes = 64,
    }};
    try std.testing.expectError(
        error.InvalidStateDescriptorBinding,
        requireInstructionStateBlockForPlan(&stateful_insn, &plan, bad_size_blocks[0..]),
    );
}

test "validateStateBlocksForDescriptors enforces descriptor coverage and constraints" {
    const descriptors = [_]StateDescriptor{
        .{
            .id = @intFromEnum(StateBlockId.kv_cache),
            .size_bytes = 64,
            .align_bytes = 64,
            .zero_init = false,
            .lifecycle = .slot_persistent,
        },
        .{
            .id = @intFromEnum(StateBlockId.shortconv),
            .size_bytes = 32,
            .align_bytes = 32,
            .zero_init = true,
            .lifecycle = .slot_persistent,
        },
    };

    var kv_bytes align(64) = [_]u8{0} ** 64;
    var sc_bytes align(64) = [_]u8{0} ** 64;
    const ok_blocks = [_]StateBlockHandle{
        .{
            .id = @intFromEnum(StateBlockId.kv_cache),
            .ptr = kv_bytes[0..].ptr,
            .size = 64,
            .align_bytes = 64,
        },
        .{
            .id = @intFromEnum(StateBlockId.shortconv),
            .ptr = sc_bytes[0..].ptr,
            .size = 64,
            .align_bytes = 64,
        },
    };
    try validateStateBlocksForDescriptors(descriptors[0..], ok_blocks[0..]);

    const missing_shortconv = [_]StateBlockHandle{ok_blocks[0]};
    try std.testing.expectError(
        error.InvalidStateDescriptorBinding,
        validateStateBlocksForDescriptors(descriptors[0..], missing_shortconv[0..]),
    );

    const bad_align_blocks = [_]StateBlockHandle{
        ok_blocks[0],
        .{
            .id = @intFromEnum(StateBlockId.shortconv),
            .ptr = sc_bytes[0..].ptr,
            .size = 64,
            .align_bytes = 16,
        },
    };
    try std.testing.expectError(
        error.InvalidStateDescriptorBinding,
        validateStateBlocksForDescriptors(descriptors[0..], bad_align_blocks[0..]),
    );
}

test "validateExecutionPlan rejects missing state binding for stateful opcode" {
    const insn = Instruction{
        .opcode = .shortconv,
        .inputs = &.{registerFromIndex(0)},
        .outputs = &.{registerFromIndex(1)},
        .weights = &.{},
        .param_block_id = null,
        .state_block_id = null,
    };
    const plan = ExecutionPlan{
        .instructions = &.{insn},
        .register_count = 2,
        .state_descs = &.{defaultStateDescriptor(.kv_cache)},
    };
    try std.testing.expectError(error.InvalidStateDescriptorBinding, validateExecutionPlan(&plan));
}

test "validateExecutionPlan allows optional attention state binding" {
    const insn = Instruction{
        .opcode = .multihead_attention,
        .inputs = &.{registerFromIndex(0)},
        .outputs = &.{registerFromIndex(1)},
        .weights = &.{},
        .param_block_id = null,
        .state_block_id = null,
    };
    const plan = ExecutionPlan{
        .instructions = &.{insn},
        .register_count = 2,
        .state_descs = &.{},
    };
    try validateExecutionPlan(&plan);
}

test "validateExecutionPlan rejects mismatched state binding for opcode" {
    const insn = Instruction{
        .opcode = .multihead_attention,
        .inputs = &.{registerFromIndex(0)},
        .outputs = &.{registerFromIndex(1)},
        .weights = &.{},
        .param_block_id = null,
        .state_block_id = @intFromEnum(StateBlockId.shortconv),
    };
    const plan = ExecutionPlan{
        .instructions = &.{insn},
        .register_count = 2,
        .state_descs = &.{
            defaultStateDescriptor(.kv_cache),
            defaultStateDescriptor(.shortconv),
        },
    };
    try std.testing.expectError(error.InvalidStateDescriptorBinding, validateExecutionPlan(&plan));
}

test "validateExecutionPlan rejects unexpected state binding for stateless opcode" {
    const insn = Instruction{
        .opcode = .residual_add,
        .inputs = &.{ registerFromIndex(0), registerFromIndex(1) },
        .outputs = &.{registerFromIndex(0)},
        .weights = &.{},
        .param_block_id = null,
        .state_block_id = @intFromEnum(StateBlockId.kv_cache),
    };
    const plan = ExecutionPlan{
        .instructions = &.{insn},
        .register_count = 2,
        .state_descs = &.{defaultStateDescriptor(.kv_cache)},
    };
    try std.testing.expectError(error.InvalidStateDescriptorBinding, validateExecutionPlan(&plan));
}

test "validateExecutionPlanForBlockKind rejects incompatible stateful opcode for block kind" {
    const insn = Instruction{
        .opcode = .shortconv,
        .inputs = &.{registerFromIndex(0)},
        .outputs = &.{registerFromIndex(1)},
        .weights = &.{},
        .param_block_id = null,
        .state_block_id = @intFromEnum(StateBlockId.shortconv),
    };
    const plan = ExecutionPlan{
        .instructions = &.{insn},
        .register_count = 2,
        .state_descs = &.{defaultStateDescriptor(.shortconv)},
    };
    try std.testing.expectError(
        error.InvalidStateDescriptorBinding,
        validateExecutionPlanForBlockKind(&plan, .attention_mlp),
    );
}

test "planFinalOutputRegister returns residual register for empty plan" {
    const plan = ExecutionPlan{
        .instructions = &.{},
        .register_count = 0,
        .state_descs = &.{},
    };
    try std.testing.expectEqual(@as(u16, 0), registerToIndex(planFinalOutputRegister(&plan)));
}

test "planFinalOutputRegister returns last instruction output register" {
    const insn = Instruction{
        .opcode = .split,
        .inputs = &.{registerFromIndex(0)},
        .outputs = &.{ registerFromIndex(3), registerFromIndex(7) },
        .weights = &.{},
        .param_block_id = null,
        .state_block_id = null,
    };
    const plan = ExecutionPlan{
        .instructions = &.{insn},
        .register_count = 8,
        .state_descs = &.{},
    };
    try std.testing.expectEqual(@as(u16, 7), registerToIndex(planFinalOutputRegister(&plan)));
}

test "planUsesInstructionWeights detects weighted instructions" {
    const unweighted = Instruction{
        .opcode = .rmsnorm,
        .inputs = &.{registerFromIndex(0)},
        .outputs = &.{registerFromIndex(1)},
        .weights = &.{},
        .param_block_id = null,
        .state_block_id = null,
    };
    const weighted = Instruction{
        .opcode = .linear,
        .inputs = &.{registerFromIndex(1)},
        .outputs = &.{registerFromIndex(2)},
        .weights = &.{.{ .index = 0 }},
        .param_block_id = null,
        .state_block_id = null,
    };
    const plan_without_weights = ExecutionPlan{
        .instructions = &.{unweighted},
        .register_count = 3,
        .state_descs = &.{},
    };
    const plan_with_weights = ExecutionPlan{
        .instructions = &.{ unweighted, weighted },
        .register_count = 3,
        .state_descs = &.{},
    };
    try std.testing.expect(!planUsesInstructionWeights(&plan_without_weights));
    try std.testing.expect(planUsesInstructionWeights(&plan_with_weights));
}

test "validatePlanWithoutInstructionWeights rejects bound-weight plans" {
    const weighted = Instruction{
        .opcode = .linear,
        .inputs = &.{registerFromIndex(0)},
        .outputs = &.{registerFromIndex(1)},
        .weights = &.{.{ .index = 0 }},
        .param_block_id = null,
        .state_block_id = null,
    };
    const compiled = CompiledPlan{
        .plan = .{
            .instructions = &.{weighted},
            .register_count = 2,
            .state_descs = &.{},
        },
        .param_blocks = &.{},
        .weight_bindings = &.{.{
            .index = 0,
            .name = "w",
        }},
        .liveness = .{
            .register_last_read = &.{ 0, 0 },
            .kill_after_instruction = &.{&.{0}},
        },
        .peak_registers = 2,
        .diagnostics = &.{},
    };
    try std.testing.expectError(error.InvalidWeightRefCount, validatePlanWithoutInstructionWeights(&compiled));
}

test "validateCompiledPlan enforces liveness dimensions" {
    const insn = Instruction{
        .opcode = .rmsnorm,
        .inputs = &.{registerFromIndex(0)},
        .outputs = &.{registerFromIndex(1)},
        .weights = &.{.{ .index = 0 }},
        .param_block_id = null,
        .state_block_id = null,
    };
    const plan = ExecutionPlan{
        .instructions = &.{insn},
        .register_count = 2,
        .state_descs = &.{},
    };
    const liveness = LivenessMap{
        .register_last_read = &.{ 0, 0 },
        .kill_after_instruction = &.{&.{0}},
    };
    const compiled = CompiledPlan{
        .plan = plan,
        .param_blocks = &.{},
        .weight_bindings = &.{.{ .index = 0, .name = "norm_w" }},
        .register_buffer_specs = &.{
            .{ .size = 1, .@"align" = 64 },
            .{ .size = 1, .@"align" = 64 },
        },
        .liveness = liveness,
        .peak_registers = 2,
        .diagnostics = &.{},
    };
    try validateCompiledPlan(&compiled);
}

test "validateCompiledPlan rejects out-of-range instruction weight refs" {
    const insn = Instruction{
        .opcode = .linear,
        .inputs = &.{registerFromIndex(0)},
        .outputs = &.{registerFromIndex(1)},
        .weights = &.{.{ .index = 1 }},
        .param_block_id = null,
        .state_block_id = null,
    };
    const plan = ExecutionPlan{
        .instructions = &.{insn},
        .register_count = 2,
        .state_descs = &.{},
    };
    const liveness = LivenessMap{
        .register_last_read = &.{ 0, 0 },
        .kill_after_instruction = &.{&.{0}},
    };
    const compiled = CompiledPlan{
        .plan = plan,
        .param_blocks = &.{},
        .weight_bindings = &.{.{ .index = 0, .name = "w0" }},
        .register_buffer_specs = &.{
            .{ .size = 1, .@"align" = 64 },
            .{ .size = 1, .@"align" = 64 },
        },
        .liveness = liveness,
        .peak_registers = 2,
        .diagnostics = &.{},
    };
    try std.testing.expectError(error.InvalidWeightRefIndex, validateCompiledPlan(&compiled));
}

test "validateCompiledPlan rejects invalid instruction weight ref count for opcode" {
    const insn = Instruction{
        .opcode = .add_param,
        .inputs = &.{registerFromIndex(0)},
        .outputs = &.{registerFromIndex(1)},
        .weights = &.{},
        .param_block_id = null,
        .state_block_id = null,
    };
    const plan = ExecutionPlan{
        .instructions = &.{insn},
        .register_count = 2,
        .state_descs = &.{},
    };
    const liveness = LivenessMap{
        .register_last_read = &.{ 0, 0 },
        .kill_after_instruction = &.{&.{0}},
    };
    const compiled = CompiledPlan{
        .plan = plan,
        .param_blocks = &.{},
        .weight_bindings = &.{},
        .register_buffer_specs = &.{
            .{ .size = 1, .@"align" = 64 },
            .{ .size = 1, .@"align" = 64 },
        },
        .liveness = liveness,
        .peak_registers = 2,
        .diagnostics = &.{},
    };
    try std.testing.expectError(error.InvalidWeightRefCount, validateCompiledPlan(&compiled));
}

test "instructionSingleWeightBindingName resolves configured binding name" {
    const insn = Instruction{
        .opcode = .linear,
        .inputs = &.{registerFromIndex(0)},
        .outputs = &.{registerFromIndex(1)},
        .weights = &.{.{ .index = 0 }},
        .param_block_id = null,
        .state_block_id = null,
    };
    const compiled = CompiledPlan{
        .plan = .{
            .instructions = &.{insn},
            .register_count = 2,
            .state_descs = &.{},
        },
        .param_blocks = &.{},
        .weight_bindings = &.{.{ .index = 0, .name = "w0" }},
        .liveness = .{
            .register_last_read = &.{ 0, 0 },
            .kill_after_instruction = &.{&.{0}},
        },
        .peak_registers = 1,
        .diagnostics = &.{},
    };
    try std.testing.expectEqualStrings("w0", try instructionSingleWeightBindingName(&compiled, 0));
}

test "instructionSingleWeightBindingName rejects non-single weight arity" {
    const insn = Instruction{
        .opcode = .linear,
        .inputs = &.{registerFromIndex(0)},
        .outputs = &.{registerFromIndex(1)},
        .weights = &.{},
        .param_block_id = null,
        .state_block_id = null,
    };
    const compiled = CompiledPlan{
        .plan = .{
            .instructions = &.{insn},
            .register_count = 2,
            .state_descs = &.{},
        },
        .param_blocks = &.{},
        .weight_bindings = &.{},
        .liveness = .{
            .register_last_read = &.{ 0, 0 },
            .kill_after_instruction = &.{&.{0}},
        },
        .peak_registers = 1,
        .diagnostics = &.{},
    };
    try std.testing.expectError(error.InvalidWeightRefCount, instructionSingleWeightBindingName(&compiled, 0));
}

test "parseKernelWeightBindingName parses kernel id and slot name" {
    const parsed = try parseKernelWeightBindingName("__kernel_weight::7::q_proj::12");
    try std.testing.expectEqual(@as(u32, 7), parsed.kernel_id);
    try std.testing.expectEqualStrings("q_proj", parsed.slot_name);
}

test "parseKernelWeightBindingName rejects malformed names" {
    try std.testing.expectError(error.InvalidWeightBindingName, parseKernelWeightBindingName("q_proj"));
    try std.testing.expectError(error.InvalidWeightBindingName, parseKernelWeightBindingName("__kernel_weight::x::q_proj::0"));
    try std.testing.expectError(error.InvalidWeightBindingName, parseKernelWeightBindingName("__kernel_weight::1::::0"));
    try std.testing.expectError(error.InvalidWeightBindingName, parseKernelWeightBindingName("__kernel_weight::1::q_proj::"));
}

test "expectedWeightRefCount returns macro-op arities" {
    try std.testing.expectEqual(@as(usize, 4), expectedWeightRefCount(.multihead_attention));
    try std.testing.expectEqual(@as(usize, 3), expectedWeightRefCount(.swiglu));
    try std.testing.expectEqual(@as(usize, 2), expectedWeightRefCount(.mamba_mixer));
    try std.testing.expectEqual(@as(usize, 3), expectedWeightRefCount(.shortconv));
}

test "AdapterTable keeps 256 opcode slots" {
    try std.testing.expectEqual(@as(usize, 256), @typeInfo(AdapterTable).array.len);
}

test "stateBlockIdForOpcode maps stateful macro ops" {
    try std.testing.expectEqual(
        @as(?u8, @intFromEnum(StateBlockId.kv_cache)),
        stateBlockIdForOpcode(.multihead_attention),
    );
    try std.testing.expectEqual(
        @as(?u8, @intFromEnum(StateBlockId.shortconv)),
        stateBlockIdForOpcode(.shortconv),
    );
    try std.testing.expectEqual(
        @as(?u8, @intFromEnum(StateBlockId.mamba)),
        stateBlockIdForOpcode(.mamba_mixer),
    );
    try std.testing.expectEqual(@as(?u8, null), stateBlockIdForOpcode(.residual_add));
}

test "requiredStateBlockIdForOpcode only requires recurrent state ops" {
    try std.testing.expectEqual(@as(?u8, null), requiredStateBlockIdForOpcode(.multihead_attention));
    try std.testing.expectEqual(
        @as(?u8, @intFromEnum(StateBlockId.shortconv)),
        requiredStateBlockIdForOpcode(.shortconv),
    );
    try std.testing.expectEqual(
        @as(?u8, @intFromEnum(StateBlockId.mamba)),
        requiredStateBlockIdForOpcode(.mamba_mixer),
    );
    try std.testing.expectEqual(@as(?u8, null), requiredStateBlockIdForOpcode(.residual_add));
}

test "defaultStateDescriptor uses stable v1 compatibility defaults" {
    const kv_desc = defaultStateDescriptor(.kv_cache);
    try std.testing.expectEqual(@as(u8, @intFromEnum(StateBlockId.kv_cache)), kv_desc.id);
    try std.testing.expectEqual(@as(u16, 64), kv_desc.align_bytes);
    try std.testing.expect(!kv_desc.zero_init);
    try std.testing.expectEqual(StateLifecycle.slot_persistent, kv_desc.lifecycle);
}

test "stateLifecyclePolicy defines matrix for all lifecycle classes" {
    const slot = stateLifecyclePolicy(.slot_persistent);
    try std.testing.expect(slot.allow_reset);
    try std.testing.expect(slot.allow_reuse);
    try std.testing.expect(slot.allow_evict);
    try std.testing.expect(!slot.allow_clone_for_fork);

    const request = stateLifecyclePolicy(.request_scoped);
    try std.testing.expect(!request.allow_reset);
    try std.testing.expect(!request.allow_reuse);
    try std.testing.expect(request.allow_evict);

    const step = stateLifecyclePolicy(.step_scoped);
    try std.testing.expect(step.allow_reset);
    try std.testing.expect(!step.allow_reuse);
    try std.testing.expect(step.allow_evict);
    try std.testing.expect(step.zero_on_alloc);
    try std.testing.expect(step.zero_on_reset);
}

test "validateStateLifecycleAction enforces lifecycle action support matrix" {
    try validateStateLifecycleAction(.slot_persistent, .reset);
    try validateStateLifecycleAction(.slot_persistent, .reuse);
    try std.testing.expectError(error.InvalidStateLifecycleAction, validateStateLifecycleAction(.request_scoped, .reset));
    try std.testing.expectError(error.InvalidStateLifecycleAction, validateStateLifecycleAction(.request_scoped, .reuse));
    try std.testing.expectError(error.InvalidStateLifecycleAction, validateStateLifecycleAction(.step_scoped, .clone_for_fork));
}

test "shouldZeroStateForLifecycleAction applies descriptor and lifecycle policy" {
    const slot_desc = StateDescriptor{
        .id = @intFromEnum(StateBlockId.kv_cache),
        .size_bytes = 0,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
    };
    try std.testing.expect(!(try shouldZeroStateForLifecycleAction(&slot_desc, .alloc)));
    try std.testing.expect(!(try shouldZeroStateForLifecycleAction(&slot_desc, .reset)));
    try std.testing.expect(!(try shouldZeroStateForLifecycleAction(&slot_desc, .evict)));

    const request_desc = StateDescriptor{
        .id = @intFromEnum(StateBlockId.shortconv),
        .size_bytes = 128,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .request_scoped,
    };
    try std.testing.expect(try shouldZeroStateForLifecycleAction(&request_desc, .alloc));
    try std.testing.expect(try shouldZeroStateForLifecycleAction(&request_desc, .evict));
    try std.testing.expectError(error.InvalidStateLifecycleAction, shouldZeroStateForLifecycleAction(&request_desc, .reset));
}

test "collectBuiltinStateFlags aggregates known descriptor ids" {
    const plan = ExecutionPlan{
        .instructions = &.{},
        .register_count = 1,
        .state_descs = &.{
            defaultStateDescriptor(.kv_cache),
            defaultStateDescriptor(.shortconv),
        },
    };
    const flags = try collectBuiltinStateFlags(&plan);
    try std.testing.expect(flags.has_kv);
    try std.testing.expect(flags.has_shortconv);
    try std.testing.expect(!flags.has_mamba);
}

test "collectBuiltinStateFlags rejects non-slot persistent lifecycle for builtin states" {
    const plan = ExecutionPlan{
        .instructions = &.{},
        .register_count = 1,
        .state_descs = &.{
            .{
                .id = @intFromEnum(StateBlockId.kv_cache),
                .size_bytes = 0,
                .align_bytes = 64,
                .zero_init = false,
                .lifecycle = .request_scoped,
            },
        },
    };
    try std.testing.expectError(error.InvalidStateDescriptorBinding, collectBuiltinStateFlags(&plan));
}

test "stateDescriptorIndex finds descriptor by id" {
    const descriptors = [_]StateDescriptor{
        defaultStateDescriptor(.kv_cache),
        defaultStateDescriptor(.shortconv),
    };
    try std.testing.expectEqual(@as(?usize, 0), stateDescriptorIndex(descriptors[0..], @intFromEnum(StateBlockId.kv_cache)));
    try std.testing.expectEqual(@as(?usize, 1), stateDescriptorIndex(descriptors[0..], @intFromEnum(StateBlockId.shortconv)));
    try std.testing.expectEqual(@as(?usize, null), stateDescriptorIndex(descriptors[0..], @intFromEnum(StateBlockId.mamba)));
}

test "appendUniqueStateDescriptor deduplicates identical descriptor and rejects mismatch" {
    var storage: [3]StateDescriptor = undefined;
    var count: u8 = 0;
    const kv_desc = defaultStateDescriptor(.kv_cache);
    try appendUniqueStateDescriptor(storage[0..], &count, kv_desc);
    try std.testing.expectEqual(@as(u8, 1), count);
    try appendUniqueStateDescriptor(storage[0..], &count, kv_desc);
    try std.testing.expectEqual(@as(u8, 1), count);

    var mismatched = kv_desc;
    mismatched.align_bytes = 32;
    try std.testing.expectError(
        error.InvalidStateDescriptorBinding,
        appendUniqueStateDescriptor(storage[0..], &count, mismatched),
    );
}

test "appendUniquePlanStateDescriptors merges unique builtin descriptors" {
    var storage: [3]StateDescriptor = undefined;
    var count: u8 = 0;
    const plan_a = ExecutionPlan{
        .instructions = &.{},
        .register_count = 1,
        .state_descs = &.{
            defaultStateDescriptor(.kv_cache),
            defaultStateDescriptor(.shortconv),
        },
    };
    const plan_b = ExecutionPlan{
        .instructions = &.{},
        .register_count = 1,
        .state_descs = &.{
            defaultStateDescriptor(.shortconv),
            defaultStateDescriptor(.mamba),
        },
    };
    try appendUniquePlanStateDescriptors(storage[0..], &count, &plan_a);
    try appendUniquePlanStateDescriptors(storage[0..], &count, &plan_b);
    try std.testing.expectEqual(@as(u8, 3), count);
    const used = storage[0..@as(usize, @intCast(count))];
    try std.testing.expectEqual(@as(?usize, 0), stateDescriptorIndex(used, @intFromEnum(StateBlockId.kv_cache)));
    try std.testing.expectEqual(@as(?usize, 1), stateDescriptorIndex(used, @intFromEnum(StateBlockId.shortconv)));
    try std.testing.expectEqual(@as(?usize, 2), stateDescriptorIndex(used, @intFromEnum(StateBlockId.mamba)));
}

test "decodeInstructionLayerOp validates param block abi version" {
    const op: layer_ops.LayerOp = .{
        .add = .{
            .branch = .branch_out,
            .scale = .one,
        },
    };
    const kill_row = [_]u64{0};
    const kill_rows = [_][]const u64{kill_row[0..]};
    const insn = Instruction{
        .opcode = opcode_map.opcodeForLayerOp(op),
        .inputs = &.{},
        .outputs = &.{registerFromIndex(0)},
        .weights = &.{},
        .param_block_id = 0,
        .state_block_id = null,
    };
    var param_block = try encodeLayerOpParam(std.testing.allocator, insn.opcode, op);
    defer if (param_block.data.len != 0) std.testing.allocator.free(param_block.data);
    param_block.version +%= 1;
    const param_blocks = [_]ParamBlock{param_block};
    const compiled = CompiledPlan{
        .plan = .{
            .instructions = &.{insn},
            .register_count = 1,
            .state_descs = &.{},
        },
        .liveness = .{
            .register_last_read = &.{0},
            .kill_after_instruction = kill_rows[0..],
        },
        .peak_registers = 1,
        .diagnostics = &.{},
        .weight_bindings = &.{},
        .param_blocks = param_blocks[0..],
    };
    try std.testing.expectError(error.InvalidParamBlockABI, decodeInstructionLayerOp(&compiled, &compiled.plan.instructions[0], 0));
}

test "decodeInstructionLayerOp returns decoded layer op when abi is valid" {
    const op: layer_ops.LayerOp = .{
        .add = .{
            .branch = .branch_out,
            .scale = .one,
        },
    };
    const kill_row = [_]u64{0};
    const kill_rows = [_][]const u64{kill_row[0..]};
    const insn = Instruction{
        .opcode = opcode_map.opcodeForLayerOp(op),
        .inputs = &.{},
        .outputs = &.{registerFromIndex(0)},
        .weights = &.{},
        .param_block_id = 0,
        .state_block_id = null,
    };
    const param_block = try encodeLayerOpParam(std.testing.allocator, insn.opcode, op);
    defer if (param_block.data.len != 0) std.testing.allocator.free(param_block.data);
    const param_blocks = [_]ParamBlock{param_block};
    const compiled = CompiledPlan{
        .plan = .{
            .instructions = &.{insn},
            .register_count = 1,
            .state_descs = &.{},
        },
        .liveness = .{
            .register_last_read = &.{0},
            .kill_after_instruction = kill_rows[0..],
        },
        .peak_registers = 1,
        .diagnostics = &.{},
        .weight_bindings = &.{},
        .param_blocks = param_blocks[0..],
    };
    const decoded = try decodeInstructionLayerOp(&compiled, &compiled.plan.instructions[0], 0);
    try std.testing.expect(std.meta.eql(op, decoded));
}

test "encodeLayerOpParam strips runtime param names for bound primitive ops" {
    const op: layer_ops.LayerOp = .{
        .add_param = .{
            .in = .tmp3,
            .out = .tmp4,
            .param_name = "w_param",
        },
    };
    const opcode = opcode_map.opcodeForLayerOp(op);
    const param_block = try encodeLayerOpParam(std.testing.allocator, opcode, op);
    defer if (param_block.data.len != 0) std.testing.allocator.free(param_block.data);

    const decoded = try decodeLayerOpFromParam(opcode, param_block.data);
    switch (decoded) {
        .add_param => |decoded_op| {
            try std.testing.expectEqual(@as(usize, 0), decoded_op.param_name.len);
            try std.testing.expectEqual(layer_ops.BufferId.tmp3, decoded_op.in);
            try std.testing.expectEqual(layer_ops.BufferId.tmp4, decoded_op.out);
        },
        else => return error.TestUnexpectedResult,
    }
}

test "encodeLayerOpParam round-trips split metadata including explicit split sizes" {
    const split_sizes = [_]usize{ 2, 4, 8 };
    const op: layer_ops.LayerOp = .{
        .split = .{
            .in = .tmp10,
            .out_start = .tmp11,
            .num_outputs = 3,
            .dim = 1,
            .split_sizes = split_sizes[0..],
        },
    };
    const opcode = opcode_map.opcodeForLayerOp(op);
    const param_block = try encodeLayerOpParam(std.testing.allocator, opcode, op);
    defer if (param_block.data.len != 0) std.testing.allocator.free(param_block.data);

    const decoded = try decodeLayerOpFromParam(opcode, param_block.data);
    switch (decoded) {
        .split => |decoded_op| {
            try std.testing.expectEqual(layer_ops.BufferId.tmp10, decoded_op.in);
            try std.testing.expectEqual(layer_ops.BufferId.tmp11, decoded_op.out_start);
            try std.testing.expectEqual(@as(u8, 3), decoded_op.num_outputs);
            try std.testing.expectEqual(@as(i8, 1), decoded_op.dim);
            try std.testing.expectEqualSlices(usize, split_sizes[0..], decoded_op.split_sizes);
        },
        else => return error.TestUnexpectedResult,
    }
}

test "blockKindSupportsState maps canonical block-state compatibility" {
    try std.testing.expect(blockKindSupportsState(.attention_mlp, @intFromEnum(StateBlockId.kv_cache)));
    try std.testing.expect(blockKindSupportsState(.shortconv, @intFromEnum(StateBlockId.shortconv)));
    try std.testing.expect(blockKindSupportsState(.mamba, @intFromEnum(StateBlockId.mamba)));
    try std.testing.expect(!blockKindSupportsState(.attention_mlp, @intFromEnum(StateBlockId.shortconv)));
    try std.testing.expect(!blockKindSupportsState(.shortconv, @intFromEnum(StateBlockId.kv_cache)));
    try std.testing.expect(!blockKindSupportsState(.mamba, @intFromEnum(StateBlockId.kv_cache)));
}

test "opcodeStateCompatibleWithBlockKind enforces stateful opcode topology compatibility" {
    try std.testing.expect(opcodeStateCompatibleWithBlockKind(.residual_add, .attention_mlp));
    try std.testing.expect(opcodeStateCompatibleWithBlockKind(.multihead_attention, .attention_mlp));
    try std.testing.expect(!opcodeStateCompatibleWithBlockKind(.multihead_attention, .shortconv));
    try std.testing.expect(opcodeStateCompatibleWithBlockKind(.shortconv, .shortconv));
    try std.testing.expect(!opcodeStateCompatibleWithBlockKind(.shortconv, .mamba));
}

test "firstLayerProgramStateMismatch returns first mismatched stateful opcode" {
    const program = [_]layer_ops.LayerOp{
        .{ .kernel = .{ .id = 0, .in = .residual, .out = .norm_out, .debug_type = .norm } },
        .{ .kernel = .{ .id = 1, .in = .norm_out, .out = .branch_out, .debug_type = .shortconv } },
    };
    const mismatch = firstLayerProgramStateMismatch(&program, .attention_mlp) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(usize, 1), mismatch.op_index);
    try std.testing.expectEqual(Opcode.shortconv, mismatch.opcode);
    try std.testing.expectEqual(@as(u8, @intFromEnum(StateBlockId.shortconv)), mismatch.state_id);
}

test "firstUnsupportedLayerProgramOpcode returns first unsupported opcode in program" {
    const table = [_]?u8{null} ** 256;
    const program = [_]layer_ops.LayerOp{
        .{ .kernel = .{
            .id = 0,
            .in = .residual,
            .out = .norm_out,
            .debug_type = .norm,
        } },
    };

    const unsupported = firstUnsupportedLayerProgramOpcode(&program, table) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(usize, 0), unsupported.op_index);
    try std.testing.expectEqual(Opcode.rmsnorm, unsupported.opcode);
}

test "firstUnsupportedInstructionOpcode returns first unsupported opcode in plan" {
    const instructions = [_]Instruction{
        .{
            .opcode = .rmsnorm,
            .inputs = &.{registerFromIndex(0)},
            .outputs = &.{registerFromIndex(1)},
            .weights = &.{},
            .param_block_id = null,
            .state_block_id = null,
        },
        .{
            .opcode = .swiglu,
            .inputs = &.{registerFromIndex(1)},
            .outputs = &.{registerFromIndex(0)},
            .weights = &.{},
            .param_block_id = null,
            .state_block_id = null,
        },
    };
    const plan = ExecutionPlan{
        .instructions = &instructions,
        .register_count = 2,
        .state_descs = &.{},
    };

    var table = [_]?u8{null} ** 256;
    table[@intFromEnum(Opcode.rmsnorm)] = 1;

    const unsupported = firstUnsupportedInstructionOpcode(&plan, table) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(usize, 1), unsupported.instruction_index);
    try std.testing.expectEqual(Opcode.swiglu, unsupported.opcode);
}

test "assertAdapterTableCoverage requires all declared opcodes to be populated" {
    comptime {
        var table = [_]?u8{null} ** 256;
        table[@intFromEnum(Opcode.rmsnorm)] = 1;
        assertAdapterTableCoverage(
            table,
            [_]Opcode{.rmsnorm},
            "runtime_contract.types.test_adapter_table",
        );
    }
}

test "firstLayerProgramCompatibilityIssue reports unsupported opcode first" {
    const table = [_]?u8{null} ** 256;
    const program = [_]layer_ops.LayerOp{
        .{ .kernel = .{
            .id = 0,
            .in = .residual,
            .out = .norm_out,
            .debug_type = .norm,
        } },
    };
    const issue = firstLayerProgramCompatibilityIssue(&program, .attention_mlp, table) orelse return error.TestUnexpectedResult;
    switch (issue) {
        .unsupported_opcode => |unsupported| {
            try std.testing.expectEqual(@as(usize, 0), unsupported.op_index);
            try std.testing.expectEqual(Opcode.rmsnorm, unsupported.opcode);
        },
        else => return error.TestUnexpectedResult,
    }
}

test "firstLayerProgramCompatibilityIssue reports state mismatch when opcodes are supported" {
    var table = [_]?u8{null} ** 256;
    table[@intFromEnum(Opcode.shortconv)] = 1;
    const program = [_]layer_ops.LayerOp{
        .{ .kernel = .{
            .id = 0,
            .in = .residual,
            .out = .branch_out,
            .debug_type = .shortconv,
        } },
    };
    const issue = firstLayerProgramCompatibilityIssue(&program, .attention_mlp, table) orelse return error.TestUnexpectedResult;
    switch (issue) {
        .state_mismatch => |mismatch| {
            try std.testing.expectEqual(@as(usize, 0), mismatch.op_index);
            try std.testing.expectEqual(Opcode.shortconv, mismatch.opcode);
            try std.testing.expectEqual(@as(u8, @intFromEnum(StateBlockId.shortconv)), mismatch.state_id);
        },
        else => return error.TestUnexpectedResult,
    }
}

test "paramAs round-trip with encodeLayerOpParam" {
    const param_block = try encodeLayerOpParam(
        std.testing.allocator,
        .residual_add,
        .{ .add = .{ .branch = .norm_out, .scale = .{ .literal = 1.5 } } },
    );
    defer std.testing.allocator.free(param_block.data);

    const p = try paramAs(ResidualAddParam, &.{param_block}, .residual_add);
    // norm_out = 1 in BufferId enum
    try std.testing.expectEqual(@as(u8, 1), p.branch_buffer_id);
    // literal = scale_tag 2
    try std.testing.expectEqual(@as(u8, 2), p.scale_tag);
    // f32 1.5 stored as u32 via @bitCast
    try std.testing.expectEqual(@as(u32, @bitCast(@as(f32, 1.5))), p.scale_literal);
}
