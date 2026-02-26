//! Typed runtime contracts for generic inference execution plans.
//!
//! This module contains backend-agnostic plan/execution ABI types.

const std = @import("std");
const dtype = @import("../../dtype.zig");
const op_types = @import("../../models/op_types.zig");
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

pub const ParamBlock = struct {
    version: u8,
    opcode: Opcode,
    data: []const u8,
};

pub const ExecutionMode = enum(u8) {
    decode = 0,
    prefill = 1,
    vision_encode = 2,
    scatter = 3,
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
    workspace: Workspace = .{},
};

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

pub fn stateBlockIdForOpcode(opcode: Opcode) ?u8 {
    return switch (opcode) {
        .multihead_attention, .mla_attention => @intFromEnum(StateBlockId.kv_cache),
        .shortconv => @intFromEnum(StateBlockId.shortconv),
        .mamba_mixer => @intFromEnum(StateBlockId.mamba),
        else => null,
    };
}

pub fn expectedWeightRefCount(opcode: Opcode) usize {
    return switch (opcode) {
        .linear, .add_param, .add_param_scalar, .mul_param => 1,
        else => 0,
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

pub fn validateTensorViewDesc(view: *const TensorViewDesc) !void {
    if (view.rank > 4) return error.UnsupportedTensorRank;
}

pub fn validateExecutionContext(ctx: *const ExecutionContext) !void {
    if (ctx.batch_size != ctx.active_slots.len) return error.InvalidBatchSize;
    if (ctx.sequence_lengths.len != ctx.active_slots.len) return error.InvalidSequenceLengthCount;
}

pub fn validateExecutionPlan(plan: *const ExecutionPlan) !void {
    var state_seen: [256]bool = [_]bool{false} ** 256;
    for (plan.state_descs) |state_desc| {
        if (state_seen[state_desc.id]) return error.DuplicateStateDescriptorId;
        state_seen[state_desc.id] = true;
        if (state_desc.align_bytes == 0) return error.InvalidStateAlignment;
    }

    for (plan.instructions) |insn| {
        const expected_state_id = stateBlockIdForOpcode(insn.opcode);
        if (expected_state_id) |state_id| {
            if (insn.state_block_id == null or insn.state_block_id.? != state_id) {
                return error.InvalidStateDescriptorBinding;
            }
        } else if (insn.state_block_id != null) {
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
            if (param_block.opcode != insn.opcode) return error.ParamBlockOpcodeMismatch;
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
    view.rank = 4;
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

test "validateExecutionPlan rejects missing state binding for stateful opcode" {
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
        .state_descs = &.{defaultStateDescriptor(.kv_cache)},
    };
    try std.testing.expectError(error.InvalidStateDescriptorBinding, validateExecutionPlan(&plan));
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

test "validateCompiledPlan enforces liveness dimensions" {
    const insn = Instruction{
        .opcode = .rmsnorm,
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

test "defaultStateDescriptor uses stable v1 compatibility defaults" {
    const kv_desc = defaultStateDescriptor(.kv_cache);
    try std.testing.expectEqual(@as(u8, @intFromEnum(StateBlockId.kv_cache)), kv_desc.id);
    try std.testing.expectEqual(@as(u16, 64), kv_desc.align_bytes);
    try std.testing.expect(!kv_desc.zero_init);
    try std.testing.expectEqual(StateLifecycle.slot_persistent, kv_desc.lifecycle);
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
