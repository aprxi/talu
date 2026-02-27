//! Plan compiler (Phase 1b).
//!
//! Compiles existing `LayerOp` programs into typed runtime `CompiledPlan`.

const std = @import("std");
const layer_ops = @import("../layer_ops.zig");
const op_types = @import("../op_types.zig");
const registry = @import("../registry.zig");
const runtime_contract = @import("../../inference/runtime_contract/root.zig");
const opcode_map = @import("opcode_map.zig");
const llama3 = @import("../llama/llama3.zig");
const granite_hybrid = @import("../granite/granite_hybrid.zig");
const qwen3_moe = @import("../qwen/qwen3_moe.zig");

pub const CompileMode = runtime_contract.ExecutionMode;

const kv_state_id: u8 = @intFromEnum(runtime_contract.StateBlockId.kv_cache);
const shortconv_state_id: u8 = @intFromEnum(runtime_contract.StateBlockId.shortconv);
const mamba_state_id: u8 = @intFromEnum(runtime_contract.StateBlockId.mamba);

fn registerFromBuffer(buffer: layer_ops.BufferId) runtime_contract.RegisterRef {
    return runtime_contract.registerFromIndex(@intFromEnum(buffer));
}

fn allocRegistersFromBuffers(
    allocator: std.mem.Allocator,
    buffers: []const layer_ops.BufferId,
) ![]runtime_contract.RegisterRef {
    var regs = try allocator.alloc(runtime_contract.RegisterRef, buffers.len);
    for (buffers, 0..) |buffer, idx| regs[idx] = registerFromBuffer(buffer);
    return regs;
}

fn allocSequentialRegisters(
    allocator: std.mem.Allocator,
    first: layer_ops.BufferId,
    count: usize,
) ![]runtime_contract.RegisterRef {
    var regs = try allocator.alloc(runtime_contract.RegisterRef, count);
    const first_index: u16 = @intFromEnum(first);
    for (0..count) |idx| {
        regs[idx] = runtime_contract.registerFromIndex(first_index + @as(u16, @intCast(idx)));
    }
    return regs;
}

fn allocWeightRefs(
    allocator: std.mem.Allocator,
    refs: []const runtime_contract.WeightRef,
) ![]runtime_contract.WeightRef {
    if (refs.len == 0) return &.{};
    const owned = try allocator.alloc(runtime_contract.WeightRef, refs.len);
    @memcpy(owned, refs);
    return owned;
}

fn ensureWeightBindingIndex(
    allocator: std.mem.Allocator,
    bindings: *std.ArrayListUnmanaged(runtime_contract.WeightBinding),
    name: []const u8,
) !u32 {
    for (bindings.items) |binding| {
        if (std.mem.eql(u8, binding.name, name)) return binding.index;
    }

    const owned_name = try allocator.dupe(u8, name);
    errdefer allocator.free(owned_name);
    const binding_index: u32 = @intCast(bindings.items.len);
    try bindings.append(allocator, .{
        .index = binding_index,
        .name = owned_name,
    });
    return binding_index;
}

fn maxRegisterInInstruction(insn: runtime_contract.Instruction) u16 {
    var max_register: u16 = 0;
    for (insn.inputs) |reg| max_register = @max(max_register, runtime_contract.registerToIndex(reg));
    for (insn.outputs) |reg| max_register = @max(max_register, runtime_contract.registerToIndex(reg));
    return max_register;
}

fn compileOneInstruction(
    allocator: std.mem.Allocator,
    op: layer_ops.LayerOp,
    mode: CompileMode,
    param_block_id: u16,
    weight_bindings: *std.ArrayListUnmanaged(runtime_contract.WeightBinding),
) !runtime_contract.Instruction {
    const opcode = opcode_map.opcodeForLayerOp(op);
    const state_block_id = switch (mode) {
        .vision_encode, .scatter => switch (opcode) {
            .multihead_attention, .mla_attention => null,
            else => runtime_contract.stateBlockIdForOpcode(opcode),
        },
        else => runtime_contract.stateBlockIdForOpcode(opcode),
    };
    return switch (op) {
        .kernel => |kernel_op| .{
            .opcode = opcode,
            .inputs = try allocRegistersFromBuffers(allocator, &.{kernel_op.in}),
            .outputs = try allocRegistersFromBuffers(allocator, &.{kernel_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = state_block_id,
        },
        .add => |add_op| .{
            .opcode = opcode,
            .inputs = try allocRegistersFromBuffers(allocator, &.{ .residual, add_op.branch }),
            .outputs = try allocRegistersFromBuffers(allocator, &.{.residual}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = state_block_id,
        },
        .linear => |linear_op| .{
            .opcode = opcode,
            .inputs = try allocRegistersFromBuffers(allocator, &.{linear_op.in}),
            .outputs = try allocRegistersFromBuffers(allocator, &.{linear_op.out}),
            .weights = try allocWeightRefs(allocator, &.{.{
                .index = try ensureWeightBindingIndex(allocator, weight_bindings, linear_op.weight_name),
            }}),
            .param_block_id = param_block_id,
            .state_block_id = state_block_id,
        },
        .matmul => |matmul_op| .{
            .opcode = opcode,
            .inputs = try allocRegistersFromBuffers(allocator, &.{ matmul_op.in_a, matmul_op.in_b }),
            .outputs = try allocRegistersFromBuffers(allocator, &.{matmul_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = state_block_id,
        },
        .split => |split_op| .{
            .opcode = opcode,
            .inputs = try allocRegistersFromBuffers(allocator, &.{split_op.in}),
            .outputs = try allocSequentialRegisters(allocator, split_op.out_start, split_op.num_outputs),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = state_block_id,
        },
        .softmax => |softmax_op| .{
            .opcode = opcode,
            .inputs = try allocRegistersFromBuffers(allocator, &.{softmax_op.in}),
            .outputs = try allocRegistersFromBuffers(allocator, &.{softmax_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = state_block_id,
        },
        .silu => |silu_op| .{
            .opcode = opcode,
            .inputs = try allocRegistersFromBuffers(allocator, &.{silu_op.in}),
            .outputs = try allocRegistersFromBuffers(allocator, &.{silu_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = state_block_id,
        },
        .gelu => |gelu_op| .{
            .opcode = opcode,
            .inputs = try allocRegistersFromBuffers(allocator, &.{gelu_op.in}),
            .outputs = try allocRegistersFromBuffers(allocator, &.{gelu_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = state_block_id,
        },
        .mul => |mul_op| .{
            .opcode = opcode,
            .inputs = try allocRegistersFromBuffers(allocator, &.{ mul_op.in, mul_op.other }),
            .outputs = try allocRegistersFromBuffers(allocator, &.{mul_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = state_block_id,
        },
        .add_tensor => |add_tensor_op| .{
            .opcode = opcode,
            .inputs = try allocRegistersFromBuffers(allocator, &.{ add_tensor_op.in_a, add_tensor_op.in_b }),
            .outputs = try allocRegistersFromBuffers(allocator, &.{add_tensor_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = state_block_id,
        },
        .add_scalar => |add_scalar_op| .{
            .opcode = opcode,
            .inputs = try allocRegistersFromBuffers(allocator, &.{add_scalar_op.in}),
            .outputs = try allocRegistersFromBuffers(allocator, &.{add_scalar_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = state_block_id,
        },
        .mul_scalar => |mul_scalar_op| .{
            .opcode = opcode,
            .inputs = try allocRegistersFromBuffers(allocator, &.{mul_scalar_op.in}),
            .outputs = try allocRegistersFromBuffers(allocator, &.{mul_scalar_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = state_block_id,
        },
        .mean => |mean_op| .{
            .opcode = opcode,
            .inputs = try allocRegistersFromBuffers(allocator, &.{mean_op.in}),
            .outputs = try allocRegistersFromBuffers(allocator, &.{mean_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = state_block_id,
        },
        .pow => |pow_op| .{
            .opcode = opcode,
            .inputs = try allocRegistersFromBuffers(allocator, &.{pow_op.in}),
            .outputs = try allocRegistersFromBuffers(allocator, &.{pow_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = state_block_id,
        },
        .rsqrt => |rsqrt_op| .{
            .opcode = opcode,
            .inputs = try allocRegistersFromBuffers(allocator, &.{rsqrt_op.in}),
            .outputs = try allocRegistersFromBuffers(allocator, &.{rsqrt_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = state_block_id,
        },
        .add_param => |add_param_op| .{
            .opcode = opcode,
            .inputs = try allocRegistersFromBuffers(allocator, &.{add_param_op.in}),
            .outputs = try allocRegistersFromBuffers(allocator, &.{add_param_op.out}),
            .weights = try allocWeightRefs(allocator, &.{.{
                .index = try ensureWeightBindingIndex(allocator, weight_bindings, add_param_op.param_name),
            }}),
            .param_block_id = param_block_id,
            .state_block_id = state_block_id,
        },
        .add_param_scalar => |add_param_scalar_op| .{
            .opcode = opcode,
            .inputs = try allocRegistersFromBuffers(allocator, &.{}),
            .outputs = try allocRegistersFromBuffers(allocator, &.{add_param_scalar_op.out}),
            .weights = try allocWeightRefs(allocator, &.{.{
                .index = try ensureWeightBindingIndex(allocator, weight_bindings, add_param_scalar_op.param_name),
            }}),
            .param_block_id = param_block_id,
            .state_block_id = state_block_id,
        },
        .mul_param => |mul_param_op| .{
            .opcode = opcode,
            .inputs = try allocRegistersFromBuffers(allocator, &.{mul_param_op.in}),
            .outputs = try allocRegistersFromBuffers(allocator, &.{mul_param_op.out}),
            .weights = try allocWeightRefs(allocator, &.{.{
                .index = try ensureWeightBindingIndex(allocator, weight_bindings, mul_param_op.param_name),
            }}),
            .param_block_id = param_block_id,
            .state_block_id = state_block_id,
        },
        .reshape => |reshape_op| .{
            .opcode = opcode,
            .inputs = try allocRegistersFromBuffers(allocator, &.{reshape_op.in}),
            .outputs = try allocRegistersFromBuffers(allocator, &.{reshape_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = state_block_id,
        },
        .transpose => |transpose_op| .{
            .opcode = opcode,
            .inputs = try allocRegistersFromBuffers(allocator, &.{transpose_op.in}),
            .outputs = try allocRegistersFromBuffers(allocator, &.{transpose_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = state_block_id,
        },
        .rope => |rope_op| .{
            .opcode = opcode,
            .inputs = try allocRegistersFromBuffers(allocator, &.{rope_op.in}),
            .outputs = try allocRegistersFromBuffers(allocator, &.{rope_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = state_block_id,
        },
        .triu => |triu_op| .{
            .opcode = opcode,
            .inputs = try allocRegistersFromBuffers(allocator, &.{triu_op.in}),
            .outputs = try allocRegistersFromBuffers(allocator, &.{triu_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = state_block_id,
        },
        .sdpa => |sdpa_op| .{
            .opcode = opcode,
            .inputs = try allocRegistersFromBuffers(allocator, &.{ sdpa_op.q, sdpa_op.k, sdpa_op.v }),
            .outputs = try allocRegistersFromBuffers(allocator, &.{sdpa_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = state_block_id,
        },
        .patch_embed => |patch_op| .{
            .opcode = opcode,
            .inputs = try allocRegistersFromBuffers(allocator, &.{patch_op.in}),
            .outputs = try allocRegistersFromBuffers(allocator, &.{patch_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = state_block_id,
        },
        .spatial_merge => |spatial_op| .{
            .opcode = opcode,
            .inputs = try allocRegistersFromBuffers(allocator, &.{spatial_op.in}),
            .outputs = try allocRegistersFromBuffers(allocator, &.{spatial_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = state_block_id,
        },
        .deepstack_extract => |deepstack_op| .{
            .opcode = opcode,
            .inputs = try allocRegistersFromBuffers(allocator, &.{deepstack_op.in}),
            .outputs = try allocRegistersFromBuffers(allocator, &.{deepstack_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = state_block_id,
        },
        .scatter => |scatter_op| .{
            .opcode = opcode,
            .inputs = try allocRegistersFromBuffers(allocator, &.{ scatter_op.text_in, scatter_op.vision_in }),
            .outputs = try allocRegistersFromBuffers(allocator, &.{scatter_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = state_block_id,
        },
    };
}

fn serializeLayerOpParam(
    allocator: std.mem.Allocator,
    opcode: runtime_contract.Opcode,
    op: layer_ops.LayerOp,
) !runtime_contract.ParamBlock {
    return runtime_contract.encodeLayerOpParam(allocator, opcode, op);
}

fn deinitParamBlock(allocator: std.mem.Allocator, param_block: runtime_contract.ParamBlock) void {
    if (param_block.data.len != 0) allocator.free(param_block.data);
}

fn buildLivenessMap(
    allocator: std.mem.Allocator,
    instructions: []const runtime_contract.Instruction,
    register_count: u16,
) !runtime_contract.LivenessMap {
    const register_count_usize = @as(usize, register_count);
    const sentinel = std.math.maxInt(u32);
    var register_last_read = try allocator.alloc(u32, register_count_usize);
    for (register_last_read) |*entry| entry.* = sentinel;

    var produced = try allocator.alloc(bool, register_count_usize);
    defer allocator.free(produced);
    @memset(produced, false);

    for (instructions, 0..) |insn, instruction_idx| {
        const idx_u32: u32 = @intCast(instruction_idx);
        for (insn.inputs) |reg| {
            const reg_idx = runtime_contract.registerToIndex(reg);
            register_last_read[reg_idx] = idx_u32;
        }
        for (insn.outputs) |reg| {
            const reg_idx = runtime_contract.registerToIndex(reg);
            produced[reg_idx] = true;
        }
    }

    if (instructions.len > 0) {
        const terminal_idx: u32 = @intCast(instructions.len - 1);
        for (register_last_read, 0..) |last_read, idx| {
            if (produced[idx] and last_read == sentinel) {
                register_last_read[idx] = terminal_idx;
            }
        }
    }

    const words = runtime_contract.LivenessMap.bitsetWordCount(register_count);
    const kill_after_instruction = try allocator.alloc([]const u64, instructions.len);
    for (kill_after_instruction) |*row| {
        const mutable = try allocator.alloc(u64, words);
        @memset(mutable, 0);
        row.* = mutable;
    }

    for (register_last_read, 0..) |last_read, register_idx| {
        if (last_read == sentinel) continue;
        const instruction_idx: usize = @intCast(last_read);
        if (instruction_idx >= instructions.len) continue;
        const word = register_idx / 64;
        const bit: u6 = @intCast(register_idx % 64);
        const mutable: []u64 = @constCast(kill_after_instruction[instruction_idx]);
        mutable[word] |= (@as(u64, 1) << bit);
    }

    return .{
        .register_last_read = register_last_read,
        .kill_after_instruction = kill_after_instruction,
    };
}

fn buildStateDescriptors(
    allocator: std.mem.Allocator,
    instructions: []const runtime_contract.Instruction,
) ![]runtime_contract.StateDescriptor {
    var has_kv = false;
    var has_shortconv = false;
    var has_mamba = false;

    for (instructions) |insn| {
        const state_id = insn.state_block_id orelse continue;
        switch (state_id) {
            kv_state_id => has_kv = true,
            shortconv_state_id => has_shortconv = true,
            mamba_state_id => has_mamba = true,
            else => return error.UnknownStateDescriptorId,
        }
    }

    const state_count: usize = @as(usize, @intFromBool(has_kv)) +
        @as(usize, @intFromBool(has_shortconv)) +
        @as(usize, @intFromBool(has_mamba));
    if (state_count == 0) return &.{};

    const descriptors = try allocator.alloc(runtime_contract.StateDescriptor, state_count);
    var idx: usize = 0;
    if (has_kv) {
        descriptors[idx] = runtime_contract.defaultStateDescriptor(.kv_cache);
        idx += 1;
    }
    if (has_shortconv) {
        descriptors[idx] = runtime_contract.defaultStateDescriptor(.shortconv);
        idx += 1;
    }
    if (has_mamba) {
        descriptors[idx] = runtime_contract.defaultStateDescriptor(.mamba);
    }
    return descriptors;
}

fn computePeakRegisters(
    allocator: std.mem.Allocator,
    compiled: *const runtime_contract.CompiledPlan,
) !u16 {
    const reg_count = @as(usize, compiled.plan.register_count);
    if (reg_count == 0 or compiled.plan.instructions.len == 0) return 0;

    var live = try allocator.alloc(bool, reg_count);
    defer allocator.free(live);
    @memset(live, false);

    var peak: usize = 0;
    for (compiled.plan.instructions, 0..) |insn, instruction_idx| {
        for (insn.outputs) |reg| live[runtime_contract.registerToIndex(reg)] = true;

        var live_count: usize = 0;
        for (live) |is_live| {
            if (is_live) live_count += 1;
        }
        peak = @max(peak, live_count);

        const row = compiled.liveness.kill_after_instruction[instruction_idx];
        for (0..reg_count) |reg_idx| {
            const word = reg_idx / 64;
            const bit: u6 = @intCast(reg_idx % 64);
            if ((row[word] & (@as(u64, 1) << bit)) != 0) live[reg_idx] = false;
        }
    }
    return @intCast(peak);
}

fn buildDiagnostics(
    allocator: std.mem.Allocator,
    mode: CompileMode,
    compiled: *const runtime_contract.CompiledPlan,
) ![]runtime_contract.PlanDiagnostic {
    var diagnostics = std.ArrayListUnmanaged(runtime_contract.PlanDiagnostic){};
    errdefer {
        for (diagnostics.items) |diag| allocator.free(diag.message);
        diagnostics.deinit(allocator);
    }

    const summary = try std.fmt.allocPrint(
        allocator,
        "plan_compiled mode={s} instructions={d} registers={d} peak_registers={d} states={d} weights={d}",
        .{
            @tagName(mode),
            compiled.plan.instructions.len,
            compiled.plan.register_count,
            compiled.peak_registers,
            compiled.plan.state_descs.len,
            compiled.weight_bindings.len,
        },
    );
    try diagnostics.append(allocator, .{
        .level = .info,
        .message = summary,
    });

    if (compiled.plan.instructions.len == 0) {
        const empty_program = try std.fmt.allocPrint(
            allocator,
            "empty_plan mode={s}",
            .{@tagName(mode)},
        );
        try diagnostics.append(allocator, .{
            .level = .warn,
            .message = empty_program,
        });
    }

    if (compiled.plan.register_count > 0 and compiled.peak_registers == compiled.plan.register_count) {
        const register_pressure = try std.fmt.allocPrint(
            allocator,
            "register_pressure peak_equals_register_count count={d}",
            .{compiled.plan.register_count},
        );
        try diagnostics.append(allocator, .{
            .level = .warn,
            .message = register_pressure,
        });
    }

    return diagnostics.toOwnedSlice(allocator);
}

fn deinitDiagnostics(allocator: std.mem.Allocator, diagnostics: []const runtime_contract.PlanDiagnostic) void {
    for (diagnostics) |diag| allocator.free(diag.message);
    if (diagnostics.len > 0) allocator.free(diagnostics);
}

fn validateProgramBlockKindStateCompatibility(
    program: []const layer_ops.LayerOp,
    block_kind: op_types.BlockKind,
) !void {
    if (runtime_contract.firstLayerProgramStateMismatch(program, block_kind) != null) {
        return error.InvalidStateDescriptorBinding;
    }
}

pub fn compileLayerProgram(
    allocator: std.mem.Allocator,
    program: []const layer_ops.LayerOp,
    mode: CompileMode,
) !runtime_contract.CompiledPlan {
    var instructions = std.ArrayListUnmanaged(runtime_contract.Instruction){};
    var param_blocks = std.ArrayListUnmanaged(runtime_contract.ParamBlock){};
    var weight_bindings = std.ArrayListUnmanaged(runtime_contract.WeightBinding){};
    errdefer {
        for (instructions.items) |insn| {
            allocator.free(insn.inputs);
            allocator.free(insn.outputs);
            if (insn.weights.len > 0) allocator.free(insn.weights);
        }
        instructions.deinit(allocator);
        for (param_blocks.items) |param_block| deinitParamBlock(allocator, param_block);
        param_blocks.deinit(allocator);
        for (weight_bindings.items) |binding| allocator.free(binding.name);
        weight_bindings.deinit(allocator);
    }

    var max_register: u16 = 0;
    for (program) |op| {
        const opcode = opcode_map.opcodeForLayerOp(op);
        const param_block_id: u16 = @intCast(param_blocks.items.len);
        try param_blocks.append(allocator, try serializeLayerOpParam(allocator, opcode, op));
        const insn = try compileOneInstruction(allocator, op, mode, param_block_id, &weight_bindings);
        if (insn.weights.len != runtime_contract.expectedWeightRefCount(insn.opcode)) {
            return error.InvalidWeightRefCount;
        }
        max_register = @max(max_register, maxRegisterInInstruction(insn));
        try instructions.append(allocator, insn);
    }

    const instruction_slice = try instructions.toOwnedSlice(allocator);
    errdefer {
        for (instruction_slice) |insn| {
            allocator.free(insn.inputs);
            allocator.free(insn.outputs);
            if (insn.weights.len > 0) allocator.free(insn.weights);
        }
        allocator.free(instruction_slice);
    }

    const register_count: u16 = if (instruction_slice.len == 0) 0 else max_register + 1;
    const liveness = try buildLivenessMap(allocator, instruction_slice, register_count);
    errdefer {
        allocator.free(liveness.register_last_read);
        for (liveness.kill_after_instruction) |row| allocator.free(row);
        allocator.free(liveness.kill_after_instruction);
    }

    const param_block_slice = try param_blocks.toOwnedSlice(allocator);
    errdefer {
        for (param_block_slice) |param_block| deinitParamBlock(allocator, param_block);
        allocator.free(param_block_slice);
    }

    const weight_binding_slice = try weight_bindings.toOwnedSlice(allocator);
    errdefer {
        for (weight_binding_slice) |binding| allocator.free(binding.name);
        allocator.free(weight_binding_slice);
    }

    const state_descs = try buildStateDescriptors(allocator, instruction_slice);
    errdefer if (state_descs.len > 0) allocator.free(state_descs);

    var compiled = runtime_contract.CompiledPlan{
        .plan = .{
            .instructions = instruction_slice,
            .register_count = register_count,
            .state_descs = state_descs,
        },
        .param_blocks = param_block_slice,
        .weight_bindings = weight_binding_slice,
        .liveness = liveness,
        .peak_registers = register_count,
        .diagnostics = &.{},
    };
    compiled.peak_registers = try computePeakRegisters(allocator, &compiled);

    try runtime_contract.validateCompiledPlan(&compiled);
    compiled.diagnostics = try buildDiagnostics(allocator, mode, &compiled);
    return compiled;
}

pub fn compileProgramForArchitecture(
    allocator: std.mem.Allocator,
    architecture_id: []const u8,
    block_kind: op_types.BlockKind,
    mode: CompileMode,
) !runtime_contract.CompiledPlan {
    const entry = registry.detectByArchitectureId(architecture_id) orelse return error.UnknownArchitecture;
    const program = registry.blockProgramFor(entry, block_kind) orelse return error.MissingBlockProgram;
    try validateProgramBlockKindStateCompatibility(program, block_kind);
    return compileLayerProgram(allocator, program, mode);
}

pub fn deinitCompiledPlan(allocator: std.mem.Allocator, compiled: *runtime_contract.CompiledPlan) void {
    for (compiled.plan.instructions) |insn| {
        allocator.free(insn.inputs);
        allocator.free(insn.outputs);
        if (insn.weights.len > 0) allocator.free(insn.weights);
    }
    allocator.free(compiled.plan.instructions);
    for (compiled.param_blocks) |param_block| deinitParamBlock(allocator, param_block);
    allocator.free(compiled.param_blocks);

    for (compiled.weight_bindings) |binding| allocator.free(binding.name);
    if (compiled.weight_bindings.len > 0) allocator.free(compiled.weight_bindings);

    if (compiled.plan.state_descs.len > 0) allocator.free(compiled.plan.state_descs);

    allocator.free(compiled.liveness.register_last_read);
    for (compiled.liveness.kill_after_instruction) |row| allocator.free(row);
    allocator.free(compiled.liveness.kill_after_instruction);
    deinitDiagnostics(allocator, compiled.diagnostics);

    compiled.* = undefined;
}

fn expectedInputCount(op: layer_ops.LayerOp) usize {
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

fn expectedOutputCount(op: layer_ops.LayerOp) usize {
    return switch (op) {
        .split => |split_op| split_op.num_outputs,
        else => 1,
    };
}

fn expectedWeightCount(op: layer_ops.LayerOp) usize {
    return switch (op) {
        .linear, .add_param, .add_param_scalar, .mul_param => 1,
        else => 0,
    };
}

fn collectExpectedRegisters(
    op: layer_ops.LayerOp,
    input_buffer: *[64]runtime_contract.RegisterRef,
    output_buffer: *[64]runtime_contract.RegisterRef,
) struct { input_count: usize, output_count: usize } {
    return switch (op) {
        .kernel => |kernel_op| blk: {
            input_buffer[0] = registerFromBuffer(kernel_op.in);
            output_buffer[0] = registerFromBuffer(kernel_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .add => |add_op| blk: {
            input_buffer[0] = registerFromBuffer(.residual);
            input_buffer[1] = registerFromBuffer(add_op.branch);
            output_buffer[0] = registerFromBuffer(.residual);
            break :blk .{ .input_count = 2, .output_count = 1 };
        },
        .linear => |linear_op| blk: {
            input_buffer[0] = registerFromBuffer(linear_op.in);
            output_buffer[0] = registerFromBuffer(linear_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .matmul => |matmul_op| blk: {
            input_buffer[0] = registerFromBuffer(matmul_op.in_a);
            input_buffer[1] = registerFromBuffer(matmul_op.in_b);
            output_buffer[0] = registerFromBuffer(matmul_op.out);
            break :blk .{ .input_count = 2, .output_count = 1 };
        },
        .split => |split_op| blk: {
            input_buffer[0] = registerFromBuffer(split_op.in);
            const first: u16 = @intFromEnum(split_op.out_start);
            for (0..split_op.num_outputs) |idx| {
                output_buffer[idx] = runtime_contract.registerFromIndex(first + @as(u16, @intCast(idx)));
            }
            break :blk .{ .input_count = 1, .output_count = split_op.num_outputs };
        },
        .softmax => |softmax_op| blk: {
            input_buffer[0] = registerFromBuffer(softmax_op.in);
            output_buffer[0] = registerFromBuffer(softmax_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .silu => |silu_op| blk: {
            input_buffer[0] = registerFromBuffer(silu_op.in);
            output_buffer[0] = registerFromBuffer(silu_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .gelu => |gelu_op| blk: {
            input_buffer[0] = registerFromBuffer(gelu_op.in);
            output_buffer[0] = registerFromBuffer(gelu_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .mul => |mul_op| blk: {
            input_buffer[0] = registerFromBuffer(mul_op.in);
            input_buffer[1] = registerFromBuffer(mul_op.other);
            output_buffer[0] = registerFromBuffer(mul_op.out);
            break :blk .{ .input_count = 2, .output_count = 1 };
        },
        .add_tensor => |add_tensor_op| blk: {
            input_buffer[0] = registerFromBuffer(add_tensor_op.in_a);
            input_buffer[1] = registerFromBuffer(add_tensor_op.in_b);
            output_buffer[0] = registerFromBuffer(add_tensor_op.out);
            break :blk .{ .input_count = 2, .output_count = 1 };
        },
        .add_scalar => |add_scalar_op| blk: {
            input_buffer[0] = registerFromBuffer(add_scalar_op.in);
            output_buffer[0] = registerFromBuffer(add_scalar_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .mul_scalar => |mul_scalar_op| blk: {
            input_buffer[0] = registerFromBuffer(mul_scalar_op.in);
            output_buffer[0] = registerFromBuffer(mul_scalar_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .mean => |mean_op| blk: {
            input_buffer[0] = registerFromBuffer(mean_op.in);
            output_buffer[0] = registerFromBuffer(mean_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .pow => |pow_op| blk: {
            input_buffer[0] = registerFromBuffer(pow_op.in);
            output_buffer[0] = registerFromBuffer(pow_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .rsqrt => |rsqrt_op| blk: {
            input_buffer[0] = registerFromBuffer(rsqrt_op.in);
            output_buffer[0] = registerFromBuffer(rsqrt_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .add_param => |add_param_op| blk: {
            input_buffer[0] = registerFromBuffer(add_param_op.in);
            output_buffer[0] = registerFromBuffer(add_param_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .add_param_scalar => |add_param_scalar_op| blk: {
            output_buffer[0] = registerFromBuffer(add_param_scalar_op.out);
            break :blk .{ .input_count = 0, .output_count = 1 };
        },
        .mul_param => |mul_param_op| blk: {
            input_buffer[0] = registerFromBuffer(mul_param_op.in);
            output_buffer[0] = registerFromBuffer(mul_param_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .reshape => |reshape_op| blk: {
            input_buffer[0] = registerFromBuffer(reshape_op.in);
            output_buffer[0] = registerFromBuffer(reshape_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .transpose => |transpose_op| blk: {
            input_buffer[0] = registerFromBuffer(transpose_op.in);
            output_buffer[0] = registerFromBuffer(transpose_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .rope => |rope_op| blk: {
            input_buffer[0] = registerFromBuffer(rope_op.in);
            output_buffer[0] = registerFromBuffer(rope_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .triu => |triu_op| blk: {
            input_buffer[0] = registerFromBuffer(triu_op.in);
            output_buffer[0] = registerFromBuffer(triu_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .sdpa => |sdpa_op| blk: {
            input_buffer[0] = registerFromBuffer(sdpa_op.q);
            input_buffer[1] = registerFromBuffer(sdpa_op.k);
            input_buffer[2] = registerFromBuffer(sdpa_op.v);
            output_buffer[0] = registerFromBuffer(sdpa_op.out);
            break :blk .{ .input_count = 3, .output_count = 1 };
        },
        .patch_embed => |patch_op| blk: {
            input_buffer[0] = registerFromBuffer(patch_op.in);
            output_buffer[0] = registerFromBuffer(patch_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .spatial_merge => |spatial_op| blk: {
            input_buffer[0] = registerFromBuffer(spatial_op.in);
            output_buffer[0] = registerFromBuffer(spatial_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .deepstack_extract => |deepstack_op| blk: {
            input_buffer[0] = registerFromBuffer(deepstack_op.in);
            output_buffer[0] = registerFromBuffer(deepstack_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .scatter => |scatter_op| blk: {
            input_buffer[0] = registerFromBuffer(scatter_op.text_in);
            input_buffer[1] = registerFromBuffer(scatter_op.vision_in);
            output_buffer[0] = registerFromBuffer(scatter_op.out);
            break :blk .{ .input_count = 2, .output_count = 1 };
        },
    };
}

fn expectProgramParity(source: []const layer_ops.LayerOp, compiled: *const runtime_contract.CompiledPlan) !void {
    try std.testing.expectEqual(source.len, compiled.plan.instructions.len);
    for (source, compiled.plan.instructions) |source_op, compiled_insn| {
        try std.testing.expectEqual(opcode_map.opcodeForLayerOp(source_op), compiled_insn.opcode);
        try std.testing.expectEqual(expectedInputCount(source_op), compiled_insn.inputs.len);
        try std.testing.expectEqual(expectedOutputCount(source_op), compiled_insn.outputs.len);
        try std.testing.expectEqual(expectedWeightCount(source_op), compiled_insn.weights.len);

        var expected_inputs: [64]runtime_contract.RegisterRef = undefined;
        var expected_outputs: [64]runtime_contract.RegisterRef = undefined;
        const expected = collectExpectedRegisters(source_op, &expected_inputs, &expected_outputs);
        try std.testing.expectEqualSlices(
            runtime_contract.RegisterRef,
            expected_inputs[0..expected.input_count],
            compiled_insn.inputs,
        );
        try std.testing.expectEqualSlices(
            runtime_contract.RegisterRef,
            expected_outputs[0..expected.output_count],
            compiled_insn.outputs,
        );
    }
}

fn hasStateDescriptor(compiled: *const runtime_contract.CompiledPlan, id: u8) bool {
    for (compiled.plan.state_descs) |state_desc| {
        if (state_desc.id == id) return true;
    }
    return false;
}

test "compileLayerProgram preserves structural parity for llama3" {
    var compiled = try compileLayerProgram(std.testing.allocator, llama3.attention_mlp_program, .decode);
    defer deinitCompiledPlan(std.testing.allocator, &compiled);

    try expectProgramParity(llama3.attention_mlp_program, &compiled);
}

test "compileLayerProgram preserves structural parity for granite_hybrid mamba" {
    var compiled = try compileLayerProgram(std.testing.allocator, granite_hybrid.mamba_program, .prefill);
    defer deinitCompiledPlan(std.testing.allocator, &compiled);

    try expectProgramParity(granite_hybrid.mamba_program, &compiled);
}

test "compileLayerProgram preserves structural parity for qwen3_moe" {
    var compiled = try compileLayerProgram(std.testing.allocator, qwen3_moe.attention_mlp_program, .decode);
    defer deinitCompiledPlan(std.testing.allocator, &compiled);

    try expectProgramParity(qwen3_moe.attention_mlp_program, &compiled);
}

test "compileLayerProgram emits KV state descriptor and attention state references" {
    var compiled = try compileLayerProgram(std.testing.allocator, llama3.attention_mlp_program, .decode);
    defer deinitCompiledPlan(std.testing.allocator, &compiled);

    try std.testing.expect(hasStateDescriptor(&compiled, kv_state_id));

    var saw_attention = false;
    for (compiled.plan.instructions) |insn| {
        if (insn.opcode == .multihead_attention) {
            saw_attention = true;
            try std.testing.expectEqual(@as(?u8, kv_state_id), insn.state_block_id);
        }
    }
    try std.testing.expect(saw_attention);
}

test "compileLayerProgram omits attention state in vision mode" {
    const program = [_]layer_ops.LayerOp{
        .{ .kernel = .{
            .id = 0,
            .in = .residual,
            .out = .norm_out,
            .debug_type = .multihead_attention,
        } },
    };

    var compiled = try compileLayerProgram(std.testing.allocator, &program, .vision_encode);
    defer deinitCompiledPlan(std.testing.allocator, &compiled);

    try std.testing.expectEqual(@as(usize, 0), compiled.plan.state_descs.len);
    try std.testing.expectEqual(@as(?u8, null), compiled.plan.instructions[0].state_block_id);
}

test "compileLayerProgram emits mamba state descriptor and mixer state references" {
    var compiled = try compileLayerProgram(std.testing.allocator, granite_hybrid.mamba_program, .prefill);
    defer deinitCompiledPlan(std.testing.allocator, &compiled);

    try std.testing.expect(hasStateDescriptor(&compiled, mamba_state_id));

    var saw_mamba = false;
    for (compiled.plan.instructions) |insn| {
        if (insn.opcode == .mamba_mixer) {
            saw_mamba = true;
            try std.testing.expectEqual(@as(?u8, mamba_state_id), insn.state_block_id);
        }
    }
    try std.testing.expect(saw_mamba);
}

test "compileLayerProgram emits shortconv state descriptor when shortconv op is present" {
    const shortconv_program = [_]layer_ops.LayerOp{
        .{ .kernel = .{
            .id = 0,
            .in = .residual,
            .out = .branch_out,
            .debug_type = .shortconv,
        } },
    };
    var compiled = try compileLayerProgram(std.testing.allocator, &shortconv_program, .decode);
    defer deinitCompiledPlan(std.testing.allocator, &compiled);

    try std.testing.expect(hasStateDescriptor(&compiled, shortconv_state_id));
    try std.testing.expectEqual(@as(?u8, shortconv_state_id), compiled.plan.instructions[0].state_block_id);
}

test "compileLayerProgram emits deterministic weight bindings for parameterized primitive ops" {
    const program = [_]layer_ops.LayerOp{
        .{ .linear = .{
            .in = .residual,
            .out = .tmp3,
            .weight_name = "w_linear",
        } },
        .{ .add_param = .{
            .in = .tmp3,
            .out = .tmp4,
            .param_name = "p_add",
        } },
        .{ .mul_param = .{
            .in = .tmp4,
            .out = .residual,
            .param_name = "p_add",
        } },
    };
    var compiled = try compileLayerProgram(std.testing.allocator, &program, .decode);
    defer deinitCompiledPlan(std.testing.allocator, &compiled);

    try std.testing.expectEqual(@as(usize, 2), compiled.weight_bindings.len);
    try std.testing.expectEqual(@as(u32, 0), compiled.weight_bindings[0].index);
    try std.testing.expectEqualStrings("w_linear", compiled.weight_bindings[0].name);
    try std.testing.expectEqual(@as(u32, 1), compiled.weight_bindings[1].index);
    try std.testing.expectEqualStrings("p_add", compiled.weight_bindings[1].name);

    try std.testing.expectEqual(@as(u32, 0), compiled.plan.instructions[0].weights[0].index);
    try std.testing.expectEqual(@as(u32, 1), compiled.plan.instructions[1].weights[0].index);
    try std.testing.expectEqual(@as(u32, 1), compiled.plan.instructions[2].weights[0].index);
}

test "compileLayerProgram strips runtime param-block weight names for bound primitives" {
    const program = [_]layer_ops.LayerOp{
        .{ .linear = .{
            .in = .residual,
            .out = .tmp3,
            .weight_name = "w_linear",
        } },
        .{ .add_param = .{
            .in = .tmp3,
            .out = .tmp4,
            .param_name = "p_add",
        } },
        .{ .mul_param = .{
            .in = .tmp4,
            .out = .residual,
            .param_name = "p_add",
        } },
    };
    var compiled = try compileLayerProgram(std.testing.allocator, &program, .decode);
    defer deinitCompiledPlan(std.testing.allocator, &compiled);

    inline for ([_]usize{ 0, 1, 2 }) |insn_idx| {
        const insn = compiled.plan.instructions[insn_idx];
        const decoded = try runtime_contract.decodeInstructionLayerOp(&compiled, &insn, insn_idx);
        switch (decoded) {
            .linear => |linear_op| try std.testing.expectEqual(@as(usize, 0), linear_op.weight_name.len),
            .add_param => |add_param_op| try std.testing.expectEqual(@as(usize, 0), add_param_op.param_name.len),
            .mul_param => |mul_param_op| try std.testing.expectEqual(@as(usize, 0), mul_param_op.param_name.len),
            else => return error.TestUnexpectedResult,
        }
    }
}

test "compileLayerProgram emits param blocks compliant with runtime ABI contract" {
    var compiled = try compileLayerProgram(std.testing.allocator, llama3.attention_mlp_program, .decode);
    defer deinitCompiledPlan(std.testing.allocator, &compiled);

    for (compiled.param_blocks) |param_block| {
        try std.testing.expectEqual(
            runtime_contract.param_block_abi_version_v1,
            param_block.version,
        );
        try std.testing.expect(param_block.data.len <= runtime_contract.max_param_block_data_bytes_v1);
        try runtime_contract.validateParamBlockAbi(&param_block);
    }
}

test "compileLayerProgram param blocks decode back to executable layer ops" {
    const program = [_]layer_ops.LayerOp{
        .{ .kernel = .{
            .id = 7,
            .in = .residual,
            .out = .norm_out,
            .debug_type = .norm,
        } },
        .{ .add = .{
            .branch = .branch_out,
            .scale = .{ .literal = 0.25 },
        } },
        .{ .mul_scalar = .{
            .in = .residual,
            .out = .tmp3,
            .scalar = 2.0,
        } },
        .{ .add_scalar = .{
            .in = .tmp3,
            .out = .residual,
            .scalar = -1.0,
        } },
    };
    var compiled = try compileLayerProgram(std.testing.allocator, &program, .decode);
    defer deinitCompiledPlan(std.testing.allocator, &compiled);

    for (program, compiled.plan.instructions, 0..) |source_op, insn, op_index| {
        const decoded = try runtime_contract.decodeInstructionLayerOp(&compiled, &insn, op_index);
        switch (source_op) {
            .kernel => |kernel| switch (decoded) {
                .kernel => |decoded_kernel| {
                    try std.testing.expectEqual(kernel.id, decoded_kernel.id);
                    try std.testing.expectEqual(kernel.in, decoded_kernel.in);
                    try std.testing.expectEqual(kernel.out, decoded_kernel.out);
                    try std.testing.expectEqual(kernel.debug_type, decoded_kernel.debug_type);
                },
                else => return error.TestUnexpectedResult,
            },
            .add => |add_op| switch (decoded) {
                .add => |decoded_add| {
                    try std.testing.expectEqual(add_op.branch, decoded_add.branch);
                    try std.testing.expect(std.meta.eql(add_op.scale, decoded_add.scale));
                },
                else => return error.TestUnexpectedResult,
            },
            .mul_scalar => |mul_scalar_op| switch (decoded) {
                .mul_scalar => |decoded_mul| {
                    try std.testing.expectEqual(mul_scalar_op.in, decoded_mul.in);
                    try std.testing.expectEqual(mul_scalar_op.out, decoded_mul.out);
                    try std.testing.expectEqual(mul_scalar_op.scalar, decoded_mul.scalar);
                },
                else => return error.TestUnexpectedResult,
            },
            .add_scalar => |add_scalar_op| switch (decoded) {
                .add_scalar => |decoded_add_scalar| {
                    try std.testing.expectEqual(add_scalar_op.in, decoded_add_scalar.in);
                    try std.testing.expectEqual(add_scalar_op.out, decoded_add_scalar.out);
                    try std.testing.expectEqual(add_scalar_op.scalar, decoded_add_scalar.scalar);
                },
                else => return error.TestUnexpectedResult,
            },
            else => return error.TestUnexpectedResult,
        }
    }
}

test "compileProgramForArchitecture resolves registry programs" {
    var compiled = try compileProgramForArchitecture(std.testing.allocator, "granite_hybrid", .mamba, .decode);
    defer deinitCompiledPlan(std.testing.allocator, &compiled);

    try std.testing.expect(compiled.plan.instructions.len > 0);
}

test "compileLayerProgram emits summary diagnostics" {
    var compiled = try compileLayerProgram(std.testing.allocator, llama3.attention_mlp_program, .decode);
    defer deinitCompiledPlan(std.testing.allocator, &compiled);

    try std.testing.expect(compiled.diagnostics.len >= 1);
    try std.testing.expectEqual(runtime_contract.PlanDiagnosticLevel.info, compiled.diagnostics[0].level);
    try std.testing.expect(std.mem.startsWith(u8, compiled.diagnostics[0].message, "plan_compiled mode=decode"));
}

test "compileLayerProgram is deterministic across repeated compiles of same program" {
    var first = try compileLayerProgram(std.testing.allocator, qwen3_moe.attention_mlp_program, .decode);
    defer deinitCompiledPlan(std.testing.allocator, &first);
    var second = try compileLayerProgram(std.testing.allocator, qwen3_moe.attention_mlp_program, .decode);
    defer deinitCompiledPlan(std.testing.allocator, &second);

    try std.testing.expectEqual(first.plan.instructions.len, second.plan.instructions.len);
    try std.testing.expectEqual(first.param_blocks.len, second.param_blocks.len);
    try std.testing.expectEqual(first.weight_bindings.len, second.weight_bindings.len);
    try std.testing.expectEqual(first.plan.register_count, second.plan.register_count);
    try std.testing.expectEqual(first.peak_registers, second.peak_registers);

    for (first.weight_bindings, second.weight_bindings) |lhs, rhs| {
        try std.testing.expectEqual(lhs.index, rhs.index);
        try std.testing.expectEqualStrings(lhs.name, rhs.name);
    }
    for (first.param_blocks, second.param_blocks) |lhs, rhs| {
        try std.testing.expectEqual(lhs.version, rhs.version);
        try std.testing.expectEqual(lhs.opcode, rhs.opcode);
        try std.testing.expectEqualSlices(u8, lhs.data, rhs.data);
    }
}

test "compileLayerProgram emits empty plan warning diagnostics" {
    var compiled = try compileLayerProgram(std.testing.allocator, &.{}, .decode);
    defer deinitCompiledPlan(std.testing.allocator, &compiled);

    try std.testing.expect(compiled.diagnostics.len >= 2);
    try std.testing.expectEqual(runtime_contract.PlanDiagnosticLevel.warn, compiled.diagnostics[1].level);
    try std.testing.expect(std.mem.startsWith(u8, compiled.diagnostics[1].message, "empty_plan mode=decode"));
}

test "validateProgramBlockKindStateCompatibility rejects mismatched stateful opcode" {
    const program = [_]layer_ops.LayerOp{
        .{ .kernel = .{
            .id = 0,
            .in = .residual,
            .out = .branch_out,
            .debug_type = .shortconv,
        } },
    };
    try std.testing.expectError(
        error.InvalidStateDescriptorBinding,
        validateProgramBlockKindStateCompatibility(&program, .attention_mlp),
    );
}
