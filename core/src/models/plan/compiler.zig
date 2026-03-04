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

const kv_state_id: u8 = runtime_contract.kv_cache_state_id;
const shortconv_state_id: u8 = runtime_contract.shortconv_state_id;
const mamba_state_id: u8 = runtime_contract.mamba_state_id;

/// Allocation-order register assignment.
///
/// Only residual (0) is architecturally fixed. All intermediates (norm_out,
/// branch_out, tmp*) are dynamically assigned dense register indices in
/// first-use order.
const RegisterMap = struct {
    buffer_to_register: [64]u16,
    next_register: u16,

    const max_registers = runtime_contract.max_register_count;
    const sentinel: u16 = std.math.maxInt(u16);

    fn init() RegisterMap {
        var map = RegisterMap{
            .buffer_to_register = [_]u16{sentinel} ** 64,
            .next_register = 1,
        };
        // Only residual is architecturally fixed (block input/output boundary).
        // All intermediates (norm_out, branch_out, tmp*) are dynamically assigned
        // by first-use order.
        map.buffer_to_register[0] = 0;
        return map;
    }

    fn registerFor(self: *RegisterMap, buffer: layer_ops.BufferId) runtime_contract.RegisterRef {
        const buf_idx: usize = @intFromEnum(buffer);
        if (self.buffer_to_register[buf_idx] != sentinel) {
            return runtime_contract.registerFromIndex(self.buffer_to_register[buf_idx]);
        }
        const reg_idx = self.next_register;
        self.next_register += 1;
        self.buffer_to_register[buf_idx] = reg_idx;
        return runtime_contract.registerFromIndex(reg_idx);
    }

    fn allocRegisters(
        self: *RegisterMap,
        allocator: std.mem.Allocator,
        buffers: []const layer_ops.BufferId,
    ) ![]runtime_contract.RegisterRef {
        var regs = try allocator.alloc(runtime_contract.RegisterRef, buffers.len);
        for (buffers, 0..) |buffer, idx| regs[idx] = self.registerFor(buffer);
        return regs;
    }

    fn allocSequential(
        self: *RegisterMap,
        allocator: std.mem.Allocator,
        first: layer_ops.BufferId,
        count: usize,
    ) ![]runtime_contract.RegisterRef {
        var regs = try allocator.alloc(runtime_contract.RegisterRef, count);
        const first_index: u16 = @intFromEnum(first);
        for (0..count) |idx| {
            const buf_val = first_index + @as(u16, @intCast(idx));
            if (buf_val < 64) {
                regs[idx] = self.registerFor(@enumFromInt(buf_val));
            } else {
                // Buffer index exceeds BufferId range — assign register directly.
                const reg_idx = self.next_register;
                self.next_register += 1;
                regs[idx] = runtime_contract.registerFromIndex(reg_idx);
            }
        }
        return regs;
    }
};

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

fn kernelWeightSlots(opcode: runtime_contract.Opcode) []const []const u8 {
    return runtime_contract.expectedKernelWeightSlots(opcode);
}

fn compileOneInstruction(
    allocator: std.mem.Allocator,
    op: layer_ops.LayerOp,
    param_block_id: u16,
    weight_bindings: *std.ArrayListUnmanaged(runtime_contract.WeightBinding),
    reg_map: *RegisterMap,
) !runtime_contract.Instruction {
    const opcode = opcode_map.opcodeForLayerOp(op);
    return switch (op) {
        .kernel => |kernel_op| blk: {
            const slots = kernelWeightSlots(opcode);
            const refs = try allocator.alloc(runtime_contract.WeightRef, slots.len);
            errdefer allocator.free(refs);
            for (slots, 0..) |slot_name, idx| {
                const binding_name = try std.fmt.allocPrint(
                    allocator,
                    "{s}{d}::{s}::{d}",
                    .{
                        runtime_contract.kernel_weight_binding_prefix,
                        kernel_op.id,
                        slot_name,
                        param_block_id,
                    },
                );
                defer allocator.free(binding_name);
                refs[idx] = .{
                    .index = try ensureWeightBindingIndex(allocator, weight_bindings, binding_name),
                };
            }
            break :blk .{
                .opcode = opcode,
                .inputs = try reg_map.allocRegisters(allocator, &.{kernel_op.in}),
                .outputs = try reg_map.allocRegisters(allocator, &.{kernel_op.out}),
                .weights = refs,
                .param_block_id = param_block_id,
                .state_block_id = kernel_op.state_block_id,
            };
        },
        .add => |add_op| .{
            .opcode = opcode,
            .inputs = try reg_map.allocRegisters(allocator, &.{ .residual, add_op.branch }),
            .outputs = try reg_map.allocRegisters(allocator, &.{.residual}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = null,
        },
        .linear => |linear_op| .{
            .opcode = opcode,
            .inputs = try reg_map.allocRegisters(allocator, &.{linear_op.in}),
            .outputs = try reg_map.allocRegisters(allocator, &.{linear_op.out}),
            .weights = try allocWeightRefs(allocator, &.{.{
                .index = try ensureWeightBindingIndex(allocator, weight_bindings, linear_op.weight_name),
            }}),
            .param_block_id = param_block_id,
            .state_block_id = null,
        },
        .matmul => |matmul_op| .{
            .opcode = opcode,
            .inputs = try reg_map.allocRegisters(allocator, &.{ matmul_op.in_a, matmul_op.in_b }),
            .outputs = try reg_map.allocRegisters(allocator, &.{matmul_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = null,
        },
        .split => |split_op| .{
            .opcode = opcode,
            .inputs = try reg_map.allocRegisters(allocator, &.{split_op.in}),
            .outputs = try reg_map.allocSequential(allocator, split_op.out_start, split_op.num_outputs),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = null,
        },
        .softmax => |softmax_op| .{
            .opcode = opcode,
            .inputs = try reg_map.allocRegisters(allocator, &.{softmax_op.in}),
            .outputs = try reg_map.allocRegisters(allocator, &.{softmax_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = null,
        },
        .silu => |silu_op| .{
            .opcode = opcode,
            .inputs = try reg_map.allocRegisters(allocator, &.{silu_op.in}),
            .outputs = try reg_map.allocRegisters(allocator, &.{silu_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = null,
        },
        .gelu => |gelu_op| .{
            .opcode = opcode,
            .inputs = try reg_map.allocRegisters(allocator, &.{gelu_op.in}),
            .outputs = try reg_map.allocRegisters(allocator, &.{gelu_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = null,
        },
        .mul => |mul_op| .{
            .opcode = opcode,
            .inputs = try reg_map.allocRegisters(allocator, &.{ mul_op.in, mul_op.other }),
            .outputs = try reg_map.allocRegisters(allocator, &.{mul_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = null,
        },
        .add_tensor => |add_tensor_op| .{
            .opcode = opcode,
            .inputs = try reg_map.allocRegisters(allocator, &.{ add_tensor_op.in_a, add_tensor_op.in_b }),
            .outputs = try reg_map.allocRegisters(allocator, &.{add_tensor_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = null,
        },
        .add_scalar => |add_scalar_op| .{
            .opcode = opcode,
            .inputs = try reg_map.allocRegisters(allocator, &.{add_scalar_op.in}),
            .outputs = try reg_map.allocRegisters(allocator, &.{add_scalar_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = null,
        },
        .mul_scalar => |mul_scalar_op| .{
            .opcode = opcode,
            .inputs = try reg_map.allocRegisters(allocator, &.{mul_scalar_op.in}),
            .outputs = try reg_map.allocRegisters(allocator, &.{mul_scalar_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = null,
        },
        .mean => |mean_op| .{
            .opcode = opcode,
            .inputs = try reg_map.allocRegisters(allocator, &.{mean_op.in}),
            .outputs = try reg_map.allocRegisters(allocator, &.{mean_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = null,
        },
        .pow => |pow_op| .{
            .opcode = opcode,
            .inputs = try reg_map.allocRegisters(allocator, &.{pow_op.in}),
            .outputs = try reg_map.allocRegisters(allocator, &.{pow_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = null,
        },
        .rsqrt => |rsqrt_op| .{
            .opcode = opcode,
            .inputs = try reg_map.allocRegisters(allocator, &.{rsqrt_op.in}),
            .outputs = try reg_map.allocRegisters(allocator, &.{rsqrt_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = null,
        },
        .add_param => |add_param_op| .{
            .opcode = opcode,
            .inputs = try reg_map.allocRegisters(allocator, &.{add_param_op.in}),
            .outputs = try reg_map.allocRegisters(allocator, &.{add_param_op.out}),
            .weights = try allocWeightRefs(allocator, &.{.{
                .index = try ensureWeightBindingIndex(allocator, weight_bindings, add_param_op.param_name),
            }}),
            .param_block_id = param_block_id,
            .state_block_id = null,
        },
        .add_param_scalar => |add_param_scalar_op| .{
            .opcode = opcode,
            .inputs = try reg_map.allocRegisters(allocator, &.{}),
            .outputs = try reg_map.allocRegisters(allocator, &.{add_param_scalar_op.out}),
            .weights = try allocWeightRefs(allocator, &.{.{
                .index = try ensureWeightBindingIndex(allocator, weight_bindings, add_param_scalar_op.param_name),
            }}),
            .param_block_id = param_block_id,
            .state_block_id = null,
        },
        .mul_param => |mul_param_op| .{
            .opcode = opcode,
            .inputs = try reg_map.allocRegisters(allocator, &.{mul_param_op.in}),
            .outputs = try reg_map.allocRegisters(allocator, &.{mul_param_op.out}),
            .weights = try allocWeightRefs(allocator, &.{.{
                .index = try ensureWeightBindingIndex(allocator, weight_bindings, mul_param_op.param_name),
            }}),
            .param_block_id = param_block_id,
            .state_block_id = null,
        },
        .reshape => |reshape_op| .{
            .opcode = opcode,
            .inputs = try reg_map.allocRegisters(allocator, &.{reshape_op.in}),
            .outputs = try reg_map.allocRegisters(allocator, &.{reshape_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = null,
        },
        .transpose => |transpose_op| .{
            .opcode = opcode,
            .inputs = try reg_map.allocRegisters(allocator, &.{transpose_op.in}),
            .outputs = try reg_map.allocRegisters(allocator, &.{transpose_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = null,
        },
        .rope => |rope_op| .{
            .opcode = opcode,
            .inputs = try reg_map.allocRegisters(allocator, &.{rope_op.in}),
            .outputs = try reg_map.allocRegisters(allocator, &.{rope_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = null,
        },
        .triu => |triu_op| .{
            .opcode = opcode,
            .inputs = try reg_map.allocRegisters(allocator, &.{triu_op.in}),
            .outputs = try reg_map.allocRegisters(allocator, &.{triu_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = null,
        },
        .sdpa => |sdpa_op| .{
            .opcode = opcode,
            .inputs = try reg_map.allocRegisters(allocator, &.{ sdpa_op.q, sdpa_op.k, sdpa_op.v }),
            .outputs = try reg_map.allocRegisters(allocator, &.{sdpa_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = null,
        },
        .patch_embed => |patch_op| .{
            .opcode = opcode,
            .inputs = try reg_map.allocRegisters(allocator, &.{patch_op.in}),
            .outputs = try reg_map.allocRegisters(allocator, &.{patch_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = null,
        },
        .spatial_merge => |spatial_op| .{
            .opcode = opcode,
            .inputs = try reg_map.allocRegisters(allocator, &.{spatial_op.in}),
            .outputs = try reg_map.allocRegisters(allocator, &.{spatial_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = null,
        },
        .deepstack_extract => |deepstack_op| .{
            .opcode = opcode,
            .inputs = try reg_map.allocRegisters(allocator, &.{deepstack_op.in}),
            .outputs = try reg_map.allocRegisters(allocator, &.{deepstack_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = null,
        },
        .scatter => |scatter_op| .{
            .opcode = opcode,
            .inputs = try reg_map.allocRegisters(allocator, &.{ scatter_op.text_in, scatter_op.vision_in }),
            .outputs = try reg_map.allocRegisters(allocator, &.{scatter_op.out}),
            .weights = &.{},
            .param_block_id = param_block_id,
            .state_block_id = null,
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

const LivenessResult = struct {
    map: runtime_contract.LivenessMap,
    never_read_registers: []const u16,
};

fn buildLivenessMap(
    allocator: std.mem.Allocator,
    instructions: []const runtime_contract.Instruction,
    register_count: u16,
) !LivenessResult {
    const register_count_usize = @as(usize, register_count);
    const sentinel = std.math.maxInt(u32);
    var register_last_read = try allocator.alloc(u32, register_count_usize);
    for (register_last_read) |*entry| entry.* = sentinel;

    var produced = try allocator.alloc(bool, register_count_usize);
    defer allocator.free(produced);
    @memset(produced, false);

    var first_write = try allocator.alloc(u32, register_count_usize);
    defer allocator.free(first_write);
    @memset(first_write, sentinel);

    for (instructions, 0..) |insn, instruction_idx| {
        const idx_u32: u32 = @intCast(instruction_idx);
        for (insn.inputs) |reg| {
            const reg_idx = runtime_contract.registerToIndex(reg);
            register_last_read[reg_idx] = idx_u32;
        }
        for (insn.outputs) |reg| {
            const reg_idx = runtime_contract.registerToIndex(reg);
            produced[reg_idx] = true;
            if (first_write[reg_idx] == sentinel) {
                first_write[reg_idx] = idx_u32;
            }
        }
    }

    var never_read = std.ArrayListUnmanaged(u16){};
    errdefer {
        if (never_read.items.len > 0) allocator.free(never_read.items);
    }

    if (instructions.len > 0) {
        const terminal_idx: u32 = @intCast(instructions.len - 1);
        // The plan's final output register is consumed by the runtime outside
        // the plan, so it is expected to have no internal readers.
        const final_output_idx: u16 = runtime_contract.registerToIndex(
            runtime_contract.planFinalOutputRegister(&.{
                .instructions = instructions,
                .register_count = register_count,
                .state_descs = &.{},
            }),
        );
        for (register_last_read, 0..) |last_read, idx| {
            if (produced[idx] and last_read == sentinel) {
                if (idx == final_output_idx) {
                    // Final output is consumed by runtime after plan execution;
                    // keep alive until terminal instruction.
                    register_last_read[idx] = terminal_idx;
                } else {
                    // Dead output: kill at producer for immediate physical buffer reclaim.
                    register_last_read[idx] = first_write[idx];
                    try never_read.append(allocator, @intCast(idx));
                }
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
        .map = .{
            .register_last_read = register_last_read,
            .kill_after_instruction = kill_after_instruction,
        },
        .never_read_registers = try never_read.toOwnedSlice(allocator),
    };
}

fn buildStateDescriptors(
    allocator: std.mem.Allocator,
    instructions: []const runtime_contract.Instruction,
    descriptor_entry: ?registry.Entry,
) ![]runtime_contract.StateDescriptor {
    var seen: [256]bool = [_]bool{false} ** 256;
    var count: usize = 0;
    for (instructions) |insn| {
        const state_id = insn.state_block_id orelse continue;
        if (!seen[state_id]) {
            seen[state_id] = true;
            count += 1;
        }
    }
    if (count == 0) return &.{};

    const descriptors = try allocator.alloc(runtime_contract.StateDescriptor, count);
    var idx: usize = 0;
    for (instructions) |insn| {
        const state_id = insn.state_block_id orelse continue;
        if (!seen[state_id]) continue;
        seen[state_id] = false;
        descriptors[idx] = if (descriptor_entry) |entry|
            registry.stateDescriptorForId(entry, state_id)
        else
            registry.defaultStateDescriptorForId(state_id);
        idx += 1;
    }
    return descriptors;
}

fn buildRegisterBufferSpecs(
    allocator: std.mem.Allocator,
    program: []const layer_ops.LayerOp,
    instructions: []const runtime_contract.Instruction,
    register_count: u16,
    size_floor: usize,
) ![]runtime_contract.PhysicalBufferSpec {
    if (register_count == 0) return &.{};
    if (instructions.len != program.len) return error.InvalidInstructionCount;

    const reg_count: usize = register_count;
    const specs = try allocator.alloc(runtime_contract.PhysicalBufferSpec, reg_count);
    const floor = @max(size_floor, 1);
    for (specs) |*spec| {
        spec.* = .{
            .size = floor,
            .@"align" = 64,
        };
    }

    for (program, 0..) |op, op_index| {
        const insn = instructions[op_index];
        var input_width: usize = 1;
        for (insn.inputs) |input_reg| {
            const input_idx = runtime_contract.registerToIndex(input_reg);
            if (input_idx < specs.len) input_width = @max(input_width, specs[input_idx].size);
        }

        // Default propagation keeps conservative non-zero sizes on all outputs.
        for (insn.outputs) |output_reg| {
            const output_idx = runtime_contract.registerToIndex(output_reg);
            if (output_idx >= specs.len) continue;
            specs[output_idx].size = @max(specs[output_idx].size, input_width);
        }

        // Override for ops that explicitly carry output-width metadata.
        switch (op) {
            .split => |split_op| {
                if (split_op.split_sizes.len == split_op.num_outputs and split_op.split_sizes.len != 0) {
                    for (insn.outputs, 0..) |output_reg, out_idx| {
                        const output_idx = runtime_contract.registerToIndex(output_reg);
                        if (output_idx >= specs.len) continue;
                        specs[output_idx].size = @max(specs[output_idx].size, split_op.split_sizes[out_idx]);
                    }
                }
            },
            .reshape => |reshape_op| {
                if (reshape_op.shape.len != 0) {
                    const tail_dim = reshape_op.shape[reshape_op.shape.len - 1];
                    if (tail_dim > 0) {
                        const width: usize = @intCast(tail_dim);
                        for (insn.outputs) |output_reg| {
                            const output_idx = runtime_contract.registerToIndex(output_reg);
                            if (output_idx >= specs.len) continue;
                            specs[output_idx].size = @max(specs[output_idx].size, width);
                        }
                    }
                }
            },
            else => {},
        }
    }

    return specs;
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
    never_read_registers: []const u16,
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

    for (never_read_registers) |reg_idx| {
        const msg = try std.fmt.allocPrint(
            allocator,
            "never_read register={d}",
            .{reg_idx},
        );
        try diagnostics.append(allocator, .{
            .level = .warn,
            .message = msg,
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
    _ = program;
    _ = block_kind;
}

/// Options for plan compilation.
pub const CompileOptions = struct {
    /// Minimum buffer size for all register specs. The compiler guarantees
    /// that every `PhysicalBufferSpec.size` in the compiled plan is at
    /// least this value. Backends should consume plan specs exactly; any
    /// model-dimension floor must be provided here, not post-hoc.
    size_floor: usize = 1,
    /// Optional architecture registry entry that owns state descriptor metadata.
    /// When provided, compiler emits state descriptors from architecture metadata
    /// rather than runtime-contract built-in defaults.
    state_descriptor_entry: ?registry.Entry = null,
};

pub fn compileLayerProgram(
    allocator: std.mem.Allocator,
    program: []const layer_ops.LayerOp,
    mode: CompileMode,
    options: CompileOptions,
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

    var reg_map = RegisterMap.init();
    for (program) |op| {
        const opcode = opcode_map.opcodeForLayerOp(op);
        const param_block_id: u16 = @intCast(param_blocks.items.len);
        try param_blocks.append(allocator, try serializeLayerOpParam(allocator, opcode, op));
        const insn = try compileOneInstruction(allocator, op, param_block_id, &weight_bindings, &reg_map);
        if (insn.weights.len != runtime_contract.expectedWeightRefCount(insn.opcode)) {
            return error.InvalidWeightRefCount;
        }
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

    const register_count: u16 = if (instruction_slice.len == 0) 0 else reg_map.next_register;
    const liveness_result = try buildLivenessMap(allocator, instruction_slice, register_count);
    const liveness = liveness_result.map;
    defer if (liveness_result.never_read_registers.len > 0)
        allocator.free(liveness_result.never_read_registers);
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

    const state_descs = try buildStateDescriptors(allocator, instruction_slice, options.state_descriptor_entry);
    errdefer if (state_descs.len > 0) allocator.free(state_descs);
    const register_buffer_specs = try buildRegisterBufferSpecs(
        allocator,
        program,
        instruction_slice,
        register_count,
        options.size_floor,
    );
    errdefer if (register_buffer_specs.len > 0) allocator.free(register_buffer_specs);

    var compiled = runtime_contract.CompiledPlan{
        .plan = .{
            .instructions = instruction_slice,
            .register_count = register_count,
            .state_descs = state_descs,
        },
        .param_blocks = param_block_slice,
        .weight_bindings = weight_binding_slice,
        .register_buffer_specs = register_buffer_specs,
        .liveness = liveness,
        .peak_registers = register_count,
        .diagnostics = &.{},
    };
    compiled.peak_registers = try computePeakRegisters(allocator, &compiled);

    try runtime_contract.validateCompiledPlan(&compiled);
    compiled.diagnostics = try buildDiagnostics(allocator, mode, &compiled, liveness_result.never_read_registers);
    return compiled;
}

pub fn compileProgramForArchitecture(
    allocator: std.mem.Allocator,
    architecture_id: []const u8,
    block_kind: op_types.BlockKind,
    mode: CompileMode,
    options: CompileOptions,
) !runtime_contract.CompiledPlan {
    const entry = registry.detectByArchitectureId(architecture_id) orelse return error.UnknownArchitecture;
    const program = registry.blockProgramFor(entry, block_kind) orelse return error.MissingBlockProgram;
    try validateProgramBlockKindStateCompatibility(program, block_kind);
    var resolved_options = options;
    if (resolved_options.state_descriptor_entry == null) {
        resolved_options.state_descriptor_entry = entry;
    }
    return compileLayerProgram(allocator, program, mode, resolved_options);
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
    if (compiled.register_buffer_specs.len > 0) allocator.free(compiled.register_buffer_specs);

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
        .kernel => runtime_contract.expectedWeightRefCount(opcode_map.opcodeForLayerOp(op)),
        .linear, .add_param, .add_param_scalar, .mul_param => 1,
        else => 0,
    };
}

fn collectExpectedRegisters(
    reg_map: *RegisterMap,
    op: layer_ops.LayerOp,
    input_buffer: *[64]runtime_contract.RegisterRef,
    output_buffer: *[64]runtime_contract.RegisterRef,
) struct { input_count: usize, output_count: usize } {
    return switch (op) {
        .kernel => |kernel_op| blk: {
            input_buffer[0] = reg_map.registerFor(kernel_op.in);
            output_buffer[0] = reg_map.registerFor(kernel_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .add => |add_op| blk: {
            input_buffer[0] = reg_map.registerFor(.residual);
            input_buffer[1] = reg_map.registerFor(add_op.branch);
            output_buffer[0] = reg_map.registerFor(.residual);
            break :blk .{ .input_count = 2, .output_count = 1 };
        },
        .linear => |linear_op| blk: {
            input_buffer[0] = reg_map.registerFor(linear_op.in);
            output_buffer[0] = reg_map.registerFor(linear_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .matmul => |matmul_op| blk: {
            input_buffer[0] = reg_map.registerFor(matmul_op.in_a);
            input_buffer[1] = reg_map.registerFor(matmul_op.in_b);
            output_buffer[0] = reg_map.registerFor(matmul_op.out);
            break :blk .{ .input_count = 2, .output_count = 1 };
        },
        .split => |split_op| blk: {
            input_buffer[0] = reg_map.registerFor(split_op.in);
            const first_index: u16 = @intFromEnum(split_op.out_start);
            for (0..split_op.num_outputs) |idx| {
                const buf_id: layer_ops.BufferId = @enumFromInt(first_index + @as(u16, @intCast(idx)));
                output_buffer[idx] = reg_map.registerFor(buf_id);
            }
            break :blk .{ .input_count = 1, .output_count = split_op.num_outputs };
        },
        .softmax => |softmax_op| blk: {
            input_buffer[0] = reg_map.registerFor(softmax_op.in);
            output_buffer[0] = reg_map.registerFor(softmax_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .silu => |silu_op| blk: {
            input_buffer[0] = reg_map.registerFor(silu_op.in);
            output_buffer[0] = reg_map.registerFor(silu_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .gelu => |gelu_op| blk: {
            input_buffer[0] = reg_map.registerFor(gelu_op.in);
            output_buffer[0] = reg_map.registerFor(gelu_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .mul => |mul_op| blk: {
            input_buffer[0] = reg_map.registerFor(mul_op.in);
            input_buffer[1] = reg_map.registerFor(mul_op.other);
            output_buffer[0] = reg_map.registerFor(mul_op.out);
            break :blk .{ .input_count = 2, .output_count = 1 };
        },
        .add_tensor => |add_tensor_op| blk: {
            input_buffer[0] = reg_map.registerFor(add_tensor_op.in_a);
            input_buffer[1] = reg_map.registerFor(add_tensor_op.in_b);
            output_buffer[0] = reg_map.registerFor(add_tensor_op.out);
            break :blk .{ .input_count = 2, .output_count = 1 };
        },
        .add_scalar => |add_scalar_op| blk: {
            input_buffer[0] = reg_map.registerFor(add_scalar_op.in);
            output_buffer[0] = reg_map.registerFor(add_scalar_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .mul_scalar => |mul_scalar_op| blk: {
            input_buffer[0] = reg_map.registerFor(mul_scalar_op.in);
            output_buffer[0] = reg_map.registerFor(mul_scalar_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .mean => |mean_op| blk: {
            input_buffer[0] = reg_map.registerFor(mean_op.in);
            output_buffer[0] = reg_map.registerFor(mean_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .pow => |pow_op| blk: {
            input_buffer[0] = reg_map.registerFor(pow_op.in);
            output_buffer[0] = reg_map.registerFor(pow_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .rsqrt => |rsqrt_op| blk: {
            input_buffer[0] = reg_map.registerFor(rsqrt_op.in);
            output_buffer[0] = reg_map.registerFor(rsqrt_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .add_param => |add_param_op| blk: {
            input_buffer[0] = reg_map.registerFor(add_param_op.in);
            output_buffer[0] = reg_map.registerFor(add_param_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .add_param_scalar => |add_param_scalar_op| blk: {
            output_buffer[0] = reg_map.registerFor(add_param_scalar_op.out);
            break :blk .{ .input_count = 0, .output_count = 1 };
        },
        .mul_param => |mul_param_op| blk: {
            input_buffer[0] = reg_map.registerFor(mul_param_op.in);
            output_buffer[0] = reg_map.registerFor(mul_param_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .reshape => |reshape_op| blk: {
            input_buffer[0] = reg_map.registerFor(reshape_op.in);
            output_buffer[0] = reg_map.registerFor(reshape_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .transpose => |transpose_op| blk: {
            input_buffer[0] = reg_map.registerFor(transpose_op.in);
            output_buffer[0] = reg_map.registerFor(transpose_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .rope => |rope_op| blk: {
            input_buffer[0] = reg_map.registerFor(rope_op.in);
            output_buffer[0] = reg_map.registerFor(rope_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .triu => |triu_op| blk: {
            input_buffer[0] = reg_map.registerFor(triu_op.in);
            output_buffer[0] = reg_map.registerFor(triu_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .sdpa => |sdpa_op| blk: {
            input_buffer[0] = reg_map.registerFor(sdpa_op.q);
            input_buffer[1] = reg_map.registerFor(sdpa_op.k);
            input_buffer[2] = reg_map.registerFor(sdpa_op.v);
            output_buffer[0] = reg_map.registerFor(sdpa_op.out);
            break :blk .{ .input_count = 3, .output_count = 1 };
        },
        .patch_embed => |patch_op| blk: {
            input_buffer[0] = reg_map.registerFor(patch_op.in);
            output_buffer[0] = reg_map.registerFor(patch_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .spatial_merge => |spatial_op| blk: {
            input_buffer[0] = reg_map.registerFor(spatial_op.in);
            output_buffer[0] = reg_map.registerFor(spatial_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .deepstack_extract => |deepstack_op| blk: {
            input_buffer[0] = reg_map.registerFor(deepstack_op.in);
            output_buffer[0] = reg_map.registerFor(deepstack_op.out);
            break :blk .{ .input_count = 1, .output_count = 1 };
        },
        .scatter => |scatter_op| blk: {
            input_buffer[0] = reg_map.registerFor(scatter_op.text_in);
            input_buffer[1] = reg_map.registerFor(scatter_op.vision_in);
            output_buffer[0] = reg_map.registerFor(scatter_op.out);
            break :blk .{ .input_count = 2, .output_count = 1 };
        },
    };
}

fn expectProgramParity(source: []const layer_ops.LayerOp, compiled: *const runtime_contract.CompiledPlan) !void {
    try std.testing.expectEqual(source.len, compiled.plan.instructions.len);
    var reg_map = RegisterMap.init();
    for (source, compiled.plan.instructions) |source_op, compiled_insn| {
        try std.testing.expectEqual(opcode_map.opcodeForLayerOp(source_op), compiled_insn.opcode);
        try std.testing.expectEqual(expectedInputCount(source_op), compiled_insn.inputs.len);
        try std.testing.expectEqual(expectedOutputCount(source_op), compiled_insn.outputs.len);
        try std.testing.expectEqual(expectedWeightCount(source_op), compiled_insn.weights.len);

        var expected_inputs: [64]runtime_contract.RegisterRef = undefined;
        var expected_outputs: [64]runtime_contract.RegisterRef = undefined;
        const expected = collectExpectedRegisters(&reg_map, source_op, &expected_inputs, &expected_outputs);
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
    var compiled = try compileLayerProgram(std.testing.allocator, llama3.attention_mlp_program, .decode, .{});
    defer deinitCompiledPlan(std.testing.allocator, &compiled);

    try expectProgramParity(llama3.attention_mlp_program, &compiled);
}

test "compileLayerProgram preserves structural parity for granite_hybrid mamba" {
    var compiled = try compileLayerProgram(std.testing.allocator, granite_hybrid.mamba_program, .prefill, .{});
    defer deinitCompiledPlan(std.testing.allocator, &compiled);

    try expectProgramParity(granite_hybrid.mamba_program, &compiled);
}

test "compileLayerProgram preserves structural parity for qwen3_moe" {
    var compiled = try compileLayerProgram(std.testing.allocator, qwen3_moe.attention_mlp_program, .decode, .{});
    defer deinitCompiledPlan(std.testing.allocator, &compiled);

    try expectProgramParity(qwen3_moe.attention_mlp_program, &compiled);
}

test "compileLayerProgram emits KV state descriptor and attention state references" {
    var compiled = try compileLayerProgram(std.testing.allocator, llama3.attention_mlp_program, .decode, .{});
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

test "compileLayerProgram preserves explicit state metadata in vision mode" {
    const program = [_]layer_ops.LayerOp{
        .{ .kernel = .{
            .id = 0,
            .in = .residual,
            .out = .norm_out,
            .debug_type = .multihead_attention,
            .state_block_id = runtime_contract.kv_cache_state_id,
        } },
    };

    var compiled = try compileLayerProgram(std.testing.allocator, &program, .vision_encode, .{});
    defer deinitCompiledPlan(std.testing.allocator, &compiled);

    try std.testing.expectEqual(@as(usize, 1), compiled.plan.state_descs.len);
    try std.testing.expectEqual(@as(?u8, runtime_contract.kv_cache_state_id), compiled.plan.instructions[0].state_block_id);
}

test "compileLayerProgram keeps attention stateless when state metadata is omitted" {
    const program = [_]layer_ops.LayerOp{
        .{ .kernel = .{
            .id = 0,
            .in = .residual,
            .out = .norm_out,
            .debug_type = .multihead_attention,
        } },
    };

    var compiled = try compileLayerProgram(std.testing.allocator, &program, .vision_encode, .{});
    defer deinitCompiledPlan(std.testing.allocator, &compiled);

    try std.testing.expectEqual(@as(usize, 0), compiled.plan.state_descs.len);
    try std.testing.expectEqual(@as(?u8, null), compiled.plan.instructions[0].state_block_id);
}

test "compileLayerProgram emits mamba state descriptor and mixer state references" {
    var compiled = try compileLayerProgram(std.testing.allocator, granite_hybrid.mamba_program, .prefill, .{});
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
            .state_block_id = runtime_contract.shortconv_state_id,
        } },
    };
    var compiled = try compileLayerProgram(std.testing.allocator, &shortconv_program, .decode, .{});
    defer deinitCompiledPlan(std.testing.allocator, &compiled);

    try std.testing.expect(hasStateDescriptor(&compiled, shortconv_state_id));
    try std.testing.expectEqual(@as(?u8, shortconv_state_id), compiled.plan.instructions[0].state_block_id);
}

test "buildStateDescriptors accepts unknown state ids" {
    const allocator = std.testing.allocator;
    const instructions = [_]runtime_contract.Instruction{
        .{
            .opcode = .residual_add,
            .inputs = &.{},
            .outputs = &.{},
            .weights = &.{},
            .param_block_id = null,
            .state_block_id = 77,
        },
    };
    const descriptors = try buildStateDescriptors(allocator, instructions[0..], null);
    defer if (descriptors.len > 0) allocator.free(descriptors);
    try std.testing.expectEqual(@as(usize, 1), descriptors.len);
    try std.testing.expectEqual(@as(u8, 77), descriptors[0].id);
    try std.testing.expectEqual(runtime_contract.StateLifecycle.request_scoped, descriptors[0].lifecycle);
}

test "buildStateDescriptors preserves builtin descriptor for builtin state ids without metadata entry" {
    const allocator = std.testing.allocator;
    const instructions = [_]runtime_contract.Instruction{
        .{
            .opcode = .multihead_attention,
            .inputs = &.{},
            .outputs = &.{},
            .weights = &.{},
            .param_block_id = null,
            .state_block_id = runtime_contract.kv_cache_state_id,
        },
    };
    const descriptors = try buildStateDescriptors(allocator, instructions[0..], null);
    defer if (descriptors.len > 0) allocator.free(descriptors);
    try std.testing.expectEqual(@as(usize, 1), descriptors.len);
    try std.testing.expectEqual(runtime_contract.kv_cache_state_id, descriptors[0].id);
    try std.testing.expectEqual(runtime_contract.StateLifecycle.slot_persistent, descriptors[0].lifecycle);
    try std.testing.expectEqual(runtime_contract.state_runtime_kind_kv_cache, descriptors[0].runtime_kind);
}

test "buildStateDescriptors uses architecture metadata when entry is provided" {
    const allocator = std.testing.allocator;
    const entry = registry.detectByArchitectureId("llama3") orelse return error.TestUnexpectedResult;
    const instructions = [_]runtime_contract.Instruction{
        .{
            .opcode = .residual_add,
            .inputs = &.{},
            .outputs = &.{},
            .weights = &.{},
            .param_block_id = null,
            .state_block_id = runtime_contract.kv_cache_state_id,
        },
    };
    const descriptors = try buildStateDescriptors(allocator, instructions[0..], entry);
    defer if (descriptors.len > 0) allocator.free(descriptors);
    try std.testing.expectEqual(@as(usize, 1), descriptors.len);
    try std.testing.expectEqual(runtime_contract.kv_cache_state_id, descriptors[0].id);
    try std.testing.expectEqual(runtime_contract.StateLifecycle.slot_persistent, descriptors[0].lifecycle);
}

test "compileLayerProgram emits structured kernel weight refs for macro attention ops" {
    const program = [_]layer_ops.LayerOp{
        .{ .kernel = .{
            .id = 7,
            .in = .residual,
            .out = .branch_out,
            .debug_type = .multihead_attention,
            .state_block_id = runtime_contract.kv_cache_state_id,
        } },
    };
    var compiled = try compileLayerProgram(std.testing.allocator, &program, .decode, .{});
    defer deinitCompiledPlan(std.testing.allocator, &compiled);

    const insn = compiled.plan.instructions[0];
    try std.testing.expectEqual(@as(usize, 11), insn.weights.len);
    const expected_slots = [_][]const u8{
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "q_norm",
        "k_norm",
        "q_bias",
        "k_bias",
        "v_bias",
        "o_bias",
        "attn_sinks",
    };
    for (expected_slots, 0..) |slot_name, idx| {
        const binding_name = try runtime_contract.instructionWeightBindingName(&compiled, 0, idx);
        const parsed = try runtime_contract.parseKernelWeightBindingName(binding_name);
        try std.testing.expectEqual(@as(u32, 7), parsed.kernel_id);
        try std.testing.expectEqualStrings(slot_name, parsed.slot_name);
    }
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
    var compiled = try compileLayerProgram(std.testing.allocator, &program, .decode, .{});
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
    var compiled = try compileLayerProgram(std.testing.allocator, &program, .decode, .{});
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
    var compiled = try compileLayerProgram(std.testing.allocator, llama3.attention_mlp_program, .decode, .{});
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
    var compiled = try compileLayerProgram(std.testing.allocator, &program, .decode, .{});
    defer deinitCompiledPlan(std.testing.allocator, &compiled);

    for (program, compiled.plan.instructions, 0..) |source_op, insn, op_index| {
        const decoded = try runtime_contract.decodeInstructionLayerOp(&compiled, &insn, op_index);
        switch (source_op) {
            .kernel => |kernel| switch (decoded) {
                .kernel => |decoded_kernel| {
                    try std.testing.expectEqual(kernel.id, decoded_kernel.id);
                    try std.testing.expectEqual(kernel.debug_type, decoded_kernel.debug_type);
                },
                else => return error.TestUnexpectedResult,
            },
            .add => |add_op| switch (decoded) {
                .add => |decoded_add| {
                    try std.testing.expect(std.meta.eql(add_op.scale, decoded_add.scale));
                },
                else => return error.TestUnexpectedResult,
            },
            .mul_scalar => |mul_scalar_op| switch (decoded) {
                .mul_scalar => |decoded_mul| {
                    try std.testing.expectEqual(mul_scalar_op.scalar, decoded_mul.scalar);
                },
                else => return error.TestUnexpectedResult,
            },
            .add_scalar => |add_scalar_op| switch (decoded) {
                .add_scalar => |decoded_add_scalar| {
                    try std.testing.expectEqual(add_scalar_op.scalar, decoded_add_scalar.scalar);
                },
                else => return error.TestUnexpectedResult,
            },
            else => return error.TestUnexpectedResult,
        }
    }
}

test "compileProgramForArchitecture resolves registry programs" {
    var compiled = try compileProgramForArchitecture(std.testing.allocator, "granite_hybrid", .mamba, .decode, .{});
    defer deinitCompiledPlan(std.testing.allocator, &compiled);

    try std.testing.expect(compiled.plan.instructions.len > 0);
}

test "compileLayerProgram emits summary diagnostics" {
    var compiled = try compileLayerProgram(std.testing.allocator, llama3.attention_mlp_program, .decode, .{});
    defer deinitCompiledPlan(std.testing.allocator, &compiled);

    try std.testing.expect(compiled.diagnostics.len >= 1);
    try std.testing.expectEqual(runtime_contract.PlanDiagnosticLevel.info, compiled.diagnostics[0].level);
    try std.testing.expect(std.mem.startsWith(u8, compiled.diagnostics[0].message, "plan_compiled mode=decode"));
}

test "compileLayerProgram is deterministic across repeated compiles of same program" {
    var first = try compileLayerProgram(std.testing.allocator, qwen3_moe.attention_mlp_program, .decode, .{});
    defer deinitCompiledPlan(std.testing.allocator, &first);
    var second = try compileLayerProgram(std.testing.allocator, qwen3_moe.attention_mlp_program, .decode, .{});
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
    var compiled = try compileLayerProgram(std.testing.allocator, &.{}, .decode, .{});
    defer deinitCompiledPlan(std.testing.allocator, &compiled);

    try std.testing.expect(compiled.diagnostics.len >= 2);
    try std.testing.expectEqual(runtime_contract.PlanDiagnosticLevel.warn, compiled.diagnostics[1].level);
    try std.testing.expect(std.mem.startsWith(u8, compiled.diagnostics[1].message, "empty_plan mode=decode"));
}

test "compileLayerProgram emits never_read diagnostic for unused output" {
    // Op 0 outputs to norm_out (reg 1), but op 1 reads residual (reg 0) only.
    // norm_out is produced but never read — should emit a never_read diagnostic.
    // branch_out (reg 2) is the plan's final output, so it is NOT flagged.
    const program = [_]layer_ops.LayerOp{
        .{ .mul_scalar = .{ .in = .residual, .out = .norm_out, .scalar = 1.0 } },
        .{ .mul_scalar = .{ .in = .residual, .out = .branch_out, .scalar = 2.0 } },
    };
    var compiled = try compileLayerProgram(std.testing.allocator, &program, .decode, .{});
    defer deinitCompiledPlan(std.testing.allocator, &compiled);

    var found_never_read = false;
    for (compiled.diagnostics) |diag| {
        if (diag.level == .warn and std.mem.startsWith(u8, diag.message, "never_read register=")) {
            // Should flag register 1 (norm_out), not register 2 (branch_out = final output)
            try std.testing.expect(std.mem.eql(u8, diag.message, "never_read register=1"));
            found_never_read = true;
        }
    }
    try std.testing.expect(found_never_read);
}

test "compileLayerProgram assigns dense registers for sparse tmp buffers" {
    const allocator = std.testing.allocator;
    // Program uses tmp3, tmp10, tmp20 — sparse BufferIds (values 3, 10, 20).
    // Dense allocation: only residual=0 is fixed; all others by first-use order.
    const program = [_]layer_ops.LayerOp{
        .{ .silu = .{ .in = .norm_out, .out = .tmp3 } },
        .{ .silu = .{ .in = .tmp3, .out = .tmp10 } },
        .{ .silu = .{ .in = .tmp10, .out = .tmp20 } },
    };
    var compiled = try compileLayerProgram(allocator, &program, .decode, .{});
    defer deinitCompiledPlan(allocator, &compiled);

    // residual=0 (fixed), norm_out→1, tmp3→2, tmp10→3, tmp20→4 (by first-use)
    try std.testing.expectEqual(@as(u16, 5), compiled.plan.register_count);

    // Verify registers in instructions are NOT identity-cast BufferIds
    const insn1 = compiled.plan.instructions[1]; // silu: tmp3→tmp10
    try std.testing.expectEqual(@as(u16, 2), runtime_contract.registerToIndex(insn1.inputs[0])); // tmp3 → register 2
    try std.testing.expectEqual(@as(u16, 3), runtime_contract.registerToIndex(insn1.outputs[0])); // tmp10 → register 3 (NOT 10)

    const insn2 = compiled.plan.instructions[2]; // silu: tmp10→tmp20
    try std.testing.expectEqual(@as(u16, 3), runtime_contract.registerToIndex(insn2.inputs[0])); // tmp10 → register 3
    try std.testing.expectEqual(@as(u16, 4), runtime_contract.registerToIndex(insn2.outputs[0])); // tmp20 → register 4 (NOT 20)
}

test "compileLayerProgram assigns dense registers for split outputs" {
    const allocator = std.testing.allocator;
    const program = [_]layer_ops.LayerOp{
        .{ .split = .{ .in = .norm_out, .out_start = .tmp10, .num_outputs = 3, .split_sizes = &.{}, .dim = -1 } },
    };
    var compiled = try compileLayerProgram(allocator, &program, .decode, .{});
    defer deinitCompiledPlan(allocator, &compiled);

    // norm_out→1 (first-use as input), split outputs tmp10→2, tmp11→3, tmp12→4
    const insn = compiled.plan.instructions[0];
    try std.testing.expectEqual(@as(u16, 2), runtime_contract.registerToIndex(insn.outputs[0])); // tmp10 → 2
    try std.testing.expectEqual(@as(u16, 3), runtime_contract.registerToIndex(insn.outputs[1])); // tmp11 → 3
    try std.testing.expectEqual(@as(u16, 4), runtime_contract.registerToIndex(insn.outputs[2])); // tmp12 → 4

}

test "compileLayerProgram does not pin intermediates to legacy IDs" {
    const allocator = std.testing.allocator;
    // Program where branch_out is used BEFORE norm_out.
    // With dynamic assignment, branch_out gets the lower register (first-use).
    const program = [_]layer_ops.LayerOp{
        .{ .silu = .{ .in = .residual, .out = .branch_out } }, // branch_out used first
        .{ .silu = .{ .in = .branch_out, .out = .norm_out } }, // norm_out used second
    };
    var compiled = try compileLayerProgram(allocator, &program, .decode, .{});
    defer deinitCompiledPlan(allocator, &compiled);
    // branch_out → register 1 (first intermediate), norm_out → register 2 (second)
    const insn0 = compiled.plan.instructions[0];
    try std.testing.expectEqual(@as(u16, 1), runtime_contract.registerToIndex(insn0.outputs[0]));
    const insn1 = compiled.plan.instructions[1];
    try std.testing.expectEqual(@as(u16, 2), runtime_contract.registerToIndex(insn1.outputs[0]));
}

test "dead output is killed at producer for immediate physical reclaim" {
    const allocator = std.testing.allocator;
    // insn0: residual → tmp3 (never read)
    // insn1: residual → tmp10 (never read)
    // insn2: residual → branch_out (final output — must survive)
    const program = [_]layer_ops.LayerOp{
        .{ .silu = .{ .in = .residual, .out = .tmp3 } },
        .{ .silu = .{ .in = .residual, .out = .tmp10 } },
        .{ .silu = .{ .in = .residual, .out = .branch_out } },
    };
    var compiled = try compileLayerProgram(allocator, &program, .decode, .{});
    defer deinitCompiledPlan(allocator, &compiled);

    // tmp3 and tmp10 are dead → killed at their producers → can share physical buffer.
    // Only branch_out (final output) stays live → reduced peak pressure.
    const tmp3_reg = runtime_contract.registerToIndex(compiled.plan.instructions[0].outputs[0]);
    const tmp10_reg = runtime_contract.registerToIndex(compiled.plan.instructions[1].outputs[0]);
    const branch_reg = runtime_contract.registerToIndex(compiled.plan.instructions[2].outputs[0]);

    // tmp3 killed at instruction 0 (its producer), tmp10 killed at instruction 1 (its producer).
    const kill0 = compiled.liveness.kill_after_instruction[0];
    const kill1 = compiled.liveness.kill_after_instruction[1];
    try std.testing.expect((kill0[tmp3_reg / 64] & (@as(u64, 1) << @intCast(tmp3_reg % 64))) != 0);
    try std.testing.expect((kill1[tmp10_reg / 64] & (@as(u64, 1) << @intCast(tmp10_reg % 64))) != 0);

    // Final output (branch_out) must NOT be killed at its producer — it stays live until terminal.
    const kill2 = compiled.liveness.kill_after_instruction[2];
    try std.testing.expect((kill2[branch_reg / 64] & (@as(u64, 1) << @intCast(branch_reg % 64))) != 0);
    // And branch_out must NOT appear in kill0 or kill1.
    try std.testing.expect((kill0[branch_reg / 64] & (@as(u64, 1) << @intCast(branch_reg % 64))) == 0);
    try std.testing.expect((kill1[branch_reg / 64] & (@as(u64, 1) << @intCast(branch_reg % 64))) == 0);
}

test "validateProgramBlockKindStateCompatibility does not enforce builtin topology" {
    const program = [_]layer_ops.LayerOp{
        .{ .kernel = .{
            .id = 0,
            .in = .residual,
            .out = .branch_out,
            .debug_type = .shortconv,
            .state_block_id = runtime_contract.shortconv_state_id,
        } },
    };
    try validateProgramBlockKindStateCompatibility(&program, .attention_mlp);
}
