//! Logical-register to physical-buffer mapping helpers (Phase 2).
//!
//! Backend-agnostic allocator: uses plan liveness to reuse physical buffers.

const std = @import("std");
const types = @import("types.zig");

pub const RegisterBufferSpec = struct {
    size: usize,
    @"align": u16,
};

const invalid_physical: u16 = std.math.maxInt(u16);

fn maxAlign(a: u16, b: u16) u16 {
    return if (a > b) a else b;
}

pub fn buildPhysicalMappingLinearScan(
    allocator: std.mem.Allocator,
    compiled: *const types.CompiledPlan,
    register_specs: []const RegisterBufferSpec,
) !types.PhysicalMapping {
    const register_count = @as(usize, compiled.plan.register_count);
    if (register_specs.len != register_count) return error.InvalidRegisterSpecCount;
    if (register_count == 0) {
        return .{
            .register_to_physical = try allocator.alloc(u16, 0),
            .physical_count = 0,
            .physical_specs = try allocator.alloc(types.PhysicalBufferSpec, 0),
        };
    }

    var register_to_physical = try allocator.alloc(u16, register_count);
    errdefer allocator.free(register_to_physical);
    @memset(register_to_physical, invalid_physical);

    var register_first_write = try allocator.alloc(u32, register_count);
    defer allocator.free(register_first_write);
    @memset(register_first_write, std.math.maxInt(u32));

    for (compiled.plan.instructions, 0..) |insn, instruction_idx| {
        const idx_u32: u32 = @intCast(instruction_idx);
        for (insn.outputs) |reg| {
            const reg_idx = types.registerToIndex(reg);
            if (register_first_write[reg_idx] == std.math.maxInt(u32)) {
                register_first_write[reg_idx] = idx_u32;
            }
        }
    }

    var physical_specs = std.ArrayList(types.PhysicalBufferSpec).empty;
    errdefer physical_specs.deinit(allocator);
    var free_list = std.ArrayList(u16).empty;
    defer free_list.deinit(allocator);

    for (compiled.plan.instructions, 0..) |insn, instruction_idx| {
        const idx_u32: u32 = @intCast(instruction_idx);

        for (insn.outputs) |reg| {
            const reg_idx = types.registerToIndex(reg);
            if (register_first_write[reg_idx] != idx_u32) continue;
            const desired = register_specs[reg_idx];

            var chosen: ?u16 = null;
            var free_idx: usize = 0;
            while (free_idx < free_list.items.len) : (free_idx += 1) {
                const candidate = free_list.items[free_idx];
                const spec = physical_specs.items[candidate];
                if (spec.size >= desired.size and spec.@"align" >= desired.@"align") {
                    chosen = candidate;
                    _ = free_list.swapRemove(free_idx);
                    break;
                }
            }

            const physical_id: u16 = if (chosen) |id|
                id
            else blk: {
                const new_id: u16 = @intCast(physical_specs.items.len);
                try physical_specs.append(allocator, .{
                    .size = desired.size,
                    .@"align" = desired.@"align",
                });
                break :blk new_id;
            };

            register_to_physical[reg_idx] = physical_id;
            var maybe_spec = &physical_specs.items[physical_id];
            maybe_spec.size = @max(maybe_spec.size, desired.size);
            maybe_spec.@"align" = maxAlign(maybe_spec.@"align", desired.@"align");
        }

        const kill_row = compiled.liveness.kill_after_instruction[instruction_idx];
        for (0..register_count) |reg_idx| {
            const word = reg_idx / 64;
            const bit: u6 = @intCast(reg_idx % 64);
            if ((kill_row[word] & (@as(u64, 1) << bit)) == 0) continue;
            const physical_id = register_to_physical[reg_idx];
            if (physical_id == invalid_physical) continue;
            try free_list.append(allocator, physical_id);
        }
    }

    const physical_specs_slice = try physical_specs.toOwnedSlice(allocator);
    return .{
        .register_to_physical = register_to_physical,
        .physical_count = @intCast(physical_specs_slice.len),
        .physical_specs = physical_specs_slice,
    };
}

pub fn deinitPhysicalMapping(allocator: std.mem.Allocator, mapping: *types.PhysicalMapping) void {
    allocator.free(mapping.register_to_physical);
    allocator.free(mapping.physical_specs);
    mapping.* = undefined;
}

test "buildPhysicalMappingLinearScan reuses physical buffers with liveness" {
    const reg0 = types.registerFromIndex(0);
    const reg1 = types.registerFromIndex(1);
    const reg2 = types.registerFromIndex(2);

    const instructions = [_]types.Instruction{
        .{
            .opcode = .rmsnorm,
            .inputs = &.{reg0},
            .outputs = &.{reg1},
            .weights = &.{},
            .param_block_id = null,
            .state_block_id = null,
        },
        .{
            .opcode = .swiglu,
            .inputs = &.{reg1},
            .outputs = &.{reg2},
            .weights = &.{},
            .param_block_id = null,
            .state_block_id = null,
        },
    };

    const kill0 = [_]u64{0b10}; // register 1 dies after instruction 0
    const kill1 = [_]u64{0b101}; // registers 0 and 2 die after instruction 1
    const liveness = types.LivenessMap{
        .register_last_read = &.{ 1, 0, 1 },
        .kill_after_instruction = &.{ kill0[0..], kill1[0..] },
    };
    const compiled = types.CompiledPlan{
        .plan = .{
            .instructions = instructions[0..],
            .register_count = 3,
            .state_descs = &.{},
        },
        .param_blocks = &.{},
        .liveness = liveness,
        .peak_registers = 2,
        .diagnostics = &.{},
    };

    const specs = [_]RegisterBufferSpec{
        .{ .size = 1024, .@"align" = 16 },
        .{ .size = 1024, .@"align" = 16 },
        .{ .size = 1024, .@"align" = 16 },
    };
    var mapping = try buildPhysicalMappingLinearScan(std.testing.allocator, &compiled, specs[0..]);
    defer deinitPhysicalMapping(std.testing.allocator, &mapping);

    try std.testing.expect(mapping.physical_count <= compiled.peak_registers);
    try std.testing.expectEqual(mapping.register_to_physical[1], mapping.register_to_physical[2]);
}
