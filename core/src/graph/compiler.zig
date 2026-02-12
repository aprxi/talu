//! Graph Compiler
//!
//! Compiles graph ops (from JSON) to LayerOp bytecode for the Zig executor.
//! This is the core "JIT" that bridges Python's declarative model definitions
//! to Zig's high-performance execution engine.
//!
//! ## Input
//!
//! A flat sequence of ops traced from Python's forward() method:
//! ```python
//! def forward(self, x):
//!     h = self.input_layernorm(x)       # → norm
//!     h = self.self_attn(h)             # → multihead_attention
//!     x = x + h                         # → add
//!     ...
//! ```
//!
//! ## Output
//!
//! An optimized LayerOp[] with explicit buffer assignments for the Zig executor.
//!
//! ## Buffer Inference
//!
//! Python doesn't specify buffers - the compiler infers them from data flow:
//!
//! 1. **residual**: The main hidden state that persists across the block
//! 2. **norm_out**: Temporary buffer for norm output (feeds into attn/ffn)
//! 3. **branch_out**: Temporary buffer for attn/ffn output (feeds into add)
//!
//! Rules:
//! - Norm after residual add → reads from `residual`, writes to `norm_out`
//! - Norm after attn/ffn → reads from `branch_out`, writes to `branch_out`
//! - Attn/FFN → reads from `norm_out`, writes to `branch_out`
//! - Add → reads from `branch_out`, adds to `residual`

const std = @import("std");
const Allocator = std.mem.Allocator;
const log = @import("../log.zig");

const types = @import("types.zig");
const Op = types.Op;
const OpType = types.OpType;
const OpInput = types.OpInput;

const ops_mod = @import("layer_ops.zig");
const LayerOp = ops_mod.LayerOp;
const BufferId = ops_mod.BufferId;
const ResidualScale = ops_mod.ResidualScale;

// =============================================================================
// Buffer Planning
// =============================================================================

const BufferPlanner = struct {
    next_temp: u8,
    tensor_to_buffer: *std.StringHashMapUnmanaged(BufferId),
    allocator: Allocator,

    fn allocTemp(self: *BufferPlanner) !BufferId {
        const max_temp = @intFromEnum(BufferId.tmp63);
        if (self.next_temp > max_temp) return error.OutOfScratchBuffers;
        const buffer_id: BufferId = @enumFromInt(self.next_temp);
        self.next_temp += 1;
        return buffer_id;
    }

    fn outputFor(self: *BufferPlanner, name: []const u8) !BufferId {
        if (self.tensor_to_buffer.get(name)) |buffer_id| return buffer_id;
        const buffer_id = try self.allocTemp();
        try self.tensor_to_buffer.put(self.allocator, name, buffer_id);
        return buffer_id;
    }
};

fn linearOutputBuffer(weight_name: []const u8) BufferId {
    if (std.mem.eql(u8, weight_name, "q_proj") or std.mem.endsWith(u8, weight_name, ".q_proj")) return .tmp3;
    if (std.mem.eql(u8, weight_name, "k_proj") or std.mem.endsWith(u8, weight_name, ".k_proj")) return .tmp4;
    if (std.mem.eql(u8, weight_name, "v_proj") or std.mem.endsWith(u8, weight_name, ".v_proj")) return .tmp5;
    if (std.mem.eql(u8, weight_name, "gate_proj") or std.mem.endsWith(u8, weight_name, ".gate_proj")) return .tmp3;
    if (std.mem.eql(u8, weight_name, "up_proj") or std.mem.endsWith(u8, weight_name, ".up_proj")) return .tmp4;
    return .branch_out;
}

fn isQKNormName(name: ?[]const u8) bool {
    if (name) |n| {
        return std.mem.endsWith(u8, n, "q_norm") or std.mem.endsWith(u8, n, "k_norm");
    }
    return false;
}

// =============================================================================
// Compiler
// =============================================================================

/// Compile graph ops to LayerOp bytecode.
pub fn compile(allocator: Allocator, graph_ops: []const Op) ![]const LayerOp {
    log.trace("graph", "Compile start", .{ .ops_count = graph_ops.len }, @src());

    var compiled_ops = std.ArrayListUnmanaged(LayerOp){};
    errdefer compiled_ops.deinit(allocator);

    // Track buffer state for proper wiring
    var last_was_residual = true;
    var last_was_attn_or_ffn = false;
    var kernel_id_counter: u32 = 0;

    // Track which buffer each tensor name is stored in (for primitive op dataflow)
    var tensor_to_buffer = std.StringHashMapUnmanaged(BufferId){};
    defer tensor_to_buffer.deinit(allocator);
    var scaled_tensors = std.StringHashMapUnmanaged(void){};
    defer scaled_tensors.deinit(allocator);
    var planner = BufferPlanner{
        .next_temp = @intFromEnum(BufferId.tmp6),
        .tensor_to_buffer = &tensor_to_buffer,
        .allocator = allocator,
    };

    // The input "x" starts in residual
    try tensor_to_buffer.put(allocator, "x", .residual);

    for (graph_ops) |graph_op| {
        switch (graph_op.op_type) {
            .norm => try compileNorm(allocator, &compiled_ops, &tensor_to_buffer, graph_op, &kernel_id_counter, &last_was_residual, &last_was_attn_or_ffn),
            .multihead_attention => try compileAttention(allocator, &compiled_ops, &tensor_to_buffer, graph_op, &kernel_id_counter, &last_was_residual, &last_was_attn_or_ffn),
            .mlp, .moe => try compileMlp(allocator, &compiled_ops, &tensor_to_buffer, graph_op, &kernel_id_counter, &last_was_residual, &last_was_attn_or_ffn),
            .mamba_mixer => try compileMambaMixer(allocator, &compiled_ops, &tensor_to_buffer, graph_op, &kernel_id_counter, &last_was_residual, &last_was_attn_or_ffn),
            .shortconv => try compileShortConv(allocator, &compiled_ops, &tensor_to_buffer, graph_op, &kernel_id_counter, &last_was_residual, &last_was_attn_or_ffn),
            .add => try compileAdd(allocator, &compiled_ops, &tensor_to_buffer, &scaled_tensors, &planner, graph_op, &last_was_residual, &last_was_attn_or_ffn),
            .linear => try compileLinear(allocator, &compiled_ops, &tensor_to_buffer, graph_op),
            .split => try compileSplit(allocator, &compiled_ops, &tensor_to_buffer, graph_op),
            .matmul => try compiled_ops.append(allocator, .{ .matmul = .{ .in_a = .tmp3, .in_b = .tmp4, .out = .tmp6 } }),
            .softmax => try compiled_ops.append(allocator, .{ .softmax = .{ .in = .tmp6, .out = .tmp6, .dim = -1 } }),
            .silu => try compileActivation(allocator, &compiled_ops, &tensor_to_buffer, graph_op, .silu),
            .gelu => try compileActivation(allocator, &compiled_ops, &tensor_to_buffer, graph_op, .gelu),
            .mul => try compileMul(allocator, &compiled_ops, &tensor_to_buffer, &scaled_tensors, &planner, graph_op),
            .mean => try compileMean(allocator, &compiled_ops, &tensor_to_buffer, &planner, graph_op),
            .pow => try compilePow(allocator, &compiled_ops, &tensor_to_buffer, &planner, graph_op),
            .rsqrt => try compileRsqrt(allocator, &compiled_ops, &tensor_to_buffer, &planner, graph_op),
            .reshape => try compileReshape(allocator, &compiled_ops, &tensor_to_buffer, graph_op),
            .transpose => try compileTranspose(allocator, &compiled_ops, &tensor_to_buffer, graph_op),
            .rope => try compileRope(allocator, &compiled_ops, &tensor_to_buffer, graph_op),
            .triu => try compiled_ops.append(allocator, .{ .triu = .{ .in = .tmp6, .out = .tmp6, .diagonal = @intCast(graph_op.dim) } }),
            .scaled_dot_product_attention => try compileSdpa(allocator, &compiled_ops, &tensor_to_buffer, graph_op),
            .embedding => {}, // Not implemented for block ops
        }
    }

    // Note: post-norm architectures (e.g., BERT) end with a norm whose output is NOT
    // in the residual buffer. The block executor handles this via finalOutputBuffer()
    // which copies the result to residual after program execution. We do NOT rewrite
    // the norm to output in-place to residual, because in-place LayerNorm (reading and
    // writing the same buffer) causes numerical corruption.

    log.trace("graph", "Compile complete", .{ .compiled_ops = compiled_ops.items.len, .next_temp = planner.next_temp }, @src());

    return try compiled_ops.toOwnedSlice(allocator);
}

/// Check if a compiled program uses primitive ops (linear, split, rope, etc.).
/// Returns true if the program requires the non-batched execution path.
pub fn usesPrimitiveOps(program: []const LayerOp) bool {
    for (program) |op| {
        switch (op) {
            // High-level ops supported by batched path
            .kernel, .add, .add_tensor, .mul_scalar => {},
            // Everything else is a primitive op
            else => return true,
        }
    }
    return false;
}

/// Return the buffer that holds the final output of the program.
/// For pre-norm architectures this is `.residual`. For post-norm (e.g., BERT)
/// the last op is a norm whose output is `.norm_out`.
pub fn finalOutputBuffer(program: []const LayerOp) BufferId {
    if (program.len == 0) return .residual;
    const last = program[program.len - 1];
    return switch (last) {
        .kernel => |k| k.out,
        .add => .residual,
        .add_tensor => |at| at.out,
        else => .residual,
    };
}

// =============================================================================
// Op Compilers
// =============================================================================

fn compileNorm(
    allocator: Allocator,
    layer_ops: *std.ArrayListUnmanaged(LayerOp),
    tensor_to_buffer: *std.StringHashMapUnmanaged(BufferId),
    graph_op: Op,
    kernel_id_counter: *u32,
    last_was_residual: *bool,
    last_was_attn_or_ffn: *bool,
) !void {
    // QK norm is handled differently - just track the buffer
    if (isQKNormName(graph_op.name)) {
        var input_buffer: BufferId = .branch_out;
        if (graph_op.inputs.len > 0) {
            switch (graph_op.inputs[0]) {
                .tensor => |tensor_name| {
                    if (tensor_to_buffer.get(tensor_name)) |buffer_id| input_buffer = buffer_id;
                },
                .scalar => {},
            }
        }
        if (graph_op.outputs.len > 0) {
            try tensor_to_buffer.put(allocator, graph_op.outputs[0], input_buffer);
        }
        return;
    }

    // Resolve input buffer: use explicit tensor_to_buffer when available (supports
    // both pre-norm and post-norm architectures), fall back to flag-based heuristic.
    const input_buffer: BufferId = blk: {
        if (graph_op.inputs.len > 0) {
            switch (graph_op.inputs[0]) {
                .tensor => |tensor_name| {
                    if (tensor_to_buffer.get(tensor_name)) |buffer_id| break :blk buffer_id;
                },
                .scalar => {},
            }
        }
        break :blk if (last_was_residual.*) .residual else .branch_out;
    };
    const output_buffer: BufferId = if (last_was_attn_or_ffn.*) .branch_out else .norm_out;

    last_was_residual.* = false;
    last_was_attn_or_ffn.* = false;

    try layer_ops.append(allocator, .{ .kernel = .{
        .id = kernel_id_counter.*,
        .in = input_buffer,
        .out = output_buffer,
        .debug_type = .norm,
    } });
    log.trace("load", "Emitting kernel {d} for {s}", .{ kernel_id_counter.*, @tagName(graph_op.op_type) }, @src());
    kernel_id_counter.* += 1;

    if (graph_op.outputs.len > 0) {
        try tensor_to_buffer.put(allocator, graph_op.outputs[0], output_buffer);
    }
}

fn compileAttention(
    allocator: Allocator,
    layer_ops: *std.ArrayListUnmanaged(LayerOp),
    tensor_to_buffer: *std.StringHashMapUnmanaged(BufferId),
    graph_op: Op,
    kernel_id_counter: *u32,
    last_was_residual: *bool,
    last_was_attn_or_ffn: *bool,
) !void {
    last_was_residual.* = false;
    last_was_attn_or_ffn.* = true;

    var input_buffer: BufferId = .norm_out;
    if (graph_op.inputs.len > 0) {
        switch (graph_op.inputs[0]) {
            .tensor => |tensor_name| {
                if (tensor_to_buffer.get(tensor_name)) |buffer_id| input_buffer = buffer_id;
            },
            .scalar => {},
        }
    }

    try layer_ops.append(allocator, .{ .kernel = .{
        .id = kernel_id_counter.*,
        .in = input_buffer,
        .out = .branch_out,
        .debug_type = .multihead_attention,
    } });
    log.trace("load", "Emitting kernel {d} for {s}", .{ kernel_id_counter.*, @tagName(graph_op.op_type) }, @src());
    kernel_id_counter.* += 1;

    if (graph_op.outputs.len > 0) {
        try tensor_to_buffer.put(allocator, graph_op.outputs[0], .branch_out);
    }
}

fn compileMlp(
    allocator: Allocator,
    layer_ops: *std.ArrayListUnmanaged(LayerOp),
    tensor_to_buffer: *std.StringHashMapUnmanaged(BufferId),
    graph_op: Op,
    kernel_id_counter: *u32,
    last_was_residual: *bool,
    last_was_attn_or_ffn: *bool,
) !void {
    last_was_residual.* = false;
    last_was_attn_or_ffn.* = true;

    var input_buffer: BufferId = .norm_out;
    if (graph_op.inputs.len > 0) {
        switch (graph_op.inputs[0]) {
            .tensor => |tensor_name| {
                if (tensor_to_buffer.get(tensor_name)) |buffer_id| input_buffer = buffer_id;
            },
            .scalar => {},
        }
    }

    try layer_ops.append(allocator, .{ .kernel = .{
        .id = kernel_id_counter.*,
        .in = input_buffer,
        .out = .branch_out,
        .debug_type = if (graph_op.op_type == .moe) .moe else .mlp,
    } });
    log.trace("load", "Emitting kernel {d} for {s}", .{ kernel_id_counter.*, @tagName(graph_op.op_type) }, @src());
    kernel_id_counter.* += 1;

    if (graph_op.outputs.len > 0) {
        try tensor_to_buffer.put(allocator, graph_op.outputs[0], .branch_out);
    }
}

fn compileMambaMixer(
    allocator: Allocator,
    layer_ops: *std.ArrayListUnmanaged(LayerOp),
    tensor_to_buffer: *std.StringHashMapUnmanaged(BufferId),
    graph_op: Op,
    kernel_id_counter: *u32,
    last_was_residual: *bool,
    last_was_attn_or_ffn: *bool,
) !void {
    // Mamba mixer follows the same buffer pattern as attention/mlp
    last_was_residual.* = false;
    last_was_attn_or_ffn.* = true;

    var input_buffer: BufferId = .norm_out;
    if (graph_op.inputs.len > 0) {
        switch (graph_op.inputs[0]) {
            .tensor => |tensor_name| {
                if (tensor_to_buffer.get(tensor_name)) |buffer_id| input_buffer = buffer_id;
            },
            .scalar => {},
        }
    }

    try layer_ops.append(allocator, .{ .kernel = .{
        .id = kernel_id_counter.*,
        .in = input_buffer,
        .out = .branch_out,
        .debug_type = .mamba_mixer,
    } });
    log.trace("load", "Emitting kernel {d} for {s}", .{ kernel_id_counter.*, @tagName(graph_op.op_type) }, @src());
    kernel_id_counter.* += 1;

    if (graph_op.outputs.len > 0) {
        try tensor_to_buffer.put(allocator, graph_op.outputs[0], .branch_out);
    }
}

fn compileShortConv(
    allocator: Allocator,
    layer_ops: *std.ArrayListUnmanaged(LayerOp),
    tensor_to_buffer: *std.StringHashMapUnmanaged(BufferId),
    graph_op: Op,
    kernel_id_counter: *u32,
    last_was_residual: *bool,
    last_was_attn_or_ffn: *bool,
) !void {
    // ShortConv follows the same buffer pattern as attention/mlp/mamba
    last_was_residual.* = false;
    last_was_attn_or_ffn.* = true;

    var input_buffer: BufferId = .norm_out;
    if (graph_op.inputs.len > 0) {
        switch (graph_op.inputs[0]) {
            .tensor => |tensor_name| {
                if (tensor_to_buffer.get(tensor_name)) |buffer_id| input_buffer = buffer_id;
            },
            .scalar => {},
        }
    }

    try layer_ops.append(allocator, .{ .kernel = .{
        .id = kernel_id_counter.*,
        .in = input_buffer,
        .out = .branch_out,
        .debug_type = .shortconv,
    } });
    log.trace("load", "Emitting kernel {d} for {s}", .{ kernel_id_counter.*, @tagName(graph_op.op_type) }, @src());
    kernel_id_counter.* += 1;

    if (graph_op.outputs.len > 0) {
        try tensor_to_buffer.put(allocator, graph_op.outputs[0], .branch_out);
    }
}

fn compileAdd(
    allocator: Allocator,
    layer_ops: *std.ArrayListUnmanaged(LayerOp),
    tensor_to_buffer: *std.StringHashMapUnmanaged(BufferId),
    scaled_tensors: *std.StringHashMapUnmanaged(void),
    planner: *BufferPlanner,
    graph_op: Op,
    last_was_residual: *bool,
    last_was_attn_or_ffn: *bool,
) !void {
    var scalar_value: ?f32 = null;
    var param_name: ?[]const u8 = null;
    var input_buffer_ids: [2]BufferId = undefined; // Safe: only [0..input_buffer_count] read
    var input_buffer_count: u8 = 0;
    var has_residual = false;
    var branch_buffer: BufferId = .branch_out;
    var branch_tensor_name: ?[]const u8 = null;

    for (graph_op.inputs) |input| {
        switch (input) {
            .scalar => |s| scalar_value = s,
            .tensor => |tensor_name| {
                if (tensor_to_buffer.get(tensor_name)) |buffer_id| {
                    if (buffer_id == .residual or std.mem.eql(u8, tensor_name, "x")) {
                        has_residual = true;
                    } else if (input_buffer_count < 2) {
                        input_buffer_ids[input_buffer_count] = buffer_id;
                        input_buffer_count += 1;
                        branch_tensor_name = tensor_name;
                    }
                } else {
                    param_name = tensor_name;
                }
            },
        }
    }

    if (has_residual) {
        if (input_buffer_count > 0) branch_buffer = input_buffer_ids[0];
        last_was_residual.* = true;
        last_was_attn_or_ffn.* = false;
        const scale: ResidualScale = if (branch_tensor_name != null and scaled_tensors.contains(branch_tensor_name.?))
            .one
        else
            .residual_multiplier;
        try layer_ops.append(allocator, .{ .add = .{ .branch = branch_buffer, .scale = scale } });

        if (graph_op.outputs.len > 0) {
            try tensor_to_buffer.put(allocator, graph_op.outputs[0], .residual);
        }
    } else if (scalar_value != null and param_name != null) {
        const out_buf = if (graph_op.outputs.len > 0) try planner.outputFor(graph_op.outputs[0]) else try planner.allocTemp();
        // Duplicate string so compiled program owns its own copy
        const param_name_dup = try allocator.dupe(u8, param_name.?);
        try layer_ops.append(allocator, .{ .add_param_scalar = .{ .out = out_buf, .param_name = param_name_dup, .scalar = scalar_value.? } });
        if (graph_op.outputs.len > 0) {
            try tensor_to_buffer.put(allocator, graph_op.outputs[0], out_buf);
        }
    } else if (scalar_value != null and input_buffer_count == 1) {
        const out_buf = if (graph_op.outputs.len > 0) try planner.outputFor(graph_op.outputs[0]) else try planner.allocTemp();
        try layer_ops.append(allocator, .{ .add_scalar = .{ .in = input_buffer_ids[0], .out = out_buf, .scalar = scalar_value.? } });
        if (graph_op.outputs.len > 0) {
            try tensor_to_buffer.put(allocator, graph_op.outputs[0], out_buf);
        }
    } else if (param_name != null and input_buffer_count == 1) {
        const out_buf = if (graph_op.outputs.len > 0) try planner.outputFor(graph_op.outputs[0]) else try planner.allocTemp();
        // Duplicate string so compiled program owns its own copy
        const param_name_dup = try allocator.dupe(u8, param_name.?);
        try layer_ops.append(allocator, .{ .add_param = .{ .in = input_buffer_ids[0], .out = out_buf, .param_name = param_name_dup } });
        if (graph_op.outputs.len > 0) {
            try tensor_to_buffer.put(allocator, graph_op.outputs[0], out_buf);
        }
    } else if (input_buffer_count == 2) {
        // Two non-residual tensor add: this is a post-norm residual connection
        // (e.g., BERT: norm_out + branch_out → new residual). Write to residual
        // so subsequent ops see this as the updated residual stream.
        try layer_ops.append(allocator, .{ .add_tensor = .{ .in_a = input_buffer_ids[0], .in_b = input_buffer_ids[1], .out = .residual } });
        last_was_residual.* = true;
        last_was_attn_or_ffn.* = false;
        if (graph_op.outputs.len > 0) {
            try tensor_to_buffer.put(allocator, graph_op.outputs[0], .residual);
        }
    }
}

fn compileLinear(
    allocator: Allocator,
    layer_ops: *std.ArrayListUnmanaged(LayerOp),
    tensor_to_buffer: *std.StringHashMapUnmanaged(BufferId),
    graph_op: Op,
) !void {
    const weight_name_src = graph_op.name orelse "_linear";
    // Duplicate the string so the compiled program owns its own copy
    const weight_name = try allocator.dupe(u8, weight_name_src);
    errdefer allocator.free(weight_name);

    var input_buffer: BufferId = .norm_out;
    if (graph_op.inputs.len > 0) {
        switch (graph_op.inputs[0]) {
            .tensor => |tensor_name| {
                if (tensor_to_buffer.get(tensor_name)) |buffer_id| input_buffer = buffer_id;
            },
            .scalar => {},
        }
    }

    try layer_ops.append(allocator, .{ .linear = .{
        .in = input_buffer,
        .out = linearOutputBuffer(weight_name_src),
        .weight_name = weight_name,
    } });

    if (graph_op.outputs.len > 0) {
        try tensor_to_buffer.put(allocator, graph_op.outputs[0], linearOutputBuffer(weight_name_src));
    }
}

fn compileSplit(
    allocator: Allocator,
    layer_ops: *std.ArrayListUnmanaged(LayerOp),
    tensor_to_buffer: *std.StringHashMapUnmanaged(BufferId),
    graph_op: Op,
) !void {
    const num_outputs: u8 = if (graph_op.num_outputs > 0)
        @intCast(graph_op.num_outputs)
    else if (graph_op.split_sizes.len > 0)
        @intCast(graph_op.split_sizes.len)
    else
        3;

    const split_sizes_usize: []const usize = blk: {
        if (graph_op.split_sizes.len == 0) break :blk &[_]usize{};
        const sizes = try allocator.alloc(usize, graph_op.split_sizes.len);
        for (graph_op.split_sizes, 0..) |s, size_idx| {
            sizes[size_idx] = @intCast(s);
        }
        break :blk sizes;
    };

    try layer_ops.append(allocator, .{ .split = .{
        .in = .branch_out,
        .out_start = .tmp3,
        .num_outputs = num_outputs,
        .dim = @intCast(graph_op.dim),
        .split_sizes = split_sizes_usize,
    } });

    const out_buffers = [_]BufferId{ .tmp3, .tmp4, .tmp5, .tmp6, .tmp7 };
    for (graph_op.outputs, 0..) |out_name, out_idx| {
        if (out_idx < out_buffers.len) {
            try tensor_to_buffer.put(allocator, out_name, out_buffers[out_idx]);
        }
    }
}

fn compileActivation(
    allocator: Allocator,
    layer_ops: *std.ArrayListUnmanaged(LayerOp),
    tensor_to_buffer: *std.StringHashMapUnmanaged(BufferId),
    graph_op: Op,
    act_type: enum { silu, gelu },
) !void {
    var input_buffer: BufferId = .tmp3;
    if (graph_op.inputs.len > 0) {
        switch (graph_op.inputs[0]) {
            .tensor => |tensor_name| {
                if (tensor_to_buffer.get(tensor_name)) |buffer_id| input_buffer = buffer_id;
            },
            .scalar => {},
        }
    }

    switch (act_type) {
        .silu => try layer_ops.append(allocator, .{ .silu = .{ .in = input_buffer, .out = input_buffer } }),
        .gelu => try layer_ops.append(allocator, .{ .gelu = .{ .in = input_buffer, .out = input_buffer } }),
    }

    if (graph_op.outputs.len > 0) {
        try tensor_to_buffer.put(allocator, graph_op.outputs[0], input_buffer);
    }
}

fn compileMul(
    allocator: Allocator,
    layer_ops: *std.ArrayListUnmanaged(LayerOp),
    tensor_to_buffer: *std.StringHashMapUnmanaged(BufferId),
    scaled_tensors: *std.StringHashMapUnmanaged(void),
    planner: *BufferPlanner,
    graph_op: Op,
) !void {
    var scalar_value: ?f32 = null;
    var param_name: ?[]const u8 = null;
    var input_buffer_ids: [2]BufferId = undefined; // Safe: only [0..input_buffer_count] read
    var input_buffer_count: u8 = 0;

    for (graph_op.inputs) |input| {
        switch (input) {
            .scalar => |s| scalar_value = s,
            .tensor => |tensor_name| {
                if (tensor_to_buffer.get(tensor_name)) |buffer_id| {
                    if (input_buffer_count < 2) {
                        input_buffer_ids[input_buffer_count] = buffer_id;
                        input_buffer_count += 1;
                    }
                } else {
                    param_name = tensor_name;
                }
            },
        }
    }

    const out_buf = if (graph_op.outputs.len > 0) try planner.outputFor(graph_op.outputs[0]) else try planner.allocTemp();

    if (scalar_value != null and input_buffer_count == 1) {
        try layer_ops.append(allocator, .{ .mul_scalar = .{ .in = input_buffer_ids[0], .out = out_buf, .scalar = scalar_value.? } });
        if (graph_op.outputs.len > 0) {
            try scaled_tensors.put(allocator, graph_op.outputs[0], {});
        }
    } else if (param_name != null and input_buffer_count == 1) {
        // Duplicate string so compiled program owns its own copy
        const param_name_dup = try allocator.dupe(u8, param_name.?);
        try layer_ops.append(allocator, .{ .mul_param = .{ .in = input_buffer_ids[0], .out = out_buf, .param_name = param_name_dup } });
    } else if (input_buffer_count == 2) {
        if (input_buffer_ids[0] == input_buffer_ids[1]) return error.InvalidMulAlias;
        try layer_ops.append(allocator, .{ .mul = .{ .in = input_buffer_ids[0], .other = input_buffer_ids[1], .out = out_buf } });
    }
    if (graph_op.outputs.len > 0) {
        try tensor_to_buffer.put(allocator, graph_op.outputs[0], out_buf);
    }
}

fn compileMean(
    allocator: Allocator,
    layer_ops: *std.ArrayListUnmanaged(LayerOp),
    tensor_to_buffer: *std.StringHashMapUnmanaged(BufferId),
    planner: *BufferPlanner,
    graph_op: Op,
) !void {
    var input_buffer: BufferId = .branch_out;
    if (graph_op.inputs.len > 0) {
        switch (graph_op.inputs[0]) {
            .tensor => |tensor_name| {
                if (tensor_to_buffer.get(tensor_name)) |buffer_id| input_buffer = buffer_id;
            },
            .scalar => {},
        }
    }
    const out_buf = if (graph_op.outputs.len > 0) try planner.outputFor(graph_op.outputs[0]) else try planner.allocTemp();
    try layer_ops.append(allocator, .{ .mean = .{ .in = input_buffer, .out = out_buf, .dim = @intCast(graph_op.dim), .keepdim = graph_op.keepdim } });
    if (graph_op.outputs.len > 0) {
        try tensor_to_buffer.put(allocator, graph_op.outputs[0], out_buf);
    }
}

fn compilePow(
    allocator: Allocator,
    layer_ops: *std.ArrayListUnmanaged(LayerOp),
    tensor_to_buffer: *std.StringHashMapUnmanaged(BufferId),
    planner: *BufferPlanner,
    graph_op: Op,
) !void {
    var input_buffer: BufferId = .branch_out;
    if (graph_op.inputs.len > 0) {
        switch (graph_op.inputs[0]) {
            .tensor => |tensor_name| {
                if (tensor_to_buffer.get(tensor_name)) |buffer_id| input_buffer = buffer_id;
            },
            .scalar => {},
        }
    }
    const out_buf = if (graph_op.outputs.len > 0) try planner.outputFor(graph_op.outputs[0]) else try planner.allocTemp();
    try layer_ops.append(allocator, .{ .pow = .{ .in = input_buffer, .out = out_buf, .exponent = graph_op.exponent } });
    if (graph_op.outputs.len > 0) {
        try tensor_to_buffer.put(allocator, graph_op.outputs[0], out_buf);
    }
}

fn compileRsqrt(
    allocator: Allocator,
    layer_ops: *std.ArrayListUnmanaged(LayerOp),
    tensor_to_buffer: *std.StringHashMapUnmanaged(BufferId),
    planner: *BufferPlanner,
    graph_op: Op,
) !void {
    var input_buffer: BufferId = .branch_out;
    if (graph_op.inputs.len > 0) {
        switch (graph_op.inputs[0]) {
            .tensor => |tensor_name| {
                if (tensor_to_buffer.get(tensor_name)) |buffer_id| input_buffer = buffer_id;
            },
            .scalar => {},
        }
    }
    const out_buf = if (graph_op.outputs.len > 0) try planner.outputFor(graph_op.outputs[0]) else try planner.allocTemp();
    try layer_ops.append(allocator, .{ .rsqrt = .{ .in = input_buffer, .out = out_buf } });
    if (graph_op.outputs.len > 0) {
        try tensor_to_buffer.put(allocator, graph_op.outputs[0], out_buf);
    }
}

fn compileReshape(
    allocator: Allocator,
    layer_ops: *std.ArrayListUnmanaged(LayerOp),
    tensor_to_buffer: *std.StringHashMapUnmanaged(BufferId),
    graph_op: Op,
) !void {
    var input_buffer: BufferId = .branch_out;
    if (graph_op.inputs.len > 0) {
        switch (graph_op.inputs[0]) {
            .tensor => |tensor_name| {
                if (tensor_to_buffer.get(tensor_name)) |buffer_id| input_buffer = buffer_id;
            },
            .scalar => {},
        }
    }
    // Duplicate shape array so compiled program owns its own copy
    const shape = if (graph_op.shape.len > 0) try allocator.dupe(i32, graph_op.shape) else &[_]i32{};
    try layer_ops.append(allocator, .{ .reshape = .{ .in = input_buffer, .out = input_buffer, .shape = shape } });

    if (graph_op.outputs.len > 0) {
        try tensor_to_buffer.put(allocator, graph_op.outputs[0], input_buffer);
    }
}

fn compileTranspose(
    allocator: Allocator,
    layer_ops: *std.ArrayListUnmanaged(LayerOp),
    tensor_to_buffer: *std.StringHashMapUnmanaged(BufferId),
    graph_op: Op,
) !void {
    var input_buffer: BufferId = .tmp4;
    if (graph_op.inputs.len > 0) {
        switch (graph_op.inputs[0]) {
            .tensor => |tensor_name| {
                if (tensor_to_buffer.get(tensor_name)) |buffer_id| input_buffer = buffer_id;
            },
            .scalar => {},
        }
    }
    const dim0: i8 = if (graph_op.dim0 != -1) @intCast(graph_op.dim0) else @intCast(graph_op.dim);
    const dim1: i8 = if (graph_op.dim1 != -1) @intCast(graph_op.dim1) else -1;
    try layer_ops.append(allocator, .{ .transpose = .{ .in = input_buffer, .out = input_buffer, .dim0 = dim0, .dim1 = dim1 } });

    if (graph_op.outputs.len > 0) {
        try tensor_to_buffer.put(allocator, graph_op.outputs[0], input_buffer);
    }
}

fn compileRope(
    allocator: Allocator,
    layer_ops: *std.ArrayListUnmanaged(LayerOp),
    tensor_to_buffer: *std.StringHashMapUnmanaged(BufferId),
    graph_op: Op,
) !void {
    var input_buffer: BufferId = .tmp3;
    if (graph_op.inputs.len > 0) {
        switch (graph_op.inputs[0]) {
            .tensor => |tensor_name| {
                if (tensor_to_buffer.get(tensor_name)) |buffer_id| input_buffer = buffer_id;
            },
            .scalar => {},
        }
    }
    try layer_ops.append(allocator, .{ .rope = .{ .in = input_buffer, .out = input_buffer } });

    if (graph_op.outputs.len > 0) {
        try tensor_to_buffer.put(allocator, graph_op.outputs[0], input_buffer);
    }
}

fn compileSdpa(
    allocator: Allocator,
    layer_ops: *std.ArrayListUnmanaged(LayerOp),
    tensor_to_buffer: *std.StringHashMapUnmanaged(BufferId),
    graph_op: Op,
) !void {
    // SDPA expects 3 inputs: Q, K, V (all 4D: [batch, heads, seq, head_dim])
    // Output is also 4D: [batch, heads, seq, head_dim]

    // Get input buffers for Q, K, V
    var q_buffer: BufferId = .tmp3;
    var k_buffer: BufferId = .tmp4;
    var v_buffer: BufferId = .tmp5;

    if (graph_op.inputs.len >= 3) {
        switch (graph_op.inputs[0]) {
            .tensor => |tensor_name| {
                if (tensor_to_buffer.get(tensor_name)) |buffer_id| q_buffer = buffer_id;
            },
            .scalar => {},
        }
        switch (graph_op.inputs[1]) {
            .tensor => |tensor_name| {
                if (tensor_to_buffer.get(tensor_name)) |buffer_id| k_buffer = buffer_id;
            },
            .scalar => {},
        }
        switch (graph_op.inputs[2]) {
            .tensor => |tensor_name| {
                if (tensor_to_buffer.get(tensor_name)) |buffer_id| v_buffer = buffer_id;
            },
            .scalar => {},
        }
    }

    // Output goes to tmp6 by default (or we could use a planner)
    const out_buf: BufferId = .tmp6;

    try layer_ops.append(allocator, .{
        .sdpa = .{
            .q = q_buffer,
            .k = k_buffer,
            .v = v_buffer,
            .out = out_buf,
            .is_causal = graph_op.is_causal,
            .scale = graph_op.sdpa_scale, // From muP attention_multiplier (null = 1/sqrt(head_dim))
        },
    });

    if (graph_op.outputs.len > 0) {
        try tensor_to_buffer.put(allocator, graph_op.outputs[0], out_buf);
    }
}

// =============================================================================
// Unit Tests
// =============================================================================

test "compile empty graph returns empty program" {
    const allocator = std.testing.allocator;

    const graph_ops: []const Op = &.{};
    const program = try compile(allocator, graph_ops);
    defer allocator.free(program);

    try std.testing.expectEqual(@as(usize, 0), program.len);
}

test "compile single norm op" {
    const allocator = std.testing.allocator;

    const graph_ops = [_]Op{
        .{ .op_type = .norm, .weight_offset = 0.0 },
    };
    const program = try compile(allocator, &graph_ops);
    defer allocator.free(program);

    try std.testing.expectEqual(@as(usize, 1), program.len);
    try std.testing.expect(program[0] == .kernel);
    try std.testing.expectEqual(@as(u32, 0), program[0].kernel.id);
    try std.testing.expectEqual(BufferId.residual, program[0].kernel.in);
    // A lone norm naturally outputs to norm_out. The block executor handles
    // copying to residual via finalOutputBuffer() after program execution.
    try std.testing.expectEqual(BufferId.norm_out, program[0].kernel.out);
    try std.testing.expectEqual(types.OpType.norm, program[0].kernel.debug_type);

    // finalOutputBuffer should report norm_out for this program.
    try std.testing.expectEqual(BufferId.norm_out, finalOutputBuffer(program));
}

test "compile norm followed by attention" {
    const allocator = std.testing.allocator;

    const graph_ops = [_]Op{
        .{ .op_type = .norm },
        .{ .op_type = .multihead_attention },
    };
    const program = try compile(allocator, &graph_ops);
    defer allocator.free(program);

    try std.testing.expectEqual(@as(usize, 2), program.len);
    try std.testing.expect(program[0] == .kernel);
    try std.testing.expect(program[1] == .kernel);
    try std.testing.expectEqual(@as(u32, 0), program[0].kernel.id);
    try std.testing.expectEqual(@as(u32, 1), program[1].kernel.id);
    try std.testing.expectEqual(BufferId.norm_out, program[1].kernel.in);
    try std.testing.expectEqual(BufferId.branch_out, program[1].kernel.out);
    try std.testing.expectEqual(types.OpType.norm, program[0].kernel.debug_type);
    try std.testing.expectEqual(types.OpType.multihead_attention, program[1].kernel.debug_type);
}

test "compile norm, attention, add, norm, ffn, add sequence" {
    const allocator = std.testing.allocator;

    const x_input = try allocator.dupe(u8, "x");
    defer allocator.free(x_input);
    const attn_out = try allocator.dupe(u8, "attn_out");
    defer allocator.free(attn_out);
    const residual1 = try allocator.dupe(u8, "residual1");
    defer allocator.free(residual1);
    const ffn_out = try allocator.dupe(u8, "ffn_out");
    defer allocator.free(ffn_out);

    const graph_ops = [_]Op{
        .{ .op_type = .norm },
        .{ .op_type = .multihead_attention, .outputs = &[_][]const u8{attn_out} },
        .{ .op_type = .add, .inputs = &[_]OpInput{ .{ .tensor = x_input }, .{ .tensor = attn_out } }, .outputs = &[_][]const u8{residual1} },
        .{ .op_type = .norm },
        .{ .op_type = .mlp, .outputs = &[_][]const u8{ffn_out} },
        .{ .op_type = .add, .inputs = &[_]OpInput{ .{ .tensor = residual1 }, .{ .tensor = ffn_out } } },
    };
    const program = try compile(allocator, &graph_ops);
    defer allocator.free(program);

    try std.testing.expectEqual(@as(usize, 6), program.len);
    try std.testing.expect(program[0] == .kernel);
    try std.testing.expect(program[1] == .kernel);
    try std.testing.expect(program[2] == .add);
    try std.testing.expect(program[3] == .kernel);
    try std.testing.expect(program[4] == .kernel);
    try std.testing.expect(program[5] == .add);
    try std.testing.expectEqual(types.OpType.norm, program[0].kernel.debug_type);
    try std.testing.expectEqual(types.OpType.multihead_attention, program[1].kernel.debug_type);
    try std.testing.expectEqual(types.OpType.norm, program[3].kernel.debug_type);
    try std.testing.expectEqual(types.OpType.mlp, program[4].kernel.debug_type);
}

test "compile linear op with weight name" {
    const allocator = std.testing.allocator;

    const name = try allocator.dupe(u8, "q_proj");
    defer allocator.free(name);

    const graph_ops = [_]Op{
        .{ .op_type = .linear, .name = name },
    };
    const program = try compile(allocator, &graph_ops);
    defer allocator.free(program);

    try std.testing.expectEqual(@as(usize, 1), program.len);
    try std.testing.expect(program[0] == .linear);
    try std.testing.expectEqualStrings("q_proj", program[0].linear.weight_name);
    try std.testing.expectEqual(BufferId.tmp3, program[0].linear.out);
}

test "compile split op" {
    const allocator = std.testing.allocator;

    const graph_ops = [_]Op{
        .{ .op_type = .split, .num_outputs = 3, .dim = -1 },
    };
    const program = try compile(allocator, &graph_ops);
    defer allocator.free(program);

    try std.testing.expectEqual(@as(usize, 1), program.len);
    try std.testing.expect(program[0] == .split);
    try std.testing.expectEqual(BufferId.branch_out, program[0].split.in);
    try std.testing.expectEqual(BufferId.tmp3, program[0].split.out_start);
    try std.testing.expectEqual(@as(u8, 3), program[0].split.num_outputs);
}

test "compile activation ops" {
    const allocator = std.testing.allocator;

    const graph_ops = [_]Op{
        .{ .op_type = .silu },
        .{ .op_type = .gelu },
    };
    const program = try compile(allocator, &graph_ops);
    defer allocator.free(program);

    try std.testing.expectEqual(@as(usize, 2), program.len);
    try std.testing.expect(program[0] == .silu);
    try std.testing.expect(program[1] == .gelu);
}

test "compile rope op" {
    const allocator = std.testing.allocator;

    const graph_ops = [_]Op{
        .{ .op_type = .rope },
    };
    const program = try compile(allocator, &graph_ops);
    defer allocator.free(program);

    try std.testing.expectEqual(@as(usize, 1), program.len);
    try std.testing.expect(program[0] == .rope);
    try std.testing.expectEqual(BufferId.tmp3, program[0].rope.in);
    try std.testing.expectEqual(BufferId.tmp3, program[0].rope.out);
}

test "compile matmul op" {
    const allocator = std.testing.allocator;

    const graph_ops = [_]Op{
        .{ .op_type = .matmul },
    };
    const program = try compile(allocator, &graph_ops);
    defer allocator.free(program);

    try std.testing.expectEqual(@as(usize, 1), program.len);
    try std.testing.expect(program[0] == .matmul);
    try std.testing.expectEqual(BufferId.tmp3, program[0].matmul.in_a);
    try std.testing.expectEqual(BufferId.tmp4, program[0].matmul.in_b);
    try std.testing.expectEqual(BufferId.tmp6, program[0].matmul.out);
}

test "compile softmax op" {
    const allocator = std.testing.allocator;

    const graph_ops = [_]Op{
        .{ .op_type = .softmax },
    };
    const program = try compile(allocator, &graph_ops);
    defer allocator.free(program);

    try std.testing.expectEqual(@as(usize, 1), program.len);
    try std.testing.expect(program[0] == .softmax);
    try std.testing.expectEqual(BufferId.tmp6, program[0].softmax.in);
    try std.testing.expectEqual(BufferId.tmp6, program[0].softmax.out);
}

test "compile triu op" {
    const allocator = std.testing.allocator;

    const graph_ops = [_]Op{
        .{ .op_type = .triu, .dim = 1 },
    };
    const program = try compile(allocator, &graph_ops);
    defer allocator.free(program);

    try std.testing.expectEqual(@as(usize, 1), program.len);
    try std.testing.expect(program[0] == .triu);
    try std.testing.expectEqual(BufferId.tmp6, program[0].triu.in);
    try std.testing.expectEqual(BufferId.tmp6, program[0].triu.out);
    try std.testing.expectEqual(@as(i32, 1), program[0].triu.diagonal);
}

test "compile embedding op is skipped" {
    const allocator = std.testing.allocator;

    const graph_ops = [_]Op{
        .{ .op_type = .embedding },
    };
    const program = try compile(allocator, &graph_ops);
    defer allocator.free(program);

    try std.testing.expectEqual(@as(usize, 0), program.len);
}

test "usesPrimitiveOps returns false for high-level ops" {
    const program = [_]LayerOp{
        .{ .kernel = .{ .id = 0, .in = .residual, .out = .norm_out, .debug_type = .norm } },
        .{ .kernel = .{ .id = 1, .in = .norm_out, .out = .branch_out, .debug_type = .multihead_attention } },
        .{ .kernel = .{ .id = 2, .in = .norm_out, .out = .branch_out, .debug_type = .mlp } },
        .{ .add = .{ .branch = .branch_out, .scale = .residual_multiplier } },
    };

    try std.testing.expect(!usesPrimitiveOps(&program));
}

test "usesPrimitiveOps returns true for primitive ops" {
    const program = [_]LayerOp{
        .{ .kernel = .{ .id = 0, .in = .residual, .out = .norm_out, .debug_type = .norm } },
        .{ .linear = .{ .in = .norm_out, .out = .tmp3, .weight_name = "q_proj" } },
    };

    try std.testing.expect(usesPrimitiveOps(&program));
}

test "usesPrimitiveOps returns true for rope" {
    const program = [_]LayerOp{
        .{ .rope = .{ .in = .tmp3, .out = .tmp3 } },
    };

    try std.testing.expect(usesPrimitiveOps(&program));
}

test "usesPrimitiveOps returns true for matmul" {
    const program = [_]LayerOp{
        .{ .matmul = .{ .in_a = .tmp3, .in_b = .tmp4, .out = .tmp6 } },
    };

    try std.testing.expect(usesPrimitiveOps(&program));
}

test "compile norm with weight offset" {
    const allocator = std.testing.allocator;

    const graph_ops = [_]Op{
        .{ .op_type = .norm, .weight_offset = 1.0 },
    };
    const program = try compile(allocator, &graph_ops);
    defer allocator.free(program);

    try std.testing.expectEqual(@as(usize, 1), program.len);
    try std.testing.expect(program[0] == .kernel);
    try std.testing.expectEqual(@as(u32, 0), program[0].kernel.id);
}

test "compile multiple norms increment norm slot" {
    const allocator = std.testing.allocator;

    const graph_ops = [_]Op{
        .{ .op_type = .norm },
        .{ .op_type = .norm },
        .{ .op_type = .norm },
    };
    const program = try compile(allocator, &graph_ops);
    defer allocator.free(program);

    try std.testing.expectEqual(@as(usize, 3), program.len);
    try std.testing.expect(program[0] == .kernel);
    try std.testing.expect(program[1] == .kernel);
    try std.testing.expect(program[2] == .kernel);
    try std.testing.expectEqual(@as(u32, 0), program[0].kernel.id);
    try std.testing.expectEqual(@as(u32, 1), program[1].kernel.id);
    try std.testing.expectEqual(@as(u32, 2), program[2].kernel.id);
}

test "compile sdpa op" {
    const allocator = std.testing.allocator;

    const graph_ops = [_]Op{
        .{ .op_type = .scaled_dot_product_attention, .is_causal = true },
    };
    const program = try compile(allocator, &graph_ops);
    defer allocator.free(program);

    try std.testing.expectEqual(@as(usize, 1), program.len);
    try std.testing.expect(program[0] == .sdpa);
    try std.testing.expectEqual(BufferId.tmp3, program[0].sdpa.q);
    try std.testing.expectEqual(BufferId.tmp4, program[0].sdpa.k);
    try std.testing.expectEqual(BufferId.tmp5, program[0].sdpa.v);
    try std.testing.expectEqual(BufferId.tmp6, program[0].sdpa.out);
    try std.testing.expect(program[0].sdpa.is_causal);
}

test "finalOutputBuffer returns residual for empty program" {
    const empty: []const LayerOp = &.{};
    try std.testing.expectEqual(BufferId.residual, finalOutputBuffer(empty));
}

test "finalOutputBuffer returns kernel output buffer" {
    // Pre-norm: last op is a residual add → kernel.out isn't the last op.
    // Post-norm: last op is a norm kernel writing to norm_out.
    const program = [_]LayerOp{
        .{ .kernel = .{ .id = 0, .in = .residual, .out = .norm_out, .debug_type = .norm } },
    };
    try std.testing.expectEqual(BufferId.norm_out, finalOutputBuffer(&program));
}

test "finalOutputBuffer returns residual for add" {
    const program = [_]LayerOp{
        .{ .kernel = .{ .id = 0, .in = .residual, .out = .norm_out, .debug_type = .norm } },
        .{ .kernel = .{ .id = 1, .in = .norm_out, .out = .branch_out, .debug_type = .multihead_attention } },
        .{ .add = .{ .branch = .branch_out, .scale = .residual_multiplier } },
    };
    try std.testing.expectEqual(BufferId.residual, finalOutputBuffer(&program));
}

test "finalOutputBuffer returns add_tensor output buffer" {
    const program = [_]LayerOp{
        .{ .add_tensor = .{ .in_a = .norm_out, .in_b = .branch_out, .out = .residual } },
    };
    try std.testing.expectEqual(BufferId.residual, finalOutputBuffer(&program));

    // add_tensor writing to a non-residual buffer
    const program2 = [_]LayerOp{
        .{ .add_tensor = .{ .in_a = .tmp3, .in_b = .tmp4, .out = .tmp6 } },
    };
    try std.testing.expectEqual(BufferId.tmp6, finalOutputBuffer(&program2));
}

test "finalOutputBuffer returns residual for primitive ops" {
    // Primitive ops (linear, silu, etc.) hit the else branch → residual.
    const program_linear = [_]LayerOp{
        .{ .linear = .{ .in = .norm_out, .out = .tmp3, .weight_name = "q_proj" } },
    };
    try std.testing.expectEqual(BufferId.residual, finalOutputBuffer(&program_linear));

    const program_silu = [_]LayerOp{
        .{ .silu = .{ .in = .tmp3, .out = .tmp3 } },
    };
    try std.testing.expectEqual(BufferId.residual, finalOutputBuffer(&program_silu));
}

test "compile BERT post-norm: attn, add, norm, mlp, add, norm" {
    const allocator = std.testing.allocator;

    // BERT post-norm pattern:
    //   attn(x) → _t0
    //   add(x, _t0) → _t1       (residual connection)
    //   norm(_t1) → _t2          (post-attn norm)
    //   mlp(_t2) → _t3
    //   add(_t2, _t3) → _t4      (non-residual add: norm_out + branch_out)
    //   norm(_t4) → _t5          (post-FFN norm, terminal → residual)

    const x = try allocator.dupe(u8, "x");
    defer allocator.free(x);
    const t0 = try allocator.dupe(u8, "_t0");
    defer allocator.free(t0);
    const t1 = try allocator.dupe(u8, "_t1");
    defer allocator.free(t1);
    const t2 = try allocator.dupe(u8, "_t2");
    defer allocator.free(t2);
    const t3 = try allocator.dupe(u8, "_t3");
    defer allocator.free(t3);
    const t4 = try allocator.dupe(u8, "_t4");
    defer allocator.free(t4);
    const t5 = try allocator.dupe(u8, "_t5");
    defer allocator.free(t5);

    const graph_ops = [_]Op{
        .{ .op_type = .multihead_attention, .inputs = &[_]OpInput{.{ .tensor = x }}, .outputs = &[_][]const u8{t0} },
        .{ .op_type = .add, .inputs = &[_]OpInput{ .{ .tensor = x }, .{ .tensor = t0 } }, .outputs = &[_][]const u8{t1} },
        .{ .op_type = .norm, .inputs = &[_]OpInput{.{ .tensor = t1 }}, .outputs = &[_][]const u8{t2} },
        .{ .op_type = .mlp, .inputs = &[_]OpInput{.{ .tensor = t2 }}, .outputs = &[_][]const u8{t3} },
        .{ .op_type = .add, .inputs = &[_]OpInput{ .{ .tensor = t2 }, .{ .tensor = t3 } }, .outputs = &[_][]const u8{t4} },
        .{ .op_type = .norm, .inputs = &[_]OpInput{.{ .tensor = t4 }}, .outputs = &[_][]const u8{t5} },
    };
    const program = try compile(allocator, &graph_ops);
    defer allocator.free(program);

    // Expected: 6 ops (kernel, add, kernel, kernel, add_tensor, kernel)
    try std.testing.expectEqual(@as(usize, 6), program.len);

    // Op 0: attention kernel reads from residual (x), writes to branch_out
    try std.testing.expect(program[0] == .kernel);
    try std.testing.expectEqual(types.OpType.multihead_attention, program[0].kernel.debug_type);
    try std.testing.expectEqual(@as(u32, 0), program[0].kernel.id);
    try std.testing.expectEqual(BufferId.residual, program[0].kernel.in);
    try std.testing.expectEqual(BufferId.branch_out, program[0].kernel.out);

    // Op 1: residual add (x + _t0)
    try std.testing.expect(program[1] == .add);
    try std.testing.expectEqual(BufferId.branch_out, program[1].add.branch);

    // Op 2: post-attn norm reads from residual (updated by add), writes to norm_out
    try std.testing.expect(program[2] == .kernel);
    try std.testing.expectEqual(types.OpType.norm, program[2].kernel.debug_type);
    try std.testing.expectEqual(@as(u32, 1), program[2].kernel.id);
    try std.testing.expectEqual(BufferId.residual, program[2].kernel.in);
    try std.testing.expectEqual(BufferId.norm_out, program[2].kernel.out);

    // Op 3: mlp reads from norm_out, writes to branch_out
    try std.testing.expect(program[3] == .kernel);
    try std.testing.expectEqual(types.OpType.mlp, program[3].kernel.debug_type);
    try std.testing.expectEqual(@as(u32, 2), program[3].kernel.id);
    try std.testing.expectEqual(BufferId.norm_out, program[3].kernel.in);
    try std.testing.expectEqual(BufferId.branch_out, program[3].kernel.out);

    // Op 4: non-residual add (_t2 + _t3 = norm_out + branch_out → residual)
    try std.testing.expect(program[4] == .add_tensor);
    try std.testing.expectEqual(BufferId.norm_out, program[4].add_tensor.in_a);
    try std.testing.expectEqual(BufferId.branch_out, program[4].add_tensor.in_b);
    try std.testing.expectEqual(BufferId.residual, program[4].add_tensor.out);

    // Op 5: terminal post-FFN norm reads from residual, writes to norm_out (natural).
    // The block executor copies norm_out → residual via finalOutputBuffer() after
    // program execution. No in-place aliasing (which would corrupt element 0).
    try std.testing.expect(program[5] == .kernel);
    try std.testing.expectEqual(types.OpType.norm, program[5].kernel.debug_type);
    try std.testing.expectEqual(@as(u32, 3), program[5].kernel.id);
    try std.testing.expectEqual(BufferId.residual, program[5].kernel.in);
    try std.testing.expectEqual(BufferId.norm_out, program[5].kernel.out);

    // finalOutputBuffer should report norm_out for this post-norm program.
    try std.testing.expectEqual(BufferId.norm_out, finalOutputBuffer(program));
}
