//! Shared vision program adapter functions, dispatch, and adapter table.
//!
//! Adapter functions are non-generic: they access backend-specific
//! VisionRuntime methods through `VisionRuntimeVTable`, which is
//! generated at comptime by `VTableFor(VRT)`.  This lets all backends
//! share a single `adapter_table` registered in the main per-backend
//! adapter tables (CPU `adapter_table`, Metal
//! `layer_program_adapter_table`).

const std = @import("std");
const runtime_contract = @import("runtime_contract/root.zig");
const image_mod = @import("../image/root.zig");

/// Per-image state machine tracking vision program progress.
pub const AdapterState = struct {
    saw_patch_embed: bool = false,
    saw_spatial_merge: bool = false,
    merged_embeddings: ?[]f32 = null,
};

/// Result of encoding a single image through the vision pipeline.
pub const EncodedSingleImage = struct {
    merged_embeddings: []f32,
    deepstack_layer_embeddings: []const []const f32,

    pub fn deinit(self: *EncodedSingleImage, allocator: std.mem.Allocator) void {
        if (self.merged_embeddings.len > 0) allocator.free(self.merged_embeddings);
        if (self.deepstack_layer_embeddings.len > 0) {
            for (self.deepstack_layer_embeddings) |layer_embed| {
                if (layer_embed.len > 0) allocator.free(layer_embed);
            }
            allocator.free(self.deepstack_layer_embeddings);
        }
        self.* = .{
            .merged_embeddings = &.{},
            .deepstack_layer_embeddings = &.{},
        };
    }
};

/// Type-erased vtable for VisionRuntime methods used by adapters.
pub const VisionRuntimeVTable = struct {
    deepstackLayerToMergerIndex: *const fn (runtime_ptr: *anyopaque, layer_idx: usize) ?usize,
    runMerger: *const fn (runtime_ptr: *anyopaque, grid: image_mod.VisionGrid, hidden: []const f32) anyerror![]f32,
};

/// Generates a `VisionRuntimeVTable` for a concrete VisionRuntime type.
///
/// `VRT` must expose:
///   - method `deepstackLayerToMergerIndex(layer_idx: usize) ?usize`
///   - method `runMerger(grid: image_mod.VisionGrid, hidden: []const f32) ![]f32`
pub fn VTableFor(comptime VRT: type) type {
    return struct {
        pub const vtable = VisionRuntimeVTable{
            .deepstackLayerToMergerIndex = struct {
                fn call(runtime_ptr: *anyopaque, layer_idx: usize) ?usize {
                    const self: *VRT = @ptrCast(@alignCast(runtime_ptr));
                    return self.deepstackLayerToMergerIndex(layer_idx);
                }
            }.call,
            .runMerger = struct {
                fn call(runtime_ptr: *anyopaque, grid: image_mod.VisionGrid, hidden: []const f32) anyerror![]f32 {
                    const self: *VRT = @ptrCast(@alignCast(runtime_ptr));
                    return self.runMerger(grid, hidden);
                }
            }.call,
        };
    };
}

/// Vision opcodes that must be covered by any adapter table supporting
/// vision programs.
pub const required_opcodes = [_]runtime_contract.Opcode{
    .vision_patch_embed,
    .vision_deepstack_extract,
    .vision_spatial_merge,
    .vision_scatter,
};

/// Non-generic adapter table entries for vision opcodes.
pub const adapter_table: runtime_contract.AdapterTable = blk: {
    var table: runtime_contract.AdapterTable = [_]?runtime_contract.KernelAdapterFn{null} ** 256;
    table[@intFromEnum(runtime_contract.Opcode.vision_patch_embed)] = patchEmbedRuntimeAdapter;
    table[@intFromEnum(runtime_contract.Opcode.vision_deepstack_extract)] = deepstackExtractRuntimeAdapter;
    table[@intFromEnum(runtime_contract.Opcode.vision_spatial_merge)] = spatialMergeRuntimeAdapter;
    table[@intFromEnum(runtime_contract.Opcode.vision_scatter)] = scatterRuntimeAdapter;
    break :blk table;
};

/// Workspace context for vision program dispatch.
pub const ExecutionContext = struct {
    runtime_ptr: *anyopaque,
    vtable: *const VisionRuntimeVTable,
    compiled_plan: *const runtime_contract.CompiledPlan,
    op_index: usize,
    grid: image_mod.VisionGrid,
    merged_hidden_in: []const f32,
    deepstack_layer_embeddings: []const []const f32,
    state: *AdapterState,
};

/// Workspace context for staged scatter dispatch.
pub const ScatterExecutionContext = struct {
    compiled_plan: *const runtime_contract.CompiledPlan,
    hidden_states: []f32,
    seq_len: usize,
    d_model: usize,
    token_ids: []const u32,
    image_token_id: u32,
    embeddings: []const f32,
};

/// Execute a compiled vision program through the main adapter table.
pub fn runVisionProgram(
    runtime_ptr: *anyopaque,
    vtable: *const VisionRuntimeVTable,
    allocator: std.mem.Allocator,
    compiled_vision_plan: *const runtime_contract.CompiledPlan,
    dispatch_counters: *runtime_contract.DispatchCounters,
    dispatch_table: runtime_contract.AdapterTable,
    grid: image_mod.VisionGrid,
    merged_hidden_in: []const f32,
    deepstack_layer_embeddings: []const []const f32,
) !EncodedSingleImage {
    var state = AdapterState{};
    errdefer if (state.merged_embeddings) |embeds| allocator.free(embeds);
    var exec_state = ExecutionContext{
        .runtime_ptr = runtime_ptr,
        .vtable = vtable,
        .compiled_plan = compiled_vision_plan,
        .op_index = 0,
        .grid = grid,
        .merged_hidden_in = merged_hidden_in,
        .deepstack_layer_embeddings = deepstack_layer_embeddings,
        .state = &state,
    };
    var active_slots: [1]usize = .{0};
    const no_seq_lengths: [0]u32 = .{};
    const no_registers: [0]runtime_contract.TensorHandle = .{};
    const no_views: [0]runtime_contract.TensorViewDesc = .{};
    const no_state_blocks: [0]runtime_contract.StateBlockHandle = .{};
    for (compiled_vision_plan.plan.instructions, 0..) |insn, op_index| {
        exec_state.op_index = op_index;
        const adapter_fn = dispatch_table[@intFromEnum(insn.opcode)] orelse return error.InvalidVisionProgram;
        var rt_ctx = runtime_contract.ExecutionContext{
            .mode = .vision_encode,
            .active_slots = active_slots[0..],
            .sequence_lengths = no_seq_lengths[0..],
            .batch_size = 1,
            .dispatch_counters = dispatch_counters,
            .workspace = .{ .any = @ptrCast(&exec_state) },
        };
        runtime_contract.recordExecutionDispatch(&rt_ctx, insn.opcode);
        try adapter_fn(
            &rt_ctx,
            &insn,
            no_registers[0..],
            no_views[0..],
            no_state_blocks[0..],
            compiled_vision_plan.param_blocks,
        );
    }

    if (!state.saw_patch_embed or !state.saw_spatial_merge or state.merged_embeddings == null) {
        return error.InvalidVisionProgram;
    }

    return .{
        .merged_embeddings = state.merged_embeddings.?,
        .deepstack_layer_embeddings = deepstack_layer_embeddings,
    };
}

/// Execute a compiled scatter stage through the main adapter table.
pub fn runScatterProgram(
    compiled_scatter_plan: *const runtime_contract.CompiledPlan,
    dispatch_counters: *runtime_contract.DispatchCounters,
    dispatch_table: runtime_contract.AdapterTable,
    hidden_states: []f32,
    seq_len: usize,
    d_model: usize,
    token_ids: []const u32,
    image_token_id: u32,
    embeddings: []const f32,
) !void {
    if (token_ids.len != seq_len) return error.InvalidArgument;

    var scatter_ctx = ScatterExecutionContext{
        .compiled_plan = compiled_scatter_plan,
        .hidden_states = hidden_states,
        .seq_len = seq_len,
        .d_model = d_model,
        .token_ids = token_ids,
        .image_token_id = image_token_id,
        .embeddings = embeddings,
    };
    var active_slots: [1]usize = .{0};
    const no_seq_lengths: [0]u32 = .{};
    const no_registers: [0]runtime_contract.TensorHandle = .{};
    const no_views: [0]runtime_contract.TensorViewDesc = .{};
    const no_state_blocks: [0]runtime_contract.StateBlockHandle = .{};
    for (compiled_scatter_plan.plan.instructions) |insn| {
        if (insn.opcode != .vision_scatter) return error.InvalidVisionProgram;
        const adapter_fn = dispatch_table[@intFromEnum(insn.opcode)] orelse return error.InvalidVisionProgram;
        var rt_ctx = runtime_contract.ExecutionContext{
            .mode = .scatter,
            .active_slots = active_slots[0..],
            .sequence_lengths = no_seq_lengths[0..],
            .batch_size = 1,
            .dispatch_counters = dispatch_counters,
            .workspace = .{ .any = @ptrCast(&scatter_ctx) },
        };
        runtime_contract.recordExecutionDispatch(&rt_ctx, insn.opcode);
        try adapter_fn(
            &rt_ctx,
            &insn,
            no_registers[0..],
            no_views[0..],
            no_state_blocks[0..],
            compiled_scatter_plan.param_blocks,
        );
    }
}

// --- ABI-stable packed param structs ---
//
// These match the byte layout produced by `encodeLayerOpParam` in
// runtime_contract/types.zig.  Adapters cast `ParamBlock.data` to
// these via `@ptrCast` — zero parsing, zero allocation, zero branching.

pub const PatchEmbedParam = packed struct {
    param_kind: u8,
    in_buffer_id: u8,
    out_buffer_id: u8,
};

pub const SpatialMergeParam = packed struct {
    param_kind: u8,
    in_buffer_id: u8,
    out_buffer_id: u8,
    merge_size: u32,
};

pub const DeepstackExtractParam = packed struct {
    param_kind: u8,
    in_buffer_id: u8,
    out_buffer_id: u8,
    layer_index: u32,
};

pub const ScatterParam = packed struct {
    param_kind: u8,
    text_in_buffer_id: u8,
    vision_in_buffer_id: u8,
    out_buffer_id: u8,
    image_token_id: u32,
};

// --- Private helpers ---

fn executionState(ctx: *runtime_contract.ExecutionContext) !*ExecutionContext {
    const raw_state = ctx.workspace.any orelse return error.InvalidDispatchState;
    return @ptrCast(@alignCast(raw_state));
}

fn scatterExecutionState(ctx: *runtime_contract.ExecutionContext) !*ScatterExecutionContext {
    const raw_state = ctx.workspace.any orelse return error.InvalidDispatchState;
    return @ptrCast(@alignCast(raw_state));
}

fn paramAs(
    comptime T: type,
    insn: *const runtime_contract.Instruction,
    params: []const runtime_contract.ParamBlock,
    expected_opcode: runtime_contract.Opcode,
) !*const T {
    const param_id = insn.param_block_id orelse return error.MissingParamBlock;
    if (param_id >= params.len) return error.MissingParamBlock;
    const param_block = params[param_id];
    if (param_block.opcode != expected_opcode) return error.ParamBlockOpcodeMismatch;
    if (param_block.data.len < @bitSizeOf(T) / 8) return error.InvalidParamBlockABI;
    return @ptrCast(@alignCast(param_block.data.ptr));
}

// --- Adapter functions (KernelAdapterFn signature) ---

fn patchEmbedRuntimeAdapter(
    ctx: *runtime_contract.ExecutionContext,
    insn: *const runtime_contract.Instruction,
    _: []runtime_contract.TensorHandle,
    _: []const runtime_contract.TensorViewDesc,
    state_blocks: []runtime_contract.StateBlockHandle,
    params: []const runtime_contract.ParamBlock,
) !void {
    const exec_ctx = try executionState(ctx);
    _ = try runtime_contract.requireInstructionStateBlockForPlan(insn, &exec_ctx.compiled_plan.plan, state_blocks);
    _ = try paramAs(PatchEmbedParam, insn, params, .vision_patch_embed);
    const state = exec_ctx.state;
    if (state.saw_patch_embed or state.saw_spatial_merge) return error.InvalidVisionProgram;
    state.saw_patch_embed = true;
}

fn deepstackExtractRuntimeAdapter(
    ctx: *runtime_contract.ExecutionContext,
    insn: *const runtime_contract.Instruction,
    _: []runtime_contract.TensorHandle,
    _: []const runtime_contract.TensorViewDesc,
    state_blocks: []runtime_contract.StateBlockHandle,
    params: []const runtime_contract.ParamBlock,
) !void {
    const exec_ctx = try executionState(ctx);
    _ = try runtime_contract.requireInstructionStateBlockForPlan(insn, &exec_ctx.compiled_plan.plan, state_blocks);
    const param = try paramAs(DeepstackExtractParam, insn, params, .vision_deepstack_extract);
    const layer_idx: usize = std.math.cast(usize, param.layer_index) orelse return error.InvalidVisionProgram;
    const state = exec_ctx.state;
    if (!state.saw_patch_embed) return error.InvalidVisionProgram;
    const merger_idx = exec_ctx.vtable.deepstackLayerToMergerIndex(exec_ctx.runtime_ptr, layer_idx) orelse return error.InvalidVisionProgram;
    if (merger_idx >= exec_ctx.deepstack_layer_embeddings.len or exec_ctx.deepstack_layer_embeddings[merger_idx].len == 0) {
        return error.InvalidState;
    }
}

fn spatialMergeRuntimeAdapter(
    ctx: *runtime_contract.ExecutionContext,
    insn: *const runtime_contract.Instruction,
    _: []runtime_contract.TensorHandle,
    _: []const runtime_contract.TensorViewDesc,
    state_blocks: []runtime_contract.StateBlockHandle,
    params: []const runtime_contract.ParamBlock,
) !void {
    const exec_ctx = try executionState(ctx);
    _ = try runtime_contract.requireInstructionStateBlockForPlan(insn, &exec_ctx.compiled_plan.plan, state_blocks);
    _ = try paramAs(SpatialMergeParam, insn, params, .vision_spatial_merge);
    const state = exec_ctx.state;
    if (!state.saw_patch_embed or state.saw_spatial_merge) return error.InvalidVisionProgram;
    state.saw_spatial_merge = true;
    state.merged_embeddings = try exec_ctx.vtable.runMerger(exec_ctx.runtime_ptr, exec_ctx.grid, exec_ctx.merged_hidden_in);
}

fn scatterRuntimeAdapter(
    ctx: *runtime_contract.ExecutionContext,
    insn: *const runtime_contract.Instruction,
    _: []runtime_contract.TensorHandle,
    _: []const runtime_contract.TensorViewDesc,
    state_blocks: []runtime_contract.StateBlockHandle,
    params: []const runtime_contract.ParamBlock,
) !void {
    const param = try paramAs(ScatterParam, insn, params, .vision_scatter);
    switch (ctx.mode) {
        .vision_encode => {
            const exec_ctx = try executionState(ctx);
            _ = try runtime_contract.requireInstructionStateBlockForPlan(insn, &exec_ctx.compiled_plan.plan, state_blocks);
            const state = exec_ctx.state;
            if (!state.saw_spatial_merge) return error.InvalidVisionProgram;
        },
        .scatter => {
            const exec_ctx = try scatterExecutionState(ctx);
            _ = try runtime_contract.requireInstructionStateBlockForPlan(insn, &exec_ctx.compiled_plan.plan, state_blocks);
            // Model-level scatter plans may carry `image_token_id=0` as a
            // placeholder. Treat zero as wildcard and use request-time id.
            if (param.image_token_id != 0 and param.image_token_id != exec_ctx.image_token_id) {
                return error.InvalidArgument;
            }
            try scatterRowsByMatchedToken(
                exec_ctx.hidden_states,
                exec_ctx.seq_len,
                exec_ctx.d_model,
                exec_ctx.token_ids,
                exec_ctx.image_token_id,
                exec_ctx.embeddings,
            );
        },
        else => return error.InvalidDispatchState,
    }
}

fn scatterRowsByMatchedToken(
    hidden_states: []f32,
    seq_len: usize,
    d_model: usize,
    token_ids: []const u32,
    image_token_id: u32,
    embeddings: []const f32,
) !void {
    const expected_hidden_values = std.math.mul(usize, seq_len, d_model) catch return error.InvalidShape;
    if (hidden_states.len != expected_hidden_values) return error.InvalidShape;
    if (token_ids.len != seq_len) return error.InvalidShape;
    if (d_model == 0) return error.InvalidShape;
    if (embeddings.len % d_model != 0) return error.InvalidShape;
    const embedding_rows = embeddings.len / d_model;

    var embedding_row_index: usize = 0;
    for (token_ids, 0..) |token_id, row_idx| {
        if (token_id != image_token_id) continue;
        if (embedding_row_index >= embedding_rows) return error.InvalidShape;
        const hidden_start = std.math.mul(usize, row_idx, d_model) catch return error.InvalidShape;
        const embed_start = std.math.mul(usize, embedding_row_index, d_model) catch return error.InvalidShape;
        @memcpy(
            hidden_states[hidden_start .. hidden_start + d_model],
            embeddings[embed_start .. embed_start + d_model],
        );
        embedding_row_index += 1;
    }

    if (embedding_row_index != embedding_rows) return error.InvalidShape;
}
