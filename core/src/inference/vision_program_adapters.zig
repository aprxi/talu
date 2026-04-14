//! Shared vision program adapter functions, dispatch, and adapter table.
//!
//! Adapter functions are non-generic: they access backend-specific
//! VisionRuntime methods through `VisionRuntimeVTable`, which is
//! generated at comptime by `VTableFor(VRT)`.  This lets all backends
//! share a single `adapter_table` registered in the main per-backend
//! adapter tables (CPU `adapter_table`, Metal
//! `layer_program_adapter_table`).

const std = @import("std");
const runtime_contract = @import("runtime_contract_pkg");
const image_mod = @import("image_pkg");

const missing_register_storage_byte: u8 = 0;
const invalid_physical_id: u16 = std.math.maxInt(u16);
const InstructionIoSlices = struct {
    inputs: []const runtime_contract.TensorHandle,
    outputs: []const runtime_contract.TensorHandle,
};

const RegisterWorkspace = struct {
    register_specs: []runtime_contract.RegisterBufferSpec,
    physical_mapping: runtime_contract.PhysicalMapping,
    slot_storage: [][]u8,
    register_handles: []runtime_contract.TensorHandle,
    instruction_handles: []runtime_contract.TensorHandle,

    fn deinit(self: *RegisterWorkspace, allocator: std.mem.Allocator) void {
        for (self.slot_storage) |slot| {
            if (slot.len > 0) allocator.free(slot);
        }
        allocator.free(self.slot_storage);
        allocator.free(self.register_handles);
        allocator.free(self.instruction_handles);
        runtime_contract.deinitPhysicalMapping(allocator, &self.physical_mapping);
        allocator.free(self.register_specs);
        self.* = undefined;
    }

    fn handleForRegister(self: *const RegisterWorkspace, reg: runtime_contract.RegisterRef) runtime_contract.TensorHandle {
        return self.register_handles[runtime_contract.registerToIndex(reg)];
    }
};

fn initRegisterWorkspace(
    allocator: std.mem.Allocator,
    compiled_plan: *const runtime_contract.CompiledPlan,
) !RegisterWorkspace {
    const register_count = @as(usize, compiled_plan.plan.register_count);
    var register_specs = try allocator.alloc(runtime_contract.RegisterBufferSpec, register_count);
    errdefer allocator.free(register_specs);

    for (register_specs) |*spec| {
        spec.* = .{
            .size = 1,
            .@"align" = 64,
            .dtype = .f32,
            .layout = .contiguous,
        };
    }
    if (compiled_plan.register_buffer_specs.len == register_count) {
        for (compiled_plan.register_buffer_specs, 0..) |spec, idx| {
            register_specs[idx] = .{
                .size = spec.size,
                .@"align" = spec.@"align",
                .dtype = spec.dtype,
                .layout = spec.layout,
            };
        }
    }

    var physical_mapping = try runtime_contract.buildPhysicalMappingLinearScan(
        allocator,
        compiled_plan,
        register_specs,
    );
    errdefer runtime_contract.deinitPhysicalMapping(allocator, &physical_mapping);

    var slot_storage = try allocator.alloc([]u8, physical_mapping.physical_count);
    errdefer allocator.free(slot_storage);
    @memset(slot_storage, &.{});
    errdefer {
        for (slot_storage) |slot| {
            if (slot.len > 0) allocator.free(slot);
        }
    }
    for (physical_mapping.physical_specs, 0..) |spec, physical_idx| {
        if (spec.size == 0) continue;
        slot_storage[physical_idx] = try allocator.alloc(u8, spec.size);
    }

    const register_handles = try allocator.alloc(runtime_contract.TensorHandle, register_count);
    errdefer allocator.free(register_handles);
    for (register_handles, 0..) |*handle, reg_idx| {
        const reg = runtime_contract.registerFromIndex(@intCast(reg_idx));
        const physical_id = physical_mapping.register_to_physical[reg_idx];
        const ptr: *anyopaque = if (physical_id == invalid_physical_id)
            @ptrCast(@constCast(&missing_register_storage_byte))
        else if (physical_id >= slot_storage.len or slot_storage[physical_id].len == 0)
            @ptrCast(@constCast(&missing_register_storage_byte))
        else
            @ptrCast(slot_storage[physical_id].ptr);
        handle.* = .{
            .register = reg,
            .ptr = ptr,
        };
    }

    var max_instruction_handles: usize = 0;
    for (compiled_plan.plan.instructions) |insn| {
        const handle_count = insn.inputs.len + insn.outputs.len + insn.weights.len;
        if (handle_count > max_instruction_handles) max_instruction_handles = handle_count;
    }
    const instruction_handle_len = @max(max_instruction_handles, 1);
    const instruction_handles = try allocator.alloc(runtime_contract.TensorHandle, instruction_handle_len);

    return .{
        .register_specs = register_specs,
        .physical_mapping = physical_mapping,
        .slot_storage = slot_storage,
        .register_handles = register_handles,
        .instruction_handles = instruction_handles,
    };
}

fn instructionIoSlices(
    insn: *const runtime_contract.Instruction,
    registers: []runtime_contract.TensorHandle,
) !InstructionIoSlices {
    const io_count = insn.inputs.len + insn.outputs.len;
    if (registers.len < io_count) return error.InvalidInstructionBinding;
    return .{
        .inputs = registers[0..insn.inputs.len],
        .outputs = registers[insn.inputs.len..io_count],
    };
}

fn validateInstructionIoBindings(
    insn: *const runtime_contract.Instruction,
    io: InstructionIoSlices,
) !void {
    for (insn.inputs, 0..) |reg, idx| {
        if (io.inputs[idx].register != reg) return error.InvalidInstructionBinding;
    }
    for (insn.outputs, 0..) |reg, idx| {
        if (io.outputs[idx].register != reg) return error.InvalidInstructionBinding;
    }
}

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
    dispatch_counters: ?*runtime_contract.DispatchCounters,
    dispatch_table: runtime_contract.AdapterTable,
    grid: image_mod.VisionGrid,
    merged_hidden_in: []const f32,
    deepstack_layer_embeddings: []const []const f32,
) !EncodedSingleImage {
    var state = AdapterState{};
    errdefer if (state.merged_embeddings) |embeds| allocator.free(embeds);
    var register_workspace = try initRegisterWorkspace(allocator, compiled_vision_plan);
    defer register_workspace.deinit(allocator);
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
    const no_views: [0]runtime_contract.TensorViewDesc = .{};
    const no_state_blocks: [0]runtime_contract.StateBlockHandle = .{};
    for (compiled_vision_plan.plan.instructions, 0..) |insn, op_index| {
        exec_state.op_index = op_index;
        const adapter_fn = dispatch_table[@intFromEnum(insn.opcode)] orelse return error.InvalidVisionProgram;
        const handle_count = insn.inputs.len + insn.outputs.len + insn.weights.len;
        if (register_workspace.instruction_handles.len < handle_count) return error.InvalidInstructionBinding;
        var handle_idx: usize = 0;
        for (insn.inputs, 0..) |reg, input_idx| {
            var handle = register_workspace.handleForRegister(reg);
            if (insn.opcode == .vision_spatial_merge and input_idx == 0) {
                handle.ptr = @ptrCast(@constCast(exec_state.merged_hidden_in.ptr));
            }
            register_workspace.instruction_handles[handle_idx] = handle;
            handle_idx += 1;
        }
        for (insn.outputs) |reg| {
            register_workspace.instruction_handles[handle_idx] = register_workspace.handleForRegister(reg);
            handle_idx += 1;
        }
        const fallback_handle = runtime_contract.TensorHandle{
            .register = runtime_contract.registerFromIndex(0),
            .ptr = @ptrCast(@constCast(&missing_register_storage_byte)),
        };
        for (insn.weights) |_| {
            register_workspace.instruction_handles[handle_idx] = fallback_handle;
            handle_idx += 1;
        }

        var rt_ctx = runtime_contract.ExecutionContext{
            .mode = .vision_encode,
            .active_slots = active_slots[0..],
            .sequence_lengths = no_seq_lengths[0..],
            .batch_size = 1,
            .dispatch_counters = dispatch_counters,
            .workspace = .{ .any = @ptrCast(&exec_state) },
        };
        runtime_contract.recordExecutionDispatch(&rt_ctx, insn.opcode);
        var param_storage: [1]runtime_contract.ParamBlock = undefined;
        const params: []const runtime_contract.ParamBlock = if (insn.param_block_id) |pid| blk: {
            if (pid >= compiled_vision_plan.param_blocks.len) return error.MissingParamBlock;
            param_storage[0] = compiled_vision_plan.param_blocks[pid];
            break :blk param_storage[0..1];
        } else &.{};
        try adapter_fn(
            &rt_ctx,
            &insn,
            register_workspace.instruction_handles[0..handle_count],
            no_views[0..],
            no_state_blocks[0..],
            params,
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
    allocator: std.mem.Allocator,
    compiled_scatter_plan: *const runtime_contract.CompiledPlan,
    dispatch_counters: ?*runtime_contract.DispatchCounters,
    dispatch_table: runtime_contract.AdapterTable,
    hidden_states: []f32,
    seq_len: usize,
    d_model: usize,
    token_ids: []const u32,
    image_token_id: u32,
    embeddings: []const f32,
) !void {
    if (token_ids.len != seq_len) return error.InvalidArgument;
    var register_workspace = try initRegisterWorkspace(allocator, compiled_scatter_plan);
    defer register_workspace.deinit(allocator);

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
    const no_views: [0]runtime_contract.TensorViewDesc = .{};
    const no_state_blocks: [0]runtime_contract.StateBlockHandle = .{};
    for (compiled_scatter_plan.plan.instructions) |insn| {
        if (insn.opcode != .vision_scatter) return error.InvalidVisionProgram;
        const adapter_fn = dispatch_table[@intFromEnum(insn.opcode)] orelse return error.InvalidVisionProgram;
        const handle_count = insn.inputs.len + insn.outputs.len + insn.weights.len;
        if (register_workspace.instruction_handles.len < handle_count) return error.InvalidInstructionBinding;
        var handle_idx: usize = 0;
        for (insn.inputs, 0..) |reg, input_idx| {
            var handle = register_workspace.handleForRegister(reg);
            switch (input_idx) {
                0 => handle.ptr = @ptrCast(hidden_states.ptr),
                1 => handle.ptr = @ptrCast(@constCast(embeddings.ptr)),
                else => {},
            }
            register_workspace.instruction_handles[handle_idx] = handle;
            handle_idx += 1;
        }
        for (insn.outputs, 0..) |reg, output_idx| {
            var handle = register_workspace.handleForRegister(reg);
            if (output_idx == 0) handle.ptr = @ptrCast(hidden_states.ptr);
            register_workspace.instruction_handles[handle_idx] = handle;
            handle_idx += 1;
        }
        const fallback_handle = runtime_contract.TensorHandle{
            .register = runtime_contract.registerFromIndex(0),
            .ptr = @ptrCast(@constCast(&missing_register_storage_byte)),
        };
        for (insn.weights) |_| {
            register_workspace.instruction_handles[handle_idx] = fallback_handle;
            handle_idx += 1;
        }
        var rt_ctx = runtime_contract.ExecutionContext{
            .mode = .scatter,
            .active_slots = active_slots[0..],
            .sequence_lengths = no_seq_lengths[0..],
            .batch_size = 1,
            .dispatch_counters = dispatch_counters,
            .workspace = .{ .any = @ptrCast(&scatter_ctx) },
        };
        runtime_contract.recordExecutionDispatch(&rt_ctx, insn.opcode);
        var param_storage: [1]runtime_contract.ParamBlock = undefined;
        const params: []const runtime_contract.ParamBlock = if (insn.param_block_id) |pid| blk: {
            if (pid >= compiled_scatter_plan.param_blocks.len) return error.MissingParamBlock;
            param_storage[0] = compiled_scatter_plan.param_blocks[pid];
            break :blk param_storage[0..1];
        } else &.{};
        try adapter_fn(
            &rt_ctx,
            &insn,
            register_workspace.instruction_handles[0..handle_count],
            no_views[0..],
            no_state_blocks[0..],
            params,
        );
    }
}

// --- Private helpers ---

fn executionState(ctx: *runtime_contract.ExecutionContext) !*ExecutionContext {
    const raw_state = ctx.workspace.any orelse return error.InvalidDispatchState;
    return @ptrCast(@alignCast(raw_state));
}

fn scatterExecutionState(ctx: *runtime_contract.ExecutionContext) !*ScatterExecutionContext {
    const raw_state = ctx.workspace.any orelse return error.InvalidDispatchState;
    return @ptrCast(@alignCast(raw_state));
}

// --- Adapter functions (KernelAdapterFn signature) ---

fn patchEmbedRuntimeAdapter(
    ctx: *runtime_contract.ExecutionContext,
    insn: *const runtime_contract.Instruction,
    registers: []runtime_contract.TensorHandle,
    _: []const runtime_contract.TensorViewDesc,
    state_blocks: []runtime_contract.StateBlockHandle,
    params: []const runtime_contract.ParamBlock,
) !void {
    const exec_ctx = try executionState(ctx);
    _ = try runtime_contract.requireInstructionStateBlockForPlan(insn, &exec_ctx.compiled_plan.plan, state_blocks);
    const io = try instructionIoSlices(insn, registers);
    try validateInstructionIoBindings(insn, io);
    _ = try runtime_contract.paramAs(runtime_contract.PatchEmbedParam, params, .vision_patch_embed);
    const state = exec_ctx.state;
    if (state.saw_patch_embed or state.saw_spatial_merge) return error.InvalidVisionProgram;
    state.saw_patch_embed = true;
}

fn deepstackExtractRuntimeAdapter(
    ctx: *runtime_contract.ExecutionContext,
    insn: *const runtime_contract.Instruction,
    registers: []runtime_contract.TensorHandle,
    _: []const runtime_contract.TensorViewDesc,
    state_blocks: []runtime_contract.StateBlockHandle,
    params: []const runtime_contract.ParamBlock,
) !void {
    const exec_ctx = try executionState(ctx);
    _ = try runtime_contract.requireInstructionStateBlockForPlan(insn, &exec_ctx.compiled_plan.plan, state_blocks);
    const io = try instructionIoSlices(insn, registers);
    try validateInstructionIoBindings(insn, io);
    const param = try runtime_contract.paramAs(runtime_contract.DeepstackExtractParam, params, .vision_deepstack_extract);
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
    registers: []runtime_contract.TensorHandle,
    _: []const runtime_contract.TensorViewDesc,
    state_blocks: []runtime_contract.StateBlockHandle,
    params: []const runtime_contract.ParamBlock,
) !void {
    const exec_ctx = try executionState(ctx);
    _ = try runtime_contract.requireInstructionStateBlockForPlan(insn, &exec_ctx.compiled_plan.plan, state_blocks);
    const io = try instructionIoSlices(insn, registers);
    try validateInstructionIoBindings(insn, io);
    if (io.inputs.len == 0) return error.InvalidInstructionBinding;
    _ = try runtime_contract.paramAs(runtime_contract.SpatialMergeParam, params, .vision_spatial_merge);
    const state = exec_ctx.state;
    if (!state.saw_patch_embed or state.saw_spatial_merge) return error.InvalidVisionProgram;
    state.saw_spatial_merge = true;
    const merged_hidden_ptr: [*]const f32 = @ptrCast(@alignCast(io.inputs[0].ptr));
    const merged_hidden = merged_hidden_ptr[0..exec_ctx.merged_hidden_in.len];
    state.merged_embeddings = try exec_ctx.vtable.runMerger(exec_ctx.runtime_ptr, exec_ctx.grid, merged_hidden);
}

fn scatterRuntimeAdapter(
    ctx: *runtime_contract.ExecutionContext,
    insn: *const runtime_contract.Instruction,
    registers: []runtime_contract.TensorHandle,
    _: []const runtime_contract.TensorViewDesc,
    state_blocks: []runtime_contract.StateBlockHandle,
    params: []const runtime_contract.ParamBlock,
) !void {
    const param = try runtime_contract.paramAs(runtime_contract.ScatterParam, params, .vision_scatter);
    const io = try instructionIoSlices(insn, registers);
    try validateInstructionIoBindings(insn, io);
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
            if (io.inputs.len < 2) return error.InvalidInstructionBinding;
            if (io.outputs.len == 0) return error.InvalidInstructionBinding;
            // Model-level scatter plans may carry `image_token_id=0` as a
            // placeholder. Treat zero as wildcard and use request-time id.
            if (param.image_token_id != 0 and param.image_token_id != exec_ctx.image_token_id) {
                return error.InvalidArgument;
            }
            const hidden_states_ptr: [*]f32 = @ptrCast(@alignCast(io.outputs[0].ptr));
            const embeddings_ptr: [*]const f32 = @ptrCast(@alignCast(io.inputs[1].ptr));
            try scatterRowsByMatchedToken(
                hidden_states_ptr[0..exec_ctx.hidden_states.len],
                exec_ctx.seq_len,
                exec_ctx.d_model,
                exec_ctx.token_ids,
                exec_ctx.image_token_id,
                embeddings_ptr[0..exec_ctx.embeddings.len],
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
