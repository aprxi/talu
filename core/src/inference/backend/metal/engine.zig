//! Metal backend for transformer inference (macOS GPU via MLX).
//!
//! Provides GPU-accelerated inference using Apple's MLX framework.
//! Supports lazy graph execution for optimal GPU utilization.

const std = @import("std");
const models = @import("../../../models/root.zig");
const contract = @import("../contract.zig");
const tensor = @import("../../../tensor.zig");
const common_mrope = @import("vision/mrope.zig");
const ModelConfig = tensor.ModelConfig;
const log = @import("../../../log.zig");

// Import compute primitives from compute/
const compute = @import("../../../compute/root.zig");
const metal_compute = compute.metal;
const graph = metal_compute.graph;
const metal_sampling = @import("sampling.zig");
const runtime_graph_mod = @import("runtime_graph.zig");
const runtime_contract = @import("../../runtime_contract/root.zig");
const LoadedModel = models.LoadedModel;

// Internal orchestration modules
const metal_executor = @import("executor/root.zig");
const weights_trait = metal_executor.weights;
const runtime_trait = metal_executor.runtime;
const vision_runtime_mod = @import("vision/root.zig");

// Re-exports for direct access if needed
pub const device = metal_compute.device;
pub const matmul = metal_compute.matmul;
pub const Graph = graph;
pub const Forward = runtime_trait;

pub const Device = metal_compute.Device;
pub const Buffer = metal_compute.Buffer;
pub const isAvailable = metal_compute.isAvailable;
pub const Cache = runtime_graph_mod.Cache;
pub const GatedDeltaCache = runtime_graph_mod.GatedDeltaCache;
pub const WeightHandles = weights_trait.WeightHandles;
/// Metal backend for GPU-accelerated transformer inference
pub const MetalBackend = struct {
    const MaxStateDescriptors: usize = runtime_contract.max_state_descriptors;
    pub const capabilities: contract.Capabilities = .{
        .vision_prefill = true,
        .decode_batch = false,
        .decode_streaming = true,
        .embedding = true,
        .warmup = false,
    };

    pub const PrefillVisionInput = vision_runtime_mod.PrefillVisionInput;

    allocator: std.mem.Allocator,
    config: ModelConfig,
    weights: *weights_trait.WeightHandles,
    layer_count: usize,
    cache_max_seq_len: usize,
    vocab_size: usize,
    d_model: usize,
    max_batch_size: usize,
    slot_in_use: bool,
    /// Slots 1..N-1 for scheduler-managed multi-slot decode.
    extra_slots: []SlotState,
    slot_logits_buffer: []f32,
    vision_runtime: ?vision_runtime_mod.VisionRuntime = null,
    slot_rope_position_delta: isize,
    state_descriptors_storage: [MaxStateDescriptors]runtime_contract.StateDescriptor,
    state_runtime_roles: [MaxStateDescriptors]StateRuntimeRole,
    state_descriptor_count: u8,
    slot0_state_binding: SlotStateBinding = .{},

    // Track position for decode
    current_position: usize,

    const SlotState = struct {
        state_binding: SlotStateBinding = .{},
        in_use: bool = false,
        position: usize = 0,
        rope_position_delta: isize = 0,
    };

    const SlotStateBinding = struct {
        handles: [MaxStateDescriptors]runtime_contract.StateBlockHandle = undefined,
        initialized: [MaxStateDescriptors]bool = [_]bool{false} ** MaxStateDescriptors,
        count: u8 = 0,
        bound: bool = false,

        fn detach(self: *SlotStateBinding) void {
            self.bound = false;
        }

        fn clear(self: *SlotStateBinding) void {
            self.initialized = [_]bool{false} ** MaxStateDescriptors;
            self.count = 0;
            self.bound = false;
        }
    };

    fn argmaxHost(values: []const f32) u32 {
        var best_idx: usize = 0;
        var best_val: f32 = -std.math.inf(f32);
        for (values, 0..) |v, idx| {
            if (v > best_val) {
                best_val = v;
                best_idx = idx;
            }
        }
        return @intCast(best_idx);
    }

    fn argminHost(values: []const f32) u32 {
        var best_idx: usize = 0;
        var best_val: f32 = std.math.inf(f32);
        for (values, 0..) |v, idx| {
            if (v < best_val) {
                best_val = v;
                best_idx = idx;
            }
        }
        return @intCast(best_idx);
    }

    fn selectNextTokenFromLogits(self: *const MetalBackend, logits: []const f32) !u32 {
        if (logits.len != self.vocab_size) return error.InvalidArgument;
        return if (self.config.logits_scaling < 0.0) argminHost(logits) else argmaxHost(logits);
    }

    fn resolveMaxBatchSize() usize {
        if (std.posix.getenv("TALU_METAL_MAX_BATCH_SIZE")) |raw| {
            const parsed = std.fmt.parseUnsigned(usize, std.mem.sliceTo(raw, 0), 10) catch return 8;
            return @max(@as(usize, 1), parsed);
        }
        return 8;
    }

    fn resolveCacheMaxSeqLen(config_max_seq_len: usize) usize {
        // Fixed-capacity KV allocation at very large context lengths can stall
        // decode startup. Switch those models to dynamic cache growth.
        const fixed_capacity_limit: usize = 65_536;
        if (config_max_seq_len > fixed_capacity_limit) return 0;
        return config_max_seq_len;
    }

    fn primeBoundSlotExecutionGraph(self: *MetalBackend, slot_index: usize) !void {
        if (self.state_descriptor_count == 0) return;
        const prime_shapes = [_]usize{ 1, 21 };
        for (prime_shapes) |prime_seq_len| {
            const prime_tokens = try self.allocator.alloc(u32, prime_seq_len);
            defer self.allocator.free(prime_tokens);
            @memset(prime_tokens, 0);
            if (prime_seq_len == 1) {
                // Decode warmup: warm the single-token logits path used by decodeSlot.
                const prime_logits = try runtime_trait.transformerForwardLazy(
                    self.allocator,
                    self.weights,
                    prime_tokens,
                    try self.slotStateBlocks(slot_index),
                    self.config,
                    0,
                );
                defer graph.freeArray(prime_logits);
                graph.eval(&[_]graph.ArrayHandle{prime_logits});
            } else {
                // Prefill warmup: warm the hidden prefill graph used by prefill.
                const prime_hidden = try runtime_trait.transformerForwardHiddenLazy(
                    self.allocator,
                    self.weights,
                    prime_tokens,
                    try self.slotStateBlocks(slot_index),
                    self.config,
                    0,
                );
                defer graph.freeArray(prime_hidden);
                graph.eval(&[_]graph.ArrayHandle{prime_hidden});
            }
        }
    }

    fn cacheMaxSeqLen(self: *const MetalBackend) usize {
        return resolveCacheMaxSeqLen(@intCast(self.config.max_seq_len));
    }

    fn extraSlotIndexFor(slot_index: usize, max_batch_size: usize) !usize {
        if (slot_index == 0) return error.InvalidArgument;
        if (slot_index >= max_batch_size) return error.InvalidArgument;
        return slot_index - 1;
    }

    fn toExtraSlotIndex(self: *const MetalBackend, slot_index: usize) !usize {
        return extraSlotIndexFor(slot_index, self.max_batch_size);
    }

    fn slotPositionPtr(self: *MetalBackend, slot_index: usize) !*usize {
        if (slot_index == 0) return &self.current_position;
        const extra_idx = try self.toExtraSlotIndex(slot_index);
        return &self.extra_slots[extra_idx].position;
    }

    fn slotRopeDeltaPtr(self: *MetalBackend, slot_index: usize) !*isize {
        if (slot_index == 0) return &self.slot_rope_position_delta;
        const extra_idx = try self.toExtraSlotIndex(slot_index);
        return &self.extra_slots[extra_idx].rope_position_delta;
    }

    fn stateObjectPtr(comptime T: type, state_block: *const runtime_contract.StateBlockHandle) !*T {
        return runtime_contract.stateValueFromBlock(*T, state_block) orelse error.InvalidStateDescriptorBinding;
    }

    const StateRuntimeRole = enum(u8) {
        none = 0,
        kv_cache = 1,
        shortconv_cache = 2,
        mamba_cache = 3,
        gated_delta_cache = 4,
    };

    fn runtimeRoleForRuntimeKind(runtime_kind: u8) !StateRuntimeRole {
        return switch (runtime_kind) {
            runtime_contract.state_runtime_kind_none => .none,
            runtime_contract.state_runtime_kind_kv_cache => .kv_cache,
            runtime_contract.state_runtime_kind_shortconv_cache => .shortconv_cache,
            runtime_contract.state_runtime_kind_mamba_cache => .mamba_cache,
            runtime_contract.state_runtime_kind_gated_delta_cache => .gated_delta_cache,
            else => error.InvalidStateDescriptorBinding,
        };
    }

    fn deriveStateRuntimeRoles(
        descriptors: []const runtime_contract.StateDescriptor,
    ) ![MaxStateDescriptors]StateRuntimeRole {
        var roles = [_]StateRuntimeRole{.none} ** MaxStateDescriptors;
        for (descriptors, 0..) |descriptor, idx| {
            roles[idx] = try runtimeRoleForRuntimeKind(descriptor.runtime_kind);
        }
        return roles;
    }

    const StateRuntimeOps = struct {
        init: *const fn (self: *MetalBackend, state_block: *const runtime_contract.StateBlockHandle) anyerror!bool,
        reset: *const fn (state_block: *const runtime_contract.StateBlockHandle) anyerror!void,
        deinit: *const fn (state_block: *const runtime_contract.StateBlockHandle) anyerror!void,
    };

    fn initNoopState(_: *MetalBackend, _: *const runtime_contract.StateBlockHandle) !bool {
        return false;
    }

    fn resetNoopState(_: *const runtime_contract.StateBlockHandle) !void {}

    fn deinitNoopState(_: *const runtime_contract.StateBlockHandle) !void {}

    fn initKvCacheState(self: *MetalBackend, state_block: *const runtime_contract.StateBlockHandle) !bool {
        const cache = try stateObjectPtr(runtime_graph_mod.Cache, state_block);
        cache.* = runtime_graph_mod.Cache.init(self.layer_count, true, self.cache_max_seq_len);
        return true;
    }

    fn resetKvCacheState(state_block: *const runtime_contract.StateBlockHandle) !void {
        (try stateObjectPtr(runtime_graph_mod.Cache, state_block)).reset();
    }

    fn deinitKvCacheState(state_block: *const runtime_contract.StateBlockHandle) !void {
        const cache = try stateObjectPtr(runtime_graph_mod.Cache, state_block);
        cache.deinit();
        cache.* = runtime_graph_mod.Cache.disabled(true);
    }

    fn initShortConvState(self: *MetalBackend, state_block: *const runtime_contract.StateBlockHandle) !bool {
        const shortconv = try stateObjectPtr(runtime_graph_mod.ShortConvCache, state_block);
        shortconv.* = runtime_graph_mod.ShortConvCache.init(self.layer_count);
        return true;
    }

    fn resetShortConvState(state_block: *const runtime_contract.StateBlockHandle) !void {
        (try stateObjectPtr(runtime_graph_mod.ShortConvCache, state_block)).reset();
    }

    fn deinitShortConvState(state_block: *const runtime_contract.StateBlockHandle) !void {
        const shortconv = try stateObjectPtr(runtime_graph_mod.ShortConvCache, state_block);
        shortconv.deinit();
        shortconv.* = runtime_graph_mod.ShortConvCache.disabled();
    }

    fn initMambaState(self: *MetalBackend, state_block: *const runtime_contract.StateBlockHandle) !bool {
        const mamba = try stateObjectPtr(runtime_graph_mod.MambaCache, state_block);
        mamba.* = runtime_graph_mod.MambaCache.init(self.layer_count);
        return true;
    }

    fn resetMambaState(state_block: *const runtime_contract.StateBlockHandle) !void {
        (try stateObjectPtr(runtime_graph_mod.MambaCache, state_block)).reset();
    }

    fn deinitMambaState(state_block: *const runtime_contract.StateBlockHandle) !void {
        const mamba = try stateObjectPtr(runtime_graph_mod.MambaCache, state_block);
        mamba.deinit();
        mamba.* = runtime_graph_mod.MambaCache.disabled();
    }

    fn initGatedDeltaState(self: *MetalBackend, state_block: *const runtime_contract.StateBlockHandle) !bool {
        const gated_delta = try stateObjectPtr(runtime_graph_mod.GatedDeltaCache, state_block);
        gated_delta.* = runtime_graph_mod.GatedDeltaCache.init(self.layer_count);
        return true;
    }

    fn resetGatedDeltaState(state_block: *const runtime_contract.StateBlockHandle) !void {
        (try stateObjectPtr(runtime_graph_mod.GatedDeltaCache, state_block)).reset();
    }

    fn deinitGatedDeltaState(state_block: *const runtime_contract.StateBlockHandle) !void {
        const gated_delta = try stateObjectPtr(runtime_graph_mod.GatedDeltaCache, state_block);
        gated_delta.deinit();
        gated_delta.* = runtime_graph_mod.GatedDeltaCache.disabled();
    }

    const state_runtime_ops: [@typeInfo(StateRuntimeRole).@"enum".fields.len]StateRuntimeOps = blk: {
        var table: [@typeInfo(StateRuntimeRole).@"enum".fields.len]StateRuntimeOps = undefined;
        table[@intFromEnum(StateRuntimeRole.none)] = .{
            .init = initNoopState,
            .reset = resetNoopState,
            .deinit = deinitNoopState,
        };
        table[@intFromEnum(StateRuntimeRole.kv_cache)] = .{
            .init = initKvCacheState,
            .reset = resetKvCacheState,
            .deinit = deinitKvCacheState,
        };
        table[@intFromEnum(StateRuntimeRole.shortconv_cache)] = .{
            .init = initShortConvState,
            .reset = resetShortConvState,
            .deinit = deinitShortConvState,
        };
        table[@intFromEnum(StateRuntimeRole.mamba_cache)] = .{
            .init = initMambaState,
            .reset = resetMambaState,
            .deinit = deinitMambaState,
        };
        table[@intFromEnum(StateRuntimeRole.gated_delta_cache)] = .{
            .init = initGatedDeltaState,
            .reset = resetGatedDeltaState,
            .deinit = deinitGatedDeltaState,
        };
        break :blk table;
    };

    fn runtimeStateOps(role: StateRuntimeRole) StateRuntimeOps {
        return state_runtime_ops[@intFromEnum(role)];
    }

    fn initStateObjectForDescriptor(
        self: *MetalBackend,
        role: StateRuntimeRole,
        state_block: *const runtime_contract.StateBlockHandle,
    ) !bool {
        const ops = runtimeStateOps(role);
        return try ops.init(self, state_block);
    }

    fn resetStateObjectForDescriptor(
        role: StateRuntimeRole,
        state_block: *const runtime_contract.StateBlockHandle,
    ) !void {
        const ops = runtimeStateOps(role);
        try ops.reset(state_block);
    }

    fn runtimeHandleLooksValid(handle: ?*anyopaque) bool {
        if (handle == null) return true;
        return @intFromPtr(handle.?) >= 4096;
    }

    fn stateObjectLooksValid(
        role: StateRuntimeRole,
        state_block: *const runtime_contract.StateBlockHandle,
    ) bool {
        return switch (role) {
            .none => true,
            .kv_cache => blk: {
                const cache = stateObjectPtr(runtime_graph_mod.Cache, state_block) catch break :blk false;
                break :blk runtimeHandleLooksValid(cache.handle);
            },
            .shortconv_cache => blk: {
                const cache = stateObjectPtr(runtime_graph_mod.ShortConvCache, state_block) catch break :blk false;
                break :blk runtimeHandleLooksValid(cache.handle);
            },
            .mamba_cache => blk: {
                const cache = stateObjectPtr(runtime_graph_mod.MambaCache, state_block) catch break :blk false;
                break :blk runtimeHandleLooksValid(cache.handle);
            },
            .gated_delta_cache => blk: {
                const cache = stateObjectPtr(runtime_graph_mod.GatedDeltaCache, state_block) catch break :blk false;
                break :blk runtimeHandleLooksValid(cache.handle);
            },
        };
    }

    fn deinitStateObjectForDescriptor(
        role: StateRuntimeRole,
        state_block: *const runtime_contract.StateBlockHandle,
    ) !void {
        const ops = runtimeStateOps(role);
        try ops.deinit(state_block);
    }

    fn deinitSlotBindingStateObjects(self: *MetalBackend, binding: *SlotStateBinding) void {
        const count = @min(@as(usize, @intCast(binding.count)), @as(usize, self.state_descriptor_count));
        for (0..count) |idx| {
            if (!binding.initialized[idx]) continue;
            deinitStateObjectForDescriptor(self.state_runtime_roles[idx], &binding.handles[idx]) catch {};
            binding.initialized[idx] = false;
        }
    }

    fn releaseNonPersistentStateObjects(self: *MetalBackend, binding: *SlotStateBinding) void {
        const descriptors = self.stateDescriptors();
        const count = @min(@as(usize, @intCast(binding.count)), descriptors.len);
        for (0..count) |idx| {
            if (!binding.initialized[idx]) continue;
            if (descriptors[idx].lifecycle == .slot_persistent) continue;
            deinitStateObjectForDescriptor(self.state_runtime_roles[idx], &binding.handles[idx]) catch {};
            binding.initialized[idx] = false;
        }
    }

    fn resetSlotState(self: *MetalBackend, slot_index: usize) !void {
        const slot_position = try self.slotPositionPtr(slot_index);
        const slot_rope_delta = try self.slotRopeDeltaPtr(slot_index);
        slot_position.* = 0;
        slot_rope_delta.* = 0;
        if (self.state_descriptor_count == 0) return;
        const binding = try self.slotStateBinding(slot_index);
        if (!binding.bound) return;
        const descriptors = self.stateDescriptors();
        const count = @min(@as(usize, @intCast(binding.count)), descriptors.len);
        for (0..count) |idx| {
            if (!binding.initialized[idx]) continue;
            const role = self.state_runtime_roles[idx];
            if (!stateObjectLooksValid(role, &binding.handles[idx])) {
                binding.initialized[idx] = try self.initStateObjectForDescriptor(role, &binding.handles[idx]);
            }
            try resetStateObjectForDescriptor(role, &binding.handles[idx]);
        }
    }

    fn prefillSlotImpl(self: *MetalBackend, slot_index: usize, tokens: []const u32, logits_out: []f32) !void {
        const sequence_len = tokens.len;
        if (sequence_len == 0) return;
        const trace_prefill_timing = std.posix.getenv("TALU_METAL_PREFILL_TIMING") != null;
        const t_start_ns: i128 = if (trace_prefill_timing) std.time.nanoTimestamp() else 0;

        const slot_position = try self.slotPositionPtr(slot_index);
        const slot_rope_delta = try self.slotRopeDeltaPtr(slot_index);
        slot_rope_delta.* = 0;

        // Reset cache for new sequence in this scheduler slot.
        try self.resetSlotState(slot_index);
        const t_reset_done_ns: i128 = if (trace_prefill_timing) std.time.nanoTimestamp() else 0;

        const hidden_handle = try runtime_trait.transformerForwardHiddenLazy(
            self.allocator,
            self.weights,
            tokens,
            try self.slotStateBlocks(slot_index),
            self.config,
            0, // pos_offset
        );
        defer graph.freeArray(hidden_handle);
        var starts: [3]c_int = .{ 0, @intCast(sequence_len - 1), 0 };
        var ends: [3]c_int = .{ 1, @intCast(sequence_len), @intCast(self.d_model) };
        const last_hidden_handle = graph.mlx_lazy_slice(hidden_handle, &starts, &ends, 3);
        defer graph.freeArray(last_hidden_handle);
        const logits_handle = metal_executor.block.TransformerBlock.projectLogits(
            last_hidden_handle,
            self.weights,
            self.config.norm_eps,
        );
        const t_graph_built_ns: i128 = if (trace_prefill_timing) std.time.nanoTimestamp() else 0;

        graph.eval(&[_]graph.ArrayHandle{logits_handle});
        const t_eval_done_ns: i128 = if (trace_prefill_timing) std.time.nanoTimestamp() else 0;

        var shape_buffer: [8]usize = undefined;
        const rank = graph.getShape(logits_handle, &shape_buffer);
        std.debug.assert(rank >= 1);
        std.debug.assert(shape_buffer[rank - 1] == self.vocab_size);
        if (rank == 2) std.debug.assert(shape_buffer[0] == 1);
        if (trace_prefill_timing) {
            std.debug.print(
                "METAL_PREFILL_LOGITS_SHAPE rank={} d0={} d1={} d2={}\n",
                .{
                    rank,
                    if (rank > 0) shape_buffer[0] else 0,
                    if (rank > 1) shape_buffer[1] else 0,
                    if (rank > 2) shape_buffer[2] else 0,
                },
            );
        }
        graph.copyToHost(logits_handle, logits_out[0..self.vocab_size]);
        const t_copy_done_ns: i128 = if (trace_prefill_timing) std.time.nanoTimestamp() else 0;

        graph.freeArray(logits_handle);
        const t_end_ns: i128 = if (trace_prefill_timing) std.time.nanoTimestamp() else 0;

        slot_position.* = sequence_len;

        if (trace_prefill_timing) {
            const reset_ns = t_reset_done_ns - t_start_ns;
            const graph_build_ns = t_graph_built_ns - t_reset_done_ns;
            const build_ns = t_graph_built_ns - t_start_ns;
            const eval_ns = t_eval_done_ns - t_graph_built_ns;
            const copy_ns = t_copy_done_ns - t_eval_done_ns;
            const finalize_ns = t_end_ns - t_copy_done_ns;
            const total_ns = t_end_ns - t_start_ns;
            std.debug.print(
                "METAL_PREFILL_TIMING slot={} seq={} reset_us={d:.3} graph_us={d:.3} build_us={d:.3} eval_us={d:.3} copy_us={d:.3} finalize_us={d:.3} total_us={d:.3}\n",
                .{
                    slot_index,
                    sequence_len,
                    @as(f64, @floatFromInt(reset_ns)) / 1000.0,
                    @as(f64, @floatFromInt(graph_build_ns)) / 1000.0,
                    @as(f64, @floatFromInt(build_ns)) / 1000.0,
                    @as(f64, @floatFromInt(eval_ns)) / 1000.0,
                    @as(f64, @floatFromInt(copy_ns)) / 1000.0,
                    @as(f64, @floatFromInt(finalize_ns)) / 1000.0,
                    @as(f64, @floatFromInt(total_ns)) / 1000.0,
                },
            );
        }
    }

    fn decodeSlot(
        self: *MetalBackend,
        slot_index: usize,
        token: u32,
        position: usize,
        logits_out: []f32,
    ) !void {
        const slot_rope_delta = try self.slotRopeDeltaPtr(slot_index);
        const effective_position = try common_mrope.applyPositionDelta(position, slot_rope_delta.*);
        const logits_handle = try runtime_trait.transformerForwardLazy(
            self.allocator,
            self.weights,
            &[_]u32{token},
            try self.slotStateBlocks(slot_index),
            self.config,
            effective_position,
        );
        defer graph.freeArray(logits_handle);
        graph.eval(&[_]graph.ArrayHandle{logits_handle});
        graph.copyToHost(logits_handle, logits_out);
        const slot_position = try self.slotPositionPtr(slot_index);
        slot_position.* = position + 1;
    }

    fn decodeSlotGreedy(
        self: *MetalBackend,
        slot_index: usize,
        token: u32,
        position: usize,
    ) !u32 {
        const slot_rope_delta = try self.slotRopeDeltaPtr(slot_index);
        const effective_position = try common_mrope.applyPositionDelta(position, slot_rope_delta.*);
        const logits_handle = try runtime_trait.transformerForwardLazy(
            self.allocator,
            self.weights,
            &[_]u32{token},
            try self.slotStateBlocks(slot_index),
            self.config,
            effective_position,
        );
        defer graph.freeArray(logits_handle);

        const selection_logits = if (self.config.logits_scaling < 0.0)
            graph.mlx_lazy_multiply_scalar(logits_handle, -1.0)
        else
            logits_handle;
        defer if (selection_logits != logits_handle) graph.freeArray(selection_logits);

        const token_handle = graph.mlx_lazy_argmax(selection_logits, -1);
        defer graph.freeArray(token_handle);
        graph.eval(&[_]graph.ArrayHandle{token_handle});

        const slot_position = try self.slotPositionPtr(slot_index);
        slot_position.* = position + 1;
        return graph.mlx_array_item_u32(token_handle);
    }

    pub fn supportsSchedulerBackendTopKDecodeRoute(
        self: *const MetalBackend,
        sampling_config: *const metal_sampling.SamplingConfig,
    ) bool {
        _ = self;
        return sampling_config.strategy == .top_k and
            sampling_config.top_k > 0 and
            sampling_config.temperature > 0.0 and
            sampling_config.min_p == 0.0;
    }

    pub fn decodeTopKCandidates(
        self: *MetalBackend,
        slot_index: usize,
        token: u32,
        top_k: usize,
        candidate_logits_out: []f32,
        candidate_ids_out: []u32,
    ) !usize {
        if (top_k == 0) return error.InvalidArgument;
        if (candidate_logits_out.len < top_k or candidate_ids_out.len < top_k) return error.InvalidArgument;

        const slot_position = try self.slotPositionPtr(slot_index);
        const slot_rope_delta = try self.slotRopeDeltaPtr(slot_index);
        const effective_position = try common_mrope.applyPositionDelta(slot_position.*, slot_rope_delta.*);
        const logits_handle = try runtime_trait.transformerForwardLazy(
            self.allocator,
            self.weights,
            &[_]u32{token},
            try self.slotStateBlocks(slot_index),
            self.config,
            effective_position,
        );
        defer graph.freeArray(logits_handle);

        var shape_buffer: [8]usize = undefined;
        const rank = graph.getShape(logits_handle, &shape_buffer);
        if (rank == 0) return error.InvalidShape;

        const vocab_dim = shape_buffer[rank - 1];
        if (vocab_dim != self.vocab_size) return error.InvalidShape;

        const k = @min(top_k, vocab_dim);
        const axis: c_int = @intCast(rank - 1);
        const partitioned_indices = graph.mlx_lazy_argpartition(logits_handle, -@as(c_int, @intCast(k)), axis);
        defer graph.freeArray(partitioned_indices);

        var starts = [_]c_int{0} ** 8;
        var ends = [_]c_int{0} ** 8;
        for (0..rank) |dim| {
            ends[dim] = @intCast(shape_buffer[dim]);
        }
        starts[rank - 1] = @intCast(vocab_dim - k);

        const top_k_indices = graph.mlx_lazy_slice(partitioned_indices, &starts, &ends, rank);
        defer graph.freeArray(top_k_indices);

        const top_k_logits = graph.mlx_lazy_take_along_axis(logits_handle, top_k_indices, axis);
        defer graph.freeArray(top_k_logits);

        graph.eval(&[_]graph.ArrayHandle{ top_k_indices, top_k_logits });
        graph.copyToHost(top_k_logits, candidate_logits_out[0..k]);
        graph.copyU32ToHost(top_k_indices, candidate_ids_out[0..k]);
        slot_position.* += 1;
        return k;
    }

    fn slotLogits(self: *MetalBackend, slot_index: usize) ![]f32 {
        if (slot_index >= self.max_batch_size) return error.InvalidArgument;
        const start = slot_index * self.vocab_size;
        return self.slot_logits_buffer[start .. start + self.vocab_size];
    }

    pub fn maxBatchSize(self: *const MetalBackend) usize {
        return self.max_batch_size;
    }

    pub fn vocabSize(self: *const MetalBackend) usize {
        return self.vocab_size;
    }

    pub fn supportsSchedulerBackendDecodeStreamingRoute(self: *const MetalBackend) bool {
        _ = self;
        return true;
    }

    pub fn init(allocator: std.mem.Allocator, loaded: *LoadedModel) !MetalBackend {
        // Load weights to GPU
        const weight_handles = try weights_trait.loadWeightsToGPU(allocator, loaded);
        errdefer weights_trait.freeWeights(allocator, weight_handles);

        // Validate compiled layer plans once at backend init so forward path
        // stays focused on hot execution.
        for (weight_handles.layers, 0..) |*layer, layer_idx| {
            if (layer.compiled_plan == null) continue;
            try metal_executor.block.TransformerBlock.validateCompiledLayerProgram(layer, layer_idx);
        }

        const layer_count: usize = @intCast(loaded.config.n_layers);
        const max_seq_len: usize = @intCast(loaded.config.max_seq_len);
        const cache_max_seq_len = resolveCacheMaxSeqLen(max_seq_len);
        log.debug("inference", "Metal cache capacity policy", .{
            .config_max_seq_len = max_seq_len,
            .cache_max_seq_len = cache_max_seq_len,
            .dynamic_growth = @as(u8, @intFromBool(cache_max_seq_len == 0)),
        }, @src());
        var state_descriptors_storage: [MaxStateDescriptors]runtime_contract.StateDescriptor = undefined;
        var state_descriptor_count: u8 = 0;
        for (weight_handles.layers) |*layer| {
            if (layer.compiled_plan) |*compiled_plan| {
                try runtime_contract.appendUniquePlanStateDescriptors(
                    state_descriptors_storage[0..],
                    &state_descriptor_count,
                    &compiled_plan.plan,
                );
            }
        }
        const state_runtime_roles = try deriveStateRuntimeRoles(state_descriptors_storage[0..state_descriptor_count]);
        var vision_runtime = try vision_runtime_mod.VisionRuntime.init(allocator, loaded);
        errdefer if (vision_runtime) |*rt| rt.deinit();

        // Scheduler-visible slot capacity for Metal backend.
        const max_batch_size: usize = resolveMaxBatchSize();
        const extra_slot_count = max_batch_size - 1;
        const extra_slots = try allocator.alloc(SlotState, extra_slot_count);
        errdefer allocator.free(extra_slots);
        for (extra_slots) |*slot| {
            slot.* = .{
                .in_use = false,
                .position = 0,
                .rope_position_delta = 0,
            };
        }

        const slot_logits_buffer = try allocator.alloc(f32, max_batch_size * @as(usize, @intCast(loaded.config.vocab_size)));
        errdefer allocator.free(slot_logits_buffer);
        var backend = MetalBackend{
            .allocator = allocator,
            .config = loaded.config,
            .weights = weight_handles,
            .layer_count = layer_count,
            .cache_max_seq_len = cache_max_seq_len,
            .vocab_size = @intCast(loaded.config.vocab_size),
            .d_model = @intCast(loaded.config.d_model),
            .max_batch_size = max_batch_size,
            .slot_in_use = false,
            .extra_slots = extra_slots,
            .slot_logits_buffer = slot_logits_buffer,
            .vision_runtime = vision_runtime,
            .slot_rope_position_delta = 0,
            .state_descriptors_storage = state_descriptors_storage,
            .state_runtime_roles = state_runtime_roles,
            .state_descriptor_count = state_descriptor_count,
            .current_position = 0,
        };
        errdefer backend.deinit();
        return backend;
    }

    pub fn deinit(self: *MetalBackend) void {
        if (self.vision_runtime) |*rt| rt.deinit();
        if (self.slot0_state_binding.bound) {
            self.deinitSlotBindingStateObjects(&self.slot0_state_binding);
        }
        self.slot0_state_binding.clear();
        for (self.extra_slots) |*slot| {
            if (slot.state_binding.bound) {
                self.deinitSlotBindingStateObjects(&slot.state_binding);
            }
            slot.state_binding.clear();
        }
        self.allocator.free(self.slot_logits_buffer);
        self.allocator.free(self.extra_slots);
        weights_trait.freeWeights(self.allocator, self.weights);
        self.* = undefined;
    }

    /// Prefill: process all prompt tokens, return logits for last position
    pub fn prefill(self: *MetalBackend, tokens: []const u32, logits_out: []f32) !void {
        // Single-sequence entrypoint uses slot 0.
        try self.ensureSlotStateBlocksBoundForScheduler(0);
        self.slot_in_use = true;
        try self.prefillSlotImpl(0, tokens, logits_out);
    }

    pub fn embed(
        self: *MetalBackend,
        tokens: []const u32,
        pooling: contract.PoolingStrategy,
        normalize: bool,
        embedding_out: []f32,
    ) !void {
        if (tokens.len == 0) return error.EmptyInput;
        if (embedding_out.len < self.d_model) return error.BufferTooSmall;
        try self.ensureSlotStateBlocksBoundForScheduler(0);

        const hidden_handle = try runtime_trait.transformerForwardHiddenLazy(
            self.allocator,
            self.weights,
            tokens,
            try self.slotStateBlocks(0),
            self.config,
            0,
        );
        defer graph.freeArray(hidden_handle);

        graph.eval(&[_]graph.ArrayHandle{hidden_handle});

        const seq_len = tokens.len;
        const hidden_values = try self.allocator.alloc(f32, seq_len * self.d_model);
        defer self.allocator.free(hidden_values);
        graph.copyToHost(hidden_handle, hidden_values);

        const out = embedding_out[0..self.d_model];
        switch (pooling) {
            .last => {
                const last_offset = (seq_len - 1) * self.d_model;
                @memcpy(out, hidden_values[last_offset .. last_offset + self.d_model]);
            },
            .first => {
                @memcpy(out, hidden_values[0..self.d_model]);
            },
            .mean => {
                @memset(out, 0);
                for (0..seq_len) |row_idx| {
                    const row = hidden_values[row_idx * self.d_model ..][0..self.d_model];
                    for (out, row) |*dst, value| {
                        dst.* += value;
                    }
                }
                const inv_n = 1.0 / @as(f32, @floatFromInt(seq_len));
                for (out) |*v| v.* *= inv_n;
            },
        }

        if (normalize) {
            var sum_sq: f64 = 0;
            for (out) |v| {
                sum_sq += @as(f64, v) * @as(f64, v);
            }
            if (sum_sq > 0) {
                const inv_norm = @as(f32, @floatCast(1.0 / @sqrt(sum_sq)));
                for (out) |*v| v.* *= inv_norm;
            }
        }
    }

    pub fn embeddingDim(self: *const MetalBackend) usize {
        return self.d_model;
    }

    /// Allocate scheduler slot.
    pub fn allocSlot(self: *MetalBackend) ?usize {
        if (!self.slot_in_use) {
            self.slot_in_use = true;
            self.resetSlotState(0) catch return null;
            return 0;
        }
        for (self.extra_slots, 0..) |*slot, idx| {
            if (slot.in_use) continue;
            slot.in_use = true;
            self.resetSlotState(idx + 1) catch return null;
            return idx + 1;
        }
        return null;
    }

    /// Release scheduler slot.
    pub fn freeSlot(self: *MetalBackend, slot_index: usize) void {
        if (slot_index == 0) {
            self.unbindSlotStateBlocks(0);
            self.slot_in_use = false;
            self.resetSlotState(0) catch {};
            return;
        }
        const extra_idx = self.toExtraSlotIndex(slot_index) catch return;
        self.unbindSlotStateBlocks(slot_index);
        self.extra_slots[extra_idx].in_use = false;
        self.resetSlotState(slot_index) catch {};
    }

    /// Reset scheduler slot state without releasing ownership.
    pub fn resetSlot(self: *MetalBackend, slot_index: usize) void {
        self.resetSlotState(slot_index) catch {};
    }

    /// Return current decode position for scheduler slot.
    pub fn getPosition(self: *const MetalBackend, slot_index: usize) usize {
        if (slot_index == 0) return self.current_position;
        const extra_idx = self.toExtraSlotIndex(slot_index) catch return 0;
        return self.extra_slots[extra_idx].position;
    }

    pub fn stateDescriptors(self: *const MetalBackend) []const runtime_contract.StateDescriptor {
        return self.state_descriptors_storage[0..self.state_descriptor_count];
    }

    fn slotStateBinding(self: *MetalBackend, slot_index: usize) !*SlotStateBinding {
        if (slot_index == 0) return &self.slot0_state_binding;
        const extra_idx = try self.toExtraSlotIndex(slot_index);
        return &self.extra_slots[extra_idx].state_binding;
    }

    fn slotInUse(self: *const MetalBackend, slot_index: usize) !bool {
        if (slot_index == 0) return self.slot_in_use;
        const extra_idx = try self.toExtraSlotIndex(slot_index);
        return self.extra_slots[extra_idx].in_use;
    }

    fn slotStateBlocks(self: *MetalBackend, slot_index: usize) ![]const runtime_contract.StateBlockHandle {
        const binding = try self.slotStateBinding(slot_index);
        return binding.handles[0..binding.count];
    }

    pub fn ensureSlotStateBlocksBoundForScheduler(self: *MetalBackend, slot_index: usize) !void {
        if (self.state_descriptor_count == 0) return;
        const binding = try self.slotStateBinding(slot_index);
        if (!binding.bound) return error.InvalidStateDescriptorBinding;
        if (comptime std.debug.runtime_safety) {
            try runtime_contract.validateStateBlocksForDescriptors(
                self.stateDescriptors(),
                try self.slotStateBlocks(slot_index),
            );
        }
    }

    pub fn bindSlotStateBlocks(
        self: *MetalBackend,
        slot_index: usize,
        state_blocks: []const runtime_contract.StateBlockHandle,
    ) !void {
        try runtime_contract.validateStateBlocksForDescriptors(self.stateDescriptors(), state_blocks);
        const binding = try self.slotStateBinding(slot_index);
        if (state_blocks.len > binding.handles.len or state_blocks.len > std.math.maxInt(u8)) {
            return error.InvalidStateDescriptorBinding;
        }
        if (state_blocks.len != self.stateDescriptors().len) return error.InvalidStateDescriptorBinding;
        binding.count = @intCast(state_blocks.len);
        for (self.stateDescriptors(), 0..) |descriptor, idx| {
            const incoming = runtime_contract.findStateBlock(state_blocks, descriptor.id) orelse {
                return error.InvalidStateDescriptorBinding;
            };
            if (binding.initialized[idx]) {
                const existing = binding.handles[idx];
                const same_block =
                    existing.id == incoming.id and
                    @intFromPtr(existing.ptr) == @intFromPtr(incoming.ptr) and
                    existing.size == incoming.size and
                    existing.align_bytes == incoming.align_bytes;
                if (!same_block) {
                    try deinitStateObjectForDescriptor(self.state_runtime_roles[idx], &binding.handles[idx]);
                    binding.initialized[idx] = false;
                }
            }
            binding.handles[idx] = .{
                .id = incoming.id,
                .ptr = incoming.ptr,
                .size = incoming.size,
                .align_bytes = incoming.align_bytes,
            };
            if (!binding.initialized[idx]) {
                binding.initialized[idx] = try self.initStateObjectForDescriptor(
                    self.state_runtime_roles[idx],
                    &binding.handles[idx],
                );
            }
        }
        binding.bound = true;
    }

    pub fn unbindSlotStateBlocks(self: *MetalBackend, slot_index: usize) void {
        const binding = self.slotStateBinding(slot_index) catch return;
        if (!binding.bound) return;
        const slot_in_use = self.slotInUse(slot_index) catch false;
        if (slot_in_use) {
            self.releaseNonPersistentStateObjects(binding);
            binding.detach();
            return;
        }
        self.deinitSlotBindingStateObjects(binding);
        binding.clear();
    }

    /// Scheduler prefill entrypoint.
    pub fn prefillSlot(
        self: *MetalBackend,
        slot_index: usize,
        tokens: []const u32,
        logits_out: []f32,
    ) !void {
        try self.ensureSlotStateBlocksBoundForScheduler(slot_index);
        if (slot_index == 0) self.slot_in_use = true else self.extra_slots[try self.toExtraSlotIndex(slot_index)].in_use = true;
        return self.prefillSlotImpl(slot_index, tokens, logits_out);
    }

    /// Scheduler prefill entrypoint with multimodal image payload.
    pub fn prefillSlotWithVision(
        self: *MetalBackend,
        slot_index: usize,
        tokens: []const u32,
        vision_input: ?*const PrefillVisionInput,
        logits_out: []f32,
    ) !void {
        try self.ensureSlotStateBlocksBoundForScheduler(slot_index);
        if (slot_index == 0) self.slot_in_use = true else self.extra_slots[try self.toExtraSlotIndex(slot_index)].in_use = true;
        if (vision_input == null) return self.prefillSlotImpl(slot_index, tokens, logits_out);
        const vi = vision_input.?;

        const sequence_len = tokens.len;
        if (sequence_len == 0) return;
        const slot_position = try self.slotPositionPtr(slot_index);
        const slot_rope_delta = try self.slotRopeDeltaPtr(slot_index);

        const vision = if (self.vision_runtime) |*rt|
            rt
        else {
            log.warn("inference", "Metal vision prefill requested but vision runtime is unavailable", .{
                .vision_hidden_size = self.config.vision_hidden_size,
                .vision_depth = self.config.vision_depth,
                .vision_num_heads = self.config.vision_num_heads,
                .vision_intermediate_size = self.config.vision_intermediate_size,
                .projector_hidden_size = self.config.projector_hidden_size,
                .vision_patch_size = self.config.vision_patch_size,
                .vision_spatial_merge_size = self.config.vision_spatial_merge_size,
                .vision_temporal_patch_size = self.config.vision_temporal_patch_size,
                .vision_num_position_embeddings = self.config.vision_num_position_embeddings,
                .vision_max_num_patches = self.config.vision_max_num_patches,
            });
            return error.UnsupportedContentType;
        };

        var encoded_vision = try vision.encodeImages(vi.images);
        defer encoded_vision.deinit(self.allocator);

        const embedding_handle = try runtime_trait.gatherTokenEmbeddingsLazy(self.weights, tokens);
        defer graph.freeArray(embedding_handle);
        graph.eval(&[_]graph.ArrayHandle{embedding_handle});

        const hidden_values = try self.allocator.alloc(f32, sequence_len * self.d_model);
        defer self.allocator.free(hidden_values);
        graph.copyToHost(embedding_handle, hidden_values);

        try vision.scatterIntoHidden(
            hidden_values,
            sequence_len,
            self.d_model,
            tokens,
            vi.image_token_id,
            encoded_vision.merged_embeddings,
        );
        slot_rope_delta.* = 0;

        var image_token_positions: []usize = &.{};
        defer if (image_token_positions.len > 0) self.allocator.free(image_token_positions);
        var deepstack_ctx: ?runtime_trait.DeepstackAdditions = null;
        if (encoded_vision.deepstack_layer_embeddings.len > 0) {
            image_token_positions = try collectTokenPositions(self.allocator, tokens, vi.image_token_id);
            if (image_token_positions.len == 0) return error.InvalidPromptImageTokens;
            deepstack_ctx = .{
                .positions = image_token_positions,
                .layer_features = encoded_vision.deepstack_layer_embeddings,
            };
        }

        var runtime_rope_cos: []f32 = &.{};
        var runtime_rope_sin: []f32 = &.{};
        defer if (runtime_rope_cos.len > 0) self.allocator.free(runtime_rope_cos);
        defer if (runtime_rope_sin.len > 0) self.allocator.free(runtime_rope_sin);
        var runtime_rope_ctx: ?runtime_trait.RuntimeRoPEOverride = null;
        const head_dim: usize = @intCast(self.config.head_dim);
        const mrope_section = common_mrope.resolveMropeSection(&self.config, head_dim);
        if (mrope_section[0] + mrope_section[1] + mrope_section[2] > 0) {
            const spatial_merge_size = std.math.cast(usize, self.config.vision_spatial_merge_size) orelse return error.InvalidShape;
            const tables = try buildMultimodalMropeTables(
                self.allocator,
                tokens,
                vi.images,
                vi.image_token_id,
                spatial_merge_size,
                head_dim,
                self.config.rope_theta,
                mrope_section,
            );
            runtime_rope_cos = tables.cos;
            runtime_rope_sin = tables.sin;
            slot_rope_delta.* = tables.position_delta;
            runtime_rope_ctx = .{
                .cos = runtime_rope_cos,
                .sin = runtime_rope_sin,
                .dim = head_dim,
            };
            log.debug("inference", "Metal text MRoPE prepared", .{
                .seq_len = sequence_len,
                .head_dim = head_dim,
                .section_t = mrope_section[0],
                .section_h = mrope_section[1],
                .section_w = mrope_section[2],
                .position_delta = slot_rope_delta.*,
            }, @src());
        }

        // Reset selected slot cache for new sequence.
        try self.resetSlotState(slot_index);

        const hidden_handle = try runtime_trait.transformerForwardHiddenLazyWithEmbeddingOverride(
            self.allocator,
            self.weights,
            tokens,
            try self.slotStateBlocks(slot_index),
            self.config,
            0,
            hidden_values,
            deepstack_ctx,
            runtime_rope_ctx,
        );
        defer graph.freeArray(hidden_handle);
        var starts: [3]c_int = .{ 0, @intCast(sequence_len - 1), 0 };
        var ends: [3]c_int = .{ 1, @intCast(sequence_len), @intCast(self.d_model) };
        const last_hidden_handle = graph.mlx_lazy_slice(hidden_handle, &starts, &ends, 3);
        defer graph.freeArray(last_hidden_handle);
        const logits_handle = metal_executor.block.TransformerBlock.projectLogits(
            last_hidden_handle,
            self.weights,
            self.config.norm_eps,
        );
        defer graph.freeArray(logits_handle);
        graph.eval(&[_]graph.ArrayHandle{logits_handle});

        var shape_buffer: [8]usize = undefined;
        const rank = graph.getShape(logits_handle, &shape_buffer);
        std.debug.assert(rank >= 1);
        std.debug.assert(shape_buffer[rank - 1] == self.vocab_size);
        if (rank == 2) std.debug.assert(shape_buffer[0] == 1);
        graph.copyToHost(logits_handle, logits_out[0..self.vocab_size]);

        slot_position.* = sequence_len;
    }

    /// Scheduler decode entrypoint.
    pub fn decodeBatch(self: *MetalBackend, requests: []const contract.DecodeRequest, results: []contract.DecodeResult) !void {
        if (requests.len == 0) return;
        if (results.len < requests.len) return error.InvalidArgument;
        if (requests.len > self.max_batch_size) return error.InvalidArgument;
        const seen_slots = try self.allocator.alloc(bool, self.max_batch_size);
        defer self.allocator.free(seen_slots);
        @memset(seen_slots, false);

        var has_duplicate_slot = false;
        for (requests) |request| {
            if (request.slot_index >= self.max_batch_size) return error.InvalidArgument;
            try self.ensureSlotStateBlocksBoundForScheduler(request.slot_index);
            if (seen_slots[request.slot_index]) {
                has_duplicate_slot = true;
                break;
            }
            seen_slots[request.slot_index] = true;
        }

        if (has_duplicate_slot) return error.InvalidArgument;

        for (requests, 0..) |request, idx| {
            const logits = try self.slotLogits(request.slot_index);
            const slot_position = try self.slotPositionPtr(request.slot_index);
            try self.decodeSlot(request.slot_index, request.token, slot_position.*, logits);
            results[idx] = .{
                .slot_index = request.slot_index,
                .logits = logits,
            };
        }
    }

    /// Decode: generate logits for a single token using KV cache
    pub fn decode(self: *MetalBackend, token: u32, position: usize, logits_out: []f32) !void {
        // Single-sequence entrypoint uses slot 0.
        try self.ensureSlotStateBlocksBoundForScheduler(0);
        self.slot_in_use = true;
        try self.decodeSlot(0, token, position, logits_out);
    }

    /// Decode with streaming — token-by-token loop through adapter table path.
    pub fn decodeStreaming(
        self: *MetalBackend,
        first_token: u32,
        start_position: usize,
        max_tokens: usize,
        eos_token_ids: []const u32,
        output_tokens: []u32,
        callback: ?*const fn (u32, ?*anyopaque) void,
        callback_data: ?*anyopaque,
    ) !usize {
        if (max_tokens == 0 or output_tokens.len == 0) return 0;
        try self.ensureSlotStateBlocksBoundForScheduler(0);
        self.slot_in_use = true;

        var decode_timer = std.time.Timer.start() catch unreachable;
        const budget = @min(max_tokens, output_tokens.len);
        var current_token = first_token;
        var current_position = start_position;
        var generated_count: usize = 0;
        while (generated_count < budget) {
            const next_token = try self.decodeSlotGreedy(0, current_token, current_position);
            output_tokens[generated_count] = next_token;
            generated_count += 1;
            current_position += 1;

            if (callback) |cb| cb(next_token, callback_data);

            var is_eos = false;
            for (eos_token_ids) |eos_id| {
                if (next_token == eos_id) {
                    is_eos = true;
                    break;
                }
            }
            if (is_eos) break;

            current_token = next_token;
        }

        const decode_ns = decode_timer.read();
        log.debug("inference", "Metal decode route complete", .{
            .route = "adapter_table_streaming",
            .requested_tokens = budget,
            .generated_tokens = generated_count,
            .duration_ms = @as(f64, @floatFromInt(decode_ns)) / 1_000_000.0,
            .tok_per_sec = if (decode_ns == 0)
                @as(f64, 0.0)
            else
                @as(f64, @floatFromInt(generated_count)) * 1_000_000_000.0 / @as(f64, @floatFromInt(decode_ns)),
            .has_callback = @as(u8, @intFromBool(callback != null)),
            .eos_count = eos_token_ids.len,
        }, @src());
        const slot_position = try self.slotPositionPtr(0);
        slot_position.* = start_position + generated_count;
        self.current_position = slot_position.*;
        return generated_count;
    }

    fn collectTokenPositions(
        allocator: std.mem.Allocator,
        token_ids: []const u32,
        needle: u32,
    ) ![]usize {
        var count: usize = 0;
        for (token_ids) |token| {
            if (token == needle) count += 1;
        }
        if (count == 0) return &.{};

        const positions = try allocator.alloc(usize, count);
        errdefer allocator.free(positions);

        var write_idx: usize = 0;
        for (token_ids, 0..) |token, idx| {
            if (token != needle) continue;
            positions[write_idx] = idx;
            write_idx += 1;
        }
        std.debug.assert(write_idx == count);
        return positions;
    }

    const MropeTables = struct {
        cos: []f32,
        sin: []f32,
        position_delta: isize,
    };

    fn buildMultimodalMropeTables(
        allocator: std.mem.Allocator,
        tokens: []const u32,
        images: []const vision_runtime_mod.PrefillVisionImage,
        image_token_id: u32,
        spatial_merge_size: usize,
        head_dim: usize,
        rope_theta: f32,
        mrope_section: [3]usize,
    ) !MropeTables {
        if (tokens.len == 0) return .{ .cos = &.{}, .sin = &.{}, .position_delta = 0 };
        if ((head_dim % 2) != 0 or mrope_section[0] + mrope_section[1] + mrope_section[2] != head_dim / 2) {
            return error.InvalidShape;
        }

        const seq_len = tokens.len;
        const pos_t = try allocator.alloc(u32, seq_len);
        defer allocator.free(pos_t);
        const pos_h = try allocator.alloc(u32, seq_len);
        defer allocator.free(pos_h);
        const pos_w = try allocator.alloc(u32, seq_len);
        defer allocator.free(pos_w);

        try common_mrope.buildMultimodalMropePositions(
            tokens,
            images,
            image_token_id,
            spatial_merge_size,
            pos_t,
            pos_h,
            pos_w,
        );

        const half_dim = head_dim / 2;
        const inv_freq = try allocator.alloc(f32, half_dim);
        defer allocator.free(inv_freq);
        for (0..half_dim) |idx| {
            const exponent = @as(f32, @floatFromInt(2 * idx)) / @as(f32, @floatFromInt(head_dim));
            inv_freq[idx] = 1.0 / std.math.pow(f32, rope_theta, exponent);
        }

        const cos = try allocator.alloc(f32, seq_len * head_dim);
        errdefer allocator.free(cos);
        const sin = try allocator.alloc(f32, seq_len * head_dim);
        errdefer allocator.free(sin);

        const h_limit = mrope_section[1] * 3;
        const w_limit = mrope_section[2] * 3;
        for (0..seq_len) |token_idx| {
            const base = token_idx * head_dim;
            for (0..half_dim) |freq_idx| {
                var pos_component = pos_t[token_idx];
                if (freq_idx < h_limit and (freq_idx % 3) == 1) pos_component = pos_h[token_idx];
                if (freq_idx < w_limit and (freq_idx % 3) == 2) pos_component = pos_w[token_idx];

                const angle = @as(f32, @floatFromInt(pos_component)) * inv_freq[freq_idx];
                const c = @cos(angle);
                const s = @sin(angle);
                cos[base + freq_idx] = c;
                sin[base + freq_idx] = s;
                cos[base + half_dim + freq_idx] = c;
                sin[base + half_dim + freq_idx] = s;
            }
        }

        const position_delta = try common_mrope.computePositionDelta(pos_t, pos_h, pos_w);

        return .{
            .cos = cos,
            .sin = sin,
            .position_delta = position_delta,
        };
    }
};

test "extraSlotIndexFor maps scheduler slots to extra-slot storage" {
    try std.testing.expectEqual(@as(usize, 0), try MetalBackend.extraSlotIndexFor(1, 4));
    try std.testing.expectEqual(@as(usize, 2), try MetalBackend.extraSlotIndexFor(3, 4));
}

test "extraSlotIndexFor rejects slot 0 and out-of-range slots" {
    try std.testing.expectError(error.InvalidArgument, MetalBackend.extraSlotIndexFor(0, 4));
    try std.testing.expectError(error.InvalidArgument, MetalBackend.extraSlotIndexFor(4, 4));
}

test "argmaxHost and argminHost select expected indices" {
    const values = [_]f32{ 0.5, -4.0, 1.25, 0.8 };
    try std.testing.expectEqual(@as(u32, 2), MetalBackend.argmaxHost(values[0..]));
    try std.testing.expectEqual(@as(u32, 1), MetalBackend.argminHost(values[0..]));
}

test "resolveCacheMaxSeqLen uses dynamic growth for very large contexts" {
    try std.testing.expectEqual(@as(usize, 0), MetalBackend.resolveCacheMaxSeqLen(262_144));
    try std.testing.expectEqual(@as(usize, 40_960), MetalBackend.resolveCacheMaxSeqLen(40_960));
}

test "applyPositionDelta rejects negative resulting positions" {
    try std.testing.expectError(error.InvalidShape, common_mrope.applyPositionDelta(2, -3));
    try std.testing.expectEqual(@as(usize, 7), try common_mrope.applyPositionDelta(4, 3));
}

test "deriveStateRuntimeRoles maps descriptor runtime_kind values" {
    const descriptors = [_]runtime_contract.StateDescriptor{
        .{
            .id = runtime_contract.kv_cache_state_id,
            .size_bytes = @sizeOf(runtime_graph_mod.Cache),
            .align_bytes = @alignOf(runtime_graph_mod.Cache),
            .zero_init = false,
            .lifecycle = .slot_persistent,
            .runtime_kind = runtime_contract.state_runtime_kind_kv_cache,
        },
        .{
            .id = runtime_contract.shortconv_state_id,
            .size_bytes = @sizeOf(runtime_graph_mod.ShortConvCache),
            .align_bytes = @alignOf(runtime_graph_mod.ShortConvCache),
            .zero_init = false,
            .lifecycle = .slot_persistent,
            .runtime_kind = runtime_contract.state_runtime_kind_shortconv_cache,
        },
        .{
            .id = runtime_contract.mamba_state_id,
            .size_bytes = @sizeOf(runtime_graph_mod.MambaCache),
            .align_bytes = @alignOf(runtime_graph_mod.MambaCache),
            .zero_init = false,
            .lifecycle = .slot_persistent,
            .runtime_kind = runtime_contract.state_runtime_kind_mamba_cache,
        },
        .{
            .id = runtime_contract.gated_delta_state_id,
            .size_bytes = @sizeOf(runtime_graph_mod.GatedDeltaCache),
            .align_bytes = @alignOf(runtime_graph_mod.GatedDeltaCache),
            .zero_init = false,
            .lifecycle = .slot_persistent,
            .runtime_kind = runtime_contract.state_runtime_kind_gated_delta_cache,
        },
    };

    const roles = try MetalBackend.deriveStateRuntimeRoles(descriptors[0..]);
    try std.testing.expectEqual(MetalBackend.StateRuntimeRole.kv_cache, roles[0]);
    try std.testing.expectEqual(MetalBackend.StateRuntimeRole.shortconv_cache, roles[1]);
    try std.testing.expectEqual(MetalBackend.StateRuntimeRole.mamba_cache, roles[2]);
    try std.testing.expectEqual(MetalBackend.StateRuntimeRole.gated_delta_cache, roles[3]);
}

test "deriveStateRuntimeRoles rejects unsupported runtime_kind" {
    const descriptors = [_]runtime_contract.StateDescriptor{
        .{
            .id = 77,
            .size_bytes = 64,
            .align_bytes = 8,
            .zero_init = false,
            .lifecycle = .request_scoped,
            .runtime_kind = 255,
        },
    };

    try std.testing.expectError(
        error.InvalidStateDescriptorBinding,
        MetalBackend.deriveStateRuntimeRoles(descriptors[0..]),
    );
}

test "buildMultimodalMropeTables computes multimodal decode position_delta" {
    const allocator = std.testing.allocator;
    const image_token_id: u32 = 151655;
    const vision_start_token_id: u32 = 151652;
    const vision_end_token_id: u32 = 151653;
    const seq_len: usize = 198;
    const image_span: usize = 169;
    const image_start: usize = 15;

    const tokens = try allocator.alloc(u32, seq_len);
    defer allocator.free(tokens);
    @memset(tokens, 42);
    tokens[image_start - 1] = vision_start_token_id;
    for (0..image_span) |idx| tokens[image_start + idx] = image_token_id;
    tokens[image_start + image_span] = vision_end_token_id;

    const images = [_]vision_runtime_mod.PrefillVisionImage{
        .{
            .pixels = &.{},
            .width = 416,
            .height = 416,
            .grid = .{ .temporal = 1, .height = 26, .width = 26 },
            .token_count = image_span,
        },
    };

    const tables = try MetalBackend.buildMultimodalMropeTables(
        allocator,
        tokens,
        images[0..],
        image_token_id,
        2,
        128,
        5_000_000.0,
        .{ 24, 20, 20 },
    );
    defer allocator.free(tables.cos);
    defer allocator.free(tables.sin);

    try std.testing.expectEqual(seq_len * 128, tables.cos.len);
    try std.testing.expectEqual(seq_len * 128, tables.sin.len);
    try std.testing.expectEqual(@as(isize, -156), tables.position_delta);
}

test "decodeBatch advances positions across multiple slots" {
    if (!isAvailable()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const loaded = try weights_trait.createTestLoadedModel(allocator);
    defer weights_trait.destroyTestLoadedModel(allocator, loaded);

    var backend = try MetalBackend.init(allocator, loaded);
    defer backend.deinit();

    const slot0 = backend.allocSlot() orelse return error.TestUnexpectedResult;
    const slot1 = backend.allocSlot() orelse return error.TestUnexpectedResult;
    defer backend.freeSlot(slot1);
    defer backend.freeSlot(slot0);

    try std.testing.expectEqual(@as(usize, 0), slot0);
    try std.testing.expectEqual(@as(usize, 1), slot1);

    const vocab_size = backend.vocabSize();
    const prefill_logits0 = try allocator.alloc(f32, vocab_size);
    defer allocator.free(prefill_logits0);
    const prefill_logits1 = try allocator.alloc(f32, vocab_size);
    defer allocator.free(prefill_logits1);

    try backend.prefillSlot(slot0, &[_]u32{ 1, 2 }, prefill_logits0);
    try backend.prefillSlot(slot1, &[_]u32{ 3, 4 }, prefill_logits1);
    try std.testing.expectEqual(@as(usize, 2), backend.getPosition(slot0));
    try std.testing.expectEqual(@as(usize, 2), backend.getPosition(slot1));

    const requests = [_]contract.DecodeRequest{
        .{ .slot_index = slot0, .token = 5 },
        .{ .slot_index = slot1, .token = 6 },
    };
    var results: [2]contract.DecodeResult = undefined;
    try backend.decodeBatch(requests[0..], results[0..]);

    try std.testing.expectEqual(slot0, results[0].slot_index);
    try std.testing.expectEqual(slot1, results[1].slot_index);
    try std.testing.expectEqual(vocab_size, results[0].logits.len);
    try std.testing.expectEqual(vocab_size, results[1].logits.len);
    try std.testing.expect(results[0].logits.ptr != results[1].logits.ptr);
    try std.testing.expectEqual(@as(usize, 3), backend.getPosition(slot0));
    try std.testing.expectEqual(@as(usize, 3), backend.getPosition(slot1));

    const duplicate_requests = [_]contract.DecodeRequest{
        .{ .slot_index = slot0, .token = 7 },
        .{ .slot_index = slot0, .token = 8 },
    };
    var duplicate_results: [2]contract.DecodeResult = undefined;
    try std.testing.expectError(error.InvalidArgument, backend.decodeBatch(duplicate_requests[0..], duplicate_results[0..]));
    try std.testing.expectEqual(@as(usize, 3), backend.getPosition(slot0));
    try std.testing.expectEqual(@as(usize, 3), backend.getPosition(slot1));
}

test "decodeStreaming greedy token matches decode logits argmax" {
    if (!isAvailable()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const loaded = try weights_trait.createTestLoadedModel(allocator);
    defer weights_trait.destroyTestLoadedModel(allocator, loaded);

    var backend = try MetalBackend.init(allocator, loaded);
    defer backend.deinit();

    const vocab_size = backend.vocabSize();
    const logits = try allocator.alloc(f32, vocab_size);
    defer allocator.free(logits);

    try backend.prefill(&[_]u32{ 1, 2 }, logits);
    const position = backend.getPosition(0);
    try backend.decode(3, position, logits);
    const expected = try backend.selectNextTokenFromLogits(logits);

    var output: [1]u32 = undefined;
    const generated = try backend.decodeStreaming(3, position, 1, &.{}, output[0..], null, null);

    try std.testing.expectEqual(@as(usize, 1), generated);
    try std.testing.expectEqual(expected, output[0]);
}

test "metal availability" {
    if (!isAvailable()) return error.SkipZigTest;
}

test "metal device creation" {
    if (!isAvailable()) return error.SkipZigTest;

    var metal_device = try Device.init();
    defer metal_device.deinit();

    const device_name = metal_device.name();
    try std.testing.expect(device_name.len > 0);
}

test "metal buffer allocation" {
    if (!isAvailable()) return error.SkipZigTest;

    var metal_device = try Device.init();
    defer metal_device.deinit();

    var metal_buffer = try metal_device.allocBuffer(1024);
    defer metal_buffer.deinit();

    const test_data = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 };
    metal_buffer.upload(&test_data);

    var result = [_]u8{0} ** 8;
    metal_buffer.download(&result);
    try std.testing.expectEqualSlices(u8, &test_data, &result);
}

test "metal f32 matmul" {
    if (!isAvailable()) return error.SkipZigTest;

    var metal_device = try Device.init();
    defer metal_device.deinit();

    const left_matrix = [_]f32{
        1, 2, 3,
        4, 5, 6,
    };
    const right_matrix = [_]f32{
        1, 0,
        0, 1,
        1, 1,
    };
    var output_matrix = [_]f32{0} ** 4;

    try matmul.matmulF32(
        &metal_device,
        &left_matrix,
        2,
        3,
        &right_matrix,
        2,
        &output_matrix,
    );

    const expected = [_]f32{ 4, 5, 10, 11 };
    for (output_matrix, expected) |actual, expected_value| {
        try std.testing.expectApproxEqAbs(expected_value, actual, 0.001);
    }
}

test {
    @import("std").testing.refAllDecls(@This());
}
