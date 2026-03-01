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
pub const WeightHandles = weights_trait.WeightHandles;
/// Metal backend for GPU-accelerated transformer inference
pub const MetalBackend = struct {
    const MaxStateDescriptors: usize = std.math.maxInt(u8);
    const ImplicitStateBlockBytes: usize = 64;

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
    /// Slot 0 state (single-sequence compatibility path).
    cache: runtime_graph_mod.Cache,
    shortconv_cache: runtime_graph_mod.ShortConvCache,
    mamba_cache: runtime_graph_mod.MambaCache,
    vocab_size: usize,
    d_model: usize,
    state_flags: runtime_contract.BuiltinStateFlags,
    max_batch_size: usize,
    slot_in_use: bool,
    /// Slots 1..N-1 for scheduler-managed multi-slot decode.
    extra_slots: []SlotState,
    slot_logits_buffer: []f32,
    vision_runtime: ?vision_runtime_mod.VisionRuntime = null,
    slot_rope_position_delta: isize,
    state_descriptors_storage: [MaxStateDescriptors]runtime_contract.StateDescriptor,
    state_descriptor_count: u8,
    slot0_state_binding: SlotStateBinding = .{},
    implicit_slot0_state_storage: [MaxStateDescriptors][ImplicitStateBlockBytes]u8 align(64) =
        [_][ImplicitStateBlockBytes]u8{[_]u8{0} ** ImplicitStateBlockBytes} ** MaxStateDescriptors,

    // Track position for decode
    current_position: usize,

    const SlotState = struct {
        cache: runtime_graph_mod.Cache,
        shortconv_cache: runtime_graph_mod.ShortConvCache,
        mamba_cache: runtime_graph_mod.MambaCache,
        state_binding: SlotStateBinding = .{},
        in_use: bool = false,
        position: usize = 0,
        rope_position_delta: isize = 0,
    };

    const SlotStateBinding = struct {
        handles: [MaxStateDescriptors]runtime_contract.StateBlockHandle = undefined,
        count: u8 = 0,
        bound: bool = false,

        fn reset(self: *SlotStateBinding) void {
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

    fn slotCache(self: *MetalBackend, slot_index: usize) !*runtime_graph_mod.Cache {
        if (slot_index == 0) return &self.cache;
        const extra_idx = try self.toExtraSlotIndex(slot_index);
        return &self.extra_slots[extra_idx].cache;
    }

    fn slotShortConvCache(self: *MetalBackend, slot_index: usize) !*runtime_graph_mod.ShortConvCache {
        if (slot_index == 0) return &self.shortconv_cache;
        const extra_idx = try self.toExtraSlotIndex(slot_index);
        return &self.extra_slots[extra_idx].shortconv_cache;
    }

    fn slotMambaCache(self: *MetalBackend, slot_index: usize) !*runtime_graph_mod.MambaCache {
        if (slot_index == 0) return &self.mamba_cache;
        const extra_idx = try self.toExtraSlotIndex(slot_index);
        return &self.extra_slots[extra_idx].mamba_cache;
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

    const BoundSlotState = struct {
        cache: *runtime_graph_mod.Cache,
        shortconv_cache: *runtime_graph_mod.ShortConvCache,
        mamba_cache: *runtime_graph_mod.MambaCache,
    };

    fn boundSlotState(self: *MetalBackend, slot_index: usize) !BoundSlotState {
        const slot_blocks = try self.slotStateBlocks(slot_index);
        var slot_cache = try self.slotCache(slot_index);
        var slot_shortconv_cache = try self.slotShortConvCache(slot_index);
        var slot_mamba_cache = try self.slotMambaCache(slot_index);

        if (self.state_flags.has_kv) {
            slot_cache = runtime_contract.findStateValue(
                *runtime_graph_mod.Cache,
                slot_blocks,
                @intFromEnum(runtime_contract.StateBlockId.kv_cache),
            ) orelse return error.InvalidStateDescriptorBinding;
        }
        if (self.state_flags.has_shortconv) {
            slot_shortconv_cache = runtime_contract.findStateValue(
                *runtime_graph_mod.ShortConvCache,
                slot_blocks,
                @intFromEnum(runtime_contract.StateBlockId.shortconv),
            ) orelse return error.InvalidStateDescriptorBinding;
        }
        if (self.state_flags.has_mamba) {
            slot_mamba_cache = runtime_contract.findStateValue(
                *runtime_graph_mod.MambaCache,
                slot_blocks,
                @intFromEnum(runtime_contract.StateBlockId.mamba),
            ) orelse return error.InvalidStateDescriptorBinding;
        }

        return .{
            .cache = slot_cache,
            .shortconv_cache = slot_shortconv_cache,
            .mamba_cache = slot_mamba_cache,
        };
    }

    fn resetSlotState(self: *MetalBackend, slot_index: usize) !void {
        const slot_cache = try self.slotCache(slot_index);
        const slot_shortconv_cache = try self.slotShortConvCache(slot_index);
        const slot_mamba_cache = try self.slotMambaCache(slot_index);
        const slot_position = try self.slotPositionPtr(slot_index);
        const slot_rope_delta = try self.slotRopeDeltaPtr(slot_index);

        if (self.state_flags.has_kv) {
            slot_cache.deinit();
            slot_cache.* = runtime_graph_mod.Cache.init(
                @intCast(self.config.n_layers),
                true,
                self.cacheMaxSeqLen(),
            );
        } else {
            slot_cache.* = runtime_graph_mod.Cache.disabled(true);
        }
        if (self.state_flags.has_shortconv) slot_shortconv_cache.reset();
        if (self.state_flags.has_mamba) slot_mamba_cache.reset();
        slot_position.* = 0;
        slot_rope_delta.* = 0;
    }

    fn prefillSlotImpl(self: *MetalBackend, slot_index: usize, tokens: []const u32, logits_out: []f32) !void {
        const sequence_len = tokens.len;
        if (sequence_len == 0) return;

        const slot_state = try self.boundSlotState(slot_index);
        const slot_cache = slot_state.cache;
        const slot_shortconv_cache = slot_state.shortconv_cache;
        const slot_mamba_cache = slot_state.mamba_cache;
        const slot_position = try self.slotPositionPtr(slot_index);
        const slot_rope_delta = try self.slotRopeDeltaPtr(slot_index);
        slot_rope_delta.* = 0;

        // Reset cache for new sequence in this scheduler slot.
        if (self.state_flags.has_kv) {
            slot_cache.deinit();
            slot_cache.* = runtime_graph_mod.Cache.init(
                @intCast(self.config.n_layers),
                true,
                self.cacheMaxSeqLen(),
            );
        } else {
            slot_cache.* = runtime_graph_mod.Cache.disabled(true);
        }
        if (self.state_flags.has_shortconv) slot_shortconv_cache.reset();
        if (self.state_flags.has_mamba) slot_mamba_cache.reset();

        const logits_handle = try runtime_trait.transformerForwardLazy(
            self.allocator,
            self.weights,
            tokens,
            try self.slotStateBlocks(slot_index),
            self.config,
            0, // pos_offset
        );

        graph.eval(&[_]graph.ArrayHandle{logits_handle});

        var shape_buffer: [8]usize = undefined;
        const rank = graph.getShape(logits_handle, &shape_buffer);
        std.debug.assert(rank == 3);
        std.debug.assert(shape_buffer[0] == 1);
        std.debug.assert(shape_buffer[1] == sequence_len);
        std.debug.assert(shape_buffer[2] == self.vocab_size);

        const logits_values = try self.allocator.alloc(f32, sequence_len * self.vocab_size);
        defer self.allocator.free(logits_values);
        graph.copyToHost(logits_handle, logits_values);

        const last_token_offset = (sequence_len - 1) * self.vocab_size;
        @memcpy(logits_out, logits_values[last_token_offset .. last_token_offset + self.vocab_size]);
        graph.freeArray(logits_handle);

        slot_position.* = sequence_len;
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

        // Initialize slot 0 cache (single-sequence compatibility path).
        const layer_count: usize = @intCast(loaded.config.n_layers);
        const max_seq_len: usize = @intCast(loaded.config.max_seq_len);
        const cache_max_seq_len = resolveCacheMaxSeqLen(max_seq_len);
        log.debug("inference", "Metal cache capacity policy", .{
            .config_max_seq_len = max_seq_len,
            .cache_max_seq_len = cache_max_seq_len,
            .dynamic_growth = @as(u8, @intFromBool(cache_max_seq_len == 0)),
        }, @src());
        const state_flags = weight_handles.state_flags;
        const has_kv_state = state_flags.has_kv;
        const has_shortconv_state = state_flags.has_shortconv;
        const has_mamba_state = state_flags.has_mamba;
        const kv_cache = if (has_kv_state)
            runtime_graph_mod.Cache.init(layer_count, true, cache_max_seq_len)
        else
            runtime_graph_mod.Cache.disabled(true);
        const shortconv_cache = if (has_shortconv_state)
            runtime_graph_mod.ShortConvCache.init(layer_count)
        else
            runtime_graph_mod.ShortConvCache.disabled();
        const mamba_cache = if (has_mamba_state)
            runtime_graph_mod.MambaCache.init(layer_count)
        else
            runtime_graph_mod.MambaCache.disabled();
        errdefer mamba_cache.deinit();
        var vision_runtime = try vision_runtime_mod.VisionRuntime.init(allocator, loaded);
        errdefer if (vision_runtime) |*rt| rt.deinit();

        // Scheduler-visible slot capacity for Metal backend.
        const max_batch_size: usize = resolveMaxBatchSize();
        const extra_slot_count = max_batch_size - 1;
        const extra_slots = try allocator.alloc(SlotState, extra_slot_count);
        errdefer allocator.free(extra_slots);
        for (extra_slots) |*slot| {
            slot.* = .{
                .cache = if (has_kv_state)
                    runtime_graph_mod.Cache.init(layer_count, true, cache_max_seq_len)
                else
                    runtime_graph_mod.Cache.disabled(true),
                .shortconv_cache = if (has_shortconv_state)
                    runtime_graph_mod.ShortConvCache.init(layer_count)
                else
                    runtime_graph_mod.ShortConvCache.disabled(),
                .mamba_cache = if (has_mamba_state)
                    runtime_graph_mod.MambaCache.init(layer_count)
                else
                    runtime_graph_mod.MambaCache.disabled(),
                .in_use = false,
                .position = 0,
                .rope_position_delta = 0,
            };
        }
        errdefer for (extra_slots) |slot| {
            slot.cache.deinit();
            slot.shortconv_cache.deinit();
            slot.mamba_cache.deinit();
        };

        const slot_logits_buffer = try allocator.alloc(f32, max_batch_size * @as(usize, @intCast(loaded.config.vocab_size)));
        errdefer allocator.free(slot_logits_buffer);

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
        const descriptor_slice = state_descriptors_storage[0..state_descriptor_count];
        const has_kv_descriptor = runtime_contract.stateDescriptorIndex(
            descriptor_slice,
            @intFromEnum(runtime_contract.StateBlockId.kv_cache),
        ) != null;
        const has_shortconv_descriptor = runtime_contract.stateDescriptorIndex(
            descriptor_slice,
            @intFromEnum(runtime_contract.StateBlockId.shortconv),
        ) != null;
        const has_mamba_descriptor = runtime_contract.stateDescriptorIndex(
            descriptor_slice,
            @intFromEnum(runtime_contract.StateBlockId.mamba),
        ) != null;
        if (has_kv_descriptor != state_flags.has_kv or
            has_shortconv_descriptor != state_flags.has_shortconv or
            has_mamba_descriptor != state_flags.has_mamba)
        {
            return error.InvalidStateDescriptorBinding;
        }

        return MetalBackend{
            .allocator = allocator,
            .config = loaded.config,
            .weights = weight_handles,
            .cache = kv_cache,
            .shortconv_cache = shortconv_cache,
            .mamba_cache = mamba_cache,
            .vocab_size = @intCast(loaded.config.vocab_size),
            .d_model = @intCast(loaded.config.d_model),
            .state_flags = state_flags,
            .max_batch_size = max_batch_size,
            .slot_in_use = false,
            .extra_slots = extra_slots,
            .slot_logits_buffer = slot_logits_buffer,
            .vision_runtime = vision_runtime,
            .slot_rope_position_delta = 0,
            .state_descriptors_storage = state_descriptors_storage,
            .state_descriptor_count = state_descriptor_count,
            .current_position = 0,
        };
    }

    pub fn deinit(self: *MetalBackend) void {
        if (self.vision_runtime) |*rt| rt.deinit();
        self.allocator.free(self.slot_logits_buffer);
        for (self.extra_slots) |slot| {
            slot.cache.deinit();
            slot.shortconv_cache.deinit();
            slot.mamba_cache.deinit();
        }
        self.allocator.free(self.extra_slots);
        self.cache.deinit();
        self.shortconv_cache.deinit();
        self.mamba_cache.deinit();
        weights_trait.freeWeights(self.allocator, self.weights);
        self.* = undefined;
    }

    /// Prefill: process all prompt tokens, return logits for last position
    pub fn prefill(self: *MetalBackend, tokens: []const u32, logits_out: []f32) !void {
        // Single-sequence compatibility path always uses slot 0.
        try self.bindImplicitSlot0StateBlocks();
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
        try self.bindImplicitSlot0StateBlocks();

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
            self.slot0_state_binding.reset();
            self.resetSlotState(0) catch return null;
            return 0;
        }
        for (self.extra_slots, 0..) |*slot, idx| {
            if (slot.in_use) continue;
            slot.in_use = true;
            slot.state_binding.reset();
            self.resetSlotState(idx + 1) catch return null;
            return idx + 1;
        }
        return null;
    }

    /// Release scheduler slot.
    pub fn freeSlot(self: *MetalBackend, slot_index: usize) void {
        if (slot_index == 0) {
            self.slot_in_use = false;
            self.slot0_state_binding.reset();
            self.resetSlotState(0) catch {};
            return;
        }
        const extra_idx = self.toExtraSlotIndex(slot_index) catch return;
        self.extra_slots[extra_idx].in_use = false;
        self.extra_slots[extra_idx].state_binding.reset();
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

    fn slotStateBlocks(self: *MetalBackend, slot_index: usize) ![]const runtime_contract.StateBlockHandle {
        const binding = try self.slotStateBinding(slot_index);
        return binding.handles[0..binding.count];
    }

    pub fn ensureSlotStateBlocksBoundForScheduler(self: *MetalBackend, slot_index: usize) !void {
        if (self.state_descriptor_count == 0) return;
        const binding = try self.slotStateBinding(slot_index);
        if (!binding.bound) return error.InvalidStateDescriptorBinding;
        try runtime_contract.validateStateBlocksForDescriptors(
            self.stateDescriptors(),
            try self.slotStateBlocks(slot_index),
        );
    }

    fn bindImplicitSlot0StateBlocks(self: *MetalBackend) !void {
        if (self.state_descriptor_count == 0 or self.slot0_state_binding.bound) return;
        const descriptors = self.stateDescriptors();
        var handles: [MaxStateDescriptors]runtime_contract.StateBlockHandle = undefined;
        for (descriptors, 0..) |descriptor, idx| {
            const size_bytes = if (descriptor.size_bytes == 0)
                @as(u64, @sizeOf(runtime_contract.OpaqueStateRef))
            else
                descriptor.size_bytes;
            const implicit_capacity = @as(u64, @intCast(self.implicit_slot0_state_storage[idx].len));
            if (size_bytes > implicit_capacity) return error.InvalidStateDescriptorBinding;
            if (descriptor.align_bytes == 0 or descriptor.align_bytes > 64) return error.InvalidStateDescriptorBinding;
            handles[idx] = .{
                .id = descriptor.id,
                .ptr = @ptrCast(&self.implicit_slot0_state_storage[idx][0]),
                .size = size_bytes,
                .align_bytes = 64,
            };
        }
        try self.bindSlotStateBlocks(0, handles[0..descriptors.len]);
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
        const slot_cache = try self.slotCache(slot_index);
        const slot_shortconv_cache = try self.slotShortConvCache(slot_index);
        const slot_mamba_cache = try self.slotMambaCache(slot_index);
        for (self.stateDescriptors(), 0..) |descriptor, idx| {
            const incoming = runtime_contract.findStateBlock(state_blocks, descriptor.id) orelse {
                return error.InvalidStateDescriptorBinding;
            };
            if (incoming.align_bytes < @alignOf(runtime_contract.OpaqueStateRef)) return error.InvalidStateDescriptorBinding;
            if (incoming.size < @sizeOf(runtime_contract.OpaqueStateRef)) return error.InvalidStateDescriptorBinding;
            const state_ptr: *anyopaque = switch (descriptor.id) {
                @intFromEnum(runtime_contract.StateBlockId.kv_cache) => @ptrCast(slot_cache),
                @intFromEnum(runtime_contract.StateBlockId.shortconv) => @ptrCast(slot_shortconv_cache),
                @intFromEnum(runtime_contract.StateBlockId.mamba) => @ptrCast(slot_mamba_cache),
                else => return error.InvalidStateDescriptorBinding,
            };
            const state_ref: *runtime_contract.OpaqueStateRef = @ptrCast(@alignCast(incoming.ptr));
            state_ref.* = .{ .ptr = state_ptr };
            binding.handles[idx] = .{
                .id = descriptor.id,
                .ptr = incoming.ptr,
                .size = if (descriptor.size_bytes == 0) incoming.size else descriptor.size_bytes,
                .align_bytes = incoming.align_bytes,
            };
        }
        binding.count = @intCast(state_blocks.len);
        binding.bound = true;
    }

    pub fn unbindSlotStateBlocks(self: *MetalBackend, slot_index: usize) void {
        const binding = self.slotStateBinding(slot_index) catch return;
        binding.reset();
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
        const slot_state = try self.boundSlotState(slot_index);
        const slot_cache = slot_state.cache;
        const slot_shortconv_cache = slot_state.shortconv_cache;
        const slot_mamba_cache = slot_state.mamba_cache;
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
        if (self.state_flags.has_kv) {
            slot_cache.deinit();
            slot_cache.* = runtime_graph_mod.Cache.init(
                @intCast(self.config.n_layers),
                true,
                self.cacheMaxSeqLen(),
            );
        } else {
            slot_cache.* = runtime_graph_mod.Cache.disabled(true);
        }
        if (self.state_flags.has_shortconv) slot_shortconv_cache.reset();
        if (self.state_flags.has_mamba) slot_mamba_cache.reset();

        const logits_handle = try runtime_trait.transformerForwardLazyWithEmbeddingOverride(
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
        defer graph.freeArray(logits_handle);

        graph.eval(&[_]graph.ArrayHandle{logits_handle});

        var shape_buffer: [8]usize = undefined;
        const rank = graph.getShape(logits_handle, &shape_buffer);
        std.debug.assert(rank == 3);
        std.debug.assert(shape_buffer[0] == 1);
        std.debug.assert(shape_buffer[1] == sequence_len);
        std.debug.assert(shape_buffer[2] == self.vocab_size);

        const all_logits = try self.allocator.alloc(f32, sequence_len * self.vocab_size);
        defer self.allocator.free(all_logits);
        graph.copyToHost(logits_handle, all_logits);

        const last_token_offset = (sequence_len - 1) * self.vocab_size;
        @memcpy(logits_out, all_logits[last_token_offset .. last_token_offset + self.vocab_size]);

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
        // Single-sequence compatibility path uses slot 0.
        try self.bindImplicitSlot0StateBlocks();
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
        try self.bindImplicitSlot0StateBlocks();
        self.slot_in_use = true;

        var decode_timer = std.time.Timer.start() catch unreachable;
        const budget = @min(max_tokens, output_tokens.len);
        var current_token = first_token;
        var current_position = start_position;
        var generated_count: usize = 0;
        const logits = self.slot_logits_buffer[0..self.vocab_size];

        while (generated_count < budget) {
            try self.decodeSlot(0, current_token, current_position, logits);
            const next_token = try self.selectNextTokenFromLogits(logits);
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
