//! CPU Scratch Buffers
//!
//! Shared scratch buffers and caches for transformer inference on CPU.
//! This module owns allocation/deallocation for temporary buffers and caches.

const std = @import("std");
const compute = @import("../../../../compute/root.zig");
const graph_types = @import("../../../../models/op_types.zig");
const log = @import("../../../../log.zig");
const runtime_contract = @import("../../../runtime_contract/root.zig");
const cpu_linalg = compute.cpu.linalg;
const cpu_common = compute.cpu.common;
const tensor = @import("../../../../tensor.zig");
const Tensor = tensor.Tensor;

const attn = @import("../kernels/attention.zig");
const mla = @import("../kernels/mla_attention.zig");
const ffn = @import("../kernels/ffn.zig");
const moe = @import("../kernels/moe.zig");
const mamba = @import("../kernels/mamba.zig");
const gated_delta = @import("../kernels/gated_delta.zig");
const shortconv = @import("../kernels/shortconv.zig");
const norm = @import("../kernels/norm.zig");
const kv_cache = @import("../kernels/kv_cache.zig");

const OpType = graph_types.OpType;

pub const AttnCache = attn.AttnCache;
pub const AttnTemp = attn.AttnTemp;
pub const MLACache = mla.MLACache;
pub const MLATemp = mla.MLATemp;
pub const MLAConfig = mla.MLAConfig;
pub const FfnScratch = ffn.FfnScratch;
pub const MoEScratch = moe.MoEScratch;
pub const MambaState = mamba.MambaState;
pub const MambaScratch = mamba.MambaScratch;
pub const GatedDeltaState = gated_delta.GatedDeltaState;
pub const GatedDeltaScratch = gated_delta.GatedDeltaScratch;
pub const ShortConvState = shortconv.ShortConvState;
pub const ShortConvScratch = shortconv.ShortConvScratch;
const BatchedKVCache = kv_cache.BatchedKVCache;

pub const BatchedKernelError = error{
    UnsupportedBatchedDecodeKernel,
};

pub const SlotContextError = error{
    MissingAttentionCache,
    MissingMlaCache,
    MissingMlaScratch,
    MissingMambaState,
    MissingMambaScratch,
    MissingGatedDeltaState,
    MissingGatedDeltaScratch,
    MissingShortConvState,
    MissingShortConvScratch,
    MissingBatchedCache,
};

pub const SlotPersistentState = struct {
    attn_cache: ?AttnCache = null,
    mla_cache: ?MLACache = null,
    mamba_state: ?MambaState = null,
    gated_delta_state: ?gated_delta.GatedDeltaState = null,
    shortconv_state: ?ShortConvState = null,
};

pub const SharedPersistentState = struct {
    batched_cache: ?*BatchedKVCache = null,
    mla_scratch: ?*MLATemp = null,
    mamba_scratch: ?*MambaScratch = null,
    gated_delta_scratch: ?*gated_delta.GatedDeltaScratch = null,
    shortconv_scratch: ?*ShortConvScratch = null,
};

/// Scratch buffers shared across transformer forward pass.
/// Slot arrays are dynamically sized based on compiled-plan physical mapping.
pub const ScratchBuffer = struct {
    allocator: std.mem.Allocator,
    d_model: usize,
    d_ff: usize,
    layer_count: usize,
    slot_count: usize,

    /// Temporary buffer array, dynamically sized. Slot 0 is layer_tmp
    /// (Model.forward alternating buffer), slots 1+ are register-mapped
    /// scratch assigned through liveness analysis.
    tmp: [][]f32 = &.{},
    /// Per-slot width hints (in f32 elements) derived from compiled-plan
    /// physical mapping.
    tmp_slot_width_hints: []usize = &.{},
    /// Per-slot active mask. Inactive slots are not ensured.
    tmp_slot_active: []bool = &.{},
    /// Last execution mode used to size scratch; enables deterministic
    /// prefill->decode shrink without per-token realloc churn.
    last_mode: runtime_contract.ExecutionMode = .decode,
    slot_states: []SlotPersistentState = &.{},

    attn_scratch: attn.AttnTemp = .{},
    ffn_scratch: ffn.FfnScratch = .{},
    moe_scratch: moe.MoEScratch = .{}, // For MoE layers
    matmul_scratch: cpu_linalg.MatmulScratch,

    // Shared recurrent scratch for heterogeneous models.
    mamba_scratch: ?mamba.MambaScratch = null,
    gated_delta_scratch: ?gated_delta.GatedDeltaScratch = null,

    shortconv_scratch: ?shortconv.ShortConvScratch = null,

    mla_scratch: ?mla.MLATemp = null,

    /// Get layer_tmp buffer (internal use, index 0).
    /// This buffer is used by Model.forward for alternating input/output between layers.
    /// Asserts the requested length fits within the allocated buffer.
    pub fn getLayerTmp(self: *ScratchBuffer, len: usize) []f32 {
        const layer_tmp = self.tmp[0];
        std.debug.assert(len <= layer_tmp.len); // buffer must be allocated via ensure()
        return layer_tmp[0..len];
    }

    pub fn init(allocator: std.mem.Allocator, d_model: usize, d_ff: usize, n_layers: usize) !ScratchBuffer {
        return initWithSlots(allocator, d_model, d_ff, n_layers, 1);
    }

    pub fn initWithSlots(
        allocator: std.mem.Allocator,
        d_model: usize,
        d_ff: usize,
        n_layers: usize,
        slot_count: usize,
    ) !ScratchBuffer {
        const effective_slot_count = @max(slot_count, 1);
        const slot_state_count = std.math.mul(usize, n_layers, effective_slot_count) catch return error.OutOfMemory;
        const slot_state_buffer = try allocator.alloc(SlotPersistentState, slot_state_count);
        errdefer allocator.free(slot_state_buffer);
        for (slot_state_buffer) |*slot_state| {
            slot_state.* = .{};
        }
        var matmul_workspace = try cpu_linalg.MatmulScratch.init(allocator);
        errdefer matmul_workspace.deinit();

        // Pre-allocate slot 0 (layer_tmp) so getLayerTmp is always valid.
        const tmp = try allocator.alloc([]f32, 1);
        errdefer allocator.free(tmp);
        tmp[0] = &.{};
        const width_hints = try allocator.alloc(usize, 1);
        errdefer allocator.free(width_hints);
        // Slot 0 is the model-level alternating residual workspace (`layer_tmp`).
        // It must always be active before any per-block layout registration,
        // because model.forward() uses tmp[0] directly.
        width_hints[0] = @max(d_model, 1);
        const active = try allocator.alloc(bool, 1);
        errdefer allocator.free(active);
        active[0] = true;

        return .{
            .allocator = allocator,
            .d_model = d_model,
            .d_ff = d_ff,
            .layer_count = n_layers,
            .slot_count = effective_slot_count,
            .tmp = tmp,
            .tmp_slot_width_hints = width_hints,
            .tmp_slot_active = active,
            .slot_states = slot_state_buffer,
            .matmul_scratch = matmul_workspace,
        };
    }

    fn slotLayerIndex(self: *const ScratchBuffer, slot_index: usize, layer_idx: usize) ?usize {
        if (slot_index >= self.slot_count) return null;
        if (layer_idx >= self.layer_count) return null;
        const base = std.math.mul(usize, slot_index, self.layer_count) catch return null;
        return base + layer_idx;
    }

    /// Initialize attention cache state for descriptor-selected layers.
    pub fn initAttention(self: *ScratchBuffer, layer_indices: []const usize) !void {
        for (layer_indices) |layer_idx| {
            if (layer_idx >= self.layer_count) return error.InvalidLayerIndex;
            for (0..self.slot_count) |slot_index| {
                const idx = self.slotLayerIndex(slot_index, layer_idx) orelse return error.InvalidLayerIndex;
                if (self.slot_states[idx].attn_cache != null) return error.AlreadyInitialized;
                self.slot_states[idx].attn_cache = .{};
            }
        }
    }

    pub fn registerTmpLayout(
        self: *ScratchBuffer,
        slot_width_hints: []const usize,
        slot_active: []const bool,
    ) !void {
        const required = @max(slot_width_hints.len, slot_active.len);
        try self.ensureSlotCapacity(required);
        for (0..required) |idx| {
            if (idx < slot_active.len and slot_active[idx]) {
                self.tmp_slot_active[idx] = true;
                if (idx < slot_width_hints.len and slot_width_hints[idx] > self.tmp_slot_width_hints[idx]) {
                    self.tmp_slot_width_hints[idx] = slot_width_hints[idx];
                }
            }
        }
    }

    fn ensureSlotCapacity(self: *ScratchBuffer, count: usize) !void {
        if (count <= self.tmp.len) return;
        const new_tmp = try self.allocator.alloc([]f32, count);
        @memcpy(new_tmp[0..self.tmp.len], self.tmp);
        @memset(new_tmp[self.tmp.len..], &.{});
        if (self.tmp.len > 0) self.allocator.free(self.tmp);
        self.tmp = new_tmp;

        const new_hints = try self.allocator.alloc(usize, count);
        @memcpy(new_hints[0..self.tmp_slot_width_hints.len], self.tmp_slot_width_hints);
        @memset(new_hints[self.tmp_slot_width_hints.len..], 0);
        if (self.tmp_slot_width_hints.len > 0) self.allocator.free(self.tmp_slot_width_hints);
        self.tmp_slot_width_hints = new_hints;

        const new_active = try self.allocator.alloc(bool, count);
        @memcpy(new_active[0..self.tmp_slot_active.len], self.tmp_slot_active);
        @memset(new_active[self.tmp_slot_active.len..], false);
        if (self.tmp_slot_active.len > 0) self.allocator.free(self.tmp_slot_active);
        self.tmp_slot_active = new_active;
    }

    pub fn ensure(self: *ScratchBuffer, seq_len: usize) !void {
        // Ensure active scratch slots; widths must come from compiled-plan
        // tmp layout registration (no runtime sizing heuristics).
        for (self.tmp, 0..) |*temp_slice, idx| {
            if (!self.tmp_slot_active[idx]) continue;
            const width_hint = if (self.tmp_slot_width_hints[idx] > 0)
                self.tmp_slot_width_hints[idx]
            else
                // Any active slot without a width is an invalid compiled layout.
                return error.InvalidScratchLayout;
            const buffer_len = seq_len * width_hint;
            try cpu_common.ensureAligned64F32Slice(self.allocator, temp_slice, buffer_len);
        }
    }

    fn shrinkForDecodeTransition(self: *ScratchBuffer, seq_len: usize) !void {
        const decode_len = @max(seq_len, 1);
        const shrink_factor: usize = 4;
        const keep_factor: usize = 2;

        for (self.tmp, 0..) |*temp_slice, idx| {
            if (!self.tmp_slot_active[idx]) continue;
            if (temp_slice.len == 0) continue;
            const width_hint = if (self.tmp_slot_width_hints[idx] > 0)
                self.tmp_slot_width_hints[idx]
            else
                return error.InvalidScratchLayout;
            const decode_target_len = decode_len * width_hint;
            if (temp_slice.len <= decode_target_len * shrink_factor) continue;
            const keep_len = @max(decode_target_len * keep_factor, width_hint);
            cpu_common.freeAligned64F32Slice(self.allocator, temp_slice.*);
            temp_slice.* = &.{};
            const aligned = try self.allocator.alignedAlloc(f32, .@"64", keep_len);
            temp_slice.* = aligned;
        }
    }

    pub fn ensureForMode(self: *ScratchBuffer, mode: runtime_contract.ExecutionMode, seq_len: usize) !void {
        if (mode == .decode and self.last_mode != .decode) {
            try self.shrinkForDecodeTransition(seq_len);
        }
        self.last_mode = mode;
        try self.ensure(seq_len);
    }

    pub fn deinit(self: *ScratchBuffer) void {
        // Free all temporary buffer contents.
        for (self.tmp) |temp_slice| {
            if (temp_slice.len > 0) {
                cpu_common.freeAligned64F32Slice(self.allocator, temp_slice);
            }
        }
        // Free the dynamic slot arrays.
        if (self.tmp.len > 0) self.allocator.free(self.tmp);
        if (self.tmp_slot_width_hints.len > 0) self.allocator.free(self.tmp_slot_width_hints);
        if (self.tmp_slot_active.len > 0) self.allocator.free(self.tmp_slot_active);

        self.attn_scratch.deinit(self.allocator);
        for (self.slot_states) |*slot_state| {
            if (slot_state.attn_cache) |*cache| cache.deinit(self.allocator);
            if (slot_state.mla_cache) |*cache| cache.deinit(self.allocator);
            if (slot_state.mamba_state) |*state| state.deinit();
            if (slot_state.gated_delta_state) |*state| state.deinit();
            if (slot_state.shortconv_state) |*state| state.deinit();
        }
        if (self.slot_states.len > 0) {
            self.allocator.free(self.slot_states);
            self.slot_states = &.{};
        }
        self.ffn_scratch.deinit(self.allocator);
        self.moe_scratch.deinit(self.allocator);
        self.matmul_scratch.deinit();

        if (self.mamba_scratch) |*scratch| {
            scratch.deinit();
            self.mamba_scratch = null;
        }

        if (self.gated_delta_scratch) |*scratch| {
            scratch.deinit();
            self.gated_delta_scratch = null;
        }

        if (self.mla_scratch) |*scratch| {
            scratch.deinit(self.allocator);
            self.mla_scratch = null;
        }
    }

    pub fn resetCaches(self: *ScratchBuffer) void {
        for (0..self.slot_count) |slot_index| {
            self.resetSlotCaches(slot_index);
        }
    }

    /// Reset persistent caches/states for a single scheduler slot.
    /// Used by prefill and slot lifecycle transitions to guarantee deterministic
    /// token-0 state without perturbing other active slots.
    pub fn resetSlotCaches(self: *ScratchBuffer, slot_index: usize) void {
        if (slot_index >= self.slot_count) return;
        for (0..self.layer_count) |layer_idx| {
            const slot_state = self.getSlotLayerState(slot_index, layer_idx) orelse continue;
            if (slot_state.attn_cache) |*cache| cache.resetCache();
            if (slot_state.mla_cache) |*cache| cache.resetCache();
            if (slot_state.mamba_state) |*state| state.reset();
            if (slot_state.gated_delta_state) |*state| state.reset();
            if (slot_state.shortconv_state) |*state| state.reset();
        }
    }

    /// Initialize Mamba state and scratch for selected global layer indices.
    /// Call this after init() when the model contains Mamba layers.
    pub fn initMamba(self: *ScratchBuffer, layer_indices: []const usize, config: mamba.MambaConfig) !void {
        if (layer_indices.len == 0) return;

        for (layer_indices) |layer_idx| {
            if (layer_idx >= self.layer_count) return error.InvalidLayerIndex;
            for (0..self.slot_count) |slot_index| {
                const idx = self.slotLayerIndex(slot_index, layer_idx) orelse return error.InvalidLayerIndex;
                if (self.slot_states[idx].mamba_state != null) return error.AlreadyInitialized;
            }
        }
        var initialized_slots: usize = 0;
        errdefer {
            const initialized_total = initialized_slots;
            for (0..initialized_total) |flat_idx| {
                const layer_offset = flat_idx % layer_indices.len;
                const slot_index = flat_idx / layer_indices.len;
                const layer_idx = layer_indices[layer_offset];
                const idx = self.slotLayerIndex(slot_index, layer_idx) orelse continue;
                if (self.slot_states[idx].mamba_state) |*state| {
                    state.deinit();
                    self.slot_states[idx].mamba_state = null;
                }
            }
        }
        for (0..self.slot_count) |slot_index| {
            for (layer_indices) |layer_idx| {
                const idx = self.slotLayerIndex(slot_index, layer_idx) orelse return error.InvalidLayerIndex;
                self.slot_states[idx].mamba_state = try mamba.MambaState.init(self.allocator, 1, config);
                initialized_slots += 1;
            }
        }

        // Allocate shared scratch buffer (same config for all layers)
        self.mamba_scratch = try mamba.MambaScratch.init(self.allocator, config);
    }

    /// Get shared Mamba scratch buffer.
    pub fn getMambaScratch(self: *ScratchBuffer) ?*mamba.MambaScratch {
        if (self.mamba_scratch) |*scratch| return scratch;
        return null;
    }

    pub fn initGatedDelta(
        self: *ScratchBuffer,
        layer_indices: []const usize,
        config: gated_delta.GatedDeltaConfig,
    ) !void {
        if (layer_indices.len == 0) return;

        for (layer_indices) |layer_idx| {
            if (layer_idx >= self.layer_count) return error.InvalidLayerIndex;
            for (0..self.slot_count) |slot_index| {
                const idx = self.slotLayerIndex(slot_index, layer_idx) orelse return error.InvalidLayerIndex;
                if (self.slot_states[idx].gated_delta_state != null) return error.AlreadyInitialized;
            }
        }
        var initialized_slots: usize = 0;
        errdefer {
            const initialized_total = initialized_slots;
            for (0..initialized_total) |flat_idx| {
                const layer_offset = flat_idx % layer_indices.len;
                const slot_index = flat_idx / layer_indices.len;
                const layer_idx = layer_indices[layer_offset];
                const idx = self.slotLayerIndex(slot_index, layer_idx) orelse continue;
                if (self.slot_states[idx].gated_delta_state) |*state| {
                    state.deinit();
                    self.slot_states[idx].gated_delta_state = null;
                }
            }
        }
        for (0..self.slot_count) |slot_index| {
            for (layer_indices) |layer_idx| {
                const idx = self.slotLayerIndex(slot_index, layer_idx) orelse return error.InvalidLayerIndex;
                self.slot_states[idx].gated_delta_state = try gated_delta.GatedDeltaState.init(self.allocator, 1, config);
                initialized_slots += 1;
            }
        }

        self.gated_delta_scratch = try gated_delta.GatedDeltaScratch.init(self.allocator, config);
    }

    pub fn getGatedDeltaScratch(self: *ScratchBuffer) ?*gated_delta.GatedDeltaScratch {
        if (self.gated_delta_scratch) |*scratch| return scratch;
        return null;
    }

    /// Initialize ShortConv state and scratch for selected global layer indices.
    /// Call this after init() when the model contains ShortConv layers.
    pub fn initShortConv(self: *ScratchBuffer, layer_indices: []const usize, config: shortconv.ShortConvConfig) !void {
        if (layer_indices.len == 0) return;

        for (layer_indices) |layer_idx| {
            if (layer_idx >= self.layer_count) return error.InvalidLayerIndex;
            for (0..self.slot_count) |slot_index| {
                const idx = self.slotLayerIndex(slot_index, layer_idx) orelse return error.InvalidLayerIndex;
                if (self.slot_states[idx].shortconv_state != null) return error.AlreadyInitialized;
            }
        }
        var initialized_slots: usize = 0;
        errdefer {
            const initialized_total = initialized_slots;
            for (0..initialized_total) |flat_idx| {
                const layer_offset = flat_idx % layer_indices.len;
                const slot_index = flat_idx / layer_indices.len;
                const layer_idx = layer_indices[layer_offset];
                const idx = self.slotLayerIndex(slot_index, layer_idx) orelse continue;
                if (self.slot_states[idx].shortconv_state) |*state| {
                    state.deinit();
                    self.slot_states[idx].shortconv_state = null;
                }
            }
        }
        for (0..self.slot_count) |slot_index| {
            for (layer_indices) |layer_idx| {
                const idx = self.slotLayerIndex(slot_index, layer_idx) orelse return error.InvalidLayerIndex;
                self.slot_states[idx].shortconv_state = try shortconv.ShortConvState.init(self.allocator, 1, config);
                initialized_slots += 1;
            }
        }

        // Allocate shared scratch buffer (same config for all layers)
        self.shortconv_scratch = try shortconv.ShortConvScratch.init(self.allocator, config);
    }

    /// Get shared ShortConv scratch buffer.
    pub fn getShortConvScratch(self: *ScratchBuffer) ?*shortconv.ShortConvScratch {
        if (self.shortconv_scratch) |*scratch| return scratch;
        return null;
    }

    /// Get FFN scratch for a specific slot in batched mode.
    ///
    /// Current behavior: returns shared scratch (safe because decodeBatch
    /// processes slots sequentially).
    ///
    /// Future parallel execution will return per-slot scratch.
    pub fn getFfnScratch(self: *ScratchBuffer, slot_index: usize) *ffn.FfnScratch {
        _ = slot_index;
        return &self.ffn_scratch;
    }

    /// Get MoE scratch for a specific slot in batched mode.
    ///
    /// Current behavior: returns shared scratch (safe because decodeBatch
    /// processes slots sequentially).
    ///
    /// Future parallel execution will return per-slot scratch.
    pub fn getMoeScratch(self: *ScratchBuffer, slot_index: usize) *moe.MoEScratch {
        _ = slot_index;
        return &self.moe_scratch;
    }

    /// Initialize MLA cache and scratch for selected global layer indices.
    /// Call this after init() for models using MLA (e.g., DeepSeek-V2, Youtu-VL).
    pub fn initMLA(self: *ScratchBuffer, layer_indices: []const usize) !void {
        if (layer_indices.len == 0) return;

        for (layer_indices) |layer_idx| {
            if (layer_idx >= self.layer_count) return error.InvalidLayerIndex;
            for (0..self.slot_count) |slot_index| {
                const idx = self.slotLayerIndex(slot_index, layer_idx) orelse return error.InvalidLayerIndex;
                if (self.slot_states[idx].mla_cache != null) return error.AlreadyInitialized;
                self.slot_states[idx].mla_cache = .{};
            }
        }

        // Allocate shared scratch buffer (initialized lazily in ensureTemp)
        self.mla_scratch = .{};
    }

    /// Get mutable persistent slot state for a specific global layer index.
    pub fn getSlotState(self: *ScratchBuffer, layer_idx: usize) ?*SlotPersistentState {
        return self.getSlotLayerState(0, layer_idx);
    }

    /// Get mutable persistent state for a scheduler slot + global layer index.
    pub fn getSlotLayerState(self: *ScratchBuffer, slot_index: usize, layer_idx: usize) ?*SlotPersistentState {
        const idx = self.slotLayerIndex(slot_index, layer_idx) orelse return null;
        return &self.slot_states[idx];
    }

    /// Get shared MLA scratch buffer.
    pub fn getMLAScratch(self: *ScratchBuffer) ?*mla.MLATemp {
        if (self.mla_scratch) |*scratch| return scratch;
        return null;
    }
};

/// Runtime resources needed by kernels during execution.
pub const SlotContext = struct {
    slot_state_ptr: *anyopaque,
    shared_state: *anyopaque,
    scratch: *ScratchBuffer,
    use_cache: bool,

    pub fn slotState(self: SlotContext) *SlotPersistentState {
        return @ptrCast(@alignCast(self.slot_state_ptr));
    }

    pub fn sharedState(self: SlotContext) *SharedPersistentState {
        return @ptrCast(@alignCast(self.shared_state));
    }
};

/// CPU kernel dispatch union.
/// Holds pointers to concrete kernel implementations.
pub const CpuKernel = union(enum) {
    attention: *const attn.MultiHeadAttention,
    mla_attention: *const mla.MLAttention,
    mamba: *const mamba.MambaKernel,
    gated_delta: *const gated_delta.GatedDeltaKernel,
    shortconv: *const shortconv.ShortConvKernel,
    swiglu: *const ffn.SwiGLU,
    moe: *const moe.MoEFFN,
    norm: *const norm.NormKernel,

    pub fn getOpType(self: CpuKernel) OpType {
        return switch (self) {
            .attention, .mla_attention => .multihead_attention,
            .mamba => .mamba_mixer,
            .gated_delta => .gated_delta_net,
            .shortconv => .shortconv,
            .swiglu => .mlp,
            .moe => .moe,
            .norm => .norm,
        };
    }

    pub fn forward(self: CpuKernel, input: *const Tensor, output: *Tensor, ctx: SlotContext) !void {
        const slot_state = ctx.slotState();
        const shared_state = ctx.sharedState();
        switch (self) {
            .attention => |k| if (slot_state.attn_cache) |*attn_cache| {
                try k.forward(input, output, attn_cache, &ctx.scratch.attn_scratch, &ctx.scratch.matmul_scratch, ctx.use_cache);
            } else return SlotContextError.MissingAttentionCache,
            .mla_attention => |k| if (slot_state.mla_cache) |*mla_cache| {
                const mla_scratch = shared_state.mla_scratch orelse return SlotContextError.MissingMlaScratch;
                try k.forward(input, output, mla_cache, mla_scratch, &ctx.scratch.matmul_scratch, ctx.use_cache);
            } else return SlotContextError.MissingMlaCache,
            .mamba => |k| if (slot_state.mamba_state) |*mamba_state| {
                const mamba_scratch = shared_state.mamba_scratch orelse return SlotContextError.MissingMambaScratch;
                try k.forward(input, output, mamba_state, mamba_scratch, &ctx.scratch.matmul_scratch);
            } else return SlotContextError.MissingMambaState,
            .gated_delta => |k| if (slot_state.gated_delta_state) |*gated_delta_state| {
                const gated_delta_scratch = shared_state.gated_delta_scratch orelse return SlotContextError.MissingGatedDeltaScratch;
                k.forward(input, output, gated_delta_state, gated_delta_scratch, &ctx.scratch.matmul_scratch) catch |err| {
                    log.warn("inference", "CPU gated-delta kernel forward failed", .{
                        .reason = @errorName(err),
                        .layer_idx = k.layer_idx,
                        .has_time_major = k.conv_weight_transposed != null,
                    });
                    return err;
                };
            } else return SlotContextError.MissingGatedDeltaState,
            .shortconv => |k| if (slot_state.shortconv_state) |*shortconv_state| {
                const shortconv_scratch = shared_state.shortconv_scratch orelse return SlotContextError.MissingShortConvScratch;
                try k.forward(input, output, shortconv_state, shortconv_scratch, &ctx.scratch.matmul_scratch);
            } else return SlotContextError.MissingShortConvState,
            .swiglu => |k| try k.forward(input, output, &ctx.scratch.ffn_scratch, &ctx.scratch.matmul_scratch),
            .moe => |k| try k.forward(input, output, &ctx.scratch.moe_scratch, &ctx.scratch.matmul_scratch),
            .norm => |k| k.forward(input, output),
        }
    }

    pub fn forwardBatched(
        self: CpuKernel,
        input: *const Tensor,
        output: *Tensor,
        ctx: SlotContext,
        slot_index: usize,
    ) !void {
        const slot_state = ctx.slotState();
        const shared_state = ctx.sharedState();
        switch (self) {
            .attention => |k| {
                const batched_cache = shared_state.batched_cache orelse return SlotContextError.MissingBatchedCache;
                try k.forwardWithBatchedCache(input, output, batched_cache, slot_index, &ctx.scratch.attn_scratch, &ctx.scratch.matmul_scratch, ctx.use_cache);
            },
            .mla_attention => |k| if (slot_state.mla_cache) |*mla_cache| {
                const mla_scratch = shared_state.mla_scratch orelse return SlotContextError.MissingMlaScratch;
                try k.forward(input, output, mla_cache, mla_scratch, &ctx.scratch.matmul_scratch, ctx.use_cache);
            } else return SlotContextError.MissingMlaCache,
            .mamba => |k| if (slot_state.mamba_state) |*mamba_state| {
                const mamba_scratch = shared_state.mamba_scratch orelse return SlotContextError.MissingMambaScratch;
                try k.forward(input, output, mamba_state, mamba_scratch, &ctx.scratch.matmul_scratch);
            } else return SlotContextError.MissingMambaState,
            .gated_delta => |k| if (slot_state.gated_delta_state) |*gated_delta_state| {
                const gated_delta_scratch = shared_state.gated_delta_scratch orelse return SlotContextError.MissingGatedDeltaScratch;
                k.forward(input, output, gated_delta_state, gated_delta_scratch, &ctx.scratch.matmul_scratch) catch |err| {
                    log.warn("inference", "CPU gated-delta batched forward failed", .{
                        .reason = @errorName(err),
                        .layer_idx = k.layer_idx,
                        .has_time_major = k.conv_weight_transposed != null,
                    });
                    return err;
                };
            } else return SlotContextError.MissingGatedDeltaState,
            .shortconv => |k| if (slot_state.shortconv_state) |*shortconv_state| {
                const shortconv_scratch = shared_state.shortconv_scratch orelse return SlotContextError.MissingShortConvScratch;
                try k.forward(input, output, shortconv_state, shortconv_scratch, &ctx.scratch.matmul_scratch);
            } else return SlotContextError.MissingShortConvState,
            .swiglu => |k| try k.forward(input, output, ctx.scratch.getFfnScratch(slot_index), &ctx.scratch.matmul_scratch),
            .moe => |k| try k.forward(input, output, ctx.scratch.getMoeScratch(slot_index), &ctx.scratch.matmul_scratch),
            .norm => |k| k.forward(input, output),
        }
    }

    /// Batched decode across multiple scheduler slots in a single kernel call.
    /// `slot_indices.len` must match `input.shape[1]` for decode tensors [1, batch, d_model].
    pub fn forwardBatchedSlots(
        self: CpuKernel,
        input: *const Tensor,
        output: *Tensor,
        ctx: SlotContext,
        slot_indices: []const usize,
    ) !void {
        const shared_state = ctx.sharedState();
        switch (self) {
            .attention => |k| try k.forwardWithBatchedCacheSlots(
                input,
                output,
                shared_state.batched_cache orelse return SlotContextError.MissingBatchedCache,
                slot_indices,
                &ctx.scratch.attn_scratch,
                &ctx.scratch.matmul_scratch,
                ctx.use_cache,
            ),
            .swiglu => |k| try k.forward(input, output, &ctx.scratch.ffn_scratch, &ctx.scratch.matmul_scratch),
            .moe => |k| try k.forward(input, output, &ctx.scratch.moe_scratch, &ctx.scratch.matmul_scratch),
            .norm => |k| k.forward(input, output),
            .mla_attention, .mamba, .gated_delta, .shortconv => return BatchedKernelError.UnsupportedBatchedDecodeKernel,
        }
    }
};

pub const kernels = @import("../kernels/root.zig");

test "CpuKernel.forward returns typed error when attention cache is missing" {
    const allocator = std.testing.allocator;
    var scratch = try ScratchBuffer.init(allocator, 8, 16, 1);
    defer scratch.deinit();

    var input_data = [_]f32{0} ** 8;
    var output_data = [_]f32{0} ** 8;
    const input = Tensor.view3DSlice(input_data[0..], 1, 8);
    var output = Tensor.view3DSlice(output_data[0..], 1, 8);

    var slot_state = SlotPersistentState{};
    var shared_state = SharedPersistentState{};
    const ctx = SlotContext{
        .slot_state_ptr = &slot_state,
        .shared_state = &shared_state,
        .scratch = &scratch,
        .use_cache = true,
    };

    // Forward validates state before dereferencing the kernel pointer.
    const fake_attention = @as(*const attn.MultiHeadAttention, @ptrFromInt(@as(usize, @alignOf(attn.MultiHeadAttention))));
    const kernel = CpuKernel{ .attention = fake_attention };
    try std.testing.expectError(SlotContextError.MissingAttentionCache, kernel.forward(&input, &output, ctx));
}

test "ScratchBuffer.ensureForMode clears tmp on decode-transition allocation failure" {
    // Regression: shrinkForDecodeTransition frees old buffers then re-allocs.
    // Before the fix, a failed realloc left tmp entries pointing to freed
    // memory. deinit() would then double-free.
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{});
    var scratch = try ScratchBuffer.init(failing.allocator(), 8, 16, 1);
    defer scratch.deinit();

    // Grow scratch for prefill — creates large buffers in tmp[0].
    try scratch.ensureForMode(.prefill, 64);
    try std.testing.expect(scratch.tmp[0].len >= 64 * 8);

    // Make the next allocation fail, then trigger decode transition.
    // shrinkForDecodeTransition frees the large buffer then tries to alloc
    // a smaller one. The alloc fails, so tmp[0] must be left empty (not dangling).
    failing.fail_index = failing.alloc_index;
    try std.testing.expectError(error.OutOfMemory, scratch.ensureForMode(.decode, 1));

    // With the fix, tmp[0] is safely empty. deinit() in defer won't double-free.
    try std.testing.expectEqual(@as(usize, 0), scratch.tmp[0].len);
}

test "ScratchBuffer.resetSlotCaches resets only target slot recurrent state" {
    const allocator = std.testing.allocator;
    var scratch = try ScratchBuffer.initWithSlots(allocator, 8, 16, 1, 2);
    defer scratch.deinit();

    const gd_config = gated_delta.GatedDeltaConfig{
        .d_model = 8,
        .d_conv = 4,
        .n_heads = 2,
        .d_head = 2,
    };
    try scratch.initGatedDelta(&.{0}, gd_config);

    const slot0 = scratch.getSlotLayerState(0, 0) orelse return error.TestUnexpectedResult;
    const slot1 = scratch.getSlotLayerState(1, 0) orelse return error.TestUnexpectedResult;
    const slot0_state = &(slot0.gated_delta_state orelse return error.TestUnexpectedResult);
    const slot1_state = &(slot1.gated_delta_state orelse return error.TestUnexpectedResult);

    @memset(slot0_state.conv_state, 1.0);
    @memset(slot0_state.ssm_state, 1.0);
    @memset(slot1_state.conv_state, 2.0);
    @memset(slot1_state.ssm_state, 2.0);

    scratch.resetSlotCaches(0);

    for (slot0_state.conv_state) |value| {
        try std.testing.expectEqual(@as(f32, 0.0), value);
    }
    for (slot0_state.ssm_state) |value| {
        try std.testing.expectEqual(@as(f32, 0.0), value);
    }
    for (slot1_state.conv_state) |value| {
        try std.testing.expectEqual(@as(f32, 2.0), value);
    }
    for (slot1_state.ssm_state) |value| {
        try std.testing.expectEqual(@as(f32, 2.0), value);
    }
}
