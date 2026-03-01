//! CPU Scratch Buffers
//!
//! Shared scratch buffers and caches for transformer inference on CPU.
//! This module owns allocation/deallocation for temporary buffers and caches.

const std = @import("std");
const compute = @import("../../../../compute/root.zig");
const graph_types = @import("../../../../models/op_types.zig");
const layer_ops = @import("../../../../models/layer_ops.zig");
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
const shortconv = @import("../kernels/shortconv.zig");
const norm = @import("../kernels/norm.zig");
const kv_cache = @import("../kernels/kv_cache.zig");

const OpType = graph_types.OpType;
pub const BufferId = layer_ops.BufferId;

pub const AttnCache = attn.AttnCache;
pub const AttnTemp = attn.AttnTemp;
pub const MLACache = mla.MLACache;
pub const MLATemp = mla.MLATemp;
pub const MLAConfig = mla.MLAConfig;
pub const FfnScratch = ffn.FfnScratch;
pub const MoEScratch = moe.MoEScratch;
pub const MambaState = mamba.MambaState;
pub const MambaScratch = mamba.MambaScratch;
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
    MissingShortConvState,
    MissingShortConvScratch,
    MissingBatchedCache,
};

pub const SlotPersistentState = struct {
    attn_cache: ?AttnCache = null,
    mla_cache: ?MLACache = null,
    mamba_state: ?MambaState = null,
    shortconv_state: ?ShortConvState = null,
};

pub const SharedPersistentState = struct {
    batched_cache: ?*BatchedKVCache = null,
    mla_scratch: ?*MLATemp = null,
    mamba_scratch: ?*MambaScratch = null,
    shortconv_scratch: ?*ShortConvScratch = null,
    state_blocks: []const runtime_contract.StateBlockHandle = &.{},
};

/// Number of temporary buffers available.
/// Array index maps to BufferId enum values (except index 0):
/// - [0] = layer_tmp (internal use for Model.forward alternating buffer)
/// - [1] = norm_out (BufferId.norm_out = 1)
/// - [2] = branch_out (BufferId.branch_out = 2)
/// - [3..63] = tmp3..tmp63 (BufferId.tmp3 = 3, etc.)
/// Note: BufferId.residual (0) is NOT stored here - it uses the model output buffer.
pub const NUM_TMP_BUFFERS: usize = 64;

/// Scratch buffers shared across transformer forward pass.
/// Uses an array for tmp buffers to simplify allocation/deallocation.
pub const ScratchBuffer = struct {
    allocator: std.mem.Allocator,
    d_model: usize,
    d_ff: usize,

    /// Unified temporary buffer array. See NUM_TMP_BUFFERS doc for index mapping.
    /// Access via getTmp(BufferId, len) or getLayerTmp(len) for index 0.
    tmp: [NUM_TMP_BUFFERS][]f32 = [_][]f32{&.{}} ** NUM_TMP_BUFFERS,
    /// Per-slot width hints (in f32 elements) derived from compiled-plan
    /// physical mapping.
    tmp_slot_width_hints: [NUM_TMP_BUFFERS]usize = [_]usize{0} ** NUM_TMP_BUFFERS,
    /// Per-slot active mask. Inactive slots are not ensured.
    tmp_slot_active: [NUM_TMP_BUFFERS]bool = [_]bool{false} ** NUM_TMP_BUFFERS,
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

    shortconv_scratch: ?shortconv.ShortConvScratch = null,

    mla_scratch: ?mla.MLATemp = null,

    /// Get a temporary buffer by BufferId and length.
    /// This is the canonical way to access scratch buffers.
    /// Asserts that:
    /// - id is not .residual (which uses the model output buffer, not scratch)
    /// - the requested length fits within the allocated buffer
    pub fn getTmp(self: *ScratchBuffer, id: BufferId, len: usize) []f32 {
        const buffer_idx = @intFromEnum(id);
        std.debug.assert(id != .residual); // residual uses model output, not scratch
        std.debug.assert(buffer_idx < NUM_TMP_BUFFERS);
        const buffer_slice = self.tmp[buffer_idx];
        std.debug.assert(len <= buffer_slice.len); // buffer must be allocated via ensure()
        return buffer_slice[0..len];
    }

    /// Get layer_tmp buffer (internal use, index 0).
    /// This buffer is used by Model.forward for alternating input/output between layers.
    /// Asserts the requested length fits within the allocated buffer.
    pub fn getLayerTmp(self: *ScratchBuffer, len: usize) []f32 {
        const layer_tmp = self.tmp[0];
        std.debug.assert(len <= layer_tmp.len); // buffer must be allocated via ensure()
        return layer_tmp[0..len];
    }

    pub fn init(allocator: std.mem.Allocator, d_model: usize, d_ff: usize, n_layers: usize) !ScratchBuffer {
        const slot_state_buffer = try allocator.alloc(SlotPersistentState, n_layers);
        errdefer allocator.free(slot_state_buffer);
        for (slot_state_buffer) |*slot_state| {
            slot_state.* = .{};
        }
        var matmul_workspace = try cpu_linalg.MatmulScratch.init(allocator);
        errdefer matmul_workspace.deinit();
        return .{
            .allocator = allocator,
            .d_model = d_model,
            .d_ff = d_ff,
            .slot_states = slot_state_buffer,
            .matmul_scratch = matmul_workspace,
        };
    }

    /// Initialize attention cache state for descriptor-selected layers.
    pub fn initAttention(self: *ScratchBuffer, layer_indices: []const usize) !void {
        for (layer_indices) |layer_idx| {
            if (layer_idx >= self.slot_states.len) return error.InvalidLayerIndex;
            if (self.slot_states[layer_idx].attn_cache != null) return error.AlreadyInitialized;
            self.slot_states[layer_idx].attn_cache = .{};
        }
    }

    pub fn registerTmpLayout(
        self: *ScratchBuffer,
        slot_width_hints: [NUM_TMP_BUFFERS]usize,
        slot_active: [NUM_TMP_BUFFERS]bool,
    ) void {
        for (0..NUM_TMP_BUFFERS) |idx| {
            if (!slot_active[idx]) continue;
            self.tmp_slot_active[idx] = true;
            if (slot_width_hints[idx] > self.tmp_slot_width_hints[idx]) {
                self.tmp_slot_width_hints[idx] = slot_width_hints[idx];
            }
        }
    }

    pub fn ensure(self: *ScratchBuffer, seq_len: usize) !void {
        // Account for fused projections which can be larger than d_model:
        // - Fused QKV: ~1.5x d_model (Q + K + V)
        // - Fused gate_up: 2x d_ff (gate + up)
        const fallback_dim = @max(self.d_model, self.d_ff * 2);
        var has_registered_layout = false;
        for (self.tmp_slot_active[1..]) |active| {
            if (active) {
                has_registered_layout = true;
                break;
            }
        }

        // Without a registered mapping, preserve legacy behavior and ensure all
        // tmp buffers to the conservative fallback size.
        if (!has_registered_layout) {
            const full_len = seq_len * fallback_dim;
            for (&self.tmp) |*temp_slice| {
                try cpu_common.ensureF32Slice(self.allocator, temp_slice, full_len);
            }
            return;
        }

        // Ensure layer_tmp (index 0) and active mapped scratch slots.
        for (&self.tmp, 0..) |*temp_slice, idx| {
            if (idx != 0 and !self.tmp_slot_active[idx]) continue;
            const width_hint = if (idx == 0 and self.tmp_slot_width_hints[0] == 0)
                // layer_tmp (index 0) is a general scratch workspace, not a
                // register-mapped buffer. Its size depends on model dimensions.
                fallback_dim
            else if (self.tmp_slot_width_hints[idx] > 0)
                self.tmp_slot_width_hints[idx]
            else
                // Active mapped slots must have a valid width from
                // buildTmpRegisterScratchMap; a zero here is a bug.
                return error.InvalidScratchLayout;
            const buffer_len = seq_len * width_hint;
            try cpu_common.ensureF32Slice(self.allocator, temp_slice, buffer_len);
        }
    }

    fn shrinkForDecodeTransition(self: *ScratchBuffer, seq_len: usize) !void {
        const decode_len = @max(seq_len, 1);
        const fallback_dim = @max(self.d_model, self.d_ff * 2);
        const shrink_factor: usize = 4;
        const keep_factor: usize = 2;

        for (&self.tmp, 0..) |*temp_slice, idx| {
            if (idx != 0 and !self.tmp_slot_active[idx]) continue;
            if (temp_slice.len == 0) continue;
            const width_hint = if (idx == 0 and self.tmp_slot_width_hints[0] == 0)
                fallback_dim
            else if (self.tmp_slot_width_hints[idx] > 0)
                self.tmp_slot_width_hints[idx]
            else
                return error.InvalidScratchLayout;
            const decode_target_len = decode_len * width_hint;
            if (temp_slice.len <= decode_target_len * shrink_factor) continue;
            const keep_len = @max(decode_target_len * keep_factor, width_hint);
            self.allocator.free(temp_slice.*);
            temp_slice.* = try self.allocator.alloc(f32, keep_len);
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
        // Free all temporary buffers in a single loop
        for (&self.tmp) |*temp_slice| {
            if (temp_slice.len > 0) {
                self.allocator.free(temp_slice.*);
                temp_slice.* = &.{};
            }
        }

        self.attn_scratch.deinit(self.allocator);
        for (self.slot_states) |*slot_state| {
            if (slot_state.attn_cache) |*cache| cache.deinit(self.allocator);
            if (slot_state.mla_cache) |*cache| cache.deinit(self.allocator);
            if (slot_state.mamba_state) |*state| state.deinit();
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

        if (self.mla_scratch) |*scratch| {
            scratch.deinit(self.allocator);
            self.mla_scratch = null;
        }
    }

    pub fn resetCaches(self: *ScratchBuffer) void {
        for (self.slot_states) |*slot_state| {
            if (slot_state.attn_cache) |*cache| cache.resetCache();
            if (slot_state.mla_cache) |*cache| cache.resetCache();
            if (slot_state.mamba_state) |*state| state.reset();
            if (slot_state.shortconv_state) |*state| state.reset();
        }
    }

    /// Initialize Mamba state and scratch for selected global layer indices.
    /// Call this after init() when the model contains Mamba layers.
    pub fn initMamba(self: *ScratchBuffer, layer_indices: []const usize, config: mamba.MambaConfig) !void {
        if (layer_indices.len == 0) return;

        for (layer_indices) |layer_idx| {
            if (layer_idx >= self.slot_states.len) return error.InvalidLayerIndex;
            if (self.slot_states[layer_idx].mamba_state != null) return error.AlreadyInitialized;
        }
        var initialized: usize = 0;
        errdefer {
            for (0..initialized) |idx| {
                const layer_idx = layer_indices[idx];
                if (self.slot_states[layer_idx].mamba_state) |*state| {
                    state.deinit();
                    self.slot_states[layer_idx].mamba_state = null;
                }
            }
        }
        for (layer_indices) |layer_idx| {
            self.slot_states[layer_idx].mamba_state = try mamba.MambaState.init(self.allocator, 1, config);
            initialized += 1;
        }

        // Allocate shared scratch buffer (same config for all layers)
        self.mamba_scratch = try mamba.MambaScratch.init(self.allocator, config);
    }

    /// Get shared Mamba scratch buffer.
    pub fn getMambaScratch(self: *ScratchBuffer) ?*mamba.MambaScratch {
        if (self.mamba_scratch) |*scratch| return scratch;
        return null;
    }

    /// Initialize ShortConv state and scratch for selected global layer indices.
    /// Call this after init() when the model contains ShortConv layers.
    pub fn initShortConv(self: *ScratchBuffer, layer_indices: []const usize, config: shortconv.ShortConvConfig) !void {
        if (layer_indices.len == 0) return;

        for (layer_indices) |layer_idx| {
            if (layer_idx >= self.slot_states.len) return error.InvalidLayerIndex;
            if (self.slot_states[layer_idx].shortconv_state != null) return error.AlreadyInitialized;
        }
        var initialized: usize = 0;
        errdefer {
            for (0..initialized) |idx| {
                const layer_idx = layer_indices[idx];
                if (self.slot_states[layer_idx].shortconv_state) |*state| {
                    state.deinit();
                    self.slot_states[layer_idx].shortconv_state = null;
                }
            }
        }
        for (layer_indices) |layer_idx| {
            self.slot_states[layer_idx].shortconv_state = try shortconv.ShortConvState.init(self.allocator, 1, config);
            initialized += 1;
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
            if (layer_idx >= self.slot_states.len) return error.InvalidLayerIndex;
            if (self.slot_states[layer_idx].mla_cache != null) return error.AlreadyInitialized;
            self.slot_states[layer_idx].mla_cache = .{};
        }

        // Allocate shared scratch buffer (initialized lazily in ensureTemp)
        self.mla_scratch = .{};
    }

    /// Get mutable persistent slot state for a specific global layer index.
    pub fn getSlotState(self: *ScratchBuffer, layer_idx: usize) ?*SlotPersistentState {
        if (layer_idx < self.slot_states.len) return &self.slot_states[layer_idx];
        return null;
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
    shortconv: *const shortconv.ShortConvKernel,
    swiglu: *const ffn.SwiGLU,
    moe: *const moe.MoEFFN,
    norm: *const norm.NormKernel,

    pub fn getOpType(self: CpuKernel) OpType {
        return switch (self) {
            .attention, .mla_attention => .multihead_attention,
            .mamba => .mamba_mixer,
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
            .mla_attention, .mamba, .shortconv => return BatchedKernelError.UnsupportedBatchedDecodeKernel,
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
