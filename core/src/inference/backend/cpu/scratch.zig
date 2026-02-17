//! CPU Scratch Buffers
//!
//! Shared scratch buffers and caches for transformer inference on CPU.
//! This module owns allocation/deallocation for temporary buffers and caches.

const std = @import("std");
const compute = @import("../../../compute/root.zig");
const matmul = compute.ops.matmul;
const ops = @import("../../graph_runtime/root.zig").layer_ops;

const attn = @import("kernels/attention.zig");
const mla = @import("kernels/mla_attention.zig");
const ffn = @import("kernels/ffn.zig");
const moe = @import("kernels/moe.zig");
const mamba = @import("kernels/mamba.zig");
const shortconv = @import("kernels/shortconv.zig");

pub const BufferId = ops.BufferId;

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

    attn_caches: []attn.AttnCache = &.{},
    attn_scratch: attn.AttnTemp = .{},
    ffn_scratch: ffn.FfnScratch = .{},
    moe_scratch: moe.MoEScratch = .{}, // For MoE layers
    matmul_scratch: matmul.MatmulScratch,

    // Mamba state/scratch for heterogeneous models (null for homogeneous attention-only)
    mamba_states: ?[]mamba.MambaState = null,
    mamba_scratch: ?mamba.MambaScratch = null,

    // ShortConv state/scratch for heterogeneous models (null for homogeneous attention-only)
    shortconv_states: ?[]shortconv.ShortConvState = null,
    shortconv_scratch: ?shortconv.ShortConvScratch = null,

    // MLA (Multi-Latent Attention) cache/scratch for MLA models (null for standard attention)
    mla_caches: ?[]mla.MLACache = null,
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
        const attn_cache_buffer = try allocator.alloc(attn.AttnCache, n_layers);
        errdefer allocator.free(attn_cache_buffer);
        for (attn_cache_buffer) |*cache| cache.* = .{};
        var matmul_workspace = try matmul.MatmulScratch.init(allocator);
        errdefer matmul_workspace.deinit();
        return .{
            .allocator = allocator,
            .d_model = d_model,
            .d_ff = d_ff,
            .attn_caches = attn_cache_buffer,
            .matmul_scratch = matmul_workspace,
        };
    }

    pub fn ensure(self: *ScratchBuffer, seq_len: usize) !void {
        // Account for fused projections which can be larger than d_model or d_ff alone:
        // - Fused QKV: ~1.5x d_model (Q + K + V)
        // - Fused gate_up: 2x d_ff (gate + up)
        // Use 2x d_ff as the max to handle all cases
        const max_buffer_dim = @max(self.d_model, self.d_ff * 2);
        const buffer_len = seq_len * max_buffer_dim;

        // Ensure all temporary buffers in a single loop
        for (&self.tmp) |*temp_slice| {
            try ensureSlice(self.allocator, temp_slice, buffer_len);
        }
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
        for (self.attn_caches) |*cache| cache.deinit(self.allocator);
        if (self.attn_caches.len > 0) self.allocator.free(self.attn_caches);
        self.ffn_scratch.deinit(self.allocator);
        self.moe_scratch.deinit(self.allocator);
        self.matmul_scratch.deinit();

        // Clean up Mamba resources if present
        if (self.mamba_states) |states| {
            for (states) |*state| state.deinit();
            self.allocator.free(states);
            self.mamba_states = null;
        }
        if (self.mamba_scratch) |*scratch| {
            scratch.deinit();
            self.mamba_scratch = null;
        }

        // Clean up MLA resources if present
        if (self.mla_caches) |caches| {
            for (caches) |*cache| cache.deinit(self.allocator);
            self.allocator.free(caches);
            self.mla_caches = null;
        }
        if (self.mla_scratch) |*scratch| {
            scratch.deinit(self.allocator);
            self.mla_scratch = null;
        }
    }

    pub fn resetCaches(self: *ScratchBuffer) void {
        for (self.attn_caches) |*cache| cache.resetCache();
        // Reset MLA caches if present
        if (self.mla_caches) |caches| {
            for (caches) |*cache| cache.resetCache();
        }
        // Reset Mamba state if present
        if (self.mamba_states) |states| {
            for (states) |*state| state.reset();
        }
    }

    /// Initialize Mamba state and scratch for heterogeneous models.
    /// Call this after init() if the model contains Mamba layers.
    pub fn initMamba(self: *ScratchBuffer, n_mamba_layers: usize, config: mamba.MambaConfig) !void {
        if (n_mamba_layers == 0) return;

        // Allocate state for each Mamba layer
        const states = try self.allocator.alloc(mamba.MambaState, n_mamba_layers);
        errdefer self.allocator.free(states);

        for (states, 0..) |*state, i| {
            state.* = mamba.MambaState.init(self.allocator, 1, config) catch |err| {
                // Clean up already-initialized states on error
                for (0..i) |j| states[j].deinit();
                return err;
            };
        }

        self.mamba_states = states;

        // Allocate shared scratch buffer (same config for all layers)
        self.mamba_scratch = try mamba.MambaScratch.init(self.allocator, config);
    }

    /// Get Mamba state for a specific layer (by Mamba layer index, not global layer index).
    pub fn getMambaState(self: *ScratchBuffer, mamba_layer_idx: usize) ?*mamba.MambaState {
        if (self.mamba_states) |states| {
            if (mamba_layer_idx < states.len) {
                return &states[mamba_layer_idx];
            }
        }
        return null;
    }

    /// Get shared Mamba scratch buffer.
    pub fn getMambaScratch(self: *ScratchBuffer) ?*mamba.MambaScratch {
        if (self.mamba_scratch) |*scratch| return scratch;
        return null;
    }

    /// Initialize ShortConv state and scratch for heterogeneous models.
    /// Call this after init() if the model contains ShortConv layers.
    pub fn initShortConv(self: *ScratchBuffer, n_shortconv_layers: usize, config: shortconv.ShortConvConfig) !void {
        if (n_shortconv_layers == 0) return;

        // Allocate state for each ShortConv layer
        const states = try self.allocator.alloc(shortconv.ShortConvState, n_shortconv_layers);
        errdefer self.allocator.free(states);

        for (states, 0..) |*state, i| {
            state.* = shortconv.ShortConvState.init(self.allocator, 1, config) catch |err| {
                // Clean up already-initialized states on error
                for (0..i) |j| states[j].deinit();
                return err;
            };
        }

        self.shortconv_states = states;

        // Allocate shared scratch buffer (same config for all layers)
        self.shortconv_scratch = try shortconv.ShortConvScratch.init(self.allocator, config);
    }

    /// Get ShortConv state for a specific layer (by ShortConv layer index, not global layer index).
    pub fn getShortConvState(self: *ScratchBuffer, shortconv_layer_idx: usize) ?*shortconv.ShortConvState {
        if (self.shortconv_states) |states| {
            if (shortconv_layer_idx < states.len) {
                return &states[shortconv_layer_idx];
            }
        }
        return null;
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

    /// Initialize MLA cache and scratch for models using Multi-Latent Attention.
    /// Call this after init() if the model uses MLA (e.g., DeepSeek-V2, Youtu-VL).
    pub fn initMLA(self: *ScratchBuffer, n_layers: usize) !void {
        if (n_layers == 0) return;

        // Allocate per-layer MLA caches
        const caches = try self.allocator.alloc(mla.MLACache, n_layers);
        errdefer self.allocator.free(caches);

        for (caches) |*cache| {
            cache.* = .{};
        }

        self.mla_caches = caches;

        // Allocate shared scratch buffer (initialized lazily in ensureTemp)
        self.mla_scratch = .{};
    }

    /// Get MLA cache for a specific layer.
    pub fn getMLACache(self: *ScratchBuffer, layer_idx: usize) ?*mla.MLACache {
        if (self.mla_caches) |caches| {
            if (layer_idx < caches.len) {
                return &caches[layer_idx];
            }
        }
        return null;
    }

    /// Get shared MLA scratch buffer.
    pub fn getMLAScratch(self: *ScratchBuffer) ?*mla.MLATemp {
        if (self.mla_scratch) |*scratch| return scratch;
        return null;
    }
};

fn ensureSlice(allocator: std.mem.Allocator, storage: *[]f32, needed: usize) !void {
    if (storage.len >= needed) return;
    if (storage.len > 0) allocator.free(storage.*);
    storage.* = try allocator.alloc(f32, needed);
}
