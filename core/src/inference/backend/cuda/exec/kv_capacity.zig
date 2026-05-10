//! Forward pass computation functions.
//!
//! Contains the main forward-pass entry points (single-token decode, batched
//! decode, prefill), KV capacity management, and recurrent state resets.
//! Functions use `self: anytype` to avoid circular imports with engine.zig.

const std = @import("std");
const compute = @import("compute_pkg");
const tensor = @import("compute_pkg").tensor;
const log = @import("log_pkg");
const trace = @import("xray_pkg").trace;
const staged_orchestrator = @import("../../staged_orchestrator.zig");
const per_layer_branch_feature = @import("../per_layer_branch.zig");

// --- Shared types from engine_types.zig ---
const engine_types = @import("../runtime/_types_impl.zig");
const BatchDecodeInfo = engine_types.BatchDecodeInfo;
const KvCacheDtype = engine_types.KvCacheDtype;
const enable_dispatch_observability = engine_types.enable_dispatch_observability;
const enable_device_embedding_lookup = engine_types.enable_device_embedding_lookup;
const AttentionKernelSet = engine_types.AttentionKernelSet;

// --- Compute ops from engine_ops.zig ---
const engine_ops = @import("../operators/root.zig");

// --- Utilities from engine_weights.zig ---
const engine_weights = @import("../weights/root.zig");
const bufferSlice = engine_weights.bufferSlice;
const populatePrefillHiddenFromTokens = engine_weights.populatePrefillHiddenFromTokens;
const tryPopulateHiddenFromToken = engine_weights.tryPopulateHiddenFromToken;

const saturatingU64FromU128 = engine_types.saturatingU64FromU128;
const logicalF32RowSlice = engine_types.logicalF32RowSlice;

fn topologyModeTag(self: anytype) ?[]const u8 {
    const SelfType = @TypeOf(self.*);
    if (comptime !@hasField(SelfType, "topology_mode")) return null;
    return @tagName(self.topology_mode);
}

fn topologyModeIs(self: anytype, comptime expected: []const u8) bool {
    const tag = topologyModeTag(self) orelse return false;
    return std.mem.eql(u8, tag, expected);
}

/// Resolve staged prefill chunk rows for a specific request length.
/// Keeps explicit env override behavior unchanged.
pub fn ensureKvCapacity(self: anytype, required_tokens: usize) !void {
    if (required_tokens == 0) return;
    if (required_tokens > self.max_seq_len) return error.InvalidArgument;
    const copy_function = self.copy_function orelse return error.CudaKernelUnavailable;
    const copy_u16_function = self.copy_u16_function orelse return error.CudaKernelUnavailable;

    var grew_any = false;
    for (self.block_runtime.blocks) |*layer| {
        const block = layer.attention_binding orelse continue;
        if (required_tokens <= block.kv_capacity) continue;
        if (self.fixed_alloc_mode) return error.OutOfMemory;

        var new_capacity = block.kv_capacity;
        if (new_capacity == 0) new_capacity = 1;
        while (new_capacity < required_tokens) {
            const doubled = std.math.mul(usize, new_capacity, 2) catch self.max_seq_len;
            const next = if (doubled > new_capacity) doubled else self.max_seq_len;
            new_capacity = @min(self.max_seq_len, next);
            if (new_capacity == self.max_seq_len) break;
        }
        if (new_capacity < required_tokens) return error.InvalidArgument;

        // KV shape can vary per layer (e.g., shared-KV layouts). Derive the
        // per-layer KV head count from the existing scale buffers when present,
        // instead of assuming the model-global n_kv_heads.
        var layer_n_kv_heads: usize = self.n_kv_heads;
        if (self.kv_cache_dtype.hasPerHeadScales()) {
            if (block.k_scale.size == 0 or block.v_scale.size == 0) return error.InvalidArgument;
            const old_scale_elems = std.math.divExact(usize, block.k_scale.size, @sizeOf(f32)) catch return error.InvalidArgument;
            if (block.kv_capacity == 0) return error.InvalidArgument;
            layer_n_kv_heads = std.math.divExact(usize, old_scale_elems, block.kv_capacity) catch return error.InvalidArgument;
            if (layer_n_kv_heads == 0) return error.InvalidArgument;
        }

        var new_kv_pair = switch (self.kv_storage_mode) {
            .device => try engine_weights.allocDeviceKvPairWithScales(
                &self.device,
                new_capacity,
                block.kv_dim,
                layer_n_kv_heads,
                self.kv_cache_dtype,
            ),
        };
        errdefer {
            if (new_kv_pair.v_scale.pointer != 0) new_kv_pair.v_scale.deinit(&self.device);
            if (new_kv_pair.k_scale.pointer != 0) new_kv_pair.k_scale.deinit(&self.device);
            new_kv_pair.v.deinit(&self.device);
            new_kv_pair.k.deinit(&self.device);
        }

        if (block.kv_capacity > 0) {
            const old_elems = std.math.mul(usize, block.kv_capacity, block.kv_dim) catch return error.InvalidArgument;
            switch (self.kv_cache_dtype) {
                .f16 => {
                    const old_count_u32: u32 = @intCast(old_elems);
                    try compute.cuda.copy_u16.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        copy_u16_function,
                        &block.k_cache,
                        &new_kv_pair.k,
                        old_count_u32,
                    );
                    try compute.cuda.copy_u16.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        copy_u16_function,
                        &block.v_cache,
                        &new_kv_pair.v,
                        old_count_u32,
                    );
                },
                .i8, .fp8 => {
                    // i8/fp8 cache: copy bytes via copy_u16 with halved count.
                    const old_u16_count: u32 = @intCast((old_elems + 1) / 2);
                    try compute.cuda.copy_u16.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        copy_u16_function,
                        &block.k_cache,
                        &new_kv_pair.k,
                        old_u16_count,
                    );
                    try compute.cuda.copy_u16.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        copy_u16_function,
                        &block.v_cache,
                        &new_kv_pair.v,
                        old_u16_count,
                    );
                    // Copy scale buffers (f32 elements: capacity * n_kv_heads).
                    const old_k_scale_elems = std.math.divExact(usize, block.k_scale.size, @sizeOf(f32)) catch return error.InvalidArgument;
                    const old_v_scale_elems = std.math.divExact(usize, block.v_scale.size, @sizeOf(f32)) catch return error.InvalidArgument;
                    const old_k_scale_count_u32: u32 = @intCast(old_k_scale_elems);
                    const old_v_scale_count_u32: u32 = @intCast(old_v_scale_elems);
                    try compute.cuda.copy.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        copy_function,
                        &block.k_scale,
                        &new_kv_pair.k_scale,
                        old_k_scale_count_u32,
                    );
                    try compute.cuda.copy.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        copy_function,
                        &block.v_scale,
                        &new_kv_pair.v_scale,
                        old_v_scale_count_u32,
                    );
                },
            }
        }

        if (block.v_scale.pointer != 0) block.v_scale.deinit(&self.device);
        if (block.k_scale.pointer != 0) block.k_scale.deinit(&self.device);
        block.k_cache.deinit(&self.device);
        block.v_cache.deinit(&self.device);
        block.k_cache = new_kv_pair.k;
        block.v_cache = new_kv_pair.v;
        block.k_scale = new_kv_pair.k_scale;
        block.v_scale = new_kv_pair.v_scale;
        block.kv_capacity = new_capacity;
        grew_any = true;
    }
    if (grew_any) {
        const SelfType = @TypeOf(self.*);
        if (comptime @hasField(SelfType, "decode_ptr_tables_dirty")) {
            self.decode_ptr_tables_dirty = true;
            if (comptime @hasField(SelfType, "decode_ptr_tables_cached_rows")) {
                self.decode_ptr_tables_cached_rows = 0;
            }
        }
    }
}
