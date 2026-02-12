//! KV Cache Core Operations
//!
//! Core KV cache data structure and attention operations with dtype conversion.
//! Used by capi modules to avoid substantial logic in the FFI layer.

const std = @import("std");
const tensor_mod = @import("../../tensor.zig");
const dtype_mod = @import("../../dtype.zig");
const attention = @import("attn_primitives.zig");
const dtype_convert = @import("dtype_convert.zig");

pub const Tensor = tensor_mod.Tensor;
pub const DType = tensor_mod.DType;

/// KV cache for transformer attention.
/// Stores K and V tensors for all layers, supporting update and retrieval.
pub const KVCache = struct {
    /// K cache: [n_layers, max_seq_len, n_kv_heads, head_dim]
    k_cache: []f32,
    /// V cache: [n_layers, max_seq_len, n_kv_heads, head_dim]
    v_cache: []f32,
    /// Current sequence position (shared across layers)
    seq_pos: usize,
    /// Configuration
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    /// Sliding window size (0 = disabled)
    sliding_window: usize,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        sliding_window: usize,
    ) !*Self {
        const cache_size = n_layers * max_seq_len * n_kv_heads * head_dim;
        const k_cache = try allocator.alloc(f32, cache_size);
        errdefer allocator.free(k_cache);
        const v_cache = try allocator.alloc(f32, cache_size);
        errdefer allocator.free(v_cache);

        @memset(k_cache, 0);
        @memset(v_cache, 0);

        const self = try allocator.create(Self);
        self.* = .{
            .k_cache = k_cache,
            .v_cache = v_cache,
            .seq_pos = 0,
            .n_layers = n_layers,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .max_seq_len = max_seq_len,
            .sliding_window = sliding_window,
        };
        return self;
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.free(self.k_cache);
        allocator.free(self.v_cache);
        allocator.destroy(self);
    }

    /// Update cache with new K/V values at current position
    pub fn update(
        self: *Self,
        layer_idx: usize,
        k_values: []const f32,
        v_values: []const f32,
        seq_len: usize,
    ) void {
        const kv_size = self.n_kv_heads * self.head_dim;
        const layer_stride = self.max_seq_len * kv_size;
        const layer_offset = layer_idx * layer_stride;

        for (0..seq_len) |s| {
            const cache_pos = (self.seq_pos + s) % self.max_seq_len;
            const cache_idx = layer_offset + cache_pos * kv_size;
            const input_idx = s * kv_size;

            @memcpy(
                self.k_cache[cache_idx..][0..kv_size],
                k_values[input_idx..][0..kv_size],
            );
            @memcpy(
                self.v_cache[cache_idx..][0..kv_size],
                v_values[input_idx..][0..kv_size],
            );
        }
    }

    /// Advance sequence position after update
    pub fn advance(self: *Self, steps: usize) void {
        self.seq_pos += steps;
    }

    /// Get cache length (number of valid tokens)
    pub fn getLength(self: *const Self) usize {
        return @min(self.seq_pos, self.max_seq_len);
    }

    /// Reset cache to empty state
    pub fn reset(self: *Self) void {
        self.seq_pos = 0;
    }
};

/// Error type for KV cache attention operations.
pub const KVCacheAttentionError = error{
    InvalidArgument,
    SequenceTooLong,
    UnsupportedDType,
    OutOfMemory,
};

/// Perform basic attention with KV cache (f32 only, no sinks/sliding window).
///
/// Q: [batch, n_heads, seq_len, head_dim]
/// K: [batch, n_kv_heads, seq_len, head_dim]
/// V: [batch, n_kv_heads, seq_len, head_dim]
pub fn attentionWithCache(
    allocator: std.mem.Allocator,
    q_tensor: *const Tensor,
    k_tensor: *const Tensor,
    v_tensor: *const Tensor,
    cache_state: *KVCache,
    layer_idx: usize,
    scale: f32,
) KVCacheAttentionError!*Tensor {
    // Validate dimensions
    if (layer_idx >= cache_state.n_layers) return error.InvalidArgument;
    if (q_tensor.n_dims != 4 or k_tensor.n_dims != 4 or v_tensor.n_dims != 4) return error.InvalidArgument;

    // Validate dtype (f32 only for this function)
    if (q_tensor.dtype != .f32 or k_tensor.dtype != .f32 or v_tensor.dtype != .f32) return error.UnsupportedDType;

    const n_heads: usize = @intCast(q_tensor.shape[1]);
    const n_kv_heads: usize = @intCast(k_tensor.shape[1]);
    const seq_len: usize = @intCast(q_tensor.shape[2]);
    const head_dim: usize = @intCast(q_tensor.shape[3]);

    // Validate cache compatibility
    if (n_kv_heads != cache_state.n_kv_heads or head_dim != cache_state.head_dim) return error.InvalidArgument;

    // Validate Q/K/V shape compatibility
    if (k_tensor.shape[0] != q_tensor.shape[0] or v_tensor.shape[0] != q_tensor.shape[0]) return error.InvalidArgument;
    if (k_tensor.shape[2] != q_tensor.shape[2] or v_tensor.shape[2] != q_tensor.shape[2]) return error.InvalidArgument;
    if (k_tensor.shape[1] != v_tensor.shape[1] or k_tensor.shape[3] != v_tensor.shape[3]) return error.InvalidArgument;

    // Get data pointers
    const q_data: [*]const f32 = @ptrCast(@alignCast(q_tensor.data_ptr));
    const k_data: [*]const f32 = @ptrCast(@alignCast(k_tensor.data_ptr));
    const v_data: [*]const f32 = @ptrCast(@alignCast(v_tensor.data_ptr));

    // Compute strides and offsets
    const kv_size = cache_state.n_kv_heads * cache_state.head_dim;
    const layer_offset = layer_idx * cache_state.max_seq_len * kv_size;
    const k_strides = dtype_convert.tensorStrides4D(k_tensor);
    const q_strides = dtype_convert.tensorStrides4D(q_tensor);

    // Update cache
    attention.updateKVCache(
        cache_state.k_cache,
        cache_state.v_cache,
        k_data,
        v_data,
        k_strides,
        layer_offset,
        cache_state.seq_pos,
        cache_state.max_seq_len,
        seq_len,
        n_kv_heads,
        head_dim,
    );

    // Allocate output
    const result_tensor = Tensor.init(allocator, q_tensor.shape[0..@intCast(q_tensor.n_dims)], q_tensor.dtype, q_tensor.device) catch return error.OutOfMemory;
    errdefer result_tensor.deinit(allocator);

    const out_strides = dtype_convert.tensorStrides4D(result_tensor);
    const scale_value = if (scale == 0) 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))) else scale;
    const total_seq_len = cache_state.seq_pos + seq_len;
    const cached_seq_len = @min(total_seq_len, cache_state.max_seq_len);

    // Compute SDPA
    attention.sdpaCached(
        @ptrCast(@alignCast(result_tensor.data_ptr)),
        out_strides,
        q_data,
        q_strides,
        cache_state.k_cache[layer_offset..][0 .. cache_state.max_seq_len * kv_size],
        cache_state.v_cache[layer_offset..][0 .. cache_state.max_seq_len * kv_size],
        n_heads,
        n_kv_heads,
        seq_len,
        cached_seq_len,
        head_dim,
        cache_state.seq_pos,
        scale_value,
        null,
        0,
        allocator,
    ) catch |err| {
        result_tensor.deinit(allocator);
        return switch (err) {
            error.SequenceTooLong => error.SequenceTooLong,
            error.OutOfMemory => error.OutOfMemory,
        };
    };

    return result_tensor;
}

/// Perform attention with KV cache, supporting sinks and sliding window.
///
/// This is a high-level operation that:
/// 1. Validates tensor dimensions and dtypes
/// 2. Converts tensors to f32 if needed
/// 3. Updates the KV cache
/// 4. Computes scaled dot-product attention
///
/// Q: [batch, n_heads, seq_len, head_dim]
/// K: [batch, n_kv_heads, seq_len, head_dim]
/// V: [batch, n_kv_heads, seq_len, head_dim]
/// sinks: optional [n_heads] - per-head sink logits
pub fn attentionWithSinks(
    allocator: std.mem.Allocator,
    q_tensor: *const Tensor,
    k_tensor: *const Tensor,
    v_tensor: *const Tensor,
    cache_state: *KVCache,
    layer_idx: usize,
    sinks: ?*const Tensor,
    sliding_window: usize,
    scale: f32,
) KVCacheAttentionError!*Tensor {
    // Validate dimensions
    if (layer_idx >= cache_state.n_layers) return error.InvalidArgument;
    if (q_tensor.n_dims != 4 or k_tensor.n_dims != 4 or v_tensor.n_dims != 4) return error.InvalidArgument;

    const batch_size: usize = @intCast(q_tensor.shape[0]);
    const n_heads: usize = @intCast(q_tensor.shape[1]);
    const seq_len: usize = @intCast(q_tensor.shape[2]);
    const head_dim: usize = @intCast(q_tensor.shape[3]);
    const n_kv_heads: usize = @intCast(k_tensor.shape[1]);

    // Validate cache compatibility
    if (n_kv_heads != cache_state.n_kv_heads or head_dim != cache_state.head_dim) return error.InvalidArgument;

    // Validate sinks shape
    if (sinks) |s| {
        if (@as(usize, @intCast(s.n_dims)) != 1) return error.InvalidArgument;
        if (@as(usize, @intCast(s.shape[0])) != n_heads) return error.InvalidArgument;
    }

    // Validate Q/K/V shape compatibility
    if (k_tensor.shape[0] != q_tensor.shape[0]) return error.InvalidArgument;
    if (k_tensor.shape[2] != q_tensor.shape[2]) return error.InvalidArgument;
    if (v_tensor.shape[0] != q_tensor.shape[0]) return error.InvalidArgument;
    if (v_tensor.shape[2] != q_tensor.shape[2]) return error.InvalidArgument;

    // Validate dtypes
    const dtype = q_tensor.simpleDType();
    if (!dtype_convert.isFloatDType(dtype)) return error.UnsupportedDType;
    if (k_tensor.simpleDType() != dtype or v_tensor.simpleDType() != dtype) return error.InvalidArgument;

    // Convert tensors to f32 using helper
    const q_element_count = batch_size * n_heads * seq_len * head_dim;
    const kv_element_count = batch_size * n_kv_heads * seq_len * head_dim;

    var q_conv = dtype_convert.tensorToF32(allocator, q_tensor, q_element_count) catch return error.OutOfMemory;
    defer q_conv.deinit(allocator);
    var k_conv = dtype_convert.tensorToF32(allocator, k_tensor, kv_element_count) catch return error.OutOfMemory;
    defer k_conv.deinit(allocator);
    var v_conv = dtype_convert.tensorToF32(allocator, v_tensor, kv_element_count) catch return error.OutOfMemory;
    defer v_conv.deinit(allocator);

    // Convert sinks if present
    var sinks_conv: ?dtype_convert.F32ConversionResult = null;
    defer if (sinks_conv) |*s| s.deinit(allocator);
    const sinks_slice: ?[]const f32 = if (sinks) |s| blk: {
        sinks_conv = dtype_convert.tensorToF32(allocator, s, n_heads) catch return error.OutOfMemory;
        break :blk sinks_conv.?.data[0..n_heads];
    } else null;

    // Update cache
    const kv_size = cache_state.n_kv_heads * cache_state.head_dim;
    const layer_stride = cache_state.max_seq_len * kv_size;
    const layer_offset = layer_idx * layer_stride;

    const shape_4d = [4]usize{ batch_size, n_kv_heads, seq_len, head_dim };
    const k_strides = if (dtype == .f32)
        dtype_convert.tensorStrides4D(k_tensor)
    else
        dtype_convert.contiguousStrides4D(shape_4d);

    attention.updateKVCache(
        cache_state.k_cache,
        cache_state.v_cache,
        k_conv.data,
        v_conv.data,
        k_strides,
        layer_offset,
        cache_state.seq_pos,
        cache_state.max_seq_len,
        seq_len,
        n_kv_heads,
        head_dim,
    );

    // Allocate output
    const out_shape = [_]i64{ @intCast(batch_size), @intCast(n_heads), @intCast(seq_len), @intCast(head_dim) };
    const result_tensor = Tensor.init(allocator, out_shape[0..4], .f32, q_tensor.device) catch return error.OutOfMemory;
    errdefer result_tensor.deinit(allocator);

    const output_data = @as([*]f32, @ptrCast(@alignCast(result_tensor.data_ptr)));
    const total_seq_len = cache_state.seq_pos + seq_len;
    const cached_seq_len = @min(total_seq_len, cache_state.max_seq_len);
    const scale_value = if (scale == 0) 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))) else scale;

    const q_shape_4d = [4]usize{ batch_size, n_heads, seq_len, head_dim };
    const q_strides = if (dtype == .f32)
        dtype_convert.tensorStrides4D(q_tensor)
    else
        dtype_convert.contiguousStrides4D(q_shape_4d);
    const out_strides = dtype_convert.tensorStrides4D(result_tensor);

    // Delegate to compute layer
    attention.sdpaCached(
        output_data,
        out_strides,
        q_conv.data,
        q_strides,
        cache_state.k_cache[layer_offset..][0 .. cache_state.max_seq_len * kv_size],
        cache_state.v_cache[layer_offset..][0 .. cache_state.max_seq_len * kv_size],
        n_heads,
        n_kv_heads,
        seq_len,
        cached_seq_len,
        head_dim,
        cache_state.seq_pos,
        scale_value,
        sinks_slice,
        sliding_window,
        allocator,
    ) catch |err| {
        result_tensor.deinit(allocator);
        return switch (err) {
            error.SequenceTooLong => error.SequenceTooLong,
            error.OutOfMemory => error.OutOfMemory,
        };
    };

    return result_tensor;
}
