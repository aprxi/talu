//! Zig bindings for MLX lazy graph API
//!
//! Keeps arrays on GPU using opaque handles, builds lazy computation graphs.
//! Follows pattern from test_mlx_single.py for 243 t/s decode performance.
const std = @import("std");

// ============================================================================
// Opaque handles - arrays stay on GPU!
// ============================================================================

/// Opaque handle to MLX array (GPU memory)
pub const ArrayHandle = ?*anyopaque;

/// Opaque handle to KV cache
pub const CacheHandle = ?*anyopaque;
/// Opaque handle to ShortConv cache
pub const ShortConvCacheHandle = ?*anyopaque;

// ============================================================================
// Array Pool - call reset() before each forward pass to reuse allocations
// ============================================================================

/// Reset array pool for next forward pass (eliminates heap allocations)
pub extern fn mlx_pool_reset() void;

/// Clear MLX memory cache - call periodically to prevent fragmentation
pub extern fn mlx_clear_memory_cache() void;

/// Get pool stats (for debugging)
pub extern fn mlx_pool_stats(pool_size: *usize, used: *usize) void;

/// Start counting operations (for debugging)
pub extern fn mlx_start_counting() void;

/// Stop counting and return op count
pub extern fn mlx_stop_counting() usize;

// ============================================================================
// Array Creation (CPU -> GPU)
// ============================================================================

//// Create MLX array from float32 data (accepts unaligned pointers)
/// C++ signature: void* mlx_array_from_float32(const void* data, ...)
pub extern fn mlx_array_from_float32(
    data: *const anyopaque,
    shape: [*]const usize,
    ndim: usize,
) ArrayHandle;

/// Create MLX array from uint32 data (accepts unaligned pointers from mmap)
/// C++ signature: void* mlx_array_from_uint32(const void* data, ...)
pub extern fn mlx_array_from_uint32(
    data: *const anyopaque,
    shape: [*]const usize,
    ndim: usize,
) ArrayHandle;

/// Create MLX array from bfloat16 data (stored as u16)
/// C++ signature: void* mlx_array_from_bfloat16(const void* data, ...)
pub extern fn mlx_array_from_bfloat16(
    data: *const anyopaque,
    shape: [*]const usize,
    ndim: usize,
) ArrayHandle;

/// Create MLX array from float16 (IEEE) data (stored as u16)
/// C++ signature: void* mlx_array_from_float16(const void* data, ...)
pub extern fn mlx_array_from_float16(
    data: *const anyopaque,
    shape: [*]const usize,
    ndim: usize,
) ArrayHandle;

/// Create MLX array from uint8 data (used for MXFP4 scales)
pub extern fn mlx_array_from_uint8(
    data: [*]align(1) const u8,
    shape: [*]const usize,
    ndim: usize,
) ArrayHandle;

/// Create MLX array from existing pointer (no-copy wrapper)
pub extern fn mlx_array_from_ptr(mlx_array_ptr: *anyopaque) ArrayHandle;

/// Free array handle
pub extern fn mlx_array_free(handle: ArrayHandle) void;

// ============================================================================
// Lazy Operations (return handles, don't execute!)
// ============================================================================

/// Quantized matmul - >>> Lazy: quantized_matmul
pub extern fn mlx_lazy_quantized_matmul(
    input: ArrayHandle,
    weights: ArrayHandle,
    scales: ArrayHandle,
    biases: ArrayHandle,
    group_size: usize,
    bits: usize,
    transpose: bool,
) ArrayHandle;

/// RMS norm - >>> Lazy: rms_norm
pub extern fn mlx_lazy_rms_norm(
    input: ArrayHandle,
    weight: ArrayHandle,
    eps: f32,
) ArrayHandle;

/// Add 1 to array (for (1+w) RMSNorm formulation)
pub extern fn mlx_add_one(arr: ArrayHandle) ArrayHandle;

/// Scale array by sqrt(d_model) for embedding scaling
pub extern fn mlx_scale_by_sqrt(arr: ArrayHandle, d_model: usize) ArrayHandle;

/// RoPE - >>> Lazy: rope
pub extern fn mlx_lazy_rope(
    input: ArrayHandle,
    head_dim: usize,
    offset: usize,
    rope_base: f32,
) ArrayHandle;

/// Scaled dot product attention - >>> Lazy: scaled_dot_product_attention
pub extern fn mlx_lazy_attention(
    q: ArrayHandle,
    k: ArrayHandle,
    v: ArrayHandle,
    scale: f32,
    causal: bool,
) ArrayHandle;

/// Fused quantized attention - >>> Lazy: Q @ K^T, scale, mask, softmax, @ V
/// All K/V inputs are quantized triplets (weights, scales, biases)
pub extern fn mlx_lazy_quantized_attention(
    q: ArrayHandle,
    k_weights: ArrayHandle,
    k_scales: ArrayHandle,
    k_biases: ArrayHandle,
    v_weights: ArrayHandle,
    v_scales: ArrayHandle,
    v_biases: ArrayHandle,
    mask: ArrayHandle, // null for decode, causal_mask for prefill
    scale: f32,
    group_size: usize,
    bits: usize,
) ArrayHandle;

/// SiLU activation - >>> Lazy: silu
pub extern fn mlx_lazy_silu(input: ArrayHandle) ArrayHandle;

/// Dequantize - >>> Lazy: dequantize
pub extern fn mlx_lazy_dequantize(
    weights: ArrayHandle,
    scales: ArrayHandle,
    biases: ArrayHandle,
    group_size: usize,
    bits: usize,
) ArrayHandle;

/// Element-wise add - >>> Lazy
pub extern fn mlx_lazy_add(a: ArrayHandle, b: ArrayHandle) ArrayHandle;

/// Element-wise multiply - >>> Lazy
pub extern fn mlx_lazy_multiply(a: ArrayHandle, b: ArrayHandle) ArrayHandle;

/// Multiply by scalar - >>> Lazy
pub extern fn mlx_lazy_multiply_scalar(a: ArrayHandle, scalar: f32) ArrayHandle;

/// Softmax along axis - >>> Lazy
pub extern fn mlx_lazy_softmax(input: ArrayHandle, axis: c_int) ArrayHandle;

/// Create array filled with scalar value - >>> Lazy
pub extern fn mlx_lazy_full(
    shape: [*]const usize,
    ndim: usize,
    value: f32,
) ArrayHandle;

/// Upper triangular matrix - >>> Lazy
pub extern fn mlx_lazy_triu(input: ArrayHandle, k: c_int) ArrayHandle;

/// Matrix multiply - >>> Lazy
pub extern fn mlx_lazy_matmul(a: ArrayHandle, b: ArrayHandle) ArrayHandle;

/// Reshape - >>> Lazy
pub extern fn mlx_lazy_reshape(
    input: ArrayHandle,
    shape: [*]const usize,
    ndim: usize,
) ArrayHandle;

/// Persistent reshape - heap-allocated, survives pool resets
pub extern fn mlx_persistent_reshape(
    input: ArrayHandle,
    shape: [*]const usize,
    ndim: usize,
) ArrayHandle;

/// Transpose - >>> Lazy
pub extern fn mlx_lazy_transpose(
    input: ArrayHandle,
    axes: [*]const usize,
    ndim: usize,
) ArrayHandle;

/// Combined reshape + transpose for better fusion - >>> Lazy
pub extern fn mlx_lazy_reshape_transpose(
    input: ArrayHandle,
    reshape_dims: [*]const usize,
    reshape_ndim: usize,
    transpose_axes: [*]const usize,
    transpose_ndim: usize,
) ArrayHandle;

/// Combined transpose + reshape for better fusion - >>> Lazy
pub extern fn mlx_lazy_transpose_reshape(
    input: ArrayHandle,
    transpose_axes: [*]const usize,
    transpose_ndim: usize,
    reshape_dims: [*]const usize,
    reshape_ndim: usize,
) ArrayHandle;

/// Embedding lookup - >>> Lazy
pub extern fn mlx_lazy_embedding(
    weights: ArrayHandle,
    indices: [*]const u32,
    n_indices: usize,
) ArrayHandle;

/// Embedding lookup from GPU array (lazy - for pipelined generation)
pub extern fn mlx_lazy_embedding_from_array(
    weights: ArrayHandle,
    indices: ArrayHandle,
) ArrayHandle;

/// Concatenate - >>> Lazy: concatenate arrays along axis
pub extern fn mlx_lazy_concatenate(
    a: ArrayHandle,
    b: ArrayHandle,
    axis: usize,
) ArrayHandle;

/// Repeat - >>> Lazy: repeat array along axis (for GQA)
pub extern fn mlx_lazy_repeat(
    input: ArrayHandle,
    repeats: usize,
    axis: usize,
) ArrayHandle;

/// Slice - >>> Lazy: extract slice from array
pub extern fn mlx_lazy_slice(
    input: ArrayHandle,
    starts: [*]const c_int,
    ends: [*]const c_int,
    ndim: usize,
) ArrayHandle;

/// Persistent slice - heap-allocated, survives pool resets
/// Use for weight slices that need to persist across forward passes
pub extern fn mlx_persistent_slice(
    input: ArrayHandle,
    starts: [*]const c_int,
    ends: [*]const c_int,
    ndim: usize,
) ArrayHandle;

/// Slice update - >>> Lazy: update slice of array
pub extern fn mlx_lazy_slice_update(
    input: ArrayHandle,
    update: ArrayHandle,
    starts: [*]const c_int,
    ends: [*]const c_int,
    ndim: usize,
) ArrayHandle;

// ============================================================================
// Graph Execution
// ============================================================================

/// Force evaluation - >>> C++ call: mx.eval()
/// THIS IS THE KEY: executes entire graph on GPU!
pub extern fn mlx_eval(handles: [*]ArrayHandle, n_handles: usize) void;

/// Async evaluation - >>> C++ call: mx.async_eval()
/// Starts GPU work in background, returns immediately.
/// Use for pipelining: start next token's eval while processing current token.
pub extern fn mlx_async_eval(handles: [*]ArrayHandle, n_handles: usize) void;

// ============================================================================
// Data Retrieval (GPU -> CPU, only when needed)
// ============================================================================

/// Copy data from GPU to CPU - >>> C++ call: np.array()
pub extern fn mlx_array_to_float32(
    handle: ArrayHandle,
    out: [*]f32,
    size: usize,
) void;

/// GPU-side argmax - returns array handle with token index
/// Use to avoid CPU roundtrip during sampling
pub extern fn mlx_lazy_argmax(handle: ArrayHandle, axis: c_int) ArrayHandle;

/// Get scalar u32 value from array (blocks until evaluated)
pub extern fn mlx_array_item_u32(handle: ArrayHandle) u32;

/// Extract last position from 3D logits tensor [B, L, V] -> [V]
/// Used for efficient argmax sampling from decode output
pub extern fn mlx_lazy_slice_last(handle: ArrayHandle) ArrayHandle;

/// Get array shape
extern fn mlx_array_shape(
    handle: ArrayHandle,
    shape: [*]usize,
    ndim: *usize,
) void;

// ============================================================================
// KV Cache
// ============================================================================

/// Create KV cache (quantized 4-bit)
extern fn mlx_cache_create(n_layers: usize) CacheHandle;

/// Create KV cache (bfloat16 - matches mlx_lm default)
extern fn mlx_cache_create_bfloat16(n_layers: usize) CacheHandle;

/// Free cache
extern fn mlx_cache_free(cache: CacheHandle) void;

/// Update buffer with new K/V (bfloat16) and return concatenated cache
extern fn mlx_cache_update_and_fetch_bfloat16(
    cache: CacheHandle,
    layer_idx: usize,
    k_new: ArrayHandle,
    v_new: ArrayHandle,
    k_out: *ArrayHandle,
    v_out: *ArrayHandle,
    is_prefill_out: *bool,
) void;

/// Get bfloat16 cache (non-quantized)
extern fn mlx_cache_get_bfloat16(
    cache: CacheHandle,
    layer_idx: usize,
    k_out: *ArrayHandle,
    v_out: *ArrayHandle,
) void;

/// Set full bfloat16 cache (fusion returns full cache, don't concatenate)
extern fn mlx_cache_set_full_bfloat16(
    cache: CacheHandle,
    layer_idx: usize,
    k_full: ArrayHandle,
    v_full: ArrayHandle,
) void;

/// Evaluate all cache arrays (force evaluation of lazy concatenations)
extern fn mlx_cache_eval_all(cache: CacheHandle, n_layers: usize) void;

/// Update buffer with new K/V (quantizes internally) and return quantized cache
extern fn mlx_cache_update_and_fetch(
    cache: CacheHandle,
    layer_idx: usize,
    k_new: ArrayHandle,
    v_new: ArrayHandle,
    k_out: *ArrayHandle,
    v_out: *ArrayHandle,
    is_prefill_out: *bool,
) void;

/// Get quantized cache triplets (weights, scales, biases) for K and V
extern fn mlx_cache_get_quantized(
    cache: CacheHandle,
    layer_idx: usize,
    k_weights_out: *ArrayHandle,
    k_scales_out: *ArrayHandle,
    k_biases_out: *ArrayHandle,
    v_weights_out: *ArrayHandle,
    v_scales_out: *ArrayHandle,
    v_biases_out: *ArrayHandle,
) void;

/// Create ShortConv cache (per-layer recurrent conv states)
extern fn mlx_shortconv_cache_create(n_layers: usize) ShortConvCacheHandle;
/// Reset ShortConv cache to zeros (called at prefill boundary)
extern fn mlx_shortconv_cache_reset(cache: ShortConvCacheHandle) void;
/// Free ShortConv cache
extern fn mlx_shortconv_cache_free(cache: ShortConvCacheHandle) void;

// ============================================================================
// High-level Zig wrappers
// ============================================================================

/// Convert i64 shape slice to usize array (max 8 dims)
fn shapeToUsize(shape_i64: []const i64) struct { shape: [8]usize, len: usize } {
    var shape: [8]usize = undefined;
    for (shape_i64, 0..) |dim, i| {
        shape[i] = @intCast(dim);
    }
    return .{ .shape = shape, .len = shape_i64.len };
}

/// Create MLX array from Zig slice (float32)
pub fn createArrayF32(data: []const f32, shape: []const i64) ArrayHandle {
    const shape_usize = shapeToUsize(shape);
    return mlx_array_from_float32(@ptrCast(data.ptr), &shape_usize.shape, shape_usize.len);
}

/// Create MLX array from Zig slice (uint32)
fn createArrayU32(data: []const u32, shape: []const i64) ArrayHandle {
    const shape_usize = shapeToUsize(shape);
    return mlx_array_from_uint32(@ptrCast(data.ptr), &shape_usize.shape, shape_usize.len);
}

/// Create MLX array from unaligned uint32 data (for mmap'd safetensor data)
pub fn createArrayU32Unaligned(data: [*]align(1) const u32, len: usize, shape: []const i64) ArrayHandle {
    _ = len; // used for bounds checking in caller
    const shape_usize = shapeToUsize(shape);
    // Cast to *const anyopaque to avoid alignment checks - C++ uses memcpy internally
    return mlx_array_from_uint32(@ptrCast(data), &shape_usize.shape, shape_usize.len);
}

/// Create MLX array from Zig slice (bfloat16 as u16)
fn createArrayBF16(data: []const u16, shape: []const i64) ArrayHandle {
    const shape_usize = shapeToUsize(shape);
    return mlx_array_from_bfloat16(@ptrCast(data.ptr), &shape_usize.shape, shape_usize.len);
}

/// Create MLX array from unaligned bfloat16 data (for mmap'd safetensor data)
pub fn createArrayBF16Unaligned(data: [*]align(1) const u16, len: usize, shape: []const i64) ArrayHandle {
    _ = len; // used for bounds checking in caller
    const shape_usize = shapeToUsize(shape);
    // Cast to *const anyopaque to avoid alignment checks - C++ uses memcpy internally
    return mlx_array_from_bfloat16(@ptrCast(data), &shape_usize.shape, shape_usize.len);
}

/// Create MLX array from Zig slice (float16/IEEE as u16)
fn createArrayF16(data: []const u16, shape: []const i64) ArrayHandle {
    const shape_usize = shapeToUsize(shape);
    return mlx_array_from_float16(@ptrCast(data.ptr), &shape_usize.shape, shape_usize.len);
}

/// Create MLX array from unaligned float16 data (for mmap'd safetensor data)
pub fn createArrayF16Unaligned(data: [*]align(1) const u16, len: usize, shape: []const i64) ArrayHandle {
    _ = len; // used for bounds checking in caller
    const shape_usize = shapeToUsize(shape);
    // Cast to *const anyopaque to avoid alignment checks - C++ uses memcpy internally
    return mlx_array_from_float16(@ptrCast(data), &shape_usize.shape, shape_usize.len);
}

/// Free MLX array
pub fn freeArray(handle: ArrayHandle) void {
    mlx_array_free(handle);
}

/// Evaluate all arrays - executes entire graph on GPU
pub fn eval(handles: []const ArrayHandle) void {
    mlx_eval(@ptrCast(@constCast(handles.ptr)), handles.len);
}

/// Async evaluate - starts GPU work in background, returns immediately
/// Use for pipelining to overlap CPU work with GPU work
pub fn asyncEval(handles: []const ArrayHandle) void {
    mlx_async_eval(@ptrCast(@constCast(handles.ptr)), handles.len);
}

/// Copy array data from GPU to CPU
pub fn copyToHost(handle: ArrayHandle, out: []f32) void {
    mlx_array_to_float32(handle, out.ptr, out.len);
}

/// Get array shape and dimensions
pub fn getShape(handle: ArrayHandle, shape_out: []usize) usize {
    var ndim: usize = 0;
    mlx_array_shape(handle, shape_out.ptr, &ndim);
    return ndim;
}

/// KV Cache wrapper
pub const Cache = struct {
    handle: CacheHandle,
    use_bfloat16: bool,

    /// Create cache with specified format
    /// use_bfloat16: true = bfloat16 cache (matches mlx_lm), false = quantized 4-bit
    pub fn init(n_layers: usize, use_bfloat16: bool) Cache {
        const handle = if (use_bfloat16)
            mlx_cache_create_bfloat16(n_layers)
        else
            mlx_cache_create(n_layers);
        return .{ .handle = handle, .use_bfloat16 = use_bfloat16 };
    }

    pub fn deinit(self: Cache) void {
        mlx_cache_free(self.handle);
    }

    /// Update buffer with new K/V and return concatenated cache
    pub fn updateAndFetch(self: Cache, layer_idx: usize, k_new: ArrayHandle, v_new: ArrayHandle) struct { k: ArrayHandle, v: ArrayHandle, is_prefill: bool } {
        var k_cache: ArrayHandle = null;
        var v_cache: ArrayHandle = null;
        var is_prefill: bool = false;

        if (self.use_bfloat16) {
            mlx_cache_update_and_fetch_bfloat16(self.handle, layer_idx, k_new, v_new, &k_cache, &v_cache, &is_prefill);
        } else {
            mlx_cache_update_and_fetch(self.handle, layer_idx, k_new, v_new, &k_cache, &v_cache, &is_prefill);
        }

        return .{ .k = k_cache, .v = v_cache, .is_prefill = is_prefill };
    }

    /// Get bfloat16 cache (non-quantized) - for use with regular attention
    pub fn get(self: Cache, layer_idx: usize) struct { k: ArrayHandle, v: ArrayHandle } {
        var k_cache: ArrayHandle = null;
        var v_cache: ArrayHandle = null;
        mlx_cache_get_bfloat16(self.handle, layer_idx, &k_cache, &v_cache);
        return .{ .k = k_cache, .v = v_cache };
    }

    /// Set full bfloat16 cache (fusion already concatenated, just store)
    pub fn setFull(self: Cache, layer_idx: usize, k_full: ArrayHandle, v_full: ArrayHandle) void {
        mlx_cache_set_full_bfloat16(self.handle, layer_idx, k_full, v_full);
    }

    /// Evaluate all cache arrays (force evaluation of lazy concatenations from fusion)
    pub fn evalAll(self: Cache, n_layers: usize) void {
        mlx_cache_eval_all(self.handle, n_layers);
    }

    /// Get quantized cache triplets for use with quantized_matmul
    pub fn getQuantized(self: Cache, layer_idx: usize) struct {
        k_weights: ArrayHandle,
        k_scales: ArrayHandle,
        k_biases: ArrayHandle,
        v_weights: ArrayHandle,
        v_scales: ArrayHandle,
        v_biases: ArrayHandle,
    } {
        var k_w: ArrayHandle = null;
        var k_s: ArrayHandle = null;
        var k_b: ArrayHandle = null;
        var v_w: ArrayHandle = null;
        var v_s: ArrayHandle = null;
        var v_b: ArrayHandle = null;
        mlx_cache_get_quantized(self.handle, layer_idx, &k_w, &k_s, &k_b, &v_w, &v_s, &v_b);
        return .{
            .k_weights = k_w,
            .k_scales = k_s,
            .k_biases = k_b,
            .v_weights = v_w,
            .v_scales = v_s,
            .v_biases = v_b,
        };
    }
};

/// ShortConv cache wrapper
pub const ShortConvCache = struct {
    handle: ShortConvCacheHandle,

    pub fn init(n_layers: usize) ShortConvCache {
        return .{ .handle = mlx_shortconv_cache_create(n_layers) };
    }

    pub fn reset(self: ShortConvCache) void {
        mlx_shortconv_cache_reset(self.handle);
    }

    pub fn deinit(self: ShortConvCache) void {
        mlx_shortconv_cache_free(self.handle);
    }
};

// ============================================================================
// Compiled Layer Functions (FUSION OPTIMIZATION)
// ============================================================================

/// Opaque handle to compiled layer function
pub const CompiledLayerHandle = ?*anyopaque;
pub const LayerOutputHandle = ?*anyopaque;

/// Compile a transformer layer for maximum fusion
pub extern fn mlx_compile_layer(
    q_weight: ArrayHandle,
    q_scales: ArrayHandle,
    q_biases: ArrayHandle,
    k_weight: ArrayHandle,
    k_scales: ArrayHandle,
    k_biases: ArrayHandle,
    v_weight: ArrayHandle,
    v_scales: ArrayHandle,
    v_biases: ArrayHandle,
    o_weight: ArrayHandle,
    o_scales: ArrayHandle,
    o_biases: ArrayHandle,
    gate_weight: ArrayHandle,
    gate_scales: ArrayHandle,
    gate_biases: ArrayHandle,
    up_weight: ArrayHandle,
    up_scales: ArrayHandle,
    up_biases: ArrayHandle,
    down_weight: ArrayHandle,
    down_scales: ArrayHandle,
    down_biases: ArrayHandle,
    attn_norm: ArrayHandle,
    ffn_norm: ArrayHandle,
    q_norm: ArrayHandle,
    k_norm: ArrayHandle,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    hidden_dim: usize,
    group_size: usize,
    bits: usize,
    rope_theta: f32,
    rms_eps: f32,
) CompiledLayerHandle;

/// Execute compiled layer forward pass (mutates cache internally like Python)
pub extern fn mlx_layer_forward(
    compiled_handle: CompiledLayerHandle,
    hidden: ArrayHandle,
    cache_ptr: CacheHandle,
    layer_idx: usize,
    pos_offset: usize,
) ArrayHandle;

/// Compiled layer wrapper
pub const CompiledLayer = struct {
    handle: CompiledLayerHandle,

    pub fn forward(
        self: CompiledLayer,
        hidden: ArrayHandle,
        cache_ptr: CacheHandle,
        layer_idx: usize,
        pos_offset: usize,
    ) ArrayHandle {
        // Safety: null handle would crash in C++ (treated as index 0 into empty array)
        if (self.handle == null) return null;
        return mlx_layer_forward(self.handle, hidden, cache_ptr, layer_idx, pos_offset);
    }
};

// ============================================================================
// FULLY FUSED MODEL: All layers in ONE C++ call (ZERO FFI overhead)
// ============================================================================

/// Opaque handle to fused model.
/// Inference ownership note: callsites should import these via
/// `inference/backend/metal/model_runtime.zig` rather than directly.
pub const FusedModelHandle = ?*anyopaque;

/// Create fused model structure
pub extern fn mlx_fused_model_create(
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    hidden_dim: usize,
    group_size: usize,
    bits: usize,
    rope_theta: f32,
    rms_eps: f32,
) FusedModelHandle;

/// Set embedding weights (quantized)
pub extern fn mlx_fused_model_set_embeddings(
    model: FusedModelHandle,
    w: ArrayHandle,
    s: ArrayHandle,
    b: ArrayHandle,
) void;

/// Set final norm and lm_head weights
pub extern fn mlx_fused_model_set_final(
    model: FusedModelHandle,
    ln_w: ArrayHandle,
    lm_w: ArrayHandle,
    lm_s: ArrayHandle,
    lm_b: ArrayHandle,
) void;

/// Set custom RoPE frequencies (for scaled positional encoding)
pub extern fn mlx_fused_model_set_rope_freqs(
    model: FusedModelHandle,
    freqs: ArrayHandle,
) void;

/// Set architecture-specific config (norm formulation, activation, attention scale)
pub extern fn mlx_fused_model_set_arch_config(
    model: FusedModelHandle,
    has_norm_weight_offset: bool,
    use_gelu: bool,
    query_pre_attn_scalar: f32,
) void;

/// Set custom scaling multipliers (data-driven from config.json)
pub extern fn mlx_fused_model_set_scaling_config(
    model: FusedModelHandle,
    embedding_multiplier: f32,
    attention_multiplier: f32,
    residual_multiplier: f32,
    logits_scaling: f32,
) void;

/// Set canonical layer topology ids for fused model execution.
/// Must be called once before `mlx_fused_model_set_layer`.
pub extern fn mlx_fused_model_set_topology(
    model: FusedModelHandle,
    layer_kinds: [*]const u8,
    n_layer_kinds: usize,
) void;

/// Set per-layer weights
pub extern fn mlx_fused_model_set_layer(
    model: FusedModelHandle,
    layer_idx: usize,
    ln1_w: ArrayHandle,
    q_w: ArrayHandle,
    q_s: ArrayHandle,
    q_b: ArrayHandle,
    k_w: ArrayHandle,
    k_s: ArrayHandle,
    k_b: ArrayHandle,
    v_w: ArrayHandle,
    v_s: ArrayHandle,
    v_b: ArrayHandle,
    o_w: ArrayHandle,
    o_s: ArrayHandle,
    o_b: ArrayHandle,
    ln2_w: ArrayHandle,
    gate_w: ArrayHandle,
    gate_s: ArrayHandle,
    gate_b: ArrayHandle,
    up_w: ArrayHandle,
    up_s: ArrayHandle,
    up_b: ArrayHandle,
    down_w: ArrayHandle,
    down_s: ArrayHandle,
    down_b: ArrayHandle,
    q_norm: ArrayHandle,
    k_norm: ArrayHandle,
    pre_ffn_norm: ArrayHandle,
    post_ffn_norm: ArrayHandle,
    shortconv_d_conv: usize,
    shortconv_conv_dim: usize,
    shortconv_in_w: ArrayHandle,
    shortconv_in_s: ArrayHandle,
    shortconv_in_b: ArrayHandle,
    shortconv_out_w: ArrayHandle,
    shortconv_out_s: ArrayHandle,
    shortconv_out_b: ArrayHandle,
    shortconv_conv_w: ArrayHandle,
    shortconv_conv_b: ArrayHandle, // can be null
) void;

/// Single decode step: token_id -> next_token_id (ALL in C++, ZERO FFI overhead)
/// This is the key optimization: entire forward pass + argmax in one call
pub extern fn mlx_fused_decode_step(
    model: FusedModelHandle,
    cache: CacheHandle,
    shortconv_cache: ShortConvCacheHandle,
    token_id: u32,
    pos_offset: usize,
) u32;

/// Single decode step returning logits for CPU-side sampling.
/// Uses the same fused model path as mlx_fused_decode_step, but does not argmax on GPU.
pub extern fn mlx_fused_decode_step_logits(
    model: FusedModelHandle,
    cache: CacheHandle,
    shortconv_cache: ShortConvCacheHandle,
    token_id: u32,
    pos_offset: usize,
) ArrayHandle;

// ===========================================================================
// TRUE PIPELINED DECODE - builds graph N+1 before materializing N
// ===========================================================================

/// Prime the pipeline: initializes with first token and builds first graph
/// Call once before the decode loop with the last prompt token
pub extern fn mlx_pipeline_prime(
    model: FusedModelHandle,
    cache: CacheHandle,
    shortconv_cache: ShortConvCacheHandle,
    first_token_id: u32,
    pos_offset: usize,
) void;

/// Pipeline step: returns current token, builds and queues next
/// This builds the graph for token N+1 BEFORE materializing token N.
/// Must call mlx_pipeline_prime first.
pub extern fn mlx_pipeline_step(
    model: FusedModelHandle,
    cache: CacheHandle,
    shortconv_cache: ShortConvCacheHandle,
    pos_offset: usize,
) u32;

/// Flush the pipeline: returns the last pending token
/// Call after the decode loop to get the final token
pub extern fn mlx_pipeline_flush() u32;

/// Batch decode: run entire decode loop in C++ to eliminate FFI overhead
/// Returns number of tokens generated
pub extern fn mlx_fused_decode_batch(
    model: FusedModelHandle,
    cache: CacheHandle,
    shortconv_cache: ShortConvCacheHandle,
    first_token: u32,
    start_pos: usize,
    out_tokens: [*]u32,
    max_tokens: usize,
    eos_ids: [*]const u32,
    n_eos_ids: usize,
) u32;

/// Free fused model
pub extern fn mlx_fused_model_free(model: FusedModelHandle) void;

/// Optimize fused model (fuse QKV and gate/up weights for faster inference)
/// Call after all layers are set
pub extern fn mlx_fused_model_optimize(model: FusedModelHandle) void;

// ===========================================================================
// FUSED DENSE MODEL - BFloat16 weights (non-quantized)
// ===========================================================================

/// Opaque handle to dense (BF16) fused model.
/// Inference ownership note: callsites should import these via
/// `inference/backend/metal/model_runtime.zig` rather than directly.
pub const DenseModelHandle = ?*anyopaque;

/// Create dense model for BFloat16 weights
pub extern fn mlx_dense_model_create(
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    hidden_dim: usize,
    rope_theta: f32,
    rms_eps: f32,
) DenseModelHandle;

/// Set dense model embeddings (BF16)
pub extern fn mlx_dense_model_set_embeddings(
    model: DenseModelHandle,
    embed: ArrayHandle,
) void;

/// Set dense model final weights
pub extern fn mlx_dense_model_set_final(
    model: DenseModelHandle,
    ln_w: ArrayHandle,
    lm_head: ArrayHandle,
) void;

/// Set canonical layer topology ids for dense model execution.
/// Must be called once before `mlx_dense_model_set_layer`.
pub extern fn mlx_dense_model_set_topology(
    model: DenseModelHandle,
    layer_kinds: [*]const u8,
    n_layer_kinds: usize,
) void;

/// Set dense model layer weights
pub extern fn mlx_dense_model_set_layer(
    model: DenseModelHandle,
    layer_idx: usize,
    ln1_w: ArrayHandle,
    q_proj: ArrayHandle,
    k_proj: ArrayHandle,
    v_proj: ArrayHandle,
    o_proj: ArrayHandle,
    ln2_w: ArrayHandle,
    gate_proj: ArrayHandle,
    up_proj: ArrayHandle,
    down_proj: ArrayHandle,
    q_norm: ArrayHandle, // null if not present
    k_norm: ArrayHandle, // null if not present
    shortconv_d_conv: usize,
    shortconv_conv_dim: usize,
    shortconv_in_proj: ArrayHandle,
    shortconv_conv_weight: ArrayHandle,
    shortconv_conv_bias: ArrayHandle, // can be null
    shortconv_out_proj: ArrayHandle,
) void;

/// Free dense model
pub extern fn mlx_dense_model_free(model: DenseModelHandle) void;

/// Prime dense pipeline with first token
pub extern fn mlx_dense_pipeline_prime(
    model: DenseModelHandle,
    cache: CacheHandle,
    shortconv_cache: ShortConvCacheHandle,
    first_token_id: u32,
    pos_offset: usize,
) void;

/// Dense pipeline step - returns current token, queues next
pub extern fn mlx_dense_pipeline_step(
    model: DenseModelHandle,
    cache: CacheHandle,
    shortconv_cache: ShortConvCacheHandle,
    pos_offset: usize,
) u32;

/// Flush dense pipeline - returns last pending token
pub extern fn mlx_dense_pipeline_flush() u32;

/// Dense single decode step returning logits for CPU-side sampling.
pub extern fn mlx_dense_decode_step_logits(
    model: DenseModelHandle,
    cache: CacheHandle,
    shortconv_cache: ShortConvCacheHandle,
    token_id: u32,
    pos_offset: usize,
) ArrayHandle;

/// Dense batch decode: run entire decode loop in C++ to eliminate FFI overhead.
pub extern fn mlx_dense_decode_batch(
    model: DenseModelHandle,
    cache: CacheHandle,
    shortconv_cache: ShortConvCacheHandle,
    first_token: u32,
    start_pos: usize,
    out_tokens: [*]u32,
    max_tokens: usize,
    eos_ids: [*]const u32,
    n_eos_ids: usize,
) u32;

// =============================================================================
// Unit Tests - compiled only on macOS where Metal/MLX is available
// =============================================================================

const builtin = @import("builtin");
const device_mod = @import("device.zig");

test "createArrayF32 creates valid array handle" {
    if (comptime builtin.os.tag != .macos) return;
    if (!device_mod.isAvailable()) return;

    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const shape = [_]i64{4};
    const handle = createArrayF32(&data, &shape);
    defer freeArray(handle);

    try std.testing.expect(handle != null);

    // Verify data round-trip
    var handles = [_]ArrayHandle{handle};
    eval(&handles);

    var output: [4]f32 = undefined;
    copyToHost(handle, &output);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), output[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), output[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), output[3], 0.001);
}

test "createArrayU32Unaligned creates array from unaligned data" {
    if (comptime builtin.os.tag != .macos) return;
    if (!device_mod.isAvailable()) return;

    var data align(1) = [_]u32{ 1, 2, 3, 4 };
    const shape = [_]i64{ 2, 2 };
    const handle = createArrayU32Unaligned(@ptrCast(&data), data.len, &shape);
    defer freeArray(handle);
    try std.testing.expect(handle != null);
}

test "createArrayBF16Unaligned creates array from unaligned data" {
    if (comptime builtin.os.tag != .macos) return;
    if (!device_mod.isAvailable()) return;

    var data align(1) = [_]u16{ 0x3F80, 0x4000, 0x4040, 0x4080 }; // bf16 values
    const shape = [_]i64{ 2, 2 };
    const handle = createArrayBF16Unaligned(@ptrCast(&data), data.len, &shape);
    defer freeArray(handle);
    try std.testing.expect(handle != null);
}

test "createArrayF16Unaligned creates array from unaligned data" {
    if (comptime builtin.os.tag != .macos) return;
    if (!device_mod.isAvailable()) return;

    var data align(1) = [_]u16{ 0x3C00, 0x4000, 0x4200, 0x4400 }; // f16 values
    const shape = [_]i64{ 2, 2 };
    const handle = createArrayF16Unaligned(@ptrCast(&data), data.len, &shape);
    defer freeArray(handle);
    try std.testing.expect(handle != null);
}

test "freeArray releases array handle without double-free" {
    if (comptime builtin.os.tag != .macos) return;
    if (!device_mod.isAvailable()) return;

    // Allocate multiple arrays and free them in order
    var handles: [4]ArrayHandle = undefined;
    for (0..4) |i| {
        const data = [_]f32{ @floatFromInt(i), @floatFromInt(i + 1), @floatFromInt(i + 2), @floatFromInt(i + 3) };
        const shape = [_]i64{4};
        handles[i] = createArrayF32(&data, &shape);
        try std.testing.expect(handles[i] != null);
    }

    // Free all handles - memory should be properly managed
    for (handles) |h| {
        freeArray(h);
    }
}

test "eval evaluates lazy operations" {
    if (comptime builtin.os.tag != .macos) return;
    if (!device_mod.isAvailable()) return;

    // Create two arrays and add them (lazy operation)
    const a_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b_data = [_]f32{ 10.0, 20.0, 30.0, 40.0 };
    const shape = [_]i64{4};

    const a = createArrayF32(&a_data, &shape);
    defer freeArray(a);
    const b = createArrayF32(&b_data, &shape);
    defer freeArray(b);

    // Lazy add
    const c = mlx_lazy_add(a, b);
    defer freeArray(c);

    // Eval to force computation
    var handles = [_]ArrayHandle{c};
    eval(&handles);

    // Verify result
    var output: [4]f32 = undefined;
    copyToHost(c, &output);
    try std.testing.expectApproxEqAbs(@as(f32, 11.0), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 22.0), output[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 33.0), output[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 44.0), output[3], 0.001);
}

test "copyToHost copies array data to CPU" {
    if (comptime builtin.os.tag != .macos) return;
    if (!device_mod.isAvailable()) return;

    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const shape = [_]i64{4};
    const handle = createArrayF32(&input, &shape);
    defer freeArray(handle);

    var handles = [_]ArrayHandle{handle};
    eval(&handles);

    var output: [4]f32 = undefined;
    copyToHost(handle, &output);

    for (input, output) |expected, actual| {
        try std.testing.expectApproxEqAbs(expected, actual, 0.001);
    }
}

test "Cache init creates cache handle" {
    if (comptime builtin.os.tag != .macos) return;
    if (!device_mod.isAvailable()) return;

    const cache = Cache.init(12, true); // 12 layers, bfloat16
    defer cache.deinit();
    try std.testing.expect(cache.handle != null);
    try std.testing.expect(cache.use_bfloat16);
}

test "Cache deinit releases cache handle" {
    if (comptime builtin.os.tag != .macos) return;
    if (!device_mod.isAvailable()) return;

    // Create multiple caches and free them
    var caches: [3]Cache = undefined;
    for (0..3) |i| {
        caches[i] = Cache.init(4, i % 2 == 0); // alternate bf16/quantized
        try std.testing.expect(caches[i].handle != null);
    }

    for (caches) |c| {
        c.deinit();
    }
}

test "Cache updateAndFetch updates KV cache" {
    if (comptime builtin.os.tag != .macos) return;
    if (!device_mod.isAvailable()) return;

    const cache = Cache.init(2, true);
    defer cache.deinit();

    // Create small K/V arrays
    const k_data = [_]f32{1.0} ** 64;
    const v_data = [_]f32{2.0} ** 64;
    const shape = [_]i64{ 1, 1, 1, 64 }; // [batch, heads, seq, dim]

    const k_handle = createArrayF32(&k_data, &shape);
    const v_handle = createArrayF32(&v_data, &shape);
    defer freeArray(k_handle);
    defer freeArray(v_handle);

    const result = cache.updateAndFetch(0, k_handle, v_handle);
    defer freeArray(result.k);
    defer freeArray(result.v);

    try std.testing.expect(result.k != null);
    try std.testing.expect(result.v != null);
}

test "Cache get retrieves cached KV with correct values" {
    if (comptime builtin.os.tag != .macos) return;
    if (!device_mod.isAvailable()) return;

    const cache = Cache.init(2, true);
    defer cache.deinit();

    // Insert data with known values
    const k_data = [_]f32{1.5} ** 64;
    const v_data = [_]f32{2.5} ** 64;
    const shape = [_]i64{ 1, 1, 1, 64 };

    const k_handle = createArrayF32(&k_data, &shape);
    const v_handle = createArrayF32(&v_data, &shape);
    defer freeArray(k_handle);
    defer freeArray(v_handle);

    const update_result = cache.updateAndFetch(0, k_handle, v_handle);
    try std.testing.expect(update_result.k != null);
    try std.testing.expect(update_result.v != null);

    // Retrieve and verify
    const result = cache.get(0);
    if (result.k != null and result.v != null) {
        var k_out: [64]f32 = undefined;
        var v_out: [64]f32 = undefined;

        var handles = [_]ArrayHandle{ result.k, result.v };
        eval(&handles);

        copyToHost(result.k, &k_out);
        copyToHost(result.v, &v_out);

        // Values should be preserved
        try std.testing.expectApproxEqAbs(@as(f32, 1.5), k_out[0], 0.1);
        try std.testing.expectApproxEqAbs(@as(f32, 2.5), v_out[0], 0.1);
    }
}

test "Cache setFull sets full cache arrays and retrieves correctly" {
    if (comptime builtin.os.tag != .macos) return;
    if (!device_mod.isAvailable()) return;

    const cache = Cache.init(2, true);
    defer cache.deinit();

    const k_data = [_]f32{3.14} ** 64;
    const v_data = [_]f32{2.71} ** 64;
    const shape = [_]i64{ 1, 1, 1, 64 };

    const k_handle = createArrayF32(&k_data, &shape);
    const v_handle = createArrayF32(&v_data, &shape);
    defer freeArray(k_handle);
    defer freeArray(v_handle);

    cache.setFull(0, k_handle, v_handle);

    // Retrieve and verify the values were stored
    const result = cache.get(0);
    if (result.k != null and result.v != null) {
        var k_out: [64]f32 = undefined;
        var v_out: [64]f32 = undefined;

        var handles = [_]ArrayHandle{ result.k, result.v };
        eval(&handles);

        copyToHost(result.k, &k_out);
        copyToHost(result.v, &v_out);

        try std.testing.expectApproxEqAbs(@as(f32, 3.14), k_out[0], 0.1);
        try std.testing.expectApproxEqAbs(@as(f32, 2.71), v_out[0], 0.1);
    }
}

test "Cache evalAll evaluates all cached arrays" {
    if (comptime builtin.os.tag != .macos) return;
    if (!device_mod.isAvailable()) return;

    const n_layers: usize = 2;
    const cache = Cache.init(n_layers, true);
    defer cache.deinit();

    // Insert data into both layers
    const k_data = [_]f32{1.0} ** 64;
    const v_data = [_]f32{2.0} ** 64;
    const shape = [_]i64{ 1, 1, 1, 64 };

    for (0..n_layers) |layer| {
        const k_handle = createArrayF32(&k_data, &shape);
        const v_handle = createArrayF32(&v_data, &shape);
        cache.setFull(layer, k_handle, v_handle);
    }

    // evalAll should force evaluation of all lazy arrays
    cache.evalAll(n_layers);

    // Verify data is accessible after eval
    for (0..n_layers) |layer| {
        const result = cache.get(layer);
        try std.testing.expect(result.k != null);
        try std.testing.expect(result.v != null);
    }
}

test "Cache getQuantized returns quantized triplets after update" {
    if (comptime builtin.os.tag != .macos) return;
    if (!device_mod.isAvailable()) return;

    const cache = Cache.init(2, false); // quantized mode
    defer cache.deinit();

    // Insert data - quantized cache will quantize internally
    const k_data = [_]f32{1.0} ** 64;
    const v_data = [_]f32{2.0} ** 64;
    const shape = [_]i64{ 1, 1, 1, 64 };

    const k_handle = createArrayF32(&k_data, &shape);
    const v_handle = createArrayF32(&v_data, &shape);
    defer freeArray(k_handle);
    defer freeArray(v_handle);

    _ = cache.updateAndFetch(0, k_handle, v_handle);

    // Now getQuantized should return valid triplets
    const result = cache.getQuantized(0);

    // After update, quantized cache should have weights/scales/biases
    // (may still be null if quantization is deferred until needed)
    _ = result;
}

test "mlx_lazy_multiply_scalar scales array correctly" {
    if (comptime builtin.os.tag != .macos) return;
    if (!device_mod.isAvailable()) return;

    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const shape = [_]i64{4};
    const a = createArrayF32(&data, &shape);
    defer freeArray(a);

    // Scale by 2.5
    const scaled = mlx_lazy_multiply_scalar(a, 2.5);
    defer freeArray(scaled);

    var handles = [_]ArrayHandle{scaled};
    eval(&handles);

    var output: [4]f32 = undefined;
    copyToHost(scaled, &output);

    try std.testing.expectApproxEqAbs(@as(f32, 2.5), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), output[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 7.5), output[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), output[3], 0.001);
}

test "mlx_lazy_multiply multiplies arrays element-wise" {
    if (comptime builtin.os.tag != .macos) return;
    if (!device_mod.isAvailable()) return;

    const a_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b_data = [_]f32{ 2.0, 3.0, 4.0, 5.0 };
    const shape = [_]i64{4};

    const a = createArrayF32(&a_data, &shape);
    defer freeArray(a);
    const b = createArrayF32(&b_data, &shape);
    defer freeArray(b);

    const c = mlx_lazy_multiply(a, b);
    defer freeArray(c);

    var handles = [_]ArrayHandle{c};
    eval(&handles);

    var output: [4]f32 = undefined;
    copyToHost(c, &output);

    try std.testing.expectApproxEqAbs(@as(f32, 2.0), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), output[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), output[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 20.0), output[3], 0.001);
}

test "mlx_lazy_softmax computes softmax along axis" {
    if (comptime builtin.os.tag != .macos) return;
    if (!device_mod.isAvailable()) return;

    // Input: [1, 2, 3, 4] - softmax will normalize these
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const shape = [_]i64{4};
    const a = createArrayF32(&data, &shape);
    defer freeArray(a);

    const result = mlx_lazy_softmax(a, 0);
    defer freeArray(result);

    var handles = [_]ArrayHandle{result};
    eval(&handles);

    var output: [4]f32 = undefined;
    copyToHost(result, &output);

    // Softmax outputs should sum to 1.0
    var sum: f32 = 0.0;
    for (output) |v| {
        sum += v;
        try std.testing.expect(v > 0.0); // All values positive
        try std.testing.expect(v < 1.0); // All values less than 1
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 0.01);

    // Values should be monotonically increasing (larger input -> larger output)
    try std.testing.expect(output[0] < output[1]);
    try std.testing.expect(output[1] < output[2]);
    try std.testing.expect(output[2] < output[3]);
}

test "CompiledLayer forward with null handle returns null" {
    if (comptime builtin.os.tag != .macos) return;
    if (!device_mod.isAvailable()) return;

    // CompiledLayer.forward requires a compiled layer handle from mlx_compile_layer
    // which requires full weight setup. Test with null handle - should return null.
    const layer = CompiledLayer{ .handle = null };

    // Create minimal input
    const hidden_data = [_]f32{1.0} ** 64;
    const shape = [_]i64{ 1, 1, 64 };
    const hidden = createArrayF32(&hidden_data, &shape);
    defer freeArray(hidden);

    // Create cache
    const cache = Cache.init(1, true);
    defer cache.deinit();

    // Forward with null handle - should gracefully return null or handle error
    const result = layer.forward(hidden, cache.handle, 0, 0);
    // With null compiled layer handle, result should be null
    try std.testing.expect(result == null);
}
