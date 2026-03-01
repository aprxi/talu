//! Zig bindings for MLX lazy graph API
//!
//! Keeps arrays on GPU using opaque handles, builds lazy computation graphs.
const std = @import("std");

// ============================================================================
// Opaque handles - arrays stay on GPU!
// ============================================================================

/// Opaque handle to MLX array (GPU memory)
pub const ArrayHandle = ?*anyopaque;
pub const MambaCacheHandle = ?*anyopaque;
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
pub extern fn mlx_pool_max_retained() usize;
pub extern fn mlx_pool_clear_if_idle() bool;
pub extern fn mlx_pool_compact_if_idle(max_retained: usize) bool;
pub extern fn mlx_gqa_index_cache_clear() void;
pub extern fn mlx_gqa_index_cache_size() usize;
pub extern fn mlx_gqa_index_cache_max_entries() usize;
pub extern fn mlx_gqa_index_cache_touch(q_heads: usize, kv_heads: usize) void;
pub extern fn mlx_array_ingest_stats(zero_copy_count: *usize, copy_count: *usize) void;
pub extern fn mlx_array_ingest_stats_reset() void;

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
    mask: ArrayHandle, // null for single-step path, causal mask for multi-step path
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

/// Fused ShortConv mixer (dense/bfloat16 path)
pub extern fn mlx_lazy_shortconv_mixer_bf16(
    input: ArrayHandle,
    in_proj: ArrayHandle,
    conv_weight: ArrayHandle,
    conv_bias: ArrayHandle,
    out_proj: ArrayHandle,
    shortconv_cache: ShortConvCacheHandle,
    layer_idx: usize,
    d_conv: usize,
    conv_dim: usize,
) ArrayHandle;

/// ShortConv recurrent state cache lifecycle
pub extern fn mlx_shortconv_cache_create(n_layers: usize) ShortConvCacheHandle;
pub extern fn mlx_shortconv_cache_reset(cache: ShortConvCacheHandle) void;
pub extern fn mlx_shortconv_cache_free(cache: ShortConvCacheHandle) void;

/// Fused Mamba block (dense/bfloat16 path)
pub extern fn mlx_lazy_mamba_block_bf16(
    input: ArrayHandle,
    ln1_weight: ArrayHandle,
    in_proj: ArrayHandle,
    conv_weight: ArrayHandle,
    conv_bias: ArrayHandle,
    a_log: ArrayHandle,
    d_skip: ArrayHandle,
    dt_bias: ArrayHandle,
    norm_weight: ArrayHandle,
    out_proj: ArrayHandle,
    ln2_weight: ArrayHandle,
    gate_up: ArrayHandle,
    down_proj: ArrayHandle,
    use_gelu: bool,
    residual_multiplier: f32,
    norm_eps: f32,
    mamba_cache: MambaCacheHandle,
    layer_idx: usize,
    d_state: usize,
    d_conv: usize,
    n_heads: usize,
    d_head: usize,
    n_groups: usize,
    gate_up_layout: u8,
) ArrayHandle;

/// Mamba recurrent state cache lifecycle
pub extern fn mlx_mamba_cache_create(n_layers: usize) MambaCacheHandle;
pub extern fn mlx_mamba_cache_reset(cache: MambaCacheHandle) void;
pub extern fn mlx_mamba_cache_free(cache: MambaCacheHandle) void;

// ============================================================================
// Graph Execution
// ============================================================================

/// Force evaluation - >>> C++ call: mx.eval()
/// THIS IS THE KEY: executes entire graph on GPU!
pub extern fn mlx_eval(handles: [*]ArrayHandle, n_handles: usize) void;

/// Async evaluation - >>> C++ call: mx.async_eval()
/// Starts GPU work in background, returns immediately.
/// Use for pipelining staged compute dispatches.
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

/// GPU-side argmax - returns array handle with selected index
/// Use to avoid CPU roundtrip during sampling
pub extern fn mlx_lazy_argmax(handle: ArrayHandle, axis: c_int) ArrayHandle;

/// Get scalar u32 value from array (blocks until evaluated)
pub extern fn mlx_array_item_u32(handle: ArrayHandle) u32;

/// Extract last position from 3D logits tensor [B, L, V] -> [V]
/// Used for efficient argmax sampling from the most recent output row.
pub extern fn mlx_lazy_slice_last(handle: ArrayHandle) ArrayHandle;

/// Get array shape
extern fn mlx_array_shape(
    handle: ArrayHandle,
    shape: [*]usize,
    ndim: *usize,
) void;

// ============================================================================
// High-level Zig wrappers for primitive array graph operations
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

    // Free all handles and free again - freeArray must be idempotent.
    for (handles) |h| {
        freeArray(h);
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

test "asyncEval starts evaluation without blocking API contract" {
    if (comptime builtin.os.tag != .macos) return;
    if (!device_mod.isAvailable()) return;

    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const shape = [_]i64{4};
    const a = createArrayF32(&data, &shape);
    defer freeArray(a);
    const doubled = mlx_lazy_multiply_scalar(a, 2.0);
    defer freeArray(doubled);

    var handles = [_]ArrayHandle{doubled};
    asyncEval(&handles);
    // Force completion and validate result.
    eval(&handles);
    var output: [4]f32 = undefined;
    copyToHost(doubled, &output);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), output[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), output[3], 0.01);
}

test "getShape returns ndim and shape entries" {
    if (comptime builtin.os.tag != .macos) return;
    if (!device_mod.isAvailable()) return;

    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const shape = [_]i64{ 2, 2 };
    const handle = createArrayF32(&data, &shape);
    defer freeArray(handle);

    var out_shape = [_]usize{ 0, 0, 0, 0 };
    const ndim = getShape(handle, &out_shape);
    try std.testing.expectEqual(@as(usize, 2), ndim);
    try std.testing.expectEqual(@as(usize, 2), out_shape[0]);
    try std.testing.expectEqual(@as(usize, 2), out_shape[1]);
}

test "mlx_lazy_mamba_block_bf16 prefill matches token-by-token path" {
    if (comptime builtin.os.tag != .macos) return;
    if (!device_mod.isAvailable()) return;

    const seq_len: usize = 4;
    const d_state: usize = 1;
    const d_conv: usize = 2;
    const n_heads: usize = 1;
    const d_head: usize = 1;
    const n_groups: usize = 1;

    const ln1_w_data = [_]f32{1.0};
    const in_proj_data = [_]f32{ 0.6, -0.2, 0.4, -0.3, 0.1 }; // [d_model=1, proj=5]
    const conv_weight_data = [_]f32{
        0.10, 0.20,  -0.05,
        0.05, -0.15, 0.12,
    }; // [d_conv=2, xbc_len=3]
    const conv_bias_data = [_]f32{ 0.01, -0.02, 0.03 };
    const a_log_data = [_]f32{-1.2};
    const d_skip_data = [_]f32{0.5};
    const dt_bias_data = [_]f32{-0.6};
    const out_proj_data = [_]f32{0.9}; // [d_inner=1, d_model=1]
    const input_seq_data = [_]f32{ 0.2, -0.1, 0.3, -0.25 }; // [1, L, 1]

    const s1 = [_]i64{1};
    const in_proj_shape = [_]i64{ 1, 5 };
    const conv_weight_shape = [_]i64{ 2, 3 };
    const s3 = [_]i64{3};
    const out_proj_shape = [_]i64{ 1, 1 };
    const input_seq_shape = [_]i64{ 1, @intCast(seq_len), 1 };
    const input_tok_shape = [_]i64{ 1, 1, 1 };

    const ln1_w = createArrayF32(&ln1_w_data, &s1);
    defer freeArray(ln1_w);
    const in_proj = createArrayF32(&in_proj_data, &in_proj_shape);
    defer freeArray(in_proj);
    const conv_weight = createArrayF32(&conv_weight_data, &conv_weight_shape);
    defer freeArray(conv_weight);
    const conv_bias = createArrayF32(&conv_bias_data, &s3);
    defer freeArray(conv_bias);
    const a_log = createArrayF32(&a_log_data, &s1);
    defer freeArray(a_log);
    const d_skip = createArrayF32(&d_skip_data, &s1);
    defer freeArray(d_skip);
    const dt_bias = createArrayF32(&dt_bias_data, &s1);
    defer freeArray(dt_bias);
    const out_proj = createArrayF32(&out_proj_data, &out_proj_shape);
    defer freeArray(out_proj);

    const prefill_cache = mlx_mamba_cache_create(1);
    defer mlx_mamba_cache_free(prefill_cache);
    const step_cache = mlx_mamba_cache_create(1);
    defer mlx_mamba_cache_free(step_cache);

    const input_seq = createArrayF32(&input_seq_data, &input_seq_shape);
    defer freeArray(input_seq);
    const prefill_out = mlx_lazy_mamba_block_bf16(
        input_seq,
        ln1_w,
        in_proj,
        conv_weight,
        conv_bias,
        a_log,
        d_skip,
        dt_bias,
        null,
        out_proj,
        null,
        null,
        null,
        false,
        1.0,
        1.0e-5,
        prefill_cache,
        0,
        d_state,
        d_conv,
        n_heads,
        d_head,
        n_groups,
        0,
    );
    defer freeArray(prefill_out);

    var prefill_eval_handles = [_]ArrayHandle{prefill_out};
    eval(&prefill_eval_handles);

    var prefill_host: [seq_len]f32 = undefined;
    copyToHost(prefill_out, &prefill_host);

    var step_host: [seq_len]f32 = undefined;
    for (input_seq_data, 0..) |tok, i| {
        const token_data = [_]f32{tok};
        const input_tok = createArrayF32(&token_data, &input_tok_shape);
        const step_out = mlx_lazy_mamba_block_bf16(
            input_tok,
            ln1_w,
            in_proj,
            conv_weight,
            conv_bias,
            a_log,
            d_skip,
            dt_bias,
            null,
            out_proj,
            null,
            null,
            null,
            false,
            1.0,
            1.0e-5,
            step_cache,
            0,
            d_state,
            d_conv,
            n_heads,
            d_head,
            n_groups,
            0,
        );

        var step_eval_handles = [_]ArrayHandle{step_out};
        eval(&step_eval_handles);
        var step_scalar: [1]f32 = undefined;
        copyToHost(step_out, &step_scalar);
        step_host[i] = step_scalar[0];

        freeArray(step_out);
        freeArray(input_tok);
    }

    for (prefill_host, step_host) |prefill_value, step_value| {
        try std.testing.expectApproxEqAbs(prefill_value, step_value, 1.0e-3);
    }
}

test "mlx_lazy_shortconv_mixer_bf16 prefill matches token-by-token path" {
    if (comptime builtin.os.tag != .macos) return;
    if (!device_mod.isAvailable()) return;

    const seq_len: usize = 5;
    const d_model: usize = 2;
    const d_conv: usize = 3;
    const conv_dim: usize = 2;

    const input_seq_data = [_]f32{
        0.2,  -0.1,
        0.3,  0.05,
        -0.2, 0.4,
        0.1,  -0.3,
        0.25, 0.15,
    }; // [1, 5, 2]
    const in_proj_data = [_]f32{
        0.7,  -0.1,
        -0.2, 0.4,
        0.3,  0.5,
        -0.4, 0.2,
        0.1,  -0.6,
        0.2,  0.3,
    }; // [3*conv_dim=6, d_model=2]
    const conv_weight_data = [_]f32{
        0.2,   -0.1, 0.05,
        -0.15, 0.25, 0.1,
    }; // [conv_dim=2, d_conv=3]
    const conv_bias_data = [_]f32{ 0.01, -0.03 };
    const out_proj_data = [_]f32{
        0.8, -0.2,
        0.1, 0.6,
    }; // [d_model=2, conv_dim=2]

    const input_seq_shape = [_]i64{ 1, @intCast(seq_len), @intCast(d_model) };
    const input_tok_shape = [_]i64{ 1, 1, @intCast(d_model) };
    const in_proj_shape = [_]i64{ 3 * @as(i64, @intCast(conv_dim)), @intCast(d_model) };
    const conv_weight_shape = [_]i64{ @intCast(conv_dim), @intCast(d_conv) };
    const conv_bias_shape = [_]i64{@intCast(conv_dim)};
    const out_proj_shape = [_]i64{ @intCast(d_model), @intCast(conv_dim) };

    const input_seq = createArrayF32(&input_seq_data, &input_seq_shape);
    defer freeArray(input_seq);
    const in_proj = createArrayF32(&in_proj_data, &in_proj_shape);
    defer freeArray(in_proj);
    const conv_weight = createArrayF32(&conv_weight_data, &conv_weight_shape);
    defer freeArray(conv_weight);
    const conv_bias = createArrayF32(&conv_bias_data, &conv_bias_shape);
    defer freeArray(conv_bias);
    const out_proj = createArrayF32(&out_proj_data, &out_proj_shape);
    defer freeArray(out_proj);

    const prefill_cache = mlx_shortconv_cache_create(1);
    defer mlx_shortconv_cache_free(prefill_cache);
    const step_cache = mlx_shortconv_cache_create(1);
    defer mlx_shortconv_cache_free(step_cache);

    const prefill_out = mlx_lazy_shortconv_mixer_bf16(
        input_seq,
        in_proj,
        conv_weight,
        conv_bias,
        out_proj,
        prefill_cache,
        0,
        d_conv,
        conv_dim,
    );
    defer freeArray(prefill_out);

    var prefill_eval_handles = [_]ArrayHandle{prefill_out};
    eval(&prefill_eval_handles);

    var prefill_host: [seq_len * d_model]f32 = undefined;
    copyToHost(prefill_out, &prefill_host);

    var step_host: [seq_len * d_model]f32 = undefined;
    for (0..seq_len) |i| {
        const token_data = [_]f32{
            input_seq_data[i * d_model],
            input_seq_data[i * d_model + 1],
        };
        const input_tok = createArrayF32(&token_data, &input_tok_shape);
        const step_out = mlx_lazy_shortconv_mixer_bf16(
            input_tok,
            in_proj,
            conv_weight,
            conv_bias,
            out_proj,
            step_cache,
            0,
            d_conv,
            conv_dim,
        );

        var step_eval_handles = [_]ArrayHandle{step_out};
        eval(&step_eval_handles);

        var out_tok: [d_model]f32 = undefined;
        copyToHost(step_out, &out_tok);
        step_host[i * d_model] = out_tok[0];
        step_host[i * d_model + 1] = out_tok[1];

        freeArray(step_out);
        freeArray(input_tok);
    }

    for (prefill_host, step_host) |prefill_value, step_value| {
        try std.testing.expectApproxEqAbs(prefill_value, step_value, 1.0e-3);
    }
}

test "mlx_lazy_mamba_block_bf16 chunked prefill matches full prefill and next token" {
    if (comptime builtin.os.tag != .macos) return;
    if (!device_mod.isAvailable()) return;

    const seq_len: usize = 5;
    const chunk_len: usize = 2;
    const rem_len: usize = seq_len - chunk_len;
    const d_state: usize = 1;
    const d_conv: usize = 2;
    const n_heads: usize = 1;
    const d_head: usize = 1;
    const n_groups: usize = 1;

    const ln1_w_data = [_]f32{1.0};
    const in_proj_data = [_]f32{ 0.6, -0.2, 0.4, -0.3, 0.1 };
    const conv_weight_data = [_]f32{
        0.10, 0.20,  -0.05,
        0.05, -0.15, 0.12,
    };
    const conv_bias_data = [_]f32{ 0.01, -0.02, 0.03 };
    const a_log_data = [_]f32{-1.2};
    const d_skip_data = [_]f32{0.5};
    const dt_bias_data = [_]f32{-0.6};
    const out_proj_data = [_]f32{0.9};
    const input_seq_data = [_]f32{ 0.2, -0.1, 0.3, -0.25, 0.15 };
    const next_token_data = [_]f32{-0.05};

    const s1 = [_]i64{1};
    const in_proj_shape = [_]i64{ 1, 5 };
    const conv_weight_shape = [_]i64{ 2, 3 };
    const s3 = [_]i64{3};
    const out_proj_shape = [_]i64{ 1, 1 };
    const input_seq_shape = [_]i64{ 1, @intCast(seq_len), 1 };
    const chunk_shape = [_]i64{ 1, @intCast(chunk_len), 1 };
    const rem_shape = [_]i64{ 1, @intCast(rem_len), 1 };
    const input_tok_shape = [_]i64{ 1, 1, 1 };

    const input_seq = createArrayF32(&input_seq_data, &input_seq_shape);
    defer freeArray(input_seq);
    const input_chunk = createArrayF32(input_seq_data[0..chunk_len], &chunk_shape);
    defer freeArray(input_chunk);
    const input_rem = createArrayF32(input_seq_data[chunk_len..], &rem_shape);
    defer freeArray(input_rem);
    const input_next = createArrayF32(&next_token_data, &input_tok_shape);
    defer freeArray(input_next);

    const ln1_w = createArrayF32(&ln1_w_data, &s1);
    defer freeArray(ln1_w);
    const in_proj = createArrayF32(&in_proj_data, &in_proj_shape);
    defer freeArray(in_proj);
    const conv_weight = createArrayF32(&conv_weight_data, &conv_weight_shape);
    defer freeArray(conv_weight);
    const conv_bias = createArrayF32(&conv_bias_data, &s3);
    defer freeArray(conv_bias);
    const a_log = createArrayF32(&a_log_data, &s1);
    defer freeArray(a_log);
    const d_skip = createArrayF32(&d_skip_data, &s1);
    defer freeArray(d_skip);
    const dt_bias = createArrayF32(&dt_bias_data, &s1);
    defer freeArray(dt_bias);
    const out_proj = createArrayF32(&out_proj_data, &out_proj_shape);
    defer freeArray(out_proj);

    const full_cache = mlx_mamba_cache_create(1);
    defer mlx_mamba_cache_free(full_cache);
    const chunk_cache = mlx_mamba_cache_create(1);
    defer mlx_mamba_cache_free(chunk_cache);

    const full_out = mlx_lazy_mamba_block_bf16(
        input_seq,
        ln1_w,
        in_proj,
        conv_weight,
        conv_bias,
        a_log,
        d_skip,
        dt_bias,
        null,
        out_proj,
        null,
        null,
        null,
        false,
        1.0,
        1.0e-5,
        full_cache,
        0,
        d_state,
        d_conv,
        n_heads,
        d_head,
        n_groups,
        0,
    );
    defer freeArray(full_out);
    var full_eval = [_]ArrayHandle{full_out};
    eval(&full_eval);

    var full_host: [seq_len]f32 = undefined;
    copyToHost(full_out, &full_host);

    const chunk_out_1 = mlx_lazy_mamba_block_bf16(
        input_chunk,
        ln1_w,
        in_proj,
        conv_weight,
        conv_bias,
        a_log,
        d_skip,
        dt_bias,
        null,
        out_proj,
        null,
        null,
        null,
        false,
        1.0,
        1.0e-5,
        chunk_cache,
        0,
        d_state,
        d_conv,
        n_heads,
        d_head,
        n_groups,
        0,
    );
    defer freeArray(chunk_out_1);
    var chunk_eval_1 = [_]ArrayHandle{chunk_out_1};
    eval(&chunk_eval_1);

    const chunk_out_2 = mlx_lazy_mamba_block_bf16(
        input_rem,
        ln1_w,
        in_proj,
        conv_weight,
        conv_bias,
        a_log,
        d_skip,
        dt_bias,
        null,
        out_proj,
        null,
        null,
        null,
        false,
        1.0,
        1.0e-5,
        chunk_cache,
        0,
        d_state,
        d_conv,
        n_heads,
        d_head,
        n_groups,
        0,
    );
    defer freeArray(chunk_out_2);
    var chunk_eval_2 = [_]ArrayHandle{chunk_out_2};
    eval(&chunk_eval_2);

    var chunk_host_1: [chunk_len]f32 = undefined;
    var chunk_host_2: [rem_len]f32 = undefined;
    copyToHost(chunk_out_1, &chunk_host_1);
    copyToHost(chunk_out_2, &chunk_host_2);

    var stitched: [seq_len]f32 = undefined;
    for (0..chunk_len) |i| stitched[i] = chunk_host_1[i];
    for (0..rem_len) |i| stitched[chunk_len + i] = chunk_host_2[i];

    for (full_host, stitched) |full_value, chunked_value| {
        try std.testing.expectApproxEqAbs(full_value, chunked_value, 1.0e-3);
    }

    const full_next = mlx_lazy_mamba_block_bf16(
        input_next,
        ln1_w,
        in_proj,
        conv_weight,
        conv_bias,
        a_log,
        d_skip,
        dt_bias,
        null,
        out_proj,
        null,
        null,
        null,
        false,
        1.0,
        1.0e-5,
        full_cache,
        0,
        d_state,
        d_conv,
        n_heads,
        d_head,
        n_groups,
        0,
    );
    defer freeArray(full_next);
    var full_next_eval = [_]ArrayHandle{full_next};
    eval(&full_next_eval);
    var full_next_host: [1]f32 = undefined;
    copyToHost(full_next, &full_next_host);

    const chunk_next = mlx_lazy_mamba_block_bf16(
        input_next,
        ln1_w,
        in_proj,
        conv_weight,
        conv_bias,
        a_log,
        d_skip,
        dt_bias,
        null,
        out_proj,
        null,
        null,
        null,
        false,
        1.0,
        1.0e-5,
        chunk_cache,
        0,
        d_state,
        d_conv,
        n_heads,
        d_head,
        n_groups,
        0,
    );
    defer freeArray(chunk_next);
    var chunk_next_eval = [_]ArrayHandle{chunk_next};
    eval(&chunk_next_eval);
    var chunk_next_host: [1]f32 = undefined;
    copyToHost(chunk_next, &chunk_next_host);

    try std.testing.expectApproxEqAbs(full_next_host[0], chunk_next_host[0], 1.0e-3);
}

test "mamba op count scales sublinearly across sequence lengths" {
    if (comptime builtin.os.tag != .macos) return;
    if (!device_mod.isAvailable()) return;

    const d_state: usize = 1;
    const d_conv: usize = 2;
    const n_heads: usize = 1;
    const d_head: usize = 1;
    const n_groups: usize = 1;

    const ln1_w_data = [_]f32{1.0};
    const in_proj_data = [_]f32{ 0.6, -0.2, 0.4, -0.3, 0.1 };
    const conv_weight_data = [_]f32{
        0.10, 0.20,  -0.05,
        0.05, -0.15, 0.12,
    };
    const conv_bias_data = [_]f32{ 0.01, -0.02, 0.03 };
    const a_log_data = [_]f32{-1.2};
    const d_skip_data = [_]f32{0.5};
    const dt_bias_data = [_]f32{-0.6};
    const out_proj_data = [_]f32{0.9};
    const input_max_data = [_]f32{
        0.2,  -0.1,  0.3,  -0.25, 0.15,  -0.05, 0.4,   -0.3,
        0.11, -0.08, 0.06, 0.19,  -0.12, 0.09,  -0.02, 0.07,
    };

    const s1 = [_]i64{1};
    const in_proj_shape = [_]i64{ 1, 5 };
    const conv_weight_shape = [_]i64{ 2, 3 };
    const s3 = [_]i64{3};
    const out_proj_shape = [_]i64{ 1, 1 };

    const ln1_w = createArrayF32(&ln1_w_data, &s1);
    defer freeArray(ln1_w);
    const in_proj = createArrayF32(&in_proj_data, &in_proj_shape);
    defer freeArray(in_proj);
    const conv_weight = createArrayF32(&conv_weight_data, &conv_weight_shape);
    defer freeArray(conv_weight);
    const conv_bias = createArrayF32(&conv_bias_data, &s3);
    defer freeArray(conv_bias);
    const a_log = createArrayF32(&a_log_data, &s1);
    defer freeArray(a_log);
    const d_skip = createArrayF32(&d_skip_data, &s1);
    defer freeArray(d_skip);
    const dt_bias = createArrayF32(&dt_bias_data, &s1);
    defer freeArray(dt_bias);
    const out_proj = createArrayF32(&out_proj_data, &out_proj_shape);
    defer freeArray(out_proj);

    const cache = mlx_mamba_cache_create(1);
    defer mlx_mamba_cache_free(cache);

    const seq_small: usize = 2;
    const seq_large: usize = 16;
    const shape_small = [_]i64{ 1, @intCast(seq_small), 1 };
    const shape_large = [_]i64{ 1, @intCast(seq_large), 1 };

    const run_count = struct {
        fn run(
            input_data: []const f32,
            shape: []const i64,
            cache_handle: MambaCacheHandle,
            ln1_weight: ArrayHandle,
            in_proj_weight: ArrayHandle,
            conv_w: ArrayHandle,
            conv_b: ArrayHandle,
            a_log_arr: ArrayHandle,
            d_skip_arr: ArrayHandle,
            dt_bias_arr: ArrayHandle,
            out_proj_weight: ArrayHandle,
            d_state_: usize,
            d_conv_: usize,
            n_heads_: usize,
            d_head_: usize,
            n_groups_: usize,
        ) usize {
            mlx_mamba_cache_reset(cache_handle);
            const input = createArrayF32(input_data, shape);
            defer freeArray(input);
            mlx_start_counting();
            const out = mlx_lazy_mamba_block_bf16(
                input,
                ln1_weight,
                in_proj_weight,
                conv_w,
                conv_b,
                a_log_arr,
                d_skip_arr,
                dt_bias_arr,
                null,
                out_proj_weight,
                null,
                null,
                null,
                false,
                1.0,
                1.0e-5,
                cache_handle,
                0,
                d_state_,
                d_conv_,
                n_heads_,
                d_head_,
                n_groups_,
                0,
            );
            defer freeArray(out);
            var handles = [_]ArrayHandle{out};
            eval(&handles);
            return mlx_stop_counting();
        }
    };

    _ = run_count.run(
        input_max_data[0..seq_small],
        &shape_small,
        cache,
        ln1_w,
        in_proj,
        conv_weight,
        conv_bias,
        a_log,
        d_skip,
        dt_bias,
        out_proj,
        d_state,
        d_conv,
        n_heads,
        d_head,
        n_groups,
    );
    const ops_small = run_count.run(
        input_max_data[0..seq_small],
        &shape_small,
        cache,
        ln1_w,
        in_proj,
        conv_weight,
        conv_bias,
        a_log,
        d_skip,
        dt_bias,
        out_proj,
        d_state,
        d_conv,
        n_heads,
        d_head,
        n_groups,
    );
    _ = run_count.run(
        input_max_data[0..seq_large],
        &shape_large,
        cache,
        ln1_w,
        in_proj,
        conv_weight,
        conv_bias,
        a_log,
        d_skip,
        dt_bias,
        out_proj,
        d_state,
        d_conv,
        n_heads,
        d_head,
        n_groups,
    );
    const ops_large = run_count.run(
        input_max_data[0..seq_large],
        &shape_large,
        cache,
        ln1_w,
        in_proj,
        conv_weight,
        conv_bias,
        a_log,
        d_skip,
        dt_bias,
        out_proj,
        d_state,
        d_conv,
        n_heads,
        d_head,
        n_groups,
    );

    try std.testing.expect(ops_small > 0);
    try std.testing.expect(ops_large <= (ops_small * 3 + 32));
}

test "shortconv op count scales sublinearly across sequence lengths" {
    if (comptime builtin.os.tag != .macos) return;
    if (!device_mod.isAvailable()) return;

    const d_model: usize = 2;
    const d_conv: usize = 3;
    const conv_dim: usize = 2;
    const seq_small: usize = 2;
    const seq_large: usize = 16;

    const input_max_data = [_]f32{
        0.2,   -0.1,  0.3,   0.05,  -0.2,  0.4,   0.1,   -0.3,
        0.25,  0.15,  -0.14, 0.08,  0.06,  -0.07, 0.11,  0.09,
        -0.12, 0.03,  0.21,  -0.17, 0.05,  0.04,  -0.09, 0.02,
        0.18,  -0.16, 0.07,  0.01,  -0.03, 0.13,  0.22,  -0.05,
    }; // [1,16,2]
    const in_proj_data = [_]f32{
        0.7,  -0.1,
        -0.2, 0.4,
        0.3,  0.5,
        -0.4, 0.2,
        0.1,  -0.6,
        0.2,  0.3,
    }; // [6,2]
    const conv_weight_data = [_]f32{
        0.2,   -0.1, 0.05,
        -0.15, 0.25, 0.1,
    }; // [2,3]
    const conv_bias_data = [_]f32{ 0.01, -0.03 };
    const out_proj_data = [_]f32{
        0.8, -0.2,
        0.1, 0.6,
    }; // [2,2]

    const in_proj_shape = [_]i64{ 3 * @as(i64, @intCast(conv_dim)), @intCast(d_model) };
    const conv_weight_shape = [_]i64{ @intCast(conv_dim), @intCast(d_conv) };
    const conv_bias_shape = [_]i64{@intCast(conv_dim)};
    const out_proj_shape = [_]i64{ @intCast(d_model), @intCast(conv_dim) };
    const input_small_shape = [_]i64{ 1, @intCast(seq_small), @intCast(d_model) };
    const input_large_shape = [_]i64{ 1, @intCast(seq_large), @intCast(d_model) };

    const in_proj = createArrayF32(&in_proj_data, &in_proj_shape);
    defer freeArray(in_proj);
    const conv_weight = createArrayF32(&conv_weight_data, &conv_weight_shape);
    defer freeArray(conv_weight);
    const conv_bias = createArrayF32(&conv_bias_data, &conv_bias_shape);
    defer freeArray(conv_bias);
    const out_proj = createArrayF32(&out_proj_data, &out_proj_shape);
    defer freeArray(out_proj);

    const cache = mlx_shortconv_cache_create(1);
    defer mlx_shortconv_cache_free(cache);

    const run_count = struct {
        fn run(
            input_data: []const f32,
            shape: []const i64,
            cache_handle: ShortConvCacheHandle,
            in_proj_weight: ArrayHandle,
            conv_w: ArrayHandle,
            conv_b: ArrayHandle,
            out_proj_weight: ArrayHandle,
            d_conv_: usize,
            conv_dim_: usize,
        ) usize {
            mlx_shortconv_cache_reset(cache_handle);
            const input = createArrayF32(input_data, shape);
            defer freeArray(input);
            mlx_start_counting();
            const out = mlx_lazy_shortconv_mixer_bf16(
                input,
                in_proj_weight,
                conv_w,
                conv_b,
                out_proj_weight,
                cache_handle,
                0,
                d_conv_,
                conv_dim_,
            );
            defer freeArray(out);
            var handles = [_]ArrayHandle{out};
            eval(&handles);
            return mlx_stop_counting();
        }
    };

    _ = run_count.run(
        input_max_data[0 .. seq_small * d_model],
        &input_small_shape,
        cache,
        in_proj,
        conv_weight,
        conv_bias,
        out_proj,
        d_conv,
        conv_dim,
    );
    const ops_small = run_count.run(
        input_max_data[0 .. seq_small * d_model],
        &input_small_shape,
        cache,
        in_proj,
        conv_weight,
        conv_bias,
        out_proj,
        d_conv,
        conv_dim,
    );
    _ = run_count.run(
        input_max_data[0 .. seq_large * d_model],
        &input_large_shape,
        cache,
        in_proj,
        conv_weight,
        conv_bias,
        out_proj,
        d_conv,
        conv_dim,
    );
    const ops_large = run_count.run(
        input_max_data[0 .. seq_large * d_model],
        &input_large_shape,
        cache,
        in_proj,
        conv_weight,
        conv_bias,
        out_proj,
        d_conv,
        conv_dim,
    );

    try std.testing.expect(ops_small > 0);
    try std.testing.expect(ops_large <= (ops_small * 3 + 32));
}

test "array pool clear and compact require idle reset barrier" {
    if (comptime builtin.os.tag != .macos) return;
    if (!device_mod.isAvailable()) return;

    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const shape = [_]i64{ 2, 2 };
    const a = createArrayF32(&data, &shape);
    defer freeArray(a);
    const b = createArrayF32(&data, &shape);
    defer freeArray(b);

    const out = mlx_lazy_add(a, b);
    var eval_handles = [_]ArrayHandle{out};
    eval(&eval_handles);

    try std.testing.expect(!mlx_pool_clear_if_idle());
    try std.testing.expect(!mlx_pool_compact_if_idle(1));

    mlx_pool_reset();
    try std.testing.expect(mlx_pool_compact_if_idle(1));

    var pool_size: usize = 0;
    var used: usize = 0;
    mlx_pool_stats(&pool_size, &used);
    try std.testing.expect(used == 0);
    try std.testing.expect(pool_size <= 1);
    try std.testing.expect(mlx_pool_clear_if_idle());
}

test "gqa index cache is bounded and resettable" {
    if (comptime builtin.os.tag != .macos) return;

    mlx_gqa_index_cache_clear();
    const max_entries = mlx_gqa_index_cache_max_entries();
    try std.testing.expect(max_entries > 0);

    var kv_heads: usize = 1;
    var touched: usize = 0;
    while (touched < max_entries * 2) : (touched += 1) {
        mlx_gqa_index_cache_touch(kv_heads * 2, kv_heads);
        kv_heads += 1;
    }

    try std.testing.expect(mlx_gqa_index_cache_size() <= max_entries);
    mlx_gqa_index_cache_clear();
    try std.testing.expect(mlx_gqa_index_cache_size() == 0);
}

test "array ingest uses zero-copy for aligned pointers and copy for unaligned pointers" {
    if (comptime builtin.os.tag != .macos) return;

    const shape = [_]usize{ 2, 2 };
    const bytes_len = shape[0] * shape[1] * @sizeOf(f32);
    mlx_array_ingest_stats_reset();

    const page_align = std.heap.page_size_min;
    const aligned_bytes = try std.heap.c_allocator.alignedAlloc(u8, .fromByteUnits(page_align), bytes_len);
    defer std.heap.c_allocator.free(aligned_bytes);
    const aligned_f32 = std.mem.bytesAsSlice(f32, aligned_bytes[0..bytes_len]);
    aligned_f32[0] = 1.0;
    aligned_f32[1] = 2.0;
    aligned_f32[2] = 3.0;
    aligned_f32[3] = 4.0;

    const aligned_arr = mlx_array_from_float32(@ptrCast(aligned_f32.ptr), &shape, shape.len);
    defer freeArray(aligned_arr);

    var unaligned_storage: [bytes_len + 1]u8 = undefined;
    @memcpy(unaligned_storage[1 .. 1 + bytes_len], aligned_bytes[0..bytes_len]);
    const unaligned_ptr: *const anyopaque = @ptrCast(unaligned_storage[1 .. 1 + bytes_len].ptr);
    const unaligned_arr = mlx_array_from_float32(unaligned_ptr, &shape, shape.len);
    defer freeArray(unaligned_arr);

    var zero_copy_count: usize = 0;
    var copy_count: usize = 0;
    mlx_array_ingest_stats(&zero_copy_count, &copy_count);

    try std.testing.expect(zero_copy_count >= 1);
    try std.testing.expect(copy_count >= 1);
}
