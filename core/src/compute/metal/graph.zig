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
