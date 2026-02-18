//! DLPack C API
//!
//! Zero-copy tensor interop with PyTorch, JAX, and other DLPack consumers.
//! Provides functions to export token arrays and batch encodings as DLPack tensors.
//!
//! Memory Safety Contract:
//! - DLPack export shares storage with the source TokenArray/SharedBuffer
//! - If a consumer mutates the exported tensor, the underlying storage is mutated
//! - TokenArray APIs treat tokens as immutable; mutation from consumers is allowed
//!   but not recommended
//! - If isolation is needed, clone/copy in the consumer framework
//!
//! Refcounted Buffer Design:
//! - TokenArrays use SharedBuffer for refcounted memory management
//! - Each DLPack export increments the buffer's refcount
//! - The DLPack deleter decrements the refcount (and frees if it reaches 0)
//! - This allows the source TokenArray to remain valid after export
//! - Multiple exports from the same TokenArray are safe

const std = @import("std");
const tensor = @import("../tensor.zig");
const compute = @import("../compute/root.zig");
const padding = compute.cpu.padding;
const capi_error = @import("error.zig");
const buffer_mod = @import("buffer.zig");
const SharedBuffer = buffer_mod.SharedBuffer;
const BufferHandle = buffer_mod.BufferHandle;

// Native build uses C allocator
const allocator = std.heap.c_allocator;

// =============================================================================
// Token Array DLPack (1D) - Refcounted Version
// =============================================================================

/// Metadata for token array DLPack export with refcounted buffer.
/// The deleter releases the buffer reference instead of freeing directly.
const TokenDLPackContext = struct {
    /// Reference to the shared buffer (refcount was incremented on export)
    buffer: *SharedBuffer,
    /// Offset into the buffer (for future view support, currently always 0)
    offset_elems: usize,
    /// Number of elements in this view
    len: usize,
    /// Shape array for DLTensor (1D: just len)
    shape: [1]i64,
    /// Strides array (1 element per token)
    strides: [1]i64,
};

/// DLPack deleter for token arrays with refcounted buffers.
/// Called by PyTorch/JAX when the DLPack capsule is consumed.
/// Safe to call from any thread (including non-Python threads).
fn tokenDLPackDeleter(managed: *tensor.DLManagedTensor) callconv(.c) void {
    if (managed.manager_ctx) |ctx| {
        const token_context: *TokenDLPackContext = @ptrCast(@alignCast(ctx));
        // Release our reference to the shared buffer
        // This may free the buffer if we were the last holder
        _ = token_context.buffer.release();
        // Free the context struct
        allocator.destroy(token_context);
    }
    // Free the DLManagedTensor itself
    allocator.destroy(managed);
}

/// Validation error for buffer-to-DLPack conversion.
const BufferDLPackError = enum { null_handle, zero_len, bounds_exceeded };

/// Validated parameters for buffer-to-DLPack conversion.
const BufferDLPackParams = struct { buf: *SharedBuffer, offset_elems: usize, len: usize };

/// Validate buffer-to-DLPack parameters. Returns null on error (error set via setBufferError).
fn validateBufferParams(buffer_handle: ?*BufferHandle, offset_elems: usize, len: usize) ?BufferDLPackParams {
    const buf: *SharedBuffer = @ptrCast(@alignCast(buffer_handle orelse {
        setBufferError(.null_handle);
        return null;
    }));
    if (len == 0) {
        setBufferError(.zero_len);
        return null;
    }
    if (offset_elems + len > buf.capacity) {
        setBufferError(.bounds_exceeded);
        return null;
    }
    return .{ .buf = buf, .offset_elems = offset_elems, .len = len };
}

/// Set buffer validation error message.
fn setBufferError(err: BufferDLPackError) void {
    switch (err) {
        .null_handle => capi_error.setError(error.InvalidArgument, "buffer_handle is null", .{}),
        .zero_len => capi_error.setError(error.InvalidArgument, "len is 0", .{}),
        .bounds_exceeded => capi_error.setError(error.InvalidArgument, "offset + len exceeds buffer capacity", .{}),
    }
}

/// Create DLManagedTensor from validated buffer parameters.
/// Caller must have already retained the buffer. On error, releases buffer.
fn createTokenDLPack(buf: *SharedBuffer, offset_elems: usize, len: usize) ?*tensor.DLManagedTensor {
    const token_context = allocator.create(TokenDLPackContext) catch |err| {
        _ = buf.release();
        capi_error.setError(err, "Failed to allocate DLPack context: {s}", .{@errorName(err)});
        return null;
    };

    token_context.* = .{
        .buffer = buf,
        .offset_elems = offset_elems,
        .len = len,
        .shape = .{@intCast(len)},
        .strides = .{1},
    };

    const managed = allocator.create(tensor.DLManagedTensor) catch |err| {
        allocator.destroy(token_context);
        _ = buf.release();
        capi_error.setError(err, "Failed to allocate DLManagedTensor: {s}", .{@errorName(err)});
        return null;
    };

    const data_ptr: [*]u32 = buf.data + offset_elems;
    managed.* = .{
        .dl_tensor = .{
            .data = @ptrCast(data_ptr),
            .device = .{ .device_type = .kDLCPU, .device_id = 0 },
            .ndim = 1,
            .dtype = .{ .code = .kDLUInt, .bits = 32, .lanes = 1 },
            .shape = &token_context.shape,
            .strides = &token_context.strides,
            .byte_offset = 0,
        },
        .manager_ctx = token_context,
        .deleter = &tokenDLPackDeleter,
    };

    return managed;
}

/// Convert a SharedBuffer to DLPack format for zero-copy export to PyTorch/JAX.
///
/// REFCOUNTED SEMANTICS:
/// - The buffer's refcount is incremented (buffer remains valid)
/// - The source TokenArray/buffer can still be used after this call
/// - Multiple exports from the same buffer are safe
/// - The consumer (PyTorch/JAX) will call the deleter which decrements refcount
///
/// Args:
///   buffer_handle: Handle to a SharedBuffer (refcount will be incremented)
///   offset_elems: Offset into the buffer (for views, usually 0)
///   len: Number of elements to export
///
/// Returns null on failure. Use talu_last_error for details.
pub export fn talu_buffer_to_dlpack(
    buffer_handle: ?*BufferHandle,
    offset_elems: usize,
    len: usize,
) callconv(.c) ?*tensor.DLManagedTensor {
    capi_error.clearError();

    const params = validateBufferParams(buffer_handle, offset_elems, len) orelse return null;

    // Retain the buffer (increment refcount) before allocating other resources
    params.buf.retain();

    return createTokenDLPack(params.buf, params.offset_elems, params.len);
}

// =============================================================================
// Batch DLPack (2D padded tensor)
// =============================================================================

/// Metadata for 2D batch DLPack export.
/// Stores shape/strides for a 2D padded tensor and owns the data.
const BatchDLPackContext = struct {
    /// The padded token data (owned, will be freed on delete)
    data: [*]u32,
    total_size: usize,
    /// Shape array for DLTensor (2D: [num_sequences, padded_length])
    shape: [2]i64,
    /// Strides array (row-major: [padded_length, 1])
    strides: [2]i64,
};

/// DLPack deleter for batch tensor.
fn batchDLPackDeleter(managed: *tensor.DLManagedTensor) callconv(.c) void {
    if (managed.manager_ctx) |ctx| {
        const batch_context: *BatchDLPackContext = @ptrCast(@alignCast(ctx));
        // Free the data
        allocator.free(batch_context.data[0..batch_context.total_size]);
        // Free the context struct
        allocator.destroy(batch_context);
    }
    // Free the DLManagedTensor itself
    allocator.destroy(managed);
}

/// Convert batch encoding to DLPack format (2D padded tensor).
///
/// Takes CSR-format batch data and returns a 2D tensor [num_sequences, padded_length].
/// Sequences are padded to max_length (or longest sequence if 0).
///
/// OWNERSHIP TRANSFER:
/// - A new padded buffer is allocated and ownership is transferred to DLManagedTensor.
/// - The original CSR data (ids, offsets) is NOT consumed - caller still owns it.
/// - The consumer (PyTorch/JAX) will call the deleter when done with the padded data.
///
/// Args:
///   ids: Flat token IDs array (CSR format)
///   offsets: Sequence boundary offsets (length = num_sequences + 1)
///   num_sequences: Number of sequences
///   pad_id: Token ID used for padding
///   max_length: Maximum length (0 = use longest sequence)
///   padding_side: 0 = right padding, 1 = left padding
///
/// Returns null on failure. Use talu_last_error for details.
pub export fn talu_batch_to_dlpack(
    ids: ?[*]const u32,
    offsets: ?[*]const usize,
    num_sequences: usize,
    pad_id: u32,
    max_length: usize,
    padding_side: u8,
) callconv(.c) ?*tensor.DLManagedTensor {
    capi_error.clearError();

    // Validate inputs
    if (num_sequences == 0) {
        capi_error.setError(error.InvalidArgument, "num_sequences is 0", .{});
        return null;
    }
    const ids_ptr = ids orelse {
        capi_error.setError(error.InvalidArgument, "ids pointer is null", .{});
        return null;
    };
    const offsets_ptr = offsets orelse {
        capi_error.setError(error.InvalidArgument, "offsets pointer is null", .{});
        return null;
    };

    // Delegate to padding helper
    var pad_result = padding.padSequences(
        allocator,
        ids_ptr,
        offsets_ptr,
        num_sequences,
        pad_id,
        max_length,
        padding_side == 1,
    ) catch |err| {
        capi_error.setError(err, "Failed to pad sequences: {s}", .{@errorName(err)});
        return null;
    };

    // Wrap in DLPack
    return wrapBatchAsDLPack(pad_result.data, pad_result.num_sequences, pad_result.padded_len) orelse {
        pad_result.deinit(allocator);
        return null;
    };
}

/// Wrap padded batch data as DLManagedTensor.
fn wrapBatchAsDLPack(data: []u32, num_sequences: usize, padded_len: usize) ?*tensor.DLManagedTensor {
    const batch_context = allocator.create(BatchDLPackContext) catch |err| {
        capi_error.setError(err, "Failed to allocate DLPack context: {s}", .{@errorName(err)});
        return null;
    };

    batch_context.* = .{
        .data = data.ptr,
        .total_size = data.len,
        .shape = .{ @intCast(num_sequences), @intCast(padded_len) },
        .strides = .{ @intCast(padded_len), 1 },
    };

    const managed = allocator.create(tensor.DLManagedTensor) catch |err| {
        allocator.destroy(batch_context);
        capi_error.setError(err, "Failed to allocate DLManagedTensor: {s}", .{@errorName(err)});
        return null;
    };

    managed.* = .{
        .dl_tensor = .{
            .data = @ptrCast(data.ptr),
            .device = .{ .device_type = .kDLCPU, .device_id = 0 },
            .ndim = 2,
            .dtype = .{ .code = .kDLUInt, .bits = 32, .lanes = 1 },
            .shape = &batch_context.shape,
            .strides = &batch_context.strides,
            .byte_offset = 0,
        },
        .manager_ctx = batch_context,
        .deleter = &batchDLPackDeleter,
    };

    return managed;
}

// =============================================================================
// Attention Mask DLPack (2D)
// =============================================================================

/// Metadata for 2D attention mask DLPack export.
const MaskDLPackContext = struct {
    data: [*]i32,
    total_size: usize,
    shape: [2]i64,
    strides: [2]i64,
};

/// DLPack deleter for mask tensor.
fn maskDLPackDeleter(managed: *tensor.DLManagedTensor) callconv(.c) void {
    if (managed.manager_ctx) |ctx| {
        const mask_context: *MaskDLPackContext = @ptrCast(@alignCast(ctx));
        allocator.free(mask_context.data[0..mask_context.total_size]);
        allocator.destroy(mask_context);
    }
    allocator.destroy(managed);
}

/// Convert batch encoding to DLPack attention mask format (2D padded tensor).
///
/// Takes CSR-format batch data and returns a 2D tensor [num_sequences, padded_length]
/// with values: 1 for real tokens, 0 for padding.
///
/// OWNERSHIP TRANSFER:
/// - The returned DLManagedTensor owns its memory
/// - Consumer must call managed.deleter(managed) when done
/// - The mask uses int32 dtype for compatibility
///
/// Args:
///   ids: CSR token data (not used for values, but needed for validation)
///   offsets: CSR offsets array (num_sequences + 1 elements)
///   num_sequences: Number of sequences in batch
///   max_length: Maximum length (0 = use longest sequence)
///   padding_side: 0=right, 1=left
///
/// Returns:
///   DLManagedTensor* on success, null on error (check talu_get_last_error)
pub export fn talu_batch_mask_to_dlpack(
    ids: ?[*]const u32,
    offsets: ?[*]const usize,
    num_sequences: usize,
    max_length: usize,
    padding_side: u8,
) callconv(.c) ?*tensor.DLManagedTensor {
    capi_error.clearError();

    // Validate inputs
    if (num_sequences == 0) {
        capi_error.setError(error.InvalidArgument, "num_sequences is 0", .{});
        return null;
    }
    _ = ids orelse {
        capi_error.setError(error.InvalidArgument, "ids pointer is null", .{});
        return null;
    };
    const offsets_ptr = offsets orelse {
        capi_error.setError(error.InvalidArgument, "offsets pointer is null", .{});
        return null;
    };

    // Delegate to padding helper
    var mask_result = padding.generateMask(
        allocator,
        offsets_ptr,
        num_sequences,
        max_length,
        padding_side == 1,
    ) catch |err| {
        capi_error.setError(err, "Failed to generate mask: {s}", .{@errorName(err)});
        return null;
    };

    // Wrap in DLPack
    return wrapMaskAsDLPack(mask_result.data, mask_result.num_sequences, mask_result.padded_len) orelse {
        mask_result.deinit(allocator);
        return null;
    };
}

/// Wrap mask data as DLManagedTensor.
fn wrapMaskAsDLPack(data: []i32, num_sequences: usize, padded_len: usize) ?*tensor.DLManagedTensor {
    const mask_context = allocator.create(MaskDLPackContext) catch |err| {
        capi_error.setError(err, "Failed to allocate DLPack context: {s}", .{@errorName(err)});
        return null;
    };

    mask_context.* = .{
        .data = data.ptr,
        .total_size = data.len,
        .shape = .{ @intCast(num_sequences), @intCast(padded_len) },
        .strides = .{ @intCast(padded_len), 1 },
    };

    const managed = allocator.create(tensor.DLManagedTensor) catch |err| {
        allocator.destroy(mask_context);
        capi_error.setError(err, "Failed to allocate DLManagedTensor: {s}", .{@errorName(err)});
        return null;
    };

    managed.* = .{
        .dl_tensor = .{
            .data = @ptrCast(data.ptr),
            .device = .{ .device_type = .kDLCPU, .device_id = 0 },
            .ndim = 2,
            .dtype = .{ .code = .kDLInt, .bits = 32, .lanes = 1 },
            .shape = &mask_context.shape,
            .strides = &mask_context.strides,
            .byte_offset = 0,
        },
        .manager_ctx = mask_context,
        .deleter = &maskDLPackDeleter,
    };

    return managed;
}
