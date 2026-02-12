//! C API tensor operations.
//!
//! Provides FFI-safe tensor creation and manipulation functions.
const std = @import("std");
const tensor_mod = @import("../tensor.zig");
const capi_error = @import("error.zig");
const error_codes = @import("error_codes.zig");

// Re-export types from tensor module
pub const Tensor = tensor_mod.Tensor;
pub const DType = tensor_mod.DType;
pub const Device = tensor_mod.Device;
pub const DLManagedTensor = tensor_mod.DLManagedTensor;
pub const MAX_NDIM = tensor_mod.MAX_NDIM;

// Native build uses C allocator
const allocator = std.heap.c_allocator;

// =============================================================================
// Basic API
// =============================================================================

pub export fn talu_hello() callconv(.c) [*:0]const u8 {
    return "Hello from Zig!";
}

// =============================================================================
// Tensor Creation API
// =============================================================================

/// Create a new tensor with the given shape and dtype
/// On success, writes the tensor handle to out_tensor
pub export fn talu_tensor_create(
    shape_ptr: [*]const i64,
    ndim: usize,
    dtype: u32,
    device_type: i32,
    device_id: i32,
    out_tensor: ?*?*Tensor,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_tensor orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_tensor is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;
    if (ndim > MAX_NDIM) {
        capi_error.setError(error.InvalidArgument, "ndim exceeds MAX_NDIM", .{});
        return @intFromEnum(error_codes.errorToCode(error.InvalidArgument));
    }

    const shape_dims = shape_ptr[0..ndim];
    const dtype_value: DType = @enumFromInt(dtype);
    const device_spec = Device{
        .device_type = @enumFromInt(device_type),
        .device_id = device_id,
    };

    const tensor_handle = Tensor.init(allocator, shape_dims, dtype_value, device_spec) catch |err| {
        capi_error.setError(err, "Tensor allocation failed: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    out.* = tensor_handle;
    return 0;
}

/// Create a tensor filled with zeros
pub export fn talu_tensor_zeros(
    shape_ptr: [*]const i64,
    ndim: usize,
    dtype: u32,
    out_tensor: ?*?*Tensor,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_tensor orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_tensor is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;
    const status = talu_tensor_create(shape_ptr, ndim, dtype, 1, 0, out_tensor);
    if (status != 0) return status;
    const tensor_handle = out.* orelse return @intFromEnum(error_codes.errorToCode(error.InternalError));

    // Zero the memory
    if (tensor_handle.data_ptr) |data_ptr| {
        const byte_size = tensor_handle.numel * tensor_handle.dtype.elementSize();
        @memset(data_ptr[0..byte_size], 0);
    }

    return 0;
}

/// Fill test embedding data with deterministic pattern.
fn fillTestEmbeddings(data: []f32, rows: usize, cols: usize) void {
    for (0..rows) |row| {
        const row_base = @as(f32, @floatFromInt(row)) * 0.1;
        for (0..cols) |col| {
            data[row * cols + col] = row_base + @as(f32, @floatFromInt(col)) * 0.001;
        }
    }
}

/// Create a test tensor with sample data (10x1536 float32 embeddings).
///
/// Caller owns the returned tensor. Call talu_tensor_free() when done.
pub export fn talu_tensor_test_embeddings(out_tensor: ?*?*Tensor) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_tensor orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_tensor is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;

    const shape = [_]i64{ 10, 1536 };
    const tensor_handle = Tensor.init(allocator, &shape, tensor_mod.DType.f32, Device.cpu()) catch |err| {
        capi_error.setError(err, "Tensor allocation failed: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    fillTestEmbeddings(tensor_handle.asSlice(f32), 10, 1536);
    out.* = tensor_handle;
    return 0;
}

/// Free a tensor
pub export fn talu_tensor_free(tensor: ?*Tensor) callconv(.c) void {
    if (tensor) |t| {
        t.deinit(allocator);
    }
}

// =============================================================================
// Tensor Accessor API (for Python __array_interface__)
// =============================================================================

/// Get pointer to tensor data
pub export fn talu_tensor_data_ptr(
    t: ?*const Tensor,
    out_ptr: ?*?*anyopaque,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_ptr orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_ptr is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;
    const tensor_handle = t orelse {
        capi_error.setError(error.InvalidHandle, "Invalid tensor handle", .{});
        return @intFromEnum(error_codes.errorToCode(error.InvalidHandle));
    };
    out.* = @ptrCast(tensor_handle.data_ptr);
    return 0;
}

/// Get number of dimensions
pub export fn talu_tensor_ndim(t: ?*const Tensor) callconv(.c) usize {
    return if (t) |p| @as(usize, @intCast(p.n_dims)) else 0;
}

/// Get pointer to shape array
pub export fn talu_tensor_shape(
    t: ?*const Tensor,
    out_shape: *?[*]const i64,
) callconv(.c) i32 {
    capi_error.clearError();
    out_shape.* = null;
    const tensor_ptr = t orelse {
        capi_error.setError(error.InvalidHandle, "Invalid tensor handle", .{});
        return @intFromEnum(error_codes.errorToCode(error.InvalidHandle));
    };
    out_shape.* = &tensor_ptr.shape;
    return 0;
}

/// Get pointer to strides array (in elements, not bytes)
pub export fn talu_tensor_strides(
    t: ?*const Tensor,
    out_strides: *?[*]const i64,
) callconv(.c) i32 {
    capi_error.clearError();
    out_strides.* = null;
    const tensor_ptr = t orelse {
        capi_error.setError(error.InvalidHandle, "Invalid tensor handle", .{});
        return @intFromEnum(error_codes.errorToCode(error.InvalidHandle));
    };
    out_strides.* = &tensor_ptr.strides;
    return 0;
}

/// Get dtype enum value (as DType for external API)
pub export fn talu_tensor_dtype(t: ?*const Tensor) callconv(.c) u32 {
    return if (t) |p| @intFromEnum(p.simpleDType()) else 0;
}

/// Get dtype as numpy typestring (e.g., "<f4")
pub export fn talu_tensor_typestr(t: ?*const Tensor) callconv(.c) [*:0]const u8 {
    return if (t) |p| p.simpleDType().toTypeStr() else "<f4";
}

/// Get device type (1=CPU, 2=CUDA, etc.)
pub export fn talu_tensor_device_type(t: ?*const Tensor) callconv(.c) i32 {
    return if (t) |p| @intFromEnum(p.device.device_type) else 1;
}

/// Get device id
pub export fn talu_tensor_device_id(t: ?*const Tensor) callconv(.c) i32 {
    return if (t) |p| p.device.device_id else 0;
}

/// Check if tensor is on CPU
pub export fn talu_tensor_is_cpu(t: ?*const Tensor) callconv(.c) bool {
    return if (t) |p| p.isCPU() else true;
}

/// Get total number of elements
pub export fn talu_tensor_numel(t: ?*const Tensor) callconv(.c) usize {
    return if (t) |p| p.numel else 0;
}

/// Get element size in bytes
pub export fn talu_tensor_element_size(t: ?*const Tensor) callconv(.c) usize {
    return if (t) |p| p.dtype.elementSize() else 4;
}

// =============================================================================
// DLPack API (for PyTorch/JAX __dlpack__)
// =============================================================================

/// Convert tensor to DLPack format
/// Returns a DLManagedTensor* that can be wrapped in a PyCapsule
/// The capsule's destructor should call the deleter function
pub export fn talu_tensor_to_dlpack(
    tensor: ?*Tensor,
    out_dlpack: ?*?*DLManagedTensor,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_dlpack orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_dlpack is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;
    const tensor_handle = tensor orelse {
        capi_error.setError(error.InvalidHandle, "Invalid tensor handle", .{});
        return @intFromEnum(error_codes.errorToCode(error.InvalidHandle));
    };
    const managed = tensor_handle.toDLPack(allocator) catch |err| {
        capi_error.setError(err, "DLPack export failed: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    out.* = managed;
    return 0;
}

/// Get the name for DLPack capsules (required by protocol)
pub export fn talu_dlpack_capsule_name() callconv(.c) [*:0]const u8 {
    return "dltensor";
}

/// Get the name for already-consumed DLPack capsules.
pub export fn talu_dlpack_used_capsule_name() callconv(.c) [*:0]const u8 {
    return "used_dltensor";
}

test "talu_tensor_data_ptr: non-zero return sets error" {
    var out_ptr: ?*anyopaque = null;
    const rc = talu_tensor_data_ptr(null, &out_ptr);
    try std.testing.expect(rc != 0);
    try std.testing.expect(capi_error.talu_last_error_code() != 0);
    capi_error.talu_clear_error();
}
