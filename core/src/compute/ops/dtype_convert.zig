//! DType Conversion Utilities
//!
//! Bulk conversion functions for tensor data between dtypes.
//! Used by capi modules to avoid inline conversion loops.
//! Also includes high-level attention helpers that require tensor dtype handling.
//!
//! Performance: Uses SIMD vectorization for BF16/F16 → F32 conversion.
//! - BF16: 8-wide vectors with bit shift (very fast, pure integer ops)
//! - F16: 8-wide vectors using hardware vcvtph2ps on F16C-capable CPUs

const std = @import("std");
const dtype_mod = @import("../../dtype.zig");
const tensor_mod = @import("../../tensor.zig");
const attention = @import("attn_primitives.zig");
const simd = @import("simd/root.zig");

pub const DType = tensor_mod.DType;
pub const Tensor = tensor_mod.Tensor;

/// SIMD vector width for bulk conversions (8 elements = 256 bits for AVX)
const VEC_WIDTH: usize = 8;

/// Result of a tensor-to-f32 conversion.
/// Contains either a pointer to the original data (if already f32)
/// or an owned buffer that must be freed.
pub const F32ConversionResult = struct {
    data: [*]const f32,
    owned_buffer: ?[]f32,

    /// Free the owned buffer if one was allocated.
    pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
        if (self.owned_buffer) |buf| {
            allocator.free(buf);
            self.owned_buffer = null;
        }
    }
};

// =============================================================================
// SIMD Bulk Conversion Functions
// =============================================================================

/// Convert BF16 array to F32 using SIMD.
/// BF16 is just the upper 16 bits of F32, so conversion is a simple left shift.
/// This is extremely fast - pure integer operations, no floating point math.
fn convertBf16ToF32Simd(src: [*]const u16, dst: [*]f32, count: usize) void {
    const U16Vec = @Vector(VEC_WIDTH, u16);
    const U32Vec = @Vector(VEC_WIDTH, u32);
    const F32Vec = @Vector(VEC_WIDTH, f32);

    var i: usize = 0;

    // SIMD loop: process VEC_WIDTH elements at a time
    while (i + VEC_WIDTH <= count) : (i += VEC_WIDTH) {
        // Load 8 BF16 values
        const bf16_vec: U16Vec = src[i..][0..VEC_WIDTH].*;
        // Extend to 32-bit and shift left by 16 (BF16 -> F32 conversion)
        const u32_vec: U32Vec = @as(U32Vec, bf16_vec) << @splat(16);
        // Reinterpret as f32
        const f32_vec: F32Vec = @bitCast(u32_vec);
        // Store result
        dst[i..][0..VEC_WIDTH].* = f32_vec;
    }

    // Scalar tail for remaining elements
    while (i < count) : (i += 1) {
        dst[i] = dtype_mod.bf16ToF32(src[i]);
    }
}

/// Convert F16 array to F32 using SIMD.
/// Uses hardware vcvtph2ps instruction on x86 with F16C, portable fallback otherwise.
fn convertF16ToF32Simd(src: [*]const u16, dst: [*]f32, count: usize) void {
    const U16Vec = @Vector(VEC_WIDTH, u16);
    const F32Vec = @Vector(VEC_WIDTH, f32);

    var i: usize = 0;

    // SIMD loop: process VEC_WIDTH elements at a time
    while (i + VEC_WIDTH <= count) : (i += VEC_WIDTH) {
        // Load 8 F16 values
        const fp16_vec: U16Vec = src[i..][0..VEC_WIDTH].*;
        // Convert using hardware intrinsic (vcvtph2ps) or portable fallback
        const f32_vec: F32Vec = simd.cvtph2ps(fp16_vec);
        // Store result
        dst[i..][0..VEC_WIDTH].* = f32_vec;
    }

    // Scalar tail for remaining elements
    while (i < count) : (i += 1) {
        dst[i] = dtype_mod.fp16ToF32(src[i]);
    }
}

/// Convert tensor data to f32, allocating a buffer if needed.
/// For f32 tensors, returns pointer to original data (no allocation).
/// For bf16/f16 tensors, allocates and converts to f32.
pub fn tensorToF32(
    allocator: std.mem.Allocator,
    tensor: *const Tensor,
    element_count: usize,
) !F32ConversionResult {
    const dtype = tensor.simpleDType();

    switch (dtype) {
        .f32 => {
            return .{
                .data = @as([*]const f32, @ptrCast(@alignCast(tensor.data_ptr))),
                .owned_buffer = null,
            };
        },
        .bf16 => {
            const buffer = try allocator.alloc(f32, element_count);
            const src = @as([*]const u16, @ptrCast(@alignCast(tensor.data_ptr)));
            convertBf16ToF32Simd(src, buffer.ptr, element_count);
            return .{
                .data = buffer.ptr,
                .owned_buffer = buffer,
            };
        },
        .f16 => {
            const buffer = try allocator.alloc(f32, element_count);
            const src = @as([*]const u16, @ptrCast(@alignCast(tensor.data_ptr)));
            convertF16ToF32Simd(src, buffer.ptr, element_count);
            return .{
                .data = buffer.ptr,
                .owned_buffer = buffer,
            };
        },
        else => return error.UnsupportedDType,
    }
}

/// Check if a dtype is a supported float type (f32, f16, bf16).
pub fn isFloatDType(dtype: DType) bool {
    return dtype == .f32 or dtype == .f16 or dtype == .bf16;
}

/// Compute contiguous strides for a 4D tensor.
pub fn contiguousStrides4D(shape: [4]usize) [4]usize {
    return .{
        shape[1] * shape[2] * shape[3],
        shape[2] * shape[3],
        shape[3],
        1,
    };
}

/// Get strides from tensor, converting to usize.
pub fn tensorStrides4D(tensor: *const Tensor) [4]usize {
    return .{
        @intCast(tensor.strides[0]),
        @intCast(tensor.strides[1]),
        @intCast(tensor.strides[2]),
        @intCast(tensor.strides[3]),
    };
}

// =============================================================================
// Tests
// =============================================================================

test "tensorToF32 with f32 tensor returns original pointer" {
    const allocator = std.testing.allocator;

    // Create a simple f32 tensor
    const tensor = try Tensor.init(allocator, &[_]i64{ 2, 3 }, .f32, .{ .device_type = .kDLCPU, .device_id = 0 });
    defer tensor.deinit(allocator);

    var result = try tensorToF32(allocator, tensor, 6);
    defer result.deinit(allocator);

    // Should not allocate for f32
    try std.testing.expect(result.owned_buffer == null);
    try std.testing.expectEqual(
        @intFromPtr(@as([*]const f32, @ptrCast(@alignCast(tensor.data_ptr)))),
        @intFromPtr(result.data),
    );
}

test "isFloatDType returns correct values" {
    try std.testing.expect(isFloatDType(.f32));
    try std.testing.expect(isFloatDType(.f16));
    try std.testing.expect(isFloatDType(.bf16));
    try std.testing.expect(!isFloatDType(.u8));
    try std.testing.expect(!isFloatDType(.i32));
}

test "contiguousStrides4D computes correct strides" {
    const strides = contiguousStrides4D(.{ 2, 4, 8, 16 });
    try std.testing.expectEqual(@as(usize, 4 * 8 * 16), strides[0]);
    try std.testing.expectEqual(@as(usize, 8 * 16), strides[1]);
    try std.testing.expectEqual(@as(usize, 16), strides[2]);
    try std.testing.expectEqual(@as(usize, 1), strides[3]);
}
