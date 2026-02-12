//! DLPack Protocol Types
//!
//! This module defines the DLPack protocol types for tensor exchange.
//! The actual Tensor implementation is in tensor.zig.
//! DLPack is a standard protocol for zero-copy tensor sharing between frameworks.

const std = @import("std");
const device_mod = @import("device.zig");
const tensor_mod = @import("../tensor.zig");

pub const DeviceType = device_mod.DeviceType;
pub const Device = device_mod.Device;

// =============================================================================
// DLPack Protocol Types
// =============================================================================

/// DLPack device type codes (from dlpack.h)
/// Note: These match DLDeviceType in the official DLPack spec
pub const DLDeviceType = enum(i32) {
    kDLCPU = 1,
    kDLCUDA = 2,
    kDLCUDAHost = 3,
    kDLOpenCL = 4,
    kDLVulkan = 7,
    kDLMetal = 8,
    kDLVPI = 9,
    kDLROCM = 10,
    kDLROCMHost = 11,
    kDLExtDev = 12,
    kDLCUDAManaged = 13,
    kDLOneAPI = 14,
    kDLWebGPU = 15,
    kDLHexagon = 16,
};

/// DLPack data type codes
pub const DLDataTypeCode = enum(u8) {
    kDLInt = 0,
    kDLUInt = 1,
    kDLFloat = 2,
    kDLBfloat = 4,
    kDLComplex = 5,
    kDLBool = 6,
};

/// DLDevice - describes where the tensor lives
pub const DLDevice = extern struct {
    device_type: DLDeviceType,
    device_id: i32,

    pub fn cpu() DLDevice {
        var d = std.mem.zeroes(DLDevice);
        d.device_type = .kDLCPU;
        d.device_id = 0;
        return d;
    }

    pub fn cuda(device_id: i32) DLDevice {
        var d = std.mem.zeroes(DLDevice);
        d.device_type = .kDLCUDA;
        d.device_id = device_id;
        return d;
    }

    /// Convert from our Device type
    pub fn fromDevice(dev: Device) DLDevice {
        var d = std.mem.zeroes(DLDevice);
        d.device_type = @enumFromInt(@intFromEnum(dev.device_type));
        d.device_id = dev.device_id;
        return d;
    }
};

/// DLDataType - describes the data type
pub const DLDataType = extern struct {
    code: DLDataTypeCode,
    bits: u8,
    lanes: u16,

    fn init(code: DLDataTypeCode, bits: u8, lanes: u16) DLDataType {
        var dt = std.mem.zeroes(DLDataType);
        dt.code = code;
        dt.bits = bits;
        dt.lanes = lanes;
        return dt;
    }

    pub fn float32() DLDataType {
        return init(.kDLFloat, 32, 1);
    }

    pub fn float64() DLDataType {
        return init(.kDLFloat, 64, 1);
    }

    pub fn int32() DLDataType {
        return init(.kDLInt, 32, 1);
    }

    pub fn int64() DLDataType {
        return init(.kDLInt, 64, 1);
    }

    /// Convert from DType
    pub fn fromDType(dtype_tag: DType) DLDataType {
        return switch (dtype_tag) {
            .f32 => init(.kDLFloat, 32, 1),
            .f64 => init(.kDLFloat, 64, 1),
            .f16 => init(.kDLFloat, 16, 1),
            .bf16 => init(.kDLBfloat, 16, 1),
            .i8 => init(.kDLInt, 8, 1),
            .i16 => init(.kDLInt, 16, 1),
            .i32 => init(.kDLInt, 32, 1),
            .i64 => init(.kDLInt, 64, 1),
            .u8 => init(.kDLUInt, 8, 1),
            .u16 => init(.kDLUInt, 16, 1),
            .u32 => init(.kDLUInt, 32, 1),
            .u64 => init(.kDLUInt, 64, 1),
            // Quantized types appear as u8 arrays
            .grouped_affine_u4, .grouped_affine_u8, .mxfp4, .f8_e4m3 => init(.kDLUInt, 8, 1),
        };
    }
};

/// DLTensor - the core tensor descriptor
pub const DLTensor = extern struct {
    /// Pointer to the data (can be CPU or GPU)
    data: ?*anyopaque,
    /// Device where data resides
    device: DLDevice,
    /// Number of dimensions
    ndim: i32,
    /// Data type
    dtype: DLDataType,
    /// Shape array (length = ndim)
    shape: [*]i64,
    /// Strides array (length = ndim), can be null for contiguous
    strides: ?[*]i64,
    /// Byte offset into data pointer
    byte_offset: u64,
};

/// Deleter function type for DLManagedTensor
pub const DLManagedTensorDeleter = *const fn (*DLManagedTensor) callconv(.c) void;

/// DLManagedTensor - tensor with lifecycle management
pub const DLManagedTensor = extern struct {
    /// The tensor descriptor
    dl_tensor: DLTensor,
    /// Context for the manager (Tensor pointer)
    manager_ctx: ?*anyopaque,
    /// Destructor function
    deleter: ?DLManagedTensorDeleter,
};

// =============================================================================
// Re-export from tensor.zig
// =============================================================================

pub const MAX_NDIM: usize = tensor_mod.MAX_NDIM;
pub const Tensor = tensor_mod.Tensor;
pub const DType = tensor_mod.DType;

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

test "DLDevice.cpu returns CPU device with id 0" {
    const device = DLDevice.cpu();
    try testing.expectEqual(DLDeviceType.kDLCPU, device.device_type);
    try testing.expectEqual(@as(i32, 0), device.device_id);
}

test "DLDevice.cuda returns CUDA device with specified id" {
    const device0 = DLDevice.cuda(0);
    try testing.expectEqual(DLDeviceType.kDLCUDA, device0.device_type);
    try testing.expectEqual(@as(i32, 0), device0.device_id);

    const device1 = DLDevice.cuda(1);
    try testing.expectEqual(DLDeviceType.kDLCUDA, device1.device_type);
    try testing.expectEqual(@as(i32, 1), device1.device_id);

    const device7 = DLDevice.cuda(7);
    try testing.expectEqual(DLDeviceType.kDLCUDA, device7.device_type);
    try testing.expectEqual(@as(i32, 7), device7.device_id);
}

test "DLDevice.fromDevice converts CPU device" {
    const dev = Device.cpu();
    const dl_device = DLDevice.fromDevice(dev);
    try testing.expectEqual(DLDeviceType.kDLCPU, dl_device.device_type);
    try testing.expectEqual(@as(i32, 0), dl_device.device_id);
}

test "DLDevice.fromDevice converts CUDA device" {
    const dev = Device.cuda(2);
    const dl_device = DLDevice.fromDevice(dev);
    try testing.expectEqual(DLDeviceType.kDLCUDA, dl_device.device_type);
    try testing.expectEqual(@as(i32, 2), dl_device.device_id);
}

test "DLDevice.fromDevice converts Metal device" {
    const dev = Device.metal(0);
    const dl_device = DLDevice.fromDevice(dev);
    try testing.expectEqual(DLDeviceType.kDLMetal, dl_device.device_type);
    try testing.expectEqual(@as(i32, 0), dl_device.device_id);
}

test "DLDataType.float32 returns correct type descriptor" {
    const dtype = DLDataType.float32();
    try testing.expectEqual(DLDataTypeCode.kDLFloat, dtype.code);
    try testing.expectEqual(@as(u8, 32), dtype.bits);
    try testing.expectEqual(@as(u16, 1), dtype.lanes);
}

test "DLDataType.float64 returns correct type descriptor" {
    const dtype = DLDataType.float64();
    try testing.expectEqual(DLDataTypeCode.kDLFloat, dtype.code);
    try testing.expectEqual(@as(u8, 64), dtype.bits);
    try testing.expectEqual(@as(u16, 1), dtype.lanes);
}

test "DLDataType.int32 returns correct type descriptor" {
    const dtype = DLDataType.int32();
    try testing.expectEqual(DLDataTypeCode.kDLInt, dtype.code);
    try testing.expectEqual(@as(u8, 32), dtype.bits);
    try testing.expectEqual(@as(u16, 1), dtype.lanes);
}

test "DLDataType.int64 returns correct type descriptor" {
    const dtype = DLDataType.int64();
    try testing.expectEqual(DLDataTypeCode.kDLInt, dtype.code);
    try testing.expectEqual(@as(u8, 64), dtype.bits);
    try testing.expectEqual(@as(u16, 1), dtype.lanes);
}

test "DLDataType.fromDType converts float types" {
    const f32_type = DLDataType.fromDType(.f32);
    try testing.expectEqual(DLDataTypeCode.kDLFloat, f32_type.code);
    try testing.expectEqual(@as(u8, 32), f32_type.bits);
    try testing.expectEqual(@as(u16, 1), f32_type.lanes);

    const f64_type = DLDataType.fromDType(.f64);
    try testing.expectEqual(DLDataTypeCode.kDLFloat, f64_type.code);
    try testing.expectEqual(@as(u8, 64), f64_type.bits);
    try testing.expectEqual(@as(u16, 1), f64_type.lanes);

    const f16_type = DLDataType.fromDType(.f16);
    try testing.expectEqual(DLDataTypeCode.kDLFloat, f16_type.code);
    try testing.expectEqual(@as(u8, 16), f16_type.bits);
    try testing.expectEqual(@as(u16, 1), f16_type.lanes);
}

test "DLDataType.fromDType converts bfloat16 type" {
    const bf16_type = DLDataType.fromDType(.bf16);
    try testing.expectEqual(DLDataTypeCode.kDLBfloat, bf16_type.code);
    try testing.expectEqual(@as(u8, 16), bf16_type.bits);
    try testing.expectEqual(@as(u16, 1), bf16_type.lanes);
}

test "DLDataType.fromDType converts signed integer types" {
    const i8_type = DLDataType.fromDType(.i8);
    try testing.expectEqual(DLDataTypeCode.kDLInt, i8_type.code);
    try testing.expectEqual(@as(u8, 8), i8_type.bits);
    try testing.expectEqual(@as(u16, 1), i8_type.lanes);

    const i16_type = DLDataType.fromDType(.i16);
    try testing.expectEqual(DLDataTypeCode.kDLInt, i16_type.code);
    try testing.expectEqual(@as(u8, 16), i16_type.bits);
    try testing.expectEqual(@as(u16, 1), i16_type.lanes);

    const i32_type = DLDataType.fromDType(.i32);
    try testing.expectEqual(DLDataTypeCode.kDLInt, i32_type.code);
    try testing.expectEqual(@as(u8, 32), i32_type.bits);
    try testing.expectEqual(@as(u16, 1), i32_type.lanes);

    const i64_type = DLDataType.fromDType(.i64);
    try testing.expectEqual(DLDataTypeCode.kDLInt, i64_type.code);
    try testing.expectEqual(@as(u8, 64), i64_type.bits);
    try testing.expectEqual(@as(u16, 1), i64_type.lanes);
}

test "DLDataType.fromDType converts unsigned integer types" {
    const u8_type = DLDataType.fromDType(.u8);
    try testing.expectEqual(DLDataTypeCode.kDLUInt, u8_type.code);
    try testing.expectEqual(@as(u8, 8), u8_type.bits);
    try testing.expectEqual(@as(u16, 1), u8_type.lanes);

    const u16_type = DLDataType.fromDType(.u16);
    try testing.expectEqual(DLDataTypeCode.kDLUInt, u16_type.code);
    try testing.expectEqual(@as(u8, 16), u16_type.bits);
    try testing.expectEqual(@as(u16, 1), u16_type.lanes);

    const u32_type = DLDataType.fromDType(.u32);
    try testing.expectEqual(DLDataTypeCode.kDLUInt, u32_type.code);
    try testing.expectEqual(@as(u8, 32), u32_type.bits);
    try testing.expectEqual(@as(u16, 1), u32_type.lanes);

    const u64_type = DLDataType.fromDType(.u64);
    try testing.expectEqual(DLDataTypeCode.kDLUInt, u64_type.code);
    try testing.expectEqual(@as(u8, 64), u64_type.bits);
    try testing.expectEqual(@as(u16, 1), u64_type.lanes);
}

test "DLDataType.fromDType converts quantized types to u8" {
    const grouped_affine_u4_type = DLDataType.fromDType(.grouped_affine_u4);
    try testing.expectEqual(DLDataTypeCode.kDLUInt, grouped_affine_u4_type.code);
    try testing.expectEqual(@as(u8, 8), grouped_affine_u4_type.bits);
    try testing.expectEqual(@as(u16, 1), grouped_affine_u4_type.lanes);

    const grouped_affine_u8_type = DLDataType.fromDType(.grouped_affine_u8);
    try testing.expectEqual(DLDataTypeCode.kDLUInt, grouped_affine_u8_type.code);
    try testing.expectEqual(@as(u8, 8), grouped_affine_u8_type.bits);
    try testing.expectEqual(@as(u16, 1), grouped_affine_u8_type.lanes);
}
