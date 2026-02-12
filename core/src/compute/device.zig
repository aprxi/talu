//! Device type codes compatible with DLPack specification.

const std = @import("std");

/// Device type codes (compatible with DLPack)
pub const DeviceType = enum(i32) {
    CPU = 1,
    CUDA = 2,
    CUDAHost = 3,
    OpenCL = 4,
    Vulkan = 7,
    Metal = 8,
    VPI = 9,
    ROCM = 10,
    ROCMHost = 11,
    ExtDev = 12,
    CUDAManaged = 13,
    OneAPI = 14,
    WebGPU = 15,
    Hexagon = 16,
};

/// Device descriptor - where the tensor data lives.
pub const Device = extern struct {
    /// Backend device type (DLPack-compatible).
    device_type: DeviceType,
    /// Device index within the backend.
    device_id: i32,

    pub fn cpu() Device {
        var d = std.mem.zeroes(Device);
        d.device_type = .CPU;
        d.device_id = 0;
        return d;
    }

    pub fn cuda(device_id: i32) Device {
        var d = std.mem.zeroes(Device);
        d.device_type = .CUDA;
        d.device_id = device_id;
        return d;
    }

    pub fn metal(device_id: i32) Device {
        var d = std.mem.zeroes(Device);
        d.device_type = .Metal;
        d.device_id = device_id;
        return d;
    }

    pub fn isCPU(self: Device) bool {
        return self.device_type == .CPU;
    }

    pub fn isCUDA(self: Device) bool {
        return self.device_type == .CUDA;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "cpu creates CPU device" {
    const dev = Device.cpu();
    try @import("std").testing.expectEqual(DeviceType.CPU, dev.device_type);
    try @import("std").testing.expectEqual(@as(i32, 0), dev.device_id);
}

test "cuda creates CUDA device" {
    const dev = Device.cuda(1);
    try @import("std").testing.expectEqual(DeviceType.CUDA, dev.device_type);
    try @import("std").testing.expectEqual(@as(i32, 1), dev.device_id);
}

test "metal creates Metal device" {
    const dev = Device.metal(2);
    try @import("std").testing.expectEqual(DeviceType.Metal, dev.device_type);
    try @import("std").testing.expectEqual(@as(i32, 2), dev.device_id);
}

test "isCPU returns true for CPU device" {
    const cpu_dev = Device.cpu();
    const cuda_dev = Device.cuda(0);
    try @import("std").testing.expect(cpu_dev.isCPU());
    try @import("std").testing.expect(!cuda_dev.isCPU());
}

test "isCUDA returns true for CUDA device" {
    const cpu_dev = Device.cpu();
    const cuda_dev = Device.cuda(0);
    try @import("std").testing.expect(!cpu_dev.isCUDA());
    try @import("std").testing.expect(cuda_dev.isCUDA());
}
