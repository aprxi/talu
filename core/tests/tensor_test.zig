//! Integration tests for Tensor
//!
//! Tensor is the unified tensor type for the entire codebase.
//! - DLPack-compatible for Python/C interop
//! - Stride-aware with contiguity assertions
//! - Supports up to 8 dimensions
//! - Supports quantized dtypes

const std = @import("std");
const main = @import("main");
const Tensor = main.Tensor;
const DType = main.DType;

// =============================================================================
// Creation Tests
// =============================================================================

test "Tensor.init creates tensor with correct shape" {
    const allocator = std.testing.allocator;

    const shape = [_]i64{ 2, 3, 4 };
    const tensor = try Tensor.init(allocator, &shape, .f32, main.tensor.Device.cpu());
    defer tensor.deinit(allocator);

    try std.testing.expectEqual(@as(i32, 3), tensor.n_dims);
    try std.testing.expectEqual(@as(i64, 2), tensor.shape[0]);
    try std.testing.expectEqual(@as(i64, 3), tensor.shape[1]);
    try std.testing.expectEqual(@as(i64, 4), tensor.shape[2]);
    try std.testing.expectEqual(@as(usize, 24), tensor.numel);
}

test "Tensor.init allocates correct data size" {
    const allocator = std.testing.allocator;

    const shape = [_]i64{ 10, 20 };
    const tensor = try Tensor.init(allocator, &shape, .f32, main.tensor.Device.cpu());
    defer tensor.deinit(allocator);

    // 10 * 20 * 4 bytes = 800 bytes
    try std.testing.expectEqual(@as(usize, 800), tensor.data_size);
    try std.testing.expect(tensor.data_ptr != null);
}

test "Tensor.init sets owns_data to true" {
    const allocator = std.testing.allocator;

    const shape = [_]i64{100};
    const tensor = try Tensor.init(allocator, &shape, .f32, main.tensor.Device.cpu());
    defer tensor.deinit(allocator);

    try std.testing.expect(tensor.owns_data);
}

// =============================================================================
// View Tests
// =============================================================================

test "Tensor.view creates non-owning view" {
    var data: [100]f32 = undefined;
    const bytes = std.mem.sliceAsBytes(&data);

    const tensor = Tensor.view(@constCast(bytes.ptr), &[_]usize{ 10, 10 }, .f32, null);

    try std.testing.expect(!tensor.owns_data);
    try std.testing.expectEqual(@as(i32, 2), tensor.n_dims);
    try std.testing.expectEqual(@as(usize, 100), tensor.numel);
}

test "Tensor.view2D creates 2D f32 view" {
    var data: [24]f32 = undefined;
    const bytes = std.mem.sliceAsBytes(&data);

    const tensor = Tensor.view2D(@constCast(bytes), 4, 6);

    try std.testing.expectEqual(@as(i32, 2), tensor.n_dims);
    try std.testing.expectEqual(@as(i64, 4), tensor.shape[0]);
    try std.testing.expectEqual(@as(i64, 6), tensor.shape[1]);
    try std.testing.expectEqual(DType.f32, tensor.dtype);
}

test "Tensor.view3D creates 3D f32 view with batch dim 1" {
    var data: [24]f32 = undefined;
    const bytes = std.mem.sliceAsBytes(&data);

    const tensor = Tensor.view3D(@constCast(bytes), 4, 6);

    try std.testing.expectEqual(@as(i32, 3), tensor.n_dims);
    try std.testing.expectEqual(@as(i64, 1), tensor.shape[0]);
    try std.testing.expectEqual(@as(i64, 4), tensor.shape[1]);
    try std.testing.expectEqual(@as(i64, 6), tensor.shape[2]);
}

// =============================================================================
// Data Access Tests
// =============================================================================

test "Tensor.data returns byte slice" {
    const allocator = std.testing.allocator;

    const shape = [_]i64{ 2, 2 };
    const tensor = try Tensor.init(allocator, &shape, .f32, main.tensor.Device.cpu());
    defer tensor.deinit(allocator);

    const data = tensor.data();
    try std.testing.expectEqual(@as(usize, 16), data.len); // 4 elements * 4 bytes
}

test "Tensor.asSlice returns typed slice" {
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const bytes = std.mem.sliceAsBytes(&data);

    const tensor = Tensor.view2D(@constCast(bytes), 2, 2);
    const slice = tensor.asSlice(f32);

    try std.testing.expectEqual(@as(usize, 4), slice.len);
    try std.testing.expectEqual(@as(f32, 1.0), slice[0]);
    try std.testing.expectEqual(@as(f32, 4.0), slice[3]);
}

test "Tensor.shapeAsUsize returns first 4 dims" {
    const allocator = std.testing.allocator;

    const shape = [_]i64{ 2, 3, 4, 5 };
    const tensor = try Tensor.init(allocator, &shape, .f32, main.tensor.Device.cpu());
    defer tensor.deinit(allocator);

    const shape_usize = tensor.shapeAsUsize();
    try std.testing.expectEqual(@as(usize, 2), shape_usize[0]);
    try std.testing.expectEqual(@as(usize, 3), shape_usize[1]);
    try std.testing.expectEqual(@as(usize, 4), shape_usize[2]);
    try std.testing.expectEqual(@as(usize, 5), shape_usize[3]);
}

// =============================================================================
// Contiguity Tests
// =============================================================================

test "Tensor.isContiguous returns true for row-major layout" {
    const allocator = std.testing.allocator;

    const shape = [_]i64{ 2, 3, 4 };
    const tensor = try Tensor.init(allocator, &shape, .f32, main.tensor.Device.cpu());
    defer tensor.deinit(allocator);

    try std.testing.expect(tensor.isContiguous());
}

test "Tensor strides are row-major (C-contiguous)" {
    const allocator = std.testing.allocator;

    const shape = [_]i64{ 2, 3, 4 };
    const tensor = try Tensor.init(allocator, &shape, .f32, main.tensor.Device.cpu());
    defer tensor.deinit(allocator);

    // Row-major strides: [3*4, 4, 1] = [12, 4, 1]
    try std.testing.expectEqual(@as(i64, 12), tensor.strides[0]);
    try std.testing.expectEqual(@as(i64, 4), tensor.strides[1]);
    try std.testing.expectEqual(@as(i64, 1), tensor.strides[2]);
}

// =============================================================================
// DType Tests
// =============================================================================

test "Tensor.simpleDType returns standard type for non-quantized" {
    const allocator = std.testing.allocator;

    const shape = [_]i64{10};
    const tensor = try Tensor.init(allocator, &shape, .f32, main.tensor.Device.cpu());
    defer tensor.deinit(allocator);

    try std.testing.expectEqual(DType.f32, tensor.simpleDType());
}

test "Tensor supports different dtypes" {
    const allocator = std.testing.allocator;
    const shape = [_]i64{10};

    // Test f32
    {
        const t = try Tensor.init(allocator, &shape, .f32, main.tensor.Device.cpu());
        defer t.deinit(allocator);
        try std.testing.expectEqual(DType.f32, t.dtype);
        try std.testing.expectEqual(@as(usize, 40), t.data_size); // 10 * 4
    }

    // Test f16
    {
        const t = try Tensor.init(allocator, &shape, .f16, main.tensor.Device.cpu());
        defer t.deinit(allocator);
        try std.testing.expectEqual(DType.f16, t.dtype);
        try std.testing.expectEqual(@as(usize, 20), t.data_size); // 10 * 2
    }

    // Test i32
    {
        const t = try Tensor.init(allocator, &shape, .i32, main.tensor.Device.cpu());
        defer t.deinit(allocator);
        try std.testing.expectEqual(DType.i32, t.dtype);
        try std.testing.expectEqual(@as(usize, 40), t.data_size); // 10 * 4
    }
}

// =============================================================================
// Device Tests
// =============================================================================

test "Tensor.isCPU returns true for CPU tensor" {
    const allocator = std.testing.allocator;

    const shape = [_]i64{10};
    const tensor = try Tensor.init(allocator, &shape, .f32, main.tensor.Device.cpu());
    defer tensor.deinit(allocator);

    try std.testing.expect(tensor.isCPU());
}

test "Tensor device can be set to Metal" {
    var data: [10]f32 = undefined;
    const bytes = std.mem.sliceAsBytes(&data);

    var tensor = Tensor.view(@constCast(bytes.ptr), &[_]usize{10}, .f32, null);
    tensor.device = main.tensor.Device.metal(0);

    try std.testing.expect(!tensor.isCPU());
    try std.testing.expectEqual(main.tensor.DLDeviceType.kDLMetal, tensor.device.device_type);
}

// =============================================================================
// DLPack Export Tests
// =============================================================================

test "Tensor.toDLPack creates DLManagedTensor" {
    const allocator = std.testing.allocator;

    const shape = [_]i64{ 2, 3 };
    var tensor = try Tensor.init(allocator, &shape, .f32, main.tensor.Device.cpu());
    // Note: toDLPack takes ownership, tensor will be freed by deleter

    const managed = try tensor.toDLPack(allocator);

    try std.testing.expectEqual(@as(i32, 2), managed.dl_tensor.ndim);
    try std.testing.expect(managed.dl_tensor.data != null);
    try std.testing.expect(managed.deleter != null);

    // Call deleter to clean up (this frees tensor too)
    managed.deleter.?(managed);
}
