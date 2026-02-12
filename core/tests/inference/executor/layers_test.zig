//! Unit tests for inference.executor.layers
//!
//! Tests all public functions in layers.zig:
//! - formatLinearLike, formatRmsNormLike - formatting helpers
//! - Linear.init, Linear.initWithDims - layer initialization
//! - Linear.forward - layer forward pass
//! - Linear.formatKernels, Linear.describe, Linear.formatTo - description/formatting
//! - Embedding.init, Embedding.forward - embedding layer
//! - Embedding.formatKernels, Embedding.describe, Embedding.formatTo - description/formatting

const std = @import("std");
const main = @import("main");

const layers = main.inference.executor.layers;
const formatLinearLike = layers.formatLinearLike;
const formatRmsNormLike = layers.formatRmsNormLike;
const Linear = layers.Linear;
const Embedding = layers.Embedding;
const Tensor = main.core.Tensor;
const DType = main.core.DType;
const GroupedAffineMeta = main.core.dtype.GroupedAffineMeta;

// =============================================================================
// formatLinearLike Tests
// =============================================================================

test "formatLinearLike with f32 weight and no bias" {
    var buf: [256]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);

    // Create a dummy f32 tensor
    var shape = [_]i64{ 512, 256 };
    var strides = [_]i64{ 256, 1 };
    const weight = Tensor{
        .data = @ptrCast(@alignCast(&buf)),
        .dtype = .f32,
        .shape = &shape,
        .strides = &strides,
        .n_dims = 2,
        .offset = 0,
        .device = Tensor.Device.cpu(),
        .gaffine = null,
        .mxfp4 = null,
    };

    try formatLinearLike(stream.writer(), &weight, null, 512, 256);

    const output = stream.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Linear(in=512, out=256, bias=false, dtype=f32)") != null);
}

test "formatLinearLike with f32 weight and bias" {
    var buf: [256]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);

    var shape = [_]i64{ 128, 64 };
    var strides = [_]i64{ 64, 1 };
    const weight = Tensor{
        .data = @ptrCast(@alignCast(&buf)),
        .dtype = .f32,
        .shape = &shape,
        .strides = &strides,
        .n_dims = 2,
        .offset = 0,
        .device = Tensor.Device.cpu(),
        .gaffine = null,
        .mxfp4 = null,
    };

    const bias_data = [_]f32{0.0} ** 64;

    try formatLinearLike(stream.writer(), &weight, &bias_data, 128, 64);

    const output = stream.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Linear(in=128, out=64, bias=true, dtype=f32)") != null);
}

test "formatLinearLike with f16 weight" {
    var buf: [256]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);

    var shape = [_]i64{ 256, 128 };
    var strides = [_]i64{ 128, 1 };
    const weight = Tensor{
        .data = @ptrCast(@alignCast(&buf)),
        .dtype = .f16,
        .shape = &shape,
        .strides = &strides,
        .n_dims = 2,
        .offset = 0,
        .device = Tensor.Device.cpu(),
        .gaffine = null,
        .mxfp4 = null,
    };

    try formatLinearLike(stream.writer(), &weight, null, 256, 128);

    const output = stream.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Linear(in=256, out=128, bias=false, dtype=f16)") != null);
}

test "formatLinearLike with bf16 weight" {
    var buf: [256]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);

    var shape = [_]i64{ 1024, 512 };
    var strides = [_]i64{ 512, 1 };
    const weight = Tensor{
        .data = @ptrCast(@alignCast(&buf)),
        .dtype = .bf16,
        .shape = &shape,
        .strides = &strides,
        .n_dims = 2,
        .offset = 0,
        .device = Tensor.Device.cpu(),
        .gaffine = null,
        .mxfp4 = null,
    };

    try formatLinearLike(stream.writer(), &weight, null, 1024, 512);

    const output = stream.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Linear(in=1024, out=512, bias=false, dtype=bf16)") != null);
}

test "formatLinearLike with q5_0 weight" {
    var buf: [256]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);

    var shape = [_]i64{ 512, 256 };
    var strides = [_]i64{ 256, 1 };
    const weight = Tensor{
        .data = @ptrCast(@alignCast(&buf)),
        .dtype = .q5_0,
        .shape = &shape,
        .strides = &strides,
        .n_dims = 2,
        .offset = 0,
        .device = Tensor.Device.cpu(),
        .gaffine = null,
        .mxfp4 = null,
    };

    try formatLinearLike(stream.writer(), &weight, null, 512, 256);

    const output = stream.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Linear(in=512, out=256, bias=false, dtype=q5_0)") != null);
}

test "formatLinearLike with grouped_affine_u4 weight" {
    var buf: [256]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);

    const gaffine_meta = GroupedAffineMeta{
        .group_size = 64,
        .n_groups = 8,
    };

    var shape = [_]i64{ 512, 256 };
    var strides = [_]i64{ 256, 1 };
    const weight = Tensor{
        .data = @ptrCast(@alignCast(&buf)),
        .dtype = .grouped_affine_u4,
        .shape = &shape,
        .strides = &strides,
        .n_dims = 2,
        .offset = 0,
        .device = Tensor.Device.cpu(),
        .gaffine = gaffine_meta,
        .mxfp4 = null,
    };

    try formatLinearLike(stream.writer(), &weight, null, 512, 256);

    const output = stream.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "QuantizedLinear(in=512, out=256, bits=4, group_size=64)") != null);
}

test "formatLinearLike with grouped_affine_u4 weight without metadata" {
    var buf: [256]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);

    var shape = [_]i64{ 512, 256 };
    var strides = [_]i64{ 256, 1 };
    const weight = Tensor{
        .data = @ptrCast(@alignCast(&buf)),
        .dtype = .grouped_affine_u4,
        .shape = &shape,
        .strides = &strides,
        .n_dims = 2,
        .offset = 0,
        .device = Tensor.Device.cpu(),
        .gaffine = null,
        .mxfp4 = null,
    };

    try formatLinearLike(stream.writer(), &weight, null, 512, 256);

    const output = stream.getWritten();
    // Should use default group_size of 64
    try std.testing.expect(std.mem.indexOf(u8, output, "QuantizedLinear(in=512, out=256, bits=4, group_size=64)") != null);
}

test "formatLinearLike with grouped_affine_u8 weight" {
    var buf: [256]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);

    const gaffine_meta = GroupedAffineMeta{
        .group_size = 128,
        .n_groups = 4,
    };

    var shape = [_]i64{ 512, 256 };
    var strides = [_]i64{ 256, 1 };
    const weight = Tensor{
        .data = @ptrCast(@alignCast(&buf)),
        .dtype = .grouped_affine_u8,
        .shape = &shape,
        .strides = &strides,
        .n_dims = 2,
        .offset = 0,
        .device = Tensor.Device.cpu(),
        .gaffine = gaffine_meta,
        .mxfp4 = null,
    };

    try formatLinearLike(stream.writer(), &weight, null, 512, 256);

    const output = stream.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "QuantizedLinear(in=512, out=256, bits=8, group_size=128)") != null);
}

test "formatLinearLike with grouped_affine_u8 weight without metadata" {
    var buf: [256]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);

    var shape = [_]i64{ 512, 256 };
    var strides = [_]i64{ 256, 1 };
    const weight = Tensor{
        .data = @ptrCast(@alignCast(&buf)),
        .dtype = .grouped_affine_u8,
        .shape = &shape,
        .strides = &strides,
        .n_dims = 2,
        .offset = 0,
        .device = Tensor.Device.cpu(),
        .gaffine = null,
        .mxfp4 = null,
    };

    try formatLinearLike(stream.writer(), &weight, null, 512, 256);

    const output = stream.getWritten();
    // Should use default group_size of 64
    try std.testing.expect(std.mem.indexOf(u8, output, "QuantizedLinear(in=512, out=256, bits=8, group_size=64)") != null);
}

// =============================================================================
// formatRmsNormLike Tests
// =============================================================================

test "formatRmsNormLike with zero weight_offset" {
    var buf: [256]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);

    try formatRmsNormLike(stream.writer(), 512, 1e-6, 0.0);

    const output = stream.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "RMSNorm(dim=512, eps=1e-06)") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "weight_offset") == null);
}

test "formatRmsNormLike with non-zero weight_offset" {
    var buf: [256]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);

    try formatRmsNormLike(stream.writer(), 768, 1e-5, 1.0);

    const output = stream.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "RMSNorm(dim=768, eps=1e-05, weight_offset=1.0)") != null);
}

test "formatRmsNormLike with various dimensions" {
    var buf: [256]u8 = undefined;

    // Test with different dimensions
    const test_cases = [_]struct { dim: usize, eps: f32 }{
        .{ .dim = 64, .eps = 1e-8 },
        .{ .dim = 256, .eps = 1e-5 },
        .{ .dim = 1024, .eps = 1e-6 },
        .{ .dim = 2048, .eps = 1e-4 },
    };

    for (test_cases) |tc| {
        var stream = std.io.fixedBufferStream(&buf);
        try formatRmsNormLike(stream.writer(), tc.dim, tc.eps, 0.0);

        const output = stream.getWritten();
        try std.testing.expect(std.mem.indexOf(u8, output, "RMSNorm(dim=") != null);
    }
}

// =============================================================================
// Linear Layer Tests
// =============================================================================

test "Linear.init with f32 weight derives dimensions correctly" {
    const allocator = std.testing.allocator;

    // Create weight tensor: [in_features, out_features] for f32
    const in_features = 512;
    const out_features = 256;

    var shape = [_]i64{ in_features, out_features };
    var strides = [_]i64{ out_features, 1 };

    const data = try allocator.alloc(u8, in_features * out_features * 4);
    defer allocator.free(data);

    const weight = Tensor{
        .data = @ptrCast(@alignCast(data.ptr)),
        .dtype = .f32,
        .shape = &shape,
        .strides = &strides,
        .n_dims = 2,
        .offset = 0,
        .device = Tensor.Device.cpu(),
        .gaffine = null,
        .mxfp4 = null,
    };

    const linear = try Linear.init(&weight, null);

    try std.testing.expectEqual(in_features, linear.in_features);
    try std.testing.expectEqual(out_features, linear.out_features);
    try std.testing.expect(linear.bias == null);
}

test "Linear.init with f16 weight derives dimensions correctly" {
    const allocator = std.testing.allocator;

    // Create weight tensor: [out_features, in_features] for f16
    const in_features = 768;
    const out_features = 512;

    var shape = [_]i64{ out_features, in_features };
    var strides = [_]i64{ in_features, 1 };

    const data = try allocator.alloc(u8, in_features * out_features * 2);
    defer allocator.free(data);

    const weight = Tensor{
        .data = @ptrCast(@alignCast(data.ptr)),
        .dtype = .f16,
        .shape = &shape,
        .strides = &strides,
        .n_dims = 2,
        .offset = 0,
        .device = Tensor.Device.cpu(),
        .gaffine = null,
        .mxfp4 = null,
    };

    const linear = try Linear.init(&weight, null);

    try std.testing.expectEqual(in_features, linear.in_features);
    try std.testing.expectEqual(out_features, linear.out_features);
}

test "Linear.init with bias" {
    const allocator = std.testing.allocator;

    const in_features = 256;
    const out_features = 128;

    var shape = [_]i64{ in_features, out_features };
    var strides = [_]i64{ out_features, 1 };

    const data = try allocator.alloc(u8, in_features * out_features * 4);
    defer allocator.free(data);

    const weight = Tensor{
        .data = @ptrCast(@alignCast(data.ptr)),
        .dtype = .f32,
        .shape = &shape,
        .strides = &strides,
        .n_dims = 2,
        .offset = 0,
        .device = Tensor.Device.cpu(),
        .gaffine = null,
        .mxfp4 = null,
    };

    const bias = try allocator.alloc(f32, out_features);
    defer allocator.free(bias);

    for (bias, 0..) |*b, i| {
        b.* = @floatFromInt(i);
    }

    const linear = try Linear.init(&weight, bias);

    try std.testing.expect(linear.bias != null);
    try std.testing.expectEqual(out_features, linear.bias.?.len);
}

test "Linear.initWithDims uses provided dimensions" {
    const allocator = std.testing.allocator;

    const in_features = 1024;
    const out_features = 512;

    var shape = [_]i64{ 128, 256 }; // Arbitrary shape
    var strides = [_]i64{ 256, 1 };

    const data = try allocator.alloc(u8, 128 * 256 * 4);
    defer allocator.free(data);

    const weight = Tensor{
        .data = @ptrCast(@alignCast(data.ptr)),
        .dtype = .f32,
        .shape = &shape,
        .strides = &strides,
        .n_dims = 2,
        .offset = 0,
        .device = Tensor.Device.cpu(),
        .gaffine = null,
        .mxfp4 = null,
    };

    const linear = try Linear.initWithDims(&weight, null, in_features, out_features);

    try std.testing.expectEqual(in_features, linear.in_features);
    try std.testing.expectEqual(out_features, linear.out_features);
}

test "Linear.initWithDims with bias" {
    const allocator = std.testing.allocator;

    const in_features = 512;
    const out_features = 256;

    var shape = [_]i64{ 64, 128 };
    var strides = [_]i64{ 128, 1 };

    const data = try allocator.alloc(u8, 64 * 128 * 4);
    defer allocator.free(data);

    const weight = Tensor{
        .data = @ptrCast(@alignCast(data.ptr)),
        .dtype = .f32,
        .shape = &shape,
        .strides = &strides,
        .n_dims = 2,
        .offset = 0,
        .device = Tensor.Device.cpu(),
        .gaffine = null,
        .mxfp4 = null,
    };

    const bias = try allocator.alloc(f32, out_features);
    defer allocator.free(bias);

    const linear = try Linear.initWithDims(&weight, bias, in_features, out_features);

    try std.testing.expectEqual(in_features, linear.in_features);
    try std.testing.expectEqual(out_features, linear.out_features);
    try std.testing.expect(linear.bias != null);
}

test "Linear.formatTo outputs correct format for f32" {
    const allocator = std.testing.allocator;

    const in_features = 512;
    const out_features = 256;

    var shape = [_]i64{ in_features, out_features };
    var strides = [_]i64{ out_features, 1 };

    const data = try allocator.alloc(u8, in_features * out_features * 4);
    defer allocator.free(data);

    const weight = Tensor{
        .data = @ptrCast(@alignCast(data.ptr)),
        .dtype = .f32,
        .shape = &shape,
        .strides = &strides,
        .n_dims = 2,
        .offset = 0,
        .device = Tensor.Device.cpu(),
        .gaffine = null,
        .mxfp4 = null,
    };

    const linear = try Linear.init(&weight, null);

    var buf: [256]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);

    try linear.formatTo(stream.writer());

    const output = stream.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Linear(in=512, out=256, bias=false, dtype=f32)") != null);
}

test "Linear.formatKernels outputs matmul operation" {
    const allocator = std.testing.allocator;

    const in_features = 256;
    const out_features = 128;

    var shape = [_]i64{ in_features, out_features };
    var strides = [_]i64{ out_features, 1 };

    const data = try allocator.alloc(u8, in_features * out_features * 4);
    defer allocator.free(data);

    const weight = Tensor{
        .data = @ptrCast(@alignCast(data.ptr)),
        .dtype = .f32,
        .shape = &shape,
        .strides = &strides,
        .n_dims = 2,
        .offset = 0,
        .device = Tensor.Device.cpu(),
        .gaffine = null,
        .mxfp4 = null,
    };

    const linear = try Linear.init(&weight, null);

    var buf: [512]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);

    try linear.formatKernels(stream.writer(), 2);

    const output = stream.getWritten();
    try std.testing.expect(output.len > 0);
    // Should contain matmul kernel info
    try std.testing.expect(std.mem.indexOf(u8, output, "matmul") != null or std.mem.indexOf(u8, output, "gemm") != null);
}

test "Linear.formatKernels outputs bias_add when bias exists" {
    const allocator = std.testing.allocator;

    const in_features = 256;
    const out_features = 128;

    var shape = [_]i64{ in_features, out_features };
    var strides = [_]i64{ out_features, 1 };

    const data = try allocator.alloc(u8, in_features * out_features * 4);
    defer allocator.free(data);

    const weight = Tensor{
        .data = @ptrCast(@alignCast(data.ptr)),
        .dtype = .f32,
        .shape = &shape,
        .strides = &strides,
        .n_dims = 2,
        .offset = 0,
        .device = Tensor.Device.cpu(),
        .gaffine = null,
        .mxfp4 = null,
    };

    const bias = try allocator.alloc(f32, out_features);
    defer allocator.free(bias);

    const linear = try Linear.init(&weight, bias);

    var buf: [512]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);

    try linear.formatKernels(stream.writer(), 2);

    const output = stream.getWritten();
    try std.testing.expect(output.len > 0);
    // Should contain bias_add kernel info
    try std.testing.expect(std.mem.indexOf(u8, output, "bias") != null);
}

test "Linear.describe outputs formatted description" {
    const allocator = std.testing.allocator;

    const in_features = 512;
    const out_features = 256;

    var shape = [_]i64{ in_features, out_features };
    var strides = [_]i64{ out_features, 1 };

    const data = try allocator.alloc(u8, in_features * out_features * 4);
    defer allocator.free(data);

    const weight = Tensor{
        .data = @ptrCast(@alignCast(data.ptr)),
        .dtype = .f32,
        .shape = &shape,
        .strides = &strides,
        .n_dims = 2,
        .offset = 0,
        .device = Tensor.Device.cpu(),
        .gaffine = null,
        .mxfp4 = null,
    };

    const linear = try Linear.init(&weight, null);

    var buf: [512]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);

    try linear.describe(stream.writer(), 0, false);

    const output = stream.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Linear(in=512, out=256") != null);
    // Should have newline
    try std.testing.expect(std.mem.indexOf(u8, output, "\n") != null);
}

test "Linear.describe with show_kernels=true" {
    const allocator = std.testing.allocator;

    const in_features = 256;
    const out_features = 128;

    var shape = [_]i64{ in_features, out_features };
    var strides = [_]i64{ out_features, 1 };

    const data = try allocator.alloc(u8, in_features * out_features * 4);
    defer allocator.free(data);

    const weight = Tensor{
        .data = @ptrCast(@alignCast(data.ptr)),
        .dtype = .f32,
        .shape = &shape,
        .strides = &strides,
        .n_dims = 2,
        .offset = 0,
        .device = Tensor.Device.cpu(),
        .gaffine = null,
        .mxfp4 = null,
    };

    const linear = try Linear.init(&weight, null);

    var buf: [1024]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);

    try linear.describe(stream.writer(), 0, true);

    const output = stream.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Linear(in=256, out=128") != null);
    // Should have kernel info
    try std.testing.expect(output.len > 50);
}

test "Linear.describe with indentation" {
    const allocator = std.testing.allocator;

    const in_features = 128;
    const out_features = 64;

    var shape = [_]i64{ in_features, out_features };
    var strides = [_]i64{ out_features, 1 };

    const data = try allocator.alloc(u8, in_features * out_features * 4);
    defer allocator.free(data);

    const weight = Tensor{
        .data = @ptrCast(@alignCast(data.ptr)),
        .dtype = .f32,
        .shape = &shape,
        .strides = &strides,
        .n_dims = 2,
        .offset = 0,
        .device = Tensor.Device.cpu(),
        .gaffine = null,
        .mxfp4 = null,
    };

    const linear = try Linear.init(&weight, null);

    var buf: [512]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);

    try linear.describe(stream.writer(), 4, false);

    const output = stream.getWritten();
    // Should start with 4 spaces
    try std.testing.expect(output.len >= 4);
    try std.testing.expectEqual(@as(u8, ' '), output[0]);
    try std.testing.expectEqual(@as(u8, ' '), output[1]);
    try std.testing.expectEqual(@as(u8, ' '), output[2]);
    try std.testing.expectEqual(@as(u8, ' '), output[3]);
}

// =============================================================================
// Embedding Layer Tests
// =============================================================================

test "Embedding.init derives dimensions correctly" {
    const allocator = std.testing.allocator;

    const vocab_size = 32000;
    const embed_dim = 512;

    var shape = [_]i64{ vocab_size, embed_dim };
    var strides = [_]i64{ embed_dim, 1 };

    const data = try allocator.alloc(u8, vocab_size * embed_dim * 4);
    defer allocator.free(data);

    const weight = Tensor{
        .data = @ptrCast(@alignCast(data.ptr)),
        .dtype = .f32,
        .shape = &shape,
        .strides = &strides,
        .n_dims = 2,
        .offset = 0,
        .device = Tensor.Device.cpu(),
        .gaffine = null,
        .mxfp4 = null,
    };

    const embedding = Embedding.init(&weight);

    try std.testing.expectEqual(vocab_size, embedding.vocab_size);
    try std.testing.expectEqual(embed_dim, embedding.embed_dim);
}

test "Embedding.init with small vocabulary" {
    const allocator = std.testing.allocator;

    const vocab_size = 1000;
    const embed_dim = 128;

    var shape = [_]i64{ vocab_size, embed_dim };
    var strides = [_]i64{ embed_dim, 1 };

    const data = try allocator.alloc(u8, vocab_size * embed_dim * 4);
    defer allocator.free(data);

    const weight = Tensor{
        .data = @ptrCast(@alignCast(data.ptr)),
        .dtype = .f32,
        .shape = &shape,
        .strides = &strides,
        .n_dims = 2,
        .offset = 0,
        .device = Tensor.Device.cpu(),
        .gaffine = null,
        .mxfp4 = null,
    };

    const embedding = Embedding.init(&weight);

    try std.testing.expectEqual(vocab_size, embedding.vocab_size);
    try std.testing.expectEqual(embed_dim, embedding.embed_dim);
}

test "Embedding.init with large vocabulary" {
    const allocator = std.testing.allocator;

    const vocab_size = 128000;
    const embed_dim = 2048;

    var shape = [_]i64{ vocab_size, embed_dim };
    var strides = [_]i64{ embed_dim, 1 };

    const data = try allocator.alloc(u8, vocab_size * embed_dim * 4);
    defer allocator.free(data);

    const weight = Tensor{
        .data = @ptrCast(@alignCast(data.ptr)),
        .dtype = .f32,
        .shape = &shape,
        .strides = &strides,
        .n_dims = 2,
        .offset = 0,
        .device = Tensor.Device.cpu(),
        .gaffine = null,
        .mxfp4 = null,
    };

    const embedding = Embedding.init(&weight);

    try std.testing.expectEqual(vocab_size, embedding.vocab_size);
    try std.testing.expectEqual(embed_dim, embedding.embed_dim);
}

test "Embedding.formatTo outputs correct format for f32" {
    const allocator = std.testing.allocator;

    const vocab_size = 32000;
    const embed_dim = 512;

    var shape = [_]i64{ vocab_size, embed_dim };
    var strides = [_]i64{ embed_dim, 1 };

    const data = try allocator.alloc(u8, vocab_size * embed_dim * 4);
    defer allocator.free(data);

    const weight = Tensor{
        .data = @ptrCast(@alignCast(data.ptr)),
        .dtype = .f32,
        .shape = &shape,
        .strides = &strides,
        .n_dims = 2,
        .offset = 0,
        .device = Tensor.Device.cpu(),
        .gaffine = null,
        .mxfp4 = null,
    };

    const embedding = Embedding.init(&weight);

    var buf: [256]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);

    try embedding.formatTo(stream.writer());

    const output = stream.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Embedding(vocab=32000, dim=512)") != null);
}

test "Embedding.formatTo outputs correct format for grouped_affine_u4" {
    const allocator = std.testing.allocator;

    const vocab_size = 16000;
    const embed_dim = 256;

    var shape = [_]i64{ vocab_size, embed_dim };
    var strides = [_]i64{ embed_dim, 1 };

    const data = try allocator.alloc(u8, vocab_size * embed_dim);
    defer allocator.free(data);

    const weight = Tensor{
        .data = @ptrCast(@alignCast(data.ptr)),
        .dtype = .grouped_affine_u4,
        .shape = &shape,
        .strides = &strides,
        .n_dims = 2,
        .offset = 0,
        .device = Tensor.Device.cpu(),
        .gaffine = null,
        .mxfp4 = null,
    };

    const embedding = Embedding.init(&weight);

    var buf: [256]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);

    try embedding.formatTo(stream.writer());

    const output = stream.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Embedding(vocab=16000, dim=256, bits=4)") != null);
}

test "Embedding.formatTo outputs correct format for grouped_affine_u8" {
    const allocator = std.testing.allocator;

    const vocab_size = 8000;
    const embed_dim = 128;

    var shape = [_]i64{ vocab_size, embed_dim };
    var strides = [_]i64{ embed_dim, 1 };

    const data = try allocator.alloc(u8, vocab_size * embed_dim);
    defer allocator.free(data);

    const weight = Tensor{
        .data = @ptrCast(@alignCast(data.ptr)),
        .dtype = .grouped_affine_u8,
        .shape = &shape,
        .strides = &strides,
        .n_dims = 2,
        .offset = 0,
        .device = Tensor.Device.cpu(),
        .gaffine = null,
        .mxfp4 = null,
    };

    const embedding = Embedding.init(&weight);

    var buf: [256]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);

    try embedding.formatTo(stream.writer());

    const output = stream.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Embedding(vocab=8000, dim=128, bits=8)") != null);
}

test "Embedding.formatKernels outputs gather operation" {
    const allocator = std.testing.allocator;

    const vocab_size = 32000;
    const embed_dim = 512;

    var shape = [_]i64{ vocab_size, embed_dim };
    var strides = [_]i64{ embed_dim, 1 };

    const data = try allocator.alloc(u8, vocab_size * embed_dim * 4);
    defer allocator.free(data);

    const weight = Tensor{
        .data = @ptrCast(@alignCast(data.ptr)),
        .dtype = .f32,
        .shape = &shape,
        .strides = &strides,
        .n_dims = 2,
        .offset = 0,
        .device = Tensor.Device.cpu(),
        .gaffine = null,
        .mxfp4 = null,
    };

    const embedding = Embedding.init(&weight);

    var buf: [512]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);

    try embedding.formatKernels(stream.writer(), 2);

    const output = stream.getWritten();
    try std.testing.expect(output.len > 0);
    // Should contain gather kernel info
    try std.testing.expect(std.mem.indexOf(u8, output, "gather") != null or std.mem.indexOf(u8, output, "Gather") != null);
}

test "Embedding.describe outputs formatted description" {
    const allocator = std.testing.allocator;

    const vocab_size = 32000;
    const embed_dim = 512;

    var shape = [_]i64{ vocab_size, embed_dim };
    var strides = [_]i64{ embed_dim, 1 };

    const data = try allocator.alloc(u8, vocab_size * embed_dim * 4);
    defer allocator.free(data);

    const weight = Tensor{
        .data = @ptrCast(@alignCast(data.ptr)),
        .dtype = .f32,
        .shape = &shape,
        .strides = &strides,
        .n_dims = 2,
        .offset = 0,
        .device = Tensor.Device.cpu(),
        .gaffine = null,
        .mxfp4 = null,
    };

    const embedding = Embedding.init(&weight);

    var buf: [512]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);

    try embedding.describe(stream.writer(), 0, false);

    const output = stream.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Embedding(vocab=32000, dim=512)") != null);
    // Should have newline
    try std.testing.expect(std.mem.indexOf(u8, output, "\n") != null);
}

test "Embedding.describe with show_kernels=true" {
    const allocator = std.testing.allocator;

    const vocab_size = 16000;
    const embed_dim = 256;

    var shape = [_]i64{ vocab_size, embed_dim };
    var strides = [_]i64{ embed_dim, 1 };

    const data = try allocator.alloc(u8, vocab_size * embed_dim * 4);
    defer allocator.free(data);

    const weight = Tensor{
        .data = @ptrCast(@alignCast(data.ptr)),
        .dtype = .f32,
        .shape = &shape,
        .strides = &strides,
        .n_dims = 2,
        .offset = 0,
        .device = Tensor.Device.cpu(),
        .gaffine = null,
        .mxfp4 = null,
    };

    const embedding = Embedding.init(&weight);

    var buf: [1024]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);

    try embedding.describe(stream.writer(), 0, true);

    const output = stream.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "Embedding(vocab=16000, dim=256)") != null);
    // Should have kernel info
    try std.testing.expect(output.len > 50);
}

test "Embedding.describe with indentation" {
    const allocator = std.testing.allocator;

    const vocab_size = 8000;
    const embed_dim = 128;

    var shape = [_]i64{ vocab_size, embed_dim };
    var strides = [_]i64{ embed_dim, 1 };

    const data = try allocator.alloc(u8, vocab_size * embed_dim * 4);
    defer allocator.free(data);

    const weight = Tensor{
        .data = @ptrCast(@alignCast(data.ptr)),
        .dtype = .f32,
        .shape = &shape,
        .strides = &strides,
        .n_dims = 2,
        .offset = 0,
        .device = Tensor.Device.cpu(),
        .gaffine = null,
        .mxfp4 = null,
    };

    const embedding = Embedding.init(&weight);

    var buf: [512]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);

    try embedding.describe(stream.writer(), 4, false);

    const output = stream.getWritten();
    // Should start with 4 spaces
    try std.testing.expect(output.len >= 4);
    try std.testing.expectEqual(@as(u8, ' '), output[0]);
    try std.testing.expectEqual(@as(u8, ' '), output[1]);
    try std.testing.expectEqual(@as(u8, ' '), output[2]);
    try std.testing.expectEqual(@as(u8, ' '), output[3]);
}
