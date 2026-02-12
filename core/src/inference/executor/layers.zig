//! Layer primitives for transformer models.
//!
//! Defines Linear, Embedding, and RMSNorm layer structures with
//! forward pass execution and formatting for inspection.

const std = @import("std");
const common = @import("common.zig");

const Tensor = common.Tensor;

// Import execution dependencies
const compute = @import("../../compute/root.zig");
const matmul = compute.ops.matmul;
const inspect = @import("../../xray/root.zig");
const kernel_info = inspect.kernel_info;
const embedding_kernel = @import("../backend/cpu/kernels/embedding.zig");

const MatmulFn = matmul.MatmulFn;
const KernelOp = kernel_info.KernelOp;

pub fn formatLinearLike(
    writer: anytype,
    weight: *const Tensor,
    bias: ?[]const f32,
    in_features: usize,
    out_features: usize,
) !void {
    const weight_dtype = weight.dtype;
    if (weight_dtype == .grouped_affine_u4) {
        const group_size = if (weight.gaffine) |m| m.group_size else 64;
        try writer.print("QuantizedLinear(in={}, out={}, bits=4, group_size={})", .{
            in_features,
            out_features,
            group_size,
        });
    } else if (weight_dtype == .grouped_affine_u8) {
        const group_size = if (weight.gaffine) |m| m.group_size else 64;
        try writer.print("QuantizedLinear(in={}, out={}, bits=8, group_size={})", .{
            in_features,
            out_features,
            group_size,
        });
    } else {
        const dtype_name: []const u8 = switch (weight_dtype) {
            .f32 => "f32",
            .f16 => "f16",
            .bf16 => "bf16",
            else => "unknown",
        };
        try writer.print("Linear(in={}, out={}, bias={}, dtype={s})", .{
            in_features,
            out_features,
            bias != null,
            dtype_name,
        });
    }
}

pub fn formatRmsNormLike(writer: anytype, dim: usize, eps: f32, weight_offset: f32) !void {
    if (weight_offset != 0.0) {
        try writer.print("RMSNorm(dim={}, eps={e}, weight_offset={d:.1})", .{ dim, eps, weight_offset });
    } else {
        try writer.print("RMSNorm(dim={}, eps={e})", .{ dim, eps });
    }
}

// =============================================================================
// Linear Layer
// =============================================================================

/// Linear transformation: y = x @ W + b
/// Owns a pointer to weight tensor (mmap'd) and optional bias.
pub const Linear = struct {
    weight: *const Tensor,
    bias: ?[]const f32 = null,
    in_features: usize,
    out_features: usize,
    matmul_fn: MatmulFn,
    pub fn init(weight: *const Tensor, bias: ?[]const f32) !Linear {
        const in_features: usize = switch (weight.dtype) {
            .f32 => @intCast(weight.shape[0]),
            else => @intCast(weight.shape[1]),
        };
        const out_features: usize = switch (weight.dtype) {
            .f32 => @intCast(weight.shape[1]),
            else => @intCast(weight.shape[0]),
        };
        const dk = try matmul.matmulKernel(weight.dtype);
        return .{
            .weight = weight,
            .bias = bias,
            .in_features = in_features,
            .out_features = out_features,
            .matmul_fn = dk.func,
        };
    }

    pub fn initWithDims(weight: *const Tensor, bias: ?[]const f32, in_features: usize, out_features: usize) !Linear {
        const dk = try matmul.matmulKernel(weight.dtype);
        return .{
            .weight = weight,
            .bias = bias,
            .in_features = in_features,
            .out_features = out_features,
            .matmul_fn = dk.func,
        };
    }

    /// Forward: y = x @ W + b
    pub inline fn forward(self: *const Linear, input_tensor: *const Tensor, output_tensor: *Tensor, scratch: *matmul.MatmulScratch) void {

        const row_count: usize = if (input_tensor.n_dims == 3) @intCast(input_tensor.shape[0] * input_tensor.shape[1]) else @intCast(input_tensor.shape[0]);
        const input_view = Tensor.view2D(input_tensor.data(), row_count, self.in_features);
        var output_view = Tensor.view2DSlice(output_tensor.asSlice(f32), row_count, self.out_features);

        self.matmul_fn(&input_view, self.weight, &output_view, scratch);

        if (self.bias) |bias| {
            const output_data = output_tensor.asSlice(f32);
            for (0..row_count) |row_idx| {
                const output_row = output_data[row_idx * self.out_features ..][0..self.out_features];
                for (0..self.out_features) |col_idx| {
                    output_row[col_idx] += bias[col_idx];
                }
            }
        }
    }

    /// Format kernel operations directly to writer (avoids dangling slice)
    pub fn formatKernels(self: *const Linear, writer: anytype, indent: usize) !void {
        const matmul_op = KernelOp{ .matmul = .{
            .m = .seq,
            .k = self.in_features,
            .n = self.out_features,
            .dtype = self.weight.dtype,
            .kernel_name = kernel_info.matmulKernelName(self.weight.dtype),
        } };
        try matmul_op.format(writer, indent);

        if (self.bias != null) {
            const bias_op = KernelOp{ .bias_add = .{ .size = self.out_features } };
            try bias_op.format(writer, indent);
        }
    }

    /// Format for introspection
    pub fn describe(self: *const Linear, writer: anytype, indent: usize, show_kernels: bool) !void {
        try writer.writeByteNTimes(' ', indent);
        try self.formatTo(writer);
        try writer.writeAll("\n");

        if (show_kernels) {
            try self.formatKernels(writer, indent + 2);
        }
    }

    pub fn formatTo(self: *const Linear, writer: anytype) !void {
        try formatLinearLike(writer, self.weight, self.bias, self.in_features, self.out_features);
    }
};

// =============================================================================
// Embedding Layer
// =============================================================================

/// Token embedding lookup table
pub const Embedding = struct {
    weight: *const Tensor,
    vocab_size: usize,
    embed_dim: usize,
    pub fn init(weight: *const Tensor) Embedding {
        return .{
            .weight = weight,
            .vocab_size = @intCast(weight.shape[0]),
            .embed_dim = @intCast(weight.shape[1]),
        };
    }

    /// Forward: gather embeddings for token IDs
    pub fn forward(self: *const Embedding, tokens: []const u32, output_tensor: *Tensor) !void {

        try embedding_kernel.gatherEmbeddings(self.weight, tokens, output_tensor);
    }

    pub fn formatKernels(self: *const Embedding, writer: anytype, indent: usize) !void {
        const gather_op = KernelOp{ .gather = .{
            .vocab_size = self.vocab_size,
            .embed_dim = self.embed_dim,
            .dtype = self.weight.dtype,
        } };
        try gather_op.format(writer, indent);
    }

    pub fn describe(self: *const Embedding, writer: anytype, indent: usize, show_kernels: bool) !void {
        try writer.writeByteNTimes(' ', indent);
        try self.formatTo(writer);
        try writer.writeAll("\n");

        if (show_kernels) {
            try self.formatKernels(writer, indent + 2);
        }
    }

    pub fn formatTo(self: *const Embedding, writer: anytype) !void {
        const dtype = self.weight.dtype;
        if (dtype == .grouped_affine_u4 or dtype == .grouped_affine_u8) {
            const bits: u8 = if (dtype == .grouped_affine_u4) 4 else 8;
            try writer.print("Embedding(vocab={}, dim={}, bits={})", .{
                self.vocab_size, self.embed_dim, bits,
            });
        } else {
            try writer.print("Embedding(vocab={}, dim={})", .{
                self.vocab_size, self.embed_dim,
            });
        }
    }
};

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;
const OwnedTensor = @import("../../tensor.zig").OwnedTensor;
const DType = @import("../../dtype.zig").DType;

test "formatLinearLike formats f32 Linear without bias" {
    var buffer: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buffer);

    const shape = [_]usize{ 128, 256 };
    var weight_owned = try OwnedTensor.init(testing.allocator, .f32, &shape);
    defer weight_owned.deinit();
    const weight = weight_owned.toTensor();

    try formatLinearLike(fbs.writer(), &weight, null, 128, 256);

    const output = fbs.getWritten();
    try testing.expect(std.mem.indexOf(u8, output, "Linear(in=128, out=256") != null);
    try testing.expect(std.mem.indexOf(u8, output, "bias=false") != null);
    try testing.expect(std.mem.indexOf(u8, output, "dtype=f32") != null);
}

test "formatLinearLike formats f32 Linear with bias" {
    var buffer: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buffer);

    const shape = [_]usize{ 128, 256 };
    var weight_owned = try OwnedTensor.init(testing.allocator, .f32, &shape);
    defer weight_owned.deinit();
    const weight = weight_owned.toTensor();

    const bias = try testing.allocator.alloc(f32, 256);
    defer testing.allocator.free(bias);

    try formatLinearLike(fbs.writer(), &weight, bias, 128, 256);

    const output = fbs.getWritten();
    try testing.expect(std.mem.indexOf(u8, output, "Linear(in=128, out=256") != null);
    try testing.expect(std.mem.indexOf(u8, output, "bias=true") != null);
    try testing.expect(std.mem.indexOf(u8, output, "dtype=f32") != null);
}

test "formatLinearLike formats f16 Linear" {
    var buffer: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buffer);

    const shape = [_]usize{ 64, 128 };
    var weight_owned = try OwnedTensor.init(testing.allocator, .f16, &shape);
    defer weight_owned.deinit();
    const weight = weight_owned.toTensor();

    try formatLinearLike(fbs.writer(), &weight, null, 64, 128);

    const output = fbs.getWritten();
    try testing.expect(std.mem.indexOf(u8, output, "Linear(in=64, out=128") != null);
    try testing.expect(std.mem.indexOf(u8, output, "dtype=f16") != null);
}

test "formatLinearLike formats bf16 Linear" {
    var buffer: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buffer);

    const shape = [_]usize{ 32, 64 };
    var weight_owned = try OwnedTensor.init(testing.allocator, .bf16, &shape);
    defer weight_owned.deinit();
    const weight = weight_owned.toTensor();

    try formatLinearLike(fbs.writer(), &weight, null, 32, 64);

    const output = fbs.getWritten();
    try testing.expect(std.mem.indexOf(u8, output, "Linear(in=32, out=64") != null);
    try testing.expect(std.mem.indexOf(u8, output, "dtype=bf16") != null);
}

test "formatLinearLike formats grouped_affine_u4 QuantizedLinear" {
    var buffer: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buffer);

    const shape = [_]usize{ 512, 256 };
    var weight_owned = try OwnedTensor.init(testing.allocator, .grouped_affine_u4, &shape);
    defer weight_owned.deinit();
    var weight = weight_owned.toTensor();

    // Set group_size in gaffine metadata
    weight.gaffine = .{ .group_size = 128, .scales = &[_]u8{}, .biases = &[_]u8{} };

    try formatLinearLike(fbs.writer(), &weight, null, 512, 256);

    const output = fbs.getWritten();
    try testing.expect(std.mem.indexOf(u8, output, "QuantizedLinear(in=512, out=256") != null);
    try testing.expect(std.mem.indexOf(u8, output, "bits=4") != null);
    try testing.expect(std.mem.indexOf(u8, output, "group_size=128") != null);
}

test "formatLinearLike formats grouped_affine_u8 QuantizedLinear" {
    var buffer: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buffer);

    const shape = [_]usize{ 256, 512 };
    var weight_owned = try OwnedTensor.init(testing.allocator, .grouped_affine_u8, &shape);
    defer weight_owned.deinit();
    var weight = weight_owned.toTensor();

    // Set group_size in gaffine metadata
    weight.gaffine = .{ .group_size = 64, .scales = &[_]u8{}, .biases = &[_]u8{} };

    try formatLinearLike(fbs.writer(), &weight, null, 256, 512);

    const output = fbs.getWritten();
    try testing.expect(std.mem.indexOf(u8, output, "QuantizedLinear(in=256, out=512") != null);
    try testing.expect(std.mem.indexOf(u8, output, "bits=8") != null);
    try testing.expect(std.mem.indexOf(u8, output, "group_size=64") != null);
}

test "formatRmsNormLike formats RMSNorm without weight_offset" {
    var buffer: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buffer);

    try formatRmsNormLike(fbs.writer(), 768, 1e-5, 0.0);

    const output = fbs.getWritten();
    try testing.expect(std.mem.indexOf(u8, output, "RMSNorm(dim=768") != null);
    try testing.expect(std.mem.indexOf(u8, output, "eps=1e-5") != null);
    try testing.expect(std.mem.indexOf(u8, output, "weight_offset") == null);
}

test "formatRmsNormLike formats RMSNorm with weight_offset" {
    var buffer: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buffer);

    try formatRmsNormLike(fbs.writer(), 1024, 1e-6, 1.0);

    const output = fbs.getWritten();
    try testing.expect(std.mem.indexOf(u8, output, "RMSNorm(dim=1024") != null);
    try testing.expect(std.mem.indexOf(u8, output, "eps=1e-6") != null);
    try testing.expect(std.mem.indexOf(u8, output, "weight_offset=1.0") != null);
}

test "Linear.init extracts dimensions from f32 weight tensor" {
    const shape = [_]usize{ 128, 256 };
    var weight_owned = try OwnedTensor.init(testing.allocator, .f32, &shape);
    defer weight_owned.deinit();
    const weight = weight_owned.toTensor();

    const linear = try Linear.init(&weight, null);

    try testing.expectEqual(@as(usize, 128), linear.in_features);
    try testing.expectEqual(@as(usize, 256), linear.out_features);
    try testing.expectEqual(@as(?[]const f32, null), linear.bias);
}

test "Linear.init extracts dimensions from f16 weight tensor" {
    const shape = [_]usize{ 512, 128 };
    var weight_owned = try OwnedTensor.init(testing.allocator, .f16, &shape);
    defer weight_owned.deinit();
    const weight = weight_owned.toTensor();

    const linear = try Linear.init(&weight, null);

    // For non-f32 dtypes, dimensions are swapped
    try testing.expectEqual(@as(usize, 128), linear.in_features);
    try testing.expectEqual(@as(usize, 512), linear.out_features);
}

test "Linear.init stores bias when provided" {
    const shape = [_]usize{ 64, 128 };
    var weight_owned = try OwnedTensor.init(testing.allocator, .f32, &shape);
    defer weight_owned.deinit();
    const weight = weight_owned.toTensor();

    const bias = try testing.allocator.alloc(f32, 128);
    defer testing.allocator.free(bias);
    bias[0] = 1.5;

    const linear = try Linear.init(&weight, bias);

    try testing.expect(linear.bias != null);
    try testing.expectEqual(@as(f32, 1.5), linear.bias.?[0]);
}

test "Linear.initWithDims creates linear layer with explicit dimensions" {
    const shape = [_]usize{ 100, 200 };
    var weight_owned = try OwnedTensor.init(testing.allocator, .f32, &shape);
    defer weight_owned.deinit();
    const weight = weight_owned.toTensor();

    const linear = try Linear.initWithDims(&weight, null, 256, 512);

    try testing.expectEqual(@as(usize, 256), linear.in_features);
    try testing.expectEqual(@as(usize, 512), linear.out_features);
}

test "Linear.forward performs matrix multiplication without bias" {
    const in_features = 4;
    const out_features = 3;
    const batch_size = 2;

    // Create weight matrix (in_features x out_features)
    const weight_shape = [_]usize{ in_features, out_features };
    var weight_owned = try OwnedTensor.init(testing.allocator, .f32, &weight_shape);
    defer weight_owned.deinit();
    const weight = weight_owned.toTensor();
    const weight_data = weight_owned.asSlice(f32);

    // Initialize weight as identity-like for testing
    for (0..in_features) |i| {
        for (0..out_features) |j| {
            weight_data[i * out_features + j] = if (i == j) 1.0 else 0.0;
        }
    }

    const linear = try Linear.init(&weight, null);

    // Create input (batch_size x in_features)
    const input_shape = [_]usize{ batch_size, in_features };
    var input_owned = try OwnedTensor.init(testing.allocator, .f32, &input_shape);
    defer input_owned.deinit();
    const input = input_owned.toTensor();
    const input_data = input_owned.asSlice(f32);

    // Initialize input
    for (0..batch_size) |i| {
        for (0..in_features) |j| {
            input_data[i * in_features + j] = @as(f32, @floatFromInt(i * in_features + j));
        }
    }

    // Create output (batch_size x out_features)
    const output_shape = [_]usize{ batch_size, out_features };
    var output_owned = try OwnedTensor.init(testing.allocator, .f32, &output_shape);
    defer output_owned.deinit();
    var output = output_owned.toTensor();

    // Create scratch buffer
    var scratch = try matmul.MatmulScratch.init(testing.allocator);
    defer scratch.deinit();

    linear.forward(&input, &output, &scratch);

    // Verify output dimensions
    try testing.expectEqual(@as(i32, 2), output.n_dims);
    try testing.expectEqual(@as(i64, batch_size), output.shape[0]);
    try testing.expectEqual(@as(i64, out_features), output.shape[1]);
}

test "Linear.forward applies bias when provided" {
    const in_features = 3;
    const out_features = 2;
    const batch_size = 1;

    // Create weight matrix
    const weight_shape = [_]usize{ in_features, out_features };
    var weight_owned = try OwnedTensor.init(testing.allocator, .f32, &weight_shape);
    defer weight_owned.deinit();
    const weight = weight_owned.toTensor();
    const weight_data = weight_owned.asSlice(f32);

    // Zero weight for simple bias test
    for (weight_data) |*w| {
        w.* = 0.0;
    }

    // Create bias
    const bias = try testing.allocator.alloc(f32, out_features);
    defer testing.allocator.free(bias);
    bias[0] = 10.0;
    bias[1] = 20.0;

    const linear = try Linear.init(&weight, bias);

    // Create input
    const input_shape = [_]usize{ batch_size, in_features };
    var input_owned = try OwnedTensor.init(testing.allocator, .f32, &input_shape);
    defer input_owned.deinit();
    const input = input_owned.toTensor();

    // Create output
    const output_shape = [_]usize{ batch_size, out_features };
    var output_owned = try OwnedTensor.init(testing.allocator, .f32, &output_shape);
    defer output_owned.deinit();
    var output = output_owned.toTensor();

    // Create scratch buffer
    var scratch = try matmul.MatmulScratch.init(testing.allocator);
    defer scratch.deinit();

    linear.forward(&input, &output, &scratch);

    const output_data = output_owned.asSlice(f32);

    // Output should contain bias values (since weight is zero)
    try testing.expectEqual(@as(f32, 10.0), output_data[0]);
    try testing.expectEqual(@as(f32, 20.0), output_data[1]);
}

test "Linear.formatKernels formats matmul operation" {
    var buffer: [512]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buffer);

    const shape = [_]usize{ 128, 256 };
    var weight_owned = try OwnedTensor.init(testing.allocator, .f32, &shape);
    defer weight_owned.deinit();
    const weight = weight_owned.toTensor();

    const linear = try Linear.init(&weight, null);

    try linear.formatKernels(fbs.writer(), 2);

    const output = fbs.getWritten();
    // Output format: matmulF32(x[seq, 128], weight[256, 128], dtype=f32) → [seq, 256]
    try testing.expect(std.mem.indexOf(u8, output, "matmul") != null);
    try testing.expect(std.mem.indexOf(u8, output, "128") != null);
    try testing.expect(std.mem.indexOf(u8, output, "256") != null);
}

test "Linear.formatKernels includes bias_add when bias present" {
    var buffer: [512]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buffer);

    const shape = [_]usize{ 128, 256 };
    var weight_owned = try OwnedTensor.init(testing.allocator, .f32, &shape);
    defer weight_owned.deinit();
    const weight = weight_owned.toTensor();

    const bias = try testing.allocator.alloc(f32, 256);
    defer testing.allocator.free(bias);

    const linear = try Linear.init(&weight, bias);

    try linear.formatKernels(fbs.writer(), 2);

    const output = fbs.getWritten();
    try testing.expect(std.mem.indexOf(u8, output, "matmul") != null);
    try testing.expect(std.mem.indexOf(u8, output, "bias_add") != null);
    try testing.expect(std.mem.indexOf(u8, output, "size=256") != null);
}

test "Linear.describe outputs layer information without kernels" {
    var buffer: [512]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buffer);

    const shape = [_]usize{ 64, 128 };
    var weight_owned = try OwnedTensor.init(testing.allocator, .f32, &shape);
    defer weight_owned.deinit();
    const weight = weight_owned.toTensor();

    const linear = try Linear.init(&weight, null);

    try linear.describe(fbs.writer(), 4, false);

    const output = fbs.getWritten();
    try testing.expect(std.mem.indexOf(u8, output, "Linear(in=64, out=128") != null);
    // Should not contain kernel info when show_kernels is false
    try testing.expect(std.mem.indexOf(u8, output, "matmul") == null);
}

test "Linear.describe outputs layer information with kernels" {
    var buffer: [512]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buffer);

    const shape = [_]usize{ 64, 128 };
    var weight_owned = try OwnedTensor.init(testing.allocator, .f32, &shape);
    defer weight_owned.deinit();
    const weight = weight_owned.toTensor();

    const linear = try Linear.init(&weight, null);

    try linear.describe(fbs.writer(), 4, true);

    const output = fbs.getWritten();
    try testing.expect(std.mem.indexOf(u8, output, "Linear(in=64, out=128") != null);
    // Should contain kernel info when show_kernels is true
    try testing.expect(std.mem.indexOf(u8, output, "matmul") != null);
}

test "Linear.formatTo calls formatLinearLike" {
    var buffer: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buffer);

    const shape = [_]usize{ 32, 64 };
    var weight_owned = try OwnedTensor.init(testing.allocator, .f32, &shape);
    defer weight_owned.deinit();
    const weight = weight_owned.toTensor();

    const linear = try Linear.init(&weight, null);

    try linear.formatTo(fbs.writer());

    const output = fbs.getWritten();
    try testing.expect(std.mem.indexOf(u8, output, "Linear(in=32, out=64") != null);
}

test "Embedding.init extracts dimensions from weight tensor" {
    const shape = [_]usize{ 50000, 768 };
    var weight_owned = try OwnedTensor.init(testing.allocator, .f32, &shape);
    defer weight_owned.deinit();
    const weight = weight_owned.toTensor();

    const embedding = Embedding.init(&weight);

    try testing.expectEqual(@as(usize, 50000), embedding.vocab_size);
    try testing.expectEqual(@as(usize, 768), embedding.embed_dim);
}

test "Embedding.forward gathers embeddings for token IDs" {
    const vocab_size = 10;
    const embed_dim = 4;

    // Create weight matrix
    const weight_shape = [_]usize{ vocab_size, embed_dim };
    var weight_owned = try OwnedTensor.init(testing.allocator, .f32, &weight_shape);
    defer weight_owned.deinit();
    const weight = weight_owned.toTensor();
    const weight_data = weight_owned.asSlice(f32);

    // Initialize embeddings with distinct values
    for (0..vocab_size) |i| {
        for (0..embed_dim) |j| {
            weight_data[i * embed_dim + j] = @as(f32, @floatFromInt(i * 10 + j));
        }
    }

    const embedding = Embedding.init(&weight);

    // Create token IDs
    const tokens = [_]u32{ 0, 2, 5 };

    // Create output tensor - gatherEmbeddings expects 3D: [1, seq_len, embed_dim]
    const output_shape = [_]usize{ 1, tokens.len, embed_dim };
    var output_owned = try OwnedTensor.init(testing.allocator, .f32, &output_shape);
    defer output_owned.deinit();
    var output = output_owned.toTensor();

    try embedding.forward(&tokens, &output);

    const output_data = output_owned.asSlice(f32);

    // Verify first token (ID=0) embedding
    try testing.expectEqual(@as(f32, 0.0), output_data[0]);
    try testing.expectEqual(@as(f32, 1.0), output_data[1]);

    // Verify second token (ID=2) embedding
    try testing.expectEqual(@as(f32, 20.0), output_data[4]);
    try testing.expectEqual(@as(f32, 21.0), output_data[5]);

    // Verify third token (ID=5) embedding
    try testing.expectEqual(@as(f32, 50.0), output_data[8]);
    try testing.expectEqual(@as(f32, 51.0), output_data[9]);
}

test "Embedding.formatKernels outputs gather operation" {
    var buffer: [512]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buffer);

    const shape = [_]usize{ 30000, 512 };
    var weight_owned = try OwnedTensor.init(testing.allocator, .f32, &shape);
    defer weight_owned.deinit();
    const weight = weight_owned.toTensor();

    const embedding = Embedding.init(&weight);

    try embedding.formatKernels(fbs.writer(), 2);

    const output = fbs.getWritten();
    // Output format: gather(indices, weight[30000, 512], dtype=f32)
    try testing.expect(std.mem.indexOf(u8, output, "gather") != null);
    try testing.expect(std.mem.indexOf(u8, output, "30000") != null);
    try testing.expect(std.mem.indexOf(u8, output, "512") != null);
}

test "Embedding.describe outputs layer information without kernels" {
    var buffer: [512]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buffer);

    const shape = [_]usize{ 50000, 768 };
    var weight_owned = try OwnedTensor.init(testing.allocator, .f32, &shape);
    defer weight_owned.deinit();
    const weight = weight_owned.toTensor();

    const embedding = Embedding.init(&weight);

    try embedding.describe(fbs.writer(), 4, false);

    const output = fbs.getWritten();
    try testing.expect(std.mem.indexOf(u8, output, "Embedding(vocab=50000, dim=768") != null);
    // Should not contain kernel info
    try testing.expect(std.mem.indexOf(u8, output, "gather") == null);
}

test "Embedding.describe outputs layer information with kernels" {
    var buffer: [512]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buffer);

    const shape = [_]usize{ 50000, 768 };
    var weight_owned = try OwnedTensor.init(testing.allocator, .f32, &shape);
    defer weight_owned.deinit();
    const weight = weight_owned.toTensor();

    const embedding = Embedding.init(&weight);

    try embedding.describe(fbs.writer(), 4, true);

    const output = fbs.getWritten();
    try testing.expect(std.mem.indexOf(u8, output, "Embedding(vocab=50000, dim=768") != null);
    // Should contain kernel info
    try testing.expect(std.mem.indexOf(u8, output, "gather") != null);
}

test "Embedding.formatTo formats standard embedding" {
    var buffer: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buffer);

    const shape = [_]usize{ 32000, 512 };
    var weight_owned = try OwnedTensor.init(testing.allocator, .f32, &shape);
    defer weight_owned.deinit();
    const weight = weight_owned.toTensor();

    const embedding = Embedding.init(&weight);

    try embedding.formatTo(fbs.writer());

    const output = fbs.getWritten();
    try testing.expect(std.mem.indexOf(u8, output, "Embedding(vocab=32000, dim=512") != null);
    try testing.expect(std.mem.indexOf(u8, output, "bits") == null);
}

test "Embedding.formatTo formats quantized embedding with u4" {
    var buffer: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buffer);

    const shape = [_]usize{ 30000, 768 };
    var weight_owned = try OwnedTensor.init(testing.allocator, .grouped_affine_u4, &shape);
    defer weight_owned.deinit();
    const weight = weight_owned.toTensor();

    const embedding = Embedding.init(&weight);

    try embedding.formatTo(fbs.writer());

    const output = fbs.getWritten();
    try testing.expect(std.mem.indexOf(u8, output, "Embedding(vocab=30000, dim=768") != null);
    try testing.expect(std.mem.indexOf(u8, output, "bits=4") != null);
}

test "Embedding.formatTo formats quantized embedding with u8" {
    var buffer: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buffer);

    const shape = [_]usize{ 40000, 1024 };
    var weight_owned = try OwnedTensor.init(testing.allocator, .grouped_affine_u8, &shape);
    defer weight_owned.deinit();
    const weight = weight_owned.toTensor();

    const embedding = Embedding.init(&weight);

    try embedding.formatTo(fbs.writer());

    const output = fbs.getWritten();
    try testing.expect(std.mem.indexOf(u8, output, "Embedding(vocab=40000, dim=1024") != null);
    try testing.expect(std.mem.indexOf(u8, output, "bits=8") != null);
}
