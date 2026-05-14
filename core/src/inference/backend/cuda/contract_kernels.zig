//! CUDA kernel contract module.
//!
//! This module keeps backend contract layout explicit while CUDA runtime kernel
//! dispatch remains in `engine.zig` and `core/src/compute/cuda/`.

const std = @import("std");

const UnsupportedKernelError = error{UnsupportedKernel};

fn UnsupportedKernelModule() type {
    return struct {
        pub const supported = false;
        pub const UnsupportedError = UnsupportedKernelError;

        pub fn unsupported() UnsupportedError {
            return error.UnsupportedKernel;
        }

        pub fn requireImplemented() UnsupportedError {
            return error.UnsupportedKernel;
        }
    };
}

pub const support = .{
    .attention = true,
    .describe_fmt = false,
    .embedding = false,
    .ffn = true,
    .fused_attention = false,
    .gated_delta = false,
    .kv_cache = false,
    .mamba = false,
    .mla_attention = false,
    .moe = false,
    .norm = true,
    .rope = false,
    .shortconv = true,
    .weights = false,
};

pub const TransformerBlock = struct {};

pub const attention = struct {
    pub const supported = true;

    pub const MultiHeadAttention = struct {
        pub const ForwardParams = struct {
            input_tensor: ?*anyopaque,
            output_tensor: ?*anyopaque,
            cache: ?*anyopaque,
            scratch: ?*anyopaque,
            matmul_scratch: ?*anyopaque,
            use_cache: bool,
        };

        pub fn forward(
            self: *@This(),
            params: ForwardParams,
            p2: ?*anyopaque,
            p3: ?*anyopaque,
            p4: ?*anyopaque,
            p5: ?*anyopaque,
            p6: ?*anyopaque,
        ) !void {
            _ = self;
            _ = params;
            _ = p2;
            _ = p3;
            _ = p4;
            _ = p5;
            _ = p6;
            return error.UnsupportedModel;
        }
    };
};

pub const describe_fmt = UnsupportedKernelModule();
pub const embedding = UnsupportedKernelModule();

pub const ffn = struct {
    pub const supported = true;

    pub const SwiGLU = struct {
        pub const ForwardParams = struct {
            input_tensor: ?*anyopaque,
            output_tensor: ?*anyopaque,
            scratch: ?*anyopaque,
            matmul_scratch: ?*anyopaque,
        };

        pub fn forward(self: *@This(), params: ForwardParams, p2: ?*anyopaque, p3: ?*anyopaque, p4: ?*anyopaque) !void {
            _ = self;
            _ = params;
            _ = p2;
            _ = p3;
            _ = p4;
            return error.UnsupportedModel;
        }
    };
};

pub const gated_delta = UnsupportedKernelModule();
pub const fused_attention = UnsupportedKernelModule();
pub const kv_cache = UnsupportedKernelModule();
pub const mamba = UnsupportedKernelModule();
pub const mla_attention = UnsupportedKernelModule();
pub const moe = UnsupportedKernelModule();

pub const norm = struct {
    pub const supported = true;

    pub const RMSNorm = struct {
        pub const ForwardParams = struct {
            input: ?*anyopaque,
            output: ?*anyopaque,
        };

        pub fn forward(self: *@This(), params: ForwardParams, p2: ?*anyopaque) !void {
            _ = self;
            _ = params;
            _ = p2;
            return error.UnsupportedModel;
        }
    };
};

pub const rope = UnsupportedKernelModule();

pub const shortconv = struct {
    pub const supported = true;

    pub const ShortConvKernel = struct {
        pub const ForwardParams = struct {
            input_tensor: ?*anyopaque,
            output_tensor: ?*anyopaque,
            state: ?*anyopaque,
            scratch: ?*anyopaque,
            matmul_scratch: ?*anyopaque,
        };

        pub fn forward(
            self: *@This(),
            params: ForwardParams,
            p2: ?*anyopaque,
            p3: ?*anyopaque,
            p4: ?*anyopaque,
            p5: ?*anyopaque,
        ) !void {
            _ = self;
            _ = params;
            _ = p2;
            _ = p3;
            _ = p4;
            _ = p5;
            return error.UnsupportedModel;
        }
    };
};

pub const weights = UnsupportedKernelModule();

test "unsupported and requireImplemented expose typed UnsupportedKernel errors" {
    const unsupported_modules = .{
        describe_fmt,
        embedding,
        fused_attention,
        gated_delta,
        kv_cache,
        mamba,
        mla_attention,
        moe,
        rope,
        weights,
    };

    inline for (unsupported_modules) |Module| {
        try std.testing.expectEqual(error.UnsupportedKernel, Module.unsupported());
        try std.testing.expectEqual(error.UnsupportedKernel, Module.requireImplemented());
    }
}
