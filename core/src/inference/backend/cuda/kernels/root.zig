//! CUDA kernel contract module.
//!
//! This module keeps backend contract layout explicit while CUDA runtime kernel
//! dispatch remains in `engine.zig` and `core/src/compute/cuda/`.

const std = @import("std");

pub const support = .{
    .attention = true,
    .describe_fmt = false,
    .embedding = false,
    .ffn = true,
    .fused_attention = false,
    .kv_cache = false,
    .mamba = false,
    .mla_attention = false,
    .moe = true,
    .norm = true,
    .rope = false,
    .shortconv = true,
    .weights = false,
};

pub const TransformerBlock = struct {};
pub const ScratchBuffer = struct {};

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

pub const describe_fmt = struct {
    pub const supported = false;
    pub const UnsupportedError = error{UnsupportedKernel};
    pub fn unsupported() UnsupportedError {
        return error.UnsupportedKernel;
    }
    pub fn requireImplemented() UnsupportedError {
        return error.UnsupportedKernel;
    }
};

pub const embedding = struct {
    pub const supported = false;
    pub const UnsupportedError = error{UnsupportedKernel};
    pub fn unsupported() UnsupportedError {
        return error.UnsupportedKernel;
    }
    pub fn requireImplemented() UnsupportedError {
        return error.UnsupportedKernel;
    }

    pub const EmbeddingLookup = struct {
        pub const ForwardParams = struct {
            token_ids: []const u32,
            output_tensor: ?*anyopaque,
        };

        pub fn forward(self: *@This(), params: ForwardParams, p2: ?*anyopaque) !void {
            _ = self;
            _ = params;
            _ = p2;
            return error.UnsupportedModel;
        }
    };
};

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

pub const fused_attention = struct {
    pub const supported = false;
    pub const UnsupportedError = error{UnsupportedKernel};
    pub fn unsupported() UnsupportedError {
        return error.UnsupportedKernel;
    }
    pub fn requireImplemented() UnsupportedError {
        return error.UnsupportedKernel;
    }

    pub const FusedAttention = struct {
        pub const ForwardParams = struct {
            input_tensor: ?*anyopaque,
            output_tensor: ?*anyopaque,
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

pub const kv_cache = struct {
    pub const supported = false;
    pub const UnsupportedError = error{UnsupportedKernel};
    pub fn unsupported() UnsupportedError {
        return error.UnsupportedKernel;
    }
    pub fn requireImplemented() UnsupportedError {
        return error.UnsupportedKernel;
    }

    pub const KVCache = struct {
        pub const ForwardParams = struct {
            cache_index: usize,
            key_input: ?*anyopaque,
            value_input: ?*anyopaque,
        };

        pub fn forward(self: *@This(), params: ForwardParams, p2: ?*anyopaque, p3: ?*anyopaque) !void {
            _ = self;
            _ = params;
            _ = p2;
            _ = p3;
            return error.UnsupportedModel;
        }
    };
};

pub const mamba = struct {
    pub const supported = false;
    pub const UnsupportedError = error{UnsupportedKernel};
    pub fn unsupported() UnsupportedError {
        return error.UnsupportedKernel;
    }
    pub fn requireImplemented() UnsupportedError {
        return error.UnsupportedKernel;
    }
};

pub const mla_attention = struct {
    pub const supported = false;
    pub const UnsupportedError = error{UnsupportedKernel};
    pub fn unsupported() UnsupportedError {
        return error.UnsupportedKernel;
    }
    pub fn requireImplemented() UnsupportedError {
        return error.UnsupportedKernel;
    }
};

pub const moe = struct {
    pub const supported = true;

    pub const MoEFFN = struct {
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

pub const rope = struct {
    pub const supported = false;
    pub const UnsupportedError = error{UnsupportedKernel};
    pub fn unsupported() UnsupportedError {
        return error.UnsupportedKernel;
    }
    pub fn requireImplemented() UnsupportedError {
        return error.UnsupportedKernel;
    }

    pub const RotaryEmbedding = struct {
        pub const ForwardParams = struct {
            input_vector: ?*anyopaque,
            output_vector: ?*anyopaque,
            position: usize,
        };

        pub fn forward(self: *@This(), params: ForwardParams, p2: ?*anyopaque, p3: ?*anyopaque) !void {
            _ = self;
            _ = params;
            _ = p2;
            _ = p3;
            return error.UnsupportedModel;
        }
    };
};

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

pub const weights = struct {
    pub const supported = false;
    pub const UnsupportedError = error{UnsupportedKernel};
    pub fn unsupported() UnsupportedError {
        return error.UnsupportedKernel;
    }
    pub fn requireImplemented() UnsupportedError {
        return error.UnsupportedKernel;
    }

    pub const WeightAccess = struct {
        pub const ForwardParams = struct {
            weight_index: usize,
            output_weight: ?*anyopaque,
        };

        pub fn forward(self: *@This(), params: ForwardParams, p2: ?*anyopaque) !void {
            _ = self;
            _ = params;
            _ = p2;
            return error.UnsupportedModel;
        }
    };
};

pub const EmbeddingLookup = embedding.EmbeddingLookup;
pub const KVCache = kv_cache.KVCache;
pub const FusedAttention = fused_attention.FusedAttention;
pub const RotaryEmbedding = rope.RotaryEmbedding;
pub const WeightAccess = weights.WeightAccess;
pub const MultiHeadAttention = attention.MultiHeadAttention;
pub const SwiGLU = ffn.SwiGLU;
pub const MoEFFN = moe.MoEFFN;
pub const RMSNorm = norm.RMSNorm;

test "unsupported kernel modules expose typed UnsupportedKernel errors" {
    try std.testing.expectEqual(error.UnsupportedKernel, describe_fmt.unsupported());
    try std.testing.expectEqual(error.UnsupportedKernel, embedding.unsupported());
    try std.testing.expectEqual(error.UnsupportedKernel, fused_attention.unsupported());
    try std.testing.expectEqual(error.UnsupportedKernel, kv_cache.unsupported());
    try std.testing.expectEqual(error.UnsupportedKernel, mamba.unsupported());
    try std.testing.expectEqual(error.UnsupportedKernel, mla_attention.unsupported());
    try std.testing.expectEqual(error.UnsupportedKernel, rope.unsupported());
    try std.testing.expectEqual(error.UnsupportedKernel, weights.unsupported());
}

test "unsupported kernel forward shims return UnsupportedModel" {
    var empty_tokens = [_]u32{};
    var embedding_kernel: embedding.EmbeddingLookup = .{};
    try std.testing.expectError(
        error.UnsupportedModel,
        embedding_kernel.forward(.{ .token_ids = empty_tokens[0..], .output_tensor = null }, null),
    );

    var kv_kernel: kv_cache.KVCache = .{};
    try std.testing.expectError(
        error.UnsupportedModel,
        kv_kernel.forward(.{ .cache_index = 0, .key_input = null, .value_input = null }, null, null),
    );

    var fused_attn: fused_attention.FusedAttention = .{};
    try std.testing.expectError(
        error.UnsupportedModel,
        fused_attn.forward(.{ .input_tensor = null, .output_tensor = null }, null, null, null, null, null),
    );

    var rotary: rope.RotaryEmbedding = .{};
    try std.testing.expectError(
        error.UnsupportedModel,
        rotary.forward(.{ .input_vector = null, .output_vector = null, .position = 0 }, null, null),
    );

    var weight_access: weights.WeightAccess = .{};
    try std.testing.expectError(
        error.UnsupportedModel,
        weight_access.forward(.{ .weight_index = 0, .output_weight = null }, null),
    );
}
