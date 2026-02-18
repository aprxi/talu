//! Backend contract (compile-time).
//!
//! Defines the trait-like surface every inference backend must expose.
//! This has zero runtime overhead: checks run at comptime only.

const std = @import("std");

/// Pooling strategy for embedding extraction.
pub const PoolingStrategy = enum(u8) {
    /// Use last token's hidden state (default for decoder models).
    last = 0,
    /// Average all token hidden states.
    mean = 1,
    /// Use first token (CLS token for BERT-style models).
    first = 2,
};

pub const Capabilities = struct {
    vision_prefill: bool = false,
    decode_batch: bool = false,
    decode_streaming: bool = false,
    embedding: bool = false,
    warmup: bool = false,
};

/// Request for one token decode step in scheduler/batched mode.
pub const DecodeRequest = struct {
    slot_index: usize,
    token: u32,
};

/// Decoded logits output for one scheduler slot.
pub const DecodeResult = struct {
    slot_index: usize,
    logits: []f32,
};

fn requireDecl(comptime T: type, comptime name: []const u8) void {
    if (!@hasDecl(T, name)) {
        @compileError("Backend type '" ++ @typeName(T) ++ "' missing required decl '" ++ name ++ "'");
    }
}

fn requireLayoutDecl(comptime M: type, comptime backend_name: []const u8, comptime name: []const u8) void {
    if (!@hasDecl(M, name)) {
        @compileError("Backend layout for '" ++ backend_name ++ "' missing required module decl '" ++ name ++ "'");
    }
}

fn requireCallableDecl(comptime M: type, comptime owner_name: []const u8, comptime name: []const u8) void {
    if (!@hasDecl(M, name)) {
        @compileError("Contract for '" ++ owner_name ++ "' missing required callable decl '" ++ name ++ "'");
    }
    const decl = @field(M, name);
    if (@typeInfo(@TypeOf(decl)) != .@"fn") {
        @compileError("Contract for '" ++ owner_name ++ "." ++ name ++ "' requires a function decl");
    }
}

fn requireStructField(comptime S: type, comptime owner_name: []const u8, comptime field_name: []const u8) void {
    if (@typeInfo(S) != .@"struct") {
        @compileError("Contract for '" ++ owner_name ++ "' requires a struct type");
    }
    if (!@hasField(S, field_name)) {
        @compileError("Contract for '" ++ owner_name ++ "' missing required field '" ++ field_name ++ "'");
    }
}

fn requireMethodArity(comptime M: type, comptime owner_name: []const u8, comptime name: []const u8, comptime arity: usize) void {
    requireCallableDecl(M, owner_name, name);
    const decl = @field(M, name);
    const fn_info = @typeInfo(@TypeOf(decl)).@"fn";
    if (fn_info.params.len != arity) {
        @compileError("Contract for '" ++ owner_name ++ "." ++ name ++ "' must have arity " ++ std.fmt.comptimePrint("{}", .{arity}));
    }
}

/// Assert module layout symmetry for backends.
///
/// This enforces the structural design (executor/kernels/engine split) so
/// architectural drift fails at compile time.
pub fn assertBackendModuleLayout(comptime M: type, comptime backend_name: []const u8) void {
    comptime {
        requireLayoutDecl(M, backend_name, "BackendType");
        requireLayoutDecl(M, backend_name, "executor");
        requireLayoutDecl(M, backend_name, "kernels");
        requireLayoutDecl(M, backend_name, "engine");
        requireLayoutDecl(M, backend_name, "scheduler");
        requireLayoutDecl(M, backend_name, "sampling");
    }
}

/// Assert executor module layout symmetry for backends.
///
/// This keeps backend internals organized under consistent responsibility traits.
pub fn assertExecutorModuleLayout(comptime E: type, comptime backend_name: []const u8) void {
    comptime {
        requireLayoutDecl(E, backend_name, "model");
        requireLayoutDecl(E, backend_name, "block");
        requireLayoutDecl(E, backend_name, "weights");
        requireLayoutDecl(E, backend_name, "runtime");
    }
}

/// Assert executor symbol symmetry (type + method names) across backends.
pub fn assertExecutorSymbolLayout(comptime E: type, comptime backend_name: []const u8) void {
    comptime {
        requireLayoutDecl(E.model, backend_name ++ ".executor.model", "Model");
        requireLayoutDecl(E.block, backend_name ++ ".executor.block", "TransformerBlock");
        requireCallableDecl(E.model.Model, backend_name ++ ".executor.model.Model", "forward");
        requireCallableDecl(E.block.TransformerBlock, backend_name ++ ".executor.block.TransformerBlock", "forward");
    }
}

/// Assert kernel module layout symmetry for backends.
///
/// Kernel modules represent backend compute capabilities and must remain
/// discoverable with stable names across architectures.
pub fn assertKernelModuleLayout(comptime K: type, comptime backend_name: []const u8) void {
    comptime {
        requireLayoutDecl(K, backend_name, "attention");
        requireLayoutDecl(K, backend_name, "describe_fmt");
        requireLayoutDecl(K, backend_name, "embedding");
        requireLayoutDecl(K, backend_name, "ffn");
        requireLayoutDecl(K, backend_name, "fused_attention");
        requireLayoutDecl(K, backend_name, "kv_cache");
        requireLayoutDecl(K, backend_name, "mamba");
        requireLayoutDecl(K, backend_name, "mla_attention");
        requireLayoutDecl(K, backend_name, "moe");
        requireLayoutDecl(K, backend_name, "norm");
        requireLayoutDecl(K, backend_name, "rope");
        requireLayoutDecl(K, backend_name, "shortconv");
        requireLayoutDecl(K, backend_name, "weights");
    }
}

/// Assert kernel support map presence and shape.
///
/// This makes unsupported capability gaps explicit and grep-friendly.
pub fn assertKernelSupportMap(comptime K: type, comptime backend_name: []const u8) void {
    comptime {
        requireLayoutDecl(K, backend_name, "support");
        const S = @TypeOf(K.support);
        if (!@hasField(S, "attention")) @compileError("Kernel support map for '" ++ backend_name ++ "' missing 'attention'");
        if (@TypeOf(K.support.attention) != bool) @compileError("Kernel support map for '" ++ backend_name ++ "' field 'attention' must be bool");
        if (!@hasField(S, "describe_fmt")) @compileError("Kernel support map for '" ++ backend_name ++ "' missing 'describe_fmt'");
        if (@TypeOf(K.support.describe_fmt) != bool) @compileError("Kernel support map for '" ++ backend_name ++ "' field 'describe_fmt' must be bool");
        if (!@hasField(S, "embedding")) @compileError("Kernel support map for '" ++ backend_name ++ "' missing 'embedding'");
        if (@TypeOf(K.support.embedding) != bool) @compileError("Kernel support map for '" ++ backend_name ++ "' field 'embedding' must be bool");
        if (!@hasField(S, "ffn")) @compileError("Kernel support map for '" ++ backend_name ++ "' missing 'ffn'");
        if (@TypeOf(K.support.ffn) != bool) @compileError("Kernel support map for '" ++ backend_name ++ "' field 'ffn' must be bool");
        if (!@hasField(S, "fused_attention")) @compileError("Kernel support map for '" ++ backend_name ++ "' missing 'fused_attention'");
        if (@TypeOf(K.support.fused_attention) != bool) @compileError("Kernel support map for '" ++ backend_name ++ "' field 'fused_attention' must be bool");
        if (!@hasField(S, "kv_cache")) @compileError("Kernel support map for '" ++ backend_name ++ "' missing 'kv_cache'");
        if (@TypeOf(K.support.kv_cache) != bool) @compileError("Kernel support map for '" ++ backend_name ++ "' field 'kv_cache' must be bool");
        if (!@hasField(S, "mamba")) @compileError("Kernel support map for '" ++ backend_name ++ "' missing 'mamba'");
        if (@TypeOf(K.support.mamba) != bool) @compileError("Kernel support map for '" ++ backend_name ++ "' field 'mamba' must be bool");
        if (!@hasField(S, "mla_attention")) @compileError("Kernel support map for '" ++ backend_name ++ "' missing 'mla_attention'");
        if (@TypeOf(K.support.mla_attention) != bool) @compileError("Kernel support map for '" ++ backend_name ++ "' field 'mla_attention' must be bool");
        if (!@hasField(S, "moe")) @compileError("Kernel support map for '" ++ backend_name ++ "' missing 'moe'");
        if (@TypeOf(K.support.moe) != bool) @compileError("Kernel support map for '" ++ backend_name ++ "' field 'moe' must be bool");
        if (!@hasField(S, "norm")) @compileError("Kernel support map for '" ++ backend_name ++ "' missing 'norm'");
        if (@TypeOf(K.support.norm) != bool) @compileError("Kernel support map for '" ++ backend_name ++ "' field 'norm' must be bool");
        if (!@hasField(S, "rope")) @compileError("Kernel support map for '" ++ backend_name ++ "' missing 'rope'");
        if (@TypeOf(K.support.rope) != bool) @compileError("Kernel support map for '" ++ backend_name ++ "' field 'rope' must be bool");
        if (!@hasField(S, "shortconv")) @compileError("Kernel support map for '" ++ backend_name ++ "' missing 'shortconv'");
        if (@TypeOf(K.support.shortconv) != bool) @compileError("Kernel support map for '" ++ backend_name ++ "' field 'shortconv' must be bool");
        if (!@hasField(S, "weights")) @compileError("Kernel support map for '" ++ backend_name ++ "' missing 'weights'");
        if (@TypeOf(K.support.weights) != bool) @compileError("Kernel support map for '" ++ backend_name ++ "' field 'weights' must be bool");
    }
}

/// Assert kernel symbol symmetry (type + method names) across backends.
pub fn assertKernelSymbolLayout(comptime K: type, comptime backend_name: []const u8) void {
    comptime {
        requireLayoutDecl(K, backend_name ++ ".kernels", "TransformerBlock");
        requireLayoutDecl(K.norm, backend_name ++ ".kernels.norm", "RMSNorm");
        requireLayoutDecl(K.attention, backend_name ++ ".kernels.attention", "MultiHeadAttention");
        requireLayoutDecl(K.ffn, backend_name ++ ".kernels.ffn", "SwiGLU");
        requireLayoutDecl(K.shortconv, backend_name ++ ".kernels.shortconv", "ShortConvKernel");
        requireLayoutDecl(K.moe, backend_name ++ ".kernels.moe", "MoEFFN");
        requireLayoutDecl(K.embedding, backend_name ++ ".kernels.embedding", "EmbeddingLookup");
        requireLayoutDecl(K.kv_cache, backend_name ++ ".kernels.kv_cache", "KVCache");
        requireLayoutDecl(K.fused_attention, backend_name ++ ".kernels.fused_attention", "FusedAttention");
        requireLayoutDecl(K.rope, backend_name ++ ".kernels.rope", "RotaryEmbedding");
        requireLayoutDecl(K.weights, backend_name ++ ".kernels.weights", "WeightAccess");

        requireLayoutDecl(K.norm.RMSNorm, backend_name ++ ".kernels.norm.RMSNorm", "ForwardParams");
        requireLayoutDecl(K.attention.MultiHeadAttention, backend_name ++ ".kernels.attention.MultiHeadAttention", "ForwardParams");
        requireLayoutDecl(K.ffn.SwiGLU, backend_name ++ ".kernels.ffn.SwiGLU", "ForwardParams");
        requireLayoutDecl(K.shortconv.ShortConvKernel, backend_name ++ ".kernels.shortconv.ShortConvKernel", "ForwardParams");
        requireLayoutDecl(K.moe.MoEFFN, backend_name ++ ".kernels.moe.MoEFFN", "ForwardParams");
        requireLayoutDecl(K.embedding.EmbeddingLookup, backend_name ++ ".kernels.embedding.EmbeddingLookup", "ForwardParams");
        requireLayoutDecl(K.kv_cache.KVCache, backend_name ++ ".kernels.kv_cache.KVCache", "ForwardParams");
        requireLayoutDecl(K.fused_attention.FusedAttention, backend_name ++ ".kernels.fused_attention.FusedAttention", "ForwardParams");
        requireLayoutDecl(K.rope.RotaryEmbedding, backend_name ++ ".kernels.rope.RotaryEmbedding", "ForwardParams");
        requireLayoutDecl(K.weights.WeightAccess, backend_name ++ ".kernels.weights.WeightAccess", "ForwardParams");

        requireStructField(K.norm.RMSNorm.ForwardParams, backend_name ++ ".kernels.norm.RMSNorm.ForwardParams", "input");
        requireStructField(K.norm.RMSNorm.ForwardParams, backend_name ++ ".kernels.norm.RMSNorm.ForwardParams", "output");
        requireStructField(K.ffn.SwiGLU.ForwardParams, backend_name ++ ".kernels.ffn.SwiGLU.ForwardParams", "input_tensor");
        requireStructField(K.ffn.SwiGLU.ForwardParams, backend_name ++ ".kernels.ffn.SwiGLU.ForwardParams", "output_tensor");
        requireStructField(K.ffn.SwiGLU.ForwardParams, backend_name ++ ".kernels.ffn.SwiGLU.ForwardParams", "scratch");
        requireStructField(K.ffn.SwiGLU.ForwardParams, backend_name ++ ".kernels.ffn.SwiGLU.ForwardParams", "matmul_scratch");
        requireStructField(K.attention.MultiHeadAttention.ForwardParams, backend_name ++ ".kernels.attention.MultiHeadAttention.ForwardParams", "input_tensor");
        requireStructField(K.attention.MultiHeadAttention.ForwardParams, backend_name ++ ".kernels.attention.MultiHeadAttention.ForwardParams", "output_tensor");
        requireStructField(K.attention.MultiHeadAttention.ForwardParams, backend_name ++ ".kernels.attention.MultiHeadAttention.ForwardParams", "cache");
        requireStructField(K.attention.MultiHeadAttention.ForwardParams, backend_name ++ ".kernels.attention.MultiHeadAttention.ForwardParams", "scratch");
        requireStructField(K.attention.MultiHeadAttention.ForwardParams, backend_name ++ ".kernels.attention.MultiHeadAttention.ForwardParams", "matmul_scratch");
        requireStructField(K.attention.MultiHeadAttention.ForwardParams, backend_name ++ ".kernels.attention.MultiHeadAttention.ForwardParams", "use_cache");
        requireStructField(K.shortconv.ShortConvKernel.ForwardParams, backend_name ++ ".kernels.shortconv.ShortConvKernel.ForwardParams", "input_tensor");
        requireStructField(K.shortconv.ShortConvKernel.ForwardParams, backend_name ++ ".kernels.shortconv.ShortConvKernel.ForwardParams", "output_tensor");
        requireStructField(K.shortconv.ShortConvKernel.ForwardParams, backend_name ++ ".kernels.shortconv.ShortConvKernel.ForwardParams", "state");
        requireStructField(K.shortconv.ShortConvKernel.ForwardParams, backend_name ++ ".kernels.shortconv.ShortConvKernel.ForwardParams", "scratch");
        requireStructField(K.shortconv.ShortConvKernel.ForwardParams, backend_name ++ ".kernels.shortconv.ShortConvKernel.ForwardParams", "matmul_scratch");
        requireStructField(K.moe.MoEFFN.ForwardParams, backend_name ++ ".kernels.moe.MoEFFN.ForwardParams", "input_tensor");
        requireStructField(K.moe.MoEFFN.ForwardParams, backend_name ++ ".kernels.moe.MoEFFN.ForwardParams", "output_tensor");
        requireStructField(K.moe.MoEFFN.ForwardParams, backend_name ++ ".kernels.moe.MoEFFN.ForwardParams", "scratch");
        requireStructField(K.moe.MoEFFN.ForwardParams, backend_name ++ ".kernels.moe.MoEFFN.ForwardParams", "matmul_scratch");
        requireStructField(K.embedding.EmbeddingLookup.ForwardParams, backend_name ++ ".kernels.embedding.EmbeddingLookup.ForwardParams", "token_ids");
        requireStructField(K.embedding.EmbeddingLookup.ForwardParams, backend_name ++ ".kernels.embedding.EmbeddingLookup.ForwardParams", "output_tensor");
        requireStructField(K.kv_cache.KVCache.ForwardParams, backend_name ++ ".kernels.kv_cache.KVCache.ForwardParams", "cache_index");
        requireStructField(K.kv_cache.KVCache.ForwardParams, backend_name ++ ".kernels.kv_cache.KVCache.ForwardParams", "key_input");
        requireStructField(K.kv_cache.KVCache.ForwardParams, backend_name ++ ".kernels.kv_cache.KVCache.ForwardParams", "value_input");
        requireStructField(K.fused_attention.FusedAttention.ForwardParams, backend_name ++ ".kernels.fused_attention.FusedAttention.ForwardParams", "input_tensor");
        requireStructField(K.fused_attention.FusedAttention.ForwardParams, backend_name ++ ".kernels.fused_attention.FusedAttention.ForwardParams", "output_tensor");
        requireStructField(K.rope.RotaryEmbedding.ForwardParams, backend_name ++ ".kernels.rope.RotaryEmbedding.ForwardParams", "input_vector");
        requireStructField(K.rope.RotaryEmbedding.ForwardParams, backend_name ++ ".kernels.rope.RotaryEmbedding.ForwardParams", "output_vector");
        requireStructField(K.rope.RotaryEmbedding.ForwardParams, backend_name ++ ".kernels.rope.RotaryEmbedding.ForwardParams", "position");
        requireStructField(K.weights.WeightAccess.ForwardParams, backend_name ++ ".kernels.weights.WeightAccess.ForwardParams", "weight_index");
        requireStructField(K.weights.WeightAccess.ForwardParams, backend_name ++ ".kernels.weights.WeightAccess.ForwardParams", "output_weight");

        requireMethodArity(K.norm.RMSNorm, backend_name ++ ".kernels.norm.RMSNorm", "forward", 3);
        requireMethodArity(K.ffn.SwiGLU, backend_name ++ ".kernels.ffn.SwiGLU", "forward", 5);
        requireMethodArity(K.attention.MultiHeadAttention, backend_name ++ ".kernels.attention.MultiHeadAttention", "forward", 7);
        requireMethodArity(K.shortconv.ShortConvKernel, backend_name ++ ".kernels.shortconv.ShortConvKernel", "forward", 6);
        requireMethodArity(K.moe.MoEFFN, backend_name ++ ".kernels.moe.MoEFFN", "forward", 5);
        requireMethodArity(K.embedding.EmbeddingLookup, backend_name ++ ".kernels.embedding.EmbeddingLookup", "forward", 3);
        requireMethodArity(K.kv_cache.KVCache, backend_name ++ ".kernels.kv_cache.KVCache", "forward", 4);
        requireMethodArity(K.fused_attention.FusedAttention, backend_name ++ ".kernels.fused_attention.FusedAttention", "forward", 7);
        requireMethodArity(K.rope.RotaryEmbedding, backend_name ++ ".kernels.rope.RotaryEmbedding", "forward", 4);
        requireMethodArity(K.weights.WeightAccess, backend_name ++ ".kernels.weights.WeightAccess", "forward", 3);
    }
}

fn assertUnsupportedKernelModule(comptime M: type, comptime backend_name: []const u8, comptime kernel_name: []const u8) void {
    if (!@hasDecl(M, "supported")) {
        @compileError("Unsupported kernel module '" ++ backend_name ++ ".kernels." ++ kernel_name ++ "' missing decl 'supported'");
    }
    if (@TypeOf(M.supported) != bool) {
        @compileError("Unsupported kernel module '" ++ backend_name ++ ".kernels." ++ kernel_name ++ "' must expose `pub const supported = bool`");
    }
    if (M.supported) {
        @compileError("Kernel '" ++ backend_name ++ ".kernels." ++ kernel_name ++ "' is marked unsupported in support map but module reports supported=true");
    }
    requireLayoutDecl(M, backend_name ++ ".kernels." ++ kernel_name, "UnsupportedError");
    requireCallableDecl(M, backend_name ++ ".kernels." ++ kernel_name, "unsupported");
    requireCallableDecl(M, backend_name ++ ".kernels." ++ kernel_name, "requireImplemented");
}

/// Assert unsupported-kernel modules expose explicit typed failures.
pub fn assertUnsupportedKernelPolicy(comptime K: type, comptime backend_name: []const u8) void {
    comptime {
        if (!K.support.mamba) {
            assertUnsupportedKernelModule(K.mamba, backend_name, "mamba");
        }
        if (!K.support.mla_attention) {
            assertUnsupportedKernelModule(K.mla_attention, backend_name, "mla_attention");
        }
    }
}

/// Assert a backend type satisfies the trait-like contract.
pub fn assertBackendType(comptime T: type) void {
    comptime {
        requireDecl(T, "capabilities");
        const caps: Capabilities = T.capabilities;

        // Core lifecycle and single-request generation
        requireDecl(T, "init");
        requireDecl(T, "deinit");
        requireDecl(T, "prefill");
        requireDecl(T, "decode");
        requireDecl(T, "maxBatchSize");
        requireDecl(T, "vocabSize");
        _ = @as(fn (*T) void, T.deinit);
        _ = @as(fn (*T, []const u32, []f32) anyerror!void, T.prefill);
        _ = @as(fn (*T, u32, usize, []f32) anyerror!void, T.decode);
        _ = @as(fn (*const T) usize, T.maxBatchSize);
        _ = @as(fn (*const T) usize, T.vocabSize);

        // Scheduler slot lifecycle + batched generation hooks
        requireDecl(T, "allocSlot");
        requireDecl(T, "freeSlot");
        requireDecl(T, "resetSlot");
        requireDecl(T, "getPosition");
        requireDecl(T, "prefillSlot");
        requireDecl(T, "prefillSlotWithVision");
        requireDecl(T, "decodeBatch");
        _ = @as(fn (*T) ?usize, T.allocSlot);
        _ = @as(fn (*T, usize) void, T.freeSlot);
        _ = @as(fn (*T, usize) void, T.resetSlot);
        _ = @as(fn (*const T, usize) usize, T.getPosition);
        _ = @as(fn (*T, usize, []const u32, []f32) anyerror!void, T.prefillSlot);
        _ = @as(fn (*T, []const DecodeRequest, []DecodeResult) anyerror!void, T.decodeBatch);

        if (caps.decode_streaming) {
            requireDecl(T, "decodeStreaming");
            _ = @as(
                fn (*T, u32, usize, usize, []const u32, []u32, ?*const fn (u32, ?*anyopaque) void, ?*anyopaque) anyerror!usize,
                T.decodeStreaming,
            );
        }
        if (caps.embedding) {
            requireDecl(T, "embed");
            requireDecl(T, "embeddingDim");
            _ = @as(fn (*T, []const u32, PoolingStrategy, bool, []f32) anyerror!void, T.embed);
            _ = @as(fn (*const T) usize, T.embeddingDim);
        }
        if (caps.warmup) {
            requireDecl(T, "warmup");
            _ = @as(fn (*T) anyerror!void, T.warmup);
        }
    }
}
