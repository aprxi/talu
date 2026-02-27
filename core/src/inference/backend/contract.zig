//! Backend contract (compile-time).
//!
//! Defines the trait-like surface every inference backend must expose.
//! This has zero runtime overhead: checks run at comptime only.

const std = @import("std");
const topology = @import("../../models/op_types.zig");

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

fn requireBoolField(comptime S: type, comptime owner_name: []const u8, comptime field_name: []const u8) void {
    if (!@hasField(S, field_name)) {
        @compileError("Contract for '" ++ owner_name ++ "' missing required bool field '" ++ field_name ++ "'");
    }
    if (@TypeOf(@field(@as(S, undefined), field_name)) != bool) {
        @compileError("Contract for '" ++ owner_name ++ "' field '" ++ field_name ++ "' must be bool");
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

fn requireTypeAliasEq(comptime actual: type, comptime expected: type, comptime alias_name: []const u8) void {
    if (actual != expected) {
        @compileError("Contract type alias mismatch for '" ++ alias_name ++ "'");
    }
}

fn requireEnumTag(comptime E: type, comptime owner_name: []const u8, comptime tag_name: []const u8) void {
    if (@typeInfo(E) != .@"enum") {
        @compileError("Contract for '" ++ owner_name ++ "' requires enum type");
    }
    inline for (@typeInfo(E).@"enum".fields) |field| {
        if (std.mem.eql(u8, field.name, tag_name)) return;
    }
    @compileError("Contract for '" ++ owner_name ++ "' missing enum tag '" ++ tag_name ++ "'");
}

const required_kernel_names = [_][]const u8{
    "attention",
    "describe_fmt",
    "embedding",
    "ffn",
    "fused_attention",
    "kv_cache",
    "mamba",
    "mla_attention",
    "moe",
    "norm",
    "rope",
    "shortconv",
    "weights",
};

fn assertKernelSupportShape(comptime K: type, comptime backend_name: []const u8) type {
    @setEvalBranchQuota(10_000);

    requireLayoutDecl(K, backend_name, "support");
    const S = @TypeOf(K.support);
    const owner = "Kernel support map for '" ++ backend_name ++ "'";

    inline for (required_kernel_names) |kernel_name| {
        requireBoolField(S, owner, kernel_name);
        requireLayoutDecl(K, backend_name, kernel_name);
    }

    const support_fields = std.meta.fields(S);
    if (support_fields.len != required_kernel_names.len) {
        @compileError(
            "Kernel support map for '" ++ backend_name ++ "' must declare exactly " ++
                std.fmt.comptimePrint("{}", .{required_kernel_names.len}) ++
                " capabilities",
        );
    }

    inline for (support_fields) |field| {
        if (field.type != bool) {
            @compileError(
                "Kernel support map for '" ++ backend_name ++ "' field '" ++ field.name ++ "' must be bool",
            );
        }
        if (!@hasDecl(K, field.name)) {
            @compileError("Kernel module layout for '" ++ backend_name ++ "' missing module '" ++ field.name ++ "'");
        }
    }

    return S;
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
        requireLayoutDecl(M, backend_name, "vision");
        requireLayoutDecl(M, backend_name, "scheduler");
        requireLayoutDecl(M, backend_name, "sampling");
    }
}

/// Assert vision module layout symmetry for backends.
pub fn assertVisionModuleLayout(comptime V: type, comptime backend_name: []const u8) void {
    comptime {
        const owner = backend_name ++ ".vision";
        requireLayoutDecl(V, owner, "PrefillVisionImage");
        requireLayoutDecl(V, owner, "PrefillVisionInput");
        requireLayoutDecl(V, owner, "EncodedVisionOutput");
        requireLayoutDecl(V, owner, "VisionRuntime");
        requireCallableDecl(V, owner, "scatterVisionEmbeddings");
        requireCallableDecl(V, owner, "maxPixels");
    }
}

/// Assert scheduler module layout symmetry.
pub fn assertSchedulerModuleLayout(comptime S: type, comptime backend_name: []const u8) void {
    comptime {
        requireLayoutDecl(S, backend_name ++ ".scheduler", "RequestState");
        requireLayoutDecl(S, backend_name ++ ".scheduler", "Request");
        requireLayoutDecl(S, backend_name ++ ".scheduler", "FinishReason");
        requireLayoutDecl(S, backend_name ++ ".scheduler", "TokenEvent");
        requireLayoutDecl(S, backend_name ++ ".scheduler", "SchedulerConfig");
        requireCallableDecl(S, backend_name ++ ".scheduler", "GenericScheduler");
        requireLayoutDecl(S, backend_name ++ ".scheduler", "Scheduler");
    }
}

/// Assert sampling module layout symmetry.
pub fn assertSamplingModuleLayout(comptime S: type, comptime backend_name: []const u8) void {
    comptime {
        requireLayoutDecl(S, backend_name ++ ".sampling", "SamplingStrategy");
        requireLayoutDecl(S, backend_name ++ ".sampling", "LogitBiasEntry");
        requireLayoutDecl(S, backend_name ++ ".sampling", "SamplingConfig");
        requireLayoutDecl(S, backend_name ++ ".sampling", "Workspace");
        requireLayoutDecl(S, backend_name ++ ".sampling", "Sampler");
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
        requireLayoutDecl(E, backend_name ++ ".executor", "Model");
        requireLayoutDecl(E, backend_name ++ ".executor", "Transformer");
        requireLayoutDecl(E, backend_name ++ ".executor", "Block");
        requireLayoutDecl(E, backend_name ++ ".executor", "TransformerBlock");
        requireLayoutDecl(E, backend_name ++ ".executor", "BlockKind");
        requireLayoutDecl(E, backend_name ++ ".executor", "Attention");
        requireLayoutDecl(E, backend_name ++ ".executor", "RMSNorm");
        requireLayoutDecl(E, backend_name ++ ".executor", "FFNLayer");
        requireLayoutDecl(E, backend_name ++ ".executor", "AttnTemp");
        requireLayoutDecl(E, backend_name ++ ".executor", "AttnCache");
        requireLayoutDecl(E, backend_name ++ ".executor", "ScratchBuffer");

        requireTypeAliasEq(E.Model, E.model.Model, backend_name ++ ".executor.Model");
        requireTypeAliasEq(E.Transformer, E.model.Model, backend_name ++ ".executor.Transformer");
        requireTypeAliasEq(E.Block, E.block.TransformerBlock, backend_name ++ ".executor.Block");
        requireTypeAliasEq(E.TransformerBlock, E.block.TransformerBlock, backend_name ++ ".executor.TransformerBlock");

        requireLayoutDecl(E.model, backend_name ++ ".executor.model", "Model");
        requireLayoutDecl(E.block, backend_name ++ ".executor.block", "TransformerBlock");
        requireLayoutDecl(E.weights, backend_name ++ ".executor.weights", "BlockType");
        requireCallableDecl(E.model.Model, backend_name ++ ".executor.model.Model", "forward");
        requireCallableDecl(E.block.TransformerBlock, backend_name ++ ".executor.block.TransformerBlock", "forward");
        requireCallableDecl(E.Model, backend_name ++ ".executor.Model", "forward");
        requireCallableDecl(E.TransformerBlock, backend_name ++ ".executor.TransformerBlock", "forward");
        requireTypeAliasEq(E.BlockKind, topology.BlockKind, backend_name ++ ".executor.BlockKind");
        requireTypeAliasEq(E.BlockKind, E.weights.BlockType, backend_name ++ ".executor.BlockKind");

        requireEnumTag(E.BlockKind, backend_name ++ ".executor.BlockKind", "attention_mlp");
        requireEnumTag(E.BlockKind, backend_name ++ ".executor.BlockKind", "mamba");
        requireEnumTag(E.BlockKind, backend_name ++ ".executor.BlockKind", "shortconv");
    }
}

/// Assert kernel module layout symmetry for backends.
///
/// Kernel modules represent backend compute capabilities and must remain
/// discoverable with stable names across architectures.
pub fn assertKernelModuleLayout(comptime K: type, comptime backend_name: []const u8) void {
    comptime {
        _ = assertKernelSupportShape(K, backend_name);
    }
}

/// Assert kernel support map presence and shape.
///
/// This makes unsupported capability gaps explicit and grep-friendly.
pub fn assertKernelSupportMap(comptime K: type, comptime backend_name: []const u8) void {
    comptime {
        const S = assertKernelSupportShape(K, backend_name);
        for (std.meta.fields(S)) |field| {
            const kernel_name = field.name;
            const expected = @field(K.support, kernel_name);
            assertKernelSupportDeclMatches(@field(K, kernel_name), backend_name, kernel_name, expected);
        }
    }
}

fn assertKernelSupportDeclMatches(
    comptime Module: type,
    comptime backend_name: []const u8,
    comptime kernel_name: []const u8,
    comptime expected: bool,
) void {
    if (!@hasDecl(Module, "supported")) {
        @compileError("Kernel module '" ++ backend_name ++ ".kernels." ++ kernel_name ++ "' missing decl 'supported'");
    }
    if (@TypeOf(Module.supported) != bool) {
        @compileError("Kernel module '" ++ backend_name ++ ".kernels." ++ kernel_name ++ "' decl `supported` must be bool");
    }
    if (Module.supported != expected) {
        @compileError("Kernel module '" ++ backend_name ++ ".kernels." ++ kernel_name ++ "' has `supported` mismatch with support map");
    }
}

/// Assert kernel symbol symmetry (type + method names) across backends.
pub fn assertKernelSymbolLayout(comptime K: type, comptime backend_name: []const u8) void {
    comptime {
        const S = assertKernelSupportShape(K, backend_name);
        requireLayoutDecl(K, backend_name ++ ".kernels", "TransformerBlock");
        for (std.meta.fields(S)) |field| {
            if (@field(K.support, field.name)) {
                assertKernelContract(K, backend_name, field.name);
            }
        }
    }
}

fn assertKernelContract(comptime K: type, comptime backend_name: []const u8, comptime kernel_name: []const u8) void {
    const Module = @field(K, kernel_name);
    const owner = backend_name ++ ".kernels." ++ kernel_name;

    if (std.mem.eql(u8, kernel_name, "norm")) {
        requireLayoutDecl(Module, owner, "RMSNorm");
        requireLayoutDecl(Module.RMSNorm, owner ++ ".RMSNorm", "ForwardParams");
        requireStructField(Module.RMSNorm.ForwardParams, owner ++ ".RMSNorm.ForwardParams", "input");
        requireStructField(Module.RMSNorm.ForwardParams, owner ++ ".RMSNorm.ForwardParams", "output");
        requireMethodArity(Module.RMSNorm, owner ++ ".RMSNorm", "forward", 3);
    } else if (std.mem.eql(u8, kernel_name, "attention")) {
        requireLayoutDecl(Module, owner, "MultiHeadAttention");
        requireLayoutDecl(Module.MultiHeadAttention, owner ++ ".MultiHeadAttention", "ForwardParams");
        requireStructField(Module.MultiHeadAttention.ForwardParams, owner ++ ".MultiHeadAttention.ForwardParams", "input_tensor");
        requireStructField(Module.MultiHeadAttention.ForwardParams, owner ++ ".MultiHeadAttention.ForwardParams", "output_tensor");
        requireStructField(Module.MultiHeadAttention.ForwardParams, owner ++ ".MultiHeadAttention.ForwardParams", "cache");
        requireStructField(Module.MultiHeadAttention.ForwardParams, owner ++ ".MultiHeadAttention.ForwardParams", "scratch");
        requireStructField(Module.MultiHeadAttention.ForwardParams, owner ++ ".MultiHeadAttention.ForwardParams", "matmul_scratch");
        requireStructField(Module.MultiHeadAttention.ForwardParams, owner ++ ".MultiHeadAttention.ForwardParams", "use_cache");
        requireMethodArity(Module.MultiHeadAttention, owner ++ ".MultiHeadAttention", "forward", 7);
    } else if (std.mem.eql(u8, kernel_name, "ffn")) {
        requireLayoutDecl(Module, owner, "SwiGLU");
        requireLayoutDecl(Module.SwiGLU, owner ++ ".SwiGLU", "ForwardParams");
        requireStructField(Module.SwiGLU.ForwardParams, owner ++ ".SwiGLU.ForwardParams", "input_tensor");
        requireStructField(Module.SwiGLU.ForwardParams, owner ++ ".SwiGLU.ForwardParams", "output_tensor");
        requireStructField(Module.SwiGLU.ForwardParams, owner ++ ".SwiGLU.ForwardParams", "scratch");
        requireStructField(Module.SwiGLU.ForwardParams, owner ++ ".SwiGLU.ForwardParams", "matmul_scratch");
        requireMethodArity(Module.SwiGLU, owner ++ ".SwiGLU", "forward", 5);
    } else if (std.mem.eql(u8, kernel_name, "shortconv")) {
        requireLayoutDecl(Module, owner, "ShortConvKernel");
        requireLayoutDecl(Module.ShortConvKernel, owner ++ ".ShortConvKernel", "ForwardParams");
        requireStructField(Module.ShortConvKernel.ForwardParams, owner ++ ".ShortConvKernel.ForwardParams", "input_tensor");
        requireStructField(Module.ShortConvKernel.ForwardParams, owner ++ ".ShortConvKernel.ForwardParams", "output_tensor");
        requireStructField(Module.ShortConvKernel.ForwardParams, owner ++ ".ShortConvKernel.ForwardParams", "state");
        requireStructField(Module.ShortConvKernel.ForwardParams, owner ++ ".ShortConvKernel.ForwardParams", "scratch");
        requireStructField(Module.ShortConvKernel.ForwardParams, owner ++ ".ShortConvKernel.ForwardParams", "matmul_scratch");
        requireMethodArity(Module.ShortConvKernel, owner ++ ".ShortConvKernel", "forward", 6);
    } else if (std.mem.eql(u8, kernel_name, "moe")) {
        requireLayoutDecl(Module, owner, "MoEFFN");
        requireLayoutDecl(Module.MoEFFN, owner ++ ".MoEFFN", "ForwardParams");
        requireStructField(Module.MoEFFN.ForwardParams, owner ++ ".MoEFFN.ForwardParams", "input_tensor");
        requireStructField(Module.MoEFFN.ForwardParams, owner ++ ".MoEFFN.ForwardParams", "output_tensor");
        requireStructField(Module.MoEFFN.ForwardParams, owner ++ ".MoEFFN.ForwardParams", "scratch");
        requireStructField(Module.MoEFFN.ForwardParams, owner ++ ".MoEFFN.ForwardParams", "matmul_scratch");
        requireMethodArity(Module.MoEFFN, owner ++ ".MoEFFN", "forward", 5);
    } else if (std.mem.eql(u8, kernel_name, "embedding")) {
        requireLayoutDecl(Module, owner, "EmbeddingLookup");
        requireLayoutDecl(Module.EmbeddingLookup, owner ++ ".EmbeddingLookup", "ForwardParams");
        requireStructField(Module.EmbeddingLookup.ForwardParams, owner ++ ".EmbeddingLookup.ForwardParams", "token_ids");
        requireStructField(Module.EmbeddingLookup.ForwardParams, owner ++ ".EmbeddingLookup.ForwardParams", "output_tensor");
        requireMethodArity(Module.EmbeddingLookup, owner ++ ".EmbeddingLookup", "forward", 3);
    } else if (std.mem.eql(u8, kernel_name, "kv_cache")) {
        requireLayoutDecl(Module, owner, "KVCache");
        requireLayoutDecl(Module.KVCache, owner ++ ".KVCache", "ForwardParams");
        requireStructField(Module.KVCache.ForwardParams, owner ++ ".KVCache.ForwardParams", "cache_index");
        requireStructField(Module.KVCache.ForwardParams, owner ++ ".KVCache.ForwardParams", "key_input");
        requireStructField(Module.KVCache.ForwardParams, owner ++ ".KVCache.ForwardParams", "value_input");
        requireMethodArity(Module.KVCache, owner ++ ".KVCache", "forward", 4);
    } else if (std.mem.eql(u8, kernel_name, "fused_attention")) {
        requireLayoutDecl(Module, owner, "FusedAttention");
        requireLayoutDecl(Module.FusedAttention, owner ++ ".FusedAttention", "ForwardParams");
        requireStructField(Module.FusedAttention.ForwardParams, owner ++ ".FusedAttention.ForwardParams", "input_tensor");
        requireStructField(Module.FusedAttention.ForwardParams, owner ++ ".FusedAttention.ForwardParams", "output_tensor");
        requireMethodArity(Module.FusedAttention, owner ++ ".FusedAttention", "forward", 7);
    } else if (std.mem.eql(u8, kernel_name, "rope")) {
        requireLayoutDecl(Module, owner, "RotaryEmbedding");
        requireLayoutDecl(Module.RotaryEmbedding, owner ++ ".RotaryEmbedding", "ForwardParams");
        requireStructField(Module.RotaryEmbedding.ForwardParams, owner ++ ".RotaryEmbedding.ForwardParams", "input_vector");
        requireStructField(Module.RotaryEmbedding.ForwardParams, owner ++ ".RotaryEmbedding.ForwardParams", "output_vector");
        requireStructField(Module.RotaryEmbedding.ForwardParams, owner ++ ".RotaryEmbedding.ForwardParams", "position");
        requireMethodArity(Module.RotaryEmbedding, owner ++ ".RotaryEmbedding", "forward", 4);
    } else if (std.mem.eql(u8, kernel_name, "weights")) {
        requireLayoutDecl(Module, owner, "WeightAccess");
        requireLayoutDecl(Module.WeightAccess, owner ++ ".WeightAccess", "ForwardParams");
        requireStructField(Module.WeightAccess.ForwardParams, owner ++ ".WeightAccess.ForwardParams", "weight_index");
        requireStructField(Module.WeightAccess.ForwardParams, owner ++ ".WeightAccess.ForwardParams", "output_weight");
        requireMethodArity(Module.WeightAccess, owner ++ ".WeightAccess", "forward", 3);
    } else if (std.mem.eql(u8, kernel_name, "describe_fmt") or
        std.mem.eql(u8, kernel_name, "mamba") or
        std.mem.eql(u8, kernel_name, "mla_attention"))
    {
        // Supported surface shape is enforced elsewhere for these modules.
    } else {
        @compileError("No symbol contract defined for kernel '" ++ kernel_name ++ "'");
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
        const S = assertKernelSupportShape(K, backend_name);
        for (std.meta.fields(S)) |field| {
            if (!@field(K.support, field.name)) {
                assertUnsupportedKernelModule(@field(K, field.name), backend_name, field.name);
            }
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

test "assertBackendModuleLayout validates CPU backend module layout" {
    const cpu = @import("cpu/root.zig");
    assertBackendModuleLayout(cpu, "cpu");
}

test "assertVisionModuleLayout validates CPU vision module layout" {
    const cpu = @import("cpu/root.zig");
    assertVisionModuleLayout(cpu.vision, "cpu");
}

test "assertSchedulerModuleLayout validates CPU scheduler module layout" {
    const cpu = @import("cpu/root.zig");
    assertSchedulerModuleLayout(cpu.scheduler, "cpu");
}

test "assertSamplingModuleLayout validates CPU sampling module layout" {
    const cpu = @import("cpu/root.zig");
    assertSamplingModuleLayout(cpu.sampling, "cpu");
}

test "assertExecutorModuleLayout validates CPU executor module layout" {
    const cpu = @import("cpu/root.zig");
    assertExecutorModuleLayout(cpu.executor, "cpu");
}

test "assertExecutorSymbolLayout validates CPU executor symbols" {
    const cpu = @import("cpu/root.zig");
    assertExecutorSymbolLayout(cpu.executor, "cpu");
}

test "assertKernelModuleLayout validates CPU kernel module layout" {
    const cpu = @import("cpu/root.zig");
    assertKernelModuleLayout(cpu.kernels, "cpu");
}

test "assertKernelSupportMap validates CPU kernel support map" {
    const cpu = @import("cpu/root.zig");
    assertKernelSupportMap(cpu.kernels, "cpu");
}

test "assertKernelSymbolLayout validates CPU kernel symbols" {
    const cpu = @import("cpu/root.zig");
    assertKernelSymbolLayout(cpu.kernels, "cpu");
}

test "assertUnsupportedKernelPolicy validates unsupported-kernel contract" {
    const cpu = @import("cpu/root.zig");
    assertUnsupportedKernelPolicy(cpu.kernels, "cpu");
}

test "assertBackendType validates backend trait contract" {
    const cpu = @import("cpu/root.zig");
    assertBackendType(cpu.BackendType);
}
