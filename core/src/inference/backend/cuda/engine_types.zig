//! Shared types, constants, and support structures for the CUDA backend engine.
//!
//! This file contains all non-CudaBackend types, constants, and support
//! structures used across the engine sub-modules.

const std = @import("std");
const build_options = @import("build_options");
const models = @import("models_pkg");
const layer_ops = models.layer_ops;
const op_types = models.op_types;
const opcode_map = models.plan.opcode_map;
const plan_compiler = models.plan.compiler;
const rope_scaling = models.rope_scaling;
const runtime_contract = @import("runtime_contract_pkg");
const backend_root = @import("../root.zig");
const contract = @import("../contract.zig");
const compute = @import("compute_pkg");
const tensor = @import("tensor_pkg");
const dtype = @import("dtype_pkg");
const log = @import("log_pkg");
const trace = @import("xray_pkg").trace;
const load_transforms = @import("models_pkg").load.transforms;
const vision_types = @import("../../vision_types.zig");
const common_mrope = @import("../../vision_mrope.zig");
const smoke_checks = @import("smoke_checks.zig");
const attention_policy = @import("attention_policy.zig");
const attention_mod = @import("attention.zig");
const decode_mod = @import("decode.zig");
const prefill_mod = @import("prefill.zig");
const sampling_mod = @import("sampling.zig");
const vision_runtime_mod = @import("vision/root.zig");
const cpu_kernels = @import("../cpu/kernels/root.zig");
const cpu_conv1d = compute.cpu.conv1d_depthwise;
const cpu_gated_delta = compute.cpu.gated_delta;
const GateUpLayout = models.runtime_blocks.GateUpLayout;

const LoadedModel = models.LoadedModel;
const Tensor = tensor.Tensor;

// --- Weight upload functions from engine_weights.zig ---
const engine_weights = @import("engine_weights.zig");
const uploadLinearWeight = engine_weights.uploadLinearWeight;
const uploadLinearWeightWithContext = engine_weights.uploadLinearWeightWithContext;
const uploadTensor = engine_weights.uploadTensor;
const uploadFusedQkvWeights = engine_weights.uploadFusedQkvWeights;
const uploadFusedGateUpWeights = engine_weights.uploadFusedGateUpWeights;
const uploadMoEWeights = engine_weights.uploadMoEWeights;
const tryPopulateFinalNormWeight = engine_weights.tryPopulateFinalNormWeight;
const canUseModelEmbeddings = engine_weights.canUseModelEmbeddings;
const tryUploadEmbeddingLookup = engine_weights.tryUploadEmbeddingLookup;
const uploadVectorTensor = engine_weights.uploadVectorTensor;
const uploadShortConvWeightTimeMajor = engine_weights.uploadShortConvWeightTimeMajor;
const allocZeroedF32Buffer = engine_weights.allocZeroedF32Buffer;
const materializeTensorF32 = engine_weights.materializeTensorF32;
const resizeScratchBuffer = engine_weights.resizeScratchBuffer;
const bufferSlice = engine_weights.bufferSlice;
const kvCacheBytesForCapacityDtype = engine_weights.kvCacheBytesForCapacityDtype;
const allocDeviceKvPairWithScales = engine_weights.allocDeviceKvPairWithScales;

pub const prototype_eps: f32 = 1e-5;
pub const initial_kv_cache_tokens: usize = 512;
pub const KvCacheDtype = enum(u8) {
    f16,
    i8,
    fp8,

    pub fn elementBytes(self: KvCacheDtype) usize {
        return switch (self) {
            .f16 => @sizeOf(u16),
            .i8 => 1,
            .fp8 => 1,
        };
    }

    pub fn hasPerHeadScales(self: KvCacheDtype) bool {
        return switch (self) {
            .f16 => false,
            .i8 => true,
            .fp8 => true,
        };
    }
};
pub fn resolveKvCacheDtype() KvCacheDtype {
    const raw = std.posix.getenv("TALU_KV_QUANT") orelse return .i8;
    if (std.ascii.eqlIgnoreCase(raw, "f16") or std.ascii.eqlIgnoreCase(raw, "fp16")) return .f16;
    if (std.ascii.eqlIgnoreCase(raw, "fp8") or std.ascii.eqlIgnoreCase(raw, "e4m3")) return .fp8;
    return .i8;
}
pub const enable_fused_attention_f16_kv: bool = false;
pub const max_fused_attention_f16_kv_seq_len: u32 = 384;
pub const default_prefill_chunk_rows_cap: usize = 1024;
pub const enable_device_embedding_lookup: bool = false;
pub const max_supported_fused_f16_kv_head_dim = 512;
// Optional dispatch observability. Keep disabled by default so production
// execution adds zero atomic overhead in the token loop.
pub const enable_dispatch_observability: bool = false;
pub const attention_policy_config = attention_policy.Config{
    .enable_fused_attention_f16_kv = enable_fused_attention_f16_kv,
    .max_fused_attention_f16_kv_seq_len = max_fused_attention_f16_kv_seq_len,
    .max_supported_fused_f16_kv_head_dim = max_supported_fused_f16_kv_head_dim,
};
pub const run_startup_selftests = build_options.cuda_startup_selftests;
pub const gaffine_scales_dtype_f16 = compute.cuda.gaffine_u4_matvec.scales_dtype_f16;
pub const gaffine_scales_dtype_bf16 = compute.cuda.gaffine_u4_matvec.scales_dtype_bf16;
pub const DenseU16Dtype = enum(u8) {
    f16,
    bf16,
};

pub const EmbeddingLookupKind = enum(u8) {
    f32,
    f16,
    bf16,
    gaffine_u4,
};

pub fn saturatingU64FromU128(value: u128) u64 {
    return if (value > std.math.maxInt(u64)) std.math.maxInt(u64) else @intCast(value);
}

pub fn saturatingAddUsize(a: usize, b: usize) usize {
    return std.math.add(usize, a, b) catch std.math.maxInt(usize);
}

pub fn parseEnvBoolValue(raw: []const u8) ?bool {
    if (std.ascii.eqlIgnoreCase(raw, "1")) return true;
    if (std.ascii.eqlIgnoreCase(raw, "true")) return true;
    if (std.ascii.eqlIgnoreCase(raw, "yes")) return true;
    if (std.ascii.eqlIgnoreCase(raw, "on")) return true;
    if (std.ascii.eqlIgnoreCase(raw, "0")) return false;
    if (std.ascii.eqlIgnoreCase(raw, "false")) return false;
    if (std.ascii.eqlIgnoreCase(raw, "no")) return false;
    if (std.ascii.eqlIgnoreCase(raw, "off")) return false;
    return null;
}

pub fn resolveEnvBool(name: []const u8, default_value: bool) bool {
    const raw = std.process.getEnvVarOwned(std.heap.c_allocator, name) catch {
        return default_value;
    };
    defer std.heap.c_allocator.free(raw);
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    return parseEnvBoolValue(trimmed) orelse blk: {
        log.warn("inference", "Invalid boolean env; using default", .{
            .name = name,
            .value = trimmed,
            .default = @as(u8, @intFromBool(default_value)),
        });
        break :blk default_value;
    };
}

pub fn resolveCudaFixedAllocMode() bool {
    return resolveEnvBool("TALU_CUDA_FIXED_ALLOC", false);
}

pub fn resolveCudaRequireFitCheck() bool {
    return resolveEnvBool("TALU_CUDA_REQUIRE_FIT", false);
}

pub fn resolveCudaStrictMemoryMode() bool {
    return resolveEnvBool("TALU_CUDA_STRICT_MEMORY", false);
}

pub fn resolveCudaGaffineU4Tile8Decode() bool {
    return resolveEnvBool("TALU_CUDA_GAFFINE_U4_TILE8", false);
}

pub fn resolveCudaGaffineU4DecodeI8() bool {
    return resolveEnvBool("TALU_CUDA_GAFFINE_U4_DECODE_I8", true);
}

pub fn resolveCudaGatedDeltaSsmI8State(default_value: bool) bool {
    return resolveEnvBool("TALU_CUDA_GD_SSM_I8_STATE", default_value);
}

pub fn resolveCudaQuantizeFp8() bool {
    return resolveEnvBool("TALU_CUDA_QUANTIZE_FP8", false);
}

pub fn resolveCudaEnableStandaloneLayerScalars() bool {
    return resolveEnvBool("TALU_CUDA_STANDALONE_LAYER_SCALARS", true);
}

pub fn resolveCudaMemoryReserveBytes() usize {
    const raw = std.process.getEnvVarOwned(std.heap.c_allocator, "TALU_CUDA_MEMORY_RESERVE_MIB") catch {
        return 0;
    };
    defer std.heap.c_allocator.free(raw);
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    const reserve_mib = std.fmt.parseUnsigned(usize, trimmed, 10) catch {
        log.warn("inference", "Invalid TALU_CUDA_MEMORY_RESERVE_MIB; using default", .{
            .value = trimmed,
            .default_mib = 0,
        });
        return 0;
    };
    return std.math.mul(usize, reserve_mib, 1024 * 1024) catch std.math.maxInt(usize);
}

pub fn resolveCudaExternalOverheadCapBytes() ?usize {
    const raw = std.process.getEnvVarOwned(std.heap.c_allocator, "TALU_CUDA_EXTERNAL_OVERHEAD_MIB") catch {
        return null;
    };
    defer std.heap.c_allocator.free(raw);
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    const cap_mib = std.fmt.parseUnsigned(usize, trimmed, 10) catch {
        log.warn("inference", "Invalid TALU_CUDA_EXTERNAL_OVERHEAD_MIB; ignoring", .{
            .value = trimmed,
        });
        return null;
    };
    return std.math.mul(usize, cap_mib, 1024 * 1024) catch std.math.maxInt(usize);
}

pub fn resolveCudaMaxSeqLen(model_max_seq_len: usize) usize {
    const raw = std.process.getEnvVarOwned(std.heap.c_allocator, "TALU_CUDA_MAX_SEQ_LEN") catch {
        return model_max_seq_len;
    };
    defer std.heap.c_allocator.free(raw);
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    const parsed = std.fmt.parseUnsigned(usize, trimmed, 10) catch {
        log.warn("inference", "Invalid TALU_CUDA_MAX_SEQ_LEN; using model max", .{
            .value = trimmed,
            .model_max_seq = model_max_seq_len,
        });
        return model_max_seq_len;
    };
    if (parsed == 0) {
        log.warn("inference", "TALU_CUDA_MAX_SEQ_LEN must be >= 1; using model max", .{
            .value = parsed,
            .model_max_seq = model_max_seq_len,
        });
        return model_max_seq_len;
    }
    const resolved = @min(model_max_seq_len, parsed);
    if (resolved != model_max_seq_len) {
        log.info("inference", "CUDA max_seq_len clamped by TALU_CUDA_MAX_SEQ_LEN", .{
            .model_max_seq = model_max_seq_len,
            .effective_max_seq = resolved,
        });
    }
    return resolved;
}

pub fn resolveSharedKvSourceLayer(config: tensor.ModelConfig, layer_idx: usize) ?usize {
    if (config.num_kv_shared_layers <= 0) return null;
    const layer_types = config.layer_types orelse return null;
    const n_layers: usize = @intCast(config.n_layers);
    if (layer_types.len != n_layers) return null;
    if (layer_idx >= n_layers) return null;

    const shared_count: usize = @min(@as(usize, @intCast(config.num_kv_shared_layers)), n_layers);
    if (shared_count == 0 or shared_count == n_layers) return null;
    const first_shared_layer = n_layers - shared_count;
    if (layer_idx < first_shared_layer or first_shared_layer == 0) return null;

    const target_layer_type = layer_types[layer_idx];
    var src = first_shared_layer;
    while (src > 0) {
        src -= 1;
        if (layer_types[src] == target_layer_type) return src;
    }
    return null;
}

/// For multi-GPU topologies, ensure the split point between GPU stages does
/// not separate KV-shared layers from their source layers. Returns the
/// adjusted split (may be lower than `proposed_split`). Returns null when
/// no valid split exists (source layer below `floor`).
pub fn adjustSplitForKvSharing(config: tensor.ModelConfig, proposed_split: usize, total_layers: usize, floor: usize) ?usize {
    if (config.num_kv_shared_layers <= 0) return proposed_split;
    var min_source = proposed_split;
    var layer_idx = proposed_split;
    while (layer_idx < total_layers) : (layer_idx += 1) {
        if (resolveSharedKvSourceLayer(config, layer_idx)) |src| {
            min_source = @min(min_source, src);
        }
    }
    if (min_source <= floor) return null;
    return min_source;
}

pub fn resolveCudaInitialKvCacheTokens(max_seq_len: usize) usize {
    const fallback = @min(max_seq_len, initial_kv_cache_tokens);
    const raw = std.process.getEnvVarOwned(std.heap.c_allocator, "TALU_CUDA_KV_INIT_TOKENS") catch {
        return fallback;
    };
    defer std.heap.c_allocator.free(raw);
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    const parsed = std.fmt.parseUnsigned(usize, trimmed, 10) catch {
        log.warn("inference", "Invalid TALU_CUDA_KV_INIT_TOKENS; using fallback", .{
            .value = trimmed,
            .fallback = fallback,
        });
        return fallback;
    };
    if (parsed == 0) {
        log.warn("inference", "TALU_CUDA_KV_INIT_TOKENS must be >= 1; using fallback", .{
            .value = parsed,
            .fallback = fallback,
        });
        return fallback;
    }
    return @min(max_seq_len, parsed);
}

pub fn resolveCudaPrefillChunkRowsCap(max_seq_len: usize) usize {
    const fallback = @min(max_seq_len, default_prefill_chunk_rows_cap);
    const raw = std.process.getEnvVarOwned(std.heap.c_allocator, "TALU_CUDA_PREFILL_CHUNK_ROWS") catch {
        return fallback;
    };
    defer std.heap.c_allocator.free(raw);
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    const parsed = std.fmt.parseUnsigned(usize, trimmed, 10) catch {
        log.warn("inference", "Invalid TALU_CUDA_PREFILL_CHUNK_ROWS; using fallback", .{
            .value = trimmed,
            .fallback = fallback,
        });
        return fallback;
    };
    if (parsed == 0) {
        log.warn("inference", "TALU_CUDA_PREFILL_CHUNK_ROWS must be >= 1; using fallback", .{
            .value = parsed,
            .fallback = fallback,
        });
        return fallback;
    }
    return @min(max_seq_len, parsed);
}

pub const KernelSlot = enum {
    vector_add,
    vector_add_scaled,
    vector_add_rows_strided,
    vector_add_scaled_rows_strided,
    mul,
    copy,
    copy_u16,
    cast_f32_to_f16,
    cast_f32_to_bf16,
    cast_bf16_to_f32,
    embedding_lookup_f32,
    embedding_lookup_u16,
    embedding_lookup_u16_rows,
    embedding_lookup_gaffine_u4,
    kv_write_f16,
    kv_write_f16_rows,
    kv_write_f16_rows_ptrs,
    rmsnorm,
    rmsnorm_rows_strided,
    rope,
    rope_store_f16,
    attn_scores_heads_f32,
    attn_scores_heads_f16_kv,
    attn_fused_heads_f16_kv,
    attn_fused_decode_heads_f16_kv_ptrs,
    attn_fused_prefill_heads_f16_kv,
    attn_fused_prefill_heads_f16_kv_gqa,
    causal_attn_softmax_f32,
    softmax_rows,
    attn_weighted_sum_heads_f32,
    attn_weighted_sum_heads_f16_kv,
    silu,
    silu_mul,
    gelu_mul,
    shortconv_step,
    gated_attention_compact_q,
    gated_attention_output_gate,
    gated_delta_conv,
    gated_delta_conv_silu,
    gated_delta_conv_silu_rows,
    gated_delta_conv_silu_rows_ptrs,
    gated_delta_advance_ring_heads,
    gated_delta_qk_norm,
    gated_delta_ssm,
    gated_delta_ssm_rows,
    gated_delta_ssm_rows_ptrs,
    gated_delta_ssm_rows_i8,
    gated_delta_ssm_rows_ptrs_i8,
    gated_delta_rmsnorm_silu_mul,
    gated_delta_rmsnorm_silu_mul_rows,
    argmax,
    matmul_f16,
    matmul_bf16,
    matvec_f16,
    matvec_bf16,
    matvec_gate_up_f16,
    matvec_gate_up_bf16,
    matvec_gate_up_silu_f16,
    matvec_gate_up_silu_bf16,
    matvec_qkv_f16,
    matvec_qkv_bf16,
    gaffine_u4_matvec,
    gaffine_u4_matvec_tile8,
    gaffine_u8_matvec,
    gaffine_u4_matvec_gate_up,
    gaffine_u4_matvec_qkv,
    gaffine_u4_matvec_qkv_tile8,
    gaffine_u8_matvec_qkv,
    gaffine_u8_matvec_gate_up,
    gaffine_u4_matvec_gate_up_silu,
    gaffine_u4_matvec_gate_up_silu_tile8,
    gaffine_u8_matvec_gate_up_silu,
    gaffine_u4_dequant_f16,
    gaffine_u8_dequant_f16,
    rope_rows_ptrs,
    attn_scores_heads_f16_kv_ptrs,
    softmax_rows_dynamic_cols_ptrs,
    attn_weighted_sum_heads_f16_kv_ptrs,
    kv_write_i8,
    kv_write_i8_rows,
    kv_write_i8_rows_ptrs,
    dequant_kv_i8_to_f16,
    rope_store_i8,
    attn_scores_heads_i8_kv,
    attn_weighted_sum_heads_i8_kv,
    attn_fused_heads_i8_kv,
    attn_fused_decode_heads_i8_kv_ptrs,
    attn_fused_prefill_heads_i8_kv,
    attn_fused_prefill_heads_i8_kv_gqa,
    attn_scores_heads_i8_kv_ptrs,
    attn_weighted_sum_heads_i8_kv_ptrs,
    kv_write_fp8,
    kv_write_fp8_rows,
    kv_write_fp8_rows_ptrs,
    dequant_kv_fp8_to_f16,
    rope_store_fp8,
    attn_scores_heads_fp8_kv,
    attn_scores_heads_fp8_kv_ptrs,
    attn_weighted_sum_heads_fp8_kv,
    attn_weighted_sum_heads_fp8_kv_ptrs,
    attn_fused_heads_fp8_kv,
    attn_fused_decode_heads_fp8_kv_ptrs,
    attn_fused_prefill_heads_fp8_kv,
    attn_fused_prefill_heads_fp8_kv_gqa,
    flash_decode_f16,
    flash_decode_i8,
    flash_decode_fp8,
    flash_decode_reduce,
    flash_prefill_f16,
    flash_prefill_i8,
    flash_prefill_fp8,
};

pub const RequiredKernel = struct {
    slot: KernelSlot,
    op_name: []const u8,
    embedded_symbol: [:0]const u8,
};

pub const ProjectionPath = enum {
    fused,
    unfused,
};

pub const Nvfp4RouteKind = enum {
    native_cublaslt,
    bf16_fallback,
    small_rows_matvec,
    fused_qkv,
    fused_gate_up,
};

pub const Nvfp4RouteCounters = struct {
    native_cublaslt: u64 = 0,
    bf16_fallback: u64 = 0,
    small_rows_matvec: u64 = 0,
    fused_qkv: u64 = 0,
    fused_gate_up: u64 = 0,

    pub fn record(self: *Nvfp4RouteCounters, kind: Nvfp4RouteKind) void {
        switch (kind) {
            .native_cublaslt => self.native_cublaslt += 1,
            .bf16_fallback => self.bf16_fallback += 1,
            .small_rows_matvec => self.small_rows_matvec += 1,
            .fused_qkv => self.fused_qkv += 1,
            .fused_gate_up => self.fused_gate_up += 1,
        }
    }

    fn saturatingSub(current: u64, start: u64) u64 {
        return if (current >= start) current - start else 0;
    }

    pub fn delta(current: Nvfp4RouteCounters, start: Nvfp4RouteCounters) Nvfp4RouteCounters {
        return .{
            .native_cublaslt = saturatingSub(current.native_cublaslt, start.native_cublaslt),
            .bf16_fallback = saturatingSub(current.bf16_fallback, start.bf16_fallback),
            .small_rows_matvec = saturatingSub(current.small_rows_matvec, start.small_rows_matvec),
            .fused_qkv = saturatingSub(current.fused_qkv, start.fused_qkv),
            .fused_gate_up = saturatingSub(current.fused_gate_up, start.fused_gate_up),
        };
    }

    pub fn total(self: *const Nvfp4RouteCounters) u64 {
        const total_u128 = @as(u128, self.native_cublaslt) +
            @as(u128, self.bf16_fallback) +
            @as(u128, self.small_rows_matvec) +
            @as(u128, self.fused_qkv) +
            @as(u128, self.fused_gate_up);
        return saturatingU64FromU128(total_u128);
    }
};

pub const Nvfp4PhaseBudgetCounters = struct {
    linear_calls: u64 = 0,
    linear_ns: u64 = 0,
    attention_calls: u64 = 0,
    attention_ns: u64 = 0,
    attention_causal_calls: u64 = 0,
    attention_noncausal_calls: u64 = 0,
    attention_context_calls: u64 = 0,
    attention_batched_prefill_calls: u64 = 0,
    layer_scalar_calls: u64 = 0,
    layer_scalar_ns: u64 = 0,
    rmsnorm_calls: u64 = 0,
    rmsnorm_ns: u64 = 0,
    residual_add_calls: u64 = 0,
    residual_add_ns: u64 = 0,
    qkv_calls: u64 = 0,
    qkv_fused_calls: u64 = 0,
    qkv_unfused_calls: u64 = 0,
    gate_up_calls: u64 = 0,
    gate_up_fused_calls: u64 = 0,
    gate_up_unfused_calls: u64 = 0,
    attention_fused_heads_f16_kv: u64 = 0,
    attention_heads_f16_kv: u64 = 0,
    attention_heads_lowbit_bridge_f16_kv: u64 = 0,
    attention_fused_heads_i8_kv: u64 = 0,
    attention_heads_i8_kv: u64 = 0,
    attention_fused_heads_fp8_kv: u64 = 0,
    attention_heads_fp8_kv: u64 = 0,
    attention_heads_f32_kv: u64 = 0,
    attention_fused_heads_f16_kv_ns: u64 = 0,
    attention_heads_f16_kv_ns: u64 = 0,
    attention_heads_lowbit_bridge_f16_kv_ns: u64 = 0,
    attention_fused_heads_i8_kv_ns: u64 = 0,
    attention_heads_i8_kv_ns: u64 = 0,
    attention_fused_heads_fp8_kv_ns: u64 = 0,
    attention_heads_fp8_kv_ns: u64 = 0,
    attention_heads_f32_kv_ns: u64 = 0,

    fn saturatingAddU64(a: u64, b: u64) u64 {
        return std.math.add(u64, a, b) catch std.math.maxInt(u64);
    }

    fn saturatingSub(current: u64, start: u64) u64 {
        return if (current >= start) current - start else 0;
    }

    pub fn recordLinear(self: *Nvfp4PhaseBudgetCounters, elapsed_ns: u64) void {
        self.linear_calls = saturatingAddU64(self.linear_calls, 1);
        self.linear_ns = saturatingAddU64(self.linear_ns, elapsed_ns);
    }

    pub fn recordAttention(self: *Nvfp4PhaseBudgetCounters, path: AttentionPath, elapsed_ns: u64) void {
        self.attention_calls = saturatingAddU64(self.attention_calls, 1);
        self.attention_ns = saturatingAddU64(self.attention_ns, elapsed_ns);
        switch (path) {
            .fused_heads_f16_kv => {
                self.attention_fused_heads_f16_kv = saturatingAddU64(self.attention_fused_heads_f16_kv, 1);
                self.attention_fused_heads_f16_kv_ns = saturatingAddU64(self.attention_fused_heads_f16_kv_ns, elapsed_ns);
            },
            .heads_f16_kv => {
                self.attention_heads_f16_kv = saturatingAddU64(self.attention_heads_f16_kv, 1);
                self.attention_heads_f16_kv_ns = saturatingAddU64(self.attention_heads_f16_kv_ns, elapsed_ns);
            },
            .heads_lowbit_bridge_f16_kv => {
                self.attention_heads_lowbit_bridge_f16_kv = saturatingAddU64(self.attention_heads_lowbit_bridge_f16_kv, 1);
                self.attention_heads_lowbit_bridge_f16_kv_ns = saturatingAddU64(self.attention_heads_lowbit_bridge_f16_kv_ns, elapsed_ns);
            },
            .fused_heads_i8_kv => {
                self.attention_fused_heads_i8_kv = saturatingAddU64(self.attention_fused_heads_i8_kv, 1);
                self.attention_fused_heads_i8_kv_ns = saturatingAddU64(self.attention_fused_heads_i8_kv_ns, elapsed_ns);
            },
            .heads_i8_kv => {
                self.attention_heads_i8_kv = saturatingAddU64(self.attention_heads_i8_kv, 1);
                self.attention_heads_i8_kv_ns = saturatingAddU64(self.attention_heads_i8_kv_ns, elapsed_ns);
            },
            .fused_heads_fp8_kv => {
                self.attention_fused_heads_fp8_kv = saturatingAddU64(self.attention_fused_heads_fp8_kv, 1);
                self.attention_fused_heads_fp8_kv_ns = saturatingAddU64(self.attention_fused_heads_fp8_kv_ns, elapsed_ns);
            },
            .heads_fp8_kv => {
                self.attention_heads_fp8_kv = saturatingAddU64(self.attention_heads_fp8_kv, 1);
                self.attention_heads_fp8_kv_ns = saturatingAddU64(self.attention_heads_fp8_kv_ns, elapsed_ns);
            },
            .heads_f32_kv => {
                self.attention_heads_f32_kv = saturatingAddU64(self.attention_heads_f32_kv, 1);
                self.attention_heads_f32_kv_ns = saturatingAddU64(self.attention_heads_f32_kv_ns, elapsed_ns);
            },
        }
    }

    pub fn recordAttentionCausality(self: *Nvfp4PhaseBudgetCounters, is_causal: bool) void {
        if (is_causal) {
            self.attention_causal_calls = saturatingAddU64(self.attention_causal_calls, 1);
        } else {
            self.attention_noncausal_calls = saturatingAddU64(self.attention_noncausal_calls, 1);
        }
    }

    pub fn recordAttentionContext(self: *Nvfp4PhaseBudgetCounters) void {
        self.attention_context_calls = saturatingAddU64(self.attention_context_calls, 1);
    }

    pub fn recordAttentionBatchedPrefill(self: *Nvfp4PhaseBudgetCounters) void {
        self.attention_batched_prefill_calls = saturatingAddU64(self.attention_batched_prefill_calls, 1);
    }

    pub fn recordLayerScalar(self: *Nvfp4PhaseBudgetCounters, elapsed_ns: u64) void {
        self.layer_scalar_calls = saturatingAddU64(self.layer_scalar_calls, 1);
        self.layer_scalar_ns = saturatingAddU64(self.layer_scalar_ns, elapsed_ns);
    }

    pub fn recordRmsnorm(self: *Nvfp4PhaseBudgetCounters, elapsed_ns: u64) void {
        self.rmsnorm_calls = saturatingAddU64(self.rmsnorm_calls, 1);
        self.rmsnorm_ns = saturatingAddU64(self.rmsnorm_ns, elapsed_ns);
    }

    pub fn recordResidualAdd(self: *Nvfp4PhaseBudgetCounters, elapsed_ns: u64) void {
        self.residual_add_calls = saturatingAddU64(self.residual_add_calls, 1);
        self.residual_add_ns = saturatingAddU64(self.residual_add_ns, elapsed_ns);
    }

    pub fn recordQkv(self: *Nvfp4PhaseBudgetCounters, path: ProjectionPath) void {
        self.qkv_calls = saturatingAddU64(self.qkv_calls, 1);
        switch (path) {
            .fused => self.qkv_fused_calls = saturatingAddU64(self.qkv_fused_calls, 1),
            .unfused => self.qkv_unfused_calls = saturatingAddU64(self.qkv_unfused_calls, 1),
        }
    }

    pub fn recordGateUp(self: *Nvfp4PhaseBudgetCounters, path: ProjectionPath) void {
        self.gate_up_calls = saturatingAddU64(self.gate_up_calls, 1);
        switch (path) {
            .fused => self.gate_up_fused_calls = saturatingAddU64(self.gate_up_fused_calls, 1),
            .unfused => self.gate_up_unfused_calls = saturatingAddU64(self.gate_up_unfused_calls, 1),
        }
    }

    pub fn knownNs(self: *const Nvfp4PhaseBudgetCounters) u64 {
        const total_u128 = @as(u128, self.linear_ns) +
            @as(u128, self.attention_ns) +
            @as(u128, self.layer_scalar_ns) +
            @as(u128, self.rmsnorm_ns) +
            @as(u128, self.residual_add_ns);
        return saturatingU64FromU128(total_u128);
    }

    /// Approximate tensor-core attention time for current routing.
    ///
    /// Contract note:
    /// `heads_f16_kv` is the GEMM-based f16 route used by the current
    /// tensor-core-oriented attention implementation.
    /// `heads_lowbit_bridge_f16_kv` is also GEMM-based (`1-byte KV -> f16`).
    /// Custom fused kernels (`fused_heads_*`) are intentionally tracked in
    /// separate buckets and are not counted as tensor-core here.
    pub fn attentionTensorCoreNsApprox(self: *const Nvfp4PhaseBudgetCounters) u64 {
        const total_u128 = @as(u128, self.attention_heads_f16_kv_ns) +
            @as(u128, self.attention_heads_lowbit_bridge_f16_kv_ns);
        return saturatingU64FromU128(total_u128);
    }

    /// Approximate scalar/non-GEMM attention time.
    ///
    /// Contract note:
    /// includes i8/fp8 prefill/decode attention paths and f32 heads path.
    pub fn attentionScalarNsApprox(self: *const Nvfp4PhaseBudgetCounters) u64 {
        const total_u128 = @as(u128, self.attention_fused_heads_i8_kv_ns) +
            @as(u128, self.attention_heads_i8_kv_ns) +
            @as(u128, self.attention_fused_heads_fp8_kv_ns) +
            @as(u128, self.attention_heads_fp8_kv_ns) +
            @as(u128, self.attention_heads_f32_kv_ns);
        return saturatingU64FromU128(total_u128);
    }

    /// Custom fused f16 attention kernels tracked separately from GEMM f16.
    pub fn attentionCustomF16Ns(self: *const Nvfp4PhaseBudgetCounters) u64 {
        return self.attention_fused_heads_f16_kv_ns;
    }

    pub fn delta(current: Nvfp4PhaseBudgetCounters, start: Nvfp4PhaseBudgetCounters) Nvfp4PhaseBudgetCounters {
        return .{
            .linear_calls = saturatingSub(current.linear_calls, start.linear_calls),
            .linear_ns = saturatingSub(current.linear_ns, start.linear_ns),
            .attention_calls = saturatingSub(current.attention_calls, start.attention_calls),
            .attention_ns = saturatingSub(current.attention_ns, start.attention_ns),
            .attention_causal_calls = saturatingSub(current.attention_causal_calls, start.attention_causal_calls),
            .attention_noncausal_calls = saturatingSub(current.attention_noncausal_calls, start.attention_noncausal_calls),
            .attention_context_calls = saturatingSub(current.attention_context_calls, start.attention_context_calls),
            .attention_batched_prefill_calls = saturatingSub(current.attention_batched_prefill_calls, start.attention_batched_prefill_calls),
            .layer_scalar_calls = saturatingSub(current.layer_scalar_calls, start.layer_scalar_calls),
            .layer_scalar_ns = saturatingSub(current.layer_scalar_ns, start.layer_scalar_ns),
            .rmsnorm_calls = saturatingSub(current.rmsnorm_calls, start.rmsnorm_calls),
            .rmsnorm_ns = saturatingSub(current.rmsnorm_ns, start.rmsnorm_ns),
            .residual_add_calls = saturatingSub(current.residual_add_calls, start.residual_add_calls),
            .residual_add_ns = saturatingSub(current.residual_add_ns, start.residual_add_ns),
            .qkv_calls = saturatingSub(current.qkv_calls, start.qkv_calls),
            .qkv_fused_calls = saturatingSub(current.qkv_fused_calls, start.qkv_fused_calls),
            .qkv_unfused_calls = saturatingSub(current.qkv_unfused_calls, start.qkv_unfused_calls),
            .gate_up_calls = saturatingSub(current.gate_up_calls, start.gate_up_calls),
            .gate_up_fused_calls = saturatingSub(current.gate_up_fused_calls, start.gate_up_fused_calls),
            .gate_up_unfused_calls = saturatingSub(current.gate_up_unfused_calls, start.gate_up_unfused_calls),
            .attention_fused_heads_f16_kv = saturatingSub(current.attention_fused_heads_f16_kv, start.attention_fused_heads_f16_kv),
            .attention_heads_f16_kv = saturatingSub(current.attention_heads_f16_kv, start.attention_heads_f16_kv),
            .attention_heads_lowbit_bridge_f16_kv = saturatingSub(current.attention_heads_lowbit_bridge_f16_kv, start.attention_heads_lowbit_bridge_f16_kv),
            .attention_fused_heads_i8_kv = saturatingSub(current.attention_fused_heads_i8_kv, start.attention_fused_heads_i8_kv),
            .attention_heads_i8_kv = saturatingSub(current.attention_heads_i8_kv, start.attention_heads_i8_kv),
            .attention_fused_heads_fp8_kv = saturatingSub(current.attention_fused_heads_fp8_kv, start.attention_fused_heads_fp8_kv),
            .attention_heads_fp8_kv = saturatingSub(current.attention_heads_fp8_kv, start.attention_heads_fp8_kv),
            .attention_heads_f32_kv = saturatingSub(current.attention_heads_f32_kv, start.attention_heads_f32_kv),
            .attention_fused_heads_f16_kv_ns = saturatingSub(current.attention_fused_heads_f16_kv_ns, start.attention_fused_heads_f16_kv_ns),
            .attention_heads_f16_kv_ns = saturatingSub(current.attention_heads_f16_kv_ns, start.attention_heads_f16_kv_ns),
            .attention_heads_lowbit_bridge_f16_kv_ns = saturatingSub(current.attention_heads_lowbit_bridge_f16_kv_ns, start.attention_heads_lowbit_bridge_f16_kv_ns),
            .attention_fused_heads_i8_kv_ns = saturatingSub(current.attention_fused_heads_i8_kv_ns, start.attention_fused_heads_i8_kv_ns),
            .attention_heads_i8_kv_ns = saturatingSub(current.attention_heads_i8_kv_ns, start.attention_heads_i8_kv_ns),
            .attention_fused_heads_fp8_kv_ns = saturatingSub(current.attention_fused_heads_fp8_kv_ns, start.attention_fused_heads_fp8_kv_ns),
            .attention_heads_fp8_kv_ns = saturatingSub(current.attention_heads_fp8_kv_ns, start.attention_heads_fp8_kv_ns),
            .attention_heads_f32_kv_ns = saturatingSub(current.attention_heads_f32_kv_ns, start.attention_heads_f32_kv_ns),
        };
    }
};

/// Runtime-selected attention route classification used by budget counters.
///
/// Naming contract:
/// - `heads_*` typically denotes the GEMM/heads-family route.
/// - `fused_heads_*` denotes custom fused kernel families.
/// - Route names encode storage dtype, not necessarily compute dtype.
pub const AttentionPath = enum {
    /// Custom fused f16 kernel family (non-GEMM bucket).
    fused_heads_f16_kv,
    /// GEMM-based f16 heads path (tensor-core-oriented bucket).
    heads_f16_kv,
    /// Low-bit KV prefill bridge route: dequant to f16 + GEMM f16 heads.
    heads_lowbit_bridge_f16_kv,
    /// Custom fused i8 KV kernel family.
    fused_heads_i8_kv,
    /// Heads-family i8 KV path (currently non-GEMM/scalar-style kernels).
    heads_i8_kv,
    /// Custom fused fp8 KV kernel family.
    fused_heads_fp8_kv,
    /// Heads-family fp8 KV path (currently non-GEMM/scalar-style kernels).
    heads_fp8_kv,
    /// Legacy f32 heads path.
    heads_f32_kv,
};

pub const KvCacheStorageMode = enum(u8) {
    device,
};

pub fn resolveCudaKvStorageMode() KvCacheStorageMode {
    const raw = std.process.getEnvVarOwned(std.heap.c_allocator, "TALU_CUDA_KV_STORAGE") catch {
        return .device;
    };
    defer std.heap.c_allocator.free(raw);
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    if (std.ascii.eqlIgnoreCase(trimmed, "device")) return .device;
    log.warn("inference", "Invalid TALU_CUDA_KV_STORAGE; using default", .{
        .value = trimmed,
        .default = "device",
    });
    return .device;
}

pub const AttentionKernelSet = struct {
    attn_scores_heads_f32_function: ?compute.cuda.Function = null,
    attn_weighted_sum_heads_f32_function: ?compute.cuda.Function = null,
    attn_scores_heads_f16_kv_function: ?compute.cuda.Function = null,
    softmax_rows_function: ?compute.cuda.Function = null,
    attn_weighted_sum_heads_f16_kv_function: ?compute.cuda.Function = null,
    attn_fused_heads_f16_kv_function: ?compute.cuda.Function = null,
    attn_fused_prefill_heads_f16_kv_function: ?compute.cuda.Function = null,
    attn_fused_prefill_heads_f16_kv_gqa_function: ?compute.cuda.Function = null,
    causal_attn_softmax_f32_function: ?compute.cuda.Function = null,
    attn_scores_heads_i8_kv_function: ?compute.cuda.Function = null,
    attn_weighted_sum_heads_i8_kv_function: ?compute.cuda.Function = null,
    attn_fused_heads_i8_kv_function: ?compute.cuda.Function = null,
    attn_fused_prefill_heads_i8_kv_function: ?compute.cuda.Function = null,
    attn_fused_prefill_heads_i8_kv_gqa_function: ?compute.cuda.Function = null,
    attn_scores_heads_fp8_kv_function: ?compute.cuda.Function = null,
    attn_weighted_sum_heads_fp8_kv_function: ?compute.cuda.Function = null,
    attn_fused_heads_fp8_kv_function: ?compute.cuda.Function = null,
    attn_fused_prefill_heads_fp8_kv_function: ?compute.cuda.Function = null,
    attn_fused_prefill_heads_fp8_kv_gqa_function: ?compute.cuda.Function = null,
};

pub const required_kernels = [_]RequiredKernel{
    .{ .slot = .vector_add, .op_name = compute.cuda.vector_add.op_name, .embedded_symbol = compute.cuda.vector_add.embedded_symbol },
    .{ .slot = .vector_add_scaled, .op_name = compute.cuda.vector_add_scaled.op_name, .embedded_symbol = compute.cuda.vector_add_scaled.embedded_symbol },
    .{ .slot = .vector_add_rows_strided, .op_name = compute.cuda.vector_add_rows_strided.op_name, .embedded_symbol = compute.cuda.vector_add_rows_strided.embedded_symbol },
    .{ .slot = .vector_add_scaled_rows_strided, .op_name = compute.cuda.vector_add_scaled_rows_strided.op_name, .embedded_symbol = compute.cuda.vector_add_scaled_rows_strided.embedded_symbol },
    .{ .slot = .mul, .op_name = compute.cuda.mul.op_name, .embedded_symbol = compute.cuda.mul.embedded_symbol },
    .{ .slot = .copy, .op_name = compute.cuda.copy.op_name, .embedded_symbol = compute.cuda.copy.embedded_symbol },
    .{ .slot = .copy_u16, .op_name = compute.cuda.copy_u16.op_name, .embedded_symbol = compute.cuda.copy_u16.embedded_symbol },
    .{ .slot = .cast_f32_to_f16, .op_name = compute.cuda.cast_f32_to_f16.op_name, .embedded_symbol = compute.cuda.cast_f32_to_f16.embedded_symbol },
    .{ .slot = .cast_f32_to_bf16, .op_name = compute.cuda.cast_f32_to_bf16.op_name, .embedded_symbol = compute.cuda.cast_f32_to_bf16.embedded_symbol },
    .{ .slot = .cast_bf16_to_f32, .op_name = compute.cuda.cast_bf16_to_f32.op_name, .embedded_symbol = compute.cuda.cast_bf16_to_f32.embedded_symbol },
    .{ .slot = .embedding_lookup_f32, .op_name = compute.cuda.embedding_lookup_f32.op_name, .embedded_symbol = compute.cuda.embedding_lookup_f32.embedded_symbol },
    .{ .slot = .embedding_lookup_u16, .op_name = compute.cuda.embedding_lookup_u16.op_name, .embedded_symbol = compute.cuda.embedding_lookup_u16.embedded_symbol },
    .{ .slot = .embedding_lookup_u16_rows, .op_name = compute.cuda.embedding_lookup_u16_rows.op_name, .embedded_symbol = compute.cuda.embedding_lookup_u16_rows.embedded_symbol },
    .{ .slot = .embedding_lookup_gaffine_u4, .op_name = compute.cuda.embedding_lookup_gaffine_u4.op_name, .embedded_symbol = compute.cuda.embedding_lookup_gaffine_u4.embedded_symbol },
    .{ .slot = .kv_write_f16, .op_name = compute.cuda.kv_write_f16.op_name, .embedded_symbol = compute.cuda.kv_write_f16.embedded_symbol },
    .{ .slot = .kv_write_f16_rows, .op_name = compute.cuda.kv_write_f16_rows.op_name, .embedded_symbol = compute.cuda.kv_write_f16_rows.embedded_symbol },
    .{ .slot = .kv_write_f16_rows_ptrs, .op_name = compute.cuda.kv_write_f16_rows_ptrs.op_name, .embedded_symbol = compute.cuda.kv_write_f16_rows_ptrs.embedded_symbol },
    .{ .slot = .rmsnorm, .op_name = compute.cuda.rmsnorm.op_name, .embedded_symbol = compute.cuda.rmsnorm.embedded_symbol },
    .{ .slot = .rmsnorm_rows_strided, .op_name = compute.cuda.rmsnorm_rows_strided.op_name, .embedded_symbol = compute.cuda.rmsnorm_rows_strided.embedded_symbol },
    .{ .slot = .rope, .op_name = compute.cuda.rope.op_name, .embedded_symbol = compute.cuda.rope.embedded_symbol },
    .{ .slot = .rope_store_f16, .op_name = compute.cuda.rope_store_f16.op_name, .embedded_symbol = compute.cuda.rope_store_f16.embedded_symbol },
    .{ .slot = .attn_scores_heads_f32, .op_name = compute.cuda.attn_scores_heads_f32.op_name, .embedded_symbol = compute.cuda.attn_scores_heads_f32.embedded_symbol },
    .{ .slot = .attn_scores_heads_f16_kv, .op_name = compute.cuda.attn_scores_heads_f16_kv.op_name, .embedded_symbol = compute.cuda.attn_scores_heads_f16_kv.embedded_symbol },
    .{ .slot = .attn_fused_heads_f16_kv, .op_name = compute.cuda.attn_fused_heads_f16_kv.op_name, .embedded_symbol = compute.cuda.attn_fused_heads_f16_kv.embedded_symbol },
    .{ .slot = .attn_fused_decode_heads_f16_kv_ptrs, .op_name = compute.cuda.attn_fused_decode_heads_f16_kv_ptrs.op_name, .embedded_symbol = compute.cuda.attn_fused_decode_heads_f16_kv_ptrs.embedded_symbol },
    .{ .slot = .attn_fused_prefill_heads_f16_kv, .op_name = compute.cuda.attn_fused_prefill_heads_f16_kv.op_name, .embedded_symbol = compute.cuda.attn_fused_prefill_heads_f16_kv.embedded_symbol },
    .{ .slot = .attn_fused_prefill_heads_f16_kv_gqa, .op_name = compute.cuda.attn_fused_prefill_heads_f16_kv_gqa.op_name, .embedded_symbol = compute.cuda.attn_fused_prefill_heads_f16_kv_gqa.embedded_symbol },
    .{ .slot = .causal_attn_softmax_f32, .op_name = compute.cuda.causal_attn_softmax_f32.op_name, .embedded_symbol = compute.cuda.causal_attn_softmax_f32.embedded_symbol },
    .{ .slot = .softmax_rows, .op_name = compute.cuda.softmax_rows.op_name, .embedded_symbol = compute.cuda.softmax_rows.embedded_symbol },
    .{ .slot = .attn_weighted_sum_heads_f32, .op_name = compute.cuda.attn_weighted_sum_heads_f32.op_name, .embedded_symbol = compute.cuda.attn_weighted_sum_heads_f32.embedded_symbol },
    .{ .slot = .attn_weighted_sum_heads_f16_kv, .op_name = compute.cuda.attn_weighted_sum_heads_f16_kv.op_name, .embedded_symbol = compute.cuda.attn_weighted_sum_heads_f16_kv.embedded_symbol },
    .{ .slot = .silu, .op_name = compute.cuda.silu.op_name, .embedded_symbol = compute.cuda.silu.embedded_symbol },
    .{ .slot = .silu_mul, .op_name = compute.cuda.silu_mul.op_name, .embedded_symbol = compute.cuda.silu_mul.embedded_symbol },
    .{ .slot = .gelu_mul, .op_name = compute.cuda.gelu_mul.op_name, .embedded_symbol = compute.cuda.gelu_mul.embedded_symbol },
    .{ .slot = .shortconv_step, .op_name = compute.cuda.shortconv_step.op_name, .embedded_symbol = compute.cuda.shortconv_step.embedded_symbol },
    .{ .slot = .gated_attention_compact_q, .op_name = compute.cuda.gated_attention_compact_q.op_name, .embedded_symbol = compute.cuda.gated_attention_compact_q.embedded_symbol },
    .{ .slot = .gated_attention_output_gate, .op_name = compute.cuda.gated_attention_output_gate.op_name, .embedded_symbol = compute.cuda.gated_attention_output_gate.embedded_symbol },
    .{ .slot = .gated_delta_conv, .op_name = compute.cuda.gated_delta_conv.op_name, .embedded_symbol = compute.cuda.gated_delta_conv.embedded_symbol },
    .{ .slot = .gated_delta_conv_silu, .op_name = compute.cuda.gated_delta_conv_silu.op_name, .embedded_symbol = compute.cuda.gated_delta_conv_silu.embedded_symbol },
    .{ .slot = .gated_delta_conv_silu_rows, .op_name = compute.cuda.gated_delta_conv_silu_rows.op_name, .embedded_symbol = compute.cuda.gated_delta_conv_silu_rows.embedded_symbol },
    .{ .slot = .gated_delta_conv_silu_rows_ptrs, .op_name = compute.cuda.gated_delta_conv_silu_rows_ptrs.op_name, .embedded_symbol = compute.cuda.gated_delta_conv_silu_rows_ptrs.embedded_symbol },
    .{ .slot = .gated_delta_advance_ring_heads, .op_name = compute.cuda.gated_delta_conv_silu_rows_ptrs.op_name_advance, .embedded_symbol = compute.cuda.gated_delta_conv_silu_rows_ptrs.embedded_symbol_advance },
    .{ .slot = .gated_delta_qk_norm, .op_name = compute.cuda.gated_delta_qk_norm.op_name, .embedded_symbol = compute.cuda.gated_delta_qk_norm.embedded_symbol },
    .{ .slot = .gated_delta_ssm, .op_name = compute.cuda.gated_delta_ssm.op_name, .embedded_symbol = compute.cuda.gated_delta_ssm.embedded_symbol },
    .{ .slot = .gated_delta_ssm_rows, .op_name = compute.cuda.gated_delta_ssm_rows.op_name, .embedded_symbol = compute.cuda.gated_delta_ssm_rows.embedded_symbol },
    .{ .slot = .gated_delta_ssm_rows_ptrs, .op_name = compute.cuda.gated_delta_ssm_rows_ptrs.op_name, .embedded_symbol = compute.cuda.gated_delta_ssm_rows_ptrs.embedded_symbol },
    .{ .slot = .gated_delta_ssm_rows_i8, .op_name = compute.cuda.gated_delta_ssm_rows_i8.op_name, .embedded_symbol = compute.cuda.gated_delta_ssm_rows_i8.embedded_symbol },
    .{ .slot = .gated_delta_ssm_rows_ptrs_i8, .op_name = compute.cuda.gated_delta_ssm_rows_ptrs_i8.op_name, .embedded_symbol = compute.cuda.gated_delta_ssm_rows_ptrs_i8.embedded_symbol },
    .{ .slot = .gated_delta_rmsnorm_silu_mul, .op_name = compute.cuda.gated_delta_rmsnorm_silu_mul.op_name, .embedded_symbol = compute.cuda.gated_delta_rmsnorm_silu_mul.embedded_symbol },
    .{ .slot = .gated_delta_rmsnorm_silu_mul_rows, .op_name = compute.cuda.gated_delta_rmsnorm_silu_mul_rows.op_name, .embedded_symbol = compute.cuda.gated_delta_rmsnorm_silu_mul_rows.embedded_symbol },
    .{ .slot = .argmax, .op_name = compute.cuda.argmax.op_name, .embedded_symbol = compute.cuda.argmax.embedded_symbol },
    .{ .slot = .matmul_f16, .op_name = compute.cuda.matmul_u16.op_name_f16, .embedded_symbol = compute.cuda.matmul_u16.embedded_symbol_f16 },
    .{ .slot = .matmul_bf16, .op_name = compute.cuda.matmul_u16.op_name_bf16, .embedded_symbol = compute.cuda.matmul_u16.embedded_symbol_bf16 },
    .{ .slot = .matvec_f16, .op_name = compute.cuda.matvec_u16.op_name_f16, .embedded_symbol = compute.cuda.matvec_u16.embedded_symbol_f16 },
    .{ .slot = .matvec_bf16, .op_name = compute.cuda.matvec_u16.op_name_bf16, .embedded_symbol = compute.cuda.matvec_u16.embedded_symbol_bf16 },
    .{ .slot = .matvec_gate_up_f16, .op_name = compute.cuda.matvec_u16_gate_up.op_name_f16, .embedded_symbol = compute.cuda.matvec_u16_gate_up.embedded_symbol_f16 },
    .{ .slot = .matvec_gate_up_bf16, .op_name = compute.cuda.matvec_u16_gate_up.op_name_bf16, .embedded_symbol = compute.cuda.matvec_u16_gate_up.embedded_symbol_bf16 },
    .{ .slot = .matvec_gate_up_silu_f16, .op_name = compute.cuda.matvec_u16_gate_up_silu.op_name_f16, .embedded_symbol = compute.cuda.matvec_u16_gate_up_silu.embedded_symbol_f16 },
    .{ .slot = .matvec_gate_up_silu_bf16, .op_name = compute.cuda.matvec_u16_gate_up_silu.op_name_bf16, .embedded_symbol = compute.cuda.matvec_u16_gate_up_silu.embedded_symbol_bf16 },
    .{ .slot = .matvec_qkv_f16, .op_name = compute.cuda.matvec_u16_qkv.op_name_f16, .embedded_symbol = compute.cuda.matvec_u16_qkv.embedded_symbol_f16 },
    .{ .slot = .matvec_qkv_bf16, .op_name = compute.cuda.matvec_u16_qkv.op_name_bf16, .embedded_symbol = compute.cuda.matvec_u16_qkv.embedded_symbol_bf16 },
    .{ .slot = .gaffine_u4_matvec, .op_name = compute.cuda.gaffine_u4_matvec.op_name, .embedded_symbol = compute.cuda.gaffine_u4_matvec.embedded_symbol },
    .{ .slot = .gaffine_u4_matvec_tile8, .op_name = compute.cuda.gaffine_u4_matvec.op_name_tile8, .embedded_symbol = compute.cuda.gaffine_u4_matvec.embedded_symbol_tile8 },
    .{ .slot = .gaffine_u8_matvec, .op_name = compute.cuda.gaffine_u8_matvec.op_name, .embedded_symbol = compute.cuda.gaffine_u8_matvec.embedded_symbol },
    .{ .slot = .gaffine_u4_matvec_gate_up, .op_name = compute.cuda.gaffine_u4_matvec_gate_up.op_name, .embedded_symbol = compute.cuda.gaffine_u4_matvec_gate_up.embedded_symbol },
    .{ .slot = .gaffine_u4_matvec_qkv, .op_name = compute.cuda.gaffine_u4_matvec_qkv.op_name, .embedded_symbol = compute.cuda.gaffine_u4_matvec_qkv.embedded_symbol },
    .{ .slot = .gaffine_u4_matvec_qkv_tile8, .op_name = compute.cuda.gaffine_u4_matvec_qkv.op_name_tile8, .embedded_symbol = compute.cuda.gaffine_u4_matvec_qkv.embedded_symbol_tile8 },
    .{ .slot = .gaffine_u8_matvec_qkv, .op_name = compute.cuda.gaffine_u8_matvec_qkv.op_name, .embedded_symbol = compute.cuda.gaffine_u8_matvec_qkv.embedded_symbol },
    .{ .slot = .gaffine_u8_matvec_gate_up, .op_name = compute.cuda.gaffine_u8_matvec_gate_up.op_name, .embedded_symbol = compute.cuda.gaffine_u8_matvec_gate_up.embedded_symbol },
    .{ .slot = .gaffine_u4_matvec_gate_up_silu, .op_name = compute.cuda.gaffine_u4_matvec_gate_up_silu.op_name, .embedded_symbol = compute.cuda.gaffine_u4_matvec_gate_up_silu.embedded_symbol },
    .{ .slot = .gaffine_u4_matvec_gate_up_silu_tile8, .op_name = compute.cuda.gaffine_u4_matvec_gate_up_silu.op_name_tile8, .embedded_symbol = compute.cuda.gaffine_u4_matvec_gate_up_silu.embedded_symbol_tile8 },
    .{ .slot = .gaffine_u8_matvec_gate_up_silu, .op_name = compute.cuda.gaffine_u8_matvec_gate_up_silu.op_name, .embedded_symbol = compute.cuda.gaffine_u8_matvec_gate_up_silu.embedded_symbol },
    .{ .slot = .gaffine_u4_dequant_f16, .op_name = compute.cuda.gaffine_u4_dequantize_f16.op_name, .embedded_symbol = compute.cuda.gaffine_u4_dequantize_f16.embedded_symbol },
    .{ .slot = .gaffine_u8_dequant_f16, .op_name = compute.cuda.gaffine_u8_dequantize_f16.op_name, .embedded_symbol = compute.cuda.gaffine_u8_dequantize_f16.embedded_symbol },
    .{ .slot = .rope_rows_ptrs, .op_name = compute.cuda.rope_rows_ptrs.op_name, .embedded_symbol = compute.cuda.rope_rows_ptrs.embedded_symbol },
    .{ .slot = .attn_scores_heads_f16_kv_ptrs, .op_name = compute.cuda.attn_scores_heads_f16_kv_ptrs.op_name, .embedded_symbol = compute.cuda.attn_scores_heads_f16_kv_ptrs.embedded_symbol },
    .{ .slot = .softmax_rows_dynamic_cols_ptrs, .op_name = compute.cuda.softmax_rows_dynamic_cols_ptrs.op_name, .embedded_symbol = compute.cuda.softmax_rows_dynamic_cols_ptrs.embedded_symbol },
    .{ .slot = .attn_weighted_sum_heads_f16_kv_ptrs, .op_name = compute.cuda.attn_weighted_sum_heads_f16_kv_ptrs.op_name, .embedded_symbol = compute.cuda.attn_weighted_sum_heads_f16_kv_ptrs.embedded_symbol },
    .{ .slot = .kv_write_i8, .op_name = compute.cuda.kv_write_i8.op_name, .embedded_symbol = compute.cuda.kv_write_i8.embedded_symbol },
    .{ .slot = .kv_write_i8_rows, .op_name = compute.cuda.kv_write_i8_rows.op_name, .embedded_symbol = compute.cuda.kv_write_i8_rows.embedded_symbol },
    .{ .slot = .kv_write_i8_rows_ptrs, .op_name = compute.cuda.kv_write_i8_rows_ptrs.op_name, .embedded_symbol = compute.cuda.kv_write_i8_rows_ptrs.embedded_symbol },
    .{ .slot = .dequant_kv_i8_to_f16, .op_name = compute.cuda.dequant_kv_i8_to_f16.op_name, .embedded_symbol = compute.cuda.dequant_kv_i8_to_f16.embedded_symbol },
    .{ .slot = .rope_store_i8, .op_name = compute.cuda.rope_store_i8.op_name, .embedded_symbol = compute.cuda.rope_store_i8.embedded_symbol },
    .{ .slot = .attn_scores_heads_i8_kv, .op_name = compute.cuda.attn_scores_heads_i8_kv.op_name, .embedded_symbol = compute.cuda.attn_scores_heads_i8_kv.embedded_symbol },
    .{ .slot = .attn_weighted_sum_heads_i8_kv, .op_name = compute.cuda.attn_weighted_sum_heads_i8_kv.op_name, .embedded_symbol = compute.cuda.attn_weighted_sum_heads_i8_kv.embedded_symbol },
    .{ .slot = .attn_fused_heads_i8_kv, .op_name = compute.cuda.attn_fused_heads_i8_kv.op_name, .embedded_symbol = compute.cuda.attn_fused_heads_i8_kv.embedded_symbol },
    .{ .slot = .attn_fused_decode_heads_i8_kv_ptrs, .op_name = compute.cuda.attn_fused_decode_heads_i8_kv_ptrs.op_name, .embedded_symbol = compute.cuda.attn_fused_decode_heads_i8_kv_ptrs.embedded_symbol },
    .{ .slot = .attn_fused_prefill_heads_i8_kv, .op_name = compute.cuda.attn_fused_prefill_heads_i8_kv.op_name, .embedded_symbol = compute.cuda.attn_fused_prefill_heads_i8_kv.embedded_symbol },
    .{ .slot = .attn_fused_prefill_heads_i8_kv_gqa, .op_name = compute.cuda.attn_fused_prefill_heads_i8_kv_gqa.op_name, .embedded_symbol = compute.cuda.attn_fused_prefill_heads_i8_kv_gqa.embedded_symbol },
    .{ .slot = .attn_scores_heads_i8_kv_ptrs, .op_name = compute.cuda.attn_scores_heads_i8_kv_ptrs.op_name, .embedded_symbol = compute.cuda.attn_scores_heads_i8_kv_ptrs.embedded_symbol },
    .{ .slot = .attn_weighted_sum_heads_i8_kv_ptrs, .op_name = compute.cuda.attn_weighted_sum_heads_i8_kv_ptrs.op_name, .embedded_symbol = compute.cuda.attn_weighted_sum_heads_i8_kv_ptrs.embedded_symbol },
    .{ .slot = .kv_write_fp8, .op_name = compute.cuda.kv_write_fp8.op_name, .embedded_symbol = compute.cuda.kv_write_fp8.embedded_symbol },
    .{ .slot = .kv_write_fp8_rows, .op_name = compute.cuda.kv_write_fp8_rows.op_name, .embedded_symbol = compute.cuda.kv_write_fp8_rows.embedded_symbol },
    .{ .slot = .kv_write_fp8_rows_ptrs, .op_name = compute.cuda.kv_write_fp8_rows_ptrs.op_name, .embedded_symbol = compute.cuda.kv_write_fp8_rows_ptrs.embedded_symbol },
    .{ .slot = .dequant_kv_fp8_to_f16, .op_name = compute.cuda.dequant_kv_fp8_to_f16.op_name, .embedded_symbol = compute.cuda.dequant_kv_fp8_to_f16.embedded_symbol },
    .{ .slot = .rope_store_fp8, .op_name = compute.cuda.rope_store_fp8.op_name, .embedded_symbol = compute.cuda.rope_store_fp8.embedded_symbol },
    .{ .slot = .attn_scores_heads_fp8_kv, .op_name = compute.cuda.attn_scores_heads_fp8_kv.op_name, .embedded_symbol = compute.cuda.attn_scores_heads_fp8_kv.embedded_symbol },
    .{ .slot = .attn_weighted_sum_heads_fp8_kv, .op_name = compute.cuda.attn_weighted_sum_heads_fp8_kv.op_name, .embedded_symbol = compute.cuda.attn_weighted_sum_heads_fp8_kv.embedded_symbol },
    .{ .slot = .attn_fused_heads_fp8_kv, .op_name = compute.cuda.attn_fused_heads_fp8_kv.op_name, .embedded_symbol = compute.cuda.attn_fused_heads_fp8_kv.embedded_symbol },
    .{ .slot = .attn_fused_decode_heads_fp8_kv_ptrs, .op_name = compute.cuda.attn_fused_decode_heads_fp8_kv_ptrs.op_name, .embedded_symbol = compute.cuda.attn_fused_decode_heads_fp8_kv_ptrs.embedded_symbol },
    .{ .slot = .attn_fused_prefill_heads_fp8_kv, .op_name = compute.cuda.attn_fused_prefill_heads_fp8_kv.op_name, .embedded_symbol = compute.cuda.attn_fused_prefill_heads_fp8_kv.embedded_symbol },
    .{ .slot = .attn_fused_prefill_heads_fp8_kv_gqa, .op_name = compute.cuda.attn_fused_prefill_heads_fp8_kv_gqa.op_name, .embedded_symbol = compute.cuda.attn_fused_prefill_heads_fp8_kv_gqa.embedded_symbol },
    .{ .slot = .attn_scores_heads_fp8_kv_ptrs, .op_name = compute.cuda.attn_scores_heads_fp8_kv_ptrs.op_name, .embedded_symbol = compute.cuda.attn_scores_heads_fp8_kv_ptrs.embedded_symbol },
    .{ .slot = .attn_weighted_sum_heads_fp8_kv_ptrs, .op_name = compute.cuda.attn_weighted_sum_heads_fp8_kv_ptrs.op_name, .embedded_symbol = compute.cuda.attn_weighted_sum_heads_fp8_kv_ptrs.embedded_symbol },
    .{ .slot = .flash_decode_f16, .op_name = compute.cuda.flash_decode.op_name_f16, .embedded_symbol = compute.cuda.flash_decode.symbol_f16 },
    .{ .slot = .flash_decode_i8, .op_name = compute.cuda.flash_decode.op_name_i8, .embedded_symbol = compute.cuda.flash_decode.symbol_i8 },
    .{ .slot = .flash_decode_fp8, .op_name = compute.cuda.flash_decode.op_name_fp8, .embedded_symbol = compute.cuda.flash_decode.symbol_fp8 },
    .{ .slot = .flash_decode_reduce, .op_name = compute.cuda.flash_decode.op_name_reduce, .embedded_symbol = compute.cuda.flash_decode.symbol_reduce },
    .{ .slot = .flash_prefill_f16, .op_name = compute.cuda.flash_prefill.op_name_f16, .embedded_symbol = compute.cuda.flash_prefill.symbol_f16 },
    .{ .slot = .flash_prefill_i8, .op_name = compute.cuda.flash_prefill.op_name_i8, .embedded_symbol = compute.cuda.flash_prefill.symbol_i8 },
    .{ .slot = .flash_prefill_fp8, .op_name = compute.cuda.flash_prefill.op_name_fp8, .embedded_symbol = compute.cuda.flash_prefill.symbol_fp8 },
};

pub const DeviceTensor = struct {
    rows: usize,
    cols: usize,
    buffer: compute.cuda.Buffer,

    pub fn deinit(self: *DeviceTensor, device: *compute.cuda.Device) void {
        self.buffer.deinit(device);
    }

    pub fn byteSize(self: *const DeviceTensor) usize {
        return self.buffer.size;
    }
};

pub const missing_device_tensor: DeviceTensor = std.mem.zeroes(DeviceTensor);
pub const missing_host_tensor: Tensor = std.mem.zeroes(Tensor);

pub const EmbeddingLookup = struct {
    kind: EmbeddingLookupKind,
    dim0: u32,
    dim1: u32,
    hidden_dim: u32,
    layout_tag: u32,
    group_size: u32 = 0,
    scales_dtype_tag: u32 = 0,
    scales: ?compute.cuda.Buffer = null,
    biases: ?compute.cuda.Buffer = null,
    multiplier: f32,
    buffer: compute.cuda.Buffer,

    pub fn deinit(self: *EmbeddingLookup, device: *compute.cuda.Device) void {
        if (self.biases) |*buf| buf.deinit(device);
        if (self.scales) |*buf| buf.deinit(device);
        self.buffer.deinit(device);
    }

    pub fn byteSize(self: *const EmbeddingLookup) usize {
        return self.buffer.size +
            (if (self.scales) |buf| buf.size else 0) +
            (if (self.biases) |buf| buf.size else 0);
    }
};

pub const GaffineU4LinearWeight = struct {
    rows: usize,
    cols: usize,
    packed_data: compute.cuda.Buffer,
    scales: compute.cuda.Buffer,
    biases: compute.cuda.Buffer,
    group_size: u32,
    scales_dtype_tag: u32,
    dequant_f16_cache: compute.cuda.Buffer = .{ .pointer = 0, .size = 0 },
    dequant_i8_cache: compute.cuda.Buffer = .{ .pointer = 0, .size = 0 },
    mean_scale_cache: compute.cuda.Buffer = .{ .pointer = 0, .size = 0 },

    pub fn deinit(self: *GaffineU4LinearWeight, device: *compute.cuda.Device) void {
        if (self.mean_scale_cache.pointer != 0) self.mean_scale_cache.deinit(device);
        if (self.dequant_i8_cache.pointer != 0) self.dequant_i8_cache.deinit(device);
        if (self.dequant_f16_cache.pointer != 0) self.dequant_f16_cache.deinit(device);
        self.biases.deinit(device);
        self.scales.deinit(device);
        self.packed_data.deinit(device);
    }

    pub fn byteSize(self: *const GaffineU4LinearWeight) usize {
        return self.packed_data.size + self.scales.size + self.biases.size +
            self.dequant_f16_cache.size + self.dequant_i8_cache.size + self.mean_scale_cache.size;
    }
};

pub const GaffineU8LinearWeight = GaffineU4LinearWeight;

pub const U16LinearWeight = struct {
    rows: usize,
    cols: usize,
    buffer: compute.cuda.Buffer,
    dtype: DenseU16Dtype,

    pub fn deinit(self: *U16LinearWeight, device: *compute.cuda.Device) void {
        self.buffer.deinit(device);
    }

    pub fn byteSize(self: *const U16LinearWeight) usize {
        return self.buffer.size;
    }
};

pub const Fp8LinearWeight = struct {
    rows: usize,
    cols: usize,
    buffer: compute.cuda.Buffer,
    /// GPU buffer holding BF16 per-block scales [scale_rows × scale_cols]
    scales_buffer: compute.cuda.Buffer,
    scale_rows: u32,
    scale_cols: u32,
    block_size: u32,
    /// Per-tensor scale (used only when scale_rows == 1 && scale_cols == 1)
    weight_scale_inv: f32,

    pub fn deinit(self: *Fp8LinearWeight, device: *compute.cuda.Device) void {
        if (self.scales_buffer.size > 0) self.scales_buffer.deinit(device);
        self.buffer.deinit(device);
    }

    pub fn byteSize(self: *const Fp8LinearWeight) usize {
        return self.buffer.size + self.scales_buffer.size;
    }
};

pub const Mxfp8LinearWeight = struct {
    rows: usize,
    cols: usize,
    /// GPU buffer holding E4M3 weight bytes [rows × cols]
    buffer: compute.cuda.Buffer,
    /// GPU buffer holding UE8M0 block-32 scales in cuBLASLt interleaved layout.
    /// Padded to 128-tile boundaries: [padded_outer × padded_sf_k] bytes.
    scales_buffer: compute.cuda.Buffer,
    /// GPU buffer holding UE8M0 block-32 scales in simple row-major layout
    /// [cols × scale_cols] for the GEMV kernel path. Same data, different layout.
    scales_raw_buffer: compute.cuda.Buffer,
    scale_cols: u32,

    pub fn deinit(self: *Mxfp8LinearWeight, device: *compute.cuda.Device) void {
        if (self.scales_raw_buffer.size > 0) self.scales_raw_buffer.deinit(device);
        if (self.scales_buffer.size > 0) self.scales_buffer.deinit(device);
        self.buffer.deinit(device);
    }

    pub fn byteSize(self: *const Mxfp8LinearWeight) usize {
        return self.buffer.size + self.scales_buffer.size + self.scales_raw_buffer.size;
    }

    /// Compute cuBLASLt-required scale tensor size for VEC32_UE8M0 block scaling.
    /// inner = contraction dimension (K), outer = non-contraction dimension (M or N).
    /// Returns the total number of UE8M0 scale bytes needed (padded to 128-tile boundaries).
    pub fn cublasLtScaleTensorSize(inner: usize, outer: usize) usize {
        const block_rows: usize = 128; // inner dimension tile
        const block_cols: usize = 128; // outer dimension tile
        const s_rows = roundoff(inner, block_rows) / 32;
        const s_cols = roundoff(outer, block_cols);
        return s_rows * s_cols;
    }

    pub fn roundoff(x: usize, granul: usize) usize {
        return granul * ((x + (granul - 1)) / granul);
    }
};

pub const Nvfp4LinearWeight = struct {
    rows: usize,
    cols: usize,
    /// Packed FP4 bytes: [out_dim × packed_in] (2 FP4 values per byte).
    buffer: compute.cuda.Buffer,
    /// FP8 E4M3 scales in row-major layout [out_dim × scale_cols].
    scales_buffer: compute.cuda.Buffer,
    /// FP8 UE4M3 block-16 scales in cuBLASLt interleaved layout.
    /// Padded to 128-tile boundaries: [padded_outer × padded_sf_k] bytes.
    scales_lt_buffer: compute.cuda.Buffer,
    packed_cols: u32,
    scale_cols: u32,
    group_size: u32,
    weight_global_scale: f32,

    pub fn deinit(self: *Nvfp4LinearWeight, device: *compute.cuda.Device) void {
        if (self.scales_lt_buffer.size > 0) self.scales_lt_buffer.deinit(device);
        if (self.scales_buffer.size > 0) self.scales_buffer.deinit(device);
        self.buffer.deinit(device);
    }

    pub fn byteSize(self: *const Nvfp4LinearWeight) usize {
        return self.buffer.size + self.scales_buffer.size + self.scales_lt_buffer.size;
    }

    /// Compute cuBLASLt-required scale tensor size for VEC16_UE4M3 block scaling.
    /// inner = contraction dimension (K), outer = non-contraction dimension (M or N).
    /// Returns total UE4M3 scale bytes padded to cuBLASLt tile boundaries.
    pub fn cublasLtScaleTensorSize(inner: usize, outer: usize) usize {
        const sf_k = roundoff((inner + 15) / 16, 4);
        return roundoff(outer, 128) * sf_k;
    }

    pub fn roundoff(x: usize, granul: usize) usize {
        return granul * ((x + (granul - 1)) / granul);
    }
};

pub const LinearWeight = union(enum) {
    dense_f32: DeviceTensor,
    dense_u16: U16LinearWeight,
    gaffine_u4: GaffineU4LinearWeight,
    gaffine_u8: GaffineU8LinearWeight,
    fp8: Fp8LinearWeight,
    mxfp8: Mxfp8LinearWeight,
    nvfp4: Nvfp4LinearWeight,

    pub fn deinit(self: *LinearWeight, device: *compute.cuda.Device) void {
        switch (self.*) {
            .dense_f32 => |*w| w.deinit(device),
            .dense_u16 => |*w| w.deinit(device),
            .gaffine_u4 => |*w| w.deinit(device),
            .gaffine_u8 => |*w| w.deinit(device),
            .fp8 => |*w| w.deinit(device),
            .mxfp8 => |*w| w.deinit(device),
            .nvfp4 => |*w| w.deinit(device),
        }
    }

    pub fn rows(self: *const LinearWeight) usize {
        return switch (self.*) {
            .dense_f32 => |w| w.rows,
            .dense_u16 => |w| w.rows,
            .gaffine_u4 => |w| w.rows,
            .gaffine_u8 => |w| w.rows,
            .fp8 => |w| w.rows,
            .mxfp8 => |w| w.rows,
            .nvfp4 => |w| w.rows,
        };
    }

    pub fn cols(self: *const LinearWeight) usize {
        return switch (self.*) {
            .dense_f32 => |w| w.cols,
            .dense_u16 => |w| w.cols,
            .gaffine_u4 => |w| w.cols,
            .gaffine_u8 => |w| w.cols,
            .fp8 => |w| w.cols,
            .mxfp8 => |w| w.cols,
            .nvfp4 => |w| w.cols,
        };
    }

    pub fn byteSize(self: *const LinearWeight) usize {
        return switch (self.*) {
            .dense_f32 => |w| w.buffer.size,
            .dense_u16 => |w| w.byteSize(),
            .gaffine_u4 => |w| w.byteSize(),
            .gaffine_u8 => |w| w.byteSize(),
            .fp8 => |w| w.byteSize(),
            .mxfp8 => |w| w.byteSize(),
            .nvfp4 => |w| w.byteSize(),
        };
    }
};

pub const RuntimeBuffers = struct {
    projected_vocab: usize,
    max_dff: usize,
    max_attn: usize,
    max_kv: usize,
    max_gdelta_proj: usize,
    max_seq_len: usize,
    head_dim: usize,
    shortconv_dim: usize,
    row_capacity: usize,
    using_model_norm: bool,
    using_model_projection: bool,
    projection_from_lm_head: bool,
    using_model_embeddings: bool,
    embedding_lookup: ?EmbeddingLookup,
    hidden_host: []f32,
    projected_logits_host: []f32,
    projected_logits_batch_host: []f32,
    prefill_tokens_dev: compute.cuda.Buffer,
    input_dev: compute.cuda.Buffer,
    norm_weight_dev: compute.cuda.Buffer,
    norm_out_dev: compute.cuda.Buffer,
    activation_u16_dev: compute.cuda.Buffer,
    attn_q_dev: compute.cuda.Buffer,
    query_gate_proj_dev: compute.cuda.Buffer,
    attn_k_dev: compute.cuda.Buffer,
    attn_v_dev: compute.cuda.Buffer,
    attn_context_dev: compute.cuda.Buffer,
    decode_key_cache_ptrs_host: []u64,
    decode_value_cache_ptrs_host: []u64,
    decode_attn_key_cache_ptrs_table_host: []u64,
    decode_attn_value_cache_ptrs_table_host: []u64,
    decode_attn_k_scale_ptrs_table_host: []u64,
    decode_attn_v_scale_ptrs_table_host: []u64,
    decode_seq_lens_host: []u32,
    decode_positions_host: []u32,
    decode_gd_conv_state_ptrs_table_host: []u64,
    decode_gd_ssm_state_ptrs_table_host: []u64,
    decode_gd_conv_ring_heads_table_host: []u32,
    decode_key_cache_ptrs_dev: compute.cuda.Buffer,
    decode_value_cache_ptrs_dev: compute.cuda.Buffer,
    decode_attn_key_cache_ptrs_table_dev: compute.cuda.Buffer,
    decode_attn_value_cache_ptrs_table_dev: compute.cuda.Buffer,
    decode_attn_k_scale_ptrs_table_dev: compute.cuda.Buffer,
    decode_attn_v_scale_ptrs_table_dev: compute.cuda.Buffer,
    decode_seq_lens_dev: compute.cuda.Buffer,
    decode_positions_dev: compute.cuda.Buffer,
    decode_gd_conv_state_ptrs_table_dev: compute.cuda.Buffer,
    decode_gd_ssm_state_ptrs_table_dev: compute.cuda.Buffer,
    decode_gd_conv_ring_heads_table_dev: compute.cuda.Buffer,
    attn_scores_dev: ?compute.cuda.Buffer,
    attn_probs_dev: ?compute.cuda.Buffer,
    attn_out_dev: compute.cuda.Buffer,
    ffn_gate_dev: compute.cuda.Buffer,
    ffn_up_dev: compute.cuda.Buffer,
    ffn_mul_dev: compute.cuda.Buffer,
    ffn_down_dev: compute.cuda.Buffer,
    deepstack_add_dev: compute.cuda.Buffer,
    shortconv_proj_dev: compute.cuda.Buffer,
    shortconv_conv_dev: compute.cuda.Buffer,
    gdelta_proj_dev: compute.cuda.Buffer,
    gdelta_ssm_dev: compute.cuda.Buffer,
    projection_weight: LinearWeight,
    logits_dev: compute.cuda.Buffer,
    topk_values_dev: compute.cuda.Buffer,
    topk_ids_dev: compute.cuda.Buffer,
    batched_attn_scores_dev: compute.cuda.Buffer,
    batched_attn_probs_dev: compute.cuda.Buffer,
    batched_attn_max_seq_len: u32,
    dequant_f16_dev: compute.cuda.Buffer,

    pub fn init(
        allocator: std.mem.Allocator,
        device: *compute.cuda.Device,
        loaded: *const LoadedModel,
        max_dff: usize,
        max_attn: usize,
        max_kv: usize,
        max_gdelta_proj: usize,
        max_shortconv_dim: usize,
        max_seq_len: usize,
        n_heads: usize,
        head_dim: usize,
        max_batch_size: usize,
        max_attn_layers: usize,
        max_gd_layers: usize,
        skip_embedding: bool,
        skip_projection: bool,
    ) !RuntimeBuffers {
        const d_model: usize = @intCast(loaded.config.d_model);
        const vocab_size: usize = @intCast(loaded.config.vocab_size);
        if (d_model == 0 or vocab_size == 0) return error.InvalidArgument;
        if (max_dff == 0) return error.InvalidArgument;
        if (max_attn == 0) return error.InvalidArgument;
        if (max_kv == 0 or max_gdelta_proj == 0 or max_seq_len == 0 or head_dim == 0) return error.InvalidArgument;
        if (max_batch_size == 0) return error.InvalidArgument;
        if (max_attn_layers == 0) return error.InvalidArgument;
        if (max_gd_layers == 0) return error.InvalidArgument;

        const d_model_bytes = std.math.mul(usize, d_model, @sizeOf(f32)) catch return error.InvalidArgument;
        const d_ff_bytes = std.math.mul(usize, max_dff, @sizeOf(f32)) catch return error.InvalidArgument;
        const d_attn_bytes = std.math.mul(usize, max_attn, @sizeOf(f32)) catch return error.InvalidArgument;
        const d_kv_bytes = std.math.mul(usize, max_kv, @sizeOf(f32)) catch return error.InvalidArgument;
        const d_gdelta_proj_bytes = std.math.mul(usize, max_gdelta_proj, @sizeOf(f32)) catch return error.InvalidArgument;
        const max_linear_in_dim = @max(@max(d_model, max_dff), @max(max_attn, max_gdelta_proj));
        const activation_u16_bytes = std.math.mul(usize, max_linear_in_dim, @sizeOf(u16)) catch return error.InvalidArgument;
        const shortconv_dim = if (max_shortconv_dim > 0) max_shortconv_dim else 1;
        const shortconv_proj_bytes = std.math.mul(usize, shortconv_dim * 3, @sizeOf(f32)) catch return error.InvalidArgument;
        const shortconv_conv_bytes = std.math.mul(usize, shortconv_dim, @sizeOf(f32)) catch return error.InvalidArgument;
        const need_attention_score_buffers = attention_policy.needAttentionScoreBuffers(
            attention_policy_config,
            max_seq_len,
            head_dim,
        );
        const attn_rows = std.math.mul(usize, max_seq_len, n_heads) catch return error.InvalidArgument;
        const attn_rows_bytes = std.math.mul(usize, attn_rows, @sizeOf(f32)) catch return error.InvalidArgument;
        const hidden_host = try allocator.alloc(f32, d_model);
        errdefer allocator.free(hidden_host);

        const norm_weight_host = try allocator.alloc(f32, d_model);
        defer allocator.free(norm_weight_host);
        const using_model_norm = tryPopulateFinalNormWeight(loaded, norm_weight_host);
        if (!using_model_norm) {
            log.warn("inference", "CUDA final norm weight unsupported", .{
                .has_ln_final = @as(u8, @intFromBool(loaded.ln_final != null)),
                .dtype = if (loaded.ln_final) |ln_final| @tagName(ln_final.dtype) else "none",
            });
            return error.UnsupportedModel;
        }
        var projection_from_lm_head = false;
        var projection_weight_opt: ?LinearWeight = null;
        errdefer if (projection_weight_opt) |*w| w.deinit(device);

        if (skip_projection) {
            // Intermediate pipeline stage — never computes logits.
            // Skip uploading projection weight to avoid init-time peak memory.
            projection_weight_opt = .{ .dense_f32 = missing_device_tensor };
        } else {
            if (loaded.lm_head) |lm_head| {
                projection_weight_opt = uploadLinearWeight(device, allocator, &lm_head, d_model) catch |err| switch (err) {
                    error.UnsupportedModel, error.InvalidArgument => null,
                    else => return err,
                };
                projection_from_lm_head = projection_weight_opt != null;
            }
            if (projection_weight_opt == null) {
                projection_weight_opt = uploadLinearWeight(device, allocator, &loaded.token_embeddings, d_model) catch |err| switch (err) {
                    error.UnsupportedModel, error.InvalidArgument => null,
                    else => return err,
                };
            }

            if (projection_weight_opt == null) {
                log.warn("inference", "CUDA projection weight unsupported", .{
                    .d_model = d_model,
                    .vocab_size = vocab_size,
                    .has_lm_head = @as(u8, @intFromBool(loaded.lm_head != null)),
                    .embed_dtype = @tagName(loaded.token_embeddings.dtype),
                    .embed_ndim = loaded.token_embeddings.n_dims,
                });
                return error.UnsupportedModel;
            }
        }
        const using_model_projection = !skip_projection;
        const projection_weight = projection_weight_opt.?;
        const projected_vocab = if (skip_projection) vocab_size else projection_weight.cols();
        const projected_logits_host = try allocator.alloc(f32, projected_vocab);
        errdefer allocator.free(projected_logits_host);
        const projected_logits_batch_count = std.math.mul(usize, projected_vocab, max_batch_size) catch return error.InvalidArgument;
        const projected_logits_batch_host = try allocator.alloc(f32, projected_logits_batch_count);
        errdefer allocator.free(projected_logits_batch_host);
        const decode_key_cache_ptrs_host = try allocator.alloc(u64, max_batch_size);
        errdefer allocator.free(decode_key_cache_ptrs_host);
        const decode_value_cache_ptrs_host = try allocator.alloc(u64, max_batch_size);
        errdefer allocator.free(decode_value_cache_ptrs_host);
        const decode_attn_table_count = std.math.mul(usize, max_batch_size, max_attn_layers) catch return error.InvalidArgument;
        const decode_gd_table_count = std.math.mul(usize, max_batch_size, max_gd_layers) catch return error.InvalidArgument;
        const decode_attn_key_cache_ptrs_table_host = try allocator.alloc(u64, decode_attn_table_count);
        errdefer allocator.free(decode_attn_key_cache_ptrs_table_host);
        const decode_attn_value_cache_ptrs_table_host = try allocator.alloc(u64, decode_attn_table_count);
        errdefer allocator.free(decode_attn_value_cache_ptrs_table_host);
        const decode_attn_k_scale_ptrs_table_host = try allocator.alloc(u64, decode_attn_table_count);
        errdefer allocator.free(decode_attn_k_scale_ptrs_table_host);
        const decode_attn_v_scale_ptrs_table_host = try allocator.alloc(u64, decode_attn_table_count);
        errdefer allocator.free(decode_attn_v_scale_ptrs_table_host);
        const decode_seq_lens_host = try allocator.alloc(u32, max_batch_size);
        errdefer allocator.free(decode_seq_lens_host);
        const decode_positions_host = try allocator.alloc(u32, max_batch_size);
        errdefer allocator.free(decode_positions_host);
        const decode_gd_conv_state_ptrs_table_host = try allocator.alloc(u64, decode_gd_table_count);
        errdefer allocator.free(decode_gd_conv_state_ptrs_table_host);
        const decode_gd_ssm_state_ptrs_table_host = try allocator.alloc(u64, decode_gd_table_count);
        errdefer allocator.free(decode_gd_ssm_state_ptrs_table_host);
        const decode_gd_conv_ring_heads_table_host = try allocator.alloc(u32, decode_gd_table_count);
        errdefer allocator.free(decode_gd_conv_ring_heads_table_host);
        const logits_bytes = std.math.mul(usize, projected_logits_batch_count, @sizeOf(f32)) catch return error.InvalidArgument;
        const topk_buffer_count = std.math.mul(usize, max_batch_size, 256) catch return error.InvalidArgument;
        const topk_values_bytes = std.math.mul(usize, topk_buffer_count, @sizeOf(f32)) catch return error.InvalidArgument;
        const topk_ids_bytes = std.math.mul(usize, topk_buffer_count, @sizeOf(u32)) catch return error.InvalidArgument;
        const decode_ptrs_bytes = std.math.mul(usize, max_batch_size, @sizeOf(u64)) catch return error.InvalidArgument;
        const decode_attn_table_ptrs_bytes = std.math.mul(usize, decode_attn_table_count, @sizeOf(u64)) catch return error.InvalidArgument;
        const decode_gd_table_ptrs_bytes = std.math.mul(usize, decode_gd_table_count, @sizeOf(u64)) catch return error.InvalidArgument;
        const decode_gd_table_idx_bytes = std.math.mul(usize, decode_gd_table_count, @sizeOf(u32)) catch return error.InvalidArgument;
        const decode_idx_bytes = std.math.mul(usize, max_batch_size, @sizeOf(u32)) catch return error.InvalidArgument;
        var embedding_lookup: ?EmbeddingLookup = null;
        errdefer if (embedding_lookup) |*lookup| lookup.deinit(device);
        const using_model_embeddings = if (skip_embedding) true else canUseModelEmbeddings(loaded, d_model);
        if (!using_model_embeddings) {
            log.warn("inference", "CUDA token embeddings unsupported", .{
                .d_model = d_model,
                .embed_dtype = @tagName(loaded.token_embeddings.dtype),
                .embed_ndim = loaded.token_embeddings.n_dims,
                .embed_shape_0 = loaded.token_embeddings.shape[0],
                .embed_shape_1 = loaded.token_embeddings.shape[1],
            });
            return error.UnsupportedModel;
        }
        if (!skip_embedding) {
            embedding_lookup = try tryUploadEmbeddingLookup(device, loaded, d_model);
        }

        var input_dev = try device.allocBuffer(d_model_bytes);
        errdefer input_dev.deinit(device);
        const prefill_tokens_bytes = std.math.mul(usize, max_seq_len, @sizeOf(u32)) catch return error.InvalidArgument;
        var prefill_tokens_dev = try device.allocBuffer(prefill_tokens_bytes);
        errdefer prefill_tokens_dev.deinit(device);
        var norm_weight_dev = try device.allocBuffer(d_model_bytes);
        errdefer norm_weight_dev.deinit(device);
        var norm_out_dev = try device.allocBuffer(d_model_bytes);
        errdefer norm_out_dev.deinit(device);
        var activation_u16_dev = try device.allocBuffer(activation_u16_bytes);
        errdefer activation_u16_dev.deinit(device);
        var attn_q_dev = try device.allocBuffer(d_attn_bytes);
        errdefer attn_q_dev.deinit(device);
        var query_gate_proj_dev = try device.allocBuffer(d_attn_bytes);
        errdefer query_gate_proj_dev.deinit(device);
        var attn_k_dev = try device.allocBuffer(d_kv_bytes);
        errdefer attn_k_dev.deinit(device);
        var attn_v_dev = try device.allocBuffer(d_kv_bytes);
        errdefer attn_v_dev.deinit(device);
        var attn_context_dev = try device.allocBuffer(d_attn_bytes);
        errdefer attn_context_dev.deinit(device);
        var decode_key_cache_ptrs_dev = try device.allocBuffer(decode_ptrs_bytes);
        errdefer decode_key_cache_ptrs_dev.deinit(device);
        var decode_value_cache_ptrs_dev = try device.allocBuffer(decode_ptrs_bytes);
        errdefer decode_value_cache_ptrs_dev.deinit(device);
        var decode_attn_key_cache_ptrs_table_dev = try device.allocBuffer(decode_attn_table_ptrs_bytes);
        errdefer decode_attn_key_cache_ptrs_table_dev.deinit(device);
        var decode_attn_value_cache_ptrs_table_dev = try device.allocBuffer(decode_attn_table_ptrs_bytes);
        errdefer decode_attn_value_cache_ptrs_table_dev.deinit(device);
        var decode_attn_k_scale_ptrs_table_dev = try device.allocBuffer(decode_attn_table_ptrs_bytes);
        errdefer decode_attn_k_scale_ptrs_table_dev.deinit(device);
        var decode_attn_v_scale_ptrs_table_dev = try device.allocBuffer(decode_attn_table_ptrs_bytes);
        errdefer decode_attn_v_scale_ptrs_table_dev.deinit(device);
        var decode_seq_lens_dev = try device.allocBuffer(decode_idx_bytes);
        errdefer decode_seq_lens_dev.deinit(device);
        var decode_positions_dev = try device.allocBuffer(decode_idx_bytes);
        errdefer decode_positions_dev.deinit(device);
        var decode_gd_conv_state_ptrs_table_dev = try device.allocBuffer(decode_gd_table_ptrs_bytes);
        errdefer decode_gd_conv_state_ptrs_table_dev.deinit(device);
        var decode_gd_ssm_state_ptrs_table_dev = try device.allocBuffer(decode_gd_table_ptrs_bytes);
        errdefer decode_gd_ssm_state_ptrs_table_dev.deinit(device);
        var decode_gd_conv_ring_heads_table_dev = try device.allocBuffer(decode_gd_table_idx_bytes);
        errdefer decode_gd_conv_ring_heads_table_dev.deinit(device);
        var attn_scores_dev: ?compute.cuda.Buffer = null;
        errdefer if (attn_scores_dev) |*buf| buf.deinit(device);
        var attn_probs_dev: ?compute.cuda.Buffer = null;
        errdefer if (attn_probs_dev) |*buf| buf.deinit(device);
        if (need_attention_score_buffers) {
            attn_scores_dev = try device.allocBuffer(attn_rows_bytes);
            attn_probs_dev = try device.allocBuffer(attn_rows_bytes);
        }
        var attn_out_dev = try device.allocBuffer(d_model_bytes);
        errdefer attn_out_dev.deinit(device);
        var ffn_gate_dev = try device.allocBuffer(d_ff_bytes);
        errdefer ffn_gate_dev.deinit(device);
        var ffn_up_dev = try device.allocBuffer(d_ff_bytes);
        errdefer ffn_up_dev.deinit(device);
        var ffn_mul_dev = try device.allocBuffer(d_ff_bytes);
        errdefer ffn_mul_dev.deinit(device);
        var ffn_down_dev = try device.allocBuffer(d_model_bytes);
        errdefer ffn_down_dev.deinit(device);
        var deepstack_add_dev = try device.allocBuffer(d_model_bytes);
        errdefer deepstack_add_dev.deinit(device);
        var shortconv_proj_dev = try device.allocBuffer(shortconv_proj_bytes);
        errdefer shortconv_proj_dev.deinit(device);
        var shortconv_conv_dev = try device.allocBuffer(shortconv_conv_bytes);
        errdefer shortconv_conv_dev.deinit(device);
        var gdelta_proj_dev = try device.allocBuffer(d_gdelta_proj_bytes);
        errdefer gdelta_proj_dev.deinit(device);
        var gdelta_ssm_dev = try device.allocBuffer(d_gdelta_proj_bytes);
        errdefer gdelta_ssm_dev.deinit(device);
        var logits_dev = try device.allocBuffer(logits_bytes);
        errdefer logits_dev.deinit(device);
        var topk_values_dev = try device.allocBuffer(topk_values_bytes);
        errdefer topk_values_dev.deinit(device);
        var topk_ids_dev = try device.allocBuffer(topk_ids_bytes);
        errdefer topk_ids_dev.deinit(device);
        // Batched attention scores/probs buffers: [max_batch_size * n_heads * max_seq_len].
        const batched_attn_max_seq_len: u32 = @intCast(max_seq_len);
        const batched_attn_elems = std.math.mul(usize, max_batch_size * n_heads, max_seq_len) catch return error.InvalidArgument;
        const batched_attn_bytes = std.math.mul(usize, batched_attn_elems, @sizeOf(f32)) catch return error.InvalidArgument;
        var batched_attn_scores_dev = try device.allocBuffer(batched_attn_bytes);
        errdefer batched_attn_scores_dev.deinit(device);
        var batched_attn_probs_dev = try device.allocBuffer(batched_attn_bytes);
        errdefer batched_attn_probs_dev.deinit(device);
        const max_dequant_dim = @max(@max(max_dff, max_attn), @max(max_kv, max_gdelta_proj));
        const dequant_f16_bytes = std.math.mul(usize, std.math.mul(usize, d_model, max_dequant_dim) catch return error.InvalidArgument, @sizeOf(u16)) catch return error.InvalidArgument;
        var dequant_f16_dev = try device.allocBuffer(dequant_f16_bytes);
        errdefer dequant_f16_dev.deinit(device);

        try norm_weight_dev.upload(device, std.mem.sliceAsBytes(norm_weight_host));

        return .{
            .projected_vocab = projected_vocab,
            .max_dff = max_dff,
            .max_attn = max_attn,
            .max_kv = max_kv,
            .max_gdelta_proj = max_gdelta_proj,
            .max_seq_len = max_seq_len,
            .head_dim = head_dim,
            .shortconv_dim = shortconv_dim,
            .row_capacity = 1,
            .using_model_norm = using_model_norm,
            .using_model_projection = using_model_projection,
            .projection_from_lm_head = projection_from_lm_head,
            .using_model_embeddings = using_model_embeddings,
            .embedding_lookup = embedding_lookup,
            .hidden_host = hidden_host,
            .projected_logits_host = projected_logits_host,
            .projected_logits_batch_host = projected_logits_batch_host,
            .prefill_tokens_dev = prefill_tokens_dev,
            .input_dev = input_dev,
            .norm_weight_dev = norm_weight_dev,
            .norm_out_dev = norm_out_dev,
            .activation_u16_dev = activation_u16_dev,
            .attn_q_dev = attn_q_dev,
            .query_gate_proj_dev = query_gate_proj_dev,
            .attn_k_dev = attn_k_dev,
            .attn_v_dev = attn_v_dev,
            .attn_context_dev = attn_context_dev,
            .decode_key_cache_ptrs_host = decode_key_cache_ptrs_host,
            .decode_value_cache_ptrs_host = decode_value_cache_ptrs_host,
            .decode_attn_key_cache_ptrs_table_host = decode_attn_key_cache_ptrs_table_host,
            .decode_attn_value_cache_ptrs_table_host = decode_attn_value_cache_ptrs_table_host,
            .decode_attn_k_scale_ptrs_table_host = decode_attn_k_scale_ptrs_table_host,
            .decode_attn_v_scale_ptrs_table_host = decode_attn_v_scale_ptrs_table_host,
            .decode_seq_lens_host = decode_seq_lens_host,
            .decode_positions_host = decode_positions_host,
            .decode_gd_conv_state_ptrs_table_host = decode_gd_conv_state_ptrs_table_host,
            .decode_gd_ssm_state_ptrs_table_host = decode_gd_ssm_state_ptrs_table_host,
            .decode_gd_conv_ring_heads_table_host = decode_gd_conv_ring_heads_table_host,
            .decode_key_cache_ptrs_dev = decode_key_cache_ptrs_dev,
            .decode_value_cache_ptrs_dev = decode_value_cache_ptrs_dev,
            .decode_attn_key_cache_ptrs_table_dev = decode_attn_key_cache_ptrs_table_dev,
            .decode_attn_value_cache_ptrs_table_dev = decode_attn_value_cache_ptrs_table_dev,
            .decode_attn_k_scale_ptrs_table_dev = decode_attn_k_scale_ptrs_table_dev,
            .decode_attn_v_scale_ptrs_table_dev = decode_attn_v_scale_ptrs_table_dev,
            .decode_seq_lens_dev = decode_seq_lens_dev,
            .decode_positions_dev = decode_positions_dev,
            .decode_gd_conv_state_ptrs_table_dev = decode_gd_conv_state_ptrs_table_dev,
            .decode_gd_ssm_state_ptrs_table_dev = decode_gd_ssm_state_ptrs_table_dev,
            .decode_gd_conv_ring_heads_table_dev = decode_gd_conv_ring_heads_table_dev,
            .attn_scores_dev = attn_scores_dev,
            .attn_probs_dev = attn_probs_dev,
            .attn_out_dev = attn_out_dev,
            .ffn_gate_dev = ffn_gate_dev,
            .ffn_up_dev = ffn_up_dev,
            .ffn_mul_dev = ffn_mul_dev,
            .ffn_down_dev = ffn_down_dev,
            .deepstack_add_dev = deepstack_add_dev,
            .shortconv_proj_dev = shortconv_proj_dev,
            .shortconv_conv_dev = shortconv_conv_dev,
            .gdelta_proj_dev = gdelta_proj_dev,
            .gdelta_ssm_dev = gdelta_ssm_dev,
            .projection_weight = projection_weight,
            .logits_dev = logits_dev,
            .topk_values_dev = topk_values_dev,
            .topk_ids_dev = topk_ids_dev,
            .batched_attn_scores_dev = batched_attn_scores_dev,
            .batched_attn_probs_dev = batched_attn_probs_dev,
            .batched_attn_max_seq_len = batched_attn_max_seq_len,
            .dequant_f16_dev = dequant_f16_dev,
        };
    }

    pub fn deinit(self: *RuntimeBuffers, allocator: std.mem.Allocator, device: *compute.cuda.Device) void {
        self.dequant_f16_dev.deinit(device);
        self.batched_attn_probs_dev.deinit(device);
        self.batched_attn_scores_dev.deinit(device);
        self.topk_ids_dev.deinit(device);
        self.topk_values_dev.deinit(device);
        self.logits_dev.deinit(device);
        self.projection_weight.deinit(device);
        if (self.embedding_lookup) |*lookup| lookup.deinit(device);
        self.shortconv_conv_dev.deinit(device);
        self.shortconv_proj_dev.deinit(device);
        self.gdelta_proj_dev.deinit(device);
        self.gdelta_ssm_dev.deinit(device);
        self.ffn_down_dev.deinit(device);
        self.ffn_mul_dev.deinit(device);
        self.ffn_up_dev.deinit(device);
        self.ffn_gate_dev.deinit(device);
        self.attn_out_dev.deinit(device);
        if (self.attn_probs_dev) |*buf| buf.deinit(device);
        if (self.attn_scores_dev) |*buf| buf.deinit(device);
        self.attn_context_dev.deinit(device);
        self.decode_positions_dev.deinit(device);
        self.decode_seq_lens_dev.deinit(device);
        self.decode_gd_conv_ring_heads_table_dev.deinit(device);
        self.decode_gd_ssm_state_ptrs_table_dev.deinit(device);
        self.decode_gd_conv_state_ptrs_table_dev.deinit(device);
        self.decode_attn_v_scale_ptrs_table_dev.deinit(device);
        self.decode_attn_k_scale_ptrs_table_dev.deinit(device);
        self.decode_attn_value_cache_ptrs_table_dev.deinit(device);
        self.decode_attn_key_cache_ptrs_table_dev.deinit(device);
        self.decode_value_cache_ptrs_dev.deinit(device);
        self.decode_key_cache_ptrs_dev.deinit(device);
        self.attn_v_dev.deinit(device);
        self.attn_k_dev.deinit(device);
        self.query_gate_proj_dev.deinit(device);
        self.attn_q_dev.deinit(device);
        self.norm_out_dev.deinit(device);
        self.norm_weight_dev.deinit(device);
        self.activation_u16_dev.deinit(device);
        self.input_dev.deinit(device);
        self.deepstack_add_dev.deinit(device);
        self.prefill_tokens_dev.deinit(device);
        allocator.free(self.projected_logits_batch_host);
        allocator.free(self.projected_logits_host);
        allocator.free(self.hidden_host);
        allocator.free(self.decode_gd_conv_ring_heads_table_host);
        allocator.free(self.decode_gd_ssm_state_ptrs_table_host);
        allocator.free(self.decode_gd_conv_state_ptrs_table_host);
        allocator.free(self.decode_positions_host);
        allocator.free(self.decode_seq_lens_host);
        allocator.free(self.decode_attn_v_scale_ptrs_table_host);
        allocator.free(self.decode_attn_k_scale_ptrs_table_host);
        allocator.free(self.decode_attn_value_cache_ptrs_table_host);
        allocator.free(self.decode_attn_key_cache_ptrs_table_host);
        allocator.free(self.decode_value_cache_ptrs_host);
        allocator.free(self.decode_key_cache_ptrs_host);
    }

    pub fn deviceByteSize(self: *const RuntimeBuffers) usize {
        return self.input_dev.size +
            self.prefill_tokens_dev.size +
            self.norm_weight_dev.size +
            self.norm_out_dev.size +
            self.activation_u16_dev.size +
            self.attn_q_dev.size +
            self.query_gate_proj_dev.size +
            self.attn_k_dev.size +
            self.attn_v_dev.size +
            self.attn_context_dev.size +
            self.decode_key_cache_ptrs_dev.size +
            self.decode_value_cache_ptrs_dev.size +
            self.decode_attn_key_cache_ptrs_table_dev.size +
            self.decode_attn_value_cache_ptrs_table_dev.size +
            self.decode_attn_k_scale_ptrs_table_dev.size +
            self.decode_attn_v_scale_ptrs_table_dev.size +
            self.decode_seq_lens_dev.size +
            self.decode_positions_dev.size +
            self.decode_gd_conv_state_ptrs_table_dev.size +
            self.decode_gd_ssm_state_ptrs_table_dev.size +
            self.decode_gd_conv_ring_heads_table_dev.size +
            (if (self.attn_scores_dev) |buf| buf.size else 0) +
            (if (self.attn_probs_dev) |buf| buf.size else 0) +
            self.attn_out_dev.size +
            self.ffn_gate_dev.size +
            self.ffn_up_dev.size +
            self.ffn_mul_dev.size +
            self.ffn_down_dev.size +
            self.deepstack_add_dev.size +
            self.shortconv_proj_dev.size +
            self.shortconv_conv_dev.size +
            self.gdelta_proj_dev.size +
            self.gdelta_ssm_dev.size +
            self.topk_values_dev.size +
            self.topk_ids_dev.size +
            self.batched_attn_scores_dev.size +
            self.batched_attn_probs_dev.size +
            self.dequant_f16_dev.size +
            self.logits_dev.size +
            (if (self.embedding_lookup) |lookup| lookup.byteSize() else 0) +
            self.projection_weight.byteSize();
    }

    pub fn requireAttentionScoresDev(self: *RuntimeBuffers) !*compute.cuda.Buffer {
        if (self.attn_scores_dev) |*buf| return buf;
        return error.CudaKernelUnavailable;
    }

    pub fn requireAttentionProbsDev(self: *RuntimeBuffers) !*compute.cuda.Buffer {
        if (self.attn_probs_dev) |*buf| return buf;
        return error.CudaKernelUnavailable;
    }

    pub fn ensureRowCapacity(
        self: *RuntimeBuffers,
        device: *compute.cuda.Device,
        required_rows: usize,
        fixed_alloc_mode: bool,
    ) !void {
        if (required_rows == 0) return error.InvalidArgument;
        if (required_rows <= self.row_capacity) return;
        if (required_rows > self.max_seq_len) return error.InvalidArgument;
        if (fixed_alloc_mode) return error.OutOfMemory;

        var new_capacity = self.row_capacity;
        if (new_capacity == 0) new_capacity = 1;
        while (new_capacity < required_rows) {
            const doubled = std.math.mul(usize, new_capacity, 2) catch self.max_seq_len;
            const next = if (doubled > new_capacity) doubled else self.max_seq_len;
            new_capacity = @min(self.max_seq_len, next);
            if (new_capacity == self.max_seq_len) break;
        }
        if (new_capacity < required_rows) return error.InvalidArgument;

        const d_model_bytes = std.math.mul(usize, self.hidden_host.len, @sizeOf(f32)) catch return error.InvalidArgument;
        const d_ff_bytes = std.math.mul(usize, self.max_dff, @sizeOf(f32)) catch return error.InvalidArgument;
        const d_attn_bytes = std.math.mul(usize, self.max_attn, @sizeOf(f32)) catch return error.InvalidArgument;
        const d_kv_bytes = std.math.mul(usize, self.max_kv, @sizeOf(f32)) catch return error.InvalidArgument;
        const d_gdelta_proj_bytes = std.math.mul(usize, self.max_gdelta_proj, @sizeOf(f32)) catch return error.InvalidArgument;
        const max_linear_in_dim = @max(@max(self.hidden_host.len, self.max_dff), @max(self.max_attn, self.max_gdelta_proj));
        const activation_u16_bytes = std.math.mul(usize, max_linear_in_dim, @sizeOf(u16)) catch return error.InvalidArgument;
        const shortconv_proj_bytes = std.math.mul(usize, self.shortconv_dim * 3, @sizeOf(f32)) catch return error.InvalidArgument;
        const shortconv_conv_bytes = std.math.mul(usize, self.shortconv_dim, @sizeOf(f32)) catch return error.InvalidArgument;
        try resizeScratchBuffer(device, &self.input_dev, std.math.mul(usize, d_model_bytes, new_capacity) catch return error.InvalidArgument);
        try resizeScratchBuffer(device, &self.norm_out_dev, std.math.mul(usize, d_model_bytes, new_capacity) catch return error.InvalidArgument);
        try resizeScratchBuffer(device, &self.activation_u16_dev, std.math.mul(usize, activation_u16_bytes, new_capacity) catch return error.InvalidArgument);
        try resizeScratchBuffer(device, &self.attn_q_dev, std.math.mul(usize, d_attn_bytes, new_capacity) catch return error.InvalidArgument);
        try resizeScratchBuffer(device, &self.query_gate_proj_dev, std.math.mul(usize, d_attn_bytes, new_capacity) catch return error.InvalidArgument);
        try resizeScratchBuffer(device, &self.attn_k_dev, std.math.mul(usize, d_kv_bytes, new_capacity) catch return error.InvalidArgument);
        try resizeScratchBuffer(device, &self.attn_v_dev, std.math.mul(usize, d_kv_bytes, new_capacity) catch return error.InvalidArgument);
        try resizeScratchBuffer(device, &self.attn_context_dev, std.math.mul(usize, d_attn_bytes, new_capacity) catch return error.InvalidArgument);
        try resizeScratchBuffer(device, &self.attn_out_dev, std.math.mul(usize, d_model_bytes, new_capacity) catch return error.InvalidArgument);
        try resizeScratchBuffer(device, &self.ffn_gate_dev, std.math.mul(usize, d_ff_bytes, new_capacity) catch return error.InvalidArgument);
        try resizeScratchBuffer(device, &self.ffn_up_dev, std.math.mul(usize, d_ff_bytes, new_capacity) catch return error.InvalidArgument);
        try resizeScratchBuffer(device, &self.ffn_mul_dev, std.math.mul(usize, d_ff_bytes, new_capacity) catch return error.InvalidArgument);
        try resizeScratchBuffer(device, &self.ffn_down_dev, std.math.mul(usize, d_model_bytes, new_capacity) catch return error.InvalidArgument);
        try resizeScratchBuffer(device, &self.deepstack_add_dev, std.math.mul(usize, d_model_bytes, new_capacity) catch return error.InvalidArgument);
        try resizeScratchBuffer(device, &self.shortconv_proj_dev, std.math.mul(usize, shortconv_proj_bytes, new_capacity) catch return error.InvalidArgument);
        try resizeScratchBuffer(device, &self.shortconv_conv_dev, std.math.mul(usize, shortconv_conv_bytes, new_capacity) catch return error.InvalidArgument);
        try resizeScratchBuffer(device, &self.gdelta_proj_dev, std.math.mul(usize, d_gdelta_proj_bytes, new_capacity) catch return error.InvalidArgument);
        try resizeScratchBuffer(device, &self.gdelta_ssm_dev, std.math.mul(usize, d_gdelta_proj_bytes, new_capacity) catch return error.InvalidArgument);

        self.row_capacity = new_capacity;
    }
};

pub const LayerAttentionRuntime = struct {
    q_dim: usize,
    q_projection_dim: usize,
    kv_dim: usize,
    d_ff: usize,
    sliding_window: usize,
    is_causal: bool,
    query_gate: bool,
    ln1_weight: DeviceTensor,
    ln2_weight: DeviceTensor,
    pre_ffn_norm_weight: ?DeviceTensor = null,
    post_ffn_norm_weight: ?DeviceTensor = null,
    q_norm_weight: ?DeviceTensor = null,
    k_norm_weight: ?DeviceTensor = null,
    q_proj: LinearWeight,
    k_proj: LinearWeight,
    v_proj: LinearWeight,
    o_proj: LinearWeight,
    w1: LinearWeight,
    w2: LinearWeight,
    w3: LinearWeight,
    k_cache: compute.cuda.Buffer,
    v_cache: compute.cuda.Buffer,
    k_scale: compute.cuda.Buffer = .{ .pointer = 0, .size = 0 },
    v_scale: compute.cuda.Buffer = .{ .pointer = 0, .size = 0 },
    kv_capacity: usize,
    qkv_i8_concat: compute.cuda.Buffer = .{ .pointer = 0, .size = 0 },
    qkv_scales_concat: compute.cuda.Buffer = .{ .pointer = 0, .size = 0 },
    qkv_concat_dims: [3]u32 = .{ 0, 0, 0 },
    slot_kv_index: usize,
    kv_shared_source_layer: ?usize = null,
    kv_shared_source_slot_kv_index: ?usize = null,
    use_v_norm: bool = false,
    cpu_kernel: ?cpu_kernels.MultiHeadAttention = null,
    cpu_cache: ?cpu_kernels.AttnCache = null,
    cpu_scratch: ?cpu_kernels.AttnTemp = null,
    cpu_matmul_scratch: ?compute.cpu.linalg.MatmulScratch = null,

    pub fn deinit(self: *LayerAttentionRuntime, device: *compute.cuda.Device) void {
        if (self.cpu_matmul_scratch) |*scratch| scratch.deinit();
        if (self.cpu_scratch) |*scratch| scratch.deinit(self.cpu_kernel.?.allocator);
        if (self.cpu_cache) |*cache| cache.deinit(self.cpu_kernel.?.allocator);
        if (self.qkv_scales_concat.pointer != 0) self.qkv_scales_concat.deinit(device);
        if (self.qkv_i8_concat.pointer != 0) self.qkv_i8_concat.deinit(device);
        if (self.v_scale.pointer != 0) self.v_scale.deinit(device);
        if (self.k_scale.pointer != 0) self.k_scale.deinit(device);
        self.v_cache.deinit(device);
        self.k_cache.deinit(device);
        if (self.post_ffn_norm_weight) |*w| w.deinit(device);
        if (self.pre_ffn_norm_weight) |*w| w.deinit(device);
        if (self.k_norm_weight) |*w| w.deinit(device);
        if (self.q_norm_weight) |*w| w.deinit(device);
        self.w3.deinit(device);
        self.w2.deinit(device);
        self.w1.deinit(device);
        self.o_proj.deinit(device);
        self.v_proj.deinit(device);
        self.k_proj.deinit(device);
        self.q_proj.deinit(device);
        self.ln2_weight.deinit(device);
        self.ln1_weight.deinit(device);
    }
};

pub const LayerAttentionExecConfig = struct {
    q_dim: usize,
    q_projection_dim: usize,
    kv_dim: usize,
    sliding_window: usize,
    is_causal: bool,
    query_gate: bool,
};

pub fn expectedAttentionQProjectionDim(cfg: *const LayerAttentionExecConfig) usize {
    return if (cfg.query_gate) cfg.q_projection_dim else cfg.q_dim;
}

pub fn tensorProjectionOutputDim(weight: *const Tensor, input_dim: usize) !usize {
    if (weight.n_dims != 2) return error.InvalidShape;
    const dim0: usize = @intCast(weight.shape[0]);
    const dim1: usize = @intCast(weight.shape[1]);
    if (dim0 == 0 or dim1 == 0) return error.InvalidShape;
    if (dim0 == input_dim and dim1 != input_dim) return dim1;
    if (dim1 == input_dim and dim0 != input_dim) return dim0;
    if (dim0 == input_dim and dim1 == input_dim) return input_dim;
    return dim0;
}

pub fn bufferF32RowCount(buffer: *const compute.cuda.Buffer, width: usize) !usize {
    if (width == 0) return error.InvalidArgument;
    const row_bytes = std.math.mul(usize, width, @sizeOf(f32)) catch return error.InvalidArgument;
    if (row_bytes == 0) return error.InvalidArgument;
    const rows = std.math.divExact(usize, buffer.size, row_bytes) catch return error.InvalidArgument;
    if (rows == 0) return error.InvalidArgument;
    return rows;
}

pub fn logicalF32RowSlice(
    buffer: *const compute.cuda.Buffer,
    rows: usize,
    row_index: usize,
    logical_width: usize,
) !compute.cuda.Buffer {
    if (rows == 0 or logical_width == 0 or row_index >= rows) return error.InvalidArgument;
    const row_bytes = std.math.mul(usize, logical_width, @sizeOf(f32)) catch return error.InvalidArgument;
    const packed_bytes = std.math.mul(usize, rows, row_bytes) catch return error.InvalidArgument;
    if (buffer.size < packed_bytes) return error.InvalidInstructionBinding;

    const row_stride = if (buffer.size == packed_bytes)
        row_bytes
    else blk: {
        if (buffer.size % rows != 0) return error.InvalidInstructionBinding;
        const stride = buffer.size / rows;
        if (stride < row_bytes) return error.InvalidInstructionBinding;
        break :blk stride;
    };

    const row_offset = std.math.mul(usize, row_index, row_stride) catch return error.InvalidArgument;
    return bufferSlice(buffer, row_offset, row_bytes);
}

pub const QkvI8ConcatRef = struct {
    i8_buf: compute.cuda.Buffer,
    scales_buf: compute.cuda.Buffer,
    dims: [3]u32,
};

pub const AttentionWeightRefs = struct {
    q_proj: ?*const LinearWeight = null,
    k_proj: ?*const LinearWeight = null,
    v_proj: ?*const LinearWeight = null,
    o_proj: ?*const LinearWeight = null,
    q_norm_weight: ?*const DeviceTensor = null,
    k_norm_weight: ?*const DeviceTensor = null,
};

pub const ShortConvBlockRuntime = struct {
    conv_dim: usize,
    d_conv: usize,
    d_ff: usize,
    ln1_weight: DeviceTensor,
    ln2_weight: ?DeviceTensor = null,
    in_proj: LinearWeight,
    out_proj: LinearWeight,
    conv_weight_time_major: DeviceTensor,
    conv_bias: ?DeviceTensor = null,
    conv_state: compute.cuda.Buffer,
    ffn_w1: ?LinearWeight = null,
    ffn_w2: ?LinearWeight = null,
    ffn_w3: ?LinearWeight = null,

    pub fn deinit(self: *ShortConvBlockRuntime, device: *compute.cuda.Device) void {
        if (self.ffn_w3) |*w| w.deinit(device);
        if (self.ffn_w2) |*w| w.deinit(device);
        if (self.ffn_w1) |*w| w.deinit(device);
        self.conv_state.deinit(device);
        if (self.conv_bias) |*w| w.deinit(device);
        self.conv_weight_time_major.deinit(device);
        self.out_proj.deinit(device);
        self.in_proj.deinit(device);
        if (self.ln2_weight) |*w| w.deinit(device);
        self.ln1_weight.deinit(device);
    }
};

pub const GatedDeltaSsmStateFormat = enum(u8) {
    f32,
    i8_per_column_scale,
};

pub const GatedDeltaBlockRuntime = struct {
    d_ff: usize,
    ln1_weight: DeviceTensor,
    ln2_weight: ?DeviceTensor = null,
    ffn_w1: ?LinearWeight = null,
    ffn_w2: ?LinearWeight = null,
    ffn_w3: ?LinearWeight = null,
    in_proj: LinearWeight,
    out_proj: LinearWeight,
    conv_weight_time_major: DeviceTensor,
    conv_bias: ?DeviceTensor = null,
    conv_state_dev: compute.cuda.Buffer,
    conv_ring_head: u32 = 0,
    a_log: DeviceTensor,
    dt_bias: ?DeviceTensor = null,
    norm_weight: DeviceTensor,
    ssm_state_dev: compute.cuda.Buffer,
    ssm_state_format: GatedDeltaSsmStateFormat = .f32,
    ssm_state_scales_offset: u32 = 0,
    kernel: cpu_kernels.GatedDeltaKernel,
    state: cpu_kernels.GatedDeltaState,
    scratch: cpu_kernels.GatedDeltaScratch,
    matmul_scratch: compute.cpu.linalg.MatmulScratch,

    pub fn deinit(self: *GatedDeltaBlockRuntime, allocator: std.mem.Allocator, device: *compute.cuda.Device) void {
        self.conv_state_dev.deinit(device);
        if (self.conv_bias) |*w| w.deinit(device);
        self.conv_weight_time_major.deinit(device);
        self.ssm_state_dev.deinit(device);
        if (self.dt_bias) |*w| w.deinit(device);
        self.norm_weight.deinit(device);
        self.a_log.deinit(device);
        self.out_proj.deinit(device);
        self.in_proj.deinit(device);
        if (self.ffn_w3) |*w| w.deinit(device);
        if (self.ffn_w2) |*w| w.deinit(device);
        if (self.ffn_w1) |*w| w.deinit(device);
        self.scratch.deinit();
        self.state.deinit();
        self.kernel.deinit();
        _ = allocator;
        self.matmul_scratch.deinit();
        if (self.ln2_weight) |*w| w.deinit(device);
        self.ln1_weight.deinit(device);
    }

    pub fn ssmStateDataBytes(self: *const GatedDeltaBlockRuntime) !usize {
        const d_head = @as(usize, self.kernel.config.d_head);
        const d_inner = @as(usize, self.kernel.config.n_heads) * d_head;
        const elems = std.math.mul(usize, d_inner, d_head) catch return error.InvalidArgument;
        return switch (self.ssm_state_format) {
            .f32 => std.math.mul(usize, elems, @sizeOf(f32)) catch return error.InvalidArgument,
            .i8_per_column_scale => elems,
        };
    }

    pub fn ssmStateScalesCount(self: *const GatedDeltaBlockRuntime) usize {
        return switch (self.ssm_state_format) {
            .f32 => 0,
            .i8_per_column_scale => @as(usize, self.kernel.config.n_heads) * @as(usize, self.kernel.config.d_head),
        };
    }
};

pub const ShortConvExecConfig = struct {
    conv_dim: usize,
    d_conv: usize,
};

pub const ShortConvWeightRefs = struct {
    in_proj: ?*const LinearWeight = null,
    conv_weight: ?*const DeviceTensor = null,
    out_proj: ?*const LinearWeight = null,
    conv_bias: ?*const DeviceTensor = null,
};

pub const GatedDeltaWeightRefs = struct {
    in_proj: ?*const Tensor = null,
    conv_weight: ?*const Tensor = null,
    a_log: ?*const Tensor = null,
    out_proj: ?*const Tensor = null,
    conv_bias: ?*const Tensor = null,
    dt_bias: ?*const Tensor = null,
    norm_weight: ?*const Tensor = null,
};

pub const GatedDeltaFfnUploadPlan = union(enum) {
    none,
    split: struct {
        w1: *const Tensor,
        w2: *const Tensor,
        w3: *const Tensor,
    },
    fused: struct {
        gate_up: Tensor,
        gate_up_layout: GateUpLayout,
        w2: *const Tensor,
    },
};

fn supportsFusedGateUpDenseUpload(dtype_tag: tensor.DType) bool {
    return switch (dtype_tag) {
        .f16, .bf16, .f32 => true,
        else => false,
    };
}

pub fn resolveGatedDeltaFfnUploadPlan(gated_delta: *const models.runtime_blocks.GatedDeltaBlockWeights) !GatedDeltaFfnUploadPlan {
    if (gated_delta.moe_weights != null) return .none;

    const split_w1 = gated_delta.w1;
    const split_w2 = gated_delta.w2 orelse gated_delta.down_proj;
    const split_w3 = gated_delta.w3;

    if (gated_delta.fused_gate_up) |fused| {
        const gate_up = fused.gate_up orelse return error.MissingWeight;
        const w2 = split_w2 orelse return error.MissingWeight;
        if (supportsFusedGateUpDenseUpload(gate_up.dtype)) {
            return .{
                .fused = .{
                    .gate_up = gate_up,
                    .gate_up_layout = fused.gate_up_layout,
                    .w2 = w2,
                },
            };
        }
        if (split_w1 != null or split_w2 != null or split_w3 != null) {
            return .{
                .split = .{
                    .w1 = split_w1 orelse return error.MissingWeight,
                    .w2 = split_w2 orelse return error.MissingWeight,
                    .w3 = split_w3 orelse return error.MissingWeight,
                },
            };
        }
        return error.UnsupportedModel;
    }

    if (split_w1 != null or split_w2 != null or split_w3 != null) {
        return .{
            .split = .{
                .w1 = split_w1 orelse return error.MissingWeight,
                .w2 = split_w2 orelse return error.MissingWeight,
                .w3 = split_w3 orelse return error.MissingWeight,
            },
        };
    }

    return .none;
}

pub const SwiGluWeightRefs = struct {
    w1: ?*const LinearWeight = null,
    w3: ?*const LinearWeight = null,
    w2: ?*const LinearWeight = null,
    w1_bias: ?*const DeviceTensor = null,
    w2_bias: ?*const DeviceTensor = null,
};

pub const MoEWeightRefs = struct {
    expert_gate_up: []LinearWeight,
    expert_down: []LinearWeight,
    shared_gate: LinearWeight,
    shared_up: LinearWeight,
    shared_down: LinearWeight,
    router_proj: LinearWeight,
    // Gemma4 MoE: router input/expert scales + 5 internal norms (all required).
    // Qwen3.5 MoE: none of these (simple softmax router, no internal norms).
    router_input_scale: ?DeviceTensor = null,
    router_per_expert_scale: ?DeviceTensor = null,
    pre_ffn_norm: ?DeviceTensor = null,
    post_shared_norm: ?DeviceTensor = null,
    pre_expert_norm: ?DeviceTensor = null,
    post_expert_norm: ?DeviceTensor = null,
    post_combine_norm: ?DeviceTensor = null,
    // Qwen3.5 MoE: sigmoid gate for shared expert output scaling.
    shared_expert_gate: ?LinearWeight = null,
    num_experts: u32,
    experts_per_token: u32,
    expert_d_ff: u32,
    shared_d_ff: u32,
    router_scalar: f32,
    use_gelu: bool = true,

    pub fn deinit(self: *MoEWeightRefs, allocator: std.mem.Allocator, device: *compute.cuda.Device) void {
        for (self.expert_gate_up) |*w| w.deinit(device);
        allocator.free(self.expert_gate_up);
        for (self.expert_down) |*w| w.deinit(device);
        allocator.free(self.expert_down);
        self.shared_gate.deinit(device);
        self.shared_up.deinit(device);
        self.shared_down.deinit(device);
        self.router_proj.deinit(device);
        if (self.router_input_scale) |*t| t.deinit(device);
        if (self.router_per_expert_scale) |*t| t.deinit(device);
        if (self.pre_ffn_norm) |*t| t.deinit(device);
        if (self.post_shared_norm) |*t| t.deinit(device);
        if (self.pre_expert_norm) |*t| t.deinit(device);
        if (self.post_expert_norm) |*t| t.deinit(device);
        if (self.post_combine_norm) |*t| t.deinit(device);
        if (self.shared_expert_gate) |*w| w.deinit(device);
    }
};

pub const BlockRuntimeLayer = struct {
    pub const invalid_slot = std.math.maxInt(u8);
    const MaxNormWeights = 4;

    compiled_plan: ?runtime_contract.CompiledPlan = null,
    instruction_norm_weight_slots: []?*const DeviceTensor = &.{},
    instruction_attention_exec_meta: []?LayerAttentionExecConfig = &.{},
    instruction_attention_weight_slots: []?AttentionWeightRefs = &.{},
    instruction_shortconv_exec_meta: []?ShortConvExecConfig = &.{},
    instruction_shortconv_weight_slots: []?ShortConvWeightRefs = &.{},
    instruction_gated_delta_weight_slots: []?GatedDeltaWeightRefs = &.{},
    instruction_swiglu_weight_slots: []?SwiGluWeightRefs = &.{},
    instruction_moe_weight_slots: []?*const MoEWeightRefs = &.{},
    instruction_weight_offsets: []u32 = &.{},
    instruction_weight_ptrs: []?*anyopaque = &.{},
    register_to_slot_map: []const u8 = &.{},
    slot_width_hints: []const usize = &.{},
    attention_runtime: ?LayerAttentionRuntime = null,
    shortconv_runtime: ?ShortConvBlockRuntime = null,
    gated_delta_runtime: ?GatedDeltaBlockRuntime = null,
    attention_binding: ?*LayerAttentionRuntime = null,
    shortconv_binding: ?*ShortConvBlockRuntime = null,
    gated_delta_binding: ?*GatedDeltaBlockRuntime = null,
    moe_runtime: ?MoEWeightRefs = null,
    moe_binding: ?*MoEWeightRefs = null,
    norm_weights: [MaxNormWeights]?*const DeviceTensor = [_]?*const DeviceTensor{null} ** MaxNormWeights,
    norm_weight_count: u8 = 0,

    pub fn instructionKernelIdFromWeightBindings(
        compiled: *const runtime_contract.CompiledPlan,
        op_index: usize,
        opcode: runtime_contract.Opcode,
    ) !u32 {
        return runtime_contract.instructionKernelBindingId(compiled, op_index, opcode);
    }

    const InstructionRefBinderFn = *const fn (
        self: *BlockRuntimeLayer,
        compiled: *const runtime_contract.CompiledPlan,
        op_index: usize,
        insn: *const runtime_contract.Instruction,
        norm_index: *usize,
    ) anyerror!void;

    pub fn bindInstructionNoop(
        _: *BlockRuntimeLayer,
        _: *const runtime_contract.CompiledPlan,
        _: usize,
        _: *const runtime_contract.Instruction,
        _: *usize,
    ) !void {}

    pub fn bindInstructionRmsNorm(
        self: *BlockRuntimeLayer,
        compiled: *const runtime_contract.CompiledPlan,
        op_index: usize,
        insn: *const runtime_contract.Instruction,
        norm_index: *usize,
    ) !void {
        _ = try instructionKernelIdFromWeightBindings(compiled, op_index, insn.opcode);
        if (norm_index.* >= self.norm_weight_count) return error.UnsupportedModel;
        const weight = self.norm_weights[norm_index.*] orelse return error.UnsupportedModel;
        self.instruction_norm_weight_slots[op_index] = weight;
        norm_index.* += 1;
    }

    pub fn bindInstructionAttention(
        self: *BlockRuntimeLayer,
        compiled: *const runtime_contract.CompiledPlan,
        op_index: usize,
        insn: *const runtime_contract.Instruction,
        _: *usize,
    ) !void {
        _ = try instructionKernelIdFromWeightBindings(compiled, op_index, insn.opcode);
        const binding = self.attention_binding orelse return error.UnsupportedModel;
        self.instruction_attention_exec_meta[op_index] = .{
            .q_dim = binding.q_dim,
            .q_projection_dim = binding.q_projection_dim,
            .kv_dim = binding.kv_dim,
            .sliding_window = binding.sliding_window,
            .is_causal = binding.is_causal,
            .query_gate = binding.query_gate,
        };
        self.instruction_attention_weight_slots[op_index] = .{
            .q_proj = &binding.q_proj,
            .k_proj = &binding.k_proj,
            .v_proj = &binding.v_proj,
            .o_proj = &binding.o_proj,
            .q_norm_weight = if (binding.q_norm_weight) |*weight| weight else null,
            .k_norm_weight = if (binding.k_norm_weight) |*weight| weight else null,
        };
    }

    pub fn bindInstructionShortConv(
        self: *BlockRuntimeLayer,
        compiled: *const runtime_contract.CompiledPlan,
        op_index: usize,
        insn: *const runtime_contract.Instruction,
        _: *usize,
    ) !void {
        _ = try instructionKernelIdFromWeightBindings(compiled, op_index, insn.opcode);
        const binding = self.shortconv_binding orelse return error.UnsupportedModel;
        self.instruction_shortconv_exec_meta[op_index] = .{
            .conv_dim = binding.conv_dim,
            .d_conv = binding.d_conv,
        };
        self.instruction_shortconv_weight_slots[op_index] = .{
            .in_proj = &binding.in_proj,
            .conv_weight = &binding.conv_weight_time_major,
            .out_proj = &binding.out_proj,
            .conv_bias = if (binding.conv_bias) |*weight| weight else null,
        };
    }

    pub fn bindInstructionGatedDelta(
        self: *BlockRuntimeLayer,
        compiled: *const runtime_contract.CompiledPlan,
        op_index: usize,
        insn: *const runtime_contract.Instruction,
        _: *usize,
    ) !void {
        _ = try instructionKernelIdFromWeightBindings(compiled, op_index, insn.opcode);
        const binding = self.gated_delta_binding orelse return error.UnsupportedModel;
        self.instruction_gated_delta_weight_slots[op_index] = .{
            .in_proj = binding.kernel.weights.in_proj,
            .conv_weight = binding.kernel.weights.conv1d_weight,
            .a_log = binding.kernel.weights.A_log,
            .out_proj = binding.kernel.weights.out_proj,
            .conv_bias = binding.kernel.weights.conv1d_bias,
            .dt_bias = binding.kernel.weights.dt_bias,
            .norm_weight = binding.kernel.weights.norm_weight,
        };
    }

    pub fn bindInstructionSwiGlu(
        self: *BlockRuntimeLayer,
        compiled: *const runtime_contract.CompiledPlan,
        op_index: usize,
        insn: *const runtime_contract.Instruction,
        _: *usize,
    ) !void {
        _ = try instructionKernelIdFromWeightBindings(compiled, op_index, insn.opcode);
        if (self.attention_binding) |binding| {
            self.instruction_swiglu_weight_slots[op_index] = .{
                .w1 = &binding.w1,
                .w3 = &binding.w3,
                .w2 = &binding.w2,
                .w1_bias = null,
                .w2_bias = null,
            };
            return;
        }
        if (self.shortconv_binding) |binding| {
            self.instruction_swiglu_weight_slots[op_index] = .{
                .w1 = if (binding.ffn_w1) |*w| w else null,
                .w3 = if (binding.ffn_w3) |*w| w else null,
                .w2 = if (binding.ffn_w2) |*w| w else null,
                .w1_bias = null,
                .w2_bias = null,
            };
            return;
        }
        if (self.gated_delta_binding) |binding| {
            self.instruction_swiglu_weight_slots[op_index] = .{
                .w1 = if (binding.ffn_w1) |*w| w else null,
                .w3 = if (binding.ffn_w3) |*w| w else null,
                .w2 = if (binding.ffn_w2) |*w| w else null,
                .w1_bias = null,
                .w2_bias = null,
            };
            return;
        }
        return error.UnsupportedModel;
    }

    pub fn bindInstructionMoE(
        self: *BlockRuntimeLayer,
        compiled: *const runtime_contract.CompiledPlan,
        op_index: usize,
        insn: *const runtime_contract.Instruction,
        _: *usize,
    ) !void {
        _ = try instructionKernelIdFromWeightBindings(compiled, op_index, insn.opcode);
        const binding = self.moe_binding orelse return error.UnsupportedModel;
        self.instruction_moe_weight_slots[op_index] = binding;
    }

    const instruction_rebind_table: [256]?InstructionRefBinderFn = blk: {
        var table: [256]?InstructionRefBinderFn = [_]?InstructionRefBinderFn{bindInstructionNoop} ** 256;
        table[@intFromEnum(runtime_contract.Opcode.rmsnorm)] = bindInstructionRmsNorm;
        table[@intFromEnum(runtime_contract.Opcode.multihead_attention)] = bindInstructionAttention;
        table[@intFromEnum(runtime_contract.Opcode.mla_attention)] = bindInstructionAttention;
        table[@intFromEnum(runtime_contract.Opcode.gated_delta_net)] = bindInstructionGatedDelta;
        table[@intFromEnum(runtime_contract.Opcode.shortconv)] = bindInstructionShortConv;
        table[@intFromEnum(runtime_contract.Opcode.swiglu)] = bindInstructionSwiGlu;
        table[@intFromEnum(runtime_contract.Opcode.moe)] = bindInstructionMoE;
        break :blk table;
    };

    pub fn rebuildInstructionMetadata(self: *BlockRuntimeLayer, allocator: std.mem.Allocator) !void {
        if (self.instruction_norm_weight_slots.len != 0) {
            allocator.free(self.instruction_norm_weight_slots);
            self.instruction_norm_weight_slots = &.{};
        }
        if (self.instruction_attention_exec_meta.len != 0) {
            allocator.free(self.instruction_attention_exec_meta);
            self.instruction_attention_exec_meta = &.{};
        }
        if (self.instruction_attention_weight_slots.len != 0) {
            allocator.free(self.instruction_attention_weight_slots);
            self.instruction_attention_weight_slots = &.{};
        }
        if (self.instruction_shortconv_exec_meta.len != 0) {
            allocator.free(self.instruction_shortconv_exec_meta);
            self.instruction_shortconv_exec_meta = &.{};
        }
        if (self.instruction_shortconv_weight_slots.len != 0) {
            allocator.free(self.instruction_shortconv_weight_slots);
            self.instruction_shortconv_weight_slots = &.{};
        }
        if (self.instruction_gated_delta_weight_slots.len != 0) {
            allocator.free(self.instruction_gated_delta_weight_slots);
            self.instruction_gated_delta_weight_slots = &.{};
        }
        if (self.instruction_swiglu_weight_slots.len != 0) {
            allocator.free(self.instruction_swiglu_weight_slots);
            self.instruction_swiglu_weight_slots = &.{};
        }
        if (self.instruction_moe_weight_slots.len != 0) {
            allocator.free(self.instruction_moe_weight_slots);
            self.instruction_moe_weight_slots = &.{};
        }
        if (self.instruction_weight_offsets.len != 0) {
            allocator.free(self.instruction_weight_offsets);
            self.instruction_weight_offsets = &.{};
        }
        if (self.instruction_weight_ptrs.len != 0) {
            allocator.free(self.instruction_weight_ptrs);
            self.instruction_weight_ptrs = &.{};
        }

        const compiled = self.compiled_plan orelse return;
        const len = compiled.plan.instructions.len;
        self.instruction_norm_weight_slots = try allocator.alloc(?*const DeviceTensor, len);
        self.instruction_attention_exec_meta = try allocator.alloc(?LayerAttentionExecConfig, len);
        self.instruction_attention_weight_slots = try allocator.alloc(?AttentionWeightRefs, len);
        self.instruction_shortconv_exec_meta = try allocator.alloc(?ShortConvExecConfig, len);
        self.instruction_shortconv_weight_slots = try allocator.alloc(?ShortConvWeightRefs, len);
        self.instruction_gated_delta_weight_slots = try allocator.alloc(?GatedDeltaWeightRefs, len);
        self.instruction_swiglu_weight_slots = try allocator.alloc(?SwiGluWeightRefs, len);
        self.instruction_moe_weight_slots = try allocator.alloc(?*const MoEWeightRefs, len);
        @memset(self.instruction_norm_weight_slots, null);
        @memset(self.instruction_attention_exec_meta, null);
        @memset(self.instruction_attention_weight_slots, null);
        @memset(self.instruction_shortconv_exec_meta, null);
        @memset(self.instruction_shortconv_weight_slots, null);
        @memset(self.instruction_gated_delta_weight_slots, null);
        @memset(self.instruction_swiglu_weight_slots, null);
        @memset(self.instruction_moe_weight_slots, null);

        var norm_index: usize = 0;
        for (compiled.plan.instructions, 0..) |insn, op_index| {
            const binder = instruction_rebind_table[@intFromEnum(insn.opcode)] orelse continue;
            try binder(self, &compiled, op_index, &insn, &norm_index);
        }
        try self.buildInstructionWeightTable(allocator, &compiled);
    }

    pub fn resolveInstructionWeightPtrForSlot(
        self: *const BlockRuntimeLayer,
        opcode: runtime_contract.Opcode,
        op_index: usize,
        slot_idx: usize,
    ) !*anyopaque {
        switch (opcode) {
            .rmsnorm => {
                return switch (slot_idx) {
                    0 => blk: {
                        const weight = try self.instructionNormWeightRef(op_index);
                        break :blk @ptrCast(@constCast(weight));
                    },
                    1 => @ptrCast(@constCast(&missing_device_tensor)),
                    else => error.InvalidWeightRefCount,
                };
            },
            .multihead_attention => {
                if (op_index >= self.instruction_attention_weight_slots.len) return error.InvalidInstructionIndex;
                const binding = self.instruction_attention_weight_slots[op_index] orelse return error.UnsupportedModel;
                return switch (slot_idx) {
                    0 => @ptrCast(@constCast(binding.q_proj orelse return error.MissingWeight)),
                    1 => @ptrCast(@constCast(binding.k_proj orelse return error.MissingWeight)),
                    2 => @ptrCast(@constCast(binding.v_proj orelse return error.MissingWeight)),
                    3 => @ptrCast(@constCast(binding.o_proj orelse return error.MissingWeight)),
                    4 => if (binding.q_norm_weight) |q_norm|
                        @ptrCast(@constCast(q_norm))
                    else
                        @ptrCast(@constCast(&missing_device_tensor)),
                    5 => if (binding.k_norm_weight) |k_norm|
                        @ptrCast(@constCast(k_norm))
                    else
                        @ptrCast(@constCast(&missing_device_tensor)),
                    6 => @ptrCast(@constCast(&missing_device_tensor)),
                    7 => @ptrCast(@constCast(&missing_device_tensor)),
                    8 => @ptrCast(@constCast(&missing_device_tensor)),
                    9 => @ptrCast(@constCast(&missing_device_tensor)),
                    10 => @ptrCast(@constCast(&missing_device_tensor)),
                    else => error.InvalidWeightRefCount,
                };
            },
            .mla_attention => {
                return error.UnsupportedModel;
            },
            .mamba_mixer => {
                return error.UnsupportedModel;
            },
            .gated_delta_net => {
                if (op_index >= self.instruction_gated_delta_weight_slots.len) return error.InvalidInstructionIndex;
                const binding = self.instruction_gated_delta_weight_slots[op_index] orelse return error.UnsupportedModel;
                const missing_tensor = &missing_host_tensor;
                return switch (slot_idx) {
                    0 => @ptrCast(@constCast(binding.in_proj orelse missing_tensor)),
                    1 => @ptrCast(@constCast(binding.conv_weight orelse missing_tensor)),
                    2 => @ptrCast(@constCast(binding.a_log orelse missing_tensor)),
                    3 => @ptrCast(@constCast(binding.out_proj orelse missing_tensor)),
                    4 => @ptrCast(@constCast(binding.conv_bias orelse missing_tensor)),
                    5 => @ptrCast(@constCast(binding.dt_bias orelse missing_tensor)),
                    6 => @ptrCast(@constCast(binding.norm_weight orelse missing_tensor)),
                    else => error.InvalidWeightRefCount,
                };
            },
            .swiglu => {
                if (op_index >= self.instruction_swiglu_weight_slots.len) return error.InvalidInstructionIndex;
                const binding = self.instruction_swiglu_weight_slots[op_index] orelse return error.UnsupportedModel;
                return switch (slot_idx) {
                    0 => @ptrCast(@constCast(binding.w1 orelse return error.MissingWeight)),
                    1 => @ptrCast(@constCast(binding.w3 orelse return error.MissingWeight)),
                    2 => @ptrCast(@constCast(binding.w2 orelse return error.MissingWeight)),
                    3 => if (binding.w1_bias) |w1_bias|
                        @ptrCast(@constCast(w1_bias))
                    else
                        @ptrCast(@constCast(&missing_device_tensor)),
                    4 => if (binding.w2_bias) |w2_bias|
                        @ptrCast(@constCast(w2_bias))
                    else
                        @ptrCast(@constCast(&missing_device_tensor)),
                    else => error.InvalidWeightRefCount,
                };
            },
            .moe => {
                // MoE weights are accessed directly via MoEWeightRefs in the adapter,
                // not through the weight handle system. Return placeholders for plan validation.
                return @ptrCast(@constCast(&missing_device_tensor));
            },
            .shortconv => {
                if (op_index >= self.instruction_shortconv_weight_slots.len) return error.InvalidInstructionIndex;
                const binding = self.instruction_shortconv_weight_slots[op_index] orelse return error.UnsupportedModel;
                return switch (slot_idx) {
                    0 => @ptrCast(@constCast(binding.in_proj orelse return error.MissingWeight)),
                    1 => @ptrCast(@constCast(binding.conv_weight orelse return error.MissingWeight)),
                    2 => @ptrCast(@constCast(binding.out_proj orelse return error.MissingWeight)),
                    3 => if (binding.conv_bias) |conv_bias|
                        @ptrCast(@constCast(conv_bias))
                    else
                        @ptrCast(@constCast(&missing_device_tensor)),
                    else => error.InvalidWeightRefCount,
                };
            },
            else => return error.InvalidInstructionBinding,
        }
    }

    pub fn resolveInstructionWeightPtr(
        self: *const BlockRuntimeLayer,
        opcode: runtime_contract.Opcode,
        op_index: usize,
        slot_name: []const u8,
        slot_idx: usize,
    ) !*anyopaque {
        const expected_slots = runtime_contract.expectedKernelWeightSlots(opcode);
        if (slot_idx >= expected_slots.len) return error.InvalidWeightRefCount;
        if (!std.mem.eql(u8, expected_slots[slot_idx], slot_name)) return error.InvalidWeightBindingName;
        return self.resolveInstructionWeightPtrForSlot(opcode, op_index, slot_idx);
    }

    pub fn buildInstructionWeightTable(
        self: *BlockRuntimeLayer,
        allocator: std.mem.Allocator,
        compiled: *const runtime_contract.CompiledPlan,
    ) !void {
        const insn_len = compiled.plan.instructions.len;
        const offsets = try allocator.alloc(u32, insn_len + 1);
        errdefer allocator.free(offsets);

        var total_slots: usize = 0;
        for (compiled.plan.instructions) |insn| total_slots += insn.weights.len;
        const ptrs = try allocator.alloc(?*anyopaque, total_slots);
        errdefer allocator.free(ptrs);
        @memset(ptrs, null);

        var cursor: usize = 0;
        for (compiled.plan.instructions, 0..) |insn, op_index| {
            offsets[op_index] = @intCast(cursor);
            const expected_slots = runtime_contract.expectedKernelWeightSlots(insn.opcode);
            if (insn.weights.len != expected_slots.len) return error.InvalidWeightRefCount;
            for (insn.weights, 0..) |_, slot_idx| {
                const parsed = try runtime_contract.instructionKernelWeightBinding(
                    compiled,
                    op_index,
                    insn.opcode,
                    slot_idx,
                );
                const weight_ptr = try self.resolveInstructionWeightPtr(insn.opcode, op_index, parsed.slot_name, slot_idx);
                ptrs[cursor] = weight_ptr;
                cursor += 1;
            }
        }
        offsets[insn_len] = @intCast(cursor);
        self.instruction_weight_offsets = offsets;
        self.instruction_weight_ptrs = ptrs;
    }

    pub fn instructionNormWeightRef(self: *const BlockRuntimeLayer, op_index: usize) !*const DeviceTensor {
        if (op_index >= self.instruction_norm_weight_slots.len) return error.InvalidInstructionIndex;
        return self.instruction_norm_weight_slots[op_index] orelse return error.UnsupportedModel;
    }

    pub fn instructionAttentionRef(self: *const BlockRuntimeLayer, op_index: usize) !*const LayerAttentionExecConfig {
        if (op_index >= self.instruction_attention_exec_meta.len) return error.InvalidInstructionIndex;
        if (self.instruction_attention_exec_meta[op_index]) |*binding| return binding;
        return error.UnsupportedModel;
    }

    pub fn instructionShortConvRef(self: *const BlockRuntimeLayer, op_index: usize) !*const ShortConvExecConfig {
        if (op_index >= self.instruction_shortconv_exec_meta.len) return error.InvalidInstructionIndex;
        if (self.instruction_shortconv_exec_meta[op_index]) |*binding| return binding;
        return error.UnsupportedModel;
    }

    pub fn appendLayerNormWeight(layer: *BlockRuntimeLayer, weight: ?*const DeviceTensor) void {
        const value = weight orelse return;
        if (layer.norm_weight_count >= layer.norm_weights.len) return;
        layer.norm_weights[layer.norm_weight_count] = value;
        layer.norm_weight_count += 1;
    }

    pub fn bindAttentionNormWeights(layer: *BlockRuntimeLayer, block: *const LayerAttentionRuntime) void {
        appendLayerNormWeight(layer, &block.ln1_weight);
        appendLayerNormWeight(layer, &block.ln2_weight);
        if (block.pre_ffn_norm_weight) |*weight| {
            appendLayerNormWeight(layer, weight);
        } else if (block.post_ffn_norm_weight) |*weight| {
            appendLayerNormWeight(layer, weight);
        }
        if (block.post_ffn_norm_weight) |*weight| {
            appendLayerNormWeight(layer, weight);
        }
    }

    pub fn bindShortConvNormWeights(layer: *BlockRuntimeLayer, block: *const ShortConvBlockRuntime) void {
        appendLayerNormWeight(layer, &block.ln1_weight);
        if (block.ln2_weight) |*weight| {
            appendLayerNormWeight(layer, weight);
        }
    }

    pub fn bindGatedDeltaNormWeights(layer: *BlockRuntimeLayer, block: *const GatedDeltaBlockRuntime) void {
        appendLayerNormWeight(layer, &block.ln1_weight);
        if (block.ln2_weight) |*weight| {
            appendLayerNormWeight(layer, weight);
        }
    }

    pub fn deinit(self: *BlockRuntimeLayer, allocator: std.mem.Allocator, device: *compute.cuda.Device) void {
        if (self.register_to_slot_map.len != 0) allocator.free(self.register_to_slot_map);
        if (self.slot_width_hints.len != 0) allocator.free(self.slot_width_hints);
        if (self.instruction_norm_weight_slots.len != 0) allocator.free(self.instruction_norm_weight_slots);
        if (self.instruction_attention_exec_meta.len != 0) allocator.free(self.instruction_attention_exec_meta);
        if (self.instruction_attention_weight_slots.len != 0) allocator.free(self.instruction_attention_weight_slots);
        if (self.instruction_shortconv_exec_meta.len != 0) allocator.free(self.instruction_shortconv_exec_meta);
        if (self.instruction_shortconv_weight_slots.len != 0) allocator.free(self.instruction_shortconv_weight_slots);
        if (self.instruction_gated_delta_weight_slots.len != 0) allocator.free(self.instruction_gated_delta_weight_slots);
        if (self.instruction_swiglu_weight_slots.len != 0) allocator.free(self.instruction_swiglu_weight_slots);
        if (self.instruction_moe_weight_slots.len != 0) allocator.free(self.instruction_moe_weight_slots);
        if (self.instruction_weight_offsets.len != 0) allocator.free(self.instruction_weight_offsets);
        if (self.instruction_weight_ptrs.len != 0) allocator.free(self.instruction_weight_ptrs);
        if (self.compiled_plan) |*compiled_plan| {
            plan_compiler.deinitCompiledPlan(allocator, compiled_plan);
            self.compiled_plan = null;
        }
        if (self.attention_runtime) |*block| block.deinit(device);
        if (self.shortconv_runtime) |*block| block.deinit(device);
        if (self.gated_delta_runtime) |*block| block.deinit(allocator, device);
        if (self.moe_runtime) |*moe| moe.deinit(allocator, device);
        self.* = .{};
    }
};

pub fn buildCudaLayerProgramRegisterSlotMap(
    allocator: std.mem.Allocator,
    compiled: *const runtime_contract.CompiledPlan,
) ![]u8 {
    const invalid_slot = BlockRuntimeLayer.invalid_slot;
    const register_count = compiled.plan.register_count;
    const register_to_slot = try allocator.alloc(u8, register_count);
    @memset(register_to_slot, invalid_slot);
    errdefer allocator.free(register_to_slot);
    if (register_count <= 1) return register_to_slot;

    const register_specs = try allocator.alloc(runtime_contract.RegisterBufferSpec, register_count);
    defer allocator.free(register_specs);
    // Register 0 (residual) uses runtime_buffers.input_dev, not a slot buffer.
    register_specs[0] = .{ .size = 0, .@"align" = 0 };
    if (compiled.register_buffer_specs.len != register_count) return error.InvalidRegisterSpecCount;
    // Plan specs already contain floors applied at compile time.
    // Backends consume specs exactly.
    for (register_specs[1..], 1..) |*spec, idx| {
        const plan_spec = compiled.register_buffer_specs[idx];
        spec.* = .{
            .size = plan_spec.size,
            .@"align" = plan_spec.@"align",
            .dtype = plan_spec.dtype,
            .layout = plan_spec.layout,
        };
    }

    var physical = try runtime_contract.buildPhysicalMappingLinearScan(allocator, compiled, register_specs);
    defer runtime_contract.deinitPhysicalMapping(allocator, &physical);
    if (physical.physical_count == 0) return register_to_slot;

    const physical_to_slot = try allocator.alloc(u8, physical.physical_count);
    defer allocator.free(physical_to_slot);
    @memset(physical_to_slot, invalid_slot);

    var next_slot: u8 = 0;
    const invalid_physical = std.math.maxInt(u16);
    var register_idx: usize = 0;
    while (register_idx < register_count) : (register_idx += 1) {
        const physical_id_u16 = physical.register_to_physical[register_idx];
        if (physical_id_u16 == invalid_physical) continue;
        const physical_id: usize = physical_id_u16;
        if (physical_id >= physical_to_slot.len) return error.UnsupportedModel;
        if (physical_to_slot[physical_id] == invalid_slot) {
            physical_to_slot[physical_id] = next_slot;
            next_slot += 1;
        }
        register_to_slot[register_idx] = physical_to_slot[physical_id];
    }

    return register_to_slot;
}

pub fn buildCudaLayerProgramSlotWidthHints(
    allocator: std.mem.Allocator,
    compiled: *const runtime_contract.CompiledPlan,
    register_to_slot_map: []const u8,
) ![]usize {
    const invalid_slot = BlockRuntimeLayer.invalid_slot;
    const register_count: usize = compiled.plan.register_count;
    if (register_to_slot_map.len != register_count) return error.InvalidRegisterSpecCount;
    if (compiled.register_buffer_specs.len != register_count) return error.InvalidRegisterSpecCount;

    var required_slots: usize = 0;
    for (register_to_slot_map) |slot_idx| {
        if (slot_idx == invalid_slot) continue;
        const next = @as(usize, slot_idx) + 1;
        if (next > required_slots) required_slots = next;
    }
    if (required_slots == 0) return &.{};

    const slot_width_hints = try allocator.alloc(usize, required_slots);
    @memset(slot_width_hints, 0);
    errdefer allocator.free(slot_width_hints);

    const register_specs = try allocator.alloc(runtime_contract.RegisterBufferSpec, register_count);
    defer allocator.free(register_specs);
    register_specs[0] = .{ .size = 0, .@"align" = 0 };
    for (register_specs[1..], 1..) |*spec, idx| {
        const plan_spec = compiled.register_buffer_specs[idx];
        spec.* = .{
            .size = plan_spec.size,
            .@"align" = plan_spec.@"align",
            .dtype = plan_spec.dtype,
            .layout = plan_spec.layout,
        };
    }
    var physical = try runtime_contract.buildPhysicalMappingLinearScan(allocator, compiled, register_specs);
    defer runtime_contract.deinitPhysicalMapping(allocator, &physical);

    const invalid_physical = std.math.maxInt(u16);
    for (0..register_count) |reg_idx| {
        const slot_idx = register_to_slot_map[reg_idx];
        if (slot_idx == invalid_slot) continue;
        const physical_id_u16 = physical.register_to_physical[reg_idx];
        if (physical_id_u16 == invalid_physical) continue;
        const physical_id: usize = physical_id_u16;
        const width = physical.physical_specs[physical_id].size;
        if (width == 0) return error.InvalidRegisterSpecSize;
        const slot_usize: usize = @intCast(slot_idx);
        if (slot_width_hints[slot_usize] == 0) {
            slot_width_hints[slot_usize] = width;
        } else if (slot_width_hints[slot_usize] != width) {
            return error.InvalidRegisterSpecSize;
        }
    }
    for (slot_width_hints) |width| {
        if (width == 0) return error.InvalidRegisterSpecSize;
    }
    return slot_width_hints;
}

pub fn validateCompiledLayerPlanForCuda(
    compiled: *const runtime_contract.CompiledPlan,
    layer_idx: usize,
    kind: op_types.BlockKind,
    adapter_table: anytype,
) !void {
    runtime_contract.validateExecutionPlanForBlockKind(&compiled.plan, kind) catch |err| {
        log.warn("inference", "CUDA compiled layer plan fails block-kind validation", .{
            .layer = layer_idx,
            .kind = @intFromEnum(kind),
            .reason = @errorName(err),
        });
        return error.UnsupportedModel;
    };
    if (runtime_contract.firstUnsupportedInstructionOpcode(&compiled.plan, adapter_table)) |unsupported| {
        log.warn("inference", "CUDA compiled layer plan contains unsupported opcode", .{
            .layer = layer_idx,
            .kind = @intFromEnum(kind),
            .op_index = unsupported.instruction_index,
            .opcode = @intFromEnum(unsupported.opcode),
        });
        return error.UnsupportedModel;
    }
}

/// Describes a CPU source layer whose KV cache is replicated to a mirror
/// entry on the GPU. Used when cpu_gpu topology places KV-shared source
/// layers on CPU while consumer layers run on GPU.
pub const ReplicatedKvSource = struct {
    /// Global (model-wide) layer index of the CPU source layer.
    global_layer_idx: usize,
    /// KV dimension (= n_kv_heads * head_dim) for this source.
    kv_dim: usize,
    /// Index into per-slot kv[] array for the mirror entry on GPU.
    mirror_kv_index: usize,
};

/// GPU-side mirror buffers for a replicated CPU KV source layer.
pub const MirrorKvBuffers = struct {
    k: compute.cuda.Buffer,
    v: compute.cuda.Buffer,
    k_scale: compute.cuda.Buffer = .{ .pointer = 0, .size = 0 },
    v_scale: compute.cuda.Buffer = .{ .pointer = 0, .size = 0 },
    capacity: usize,
};

pub const BlockRuntime = struct {
    blocks: []BlockRuntimeLayer,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    attention_block_count: usize,
    shortconv_block_count: usize,
    gated_delta_block_count: usize,
    q_norm_blocks: usize,
    k_norm_blocks: usize,
    linear_weight_bytes: usize,
    norm_weight_bytes: usize,
    kv_cache_bytes: usize,
    shortconv_state_bytes: usize,
    gated_delta_state_bytes: usize,
    max_shortconv_dim: usize,
    max_gdelta_proj: usize,
    /// CPU source layers whose KV is replicated to GPU mirror entries.
    replicated_kv_sources: []ReplicatedKvSource = &.{},
    /// GPU-side mirror KV buffers for slot 0. loadKvSlot syncs these from
    /// slot_kv_states for the active slot. Indexed by mirror offset
    /// (mirror_kv_index - attention_block_count).
    mirror_kv: []MirrorKvBuffers = &.{},

    pub fn init(
        allocator: std.mem.Allocator,
        device: *compute.cuda.Device,
        loaded: *const LoadedModel,
        max_seq_len: usize,
        kv_init_tokens: usize,
        gated_delta_ssm_i8_state: bool,
        adapter_table: anytype,
        kv_cache_dtype: KvCacheDtype,
    ) !BlockRuntime {
        return initRange(
            allocator,
            device,
            loaded,
            max_seq_len,
            kv_init_tokens,
            0,
            loaded.blocks.len,
            gated_delta_ssm_i8_state,
            adapter_table,
            kv_cache_dtype,
        );
    }

    /// Initialize a BlockRuntime for a contiguous range of decoder layers
    /// [layer_start, layer_end). Used by pipeline parallel to split layers
    /// across devices.
    pub fn initRange(
        allocator: std.mem.Allocator,
        device: *compute.cuda.Device,
        loaded: *const LoadedModel,
        max_seq_len: usize,
        kv_init_tokens: usize,
        layer_start: usize,
        layer_end: usize,
        gated_delta_ssm_i8_state: bool,
        adapter_table: anytype,
        kv_cache_dtype: KvCacheDtype,
    ) !BlockRuntime {
        const d_model: usize = @intCast(loaded.config.d_model);
        const n_heads: usize = @intCast(loaded.config.n_heads);
        const n_kv_heads: usize = @intCast(loaded.config.n_kv_groups);
        const head_dim: usize = @intCast(loaded.config.head_dim);
        if (n_heads == 0 or n_kv_heads == 0 or head_dim == 0 or max_seq_len == 0) return error.InvalidArgument;
        if (n_heads % n_kv_heads != 0) return error.UnsupportedModel;
        if (layer_end > loaded.blocks.len or layer_start > layer_end) return error.InvalidArgument;
        const initial_kv_tokens = @min(max_seq_len, @max(@as(usize, 1), kv_init_tokens));
        const arena_allocator = @constCast(&loaded.arena).allocator();
        const static_entry = if (loaded.runtime.architecture_id) |arch_id|
            models.registry.detectByArchitectureId(arch_id)
        else
            null;
        const layer_count = layer_end - layer_start;
        var attention_block_count: usize = 0;
        var shortconv_block_count: usize = 0;
        var gated_delta_block_count: usize = 0;
        var q_norm_blocks: usize = 0;
        var k_norm_blocks: usize = 0;
        var linear_weight_bytes: usize = 0;
        var norm_weight_bytes: usize = 0;
        var kv_cache_bytes: usize = 0;
        var shortconv_state_bytes: usize = 0;
        var gated_delta_state_bytes: usize = 0;
        var max_shortconv_dim: usize = 0;

        // Track cross-device KV sharing references for mirror replication.
        const PendingMirror = struct { local_idx: usize, source_global: usize, kv_dim: usize };
        var pending_mirrors: std.ArrayListUnmanaged(PendingMirror) = .{};
        defer pending_mirrors.deinit(allocator);
        var max_gdelta_proj: usize = 0;
        var blocks = try allocator.alloc(BlockRuntimeLayer, layer_count);
        errdefer allocator.free(blocks);
        for (blocks) |*layer| layer.* = .{};

        var initialized: usize = 0;
        errdefer {
            while (initialized > 0) {
                initialized -= 1;
                blocks[initialized].deinit(allocator, device);
            }
        }
        const layer_blocks = loaded.blocks[layer_start..layer_end];
        for (layer_blocks, 0..) |*layer_weights, local_idx| {
            const layer_idx = layer_start + local_idx;
            const block_weights = try models.runtime_blocks.layerToBlockWeights(arena_allocator, layer_weights);
            switch (block_weights) {
                .attention_mlp => |attn| {
                    const entry = static_entry orelse {
                        log.warn("inference", "CUDA block runtime missing architecture metadata", .{ .layer = layer_idx });
                        return error.UnsupportedModel;
                    };
                    const program = models.registry.blockProgramFor(entry, .attention_mlp) orelse {
                        log.warn("inference", "CUDA block runtime missing LayerOp program", .{
                            .layer = layer_idx,
                            .kind = @intFromEnum(op_types.BlockKind.attention_mlp),
                            .architecture = entry.id,
                        });
                        return error.UnsupportedModel;
                    };
                    blocks[local_idx].compiled_plan = try plan_compiler.compileLayerProgram(
                        allocator,
                        program,
                        .decode,
                        .{
                            .size_floor = d_model,
                            .state_descriptor_entry = entry,
                        },
                    );
                    try validateCompiledLayerPlanForCuda(&blocks[local_idx].compiled_plan.?, layer_idx, .attention_mlp, adapter_table);
                    errdefer if (blocks[local_idx].compiled_plan) |*compiled_plan| {
                        plan_compiler.deinitCompiledPlan(allocator, compiled_plan);
                        blocks[local_idx].compiled_plan = null;
                    };
                    blocks[local_idx].register_to_slot_map = try buildCudaLayerProgramRegisterSlotMap(
                        allocator,
                        &blocks[local_idx].compiled_plan.?,
                    );
                    errdefer if (blocks[local_idx].register_to_slot_map.len != 0) {
                        allocator.free(blocks[local_idx].register_to_slot_map);
                        blocks[local_idx].register_to_slot_map = &.{};
                    };
                    blocks[local_idx].slot_width_hints = try buildCudaLayerProgramSlotWidthHints(
                        allocator,
                        &blocks[local_idx].compiled_plan.?,
                        blocks[local_idx].register_to_slot_map,
                    );
                    errdefer if (blocks[local_idx].slot_width_hints.len != 0) {
                        allocator.free(blocks[local_idx].slot_width_hints);
                        blocks[local_idx].slot_width_hints = &.{};
                    };
                    if (attn.mla_config != null) {
                        log.warn("inference", "CUDA block runtime MLA not supported yet", .{ .layer = layer_idx });
                        return error.UnsupportedModel;
                    }
                    const has_moe = attn.moe_weights != null;
                    const w2 = if (!has_moe) (attn.w2 orelse return error.MissingWeight) else null;
                    const q_proj_src = attn.q_proj orelse return error.MissingWeight;
                    const k_proj_src = attn.k_proj orelse return error.MissingWeight;
                    const v_proj_src = attn.v_proj orelse return error.MissingWeight;
                    const q_proj_out = try tensorProjectionOutputDim(q_proj_src, d_model);
                    const q_out = if (attn.attention_config.query_gate) blk: {
                        if ((q_proj_out % 2) != 0) {
                            log.warn("inference", "CUDA block runtime q_proj dim unsupported for query_gate", .{
                                .layer = layer_idx,
                                .q_proj_out = q_proj_out,
                            });
                            return error.UnsupportedModel;
                        }
                        break :blk q_proj_out / 2;
                    } else q_proj_out;
                    const kv_out = try tensorProjectionOutputDim(k_proj_src, d_model);
                    const v_out = try tensorProjectionOutputDim(v_proj_src, d_model);
                    if (v_out != kv_out) {
                        log.warn("inference", "CUDA block runtime k/v dim mismatch", .{
                            .layer = layer_idx,
                            .k_cols = kv_out,
                            .v_cols = v_out,
                        });
                        return error.UnsupportedModel;
                    }
                    var layer_head_dim = head_dim;
                    if (attn.q_norm) |q_norm_src| {
                        const q_norm_dim: usize = switch (q_norm_src.n_dims) {
                            1 => @intCast(q_norm_src.shape[0]),
                            2 => if (q_norm_src.shape[0] == 1 and q_norm_src.shape[1] > 0)
                                @intCast(q_norm_src.shape[1])
                            else if (q_norm_src.shape[1] == 1 and q_norm_src.shape[0] > 0)
                                @intCast(q_norm_src.shape[0])
                            else
                                return error.UnsupportedModel,
                            else => return error.UnsupportedModel,
                        };
                        if (q_norm_dim == 0) return error.UnsupportedModel;
                        layer_head_dim = q_norm_dim;
                    } else if (attn.k_norm) |k_norm_src| {
                        const k_norm_dim: usize = switch (k_norm_src.n_dims) {
                            1 => @intCast(k_norm_src.shape[0]),
                            2 => if (k_norm_src.shape[0] == 1 and k_norm_src.shape[1] > 0)
                                @intCast(k_norm_src.shape[1])
                            else if (k_norm_src.shape[1] == 1 and k_norm_src.shape[0] > 0)
                                @intCast(k_norm_src.shape[0])
                            else
                                return error.UnsupportedModel,
                            else => return error.UnsupportedModel,
                        };
                        if (k_norm_dim == 0) return error.UnsupportedModel;
                        layer_head_dim = k_norm_dim;
                    } else if ((q_out % n_heads) == 0 and n_heads > 0) {
                        layer_head_dim = q_out / n_heads;
                    } else if ((kv_out % n_kv_heads) == 0 and n_kv_heads > 0) {
                        layer_head_dim = kv_out / n_kv_heads;
                    }
                    if (layer_head_dim == 0) return error.UnsupportedModel;
                    const layer_n_heads = if ((q_out % layer_head_dim) == 0) q_out / layer_head_dim else n_heads;
                    const layer_n_kv_heads = if ((kv_out % layer_head_dim) == 0) kv_out / layer_head_dim else n_kv_heads;
                    if (layer_n_heads == 0 or layer_n_kv_heads == 0 or (layer_n_heads % layer_n_kv_heads) != 0) {
                        log.warn("inference", "CUDA block runtime inferred attention shape unsupported", .{
                            .layer = layer_idx,
                            .q_dim = q_out,
                            .kv_dim = kv_out,
                            .head_dim = layer_head_dim,
                            .n_heads = layer_n_heads,
                            .n_kv_heads = layer_n_kv_heads,
                        });
                        return error.UnsupportedModel;
                    }
                    if (attention_block_count == 0) {
                        if (has_moe) {
                            const moe = attn.moe_weights.?;
                            log.info("inference", "CUDA block0 MoE weight mode", .{
                                .num_experts = moe.num_experts,
                                .experts_per_token = moe.experts_per_token,
                                .q_out = q_out,
                                .q_proj_out = q_proj_out,
                                .kv_out = kv_out,
                                .head_dim = layer_head_dim,
                                .n_heads = layer_n_heads,
                                .n_kv_heads = layer_n_kv_heads,
                            });
                        } else if (attn.fused.qkv_proj != null or attn.fused.gate_up != null) {
                            log.info("inference", "CUDA block0 fused weight mode", .{
                                .qkv_fused = @as(u8, @intFromBool(attn.fused.qkv_proj != null)),
                                .gate_up_fused = @as(u8, @intFromBool(attn.fused.gate_up != null)),
                                .gate_up_layout = @tagName(attn.fused.gate_up_layout),
                                .qkv_dtype = if (attn.fused.qkv_proj) |qkv| @tagName(qkv.dtype) else "none",
                                .gate_up_dtype = if (attn.fused.gate_up) |gate_up| @tagName(gate_up.dtype) else "none",
                                .w2_dtype = @tagName(w2.?.dtype),
                                .q_out = q_out,
                                .q_proj_out = q_proj_out,
                                .kv_out = kv_out,
                                .head_dim = layer_head_dim,
                                .n_heads = layer_n_heads,
                                .n_kv_heads = layer_n_kv_heads,
                            });
                        } else {
                            const w1 = attn.w1 orelse return error.MissingWeight;
                            const w3 = attn.w3 orelse return error.MissingWeight;
                            log.info("inference", "CUDA block0 weight dtypes", .{
                                .q_proj = @tagName(q_proj_src.dtype),
                                .k_proj = @tagName(k_proj_src.dtype),
                                .v_proj = @tagName(v_proj_src.dtype),
                                .o_proj = @tagName(attn.o_proj.dtype),
                                .w1 = @tagName(w1.dtype),
                                .w2 = @tagName(w2.?.dtype),
                                .w3 = @tagName(w3.dtype),
                            });
                            log.info("inference", "CUDA block0 weight shapes", .{
                                .q0 = q_proj_src.shape[0],
                                .q1 = q_proj_src.shape[1],
                                .k0 = k_proj_src.shape[0],
                                .k1 = k_proj_src.shape[1],
                                .v0 = v_proj_src.shape[0],
                                .v1 = v_proj_src.shape[1],
                                .o0 = attn.o_proj.shape[0],
                                .o1 = attn.o_proj.shape[1],
                                .w10 = w1.shape[0],
                                .w11 = w1.shape[1],
                                .w20 = w2.?.shape[0],
                                .w21 = w2.?.shape[1],
                                .w30 = w3.shape[0],
                                .w31 = w3.shape[1],
                            });
                        }
                    }

                    var ln1_weight = try uploadTensor(device, allocator, attn.ln1_weight);
                    errdefer ln1_weight.deinit(device);
                    var ln2_weight = try uploadTensor(device, allocator, attn.ln2_weight);
                    errdefer ln2_weight.deinit(device);
                    if (!(ln1_weight.rows == d_model and ln1_weight.cols == 1)) {
                        log.warn("inference", "CUDA block runtime ln1 shape unsupported", .{
                            .layer = layer_idx,
                            .rows = ln1_weight.rows,
                            .cols = ln1_weight.cols,
                            .d_model = d_model,
                        });
                        return error.UnsupportedModel;
                    }
                    if (!(ln2_weight.rows == d_model and ln2_weight.cols == 1)) {
                        log.warn("inference", "CUDA block runtime ln2 shape unsupported", .{
                            .layer = layer_idx,
                            .rows = ln2_weight.rows,
                            .cols = ln2_weight.cols,
                            .d_model = d_model,
                        });
                        return error.UnsupportedModel;
                    }

                    var pre_ffn_norm_weight: ?DeviceTensor = null;
                    if (attn.pre_ffn_norm) |pre_ffn_norm| {
                        var pre_ffn = try uploadTensor(device, allocator, pre_ffn_norm);
                        errdefer pre_ffn.deinit(device);
                        if (!(pre_ffn.rows == d_model and pre_ffn.cols == 1)) {
                            log.warn("inference", "CUDA block runtime pre_ffn_norm shape unsupported", .{
                                .layer = layer_idx,
                                .rows = pre_ffn.rows,
                                .cols = pre_ffn.cols,
                                .d_model = d_model,
                            });
                            return error.UnsupportedModel;
                        }
                        pre_ffn_norm_weight = pre_ffn;
                    }
                    errdefer if (pre_ffn_norm_weight) |*w| w.deinit(device);

                    var post_ffn_norm_weight: ?DeviceTensor = null;
                    if (attn.post_ffn_norm) |post_ffn_norm| {
                        var post_ffn = try uploadTensor(device, allocator, post_ffn_norm);
                        errdefer post_ffn.deinit(device);
                        if (!(post_ffn.rows == d_model and post_ffn.cols == 1)) {
                            log.warn("inference", "CUDA block runtime post_ffn_norm shape unsupported", .{
                                .layer = layer_idx,
                                .rows = post_ffn.rows,
                                .cols = post_ffn.cols,
                                .d_model = d_model,
                            });
                            return error.UnsupportedModel;
                        }
                        post_ffn_norm_weight = post_ffn;
                    }
                    errdefer if (post_ffn_norm_weight) |*w| w.deinit(device);

                    var q_norm_weight: ?DeviceTensor = null;
                    if (attn.q_norm) |q_norm| {
                        var qn = try uploadTensor(device, allocator, q_norm);
                        errdefer qn.deinit(device);
                        if (!(qn.rows == layer_head_dim and qn.cols == 1)) {
                            log.warn("inference", "CUDA block runtime q_norm shape unsupported", .{
                                .layer = layer_idx,
                                .rows = qn.rows,
                                .cols = qn.cols,
                                .head_dim = layer_head_dim,
                            });
                            return error.UnsupportedModel;
                        }
                        q_norm_weight = qn;
                        q_norm_blocks += 1;
                    }
                    errdefer if (q_norm_weight) |*w| w.deinit(device);

                    var k_norm_weight: ?DeviceTensor = null;
                    if (attn.k_norm) |k_norm| {
                        var kn = try uploadTensor(device, allocator, k_norm);
                        errdefer kn.deinit(device);
                        if (!(kn.rows == layer_head_dim and kn.cols == 1)) {
                            log.warn("inference", "CUDA block runtime k_norm shape unsupported", .{
                                .layer = layer_idx,
                                .rows = kn.rows,
                                .cols = kn.cols,
                                .head_dim = layer_head_dim,
                            });
                            return error.UnsupportedModel;
                        }
                        k_norm_weight = kn;
                        k_norm_blocks += 1;
                    }
                    errdefer if (k_norm_weight) |*w| w.deinit(device);

                    var q_proj_dev: LinearWeight = undefined;
                    var k_proj_dev: LinearWeight = undefined;
                    var v_proj_dev: LinearWeight = undefined;
                    if (attn.fused.qkv_proj) |qkv_proj| {
                        if (attn.attention_config.query_gate) {
                            log.warn("inference", "CUDA block runtime fused qkv with query_gate unsupported", .{
                                .layer = layer_idx,
                            });
                            return error.UnsupportedModel;
                        }
                        const fused_qkv = try uploadFusedQkvWeights(
                            device,
                            allocator,
                            &qkv_proj,
                            d_model,
                            q_out,
                            kv_out,
                        );
                        q_proj_dev = fused_qkv.q;
                        k_proj_dev = fused_qkv.k;
                        v_proj_dev = fused_qkv.v;
                    } else {
                        q_proj_dev = try uploadLinearWeightWithContext(device, allocator, q_proj_src, d_model, layer_idx, "self_attn.q_proj.weight");
                        k_proj_dev = try uploadLinearWeightWithContext(device, allocator, k_proj_src, d_model, layer_idx, "self_attn.k_proj.weight");
                        v_proj_dev = try uploadLinearWeightWithContext(device, allocator, v_proj_src, d_model, layer_idx, "self_attn.v_proj.weight");
                    }
                    errdefer q_proj_dev.deinit(device);
                    errdefer k_proj_dev.deinit(device);
                    errdefer v_proj_dev.deinit(device);

                    var o_proj_dev = try uploadLinearWeightWithContext(device, allocator, attn.o_proj, q_out, layer_idx, "self_attn.o_proj.weight");
                    errdefer o_proj_dev.deinit(device);

                    // Upload FFN weights: either standard SwiGLU (w1/w2/w3) or MoE
                    var w1_dev: LinearWeight = undefined;
                    var w3_dev: LinearWeight = undefined;
                    var w2_dev: LinearWeight = undefined;
                    var d_ff: usize = 0;
                    var moe_weight_refs: ?MoEWeightRefs = null;
                    if (has_moe) {
                        const moe = attn.moe_weights.?;
                        const moe_result = try uploadMoEWeights(device, allocator, moe, d_model, layer_idx, loaded.config.use_gelu);
                        moe_weight_refs = moe_result;
                        // Use dummy LinearWeights for w1/w2/w3 — not accessed for MoE layers
                        w1_dev = moe_result.shared_gate;
                        w3_dev = moe_result.shared_up;
                        w2_dev = moe_result.shared_down;
                        d_ff = @max(moe_result.shared_d_ff, 2 * @as(usize, moe_result.expert_d_ff));
                    } else if (attn.fused.gate_up) |gate_up| {
                        if (supportsFusedGateUpDenseUpload(gate_up.dtype)) {
                            const fused_gate_up = try uploadFusedGateUpWeights(
                                device,
                                allocator,
                                &gate_up,
                                d_model,
                                attn.fused.gate_up_layout,
                            );
                            w1_dev = fused_gate_up.gate;
                            w3_dev = fused_gate_up.up;
                            d_ff = w1_dev.cols();
                            w2_dev = try uploadLinearWeightWithContext(device, allocator, w2.?, d_ff, layer_idx, "mlp.down_proj.weight");
                        } else {
                            const w1 = attn.w1 orelse return error.MissingWeight;
                            const w3 = attn.w3 orelse return error.MissingWeight;
                            w1_dev = try uploadLinearWeightWithContext(device, allocator, w1, d_model, layer_idx, "mlp.gate_proj.weight");
                            w3_dev = try uploadLinearWeightWithContext(device, allocator, w3, d_model, layer_idx, "mlp.up_proj.weight");
                            if (w1_dev.cols() != w3_dev.cols()) {
                                log.warn("inference", "CUDA block runtime gate/up dim mismatch", .{
                                    .layer = layer_idx,
                                    .w1_cols = w1_dev.cols(),
                                    .w3_cols = w3_dev.cols(),
                                });
                                return error.UnsupportedModel;
                            }
                            d_ff = w1_dev.cols();
                            w2_dev = try uploadLinearWeightWithContext(device, allocator, w2.?, d_ff, layer_idx, "mlp.down_proj.weight");
                        }
                    } else {
                        const w1 = attn.w1 orelse return error.MissingWeight;
                        const w3 = attn.w3 orelse return error.MissingWeight;
                        w1_dev = try uploadLinearWeightWithContext(device, allocator, w1, d_model, layer_idx, "mlp.gate_proj.weight");
                        w3_dev = try uploadLinearWeightWithContext(device, allocator, w3, d_model, layer_idx, "mlp.up_proj.weight");
                        if (w1_dev.cols() != w3_dev.cols()) {
                            log.warn("inference", "CUDA block runtime gate/up dim mismatch", .{
                                .layer = layer_idx,
                                .w1_cols = w1_dev.cols(),
                                .w3_cols = w3_dev.cols(),
                            });
                            return error.UnsupportedModel;
                        }
                        d_ff = w1_dev.cols();
                        w2_dev = try uploadLinearWeightWithContext(device, allocator, w2.?, d_ff, layer_idx, "mlp.down_proj.weight");
                    }
                    errdefer w1_dev.deinit(device);
                    errdefer w3_dev.deinit(device);
                    errdefer w2_dev.deinit(device);
                    const cpu_attention_kernel: ?cpu_kernels.MultiHeadAttention = null;
                    const cpu_attention_cache: ?cpu_kernels.AttnCache = null;
                    const cpu_attention_scratch: ?cpu_kernels.AttnTemp = null;
                    const cpu_attention_matmul_scratch: ?compute.cpu.linalg.MatmulScratch = null;
                    if (k_proj_dev.cols() != v_proj_dev.cols()) {
                        log.warn("inference", "CUDA block runtime k/v dim mismatch", .{
                            .layer = layer_idx,
                            .k_cols = k_proj_dev.cols(),
                            .v_cols = v_proj_dev.cols(),
                        });
                        return error.UnsupportedModel;
                    }
                    if (o_proj_dev.cols() != d_model) {
                        log.warn("inference", "CUDA block runtime o_proj out dim unsupported", .{
                            .layer = layer_idx,
                            .o_proj_cols = o_proj_dev.cols(),
                            .d_model = d_model,
                        });
                        return error.UnsupportedModel;
                    }
                    if (w2_dev.cols() != d_model) {
                        log.warn("inference", "CUDA block runtime down_proj out dim unsupported", .{
                            .layer = layer_idx,
                            .w2_cols = w2_dev.cols(),
                            .d_model = d_model,
                        });
                        return error.UnsupportedModel;
                    }
                    if (q_proj_dev.cols() != q_proj_out) {
                        log.warn("inference", "CUDA block runtime q_proj dim unsupported", .{
                            .layer = layer_idx,
                            .q_cols = q_proj_dev.cols(),
                            .expected = q_proj_out,
                            .query_gate = @as(u8, @intFromBool(attn.attention_config.query_gate)),
                        });
                        return error.UnsupportedModel;
                    }
                    if (k_proj_dev.cols() != kv_out) {
                        log.warn("inference", "CUDA block runtime kv dim unsupported", .{
                            .layer = layer_idx,
                            .kv_cols = k_proj_dev.cols(),
                            .expected = kv_out,
                        });
                        return error.UnsupportedModel;
                    }

                    const kv_capacity = initial_kv_tokens;
                    if (kv_capacity == 0) return error.InvalidArgument;
                    const kv_cache_bytes_per_buffer = try kvCacheBytesForCapacityDtype(kv_capacity, k_proj_dev.cols(), kv_cache_dtype);
                    var kv_pair = try allocDeviceKvPairWithScales(device, kv_capacity, k_proj_dev.cols(), layer_n_kv_heads, kv_cache_dtype);
                    errdefer {
                        if (kv_pair.v_scale.pointer != 0) kv_pair.v_scale.deinit(device);
                        if (kv_pair.k_scale.pointer != 0) kv_pair.k_scale.deinit(device);
                        kv_pair.v.deinit(device);
                        kv_pair.k.deinit(device);
                    }

                    const layer_norm_bytes = ln1_weight.byteSize() +
                        ln2_weight.byteSize() +
                        (if (pre_ffn_norm_weight) |w| w.byteSize() else 0) +
                        (if (post_ffn_norm_weight) |w| w.byteSize() else 0) +
                        (if (q_norm_weight) |w| w.byteSize() else 0) +
                        (if (k_norm_weight) |w| w.byteSize() else 0);
                    norm_weight_bytes = std.math.add(usize, norm_weight_bytes, layer_norm_bytes) catch return error.InvalidArgument;

                    const layer_linear_bytes = q_proj_dev.byteSize() +
                        k_proj_dev.byteSize() +
                        v_proj_dev.byteSize() +
                        o_proj_dev.byteSize() +
                        w1_dev.byteSize() +
                        w2_dev.byteSize() +
                        w3_dev.byteSize();
                    linear_weight_bytes = std.math.add(usize, linear_weight_bytes, layer_linear_bytes) catch return error.InvalidArgument;

                    const layer_kv_bytes = std.math.mul(usize, kv_cache_bytes_per_buffer, 2) catch return error.InvalidArgument;
                    kv_cache_bytes = std.math.add(usize, kv_cache_bytes, layer_kv_bytes) catch return error.InvalidArgument;

                    const slot_kv_index = attention_block_count;
                    const kv_shared_source_layer_global = resolveSharedKvSourceLayer(loaded.config, layer_idx);
                    // Convert global source layer to local index. If the source
                    // is on a different device, record it for mirror replication.
                    const kv_shared_source_layer: ?usize = if (kv_shared_source_layer_global) |src_layer_idx| blk: {
                        if (src_layer_idx < layer_start or src_layer_idx >= layer_end) {
                            try pending_mirrors.append(allocator, .{
                                .local_idx = local_idx,
                                .source_global = src_layer_idx,
                                .kv_dim = k_proj_dev.cols(),
                            });
                            break :blk null;
                        }
                        break :blk src_layer_idx - layer_start;
                    } else null;
                    const kv_shared_source_slot_kv_index: ?usize = if (kv_shared_source_layer) |src_local_idx| blk: {
                        if (src_local_idx >= blocks.len) break :blk null;
                        const src_binding = blocks[src_local_idx].attention_binding orelse break :blk null;
                        break :blk src_binding.slot_kv_index;
                    } else null;

                    blocks[local_idx].attention_runtime = .{
                        .q_dim = q_out,
                        .q_projection_dim = q_proj_dev.cols(),
                        .kv_dim = k_proj_dev.cols(),
                        .d_ff = d_ff,
                        .sliding_window = attn.sliding_window,
                        .is_causal = attn.is_causal,
                        .query_gate = attn.attention_config.query_gate,
                        .ln1_weight = ln1_weight,
                        .ln2_weight = ln2_weight,
                        .pre_ffn_norm_weight = pre_ffn_norm_weight,
                        .post_ffn_norm_weight = post_ffn_norm_weight,
                        .q_norm_weight = q_norm_weight,
                        .k_norm_weight = k_norm_weight,
                        .q_proj = q_proj_dev,
                        .k_proj = k_proj_dev,
                        .v_proj = v_proj_dev,
                        .o_proj = o_proj_dev,
                        .w1 = w1_dev,
                        .w2 = w2_dev,
                        .w3 = w3_dev,
                        .k_cache = kv_pair.k,
                        .v_cache = kv_pair.v,
                        .k_scale = kv_pair.k_scale,
                        .v_scale = kv_pair.v_scale,
                        .kv_capacity = kv_capacity,
                        .slot_kv_index = slot_kv_index,
                        .kv_shared_source_layer = kv_shared_source_layer,
                        .kv_shared_source_slot_kv_index = kv_shared_source_slot_kv_index,
                        .use_v_norm = loaded.config.use_v_norm,
                        .cpu_kernel = cpu_attention_kernel,
                        .cpu_cache = cpu_attention_cache,
                        .cpu_scratch = cpu_attention_scratch,
                        .cpu_matmul_scratch = cpu_attention_matmul_scratch,
                    };
                    blocks[local_idx].attention_binding = &blocks[local_idx].attention_runtime.?;
                    BlockRuntimeLayer.bindAttentionNormWeights(&blocks[local_idx], &blocks[local_idx].attention_runtime.?);
                    if (moe_weight_refs) |moe_refs| {
                        blocks[local_idx].moe_runtime = moe_refs;
                        blocks[local_idx].moe_binding = &blocks[local_idx].moe_runtime.?;
                        // Add MoE expert weight bytes to linear total
                        var moe_linear_bytes: usize = 0;
                        for (moe_refs.expert_gate_up) |w| moe_linear_bytes += w.byteSize();
                        for (moe_refs.expert_down) |w| moe_linear_bytes += w.byteSize();
                        moe_linear_bytes += moe_refs.router_proj.byteSize();
                        linear_weight_bytes = std.math.add(usize, linear_weight_bytes, moe_linear_bytes) catch return error.InvalidArgument;
                        // Add MoE norm bytes
                        var moe_norm_bytes: usize = 0;
                        if (moe_refs.pre_ffn_norm) |t| moe_norm_bytes += t.byteSize();
                        if (moe_refs.post_shared_norm) |t| moe_norm_bytes += t.byteSize();
                        if (moe_refs.pre_expert_norm) |t| moe_norm_bytes += t.byteSize();
                        if (moe_refs.post_expert_norm) |t| moe_norm_bytes += t.byteSize();
                        if (moe_refs.post_combine_norm) |t| moe_norm_bytes += t.byteSize();
                        if (moe_refs.router_input_scale) |t| moe_norm_bytes += t.byteSize();
                        if (moe_refs.router_per_expert_scale) |t| moe_norm_bytes += t.byteSize();
                        norm_weight_bytes = std.math.add(usize, norm_weight_bytes, moe_norm_bytes) catch return error.InvalidArgument;
                    }
                    attention_block_count += 1;
                },
                .gated_delta => |gated_delta| {
                    const in_proj_cols = try tensorProjectionOutputDim(gated_delta.weights.in_proj, d_model);
                    max_gdelta_proj = @max(max_gdelta_proj, in_proj_cols);
                    const entry = static_entry orelse {
                        log.warn("inference", "CUDA gated-delta runtime missing architecture metadata", .{ .layer = layer_idx });
                        return error.UnsupportedModel;
                    };
                    const program = models.registry.blockProgramFor(entry, .gated_delta) orelse {
                        log.warn("inference", "CUDA gated-delta runtime missing LayerOp program", .{
                            .layer = layer_idx,
                            .kind = @intFromEnum(op_types.BlockKind.gated_delta),
                            .architecture = entry.id,
                        });
                        return error.UnsupportedModel;
                    };
                    blocks[local_idx].compiled_plan = try plan_compiler.compileLayerProgram(
                        allocator,
                        program,
                        .decode,
                        .{
                            .size_floor = d_model,
                            .state_descriptor_entry = entry,
                            .gated_delta_config_override = .{
                                .d_conv = @intCast(gated_delta.config.d_conv),
                                .n_heads = @intCast(gated_delta.config.n_heads),
                                .d_head = @intCast(gated_delta.config.d_head),
                                .d_inner = @intCast(@as(usize, gated_delta.config.n_heads) * @as(usize, gated_delta.config.d_head)),
                            },
                        },
                    );
                    try validateCompiledLayerPlanForCuda(&blocks[local_idx].compiled_plan.?, layer_idx, .gated_delta, adapter_table);
                    errdefer if (blocks[local_idx].compiled_plan) |*compiled_plan| {
                        plan_compiler.deinitCompiledPlan(allocator, compiled_plan);
                        blocks[local_idx].compiled_plan = null;
                    };
                    blocks[local_idx].register_to_slot_map = try buildCudaLayerProgramRegisterSlotMap(
                        allocator,
                        &blocks[local_idx].compiled_plan.?,
                    );
                    errdefer if (blocks[local_idx].register_to_slot_map.len != 0) {
                        allocator.free(blocks[local_idx].register_to_slot_map);
                        blocks[local_idx].register_to_slot_map = &.{};
                    };
                    blocks[local_idx].slot_width_hints = try buildCudaLayerProgramSlotWidthHints(
                        allocator,
                        &blocks[local_idx].compiled_plan.?,
                        blocks[local_idx].register_to_slot_map,
                    );
                    errdefer if (blocks[local_idx].slot_width_hints.len != 0) {
                        allocator.free(blocks[local_idx].slot_width_hints);
                        blocks[local_idx].slot_width_hints = &.{};
                    };
                    if (gated_delta.config.d_model != d_model) {
                        log.warn("inference", "CUDA gated-delta d_model mismatch", .{
                            .layer = layer_idx,
                            .config_d_model = gated_delta.config.d_model,
                            .model_d_model = d_model,
                        });
                        return error.UnsupportedModel;
                    }

                    var ln1_weight = try uploadTensor(device, allocator, gated_delta.ln1_weight);
                    errdefer ln1_weight.deinit(device);
                    if (!(ln1_weight.rows == d_model and ln1_weight.cols == 1)) {
                        log.warn("inference", "CUDA gated-delta ln1 shape unsupported", .{
                            .layer = layer_idx,
                            .rows = ln1_weight.rows,
                            .cols = ln1_weight.cols,
                            .d_model = d_model,
                        });
                        return error.UnsupportedModel;
                    }

                    var ln2_weight: ?DeviceTensor = null;
                    if (gated_delta.ln2_weight) |ln2| {
                        var ln2_dev = try uploadTensor(device, allocator, ln2);
                        errdefer ln2_dev.deinit(device);
                        if (!(ln2_dev.rows == d_model and ln2_dev.cols == 1)) {
                            log.warn("inference", "CUDA gated-delta ln2 shape unsupported", .{
                                .layer = layer_idx,
                                .rows = ln2_dev.rows,
                                .cols = ln2_dev.cols,
                                .d_model = d_model,
                            });
                            return error.UnsupportedModel;
                        }
                        ln2_weight = ln2_dev;
                    }
                    errdefer if (ln2_weight) |*w| w.deinit(device);

                    // Upload FFN weights: either MoE or standard SwiGLU (w1/w2/w3)
                    const has_gd_moe = gated_delta.moe_weights != null;
                    var moe_weight_refs: ?MoEWeightRefs = null;
                    if (has_gd_moe) {
                        const gd_moe = gated_delta.moe_weights.?;
                        moe_weight_refs = try uploadMoEWeights(device, allocator, gd_moe, d_model, layer_idx, loaded.config.use_gelu);
                    }
                    errdefer if (moe_weight_refs) |*mwr| mwr.deinit(allocator, device);

                    var ffn_w1: ?LinearWeight = null;
                    var ffn_w2: ?LinearWeight = null;
                    var ffn_w3: ?LinearWeight = null;
                    var d_ff: usize = 0;
                    if (!has_gd_moe) {
                        const ffn_plan = try resolveGatedDeltaFfnUploadPlan(&gated_delta);
                        switch (ffn_plan) {
                            .none => {
                                log.warn("inference", "CUDA gated-delta missing FFN weights", .{
                                    .layer = layer_idx,
                                });
                                return error.UnsupportedModel;
                            },
                            .split => |split| {
                                if (ln2_weight == null) {
                                    log.warn("inference", "CUDA gated-delta ffn requires ln2", .{
                                        .layer = layer_idx,
                                    });
                                    return error.UnsupportedModel;
                                }
                                var w1_dev = try uploadLinearWeightWithContext(device, allocator, split.w1, d_model, layer_idx, "mlp.gate_proj.weight");
                                errdefer w1_dev.deinit(device);
                                var w3_dev = try uploadLinearWeightWithContext(device, allocator, split.w3, d_model, layer_idx, "mlp.up_proj.weight");
                                errdefer w3_dev.deinit(device);
                                if (w1_dev.cols() != w3_dev.cols()) {
                                    log.warn("inference", "CUDA gated-delta gate/up dim mismatch", .{
                                        .layer = layer_idx,
                                        .w1_cols = w1_dev.cols(),
                                        .w3_cols = w3_dev.cols(),
                                    });
                                    return error.UnsupportedModel;
                                }
                                d_ff = w1_dev.cols();
                                var w2_dev = try uploadLinearWeightWithContext(device, allocator, split.w2, d_ff, layer_idx, "mlp.down_proj.weight");
                                errdefer w2_dev.deinit(device);
                                if (w2_dev.cols() != d_model) {
                                    log.warn("inference", "CUDA gated-delta down_proj out dim unsupported", .{
                                        .layer = layer_idx,
                                        .w2_cols = w2_dev.cols(),
                                        .d_model = d_model,
                                    });
                                    return error.UnsupportedModel;
                                }
                                ffn_w1 = w1_dev;
                                ffn_w2 = w2_dev;
                                ffn_w3 = w3_dev;
                            },
                            .fused => |fused| {
                                if (ln2_weight == null) {
                                    log.warn("inference", "CUDA gated-delta ffn requires ln2", .{
                                        .layer = layer_idx,
                                    });
                                    return error.UnsupportedModel;
                                }
                                const fused_gate_up = try uploadFusedGateUpWeights(
                                    device,
                                    allocator,
                                    &fused.gate_up,
                                    d_model,
                                    fused.gate_up_layout,
                                );
                                var w1_dev = fused_gate_up.gate;
                                errdefer w1_dev.deinit(device);
                                var w3_dev = fused_gate_up.up;
                                errdefer w3_dev.deinit(device);
                                d_ff = w1_dev.cols();
                                var w2_dev = try uploadLinearWeightWithContext(device, allocator, fused.w2, d_ff, layer_idx, "mlp.down_proj.weight");
                                errdefer w2_dev.deinit(device);
                                if (w2_dev.cols() != d_model) {
                                    log.warn("inference", "CUDA gated-delta down_proj out dim unsupported", .{
                                        .layer = layer_idx,
                                        .w2_cols = w2_dev.cols(),
                                        .d_model = d_model,
                                    });
                                    return error.UnsupportedModel;
                                }
                                ffn_w1 = w1_dev;
                                ffn_w2 = w2_dev;
                                ffn_w3 = w3_dev;
                            },
                        }
                    }
                    errdefer if (ffn_w1) |*w| w.deinit(device);
                    errdefer if (ffn_w2) |*w| w.deinit(device);
                    errdefer if (ffn_w3) |*w| w.deinit(device);

                    const gated_delta_kernel_config = cpu_kernels.GatedDeltaConfig{
                        .d_model = gated_delta.config.d_model,
                        .d_conv = gated_delta.config.d_conv,
                        .n_heads = gated_delta.config.n_heads,
                        .d_head = gated_delta.config.d_head,
                        .n_key_heads = gated_delta.config.n_key_heads,
                    };
                    const conv_values = try materializeTensorF32(allocator, gated_delta.weights.conv1d_weight);
                    defer allocator.free(conv_values);
                    if (gated_delta_kernel_config.d_conv == 0 or (conv_values.len % gated_delta_kernel_config.d_conv) != 0) return error.UnsupportedModel;
                    const gated_delta_conv_dim = conv_values.len / gated_delta_kernel_config.d_conv;

                    var in_proj_dev = try uploadLinearWeightWithContext(device, allocator, gated_delta.weights.in_proj, d_model, layer_idx, "gated_delta.in_proj");
                    errdefer in_proj_dev.deinit(device);
                    var out_proj_dev = try uploadLinearWeightWithContext(device, allocator, gated_delta.weights.out_proj, @as(usize, gated_delta.config.n_heads) * @as(usize, gated_delta.config.d_head), layer_idx, "gated_delta.out_proj");
                    errdefer out_proj_dev.deinit(device);
                    var conv_weight_time_major = try uploadShortConvWeightTimeMajor(
                        device,
                        allocator,
                        gated_delta.weights.conv1d_weight,
                        gated_delta_conv_dim,
                        gated_delta_kernel_config.d_conv,
                    );
                    errdefer conv_weight_time_major.deinit(device);
                    var conv_bias_dev: ?DeviceTensor = null;
                    if (gated_delta.weights.conv1d_bias) |bias| {
                        var bias_dev = try uploadVectorTensor(device, allocator, bias, gated_delta_conv_dim);
                        errdefer bias_dev.deinit(device);
                        conv_bias_dev = bias_dev;
                    }
                    errdefer if (conv_bias_dev) |*w| w.deinit(device);
                    var a_log_dev = try uploadTensor(device, allocator, gated_delta.weights.A_log);
                    errdefer a_log_dev.deinit(device);
                    var dt_bias_dev: ?DeviceTensor = null;
                    if (gated_delta.weights.dt_bias) |bias| {
                        var bias_dev = try uploadTensor(device, allocator, bias);
                        errdefer bias_dev.deinit(device);
                        dt_bias_dev = bias_dev;
                    }
                    errdefer if (dt_bias_dev) |*w| w.deinit(device);
                    const norm_weight_dev = blk: {
                        if (gated_delta.weights.norm_weight) |norm_weight| {
                            break :blk try uploadTensor(device, allocator, norm_weight);
                        }
                        const default_len: usize = gated_delta.config.d_head;
                        const ones = try allocator.alloc(f32, default_len);
                        defer allocator.free(ones);
                        @memset(ones, 1.0);
                        var buffer = try device.allocBuffer(default_len * @sizeOf(f32));
                        errdefer buffer.deinit(device);
                        try buffer.upload(device, std.mem.sliceAsBytes(ones));
                        break :blk DeviceTensor{ .rows = 1, .cols = default_len, .buffer = buffer };
                    };
                    errdefer {
                        var norm = norm_weight_dev;
                        norm.deinit(device);
                    }

                    // CPU matmul dispatchers for the GatedDeltaKernel struct. The CUDA engine
                    // never calls these (GPU handles all matmul), but the struct requires them.
                    // Fall back to matmulF32 for dtypes without a CPU kernel (e.g. FP8).
                    const in_proj_fn = (compute.cpu.linalg.matmulKernel(gated_delta.weights.in_proj.dtype) catch
                        compute.cpu.linalg.DispatchedKernel{ .func = compute.cpu.linalg.matmulF32, .name = "matmulF32" }).func;
                    const out_proj_fn = (compute.cpu.linalg.matmulKernel(gated_delta.weights.out_proj.dtype) catch
                        compute.cpu.linalg.DispatchedKernel{ .func = compute.cpu.linalg.matmulF32, .name = "matmulF32" }).func;
                    var gated_delta_kernel = cpu_kernels.GatedDeltaKernel.init(
                        gated_delta_kernel_config,
                        .{
                            .in_proj = gated_delta.weights.in_proj,
                            .conv1d_weight = gated_delta.weights.conv1d_weight,
                            .conv1d_bias = gated_delta.weights.conv1d_bias,
                            .A_log = gated_delta.weights.A_log,
                            .dt_bias = gated_delta.weights.dt_bias,
                            .norm_weight = gated_delta.weights.norm_weight,
                            .out_proj = gated_delta.weights.out_proj,
                        },
                        in_proj_fn,
                        out_proj_fn,
                    );
                    gated_delta_kernel.layer_idx = @intCast(layer_idx);
                    try gated_delta_kernel.initTransposedWeights(allocator);
                    errdefer gated_delta_kernel.deinit();

                    var gated_delta_state = try cpu_kernels.GatedDeltaState.init(
                        allocator,
                        1,
                        .{
                            .d_model = gated_delta.config.d_model,
                            .d_conv = gated_delta.config.d_conv,
                            .n_heads = gated_delta.config.n_heads,
                            .d_head = gated_delta.config.d_head,
                            .n_key_heads = gated_delta.config.n_key_heads,
                        },
                    );
                    errdefer gated_delta_state.deinit();
                    const conv_state_bytes = gated_delta_state.conv_state.len * @sizeOf(f32);
                    var conv_state_dev = try device.allocBuffer(conv_state_bytes);
                    errdefer conv_state_dev.deinit(device);
                    const zero_conv = try allocator.alloc(f32, gated_delta_state.conv_state.len);
                    defer allocator.free(zero_conv);
                    @memset(zero_conv, 0.0);
                    try conv_state_dev.upload(device, std.mem.sliceAsBytes(zero_conv));
                    const ssm_state_format: GatedDeltaSsmStateFormat = if (gated_delta_ssm_i8_state)
                        .i8_per_column_scale
                    else
                        .f32;
                    var ssm_state_scales_offset_bytes: usize = 0;
                    var ssm_state_storage_bytes: usize = 0;
                    var ssm_state_dev: compute.cuda.Buffer = undefined;
                    switch (ssm_state_format) {
                        .f32 => {
                            const ssm_state_bytes = gated_delta_state.ssm_state.len * @sizeOf(f32);
                            ssm_state_storage_bytes = ssm_state_bytes;
                            ssm_state_dev = try device.allocBuffer(ssm_state_bytes);
                            errdefer ssm_state_dev.deinit(device);
                            const zero_ssm = try allocator.alloc(f32, gated_delta_state.ssm_state.len);
                            defer allocator.free(zero_ssm);
                            @memset(zero_ssm, 0.0);
                            try ssm_state_dev.upload(device, std.mem.sliceAsBytes(zero_ssm));
                        },
                        .i8_per_column_scale => {
                            const d_inner = @as(usize, gated_delta.config.n_heads) * @as(usize, gated_delta.config.d_head);
                            const ssm_state_i8_bytes = gated_delta_state.ssm_state.len;
                            const ssm_scales_bytes = std.math.mul(usize, d_inner, @sizeOf(f32)) catch return error.InvalidArgument;
                            ssm_state_scales_offset_bytes = std.mem.alignForward(usize, ssm_state_i8_bytes, @alignOf(f32));
                            ssm_state_storage_bytes = std.math.add(
                                usize,
                                ssm_state_scales_offset_bytes,
                                ssm_scales_bytes,
                            ) catch return error.InvalidArgument;
                            ssm_state_dev = try device.allocBuffer(ssm_state_storage_bytes);
                            errdefer ssm_state_dev.deinit(device);

                            const zero_ssm = try allocator.alloc(i8, gated_delta_state.ssm_state.len);
                            defer allocator.free(zero_ssm);
                            @memset(zero_ssm, 0);
                            var ssm_state_i8_dev = try bufferSlice(&ssm_state_dev, 0, ssm_state_i8_bytes);
                            try ssm_state_i8_dev.upload(device, std.mem.sliceAsBytes(zero_ssm));

                            const init_scales = try allocator.alloc(f32, d_inner);
                            defer allocator.free(init_scales);
                            @memset(init_scales, 1.0);
                            var ssm_state_scales_dev = try bufferSlice(&ssm_state_dev, ssm_state_scales_offset_bytes, ssm_scales_bytes);
                            try ssm_state_scales_dev.upload(device, std.mem.sliceAsBytes(init_scales));
                        },
                    }

                    var gated_delta_scratch = try cpu_kernels.GatedDeltaScratch.init(
                        allocator,
                        .{
                            .d_model = gated_delta.config.d_model,
                            .d_conv = gated_delta.config.d_conv,
                            .n_heads = gated_delta.config.n_heads,
                            .d_head = gated_delta.config.d_head,
                            .n_key_heads = gated_delta.config.n_key_heads,
                        },
                    );
                    errdefer gated_delta_scratch.deinit();

                    var gated_delta_matmul_scratch = try compute.cpu.linalg.MatmulScratch.init(allocator);
                    errdefer gated_delta_matmul_scratch.deinit();

                    const layer_norm_bytes = ln1_weight.byteSize() + (if (ln2_weight) |w| w.byteSize() else 0);
                    norm_weight_bytes = std.math.add(usize, norm_weight_bytes, layer_norm_bytes) catch return error.InvalidArgument;

                    var layer_linear_bytes: usize = in_proj_dev.byteSize() + out_proj_dev.byteSize();
                    if (ffn_w1) |w| layer_linear_bytes = std.math.add(usize, layer_linear_bytes, w.byteSize()) catch return error.InvalidArgument;
                    if (ffn_w2) |w| layer_linear_bytes = std.math.add(usize, layer_linear_bytes, w.byteSize()) catch return error.InvalidArgument;
                    if (ffn_w3) |w| layer_linear_bytes = std.math.add(usize, layer_linear_bytes, w.byteSize()) catch return error.InvalidArgument;
                    linear_weight_bytes = std.math.add(usize, linear_weight_bytes, layer_linear_bytes) catch return error.InvalidArgument;

                    const gated_delta_state_bytes_layer = std.math.add(
                        usize,
                        conv_state_bytes,
                        ssm_state_storage_bytes,
                    ) catch return error.InvalidArgument;
                    gated_delta_state_bytes = std.math.add(usize, gated_delta_state_bytes, gated_delta_state_bytes_layer) catch return error.InvalidArgument;

                    blocks[local_idx].gated_delta_runtime = .{
                        .d_ff = d_ff,
                        .ln1_weight = ln1_weight,
                        .ln2_weight = ln2_weight,
                        .ffn_w1 = ffn_w1,
                        .ffn_w2 = ffn_w2,
                        .ffn_w3 = ffn_w3,
                        .in_proj = in_proj_dev,
                        .out_proj = out_proj_dev,
                        .conv_weight_time_major = conv_weight_time_major,
                        .conv_bias = conv_bias_dev,
                        .conv_state_dev = conv_state_dev,
                        .conv_ring_head = 0,
                        .a_log = a_log_dev,
                        .dt_bias = dt_bias_dev,
                        .norm_weight = norm_weight_dev,
                        .ssm_state_dev = ssm_state_dev,
                        .ssm_state_format = ssm_state_format,
                        .ssm_state_scales_offset = @intCast(ssm_state_scales_offset_bytes),
                        .kernel = gated_delta_kernel,
                        .state = gated_delta_state,
                        .scratch = gated_delta_scratch,
                        .matmul_scratch = gated_delta_matmul_scratch,
                    };
                    blocks[local_idx].gated_delta_binding = &blocks[local_idx].gated_delta_runtime.?;
                    BlockRuntimeLayer.bindGatedDeltaNormWeights(&blocks[local_idx], &blocks[local_idx].gated_delta_runtime.?);
                    if (moe_weight_refs) |moe_refs| {
                        blocks[local_idx].moe_runtime = moe_refs;
                        blocks[local_idx].moe_binding = &blocks[local_idx].moe_runtime.?;
                        var moe_linear_bytes: usize = 0;
                        for (moe_refs.expert_gate_up) |w| moe_linear_bytes += w.byteSize();
                        for (moe_refs.expert_down) |w| moe_linear_bytes += w.byteSize();
                        moe_linear_bytes += moe_refs.router_proj.byteSize();
                        linear_weight_bytes = std.math.add(usize, linear_weight_bytes, moe_linear_bytes) catch return error.InvalidArgument;
                        var moe_norm_bytes: usize = 0;
                        if (moe_refs.pre_ffn_norm) |t| moe_norm_bytes += t.byteSize();
                        if (moe_refs.post_shared_norm) |t| moe_norm_bytes += t.byteSize();
                        if (moe_refs.pre_expert_norm) |t| moe_norm_bytes += t.byteSize();
                        if (moe_refs.post_expert_norm) |t| moe_norm_bytes += t.byteSize();
                        if (moe_refs.post_combine_norm) |t| moe_norm_bytes += t.byteSize();
                        if (moe_refs.router_input_scale) |t| moe_norm_bytes += t.byteSize();
                        if (moe_refs.router_per_expert_scale) |t| moe_norm_bytes += t.byteSize();
                        norm_weight_bytes = std.math.add(usize, norm_weight_bytes, moe_norm_bytes) catch return error.InvalidArgument;
                    }
                    gated_delta_block_count += 1;
                },
                .shortconv => |shortconv| {
                    const entry = static_entry orelse {
                        log.warn("inference", "CUDA shortconv runtime missing architecture metadata", .{ .layer = layer_idx });
                        return error.UnsupportedModel;
                    };
                    const program = models.registry.blockProgramFor(entry, .shortconv) orelse {
                        log.warn("inference", "CUDA shortconv runtime missing LayerOp program", .{
                            .layer = layer_idx,
                            .kind = @intFromEnum(op_types.BlockKind.shortconv),
                            .architecture = entry.id,
                        });
                        return error.UnsupportedModel;
                    };
                    blocks[local_idx].compiled_plan = try plan_compiler.compileLayerProgram(
                        allocator,
                        program,
                        .decode,
                        .{
                            .size_floor = d_model,
                            .state_descriptor_entry = entry,
                        },
                    );
                    try validateCompiledLayerPlanForCuda(&blocks[local_idx].compiled_plan.?, layer_idx, .shortconv, adapter_table);
                    errdefer if (blocks[local_idx].compiled_plan) |*compiled_plan| {
                        plan_compiler.deinitCompiledPlan(allocator, compiled_plan);
                        blocks[local_idx].compiled_plan = null;
                    };
                    blocks[local_idx].register_to_slot_map = try buildCudaLayerProgramRegisterSlotMap(
                        allocator,
                        &blocks[local_idx].compiled_plan.?,
                    );
                    errdefer if (blocks[local_idx].register_to_slot_map.len != 0) {
                        allocator.free(blocks[local_idx].register_to_slot_map);
                        blocks[local_idx].register_to_slot_map = &.{};
                    };
                    blocks[local_idx].slot_width_hints = try buildCudaLayerProgramSlotWidthHints(
                        allocator,
                        &blocks[local_idx].compiled_plan.?,
                        blocks[local_idx].register_to_slot_map,
                    );
                    errdefer if (blocks[local_idx].slot_width_hints.len != 0) {
                        allocator.free(blocks[local_idx].slot_width_hints);
                        blocks[local_idx].slot_width_hints = &.{};
                    };
                    if (shortconv.fused_gate_up != null) {
                        log.warn("inference", "CUDA block runtime fused shortconv gate_up not supported yet", .{
                            .layer = layer_idx,
                        });
                        return error.UnsupportedModel;
                    }
                    const conv_dim: usize = @intCast(shortconv.config.conv_dim);
                    const d_conv: usize = @intCast(shortconv.config.d_conv);
                    if (shortconv_block_count == 0) {
                        log.info("inference", "CUDA shortconv block0 config", .{
                            .layer = layer_idx,
                            .d_model = shortconv.config.d_model,
                            .conv_dim = shortconv.config.conv_dim,
                            .conv_dim_out = shortconv.config.conv_dim_out,
                            .d_conv = shortconv.config.d_conv,
                            .has_bias = @as(u8, @intFromBool(shortconv.config.has_bias)),
                            .in_proj_dtype = @tagName(shortconv.weights.in_proj.dtype),
                            .in_proj_0 = shortconv.weights.in_proj.shape[0],
                            .in_proj_1 = shortconv.weights.in_proj.shape[1],
                            .conv_weight_dtype = @tagName(shortconv.weights.conv1d_weight.dtype),
                            .conv_weight_n_dims = shortconv.weights.conv1d_weight.n_dims,
                            .conv_weight_0 = shortconv.weights.conv1d_weight.shape[0],
                            .conv_weight_1 = shortconv.weights.conv1d_weight.shape[1],
                            .conv_weight_2 = shortconv.weights.conv1d_weight.shape[2],
                            .out_proj_dtype = @tagName(shortconv.weights.out_proj.dtype),
                            .out_proj_0 = shortconv.weights.out_proj.shape[0],
                            .out_proj_1 = shortconv.weights.out_proj.shape[1],
                        });
                    }
                    if (conv_dim == 0 or d_conv == 0) return error.UnsupportedModel;
                    if (shortconv.config.d_model != d_model) {
                        log.warn("inference", "CUDA shortconv d_model mismatch", .{
                            .layer = layer_idx,
                            .config_d_model = shortconv.config.d_model,
                            .model_d_model = d_model,
                        });
                        return error.UnsupportedModel;
                    }

                    var ln1_weight = try uploadTensor(device, allocator, shortconv.ln1_weight);
                    errdefer ln1_weight.deinit(device);
                    if (!(ln1_weight.rows == d_model and ln1_weight.cols == 1)) {
                        log.warn("inference", "CUDA shortconv ln1 shape unsupported", .{
                            .layer = layer_idx,
                            .rows = ln1_weight.rows,
                            .cols = ln1_weight.cols,
                            .d_model = d_model,
                        });
                        return error.UnsupportedModel;
                    }

                    var ln2_weight: ?DeviceTensor = null;
                    if (shortconv.ln2_weight) |ln2| {
                        var ln2_dev = try uploadTensor(device, allocator, ln2);
                        errdefer ln2_dev.deinit(device);
                        if (!(ln2_dev.rows == d_model and ln2_dev.cols == 1)) {
                            log.warn("inference", "CUDA shortconv ln2 shape unsupported", .{
                                .layer = layer_idx,
                                .rows = ln2_dev.rows,
                                .cols = ln2_dev.cols,
                                .d_model = d_model,
                            });
                            return error.UnsupportedModel;
                        }
                        ln2_weight = ln2_dev;
                    }
                    errdefer if (ln2_weight) |*w| w.deinit(device);

                    var in_proj_dev = try uploadLinearWeightWithContext(device, allocator, shortconv.weights.in_proj, d_model, layer_idx, "conv.in_proj.weight");
                    errdefer in_proj_dev.deinit(device);
                    if (in_proj_dev.cols() != 3 * conv_dim) {
                        log.warn("inference", "CUDA shortconv in_proj dim unsupported", .{
                            .layer = layer_idx,
                            .cols = in_proj_dev.cols(),
                            .expected = 3 * conv_dim,
                        });
                        return error.UnsupportedModel;
                    }

                    var out_proj_dev = try uploadLinearWeightWithContext(device, allocator, shortconv.weights.out_proj, conv_dim, layer_idx, "conv.out_proj.weight");
                    errdefer out_proj_dev.deinit(device);
                    if (out_proj_dev.cols() != d_model) {
                        log.warn("inference", "CUDA shortconv out_proj dim unsupported", .{
                            .layer = layer_idx,
                            .cols = out_proj_dev.cols(),
                            .expected = d_model,
                        });
                        return error.UnsupportedModel;
                    }

                    var conv_weight_time_major = try uploadShortConvWeightTimeMajor(
                        device,
                        allocator,
                        shortconv.weights.conv1d_weight,
                        conv_dim,
                        d_conv,
                    );
                    errdefer conv_weight_time_major.deinit(device);

                    var conv_bias: ?DeviceTensor = null;
                    if (shortconv.weights.conv1d_bias) |bias| {
                        var bias_dev = try uploadVectorTensor(device, allocator, bias, conv_dim);
                        errdefer bias_dev.deinit(device);
                        conv_bias = bias_dev;
                    }
                    errdefer if (conv_bias) |*w| w.deinit(device);

                    const conv_state_count = std.math.mul(usize, conv_dim, d_conv) catch return error.InvalidArgument;
                    var conv_state = try allocZeroedF32Buffer(device, allocator, conv_state_count);
                    errdefer conv_state.deinit(device);

                    var ffn_w1: ?LinearWeight = null;
                    var ffn_w2: ?LinearWeight = null;
                    var ffn_w3: ?LinearWeight = null;
                    var d_ff: usize = 0;
                    if (shortconv.w1 != null or shortconv.w2 != null or shortconv.w3 != null) {
                        const w1 = shortconv.w1 orelse return error.MissingWeight;
                        const w2 = shortconv.w2 orelse return error.MissingWeight;
                        const w3 = shortconv.w3 orelse return error.MissingWeight;
                        if (ln2_weight == null) {
                            log.warn("inference", "CUDA shortconv ffn requires ln2", .{
                                .layer = layer_idx,
                            });
                            return error.UnsupportedModel;
                        }

                        var w1_dev = try uploadLinearWeightWithContext(device, allocator, w1, d_model, layer_idx, "mlp.gate_proj.weight");
                        errdefer w1_dev.deinit(device);
                        var w3_dev = try uploadLinearWeightWithContext(device, allocator, w3, d_model, layer_idx, "mlp.up_proj.weight");
                        errdefer w3_dev.deinit(device);
                        if (w1_dev.cols() != w3_dev.cols()) {
                            log.warn("inference", "CUDA shortconv gate/up dim mismatch", .{
                                .layer = layer_idx,
                                .w1_cols = w1_dev.cols(),
                                .w3_cols = w3_dev.cols(),
                            });
                            return error.UnsupportedModel;
                        }
                        d_ff = w1_dev.cols();
                        var w2_dev = try uploadLinearWeightWithContext(device, allocator, w2, d_ff, layer_idx, "mlp.down_proj.weight");
                        errdefer w2_dev.deinit(device);
                        if (w2_dev.cols() != d_model) {
                            log.warn("inference", "CUDA shortconv down_proj out dim unsupported", .{
                                .layer = layer_idx,
                                .w2_cols = w2_dev.cols(),
                                .d_model = d_model,
                            });
                            return error.UnsupportedModel;
                        }
                        ffn_w1 = w1_dev;
                        ffn_w2 = w2_dev;
                        ffn_w3 = w3_dev;
                    }
                    errdefer if (ffn_w1) |*w| w.deinit(device);
                    errdefer if (ffn_w2) |*w| w.deinit(device);
                    errdefer if (ffn_w3) |*w| w.deinit(device);

                    const layer_norm_bytes = ln1_weight.byteSize() + (if (ln2_weight) |w| w.byteSize() else 0);
                    norm_weight_bytes = std.math.add(usize, norm_weight_bytes, layer_norm_bytes) catch return error.InvalidArgument;

                    var layer_linear_bytes = in_proj_dev.byteSize() +
                        out_proj_dev.byteSize() +
                        conv_weight_time_major.byteSize() +
                        (if (conv_bias) |w| w.byteSize() else 0);
                    if (ffn_w1) |w| layer_linear_bytes = std.math.add(usize, layer_linear_bytes, w.byteSize()) catch return error.InvalidArgument;
                    if (ffn_w2) |w| layer_linear_bytes = std.math.add(usize, layer_linear_bytes, w.byteSize()) catch return error.InvalidArgument;
                    if (ffn_w3) |w| layer_linear_bytes = std.math.add(usize, layer_linear_bytes, w.byteSize()) catch return error.InvalidArgument;
                    linear_weight_bytes = std.math.add(usize, linear_weight_bytes, layer_linear_bytes) catch return error.InvalidArgument;
                    shortconv_state_bytes = std.math.add(usize, shortconv_state_bytes, conv_state.size) catch return error.InvalidArgument;
                    max_shortconv_dim = @max(max_shortconv_dim, conv_dim);

                    blocks[local_idx].shortconv_runtime = .{
                        .conv_dim = conv_dim,
                        .d_conv = d_conv,
                        .d_ff = d_ff,
                        .ln1_weight = ln1_weight,
                        .ln2_weight = ln2_weight,
                        .in_proj = in_proj_dev,
                        .out_proj = out_proj_dev,
                        .conv_weight_time_major = conv_weight_time_major,
                        .conv_bias = conv_bias,
                        .conv_state = conv_state,
                        .ffn_w1 = ffn_w1,
                        .ffn_w2 = ffn_w2,
                        .ffn_w3 = ffn_w3,
                    };
                    blocks[local_idx].shortconv_binding = &blocks[local_idx].shortconv_runtime.?;
                    BlockRuntimeLayer.bindShortConvNormWeights(&blocks[local_idx], &blocks[local_idx].shortconv_runtime.?);
                    shortconv_block_count += 1;
                },
                else => {
                    log.warn("inference", "CUDA block runtime unsupported block kind", .{
                        .layer = layer_idx,
                    });
                    return error.UnsupportedModel;
                },
            }
            try blocks[local_idx].rebuildInstructionMetadata(allocator);
            initialized += 1;
        }

        // Resolve cross-device KV sharing: deduplicate sources, assign mirror
        // indices, allocate mirror KV buffers, and fixup consumer blocks.
        var replicated_kv_sources: []ReplicatedKvSource = &.{};
        var mirror_kv: []MirrorKvBuffers = &.{};
        if (pending_mirrors.items.len > 0) {
            var unique_sources: std.ArrayListUnmanaged(ReplicatedKvSource) = .{};
            errdefer unique_sources.deinit(allocator);

            for (pending_mirrors.items) |pm| {
                var found = false;
                for (unique_sources.items) |src| {
                    if (src.global_layer_idx == pm.source_global) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    try unique_sources.append(allocator, .{
                        .global_layer_idx = pm.source_global,
                        .kv_dim = pm.kv_dim,
                        .mirror_kv_index = attention_block_count + unique_sources.items.len,
                    });
                }
            }

            // Allocate mirror KV buffers at max_seq_len capacity (avoids
            // ensureKvCapacity changes; bandwidth cost is negligible).
            const n_mirrors = unique_sources.items.len;
            var mirrors = try allocator.alloc(MirrorKvBuffers, n_mirrors);
            errdefer allocator.free(mirrors);
            var mirrors_allocated: usize = 0;
            errdefer for (mirrors[0..mirrors_allocated]) |*mk| {
                if (mk.v_scale.pointer != 0) mk.v_scale.deinit(device);
                if (mk.k_scale.pointer != 0) mk.k_scale.deinit(device);
                mk.v.deinit(device);
                mk.k.deinit(device);
            };

            for (unique_sources.items, 0..) |src, mi| {
                const n_mirror_kv_heads: usize = if (src.kv_dim > 0 and head_dim > 0) src.kv_dim / head_dim else n_kv_heads;
                const kv_pair = try allocDeviceKvPairWithScales(device, max_seq_len, src.kv_dim, n_mirror_kv_heads, kv_cache_dtype);
                mirrors[mi] = .{
                    .k = kv_pair.k,
                    .v = kv_pair.v,
                    .k_scale = kv_pair.k_scale,
                    .v_scale = kv_pair.v_scale,
                    .capacity = max_seq_len,
                };
                mirrors_allocated += 1;
            }

            // Fixup consumer blocks: set kv_shared_source_slot_kv_index to mirror.
            for (pending_mirrors.items) |pm| {
                for (unique_sources.items) |src| {
                    if (src.global_layer_idx == pm.source_global) {
                        blocks[pm.local_idx].attention_runtime.?.kv_shared_source_slot_kv_index = src.mirror_kv_index;
                        break;
                    }
                }
            }

            replicated_kv_sources = try unique_sources.toOwnedSlice(allocator);
            mirror_kv = mirrors;

            log.info("inference", "KV sharing: allocated mirror entries for cross-device sources", .{
                .n_mirrors = n_mirrors,
                .layer_start = layer_start,
                .layer_end = layer_end,
            });
        }

        return .{
            .blocks = blocks,
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .max_seq_len = max_seq_len,
            .attention_block_count = attention_block_count,
            .shortconv_block_count = shortconv_block_count,
            .gated_delta_block_count = gated_delta_block_count,
            .q_norm_blocks = q_norm_blocks,
            .k_norm_blocks = k_norm_blocks,
            .linear_weight_bytes = linear_weight_bytes,
            .norm_weight_bytes = norm_weight_bytes,
            .kv_cache_bytes = kv_cache_bytes,
            .shortconv_state_bytes = shortconv_state_bytes,
            .gated_delta_state_bytes = gated_delta_state_bytes,
            .max_shortconv_dim = max_shortconv_dim,
            .max_gdelta_proj = max_gdelta_proj,
            .replicated_kv_sources = replicated_kv_sources,
            .mirror_kv = mirror_kv,
        };
    }

    pub fn deinit(self: *BlockRuntime, allocator: std.mem.Allocator, device: *compute.cuda.Device) void {
        for (self.mirror_kv) |*mk| {
            if (mk.v_scale.pointer != 0) mk.v_scale.deinit(device);
            if (mk.k_scale.pointer != 0) mk.k_scale.deinit(device);
            mk.v.deinit(device);
            mk.k.deinit(device);
        }
        if (self.mirror_kv.len > 0) allocator.free(self.mirror_kv);
        if (self.replicated_kv_sources.len > 0) allocator.free(self.replicated_kv_sources);
        for (self.blocks) |*block| block.deinit(allocator, device);
        allocator.free(self.blocks);
    }

    pub fn maxDff(self: *const BlockRuntime) usize {
        var max_dff: usize = 0;
        for (self.blocks) |layer| {
            if (layer.attention_binding) |block| {
                if (block.d_ff > max_dff) max_dff = block.d_ff;
            }
            if (layer.shortconv_binding) |block| {
                if (block.d_ff > max_dff) max_dff = block.d_ff;
            }
            if (layer.gated_delta_binding) |block| {
                if (block.d_ff > max_dff) max_dff = block.d_ff;
            }
        }
        return max_dff;
    }

    pub fn maxAttn(self: *const BlockRuntime) usize {
        var max_attn: usize = 0;
        for (self.blocks) |layer| {
            if (layer.attention_binding) |block| {
                if (block.q_projection_dim > max_attn) max_attn = block.q_projection_dim;
            }
        }
        return if (max_attn > 0) max_attn else self.n_heads * self.head_dim;
    }

    pub fn maxKv(self: *const BlockRuntime) usize {
        var max_kv: usize = 0;
        for (self.blocks) |layer| {
            if (layer.attention_binding) |block| {
                if (block.kv_dim > max_kv) max_kv = block.kv_dim;
            }
        }
        return if (max_kv > 0) max_kv else self.n_kv_heads * self.head_dim;
    }

    pub fn maxShortConvDim(self: *const BlockRuntime) usize {
        return self.max_shortconv_dim;
    }

    pub fn maxGatedDeltaProj(self: *const BlockRuntime) usize {
        return if (self.max_gdelta_proj > 0) self.max_gdelta_proj else 1;
    }
};

pub const KvRuntimeState = extern struct {
    runtime_kind: u8,
    _pad: [7]u8 = [_]u8{0} ** 7,
    block_runtime: *BlockRuntime,
    slot_index: usize,
};

pub const RecurrentRuntimeState = extern struct {
    runtime_kind: u8,
    _pad: [7]u8 = [_]u8{0} ** 7,
    block_runtime: *BlockRuntime,
    slot_index: usize,
};

pub const ShortConvRuntimeState = RecurrentRuntimeState;
pub const MambaRuntimeState = RecurrentRuntimeState;
pub const GatedDeltaRuntimeState = RecurrentRuntimeState;

/// Per-row batch info for batched decode (N tokens at different positions/slots).
/// Null for single-token decode and prefill.
pub const BatchDecodeInfo = struct {
    slot_indices: []const usize,
    positions: []const usize,
    seq_lens: []const u32,
    attn_ptrs_row_stride: usize,
    attn_key_cache_ptrs_table_dev: *const compute.cuda.Buffer,
    attn_value_cache_ptrs_table_dev: *const compute.cuda.Buffer,
    gd_ptrs_row_stride: usize,
    gd_conv_state_ptrs_table_dev: *const compute.cuda.Buffer,
    gd_ssm_state_ptrs_table_dev: *const compute.cuda.Buffer,
    gd_conv_ring_heads_table_dev: *const compute.cuda.Buffer,
    attn_k_scale_ptrs_table_dev: *const compute.cuda.Buffer,
    attn_v_scale_ptrs_table_dev: *const compute.cuda.Buffer,
    attn_layer_index: usize,
    gd_layer_index: usize,
    sc_layer_index: usize,
};
