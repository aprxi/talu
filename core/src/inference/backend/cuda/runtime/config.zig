//! CUDA runtime configuration, feature switches, and numeric helpers.

const std = @import("std");
const build_options = @import("build_options");
const models = @import("models_pkg");
const compute = @import("compute_pkg");
const log = @import("log_pkg");
const attention_policy = @import("../attention_policy.zig");

pub const default_norm_eps: f32 = 1e-5;
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
    const raw = @import("env_pkg").getenv("TALU_KV_QUANT") orelse return .i8;
    if (std.ascii.eqlIgnoreCase(raw, "f16") or std.ascii.eqlIgnoreCase(raw, "fp16")) return .f16;
    if (std.ascii.eqlIgnoreCase(raw, "f8") or std.ascii.eqlIgnoreCase(raw, "fp8") or std.ascii.eqlIgnoreCase(raw, "e4m3")) return .fp8;
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

pub fn resolveSharedKvSourceLayer(config: models.config.ModelConfig, layer_idx: usize) ?usize {
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
pub fn adjustSplitForKvSharing(config: models.config.ModelConfig, proposed_split: usize, total_layers: usize, floor: usize) ?usize {
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
    const default_value = @min(max_seq_len, initial_kv_cache_tokens);
    const raw = std.process.getEnvVarOwned(std.heap.c_allocator, "TALU_CUDA_KV_INIT_TOKENS") catch {
        return default_value;
    };
    defer std.heap.c_allocator.free(raw);
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    const parsed = std.fmt.parseUnsigned(usize, trimmed, 10) catch {
        log.warn("inference", "Invalid TALU_CUDA_KV_INIT_TOKENS; using default", .{
            .value = trimmed,
            .default_value = default_value,
        });
        return default_value;
    };
    if (parsed == 0) {
        log.warn("inference", "TALU_CUDA_KV_INIT_TOKENS must be >= 1; using default", .{
            .value = parsed,
            .default_value = default_value,
        });
        return default_value;
    }
    return @min(max_seq_len, parsed);
}

pub fn resolveCudaPrefillChunkRowsCap(max_seq_len: usize) usize {
    const default_value = @min(max_seq_len, default_prefill_chunk_rows_cap);
    const raw = std.process.getEnvVarOwned(std.heap.c_allocator, "TALU_CUDA_PREFILL_CHUNK_ROWS") catch {
        return default_value;
    };
    defer std.heap.c_allocator.free(raw);
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    const parsed = std.fmt.parseUnsigned(usize, trimmed, 10) catch {
        log.warn("inference", "Invalid TALU_CUDA_PREFILL_CHUNK_ROWS; using default", .{
            .value = trimmed,
            .default_value = default_value,
        });
        return default_value;
    };
    if (parsed == 0) {
        log.warn("inference", "TALU_CUDA_PREFILL_CHUNK_ROWS must be >= 1; using default", .{
            .value = parsed,
            .default_value = default_value,
        });
        return default_value;
    }
    return @min(max_seq_len, parsed);
}
