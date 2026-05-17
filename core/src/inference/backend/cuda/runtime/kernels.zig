//! CUDA runtime kernel catalog and route counters.

const std = @import("std");
const compute = @import("compute_pkg");
const log = @import("log_pkg");
const config = @import("config.zig");

const saturatingU64FromU128 = config.saturatingU64FromU128;

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
    bf16_dense_route,
    small_rows_nvfp4_matvec,
    small_rows_i8_matvec,
    fused_qkv_custom,
    fused_qkv_native_cublaslt,
    fused_gate_up_custom,
    fused_gate_up_native_cublaslt,
};

pub const Nvfp4RouteCounters = struct {
    native_cublaslt: u64 = 0,
    bf16_dense_route: u64 = 0,
    small_rows_nvfp4_matvec: u64 = 0,
    small_rows_i8_matvec: u64 = 0,
    fused_qkv_custom: u64 = 0,
    fused_qkv_native_cublaslt: u64 = 0,
    fused_gate_up_custom: u64 = 0,
    fused_gate_up_native_cublaslt: u64 = 0,

    pub fn record(self: *Nvfp4RouteCounters, kind: Nvfp4RouteKind) void {
        switch (kind) {
            .native_cublaslt => self.native_cublaslt += 1,
            .bf16_dense_route => self.bf16_dense_route += 1,
            .small_rows_nvfp4_matvec => self.small_rows_nvfp4_matvec += 1,
            .small_rows_i8_matvec => self.small_rows_i8_matvec += 1,
            .fused_qkv_custom => self.fused_qkv_custom += 1,
            .fused_qkv_native_cublaslt => self.fused_qkv_native_cublaslt += 1,
            .fused_gate_up_custom => self.fused_gate_up_custom += 1,
            .fused_gate_up_native_cublaslt => self.fused_gate_up_native_cublaslt += 1,
        }
    }

    fn saturatingSub(current: u64, start: u64) u64 {
        return if (current >= start) current - start else 0;
    }

    pub fn delta(current: Nvfp4RouteCounters, start: Nvfp4RouteCounters) Nvfp4RouteCounters {
        return .{
            .native_cublaslt = saturatingSub(current.native_cublaslt, start.native_cublaslt),
            .bf16_dense_route = saturatingSub(current.bf16_dense_route, start.bf16_dense_route),
            .small_rows_nvfp4_matvec = saturatingSub(current.small_rows_nvfp4_matvec, start.small_rows_nvfp4_matvec),
            .small_rows_i8_matvec = saturatingSub(current.small_rows_i8_matvec, start.small_rows_i8_matvec),
            .fused_qkv_custom = saturatingSub(current.fused_qkv_custom, start.fused_qkv_custom),
            .fused_qkv_native_cublaslt = saturatingSub(current.fused_qkv_native_cublaslt, start.fused_qkv_native_cublaslt),
            .fused_gate_up_custom = saturatingSub(current.fused_gate_up_custom, start.fused_gate_up_custom),
            .fused_gate_up_native_cublaslt = saturatingSub(current.fused_gate_up_native_cublaslt, start.fused_gate_up_native_cublaslt),
        };
    }

    pub fn total(self: *const Nvfp4RouteCounters) u64 {
        const total_u128 = @as(u128, self.native_cublaslt) +
            @as(u128, self.bf16_dense_route) +
            @as(u128, self.small_rows_nvfp4_matvec) +
            @as(u128, self.small_rows_i8_matvec) +
            @as(u128, self.fused_qkv_custom) +
            @as(u128, self.fused_qkv_native_cublaslt) +
            @as(u128, self.fused_gate_up_custom) +
            @as(u128, self.fused_gate_up_native_cublaslt);
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
    attention_heads_lowbit_dequant_f16_kv: u64 = 0,
    attention_fused_heads_i8_kv: u64 = 0,
    attention_heads_i8_kv: u64 = 0,
    attention_fused_heads_fp8_kv: u64 = 0,
    attention_heads_fp8_kv: u64 = 0,
    attention_heads_f32_kv: u64 = 0,
    attention_fused_heads_f16_kv_ns: u64 = 0,
    attention_heads_f16_kv_ns: u64 = 0,
    attention_heads_lowbit_dequant_f16_kv_ns: u64 = 0,
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
            .heads_lowbit_dequant_f16_kv => {
                self.attention_heads_lowbit_dequant_f16_kv = saturatingAddU64(self.attention_heads_lowbit_dequant_f16_kv, 1);
                self.attention_heads_lowbit_dequant_f16_kv_ns = saturatingAddU64(self.attention_heads_lowbit_dequant_f16_kv_ns, elapsed_ns);
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
    /// `heads_lowbit_dequant_f16_kv` is also GEMM-based (`1-byte KV -> f16`).
    /// Custom fused kernels (`fused_heads_*`) are intentionally tracked in
    /// separate buckets and are not counted as tensor-core here.
    pub fn attentionTensorCoreNsApprox(self: *const Nvfp4PhaseBudgetCounters) u64 {
        const total_u128 = @as(u128, self.attention_heads_f16_kv_ns) +
            @as(u128, self.attention_heads_lowbit_dequant_f16_kv_ns);
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
            .attention_heads_lowbit_dequant_f16_kv = saturatingSub(current.attention_heads_lowbit_dequant_f16_kv, start.attention_heads_lowbit_dequant_f16_kv),
            .attention_fused_heads_i8_kv = saturatingSub(current.attention_fused_heads_i8_kv, start.attention_fused_heads_i8_kv),
            .attention_heads_i8_kv = saturatingSub(current.attention_heads_i8_kv, start.attention_heads_i8_kv),
            .attention_fused_heads_fp8_kv = saturatingSub(current.attention_fused_heads_fp8_kv, start.attention_fused_heads_fp8_kv),
            .attention_heads_fp8_kv = saturatingSub(current.attention_heads_fp8_kv, start.attention_heads_fp8_kv),
            .attention_heads_f32_kv = saturatingSub(current.attention_heads_f32_kv, start.attention_heads_f32_kv),
            .attention_fused_heads_f16_kv_ns = saturatingSub(current.attention_fused_heads_f16_kv_ns, start.attention_fused_heads_f16_kv_ns),
            .attention_heads_f16_kv_ns = saturatingSub(current.attention_heads_f16_kv_ns, start.attention_heads_f16_kv_ns),
            .attention_heads_lowbit_dequant_f16_kv_ns = saturatingSub(current.attention_heads_lowbit_dequant_f16_kv_ns, start.attention_heads_lowbit_dequant_f16_kv_ns),
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
    /// Low-bit KV prefill dequant route: dequant to f16 + GEMM f16 heads.
    heads_lowbit_dequant_f16_kv,
    /// Custom fused i8 KV kernel family.
    fused_heads_i8_kv,
    /// Heads-family i8 KV path (currently non-GEMM/scalar-style kernels).
    heads_i8_kv,
    /// Custom fused fp8 KV kernel family.
    fused_heads_fp8_kv,
    /// Heads-family fp8 KV path (currently non-GEMM/scalar-style kernels).
    heads_fp8_kv,
    /// Scalar f32 heads path.
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
