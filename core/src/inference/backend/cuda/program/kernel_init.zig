//! Layer program dispatch, adapter implementations, kernel initialization.
//!
//! Contains the per-instruction adapter implementations (norm, attention,
//! short conv, gated delta, SwiGLU, residual add), the layer program
//! dispatch loop, tryExecuteLayerProgram, runAttentionContext, kernel
//! function resolution, and CPU ROPE initialization.
//! Functions use `self: anytype` to avoid circular imports with engine.zig.

const std = @import("std");
const build_options = @import("build_options");
const compute = @import("compute_pkg");
const tensor = @import("tensor_pkg");
const dtype = @import("dtype_pkg");
const log = @import("log_pkg");
const trace = @import("xray_pkg").trace;
const models = @import("models_pkg");
const layer_ops = models.layer_ops;
const op_types = models.op_types;
const opcode_map = models.plan.opcode_map;
const rope_scaling = models.rope_scaling;
const runtime_contract = @import("runtime_contract_pkg");
const attention_mod = @import("../attention_path.zig");
const smoke_checks = @import("../selftest.zig");
const cpu_kernels = @import("../../cpu/kernels/root.zig");
const cpu_conv1d = compute.cpu.conv1d_depthwise;
const cpu_gated_delta = compute.cpu.gated_delta;

// --- Types from engine.zig (mutual import) ---
const engine = @import("../engine.zig");
const CudaBackend = engine.CudaBackend;
const LayerProgramExecutionContext = CudaBackend.LayerProgramExecutionContext;
const BuiltLayerProgramHandles = CudaBackend.BuiltLayerProgramHandles;
const LayerProgramInstructionStateBlocks = CudaBackend.LayerProgramInstructionStateBlocks;
const layer_program_adapter_table = CudaBackend.layer_program_adapter_table;
const traceShapeBsd = CudaBackend.traceShapeBsd;
const traceTokenIndex = CudaBackend.traceTokenIndex;
const tracePositionForPoint = CudaBackend.tracePositionForPoint;
const layerProgramExecutionState = CudaBackend.layerProgramExecutionState;
const layer_program_adapter_capabilities = CudaBackend.layer_program_adapter_capabilities;

// --- Shared types from engine_types.zig ---
const engine_types = @import("../runtime/_types_impl.zig");
const DeviceTensor = engine_types.DeviceTensor;
const LinearWeight = engine_types.LinearWeight;
const BlockRuntimeLayer = engine_types.BlockRuntimeLayer;
const BlockRuntime = engine_types.BlockRuntime;
const RuntimeBuffers = engine_types.RuntimeBuffers;
const LayerAttentionRuntime = engine_types.LayerAttentionRuntime;
const LayerAttentionExecConfig = engine_types.LayerAttentionExecConfig;
const ShortConvBlockRuntime = engine_types.ShortConvBlockRuntime;
const ShortConvExecConfig = engine_types.ShortConvExecConfig;
const GatedDeltaBlockRuntime = engine_types.GatedDeltaBlockRuntime;
const KvRuntimeState = engine_types.KvRuntimeState;
const ShortConvRuntimeState = engine_types.ShortConvRuntimeState;
const GatedDeltaRuntimeState = engine_types.GatedDeltaRuntimeState;
const AttentionKernelSet = engine_types.AttentionKernelSet;
const BatchDecodeInfo = engine_types.BatchDecodeInfo;
const KernelSlot = engine_types.KernelSlot;
const RequiredKernel = engine_types.RequiredKernel;
const required_kernels = engine_types.required_kernels;
const KvCacheDtype = engine_types.KvCacheDtype;
const enable_fused_attention_f16_kv = engine_types.enable_fused_attention_f16_kv;
const max_fused_attention_f16_kv_seq_len = engine_types.max_fused_attention_f16_kv_seq_len;
const max_supported_fused_f16_kv_head_dim = engine_types.max_supported_fused_f16_kv_head_dim;
const enable_dispatch_observability = engine_types.enable_dispatch_observability;
const attention_policy_config = engine_types.attention_policy_config;
const missing_device_tensor = engine_types.missing_device_tensor;
const expectedAttentionQProjectionDim = engine_types.expectedAttentionQProjectionDim;
const logicalF32RowSlice = engine_types.logicalF32RowSlice;
const GaffineU4LinearWeight = engine_types.GaffineU4LinearWeight;
const GaffineU8LinearWeight = engine_types.GaffineU8LinearWeight;
const Nvfp4LinearWeight = engine_types.Nvfp4LinearWeight;
const AttentionPath = engine_types.AttentionPath;
const gqa_prefill_f16_dynamic_smem_bytes: u32 = 65536;

// --- Compute ops from engine_ops.zig ---
const engine_ops = @import("../operators/root.zig");

// --- Mixer functions from engine_mixers.zig ---
const engine_mixers = @import("../operators/root.zig");

// --- Utilities from engine_weights.zig ---
const engine_weights = @import("../weights/root.zig");
const bufferSlice = engine_weights.bufferSlice;

const ResolvedAttentionShape = struct {
    head_dim_u32: u32,
    rope_dim_u32: u32,
    n_heads_u32: u32,
    n_kv_heads_u32: u32,
};

fn resolveAttentionShapeForInstruction(
    cfg: *const LayerAttentionExecConfig,
    q_norm_weight: ?*const DeviceTensor,
    k_norm_weight: ?*const DeviceTensor,
    ctx: *const LayerProgramExecutionContext,
) !ResolvedAttentionShape {
    var head_dim_u32 = ctx.head_dim_u32;
    if (q_norm_weight) |q_norm| {
        head_dim_u32 = @intCast(q_norm.rows);
    } else if (k_norm_weight) |k_norm| {
        head_dim_u32 = @intCast(k_norm.rows);
    }
    if (head_dim_u32 == 0) return error.InvalidInstructionBinding;
    const head_dim_usize: usize = @intCast(head_dim_u32);

    var n_heads_u32 = ctx.n_heads_u32;
    if ((cfg.q_dim % head_dim_usize) == 0) {
        n_heads_u32 = @intCast(cfg.q_dim / head_dim_usize);
    } else if (q_norm_weight != null) {
        return error.InvalidInstructionBinding;
    }

    var n_kv_heads_u32 = ctx.n_kv_heads_u32;
    if ((cfg.kv_dim % head_dim_usize) == 0) {
        n_kv_heads_u32 = @intCast(cfg.kv_dim / head_dim_usize);
    } else if (k_norm_weight != null) {
        return error.InvalidInstructionBinding;
    }
    if (n_heads_u32 == 0 or n_kv_heads_u32 == 0 or (n_heads_u32 % n_kv_heads_u32) != 0) {
        return error.InvalidInstructionBinding;
    }

    var rope_dim_u32 = @min(ctx.rope_dim_u32, head_dim_u32);
    // Mixed-attention models use proportional rotary width for full-attention
    // layers, but sliding-window layers run local RoPE across the full per-head
    // width.
    if (cfg.sliding_window > 0 and ctx.backend.loaded.config.global_head_dim > 0) {
        rope_dim_u32 = head_dim_u32;
    }

    return .{
        .head_dim_u32 = head_dim_u32,
        .rope_dim_u32 = rope_dim_u32,
        .n_heads_u32 = n_heads_u32,
        .n_kv_heads_u32 = n_kv_heads_u32,
    };
}

const handles = @import("handles.zig");
const tensorViewDescForCudaBuffer = handles.tensorViewDescForCudaBuffer;

pub fn initCpuRuntimeRopeHandles(self: anytype) !void {
    if (self.loaded.position_embeddings != null) return;
    if (self.rope_dim == 0) return;

    var global_freqs = try rope_scaling.materializeInverseFrequencies(
        self.allocator,
        self.rope_dim,
        self.loaded.config.rope_theta,
        self.loaded.config.rope_scaling,
    );
    defer global_freqs.deinit(self.allocator);

    const global_rope = try self.allocator.create(cpu_kernels.RoPE);
    errdefer self.allocator.destroy(global_rope);
    global_rope.* = try cpu_kernels.RoPE.initFromInvFreq(
        self.allocator,
        self.rope_dim,
        @intCast(self.max_seq_len),
        global_freqs.inv_freq,
        global_freqs.attention_scaling,
    );
    self.cpu_rope_global = global_rope;

    if (self.loaded.config.rope_local_theta > 0 and self.loaded.config.sliding_window > 0) {
        var local_freqs = try rope_scaling.materializeInverseFrequencies(
            self.allocator,
            self.rope_dim,
            self.loaded.config.rope_local_theta,
            self.loaded.config.rope_scaling,
        );
        defer local_freqs.deinit(self.allocator);

        const local_rope = try self.allocator.create(cpu_kernels.RoPE);
        errdefer self.allocator.destroy(local_rope);
        local_rope.* = try cpu_kernels.RoPE.initFromInvFreq(
            self.allocator,
            self.rope_dim,
            @intCast(self.max_seq_len),
            local_freqs.inv_freq,
            local_freqs.attention_scaling,
        );
        self.cpu_rope_local = local_rope;
    }
}

pub fn assignCpuRuntimeRopeToAttentionFallbacks(self: anytype) void {
    for (self.block_runtime.blocks) |*layer| {
        const block = layer.attention_binding orelse continue;
        if (block.cpu_kernel) |*kernel| {
            kernel.rope = if (kernel.sliding_window > 0 and self.cpu_rope_local != null)
                self.cpu_rope_local
            else
                self.cpu_rope_global;
        }
    }
}

pub fn initKernelFunctions(self: anytype) !void {
    if (!self.device.supportsModuleLaunch()) return;

    try self.kernel_registry.loadEmbeddedModule(compute.cuda.vector_add.embedded_module);
    const sideload_loaded = tryLoadSideloadModule(
        self,
    ) catch |err| blk: {
        log.warn("inference", "CUDA sideload unavailable; using embedded PTX", .{
            .reason = @errorName(err),
        });
        break :blk false;
    };
    if (sideload_loaded) {
        log.info("inference", "CUDA sideload kernel module active", .{});
    }

    try resolveRequiredKernels(
        self,
    );

    // Optional fusion kernel. If unavailable (e.g. stale sideload payload),
    // keep the canonical residual_add + rmsnorm split path.
    if (self.kernel_registry.resolveFunction(
        compute.cuda.residual_scaled_rmsnorm_rows_strided.op_name,
        compute.cuda.residual_scaled_rmsnorm_rows_strided.embedded_symbol,
    )) |resolved| {
        self.residual_scaled_rmsnorm_rows_strided_function = resolved.function;
        self.residual_scaled_rmsnorm_rows_strided_source = resolved.source;
    } else |_| {
        self.residual_scaled_rmsnorm_rows_strided_function = null;
        self.residual_scaled_rmsnorm_rows_strided_source = null;
    }
}

pub fn warmupDequantF16Cache(self: anytype) !void {
    // Resolve INT8 GEMM helper kernels (optional — graceful degradation to F16 path).
    if (self.kernel_registry.resolveFunction("quantize_f32_to_i8", "talu_quantize_f32_to_i8")) |resolved| {
        self.quantize_f32_to_i8_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("dequant_i32_gaffine", "talu_dequant_i32_gaffine")) |resolved| {
        self.dequant_i32_gaffine_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("i8_rowsum", "talu_i8_rowsum")) |resolved| {
        self.i8_rowsum_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("u8_xor_to_i8", "talu_u8_xor_to_i8")) |resolved| {
        self.u8_xor_to_i8_function = resolved.function;
    } else |_| {}
    // Symmetric INT8 kernels.
    if (self.kernel_registry.resolveFunction("i8_matvec_f32", "talu_i8_matvec_f32")) |resolved| {
        self.i8_matvec_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("gaffine_u8_to_i8", "talu_gaffine_u8_to_i8")) |resolved| {
        self.gaffine_u8_to_i8_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("gaffine_u4_to_i8", "talu_gaffine_u4_to_i8")) |resolved| {
        self.gaffine_u4_to_i8_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("nvfp4_to_i8", "talu_nvfp4_to_i8")) |resolved| {
        self.nvfp4_to_i8_function = resolved.function;
    } else |err| {
        log.warn("inference", "CUDA NVFP4->I8 kernel resolve failed", .{
            .reason = @errorName(err),
        });
    }
    if (self.kernel_registry.resolveFunction("quantize_f16_to_i8", "talu_quantize_f16_to_i8")) |resolved| {
        self.quantize_f16_to_i8_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("quantize_f32_to_i8_simple", "talu_quantize_f32_to_i8_simple")) |resolved| {
        self.quantize_f32_to_i8_simple_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("dequant_i32_scales", "talu_dequant_i32_scales")) |resolved| {
        self.dequant_i32_scales_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("dequant_i32_scales_split3", "talu_dequant_i32_scales_split3")) |resolved| {
        self.dequant_i32_scales_split3_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("quantize_f32_to_fp8_e4m3", "talu_quantize_f32_to_fp8_e4m3")) |resolved| {
        self.quantize_f32_to_fp8_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("quantize_f32_to_mxfp8", "talu_quantize_f32_to_mxfp8")) |resolved| {
        self.quantize_f32_to_mxfp8_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("quantize_f32_to_nvfp4", "talu_quantize_f32_to_nvfp4")) |resolved| {
        self.quantize_f32_to_nvfp4_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("dequant_mxfp8_to_bf16", "talu_dequant_mxfp8_to_bf16")) |resolved| {
        self.mxfp8_dequant_to_bf16_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("dequant_nvfp4_to_bf16", "talu_dequant_nvfp4_to_bf16")) |resolved| {
        self.nvfp4_dequant_to_bf16_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("scale_rows_f32", "talu_scale_rows_f32")) |resolved| {
        self.scale_rows_f32_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("fp8_e4m3_matvec_f32", "talu_fp8_e4m3_matvec_f32")) |resolved| {
        self.fp8_matvec_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("dequant_fp8_e4m3_to_bf16", "talu_dequant_fp8_e4m3_to_bf16")) |resolved| {
        self.fp8_dequant_to_bf16_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("fp8_e4m3_matvec_gate_up_silu_f32", "talu_fp8_e4m3_matvec_gate_up_silu_f32")) |resolved| {
        self.fp8_matvec_gate_up_silu_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("fp8_e4m3_matvec_gate_up_f32", "talu_fp8_e4m3_matvec_gate_up_f32")) |resolved| {
        self.fp8_matvec_gate_up_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("fp8_e4m3_matvec_f32_tile8", "talu_fp8_e4m3_matvec_f32_tile8")) |resolved| {
        self.fp8_matvec_tile8_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("fp8_e4m3_matvec_gate_up_silu_f32_tile8", "talu_fp8_e4m3_matvec_gate_up_silu_f32_tile8")) |resolved| {
        self.fp8_matvec_gate_up_silu_tile8_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("fp8_e4m3_matvec_gate_up_f32_tile8", "talu_fp8_e4m3_matvec_gate_up_f32_tile8")) |resolved| {
        self.fp8_matvec_gate_up_tile8_function = resolved.function;
    } else |_| {}

    if (self.kernel_registry.resolveFunction("mxfp8_matvec_f32", "talu_mxfp8_matvec_f32")) |resolved| {
        self.mxfp8_matvec_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("mxfp8_matvec_f32_tile8", "talu_mxfp8_matvec_f32_tile8")) |resolved| {
        self.mxfp8_matvec_tile8_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("nvfp4_matvec_f32", "talu_nvfp4_matvec_f32")) |resolved| {
        self.nvfp4_matvec_function = resolved.function;
    } else |err| {
        log.warn("inference", "CUDA NVFP4 matvec resolve failed", .{
            .reason = @errorName(err),
        });
    }
    if (self.kernel_registry.resolveFunction("nvfp4_matvec_f32_tile8", "talu_nvfp4_matvec_f32_tile8")) |resolved| {
        self.nvfp4_matvec_tile8_function = resolved.function;
    } else |err| {
        log.warn("inference", "CUDA NVFP4 matvec tile8 resolve failed", .{
            .reason = @errorName(err),
        });
    }
    if (self.kernel_registry.resolveFunction("nvfp4_matvec_qkv_f32", "talu_nvfp4_matvec_qkv_f32")) |resolved| {
        self.nvfp4_matvec_qkv_function = resolved.function;
    } else |err| {
        log.warn("inference", "CUDA NVFP4 qkv resolve failed", .{
            .reason = @errorName(err),
        });
    }
    if (self.kernel_registry.resolveFunction("nvfp4_matvec_qkv_f32_tile8", "talu_nvfp4_matvec_qkv_f32_tile8")) |resolved| {
        self.nvfp4_matvec_qkv_tile8_function = resolved.function;
    } else |err| {
        log.warn("inference", "CUDA NVFP4 qkv tile8 resolve failed", .{
            .reason = @errorName(err),
        });
    }
    if (self.kernel_registry.resolveFunction("nvfp4_matvec_gate_up_f32", "talu_nvfp4_matvec_gate_up_f32")) |resolved| {
        self.nvfp4_matvec_gate_up_function = resolved.function;
    } else |err| {
        log.warn("inference", "CUDA NVFP4 gate_up resolve failed", .{
            .reason = @errorName(err),
        });
    }
    if (self.kernel_registry.resolveFunction("nvfp4_matvec_gate_up_f32_tile8", "talu_nvfp4_matvec_gate_up_f32_tile8")) |resolved| {
        self.nvfp4_matvec_gate_up_tile8_function = resolved.function;
    } else |err| {
        log.warn("inference", "CUDA NVFP4 gate_up tile8 resolve failed", .{
            .reason = @errorName(err),
        });
    }
    if (self.kernel_registry.resolveFunction("nvfp4_matvec_gate_up_silu_f32", "talu_nvfp4_matvec_gate_up_silu_f32")) |resolved| {
        self.nvfp4_matvec_gate_up_silu_function = resolved.function;
    } else |err| {
        log.warn("inference", "CUDA NVFP4 gate_up_silu resolve failed", .{
            .reason = @errorName(err),
        });
    }
    if (self.kernel_registry.resolveFunction("nvfp4_matvec_gate_up_silu_f32_tile8", "talu_nvfp4_matvec_gate_up_silu_f32_tile8")) |resolved| {
        self.nvfp4_matvec_gate_up_silu_tile8_function = resolved.function;
    } else |err| {
        log.warn("inference", "CUDA NVFP4 gate_up_silu tile8 resolve failed", .{
            .reason = @errorName(err),
        });
    }
    if (self.kernel_registry.resolveFunction("nvfp4_matvec_gate_up_gelu_f32", "talu_nvfp4_matvec_gate_up_gelu_f32")) |resolved| {
        self.nvfp4_matvec_gate_up_gelu_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("nvfp4_matvec_gate_up_gelu_f32_tile8", "talu_nvfp4_matvec_gate_up_gelu_f32_tile8")) |resolved| {
        self.nvfp4_matvec_gate_up_gelu_tile8_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("mxfp8_matvec_gate_up_silu_f32", "talu_mxfp8_matvec_gate_up_silu_f32")) |resolved| {
        self.mxfp8_matvec_gate_up_silu_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("mxfp8_matvec_gate_up_silu_f32_tile8", "talu_mxfp8_matvec_gate_up_silu_f32_tile8")) |resolved| {
        self.mxfp8_matvec_gate_up_silu_tile8_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("mxfp8_matvec_gate_up_f32", "talu_mxfp8_matvec_gate_up_f32")) |resolved| {
        self.mxfp8_matvec_gate_up_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("mxfp8_matvec_gate_up_f32_tile8", "talu_mxfp8_matvec_gate_up_f32_tile8")) |resolved| {
        self.mxfp8_matvec_gate_up_tile8_function = resolved.function;
    } else |_| {}

    const has_u8_dequant = self.gaffine_u8_dequant_f16_function != null;
    const has_u4_dequant = self.gaffine_u4_dequant_f16_function != null;
    const has_u4_to_i8 = self.gaffine_u4_to_i8_function != null;
    const has_u8_to_i8 = self.gaffine_u8_to_i8_function != null;
    const has_nvfp4_to_i8 = self.nvfp4_to_i8_function != null;
    // NVFP4->I8 cache can improve some decode kernels but substantially
    // increases VRAM footprint; keep disabled by default until budgets and
    // route quality are validated across long-context prefill.
    const enable_nvfp4_i8_cache = has_nvfp4_to_i8 and (std.posix.getenv("TALU_NVFP4_I8_CACHE") != null);
    if (!has_u8_dequant and !has_u4_dequant and !has_u4_to_i8 and !has_u8_to_i8 and !enable_nvfp4_i8_cache) {
        return;
    }

    // Estimate total dequant cache bytes and skip if insufficient VRAM remains.
    // The I8 cache accelerates prefill and can unlock fused/smaller-kernel decode
    // routes for low-bit weights. It is optional and skipped when memory headroom
    // is insufficient.
    {
        var estimated_cache_bytes: usize = 0;
        const countLowBitI8CacheBytes = struct {
            fn run(weight: *const LinearWeight, nvfp4_i8_enabled: bool) usize {
                return switch (weight.*) {
                    // I8: rows*cols + cols*4 (scale). Use rows*cols as lower bound.
                    .gaffine_u4, .gaffine_u8 => |w| std.math.mul(usize, w.rows, w.cols) catch 0,
                    .nvfp4 => |w| if (nvfp4_i8_enabled) std.math.mul(usize, w.rows, w.cols) catch 0 else 0,
                    else => 0,
                };
            }
        }.run;
        for (self.block_runtime.blocks) |*layer| {
            if (layer.attention_runtime) |*attn| {
                estimated_cache_bytes = engine_types.saturatingAddUsize(estimated_cache_bytes, countLowBitI8CacheBytes(&attn.q_proj, enable_nvfp4_i8_cache));
                estimated_cache_bytes = engine_types.saturatingAddUsize(estimated_cache_bytes, countLowBitI8CacheBytes(&attn.k_proj, enable_nvfp4_i8_cache));
                estimated_cache_bytes = engine_types.saturatingAddUsize(estimated_cache_bytes, countLowBitI8CacheBytes(&attn.v_proj, enable_nvfp4_i8_cache));
                estimated_cache_bytes = engine_types.saturatingAddUsize(estimated_cache_bytes, countLowBitI8CacheBytes(&attn.o_proj, enable_nvfp4_i8_cache));
                estimated_cache_bytes = engine_types.saturatingAddUsize(estimated_cache_bytes, countLowBitI8CacheBytes(&attn.w1, enable_nvfp4_i8_cache));
                estimated_cache_bytes = engine_types.saturatingAddUsize(estimated_cache_bytes, countLowBitI8CacheBytes(&attn.w2, enable_nvfp4_i8_cache));
                estimated_cache_bytes = engine_types.saturatingAddUsize(estimated_cache_bytes, countLowBitI8CacheBytes(&attn.w3, enable_nvfp4_i8_cache));
            }
            if (layer.shortconv_runtime) |*sc| {
                estimated_cache_bytes = engine_types.saturatingAddUsize(estimated_cache_bytes, countLowBitI8CacheBytes(&sc.in_proj, enable_nvfp4_i8_cache));
                estimated_cache_bytes = engine_types.saturatingAddUsize(estimated_cache_bytes, countLowBitI8CacheBytes(&sc.out_proj, enable_nvfp4_i8_cache));
            }
            if (layer.gated_delta_runtime) |*gd| {
                estimated_cache_bytes = engine_types.saturatingAddUsize(estimated_cache_bytes, countLowBitI8CacheBytes(&gd.in_proj, enable_nvfp4_i8_cache));
                estimated_cache_bytes = engine_types.saturatingAddUsize(estimated_cache_bytes, countLowBitI8CacheBytes(&gd.out_proj, enable_nvfp4_i8_cache));
            }
        }
        estimated_cache_bytes = engine_types.saturatingAddUsize(estimated_cache_bytes, countLowBitI8CacheBytes(&self.runtime_buffers.projection_weight, enable_nvfp4_i8_cache));
        // Add overhead for per-col scale buffers and concatenated QKV caches.
        estimated_cache_bytes = engine_types.saturatingAddUsize(estimated_cache_bytes, estimated_cache_bytes / 4);

        if (self.device.memoryInfo()) |mem_info| {
            // Keep a fixed safety headroom for runtime allocations and KV growth.
            // The previous free/2 cap was too conservative for medium NVFP4 models
            // (e.g. 4B) and prevented decode caches from being created at all.
            const safety_bytes: usize = 2 * 1024 * 1024 * 1024; // 2 GiB
            const cache_budget = if (mem_info.free > safety_bytes)
                mem_info.free - safety_bytes
            else
                0;
            if (estimated_cache_bytes > cache_budget) {
                return;
            }
        } else |_| {}
    }

    var total_bytes: usize = 0;
    var weight_count: usize = 0;

    // Helper to launch a fused gaffine→I8 dequant kernel for a single weight.
    // Shared by both U8 and U4 paths — only the kernel function differs.
    const launchFusedToI8 = struct {
        fn run(
            backend: *CudaBackend,
            fused_fn: compute.cuda.Function,
            w: *GaffineU4LinearWeight,
            bytes_out: *usize,
        ) void {
            const weight_elems = std.math.mul(usize, w.rows, w.cols) catch return;
            if (weight_elems == 0) return;
            const i8_bytes = weight_elems;
            const scale_bytes = std.math.mul(usize, w.cols, @sizeOf(f32)) catch return;

            var i8_buf = backend.device.allocBuffer(i8_bytes) catch return;
            var scale_buf = backend.device.allocBuffer(scale_bytes) catch {
                i8_buf.deinit(&backend.device);
                return;
            };

            // Launch: grid=(out_dim=w.cols), block=(256)
            backend.kernel_arg_pack.reset();
            backend.kernel_arg_pack.appendBufferPtr(&w.packed_data) catch {
                scale_buf.deinit(&backend.device);
                i8_buf.deinit(&backend.device);
                return;
            };
            backend.kernel_arg_pack.appendBufferPtr(&w.scales) catch {
                scale_buf.deinit(&backend.device);
                i8_buf.deinit(&backend.device);
                return;
            };
            backend.kernel_arg_pack.appendBufferPtr(&w.biases) catch {
                scale_buf.deinit(&backend.device);
                i8_buf.deinit(&backend.device);
                return;
            };
            backend.kernel_arg_pack.appendBufferPtr(&i8_buf) catch {
                scale_buf.deinit(&backend.device);
                i8_buf.deinit(&backend.device);
                return;
            };
            backend.kernel_arg_pack.appendBufferPtr(&scale_buf) catch {
                scale_buf.deinit(&backend.device);
                i8_buf.deinit(&backend.device);
                return;
            };
            backend.kernel_arg_pack.appendScalar(u32, @intCast(w.rows)) catch {
                scale_buf.deinit(&backend.device);
                i8_buf.deinit(&backend.device);
                return;
            };
            backend.kernel_arg_pack.appendScalar(u32, w.group_size) catch {
                scale_buf.deinit(&backend.device);
                i8_buf.deinit(&backend.device);
                return;
            };
            backend.kernel_arg_pack.appendScalar(u32, w.scales_dtype_tag) catch {
                scale_buf.deinit(&backend.device);
                i8_buf.deinit(&backend.device);
                return;
            };
            compute.cuda.launch.launchWithFamily(&backend.device, fused_fn, .{
                .grid_x = @intCast(w.cols),
                .block_x = 256,
            }, &backend.kernel_arg_pack, .other) catch {
                scale_buf.deinit(&backend.device);
                i8_buf.deinit(&backend.device);
                return;
            };

            w.dequant_i8_cache = i8_buf;
            w.mean_scale_cache = scale_buf;
            bytes_out.* += i8_bytes + scale_bytes;
        }
    }.run;

    // Helper to create I8 cache for a single gaffine_u8 weight.
    // Prefers fused U8→I8 kernel (no F16 intermediate), falls back to F16→I8.
    const dequantU8Weight = struct {
        fn run(
            backend: *CudaBackend,
            w: *GaffineU8LinearWeight,
            bytes_out: *usize,
        ) void {
            // Try fused U8→I8 path (no F16 intermediate, saves ~50% VRAM).
            if (backend.gaffine_u8_to_i8_function) |fused_fn| {
                launchFusedToI8(backend, fused_fn, w, bytes_out);
                return;
            }

            // Fallback: dequant to F16 cache (for F16 GEMM path).
            const dequant_f16_fn = backend.gaffine_u8_dequant_f16_function orelse return;
            const weight_elems = std.math.mul(usize, w.rows, w.cols) catch return;
            if (weight_elems == 0) return;
            const weight_f16_bytes = std.math.mul(usize, weight_elems, @sizeOf(u16)) catch return;
            var cache_buf = backend.device.allocBuffer(weight_f16_bytes) catch return;
            errdefer cache_buf.deinit(&backend.device);

            compute.cuda.gaffine_u8_dequantize_f16.runWithFunction(
                &backend.kernel_arg_pack,
                &backend.device,
                dequant_f16_fn,
                &w.packed_data,
                &w.scales,
                &w.biases,
                &cache_buf,
                @intCast(w.cols),
                @intCast(w.rows),
                w.group_size,
                w.scales_dtype_tag,
            ) catch {
                cache_buf.deinit(&backend.device);
                return;
            };

            w.dequant_f16_cache = cache_buf;
            bytes_out.* += weight_f16_bytes;
        }
    }.run;

    // Helper to create I8 cache for a single NVFP4 weight.
    // This is used by decode fused I8 paths and prefill tensor-core I8 paths.
    const dequantNvfp4Weight = struct {
        fn run(
            backend: *CudaBackend,
            w: *Nvfp4LinearWeight,
            bytes_out: *usize,
        ) void {
            const dequant_i8_fn = backend.nvfp4_to_i8_function orelse return;
            const weight_elems = std.math.mul(usize, w.rows, w.cols) catch return;
            if (weight_elems == 0) return;
            const i8_bytes = weight_elems;
            const scale_bytes = std.math.mul(usize, w.cols, @sizeOf(f32)) catch return;

            var i8_buf = backend.device.allocBuffer(i8_bytes) catch return;
            var scale_buf = backend.device.allocBuffer(scale_bytes) catch {
                i8_buf.deinit(&backend.device);
                return;
            };

            backend.kernel_arg_pack.reset();
            backend.kernel_arg_pack.appendBufferPtr(&w.buffer) catch {
                scale_buf.deinit(&backend.device);
                i8_buf.deinit(&backend.device);
                return;
            };
            backend.kernel_arg_pack.appendBufferPtr(&w.scales_buffer) catch {
                scale_buf.deinit(&backend.device);
                i8_buf.deinit(&backend.device);
                return;
            };
            backend.kernel_arg_pack.appendBufferPtr(&i8_buf) catch {
                scale_buf.deinit(&backend.device);
                i8_buf.deinit(&backend.device);
                return;
            };
            backend.kernel_arg_pack.appendBufferPtr(&scale_buf) catch {
                scale_buf.deinit(&backend.device);
                i8_buf.deinit(&backend.device);
                return;
            };
            backend.kernel_arg_pack.appendScalar(u32, @intCast(w.rows)) catch {
                scale_buf.deinit(&backend.device);
                i8_buf.deinit(&backend.device);
                return;
            };
            backend.kernel_arg_pack.appendScalar(u32, @intCast(w.scale_cols)) catch {
                scale_buf.deinit(&backend.device);
                i8_buf.deinit(&backend.device);
                return;
            };
            backend.kernel_arg_pack.appendScalar(f32, w.weight_global_scale) catch {
                scale_buf.deinit(&backend.device);
                i8_buf.deinit(&backend.device);
                return;
            };

            compute.cuda.launch.launchWithFamily(&backend.device, dequant_i8_fn, .{
                .grid_x = @intCast(w.cols),
                .block_x = 256,
            }, &backend.kernel_arg_pack, .other) catch {
                scale_buf.deinit(&backend.device);
                i8_buf.deinit(&backend.device);
                return;
            };

            w.dequant_i8_cache = i8_buf;
            w.mean_scale_cache = scale_buf;
            bytes_out.* += i8_bytes + scale_bytes;
        }
    }.run;

    // Helper to process a low-bit LinearWeight and materialize I8 caches.
    const maybeProcess = struct {
        fn run(
            backend: *CudaBackend,
            weight: *LinearWeight,
            nvfp4_i8_enabled: bool,
            bytes_out: *usize,
            count_out: *usize,
        ) void {
            switch (weight.*) {
                .gaffine_u8 => |*w| {
                    dequantU8Weight(backend, w, bytes_out);
                    count_out.* += 1;
                },
                .gaffine_u4 => |*w| {
                    if (backend.gaffine_u4_to_i8_function) |fused_fn| {
                        launchFusedToI8(backend, fused_fn, w, bytes_out);
                        count_out.* += 1;
                    }
                },
                .nvfp4 => |*w| {
                    if (nvfp4_i8_enabled) {
                        dequantNvfp4Weight(backend, w, bytes_out);
                        count_out.* += 1;
                    }
                },
                else => {},
            }
        }
    }.run;

    // Helper to process an optional LinearWeight.
    const maybeProcessOpt = struct {
        fn run(
            backend: *CudaBackend,
            weight_opt: *?LinearWeight,
            nvfp4_i8_enabled: bool,
            bytes_out: *usize,
            count_out: *usize,
        ) void {
            if (weight_opt.*) |*w| {
                maybeProcess(backend, w, nvfp4_i8_enabled, bytes_out, count_out);
            }
        }
    }.run;

    for (self.block_runtime.blocks) |*layer| {
        if (layer.attention_runtime) |*attn| {
            maybeProcess(self, &attn.q_proj, enable_nvfp4_i8_cache, &total_bytes, &weight_count);
            maybeProcess(self, &attn.k_proj, enable_nvfp4_i8_cache, &total_bytes, &weight_count);
            maybeProcess(self, &attn.v_proj, enable_nvfp4_i8_cache, &total_bytes, &weight_count);
            maybeProcess(self, &attn.o_proj, enable_nvfp4_i8_cache, &total_bytes, &weight_count);
            maybeProcess(self, &attn.w1, enable_nvfp4_i8_cache, &total_bytes, &weight_count);
            maybeProcess(self, &attn.w2, enable_nvfp4_i8_cache, &total_bytes, &weight_count);
            maybeProcess(self, &attn.w3, enable_nvfp4_i8_cache, &total_bytes, &weight_count);
        }
        if (layer.shortconv_runtime) |*sc| {
            maybeProcess(self, &sc.in_proj, enable_nvfp4_i8_cache, &total_bytes, &weight_count);
            maybeProcess(self, &sc.out_proj, enable_nvfp4_i8_cache, &total_bytes, &weight_count);
            maybeProcessOpt(self, &sc.ffn_w1, enable_nvfp4_i8_cache, &total_bytes, &weight_count);
            maybeProcessOpt(self, &sc.ffn_w2, enable_nvfp4_i8_cache, &total_bytes, &weight_count);
            maybeProcessOpt(self, &sc.ffn_w3, enable_nvfp4_i8_cache, &total_bytes, &weight_count);
        }
        if (layer.gated_delta_runtime) |*gd| {
            maybeProcess(self, &gd.in_proj, enable_nvfp4_i8_cache, &total_bytes, &weight_count);
            maybeProcess(self, &gd.out_proj, enable_nvfp4_i8_cache, &total_bytes, &weight_count);
            maybeProcessOpt(self, &gd.ffn_w1, enable_nvfp4_i8_cache, &total_bytes, &weight_count);
            maybeProcessOpt(self, &gd.ffn_w2, enable_nvfp4_i8_cache, &total_bytes, &weight_count);
            maybeProcessOpt(self, &gd.ffn_w3, enable_nvfp4_i8_cache, &total_bytes, &weight_count);
        }
    }

    // Projection weight (lm_head).
    maybeProcess(self, &self.runtime_buffers.projection_weight, enable_nvfp4_i8_cache, &total_bytes, &weight_count);

    // Build concatenated I8 QKV caches for fused prefill GEMM.
    // This merges Q+K+V I8 weights into one contiguous buffer so prefill
    // can run a single large GEMM instead of 3 separate ones.
    const I8CacheRef = struct { i8_buf: compute.cuda.Buffer, scales_buf: compute.cuda.Buffer };
    const getI8Cache = struct {
        fn get(weight: *const LinearWeight) ?I8CacheRef {
            return switch (weight.*) {
                .gaffine_u4 => |w| if (w.dequant_i8_cache.pointer != 0 and w.mean_scale_cache.pointer != 0)
                    .{ .i8_buf = w.dequant_i8_cache, .scales_buf = w.mean_scale_cache }
                else
                    null,
                .gaffine_u8 => |w| if (w.dequant_i8_cache.pointer != 0 and w.mean_scale_cache.pointer != 0)
                    .{ .i8_buf = w.dequant_i8_cache, .scales_buf = w.mean_scale_cache }
                else
                    null,
                .nvfp4 => |w| if (w.dequant_i8_cache.pointer != 0 and w.mean_scale_cache.pointer != 0)
                    .{ .i8_buf = w.dequant_i8_cache, .scales_buf = w.mean_scale_cache }
                else
                    null,
                else => null,
            };
        }
    }.get;

    for (self.block_runtime.blocks) |*layer| {
        const attn = &(layer.attention_runtime orelse continue);
        const q_ref = getI8Cache(&attn.q_proj) orelse continue;
        const k_ref = getI8Cache(&attn.k_proj) orelse continue;
        const v_ref = getI8Cache(&attn.v_proj) orelse continue;

        const in_dim = attn.q_proj.rows();
        if (in_dim != attn.k_proj.rows() or in_dim != attn.v_proj.rows()) continue;
        if (in_dim == 0) continue;

        const q_dim: u32 = @intCast(attn.q_proj.cols());
        const k_dim: u32 = @intCast(attn.k_proj.cols());
        const v_dim: u32 = @intCast(attn.v_proj.cols());

        const total_dim: usize = @as(usize, q_dim) + k_dim + v_dim;
        const i8_bytes = std.math.mul(usize, total_dim, in_dim) catch continue;
        const scales_bytes = std.math.mul(usize, total_dim, @sizeOf(f32)) catch continue;

        var concat_i8 = self.device.allocBuffer(i8_bytes) catch continue;
        var concat_scales = self.device.allocBuffer(scales_bytes) catch {
            concat_i8.deinit(&self.device);
            continue;
        };

        // D2D copy each weight's I8 cache into the concatenated buffer.
        const q_i8_bytes = @as(usize, q_dim) * in_dim;
        const k_i8_bytes = @as(usize, k_dim) * in_dim;
        const v_i8_bytes = @as(usize, v_dim) * in_dim;
        const q_scale_bytes = @as(usize, q_dim) * @sizeOf(f32);
        const k_scale_bytes = @as(usize, k_dim) * @sizeOf(f32);
        const v_scale_bytes = @as(usize, v_dim) * @sizeOf(f32);

        const ok = blk: {
            var dst = bufferSlice(&concat_i8, 0, q_i8_bytes) catch break :blk false;
            dst.copyFrom(&self.device, &q_ref.i8_buf, q_i8_bytes) catch break :blk false;
            dst = bufferSlice(&concat_i8, q_i8_bytes, k_i8_bytes) catch break :blk false;
            dst.copyFrom(&self.device, &k_ref.i8_buf, k_i8_bytes) catch break :blk false;
            dst = bufferSlice(&concat_i8, q_i8_bytes + k_i8_bytes, v_i8_bytes) catch break :blk false;
            dst.copyFrom(&self.device, &v_ref.i8_buf, v_i8_bytes) catch break :blk false;

            dst = bufferSlice(&concat_scales, 0, q_scale_bytes) catch break :blk false;
            dst.copyFrom(&self.device, &q_ref.scales_buf, q_scale_bytes) catch break :blk false;
            dst = bufferSlice(&concat_scales, q_scale_bytes, k_scale_bytes) catch break :blk false;
            dst.copyFrom(&self.device, &k_ref.scales_buf, k_scale_bytes) catch break :blk false;
            dst = bufferSlice(&concat_scales, q_scale_bytes + k_scale_bytes, v_scale_bytes) catch break :blk false;
            dst.copyFrom(&self.device, &v_ref.scales_buf, v_scale_bytes) catch break :blk false;
            break :blk true;
        };

        if (ok) {
            attn.qkv_i8_concat = concat_i8;
            attn.qkv_scales_concat = concat_scales;
            attn.qkv_concat_dims = .{ q_dim, k_dim, v_dim };
            total_bytes += i8_bytes + scales_bytes;
        } else {
            concat_scales.deinit(&self.device);
            concat_i8.deinit(&self.device);
        }
    }

    self.dequant_cache_bytes = total_bytes;
    if (weight_count > 0) {
        try self.device.synchronize();
        log.info("inference", "CUDA low-bit i8 cache ready", .{
            .weights = weight_count,
            .cache_mib = total_bytes / (1024 * 1024),
        });
    }
}

pub fn resolveRequiredKernels(self: anytype) !void {
    for (required_kernels) |kernel| {
        const resolved = self.kernel_registry.resolveFunction(
            kernel.op_name,
            kernel.embedded_symbol,
        ) catch |err| {
            log.warn("inference", "CUDA kernel resolve failed", .{
                .op = kernel.op_name,
                .symbol = kernel.embedded_symbol,
                .reason = @errorName(err),
            });
            return err;
        };
        assignResolvedKernel(self, kernel.slot, resolved);
    }
}

pub fn assignResolvedKernel(
    self: anytype,
    slot: KernelSlot,
    resolved: compute.cuda.registry.ResolvedFunction,
) void {
    switch (slot) {
        .vector_add => {
            self.vector_add_function = resolved.function;
            self.vector_add_source = resolved.source;
        },
        .vector_add_scaled => {
            self.vector_add_scaled_function = resolved.function;
            self.vector_add_scaled_source = resolved.source;
        },
        .vector_add_rows_strided => {
            self.vector_add_rows_strided_function = resolved.function;
            self.vector_add_rows_strided_source = resolved.source;
        },
        .vector_add_scaled_rows_strided => {
            self.vector_add_scaled_rows_strided_function = resolved.function;
            self.vector_add_scaled_rows_strided_source = resolved.source;
        },
        .mul => {
            self.mul_function = resolved.function;
            self.mul_source = resolved.source;
        },
        .copy => {
            self.copy_function = resolved.function;
            self.copy_source = resolved.source;
        },
        .copy_u16 => {
            self.copy_u16_function = resolved.function;
            self.copy_u16_source = resolved.source;
        },
        .cast_f32_to_f16 => {
            self.cast_f32_to_f16_function = resolved.function;
            self.cast_f32_to_f16_source = resolved.source;
        },
        .cast_f32_to_bf16 => {
            self.cast_f32_to_bf16_function = resolved.function;
            self.cast_f32_to_bf16_source = resolved.source;
        },
        .cast_bf16_to_f32 => {
            self.cast_bf16_to_f32_function = resolved.function;
        },
        .embedding_lookup_f32 => {
            self.embedding_lookup_f32_function = resolved.function;
            self.embedding_lookup_f32_source = resolved.source;
        },
        .embedding_lookup_u16 => {
            self.embedding_lookup_u16_function = resolved.function;
            self.embedding_lookup_u16_source = resolved.source;
        },
        .embedding_lookup_u16_rows => {
            self.embedding_lookup_u16_rows_function = resolved.function;
            self.embedding_lookup_u16_rows_source = resolved.source;
        },
        .embedding_lookup_gaffine_u4 => {
            self.embedding_lookup_gaffine_u4_function = resolved.function;
            self.embedding_lookup_gaffine_u4_source = resolved.source;
        },
        .kv_write_f16 => {
            self.kv_write_f16_function = resolved.function;
            self.kv_write_f16_source = resolved.source;
        },
        .kv_write_f16_rows => {
            self.kv_write_f16_rows_function = resolved.function;
            self.kv_write_f16_rows_source = resolved.source;
        },
        .kv_write_f16_rows_ptrs => {
            self.kv_write_f16_rows_ptrs_function = resolved.function;
            self.kv_write_f16_rows_ptrs_source = resolved.source;
        },
        .rmsnorm => {
            self.rmsnorm_function = resolved.function;
            self.rmsnorm_source = resolved.source;
        },
        .rmsnorm_rows_strided => {
            self.rmsnorm_rows_strided_function = resolved.function;
            self.rmsnorm_rows_strided_source = resolved.source;
        },
        .rope => {
            self.rope_function = resolved.function;
            self.rope_source = resolved.source;
        },
        .rope_store_f16 => {
            self.rope_store_f16_function = resolved.function;
            self.rope_store_f16_source = resolved.source;
        },
        .attn_scores_heads_f32 => {
            self.attn_scores_heads_f32_function = resolved.function;
            self.attn_scores_heads_f32_source = resolved.source;
        },
        .attn_scores_heads_f16_kv => {
            self.attn_scores_heads_f16_kv_function = resolved.function;
            self.attn_scores_heads_f16_kv_source = resolved.source;
        },
        .attn_fused_heads_f16_kv => {
            self.attn_fused_heads_f16_kv_function = resolved.function;
            self.attn_fused_heads_f16_kv_source = resolved.source;
        },
        .attn_fused_decode_heads_f16_kv_ptrs => {
            self.attn_fused_decode_heads_f16_kv_ptrs_function = resolved.function;
            self.attn_fused_decode_heads_f16_kv_ptrs_source = resolved.source;
        },
        .attn_fused_prefill_heads_f16_kv => {
            self.attn_fused_prefill_heads_f16_kv_function = resolved.function;
            self.attn_fused_prefill_heads_f16_kv_source = resolved.source;
        },
        .attn_fused_prefill_heads_f16_kv_gqa => {
            self.attn_fused_prefill_heads_f16_kv_gqa_function = resolved.function;
            self.attn_fused_prefill_heads_f16_kv_gqa_source = resolved.source;
            self.device.setFunctionMaxDynamicSharedMemory(
                resolved.function.handle,
                gqa_prefill_f16_dynamic_smem_bytes,
            ) catch |err| {
                log.warn("inference", "CUDA could not raise dynamic shared memory for fused prefill f16 GQA", .{
                    .requested_bytes = gqa_prefill_f16_dynamic_smem_bytes,
                    .reason = @errorName(err),
                });
            };
        },
        .causal_attn_softmax_f32 => {
            self.causal_attn_softmax_f32_function = resolved.function;
            self.causal_attn_softmax_f32_source = resolved.source;
        },
        .softmax_rows => {
            self.softmax_rows_function = resolved.function;
            self.softmax_rows_source = resolved.source;
        },
        .attn_weighted_sum_heads_f32 => {
            self.attn_weighted_sum_heads_f32_function = resolved.function;
            self.attn_weighted_sum_heads_f32_source = resolved.source;
        },
        .attn_weighted_sum_heads_f16_kv => {
            self.attn_weighted_sum_heads_f16_kv_function = resolved.function;
            self.attn_weighted_sum_heads_f16_kv_source = resolved.source;
        },
        .rope_rows_ptrs => {
            self.rope_rows_ptrs_function = resolved.function;
            self.rope_rows_ptrs_source = resolved.source;
        },
        .attn_scores_heads_f16_kv_ptrs => {
            self.attn_scores_heads_f16_kv_ptrs_function = resolved.function;
            self.attn_scores_heads_f16_kv_ptrs_source = resolved.source;
        },
        .softmax_rows_dynamic_cols_ptrs => {
            self.softmax_rows_dynamic_cols_ptrs_function = resolved.function;
            self.softmax_rows_dynamic_cols_ptrs_source = resolved.source;
        },
        .attn_weighted_sum_heads_f16_kv_ptrs => {
            self.attn_weighted_sum_heads_f16_kv_ptrs_function = resolved.function;
            self.attn_weighted_sum_heads_f16_kv_ptrs_source = resolved.source;
        },
        .kv_write_i8 => {
            self.kv_write_i8_function = resolved.function;
            self.kv_write_i8_source = resolved.source;
        },
        .kv_write_i8_rows => {
            self.kv_write_i8_rows_function = resolved.function;
            self.kv_write_i8_rows_source = resolved.source;
        },
        .kv_write_i8_rows_ptrs => {
            self.kv_write_i8_rows_ptrs_function = resolved.function;
            self.kv_write_i8_rows_ptrs_source = resolved.source;
        },
        .dequant_kv_i8_to_f16 => {
            self.dequant_kv_i8_to_f16_function = resolved.function;
            self.dequant_kv_i8_to_f16_source = resolved.source;
        },
        .rope_store_i8 => {
            self.rope_store_i8_function = resolved.function;
            self.rope_store_i8_source = resolved.source;
        },
        .attn_scores_heads_i8_kv => {
            self.attn_scores_heads_i8_kv_function = resolved.function;
            self.attn_scores_heads_i8_kv_source = resolved.source;
        },
        .attn_weighted_sum_heads_i8_kv => {
            self.attn_weighted_sum_heads_i8_kv_function = resolved.function;
            self.attn_weighted_sum_heads_i8_kv_source = resolved.source;
        },
        .attn_fused_heads_i8_kv => {
            self.attn_fused_heads_i8_kv_function = resolved.function;
            self.attn_fused_heads_i8_kv_source = resolved.source;
        },
        .attn_fused_decode_heads_i8_kv_ptrs => {
            self.attn_fused_decode_heads_i8_kv_ptrs_function = resolved.function;
            self.attn_fused_decode_heads_i8_kv_ptrs_source = resolved.source;
        },
        .attn_fused_prefill_heads_i8_kv => {
            self.attn_fused_prefill_heads_i8_kv_function = resolved.function;
            self.attn_fused_prefill_heads_i8_kv_source = resolved.source;
        },
        .attn_fused_prefill_heads_i8_kv_gqa => {
            self.attn_fused_prefill_heads_i8_kv_gqa_function = resolved.function;
            self.attn_fused_prefill_heads_i8_kv_gqa_source = resolved.source;
        },
        .attn_scores_heads_i8_kv_ptrs => {
            self.attn_scores_heads_i8_kv_ptrs_function = resolved.function;
            self.attn_scores_heads_i8_kv_ptrs_source = resolved.source;
        },
        .attn_weighted_sum_heads_i8_kv_ptrs => {
            self.attn_weighted_sum_heads_i8_kv_ptrs_function = resolved.function;
            self.attn_weighted_sum_heads_i8_kv_ptrs_source = resolved.source;
        },
        .kv_write_fp8 => {
            self.kv_write_fp8_function = resolved.function;
            self.kv_write_fp8_source = resolved.source;
        },
        .kv_write_fp8_rows => {
            self.kv_write_fp8_rows_function = resolved.function;
            self.kv_write_fp8_rows_source = resolved.source;
        },
        .kv_write_fp8_rows_ptrs => {
            self.kv_write_fp8_rows_ptrs_function = resolved.function;
            self.kv_write_fp8_rows_ptrs_source = resolved.source;
        },
        .dequant_kv_fp8_to_f16 => {
            self.dequant_kv_fp8_to_f16_function = resolved.function;
            self.dequant_kv_fp8_to_f16_source = resolved.source;
        },
        .rope_store_fp8 => {
            self.rope_store_fp8_function = resolved.function;
            self.rope_store_fp8_source = resolved.source;
        },
        .attn_scores_heads_fp8_kv => {
            self.attn_scores_heads_fp8_kv_function = resolved.function;
            self.attn_scores_heads_fp8_kv_source = resolved.source;
        },
        .attn_scores_heads_fp8_kv_ptrs => {
            self.attn_scores_heads_fp8_kv_ptrs_function = resolved.function;
            self.attn_scores_heads_fp8_kv_ptrs_source = resolved.source;
        },
        .attn_weighted_sum_heads_fp8_kv => {
            self.attn_weighted_sum_heads_fp8_kv_function = resolved.function;
            self.attn_weighted_sum_heads_fp8_kv_source = resolved.source;
        },
        .attn_weighted_sum_heads_fp8_kv_ptrs => {
            self.attn_weighted_sum_heads_fp8_kv_ptrs_function = resolved.function;
            self.attn_weighted_sum_heads_fp8_kv_ptrs_source = resolved.source;
        },
        .attn_fused_heads_fp8_kv => {
            self.attn_fused_heads_fp8_kv_function = resolved.function;
            self.attn_fused_heads_fp8_kv_source = resolved.source;
        },
        .attn_fused_decode_heads_fp8_kv_ptrs => {
            self.attn_fused_decode_heads_fp8_kv_ptrs_function = resolved.function;
            self.attn_fused_decode_heads_fp8_kv_ptrs_source = resolved.source;
        },
        .attn_fused_prefill_heads_fp8_kv => {
            self.attn_fused_prefill_heads_fp8_kv_function = resolved.function;
            self.attn_fused_prefill_heads_fp8_kv_source = resolved.source;
        },
        .attn_fused_prefill_heads_fp8_kv_gqa => {
            self.attn_fused_prefill_heads_fp8_kv_gqa_function = resolved.function;
            self.attn_fused_prefill_heads_fp8_kv_gqa_source = resolved.source;
        },
        .flash_decode_f16 => {
            self.flash_decode_f16_function = resolved.function;
            self.flash_decode_f16_source = resolved.source;
        },
        .flash_decode_i8 => {
            self.flash_decode_i8_function = resolved.function;
            self.flash_decode_i8_source = resolved.source;
        },
        .flash_decode_fp8 => {
            self.flash_decode_fp8_function = resolved.function;
            self.flash_decode_fp8_source = resolved.source;
        },
        .flash_decode_reduce => {
            self.flash_decode_reduce_function = resolved.function;
            self.flash_decode_reduce_source = resolved.source;
        },
        .flash_prefill_f16 => {
            self.flash_prefill_f16_function = resolved.function;
            self.flash_prefill_f16_source = resolved.source;
        },
        .flash_prefill_i8 => {
            self.flash_prefill_i8_function = resolved.function;
            self.flash_prefill_i8_source = resolved.source;
        },
        .flash_prefill_fp8 => {
            self.flash_prefill_fp8_function = resolved.function;
            self.flash_prefill_fp8_source = resolved.source;
        },
        .silu => {
            self.silu_function = resolved.function;
            self.silu_source = resolved.source;
        },
        .silu_mul => {
            self.silu_mul_function = resolved.function;
            self.silu_mul_source = resolved.source;
        },
        .gelu_mul => {
            self.gelu_mul_function = resolved.function;
            self.gelu_mul_source = resolved.source;
        },
        .shortconv_step => {
            self.shortconv_step_function = resolved.function;
            self.shortconv_step_source = resolved.source;
        },
        .gated_attention_compact_q => {
            self.gated_attention_compact_q_function = resolved.function;
            self.gated_attention_compact_q_source = resolved.source;
        },
        .gated_attention_output_gate => {
            self.gated_attention_output_gate_function = resolved.function;
            self.gated_attention_output_gate_source = resolved.source;
        },
        .gated_delta_conv => {
            self.gated_delta_conv_function = resolved.function;
            self.gated_delta_conv_source = resolved.source;
        },
        .gated_delta_conv_silu => {
            self.gated_delta_conv_silu_function = resolved.function;
            self.gated_delta_conv_silu_source = resolved.source;
        },
        .gated_delta_conv_silu_rows => {
            self.gated_delta_conv_silu_rows_function = resolved.function;
            self.gated_delta_conv_silu_rows_source = resolved.source;
        },
        .gated_delta_conv_silu_rows_ptrs => {
            self.gated_delta_conv_silu_rows_ptrs_function = resolved.function;
            self.gated_delta_conv_silu_rows_ptrs_source = resolved.source;
        },
        .gated_delta_advance_ring_heads => {
            self.gated_delta_advance_ring_heads_function = resolved.function;
            self.gated_delta_advance_ring_heads_source = resolved.source;
        },
        .gated_delta_qk_norm => {
            self.gated_delta_qk_norm_function = resolved.function;
            self.gated_delta_qk_norm_source = resolved.source;
        },
        .gated_delta_ssm => {
            self.gated_delta_ssm_function = resolved.function;
            self.gated_delta_ssm_source = resolved.source;
        },
        .gated_delta_ssm_rows => {
            self.gated_delta_ssm_rows_function = resolved.function;
            self.gated_delta_ssm_rows_source = resolved.source;
        },
        .gated_delta_ssm_rows_ptrs => {
            self.gated_delta_ssm_rows_ptrs_function = resolved.function;
            self.gated_delta_ssm_rows_ptrs_source = resolved.source;
        },
        .gated_delta_ssm_rows_i8 => {
            self.gated_delta_ssm_rows_i8_function = resolved.function;
            self.gated_delta_ssm_rows_i8_source = resolved.source;
        },
        .gated_delta_ssm_rows_ptrs_i8 => {
            self.gated_delta_ssm_rows_ptrs_i8_function = resolved.function;
            self.gated_delta_ssm_rows_ptrs_i8_source = resolved.source;
        },
        .gated_delta_rmsnorm_silu_mul => {
            self.gated_delta_rmsnorm_silu_mul_function = resolved.function;
            self.gated_delta_rmsnorm_silu_mul_source = resolved.source;
        },
        .gated_delta_rmsnorm_silu_mul_rows => {
            self.gated_delta_rmsnorm_silu_mul_rows_function = resolved.function;
            self.gated_delta_rmsnorm_silu_mul_rows_source = resolved.source;
        },
        .argmax => {
            self.argmax_function = resolved.function;
            self.argmax_source = resolved.source;
        },
        .matmul_f16 => {
            self.matmul_f16_function = resolved.function;
            self.matmul_f16_source = resolved.source;
        },
        .matmul_bf16 => {
            self.matmul_bf16_function = resolved.function;
            self.matmul_bf16_source = resolved.source;
        },
        .matvec_f16 => {
            self.matvec_f16_function = resolved.function;
            self.matvec_f16_source = resolved.source;
        },
        .matvec_bf16 => {
            self.matvec_bf16_function = resolved.function;
            self.matvec_bf16_source = resolved.source;
        },
        .matvec_gate_up_f16 => {
            self.matvec_gate_up_f16_function = resolved.function;
            self.matvec_gate_up_f16_source = resolved.source;
        },
        .matvec_gate_up_bf16 => {
            self.matvec_gate_up_bf16_function = resolved.function;
            self.matvec_gate_up_bf16_source = resolved.source;
        },
        .matvec_gate_up_silu_f16 => {
            self.matvec_gate_up_silu_f16_function = resolved.function;
            self.matvec_gate_up_silu_f16_source = resolved.source;
        },
        .matvec_gate_up_silu_bf16 => {
            self.matvec_gate_up_silu_bf16_function = resolved.function;
            self.matvec_gate_up_silu_bf16_source = resolved.source;
        },
        .matvec_qkv_f16 => {
            self.matvec_qkv_f16_function = resolved.function;
            self.matvec_qkv_f16_source = resolved.source;
        },
        .matvec_qkv_bf16 => {
            self.matvec_qkv_bf16_function = resolved.function;
            self.matvec_qkv_bf16_source = resolved.source;
        },
        .gaffine_u4_matvec => {
            self.gaffine_u4_matvec_function = resolved.function;
            self.gaffine_u4_matvec_source = resolved.source;
        },
        .gaffine_u4_matvec_tile8 => {
            self.gaffine_u4_matvec_tile8_function = resolved.function;
            self.gaffine_u4_matvec_tile8_source = resolved.source;
        },
        .gaffine_u8_matvec => {
            self.gaffine_u8_matvec_function = resolved.function;
            self.gaffine_u8_matvec_source = resolved.source;
        },
        .gaffine_u4_matvec_gate_up => {
            self.gaffine_u4_matvec_gate_up_function = resolved.function;
            self.gaffine_u4_matvec_gate_up_source = resolved.source;
        },
        .gaffine_u4_matvec_qkv => {
            self.gaffine_u4_matvec_qkv_function = resolved.function;
            self.gaffine_u4_matvec_qkv_source = resolved.source;
        },
        .gaffine_u4_matvec_qkv_tile8 => {
            self.gaffine_u4_matvec_qkv_tile8_function = resolved.function;
            self.gaffine_u4_matvec_qkv_tile8_source = resolved.source;
        },
        .gaffine_u8_matvec_qkv => {
            self.gaffine_u8_matvec_qkv_function = resolved.function;
            self.gaffine_u8_matvec_qkv_source = resolved.source;
        },
        .gaffine_u8_matvec_gate_up => {
            self.gaffine_u8_matvec_gate_up_function = resolved.function;
            self.gaffine_u8_matvec_gate_up_source = resolved.source;
        },
        .gaffine_u4_matvec_gate_up_silu => {
            self.gaffine_u4_matvec_gate_up_silu_function = resolved.function;
            self.gaffine_u4_matvec_gate_up_silu_source = resolved.source;
        },
        .gaffine_u4_matvec_gate_up_silu_tile8 => {
            self.gaffine_u4_matvec_gate_up_silu_tile8_function = resolved.function;
            self.gaffine_u4_matvec_gate_up_silu_tile8_source = resolved.source;
        },
        .gaffine_u8_matvec_gate_up_silu => {
            self.gaffine_u8_matvec_gate_up_silu_function = resolved.function;
            self.gaffine_u8_matvec_gate_up_silu_source = resolved.source;
        },
        .gaffine_u4_dequant_f16 => {
            self.gaffine_u4_dequant_f16_function = resolved.function;
            self.gaffine_u4_dequant_f16_source = resolved.source;
        },
        .gaffine_u8_dequant_f16 => {
            self.gaffine_u8_dequant_f16_function = resolved.function;
            self.gaffine_u8_dequant_f16_source = resolved.source;
        },
    }
}

pub fn tryLoadSideloadModule(self: anytype) !bool {
    const base_url_raw = std.process.getEnvVarOwned(self.allocator, compute.cuda.sideload.kernel_base_url_env) catch |err| switch (err) {
        error.EnvironmentVariableNotFound => return false,
        else => return err,
    };
    defer self.allocator.free(base_url_raw);
    const base_url = std.mem.trim(u8, base_url_raw, " \t\r\n");
    if (base_url.len == 0) return false;

    const capability = self.device.computeCapability() catch |err| switch (err) {
        error.CudaQueryUnavailable => return false,
        else => return err,
    };
    const arch = try compute.cuda.sideload.archTag(self.allocator, capability.major, capability.minor);
    defer self.allocator.free(arch);

    const cache_dir = try compute.cuda.sideload.resolveCacheDir(self.allocator);
    defer self.allocator.free(cache_dir);
    try compute.cuda.sideload.ensureCacheDir(cache_dir);

    const manifest_bytes = try compute.cuda.sideload.loadOrFetchManifest(
        self.allocator,
        cache_dir,
        arch,
        base_url,
    );
    defer self.allocator.free(manifest_bytes);
    var parsed_manifest = try compute.cuda.manifest.parse(self.allocator, manifest_bytes);
    defer parsed_manifest.deinit();
    try compute.cuda.manifest.ensureCompatible(
        parsed_manifest.manifest,
        arch,
        compute.cuda.manifest.kernel_abi_version,
    );

    const artifact_bytes = try compute.cuda.sideload.loadOrFetchArtifact(
        self.allocator,
        cache_dir,
        arch,
        base_url,
        parsed_manifest.manifest.sha256,
    );
    defer self.allocator.free(artifact_bytes);

    try self.kernel_registry.loadSideloadModule(
        manifest_bytes,
        artifact_bytes,
        arch,
        compute.cuda.manifest.kernel_abi_version,
    );
    log.info("inference", "CUDA sideload payload loaded", .{
        .arch = arch,
        .cache_dir = cache_dir,
    });
    return true;
}
