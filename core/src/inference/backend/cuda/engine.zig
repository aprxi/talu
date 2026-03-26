//! CUDA backend engine.
//!
//! Contains the CudaBackend struct — the primary CUDA inference backend.
//! Shared types, constants, and support structures live in engine_types.zig.

const std = @import("std");
const build_options = @import("build_options");
const models = @import("../../../models/root.zig");
const layer_ops = models.layer_ops;
const op_types = models.op_types;
const opcode_map = models.plan.opcode_map;
const plan_compiler = models.plan.compiler;
const rope_scaling = models.rope_scaling;
const runtime_contract = @import("../../runtime_contract/root.zig");
const backend_root = @import("../root.zig");
const contract = @import("../contract.zig");
const compute = @import("../../../compute/root.zig");
const tensor = @import("../../../tensor.zig");
const dtype = @import("../../../dtype.zig");
const log = @import("../../../log.zig");
const trace = @import("../../../xray/trace.zig");
const load_transforms = @import("../../../models/load/transforms.zig");
const vision_types = @import("../../vision_types.zig");
const common_mrope = @import("../../vision_mrope.zig");
const smoke_checks = @import("smoke_checks.zig");
const attention_policy = @import("attention_policy.zig");
const attention_mod = @import("attention.zig");
const decode_mod = @import("decode.zig");
const prefill_mod = @import("prefill.zig");
const cpu_backend = @import("../cpu/root.zig");
const cuda_stage = @import("stage.zig");
const cpu_stage = @import("../cpu/stage.zig");
const progress_mod = @import("../../../progress.zig");
const sampling_mod = @import("sampling.zig");
const vision_runtime_mod = @import("vision/root.zig");
const cpu_kernels = @import("../cpu/kernels/root.zig");
const cpu_conv1d = compute.cpu.conv1d_depthwise;
const cpu_gated_delta = compute.cpu.gated_delta;
const GateUpLayout = models.runtime_blocks.GateUpLayout;

const LoadedModel = models.LoadedModel;
const Tensor = tensor.Tensor;

// --- Re-exported types from engine_types.zig ---
const engine_types = @import("engine_types.zig");
const prototype_eps = engine_types.prototype_eps;
const initial_kv_cache_tokens = engine_types.initial_kv_cache_tokens;
const kv_cache_dtype_fp16 = engine_types.kv_cache_dtype_fp16;
const enable_fused_attention_f16_kv = engine_types.enable_fused_attention_f16_kv;
const max_fused_attention_f16_kv_seq_len = engine_types.max_fused_attention_f16_kv_seq_len;
const default_prefill_chunk_rows_cap = engine_types.default_prefill_chunk_rows_cap;
const enable_device_embedding_lookup = engine_types.enable_device_embedding_lookup;
const max_supported_fused_f16_kv_head_dim = engine_types.max_supported_fused_f16_kv_head_dim;
const enable_dispatch_observability = engine_types.enable_dispatch_observability;
const attention_policy_config = engine_types.attention_policy_config;
const run_startup_selftests = engine_types.run_startup_selftests;
const gaffine_scales_dtype_f16 = engine_types.gaffine_scales_dtype_f16;
const gaffine_scales_dtype_bf16 = engine_types.gaffine_scales_dtype_bf16;
const DenseU16Dtype = engine_types.DenseU16Dtype;
const EmbeddingLookupKind = engine_types.EmbeddingLookupKind;
const KernelSlot = engine_types.KernelSlot;
const RequiredKernel = engine_types.RequiredKernel;
const required_kernels = engine_types.required_kernels;
const ProjectionPath = engine_types.ProjectionPath;
const AttentionPath = engine_types.AttentionPath;
const KvCacheStorageMode = engine_types.KvCacheStorageMode;
const AttentionKernelSet = engine_types.AttentionKernelSet;
const DeviceTensor = engine_types.DeviceTensor;
const missing_device_tensor = engine_types.missing_device_tensor;
const missing_host_tensor = engine_types.missing_host_tensor;
const EmbeddingLookup = engine_types.EmbeddingLookup;
const GaffineU4LinearWeight = engine_types.GaffineU4LinearWeight;
const GaffineU8LinearWeight = engine_types.GaffineU8LinearWeight;
const U16LinearWeight = engine_types.U16LinearWeight;
const LinearWeight = engine_types.LinearWeight;
const RuntimeBuffers = engine_types.RuntimeBuffers;
const LayerAttentionRuntime = engine_types.LayerAttentionRuntime;
const LayerAttentionExecConfig = engine_types.LayerAttentionExecConfig;
const AttentionWeightRefs = engine_types.AttentionWeightRefs;
const QkvI8ConcatRef = engine_types.QkvI8ConcatRef;
const ShortConvBlockRuntime = engine_types.ShortConvBlockRuntime;
const ShortConvExecConfig = engine_types.ShortConvExecConfig;
const ShortConvWeightRefs = engine_types.ShortConvWeightRefs;
const GatedDeltaBlockRuntime = engine_types.GatedDeltaBlockRuntime;
const GatedDeltaWeightRefs = engine_types.GatedDeltaWeightRefs;
const SwiGluWeightRefs = engine_types.SwiGluWeightRefs;
const BlockRuntimeLayer = engine_types.BlockRuntimeLayer;
const BlockRuntime = engine_types.BlockRuntime;
const RecurrentRuntimeState = engine_types.RecurrentRuntimeState;
const ShortConvRuntimeState = engine_types.ShortConvRuntimeState;
const MambaRuntimeState = engine_types.MambaRuntimeState;
const GatedDeltaRuntimeState = engine_types.GatedDeltaRuntimeState;
const KvRuntimeState = engine_types.KvRuntimeState;
const saturatingU64FromU128 = engine_types.saturatingU64FromU128;
const saturatingAddUsize = engine_types.saturatingAddUsize;
const resolveCudaFixedAllocMode = engine_types.resolveCudaFixedAllocMode;
const resolveCudaRequireFitCheck = engine_types.resolveCudaRequireFitCheck;
const resolveCudaStrictMemoryMode = engine_types.resolveCudaStrictMemoryMode;
const resolveCudaMemoryReserveBytes = engine_types.resolveCudaMemoryReserveBytes;
const resolveCudaExternalOverheadCapBytes = engine_types.resolveCudaExternalOverheadCapBytes;
const resolveCudaMaxSeqLen = engine_types.resolveCudaMaxSeqLen;
const resolveCudaInitialKvCacheTokens = engine_types.resolveCudaInitialKvCacheTokens;
const resolveCudaPrefillChunkRowsCap = engine_types.resolveCudaPrefillChunkRowsCap;
const resolveCudaKvStorageMode = engine_types.resolveCudaKvStorageMode;
const expectedAttentionQProjectionDim = engine_types.expectedAttentionQProjectionDim;
const bufferF32RowCount = engine_types.bufferF32RowCount;
const logicalF32RowSlice = engine_types.logicalF32RowSlice;
const buildCudaLayerProgramRegisterSlotMap = engine_types.buildCudaLayerProgramRegisterSlotMap;
const buildCudaLayerProgramSlotWidthHints = engine_types.buildCudaLayerProgramSlotWidthHints;
const validateCompiledLayerPlanForCuda = engine_types.validateCompiledLayerPlanForCuda;

// --- Compute ops from engine_ops.zig ---
const engine_ops = @import("engine_ops.zig");

// --- Mixer functions from engine_mixers.zig ---
const engine_mixers = @import("engine_mixers.zig");

// --- Forward pass from engine_forward.zig ---
const engine_forward = @import("engine_forward.zig");

// --- Layer program from engine_layer_program.zig ---
const engine_layer_program = @import("engine_layer_program.zig");

pub const CudaBackend = struct {
    pub const capabilities: contract.Capabilities = .{
        .vision_prefill = true,
        .decode_batch = true,
        .decode_streaming = true,
        .embedding = false,
        .warmup = false,
    };

    pub const PrefillVisionInput = vision_types.PrefillVisionInput;

    allocator: std.mem.Allocator,
    loaded: *LoadedModel,
    vision_runtime: ?vision_runtime_mod.VisionRuntime = null,
    device: compute.cuda.Device,
    compute_stream: ?compute.cuda.StreamHandle = null,
    decode_graph_exec: ?compute.cuda.GraphExecHandle = null,
    batched_decode_graph_exec: ?compute.cuda.GraphExecHandle = null,
    batched_decode_graph_seq_tier: u32 = 0,
    kernel_registry: compute.cuda.Registry,
    vector_add_function: ?compute.cuda.Function = null,
    vector_add_source: ?compute.cuda.registry.KernelSource = null,
    vector_add_scaled_function: ?compute.cuda.Function = null,
    vector_add_scaled_source: ?compute.cuda.registry.KernelSource = null,
    mul_function: ?compute.cuda.Function = null,
    mul_source: ?compute.cuda.registry.KernelSource = null,
    copy_function: ?compute.cuda.Function = null,
    copy_source: ?compute.cuda.registry.KernelSource = null,
    copy_u16_function: ?compute.cuda.Function = null,
    copy_u16_source: ?compute.cuda.registry.KernelSource = null,
    cast_f32_to_f16_function: ?compute.cuda.Function = null,
    cast_f32_to_f16_source: ?compute.cuda.registry.KernelSource = null,
    cast_f32_to_bf16_function: ?compute.cuda.Function = null,
    cast_f32_to_bf16_source: ?compute.cuda.registry.KernelSource = null,
    embedding_lookup_f32_function: ?compute.cuda.Function = null,
    embedding_lookup_f32_source: ?compute.cuda.registry.KernelSource = null,
    embedding_lookup_u16_function: ?compute.cuda.Function = null,
    embedding_lookup_u16_source: ?compute.cuda.registry.KernelSource = null,
    embedding_lookup_u16_rows_function: ?compute.cuda.Function = null,
    embedding_lookup_u16_rows_source: ?compute.cuda.registry.KernelSource = null,
    embedding_lookup_gaffine_u4_function: ?compute.cuda.Function = null,
    embedding_lookup_gaffine_u4_source: ?compute.cuda.registry.KernelSource = null,
    kv_write_f16_function: ?compute.cuda.Function = null,
    kv_write_f16_source: ?compute.cuda.registry.KernelSource = null,
    kv_write_f16_rows_function: ?compute.cuda.Function = null,
    kv_write_f16_rows_source: ?compute.cuda.registry.KernelSource = null,
    kv_write_f16_rows_ptrs_function: ?compute.cuda.Function = null,
    kv_write_f16_rows_ptrs_source: ?compute.cuda.registry.KernelSource = null,
    rmsnorm_function: ?compute.cuda.Function = null,
    rmsnorm_source: ?compute.cuda.registry.KernelSource = null,
    rope_function: ?compute.cuda.Function = null,
    rope_source: ?compute.cuda.registry.KernelSource = null,
    rope_store_f16_function: ?compute.cuda.Function = null,
    rope_store_f16_source: ?compute.cuda.registry.KernelSource = null,
    attn_scores_heads_f32_function: ?compute.cuda.Function = null,
    attn_scores_heads_f32_source: ?compute.cuda.registry.KernelSource = null,
    attn_scores_heads_f16_kv_function: ?compute.cuda.Function = null,
    attn_scores_heads_f16_kv_source: ?compute.cuda.registry.KernelSource = null,
    attn_fused_heads_f16_kv_function: ?compute.cuda.Function = null,
    attn_fused_heads_f16_kv_source: ?compute.cuda.registry.KernelSource = null,
    attn_fused_decode_heads_f16_kv_ptrs_function: ?compute.cuda.Function = null,
    attn_fused_decode_heads_f16_kv_ptrs_source: ?compute.cuda.registry.KernelSource = null,
    attn_fused_prefill_heads_f16_kv_function: ?compute.cuda.Function = null,
    attn_fused_prefill_heads_f16_kv_source: ?compute.cuda.registry.KernelSource = null,
    attn_fused_prefill_heads_f16_kv_gqa_function: ?compute.cuda.Function = null,
    attn_fused_prefill_heads_f16_kv_gqa_source: ?compute.cuda.registry.KernelSource = null,
    causal_attn_softmax_f32_function: ?compute.cuda.Function = null,
    causal_attn_softmax_f32_source: ?compute.cuda.registry.KernelSource = null,
    /// Workspace buffer for GEMM-based attention scores/probs (lazy allocation, f32).
    attn_scores_workspace_dev: ?compute.cuda.Buffer = null,
    /// Workspace buffer for f16 copies of Q and probs used by GEMM attention (lazy allocation).
    attn_u16_workspace_dev: ?compute.cuda.Buffer = null,
    /// Optional guard allocation used by strict-memory mode to reserve the
    /// allowed external-overhead envelope and fail closed on drift.
    strict_external_guard_dev: ?compute.cuda.Buffer = null,
    softmax_rows_function: ?compute.cuda.Function = null,
    softmax_rows_source: ?compute.cuda.registry.KernelSource = null,
    attn_weighted_sum_heads_f32_function: ?compute.cuda.Function = null,
    attn_weighted_sum_heads_f32_source: ?compute.cuda.registry.KernelSource = null,
    attn_weighted_sum_heads_f16_kv_function: ?compute.cuda.Function = null,
    attn_weighted_sum_heads_f16_kv_source: ?compute.cuda.registry.KernelSource = null,
    rope_rows_ptrs_function: ?compute.cuda.Function = null,
    rope_rows_ptrs_source: ?compute.cuda.registry.KernelSource = null,
    attn_scores_heads_f16_kv_ptrs_function: ?compute.cuda.Function = null,
    attn_scores_heads_f16_kv_ptrs_source: ?compute.cuda.registry.KernelSource = null,
    softmax_rows_dynamic_cols_ptrs_function: ?compute.cuda.Function = null,
    softmax_rows_dynamic_cols_ptrs_source: ?compute.cuda.registry.KernelSource = null,
    attn_weighted_sum_heads_f16_kv_ptrs_function: ?compute.cuda.Function = null,
    attn_weighted_sum_heads_f16_kv_ptrs_source: ?compute.cuda.registry.KernelSource = null,
    silu_function: ?compute.cuda.Function = null,
    silu_source: ?compute.cuda.registry.KernelSource = null,
    silu_mul_function: ?compute.cuda.Function = null,
    silu_mul_source: ?compute.cuda.registry.KernelSource = null,
    gelu_mul_function: ?compute.cuda.Function = null,
    gelu_mul_source: ?compute.cuda.registry.KernelSource = null,
    shortconv_step_function: ?compute.cuda.Function = null,
    shortconv_step_source: ?compute.cuda.registry.KernelSource = null,
    gated_attention_compact_q_function: ?compute.cuda.Function = null,
    gated_attention_compact_q_source: ?compute.cuda.registry.KernelSource = null,
    gated_attention_output_gate_function: ?compute.cuda.Function = null,
    gated_attention_output_gate_source: ?compute.cuda.registry.KernelSource = null,
    gated_delta_conv_function: ?compute.cuda.Function = null,
    gated_delta_conv_source: ?compute.cuda.registry.KernelSource = null,
    gated_delta_conv_silu_function: ?compute.cuda.Function = null,
    gated_delta_conv_silu_source: ?compute.cuda.registry.KernelSource = null,
    gated_delta_conv_silu_rows_function: ?compute.cuda.Function = null,
    gated_delta_conv_silu_rows_source: ?compute.cuda.registry.KernelSource = null,
    gated_delta_conv_silu_rows_ptrs_function: ?compute.cuda.Function = null,
    gated_delta_conv_silu_rows_ptrs_source: ?compute.cuda.registry.KernelSource = null,
    gated_delta_advance_ring_heads_function: ?compute.cuda.Function = null,
    gated_delta_advance_ring_heads_source: ?compute.cuda.registry.KernelSource = null,
    gated_delta_qk_norm_function: ?compute.cuda.Function = null,
    gated_delta_qk_norm_source: ?compute.cuda.registry.KernelSource = null,
    gated_delta_ssm_function: ?compute.cuda.Function = null,
    gated_delta_ssm_source: ?compute.cuda.registry.KernelSource = null,
    gated_delta_ssm_rows_function: ?compute.cuda.Function = null,
    gated_delta_ssm_rows_source: ?compute.cuda.registry.KernelSource = null,
    gated_delta_ssm_rows_ptrs_function: ?compute.cuda.Function = null,
    gated_delta_ssm_rows_ptrs_source: ?compute.cuda.registry.KernelSource = null,
    gated_delta_rmsnorm_silu_mul_function: ?compute.cuda.Function = null,
    gated_delta_rmsnorm_silu_mul_source: ?compute.cuda.registry.KernelSource = null,
    gated_delta_rmsnorm_silu_mul_rows_function: ?compute.cuda.Function = null,
    gated_delta_rmsnorm_silu_mul_rows_source: ?compute.cuda.registry.KernelSource = null,
    argmax_function: ?compute.cuda.Function = null,
    argmax_source: ?compute.cuda.registry.KernelSource = null,
    topk_rows_function: ?compute.cuda.Function = null,
    topk_rows_source: ?compute.cuda.registry.KernelSource = null,
    decode_u32_increment_function: ?compute.cuda.Function = null,
    decode_u32_increment_source: ?compute.cuda.registry.KernelSource = null,
    matmul_f16_function: ?compute.cuda.Function = null,
    matmul_f16_source: ?compute.cuda.registry.KernelSource = null,
    matmul_bf16_function: ?compute.cuda.Function = null,
    matmul_bf16_source: ?compute.cuda.registry.KernelSource = null,
    matvec_f16_function: ?compute.cuda.Function = null,
    matvec_f16_source: ?compute.cuda.registry.KernelSource = null,
    matvec_bf16_function: ?compute.cuda.Function = null,
    matvec_bf16_source: ?compute.cuda.registry.KernelSource = null,
    matvec_gate_up_f16_function: ?compute.cuda.Function = null,
    matvec_gate_up_f16_source: ?compute.cuda.registry.KernelSource = null,
    matvec_gate_up_bf16_function: ?compute.cuda.Function = null,
    matvec_gate_up_bf16_source: ?compute.cuda.registry.KernelSource = null,
    matvec_gate_up_silu_f16_function: ?compute.cuda.Function = null,
    matvec_gate_up_silu_f16_source: ?compute.cuda.registry.KernelSource = null,
    matvec_gate_up_silu_bf16_function: ?compute.cuda.Function = null,
    matvec_gate_up_silu_bf16_source: ?compute.cuda.registry.KernelSource = null,
    matvec_qkv_f16_function: ?compute.cuda.Function = null,
    matvec_qkv_f16_source: ?compute.cuda.registry.KernelSource = null,
    matvec_qkv_bf16_function: ?compute.cuda.Function = null,
    matvec_qkv_bf16_source: ?compute.cuda.registry.KernelSource = null,
    gaffine_u4_matvec_function: ?compute.cuda.Function = null,
    gaffine_u4_matvec_source: ?compute.cuda.registry.KernelSource = null,
    gaffine_u8_matvec_function: ?compute.cuda.Function = null,
    gaffine_u8_matvec_source: ?compute.cuda.registry.KernelSource = null,
    gaffine_u4_matvec_gate_up_function: ?compute.cuda.Function = null,
    gaffine_u4_matvec_gate_up_source: ?compute.cuda.registry.KernelSource = null,
    gaffine_u4_matvec_qkv_function: ?compute.cuda.Function = null,
    gaffine_u4_matvec_qkv_source: ?compute.cuda.registry.KernelSource = null,
    gaffine_u8_matvec_qkv_function: ?compute.cuda.Function = null,
    gaffine_u8_matvec_qkv_source: ?compute.cuda.registry.KernelSource = null,
    gaffine_u8_matvec_gate_up_function: ?compute.cuda.Function = null,
    gaffine_u8_matvec_gate_up_source: ?compute.cuda.registry.KernelSource = null,
    gaffine_u4_matvec_gate_up_silu_function: ?compute.cuda.Function = null,
    gaffine_u4_matvec_gate_up_silu_source: ?compute.cuda.registry.KernelSource = null,
    gaffine_u8_matvec_gate_up_silu_function: ?compute.cuda.Function = null,
    gaffine_u8_matvec_gate_up_silu_source: ?compute.cuda.registry.KernelSource = null,
    gaffine_u4_dequant_f16_function: ?compute.cuda.Function = null,
    gaffine_u4_dequant_f16_source: ?compute.cuda.registry.KernelSource = null,
    gaffine_u8_dequant_f16_function: ?compute.cuda.Function = null,
    gaffine_u8_dequant_f16_source: ?compute.cuda.registry.KernelSource = null,
    quantize_f32_to_i8_function: ?compute.cuda.Function = null,
    dequant_i32_gaffine_function: ?compute.cuda.Function = null,
    i8_rowsum_function: ?compute.cuda.Function = null,
    u8_xor_to_i8_function: ?compute.cuda.Function = null,
    i8_matvec_function: ?compute.cuda.Function = null,
    i8_matvec_qkv_function: ?compute.cuda.Function = null,
    i8_matvec_gate_up_silu_function: ?compute.cuda.Function = null,
    gaffine_u8_to_i8_function: ?compute.cuda.Function = null,
    gaffine_u4_to_i8_function: ?compute.cuda.Function = null,
    quantize_f16_to_i8_function: ?compute.cuda.Function = null,
    quantize_f32_to_i8_simple_function: ?compute.cuda.Function = null,
    dequant_i32_scales_function: ?compute.cuda.Function = null,
    dequant_i32_scales_split3_function: ?compute.cuda.Function = null,
    i8_blas_supported: bool = true,
    // Transient: set before QKV projection to provide concat cache for fused I8 prefill.
    active_qkv_concat: ?QkvI8ConcatRef = null,
    // Transient: set before linearForwardRows to fuse residual add into GEMV output.
    // Consumed (cleared) inside linearForwardRows for rows==1.
    pending_residual_add_buf: ?compute.cuda.Buffer = null,
    // Set when fused GEMV+residual succeeded; checked/cleared by residual add adapter.
    skip_next_residual_add: bool = false,
    gaffine_sequence_rows_supported: bool = false,
    gaffine_sequence_fused_qkv_supported: bool = false,
    gaffine_sequence_fused_gate_up_supported: bool = false,
    u16_blas_f16_supported: bool = true,
    u16_blas_bf16_supported: bool = true,
    kernel_arg_pack: compute.cuda.ArgPack,
    blas: compute.cuda.Blas,
    runtime_buffers: RuntimeBuffers,
    block_runtime: BlockRuntime,
    d_model: usize,
    vocab_size: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    rope_dim: usize,
    attention_scale: f32,
    norm_eps: f32,
    cpu_rope_global: ?*cpu_kernels.RoPE = null,
    cpu_rope_local: ?*cpu_kernels.RoPE = null,
    /// Pipeline topology mode. `.single` for one device, `.pipeline2` for two.
    topology_mode: backend_root.CudaTopologyMode = .single,
    /// Layer index at which stage 0 ends and stage 1 begins (pipeline2 only).
    split_layer: usize = 0,
    /// Layer index at which stage 1 ends and stage 2 begins (cpu+gpu+gpu only).
    split_layer_stage2: usize = 0,
    /// CPU stage backend for cpu+gpu topology mode.
    pipeline_backend0_cpu: ?*cpu_backend.BackendType = null,
    /// Second-device state for pipeline2 mode. Null for single-device mode.
    pipeline_backend1: ?*CudaBackend = null,
    /// Activation transfer mechanism (pipeline2 only).
    pipeline_transfer_mode: PipelineTransferMode = .none,
    pipeline_host_staging: ?[]align(4096) u8 = null,
    pipeline_host_staging_stage12: ?[]align(4096) u8 = null,
    pipeline_boundary_dtype: backend_root.pipeline.BoundaryDType = .f32,
    pipeline_boundary_layout: backend_root.pipeline.BoundaryLayout = .row_major,
    pipeline_stage0_boundary_conversion: bool = false,
    pipeline_stage1_boundary_conversion: bool = false,
    pipeline_boundary_dtype_stage12: backend_root.pipeline.BoundaryDType = .f32,
    pipeline_boundary_layout_stage12: backend_root.pipeline.BoundaryLayout = .row_major,
    pipeline_stage1_boundary_conversion_stage12: bool = false,
    pipeline_stage2_boundary_conversion_stage12: bool = false,

    kv_storage_mode: KvCacheStorageMode = .device,
    kv_init_tokens: usize = initial_kv_cache_tokens,
    prefill_chunk_rows_cap: usize = default_prefill_chunk_rows_cap,
    max_batch_size: usize = 1,
    fixed_alloc_mode: bool = false,
    require_fit_check: bool = false,
    strict_memory_mode: bool = false,
    memory_reserve_bytes: usize = 0,
    external_overhead_cap_bytes: ?usize = null,
    model_max_seq_len: usize = 0,
    dequant_cache_bytes: usize = 0,
    strict_guard_bytes: usize = 0,
    measured_external_overhead_bytes: usize = 0,
    slot_in_use: []bool,
    slot_positions: []usize,
    slot_rope_position_deltas: []isize,
    slot_logits: []f32,
    /// Cached slot set for batched decode pointer tables.
    decode_ptr_tables_cached_rows: usize = 0,
    decode_ptr_tables_cached_slots: []usize = &.{},
    /// Marks cached pointer tables stale after slot/KV mutations.
    decode_ptr_tables_dirty: bool = true,
    /// Which slot's KV/state buffers are currently loaded into block_runtime.
    /// Call activateKvSlot() before accessing per-slot KV or gated delta state.
    active_kv_slot: usize = 0,
    slot_kv_states: []SlotKvStates,
    state_descriptors_storage: [runtime_contract.max_state_descriptors]runtime_contract.StateDescriptor = undefined,
    state_descriptor_count: u8 = 0,
    slot_state_bindings: []SlotStateBinding = &.{},
    runtime_dispatch_counters: runtime_contract.DispatchCounters = .{},
    layer_program_dispatch_total: [256]u64 = [_]u64{0} ** 256,
    prefill_dispatch_window_start: [256]u64 = [_]u64{0} ** 256,
    layer_program_slot_buffers: []compute.cuda.Buffer = &.{},
    layer_program_slot_ptrs: []*compute.cuda.Buffer = &.{},
    layer_program_slot_widths: []usize = &.{},
    layer_program_row_capacity: usize = 1,
    argmax_index_dev: compute.cuda.Buffer,
    gated_delta_stage_input_host: []f32,
    gated_delta_stage_mid_host: []f32,
    gated_delta_stage_output_host: []f32,
    trace_checkpoint_host: []f32,
    parity_prefill_seq_len: usize,
    parity_prefill_token_index: usize,
    parity_prefill_layer_attn_norm_host: []f32,
    parity_prefill_layer_ffn_norm_host: []f32,
    parity_prefill_block_out_host: []f32,
    parity_checkpoint_warned: [256]bool,

    const PipelineTransferMode = enum { none, peer_to_peer, host_staged };

    const max_state_bindings_per_slot: usize = runtime_contract.max_state_descriptors;

    /// Per-slot KV cache, gated delta, and shortconv state buffer pointers.
    /// Buffer handles are swapped into block_runtime via activateKvSlot().
    const SlotKvStates = struct {
        const KvEntry = struct { k: compute.cuda.Buffer, v: compute.cuda.Buffer, capacity: usize };
        const GdEntry = struct { conv: compute.cuda.Buffer, ssm: compute.cuda.Buffer, conv_ring_head: u32 };
        const ScEntry = struct { conv: compute.cuda.Buffer };

        kv: []KvEntry,
        gd: []GdEntry,
        sc: []ScEntry,
    };

    pub const SlotStateBinding = struct {
        const local_state_block_bytes: usize = @intCast(runtime_contract.builtin_state_block_bytes);

        handles: [max_state_bindings_per_slot]runtime_contract.StateBlockHandle = undefined,
        local_blocks: [max_state_bindings_per_slot][local_state_block_bytes]u8 align(64) = undefined,
        count: u8 = 0,
        bound: bool = false,

        fn reset(self: *SlotStateBinding) void {
            self.count = 0;
            self.bound = false;
        }
    };

    const DeviceMemoryBudget = struct {
        weights_bytes: usize,
        runtime_bytes: usize,
        kv_state_bytes: usize,
        gated_delta_state_bytes: usize,
        shortconv_state_bytes: usize,
        layer_program_bytes: usize,
        workspace_bytes: usize,
        dequant_cache_bytes: usize,
        strict_guard_bytes: usize,
        misc_bytes: usize,

        fn slotStateBytes(self: *const DeviceMemoryBudget) usize {
            var total: usize = 0;
            total = saturatingAddUsize(total, self.kv_state_bytes);
            total = saturatingAddUsize(total, self.gated_delta_state_bytes);
            total = saturatingAddUsize(total, self.shortconv_state_bytes);
            return total;
        }

        fn totalBytes(self: *const DeviceMemoryBudget) usize {
            var total: usize = 0;
            total = saturatingAddUsize(total, self.weights_bytes);
            total = saturatingAddUsize(total, self.runtime_bytes);
            total = saturatingAddUsize(total, self.slotStateBytes());
            total = saturatingAddUsize(total, self.layer_program_bytes);
            total = saturatingAddUsize(total, self.workspace_bytes);
            total = saturatingAddUsize(total, self.dequant_cache_bytes);
            total = saturatingAddUsize(total, self.strict_guard_bytes);
            total = saturatingAddUsize(total, self.misc_bytes);
            return total;
        }
    };

    pub const InitOptions = struct {
        device_ordinal: usize = 0,
        topology_mode: backend_root.CudaTopologyMode = .single,
        stage_device_ordinals: [2]usize = .{ 0, 1 },
        split_layer: ?usize = null,
        split_layer_stage2: ?usize = null,
        /// Internal: restricts layer-dependent allocations to [start, end).
        /// Used by topology init to create stage backends without full-model allocation.
        init_layer_range: ?struct { start: usize, end: usize } = null,
        /// Progress context for "Devices" progress bar.
        progress: progress_mod.Context = progress_mod.Context.NONE,
    };

    pub fn init(
        allocator: std.mem.Allocator,
        loaded: *LoadedModel,
        max_batch_size: usize,
        init_options: InitOptions,
    ) !CudaBackend {
        var device = try compute.cuda.Device.initAt(init_options.device_ordinal);
        errdefer device.deinit();
        if (max_batch_size == 0) return error.InvalidArgument;
        const model_max_seq_len: usize = @intCast(loaded.config.max_seq_len);
        const resolved_max_seq_len = resolveCudaMaxSeqLen(model_max_seq_len);
        const resolved_kv_init_tokens = resolveCudaInitialKvCacheTokens(resolved_max_seq_len);
        const resolved_prefill_chunk_rows_cap = resolveCudaPrefillChunkRowsCap(resolved_max_seq_len);
        const resolved_require_fit_check = resolveCudaRequireFitCheck();
        const resolved_strict_memory_mode = resolveCudaStrictMemoryMode();
        const resolved_memory_reserve_bytes = resolveCudaMemoryReserveBytes();
        const resolved_external_overhead_cap_bytes = resolveCudaExternalOverheadCapBytes();

        log.info("inference", "CUDA device ready", .{
            .name = device.name(),
            .ordinal = device.ordinal(),
        });
        var backend = CudaBackend{
            .allocator = allocator,
            .loaded = loaded,
            .vision_runtime = null,
            .device = device,
            .compute_stream = null,
            .kernel_registry = undefined,
            .kernel_arg_pack = compute.cuda.ArgPack.init(allocator),
            .blas = undefined,
            .runtime_buffers = undefined,
            .block_runtime = undefined,
            .d_model = @intCast(loaded.config.d_model),
            .vocab_size = @intCast(loaded.config.vocab_size),
            .n_heads = @intCast(loaded.config.n_heads),
            .n_kv_heads = @intCast(loaded.config.n_kv_groups),
            .head_dim = @intCast(loaded.config.head_dim),
            .max_seq_len = resolved_max_seq_len,
            .rope_dim = 0,
            .attention_scale = 0.0,
            .kv_storage_mode = resolveCudaKvStorageMode(),
            .kv_init_tokens = resolved_kv_init_tokens,
            .prefill_chunk_rows_cap = resolved_prefill_chunk_rows_cap,
            .max_batch_size = max_batch_size,
            .fixed_alloc_mode = resolveCudaFixedAllocMode(),
            .require_fit_check = resolved_require_fit_check,
            .strict_memory_mode = resolved_strict_memory_mode,
            .memory_reserve_bytes = resolved_memory_reserve_bytes,
            .external_overhead_cap_bytes = resolved_external_overhead_cap_bytes,
            .model_max_seq_len = model_max_seq_len,
            .dequant_cache_bytes = 0,
            .strict_guard_bytes = 0,
            .measured_external_overhead_bytes = 0,
            .norm_eps = prototype_eps,
            .cpu_rope_global = null,
            .cpu_rope_local = null,
            .slot_in_use = &.{},
            .slot_positions = &.{},
            .slot_rope_position_deltas = &.{},
            .slot_logits = &.{},
            .decode_ptr_tables_cached_rows = 0,
            .decode_ptr_tables_cached_slots = &.{},
            .decode_ptr_tables_dirty = true,
            .slot_kv_states = &.{},
            .state_descriptors_storage = undefined,
            .state_descriptor_count = 0,
            .slot_state_bindings = &.{},
            .runtime_dispatch_counters = .{},
            .layer_program_dispatch_total = [_]u64{0} ** 256,
            .prefill_dispatch_window_start = [_]u64{0} ** 256,
            .layer_program_slot_buffers = &.{},
            .layer_program_slot_ptrs = &.{},
            .layer_program_slot_widths = &.{},
            .layer_program_row_capacity = 1,
            .argmax_index_dev = undefined,
            .gated_delta_stage_input_host = &.{},
            .gated_delta_stage_mid_host = &.{},
            .gated_delta_stage_output_host = &.{},
            .trace_checkpoint_host = &.{},
            .parity_prefill_seq_len = 0,
            .parity_prefill_token_index = 0,
            .parity_prefill_layer_attn_norm_host = &.{},
            .parity_prefill_layer_ffn_norm_host = &.{},
            .parity_prefill_block_out_host = &.{},
            .parity_checkpoint_warned = [_]bool{false} ** 256,
        };
        if (backend.n_heads == 0 or backend.n_kv_heads == 0 or backend.head_dim == 0 or backend.max_seq_len == 0) {
            return error.InvalidArgument;
        }
        if (backend.strict_memory_mode and !backend.fixed_alloc_mode) {
            // Strict memory mode is only meaningful with fixed allocations.
            backend.fixed_alloc_mode = true;
            log.info("inference", "Enabling TALU_CUDA_FIXED_ALLOC because TALU_CUDA_STRICT_MEMORY is set", .{});
        }
        if (backend.fixed_alloc_mode and !backend.require_fit_check) {
            // Fixed-allocation mode is intended to provide a deterministic
            // startup envelope. Enforce fit-check in this mode by default.
            backend.require_fit_check = true;
            log.info("inference", "Enabling TALU_CUDA_REQUIRE_FIT because TALU_CUDA_FIXED_ALLOC is set", .{});
        }
        if (backend.strict_memory_mode and backend.external_overhead_cap_bytes == null) {
            log.err("inference", "TALU_CUDA_STRICT_MEMORY requires TALU_CUDA_EXTERNAL_OVERHEAD_MIB", .{}, @src());
            return error.InvalidArgument;
        }
        if (backend.n_heads % backend.n_kv_heads != 0) return error.UnsupportedModel;
        backend.rope_dim = if (loaded.config.rope_dim > 0)
            @intCast(loaded.config.rope_dim)
        else
            backend.head_dim;
        if (backend.rope_dim == 0 or backend.rope_dim > backend.head_dim or (backend.rope_dim & 1) != 0) {
            return error.UnsupportedModel;
        }
        backend.attention_scale = if (loaded.config.attention_multiplier > 0.0)
            loaded.config.attention_multiplier
        else if (loaded.config.query_pre_attn_scalar > 0.0)
            1.0 / std.math.sqrt(loaded.config.query_pre_attn_scalar)
        else
            1.0 / std.math.sqrt(@as(f32, @floatFromInt(backend.head_dim)));
        try engine_layer_program.initCpuRuntimeRopeHandles(&backend);
        backend.norm_eps = if (loaded.config.norm_eps > 0.0) loaded.config.norm_eps else prototype_eps;
        if (backend.device.supportsStreams()) {
            const stream = try backend.device.createStream();
            backend.compute_stream = stream;
            backend.device.setLaunchStream(stream);
        }
        errdefer {
            backend.device.setLaunchStream(null);
            if (backend.compute_stream) |stream| backend.device.destroyStream(stream);
            backend.compute_stream = null;
        }
        backend.kernel_registry = compute.cuda.Registry.init(allocator, &backend.device);
        errdefer backend.kernel_registry.deinit();
        backend.slot_in_use = try allocator.alloc(bool, backend.max_batch_size);
        errdefer allocator.free(backend.slot_in_use);
        @memset(backend.slot_in_use, false);
        backend.slot_positions = try allocator.alloc(usize, backend.max_batch_size);
        errdefer allocator.free(backend.slot_positions);
        @memset(backend.slot_positions, 0);
        backend.slot_rope_position_deltas = try allocator.alloc(isize, backend.max_batch_size);
        errdefer allocator.free(backend.slot_rope_position_deltas);
        @memset(backend.slot_rope_position_deltas, 0);
        backend.decode_ptr_tables_cached_slots = try allocator.alloc(usize, backend.max_batch_size);
        errdefer allocator.free(backend.decode_ptr_tables_cached_slots);
        backend.decode_ptr_tables_cached_rows = 0;
        backend.decode_ptr_tables_dirty = true;
        backend.slot_logits = try allocator.alloc(f32, backend.max_batch_size * backend.vocab_size);
        errdefer allocator.free(backend.slot_logits);
        backend.slot_state_bindings = try allocator.alloc(SlotStateBinding, backend.max_batch_size);
        errdefer allocator.free(backend.slot_state_bindings);
        for (backend.slot_state_bindings) |*binding| binding.* = .{};
        backend.argmax_index_dev = try backend.device.allocBuffer(@sizeOf(u32));
        errdefer backend.argmax_index_dev.deinit(&backend.device);
        // Determine layer range for this backend instance. Stage backends
        // receive an explicit range; topology backends compute their own range
        // from the split points so BlockRuntime only allocates the needed layers.
        const total_layers = loaded.blocks.len;
        const layer_range = try computeInitLayerRange(init_options, total_layers);
        backend.block_runtime = try BlockRuntime.initRange(
            allocator,
            &backend.device,
            loaded,
            backend.max_seq_len,
            backend.kv_init_tokens,
            layer_range.start,
            layer_range.end,
            CudaBackend.layer_program_adapter_table,
        );
        errdefer backend.block_runtime.deinit(allocator, &backend.device);
        // A10 invariant: BlockRuntime must hold exactly the requested layer count.
        std.debug.assert(backend.block_runtime.blocks.len == layer_range.end - layer_range.start);
        engine_layer_program.assignCpuRuntimeRopeToAttentionFallbacks(&backend);
        try backend.initSlotKvStates();
        errdefer backend.deinitSlotKvStates();
        for (backend.block_runtime.blocks) |*layer| {
            if (layer.compiled_plan) |*compiled_plan| {
                try runtime_contract.appendUniquePlanStateDescriptors(
                    backend.state_descriptors_storage[0..],
                    &backend.state_descriptor_count,
                    &compiled_plan.plan,
                );
            }
        }
        backend.vision_runtime = try vision_runtime_mod.VisionRuntime.init(allocator, loaded);
        errdefer if (backend.vision_runtime) |*rt| rt.deinit();
        if (loaded.config.use_qk_norm and
            (backend.block_runtime.q_norm_blocks != backend.block_runtime.attention_block_count or
                backend.block_runtime.k_norm_blocks != backend.block_runtime.attention_block_count))
        {
            log.warn("inference", "CUDA backend requires explicit q/k norm weights when qk_norm is enabled", .{
                .q_norm_blocks = backend.block_runtime.q_norm_blocks,
                .k_norm_blocks = backend.block_runtime.k_norm_blocks,
                .layers = backend.block_runtime.attention_block_count,
            });
            return error.UnsupportedModel;
        }
        const max_dff = backend.block_runtime.maxDff();
        const max_attn = backend.block_runtime.maxAttn();
        const max_kv = backend.block_runtime.maxKv();
        const max_gdelta_proj = backend.block_runtime.maxGatedDeltaProj();
        const max_shortconv_dim = backend.block_runtime.maxShortConvDim();
        backend.blas = try compute.cuda.Blas.init(&backend.device);
        errdefer backend.blas.deinit(&backend.device);
        backend.runtime_buffers = try RuntimeBuffers.init(
            allocator,
            &backend.device,
            loaded,
            max_dff,
            max_attn,
            max_kv,
            max_gdelta_proj,
            max_shortconv_dim,
            backend.max_seq_len,
            backend.n_heads,
            backend.head_dim,
            backend.max_batch_size,
            @max(@as(usize, 1), backend.block_runtime.attention_block_count),
            @max(@as(usize, 1), backend.block_runtime.gated_delta_block_count),
        );
        errdefer backend.runtime_buffers.deinit(allocator, &backend.device);

        // Pipeline stages that don't start at layer 0 receive hidden states
        // from the preceding stage — they never perform embedding lookup.
        // Pipeline stages that don't end at the last layer output hidden states
        // to the next stage — they never compute logits via projection.
        // Free these unused GPU allocations to reclaim VRAM.
        if (layer_range.start > 0) {
            if (backend.runtime_buffers.embedding_lookup) |*lookup| {
                lookup.deinit(&backend.device);
                backend.runtime_buffers.embedding_lookup = null;
            }
        }
        if (layer_range.end < total_layers) {
            backend.runtime_buffers.projection_weight.deinit(&backend.device);
            backend.runtime_buffers.projection_weight = .{ .dense_f32 = missing_device_tensor };
        }

        try backend.initLayerProgramSlotBuffers();
        errdefer backend.deinitLayerProgramSlotBuffers();
        try engine_layer_program.initKernelFunctions(&backend);
        try backend.preallocateFixedAllocBuffers();
        try engine_layer_program.warmupDequantF16Cache(&backend);
        var memory_budget = backend.computeDeviceMemoryBudget();
        const slot_state_bytes = memory_budget.slotStateBytes();
        const slot_state_per_slot_bytes: usize = if (backend.max_batch_size > 0)
            slot_state_bytes / backend.max_batch_size
        else
            0;
        var known_device_bytes = memory_budget.totalBytes();
        const reserve_device_bytes = backend.memory_reserve_bytes;
        const require_fit_check = backend.require_fit_check;
        const total_device_bytes = backend.device.totalMemory() catch 0;
        const external_overhead_cap_bytes = backend.external_overhead_cap_bytes orelse 0;
        var used_device_bytes: usize = 0;
        var used_device_query_available = false;

        if (backend.strict_memory_mode) {
            const overhead_cap_bytes = backend.external_overhead_cap_bytes.?;
            const used_before_guard = backend.device.usedMemory() catch |err| {
                log.err("inference", "TALU_CUDA_STRICT_MEMORY failed to query device used memory", .{
                    .err = @errorName(err),
                }, @src());
                return err;
            };
            used_device_query_available = true;
            backend.measured_external_overhead_bytes = if (used_before_guard > known_device_bytes)
                used_before_guard - known_device_bytes
            else
                0;
            if (backend.measured_external_overhead_bytes > overhead_cap_bytes) {
                log.err("inference", "CUDA strict-memory external overhead exceeds configured cap", .{
                    .external_overhead_mib = bytesToMiB(backend.measured_external_overhead_bytes),
                    .external_overhead_cap_mib = bytesToMiB(overhead_cap_bytes),
                    .known_device_mib = bytesToMiB(known_device_bytes),
                    .used_device_mib = bytesToMiB(used_before_guard),
                }, @src());
                return error.OutOfMemory;
            }
            const guard_bytes = overhead_cap_bytes - backend.measured_external_overhead_bytes;
            if (guard_bytes > 0) {
                backend.strict_external_guard_dev = try backend.device.allocBuffer(guard_bytes);
                errdefer if (backend.strict_external_guard_dev) |*buf| buf.deinit(&backend.device);
                backend.strict_guard_bytes = backend.strict_external_guard_dev.?.size;
                memory_budget = backend.computeDeviceMemoryBudget();
                known_device_bytes = memory_budget.totalBytes();
            }
            used_device_bytes = backend.device.usedMemory() catch used_before_guard;
        } else {
            const maybe_used = backend.device.usedMemory() catch null;
            if (maybe_used) |used| {
                used_device_query_available = true;
                used_device_bytes = used;
                backend.measured_external_overhead_bytes = if (used > known_device_bytes)
                    used - known_device_bytes
                else
                    0;
            }
        }

        const required_with_reserve_bytes = saturatingAddUsize(known_device_bytes, reserve_device_bytes);
        const headroom_after_reserve_bytes: usize = if (total_device_bytes > required_with_reserve_bytes)
            total_device_bytes - required_with_reserve_bytes
        else
            0;
        const known_device_pct: f32 = if (total_device_bytes > 0)
            (@as(f32, @floatFromInt(known_device_bytes)) * 100.0) / @as(f32, @floatFromInt(total_device_bytes))
        else
            0.0;
        const required_with_reserve_pct: f32 = if (total_device_bytes > 0)
            (@as(f32, @floatFromInt(required_with_reserve_bytes)) * 100.0) / @as(f32, @floatFromInt(total_device_bytes))
        else
            0.0;
        log.info("inference", "CUDA device memory budget", .{
            .known_device_mib = bytesToMiB(known_device_bytes),
            .required_with_reserve_mib = bytesToMiB(required_with_reserve_bytes),
            .weights_mib = bytesToMiB(memory_budget.weights_bytes),
            .runtime_mib = bytesToMiB(memory_budget.runtime_bytes),
            .slot_state_mib = bytesToMiB(slot_state_bytes),
            .slot_state_per_slot_mib = bytesToMiB(slot_state_per_slot_bytes),
            .kv_state_mib = bytesToMiB(memory_budget.kv_state_bytes),
            .gated_delta_state_mib = bytesToMiB(memory_budget.gated_delta_state_bytes),
            .shortconv_state_mib = bytesToMiB(memory_budget.shortconv_state_bytes),
            .layer_program_mib = bytesToMiB(memory_budget.layer_program_bytes),
            .workspace_mib = bytesToMiB(memory_budget.workspace_bytes),
            .dequant_cache_mib = bytesToMiB(memory_budget.dequant_cache_bytes),
            .strict_guard_mib = bytesToMiB(memory_budget.strict_guard_bytes),
            .misc_mib = bytesToMiB(memory_budget.misc_bytes),
            .device_total_mib = bytesToMiB(total_device_bytes),
            .used_device_mib = bytesToMiB(used_device_bytes),
            .known_device_pct = known_device_pct,
            .required_with_reserve_pct = required_with_reserve_pct,
            .external_overhead_mib = bytesToMiB(backend.measured_external_overhead_bytes),
            .external_overhead_cap_mib = bytesToMiB(external_overhead_cap_bytes),
            .headroom_after_reserve_mib = bytesToMiB(headroom_after_reserve_bytes),
            .reserve_mib = bytesToMiB(reserve_device_bytes),
            .kv_storage = @tagName(backend.kv_storage_mode),
            .kv_init_tokens = backend.kv_init_tokens,
            .prefill_chunk_rows = backend.prefill_chunk_rows_cap,
            .max_batch = backend.max_batch_size,
            .max_seq = backend.max_seq_len,
            .model_max_seq = backend.model_max_seq_len,
            .fixed_alloc = @as(u8, @intFromBool(backend.fixed_alloc_mode)),
            .require_fit = @as(u8, @intFromBool(require_fit_check)),
            .strict_memory = @as(u8, @intFromBool(backend.strict_memory_mode)),
            .used_query = @as(u8, @intFromBool(used_device_query_available)),
        });
        if (total_device_bytes > 0 and required_with_reserve_bytes > total_device_bytes) {
            log.warn("inference", "CUDA known device allocations exceed fit envelope", .{
                .known_device_mib = bytesToMiB(known_device_bytes),
                .required_with_reserve_mib = bytesToMiB(required_with_reserve_bytes),
                .reserve_mib = bytesToMiB(reserve_device_bytes),
                .device_total_mib = bytesToMiB(total_device_bytes),
                .max_batch = backend.max_batch_size,
                .max_seq = backend.max_seq_len,
                .model_max_seq = backend.model_max_seq_len,
                .fixed_alloc = @as(u8, @intFromBool(backend.fixed_alloc_mode)),
                .require_fit = @as(u8, @intFromBool(require_fit_check)),
            });
            if (require_fit_check) return error.OutOfMemory;
        }

        if (loaded.original_weight_dtype == .grouped_affine_u4) {
            backend.gaffine_sequence_rows_supported = smoke_checks.probeGaffineU4SequenceRowsSupport(&backend) catch false;
            if (!backend.gaffine_sequence_rows_supported) {
                log.warn("inference", "CUDA gaffine batch-rows linear degraded mode active (multi-row parity probe failed)", .{
                    .reason = "gaffine_batch_rows_probe_failed",
                });
            } else {
                backend.gaffine_sequence_fused_qkv_supported = smoke_checks.probeGaffineU4SequenceFusedQkvSupport(&backend) catch false;
                if (!backend.gaffine_sequence_fused_qkv_supported) {
                    log.warn("inference", "CUDA gaffine batch-rows unfused QKV degraded mode active (multi-row fused parity probe failed)", .{
                        .reason = "gaffine_batch_rows_fused_qkv_probe_failed",
                    });
                }

                backend.gaffine_sequence_fused_gate_up_supported = smoke_checks.probeGaffineU4SequenceFusedGateUpSupport(&backend) catch false;
                if (!backend.gaffine_sequence_fused_gate_up_supported) {
                    log.warn("inference", "CUDA gaffine batch-rows unfused gate/up degraded mode active (multi-row fused parity probe failed)", .{
                        .reason = "gaffine_batch_rows_fused_gate_up_probe_failed",
                    });
                }
            }
        }

        if (run_startup_selftests) {
            try smoke_checks.runMatmulSmoke(&backend);
            try smoke_checks.runKernelSmoke(&backend);
        }

        switch (init_options.topology_mode) {
            .single => {
                init_options.progress.updateLine(1, @intCast(total_layers), null);
            },
            .pipeline2 => {
                backend.topology_mode = .pipeline2;
                const split = layer_range.split_layer;
                backend.split_layer = split;
                init_options.progress.updateLine(1, @intCast(split), null);

                // Stage 1: dedicated backend on device1 with layers [split, total_layers).
                const stage1_ptr = try allocator.create(CudaBackend);
                errdefer allocator.destroy(stage1_ptr);
                stage1_ptr.* = try CudaBackend.init(
                    allocator,
                    loaded,
                    max_batch_size,
                    .{
                        .device_ordinal = init_options.stage_device_ordinals[1],
                        .topology_mode = .single,
                        .stage_device_ordinals = init_options.stage_device_ordinals,
                        .init_layer_range = .{ .start = split, .end = total_layers },
                        .progress = init_options.progress,
                    },
                );
                errdefer {
                    stage1_ptr.deinit();
                    allocator.destroy(stage1_ptr);
                }
                backend.pipeline_backend1 = stage1_ptr;

                const boundary = try backend_root.pipeline.negotiateBoundaryContract(.{
                    .stage0_native_dtype = .f32,
                    .stage1_native_dtype = .f32,
                    .stage0_supported_boundary_dtypes = cuda_stage.CudaStage.supported_boundary_dtypes[0..],
                    .stage1_supported_boundary_dtypes = cuda_stage.CudaStage.supported_boundary_dtypes[0..],
                });
                backend.pipeline_boundary_dtype = boundary.boundary_dtype;
                backend.pipeline_boundary_layout = boundary.layout;
                backend.pipeline_stage0_boundary_conversion = boundary.stage0_requires_conversion;
                backend.pipeline_stage1_boundary_conversion = boundary.stage1_requires_conversion;

                log.info("inference", "CUDA pipeline2 stage 1 device ready", .{
                    .name = stage1_ptr.device.name(),
                    .ordinal = stage1_ptr.device.ordinal(),
                    .split_layer = split,
                    .total_layers = total_layers,
                });

                // Determine transfer mode.
                if (backend.device.canAccessPeer(&stage1_ptr.device)) {
                    backend.device.enablePeerAccess(&stage1_ptr.device) catch {};
                    stage1_ptr.device.enablePeerAccess(&backend.device) catch {};
                    backend.pipeline_transfer_mode = .peer_to_peer;
                    log.info("inference", "CUDA pipeline2 using peer-to-peer transfer", .{});
                } else {
                    const transfer_bytes = try backend.pipelineActivationByteCount();
                    backend.pipeline_host_staging = try allocator.alignedAlloc(u8, .fromByteUnits(4096), transfer_bytes);
                    backend.pipeline_transfer_mode = .host_staged;
                    log.info("inference", "CUDA pipeline2 using host-staged transfer", .{
                        .staging_mib = bytesToMiB(transfer_bytes),
                    });
                }
                errdefer if (backend.pipeline_host_staging) |buf| {
                    allocator.free(buf);
                    backend.pipeline_host_staging = null;
                };

                const stage0_budget = backend.computeDeviceMemoryBudget();
                const stage1_budget = stage1_ptr.computeDeviceMemoryBudget();
                log.info("inference", "CUDA pipeline2 topology", .{
                    .stage0_ordinal = backend.device.ordinal(),
                    .stage1_ordinal = stage1_ptr.device.ordinal(),
                    .stage0_layers_start = @as(usize, 0),
                    .stage0_layers_end = split,
                    .stage1_layers_start = split,
                    .stage1_layers_end = total_layers,
                    .transfer_mode = @tagName(backend.pipeline_transfer_mode),
                    .stage0_known_mib = bytesToMiB(stage0_budget.totalBytes()),
                    .stage1_known_mib = bytesToMiB(stage1_budget.totalBytes()),
                    .stage0_weights_mib = bytesToMiB(stage0_budget.weights_bytes),
                    .stage1_weights_mib = bytesToMiB(stage1_budget.weights_bytes),
                    .stage0_runtime_mib = bytesToMiB(stage0_budget.runtime_bytes),
                    .stage1_runtime_mib = bytesToMiB(stage1_budget.runtime_bytes),
                    .stage0_slot_state_mib = bytesToMiB(stage0_budget.slotStateBytes()),
                    .stage1_slot_state_mib = bytesToMiB(stage1_budget.slotStateBytes()),
                    .stage0_workspace_mib = bytesToMiB(stage0_budget.workspace_bytes),
                    .stage1_workspace_mib = bytesToMiB(stage1_budget.workspace_bytes),
                    .stage0_fixed_alloc = @as(u8, @intFromBool(backend.fixed_alloc_mode)),
                    .stage1_fixed_alloc = @as(u8, @intFromBool(stage1_ptr.fixed_alloc_mode)),
                    .stage0_strict_memory = @as(u8, @intFromBool(backend.strict_memory_mode)),
                    .stage1_strict_memory = @as(u8, @intFromBool(stage1_ptr.strict_memory_mode)),
                });
                init_options.progress.updateLine(1, @intCast(total_layers), null);
            },
            .cpu_gpu => {
                backend.topology_mode = .cpu_gpu;
                const split = layer_range.split_layer;
                backend.split_layer = split;

                // CPU stage 0 executes layers [0, split).
                const stage0_cpu_ptr = try allocator.create(cpu_backend.BackendType);
                errdefer allocator.destroy(stage0_cpu_ptr);
                stage0_cpu_ptr.* = try cpu_backend.BackendType.init(
                    allocator,
                    loaded,
                    max_batch_size,
                    progress_mod.Context.NONE,
                );
                errdefer {
                    stage0_cpu_ptr.deinit();
                    allocator.destroy(stage0_cpu_ptr);
                }
                backend.pipeline_backend0_cpu = stage0_cpu_ptr;
                init_options.progress.updateLine(1, @intCast(split), null);

                const boundary = try backend_root.pipeline.negotiateBoundaryContract(.{
                    .stage0_native_dtype = .f32,
                    .stage1_native_dtype = .f32,
                    .stage0_supported_boundary_dtypes = cpu_stage.CpuStage.supported_boundary_dtypes[0..],
                    .stage1_supported_boundary_dtypes = cuda_stage.CudaStage.supported_boundary_dtypes[0..],
                });
                backend.pipeline_boundary_dtype = boundary.boundary_dtype;
                backend.pipeline_boundary_layout = boundary.layout;
                backend.pipeline_stage0_boundary_conversion = boundary.stage0_requires_conversion;
                backend.pipeline_stage1_boundary_conversion = boundary.stage1_requires_conversion;
                if (boundary.stage0_requires_conversion or boundary.stage1_requires_conversion) {
                    return error.InvalidTopologyConfig;
                }

                // CUDA stage 1 (this backend) already initialized with layers [split, total_layers)
                // via init_layer_range computed above.

                const transfer_bytes = try backend.pipelineActivationByteCount();
                backend.pipeline_host_staging = try allocator.alignedAlloc(u8, .fromByteUnits(4096), transfer_bytes);
                errdefer {
                    allocator.free(backend.pipeline_host_staging.?);
                    backend.pipeline_host_staging = null;
                }
                backend.pipeline_transfer_mode = .host_staged;
                init_options.progress.updateLine(1, @intCast(total_layers), null);

                const stage1_budget = backend.computeDeviceMemoryBudget();
                log.info("inference", "CUDA cpu+gpu topology", .{
                    .cpu_stage_layers_start = @as(usize, 0),
                    .cpu_stage_layers_end = split,
                    .gpu_stage_layers_start = split,
                    .gpu_stage_layers_end = total_layers,
                    .gpu_ordinal = backend.device.ordinal(),
                    .transfer_mode = @tagName(backend.pipeline_transfer_mode),
                    .boundary_dtype = @tagName(backend.pipeline_boundary_dtype),
                    .boundary_layout = @tagName(backend.pipeline_boundary_layout),
                    .stage0_boundary_conversion = @as(u8, @intFromBool(backend.pipeline_stage0_boundary_conversion)),
                    .stage1_boundary_conversion = @as(u8, @intFromBool(backend.pipeline_stage1_boundary_conversion)),
                    .gpu_known_mib = bytesToMiB(stage1_budget.totalBytes()),
                    .gpu_weights_mib = bytesToMiB(stage1_budget.weights_bytes),
                    .gpu_runtime_mib = bytesToMiB(stage1_budget.runtime_bytes),
                    .gpu_slot_state_mib = bytesToMiB(stage1_budget.slotStateBytes()),
                    .gpu_workspace_mib = bytesToMiB(stage1_budget.workspace_bytes),
                });
            },
            .cpu_gpu_gpu => {
                backend.topology_mode = .cpu_gpu_gpu;
                const split = layer_range.split_layer;
                const split_stage2 = layer_range.split_layer_stage2;

                backend.split_layer = split;
                backend.split_layer_stage2 = split_stage2;

                // CPU stage 0 executes layers [0, split).
                const stage0_cpu_ptr = try allocator.create(cpu_backend.BackendType);
                errdefer allocator.destroy(stage0_cpu_ptr);
                stage0_cpu_ptr.* = try cpu_backend.BackendType.init(
                    allocator,
                    loaded,
                    max_batch_size,
                    progress_mod.Context.NONE,
                );
                errdefer {
                    stage0_cpu_ptr.deinit();
                    allocator.destroy(stage0_cpu_ptr);
                }
                backend.pipeline_backend0_cpu = stage0_cpu_ptr;
                init_options.progress.updateLine(1, @intCast(split), null);

                // GPU stage 1 executes layers [split, split_stage2).
                const stage1_ptr = try allocator.create(CudaBackend);
                errdefer allocator.destroy(stage1_ptr);
                stage1_ptr.* = try CudaBackend.init(
                    allocator,
                    loaded,
                    max_batch_size,
                    .{
                        .device_ordinal = init_options.stage_device_ordinals[0],
                        .topology_mode = .single,
                        .stage_device_ordinals = init_options.stage_device_ordinals,
                        .init_layer_range = .{ .start = split, .end = split_stage2 },
                        .progress = init_options.progress,
                    },
                );
                errdefer {
                    stage1_ptr.deinit();
                    allocator.destroy(stage1_ptr);
                }
                backend.pipeline_backend1 = stage1_ptr;
                init_options.progress.updateLine(1, @intCast(split_stage2), null);

                // GPU stage 2 (this backend) already initialized with layers [split_stage2, total_layers)
                // via init_layer_range computed above.

                // Boundary 0->1 (CPU->GPU1) contract.
                const boundary_01 = try backend_root.pipeline.negotiateBoundaryContract(.{
                    .stage0_native_dtype = .f32,
                    .stage1_native_dtype = .f32,
                    .stage0_supported_boundary_dtypes = cpu_stage.CpuStage.supported_boundary_dtypes[0..],
                    .stage1_supported_boundary_dtypes = cuda_stage.CudaStage.supported_boundary_dtypes[0..],
                });
                backend.pipeline_boundary_dtype = boundary_01.boundary_dtype;
                backend.pipeline_boundary_layout = boundary_01.layout;
                backend.pipeline_stage0_boundary_conversion = boundary_01.stage0_requires_conversion;
                backend.pipeline_stage1_boundary_conversion = boundary_01.stage1_requires_conversion;
                if (boundary_01.stage0_requires_conversion or boundary_01.stage1_requires_conversion) {
                    return error.InvalidTopologyConfig;
                }

                // Boundary 1->2 (GPU1->GPU2) contract.
                const boundary_12 = try backend_root.pipeline.negotiateBoundaryContract(.{
                    .stage0_native_dtype = .f32,
                    .stage1_native_dtype = .f32,
                    .stage0_supported_boundary_dtypes = cuda_stage.CudaStage.supported_boundary_dtypes[0..],
                    .stage1_supported_boundary_dtypes = cuda_stage.CudaStage.supported_boundary_dtypes[0..],
                });
                backend.pipeline_boundary_dtype_stage12 = boundary_12.boundary_dtype;
                backend.pipeline_boundary_layout_stage12 = boundary_12.layout;
                backend.pipeline_stage1_boundary_conversion_stage12 = boundary_12.stage0_requires_conversion;
                backend.pipeline_stage2_boundary_conversion_stage12 = boundary_12.stage1_requires_conversion;
                if (boundary_12.stage0_requires_conversion or boundary_12.stage1_requires_conversion) {
                    return error.InvalidTopologyConfig;
                }

                const transfer01_bytes = try backend.pipelineActivationByteCount();
                backend.pipeline_host_staging = try allocator.alignedAlloc(u8, .fromByteUnits(4096), transfer01_bytes);
                errdefer {
                    allocator.free(backend.pipeline_host_staging.?);
                    backend.pipeline_host_staging = null;
                }
                backend.pipeline_host_staging_stage12 = try allocator.alignedAlloc(u8, .fromByteUnits(4096), transfer01_bytes);
                errdefer {
                    allocator.free(backend.pipeline_host_staging_stage12.?);
                    backend.pipeline_host_staging_stage12 = null;
                }
                backend.pipeline_transfer_mode = .host_staged;

                const stage1_budget = stage1_ptr.computeDeviceMemoryBudget();
                const stage2_budget = backend.computeDeviceMemoryBudget();
                log.info("inference", "CUDA cpu+gpu+gpu topology", .{
                    .cpu_stage_layers_start = @as(usize, 0),
                    .cpu_stage_layers_end = split,
                    .gpu_stage1_layers_start = split,
                    .gpu_stage1_layers_end = split_stage2,
                    .gpu_stage2_layers_start = split_stage2,
                    .gpu_stage2_layers_end = total_layers,
                    .gpu_stage1_ordinal = stage1_ptr.device.ordinal(),
                    .gpu_stage2_ordinal = backend.device.ordinal(),
                    .boundary01_dtype = @tagName(backend.pipeline_boundary_dtype),
                    .boundary12_dtype = @tagName(backend.pipeline_boundary_dtype_stage12),
                    .boundary01_layout = @tagName(backend.pipeline_boundary_layout),
                    .boundary12_layout = @tagName(backend.pipeline_boundary_layout_stage12),
                    .gpu_stage1_known_mib = bytesToMiB(stage1_budget.totalBytes()),
                    .gpu_stage2_known_mib = bytesToMiB(stage2_budget.totalBytes()),
                    .gpu_stage1_weights_mib = bytesToMiB(stage1_budget.weights_bytes),
                    .gpu_stage2_weights_mib = bytesToMiB(stage2_budget.weights_bytes),
                    .gpu_stage1_runtime_mib = bytesToMiB(stage1_budget.runtime_bytes),
                    .gpu_stage2_runtime_mib = bytesToMiB(stage2_budget.runtime_bytes),
                    .gpu_stage1_slot_state_mib = bytesToMiB(stage1_budget.slotStateBytes()),
                    .gpu_stage2_slot_state_mib = bytesToMiB(stage2_budget.slotStateBytes()),
                });
                init_options.progress.updateLine(1, @intCast(total_layers), null);
            },
        }

        log.info("inference", "CUDA decode path ready", .{
            .d_model = backend.d_model,
            .n_heads = backend.n_heads,
            .n_kv = backend.n_kv_heads,
            .head_dim = backend.head_dim,
            .max_seq = backend.max_seq_len,
            .model_max_seq = backend.model_max_seq_len,
            .kv_storage = @tagName(backend.kv_storage_mode),
            .kv_dtype = if (kv_cache_dtype_fp16) "f16" else "f32",
            .kv_init_tokens = backend.kv_init_tokens,
            .kv_capacity_init = backend.initialKvCapacity(),
            .prefill_chunk_rows = backend.prefill_chunk_rows_cap,
            .device_blocks = backend.block_runtime.blocks.len,
            .attention_blocks = backend.block_runtime.attention_block_count,
            .shortconv_blocks = backend.block_runtime.shortconv_block_count,
            .gated_delta_blocks = backend.block_runtime.gated_delta_block_count,
            .embed_dtype = @tagName(loaded.token_embeddings.dtype),
            .embed_shape_0 = loaded.token_embeddings.shape[0],
            .embed_shape_1 = loaded.token_embeddings.shape[1],
        });
        log.info("inference", "CUDA memory budget", .{
            .projected_vocab = backend.runtime_buffers.projected_vocab,
            .max_dff = backend.runtime_buffers.max_dff,
            .max_attn = backend.runtime_buffers.max_attn,
            .max_kv = backend.runtime_buffers.max_kv,
            .max_seq = backend.max_seq_len,
            .model_max_seq = backend.model_max_seq_len,
            .kv_storage = @tagName(backend.kv_storage_mode),
            .kv_init_tokens = backend.kv_init_tokens,
            .prefill_chunk_rows = backend.prefill_chunk_rows_cap,
            .kv_capacity_init = backend.initialKvCapacity(),
            .n_heads = backend.n_heads,
            .n_kv = backend.n_kv_heads,
            .head_dim = backend.head_dim,
            .kv_dtype = if (kv_cache_dtype_fp16) "f16" else "f32",
            .linear_weight_mib = bytesToMiB(backend.block_runtime.linear_weight_bytes),
            .norm_weight_mib = bytesToMiB(backend.block_runtime.norm_weight_bytes),
            .kv_cache_mib = bytesToMiB(backend.block_runtime.kv_cache_bytes),
            .shortconv_state_mib = bytesToMiB(backend.block_runtime.shortconv_state_bytes),
            .gated_delta_state_mib = bytesToMiB(backend.block_runtime.gated_delta_state_bytes),
            .prototype_mib = bytesToMiB(backend.runtime_buffers.deviceByteSize()),
            .slot_logits_mib = bytesToMiB(std.math.mul(usize, backend.slot_logits.len, @sizeOf(f32)) catch 0),
            .reserve_mib = bytesToMiB(backend.memory_reserve_bytes),
            .fixed_alloc = @as(u8, @intFromBool(backend.fixed_alloc_mode)),
            .require_fit = @as(u8, @intFromBool(backend.require_fit_check)),
        });
        log.debug("inference", "CUDA model features", .{
            .use_qk_norm = @as(u8, @intFromBool(loaded.config.use_qk_norm)),
            .attention_bias = @as(u8, @intFromBool(loaded.config.attention_bias)),
            .norm_weight_offset = loaded.runtime.weight_offset,
            .qk_norm_weight_offset = loaded.runtime.qk_norm_weight_offset,
            .q_norm_blocks = backend.block_runtime.q_norm_blocks,
            .k_norm_blocks = backend.block_runtime.k_norm_blocks,
            .model_norm = @as(u8, @intFromBool(backend.runtime_buffers.using_model_norm)),
            .model_projection = @as(u8, @intFromBool(backend.runtime_buffers.using_model_projection)),
            .projection_lm_head = @as(u8, @intFromBool(backend.runtime_buffers.projection_from_lm_head)),
            .has_lm_head = @as(u8, @intFromBool(loaded.lm_head != null)),
            .model_embeddings = @as(u8, @intFromBool(backend.runtime_buffers.using_model_embeddings)),
            .embedding_lookup_device = @as(u8, @intFromBool(backend.runtime_buffers.embedding_lookup != null)),
            .stream_token_select = "gpu_argmax",
            .stream_enabled = @as(u8, @intFromBool(backend.compute_stream != null)),
            .attn_fused_heads_f16_kv_enabled = @as(u8, @intFromBool(enable_fused_attention_f16_kv)),
            .attn_fused_heads_f16_kv_max_seq = max_fused_attention_f16_kv_seq_len,
            .attn_score_buffers = @as(u8, @intFromBool(backend.runtime_buffers.attn_scores_dev != null and backend.runtime_buffers.attn_probs_dev != null)),
        }, @src());
        log.debug("inference", "CUDA kernel availability", .{
            .vector_add = @as(u8, @intFromBool(backend.vector_add_function != null)),
            .vector_add_scaled = @as(u8, @intFromBool(backend.vector_add_scaled_function != null)),
            .rmsnorm = @as(u8, @intFromBool(backend.rmsnorm_function != null)),
            .mul = @as(u8, @intFromBool(backend.mul_function != null)),
            .copy = @as(u8, @intFromBool(backend.copy_function != null)),
            .copy_u16 = @as(u8, @intFromBool(backend.copy_u16_function != null)),
            .cast_f32_to_f16 = @as(u8, @intFromBool(backend.cast_f32_to_f16_function != null)),
            .embed_f32 = @as(u8, @intFromBool(backend.embedding_lookup_f32_function != null)),
            .embed_u16 = @as(u8, @intFromBool(backend.embedding_lookup_u16_function != null)),
            .embed_gaffine_u4 = @as(u8, @intFromBool(backend.embedding_lookup_gaffine_u4_function != null)),
            .kv_write_f16 = @as(u8, @intFromBool(backend.kv_write_f16_function != null)),
            .rope = @as(u8, @intFromBool(backend.rope_function != null)),
            .rope_store_f16 = @as(u8, @intFromBool(backend.rope_store_f16_function != null)),
            .silu = @as(u8, @intFromBool(backend.silu_function != null)),
            .silu_mul = @as(u8, @intFromBool(backend.silu_mul_function != null)),
            .gelu_mul = @as(u8, @intFromBool(backend.gelu_mul_function != null)),
            .shortconv_step = @as(u8, @intFromBool(backend.shortconv_step_function != null)),
            .argmax = @as(u8, @intFromBool(backend.argmax_function != null)),
            .softmax_rows = @as(u8, @intFromBool(backend.softmax_rows_function != null)),
        }, @src());
        log.debug("inference", "CUDA matmul/attention kernels", .{
            .attn_scores_f32 = @as(u8, @intFromBool(backend.attn_scores_heads_f32_function != null)),
            .attn_scores_f16_kv = @as(u8, @intFromBool(backend.attn_scores_heads_f16_kv_function != null)),
            .attn_fused_f16_kv = @as(u8, @intFromBool(backend.attn_fused_heads_f16_kv_function != null)),
            .attn_wsum_f32 = @as(u8, @intFromBool(backend.attn_weighted_sum_heads_f32_function != null)),
            .attn_wsum_f16_kv = @as(u8, @intFromBool(backend.attn_weighted_sum_heads_f16_kv_function != null)),
            .matmul_f16 = @as(u8, @intFromBool(backend.matmul_f16_function != null)),
            .matmul_bf16 = @as(u8, @intFromBool(backend.matmul_bf16_function != null)),
            .matvec_f16 = @as(u8, @intFromBool(backend.matvec_f16_function != null)),
            .matvec_bf16 = @as(u8, @intFromBool(backend.matvec_bf16_function != null)),
            .matvec_gate_up_f16 = @as(u8, @intFromBool(backend.matvec_gate_up_f16_function != null)),
            .matvec_gate_up_bf16 = @as(u8, @intFromBool(backend.matvec_gate_up_bf16_function != null)),
            .matvec_qkv_f16 = @as(u8, @intFromBool(backend.matvec_qkv_f16_function != null)),
            .matvec_qkv_bf16 = @as(u8, @intFromBool(backend.matvec_qkv_bf16_function != null)),
            .gaffine_u4_matvec = @as(u8, @intFromBool(backend.gaffine_u4_matvec_function != null)),
            .gaffine_u8_matvec = @as(u8, @intFromBool(backend.gaffine_u8_matvec_function != null)),
            .gaffine_u4_gate_up = @as(u8, @intFromBool(backend.gaffine_u4_matvec_gate_up_function != null)),
            .gaffine_u4_qkv = @as(u8, @intFromBool(backend.gaffine_u4_matvec_qkv_function != null)),
            .gaffine_u8_qkv = @as(u8, @intFromBool(backend.gaffine_u8_matvec_qkv_function != null)),
            .gaffine_u8_gate_up = @as(u8, @intFromBool(backend.gaffine_u8_matvec_gate_up_function != null)),
            .gaffine_u4_gate_up_silu = @as(u8, @intFromBool(backend.gaffine_u4_matvec_gate_up_silu_function != null)),
            .gaffine_u8_gate_up_silu = @as(u8, @intFromBool(backend.gaffine_u8_matvec_gate_up_silu_function != null)),
            .gaffine_seq_rows = @as(u8, @intFromBool(backend.gaffine_sequence_rows_supported)),
            .gaffine_seq_qkv = @as(u8, @intFromBool(backend.gaffine_sequence_fused_qkv_supported)),
            .gaffine_seq_gate_up = @as(u8, @intFromBool(backend.gaffine_sequence_fused_gate_up_supported)),
        }, @src());
        return backend;
    }

    /// Resolved layer range and split points for a backend instance.
    pub const LayerRangeResult = struct {
        /// Layer range for this backend instance: [start, end).
        start: usize,
        end: usize,
        /// Resolved first split point. Zero for single topology or explicit range override.
        split_layer: usize = 0,
        /// Resolved second split point. Non-zero only for cpu_gpu_gpu.
        split_layer_stage2: usize = 0,
    };

    /// Compute the layer range and resolved split points from topology mode,
    /// split points, and optional explicit range override.
    /// Pure validation — no GPU or allocator dependencies.
    pub fn computeInitLayerRange(
        opts: InitOptions,
        total_layers: usize,
    ) error{InvalidTopologyConfig}!LayerRangeResult {
        if (opts.init_layer_range) |r| {
            // init_layer_range is internal: used by topology init to create stage
            // backends which are always .single. Reject staged modes to prevent
            // the topology switch from consuming zero-valued split metadata.
            if (opts.topology_mode != .single) return error.InvalidTopologyConfig;
            if (r.start >= r.end or r.end > total_layers) return error.InvalidTopologyConfig;
            return .{ .start = r.start, .end = r.end };
        }
        return switch (opts.topology_mode) {
            .single => .{ .start = 0, .end = total_layers },
            .pipeline2 => {
                if (total_layers < 2) return error.InvalidTopologyConfig;
                const split = opts.split_layer orelse total_layers / 2;
                if (split == 0 or split >= total_layers) return error.InvalidTopologyConfig;
                return .{ .start = 0, .end = split, .split_layer = split };
            },
            .cpu_gpu => {
                if (total_layers < 2) return error.InvalidTopologyConfig;
                const split = opts.split_layer orelse total_layers / 2;
                if (split == 0 or split >= total_layers) return error.InvalidTopologyConfig;
                return .{ .start = split, .end = total_layers, .split_layer = split };
            },
            .cpu_gpu_gpu => {
                if (total_layers < 3) return error.InvalidTopologyConfig;
                const split_default = @max(@as(usize, 1), total_layers / 3);
                const split = opts.split_layer orelse split_default;
                if (split == 0 or split >= total_layers) return error.InvalidTopologyConfig;
                const split_stage2_default = split + @max(@as(usize, 1), (total_layers - split) / 2);
                const split_stage2 = opts.split_layer_stage2 orelse split_stage2_default;
                if (split_stage2 <= split or split_stage2 >= total_layers) return error.InvalidTopologyConfig;
                return .{ .start = split_stage2, .end = total_layers, .split_layer = split, .split_layer_stage2 = split_stage2 };
            },
        };
    }

    pub fn deinit(self: *CudaBackend) void {
        if (self.vision_runtime) |*rt| rt.deinit();
        if (self.decode_graph_exec) |exec| {
            self.device.graphExecDestroy(exec);
            self.decode_graph_exec = null;
        }
        if (self.batched_decode_graph_exec) |exec| {
            self.device.graphExecDestroy(exec);
            self.batched_decode_graph_exec = null;
        }
        self.device.setLaunchStream(null);
        if (self.compute_stream) |stream| {
            _ = self.device.synchronizeStream(stream) catch {};
            self.device.destroyStream(stream);
            self.compute_stream = null;
        }
        if (self.strict_external_guard_dev) |*buf| {
            buf.deinit(&self.device);
            self.strict_external_guard_dev = null;
        }
        self.argmax_index_dev.deinit(&self.device);
        if (self.slot_state_bindings.len > 0) self.allocator.free(self.slot_state_bindings);
        if (self.trace_checkpoint_host.len > 0) self.allocator.free(self.trace_checkpoint_host);
        if (self.parity_prefill_layer_attn_norm_host.len > 0) self.allocator.free(self.parity_prefill_layer_attn_norm_host);
        if (self.parity_prefill_layer_ffn_norm_host.len > 0) self.allocator.free(self.parity_prefill_layer_ffn_norm_host);
        if (self.parity_prefill_block_out_host.len > 0) self.allocator.free(self.parity_prefill_block_out_host);
        if (self.gated_delta_stage_input_host.len > 0) self.allocator.free(self.gated_delta_stage_input_host);
        if (self.gated_delta_stage_mid_host.len > 0) self.allocator.free(self.gated_delta_stage_mid_host);
        if (self.gated_delta_stage_output_host.len > 0) self.allocator.free(self.gated_delta_stage_output_host);
        if (self.cpu_rope_local) |rope| {
            rope.deinit(self.allocator);
            self.allocator.destroy(rope);
        }
        if (self.cpu_rope_global) |rope| {
            rope.deinit(self.allocator);
            self.allocator.destroy(rope);
        }
        self.allocator.free(self.slot_logits);
        if (self.decode_ptr_tables_cached_slots.len > 0) self.allocator.free(self.decode_ptr_tables_cached_slots);
        if (self.slot_in_use.len > 0) self.allocator.free(self.slot_in_use);
        if (self.slot_positions.len > 0) self.allocator.free(self.slot_positions);
        if (self.slot_rope_position_deltas.len > 0) self.allocator.free(self.slot_rope_position_deltas);
        self.deinitSlotKvStates();
        self.deinitLayerProgramSlotBuffers();
        if (self.attn_scores_workspace_dev) |*buf| buf.deinit(&self.device);
        if (self.attn_u16_workspace_dev) |*buf| buf.deinit(&self.device);
        if (self.pipeline_backend1) |stage1| {
            stage1.deinit();
            self.allocator.destroy(stage1);
            self.pipeline_backend1 = null;
        }
        if (self.pipeline_backend0_cpu) |stage0_cpu| {
            stage0_cpu.deinit();
            self.allocator.destroy(stage0_cpu);
            self.pipeline_backend0_cpu = null;
        }
        // Pipeline2: release stage 1 resources before stage 0 (reverse init order).
        if (self.pipeline_host_staging_stage12) |buf| {
            self.allocator.free(buf);
            self.pipeline_host_staging_stage12 = null;
        }
        if (self.pipeline_host_staging) |buf| {
            self.allocator.free(buf);
            self.pipeline_host_staging = null;
        }
        self.block_runtime.deinit(self.allocator, &self.device);
        self.runtime_buffers.deinit(self.allocator, &self.device);
        self.blas.deinit(&self.device);
        self.kernel_arg_pack.deinit();
        self.kernel_registry.deinit();
        self.device.deinit();
        self.* = undefined;
    }

    pub fn synchronize(self: *CudaBackend) !void {
        if (self.compute_stream) |stream| {
            try self.device.synchronizeStream(stream);
            return;
        }
        try self.device.synchronize();
    }

    pub fn pipelineStage1(self: *CudaBackend) ?*CudaBackend {
        return self.pipeline_backend1;
    }

    pub fn pipelineCpuStage0(self: *CudaBackend) ?*cpu_backend.BackendType {
        return self.pipeline_backend0_cpu;
    }

    pub fn pipelineSplitLayer(self: *const CudaBackend) usize {
        return self.split_layer;
    }

    pub fn pipelineSplitLayerStage2(self: *const CudaBackend) usize {
        return self.split_layer_stage2;
    }

    pub fn pipelineActivationByteCount(self: *const CudaBackend) !usize {
        const element_bytes: usize = switch (self.pipeline_boundary_dtype) {
            .bf16, .f16 => @sizeOf(u16),
            .f32 => @sizeOf(f32),
        };
        return std.math.mul(usize, self.d_model, element_bytes) catch error.InvalidArgument;
    }

    pub fn transferPipelineActivation(self: *CudaBackend, dst: *CudaBackend, byte_count: usize) !void {
        if (byte_count == 0) return;
        switch (self.pipeline_transfer_mode) {
            .peer_to_peer => {
                try self.device.memcpyPeerAsync(
                    dst.runtime_buffers.input_dev.pointer,
                    dst.device.context,
                    self.runtime_buffers.input_dev.pointer,
                    self.device.context,
                    byte_count,
                    self.compute_stream,
                );
                if (self.compute_stream) |stream| {
                    try self.device.synchronizeStream(stream);
                } else {
                    try self.device.synchronize();
                }
            },
            .host_staged => {
                const staging = self.pipeline_host_staging orelse return error.PipelineTransferNotInitialized;
                if (byte_count > staging.len) return error.PipelineTransferBufferTooSmall;
                try self.runtime_buffers.input_dev.download(&self.device, staging[0..byte_count]);
                try dst.runtime_buffers.input_dev.upload(&dst.device, staging[0..byte_count]);
            },
            .none => return error.InvalidTopologyConfig,
        }
    }

    pub fn transferPipelineActivationFromCpu(
        self: *CudaBackend,
        src: *cpu_backend.BackendType,
        slot_index: usize,
        byte_count: usize,
    ) !void {
        if (byte_count == 0) return;
        const staging = self.pipeline_host_staging orelse return error.PipelineTransferNotInitialized;
        if (byte_count > staging.len) return error.PipelineTransferBufferTooSmall;
        const src_activation = src.slotActivationBytes(slot_index);
        if (byte_count > src_activation.len) return error.InvalidArgument;
        @memcpy(staging[0..byte_count], src_activation[0..byte_count]);
        try self.runtime_buffers.input_dev.upload(&self.device, staging[0..byte_count]);
    }

    pub fn maxBatchSize(self: *const CudaBackend) usize {
        return self.max_batch_size;
    }

    /// Central KV allocation seam for future storage backends.
    pub fn allocKvPair(self: *CudaBackend, capacity: usize, kv_dim: usize) !DeviceKvPair {
        return switch (self.kv_storage_mode) {
            .device => allocDeviceKvPair(&self.device, capacity, kv_dim),
        };
    }

    fn computeDeviceMemoryBudget(self: *const CudaBackend) DeviceMemoryBudget {
        const weights_bytes = saturatingAddUsize(
            self.block_runtime.linear_weight_bytes,
            self.block_runtime.norm_weight_bytes,
        );
        const runtime_bytes = self.runtime_buffers.deviceByteSize();

        var kv_state_bytes: usize = 0;
        var gated_delta_state_bytes: usize = 0;
        var shortconv_state_bytes: usize = 0;
        for (self.slot_kv_states) |slot| {
            for (slot.kv) |entry| {
                kv_state_bytes = saturatingAddUsize(kv_state_bytes, entry.k.size);
                kv_state_bytes = saturatingAddUsize(kv_state_bytes, entry.v.size);
            }
            for (slot.gd) |entry| {
                gated_delta_state_bytes = saturatingAddUsize(gated_delta_state_bytes, entry.conv.size);
                gated_delta_state_bytes = saturatingAddUsize(gated_delta_state_bytes, entry.ssm.size);
            }
            for (slot.sc) |entry| {
                shortconv_state_bytes = saturatingAddUsize(shortconv_state_bytes, entry.conv.size);
            }
        }

        var layer_program_bytes: usize = 0;
        for (self.layer_program_slot_buffers) |buf| {
            layer_program_bytes = saturatingAddUsize(layer_program_bytes, buf.size);
        }

        var workspace_bytes: usize = 0;
        if (self.attn_scores_workspace_dev) |buf| {
            workspace_bytes = saturatingAddUsize(workspace_bytes, buf.size);
        }
        if (self.attn_u16_workspace_dev) |buf| {
            workspace_bytes = saturatingAddUsize(workspace_bytes, buf.size);
        }

        const misc_bytes = self.argmax_index_dev.size;

        return .{
            .weights_bytes = weights_bytes,
            .runtime_bytes = runtime_bytes,
            .kv_state_bytes = kv_state_bytes,
            .gated_delta_state_bytes = gated_delta_state_bytes,
            .shortconv_state_bytes = shortconv_state_bytes,
            .layer_program_bytes = layer_program_bytes,
            .workspace_bytes = workspace_bytes,
            .dequant_cache_bytes = self.dequant_cache_bytes,
            .strict_guard_bytes = self.strict_guard_bytes,
            .misc_bytes = misc_bytes,
        };
    }

    /// Returns a per-slot logits slice for the given slot index.
    pub fn slotLogits(self: *CudaBackend, slot_index: usize) []f32 {
        const offset = slot_index * self.vocab_size;
        return self.slot_logits[offset .. offset + self.vocab_size];
    }

    /// Initialize per-slot KV, gated delta, and shortconv state buffers.
    /// Called after block_runtime is initialized.
    fn initSlotKvStates(self: *CudaBackend) !void {
        const n_attn = self.block_runtime.attention_block_count;
        const n_gd = self.block_runtime.gated_delta_block_count;
        const n_sc = self.block_runtime.shortconv_block_count;
        self.slot_kv_states = try self.allocator.alloc(SlotKvStates, self.max_batch_size);
        errdefer self.allocator.free(self.slot_kv_states);
        var initialized_slots: usize = 0;
        errdefer {
            // Slot 0's device buffers are aliases of block_runtime — skip deinit.
            for (self.slot_kv_states[0..initialized_slots], 0..) |*sks, err_slot_idx| {
                if (err_slot_idx > 0) {
                    for (sks.kv) |*kv| {
                        kv.k.deinit(&self.device);
                        kv.v.deinit(&self.device);
                    }
                    for (sks.gd) |*gd| {
                        gd.conv.deinit(&self.device);
                        gd.ssm.deinit(&self.device);
                    }
                    for (sks.sc) |*sc| {
                        sc.conv.deinit(&self.device);
                    }
                }
                self.allocator.free(sks.kv);
                self.allocator.free(sks.gd);
                self.allocator.free(sks.sc);
            }
        }
        // Slot 0 takes the existing buffers from block_runtime.
        // Slots 1..N-1 allocate fresh buffers.
        for (self.slot_kv_states, 0..) |*sks, slot_idx| {
            sks.kv = try self.allocator.alloc(SlotKvStates.KvEntry, n_attn);
            errdefer self.allocator.free(sks.kv);
            sks.gd = try self.allocator.alloc(SlotKvStates.GdEntry, n_gd);
            errdefer self.allocator.free(sks.gd);
            sks.sc = try self.allocator.alloc(SlotKvStates.ScEntry, n_sc);
            errdefer self.allocator.free(sks.sc);
            var attn_i: usize = 0;
            var gd_i: usize = 0;
            var sc_i: usize = 0;
            // Clean up device buffers stored in this slot's entries so far.
            // Slot 0 entries are block_runtime aliases — only free for slot > 0.
            errdefer if (slot_idx > 0) {
                for (sks.kv[0..attn_i]) |*kv| {
                    kv.k.deinit(&self.device);
                    kv.v.deinit(&self.device);
                }
                for (sks.gd[0..gd_i]) |*gd| {
                    gd.conv.deinit(&self.device);
                    gd.ssm.deinit(&self.device);
                }
                for (sks.sc[0..sc_i]) |*sc| {
                    sc.conv.deinit(&self.device);
                }
            };
            for (self.block_runtime.blocks) |*layer| {
                if (layer.attention_binding) |block| {
                    if (slot_idx == 0) {
                        sks.kv[attn_i] = .{ .k = block.k_cache, .v = block.v_cache, .capacity = block.kv_capacity };
                    } else {
                        var kv_pair = try self.allocKvPair(block.kv_capacity, block.kv_dim);
                        errdefer {
                            kv_pair.v.deinit(&self.device);
                            kv_pair.k.deinit(&self.device);
                        }
                        sks.kv[attn_i] = .{ .k = kv_pair.k, .v = kv_pair.v, .capacity = block.kv_capacity };
                    }
                    attn_i += 1;
                }
                if (layer.gated_delta_binding) |block| {
                    if (slot_idx == 0) {
                        sks.gd[gd_i] = .{ .conv = block.conv_state_dev, .ssm = block.ssm_state_dev, .conv_ring_head = block.conv_ring_head };
                    } else {
                        var conv = try self.device.allocBuffer(block.conv_state_dev.size);
                        errdefer conv.deinit(&self.device);
                        var ssm = try self.device.allocBuffer(block.ssm_state_dev.size);
                        errdefer ssm.deinit(&self.device);
                        sks.gd[gd_i] = .{ .conv = conv, .ssm = ssm, .conv_ring_head = 0 };
                    }
                    gd_i += 1;
                }
                if (layer.shortconv_binding) |block| {
                    if (slot_idx == 0) {
                        sks.sc[sc_i] = .{ .conv = block.conv_state };
                    } else {
                        var conv = try self.device.allocBuffer(block.conv_state.size);
                        errdefer conv.deinit(&self.device);
                        const elems = std.math.divExact(usize, conv.size, @sizeOf(f32)) catch return error.InvalidArgument;
                        const zeros = try self.allocator.alloc(f32, elems);
                        defer self.allocator.free(zeros);
                        @memset(zeros, 0.0);
                        try conv.upload(&self.device, std.mem.sliceAsBytes(zeros));
                        sks.sc[sc_i] = .{ .conv = conv };
                    }
                    sc_i += 1;
                }
            }
            initialized_slots += 1;
        }
    }

    fn deinitSlotKvStates(self: *CudaBackend) void {
        // Slot 0's buffers are owned by block_runtime — skip them.
        for (self.slot_kv_states, 0..) |*sks, slot_idx| {
            if (slot_idx > 0) {
                for (sks.kv) |*kv| {
                    kv.k.deinit(&self.device);
                    kv.v.deinit(&self.device);
                }
                for (sks.gd) |*gd| {
                    gd.conv.deinit(&self.device);
                    gd.ssm.deinit(&self.device);
                }
                for (sks.sc) |*sc| {
                    sc.conv.deinit(&self.device);
                }
            }
            self.allocator.free(sks.kv);
            self.allocator.free(sks.gd);
            self.allocator.free(sks.sc);
        }
        if (self.slot_kv_states.len > 0) self.allocator.free(self.slot_kv_states);
    }

    /// Swap the active slot's KV/state buffer pointers out of block_runtime
    /// and load the target slot's buffers in.
    pub fn activateKvSlot(self: *CudaBackend, slot_index: usize) void {
        if (self.active_kv_slot == slot_index) return;
        self.saveActiveKvSlot();
        self.loadKvSlot(slot_index);
        self.active_kv_slot = slot_index;
    }

    pub fn saveActiveKvSlot(self: *CudaBackend) void {
        const sks = &self.slot_kv_states[self.active_kv_slot];
        var attn_i: usize = 0;
        var gd_i: usize = 0;
        var sc_i: usize = 0;
        for (self.block_runtime.blocks) |*layer| {
            if (layer.attention_binding) |block| {
                sks.kv[attn_i] = .{ .k = block.k_cache, .v = block.v_cache, .capacity = block.kv_capacity };
                attn_i += 1;
            }
            if (layer.gated_delta_binding) |block| {
                sks.gd[gd_i] = .{ .conv = block.conv_state_dev, .ssm = block.ssm_state_dev, .conv_ring_head = block.conv_ring_head };
                gd_i += 1;
            }
            if (layer.shortconv_binding) |block| {
                sks.sc[sc_i] = .{ .conv = block.conv_state };
                sc_i += 1;
            }
        }
    }

    pub fn loadKvSlot(self: *CudaBackend, slot_index: usize) void {
        const sks = &self.slot_kv_states[slot_index];
        var attn_i: usize = 0;
        var gd_i: usize = 0;
        var sc_i: usize = 0;
        for (self.block_runtime.blocks) |*layer| {
            if (layer.attention_binding) |block| {
                block.k_cache = sks.kv[attn_i].k;
                block.v_cache = sks.kv[attn_i].v;
                block.kv_capacity = sks.kv[attn_i].capacity;
                attn_i += 1;
            }
            if (layer.gated_delta_binding) |block| {
                block.conv_state_dev = sks.gd[gd_i].conv;
                block.ssm_state_dev = sks.gd[gd_i].ssm;
                block.conv_ring_head = sks.gd[gd_i].conv_ring_head;
                gd_i += 1;
            }
            if (layer.shortconv_binding) |block| {
                block.conv_state = sks.sc[sc_i].conv;
                sc_i += 1;
            }
        }
    }

    fn layerProgramRequiredSlotCount(self: *const CudaBackend) usize {
        var required: usize = 0;
        for (self.block_runtime.blocks) |layer| {
            for (layer.register_to_slot_map) |slot_idx| {
                if (slot_idx == BlockRuntimeLayer.invalid_slot) continue;
                const next = @as(usize, slot_idx) + 1;
                if (next > required) required = next;
            }
        }
        return required;
    }

    fn initLayerProgramSlotBuffers(self: *CudaBackend) !void {
        const required = self.layerProgramRequiredSlotCount();
        if (required == 0) {
            self.layer_program_slot_buffers = &.{};
            self.layer_program_slot_ptrs = &.{};
            self.layer_program_slot_widths = &.{};
            self.layer_program_row_capacity = 1;
            return;
        }

        self.layer_program_slot_widths = try self.allocator.alloc(usize, required);
        errdefer self.allocator.free(self.layer_program_slot_widths);
        @memset(self.layer_program_slot_widths, 0);
        for (self.block_runtime.blocks) |layer| {
            for (layer.slot_width_hints, 0..) |width, slot_idx| {
                if (slot_idx >= self.layer_program_slot_widths.len) continue;
                if (self.layer_program_slot_widths[slot_idx] == 0) {
                    self.layer_program_slot_widths[slot_idx] = width;
                } else if (width != 0 and self.layer_program_slot_widths[slot_idx] != width) {
                    return error.InvalidRegisterSpecSize;
                }
            }
        }
        self.layer_program_slot_buffers = try self.allocator.alloc(compute.cuda.Buffer, required);
        errdefer self.allocator.free(self.layer_program_slot_buffers);

        var initialized: usize = 0;
        errdefer {
            for (self.layer_program_slot_buffers[0..initialized]) |*buf| {
                buf.deinit(&self.device);
            }
        }

        for (0..required) |idx| {
            const width = self.layer_program_slot_widths[idx];
            if (width == 0) return error.InvalidRegisterSpecSize;
            const bytes = std.math.mul(usize, width, @sizeOf(f32)) catch return error.InvalidArgument;
            self.layer_program_slot_buffers[idx] = try self.device.allocBuffer(bytes);
            initialized += 1;
        }

        // Pre-allocate pointer array for execution dispatch.
        self.layer_program_slot_ptrs = try self.allocator.alloc(*compute.cuda.Buffer, required);
        for (self.layer_program_slot_buffers, 0..) |*buf, idx| {
            self.layer_program_slot_ptrs[idx] = buf;
        }
        self.layer_program_row_capacity = 1;
    }

    pub fn ensureLayerProgramSlotRowCapacity(
        self: *CudaBackend,
        required_rows: usize,
        fixed_alloc_mode: bool,
    ) !void {
        if (required_rows == 0) return error.InvalidArgument;
        if (required_rows <= self.layer_program_row_capacity) return;
        if (required_rows > self.max_seq_len) return error.InvalidArgument;
        if (self.layer_program_slot_buffers.len == 0) return;
        if (fixed_alloc_mode) return error.OutOfMemory;

        var new_capacity = self.layer_program_row_capacity;
        if (new_capacity == 0) new_capacity = 1;
        while (new_capacity < required_rows) {
            const doubled = std.math.mul(usize, new_capacity, 2) catch self.max_seq_len;
            const next = if (doubled > new_capacity) doubled else self.max_seq_len;
            new_capacity = @min(self.max_seq_len, next);
            if (new_capacity == self.max_seq_len) break;
        }
        if (new_capacity < required_rows) return error.InvalidArgument;
        for (self.layer_program_slot_buffers, 0..) |*buf, idx| {
            const width = self.layer_program_slot_widths[idx];
            const row_bytes = std.math.mul(usize, width, @sizeOf(f32)) catch return error.InvalidArgument;
            try resizeScratchBuffer(&self.device, buf, std.math.mul(usize, row_bytes, new_capacity) catch return error.InvalidArgument);
        }
        self.layer_program_row_capacity = new_capacity;
    }

    fn preallocateFixedAllocBuffers(self: *CudaBackend) !void {
        if (!self.fixed_alloc_mode) return;

        // Preallocate all grow-only CUDA buffers to their configured maxima.
        // After this point, fixed_alloc_mode enforces "no growth at runtime".
        const prev_fixed_mode = self.fixed_alloc_mode;
        self.fixed_alloc_mode = false;
        defer self.fixed_alloc_mode = prev_fixed_mode;

        var slot_index: usize = 0;
        while (slot_index < self.max_batch_size) : (slot_index += 1) {
            self.activateKvSlot(slot_index);
            try engine_forward.ensureKvCapacity(self, self.max_seq_len);
        }
        self.saveActiveKvSlot();
        self.activateKvSlot(0);

        try self.runtime_buffers.ensureRowCapacity(&self.device, self.max_seq_len, false);
        try self.ensureLayerProgramSlotRowCapacity(self.max_seq_len, false);

        const kv_groups_u32: u32 = @intCast(self.n_heads / self.n_kv_heads);
        const prefill_rows: usize = @min(self.max_seq_len, self.prefill_chunk_rows_cap);
        const prefill_rows_u32: u32 = @intCast(prefill_rows);
        const max_seq_len_u32: u32 = @intCast(self.max_seq_len);

        _ = try engine_mixers.ensureAttnScoresWorkspace(
            self,
            kv_groups_u32,
            prefill_rows_u32,
            max_seq_len_u32,
        );

        const q_f16_elems = std.math.mul(usize, prefill_rows, self.runtime_buffers.max_attn) catch return error.InvalidArgument;
        const probs_f16_elems = std.math.mul(
            usize,
            std.math.mul(usize, @as(usize, kv_groups_u32), prefill_rows) catch return error.InvalidArgument,
            self.max_seq_len,
        ) catch return error.InvalidArgument;
        const u16_workspace_bytes = std.math.mul(
            usize,
            q_f16_elems + probs_f16_elems,
            @sizeOf(u16),
        ) catch return error.InvalidArgument;
        _ = try engine_mixers.ensureAttnU16Workspace(self, u16_workspace_bytes);
    }

    fn deinitLayerProgramSlotBuffers(self: *CudaBackend) void {
        if (self.layer_program_slot_ptrs.len > 0) {
            self.allocator.free(self.layer_program_slot_ptrs);
            self.layer_program_slot_ptrs = &.{};
        }
        if (self.layer_program_slot_buffers.len == 0) return;
        for (self.layer_program_slot_buffers) |*buf| {
            buf.deinit(&self.device);
        }
        self.allocator.free(self.layer_program_slot_buffers);
        self.layer_program_slot_buffers = &.{};
        if (self.layer_program_slot_widths.len > 0) {
            self.allocator.free(self.layer_program_slot_widths);
            self.layer_program_slot_widths = &.{};
        }
        self.layer_program_row_capacity = 1;
    }

    pub fn vocabSize(self: *const CudaBackend) usize {
        return self.vocab_size;
    }

    fn initialKvCapacity(self: *const CudaBackend) usize {
        for (self.block_runtime.blocks) |layer| {
            if (layer.attention_binding) |block| return block.kv_capacity;
        }
        return 0;
    }

    pub fn prefill(self: *CudaBackend, tokens: []const u32, logits_out: []f32) !void {
        try self.ensureSlotStateBlocksBoundForScheduler(0);
        return prefill_mod.prefill(self, tokens, logits_out);
    }

    pub fn decode(self: *CudaBackend, token: u32, position: usize, logits_out: []f32) !void {
        try self.ensureSlotStateBlocksBoundForScheduler(0);
        return decode_mod.decode(self, token, position, logits_out);
    }

    pub fn decodeStreaming(
        self: *CudaBackend,
        first_token: u32,
        start_position: usize,
        max_tokens: usize,
        eos_token_ids: []const u32,
        output_tokens: []u32,
        callback: ?*const fn (u32, ?*anyopaque) void,
        callback_data: ?*anyopaque,
    ) !usize {
        try self.ensureSlotStateBlocksBoundForScheduler(0);
        return decode_mod.decodeStreaming(
            self,
            first_token,
            start_position,
            max_tokens,
            eos_token_ids,
            output_tokens,
            callback,
            callback_data,
        );
    }

    pub fn supportsSchedulerBackendDecodeStreamingRoute(self: *const CudaBackend) bool {
        _ = self;
        // Parity mode: force n=1 and n>1 through queued decodeBatch.
        // Do not take single-request streaming shortcut route.
        return false;
    }

    pub fn supportsSchedulerBackendTopKDecodeRoute(
        self: *const CudaBackend,
        sampling_config: *const sampling_mod.SamplingConfig,
    ) bool {
        _ = self;
        _ = sampling_config;
        // Keep single-request top-k candidate route disabled for CUDA.
        // Batched top-k routing is exposed via supportsSchedulerBackendBatchedTopKDecodeRoute().
        return false;
    }

    pub fn supportsSchedulerBackendBatchedTopKDecodeRoute(
        self: *const CudaBackend,
        sampling_config: *const sampling_mod.SamplingConfig,
    ) bool {
        return sampling_config.strategy == .top_k and
            sampling_config.top_k > 0 and
            sampling_config.top_k <= 256 and
            sampling_config.temperature > 0.0 and
            self.loaded.config.logits_scaling > 0.0;
    }

    pub fn incrementDecodeMetadataInPlace(
        self: *CudaBackend,
        decode_seq_lens_dev: *compute.cuda.Buffer,
        decode_positions_dev: *compute.cuda.Buffer,
        rows: usize,
    ) !void {
        if (rows == 0) return error.InvalidArgument;
        const rows_u32 = std.math.cast(u32, rows) orelse return error.InvalidArgument;
        if (self.decode_u32_increment_function == null) {
            if (self.kernel_registry.embedded_module == null) {
                try self.kernel_registry.loadEmbeddedModule(compute.cuda.decode_u32_increment.embedded_module);
            }
            const resolved = try self.kernel_registry.resolveFunction(
                compute.cuda.decode_u32_increment.op_name,
                compute.cuda.decode_u32_increment.embedded_symbol,
            );
            self.decode_u32_increment_function = resolved.function;
            self.decode_u32_increment_source = resolved.source;
        }
        try compute.cuda.decode_u32_increment.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            self.decode_u32_increment_function orelse return error.CudaKernelUnavailable,
            decode_seq_lens_dev,
            decode_positions_dev,
            rows_u32,
        );
    }

    fn extractTopKFromBatchedDeviceLogits(
        self: *CudaBackend,
        batch_rows: usize,
        top_k: usize,
        candidate_logits_out: []f32,
        candidate_ids_out: []u32,
        candidate_counts_out: []usize,
    ) !void {
        if (batch_rows == 0) return error.InvalidArgument;
        if (top_k == 0) return error.InvalidArgument;
        if (candidate_counts_out.len < batch_rows) return error.InvalidArgument;
        const total_candidates = std.math.mul(usize, batch_rows, top_k) catch return error.InvalidArgument;
        if (candidate_logits_out.len < total_candidates or candidate_ids_out.len < total_candidates) {
            return error.InvalidArgument;
        }

        const vocab = self.runtime_buffers.projected_vocab;
        if (vocab == 0) return error.InvalidArgument;
        const per_row_count = @min(top_k, vocab);
        const logits_scaling = self.loaded.config.logits_scaling;
        if (logits_scaling <= 0.0) return error.UnsupportedModel;
        if (top_k > 256) return error.InvalidArgument;

        const total_logits = std.math.mul(usize, batch_rows, vocab) catch return error.InvalidArgument;
        const total_logits_bytes = std.math.mul(usize, total_logits, @sizeOf(f32)) catch return error.InvalidArgument;
        var logits_batch = try engine_weights.bufferSlice(&self.runtime_buffers.logits_dev, 0, total_logits_bytes);

        if (self.topk_rows_function == null) {
            if (self.kernel_registry.embedded_module == null) {
                try self.kernel_registry.loadEmbeddedModule(compute.cuda.topk_rows_f32.embedded_module);
            }
            const resolved = try self.kernel_registry.resolveFunction(
                compute.cuda.topk_rows_f32.op_name,
                compute.cuda.topk_rows_f32.embedded_symbol,
            );
            self.topk_rows_function = resolved.function;
            self.topk_rows_source = resolved.source;
        }
        const topk_rows_function = self.topk_rows_function orelse return error.CudaKernelUnavailable;
        const rows_u32 = std.math.cast(u32, batch_rows) orelse return error.InvalidArgument;
        const vocab_u32 = std.math.cast(u32, vocab) orelse return error.InvalidArgument;
        const row_stride_u32 = std.math.cast(u32, top_k) orelse return error.InvalidArgument;
        const per_row_count_u32 = std.math.cast(u32, per_row_count) orelse return error.InvalidArgument;

        try compute.cuda.topk_rows_f32.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            topk_rows_function,
            &self.runtime_buffers.topk_values_dev,
            &self.runtime_buffers.topk_ids_dev,
            &logits_batch,
            rows_u32,
            vocab_u32,
            row_stride_u32,
            per_row_count_u32,
        );

        const topk_total = std.math.mul(usize, batch_rows, top_k) catch return error.InvalidArgument;
        const topk_total_logits = candidate_logits_out[0..topk_total];
        const topk_total_ids = candidate_ids_out[0..topk_total];
        try self.runtime_buffers.topk_values_dev.download(&self.device, std.mem.sliceAsBytes(topk_total_logits));
        try self.runtime_buffers.topk_ids_dev.download(&self.device, std.mem.sliceAsBytes(topk_total_ids));

        for (0..batch_rows) |row_index| {
            candidate_counts_out[row_index] = per_row_count;
        }
        if (logits_scaling != 1.0) {
            for (0..batch_rows) |row_index| {
                const row_base = std.math.mul(usize, row_index, top_k) catch return error.InvalidArgument;
                for (0..per_row_count) |k_index| {
                    const out_idx = row_base + k_index;
                    candidate_logits_out[out_idx] /= logits_scaling;
                }
            }
        }
    }

    pub fn decodeBatchTopKCandidates(
        self: *CudaBackend,
        requests: []const contract.DecodeRequest,
        top_k: usize,
        candidate_logits_out: []f32,
        candidate_ids_out: []u32,
        candidate_counts_out: []usize,
    ) !void {
        if (requests.len == 0) return error.InvalidArgument;
        if (top_k == 0) return error.InvalidArgument;
        if (top_k > 256) return error.InvalidArgument;
        if (self.loaded.config.logits_scaling <= 0.0) return error.UnsupportedModel;
        const max_n = 128;
        if (requests.len > max_n) return error.InvalidArgument;

        var tokens_buf: [max_n]u32 = undefined;
        var slot_indices_buf: [max_n]usize = undefined;
        var positions_buf: [max_n]usize = undefined;
        var raw_positions_buf: [max_n]usize = undefined;
        for (requests, 0..) |req, i| {
            if (!self.slotIndexSupported(req.slot_index) or !self.slot_in_use[req.slot_index]) {
                return error.InvalidArgument;
            }
            try self.ensureSlotStateBlocksBoundForScheduler(req.slot_index);
            const raw_position = self.slot_positions[req.slot_index];
            const effective_position = try common_mrope.applyPositionDelta(
                raw_position,
                self.slot_rope_position_deltas[req.slot_index],
            );
            tokens_buf[i] = req.token;
            slot_indices_buf[i] = req.slot_index;
            positions_buf[i] = effective_position;
            raw_positions_buf[i] = raw_position;
        }
        try self.computeBatchedDecodeLogitsDeviceOnly(
            tokens_buf[0..requests.len],
            slot_indices_buf[0..requests.len],
            positions_buf[0..requests.len],
        );
        for (requests, 0..) |req, i| {
            self.slot_positions[req.slot_index] = raw_positions_buf[i] + 1;
        }

        try self.extractTopKFromBatchedDeviceLogits(
            requests.len,
            top_k,
            candidate_logits_out,
            candidate_ids_out,
            candidate_counts_out,
        );
    }

    pub fn decodeTopKCandidates(
        self: *CudaBackend,
        slot_index: usize,
        token: u32,
        top_k: usize,
        candidate_logits_out: []f32,
        candidate_ids_out: []u32,
    ) !usize {
        if (top_k == 0) return error.InvalidArgument;
        if (candidate_logits_out.len < top_k or candidate_ids_out.len < top_k) return error.InvalidArgument;
        var counts: [1]usize = .{0};
        const request = [_]contract.DecodeRequest{
            .{
                .slot_index = slot_index,
                .token = token,
            },
        };
        try self.decodeBatchTopKCandidates(
            request[0..],
            top_k,
            candidate_logits_out[0..top_k],
            candidate_ids_out[0..top_k],
            counts[0..],
        );
        return counts[0];
    }

    pub fn allocSlot(self: *CudaBackend) ?usize {
        const slot_index = decode_mod.allocSlot(self) orelse return null;
        if (self.pipeline_backend1) |stage1| {
            const stage1_slot_index = stage1.allocSlot() orelse {
                decode_mod.freeSlot(self, slot_index);
                self.unbindSlotStateBlocks(slot_index);
                return null;
            };
            if (stage1_slot_index != slot_index) {
                stage1.freeSlot(stage1_slot_index);
                decode_mod.freeSlot(self, slot_index);
                self.unbindSlotStateBlocks(slot_index);
                return null;
            }
            stage1.unbindSlotStateBlocks(slot_index);
        }
        if (self.pipeline_backend0_cpu) |stage0_cpu| {
            const cpu_slot_index = stage0_cpu.allocSlot() orelse {
                if (self.pipeline_backend1) |stage1| stage1.freeSlot(slot_index);
                decode_mod.freeSlot(self, slot_index);
                self.unbindSlotStateBlocks(slot_index);
                return null;
            };
            if (cpu_slot_index != slot_index) {
                stage0_cpu.freeSlot(cpu_slot_index);
                if (self.pipeline_backend1) |stage1| stage1.freeSlot(slot_index);
                decode_mod.freeSlot(self, slot_index);
                self.unbindSlotStateBlocks(slot_index);
                return null;
            }
            stage0_cpu.unbindSlotStateBlocks(slot_index);
        }
        self.unbindSlotStateBlocks(slot_index);
        return slot_index;
    }

    pub fn freeSlot(self: *CudaBackend, slot_index: usize) void {
        decode_mod.freeSlot(self, slot_index);
        if (self.pipeline_backend1) |stage1| {
            stage1.freeSlot(slot_index);
        }
        if (self.pipeline_backend0_cpu) |stage0_cpu| {
            stage0_cpu.freeSlot(slot_index);
        }
        self.unbindSlotStateBlocks(slot_index);
    }

    pub fn resetSlot(self: *CudaBackend, slot_index: usize) void {
        decode_mod.resetSlot(self, slot_index);
        if (self.pipeline_backend1) |stage1| {
            stage1.resetSlot(slot_index);
        }
        if (self.pipeline_backend0_cpu) |stage0_cpu| {
            stage0_cpu.resetSlot(slot_index);
        }
        if (self.state_descriptor_count == 0) return;
        if (!self.slotIndexSupported(slot_index)) return;
        if (!self.slot_state_bindings[slot_index].bound) return;
        engine_forward.resetShortConvStates(self) catch |err| {
            log.warn("inference", "CUDA resetSlot shortconv reset failed", .{
                .slot_index = slot_index,
                .reason = @errorName(err),
            });
        };
        engine_forward.resetAttentionCpuStates(self);
        engine_forward.resetGatedDeltaStates(self);
    }

    pub fn getPosition(self: *const CudaBackend, slot_index: usize) usize {
        return decode_mod.getPosition(self, slot_index);
    }

    pub fn stateDescriptors(self: *const CudaBackend) []const runtime_contract.StateDescriptor {
        return self.state_descriptors_storage[0..self.state_descriptor_count];
    }

    fn bindRuntimeState(
        self: *CudaBackend,
        slot_index: usize,
        runtime_kind: u8,
        state_block: *runtime_contract.StateBlockHandle,
    ) !void {
        if (runtime_kind == runtime_contract.state_runtime_kind_none) {
            return;
        }
        if (runtime_kind == runtime_contract.state_runtime_kind_kv_cache) {
            const state_value = runtime_contract.stateValueFromBlock(*KvRuntimeState, state_block) orelse {
                return error.InvalidStateDescriptorBinding;
            };
            state_value.* = .{
                .runtime_kind = runtime_kind,
                .block_runtime = &self.block_runtime,
                .slot_index = slot_index,
            };
            return;
        }
        const state_value = runtime_contract.stateValueFromBlock(*RecurrentRuntimeState, state_block) orelse {
            return error.InvalidStateDescriptorBinding;
        };
        state_value.* = .{
            .runtime_kind = runtime_kind,
            .block_runtime = &self.block_runtime,
            .slot_index = slot_index,
        };
    }

    fn synthesizeRuntimeStateBlockForDescriptor(
        self: *CudaBackend,
        binding: *SlotStateBinding,
        descriptor: runtime_contract.StateDescriptor,
        slot_index: usize,
        binding_index: usize,
    ) !runtime_contract.StateBlockHandle {
        const block_storage = &binding.local_blocks[binding_index];
        switch (descriptor.runtime_kind) {
            runtime_contract.state_runtime_kind_kv_cache,
            runtime_contract.state_runtime_kind_shortconv_cache,
            runtime_contract.state_runtime_kind_mamba_cache,
            runtime_contract.state_runtime_kind_gated_delta_cache,
            => {},
            else => return error.InvalidStateDescriptorBinding,
        }
        if (descriptor.align_bytes > 64) return error.InvalidStateDescriptorBinding;
        if (descriptor.size_bytes > block_storage.len) return error.InvalidStateDescriptorBinding;
        if (descriptor.zero_init) @memset(block_storage, 0);

        var local_handle: runtime_contract.StateBlockHandle = .{
            .id = descriptor.id,
            .ptr = block_storage[0..].ptr,
            .size = block_storage.len,
            .align_bytes = 64,
        };
        try bindRuntimeState(self, slot_index, descriptor.runtime_kind, &local_handle);
        return local_handle;
    }

    pub fn bindSlotStateBlocks(
        self: *CudaBackend,
        slot_index: usize,
        state_blocks: []const runtime_contract.StateBlockHandle,
    ) !void {
        if (!self.slotIndexSupported(slot_index)) return error.InvalidArgument;
        runtime_contract.validateStateBlocksForDescriptors(self.stateDescriptors(), state_blocks) catch |err| {
            log.warn("inference", "CUDA bindSlotStateBlocks descriptor validation failed", .{
                .slot_index = slot_index,
                .state_blocks = state_blocks.len,
                .state_descriptors = self.stateDescriptors().len,
                .reason = @errorName(err),
            });
            return err;
        };
        var binding = &self.slot_state_bindings[slot_index];
        if (state_blocks.len > binding.handles.len) {
            log.warn("inference", "CUDA bindSlotStateBlocks too many state blocks", .{
                .slot_index = slot_index,
                .state_blocks = state_blocks.len,
                .capacity = binding.handles.len,
            });
            return error.InvalidStateDescriptorBinding;
        }
        for (self.stateDescriptors(), 0..) |descriptor, idx| {
            const incoming = runtime_contract.findStateBlock(state_blocks, descriptor.id) orelse {
                log.warn("inference", "CUDA bindSlotStateBlocks missing descriptor state id", .{
                    .slot_index = slot_index,
                    .state_id = descriptor.id,
                });
                return error.InvalidStateDescriptorBinding;
            };
            var bound = incoming.*;
            try bindRuntimeState(self, slot_index, descriptor.runtime_kind, &bound);
            binding.handles[idx] = .{
                .id = descriptor.id,
                .ptr = bound.ptr,
                .size = bound.size,
                .align_bytes = bound.align_bytes,
            };
        }
        binding.count = @intCast(state_blocks.len);
        binding.bound = true;
        if (self.pipeline_backend1) |stage1| {
            stage1.mirrorSlotStateBlocksFrom(self, slot_index) catch |err| {
                // Rollback: unbind self before propagating.
                self.slot_state_bindings[slot_index].reset();
                return err;
            };
        }
        if (self.pipeline_backend0_cpu) |stage0_cpu| {
            stage0_cpu.bindSlotStateBlocks(slot_index, state_blocks) catch |err| {
                // Rollback: unbind stage1 mirror and self before propagating.
                if (self.pipeline_backend1) |stage1| {
                    stage1.unbindSlotStateBlocks(slot_index);
                }
                self.slot_state_bindings[slot_index].reset();
                return err;
            };
        }
    }

    pub fn mirrorSlotStateBlocksFrom(
        self: *CudaBackend,
        source: *const CudaBackend,
        slot_index: usize,
    ) !void {
        if (!self.slotIndexSupported(slot_index) or !source.slotIndexSupported(slot_index)) return error.InvalidArgument;
        if (self.state_descriptor_count == 0) return;
        const source_binding = &source.slot_state_bindings[slot_index];

        var binding = &self.slot_state_bindings[slot_index];
        binding.reset();
        const source_blocks: []const runtime_contract.StateBlockHandle = if (source_binding.bound)
            source.slotStateBlocks(slot_index)
        else
            &.{};
        for (self.stateDescriptors(), 0..) |descriptor, idx| {
            if (descriptor.runtime_kind != runtime_contract.state_runtime_kind_none) {
                // Runtime descriptors are always rebound into stage-local wrapper blocks.
                // Rebinding shared source blocks mutates their payload and can alias state
                // across stages, so stage1 must never reuse stage0 runtime block pointers.
                binding.handles[idx] = try synthesizeRuntimeStateBlockForDescriptor(
                    self,
                    binding,
                    descriptor,
                    slot_index,
                    idx,
                );
                continue;
            }
            const incoming = runtime_contract.findStateBlock(source_blocks, descriptor.id) orelse {
                return error.InvalidStateDescriptorBinding;
            };
            binding.handles[idx] = .{
                .id = descriptor.id,
                .ptr = incoming.ptr,
                .size = incoming.size,
                .align_bytes = incoming.align_bytes,
            };
        }
        binding.count = self.state_descriptor_count;
        binding.bound = true;
    }

    pub fn unbindSlotStateBlocks(self: *CudaBackend, slot_index: usize) void {
        if (!self.slotIndexSupported(slot_index)) return;
        if (self.pipeline_backend1) |stage1| {
            stage1.unbindSlotStateBlocks(slot_index);
        }
        if (self.pipeline_backend0_cpu) |stage0_cpu| {
            stage0_cpu.unbindSlotStateBlocks(slot_index);
        }
        self.slot_state_bindings[slot_index].reset();
    }

    pub fn slotStateBlocks(self: *const CudaBackend, slot_index: usize) []const runtime_contract.StateBlockHandle {
        const binding = &self.slot_state_bindings[slot_index];
        return binding.handles[0..binding.count];
    }

    pub inline fn slotIndexSupported(self: *const CudaBackend, slot_index: usize) bool {
        return slot_index < self.max_batch_size;
    }

    pub fn ensureSlotStateBlocksBoundForScheduler(self: *CudaBackend, slot_index: usize) !void {
        if (!self.slotIndexSupported(slot_index)) return error.InvalidArgument;
        if (self.state_descriptor_count == 0) return;
        if (!self.slot_state_bindings[slot_index].bound) return error.InvalidStateDescriptorBinding;
        try runtime_contract.validateStateBlocksForDescriptors(
            self.stateDescriptors(),
            self.slotStateBlocks(slot_index),
        );
    }

    pub fn prefillSlot(
        self: *CudaBackend,
        slot_index: usize,
        tokens: []const u32,
        logits_out: []f32,
    ) !void {
        return prefill_mod.prefillSlot(self, slot_index, tokens, logits_out);
    }

    pub fn prefillSlotWithVision(
        self: *CudaBackend,
        slot_index: usize,
        tokens: []const u32,
        vision_input: ?*const PrefillVisionInput,
        logits_out: []f32,
    ) !void {
        if (vision_input == null) return self.prefillSlot(slot_index, tokens, logits_out);
        try self.ensureSlotStateBlocksBoundForScheduler(slot_index);

        if (tokens.len == 0) {
            log.warn("inference", "CUDA prefillSlotWithVision invalid args", .{
                .reason = "empty_tokens",
                .slot_index = slot_index,
            });
            return error.InvalidArgument;
        }
        if (logits_out.len != self.vocab_size) {
            log.warn("inference", "CUDA prefillSlotWithVision invalid args", .{
                .reason = "logits_len_mismatch",
                .slot_index = slot_index,
                .logits_len = logits_out.len,
                .vocab_size = self.vocab_size,
            });
            return error.InvalidArgument;
        }
        if (!self.slot_in_use[slot_index] or !self.slotIndexSupported(slot_index)) {
            log.warn("inference", "CUDA prefillSlotWithVision invalid args", .{
                .reason = "slot_state",
                .slot_index = slot_index,
                .slot_in_use = @as(u8, @intFromBool(self.slot_in_use[slot_index])),
            });
            return error.InvalidArgument;
        }
        if (tokens.len > self.max_seq_len) return error.InvalidArgument;

        const vision = if (self.vision_runtime) |*rt|
            rt
        else {
            log.warn("inference", "CUDA vision prefill requested but vision runtime is unavailable", .{
                .slot_index = slot_index,
                .tokens = tokens.len,
            });
            return error.UnsupportedContentType;
        };

        const vi = vision_input.?;
        var encoded_vision_output = vision.encodeImages(vi.images) catch |err| {
            log.warn("inference", "CUDA vision encode failed", .{
                .slot_index = slot_index,
                .reason = @errorName(err),
                .images = vi.images.len,
            });
            return err;
        };
        defer encoded_vision_output.deinit(self.allocator);

        var image_token_positions: []usize = &.{};
        defer if (image_token_positions.len > 0) self.allocator.free(image_token_positions);
        var deepstack_layer_features_opt: ?[]const []const f32 = null;
        if (encoded_vision_output.deepstack_layer_embeddings.len > 0) {
            image_token_positions = try collectTokenPositions(self.allocator, tokens, vi.image_token_id);
            if (image_token_positions.len == 0) return error.InvalidPromptImageTokens;
            if (deepstackLayersCompatibleWithPrompt(
                encoded_vision_output.deepstack_layer_embeddings,
                image_token_positions.len,
                self.d_model,
            )) {
                deepstack_layer_features_opt = encoded_vision_output.deepstack_layer_embeddings;
            } else {
                log.warn("inference", "CUDA vision deepstack disabled: invalid layer feature shapes", .{
                    .slot_index = slot_index,
                    .deepstack_layers = encoded_vision_output.deepstack_layer_embeddings.len,
                    .image_positions = image_token_positions.len,
                    .d_model = self.d_model,
                });
            }
        }

        self.slot_rope_position_deltas[slot_index] = 0;
        self.activateKvSlot(slot_index);
        self.beginPrefillDispatchWindow();
        const prefill_start_ns: i128 = std.time.nanoTimestamp();
        try engine_forward.ensureKvCapacity(self, tokens.len);

        const hidden_count = std.math.mul(usize, tokens.len, self.d_model) catch return error.InvalidArgument;
        const hidden_host = try self.allocator.alloc(f32, hidden_count);
        defer self.allocator.free(hidden_host);

        populatePrefillHiddenFromTokens(
            self.loaded,
            tokens,
            self.d_model,
            hidden_host,
            vi.image_token_id,
        ) catch |err| {
            log.warn("inference", "CUDA vision prefill hidden population failed", .{
                .slot_index = slot_index,
                .tokens = tokens.len,
                .reason = @errorName(err),
            });
            return err;
        };
        vision.scatterIntoHidden(
            hidden_host,
            tokens.len,
            self.d_model,
            tokens,
            vi.image_token_id,
            encoded_vision_output.merged_embeddings,
        ) catch |err| {
            log.warn("inference", "CUDA vision scatter into hidden failed", .{
                .slot_index = slot_index,
                .tokens = tokens.len,
                .reason = @errorName(err),
                .merged_embeddings = encoded_vision_output.merged_embeddings.len,
                .d_model = self.d_model,
            });
            return err;
        };

        var i: usize = 0;
        while (i < tokens.len) : (i += 1) {
            const row_start = std.math.mul(usize, i, self.d_model) catch return error.InvalidArgument;
            const hidden_override = hidden_host[row_start .. row_start + self.d_model];
            const download_logits = self.shouldDownloadPrefillLogitsImpl(i, tokens.len);
            const deepstack_feature_index = if (deepstack_layer_features_opt != null and image_token_positions.len > 0)
                findPositionIndex(image_token_positions, i)
            else
                null;
            engine_forward.computeGpuPrototypeLogitsWithLayerLimit(
                self,
                tokens[i],
                i,
                slot_index,
                if (download_logits) self.slotLogits(slot_index) else null,
                self.block_runtime.blocks.len,
                download_logits,
                download_logits,
                false,
                @intCast(i + 1),
                i,
                hidden_override,
                deepstack_layer_features_opt,
                deepstack_feature_index,
                false,
            ) catch |err| {
                log.warn("inference", "CUDA vision token prefill step failed", .{
                    .slot_index = slot_index,
                    .token_index = i,
                    .token_id = tokens[i],
                    .reason = @errorName(err),
                    .has_deepstack = @as(u8, @intFromBool(deepstack_layer_features_opt != null)),
                    .deepstack_feature_index = if (deepstack_feature_index) |idx| idx else std.math.maxInt(usize),
                });
                return err;
            };
        }

        const prefill_elapsed_ns: u64 = @intCast(std.time.nanoTimestamp() - prefill_start_ns);
        self.logPrefillTimingImpl("prefill_slot_vision", tokens.len, prefill_elapsed_ns);
        @memcpy(logits_out, self.slotLogits(slot_index));
        self.slot_positions[slot_index] = tokens.len;
    }

    fn linearWeightSupportsSequenceRows(self: *const CudaBackend, weight: *const LinearWeight) bool {
        return linearWeightSupportsSequenceRowsForKernels(
            weight,
            self.matmul_f16_function != null,
            self.matmul_bf16_function != null,
            self.gaffine_u4_matvec_function != null,
            self.gaffine_u8_matvec_function != null,
        );
    }

    pub fn linearWeightSupportsSequenceRowsForKernels(
        weight: *const LinearWeight,
        matmul_f16_available: bool,
        matmul_bf16_available: bool,
        gaffine_matvec_available: bool,
        gaffine_u8_matvec_available: bool,
    ) bool {
        return switch (weight.*) {
            .dense_f32 => true,
            .dense_u16 => |w| switch (w.dtype) {
                .f16 => matmul_f16_available,
                .bf16 => matmul_bf16_available,
            },
            .gaffine_u4 => gaffine_matvec_available,
            .gaffine_u8 => gaffine_u8_matvec_available,
        };
    }

    pub fn decodeBatch(
        self: *CudaBackend,
        requests: []const contract.DecodeRequest,
        results: []contract.DecodeResult,
    ) !void {
        return decode_mod.decodeBatch(self, requests, results);
    }

    pub fn selectNextTokenFromDeviceLogitsImpl(self: *CudaBackend) !u32 {
        return selectNextTokenFromDeviceLogits(self);
    }

    pub fn shouldDownloadPrefillLogitsImpl(self: *const CudaBackend, token_index: usize, token_count: usize) bool {
        _ = self;
        return shouldDownloadPrefillLogits(token_index, token_count);
    }

    pub fn beginPrefillDispatchWindow(self: *CudaBackend) void {
        @memcpy(self.prefill_dispatch_window_start[0..], self.layer_program_dispatch_total[0..]);
    }

    pub fn logPrefillTimingImpl(self: *const CudaBackend, mode: []const u8, token_count: usize, elapsed_ns: u64) void {
        logPrefillTiming(self, mode, token_count, elapsed_ns);
    }

    pub fn computeGpuPrototypeLogits(self: *CudaBackend, token: u32, position: usize, logits_out: []f32) !void {
        return engine_forward.computeGpuPrototypeLogitsWithLayerLimit(
            self,
            token,
            position,
            0,
            logits_out,
            self.block_runtime.blocks.len,
            true,
            true,
            true,
            1,
            position,
            null,
            null,
            null,
            false,
        );
    }

    // --- Delegation to engine_forward.zig ---
    pub fn computeGpuPrototypeLogitsWithLayerLimit(
        self: *CudaBackend,
        token: u32,
        position: usize,
        slot_index: usize,
        logits_out_opt: ?[]f32,
        layer_limit: usize,
        compute_logits: bool,
        download_logits: bool,
        ensure_kv_capacity: bool,
        trace_seq_len_u32: u32,
        trace_pos_offset: usize,
        hidden_override: ?[]const f32,
        deepstack_layer_features_opt: ?[]const []const f32,
        deepstack_feature_index_opt: ?usize,
        use_preloaded_input: bool,
    ) !void {
        return engine_forward.computeGpuPrototypeLogitsWithLayerLimit(self, token, position, slot_index, logits_out_opt, layer_limit, compute_logits, download_logits, ensure_kv_capacity, trace_seq_len_u32, trace_pos_offset, hidden_override, deepstack_layer_features_opt, deepstack_feature_index_opt, use_preloaded_input);
    }

    pub fn computeBatchedDecodeLogits(
        self: *CudaBackend,
        tokens: []const u32,
        slot_indices: []const usize,
        positions: []const usize,
    ) !void {
        return engine_forward.computeBatchedDecodeLogits(self, tokens, slot_indices, positions);
    }

    pub fn computeBatchedDecodeLogitsDeviceOnly(
        self: *CudaBackend,
        tokens: []const u32,
        slot_indices: []const usize,
        positions: []const usize,
    ) !void {
        return engine_forward.computeBatchedDecodeLogitsDeviceOnly(self, tokens, slot_indices, positions);
    }

    pub fn batchedHostLogitsRow(self: *CudaBackend, row_index: usize) ?[]f32 {
        const vocab = self.runtime_buffers.projected_vocab;
        if (vocab == 0) return null;
        const start = std.math.mul(usize, row_index, vocab) catch return null;
        const end = std.math.add(usize, start, vocab) catch return null;
        if (end > self.runtime_buffers.projected_logits_batch_host.len) return null;
        return self.runtime_buffers.projected_logits_batch_host[start..end];
    }

    pub fn computeGpuPrototypePrefillLogitsWithLayerLimit(
        self: *CudaBackend,
        tokens: []const u32,
        slot_index: usize,
        logits_out: []f32,
        layer_limit: usize,
    ) !void {
        return engine_forward.computeGpuPrototypePrefillLogitsWithLayerLimit(self, tokens, slot_index, logits_out, layer_limit);
    }

    pub fn ensureKvCapacity(self: *CudaBackend, required_tokens: usize) !void {
        return engine_forward.ensureKvCapacity(self, required_tokens);
    }

    pub fn tryExecuteLayerProgram(
        self: *CudaBackend,
        layer: *BlockRuntimeLayer,
        slot_index: usize,
        layer_index: usize,
        d_model_u32: u32,
        head_dim_u32: u32,
        rope_dim_u32: u32,
        n_heads_u32: u32,
        n_kv_heads_u32: u32,
        active_rows_u32: u32,
        seq_len_u32: u32,
        trace_seq_len_u32: u32,
        trace_pos_offset: usize,
        position: usize,
        position_u32: u32,
        global_rope_theta: f32,
        local_rope_theta: f32,
        rope_function: compute.cuda.Function,
        copy_function: compute.cuda.Function,
        cast_f32_to_f16_function: ?compute.cuda.Function,
        kv_write_f16_function: ?compute.cuda.Function,
        rope_store_f16_function: ?compute.cuda.Function,
        shortconv_step_function: compute.cuda.Function,
        attention_kernels: AttentionKernelSet,
        batch_info: ?*const BatchDecodeInfo,
    ) !compute.cuda.Buffer {
        return engine_layer_program.tryExecuteLayerProgram(self, layer, slot_index, layer_index, d_model_u32, head_dim_u32, rope_dim_u32, n_heads_u32, n_kv_heads_u32, active_rows_u32, seq_len_u32, trace_seq_len_u32, trace_pos_offset, position, position_u32, global_rope_theta, local_rope_theta, rope_function, copy_function, cast_f32_to_f16_function, kv_write_f16_function, rope_store_f16_function, shortconv_step_function, attention_kernels, batch_info);
    }

    pub fn runAttentionContext(
        self: *CudaBackend,
        cfg: *const LayerAttentionExecConfig,
        q_stage: *const compute.cuda.Buffer,
        context_stage: *compute.cuda.Buffer,
        k_cache: *const compute.cuda.Buffer,
        v_cache: *const compute.cuda.Buffer,
        kernels: AttentionKernelSet,
        seq_len_u32: u32,
        head_dim_u32: u32,
        kv_dim_u32: u32,
        kv_groups_u32: u32,
        rope_dim_u32: u32,
        position_u32: u32,
        theta: f32,
    ) !AttentionPath {
        return engine_layer_program.runAttentionContext(self, cfg, q_stage, context_stage, k_cache, v_cache, kernels, seq_len_u32, head_dim_u32, kv_dim_u32, kv_groups_u32, rope_dim_u32, position_u32, theta);
    }

    pub fn prefillDispatchDelta(self: *const CudaBackend, opcode: opcode_map.Opcode) u64 {
        return engine_layer_program.prefillDispatchDelta(self, opcode);
    }

    pub fn prefillDispatchTotal(self: *const CudaBackend) u64 {
        return engine_layer_program.prefillDispatchTotal(self);
    }

    const BatchDecodeInfo = engine_types.BatchDecodeInfo;

    pub const LayerProgramExecutionContext = struct {
        backend: *CudaBackend,
        layer: *BlockRuntimeLayer,
        slot_index: usize,
        layer_index: usize,
        op_index: usize,
        d_model_u32: u32,
        head_dim_u32: u32,
        rope_dim_u32: u32,
        n_heads_u32: u32,
        n_kv_heads_u32: u32,
        active_rows_u32: u32,
        seq_len_u32: u32,
        trace_seq_len_u32: u32,
        trace_pos_offset: usize,
        position: usize,
        position_u32: u32,
        global_rope_theta: f32,
        local_rope_theta: f32,
        rope_function: compute.cuda.Function,
        copy_function: compute.cuda.Function,
        cast_f32_to_f16_function: ?compute.cuda.Function,
        kv_write_f16_function: ?compute.cuda.Function,
        rope_store_f16_function: ?compute.cuda.Function,
        shortconv_step_function: compute.cuda.Function,
        attention_kernels: AttentionKernelSet,
        register_to_slot_map: []const u8,
        input_view: compute.cuda.Buffer,
        slot_buffers: []compute.cuda.Buffer,
        instruction_handles: []runtime_contract.TensorHandle,
        instruction_views: []runtime_contract.TensorViewDesc,
        batch_info: ?*const BatchDecodeInfo = null,
    };

    pub const BuiltLayerProgramHandles = struct {
        registers: []runtime_contract.TensorHandle,
        views: []const runtime_contract.TensorViewDesc,
    };

    pub const LayerProgramInstructionStateBlocks = struct {
        handles: [1]runtime_contract.StateBlockHandle = undefined,
        len: usize = 0,

        pub fn slice(self: *LayerProgramInstructionStateBlocks) []runtime_contract.StateBlockHandle {
            return self.handles[0..self.len];
        }
    };

    const layer_program_required_opcodes = [_]opcode_map.Opcode{
        .rmsnorm,
        .multihead_attention,
        .gated_delta_net,
        .shortconv,
        .swiglu,
        .residual_add,
    };

    pub const layer_program_adapter_table: runtime_contract.AdapterTable = blk: {
        var table: runtime_contract.AdapterTable = [_]?runtime_contract.KernelAdapterFn{null} ** 256;
        table[@intFromEnum(opcode_map.Opcode.rmsnorm)] = layerProgramNormRuntimeAdapter;
        table[@intFromEnum(opcode_map.Opcode.multihead_attention)] = layerProgramAttentionRuntimeAdapter;
        table[@intFromEnum(opcode_map.Opcode.gated_delta_net)] = layerProgramGatedDeltaRuntimeAdapter;
        table[@intFromEnum(opcode_map.Opcode.shortconv)] = layerProgramShortConvRuntimeAdapter;
        table[@intFromEnum(opcode_map.Opcode.swiglu)] = layerProgramSwiGluRuntimeAdapter;
        table[@intFromEnum(opcode_map.Opcode.residual_add)] = layerProgramResidualAddRuntimeAdapter;
        break :blk table;
    };

    pub const layer_program_adapter_capabilities: runtime_contract.AdapterCapabilities = blk: {
        var caps: runtime_contract.AdapterCapabilities = [_]runtime_contract.AdapterCapability{.{
            .supports_batch = false,
            .supports_graph_emit = false,
            .max_batch_size = 1,
        }} ** 256;
        for (layer_program_required_opcodes) |opcode| {
            caps[@intFromEnum(opcode)] = .{
                .supports_batch = false,
                .supports_graph_emit = false,
                .max_batch_size = 1,
            };
        }
        break :blk caps;
    };

    comptime {
        runtime_contract.assertAdapterTableCoverage(
            layer_program_adapter_table,
            layer_program_required_opcodes,
            "cuda.engine.layer_program_adapter_table",
        );
    }

    pub fn layerProgramExecutionState(ctx: *runtime_contract.ExecutionContext) !*LayerProgramExecutionContext {
        const raw_state = ctx.workspace.any orelse return error.InvalidDispatchState;
        return @ptrCast(@alignCast(raw_state));
    }

    pub fn traceShapeBsd(seq_len: u32, dim: u32) [4]u32 {
        return .{ 1, seq_len, dim, 0 };
    }

    pub fn traceTokenIndex(seq_len: u32) u32 {
        if (seq_len == 0) return 0;
        return seq_len - 1;
    }

    pub fn tracePositionForPoint(point: trace.TracePoint, pos_offset: usize, seq_len: u32) u32 {
        if (seq_len == 0) return 0;
        return switch (point) {
            .attn_q, .attn_k, .attn_v, .embed_pos => if (seq_len == 1)
                @intCast(@min(pos_offset, std.math.maxInt(u32)))
            else
                seq_len,
            .attn_qk, .attn_weights, .attn_out => if (seq_len == 1)
                @intCast(@min(pos_offset + 1, std.math.maxInt(u32)))
            else
                seq_len,
            else => seq_len,
        };
    }

    test "tracePositionForPoint prefill uses sequence length across points" {
        const pos_offset: usize = 13;
        const seq_len: u32 = 14;
        try std.testing.expectEqual(@as(u32, 14), tracePositionForPoint(.layer_attn_norm, pos_offset, seq_len));
        try std.testing.expectEqual(@as(u32, 14), tracePositionForPoint(.attn_q, pos_offset, seq_len));
        try std.testing.expectEqual(@as(u32, 14), tracePositionForPoint(.attn_out, pos_offset, seq_len));
    }

    test "tracePositionForPoint decode matches CPU trace semantics" {
        const pos_offset: usize = 13;
        const seq_len: u32 = 1;
        try std.testing.expectEqual(@as(u32, 1), tracePositionForPoint(.layer_attn_norm, pos_offset, seq_len));
        try std.testing.expectEqual(@as(u32, 13), tracePositionForPoint(.attn_q, pos_offset, seq_len));
        try std.testing.expectEqual(@as(u32, 14), tracePositionForPoint(.attn_out, pos_offset, seq_len));
        try std.testing.expectEqual(@as(u32, 1), tracePositionForPoint(.ffn_down, pos_offset, seq_len));
    }

    fn ensureParityPrefillBufferCapacity(self: *CudaBackend, buffer: *[]f32, elements: usize) !void {
        if (buffer.*.len >= elements) return;
        if (buffer.*.len > 0) self.allocator.free(buffer.*);
        buffer.* = try self.allocator.alloc(f32, elements);
    }

    pub fn beginParityPrefillCapture(self: *CudaBackend, seq_len: usize) !void {
        self.parity_prefill_seq_len = 0;
        self.parity_prefill_token_index = 0;
        @memset(&self.parity_checkpoint_warned, false);
        if (!trace.isEnabled() or seq_len == 0) return;
        const layer_count = self.block_runtime.blocks.len;
        const elements = std.math.mul(usize, layer_count, std.math.mul(usize, seq_len, self.d_model) catch return error.InvalidArgument) catch return error.InvalidArgument;
        try self.ensureParityPrefillBufferCapacity(&self.parity_prefill_layer_attn_norm_host, elements);
        try self.ensureParityPrefillBufferCapacity(&self.parity_prefill_layer_ffn_norm_host, elements);
        try self.ensureParityPrefillBufferCapacity(&self.parity_prefill_block_out_host, elements);
        self.parity_prefill_seq_len = seq_len;
    }

    pub fn endParityPrefillCapture(self: *CudaBackend) void {
        self.parity_prefill_seq_len = 0;
        self.parity_prefill_token_index = 0;
        @memset(&self.parity_checkpoint_warned, false);
    }

    pub fn parityPrefillBufferForPoint(self: *CudaBackend, point: trace.TracePoint) ?[]f32 {
        return switch (point) {
            .layer_attn_norm => self.parity_prefill_layer_attn_norm_host,
            .layer_ffn_norm => self.parity_prefill_layer_ffn_norm_host,
            .block_out => self.parity_prefill_block_out_host,
            else => null,
        };
    }

    pub fn ensureTraceCheckpointHostCapacity(self: *CudaBackend, elements: usize) !void {
        if (self.trace_checkpoint_host.len >= elements) return;
        if (self.trace_checkpoint_host.len > 0) self.allocator.free(self.trace_checkpoint_host);
        self.trace_checkpoint_host = try self.allocator.alloc(f32, elements);
    }

    fn layerProgramNormRuntimeAdapter(
        rt_ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        _: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        _: []const runtime_contract.ParamBlock,
    ) !void {
        const exec_ctx = try layerProgramExecutionState(rt_ctx);
        const io = try engine_layer_program.instructionIoSlices(insn, registers);
        if (io.inputs.len != 1 or io.outputs.len != 1) return error.InvalidInstructionBinding;
        _ = try runtime_contract.requireInstructionStateBlockForPlan(
            insn,
            &exec_ctx.layer.compiled_plan.?.plan,
            state_blocks,
        );
        const layer = try engine_layer_program.requireLayerProgramRuntimeState(exec_ctx, insn, state_blocks);
        try engine_layer_program.layerProgramNormAdapter(exec_ctx.backend, layer, insn, registers, exec_ctx);
        if (trace.isEnabled()) {
            engine_layer_program.emitLayerProgramTracePoint(
                exec_ctx,
                engine_layer_program.inferNormTracePoint(layer, exec_ctx.op_index),
                traceShapeBsd(exec_ctx.trace_seq_len_u32, exec_ctx.d_model_u32),
                3,
                "cuda_rmsnorm",
                engine_layer_program.bufferFromTensorHandle(io.outputs[0]),
            );
        }
    }

    fn layerProgramAttentionRuntimeAdapter(
        rt_ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        _: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        _: []const runtime_contract.ParamBlock,
    ) !void {
        const exec_ctx = try layerProgramExecutionState(rt_ctx);
        _ = try runtime_contract.requireInstructionStateBlockForPlan(
            insn,
            &exec_ctx.layer.compiled_plan.?.plan,
            state_blocks,
        );
        const layer = try engine_layer_program.requireLayerProgramRuntimeState(exec_ctx, insn, state_blocks);
        const trace_enabled = trace.isEnabled();
        const emits_traced_inside_cpu_fallback = if (trace_enabled) if (layer.instructionAttentionRef(exec_ctx.op_index)) |cfg| blk: {
            // Query-gated attention currently runs through the traced CPU fallback inside
            // the CUDA backend. Skipping the wrapper emit here avoids duplicate attn.q/out
            // rows with one synthetic metadata record and one real host-backed record.
            if (!cfg.query_gate) {
                engine_layer_program.emitLayerProgramTracePoint(
                    exec_ctx,
                    .attn_q,
                    traceShapeBsd(exec_ctx.trace_seq_len_u32, @intCast(cfg.q_dim)),
                    3,
                    "cuda_attention_q",
                    null,
                );
            }
            break :blk cfg.query_gate;
        } else |_| false else false;
        try engine_layer_program.layerProgramAttentionAdapter(exec_ctx.backend, layer, insn, registers, state_blocks, exec_ctx);
        const io = try engine_layer_program.instructionIoSlices(insn, registers);
        if (io.inputs.len != 1 or io.outputs.len != 1) return error.InvalidInstructionBinding;
        if (trace_enabled and !emits_traced_inside_cpu_fallback) {
            engine_layer_program.emitLayerProgramTracePoint(
                exec_ctx,
                .attn_out,
                traceShapeBsd(exec_ctx.trace_seq_len_u32, exec_ctx.d_model_u32),
                3,
                "cuda_attention_out",
                engine_layer_program.bufferFromTensorHandle(io.outputs[0]),
            );
        }
    }

    fn layerProgramShortConvRuntimeAdapter(
        rt_ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        _: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        _: []const runtime_contract.ParamBlock,
    ) !void {
        const exec_ctx = try layerProgramExecutionState(rt_ctx);
        _ = try runtime_contract.requireInstructionStateBlockForPlan(
            insn,
            &exec_ctx.layer.compiled_plan.?.plan,
            state_blocks,
        );
        const layer = try engine_layer_program.requireLayerProgramRuntimeState(exec_ctx, insn, state_blocks);
        if (trace.isEnabled()) {
            if (layer.instructionShortConvRef(exec_ctx.op_index)) |cfg| {
                engine_layer_program.emitLayerProgramTracePoint(
                    exec_ctx,
                    .conv_in_proj,
                    traceShapeBsd(exec_ctx.trace_seq_len_u32, @intCast(cfg.conv_dim * 3)),
                    3,
                    "cuda_shortconv_in_proj",
                    null,
                );
            } else |_| {}
        }
        try engine_layer_program.layerProgramShortConvAdapter(exec_ctx.backend, layer, insn, registers, state_blocks, exec_ctx);
        const io = try engine_layer_program.instructionIoSlices(insn, registers);
        if (io.inputs.len != 1 or io.outputs.len != 1) return error.InvalidInstructionBinding;
        if (trace.isEnabled()) {
            engine_layer_program.emitLayerProgramTracePoint(
                exec_ctx,
                .conv_out_proj,
                traceShapeBsd(exec_ctx.trace_seq_len_u32, exec_ctx.d_model_u32),
                3,
                "cuda_shortconv_out_proj",
                engine_layer_program.bufferFromTensorHandle(io.outputs[0]),
            );
        }
    }

    fn layerProgramGatedDeltaRuntimeAdapter(
        rt_ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        _: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        params: []const runtime_contract.ParamBlock,
    ) !void {
        const exec_ctx = try layerProgramExecutionState(rt_ctx);
        _ = try runtime_contract.requireInstructionStateBlockForPlan(
            insn,
            &exec_ctx.layer.compiled_plan.?.plan,
            state_blocks,
        );
        const layer = try engine_layer_program.requireLayerProgramRuntimeState(exec_ctx, insn, state_blocks);
        try engine_layer_program.layerProgramGatedDeltaAdapter(exec_ctx.backend, layer, insn, registers, state_blocks, params, exec_ctx);
    }

    fn layerProgramSwiGluRuntimeAdapter(
        rt_ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        _: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        _: []const runtime_contract.ParamBlock,
    ) !void {
        const exec_ctx = try layerProgramExecutionState(rt_ctx);
        _ = try runtime_contract.requireInstructionStateBlockForPlan(
            insn,
            &exec_ctx.layer.compiled_plan.?.plan,
            state_blocks,
        );
        const layer = try engine_layer_program.requireLayerProgramRuntimeState(exec_ctx, insn, state_blocks);
        const weight_handles = try engine_layer_program.instructionWeightSlice(insn, registers);
        if (trace.isEnabled() and weight_handles.len >= 2) {
            const gate = engine_layer_program.linearWeightFromWeightHandle(weight_handles[0]);
            engine_layer_program.emitLayerProgramTracePoint(
                exec_ctx,
                .ffn_gate,
                traceShapeBsd(exec_ctx.trace_seq_len_u32, @intCast(gate.cols())),
                3,
                "cuda_ffn_gate",
                null,
            );
        }
        try engine_layer_program.layerProgramSwiGluAdapter(exec_ctx.backend, layer, insn, registers, exec_ctx);
        const io = try engine_layer_program.instructionIoSlices(insn, registers);
        if (io.inputs.len != 1 or io.outputs.len != 1) return error.InvalidInstructionBinding;
        if (trace.isEnabled()) {
            engine_layer_program.emitLayerProgramTracePoint(
                exec_ctx,
                .ffn_down,
                traceShapeBsd(exec_ctx.trace_seq_len_u32, exec_ctx.d_model_u32),
                3,
                "cuda_ffn_down",
                engine_layer_program.bufferFromTensorHandle(io.outputs[0]),
            );
        }
    }

    fn layerProgramResidualAddRuntimeAdapter(
        rt_ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        _: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        params: []const runtime_contract.ParamBlock,
    ) !void {
        const exec_ctx = try layerProgramExecutionState(rt_ctx);
        _ = try runtime_contract.requireInstructionStateBlockForPlan(
            insn,
            &exec_ctx.layer.compiled_plan.?.plan,
            state_blocks,
        );
        _ = try engine_layer_program.requireLayerProgramRuntimeState(exec_ctx, insn, state_blocks);
        const scale = try engine_layer_program.decodeResidualScaleFromParams(params);
        try engine_layer_program.layerProgramResidualAddAdapter(exec_ctx.backend, insn, registers, scale, exec_ctx);
        const io = try engine_layer_program.instructionIoSlices(insn, registers);
        if (io.inputs.len != 2 or io.outputs.len != 1) return error.InvalidInstructionBinding;
        if (trace.isEnabled()) {
            engine_layer_program.emitLayerProgramTracePoint(
                exec_ctx,
                .block_out,
                traceShapeBsd(exec_ctx.trace_seq_len_u32, exec_ctx.d_model_u32),
                3,
                "cuda_residual_add",
                engine_layer_program.bufferFromTensorHandle(io.outputs[0]),
            );
        }
    }
};

// --- Free functions from engine_weights.zig ---
const engine_weights = @import("engine_weights.zig");
const argmaxHost = engine_weights.argmaxHost;
const argminHost = engine_weights.argminHost;
const bytesToMiB = engine_weights.bytesToMiB;
const populatePrefillHiddenFromTokens = engine_weights.populatePrefillHiddenFromTokens;
const selectNextTokenFromLogits = engine_weights.selectNextTokenFromLogits;
const selectNextTokenFromDeviceLogits = engine_weights.selectNextTokenFromDeviceLogits;
const bufferSlice = engine_weights.bufferSlice;
const DeviceKvPair = engine_weights.DeviceKvPair;
const kvCacheElementBytes = engine_weights.kvCacheElementBytes;
const kvCacheBytesForCapacity = engine_weights.kvCacheBytesForCapacity;
const allocDeviceKvPair = engine_weights.allocDeviceKvPair;
const resizeScratchBuffer = engine_weights.resizeScratchBuffer;
const freeOwnedTensorView = engine_weights.freeOwnedTensorView;
const shouldDownloadPrefillLogits = engine_weights.shouldDownloadPrefillLogits;
const logPrefillTiming = engine_weights.logPrefillTiming;
const deepstackLayersCompatibleWithPrompt = engine_weights.deepstackLayersCompatibleWithPrompt;
const collectTokenPositions = engine_weights.collectTokenPositions;
const findPositionIndex = engine_weights.findPositionIndex;
const DenseLinearLayout = engine_weights.DenseLinearLayout;
const resolveDenseInOutLayout = engine_weights.resolveDenseInOutLayout;
const resolveDenseOutInLayout = engine_weights.resolveDenseOutInLayout;
const transposeRowMajor = engine_weights.transposeRowMajor;
const uploadTensor = engine_weights.uploadTensor;
const uploadLinearWeight = engine_weights.uploadLinearWeight;
const uploadLinearWeightWithContext = engine_weights.uploadLinearWeightWithContext;
const DenseOutInU16 = engine_weights.DenseOutInU16;
const DenseOutInF32 = engine_weights.DenseOutInF32;
const FusedQkvUpload = engine_weights.FusedQkvUpload;
const FusedGateUpUpload = engine_weights.FusedGateUpUpload;
const materializeDenseOutInU16 = engine_weights.materializeDenseOutInU16;
const materializeDenseOutInF32 = engine_weights.materializeDenseOutInF32;
const uploadFusedQkvWeights = engine_weights.uploadFusedQkvWeights;
const uploadFusedGateUpWeights = engine_weights.uploadFusedGateUpWeights;
const uploadLinearWeightDense = engine_weights.uploadLinearWeightDense;
const uploadLinearWeightDenseU16 = engine_weights.uploadLinearWeightDenseU16;
const uploadLinearWeightGroupedAffineU4 = engine_weights.uploadLinearWeightGroupedAffineU4;
const uploadLinearWeightGroupedAffineU8 = engine_weights.uploadLinearWeightGroupedAffineU8;
const uploadVectorTensor = engine_weights.uploadVectorTensor;
const allocZeroedF32Buffer = engine_weights.allocZeroedF32Buffer;
const tryUploadEmbeddingLookup = engine_weights.tryUploadEmbeddingLookup;
const uploadShortConvWeightTimeMajor = engine_weights.uploadShortConvWeightTimeMajor;
const materializeTensorF32 = engine_weights.materializeTensorF32;
const canUseModelEmbeddings = engine_weights.canUseModelEmbeddings;
const tryPopulateHiddenFromToken = engine_weights.tryPopulateHiddenFromToken;
const decodeGaffineRow = engine_weights.decodeGaffineRow;
const tryPopulateFinalNormWeight = engine_weights.tryPopulateFinalNormWeight;
const tryPopulateProjectionFromWeight = engine_weights.tryPopulateProjectionFromWeight;
const gaffineScaleBiasToF32 = engine_weights.gaffineScaleBiasToF32;
const gaffineValueAt = engine_weights.gaffineValueAt;

test {
    _ = @import("engine_tests.zig");
}
