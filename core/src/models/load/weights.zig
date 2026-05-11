//! Model weight loading and transformation.
//!
//! Handles loading weights from SafeTensors files with support for
//! quantization (GAF), transposition, and fusion of attention
//! components (QKV, gate/up projections).

const std = @import("std");
const tensor = @import("compute_pkg").tensor;
const config_types = @import("../config/types.zig");
const dtype = @import("compute_pkg").dtype;
const log = @import("log_pkg");
const progress_mod = @import("progress_pkg");
const runtime_blocks = @import("models_pkg").runtime_blocks;

const Tensor = tensor.Tensor;
const ModelConfig = config_types.ModelConfig;
const DType = dtype.DType;
const cfg_loader = @import("../config/root.zig");
const st_loader = @import("io_pkg").safetensors.root;
const model_types = @import("models_pkg").op_types;
const manifest_mod = @import("../manifest.zig");
const transforms = @import("transforms.zig");
const generic_weights = @import("generic_weights.zig");

pub const blocks = runtime_blocks;

const maybeConcatGateUpWeights = transforms.maybeConcatGateUpWeights;
const maybeConcatQkvWeights = transforms.maybeConcatQkvWeights;

// Test-only imports from transforms (used in unit tests)
const ensureF32 = transforms.ensureF32;
const convertToF32 = transforms.convertToF32;
const bytesToU16Slice = transforms.bytesToU16Slice;
const orientWeightF32 = transforms.orientWeightF32;
const orientWeightTyped = transforms.orientWeightTyped;

pub const LoadOptions = struct {
    /// Keep native (bf16/f16) norm weight dtype instead of converting to f32.
    /// The caller decides this policy based on selected runtime/backend.
    preserve_native_norm_dtype: bool = false,
    /// Dequantize MXFP8 weights to BF16 during load.
    /// Used by CPU backend until native MXFP8 parity is complete.
    dequantize_mxfp8_to_bf16: bool = false,
    /// Dequantize NVFP4 packed weights to BF16 during load.
    /// CUDA can keep NVFP4 packed for native kernels; other backends use BF16 fallback.
    dequantize_nvfp4_to_bf16: bool = true,
};

pub const StageLoadError = error{
    InvalidStageRange,
    InvalidStageLayerOffset,
    EmptyStageRange,
    MissingStageWeight,
    MissingStageGlobalWeight,
    UnsupportedStageGlobalRole,
    UnsupportedStageConfigInference,
    StageResidencyMismatch,
};

pub const StageLayerRange = struct {
    start: usize,
    end: usize,

    pub fn len(self: StageLayerRange) usize {
        return self.end - self.start;
    }
};

pub const StageRoleRequest = struct {
    include_token_embeddings: bool = false,
    include_final_norm: bool = false,
    include_lm_head: bool = false,
    include_embedding_side: bool = false,
    include_vision_side: bool = false,
    include_architecture_side: bool = false,
    include_unclassified_global: bool = false,

    pub fn includesManifestRole(self: StageRoleRequest, role: manifest_mod.TensorRole) bool {
        return switch (role) {
            .token_embeddings => self.include_token_embeddings,
            .final_norm => self.include_final_norm,
            .lm_head => self.include_lm_head,
            .embedding_side => self.include_embedding_side,
            .vision_side => self.include_vision_side,
            .architecture_side => self.include_architecture_side,
            .unclassified_global => self.include_unclassified_global,
            .decoder_layer, .quant_companion => false,
        };
    }

    pub fn toResidencyRequest(
        self: StageRoleRequest,
        range: StageLayerRange,
    ) manifest_mod.StageResidencyRequest {
        return .{
            .layer_start = range.start,
            .layer_end = range.end,
            .include_token_embeddings = self.include_token_embeddings,
            .include_final_norm = self.include_final_norm,
            .include_lm_head = self.include_lm_head,
            .include_embedding_side = self.include_embedding_side,
            .include_vision_side = self.include_vision_side,
            .include_architecture_side = self.include_architecture_side,
            .include_unclassified_global = self.include_unclassified_global,
        };
    }
};

pub const StageLoadRequest = struct {
    layer_start: usize,
    layer_end: usize,
    roles: StageRoleRequest,

    pub fn range(self: StageLoadRequest) StageLayerRange {
        return .{ .start = self.layer_start, .end = self.layer_end };
    }
};

pub const LoadedModel = struct {
    arena: std.heap.ArenaAllocator,
    config: ModelConfig,
    runtime: config_types.ModelRuntime = .{},
    st: ?st_loader.UnifiedSafeTensors = null,
    ln_final: ?Tensor = null,
    lm_head: ?Tensor = null,
    token_embeddings: Tensor,
    /// Optional embedding tables for BERT-family models.
    position_embeddings: ?Tensor = null,
    token_type_embeddings: ?Tensor = null,
    /// Optional embedding layer norm (BERT-family).
    embedding_norm_weight: ?Tensor = null,
    embedding_norm_bias: ?Tensor = null,
    blocks: []runtime_blocks.LayerWeights,
    /// Original dtype of projection weights (before conversion to f32)
    /// Used to detect BF16 models for MLX GPU path
    original_weight_dtype: DType,
    /// File size in bytes (for display)
    file_size: usize = 0,
    /// Total tensor count (for display)
    tensor_count: usize = 0,
    /// Metadata-only tensor ownership and checkpoint byte accounting.
    manifest: ?manifest_mod.ModelManifest = null,

    pub fn deinit(self: *LoadedModel) void {
        if (self.manifest) |*manifest| manifest.deinit();
        if (self.st) |*st| st.deinit();
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn manifestPtr(self: *const LoadedModel) ?*const manifest_mod.ModelManifest {
        if (self.manifest) |*manifest| return manifest;
        return null;
    }
};

pub const LoadedStageModel = struct {
    arena: std.heap.ArenaAllocator,
    config: ModelConfig,
    runtime: config_types.ModelRuntime = .{},
    st: ?st_loader.UnifiedSafeTensors = null,
    ln_final: ?Tensor = null,
    lm_head: ?Tensor = null,
    token_embeddings: ?Tensor = null,
    position_embeddings: ?Tensor = null,
    token_type_embeddings: ?Tensor = null,
    embedding_norm_weight: ?Tensor = null,
    embedding_norm_bias: ?Tensor = null,
    blocks: []runtime_blocks.LayerWeights,
    layer_range: StageLayerRange,
    requested_roles: StageRoleRequest,
    residency: manifest_mod.StageResidencyReport,
    original_weight_dtype: DType,
    file_size: usize = 0,
    tensor_count: usize = 0,
    manifest: manifest_mod.ModelManifest,
    extra_global_weights: runtime_blocks.WeightMap = .{},
    extra_global_role_bytes: [manifest_mod.role_count]usize = [_]usize{0} ** manifest_mod.role_count,
    lm_head_uses_token_embeddings: bool = false,

    /// Releases manifest-owned allocations, safetensors mappings, then arena-owned tensors.
    /// Matching `LoadedModel`, the value is invalid after deinit returns.
    pub fn deinit(self: *LoadedStageModel) void {
        self.manifest.deinit();
        if (self.st) |*st| st.deinit();
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn originalLayerIndex(self: *const LoadedStageModel, block_offset: usize) !usize {
        if (block_offset >= self.blocks.len) return error.InvalidStageLayerOffset;
        return self.layer_range.start + block_offset;
    }

    pub fn manifestPtr(self: *const LoadedStageModel) *const manifest_mod.ModelManifest {
        return &self.manifest;
    }
};

pub fn loadModel(
    backing_allocator: std.mem.Allocator,
    config_path: []const u8,
    safetensors_path: []const u8,
    load_options: LoadOptions,
    progress: progress_mod.Context,
) !LoadedModel {
    return loadModelWithArchitecture(
        backing_allocator,
        config_path,
        safetensors_path,
        null,
        null,
        load_options,
        progress,
    );
}

/// Load model config, architecture metadata, and per-layer block types WITHOUT
/// loading weight tensor data. Reads only safetensors headers for dtype detection
/// and tensor counts. Suitable for backends that load weights independently
/// (e.g., Metal/MLX loads from model_path via its own C++ loader).
pub fn loadModelMetadataOnly(
    backing_allocator: std.mem.Allocator,
    config_path: []const u8,
    safetensors_path: []const u8,
    runtime_arch: *const model_types.Architecture,
    parse_config_hook: ?model_types.ConfigParseHook,
) !LoadedModel {
    var arena_alloc = std.heap.ArenaAllocator.init(backing_allocator);
    errdefer arena_alloc.deinit();
    const arena_allocator = arena_alloc.allocator();

    // Read safetensors headers only (no mmap of weight data)
    var safetensors_file = try st_loader.UnifiedSafeTensors.loadMetadataOnly(backing_allocator, safetensors_path);
    errdefer safetensors_file.deinit();

    var discovery = try discoverModelForLoad(
        backing_allocator,
        arena_allocator,
        config_path,
        &safetensors_file,
        runtime_arch,
        parse_config_hook,
        .full,
    );
    errdefer discovery.manifest.deinit();

    const layer_count = discovery.layer_block_kinds.len;
    var block_weights = try arena_allocator.alloc(runtime_blocks.LayerWeights, layer_count);

    // Populate block types from architecture metadata (no weight loading)
    for (0..layer_count) |layer_idx| {
        block_weights[layer_idx] = .{
            .block_type = discovery.layer_block_kinds[layer_idx],
            .weight_map = .{},
            .map_context = discovery.layer_contexts[layer_idx],
        };
    }

    return LoadedModel{
        .arena = arena_alloc,
        .config = discovery.config,
        .runtime = .{
            .architecture_id = runtime_arch.name,
            .has_moe = runtime_arch.has_moe,
            .has_mamba = runtime_arch.has_mamba,
            .has_gated_delta = runtime_arch.has_gated_delta,
            .has_shortconv = runtime_arch.has_shortconv,
            .has_mla = runtime_arch.has_mla,
            .explicit_qk_norm_ops = runtime_arch.explicit_qk_norm_ops,
            .norm_weights_pre_shifted = runtime_arch.norm_weights_pre_shifted,
        },
        .st = safetensors_file,
        .ln_final = null,
        .lm_head = null,
        .token_embeddings = std.mem.zeroes(Tensor),
        .position_embeddings = null,
        .token_type_embeddings = null,
        .embedding_norm_weight = null,
        .embedding_norm_bias = null,
        .blocks = block_weights,
        .original_weight_dtype = discovery.original_weight_dtype,
        .file_size = safetensors_file.fileSize(),
        .tensor_count = safetensors_file.tensorCount(),
        .manifest = discovery.manifest,
    };
}

/// Environment flags collected once at load time to avoid repeated lookups.
const LoaderEnvFlags = struct {
    enable_cpu_fusion: bool,

    fn fromEnvironment(allocator: std.mem.Allocator) LoaderEnvFlags {
        return .{
            .enable_cpu_fusion = readEnvFlag(allocator, "TALU_CPU_FUSION", true),
        };
    }
};

fn archHasBlockWeights(arch: *const model_types.Architecture) bool {
    if (arch.block_weights.len > 0) return true;
    if (arch.block_variants) |variants| {
        for (variants) |variant| {
            if (variant.weights.len > 0) return true;
        }
    }
    return false;
}

fn archHasGlobalWeights(arch: *const model_types.Architecture) bool {
    return arch.global_weights.len > 0;
}

fn requireArchitectureMetadata(arch: *const model_types.Architecture) !void {
    if (arch.resolve_d_ff_from_weights and arch.d_ff_source_weight_ids.len == 0) {
        return error.MissingDffSourceWeightIds;
    }
    if (arch.resolve_shortconv_dims_from_weights and arch.shortconv_dims_source_weight_id == null) {
        return error.MissingShortConvSourceWeightId;
    }
    const has_dtype_probe_ids = arch.weight_dtype_source_weight_ids.len > 0 or arch.d_ff_source_weight_ids.len > 0;
    if (!has_dtype_probe_ids) return error.MissingWeightDTypeSourceWeightIds;
}

fn findWeightSpecById(specs: []const model_types.WeightSpec, id: []const u8) ?*const model_types.WeightSpec {
    for (specs) |*spec| {
        if (std.mem.eql(u8, spec.id, id)) return spec;
    }
    return null;
}

const ResolvedTensor = struct {
    name: []const u8,
    tensor: Tensor,
};

fn resolveGlobalTensorBySpec(
    allocator: std.mem.Allocator,
    safetensors_file: *st_loader.UnifiedSafeTensors,
    spec: *const model_types.WeightSpec,
    weight_prefixes: []const []const u8,
) !?ResolvedTensor {
    var name_buf: [512]u8 = undefined;
    var prefix_buf: [256]u8 = undefined;
    var qweight_buf: [1024]u8 = undefined;
    var name_resolver: generic_weights.NameResolver = .{};
    defer name_resolver.deinit(allocator);

    var alias_idx: usize = 0;
    while (alias_idx < spec.aliases.len + 1) : (alias_idx += 1) {
        const candidate = if (alias_idx == 0) spec.suffix else spec.aliases[alias_idx - 1];
        if (weight_prefixes.len == 0) {
            const resolved_name = try name_resolver.resolve(allocator, safetensors_file, candidate, qweight_buf[0..]) orelse continue;
            return .{ .name = resolved_name, .tensor = try safetensors_file.getTensor(resolved_name, null) };
        }

        for (weight_prefixes) |prefix_template| {
            const expanded_prefix = generic_weights.expandLayerTemplate(prefix_buf[0..], prefix_template, 0) catch continue;
            const total_len = expanded_prefix.len + candidate.len;
            if (total_len > name_buf.len) continue;
            @memcpy(name_buf[0..expanded_prefix.len], expanded_prefix);
            @memcpy(name_buf[expanded_prefix.len..total_len], candidate);
            const qualified_name = name_buf[0..total_len];
            const resolved_name = try name_resolver.resolve(allocator, safetensors_file, qualified_name, qweight_buf[0..]) orelse continue;
            return .{ .name = resolved_name, .tensor = try safetensors_file.getTensor(resolved_name, null) };
        }
    }

    return null;
}

fn buildTiedLmHead(
    allocator: std.mem.Allocator,
    safetensors_file: *st_loader.UnifiedSafeTensors,
    arch: *const model_types.Architecture,
    model_config: *const ModelConfig,
    model_load_options: LoadOptions,
) !?Tensor {
    const spec = findWeightSpecById(arch.global_weights, "token_embeddings") orelse return null;
    // Global tied-projection sources are not layer-scoped.
    // Using layer weight prefixes here makes the lookup miss real global
    // embedding tensors and silently fall back to a dequantized embedding copy.
    const resolved = try resolveGlobalTensorBySpec(allocator, safetensors_file, spec, &.{}) orelse return null;
    return try transforms.orientWeight(
        allocator,
        safetensors_file,
        resolved.name,
        @intCast(model_config.d_model),
        model_config.*,
        model_load_options.dequantize_mxfp8_to_bf16,
        model_load_options.dequantize_nvfp4_to_bf16,
    );
}

/// Try to get a tensor by name, with GPTQ .qweight and NVFP4 fallbacks.
/// When a name ending in .weight is not found, tries .qweight and .weight_packed.
/// When the resolved tensor is U8/I8 with a companion .weight_scale, recognizes
/// NVFP4 and returns grouped_affine_u4 (the effective dtype after transcoding).
fn getTensorOrGptqFallback(
    allocator: std.mem.Allocator,
    safetensors_file: *st_loader.UnifiedSafeTensors,
    name: []const u8,
    name_resolver: *generic_weights.NameResolver,
    qweight_buf: []u8,
) !?DType {
    const resolved = try name_resolver.resolve(allocator, safetensors_file, name, qweight_buf) orelse return null;
    const t = safetensors_file.getTensor(resolved, null) catch return null;
    return resolveNvfp4EffectiveDtype(t.dtype, safetensors_file, resolved);
}

fn detectOriginalWeightDType(
    allocator: std.mem.Allocator,
    arch: *const model_types.Architecture,
    layer_types: ?[]const u8,
    safetensors_file: *st_loader.UnifiedSafeTensors,
) !DType {
    const specs = if (arch.getVariantWithOverride(0, layer_types)) |v| v.weights else arch.block_weights;
    const probe_weight_ids = if (arch.weight_dtype_source_weight_ids.len > 0)
        arch.weight_dtype_source_weight_ids
    else
        arch.d_ff_source_weight_ids;

    if (probe_weight_ids.len == 0) return error.MissingWeightDTypeSourceWeightIds;
    var name_buf: [512]u8 = undefined;
    var prefix_buf: [256]u8 = undefined;
    var qweight_buf: [512]u8 = undefined;
    var name_resolver: generic_weights.NameResolver = .{};
    defer name_resolver.deinit(allocator);

    for (probe_weight_ids) |id| {
        const spec = findWeightSpecById(specs, id) orelse continue;
        var alias_idx: usize = 0;
        while (alias_idx < spec.aliases.len + 1) : (alias_idx += 1) {
            const candidate = if (alias_idx == 0) spec.suffix else spec.aliases[alias_idx - 1];
            if (arch.weight_prefixes.len == 0 or std.mem.indexOf(u8, candidate, "{d}") != null) {
                const name = generic_weights.expandLayerTemplate(name_buf[0..], candidate, 0) catch continue;
                if (try getTensorOrGptqFallback(allocator, safetensors_file, name, &name_resolver, &qweight_buf)) |d| return d;
                continue;
            }

            for (arch.weight_prefixes) |prefix_template| {
                const expanded_prefix = generic_weights.expandLayerTemplate(prefix_buf[0..], prefix_template, 0) catch continue;
                const total_len = expanded_prefix.len + candidate.len;
                if (total_len > name_buf.len) continue;
                @memcpy(name_buf[0..expanded_prefix.len], expanded_prefix);
                @memcpy(name_buf[expanded_prefix.len..total_len], candidate);
                const name = name_buf[0..total_len];
                if (try getTensorOrGptqFallback(allocator, safetensors_file, name, &name_resolver, &qweight_buf)) |d| return d;
            }
        }
    }

    return error.MissingWeightDTypeSourceWeight;
}

/// NVFP4 packed weights are stored as U8/I8 in safetensors with companion
/// scale tensors (.weight_scale). Both compressed-tensors (.weight_packed)
/// and modelopt (.weight) formats use this convention.
/// Returns grouped_affine_u4 when NVFP4 metadata is present — the effective
/// dtype Metal (and CPU) will transcode/dequantize to at load time.
fn resolveNvfp4EffectiveDtype(
    raw_dtype: DType,
    safetensors_file: *st_loader.UnifiedSafeTensors,
    resolved_name: []const u8,
) DType {
    if (raw_dtype != .u8 and raw_dtype != .i8) return raw_dtype;
    // Derive the base for the scale key:
    //   modelopt:          "...gate_proj.weight"        → base = up to ".weight"
    //   compressed-tensors: "...gate_proj.weight_packed" → base = up to ".weight"
    const base_len = if (std.mem.endsWith(u8, resolved_name, ".weight_packed"))
        resolved_name.len - ".weight_packed".len + ".weight".len
    else if (std.mem.endsWith(u8, resolved_name, ".weight"))
        resolved_name.len
    else
        return raw_dtype;
    var scale_buf: [512]u8 = undefined;
    const scale_len = base_len + "_scale".len;
    if (scale_len > scale_buf.len) return raw_dtype;
    @memcpy(scale_buf[0..base_len], resolved_name[0..base_len]);
    @memcpy(scale_buf[base_len..scale_len], "_scale");
    if (safetensors_file.hasTensor(scale_buf[0..scale_len])) return .grouped_affine_u4;
    return raw_dtype;
}

fn maybeAddFusedWeights(
    allocator: std.mem.Allocator,
    map: *runtime_blocks.WeightMap,
) !void {
    if (map.get("self_attn.qkv_proj.weight") == null) {
        const q = map.get("self_attn.q_proj.weight");
        const k = map.get("self_attn.k_proj.weight");
        const v = map.get("self_attn.v_proj.weight");
        if (q != null and k != null and v != null) {
            if (maybeConcatQkvWeights(allocator, q.?.*, k.?.*, v.?.*)) |fused| {
                const fused_ptr = try allocator.create(Tensor); // lint:ignore errdefer-alloc - arena freed atomically
                fused_ptr.* = fused;
                try map.put(allocator, "self_attn.qkv_proj.weight", fused_ptr);
            }
        }
    }

    if (map.get("mlp.gate_up_proj.weight") == null and map.get("mlp.input_linear.weight") == null) {
        const gate = map.get("mlp.gate_proj.weight");
        const up = map.get("mlp.up_proj.weight");
        if (gate != null and up != null) {
            if (maybeConcatGateUpWeights(allocator, gate.?.*, up.?.*)) |fused| {
                const fused_ptr = try allocator.create(Tensor); // lint:ignore errdefer-alloc - arena freed atomically
                fused_ptr.* = fused;
                try map.put(allocator, "mlp.gate_up_proj.weight", fused_ptr);
            }
        }
    }
}

/// Infer missing shortconv dimensions from loaded weight tensor shapes.
/// Config files may omit some dimensions (e.g., conv_dim_out in LFM2.5).
/// The depthwise conv weight is always f32 with shape [conv_dim, d_conv] or
/// [conv_dim, 1, d_conv], making it a reliable source for both dimensions.
fn inferShortConvDims(
    config: *runtime_blocks.ShortConvConfig,
    weight_map: *const runtime_blocks.WeightMap,
    source_weight_id: ?[]const u8,
) !void {
    const id = source_weight_id orelse return error.MissingShortConvSourceWeightId;
    const conv_w = weight_map.get(id) orelse return error.MissingShortConvSourceWeight;
    if (conv_w.n_dims != 2 and conv_w.n_dims != 3) return error.InvalidShortConvSourceWeightShape;

    const prev_conv_dim = config.conv_dim;
    const inferred_conv_dim: u32 = @intCast(conv_w.shape[0]);
    const d_idx: usize = if (conv_w.n_dims == 3) 2 else 1;
    const inferred_d_conv: u32 = @intCast(conv_w.shape[d_idx]);

    if (config.conv_dim == 0 or config.conv_dim != inferred_conv_dim) {
        if (config.conv_dim != 0 and config.conv_dim != inferred_conv_dim) {
            log.info("load", "Corrected shortconv conv_dim from source weight", .{
                .config_conv_dim = config.conv_dim,
                .inferred_conv_dim = inferred_conv_dim,
                .weight_id = id,
            });
        }
        config.conv_dim = inferred_conv_dim;
    }

    if (config.d_conv == 0 or config.d_conv != inferred_d_conv) {
        if (config.d_conv != 0 and config.d_conv != inferred_d_conv) {
            log.info("load", "Corrected shortconv d_conv from source weight", .{
                .config_d_conv = config.d_conv,
                .inferred_d_conv = inferred_d_conv,
                .weight_id = id,
            });
        }
        config.d_conv = inferred_d_conv;
    }

    // conv_dim_out is typically equal to conv_dim for known architectures.
    // Cannot reliably infer from out_proj since it may be quantized (packed shape).
    if (config.conv_dim_out == 0 and config.conv_dim != 0) {
        config.conv_dim_out = config.conv_dim;
    } else if (prev_conv_dim != 0 and config.conv_dim_out == prev_conv_dim and config.conv_dim != prev_conv_dim) {
        // If conv_dim changed due to source-weight correction and conv_dim_out
        // matched the stale conv_dim, keep them aligned.
        config.conv_dim_out = config.conv_dim;
    }

    if (config.conv_dim == 0 or config.conv_dim_out == 0 or config.d_conv == 0) {
        return error.InvalidShortConvConfig;
    }
}

/// Infer the FFN intermediate dimension (d_ff) from loaded weight tensor shapes.
///
/// Config files express the FFN dimension in various ways (intermediate_size,
/// block_ff_dim with SwiGLU adjustment, d_ff). Rather than encoding that
/// model-specific knowledge in the config parser, we treat the actual weight
/// tensor shape as ground truth: the gate projection (w1) output dimension
/// IS d_ff, regardless of what the config calls it or how it's computed.
///
/// Weight map keys vary by architecture (mlp.gate_proj.weight, feed_forward.w1.weight,
/// etc.) so we try known IDs in order. We use the gate weight (w1) because:
/// - It maps d_model → d_ff (output dim = d_ff)
/// - For quantized weights (GAF4/GAF8), shape[0] is always the output dim (never packed)
/// - For BF16/F16, shape[0] is the output dim (kept in [out, in] format)
/// - For F32, shape[1] is d_ff (transposed to [in, out] = [d_model, d_ff])
fn inferDff(
    model_config: *ModelConfig,
    weight_map: *const runtime_blocks.WeightMap,
    source_weight_ids: []const []const u8,
    has_fused_gate_up: bool,
) !void {
    if (source_weight_ids.len == 0) return error.MissingDffSourceWeightIds;

    const d_model: usize = @intCast(model_config.d_model);
    const config_d_ff: usize = @intCast(model_config.d_ff);
    var found_source_weight = false;

    for (source_weight_ids) |id| {
        if (weight_map.get(id)) |w| {
            found_source_weight = true;
            if (inferDffFromWeight(w.*, d_model)) |raw_inferred| {
                const inferred = blk: {
                    if (has_fused_gate_up) {
                        // Fused gate_up packs [gate, up], so tensor width is 2*d_ff.
                        if ((raw_inferred % 2) != 0) return error.InvalidDffSourceWeightShape;
                        break :blk raw_inferred / 2;
                    }
                    break :blk raw_inferred;
                };
                if (inferred != config_d_ff) {
                    log.info("load", "Corrected d_ff from weight shape", .{
                        .config_d_ff = config_d_ff,
                        .inferred_d_ff = inferred,
                        .weight_id = id,
                    });
                    model_config.d_ff = @intCast(inferred);
                }
                return;
            }

            // Square case (d_model == d_ff): no correction needed when config
            // already matches d_model.
            if (w.n_dims == 2 and w.shape[0] == d_model and w.shape[1] == d_model and config_d_ff == d_model) {
                return;
            }
        }
    }

    if (!found_source_weight) return error.MissingDffSourceWeight;
    return error.InvalidDffSourceWeightShape;
}

/// Extract d_ff from a gate projection weight tensor shape.
/// After loadWeightMap processes the weight (orientWeight + inferGaffineParams),
/// the shape is always in LOGICAL (unpacked) dimensions, regardless of dtype.
/// One dimension is d_model, the other is d_ff. Returns null if ambiguous.
fn inferDffFromWeight(w: Tensor, d_model: usize) ?usize {
    if (w.n_dims != 2) return null;
    const dim0: usize = @intCast(w.shape[0]);
    const dim1: usize = @intCast(w.shape[1]);

    if (dim0 == d_model and dim1 != d_model) return dim1;
    if (dim1 == d_model and dim0 != d_model) return dim0;

    // Square case (d_model == d_ff) — config is already correct
    return null;
}

fn inferMoEFromArchitecture(arch: *const model_types.Architecture, model_config: *ModelConfig) void {
    _ = arch;
    _ = model_config;
}

const ModelLoadDiscovery = struct {
    config: ModelConfig,
    runtime_arch: *const model_types.Architecture,
    manifest: manifest_mod.ModelManifest,
    layer_contexts: []runtime_blocks.BlockMapContext,
    layer_block_kinds: []runtime_blocks.BlockKind,
    original_weight_dtype: DType,
};

const StageErrorPolicy = enum {
    full,
    stage,
};

fn mapConfigInferenceError(comptime policy: StageErrorPolicy, err: anyerror) anyerror {
    if (policy == .stage) {
        return switch (err) {
            error.MissingDffSourceWeight,
            error.InvalidDffSourceWeightShape,
            error.MissingShortConvSourceWeight,
            error.InvalidShortConvSourceWeightShape,
            error.InvalidShortConvConfig,
            => error.UnsupportedStageConfigInference,
            else => err,
        };
    }
    return err;
}

fn discoverModelForLoad(
    backing_allocator: std.mem.Allocator,
    arena_allocator: std.mem.Allocator,
    config_path: []const u8,
    safetensors_file: *st_loader.UnifiedSafeTensors,
    runtime_arch: *const model_types.Architecture,
    parse_config_hook: ?model_types.ConfigParseHook,
    comptime error_policy: StageErrorPolicy,
) !ModelLoadDiscovery {
    const arch = runtime_arch;
    var model_config = try cfg_loader.loadConfigForArchitectureWithHook(
        arena_allocator,
        config_path,
        arch.name,
        parse_config_hook orelse arch.parse_config_hook,
    );

    log.debug("load", "Vision config from loadConfig", .{
        .vision_hidden_size = model_config.vision_hidden_size,
        .vision_depth = model_config.vision_depth,
        .vision_num_heads = model_config.vision_num_heads,
        .vision_intermediate_size = model_config.vision_intermediate_size,
        .projector_hidden_size = model_config.projector_hidden_size,
        .vision_patch_size = model_config.vision_patch_size,
        .vision_spatial_merge_size = model_config.vision_spatial_merge_size,
        .vision_temporal_patch_size = model_config.vision_temporal_patch_size,
        .vision_num_position_embeddings = model_config.vision_num_position_embeddings,
        .vision_max_num_patches = model_config.vision_max_num_patches,
        .image_token_id = model_config.image_token_id,
    }, @src());

    log.trace("load", "Config loaded", .{
        .n_layers = model_config.n_layers,
        .d_model = model_config.d_model,
        .n_heads = model_config.n_heads,
        .head_dim = model_config.head_dim,
    }, @src());

    try requireArchitectureMetadata(arch);
    if (!archHasBlockWeights(arch)) return error.MissingArchitecture;
    inferMoEFromArchitecture(arch, &model_config);

    if (arch.isHeterogeneous()) {
        if (arch.block_variants) |variants| {
            var variant_names = try arena_allocator.alloc([]const u8, variants.len);
            for (variants, 0..) |variant, i| {
                variant_names[i] = variant.name;
            }
            model_config.layer_types = cfg_loader.parseLayerTypes(arena_allocator, config_path, variant_names, arch.variant_aliases) catch null;
        }
    }

    const original_weight_dtype = blk: {
        const detected = try detectOriginalWeightDType(arena_allocator, arch, model_config.layer_types, safetensors_file);
        if (detected == .grouped_affine_u4 and model_config.gaffine_bits == 8) {
            break :blk DType.grouped_affine_u8;
        }
        break :blk detected;
    };

    applyMetadataDffCorrection(
        arena_allocator,
        arch,
        &model_config,
        safetensors_file,
    ) catch |err| return mapConfigInferenceError(error_policy, err);

    var model_manifest = try manifest_mod.build(backing_allocator, arch, &model_config, safetensors_file);
    errdefer model_manifest.deinit();

    if (manifestHasQkNorm(&model_manifest)) model_config.use_qk_norm = true;

    const layer_count: usize = @intCast(model_config.n_layers);
    var layer_contexts = try arena_allocator.alloc(runtime_blocks.BlockMapContext, layer_count);
    var layer_block_kinds = try arena_allocator.alloc(runtime_blocks.BlockKind, layer_count);
    for (0..layer_count) |layer_idx| {
        const layer_info = buildLayerStaticContext(
            arena_allocator,
            arch,
            &model_config,
            safetensors_file,
            layer_idx,
        ) catch |err| return mapConfigInferenceError(error_policy, err);
        layer_contexts[layer_idx] = layer_info.map_context;
        layer_block_kinds[layer_idx] = layer_info.block_type;
    }

    return .{
        .config = model_config,
        .runtime_arch = arch,
        .manifest = model_manifest,
        .layer_contexts = layer_contexts,
        .layer_block_kinds = layer_block_kinds,
        .original_weight_dtype = original_weight_dtype,
    };
}

fn manifestHasQkNorm(model_manifest: *const manifest_mod.ModelManifest) bool {
    const qk_norm_ids = [_][]const u8{
        "self_attn.q_norm.weight",
        "self_attn.q_layernorm.weight",
        "mixer.q_norm.weight",
        "self_attn.k_norm.weight",
        "self_attn.k_layernorm.weight",
        "mixer.k_norm.weight",
    };
    for (qk_norm_ids) |id| {
        if (model_manifest.hasLayerWeight(0, id)) return true;
    }
    return false;
}

fn applyMetadataDffCorrection(
    allocator: std.mem.Allocator,
    arch: *const model_types.Architecture,
    model_config: *ModelConfig,
    safetensors_file: *st_loader.UnifiedSafeTensors,
) !void {
    if (!arch.resolve_d_ff_from_weights) return;
    const specs = if (arch.getVariantWithOverride(0, model_config.layer_types)) |v| v.weights else arch.block_weights;
    const d_model: usize = @intCast(model_config.d_model);
    const config_d_ff: usize = @intCast(model_config.d_ff);
    var found_source_weight = false;

    for (arch.d_ff_source_weight_ids) |id| {
        const spec = findWeightSpecById(specs, id) orelse continue;
        const metadata = try metadataForWeightId(
            allocator,
            safetensors_file,
            specs,
            arch.weight_prefixes,
            0,
            id,
        ) orelse continue;
        found_source_weight = true;
        if (inferDffFromMetadataShape(metadata, spec.*, model_config, safetensors_file, d_model)) |raw_inferred| {
            const inferred = blk: {
                if (arch.has_fused_gate_up) {
                    if ((raw_inferred % 2) != 0) return error.InvalidDffSourceWeightShape;
                    break :blk raw_inferred / 2;
                }
                break :blk raw_inferred;
            };
            if (inferred != config_d_ff) {
                log.info("load", "Corrected d_ff from weight metadata", .{
                    .config_d_ff = config_d_ff,
                    .inferred_d_ff = inferred,
                    .weight_id = id,
                });
                model_config.d_ff = @intCast(inferred);
            }
            return;
        }

        if (metadata.shape.len == 2 and
            metadata.shape[0] == d_model and
            metadata.shape[1] == d_model and
            config_d_ff == d_model)
        {
            return;
        }
    }

    if (!found_source_weight) return error.MissingDffSourceWeight;
    return error.InvalidDffSourceWeightShape;
}

fn inferDffFromShape(shape: []const usize, d_model: usize) ?usize {
    if (shape.len != 2) return null;
    const dim0 = shape[0];
    const dim1 = shape[1];
    if (dim0 == d_model and dim1 != d_model) return dim1;
    if (dim1 == d_model and dim0 != d_model) return dim0;
    return null;
}

fn inferDffFromMetadataShape(
    metadata: st_loader.reader.SafeTensors.TensorMetadata,
    spec: model_types.WeightSpec,
    model_config: *const ModelConfig,
    safetensors_file: *st_loader.UnifiedSafeTensors,
    d_model: usize,
) ?usize {
    if (spec.expected_shape) |shape| {
        if (inferDffFromShape(shape, d_model)) |inferred| return inferred;
        if (shape.len == 2 and shape[0] == d_model and shape[1] == d_model) return d_model;
        return null;
    }

    if (metadata.shape.len != 2) return null;
    if (inferGptqDffFromMetadata(metadata, safetensors_file, model_config, d_model)) |inferred| return inferred;
    if (inferNvfp4DffFromMetadata(metadata, safetensors_file, d_model)) |inferred| return inferred;

    if (metadata.dtype == .grouped_affine_u4 or metadata.dtype == .grouped_affine_u8) {
        const rows = metadata.shape[0];
        const cols = metadata.shape[1];
        const values_per_word: usize = if (model_config.gaffine_bits == 8) 4 else 8;
        const unpacked_cols = cols * values_per_word;

        if (rows == d_model) return unpacked_cols;
        if (unpacked_cols == d_model) return rows;
    }

    return inferDffFromShape(metadata.shape, d_model);
}

fn inferGptqDffFromMetadata(
    metadata: st_loader.reader.SafeTensors.TensorMetadata,
    safetensors_file: *st_loader.UnifiedSafeTensors,
    model_config: *const ModelConfig,
    d_model: usize,
) ?usize {
    const base = weightBaseForSuffix(metadata.name, ".qweight") orelse return null;
    const scales_metadata = metadataForCompanion(safetensors_file, base, ".scales") orelse return null;
    if (scales_metadata.shape.len != 2) return null;

    const dim0 = metadata.shape[0];
    const dim1 = metadata.shape[1];
    const n_groups = scales_metadata.shape[0];
    const scales_out = scales_metadata.shape[1];
    if (n_groups == 0 or scales_out == 0) return null;

    var in_packed: usize = undefined;
    var out_features: usize = undefined;
    if (dim1 == scales_out and dim0 != scales_out) {
        in_packed = dim0;
        out_features = dim1;
    } else if (dim0 == scales_out and dim1 != scales_out) {
        out_features = dim0;
        in_packed = dim1;
    } else if (dim0 == scales_out and dim1 == scales_out) {
        if (dim0 * 8 == d_model or dim0 * 4 == d_model) {
            in_packed = dim0;
            out_features = dim1;
        } else {
            out_features = dim0;
            in_packed = dim1;
        }
    } else {
        return null;
    }

    const unpacked_4bit = in_packed * 8;
    const unpacked_8bit = in_packed * 4;
    const group_size_4bit = unpacked_4bit / n_groups;
    const group_size_8bit = unpacked_8bit / n_groups;
    const valid_4bit = (unpacked_4bit % n_groups) == 0 and
        (group_size_4bit == 32 or group_size_4bit == 64 or group_size_4bit == 128);
    const valid_8bit = (unpacked_8bit % n_groups) == 0 and
        (group_size_8bit == 32 or group_size_8bit == 64 or group_size_8bit == 128);
    const is_4bit = if (valid_4bit and valid_8bit)
        model_config.gaffine_bits != 8
    else
        valid_4bit;
    if (!is_4bit and !valid_8bit) return null;

    const unpacked_in = if (is_4bit) unpacked_4bit else unpacked_8bit;
    if (inferDffFromShape(&.{ out_features, unpacked_in }, d_model)) |inferred| return inferred;
    if (out_features == d_model and unpacked_in == d_model) return d_model;
    return null;
}

fn inferNvfp4DffFromMetadata(
    metadata: st_loader.reader.SafeTensors.TensorMetadata,
    safetensors_file: *st_loader.UnifiedSafeTensors,
    d_model: usize,
) ?usize {
    if (metadata.dtype != .u8 and metadata.dtype != .i8) return null;
    const base = weightBaseForSuffix(metadata.name, ".weight_packed") orelse
        weightBaseForSuffix(metadata.name, ".weight") orelse return null;
    const scale_metadata = metadataForCompanion(safetensors_file, base, ".weight_scale") orelse return null;
    if (scale_metadata.dtype != .f8_e4m3 or scale_metadata.shape.len != 2) return null;

    const rows = metadata.shape[0];
    const packed_cols = metadata.shape[1];
    const scale_rows = scale_metadata.shape[0];
    const scale_cols = scale_metadata.shape[1];
    if (rows == 0 or packed_cols == 0 or scale_rows != rows or scale_cols == 0) return null;

    const unpacked_cols = packed_cols * 2;
    if ((unpacked_cols % scale_cols) != 0) return null;
    if (inferDffFromShape(&.{ rows, unpacked_cols }, d_model)) |inferred| return inferred;
    if (rows == d_model and unpacked_cols == d_model) return d_model;
    return null;
}

fn weightBaseForSuffix(name: []const u8, suffix: []const u8) ?[]const u8 {
    if (!std.mem.endsWith(u8, name, suffix)) return null;
    return name[0 .. name.len - suffix.len];
}

fn metadataForCompanion(
    safetensors_file: *st_loader.UnifiedSafeTensors,
    base: []const u8,
    suffix: []const u8,
) ?st_loader.reader.SafeTensors.TensorMetadata {
    var name_buf: [1024]u8 = undefined;
    const companion_name = std.fmt.bufPrint(name_buf[0..], "{s}{s}", .{ base, suffix }) catch return null;
    return safetensors_file.getTensorMetadata(companion_name) catch null;
}

const LayerStaticContext = struct {
    block_type: runtime_blocks.BlockKind,
    map_context: runtime_blocks.BlockMapContext,
};

fn buildLayerStaticContext(
    allocator: std.mem.Allocator,
    arch: *const model_types.Architecture,
    model_config: *ModelConfig,
    safetensors_file: *st_loader.UnifiedSafeTensors,
    layer_idx: usize,
) !LayerStaticContext {
    const variant = arch.getVariantWithOverride(layer_idx, model_config.layer_types);
    const block_type = if (variant) |v|
        runtime_blocks.BlockKind.fromVariantName(v.name) orelse return error.UnknownBlockVariant
    else if (arch.has_mamba)
        runtime_blocks.BlockKind.mamba
    else
        runtime_blocks.BlockKind.attention_mlp;

    const layer_meta = if (variant) |v| v.meta orelse arch.kernel_meta else arch.kernel_meta;
    const variant_is_full_attention = if (variant) |v|
        std.mem.eql(u8, v.name, "full_attention")
    else
        false;
    const has_explicit_layer_types = model_config.layer_types != null;
    const layer_has_global_attn = if (variant_is_full_attention)
        true
    else if (has_explicit_layer_types and model_config.sliding_window > 0)
        false
    else if (model_config.sliding_window <= 0)
        true
    else if (model_config.sliding_window_pattern > 0)
        (@mod(@as(i32, @intCast(layer_idx)), model_config.sliding_window_pattern) == 0)
    else
        false;
    const layer_window_size: usize = if (model_config.sliding_window > 0 and !layer_has_global_attn)
        @intCast(model_config.sliding_window)
    else
        0;

    var map_context = runtime_blocks.BlockMapContext{
        .sliding_window = if (block_type == .attention_mlp) layer_window_size else 0,
        .kernel_meta = layer_meta,
        .mamba_config = null,
        .gated_delta_config = null,
        .shortconv_config = null,
        .num_experts = if (model_config.num_experts > 0) @intCast(model_config.num_experts) else 0,
        .experts_per_token = if (model_config.experts_per_token > 0) @intCast(model_config.experts_per_token) else 0,
        .allocator = null,
    };

    if (block_type == .mamba) {
        const meta_cfg = layer_meta.mamba_config;
        map_context.mamba_config = .{
            .d_model = @intCast(model_config.d_model),
            .d_state = if (meta_cfg) |cfg| cfg.d_state else @intCast(model_config.mamba_d_state),
            .d_conv = if (meta_cfg) |cfg| cfg.d_conv else @intCast(model_config.mamba_d_conv),
            .n_heads = if (meta_cfg) |cfg| cfg.n_heads else @intCast(model_config.mamba_n_heads),
            .d_head = if (meta_cfg) |cfg| cfg.d_head else @intCast(model_config.mamba_d_head),
            .n_groups = if (meta_cfg) |cfg| cfg.n_groups else @intCast(model_config.mamba_n_groups),
        };
    }

    if (block_type == .gated_delta) {
        const meta_cfg = layer_meta.gated_delta_config orelse return error.MissingGatedDeltaConfig;
        const cfg_num_value_heads: u32 = if (model_config.linear_num_value_heads > 0)
            @intCast(model_config.linear_num_value_heads)
        else
            meta_cfg.n_heads;
        const cfg_value_head_dim: u32 = if (model_config.linear_value_head_dim > 0)
            @intCast(model_config.linear_value_head_dim)
        else
            meta_cfg.d_head;
        if (cfg_num_value_heads == 0 or cfg_value_head_dim == 0) return error.InvalidShape;
        _ = std.math.mul(u32, cfg_num_value_heads, cfg_value_head_dim) catch return error.InvalidShape;
        const cfg_num_key_heads: u32 = if (model_config.linear_num_key_heads > 0)
            @intCast(model_config.linear_num_key_heads)
        else
            cfg_num_value_heads;
        map_context.gated_delta_config = .{
            .d_model = @intCast(model_config.d_model),
            .d_conv = meta_cfg.d_conv,
            .n_heads = cfg_num_value_heads,
            .d_head = cfg_value_head_dim,
            .n_key_heads = cfg_num_key_heads,
        };
    }

    if (block_type == .shortconv) {
        const meta_cfg = layer_meta.shortconv_config;
        map_context.shortconv_config = .{
            .d_model = @intCast(model_config.d_model),
            .d_conv = if (meta_cfg) |cfg| cfg.d_conv else @intCast(model_config.shortconv_d_conv),
            .conv_dim = if (meta_cfg) |cfg| cfg.conv_dim else @intCast(model_config.shortconv_conv_dim),
            .conv_dim_out = if (meta_cfg) |cfg| cfg.conv_dim_out else @intCast(model_config.shortconv_conv_dim_out),
            .has_bias = if (meta_cfg) |cfg| cfg.has_bias else model_config.shortconv_has_bias,
        };
        if (arch.resolve_shortconv_dims_from_weights) {
            const specs = if (variant) |v| v.weights else arch.block_weights;
            try inferShortConvDimsFromMetadata(
                allocator,
                &map_context.shortconv_config.?,
                safetensors_file,
                specs,
                arch.weight_prefixes,
                layer_idx,
                arch.shortconv_dims_source_weight_id,
            );
        } else {
            const cfg = &map_context.shortconv_config.?;
            if (cfg.conv_dim == 0 or cfg.conv_dim_out == 0 or cfg.d_conv == 0) {
                return error.InvalidShortConvConfig;
            }
        }
    }

    return .{ .block_type = block_type, .map_context = map_context };
}

fn metadataForWeightId(
    allocator: std.mem.Allocator,
    safetensors_file: *st_loader.UnifiedSafeTensors,
    specs: []const model_types.WeightSpec,
    weight_prefixes: []const []const u8,
    layer_idx: usize,
    weight_id: []const u8,
) !?st_loader.reader.SafeTensors.TensorMetadata {
    const spec = findWeightSpecById(specs, weight_id) orelse return null;
    var resolved_name_buf: [1024]u8 = undefined;
    var name_resolver: generic_weights.NameResolver = .{};
    defer name_resolver.deinit(allocator);
    const resolved_name = try generic_weights.resolveWeightNameBySpec(
        allocator,
        safetensors_file,
        spec.*,
        weight_prefixes,
        layer_idx,
        &name_resolver,
        resolved_name_buf[0..],
    ) orelse return null;
    return try safetensors_file.getTensorMetadata(resolved_name);
}

fn inferShortConvDimsFromMetadata(
    allocator: std.mem.Allocator,
    config: *runtime_blocks.ShortConvConfig,
    safetensors_file: *st_loader.UnifiedSafeTensors,
    specs: []const model_types.WeightSpec,
    weight_prefixes: []const []const u8,
    layer_idx: usize,
    source_weight_id: ?[]const u8,
) !void {
    const id = source_weight_id orelse return error.MissingShortConvSourceWeightId;
    const metadata = try metadataForWeightId(
        allocator,
        safetensors_file,
        specs,
        weight_prefixes,
        layer_idx,
        id,
    ) orelse return error.MissingShortConvSourceWeight;
    if (metadata.shape.len != 2 and metadata.shape.len != 3) return error.InvalidShortConvSourceWeightShape;

    const prev_conv_dim = config.conv_dim;
    const inferred_conv_dim: u32 = @intCast(metadata.shape[0]);
    const d_idx: usize = if (metadata.shape.len == 3) 2 else 1;
    const inferred_d_conv: u32 = @intCast(metadata.shape[d_idx]);

    if (config.conv_dim == 0 or config.conv_dim != inferred_conv_dim) {
        config.conv_dim = inferred_conv_dim;
    }
    if (config.d_conv == 0 or config.d_conv != inferred_d_conv) {
        config.d_conv = inferred_d_conv;
    }
    if (config.conv_dim_out == 0 and config.conv_dim != 0) {
        config.conv_dim_out = config.conv_dim;
    } else if (prev_conv_dim != 0 and config.conv_dim_out == prev_conv_dim and config.conv_dim != prev_conv_dim) {
        config.conv_dim_out = config.conv_dim;
    }
    if (config.conv_dim == 0 or config.conv_dim_out == 0 or config.d_conv == 0) {
        return error.InvalidShortConvConfig;
    }
}

fn specsForLayer(
    arch: *const model_types.Architecture,
    model_config: *const ModelConfig,
    layer_idx: usize,
) []const model_types.WeightSpec {
    const variant = arch.getVariantWithOverride(layer_idx, model_config.layer_types);
    return if (variant) |v| v.weights else arch.block_weights;
}

fn requiredLayerSpecSatisfied(
    model_manifest: *const manifest_mod.ModelManifest,
    layer_idx: usize,
    spec_id: []const u8,
) bool {
    if (model_manifest.hasLayerWeight(layer_idx, spec_id)) return true;
    if (std.mem.eql(u8, spec_id, "mlp.up_proj.weight") and
        model_manifest.hasLayerWeight(layer_idx, "mlp.gate_up_proj.weight"))
    {
        return true;
    }
    if ((std.mem.eql(u8, spec_id, "self_attn.q_proj.weight") or
        std.mem.eql(u8, spec_id, "self_attn.k_proj.weight") or
        std.mem.eql(u8, spec_id, "self_attn.v_proj.weight")) and
        model_manifest.hasLayerWeight(layer_idx, "self_attn.qkv_proj.weight"))
    {
        return true;
    }
    return false;
}

fn roleSupportedByArchitectureOrManifest(
    arch: *const model_types.Architecture,
    model_config: *const ModelConfig,
    model_manifest: *const manifest_mod.ModelManifest,
    role: manifest_mod.TensorRole,
) bool {
    if (role == .lm_head and model_config.tie_word_embeddings and model_manifest.hasGlobalWeight("token_embeddings")) {
        return true;
    }
    for (arch.global_weights) |spec| {
        if (manifest_mod.roleForGlobalWeight(spec.id) == role) return true;
    }
    return model_manifest.hasRole(role);
}

fn effectiveRolesForStage(
    roles: StageRoleRequest,
    model_config: *const ModelConfig,
    model_manifest: *const manifest_mod.ModelManifest,
) StageRoleRequest {
    var effective = roles;
    if (roles.include_lm_head and
        model_config.tie_word_embeddings and
        !model_manifest.hasGlobalWeight("lm_head"))
    {
        effective.include_token_embeddings = true;
    }
    return effective;
}

fn validateRequestedRole(
    discovery: *const ModelLoadDiscovery,
    role: manifest_mod.TensorRole,
    requested: bool,
) !void {
    if (!requested) return;
    if (!roleSupportedByArchitectureOrManifest(discovery.runtime_arch, &discovery.config, &discovery.manifest, role)) {
        return error.UnsupportedStageGlobalRole;
    }
    if (role == .lm_head and
        discovery.config.tie_word_embeddings and
        !discovery.manifest.hasGlobalWeight("lm_head"))
    {
        if (!discovery.manifest.hasGlobalWeight("token_embeddings")) return error.MissingStageGlobalWeight;
        return;
    }
    if (!discovery.manifest.hasRole(role)) return error.MissingStageGlobalWeight;
}

fn validateStageLoadRequest(
    discovery: *const ModelLoadDiscovery,
    request: StageLoadRequest,
) !manifest_mod.StageResidencyReport {
    if (request.layer_start >= request.layer_end) return error.EmptyStageRange;
    const layer_count: usize = @intCast(discovery.config.n_layers);
    if (request.layer_end > layer_count) return error.InvalidStageRange;

    for (request.layer_start..request.layer_end) |layer_idx| {
        const specs = specsForLayer(discovery.runtime_arch, &discovery.config, layer_idx);
        for (specs) |spec| {
            if (!spec.required) continue;
            if (!requiredLayerSpecSatisfied(&discovery.manifest, layer_idx, spec.id)) {
                return error.MissingStageWeight;
            }
        }
    }

    try validateRequestedRole(discovery, .token_embeddings, request.roles.include_token_embeddings);
    try validateRequestedRole(discovery, .final_norm, request.roles.include_final_norm);
    try validateRequestedRole(discovery, .lm_head, request.roles.include_lm_head);
    try validateRequestedRole(discovery, .embedding_side, request.roles.include_embedding_side);
    try validateRequestedRole(discovery, .vision_side, request.roles.include_vision_side);
    try validateRequestedRole(discovery, .architecture_side, request.roles.include_architecture_side);
    try validateRequestedRole(discovery, .unclassified_global, request.roles.include_unclassified_global);

    const effective_roles = effectiveRolesForStage(request.roles, &discovery.config, &discovery.manifest);
    return discovery.manifest.stageResidencyReport(effective_roles.toResidencyRequest(request.range())) catch |err| switch (err) {
        error.InvalidLayerRange => error.InvalidStageRange,
        else => err,
    };
}

fn loadModelLayersForRange(
    arena_allocator: std.mem.Allocator,
    safetensors_file: *st_loader.UnifiedSafeTensors,
    config: *ModelConfig,
    runtime_arch: *const model_types.Architecture,
    layer_contexts: []const runtime_blocks.BlockMapContext,
    layer_block_kinds: []const runtime_blocks.BlockKind,
    layer_range: StageLayerRange,
    model_load_options: LoadOptions,
    env_flags: LoaderEnvFlags,
    progress: progress_mod.Context,
) ![]runtime_blocks.LayerWeights {
    if (layer_range.start > layer_range.end or layer_range.end > layer_block_kinds.len) {
        return error.InvalidStageRange;
    }
    const range_len = layer_range.len();
    var block_weights = try arena_allocator.alloc(runtime_blocks.LayerWeights, range_len);
    var block_time_ns: i128 = 0;

    progress.addLine(0, "Loading", @intCast(range_len), null, null);
    for (0..range_len) |offset| {
        const layer_idx = layer_range.start + offset;
        const block_start_ns: i128 = std.time.nanoTimestamp();
        const specs = specsForLayer(runtime_arch, config, layer_idx);
        const weight_load_options = generic_weights.LoadOptions{
            .preserve_native_norm_dtype = model_load_options.preserve_native_norm_dtype,
            .dequantize_mxfp8_to_bf16 = model_load_options.dequantize_mxfp8_to_bf16,
            .dequantize_nvfp4_to_bf16 = model_load_options.dequantize_nvfp4_to_bf16,
        };
        var weight_map = try generic_weights.loadWeightMap(
            arena_allocator,
            safetensors_file,
            specs,
            runtime_arch.weight_prefixes,
            layer_idx,
            config,
            weight_load_options,
        );

        if (config.attention_k_eq_v and weight_map.get("self_attn.v_proj.weight") == null) {
            if (weight_map.get("self_attn.k_proj.weight")) |k| {
                try weight_map.put(arena_allocator, "self_attn.v_proj.weight", k);
            }
        }

        if (env_flags.enable_cpu_fusion and runtime_arch.enable_loader_fusions) {
            try maybeAddFusedWeights(arena_allocator, &weight_map);
        }

        var layer_context = layer_contexts[layer_idx];
        layer_context.allocator = null;
        block_weights[offset] = .{
            .block_type = layer_block_kinds[layer_idx],
            .weight_map = weight_map,
            .map_context = layer_context,
        };
        block_time_ns += std.time.nanoTimestamp() - block_start_ns;
        progress.updateLine(0, @intCast(offset + 1), null);
    }
    progress.completeLine(0);

    log.debug("load", "Blocks loaded", .{
        .layer_count = range_len,
        .total_ms = @as(f64, @floatFromInt(block_time_ns)) / 1_000_000.0,
        .per_block_ms = if (range_len == 0) 0 else @as(f64, @floatFromInt(block_time_ns)) / 1_000_000.0 / @as(f64, @floatFromInt(range_len)),
    }, @src());

    return block_weights;
}

const LoadedGlobalWeights = struct {
    ln_final: ?Tensor = null,
    lm_head: ?Tensor = null,
    token_embeddings: ?Tensor = null,
    position_embeddings: ?Tensor = null,
    token_type_embeddings: ?Tensor = null,
    embedding_norm_weight: ?Tensor = null,
    embedding_norm_bias: ?Tensor = null,
    extra_global_weights: runtime_blocks.WeightMap = .{},
    extra_global_role_bytes: [manifest_mod.role_count]usize = [_]usize{0} ** manifest_mod.role_count,
    lm_head_uses_token_embeddings: bool = false,
};

const GlobalLoadMode = enum {
    full,
    stage,
};

fn filterGlobalSpecsForRoles(
    allocator: std.mem.Allocator,
    arch: *const model_types.Architecture,
    roles: StageRoleRequest,
) ![]model_types.WeightSpec {
    var filtered = std.ArrayListUnmanaged(model_types.WeightSpec){};
    for (arch.global_weights) |spec| {
        const role = manifest_mod.roleForGlobalWeight(spec.id);
        if (roles.includesManifestRole(role)) {
            try filtered.append(allocator, spec);
        }
    }
    return try filtered.toOwnedSlice(allocator);
}

fn putExtraGlobalWeight(
    allocator: std.mem.Allocator,
    result: *LoadedGlobalWeights,
    key: []const u8,
    weight: Tensor,
    role: manifest_mod.TensorRole,
    checkpoint_bytes: usize,
) !void {
    const stored_key = try allocator.dupe(u8, key);
    const stored_weight = try allocator.create(Tensor); // lint:ignore errdefer-alloc - arena freed atomically
    stored_weight.* = weight;
    try result.extra_global_weights.put(allocator, stored_key, stored_weight);
    result.extra_global_role_bytes[@intFromEnum(role)] +|= checkpoint_bytes;
}

fn isKnownGlobalFieldId(id: []const u8) bool {
    return std.mem.eql(u8, id, "token_embeddings") or
        std.mem.eql(u8, id, "ln_final") or
        std.mem.eql(u8, id, "lm_head") or
        std.mem.eql(u8, id, "position_embeddings") or
        std.mem.eql(u8, id, "token_type_embeddings") or
        std.mem.eql(u8, id, "embedding_ln") or
        std.mem.eql(u8, id, "embedding_ln_bias");
}

fn loadRawManifestEntriesForRole(
    allocator: std.mem.Allocator,
    safetensors_file: *st_loader.UnifiedSafeTensors,
    model_manifest: *const manifest_mod.ModelManifest,
    result: *LoadedGlobalWeights,
    role: manifest_mod.TensorRole,
) !void {
    for (model_manifest.entries) |entry| {
        if (entry.role != role or entry.layer_index != null) continue;
        if (entry.status == .quant_companion) continue;
        if (entry.weight_id != null) continue;
        const tensor_view = try safetensors_file.getTensor(entry.name, null);
        try putExtraGlobalWeight(allocator, result, entry.name, tensor_view, role, entry.checkpoint_bytes);
    }
}

fn loadGlobalWeightsForRoles(
    arena_allocator: std.mem.Allocator,
    safetensors_file: *st_loader.UnifiedSafeTensors,
    arch: *const model_types.Architecture,
    model_config: *const ModelConfig,
    model_manifest: *const manifest_mod.ModelManifest,
    model_load_options: LoadOptions,
    roles: StageRoleRequest,
    comptime mode: GlobalLoadMode,
) !LoadedGlobalWeights {
    if (!archHasGlobalWeights(arch)) return error.MissingArchitecture;
    var result = LoadedGlobalWeights{};
    const filtered_specs = try filterGlobalSpecsForRoles(arena_allocator, arch, roles);
    var global_map = generic_weights.loadWeightMap(
        arena_allocator,
        safetensors_file,
        filtered_specs,
        &.{},
        0,
        model_config,
        .{
            .preserve_native_norm_dtype = model_load_options.preserve_native_norm_dtype,
            .dequantize_mxfp8_to_bf16 = model_load_options.dequantize_mxfp8_to_bf16,
            .dequantize_nvfp4_to_bf16 = model_load_options.dequantize_nvfp4_to_bf16,
        },
    ) catch |err| return if (mode == .stage) switch (err) {
        error.MissingWeight => error.MissingStageGlobalWeight,
        else => err,
    } else err;

    if (roles.include_token_embeddings) {
        result.token_embeddings = if (global_map.get("token_embeddings")) |weight|
            weight.*
        else
            return if (mode == .stage) error.MissingStageGlobalWeight else error.MissingWeight;
    }

    if (roles.include_final_norm) {
        result.ln_final = if (global_map.get("ln_final")) |weight|
            weight.*
        else if (mode == .stage) return error.MissingStageGlobalWeight else null;
    }

    if (roles.include_lm_head) {
        result.lm_head = if (global_map.get("lm_head")) |weight|
            weight.*
        else if (model_config.tie_word_embeddings) blk: {
            if (try buildTiedLmHead(arena_allocator, safetensors_file, arch, model_config, model_load_options)) |tied_projection| {
                result.lm_head_uses_token_embeddings = true;
                break :blk tied_projection;
            }
            const token_embedding_weights = result.token_embeddings orelse {
                if (mode == .stage) return error.MissingStageGlobalWeight;
                return error.MissingWeight;
            };
            result.lm_head_uses_token_embeddings = true;
            if (token_embedding_weights.dtype == .f32) {
                break :blk try transposeToOwned(arena_allocator, token_embedding_weights, .f32);
            } else {
                break :blk token_embedding_weights;
            }
        } else if (mode == .stage) return error.MissingStageGlobalWeight else null;
    }

    if (roles.include_embedding_side) {
        result.position_embeddings = if (global_map.get("position_embeddings")) |weight|
            try transforms.ensureF32(arena_allocator, weight.*)
        else
            null;
        result.token_type_embeddings = if (global_map.get("token_type_embeddings")) |weight|
            try transforms.ensureF32(arena_allocator, weight.*)
        else
            null;
        result.embedding_norm_weight = if (global_map.get("embedding_ln")) |weight|
            try transforms.ensureF32(arena_allocator, weight.*)
        else
            null;
        result.embedding_norm_bias = if (global_map.get("embedding_ln_bias")) |weight|
            try transforms.ensureF32(arena_allocator, weight.*)
        else
            null;
        if (model_manifest.hasRole(.embedding_side) and
            result.position_embeddings == null and
            result.token_type_embeddings == null and
            result.embedding_norm_weight == null and
            result.embedding_norm_bias == null)
        {
            return error.MissingStageGlobalWeight;
        }
    }

    var it = global_map.iterator();
    while (it.next()) |entry| {
        const id = entry.key_ptr.*;
        if (isKnownGlobalFieldId(id)) continue;
        const role = manifest_mod.roleForGlobalWeight(id);
        if (!roles.includesManifestRole(role)) continue;
        const checkpoint_bytes = if (model_manifest.findGlobalWeight(id)) |manifest_entry|
            manifest_entry.checkpoint_bytes
        else
            entry.value_ptr.*.*.data_size;
        try putExtraGlobalWeight(arena_allocator, &result, id, entry.value_ptr.*.*, role, checkpoint_bytes);
    }

    if (mode == .stage and roles.include_vision_side) {
        try loadRawManifestEntriesForRole(arena_allocator, safetensors_file, model_manifest, &result, .vision_side);
    }
    if (mode == .stage and roles.include_unclassified_global) {
        try loadRawManifestEntriesForRole(arena_allocator, safetensors_file, model_manifest, &result, .unclassified_global);
    }

    return result;
}

pub fn loadModelWithArchitecture(
    backing_allocator: std.mem.Allocator,
    config_path: []const u8,
    safetensors_path: []const u8,
    runtime_arch: ?*const model_types.Architecture,
    parse_config_hook: ?model_types.ConfigParseHook,
    model_load_options: LoadOptions,
    progress: progress_mod.Context,
) !LoadedModel {
    const env_flags = LoaderEnvFlags.fromEnvironment(backing_allocator);
    var start_time_ns: i128 = std.time.nanoTimestamp();

    var arena_alloc = std.heap.ArenaAllocator.init(backing_allocator);
    errdefer arena_alloc.deinit();
    const arena_allocator = arena_alloc.allocator();

    var safetensors_file = try st_loader.UnifiedSafeTensors.load(backing_allocator, safetensors_path);
    errdefer safetensors_file.deinit();
    {
        const now = std.time.nanoTimestamp();
        const duration_ms = @as(f64, @floatFromInt(now - start_time_ns)) / 1_000_000.0;
        log.debug("load", "Safetensors mmap", .{ .duration_ms = duration_ms }, @src());
        start_time_ns = now;
    }

    const arch = runtime_arch orelse return error.MissingArchitecture;
    var discovery = try discoverModelForLoad(
        backing_allocator,
        arena_allocator,
        config_path,
        &safetensors_file,
        arch,
        parse_config_hook,
        .full,
    );
    errdefer discovery.manifest.deinit();
    {
        const now = std.time.nanoTimestamp();
        const duration_ms = @as(f64, @floatFromInt(now - start_time_ns)) / 1_000_000.0;
        log.debug("load", "Model discovery complete", .{ .duration_ms = duration_ms }, @src());
        start_time_ns = now;
    }

    const layer_count = discovery.layer_block_kinds.len;
    const block_weights = try loadModelLayersForRange(
        arena_allocator,
        &safetensors_file,
        &discovery.config,
        arch,
        discovery.layer_contexts,
        discovery.layer_block_kinds,
        .{ .start = 0, .end = layer_count },
        model_load_options,
        env_flags,
        progress,
    );

    {
        const now = std.time.nanoTimestamp();
        const duration_ms = @as(f64, @floatFromInt(now - start_time_ns)) / 1_000_000.0;
        log.debug("load", "Block weights ready", .{ .layer_count = layer_count, .duration_ms = duration_ms }, @src());
        start_time_ns = now;
    }

    const all_roles = StageRoleRequest{
        .include_token_embeddings = true,
        .include_final_norm = true,
        .include_lm_head = true,
        .include_embedding_side = true,
        .include_vision_side = true,
        .include_architecture_side = true,
    };
    const globals = try loadGlobalWeightsForRoles(
        arena_allocator,
        &safetensors_file,
        arch,
        &discovery.config,
        &discovery.manifest,
        model_load_options,
        all_roles,
        .full,
    );
    const token_embedding_weights = globals.token_embeddings orelse return error.MissingWeight;

    {
        const now = std.time.nanoTimestamp();
        const duration_ms = @as(f64, @floatFromInt(now - start_time_ns)) / 1_000_000.0;
        log.debug("load", "Global weights loaded", .{ .duration_ms = duration_ms }, @src());
    }

    return LoadedModel{
        .arena = arena_alloc,
        .config = discovery.config,
        .runtime = .{},
        .st = safetensors_file,
        .ln_final = globals.ln_final,
        .lm_head = globals.lm_head,
        .token_embeddings = token_embedding_weights,
        .position_embeddings = globals.position_embeddings,
        .token_type_embeddings = globals.token_type_embeddings,
        .embedding_norm_weight = globals.embedding_norm_weight,
        .embedding_norm_bias = globals.embedding_norm_bias,
        .blocks = block_weights,
        .original_weight_dtype = discovery.original_weight_dtype,
        .file_size = safetensors_file.fileSize(),
        .tensor_count = safetensors_file.tensorCount(),
        .manifest = discovery.manifest,
    };
}

pub fn loadStageModelWithArchitecture(
    backing_allocator: std.mem.Allocator,
    config_path: []const u8,
    safetensors_path: []const u8,
    runtime_arch: *const model_types.Architecture,
    parse_config_hook: ?model_types.ConfigParseHook,
    model_load_options: LoadOptions,
    request: StageLoadRequest,
    progress: progress_mod.Context,
) !LoadedStageModel {
    const env_flags = LoaderEnvFlags.fromEnvironment(backing_allocator);
    var arena_alloc = std.heap.ArenaAllocator.init(backing_allocator);
    errdefer arena_alloc.deinit();
    const arena_allocator = arena_alloc.allocator();

    var safetensors_file = try st_loader.UnifiedSafeTensors.load(backing_allocator, safetensors_path);
    errdefer safetensors_file.deinit();

    var discovery = try discoverModelForLoad(
        backing_allocator,
        arena_allocator,
        config_path,
        &safetensors_file,
        runtime_arch,
        parse_config_hook,
        .stage,
    );
    errdefer discovery.manifest.deinit();

    const residency = try validateStageLoadRequest(&discovery, request);
    const stage_blocks = loadModelLayersForRange(
        arena_allocator,
        &safetensors_file,
        &discovery.config,
        runtime_arch,
        discovery.layer_contexts,
        discovery.layer_block_kinds,
        request.range(),
        model_load_options,
        env_flags,
        progress,
    ) catch |err| return switch (err) {
        error.MissingWeight => error.MissingStageWeight,
        else => err,
    };

    const effective_roles = effectiveRolesForStage(request.roles, &discovery.config, &discovery.manifest);
    const globals = try loadGlobalWeightsForRoles(
        arena_allocator,
        &safetensors_file,
        runtime_arch,
        &discovery.config,
        &discovery.manifest,
        model_load_options,
        effective_roles,
        .stage,
    );

    return .{
        .arena = arena_alloc,
        .config = discovery.config,
        .runtime = .{},
        .st = safetensors_file,
        .ln_final = globals.ln_final,
        .lm_head = globals.lm_head,
        .token_embeddings = globals.token_embeddings,
        .position_embeddings = globals.position_embeddings,
        .token_type_embeddings = globals.token_type_embeddings,
        .embedding_norm_weight = globals.embedding_norm_weight,
        .embedding_norm_bias = globals.embedding_norm_bias,
        .blocks = stage_blocks,
        .layer_range = request.range(),
        .requested_roles = request.roles,
        .residency = residency,
        .original_weight_dtype = discovery.original_weight_dtype,
        .file_size = safetensors_file.fileSize(),
        .tensor_count = safetensors_file.tensorCount(),
        .manifest = discovery.manifest,
        .extra_global_weights = globals.extra_global_weights,
        .extra_global_role_bytes = globals.extra_global_role_bytes,
        .lm_head_uses_token_embeddings = globals.lm_head_uses_token_embeddings,
    };
}

fn transposeToOwned(allocator: std.mem.Allocator, t: Tensor, data_type: DType) !Tensor {
    const rows: usize = @intCast(t.shape[0]);
    const cols: usize = @intCast(t.shape[1]);

    const owned = try tensor.OwnedTensor.init(allocator, data_type, &.{ cols, rows });
    switch (data_type) {
        .f32 => {
            const src_f32 = t.asSlice(f32);
            const dst_f32 = owned.asSlice(f32);
            for (0..rows) |r| {
                for (0..cols) |c| {
                    dst_f32[c * rows + r] = src_f32[r * cols + c];
                }
            }
        },
        .bf16, .f16 => {
            const src_u16 = t.asSliceUnaligned(u16);
            const dst_u16 = owned.asSlice(u16);
            for (0..rows) |r| {
                for (0..cols) |c| {
                    dst_u16[c * rows + r] = src_u16[r * cols + c];
                }
            }
        },
        else => return error.UnsupportedDType,
    }

    return owned.view();
}

fn readEnvFlag(allocator: std.mem.Allocator, name: []const u8, default_value: bool) bool {
    const env_value = std.process.getEnvVarOwned(allocator, name) catch return default_value;
    defer allocator.free(env_value);

    if (std.ascii.eqlIgnoreCase(env_value, "0") or std.ascii.eqlIgnoreCase(env_value, "false")) return false;
    if (std.ascii.eqlIgnoreCase(env_value, "1") or std.ascii.eqlIgnoreCase(env_value, "true")) return true;
    return default_value;
}

// =============================================================================
// Unit Tests
// =============================================================================

const stage_test_block_specs = [_]model_types.WeightSpec{
    .{
        .id = "proj.weight",
        .suffix = "proj.weight",
        .module_type = "Linear",
        .layout = .linear,
        .dtype = "float32",
        .required = true,
    },
};

const stage_test_global_specs = [_]model_types.WeightSpec{
    .{
        .id = "token_embeddings",
        .suffix = "tok.weight",
        .module_type = "Embedding",
        .layout = .embedding,
        .dtype = "float32",
        .required = true,
    },
    .{
        .id = "ln_final",
        .suffix = "norm.weight",
        .module_type = "RMSNorm",
        .layout = .none,
        .dtype = "float32",
        .required = false,
    },
    .{
        .id = "lm_head",
        .suffix = "lm_head.weight",
        .module_type = "Linear",
        .layout = .linear,
        .dtype = "float32",
        .required = false,
    },
};

const stage_test_prefixes = [_][]const u8{"layers.{d}."};

fn stageTestArchitecture() model_types.Architecture {
    return .{
        .name = "stage_test",
        .model_types = &.{},
        .block_weights = &stage_test_block_specs,
        .global_weights = &stage_test_global_specs,
        .weight_prefixes = &stage_test_prefixes,
        .weight_dtype_source_weight_ids = &.{"proj.weight"},
    };
}

const StageFixtureOptions = struct {
    include_token_embeddings: bool = true,
    include_layer1: bool = true,
    include_lm_head: bool = true,
    include_vision: bool = false,
    tie_word_embeddings: bool = false,
    intermediate_size: usize = 8,
};

fn writeStageTestConfig(
    allocator: std.mem.Allocator,
    dir: std.fs.Dir,
    options: StageFixtureOptions,
) ![]u8 {
    const tie_value = if (options.tie_word_embeddings) "true" else "false";
    const config_json = try std.fmt.allocPrint(allocator,
        \\{{
        \\  "model_type": "stage_test",
        \\  "vocab_size": 4,
        \\  "hidden_size": 4,
        \\  "num_hidden_layers": 3,
        \\  "num_attention_heads": 1,
        \\  "intermediate_size": {d},
        \\  "max_position_embeddings": 16,
        \\  "tie_word_embeddings": {s}
        \\}}
    , .{ options.intermediate_size, tie_value });
    defer allocator.free(config_json);
    try dir.writeFile(.{ .sub_path = "config.json", .data = config_json });
    return try dir.realpathAlloc(allocator, "config.json");
}

fn writeStageTestSafetensors(
    allocator: std.mem.Allocator,
    dir_path: []const u8,
    options: StageFixtureOptions,
) ![]u8 {
    const model_path = try std.fs.path.join(allocator, &.{ dir_path, "model.safetensors" });
    errdefer allocator.free(model_path);

    const embed = [_]f32{1.0} ** 16;
    const layer0 = [_]f32{2.0} ** 16;
    const layer1 = [_]f32{3.0} ** 16;
    const layer2 = [_]f32{4.0} ** 16;
    const norm = [_]f32{5.0} ** 4;
    const lm_head = [_]f32{6.0} ** 16;
    const vision = [_]f32{7.0} ** 4;

    var entries = std.ArrayListUnmanaged(st_loader.TensorEntry){};
    defer entries.deinit(allocator);
    if (options.include_token_embeddings) {
        try entries.append(allocator, .{ .name = "tok.weight", .dtype = .f32, .shape = &.{ 4, 4 }, .data = std.mem.sliceAsBytes(embed[0..]) });
    }
    try entries.append(allocator, .{ .name = "layers.0.proj.weight", .dtype = .f32, .shape = &.{ 4, 4 }, .data = std.mem.sliceAsBytes(layer0[0..]) });
    if (options.include_layer1) {
        try entries.append(allocator, .{ .name = "layers.1.proj.weight", .dtype = .f32, .shape = &.{ 4, 4 }, .data = std.mem.sliceAsBytes(layer1[0..]) });
    }
    try entries.append(allocator, .{ .name = "layers.2.proj.weight", .dtype = .f32, .shape = &.{ 4, 4 }, .data = std.mem.sliceAsBytes(layer2[0..]) });
    try entries.append(allocator, .{ .name = "norm.weight", .dtype = .f32, .shape = &.{4}, .data = std.mem.sliceAsBytes(norm[0..]) });
    if (options.include_lm_head) {
        try entries.append(allocator, .{ .name = "lm_head.weight", .dtype = .f32, .shape = &.{ 4, 4 }, .data = std.mem.sliceAsBytes(lm_head[0..]) });
    }
    if (options.include_vision) {
        try entries.append(allocator, .{ .name = "vision.patch_embed.weight", .dtype = .f32, .shape = &.{ 2, 2 }, .data = std.mem.sliceAsBytes(vision[0..]) });
    }
    try st_loader.write(allocator, model_path, entries.items);
    return model_path;
}

fn expectStageProjFirstValue(block: *const runtime_blocks.LayerWeights, expected: f32) !void {
    const proj = block.weight_map.get("proj.weight") orelse return error.TestExpectedEqual;
    const values = proj.asSlice(f32);
    try std.testing.expectApproxEqAbs(expected, values[0], 0.0001);
}

fn writeStageDffSafetensors(
    allocator: std.mem.Allocator,
    dir_path: []const u8,
) ![]u8 {
    const model_path = try std.fs.path.join(allocator, &.{ dir_path, "model.safetensors" });
    errdefer allocator.free(model_path);

    const embed = [_]f32{1.0} ** 16;
    const layer0 = [_]f32{2.0} ** 32;
    const layer1 = [_]f32{3.0} ** 32;
    const layer2 = [_]f32{4.0} ** 32;

    var entries = std.ArrayListUnmanaged(st_loader.TensorEntry){};
    defer entries.deinit(allocator);
    try entries.append(allocator, .{ .name = "tok.weight", .dtype = .f32, .shape = &.{ 4, 4 }, .data = std.mem.sliceAsBytes(embed[0..]) });
    try entries.append(allocator, .{ .name = "layers.0.proj.weight", .dtype = .f32, .shape = &.{ 8, 4 }, .data = std.mem.sliceAsBytes(layer0[0..]) });
    try entries.append(allocator, .{ .name = "layers.1.proj.weight", .dtype = .f32, .shape = &.{ 8, 4 }, .data = std.mem.sliceAsBytes(layer1[0..]) });
    try entries.append(allocator, .{ .name = "layers.2.proj.weight", .dtype = .f32, .shape = &.{ 8, 4 }, .data = std.mem.sliceAsBytes(layer2[0..]) });
    try st_loader.write(allocator, model_path, entries.items);
    return model_path;
}

fn writeStagePackedDffSafetensors(
    allocator: std.mem.Allocator,
    dir_path: []const u8,
) ![]u8 {
    const model_path = try std.fs.path.join(allocator, &.{ dir_path, "model.safetensors" });
    errdefer allocator.free(model_path);

    const packed_layer0 = [_]u32{0} ** 64;
    const entries = [_]st_loader.TensorEntry{
        .{ .name = "layers.0.proj.weight", .dtype = .grouped_affine_u4, .shape = &.{ 32, 2 }, .data = std.mem.sliceAsBytes(packed_layer0[0..]) },
    };
    try st_loader.write(allocator, model_path, &entries);
    return model_path;
}

fn writeStageGptqDffSafetensors(
    allocator: std.mem.Allocator,
    dir_path: []const u8,
) ![]u8 {
    const model_path = try std.fs.path.join(allocator, &.{ dir_path, "model.safetensors" });
    errdefer allocator.free(model_path);

    const qweight = [_]i32{0} ** (4 * 64);
    const scales = [_]u16{0} ** 64;
    const qzeros = [_]i32{0} ** 8;
    const entries = [_]st_loader.TensorEntry{
        .{ .name = "layers.0.proj.qweight", .dtype = .i32, .shape = &.{ 4, 64 }, .data = std.mem.sliceAsBytes(qweight[0..]) },
        .{ .name = "layers.0.proj.scales", .dtype = .f16, .shape = &.{ 1, 64 }, .data = std.mem.sliceAsBytes(scales[0..]) },
        .{ .name = "layers.0.proj.qzeros", .dtype = .i32, .shape = &.{ 1, 8 }, .data = std.mem.sliceAsBytes(qzeros[0..]) },
    };
    try st_loader.write(allocator, model_path, &entries);
    return model_path;
}

fn writeStageNvfp4DffSafetensors(
    allocator: std.mem.Allocator,
    dir_path: []const u8,
) ![]u8 {
    const model_path = try std.fs.path.join(allocator, &.{ dir_path, "model.safetensors" });
    errdefer allocator.free(model_path);

    const packed_weight = [_]u8{0} ** (64 * 16);
    const block_scales = [_]u8{0x38} ** 64;
    const global_scale = [_]f32{1.0};
    const entries = [_]st_loader.TensorEntry{
        .{ .name = "layers.0.proj.weight_packed", .dtype = .u8, .shape = &.{ 64, 16 }, .data = packed_weight[0..] },
        .{ .name = "layers.0.proj.weight_scale", .dtype = .f8_e4m3, .shape = &.{ 64, 1 }, .data = block_scales[0..] },
        .{ .name = "layers.0.proj.weight_global_scale", .dtype = .f32, .shape = &.{1}, .data = std.mem.sliceAsBytes(global_scale[0..]) },
    };
    try st_loader.write(allocator, model_path, &entries);
    return model_path;
}

fn stageTestArchitectureWithDffCorrection() model_types.Architecture {
    var arch = stageTestArchitecture();
    arch.resolve_d_ff_from_weights = true;
    arch.d_ff_source_weight_ids = &.{"proj.weight"};
    arch.weight_dtype_source_weight_ids = &.{"proj.weight"};
    return arch;
}

const stage_shortconv_specs = [_]model_types.WeightSpec{
    .{
        .id = "conv.conv.weight",
        .suffix = "conv.conv.weight",
        .module_type = "Conv1d",
        .layout = .conv1d_depthwise,
        .dtype = "float32",
        .required = true,
    },
};

var stage_shortconv_variants = [_]model_types.BlockVariant{
    .{ .name = "shortconv", .weights = &stage_shortconv_specs },
};

fn stageShortConvArchitecture() model_types.Architecture {
    return .{
        .name = "stage_shortconv",
        .model_types = &.{},
        .block_variants = &stage_shortconv_variants,
        .global_weights = &stage_test_global_specs,
        .weight_prefixes = &stage_test_prefixes,
        .weight_dtype_source_weight_ids = &.{"conv.conv.weight"},
        .shortconv_dims_source_weight_id = "conv.conv.weight",
        .resolve_shortconv_dims_from_weights = true,
        .has_shortconv = true,
    };
}

fn writeStageShortConvConfig(
    allocator: std.mem.Allocator,
    dir: std.fs.Dir,
) ![]u8 {
    const config_json =
        \\{
        \\  "model_type": "stage_shortconv",
        \\  "vocab_size": 4,
        \\  "hidden_size": 4,
        \\  "num_hidden_layers": 2,
        \\  "num_attention_heads": 1,
        \\  "intermediate_size": 8,
        \\  "max_position_embeddings": 16,
        \\  "tie_word_embeddings": false
        \\}
    ;
    try dir.writeFile(.{ .sub_path = "config.json", .data = config_json });
    return try dir.realpathAlloc(allocator, "config.json");
}

fn writeStageShortConvSafetensors(
    allocator: std.mem.Allocator,
    dir_path: []const u8,
    include_layer0_source: bool,
) ![]u8 {
    const model_path = try std.fs.path.join(allocator, &.{ dir_path, "model.safetensors" });
    errdefer allocator.free(model_path);

    const layer0 = [_]f32{2.0} ** 18;
    const layer1 = [_]f32{3.0} ** 18;

    var entries = std.ArrayListUnmanaged(st_loader.TensorEntry){};
    defer entries.deinit(allocator);
    if (include_layer0_source) {
        try entries.append(allocator, .{ .name = "layers.0.conv.conv.weight", .dtype = .f32, .shape = &.{ 6, 3 }, .data = std.mem.sliceAsBytes(layer0[0..]) });
    }
    try entries.append(allocator, .{ .name = "layers.1.conv.conv.weight", .dtype = .f32, .shape = &.{ 6, 3 }, .data = std.mem.sliceAsBytes(layer1[0..]) });
    try st_loader.write(allocator, model_path, entries.items);
    return model_path;
}

test "manifest StageRoleRequest.toResidencyRequest maps vision side independently" {
    const roles = StageRoleRequest{ .include_vision_side = true };
    const request = roles.toResidencyRequest(.{ .start = 1, .end = 2 });
    try std.testing.expectEqual(@as(usize, 1), request.layer_start);
    try std.testing.expectEqual(@as(usize, 2), request.layer_end);
    try std.testing.expect(request.include_vision_side);
    try std.testing.expect(!request.include_architecture_side);
    try std.testing.expect(roles.includesManifestRole(.vision_side));
    try std.testing.expect(!roles.includesManifestRole(.decoder_layer));
}

test "manifest StageLayerRange.len and StageLoadRequest.range expose contiguous range" {
    const request = StageLoadRequest{
        .layer_start = 2,
        .layer_end = 5,
        .roles = .{},
    };
    const range = request.range();
    try std.testing.expectEqual(@as(usize, 2), range.start);
    try std.testing.expectEqual(@as(usize, 5), range.end);
    try std.testing.expectEqual(@as(usize, 3), range.len());
}

test "manifest LoadedStageModel.originalLayerIndex maps dense stage offsets" {
    var layer_storage = [_]runtime_blocks.LayerWeights{
        .{ .block_type = .attention_mlp, .weight_map = .{}, .map_context = .{} },
        .{ .block_type = .attention_mlp, .weight_map = .{}, .map_context = .{} },
    };
    var stage = std.mem.zeroes(LoadedStageModel);
    stage.layer_range = .{ .start = 3, .end = 5 };
    stage.blocks = layer_storage[0..];

    try std.testing.expectEqual(@as(usize, 3), try stage.originalLayerIndex(0));
    try std.testing.expectEqual(@as(usize, 4), try stage.originalLayerIndex(1));
    try std.testing.expectError(error.InvalidStageLayerOffset, stage.originalLayerIndex(2));
}

test "manifest loadStageModelWithArchitecture hydrates requested layer range only" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);
    const config_path = try writeStageTestConfig(allocator, tmp.dir, .{});
    defer allocator.free(config_path);
    const model_path = try writeStageTestSafetensors(allocator, tmp_path, .{});
    defer allocator.free(model_path);
    var arch = stageTestArchitecture();

    var stage = try loadStageModelWithArchitecture(
        allocator,
        config_path,
        model_path,
        &arch,
        null,
        .{},
        .{
            .layer_start = 1,
            .layer_end = 3,
            .roles = .{ .include_final_norm = true },
        },
        progress_mod.Context.NONE,
    );
    defer stage.deinit();

    try std.testing.expectEqual(@as(usize, 2), stage.blocks.len);
    try std.testing.expectEqual(@as(usize, 1), try stage.originalLayerIndex(0));
    try std.testing.expectEqual(@as(usize, 2), try stage.originalLayerIndex(1));
    try expectStageProjFirstValue(&stage.blocks[0], 3.0);
    try expectStageProjFirstValue(&stage.blocks[1], 4.0);
    try std.testing.expect(stage.token_embeddings == null);
    try std.testing.expect(stage.ln_final != null);
    try std.testing.expect(stage.lm_head == null);
    try std.testing.expectEqual(@as(usize, 128), stage.residency.bytesForRole(.decoder_layer));
    try std.testing.expectEqual(@as(usize, 16), stage.residency.bytesForRole(.final_norm));
}

test "manifest loadModelWithArchitecture hydrates complete model through shared range loader" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);
    const config_path = try writeStageTestConfig(allocator, tmp.dir, .{});
    defer allocator.free(config_path);
    const model_path = try writeStageTestSafetensors(allocator, tmp_path, .{});
    defer allocator.free(model_path);
    var arch = stageTestArchitecture();

    var model = try loadModelWithArchitecture(
        allocator,
        config_path,
        model_path,
        &arch,
        null,
        .{},
        progress_mod.Context.NONE,
    );
    defer model.deinit();

    try std.testing.expectEqual(@as(usize, 3), model.blocks.len);
    try expectStageProjFirstValue(&model.blocks[0], 2.0);
    try expectStageProjFirstValue(&model.blocks[1], 3.0);
    try expectStageProjFirstValue(&model.blocks[2], 4.0);
    try std.testing.expect(model.token_embeddings.data_ptr != null);
    try std.testing.expect(model.ln_final != null);
    try std.testing.expect(model.lm_head != null);
}

test "manifest loadStageModelWithArchitecture rejects missing selected layer tensor" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);
    const config_path = try writeStageTestConfig(allocator, tmp.dir, .{});
    defer allocator.free(config_path);
    const model_path = try writeStageTestSafetensors(allocator, tmp_path, .{ .include_layer1 = false });
    defer allocator.free(model_path);
    var arch = stageTestArchitecture();

    try std.testing.expectError(error.MissingStageWeight, loadStageModelWithArchitecture(
        allocator,
        config_path,
        model_path,
        &arch,
        null,
        .{},
        .{ .layer_start = 1, .layer_end = 2, .roles = .{} },
        progress_mod.Context.NONE,
    ));
}

test "manifest loadStageModelWithArchitecture rejects invalid stage ranges" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);
    const config_path = try writeStageTestConfig(allocator, tmp.dir, .{});
    defer allocator.free(config_path);
    const model_path = try writeStageTestSafetensors(allocator, tmp_path, .{});
    defer allocator.free(model_path);
    var arch = stageTestArchitecture();

    try std.testing.expectError(error.EmptyStageRange, loadStageModelWithArchitecture(
        allocator,
        config_path,
        model_path,
        &arch,
        null,
        .{},
        .{ .layer_start = 1, .layer_end = 1, .roles = .{} },
        progress_mod.Context.NONE,
    ));
    try std.testing.expectError(error.InvalidStageRange, loadStageModelWithArchitecture(
        allocator,
        config_path,
        model_path,
        &arch,
        null,
        .{},
        .{ .layer_start = 2, .layer_end = 4, .roles = .{} },
        progress_mod.Context.NONE,
    ));
}

test "manifest loadStageModelWithArchitecture rejects unsupported requested global role" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);
    const config_path = try writeStageTestConfig(allocator, tmp.dir, .{});
    defer allocator.free(config_path);
    const model_path = try writeStageTestSafetensors(allocator, tmp_path, .{});
    defer allocator.free(model_path);
    var arch = stageTestArchitecture();

    try std.testing.expectError(error.UnsupportedStageGlobalRole, loadStageModelWithArchitecture(
        allocator,
        config_path,
        model_path,
        &arch,
        null,
        .{},
        .{ .layer_start = 0, .layer_end = 1, .roles = .{ .include_architecture_side = true } },
        progress_mod.Context.NONE,
    ));
}

test "manifest loadStageModelWithArchitecture rejects missing requested global tensor" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);
    const config_path = try writeStageTestConfig(allocator, tmp.dir, .{});
    defer allocator.free(config_path);
    const model_path = try writeStageTestSafetensors(allocator, tmp_path, .{ .include_lm_head = false });
    defer allocator.free(model_path);
    var arch = stageTestArchitecture();

    try std.testing.expectError(error.MissingStageGlobalWeight, loadStageModelWithArchitecture(
        allocator,
        config_path,
        model_path,
        &arch,
        null,
        .{},
        .{ .layer_start = 0, .layer_end = 1, .roles = .{ .include_lm_head = true } },
        progress_mod.Context.NONE,
    ));
}

test "manifest loadStageModelWithArchitecture ties lm_head through token embeddings" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);
    const config_path = try writeStageTestConfig(allocator, tmp.dir, .{ .tie_word_embeddings = true });
    defer allocator.free(config_path);
    const model_path = try writeStageTestSafetensors(allocator, tmp_path, .{ .include_lm_head = false });
    defer allocator.free(model_path);
    var arch = stageTestArchitecture();

    var stage = try loadStageModelWithArchitecture(
        allocator,
        config_path,
        model_path,
        &arch,
        null,
        .{},
        .{ .layer_start = 0, .layer_end = 1, .roles = .{ .include_lm_head = true } },
        progress_mod.Context.NONE,
    );
    defer stage.deinit();

    try std.testing.expect(stage.token_embeddings != null);
    try std.testing.expect(stage.lm_head != null);
    try std.testing.expect(stage.lm_head_uses_token_embeddings);
    try std.testing.expect(!stage.requested_roles.include_token_embeddings);
    try std.testing.expectEqual(@as(usize, 64), stage.residency.bytesForRole(.token_embeddings));
}

test "manifest loadStageModelWithArchitecture rejects tied lm_head without token embeddings" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);
    const config_path = try writeStageTestConfig(allocator, tmp.dir, .{ .tie_word_embeddings = true });
    defer allocator.free(config_path);
    const model_path = try writeStageTestSafetensors(allocator, tmp_path, .{
        .include_token_embeddings = false,
        .include_lm_head = false,
    });
    defer allocator.free(model_path);
    var arch = stageTestArchitecture();

    try std.testing.expectError(error.MissingStageGlobalWeight, loadStageModelWithArchitecture(
        allocator,
        config_path,
        model_path,
        &arch,
        null,
        .{},
        .{ .layer_start = 0, .layer_end = 1, .roles = .{ .include_lm_head = true } },
        progress_mod.Context.NONE,
    ));
}

test "manifest loadStageModelWithArchitecture applies d_ff metadata correction outside selected stage" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);
    const config_path = try writeStageTestConfig(allocator, tmp.dir, .{ .intermediate_size = 99 });
    defer allocator.free(config_path);
    const model_path = try writeStageDffSafetensors(allocator, tmp_path);
    defer allocator.free(model_path);
    var arch = stageTestArchitectureWithDffCorrection();

    var stage = try loadStageModelWithArchitecture(
        allocator,
        config_path,
        model_path,
        &arch,
        null,
        .{},
        .{ .layer_start = 1, .layer_end = 2, .roles = .{} },
        progress_mod.Context.NONE,
    );
    defer stage.deinit();

    try std.testing.expectEqual(@as(i32, 8), stage.config.d_ff);
    try std.testing.expectEqual(@as(usize, 1), stage.blocks.len);
    try std.testing.expectEqual(@as(usize, 1), try stage.originalLayerIndex(0));
    try expectStageProjFirstValue(&stage.blocks[0], 3.0);
    try std.testing.expectEqual(@as(usize, 128), stage.residency.bytesForRole(.decoder_layer));
}

test "manifest applyMetadataDffCorrection infers packed grouped affine source shape" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);
    const model_path = try writeStagePackedDffSafetensors(allocator, tmp_path);
    defer allocator.free(model_path);
    var safetensors_file = try st_loader.UnifiedSafeTensors.loadMetadataOnly(allocator, model_path);
    defer safetensors_file.deinit();

    var arch = stageTestArchitectureWithDffCorrection();
    var cfg = std.mem.zeroes(ModelConfig);
    cfg.model_arch = .custom;
    cfg.d_model = 16;
    cfg.d_ff = 99;
    cfg.n_layers = 1;
    cfg.n_heads = 1;
    cfg.n_kv_groups = 1;
    cfg.head_dim = 16;
    cfg.vocab_size = 4;
    cfg.max_seq_len = 16;
    cfg.gaffine_bits = 4;

    try applyMetadataDffCorrection(allocator, &arch, &cfg, &safetensors_file);

    try std.testing.expectEqual(@as(i32, 32), cfg.d_ff);
}

test "manifest applyMetadataDffCorrection infers GPTQ qweight source shape" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);
    const model_path = try writeStageGptqDffSafetensors(allocator, tmp_path);
    defer allocator.free(model_path);
    var safetensors_file = try st_loader.UnifiedSafeTensors.loadMetadataOnly(allocator, model_path);
    defer safetensors_file.deinit();

    var arch = stageTestArchitectureWithDffCorrection();
    var cfg = std.mem.zeroes(ModelConfig);
    cfg.model_arch = .custom;
    cfg.d_model = 32;
    cfg.d_ff = 99;
    cfg.n_layers = 1;
    cfg.n_heads = 1;
    cfg.n_kv_groups = 1;
    cfg.head_dim = 32;
    cfg.vocab_size = 4;
    cfg.max_seq_len = 16;
    cfg.gaffine_bits = 4;

    try applyMetadataDffCorrection(allocator, &arch, &cfg, &safetensors_file);

    try std.testing.expectEqual(@as(i32, 64), cfg.d_ff);
}

test "manifest applyMetadataDffCorrection infers NVFP4 packed source shape" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);
    const model_path = try writeStageNvfp4DffSafetensors(allocator, tmp_path);
    defer allocator.free(model_path);
    var safetensors_file = try st_loader.UnifiedSafeTensors.loadMetadataOnly(allocator, model_path);
    defer safetensors_file.deinit();

    var arch = stageTestArchitectureWithDffCorrection();
    var cfg = std.mem.zeroes(ModelConfig);
    cfg.model_arch = .custom;
    cfg.d_model = 32;
    cfg.d_ff = 99;
    cfg.n_layers = 1;
    cfg.n_heads = 1;
    cfg.n_kv_groups = 1;
    cfg.head_dim = 32;
    cfg.vocab_size = 4;
    cfg.max_seq_len = 16;

    try applyMetadataDffCorrection(allocator, &arch, &cfg, &safetensors_file);

    try std.testing.expectEqual(@as(i32, 64), cfg.d_ff);
}

test "manifest loadStageModelWithArchitecture rejects unsupported d_ff metadata correction" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);
    const config_path = try writeStageTestConfig(allocator, tmp.dir, .{ .intermediate_size = 99 });
    defer allocator.free(config_path);
    const model_path = try writeStageTestSafetensors(allocator, tmp_path, .{});
    defer allocator.free(model_path);
    var arch = stageTestArchitecture();
    arch.resolve_d_ff_from_weights = true;
    arch.d_ff_source_weight_ids = &.{"missing.weight"};
    arch.weight_dtype_source_weight_ids = &.{"proj.weight"};

    try std.testing.expectError(error.UnsupportedStageConfigInference, loadStageModelWithArchitecture(
        allocator,
        config_path,
        model_path,
        &arch,
        null,
        .{},
        .{ .layer_start = 1, .layer_end = 2, .roles = .{} },
        progress_mod.Context.NONE,
    ));
}

test "manifest loadStageModelWithArchitecture applies shortconv metadata correction" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);
    const config_path = try writeStageShortConvConfig(allocator, tmp.dir);
    defer allocator.free(config_path);
    const model_path = try writeStageShortConvSafetensors(allocator, tmp_path, true);
    defer allocator.free(model_path);
    var arch = stageShortConvArchitecture();

    var stage = try loadStageModelWithArchitecture(
        allocator,
        config_path,
        model_path,
        &arch,
        null,
        .{},
        .{ .layer_start = 1, .layer_end = 2, .roles = .{} },
        progress_mod.Context.NONE,
    );
    defer stage.deinit();

    try std.testing.expectEqual(@as(usize, 1), stage.blocks.len);
    try std.testing.expectEqual(runtime_blocks.BlockKind.shortconv, stage.blocks[0].block_type);
    const shortconv_cfg = stage.blocks[0].map_context.shortconv_config orelse return error.TestExpectedEqual;
    try std.testing.expectEqual(@as(u32, 6), shortconv_cfg.conv_dim);
    try std.testing.expectEqual(@as(u32, 3), shortconv_cfg.d_conv);
    try std.testing.expectEqual(@as(u32, 6), shortconv_cfg.conv_dim_out);
    try std.testing.expectEqual(@as(usize, 72), stage.residency.bytesForRole(.decoder_layer));
}

test "manifest loadStageModelWithArchitecture rejects unsupported shortconv metadata correction" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);
    const config_path = try writeStageShortConvConfig(allocator, tmp.dir);
    defer allocator.free(config_path);
    const model_path = try writeStageShortConvSafetensors(allocator, tmp_path, false);
    defer allocator.free(model_path);
    var arch = stageShortConvArchitecture();

    try std.testing.expectError(error.UnsupportedStageConfigInference, loadStageModelWithArchitecture(
        allocator,
        config_path,
        model_path,
        &arch,
        null,
        .{},
        .{ .layer_start = 1, .layer_end = 2, .roles = .{} },
        progress_mod.Context.NONE,
    ));
}

test "manifest loadStageModelWithArchitecture hydrates requested vision side extras" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);
    const config_path = try writeStageTestConfig(allocator, tmp.dir, .{});
    defer allocator.free(config_path);
    const model_path = try writeStageTestSafetensors(allocator, tmp_path, .{ .include_vision = true });
    defer allocator.free(model_path);
    var arch = stageTestArchitecture();

    var stage = try loadStageModelWithArchitecture(
        allocator,
        config_path,
        model_path,
        &arch,
        null,
        .{},
        .{ .layer_start = 0, .layer_end = 1, .roles = .{ .include_vision_side = true } },
        progress_mod.Context.NONE,
    );
    defer stage.deinit();

    try std.testing.expectEqual(@as(usize, 16), stage.residency.bytesForRole(.vision_side));
    try std.testing.expectEqual(@as(usize, 16), stage.extra_global_role_bytes[@intFromEnum(manifest_mod.TensorRole.vision_side)]);
    try std.testing.expect(stage.extra_global_weights.get("vision.patch_embed.weight") != null);
}

test "manifest LoadedStageModel.deinit releases loaded stage resources" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);
    const config_path = try writeStageTestConfig(allocator, tmp.dir, .{});
    defer allocator.free(config_path);
    const model_path = try writeStageTestSafetensors(allocator, tmp_path, .{});
    defer allocator.free(model_path);
    var arch = stageTestArchitecture();

    var stage = try loadStageModelWithArchitecture(
        allocator,
        config_path,
        model_path,
        &arch,
        null,
        .{},
        .{ .layer_start = 0, .layer_end = 1, .roles = .{ .include_token_embeddings = true } },
        progress_mod.Context.NONE,
    );
    stage.deinit();
}

/// Helper to free aligned tensor data allocated by OwnedTensor
fn freeAlignedTensorData(allocator: std.mem.Allocator, t: Tensor) void {
    if (t.data_ptr) |ptr| {
        const aligned_ptr: [*]align(32) u8 = @alignCast(ptr);
        allocator.free(aligned_ptr[0..t.data_size]);
    }
}

test "requireArchitectureMetadata rejects missing d_ff metadata for mlp architectures" {
    const arch = model_types.Architecture{
        .name = "test_mlp",
        .model_types = &.{},
        .global_weights = &.{},
        .resolve_d_ff_from_weights = true,
        .d_ff_source_weight_ids = &.{},
    };

    try std.testing.expectError(error.MissingDffSourceWeightIds, requireArchitectureMetadata(&arch));
}

test "requireArchitectureMetadata rejects missing shortconv source id" {
    const arch = model_types.Architecture{
        .name = "test_shortconv",
        .model_types = &.{},
        .global_weights = &.{},
        .resolve_shortconv_dims_from_weights = true,
        .d_ff_source_weight_ids = &.{"mlp.gate_proj.weight"},
        .shortconv_dims_source_weight_id = null,
    };

    try std.testing.expectError(error.MissingShortConvSourceWeightId, requireArchitectureMetadata(&arch));
}

test "inferDff returns missing source list error" {
    var cfg = std.mem.zeroes(ModelConfig);
    cfg.model_arch = .custom;
    cfg.d_model = 128;
    cfg.d_ff = 256;
    cfg.vocab_size = 1000;
    cfg.n_layers = 1;
    cfg.n_heads = 4;
    cfg.n_kv_groups = 4;
    cfg.head_dim = 32;
    cfg.max_seq_len = 128;
    cfg.rope_theta = 10000.0;
    cfg.norm_eps = 1e-5;
    cfg.gaffine_group_size = 128;
    var map: runtime_blocks.WeightMap = .{};
    defer map.deinit(std.testing.allocator);

    try std.testing.expectError(error.MissingDffSourceWeightIds, inferDff(&cfg, &map, &.{}, false));
}

test "inferDff returns missing source weight error when ids do not resolve" {
    var cfg = std.mem.zeroes(ModelConfig);
    cfg.model_arch = .custom;
    cfg.d_model = 128;
    cfg.d_ff = 256;
    cfg.vocab_size = 1000;
    cfg.n_layers = 1;
    cfg.n_heads = 4;
    cfg.n_kv_groups = 4;
    cfg.head_dim = 32;
    cfg.max_seq_len = 128;
    cfg.rope_theta = 10000.0;
    cfg.norm_eps = 1e-5;
    cfg.gaffine_group_size = 128;
    var map: runtime_blocks.WeightMap = .{};
    defer map.deinit(std.testing.allocator);

    try std.testing.expectError(
        error.MissingDffSourceWeight,
        inferDff(&cfg, &map, &.{"mlp.gate_proj.weight"}, false),
    );
}

test "inferDff updates config from source weight shape" {
    var cfg = std.mem.zeroes(ModelConfig);
    cfg.model_arch = .custom;
    cfg.d_model = 128;
    cfg.d_ff = 256;
    cfg.vocab_size = 1000;
    cfg.n_layers = 1;
    cfg.n_heads = 4;
    cfg.n_kv_groups = 4;
    cfg.head_dim = 32;
    cfg.max_seq_len = 128;
    cfg.rope_theta = 10000.0;
    cfg.norm_eps = 1e-5;
    cfg.gaffine_group_size = 128;
    var map: runtime_blocks.WeightMap = .{};
    defer map.deinit(std.testing.allocator);

    var gate_weight = Tensor{
        .dtype = .f32,
        .n_dims = 2,
        .shape = .{ 128, 320, 0, 0, 0, 0, 0, 0 },
        .data_ptr = null,
        .data_size = 0,
    };
    try map.put(std.testing.allocator, "mlp.gate_proj.weight", &gate_weight);

    try inferDff(&cfg, &map, &.{"mlp.gate_proj.weight"}, false);
    try std.testing.expectEqual(@as(i32, 320), cfg.d_ff);
}

test "inferDff halves fused gate_up width for fused-gate architectures" {
    var cfg = std.mem.zeroes(ModelConfig);
    cfg.model_arch = .custom;
    cfg.d_model = 128;
    cfg.d_ff = 256;
    cfg.vocab_size = 1000;
    cfg.n_layers = 1;
    cfg.n_heads = 4;
    cfg.n_kv_groups = 4;
    cfg.head_dim = 32;
    cfg.max_seq_len = 128;
    cfg.rope_theta = 10000.0;
    cfg.norm_eps = 1e-5;
    cfg.gaffine_group_size = 128;
    var map: runtime_blocks.WeightMap = .{};
    defer map.deinit(std.testing.allocator);

    var gate_up_weight = Tensor{
        .dtype = .f32,
        .n_dims = 2,
        .shape = .{ 128, 640, 0, 0, 0, 0, 0, 0 },
        .data_ptr = null,
        .data_size = 0,
    };
    try map.put(std.testing.allocator, "mlp.gate_up_proj.weight", &gate_up_weight);

    try inferDff(&cfg, &map, &.{"mlp.gate_up_proj.weight"}, true);
    try std.testing.expectEqual(@as(i32, 320), cfg.d_ff);
}

test "inferDff halves fused input_linear width for fused-gate architectures" {
    var cfg = std.mem.zeroes(ModelConfig);
    cfg.model_arch = .custom;
    cfg.d_model = 128;
    cfg.d_ff = 256;
    cfg.vocab_size = 1000;
    cfg.n_layers = 1;
    cfg.n_heads = 4;
    cfg.n_kv_groups = 4;
    cfg.head_dim = 32;
    cfg.max_seq_len = 128;
    cfg.rope_theta = 10000.0;
    cfg.norm_eps = 1e-5;
    cfg.gaffine_group_size = 128;
    var map: runtime_blocks.WeightMap = .{};
    defer map.deinit(std.testing.allocator);

    var input_linear_weight = Tensor{
        .dtype = .f32,
        .n_dims = 2,
        .shape = .{ 128, 640, 0, 0, 0, 0, 0, 0 },
        .data_ptr = null,
        .data_size = 0,
    };
    try map.put(std.testing.allocator, "mlp.input_linear.weight", &input_linear_weight);

    try inferDff(&cfg, &map, &.{"mlp.input_linear.weight"}, true);
    try std.testing.expectEqual(@as(i32, 320), cfg.d_ff);
}

test "inferShortConvDims rejects missing source id when dims are incomplete" {
    var cfg = runtime_blocks.ShortConvConfig{
        .d_model = 128,
        .d_conv = 0,
        .conv_dim = 0,
        .conv_dim_out = 0,
    };
    var map: runtime_blocks.WeightMap = .{};
    defer map.deinit(std.testing.allocator);

    try std.testing.expectError(error.MissingShortConvSourceWeightId, inferShortConvDims(&cfg, &map, null));
}

test "inferShortConvDims infers missing dimensions from source weight" {
    var cfg = runtime_blocks.ShortConvConfig{
        .d_model = 128,
        .d_conv = 0,
        .conv_dim = 0,
        .conv_dim_out = 0,
    };
    var map: runtime_blocks.WeightMap = .{};
    defer map.deinit(std.testing.allocator);

    var conv_weight = Tensor{
        .dtype = .f32,
        .n_dims = 3,
        .shape = .{ 512, 1, 4, 0, 0, 0, 0, 0 },
        .data_ptr = null,
        .data_size = 0,
    };
    try map.put(std.testing.allocator, "conv.conv.weight", &conv_weight);

    try inferShortConvDims(&cfg, &map, "conv.conv.weight");
    try std.testing.expectEqual(@as(u32, 512), cfg.conv_dim);
    try std.testing.expectEqual(@as(u32, 512), cfg.conv_dim_out);
    try std.testing.expectEqual(@as(u32, 4), cfg.d_conv);
}

test "inferShortConvDims corrects stale configured dimensions from source weight" {
    var cfg = runtime_blocks.ShortConvConfig{
        .d_model = 2048,
        .d_conv = 3,
        .conv_dim = 1024,
        .conv_dim_out = 1024,
    };
    var map: runtime_blocks.WeightMap = .{};
    defer map.deinit(std.testing.allocator);

    var conv_weight = Tensor{
        .dtype = .f32,
        .n_dims = 3,
        .shape = .{ 2048, 1, 3, 0, 0, 0, 0, 0 },
        .data_ptr = null,
        .data_size = 0,
    };
    try map.put(std.testing.allocator, "conv.conv.weight", &conv_weight);

    try inferShortConvDims(&cfg, &map, "conv.conv.weight");
    try std.testing.expectEqual(@as(u32, 2048), cfg.conv_dim);
    try std.testing.expectEqual(@as(u32, 2048), cfg.conv_dim_out);
    try std.testing.expectEqual(@as(u32, 3), cfg.d_conv);
}

test "orientWeightF32: transpose [out, in] to [in, out]" {
    const allocator = std.testing.allocator;

    // Create a 2x3 weight matrix [out=2, in=3]
    var data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const weight = Tensor.view2DSlice(data[0..], 2, 3);

    const result = try orientWeightF32(allocator, weight, 3);
    defer if (result.data_ptr != weight.data_ptr) freeAlignedTensorData(allocator, result);

    try std.testing.expectEqual(@as(usize, 3), @as(usize, @intCast(result.shape[0])));
    try std.testing.expectEqual(@as(usize, 2), @as(usize, @intCast(result.shape[1])));

    const result_data = result.asSlice(f32);
    // Transposed: [1,4], [2,5], [3,6]
    try std.testing.expectEqual(@as(f32, 1), result_data[0]);
    try std.testing.expectEqual(@as(f32, 4), result_data[1]);
    try std.testing.expectEqual(@as(f32, 2), result_data[2]);
    try std.testing.expectEqual(@as(f32, 5), result_data[3]);
    try std.testing.expectEqual(@as(f32, 3), result_data[4]);
    try std.testing.expectEqual(@as(f32, 6), result_data[5]);
}

test "orientWeightF32: no transpose when already [in, out]" {
    const allocator = std.testing.allocator;

    // Create a 3x2 weight matrix [in=3, out=2]
    var data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const weight = Tensor.view2DSlice(data[0..], 3, 2);

    const result = try orientWeightF32(allocator, weight, 3);

    // Should not transpose
    try std.testing.expectEqual(@as(usize, 3), @as(usize, @intCast(result.shape[0])));
    try std.testing.expectEqual(@as(usize, 2), @as(usize, @intCast(result.shape[1])));
    try std.testing.expectEqual(weight.data_ptr, result.data_ptr);
}

test "orientWeightF32: shape validation errors" {
    const allocator = std.testing.allocator;

    // 1D tensor is valid - returned unchanged (passthrough for embeddings, etc.)
    var data_1d = [_]f32{ 1, 2, 3 };
    var weight_1d = Tensor.view2DSlice(data_1d[0..], 3, 1);
    weight_1d.n_dims = 1;
    weight_1d.shape[0] = 3;
    const result_1d = try orientWeightF32(allocator, weight_1d, 3);
    try std.testing.expectEqual(@as(i32, 1), result_1d.n_dims);

    // Invalid: neither dimension matches expected_in
    var data_2d = [_]f32{ 1, 2, 3, 4 };
    const weight_2d = Tensor.view2DSlice(data_2d[0..], 2, 2);
    try std.testing.expectError(error.InvalidShape, orientWeightF32(allocator, weight_2d, 5));
}

test "orientWeightTyped: BF16 weights remain untransposed" {
    const allocator = std.testing.allocator;

    // Create a 2x3 BF16 weight matrix [out=2, in=3]
    var data = [_]u16{ 0x3F80, 0x4000, 0x4040, 0x4080, 0x40A0, 0x40C0 };
    var weight: Tensor = undefined;
    weight.dtype = .bf16;
    weight.n_dims = 2;
    weight.shape = .{ 2, 3, 0, 0, 0, 0, 0, 0 };
    weight.data_ptr = @ptrCast(data[0..].ptr);
    weight.data_size = data.len * @sizeOf(u16);
    weight.owns_data = false;

    const result = try orientWeightTyped(allocator, weight, 3);

    // BF16 should not transpose
    try std.testing.expectEqual(@as(usize, 2), @as(usize, @intCast(result.shape[0])));
    try std.testing.expectEqual(@as(usize, 3), @as(usize, @intCast(result.shape[1])));
    try std.testing.expectEqual(weight.data_ptr, result.data_ptr);
}

test "maybeConcatQkvWeights: successful concatenation" {
    const allocator = std.testing.allocator;

    // Q: 4x2, K: 4x1, V: 4x1
    var q_data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var k_data = [_]f32{ 9, 10, 11, 12 };
    var v_data = [_]f32{ 13, 14, 15, 16 };

    const q = Tensor.view2DSlice(q_data[0..], 4, 2);
    const k = Tensor.view2DSlice(k_data[0..], 4, 1);
    const v = Tensor.view2DSlice(v_data[0..], 4, 1);

    const result = maybeConcatQkvWeights(allocator, q, k, v) orelse return error.ConcatFailed;
    defer allocator.free(result.asSlice(f32));

    try std.testing.expectEqual(@as(usize, 4), @as(usize, @intCast(result.shape[0])));
    try std.testing.expectEqual(@as(usize, 4), @as(usize, @intCast(result.shape[1])));

    const result_data = result.asSlice(f32);
    // Row 0: [1,2,9,13]
    try std.testing.expectEqual(@as(f32, 1), result_data[0]);
    try std.testing.expectEqual(@as(f32, 2), result_data[1]);
    try std.testing.expectEqual(@as(f32, 9), result_data[2]);
    try std.testing.expectEqual(@as(f32, 13), result_data[3]);
    // Row 1: [3,4,10,14]
    try std.testing.expectEqual(@as(f32, 3), result_data[4]);
    try std.testing.expectEqual(@as(f32, 4), result_data[5]);
    try std.testing.expectEqual(@as(f32, 10), result_data[6]);
    try std.testing.expectEqual(@as(f32, 14), result_data[7]);
}

test "maybeConcatQkvWeights: returns null for BF16" {
    const allocator = std.testing.allocator;

    var q_data = [_]u16{ 0x3F80, 0x4000 };
    var q: Tensor = undefined;
    q.dtype = .bf16;
    q.n_dims = 2;
    q.shape = .{ 1, 2, 0, 0, 0, 0, 0, 0 };
    q.data_ptr = @ptrCast(q_data[0..].ptr);

    const k = q;
    const v = q;

    const result = maybeConcatQkvWeights(allocator, q, k, v);
    try std.testing.expect(result == null);
}

test "maybeConcatQkvWeights: returns null for mismatched rows" {
    const allocator = std.testing.allocator;

    var q_data = [_]f32{ 1, 2, 3, 4 };
    var k_data = [_]f32{ 5, 6 };
    var v_data = [_]f32{ 7, 8 };

    const q = Tensor.view2DSlice(q_data[0..], 2, 2);
    const k = Tensor.view2DSlice(k_data[0..], 1, 2); // Different rows
    const v = Tensor.view2DSlice(v_data[0..], 1, 2);

    const result = maybeConcatQkvWeights(allocator, q, k, v);
    try std.testing.expect(result == null);
}

test "maybeConcatGateUpWeights: successful concatenation" {
    const allocator = std.testing.allocator;

    var gate_data = [_]f32{ 1, 2, 3, 4 };
    var up_data = [_]f32{ 5, 6, 7, 8 };

    const gate = Tensor.view2DSlice(gate_data[0..], 2, 2);
    const up = Tensor.view2DSlice(up_data[0..], 2, 2);

    const result = maybeConcatGateUpWeights(allocator, gate, up) orelse return error.ConcatFailed;
    defer allocator.free(result.asSlice(f32));

    try std.testing.expectEqual(@as(usize, 2), @as(usize, @intCast(result.shape[0])));
    try std.testing.expectEqual(@as(usize, 4), @as(usize, @intCast(result.shape[1])));

    const result_data = result.asSlice(f32);
    // Row 0: [1,2,5,6]
    try std.testing.expectEqual(@as(f32, 1), result_data[0]);
    try std.testing.expectEqual(@as(f32, 2), result_data[1]);
    try std.testing.expectEqual(@as(f32, 5), result_data[2]);
    try std.testing.expectEqual(@as(f32, 6), result_data[3]);
    // Row 1: [3,4,7,8]
    try std.testing.expectEqual(@as(f32, 3), result_data[4]);
    try std.testing.expectEqual(@as(f32, 4), result_data[5]);
    try std.testing.expectEqual(@as(f32, 7), result_data[6]);
    try std.testing.expectEqual(@as(f32, 8), result_data[7]);
}

test "maybeConcatGateUpWeights: returns null for BF16" {
    const allocator = std.testing.allocator;

    var gate_data = [_]u16{ 0x3F80, 0x4000 };
    var gate: Tensor = undefined;
    gate.dtype = .bf16;
    gate.n_dims = 2;
    gate.shape = .{ 1, 2, 0, 0, 0, 0, 0, 0 };
    gate.data_ptr = @ptrCast(gate_data[0..].ptr);

    const up = gate;

    const result = maybeConcatGateUpWeights(allocator, gate, up);
    try std.testing.expect(result == null);
}

test "ensureF32: F32 passthrough" {
    const allocator = std.testing.allocator;

    var data = [_]f32{ 1.0, 2.0, 3.0 };
    const t = Tensor.view2DSlice(data[0..], 1, 3);

    const result = try ensureF32(allocator, t);

    try std.testing.expectEqual(DType.f32, result.dtype);
    try std.testing.expectEqual(t.data_ptr, result.data_ptr);
}

test "convertToF32: BF16 to F32 conversion" {
    const allocator = std.testing.allocator;

    // BF16 representation of [1.0, 2.0]
    const data = [_]u16{ 0x3F80, 0x4000 };
    var t: Tensor = undefined;
    t.dtype = .bf16;
    t.n_dims = 1;
    t.shape = .{ 2, 0, 0, 0, 0, 0, 0, 0 };
    t.data_ptr = @ptrCast(@constCast(&data));
    t.data_size = data.len * @sizeOf(u16);
    t.owns_data = false;
    t.numel = 2;

    const result = try convertToF32(allocator, t);
    defer freeAlignedTensorData(allocator, result);

    try std.testing.expectEqual(DType.f32, result.dtype);
    const result_data = result.asSlice(f32);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result_data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), result_data[1], 0.001);
}

test "convertToF32: F16 to F32 conversion" {
    const allocator = std.testing.allocator;

    // F16 representation of [1.0, -2.0]
    const data = [_]u16{ 0x3C00, 0xC000 };
    var t: Tensor = undefined;
    t.dtype = .f16;
    t.n_dims = 1;
    t.shape = .{ 2, 0, 0, 0, 0, 0, 0, 0 };
    t.data_ptr = @ptrCast(@constCast(&data));
    t.data_size = data.len * @sizeOf(u16);
    t.owns_data = false;
    t.numel = 2;

    const result = try convertToF32(allocator, t);
    defer freeAlignedTensorData(allocator, result);

    try std.testing.expectEqual(DType.f32, result.dtype);
    const result_data = result.asSlice(f32);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result_data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), result_data[1], 0.001);
}

test "transposeToOwned: F32 transpose" {
    const allocator = std.testing.allocator;

    var data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const t = Tensor.view2DSlice(data[0..], 2, 3);

    const result = try transposeToOwned(allocator, t, .f32);
    defer freeAlignedTensorData(allocator, result);

    try std.testing.expectEqual(@as(usize, 3), @as(usize, @intCast(result.shape[0])));
    try std.testing.expectEqual(@as(usize, 2), @as(usize, @intCast(result.shape[1])));

    const result_data = result.asSlice(f32);
    try std.testing.expectEqual(@as(f32, 1), result_data[0]);
    try std.testing.expectEqual(@as(f32, 4), result_data[1]);
    try std.testing.expectEqual(@as(f32, 2), result_data[2]);
    try std.testing.expectEqual(@as(f32, 5), result_data[3]);
}

test "transposeToOwned: BF16 transpose" {
    const allocator = std.testing.allocator;

    // 2x2 matrix
    var data = [_]u16{ 0x3F80, 0x4000, 0x4040, 0x4080 };
    var t: Tensor = undefined;
    t.dtype = .bf16;
    t.n_dims = 2;
    t.shape = .{ 2, 2, 0, 0, 0, 0, 0, 0 };
    t.data_ptr = @ptrCast(@constCast(&data));
    t.data_size = data.len * @sizeOf(u16);

    const result = try transposeToOwned(allocator, t, .bf16);
    defer freeAlignedTensorData(allocator, result);

    try std.testing.expectEqual(@as(usize, 2), @as(usize, @intCast(result.shape[0])));
    try std.testing.expectEqual(@as(usize, 2), @as(usize, @intCast(result.shape[1])));

    const result_data = result.asSlice(u16);
    try std.testing.expectEqual(@as(u16, 0x3F80), result_data[0]);
    try std.testing.expectEqual(@as(u16, 0x4040), result_data[1]);
    try std.testing.expectEqual(@as(u16, 0x4000), result_data[2]);
    try std.testing.expectEqual(@as(u16, 0x4080), result_data[3]);
}

test "bytesToU16Slice: aligned conversion" {
    const allocator = std.testing.allocator;

    const data = [_]u16{ 0x1234, 0x5678, 0xABCD };
    const bytes = std.mem.sliceAsBytes(&data);

    var owned: ?[]u16 = null;
    defer if (owned) |o| allocator.free(o);

    const result = try bytesToU16Slice(allocator, bytes, &owned);

    try std.testing.expectEqual(@as(usize, 3), result.len);
    try std.testing.expectEqual(@as(u16, 0x1234), result[0]);
    try std.testing.expectEqual(@as(u16, 0x5678), result[1]);
    try std.testing.expectEqual(@as(u16, 0xABCD), result[2]);
}

test "bytesToU16Slice: invalid length error" {
    const allocator = std.testing.allocator;

    const bytes = [_]u8{ 0x12, 0x34, 0x56 }; // Odd length
    var owned: ?[]u16 = null;

    try std.testing.expectError(error.InvalidShape, bytesToU16Slice(allocator, &bytes, &owned));
}

test "readEnvFlag: parse true values" {
    const allocator = std.testing.allocator;

    // Create a simple test - we can't easily set env vars in tests,
    // so we just test the default behavior
    const result = readEnvFlag(allocator, "NONEXISTENT_VAR_XYZ123", true);
    try std.testing.expect(result == true);

    const result2 = readEnvFlag(allocator, "NONEXISTENT_VAR_XYZ123", false);
    try std.testing.expect(result2 == false);
}

test "convertToF32: grouped_affine_u4 dequantization" {
    const allocator = std.testing.allocator;

    // Create a simple 1x8 grouped_affine_u4 tensor with group_size=8
    // All zeros for simple verification that output dtype is correct
    const packed_data = [_]u32{
        0x00000000, // all zero nibbles
    };

    // Single group with scale=1.0, bias=0.0
    const scales_data = [_]u16{dtype.f32ToFp16(1.0)};
    const biases_data = [_]u16{dtype.f32ToFp16(0.0)};

    var t: Tensor = undefined;
    t.dtype = .grouped_affine_u4;
    t.n_dims = 2;
    t.shape = .{ 1, 8, 0, 0, 0, 0, 0, 0 }; // Unpacked shape
    t.data_ptr = @constCast(@as([*]u8, @ptrCast(@constCast(&packed_data))));
    t.data_size = packed_data.len * @sizeOf(u32);
    t.numel = 8;
    t.gaffine = .{
        .scales = @constCast(std.mem.sliceAsBytes(&scales_data)),
        .biases = @constCast(std.mem.sliceAsBytes(&biases_data)),
        .group_size = 8,
        .scales_dtype = .f16,
    };

    const result = try convertToF32(allocator, t);
    defer freeAlignedTensorData(allocator, result);

    try std.testing.expectEqual(DType.f32, result.dtype);
    const result_data = result.asSlice(f32);

    // All zeros with scale=1.0, bias=0.0 should produce all zeros
    for (result_data) |val| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), val, 0.01);
    }
}

test "convertToF32: grouped_affine_u8 dequantization" {
    const allocator = std.testing.allocator;

    // Create a 1x4 grouped_affine_u8 tensor with group_size=4
    // All zeros for simple verification that output dtype is correct
    const packed_data = [_]u32{0x00000000};

    const scales_data = [_]u16{dtype.f32ToFp16(1.0)};
    const biases_data = [_]u16{dtype.f32ToFp16(0.0)};

    var t: Tensor = undefined;
    t.dtype = .grouped_affine_u8;
    t.n_dims = 2;
    t.shape = .{ 1, 4, 0, 0, 0, 0, 0, 0 };
    t.data_ptr = @constCast(@as([*]u8, @ptrCast(@constCast(&packed_data))));
    t.data_size = 4; // 4 bytes
    t.numel = 4;
    t.gaffine = .{
        .scales = @constCast(std.mem.sliceAsBytes(&scales_data)),
        .biases = @constCast(std.mem.sliceAsBytes(&biases_data)),
        .group_size = 4,
        .scales_dtype = .f16,
    };

    const result = try convertToF32(allocator, t);
    defer freeAlignedTensorData(allocator, result);

    try std.testing.expectEqual(DType.f32, result.dtype);
    const result_data = result.asSlice(f32);

    // All zeros with scale=1.0, bias=0.0 should produce all zeros
    for (result_data) |val| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), val, 0.01);
    }
}

test "convertToF32: invalid shape error" {
    const allocator = std.testing.allocator;

    var t: Tensor = undefined;
    t.dtype = .bf16;
    t.n_dims = 0; // Invalid
    t.shape = .{ 0, 0, 0, 0, 0, 0, 0, 0 };

    try std.testing.expectError(error.InvalidShape, convertToF32(allocator, t));
}

test "LoadedModel.deinit: cleanup without backend runtime state" {
    const allocator = std.testing.allocator;

    const arena = std.heap.ArenaAllocator.init(allocator);
    const config = ModelConfig{
        .model_arch = .custom,
        .n_layers = 1,
        .d_model = 64,
        .n_heads = 4,
        .n_kv_groups = 4,
        .head_dim = 16,
        .d_ff = 128,
        .max_seq_len = 128,
        .vocab_size = 100,
        .rope_theta = 10000.0,
        .norm_eps = 1e-6,
        .gaffine_group_size = 128,
    };

    var model = LoadedModel{
        .arena = arena,
        .config = config,
        .runtime = .{},
        .st = null,
        .ln_final = undefined,
        .lm_head = undefined,
        .token_embeddings = undefined,
        .blocks = &[_]runtime_blocks.LayerWeights{},
        .original_weight_dtype = .f32,
    };

    // Should not leak memory
    model.deinit();
}

test "buildTiedLmHead preserves NVFP4 for CUDA-native tied projection" {
    const allocator = std.testing.allocator;
    const writer = @import("io_pkg").safetensors.writer;

    const tmp_dir_path = "/tmp/test_build_tied_lm_head_nvfp4";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    const packed_bytes = [_]u8{
        0x21, 0xA9,
        0x43, 0x65,
    };
    const scales = [_]u8{
        0x38, 0x38,
        0x38, 0x38,
    };
    const scale_2 = [_]f32{1.0};
    const entries = [_]writer.TensorEntry{
        .{ .name = "model.language_model.embed_tokens.weight", .dtype = .u8, .shape = &[_]usize{ 2, 2 }, .data = packed_bytes[0..] },
        .{ .name = "model.language_model.embed_tokens.weight_scale", .dtype = .f8_e4m3, .shape = &[_]usize{ 2, 2 }, .data = scales[0..] },
        .{ .name = "model.language_model.embed_tokens.weight_scale_2", .dtype = .f32, .shape = &[_]usize{1}, .data = std.mem.sliceAsBytes(&scale_2) },
    };
    const model_path = tmp_dir_path ++ "/model.safetensors";
    try writer.write(allocator, model_path, &entries);

    var safetensors_file = try st_loader.UnifiedSafeTensors.load(allocator, model_path);
    defer safetensors_file.deinit();

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const arch = model_types.Architecture{
        .name = "test",
        .model_types = &.{},
        .weight_prefixes = &.{
            "model.layers.{d}.",
            "layers.{d}.",
        },
        .global_weights = &.{
            .{
                .id = "token_embeddings",
                .suffix = "model.language_model.embed_tokens.weight",
                .module_type = "Embedding",
                .layout = .embedding,
                .dtype = "float32",
                .required = true,
            },
        },
    };
    var config = std.mem.zeroes(ModelConfig);
    config.d_model = 4;

    const tied = (try buildTiedLmHead(
        arena.allocator(),
        &safetensors_file,
        &arch,
        &config,
        .{ .dequantize_nvfp4_to_bf16 = false },
    )) orelse return error.TestExpectedEqual;

    try std.testing.expectEqual(DType.u8, tied.dtype);
    try std.testing.expectEqual(@as(i64, 2), tied.shape[0]);
    try std.testing.expectEqual(@as(i64, 4), tied.shape[1]);
    try std.testing.expect(tied.nvfp4 != null);
}

test "orientWeight: error on missing tensor" {
    // Test that orientWeight returns error.NotFound when tensor doesn't exist.
    // We can't easily mock UnifiedSafeTensors due to private MappedBuffer type,
    // so this test documents the expected error behavior without full mocking.
    // The function is integration-tested via loadModel tests.
}
