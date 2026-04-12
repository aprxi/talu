//! Transformer model structure and execution.
//!
//! Assembles transformer layers into a complete model with embeddings,
//! blocks, final normalization, and output projection.

const std = @import("std");
const build_options = @import("build_options");
const log = @import("../../../../log.zig");
const models = @import("../../../../models/root.zig");
const Block = @import("block.zig").Block;
const layer_ops = @import("../../../../models/layer_ops.zig");
const tensor_mod = @import("../../../../tensor.zig");
const dtype_mod = @import("../../../../dtype.zig");
const st_loader = @import("../../../../io/safetensors/root.zig");
const compute = @import("../../../../compute/root.zig");
const cpu_linalg = compute.cpu.linalg;
const cpu_common = compute.cpu.common;
const cpu_rowwise = compute.cpu.rowwise;
const cpu_memory = compute.cpu.memory;
const cpu_activation = compute.cpu.activation;
const inspect = @import("../../../../xray/root.zig");
const kernel_info = inspect.kernel_info;
const perf_estimate = inspect.perf_estimate;
const trace = @import("../../../../xray/trace.zig");
const runtime_contract = @import("../../../runtime_contract/root.zig");
const state_bindings = @import("../state_bindings.zig");
const dump = if (build_options.dump_tensors) @import("../../../../xray/dump/capture.zig") else struct {
    pub fn recordGlobal(_: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) void {}
};
const embedding_kernel = @import("../kernels/embedding.zig");
const attn_kernel = @import("../kernels/attention.zig");
const norm_kernel = @import("../kernels/norm.zig");
const per_layer_branch_kernel = @import("../kernels/per_layer_branch.zig");

const Tensor = tensor_mod.Tensor;
const DType = dtype_mod.DType;
const Attention = attn_kernel.MultiHeadAttention;
const FFNLayer = block_kernels.FfnLayer;
const RMSNorm = norm_kernel.RMSNorm;
const ScratchBuffer = @import("runtime.zig").ScratchBuffer;
const PerfEstimate = perf_estimate.PerfEstimate;
const MatmulFn = cpu_linalg.MatmulFn;
const KernelOp = kernel_info.KernelOp;

const block_kernels = @import("weights.zig");
const cpu_blocks = block_kernels;

const kv_cache = @import("../kernels/kv_cache.zig");
const LayeredBatchedKVCache = kv_cache.LayeredBatchedKVCache;

const LoadedModel = models.LoadedModel;
const ModelConfig = tensor_mod.ModelConfig;
const LayerOp = layer_ops.LayerOp;
const BlockKind = block_kernels.BlockType;

pub fn formatLinearLike(
    writer: anytype,
    weight: *const Tensor,
    bias: ?[]const f32,
    in_features: usize,
    out_features: usize,
) !void {
    const weight_dtype = weight.dtype;
    if (weight_dtype == .grouped_affine_u4) {
        const group_size = if (weight.gaffine) |m| m.group_size else 64;
        try writer.print("QuantizedLinear(in={}, out={}, bits=4, group_size={})", .{
            in_features,
            out_features,
            group_size,
        });
    } else if (weight_dtype == .grouped_affine_u8) {
        const group_size = if (weight.gaffine) |m| m.group_size else 64;
        try writer.print("QuantizedLinear(in={}, out={}, bits=8, group_size={})", .{
            in_features,
            out_features,
            group_size,
        });
    } else {
        const dtype_name: []const u8 = switch (weight_dtype) {
            .f32 => "f32",
            .f16 => "f16",
            .bf16 => "bf16",
            else => "unknown",
        };
        try writer.print("Linear(in={}, out={}, bias={}, dtype={s})", .{
            in_features,
            out_features,
            bias != null,
            dtype_name,
        });
    }
}

pub fn formatRmsNormLike(writer: anytype, dim: usize, eps: f32, weight_offset: f32) !void {
    if (weight_offset != 0.0) {
        try writer.print("RMSNorm(dim={}, eps={e}, weight_offset={d:.1})", .{ dim, eps, weight_offset });
    } else {
        try writer.print("RMSNorm(dim={}, eps={e})", .{ dim, eps });
    }
}

/// Linear transformation: y = x @ W + b
/// Owns a pointer to weight tensor (mmap'd) and optional bias.
pub const Linear = struct {
    weight: *const Tensor,
    bias: ?[]const f32 = null,
    in_features: usize,
    out_features: usize,
    matmul_fn: MatmulFn,
    pub fn init(weight: *const Tensor, bias: ?[]const f32) !Linear {
        const in_features: usize = switch (weight.dtype) {
            .f32 => @intCast(weight.shape[0]),
            else => @intCast(weight.shape[1]),
        };
        const out_features: usize = switch (weight.dtype) {
            .f32 => @intCast(weight.shape[1]),
            else => @intCast(weight.shape[0]),
        };
        const dk = try cpu_linalg.matmulKernel(weight.dtype);
        return .{
            .weight = weight,
            .bias = bias,
            .in_features = in_features,
            .out_features = out_features,
            .matmul_fn = dk.func,
        };
    }

    pub fn initWithDims(weight: *const Tensor, bias: ?[]const f32, in_features: usize, out_features: usize) !Linear {
        const dk = try cpu_linalg.matmulKernel(weight.dtype);
        return .{
            .weight = weight,
            .bias = bias,
            .in_features = in_features,
            .out_features = out_features,
            .matmul_fn = dk.func,
        };
    }

    pub inline fn forward(self: *const Linear, input_tensor: *const Tensor, output_tensor: *Tensor, scratch: *cpu_linalg.MatmulScratch) void {
        const row_count: usize = if (input_tensor.n_dims == 3) @intCast(input_tensor.shape[0] * input_tensor.shape[1]) else @intCast(input_tensor.shape[0]);
        const input_view = Tensor.view2D(input_tensor.data(), row_count, self.in_features);
        var output_view = Tensor.view2DSlice(output_tensor.asSlice(f32), row_count, self.out_features);

        self.matmul_fn(&input_view, self.weight, &output_view, scratch);

        if (self.bias) |bias| {
            cpu_common.addBiasRows(output_tensor.asSlice(f32), bias, row_count, self.out_features);
        }
    }

    pub fn formatKernels(self: *const Linear, writer: anytype, indent: usize) !void {
        const matmul_op = KernelOp{ .matmul = .{
            .m = .seq,
            .k = self.in_features,
            .n = self.out_features,
            .dtype = self.weight.dtype,
            .kernel_name = kernel_info.matmulKernelName(self.weight.dtype),
        } };
        try matmul_op.format(writer, indent);

        if (self.bias != null) {
            const bias_op = KernelOp{ .bias_add = .{ .size = self.out_features } };
            try bias_op.format(writer, indent);
        }
    }

    pub fn describe(self: *const Linear, writer: anytype, indent: usize, show_kernels: bool) !void {
        try writer.writeByteNTimes(' ', indent);
        try self.formatTo(writer);
        try writer.writeAll("\n");

        if (show_kernels) {
            try self.formatKernels(writer, indent + 2);
        }
    }

    pub fn formatTo(self: *const Linear, writer: anytype) !void {
        try formatLinearLike(writer, self.weight, self.bias, self.in_features, self.out_features);
    }
};

/// Token embedding lookup table
pub const Embedding = struct {
    weight: *const Tensor,
    vocab_size: usize,
    embed_dim: usize,
    pub fn init(weight: *const Tensor) Embedding {
        return .{
            .weight = weight,
            .vocab_size = @intCast(weight.shape[0]),
            .embed_dim = @intCast(weight.shape[1]),
        };
    }

    pub fn forward(self: *const Embedding, tokens: []const u32, output_tensor: *Tensor) !void {
        try embedding_kernel.gatherEmbeddings(self.weight, tokens, output_tensor);
    }

    pub fn formatKernels(self: *const Embedding, writer: anytype, indent: usize) !void {
        const gather_op = KernelOp{ .gather = .{
            .vocab_size = self.vocab_size,
            .embed_dim = self.embed_dim,
            .dtype = self.weight.dtype,
        } };
        try gather_op.format(writer, indent);
    }

    pub fn describe(self: *const Embedding, writer: anytype, indent: usize, show_kernels: bool) !void {
        try writer.writeByteNTimes(' ', indent);
        try self.formatTo(writer);
        try writer.writeAll("\n");

        if (show_kernels) {
            try self.formatKernels(writer, indent + 2);
        }
    }

    pub fn formatTo(self: *const Embedding, writer: anytype) !void {
        const dtype = self.weight.dtype;
        if (dtype == .grouped_affine_u4 or dtype == .grouped_affine_u8) {
            const bits: u8 = if (dtype == .grouped_affine_u4) 4 else 8;
            try writer.print("Embedding(vocab={}, dim={}, bits={})", .{
                self.vocab_size, self.embed_dim, bits,
            });
        } else {
            try writer.print("Embedding(vocab={}, dim={})", .{
                self.vocab_size, self.embed_dim,
            });
        }
    }
};

const PerLayerBranchScratch = struct {
    projection: []f32,
    per_layer_input: []f32,
    gated: []f32,
    branch: []f32,
    source_embeddings: []f32,
    row_count: usize,
    hidden_size: usize,
    hidden_size_per_layer_input: usize,

    fn init(
        allocator: std.mem.Allocator,
        row_count: usize,
        hidden_size: usize,
        hidden_size_per_layer_input: usize,
    ) !PerLayerBranchScratch {
        return .{
            .projection = try allocator.alloc(f32, row_count * hidden_size_per_layer_input),
            .per_layer_input = try allocator.alloc(f32, row_count * hidden_size_per_layer_input),
            .gated = try allocator.alloc(f32, row_count * hidden_size_per_layer_input),
            .branch = try allocator.alloc(f32, row_count * hidden_size),
            .source_embeddings = try allocator.alloc(f32, row_count * hidden_size),
            .row_count = row_count,
            .hidden_size = hidden_size,
            .hidden_size_per_layer_input = hidden_size_per_layer_input,
        };
    }

    fn deinit(self: *PerLayerBranchScratch, allocator: std.mem.Allocator) void {
        allocator.free(self.projection);
        allocator.free(self.per_layer_input);
        allocator.free(self.gated);
        allocator.free(self.branch);
        allocator.free(self.source_embeddings);
        self.* = undefined;
    }
};

const PerLayerBranchRuntime = struct {
    hidden_size_per_layer_input: usize,
    use_gelu: bool,
    per_layer_input_scale: f32,
    per_layer_embed_scale: f32,
    per_layer_model_projection_scale: f32,
    per_layer_embedding: Tensor,
    per_layer_projection_norm_weight: Tensor,
    per_layer_projection_norm_eps: f32,
    per_layer_model_projection_weights: []Tensor,
    per_layer_input_gate_weights: []Tensor,
    per_layer_projection_weights: []Tensor,
    post_per_layer_input_norm_weights: []Tensor,
    layer_scalars: []f32,
    matmul_fn: MatmulFn,

    fn deinit(self: *PerLayerBranchRuntime, allocator: std.mem.Allocator) void {
        allocator.free(self.per_layer_model_projection_weights);
        allocator.free(self.per_layer_input_gate_weights);
        allocator.free(self.per_layer_projection_weights);
        allocator.free(self.post_per_layer_input_norm_weights);
        allocator.free(self.layer_scalars);
        self.* = undefined;
    }
};

fn tensorScalarToF32(tensor_value: *const Tensor) !f32 {
    if (tensor_value.numel < 1) return error.InvalidShape;
    return switch (tensor_value.dtype) {
        .f32 => tensor_value.asSlice(f32)[0],
        .bf16 => dtype_mod.bf16ToF32(tensor_value.asSlice(u16)[0]),
        .f16 => dtype_mod.fp16ToF32(tensor_value.asSlice(u16)[0]),
        else => error.UnsupportedDType,
    };
}

fn loadTensorAny(
    safetensors: *st_loader.UnifiedSafeTensors,
    names: []const []const u8,
) !Tensor {
    for (names) |name| {
        const maybe = safetensors.getTensor(name, null) catch null;
        if (maybe) |tensor_value| return tensor_value;
    }
    return error.MissingWeight;
}

fn loadLayerTensorBySuffix(
    safetensors: *st_loader.UnifiedSafeTensors,
    layer_idx: usize,
    suffix: []const u8,
) !Tensor {
    var name_buf: [256]u8 = undefined;
    const first = std.fmt.bufPrint(name_buf[0..], "model.language_model.layers.{d}.{s}", .{ layer_idx, suffix }) catch null;
    if (first) |full_name| {
        if (safetensors.getTensor(full_name, null) catch null) |tensor_value| return tensor_value;
    }
    const second = std.fmt.bufPrint(name_buf[0..], "model.layers.{d}.{s}", .{ layer_idx, suffix }) catch null;
    if (second) |full_name| {
        if (safetensors.getTensor(full_name, null) catch null) |tensor_value| return tensor_value;
    }
    const third = std.fmt.bufPrint(name_buf[0..], "language_model.model.layers.{d}.{s}", .{ layer_idx, suffix }) catch null;
    if (third) |full_name| {
        if (safetensors.getTensor(full_name, null) catch null) |tensor_value| return tensor_value;
    }
    const fourth = std.fmt.bufPrint(name_buf[0..], "layers.{d}.{s}", .{ layer_idx, suffix }) catch null;
    if (fourth) |full_name| {
        if (safetensors.getTensor(full_name, null) catch null) |tensor_value| return tensor_value;
    }
    return error.MissingWeight;
}

fn snapshotSourceEmbeddings(
    plb_scratch: *PerLayerBranchScratch,
    input_tensor: *const Tensor,
) !Tensor {
    if (input_tensor.dtype != .f32) return error.UnsupportedDType;
    const values = input_tensor.asSlice(f32);
    const total_values = plb_scratch.row_count * plb_scratch.hidden_size;
    if (values.len < total_values or plb_scratch.source_embeddings.len < total_values) return error.InvalidShape;
    @memcpy(plb_scratch.source_embeddings[0..total_values], values[0..total_values]);
    return Tensor.view3DSlice(plb_scratch.source_embeddings[0..total_values], plb_scratch.row_count, plb_scratch.hidden_size);
}

fn matrixRowsView(
    source: *const Tensor,
    row_start: usize,
    row_count: usize,
    col_count: usize,
) !Tensor {
    if (source.n_dims != 2) return error.InvalidShape;
    if (source.dtype.isQuantized()) return error.UnsupportedDType;
    const rows: usize = @intCast(source.shape[0]);
    const cols: usize = @intCast(source.shape[1]);
    if (row_start + row_count > rows or cols != col_count) return error.InvalidShape;
    const data_ptr = source.data_ptr orelse return error.InvalidShape;
    const elem_size = source.dtype.elementSize();
    const row_stride_bytes = cols * elem_size;
    const offset_bytes = row_start * row_stride_bytes;

    var view = source.*;
    view.n_dims = 2;
    view.shape = [_]i64{ @intCast(row_count), @intCast(col_count), 0, 0, 0, 0, 0, 0 };
    view.strides = [_]i64{ @intCast(col_count), 1, 0, 0, 0, 0, 0, 0 };
    view.numel = row_count * col_count;
    view.data_ptr = data_ptr + offset_bytes;
    view.data_size = view.numel * elem_size;
    return view;
}

fn initPerLayerBranchRuntime(
    allocator: std.mem.Allocator,
    loaded: *LoadedModel,
    layer_count: usize,
) !?PerLayerBranchRuntime {
    if (loaded.config.hidden_size_per_layer_input <= 0) return null;
    if (loaded.st == null) return error.MissingWeight;
    const safetensors = &(loaded.st.?);

    const hidden_size_per_layer_input: usize = @intCast(loaded.config.hidden_size_per_layer_input);
    const hidden_size: usize = @intCast(loaded.config.d_model);
    const layer_width_total = layer_count * hidden_size_per_layer_input;

    const per_layer_embedding = try loadTensorAny(safetensors, &.{
        "model.language_model.embed_tokens_per_layer.weight",
        "model.embed_tokens_per_layer.weight",
        "embed_tokens_per_layer.weight",
    });
    const per_layer_model_projection_weight = try loadTensorAny(safetensors, &.{
        "model.language_model.per_layer_model_projection.weight",
        "model.per_layer_model_projection.weight",
        "per_layer_model_projection.weight",
    });
    const per_layer_projection_norm_weight = try loadTensorAny(safetensors, &.{
        "model.language_model.per_layer_projection_norm.weight",
        "model.per_layer_projection_norm.weight",
        "per_layer_projection_norm.weight",
    });

    if (per_layer_embedding.n_dims != 2 or @as(usize, @intCast(per_layer_embedding.shape[1])) < layer_width_total) {
        return error.InvalidShape;
    }
    if (per_layer_model_projection_weight.n_dims != 2 or
        @as(usize, @intCast(per_layer_model_projection_weight.shape[0])) != layer_width_total or
        @as(usize, @intCast(per_layer_model_projection_weight.shape[1])) != hidden_size)
    {
        return error.InvalidShape;
    }
    if (per_layer_projection_norm_weight.n_dims != 1 or
        @as(usize, @intCast(per_layer_projection_norm_weight.shape[0])) != hidden_size_per_layer_input)
    {
        return error.InvalidShape;
    }

    const per_layer_model_projection_weights = try allocator.alloc(Tensor, layer_count);
    errdefer allocator.free(per_layer_model_projection_weights);
    const per_layer_input_gate_weights = try allocator.alloc(Tensor, layer_count);
    errdefer allocator.free(per_layer_input_gate_weights);
    const per_layer_projection_weights = try allocator.alloc(Tensor, layer_count);
    errdefer allocator.free(per_layer_projection_weights);
    const post_per_layer_input_norm_weights = try allocator.alloc(Tensor, layer_count);
    errdefer allocator.free(post_per_layer_input_norm_weights);
    const layer_scalars = try allocator.alloc(f32, layer_count);
    errdefer allocator.free(layer_scalars);

    for (0..layer_count) |layer_idx| {
        const gate_weight = try loadLayerTensorBySuffix(safetensors, layer_idx, "per_layer_input_gate.weight");
        const projection_weight = try loadLayerTensorBySuffix(safetensors, layer_idx, "per_layer_projection.weight");
        const post_norm_weight = try loadLayerTensorBySuffix(safetensors, layer_idx, "post_per_layer_input_norm.weight");
        const layer_scalar = try loadLayerTensorBySuffix(safetensors, layer_idx, "layer_scalar");

        if (gate_weight.n_dims != 2 or
            @as(usize, @intCast(gate_weight.shape[0])) != hidden_size_per_layer_input or
            @as(usize, @intCast(gate_weight.shape[1])) != hidden_size)
        {
            return error.InvalidShape;
        }
        if (projection_weight.n_dims != 2 or
            @as(usize, @intCast(projection_weight.shape[0])) != hidden_size or
            @as(usize, @intCast(projection_weight.shape[1])) != hidden_size_per_layer_input)
        {
            return error.InvalidShape;
        }
        if (post_norm_weight.n_dims != 1 or
            @as(usize, @intCast(post_norm_weight.shape[0])) != hidden_size)
        {
            return error.InvalidShape;
        }
        // All projection weights must share the same dtype for cached matmul dispatch.
        if (gate_weight.dtype != per_layer_model_projection_weight.dtype or
            projection_weight.dtype != per_layer_model_projection_weight.dtype)
        {
            return error.UnsupportedDType;
        }

        const row_start = layer_idx * hidden_size_per_layer_input;
        const proj_weight_view = try matrixRowsView(
            &per_layer_model_projection_weight,
            row_start,
            hidden_size_per_layer_input,
            hidden_size,
        );

        per_layer_model_projection_weights[layer_idx] = proj_weight_view;
        per_layer_input_gate_weights[layer_idx] = gate_weight;
        per_layer_projection_weights[layer_idx] = projection_weight;
        post_per_layer_input_norm_weights[layer_idx] = post_norm_weight;
        layer_scalars[layer_idx] = try tensorScalarToF32(&layer_scalar);
    }

    return .{
        .hidden_size_per_layer_input = hidden_size_per_layer_input,
        .use_gelu = loaded.config.use_gelu,
        .per_layer_input_scale = 0.70710677, // 2^-0.5
        .per_layer_embed_scale = @sqrt(@as(f32, @floatFromInt(hidden_size_per_layer_input))),
        .per_layer_model_projection_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hidden_size))),
        .per_layer_embedding = per_layer_embedding,
        .per_layer_projection_norm_weight = per_layer_projection_norm_weight,
        .per_layer_projection_norm_eps = loaded.config.norm_eps,
        .per_layer_model_projection_weights = per_layer_model_projection_weights,
        .per_layer_input_gate_weights = per_layer_input_gate_weights,
        .per_layer_projection_weights = per_layer_projection_weights,
        .post_per_layer_input_norm_weights = post_per_layer_input_norm_weights,
        .layer_scalars = layer_scalars,
        .matmul_fn = (try cpu_linalg.matmulKernel(per_layer_model_projection_weight.dtype)).func,
    };
}

/// Load per-layer scalar multipliers from safetensors. Returns null when the
/// weights are absent (most models) or when PLE already owns them.
fn initStandaloneLayerScalars(
    allocator: std.mem.Allocator,
    loaded: *LoadedModel,
    layer_count: usize,
) !?[]f32 {
    // PLE already loads and applies layer_scalars — don't double-apply.
    if (loaded.config.hidden_size_per_layer_input > 0) return null;
    if (loaded.st == null) return null;
    const safetensors = &(loaded.st.?);

    // Probe layer 0 to determine if layer_scalar weights exist.
    _ = loadLayerTensorBySuffix(safetensors, 0, "layer_scalar") catch return null;

    const scalars = try allocator.alloc(f32, layer_count);
    errdefer allocator.free(scalars);
    for (0..layer_count) |layer_idx| {
        const t = try loadLayerTensorBySuffix(safetensors, layer_idx, "layer_scalar");
        scalars[layer_idx] = try tensorScalarToF32(&t);
    }
    return scalars;
}

/// Complete transformer model for inference
pub const Transformer = struct {
    model_type: []const u8,
    embed_tokens: Embedding,
    layers: []Block,
    norm: ?RMSNorm = null,
    lm_head: ?Linear = null,
    tie_word_embeddings: bool = true,

    // Dimensions
    hidden_size: usize,
    vocab_size: usize,
    num_hidden_layers: usize,

    // Original weight dtype for summary
    weight_dtype: DType,

    // File info for summary (from LoadedModel)
    file_size: usize = 0,
    per_layer_branch: ?PerLayerBranchRuntime = null,

    // Per-layer scalar multiplier applied to the residual stream at the end of
    // each decoder layer. Loaded independently of the PLE system so models
    // with layer_scalar weights but no per-layer embeddings still apply them.
    layer_scalars: ?[]f32 = null,

    // Optional prefill progress callback (set transiently by prefillSlot)
    prefill_progress_fn: ?PrefillProgressFn = null,
    prefill_progress_ctx: ?*anyopaque = null,

    // Optional stop flag checked per-layer during prefill forward pass.
    // Set transiently by prefillSlot alongside the progress callback.
    stop_flag: ?*const std.atomic.Value(bool) = null,

    tensor_count: usize = 0,

    pub const PrefillProgressFn = *const fn (usize, usize, ?*anyopaque) callconv(.c) void;
    pub const DeepstackAdditions = struct {
        /// Token positions in the prompt where visual placeholders were scattered.
        positions: []const usize,
        /// Per-decoder-layer visual embeddings to add at `positions`.
        /// Layer 0 consumes layer_features[0], layer 1 consumes layer_features[1], etc.
        layer_features: []const []const f32,
    };

    fn resolveLayeredCache(
        state_blocks: []const runtime_contract.StateBlockHandle,
    ) ?*LayeredBatchedKVCache {
        for (state_blocks) |*state_block| {
            const state_value = runtime_contract.stateValueFromBlock(*state_bindings.KvRuntimeState, state_block) orelse continue;
            if (state_value.runtime_kind != runtime_contract.state_runtime_kind_kv_cache) continue;
            return state_value.layered_cache;
        }
        return null;
    }

    fn validatePerLayerBranchTokenIds(
        self: *const Transformer,
        plb: *const PerLayerBranchRuntime,
        token_ids: ?[]const u32,
        row_count: usize,
    ) ![]const u32 {
        _ = self;
        const ids = token_ids orelse return error.InvalidArgument;
        if (ids.len != row_count) return error.InvalidArgument;
        const vocab_limit: usize = @intCast(plb.per_layer_embedding.shape[0]);
        for (ids) |token_id| {
            if (token_id >= vocab_limit) return error.InvalidToken;
        }
        return ids;
    }

    /// Log first 8 f32 values + L2 norm from a CPU tensor row.
    /// Gated by TALU_DUMP_HIDDEN env var. Uses log.warn so it survives ReleaseFast.
    fn dumpCpuHiddenState(data: []const f32, d_model: usize, global_layer_idx: usize, label: []const u8) void {
        const dump_env = std.posix.getenv("TALU_DUMP_HIDDEN");
        if (dump_env == null) return;
        if (data.len < d_model) return;

        const row = data[0..d_model];
        var host_buf: [8]f32 = .{0} ** 8;
        const n = @min(8, d_model);
        @memcpy(host_buf[0..n], row[0..n]);

        var sum: f64 = 0.0;
        for (row) |v| {
            sum += @as(f64, v) * @as(f64, v);
        }
        const l2_norm: f32 = @floatCast(@sqrt(sum));

        log.warn("inference", "DUMP_HIDDEN", .{
            .layer = global_layer_idx,
            .label = label,
            .l2_norm = l2_norm,
            .v0 = host_buf[0],
            .v1 = host_buf[1],
            .v2 = host_buf[2],
            .v3 = host_buf[3],
            .v4 = host_buf[4],
            .v5 = host_buf[5],
            .v6 = host_buf[6],
            .v7 = host_buf[7],
        });
    }

    fn applyPerLayerBranch(
        self: *const Transformer,
        plb: *const PerLayerBranchRuntime,
        plb_scratch: *PerLayerBranchScratch,
        layer_idx: usize,
        source_embeddings: *const Tensor,
        hidden_states: *Tensor,
        token_ids: []const u32,
        scratch: *ScratchBuffer,
    ) !void {
        var kernel_scratch = per_layer_branch_kernel.PerLayerBranchScratch{
            .projection = plb_scratch.projection,
            .per_layer_input = plb_scratch.per_layer_input,
            .gated = plb_scratch.gated,
            .branch = plb_scratch.branch,
        };
        const params = per_layer_branch_kernel.ForwardParams{
            .hidden_states = hidden_states,
            .source_embeddings = source_embeddings,
            .token_ids = token_ids,
            .per_layer_embedding = &plb.per_layer_embedding,
            .model_projection_weight = &plb.per_layer_model_projection_weights[layer_idx],
            .projection_norm_weight = &plb.per_layer_projection_norm_weight,
            .input_gate_weight = &plb.per_layer_input_gate_weights[layer_idx],
            .projection_weight = &plb.per_layer_projection_weights[layer_idx],
            .post_norm_weight = &plb.post_per_layer_input_norm_weights[layer_idx],
            .config = .{
                .hidden_size = self.hidden_size,
                .hidden_size_per_layer_input = plb.hidden_size_per_layer_input,
                .use_gelu = plb.use_gelu,
                .per_layer_input_scale = plb.per_layer_input_scale,
                .per_layer_embed_scale = plb.per_layer_embed_scale,
                .per_layer_model_projection_scale = plb.per_layer_model_projection_scale,
                .norm_eps = plb.per_layer_projection_norm_eps,
                .norm_weight_offset = 0.0,
                .layer_scalar = plb.layer_scalars[layer_idx],
                .layer_idx = layer_idx,
            },
            .scratch = &kernel_scratch,
            .matmul_scratch = &scratch.matmul_scratch,
            .matmul_fn = plb.matmul_fn,
        };
        try per_layer_branch_kernel.forward(&params);
    }

    /// Forward pass through transformer layers only (not embedding or final norm).
    /// This is the core transformer body: hidden_states -> layers -> hidden_states
    pub fn forward(
        self: *const Transformer,
        input_tensor: *const Tensor,
        output_tensor: *Tensor,
        scratch: *ScratchBuffer,
        use_cache: bool,
    ) !void {
        return self.forwardWithTokenIds(input_tensor, output_tensor, scratch, null, use_cache);
    }

    pub fn forwardWithTokenIds(
        self: *const Transformer,
        input_tensor: *const Tensor,
        output_tensor: *Tensor,
        scratch: *ScratchBuffer,
        token_ids: ?[]const u32,
        use_cache: bool,
    ) !void {
        if (!use_cache) scratch.resetCaches();
        const row_count: usize = @intCast(input_tensor.shape[1]);
        for (self.layers) |*layer| {
            try layer.registerScratchLayout(scratch);
        }
        try scratch.ensureForMode(if (use_cache) .decode else .prefill, row_count);

        var plb_scratch: ?PerLayerBranchScratch = null;
        defer if (plb_scratch) |*state| state.deinit(scratch.allocator);
        var resolved_token_ids: ?[]const u32 = null;
        var plb_source_embeddings: Tensor = undefined;
        var has_plb_source_embeddings = false;
        if (self.per_layer_branch) |*plb| {
            resolved_token_ids = try self.validatePerLayerBranchTokenIds(plb, token_ids, row_count);
            plb_scratch = try PerLayerBranchScratch.init(
                scratch.allocator,
                row_count,
                self.hidden_size,
                plb.hidden_size_per_layer_input,
            );
            const plb_state = if (plb_scratch) |*state| state else unreachable;
            plb_source_embeddings = try snapshotSourceEmbeddings(plb_state, input_tensor);
            has_plb_source_embeddings = true;
        }

        // Use pre-allocated scratch buffer for alternating input/output
        var scratch_tensor_view = Tensor.view3DSlice(scratch.tmp[0], row_count, self.hidden_size);
        var current_input_tensor: *const Tensor = input_tensor;
        var write_to_scratch_view = false;

        for (self.layers, 0..) |*layer, layer_idx| {
            // Emit trace point for layer input (if handler installed)
            trace.emit(
                .layer_input,
                @intCast(layer_idx),
                0, // token (batch dimension)
                @intCast(row_count), // position
                current_input_tensor.data().ptr,
                .f32,
                .{ 1, @intCast(row_count), @intCast(self.hidden_size), 0 },
                3,
                null, // no kernel - this is the input activation
            );

            var layer_output_tensor: *Tensor = undefined;
            if (write_to_scratch_view) {
                try layer.forward(current_input_tensor, &scratch_tensor_view, scratch, use_cache);
                layer_output_tensor = &scratch_tensor_view;
            } else {
                try layer.forward(current_input_tensor, output_tensor, scratch, use_cache);
                layer_output_tensor = output_tensor;
            }
            if (self.per_layer_branch) |*plb| {
                const plb_state = if (plb_scratch) |*state| state else unreachable;
                const source_embeddings = if (has_plb_source_embeddings) &plb_source_embeddings else unreachable;
                try self.applyPerLayerBranch(
                    plb,
                    plb_state,
                    layer_idx,
                    source_embeddings,
                    layer_output_tensor,
                    resolved_token_ids orelse unreachable,
                    scratch,
                );
            } else if (self.layer_scalars) |scalars| {
                cpu_rowwise.scaleInPlace(layer_output_tensor.asSliceMut(f32), scalars[layer_idx]);
            }
            current_input_tensor = layer_output_tensor;
            write_to_scratch_view = !write_to_scratch_view;

            // Emit trace point for layer output (if handler installed)
            trace.emit(
                .block_out,
                @intCast(layer_idx),
                0, // token (batch dimension)
                @intCast(row_count), // position
                current_input_tensor.data().ptr,
                .f32,
                .{ 1, @intCast(row_count), @intCast(self.hidden_size), 0 },
                3,
                null, // residual is element-wise add, not a matmul kernel
            );
            // Dump capture (compiled in only for dump binary)
            if (build_options.dump_tensors) {
                const shape = [4]usize{ 1, row_count, self.hidden_size, 0 };
                dump.recordGlobal(.block_out, @intCast(layer_idx), current_input_tensor.data().ptr, .f32, shape, 3);
                // Early stop for partial dump (big runtime win when debugging specific layers)
                if (dump.shouldStopGlobal()) break;
            }
        }

        // Copy final result to out if needed
        if (current_input_tensor != output_tensor) {
            block_kernels.copyTensor(current_input_tensor, output_tensor);
        }
    }

    /// Forward pass through transformer layers using batched KV cache.
    /// This enables continuous batching by using LayeredBatchedKVCache with a slot index.
    pub fn forwardWithBatchedCache(
        self: *const Transformer,
        input_tensor: *const Tensor,
        output_tensor: *Tensor,
        scratch: *ScratchBuffer,
        state_blocks: []const runtime_contract.StateBlockHandle,
        slot_index: usize,
        use_cache: bool,
    ) !void {
        return self.forwardWithBatchedCacheTokenIds(
            input_tensor,
            output_tensor,
            scratch,
            state_blocks,
            slot_index,
            null,
            use_cache,
        );
    }

    pub fn forwardWithBatchedCacheTokenIds(
        self: *const Transformer,
        input_tensor: *const Tensor,
        output_tensor: *Tensor,
        scratch: *ScratchBuffer,
        state_blocks: []const runtime_contract.StateBlockHandle,
        slot_index: usize,
        token_ids: ?[]const u32,
        use_cache: bool,
    ) !void {
        return self.forwardWithBatchedCacheWithDeepstackLayerRange(
            input_tensor,
            output_tensor,
            scratch,
            state_blocks,
            slot_index,
            token_ids,
            use_cache,
            null,
            0,
            self.layers.len,
        );
    }

    /// Forward pass through a subset of transformer layers using batched KV cache.
    /// Executes layers in the half-open range [layer_start, layer_end).
    pub fn forwardWithBatchedCacheLayerRange(
        self: *const Transformer,
        input_tensor: *const Tensor,
        output_tensor: *Tensor,
        scratch: *ScratchBuffer,
        state_blocks: []const runtime_contract.StateBlockHandle,
        slot_index: usize,
        use_cache: bool,
        layer_start: usize,
        layer_end: usize,
    ) !void {
        return self.forwardWithBatchedCacheLayerRangeTokenIds(
            input_tensor,
            output_tensor,
            scratch,
            state_blocks,
            slot_index,
            null,
            use_cache,
            layer_start,
            layer_end,
        );
    }

    pub fn forwardWithBatchedCacheLayerRangeTokenIds(
        self: *const Transformer,
        input_tensor: *const Tensor,
        output_tensor: *Tensor,
        scratch: *ScratchBuffer,
        state_blocks: []const runtime_contract.StateBlockHandle,
        slot_index: usize,
        token_ids: ?[]const u32,
        use_cache: bool,
        layer_start: usize,
        layer_end: usize,
    ) !void {
        return self.forwardWithBatchedCacheWithDeepstackLayerRange(
            input_tensor,
            output_tensor,
            scratch,
            state_blocks,
            slot_index,
            token_ids,
            use_cache,
            null,
            layer_start,
            layer_end,
        );
    }

    /// Returns true when every layer can execute decode in slot-batched mode.
    pub fn supportsBatchedDecodeSlots(self: *const Transformer) bool {
        for (self.layers) |*layer| {
            if (!layer.supportsBatchedDecodeSlots()) return false;
        }
        return true;
    }

    pub fn forwardWithBatchedCacheWithDeepstack(
        self: *const Transformer,
        input_tensor: *const Tensor,
        output_tensor: *Tensor,
        scratch: *ScratchBuffer,
        state_blocks: []const runtime_contract.StateBlockHandle,
        slot_index: usize,
        use_cache: bool,
        deepstack: ?*const DeepstackAdditions,
    ) !void {
        return self.forwardWithBatchedCacheWithDeepstackTokenIds(
            input_tensor,
            output_tensor,
            scratch,
            state_blocks,
            slot_index,
            null,
            use_cache,
            deepstack,
        );
    }

    pub fn forwardWithBatchedCacheWithDeepstackTokenIds(
        self: *const Transformer,
        input_tensor: *const Tensor,
        output_tensor: *Tensor,
        scratch: *ScratchBuffer,
        state_blocks: []const runtime_contract.StateBlockHandle,
        slot_index: usize,
        token_ids: ?[]const u32,
        use_cache: bool,
        deepstack: ?*const DeepstackAdditions,
    ) !void {
        return self.forwardWithBatchedCacheWithDeepstackLayerRange(
            input_tensor,
            output_tensor,
            scratch,
            state_blocks,
            slot_index,
            token_ids,
            use_cache,
            deepstack,
            0,
            self.layers.len,
        );
    }

    /// Forward pass through a subset of transformer layers using batched KV cache.
    /// Executes layers in the half-open range [layer_start, layer_end).
    pub fn forwardWithBatchedCacheWithDeepstackLayerRange(
        self: *const Transformer,
        input_tensor: *const Tensor,
        output_tensor: *Tensor,
        scratch: *ScratchBuffer,
        state_blocks: []const runtime_contract.StateBlockHandle,
        slot_index: usize,
        token_ids: ?[]const u32,
        use_cache: bool,
        deepstack: ?*const DeepstackAdditions,
        layer_start: usize,
        layer_end: usize,
    ) !void {
        if (layer_end < layer_start or layer_end > self.layers.len) return error.InvalidArgument;
        if (self.per_layer_branch != null and layer_start != 0) return error.InvalidArgument;
        const layered_cache = resolveLayeredCache(state_blocks);
        if (!use_cache and layer_start == 0) {
            if (layered_cache) |cache| cache.resetSlot(slot_index);
            scratch.resetSlotCaches(slot_index);
        }
        const seq_len: usize = @intCast(input_tensor.shape[1]);
        for (self.layers[layer_start..layer_end]) |*layer| {
            try layer.registerScratchLayout(scratch);
        }
        try scratch.ensureForMode(if (use_cache) .decode else .prefill, seq_len);

        var plb_scratch: ?PerLayerBranchScratch = null;
        defer if (plb_scratch) |*state| state.deinit(scratch.allocator);
        var resolved_token_ids: ?[]const u32 = null;
        var plb_source_embeddings: Tensor = undefined;
        var has_plb_source_embeddings = false;
        if (self.per_layer_branch) |*plb| {
            resolved_token_ids = try self.validatePerLayerBranchTokenIds(plb, token_ids, seq_len);
            plb_scratch = try PerLayerBranchScratch.init(
                scratch.allocator,
                seq_len,
                self.hidden_size,
                plb.hidden_size_per_layer_input,
            );
            const plb_state = if (plb_scratch) |*state| state else unreachable;
            plb_source_embeddings = try snapshotSourceEmbeddings(plb_state, input_tensor);
            has_plb_source_embeddings = true;
        }

        // Use pre-allocated scratch buffer for alternating input/output
        var scratch_tensor_view = Tensor.view3DSlice(scratch.tmp[0], seq_len, self.hidden_size);
        var current_input_tensor: *const Tensor = input_tensor;
        var write_to_scratch_view = false;

        for (self.layers[layer_start..layer_end], 0..) |*layer, local_layer_idx| {
            const layer_idx = layer_start + local_layer_idx;
            // Emit trace point for layer input (if handler installed)
            trace.emit(
                .layer_input,
                @intCast(layer_idx),
                0,
                @intCast(seq_len),
                current_input_tensor.data().ptr,
                .f32,
                .{ 1, @intCast(seq_len), @intCast(self.hidden_size), 0 },
                3,
                null,
            );

            var layer_output_tensor: *Tensor = undefined;
            if (write_to_scratch_view) {
                try layer.forwardWithBatchedCache(current_input_tensor, &scratch_tensor_view, scratch, state_blocks, slot_index, use_cache);
                layer_output_tensor = &scratch_tensor_view;
                if (deepstack) |ctx| {
                    if (local_layer_idx < ctx.layer_features.len) {
                        try applyDeepstackAdditions(layer_output_tensor, seq_len, self.hidden_size, ctx.positions, ctx.layer_features[local_layer_idx]);
                    }
                }
            } else {
                try layer.forwardWithBatchedCache(current_input_tensor, output_tensor, scratch, state_blocks, slot_index, use_cache);
                layer_output_tensor = output_tensor;
                if (deepstack) |ctx| {
                    if (local_layer_idx < ctx.layer_features.len) {
                        try applyDeepstackAdditions(layer_output_tensor, seq_len, self.hidden_size, ctx.positions, ctx.layer_features[local_layer_idx]);
                    }
                }
            }
            dumpCpuHiddenState(layer_output_tensor.asSlice(f32), self.hidden_size, layer_idx, "post_layer");
            if (self.per_layer_branch) |*plb| {
                const plb_state = if (plb_scratch) |*state| state else unreachable;
                const source_embeddings = if (has_plb_source_embeddings) &plb_source_embeddings else unreachable;
                try self.applyPerLayerBranch(
                    plb,
                    plb_state,
                    layer_idx,
                    source_embeddings,
                    layer_output_tensor,
                    resolved_token_ids orelse unreachable,
                    scratch,
                );
            } else if (self.layer_scalars) |scalars| {
                cpu_rowwise.scaleInPlace(layer_output_tensor.asSliceMut(f32), scalars[layer_idx]);
            }
            dumpCpuHiddenState(layer_output_tensor.asSlice(f32), self.hidden_size, layer_idx, "post_ple");
            current_input_tensor = layer_output_tensor;
            write_to_scratch_view = !write_to_scratch_view;

            // Emit trace point for layer output
            trace.emit(
                .block_out,
                @intCast(layer_idx),
                0,
                @intCast(seq_len),
                current_input_tensor.data().ptr,
                .f32,
                .{ 1, @intCast(seq_len), @intCast(self.hidden_size), 0 },
                3,
                null,
            );
            // Dump capture (compiled in only for dump binary)
            if (build_options.dump_tensors) {
                const shape = [4]usize{ 1, seq_len, self.hidden_size, 0 };
                dump.recordGlobal(.block_out, @intCast(layer_idx), current_input_tensor.data().ptr, .f32, shape, 3);
                // Early stop for partial dump (big runtime win when debugging specific layers)
                if (dump.shouldStopGlobal()) break;
            }

            // Prefill progress (fires once per layer, outside hot path)
            if (self.prefill_progress_fn) |progress_fn| {
                progress_fn(layer_idx + 1, self.num_hidden_layers, self.prefill_progress_ctx);
            }

            // Check stop flag after each layer for responsive cancellation.
            if (self.stop_flag) |flag| {
                if (flag.load(.acquire)) return error.Cancelled;
            }
        }

        // Copy final result to out if needed
        if (current_input_tensor != output_tensor) {
            block_kernels.copyTensor(current_input_tensor, output_tensor);
        }
    }

    /// Forward pass through transformer layers using batched cache and a slot map.
    /// The decode-batch tensor layout is [1, batch_size, d_model].
    pub fn forwardWithBatchedCacheSlots(
        self: *const Transformer,
        input_tensor: *const Tensor,
        output_tensor: *Tensor,
        scratch: *ScratchBuffer,
        state_blocks: []const runtime_contract.StateBlockHandle,
        slot_indices: []const usize,
        use_cache: bool,
    ) !void {
        return self.forwardWithBatchedCacheSlotsTokenIds(
            input_tensor,
            output_tensor,
            scratch,
            state_blocks,
            slot_indices,
            null,
            use_cache,
        );
    }

    pub fn forwardWithBatchedCacheSlotsTokenIds(
        self: *const Transformer,
        input_tensor: *const Tensor,
        output_tensor: *Tensor,
        scratch: *ScratchBuffer,
        state_blocks: []const runtime_contract.StateBlockHandle,
        slot_indices: []const usize,
        token_ids: ?[]const u32,
        use_cache: bool,
    ) !void {
        std.debug.assert(input_tensor.shape[0] == 1 and output_tensor.shape[0] == 1);
        std.debug.assert(@as(usize, @intCast(input_tensor.shape[1])) == slot_indices.len);
        std.debug.assert(@as(usize, @intCast(output_tensor.shape[1])) == slot_indices.len);
        const layered_cache = resolveLayeredCache(state_blocks);
        if (!use_cache) {
            if (layered_cache) |cache| {
                for (slot_indices) |slot_index| cache.resetSlot(slot_index);
            }
            for (slot_indices) |slot_index| {
                scratch.resetSlotCaches(slot_index);
            }
        }

        const batch_size: usize = @intCast(input_tensor.shape[1]);
        for (self.layers) |*layer| {
            try layer.registerScratchLayout(scratch);
        }
        try scratch.ensureForMode(if (use_cache) .decode else .prefill, batch_size);

        var plb_scratch: ?PerLayerBranchScratch = null;
        defer if (plb_scratch) |*state| state.deinit(scratch.allocator);
        var resolved_token_ids: ?[]const u32 = null;
        var plb_source_embeddings: Tensor = undefined;
        var has_plb_source_embeddings = false;
        if (self.per_layer_branch) |*plb| {
            resolved_token_ids = try self.validatePerLayerBranchTokenIds(plb, token_ids, batch_size);
            plb_scratch = try PerLayerBranchScratch.init(
                scratch.allocator,
                batch_size,
                self.hidden_size,
                plb.hidden_size_per_layer_input,
            );
            const plb_state = if (plb_scratch) |*state| state else unreachable;
            plb_source_embeddings = try snapshotSourceEmbeddings(plb_state, input_tensor);
            has_plb_source_embeddings = true;
        }

        var scratch_tensor_view = Tensor.view3DSlice(scratch.tmp[0], batch_size, self.hidden_size);
        var current_input_tensor: *const Tensor = input_tensor;
        var write_to_scratch_view = false;

        for (self.layers, 0..) |*layer, layer_idx| {
            trace.emit(
                .layer_input,
                @intCast(layer_idx),
                0,
                @intCast(batch_size),
                current_input_tensor.data().ptr,
                .f32,
                .{ 1, @intCast(batch_size), @intCast(self.hidden_size), 0 },
                3,
                null,
            );

            var layer_output_tensor: *Tensor = undefined;
            if (write_to_scratch_view) {
                try layer.forwardWithBatchedCacheSlots(current_input_tensor, &scratch_tensor_view, scratch, state_blocks, slot_indices, use_cache);
                layer_output_tensor = &scratch_tensor_view;
            } else {
                try layer.forwardWithBatchedCacheSlots(current_input_tensor, output_tensor, scratch, state_blocks, slot_indices, use_cache);
                layer_output_tensor = output_tensor;
            }
            if (self.per_layer_branch) |*plb| {
                const plb_state = if (plb_scratch) |*state| state else unreachable;
                const source_embeddings = if (has_plb_source_embeddings) &plb_source_embeddings else unreachable;
                try self.applyPerLayerBranch(
                    plb,
                    plb_state,
                    layer_idx,
                    source_embeddings,
                    layer_output_tensor,
                    resolved_token_ids orelse unreachable,
                    scratch,
                );
            } else if (self.layer_scalars) |scalars| {
                cpu_rowwise.scaleInPlace(layer_output_tensor.asSliceMut(f32), scalars[layer_idx]);
            }
            current_input_tensor = layer_output_tensor;
            write_to_scratch_view = !write_to_scratch_view;

            trace.emit(
                .block_out,
                @intCast(layer_idx),
                0,
                @intCast(batch_size),
                current_input_tensor.data().ptr,
                .f32,
                .{ 1, @intCast(batch_size), @intCast(self.hidden_size), 0 },
                3,
                null,
            );
        }

        if (current_input_tensor != output_tensor) {
            block_kernels.copyTensor(current_input_tensor, output_tensor);
        }
    }

    fn applyDeepstackAdditions(
        hidden: *Tensor,
        seq_len: usize,
        hidden_size: usize,
        positions: []const usize,
        features: []const f32,
    ) !void {
        if (positions.len == 0) return;
        try cpu_memory.scatterAddRowsByPositions(
            hidden.asSliceMut(f32),
            seq_len,
            hidden_size,
            positions,
            features,
        );
    }

    pub fn describe(self: *const Transformer, writer: anytype, show_kernels: bool) !void {
        try self.describeCondensed(writer, show_kernels, 3); // Show first 3, last 1
    }

    const DescribeMode = union(enum) {
        condensed: usize,
        all,
    };

    fn describeImpl(self: *const Transformer, writer: anytype, show_kernels: bool, mode: DescribeMode) !void {
        try writer.print("{s}(\n", .{self.model_type});

        try writer.writeAll("  (embed_tokens): ");
        try self.embed_tokens.formatTo(writer);
        try writer.writeAll("\n");

        if (show_kernels) {
            try self.embed_tokens.formatKernels(writer, 4);
        }

        try writer.writeAll("  (layers): [\n");

        const layer_count = self.layers.len;
        switch (mode) {
            .all => {
                for (self.layers) |*layer| {
                    try layer.describe(writer, 4, show_kernels);
                }
            },
            .condensed => |show_first| {
                if (layer_count <= show_first + 2) {
                    for (self.layers) |*layer| {
                        try layer.describe(writer, 4, show_kernels);
                    }
                } else {
                    for (self.layers[0..show_first]) |*layer| {
                        try layer.describe(writer, 4, show_kernels);
                    }
                    const remaining_layers = layer_count - show_first - 1;
                    try writer.print("    ... {d} more identical layers ...\n", .{remaining_layers});
                    try self.layers[layer_count - 1].describe(writer, 4, show_kernels);
                }
            },
        }
        try writer.writeAll("  ]\n");

        if (self.norm) |n| {
            try writer.writeAll("  (norm): ");
            try formatRmsNormLike(writer, n.dim, n.eps, n.weight_offset);
            try writer.writeAll("\n");
        }

        if (self.lm_head) |*head| {
            try writer.writeAll("  (lm_head): ");
            try head.formatTo(writer);
            try writer.writeAll("\n");
            if (show_kernels) {
                try head.formatKernels(writer, 4);
            }
        } else if (self.tie_word_embeddings) {
            try writer.writeAll("  (lm_head): [tied to embed_tokens]\n");
        }

        try writer.writeAll(")\n");
    }

    /// Describe model with condensed layer output.
    /// Shows first `show_first` layers, then "... N more layers ...", then last layer.
    pub fn describeCondensed(self: *const Transformer, writer: anytype, show_kernels: bool, show_first: usize) !void {
        try self.describeImpl(writer, show_kernels, .{ .condensed = show_first });
    }

    /// Describe model showing all layers (no condensing)
    pub fn describeAll(self: *const Transformer, writer: anytype, show_kernels: bool) !void {
        try self.describeImpl(writer, show_kernels, .all);
    }

    pub fn summary(self: *const Transformer, writer: anytype) !void {
        try self.summaryWithSeqLen(writer, 512); // Default seq_len for memory estimates
    }

    pub fn summaryWithSeqLen(self: *const Transformer, writer: anytype, seq_len: usize) !void {
        _ = seq_len; // Reserved for future use

        try writer.print("Model: {s}\n", .{self.model_type});

        // Estimate parameters
        var total_params: usize = 0;
        total_params += self.embed_tokens.vocab_size * self.embed_tokens.embed_dim;

        if (self.firstLayerGeom()) |geom| {
            var layer_params: usize = 0;
            layer_params += geom.total_layer_params;
            total_params += layer_params * self.num_hidden_layers;
        }

        if (self.lm_head) |head| {
            total_params += head.in_features * head.out_features;
        }

        // Parameters
        if (total_params >= 1_000_000_000) {
            const billions = @as(f64, @floatFromInt(total_params)) / 1_000_000_000.0;
            try writer.print("  Parameters: {d:.2}B\n", .{billions});
        } else if (total_params >= 1_000_000) {
            const millions = @as(f64, @floatFromInt(total_params)) / 1_000_000.0;
            try writer.print("  Parameters: {d:.2}M\n", .{millions});
        } else {
            try writer.print("  Parameters: {}\n", .{total_params});
        }

        // Quantization
        const quant_info: []const u8 = switch (self.weight_dtype) {
            .grouped_affine_u4 => "4-bit (grouped affine)",
            .grouped_affine_u8 => "8-bit (grouped affine)",
            .f8_e4m3 => "FP8 (E4M3)",
            .bf16 => "BF16",
            .f16 => "F16",
            .f32 => "F32",
            else => "unknown",
        };
        try writer.print("  Quantization: {s}\n", .{quant_info});

        // Weights (file size)
        if (self.file_size > 0) {
            try writer.writeAll("  Weights: ");
            try perf_estimate.formatBytes(writer, self.file_size);
            try writer.writeAll("\n");
        }

        // Architecture
        try writer.print("  Layers: {}\n", .{self.num_hidden_layers});
        try writer.print("  Hidden size: {}\n", .{self.hidden_size});
        try writer.print("  Vocab size: {}\n", .{self.vocab_size});

        // Help hint
        try writer.writeAll("\nUse -v for module graph, -vv for kernel operations\n");
    }

    /// Estimate weight memory in bytes (accounts for quantization)
    pub fn estimateWeightMemory(self: *const Transformer) u64 {
        var total_params: u64 = 0;

        // Embedding
        total_params += self.embed_tokens.vocab_size * self.embed_tokens.embed_dim;

        // Layers
        if (self.firstLayerGeom()) |geom| {
            var layer_params: u64 = 0;

            // Per-layer weights
            layer_params += @intCast(geom.total_layer_params);

            // Norms (always f32)
            const norm_params: u64 = self.hidden_size * 2; // input + post-attn norm

            total_params += layer_params * self.num_hidden_layers;
            // Norm params are always f32
            total_params += norm_params * self.num_hidden_layers * 4 / self.bytesPerParam();
        }

        // Final norm (f32)
        total_params += self.hidden_size * 4 / self.bytesPerParam();

        // LM head
        if (self.lm_head) |head| {
            total_params += head.in_features * head.out_features;
        }

        return total_params * self.bytesPerParam();
    }

    /// Estimate scratch buffer memory for given sequence length
    pub fn estimateScratchMemory(self: *const Transformer, seq_len: usize) u64 {
        var total: u64 = 0;

        // Activation buffers (f32): tmp[0..2] (layer_tmp, norm_out, branch_out)
        // Each is [seq_len, hidden_size]
        const activation_buf = seq_len * self.hidden_size * 4;
        total += activation_buf * 3;

        if (self.firstLayerKernels()) |layer| {
            const attn = layer.attn;
            const ffn = layer.ffn;

            const kv_per_layer = seq_len * attn.n_kv_heads * attn.head_dim * 4 * 2; // f32, K+V
            total += kv_per_layer * self.num_hidden_layers;

            // Scores: [n_heads, seq_len, seq_len] for prefill
            total += attn.n_heads * seq_len * seq_len * 4;

            const ffn_size = switch (ffn.*) {
                .swiglu => |mlp| seq_len * mlp.d_ff * 4,
                .moe_ffn => |moe| seq_len * moe.d_ff * moe.experts_per_token * 4,
            };
            total += ffn_size;
        }

        return total;
    }

    fn bytesPerParam(self: *const Transformer) u64 {
        return switch (self.weight_dtype) {
            .grouped_affine_u4 => 1, // ~0.5 bytes, round up to 1 for scale overhead
            .grouped_affine_u8 => 1,
            .f8_e4m3 => 1,
            .f16, .bf16 => 2,
            .f32 => 4,
            else => 2,
        };
    }

    fn firstLayerKernels(self: *const Transformer) ?struct { attn: *const Attention, ffn: *const FFNLayer } {
        if (self.layers.len == 0) return null;
        const first = &self.layers[0];
        const attn = first.getAttention() orelse return null;
        const ffn = first.getFFN() orelse return null;
        return .{
            .attn = attn,
            .ffn = ffn,
        };
    }

    fn firstLayerGeom(self: *const Transformer) ?perf_estimate.LayerGeometry {
        const layer = self.firstLayerKernels() orelse return null;
        return perf_estimate.LayerGeometry.init(
            attnToConfig(layer.attn),
            ffnToConfig(layer.ffn),
        );
    }

    /// Convert runtime Attention to primitive AttnConfig for perf estimation.
    fn attnToConfig(attn: *const Attention) perf_estimate.AttnConfig {
        return .{
            .n_heads = attn.n_heads,
            .n_kv_heads = attn.n_kv_heads,
            .head_dim = attn.head_dim,
            .d_model = attn.d_model,
            .has_q_bias = attn.q_bias != null,
            .has_k_bias = attn.k_bias != null,
            .has_v_bias = attn.v_bias != null,
            .has_o_bias = false, // o_bias not present in MultiHeadAttention
        };
    }

    /// Convert runtime FFNLayer to primitive FfnConfig for perf estimation.
    fn ffnToConfig(ffn: *const FFNLayer) perf_estimate.FfnConfig {
        return switch (ffn.*) {
            .swiglu => |mlp| .{ .swiglu = .{
                .d_model = mlp.d_model,
                .d_ff = mlp.d_ff,
            } },
            .moe_ffn => |moe| .{ .moe_ffn = .{
                .d_model = moe.d_model,
                .d_ff = moe.d_ff,
                .num_experts = moe.num_experts,
                .experts_per_token = moe.experts_per_token,
            } },
        };
    }

    /// Estimate FLOPs for a forward pass with given sequence length.
    /// Returns struct with prefill and per-token decode FLOPs.
    pub fn estimateFlops(self: *const Transformer, seq_len: usize) PerfEstimate {
        return self.estimatePerf(seq_len);
    }

    /// Estimate performance characteristics (FLOPs and memory bandwidth) for inference.
    /// Returns struct with prefill and per-token decode estimates.
    pub fn estimatePerf(self: *const Transformer, seq_len: usize) PerfEstimate {
        const layer = self.firstLayerKernels() orelse return .{
            .prefill_flops = 0,
            .per_token_flops = 0,
            .prefill_mem_bytes = 0,
            .per_token_mem_bytes = 0,
            .seq_len = seq_len,
            .weight_dtype = self.weight_dtype,
        };
        return perf_estimate.estimatePerf(.{
            .seq_len = seq_len,
            .weight_dtype = self.weight_dtype,
            .hidden_size = self.hidden_size,
            .vocab_size = self.vocab_size,
            .num_hidden_layers = self.num_hidden_layers,
            .attn = attnToConfig(layer.attn),
            .ffn = ffnToConfig(layer.ffn),
        });
    }

    /// Build a Transformer from LoadedModel weights and CPU kernel blocks.
    pub fn build(
        allocator: std.mem.Allocator,
        loaded: *LoadedModel,
        blocks: []const cpu_blocks.TransformerBlock,
    ) !Transformer {
        const model_config = loaded.config;
        const layer_count = blocks.len;

        // Build layers as Block (with ops-based execution)
        var block_layers = try allocator.alloc(Block, layer_count);
        var built_layers: usize = 0;
        errdefer {
            for (block_layers[0..built_layers]) |*layer| layer.deinit(allocator);
            allocator.free(block_layers);
        }

        const static_entry = detectStaticTopologyEntry(loaded);

        // Get block program for each layer (handles heterogeneous models)
        // For heterogeneous models, each layer may have a different program (e.g., Mamba vs Attention).
        for (blocks, 0..) |*block, layer_idx| {
            const block_kind = block.block_type;
            const layer_program = try resolveLayerProgram(static_entry, block_kind);
            block_layers[layer_idx] = try buildBlock(allocator, block, model_config, layer_idx, layer_program, static_entry);
            built_layers = layer_idx + 1;
            try block_layers[layer_idx].validate();
        }

        // Build embedding
        const embed_tokens = Embedding.init(&loaded.token_embeddings);

        // Build final norm (optional — embed-only models may not have one)
        const final_norm: ?RMSNorm = if (loaded.ln_final) |*ln_f|
            RMSNorm{
                .weight = ln_f,
                .dim = @intCast(model_config.d_model),
                .eps = model_config.norm_eps,
                .weight_offset = loaded.runtime.weight_offset,
                .trace_point = .final_norm,
            }
        else
            null;

        // Build LM head if not tied (optional — embed-only models may not have one)
        const lm_head: ?Linear = if (loaded.lm_head) |*head|
            try Linear.init(head, null)
        else if (model_config.tie_word_embeddings)
            null
        else
            null;
        const per_layer_branch = try initPerLayerBranchRuntime(allocator, loaded, layer_count);
        const layer_scalars = try initStandaloneLayerScalars(allocator, loaded, layer_count);

        // Model type for debug output.
        const model_type: []const u8 = if (static_entry) |entry| entry.id else "TransformerModel";

        return .{
            .model_type = model_type,
            .embed_tokens = embed_tokens,
            .layers = block_layers,
            .norm = final_norm,
            .lm_head = lm_head,
            .tie_word_embeddings = model_config.tie_word_embeddings,
            .hidden_size = @intCast(model_config.d_model),
            .vocab_size = @intCast(model_config.vocab_size),
            .num_hidden_layers = layer_count,
            .weight_dtype = loaded.original_weight_dtype,
            .file_size = loaded.file_size,
            .tensor_count = loaded.tensor_count,
            .per_layer_branch = per_layer_branch,
            .layer_scalars = layer_scalars,
        };
    }

    /// Build a Block with ops-based execution from cpu_blocks.TransformerBlock
    fn buildBlock(
        allocator: std.mem.Allocator,
        block: *const cpu_blocks.TransformerBlock,
        model_config: ModelConfig,
        layer_idx: usize,
        program: []const LayerOp,
        static_entry: ?models.ModelDescriptor,
    ) !Block {
        return Block.initWithProgramOptions(
            allocator,
            block,
            layer_idx,
            @intCast(model_config.d_model),
            program,
            .decode,
            .{
                .state_descriptor_entry = static_entry,
            },
        );
    }

    fn detectStaticTopologyEntry(loaded: *const LoadedModel) ?models.ModelDescriptor {
        if (loaded.runtime.architecture_id) |arch_id| {
            return models.registry.detectByArchitectureId(arch_id);
        }
        return null;
    }

    fn resolveLayerProgram(
        static_entry: ?models.ModelDescriptor,
        block_kind: BlockKind,
    ) ![]const LayerOp {
        if (static_entry) |entry| {
            return models.registry.blockProgramFor(entry, block_kind) orelse error.UnsupportedModel;
        }
        return error.UnsupportedModel;
    }

    /// Free model allocated by build
    pub fn deinit(self: *Transformer, allocator: std.mem.Allocator) void {
        if (self.per_layer_branch) |*plb| plb.deinit(allocator);
        if (self.layer_scalars) |s| allocator.free(s);
        for (self.layers) |*layer| layer.deinit(allocator);
        allocator.free(self.layers);
        self.* = undefined;
    }
};

pub const Model = Transformer;

// =============================================================================
// Unit Tests
// =============================================================================

const testing = std.testing;
const expect = testing.expect;

/// Create a minimal mock model for testing inspection/estimation functions only.
/// Does not support forward passes - use for testing describe/summary/estimate* only.
fn createMockTransformer(num_layers: usize, hidden_size: usize, vocab_size: usize) Transformer {
    // Create a transformer with placeholder values - this is only for testing
    // the estimation functions which use the dimensions, not the actual layer data.
    const model = Transformer{
        .model_type = "test_model",
        .embed_tokens = .{
            .weight = undefined, // Not dereferenced in estimation functions
            .vocab_size = vocab_size,
            .embed_dim = hidden_size,
        },
        .layers = &[_]Block{}, // Empty layers slice - firstLayerKernels returns null
        .norm = undefined, // Not dereferenced in estimation
        .lm_head = null,
        .tie_word_embeddings = false,
        .hidden_size = hidden_size,
        .vocab_size = vocab_size,
        .num_hidden_layers = num_layers,
        .weight_dtype = .f32,
        .file_size = 1024 * 1024,
        .tensor_count = 10,
    };
    return model;
}

test "Transformer.summary writes model summary without layers" {
    const model = createMockTransformer(2, 64, 128);

    var buf: std.ArrayList(u8) = .{};
    defer buf.deinit(testing.allocator);

    try model.summary(buf.writer(testing.allocator));
    const output = buf.items;

    // Verify summary contains key information
    try expect(std.mem.indexOf(u8, output, "Model: test_model") != null);
    try expect(std.mem.indexOf(u8, output, "Quantization:") != null);
    try expect(std.mem.indexOf(u8, output, "Layers: 2") != null);
    try expect(std.mem.indexOf(u8, output, "Hidden size: 64") != null);
    try expect(std.mem.indexOf(u8, output, "Vocab size: 128") != null);
}

test "Transformer.summaryWithSeqLen writes model summary with custom seq_len" {
    const model = createMockTransformer(2, 64, 128);

    var buf: std.ArrayList(u8) = .{};
    defer buf.deinit(testing.allocator);

    try model.summaryWithSeqLen(buf.writer(testing.allocator), 1024);
    const output = buf.items;

    try expect(std.mem.indexOf(u8, output, "Model: test_model") != null);
    try expect(std.mem.indexOf(u8, output, "Quantization") != null);
}

test "Transformer.estimateWeightMemory returns reasonable value" {
    const model = createMockTransformer(2, 64, 128);
    const weight_mem = model.estimateWeightMemory();

    // Should be non-zero
    try expect(weight_mem > 0);

    // For f32 model with 128 vocab, 64 hidden, should be reasonable
    // Embedding: 128 * 64 * 4 = 32768 bytes
    // Should be at least embedding size
    try expect(weight_mem >= 32768);
}

test "Transformer.estimateScratchMemory returns reasonable value for seq_len" {
    const model = createMockTransformer(2, 64, 128);

    const scratch_mem_small = model.estimateScratchMemory(16);
    const scratch_mem_large = model.estimateScratchMemory(512);

    // Should be non-zero
    try expect(scratch_mem_small > 0);
    try expect(scratch_mem_large > 0);

    // Larger seq_len should require more scratch memory
    try expect(scratch_mem_large > scratch_mem_small);
}

test "Transformer.estimateFlops returns PerfEstimate struct with no layers" {
    const model = createMockTransformer(2, 64, 128);
    const perf = model.estimateFlops(128);

    // Verify fields are populated
    try expect(perf.seq_len == 128);
    try expect(perf.weight_dtype == .f32);

    // With no layers, FLOPs should be zero
    try expect(perf.prefill_flops == 0);
    try expect(perf.per_token_flops == 0);
}

test "Transformer.estimatePerf returns PerfEstimate with memory estimates" {
    const model = createMockTransformer(2, 64, 128);
    const perf = model.estimatePerf(128);

    try expect(perf.seq_len == 128);
    try expect(perf.weight_dtype == .f32);

    // With no layers (firstLayerKernels returns null), should return zeros
    try expect(perf.prefill_flops == 0);
    try expect(perf.per_token_flops == 0);
    try expect(perf.prefill_mem_bytes == 0);
    try expect(perf.per_token_mem_bytes == 0);
}

test "Transformer.estimateWeightMemory handles quantized dtypes" {
    var model = createMockTransformer(1, 64, 128);
    const weight_mem_f32 = model.estimateWeightMemory();

    // Test with different dtypes
    model.weight_dtype = .f16;
    const weight_mem_f16 = model.estimateWeightMemory();

    model.weight_dtype = .grouped_affine_u4;
    const weight_mem_q4 = model.estimateWeightMemory();

    // f32 should be larger than f16
    try expect(weight_mem_f32 > weight_mem_f16);

    // f16 should be larger than q4
    try expect(weight_mem_f16 > weight_mem_q4);
}

test "Transformer.estimateScratchMemory scales with seq_len and hidden_size" {
    const model_small = createMockTransformer(1, 32, 64);
    const model_large = createMockTransformer(1, 128, 256);

    const scratch_small = model_small.estimateScratchMemory(16);
    const scratch_large = model_large.estimateScratchMemory(16);

    // Larger model should require more scratch memory
    try expect(scratch_large > scratch_small);
}

test "estimateWeightMemory bytesPerParam dtypes" {
    var model = createMockTransformer(1, 64, 128);

    model.weight_dtype = .f32;
    try expect(model.bytesPerParam() == 4);

    model.weight_dtype = .f16;
    try expect(model.bytesPerParam() == 2);

    model.weight_dtype = .bf16;
    try expect(model.bytesPerParam() == 2);

    model.weight_dtype = .grouped_affine_u4;
    try expect(model.bytesPerParam() == 1);

    model.weight_dtype = .grouped_affine_u8;
    try expect(model.bytesPerParam() == 1);
}

test "Transformer.resolveLayeredCache resolves typed runtime state without builtin id routing" {
    var layered_cache = try LayeredBatchedKVCache.init(std.testing.allocator, 1, 1, 1, 1, 8);
    defer layered_cache.deinit();
    var scratch: ScratchBuffer = undefined;
    var storage: [@sizeOf(state_bindings.KvRuntimeState)]u8 align(64) =
        [_]u8{0} ** @sizeOf(state_bindings.KvRuntimeState);
    var state_block = runtime_contract.StateBlockHandle{
        .id = 77,
        .ptr = @ptrCast(storage[0..].ptr),
        .size = @sizeOf(state_bindings.KvRuntimeState),
        .align_bytes = 64,
    };
    const runtime_state = runtime_contract.stateValueFromBlock(*state_bindings.KvRuntimeState, &state_block) orelse {
        return error.TestUnexpectedResult;
    };
    runtime_state.* = .{
        .runtime_kind = runtime_contract.state_runtime_kind_kv_cache,
        .layered_cache = &layered_cache,
        .scratch = &scratch,
        .slot_index = 0,
    };

    const resolved = Transformer.resolveLayeredCache(&.{state_block}) orelse {
        return error.TestUnexpectedResult;
    };
    try std.testing.expectEqual(@intFromPtr(&layered_cache), @intFromPtr(resolved));
}

test "Transformer.resolveLayeredCache ignores non-kv runtime state blocks" {
    var layered_cache = try LayeredBatchedKVCache.init(std.testing.allocator, 1, 1, 1, 1, 8);
    defer layered_cache.deinit();
    var scratch: ScratchBuffer = undefined;
    var storage: [@sizeOf(state_bindings.KvRuntimeState)]u8 align(64) =
        [_]u8{0} ** @sizeOf(state_bindings.KvRuntimeState);
    var state_block = runtime_contract.StateBlockHandle{
        .id = 78,
        .ptr = @ptrCast(storage[0..].ptr),
        .size = @sizeOf(state_bindings.KvRuntimeState),
        .align_bytes = 64,
    };
    const runtime_state = runtime_contract.stateValueFromBlock(*state_bindings.KvRuntimeState, &state_block) orelse {
        return error.TestUnexpectedResult;
    };
    runtime_state.* = .{
        .runtime_kind = runtime_contract.state_runtime_kind_shortconv_cache,
        .layered_cache = &layered_cache,
        .scratch = &scratch,
        .slot_index = 0,
    };

    try std.testing.expect(Transformer.resolveLayeredCache(&.{state_block}) == null);
}

test "forwardWithBatchedCache resets only target slot recurrent state when use_cache=false" {
    const allocator = std.testing.allocator;
    const model = createMockTransformer(1, 8, 16);
    var scratch = try ScratchBuffer.initWithSlots(allocator, 8, 16, 1, 2);
    defer scratch.deinit();

    const gd_config = @import("../kernels/gated_delta.zig").GatedDeltaConfig{
        .d_model = 8,
        .d_conv = 4,
        .n_heads = 2,
        .d_head = 2,
    };
    try scratch.initGatedDelta(&.{0}, gd_config);

    const slot0 = scratch.getSlotLayerState(0, 0) orelse return error.TestUnexpectedResult;
    const slot1 = scratch.getSlotLayerState(1, 0) orelse return error.TestUnexpectedResult;
    const slot0_state = &(slot0.gated_delta_state orelse return error.TestUnexpectedResult);
    const slot1_state = &(slot1.gated_delta_state orelse return error.TestUnexpectedResult);
    @memset(slot0_state.conv_state, 1.0);
    @memset(slot0_state.ssm_state, 1.0);
    @memset(slot1_state.conv_state, 2.0);
    @memset(slot1_state.ssm_state, 2.0);

    var input_storage = [_]f32{ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 };
    var output_storage = [_]f32{0.0} ** 8;
    const input_tensor = Tensor.view3DSlice(input_storage[0..], 1, 8);
    var output_tensor = Tensor.view3DSlice(output_storage[0..], 1, 8);
    try model.forwardWithBatchedCache(
        &input_tensor,
        &output_tensor,
        &scratch,
        &.{},
        0,
        false,
    );

    for (slot0_state.conv_state) |value| {
        try std.testing.expectEqual(@as(f32, 0.0), value);
    }
    for (slot0_state.ssm_state) |value| {
        try std.testing.expectEqual(@as(f32, 0.0), value);
    }
    for (slot1_state.conv_state) |value| {
        try std.testing.expectEqual(@as(f32, 2.0), value);
    }
    for (slot1_state.ssm_state) |value| {
        try std.testing.expectEqual(@as(f32, 2.0), value);
    }
}

test "forwardWithBatchedCacheLayerRange rejects invalid layer ranges" {
    const allocator = std.testing.allocator;
    const model = createMockTransformer(0, 8, 16);
    var scratch = try ScratchBuffer.init(allocator, 8, 16, 1);
    defer scratch.deinit();

    var input_storage = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var output_storage = [_]f32{0} ** 8;
    const input_tensor = Tensor.view3DSlice(input_storage[0..], 1, 8);
    var output_tensor = Tensor.view3DSlice(output_storage[0..], 1, 8);

    try std.testing.expectError(
        error.InvalidArgument,
        model.forwardWithBatchedCacheLayerRange(
            &input_tensor,
            &output_tensor,
            &scratch,
            &.{},
            0,
            true,
            1,
            0,
        ),
    );
    try std.testing.expectError(
        error.InvalidArgument,
        model.forwardWithBatchedCacheLayerRange(
            &input_tensor,
            &output_tensor,
            &scratch,
            &.{},
            0,
            true,
            0,
            1,
        ),
    );
}

test "forwardWithBatchedCacheLayerRange processes valid empty range" {
    const allocator = std.testing.allocator;
    const model = createMockTransformer(0, 8, 16);
    var scratch = try ScratchBuffer.init(allocator, 8, 16, 1);
    defer scratch.deinit();

    var input_storage = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var output_storage = [_]f32{0} ** 8;
    const input_tensor = Tensor.view3DSlice(input_storage[0..], 1, 8);
    var output_tensor = Tensor.view3DSlice(output_storage[0..], 1, 8);

    try model.forwardWithBatchedCacheLayerRange(
        &input_tensor,
        &output_tensor,
        &scratch,
        &.{},
        0,
        true,
        0,
        0,
    );
    try std.testing.expectEqualSlices(f32, input_storage[0..], output_storage[0..]);
}

test "snapshotSourceEmbeddings preserves source when input buffer mutates" {
    const allocator = std.testing.allocator;

    var plb_scratch = try PerLayerBranchScratch.init(allocator, 2, 4, 2);
    defer plb_scratch.deinit(allocator);

    var input_storage = [_]f32{
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
    };
    const input_tensor = Tensor.view3DSlice(input_storage[0..], 2, 4);

    const snapshot_tensor = try snapshotSourceEmbeddings(&plb_scratch, &input_tensor);
    const snapshot_values = snapshot_tensor.asSlice(f32);

    @memset(input_storage[0..], -1.0);

    try std.testing.expectEqualSlices(f32, &[_]f32{
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
    }, snapshot_values);
}
