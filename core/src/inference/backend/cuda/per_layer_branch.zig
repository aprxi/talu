//! Generic CUDA helpers for the per-layer branch feature.

const std = @import("std");
const models = @import("models_pkg");
const compute = @import("compute_pkg");
const tensor = @import("compute_pkg").tensor;
const dtype = @import("compute_pkg").dtype;
const log = @import("log_pkg");
const st_loader = @import("io_pkg").safetensors.root;

const Tensor = tensor.Tensor;
const per_layer_branch_models = models.per_layer_branch;
const engine_types = @import("runtime/root.zig");
const DeviceTensor = engine_types.DeviceTensor;
const LinearWeight = engine_types.LinearWeight;
const engine_ops = @import("operators/root.zig");
const engine_weights = @import("weights/root.zig");
const bufferSlice = engine_weights.bufferSlice;
const uploadLinearWeight = engine_weights.uploadLinearWeight;
const uploadVectorTensor = engine_weights.uploadVectorTensor;
const materializeTensorF32 = engine_weights.materializeTensorF32;

pub const PerLayerBranchRuntime = struct {
    hidden_size_per_layer_input: usize,
    per_layer_input_scale: f32,
    per_layer_embed_add_scale: f32,
    per_layer_embedding: Tensor,
    per_layer_embedding_width: usize,
    per_layer_projection_norm_weight: DeviceTensor,
    per_layer_model_projection: []LinearWeight,
    per_layer_input_gate: []LinearWeight,
    per_layer_projection: []LinearWeight,
    post_per_layer_input_norm_weight: []DeviceTensor,
    layer_scalars: []f32,
    layer_offset: usize,

    pub fn deinit(self: *PerLayerBranchRuntime, allocator: std.mem.Allocator, device: *compute.cuda.Device) void {
        for (self.post_per_layer_input_norm_weight) |*weight| weight.deinit(device);
        for (self.per_layer_projection) |*weight| weight.deinit(device);
        for (self.per_layer_input_gate) |*weight| weight.deinit(device);
        for (self.per_layer_model_projection) |*weight| weight.deinit(device);
        allocator.free(self.post_per_layer_input_norm_weight);
        allocator.free(self.per_layer_projection);
        allocator.free(self.per_layer_input_gate);
        allocator.free(self.per_layer_model_projection);
        allocator.free(self.layer_scalars);
        self.per_layer_projection_norm_weight.deinit(device);
        self.* = undefined;
    }
};

fn tensorScalarToF32Cuda(tensor_value: *const Tensor) !f32 {
    if (tensor_value.numel < 1) return error.InvalidShape;
    return switch (tensor_value.dtype) {
        .f32 => tensor_value.asSlice(f32)[0],
        .bf16 => dtype.bf16ToF32(tensor_value.asSliceUnaligned(u16)[0]),
        .f16 => dtype.fp16ToF32(tensor_value.asSliceUnaligned(u16)[0]),
        else => error.UnsupportedDType,
    };
}

fn loadTensorAnyCuda(
    safetensors: *st_loader.UnifiedSafeTensors,
    names: []const []const u8,
) !Tensor {
    for (names) |name| {
        const maybe = safetensors.getTensor(name, null) catch null;
        if (maybe) |tensor_value| return tensor_value;
    }
    return error.MissingWeight;
}

fn loadLayerTensorBySuffixCuda(
    safetensors: *st_loader.UnifiedSafeTensors,
    prefixes: []const []const u8,
    layer_idx: usize,
    suffix: []const u8,
) !Tensor {
    var name_buf: [256]u8 = undefined;
    for (prefixes) |prefix| {
        const full_name = std.fmt.bufPrint(name_buf[0..], "{s}.{d}.{s}", .{ prefix, layer_idx, suffix }) catch continue;
        if (safetensors.getTensor(full_name, null) catch null) |tensor_value| return tensor_value;
    }
    return error.MissingWeight;
}

fn matrixRowsViewCuda(
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

fn uploadScaledVectorTensor(
    device: *compute.cuda.Device,
    allocator: std.mem.Allocator,
    src: *const Tensor,
    scale: f32,
) !DeviceTensor {
    const values = try materializeTensorF32(allocator, src);
    defer allocator.free(values);
    for (values) |*value| value.* *= scale;
    var buffer = try device.allocBuffer(values.len * @sizeOf(f32));
    errdefer buffer.deinit(device);
    try buffer.upload(device, std.mem.sliceAsBytes(values));
    return .{
        .rows = values.len,
        .cols = 1,
        .buffer = buffer,
    };
}

fn uploadScaledLinearWeight(
    device: *compute.cuda.Device,
    allocator: std.mem.Allocator,
    src: *const Tensor,
    input_dim: usize,
    scale: f32,
) !LinearWeight {
    if (src.n_dims != 2) return error.InvalidShape;
    if (src.shape[0] <= 0 or src.shape[1] <= 0) return error.InvalidShape;
    const rows: usize = @intCast(src.shape[0]);
    const cols: usize = @intCast(src.shape[1]);
    const host = try materializeTensorF32(allocator, src);
    defer allocator.free(host);
    for (host) |*value| value.* *= scale;
    var scaled = Tensor.view2DSlice(host, rows, cols);
    return uploadLinearWeight(device, allocator, &scaled, input_dim);
}

pub fn hasPerLayerBranchRuntime(self: anytype) bool {
    const SelfType = @TypeOf(self.*);
    if (comptime @hasField(SelfType, "per_layer_branch_runtime")) {
        return self.per_layer_branch_runtime != null;
    }
    return false;
}

pub fn hasStandaloneLayerScalars(self: anytype) bool {
    const SelfType = @TypeOf(self.*);
    if (comptime @hasField(SelfType, "enable_layer_scalars") and @hasField(SelfType, "standalone_layer_scalars")) {
        return self.enable_layer_scalars and self.standalone_layer_scalars != null;
    }
    return false;
}

pub fn initPerLayerBranchRuntime(
    self: anytype,
    layer_count: usize,
    layer_offset: usize,
) !?PerLayerBranchRuntime {
    if (self.loaded.config.hidden_size_per_layer_input <= 0) return null;
    const per_layer_branch_spec = per_layer_branch_models.specForLoadedModel(self.loaded) orelse return null;
    if (self.loaded.st == null) return error.MissingWeight;
    const safetensors = &(self.loaded.st.?);

    const hidden_size_per_layer_input: usize = @intCast(self.loaded.config.hidden_size_per_layer_input);
    const hidden_size = self.d_model;
    const required_width = std.math.mul(usize, layer_offset + layer_count, hidden_size_per_layer_input) catch return error.InvalidArgument;

    const per_layer_embedding = try loadTensorAnyCuda(safetensors, per_layer_branch_spec.per_layer_embedding_names);
    const per_layer_model_projection_weight = try loadTensorAnyCuda(safetensors, per_layer_branch_spec.per_layer_model_projection_names);
    const per_layer_projection_norm_weight = try loadTensorAnyCuda(safetensors, per_layer_branch_spec.per_layer_projection_norm_names);

    if (per_layer_embedding.n_dims != 2) return error.InvalidShape;
    if (per_layer_embedding.shape[0] <= 0 or per_layer_embedding.shape[1] <= 0) return error.InvalidShape;
    const per_layer_embedding_width: usize = @intCast(per_layer_embedding.shape[1]);
    if (per_layer_embedding_width < required_width) return error.InvalidShape;
    if (per_layer_model_projection_weight.n_dims != 2 or
        @as(usize, @intCast(per_layer_model_projection_weight.shape[0])) < required_width or
        @as(usize, @intCast(per_layer_model_projection_weight.shape[1])) != hidden_size)
    {
        return error.InvalidShape;
    }
    if (per_layer_projection_norm_weight.n_dims != 1 or
        @as(usize, @intCast(per_layer_projection_norm_weight.shape[0])) != hidden_size_per_layer_input)
    {
        return error.InvalidShape;
    }

    const per_layer_model_projection = try self.allocator.alloc(LinearWeight, layer_count);
    errdefer self.allocator.free(per_layer_model_projection);
    const per_layer_input_gate = try self.allocator.alloc(LinearWeight, layer_count);
    errdefer self.allocator.free(per_layer_input_gate);
    const per_layer_projection = try self.allocator.alloc(LinearWeight, layer_count);
    errdefer self.allocator.free(per_layer_projection);
    const post_per_layer_input_norm_weight = try self.allocator.alloc(DeviceTensor, layer_count);
    errdefer self.allocator.free(post_per_layer_input_norm_weight);
    const layer_scalars = try self.allocator.alloc(f32, layer_count);
    errdefer self.allocator.free(layer_scalars);

    var built_layers: usize = 0;
    errdefer {
        for (0..built_layers) |idx| {
            post_per_layer_input_norm_weight[idx].deinit(&self.device);
            per_layer_projection[idx].deinit(&self.device);
            per_layer_input_gate[idx].deinit(&self.device);
            per_layer_model_projection[idx].deinit(&self.device);
        }
    }

    const per_layer_input_scale: f32 = per_layer_branch_spec.per_layer_input_scale;
    const per_layer_model_projection_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hidden_size)));
    var per_layer_projection_norm_weight_dev = try uploadVectorTensor(
        &self.device,
        self.allocator,
        &per_layer_projection_norm_weight,
        hidden_size_per_layer_input,
    );
    errdefer per_layer_projection_norm_weight_dev.deinit(&self.device);

    for (0..layer_count) |layer_idx| {
        const global_layer_idx = layer_offset + layer_idx;
        const gate_weight = try loadLayerTensorBySuffixCuda(safetensors, per_layer_branch_spec.layer_prefixes, global_layer_idx, per_layer_branch_spec.per_layer_input_gate_suffix);
        const projection_weight = try loadLayerTensorBySuffixCuda(safetensors, per_layer_branch_spec.layer_prefixes, global_layer_idx, per_layer_branch_spec.per_layer_projection_suffix);
        const post_norm_weight = try loadLayerTensorBySuffixCuda(safetensors, per_layer_branch_spec.layer_prefixes, global_layer_idx, per_layer_branch_spec.post_per_layer_input_norm_suffix);
        const layer_scalar = try loadLayerTensorBySuffixCuda(safetensors, per_layer_branch_spec.layer_prefixes, global_layer_idx, per_layer_branch_spec.layer_scalar_suffix);

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

        const row_start = std.math.mul(usize, global_layer_idx, hidden_size_per_layer_input) catch return error.InvalidArgument;
        const projection_view = try matrixRowsViewCuda(
            &per_layer_model_projection_weight,
            row_start,
            hidden_size_per_layer_input,
            hidden_size,
        );

        per_layer_model_projection[layer_idx] = try uploadScaledLinearWeight(
            &self.device,
            self.allocator,
            &projection_view,
            hidden_size,
            per_layer_model_projection_scale,
        );
        per_layer_input_gate[layer_idx] = try uploadLinearWeight(
            &self.device,
            self.allocator,
            &gate_weight,
            hidden_size,
        );
        per_layer_projection[layer_idx] = try uploadLinearWeight(
            &self.device,
            self.allocator,
            &projection_weight,
            hidden_size_per_layer_input,
        );
        post_per_layer_input_norm_weight[layer_idx] = try uploadVectorTensor(
            &self.device,
            self.allocator,
            &post_norm_weight,
            hidden_size,
        );
        layer_scalars[layer_idx] = try tensorScalarToF32Cuda(&layer_scalar);
        built_layers += 1;
    }

    return .{
        .hidden_size_per_layer_input = hidden_size_per_layer_input,
        .per_layer_input_scale = per_layer_input_scale,
        .per_layer_embed_add_scale = @sqrt(@as(f32, @floatFromInt(hidden_size_per_layer_input))),
        .per_layer_embedding = per_layer_embedding,
        .per_layer_embedding_width = per_layer_embedding_width,
        .per_layer_projection_norm_weight = per_layer_projection_norm_weight_dev,
        .per_layer_model_projection = per_layer_model_projection,
        .per_layer_input_gate = per_layer_input_gate,
        .per_layer_projection = per_layer_projection,
        .post_per_layer_input_norm_weight = post_per_layer_input_norm_weight,
        .layer_scalars = layer_scalars,
        .layer_offset = layer_offset,
    };
}

pub fn deinitPerLayerBranchRuntime(self: anytype) void {
    if (self.per_layer_branch_runtime) |*branch_runtime| {
        branch_runtime.deinit(self.allocator, &self.device);
        self.per_layer_branch_runtime = null;
    }
}

pub fn initStandaloneLayerScalars(
    self: anytype,
    layer_count: usize,
    layer_offset: usize,
) !?[]f32 {
    if (self.loaded.config.hidden_size_per_layer_input > 0) return null;
    const per_layer_branch_spec = per_layer_branch_models.specForLoadedModel(self.loaded) orelse return null;
    if (self.loaded.st == null) return null;
    const safetensors = &(self.loaded.st.?);
    _ = loadLayerTensorBySuffixCuda(safetensors, per_layer_branch_spec.layer_prefixes, 0, per_layer_branch_spec.layer_scalar_suffix) catch return null;

    const scalars = try self.allocator.alloc(f32, layer_count);
    errdefer self.allocator.free(scalars);
    for (0..layer_count) |layer_idx| {
        const global_layer_idx = layer_offset + layer_idx;
        const t = try loadLayerTensorBySuffixCuda(safetensors, per_layer_branch_spec.layer_prefixes, global_layer_idx, per_layer_branch_spec.layer_scalar_suffix);
        scalars[layer_idx] = try tensorScalarToF32Cuda(&t);
    }
    return scalars;
}

pub fn initStandaloneLayerScalarFusionMap(self: anytype, scalars: []const f32) ![]bool {
    const fused = try self.allocator.alloc(bool, scalars.len);
    errdefer self.allocator.free(fused);
    @memset(fused, false);

    const layer_count = @min(scalars.len, self.block_runtime.blocks.len);
    for (0..layer_count) |layer_idx| {
        if (scalars[layer_idx] == 1.0) continue;
        const compiled = self.block_runtime.blocks[layer_idx].compiled_plan orelse continue;
        const instructions = compiled.plan.instructions;
        if (instructions.len == 0) continue;
        if (instructions[instructions.len - 1].opcode != .residual_add) continue;
        fused[layer_idx] = true;
    }
    return fused;
}

pub fn ensurePerLayerEmbedAddHostCapacity(self: anytype, elements: usize) !void {
    if (elements == 0) return error.InvalidArgument;
    if (self.per_layer_branch_embed_add_host.len >= elements) return;
    if (self.per_layer_branch_embed_add_host.len > 0) self.allocator.free(self.per_layer_branch_embed_add_host);
    self.per_layer_branch_embed_add_host = try self.allocator.alloc(f32, elements);
}

pub fn gatherPerLayerEmbedding(
    self: anytype,
    branch_runtime: *const PerLayerBranchRuntime,
    token_ids: []const u32,
    layer_idx: usize,
    out: []f32,
) !void {
    _ = self;
    if (layer_idx >= branch_runtime.per_layer_model_projection.len) return error.InvalidArgument;
    const hpl = branch_runtime.hidden_size_per_layer_input;
    const required = std.math.mul(usize, token_ids.len, hpl) catch return error.InvalidArgument;
    if (out.len < required) return error.InvalidArgument;
    const vocab_limit: usize = @intCast(branch_runtime.per_layer_embedding.shape[0]);
    const global_layer_idx = branch_runtime.layer_offset + layer_idx;
    const layer_offset = std.math.mul(usize, global_layer_idx, hpl) catch return error.InvalidArgument;

    switch (branch_runtime.per_layer_embedding.dtype) {
        .f32 => {
            const values = branch_runtime.per_layer_embedding.asSlice(f32);
            for (token_ids, 0..) |token_id, row_idx| {
                if (token_id >= vocab_limit) return error.InvalidToken;
                const src_base = std.math.mul(usize, @as(usize, token_id), branch_runtime.per_layer_embedding_width) catch return error.InvalidArgument;
                const dst_base = std.math.mul(usize, row_idx, hpl) catch return error.InvalidArgument;
                for (0..hpl) |col| {
                    out[dst_base + col] = values[src_base + layer_offset + col] * branch_runtime.per_layer_embed_add_scale;
                }
            }
        },
        .bf16 => {
            const values = branch_runtime.per_layer_embedding.asSliceUnaligned(u16);
            for (token_ids, 0..) |token_id, row_idx| {
                if (token_id >= vocab_limit) return error.InvalidToken;
                const src_base = std.math.mul(usize, @as(usize, token_id), branch_runtime.per_layer_embedding_width) catch return error.InvalidArgument;
                const dst_base = std.math.mul(usize, row_idx, hpl) catch return error.InvalidArgument;
                for (0..hpl) |col| {
                    out[dst_base + col] = dtype.bf16ToF32(values[src_base + layer_offset + col]) * branch_runtime.per_layer_embed_add_scale;
                }
            }
        },
        .f16 => {
            const values = branch_runtime.per_layer_embedding.asSliceUnaligned(u16);
            for (token_ids, 0..) |token_id, row_idx| {
                if (token_id >= vocab_limit) return error.InvalidToken;
                const src_base = std.math.mul(usize, @as(usize, token_id), branch_runtime.per_layer_embedding_width) catch return error.InvalidArgument;
                const dst_base = std.math.mul(usize, row_idx, hpl) catch return error.InvalidArgument;
                for (0..hpl) |col| {
                    out[dst_base + col] = dtype.fp16ToF32(values[src_base + layer_offset + col]) * branch_runtime.per_layer_embed_add_scale;
                }
            }
        },
        else => return error.UnsupportedDType,
    }
}

pub fn maybeCapturePerLayerSourceEmbeddings(self: anytype, rows: usize) !?compute.cuda.Buffer {
    _ = self.per_layer_branch_runtime orelse return null;
    if (rows == 0) return error.InvalidArgument;
    const row_bytes = std.math.mul(usize, self.d_model, @sizeOf(f32)) catch return error.InvalidArgument;
    const bytes = std.math.mul(usize, rows, row_bytes) catch return error.InvalidArgument;
    var src = try bufferSlice(&self.runtime_buffers.input_dev, 0, bytes);
    var dst = try bufferSlice(&self.runtime_buffers.deepstack_add_dev, 0, bytes);
    try compute.cuda.copy.runWithFunction(
        &self.kernel_arg_pack,
        &self.device,
        self.copy_function orelse return error.CudaKernelUnavailable,
        &src,
        &dst,
        std.math.mul(u32, @intCast(rows), @intCast(self.d_model)) catch return error.InvalidArgument,
    );
    return dst;
}

pub fn capturePerLayerSourceEmbeddingsForLocalStage(self: anytype, token: u32) !?compute.cuda.Buffer {
    _ = self.per_layer_branch_runtime orelse return null;
    const used = try engine_weights.tryPopulateHiddenFromToken(self.loaded, token, self.runtime_buffers.hidden_host);
    if (!used) {
        log.warn("inference", "capturePerLayerSourceEmbeddingsForLocalStage: tryPopulateHiddenFromToken returned false", .{
            .token = token,
            .embed_dtype = @tagName(self.loaded.token_embeddings.dtype),
            .embed_ndim = self.loaded.token_embeddings.n_dims,
            .embed_data_ptr_null = self.loaded.token_embeddings.data_ptr == null,
        });
        return error.UnsupportedModel;
    }
    if (self.loaded.config.embedding_multiplier != 1.0) {
        for (self.runtime_buffers.hidden_host) |*v| v.* *= self.loaded.config.embedding_multiplier;
    }
    const row_bytes = std.math.mul(usize, self.d_model, @sizeOf(f32)) catch return error.InvalidArgument;
    var dst = try bufferSlice(&self.runtime_buffers.deepstack_add_dev, 0, row_bytes);
    try dst.upload(&self.device, std.mem.sliceAsBytes(self.runtime_buffers.hidden_host));
    return dst;
}

pub fn applyPerLayerBranch(
    self: anytype,
    layer_idx: usize,
    token_ids: []const u32,
    source_embeddings: *const compute.cuda.Buffer,
    hidden_rows: *compute.cuda.Buffer,
) !void {
    const branch_runtime = self.per_layer_branch_runtime orelse return;
    if (layer_idx >= branch_runtime.per_layer_model_projection.len) return error.InvalidArgument;
    const rows = token_ids.len;
    if (rows == 0) return error.InvalidArgument;
    const rows_u32: u32 = @intCast(rows);
    const hpl = branch_runtime.hidden_size_per_layer_input;
    const hpl_u32: u32 = @intCast(hpl);
    const d_model_u32: u32 = @intCast(self.d_model);
    const hpl_bytes = std.math.mul(usize, std.math.mul(usize, rows, hpl) catch return error.InvalidArgument, @sizeOf(f32)) catch return error.InvalidArgument;
    const hidden_bytes = std.math.mul(usize, std.math.mul(usize, rows, self.d_model) catch return error.InvalidArgument, @sizeOf(f32)) catch return error.InvalidArgument;

    var projection = try bufferSlice(&self.runtime_buffers.ffn_gate_dev, 0, hpl_bytes);
    var per_layer_input = try bufferSlice(&self.runtime_buffers.ffn_up_dev, 0, hpl_bytes);
    var gated = try bufferSlice(&self.runtime_buffers.ffn_mul_dev, 0, hpl_bytes);
    var embed_add = try bufferSlice(&self.runtime_buffers.query_gate_proj_dev, 0, hpl_bytes);
    var branch = try bufferSlice(&self.runtime_buffers.ffn_down_dev, 0, hidden_bytes);
    var branch_norm = try bufferSlice(&self.runtime_buffers.norm_out_dev, 0, hidden_bytes);

    try engine_ops.linearForwardRows(
        self,
        source_embeddings,
        rows,
        &branch_runtime.per_layer_model_projection[layer_idx],
        &projection,
    );
    try compute.cuda.rmsnorm.runWithFunction(
        &self.kernel_arg_pack,
        &self.device,
        self.rmsnorm_function orelse return error.CudaKernelUnavailable,
        &projection,
        &branch_runtime.per_layer_projection_norm_weight.buffer,
        &per_layer_input,
        rows_u32,
        hpl_u32,
        self.norm_eps,
        0.0,
    );

    const embed_count = std.math.mul(usize, rows, hpl) catch return error.InvalidArgument;
    try ensurePerLayerEmbedAddHostCapacity(self, embed_count);
    const embed_host = self.per_layer_branch_embed_add_host[0..embed_count];
    try gatherPerLayerEmbedding(self, &branch_runtime, token_ids, layer_idx, embed_host);
    try embed_add.upload(&self.device, std.mem.sliceAsBytes(embed_host));
    try compute.cuda.vector_add.runWithFunction(
        &self.kernel_arg_pack,
        &self.device,
        self.vector_add_function orelse return error.CudaKernelUnavailable,
        &per_layer_input,
        &embed_add,
        &per_layer_input,
        @intCast(embed_count),
    );
    if (branch_runtime.per_layer_input_scale != 1.0) {
        try compute.cuda.vector_add_scaled.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            self.vector_add_scaled_function orelse return error.CudaKernelUnavailable,
            &per_layer_input,
            &per_layer_input,
            &per_layer_input,
            branch_runtime.per_layer_input_scale - 1.0,
            @intCast(embed_count),
        );
    }

    try engine_ops.linearForwardRows(
        self,
        hidden_rows,
        rows,
        &branch_runtime.per_layer_input_gate[layer_idx],
        &gated,
    );
    if (self.loaded.config.use_gelu) {
        try compute.cuda.gelu_mul.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            self.gelu_mul_function orelse return error.CudaKernelUnavailable,
            &gated,
            &per_layer_input,
            &gated,
            @intCast(embed_count),
        );
    } else {
        try compute.cuda.silu_mul.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            self.silu_mul_function orelse return error.CudaKernelUnavailable,
            &gated,
            &per_layer_input,
            &gated,
            @intCast(embed_count),
        );
    }

    try engine_ops.linearForwardRows(
        self,
        &gated,
        rows,
        &branch_runtime.per_layer_projection[layer_idx],
        &branch,
    );
    try compute.cuda.rmsnorm.runWithFunction(
        &self.kernel_arg_pack,
        &self.device,
        self.rmsnorm_function orelse return error.CudaKernelUnavailable,
        &branch,
        &branch_runtime.post_per_layer_input_norm_weight[layer_idx].buffer,
        &branch_norm,
        rows_u32,
        d_model_u32,
        self.norm_eps,
        0.0,
    );
    try compute.cuda.vector_add.runWithFunction(
        &self.kernel_arg_pack,
        &self.device,
        self.vector_add_function orelse return error.CudaKernelUnavailable,
        hidden_rows,
        &branch_norm,
        hidden_rows,
        std.math.mul(u32, rows_u32, d_model_u32) catch return error.InvalidArgument,
    );

    const layer_scalar = branch_runtime.layer_scalars[layer_idx];
    if (self.enable_layer_scalars and layer_scalar != 1.0) {
        const layer_scalar_start_ns: i128 = std.time.nanoTimestamp();
        try compute.cuda.vector_add_scaled.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            self.vector_add_scaled_function orelse return error.CudaKernelUnavailable,
            hidden_rows,
            hidden_rows,
            hidden_rows,
            layer_scalar - 1.0,
            std.math.mul(u32, rows_u32, d_model_u32) catch return error.InvalidArgument,
        );
        const layer_scalar_elapsed_ns: u64 = @intCast(std.time.nanoTimestamp() - layer_scalar_start_ns);
        self.nvfp4_phase_counters.recordLayerScalar(layer_scalar_elapsed_ns);
    }
}

pub fn applyStandaloneLayerScalar(
    self: anytype,
    layer_idx: usize,
    hidden_rows: *compute.cuda.Buffer,
    rows: usize,
) !void {
    if (!self.enable_layer_scalars) return;
    if (self.standalone_layer_scalar_fused_layers) |fused| {
        if (layer_idx < fused.len and fused[layer_idx]) return;
    }
    const scalars = self.standalone_layer_scalars orelse return;
    if (layer_idx >= scalars.len) return;
    const scalar = scalars[layer_idx];
    if (scalar == 1.0) return;
    const d_model_u32: u32 = @intCast(self.d_model);
    const rows_u32: u32 = @intCast(rows);
    const layer_scalar_start_ns: i128 = std.time.nanoTimestamp();
    try compute.cuda.vector_add_scaled.runWithFunction(
        &self.kernel_arg_pack,
        &self.device,
        self.vector_add_scaled_function orelse return error.CudaKernelUnavailable,
        hidden_rows,
        hidden_rows,
        hidden_rows,
        scalar - 1.0,
        std.math.mul(u32, rows_u32, d_model_u32) catch return error.InvalidArgument,
    );
    const layer_scalar_elapsed_ns: u64 = @intCast(std.time.nanoTimestamp() - layer_scalar_start_ns);
    self.nvfp4_phase_counters.recordLayerScalar(layer_scalar_elapsed_ns);
}
