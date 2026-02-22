//! CUDA backend engine (Phase 1 stub).
//!
//! This implements the backend contract while returning explicit typed errors
//! for unimplemented execution methods.

const std = @import("std");
const models = @import("../../../models/root.zig");
const contract = @import("../contract.zig");
const cpu_vision = @import("../cpu/vision/root.zig");
const compute = @import("../../../compute/root.zig");
const tensor = @import("../../../tensor.zig");
const dtype = @import("../../../dtype.zig");
const log = @import("../../../log.zig");

const LoadedModel = models.LoadedModel;
const Tensor = tensor.Tensor;
const prototype_eps: f32 = 1e-5;
const prototype_projected_vocab_cap: usize = 256;

const PrototypeRuntime = struct {
    projected_vocab: usize,
    using_model_norm: bool,
    using_model_projection: bool,
    using_model_embeddings: bool,
    hidden_host: []f32,
    projected_logits_host: []f32,
    input_dev: compute.cuda.Buffer,
    norm_weight_dev: compute.cuda.Buffer,
    norm_out_dev: compute.cuda.Buffer,
    projection_dev: compute.cuda.Buffer,
    logits_dev: compute.cuda.Buffer,

    fn init(
        allocator: std.mem.Allocator,
        device: *compute.cuda.Device,
        loaded: *const LoadedModel,
    ) !PrototypeRuntime {
        const d_model: usize = @intCast(loaded.config.d_model);
        const vocab_size: usize = @intCast(loaded.config.vocab_size);
        if (d_model == 0 or vocab_size == 0) return error.InvalidArgument;

        const projected_vocab = @min(vocab_size, prototype_projected_vocab_cap);
        const projection_elements = std.math.mul(usize, d_model, projected_vocab) catch return error.InvalidArgument;
        const d_model_bytes = std.math.mul(usize, d_model, @sizeOf(f32)) catch return error.InvalidArgument;
        const projection_bytes = std.math.mul(usize, projection_elements, @sizeOf(f32)) catch return error.InvalidArgument;
        const logits_bytes = std.math.mul(usize, projected_vocab, @sizeOf(f32)) catch return error.InvalidArgument;

        const hidden_host = try allocator.alloc(f32, d_model);
        errdefer allocator.free(hidden_host);
        const projected_logits_host = try allocator.alloc(f32, projected_vocab);
        errdefer allocator.free(projected_logits_host);

        const norm_weight_host = try allocator.alloc(f32, d_model);
        defer allocator.free(norm_weight_host);
        const using_model_norm = tryPopulateFinalNormWeight(loaded, norm_weight_host);
        if (!using_model_norm) fillPrototypeNormWeight(norm_weight_host);

        const projection_host = try allocator.alloc(f32, projection_elements);
        defer allocator.free(projection_host);
        const using_model_projection = tryPopulateProjectionFromLoadedModel(loaded, d_model, projected_vocab, projection_host);
        if (!using_model_projection) fillPrototypeProjection(projection_host, d_model, projected_vocab);

        var input_dev = try device.allocBuffer(d_model_bytes);
        errdefer input_dev.deinit(device);
        var norm_weight_dev = try device.allocBuffer(d_model_bytes);
        errdefer norm_weight_dev.deinit(device);
        var norm_out_dev = try device.allocBuffer(d_model_bytes);
        errdefer norm_out_dev.deinit(device);
        var projection_dev = try device.allocBuffer(projection_bytes);
        errdefer projection_dev.deinit(device);
        var logits_dev = try device.allocBuffer(logits_bytes);
        errdefer logits_dev.deinit(device);

        try norm_weight_dev.upload(device, std.mem.sliceAsBytes(norm_weight_host));
        try projection_dev.upload(device, std.mem.sliceAsBytes(projection_host));

        return .{
            .projected_vocab = projected_vocab,
            .using_model_norm = using_model_norm,
            .using_model_projection = using_model_projection,
            .using_model_embeddings = canUseModelEmbeddings(loaded),
            .hidden_host = hidden_host,
            .projected_logits_host = projected_logits_host,
            .input_dev = input_dev,
            .norm_weight_dev = norm_weight_dev,
            .norm_out_dev = norm_out_dev,
            .projection_dev = projection_dev,
            .logits_dev = logits_dev,
        };
    }

    fn deinit(self: *PrototypeRuntime, allocator: std.mem.Allocator, device: *compute.cuda.Device) void {
        self.logits_dev.deinit(device);
        self.projection_dev.deinit(device);
        self.norm_out_dev.deinit(device);
        self.norm_weight_dev.deinit(device);
        self.input_dev.deinit(device);
        allocator.free(self.projected_logits_host);
        allocator.free(self.hidden_host);
    }
};

pub const CudaBackend = struct {
    pub const capabilities: contract.Capabilities = .{
        .vision_prefill = true,
        .decode_batch = true,
        .decode_streaming = false,
        .embedding = false,
        .warmup = false,
    };

    pub const PrefillVisionInput = cpu_vision.PrefillVisionInput;

    allocator: std.mem.Allocator,
    loaded: *LoadedModel,
    device: compute.cuda.Device,
    kernel_registry: compute.cuda.Registry,
    vector_add_function: ?compute.cuda.Function = null,
    vector_add_source: ?compute.cuda.registry.KernelSource = null,
    rmsnorm_function: ?compute.cuda.Function = null,
    rmsnorm_source: ?compute.cuda.registry.KernelSource = null,
    kernel_arg_pack: compute.cuda.ArgPack,
    blas: compute.cuda.Blas,
    prototype: PrototypeRuntime,
    d_model: usize,
    vocab_size: usize,
    max_batch_size: usize = 1,
    slot_in_use: bool = false,
    slot_position: usize = 0,
    slot_logits: []f32,

    pub fn init(allocator: std.mem.Allocator, loaded: *LoadedModel) !CudaBackend {
        var device = try compute.cuda.Device.init();
        errdefer device.deinit();

        log.info("inference", "CUDA device ready", .{ .name = device.name() });
        var backend = CudaBackend{
            .allocator = allocator,
            .loaded = loaded,
            .device = device,
            .kernel_registry = undefined,
            .kernel_arg_pack = compute.cuda.ArgPack.init(allocator),
            .blas = undefined,
            .prototype = undefined,
            .d_model = @intCast(loaded.config.d_model),
            .vocab_size = @intCast(loaded.config.vocab_size),
            .slot_logits = undefined,
        };
        backend.kernel_registry = compute.cuda.Registry.init(allocator, &backend.device);
        errdefer backend.kernel_registry.deinit();
        backend.slot_logits = try allocator.alloc(f32, backend.vocab_size);
        errdefer allocator.free(backend.slot_logits);
        backend.blas = try compute.cuda.Blas.init(&backend.device);
        errdefer backend.blas.deinit(&backend.device);
        backend.prototype = try PrototypeRuntime.init(allocator, &backend.device, loaded);
        errdefer backend.prototype.deinit(allocator, &backend.device);
        try backend.initKernelFunctions();

        try runMatmulSmoke(&backend);
        try runKernelSmoke(&backend);
        log.info("inference", "CUDA prototype decode path ready", .{
            .d_model = backend.d_model,
            .projected_vocab = backend.prototype.projected_vocab,
            .rmsnorm_kernel = @as(u8, @intFromBool(backend.rmsnorm_function != null)),
            .model_norm = @as(u8, @intFromBool(backend.prototype.using_model_norm)),
            .model_projection = @as(u8, @intFromBool(backend.prototype.using_model_projection)),
            .model_embeddings = @as(u8, @intFromBool(backend.prototype.using_model_embeddings)),
        });
        return backend;
    }

    pub fn deinit(self: *CudaBackend) void {
        self.allocator.free(self.slot_logits);
        self.prototype.deinit(self.allocator, &self.device);
        self.blas.deinit(&self.device);
        self.kernel_arg_pack.deinit();
        self.kernel_registry.deinit();
        self.device.deinit();
        self.* = undefined;
    }

    pub fn maxBatchSize(self: *const CudaBackend) usize {
        return self.max_batch_size;
    }

    pub fn vocabSize(self: *const CudaBackend) usize {
        return self.vocab_size;
    }

    pub fn prefill(self: *CudaBackend, tokens: []const u32, logits_out: []f32) !void {
        if (tokens.len == 0) {
            log.warn("inference", "CUDA prefill invalid args", .{
                .reason = "empty_tokens",
            });
            return error.InvalidArgument;
        }
        if (logits_out.len != self.vocab_size) {
            log.warn("inference", "CUDA prefill invalid args", .{
                .reason = "logits_len_mismatch",
                .logits_len = logits_out.len,
                .vocab_size = self.vocab_size,
            });
            return error.InvalidArgument;
        }

        const last_token = tokens[tokens.len - 1];
        try self.computeGpuPrototypeLogits(last_token, logits_out);
        self.slot_position = tokens.len;
    }

    pub fn decode(self: *CudaBackend, token: u32, position: usize, logits_out: []f32) !void {
        if (logits_out.len != self.vocab_size) {
            log.warn("inference", "CUDA decode invalid args", .{
                .reason = "logits_len_mismatch",
                .logits_len = logits_out.len,
                .vocab_size = self.vocab_size,
            });
            return error.InvalidArgument;
        }
        try self.computeGpuPrototypeLogits(token, logits_out);
        self.slot_position = position + 1;
    }

    pub fn allocSlot(self: *CudaBackend) ?usize {
        if (self.slot_in_use) return null;
        self.slot_in_use = true;
        self.slot_position = 0;
        return 0;
    }

    pub fn freeSlot(self: *CudaBackend, slot_index: usize) void {
        if (slot_index != 0) return;
        self.slot_in_use = false;
        self.slot_position = 0;
    }

    pub fn resetSlot(self: *CudaBackend, slot_index: usize) void {
        if (slot_index != 0) return;
        self.slot_position = 0;
    }

    pub fn getPosition(self: *const CudaBackend, slot_index: usize) usize {
        if (slot_index != 0) return 0;
        return self.slot_position;
    }

    pub fn prefillSlot(
        self: *CudaBackend,
        slot_index: usize,
        tokens: []const u32,
        logits_out: []f32,
    ) !void {
        if (tokens.len == 0) {
            log.warn("inference", "CUDA prefillSlot invalid args", .{
                .reason = "empty_tokens",
                .slot_index = slot_index,
            });
            return error.InvalidArgument;
        }
        if (logits_out.len != self.vocab_size) {
            log.warn("inference", "CUDA prefillSlot invalid args", .{
                .reason = "logits_len_mismatch",
                .slot_index = slot_index,
                .logits_len = logits_out.len,
                .vocab_size = self.vocab_size,
            });
            return error.InvalidArgument;
        }
        if (!self.slot_in_use or slot_index != 0) {
            log.warn("inference", "CUDA prefillSlot invalid args", .{
                .reason = "slot_state",
                .slot_index = slot_index,
                .slot_in_use = @as(u8, @intFromBool(self.slot_in_use)),
            });
            return error.InvalidArgument;
        }

        const last_token = tokens[tokens.len - 1];
        try self.computeGpuPrototypeLogits(last_token, logits_out);
        self.slot_position = tokens.len;
    }

    pub fn prefillSlotWithVision(
        self: *CudaBackend,
        slot_index: usize,
        tokens: []const u32,
        vision_input: ?*const PrefillVisionInput,
        logits_out: []f32,
    ) !void {
        _ = vision_input;
        return self.prefillSlot(slot_index, tokens, logits_out);
    }

    pub fn decodeBatch(
        self: *CudaBackend,
        requests: []const contract.DecodeRequest,
        results: []contract.DecodeResult,
    ) !void {
        if (results.len < requests.len) {
            log.warn("inference", "CUDA decodeBatch invalid args", .{
                .reason = "results_short",
                .requests = requests.len,
                .results = results.len,
            });
            return error.InvalidArgument;
        }
        if (requests.len == 0) return;
        if (requests.len > 1) {
            log.warn("inference", "CUDA decodeBatch invalid args", .{
                .reason = "batch_gt_one",
                .requests = requests.len,
            });
            return error.InvalidArgument;
        }

        const req = requests[0];
        if (!self.slot_in_use or req.slot_index != 0) {
            log.warn("inference", "CUDA decodeBatch invalid args", .{
                .reason = "slot_state",
                .slot_index = req.slot_index,
                .slot_in_use = @as(u8, @intFromBool(self.slot_in_use)),
            });
            return error.InvalidArgument;
        }

        try self.computeGpuPrototypeLogits(req.token, self.slot_logits);
        results[0] = .{
            .slot_index = req.slot_index,
            .logits = self.slot_logits,
        };
        self.slot_position += 1;
    }

    fn computeGpuPrototypeLogits(self: *CudaBackend, token: u32, logits_out: []f32) !void {
        if (logits_out.len != self.vocab_size) return error.InvalidArgument;

        const used_model_embeddings = tryPopulateHiddenFromToken(self.loaded, token, self.prototype.hidden_host) catch |err| switch (err) {
            error.InvalidArgument => return error.InvalidArgument,
            else => return err,
        };
        if (!used_model_embeddings) fillPrototypeInput(self.prototype.hidden_host, token);
        try self.prototype.input_dev.upload(&self.device, std.mem.sliceAsBytes(self.prototype.hidden_host));

        if (self.rmsnorm_function) |rmsnorm_function| {
            const cols_u32 = std.math.cast(u32, self.d_model) orelse return error.InvalidArgument;
            try compute.cuda.rmsnorm.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                rmsnorm_function,
                &self.prototype.input_dev,
                &self.prototype.norm_weight_dev,
                &self.prototype.norm_out_dev,
                1,
                cols_u32,
                prototype_eps,
            );
        } else {
            try self.prototype.norm_out_dev.upload(&self.device, std.mem.sliceAsBytes(self.prototype.hidden_host));
        }

        try self.blas.matmulF32(
            &self.device,
            &self.prototype.norm_out_dev,
            1,
            self.d_model,
            &self.prototype.projection_dev,
            self.prototype.projected_vocab,
            &self.prototype.logits_dev,
        );
        try self.device.synchronize();
        try self.prototype.logits_dev.download(&self.device, std.mem.sliceAsBytes(self.prototype.projected_logits_host));

        @memset(logits_out, -1.0e9);
        @memcpy(logits_out[0..self.prototype.projected_vocab], self.prototype.projected_logits_host);
    }

    fn initKernelFunctions(self: *CudaBackend) !void {
        if (!self.device.supportsModuleLaunch()) return;

        try self.kernel_registry.loadEmbeddedModule(compute.cuda.vector_add.embedded_ptx);

        const vector_add = try self.kernel_registry.resolveFunction(
            "vector_add_f32",
            compute.cuda.vector_add.embedded_symbol,
        );
        self.vector_add_function = vector_add.function;
        self.vector_add_source = vector_add.source;

        const rmsnorm = try self.kernel_registry.resolveFunction(
            "rmsnorm_f32",
            compute.cuda.rmsnorm.embedded_symbol,
        );
        self.rmsnorm_function = rmsnorm.function;
        self.rmsnorm_source = rmsnorm.source;
    }
};

fn runMatmulSmoke(backend: *CudaBackend) !void {
    const device = &backend.device;
    const m: usize = 2;
    const k: usize = 2;
    const n: usize = 2;

    const a = [_]f32{
        1.0, 2.0,
        3.0, 4.0,
    };
    const b = [_]f32{
        5.0, 6.0,
        7.0, 8.0,
    };
    const expected = [_]f32{
        19.0, 22.0,
        43.0, 50.0,
    };
    var actual = [_]f32{0.0} ** (m * n);

    var a_dev = try device.allocBuffer(@sizeOf(f32) * a.len);
    defer a_dev.deinit(device);
    var b_dev = try device.allocBuffer(@sizeOf(f32) * b.len);
    defer b_dev.deinit(device);
    var c_dev = try device.allocBuffer(@sizeOf(f32) * actual.len);
    defer c_dev.deinit(device);

    try a_dev.upload(device, std.mem.sliceAsBytes(a[0..]));
    try b_dev.upload(device, std.mem.sliceAsBytes(b[0..]));
    try backend.blas.matmulF32(device, &a_dev, m, k, &b_dev, n, &c_dev);
    try device.synchronize();
    try c_dev.download(device, std.mem.sliceAsBytes(actual[0..]));

    for (expected, actual) |want, got| {
        if (@abs(want - got) > 0.001) return error.CudaSmokeMismatch;
    }

    log.info("inference", "CUDA matmul smoke passed", .{
        .m = m,
        .k = k,
        .n = n,
        .c00 = actual[0],
    });
}

fn runKernelSmoke(
    backend: *CudaBackend,
) !void {
    if (!backend.device.supportsModuleLaunch()) {
        log.info("inference", "CUDA module launch API unavailable; skipping kernel smoke", .{});
        return;
    }
    if (backend.vector_add_function == null or backend.rmsnorm_function == null) {
        return error.CudaKernelUnavailable;
    }

    try runVectorAddSmoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.vector_add_function.?,
        backend.vector_add_source orelse .embedded_ptx,
    );
    try runRmsNormSmoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.rmsnorm_function.?,
        backend.rmsnorm_source orelse .embedded_ptx,
    );
}

fn runVectorAddSmoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function: compute.cuda.Function,
    source: compute.cuda.registry.KernelSource,
) !void {
    const lhs = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const rhs = [_]f32{ 10.0, 20.0, 30.0, 40.0 };
    const expected = [_]f32{ 11.0, 22.0, 33.0, 44.0 };
    var actual = [_]f32{0.0} ** lhs.len;

    var lhs_dev = try device.allocBuffer(lhs.len * @sizeOf(f32));
    defer lhs_dev.deinit(device);
    var rhs_dev = try device.allocBuffer(rhs.len * @sizeOf(f32));
    defer rhs_dev.deinit(device);
    var out_dev = try device.allocBuffer(actual.len * @sizeOf(f32));
    defer out_dev.deinit(device);

    try lhs_dev.upload(device, std.mem.sliceAsBytes(lhs[0..]));
    try rhs_dev.upload(device, std.mem.sliceAsBytes(rhs[0..]));
    try compute.cuda.vector_add.runWithFunction(
        arg_pack,
        device,
        function,
        &lhs_dev,
        &rhs_dev,
        &out_dev,
        @intCast(lhs.len),
    );
    try device.synchronize();
    try out_dev.download(device, std.mem.sliceAsBytes(actual[0..]));

    for (expected, actual) |want, got| {
        if (@abs(want - got) > 0.0001) return error.CudaKernelSmokeMismatch;
    }

    log.info("inference", "CUDA vector_add smoke passed", .{
        .n = lhs.len,
        .source = @tagName(source),
        .out0 = actual[0],
    });
}

fn runRmsNormSmoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function: compute.cuda.Function,
    source: compute.cuda.registry.KernelSource,
) !void {
    const rows: u32 = 2;
    const cols: u32 = 4;
    const eps: f32 = 1e-5;

    const input = [_]f32{
        1.0,  2.0, 3.0, 4.0,
        -1.0, 0.0, 1.0, 2.0,
    };
    const weight = [_]f32{ 1.0, 1.5, 0.5, 2.0 };
    var expected = [_]f32{0.0} ** input.len;
    var actual = [_]f32{0.0} ** input.len;

    computeRmsNormReference(&expected, &input, &weight, rows, cols, eps);

    var input_dev = try device.allocBuffer(input.len * @sizeOf(f32));
    defer input_dev.deinit(device);
    var weight_dev = try device.allocBuffer(weight.len * @sizeOf(f32));
    defer weight_dev.deinit(device);
    var output_dev = try device.allocBuffer(actual.len * @sizeOf(f32));
    defer output_dev.deinit(device);

    try input_dev.upload(device, std.mem.sliceAsBytes(input[0..]));
    try weight_dev.upload(device, std.mem.sliceAsBytes(weight[0..]));
    try compute.cuda.rmsnorm.runWithFunction(
        arg_pack,
        device,
        function,
        &input_dev,
        &weight_dev,
        &output_dev,
        rows,
        cols,
        eps,
    );
    try device.synchronize();
    try output_dev.download(device, std.mem.sliceAsBytes(actual[0..]));

    for (expected, actual) |want, got| {
        if (@abs(want - got) > 0.001) return error.CudaKernelSmokeMismatch;
    }

    log.info("inference", "CUDA rmsnorm smoke passed", .{
        .rows = rows,
        .cols = cols,
        .source = @tagName(source),
        .out0 = actual[0],
    });
}

fn computeRmsNormReference(
    out: []f32,
    input: []const f32,
    weight: []const f32,
    rows: u32,
    cols: u32,
    eps: f32,
) void {
    const rows_usize: usize = @intCast(rows);
    const cols_usize: usize = @intCast(cols);
    var row: usize = 0;
    while (row < rows_usize) : (row += 1) {
        const base = row * cols_usize;
        var sum_sq: f32 = 0.0;
        var col: usize = 0;
        while (col < cols_usize) : (col += 1) {
            const v = input[base + col];
            sum_sq += v * v;
        }
        const mean_sq = sum_sq / @as(f32, @floatFromInt(cols_usize));
        const inv_rms = 1.0 / std.math.sqrt(mean_sq + eps);
        col = 0;
        while (col < cols_usize) : (col += 1) {
            out[base + col] = input[base + col] * inv_rms * weight[col];
        }
    }
}

fn fillPrototypeInput(out: []f32, token: u32) void {
    for (out, 0..) |*value, i| {
        const i_u32: u32 = @intCast(i);
        const hashed = token +% (i_u32 *% 2654435761);
        const centered: i32 = @intCast(hashed & 0x3ff);
        value.* = (@as(f32, @floatFromInt(centered)) - 512.0) / 512.0;
    }
}

fn canUseModelEmbeddings(loaded: *const LoadedModel) bool {
    const embeddings = loaded.token_embeddings;
    if (embeddings.data_ptr == null) return false;
    return embeddings.dtype == .f32 or
        embeddings.dtype == .f16 or
        embeddings.dtype == .bf16 or
        embeddings.dtype == .grouped_affine_u4 or
        embeddings.dtype == .grouped_affine_u8;
}

fn tryPopulateHiddenFromToken(
    loaded: *const LoadedModel,
    token: u32,
    out: []f32,
) !bool {
    const embeddings = &loaded.token_embeddings;
    if (embeddings.data_ptr == null) return false;
    if (embeddings.n_dims != 2) return false;
    if (embeddings.shape[0] <= 0 or embeddings.shape[1] <= 0) return false;

    const dim0: usize = @intCast(embeddings.shape[0]);
    const dim1: usize = @intCast(embeddings.shape[1]);
    const token_idx: usize = @intCast(token);
    const hidden_dim = out.len;

    switch (embeddings.dtype) {
        .f32 => {
            const src = embeddings.asSlice(f32);
            // Common layout: [vocab, d_model].
            if (dim1 == hidden_dim) {
                if (token_idx >= dim0) return error.InvalidArgument;
                const row_start = token_idx * dim1;
                @memcpy(out, src[row_start .. row_start + hidden_dim]);
                return true;
            }

            // Transposed layout: [d_model, vocab].
            if (dim0 == hidden_dim) {
                if (token_idx >= dim1) return error.InvalidArgument;
                var i: usize = 0;
                while (i < hidden_dim) : (i += 1) {
                    out[i] = src[i * dim1 + token_idx];
                }
                return true;
            }

            return false;
        },
        .f16, .bf16 => {
            const src_u16 = embeddings.asSliceUnaligned(u16);
            // Common layout: [vocab, d_model].
            if (dim1 == hidden_dim) {
                if (token_idx >= dim0) return error.InvalidArgument;
                var i: usize = 0;
                while (i < hidden_dim) : (i += 1) {
                    const raw = src_u16[token_idx * dim1 + i];
                    out[i] = if (embeddings.dtype == .f16) dtype.fp16ToF32(raw) else dtype.bf16ToF32(raw);
                }
                return true;
            }

            // Transposed layout: [d_model, vocab].
            if (dim0 == hidden_dim) {
                if (token_idx >= dim1) return error.InvalidArgument;
                var i: usize = 0;
                while (i < hidden_dim) : (i += 1) {
                    const raw = src_u16[i * dim1 + token_idx];
                    out[i] = if (embeddings.dtype == .f16) dtype.fp16ToF32(raw) else dtype.bf16ToF32(raw);
                }
                return true;
            }

            return false;
        },
        .grouped_affine_u4, .grouped_affine_u8 => {
            // Common layout: [vocab, d_model].
            if (dim1 == hidden_dim) {
                if (token_idx >= dim0) return error.InvalidArgument;
                var i: usize = 0;
                while (i < hidden_dim) : (i += 1) {
                    out[i] = try gaffineValueAt(embeddings, token_idx, i);
                }
                return true;
            }

            // Transposed layout: [d_model, vocab].
            if (dim0 == hidden_dim) {
                if (token_idx >= dim1) return error.InvalidArgument;
                var i: usize = 0;
                while (i < hidden_dim) : (i += 1) {
                    out[i] = try gaffineValueAt(embeddings, i, token_idx);
                }
                return true;
            }

            return false;
        },
        else => return false,
    }
}

fn fillPrototypeNormWeight(out: []f32) void {
    for (out, 0..) |*value, i| {
        const i_u32: u32 = @intCast(i);
        value.* = 1.0 + @as(f32, @floatFromInt(i_u32 % 7)) * 0.01;
    }
}

fn fillPrototypeProjection(out: []f32, hidden_dim: usize, projected_vocab: usize) void {
    var row: usize = 0;
    while (row < hidden_dim) : (row += 1) {
        const row_u32: u32 = @intCast(row + 1);
        var col: usize = 0;
        while (col < projected_vocab) : (col += 1) {
            const col_u32: u32 = @intCast(col + 1);
            const mixed = (row_u32 *% 1664525) +% (col_u32 *% 1013904223) +% 12345;
            const centered: i32 = @intCast(mixed & 0x1ff);
            out[row * projected_vocab + col] = (@as(f32, @floatFromInt(centered)) - 256.0) * 0.0005;
        }
    }
}

fn tryPopulateFinalNormWeight(loaded: *const LoadedModel, out: []f32) bool {
    if (loaded.ln_final) |ln_final| {
        if (ln_final.data_ptr == null or ln_final.numel < out.len) return false;
        switch (ln_final.dtype) {
            .f32 => {
                const src = ln_final.asSlice(f32);
                if (src.len < out.len) return false;
                @memcpy(out, src[0..out.len]);
                return true;
            },
            .f16, .bf16 => {
                const src = ln_final.asSliceUnaligned(u16);
                if (src.len < out.len) return false;
                for (out, 0..) |*v, i| {
                    const raw = src[i];
                    v.* = if (ln_final.dtype == .f16) dtype.fp16ToF32(raw) else dtype.bf16ToF32(raw);
                }
                return true;
            },
            else => return false,
        }
    }
    return false;
}

fn tryPopulateProjectionFromLoadedModel(
    loaded: *const LoadedModel,
    d_model: usize,
    projected_vocab: usize,
    out: []f32,
) bool {
    if (loaded.lm_head) |lm_head| {
        if (tryPopulateProjectionFromWeight(&lm_head, d_model, projected_vocab, out)) return true;
    }
    return tryPopulateProjectionFromWeight(&loaded.token_embeddings, d_model, projected_vocab, out);
}

fn tryPopulateProjectionFromWeight(
    weight: *const Tensor,
    d_model: usize,
    projected_vocab: usize,
    out: []f32,
) bool {
    if (weight.data_ptr == null) return false;
    if (weight.n_dims != 2) return false;

    if (weight.shape[0] <= 0 or weight.shape[1] <= 0) return false;
    const dim0: usize = @intCast(weight.shape[0]);
    const dim1: usize = @intCast(weight.shape[1]);
    switch (weight.dtype) {
        .f32 => {
            const expected_len = std.math.mul(usize, dim0, dim1) catch return false;
            const src = weight.asSlice(f32);
            if (src.len < expected_len) return false;

            // Direct layout: [d_model, vocab] so each row can be copied contiguously.
            if (dim0 == d_model and dim1 >= projected_vocab) {
                var row: usize = 0;
                while (row < d_model) : (row += 1) {
                    const src_start = row * dim1;
                    const dst_start = row * projected_vocab;
                    @memcpy(out[dst_start .. dst_start + projected_vocab], src[src_start .. src_start + projected_vocab]);
                }
                return true;
            }

            // Transposed layout: [vocab, d_model], so gather one column per token row.
            if (dim1 == d_model and dim0 >= projected_vocab) {
                var row: usize = 0;
                while (row < d_model) : (row += 1) {
                    var col: usize = 0;
                    while (col < projected_vocab) : (col += 1) {
                        out[row * projected_vocab + col] = src[col * dim1 + row];
                    }
                }
                return true;
            }

            return false;
        },
        .f16, .bf16 => {
            const expected_len = std.math.mul(usize, dim0, dim1) catch return false;
            const src_u16 = weight.asSliceUnaligned(u16);
            if (src_u16.len < expected_len) return false;

            // Direct layout: [d_model, vocab]
            if (dim0 == d_model and dim1 >= projected_vocab) {
                var row: usize = 0;
                while (row < d_model) : (row += 1) {
                    var col: usize = 0;
                    while (col < projected_vocab) : (col += 1) {
                        const raw = src_u16[row * dim1 + col];
                        out[row * projected_vocab + col] = if (weight.dtype == .f16) dtype.fp16ToF32(raw) else dtype.bf16ToF32(raw);
                    }
                }
                return true;
            }

            // Transposed layout: [vocab, d_model]
            if (dim1 == d_model and dim0 >= projected_vocab) {
                var row: usize = 0;
                while (row < d_model) : (row += 1) {
                    var col: usize = 0;
                    while (col < projected_vocab) : (col += 1) {
                        const raw = src_u16[col * dim1 + row];
                        out[row * projected_vocab + col] = if (weight.dtype == .f16) dtype.fp16ToF32(raw) else dtype.bf16ToF32(raw);
                    }
                }
                return true;
            }

            return false;
        },
        .grouped_affine_u4, .grouped_affine_u8 => {
            // Direct layout: [d_model, vocab]
            if (dim0 == d_model and dim1 >= projected_vocab) {
                var row: usize = 0;
                while (row < d_model) : (row += 1) {
                    var col: usize = 0;
                    while (col < projected_vocab) : (col += 1) {
                        out[row * projected_vocab + col] = gaffineValueAt(weight, row, col) catch return false;
                    }
                }
                return true;
            }

            // Transposed layout: [vocab, d_model]
            if (dim1 == d_model and dim0 >= projected_vocab) {
                var row: usize = 0;
                while (row < d_model) : (row += 1) {
                    var col: usize = 0;
                    while (col < projected_vocab) : (col += 1) {
                        out[row * projected_vocab + col] = gaffineValueAt(weight, col, row) catch return false;
                    }
                }
                return true;
            }

            return false;
        },
        else => return false,
    }
}

fn gaffineScaleBiasToF32(scales_dtype: tensor.DType, bytes: []const u8, idx: usize) !f32 {
    const byte_offset = std.math.mul(usize, idx, @sizeOf(u16)) catch return error.InvalidArgument;
    if (byte_offset + @sizeOf(u16) > bytes.len) return error.InvalidArgument;

    const v = std.mem.readInt(u16, bytes[byte_offset..][0..2], .little);
    return switch (scales_dtype) {
        .f16 => dtype.fp16ToF32(v),
        .bf16 => dtype.bf16ToF32(v),
        else => error.InvalidArgument,
    };
}

fn gaffineValueAt(weight: *const Tensor, row: usize, col: usize) !f32 {
    if (weight.dtype != .grouped_affine_u4 and weight.dtype != .grouped_affine_u8) return error.InvalidArgument;
    const gaffine = weight.gaffine orelse return error.InvalidArgument;
    if (weight.n_dims != 2) return error.InvalidArgument;
    if (weight.shape[0] <= 0 or weight.shape[1] <= 0) return error.InvalidArgument;

    const rows: usize = @intCast(weight.shape[0]);
    const cols: usize = @intCast(weight.shape[1]);
    if (row >= rows or col >= cols) return error.InvalidArgument;

    const values_per_word: usize = if (weight.dtype == .grouped_affine_u4) 8 else 4;
    const bits: u5 = if (weight.dtype == .grouped_affine_u4) 4 else 8;
    const mask: u32 = if (weight.dtype == .grouped_affine_u4) 0xF else 0xFF;
    if (values_per_word == 0 or cols % values_per_word != 0) return error.InvalidArgument;
    if (gaffine.group_size == 0 or cols % gaffine.group_size != 0) return error.InvalidArgument;

    const packed_stride = cols / values_per_word;
    const group_stride = cols / gaffine.group_size;
    if (group_stride == 0) return error.InvalidArgument;

    const pack_idx = row * packed_stride + (col / values_per_word);
    const pack_byte_offset = std.math.mul(usize, pack_idx, @sizeOf(u32)) catch return error.InvalidArgument;
    if (pack_byte_offset + @sizeOf(u32) > weight.data().len) return error.InvalidArgument;
    const packed_word = std.mem.readInt(u32, weight.data()[pack_byte_offset..][0..4], .little);
    const shift: u5 = @intCast((col % values_per_word) * bits);
    const quant = (packed_word >> shift) & mask;

    const group_idx = col / gaffine.group_size;
    const sb_idx = row * group_stride + group_idx;
    const scale = try gaffineScaleBiasToF32(gaffine.scales_dtype, gaffine.scales, sb_idx);
    const bias = try gaffineScaleBiasToF32(gaffine.scales_dtype, gaffine.biases, sb_idx);

    return @as(f32, @floatFromInt(quant)) * scale + bias;
}

test "tryPopulateProjectionFromWeight supports [d_model, vocab] layout" {
    const d_model: usize = 3;
    const projected_vocab: usize = 2;
    var weights = [_]f32{
        1.0, 2.0,  3.0,  4.0,
        5.0, 6.0,  7.0,  8.0,
        9.0, 10.0, 11.0, 12.0,
    };
    const weight_bytes = std.mem.sliceAsBytes(weights[0..]);
    const weight = Tensor.view(weight_bytes.ptr, &.{ d_model, 4 }, .f32, weight_bytes.len);
    var out = [_]f32{0.0} ** (d_model * projected_vocab);

    try std.testing.expect(tryPopulateProjectionFromWeight(&weight, d_model, projected_vocab, out[0..]));
    const expected = [_]f32{ 1.0, 2.0, 5.0, 6.0, 9.0, 10.0 };
    for (expected, out) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.0);
    }
}

test "tryPopulateProjectionFromWeight supports [vocab, d_model] layout" {
    const d_model: usize = 3;
    const projected_vocab: usize = 2;
    var weights = [_]f32{
        1.0,  2.0,  3.0,
        4.0,  5.0,  6.0,
        7.0,  8.0,  9.0,
        10.0, 11.0, 12.0,
    };
    const weight_bytes = std.mem.sliceAsBytes(weights[0..]);
    const weight = Tensor.view(weight_bytes.ptr, &.{ 4, d_model }, .f32, weight_bytes.len);
    var out = [_]f32{0.0} ** (d_model * projected_vocab);

    try std.testing.expect(tryPopulateProjectionFromWeight(&weight, d_model, projected_vocab, out[0..]));
    const expected = [_]f32{ 1.0, 4.0, 2.0, 5.0, 3.0, 6.0 };
    for (expected, out) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.0);
    }
}

test "tryPopulateHiddenFromToken supports [vocab, d_model] layout" {
    var embedding_data = [_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };
    const bytes = std.mem.sliceAsBytes(embedding_data[0..]);
    const embeddings = Tensor.view(bytes.ptr, &.{ 2, 3 }, .f32, bytes.len);

    var loaded = LoadedModel{
        .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
        .config = std.mem.zeroes(tensor.ModelConfig),
        .token_embeddings = embeddings,
        .blocks = &.{},
        .original_weight_dtype = .f32,
    };
    defer loaded.arena.deinit();

    var out = [_]f32{0.0} ** 3;
    try std.testing.expect(try tryPopulateHiddenFromToken(&loaded, 1, out[0..]));
    const expected = [_]f32{ 4.0, 5.0, 6.0 };
    for (expected, out) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.0);
    }
}

test "tryPopulateHiddenFromToken supports [d_model, vocab] layout" {
    var embedding_data = [_]f32{
        1.0, 4.0,
        2.0, 5.0,
        3.0, 6.0,
    };
    const bytes = std.mem.sliceAsBytes(embedding_data[0..]);
    const embeddings = Tensor.view(bytes.ptr, &.{ 3, 2 }, .f32, bytes.len);

    var loaded = LoadedModel{
        .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
        .config = std.mem.zeroes(tensor.ModelConfig),
        .token_embeddings = embeddings,
        .blocks = &.{},
        .original_weight_dtype = .f32,
    };
    defer loaded.arena.deinit();

    var out = [_]f32{0.0} ** 3;
    try std.testing.expect(try tryPopulateHiddenFromToken(&loaded, 1, out[0..]));
    const expected = [_]f32{ 4.0, 5.0, 6.0 };
    for (expected, out) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.0);
    }
}

test "gaffineValueAt decodes grouped_affine_u4 values" {
    var packed_words = [_]u32{
        // 8 packed 4-bit values: 0,1,2,3,4,5,6,7
        0x7654_3210,
    };
    const packed_bytes = std.mem.sliceAsBytes(packed_words[0..]);

    const one_bf16 = dtype.f32ToBf16(1.0);
    const zero_bf16 = dtype.f32ToBf16(0.0);
    var scales_u16 = [_]u16{one_bf16};
    var biases_u16 = [_]u16{zero_bf16};
    const scales_bytes = std.mem.sliceAsBytes(scales_u16[0..]);
    const biases_bytes = std.mem.sliceAsBytes(biases_u16[0..]);

    var weight = Tensor.view(packed_bytes.ptr, &.{ 1, 8 }, .grouped_affine_u4, packed_bytes.len);
    weight.gaffine = .{
        .scales = scales_bytes,
        .biases = biases_bytes,
        .group_size = 8,
        .scales_dtype = .bf16,
    };

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), try gaffineValueAt(&weight, 0, 0), 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), try gaffineValueAt(&weight, 0, 3), 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), try gaffineValueAt(&weight, 0, 7), 0.0);
}

test "tryPopulateFinalNormWeight supports bf16 weights" {
    var norm_u16 = [_]u16{
        dtype.f32ToBf16(1.25),
        dtype.f32ToBf16(-0.5),
    };
    const norm_bytes = std.mem.sliceAsBytes(norm_u16[0..]);
    const norm_tensor = Tensor.view(norm_bytes.ptr, &.{2}, .bf16, norm_bytes.len);

    var loaded = LoadedModel{
        .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
        .config = std.mem.zeroes(tensor.ModelConfig),
        .ln_final = norm_tensor,
        .token_embeddings = std.mem.zeroes(Tensor),
        .blocks = &.{},
        .original_weight_dtype = .bf16,
    };
    defer loaded.arena.deinit();

    var out = [_]f32{0.0} ** 2;
    try std.testing.expect(tryPopulateFinalNormWeight(&loaded, out[0..]));
    try std.testing.expectApproxEqAbs(@as(f32, 1.25), out[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), out[1], 0.01);
}

test "allocSlot allows only a single slot in stub backend" {
    var backend: CudaBackend = undefined;
    backend.slot_in_use = false;
    backend.slot_position = 999;
    try std.testing.expectEqual(@as(?usize, 0), backend.allocSlot());
    try std.testing.expectEqual(@as(usize, 0), backend.slot_position);
    try std.testing.expectEqual(@as(?usize, null), backend.allocSlot());
}

test "fillPrototypeInput is deterministic for token id" {
    var a: [8]f32 = undefined;
    var b: [8]f32 = undefined;
    fillPrototypeInput(a[0..], 42);
    fillPrototypeInput(b[0..], 42);

    for (a, b) |lhs, rhs| {
        try std.testing.expectApproxEqAbs(lhs, rhs, 0.0);
    }
}

test "fillPrototypeProjection writes non-zero coefficients" {
    var coeffs: [4 * 6]f32 = undefined;
    fillPrototypeProjection(coeffs[0..], 4, 6);

    var has_non_zero = false;
    for (coeffs) |value| {
        if (value != 0.0) {
            has_non_zero = true;
            break;
        }
    }
    try std.testing.expect(has_non_zero);
}
