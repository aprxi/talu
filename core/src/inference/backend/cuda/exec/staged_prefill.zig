//! Bridge-driven local staged execution for CUDA-backed local topologies.
//!
//! CPU->CUDA, CUDA->CUDA, and CPU->CUDA->CUDA are placement instances of this local stage-chain route.

const std = @import("std");
const compute = @import("compute_pkg");
const bridge = @import("../../../bridge/root.zig");
const transport = @import("../../../transport/root.zig");
const per_layer_branch_feature = @import("../per_layer_branch.zig");

const engine_weights = @import("../weights/root.zig");
const bufferSlice = engine_weights.bufferSlice;
const populatePrefillHiddenFromTokens = engine_weights.populatePrefillHiddenFromTokens;

const common = @import("common.zig");
const stage_adapters = @import("stage_adapters.zig");
const resolveStagedPrefillChunkRows = @import("prefill_route.zig").resolveStagedPrefillChunkRows;
const uploadCpuKvToMirrors = transport.uploadCpuKvToCudaMirrors;
const buildAttentionKernelSet = common.buildAttentionKernelSet;
const prefillMath = common.prefillMath;
const prefillChunkContext = common.prefillChunkContext;
const prepareCudaPrefillBackend = common.prepareCudaPrefillBackend;
const populateCudaPrefillInputRows = common.populateCudaPrefillInputRows;
const executeGpuPrefillLayers = common.executeGpuPrefillLayers;
const projectFinalLogitsFromCudaStage = common.projectFinalLogitsFromCudaStage;

fn resolveTwoCudaChunkCap(total_rows: usize, backend_a: anytype, backend_b: anytype) usize {
    return resolveStagedPrefillChunkRows(
        total_rows,
        @min(backend_a.prefill_chunk_rows_cap, backend_b.prefill_chunk_rows_cap),
        @import("env_pkg").getenv("TALU_CUDA_PREFILL_CHUNK_ROWS") != null,
    );
}

fn hasActivePerLayerBranch(backend: anytype) bool {
    const BackendType = @TypeOf(backend.*);
    if (!per_layer_branch_feature.hasPerLayerBranchRuntime(backend)) return false;
    if (comptime @hasField(BackendType, "per_layer_branch_runtime")) {
        return backend.per_layer_branch_runtime != null;
    }
    return false;
}

fn allocateSourceEmbeddings(
    allocator: std.mem.Allocator,
    needed: bool,
    total_rows: usize,
    d_model: usize,
) !?[]f32 {
    if (!needed) return null;
    return try allocator.alloc(f32, total_rows * d_model);
}

fn uploadChunkSourceEmbeddings(
    backend: anytype,
    source_embeddings_host: []const f32,
    pos_base: usize,
    rows: usize,
    d_model: usize,
) !compute.cuda.Buffer {
    const se_chunk_offset = pos_base * d_model;
    const se_chunk_f32s = rows * d_model;
    const se_bytes = std.math.mul(usize, se_chunk_f32s, @sizeOf(f32)) catch return error.InvalidArgument;
    var se_dst = try bufferSlice(&backend.runtime_buffers.deepstack_add_dev, 0, se_bytes);
    try se_dst.upload(&backend.device, std.mem.sliceAsBytes(source_embeddings_host[se_chunk_offset..][0..se_chunk_f32s]));
    return se_dst;
}

fn ensureCudaChunkBuffers(backend: anytype, rows: usize) !void {
    try backend.runtime_buffers.ensureRowCapacity(&backend.device, rows, backend.fixed_alloc_mode);
    try backend.ensureLayerProgramSlotRowCapacity(rows, backend.fixed_alloc_mode);
}

fn canExecuteConcreteCudaPrefill(comptime BackendType: type) bool {
    return @hasField(BackendType, "loaded") and
        @hasField(BackendType, "runtime_buffers") and
        @hasDecl(BackendType, "tryExecuteLayerProgram");
}

pub fn executeLocalPrefillPipeline(
    self: anytype,
    placement_kind: bridge.LocalPipelinePlacementKind,
    tokens: []const u32,
    slot_index: usize,
    logits_out: []f32,
) !void {
    const SelfType = @TypeOf(self.*);
    switch (placement_kind) {
        .cuda_cuda => {
            if (comptime @hasDecl(SelfType, "localCudaStage1")) {
                var stage1 = self.localCudaStage1() orelse return error.InvalidTopologyConfig;
                const Stage1Type = @TypeOf(stage1.*);
                if (comptime @hasField(Stage1Type, "state_descriptor_count") and @hasDecl(Stage1Type, "mirrorSlotStateBlocksFrom")) {
                    if (stage1.state_descriptor_count > 0) try stage1.mirrorSlotStateBlocksFrom(self, slot_index);
                }
                if (comptime @hasDecl(Stage1Type, "activateKvSlot")) {
                    stage1.activateKvSlot(slot_index);
                } else return error.InvalidTopologyConfig;
                if (comptime @hasDecl(SelfType, "activateKvSlot")) {
                    self.activateKvSlot(slot_index);
                } else return error.InvalidTopologyConfig;
                if (comptime canExecuteConcreteCudaPrefill(SelfType) and canExecuteConcreteCudaPrefill(Stage1Type)) {
                    return executeLocalPrefillCudaCuda(self, stage1, tokens, slot_index, logits_out);
                }
                return error.InvalidTopologyConfig;
            }
        },
        .cpu_cuda => {
            if (comptime @hasDecl(SelfType, "localCpuStage0")) {
                const cpu_stage0 = self.localCpuStage0() orelse return error.InvalidTopologyConfig;
                if (comptime @hasDecl(SelfType, "activateKvSlot")) {
                    self.activateKvSlot(slot_index);
                } else return error.InvalidTopologyConfig;
                if (comptime canExecuteConcreteCudaPrefill(SelfType)) {
                    return executeLocalPrefillCpuCuda(self, cpu_stage0, tokens, slot_index, logits_out);
                }
                return error.InvalidTopologyConfig;
            }
        },
        .cpu_cuda_cuda => {
            if (comptime @hasDecl(SelfType, "localCpuStage0") and @hasDecl(SelfType, "localCudaStage1")) {
                const cpu_stage0 = self.localCpuStage0() orelse return error.InvalidTopologyConfig;
                var gpu_stage1 = self.localCudaStage1() orelse return error.InvalidTopologyConfig;
                const Stage1Type = @TypeOf(gpu_stage1.*);
                if (comptime @hasDecl(SelfType, "activateKvSlot")) {
                    self.activateKvSlot(slot_index);
                } else return error.InvalidTopologyConfig;
                if (comptime @hasField(Stage1Type, "state_descriptor_count") and @hasDecl(Stage1Type, "mirrorSlotStateBlocksFrom")) {
                    if (gpu_stage1.state_descriptor_count > 0) try gpu_stage1.mirrorSlotStateBlocksFrom(self, slot_index);
                }
                if (comptime @hasDecl(Stage1Type, "activateKvSlot")) {
                    gpu_stage1.activateKvSlot(slot_index);
                } else return error.InvalidTopologyConfig;
                if (comptime canExecuteConcreteCudaPrefill(SelfType) and canExecuteConcreteCudaPrefill(Stage1Type)) {
                    return executeLocalPrefillCpuCudaCuda(self, cpu_stage0, gpu_stage1, tokens, slot_index, logits_out);
                }
                return error.InvalidTopologyConfig;
            }
        },
        .generic_local_chain => {},
    }
    return error.InvalidTopologyConfig;
}

fn executeLocalPrefillCpuCuda(
    self: anytype,
    cpu_stage0: anytype,
    tokens: []const u32,
    slot_index: usize,
    logits_out: []f32,
) !void {
    const SelfType = @TypeOf(self.*);
    if (comptime !@hasField(SelfType, "device")) return error.InvalidArgument;

    const previous_launch_phase = self.device.setLaunchPhase(.prefill);
    defer _ = self.device.setLaunchPhase(previous_launch_phase);

    const total_rows = tokens.len;
    const math = try prefillMath(self);

    try prepareCudaPrefillBackend(self, total_rows);
    const attention_kernels = buildAttentionKernelSet(self) catch return error.CudaKernelUnavailable;

    // ── CPU stage0: batched embed + forward through [0, split_layer) ──
    const prefill_buffer = try self.allocator.alloc(f32, total_rows * math.d_model);
    defer self.allocator.free(prefill_buffer);

    // For per-layer branch-input: capture source embeddings (raw scaled
    // embed_tokens output) before CPU layers modify the hidden states.
    const source_embeddings_host = try allocateSourceEmbeddings(
        self.allocator,
        hasActivePerLayerBranch(self),
        total_rows,
        math.d_model,
    );
    defer if (source_embeddings_host) |buf| self.allocator.free(buf);

    try cpu_stage0.prefillSlotLayerRange(slot_index, tokens, prefill_buffer, stage_adapters.localLayerOffset(self), source_embeddings_host);

    // Upload CPU source layer KV to GPU mirror buffers for cross-device sharing.
    try uploadCpuKvToMirrors(self, cpu_stage0, slot_index, 0, total_rows);

    // ── GPU stage1: chunked forward through GPU layers ──
    var pos_base: usize = 0;
    while (pos_base < total_rows) {
        const rows = @min(total_rows - pos_base, self.prefill_chunk_rows_cap);
        const chunk = try prefillChunkContext(pos_base, rows);
        const chunk_tokens = tokens[pos_base .. pos_base + rows];

        try ensureCudaChunkBuffers(self, rows);

        // Upload chunk from CPU host buffer to GPU input_dev.
        const chunk_offset = pos_base * math.d_model;
        const chunk_f32s = rows * math.d_model;
        const chunk_bytes = std.mem.sliceAsBytes(prefill_buffer[chunk_offset..][0..chunk_f32s]);
        try stage_adapters.executeHostToCudaPrefillBoundary(
            self,
            self,
            0,
            slot_index,
            pos_base,
            rows,
            chunk_bytes,
            chunk_bytes.len,
        );

        var per_layer_source_embeddings_opt: ?compute.cuda.Buffer = null;
        if (source_embeddings_host) |se_host| {
            per_layer_source_embeddings_opt = try uploadChunkSourceEmbeddings(self, se_host, pos_base, rows, math.d_model);
        } else if (per_layer_branch_feature.hasPerLayerBranchRuntime(self)) {
            per_layer_source_embeddings_opt = try per_layer_branch_feature.maybeCapturePerLayerSourceEmbeddings(self, rows);
        }
        const final_hidden_rows = try executeGpuPrefillLayers(
            self,
            slot_index,
            chunk_tokens,
            math,
            chunk,
            attention_kernels,
            per_layer_source_embeddings_opt,
            per_layer_branch_feature.hasPerLayerBranchRuntime(self),
            stage_adapters.localLayerOffset(self),
            self.block_runtime.blocks.len,
        );

        // Extract logits from the last row of the final chunk.
        if (pos_base + rows >= total_rows) {
            try projectFinalLogitsFromCudaStage(self, final_hidden_rows, rows, math.row_bytes, chunk.last_position, logits_out, null);
        }

        pos_base += rows;
    }
}

fn executeLocalPrefillCudaCuda(
    self: anytype,
    stage1: anytype,
    tokens: []const u32,
    slot_index: usize,
    logits_out: []f32,
) !void {
    const SelfType = @TypeOf(self.*);
    if (comptime !@hasField(SelfType, "device")) return error.InvalidArgument;
    if (comptime !@hasField(@TypeOf(stage1.*), "device")) return error.InvalidArgument;

    const previous_launch_phase0 = self.device.setLaunchPhase(.prefill);
    defer _ = self.device.setLaunchPhase(previous_launch_phase0);
    const previous_launch_phase1 = stage1.device.setLaunchPhase(.prefill);
    defer _ = stage1.device.setLaunchPhase(previous_launch_phase1);

    const total_rows = tokens.len;
    const math = try prefillMath(self);

    try prepareCudaPrefillBackend(self, total_rows);
    try prepareCudaPrefillBackend(stage1, total_rows);
    const attn_kernels_0 = buildAttentionKernelSet(self) catch return error.CudaKernelUnavailable;
    const attn_kernels_1 = buildAttentionKernelSet(stage1) catch return error.CudaKernelUnavailable;
    const chunk_cap = resolveTwoCudaChunkCap(total_rows, self, stage1);

    // ── per-layer branch input: compute source embeddings on host ──
    const has_per_layer_branch_0 = per_layer_branch_feature.hasPerLayerBranchRuntime(self);
    const has_per_layer_branch_1 = per_layer_branch_feature.hasPerLayerBranchRuntime(stage1);
    const source_embeddings_host = try allocateSourceEmbeddings(
        self.allocator,
        hasActivePerLayerBranch(self) or hasActivePerLayerBranch(stage1),
        total_rows,
        math.d_model,
    );
    defer if (source_embeddings_host) |buf| self.allocator.free(buf);

    if (source_embeddings_host) |se_host| {
        try populatePrefillHiddenFromTokens(self.loaded, tokens, math.d_model, se_host, null);
    }

    var pos_base: usize = 0;
    while (pos_base < total_rows) {
        const rows = @min(total_rows - pos_base, chunk_cap);
        const chunk = try prefillChunkContext(pos_base, rows);
        const chunk_tokens = tokens[pos_base .. pos_base + rows];

        // ── Stage 0: embedding + layers on GPU0 ──
        try ensureCudaChunkBuffers(self, rows);
        try populateCudaPrefillInputRows(self, chunk_tokens, rows, math.row_bytes);

        // Upload source embeddings for this chunk to stage0's deepstack_add_dev.
        var per_layer_source_embeddings_0: ?compute.cuda.Buffer = null;
        if (hasActivePerLayerBranch(self)) {
            if (source_embeddings_host) |se_host| {
                per_layer_source_embeddings_0 = try uploadChunkSourceEmbeddings(self, se_host, pos_base, rows, math.d_model);
            }
        }

        _ = try executeGpuPrefillLayers(
            self,
            slot_index,
            chunk_tokens,
            math,
            chunk,
            attn_kernels_0,
            per_layer_source_embeddings_0,
            has_per_layer_branch_0,
            null,
            self.block_runtime.blocks.len,
        );

        // ── Stage 1 buffer setup (must precede transfer into input_dev) ──
        try ensureCudaChunkBuffers(stage1, rows);

        // ── Bulk transfer stage0 → stage1 ──
        const transfer_bytes = std.math.mul(usize, rows, math.row_bytes) catch return error.InvalidArgument;
        try stage_adapters.executeCudaDevicePrefillBoundary(
            self,
            self,
            stage1,
            0,
            slot_index,
            pos_base,
            rows,
            transfer_bytes,
            .source_event_target_stream,
        );

        // Upload source embeddings for this chunk to stage1's deepstack_add_dev.
        var per_layer_source_embeddings_1: ?compute.cuda.Buffer = null;
        if (hasActivePerLayerBranch(stage1)) {
            if (source_embeddings_host) |se_host| {
                per_layer_source_embeddings_1 = try uploadChunkSourceEmbeddings(stage1, se_host, pos_base, rows, math.d_model);
            }
        }

        const stage1_final_hidden = try executeGpuPrefillLayers(
            stage1,
            slot_index,
            chunk_tokens,
            math,
            chunk,
            attn_kernels_1,
            per_layer_source_embeddings_1,
            has_per_layer_branch_1,
            null,
            stage1.block_runtime.blocks.len,
        );

        // ── Logits from last chunk (on stage1) ──
        if (pos_base + rows >= total_rows) {
            try projectFinalLogitsFromCudaStage(stage1, stage1_final_hidden, rows, math.row_bytes, chunk.last_position, logits_out, "cuda_pipeline2_final_norm_host");
        }

        pos_base += rows;
    }
}

fn executeLocalPrefillCpuCudaCuda(
    self: anytype,
    cpu_stage0: anytype,
    gpu_stage1: anytype,
    tokens: []const u32,
    slot_index: usize,
    logits_out: []f32,
) !void {
    const SelfType = @TypeOf(self.*);
    if (comptime !@hasField(SelfType, "device")) return error.InvalidArgument;
    if (comptime !@hasField(@TypeOf(gpu_stage1.*), "device")) return error.InvalidArgument;

    const previous_launch_phase1 = gpu_stage1.device.setLaunchPhase(.prefill);
    defer _ = gpu_stage1.device.setLaunchPhase(previous_launch_phase1);
    const previous_launch_phase2 = self.device.setLaunchPhase(.prefill);
    defer _ = self.device.setLaunchPhase(previous_launch_phase2);

    const total_rows = tokens.len;
    const math = try prefillMath(self);

    // ── GPU1 + GPU2 setup ──
    try prepareCudaPrefillBackend(gpu_stage1, total_rows);
    try prepareCudaPrefillBackend(self, total_rows);
    const attn_kernels_1 = buildAttentionKernelSet(gpu_stage1) catch return error.CudaKernelUnavailable;
    const attn_kernels_2 = buildAttentionKernelSet(self) catch return error.CudaKernelUnavailable;
    const chunk_cap = resolveTwoCudaChunkCap(total_rows, self, gpu_stage1);

    // ── CPU stage0: batched embed + forward through [0, split_layer) ──
    const prefill_buffer = try self.allocator.alloc(f32, total_rows * math.d_model);
    defer self.allocator.free(prefill_buffer);

    // For per-layer branch-input: capture source embeddings from CPU.
    const per_layer_branch_active_1 = per_layer_branch_feature.hasPerLayerBranchRuntime(gpu_stage1);
    const per_layer_branch_active_2 = per_layer_branch_feature.hasPerLayerBranchRuntime(self);
    const source_embeddings_host = try allocateSourceEmbeddings(
        self.allocator,
        hasActivePerLayerBranch(gpu_stage1) or hasActivePerLayerBranch(self),
        total_rows,
        math.d_model,
    );
    defer if (source_embeddings_host) |buf| self.allocator.free(buf);

    try cpu_stage0.prefillSlotLayerRange(slot_index, tokens, prefill_buffer, stage_adapters.localLayerOffset(self), source_embeddings_host);

    // ── GPU1 → GPU2: chunked forward ──
    var pos_base: usize = 0;
    while (pos_base < total_rows) {
        const rows = @min(total_rows - pos_base, chunk_cap);
        const chunk = try prefillChunkContext(pos_base, rows);
        const chunk_tokens = tokens[pos_base .. pos_base + rows];

        // GPU1: upload CPU output + layer loop.
        try ensureCudaChunkBuffers(gpu_stage1, rows);

        const chunk_offset = pos_base * math.d_model;
        const chunk_f32s = rows * math.d_model;
        const chunk_bytes = std.mem.sliceAsBytes(prefill_buffer[chunk_offset..][0..chunk_f32s]);
        try stage_adapters.executeHostToCudaPrefillBoundary(
            self,
            gpu_stage1,
            0,
            slot_index,
            pos_base,
            rows,
            chunk_bytes,
            chunk_bytes.len,
        );

        // Upload source embeddings for GPU1 per-layer branch branch.
        var per_layer_source_embeddings_1: ?compute.cuda.Buffer = null;
        if (hasActivePerLayerBranch(gpu_stage1)) {
            if (source_embeddings_host) |se_host| {
                per_layer_source_embeddings_1 = try uploadChunkSourceEmbeddings(gpu_stage1, se_host, pos_base, rows, math.d_model);
            }
        }

        _ = try executeGpuPrefillLayers(
            gpu_stage1,
            slot_index,
            chunk_tokens,
            math,
            chunk,
            attn_kernels_1,
            per_layer_source_embeddings_1,
            per_layer_branch_active_1,
            null,
            gpu_stage1.block_runtime.blocks.len,
        );

        // GPU2: transfer from GPU1 + layer loop.
        try ensureCudaChunkBuffers(self, rows);

        // Bulk stage1→stage2 transfer for the full chunk.
        const transfer_bytes = std.math.mul(usize, rows, math.row_bytes) catch return error.InvalidArgument;
        try stage_adapters.executeCudaDevicePrefillBoundary(
            self,
            gpu_stage1,
            self,
            1,
            slot_index,
            pos_base,
            rows,
            transfer_bytes,
            .source_stream,
        );

        // Upload source embeddings for GPU2 per-layer branch branch.
        var per_layer_source_embeddings_2: ?compute.cuda.Buffer = null;
        if (hasActivePerLayerBranch(self)) {
            if (source_embeddings_host) |se_host| {
                per_layer_source_embeddings_2 = try uploadChunkSourceEmbeddings(self, se_host, pos_base, rows, math.d_model);
            }
        }

        const final_hidden_rows = try executeGpuPrefillLayers(
            self,
            slot_index,
            chunk_tokens,
            math,
            chunk,
            attn_kernels_2,
            per_layer_source_embeddings_2,
            per_layer_branch_active_2,
            null,
            self.block_runtime.blocks.len,
        );

        // Logits from last chunk (on GPU2).
        if (pos_base + rows >= total_rows) {
            try projectFinalLogitsFromCudaStage(self, final_hidden_rows, rows, math.row_bytes, chunk.last_position, logits_out, null);
        }

        pos_base += rows;
    }
}

test "executeLocalPrefillPipeline rejects missing placement adapters" {
    const MockBackend = struct {};
    var backend = MockBackend{};
    const tokens = [_]u32{1};
    var logits: [1]f32 = undefined;

    try std.testing.expectError(
        error.InvalidTopologyConfig,
        executeLocalPrefillPipeline(&backend, .cuda_cuda, tokens[0..], 0, logits[0..]),
    );
}
