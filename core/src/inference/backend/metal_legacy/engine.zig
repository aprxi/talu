//! Metal backend for transformer inference (macOS GPU via MLX).
//!
//! Provides GPU-accelerated inference using Apple's MLX framework.
//! Supports lazy graph execution for optimal GPU utilization.

const std = @import("std");
const models = @import("../../../models/root.zig");
const rope_scaling = @import("../../../models/rope_scaling.zig");
const contract = @import("../contract.zig");
const tensor = @import("../../../tensor.zig");
const common_mrope = @import("vision/mrope.zig");
const cpu_math_rope = @import("../../../compute/cpu/math_rope.zig");
const cpu_linalg = @import("../../../compute/cpu/linalg.zig");
const cpu_rowwise = @import("../../../compute/cpu/rowwise.zig");
const ModelConfig = tensor.ModelConfig;
const Tensor = tensor.Tensor;
const log = @import("../../../log.zig");
const trace = @import("../../../xray/trace.zig");
const xray = @import("../../../xray/root.zig");

// Import compute primitives from compute/
const compute = @import("../../../compute/root.zig");
const metal_compute = compute.metal;
const graph = metal_compute.graph;
const metal_sampling = @import("sampling.zig");
const runtime_graph_mod = @import("runtime_graph.zig");
const runtime_contract = @import("../../runtime_contract/root.zig");
const LoadedModel = models.LoadedModel;

// Internal orchestration modules
const metal_executor = @import("executor/root.zig");
const weights_trait = metal_executor.weights;
const runtime_trait = metal_executor.runtime;
const vision_runtime_mod = @import("vision/root.zig");

// Re-exports for direct access if needed
pub const device = metal_compute.device;
pub const matmul = metal_compute.matmul;
pub const Graph = graph;
pub const Forward = runtime_trait;

pub const Device = metal_compute.Device;
pub const Buffer = metal_compute.Buffer;
pub const isAvailable = metal_compute.isAvailable;
pub const Cache = runtime_graph_mod.Cache;
pub const GatedDeltaCache = runtime_graph_mod.GatedDeltaCache;
pub const WeightHandles = weights_trait.WeightHandles;
/// Metal backend for GPU-accelerated transformer inference
pub const MetalBackend = struct {
    const MaxStateDescriptors: usize = runtime_contract.max_state_descriptors;
    pub const capabilities: contract.Capabilities = .{
        .vision_prefill = true,
        .decode_batch = false,
        .decode_streaming = true,
        .embedding = true,
        .warmup = true,
    };

    pub const PrefillVisionInput = vision_runtime_mod.PrefillVisionInput;

    allocator: std.mem.Allocator,
    loaded: *LoadedModel,
    config: ModelConfig,
    weights: *weights_trait.WeightHandles,
    cpu_lm_head_scratch: cpu_linalg.MatmulScratch,
    cpu_lm_head_fallback: bool,
    layer_count: usize,
    cache_max_seq_len: usize,
    vocab_size: usize,
    d_model: usize,
    max_batch_size: usize,
    slot_in_use: bool,
    /// Slots 1..N-1 for scheduler-managed multi-slot decode.
    extra_slots: []SlotState,
    slot_logits_buffer: []f32,
    vision_runtime: ?vision_runtime_mod.VisionRuntime = null,
    slot_rope_position_delta: isize,
    state_descriptors_storage: [MaxStateDescriptors]runtime_contract.StateDescriptor,
    state_runtime_roles: [MaxStateDescriptors]StateRuntimeRole,
    state_descriptor_count: u8,
    slot0_state_binding: SlotStateBinding = .{},
    text_runtime_rope: ?cpu_math_rope.RoPE = null,

    // Track position for decode
    current_position: usize,

    const SlotState = struct {
        state_binding: SlotStateBinding = .{},
        in_use: bool = false,
        position: usize = 0,
        rope_position_delta: isize = 0,
    };

    const SlotStateBinding = struct {
        handles: [MaxStateDescriptors]runtime_contract.StateBlockHandle = undefined,
        initialized: [MaxStateDescriptors]bool = [_]bool{false} ** MaxStateDescriptors,
        count: u8 = 0,
        bound: bool = false,

        fn detach(self: *SlotStateBinding) void {
            self.bound = false;
        }

        fn clear(self: *SlotStateBinding) void {
            self.initialized = [_]bool{false} ** MaxStateDescriptors;
            self.count = 0;
            self.bound = false;
        }
    };

    fn argmaxHost(values: []const f32) u32 {
        var best_idx: usize = 0;
        var best_val: f32 = -std.math.inf(f32);
        for (values, 0..) |v, idx| {
            if (v > best_val) {
                best_val = v;
                best_idx = idx;
            }
        }
        return @intCast(best_idx);
    }

    fn argminHost(values: []const f32) u32 {
        var best_idx: usize = 0;
        var best_val: f32 = std.math.inf(f32);
        for (values, 0..) |v, idx| {
            if (v < best_val) {
                best_val = v;
                best_idx = idx;
            }
        }
        return @intCast(best_idx);
    }

    const LogitsFingerprint = struct {
        min: f32,
        max: f32,
        finite_count: usize,
        non_finite_count: usize,
        checksum: u64,
    };

    fn fingerprintLogits(logits: []const f32) LogitsFingerprint {
        var min_value = std.math.inf(f32);
        var max_value = -std.math.inf(f32);
        var finite_count: usize = 0;
        var non_finite_count: usize = 0;
        var checksum: u64 = 0xcbf29ce484222325;
        for (logits, 0..) |value, idx| {
            if (!std.math.isFinite(value)) {
                non_finite_count += 1;
                continue;
            }
            finite_count += 1;
            min_value = @min(min_value, value);
            max_value = @max(max_value, value);
            const bits: u32 = @bitCast(value);
            checksum ^= (@as(u64, @intCast(idx)) << 32) ^ bits;
            checksum *%= 0x100000001b3;
        }
        if (finite_count == 0) {
            min_value = std.math.nan(f32);
            max_value = std.math.nan(f32);
        }
        return .{
            .min = min_value,
            .max = max_value,
            .finite_count = finite_count,
            .non_finite_count = non_finite_count,
            .checksum = checksum,
        };
    }

    fn selectNextTokenFromLogits(self: *const MetalBackend, logits: []const f32) !u32 {
        if (logits.len != self.vocab_size) return error.InvalidArgument;
        return if (self.config.logits_scaling < 0.0) argminHost(logits) else argmaxHost(logits);
    }

    fn emitParityFinalPathCheckpoints() bool {
        // XRAY ACCEPTABLE USE:
        // Verify is observability-only. It MUST NOT toggle prefill/decode route
        // selection, fusion eligibility, or kernel choice. Keep this disabled
        // unless we can emit final-path probes from the already-selected
        // production route. Final norm / lm_head tracing here only forces
        // materialization and host copies of tensors that the selected route
        // already computed; it does not alter route or kernel selection.
        return trace.shouldEmit(.final_norm) or
            trace.shouldEmit(.lm_head) or
            trace.shouldEmit(.logits_scaled);
    }

    fn useReferenceFinalProjection() bool {
        const S = struct {
            var enabled: ?bool = null;
        };
        if (S.enabled) |cached| return cached;
        const raw = std.posix.getenv("TALU_METAL_ATTN_REFERENCE") orelse {
            S.enabled = false;
            return false;
        };
        const value = std.mem.sliceTo(raw, 0);
        const result = value.len != 0 and
            !std.ascii.eqlIgnoreCase(value, "0") and
            !std.ascii.eqlIgnoreCase(value, "false") and
            !std.ascii.eqlIgnoreCase(value, "off") and
            !std.ascii.eqlIgnoreCase(value, "no");
        S.enabled = result;
        return result;
    }

    fn applyLogitsScalingHandle(self: *MetalBackend, logits_handle: graph.ArrayHandle) graph.ArrayHandle {
        if (self.weights.logits_scaling == 1.0) return logits_handle;
        return graph.mlx_lazy_multiply_scalar(logits_handle, 1.0 / self.weights.logits_scaling);
    }

    fn emitLmHeadFromReadyLogits(self: *MetalBackend, logits_out: []const f32) !void {
        if (!trace.shouldEmit(.lm_head)) return;
        if (logits_out.len < self.vocab_size) return error.InvalidArgument;
        trace.emitFinal(
            .lm_head,
            0,
            0,
            @ptrCast(logits_out.ptr),
            .f32,
            .{ @intCast(self.vocab_size), 0, 0, 0 },
            1,
            "metal_lm_head_host",
        );
    }

    fn projectLastNormedHiddenToLogits(
        self: *MetalBackend,
        last_normed_hidden_handle: graph.ArrayHandle,
        logits_out: []f32,
        emit_final_path_points: bool,
    ) !void {
        if (logits_out.len < self.vocab_size) return error.InvalidArgument;

        if (self.cpu_lm_head_fallback or useReferenceFinalProjection()) {
            const final_norm_host = try self.allocator.alloc(f32, self.d_model);
            defer self.allocator.free(final_norm_host);
            graph.eval(&[_]graph.ArrayHandle{last_normed_hidden_handle});
            graph.copyToHost(last_normed_hidden_handle, final_norm_host);

            const lm_head_tensor = self.loaded.lm_head orelse return error.MissingLmHead;
            if (lm_head_tensor.dtype == .bf16) {
                cpu_linalg.matmulLmHeadRowsBf16(
                    final_norm_host,
                    1,
                    &lm_head_tensor,
                    logits_out[0..self.vocab_size],
                    self.loaded.config.logits_scaling,
                    &self.cpu_lm_head_scratch,
                );
            } else {
                var hidden_view = Tensor.view2DSlice(final_norm_host, 1, self.d_model);
                var logits_view = Tensor.view2DSlice(logits_out[0..self.vocab_size], 1, self.vocab_size);
                try cpu_linalg.matmulAuto(&hidden_view, &lm_head_tensor, &logits_view, &self.cpu_lm_head_scratch);
                cpu_rowwise.scaleInPlaceReciprocal(logits_out[0..self.vocab_size], self.loaded.config.logits_scaling);
            }

            if (!emit_final_path_points) return;

            trace.emitFinal(
                .final_norm,
                0,
                1,
                @ptrCast(final_norm_host.ptr),
                .f32,
                .{ @intCast(self.d_model), 0, 0, 0 },
                1,
                "metal_final_norm_host",
            );
            trace.emitFinal(
                .lm_head,
                0,
                0,
                @ptrCast(logits_out.ptr),
                .f32,
                .{ @intCast(self.vocab_size), 0, 0, 0 },
                1,
                "metal_cpu_lm_head_host",
            );
            if (self.weights.logits_scaling != 1.0) {
                trace.emitFinal(
                    .logits_scaled,
                    0,
                    0,
                    @ptrCast(logits_out.ptr),
                    .f32,
                    .{ @intCast(self.vocab_size), 0, 0, 0 },
                    1,
                    "metal_cpu_lm_head_host",
                );
            }
            return;
        }

        const lm_head_handle = if (self.weights.lm_head_quantized) |quantized_lm_head| blk: {
            break :blk graph.mlx_lazy_quantized_matmul(
                last_normed_hidden_handle,
                quantized_lm_head.weights,
                quantized_lm_head.scales,
                quantized_lm_head.biases,
                quantized_lm_head.group_size,
                quantized_lm_head.bits,
                true,
            );
        } else blk: {
            break :blk graph.mlx_lazy_matmul(last_normed_hidden_handle, self.weights.lm_head.?);
        };
        defer graph.freeArray(lm_head_handle);

        const logits_handle = applyLogitsScalingHandle(self, lm_head_handle);
        defer if (logits_handle != lm_head_handle) graph.freeArray(logits_handle);

        if (emit_final_path_points) {
            graph.eval(&[_]graph.ArrayHandle{ last_normed_hidden_handle, lm_head_handle, logits_handle });
        } else {
            graph.eval(&[_]graph.ArrayHandle{logits_handle});
        }
        graph.copyToHost(logits_handle, logits_out[0..self.vocab_size]);

        if (!emit_final_path_points) return;

        const final_norm_host = try self.allocator.alloc(f32, self.d_model);
        defer self.allocator.free(final_norm_host);
        graph.copyToHost(last_normed_hidden_handle, final_norm_host);
        trace.emitFinal(
            .final_norm,
            0,
            1,
            @ptrCast(final_norm_host.ptr),
            .f32,
            .{ @intCast(self.d_model), 0, 0, 0 },
            1,
            "metal_final_norm_host",
        );

        if (self.weights.logits_scaling != 1.0) {
            // Keep lm_head trace semantics aligned with CPU: emit the same
            // scaled logits buffer used for token selection.
            trace.emitFinal(
                .lm_head,
                0,
                0,
                @ptrCast(logits_out.ptr),
                .f32,
                .{ @intCast(self.vocab_size), 0, 0, 0 },
                1,
                "metal_lm_head_host",
            );
            trace.emitFinal(
                .logits_scaled,
                0,
                0,
                @ptrCast(logits_out.ptr),
                .f32,
                .{ @intCast(self.vocab_size), 0, 0, 0 },
                1,
                "metal_logits_scaled_host",
            );
            return;
        }

        trace.emitFinal(
            .lm_head,
            0,
            0,
            @ptrCast(logits_out.ptr),
            .f32,
            .{ @intCast(self.vocab_size), 0, 0, 0 },
            1,
            "metal_lm_head_host",
        );
    }

    fn resolveMaxBatchSize() usize {
        if (std.posix.getenv("TALU_METAL_MAX_BATCH_SIZE")) |raw| {
            const parsed = std.fmt.parseUnsigned(usize, std.mem.sliceTo(raw, 0), 10) catch return 8;
            return @max(@as(usize, 1), parsed);
        }
        return 8;
    }

    fn resolveCpuLmHeadFallback() bool {
        const raw = std.posix.getenv("TALU_METAL_CPU_LM_HEAD") orelse return false;
        const value = std.mem.sliceTo(raw, 0);
        if (value.len == 0) return false;
        if (std.ascii.eqlIgnoreCase(value, "0")) return false;
        if (std.ascii.eqlIgnoreCase(value, "false")) return false;
        if (std.ascii.eqlIgnoreCase(value, "off")) return false;
        if (std.ascii.eqlIgnoreCase(value, "no")) return false;
        return true;
    }

    fn synchronizeDefaultDevice() void {
        if (!metal_compute.isAvailable()) return;
        graph.mlx_synchronize_default_stream();
    }

    fn resolveCacheMaxSeqLen(config_max_seq_len: usize) usize {
        // Fixed-capacity KV allocation at very large context lengths can stall
        // decode startup. Switch those models to dynamic cache growth.
        const fixed_capacity_limit: usize = 65_536;
        if (config_max_seq_len > fixed_capacity_limit) return 0;
        return config_max_seq_len;
    }

    fn buildTextRuntimeRoPE(_: std.mem.Allocator, loaded: *LoadedModel) !?cpu_math_rope.RoPE {
        const rope_allocator = std.heap.c_allocator;
        if (loaded.position_embeddings != null) return null;
        const rope_dim: usize = if (loaded.config.rope_dim > 0)
            @intCast(loaded.config.rope_dim)
        else
            @intCast(loaded.config.head_dim);
        if (rope_dim == 0) return null;

        // Standard RoPE (no scaling) is handled by MLX fast::rope which is a
        // single fused Metal kernel. Only build runtime tables for non-standard
        // scaling (llama3, yarn, linear) where fast::rope cannot reproduce the
        // modified inverse frequencies.
        const rs = loaded.config.rope_scaling;
        if (rs.rope_type == .none) return null;

        var freqs = try rope_scaling.materializeInverseFrequencies(
            rope_allocator,
            rope_dim,
            loaded.config.rope_theta,
            loaded.config.rope_scaling,
        );
        defer freqs.deinit(rope_allocator);

        return try cpu_math_rope.RoPE.initFromInvFreq(
            rope_allocator,
            rope_dim,
            @intCast(loaded.config.max_seq_len),
            freqs.inv_freq,
            freqs.attention_scaling,
        );
    }

    fn textRuntimeRoPEOverride(self: *MetalBackend, pos_offset: usize, sequence_len: usize) ?runtime_trait.RuntimeRoPEOverride {
        if (sequence_len == 0) return null;
        if (self.text_runtime_rope) |*rope| {
            const max_pos = pos_offset + (sequence_len - 1);
            _ = rope.getCos(max_pos);
            _ = rope.getSin(max_pos);
            const rows = max_pos + 1;
            const table_len = rows * rope.dim;
            return .{
                .cos = rope.freqs_cos[0..table_len],
                .sin = rope.freqs_sin[0..table_len],
                .dim = rope.dim,
            };
        }
        return null;
    }

    fn primeBoundSlotExecutionGraph(self: *MetalBackend, slot_index: usize) !void {
        if (self.state_descriptor_count == 0) return;
        const prime_shapes = [_]usize{ 1, 21 };
        for (prime_shapes) |prime_seq_len| {
            const prime_tokens = try self.allocator.alloc(u32, prime_seq_len);
            defer self.allocator.free(prime_tokens);
            @memset(prime_tokens, 0);
            const prime_rope = self.textRuntimeRoPEOverride(0, prime_seq_len);
            if (prime_seq_len == 1) {
                // Decode warmup: warm the single-token logits path used by decodeSlot.
                const prime_logits = try runtime_trait.transformerForwardLazyWithEmbeddingOverride(
                    self.allocator,
                    self.weights,
                    prime_tokens,
                    try self.slotStateBlocks(slot_index),
                    self.config,
                    0,
                    null,
                    null,
                    prime_rope,
                );
                defer graph.freeArray(prime_logits);
                graph.eval(&[_]graph.ArrayHandle{prime_logits});
            } else {
                // Prefill warmup: warm the hidden prefill graph used by prefill.
                const prime_hidden = try runtime_trait.transformerForwardHiddenLazyWithEmbeddingOverride(
                    self.allocator,
                    self.weights,
                    prime_tokens,
                    try self.slotStateBlocks(slot_index),
                    self.config,
                    0,
                    null,
                    null,
                    prime_rope,
                );
                defer graph.freeArray(prime_hidden);
                graph.eval(&[_]graph.ArrayHandle{prime_hidden});
            }
        }
    }

    fn cacheMaxSeqLen(self: *const MetalBackend) usize {
        return resolveCacheMaxSeqLen(@intCast(self.config.max_seq_len));
    }

    fn extraSlotIndexFor(slot_index: usize, max_batch_size: usize) !usize {
        if (slot_index == 0) return error.InvalidArgument;
        if (slot_index >= max_batch_size) return error.InvalidArgument;
        return slot_index - 1;
    }

    fn toExtraSlotIndex(self: *const MetalBackend, slot_index: usize) !usize {
        return extraSlotIndexFor(slot_index, self.max_batch_size);
    }

    fn slotPositionPtr(self: *MetalBackend, slot_index: usize) !*usize {
        if (slot_index == 0) return &self.current_position;
        const extra_idx = try self.toExtraSlotIndex(slot_index);
        return &self.extra_slots[extra_idx].position;
    }

    fn slotRopeDeltaPtr(self: *MetalBackend, slot_index: usize) !*isize {
        if (slot_index == 0) return &self.slot_rope_position_delta;
        const extra_idx = try self.toExtraSlotIndex(slot_index);
        return &self.extra_slots[extra_idx].rope_position_delta;
    }

    fn stateObjectPtr(comptime T: type, state_block: *const runtime_contract.StateBlockHandle) !*T {
        return runtime_contract.stateValueFromBlock(*T, state_block) orelse error.InvalidStateDescriptorBinding;
    }

    const StateRuntimeRole = enum(u8) {
        none = 0,
        kv_cache = 1,
        shortconv_cache = 2,
        mamba_cache = 3,
        gated_delta_cache = 4,
    };

    fn runtimeRoleForRuntimeKind(runtime_kind: u8) !StateRuntimeRole {
        return switch (runtime_kind) {
            runtime_contract.state_runtime_kind_none => .none,
            runtime_contract.state_runtime_kind_kv_cache => .kv_cache,
            runtime_contract.state_runtime_kind_shortconv_cache => .shortconv_cache,
            runtime_contract.state_runtime_kind_mamba_cache => .mamba_cache,
            runtime_contract.state_runtime_kind_gated_delta_cache => .gated_delta_cache,
            else => error.InvalidStateDescriptorBinding,
        };
    }

    fn deriveStateRuntimeRoles(
        descriptors: []const runtime_contract.StateDescriptor,
    ) ![MaxStateDescriptors]StateRuntimeRole {
        var roles = [_]StateRuntimeRole{.none} ** MaxStateDescriptors;
        for (descriptors, 0..) |descriptor, idx| {
            roles[idx] = try runtimeRoleForRuntimeKind(descriptor.runtime_kind);
        }
        return roles;
    }

    const StateRuntimeOps = struct {
        init: *const fn (self: *MetalBackend, state_block: *const runtime_contract.StateBlockHandle) anyerror!bool,
        reset: *const fn (state_block: *const runtime_contract.StateBlockHandle) anyerror!void,
        deinit: *const fn (state_block: *const runtime_contract.StateBlockHandle) anyerror!void,
    };

    fn initNoopState(_: *MetalBackend, _: *const runtime_contract.StateBlockHandle) !bool {
        return false;
    }

    fn resetNoopState(_: *const runtime_contract.StateBlockHandle) !void {}

    fn deinitNoopState(_: *const runtime_contract.StateBlockHandle) !void {}

    fn initKvCacheState(self: *MetalBackend, state_block: *const runtime_contract.StateBlockHandle) !bool {
        const cache = try stateObjectPtr(runtime_graph_mod.Cache, state_block);
        cache.* = runtime_graph_mod.Cache.init(self.layer_count, true, self.cache_max_seq_len);
        if (cache.handle == null) return error.OutOfMemory;
        return true;
    }

    fn resetKvCacheState(state_block: *const runtime_contract.StateBlockHandle) !void {
        (try stateObjectPtr(runtime_graph_mod.Cache, state_block)).reset();
    }

    fn deinitKvCacheState(state_block: *const runtime_contract.StateBlockHandle) !void {
        const cache = try stateObjectPtr(runtime_graph_mod.Cache, state_block);
        cache.deinit();
        cache.* = runtime_graph_mod.Cache.disabled(true);
    }

    fn initShortConvState(self: *MetalBackend, state_block: *const runtime_contract.StateBlockHandle) !bool {
        const shortconv = try stateObjectPtr(runtime_graph_mod.ShortConvCache, state_block);
        shortconv.* = runtime_graph_mod.ShortConvCache.init(self.layer_count);
        if (shortconv.handle == null) return error.OutOfMemory;
        return true;
    }

    fn resetShortConvState(state_block: *const runtime_contract.StateBlockHandle) !void {
        (try stateObjectPtr(runtime_graph_mod.ShortConvCache, state_block)).reset();
    }

    fn deinitShortConvState(state_block: *const runtime_contract.StateBlockHandle) !void {
        const shortconv = try stateObjectPtr(runtime_graph_mod.ShortConvCache, state_block);
        shortconv.deinit();
        shortconv.* = runtime_graph_mod.ShortConvCache.disabled();
    }

    fn initMambaState(self: *MetalBackend, state_block: *const runtime_contract.StateBlockHandle) !bool {
        const mamba = try stateObjectPtr(runtime_graph_mod.MambaCache, state_block);
        mamba.* = runtime_graph_mod.MambaCache.init(self.layer_count);
        if (mamba.handle == null) return error.OutOfMemory;
        return true;
    }

    fn resetMambaState(state_block: *const runtime_contract.StateBlockHandle) !void {
        (try stateObjectPtr(runtime_graph_mod.MambaCache, state_block)).reset();
    }

    fn deinitMambaState(state_block: *const runtime_contract.StateBlockHandle) !void {
        const mamba = try stateObjectPtr(runtime_graph_mod.MambaCache, state_block);
        mamba.deinit();
        mamba.* = runtime_graph_mod.MambaCache.disabled();
    }

    fn initGatedDeltaState(self: *MetalBackend, state_block: *const runtime_contract.StateBlockHandle) !bool {
        const gated_delta = try stateObjectPtr(runtime_graph_mod.GatedDeltaCache, state_block);
        gated_delta.* = runtime_graph_mod.GatedDeltaCache.init(self.layer_count);
        if (gated_delta.handle == null) return error.OutOfMemory;
        return true;
    }

    fn resetGatedDeltaState(state_block: *const runtime_contract.StateBlockHandle) !void {
        (try stateObjectPtr(runtime_graph_mod.GatedDeltaCache, state_block)).reset();
    }

    fn deinitGatedDeltaState(state_block: *const runtime_contract.StateBlockHandle) !void {
        const gated_delta = try stateObjectPtr(runtime_graph_mod.GatedDeltaCache, state_block);
        gated_delta.deinit();
        gated_delta.* = runtime_graph_mod.GatedDeltaCache.disabled();
    }

    const state_runtime_ops: [@typeInfo(StateRuntimeRole).@"enum".fields.len]StateRuntimeOps = blk: {
        var table: [@typeInfo(StateRuntimeRole).@"enum".fields.len]StateRuntimeOps = undefined;
        table[@intFromEnum(StateRuntimeRole.none)] = .{
            .init = initNoopState,
            .reset = resetNoopState,
            .deinit = deinitNoopState,
        };
        table[@intFromEnum(StateRuntimeRole.kv_cache)] = .{
            .init = initKvCacheState,
            .reset = resetKvCacheState,
            .deinit = deinitKvCacheState,
        };
        table[@intFromEnum(StateRuntimeRole.shortconv_cache)] = .{
            .init = initShortConvState,
            .reset = resetShortConvState,
            .deinit = deinitShortConvState,
        };
        table[@intFromEnum(StateRuntimeRole.mamba_cache)] = .{
            .init = initMambaState,
            .reset = resetMambaState,
            .deinit = deinitMambaState,
        };
        table[@intFromEnum(StateRuntimeRole.gated_delta_cache)] = .{
            .init = initGatedDeltaState,
            .reset = resetGatedDeltaState,
            .deinit = deinitGatedDeltaState,
        };
        break :blk table;
    };

    fn runtimeStateOps(role: StateRuntimeRole) StateRuntimeOps {
        return state_runtime_ops[@intFromEnum(role)];
    }

    fn initStateObjectForDescriptor(
        self: *MetalBackend,
        role: StateRuntimeRole,
        state_block: *const runtime_contract.StateBlockHandle,
    ) !bool {
        const ops = runtimeStateOps(role);
        return try ops.init(self, state_block);
    }

    fn resetStateObjectForDescriptor(
        role: StateRuntimeRole,
        state_block: *const runtime_contract.StateBlockHandle,
    ) !void {
        const ops = runtimeStateOps(role);
        try ops.reset(state_block);
    }

    fn runtimeHandleLooksValid(handle: ?*anyopaque) bool {
        if (handle == null) return false;
        return @intFromPtr(handle.?) >= 4096;
    }

    fn stateRoleName(role: StateRuntimeRole) []const u8 {
        return switch (role) {
            .none => "none",
            .kv_cache => "kv_cache",
            .shortconv_cache => "shortconv_cache",
            .mamba_cache => "mamba_cache",
            .gated_delta_cache => "gated_delta_cache",
        };
    }

    const LayerRoleCounts = struct {
        attention_layers: usize = 0,
        shortconv_layers: usize = 0,
        mamba_layers: usize = 0,
        gated_delta_layers: usize = 0,
    };

    fn countLayerRoles(handles: *const weights_trait.WeightHandles) LayerRoleCounts {
        var counts = LayerRoleCounts{};
        for (handles.layers) |*layer| {
            switch (layer.kind) {
                .attention_mlp => counts.attention_layers += 1,
                .shortconv => counts.shortconv_layers += 1,
                .mamba => counts.mamba_layers += 1,
                .gated_delta => counts.gated_delta_layers += 1,
            }
        }
        return counts;
    }

    const StateRoleCounts = struct {
        kv_roles: usize = 0,
        shortconv_roles: usize = 0,
        mamba_roles: usize = 0,
        gated_delta_roles: usize = 0,
    };

    fn countStateRoles(roles: []const StateRuntimeRole) StateRoleCounts {
        var counts = StateRoleCounts{};
        for (roles) |role| {
            switch (role) {
                .kv_cache => counts.kv_roles += 1,
                .shortconv_cache => counts.shortconv_roles += 1,
                .mamba_cache => counts.mamba_roles += 1,
                .gated_delta_cache => counts.gated_delta_roles += 1,
                .none => {},
            }
        }
        return counts;
    }

    fn validateModelStateTopology(
        layer_counts: LayerRoleCounts,
        state_counts: StateRoleCounts,
    ) !void {
        if (layer_counts.attention_layers > 0 and state_counts.kv_roles == 0) {
            return error.InvalidStateDescriptorBinding;
        }
        if (layer_counts.shortconv_layers > 0 and state_counts.shortconv_roles == 0) {
            return error.InvalidStateDescriptorBinding;
        }
        if (layer_counts.mamba_layers > 0 and state_counts.mamba_roles == 0) {
            return error.InvalidStateDescriptorBinding;
        }
        if (layer_counts.gated_delta_layers > 0 and state_counts.gated_delta_roles == 0) {
            return error.InvalidStateDescriptorBinding;
        }
    }

    fn panicInvalidSlotState(
        self: *MetalBackend,
        slot_index: usize,
        descriptor_idx: usize,
        reason: []const u8,
    ) noreturn {
        const descriptors = self.stateDescriptors();
        const descriptor = descriptors[descriptor_idx];
        const role = self.state_runtime_roles[descriptor_idx];
        log.warn("inference", "Metal slot state integrity failure (fatal)", .{
            .slot = slot_index,
            .descriptor_index = descriptor_idx,
            .descriptor_id = descriptor.id,
            .runtime_kind = descriptor.runtime_kind,
            .role = stateRoleName(role),
            .reason = reason,
        });
        std.debug.panic(
            "metal slot state integrity failure: slot={} descriptor_index={} descriptor_id={} role={s} reason={s}",
            .{ slot_index, descriptor_idx, descriptor.id, stateRoleName(role), reason },
        );
    }

    fn stateObjectLooksValid(
        role: StateRuntimeRole,
        state_block: *const runtime_contract.StateBlockHandle,
    ) bool {
        return switch (role) {
            .none => true,
            .kv_cache => blk: {
                const cache = stateObjectPtr(runtime_graph_mod.Cache, state_block) catch break :blk false;
                break :blk cache.isValid() and runtimeHandleLooksValid(cache.handle);
            },
            .shortconv_cache => blk: {
                const cache = stateObjectPtr(runtime_graph_mod.ShortConvCache, state_block) catch break :blk false;
                break :blk cache.isValid() and runtimeHandleLooksValid(cache.handle);
            },
            .mamba_cache => blk: {
                const cache = stateObjectPtr(runtime_graph_mod.MambaCache, state_block) catch break :blk false;
                break :blk cache.isValid() and runtimeHandleLooksValid(cache.handle);
            },
            .gated_delta_cache => blk: {
                const cache = stateObjectPtr(runtime_graph_mod.GatedDeltaCache, state_block) catch break :blk false;
                break :blk cache.isValid() and runtimeHandleLooksValid(cache.handle);
            },
        };
    }

    fn deinitStateObjectForDescriptor(
        role: StateRuntimeRole,
        state_block: *const runtime_contract.StateBlockHandle,
    ) !void {
        const ops = runtimeStateOps(role);
        try ops.deinit(state_block);
    }

    fn deinitSlotBindingStateObjects(self: *MetalBackend, binding: *SlotStateBinding) void {
        const count = @min(@as(usize, @intCast(binding.count)), @as(usize, self.state_descriptor_count));
        for (0..count) |idx| {
            if (!binding.initialized[idx]) continue;
            deinitStateObjectForDescriptor(self.state_runtime_roles[idx], &binding.handles[idx]) catch {};
            binding.initialized[idx] = false;
        }
    }

    fn releaseNonPersistentStateObjects(self: *MetalBackend, binding: *SlotStateBinding) void {
        const descriptors = self.stateDescriptors();
        const count = @min(@as(usize, @intCast(binding.count)), descriptors.len);
        for (0..count) |idx| {
            if (!binding.initialized[idx]) continue;
            if (descriptors[idx].lifecycle == .slot_persistent) continue;
            deinitStateObjectForDescriptor(self.state_runtime_roles[idx], &binding.handles[idx]) catch {};
            binding.initialized[idx] = false;
        }
    }

    fn resetSlotState(self: *MetalBackend, slot_index: usize) !void {
        const slot_position = try self.slotPositionPtr(slot_index);
        const slot_rope_delta = try self.slotRopeDeltaPtr(slot_index);
        slot_position.* = 0;
        slot_rope_delta.* = 0;
        if (self.state_descriptor_count == 0) return;
        const binding = try self.slotStateBinding(slot_index);
        if (!binding.bound) return;
        const descriptors = self.stateDescriptors();
        const count = @min(@as(usize, @intCast(binding.count)), descriptors.len);
        for (0..count) |idx| {
            if (!binding.initialized[idx]) continue;
            const role = self.state_runtime_roles[idx];
            if (!stateObjectLooksValid(role, &binding.handles[idx])) {
                binding.initialized[idx] = try self.initStateObjectForDescriptor(role, &binding.handles[idx]);
            }
            try resetStateObjectForDescriptor(role, &binding.handles[idx]);
        }
    }

    fn prefillSlotImpl(self: *MetalBackend, slot_index: usize, tokens: []const u32, logits_out: []f32) !void {
        const prev_backend = trace.setBackendContext(.metal);
        defer _ = trace.setBackendContext(prev_backend);
        const sequence_len = tokens.len;
        if (sequence_len == 0) return;
        const trace_prefill_timing = std.posix.getenv("TALU_METAL_PREFILL_TIMING") != null;
        const t_start_ns: i128 = if (trace_prefill_timing) std.time.nanoTimestamp() else 0;

        const slot_position = try self.slotPositionPtr(slot_index);
        const slot_rope_delta = try self.slotRopeDeltaPtr(slot_index);
        slot_rope_delta.* = 0;

        // Reset cache for new sequence in this scheduler slot.
        try self.resetSlotState(slot_index);
        const t_reset_done_ns: i128 = if (trace_prefill_timing) std.time.nanoTimestamp() else 0;
        const state_blocks = try self.slotStateBlocks(slot_index);

        const emit_final_path_points = emitParityFinalPathCheckpoints();
        const runtime_rope_ctx = self.textRuntimeRoPEOverride(0, sequence_len);
        const t_graph_built_ns: i128 = if (trace_prefill_timing) std.time.nanoTimestamp() else 0;
        var t_eval_done_ns: i128 = t_graph_built_ns;
        var t_copy_done_ns: i128 = t_graph_built_ns;

        if (!emit_final_path_points) {
            // Staged prefill (mlx parity): prefill prefix hidden state, then
            // run one-token decode projection on the last prompt token.
            if (sequence_len > 1) {
                const prefix_hidden = try runtime_trait.transformerForwardHiddenLazyWithEmbeddingOverride(
                    self.allocator,
                    self.weights,
                    tokens[0 .. sequence_len - 1],
                    state_blocks,
                    self.config,
                    0,
                    null,
                    null,
                    runtime_rope_ctx,
                );
                defer graph.freeArray(prefix_hidden);
                graph.eval(&[_]graph.ArrayHandle{prefix_hidden});
            }

            const raw_logits = try runtime_trait.transformerForwardLazyWithEmbeddingOverride(
                self.allocator,
                self.weights,
                tokens[sequence_len - 1 .. sequence_len],
                state_blocks,
                self.config,
                sequence_len - 1,
                null,
                null,
                runtime_rope_ctx,
            );
            defer graph.freeArray(raw_logits);
            const logits_handle = applyLogitsScalingHandle(self, raw_logits);
            if (logits_handle != raw_logits) {
                defer graph.freeArray(logits_handle);
            }
            graph.eval(&[_]graph.ArrayHandle{logits_handle});
            t_eval_done_ns = if (trace_prefill_timing) std.time.nanoTimestamp() else 0;
            graph.copyToHost(logits_handle, logits_out[0..self.vocab_size]);
            t_copy_done_ns = if (trace_prefill_timing) std.time.nanoTimestamp() else 0;
        } else {
            const hidden_handle = try runtime_trait.transformerForwardHiddenLazyWithEmbeddingOverride(
                self.allocator,
                self.weights,
                tokens,
                state_blocks,
                self.config,
                0, // pos_offset
                null,
                null,
                runtime_rope_ctx,
            );
            defer graph.freeArray(hidden_handle);
            var starts: [3]c_int = .{ 0, @intCast(sequence_len - 1), 0 };
            var ends: [3]c_int = .{ 1, @intCast(sequence_len), @intCast(self.d_model) };
            const last_hidden_handle = graph.mlx_lazy_slice(hidden_handle, &starts, &ends, 3);
            defer graph.freeArray(last_hidden_handle);
            try self.projectLastNormedHiddenToLogits(
                last_hidden_handle,
                logits_out,
                emit_final_path_points,
            );
            const t_done = if (trace_prefill_timing) std.time.nanoTimestamp() else 0;
            t_eval_done_ns = t_done;
            t_copy_done_ns = t_done;
        }
        try self.emitLmHeadFromReadyLogits(logits_out);
        if (trace.shouldEmit(.logits_ready)) {
            trace.emitFinal(
                .logits_ready,
                @intCast(sequence_len - 1),
                @intCast(sequence_len),
                @ptrCast(logits_out.ptr),
                .f32,
                .{ @intCast(self.vocab_size), 0, 0, 0 },
                1,
                "metal_logits_host",
            );
        }

        const t_end_ns: i128 = if (trace_prefill_timing) std.time.nanoTimestamp() else 0;

        slot_position.* = sequence_len;

        if (trace_prefill_timing) {
            const reset_ns = t_reset_done_ns - t_start_ns;
            const graph_build_ns = t_graph_built_ns - t_reset_done_ns;
            const build_ns = t_graph_built_ns - t_start_ns;
            const eval_ns = t_eval_done_ns - t_graph_built_ns;
            const copy_ns = t_copy_done_ns - t_eval_done_ns;
            const finalize_ns = t_end_ns - t_copy_done_ns;
            const total_ns = t_end_ns - t_start_ns;
            std.debug.print(
                "METAL_PREFILL_TIMING slot={} seq={} reset_us={d:.3} graph_us={d:.3} build_us={d:.3} eval_us={d:.3} copy_us={d:.3} finalize_us={d:.3} total_us={d:.3}\n",
                .{
                    slot_index,
                    sequence_len,
                    @as(f64, @floatFromInt(reset_ns)) / 1000.0,
                    @as(f64, @floatFromInt(graph_build_ns)) / 1000.0,
                    @as(f64, @floatFromInt(build_ns)) / 1000.0,
                    @as(f64, @floatFromInt(eval_ns)) / 1000.0,
                    @as(f64, @floatFromInt(copy_ns)) / 1000.0,
                    @as(f64, @floatFromInt(finalize_ns)) / 1000.0,
                    @as(f64, @floatFromInt(total_ns)) / 1000.0,
                },
            );
        }
    }

    fn prefillGreedySeedTokenImpl(self: *MetalBackend, slot_index: usize, tokens: []const u32) !u32 {
        const prev_backend = trace.setBackendContext(.metal);
        defer _ = trace.setBackendContext(prev_backend);
        const sequence_len = tokens.len;
        if (sequence_len == 0) return error.InvalidArgument;

        if (emitParityFinalPathCheckpoints()) {
            const logits_host = try self.allocator.alloc(f32, self.vocab_size);
            defer self.allocator.free(logits_host);
            try self.prefillSlotImpl(slot_index, tokens, logits_host);
            return self.selectNextTokenFromLogits(logits_host[0..self.vocab_size]);
        }

        const slot_position = try self.slotPositionPtr(slot_index);
        const slot_rope_delta = try self.slotRopeDeltaPtr(slot_index);
        slot_rope_delta.* = 0;

        // Reset cache for new sequence in this scheduler slot.
        try self.resetSlotState(slot_index);
        const state_blocks = try self.slotStateBlocks(slot_index);
        const runtime_rope_ctx = self.textRuntimeRoPEOverride(0, sequence_len);

        // Staged prefill (mlx parity): prefill prefix hidden state, then
        // run one-token decode projection on the last prompt token.
        if (sequence_len > 1) {
            const prefix_hidden = try runtime_trait.transformerForwardHiddenLazyWithEmbeddingOverride(
                self.allocator,
                self.weights,
                tokens[0 .. sequence_len - 1],
                state_blocks,
                self.config,
                0,
                null,
                null,
                runtime_rope_ctx,
            );
            defer graph.freeArray(prefix_hidden);
            graph.eval(&[_]graph.ArrayHandle{prefix_hidden});
        }

        const raw_logits = try runtime_trait.transformerForwardLazyWithEmbeddingOverride(
            self.allocator,
            self.weights,
            tokens[sequence_len - 1 .. sequence_len],
            state_blocks,
            self.config,
            sequence_len - 1,
            null,
            null,
            runtime_rope_ctx,
        );
        defer graph.freeArray(raw_logits);
        const logits_handle = applyLogitsScalingHandle(self, raw_logits);
        if (logits_handle != raw_logits) {
            defer graph.freeArray(logits_handle);
        }

        const selection_logits = if (self.config.logits_scaling < 0.0)
            graph.mlx_lazy_multiply_scalar(logits_handle, -1.0)
        else
            logits_handle;
        defer if (selection_logits != logits_handle) graph.freeArray(selection_logits);

        const token_handle = graph.mlx_lazy_argmax_owned(selection_logits, -1) orelse return error.OutOfMemory;
        defer graph.freeArray(token_handle);
        graph.eval(&[_]graph.ArrayHandle{token_handle});
        const token_id = graph.mlx_array_item_u32(token_handle);

        // XRAY observability for parity runs: keep this on the selected route
        // and only materialize host logits when explicitly requested.
        if (trace.shouldEmit(.logits_ready) or trace.shouldEmit(.lm_head)) {
            const logits_host = try self.allocator.alloc(f32, self.vocab_size);
            defer self.allocator.free(logits_host);
            graph.copyToHost(logits_handle, logits_host);
            try self.emitLmHeadFromReadyLogits(logits_host);
            if (trace.shouldEmit(.logits_ready)) {
                trace.emitFinal(
                    .logits_ready,
                    @intCast(sequence_len - 1),
                    @intCast(sequence_len),
                    @ptrCast(logits_host.ptr),
                    .f32,
                    .{ @intCast(self.vocab_size), 0, 0, 0 },
                    1,
                    "metal_logits_host",
                );
            }
        }

        slot_position.* = sequence_len;
        return token_id;
    }

    fn decodeSlot(
        self: *MetalBackend,
        slot_index: usize,
        token: u32,
        position: usize,
        logits_out: []f32,
    ) !void {
        const prev_backend = trace.setBackendContext(.metal);
        defer _ = trace.setBackendContext(prev_backend);
        const parity_log = @intFromEnum(log.Level.trace) >= @intFromEnum(log.getLogLevel());
        const slot_position_ptr = try self.slotPositionPtr(slot_index);
        const slot_position_before = slot_position_ptr.*;
        const slot_rope_delta = try self.slotRopeDeltaPtr(slot_index);
        const effective_position = try common_mrope.applyPositionDelta(position, slot_rope_delta.*);
        const runtime_rope_ctx = self.textRuntimeRoPEOverride(effective_position, 1);
        const state_blocks = try self.slotStateBlocks(slot_index);
        const emit_final_path_points = emitParityFinalPathCheckpoints();
        if (emit_final_path_points) {
            const hidden_handle = try runtime_trait.transformerForwardHiddenLazyWithEmbeddingOverride(
                self.allocator,
                self.weights,
                &[_]u32{token},
                state_blocks,
                self.config,
                effective_position,
                null,
                null,
                runtime_rope_ctx,
            );
            defer graph.freeArray(hidden_handle);
            try self.projectLastNormedHiddenToLogits(hidden_handle, logits_out, true);
        } else {
            const trace_decode_timing = std.posix.getenv("TALU_METAL_DECODE_TIMING") != null;
            const t_build_start: i128 = if (trace_decode_timing) std.time.nanoTimestamp() else 0;
            const raw_logits_handle = try runtime_trait.transformerForwardLazyWithEmbeddingOverride(
                self.allocator,
                self.weights,
                &[_]u32{token},
                state_blocks,
                self.config,
                effective_position,
                null,
                null,
                runtime_rope_ctx,
            );
            const t_build_done: i128 = if (trace_decode_timing) std.time.nanoTimestamp() else 0;
            const logits_handle = applyLogitsScalingHandle(self, raw_logits_handle);
            defer if (logits_handle != raw_logits_handle) graph.freeArray(raw_logits_handle);
            defer graph.freeArray(logits_handle);
            graph.eval(&[_]graph.ArrayHandle{logits_handle});
            const t_eval_done: i128 = if (trace_decode_timing) std.time.nanoTimestamp() else 0;
            graph.copyToHost(logits_handle, logits_out);
            const t_copy_done: i128 = if (trace_decode_timing) std.time.nanoTimestamp() else 0;
            if (trace_decode_timing) {
                std.debug.print(
                    "METAL_DECODE_TIMING slot={} pos={} build_us={d:.3} eval_us={d:.3} copy_us={d:.3} total_us={d:.3}\n",
                    .{
                        slot_index,
                        position,
                        @as(f64, @floatFromInt(t_build_done - t_build_start)) / 1000.0,
                        @as(f64, @floatFromInt(t_eval_done - t_build_done)) / 1000.0,
                        @as(f64, @floatFromInt(t_copy_done - t_eval_done)) / 1000.0,
                        @as(f64, @floatFromInt(t_copy_done - t_build_start)) / 1000.0,
                    },
                );
            }
        }
        if (parity_log) {
            const fp = fingerprintLogits(logits_out);
            log.trace(
                "inference",
                "PARITY_METAL decode logits",
                .{
                    .slot = slot_index,
                    .token = token,
                    .pos_arg = position,
                    .pos_before = slot_position_before,
                    .pos_effective = effective_position,
                    .rope_delta = slot_rope_delta.*,
                    .finite = fp.finite_count,
                    .non_finite = fp.non_finite_count,
                    .min = fp.min,
                    .max = fp.max,
                    .checksum = fp.checksum,
                },
                @src(),
            );
        }
        try self.emitLmHeadFromReadyLogits(logits_out);
        if (trace.shouldEmit(.logits_ready)) {
            trace.emitFinal(
                .logits_ready,
                0,
                @intCast(position + 1),
                @ptrCast(logits_out.ptr),
                .f32,
                .{ @intCast(self.vocab_size), 0, 0, 0 },
                1,
                "metal_logits_host",
            );
        }
        slot_position_ptr.* = position + 1;
    }

    fn decodeSlotGreedyHandleFromHostToken(
        self: *MetalBackend,
        slot_index: usize,
        token: u32,
        position: usize,
    ) !graph.ArrayHandle {
        const slot_rope_delta = try self.slotRopeDeltaPtr(slot_index);
        const effective_position = try common_mrope.applyPositionDelta(position, slot_rope_delta.*);
        const runtime_rope_ctx = self.textRuntimeRoPEOverride(effective_position, 1);
        const state_blocks = try self.slotStateBlocks(slot_index);
        const raw_logits_handle = try runtime_trait.transformerForwardLazyWithEmbeddingOverride(
            self.allocator,
            self.weights,
            &[_]u32{token},
            state_blocks,
            self.config,
            effective_position,
            null,
            null,
            runtime_rope_ctx,
        );
        const logits_handle = applyLogitsScalingHandle(self, raw_logits_handle);
        defer if (logits_handle != raw_logits_handle) graph.freeArray(raw_logits_handle);
        defer graph.freeArray(logits_handle);

        const selection_logits = if (self.config.logits_scaling < 0.0)
            graph.mlx_lazy_multiply_scalar(logits_handle, -1.0)
        else
            logits_handle;
        defer if (selection_logits != logits_handle) graph.freeArray(selection_logits);

        return graph.mlx_lazy_argmax_owned(selection_logits, -1) orelse error.OutOfMemory;
    }

    fn decodeSlotGreedyHandleFromGPUToken(
        self: *MetalBackend,
        slot_index: usize,
        token_handle: graph.ArrayHandle,
        position: usize,
    ) !graph.ArrayHandle {
        const slot_rope_delta = try self.slotRopeDeltaPtr(slot_index);
        const effective_position = try common_mrope.applyPositionDelta(position, slot_rope_delta.*);
        const runtime_rope_ctx = self.textRuntimeRoPEOverride(effective_position, 1);
        const state_blocks = try self.slotStateBlocks(slot_index);
        const raw_logits_handle = try runtime_trait.transformerForwardFromGPUToken(
            self.allocator,
            self.weights,
            token_handle,
            state_blocks,
            self.config,
            effective_position,
            runtime_rope_ctx,
        );
        const logits_handle = applyLogitsScalingHandle(self, raw_logits_handle);
        defer if (logits_handle != raw_logits_handle) graph.freeArray(raw_logits_handle);
        defer graph.freeArray(logits_handle);

        const selection_logits = if (self.config.logits_scaling < 0.0)
            graph.mlx_lazy_multiply_scalar(logits_handle, -1.0)
        else
            logits_handle;
        defer if (selection_logits != logits_handle) graph.freeArray(selection_logits);

        return graph.mlx_lazy_argmax_owned(selection_logits, -1) orelse error.OutOfMemory;
    }

    fn emitTokenSelectTrace(position: usize, token: u32) void {
        if (!trace.shouldEmit(.token_select)) return;
        trace.emitFinal(
            .token_select,
            0,
            @intCast(position + 1),
            @ptrCast(std.mem.asBytes(&token).ptr),
            .u32,
            .{ 1, 0, 0, 0 },
            1,
            "gpu_argmax",
        );
    }

    pub fn supportsSchedulerBackendTopKDecodeRoute(
        self: *const MetalBackend,
        sampling_config: *const metal_sampling.SamplingConfig,
    ) bool {
        _ = self;
        // Match scheduler top-k candidate route contract.
        return sampling_config.strategy == .top_k and
            sampling_config.top_k > 0 and
            sampling_config.temperature > 0.0 and
            sampling_config.min_p == 0.0;
    }

    pub fn decodeTopKCandidates(
        self: *MetalBackend,
        slot_index: usize,
        token: u32,
        top_k: usize,
        candidate_logits_out: []f32,
        candidate_ids_out: []u32,
    ) !usize {
        const prev_backend = trace.setBackendContext(.metal);
        defer _ = trace.setBackendContext(prev_backend);
        if (top_k == 0) return error.InvalidArgument;
        if (candidate_logits_out.len < top_k or candidate_ids_out.len < top_k) return error.InvalidArgument;

        const slot_position = try self.slotPositionPtr(slot_index);
        const slot_rope_delta = try self.slotRopeDeltaPtr(slot_index);
        const effective_position = try common_mrope.applyPositionDelta(slot_position.*, slot_rope_delta.*);
        const runtime_rope_ctx = self.textRuntimeRoPEOverride(effective_position, 1);
        const state_blocks = try self.slotStateBlocks(slot_index);
        const trace_decode_timing = std.posix.getenv("TALU_METAL_DECODE_TIMING") != null;
        const t_build_start: i128 = if (trace_decode_timing) std.time.nanoTimestamp() else 0;
        const raw_logits_handle = try runtime_trait.transformerForwardLazyWithEmbeddingOverride(
            self.allocator,
            self.weights,
            &[_]u32{token},
            state_blocks,
            self.config,
            effective_position,
            null,
            null,
            runtime_rope_ctx,
        );
        const t_build_done: i128 = if (trace_decode_timing) std.time.nanoTimestamp() else 0;
        const logits_handle = applyLogitsScalingHandle(self, raw_logits_handle);
        defer if (logits_handle != raw_logits_handle) graph.freeArray(raw_logits_handle);
        defer graph.freeArray(logits_handle);

        var shape_buffer: [8]usize = undefined;
        const rank = graph.getShape(logits_handle, &shape_buffer);
        if (rank == 0) return error.InvalidShape;

        const vocab_dim = shape_buffer[rank - 1];
        if (vocab_dim != self.vocab_size) return error.InvalidShape;

        const k = @min(top_k, vocab_dim);
        const axis: c_int = @intCast(rank - 1);
        const partitioned_indices = graph.mlx_lazy_argpartition(logits_handle, -@as(c_int, @intCast(k)), axis);
        defer graph.freeArray(partitioned_indices);

        var starts = [_]c_int{0} ** 8;
        var ends = [_]c_int{0} ** 8;
        for (0..rank) |dim| {
            ends[dim] = @intCast(shape_buffer[dim]);
        }
        starts[rank - 1] = @intCast(vocab_dim - k);

        const top_k_indices = graph.mlx_lazy_slice(partitioned_indices, &starts, &ends, rank);
        defer graph.freeArray(top_k_indices);

        const top_k_logits = graph.mlx_lazy_take_along_axis(logits_handle, top_k_indices, axis);
        defer graph.freeArray(top_k_logits);

        graph.eval(&[_]graph.ArrayHandle{ top_k_indices, top_k_logits });
        const t_eval_done: i128 = if (trace_decode_timing) std.time.nanoTimestamp() else 0;
        if (trace_decode_timing) {
            std.debug.print(
                "METAL_DECODE_TIMING slot={} pos={} build_us={d:.1} eval_us={d:.1}\n",
                .{
                    slot_index,
                    slot_position.*,
                    @as(f64, @floatFromInt(t_build_done - t_build_start)) / 1000.0,
                    @as(f64, @floatFromInt(t_eval_done - t_build_done)) / 1000.0,
                },
            );
        }
        graph.copyToHost(top_k_logits, candidate_logits_out[0..k]);
        graph.copyU32ToHost(top_k_indices, candidate_ids_out[0..k]);
        slot_position.* += 1;
        return k;
    }

    fn slotLogits(self: *MetalBackend, slot_index: usize) ![]f32 {
        if (slot_index >= self.max_batch_size) return error.InvalidArgument;
        const start = slot_index * self.vocab_size;
        return self.slot_logits_buffer[start .. start + self.vocab_size];
    }

    pub fn maxBatchSize(self: *const MetalBackend) usize {
        return self.max_batch_size;
    }

    pub fn vocabSize(self: *const MetalBackend) usize {
        return self.vocab_size;
    }

    pub fn supportsSchedulerBackendDecodeStreamingRoute(self: *const MetalBackend) bool {
        _ = self;
        // XRAY ACCEPTABLE USE:
        // Verify must never disable/enable scheduler routes.
        return true;
    }

    pub fn init(allocator: std.mem.Allocator, loaded: *LoadedModel) !MetalBackend {
        // Establish Metal device availability through the Objective-C wrapper
        // before MLX array ingestion touches its own implicit device path.
        // This keeps backend init on one explicit lifecycle boundary and turns
        // "no device" into a typed failure instead of an Objective-C abort.
        var init_device = try metal_compute.Device.init();
        defer init_device.deinit();
        // Backend init performs MLX weight ingest/transform work on this
        // thread before the first real generation starts. Clear pooled
        // run-scoped temporaries before returning so request 0 never inherits
        // leftover init-time array-pool state.
        defer graph.mlx_clear_thread_local_run_state();

        // Load weights to GPU
        const weight_handles = try weights_trait.loadWeightsToGPU(allocator, loaded);
        errdefer weights_trait.freeWeights(allocator, weight_handles);

        // Validate compiled layer plans once at backend init so forward path
        // stays focused on hot execution.
        for (weight_handles.layers, 0..) |*layer, layer_idx| {
            if (layer.compiled_plan == null) continue;
            try metal_executor.block.TransformerBlock.validateCompiledLayerProgram(layer, layer_idx);
        }

        const layer_count: usize = @intCast(loaded.config.n_layers);
        const max_seq_len: usize = @intCast(loaded.config.max_seq_len);
        const cache_max_seq_len = resolveCacheMaxSeqLen(max_seq_len);
        var text_runtime_rope = try buildTextRuntimeRoPE(allocator, loaded);
        errdefer if (text_runtime_rope) |*rope| rope.deinit(std.heap.c_allocator);
        var cpu_lm_head_scratch = try cpu_linalg.MatmulScratch.init(allocator);
        errdefer cpu_lm_head_scratch.deinit();
        const cpu_lm_head_fallback = resolveCpuLmHeadFallback();
        if (cpu_lm_head_fallback) {
            log.warn("inference", "Metal CPU lm_head fallback enabled", .{});
        }
        log.debug("inference", "Metal cache capacity policy", .{
            .config_max_seq_len = max_seq_len,
            .cache_max_seq_len = cache_max_seq_len,
            .dynamic_growth = @as(u8, @intFromBool(cache_max_seq_len == 0)),
        }, @src());
        var state_descriptors_storage: [MaxStateDescriptors]runtime_contract.StateDescriptor = undefined;
        var state_descriptor_count: u8 = 0;
        for (weight_handles.layers) |*layer| {
            if (layer.compiled_plan) |*compiled_plan| {
                try runtime_contract.appendUniquePlanStateDescriptors(
                    state_descriptors_storage[0..],
                    &state_descriptor_count,
                    &compiled_plan.plan,
                );
            }
        }
        const state_runtime_roles = try deriveStateRuntimeRoles(state_descriptors_storage[0..state_descriptor_count]);
        const layer_counts = countLayerRoles(weight_handles);
        const state_counts = countStateRoles(state_runtime_roles[0..state_descriptor_count]);
        log.debug("inference", "Metal runtime state topology", .{
            .attention_layers = layer_counts.attention_layers,
            .shortconv_layers = layer_counts.shortconv_layers,
            .mamba_layers = layer_counts.mamba_layers,
            .gated_delta_layers = layer_counts.gated_delta_layers,
            .kv_state_roles = state_counts.kv_roles,
            .shortconv_state_roles = state_counts.shortconv_roles,
            .mamba_state_roles = state_counts.mamba_roles,
            .gated_delta_state_roles = state_counts.gated_delta_roles,
            .state_descriptors = state_descriptor_count,
        }, @src());
        validateModelStateTopology(layer_counts, state_counts) catch |err| {
            log.err("inference", "Metal runtime state topology mismatch (fatal)", .{
                .attention_layers = layer_counts.attention_layers,
                .shortconv_layers = layer_counts.shortconv_layers,
                .mamba_layers = layer_counts.mamba_layers,
                .gated_delta_layers = layer_counts.gated_delta_layers,
                .kv_state_roles = state_counts.kv_roles,
                .shortconv_state_roles = state_counts.shortconv_roles,
                .mamba_state_roles = state_counts.mamba_roles,
                .gated_delta_state_roles = state_counts.gated_delta_roles,
                .reason = @errorName(err),
            }, @src());
            return err;
        };
        var vision_runtime = try vision_runtime_mod.VisionRuntime.init(allocator, loaded);
        errdefer if (vision_runtime) |*rt| rt.deinit();

        // Scheduler-visible slot capacity for Metal backend.
        const max_batch_size: usize = resolveMaxBatchSize();
        const extra_slot_count = max_batch_size - 1;
        const extra_slots = try allocator.alloc(SlotState, extra_slot_count);
        errdefer allocator.free(extra_slots);
        for (extra_slots) |*slot| {
            slot.* = .{
                .in_use = false,
                .position = 0,
                .rope_position_delta = 0,
            };
        }

        const slot_logits_buffer = try allocator.alloc(f32, max_batch_size * @as(usize, @intCast(loaded.config.vocab_size)));
        errdefer allocator.free(slot_logits_buffer);
        var backend = MetalBackend{
            .allocator = allocator,
            .loaded = loaded,
            .config = loaded.config,
            .weights = weight_handles,
            .cpu_lm_head_scratch = cpu_lm_head_scratch,
            .cpu_lm_head_fallback = cpu_lm_head_fallback,
            .layer_count = layer_count,
            .cache_max_seq_len = cache_max_seq_len,
            .vocab_size = @intCast(loaded.config.vocab_size),
            .d_model = @intCast(loaded.config.d_model),
            .max_batch_size = max_batch_size,
            .slot_in_use = false,
            .extra_slots = extra_slots,
            .slot_logits_buffer = slot_logits_buffer,
            .vision_runtime = vision_runtime,
            .slot_rope_position_delta = 0,
            .state_descriptors_storage = state_descriptors_storage,
            .state_runtime_roles = state_runtime_roles,
            .state_descriptor_count = state_descriptor_count,
            .current_position = 0,
            .text_runtime_rope = text_runtime_rope,
        };
        errdefer backend.deinit();
        return backend;
    }

    pub fn warmup(self: *MetalBackend) !void {
        // Warmup executes once during engine initialization, before first user
        // request, and must leave recurrent state reset afterward.
        const prev_slot_in_use = self.slot_in_use;
        defer self.slot_in_use = prev_slot_in_use;
        try self.ensureSlotStateBlocksBoundForScheduler(0);
        self.slot_in_use = true;
        try self.resetSlotState(0);

        // Prime both decode (seq=1) and prefill (seq>1) execution graphs.
        try self.primeBoundSlotExecutionGraph(0);

        // Prime scheduler top-k decode route, which has extra argpartition /
        // gather nodes beyond the plain decode path.
        var candidate_logits: [32]f32 = undefined;
        var candidate_ids: [32]u32 = undefined;
        _ = try self.decodeTopKCandidates(0, 0, 20, candidate_logits[0..20], candidate_ids[0..20]);

        // Reset warmup mutations so request 0 starts from pristine slot state.
        try self.resetSlotState(0);
        // Warmup bindings are scratch-only; force fresh state allocation on the
        // first real request to avoid carrying potentially stale handles.
        self.unbindSlotStateBlocks(0);
    }

    pub fn deinit(self: *MetalBackend) void {
        // Shutdown must happen after all queued MLX/Metal work on the default
        // device has completed. Otherwise array/cache destruction can race the
        // MLX stream thread and free backend-owned weights while GPU work still
        // references them. This is a deinit-only barrier, not a hot-path sync.
        synchronizeDefaultDevice();

        // Run-scoped pooled temporaries can still hold lazy graph references to
        // backend-owned weights/state at shutdown. Clear those transient arrays
        // before tearing down slot state objects or weight handles so pooled
        // temporaries never outlive their dependencies.
        //
        // Do not explicitly clear thread-local transform caches here. Those
        // caches are bounded, intentionally never-destroyed, and on macOS they
        // may hold ARC-managed Metal resources whose eager teardown has been
        // crashing verify runs after otherwise-successful execution.
        graph.mlx_clear_thread_local_run_state();
        self.cpu_lm_head_scratch.deinit();
        if (self.text_runtime_rope) |*rope| rope.deinit(std.heap.c_allocator);
        if (self.vision_runtime) |*rt| rt.deinit();
        if (self.slot0_state_binding.bound) {
            self.deinitSlotBindingStateObjects(&self.slot0_state_binding);
        }
        self.slot0_state_binding.clear();
        for (self.extra_slots) |*slot| {
            if (slot.state_binding.bound) {
                self.deinitSlotBindingStateObjects(&slot.state_binding);
            }
            slot.state_binding.clear();
        }

        // Cache/state teardown above destroys arrays that can still own the
        // last references to persistent weights. Wait for those releases to
        // quiesce before freeing the weight handles themselves.
        synchronizeDefaultDevice();

        self.allocator.free(self.slot_logits_buffer);
        self.allocator.free(self.extra_slots);
        weights_trait.freeWeights(self.allocator, self.weights);

        // Weight destruction can enqueue deferred MLX/Metal release work.
        // Flush that work before returning, but do not explicitly purge the
        // global MLX allocator cache here. On macOS, cached Metal buffers are
        // ARC-managed Objective-C objects and eager cache purges during deinit
        // have been crashing xray verify inside MLX buffer-cache teardown.
        //
        // The verify harness already isolates passes in short-lived child
        // processes, so letting process teardown reclaim the allocator cache is
        // the safer lifecycle boundary.
        synchronizeDefaultDevice();
        synchronizeDefaultDevice();

        self.* = undefined;
    }

    /// Barrier for xray/teardown code that needs MLX default-stream work to be
    /// fully visible to host-side capture serializers before they inspect or
    /// destroy traced tensors.
    pub fn synchronize(self: *MetalBackend) void {
        _ = self;
        synchronizeDefaultDevice();
    }

    /// Clear per-thread MLX transient state after a full generation run.
    /// This is an explicit lifecycle barrier for pooled temporaries on the
    /// thread that executed the run; persistent transform caches are cleared at
    /// backend shutdown, not between live runs.
    pub fn cleanupExecutionThreadState(self: *MetalBackend) void {
        _ = self;
        graph.mlx_clear_thread_local_run_state();
    }

    /// Reset thread-local MLX transient state on the execution thread before a
    /// new logical run begins. Persistent transform caches remain live across
    /// runs on purpose: they are bounded, keyed by backend-owned weights, and
    /// explicit cache destruction on macOS has proven unsafe during verify
    /// teardown.
    pub fn teardownExecutionThreadState(self: *MetalBackend) void {
        _ = self;
        graph.mlx_clear_thread_local_run_state();
    }

    /// Prefill: process all prompt tokens, return logits for last position
    pub fn prefill(self: *MetalBackend, tokens: []const u32, logits_out: []f32) !void {
        // Single-sequence entrypoint uses slot 0.
        try self.ensureSlotStateBlocksBoundForScheduler(0);
        self.slot_in_use = true;
        try self.prefillSlotImpl(0, tokens, logits_out);
    }

    pub fn embed(
        self: *MetalBackend,
        tokens: []const u32,
        pooling: contract.PoolingStrategy,
        normalize: bool,
        embedding_out: []f32,
    ) !void {
        if (tokens.len == 0) return error.EmptyInput;
        if (embedding_out.len < self.d_model) return error.BufferTooSmall;
        try self.ensureSlotStateBlocksBoundForScheduler(0);

        const hidden_handle = try runtime_trait.transformerForwardHiddenLazy(
            self.allocator,
            self.weights,
            tokens,
            try self.slotStateBlocks(0),
            self.config,
            0,
        );
        defer graph.freeArray(hidden_handle);

        graph.eval(&[_]graph.ArrayHandle{hidden_handle});

        const seq_len = tokens.len;
        const hidden_values = try self.allocator.alloc(f32, seq_len * self.d_model);
        defer self.allocator.free(hidden_values);
        graph.copyToHost(hidden_handle, hidden_values);

        const out = embedding_out[0..self.d_model];
        switch (pooling) {
            .last => {
                const last_offset = (seq_len - 1) * self.d_model;
                @memcpy(out, hidden_values[last_offset .. last_offset + self.d_model]);
            },
            .first => {
                @memcpy(out, hidden_values[0..self.d_model]);
            },
            .mean => {
                @memset(out, 0);
                for (0..seq_len) |row_idx| {
                    const row = hidden_values[row_idx * self.d_model ..][0..self.d_model];
                    for (out, row) |*dst, value| {
                        dst.* += value;
                    }
                }
                const inv_n = 1.0 / @as(f32, @floatFromInt(seq_len));
                for (out) |*v| v.* *= inv_n;
            },
        }

        if (normalize) {
            var sum_sq: f64 = 0;
            for (out) |v| {
                sum_sq += @as(f64, v) * @as(f64, v);
            }
            if (sum_sq > 0) {
                const inv_norm = @as(f32, @floatCast(1.0 / @sqrt(sum_sq)));
                for (out) |*v| v.* *= inv_norm;
            }
        }
    }

    pub fn embeddingDim(self: *const MetalBackend) usize {
        return self.d_model;
    }

    /// Allocate scheduler slot.
    pub fn allocSlot(self: *MetalBackend) ?usize {
        if (!self.slot_in_use) {
            self.slot_in_use = true;
            self.resetSlotState(0) catch return null;
            return 0;
        }
        for (self.extra_slots, 0..) |*slot, idx| {
            if (slot.in_use) continue;
            slot.in_use = true;
            self.resetSlotState(idx + 1) catch return null;
            return idx + 1;
        }
        return null;
    }

    /// Release scheduler slot.
    pub fn freeSlot(self: *MetalBackend, slot_index: usize) void {
        if (slot_index == 0) {
            self.unbindSlotStateBlocks(0);
            self.slot_in_use = false;
            self.resetSlotState(0) catch {};
            return;
        }
        const extra_idx = self.toExtraSlotIndex(slot_index) catch return;
        self.unbindSlotStateBlocks(slot_index);
        self.extra_slots[extra_idx].in_use = false;
        self.resetSlotState(slot_index) catch {};
    }

    /// Reset scheduler slot state without releasing ownership.
    pub fn resetSlot(self: *MetalBackend, slot_index: usize) void {
        self.resetSlotState(slot_index) catch {};
    }

    /// Return current decode position for scheduler slot.
    pub fn getPosition(self: *const MetalBackend, slot_index: usize) usize {
        if (slot_index == 0) return self.current_position;
        const extra_idx = self.toExtraSlotIndex(slot_index) catch return 0;
        return self.extra_slots[extra_idx].position;
    }

    pub fn stateDescriptors(self: *const MetalBackend) []const runtime_contract.StateDescriptor {
        return self.state_descriptors_storage[0..self.state_descriptor_count];
    }

    fn slotStateBinding(self: *MetalBackend, slot_index: usize) !*SlotStateBinding {
        if (slot_index == 0) return &self.slot0_state_binding;
        const extra_idx = try self.toExtraSlotIndex(slot_index);
        return &self.extra_slots[extra_idx].state_binding;
    }

    fn slotInUse(self: *const MetalBackend, slot_index: usize) !bool {
        if (slot_index == 0) return self.slot_in_use;
        const extra_idx = try self.toExtraSlotIndex(slot_index);
        return self.extra_slots[extra_idx].in_use;
    }

    fn slotStateBlocks(self: *MetalBackend, slot_index: usize) ![]const runtime_contract.StateBlockHandle {
        const binding = try self.slotStateBinding(slot_index);
        return binding.handles[0..binding.count];
    }

    pub fn ensureSlotStateBlocksBoundForScheduler(self: *MetalBackend, slot_index: usize) !void {
        if (self.state_descriptor_count == 0) return;
        const binding = try self.slotStateBinding(slot_index);
        if (!binding.bound) return error.InvalidStateDescriptorBinding;
        const descriptors = self.stateDescriptors();
        const count = @as(usize, @intCast(binding.count));
        if (count != descriptors.len) {
            self.panicInvalidSlotState(slot_index, 0, "bound descriptor count mismatch");
        }
        for (0..count) |idx| {
            const role = self.state_runtime_roles[idx];
            if (role == .none) continue;
            if (!binding.initialized[idx]) {
                binding.initialized[idx] = try self.initStateObjectForDescriptor(role, &binding.handles[idx]);
            }
            if (!stateObjectLooksValid(role, &binding.handles[idx])) {
                deinitStateObjectForDescriptor(role, &binding.handles[idx]) catch {};
                binding.initialized[idx] = try self.initStateObjectForDescriptor(role, &binding.handles[idx]);
                if (!binding.initialized[idx] or !stateObjectLooksValid(role, &binding.handles[idx])) {
                    self.panicInvalidSlotState(slot_index, idx, "runtime state object invalid");
                }
            }
        }
        if (comptime std.debug.runtime_safety) {
            try runtime_contract.validateStateBlocksForDescriptors(
                self.stateDescriptors(),
                try self.slotStateBlocks(slot_index),
            );
        }
    }

    pub fn bindSlotStateBlocks(
        self: *MetalBackend,
        slot_index: usize,
        state_blocks: []const runtime_contract.StateBlockHandle,
    ) !void {
        try runtime_contract.validateStateBlocksForDescriptors(self.stateDescriptors(), state_blocks);
        const binding = try self.slotStateBinding(slot_index);
        if (state_blocks.len > binding.handles.len or state_blocks.len > std.math.maxInt(u8)) {
            return error.InvalidStateDescriptorBinding;
        }
        if (state_blocks.len != self.stateDescriptors().len) return error.InvalidStateDescriptorBinding;
        binding.count = @intCast(state_blocks.len);
        for (self.stateDescriptors(), 0..) |descriptor, idx| {
            const incoming = runtime_contract.findStateBlock(state_blocks, descriptor.id) orelse {
                return error.InvalidStateDescriptorBinding;
            };
            if (binding.initialized[idx]) {
                const existing = binding.handles[idx];
                const same_block =
                    existing.id == incoming.id and
                    @intFromPtr(existing.ptr) == @intFromPtr(incoming.ptr) and
                    existing.size == incoming.size and
                    existing.align_bytes == incoming.align_bytes;
                if (!same_block) {
                    try deinitStateObjectForDescriptor(self.state_runtime_roles[idx], &binding.handles[idx]);
                    binding.initialized[idx] = false;
                }
            }
            binding.handles[idx] = .{
                .id = incoming.id,
                .ptr = incoming.ptr,
                .size = incoming.size,
                .align_bytes = incoming.align_bytes,
            };
            if (!binding.initialized[idx]) {
                binding.initialized[idx] = try self.initStateObjectForDescriptor(
                    self.state_runtime_roles[idx],
                    &binding.handles[idx],
                );
            }
        }
        binding.bound = true;
    }

    pub fn unbindSlotStateBlocks(self: *MetalBackend, slot_index: usize) void {
        const binding = self.slotStateBinding(slot_index) catch return;
        if (!binding.bound) return;
        const slot_in_use = self.slotInUse(slot_index) catch false;
        if (slot_in_use) {
            self.releaseNonPersistentStateObjects(binding);
            binding.detach();
            return;
        }
        self.deinitSlotBindingStateObjects(binding);
        binding.clear();
    }

    /// Scheduler prefill entrypoint.
    pub fn prefillSlot(
        self: *MetalBackend,
        slot_index: usize,
        tokens: []const u32,
        logits_out: []f32,
    ) !void {
        try self.ensureSlotStateBlocksBoundForScheduler(slot_index);
        if (slot_index == 0) self.slot_in_use = true else self.extra_slots[try self.toExtraSlotIndex(slot_index)].in_use = true;
        return self.prefillSlotImpl(slot_index, tokens, logits_out);
    }

    /// Prefill selected slot and return greedy seed token without host logits copy.
    /// Used by scheduler greedy-streaming route to match mlx token flow.
    pub fn prefillGreedySeedToken(
        self: *MetalBackend,
        slot_index: usize,
        tokens: []const u32,
    ) !u32 {
        try self.ensureSlotStateBlocksBoundForScheduler(slot_index);
        if (slot_index == 0) self.slot_in_use = true else self.extra_slots[try self.toExtraSlotIndex(slot_index)].in_use = true;
        return self.prefillGreedySeedTokenImpl(slot_index, tokens);
    }

    /// Scheduler prefill entrypoint with multimodal image payload.
    pub fn prefillSlotWithVision(
        self: *MetalBackend,
        slot_index: usize,
        tokens: []const u32,
        vision_input: ?*const PrefillVisionInput,
        logits_out: []f32,
    ) !void {
        const prev_backend = trace.setBackendContext(.metal);
        defer _ = trace.setBackendContext(prev_backend);
        try self.ensureSlotStateBlocksBoundForScheduler(slot_index);
        if (slot_index == 0) self.slot_in_use = true else self.extra_slots[try self.toExtraSlotIndex(slot_index)].in_use = true;
        if (vision_input == null) return self.prefillSlotImpl(slot_index, tokens, logits_out);
        const vi = vision_input.?;

        const sequence_len = tokens.len;
        if (sequence_len == 0) return;
        const slot_position = try self.slotPositionPtr(slot_index);
        const slot_rope_delta = try self.slotRopeDeltaPtr(slot_index);

        const vision = if (self.vision_runtime) |*rt|
            rt
        else {
            log.warn("inference", "Metal vision prefill requested but vision runtime is unavailable", .{
                .vision_hidden_size = self.config.vision_hidden_size,
                .vision_depth = self.config.vision_depth,
                .vision_num_heads = self.config.vision_num_heads,
                .vision_intermediate_size = self.config.vision_intermediate_size,
                .projector_hidden_size = self.config.projector_hidden_size,
                .vision_patch_size = self.config.vision_patch_size,
                .vision_spatial_merge_size = self.config.vision_spatial_merge_size,
                .vision_temporal_patch_size = self.config.vision_temporal_patch_size,
                .vision_num_position_embeddings = self.config.vision_num_position_embeddings,
                .vision_max_num_patches = self.config.vision_max_num_patches,
            });
            return error.UnsupportedContentType;
        };

        var encoded_vision = try vision.encodeImages(vi.images);
        defer encoded_vision.deinit(self.allocator);

        const embedding_handle = try runtime_trait.gatherTokenEmbeddingsLazy(self.weights, tokens);
        defer graph.freeArray(embedding_handle);
        graph.eval(&[_]graph.ArrayHandle{embedding_handle});

        const hidden_values = try self.allocator.alloc(f32, sequence_len * self.d_model);
        defer self.allocator.free(hidden_values);
        graph.copyToHost(embedding_handle, hidden_values);

        try vision.scatterIntoHidden(
            hidden_values,
            sequence_len,
            self.d_model,
            tokens,
            vi.image_token_id,
            encoded_vision.merged_embeddings,
        );
        slot_rope_delta.* = 0;

        var image_token_positions: []usize = &.{};
        defer if (image_token_positions.len > 0) self.allocator.free(image_token_positions);
        var deepstack_ctx: ?runtime_trait.DeepstackAdditions = null;
        if (encoded_vision.deepstack_layer_embeddings.len > 0) {
            image_token_positions = try collectTokenPositions(self.allocator, tokens, vi.image_token_id);
            if (image_token_positions.len == 0) return error.InvalidPromptImageTokens;
            deepstack_ctx = .{
                .positions = image_token_positions,
                .layer_features = encoded_vision.deepstack_layer_embeddings,
            };
        }

        var runtime_rope_cos: []f32 = &.{};
        var runtime_rope_sin: []f32 = &.{};
        defer if (runtime_rope_cos.len > 0) self.allocator.free(runtime_rope_cos);
        defer if (runtime_rope_sin.len > 0) self.allocator.free(runtime_rope_sin);
        var runtime_rope_ctx: ?runtime_trait.RuntimeRoPEOverride = null;
        const head_dim: usize = @intCast(self.config.head_dim);
        const mrope_section = common_mrope.resolveMropeSection(&self.config, head_dim);
        if (mrope_section[0] + mrope_section[1] + mrope_section[2] > 0) {
            const spatial_merge_size = std.math.cast(usize, self.config.vision_spatial_merge_size) orelse return error.InvalidShape;
            const tables = try buildMultimodalMropeTables(
                self.allocator,
                tokens,
                vi.images,
                vi.image_token_id,
                spatial_merge_size,
                head_dim,
                self.config.rope_theta,
                mrope_section,
            );
            runtime_rope_cos = tables.cos;
            runtime_rope_sin = tables.sin;
            slot_rope_delta.* = tables.position_delta;
            runtime_rope_ctx = .{
                .cos = runtime_rope_cos,
                .sin = runtime_rope_sin,
                .dim = head_dim,
            };
            log.debug("inference", "Metal text MRoPE prepared", .{
                .seq_len = sequence_len,
                .head_dim = head_dim,
                .section_t = mrope_section[0],
                .section_h = mrope_section[1],
                .section_w = mrope_section[2],
                .position_delta = slot_rope_delta.*,
            }, @src());
        }

        // Reset selected slot cache for new sequence.
        try self.resetSlotState(slot_index);

        const hidden_handle = try runtime_trait.transformerForwardHiddenLazyWithEmbeddingOverride(
            self.allocator,
            self.weights,
            tokens,
            try self.slotStateBlocks(slot_index),
            self.config,
            0,
            hidden_values,
            deepstack_ctx,
            runtime_rope_ctx,
        );
        defer graph.freeArray(hidden_handle);
        var starts: [3]c_int = .{ 0, @intCast(sequence_len - 1), 0 };
        var ends: [3]c_int = .{ 1, @intCast(sequence_len), @intCast(self.d_model) };
        const last_hidden_handle = graph.mlx_lazy_slice(hidden_handle, &starts, &ends, 3);
        defer graph.freeArray(last_hidden_handle);
        try self.projectLastNormedHiddenToLogits(
            last_hidden_handle,
            logits_out,
            emitParityFinalPathCheckpoints(),
        );
        if (trace.shouldEmit(.logits_ready)) {
            trace.emitFinal(
                .logits_ready,
                @intCast(sequence_len - 1),
                @intCast(sequence_len),
                @ptrCast(logits_out.ptr),
                .f32,
                .{ @intCast(self.vocab_size), 0, 0, 0 },
                1,
                "metal_logits_host",
            );
        }

        slot_position.* = sequence_len;
    }

    /// Scheduler decode entrypoint.
    pub fn decodeBatch(self: *MetalBackend, requests: []const contract.DecodeRequest, results: []contract.DecodeResult) !void {
        if (requests.len == 0) return;
        if (results.len < requests.len) return error.InvalidArgument;
        if (requests.len > self.max_batch_size) return error.InvalidArgument;

        // Validate bounds and detect duplicates without heap allocation.
        // Batch sizes are small (≤ max_batch_size, default 8), so O(N²) is fine.
        for (requests, 0..) |request, i| {
            if (request.slot_index >= self.max_batch_size) return error.InvalidArgument;
            for (requests[0..i]) |prev| {
                if (request.slot_index == prev.slot_index) return error.InvalidArgument;
            }
            try self.ensureSlotStateBlocksBoundForScheduler(request.slot_index);
        }

        for (requests, 0..) |request, idx| {
            const logits = try self.slotLogits(request.slot_index);
            const slot_position = try self.slotPositionPtr(request.slot_index);
            try self.decodeSlot(request.slot_index, request.token, slot_position.*, logits);
            results[idx] = .{
                .slot_index = request.slot_index,
                .logits = logits,
            };
        }
    }

    /// Decode: generate logits for a single token using KV cache
    pub fn decode(self: *MetalBackend, token: u32, position: usize, logits_out: []f32) !void {
        // Single-sequence entrypoint uses slot 0.
        try self.ensureSlotStateBlocksBoundForScheduler(0);
        self.slot_in_use = true;
        try self.decodeSlot(0, token, position, logits_out);
    }

    /// Decode with streaming — token-by-token loop through adapter table path.
    pub fn decodeStreaming(
        self: *MetalBackend,
        first_token: u32,
        start_position: usize,
        max_tokens: usize,
        eos_token_ids: []const u32,
        output_tokens: []u32,
        callback: ?*const fn (u32, ?*anyopaque) void,
        callback_data: ?*anyopaque,
    ) !usize {
        if (max_tokens == 0 or output_tokens.len == 0) return 0;
        try self.ensureSlotStateBlocksBoundForScheduler(0);
        self.slot_in_use = true;
        const prev_backend = trace.setBackendContext(.metal);
        defer _ = trace.setBackendContext(prev_backend);

        var decode_timer = std.time.Timer.start() catch unreachable;
        const budget = @min(max_tokens, output_tokens.len);
        var current_position = start_position;
        var generated_count: usize = 0;
        var current_token_handle = try self.decodeSlotGreedyHandleFromHostToken(0, first_token, current_position);
        defer if (current_token_handle != null) graph.freeArray(current_token_handle);
        graph.asyncEval(&[_]graph.ArrayHandle{current_token_handle});

        while (generated_count < budget) {
            if (xray.isVerifyStopRequested()) break;
            var next_token_handle: graph.ArrayHandle = null;
            const has_next_step = generated_count + 1 < budget;
            if (has_next_step) {
                next_token_handle = try self.decodeSlotGreedyHandleFromGPUToken(
                    0,
                    current_token_handle,
                    current_position + 1,
                );
                graph.asyncEval(&[_]graph.ArrayHandle{next_token_handle});
            }
            graph.eval(&[_]graph.ArrayHandle{current_token_handle});
            const next_token = graph.mlx_array_item_u32(current_token_handle);
            output_tokens[generated_count] = next_token;
            generated_count += 1;
            emitTokenSelectTrace(current_position, next_token);
            current_position += 1;

            if (callback) |cb| cb(next_token, callback_data);

            var is_eos = false;
            for (eos_token_ids) |eos_id| {
                if (next_token == eos_id) {
                    is_eos = true;
                    break;
                }
            }
            if (is_eos or generated_count >= budget) {
                if (next_token_handle != null) graph.freeArray(next_token_handle);
                break;
            }

            const prev_token_handle = current_token_handle;
            current_token_handle = next_token_handle;
            graph.freeArray(prev_token_handle);
        }

        const decode_ns = decode_timer.read();
        log.debug("inference", "Metal decode route complete", .{
            .route = "adapter_table_streaming",
            .requested_tokens = budget,
            .generated_tokens = generated_count,
            .duration_ms = @as(f64, @floatFromInt(decode_ns)) / 1_000_000.0,
            .tok_per_sec = if (decode_ns == 0)
                @as(f64, 0.0)
            else
                @as(f64, @floatFromInt(generated_count)) * 1_000_000_000.0 / @as(f64, @floatFromInt(decode_ns)),
            .has_callback = @as(u8, @intFromBool(callback != null)),
            .eos_count = eos_token_ids.len,
        }, @src());
        const slot_position = try self.slotPositionPtr(0);
        slot_position.* = start_position + generated_count;
        self.current_position = slot_position.*;
        return generated_count;
    }

    fn collectTokenPositions(
        allocator: std.mem.Allocator,
        token_ids: []const u32,
        needle: u32,
    ) ![]usize {
        var count: usize = 0;
        for (token_ids) |token| {
            if (token == needle) count += 1;
        }
        if (count == 0) return &.{};

        const positions = try allocator.alloc(usize, count);
        errdefer allocator.free(positions);

        var write_idx: usize = 0;
        for (token_ids, 0..) |token, idx| {
            if (token != needle) continue;
            positions[write_idx] = idx;
            write_idx += 1;
        }
        std.debug.assert(write_idx == count);
        return positions;
    }

    const MropeTables = struct {
        cos: []f32,
        sin: []f32,
        position_delta: isize,
    };

    fn buildMultimodalMropeTables(
        allocator: std.mem.Allocator,
        tokens: []const u32,
        images: []const vision_runtime_mod.PrefillVisionImage,
        image_token_id: u32,
        spatial_merge_size: usize,
        head_dim: usize,
        rope_theta: f32,
        mrope_section: [3]usize,
    ) !MropeTables {
        if (tokens.len == 0) return .{ .cos = &.{}, .sin = &.{}, .position_delta = 0 };
        if ((head_dim % 2) != 0 or mrope_section[0] + mrope_section[1] + mrope_section[2] != head_dim / 2) {
            return error.InvalidShape;
        }

        const seq_len = tokens.len;
        const pos_t = try allocator.alloc(u32, seq_len);
        defer allocator.free(pos_t);
        const pos_h = try allocator.alloc(u32, seq_len);
        defer allocator.free(pos_h);
        const pos_w = try allocator.alloc(u32, seq_len);
        defer allocator.free(pos_w);

        try common_mrope.buildMultimodalMropePositions(
            tokens,
            images,
            image_token_id,
            spatial_merge_size,
            pos_t,
            pos_h,
            pos_w,
        );

        const half_dim = head_dim / 2;
        const inv_freq = try allocator.alloc(f32, half_dim);
        defer allocator.free(inv_freq);
        for (0..half_dim) |idx| {
            const exponent = @as(f32, @floatFromInt(2 * idx)) / @as(f32, @floatFromInt(head_dim));
            inv_freq[idx] = 1.0 / std.math.pow(f32, rope_theta, exponent);
        }

        const cos = try allocator.alloc(f32, seq_len * head_dim);
        errdefer allocator.free(cos);
        const sin = try allocator.alloc(f32, seq_len * head_dim);
        errdefer allocator.free(sin);

        const h_limit = mrope_section[1] * 3;
        const w_limit = mrope_section[2] * 3;
        for (0..seq_len) |token_idx| {
            const base = token_idx * head_dim;
            for (0..half_dim) |freq_idx| {
                var pos_component = pos_t[token_idx];
                if (freq_idx < h_limit and (freq_idx % 3) == 1) pos_component = pos_h[token_idx];
                if (freq_idx < w_limit and (freq_idx % 3) == 2) pos_component = pos_w[token_idx];

                const angle = @as(f32, @floatFromInt(pos_component)) * inv_freq[freq_idx];
                const c = @cos(angle);
                const s = @sin(angle);
                cos[base + freq_idx] = c;
                sin[base + freq_idx] = s;
                cos[base + half_dim + freq_idx] = c;
                sin[base + half_dim + freq_idx] = s;
            }
        }

        const position_delta = try common_mrope.computePositionDelta(pos_t, pos_h, pos_w);

        return .{
            .cos = cos,
            .sin = sin,
            .position_delta = position_delta,
        };
    }
};

test "extraSlotIndexFor maps scheduler slots to extra-slot storage" {
    try std.testing.expectEqual(@as(usize, 0), try MetalBackend.extraSlotIndexFor(1, 4));
    try std.testing.expectEqual(@as(usize, 2), try MetalBackend.extraSlotIndexFor(3, 4));
}

test "extraSlotIndexFor rejects slot 0 and out-of-range slots" {
    try std.testing.expectError(error.InvalidArgument, MetalBackend.extraSlotIndexFor(0, 4));
    try std.testing.expectError(error.InvalidArgument, MetalBackend.extraSlotIndexFor(4, 4));
}

test "argmaxHost and argminHost select expected indices" {
    const values = [_]f32{ 0.5, -4.0, 1.25, 0.8 };
    try std.testing.expectEqual(@as(u32, 2), MetalBackend.argmaxHost(values[0..]));
    try std.testing.expectEqual(@as(u32, 1), MetalBackend.argminHost(values[0..]));
}

test "resolveCacheMaxSeqLen uses dynamic growth for very large contexts" {
    try std.testing.expectEqual(@as(usize, 0), MetalBackend.resolveCacheMaxSeqLen(262_144));
    try std.testing.expectEqual(@as(usize, 40_960), MetalBackend.resolveCacheMaxSeqLen(40_960));
}

test "applyPositionDelta rejects negative resulting positions" {
    try std.testing.expectError(error.InvalidShape, common_mrope.applyPositionDelta(2, -3));
    try std.testing.expectEqual(@as(usize, 7), try common_mrope.applyPositionDelta(4, 3));
}

test "deriveStateRuntimeRoles maps descriptor runtime_kind values" {
    const descriptors = [_]runtime_contract.StateDescriptor{
        .{
            .id = runtime_contract.kv_cache_state_id,
            .size_bytes = @sizeOf(runtime_graph_mod.Cache),
            .align_bytes = @alignOf(runtime_graph_mod.Cache),
            .zero_init = false,
            .lifecycle = .slot_persistent,
            .runtime_kind = runtime_contract.state_runtime_kind_kv_cache,
        },
        .{
            .id = runtime_contract.shortconv_state_id,
            .size_bytes = @sizeOf(runtime_graph_mod.ShortConvCache),
            .align_bytes = @alignOf(runtime_graph_mod.ShortConvCache),
            .zero_init = false,
            .lifecycle = .slot_persistent,
            .runtime_kind = runtime_contract.state_runtime_kind_shortconv_cache,
        },
        .{
            .id = runtime_contract.mamba_state_id,
            .size_bytes = @sizeOf(runtime_graph_mod.MambaCache),
            .align_bytes = @alignOf(runtime_graph_mod.MambaCache),
            .zero_init = false,
            .lifecycle = .slot_persistent,
            .runtime_kind = runtime_contract.state_runtime_kind_mamba_cache,
        },
        .{
            .id = runtime_contract.gated_delta_state_id,
            .size_bytes = @sizeOf(runtime_graph_mod.GatedDeltaCache),
            .align_bytes = @alignOf(runtime_graph_mod.GatedDeltaCache),
            .zero_init = false,
            .lifecycle = .slot_persistent,
            .runtime_kind = runtime_contract.state_runtime_kind_gated_delta_cache,
        },
    };

    const roles = try MetalBackend.deriveStateRuntimeRoles(descriptors[0..]);
    try std.testing.expectEqual(MetalBackend.StateRuntimeRole.kv_cache, roles[0]);
    try std.testing.expectEqual(MetalBackend.StateRuntimeRole.shortconv_cache, roles[1]);
    try std.testing.expectEqual(MetalBackend.StateRuntimeRole.mamba_cache, roles[2]);
    try std.testing.expectEqual(MetalBackend.StateRuntimeRole.gated_delta_cache, roles[3]);
}

test "deriveStateRuntimeRoles rejects unsupported runtime_kind" {
    const descriptors = [_]runtime_contract.StateDescriptor{
        .{
            .id = 77,
            .size_bytes = 64,
            .align_bytes = 8,
            .zero_init = false,
            .lifecycle = .request_scoped,
            .runtime_kind = 255,
        },
    };

    try std.testing.expectError(
        error.InvalidStateDescriptorBinding,
        MetalBackend.deriveStateRuntimeRoles(descriptors[0..]),
    );
}

test "countStateRoles tallies runtime descriptor roles" {
    const roles = [_]MetalBackend.StateRuntimeRole{
        .kv_cache,
        .shortconv_cache,
        .gated_delta_cache,
        .gated_delta_cache,
        .none,
    };
    const counts = MetalBackend.countStateRoles(roles[0..]);
    try std.testing.expectEqual(@as(usize, 1), counts.kv_roles);
    try std.testing.expectEqual(@as(usize, 1), counts.shortconv_roles);
    try std.testing.expectEqual(@as(usize, 0), counts.mamba_roles);
    try std.testing.expectEqual(@as(usize, 2), counts.gated_delta_roles);
}

test "validateModelStateTopology rejects missing recurrent state role" {
    const layer_counts = MetalBackend.LayerRoleCounts{
        .attention_layers = 6,
        .gated_delta_layers = 18,
    };
    const state_counts = MetalBackend.StateRoleCounts{
        .kv_roles = 1,
        .gated_delta_roles = 0,
    };
    try std.testing.expectError(
        error.InvalidStateDescriptorBinding,
        MetalBackend.validateModelStateTopology(layer_counts, state_counts),
    );
}

test "runtimeHandleLooksValid rejects null and low-pointer handles" {
    try std.testing.expect(!MetalBackend.runtimeHandleLooksValid(null));
    try std.testing.expect(!MetalBackend.runtimeHandleLooksValid(@ptrFromInt(1024)));
    try std.testing.expect(MetalBackend.runtimeHandleLooksValid(@ptrFromInt(4096)));
}

test "buildMultimodalMropeTables computes multimodal decode position_delta" {
    const allocator = std.testing.allocator;
    const image_token_id: u32 = 151655;
    const vision_start_token_id: u32 = 151652;
    const vision_end_token_id: u32 = 151653;
    const seq_len: usize = 198;
    const image_span: usize = 169;
    const image_start: usize = 15;

    const tokens = try allocator.alloc(u32, seq_len);
    defer allocator.free(tokens);
    @memset(tokens, 42);
    tokens[image_start - 1] = vision_start_token_id;
    for (0..image_span) |idx| tokens[image_start + idx] = image_token_id;
    tokens[image_start + image_span] = vision_end_token_id;

    const images = [_]vision_runtime_mod.PrefillVisionImage{
        .{
            .pixels = &.{},
            .width = 416,
            .height = 416,
            .grid = .{ .temporal = 1, .height = 26, .width = 26 },
            .token_count = image_span,
        },
    };

    const tables = try MetalBackend.buildMultimodalMropeTables(
        allocator,
        tokens,
        images[0..],
        image_token_id,
        2,
        128,
        5_000_000.0,
        .{ 24, 20, 20 },
    );
    defer allocator.free(tables.cos);
    defer allocator.free(tables.sin);

    try std.testing.expectEqual(seq_len * 128, tables.cos.len);
    try std.testing.expectEqual(seq_len * 128, tables.sin.len);
    try std.testing.expectEqual(@as(isize, -156), tables.position_delta);
}

test "decodeBatch advances positions across multiple slots" {
    if (!isAvailable()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const loaded = try weights_trait.createTestLoadedModel(allocator);
    defer weights_trait.destroyTestLoadedModel(allocator, loaded);

    var backend = MetalBackend.init(allocator, loaded) catch |err| switch (err) {
        error.NotImplemented, error.InvalidStateDescriptorBinding => return error.SkipZigTest,
        else => return err,
    };
    defer backend.deinit();

    const slot0 = backend.allocSlot() orelse return error.TestUnexpectedResult;
    const slot1 = backend.allocSlot() orelse return error.TestUnexpectedResult;
    defer backend.freeSlot(slot1);
    defer backend.freeSlot(slot0);

    try std.testing.expectEqual(@as(usize, 0), slot0);
    try std.testing.expectEqual(@as(usize, 1), slot1);

    const vocab_size = backend.vocabSize();
    const prefill_logits0 = try allocator.alloc(f32, vocab_size);
    defer allocator.free(prefill_logits0);
    const prefill_logits1 = try allocator.alloc(f32, vocab_size);
    defer allocator.free(prefill_logits1);

    try backend.prefillSlot(slot0, &[_]u32{ 1, 2 }, prefill_logits0);
    try backend.prefillSlot(slot1, &[_]u32{ 3, 4 }, prefill_logits1);
    try std.testing.expectEqual(@as(usize, 2), backend.getPosition(slot0));
    try std.testing.expectEqual(@as(usize, 2), backend.getPosition(slot1));

    const requests = [_]contract.DecodeRequest{
        .{ .slot_index = slot0, .token = 5 },
        .{ .slot_index = slot1, .token = 6 },
    };
    var results: [2]contract.DecodeResult = undefined;
    try backend.decodeBatch(requests[0..], results[0..]);

    try std.testing.expectEqual(slot0, results[0].slot_index);
    try std.testing.expectEqual(slot1, results[1].slot_index);
    try std.testing.expectEqual(vocab_size, results[0].logits.len);
    try std.testing.expectEqual(vocab_size, results[1].logits.len);
    try std.testing.expect(results[0].logits.ptr != results[1].logits.ptr);
    try std.testing.expectEqual(@as(usize, 3), backend.getPosition(slot0));
    try std.testing.expectEqual(@as(usize, 3), backend.getPosition(slot1));

    const duplicate_requests = [_]contract.DecodeRequest{
        .{ .slot_index = slot0, .token = 7 },
        .{ .slot_index = slot0, .token = 8 },
    };
    var duplicate_results: [2]contract.DecodeResult = undefined;
    try std.testing.expectError(error.InvalidArgument, backend.decodeBatch(duplicate_requests[0..], duplicate_results[0..]));
    try std.testing.expectEqual(@as(usize, 3), backend.getPosition(slot0));
    try std.testing.expectEqual(@as(usize, 3), backend.getPosition(slot1));
}

test "decodeStreaming greedy token matches decode logits argmax" {
    if (!isAvailable()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const loaded = try weights_trait.createTestLoadedModel(allocator);
    defer weights_trait.destroyTestLoadedModel(allocator, loaded);

    var backend = MetalBackend.init(allocator, loaded) catch |err| switch (err) {
        error.NotImplemented, error.InvalidStateDescriptorBinding => return error.SkipZigTest,
        else => return err,
    };
    defer backend.deinit();

    const vocab_size = backend.vocabSize();
    const logits = try allocator.alloc(f32, vocab_size);
    defer allocator.free(logits);

    try backend.prefill(&[_]u32{ 1, 2 }, logits);
    const position = backend.getPosition(0);
    try backend.decode(3, position, logits);
    const expected = try backend.selectNextTokenFromLogits(logits);

    var output: [1]u32 = undefined;
    const generated = try backend.decodeStreaming(3, position, 1, &.{}, output[0..], null, null);

    try std.testing.expectEqual(@as(usize, 1), generated);
    try std.testing.expectEqual(expected, output[0]);
}

test "metal availability" {
    if (!isAvailable()) return error.SkipZigTest;
}

test "metal device creation" {
    if (!isAvailable()) return error.SkipZigTest;

    var metal_device = try Device.init();
    defer metal_device.deinit();

    const device_name = metal_device.name();
    try std.testing.expect(device_name.len > 0);
}

test "metal buffer allocation" {
    if (!isAvailable()) return error.SkipZigTest;

    var metal_device = try Device.init();
    defer metal_device.deinit();

    var metal_buffer = try metal_device.allocBuffer(1024);
    defer metal_buffer.deinit();

    const test_data = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 };
    metal_buffer.upload(&test_data);

    var result = [_]u8{0} ** 8;
    metal_buffer.download(&result);
    try std.testing.expectEqualSlices(u8, &test_data, &result);
}

test "metal f32 matmul" {
    if (!isAvailable()) return error.SkipZigTest;

    var metal_device = try Device.init();
    defer metal_device.deinit();

    const left_matrix = [_]f32{
        1, 2, 3,
        4, 5, 6,
    };
    const right_matrix = [_]f32{
        1, 0,
        0, 1,
        1, 1,
    };
    var output_matrix = [_]f32{0} ** 4;

    try matmul.matmulF32(
        &metal_device,
        &left_matrix,
        2,
        3,
        &right_matrix,
        2,
        &output_matrix,
    );

    const expected = [_]f32{ 4, 5, 10, 11 };
    for (output_matrix, expected) |actual, expected_value| {
        try std.testing.expectApproxEqAbs(expected_value, actual, 0.001);
    }
}

test {
    @import("std").testing.refAllDecls(@This());
}
