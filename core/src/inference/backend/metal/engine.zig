const std = @import("std");
const contract = @import("../contract.zig");
const runtime_contract = @import("../../runtime_contract/root.zig");
const models = @import("../../../models/root.zig");
const log = @import("../../../log.zig");
const progress_mod = @import("../../../progress.zig");
const metal_vision = @import("vision/root.zig");
const cpu_engine = @import("../cpu/engine.zig");
const sampling = @import("../cpu/sampling.zig");
const compute = @import("../../../compute/root.zig");
const trace = @import("../../../xray/trace.zig");
const cpu_sampling_ops = compute.cpu.sampling_ops;

const LoadedModel = models.LoadedModel;
const topk_route_candidate_capacity: usize = 256;
const default_metal_max_batch_size: usize = 4;

const mlx_ctx = opaque {};

extern fn mlx_is_available() c_int;
extern fn mlx_validate_config(model_path: [*:0]const u8) c_int;
extern fn mlx_create(model_id: [*:0]const u8, model_path: [*:0]const u8, seed: c_int) ?*mlx_ctx;
extern fn mlx_clone(source_ctx: ?*mlx_ctx, seed: c_int) ?*mlx_ctx;
extern fn mlx_destroy(ctx: ?*mlx_ctx) void;
extern fn mlx_reset(ctx: ?*mlx_ctx) c_int;
extern fn mlx_last_error() ?[*:0]const u8;

extern fn mlx_prefill_first(
    ctx: ?*mlx_ctx,
    prompt_ids: [*]const i32,
    prompt_len: c_int,
    out_first_token: *c_int,
) c_int;

extern fn mlx_prefill_logits(
    ctx: ?*mlx_ctx,
    prompt_ids: [*]const i32,
    prompt_len: c_int,
    out_logits: [*]f32,
    logits_len: c_int,
) c_int;

extern fn mlx_prefill_logits_batch(
    ctxs: [*]const *mlx_ctx,
    prompt_ids_ptrs: [*]const [*]const i32,
    prompt_lens: [*]const i32,
    out_logits_ptrs: [*]const [*]f32,
    batch_size: c_int,
    logits_len: c_int,
) c_int;

extern fn mlx_embed(
    ctx: ?*mlx_ctx,
    token_ids: [*]const i32,
    token_len: c_int,
    pooling: c_int,
    normalize: c_int,
    out_embedding: [*]f32,
    embedding_len: c_int,
) c_int;

extern fn mlx_decode_logits(
    ctx: ?*mlx_ctx,
    token: c_int,
    out_logits: [*]f32,
    logits_len: c_int,
) c_int;

extern fn mlx_decode_logits_batch(
    ctxs: [*]const *mlx_ctx,
    tokens: [*]const i32,
    out_logits_ptrs: [*]const [*]f32,
    batch_size: c_int,
    logits_len: c_int,
) c_int;

extern fn mlx_decode_stream(
    ctx: ?*mlx_ctx,
    first_token: c_int,
    decode_tokens: c_int,
    eos_ids: ?[*]const i32,
    eos_len: c_int,
    out_generated_ids: *[*c]i32,
    out_generated_len: *c_int,
) c_int;

extern fn mlx_decode_topk_candidates(
    ctx: ?*mlx_ctx,
    token: c_int,
    top_k: c_int,
    out_candidate_logits: [*]f32,
    out_candidate_ids: [*]i32,
    out_candidate_count: *c_int,
) c_int;

extern fn mlx_decode_topk_candidates_with_sampling(
    ctx: ?*mlx_ctx,
    token: c_int,
    top_k: c_int,
    repetition_penalty: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    context_ids: ?[*]const i32,
    context_len: c_int,
    out_candidate_logits: [*]f32,
    out_candidate_ids: [*]i32,
    out_candidate_count: *c_int,
) c_int;

extern fn mlx_tokens_free(ids: [*c]i32) void;

fn decodeTracePoint(point_id: u8) ?trace.TracePoint {
    const point: trace.TracePoint = @enumFromInt(point_id);
    return switch (point) {
        .embed,
        .embed_pos,
        .layer_input,
        .layer_attn_norm,
        .attn_q,
        .attn_k,
        .attn_v,
        .attn_q_proj_raw,
        .attn_k_proj_raw,
        .attn_q_norm,
        .attn_k_norm,
        .attn_q_rope,
        .attn_k_rope,
        .attn_qk,
        .attn_weights,
        .attn_out,
        .layer_ffn_norm,
        .ffn_gate,
        .ffn_up,
        .ffn_act,
        .ffn_down,
        .block_out,
        .mamba_out,
        .conv_in_proj,
        .conv_conv,
        .conv_out_proj,
        .final_norm,
        .lm_head,
        .logits_scaled,
        .logits_ready,
        .token_select,
        .ffn_act_map,
        .ffn_act_mix,
        .gdelta_in_proj,
        .gdelta_conv,
        .gdelta_ssm,
        .gdelta_norm,
        .gdelta_out,
        .gdelta_state_conv,
        .gdelta_state_ssm,
        => point,
        else => null,
    };
}

pub export fn talu_metal_xray_should_emit(point_id: u8, layer: u16, position: u32) c_int {
    if (!trace.isEnabled()) return 0;
    const point = decodeTracePoint(point_id) orelse return 0;
    return if (trace.shouldEmitEmission(point, layer, position)) 1 else 0;
}

pub export fn talu_metal_xray_is_enabled() c_int {
    return if (trace.isEnabled()) 1 else 0;
}

pub export fn talu_metal_xray_emit_f32(
    point_id: u8,
    layer: u16,
    token: u32,
    position: u32,
    ptr: [*]const f32,
    dim0: u32,
    dim1: u32,
    dim2: u32,
    dim3: u32,
    ndim: u8,
    kernel_name: ?[*:0]const u8,
) void {
    if (!trace.isEnabled()) return;
    if (ndim == 0 or ndim > 4) return;
    const point = decodeTracePoint(point_id) orelse return;

    const prev_backend = trace.setBackendContext(.metal);
    defer _ = trace.setBackendContext(prev_backend);

    const kernel = if (kernel_name) |name| std.mem.sliceTo(name, 0) else null;
    trace.emit(
        point,
        layer,
        token,
        position,
        @ptrCast(ptr),
        .f32,
        .{ dim0, dim1, dim2, dim3 },
        ndim,
        kernel,
    );
}

pub const MetalBackend = struct {
    pub const InitConfig = struct {
        model_path: ?[]const u8 = null,
        model_id: ?[]const u8 = null,
        memory_fit_is_error: bool = false,
    };

    pub const capabilities: contract.Capabilities = .{
        .vision_prefill = true,
        .decode_batch = true,
        .decode_streaming = true,
        .embedding = true,
        .warmup = false,
    };

    pub const PrefillVisionInput = metal_vision.PrefillVisionInput;

    allocator: std.mem.Allocator,
    loaded: *LoadedModel,
    model_id_z: [:0]u8,
    model_path_z: [:0]u8,
    seed: c_int,

    vocab_size: usize,
    d_model: usize,
    max_batch_size: usize = default_metal_max_batch_size,

    slot_ctxs: []?*mlx_ctx,
    slot_in_use: []bool,
    slot_positions: []usize,
    slot_logits_buffers: []f32,
    slot_token_scratch: []std.ArrayListUnmanaged(i32),
    slot_topk_ids_i32: []i32,
    topk_scratch_capacity: usize,
    batch_decode_ctxs: []*mlx_ctx,
    batch_decode_tokens_i32: []i32,
    batch_decode_logits_ptrs: [][*]f32,
    batch_prefill_ctxs: []*mlx_ctx,
    batch_prefill_prompt_ptrs: [][*]const i32,
    batch_prefill_prompt_lens_i32: []i32,
    batch_prefill_logits_ptrs: [][*]f32,
    slot_state_bindings: []SlotStateBinding,
    slot_route_modes: []SlotRouteMode,
    slot_delegate_slots: []?usize,
    vision_delegate: ?cpu_engine.FusedCpuBackend = null,
    state_descriptors_storage: [runtime_contract.max_state_descriptors]runtime_contract.StateDescriptor = undefined,
    state_descriptor_count: u8 = 0,

    const max_state_bindings_per_slot: usize = runtime_contract.max_state_descriptors;

    const SlotStateBinding = struct {
        bound: bool = false,
        count: u8 = 0,
        handles: [max_state_bindings_per_slot]runtime_contract.StateBlockHandle = undefined,

        fn clear(self: *SlotStateBinding) void {
            self.bound = false;
            self.count = 0;
        }
    };

    const SlotRouteMode = enum(u8) {
        mlx = 0,
        vision_delegate = 1,
    };

    fn appendProgramStateDescriptors(
        storage: []runtime_contract.StateDescriptor,
        count: *u8,
        entry: models.registry.Entry,
        program: []const models.layer_ops.LayerOp,
    ) !void {
        for (program) |op| {
            const state_id = switch (op) {
                .kernel => |kernel_op| kernel_op.state_block_id orelse continue,
                else => continue,
            };
            try runtime_contract.appendUniqueStateDescriptor(
                storage,
                count,
                try models.registry.stateDescriptorForId(entry, state_id),
            );
        }
    }

    fn collectPlanStateDescriptors(self: *MetalBackend) !void {
        const arch_id = self.loaded.runtime.architecture_id orelse return;
        const entry = models.registry.detectByArchitectureId(arch_id) orelse return;
        var count: u8 = 0;

        for (self.loaded.blocks) |layer| {
            const program = models.registry.blockProgramFor(entry, layer.block_type) orelse continue;
            try appendProgramStateDescriptors(
                self.state_descriptors_storage[0..],
                &count,
                entry,
                program,
            );
        }
        if (models.registry.visionProgramByArchitectureId(entry.id)) |program| {
            try appendProgramStateDescriptors(
                self.state_descriptors_storage[0..],
                &count,
                entry,
                program,
            );
        }
        self.state_descriptor_count = count;
    }

    fn readSeedFromEnv(allocator: std.mem.Allocator) c_int {
        const raw = std.process.getEnvVarOwned(allocator, "TALU_METAL_SEED") catch return 42;
        defer allocator.free(raw);
        const trimmed = std.mem.trim(u8, raw, " \t\r\n");
        const parsed = std.fmt.parseInt(i32, trimmed, 10) catch return 42;
        return @intCast(parsed);
    }

    fn resolveMaxBatchSize(allocator: std.mem.Allocator) usize {
        const parse_raw = struct {
            fn run(raw: []const u8) ?usize {
                const trimmed = std.mem.trim(u8, raw, " \t\r\n");
                if (trimmed.len == 0) return null;
                const parsed = std.fmt.parseUnsigned(usize, trimmed, 10) catch return null;
                return @max(@as(usize, 1), parsed);
            }
        }.run;

        if (std.process.getEnvVarOwned(allocator, "TALU_METAL_MAX_BATCH_SIZE")) |raw_metal| {
            defer allocator.free(raw_metal);
            if (parse_raw(raw_metal)) |parsed| return parsed;
            return default_metal_max_batch_size;
        } else |_| {}

        // Backward-compatible alias used by bench scenarios that set
        // TALU_CUDA_MAX_BATCH_SIZE across backends.
        if (std.process.getEnvVarOwned(allocator, "TALU_CUDA_MAX_BATCH_SIZE")) |raw_cuda| {
            defer allocator.free(raw_cuda);
            if (parse_raw(raw_cuda)) |parsed| return parsed;
        } else |_| {}

        return default_metal_max_batch_size;
    }

    fn resolveLastError() []const u8 {
        const raw = mlx_last_error() orelse return "mlx_last_error unavailable";
        return std.mem.sliceTo(raw, 0);
    }

    fn isMemoryError(message: []const u8) bool {
        return std.mem.indexOf(u8, message, "metal memory budget exceeded") != null or
            std.mem.indexOf(u8, message, "Insufficient Memory") != null or
            std.mem.indexOf(u8, message, "out of memory") != null or
            std.mem.indexOf(u8, message, "OutOfMemory") != null;
    }

    fn resetCtx(self: *MetalBackend, slot_index: usize) void {
        if (slot_index >= self.max_batch_size) return;
        const ctx = self.slot_ctxs[slot_index] orelse return;
        if (mlx_reset(ctx) == 0) {
            log.warn("inference", "metal mlx_reset failed", .{
                .mlx_error = resolveLastError(),
                .slot = slot_index,
            });
        }
    }

    fn parseOptionalEnvVarOwned(
        allocator: std.mem.Allocator,
        key: []const u8,
    ) ?[]u8 {
        return std.process.getEnvVarOwned(allocator, key) catch null;
    }

    fn clearSlotState(self: *MetalBackend, slot_index: usize) void {
        self.slot_positions[slot_index] = 0;
    }

    fn ensureSlotIndex(self: *const MetalBackend, slot_index: usize) !void {
        if (slot_index >= self.max_batch_size) return error.InvalidArgument;
    }

    fn slotLogitsSlice(self: *MetalBackend, slot_index: usize) []f32 {
        const offset = slot_index * self.vocab_size;
        return self.slot_logits_buffers[offset .. offset + self.vocab_size];
    }

    fn slotTopKIdsSlice(self: *MetalBackend, slot_index: usize) []i32 {
        const offset = slot_index * self.topk_scratch_capacity;
        return self.slot_topk_ids_i32[offset .. offset + self.topk_scratch_capacity];
    }

    fn fillTopKFromLogits(
        logits: []const f32,
        top_k: usize,
        candidate_logits_out: []f32,
        candidate_ids_out: []u32,
    ) usize {
        var count: usize = 0;
        for (logits, 0..) |logit, idx| {
            var insert_at: usize = count;
            while (insert_at > 0 and candidate_logits_out[insert_at - 1] < logit) {
                insert_at -= 1;
            }
            if (insert_at >= top_k) continue;
            const upper = @min(count, top_k - 1);
            var shift = upper;
            while (shift > insert_at) : (shift -= 1) {
                candidate_logits_out[shift] = candidate_logits_out[shift - 1];
                candidate_ids_out[shift] = candidate_ids_out[shift - 1];
            }
            candidate_logits_out[insert_at] = logit;
            candidate_ids_out[insert_at] = @intCast(idx);
            if (count < top_k) count += 1;
        }
        return count;
    }

    fn applySamplingMutationsToLogits(
        logits: []f32,
        sampling_config: *const sampling.SamplingConfig,
    ) void {
        const context_tokens = sampling_config.context_tokens orelse &.{};
        if (context_tokens.len == 0) return;
        if (sampling_config.repetition_penalty != 1.0) {
            cpu_sampling_ops.applyIndexPenalty(
                logits,
                context_tokens,
                sampling_config.repetition_penalty,
            );
        }
        if (sampling_config.presence_penalty != 0.0 or sampling_config.frequency_penalty != 0.0) {
            cpu_sampling_ops.applyAdditivePenalties(
                logits,
                context_tokens,
                sampling_config.presence_penalty,
                sampling_config.frequency_penalty,
            );
        }
    }

    fn ensureSlotTokenScratch(self: *MetalBackend, slot_index: usize, len: usize) ![]i32 {
        var scratch = &self.slot_token_scratch[slot_index];
        try scratch.ensureTotalCapacity(self.allocator, len);
        scratch.items.len = len;
        return scratch.items;
    }

    fn ensureSlotCtx(self: *MetalBackend, slot_index: usize) !*mlx_ctx {
        try self.ensureSlotIndex(slot_index);
        if (self.slot_ctxs[slot_index]) |ctx| return ctx;

        var created: ?*mlx_ctx = null;
        if (slot_index > 0) {
            if (self.slot_ctxs[0]) |base_ctx| {
                created = mlx_clone(base_ctx, self.seed);
                if (created == null) {
                    log.warn("inference", "metal mlx_clone failed; falling back to mlx_create", .{
                        .mlx_error = resolveLastError(),
                        .slot = slot_index,
                    });
                }
            }
        }
        if (created == null) {
            created = mlx_create(self.model_id_z.ptr, self.model_path_z.ptr, self.seed);
        }
        const ctx = created orelse {
            log.warn("inference", "metal slot context creation failed", .{
                .mlx_error = resolveLastError(),
                .slot = slot_index,
            });
            return error.InvalidState;
        };
        self.slot_ctxs[slot_index] = ctx;
        return ctx;
    }

    fn ensureSlotStateBlocksBoundForExecution(self: *const MetalBackend, slot_index: usize) !void {
        if (self.state_descriptor_count == 0) return;
        if (slot_index >= self.max_batch_size) return error.InvalidArgument;
        const binding = self.slot_state_bindings[slot_index];
        if (!binding.bound) return error.InvalidStateDescriptorBinding;
        if (@as(usize, @intCast(binding.count)) != @as(usize, self.state_descriptor_count)) {
            return error.InvalidStateDescriptorBinding;
        }
    }

    fn ensureVisionDelegate(self: *MetalBackend) !*cpu_engine.FusedCpuBackend {
        if (self.vision_delegate) |*delegate| return delegate;
        self.vision_delegate = try cpu_engine.FusedCpuBackend.init(
            self.allocator,
            self.loaded,
            self.max_batch_size,
            progress_mod.Context.NONE,
        );
        return &self.vision_delegate.?;
    }

    fn releaseSlotDelegateRouting(self: *MetalBackend, slot_index: usize) void {
        if (slot_index >= self.max_batch_size) return;
        if (self.slot_route_modes[slot_index] != .vision_delegate) return;
        if (self.vision_delegate) |*delegate| {
            if (self.slot_delegate_slots[slot_index]) |delegate_slot| {
                delegate.unbindSlotStateBlocks(delegate_slot);
                delegate.freeSlot(delegate_slot);
            }
        }
        self.slot_delegate_slots[slot_index] = null;
        self.slot_route_modes[slot_index] = .mlx;
    }

    fn prefillFirst(self: *MetalBackend, slot_index: usize, ctx: *mlx_ctx, prompt_tokens: []const u32) !u32 {
        if (prompt_tokens.len == 0) return error.InvalidArgument;
        if (prompt_tokens.len > @as(usize, @intCast(std.math.maxInt(i32)))) return error.InvalidArgument;

        const prompt_i32 = try self.ensureSlotTokenScratch(slot_index, prompt_tokens.len);
        for (prompt_tokens, 0..) |tok, i| {
            if (tok > @as(u32, @intCast(std.math.maxInt(i32)))) return error.InvalidArgument;
            prompt_i32[i] = @intCast(tok);
        }

        var first_token: c_int = 0;
        const status = mlx_prefill_first(
            ctx,
            @ptrCast(prompt_i32.ptr),
            @intCast(prompt_i32.len),
            &first_token,
        );
        if (status == 0 or first_token < 0) {
            log.warn("inference", "metal prefill_first failed", .{
                .mlx_error = resolveLastError(),
                .prompt_len = prompt_tokens.len,
            });
            return error.InvalidArgument;
        }

        return @intCast(first_token);
    }

    fn prefillLogits(
        self: *MetalBackend,
        slot_index: usize,
        ctx: *mlx_ctx,
        prompt_tokens: []const u32,
        logits_out: []f32,
    ) !void {
        if (prompt_tokens.len == 0) return error.InvalidArgument;
        if (prompt_tokens.len > @as(usize, @intCast(std.math.maxInt(i32)))) return error.InvalidArgument;
        if (logits_out.len < self.vocab_size) return error.InvalidArgument;
        if (self.vocab_size > @as(usize, @intCast(std.math.maxInt(i32)))) return error.InvalidArgument;

        const prompt_i32 = try self.ensureSlotTokenScratch(slot_index, prompt_tokens.len);
        for (prompt_tokens, 0..) |tok, i| {
            if (tok > @as(u32, @intCast(std.math.maxInt(i32)))) return error.InvalidArgument;
            prompt_i32[i] = @intCast(tok);
        }

        const status = mlx_prefill_logits(
            ctx,
            @ptrCast(prompt_i32.ptr),
            @intCast(prompt_i32.len),
            @ptrCast(logits_out.ptr),
            @intCast(self.vocab_size),
        );
        if (status == 0) {
            log.warn("inference", "metal prefill_logits failed", .{
                .mlx_error = resolveLastError(),
                .prompt_len = prompt_tokens.len,
            });
            return error.InvalidArgument;
        }
    }

    fn decodeLogits(self: *MetalBackend, ctx: *mlx_ctx, token: u32, logits_out: []f32) !void {
        if (token > @as(u32, @intCast(std.math.maxInt(i32)))) return error.InvalidArgument;
        if (logits_out.len < self.vocab_size) return error.InvalidArgument;
        if (self.vocab_size > @as(usize, @intCast(std.math.maxInt(i32)))) return error.InvalidArgument;

        const status = mlx_decode_logits(
            ctx,
            @intCast(token),
            @ptrCast(logits_out.ptr),
            @intCast(self.vocab_size),
        );
        if (status == 0) {
            log.warn("inference", "metal decode_logits failed", .{
                .mlx_error = resolveLastError(),
                .token = token,
            });
            return error.InvalidArgument;
        }
    }

    fn decodeStream(
        self: *MetalBackend,
        ctx: *mlx_ctx,
        first_token: u32,
        max_new_tokens: usize,
        eos_token_ids: []const u32,
        output_tokens: []u32,
    ) !usize {
        if (max_new_tokens == 0 or output_tokens.len == 0) return 0;
        if (first_token > @as(u32, @intCast(std.math.maxInt(i32)))) return error.InvalidArgument;

        const budget = @min(max_new_tokens, output_tokens.len);
        if (budget > @as(usize, @intCast(std.math.maxInt(i32)))) return error.InvalidArgument;

        var eos_i32_storage: [16]i32 = undefined;
        var eos_i32_owned: ?[]i32 = null;
        defer if (eos_i32_owned) |owned| self.allocator.free(owned);
        var eos_i32: []i32 = undefined;
        if (eos_token_ids.len == 0) {
            eos_i32 = eos_i32_storage[0..0];
        } else if (eos_token_ids.len <= eos_i32_storage.len) {
            eos_i32 = eos_i32_storage[0..eos_token_ids.len];
        } else {
            const owned = try self.allocator.alloc(i32, eos_token_ids.len);
            eos_i32_owned = owned;
            eos_i32 = owned;
        }
        for (eos_token_ids, 0..) |tok, i| {
            if (tok > @as(u32, @intCast(std.math.maxInt(i32)))) return error.InvalidArgument;
            eos_i32[i] = @intCast(tok);
        }

        var generated_ids_ptr: [*c]i32 = null;
        defer mlx_tokens_free(generated_ids_ptr);
        var generated_len: c_int = 0;
        const status = mlx_decode_stream(
            ctx,
            @intCast(first_token),
            @intCast(budget),
            if (eos_i32.len > 0) @ptrCast(eos_i32.ptr) else null,
            @intCast(eos_i32.len),
            &generated_ids_ptr,
            &generated_len,
        );
        if (status == 0 or generated_len < 0) {
            log.warn("inference", "metal decode_stream failed", .{
                .mlx_error = resolveLastError(),
                .max_new_tokens = budget,
            });
            return error.InvalidArgument;
        }
        if (generated_ids_ptr == null and generated_len > 0) return error.InvalidArgument;

        const generated: []const i32 = if (generated_ids_ptr != null and generated_len > 0)
            generated_ids_ptr[0..@intCast(generated_len)]
        else
            &.{};

        const produced = @min(generated.len, budget);
        for (0..produced) |i| {
            const tok_i32 = generated[i];
            if (tok_i32 < 0) return error.InvalidArgument;
            output_tokens[i] = @intCast(tok_i32);
        }
        return produced;
    }

    pub fn init(allocator: std.mem.Allocator, loaded: *LoadedModel, init_config: InitConfig) !MetalBackend {
        const model_path_env_owned = parseOptionalEnvVarOwned(allocator, "TALU_METAL_MODEL_PATH");
        defer if (model_path_env_owned) |buf| allocator.free(buf);
        const model_path_src = blk: {
            if (init_config.model_path) |configured| {
                const trimmed = std.mem.trim(u8, configured, " \t\r\n");
                if (trimmed.len > 0) break :blk trimmed;
            }
            if (model_path_env_owned) |env_buf| {
                const trimmed = std.mem.trim(u8, env_buf, " \t\r\n");
                if (trimmed.len > 0) break :blk trimmed;
            }
            return error.InvalidArgument;
        };

        const model_id_env_owned = parseOptionalEnvVarOwned(allocator, "TALU_METAL_MODEL_ID");
        defer if (model_id_env_owned) |buf| allocator.free(buf);
        const model_id_src = blk: {
            if (init_config.model_id) |configured| {
                const trimmed = std.mem.trim(u8, configured, " \t\r\n");
                if (trimmed.len > 0) break :blk trimmed;
            }
            if (model_id_env_owned) |env_buf| {
                const trimmed = std.mem.trim(u8, env_buf, " \t\r\n");
                if (trimmed.len > 0) break :blk trimmed;
            }
            break :blk "metal";
        };

        const model_path_z = try allocator.dupeZ(u8, model_path_src);
        errdefer allocator.free(model_path_z);

        const model_id_z = try allocator.dupeZ(u8, model_id_src);
        errdefer allocator.free(model_id_z);

        const seed = readSeedFromEnv(allocator);
        const max_batch_size = resolveMaxBatchSize(allocator);
        if (mlx_validate_config(model_path_z.ptr) == 0) {
            const mlx_error = resolveLastError();
            if (isMemoryError(mlx_error)) {
                if (init_config.memory_fit_is_error) {
                    log.err("inference", "metal mlx_validate_config failed", .{ .mlx_error = mlx_error }, @src());
                } else {
                    log.warn("inference", "metal mlx_validate_config failed", .{ .mlx_error = mlx_error });
                }
                return error.OutOfMemory;
            }
            log.warn("inference", "metal mlx_validate_config failed", .{ .mlx_error = mlx_error });
            return error.InvalidArgument;
        }
        const ctx = mlx_create(model_id_z.ptr, model_path_z.ptr, seed) orelse {
            const mlx_error = resolveLastError();
            if (isMemoryError(mlx_error)) {
                if (init_config.memory_fit_is_error) {
                    log.err("inference", "metal mlx_create failed", .{ .mlx_error = mlx_error }, @src());
                } else {
                    log.warn("inference", "metal mlx_create failed", .{ .mlx_error = mlx_error });
                }
                return error.OutOfMemory;
            }
            log.warn("inference", "metal mlx_create failed", .{ .mlx_error = mlx_error });
            return error.InvalidArgument;
        };

        const vocab_size: usize = @intCast(loaded.config.vocab_size);
        const d_model = std.math.cast(usize, loaded.config.d_model) orelse return error.InvalidArgument;
        const slot_ctxs = try allocator.alloc(?*mlx_ctx, max_batch_size);
        errdefer allocator.free(slot_ctxs);
        @memset(slot_ctxs, null);
        slot_ctxs[0] = ctx;

        const slot_in_use = try allocator.alloc(bool, max_batch_size);
        errdefer allocator.free(slot_in_use);
        @memset(slot_in_use, false);

        const slot_positions = try allocator.alloc(usize, max_batch_size);
        errdefer allocator.free(slot_positions);
        @memset(slot_positions, 0);

        const slot_logits_buffers = try allocator.alloc(f32, max_batch_size * vocab_size);
        errdefer allocator.free(slot_logits_buffers);
        const slot_token_scratch = try allocator.alloc(std.ArrayListUnmanaged(i32), max_batch_size);
        errdefer allocator.free(slot_token_scratch);
        for (slot_token_scratch) |*scratch| scratch.* = .{};
        const topk_scratch_capacity = @min(vocab_size, topk_route_candidate_capacity);
        const slot_topk_ids_i32 = try allocator.alloc(i32, max_batch_size * topk_scratch_capacity);
        errdefer allocator.free(slot_topk_ids_i32);
        const batch_decode_ctxs = try allocator.alloc(*mlx_ctx, max_batch_size);
        errdefer allocator.free(batch_decode_ctxs);
        const batch_decode_tokens_i32 = try allocator.alloc(i32, max_batch_size);
        errdefer allocator.free(batch_decode_tokens_i32);
        const batch_decode_logits_ptrs = try allocator.alloc([*]f32, max_batch_size);
        errdefer allocator.free(batch_decode_logits_ptrs);
        const batch_prefill_ctxs = try allocator.alloc(*mlx_ctx, max_batch_size);
        errdefer allocator.free(batch_prefill_ctxs);
        const batch_prefill_prompt_ptrs = try allocator.alloc([*]const i32, max_batch_size);
        errdefer allocator.free(batch_prefill_prompt_ptrs);
        const batch_prefill_prompt_lens_i32 = try allocator.alloc(i32, max_batch_size);
        errdefer allocator.free(batch_prefill_prompt_lens_i32);
        const batch_prefill_logits_ptrs = try allocator.alloc([*]f32, max_batch_size);
        errdefer allocator.free(batch_prefill_logits_ptrs);

        const slot_state_bindings = try allocator.alloc(SlotStateBinding, max_batch_size);
        errdefer allocator.free(slot_state_bindings);
        for (slot_state_bindings) |*binding| binding.* = .{};

        const slot_route_modes = try allocator.alloc(SlotRouteMode, max_batch_size);
        errdefer allocator.free(slot_route_modes);
        @memset(slot_route_modes, .mlx);

        const slot_delegate_slots = try allocator.alloc(?usize, max_batch_size);
        errdefer allocator.free(slot_delegate_slots);
        @memset(slot_delegate_slots, null);

        var backend = MetalBackend{
            .allocator = allocator,
            .loaded = loaded,
            .model_id_z = model_id_z,
            .model_path_z = model_path_z,
            .seed = seed,
            .vocab_size = vocab_size,
            .d_model = d_model,
            .max_batch_size = max_batch_size,
            .slot_ctxs = slot_ctxs,
            .slot_in_use = slot_in_use,
            .slot_positions = slot_positions,
            .slot_logits_buffers = slot_logits_buffers,
            .slot_token_scratch = slot_token_scratch,
            .slot_topk_ids_i32 = slot_topk_ids_i32,
            .topk_scratch_capacity = topk_scratch_capacity,
            .batch_decode_ctxs = batch_decode_ctxs,
            .batch_decode_tokens_i32 = batch_decode_tokens_i32,
            .batch_decode_logits_ptrs = batch_decode_logits_ptrs,
            .batch_prefill_ctxs = batch_prefill_ctxs,
            .batch_prefill_prompt_ptrs = batch_prefill_prompt_ptrs,
            .batch_prefill_prompt_lens_i32 = batch_prefill_prompt_lens_i32,
            .batch_prefill_logits_ptrs = batch_prefill_logits_ptrs,
            .slot_state_bindings = slot_state_bindings,
            .slot_route_modes = slot_route_modes,
            .slot_delegate_slots = slot_delegate_slots,
        };
        errdefer backend.deinit();
        try backend.collectPlanStateDescriptors();
        return backend;
    }

    pub fn deinit(self: *MetalBackend) void {
        if (self.vision_delegate) |*delegate| {
            delegate.deinit();
            self.vision_delegate = null;
        }
        for (self.slot_ctxs) |ctx| {
            if (ctx != null) mlx_destroy(ctx);
        }
        self.allocator.free(self.slot_delegate_slots);
        self.allocator.free(self.slot_route_modes);
        self.allocator.free(self.slot_state_bindings);
        for (self.slot_token_scratch) |*scratch| {
            scratch.deinit(self.allocator);
        }
        self.allocator.free(self.slot_token_scratch);
        self.allocator.free(self.slot_topk_ids_i32);
        self.allocator.free(self.batch_prefill_logits_ptrs);
        self.allocator.free(self.batch_prefill_prompt_lens_i32);
        self.allocator.free(self.batch_prefill_prompt_ptrs);
        self.allocator.free(self.batch_prefill_ctxs);
        self.allocator.free(self.batch_decode_logits_ptrs);
        self.allocator.free(self.batch_decode_tokens_i32);
        self.allocator.free(self.batch_decode_ctxs);
        self.allocator.free(self.slot_logits_buffers);
        self.allocator.free(self.slot_positions);
        self.allocator.free(self.slot_in_use);
        self.allocator.free(self.slot_ctxs);
        self.allocator.free(self.model_id_z);
        self.allocator.free(self.model_path_z);
    }

    pub fn synchronize(self: *MetalBackend) void {
        _ = self;
    }

    pub fn cleanupExecutionThreadState(self: *MetalBackend) void {
        _ = self;
    }

    pub fn teardownExecutionThreadState(self: *MetalBackend) void {
        _ = self;
    }

    pub fn prefill(self: *MetalBackend, tokens: []const u32, logits_out: []f32) !void {
        self.slot_in_use[0] = true;
        self.releaseSlotDelegateRouting(0);
        return self.prefillSlot(0, tokens, logits_out);
    }

    pub fn decode(self: *MetalBackend, token: u32, position: usize, logits_out: []f32) !void {
        if (!self.slot_in_use[0]) return error.InvalidArgument;
        if (logits_out.len < self.vocab_size) return error.InvalidArgument;
        try self.ensureSlotStateBlocksBoundForExecution(0);
        if (self.slot_route_modes[0] == .vision_delegate) {
            const delegate = if (self.vision_delegate) |*d| d else return error.InvalidState;
            const delegate_slot = self.slot_delegate_slots[0] orelse return error.InvalidState;
            var delegate_requests = [_]contract.DecodeRequest{.{
                .slot_index = delegate_slot,
                .token = token,
            }};
            var delegate_results = [_]contract.DecodeResult{undefined};
            try delegate.decodeBatch(delegate_requests[0..], delegate_results[0..]);
            const delegate_logits = delegate_results[0].logits;
            if (delegate_logits.len < self.vocab_size) return error.InvalidState;
            const slot_logits = self.slotLogitsSlice(0);
            @memcpy(slot_logits, delegate_logits[0..self.vocab_size]);
            @memcpy(logits_out[0..self.vocab_size], slot_logits);
            self.slot_positions[0] = position + 1;
            return;
        }
        const ctx = try self.ensureSlotCtx(0);
        const logits_view = logits_out[0..self.vocab_size];
        try self.decodeLogits(ctx, token, logits_view);
        const slot_logits = self.slotLogitsSlice(0);
        if (logits_view.ptr != slot_logits.ptr) {
            @memcpy(slot_logits, logits_view);
        }
        self.slot_positions[0] = position + 1;
    }

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
        if (!self.slot_in_use[0]) return error.InvalidArgument;
        try self.ensureSlotStateBlocksBoundForExecution(0);
        const budget = @min(max_tokens, output_tokens.len);
        if (self.slot_route_modes[0] == .vision_delegate) {
            const delegate = if (self.vision_delegate) |*d| d else return error.InvalidState;
            const delegate_slot = self.slot_delegate_slots[0] orelse return error.InvalidState;
            var produced: usize = 0;
            if (delegate_slot == 0) {
                produced = try delegate.decodeStreaming(
                    first_token,
                    start_position,
                    budget,
                    eos_token_ids,
                    output_tokens,
                    callback,
                    callback_data,
                );
            } else {
                var current_token = first_token;
                while (produced < budget) : (produced += 1) {
                    var delegate_requests = [_]contract.DecodeRequest{.{
                        .slot_index = delegate_slot,
                        .token = current_token,
                    }};
                    var delegate_results = [_]contract.DecodeResult{undefined};
                    try delegate.decodeBatch(delegate_requests[0..], delegate_results[0..]);
                    const logits = delegate_results[0].logits;
                    if (logits.len == 0) return error.InvalidState;
                    var best_idx: usize = 0;
                    var best_val: f32 = -std.math.inf(f32);
                    for (logits, 0..) |value, idx| {
                        if (value > best_val) {
                            best_val = value;
                            best_idx = idx;
                        }
                    }
                    current_token = @intCast(best_idx);
                    output_tokens[produced] = current_token;
                    if (callback) |cb| cb(current_token, callback_data);
                    for (eos_token_ids) |eos_id| {
                        if (current_token == eos_id) {
                            produced += 1;
                            self.slot_positions[0] = start_position + produced;
                            return produced;
                        }
                    }
                }
            }
            self.slot_positions[0] = start_position + produced;
            return produced;
        }
        const ctx = try self.ensureSlotCtx(0);

        const produced = try self.decodeStream(ctx, first_token, budget, eos_token_ids, output_tokens);
        if (callback) |cb| {
            for (output_tokens[0..produced]) |token| cb(token, callback_data);
        }

        self.slot_positions[0] = start_position + produced;
        return produced;
    }

    pub fn supportsSchedulerBackendDecodeStreamingRoute(self: *const MetalBackend) bool {
        _ = self;
        return true;
    }

    pub fn supportsSchedulerBackendTopKDecodeRoute(self: *const MetalBackend, sampling_config: *const sampling.SamplingConfig) bool {
        _ = self;
        _ = sampling_config;
        // Intentionally disabled: enabling backend top-k route regressed decode
        // throughput in A/B benchmarking versus decodeBatch+host sampling.
        // Keep a single default route for now; revisit only with a proven
        // non-regressing implementation.
        return false;
    }

    pub fn supportsSchedulerBackendTopKCandidateSamplingRoute(self: *const MetalBackend, sampling_config: *const sampling.SamplingConfig) bool {
        _ = self;
        _ = sampling_config;
        // Must stay aligned with supportsSchedulerBackendTopKDecodeRoute.
        return false;
    }

    pub fn supportsSchedulerBackendInPlaceSamplingMutation(self: *const MetalBackend) bool {
        _ = self;
        return true;
    }

    pub fn decodeTopKCandidates(
        self: *MetalBackend,
        slot_index: usize,
        token: u32,
        top_k: usize,
        candidate_logits_out: []f32,
        candidate_ids_out: []u32,
    ) !usize {
        try self.ensureSlotIndex(slot_index);
        if (!self.slot_in_use[slot_index]) return error.InvalidArgument;
        try self.ensureSlotStateBlocksBoundForExecution(slot_index);
        if (top_k == 0) return error.InvalidArgument;
        if (top_k > self.vocab_size) return error.InvalidArgument;
        if (candidate_logits_out.len < top_k) return error.InvalidArgument;
        if (candidate_ids_out.len < top_k) return error.InvalidArgument;
        if (token > @as(u32, @intCast(std.math.maxInt(i32)))) return error.InvalidArgument;
        if (top_k > @as(usize, @intCast(std.math.maxInt(i32)))) return error.InvalidArgument;
        if (self.slot_route_modes[slot_index] == .vision_delegate) {
            const delegate = if (self.vision_delegate) |*d| d else return error.InvalidState;
            const delegate_slot = self.slot_delegate_slots[slot_index] orelse return error.InvalidState;
            var delegate_requests = [_]contract.DecodeRequest{.{
                .slot_index = delegate_slot,
                .token = token,
            }};
            var delegate_results = [_]contract.DecodeResult{undefined};
            try delegate.decodeBatch(delegate_requests[0..], delegate_results[0..]);
            const delegate_logits = delegate_results[0].logits;
            if (delegate_logits.len < self.vocab_size) return error.InvalidState;
            const slot_logits = self.slotLogitsSlice(slot_index);
            @memcpy(slot_logits, delegate_logits[0..self.vocab_size]);
            const candidate_count = fillTopKFromLogits(
                slot_logits[0..self.vocab_size],
                top_k,
                candidate_logits_out,
                candidate_ids_out,
            );
            self.slot_positions[slot_index] += 1;
            return candidate_count;
        }

        const ctx = try self.ensureSlotCtx(slot_index);
        var candidate_ids_owned: ?[]i32 = null;
        defer if (candidate_ids_owned) |owned| self.allocator.free(owned);
        const candidate_ids_i32 = if (top_k <= self.topk_scratch_capacity)
            self.slotTopKIdsSlice(slot_index)[0..top_k]
        else blk: {
            const owned = try self.allocator.alloc(i32, top_k);
            candidate_ids_owned = owned;
            break :blk owned;
        };
        var candidate_count_i32: c_int = 0;
        const status = mlx_decode_topk_candidates(
            ctx,
            @intCast(token),
            @intCast(top_k),
            @ptrCast(candidate_logits_out.ptr),
            @ptrCast(candidate_ids_i32.ptr),
            &candidate_count_i32,
        );
        if (status == 0 or candidate_count_i32 <= 0) {
            log.warn("inference", "metal decode_topk_candidates failed", .{
                .mlx_error = resolveLastError(),
                .slot = slot_index,
                .token = token,
                .top_k = top_k,
            });
            return error.InvalidArgument;
        }
        const candidate_count: usize = @intCast(candidate_count_i32);
        if (candidate_count > top_k) return error.InvalidState;
        for (candidate_ids_i32[0..candidate_count], 0..) |candidate_id_i32, idx| {
            if (candidate_id_i32 < 0) return error.InvalidState;
            candidate_ids_out[idx] = @intCast(candidate_id_i32);
        }
        self.slot_positions[slot_index] += 1;
        return candidate_count;
    }

    pub fn decodeTopKCandidatesWithSampling(
        self: *MetalBackend,
        slot_index: usize,
        token: u32,
        sampling_config: *const sampling.SamplingConfig,
        candidate_logits_out: []f32,
        candidate_ids_out: []u32,
    ) !usize {
        try self.ensureSlotIndex(slot_index);
        if (!self.slot_in_use[slot_index]) return error.InvalidArgument;
        try self.ensureSlotStateBlocksBoundForExecution(slot_index);
        if (sampling_config.strategy != .top_k) return error.InvalidArgument;
        if (sampling_config.top_k == 0) return error.InvalidArgument;
        if (sampling_config.top_k > self.vocab_size) return error.InvalidArgument;
        if (candidate_logits_out.len < sampling_config.top_k) return error.InvalidArgument;
        if (candidate_ids_out.len < sampling_config.top_k) return error.InvalidArgument;
        if (token > @as(u32, @intCast(std.math.maxInt(i32)))) return error.InvalidArgument;
        if (sampling_config.top_k > @as(usize, @intCast(std.math.maxInt(i32)))) return error.InvalidArgument;
        if (sampling_config.logit_bias != null) return error.InvalidArgument;
        if (self.slot_route_modes[slot_index] == .vision_delegate) {
            const delegate = if (self.vision_delegate) |*d| d else return error.InvalidState;
            const delegate_slot = self.slot_delegate_slots[slot_index] orelse return error.InvalidState;
            var delegate_requests = [_]contract.DecodeRequest{.{
                .slot_index = delegate_slot,
                .token = token,
            }};
            var delegate_results = [_]contract.DecodeResult{undefined};
            try delegate.decodeBatch(delegate_requests[0..], delegate_results[0..]);
            const delegate_logits = delegate_results[0].logits;
            if (delegate_logits.len < self.vocab_size) return error.InvalidState;
            const slot_logits = self.slotLogitsSlice(slot_index);
            @memcpy(slot_logits, delegate_logits[0..self.vocab_size]);
            applySamplingMutationsToLogits(slot_logits, sampling_config);

            const candidate_count = fillTopKFromLogits(
                slot_logits[0..self.vocab_size],
                sampling_config.top_k,
                candidate_logits_out,
                candidate_ids_out,
            );
            self.slot_positions[slot_index] += 1;
            return candidate_count;
        }

        const ctx = try self.ensureSlotCtx(slot_index);
        const context_tokens = sampling_config.context_tokens orelse &.{};
        if (context_tokens.len > @as(usize, @intCast(std.math.maxInt(i32)))) return error.InvalidArgument;
        const context_ids_i32 = try self.ensureSlotTokenScratch(slot_index, context_tokens.len);
        for (context_tokens, 0..) |context_token, idx| {
            if (context_token > @as(u32, @intCast(std.math.maxInt(i32)))) return error.InvalidArgument;
            context_ids_i32[idx] = @intCast(context_token);
        }

        var candidate_ids_owned: ?[]i32 = null;
        defer if (candidate_ids_owned) |owned| self.allocator.free(owned);
        const candidate_ids_i32 = if (sampling_config.top_k <= self.topk_scratch_capacity)
            self.slotTopKIdsSlice(slot_index)[0..sampling_config.top_k]
        else blk: {
            const owned = try self.allocator.alloc(i32, sampling_config.top_k);
            candidate_ids_owned = owned;
            break :blk owned;
        };

        var candidate_count_i32: c_int = 0;
        const status = mlx_decode_topk_candidates_with_sampling(
            ctx,
            @intCast(token),
            @intCast(sampling_config.top_k),
            sampling_config.repetition_penalty,
            sampling_config.presence_penalty,
            sampling_config.frequency_penalty,
            if (context_ids_i32.len > 0) @ptrCast(context_ids_i32.ptr) else null,
            @intCast(context_ids_i32.len),
            @ptrCast(candidate_logits_out.ptr),
            @ptrCast(candidate_ids_i32.ptr),
            &candidate_count_i32,
        );
        if (status == 0 or candidate_count_i32 <= 0) {
            log.warn("inference", "metal decode_topk_candidates_with_sampling failed", .{
                .mlx_error = resolveLastError(),
                .slot = slot_index,
                .token = token,
                .top_k = sampling_config.top_k,
                .repetition_penalty = sampling_config.repetition_penalty,
                .presence_penalty = sampling_config.presence_penalty,
                .frequency_penalty = sampling_config.frequency_penalty,
                .context_len = context_tokens.len,
            });
            return error.InvalidArgument;
        }

        const candidate_count: usize = @intCast(candidate_count_i32);
        if (candidate_count > sampling_config.top_k) return error.InvalidState;
        for (candidate_ids_i32[0..candidate_count], 0..) |candidate_id_i32, idx| {
            if (candidate_id_i32 < 0) return error.InvalidState;
            candidate_ids_out[idx] = @intCast(candidate_id_i32);
        }
        self.slot_positions[slot_index] += 1;
        return candidate_count;
    }

    pub fn maxBatchSize(self: *const MetalBackend) usize {
        return self.max_batch_size;
    }

    pub fn vocabSize(self: *const MetalBackend) usize {
        return self.vocab_size;
    }

    pub fn warmup(self: *MetalBackend) !void {
        _ = self;
    }

    pub fn embed(
        self: *MetalBackend,
        tokens: []const u32,
        pooling: contract.PoolingStrategy,
        normalize: bool,
        embedding_buffer: []f32,
    ) !void {
        if (tokens.len == 0) return error.EmptyInput;
        if (embedding_buffer.len < self.d_model) return error.BufferTooSmall;
        if (tokens.len > @as(usize, @intCast(std.math.maxInt(i32)))) return error.InvalidArgument;
        if (self.d_model > @as(usize, @intCast(std.math.maxInt(i32)))) return error.InvalidArgument;

        const ctx = try self.ensureSlotCtx(0);
        const token_ids_i32 = try self.allocator.alloc(i32, tokens.len);
        defer self.allocator.free(token_ids_i32);
        for (tokens, 0..) |tok, i| {
            if (tok > @as(u32, @intCast(std.math.maxInt(i32)))) return error.InvalidArgument;
            token_ids_i32[i] = @intCast(tok);
        }

        const status = mlx_embed(
            ctx,
            @ptrCast(token_ids_i32.ptr),
            @intCast(token_ids_i32.len),
            @intFromEnum(pooling),
            if (normalize) 1 else 0,
            @ptrCast(embedding_buffer.ptr),
            @intCast(self.d_model),
        );
        if (status == 0) {
            log.warn("inference", "metal mlx_embed failed", .{
                .mlx_error = resolveLastError(),
                .token_len = tokens.len,
                .normalize = @as(u8, @intFromBool(normalize)),
                .pooling = @tagName(pooling),
            });
            return error.InvalidArgument;
        }
    }

    pub fn embeddingDim(self: *const MetalBackend) usize {
        return self.d_model;
    }

    pub fn allocSlot(self: *MetalBackend) ?usize {
        for (self.slot_in_use, 0..) |used, slot_index| {
            if (used) continue;
            self.slot_in_use[slot_index] = true;
            return slot_index;
        }
        return null;
    }

    pub fn freeSlot(self: *MetalBackend, slot_index: usize) void {
        self.ensureSlotIndex(slot_index) catch return;
        self.releaseSlotDelegateRouting(slot_index);
        self.unbindSlotStateBlocks(slot_index);
        self.slot_in_use[slot_index] = false;
        self.clearSlotState(slot_index);
        // Keep slot contexts warm across request lifecycles. Destroying and
        // recreating non-zero slot contexts adds avoidable fixed latency per
        // request and defeats continuous-batching admission at steady state.
        self.resetCtx(slot_index);
    }

    pub fn resetSlot(self: *MetalBackend, slot_index: usize) void {
        self.ensureSlotIndex(slot_index) catch return;
        if (self.slot_route_modes[slot_index] == .vision_delegate) {
            if (self.vision_delegate) |*delegate| {
                if (self.slot_delegate_slots[slot_index]) |delegate_slot| {
                    delegate.resetSlot(delegate_slot);
                }
            }
        } else {
            self.resetCtx(slot_index);
        }
        self.clearSlotState(slot_index);
    }

    pub fn getPosition(self: *const MetalBackend, slot_index: usize) usize {
        if (slot_index >= self.max_batch_size) return 0;
        if (self.slot_route_modes[slot_index] == .vision_delegate) {
            if (self.vision_delegate) |*delegate| {
                if (self.slot_delegate_slots[slot_index]) |delegate_slot| {
                    return delegate.getPosition(delegate_slot);
                }
            }
        }
        return self.slot_positions[slot_index];
    }

    pub fn stateDescriptors(self: *const MetalBackend) []const runtime_contract.StateDescriptor {
        return self.state_descriptors_storage[0..self.state_descriptor_count];
    }

    pub fn bindSlotStateBlocks(
        self: *MetalBackend,
        slot_index: usize,
        state_blocks: []const runtime_contract.StateBlockHandle,
    ) !void {
        try self.ensureSlotIndex(slot_index);
        if (!self.slot_in_use[slot_index]) return error.InvalidArgument;

        const descriptors = self.stateDescriptors();
        runtime_contract.validateStateBlocksForDescriptors(descriptors, state_blocks) catch |err| {
            log.warn("inference", "metal bindSlotStateBlocks descriptor validation failed", .{
                .reason = @errorName(err),
                .slot = slot_index,
                .descriptors = descriptors.len,
                .state_blocks = state_blocks.len,
                .desc0 = if (descriptors.len > 0) descriptors[0].id else @as(u8, 0),
                .block0 = if (state_blocks.len > 0) state_blocks[0].id else @as(u8, 0),
            });
            return err;
        };
        if (state_blocks.len != descriptors.len) return error.InvalidStateDescriptorBinding;
        const binding = &self.slot_state_bindings[slot_index];
        if (state_blocks.len > binding.handles.len) return error.InvalidStateDescriptorBinding;

        for (descriptors, 0..) |descriptor, idx| {
            const incoming = runtime_contract.findStateBlock(state_blocks, descriptor.id) orelse {
                return error.InvalidStateDescriptorBinding;
            };
            binding.handles[idx] = .{
                .id = incoming.id,
                .ptr = incoming.ptr,
                .size = incoming.size,
                .align_bytes = incoming.align_bytes,
            };
        }

        binding.count = @intCast(state_blocks.len);
        binding.bound = true;
        if (self.slot_route_modes[slot_index] == .vision_delegate) {
            const delegate = if (self.vision_delegate) |*d| d else return error.InvalidState;
            const delegate_slot = self.slot_delegate_slots[slot_index] orelse return error.InvalidState;
            try delegate.bindSlotStateBlocks(delegate_slot, state_blocks);
        }
    }

    pub fn unbindSlotStateBlocks(self: *MetalBackend, slot_index: usize) void {
        if (slot_index >= self.max_batch_size) return;
        if (self.slot_route_modes[slot_index] == .vision_delegate) {
            if (self.vision_delegate) |*delegate| {
                if (self.slot_delegate_slots[slot_index]) |delegate_slot| {
                    delegate.unbindSlotStateBlocks(delegate_slot);
                }
            }
        }
        self.slot_state_bindings[slot_index].clear();
    }

    pub fn prefillSlot(self: *MetalBackend, slot_index: usize, tokens: []const u32, logits_out: []f32) !void {
        try self.ensureSlotIndex(slot_index);
        if (!self.slot_in_use[slot_index]) return error.InvalidArgument;
        try self.ensureSlotStateBlocksBoundForExecution(slot_index);
        if (tokens.len == 0) return error.InvalidArgument;
        if (logits_out.len < self.vocab_size) return error.InvalidArgument;
        self.releaseSlotDelegateRouting(slot_index);
        const ctx = try self.ensureSlotCtx(slot_index);
        self.clearSlotState(slot_index);
        const logits_view = logits_out[0..self.vocab_size];
        try self.prefillLogits(slot_index, ctx, tokens, logits_view);
        const slot_logits = self.slotLogitsSlice(slot_index);
        if (logits_view.ptr != slot_logits.ptr) {
            @memcpy(slot_logits, logits_view);
        }
        self.slot_positions[slot_index] = tokens.len;
    }

    pub fn prefillBatch(self: *MetalBackend, requests: []const contract.PrefillBatchRequest) !void {
        try runtime_contract.validateBatchCapability(.{
            .supports_batch = true,
            .supports_graph_emit = false,
            .max_batch_size = self.max_batch_size,
        }, requests.len);
        if (requests.len == 0) return;
        if (self.vocab_size > @as(usize, @intCast(std.math.maxInt(i32)))) return error.InvalidArgument;

        var batch_prefill_count: usize = 0;
        for (requests, 0..) |request_entry, idx| {
            try self.ensureSlotIndex(request_entry.slot_index);
            if (!self.slot_in_use[request_entry.slot_index]) return error.InvalidArgument;
            try self.ensureSlotStateBlocksBoundForExecution(request_entry.slot_index);
            if (request_entry.prompt_tokens.len == 0) return error.InvalidArgument;
            if (request_entry.prompt_tokens.len > @as(usize, @intCast(std.math.maxInt(i32)))) return error.InvalidArgument;
            if (request_entry.logits_out.len < self.vocab_size) return error.InvalidArgument;
            for (requests[0..idx]) |prev| {
                if (prev.slot_index == request_entry.slot_index) return error.InvalidBatchSize;
            }
            if (self.slot_route_modes[request_entry.slot_index] == .vision_delegate) {
                return error.UnsupportedContentType;
            }

            self.releaseSlotDelegateRouting(request_entry.slot_index);
            const ctx = try self.ensureSlotCtx(request_entry.slot_index);
            self.clearSlotState(request_entry.slot_index);

            const prompt_i32 = try self.ensureSlotTokenScratch(request_entry.slot_index, request_entry.prompt_tokens.len);
            for (request_entry.prompt_tokens, 0..) |token, token_idx| {
                if (token > @as(u32, @intCast(std.math.maxInt(i32)))) return error.InvalidArgument;
                prompt_i32[token_idx] = @intCast(token);
            }

            self.batch_prefill_ctxs[batch_prefill_count] = ctx;
            self.batch_prefill_prompt_ptrs[batch_prefill_count] = prompt_i32.ptr;
            self.batch_prefill_prompt_lens_i32[batch_prefill_count] = @intCast(prompt_i32.len);
            self.batch_prefill_logits_ptrs[batch_prefill_count] = request_entry.logits_out.ptr;
            batch_prefill_count += 1;
        }

        const status = mlx_prefill_logits_batch(
            self.batch_prefill_ctxs.ptr,
            self.batch_prefill_prompt_ptrs.ptr,
            self.batch_prefill_prompt_lens_i32.ptr,
            self.batch_prefill_logits_ptrs.ptr,
            @intCast(batch_prefill_count),
            @intCast(self.vocab_size),
        );
        if (status == 0) {
            log.warn("inference", "metal prefill_logits_batch failed", .{
                .mlx_error = resolveLastError(),
                .batch = batch_prefill_count,
            });
            return error.InvalidArgument;
        }

        for (requests) |request_entry| {
            const slot_logits = self.slotLogitsSlice(request_entry.slot_index);
            const logits_view = request_entry.logits_out[0..self.vocab_size];
            if (slot_logits.ptr != logits_view.ptr) {
                @memcpy(slot_logits, logits_view);
            }
            self.slot_positions[request_entry.slot_index] = request_entry.prompt_tokens.len;
        }
    }

    pub fn prefillGreedySeedToken(self: *MetalBackend, slot_index: usize, tokens: []const u32) !u32 {
        try self.ensureSlotIndex(slot_index);
        if (!self.slot_in_use[slot_index]) return error.InvalidArgument;
        try self.ensureSlotStateBlocksBoundForExecution(slot_index);
        if (tokens.len == 0) return error.InvalidArgument;

        self.releaseSlotDelegateRouting(slot_index);
        const ctx = try self.ensureSlotCtx(slot_index);
        self.clearSlotState(slot_index);
        const first_token = try self.prefillFirst(slot_index, ctx, tokens);
        self.slot_positions[slot_index] = tokens.len;
        return first_token;
    }

    pub fn prefillSlotWithVision(
        self: *MetalBackend,
        slot_index: usize,
        tokens: []const u32,
        vision_input: ?*const PrefillVisionInput,
        logits_out: []f32,
    ) !void {
        if (vision_input == null) return self.prefillSlot(slot_index, tokens, logits_out);

        try self.ensureSlotIndex(slot_index);
        if (!self.slot_in_use[slot_index]) return error.InvalidArgument;
        try self.ensureSlotStateBlocksBoundForExecution(slot_index);
        if (tokens.len == 0) return error.InvalidArgument;
        if (logits_out.len < self.vocab_size) return error.InvalidArgument;

        var delegate = try self.ensureVisionDelegate();
        var delegate_slot = self.slot_delegate_slots[slot_index];
        var allocated_delegate_slot = false;
        if (delegate_slot == null) {
            delegate_slot = delegate.allocSlot() orelse return error.NoSlotsAvailable;
            self.slot_delegate_slots[slot_index] = delegate_slot;
            allocated_delegate_slot = true;
        }
        errdefer if (allocated_delegate_slot) {
            if (delegate_slot) |slot| delegate.freeSlot(slot);
            self.slot_delegate_slots[slot_index] = null;
        };
        const previous_route_mode = self.slot_route_modes[slot_index];
        errdefer self.slot_route_modes[slot_index] = previous_route_mode;

        const binding = &self.slot_state_bindings[slot_index];
        if (binding.bound) {
            const count: usize = @intCast(binding.count);
            try delegate.bindSlotStateBlocks(delegate_slot.?, binding.handles[0..count]);
        }

        try delegate.prefillSlotWithVision(
            delegate_slot.?,
            tokens,
            vision_input,
            logits_out[0..self.vocab_size],
        );
        self.slot_route_modes[slot_index] = .vision_delegate;
        const slot_logits = self.slotLogitsSlice(slot_index);
        if (slot_logits.ptr != logits_out.ptr) {
            @memcpy(slot_logits, logits_out[0..self.vocab_size]);
        }
        self.slot_positions[slot_index] = tokens.len;
    }

    pub fn decodeBatch(self: *MetalBackend, requests: []const contract.DecodeRequest, results: []contract.DecodeResult) !void {
        try runtime_contract.validateBatchCapability(.{
            .supports_batch = true,
            .supports_graph_emit = false,
            .max_batch_size = self.max_batch_size,
        }, requests.len);
        if (results.len < requests.len) return error.InvalidArgument;

        var batch_decode_count: usize = 0;
        for (requests, 0..) |request, idx| {
            try self.ensureSlotIndex(request.slot_index);
            if (!self.slot_in_use[request.slot_index]) return error.InvalidArgument;
            try self.ensureSlotStateBlocksBoundForExecution(request.slot_index);
            if (request.token > @as(u32, @intCast(std.math.maxInt(i32)))) return error.InvalidArgument;
            for (requests[0..idx]) |prev| {
                if (prev.slot_index == request.slot_index) return error.InvalidBatchSize;
            }
            if (self.slot_route_modes[request.slot_index] == .vision_delegate) continue;
            const ctx = try self.ensureSlotCtx(request.slot_index);
            self.batch_decode_ctxs[batch_decode_count] = ctx;
            self.batch_decode_tokens_i32[batch_decode_count] = @intCast(request.token);
            self.batch_decode_logits_ptrs[batch_decode_count] = self.slotLogitsSlice(request.slot_index).ptr;
            batch_decode_count += 1;
        }

        if (batch_decode_count > 0) {
            const status = mlx_decode_logits_batch(
                self.batch_decode_ctxs.ptr,
                self.batch_decode_tokens_i32.ptr,
                self.batch_decode_logits_ptrs.ptr,
                @intCast(batch_decode_count),
                @intCast(self.vocab_size),
            );
            if (status == 0) {
                log.warn("inference", "metal decode_logits_batch failed", .{
                    .mlx_error = resolveLastError(),
                    .batch = batch_decode_count,
                });
                return error.InvalidArgument;
            }
        }

        for (requests, 0..) |request, idx| {
            const slot_logits = self.slotLogitsSlice(request.slot_index);
            if (self.slot_route_modes[request.slot_index] == .vision_delegate) {
                const delegate = if (self.vision_delegate) |*d| d else return error.InvalidState;
                const delegate_slot = self.slot_delegate_slots[request.slot_index] orelse return error.InvalidState;
                var delegate_requests = [_]contract.DecodeRequest{.{
                    .slot_index = delegate_slot,
                    .token = request.token,
                }};
                var delegate_results = [_]contract.DecodeResult{undefined};
                try delegate.decodeBatch(delegate_requests[0..], delegate_results[0..]);
                const delegate_logits = delegate_results[0].logits;
                if (delegate_logits.len < self.vocab_size) return error.InvalidState;
                @memcpy(slot_logits, delegate_logits[0..self.vocab_size]);
            }
            self.slot_positions[request.slot_index] += 1;
            results[idx] = .{
                .slot_index = request.slot_index,
                .logits = slot_logits,
            };
        }
    }

    pub fn isAvailable() bool {
        return mlx_is_available() != 0;
    }
};

test "MetalBackend.applySamplingMutationsToLogits applies repetition penalty" {
    var logits = [_]f32{ 2.0, -2.0, 1.0, 0.5 };
    const context_tokens = [_]u32{ 0, 1 };
    const cfg = sampling.SamplingConfig{
        .strategy = .top_k,
        .top_k = 2,
        .repetition_penalty = 2.0,
        .context_tokens = context_tokens[0..],
    };
    MetalBackend.applySamplingMutationsToLogits(logits[0..], &cfg);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), logits[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -4.0), logits[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), logits[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), logits[3], 1e-6);
}

test "MetalBackend.applySamplingMutationsToLogits applies additive penalties" {
    var logits = [_]f32{ 10.0, 10.0, 10.0 };
    const context_tokens = [_]u32{ 0, 1, 0 };
    const cfg = sampling.SamplingConfig{
        .strategy = .top_k,
        .top_k = 2,
        .presence_penalty = 1.0,
        .frequency_penalty = 0.5,
        .context_tokens = context_tokens[0..],
    };
    MetalBackend.applySamplingMutationsToLogits(logits[0..], &cfg);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), logits[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 8.5), logits[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), logits[2], 1e-6);
}
