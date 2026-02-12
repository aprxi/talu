//! Metal backend for transformer inference (macOS GPU via MLX).
//!
//! Provides GPU-accelerated inference using Apple's MLX framework.
//! Supports lazy graph execution for optimal GPU utilization.

const std = @import("std");
const loader = @import("../../../io/root.zig").weights;
const tensor = @import("../../../tensor.zig");
const ModelConfig = tensor.ModelConfig;
const log = @import("../../../log.zig");

// Import compute primitives from compute/
const compute = @import("../../../compute/root.zig");
const metal_compute = compute.metal;
const graph = metal_compute.graph;

// Internal orchestration modules
const mlx_forward = @import("mlx_forward.zig");

// Re-exports for direct access if needed
pub const device = metal_compute.device;
pub const matmul = metal_compute.matmul;
pub const Graph = graph;
pub const Forward = mlx_forward;

pub const Device = metal_compute.Device;
pub const Buffer = metal_compute.Buffer;
pub const isAvailable = metal_compute.isAvailable;
pub const Cache = metal_compute.Cache;
pub const WeightHandles = mlx_forward.WeightHandles;

/// Metal backend for GPU-accelerated transformer inference
pub const MetalBackend = struct {
    allocator: std.mem.Allocator,
    config: ModelConfig,
    weights: *mlx_forward.WeightHandles,
    cache: graph.Cache,
    shortconv_cache: graph.ShortConvCache,
    vocab_size: usize,
    d_model: usize,
    has_shortconv: bool,

    // Track position for decode
    current_position: usize,

    // Fused model handles (quantized or dense)
    fused_model: ?*anyopaque,
    dense_model: ?*anyopaque,

    fn shouldUseFusedDecodePath(
        force_non_fused_path: bool,
        fused_model: ?*anyopaque,
        dense_model: ?*anyopaque,
    ) bool {
        if (force_non_fused_path) return false;
        return fused_model != null or dense_model != null;
    }

    fn shouldUseBatchDecode(
        env_batch_decode: bool,
        callback: ?*const fn (u32, ?*anyopaque) void,
        fused_model: ?*anyopaque,
        dense_model: ?*anyopaque,
    ) bool {
        if (env_batch_decode) return true;
        return callback == null and (fused_model != null or dense_model != null);
    }

    pub fn init(allocator: std.mem.Allocator, loaded: *loader.LoadedModel) !MetalBackend {
        // Load weights to GPU
        const weight_handles = try mlx_forward.loadWeightsToGPU(allocator, loaded);
        errdefer mlx_forward.freeWeights(allocator, weight_handles);

        // Create fused model for decode optimization
        mlx_forward.createFusedModel(allocator, weight_handles, loaded.config) catch |err| {
            log.warn("inference", "Failed to create fused model", .{ .err = @errorName(err) });
            // Continue without fused model - will fall back to per-layer calls
        };

        log.debug("inference", "Metal decode model selection", .{
            .fused_model = @as(u8, @intFromBool(weight_handles.fused_model != null)),
            .dense_model = @as(u8, @intFromBool(weight_handles.dense_model != null)),
            .has_shortconv = @as(u8, @intFromBool(weight_handles.has_shortconv)),
            .is_quantized = @as(u8, @intFromBool(weight_handles.is_quantized)),
            .is_moe = @as(u8, @intFromBool(weight_handles.is_moe)),
        }, @src());

        // Initialize KV cache (bfloat16)
        const layer_count: usize = @intCast(loaded.config.n_layers);
        const kv_cache = graph.Cache.init(layer_count, true);
        const shortconv_cache = graph.ShortConvCache.init(layer_count);

        return MetalBackend{
            .allocator = allocator,
            .config = loaded.config,
            .weights = weight_handles,
            .cache = kv_cache,
            .shortconv_cache = shortconv_cache,
            .vocab_size = @intCast(loaded.config.vocab_size),
            .d_model = @intCast(loaded.config.d_model),
            .has_shortconv = weight_handles.has_shortconv,
            .current_position = 0,
            .fused_model = weight_handles.fused_model,
            .dense_model = weight_handles.dense_model,
        };
    }

    pub fn deinit(self: *MetalBackend) void {
        self.cache.deinit();
        self.shortconv_cache.deinit();
        mlx_forward.freeWeights(self.allocator, self.weights);
        self.* = undefined;
    }

    /// Prefill: process all prompt tokens, return logits for last position
    pub fn prefill(self: *MetalBackend, tokens: []const u32, logits_out: []f32) !void {
        const sequence_len = tokens.len;

        // Reset cache for new sequence
        self.cache.deinit();
        self.cache = graph.Cache.init(@intCast(self.config.n_layers), true);
        self.shortconv_cache.reset();

        // Build lazy computation graph
        const logits_handle = try mlx_forward.transformerForwardLazy(
            self.allocator,
            self.weights,
            tokens,
            self.config,
            self.cache,
            self.shortconv_cache,
            0, // pos_offset
            false, // use_compiled (prefill must use manual path)
        );

        // Execute graph on GPU
        graph.eval(&[_]graph.ArrayHandle{logits_handle});

        // Get shape to verify
        var shape_buffer: [8]usize = undefined;
        const rank = graph.getShape(logits_handle, &shape_buffer);
        std.debug.assert(rank == 3);
        std.debug.assert(shape_buffer[0] == 1);
        std.debug.assert(shape_buffer[1] == sequence_len);
        std.debug.assert(shape_buffer[2] == self.vocab_size);

        // Copy full logits from GPU
        const logits_values = try self.allocator.alloc(f32, sequence_len * self.vocab_size);
        defer self.allocator.free(logits_values);
        graph.copyToHost(logits_handle, logits_values);

        // Extract last position
        const last_token_offset = (sequence_len - 1) * self.vocab_size;
        @memcpy(logits_out, logits_values[last_token_offset .. last_token_offset + self.vocab_size]);

        // Free handle
        graph.freeArray(logits_handle);

        // Update position
        self.current_position = sequence_len;
    }

    /// Decode: generate logits for a single token using KV cache
    pub fn decode(self: *MetalBackend, token: u32, position: usize, logits_out: []f32) !void {
        // Sampled decode should still use fused decode kernels when available.
        // This keeps core behavior architecture-agnostic and avoids per-token lazy graph rebuilds.
        if (self.fused_model) |fused| {
            const logits_handle = graph.mlx_fused_decode_step_logits(
                fused,
                self.cache.handle,
                self.shortconv_cache.handle,
                token,
                position,
            );
            graph.copyToHost(logits_handle, logits_out);
            self.current_position = position + 1;
            return;
        }
        if (self.dense_model) |dense| {
            const logits_handle = graph.mlx_dense_decode_step_logits(
                dense,
                self.cache.handle,
                self.shortconv_cache.handle,
                token,
                position,
            );
            graph.copyToHost(logits_handle, logits_out);
            self.current_position = position + 1;
            return;
        }

        const token_id_slice = &[_]u32{token};

        // Build lazy computation graph for single token
        const logits_handle = try mlx_forward.transformerForwardLazy(
            self.allocator,
            self.weights,
            token_id_slice,
            self.config,
            self.cache,
            self.shortconv_cache,
            position,
            false,
        );

        // Execute graph
        graph.eval(&[_]graph.ArrayHandle{logits_handle});

        // Copy logits from GPU
        graph.copyToHost(logits_handle, logits_out);

        // Free handle
        graph.freeArray(logits_handle);

        self.current_position = position + 1;
    }

    /// Decode with streaming - Metal uses pipelined execution for better throughput
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
        // Allow forcing non-fused path for debugging
        const force_non_fused_path = std.process.hasEnvVar(self.allocator, "TALU_NO_FUSED") catch false;
        const env_batch_decode = std.process.hasEnvVar(self.allocator, "TALU_BATCH_DECODE") catch false;
        // When no streaming callback is required, run decode fully in C++ to avoid
        // per-token Zig/FFI overhead on the fused quantized path.
        const batch_decode_enabled = shouldUseBatchDecode(
            env_batch_decode,
            callback,
            self.fused_model,
            self.dense_model,
        );
        const use_fused_path = shouldUseFusedDecodePath(
            force_non_fused_path,
            self.fused_model,
            self.dense_model,
        );
        const decode_path: []const u8 = if (batch_decode_enabled and use_fused_path and self.fused_model != null)
            "batch_fused"
        else if (batch_decode_enabled and use_fused_path and self.dense_model != null)
            "batch_dense"
        else if (use_fused_path and self.fused_model != null)
            "pipeline_fused"
        else if (use_fused_path and self.dense_model != null)
            "pipeline_dense"
        else
            "non_fused";
        log.debug("inference", "Metal decode path", .{
            .fused = @as(u8, @intFromBool(self.fused_model != null)),
            .dense = @as(u8, @intFromBool(self.dense_model != null)),
            .selected_fused_path = @as(u8, @intFromBool(use_fused_path)),
            .force_non_fused_path = @as(u8, @intFromBool(force_non_fused_path)),
            .batch_decode_enabled = @as(u8, @intFromBool(batch_decode_enabled)),
            .path = decode_path,
        }, @src());

        // Batch decode: run entire loop in C++ to eliminate FFI overhead
        // Note: callbacks are called AFTER batch completes (not during)
        if (batch_decode_enabled and use_fused_path) {
            const generated_count = if (self.fused_model) |fused|
                graph.mlx_fused_decode_batch(
                    fused,
                    self.cache.handle,
                    self.shortconv_cache.handle,
                    first_token,
                    start_position,
                    output_tokens.ptr,
                    max_tokens,
                    eos_token_ids.ptr,
                    eos_token_ids.len,
                )
            else if (self.dense_model) |dense|
                graph.mlx_dense_decode_batch(
                    dense,
                    self.cache.handle,
                    self.shortconv_cache.handle,
                    first_token,
                    start_position,
                    output_tokens.ptr,
                    max_tokens,
                    eos_token_ids.ptr,
                    eos_token_ids.len,
                )
            else
                0;

            if (generated_count > 0) {
                // Call callbacks after batch (for streaming output)
                if (callback) |cb| {
                    for (0..generated_count) |token_index| {
                        cb(output_tokens[token_index], callback_data);
                    }
                }
                self.current_position = start_position + generated_count;
                return generated_count;
            }
        }

        if (use_fused_path) {
            return self.decodeStreamingFused(
                first_token,
                start_position,
                max_tokens,
                eos_token_ids,
                output_tokens,
                callback,
                callback_data,
            );
        } else {
            return self.decodeStreamingNonFused(
                first_token,
                start_position,
                max_tokens,
                eos_token_ids,
                output_tokens,
                callback,
                callback_data,
            );
        }
    }

    /// Fused decode path - uses pipelined C++ execution
    fn decodeStreamingFused(
        self: *MetalBackend,
        first_token: u32,
        start_position: usize,
        max_tokens: usize,
        eos_token_ids: []const u32,
        output_tokens: []u32,
        callback: ?*const fn (u32, ?*anyopaque) void,
        callback_data: ?*anyopaque,
    ) !usize {
        var generated_count: usize = 0;
        var position_index = start_position;

        // Prime the pipeline
        if (self.fused_model) |fused| {
            graph.mlx_pipeline_prime(
                fused,
                self.cache.handle,
                self.shortconv_cache.handle,
                first_token,
                position_index,
            );
        } else if (self.dense_model) |dense| {
            graph.mlx_dense_pipeline_prime(
                dense,
                self.cache.handle,
                self.shortconv_cache.handle,
                first_token,
                position_index,
            );
        }
        position_index += 1;

        while (generated_count < max_tokens) : (generated_count += 1) {
            var sampled_token_id: u32 = undefined; // Safe: both branches assign before use

            if (generated_count + 1 < max_tokens) {
                // Normal step: returns current, queues next
                if (self.fused_model) |fused| {
                    sampled_token_id = graph.mlx_pipeline_step(
                        fused,
                        self.cache.handle,
                        self.shortconv_cache.handle,
                        position_index,
                    );
                } else if (self.dense_model) |dense| {
                    sampled_token_id = graph.mlx_dense_pipeline_step(
                        dense,
                        self.cache.handle,
                        self.shortconv_cache.handle,
                        position_index,
                    );
                }
            } else {
                // Last iteration: flush
                if (self.fused_model != null) {
                    sampled_token_id = graph.mlx_pipeline_flush();
                } else {
                    sampled_token_id = graph.mlx_dense_pipeline_flush();
                }
            }
            position_index += 1;

            // Store token
            output_tokens[generated_count] = sampled_token_id;
            log.trace("inference", "Generated token", .{ .idx = generated_count, .token_id = sampled_token_id, .position = position_index - 1 }, @src());

            // Check for EOS
            var is_eos_token = false;
            for (eos_token_ids) |eos_id| {
                if (sampled_token_id == eos_id) {
                    is_eos_token = true;
                    break;
                }
            }

            // Invoke callback
            if (callback) |cb| {
                cb(sampled_token_id, callback_data);
            }

            if (is_eos_token) {
                generated_count += 1;
                break;
            }
        }

        self.current_position = position_index;
        return generated_count;
    }

    /// Non-fused decode path - uses lazy graph API with pipelining
    fn decodeStreamingNonFused(
        self: *MetalBackend,
        first_token: u32,
        start_position: usize,
        max_tokens: usize,
        eos_token_ids: []const u32,
        output_tokens: []u32,
        callback: ?*const fn (u32, ?*anyopaque) void,
        callback_data: ?*anyopaque,
    ) !usize {
        var generated_count: usize = 0;
        var position_index = start_position;

        // Prime: Build first token graph
        const first_token_ids = &[_]u32{first_token};
        var current_logits_handle = try mlx_forward.transformerForwardLazy(
            self.allocator,
            self.weights,
            first_token_ids,
            self.config,
            self.cache,
            self.shortconv_cache,
            position_index,
            false,
        );
        var current_logits_last = graph.mlx_lazy_slice_last(current_logits_handle);
        var current_token_handle = graph.mlx_lazy_argmax(current_logits_last, -1);
        graph.asyncEval(&[_]graph.ArrayHandle{current_token_handle});
        position_index += 1;

        while (generated_count < max_tokens) : (generated_count += 1) {
            var sampled_token_id: u32 = undefined; // Safe: both branches assign before use

            if (generated_count + 1 < max_tokens) {
                // Build graph for NEXT token using current (lazy) token
                const next_logits_handle = try mlx_forward.transformerForwardFromGPUToken(
                    self.allocator,
                    self.weights,
                    current_token_handle,
                    self.config,
                    self.cache,
                    self.shortconv_cache,
                    position_index,
                );
                const next_logits_last = graph.mlx_lazy_slice_last(next_logits_handle);
                const next_token_handle = graph.mlx_lazy_argmax(next_logits_last, -1);

                // Queue next token computation
                graph.asyncEval(&[_]graph.ArrayHandle{next_token_handle});

                // Materialize current token
                sampled_token_id = graph.mlx_array_item_u32(current_token_handle);

                // Free old handles
                graph.freeArray(current_logits_handle);

                // Rotate
                current_logits_handle = next_logits_handle;
                current_logits_last = next_logits_last;
                current_token_handle = next_token_handle;
            } else {
                // Last iteration - just materialize current
                sampled_token_id = graph.mlx_array_item_u32(current_token_handle);
                graph.freeArray(current_logits_handle);
            }
            position_index += 1;

            // Reset pool periodically
            if (generated_count != 0 and generated_count % 64 == 0) {
                graph.mlx_pool_reset();
            }

            // Store token
            output_tokens[generated_count] = sampled_token_id;

            // Check for EOS
            var is_eos_token = false;
            for (eos_token_ids) |eos_id| {
                if (sampled_token_id == eos_id) {
                    is_eos_token = true;
                    break;
                }
            }

            // Invoke callback
            if (callback) |cb| {
                cb(sampled_token_id, callback_data);
            }

            // Clear memory cache periodically
            if (generated_count != 0 and generated_count % 256 == 0) {
                graph.mlx_clear_memory_cache();
            }

            if (is_eos_token) {
                generated_count += 1;
                break;
            }
        }

        self.current_position = position_index;
        return generated_count;
    }
};

test "shouldUseFusedDecodePath returns false when forced non-fused" {
    try std.testing.expect(!MetalBackend.shouldUseFusedDecodePath(true, @ptrFromInt(1), null));
}

test "shouldUseFusedDecodePath returns false when fused and dense models are null" {
    try std.testing.expect(!MetalBackend.shouldUseFusedDecodePath(false, null, null));
}

test "shouldUseBatchDecode enables auto batch for fused model without callback" {
    try std.testing.expect(MetalBackend.shouldUseBatchDecode(false, null, @ptrFromInt(1), null));
}

test "shouldUseBatchDecode enables auto batch for dense model without callback" {
    try std.testing.expect(MetalBackend.shouldUseBatchDecode(false, null, null, @ptrFromInt(1)));
}

test "shouldUseBatchDecode keeps streaming path when callback is set" {
    const TestCallback = struct {
        fn cb(_: u32, _: ?*anyopaque) void {}
    };
    try std.testing.expect(!MetalBackend.shouldUseBatchDecode(
        false,
        TestCallback.cb,
        @ptrFromInt(1),
        null,
    ));
}

test "shouldUseBatchDecode honors env override" {
    try std.testing.expect(MetalBackend.shouldUseBatchDecode(
        true,
        null,
        null,
        null,
    ));
}

test "shouldUseFusedDecodePath returns true when fused model exists" {
    try std.testing.expect(MetalBackend.shouldUseFusedDecodePath(false, @ptrFromInt(1), null));
}

test "shouldUseFusedDecodePath returns true when dense model exists" {
    try std.testing.expect(MetalBackend.shouldUseFusedDecodePath(false, null, @ptrFromInt(1)));
}

test {
    @import("std").testing.refAllDecls(@This());
    _ = @import("test.zig");
}
