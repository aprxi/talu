//! Model Validation
//!
//! Validates loaded models for correctness (tensor shapes, config consistency).

const std = @import("std");
const weights = @import("weights.zig");
const tensor = @import("../../tensor.zig");
const log = @import("../../log.zig");

pub const Error = error{ValidationFailed};

pub const Reporter = struct {
    writer: *std.io.AnyWriter,
    is_verbose: bool,

    pub fn init(writer: *std.io.AnyWriter, is_verbose: bool) Reporter {
        return .{ .writer = writer, .is_verbose = is_verbose };
    }

    pub fn reportInfo(self: *Reporter, comptime fmt: []const u8, args: anytype) void {
        if (self.is_verbose) {
            self.writer.print(fmt ++ "\n", args) catch {};
        }
        log.trace("load", fmt, args, @src());
    }

    pub fn reportError(self: *Reporter, comptime fmt: []const u8, args: anytype) Error {
        self.writer.print("validation failed: " ++ fmt ++ "\n", args) catch {};
        return error.ValidationFailed;
    }
};

/// Validate a loaded model's weight shapes against its config.
///
/// Called automatically after model loading to catch config-vs-weight
/// mismatches at load time (e.g., d_ff from config doesn't match actual
/// weight dimensions). Returns error.ValidationFailed on shape mismatch,
/// preventing silent corruption during inference.
pub fn validate(loaded_model: *weights.LoadedModel) Error!void {
    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var writer = fbs.writer().any();
    var reporter = Reporter.init(&writer, false);
    validateCommon(&reporter, loaded_model) catch |err| {
        const written = fbs.getWritten();
        if (written.len > 0) {
            log.err("load", "Model validation failed", .{
                .detail = written,
            }, @src());
        }
        return err;
    };
}

fn validateCommon(reporter: *Reporter, loaded_model: *weights.LoadedModel) !void {
    const d_model: usize = @intCast(loaded_model.config.d_model);
    const d_ff: usize = @intCast(loaded_model.config.d_ff);
    const head_dim: usize = @intCast(loaded_model.config.head_dim);
    const n_heads: usize = @intCast(loaded_model.config.n_heads);
    const n_kv_groups: usize = @intCast(loaded_model.config.n_kv_groups);
    const vocab_size: usize = @intCast(loaded_model.config.vocab_size);

    if (loaded_model.token_embeddings.n_dims != 2) return reporter.reportError("token_embeddings not 2D (n_dims={})", .{loaded_model.token_embeddings.n_dims});
    if (!is2DShapeMatch(&loaded_model.token_embeddings, vocab_size, d_model))
        return reporter.reportError("token_embeddings shape mismatch (shape=[{},{}], expected=[{},{}])", .{
            loaded_model.token_embeddings.shape[0], loaded_model.token_embeddings.shape[1], vocab_size, d_model,
        });

    if (loaded_model.blocks.len != @as(usize, @intCast(loaded_model.config.n_layers)))
        return reporter.reportError("blocks len mismatch (blocks={} config={})", .{ loaded_model.blocks.len, loaded_model.config.n_layers });

    for (loaded_model.blocks, 0..) |block_weights, layer_idx| {
        switch (block_weights) {
            .attention_mlp => |block| {
                if (!isVectorShape(block.ln1_weight, d_model))
                    return reporter.reportError("layer {} ln1 shape mismatch (n_dims={}, shape0={}, expected={})", .{
                        layer_idx, block.ln1_weight.n_dims, block.ln1_weight.shape[0], d_model,
                    });
                if (!isVectorShape(block.ln2_weight, d_model))
                    return reporter.reportError("layer {} ln2 shape mismatch (n_dims={}, shape0={}, expected={})", .{
                        layer_idx, block.ln2_weight.n_dims, block.ln2_weight.shape[0], d_model,
                    });

                const q_out: usize = n_heads * head_dim;
                const kv_out: usize = n_kv_groups * head_dim;

                // Check for MLA (Multi-Latent Attention) weights first
                const is_mla = block.q_a_proj != null;
                if (is_mla) {
                    // MLA has compressed Q/KV projections - skip standard Q/K/V validation
                    // Shape validation for MLA weights is model-specific (handled by expected_shape in graph)
                } else if (block.fused.qkv_proj) |qkv| {
                    const qkv_out: usize = q_out + 2 * kv_out;
                    if (!is2DShapeMatch(&qkv, d_model, qkv_out))
                        return reporter.reportError("layer {} qkv_proj shape mismatch (shape=[{},{}], expected=[{},{}])", .{
                            layer_idx, qkv.shape[0], qkv.shape[1], d_model, qkv_out,
                        });
                } else {
                    if (block.q_proj == null or block.k_proj == null or block.v_proj == null)
                        return reporter.reportError("layer {} missing q/k/v weights without fused qkv", .{layer_idx});
                    if (!is2DShapeMatch(block.q_proj.?, d_model, q_out))
                        return reporter.reportError("layer {} q_proj shape mismatch (shape=[{},{}], expected=[{},{}])", .{
                            layer_idx, block.q_proj.?.shape[0], block.q_proj.?.shape[1], d_model, q_out,
                        });
                    if (!is2DShapeMatch(block.k_proj.?, d_model, kv_out))
                        return reporter.reportError("layer {} k_proj shape mismatch (shape=[{},{}], expected=[{},{}])", .{
                            layer_idx, block.k_proj.?.shape[0], block.k_proj.?.shape[1], d_model, kv_out,
                        });
                    if (!is2DShapeMatch(block.v_proj.?, d_model, kv_out))
                        return reporter.reportError("layer {} v_proj shape mismatch (shape=[{},{}], expected=[{},{}])", .{
                            layer_idx, block.v_proj.?.shape[0], block.v_proj.?.shape[1], d_model, kv_out,
                        });
                }

                // o_proj validation - skip for MLA which has different dimensions
                if (!is_mla and !is2DShapeMatch(block.o_proj, q_out, d_model))
                    return reporter.reportError("layer {} o_proj shape mismatch (shape=[{},{}], expected=[{},{}])", .{
                        layer_idx, block.o_proj.shape[0], block.o_proj.shape[1], q_out, d_model,
                    });

                if (block.moe_weights) |moe_w| {
                    if (!is2DShapeMatch(&moe_w.router_weight, d_model, @intCast(moe_w.num_experts)))
                        return reporter.reportError("layer {} router_weight shape mismatch (shape=[{},{}], expected=[{},{}])", .{
                            layer_idx, moe_w.router_weight.shape[0], moe_w.router_weight.shape[1], d_model, moe_w.num_experts,
                        });
                } else {
                    if (block.fused.gate_up) |gate_up| {
                        if (!is2DShapeMatch(&gate_up, d_model, d_ff * 2))
                            return reporter.reportError("layer {} gate_up shape mismatch (shape=[{},{}], expected=[{},{}])", .{
                                layer_idx, gate_up.shape[0], gate_up.shape[1], d_model, d_ff * 2,
                            });
                    } else {
                        // Dense FFN: w1 (gate/dense_in) and w2 (down/dense_out) required.
                        // w3 (up projection) is optional - absent in non-gated MLPs (BERT/MiniLM).
                        if (block.w1 == null or block.w2 == null)
                            return reporter.reportError("layer {} missing dense FFN weights", .{layer_idx});
                        if (!is2DShapeMatch(block.w1.?, d_model, d_ff))
                            return reporter.reportError("layer {} w1 shape mismatch (shape=[{},{}], expected=[{},{}])", .{
                                layer_idx, block.w1.?.shape[0], block.w1.?.shape[1], d_model, d_ff,
                            });
                        if (block.w3) |w3| {
                            if (!is2DShapeMatch(w3, d_model, d_ff))
                                return reporter.reportError("layer {} w3 shape mismatch (shape=[{},{}], expected=[{},{}])", .{
                                    layer_idx, w3.shape[0], w3.shape[1], d_model, d_ff,
                                });
                        }
                        if (!is2DShapeMatch(block.w2.?, d_ff, d_model))
                            return reporter.reportError("layer {} w2 shape mismatch (shape=[{},{}], expected=[{},{}])", .{
                                layer_idx, block.w2.?.shape[0], block.w2.?.shape[1], d_ff, d_model,
                            });
                    }
                }

                if (block.q_norm) |q_norm| {
                    if (!isVectorShape(q_norm, head_dim))
                        return reporter.reportError("layer {} q_norm shape mismatch (n_dims={}, shape0={}, expected={})", .{
                            layer_idx, q_norm.n_dims, q_norm.shape[0], head_dim,
                        });
                }
                if (block.k_norm) |k_norm| {
                    if (!isVectorShape(k_norm, head_dim))
                        return reporter.reportError("layer {} k_norm shape mismatch (n_dims={}, shape0={}, expected={})", .{
                            layer_idx, k_norm.n_dims, k_norm.shape[0], head_dim,
                        });
                }
                if (block.pre_ffn_norm) |pre_ffn_norm| {
                    if (!isVectorShape(pre_ffn_norm, d_model))
                        return reporter.reportError("layer {} pre_ffn_norm shape mismatch (n_dims={}, shape0={}, expected={})", .{
                            layer_idx, pre_ffn_norm.n_dims, pre_ffn_norm.shape[0], d_model,
                        });
                }
                if (block.post_ffn_norm) |post_ffn_norm| {
                    if (!isVectorShape(post_ffn_norm, d_model))
                        return reporter.reportError("layer {} post_ffn_norm shape mismatch (n_dims={}, shape0={}, expected={})", .{
                            layer_idx, post_ffn_norm.n_dims, post_ffn_norm.shape[0], d_model,
                        });
                }
            },
            .shortconv => |block| {
                if (!isVectorShape(block.ln1_weight, d_model))
                    return reporter.reportError("layer {} ln1 shape mismatch (n_dims={}, shape0={}, expected={})", .{
                        layer_idx, block.ln1_weight.n_dims, block.ln1_weight.shape[0], d_model,
                    });
                if (block.ln2_weight) |ln2| {
                    if (!isVectorShape(ln2, d_model))
                        return reporter.reportError("layer {} ln2 shape mismatch (n_dims={}, shape0={}, expected={})", .{
                            layer_idx, ln2.n_dims, ln2.shape[0], d_model,
                        });
                }
                // Validate FFN weights (fused or separate)
                if (block.fused_gate_up) |fused| {
                    if (!is2DShapeMatch(&fused.gate_up.?, d_model, d_ff * 2))
                        return reporter.reportError("layer {} gate_up shape mismatch (shape=[{},{}], expected=[{},{}])", .{
                            layer_idx, fused.gate_up.?.shape[0], fused.gate_up.?.shape[1], d_model, d_ff * 2,
                        });
                } else {
                    if (block.w1) |w1| {
                        if (!is2DShapeMatch(w1, d_model, d_ff))
                            return reporter.reportError("layer {} w1 shape mismatch (shape=[{},{}], expected=[{},{}])", .{
                                layer_idx, w1.shape[0], w1.shape[1], d_model, d_ff,
                            });
                    }
                    if (block.w2) |w2| {
                        if (!is2DShapeMatch(w2, d_ff, d_model))
                            return reporter.reportError("layer {} w2 shape mismatch (shape=[{},{}], expected=[{},{}])", .{
                                layer_idx, w2.shape[0], w2.shape[1], d_ff, d_model,
                            });
                    }
                    if (block.w3) |w3| {
                        if (!is2DShapeMatch(w3, d_model, d_ff))
                            return reporter.reportError("layer {} w3 shape mismatch (shape=[{},{}], expected=[{},{}])", .{
                                layer_idx, w3.shape[0], w3.shape[1], d_model, d_ff,
                            });
                    }
                }
            },
            .mamba => |block| {
                // Mamba blocks have different structure - validate ln1 only (ln2 is optional)
                if (!isVectorShape(block.ln1_weight, d_model))
                    return reporter.reportError("layer {} ln1 shape mismatch (n_dims={}, shape0={}, expected={})", .{
                        layer_idx, block.ln1_weight.n_dims, block.ln1_weight.shape[0], d_model,
                    });
            },
        }
    }

    if (loaded_model.lm_head) |lm_head| {
        if (lm_head.n_dims != 2)
            return reporter.reportError("lm_head not 2D (n_dims={})", .{lm_head.n_dims});
        if (!is2DShapeMatch(&lm_head, vocab_size, d_model))
            return reporter.reportError("lm_head shape mismatch (shape=[{},{}], expected=[{},{}])", .{
                lm_head.shape[0], lm_head.shape[1], vocab_size, d_model,
            });
    }
}

fn is2DShapeMatch(tensor_view: *const tensor.Tensor, dim_a: usize, dim_b: usize) bool {
    return tensor_view.n_dims == 2 and ((tensor_view.shape[0] == dim_a and tensor_view.shape[1] == dim_b) or (tensor_view.shape[0] == dim_b and tensor_view.shape[1] == dim_a));
}

fn isVectorShape(tensor_view: *const tensor.Tensor, len: usize) bool {
    if (tensor_view.n_dims == 1) return tensor_view.shape[0] == len;
    if (tensor_view.n_dims == 2) return (tensor_view.shape[0] == len and tensor_view.shape[1] == 1) or (tensor_view.shape[0] == 1 and tensor_view.shape[1] == len);
    return false;
}

// =============================================================================
// Unit Tests
// =============================================================================

test "Reporter.init: creates reporter with correct fields" {
    var buffer: std.ArrayList(u8) = .{};
    defer buffer.deinit(std.testing.allocator);

    var any_writer = std.io.AnyWriter{ .context = &buffer, .writeFn = struct {
        fn writeFn(context: *const anyopaque, bytes: []const u8) anyerror!usize {
            const buf: *std.ArrayList(u8) = @ptrCast(@alignCast(@constCast(context)));
            try buf.appendSlice(std.testing.allocator, bytes);
            return bytes.len;
        }
    }.writeFn };

    // Test verbose = false
    const reporter_quiet = Reporter.init(&any_writer, false);
    try std.testing.expectEqual(false, reporter_quiet.is_verbose);
    try std.testing.expectEqual(&any_writer, reporter_quiet.writer);

    // Test verbose = true
    const reporter_verbose = Reporter.init(&any_writer, true);
    try std.testing.expectEqual(true, reporter_verbose.is_verbose);
    try std.testing.expectEqual(&any_writer, reporter_verbose.writer);
}

test "Reporter.reportInfo: outputs formatted message when verbose" {
    var buffer: std.ArrayList(u8) = .{};
    defer buffer.deinit(std.testing.allocator);

    var any_writer = std.io.AnyWriter{ .context = &buffer, .writeFn = struct {
        fn writeFn(context: *const anyopaque, bytes: []const u8) anyerror!usize {
            const buf: *std.ArrayList(u8) = @ptrCast(@alignCast(@constCast(context)));
            try buf.appendSlice(std.testing.allocator, bytes);
            return bytes.len;
        }
    }.writeFn };

    var reporter = Reporter.init(&any_writer, true);
    reporter.reportInfo("validation check: {} tensors processed", .{42});

    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "validation check: 42 tensors processed\n") != null);
}

test "Reporter.reportInfo: suppresses output when not verbose" {
    var buffer: std.ArrayList(u8) = .{};
    defer buffer.deinit(std.testing.allocator);

    var any_writer = std.io.AnyWriter{ .context = &buffer, .writeFn = struct {
        fn writeFn(context: *const anyopaque, bytes: []const u8) anyerror!usize {
            const buf: *std.ArrayList(u8) = @ptrCast(@alignCast(@constCast(context)));
            try buf.appendSlice(std.testing.allocator, bytes);
            return bytes.len;
        }
    }.writeFn };

    var reporter = Reporter.init(&any_writer, false);
    reporter.reportInfo("this should not appear", .{});

    try std.testing.expectEqual(@as(usize, 0), buffer.items.len);
}

test "Reporter.reportInfo: handles multiple arguments" {
    var buffer: std.ArrayList(u8) = .{};
    defer buffer.deinit(std.testing.allocator);

    var any_writer = std.io.AnyWriter{ .context = &buffer, .writeFn = struct {
        fn writeFn(context: *const anyopaque, bytes: []const u8) anyerror!usize {
            const buf: *std.ArrayList(u8) = @ptrCast(@alignCast(@constCast(context)));
            try buf.appendSlice(std.testing.allocator, bytes);
            return bytes.len;
        }
    }.writeFn };

    var reporter = Reporter.init(&any_writer, true);
    reporter.reportInfo("layer {}: shape [{}, {}]", .{ 5, 128, 256 });

    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "layer 5: shape [128, 256]\n") != null);
}

test "Reporter.reportInfo: handles empty format string" {
    var buffer: std.ArrayList(u8) = .{};
    defer buffer.deinit(std.testing.allocator);

    var any_writer = std.io.AnyWriter{ .context = &buffer, .writeFn = struct {
        fn writeFn(context: *const anyopaque, bytes: []const u8) anyerror!usize {
            const buf: *std.ArrayList(u8) = @ptrCast(@alignCast(@constCast(context)));
            try buf.appendSlice(std.testing.allocator, bytes);
            return bytes.len;
        }
    }.writeFn };

    var reporter = Reporter.init(&any_writer, true);
    reporter.reportInfo("", .{});

    try std.testing.expectEqualStrings("\n", buffer.items);
}

test "Reporter.reportError: returns ValidationFailed error" {
    var buffer: std.ArrayList(u8) = .{};
    defer buffer.deinit(std.testing.allocator);

    var any_writer = std.io.AnyWriter{ .context = &buffer, .writeFn = struct {
        fn writeFn(context: *const anyopaque, bytes: []const u8) anyerror!usize {
            const buf: *std.ArrayList(u8) = @ptrCast(@alignCast(@constCast(context)));
            try buf.appendSlice(std.testing.allocator, bytes);
            return bytes.len;
        }
    }.writeFn };

    var reporter = Reporter.init(&any_writer, false);
    const err = reporter.reportError("test error: {}", .{42});

    try std.testing.expectEqual(Error.ValidationFailed, err);
}

test "Reporter.reportError: outputs formatted error message" {
    var buffer: std.ArrayList(u8) = .{};
    defer buffer.deinit(std.testing.allocator);

    var any_writer = std.io.AnyWriter{ .context = &buffer, .writeFn = struct {
        fn writeFn(context: *const anyopaque, bytes: []const u8) anyerror!usize {
            const buf: *std.ArrayList(u8) = @ptrCast(@alignCast(@constCast(context)));
            try buf.appendSlice(std.testing.allocator, bytes);
            return bytes.len;
        }
    }.writeFn };

    var reporter = Reporter.init(&any_writer, false);
    reporter.reportError("shape mismatch: expected {}, got {}", .{ 128, 64 }) catch {};

    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "validation failed: shape mismatch: expected 128, got 64\n") != null);
}

test "Reporter.reportError: includes prefix in output" {
    var buffer: std.ArrayList(u8) = .{};
    defer buffer.deinit(std.testing.allocator);

    var any_writer = std.io.AnyWriter{ .context = &buffer, .writeFn = struct {
        fn writeFn(context: *const anyopaque, bytes: []const u8) anyerror!usize {
            const buf: *std.ArrayList(u8) = @ptrCast(@alignCast(@constCast(context)));
            try buf.appendSlice(std.testing.allocator, bytes);
            return bytes.len;
        }
    }.writeFn };

    var reporter = Reporter.init(&any_writer, true);
    reporter.reportError("config invalid", .{}) catch {};

    const output = buffer.items;
    try std.testing.expect(std.mem.startsWith(u8, output, "validation failed: "));
    try std.testing.expect(std.mem.indexOf(u8, output, "config invalid") != null);
}

test "Reporter.reportError: works with verbose and non-verbose modes" {
    // Test that reportError writes regardless of verbose flag
    var buffer_verbose: std.ArrayList(u8) = .{};
    defer buffer_verbose.deinit(std.testing.allocator);
    var buffer_quiet: std.ArrayList(u8) = .{};
    defer buffer_quiet.deinit(std.testing.allocator);

    var any_writer_verbose = std.io.AnyWriter{ .context = &buffer_verbose, .writeFn = struct {
        fn writeFn(context: *const anyopaque, bytes: []const u8) anyerror!usize {
            const buf: *std.ArrayList(u8) = @ptrCast(@alignCast(@constCast(context)));
            try buf.appendSlice(std.testing.allocator, bytes);
            return bytes.len;
        }
    }.writeFn };

    var any_writer_quiet = std.io.AnyWriter{ .context = &buffer_quiet, .writeFn = struct {
        fn writeFn(context: *const anyopaque, bytes: []const u8) anyerror!usize {
            const buf: *std.ArrayList(u8) = @ptrCast(@alignCast(@constCast(context)));
            try buf.appendSlice(std.testing.allocator, bytes);
            return bytes.len;
        }
    }.writeFn };

    var reporter_verbose = Reporter.init(&any_writer_verbose, true);
    var reporter_quiet = Reporter.init(&any_writer_quiet, false);

    reporter_verbose.reportError("error message", .{}) catch {};
    reporter_quiet.reportError("error message", .{}) catch {};

    // Both should write the error
    try std.testing.expect(buffer_verbose.items.len > 0);
    try std.testing.expect(buffer_quiet.items.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, buffer_verbose.items, "validation failed: error message") != null);
    try std.testing.expect(std.mem.indexOf(u8, buffer_quiet.items, "validation failed: error message") != null);
}

test "Reporter.reportError: handles complex format strings" {
    var buffer: std.ArrayList(u8) = .{};
    defer buffer.deinit(std.testing.allocator);

    var any_writer = std.io.AnyWriter{ .context = &buffer, .writeFn = struct {
        fn writeFn(context: *const anyopaque, bytes: []const u8) anyerror!usize {
            const buf: *std.ArrayList(u8) = @ptrCast(@alignCast(@constCast(context)));
            try buf.appendSlice(std.testing.allocator, bytes);
            return bytes.len;
        }
    }.writeFn };

    var reporter = Reporter.init(&any_writer, false);
    reporter.reportError("layer {} qkv_proj shape mismatch (shape=[{},{}], expected=[{},{}])", .{ 3, 64, 128, 64, 192 }) catch {};

    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "validation failed: layer 3 qkv_proj shape mismatch (shape=[64,128], expected=[64,192])\n") != null);
}

test "Reporter: error reporting" {
    var buffer: std.ArrayList(u8) = .{};
    defer buffer.deinit(std.testing.allocator);

    var any_writer = std.io.AnyWriter{ .context = &buffer, .writeFn = struct {
        fn writeFn(context: *const anyopaque, bytes: []const u8) anyerror!usize {
            const buf: *std.ArrayList(u8) = @ptrCast(@alignCast(@constCast(context)));
            try buf.appendSlice(std.testing.allocator, bytes);
            return bytes.len;
        }
    }.writeFn };

    var reporter = Reporter.init(&any_writer, false);
    const err = reporter.reportError("test error: {}", .{42});

    try std.testing.expectEqual(Error.ValidationFailed, err);
    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "validation failed: test error: 42") != null);
}

test "Reporter: verbose info reporting" {
    var buffer: std.ArrayList(u8) = .{};
    defer buffer.deinit(std.testing.allocator);

    var any_writer = std.io.AnyWriter{ .context = &buffer, .writeFn = struct {
        fn writeFn(context: *const anyopaque, bytes: []const u8) anyerror!usize {
            const buf: *std.ArrayList(u8) = @ptrCast(@alignCast(@constCast(context)));
            try buf.appendSlice(std.testing.allocator, bytes);
            return bytes.len;
        }
    }.writeFn };

    var reporter = Reporter.init(&any_writer, true);
    reporter.reportInfo("test info: {}", .{123});

    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "test info: 123") != null);
}

test "Reporter: non-verbose info suppression" {
    var buffer: std.ArrayList(u8) = .{};
    defer buffer.deinit(std.testing.allocator);

    var any_writer = std.io.AnyWriter{ .context = &buffer, .writeFn = struct {
        fn writeFn(context: *const anyopaque, bytes: []const u8) anyerror!usize {
            const buf: *std.ArrayList(u8) = @ptrCast(@alignCast(@constCast(context)));
            try buf.appendSlice(std.testing.allocator, bytes);
            return bytes.len;
        }
    }.writeFn };

    var reporter = Reporter.init(&any_writer, false);
    reporter.reportInfo("should not appear", .{});

    try std.testing.expectEqual(@as(usize, 0), buffer.items.len);
}

test "is2DShapeMatch: correct shapes" {
    // Test matching shapes in [dim_a, dim_b] order
    var t1 = tensor.Tensor{
        .dtype = .f32,
        .n_dims = 2,
        .shape = .{ 10, 20, 0, 0, 0, 0, 0, 0 },
        .data_ptr = null,
        .data_size = 0,
    };
    try std.testing.expect(is2DShapeMatch(&t1, 10, 20));
    try std.testing.expect(is2DShapeMatch(&t1, 20, 10)); // symmetric check

    // Test non-matching shapes
    try std.testing.expect(!is2DShapeMatch(&t1, 10, 30));
    try std.testing.expect(!is2DShapeMatch(&t1, 15, 20));
}

test "is2DShapeMatch: wrong dimensions" {
    // Test 1D tensor (should fail)
    var t1d = tensor.Tensor{
        .dtype = .f32,
        .n_dims = 1,
        .shape = .{ 10, 0, 0, 0, 0, 0, 0, 0 },
        .data_ptr = null,
        .data_size = 0,
    };
    try std.testing.expect(!is2DShapeMatch(&t1d, 10, 1));

    // Test 3D tensor (should fail)
    var t3d = tensor.Tensor{
        .dtype = .f32,
        .n_dims = 3,
        .shape = .{ 10, 20, 30, 0, 0, 0, 0, 0 },
        .data_ptr = null,
        .data_size = 0,
    };
    try std.testing.expect(!is2DShapeMatch(&t3d, 10, 20));
}

test "isVectorShape: 1D vector" {
    // Test correct 1D vector
    var t1d = tensor.Tensor{
        .dtype = .f32,
        .n_dims = 1,
        .shape = .{ 100, 0, 0, 0, 0, 0, 0, 0 },
        .data_ptr = null,
        .data_size = 0,
    };
    try std.testing.expect(isVectorShape(&t1d, 100));
    try std.testing.expect(!isVectorShape(&t1d, 50));
}

test "isVectorShape: 2D column vector" {
    // Test [n, 1] shape (column vector)
    var t_col = tensor.Tensor{
        .dtype = .f32,
        .n_dims = 2,
        .shape = .{ 100, 1, 0, 0, 0, 0, 0, 0 },
        .data_ptr = null,
        .data_size = 0,
    };
    try std.testing.expect(isVectorShape(&t_col, 100));
    try std.testing.expect(!isVectorShape(&t_col, 50));
}

test "isVectorShape: 2D row vector" {
    // Test [1, n] shape (row vector)
    var t_row = tensor.Tensor{
        .dtype = .f32,
        .n_dims = 2,
        .shape = .{ 1, 100, 0, 0, 0, 0, 0, 0 },
        .data_ptr = null,
        .data_size = 0,
    };
    try std.testing.expect(isVectorShape(&t_row, 100));
    try std.testing.expect(!isVectorShape(&t_row, 50));
}

test "isVectorShape: non-vector 2D tensor" {
    // Test [m, n] where both m,n > 1 (not a vector)
    var t_matrix = tensor.Tensor{
        .dtype = .f32,
        .n_dims = 2,
        .shape = .{ 10, 20, 0, 0, 0, 0, 0, 0 },
        .data_ptr = null,
        .data_size = 0,
    };
    try std.testing.expect(!isVectorShape(&t_matrix, 10));
    try std.testing.expect(!isVectorShape(&t_matrix, 20));
}

test "isVectorShape: 3D tensor fails" {
    // Test that 3D tensors are never vectors
    var t3d = tensor.Tensor{
        .dtype = .f32,
        .n_dims = 3,
        .shape = .{ 1, 100, 1, 0, 0, 0, 0, 0 },
        .data_ptr = null,
        .data_size = 0,
    };
    try std.testing.expect(!isVectorShape(&t3d, 100));
}

test "validateCommon: valid minimal model" {
    // Create a minimal valid model configuration
    const d_model: i32 = 64;
    const d_ff: i32 = 256;
    const head_dim: i32 = 32;
    const n_heads: i32 = 2;
    const n_kv_groups: i32 = 2;
    const vocab_size: i32 = 1000;
    const n_layers: i32 = 2;

    const config = tensor.ModelConfig{
        .vocab_size = vocab_size,
        .d_model = d_model,
        .n_layers = n_layers,
        .n_heads = n_heads,
        .n_kv_groups = n_kv_groups,
        .d_ff = d_ff,
        .max_seq_len = 512,
        .head_dim = head_dim,
        .rope_theta = 10000.0,
        .norm_eps = 1e-5,
        .gaffine_group_size = 128,
    };

    const InferenceBackend = @import("../../inference/backend/cpu/executor/weights.zig");

    // Create mock block weights
    var blocks = [_]InferenceBackend.BlockWeights{
        .{ .attention_mlp = .{
            .ln1_weight = &tensor.Tensor{
                .dtype = .f32,
                .n_dims = 1,
                .shape = .{ d_model, 0, 0, 0, 0, 0, 0, 0 },
                .data_ptr = null,
                .data_size = 0,
            },
            .ln2_weight = &tensor.Tensor{
                .dtype = .f32,
                .n_dims = 1,
                .shape = .{ d_model, 0, 0, 0, 0, 0, 0, 0 },
                .data_ptr = null,
                .data_size = 0,
            },
            .q_proj = &tensor.Tensor{
                .dtype = .f32,
                .n_dims = 2,
                .shape = .{ d_model, head_dim * n_heads, 0, 0, 0, 0, 0, 0 },
                .data_ptr = null,
                .data_size = 0,
            },
            .k_proj = &tensor.Tensor{
                .dtype = .f32,
                .n_dims = 2,
                .shape = .{ d_model, head_dim * n_kv_groups, 0, 0, 0, 0, 0, 0 },
                .data_ptr = null,
                .data_size = 0,
            },
            .v_proj = &tensor.Tensor{
                .dtype = .f32,
                .n_dims = 2,
                .shape = .{ d_model, head_dim * n_kv_groups, 0, 0, 0, 0, 0, 0 },
                .data_ptr = null,
                .data_size = 0,
            },
            .o_proj = &tensor.Tensor{
                .dtype = .f32,
                .n_dims = 2,
                .shape = .{ head_dim * n_heads, d_model, 0, 0, 0, 0, 0, 0 },
                .data_ptr = null,
                .data_size = 0,
            },
            .w1 = &tensor.Tensor{
                .dtype = .f32,
                .n_dims = 2,
                .shape = .{ d_model, d_ff, 0, 0, 0, 0, 0, 0 },
                .data_ptr = null,
                .data_size = 0,
            },
            .w2 = &tensor.Tensor{
                .dtype = .f32,
                .n_dims = 2,
                .shape = .{ d_ff, d_model, 0, 0, 0, 0, 0, 0 },
                .data_ptr = null,
                .data_size = 0,
            },
            .w3 = &tensor.Tensor{
                .dtype = .f32,
                .n_dims = 2,
                .shape = .{ d_model, d_ff, 0, 0, 0, 0, 0, 0 },
                .data_ptr = null,
                .data_size = 0,
            },
        } },
        .{ .attention_mlp = .{
            .ln1_weight = &tensor.Tensor{
                .dtype = .f32,
                .n_dims = 1,
                .shape = .{ d_model, 0, 0, 0, 0, 0, 0, 0 },
                .data_ptr = null,
                .data_size = 0,
            },
            .ln2_weight = &tensor.Tensor{
                .dtype = .f32,
                .n_dims = 1,
                .shape = .{ d_model, 0, 0, 0, 0, 0, 0, 0 },
                .data_ptr = null,
                .data_size = 0,
            },
            .q_proj = &tensor.Tensor{
                .dtype = .f32,
                .n_dims = 2,
                .shape = .{ d_model, head_dim * n_heads, 0, 0, 0, 0, 0, 0 },
                .data_ptr = null,
                .data_size = 0,
            },
            .k_proj = &tensor.Tensor{
                .dtype = .f32,
                .n_dims = 2,
                .shape = .{ d_model, head_dim * n_kv_groups, 0, 0, 0, 0, 0, 0 },
                .data_ptr = null,
                .data_size = 0,
            },
            .v_proj = &tensor.Tensor{
                .dtype = .f32,
                .n_dims = 2,
                .shape = .{ d_model, head_dim * n_kv_groups, 0, 0, 0, 0, 0, 0 },
                .data_ptr = null,
                .data_size = 0,
            },
            .o_proj = &tensor.Tensor{
                .dtype = .f32,
                .n_dims = 2,
                .shape = .{ head_dim * n_heads, d_model, 0, 0, 0, 0, 0, 0 },
                .data_ptr = null,
                .data_size = 0,
            },
            .w1 = &tensor.Tensor{
                .dtype = .f32,
                .n_dims = 2,
                .shape = .{ d_model, d_ff, 0, 0, 0, 0, 0, 0 },
                .data_ptr = null,
                .data_size = 0,
            },
            .w2 = &tensor.Tensor{
                .dtype = .f32,
                .n_dims = 2,
                .shape = .{ d_ff, d_model, 0, 0, 0, 0, 0, 0 },
                .data_ptr = null,
                .data_size = 0,
            },
            .w3 = &tensor.Tensor{
                .dtype = .f32,
                .n_dims = 2,
                .shape = .{ d_model, d_ff, 0, 0, 0, 0, 0, 0 },
                .data_ptr = null,
                .data_size = 0,
            },
        } },
    };

    var loaded_model = weights.LoadedModel{
        .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
        .config = config,
        .ln_final = tensor.Tensor{
            .dtype = .f32,
            .n_dims = 1,
            .shape = .{ d_model, 0, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .lm_head = tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ vocab_size, d_model, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .token_embeddings = tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ vocab_size, d_model, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .blocks = &blocks,
        .original_weight_dtype = .f32,
    };
    defer loaded_model.arena.deinit();

    var buffer: std.ArrayList(u8) = .{};
    defer buffer.deinit(std.testing.allocator);

    var any_writer = std.io.AnyWriter{ .context = &buffer, .writeFn = struct {
        fn writeFn(context: *const anyopaque, bytes: []const u8) anyerror!usize {
            const buf: *std.ArrayList(u8) = @ptrCast(@alignCast(@constCast(context)));
            try buf.appendSlice(std.testing.allocator, bytes);
            return bytes.len;
        }
    }.writeFn };

    var reporter = Reporter.init(&any_writer, false);

    // Should succeed
    try validateCommon(&reporter, &loaded_model);
}

test "validateCommon: token_embeddings wrong dimensions" {
    // Create model with invalid token_embeddings (1D instead of 2D)
    const d_model: i32 = 64;
    const vocab_size: i32 = 1000;

    const config = tensor.ModelConfig{
        .vocab_size = vocab_size,
        .d_model = d_model,
        .n_layers = 1,
        .n_heads = 2,
        .n_kv_groups = 2,
        .d_ff = 256,
        .max_seq_len = 512,
        .head_dim = 32,
        .rope_theta = 10000.0,
        .norm_eps = 1e-5,
        .gaffine_group_size = 128,
    };

    const InferenceBackend = @import("../../inference/backend/cpu/executor/weights.zig");
    var blocks = [_]InferenceBackend.BlockWeights{.{ .attention_mlp = .{
        .ln1_weight = &tensor.Tensor{
            .dtype = .f32,
            .n_dims = 1,
            .shape = .{ d_model, 0, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .ln2_weight = &tensor.Tensor{
            .dtype = .f32,
            .n_dims = 1,
            .shape = .{ d_model, 0, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .q_proj = &tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ d_model, 64, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .k_proj = &tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ d_model, 64, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .v_proj = &tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ d_model, 64, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .o_proj = &tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ 64, d_model, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .w1 = &tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ d_model, 256, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .w2 = &tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ 256, d_model, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .w3 = &tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ d_model, 256, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
    } }};

    var loaded_model = weights.LoadedModel{
        .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
        .config = config,
        .ln_final = tensor.Tensor{
            .dtype = .f32,
            .n_dims = 1,
            .shape = .{ d_model, 0, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .lm_head = tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ vocab_size, d_model, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .token_embeddings = tensor.Tensor{
            .dtype = .f32,
            .n_dims = 1, // Wrong! Should be 2D
            .shape = .{ vocab_size, 0, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .blocks = &blocks,
        .original_weight_dtype = .f32,
    };
    defer loaded_model.arena.deinit();

    var buffer: std.ArrayList(u8) = .{};
    defer buffer.deinit(std.testing.allocator);

    var any_writer = std.io.AnyWriter{ .context = &buffer, .writeFn = struct {
        fn writeFn(context: *const anyopaque, bytes: []const u8) anyerror!usize {
            const buf: *std.ArrayList(u8) = @ptrCast(@alignCast(@constCast(context)));
            try buf.appendSlice(std.testing.allocator, bytes);
            return bytes.len;
        }
    }.writeFn };

    var reporter = Reporter.init(&any_writer, false);

    // Should fail
    try std.testing.expectError(Error.ValidationFailed, validateCommon(&reporter, &loaded_model));
    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "token_embeddings not 2D") != null);
}

test "validateCommon: token_embeddings shape mismatch" {
    // Create model with wrong token_embeddings shape
    const d_model: i32 = 64;
    const vocab_size: i32 = 1000;

    const config = tensor.ModelConfig{
        .vocab_size = vocab_size,
        .d_model = d_model,
        .n_layers = 1,
        .n_heads = 2,
        .n_kv_groups = 2,
        .d_ff = 256,
        .max_seq_len = 512,
        .head_dim = 32,
        .rope_theta = 10000.0,
        .norm_eps = 1e-5,
        .gaffine_group_size = 128,
    };

    const InferenceBackend = @import("../../inference/backend/cpu/executor/weights.zig");
    var blocks = [_]InferenceBackend.BlockWeights{.{ .attention_mlp = .{
        .ln1_weight = &tensor.Tensor{
            .dtype = .f32,
            .n_dims = 1,
            .shape = .{ d_model, 0, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .ln2_weight = &tensor.Tensor{
            .dtype = .f32,
            .n_dims = 1,
            .shape = .{ d_model, 0, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .q_proj = &tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ d_model, 64, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .k_proj = &tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ d_model, 64, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .v_proj = &tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ d_model, 64, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .o_proj = &tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ 64, d_model, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .w1 = &tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ d_model, 256, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .w2 = &tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ 256, d_model, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .w3 = &tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ d_model, 256, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
    } }};

    var loaded_model = weights.LoadedModel{
        .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
        .config = config,
        .ln_final = tensor.Tensor{
            .dtype = .f32,
            .n_dims = 1,
            .shape = .{ d_model, 0, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .lm_head = tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ vocab_size, d_model, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .token_embeddings = tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ vocab_size, 128, 0, 0, 0, 0, 0, 0 }, // Wrong d_model (128 vs 64)
            .data_ptr = null,
            .data_size = 0,
        },
        .blocks = &blocks,
        .original_weight_dtype = .f32,
    };
    defer loaded_model.arena.deinit();

    var buffer: std.ArrayList(u8) = .{};
    defer buffer.deinit(std.testing.allocator);

    var any_writer = std.io.AnyWriter{ .context = &buffer, .writeFn = struct {
        fn writeFn(context: *const anyopaque, bytes: []const u8) anyerror!usize {
            const buf: *std.ArrayList(u8) = @ptrCast(@alignCast(@constCast(context)));
            try buf.appendSlice(std.testing.allocator, bytes);
            return bytes.len;
        }
    }.writeFn };

    var reporter = Reporter.init(&any_writer, false);

    // Should fail
    try std.testing.expectError(Error.ValidationFailed, validateCommon(&reporter, &loaded_model));
    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "token_embeddings shape mismatch") != null);
}

test "validateCommon: blocks count mismatch" {
    // Create model with wrong number of blocks
    const d_model: i32 = 64;
    const vocab_size: i32 = 1000;
    const n_layers: i32 = 3; // Config says 3 layers

    const config = tensor.ModelConfig{
        .vocab_size = vocab_size,
        .d_model = d_model,
        .n_layers = n_layers,
        .n_heads = 2,
        .n_kv_groups = 2,
        .d_ff = 256,
        .max_seq_len = 512,
        .head_dim = 32,
        .rope_theta = 10000.0,
        .norm_eps = 1e-5,
        .gaffine_group_size = 128,
    };

    const InferenceBackend = @import("../../inference/backend/cpu/executor/weights.zig");
    // Only provide 1 block instead of 3
    var blocks = [_]InferenceBackend.BlockWeights{.{ .attention_mlp = .{
        .ln1_weight = &tensor.Tensor{
            .dtype = .f32,
            .n_dims = 1,
            .shape = .{ d_model, 0, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .ln2_weight = &tensor.Tensor{
            .dtype = .f32,
            .n_dims = 1,
            .shape = .{ d_model, 0, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .q_proj = &tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ d_model, 64, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .k_proj = &tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ d_model, 64, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .v_proj = &tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ d_model, 64, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .o_proj = &tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ 64, d_model, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .w1 = &tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ d_model, 256, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .w2 = &tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ 256, d_model, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .w3 = &tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ d_model, 256, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
    } }};

    var loaded_model = weights.LoadedModel{
        .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
        .config = config,
        .ln_final = tensor.Tensor{
            .dtype = .f32,
            .n_dims = 1,
            .shape = .{ d_model, 0, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .lm_head = tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ vocab_size, d_model, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .token_embeddings = tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ vocab_size, d_model, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .blocks = &blocks, // Only 1 block, but config expects 3
        .original_weight_dtype = .f32,
    };
    defer loaded_model.arena.deinit();

    var buffer: std.ArrayList(u8) = .{};
    defer buffer.deinit(std.testing.allocator);

    var any_writer = std.io.AnyWriter{ .context = &buffer, .writeFn = struct {
        fn writeFn(context: *const anyopaque, bytes: []const u8) anyerror!usize {
            const buf: *std.ArrayList(u8) = @ptrCast(@alignCast(@constCast(context)));
            try buf.appendSlice(std.testing.allocator, bytes);
            return bytes.len;
        }
    }.writeFn };

    var reporter = Reporter.init(&any_writer, false);

    // Should fail
    try std.testing.expectError(Error.ValidationFailed, validateCommon(&reporter, &loaded_model));
    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "blocks len mismatch") != null);
}

test "validateCommon: layer ln1 shape mismatch" {
    // Test validation of layer normalization weights
    const d_model: i32 = 64;
    const vocab_size: i32 = 1000;

    const config = tensor.ModelConfig{
        .vocab_size = vocab_size,
        .d_model = d_model,
        .n_layers = 1,
        .n_heads = 2,
        .n_kv_groups = 2,
        .d_ff = 256,
        .max_seq_len = 512,
        .head_dim = 32,
        .rope_theta = 10000.0,
        .norm_eps = 1e-5,
        .gaffine_group_size = 128,
    };

    const InferenceBackend = @import("../../inference/backend/cpu/executor/weights.zig");
    var blocks = [_]InferenceBackend.BlockWeights{.{
        .attention_mlp = .{
            .ln1_weight = &tensor.Tensor{
                .dtype = .f32,
                .n_dims = 1,
                .shape = .{ 128, 0, 0, 0, 0, 0, 0, 0 }, // Wrong! Should be d_model=64
                .data_ptr = null,
                .data_size = 0,
            },
            .ln2_weight = &tensor.Tensor{
                .dtype = .f32,
                .n_dims = 1,
                .shape = .{ d_model, 0, 0, 0, 0, 0, 0, 0 },
                .data_ptr = null,
                .data_size = 0,
            },
            .q_proj = &tensor.Tensor{
                .dtype = .f32,
                .n_dims = 2,
                .shape = .{ d_model, 64, 0, 0, 0, 0, 0, 0 },
                .data_ptr = null,
                .data_size = 0,
            },
            .k_proj = &tensor.Tensor{
                .dtype = .f32,
                .n_dims = 2,
                .shape = .{ d_model, 64, 0, 0, 0, 0, 0, 0 },
                .data_ptr = null,
                .data_size = 0,
            },
            .v_proj = &tensor.Tensor{
                .dtype = .f32,
                .n_dims = 2,
                .shape = .{ d_model, 64, 0, 0, 0, 0, 0, 0 },
                .data_ptr = null,
                .data_size = 0,
            },
            .o_proj = &tensor.Tensor{
                .dtype = .f32,
                .n_dims = 2,
                .shape = .{ 64, d_model, 0, 0, 0, 0, 0, 0 },
                .data_ptr = null,
                .data_size = 0,
            },
            .w1 = &tensor.Tensor{
                .dtype = .f32,
                .n_dims = 2,
                .shape = .{ d_model, 256, 0, 0, 0, 0, 0, 0 },
                .data_ptr = null,
                .data_size = 0,
            },
            .w2 = &tensor.Tensor{
                .dtype = .f32,
                .n_dims = 2,
                .shape = .{ 256, d_model, 0, 0, 0, 0, 0, 0 },
                .data_ptr = null,
                .data_size = 0,
            },
            .w3 = &tensor.Tensor{
                .dtype = .f32,
                .n_dims = 2,
                .shape = .{ d_model, 256, 0, 0, 0, 0, 0, 0 },
                .data_ptr = null,
                .data_size = 0,
            },
        },
    }};

    var loaded_model = weights.LoadedModel{
        .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
        .config = config,
        .ln_final = tensor.Tensor{
            .dtype = .f32,
            .n_dims = 1,
            .shape = .{ d_model, 0, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .lm_head = tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ vocab_size, d_model, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .token_embeddings = tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ vocab_size, d_model, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .blocks = &blocks,
        .original_weight_dtype = .f32,
    };
    defer loaded_model.arena.deinit();

    var buffer: std.ArrayList(u8) = .{};
    defer buffer.deinit(std.testing.allocator);

    var any_writer = std.io.AnyWriter{ .context = &buffer, .writeFn = struct {
        fn writeFn(context: *const anyopaque, bytes: []const u8) anyerror!usize {
            const buf: *std.ArrayList(u8) = @ptrCast(@alignCast(@constCast(context)));
            try buf.appendSlice(std.testing.allocator, bytes);
            return bytes.len;
        }
    }.writeFn };

    var reporter = Reporter.init(&any_writer, false);

    // Should fail
    try std.testing.expectError(Error.ValidationFailed, validateCommon(&reporter, &loaded_model));
    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "layer 0 ln1 shape mismatch") != null);
}

test "validateCommon: missing q/k/v without fused qkv" {
    // Test that validation fails when q/k/v projections are missing without fused qkv
    const d_model: i32 = 64;
    const vocab_size: i32 = 1000;

    const config = tensor.ModelConfig{
        .vocab_size = vocab_size,
        .d_model = d_model,
        .n_layers = 1,
        .n_heads = 2,
        .n_kv_groups = 2,
        .d_ff = 256,
        .max_seq_len = 512,
        .head_dim = 32,
        .rope_theta = 10000.0,
        .norm_eps = 1e-5,
        .gaffine_group_size = 128,
    };

    const InferenceBackend = @import("../../inference/backend/cpu/executor/weights.zig");
    var blocks = [_]InferenceBackend.BlockWeights{.{
        .attention_mlp = .{
            .ln1_weight = &tensor.Tensor{
                .dtype = .f32,
                .n_dims = 1,
                .shape = .{ d_model, 0, 0, 0, 0, 0, 0, 0 },
                .data_ptr = null,
                .data_size = 0,
            },
            .ln2_weight = &tensor.Tensor{
                .dtype = .f32,
                .n_dims = 1,
                .shape = .{ d_model, 0, 0, 0, 0, 0, 0, 0 },
                .data_ptr = null,
                .data_size = 0,
            },
            .q_proj = null, // Missing!
            .k_proj = null, // Missing!
            .v_proj = null, // Missing!
            .o_proj = &tensor.Tensor{
                .dtype = .f32,
                .n_dims = 2,
                .shape = .{ 64, d_model, 0, 0, 0, 0, 0, 0 },
                .data_ptr = null,
                .data_size = 0,
            },
            .w1 = &tensor.Tensor{
                .dtype = .f32,
                .n_dims = 2,
                .shape = .{ d_model, 256, 0, 0, 0, 0, 0, 0 },
                .data_ptr = null,
                .data_size = 0,
            },
            .w2 = &tensor.Tensor{
                .dtype = .f32,
                .n_dims = 2,
                .shape = .{ 256, d_model, 0, 0, 0, 0, 0, 0 },
                .data_ptr = null,
                .data_size = 0,
            },
            .w3 = &tensor.Tensor{
                .dtype = .f32,
                .n_dims = 2,
                .shape = .{ d_model, 256, 0, 0, 0, 0, 0, 0 },
                .data_ptr = null,
                .data_size = 0,
            },
            .fused = .{ .qkv_proj = null }, // No fused QKV either
        },
    }};

    var loaded_model = weights.LoadedModel{
        .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
        .config = config,
        .ln_final = tensor.Tensor{
            .dtype = .f32,
            .n_dims = 1,
            .shape = .{ d_model, 0, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .lm_head = tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ vocab_size, d_model, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .token_embeddings = tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ vocab_size, d_model, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .blocks = &blocks,
        .original_weight_dtype = .f32,
    };
    defer loaded_model.arena.deinit();

    var buffer: std.ArrayList(u8) = .{};
    defer buffer.deinit(std.testing.allocator);

    var any_writer = std.io.AnyWriter{ .context = &buffer, .writeFn = struct {
        fn writeFn(context: *const anyopaque, bytes: []const u8) anyerror!usize {
            const buf: *std.ArrayList(u8) = @ptrCast(@alignCast(@constCast(context)));
            try buf.appendSlice(std.testing.allocator, bytes);
            return bytes.len;
        }
    }.writeFn };

    var reporter = Reporter.init(&any_writer, false);

    // Should fail
    try std.testing.expectError(Error.ValidationFailed, validateCommon(&reporter, &loaded_model));
    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "missing q/k/v weights") != null);
}

test "validateCommon: lm_head not 2D" {
    // Test that lm_head must be 2D
    const d_model: i32 = 64;
    const vocab_size: i32 = 1000;

    const config = tensor.ModelConfig{
        .vocab_size = vocab_size,
        .d_model = d_model,
        .n_layers = 1,
        .n_heads = 2,
        .n_kv_groups = 2,
        .d_ff = 256,
        .max_seq_len = 512,
        .head_dim = 32,
        .rope_theta = 10000.0,
        .norm_eps = 1e-5,
        .gaffine_group_size = 128,
    };

    const InferenceBackend = @import("../../inference/backend/cpu/executor/weights.zig");
    var blocks = [_]InferenceBackend.BlockWeights{.{ .attention_mlp = .{
        .ln1_weight = &tensor.Tensor{
            .dtype = .f32,
            .n_dims = 1,
            .shape = .{ d_model, 0, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .ln2_weight = &tensor.Tensor{
            .dtype = .f32,
            .n_dims = 1,
            .shape = .{ d_model, 0, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .q_proj = &tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ d_model, 64, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .k_proj = &tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ d_model, 64, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .v_proj = &tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ d_model, 64, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .o_proj = &tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ 64, d_model, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .w1 = &tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ d_model, 256, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .w2 = &tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ 256, d_model, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .w3 = &tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ d_model, 256, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
    } }};

    var loaded_model = weights.LoadedModel{
        .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
        .config = config,
        .ln_final = tensor.Tensor{
            .dtype = .f32,
            .n_dims = 1,
            .shape = .{ d_model, 0, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .lm_head = tensor.Tensor{
            .dtype = .f32,
            .n_dims = 1, // Wrong! Should be 2D
            .shape = .{ vocab_size, 0, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .token_embeddings = tensor.Tensor{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ vocab_size, d_model, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
        },
        .blocks = &blocks,
        .original_weight_dtype = .f32,
    };
    defer loaded_model.arena.deinit();

    var buffer: std.ArrayList(u8) = .{};
    defer buffer.deinit(std.testing.allocator);

    var any_writer = std.io.AnyWriter{ .context = &buffer, .writeFn = struct {
        fn writeFn(context: *const anyopaque, bytes: []const u8) anyerror!usize {
            const buf: *std.ArrayList(u8) = @ptrCast(@alignCast(@constCast(context)));
            try buf.appendSlice(std.testing.allocator, bytes);
            return bytes.len;
        }
    }.writeFn };

    var reporter = Reporter.init(&any_writer, false);

    // Should fail
    try std.testing.expectError(Error.ValidationFailed, validateCommon(&reporter, &loaded_model));
    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "lm_head not 2D") != null);
}
