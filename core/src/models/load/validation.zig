//! Model Validation
//!
//! Validates loaded models for correctness (tensor shapes, config consistency).

const std = @import("std");
const weights = @import("weights.zig");
const tensor = @import("compute_pkg").tensor;
const config_types = @import("../config/types.zig");
const block_geometry = @import("../block_geometry.zig");
const log = @import("log_pkg");
const op_types = @import("models_pkg").op_types;
const manifest_mod = @import("../manifest.zig");

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

pub fn validateStageModel(stage: *weights.LoadedStageModel) !void {
    if (stage.layer_range.start >= stage.layer_range.end) return error.EmptyStageRange;
    if (stage.blocks.len != stage.layer_range.end - stage.layer_range.start) {
        return error.StageResidencyMismatch;
    }
    if (stage.config.n_layers < 0) return error.InvalidStageRange;
    const configured_layers: usize = @intCast(stage.config.n_layers);
    if (stage.layer_range.end > configured_layers) return error.InvalidStageRange;
    if (stage.manifest.layer_count != configured_layers) return error.StageResidencyMismatch;
    for (stage.blocks, 0..) |_, offset| {
        const original_idx = try stage.originalLayerIndex(offset);
        if (original_idx != stage.layer_range.start + offset) return error.InvalidStageLayerOffset;
    }

    var effective_roles = stage.requested_roles;
    if (stage.lm_head_uses_token_embeddings) effective_roles.include_token_embeddings = true;

    try validateStageRoleHydrated(stage, .token_embeddings, effective_roles.include_token_embeddings);
    try validateStageRoleHydrated(stage, .final_norm, effective_roles.include_final_norm);
    try validateStageRoleHydrated(stage, .lm_head, effective_roles.include_lm_head);
    try validateStageRoleHydrated(stage, .embedding_side, effective_roles.include_embedding_side);
    try validateStageRoleHydrated(stage, .vision_side, effective_roles.include_vision_side);
    try validateStageRoleHydrated(stage, .architecture_side, effective_roles.include_architecture_side);
    try validateStageRoleHydrated(stage, .unclassified_global, effective_roles.include_unclassified_global);

    const expected = try stage.manifest.stageResidencyReport(effective_roles.toResidencyRequest(stage.layer_range));
    if (expected.layer_start != stage.residency.layer_start or expected.layer_end != stage.residency.layer_end) {
        return error.StageResidencyMismatch;
    }
    if (expected.total_checkpoint_bytes != stage.residency.total_checkpoint_bytes) {
        return error.StageResidencyMismatch;
    }
    inline for (std.meta.fields(manifest_mod.TensorRole)) |field| {
        const role: manifest_mod.TensorRole = @field(manifest_mod.TensorRole, field.name);
        if (expected.bytesForRole(role) != stage.residency.bytesForRole(role)) {
            return error.StageResidencyMismatch;
        }
    }

    try validateStageShapeContract(stage);
}

fn validateStageRoleHydrated(
    stage: *const weights.LoadedStageModel,
    role: manifest_mod.TensorRole,
    requested: bool,
) !void {
    if (!requested) return;
    const planned_bytes = stage.residency.bytesForRole(role);
    if (planned_bytes == 0) return;
    const hydrated = switch (role) {
        .token_embeddings => stage.token_embeddings != null,
        .final_norm => stage.ln_final != null,
        .lm_head => stage.lm_head != null,
        .embedding_side => stage.position_embeddings != null or
            stage.token_type_embeddings != null or
            stage.embedding_norm_weight != null or
            stage.embedding_norm_bias != null or
            stage.extra_global_role_bytes[@intFromEnum(role)] > 0,
        .vision_side, .architecture_side, .unclassified_global => stage.extra_global_role_bytes[@intFromEnum(role)] > 0,
        .decoder_layer => stage.blocks.len > 0,
        .quant_companion => true,
    };
    if (!hydrated) return error.MissingStageGlobalWeight;
}

fn validateStageShapeContract(stage: *weights.LoadedStageModel) Error!void {
    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var writer = fbs.writer().any();
    var reporter = Reporter.init(&writer, false);
    validateStageShapeContractCommon(&reporter, stage) catch |err| {
        const written = fbs.getWritten();
        if (written.len > 0) {
            log.err("load", "Stage model validation failed", .{
                .detail = written,
            }, @src());
        }
        return err;
    };
}

fn validateStageShapeContractCommon(reporter: *Reporter, stage: *weights.LoadedStageModel) Error!void {
    const d_model: usize = @intCast(stage.config.d_model);
    const vocab_size: usize = @intCast(stage.config.vocab_size);

    var validation_config = stage.config;
    validation_config.n_layers = @intCast(stage.blocks.len);
    var validation_model = weights.LoadedModel{
        .arena = std.heap.ArenaAllocator.init(std.heap.page_allocator),
        .config = validation_config,
        .runtime = stage.runtime,
        .st = null,
        .ln_final = stage.ln_final,
        .lm_head = stage.lm_head,
        .token_embeddings = stage.token_embeddings orelse shapeOnlyTensor2D(vocab_size, d_model),
        .position_embeddings = stage.position_embeddings,
        .token_type_embeddings = stage.token_type_embeddings,
        .embedding_norm_weight = stage.embedding_norm_weight,
        .embedding_norm_bias = stage.embedding_norm_bias,
        .blocks = stage.blocks,
        .original_weight_dtype = stage.original_weight_dtype,
        .file_size = stage.file_size,
        .tensor_count = stage.tensor_count,
        .manifest = null,
    };
    defer validation_model.arena.deinit();

    try validateCommon(reporter, &validation_model);

    if (stage.ln_final) |ln_final| {
        if (!isVectorShape(&ln_final, d_model))
            return reporter.reportError("final norm shape mismatch (n_dims={}, shape0={}, expected={})", .{
                ln_final.n_dims, ln_final.shape[0], d_model,
            });
    }
}

fn shapeOnlyTensor2D(dim0: usize, dim1: usize) tensor.Tensor {
    var tensor_view = std.mem.zeroes(tensor.Tensor);
    tensor_view.dtype = .f32;
    tensor_view.n_dims = 2;
    tensor_view.shape[0] = @intCast(dim0);
    tensor_view.shape[1] = @intCast(dim1);
    return tensor_view;
}

fn validateCommon(reporter: *Reporter, loaded_model: *weights.LoadedModel) !void {
    const d_model: usize = @intCast(loaded_model.config.d_model);
    const d_ff: usize = @intCast(loaded_model.config.d_ff);
    const vocab_size: usize = @intCast(loaded_model.config.vocab_size);

    if (loaded_model.token_embeddings.n_dims != 2) return reporter.reportError("token_embeddings not 2D (n_dims={})", .{loaded_model.token_embeddings.n_dims});
    if (!is2DShapeMatch(&loaded_model.token_embeddings, vocab_size, d_model))
        return reporter.reportError("token_embeddings shape mismatch (shape=[{},{}], expected=[{},{}])", .{
            loaded_model.token_embeddings.shape[0], loaded_model.token_embeddings.shape[1], vocab_size, d_model,
        });

    if (loaded_model.blocks.len != @as(usize, @intCast(loaded_model.config.n_layers)))
        return reporter.reportError("blocks len mismatch (blocks={} config={})", .{ loaded_model.blocks.len, loaded_model.config.n_layers });

    const arena_allocator = loaded_model.arena.allocator();
    for (loaded_model.blocks, 0..) |*layer, layer_idx| {
        const block_weights_at_layer = weights.blocks.layerToBlockWeights(arena_allocator, layer) catch |err| {
            return reporter.reportError("failed to materialize layer {} block weights for validation ({s})", .{
                layer_idx,
                @errorName(err),
            });
        };
        switch (block_weights_at_layer) {
            .attention_mlp => |block| {
                if (!isVectorShape(block.ln1_weight, d_model))
                    return reporter.reportError("layer {} ln1 shape mismatch (n_dims={}, shape0={}, expected={})", .{
                        layer_idx, block.ln1_weight.n_dims, block.ln1_weight.shape[0], d_model,
                    });
                if (!isVectorShape(block.ln2_weight, d_model))
                    return reporter.reportError("layer {} ln2 shape mismatch (n_dims={}, shape0={}, expected={})", .{
                        layer_idx, block.ln2_weight.n_dims, block.ln2_weight.shape[0], d_model,
                    });

                const inferred_shape = block_geometry.inferAttentionShape(loaded_model.config, block);
                const layer_n_heads = inferred_shape.n_heads;
                const layer_n_kv_heads = inferred_shape.n_kv_heads;
                const layer_head_dim = inferred_shape.head_dim;
                const layer_d_ff = block_geometry.inferAttentionDff(loaded_model.config, block);
                const q_out: usize = layer_n_heads * layer_head_dim;
                const kv_out: usize = layer_n_kv_heads * layer_head_dim;

                // Check for MLA (Multi-Latent Attention) weights first
                const is_mla = block.q_a_proj != null;
                if (is_mla) {
                    // MLA has compressed Q/KV projections - skip standard Q/K/V validation
                    // Shape validation for MLA weights is model-specific (handled by expected_shape in architecture metadata)
                } else if (block.fused.qkv_proj) |qkv| {
                    const qkv_out: usize = q_out + 2 * kv_out;
                    const gated_qkv_out: usize = q_out * 2 + 2 * kv_out;
                    const expected_qkv_out = if (block.attention_config.query_gate) gated_qkv_out else qkv_out;
                    if (!is2DShapeMatch(&qkv, d_model, expected_qkv_out))
                        return reporter.reportError("layer {} qkv_proj shape mismatch (shape=[{},{}], expected=[{},{}])", .{
                            layer_idx, qkv.shape[0], qkv.shape[1], d_model, expected_qkv_out,
                        });
                } else {
                    if (block.q_proj == null or block.k_proj == null or block.v_proj == null)
                        return reporter.reportError("layer {} missing q/k/v weights without fused qkv", .{layer_idx});
                    const expected_q_proj_out = if (block.attention_config.query_gate) q_out * 2 else q_out;
                    if (!is2DShapeMatch(block.q_proj.?, d_model, expected_q_proj_out))
                        return reporter.reportError("layer {} q_proj shape mismatch (shape=[{},{}], expected=[{},{}])", .{
                            layer_idx, block.q_proj.?.shape[0], block.q_proj.?.shape[1], d_model, expected_q_proj_out,
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
                        if (!is2DShapeMatch(&gate_up, d_model, layer_d_ff * 2))
                            return reporter.reportError("layer {} gate_up shape mismatch (shape=[{},{}], expected=[{},{}])", .{
                                layer_idx, gate_up.shape[0], gate_up.shape[1], d_model, layer_d_ff * 2,
                            });
                    } else {
                        // Dense FFN: w1 (gate/dense_in) and w2 (down/dense_out) required.
                        // w3 (up projection) is optional - absent in non-gated MLPs (BERT/MiniLM).
                        if (block.w1 == null or block.w2 == null)
                            return reporter.reportError("layer {} missing dense FFN weights", .{layer_idx});
                        if (!is2DShapeMatch(block.w1.?, d_model, layer_d_ff))
                            return reporter.reportError("layer {} w1 shape mismatch (shape=[{},{}], expected=[{},{}])", .{
                                layer_idx, block.w1.?.shape[0], block.w1.?.shape[1], d_model, layer_d_ff,
                            });
                        if (block.w3) |w3| {
                            if (!is2DShapeMatch(w3, d_model, layer_d_ff))
                                return reporter.reportError("layer {} w3 shape mismatch (shape=[{},{}], expected=[{},{}])", .{
                                    layer_idx, w3.shape[0], w3.shape[1], d_model, layer_d_ff,
                                });
                        }
                        if (!is2DShapeMatch(block.w2.?, layer_d_ff, d_model))
                            return reporter.reportError("layer {} w2 shape mismatch (shape=[{},{}], expected=[{},{}])", .{
                                layer_idx, block.w2.?.shape[0], block.w2.?.shape[1], layer_d_ff, d_model,
                            });
                    }
                }

                if (block.q_norm) |q_norm| {
                    if (!isVectorShape(q_norm, layer_head_dim))
                        return reporter.reportError("layer {} q_norm shape mismatch (n_dims={}, shape0={}, expected={})", .{
                            layer_idx, q_norm.n_dims, q_norm.shape[0], layer_head_dim,
                        });
                }
                if (block.k_norm) |k_norm| {
                    if (!isVectorShape(k_norm, layer_head_dim))
                        return reporter.reportError("layer {} k_norm shape mismatch (n_dims={}, shape0={}, expected={})", .{
                            layer_idx, k_norm.n_dims, k_norm.shape[0], layer_head_dim,
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
            .gated_delta => |block| {
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

                const d_inner: usize = @intCast(block.config.n_heads * block.config.d_head);
                const d_conv: usize = @intCast(block.config.d_conv);
                if (d_conv == 0) return reporter.reportError("layer {} gated_delta d_conv invalid (0)", .{layer_idx});
                const conv_numel = tensorNumel(block.weights.conv1d_weight);
                if ((conv_numel % d_conv) != 0)
                    return reporter.reportError("layer {} gated_delta conv1d_weight numel {} not divisible by d_conv {}", .{
                        layer_idx, conv_numel, d_conv,
                    });
                const qkv_out = conv_numel / d_conv;
                const proj_out = qkv_out + d_inner + (2 * @as(usize, @intCast(block.config.n_heads)));
                if (!isProjected2DShapeMatch(block.weights.in_proj, d_model, proj_out))
                    return reporter.reportError("layer {} gated_delta in_proj shape mismatch (shape=[{},{}], expected d_model={} proj_out={})", .{
                        layer_idx, block.weights.in_proj.shape[0], block.weights.in_proj.shape[1], d_model, proj_out,
                    });
                if (!is2DShapeMatch(block.weights.out_proj, d_inner, d_model))
                    return reporter.reportError("layer {} gated_delta out_proj shape mismatch (shape=[{},{}], expected=[{},{}])", .{
                        layer_idx, block.weights.out_proj.shape[0], block.weights.out_proj.shape[1], d_inner, d_model,
                    });
                if (!isConvWeightShapeMatch(block.weights.conv1d_weight, qkv_out, d_conv))
                    return reporter.reportError("layer {} gated_delta conv1d_weight shape mismatch", .{layer_idx});
                if (block.weights.conv1d_bias) |bias| {
                    if (!isVectorShape(bias, qkv_out))
                        return reporter.reportError("layer {} gated_delta conv1d_bias shape mismatch", .{layer_idx});
                }
                if (!isVectorShape(block.weights.A_log, @intCast(block.config.n_heads)))
                    return reporter.reportError("layer {} gated_delta A_log shape mismatch", .{layer_idx});
                if (block.weights.dt_bias) |dt_bias| {
                    if (!isVectorShape(dt_bias, @intCast(block.config.n_heads)))
                        return reporter.reportError("layer {} gated_delta dt_bias shape mismatch", .{layer_idx});
                }
                if (block.weights.norm_weight) |norm_weight| {
                    const n_head_len: usize = @intCast(block.config.d_head);
                    if (!isVectorShape(norm_weight, d_inner) and !isVectorShape(norm_weight, n_head_len))
                        return reporter.reportError("layer {} gated_delta norm_weight shape mismatch", .{layer_idx});
                }
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

fn isProjected2DShapeMatch(tensor_view: *const tensor.Tensor, d_model: usize, out_dim: usize) bool {
    return tensor_view.n_dims == 2 and
        ((tensor_view.shape[0] == d_model and tensor_view.shape[1] == out_dim) or
            (tensor_view.shape[0] == out_dim and tensor_view.shape[1] == d_model));
}

fn tensorNumel(tensor_view: *const tensor.Tensor) usize {
    var total: usize = 1;
    var idx: usize = 0;
    while (idx < @as(usize, @intCast(tensor_view.n_dims))) : (idx += 1) {
        total *= @intCast(tensor_view.shape[idx]);
    }
    return total;
}

fn isConvWeightShapeMatch(tensor_view: *const tensor.Tensor, channels: usize, d_conv: usize) bool {
    if (tensor_view.n_dims != 2 and tensor_view.n_dims != 3) return false;
    return tensorNumel(tensor_view) == channels * d_conv;
}

fn isVectorShape(tensor_view: *const tensor.Tensor, len: usize) bool {
    if (tensor_view.n_dims == 1) return tensor_view.shape[0] == len;
    if (tensor_view.n_dims == 2) return (tensor_view.shape[0] == len and tensor_view.shape[1] == 1) or (tensor_view.shape[0] == 1 and tensor_view.shape[1] == len);
    return false;
}

fn allocTensorCopy(allocator: std.mem.Allocator, t: tensor.Tensor) !*const tensor.Tensor {
    const copy = try allocator.create(tensor.Tensor);
    copy.* = t;
    return copy;
}

fn legacyBlockToLayer(
    allocator: std.mem.Allocator,
    legacy_block: weights.blocks.BlockWeights,
) !weights.blocks.LayerWeights {
    var map: weights.blocks.WeightMap = .{};

    switch (legacy_block) {
        .attention_mlp => |block| {
            try map.put(allocator, "input_layernorm.weight", block.ln1_weight);
            try map.put(allocator, "post_attention_layernorm.weight", block.ln2_weight);
            if (block.q_proj) |q| try map.put(allocator, "self_attn.q_proj.weight", q);
            if (block.k_proj) |k| try map.put(allocator, "self_attn.k_proj.weight", k);
            if (block.v_proj) |v| try map.put(allocator, "self_attn.v_proj.weight", v);
            try map.put(allocator, "self_attn.o_proj.weight", block.o_proj);
            if (block.fused.qkv_proj) |qkv| {
                try map.put(allocator, "self_attn.qkv_proj.weight", try allocTensorCopy(allocator, qkv));
            }
            if (block.w1) |w1| try map.put(allocator, "mlp.gate_proj.weight", w1);
            if (block.w2) |w2| try map.put(allocator, "mlp.down_proj.weight", w2);
            if (block.w3) |w3| try map.put(allocator, "mlp.up_proj.weight", w3);
            if (block.fused.gate_up) |gate_up| {
                try map.put(allocator, "mlp.gate_up_proj.weight", try allocTensorCopy(allocator, gate_up));
            }
            if (block.q_norm) |q_norm| try map.put(allocator, "self_attn.q_norm.weight", q_norm);
            if (block.k_norm) |k_norm| try map.put(allocator, "self_attn.k_norm.weight", k_norm);
            if (block.pre_ffn_norm) |pre_ffn_norm| try map.put(allocator, "pre_feedforward_layernorm.weight", pre_ffn_norm);
            if (block.post_ffn_norm) |post_ffn_norm| try map.put(allocator, "post_feedforward_layernorm.weight", post_ffn_norm);
            if (block.q_a_proj) |t| try map.put(allocator, "self_attn.q_a_proj.weight", t);
            if (block.q_a_norm) |t| try map.put(allocator, "self_attn.q_a_layernorm.weight", t);
            if (block.q_b_proj) |t| try map.put(allocator, "self_attn.q_b_proj.weight", t);
            if (block.kv_a_proj) |t| try map.put(allocator, "self_attn.kv_a_proj_with_mqa.weight", t);
            if (block.kv_a_norm) |t| try map.put(allocator, "self_attn.kv_a_layernorm.weight", t);
            if (block.kv_b_proj) |t| try map.put(allocator, "self_attn.kv_b_proj.weight", t);

            var kernel_meta = op_types.KernelMeta{
                .is_causal = block.is_causal,
            };
            if (block.mla_config) |mla_cfg| {
                kernel_meta.mla_config = .{
                    .q_lora_rank = @intCast(mla_cfg.q_lora_rank),
                    .kv_lora_rank = @intCast(mla_cfg.kv_lora_rank),
                    .qk_head_dim = @intCast(mla_cfg.qk_head_dim),
                    .qk_rope_head_dim = @intCast(mla_cfg.qk_rope_head_dim),
                    .qk_nope_head_dim = @intCast(mla_cfg.qk_nope_head_dim),
                    .v_head_dim = @intCast(mla_cfg.v_head_dim),
                    .rope_interleave = mla_cfg.rope_interleave,
                };
            }

            return .{
                .block_type = .attention_mlp,
                .weight_map = map,
                .map_context = .{
                    .sliding_window = block.sliding_window,
                    .kernel_meta = kernel_meta,
                    .mamba_config = null,
                    .shortconv_config = null,
                    .num_experts = if (block.moe_weights) |moe_w| moe_w.num_experts else 0,
                    .experts_per_token = if (block.moe_weights) |moe_w| moe_w.experts_per_token else 0,
                    .allocator = null,
                },
            };
        },
        .mamba => |block| {
            try map.put(allocator, "input_layernorm.weight", block.ln1_weight);
            try map.put(allocator, "mixer.in_proj.weight", block.weights.in_proj);
            try map.put(allocator, "mixer.conv1d.weight", block.weights.conv1d_weight);
            try map.put(allocator, "mixer.A_log", block.weights.A_log);
            try map.put(allocator, "mixer.D", block.weights.D);
            try map.put(allocator, "mixer.out_proj.weight", block.weights.out_proj);
            if (block.ln2_weight) |ln2| try map.put(allocator, "post_attention_layernorm.weight", ln2);
            if (block.weights.conv1d_bias) |b| try map.put(allocator, "mixer.conv1d.bias", b);
            if (block.weights.dt_bias) |b| try map.put(allocator, "mixer.dt_bias", b);
            if (block.weights.norm_weight) |w| try map.put(allocator, "mixer.norm.weight", w);

            return .{
                .block_type = .mamba,
                .weight_map = map,
                .map_context = .{
                    .kernel_meta = .{},
                    .mamba_config = block.config,
                    .shortconv_config = null,
                    .allocator = null,
                },
            };
        },
        .gated_delta => |block| {
            try map.put(allocator, "input_layernorm.weight", block.ln1_weight);
            // Validation rematerialization emits the already-fused in-proj under
            // the fused loader name accepted by blockWeightsFromMap.
            try map.put(allocator, "mixer.in_proj.weight", block.weights.in_proj);
            try map.put(allocator, "linear_attn.conv1d.weight", block.weights.conv1d_weight);
            try map.put(allocator, "linear_attn.A_log", block.weights.A_log);
            try map.put(allocator, "linear_attn.out_proj.weight", block.weights.out_proj);
            if (block.ln2_weight) |ln2| try map.put(allocator, "post_attention_layernorm.weight", ln2);
            if (block.weights.conv1d_bias) |b| try map.put(allocator, "linear_attn.conv1d.bias", b);
            if (block.weights.dt_bias) |b| try map.put(allocator, "linear_attn.dt_bias", b);
            if (block.weights.norm_weight) |w| try map.put(allocator, "linear_attn.norm.weight", w);

            return .{
                .block_type = .gated_delta,
                .weight_map = map,
                .map_context = .{
                    .kernel_meta = .{},
                    .mamba_config = null,
                    .gated_delta_config = block.config,
                    .shortconv_config = null,
                    .allocator = null,
                },
            };
        },
        .shortconv => |block| {
            try map.put(allocator, "operator_norm.weight", block.ln1_weight);
            try map.put(allocator, "conv.in_proj.weight", block.weights.in_proj);
            try map.put(allocator, "conv.conv.weight", block.weights.conv1d_weight);
            try map.put(allocator, "conv.out_proj.weight", block.weights.out_proj);
            if (block.ln2_weight) |ln2| try map.put(allocator, "ffn_norm.weight", ln2);
            if (block.fused_gate_up) |fused| {
                if (fused.gate_up) |gate_up| {
                    try map.put(allocator, "feed_forward.gate_up_proj.weight", try allocTensorCopy(allocator, gate_up));
                }
            } else {
                if (block.w1) |w1| try map.put(allocator, "feed_forward.w1.weight", w1);
                if (block.w2) |w2| try map.put(allocator, "feed_forward.w2.weight", w2);
                if (block.w3) |w3| try map.put(allocator, "feed_forward.w3.weight", w3);
            }
            return .{
                .block_type = .shortconv,
                .weight_map = map,
                .map_context = .{
                    .kernel_meta = .{},
                    .mamba_config = null,
                    .shortconv_config = block.config,
                    .allocator = null,
                },
            };
        },
    }
}

fn legacyBlocksToLayers(
    allocator: std.mem.Allocator,
    legacy_blocks: []const weights.blocks.BlockWeights,
) ![]weights.blocks.LayerWeights {
    const layer_blocks = try allocator.alloc(weights.blocks.LayerWeights, legacy_blocks.len);
    errdefer allocator.free(layer_blocks);
    for (legacy_blocks, 0..) |legacy_block, idx| {
        layer_blocks[idx] = try legacyBlockToLayer(allocator, legacy_block);
    }
    return layer_blocks;
}

fn freeLayerBlocks(
    allocator: std.mem.Allocator,
    layer_blocks: []weights.blocks.LayerWeights,
) void {
    for (layer_blocks) |*layer| {
        layer.weight_map.deinit(allocator);
    }
    allocator.free(layer_blocks);
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

const validation_stage_shape_2d = [_]usize{ 4, 4 };

fn makeValidationStageManifest(
    allocator: std.mem.Allocator,
    layer_count: usize,
    entries: []const manifest_mod.TensorManifestEntry,
) !manifest_mod.ModelManifest {
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_allocator = arena.allocator();

    const stored_entries = try arena_allocator.alloc(manifest_mod.TensorManifestEntry, entries.len);
    @memcpy(stored_entries, entries);

    var role_bytes = [_]usize{0} ** manifest_mod.role_count;
    var total_checkpoint_bytes: usize = 0;
    for (stored_entries) |entry| {
        role_bytes[@intFromEnum(entry.role)] +|= entry.checkpoint_bytes;
        total_checkpoint_bytes +|= entry.checkpoint_bytes;
    }

    return .{
        .arena = arena,
        .architecture_id = "validation_stage",
        .layer_count = layer_count,
        .entries = stored_entries,
        .total_checkpoint_bytes = total_checkpoint_bytes,
        .role_bytes = role_bytes,
    };
}

fn stageBlockStorage() [1]weights.blocks.LayerWeights {
    return .{.{ .block_type = .attention_mlp, .weight_map = .{}, .map_context = .{} }};
}

test "manifest validateStageModel rejects blocks length mismatch" {
    var stage = std.mem.zeroes(weights.LoadedStageModel);
    stage.layer_range = .{ .start = 0, .end = 2 };
    stage.blocks = &.{};

    try std.testing.expectError(error.StageResidencyMismatch, validateStageModel(&stage));
}

test "manifest validateStageModel rejects empty stage range" {
    var stage = std.mem.zeroes(weights.LoadedStageModel);
    stage.layer_range = .{ .start = 1, .end = 1 };
    stage.blocks = &.{};

    try std.testing.expectError(error.EmptyStageRange, validateStageModel(&stage));
}

test "manifest validateStageModel rejects stage range beyond config layers" {
    var block_storage = [_]weights.blocks.LayerWeights{
        .{ .block_type = .attention_mlp, .weight_map = .{}, .map_context = .{} },
        .{ .block_type = .attention_mlp, .weight_map = .{}, .map_context = .{} },
    };
    var stage = std.mem.zeroes(weights.LoadedStageModel);
    stage.config.n_layers = 2;
    stage.layer_range = .{ .start = 1, .end = 3 };
    stage.blocks = block_storage[0..];
    stage.manifest = try makeValidationStageManifest(std.testing.allocator, 2, &.{});
    defer stage.manifest.deinit();

    try std.testing.expectError(error.InvalidStageRange, validateStageModel(&stage));
}

test "manifest validateStageModel rejects missing requested global role" {
    var blocks = stageBlockStorage();
    var stage = std.mem.zeroes(weights.LoadedStageModel);
    stage.config.n_layers = 1;
    stage.layer_range = .{ .start = 0, .end = 1 };
    stage.blocks = blocks[0..];
    stage.requested_roles = .{ .include_token_embeddings = true };
    stage.manifest = try makeValidationStageManifest(std.testing.allocator, 1, &.{
        .{
            .name = "tok.weight",
            .dtype = .f32,
            .shape = &validation_stage_shape_2d,
            .checkpoint_bytes = 64,
            .role = .token_embeddings,
            .weight_id = "token_embeddings",
            .status = .architecture_weight,
        },
    });
    defer stage.manifest.deinit();
    stage.residency = try stage.manifest.stageResidencyReport(stage.requested_roles.toResidencyRequest(stage.layer_range));

    try std.testing.expectError(error.MissingStageGlobalWeight, validateStageModel(&stage));
}

test "manifest validateStageModel rejects tied lm_head missing token embedding" {
    var blocks = stageBlockStorage();
    const dummy_lm_head = std.mem.zeroes(tensor.Tensor);
    var stage = std.mem.zeroes(weights.LoadedStageModel);
    stage.config.n_layers = 1;
    stage.layer_range = .{ .start = 0, .end = 1 };
    stage.blocks = blocks[0..];
    stage.requested_roles = .{ .include_lm_head = true };
    stage.lm_head = dummy_lm_head;
    stage.lm_head_uses_token_embeddings = true;
    stage.manifest = try makeValidationStageManifest(std.testing.allocator, 1, &.{
        .{
            .name = "tok.weight",
            .dtype = .f32,
            .shape = &validation_stage_shape_2d,
            .checkpoint_bytes = 64,
            .role = .token_embeddings,
            .weight_id = "token_embeddings",
            .status = .architecture_weight,
        },
    });
    defer stage.manifest.deinit();
    const effective_roles = weights.StageRoleRequest{
        .include_token_embeddings = true,
        .include_lm_head = true,
    };
    stage.residency = try stage.manifest.stageResidencyReport(effective_roles.toResidencyRequest(stage.layer_range));

    try std.testing.expectError(error.MissingStageGlobalWeight, validateStageModel(&stage));
}

test "manifest validateStageModel rejects residency mismatch" {
    var blocks = stageBlockStorage();
    var stage = std.mem.zeroes(weights.LoadedStageModel);
    stage.config.n_layers = 1;
    stage.layer_range = .{ .start = 0, .end = 1 };
    stage.blocks = blocks[0..];
    stage.manifest = try makeValidationStageManifest(std.testing.allocator, 1, &.{
        .{
            .name = "layers.0.proj.weight",
            .dtype = .f32,
            .shape = &validation_stage_shape_2d,
            .checkpoint_bytes = 64,
            .role = .decoder_layer,
            .layer_index = 0,
            .weight_id = "proj.weight",
            .status = .architecture_weight,
        },
    });
    defer stage.manifest.deinit();
    stage.residency = try stage.manifest.stageResidencyReport(stage.requested_roles.toResidencyRequest(stage.layer_range));
    stage.residency.role_bytes[@intFromEnum(manifest_mod.TensorRole.decoder_layer)] = 32;

    try std.testing.expectError(error.StageResidencyMismatch, validateStageModel(&stage));
}

test "manifest validateStageModel rejects residency range metadata mismatch" {
    var blocks = stageBlockStorage();
    var stage = std.mem.zeroes(weights.LoadedStageModel);
    stage.config.n_layers = 1;
    stage.layer_range = .{ .start = 0, .end = 1 };
    stage.blocks = blocks[0..];
    stage.manifest = try makeValidationStageManifest(std.testing.allocator, 1, &.{});
    defer stage.manifest.deinit();
    stage.residency = .{
        .layer_start = 1,
        .layer_end = 2,
    };

    try std.testing.expectError(error.StageResidencyMismatch, validateStageModel(&stage));
}

test "manifest validateStageModel rejects malformed selected layer weights" {
    var blocks = stageBlockStorage();
    var stage = std.mem.zeroes(weights.LoadedStageModel);
    stage.config.n_layers = 1;
    stage.config.d_model = 4;
    stage.config.d_ff = 8;
    stage.config.head_dim = 4;
    stage.config.n_heads = 1;
    stage.config.n_kv_groups = 1;
    stage.config.vocab_size = 4;
    stage.layer_range = .{ .start = 0, .end = 1 };
    stage.blocks = blocks[0..];
    stage.manifest = try makeValidationStageManifest(std.testing.allocator, 1, &.{});
    defer stage.manifest.deinit();
    stage.residency = try stage.manifest.stageResidencyReport(stage.requested_roles.toResidencyRequest(stage.layer_range));

    try std.testing.expectError(error.ValidationFailed, validateStageModel(&stage));
}

test "manifest validateStageModel rejects manifest layer count mismatch" {
    var blocks = stageBlockStorage();
    var stage = std.mem.zeroes(weights.LoadedStageModel);
    stage.config.n_layers = 1;
    stage.layer_range = .{ .start = 0, .end = 1 };
    stage.blocks = blocks[0..];
    stage.manifest = try makeValidationStageManifest(std.testing.allocator, 2, &.{});
    defer stage.manifest.deinit();

    try std.testing.expectError(error.StageResidencyMismatch, validateStageModel(&stage));
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

    const config = config_types.ModelConfig{
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

    const InferenceBackend = weights.blocks;

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

    const layer_blocks = try legacyBlocksToLayers(std.testing.allocator, &blocks);
    defer freeLayerBlocks(std.testing.allocator, layer_blocks);

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
        .blocks = layer_blocks,
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

    const config = config_types.ModelConfig{
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

    const InferenceBackend = weights.blocks;
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

    const layer_blocks = try legacyBlocksToLayers(std.testing.allocator, &blocks);
    defer freeLayerBlocks(std.testing.allocator, layer_blocks);

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
        .blocks = layer_blocks,
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

    const config = config_types.ModelConfig{
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

    const InferenceBackend = weights.blocks;
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

    const layer_blocks = try legacyBlocksToLayers(std.testing.allocator, &blocks);
    defer freeLayerBlocks(std.testing.allocator, layer_blocks);

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
        .blocks = layer_blocks,
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

    const config = config_types.ModelConfig{
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

    const InferenceBackend = weights.blocks;
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

    const layer_blocks = try legacyBlocksToLayers(std.testing.allocator, &blocks);
    defer freeLayerBlocks(std.testing.allocator, layer_blocks);

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
        .blocks = layer_blocks, // Only 1 block, but config expects 3
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

    const config = config_types.ModelConfig{
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

    const InferenceBackend = weights.blocks;
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

    const layer_blocks = try legacyBlocksToLayers(std.testing.allocator, &blocks);
    defer freeLayerBlocks(std.testing.allocator, layer_blocks);

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
        .blocks = layer_blocks,
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

    const config = config_types.ModelConfig{
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

    const InferenceBackend = weights.blocks;
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

    const layer_blocks = try legacyBlocksToLayers(std.testing.allocator, &blocks);
    defer freeLayerBlocks(std.testing.allocator, layer_blocks);

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
        .blocks = layer_blocks,
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
    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "failed to materialize layer 0 block weights for validation") != null);
}

test "legacyBlockToLayer round-trips fused gated delta block through blockWeightsFromMap" {
    const allocator = std.testing.allocator;

    var ln1_data = [_]f32{ 1, 1 };
    var in_proj_data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var conv_data = [_]f32{ 0, 1, 2, 3, 4, 5 };
    var a_log_data = [_]f32{ 0.1, 0.2 };
    var dt_bias_data = [_]f32{ 0.3, 0.4 };
    var norm_data = [_]f32{ 1, 1 };
    var out_proj_data = [_]f32{ 1, 0, 0, 1 };

    var ln1 = tensor.Tensor.view(@ptrCast(ln1_data[0..].ptr), &.{2}, .f32, null);
    var in_proj = tensor.Tensor.view2DSlice(in_proj_data[0..], 4, 2);
    var conv = tensor.Tensor.view(@ptrCast(conv_data[0..].ptr), &.{ 6, 1, 1 }, .f32, null);
    var a_log = tensor.Tensor.view(@ptrCast(a_log_data[0..].ptr), &.{2}, .f32, null);
    var dt_bias = tensor.Tensor.view(@ptrCast(dt_bias_data[0..].ptr), &.{2}, .f32, null);
    var norm = tensor.Tensor.view(@ptrCast(norm_data[0..].ptr), &.{2}, .f32, null);
    var out_proj = tensor.Tensor.view2DSlice(out_proj_data[0..], 2, 2);

    const legacy_block = weights.blocks.BlockWeights{
        .gated_delta = .{
            .ln1_weight = &ln1,
            .config = .{
                .d_model = 2,
                .d_conv = 1,
                .n_heads = 2,
                .d_head = 1,
            },
            .weights = .{
                .in_proj = &in_proj,
                .out_proj = &out_proj,
                .conv1d_weight = &conv,
                .conv1d_bias = null,
                .A_log = &a_log,
                .dt_bias = &dt_bias,
                .norm_weight = &norm,
            },
        },
    };

    var layer = try legacyBlockToLayer(allocator, legacy_block);
    defer layer.weight_map.deinit(allocator);

    const rematerialized = try weights.blocks.blockWeightsFromMap(
        &layer.weight_map,
        layer.block_type,
        layer.map_context,
    );
    try std.testing.expect(rematerialized == .gated_delta);
}

test "validateCommon: lm_head not 2D" {
    // Test that lm_head must be 2D
    const d_model: i32 = 64;
    const vocab_size: i32 = 1000;

    const config = config_types.ModelConfig{
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

    const InferenceBackend = weights.blocks;
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

    const layer_blocks = try legacyBlocksToLayers(std.testing.allocator, &blocks);
    defer freeLayerBlocks(std.testing.allocator, layer_blocks);

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
        .blocks = layer_blocks,
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
