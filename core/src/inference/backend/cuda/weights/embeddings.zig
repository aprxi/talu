//! Embedding and token-materialization helpers for the CUDA inference backend.

const host = @import("host.zig");
const gaffine_utils = @import("gaffine.zig");
const freeOwnedTensorView = host.freeOwnedTensorView;
const decodeGaffineRow = gaffine_utils.decodeGaffineRow;
const gaffineValueAt = gaffine_utils.gaffineValueAt;

const std = @import("std");
const compute = @import("compute_pkg");
const tensor = @import("compute_pkg").tensor;
const dtype = @import("compute_pkg").dtype;
const log = @import("log_pkg");
const load_transforms = @import("models_pkg").load.transforms;
const models = @import("models_pkg");
const GateUpLayout = models.runtime_blocks.GateUpLayout;

const LoadedModel = models.LoadedModel;
const Tensor = tensor.Tensor;

/// Convert UE8M0 block scale exponent to f32 scale factor.
inline fn ue8m0ToScale(e8m0: u8) f32 {
    const exp_bits = @as(u32, e8m0) << 23;
    return @bitCast(exp_bits);
}

// --- Shared types from engine_types.zig ---
const engine_types = @import("../runtime/_types_impl.zig");
const KvCacheDtype = engine_types.KvCacheDtype;
const gaffine_scales_dtype_f16 = engine_types.gaffine_scales_dtype_f16;
const gaffine_scales_dtype_bf16 = engine_types.gaffine_scales_dtype_bf16;
const DenseU16Dtype = engine_types.DenseU16Dtype;
const EmbeddingLookupKind = engine_types.EmbeddingLookupKind;
const EmbeddingLookup = engine_types.EmbeddingLookup;
const LinearWeight = engine_types.LinearWeight;
const DeviceTensor = engine_types.DeviceTensor;
const MoEWeightRefs = engine_types.MoEWeightRefs;
const MoEWeights = models.runtime_blocks.MoEWeights;

pub fn populatePrefillHiddenFromTokens(
    loaded: *const LoadedModel,
    tokens: []const u32,
    d_model: usize,
    out: []f32,
    skip_token_id: ?u32,
) !void {
    if (d_model == 0) return error.InvalidArgument;
    const expected = std.math.mul(usize, tokens.len, d_model) catch return error.InvalidArgument;
    if (out.len != expected) return error.InvalidArgument;

    var idx: usize = 0;
    while (idx < tokens.len) : (idx += 1) {
        const row_start = std.math.mul(usize, idx, d_model) catch return error.InvalidArgument;
        const row = out[row_start .. row_start + d_model];
        if (skip_token_id) |skip_id| {
            if (tokens[idx] == skip_id) {
                @memset(row, 0.0);
                continue;
            }
        }
        const used_model_embeddings = tryPopulateHiddenFromToken(loaded, tokens[idx], row) catch |err| switch (err) {
            error.InvalidArgument => return error.InvalidArgument,
            else => return err,
        };
        if (!used_model_embeddings) return error.UnsupportedModel;
        if (loaded.config.embedding_multiplier != 1.0) {
            for (row) |*value| value.* *= loaded.config.embedding_multiplier;
        }
    }
}

pub fn tryUploadEmbeddingLookup(
    device: *compute.cuda.Device,
    loaded: *const LoadedModel,
    d_model: usize,
) !?EmbeddingLookup {
    const embeddings = &loaded.token_embeddings;
    if (embeddings.data_ptr == null) return null;
    if (embeddings.n_dims != 2) return null;
    if (embeddings.shape[0] <= 0 or embeddings.shape[1] <= 0) return null;
    const kind: EmbeddingLookupKind = switch (embeddings.dtype) {
        .f32 => .f32,
        .f16 => .f16,
        .bf16 => .bf16,
        .grouped_affine_u4 => .gaffine_u4,
        else => return null,
    };

    const dim0: usize = @intCast(embeddings.shape[0]);
    const dim1: usize = @intCast(embeddings.shape[1]);
    var layout_tag: u32 = undefined;
    var hidden_dim: usize = undefined;
    if (dim1 == d_model) {
        layout_tag = compute.cuda.embedding_lookup_f32.layout_vocab_hidden;
        hidden_dim = dim1;
    } else if (dim0 == d_model) {
        if (kind == .gaffine_u4) return null;
        layout_tag = compute.cuda.embedding_lookup_f32.layout_hidden_vocab;
        hidden_dim = dim0;
    } else {
        return null;
    }
    const dim0_u32 = std.math.cast(u32, dim0) orelse return error.InvalidArgument;
    const dim1_u32 = std.math.cast(u32, dim1) orelse return error.InvalidArgument;
    const hidden_dim_u32 = std.math.cast(u32, hidden_dim) orelse return error.InvalidArgument;

    if (kind == .gaffine_u4) {
        const gaffine = embeddings.gaffine orelse return null;
        if (gaffine.group_size == 0) return null;
        const group_size: usize = gaffine.group_size;
        if ((hidden_dim % group_size) != 0 or (group_size % 8) != 0) return null;
        const groups_per_row = hidden_dim / group_size;
        const packed_words_per_row = hidden_dim / 8;
        const packed_words_total = std.math.mul(usize, dim0, packed_words_per_row) catch return error.InvalidArgument;
        const sb_count = std.math.mul(usize, dim0, groups_per_row) catch return error.InvalidArgument;
        const packed_bytes = std.math.mul(usize, packed_words_total, @sizeOf(u32)) catch return error.InvalidArgument;
        const sb_bytes = std.math.mul(usize, sb_count, @sizeOf(u16)) catch return error.InvalidArgument;
        const packed_vals = embeddings.asSliceUnaligned(u32);
        if (packed_vals.len < packed_words_total) return error.InvalidArgument;
        if (gaffine.scales.len < sb_bytes or gaffine.biases.len < sb_bytes) return error.InvalidArgument;
        const scales_dtype_tag = switch (gaffine.scales_dtype) {
            .f16 => gaffine_scales_dtype_f16,
            .bf16 => gaffine_scales_dtype_bf16,
            else => return error.UnsupportedModel,
        };

        var packed_dev = try device.allocBuffer(packed_bytes);
        errdefer packed_dev.deinit(device);
        var scales_dev = try device.allocBuffer(sb_bytes);
        errdefer scales_dev.deinit(device);
        var biases_dev = try device.allocBuffer(sb_bytes);
        errdefer biases_dev.deinit(device);
        try packed_dev.upload(device, std.mem.sliceAsBytes(packed_vals[0..packed_words_total]));
        try scales_dev.upload(device, gaffine.scales[0..sb_bytes]);
        try biases_dev.upload(device, gaffine.biases[0..sb_bytes]);

        return .{
            .kind = .gaffine_u4,
            .dim0 = dim0_u32,
            .dim1 = dim1_u32,
            .hidden_dim = hidden_dim_u32,
            .layout_tag = layout_tag,
            .group_size = std.math.cast(u32, group_size) orelse return error.InvalidArgument,
            .scales_dtype_tag = scales_dtype_tag,
            .scales = scales_dev,
            .biases = biases_dev,
            .multiplier = loaded.config.embedding_multiplier,
            .buffer = packed_dev,
        };
    }

    const elem_count = std.math.mul(usize, dim0, dim1) catch return error.InvalidArgument;
    const elem_bytes: usize = switch (kind) {
        .f32 => @sizeOf(f32),
        .f16, .bf16 => @sizeOf(u16),
        .gaffine_u4 => unreachable,
    };
    const bytes = std.math.mul(usize, elem_count, elem_bytes) catch return error.InvalidArgument;
    var buffer = try device.allocBuffer(bytes);
    errdefer buffer.deinit(device);
    switch (kind) {
        .f32 => {
            const src = embeddings.asSlice(f32);
            if (src.len < elem_count) return error.InvalidArgument;
            try buffer.upload(device, std.mem.sliceAsBytes(src[0..elem_count]));
        },
        .f16, .bf16 => {
            const src_u16 = embeddings.asSliceUnaligned(u16);
            if (src_u16.len < elem_count) return error.InvalidArgument;
            try buffer.upload(device, std.mem.sliceAsBytes(src_u16[0..elem_count]));
        },
        .gaffine_u4 => unreachable,
    }

    return .{
        .kind = kind,
        .dim0 = dim0_u32,
        .dim1 = dim1_u32,
        .hidden_dim = hidden_dim_u32,
        .layout_tag = layout_tag,
        .multiplier = loaded.config.embedding_multiplier,
        .buffer = buffer,
    };
}

pub fn canUseModelEmbeddings(loaded: *const LoadedModel, d_model: usize) bool {
    if (d_model == 0) return false;
    const embeddings = loaded.token_embeddings;
    if (embeddings.data_ptr == null) return false;
    if (embeddings.n_dims != 2) return false;
    if (embeddings.shape[0] <= 0 or embeddings.shape[1] <= 0) return false;
    const dim0: usize = @intCast(embeddings.shape[0]);
    const dim1: usize = @intCast(embeddings.shape[1]);
    if (dim0 != d_model and dim1 != d_model) return false;
    return switch (embeddings.dtype) {
        .f32, .f16, .bf16, .f8_e4m3, .grouped_affine_u4, .grouped_affine_u8 => true,
        else => false,
    };
}

pub fn tryPopulateHiddenFromToken(
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
                try decodeGaffineRow(embeddings, token_idx, out);
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
        .f8_e4m3 => {
            // MXFP8 embedding: dequant-after-take with block-32 UE8M0 scales.
            const mxfp8 = embeddings.mxfp8 orelse return false;
            if (dim1 != hidden_dim) return false;
            if (token_idx >= dim0) return error.InvalidArgument;
            const fp8_data = embeddings.data();
            const row_offset = token_idx * dim1;
            const scale_cols: usize = mxfp8.scale_cols;
            const scales_ptr = mxfp8.block_scales_data orelse return false;
            const scales = scales_ptr[0..mxfp8.block_scales_len];
            const scale_row_offset = token_idx * scale_cols;
            for (0..hidden_dim) |col| {
                const block_idx = col / 32;
                const scale = ue8m0ToScale(scales[scale_row_offset + block_idx]);
                out[col] = dtype.fp8e4m3ToF32(fp8_data[row_offset + col]) * scale;
            }
            return true;
        },
        else => return false,
    }
}

pub fn tryPopulateFinalNormWeight(loaded: *const LoadedModel, out: []f32) bool {
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

pub fn tryPopulateProjectionFromWeight(
    allocator: std.mem.Allocator,
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
            const dequantized = load_transforms.convertToF32(allocator, weight.*) catch return false;
            defer freeOwnedTensorView(allocator, dequantized);
            const src = dequantized.asSlice(f32);

            if (dim0 == d_model and dim1 >= projected_vocab) {
                var row: usize = 0;
                while (row < d_model) : (row += 1) {
                    const src_start = row * dim1;
                    const dst_start = row * projected_vocab;
                    @memcpy(out[dst_start .. dst_start + projected_vocab], src[src_start .. src_start + projected_vocab]);
                }
                return true;
            }
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
        else => return false,
    }
}
