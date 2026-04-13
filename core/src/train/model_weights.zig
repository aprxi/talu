//! Model weight storage for from-scratch transformer training.
//!
//! Owns all weight tensors (f32) and their corresponding gradient tensors.
//! Provides random initialization (Xavier uniform) and bulk gradient zeroing.
//!
//! Weight layout matches standard Llama/GPT naming:
//!   - token_embedding:  [vocab_size, d_model]
//!   - per-layer attention: attn_norm, q/k/v/o projections
//!   - per-layer FFN: ffn_norm, gate/up/down projections
//!   - final_norm:  [d_model]
//!   - lm_head:     [vocab_size, d_model]

const std = @import("std");
const tensor_mod = @import("tensor_pkg");
const model_config = @import("model_config.zig");
const grad_mod = @import("grad.zig");

const Allocator = std.mem.Allocator;
const OwnedTensor = tensor_mod.OwnedTensor;
const DType = tensor_mod.DType;
const GradTensor = grad_mod.GradTensor;
const TransformerConfig = model_config.TransformerConfig;

/// Weights and gradients for a single transformer layer.
pub const LayerWeights = struct {
    allocator: Allocator,

    // Attention block
    attn_norm: OwnedTensor, // [d_model]
    q_proj: OwnedTensor, // [num_heads * head_dim, d_model]
    k_proj: OwnedTensor, // [num_kv_heads * head_dim, d_model]
    v_proj: OwnedTensor, // [num_kv_heads * head_dim, d_model]
    o_proj: OwnedTensor, // [d_model, num_heads * head_dim]

    /// Fused QKV weight buffer for single-matmul projection.
    /// [q_dim + 2*kv_dim, d_model] — concatenation of q_proj, k_proj, v_proj.
    /// Synced from individual projections via syncQkvBuf().
    qkv_proj_buf: []f32,

    // FFN block (SwiGLU)
    ffn_norm: OwnedTensor, // [d_model]
    gate_proj: OwnedTensor, // [d_ff, d_model]
    up_proj: OwnedTensor, // [d_ff, d_model]
    down_proj: OwnedTensor, // [d_model, d_ff]

    // Gradients (always f32)
    grad_attn_norm: GradTensor,
    grad_q_proj: GradTensor,
    grad_k_proj: GradTensor,
    grad_v_proj: GradTensor,
    grad_o_proj: GradTensor,
    grad_ffn_norm: GradTensor,
    grad_gate_proj: GradTensor,
    grad_up_proj: GradTensor,
    grad_down_proj: GradTensor,

    pub fn init(allocator: Allocator, config: TransformerConfig) !LayerWeights {
        const d: usize = config.d_model;
        const hd: usize = config.headDim();
        const nh: usize = config.num_heads;
        const nkv: usize = config.num_kv_heads;
        const ff: usize = config.d_ff;
        const q_dim = nh * hd;
        const kv_dim = nkv * hd;

        var self: LayerWeights = undefined;
        self.allocator = allocator;
        var init_count: u32 = 0;
        errdefer {
            // Deinit in reverse order for each successfully initialized field.
            const fields = @typeInfo(LayerWeights).@"struct".fields;
            inline for (0..fields.len) |i| {
                const fi = fields.len - 1 - i;
                const field = fields[fi];
                if (fi < init_count) {
                    if (field.type == OwnedTensor) {
                        @field(self, field.name).deinit();
                    } else if (field.type == GradTensor) {
                        @field(self, field.name).deinit();
                    }
                }
            }
        }

        // Weights
        self.attn_norm = try OwnedTensor.init(allocator, .f32, &.{d});
        init_count += 1;
        self.q_proj = try OwnedTensor.init(allocator, .f32, &.{ q_dim, d });
        init_count += 1;
        self.k_proj = try OwnedTensor.init(allocator, .f32, &.{ kv_dim, d });
        init_count += 1;
        self.v_proj = try OwnedTensor.init(allocator, .f32, &.{ kv_dim, d });
        init_count += 1;
        self.o_proj = try OwnedTensor.init(allocator, .f32, &.{ d, q_dim });
        init_count += 1;
        self.qkv_proj_buf = try allocator.alloc(f32, (q_dim + 2 * kv_dim) * d);
        errdefer allocator.free(self.qkv_proj_buf);
        init_count += 1;
        self.ffn_norm = try OwnedTensor.init(allocator, .f32, &.{d});
        init_count += 1;
        self.gate_proj = try OwnedTensor.init(allocator, .f32, &.{ ff, d });
        init_count += 1;
        self.up_proj = try OwnedTensor.init(allocator, .f32, &.{ ff, d });
        init_count += 1;
        self.down_proj = try OwnedTensor.init(allocator, .f32, &.{ d, ff });
        init_count += 1;

        // Gradients
        self.grad_attn_norm = try GradTensor.init(allocator, &.{d});
        init_count += 1;
        self.grad_q_proj = try GradTensor.init(allocator, &.{ q_dim, d });
        init_count += 1;
        self.grad_k_proj = try GradTensor.init(allocator, &.{ kv_dim, d });
        init_count += 1;
        self.grad_v_proj = try GradTensor.init(allocator, &.{ kv_dim, d });
        init_count += 1;
        self.grad_o_proj = try GradTensor.init(allocator, &.{ d, q_dim });
        init_count += 1;
        self.grad_ffn_norm = try GradTensor.init(allocator, &.{d});
        init_count += 1;
        self.grad_gate_proj = try GradTensor.init(allocator, &.{ ff, d });
        init_count += 1;
        self.grad_up_proj = try GradTensor.init(allocator, &.{ ff, d });
        init_count += 1;
        self.grad_down_proj = try GradTensor.init(allocator, &.{ d, ff });
        init_count += 1;

        return self;
    }

    /// Copy q_proj, k_proj, v_proj into the contiguous qkv_proj_buf
    /// for fused QKV matmul. Call after weight initialization or optimizer step.
    pub fn syncQkvBuf(self: *LayerWeights) void {
        const q_data = self.q_proj.asSlice(f32);
        const k_data = self.k_proj.asSlice(f32);
        const v_data = self.v_proj.asSlice(f32);
        @memcpy(self.qkv_proj_buf[0..q_data.len], q_data);
        @memcpy(self.qkv_proj_buf[q_data.len .. q_data.len + k_data.len], k_data);
        @memcpy(self.qkv_proj_buf[q_data.len + k_data.len .. q_data.len + k_data.len + v_data.len], v_data);
    }

    /// Zero all gradient buffers between training steps.
    pub fn zeroGrads(self: *LayerWeights) void {
        self.grad_attn_norm.zero();
        self.grad_q_proj.zero();
        self.grad_k_proj.zero();
        self.grad_v_proj.zero();
        self.grad_o_proj.zero();
        self.grad_ffn_norm.zero();
        self.grad_gate_proj.zero();
        self.grad_up_proj.zero();
        self.grad_down_proj.zero();
    }

    pub fn deinit(self: *LayerWeights) void {
        self.grad_down_proj.deinit();
        self.grad_up_proj.deinit();
        self.grad_gate_proj.deinit();
        self.grad_ffn_norm.deinit();
        self.grad_o_proj.deinit();
        self.grad_v_proj.deinit();
        self.grad_k_proj.deinit();
        self.grad_q_proj.deinit();
        self.grad_attn_norm.deinit();

        self.down_proj.deinit();
        self.up_proj.deinit();
        self.gate_proj.deinit();
        self.ffn_norm.deinit();
        self.allocator.free(self.qkv_proj_buf);
        self.o_proj.deinit();
        self.v_proj.deinit();
        self.k_proj.deinit();
        self.q_proj.deinit();
        self.attn_norm.deinit();

        self.* = undefined;
    }
};

/// All model weights and gradients for a transformer language model.
pub const ModelWeights = struct {
    allocator: Allocator,
    config: TransformerConfig,

    token_embedding: OwnedTensor, // [vocab_size, d_model]
    final_norm: OwnedTensor, // [d_model]
    lm_head: OwnedTensor, // [vocab_size, d_model]

    grad_token_embedding: GradTensor,
    grad_final_norm: GradTensor,
    grad_lm_head: GradTensor,

    layers: []LayerWeights,

    pub fn init(allocator: Allocator, config: TransformerConfig) !ModelWeights {
        const d: usize = config.d_model;
        const v: usize = config.vocab_size;

        var self: ModelWeights = undefined;
        self.allocator = allocator;
        self.config = config;

        self.token_embedding = try OwnedTensor.init(allocator, .f32, &.{ v, d });
        errdefer self.token_embedding.deinit();

        self.final_norm = try OwnedTensor.init(allocator, .f32, &.{d});
        errdefer self.final_norm.deinit();

        self.lm_head = try OwnedTensor.init(allocator, .f32, &.{ v, d });
        errdefer self.lm_head.deinit();

        self.grad_token_embedding = try GradTensor.init(allocator, &.{ v, d });
        errdefer self.grad_token_embedding.deinit();

        self.grad_final_norm = try GradTensor.init(allocator, &.{d});
        errdefer self.grad_final_norm.deinit();

        self.grad_lm_head = try GradTensor.init(allocator, &.{ v, d });
        errdefer self.grad_lm_head.deinit();

        self.layers = try allocator.alloc(LayerWeights, config.num_layers);
        var initialized_layers: u32 = 0;
        errdefer {
            for (self.layers[0..initialized_layers]) |*layer| {
                layer.deinit();
            }
            allocator.free(self.layers);
        }

        for (self.layers) |*layer| {
            layer.* = try LayerWeights.init(allocator, config);
            initialized_layers += 1;
        }

        return self;
    }

    /// Initialize weights with Xavier uniform: U(-limit, +limit) where limit = sqrt(6 / (fan_in + fan_out)).
    /// RMSNorm weights are initialized to 1.0 (identity scaling).
    /// Seed controls the PRNG for reproducibility.
    pub fn initRandom(self: *ModelWeights, seed: u64) void {
        var rng = std.Random.DefaultPrng.init(seed);
        const random = rng.random();

        // Token embedding: fan_in = vocab_size, fan_out = d_model
        xavierUniform(self.token_embedding.asSlice(f32), self.config.vocab_size, self.config.d_model, random);

        // LM head: fan_in = d_model, fan_out = vocab_size
        xavierUniform(self.lm_head.asSlice(f32), self.config.d_model, self.config.vocab_size, random);

        // Final norm: ones
        fillOnes(self.final_norm.asSlice(f32));

        for (self.layers) |*layer| {
            const d = self.config.d_model;
            const hd = self.config.headDim();
            const nh = self.config.num_heads;
            const nkv = self.config.num_kv_heads;
            const ff = self.config.d_ff;

            // Attention norms: ones
            fillOnes(layer.attn_norm.asSlice(f32));
            fillOnes(layer.ffn_norm.asSlice(f32));

            // Projections: Xavier uniform
            xavierUniform(layer.q_proj.asSlice(f32), d, nh * hd, random);
            xavierUniform(layer.k_proj.asSlice(f32), d, nkv * hd, random);
            xavierUniform(layer.v_proj.asSlice(f32), d, nkv * hd, random);
            xavierUniform(layer.o_proj.asSlice(f32), nh * hd, d, random);

            // SwiGLU projections
            xavierUniform(layer.gate_proj.asSlice(f32), d, ff, random);
            xavierUniform(layer.up_proj.asSlice(f32), d, ff, random);
            xavierUniform(layer.down_proj.asSlice(f32), ff, d, random);

            layer.syncQkvBuf();
        }
    }

    /// Zero all gradient buffers across all layers and global weights.
    pub fn zeroGrads(self: *ModelWeights) void {
        self.grad_token_embedding.zero();
        self.grad_final_norm.zero();
        self.grad_lm_head.zero();
        for (self.layers) |*layer| {
            layer.zeroGrads();
        }
    }

    /// Copy all trainable weights into a flat f32 buffer in TinyLLM checkpoint order.
    /// Order:
    ///   token_embedding, final_norm, lm_head,
    ///   for each layer: attn_norm, q_proj, k_proj, v_proj, o_proj,
    ///                   ffn_norm, gate_proj, up_proj, down_proj.
    pub fn copyFlatF32(self: *const ModelWeights, out: []f32) void {
        std.debug.assert(out.len == @as(usize, @intCast(self.totalParams())));

        var cursor: usize = 0;

        cursor += copySlice(out[cursor..], self.token_embedding.asSlice(f32));
        cursor += copySlice(out[cursor..], self.final_norm.asSlice(f32));
        cursor += copySlice(out[cursor..], self.lm_head.asSlice(f32));

        for (self.layers) |*layer| {
            cursor += copySlice(out[cursor..], layer.attn_norm.asSlice(f32));
            cursor += copySlice(out[cursor..], layer.q_proj.asSlice(f32));
            cursor += copySlice(out[cursor..], layer.k_proj.asSlice(f32));
            cursor += copySlice(out[cursor..], layer.v_proj.asSlice(f32));
            cursor += copySlice(out[cursor..], layer.o_proj.asSlice(f32));
            cursor += copySlice(out[cursor..], layer.ffn_norm.asSlice(f32));
            cursor += copySlice(out[cursor..], layer.gate_proj.asSlice(f32));
            cursor += copySlice(out[cursor..], layer.up_proj.asSlice(f32));
            cursor += copySlice(out[cursor..], layer.down_proj.asSlice(f32));
        }

        std.debug.assert(cursor == out.len);
    }

    /// Load all trainable weights from a flat f32 buffer in TinyLLM checkpoint order.
    pub fn loadFlatF32(self: *ModelWeights, flat: []const f32) void {
        std.debug.assert(flat.len == @as(usize, @intCast(self.totalParams())));

        var cursor: usize = 0;

        cursor += loadSlice(self.token_embedding.asSlice(f32), flat[cursor..]);
        cursor += loadSlice(self.final_norm.asSlice(f32), flat[cursor..]);
        cursor += loadSlice(self.lm_head.asSlice(f32), flat[cursor..]);

        for (self.layers) |*layer| {
            cursor += loadSlice(layer.attn_norm.asSlice(f32), flat[cursor..]);
            cursor += loadSlice(layer.q_proj.asSlice(f32), flat[cursor..]);
            cursor += loadSlice(layer.k_proj.asSlice(f32), flat[cursor..]);
            cursor += loadSlice(layer.v_proj.asSlice(f32), flat[cursor..]);
            cursor += loadSlice(layer.o_proj.asSlice(f32), flat[cursor..]);
            cursor += loadSlice(layer.ffn_norm.asSlice(f32), flat[cursor..]);
            cursor += loadSlice(layer.gate_proj.asSlice(f32), flat[cursor..]);
            cursor += loadSlice(layer.up_proj.asSlice(f32), flat[cursor..]);
            cursor += loadSlice(layer.down_proj.asSlice(f32), flat[cursor..]);
            layer.syncQkvBuf();
        }

        std.debug.assert(cursor == flat.len);
    }

    /// Total number of trainable parameters.
    pub fn totalParams(self: *const ModelWeights) u64 {
        var total: u64 = 0;
        total += self.token_embedding.numElements();
        total += self.final_norm.numElements();
        total += self.lm_head.numElements();
        for (self.layers) |*layer| {
            total += layer.attn_norm.numElements();
            total += layer.q_proj.numElements();
            total += layer.k_proj.numElements();
            total += layer.v_proj.numElements();
            total += layer.o_proj.numElements();
            total += layer.ffn_norm.numElements();
            total += layer.gate_proj.numElements();
            total += layer.up_proj.numElements();
            total += layer.down_proj.numElements();
        }
        return total;
    }

    pub fn deinit(self: *ModelWeights) void {
        for (self.layers) |*layer| {
            layer.deinit();
        }
        self.allocator.free(self.layers);

        self.grad_lm_head.deinit();
        self.grad_final_norm.deinit();
        self.grad_token_embedding.deinit();

        self.lm_head.deinit();
        self.final_norm.deinit();
        self.token_embedding.deinit();

        self.* = undefined;
    }
};

// =============================================================================
// Initialization helpers
// =============================================================================

/// Xavier uniform initialization: U(-limit, +limit) where limit = sqrt(6 / (fan_in + fan_out)).
fn xavierUniform(data: []f32, fan_in: anytype, fan_out: anytype, random: std.Random) void {
    const fi: f32 = @floatFromInt(fan_in);
    const fo: f32 = @floatFromInt(fan_out);
    const limit = @sqrt(6.0 / (fi + fo));
    for (data) |*v| {
        // Map uniform [0, 1) to [-limit, +limit)
        v.* = (random.float(f32) * 2.0 - 1.0) * limit;
    }
}

fn fillOnes(data: []f32) void {
    for (data) |*v| {
        v.* = 1.0;
    }
}

fn copySlice(dst: []f32, src: []const f32) usize {
    std.debug.assert(dst.len >= src.len);
    @memcpy(dst[0..src.len], src);
    return src.len;
}

fn loadSlice(dst: []f32, src: []const f32) usize {
    std.debug.assert(src.len >= dst.len);
    @memcpy(dst, src[0..dst.len]);
    return dst.len;
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

fn testConfig() TransformerConfig {
    return .{
        .vocab_size = 32,
        .d_model = 16,
        .num_layers = 2,
        .num_heads = 2,
        .num_kv_heads = 2,
        .d_ff = 32,
        .seq_len = 8,
    };
}

test "ModelWeights init allocates correct shapes" {
    const config = testConfig();
    var weights = try ModelWeights.init(testing.allocator, config);
    defer weights.deinit();

    // Token embedding: [32, 16]
    try testing.expectEqual(@as(usize, 32 * 16), weights.token_embedding.numElements());
    // Final norm: [16]
    try testing.expectEqual(@as(usize, 16), weights.final_norm.numElements());
    // LM head: [32, 16]
    try testing.expectEqual(@as(usize, 32 * 16), weights.lm_head.numElements());
    // 2 layers
    try testing.expectEqual(@as(usize, 2), weights.layers.len);
}

test "ModelWeights init layer shapes match config" {
    const config = testConfig();
    var weights = try ModelWeights.init(testing.allocator, config);
    defer weights.deinit();

    const layer = &weights.layers[0];
    // attn_norm: [16]
    try testing.expectEqual(@as(usize, 16), layer.attn_norm.numElements());
    // q_proj: [num_heads * head_dim, d_model] = [16, 16]
    try testing.expectEqual(@as(usize, 16 * 16), layer.q_proj.numElements());
    // k_proj: [num_kv_heads * head_dim, d_model] = [16, 16]
    try testing.expectEqual(@as(usize, 16 * 16), layer.k_proj.numElements());
    // gate_proj: [d_ff, d_model] = [32, 16]
    try testing.expectEqual(@as(usize, 32 * 16), layer.gate_proj.numElements());
    // down_proj: [d_model, d_ff] = [16, 32]
    try testing.expectEqual(@as(usize, 16 * 32), layer.down_proj.numElements());
}

test "ModelWeights initRandom produces non-zero weights" {
    const config = testConfig();
    var weights = try ModelWeights.init(testing.allocator, config);
    defer weights.deinit();

    weights.initRandom(42);

    // Token embedding should have non-zero values
    var has_nonzero = false;
    for (weights.token_embedding.asSlice(f32)) |v| {
        if (v != 0.0) {
            has_nonzero = true;
            break;
        }
    }
    try testing.expect(has_nonzero);

    // LM head should have non-zero values
    has_nonzero = false;
    for (weights.lm_head.asSlice(f32)) |v| {
        if (v != 0.0) {
            has_nonzero = true;
            break;
        }
    }
    try testing.expect(has_nonzero);
}

test "ModelWeights initRandom norm weights are ones" {
    const config = testConfig();
    var weights = try ModelWeights.init(testing.allocator, config);
    defer weights.deinit();

    weights.initRandom(42);

    for (weights.final_norm.asSlice(f32)) |v| {
        try testing.expectApproxEqAbs(@as(f32, 1.0), v, 1e-6);
    }
    for (weights.layers[0].attn_norm.asSlice(f32)) |v| {
        try testing.expectApproxEqAbs(@as(f32, 1.0), v, 1e-6);
    }
    for (weights.layers[0].ffn_norm.asSlice(f32)) |v| {
        try testing.expectApproxEqAbs(@as(f32, 1.0), v, 1e-6);
    }
}

test "ModelWeights initRandom is deterministic with same seed" {
    const config = testConfig();
    var w1 = try ModelWeights.init(testing.allocator, config);
    defer w1.deinit();
    var w2 = try ModelWeights.init(testing.allocator, config);
    defer w2.deinit();

    w1.initRandom(123);
    w2.initRandom(123);

    const d1 = w1.token_embedding.asSlice(f32);
    const d2 = w2.token_embedding.asSlice(f32);
    for (d1, d2) |a, b| {
        try testing.expectEqual(a, b);
    }
}

test "ModelWeights initRandom different seeds produce different weights" {
    const config = testConfig();
    var w1 = try ModelWeights.init(testing.allocator, config);
    defer w1.deinit();
    var w2 = try ModelWeights.init(testing.allocator, config);
    defer w2.deinit();

    w1.initRandom(1);
    w2.initRandom(2);

    var differ = false;
    const d1 = w1.token_embedding.asSlice(f32);
    const d2 = w2.token_embedding.asSlice(f32);
    for (d1, d2) |a, b| {
        if (a != b) {
            differ = true;
            break;
        }
    }
    try testing.expect(differ);
}

test "ModelWeights zeroGrads zeros all gradient buffers" {
    const config = testConfig();
    var weights = try ModelWeights.init(testing.allocator, config);
    defer weights.deinit();

    // Write something into gradients
    weights.grad_token_embedding.asSliceMut()[0] = 1.0;
    weights.layers[0].grad_q_proj.asSliceMut()[0] = 2.0;

    weights.zeroGrads();

    try testing.expectEqual(@as(f32, 0.0), weights.grad_token_embedding.asSlice()[0]);
    try testing.expectEqual(@as(f32, 0.0), weights.layers[0].grad_q_proj.asSlice()[0]);
}

test "ModelWeights totalParams matches config calculation" {
    const config = testConfig();
    var weights = try ModelWeights.init(testing.allocator, config);
    defer weights.deinit();

    try testing.expectEqual(config.totalParams(), weights.totalParams());
}

test "ModelWeights initRandom values within Xavier bounds" {
    const config = testConfig();
    var weights = try ModelWeights.init(testing.allocator, config);
    defer weights.deinit();

    weights.initRandom(42);

    // q_proj: fan_in=16, fan_out=16, limit = sqrt(6/32) ≈ 0.433
    const limit = @sqrt(6.0 / 32.0);
    for (weights.layers[0].q_proj.asSlice(f32)) |v| {
        try testing.expect(v >= -limit and v <= limit);
    }
}

test "ModelWeights copyFlatF32 preserves tensor order" {
    var config = testConfig();
    config.num_layers = 1;
    var weights = try ModelWeights.init(testing.allocator, config);
    defer weights.deinit();

    @memset(weights.token_embedding.asSliceMut(f32), 1.0);
    @memset(weights.final_norm.asSliceMut(f32), 2.0);
    @memset(weights.lm_head.asSliceMut(f32), 3.0);

    const layer0 = &weights.layers[0];
    @memset(layer0.attn_norm.asSliceMut(f32), 4.0);
    @memset(layer0.q_proj.asSliceMut(f32), 5.0);
    @memset(layer0.k_proj.asSliceMut(f32), 6.0);
    @memset(layer0.v_proj.asSliceMut(f32), 7.0);
    @memset(layer0.o_proj.asSliceMut(f32), 8.0);
    @memset(layer0.ffn_norm.asSliceMut(f32), 9.0);
    @memset(layer0.gate_proj.asSliceMut(f32), 10.0);
    @memset(layer0.up_proj.asSliceMut(f32), 11.0);
    @memset(layer0.down_proj.asSliceMut(f32), 12.0);

    var flat = try testing.allocator.alloc(f32, @intCast(weights.totalParams()));
    defer testing.allocator.free(flat);
    weights.copyFlatF32(flat);

    var cursor: usize = 0;
    inline for (.{
        .{ @as(f32, 1.0), weights.token_embedding.numElements() },
        .{ @as(f32, 2.0), weights.final_norm.numElements() },
        .{ @as(f32, 3.0), weights.lm_head.numElements() },
        .{ @as(f32, 4.0), layer0.attn_norm.numElements() },
        .{ @as(f32, 5.0), layer0.q_proj.numElements() },
        .{ @as(f32, 6.0), layer0.k_proj.numElements() },
        .{ @as(f32, 7.0), layer0.v_proj.numElements() },
        .{ @as(f32, 8.0), layer0.o_proj.numElements() },
        .{ @as(f32, 9.0), layer0.ffn_norm.numElements() },
        .{ @as(f32, 10.0), layer0.gate_proj.numElements() },
        .{ @as(f32, 11.0), layer0.up_proj.numElements() },
        .{ @as(f32, 12.0), layer0.down_proj.numElements() },
    }) |section| {
        for (flat[cursor .. cursor + section[1]]) |v| {
            try testing.expectEqual(section[0], v);
        }
        cursor += section[1];
    }
    try testing.expectEqual(flat.len, cursor);
}

test "ModelWeights loadFlatF32 roundtrips copyFlatF32" {
    const config = testConfig();
    var source = try ModelWeights.init(testing.allocator, config);
    defer source.deinit();
    source.initRandom(123);

    const flat = try testing.allocator.alloc(f32, @intCast(source.totalParams()));
    defer testing.allocator.free(flat);
    source.copyFlatF32(flat);

    var restored = try ModelWeights.init(testing.allocator, config);
    defer restored.deinit();
    restored.loadFlatF32(flat);

    const restored_flat = try testing.allocator.alloc(f32, @intCast(restored.totalParams()));
    defer testing.allocator.free(restored_flat);
    restored.copyFlatF32(restored_flat);

    for (flat, restored_flat) |expected, actual| {
        try testing.expectEqual(expected, actual);
    }
}
