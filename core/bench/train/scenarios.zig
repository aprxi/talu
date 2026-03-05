//! Training benchmark scenarios.
//!
//! Each scenario isolates a specific kernel or pass for measurement.
//! Default dimensions match the shakespeare model config.

const std = @import("std");
const core = @import("main");
const harness = @import("harness.zig");
const metrics = @import("metrics.zig");

const train = core.train;
const compute = core.compute;
const TransformerConfig = train.TransformerConfig;
const ModelWeights = train.ModelWeights;
const ActivationCache = train.ActivationCache;
const MatmulScratch = compute.cpu.linalg.MatmulScratch;

// Forward kernels
const linear_fwd = train.forward.linear;
const attention_fwd = train.forward.attention;
const norm_fwd = train.forward.norm;
const activation_fwd = train.forward.activation;
const rope_fwd = train.forward.rope;
const embedding_fwd = train.forward.embedding;
const loss_fwd = train.forward.loss;
const forward_pass = train.forward.pass;

// Backward kernels
const backward_pass = train.backward_pass;
const linear_bw = train.backward.linear;
const attention_bw = train.backward.attention;
const norm_bw = train.backward.rmsnorm;
const activation_bw = train.backward.activation;
const rope_bw = train.backward.rope;
const cross_entropy_bw = train.backward.cross_entropy;

pub const Profile = enum { ci, bw };

pub const RunConfig = struct {
    warmup: usize = 4,
    iters: usize = 20,
    profile: Profile = .bw,
};

pub const Scenario = enum {
    forward_linear,
    forward_attention,
    forward_norm,
    forward_activation,
    forward_rope,
    forward_embedding,
    forward_loss,
    forward_full,
    backward_linear,
    backward_attention,
    backward_norm,
    backward_activation,
    backward_rope,
    backward_loss,
    backward_full,
    step_full,
    optimizer_step,
    all,

    pub fn label(self: Scenario) []const u8 {
        return switch (self) {
            .forward_linear => "fwd/linear",
            .forward_attention => "fwd/attention",
            .forward_norm => "fwd/norm",
            .forward_activation => "fwd/activation",
            .forward_rope => "fwd/rope",
            .forward_embedding => "fwd/embedding",
            .forward_loss => "fwd/loss",
            .forward_full => "fwd/full",
            .backward_linear => "bwd/linear",
            .backward_attention => "bwd/attention",
            .backward_norm => "bwd/norm",
            .backward_activation => "bwd/activation",
            .backward_rope => "bwd/rope",
            .backward_loss => "bwd/loss",
            .backward_full => "bwd/full",
            .step_full => "step/full",
            .optimizer_step => "step/optimizer",
            .all => "all",
        };
    }

    pub fn fromString(s: []const u8) ?Scenario {
        inline for (std.meta.fields(Scenario)) |f| {
            if (std.mem.eql(u8, s, f.name)) return @enumFromInt(f.value);
        }
        return null;
    }
};

pub const ScenarioResult = struct {
    name: []const u8,
    summary: harness.Summary,
    flops: u64 = 0,
    note: []const u8 = "",
};

/// Default shakespeare-sized config for benchmarks.
fn defaultConfig() TransformerConfig {
    return .{
        .vocab_size = 512,
        .d_model = 256,
        .num_layers = 4,
        .num_heads = 4,
        .num_kv_heads = 4,
        .d_ff = 1024,
        .seq_len = 256,
    };
}

const batch_size: u32 = 4;

/// Run a scenario and return results.
pub fn run(scenario: Scenario, config: RunConfig, allocator: std.mem.Allocator) ![]ScenarioResult {
    if (scenario == .all) {
        return runAll(config, allocator);
    }

    const results = try allocator.alloc(ScenarioResult, 1);
    results[0] = try runOne(scenario, config, allocator);
    return results;
}

fn runAll(config: RunConfig, allocator: std.mem.Allocator) ![]ScenarioResult {
    const scenarios = [_]Scenario{
        .forward_linear,
        .forward_attention,
        .forward_norm,
        .forward_activation,
        .forward_rope,
        .forward_embedding,
        .forward_loss,
        .forward_full,
        .backward_linear,
        .backward_attention,
        .backward_norm,
        .backward_activation,
        .backward_rope,
        .backward_loss,
        .backward_full,
        .step_full,
        .optimizer_step,
    };
    const results = try allocator.alloc(ScenarioResult, scenarios.len);
    for (scenarios, 0..) |s, i| {
        results[i] = try runOne(s, config, allocator);
    }
    return results;
}

fn runOne(scenario: Scenario, config: RunConfig, allocator: std.mem.Allocator) !ScenarioResult {
    return switch (scenario) {
        .forward_linear => try runForwardLinear(config, allocator),
        .forward_attention => try runForwardAttention(config, allocator),
        .forward_norm => try runForwardNorm(config, allocator),
        .forward_activation => try runForwardActivation(config, allocator),
        .forward_rope => try runForwardRope(config, allocator),
        .forward_embedding => try runForwardEmbedding(config, allocator),
        .forward_loss => try runForwardLoss(config, allocator),
        .forward_full => try runForwardFull(config, allocator),
        .backward_linear => try runBackwardLinear(config, allocator),
        .backward_attention => try runBackwardAttention(config, allocator),
        .backward_norm => try runBackwardNorm(config, allocator),
        .backward_activation => try runBackwardActivation(config, allocator),
        .backward_rope => try runBackwardRope(config, allocator),
        .backward_loss => try runBackwardLoss(config, allocator),
        .backward_full => try runBackwardFull(config, allocator),
        .step_full => try runStepFull(config, allocator),
        .optimizer_step => try runOptimizerStep(config, allocator),
        .all => unreachable,
    };
}

// =========================================================================
// Scenario Runners
// =========================================================================

fn runForwardLinear(config: RunConfig, allocator: std.mem.Allocator) !ScenarioResult {
    const mc = defaultConfig();
    const bs: usize = @as(usize, batch_size) * mc.seq_len;
    const d = mc.d_model;
    const ff = mc.d_ff;

    const input = try allocator.alloc(f32, bs * d);
    defer allocator.free(input);
    const output = try allocator.alloc(f32, bs * ff);
    defer allocator.free(output);
    const weight = try allocator.alloc(f32, ff * d);
    defer allocator.free(weight);
    var scratch = try MatmulScratch.init(allocator);
    defer scratch.deinit();

    fillRandom(input);
    fillRandom(weight);

    const total = config.warmup + config.iters;
    const samples = try allocator.alloc(u64, config.iters);
    defer allocator.free(samples);

    for (0..total) |i| {
        var timer = std.time.Timer.start() catch unreachable;
        linear_fwd.linearForward(output, input, weight, bs, d, ff, &scratch);
        const elapsed = timer.read();
        if (i >= config.warmup) samples[i - config.warmup] = elapsed;
    }

    // gate+up+down per layer × num_layers + Q+K+V+O per layer × num_layers + LM head
    const per_call = metrics.matmulFlops(bs, d, ff);
    return .{
        .name = "fwd/linear",
        .summary = harness.summarizeValues(samples),
        .flops = per_call,
        .note = "single [bs,d]@[ff,d]^T",
    };
}

fn runForwardAttention(config: RunConfig, allocator: std.mem.Allocator) !ScenarioResult {
    const mc = defaultConfig();
    const bs: usize = @as(usize, batch_size) * mc.seq_len;
    const nh = mc.num_heads;
    const nkv = mc.num_kv_heads;
    const hd = mc.headDim();
    const b: usize = batch_size;
    const s = mc.seq_len;

    const q = try allocator.alloc(f32, bs * nh * hd);
    defer allocator.free(q);
    const k = try allocator.alloc(f32, bs * nkv * hd);
    defer allocator.free(k);
    const v = try allocator.alloc(f32, bs * nkv * hd);
    defer allocator.free(v);
    const output = try allocator.alloc(f32, bs * nh * hd);
    defer allocator.free(output);
    const probs = try allocator.alloc(f32, b * nh * s * s);
    defer allocator.free(probs);

    fillRandom(q);
    fillRandom(k);
    fillRandom(v);

    const total = config.warmup + config.iters;
    const samples = try allocator.alloc(u64, config.iters);
    defer allocator.free(samples);

    for (0..total) |i| {
        var timer = std.time.Timer.start() catch unreachable;
        attention_fwd.attentionForward(output, probs, q, k, v, b, s, nh, nkv, hd);
        const elapsed = timer.read();
        if (i >= config.warmup) samples[i - config.warmup] = elapsed;
    }

    const flops = @as(u64, b) * @as(u64, nh) * (metrics.matmulFlops(s, hd, s) + metrics.matmulFlops(s, s, hd));
    return .{
        .name = "fwd/attention",
        .summary = harness.summarizeValues(samples),
        .flops = flops,
        .note = "causal full-seq",
    };
}

fn runForwardNorm(config: RunConfig, allocator: std.mem.Allocator) !ScenarioResult {
    const mc = defaultConfig();
    const bs: usize = @as(usize, batch_size) * mc.seq_len;
    const d = mc.d_model;

    const input = try allocator.alloc(f32, bs * d);
    defer allocator.free(input);
    const output = try allocator.alloc(f32, bs * d);
    defer allocator.free(output);
    const inv_rms = try allocator.alloc(f32, bs);
    defer allocator.free(inv_rms);
    const gamma = try allocator.alloc(f32, d);
    defer allocator.free(gamma);

    fillRandom(input);
    for (gamma) |*g| g.* = 1.0;

    const total = config.warmup + config.iters;
    const samples = try allocator.alloc(u64, config.iters);
    defer allocator.free(samples);

    for (0..total) |i| {
        var timer = std.time.Timer.start() catch unreachable;
        norm_fwd.rmsnormForwardSave(output, inv_rms, input, gamma, mc.norm_eps, bs, d);
        const elapsed = timer.read();
        if (i >= config.warmup) samples[i - config.warmup] = elapsed;
    }

    return .{
        .name = "fwd/norm",
        .summary = harness.summarizeValues(samples),
        .flops = @as(u64, bs) * @as(u64, d) * 3, // square + mean + scale
    };
}

fn runForwardActivation(config: RunConfig, allocator: std.mem.Allocator) !ScenarioResult {
    const mc = defaultConfig();
    const bs: usize = @as(usize, batch_size) * mc.seq_len;
    const ff = mc.d_ff;

    const gate = try allocator.alloc(f32, bs * ff);
    defer allocator.free(gate);
    const up = try allocator.alloc(f32, bs * ff);
    defer allocator.free(up);
    const output = try allocator.alloc(f32, bs * ff);
    defer allocator.free(output);

    fillRandom(gate);
    fillRandom(up);

    const total = config.warmup + config.iters;
    const samples = try allocator.alloc(u64, config.iters);
    defer allocator.free(samples);

    for (0..total) |i| {
        var timer = std.time.Timer.start() catch unreachable;
        activation_fwd.swigluForward(output, gate, up, bs * ff);
        const elapsed = timer.read();
        if (i >= config.warmup) samples[i - config.warmup] = elapsed;
    }

    return .{
        .name = "fwd/activation",
        .summary = harness.summarizeValues(samples),
        .flops = @as(u64, bs) * @as(u64, ff) * 4, // sigmoid + mul + mul + silu
    };
}

fn runForwardRope(config: RunConfig, allocator: std.mem.Allocator) !ScenarioResult {
    const mc = defaultConfig();
    const bs: usize = @as(usize, batch_size) * mc.seq_len;
    const nh = mc.num_heads;
    const hd = mc.headDim();
    const s = mc.seq_len;

    const q = try allocator.alloc(f32, bs * nh * hd);
    defer allocator.free(q);

    fillRandom(q);

    const total = config.warmup + config.iters;
    const samples = try allocator.alloc(u64, config.iters);
    defer allocator.free(samples);

    for (0..total) |i| {
        var timer = std.time.Timer.start() catch unreachable;
        rope_fwd.ropeForwardBatch(q, batch_size, s, nh, hd, hd, mc.rope_theta);
        const elapsed = timer.read();
        if (i >= config.warmup) samples[i - config.warmup] = elapsed;
    }

    return .{
        .name = "fwd/rope",
        .summary = harness.summarizeValues(samples),
        .flops = @as(u64, bs) * @as(u64, nh) * @as(u64, hd) * 6, // sin, cos, rotate pairs
    };
}

fn runForwardEmbedding(config: RunConfig, allocator: std.mem.Allocator) !ScenarioResult {
    const mc = defaultConfig();
    const bs: usize = @as(usize, batch_size) * mc.seq_len;
    const d = mc.d_model;

    const output = try allocator.alloc(f32, bs * d);
    defer allocator.free(output);
    const embed_table = try allocator.alloc(f32, mc.vocab_size * d);
    defer allocator.free(embed_table);
    const tokens = try allocator.alloc(u32, bs);
    defer allocator.free(tokens);

    fillRandom(embed_table);
    for (tokens, 0..) |*t, idx| t.* = @intCast(idx % mc.vocab_size);

    const total = config.warmup + config.iters;
    const samples = try allocator.alloc(u64, config.iters);
    defer allocator.free(samples);

    for (0..total) |i| {
        var timer = std.time.Timer.start() catch unreachable;
        embedding_fwd.embeddingForward(output, embed_table, tokens, d);
        const elapsed = timer.read();
        if (i >= config.warmup) samples[i - config.warmup] = elapsed;
    }

    return .{
        .name = "fwd/embedding",
        .summary = harness.summarizeValues(samples),
        .note = "memcpy gather",
    };
}

fn runForwardLoss(config: RunConfig, allocator: std.mem.Allocator) !ScenarioResult {
    const mc = defaultConfig();
    const bs: usize = @as(usize, batch_size) * mc.seq_len;
    const v = mc.vocab_size;

    const logits = try allocator.alloc(f32, bs * v);
    defer allocator.free(logits);
    const targets = try allocator.alloc(u32, bs);
    defer allocator.free(targets);

    fillRandom(logits);
    for (targets, 0..) |*t, idx| t.* = @intCast(idx % v);

    const total = config.warmup + config.iters;
    const samples = try allocator.alloc(u64, config.iters);
    defer allocator.free(samples);

    for (0..total) |i| {
        var timer = std.time.Timer.start() catch unreachable;
        const loss = loss_fwd.crossEntropyLoss(logits, targets, bs, v);
        const elapsed = timer.read();
        std.mem.doNotOptimizeAway(loss);
        if (i >= config.warmup) samples[i - config.warmup] = elapsed;
    }

    return .{
        .name = "fwd/loss",
        .summary = harness.summarizeValues(samples),
        .flops = @as(u64, bs) * @as(u64, v) * 3, // exp + sum + log
    };
}

fn runForwardFull(config: RunConfig, allocator: std.mem.Allocator) !ScenarioResult {
    const mc = defaultConfig();
    const bs: usize = @as(usize, batch_size) * mc.seq_len;

    var weights = try ModelWeights.init(allocator, mc);
    defer weights.deinit();
    weights.initRandom(42);

    var cache = try ActivationCache.init(allocator, mc, batch_size);
    defer cache.deinit();

    var scratch = try MatmulScratch.init(allocator);
    defer scratch.deinit();

    const tokens = try allocator.alloc(u32, bs);
    defer allocator.free(tokens);
    const targets = try allocator.alloc(u32, bs);
    defer allocator.free(targets);

    for (tokens, 0..) |*t, idx| t.* = @intCast(idx % mc.vocab_size);
    for (targets, 0..) |*t, idx| t.* = @intCast((idx + 1) % mc.vocab_size);

    const total = config.warmup + config.iters;
    const samples = try allocator.alloc(u64, config.iters);
    defer allocator.free(samples);

    for (0..total) |i| {
        var timer = std.time.Timer.start() catch unreachable;
        _ = forward_pass.forward(&weights, &cache, tokens, targets, &scratch);
        const elapsed = timer.read();
        if (i >= config.warmup) samples[i - config.warmup] = elapsed;
    }

    const flops = metrics.forwardFlops(bs, mc.d_model, mc.num_heads, mc.num_kv_heads, mc.headDim(), mc.d_ff, mc.seq_len, mc.num_layers, mc.vocab_size);
    return .{
        .name = "fwd/full",
        .summary = harness.summarizeValues(samples),
        .flops = flops,
    };
}

fn runBackwardLinear(config: RunConfig, allocator: std.mem.Allocator) !ScenarioResult {
    const mc = defaultConfig();
    const bs: usize = @as(usize, batch_size) * mc.seq_len;
    const d = mc.d_model;
    const ff = mc.d_ff;

    const grad_output = try allocator.alloc(f32, bs * d);
    defer allocator.free(grad_output);
    const input = try allocator.alloc(f32, bs * ff);
    defer allocator.free(input);
    const grad_weight = try allocator.alloc(f32, d * ff);
    defer allocator.free(grad_weight);
    const grad_input = try allocator.alloc(f32, bs * ff);
    defer allocator.free(grad_input);
    const weight = try allocator.alloc(f32, d * ff);
    defer allocator.free(weight);
    var scratch = try MatmulScratch.init(allocator);
    defer scratch.deinit();

    fillRandom(grad_output);
    fillRandom(input);
    fillRandom(weight);

    const total = config.warmup + config.iters;
    const samples = try allocator.alloc(u64, config.iters);
    defer allocator.free(samples);

    for (0..total) |i| {
        @memset(grad_weight, 0);
        var timer = std.time.Timer.start() catch unreachable;
        linear_bw.gradWeight(grad_weight, grad_output, input, bs, d, ff, &scratch);
        linear_bw.gradInput(grad_input, grad_output, weight, bs, d, ff, &scratch);
        const elapsed = timer.read();
        if (i >= config.warmup) samples[i - config.warmup] = elapsed;
    }

    const flops = metrics.matmulFlops(d, bs, ff) + metrics.matmulFlops(bs, d, ff);
    return .{
        .name = "bwd/linear",
        .summary = harness.summarizeValues(samples),
        .flops = flops,
        .note = "gradWeight+gradInput",
    };
}

fn runBackwardAttention(config: RunConfig, allocator: std.mem.Allocator) !ScenarioResult {
    const mc = defaultConfig();
    const b: usize = batch_size;
    const s = mc.seq_len;
    const nh = mc.num_heads;
    const nkv = mc.num_kv_heads;
    const hd = mc.headDim();
    const bs: usize = b * s;

    const grad_q = try allocator.alloc(f32, bs * nh * hd);
    defer allocator.free(grad_q);
    const grad_k = try allocator.alloc(f32, bs * nkv * hd);
    defer allocator.free(grad_k);
    const grad_v = try allocator.alloc(f32, bs * nkv * hd);
    defer allocator.free(grad_v);
    const grad_output = try allocator.alloc(f32, bs * nh * hd);
    defer allocator.free(grad_output);
    const query = try allocator.alloc(f32, bs * nh * hd);
    defer allocator.free(query);
    const key_cache = try allocator.alloc(f32, bs * nkv * hd);
    defer allocator.free(key_cache);
    const value_cache = try allocator.alloc(f32, bs * nkv * hd);
    defer allocator.free(value_cache);
    const attn_probs = try allocator.alloc(f32, b * nh * s * s);
    defer allocator.free(attn_probs);
    const d_scores = try allocator.alloc(f32, b * nh * s * s);
    defer allocator.free(d_scores);

    fillRandom(grad_output);
    fillRandom(query);
    fillRandom(key_cache);
    fillRandom(value_cache);
    // Fill probs with valid softmax-like values
    for (attn_probs) |*p| p.* = 0.0;
    var rng = std.Random.DefaultPrng.init(99);
    const random = rng.random();
    for (0..b) |bi| {
        for (0..nh) |h| {
            for (0..s) |qi| {
                const row = attn_probs[(bi * nh + h) * s * s + qi * s ..][0..s];
                var sum: f32 = 0.0;
                for (0..qi + 1) |t| {
                    row[t] = random.float(f32) + 0.01;
                    sum += row[t];
                }
                for (0..qi + 1) |t| row[t] /= sum;
            }
        }
    }

    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

    const total = config.warmup + config.iters;
    const samples = try allocator.alloc(u64, config.iters);
    defer allocator.free(samples);

    for (0..total) |i| {
        var timer = std.time.Timer.start() catch unreachable;
        attention_bw.attentionBackwardBatch(
            grad_q, grad_k, grad_v, grad_output,
            query, key_cache, value_cache, attn_probs, d_scores,
            b, s, nh, nkv, hd, scale,
        );
        const elapsed = timer.read();
        if (i >= config.warmup) samples[i - config.warmup] = elapsed;
    }

    return .{
        .name = "bwd/attention",
        .summary = harness.summarizeValues(samples),
        .note = "threaded 2-phase",
    };
}

fn runBackwardNorm(config: RunConfig, allocator: std.mem.Allocator) !ScenarioResult {
    const mc = defaultConfig();
    const bs: usize = @as(usize, batch_size) * mc.seq_len;
    const d = mc.d_model;

    const grad_input = try allocator.alloc(f32, bs * d);
    defer allocator.free(grad_input);
    const grad_weight = try allocator.alloc(f32, d);
    defer allocator.free(grad_weight);
    const grad_output = try allocator.alloc(f32, bs * d);
    defer allocator.free(grad_output);
    const input = try allocator.alloc(f32, bs * d);
    defer allocator.free(input);
    const inv_rms = try allocator.alloc(f32, bs);
    defer allocator.free(inv_rms);
    const weight = try allocator.alloc(f32, d);
    defer allocator.free(weight);

    fillRandom(grad_output);
    fillRandom(input);
    for (inv_rms) |*v| v.* = 1.0;
    for (weight) |*w| w.* = 1.0;

    const total = config.warmup + config.iters;
    const samples = try allocator.alloc(u64, config.iters);
    defer allocator.free(samples);

    for (0..total) |i| {
        @memset(grad_weight, 0);
        var timer = std.time.Timer.start() catch unreachable;
        norm_bw.rmsnormBackward(grad_input, grad_weight, grad_output, input, inv_rms, weight, bs, d, 0.0);
        const elapsed = timer.read();
        if (i >= config.warmup) samples[i - config.warmup] = elapsed;
    }

    return .{
        .name = "bwd/norm",
        .summary = harness.summarizeValues(samples),
        .flops = @as(u64, bs) * @as(u64, d) * 5,
    };
}

fn runBackwardActivation(config: RunConfig, allocator: std.mem.Allocator) !ScenarioResult {
    const mc = defaultConfig();
    const bs: usize = @as(usize, batch_size) * mc.seq_len;
    const ff = mc.d_ff;

    const grad_gate = try allocator.alloc(f32, bs * ff);
    defer allocator.free(grad_gate);
    const grad_up = try allocator.alloc(f32, bs * ff);
    defer allocator.free(grad_up);
    const grad_output = try allocator.alloc(f32, bs * ff);
    defer allocator.free(grad_output);
    const gate = try allocator.alloc(f32, bs * ff);
    defer allocator.free(gate);
    const up = try allocator.alloc(f32, bs * ff);
    defer allocator.free(up);

    fillRandom(grad_output);
    fillRandom(gate);
    fillRandom(up);

    const total = config.warmup + config.iters;
    const samples = try allocator.alloc(u64, config.iters);
    defer allocator.free(samples);

    for (0..total) |i| {
        var timer = std.time.Timer.start() catch unreachable;
        activation_bw.swigluBackward(grad_gate, grad_up, grad_output, gate, up);
        const elapsed = timer.read();
        if (i >= config.warmup) samples[i - config.warmup] = elapsed;
    }

    return .{
        .name = "bwd/activation",
        .summary = harness.summarizeValues(samples),
        .flops = @as(u64, bs) * @as(u64, ff) * 6,
    };
}

fn runBackwardRope(config: RunConfig, allocator: std.mem.Allocator) !ScenarioResult {
    const mc = defaultConfig();
    const bs: usize = @as(usize, batch_size) * mc.seq_len;
    const nh = mc.num_heads;
    const hd = mc.headDim();

    const grad = try allocator.alloc(f32, bs * nh * hd);
    defer allocator.free(grad);

    fillRandom(grad);

    const total = config.warmup + config.iters;
    const samples = try allocator.alloc(u64, config.iters);
    defer allocator.free(samples);

    for (0..total) |i| {
        var timer = std.time.Timer.start() catch unreachable;
        rope_bw.ropeBackwardBatch(grad, batch_size, mc.seq_len, nh, hd, hd, mc.rope_theta);
        const elapsed = timer.read();
        if (i >= config.warmup) samples[i - config.warmup] = elapsed;
    }

    return .{
        .name = "bwd/rope",
        .summary = harness.summarizeValues(samples),
        .flops = @as(u64, bs) * @as(u64, nh) * @as(u64, hd) * 6,
    };
}

fn runBackwardLoss(config: RunConfig, allocator: std.mem.Allocator) !ScenarioResult {
    const mc = defaultConfig();
    const bs: usize = @as(usize, batch_size) * mc.seq_len;
    const v = mc.vocab_size;

    const grad_logits = try allocator.alloc(f32, bs * v);
    defer allocator.free(grad_logits);
    const logits = try allocator.alloc(f32, bs * v);
    defer allocator.free(logits);
    const targets = try allocator.alloc(u32, bs);
    defer allocator.free(targets);

    fillRandom(logits);
    for (targets, 0..) |*t, idx| t.* = @intCast(idx % v);

    const total = config.warmup + config.iters;
    const samples = try allocator.alloc(u64, config.iters);
    defer allocator.free(samples);

    for (0..total) |i| {
        var timer = std.time.Timer.start() catch unreachable;
        cross_entropy_bw.crossEntropyBackward(grad_logits, logits, targets, bs, v);
        const elapsed = timer.read();
        if (i >= config.warmup) samples[i - config.warmup] = elapsed;
    }

    return .{
        .name = "bwd/loss",
        .summary = harness.summarizeValues(samples),
        .flops = @as(u64, bs) * @as(u64, v) * 4,
    };
}

fn runBackwardFull(config: RunConfig, allocator: std.mem.Allocator) !ScenarioResult {
    const mc = defaultConfig();
    const bs: usize = @as(usize, batch_size) * mc.seq_len;

    var weights = try ModelWeights.init(allocator, mc);
    defer weights.deinit();
    weights.initRandom(42);

    var cache = try ActivationCache.init(allocator, mc, batch_size);
    defer cache.deinit();

    var scratch = try MatmulScratch.init(allocator);
    defer scratch.deinit();

    const tokens = try allocator.alloc(u32, bs);
    defer allocator.free(tokens);
    const targets = try allocator.alloc(u32, bs);
    defer allocator.free(targets);

    for (tokens, 0..) |*t, idx| t.* = @intCast(idx % mc.vocab_size);
    for (targets, 0..) |*t, idx| t.* = @intCast((idx + 1) % mc.vocab_size);

    // Run forward once to populate cache
    _ = forward_pass.forward(&weights, &cache, tokens, targets, &scratch);

    const total = config.warmup + config.iters;
    const samples = try allocator.alloc(u64, config.iters);
    defer allocator.free(samples);

    for (0..total) |i| {
        weights.zeroGrads();
        var timer = std.time.Timer.start() catch unreachable;
        backward_pass.backward(&weights, &cache, tokens, targets, &scratch);
        const elapsed = timer.read();
        if (i >= config.warmup) samples[i - config.warmup] = elapsed;
    }

    return .{
        .name = "bwd/full",
        .summary = harness.summarizeValues(samples),
    };
}

fn runStepFull(config: RunConfig, allocator: std.mem.Allocator) !ScenarioResult {
    const mc = defaultConfig();
    const FullTrainingSession = train.FullTrainingSession;

    var session = FullTrainingSession.init(allocator);
    defer session.deinit();

    try session.initModel(mc, 42);
    try session.configure(.{
        .total_steps = 10000,
        .batch_size = batch_size,
        .warmup_steps = 0,
        .learning_rate = 1e-3,
    });

    const bs: usize = @as(usize, batch_size) * mc.seq_len;
    const num_tokens = bs * 10; // enough data for multiple batches
    const tokens = try allocator.alloc(u32, num_tokens);
    defer allocator.free(tokens);
    for (tokens, 0..) |*t, idx| t.* = @intCast(idx % mc.vocab_size);

    try session.setData(tokens);

    const total = config.warmup + config.iters;
    const samples = try allocator.alloc(u64, config.iters);
    defer allocator.free(samples);

    for (0..total) |i| {
        var timer = std.time.Timer.start() catch unreachable;
        _ = try session.trainStep();
        const elapsed = timer.read();
        if (i >= config.warmup) samples[i - config.warmup] = elapsed;
    }

    return .{
        .name = "step/full",
        .summary = harness.summarizeValues(samples),
        .note = "fwd+bwd+clip+opt",
    };
}

fn runOptimizerStep(config: RunConfig, allocator: std.mem.Allocator) !ScenarioResult {
    const mc = defaultConfig();
    const AdamW = train.AdamW;

    // Create parameter buffers matching a Q projection
    const param_size: usize = mc.d_model * mc.d_model;
    const params = try allocator.alloc(f32, param_size);
    defer allocator.free(params);
    const grads = try allocator.alloc(f32, param_size);
    defer allocator.free(grads);

    fillRandom(params);
    fillRandom(grads);

    var opt = AdamW.init(.{
        .lr = 1e-3,
        .weight_decay = 0.1,
    });

    const optimizer_mod = @import("main").train.optimizer;
    var opt_state = try optimizer_mod.ParamState.init(allocator, param_size);
    defer opt_state.deinit();

    const total = config.warmup + config.iters;
    const samples = try allocator.alloc(u64, config.iters);
    defer allocator.free(samples);

    for (0..total) |i| {
        var timer = std.time.Timer.start() catch unreachable;
        opt.step(params, grads, &opt_state, 1e-3);
        const elapsed = timer.read();
        if (i >= config.warmup) samples[i - config.warmup] = elapsed;
    }

    return .{
        .name = "step/optimizer",
        .summary = harness.summarizeValues(samples),
        .flops = @as(u64, param_size) * 10, // ~10 ops per param (adam update)
        .note = "AdamW single tensor",
    };
}

// =========================================================================
// Helpers
// =========================================================================

fn fillRandom(buf: []f32) void {
    var rng = std.Random.DefaultPrng.init(12345);
    const random = rng.random();
    for (buf) |*v| {
        v.* = random.float(f32) * 2.0 - 1.0;
    }
}
