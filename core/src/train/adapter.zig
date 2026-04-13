//! LoRA adapter abstraction.
//!
//! Adapters match against WeightSpec IDs so they work across any supported
//! model architecture without per-model adapter code. All architectures use
//! standardized IDs (e.g., "self_attn.q_proj.weight", "mlp.gate_proj.weight")
//! so a single LoRA config works for Llama, Qwen, Gemma, Granite, etc.

const std = @import("std");
const tensor_mod = @import("tensor_pkg");
const op_types = @import("models_pkg").op_types;

const Tensor = tensor_mod.Tensor;
const OwnedTensor = tensor_mod.OwnedTensor;
const DType = tensor_mod.DType;
const WeightSpec = op_types.WeightSpec;
const Architecture = op_types.Architecture;
const Allocator = std.mem.Allocator;

/// Configuration for LoRA adapter layers.
pub const LoraConfig = struct {
    /// LoRA rank (r). Typical values: 4, 8, 16, 32, 64.
    rank: u32 = 16,
    /// LoRA alpha (scaling factor). scaling = alpha / rank.
    alpha: f32 = 32.0,
};

/// A single LoRA adapter applied to one weight matrix.
///
/// LoRA decomposes a weight update as: delta_W = (A @ B) * scaling
/// where A is [rank, in_dim] and B is [out_dim, rank].
///
/// Forward: output += (input @ A^T @ B^T) * scaling
///
/// A is initialized with Kaiming uniform, B is initialized to zero,
/// so the adapter has no effect at initialization.
pub const LoraLayer = struct {
    /// WeightSpec.id this adapter applies to (e.g., "self_attn.q_proj.weight").
    weight_id: []const u8,
    /// Layer index this adapter applies to (which transformer block).
    layer_index: usize,
    /// Low-rank matrix A: [rank, in_dim], initialized with scaled random values.
    A: OwnedTensor,
    /// Low-rank matrix B: [out_dim, rank], initialized to zero.
    B: OwnedTensor,
    /// Scaling factor: alpha / rank.
    scaling: f32,
    /// The rank used.
    rank: u32,
    /// Input dimension of the adapted weight.
    in_dim: usize,
    /// Output dimension of the adapted weight.
    out_dim: usize,

    /// Create a new LoRA layer for a weight with the given dimensions.
    ///
    /// A is initialized with Kaiming uniform: values in [-1/sqrt(rank), 1/sqrt(rank)]
    /// using a simple deterministic pattern (not random — deterministic for reproducibility).
    /// B is initialized to zero so the adapter starts with no effect.
    pub fn init(
        allocator: Allocator,
        weight_id: []const u8,
        layer_index: usize,
        in_dim: usize,
        out_dim: usize,
        config: LoraConfig,
    ) !LoraLayer {
        const rank: usize = config.rank;
        var A = try OwnedTensor.init(allocator, .f32, &.{ rank, in_dim });
        errdefer A.deinit();

        // Kaiming-style initialization: scale by 1/sqrt(rank)
        const bound: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(rank)));
        const a_data = A.asSlice(f32);
        for (a_data, 0..) |*v, i| {
            // Deterministic pseudo-random: simple hash-based pattern
            const hash = hashIndex(i);
            v.* = (hash - 0.5) * 2.0 * bound;
        }

        const B = try OwnedTensor.init(allocator, .f32, &.{ out_dim, rank });
        // B is already zero-initialized by OwnedTensor.init

        return .{
            .weight_id = weight_id,
            .layer_index = layer_index,
            .A = A,
            .B = B,
            .scaling = config.alpha / @as(f32, @floatFromInt(config.rank)),
            .rank = config.rank,
            .in_dim = in_dim,
            .out_dim = out_dim,
        };
    }

    /// Total number of trainable parameters in this adapter layer.
    pub fn paramCount(self: *const LoraLayer) usize {
        return self.A.numElements() + self.B.numElements();
    }

    pub fn deinit(self: *LoraLayer) void {
        self.A.deinit();
        self.B.deinit();
        self.* = undefined;
    }
};

/// Pattern for selecting which weights get adapters.
pub const TargetPattern = union(enum) {
    /// Match exact WeightSpec IDs.
    exact: []const []const u8,
    /// Match weights whose module_type matches (e.g., "Linear").
    module_type: []const u8,
    /// Match weights whose ID contains the given substring.
    contains: []const u8,
};

/// Collection of LoRA layers applied to a model.
///
/// Created by matching a TargetPattern against an Architecture's WeightSpecs.
/// The adapter set is model-agnostic — it works with any architecture that
/// follows the WeightSpec naming convention.
pub const LoraAdapter = struct {
    layers: std.ArrayListUnmanaged(LoraLayer),
    config: LoraConfig,
    allocator: Allocator,

    /// Create a LoRA adapter set for a given architecture.
    ///
    /// Walks the architecture's block_weights (per-layer) and global_weights,
    /// matching each WeightSpec.id against the target pattern. Creates a LoraLayer
    /// for each match.
    ///
    /// `num_layers` is the number of transformer blocks in the model.
    /// `dim_resolver` provides (in_dim, out_dim) for a given weight spec ID.
    pub fn init(
        allocator: Allocator,
        architecture: *const Architecture,
        num_layers: usize,
        target: TargetPattern,
        config: LoraConfig,
        dim_resolver: DimResolver,
    ) !LoraAdapter {
        var layers = std.ArrayListUnmanaged(LoraLayer){};
        errdefer {
            for (layers.items) |*layer| layer.deinit();
            layers.deinit(allocator);
        }

        // Match against block weights (repeated per layer)
        const block_specs = if (architecture.block_variants) |variants|
            // Heterogeneous: use first variant's weights (attention variant)
            variants[0].weights
        else
            architecture.block_weights;

        for (0..num_layers) |layer_idx| {
            for (block_specs) |spec| {
                if (matchesPattern(spec, target)) {
                    const dims = dim_resolver.resolve(spec.id, layer_idx);
                    if (dims) |d| {
                        const layer = try LoraLayer.init(
                            allocator,
                            spec.id,
                            layer_idx,
                            d.in_dim,
                            d.out_dim,
                            config,
                        );
                        try layers.append(allocator, layer);
                    }
                }
            }

            // For heterogeneous models, also check other variants
            if (architecture.block_variants) |variants| {
                for (variants[1..]) |variant| {
                    const variant_idx = architecture.getVariantIndex(layer_idx);
                    // Only add adapters for the variant actually used by this layer
                    if (variant_idx == 0) continue;
                    for (variant.weights) |spec| {
                        if (matchesPattern(spec, target)) {
                            const dims = dim_resolver.resolve(spec.id, layer_idx);
                            if (dims) |d| {
                                const layer = try LoraLayer.init(
                                    allocator,
                                    spec.id,
                                    layer_idx,
                                    d.in_dim,
                                    d.out_dim,
                                    config,
                                );
                                try layers.append(allocator, layer);
                            }
                        }
                    }
                }
            }
        }

        return .{
            .layers = layers,
            .config = config,
            .allocator = allocator,
        };
    }

    /// Create a LoRA adapter with explicitly listed layers (no architecture needed).
    /// Useful for testing or when dimensions are known ahead of time.
    pub fn initExplicit(allocator: Allocator, config: LoraConfig) LoraAdapter {
        return .{
            .layers = .{},
            .config = config,
            .allocator = allocator,
        };
    }

    /// Add a LoRA layer explicitly.
    pub fn addLayer(self: *LoraAdapter, layer: LoraLayer) !void {
        try self.layers.append(self.allocator, layer);
    }

    /// Find the LoRA layer for a given weight_id and layer_index.
    /// Returns null if no adapter is applied to this weight at this layer.
    pub fn getLayer(self: *const LoraAdapter, weight_id: []const u8, layer_index: usize) ?*const LoraLayer {
        for (self.layers.items) |*layer| {
            if (layer.layer_index == layer_index and std.mem.eql(u8, layer.weight_id, weight_id)) {
                return layer;
            }
        }
        return null;
    }

    /// Total trainable parameter count across all adapter layers.
    pub fn trainableParamCount(self: *const LoraAdapter) usize {
        var total: usize = 0;
        for (self.layers.items) |*layer| {
            total += layer.paramCount();
        }
        return total;
    }

    /// Number of adapter layers.
    pub fn layerCount(self: *const LoraAdapter) usize {
        return self.layers.items.len;
    }

    pub fn deinit(self: *LoraAdapter) void {
        for (self.layers.items) |*layer| {
            layer.deinit();
        }
        self.layers.deinit(self.allocator);
        self.* = undefined;
    }
};

/// Resolves (in_dim, out_dim) for a given weight spec ID and layer index.
pub const Dims = struct {
    in_dim: usize,
    out_dim: usize,
};

pub const DimResolver = struct {
    ctx: *const anyopaque,
    resolveFn: *const fn (ctx: *const anyopaque, weight_id: []const u8, layer_index: usize) ?Dims,

    pub fn resolve(self: DimResolver, weight_id: []const u8, layer_index: usize) ?Dims {
        return self.resolveFn(self.ctx, weight_id, layer_index);
    }

    /// Create a DimResolver from a simple fixed-dimension config.
    /// All matched weights get the same (in_dim, out_dim).
    pub fn fixed(config: *const FixedDimConfig) DimResolver {
        return .{
            .ctx = @ptrCast(config),
            .resolveFn = FixedDimConfig.resolveFn,
        };
    }
};

/// Simple dimension config where all adapted weights share the same dimensions.
pub const FixedDimConfig = struct {
    in_dim: usize,
    out_dim: usize,

    fn resolveFn(ctx: *const anyopaque, _: []const u8, _: usize) ?Dims {
        const self: *const FixedDimConfig = @ptrCast(@alignCast(ctx));
        return .{ .in_dim = self.in_dim, .out_dim = self.out_dim };
    }
};

// =============================================================================
// Internal helpers
// =============================================================================

/// Check if a WeightSpec matches a target pattern.
fn matchesPattern(spec: WeightSpec, target: TargetPattern) bool {
    return switch (target) {
        .exact => |ids| {
            for (ids) |id| {
                if (std.mem.eql(u8, spec.id, id)) return true;
            }
            return false;
        },
        .module_type => |mt| std.mem.eql(u8, spec.module_type, mt),
        .contains => |substr| std.mem.indexOf(u8, spec.id, substr) != null,
    };
}

/// Deterministic pseudo-random value in [0, 1) for initialization.
/// Uses a simple hash to avoid needing a PRNG.
fn hashIndex(i: usize) f32 {
    // Fibonacci hashing
    const golden: u64 = 0x9E3779B97F4A7C15;
    const h = @as(u64, @intCast(i)) *% golden;
    // Map to [0, 1)
    return @as(f32, @floatFromInt(h >> 40)) / @as(f32, @floatFromInt(@as(u64, 1) << 24));
}

// =============================================================================
// Tests
// =============================================================================

test "LoraLayer init creates correct shapes" {
    const allocator = std.testing.allocator;
    var layer = try LoraLayer.init(allocator, "self_attn.q_proj.weight", 0, 256, 256, .{ .rank = 8, .alpha = 16.0 });
    defer layer.deinit();

    // A: [rank, in_dim] = [8, 256]
    try std.testing.expectEqual(@as(usize, 8), @as(usize, @intCast(layer.A.shape[0])));
    try std.testing.expectEqual(@as(usize, 256), @as(usize, @intCast(layer.A.shape[1])));

    // B: [out_dim, rank] = [256, 8]
    try std.testing.expectEqual(@as(usize, 256), @as(usize, @intCast(layer.B.shape[0])));
    try std.testing.expectEqual(@as(usize, 8), @as(usize, @intCast(layer.B.shape[1])));

    // scaling = alpha / rank = 16 / 8 = 2.0
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), layer.scaling, 1e-6);
}

test "LoraLayer B is zero-initialized" {
    const allocator = std.testing.allocator;
    var layer = try LoraLayer.init(allocator, "test", 0, 16, 16, .{ .rank = 4, .alpha = 8.0 });
    defer layer.deinit();

    for (layer.B.asSlice(f32)) |v| {
        try std.testing.expectEqual(@as(f32, 0.0), v);
    }
}

test "LoraLayer A has non-zero values" {
    const allocator = std.testing.allocator;
    var layer = try LoraLayer.init(allocator, "test", 0, 32, 32, .{ .rank = 8, .alpha = 16.0 });
    defer layer.deinit();

    var has_nonzero = false;
    for (layer.A.asSlice(f32)) |v| {
        if (v != 0.0) {
            has_nonzero = true;
            break;
        }
    }
    try std.testing.expect(has_nonzero);
}

test "LoraLayer paramCount" {
    const allocator = std.testing.allocator;
    var layer = try LoraLayer.init(allocator, "test", 0, 256, 128, .{ .rank = 16, .alpha = 32.0 });
    defer layer.deinit();

    // A: 16*256 = 4096, B: 128*16 = 2048, total = 6144
    try std.testing.expectEqual(@as(usize, 6144), layer.paramCount());
}

test "matchesPattern exact" {
    const spec = WeightSpec{
        .id = "self_attn.q_proj.weight",
        .suffix = "self_attn.q_proj.weight",
        .module_type = "Linear",
        .layout = .linear,
        .dtype = "float32",
        .required = true,
    };

    const target_ids = [_][]const u8{ "self_attn.q_proj.weight", "self_attn.v_proj.weight" };
    try std.testing.expect(matchesPattern(spec, .{ .exact = &target_ids }));

    const non_match = [_][]const u8{"mlp.gate_proj.weight"};
    try std.testing.expect(!matchesPattern(spec, .{ .exact = &non_match }));
}

test "matchesPattern module_type" {
    const linear_spec = WeightSpec{
        .id = "self_attn.q_proj.weight",
        .suffix = "self_attn.q_proj.weight",
        .module_type = "Linear",
        .layout = .linear,
        .dtype = "float32",
        .required = true,
    };
    const norm_spec = WeightSpec{
        .id = "input_layernorm.weight",
        .suffix = "input_layernorm.weight",
        .module_type = "RMSNorm",
        .layout = .none,
        .dtype = "float32",
        .required = true,
    };

    try std.testing.expect(matchesPattern(linear_spec, .{ .module_type = "Linear" }));
    try std.testing.expect(!matchesPattern(norm_spec, .{ .module_type = "Linear" }));
    try std.testing.expect(matchesPattern(norm_spec, .{ .module_type = "RMSNorm" }));
}

test "matchesPattern contains" {
    const q_spec = WeightSpec{
        .id = "self_attn.q_proj.weight",
        .suffix = "self_attn.q_proj.weight",
        .module_type = "Linear",
        .layout = .linear,
        .dtype = "float32",
        .required = true,
    };
    const gate_spec = WeightSpec{
        .id = "mlp.gate_proj.weight",
        .suffix = "mlp.gate_proj.weight",
        .module_type = "Linear",
        .layout = .linear,
        .dtype = "float32",
        .required = true,
    };

    try std.testing.expect(matchesPattern(q_spec, .{ .contains = "self_attn" }));
    try std.testing.expect(!matchesPattern(gate_spec, .{ .contains = "self_attn" }));
    try std.testing.expect(matchesPattern(gate_spec, .{ .contains = "mlp" }));
}

test "LoraAdapter initExplicit and addLayer" {
    const allocator = std.testing.allocator;
    var adapter = LoraAdapter.initExplicit(allocator, .{ .rank = 8, .alpha = 16.0 });
    defer adapter.deinit();

    var layer1 = try LoraLayer.init(allocator, "self_attn.q_proj.weight", 0, 64, 64, .{ .rank = 8, .alpha = 16.0 });
    errdefer layer1.deinit();
    try adapter.addLayer(layer1);

    var layer2 = try LoraLayer.init(allocator, "self_attn.v_proj.weight", 0, 64, 64, .{ .rank = 8, .alpha = 16.0 });
    errdefer layer2.deinit();
    try adapter.addLayer(layer2);

    try std.testing.expectEqual(@as(usize, 2), adapter.layerCount());
}

test "LoraAdapter getLayer finds by id and layer_index" {
    const allocator = std.testing.allocator;
    var adapter = LoraAdapter.initExplicit(allocator, .{ .rank = 4, .alpha = 8.0 });
    defer adapter.deinit();

    var layer = try LoraLayer.init(allocator, "self_attn.q_proj.weight", 2, 32, 32, .{ .rank = 4, .alpha = 8.0 });
    errdefer layer.deinit();
    try adapter.addLayer(layer);

    // Should find it at layer 2
    const found = adapter.getLayer("self_attn.q_proj.weight", 2);
    try std.testing.expect(found != null);
    try std.testing.expectEqual(@as(usize, 2), found.?.layer_index);

    // Should not find it at layer 0
    try std.testing.expect(adapter.getLayer("self_attn.q_proj.weight", 0) == null);

    // Should not find wrong id
    try std.testing.expect(adapter.getLayer("mlp.gate_proj.weight", 2) == null);
}

test "LoraAdapter trainableParamCount sums all layers" {
    const allocator = std.testing.allocator;
    var adapter = LoraAdapter.initExplicit(allocator, .{ .rank = 4, .alpha = 8.0 });
    defer adapter.deinit();

    // Layer 1: in=16, out=16, rank=4 -> A=4*16=64, B=16*4=64, total=128
    var l1 = try LoraLayer.init(allocator, "a", 0, 16, 16, .{ .rank = 4, .alpha = 8.0 });
    errdefer l1.deinit();
    try adapter.addLayer(l1);

    // Layer 2: in=32, out=16, rank=4 -> A=4*32=128, B=16*4=64, total=192
    var l2 = try LoraLayer.init(allocator, "b", 0, 32, 16, .{ .rank = 4, .alpha = 8.0 });
    errdefer l2.deinit();
    try adapter.addLayer(l2);

    try std.testing.expectEqual(@as(usize, 128 + 192), adapter.trainableParamCount());
}

test "hashIndex returns values in [0, 1)" {
    for (0..100) |i| {
        const v = hashIndex(i);
        try std.testing.expect(v >= 0.0);
        try std.testing.expect(v < 1.0);
    }
}

test "hashIndex is deterministic" {
    try std.testing.expectEqual(hashIndex(42), hashIndex(42));
    try std.testing.expectEqual(hashIndex(0), hashIndex(0));
}
