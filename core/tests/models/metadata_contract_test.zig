//! Integration tests for model metadata contracts and registry invariants.

const std = @import("std");
const main = @import("main");
const report = @import("report.zig");

const registry = main.models.dispatcher.registry;
const op_types = main.models.dispatcher.op_types;
const WeightSpec = op_types.WeightSpec;

fn hasWeightId(specs: []const WeightSpec, id: []const u8) bool {
    for (specs) |spec| {
        if (std.mem.eql(u8, spec.id, id)) return true;
    }
    return false;
}

fn expectUniqueWeightSpecIds(specs: []const WeightSpec) !void {
    var ids = std.StringHashMap(void).init(std.testing.allocator);
    defer ids.deinit();

    for (specs) |spec| {
        try std.testing.expect(!ids.contains(spec.id));
        try ids.put(spec.id, {});
    }
}

test "registry architecture ids are unique and runtime architecture exists" {
    var ids = std.StringHashMap(void).init(std.testing.allocator);
    defer ids.deinit();

    for (registry.entries) |entry| {
        try std.testing.expect(!ids.contains(entry.id));
        try ids.put(entry.id, {});
        try std.testing.expect(registry.runtimeArchitectureById(entry.id) != null);
    }
}

test "registry model_type aliases are unique and resolve to owning architecture" {
    var aliases = std.StringHashMap([]const u8).init(std.testing.allocator);
    defer aliases.deinit();

    for (registry.entries) |entry| {
        const arch = registry.runtimeArchitectureById(entry.id) orelse return error.MissingArchitecture;
        for (entry.model_types) |model_type| {
            if (aliases.get(model_type)) |existing_owner| {
                std.debug.print("duplicate model_type alias: {s} owners={s},{s}\n", .{ model_type, existing_owner, entry.id });
                return error.DuplicateModelTypeAlias;
            }
            try aliases.put(model_type, entry.id);

            const detected = registry.runtimeArchitectureByModelType(model_type) orelse return error.MissingArchitecture;
            try std.testing.expect(detected == arch);
        }
    }
}

test "variant layer mapping and aliases are in range" {
    for (registry.entries) |entry| {
        const arch = registry.runtimeArchitectureById(entry.id) orelse return error.MissingArchitecture;
        if (arch.block_variants) |variants| {
            if (arch.layer_map) |layer_map| {
                for (layer_map) |variant_index| {
                    try std.testing.expect(@as(usize, variant_index) < variants.len);
                }
            }
            if (arch.variant_aliases) |aliases| {
                for (aliases) |alias| {
                    try std.testing.expect(@as(usize, alias.variant_index) < variants.len);
                }
            }
        } else {
            try std.testing.expect(arch.layer_map == null);
            try std.testing.expect(arch.variant_aliases == null);
        }
    }
}

test "metadata completeness contract passes for all registered architectures" {
    for (registry.entries) |entry| {
        const arch = registry.runtimeArchitectureById(entry.id) orelse return error.MissingArchitecture;
        const completeness = try report.evaluateArchitecture(std.testing.allocator, arch);
        if (!completeness.ok()) {
            std.debug.print(
                "metadata completeness failed for {s}: has_mlp_or_moe={d} d_ff_source_configured={d} d_ff_source_ids_exist={d} has_shortconv_feature={d} shortconv_source_configured={d} shortconv_source_exists={d} moe_params_declared={d}\n",
                .{
                    entry.id,
                    @intFromBool(completeness.has_mlp_or_moe),
                    @intFromBool(completeness.d_ff_source_configured),
                    @intFromBool(completeness.d_ff_source_ids_exist),
                    @intFromBool(completeness.has_shortconv_feature),
                    @intFromBool(completeness.shortconv_source_configured),
                    @intFromBool(completeness.shortconv_source_exists),
                    @intFromBool(completeness.moe_params_declared),
                },
            );
        }
        try std.testing.expect(completeness.ok());
    }
}

test "weight spec ids are unique in each declared scope" {
    for (registry.entries) |entry| {
        const arch = registry.runtimeArchitectureById(entry.id) orelse return error.MissingArchitecture;
        try expectUniqueWeightSpecIds(arch.global_weights);
        try expectUniqueWeightSpecIds(arch.block_weights);
        if (arch.block_variants) |variants| {
            for (variants) |variant| {
                try expectUniqueWeightSpecIds(variant.weights);
            }
        }
    }
}

test "architectures declare token_embeddings in global weights" {
    for (registry.entries) |entry| {
        const arch = registry.runtimeArchitectureById(entry.id) orelse return error.MissingArchitecture;
        try std.testing.expect(hasWeightId(arch.global_weights, "token_embeddings"));
    }
}
