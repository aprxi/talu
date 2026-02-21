//! Integration tests for models onboarding and runtime-architecture contracts.

const std = @import("std");
const main = @import("main");

const models = main.models.dispatcher;
const registry = models.registry;

fn writeConfigForModelType(
    allocator: std.mem.Allocator,
    tmp: *std.testing.TmpDir,
    model_type: []const u8,
) ![]u8 {
    const config_json = try std.fmt.allocPrint(allocator, "{{\"model_type\":\"{s}\"}}", .{model_type});
    defer allocator.free(config_json);

    try tmp.dir.writeFile(.{ .sub_path = "config.json", .data = config_json });
    return tmp.dir.realpathAlloc(allocator, "config.json");
}

test "onboarding resolves runtime architecture for every registered model_type" {
    for (registry.entries) |entry| {
        const expected_arch = registry.runtimeArchitectureById(entry.id) orelse return error.MissingArchitecture;
        for (entry.model_types) |model_type| {
            var tmp = std.testing.tmpDir(.{});
            defer tmp.cleanup();

            const config_path = try writeConfigForModelType(std.testing.allocator, &tmp, model_type);
            defer std.testing.allocator.free(config_path);

            const resolved = try models.resolveModelKindForConfig(std.testing.allocator, config_path);
            try std.testing.expectEqualStrings(entry.id, resolved.descriptor.id);
            try std.testing.expect(resolved.runtime_arch == expected_arch);

            const detected_arch = try models.runtimeArchitectureForConfig(std.testing.allocator, config_path);
            try std.testing.expect(detected_arch == expected_arch);
        }
    }
}

test "onboarding rejects unsupported model_type with typed error" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{ .sub_path = "config.json", .data = "{\"model_type\":\"not_supported\"}" });
    const config_path = try tmp.dir.realpathAlloc(std.testing.allocator, "config.json");
    defer std.testing.allocator.free(config_path);

    try std.testing.expectError(error.UnsupportedModel, models.runtimeArchitectureForConfig(std.testing.allocator, config_path));
}

test "onboarding rejects missing model_type with typed error" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{ .sub_path = "config.json", .data = "{}" });
    const config_path = try tmp.dir.realpathAlloc(std.testing.allocator, "config.json");
    defer std.testing.allocator.free(config_path);

    try std.testing.expectError(error.UnsupportedModel, models.runtimeArchitectureForConfig(std.testing.allocator, config_path));
}

test "applyRuntimeArchitectureMetadata writes runtime contract fields" {
    const arch = registry.runtimeArchitectureById("llama3") orelse return error.MissingArchitecture;

    var cfg = std.mem.zeroes(@FieldType(models.LoadedModel, "config"));
    cfg.model_arch = .custom;
    cfg.vocab_size = 1;
    cfg.n_layers = 0;
    cfg.d_model = 4;
    cfg.n_heads = 1;
    cfg.n_kv_groups = 1;
    cfg.head_dim = 4;
    cfg.d_ff = 4;
    cfg.max_seq_len = 16;
    cfg.rope_theta = 10000.0;
    cfg.norm_eps = 1e-5;
    cfg.gaffine_group_size = 128;

    var loaded: models.LoadedModel = undefined;
    loaded.config = cfg;
    loaded.runtime = .{};

    models.applyRuntimeArchitectureMetadata(&loaded, arch);

    try std.testing.expect(loaded.runtime.architecture_id != null);
    try std.testing.expectEqualStrings(arch.name, loaded.runtime.architecture_id.?);
    try std.testing.expectEqual(arch.has_moe, loaded.runtime.has_moe);
    try std.testing.expectEqual(arch.has_mamba, loaded.runtime.has_mamba);
    try std.testing.expectEqual(arch.has_shortconv, loaded.runtime.has_shortconv);
    try std.testing.expectEqual(arch.has_mla, loaded.runtime.has_mla);
    try std.testing.expectEqual(arch.explicit_qk_norm_ops, loaded.runtime.explicit_qk_norm_ops);
}
