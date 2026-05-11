//! Model tensor manifest and checkpoint residency accounting.
//!
//! This module owns metadata-only tensor ownership. It does not load tensor
//! payloads and does not estimate backend-transformed resident memory.

const std = @import("std");
const dtype_mod = @import("compute_pkg").dtype;
const st_loader = @import("io_pkg").safetensors.root;
const config_types = @import("config/types.zig");
const model_types = @import("op_types.zig");
const generic_weights = @import("load/generic_weights.zig");

const Allocator = std.mem.Allocator;
const DType = dtype_mod.DType;
const Architecture = model_types.Architecture;
const ModelConfig = config_types.ModelConfig;
const WeightSpec = model_types.WeightSpec;

pub const TensorRole = enum(u8) {
    token_embeddings,
    decoder_layer,
    final_norm,
    lm_head,
    embedding_side,
    quant_companion,
    vision_side,
    architecture_side,
    unclassified_global,
};

pub const ClassificationStatus = enum(u8) {
    architecture_weight,
    quant_companion,
    unclassified,
};

pub const TensorManifestEntry = struct {
    name: []const u8,
    dtype: DType,
    shape: []const usize,
    checkpoint_bytes: usize,
    role: TensorRole,
    owner_role: ?TensorRole = null,
    layer_index: ?usize = null,
    weight_id: ?[]const u8 = null,
    primary_name: ?[]const u8 = null,
    status: ClassificationStatus,
};

pub const StageResidencyRequest = struct {
    layer_start: usize,
    layer_end: usize,
    include_token_embeddings: bool = false,
    include_final_norm: bool = false,
    include_lm_head: bool = false,
    include_embedding_side: bool = false,
    include_architecture_side: bool = false,
    include_unclassified_global: bool = false,
};

pub const role_count = std.meta.fields(TensorRole).len;

pub const BudgetExceeded = struct {
    budget_bytes: usize,
    total_bytes: usize,
    largest_role: TensorRole,
    largest_role_bytes: usize,
};

pub const StageResidencyReport = struct {
    layer_start: usize,
    layer_end: usize,
    total_checkpoint_bytes: usize = 0,
    role_bytes: [role_count]usize = [_]usize{0} ** role_count,

    pub fn add(self: *StageResidencyReport, role: TensorRole, bytes: usize) void {
        self.total_checkpoint_bytes +|= bytes;
        self.role_bytes[@intFromEnum(role)] +|= bytes;
    }

    pub fn bytesForRole(self: *const StageResidencyReport, role: TensorRole) usize {
        return self.role_bytes[@intFromEnum(role)];
    }

    pub fn budgetExceeded(self: *const StageResidencyReport, budget_bytes: usize) ?BudgetExceeded {
        if (self.total_checkpoint_bytes <= budget_bytes) return null;

        var largest_role: TensorRole = .token_embeddings;
        var largest_role_bytes: usize = 0;
        inline for (std.meta.fields(TensorRole)) |field| {
            const role: TensorRole = @field(TensorRole, field.name);
            const bytes = self.bytesForRole(role);
            if (bytes > largest_role_bytes) {
                largest_role = role;
                largest_role_bytes = bytes;
            }
        }

        return .{
            .budget_bytes = budget_bytes,
            .total_bytes = self.total_checkpoint_bytes,
            .largest_role = largest_role,
            .largest_role_bytes = largest_role_bytes,
        };
    }
};

pub const ModelManifest = struct {
    arena: std.heap.ArenaAllocator,
    architecture_id: []const u8,
    layer_count: usize,
    entries: []const TensorManifestEntry,
    total_checkpoint_bytes: usize,
    role_bytes: [role_count]usize,

    pub fn deinit(self: *ModelManifest) void {
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn bytesForRole(self: *const ModelManifest, role: TensorRole) usize {
        return self.role_bytes[@intFromEnum(role)];
    }

    pub fn stageResidencyReport(
        self: *const ModelManifest,
        request: StageResidencyRequest,
    ) !StageResidencyReport {
        if (request.layer_start > request.layer_end or request.layer_end > self.layer_count) {
            return error.InvalidLayerRange;
        }

        var report = StageResidencyReport{
            .layer_start = request.layer_start,
            .layer_end = request.layer_end,
        };

        for (self.entries) |entry| {
            if (!shouldIncludeEntry(entry, request)) continue;
            report.add(entry.role, entry.checkpoint_bytes);
        }

        return report;
    }
};

pub fn build(
    backing_allocator: Allocator,
    arch: *const Architecture,
    model_config: *const ModelConfig,
    safetensors: *st_loader.UnifiedSafeTensors,
) !ModelManifest {
    var arena = std.heap.ArenaAllocator.init(backing_allocator);
    errdefer arena.deinit();
    const arena_allocator = arena.allocator();

    var entries = std.ArrayListUnmanaged(TensorManifestEntry){};
    var name_index = std.StringHashMapUnmanaged(usize){};
    defer name_index.deinit(backing_allocator);

    const layer_count: usize = @intCast(@max(@as(i32, 0), model_config.n_layers));

    var global_resolver: generic_weights.NameResolver = .{};
    defer global_resolver.deinit(backing_allocator);
    for (arch.global_weights) |spec| {
        var resolved_name_buf: [1024]u8 = undefined;
        if (try generic_weights.resolveWeightNameBySpec(
            backing_allocator,
            safetensors,
            spec,
            &.{},
            0,
            &global_resolver,
            resolved_name_buf[0..],
        )) |resolved_name| {
            const role = roleForGlobalWeight(spec.id);
            try addManifestEntry(
                backing_allocator,
                arena_allocator,
                &entries,
                &name_index,
                safetensors,
                resolved_name,
                role,
                null,
                null,
                spec.id,
                null,
                .architecture_weight,
            );
            try addCompanionEntries(
                backing_allocator,
                arena_allocator,
                &entries,
                &name_index,
                safetensors,
                resolved_name,
                role,
                null,
                spec.id,
            );
        }
    }

    var layer_resolver: generic_weights.NameResolver = .{};
    defer layer_resolver.deinit(backing_allocator);
    for (0..layer_count) |layer_index| {
        const variant = arch.getVariantWithOverride(layer_index, model_config.layer_types);
        const specs = if (variant) |v| v.weights else arch.block_weights;
        for (specs) |spec| {
            var resolved_name_buf: [1024]u8 = undefined;
            if (try generic_weights.resolveWeightNameBySpec(
                backing_allocator,
                safetensors,
                spec,
                arch.weight_prefixes,
                layer_index,
                &layer_resolver,
                resolved_name_buf[0..],
            )) |resolved_name| {
                try addManifestEntry(
                    backing_allocator,
                    arena_allocator,
                    &entries,
                    &name_index,
                    safetensors,
                    resolved_name,
                    .decoder_layer,
                    null,
                    layer_index,
                    spec.id,
                    null,
                    .architecture_weight,
                );
                try addCompanionEntries(
                    backing_allocator,
                    arena_allocator,
                    &entries,
                    &name_index,
                    safetensors,
                    resolved_name,
                    .decoder_layer,
                    layer_index,
                    spec.id,
                );
            }
        }
    }

    const tensor_names = try safetensors.tensorNames(backing_allocator);
    defer backing_allocator.free(tensor_names);
    for (tensor_names) |tensor_name| {
        if (name_index.contains(tensor_name)) continue;
        const role: TensorRole = if (looksLikeVisionTensor(tensor_name)) .vision_side else .unclassified_global;
        try addManifestEntry(
            backing_allocator,
            arena_allocator,
            &entries,
            &name_index,
            safetensors,
            tensor_name,
            role,
            null,
            null,
            null,
            null,
            .unclassified,
        );
    }

    var role_bytes = [_]usize{0} ** role_count;
    var total_checkpoint_bytes: usize = 0;
    for (entries.items) |entry| {
        total_checkpoint_bytes +|= entry.checkpoint_bytes;
        role_bytes[@intFromEnum(entry.role)] +|= entry.checkpoint_bytes;
    }

    return .{
        .arena = arena,
        .architecture_id = try arena_allocator.dupe(u8, arch.name),
        .layer_count = layer_count,
        .entries = entries.items,
        .total_checkpoint_bytes = total_checkpoint_bytes,
        .role_bytes = role_bytes,
    };
}

fn addManifestEntry(
    index_allocator: Allocator,
    arena_allocator: Allocator,
    entries: *std.ArrayListUnmanaged(TensorManifestEntry),
    name_index: *std.StringHashMapUnmanaged(usize),
    safetensors: *const st_loader.UnifiedSafeTensors,
    tensor_name: []const u8,
    role: TensorRole,
    owner_role: ?TensorRole,
    layer_index: ?usize,
    weight_id: ?[]const u8,
    primary_name: ?[]const u8,
    status: ClassificationStatus,
) !void {
    if (name_index.contains(tensor_name)) return;

    const metadata = try safetensors.getTensorMetadata(tensor_name);
    const stored_name = try arena_allocator.dupe(u8, metadata.name);
    const stored_shape = try arena_allocator.dupe(usize, metadata.shape);
    const stored_weight_id = if (weight_id) |id| try arena_allocator.dupe(u8, id) else null;
    const stored_primary_name = if (primary_name) |name| try arena_allocator.dupe(u8, name) else null;

    try entries.append(arena_allocator, .{
        .name = stored_name,
        .dtype = metadata.dtype,
        .shape = stored_shape,
        .checkpoint_bytes = metadata.byte_count,
        .role = role,
        .owner_role = owner_role,
        .layer_index = layer_index,
        .weight_id = stored_weight_id,
        .primary_name = stored_primary_name,
        .status = status,
    });
    try name_index.put(index_allocator, stored_name, entries.items.len - 1);
}

fn addCompanionEntries(
    index_allocator: Allocator,
    arena_allocator: Allocator,
    entries: *std.ArrayListUnmanaged(TensorManifestEntry),
    name_index: *std.StringHashMapUnmanaged(usize),
    safetensors: *const st_loader.UnifiedSafeTensors,
    primary_name: []const u8,
    owner_role: TensorRole,
    layer_index: ?usize,
    weight_id: []const u8,
) !void {
    const base = companionBase(primary_name) orelse return;
    const companion_suffixes = [_][]const u8{
        ".scales",
        ".biases",
        ".qzeros",
        ".weight_scale",
        ".weight_scale_2",
        ".weight_global_scale",
        ".weight_block_scale",
        ".weight_scale_inv",
    };

    var name_buf: [1024]u8 = undefined;
    for (companion_suffixes) |suffix| {
        const companion_name = std.fmt.bufPrint(&name_buf, "{s}{s}", .{ base, suffix }) catch continue;
        if (!safetensors.hasTensor(companion_name)) continue;
        try addManifestEntry(
            index_allocator,
            arena_allocator,
            entries,
            name_index,
            safetensors,
            companion_name,
            .quant_companion,
            owner_role,
            layer_index,
            weight_id,
            primary_name,
            .quant_companion,
        );
    }
}

fn companionBase(name: []const u8) ?[]const u8 {
    if (std.mem.endsWith(u8, name, ".weight_packed")) {
        return name[0 .. name.len - ".weight_packed".len];
    }
    if (std.mem.endsWith(u8, name, ".weight")) {
        return name[0 .. name.len - ".weight".len];
    }
    if (std.mem.endsWith(u8, name, ".qweight")) {
        return name[0 .. name.len - ".qweight".len];
    }
    return null;
}

fn roleForGlobalWeight(weight_id: []const u8) TensorRole {
    if (std.mem.eql(u8, weight_id, "token_embeddings")) return .token_embeddings;
    if (std.mem.eql(u8, weight_id, "ln_final")) return .final_norm;
    if (std.mem.eql(u8, weight_id, "lm_head")) return .lm_head;
    if (std.mem.indexOf(u8, weight_id, "embedding") != null or
        std.mem.indexOf(u8, weight_id, "embedding_ln") != null)
    {
        return .embedding_side;
    }
    return .architecture_side;
}

fn looksLikeVisionTensor(name: []const u8) bool {
    return std.mem.indexOf(u8, name, "vision") != null or
        std.mem.indexOf(u8, name, "visual") != null or
        std.mem.indexOf(u8, name, "patch_embed") != null or
        std.mem.indexOf(u8, name, "merger") != null;
}

fn shouldIncludeEntry(entry: TensorManifestEntry, request: StageResidencyRequest) bool {
    return switch (entry.role) {
        .decoder_layer => if (entry.layer_index) |layer|
            layer >= request.layer_start and layer < request.layer_end
        else
            false,
        .quant_companion => shouldIncludeCompanion(entry, request),
        .token_embeddings => request.include_token_embeddings,
        .final_norm => request.include_final_norm,
        .lm_head => request.include_lm_head,
        .embedding_side => request.include_embedding_side,
        .architecture_side, .vision_side => request.include_architecture_side,
        .unclassified_global => request.include_unclassified_global,
    };
}

fn shouldIncludeCompanion(entry: TensorManifestEntry, request: StageResidencyRequest) bool {
    if (entry.layer_index) |layer| {
        return layer >= request.layer_start and layer < request.layer_end;
    }
    return switch (entry.owner_role orelse .unclassified_global) {
        .token_embeddings => request.include_token_embeddings,
        .final_norm => request.include_final_norm,
        .lm_head => request.include_lm_head,
        .embedding_side => request.include_embedding_side,
        .architecture_side, .vision_side => request.include_architecture_side,
        .unclassified_global => request.include_unclassified_global,
        .decoder_layer, .quant_companion => false,
    };
}

test "manifest classifies tensor roles and stage residency bytes" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);
    const model_path = try std.fs.path.join(allocator, &.{ tmp_path, "model.safetensors" });
    defer allocator.free(model_path);

    const embed_data = [_]u8{0} ** 128;
    const layer0_q_data = [_]u8{1} ** 32;
    const layer1_q_data = [_]u8{2} ** 64;
    const layer1_gate_data = [_]u8{3} ** 8;
    const layer1_scale_data = [_]u8{4} ** 4;
    const layer1_scale2 = [_]u8{5} ** 4;
    const norm_data = [_]u8{6} ** 16;
    const lm_head_data = [_]u8{7} ** 128;
    const vision_data = [_]u8{8} ** 12;
    const untracked_data = [_]u8{9} ** 10;

    const entries = [_]st_loader.TensorEntry{
        .{ .name = "model.embed_tokens.weight", .dtype = .f16, .shape = &.{ 16, 4 }, .data = embed_data[0..] },
        .{ .name = "model.layers.0.self_attn.q_proj.weight", .dtype = .f16, .shape = &.{ 4, 4 }, .data = layer0_q_data[0..] },
        .{ .name = "model.layers.1.self_attn.q_proj.weight", .dtype = .f16, .shape = &.{ 8, 4 }, .data = layer1_q_data[0..] },
        .{ .name = "model.layers.1.mlp.gate_proj.weight", .dtype = .u8, .shape = &.{ 4, 2 }, .data = layer1_gate_data[0..] },
        .{ .name = "model.layers.1.mlp.gate_proj.weight_scale", .dtype = .f8_e4m3, .shape = &.{ 4, 1 }, .data = layer1_scale_data[0..] },
        .{ .name = "model.layers.1.mlp.gate_proj.weight_scale_2", .dtype = .f32, .shape = &.{1}, .data = layer1_scale2[0..] },
        .{ .name = "model.norm.weight", .dtype = .f32, .shape = &.{4}, .data = norm_data[0..] },
        .{ .name = "lm_head.weight", .dtype = .f16, .shape = &.{ 4, 16 }, .data = lm_head_data[0..] },
        .{ .name = "vision.patch_embed.weight", .dtype = .f16, .shape = &.{ 3, 2 }, .data = vision_data[0..] },
        .{ .name = "untracked.weight", .dtype = .f16, .shape = &.{5}, .data = untracked_data[0..] },
    };
    try st_loader.write(allocator, model_path, &entries);

    var safetensors = try st_loader.UnifiedSafeTensors.loadMetadataOnly(allocator, model_path);
    defer safetensors.deinit();

    const block_specs = [_]WeightSpec{
        .{
            .id = "self_attn.q_proj.weight",
            .suffix = "self_attn.q_proj.weight",
            .module_type = "Linear",
            .layout = .linear,
            .dtype = "F16",
            .required = true,
        },
        .{
            .id = "mlp.gate_proj.weight",
            .suffix = "mlp.gate_proj.weight",
            .module_type = "Linear",
            .layout = .linear,
            .dtype = "U8",
            .required = false,
        },
    };
    const global_specs = [_]WeightSpec{
        .{
            .id = "token_embeddings",
            .suffix = "model.embed_tokens.weight",
            .module_type = "Embedding",
            .layout = .embedding,
            .dtype = "F16",
            .required = true,
        },
        .{
            .id = "ln_final",
            .suffix = "model.norm.weight",
            .module_type = "RMSNorm",
            .layout = .none,
            .dtype = "F32",
            .required = false,
        },
        .{
            .id = "lm_head",
            .suffix = "lm_head.weight",
            .module_type = "Linear",
            .layout = .linear,
            .dtype = "F16",
            .required = false,
        },
    };
    const prefixes = [_][]const u8{"model.layers.{d}."};
    const arch = Architecture{
        .name = "manifest_test",
        .model_types = &.{"manifest_test"},
        .block_weights = block_specs[0..],
        .global_weights = global_specs[0..],
        .weight_prefixes = prefixes[0..],
        .weight_dtype_source_weight_ids = &.{"self_attn.q_proj.weight"},
    };
    const config = ModelConfig{
        .vocab_size = 16,
        .d_model = 4,
        .n_layers = 2,
        .n_heads = 1,
        .n_kv_groups = 1,
        .d_ff = 8,
        .max_seq_len = 16,
        .head_dim = 4,
        .rope_theta = 10000,
        .norm_eps = 0.00001,
        .gaffine_group_size = 32,
    };

    var model_manifest = try build(allocator, &arch, &config, &safetensors);
    defer model_manifest.deinit();

    try std.testing.expectEqual(@as(usize, 2), model_manifest.layer_count);
    try std.testing.expectEqual(@as(usize, 128), model_manifest.bytesForRole(.token_embeddings));
    try std.testing.expectEqual(@as(usize, 104), model_manifest.bytesForRole(.decoder_layer));
    try std.testing.expectEqual(@as(usize, 8), model_manifest.bytesForRole(.quant_companion));
    try std.testing.expectEqual(@as(usize, 12), model_manifest.bytesForRole(.vision_side));
    try std.testing.expectEqual(@as(usize, 10), model_manifest.bytesForRole(.unclassified_global));

    const layer1_report = try model_manifest.stageResidencyReport(.{
        .layer_start = 1,
        .layer_end = 2,
        .include_final_norm = true,
        .include_lm_head = true,
    });
    try std.testing.expectEqual(@as(usize, 64 + 8), layer1_report.bytesForRole(.decoder_layer));
    try std.testing.expectEqual(@as(usize, 8), layer1_report.bytesForRole(.quant_companion));
    try std.testing.expectEqual(@as(usize, 16), layer1_report.bytesForRole(.final_norm));
    try std.testing.expectEqual(@as(usize, 128), layer1_report.bytesForRole(.lm_head));
    try std.testing.expect(layer1_report.budgetExceeded(100) != null);
}
