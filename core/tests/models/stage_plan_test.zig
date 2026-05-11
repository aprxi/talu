//! Integration tests for the models stage plan contract.

const std = @import("std");
const main = @import("main");

const models = main.models.dispatcher;
const stage_plan = models.stage_plan;
const manifest_mod = models.manifest;
const TensorManifestEntry = manifest_mod.TensorManifestEntry;

test "StagePlan.deinit StagePlan.stage StagePlan.stageLoadRequest and validateStagePlan are exported through models root" {
    _ = stage_plan.StagePlanId;
    _ = stage_plan.PartitionConstraintSource;
    _ = stage_plan.StageRoleSemantics;
    _ = stage_plan.StagePlanValidationOptions;
    _ = stage_plan.graphIdentityEql;
    _ = stage_plan.dupeGraphIdentity;
    _ = stage_plan.deinitGraphIdentity;
    _ = stage_plan.validateGraphIdentity;
    _ = stage_plan.assertGraphIdentity;
    _ = stage_plan.validateStagePlan;

    var config = models.config.ModelConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_layers = 2,
        .n_heads = 1,
        .n_kv_groups = 1,
        .d_ff = 16,
        .max_seq_len = 32,
        .head_dim = 4,
        .rope_theta = 10000.0,
        .norm_eps = 0.00001,
        .gaffine_group_size = 32,
        .tie_word_embeddings = false,
    };
    const arch = models.op_types.Architecture{
        .name = "integration_stage_plan",
        .model_types = &.{"integration_stage_plan"},
    };
    var manifest = try makeStagePlanManifest(std.testing.allocator);
    defer manifest.deinit();

    var plan = try stage_plan.buildStagePlan(std.testing.allocator, .{
        .n_layers = 2,
        .split_points = &.{1},
        .architecture = &arch,
        .model_config = &config,
        .manifest = &manifest,
        .partition_constraints = .{ .decoder_cuts_allowed = true },
    });
    defer plan.deinit();

    try std.testing.expectEqual(stage_plan.stage_plan_contract_version, plan.stage_contract_version);
    try std.testing.expectEqual(stage_plan.PartitionConstraintSource.explicit, plan.partition_constraint_source);
    try std.testing.expectEqualSlices(usize, &.{1}, plan.split_points);
    try std.testing.expect(stage_plan.graphIdentityEql(plan.graph_identity, plan.graph_identity));
    try stage_plan.assertGraphIdentity(&plan, plan.graph_identity);
    try stage_plan.validateStagePlan(&plan, .{ .expected_graph_identity = plan.graph_identity, .manifest = &manifest });

    const first = try plan.stage(0);
    try std.testing.expectEqual(@as(usize, 0), first.layer_start);
    try std.testing.expectEqual(@as(usize, 1), first.layer_end);
    try std.testing.expectEqual(@as(usize, 0), first.residency.layer_start);
    try std.testing.expectEqual(@as(usize, 1), first.residency.layer_end);

    const load_request = try plan.stageLoadRequest(1);
    try std.testing.expectEqual(@as(usize, 1), load_request.layer_start);
    try std.testing.expectEqual(@as(usize, 2), load_request.layer_end);
    try std.testing.expect(load_request.roles.include_lm_head);
}

fn makeStagePlanManifest(allocator: std.mem.Allocator) !manifest_mod.ModelManifest {
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_allocator = arena.allocator();

    const entries = try arena_allocator.alloc(TensorManifestEntry, 5);
    entries[0] = .{
        .name = try arena_allocator.dupe(u8, "model.embed_tokens.weight"),
        .dtype = .f16,
        .shape = try arena_allocator.dupe(usize, &.{ 8, 4 }),
        .checkpoint_bytes = 64,
        .role = .token_embeddings,
        .weight_id = try arena_allocator.dupe(u8, "token_embeddings"),
        .status = .architecture_weight,
    };
    entries[1] = .{
        .name = try arena_allocator.dupe(u8, "model.layers.0.self_attn.q_proj.weight"),
        .dtype = .f16,
        .shape = try arena_allocator.dupe(usize, &.{ 4, 4 }),
        .checkpoint_bytes = 32,
        .role = .decoder_layer,
        .layer_index = 0,
        .weight_id = try arena_allocator.dupe(u8, "self_attn.q_proj.weight"),
        .status = .architecture_weight,
    };
    entries[2] = .{
        .name = try arena_allocator.dupe(u8, "model.layers.1.self_attn.q_proj.weight"),
        .dtype = .f16,
        .shape = try arena_allocator.dupe(usize, &.{ 4, 4 }),
        .checkpoint_bytes = 32,
        .role = .decoder_layer,
        .layer_index = 1,
        .weight_id = try arena_allocator.dupe(u8, "self_attn.q_proj.weight"),
        .status = .architecture_weight,
    };
    entries[3] = .{
        .name = try arena_allocator.dupe(u8, "model.norm.weight"),
        .dtype = .f32,
        .shape = try arena_allocator.dupe(usize, &.{4}),
        .checkpoint_bytes = 16,
        .role = .final_norm,
        .weight_id = try arena_allocator.dupe(u8, "ln_final"),
        .status = .architecture_weight,
    };
    entries[4] = .{
        .name = try arena_allocator.dupe(u8, "lm_head.weight"),
        .dtype = .f16,
        .shape = try arena_allocator.dupe(usize, &.{ 8, 4 }),
        .checkpoint_bytes = 64,
        .role = .lm_head,
        .weight_id = try arena_allocator.dupe(u8, "lm_head"),
        .status = .architecture_weight,
    };

    var role_bytes = [_]usize{0} ** manifest_mod.role_count;
    var total_checkpoint_bytes: usize = 0;
    for (entries) |entry| {
        total_checkpoint_bytes += entry.checkpoint_bytes;
        role_bytes[@intFromEnum(entry.role)] += entry.checkpoint_bytes;
    }

    return .{
        .arena = arena,
        .architecture_id = try arena_allocator.dupe(u8, "integration_stage_plan"),
        .layer_count = 2,
        .entries = entries,
        .total_checkpoint_bytes = total_checkpoint_bytes,
        .role_bytes = role_bytes,
    };
}
