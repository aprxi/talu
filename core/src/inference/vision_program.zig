//! Vision LayerOp program parsing helpers.
//!
//! Keeps backend vision runtime initialization aligned with model-declared
//! vision pipeline metadata.

const std = @import("std");
const layer_ops = @import("../models/layer_ops.zig");
const runtime_contract = @import("runtime_contract/root.zig");
const plan_compiler = @import("../models/plan/compiler.zig");

pub const ParsedVisionProgram = struct {
    spatial_merge_size: usize,
    deepstack_visual_layers: [8]usize,
    deepstack_layer_count: usize,
    scatter_image_token_id: u32,
};

/// Fixed staged handoff contract for multimodal vision execution.
///
/// `vision_encode` drives patch/deepstack/spatial-merge operations.
/// `scatter` drives token-stream embedding scatter.
pub const VisionStagePlans = struct {
    vision_encode: runtime_contract.CompiledPlan,
    scatter: runtime_contract.CompiledPlan,
    scatter_image_token_id: u32,
};

/// Parse model-declared vision pipeline ops and resolve runtime parameters.
/// The program must declare at least one `.patch_embed`, `.spatial_merge`,
/// and `.scatter` op.
pub fn parseVisionProgram(
    program: []const layer_ops.LayerOp,
    default_spatial_merge_size: usize,
    default_deepstack_visual_layers: [8]usize,
    default_deepstack_layer_count: usize,
) !ParsedVisionProgram {
    if (default_spatial_merge_size == 0) return error.InvalidSpatialMergeSize;
    if (default_deepstack_layer_count > default_deepstack_visual_layers.len) {
        return error.InvalidDeepstackConfig;
    }

    var has_patch_embed = false;
    var has_spatial_merge = false;
    var has_scatter = false;
    var spatial_merge_size = default_spatial_merge_size;
    var scatter_image_token_id: ?u32 = null;

    var deepstack_visual_layers: [8]usize = [_]usize{0} ** 8;
    var deepstack_layer_count: usize = 0;

    for (program) |op| {
        switch (op) {
            .patch_embed => has_patch_embed = true,
            .spatial_merge => |merge_op| {
                has_spatial_merge = true;
                const merge_size = std.math.cast(usize, merge_op.merge_size) orelse return error.InvalidSpatialMergeSize;
                if (merge_size == 0) return error.InvalidSpatialMergeSize;
                spatial_merge_size = merge_size;
            },
            .deepstack_extract => |extract_op| {
                if (deepstack_layer_count >= deepstack_visual_layers.len) return error.TooManyDeepstackLayers;
                const layer_idx = std.math.cast(usize, extract_op.layer_index) orelse return error.InvalidDeepstackLayer;
                deepstack_visual_layers[deepstack_layer_count] = layer_idx;
                deepstack_layer_count += 1;
            },
            .scatter => |scatter_op| {
                has_scatter = true;
                if (scatter_image_token_id) |existing| {
                    if (existing != scatter_op.image_token_id) return error.InvalidScatterImageTokenId;
                } else {
                    scatter_image_token_id = scatter_op.image_token_id;
                }
            },
            else => {},
        }
    }

    if (!has_patch_embed) return error.MissingPatchEmbedOp;
    if (!has_spatial_merge) return error.MissingSpatialMergeOp;
    if (!has_scatter) return error.MissingScatterOp;
    if (scatter_image_token_id == null) return error.MissingScatterOp;

    if (deepstack_layer_count == 0) {
        deepstack_layer_count = default_deepstack_layer_count;
        if (deepstack_layer_count > deepstack_visual_layers.len) {
            return error.InvalidDeepstackConfig;
        }
        for (0..deepstack_layer_count) |idx| {
            deepstack_visual_layers[idx] = default_deepstack_visual_layers[idx];
        }
    }

    return .{
        .spatial_merge_size = spatial_merge_size,
        .deepstack_visual_layers = deepstack_visual_layers,
        .deepstack_layer_count = deepstack_layer_count,
        .scatter_image_token_id = scatter_image_token_id.?,
    };
}

/// Validate register-level handoff between vision_encode and scatter plans.
///
/// Ensures the vision encoder's final output register matches the scatter
/// stage's expected vision embedding input, and that scatter outputs to
/// register 0 (residual) for decoder consumption.
fn validateVisionStageHandoff(
    vision_encode: *const runtime_contract.CompiledPlan,
    scatter: *const runtime_contract.CompiledPlan,
) !void {
    if (vision_encode.plan.instructions.len == 0) return error.InvalidVisionProgram;
    if (scatter.plan.instructions.len == 0) return error.InvalidVisionProgram;

    const encode_output = runtime_contract.planFinalOutputRegister(&vision_encode.plan);
    const scatter_insn = scatter.plan.instructions[0];

    // Scatter inputs: [text_in, vision_in]. Vision encode output must match vision_in.
    if (scatter_insn.inputs.len < 2) return error.InvalidVisionStageHandoff;
    if (@intFromEnum(encode_output) != @intFromEnum(scatter_insn.inputs[1])) {
        return error.InvalidVisionStageHandoff;
    }

    // Scatter output must be register 0 (residual) for decoder handoff.
    if (scatter_insn.outputs.len == 0) return error.InvalidVisionStageHandoff;
    if (runtime_contract.registerToIndex(scatter_insn.outputs[0]) != 0) {
        return error.InvalidVisionStageHandoff;
    }
}

/// Compile staged vision plans from a single model-declared vision program.
///
/// This creates explicit handoff boundaries:
/// - `vision_encode`: patch/deepstack/spatial-merge
/// - `scatter`: token scatter
pub fn compileVisionStagePlans(
    allocator: std.mem.Allocator,
    program: []const layer_ops.LayerOp,
) !VisionStagePlans {
    // Ensure required stage ops are present and coherent first.
    const parsed = try parseVisionProgram(program, 1, [_]usize{0} ** 8, 0);

    var encode_ops = std.ArrayListUnmanaged(layer_ops.LayerOp){};
    defer encode_ops.deinit(allocator);
    var scatter_ops = std.ArrayListUnmanaged(layer_ops.LayerOp){};
    defer scatter_ops.deinit(allocator);

    for (program) |op| {
        switch (op) {
            .scatter => try scatter_ops.append(allocator, op),
            else => try encode_ops.append(allocator, op),
        }
    }

    if (encode_ops.items.len == 0 or scatter_ops.items.len == 0) {
        return error.InvalidVisionProgram;
    }

    var vision_encode = try plan_compiler.compileLayerProgram(
        allocator,
        encode_ops.items,
        .vision_encode,
    );
    errdefer plan_compiler.deinitCompiledPlan(allocator, &vision_encode);

    var scatter = try plan_compiler.compileLayerProgram(
        allocator,
        scatter_ops.items,
        .scatter,
    );
    errdefer plan_compiler.deinitCompiledPlan(allocator, &scatter);

    // Scatter stage must contain only scatter opcodes.
    for (scatter.plan.instructions) |insn| {
        if (insn.opcode != .vision_scatter) return error.InvalidVisionProgram;
    }

    try validateVisionStageHandoff(&vision_encode, &scatter);

    return .{
        .vision_encode = vision_encode,
        .scatter = scatter,
        .scatter_image_token_id = parsed.scatter_image_token_id,
    };
}

pub fn deinitVisionStagePlans(
    allocator: std.mem.Allocator,
    plans: *VisionStagePlans,
) void {
    plan_compiler.deinitCompiledPlan(allocator, &plans.vision_encode);
    plan_compiler.deinitCompiledPlan(allocator, &plans.scatter);
    plans.* = undefined;
}

test "parseVisionProgram enforces required patch and merge ops" {
    const program = [_]layer_ops.LayerOp{
        .{ .deepstack_extract = .{ .in = .norm_out, .out = .tmp3, .layer_index = 1 } },
    };
    try std.testing.expectError(
        error.MissingPatchEmbedOp,
        parseVisionProgram(&program, 2, [_]usize{0} ** 8, 0),
    );
}

test "parseVisionProgram applies merge and deepstack overrides from program" {
    const program = [_]layer_ops.LayerOp{
        .{ .patch_embed = .{ .in = .residual, .out = .norm_out } },
        .{ .deepstack_extract = .{ .in = .norm_out, .out = .tmp3, .layer_index = 4 } },
        .{ .deepstack_extract = .{ .in = .norm_out, .out = .tmp4, .layer_index = 9 } },
        .{ .spatial_merge = .{ .in = .norm_out, .out = .branch_out, .merge_size = 3 } },
        .{ .scatter = .{
            .text_in = .residual,
            .vision_in = .branch_out,
            .out = .residual,
            .image_token_id = 42,
        } },
    };
    const parsed = try parseVisionProgram(&program, 2, [_]usize{0} ** 8, 0);
    try std.testing.expectEqual(@as(usize, 3), parsed.spatial_merge_size);
    try std.testing.expectEqual(@as(usize, 2), parsed.deepstack_layer_count);
    try std.testing.expectEqual(@as(usize, 4), parsed.deepstack_visual_layers[0]);
    try std.testing.expectEqual(@as(usize, 9), parsed.deepstack_visual_layers[1]);
    try std.testing.expectEqual(@as(u32, 42), parsed.scatter_image_token_id);
}

test "compileVisionStagePlans splits encode and scatter stages" {
    const program = [_]layer_ops.LayerOp{
        .{ .patch_embed = .{ .in = .residual, .out = .norm_out } },
        .{ .spatial_merge = .{ .in = .norm_out, .out = .branch_out, .merge_size = 2 } },
        .{ .scatter = .{
            .text_in = .residual,
            .vision_in = .branch_out,
            .out = .residual,
            .image_token_id = 151655,
        } },
    };
    var plans = try compileVisionStagePlans(std.testing.allocator, &program);
    defer deinitVisionStagePlans(std.testing.allocator, &plans);

    try std.testing.expect(plans.vision_encode.plan.instructions.len >= 2);
    try std.testing.expectEqual(@as(usize, 1), plans.scatter.plan.instructions.len);
    try std.testing.expectEqual(runtime_contract.Opcode.vision_scatter, plans.scatter.plan.instructions[0].opcode);
    try std.testing.expectEqual(@as(u32, 151655), plans.scatter_image_token_id);
}

test "compileVisionStagePlans rejects mismatched handoff registers" {
    // spatial_merge outputs to norm_out, but scatter expects vision_in = branch_out.
    // This mismatch should be caught by handoff validation.
    const program = [_]layer_ops.LayerOp{
        .{ .patch_embed = .{ .in = .residual, .out = .norm_out } },
        .{ .spatial_merge = .{ .in = .norm_out, .out = .norm_out, .merge_size = 2 } },
        .{ .scatter = .{
            .text_in = .residual,
            .vision_in = .branch_out,
            .out = .residual,
            .image_token_id = 151655,
        } },
    };
    try std.testing.expectError(
        error.InvalidVisionStageHandoff,
        compileVisionStagePlans(std.testing.allocator, &program),
    );
}
