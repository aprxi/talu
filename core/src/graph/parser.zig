//! Graph Parser
//!
//! Parses compute graph definitions from JSON.
//! Used for runtime architecture registration via the C API.

const std = @import("std");
const json_mod = @import("../io/json/root.zig");
const Allocator = std.mem.Allocator;

const types = @import("types.zig");
const Op = types.Op;
const OpType = types.OpType;
const OpInput = types.OpInput;
const Architecture = types.Architecture;

// =============================================================================
// Public API
// =============================================================================

/// Parse architecture definition from JSON string.
pub fn parseFromJson(allocator: Allocator, json_str: []const u8) !Architecture {
    const parsed = json_mod.parseValue(allocator, json_str, .{
        .max_size_bytes = 10 * 1024 * 1024,
        .max_value_bytes = 10 * 1024 * 1024,
        .max_string_bytes = 1 * 1024 * 1024,
    }) catch |err| {
        return switch (err) {
            error.InputTooLarge => error.InvalidJson,
            error.InputTooDeep => error.InvalidJson,
            error.StringTooLong => error.InvalidJson,
            error.InvalidJson => error.InvalidJson,
            error.OutOfMemory => error.OutOfMemory,
        };
    };
    defer parsed.deinit();

    if (parsed.value != .object) return error.InvalidJson;
    const obj = parsed.value.object;

    // Parse name
    const name_value = obj.get("name") orelse return error.MissingName;
    const name = switch (name_value) {
        .string => |s| try allocator.dupe(u8, s),
        else => return error.InvalidName,
    };
    errdefer allocator.free(name);

    // Parse model_types
    const model_types_json = obj.get("model_types") orelse return error.MissingModelTypes;
    const model_types = switch (model_types_json) {
        .array => |arr| blk: {
            var model_type_names = try allocator.alloc([]const u8, arr.items.len);
            errdefer allocator.free(model_type_names);
            var filled: usize = 0;
            errdefer for (model_type_names[0..filled]) |s| allocator.free(s);
            for (arr.items) |model_type_item| {
                model_type_names[filled] = switch (model_type_item) {
                    .string => |s| try allocator.dupe(u8, s),
                    else => return error.InvalidModelType,
                };
                filled += 1;
            }
            break :blk model_type_names;
        },
        else => return error.InvalidModelTypes,
    };
    errdefer {
        for (model_types) |model_type_str| allocator.free(model_type_str);
        allocator.free(model_types);
    }

    // Parse pre_block ops (optional)
    const pre_block_ops: []const Op = if (obj.get("pre_block")) |json_val| switch (json_val) {
        .array => |arr| try parseOps(allocator, arr.items),
        else => &.{},
    } else &.{};
    errdefer if (pre_block_ops.len > 0) allocator.free(pre_block_ops);

    // Parse post_block ops (optional)
    const post_block_ops: []const Op = if (obj.get("post_block")) |json_val| switch (json_val) {
        .array => |arr| try parseOps(allocator, arr.items),
        else => &.{},
    } else &.{};
    errdefer if (post_block_ops.len > 0) allocator.free(post_block_ops);

    // Parse weight_prefixes (optional) - used to generate candidates for block weights
    const weight_prefixes = try parseWeightPrefixes(allocator, obj.get("weight_prefixes"));
    errdefer if (weight_prefixes.len > 0) {
        for (weight_prefixes) |p| allocator.free(p);
        allocator.free(weight_prefixes);
    };

    // Parse global weights (optional) - these have explicit candidates, not prefixes
    const global_weights = try parseWeightSpecs(allocator, obj.get("global_weights"), &.{});
    errdefer if (global_weights.len > 0) freeWeightSpecs(allocator, global_weights);

    // Check for heterogeneous model (block_variants) or homogeneous (block)
    var block_ops: []const Op = &.{};
    var block_variants: ?[]types.BlockVariant = null;
    var layer_map: ?[]const u8 = null;
    var variant_aliases: ?[]const types.VariantAlias = null;
    var has_mamba = false;
    var has_shortconv = false;
    var block_weights: []const types.WeightSpec = &.{};

    if (obj.get("block_variants")) |variants_json| {
        // Heterogeneous model: parse block_variants and layer_map
        var variant_names: ?[]const []const u8 = null;
        if (obj.get("variant_names")) |names_json| {
            variant_names = switch (names_json) {
                .array => |arr| blk: {
                    var names = try allocator.alloc([]const u8, arr.items.len);
                    errdefer allocator.free(names);
                    var filled: usize = 0;
                    errdefer for (names[0..filled]) |n| allocator.free(n);
                    for (arr.items) |item| {
                        const variant_name = switch (item) {
                            .string => |s| try allocator.dupe(u8, s),
                            else => return error.InvalidVariantNames,
                        };
                        names[filled] = variant_name;
                        filled += 1;
                    }
                    break :blk names;
                },
                else => return error.InvalidVariantNames,
            };
        }
        defer if (variant_names) |names| {
            for (names) |n| allocator.free(n);
            allocator.free(names);
        };

        const result = try parseBlockVariants(allocator, variants_json, variant_names, weight_prefixes);
        block_variants = result.variants;
        has_mamba = result.has_mamba;
        has_shortconv = result.has_shortconv;
        errdefer if (block_variants) |v| {
            for (v) |variant| {
                allocator.free(variant.name);
                allocator.free(variant.ops);
                if (variant.weights.len > 0) {
                    freeWeightSpecs(allocator, variant.weights);
                }
            }
            allocator.free(v);
        };

        // Parse layer_map (required for heterogeneous models)
        layer_map = if (obj.get("layer_map")) |map_json| switch (map_json) {
            .array => |arr| try parseLayerMap(allocator, arr.items),
            else => null,
        } else null;
        errdefer if (layer_map) |m| allocator.free(m);

        if (layer_map == null) return error.MissingLayerMap;

        // Parse variant_aliases (optional - maps alternate config strings to variant indices)
        if (obj.get("variant_aliases")) |aliases_json| {
            switch (aliases_json) {
                .object => |alias_obj| {
                    if (alias_obj.count() > 0) {
                        var aliases = try allocator.alloc(types.VariantAlias, alias_obj.count());
                        errdefer allocator.free(aliases);
                        var idx: usize = 0;
                        var iter = alias_obj.iterator();
                        while (iter.next()) |entry| {
                            const target_name = switch (entry.value_ptr.*) {
                                .string => |s| s,
                                else => continue,
                            };
                            // Resolve target variant name to index
                            var target_idx: u8 = 0;
                            if (variant_names) |names| {
                                for (names, 0..) |vname, vi| {
                                    if (std.mem.eql(u8, target_name, vname)) {
                                        target_idx = @intCast(vi);
                                        break;
                                    }
                                }
                            }
                            aliases[idx] = .{
                                .alias = try allocator.dupe(u8, entry.key_ptr.*),
                                .variant_index = target_idx,
                            };
                            idx += 1;
                        }
                        variant_aliases = aliases[0..idx];
                    }
                },
                else => {},
            }
        }
    } else if (obj.get("block")) |block_json| {
        // Homogeneous model: parse single block
        block_ops = switch (block_json) {
            .array => |arr| try parseOps(allocator, arr.items),
            else => return error.InvalidBlock,
        };
        errdefer allocator.free(block_ops);

        block_weights = try parseWeightSpecs(allocator, obj.get("block_weights"), weight_prefixes);
        errdefer if (block_weights.len > 0) freeWeightSpecs(allocator, block_weights);
    } else {
        return error.MissingBlock;
    }

    // Analyze ops to derive flags (use first variant's ops for heterogeneous)
    const ops_to_analyze = if (block_variants) |v|
        if (v.len > 0) v[0].ops else &[_]Op{}
    else
        block_ops;
    const flags = analyzeOps(ops_to_analyze, pre_block_ops);

    return Architecture{
        .name = name,
        .model_types = model_types,
        .block_ops = block_ops,
        .pre_block_ops = pre_block_ops,
        .post_block_ops = post_block_ops,
        .block_variants = block_variants,
        .layer_map = layer_map,
        .variant_aliases = variant_aliases,
        .block_weights = block_weights,
        .global_weights = global_weights,
        .weight_prefixes = weight_prefixes,
        .has_qk_norm = flags.has_qk_norm,
        .has_moe = flags.has_moe,
        .has_mamba = has_mamba,
        .has_shortconv = has_shortconv,
        .has_mla = flags.has_mla,
        .has_fused_qkv = flags.has_fused_qkv,
        .has_fused_gate_up = flags.has_fused_gate_up,
        .num_norms_per_block = flags.num_norms,
        .use_gelu = flags.use_gelu,
        .use_swiglu_oss = flags.use_swiglu_oss,
        .embedding_multiplier = flags.embedding_multiplier,
        .norm_weight_offset = flags.norm_weight_offset,
        .explicit_qk_norm_ops = flags.explicit_qk_norm_ops,
    };
}

// =============================================================================
// Block Variant Parsing (for heterogeneous models)
// =============================================================================

const BlockVariantResult = struct {
    variants: []types.BlockVariant,
    has_mamba: bool,
    has_shortconv: bool,
};

fn parseBlockVariants(
    allocator: Allocator,
    variants_json: std.json.Value,
    variant_names: ?[]const []const u8,
    weight_prefixes: []const []const u8,
) !BlockVariantResult {
    // block_variants is an object: { "mamba": { "ops": [...] }, "attention": { "ops": [...] } }
    if (variants_json != .object) return error.InvalidBlockVariants;
    const variants_obj = variants_json.object;

    const variant_count = variants_obj.count();
    if (variant_names) |names| {
        if (names.len != variant_count) return error.InvalidVariantNames;
    }

    var variants = try allocator.alloc(types.BlockVariant, variant_count);
    errdefer allocator.free(variants);

    var idx: usize = 0;
    var has_mamba = false;
    var has_shortconv = false;
    var filled: usize = 0;
    errdefer for (variants[0..filled]) |variant| {
        allocator.free(variant.name);
        allocator.free(variant.ops);
        if (variant.weights.len > 0) freeWeightSpecs(allocator, variant.weights);
        if (variant.compiled_program) |prog| allocator.free(prog);
    };

    if (variant_names) |names| {
        for (names) |variant_name_raw| {
            const variant_obj_val = variants_obj.get(variant_name_raw) orelse return error.MissingBlockVariant;
            const variant_name = try allocator.dupe(u8, variant_name_raw);
            errdefer allocator.free(variant_name);

            // Each variant is { "ops": [...] }
            const variant_obj = switch (variant_obj_val) {
                .object => |o| o,
                else => return error.InvalidBlockVariant,
            };

            const ops_json = variant_obj.get("ops") orelse return error.MissingVariantOps;
            const ops = switch (ops_json) {
                .array => |arr| try parseOps(allocator, arr.items),
                else => return error.InvalidVariantOps,
            };
            errdefer allocator.free(ops);

            const weights = try parseWeightSpecs(allocator, variant_obj.get("weights"), weight_prefixes);
            errdefer if (weights.len > 0) freeWeightSpecs(allocator, weights);

            // Check if this variant has mamba_mixer or shortconv ops
            for (ops) |op| {
                if (op.op_type == .mamba_mixer) has_mamba = true;
                if (op.op_type == .shortconv) has_shortconv = true;
            }

            variants[idx] = types.BlockVariant{
                .name = variant_name,
                .ops = ops,
                .weights = weights,
                .compiled_program = null,
            };
            idx += 1;
            filled += 1;
        }
    } else {
        var it = variants_obj.iterator();
        while (it.next()) |entry| {
            const variant_name = try allocator.dupe(u8, entry.key_ptr.*);
            errdefer allocator.free(variant_name);

            // Each variant is { "ops": [...] }
            const variant_obj = switch (entry.value_ptr.*) {
                .object => |o| o,
                else => return error.InvalidBlockVariant,
            };

            const ops_json = variant_obj.get("ops") orelse return error.MissingVariantOps;
            const ops = switch (ops_json) {
                .array => |arr| try parseOps(allocator, arr.items),
                else => return error.InvalidVariantOps,
            };
            errdefer allocator.free(ops);

            const weights = try parseWeightSpecs(allocator, variant_obj.get("weights"), weight_prefixes);
            errdefer if (weights.len > 0) freeWeightSpecs(allocator, weights);

            // Check if this variant has mamba_mixer or shortconv ops
            for (ops) |op| {
                if (op.op_type == .mamba_mixer) has_mamba = true;
                if (op.op_type == .shortconv) has_shortconv = true;
            }

            variants[idx] = types.BlockVariant{
                .name = variant_name,
                .ops = ops,
                .weights = weights,
                .compiled_program = null,
            };
            idx += 1;
            filled += 1;
        }
    }

    return BlockVariantResult{
        .variants = variants,
        .has_mamba = has_mamba,
        .has_shortconv = has_shortconv,
    };
}

fn parseLayerMap(allocator: Allocator, items: []const std.json.Value) ![]const u8 {
    var layer_map = try allocator.alloc(u8, items.len);
    errdefer allocator.free(layer_map);

    for (items, 0..) |item, i| {
        layer_map[i] = switch (item) {
            .integer => |int| if (int >= 0 and int <= 255) @intCast(int) else return error.InvalidLayerMapValue,
            else => return error.InvalidLayerMapValue,
        };
    }
    return layer_map;
}

// =============================================================================
// Weight Spec Parsing
// =============================================================================

/// Parse weight_prefixes array from JSON.
fn parseWeightPrefixes(allocator: Allocator, obj: ?std.json.Value) ![]const []const u8 {
    if (obj == null) return &.{};
    const value = obj.?;
    if (value != .array) return error.InvalidWeightPrefixes;
    const items = value.array.items;
    if (items.len == 0) return &.{};

    var prefixes = try allocator.alloc([]const u8, items.len);
    errdefer allocator.free(prefixes);
    var filled: usize = 0;
    errdefer for (prefixes[0..filled]) |p| allocator.free(p);

    for (items, 0..) |item, i| {
        if (item != .string) return error.InvalidWeightPrefix;
        prefixes[i] = try allocator.dupe(u8, item.string);
        filled += 1;
    }
    return prefixes;
}

/// Parse weight specs from JSON.
/// If weight_prefixes is provided and a spec lacks explicit candidates,
/// generates candidates by joining each prefix with the weight's id (suffix).
fn parseWeightSpecs(
    allocator: Allocator,
    obj: ?std.json.Value,
    weight_prefixes: []const []const u8,
) ![]types.WeightSpec {
    if (obj == null) return &.{};
    const value = obj.?;
    if (value != .array) return error.InvalidWeightSpecs;

    const items = value.array.items;
    if (items.len == 0) return &.{};

    var specs = try allocator.alloc(types.WeightSpec, items.len);
    errdefer allocator.free(specs);

    var filled: usize = 0;
    errdefer for (specs[0..filled]) |spec| freeWeightSpec(allocator, spec);

    for (items, 0..) |item, i| {
        if (item != .object) return error.InvalidWeightSpec;
        const spec_obj = item.object;

        const id_val = spec_obj.get("id") orelse return error.MissingWeightId;
        if (id_val != .string) return error.InvalidWeightId;
        const id = try allocator.dupe(u8, id_val.string);
        errdefer allocator.free(id);

        // Parse candidates: either explicit array or generate from weight_prefixes + id
        const candidates: []const []const u8 = if (spec_obj.get("candidates")) |candidates_val| blk: {
            // Explicit candidates array
            if (candidates_val != .array) return error.InvalidCandidates;
            const candidates_items = candidates_val.array.items;
            var cands = try allocator.alloc([]const u8, candidates_items.len);
            errdefer allocator.free(cands);
            var candidates_filled: usize = 0;
            errdefer for (cands[0..candidates_filled]) |c| allocator.free(c);
            for (candidates_items, 0..) |cand, j| {
                if (cand != .string) return error.InvalidCandidate;
                cands[j] = try allocator.dupe(u8, cand.string);
                candidates_filled += 1;
            }
            break :blk cands;
        } else if (weight_prefixes.len > 0) blk: {
            // Generate candidates from prefixes + id (suffix)
            var cands = try allocator.alloc([]const u8, weight_prefixes.len);
            errdefer allocator.free(cands);
            var candidates_filled: usize = 0;
            errdefer for (cands[0..candidates_filled]) |c| allocator.free(c);
            for (weight_prefixes, 0..) |prefix, j| {
                cands[j] = try std.fmt.allocPrint(allocator, "{s}{s}", .{ prefix, id_val.string });
                candidates_filled += 1;
            }
            break :blk cands;
        } else {
            return error.MissingCandidates;
        };
        errdefer {
            for (candidates) |c| allocator.free(c);
            allocator.free(candidates);
        }

        const module_type_val = spec_obj.get("module_type") orelse return error.MissingModuleType;
        if (module_type_val != .string) return error.InvalidModuleType;
        const module_type = try allocator.dupe(u8, module_type_val.string);
        errdefer allocator.free(module_type);

        const layout = try parseWeightLayout(spec_obj.get("layout"));

        const dtype_val = spec_obj.get("dtype") orelse return error.MissingWeightDtype;
        if (dtype_val != .string) return error.InvalidWeightDtype;
        const dtype = try allocator.dupe(u8, dtype_val.string);
        errdefer allocator.free(dtype);

        const required = if (spec_obj.get("required")) |req_val| switch (req_val) {
            .bool => |b| b,
            else => return error.InvalidWeightRequired,
        } else true;

        const expected_shape = try parseOptionalShape(allocator, spec_obj.get("expected_shape"));
        errdefer if (expected_shape) |shape| allocator.free(shape);

        const transforms = try parseTransforms(allocator, spec_obj.get("transforms"));
        errdefer if (transforms.len > 0) allocator.free(transforms);

        specs[i] = .{
            .id = id,
            .candidates = candidates,
            .module_type = module_type,
            .layout = layout,
            .dtype = dtype,
            .required = required,
            .expected_shape = expected_shape,
            .transforms = transforms,
        };
        filled += 1;
    }

    return specs;
}

fn parseWeightLayout(obj: ?std.json.Value) !types.WeightLayout {
    if (obj == null) return .none;
    if (obj.? != .string) return error.InvalidWeightLayout;
    const layout_str = obj.?.string;
    if (std.mem.eql(u8, layout_str, "none")) return .none;
    if (std.mem.eql(u8, layout_str, "linear")) return .linear;
    if (std.mem.eql(u8, layout_str, "conv1d_depthwise")) return .conv1d_depthwise;
    if (std.mem.eql(u8, layout_str, "embedding")) return .embedding;
    if (std.mem.eql(u8, layout_str, "gaffine")) return .gaffine;
    return error.UnknownWeightLayout;
}

fn parseTransforms(allocator: Allocator, obj: ?std.json.Value) ![]types.WeightTransform {
    if (obj == null) return &.{};
    const value = obj.?;
    if (value != .array) return error.InvalidWeightTransforms;
    const items = value.array.items;
    if (items.len == 0) return &.{};

    var transforms = try allocator.alloc(types.WeightTransform, items.len);
    errdefer allocator.free(transforms);

    for (items, 0..) |item, i| {
        if (item != .string) return error.InvalidWeightTransform;
        const name = item.string;
        transforms[i] = if (std.mem.eql(u8, name, "transpose"))
            .transpose
        else if (std.mem.eql(u8, name, "maybe_transpose"))
            .maybe_transpose
        else if (std.mem.eql(u8, name, "quantize_gaffine"))
            .quantize_gaffine
        else if (std.mem.eql(u8, name, "quantize_fp8"))
            .quantize_fp8
        else if (std.mem.eql(u8, name, "dtype_f32"))
            .dtype_f32
        else
            return error.UnknownWeightTransform;
    }

    return transforms;
}

fn parseOptionalShape(allocator: Allocator, obj: ?std.json.Value) !?[]const usize {
    if (obj == null) return null;
    const value = obj.?;
    if (value != .array) return error.InvalidWeightShape;
    const items = value.array.items;
    if (items.len == 0) return &.{};

    var shape = try allocator.alloc(usize, items.len);
    errdefer allocator.free(shape);

    for (items, 0..) |item, i| {
        shape[i] = switch (item) {
            .integer => |int| if (int >= 0) @intCast(int) else return error.InvalidWeightShapeValue,
            else => return error.InvalidWeightShapeValue,
        };
    }
    return shape;
}

fn freeWeightSpec(allocator: Allocator, spec: types.WeightSpec) void {
    allocator.free(spec.id);
    for (spec.candidates) |cand| allocator.free(cand);
    allocator.free(spec.candidates);
    allocator.free(spec.module_type);
    allocator.free(spec.dtype);
    if (spec.expected_shape) |shape| allocator.free(shape);
    if (spec.transforms.len > 0) allocator.free(spec.transforms);
}

fn freeWeightSpecs(allocator: Allocator, specs: []const types.WeightSpec) void {
    for (specs) |spec| freeWeightSpec(allocator, spec);
    allocator.free(specs);
}

// =============================================================================
// Op Parsing
// =============================================================================

fn parseOps(allocator: Allocator, items: []const std.json.Value) ![]const Op {
    var ops = try allocator.alloc(Op, items.len);
    errdefer allocator.free(ops);

    for (items, 0..) |item, op_idx| {
        ops[op_idx] = try parseOp(allocator, item);
    }
    return ops;
}

fn parseOp(allocator: Allocator, value: std.json.Value) !Op {
    if (value != .object) return error.InvalidBlockOp;
    const op_json = value.object;

    // Parse op type
    const op_name = switch (op_json.get("op") orelse return error.MissingOpType) {
        .string => |s| s,
        else => return error.InvalidOpType,
    };

    const op_type = parseOpType(op_name) orelse return error.UnknownOpType;

    // Parse optional fields
    const name: ?[]const u8 = if (op_json.get("name")) |json_val| switch (json_val) {
        .string => |s| try allocator.dupe(u8, s),
        else => null,
    } else null;

    const activation: ?[]const u8 = if (op_json.get("activation")) |json_val| switch (json_val) {
        .string => |s| try allocator.dupe(u8, s),
        else => null,
    } else null;

    const weight_offset: f32 = if (op_json.get("weight_offset")) |json_val| switch (json_val) {
        .float => |f| @floatCast(f),
        .integer => |int_val| @floatFromInt(int_val),
        else => 0.0,
    } else 0.0;

    // Parse inputs array
    const inputs: []const OpInput = if (op_json.get("inputs")) |json_val| switch (json_val) {
        .array => |arr| try parseInputs(allocator, arr.items),
        else => &.{},
    } else &.{};

    // Parse split_sizes array
    const split_sizes: []const i32 = if (op_json.get("split_sizes")) |json_val| switch (json_val) {
        .array => |arr| blk: {
            var sizes = allocator.alloc(i32, arr.items.len) catch break :blk &[_]i32{};
            for (arr.items, 0..) |item, size_idx| {
                sizes[size_idx] = switch (item) {
                    .integer => |int| @intCast(int),
                    else => 0,
                };
            }
            break :blk sizes;
        },
        else => &.{},
    } else &.{};

    const dim0: i32 = if (op_json.get("dim0")) |json_val| switch (json_val) {
        .integer => |int| @intCast(int),
        else => -1,
    } else -1;

    const dim1: i32 = if (op_json.get("dim1")) |json_val| switch (json_val) {
        .integer => |int| @intCast(int),
        else => -1,
    } else -1;

    // Parse shape array
    const shape: []const i32 = if (op_json.get("shape")) |json_val| switch (json_val) {
        .array => |arr| blk: {
            var dims = allocator.alloc(i32, arr.items.len) catch break :blk &[_]i32{};
            for (arr.items, 0..) |item, dim_idx| {
                dims[dim_idx] = switch (item) {
                    .integer => |int| @intCast(int),
                    .string => |s| if (std.mem.eql(u8, s, "B"))
                        -2
                    else if (std.mem.eql(u8, s, "T"))
                        -3
                    else
                        -1,
                    else => -1,
                };
            }
            break :blk dims;
        },
        else => &.{},
    } else &.{};

    // Parse outputs array
    const outputs: []const []const u8 = if (op_json.get("outputs")) |json_val| switch (json_val) {
        .array => |arr| blk: {
            var outs = allocator.alloc([]const u8, arr.items.len) catch break :blk &[_][]const u8{};
            for (arr.items, 0..) |item, out_idx| {
                outs[out_idx] = switch (item) {
                    .string => |s| allocator.dupe(u8, s) catch "",
                    else => "",
                };
            }
            break :blk outs;
        },
        else => &.{},
    } else &.{};

    // Parse config flags
    var qk_norm = getBool(op_json, "qk_norm") orelse false;
    var fused_qkv = getBool(op_json, "fused_qkv") orelse false;
    var fused_gate_up = getBool(op_json, "fused_gate_up") orelse false;

    // Also check "config" array for flags
    if (op_json.get("config")) |config_val| {
        if (config_val == .array) {
            for (config_val.array.items) |item| {
                if (item == .string) {
                    if (std.mem.eql(u8, item.string, "qk_norm")) {
                        qk_norm = true;
                    } else if (std.mem.eql(u8, item.string, "fused_qkv")) {
                        fused_qkv = true;
                    } else if (std.mem.eql(u8, item.string, "fused_gate_up")) {
                        fused_gate_up = true;
                    }
                }
            }
        }
    }

    return Op{
        .op_type = op_type,
        .name = name,
        .inputs = inputs,
        .outputs = outputs,
        .weight_offset = weight_offset,
        .qk_norm = qk_norm,
        .fused_qkv = fused_qkv,
        .fused_gate_up = fused_gate_up,
        .sliding_window = getInt(op_json, "sliding_window"),
        .activation = activation,
        .num_experts = getInt(op_json, "num_experts") orelse 0,
        .experts_per_token = getInt(op_json, "experts_per_token") orelse 0,
        .scale = getFloat(op_json, "scale") orelse 1.0,
        .num_outputs = getInt(op_json, "num_outputs") orelse 0,
        .dim = getInt(op_json, "dim") orelse -1,
        .keepdim = getBool(op_json, "keepdim") orelse false,
        .exponent = getFloat(op_json, "exponent") orelse 1.0,
        .shape = shape,
        .split_sizes = split_sizes,
        .dim0 = dim0,
        .dim1 = dim1,
        .is_causal = getBool(op_json, "is_causal") orelse true,
        .sdpa_scale = getFloat(op_json, "sdpa_scale"),
        // Mamba-specific fields
        .d_state = getU32(op_json, "d_state"),
        .d_conv = getU32(op_json, "d_conv"),
        .n_heads = getU32(op_json, "n_heads"),
        .d_head = getU32(op_json, "d_head"),
        .n_groups = getU32(op_json, "n_groups"),
        .d_inner = getU32(op_json, "d_inner"),
        // MLA (Multi-Latent Attention) fields
        .mla = getBool(op_json, "mla") orelse false,
        .q_lora_rank = getU32(op_json, "q_lora_rank"),
        .kv_lora_rank = getU32(op_json, "kv_lora_rank"),
        .qk_head_dim = getU32(op_json, "qk_head_dim"),
        .qk_rope_head_dim = getU32(op_json, "qk_rope_head_dim"),
        .qk_nope_head_dim = getU32(op_json, "qk_nope_head_dim"),
        .v_head_dim = getU32(op_json, "v_head_dim"),
        .rope_interleave = getBool(op_json, "rope_interleave") orelse true,
    };
}

fn parseOpType(s: []const u8) ?OpType {
    if (std.mem.eql(u8, s, "norm")) return .norm;
    if (std.mem.eql(u8, s, "multihead_attention") or std.mem.eql(u8, s, "attention")) return .multihead_attention;
    if (std.mem.eql(u8, s, "mlp")) return .mlp;
    if (std.mem.eql(u8, s, "moe")) return .moe;
    if (std.mem.eql(u8, s, "mamba_mixer")) return .mamba_mixer;
    if (std.mem.eql(u8, s, "shortconv")) return .shortconv;
    if (std.mem.eql(u8, s, "add")) return .add;
    if (std.mem.eql(u8, s, "mul")) return .mul;
    if (std.mem.eql(u8, s, "mean")) return .mean;
    if (std.mem.eql(u8, s, "pow")) return .pow;
    if (std.mem.eql(u8, s, "rsqrt")) return .rsqrt;
    if (std.mem.eql(u8, s, "matmul")) return .matmul;
    if (std.mem.eql(u8, s, "split")) return .split;
    if (std.mem.eql(u8, s, "transpose")) return .transpose;
    if (std.mem.eql(u8, s, "reshape")) return .reshape;
    if (std.mem.eql(u8, s, "softmax")) return .softmax;
    if (std.mem.eql(u8, s, "silu")) return .silu;
    if (std.mem.eql(u8, s, "gelu")) return .gelu;
    if (std.mem.eql(u8, s, "embedding")) return .embedding;
    if (std.mem.eql(u8, s, "linear")) return .linear;
    if (std.mem.eql(u8, s, "rope")) return .rope;
    if (std.mem.eql(u8, s, "triu")) return .triu;
    if (std.mem.eql(u8, s, "scaled_dot_product_attention")) return .scaled_dot_product_attention;
    return null;
}

fn parseInputs(allocator: Allocator, items: []const std.json.Value) ![]const OpInput {
    var inputs = try allocator.alloc(OpInput, items.len);
    errdefer allocator.free(inputs);

    for (items, 0..) |item, input_idx| {
        inputs[input_idx] = try parseInput(allocator, item);
    }
    return inputs;
}

fn parseInput(allocator: Allocator, value: std.json.Value) !OpInput {
    const obj = switch (value) {
        .object => |o| o,
        else => return error.InvalidInput,
    };

    if (obj.get("tensor")) |json_val| {
        return switch (json_val) {
            .string => |s| .{ .tensor = try allocator.dupe(u8, s) },
            else => error.InvalidInput,
        };
    }

    if (obj.get("scalar")) |json_val| {
        return switch (json_val) {
            .float => |f| .{ .scalar = @floatCast(f) },
            .integer => |int_val| .{ .scalar = @floatFromInt(int_val) },
            else => error.InvalidInput,
        };
    }

    return error.InvalidInput;
}

// =============================================================================
// JSON Helpers
// =============================================================================

fn getBool(obj: std.json.ObjectMap, key: []const u8) ?bool {
    const value = obj.get(key) orelse return null;
    return switch (value) {
        .bool => |b| b,
        else => null,
    };
}

fn getInt(obj: std.json.ObjectMap, key: []const u8) ?i32 {
    const value = obj.get(key) orelse return null;
    return switch (value) {
        .integer => |int_val| @intCast(int_val),
        else => null,
    };
}

fn getFloat(obj: std.json.ObjectMap, key: []const u8) ?f32 {
    const value = obj.get(key) orelse return null;
    return switch (value) {
        .float => |f| @floatCast(f),
        .integer => |int_val| @floatFromInt(int_val),
        else => null,
    };
}

fn getU32(obj: std.json.ObjectMap, key: []const u8) ?u32 {
    const value = obj.get(key) orelse return null;
    return switch (value) {
        .integer => |int_val| if (int_val >= 0) @intCast(int_val) else null,
        else => null,
    };
}

// =============================================================================
// Op Analysis
// =============================================================================

const AnalysisFlags = struct {
    has_qk_norm: bool,
    has_moe: bool,
    has_mla: bool,
    has_fused_qkv: bool,
    has_fused_gate_up: bool,
    use_gelu: bool,
    use_swiglu_oss: bool,
    num_norms: u8,
    norm_weight_offset: f32,
    explicit_qk_norm_ops: bool,
    embedding_multiplier: f32,
};

fn analyzeOps(block_ops: []const Op, pre_block_ops: []const Op) AnalysisFlags {
    var flags = AnalysisFlags{
        .has_qk_norm = false,
        .has_moe = false,
        .has_mla = false,
        .has_fused_qkv = false,
        .has_fused_gate_up = false,
        .use_gelu = false,
        .use_swiglu_oss = false,
        .num_norms = 0,
        .norm_weight_offset = 0.0,
        .explicit_qk_norm_ops = false,
        .embedding_multiplier = 1.0,
    };

    for (block_ops) |op| {
        switch (op.op_type) {
            .norm => {
                if (isQKNormName(op.name)) {
                    flags.has_qk_norm = true;
                } else {
                    flags.num_norms += 1;
                }
                if (op.weight_offset != 0.0 and flags.norm_weight_offset == 0.0) {
                    flags.norm_weight_offset = op.weight_offset;
                }
            },
            .multihead_attention => {
                if (op.qk_norm) flags.has_qk_norm = true;
                if (op.fused_qkv) flags.has_fused_qkv = true;
                if (op.mla) flags.has_mla = true;
            },
            .mlp => {
                if (op.activation) |act| {
                    if (std.mem.eql(u8, act, "gelu")) flags.use_gelu = true;
                    if (std.mem.eql(u8, act, "swiglu_oss")) flags.use_swiglu_oss = true;
                }
                if (op.fused_gate_up) flags.has_fused_gate_up = true;
            },
            .moe => {
                flags.has_moe = true;
                if (op.activation) |act| {
                    if (std.mem.eql(u8, act, "swiglu_oss")) flags.use_swiglu_oss = true;
                }
            },
            else => {},
        }

        // Check for explicit QK norm ops in inputs
        if (!flags.explicit_qk_norm_ops) {
            for (op.inputs) |inp| {
                switch (inp) {
                    .tensor => |t| {
                        if (std.mem.endsWith(u8, t, "q_norm.weight") or std.mem.endsWith(u8, t, "k_norm.weight")) {
                            flags.explicit_qk_norm_ops = true;
                            break;
                        }
                    },
                    else => {},
                }
            }
        }
    }

    // Analyze pre_block ops for embedding_multiplier
    for (pre_block_ops) |op| {
        if (op.op_type == .mul) {
            for (op.inputs) |inp| {
                switch (inp) {
                    .scalar => |s| {
                        flags.embedding_multiplier = s;
                    },
                    .tensor => {},
                }
            }
        }
    }

    return flags;
}

fn isQKNormName(name: ?[]const u8) bool {
    if (name) |n| {
        return std.mem.endsWith(u8, n, "q_norm") or std.mem.endsWith(u8, n, "k_norm");
    }
    return false;
}

// =============================================================================
// Unit Tests
// =============================================================================

test "parseFromJson parses minimal valid architecture" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "name": "test_arch",
        \\  "model_types": ["TestModel"],
        \\  "block": [
        \\    {"op": "norm"},
        \\    {"op": "add"}
        \\  ]
        \\}
    ;

    const arch = try parseFromJson(allocator, json);
    defer {
        allocator.free(arch.name);
        for (arch.model_types) |mt| allocator.free(mt);
        allocator.free(arch.model_types);
        allocator.free(arch.block_ops);
    }

    try std.testing.expectEqualStrings("test_arch", arch.name);
    try std.testing.expectEqual(@as(usize, 1), arch.model_types.len);
    try std.testing.expectEqualStrings("TestModel", arch.model_types[0]);
    try std.testing.expectEqual(@as(usize, 2), arch.block_ops.len);
    try std.testing.expectEqual(OpType.norm, arch.block_ops[0].op_type);
    try std.testing.expectEqual(OpType.add, arch.block_ops[1].op_type);
}

test "parseFromJson parses all op types" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "name": "op_types_test",
        \\  "model_types": ["Test"],
        \\  "block": [
        \\    {"op": "norm"},
        \\    {"op": "multihead_attention"},
        \\    {"op": "attention"},
        \\    {"op": "mlp"},
        \\    {"op": "moe"},
        \\    {"op": "add"},
        \\    {"op": "mul"},
        \\    {"op": "matmul"},
        \\    {"op": "silu"},
        \\    {"op": "gelu"},
        \\    {"op": "embedding"},
        \\    {"op": "linear"},
        \\    {"op": "rope"},
        \\    {"op": "softmax"}
        \\  ]
        \\}
    ;

    const arch = try parseFromJson(allocator, json);
    defer {
        allocator.free(arch.name);
        for (arch.model_types) |mt| allocator.free(mt);
        allocator.free(arch.model_types);
        allocator.free(arch.block_ops);
    }

    try std.testing.expectEqual(@as(usize, 14), arch.block_ops.len);
    try std.testing.expectEqual(OpType.norm, arch.block_ops[0].op_type);
    try std.testing.expectEqual(OpType.multihead_attention, arch.block_ops[1].op_type);
    try std.testing.expectEqual(OpType.multihead_attention, arch.block_ops[2].op_type); // "attention" alias
    try std.testing.expectEqual(OpType.mlp, arch.block_ops[3].op_type);
    try std.testing.expectEqual(OpType.moe, arch.block_ops[4].op_type);
}

test "parseFromJson parses op with inputs" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "name": "inputs_test",
        \\  "model_types": ["Test"],
        \\  "block": [
        \\    {
        \\      "op": "add",
        \\      "inputs": [
        \\        {"tensor": "hidden_states"},
        \\        {"scalar": 1.5}
        \\      ]
        \\    }
        \\  ]
        \\}
    ;

    const arch = try parseFromJson(allocator, json);
    defer {
        allocator.free(arch.name);
        for (arch.model_types) |mt| allocator.free(mt);
        allocator.free(arch.model_types);
        for (arch.block_ops) |op| {
            for (op.inputs) |inp| {
                switch (inp) {
                    .tensor => |t| allocator.free(t),
                    .scalar => {},
                }
            }
            allocator.free(op.inputs);
        }
        allocator.free(arch.block_ops);
    }

    try std.testing.expectEqual(@as(usize, 1), arch.block_ops.len);
    const inputs = arch.block_ops[0].inputs;
    try std.testing.expectEqual(@as(usize, 2), inputs.len);
    try std.testing.expectEqualStrings("hidden_states", inputs[0].tensor);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), inputs[1].scalar, 0.001);
}

test "parseFromJson parses pre_block and post_block" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "name": "blocks_test",
        \\  "model_types": ["Test"],
        \\  "pre_block": [{"op": "embedding"}],
        \\  "block": [{"op": "norm"}],
        \\  "post_block": [{"op": "linear"}]
        \\}
    ;

    const arch = try parseFromJson(allocator, json);
    defer {
        allocator.free(arch.name);
        for (arch.model_types) |mt| allocator.free(mt);
        allocator.free(arch.model_types);
        allocator.free(arch.block_ops);
        allocator.free(arch.pre_block_ops);
        allocator.free(arch.post_block_ops);
    }

    try std.testing.expectEqual(@as(usize, 1), arch.pre_block_ops.len);
    try std.testing.expectEqual(@as(usize, 1), arch.block_ops.len);
    try std.testing.expectEqual(@as(usize, 1), arch.post_block_ops.len);
    try std.testing.expectEqual(OpType.embedding, arch.pre_block_ops[0].op_type);
    try std.testing.expectEqual(OpType.norm, arch.block_ops[0].op_type);
    try std.testing.expectEqual(OpType.linear, arch.post_block_ops[0].op_type);
}

test "parseFromJson parses weight specs" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "name": "weights_test",
        \\  "model_types": ["Test"],
        \\  "block": [{"op": "norm"}],
        \\  "block_weights": [
        \\    {
        \\      "id": "self_attn.q_proj.weight",
        \\      "candidates": ["model.layers.{d}.self_attn.q_proj.weight"],
        \\      "module_type": "Linear",
        \\      "layout": "linear",
        \\      "dtype": "float16",
        \\      "required": true,
        \\      "expected_shape": [4, 4],
        \\      "transforms": ["maybe_transpose", "dtype_f32"]
        \\    }
        \\  ],
        \\  "global_weights": [
        \\    {
        \\      "id": "token_embeddings",
        \\      "candidates": ["model.embed_tokens.weight"],
        \\      "module_type": "Embedding",
        \\      "layout": "embedding",
        \\      "dtype": "float16",
        \\      "required": true,
        \\      "expected_shape": [8, 4]
        \\    }
        \\  ]
        \\}
    ;

    const arch = try parseFromJson(allocator, json);
    defer {
        allocator.free(arch.name);
        for (arch.model_types) |mt| allocator.free(mt);
        allocator.free(arch.model_types);
        allocator.free(arch.block_ops);
        if (arch.block_weights.len > 0) freeWeightSpecs(allocator, arch.block_weights);
        if (arch.global_weights.len > 0) freeWeightSpecs(allocator, arch.global_weights);
    }

    try std.testing.expectEqual(@as(usize, 1), arch.block_weights.len);
    try std.testing.expectEqualStrings("self_attn.q_proj.weight", arch.block_weights[0].id);
    try std.testing.expectEqual(types.WeightLayout.linear, arch.block_weights[0].layout);
    try std.testing.expectEqual(@as(usize, 2), arch.block_weights[0].transforms.len);
    try std.testing.expectEqual(types.WeightTransform.maybe_transpose, arch.block_weights[0].transforms[0]);
    try std.testing.expectEqual(@as(usize, 1), arch.global_weights.len);
    try std.testing.expectEqualStrings("token_embeddings", arch.global_weights[0].id);
    try std.testing.expectEqual(types.WeightLayout.embedding, arch.global_weights[0].layout);
}

test "parseFromJson orders block variants by variant_names" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "name": "hetero_order_test",
        \\  "model_types": ["Test"],
        \\  "block_variants": {
        \\    "attention": { "ops": [ { "op": "norm" } ] },
        \\    "mamba": { "ops": [ { "op": "mamba_mixer" } ] }
        \\  },
        \\  "variant_names": ["mamba", "attention"],
        \\  "layer_map": [0, 1]
        \\}
    ;

    const arch = try parseFromJson(allocator, json);
    defer {
        allocator.free(arch.name);
        for (arch.model_types) |mt| allocator.free(mt);
        allocator.free(arch.model_types);
        if (arch.block_ops.len > 0) allocator.free(arch.block_ops);
        if (arch.pre_block_ops.len > 0) allocator.free(arch.pre_block_ops);
        if (arch.post_block_ops.len > 0) allocator.free(arch.post_block_ops);
        if (arch.layer_map) |map| allocator.free(map);
        if (arch.block_variants) |variants| {
            for (variants) |variant| {
                allocator.free(variant.name);
                allocator.free(variant.ops);
                if (variant.weights.len > 0) freeWeightSpecs(allocator, variant.weights);
            }
            allocator.free(variants);
        }
        if (arch.block_weights.len > 0) freeWeightSpecs(allocator, arch.block_weights);
        if (arch.global_weights.len > 0) freeWeightSpecs(allocator, arch.global_weights);
    }

    const first_variant = arch.getVariant(0) orelse return error.MissingBlockVariant;
    try std.testing.expectEqualStrings("mamba", first_variant.name);
}

test "parseFromJson detects qk_norm flag" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "name": "qk_norm_test",
        \\  "model_types": ["Test"],
        \\  "block": [
        \\    {"op": "multihead_attention", "qk_norm": true}
        \\  ]
        \\}
    ;

    const arch = try parseFromJson(allocator, json);
    defer {
        allocator.free(arch.name);
        for (arch.model_types) |mt| allocator.free(mt);
        allocator.free(arch.model_types);
        allocator.free(arch.block_ops);
    }

    try std.testing.expect(arch.has_qk_norm);
}

test "parseFromJson detects moe flag" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "name": "moe_test",
        \\  "model_types": ["Test"],
        \\  "block": [
        \\    {"op": "moe", "num_experts": 8, "experts_per_token": 2}
        \\  ]
        \\}
    ;

    const arch = try parseFromJson(allocator, json);
    defer {
        allocator.free(arch.name);
        for (arch.model_types) |mt| allocator.free(mt);
        allocator.free(arch.model_types);
        allocator.free(arch.block_ops);
    }

    try std.testing.expect(arch.has_moe);
    try std.testing.expectEqual(@as(i32, 8), arch.block_ops[0].num_experts);
    try std.testing.expectEqual(@as(i32, 2), arch.block_ops[0].experts_per_token);
}

test "parseFromJson detects gelu activation" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "name": "gelu_test",
        \\  "model_types": ["Test"],
        \\  "block": [
        \\    {"op": "mlp", "activation": "gelu"}
        \\  ]
        \\}
    ;

    const arch = try parseFromJson(allocator, json);
    defer {
        allocator.free(arch.name);
        for (arch.model_types) |mt| allocator.free(mt);
        allocator.free(arch.model_types);
        for (arch.block_ops) |op| {
            if (op.activation) |a| allocator.free(a);
        }
        allocator.free(arch.block_ops);
    }

    try std.testing.expect(arch.use_gelu);
}

test "parseFromJson detects swiglu_oss activation" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "name": "swiglu_oss_test",
        \\  "model_types": ["Test"],
        \\  "block": [
        \\    {"op": "mlp", "activation": "swiglu_oss"}
        \\  ]
        \\}
    ;

    const arch = try parseFromJson(allocator, json);
    defer {
        allocator.free(arch.name);
        for (arch.model_types) |mt| allocator.free(mt);
        allocator.free(arch.model_types);
        for (arch.block_ops) |op| {
            if (op.activation) |a| allocator.free(a);
        }
        allocator.free(arch.block_ops);
    }

    try std.testing.expect(arch.use_swiglu_oss);
    try std.testing.expect(!arch.use_gelu); // Ensure gelu is not set
}

test "parseFromJson returns error for missing name" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "model_types": ["Test"],
        \\  "block": []
        \\}
    ;

    const result = parseFromJson(allocator, json);
    try std.testing.expectError(error.MissingName, result);
}

test "parseFromJson returns error for missing model_types" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "name": "test",
        \\  "block": []
        \\}
    ;

    const result = parseFromJson(allocator, json);
    try std.testing.expectError(error.MissingModelTypes, result);
}

test "parseFromJson returns error for missing block" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "name": "test",
        \\  "model_types": ["Test"]
        \\}
    ;

    const result = parseFromJson(allocator, json);
    try std.testing.expectError(error.MissingBlock, result);
}

test "parseFromJson returns error for unknown op type" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "name": "test",
        \\  "model_types": ["Test"],
        \\  "block": [{"op": "unknown_op"}]
        \\}
    ;

    const result = parseFromJson(allocator, json);
    try std.testing.expectError(error.UnknownOpType, result);
}

test "parseFromJson returns error for invalid json" {
    const allocator = std.testing.allocator;
    const json = "not valid json";

    const result = parseFromJson(allocator, json);
    try std.testing.expectError(error.InvalidJson, result);
}
