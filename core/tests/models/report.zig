//! Deterministic models metadata reporting for hardening baselines.
//!
//! Usage:
//!   zig run core/tests/models/report.zig -- registry
//!   zig run core/tests/models/report.zig -- metadata

const std = @import("std");
const root_main = @import("main");

const registry = root_main.models.dispatcher.registry;
const op_types = root_main.models.dispatcher.op_types;

const Architecture = op_types.Architecture;
const Op = op_types.Op;
const OpType = op_types.OpType;

pub const Completeness = struct {
    has_mlp_or_moe: bool,
    d_ff_source_configured: bool,
    d_ff_source_ids_exist: bool,
    has_shortconv_feature: bool,
    shortconv_source_configured: bool,
    shortconv_source_exists: bool,
    moe_params_declared: bool,

    pub fn ok(self: Completeness) bool {
        const d_ff_ok = !self.has_mlp_or_moe or (self.d_ff_source_configured and self.d_ff_source_ids_exist);
        const shortconv_ok = !self.has_shortconv_feature or (self.shortconv_source_configured and self.shortconv_source_exists);
        return d_ff_ok and shortconv_ok and self.moe_params_declared;
    }
};

fn gatherSortedEntryIndices(allocator: std.mem.Allocator) ![]usize {
    const count = registry.entries.len;
    const indices = try allocator.alloc(usize, count);
    for (indices, 0..) |*entry_index, i| entry_index.* = i;
    const Ctx = struct {
        fn less(_: void, lhs: usize, rhs: usize) bool {
            return std.mem.lessThan(u8, registry.entries[lhs].id, registry.entries[rhs].id);
        }
    };
    std.sort.block(usize, indices, {}, Ctx.less);
    return indices;
}

fn appendWeightId(set: *std.StringHashMap(void), allocator: std.mem.Allocator, id: []const u8) !void {
    if (set.contains(id)) return;
    const owned = try allocator.dupe(u8, id);
    errdefer allocator.free(owned);
    try set.put(owned, {});
}

fn freeOwnedStringSet(allocator: std.mem.Allocator, set: *std.StringHashMap(void)) void {
    var it = set.iterator();
    while (it.next()) |entry| allocator.free(entry.key_ptr.*);
    set.deinit();
}

fn collectWeightIds(allocator: std.mem.Allocator, arch: *const Architecture) !std.StringHashMap(void) {
    var ids = std.StringHashMap(void).init(allocator);
    errdefer freeOwnedStringSet(allocator, &ids);

    for (arch.block_weights) |spec| try appendWeightId(&ids, allocator, spec.id);
    for (arch.global_weights) |spec| try appendWeightId(&ids, allocator, spec.id);
    if (arch.block_variants) |variants| {
        for (variants) |variant| {
            for (variant.weights) |spec| try appendWeightId(&ids, allocator, spec.id);
        }
    }
    return ids;
}

fn hasOpTypeIn(ops: []const Op, tag: OpType) bool {
    for (ops) |op| {
        if (op.op_type == tag) return true;
    }
    return false;
}

fn hasOpType(arch: *const Architecture, tag: OpType) bool {
    if (hasOpTypeIn(arch.block_ops, tag)) return true;
    if (arch.block_variants) |variants| {
        for (variants) |variant| {
            if (hasOpTypeIn(variant.ops, tag)) return true;
        }
    }
    return false;
}

fn moeOpsDeclareParamsIn(ops: []const Op) bool {
    for (ops) |op| {
        if (op.op_type != .moe) continue;
        if (op.num_experts <= 0 or op.experts_per_token <= 0) return false;
    }
    return true;
}

fn moeOpsDeclareParams(arch: *const Architecture) bool {
    if (!moeOpsDeclareParamsIn(arch.block_ops)) return false;
    if (arch.block_variants) |variants| {
        for (variants) |variant| {
            if (!moeOpsDeclareParamsIn(variant.ops)) return false;
        }
    }
    return true;
}

pub fn evaluateArchitecture(allocator: std.mem.Allocator, arch: *const Architecture) !Completeness {
    const has_mlp_or_moe = hasOpType(arch, .mlp) or hasOpType(arch, .moe);
    const has_shortconv_feature = arch.has_shortconv or hasOpType(arch, .shortconv);

    var ids = try collectWeightIds(allocator, arch);
    defer freeOwnedStringSet(allocator, &ids);

    var d_ff_source_ids_exist = true;
    for (arch.d_ff_source_weight_ids) |id| {
        if (!ids.contains(id)) {
            d_ff_source_ids_exist = false;
            break;
        }
    }

    var shortconv_source_exists = true;
    if (arch.shortconv_dims_source_weight_id) |id| {
        shortconv_source_exists = ids.contains(id);
    }

    return .{
        .has_mlp_or_moe = has_mlp_or_moe,
        .d_ff_source_configured = arch.d_ff_source_weight_ids.len > 0,
        .d_ff_source_ids_exist = d_ff_source_ids_exist,
        .has_shortconv_feature = has_shortconv_feature,
        .shortconv_source_configured = arch.shortconv_dims_source_weight_id != null,
        .shortconv_source_exists = shortconv_source_exists,
        .moe_params_declared = moeOpsDeclareParams(arch),
    };
}

pub fn writeRegistryReport(writer: anytype, allocator: std.mem.Allocator) !void {
    const sorted = try gatherSortedEntryIndices(allocator);
    defer allocator.free(sorted);

    try writer.writeAll("id\tfamily\tversion\tmodel_types\n");
    for (sorted) |idx| {
        const entry = registry.entries[idx];
        try writer.print("{s}\t{s}\t{s}\t", .{ entry.id, entry.family, entry.version });
        for (entry.model_types, 0..) |model_type, mt_idx| {
            if (mt_idx != 0) try writer.writeByte(',');
            try writer.writeAll(model_type);
        }
        try writer.writeByte('\n');
    }
}

pub fn writeMetadataReport(writer: anytype, allocator: std.mem.Allocator) !void {
    const sorted = try gatherSortedEntryIndices(allocator);
    defer allocator.free(sorted);

    try writer.writeAll("id\tok\thas_mlp_or_moe\td_ff_source_configured\td_ff_source_ids_exist\thas_shortconv_feature\tshortconv_source_configured\tshortconv_source_exists\tmoe_params_declared\n");
    for (sorted) |idx| {
        const entry = registry.entries[idx];
        const arch = registry.runtimeArchitectureById(entry.id) orelse return error.MissingArchitecture;
        const report = try evaluateArchitecture(allocator, arch);
        try writer.print(
            "{s}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\n",
            .{
                entry.id,
                @intFromBool(report.ok()),
                @intFromBool(report.has_mlp_or_moe),
                @intFromBool(report.d_ff_source_configured),
                @intFromBool(report.d_ff_source_ids_exist),
                @intFromBool(report.has_shortconv_feature),
                @intFromBool(report.shortconv_source_configured),
                @intFromBool(report.shortconv_source_exists),
                @intFromBool(report.moe_params_declared),
            },
        );
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len != 2) {
        std.debug.print("usage: report.zig <registry|metadata>\n", .{});
        return error.InvalidArguments;
    }

    var output = std.ArrayListUnmanaged(u8){};
    defer output.deinit(allocator);

    if (std.mem.eql(u8, args[1], "registry")) {
        try writeRegistryReport(output.writer(allocator), allocator);
        std.debug.print("{s}", .{output.items});
        return;
    }
    if (std.mem.eql(u8, args[1], "metadata")) {
        try writeMetadataReport(output.writer(allocator), allocator);
        std.debug.print("{s}", .{output.items});
        return;
    }

    std.debug.print("unknown mode: {s}\n", .{args[1]});
    return error.InvalidArguments;
}

test "writeRegistryReport is deterministic" {
    var buf_a = std.ArrayListUnmanaged(u8){};
    defer buf_a.deinit(std.testing.allocator);
    var buf_b = std.ArrayListUnmanaged(u8){};
    defer buf_b.deinit(std.testing.allocator);

    try writeRegistryReport(buf_a.writer(std.testing.allocator), std.testing.allocator);
    try writeRegistryReport(buf_b.writer(std.testing.allocator), std.testing.allocator);

    try std.testing.expectEqualStrings(buf_a.items, buf_b.items);
}

test "writeMetadataReport is deterministic" {
    var buf_a = std.ArrayListUnmanaged(u8){};
    defer buf_a.deinit(std.testing.allocator);
    var buf_b = std.ArrayListUnmanaged(u8){};
    defer buf_b.deinit(std.testing.allocator);

    try writeMetadataReport(buf_a.writer(std.testing.allocator), std.testing.allocator);
    try writeMetadataReport(buf_b.writer(std.testing.allocator), std.testing.allocator);

    try std.testing.expectEqualStrings(buf_a.items, buf_b.items);
}

test "registry report matches checked-in baseline snapshot" {
    var buf = std.ArrayListUnmanaged(u8){};
    defer buf.deinit(std.testing.allocator);
    try writeRegistryReport(buf.writer(std.testing.allocator), std.testing.allocator);
    try std.testing.expectEqualStrings(@embedFile("baseline_registry.tsv"), buf.items);
}

test "metadata report matches checked-in baseline snapshot" {
    var buf = std.ArrayListUnmanaged(u8){};
    defer buf.deinit(std.testing.allocator);
    try writeMetadataReport(buf.writer(std.testing.allocator), std.testing.allocator);
    try std.testing.expectEqualStrings(@embedFile("baseline_metadata.tsv"), buf.items);
}
