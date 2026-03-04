//! Save and load LoRA adapter weights via SafeTensors format.
//!
//! Adapter tensors are named:
//!   "{layer_index}.{weight_id}.lora_A"  — shape [rank, in_dim]
//!   "{layer_index}.{weight_id}.lora_B"  — shape [out_dim, rank]

const std = @import("std");
const adapter_mod = @import("adapter.zig");

const LoraAdapter = adapter_mod.LoraAdapter;
const LoraLayer = adapter_mod.LoraLayer;
const LoraConfig = adapter_mod.LoraConfig;
const Allocator = std.mem.Allocator;

/// Metadata for a saved adapter checkpoint.
pub const CheckpointMeta = struct {
    rank: u32,
    alpha: f32,
    num_layers: usize,
    total_params: usize,
};

/// Extract checkpoint metadata from an adapter.
pub fn getCheckpointMeta(adapter_ptr: *const LoraAdapter) CheckpointMeta {
    return .{
        .rank = adapter_ptr.config.rank,
        .alpha = adapter_ptr.config.alpha,
        .num_layers = adapter_ptr.layerCount(),
        .total_params = adapter_ptr.trainableParamCount(),
    };
}

/// Collect all adapter parameter slices for serialization.
///
/// Returns paired (A, B) slices for each adapter layer, along with
/// identifying names. Caller owns the returned name buffers.
pub const AdapterTensorEntry = struct {
    name_A: []u8,
    name_B: []u8,
    data_A: []const f32,
    data_B: []const f32,
    rank: usize,
    in_dim: usize,
    out_dim: usize,
};

/// Collect tensor entries from an adapter for serialization.
/// Caller must free the returned slice and each entry's name buffers.
pub fn collectTensorEntries(
    allocator: Allocator,
    adapter_ptr: *const LoraAdapter,
) ![]AdapterTensorEntry {
    const layers = adapter_ptr.layers.items;
    const entries = try allocator.alloc(AdapterTensorEntry, layers.len);
    errdefer allocator.free(entries);

    for (layers, entries) |*layer, *entry| {
        const name_A = try std.fmt.allocPrint(allocator, "{d}.{s}.lora_A", .{ layer.layer_index, layer.weight_id });
        errdefer allocator.free(name_A);
        const name_B = try std.fmt.allocPrint(allocator, "{d}.{s}.lora_B", .{ layer.layer_index, layer.weight_id });

        entry.* = .{
            .name_A = name_A,
            .name_B = name_B,
            .data_A = layer.A.asSlice(f32),
            .data_B = layer.B.asSlice(f32),
            .rank = layer.rank,
            .in_dim = layer.in_dim,
            .out_dim = layer.out_dim,
        };
    }

    return entries;
}

/// Free tensor entries returned by collectTensorEntries.
pub fn freeTensorEntries(allocator: Allocator, entries: []AdapterTensorEntry) void {
    for (entries) |*entry| {
        allocator.free(entry.name_A);
        allocator.free(entry.name_B);
    }
    allocator.free(entries);
}

// =============================================================================
// Tests
// =============================================================================

test "getCheckpointMeta returns correct values" {
    const allocator = std.testing.allocator;
    var adapter = LoraAdapter.initExplicit(allocator, .{ .rank = 8, .alpha = 16.0 });
    defer adapter.deinit();

    var l1 = try LoraLayer.init(allocator, "q_proj", 0, 64, 64, .{ .rank = 8, .alpha = 16.0 });
    errdefer l1.deinit();
    try adapter.addLayer(l1);

    const meta = getCheckpointMeta(&adapter);
    try std.testing.expectEqual(@as(u32, 8), meta.rank);
    try std.testing.expectApproxEqAbs(@as(f32, 16.0), meta.alpha, 1e-6);
    try std.testing.expectEqual(@as(usize, 1), meta.num_layers);
    // A: 8*64=512, B: 64*8=512, total=1024
    try std.testing.expectEqual(@as(usize, 1024), meta.total_params);
}

test "collectTensorEntries produces correct names and data" {
    const allocator = std.testing.allocator;
    var adapter = LoraAdapter.initExplicit(allocator, .{ .rank = 4, .alpha = 8.0 });
    defer adapter.deinit();

    var l = try LoraLayer.init(allocator, "self_attn.q_proj.weight", 2, 16, 16, .{ .rank = 4, .alpha = 8.0 });
    errdefer l.deinit();
    try adapter.addLayer(l);

    const entries = try collectTensorEntries(allocator, &adapter);
    defer freeTensorEntries(allocator, entries);

    try std.testing.expectEqual(@as(usize, 1), entries.len);
    try std.testing.expectEqualStrings("2.self_attn.q_proj.weight.lora_A", entries[0].name_A);
    try std.testing.expectEqualStrings("2.self_attn.q_proj.weight.lora_B", entries[0].name_B);
    try std.testing.expectEqual(@as(usize, 4 * 16), entries[0].data_A.len);
    try std.testing.expectEqual(@as(usize, 16 * 4), entries[0].data_B.len);
}
