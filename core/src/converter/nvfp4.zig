//! NVFP4 Converter
//!
//! Dedicated NVFP4 conversion surface. Conversion emits 4-bit grouped-affine
//! weights (group_size=32) and rewrites config metadata to canonical NVFP4
//! contract fields.

const std = @import("std");
const grouped_affine = @import("grouped_affine.zig");
const json = @import("../io/json/root.zig");

pub const modelIdFromOutputPath = grouped_affine.modelIdFromOutputPath;

pub const ConvertOptions = struct {
    output_dir: []const u8 = "models",
    destination: ?[]const u8 = null,
    output_suffix: ?[]const u8 = null,
    force: bool = false,
    max_shard_size: u64 = 0,
    progress: grouped_affine.ProgressContext = grouped_affine.ProgressContext.NONE,
    profile: @TypeOf((grouped_affine.ConvertOptions{}).profile) = .fast,
    calib_iters: u32 = 300,
    calib_nsamples: u32 = 128,
    calib_seqlen: u32 = 2048,
    calib_batch_size: u32 = 1,
    calib_nblocks: u32 = 1,
    calib_seed: u64 = 42,
};

pub fn convertToNvfp4(
    allocator: std.mem.Allocator,
    input_path: []const u8,
    options: ConvertOptions,
) ![]const u8 {
    const output_path = try grouped_affine.convertToGroupedAffine(allocator, input_path, .{
        .quant = .{
            .bits = 4,
            .group_size = 32,
        },
        .output_dir = options.output_dir,
        .destination = options.destination,
        .output_suffix = options.output_suffix orelse "NVFP4",
        .force = options.force,
        .max_shard_size = options.max_shard_size,
        .progress = options.progress,
        .profile = options.profile,
        .calib_iters = options.calib_iters,
        .calib_nsamples = options.calib_nsamples,
        .calib_seqlen = options.calib_seqlen,
        .calib_batch_size = options.calib_batch_size,
        .calib_nblocks = options.calib_nblocks,
        .calib_seed = options.calib_seed,
    });
    errdefer allocator.free(output_path);
    try rewriteConfigToCanonical(allocator, output_path);
    return output_path;
}

pub fn rewriteConfigToCanonical(allocator: std.mem.Allocator, output_path: []const u8) !void {
    const config_path = try std.fs.path.join(allocator, &.{ output_path, "config.json" });
    defer allocator.free(config_path);

    const config_bytes = try std.fs.cwd().readFileAlloc(allocator, config_path, 1024 * 1024);
    defer allocator.free(config_bytes);

    var parsed = json.parseValue(allocator, config_bytes, .{
        .max_size_bytes = 1024 * 1024,
        .max_value_bytes = 1024 * 1024,
        .max_string_bytes = 256 * 1024,
    }) catch |err| {
        return switch (err) {
            error.InputTooLarge, error.InputTooDeep, error.StringTooLong, error.InvalidJson => error.InvalidConfig,
            else => err,
        };
    };
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidConfig;

    var output_buf = std.ArrayListUnmanaged(u8){};
    defer output_buf.deinit(allocator);
    try output_buf.append(allocator, '{');

    var first_field = true;
    var iter = parsed.value.object.iterator();
    while (iter.next()) |kv| {
        if (std.mem.eql(u8, kv.key_ptr.*, "quantization")) continue;
        if (std.mem.eql(u8, kv.key_ptr.*, "quantization_config")) continue;

        if (!first_field) try output_buf.append(allocator, ',');
        first_field = false;

        try output_buf.append(allocator, '"');
        try output_buf.appendSlice(allocator, kv.key_ptr.*);
        try output_buf.appendSlice(allocator, "\":");

        const value_json = try std.json.Stringify.valueAlloc(allocator, kv.value_ptr.*, .{});
        defer allocator.free(value_json);
        try output_buf.appendSlice(allocator, value_json);
    }

    if (!first_field) try output_buf.append(allocator, ',');
    try output_buf.appendSlice(allocator, "\"quantization\":{\"group_size\":32,\"bits\":4}");
    try output_buf.appendSlice(
        allocator,
        ",\"quantization_config\":{\"quant_method\":\"nvfp4\",\"quant_type\":\"nvfp4\",\"bits\":4,\"group_size\":32,\"quant_contract_version\":1}",
    );
    try output_buf.append(allocator, '}');

    var out_file = try std.fs.cwd().createFile(config_path, .{});
    defer out_file.close();
    try out_file.writeAll(output_buf.items);
}
