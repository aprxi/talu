//! Integration checks for mlx_bridge config parsing edge cases.

const std = @import("std");
const builtin = @import("builtin");

const has_metal = builtin.os.tag == .macos;

extern fn mlx_validate_config(model_path: [*:0]const u8) c_int;
extern fn mlx_last_error() [*:0]const u8;

test "mlx_validate_config accepts block_ff_dim fallback for lfm2 config" {
    if (comptime !has_metal) return;

    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const config_json =
        \\{
        \\  "model_type": "lfm2",
        \\  "hidden_size": 1024,
        \\  "num_hidden_layers": 16,
        \\  "num_attention_heads": 16,
        \\  "num_key_value_heads": 8,
        \\  "block_ff_dim": 6656
        \\}
    ;
    try tmp.dir.writeFile(.{ .sub_path = "config.json", .data = config_json });

    const model_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(model_path);
    const model_path_z = try allocator.dupeZ(u8, model_path);
    defer allocator.free(model_path_z);

    const status = mlx_validate_config(model_path_z.ptr);
    if (status != 1) {
        std.debug.print("mlx_validate_config failed: {s}\n", .{std.mem.span(mlx_last_error())});
    }
    try std.testing.expectEqual(@as(c_int, 1), status);
}

test "mlx_validate_config rejects config missing all d_ff aliases" {
    if (comptime !has_metal) return;

    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const config_json =
        \\{
        \\  "model_type": "lfm2",
        \\  "hidden_size": 1024,
        \\  "num_hidden_layers": 16,
        \\  "num_attention_heads": 16,
        \\  "num_key_value_heads": 8
        \\}
    ;
    try tmp.dir.writeFile(.{ .sub_path = "config.json", .data = config_json });

    const model_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(model_path);
    const model_path_z = try allocator.dupeZ(u8, model_path);
    defer allocator.free(model_path_z);

    const status = mlx_validate_config(model_path_z.ptr);
    try std.testing.expectEqual(@as(c_int, 0), status);
    try std.testing.expect(
        std.mem.indexOf(
            u8,
            std.mem.span(mlx_last_error()),
            "intermediate_size/d_ff/block_ff_dim",
        ) != null,
    );
}
