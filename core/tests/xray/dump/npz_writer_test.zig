//! Integration tests for dump.NpzWriter.
//!
//! Tests the NpzWriter type exported from core/src/xray/dump/root.zig.
//! NpzWriter writes captured tensors to NPZ format (numpy archive).

const std = @import("std");
const main = @import("main");
const dump = main.dump;
const NpzWriter = dump.NpzWriter;
const CapturedTensor = dump.capture.CapturedTensor;

// ============================================================================
// Lifecycle Tests
// ============================================================================

test "NpzWriter init and deinit" {
    var writer = NpzWriter.init(std.testing.allocator);
    defer writer.deinit();

    try std.testing.expectEqual(@as(usize, 0), writer.entries.items.len);
}

// ============================================================================
// addTensor Tests
// ============================================================================

test "NpzWriter addTensor adds entry" {
    var writer = NpzWriter.init(std.testing.allocator);
    defer writer.deinit();

    // Create a captured tensor
    const name = try std.testing.allocator.dupe(u8, "test_array");
    const data = try std.testing.allocator.alloc(f32, 6);
    @memcpy(data, &[_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });

    var tensor = CapturedTensor{
        .name = name,
        .data = data,
        .shape = .{ 2, 3, 0, 0 },
        .ndim = 2,
    };
    defer tensor.deinit(std.testing.allocator);

    try writer.addTensor(&tensor);

    try std.testing.expectEqual(@as(usize, 1), writer.entries.items.len);
    try std.testing.expectEqualStrings("test_array.npy", writer.entries.items[0].name);
}

test "NpzWriter addTensor multiple tensors" {
    var writer = NpzWriter.init(std.testing.allocator);
    defer writer.deinit();

    // Add first tensor
    const name1 = try std.testing.allocator.dupe(u8, "array1");
    const data1 = try std.testing.allocator.alloc(f32, 4);
    @memset(data1, 1.0);
    var tensor1 = CapturedTensor{
        .name = name1,
        .data = data1,
        .shape = .{ 4, 0, 0, 0 },
        .ndim = 1,
    };
    defer tensor1.deinit(std.testing.allocator);
    try writer.addTensor(&tensor1);

    // Add second tensor
    const name2 = try std.testing.allocator.dupe(u8, "array2");
    const data2 = try std.testing.allocator.alloc(f32, 9);
    @memset(data2, 2.0);
    var tensor2 = CapturedTensor{
        .name = name2,
        .data = data2,
        .shape = .{ 3, 3, 0, 0 },
        .ndim = 2,
    };
    defer tensor2.deinit(std.testing.allocator);
    try writer.addTensor(&tensor2);

    try std.testing.expectEqual(@as(usize, 2), writer.entries.items.len);
    try std.testing.expectEqualStrings("array1.npy", writer.entries.items[0].name);
    try std.testing.expectEqualStrings("array2.npy", writer.entries.items[1].name);
}

// ============================================================================
// NPY Format Tests
// ============================================================================

test "NpzWriter generates valid NPY header" {
    var writer = NpzWriter.init(std.testing.allocator);
    defer writer.deinit();

    const name = try std.testing.allocator.dupe(u8, "test");
    const data = try std.testing.allocator.alloc(f32, 4);
    @memcpy(data, &[_]f32{ 1.0, 2.0, 3.0, 4.0 });
    var tensor = CapturedTensor{
        .name = name,
        .data = data,
        .shape = .{ 2, 2, 0, 0 },
        .ndim = 2,
    };
    defer tensor.deinit(std.testing.allocator);

    try writer.addTensor(&tensor);

    const npy_data = writer.entries.items[0].data;

    // Check NPY magic number
    try std.testing.expectEqual(@as(u8, 0x93), npy_data[0]);
    try std.testing.expectEqualStrings("NUMPY", npy_data[1..6]);

    // Check version (1.0)
    try std.testing.expectEqual(@as(u8, 1), npy_data[6]);
    try std.testing.expectEqual(@as(u8, 0), npy_data[7]);

    // Header length at bytes 8-9 (little endian u16)
    const header_len = @as(u16, npy_data[8]) | (@as(u16, npy_data[9]) << 8);
    try std.testing.expect(header_len > 0);

    // Check that header contains expected descriptor
    const header_start = 10;
    const header_end = header_start + header_len;
    const header = npy_data[header_start..header_end];

    try std.testing.expect(std.mem.indexOf(u8, header, "'descr': '<f4'") != null);
    try std.testing.expect(std.mem.indexOf(u8, header, "'fortran_order': False") != null);
    try std.testing.expect(std.mem.indexOf(u8, header, "'shape': (2, 2, )") != null);
}

// ============================================================================
// Write Tests
// ============================================================================

test "NpzWriter write creates valid ZIP file" {
    var writer = NpzWriter.init(std.testing.allocator);
    defer writer.deinit();

    // Add a tensor
    const name = try std.testing.allocator.dupe(u8, "data");
    const data = try std.testing.allocator.alloc(f32, 3);
    @memcpy(data, &[_]f32{ 1.0, 2.0, 3.0 });
    var tensor = CapturedTensor{
        .name = name,
        .data = data,
        .shape = .{ 3, 0, 0, 0 },
        .ndim = 1,
    };
    defer tensor.deinit(std.testing.allocator);
    try writer.addTensor(&tensor);

    // Write to a temp file
    const tmp_path = "/tmp/test_npz_writer.npz";
    try writer.write(tmp_path);
    defer std.fs.cwd().deleteFile(tmp_path) catch {};

    // Read the file and verify ZIP structure
    const file = try std.fs.cwd().openFile(tmp_path, .{});
    defer file.close();

    var header: [4]u8 = undefined;
    _ = try file.read(&header);

    // ZIP local file header signature
    try std.testing.expectEqualSlices(u8, &[_]u8{ 0x50, 0x4B, 0x03, 0x04 }, &header);
}

test "NpzWriter write empty archive" {
    var writer = NpzWriter.init(std.testing.allocator);
    defer writer.deinit();

    const tmp_path = "/tmp/test_npz_empty.npz";
    try writer.write(tmp_path);
    defer std.fs.cwd().deleteFile(tmp_path) catch {};

    // Verify file was created
    const stat = try std.fs.cwd().statFile(tmp_path);
    // Empty ZIP has at least end-of-central-directory record (22 bytes)
    try std.testing.expect(stat.size >= 22);
}

// ============================================================================
// addAll Tests
// ============================================================================

test "NpzWriter addAll from Capture" {
    var cap = dump.Capture.init(std.testing.allocator);
    defer cap.deinit();

    // Manually add tensors to capture
    const name1 = try std.testing.allocator.dupe(u8, "tensor1");
    const data1 = try std.testing.allocator.alloc(f32, 2);
    @memcpy(data1, &[_]f32{ 1.0, 2.0 });
    try cap.tensors.append(std.testing.allocator, .{
        .name = name1,
        .data = data1,
        .shape = .{ 2, 0, 0, 0 },
        .ndim = 1,
    });

    const name2 = try std.testing.allocator.dupe(u8, "tensor2");
    const data2 = try std.testing.allocator.alloc(f32, 4);
    @memcpy(data2, &[_]f32{ 3.0, 4.0, 5.0, 6.0 });
    try cap.tensors.append(std.testing.allocator, .{
        .name = name2,
        .data = data2,
        .shape = .{ 2, 2, 0, 0 },
        .ndim = 2,
    });

    var writer = NpzWriter.init(std.testing.allocator);
    defer writer.deinit();

    try writer.addAll(&cap);

    try std.testing.expectEqual(@as(usize, 2), writer.entries.items.len);
    try std.testing.expectEqualStrings("tensor1.npy", writer.entries.items[0].name);
    try std.testing.expectEqualStrings("tensor2.npy", writer.entries.items[1].name);
}
