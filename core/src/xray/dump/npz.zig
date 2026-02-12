//! NPZ Writer
//!
//! Writes captured tensors to NPZ format (ZIP file containing .npy arrays).
//! NPZ is numpy's standard archive format for multiple arrays.
//!
//! Format: ZIP archive where each entry is "{name}.npy" with numpy array data.

const std = @import("std");
const capture_mod = @import("capture.zig");
const CapturedTensor = capture_mod.CapturedTensor;

/// NPZ file writer.
pub const NpzWriter = struct {
    allocator: std.mem.Allocator,
    entries: std.ArrayListUnmanaged(ZipEntry),

    const ZipEntry = struct {
        name: []const u8, // e.g., "layer0.attn_out.npy"
        data: []const u8, // .npy file content
    };

    pub fn init(allocator: std.mem.Allocator) NpzWriter {
        return .{
            .allocator = allocator,
            .entries = .{},
        };
    }

    pub fn deinit(self: *NpzWriter) void {
        for (self.entries.items) |entry| {
            self.allocator.free(entry.name);
            self.allocator.free(entry.data);
        }
        self.entries.deinit(self.allocator);
    }

    /// Add a tensor to the archive.
    pub fn addTensor(self: *NpzWriter, tensor: *const CapturedTensor) !void {
        // Build .npy file content
        const npy_data = try self.buildNpyData(tensor);
        errdefer self.allocator.free(npy_data);

        // Build entry name: "name.npy"
        const name = try std.fmt.allocPrint(self.allocator, "{s}.npy", .{tensor.name});
        errdefer self.allocator.free(name);

        try self.entries.append(self.allocator, .{ .name = name, .data = npy_data });
    }

    /// Add all tensors from a Capture.
    pub fn addAll(self: *NpzWriter, cap: *const capture_mod.Capture) !void {
        for (cap.tensors.items) |*tensor| {
            try self.addTensor(tensor);
        }
    }

    /// Write the NPZ file to disk.
    pub fn write(self: *NpzWriter, path: []const u8) !void {
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();

        try self.writeZip(file);
    }

    fn writeZip(self: *NpzWriter, file: std.fs.File) !void {
        var offsets = try self.allocator.alloc(u32, self.entries.items.len);
        defer self.allocator.free(offsets);

        // Write local file headers + data
        var offset: u32 = 0;
        for (self.entries.items, 0..) |entry, i| {
            offsets[i] = offset;
            const header_size = try writeLocalFileHeader(file, entry.name, entry.data);
            offset += header_size;
        }

        // Write central directory
        const cd_start = offset;
        for (self.entries.items, 0..) |entry, i| {
            const cd_size = try writeCentralDirectoryHeader(file, entry.name, entry.data, offsets[i]);
            offset += cd_size;
        }
        const cd_size = offset - cd_start;

        // Write end of central directory
        try writeEndOfCentralDirectory(file, @intCast(self.entries.items.len), cd_size, cd_start);
    }

    fn buildNpyData(self: *NpzWriter, tensor: *const CapturedTensor) ![]const u8 {
        // Build header string: "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 2, 3), }"
        var header_buf: [256]u8 = undefined; // lint:ignore undefined-usage - populated by bufPrint below
        var header_len: usize = 0;

        // Shape tuple string
        var shape_buf: [64]u8 = undefined; // lint:ignore undefined-usage - populated char-by-char below
        var shape_len: usize = 0;
        shape_buf[0] = '(';
        shape_len = 1;
        for (0..tensor.ndim) |i| {
            const dim = tensor.shape[i];
            const written = std.fmt.bufPrint(shape_buf[shape_len..], "{d}, ", .{dim}) catch return error.ShapeTooLong;
            shape_len += written.len;
        }
        shape_buf[shape_len] = ')';
        shape_len += 1;

        // Full header
        const header_str = std.fmt.bufPrint(&header_buf, "{{'descr': '<f4', 'fortran_order': False, 'shape': {s}, }}", .{shape_buf[0..shape_len]}) catch return error.HeaderTooLong;
        header_len = header_str.len;

        // Pad header to 64-byte alignment (including magic + version + header_len)
        const prefix_len = 10; // magic(6) + version(2) + header_len(2)
        const total_header = prefix_len + header_len;
        const padding = (64 - (total_header % 64)) % 64;
        const padded_header_len = header_len + padding;

        // Calculate total size
        const data_bytes = tensor.data.len * @sizeOf(f32);
        const total_size = prefix_len + padded_header_len + data_bytes;

        // Allocate buffer
        const buffer = try self.allocator.alloc(u8, total_size);
        var pos: usize = 0;

        // Magic number
        buffer[pos] = 0x93;
        pos += 1;
        @memcpy(buffer[pos..][0..5], "NUMPY");
        pos += 5;

        // Version 1.0
        buffer[pos] = 1;
        pos += 1;
        buffer[pos] = 0;
        pos += 1;

        // Header length (little-endian u16)
        buffer[pos] = @truncate(padded_header_len & 0xFF);
        pos += 1;
        buffer[pos] = @truncate((padded_header_len >> 8) & 0xFF);
        pos += 1;

        // Header string
        @memcpy(buffer[pos..][0..header_len], header_str);
        pos += header_len;

        // Padding (spaces, ending with newline)
        for (0..padding) |i| {
            buffer[pos + i] = if (i == padding - 1) '\n' else ' ';
        }
        pos += padding;

        // Data (f32 little-endian)
        const data_slice = std.mem.sliceAsBytes(tensor.data);
        @memcpy(buffer[pos..][0..data_bytes], data_slice);

        return buffer;
    }
};

fn writeInt(file: std.fs.File, comptime T: type, value: T, endian: std.builtin.Endian) !void {
    var bytes: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &bytes, value, endian);
    try file.writeAll(&bytes);
}

fn crc32(data: []const u8) u32 {
    return std.hash.Crc32.hash(data);
}

fn writeLocalFileHeader(file: std.fs.File, name: []const u8, data: []const u8) !u32 {
    const name_len: u16 = @intCast(name.len);
    const data_len: u32 = @intCast(data.len);
    const data_crc = crc32(data);

    // Local file header signature
    try file.writeAll(&[_]u8{ 0x50, 0x4B, 0x03, 0x04 });
    // Version needed (2.0)
    try file.writeAll(&[_]u8{ 0x14, 0x00 });
    // General purpose bit flag
    try file.writeAll(&[_]u8{ 0x00, 0x00 });
    // Compression method (0 = stored)
    try file.writeAll(&[_]u8{ 0x00, 0x00 });
    // Last mod time/date
    try file.writeAll(&[_]u8{ 0x00, 0x00, 0x00, 0x00 });
    // CRC-32
    try writeInt(file, u32, data_crc, .little);
    // Compressed size
    try writeInt(file, u32, data_len, .little);
    // Uncompressed size
    try writeInt(file, u32, data_len, .little);
    // File name length
    try writeInt(file, u16, name_len, .little);
    // Extra field length
    try writeInt(file, u16, 0, .little);
    // File name
    try file.writeAll(name);
    // Data
    try file.writeAll(data);

    return 30 + name_len + data_len;
}

fn writeCentralDirectoryHeader(file: std.fs.File, name: []const u8, data: []const u8, local_offset: u32) !u32 {
    const name_len: u16 = @intCast(name.len);
    const data_len: u32 = @intCast(data.len);
    const data_crc = crc32(data);

    // Central directory header signature
    try file.writeAll(&[_]u8{ 0x50, 0x4B, 0x01, 0x02 });
    // Version made by
    try file.writeAll(&[_]u8{ 0x14, 0x00 });
    // Version needed
    try file.writeAll(&[_]u8{ 0x14, 0x00 });
    // General purpose bit flag
    try file.writeAll(&[_]u8{ 0x00, 0x00 });
    // Compression method
    try file.writeAll(&[_]u8{ 0x00, 0x00 });
    // Last mod time/date
    try file.writeAll(&[_]u8{ 0x00, 0x00, 0x00, 0x00 });
    // CRC-32
    try writeInt(file, u32, data_crc, .little);
    // Compressed size
    try writeInt(file, u32, data_len, .little);
    // Uncompressed size
    try writeInt(file, u32, data_len, .little);
    // File name length
    try writeInt(file, u16, name_len, .little);
    // Extra field length
    try writeInt(file, u16, 0, .little);
    // File comment length
    try writeInt(file, u16, 0, .little);
    // Disk number start
    try writeInt(file, u16, 0, .little);
    // Internal file attributes
    try writeInt(file, u16, 0, .little);
    // External file attributes
    try writeInt(file, u32, 0, .little);
    // Relative offset of local header
    try writeInt(file, u32, local_offset, .little);
    // File name
    try file.writeAll(name);

    return 46 + name_len;
}

fn writeEndOfCentralDirectory(file: std.fs.File, entry_count: u16, cd_size: u32, cd_offset: u32) !void {
    // End of central directory signature
    try file.writeAll(&[_]u8{ 0x50, 0x4B, 0x05, 0x06 });
    // Disk number
    try writeInt(file, u16, 0, .little);
    // Disk number with central directory
    try writeInt(file, u16, 0, .little);
    // Number of central directory records on this disk
    try writeInt(file, u16, entry_count, .little);
    // Total number of central directory records
    try writeInt(file, u16, entry_count, .little);
    // Size of central directory
    try writeInt(file, u32, cd_size, .little);
    // Offset of central directory
    try writeInt(file, u32, cd_offset, .little);
    // Comment length
    try writeInt(file, u16, 0, .little);
}
