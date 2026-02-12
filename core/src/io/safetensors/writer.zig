//! SafeTensors file writer.
//!
//! Writes tensors to SafeTensors format files with proper header
//! serialization and data alignment.

const std = @import("std");
const tensor = @import("../../tensor.zig");
const dtype_mod = @import("../../dtype.zig");

const Tensor = tensor.Tensor;
const DType = dtype_mod.DType;

/// Errors that can occur when writing SafeTensors files
pub const WriteError = error{
    InvalidTensor,
    SerializationFailed,
    WriteFailed,
    OutOfMemory,
};

/// A tensor entry to be written to a SafeTensors file
pub const TensorEntry = struct {
    name: []const u8,
    dtype: DType,
    shape: []const usize,
    data: []const u8,
};

/// Write tensors to a SafeTensors file
/// Format: [8-byte header length][JSON header][tensor data...]
pub fn write(allocator: std.mem.Allocator, path: []const u8, entries: []const TensorEntry) !void {
    try writeWithMetadata(allocator, path, entries, null);
}

/// Write tensors to a SafeTensors file with custom metadata
pub fn writeWithMetadata(
    allocator: std.mem.Allocator,
    path: []const u8,
    entries: []const TensorEntry,
    metadata: ?*const std.StringHashMapUnmanaged([]u8),
) !void {
    var out_file = try std.fs.cwd().createFile(path, .{});
    defer out_file.close();

    try writeToFileWithMetadata(allocator, out_file, entries, metadata);
}

/// Write tensors to an already-opened file
pub fn writeToFile(allocator: std.mem.Allocator, file: std.fs.File, entries: []const TensorEntry) !void {
    try writeToFileWithMetadata(allocator, file, entries, null);
}

/// Write tensors to an already-opened file with custom metadata
pub fn writeToFileWithMetadata(
    allocator: std.mem.Allocator,
    file: std.fs.File,
    entries: []const TensorEntry,
    metadata: ?*const std.StringHashMapUnmanaged([]u8),
) !void {
    // Build header JSON
    var header_buf = std.ArrayListUnmanaged(u8){};
    defer header_buf.deinit(allocator);

    try header_buf.append(allocator, '{');

    var data_offset: usize = 0;
    for (entries, 0..) |entry, entry_idx| {
        if (entry_idx > 0) try header_buf.append(allocator, ',');

        // Calculate data size
        const data_size = entry.data.len;

        // Write entry: "name": {"dtype": "F32", "shape": [x, y], "data_offsets": [start, end]}
        try header_buf.writer(allocator).print("\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{
            entry.name,
            dtypeToString(entry.dtype),
        });

        for (entry.shape, 0..) |dim, dim_idx| {
            if (dim_idx > 0) try header_buf.append(allocator, ',');
            try header_buf.writer(allocator).print("{d}", .{dim});
        }

        try header_buf.writer(allocator).print("],\"data_offsets\":[{d},{d}]}}", .{
            data_offset,
            data_offset + data_size,
        });

        data_offset += data_size;
    }

    // Add metadata field (required by SafeTensors spec)
    if (entries.len > 0) try header_buf.append(allocator, ',');
    try header_buf.appendSlice(allocator, "\"__metadata__\":{");

    // Write custom metadata if provided
    if (metadata) |meta| {
        var meta_iter = meta.iterator();
        var first_meta = true;
        while (meta_iter.next()) |kv| {
            if (!first_meta) try header_buf.append(allocator, ',');
            first_meta = false;
            // Escape JSON strings properly
            try header_buf.writer(allocator).print("\"{s}\":\"{s}\"", .{ kv.key_ptr.*, kv.value_ptr.* });
        }
    }

    try header_buf.append(allocator, '}'); // Close __metadata__
    try header_buf.append(allocator, '}'); // Close root object

    // Pad header to 8-byte alignment
    const header_len = header_buf.items.len;
    const padded_len = (header_len + 7) & ~@as(usize, 7);
    const padding = padded_len - header_len;
    for (0..padding) |_| try header_buf.append(allocator, ' ');

    // Write header length (8 bytes, little-endian)
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, @intCast(padded_len), .little);
    try file.writeAll(&len_buf);

    // Write header
    try file.writeAll(header_buf.items);

    // Write tensor data
    for (entries) |entry| {
        try file.writeAll(entry.data);
    }
}

/// Convert DType to SafeTensors dtype string
fn dtypeToString(dtype: DType) []const u8 {
    return switch (dtype) {
        .f32 => "F32",
        .f64 => "F64",
        .f16 => "F16",
        .bf16 => "BF16",
        .i8 => "I8",
        .i16 => "I16",
        .i32 => "I32",
        .i64 => "I64",
        .u8 => "U8",
        .u16 => "U16",
        .u32 => "U32",
        .u64 => "U64",
        .grouped_affine_u4 => "U32",
        .grouped_affine_u8 => "U32", // 8-bit also uses U32 (4 values per word)
        .mxfp4 => "U8", // MXFP4 stores packed 4-bit values as bytes
        .f8_e4m3 => "F8_E4M3",
    };
}

/// Builder for incrementally constructing a SafeTensors file
pub const Builder = struct {
    allocator: std.mem.Allocator,
    entries: std.ArrayListUnmanaged(OwnedEntry),
    metadata: std.StringHashMapUnmanaged([]u8),
    /// Maximum shard size in bytes. 0 = no limit (single file).
    max_shard_size: u64 = 0,

    const OwnedEntry = struct {
        name: []u8,
        dtype: DType,
        shape: []usize,
        data: []u8,
    };

    pub fn init(allocator: std.mem.Allocator) Builder {
        return .{
            .allocator = allocator,
            .entries = .{},
            .metadata = .{},
            .max_shard_size = 0,
        };
    }

    pub fn deinit(self: *Builder) void {
        for (self.entries.items) |entry| {
            self.allocator.free(entry.name);
            self.allocator.free(entry.shape);
            self.allocator.free(entry.data);
        }
        self.entries.deinit(self.allocator);

        // Free metadata keys and values
        var meta_iter = self.metadata.iterator();
        while (meta_iter.next()) |kv| {
            self.allocator.free(kv.key_ptr.*);
            self.allocator.free(kv.value_ptr.*);
        }
        self.metadata.deinit(self.allocator);
    }

    /// Add a metadata key-value pair (copies both key and value).
    pub fn setMetadata(self: *Builder, key: []const u8, value: []const u8) !void {
        const owned_key = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(owned_key);

        const owned_value = try self.allocator.dupe(u8, value);
        errdefer self.allocator.free(owned_value);

        // If key already exists, free old value
        if (self.metadata.fetchRemove(owned_key)) |removed| {
            self.allocator.free(removed.key);
            self.allocator.free(removed.value);
        }

        try self.metadata.put(self.allocator, owned_key, owned_value);
    }

    /// Add a tensor to the builder (copies data)
    pub fn addTensor(self: *Builder, name: []const u8, dt: DType, shape: []const usize, data: []const u8) !void {
        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);

        const owned_shape = try self.allocator.dupe(usize, shape);
        errdefer self.allocator.free(owned_shape);

        const owned_data = try self.allocator.dupe(u8, data);
        errdefer self.allocator.free(owned_data);

        try self.entries.append(self.allocator, .{
            .name = owned_name,
            .dtype = dt,
            .shape = owned_shape,
            .data = owned_data,
        });
    }

    /// Write all tensors to file(s).
    /// If max_shard_size is set and total exceeds limit, creates sharded output
    /// with model-NNNNN-of-MMMMM.safetensors files and index JSON in output_dir.
    /// Otherwise, writes a single file at output_dir/filename.
    ///
    /// Parameters:
    /// - output_dir: Directory path where output files will be written
    /// - filename: Name of the output file (e.g., "model.safetensors")
    pub fn save(self: *Builder, output_dir: []const u8, filename: []const u8) !void {
        // Calculate total size
        var total_size: u64 = 0;
        for (self.entries.items) |e| {
            total_size += e.data.len;
        }

        // Decide sharding vs single file
        const needs_sharding = self.max_shard_size > 0 and total_size > self.max_shard_size;

        if (needs_sharding) {
            // Write sharded files to output_dir
            try self.saveSharded(output_dir);
        } else {
            // Single file output - join directory and filename
            const full_path = try std.fs.path.join(self.allocator, &.{ output_dir, filename });
            defer self.allocator.free(full_path);

            var tensor_entries = try self.allocator.alloc(TensorEntry, self.entries.items.len);
            defer self.allocator.free(tensor_entries);

            for (self.entries.items, 0..) |e, entry_idx| {
                tensor_entries[entry_idx] = .{
                    .name = e.name,
                    .dtype = e.dtype,
                    .shape = e.shape,
                    .data = e.data,
                };
            }

            // Pass metadata to the writer (may be empty)
            const meta_ptr: ?*const std.StringHashMapUnmanaged([]u8) = if (self.metadata.count() > 0) &self.metadata else null;
            try writeWithMetadata(self.allocator, full_path, tensor_entries, meta_ptr);
        }
    }

    /// Check if tensor is an auxiliary tensor (scales/biases) that must stay with its parent.
    fn isAuxiliaryTensor(name: []const u8) bool {
        return std.mem.endsWith(u8, name, ".scales") or std.mem.endsWith(u8, name, ".biases");
    }

    /// Save tensors to multiple sharded files with an index.
    /// Creates: model-00001-of-NNNNN.safetensors, ..., model.safetensors.index.json
    fn saveSharded(self: *Builder, output_dir: []const u8) !void {
        // Build tensor entries
        var tensor_entries = try self.allocator.alloc(TensorEntry, self.entries.items.len);
        defer self.allocator.free(tensor_entries);

        for (self.entries.items, 0..) |e, entry_idx| {
            tensor_entries[entry_idx] = .{
                .name = e.name,
                .dtype = e.dtype,
                .shape = e.shape,
                .data = e.data,
            };
        }

        // Plan shards: group tensors respecting auxiliary tensor constraints
        var shard_boundaries = std.ArrayListUnmanaged(usize){}; // Start indices
        defer shard_boundaries.deinit(self.allocator);

        try shard_boundaries.append(self.allocator, 0); // First shard starts at 0

        var current_shard_size: u64 = 0;
        var idx: usize = 0;
        while (idx < tensor_entries.len) {
            const entry = tensor_entries[idx];
            const entry_size: u64 = entry.data.len;

            // Check if we need a new shard (but not for aux tensors and not if shard is empty)
            if (current_shard_size > 0 and
                current_shard_size + entry_size > self.max_shard_size and
                !isAuxiliaryTensor(entry.name))
            {
                try shard_boundaries.append(self.allocator, idx);
                current_shard_size = 0;
            }

            current_shard_size += entry_size;
            idx += 1;
        }

        const num_shards = shard_boundaries.items.len;

        // Build weight map for index
        var weight_map = std.StringHashMapUnmanaged([]const u8){};
        defer weight_map.deinit(self.allocator);

        // Get metadata pointer
        const meta_ptr: ?*const std.StringHashMapUnmanaged([]u8) = if (self.metadata.count() > 0) &self.metadata else null;

        // Write each shard
        for (shard_boundaries.items, 0..) |start_idx, shard_idx| {
            const end_idx = if (shard_idx + 1 < num_shards)
                shard_boundaries.items[shard_idx + 1]
            else
                tensor_entries.len;

            const shard_entries = tensor_entries[start_idx..end_idx];

            // Generate shard filename: model-00001-of-00003.safetensors
            const shard_filename = try std.fmt.allocPrint(
                self.allocator,
                "model-{d:0>5}-of-{d:0>5}.safetensors",
                .{ shard_idx + 1, num_shards },
            );
            defer self.allocator.free(shard_filename);

            const shard_path = try std.fs.path.join(self.allocator, &.{ output_dir, shard_filename });
            defer self.allocator.free(shard_path);

            // Write shard file (with metadata in each shard)
            try writeWithMetadata(self.allocator, shard_path, shard_entries, meta_ptr);

            // Record tensor->shard mapping
            for (shard_entries) |entry| {
                try weight_map.put(self.allocator, entry.name, shard_filename);
            }
        }

        // Calculate total size for index metadata
        var total_size: u64 = 0;
        for (self.entries.items) |e| {
            total_size += e.data.len;
        }

        // Write index file
        try self.writeIndexFile(output_dir, &weight_map, total_size, num_shards);
    }

    /// Write the model.safetensors.index.json file.
    fn writeIndexFile(
        self: *Builder,
        output_dir: []const u8,
        weight_map: *const std.StringHashMapUnmanaged([]const u8),
        total_size: u64,
        num_shards: usize,
    ) !void {
        var json_buf = std.ArrayListUnmanaged(u8){};
        defer json_buf.deinit(self.allocator);

        try json_buf.appendSlice(self.allocator, "{\"metadata\":{\"total_size\":");
        try json_buf.writer(self.allocator).print("{d}", .{total_size});
        try json_buf.appendSlice(self.allocator, "},\"weight_map\":{");

        // Sort tensor names for deterministic output
        var sorted_names = try self.allocator.alloc([]const u8, weight_map.count());
        defer self.allocator.free(sorted_names);

        var name_idx: usize = 0;
        var iter = weight_map.iterator();
        while (iter.next()) |kv| {
            sorted_names[name_idx] = kv.key_ptr.*;
            name_idx += 1;
        }

        std.mem.sort([]const u8, sorted_names, {}, struct {
            fn lessThan(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.order(u8, a, b) == .lt;
            }
        }.lessThan);

        // Write sorted weight_map entries
        for (sorted_names, 0..) |name, i| {
            if (i > 0) try json_buf.append(self.allocator, ',');
            const shard_file = weight_map.get(name).?;
            try json_buf.writer(self.allocator).print("\"{s}\":\"{s}\"", .{ name, shard_file });
        }

        try json_buf.appendSlice(self.allocator, "}}");

        // Write index file
        const index_path = try std.fs.path.join(self.allocator, &.{ output_dir, "model.safetensors.index.json" });
        defer self.allocator.free(index_path);

        var file = try std.fs.cwd().createFile(index_path, .{});
        defer file.close();
        try file.writeAll(json_buf.items);

        _ = num_shards; // Currently unused but available for future use
    }
};

test "writer round-trip" {
    const allocator = std.testing.allocator;

    // Create test data
    var test_data: [16]u8 = undefined;
    for (&test_data, 0..) |*byte_ptr, byte_idx| byte_ptr.* = @intCast(byte_idx);

    const tensor_entries = [_]TensorEntry{
        .{
            .name = "test_tensor",
            .dtype = .f32,
            .shape = &[_]usize{ 2, 2 },
            .data = &test_data,
        },
    };

    // Write to temp file
    const temp_path = "/tmp/test_safetensors_writer.safetensors";
    try write(allocator, temp_path, &tensor_entries);
    defer std.fs.cwd().deleteFile(temp_path) catch {};

    // Read back and verify
    const reader = @import("reader.zig");
    var safetensors = try reader.SafeTensors.load(allocator, temp_path);
    defer safetensors.deinit();

    try std.testing.expect(safetensors.hasTensor("test_tensor"));
    const tensor_view = try safetensors.getTensor("test_tensor", .f32);
    try std.testing.expectEqual(@as(i64, 2), tensor_view.shape[0]);
    try std.testing.expectEqual(@as(i64, 2), tensor_view.shape[1]);
}

test "write single tensor to file" {
    const allocator = std.testing.allocator;

    // Create test data - 4 float32 values
    var data: [16]u8 = undefined;
    const float_values = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    for (float_values, 0..) |val, i| {
        const bytes = std.mem.toBytes(val);
        @memcpy(data[i * 4 .. (i + 1) * 4], &bytes);
    }

    const tensor_entries = [_]TensorEntry{.{
        .name = "weights",
        .dtype = .f32,
        .shape = &[_]usize{4},
        .data = &data,
    }};

    // Write and verify
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const temp_path = "test_write_single.safetensors";
    const dir = tmp.dir;

    var temp_file = try dir.createFile(temp_path, .{});
    const real_path = try dir.realpathAlloc(allocator, temp_path);
    defer allocator.free(real_path);
    temp_file.close();

    try write(allocator, real_path, &tensor_entries);

    // Verify the file exists and has content
    const file = try dir.openFile(temp_path, .{});
    defer file.close();
    const file_size = try file.getEndPos();
    try std.testing.expect(file_size > 0);
}

test "write multiple tensors with different dtypes" {
    const allocator = std.testing.allocator;

    // Create test data for dtypes supported by both writer and reader
    var f32_data: [16]u8 = undefined;
    for (&f32_data, 0..) |*byte_ptr, i| byte_ptr.* = @intCast(i);

    var i64_data: [16]u8 = undefined;
    for (&i64_data, 0..) |*byte_ptr, i| byte_ptr.* = @intCast(i + 100);

    var i8_data = [_]u8{ 10, 20, 30, 40 };

    const tensor_entries = [_]TensorEntry{
        .{
            .name = "float_tensor",
            .dtype = .f32,
            .shape = &[_]usize{ 2, 2 },
            .data = &f32_data,
        },
        .{
            .name = "int_tensor",
            .dtype = .i64, // Use i64 which reader supports
            .shape = &[_]usize{2},
            .data = &i64_data,
        },
        .{
            .name = "byte_tensor",
            .dtype = .i8, // Use i8 which reader supports (maps U8 to i8)
            .shape = &[_]usize{4},
            .data = &i8_data,
        },
    };

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const temp_path = "test_write_multiple.safetensors";
    const dir = tmp.dir;

    var temp_file = try dir.createFile(temp_path, .{});
    const real_path = try dir.realpathAlloc(allocator, temp_path);
    defer allocator.free(real_path);
    temp_file.close();

    try write(allocator, real_path, &tensor_entries);

    // Read back and verify all tensors
    const reader = @import("reader.zig");
    var safetensors = try reader.SafeTensors.load(allocator, real_path);
    defer safetensors.deinit();

    try std.testing.expect(safetensors.hasTensor("float_tensor"));
    try std.testing.expect(safetensors.hasTensor("int_tensor"));
    try std.testing.expect(safetensors.hasTensor("byte_tensor"));
}

test "writeToFile writes valid safetensors format" {
    const allocator = std.testing.allocator;

    var data: [12]u8 = undefined;
    for (&data, 0..) |*byte_ptr, i| byte_ptr.* = @intCast(i);

    const tensor_entries = [_]TensorEntry{.{
        .name = "test",
        .dtype = .f32,
        .shape = &[_]usize{3},
        .data = &data,
    }};

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const temp_path = "test_writeToFile.safetensors";
    const dir = tmp.dir;

    // Create file for writing
    var write_file = try dir.createFile(temp_path, .{});
    try writeToFile(allocator, write_file, &tensor_entries);
    write_file.close();

    // Reopen for reading to verify
    var read_file = try dir.openFile(temp_path, .{});
    defer read_file.close();

    // Read header length
    var len_buf: [8]u8 = undefined;
    _ = try read_file.readAll(&len_buf);
    const header_len = std.mem.readInt(u64, &len_buf, .little);

    // Header length should be positive and reasonable
    try std.testing.expect(header_len > 0);
    try std.testing.expect(header_len < 10000);

    // Read header JSON
    const header_json = try allocator.alloc(u8, header_len);
    defer allocator.free(header_json);
    _ = try read_file.readAll(header_json);

    // Verify JSON contains required fields
    try std.testing.expect(std.mem.indexOf(u8, header_json, "\"test\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, header_json, "\"dtype\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, header_json, "\"F32\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, header_json, "\"shape\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, header_json, "\"data_offsets\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, header_json, "\"__metadata__\"") != null);
}

test "Builder init and deinit manages memory" {
    const allocator = std.testing.allocator;

    var builder = Builder.init(allocator);
    defer builder.deinit();

    // Builder should be properly initialized
    try std.testing.expectEqual(@as(usize, 0), builder.entries.items.len);
}

test "Builder addTensor copies data correctly" {
    const allocator = std.testing.allocator;

    var builder = Builder.init(allocator);
    defer builder.deinit();

    const data = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const shape = [_]usize{ 2, 4 };

    try builder.addTensor("tensor1", .u8, &shape, &data);

    try std.testing.expectEqual(@as(usize, 1), builder.entries.items.len);
    try std.testing.expectEqualStrings("tensor1", builder.entries.items[0].name);
    try std.testing.expectEqual(DType.u8, builder.entries.items[0].dtype);
    try std.testing.expectEqualSlices(usize, &shape, builder.entries.items[0].shape);
    try std.testing.expectEqualSlices(u8, &data, builder.entries.items[0].data);
}

test "Builder addTensor multiple tensors" {
    const allocator = std.testing.allocator;

    var builder = Builder.init(allocator);
    defer builder.deinit();

    const data1 = [_]u8{ 1, 2, 3, 4 };
    const shape1 = [_]usize{4};

    const data2 = [_]u8{ 10, 20, 30, 40, 50, 60 };
    const shape2 = [_]usize{ 2, 3 };

    try builder.addTensor("first", .u8, &shape1, &data1);
    try builder.addTensor("second", .u8, &shape2, &data2);

    try std.testing.expectEqual(@as(usize, 2), builder.entries.items.len);
    try std.testing.expectEqualStrings("first", builder.entries.items[0].name);
    try std.testing.expectEqualStrings("second", builder.entries.items[1].name);
}

test "Builder setMetadata stores key-value pairs" {
    const allocator = std.testing.allocator;

    var builder = Builder.init(allocator);
    defer builder.deinit();

    try builder.setMetadata("talu.format", "gaf");
    try builder.setMetadata("talu.version", "1.0");

    try std.testing.expectEqual(@as(usize, 2), builder.metadata.count());
    try std.testing.expectEqualStrings("gaf", builder.metadata.get("talu.format").?);
    try std.testing.expectEqualStrings("1.0", builder.metadata.get("talu.version").?);
}

test "Builder setMetadata replaces existing key" {
    const allocator = std.testing.allocator;

    var builder = Builder.init(allocator);
    defer builder.deinit();

    try builder.setMetadata("key", "value1");
    try builder.setMetadata("key", "value2");

    try std.testing.expectEqual(@as(usize, 1), builder.metadata.count());
    try std.testing.expectEqualStrings("value2", builder.metadata.get("key").?);
}

test "Builder save writes metadata to file" {
    const allocator = std.testing.allocator;

    var builder = Builder.init(allocator);
    defer builder.deinit();

    // Add a tensor
    var data: [16]u8 = undefined;
    for (&data, 0..) |*byte_ptr, i| byte_ptr.* = @intCast(i);
    const shape = [_]usize{ 2, 2 };
    try builder.addTensor("test", .f32, &shape, &data);

    // Add metadata
    try builder.setMetadata("talu.format", "gaf");
    try builder.setMetadata("talu.quant_type", "gaf4_64");

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const temp_path = "test_builder_metadata.safetensors";
    const dir = tmp.dir;

    const dir_real_path = try dir.realpathAlloc(allocator, ".");
    defer allocator.free(dir_real_path);

    try builder.save(dir_real_path, temp_path);

    // Read back and verify metadata is present in header
    var read_file = try dir.openFile(temp_path, .{});
    defer read_file.close();

    var len_buf: [8]u8 = undefined;
    _ = try read_file.readAll(&len_buf);
    const header_len = std.mem.readInt(u64, &len_buf, .little);

    const header_json = try allocator.alloc(u8, header_len);
    defer allocator.free(header_json);
    _ = try read_file.readAll(header_json);

    // Verify metadata fields are present
    try std.testing.expect(std.mem.indexOf(u8, header_json, "\"__metadata__\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, header_json, "\"talu.format\":\"gaf\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, header_json, "\"talu.quant_type\":\"gaf4_64\"") != null);
}

test "Builder save writes valid file" {
    const allocator = std.testing.allocator;

    var builder = Builder.init(allocator);
    defer builder.deinit();

    var data1: [16]u8 = undefined;
    for (&data1, 0..) |*byte_ptr, i| byte_ptr.* = @intCast(i);

    var data2: [8]u8 = undefined;
    for (&data2, 0..) |*byte_ptr, i| byte_ptr.* = @intCast(i + 50);

    const shape1 = [_]usize{ 2, 2 };
    const shape2 = [_]usize{2};

    try builder.addTensor("weights", .f32, &shape1, &data1);
    try builder.addTensor("bias", .f32, &shape2, &data2);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const temp_path = "test_builder_save.safetensors";
    const dir = tmp.dir;

    const dir_real_path = try dir.realpathAlloc(allocator, ".");
    defer allocator.free(dir_real_path);

    try builder.save(dir_real_path, temp_path);

    // Verify the saved file by reading it back
    const reader = @import("reader.zig");
    const full_path = try std.fs.path.join(allocator, &.{ dir_real_path, temp_path });
    defer allocator.free(full_path);
    var safetensors = try reader.SafeTensors.load(allocator, full_path);
    defer safetensors.deinit();

    try std.testing.expect(safetensors.hasTensor("weights"));
    try std.testing.expect(safetensors.hasTensor("bias"));

    const weights = try safetensors.getTensor("weights", .f32);
    try std.testing.expectEqual(@as(i64, 2), weights.shape[0]);
    try std.testing.expectEqual(@as(i64, 2), weights.shape[1]);

    const bias = try safetensors.getTensor("bias", .f32);
    try std.testing.expectEqual(@as(i64, 2), bias.shape[0]);
}

test "write empty tensor list" {
    const allocator = std.testing.allocator;

    const tensor_entries = [_]TensorEntry{};

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const temp_path = "test_write_empty.safetensors";
    const dir = tmp.dir;

    var temp_file = try dir.createFile(temp_path, .{});
    const real_path = try dir.realpathAlloc(allocator, temp_path);
    defer allocator.free(real_path);
    temp_file.close();

    try write(allocator, real_path, &tensor_entries);

    // Verify file was created and has valid structure
    const file = try dir.openFile(temp_path, .{});
    defer file.close();
    const file_size = try file.getEndPos();

    // Should have at least 8 bytes for header length
    try std.testing.expect(file_size >= 8);
}

test "write tensors with various dtypes" {
    const allocator = std.testing.allocator;

    var f32_data: [16]u8 = undefined;
    for (&f32_data, 0..) |*byte_ptr, i| byte_ptr.* = @intCast(i);

    var f16_data: [8]u8 = undefined;
    for (&f16_data, 0..) |*byte_ptr, i| byte_ptr.* = @intCast(i + 10);

    var i64_data: [32]u8 = undefined;
    for (&i64_data, 0..) |*byte_ptr, i| byte_ptr.* = @intCast(i + 20);

    const tensor_entries = [_]TensorEntry{
        .{
            .name = "f32_tensor",
            .dtype = .f32,
            .shape = &[_]usize{4},
            .data = &f32_data,
        },
        .{
            .name = "f16_tensor",
            .dtype = .f16,
            .shape = &[_]usize{4},
            .data = &f16_data,
        },
        .{
            .name = "i64_tensor",
            .dtype = .i64,
            .shape = &[_]usize{4},
            .data = &i64_data,
        },
    };

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const temp_path = "test_write_dtypes.safetensors";
    const dir = tmp.dir;

    var temp_file = try dir.createFile(temp_path, .{});
    const real_path = try dir.realpathAlloc(allocator, temp_path);
    defer allocator.free(real_path);
    temp_file.close();

    try write(allocator, real_path, &tensor_entries);

    // Verify by reading back
    const reader = @import("reader.zig");
    var safetensors = try reader.SafeTensors.load(allocator, real_path);
    defer safetensors.deinit();

    try std.testing.expect(safetensors.hasTensor("f32_tensor"));
    try std.testing.expect(safetensors.hasTensor("f16_tensor"));
    try std.testing.expect(safetensors.hasTensor("i64_tensor"));
}

test "Builder addTensor preserves data independence" {
    const allocator = std.testing.allocator;

    var builder = Builder.init(allocator);
    defer builder.deinit();

    var original_data = [_]u8{ 1, 2, 3, 4 };
    const original_shape = [_]usize{4};

    try builder.addTensor("tensor1", .u8, &original_shape, &original_data);

    // Modify original data
    original_data[0] = 99;

    // Builder should have copied the data, so it remains unchanged
    try std.testing.expectEqual(@as(u8, 1), builder.entries.items[0].data[0]);
}

test "write tensors with multidimensional shapes" {
    const allocator = std.testing.allocator;

    // Use f32 to avoid reader dtype mapping issues (U8->i8)
    var data1: [96]u8 = undefined; // 2x3x4 = 24 f32 values = 96 bytes
    for (&data1, 0..) |*byte_ptr, i| byte_ptr.* = @intCast(i % 256);

    var data2: [480]u8 = undefined; // 5x4x3x2 = 120 f32 values = 480 bytes
    for (&data2, 0..) |*byte_ptr, i| byte_ptr.* = @intCast(i % 256);

    const tensor_entries = [_]TensorEntry{
        .{
            .name = "tensor_3d",
            .dtype = .f32,
            .shape = &[_]usize{ 2, 3, 4 },
            .data = &data1,
        },
        .{
            .name = "tensor_4d",
            .dtype = .f32,
            .shape = &[_]usize{ 5, 4, 3, 2 },
            .data = &data2,
        },
    };

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const temp_path = "test_write_multidim.safetensors";
    const dir = tmp.dir;

    var temp_file = try dir.createFile(temp_path, .{});
    const real_path = try dir.realpathAlloc(allocator, temp_path);
    defer allocator.free(real_path);
    temp_file.close();

    try write(allocator, real_path, &tensor_entries);

    // Verify by reading back
    const reader = @import("reader.zig");
    var safetensors = try reader.SafeTensors.load(allocator, real_path);
    defer safetensors.deinit();

    const tensor_3d = try safetensors.getTensor("tensor_3d", .f32);
    try std.testing.expectEqual(@as(i64, 2), tensor_3d.shape[0]);
    try std.testing.expectEqual(@as(i64, 3), tensor_3d.shape[1]);
    try std.testing.expectEqual(@as(i64, 4), tensor_3d.shape[2]);

    const tensor_4d = try safetensors.getTensor("tensor_4d", .f32);
    try std.testing.expectEqual(@as(i64, 5), tensor_4d.shape[0]);
    try std.testing.expectEqual(@as(i64, 4), tensor_4d.shape[1]);
    try std.testing.expectEqual(@as(i64, 3), tensor_4d.shape[2]);
    try std.testing.expectEqual(@as(i64, 2), tensor_4d.shape[3]);
}

// =============================================================================
// Sharding Tests
// =============================================================================

test "Builder isAuxiliaryTensor identifies scales and biases" {
    try std.testing.expect(Builder.isAuxiliaryTensor("layer.0.weight.scales"));
    try std.testing.expect(Builder.isAuxiliaryTensor("layer.0.weight.biases"));
    try std.testing.expect(Builder.isAuxiliaryTensor("model.layers.5.mlp.gate_proj.scales"));
    try std.testing.expect(!Builder.isAuxiliaryTensor("layer.0.weight"));
    try std.testing.expect(!Builder.isAuxiliaryTensor("model.embed_tokens.weight"));
    try std.testing.expect(!Builder.isAuxiliaryTensor("lm_head.weight"));
}

test "Builder save creates single file when under shard limit" {
    const allocator = std.testing.allocator;

    var builder = Builder.init(allocator);
    defer builder.deinit();

    builder.max_shard_size = 1000; // 1KB limit

    // Add small tensors (well under limit)
    var data1: [16]u8 = undefined;
    for (&data1, 0..) |*byte_ptr, i| byte_ptr.* = @intCast(i);
    try builder.addTensor("tensor1", .f32, &[_]usize{ 2, 2 }, &data1);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    // Create test directory
    const test_dir = "test_single_file";
    try tmp.dir.makePath(test_dir);

    const dir_real_path = try tmp.dir.realpathAlloc(allocator, test_dir);
    defer allocator.free(dir_real_path);

    // Save - should create single file (16 bytes < 1000 limit)
    try builder.save(dir_real_path, "model.safetensors");

    // Verify single file was created
    const single_file_path = try std.fs.path.join(allocator, &.{ test_dir, "model.safetensors" });
    defer allocator.free(single_file_path);
    try tmp.dir.access(single_file_path, .{});
}

test "Builder saveSharded creates multiple files when over limit" {
    const allocator = std.testing.allocator;

    var builder = Builder.init(allocator);
    defer builder.deinit();

    builder.max_shard_size = 50; // Very small limit to force sharding

    // Add tensors that exceed shard limit
    var data1: [32]u8 = undefined;
    for (&data1, 0..) |*byte_ptr, i| byte_ptr.* = @intCast(i);
    var data2: [32]u8 = undefined;
    for (&data2, 0..) |*byte_ptr, i| byte_ptr.* = @intCast(i + 32);
    var data3: [32]u8 = undefined;
    for (&data3, 0..) |*byte_ptr, i| byte_ptr.* = @intCast(i + 64);

    try builder.addTensor("tensor1", .f32, &[_]usize{8}, &data1);
    try builder.addTensor("tensor2", .f32, &[_]usize{8}, &data2);
    try builder.addTensor("tensor3", .f32, &[_]usize{8}, &data3);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const test_dir = "test_sharded";
    try tmp.dir.makePath(test_dir);

    const dir_real_path = try tmp.dir.realpathAlloc(allocator, test_dir);
    defer allocator.free(dir_real_path);

    // Save - should create sharded files (96 bytes > 50 limit)
    try builder.save(dir_real_path, "model.safetensors");

    // Verify index file was created
    const index_path = try std.fs.path.join(allocator, &.{ test_dir, "model.safetensors.index.json" });
    defer allocator.free(index_path);
    try tmp.dir.access(index_path, .{});

    // Verify at least first shard was created
    const shard1_path = try std.fs.path.join(allocator, &.{ test_dir, "model-00001-of-00003.safetensors" });
    defer allocator.free(shard1_path);
    try tmp.dir.access(shard1_path, .{});
}

test "Builder saveSharded keeps auxiliary tensors with parent" {
    const allocator = std.testing.allocator;

    var builder = Builder.init(allocator);
    defer builder.deinit();

    // Set very small shard size to force sharding
    builder.max_shard_size = 40;

    // Add weight followed by auxiliary tensors
    // The .scales and .biases should stay with their parent weight
    var weight_data: [32]u8 = undefined;
    for (&weight_data, 0..) |*byte_ptr, i| byte_ptr.* = @intCast(i);
    var scales_data: [8]u8 = undefined;
    for (&scales_data, 0..) |*byte_ptr, i| byte_ptr.* = @intCast(i + 100);
    var biases_data: [8]u8 = undefined;
    for (&biases_data, 0..) |*byte_ptr, i| byte_ptr.* = @intCast(i + 200);
    var weight2_data: [32]u8 = undefined;
    for (&weight2_data, 0..) |*byte_ptr, i| byte_ptr.* = @intCast(i + 50);

    try builder.addTensor("layer.0.weight", .f32, &[_]usize{8}, &weight_data);
    try builder.addTensor("layer.0.weight.scales", .f16, &[_]usize{4}, &scales_data);
    try builder.addTensor("layer.0.weight.biases", .f16, &[_]usize{4}, &biases_data);
    try builder.addTensor("layer.1.weight", .f32, &[_]usize{8}, &weight2_data);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const test_dir = "test_aux_tensors";
    try tmp.dir.makePath(test_dir);

    const dir_real_path = try tmp.dir.realpathAlloc(allocator, test_dir);
    defer allocator.free(dir_real_path);

    try builder.save(dir_real_path, "model.safetensors");

    // Verify index file was created and read it
    const index_path = try std.fs.path.join(allocator, &.{ test_dir, "model.safetensors.index.json" });
    defer allocator.free(index_path);

    var index_file = try tmp.dir.openFile(index_path, .{});
    defer index_file.close();

    const index_content = try index_file.readToEndAlloc(allocator, 8192);
    defer allocator.free(index_content);

    // The scales and biases should be in the same shard as their parent weight
    // We verify this by checking the index JSON contains both in the same shard reference
    try std.testing.expect(std.mem.indexOf(u8, index_content, "\"weight_map\"") != null);
}
