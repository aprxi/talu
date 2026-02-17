//! SafeTensors file reader.
//!
//! Parses SafeTensors format files, validating headers and providing
//! lazy tensor loading with memory mapping support.

const std = @import("std");
const tensor = @import("../../tensor.zig");
const dtype = @import("../../dtype.zig");
const json = @import("../json/root.zig");
const mmap_policy = @import("../../compute/mmap_policy.zig");

const Tensor = tensor.Tensor;
const DType = dtype.DType;

/// Errors that can occur when loading SafeTensors files
pub const LoadError = error{
    InvalidFile,
    IncompleteRead,
    InvalidHeader,
    NotFound,
    UnexpectedDType,
    ShapeTooLarge,
    OutOfMemory,
    ResourceExhausted,
};

/// Buffer for model weights - uses mmap with MAP_POPULATE for zero-copy loading
const MappedBuffer = struct {
    mapped_data: []align(std.heap.page_size_min) u8,

    /// Validate that mapped memory is accessible by touching pages.
    /// This converts potential SIGBUS (fatal) into a catchable error.
    /// Only validates the header region which is needed immediately.
    fn validateHeaderRegion(self: *const MappedBuffer, header_size: usize) !void {
        if (self.mapped_data.len == 0) return;

        const page_size = std.heap.page_size_min;
        const validate_size = @min(header_size, self.mapped_data.len);

        // Touch each page in the header region to force page faults now
        // rather than later when we try to parse. If memory is exhausted,
        // the page fault will fail and we return ResourceExhausted.
        var offset: usize = 0;
        while (offset < validate_size) : (offset += page_size) {
            // Volatile read to prevent optimization
            const ptr: *const volatile u8 = @ptrCast(&self.mapped_data[offset]);
            _ = ptr.*;
        }
    }

    fn mapFromFile(file: std.fs.File, size: usize) !MappedBuffer {
        // Optionally use MAP_POPULATE to fault in all pages immediately.
        // For large shards this can take minutes and appears like a hang.
        var flags: std.posix.MAP = .{ .TYPE = .PRIVATE };
        mmap_policy.applyReadOnlyMapPolicy(&flags, size);

        const data = try std.posix.mmap(
            null,
            size,
            std.posix.PROT.READ,
            flags,
            file.handle,
            0,
        );

        return .{ .mapped_data = data };
    }

    fn deinit(self: MappedBuffer) void {
        if (self.mapped_data.len > 0) std.posix.munmap(self.mapped_data);
    }
};

pub const SafeTensors = struct {
    allocator: std.mem.Allocator,
    buffer: MappedBuffer,
    entries: std.StringHashMapUnmanaged(Entry) = .{},
    data_start: usize,

    pub const Entry = struct {
        dtype: DType,
        shape: []usize,
        data: []const u8,
    };

    pub fn load(allocator: std.mem.Allocator, path: []const u8) !SafeTensors {
        var st_file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });
        defer st_file.close();

        const file_stat = try st_file.stat();
        if (file_stat.size < 8) return error.InvalidFile;
        const mapped_size: usize = @intCast(file_stat.size);

        // mmap with MAP_POPULATE - zero-copy, shares page cache, pre-faults all pages
        var mapped_buffer = try MappedBuffer.mapFromFile(st_file, mapped_size);
        errdefer mapped_buffer.deinit();

        // Read header size first (8 bytes)
        const max_header_bytes: usize = 10 * 1024 * 1024;
        const header_bytes = std.mem.readInt(u64, mapped_buffer.mapped_data[0..8], .little);
        if (header_bytes > @as(u64, max_header_bytes)) return error.InvalidHeader;
        const header_end_offset = 8 + header_bytes;

        // Validate header region is accessible before parsing.
        // This converts potential SIGBUS into a catchable ResourceExhausted error.
        mapped_buffer.validateHeaderRegion(@intCast(header_end_offset)) catch {
            return error.ResourceExhausted;
        };

        if (header_end_offset > mapped_buffer.mapped_data.len) return error.InvalidHeader;

        var safetensors = SafeTensors{
            .allocator = allocator,
            .buffer = mapped_buffer,
            .data_start = @intCast(header_end_offset),
        };

        var parsed_json = json.parseValue(allocator, mapped_buffer.mapped_data[8..header_end_offset], .{ .max_size_bytes = max_header_bytes }) catch |err| {
            return switch (err) {
                error.InputTooLarge => error.InvalidHeader,
                error.InputTooDeep => error.InvalidHeader,
                error.StringTooLong => error.InvalidHeader,
                error.InvalidJson => error.InvalidHeader,
                error.OutOfMemory => error.OutOfMemory,
            };
        };
        defer parsed_json.deinit();

        if (parsed_json.value != .object) return error.InvalidHeader;

        var entry_iter = parsed_json.value.object.iterator();
        while (entry_iter.next()) |kv| {
            const tensor_name = kv.key_ptr.*;
            const meta = kv.value_ptr.*;
            if (meta != .object) continue;

            const entry_dtype = parseDType(meta.object.get("dtype") orelse continue) orelse continue;
            const shape = try parseShape(allocator, meta.object.get("shape") orelse continue);
            errdefer allocator.free(shape);

            const offsets = parseDataOffsets(meta.object.get("data_offsets") orelse {
                allocator.free(shape);
                continue;
            }) catch {
                allocator.free(shape);
                continue;
            };
            const start = safetensors.data_start + offsets[0];
            const end = safetensors.data_start + offsets[1];
            if (start > end or end > mapped_buffer.mapped_data.len) {
                allocator.free(shape);
                continue;
            }

            const stored_name = try allocator.dupe(u8, tensor_name);
            try safetensors.entries.put(allocator, stored_name, .{
                .dtype = entry_dtype,
                .shape = shape,
                .data = mapped_buffer.mapped_data[start..end],
            });
        }

        return safetensors;
    }

    pub fn deinit(self: *SafeTensors) void {
        var entry_iter = self.entries.iterator();
        while (entry_iter.next()) |kv| {
            self.allocator.free(kv.key_ptr.*);
            self.allocator.free(kv.value_ptr.shape);
        }
        self.entries.deinit(self.allocator);
        self.buffer.deinit();
        self.* = undefined;
    }

    pub fn getTensor(self: *const SafeTensors, name: []const u8, expected_dtype: ?DType) !Tensor {
        const tensor_entry = self.entries.get(name) orelse return error.NotFound;
        if (expected_dtype) |dt| {
            if (tensor_entry.dtype != dt) return error.UnexpectedDType;
        }

        if (tensor_entry.shape.len > tensor.MAX_NDIM) return error.ShapeTooLarge;

        var tensor_view: tensor.Tensor = undefined;
        tensor_view.dtype = tensor_entry.dtype;
        tensor_view.n_dims = @intCast(tensor_entry.shape.len);
        tensor_view.data_ptr = @constCast(tensor_entry.data.ptr);
        tensor_view.data_size = tensor_entry.data.len;
        tensor_view.device = tensor.Device.cpu();
        tensor_view.owns_data = false;
        tensor_view.gaffine = null;

        // Copy shape and compute numel
        var element_count: usize = 1;
        for (0..tensor_entry.shape.len) |dim_index| {
            tensor_view.shape[dim_index] = @intCast(tensor_entry.shape[dim_index]);
            element_count *= tensor_entry.shape[dim_index];
        }
        for (tensor_entry.shape.len..tensor.MAX_NDIM) |dim_index| {
            tensor_view.shape[dim_index] = 0;
        }
        tensor_view.numel = element_count;

        // Compute strides
        var stride_value: i64 = 1;
        var dim_index: usize = tensor_entry.shape.len;
        while (dim_index > 0) {
            dim_index -= 1;
            tensor_view.strides[dim_index] = stride_value;
            stride_value *= @intCast(tensor_entry.shape[dim_index]);
        }
        for (tensor_entry.shape.len..tensor.MAX_NDIM) |stride_index| {
            tensor_view.strides[stride_index] = 0;
        }

        return tensor_view;
    }

    pub fn hasTensor(self: *const SafeTensors, name: []const u8) bool {
        return self.entries.contains(name);
    }

    /// Get a list of all tensor names in the file
    pub fn tensorNames(self: *const SafeTensors, allocator: std.mem.Allocator) ![][]const u8 {
        var tensor_names = try allocator.alloc([]const u8, self.entries.count());
        var name_index: usize = 0;
        var entry_iter = self.entries.iterator();
        while (entry_iter.next()) |kv| {
            tensor_names[name_index] = kv.key_ptr.*;
            name_index += 1;
        }
        return tensor_names;
    }

    /// Get file size in bytes
    pub fn fileSize(self: *const SafeTensors) usize {
        return self.buffer.mapped_data.len;
    }

    /// Get number of tensors in the file
    pub fn tensorCount(self: *const SafeTensors) usize {
        return self.entries.count();
    }
};

/// Parse SafeTensors dtype string to internal DType.
///
/// NOTE: U8 is mapped to .i8 because SafeTensors uses unsigned, but our
/// tensor ops treat int8 indices uniformly. U32 is mapped to .grouped_affine_u4
/// because MLX-quantized models store packed 4-bit weights as U32 with separate
/// scales/biases tensors. The actual bit-width (4 or 8) is auto-detected at
/// load time in orientWeight() based on scales shape.
fn parseDType(value: std.json.Value) ?DType {
    if (value != .string) return null;
    return std.StaticStringMap(DType).initComptime(.{
        .{ "F32", .f32 },
        .{ "F16", .f16 },
        .{ "BF16", .bf16 },
        .{ "I8", .i8 },
        .{ "U8", .i8 }, // Unsigned treated as signed for index ops
        .{ "I64", .i64 },
        .{ "U32", .grouped_affine_u4 }, // MLX packed weights; actual bits detected later
        .{ "F8_E4M3", .f8_e4m3 },
    }).get(value.string);
}

fn parseShape(allocator: std.mem.Allocator, value: std.json.Value) ![]usize {
    if (value != .array) return error.InvalidShape;
    const shape_items = value.array.items;
    const dims = try allocator.alloc(usize, shape_items.len);
    errdefer allocator.free(dims);
    for (shape_items, 0..) |item, dim_idx| {
        dims[dim_idx] = switch (item) {
            .integer => |n| @intCast(n),
            .float => |f| @intFromFloat(f),
            else => return error.InvalidShape,
        };
    }
    return dims;
}

fn parseDataOffsets(value: std.json.Value) ![2]usize {
    if (value != .array or value.array.items.len != 2) return error.InvalidOffsets;
    var offsets: [2]usize = undefined;
    for (value.array.items, 0..) |item, offset_idx| {
        offsets[offset_idx] = switch (item) {
            .integer => |n| @intCast(n),
            .float => |f| @intFromFloat(f),
            else => return error.InvalidOffsets,
        };
    }
    return offsets;
}

pub fn tryGetBytes(st: *const SafeTensors, base: []const u8, suffix: []const u8) ?[]u8 {
    var name_buffer: [256]u8 = undefined;
    const tensor_name = std.fmt.bufPrint(&name_buffer, "{s}{s}", .{ base, suffix }) catch return null;
    const tensor_entry = st.entries.get(tensor_name) orelse return null;
    return @constCast(tensor_entry.data);
}

// ============================================================================
// Unit Tests
// ============================================================================

test "parseDType: valid dtype strings" {
    const testing = std.testing;

    // Test float types
    try testing.expectEqual(DType.f32, parseDType(.{ .string = "F32" }).?);
    try testing.expectEqual(DType.f16, parseDType(.{ .string = "F16" }).?);
    try testing.expectEqual(DType.bf16, parseDType(.{ .string = "BF16" }).?);
    try testing.expectEqual(DType.f8_e4m3, parseDType(.{ .string = "F8_E4M3" }).?);

    // Test integer types
    try testing.expectEqual(DType.i8, parseDType(.{ .string = "I8" }).?);
    try testing.expectEqual(DType.i8, parseDType(.{ .string = "U8" }).?); // U8 maps to i8
    try testing.expectEqual(DType.i64, parseDType(.{ .string = "I64" }).?);
    try testing.expectEqual(DType.grouped_affine_u4, parseDType(.{ .string = "U32" }).?); // U32 maps to grouped_affine_u4

}

test "parseDType: invalid dtype strings" {
    // Unknown dtype should return null
    try std.testing.expectEqual(@as(?DType, null), parseDType(.{ .string = "UNKNOWN" }));
    try std.testing.expectEqual(@as(?DType, null), parseDType(.{ .string = "f32" })); // Wrong case
    try std.testing.expectEqual(@as(?DType, null), parseDType(.{ .string = "" }));
}

test "parseDType: non-string values" {
    // Non-string JSON values should return null
    try std.testing.expectEqual(@as(?DType, null), parseDType(.{ .integer = 42 }));
    try std.testing.expectEqual(@as(?DType, null), parseDType(.{ .float = 3.14 }));
    try std.testing.expectEqual(@as(?DType, null), parseDType(.{ .null = {} }));
}

test "parseShape: valid shapes" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test 1D shape
    {
        var shape_arr = std.json.Array.init(allocator);
        defer shape_arr.deinit();
        try shape_arr.append(.{ .integer = 10 });

        const shape = try parseShape(allocator, .{ .array = shape_arr });
        defer allocator.free(shape);

        try testing.expectEqual(@as(usize, 1), shape.len);
        try testing.expectEqual(@as(usize, 10), shape[0]);
    }

    // Test 2D shape
    {
        var shape_arr = std.json.Array.init(allocator);
        defer shape_arr.deinit();
        try shape_arr.append(.{ .integer = 3 });
        try shape_arr.append(.{ .integer = 4 });

        const shape = try parseShape(allocator, .{ .array = shape_arr });
        defer allocator.free(shape);

        try testing.expectEqual(@as(usize, 2), shape.len);
        try testing.expectEqual(@as(usize, 3), shape[0]);
        try testing.expectEqual(@as(usize, 4), shape[1]);
    }

    // Test 4D shape
    {
        var shape_arr = std.json.Array.init(allocator);
        defer shape_arr.deinit();
        try shape_arr.append(.{ .integer = 2 });
        try shape_arr.append(.{ .integer = 3 });
        try shape_arr.append(.{ .integer = 4 });
        try shape_arr.append(.{ .integer = 5 });

        const shape = try parseShape(allocator, .{ .array = shape_arr });
        defer allocator.free(shape);

        try testing.expectEqual(@as(usize, 4), shape.len);
        try testing.expectEqual(@as(usize, 2), shape[0]);
        try testing.expectEqual(@as(usize, 3), shape[1]);
        try testing.expectEqual(@as(usize, 4), shape[2]);
        try testing.expectEqual(@as(usize, 5), shape[3]);
    }

    // Test empty shape (scalar)
    {
        var shape_arr = std.json.Array.init(allocator);
        defer shape_arr.deinit();

        const shape = try parseShape(allocator, .{ .array = shape_arr });
        defer allocator.free(shape);

        try testing.expectEqual(@as(usize, 0), shape.len);
    }
}

test "parseShape: float values in shape" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var shape_arr = std.json.Array.init(allocator);
    defer shape_arr.deinit();
    try shape_arr.append(.{ .float = 10.0 });
    try shape_arr.append(.{ .float = 20.5 }); // Will truncate to 20

    const shape = try parseShape(allocator, .{ .array = shape_arr });
    defer allocator.free(shape);

    try testing.expectEqual(@as(usize, 2), shape.len);
    try testing.expectEqual(@as(usize, 10), shape[0]);
    try testing.expectEqual(@as(usize, 20), shape[1]);
}

test "parseShape: invalid shape values" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test non-array value
    try testing.expectError(error.InvalidShape, parseShape(allocator, .{ .integer = 42 }));
    try testing.expectError(error.InvalidShape, parseShape(allocator, .{ .string = "[1,2,3]" }));

    // Test array with invalid item types
    {
        var shape_arr = std.json.Array.init(allocator);
        defer shape_arr.deinit();
        try shape_arr.append(.{ .integer = 10 });
        try shape_arr.append(.{ .string = "invalid" });

        try testing.expectError(error.InvalidShape, parseShape(allocator, .{ .array = shape_arr }));
    }
}

test "parseDataOffsets: valid offsets" {
    const testing = std.testing;

    // Test with integers
    {
        var offset_arr = std.json.Array.init(testing.allocator);
        defer offset_arr.deinit();
        try offset_arr.append(.{ .integer = 0 });
        try offset_arr.append(.{ .integer = 1024 });

        const offsets = try parseDataOffsets(.{ .array = offset_arr });
        try testing.expectEqual(@as(usize, 0), offsets[0]);
        try testing.expectEqual(@as(usize, 1024), offsets[1]);
    }

    // Test with floats
    {
        var offset_arr = std.json.Array.init(testing.allocator);
        defer offset_arr.deinit();
        try offset_arr.append(.{ .float = 100.0 });
        try offset_arr.append(.{ .float = 200.5 }); // Will truncate to 200

        const offsets = try parseDataOffsets(.{ .array = offset_arr });
        try testing.expectEqual(@as(usize, 100), offsets[0]);
        try testing.expectEqual(@as(usize, 200), offsets[1]);
    }
}

test "parseDataOffsets: invalid offsets" {
    const testing = std.testing;

    // Non-array value
    try testing.expectError(error.InvalidOffsets, parseDataOffsets(.{ .integer = 42 }));

    // Wrong array length
    {
        var offset_arr = std.json.Array.init(testing.allocator);
        defer offset_arr.deinit();
        try offset_arr.append(.{ .integer = 0 });

        try testing.expectError(error.InvalidOffsets, parseDataOffsets(.{ .array = offset_arr }));
    }

    {
        var offset_arr = std.json.Array.init(testing.allocator);
        defer offset_arr.deinit();
        try offset_arr.append(.{ .integer = 0 });
        try offset_arr.append(.{ .integer = 100 });
        try offset_arr.append(.{ .integer = 200 });

        try testing.expectError(error.InvalidOffsets, parseDataOffsets(.{ .array = offset_arr }));
    }

    // Invalid item types
    {
        var offset_arr = std.json.Array.init(testing.allocator);
        defer offset_arr.deinit();
        try offset_arr.append(.{ .string = "0" });
        try offset_arr.append(.{ .integer = 100 });

        try testing.expectError(error.InvalidOffsets, parseDataOffsets(.{ .array = offset_arr }));
    }
}

/// Helper to create a minimal valid SafeTensors file in memory
/// Format: [8 bytes header length][JSON header][tensor data]
fn createMockSafeTensorsFile(allocator: std.mem.Allocator, header_json: []const u8, tensor_data: []const u8) ![]u8 {
    const header_len = header_json.len;
    const total_size = 8 + header_len + tensor_data.len;

    var buffer = try allocator.alloc(u8, total_size);

    // Write header length (little-endian u64)
    std.mem.writeInt(u64, buffer[0..8], @intCast(header_len), .little);

    // Write JSON header
    @memcpy(buffer[8..8 + header_len], header_json);

    // Write tensor data
    @memcpy(buffer[8 + header_len..], tensor_data);

    return buffer;
}

test "SafeTensors: load valid file with single tensor" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Create a simple F32 tensor with shape [2, 3]
    const tensor_data = [_]u8{
        0, 0, 0, 0,  // 0.0
        0, 0, 128, 63,  // 1.0
        0, 0, 0, 64,  // 2.0
        0, 0, 64, 64,  // 3.0
        0, 0, 128, 64,  // 4.0
        0, 0, 160, 64,  // 5.0
    };

    const header =
        \\{"weight": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}}
    ;

    const file_data = try createMockSafeTensorsFile(allocator, header, &tensor_data);
    defer allocator.free(file_data);

    // Write to a temporary file
    const tmp_dir = testing.tmpDir(.{});
    var tmp_file = try tmp_dir.dir.createFile("test.safetensors", .{ .read = true });
    defer tmp_file.close();
    try tmp_file.writeAll(file_data);
    try tmp_file.seekTo(0);

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path = try tmp_dir.dir.realpath("test.safetensors", &path_buf);

    // Load SafeTensors
    var st = try SafeTensors.load(allocator, tmp_path);
    defer st.deinit();

    // Verify tensor count and names
    try testing.expectEqual(@as(usize, 1), st.tensorCount());
    try testing.expect(st.hasTensor("weight"));
    try testing.expect(!st.hasTensor("nonexistent"));

    // Get tensor and verify properties
    const t = try st.getTensor("weight", .f32);
    try testing.expectEqual(DType.f32, t.dtype);
    try testing.expectEqual(@as(i32, 2), t.n_dims);
    try testing.expectEqual(@as(i64, 2), t.shape[0]);
    try testing.expectEqual(@as(i64, 3), t.shape[1]);
    try testing.expectEqual(@as(usize, 6), t.numel);
    try testing.expect(!t.owns_data);
}

test "SafeTensors: load file with multiple tensors" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Create two tensors
    const tensor_data = [_]u8{
        0, 0, 128, 63,  // weight1: [1.0]
        0, 0, 0, 64,  // weight2: [2.0]
        0, 0, 64, 64,  // [3.0]
    };

    const header =
        \\{"weight1": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        \\ "weight2": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]}}
    ;

    const file_data = try createMockSafeTensorsFile(allocator, header, &tensor_data);
    defer allocator.free(file_data);

    const tmp_dir = testing.tmpDir(.{});
    var tmp_file = try tmp_dir.dir.createFile("test.safetensors", .{ .read = true });
    defer tmp_file.close();
    try tmp_file.writeAll(file_data);
    try tmp_file.seekTo(0);

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path = try tmp_dir.dir.realpath("test.safetensors", &path_buf);

    var st = try SafeTensors.load(allocator, tmp_path);
    defer st.deinit();

    try testing.expectEqual(@as(usize, 2), st.tensorCount());
    try testing.expect(st.hasTensor("weight1"));
    try testing.expect(st.hasTensor("weight2"));

    // Verify both tensors
    const t1 = try st.getTensor("weight1", .f32);
    try testing.expectEqual(@as(usize, 1), t1.numel);
    try testing.expectEqual(@as(i64, 1), t1.shape[0]);

    const t2 = try st.getTensor("weight2", .f32);
    try testing.expectEqual(@as(usize, 2), t2.numel);
    try testing.expectEqual(@as(i64, 2), t2.shape[0]);
}

test "SafeTensors: getTensor with dtype validation" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const tensor_data = [_]u8{0, 0, 128, 63}; // 1.0 in F32
    const header =
        \\{"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    ;

    const file_data = try createMockSafeTensorsFile(allocator, header, &tensor_data);
    defer allocator.free(file_data);

    const tmp_dir = testing.tmpDir(.{});
    var tmp_file = try tmp_dir.dir.createFile("test.safetensors", .{ .read = true });
    defer tmp_file.close();
    try tmp_file.writeAll(file_data);
    try tmp_file.seekTo(0);

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path = try tmp_dir.dir.realpath("test.safetensors", &path_buf);

    var st = try SafeTensors.load(allocator, tmp_path);
    defer st.deinit();

    // Should succeed with correct dtype
    _ = try st.getTensor("weight", .f32);

    // Should succeed with null (no dtype check)
    _ = try st.getTensor("weight", null);

    // Should fail with wrong dtype
    try testing.expectError(error.UnexpectedDType, st.getTensor("weight", .f16));
    try testing.expectError(error.UnexpectedDType, st.getTensor("weight", .i8));
}

test "SafeTensors: getTensor for nonexistent tensor" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const tensor_data = [_]u8{0, 0, 128, 63};
    const header =
        \\{"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    ;

    const file_data = try createMockSafeTensorsFile(allocator, header, &tensor_data);
    defer allocator.free(file_data);

    const tmp_dir = testing.tmpDir(.{});
    var tmp_file = try tmp_dir.dir.createFile("test.safetensors", .{ .read = true });
    defer tmp_file.close();
    try tmp_file.writeAll(file_data);
    try tmp_file.seekTo(0);

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path = try tmp_dir.dir.realpath("test.safetensors", &path_buf);

    var st = try SafeTensors.load(allocator, tmp_path);
    defer st.deinit();

    try testing.expectError(error.NotFound, st.getTensor("nonexistent", null));
}

test "SafeTensors: load file with different dtypes" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Create tensors with different dtypes
    const tensor_data = [_]u8{
        0, 0, 128, 63,  // f32_weight: 1.0 (F32, 4 bytes)
        0, 60,          // f16_weight: 1.0 (F16, 2 bytes)
        42,             // i8_weight: 42 (I8, 1 byte)
    };

    const header =
        \\{"f32_weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        \\ "f16_weight": {"dtype": "F16", "shape": [1], "data_offsets": [4, 6]},
        \\ "i8_weight": {"dtype": "I8", "shape": [1], "data_offsets": [6, 7]}}
    ;

    const file_data = try createMockSafeTensorsFile(allocator, header, &tensor_data);
    defer allocator.free(file_data);

    const tmp_dir = testing.tmpDir(.{});
    var tmp_file = try tmp_dir.dir.createFile("test.safetensors", .{ .read = true });
    defer tmp_file.close();
    try tmp_file.writeAll(file_data);
    try tmp_file.seekTo(0);

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path = try tmp_dir.dir.realpath("test.safetensors", &path_buf);

    var st = try SafeTensors.load(allocator, tmp_path);
    defer st.deinit();

    try testing.expectEqual(@as(usize, 3), st.tensorCount());

    const t1 = try st.getTensor("f32_weight", .f32);
    try testing.expectEqual(DType.f32, t1.dtype);
    try testing.expectEqual(@as(usize, 4), t1.data_size);

    const t2 = try st.getTensor("f16_weight", .f16);
    try testing.expectEqual(DType.f16, t2.dtype);
    try testing.expectEqual(@as(usize, 2), t2.data_size);

    const t3 = try st.getTensor("i8_weight", .i8);
    try testing.expectEqual(DType.i8, t3.dtype);
    try testing.expectEqual(@as(usize, 1), t3.data_size);
}

test "SafeTensors: tensor strides calculation" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Create a 3D tensor [2, 3, 4]
    const tensor_data = try allocator.alloc(u8, 2 * 3 * 4 * 4); // F32
    defer allocator.free(tensor_data);
    @memset(tensor_data, 0);

    const header =
        \\{"weight": {"dtype": "F32", "shape": [2, 3, 4], "data_offsets": [0, 96]}}
    ;

    const file_data = try createMockSafeTensorsFile(allocator, header, tensor_data);
    defer allocator.free(file_data);

    const tmp_dir = testing.tmpDir(.{});
    var tmp_file = try tmp_dir.dir.createFile("test.safetensors", .{ .read = true });
    defer tmp_file.close();
    try tmp_file.writeAll(file_data);
    try tmp_file.seekTo(0);

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path = try tmp_dir.dir.realpath("test.safetensors", &path_buf);

    var st = try SafeTensors.load(allocator, tmp_path);
    defer st.deinit();

    const t = try st.getTensor("weight", .f32);

    // Verify strides for [2, 3, 4] tensor (row-major)
    // strides should be [12, 4, 1]
    try testing.expectEqual(@as(i64, 12), t.strides[0]);
    try testing.expectEqual(@as(i64, 4), t.strides[1]);
    try testing.expectEqual(@as(i64, 1), t.strides[2]);

    // Unused dimensions should have 0 stride
    try testing.expectEqual(@as(i64, 0), t.strides[3]);
}

test "SafeTensors: tensorNames listing" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const tensor_data = [_]u8{0} ** 12;
    const header =
        \\{"weight1": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        \\ "weight2": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]},
        \\ "bias": {"dtype": "F32", "shape": [1], "data_offsets": [8, 12]}}
    ;

    const file_data = try createMockSafeTensorsFile(allocator, header, &tensor_data);
    defer allocator.free(file_data);

    const tmp_dir = testing.tmpDir(.{});
    var tmp_file = try tmp_dir.dir.createFile("test.safetensors", .{ .read = true });
    defer tmp_file.close();
    try tmp_file.writeAll(file_data);
    try tmp_file.seekTo(0);

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path = try tmp_dir.dir.realpath("test.safetensors", &path_buf);

    var st = try SafeTensors.load(allocator, tmp_path);
    defer st.deinit();

    const names = try st.tensorNames(allocator);
    defer allocator.free(names);

    try testing.expectEqual(@as(usize, 3), names.len);

    // Check that all expected names are present (order may vary)
    var found_weight1 = false;
    var found_weight2 = false;
    var found_bias = false;

    for (names) |name| {
        if (std.mem.eql(u8, name, "weight1")) found_weight1 = true;
        if (std.mem.eql(u8, name, "weight2")) found_weight2 = true;
        if (std.mem.eql(u8, name, "bias")) found_bias = true;
    }

    try testing.expect(found_weight1);
    try testing.expect(found_weight2);
    try testing.expect(found_bias);
}

test "SafeTensors: fileSize reporting" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const tensor_data = [_]u8{0} ** 100;
    const header =
        \\{"weight": {"dtype": "F32", "shape": [25], "data_offsets": [0, 100]}}
    ;

    const file_data = try createMockSafeTensorsFile(allocator, header, &tensor_data);
    defer allocator.free(file_data);

    const expected_size = file_data.len;

    const tmp_dir = testing.tmpDir(.{});
    var tmp_file = try tmp_dir.dir.createFile("test.safetensors", .{ .read = true });
    defer tmp_file.close();
    try tmp_file.writeAll(file_data);
    try tmp_file.seekTo(0);

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path = try tmp_dir.dir.realpath("test.safetensors", &path_buf);

    var st = try SafeTensors.load(allocator, tmp_path);
    defer st.deinit();

    try testing.expectEqual(expected_size, st.fileSize());
}

test "SafeTensors: tryGetBytes helper" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const tensor_data = [_]u8{1, 2, 3, 4, 5, 6, 7, 8};
    const header =
        \\{"model.weight": {"dtype": "I8", "shape": [4], "data_offsets": [0, 4]},
        \\ "model.bias": {"dtype": "I8", "shape": [4], "data_offsets": [4, 8]}}
    ;

    const file_data = try createMockSafeTensorsFile(allocator, header, &tensor_data);
    defer allocator.free(file_data);

    const tmp_dir = testing.tmpDir(.{});
    var tmp_file = try tmp_dir.dir.createFile("test.safetensors", .{ .read = true });
    defer tmp_file.close();
    try tmp_file.writeAll(file_data);
    try tmp_file.seekTo(0);

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path = try tmp_dir.dir.realpath("test.safetensors", &path_buf);

    var st = try SafeTensors.load(allocator, tmp_path);
    defer st.deinit();

    // Test successful lookup
    const weight_data = tryGetBytes(&st, "model.", "weight");
    try testing.expect(weight_data != null);
    try testing.expectEqual(@as(usize, 4), weight_data.?.len);

    const bias_data = tryGetBytes(&st, "model.", "bias");
    try testing.expect(bias_data != null);
    try testing.expectEqual(@as(usize, 4), bias_data.?.len);

    // Test failed lookup
    const missing_data = tryGetBytes(&st, "model.", "missing");
    try testing.expect(missing_data == null);
}

test "SafeTensors: invalid file - too small" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Create a file that's less than 8 bytes (minimum for header length)
    const small_data = [_]u8{1, 2, 3};

    const tmp_dir = testing.tmpDir(.{});
    var tmp_file = try tmp_dir.dir.createFile("test.safetensors", .{ .read = true });
    defer tmp_file.close();
    try tmp_file.writeAll(&small_data);
    try tmp_file.seekTo(0);

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path = try tmp_dir.dir.realpath("test.safetensors", &path_buf);

    try testing.expectError(error.InvalidFile, SafeTensors.load(allocator, tmp_path));
}

test "SafeTensors: invalid file - header extends beyond file" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Create a file where header length claims more data than exists
    var buffer: [100]u8 = undefined;
    std.mem.writeInt(u64, buffer[0..8], 1000, .little); // Claims 1000 bytes of header
    @memset(buffer[8..], 0);

    const tmp_dir = testing.tmpDir(.{});
    var tmp_file = try tmp_dir.dir.createFile("test.safetensors", .{ .read = true });
    defer tmp_file.close();
    try tmp_file.writeAll(&buffer);
    try tmp_file.seekTo(0);

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path = try tmp_dir.dir.realpath("test.safetensors", &path_buf);

    try testing.expectError(error.InvalidHeader, SafeTensors.load(allocator, tmp_path));
}

test "SafeTensors: invalid file - malformed JSON header" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const invalid_json = "not valid json at all";
    const file_data = try createMockSafeTensorsFile(allocator, invalid_json, &[_]u8{});
    defer allocator.free(file_data);

    const tmp_dir = testing.tmpDir(.{});
    var tmp_file = try tmp_dir.dir.createFile("test.safetensors", .{ .read = true });
    defer tmp_file.close();
    try tmp_file.writeAll(file_data);
    try tmp_file.seekTo(0);

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path = try tmp_dir.dir.realpath("test.safetensors", &path_buf);

    try testing.expectError(error.InvalidHeader, SafeTensors.load(allocator, tmp_path));
}

test "SafeTensors: invalid file - non-object JSON header" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const array_json = "[1, 2, 3]";
    const file_data = try createMockSafeTensorsFile(allocator, array_json, &[_]u8{});
    defer allocator.free(file_data);

    const tmp_dir = testing.tmpDir(.{});
    var tmp_file = try tmp_dir.dir.createFile("test.safetensors", .{ .read = true });
    defer tmp_file.close();
    try tmp_file.writeAll(file_data);
    try tmp_file.seekTo(0);

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path = try tmp_dir.dir.realpath("test.safetensors", &path_buf);

    try testing.expectError(error.InvalidHeader, SafeTensors.load(allocator, tmp_path));
}

test "SafeTensors: skip invalid tensor entries" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Mix of valid and invalid tensor entries
    // Invalid entries should be silently skipped
    const tensor_data = [_]u8{0, 0, 128, 63}; // 4 bytes for valid tensor
    const header =
        \\{"valid": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        \\ "missing_dtype": {"shape": [1], "data_offsets": [0, 4]},
        \\ "missing_shape": {"dtype": "F32", "data_offsets": [0, 4]},
        \\ "missing_offsets": {"dtype": "F32", "shape": [1]},
        \\ "invalid_dtype": {"dtype": "UNKNOWN", "shape": [1], "data_offsets": [0, 4]}}
    ;

    const file_data = try createMockSafeTensorsFile(allocator, header, &tensor_data);
    defer allocator.free(file_data);

    const tmp_dir = testing.tmpDir(.{});
    var tmp_file = try tmp_dir.dir.createFile("test.safetensors", .{ .read = true });
    defer tmp_file.close();
    try tmp_file.writeAll(file_data);
    try tmp_file.seekTo(0);

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path = try tmp_dir.dir.realpath("test.safetensors", &path_buf);

    var st = try SafeTensors.load(allocator, tmp_path);
    defer st.deinit();

    // Only the valid tensor should be loaded
    try testing.expectEqual(@as(usize, 1), st.tensorCount());
    try testing.expect(st.hasTensor("valid"));
    try testing.expect(!st.hasTensor("missing_dtype"));
    try testing.expect(!st.hasTensor("missing_shape"));
    try testing.expect(!st.hasTensor("missing_offsets"));
    try testing.expect(!st.hasTensor("invalid_dtype"));
}

test "SafeTensors: skip tensor with out-of-bounds offsets" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const tensor_data = [_]u8{0, 0, 128, 63}; // Only 4 bytes of data
    const header =
        \\{"valid": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        \\ "out_of_bounds": {"dtype": "F32", "shape": [10], "data_offsets": [0, 1000]},
        \\ "reversed": {"dtype": "F32", "shape": [1], "data_offsets": [4, 0]}}
    ;

    const file_data = try createMockSafeTensorsFile(allocator, header, &tensor_data);
    defer allocator.free(file_data);

    const tmp_dir = testing.tmpDir(.{});
    var tmp_file = try tmp_dir.dir.createFile("test.safetensors", .{ .read = true });
    defer tmp_file.close();
    try tmp_file.writeAll(file_data);
    try tmp_file.seekTo(0);

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path = try tmp_dir.dir.realpath("test.safetensors", &path_buf);

    var st = try SafeTensors.load(allocator, tmp_path);
    defer st.deinit();

    // Only the valid tensor should be loaded
    try testing.expectEqual(@as(usize, 1), st.tensorCount());
    try testing.expect(st.hasTensor("valid"));
    try testing.expect(!st.hasTensor("out_of_bounds"));
    try testing.expect(!st.hasTensor("reversed"));
}

test "SafeTensors: empty file (no tensors)" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const header = "{}";
    const file_data = try createMockSafeTensorsFile(allocator, header, &[_]u8{});
    defer allocator.free(file_data);

    const tmp_dir = testing.tmpDir(.{});
    var tmp_file = try tmp_dir.dir.createFile("test.safetensors", .{ .read = true });
    defer tmp_file.close();
    try tmp_file.writeAll(file_data);
    try tmp_file.seekTo(0);

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path = try tmp_dir.dir.realpath("test.safetensors", &path_buf);

    var st = try SafeTensors.load(allocator, tmp_path);
    defer st.deinit();

    try testing.expectEqual(@as(usize, 0), st.tensorCount());

    const names = try st.tensorNames(allocator);
    defer allocator.free(names);
    try testing.expectEqual(@as(usize, 0), names.len);
}

test "SafeTensors.deinit: properly cleans up resources" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const tensor_data = [_]u8{
        0, 0, 128, 63,  // weight1: [1.0]
        0, 0, 0, 64,    // weight2: [2.0]
    };

    const header =
        \\{"weight1": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        \\ "weight2": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]}}
    ;

    const file_data = try createMockSafeTensorsFile(allocator, header, &tensor_data);
    defer allocator.free(file_data);

    const tmp_dir = testing.tmpDir(.{});
    var tmp_file = try tmp_dir.dir.createFile("test.safetensors", .{ .read = true });
    defer tmp_file.close();
    try tmp_file.writeAll(file_data);
    try tmp_file.seekTo(0);

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path = try tmp_dir.dir.realpath("test.safetensors", &path_buf);

    var st = try SafeTensors.load(allocator, tmp_path);

    // Verify we have tensors before cleanup
    try testing.expectEqual(@as(usize, 2), st.tensorCount());
    try testing.expect(st.hasTensor("weight1"));
    try testing.expect(st.hasTensor("weight2"));

    // Clean up - this should not leak memory
    st.deinit();
}

test "SafeTensors.hasTensor: returns true for existing tensors" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const tensor_data = [_]u8{
        0, 0, 128, 63,  // model.weight: [1.0]
        0, 0, 0, 64,    // model.bias: [2.0]
    };

    const header =
        \\{"model.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        \\ "model.bias": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]}}
    ;

    const file_data = try createMockSafeTensorsFile(allocator, header, &tensor_data);
    defer allocator.free(file_data);

    const tmp_dir = testing.tmpDir(.{});
    var tmp_file = try tmp_dir.dir.createFile("test.safetensors", .{ .read = true });
    defer tmp_file.close();
    try tmp_file.writeAll(file_data);
    try tmp_file.seekTo(0);

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path = try tmp_dir.dir.realpath("test.safetensors", &path_buf);

    var st = try SafeTensors.load(allocator, tmp_path);
    defer st.deinit();

    // Test existing tensors return true
    try testing.expect(st.hasTensor("model.weight"));
    try testing.expect(st.hasTensor("model.bias"));
}

test "SafeTensors.hasTensor: returns false for nonexistent tensors" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const tensor_data = [_]u8{0, 0, 128, 63};
    const header =
        \\{"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    ;

    const file_data = try createMockSafeTensorsFile(allocator, header, &tensor_data);
    defer allocator.free(file_data);

    const tmp_dir = testing.tmpDir(.{});
    var tmp_file = try tmp_dir.dir.createFile("test.safetensors", .{ .read = true });
    defer tmp_file.close();
    try tmp_file.writeAll(file_data);
    try tmp_file.seekTo(0);

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path = try tmp_dir.dir.realpath("test.safetensors", &path_buf);

    var st = try SafeTensors.load(allocator, tmp_path);
    defer st.deinit();

    // Test nonexistent tensors return false
    try testing.expect(!st.hasTensor("nonexistent"));
    try testing.expect(!st.hasTensor("missing_tensor"));
    try testing.expect(!st.hasTensor(""));
    try testing.expect(!st.hasTensor("weight2"));
}

test "SafeTensors.hasTensor: case sensitive and exact match" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const tensor_data = [_]u8{0, 0, 128, 63};
    const header =
        \\{"MyTensor": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    ;

    const file_data = try createMockSafeTensorsFile(allocator, header, &tensor_data);
    defer allocator.free(file_data);

    const tmp_dir = testing.tmpDir(.{});
    var tmp_file = try tmp_dir.dir.createFile("test.safetensors", .{ .read = true });
    defer tmp_file.close();
    try tmp_file.writeAll(file_data);
    try tmp_file.seekTo(0);

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path = try tmp_dir.dir.realpath("test.safetensors", &path_buf);

    var st = try SafeTensors.load(allocator, tmp_path);
    defer st.deinit();

    // Test exact match is required (case sensitive)
    try testing.expect(st.hasTensor("MyTensor"));
    try testing.expect(!st.hasTensor("mytensor"));
    try testing.expect(!st.hasTensor("MYTENSOR"));
    try testing.expect(!st.hasTensor("MyTenso")); // Prefix
    try testing.expect(!st.hasTensor("MyTensorX")); // Suffix
}

test "SafeTensors.tensorCount: returns correct count with multiple tensors" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const tensor_data = [_]u8{0} ** 20;
    const header =
        \\{"weight1": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        \\ "weight2": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]},
        \\ "bias1": {"dtype": "F32", "shape": [1], "data_offsets": [8, 12]},
        \\ "bias2": {"dtype": "F32", "shape": [1], "data_offsets": [12, 16]},
        \\ "scale": {"dtype": "F32", "shape": [1], "data_offsets": [16, 20]}}
    ;

    const file_data = try createMockSafeTensorsFile(allocator, header, &tensor_data);
    defer allocator.free(file_data);

    const tmp_dir = testing.tmpDir(.{});
    var tmp_file = try tmp_dir.dir.createFile("test.safetensors", .{ .read = true });
    defer tmp_file.close();
    try tmp_file.writeAll(file_data);
    try tmp_file.seekTo(0);

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path = try tmp_dir.dir.realpath("test.safetensors", &path_buf);

    var st = try SafeTensors.load(allocator, tmp_path);
    defer st.deinit();

    try testing.expectEqual(@as(usize, 5), st.tensorCount());
}

test "SafeTensors.tensorCount: returns zero for empty file" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const header = "{}";
    const file_data = try createMockSafeTensorsFile(allocator, header, &[_]u8{});
    defer allocator.free(file_data);

    const tmp_dir = testing.tmpDir(.{});
    var tmp_file = try tmp_dir.dir.createFile("test.safetensors", .{ .read = true });
    defer tmp_file.close();
    try tmp_file.writeAll(file_data);
    try tmp_file.seekTo(0);

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path = try tmp_dir.dir.realpath("test.safetensors", &path_buf);

    var st = try SafeTensors.load(allocator, tmp_path);
    defer st.deinit();

    try testing.expectEqual(@as(usize, 0), st.tensorCount());
}

test "SafeTensors.tensorCount: matches actual loaded tensors" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Create file with mix of valid and invalid entries
    const tensor_data = [_]u8{0, 0, 128, 63, 0, 0, 0, 64};
    const header =
        \\{"valid1": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        \\ "valid2": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]},
        \\ "invalid": {"dtype": "UNKNOWN", "shape": [1], "data_offsets": [0, 4]},
        \\ "missing_shape": {"dtype": "F32", "data_offsets": [0, 4]}}
    ;

    const file_data = try createMockSafeTensorsFile(allocator, header, &tensor_data);
    defer allocator.free(file_data);

    const tmp_dir = testing.tmpDir(.{});
    var tmp_file = try tmp_dir.dir.createFile("test.safetensors", .{ .read = true });
    defer tmp_file.close();
    try tmp_file.writeAll(file_data);
    try tmp_file.seekTo(0);

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path = try tmp_dir.dir.realpath("test.safetensors", &path_buf);

    var st = try SafeTensors.load(allocator, tmp_path);
    defer st.deinit();

    // Should only count the valid tensors (invalid entries are skipped)
    try testing.expectEqual(@as(usize, 2), st.tensorCount());

    // Verify the count matches the actual loaded tensors
    const names = try st.tensorNames(allocator);
    defer allocator.free(names);
    try testing.expectEqual(st.tensorCount(), names.len);
}
