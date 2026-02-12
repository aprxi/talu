//! Sharded SafeTensors support
//!
//! Handles models split across multiple safetensors files with an index file.
//! Format: model-00001-of-00003.safetensors, model-00002-of-00003.safetensors, etc.
//! Index: model.safetensors.index.json

const std = @import("std");
const json = @import("../json/root.zig");
const reader = @import("reader.zig");
const tensor = @import("../../tensor.zig");
const dtype = @import("../../dtype.zig");

const SafeTensors = reader.SafeTensors;
const Tensor = tensor.Tensor;
const DType = dtype.DType;

pub const ShardedLoadError = error{
    InvalidIndexFile,
    MissingWeightMap,
    ShardNotFound,
    TensorNotFound,
    OutOfMemory,
    FileNotFound,
};

/// A collection of sharded SafeTensors files with unified access
pub const ShardedSafeTensors = struct {
    allocator: std.mem.Allocator,
    /// Map from tensor name to shard filename
    weight_map: std.StringHashMapUnmanaged([]const u8),
    /// Map from shard filename to loaded SafeTensors
    shards: std.StringHashMapUnmanaged(SafeTensors),
    /// Base directory containing the shard files
    base_dir: []const u8,

    /// Load sharded safetensors from an index file
    /// index_path: path to model.safetensors.index.json
    pub fn load(allocator: std.mem.Allocator, index_path: []const u8) !ShardedSafeTensors {
        // Get base directory from index path
        const base_dir_path = std.fs.path.dirname(index_path) orelse ".";
        const base_dir_storage = try allocator.dupe(u8, base_dir_path);
        errdefer allocator.free(base_dir_storage);

        // Read and parse index file
        const index_bytes = std.fs.cwd().readFileAlloc(allocator, index_path, 10 * 1024 * 1024) catch {
            return ShardedLoadError.FileNotFound;
        };
        defer allocator.free(index_bytes);

        const parsed_index = json.parseStruct(allocator, IndexJson, index_bytes, .{
            .max_size_bytes = 10 * 1024 * 1024,
            .max_value_bytes = 10 * 1024 * 1024,
            .max_string_bytes = 1 * 1024 * 1024,
            .ignore_unknown_fields = true,
        }) catch |err| {
            return switch (err) {
                error.InputTooLarge => ShardedLoadError.InvalidIndexFile,
                error.InputTooDeep => ShardedLoadError.InvalidIndexFile,
                error.StringTooLong => ShardedLoadError.InvalidIndexFile,
                error.InvalidJson => ShardedLoadError.InvalidIndexFile,
                error.OutOfMemory => ShardedLoadError.OutOfMemory,
            };
        };
        defer parsed_index.deinit();

        const index_root = parsed_index.value;
        if (index_root.weight_map == null) {
            return ShardedLoadError.MissingWeightMap;
        }

        // Build weight map
        var weight_map = std.StringHashMapUnmanaged([]const u8){};
        errdefer {
            var entry_iter = weight_map.iterator();
            while (entry_iter.next()) |kv| {
                allocator.free(kv.key_ptr.*);
                allocator.free(kv.value_ptr.*);
            }
            weight_map.deinit(allocator);
        }

        // ArrayHashMap wraps a StringArrayHashMapUnmanaged
        const weight_map_json = index_root.weight_map.?;
        const keys = weight_map_json.map.keys();
        const values = weight_map_json.map.values();
        for (keys, values) |key, value| {
            const tensor_name = try allocator.dupe(u8, key);
            errdefer allocator.free(tensor_name);
            const shard_name = try allocator.dupe(u8, value);
            try weight_map.put(allocator, tensor_name, shard_name);
        }

        return ShardedSafeTensors{
            .allocator = allocator,
            .weight_map = weight_map,
            .shards = .{},
            .base_dir = base_dir_storage,
        };
    }

    pub fn deinit(self: *ShardedSafeTensors) void {
        // Free weight map
        var weight_map_it = self.weight_map.iterator();
        while (weight_map_it.next()) |kv| {
            self.allocator.free(kv.key_ptr.*);
            self.allocator.free(kv.value_ptr.*);
        }
        self.weight_map.deinit(self.allocator);

        // Free shards
        var shard_it = self.shards.iterator();
        while (shard_it.next()) |kv| {
            self.allocator.free(kv.key_ptr.*);
            kv.value_ptr.deinit();
        }
        self.shards.deinit(self.allocator);

        self.allocator.free(self.base_dir);
        self.* = undefined;
    }

    /// Get a tensor by name, loading the appropriate shard if needed
    pub fn getTensor(self: *ShardedSafeTensors, name: []const u8, expected_dtype: ?DType) !Tensor {
        // Find which shard contains this tensor
        const shard_name = self.weight_map.get(name) orelse return error.NotFound;

        // Load shard if not already loaded
        const shard_tensor_file = try self.getOrLoadShard(shard_name);

        // Get tensor from shard
        return shard_tensor_file.getTensor(name, expected_dtype);
    }

    /// Check if a tensor exists
    pub fn hasTensor(self: *const ShardedSafeTensors, name: []const u8) bool {
        return self.weight_map.contains(name);
    }

    /// Get a list of all tensor names
    pub fn tensorNames(self: *const ShardedSafeTensors, allocator: std.mem.Allocator) ![][]const u8 {
        var tensor_names = try allocator.alloc([]const u8, self.weight_map.count());
        var name_index: usize = 0;
        var name_iter = self.weight_map.iterator();
        while (name_iter.next()) |kv| {
            tensor_names[name_index] = kv.key_ptr.*;
            name_index += 1;
        }
        return tensor_names;
    }

    /// Get number of tensors across all shards
    pub fn tensorCount(self: *const ShardedSafeTensors) usize {
        return self.weight_map.count();
    }

    /// Get total file size across all loaded shards (best effort - only counts loaded shards)
    pub fn fileSize(self: *const ShardedSafeTensors) usize {
        var total_bytes: usize = 0;
        var shard_iter = self.shards.iterator();
        while (shard_iter.next()) |kv| {
            total_bytes += kv.value_ptr.fileSize();
        }
        return total_bytes;
    }

    /// Ensure a shard is loaded, loading it if necessary
    fn getOrLoadShard(self: *ShardedSafeTensors, shard_name: []const u8) !*SafeTensors {
        // Check if already loaded
        if (self.shards.getPtr(shard_name)) |shard_file| {
            return shard_file;
        }

        // Load the shard
        const shard_file_path = try std.fs.path.join(self.allocator, &.{ self.base_dir, shard_name });
        defer self.allocator.free(shard_file_path);

        var shard_file = SafeTensors.load(self.allocator, shard_file_path) catch {
            return ShardedLoadError.ShardNotFound;
        };
        errdefer shard_file.deinit();

        const shard_name_storage = try self.allocator.dupe(u8, shard_name);
        try self.shards.put(self.allocator, shard_name_storage, shard_file);

        return self.shards.getPtr(shard_name).?;
    }

    /// Get the number of shards
    pub fn shardCount(self: *const ShardedSafeTensors) usize {
        // Count unique shard files
        var unique_shard_names = std.StringHashMapUnmanaged(void){};
        defer unique_shard_names.deinit(self.allocator);

        var name_iter = self.weight_map.iterator();
        while (name_iter.next()) |kv| {
            unique_shard_names.put(self.allocator, kv.value_ptr.*, {}) catch continue;
        }
        return unique_shard_names.count();
    }

    /// Get raw bytes for a tensor by base name + suffix (e.g., for MLX scales/biases)
    pub fn tryGetBytes(self: *ShardedSafeTensors, base: []const u8, suffix: []const u8) ?[]u8 {
        var name_buffer: [256]u8 = undefined;
        const tensor_name = std.fmt.bufPrint(&name_buffer, "{s}{s}", .{ base, suffix }) catch return null;

        // Find which shard contains this tensor
        const shard_name = self.weight_map.get(tensor_name) orelse return null;

        // Load shard if not already loaded
        const shard_file = self.getOrLoadShard(shard_name) catch return null;

        // Get entry from shard
        const tensor_entry = shard_file.entries.get(tensor_name) orelse return null;
        return @constCast(tensor_entry.data);
    }
};

/// JSON structure for model.safetensors.index.json
const IndexJson = struct {
    metadata: ?struct {
        total_size: ?i64 = null,
        total_parameters: ?i64 = null,
    } = null,
    weight_map: ?std.json.ArrayHashMap([]const u8) = null,
};

/// Detect if a model uses sharded weights
fn isShardedModel(allocator: std.mem.Allocator, model_dir: []const u8) bool {
    const index_path = std.fs.path.join(allocator, &.{ model_dir, "model.safetensors.index.json" }) catch return false;
    defer allocator.free(index_path);

    std.fs.cwd().access(index_path, .{}) catch return false;
    return true;
}

/// Get the path to the index file if it exists
fn getIndexPath(allocator: std.mem.Allocator, model_dir: []const u8) !?[]const u8 {
    const index_path = try std.fs.path.join(allocator, &.{ model_dir, "model.safetensors.index.json" });
    errdefer allocator.free(index_path);

    std.fs.cwd().access(index_path, .{}) catch {
        allocator.free(index_path);
        return null;
    };

    return index_path;
}

/// Unified interface for both single and sharded safetensors
/// This allows model loaders to work with either format transparently
pub const UnifiedSafeTensors = union(enum) {
    single: SafeTensors,
    sharded: ShardedSafeTensors,

    /// Load safetensors, automatically detecting if sharded
    pub fn load(allocator: std.mem.Allocator, path: []const u8) !UnifiedSafeTensors {
        // Check if path points to an index file or if directory contains one
        if (std.mem.endsWith(u8, path, ".index.json")) {
            // Explicit index file path
            const sharded_tensors = try ShardedSafeTensors.load(allocator, path);
            return .{ .sharded = sharded_tensors };
        }

        // Check if there's an index file in the same directory
        const model_dir = std.fs.path.dirname(path) orelse ".";
        if (try getIndexPath(allocator, model_dir)) |index_path| {
            defer allocator.free(index_path);
            const sharded_tensors = try ShardedSafeTensors.load(allocator, index_path);
            return .{ .sharded = sharded_tensors };
        }

        // Load as single file
        const single_tensors = try SafeTensors.load(allocator, path);
        return .{ .single = single_tensors };
    }

    pub fn deinit(self: *UnifiedSafeTensors) void {
        switch (self.*) {
            .single => |*s| s.deinit(),
            .sharded => |*s| s.deinit(),
        }
        self.* = undefined;
    }

    pub fn getTensor(self: *UnifiedSafeTensors, name: []const u8, expected_dtype: ?DType) !Tensor {
        return switch (self.*) {
            .single => |*s| s.getTensor(name, expected_dtype),
            .sharded => |*s| s.getTensor(name, expected_dtype),
        };
    }

    pub fn hasTensor(self: *const UnifiedSafeTensors, name: []const u8) bool {
        return switch (self.*) {
            .single => |*s| s.hasTensor(name),
            .sharded => |s| s.hasTensor(name),
        };
    }

    pub fn tensorNames(self: *const UnifiedSafeTensors, allocator: std.mem.Allocator) ![][]const u8 {
        return switch (self.*) {
            .single => |*s| s.tensorNames(allocator),
            .sharded => |s| s.tensorNames(allocator),
        };
    }

    /// Get raw bytes for a tensor by base name + suffix (e.g., for MLX scales/biases)
    pub fn tryGetBytes(self: *UnifiedSafeTensors, base: []const u8, suffix: []const u8) ?[]u8 {
        return switch (self.*) {
            .single => |*s| reader.tryGetBytes(s, base, suffix),
            .sharded => |*s| s.tryGetBytes(base, suffix),
        };
    }

    /// Get total file size in bytes
    pub fn fileSize(self: *const UnifiedSafeTensors) usize {
        return switch (self.*) {
            .single => |*s| s.fileSize(),
            .sharded => |*s| s.fileSize(),
        };
    }

    /// Get total number of tensors
    pub fn tensorCount(self: *const UnifiedSafeTensors) usize {
        return switch (self.*) {
            .single => |*s| s.tensorCount(),
            .sharded => |*s| s.tensorCount(),
        };
    }
};

// Tests
test "isShardedModel returns false for non-sharded" {
    const allocator = std.testing.allocator;
    try std.testing.expect(!isShardedModel(allocator, "/nonexistent/path"));
}

test "ShardedSafeTensors: load from index file" {
    const allocator = std.testing.allocator;
    const writer = @import("writer.zig");

    // Create temporary directory structure
    const tmp_dir_path = "/tmp/test_sharded_safetensors";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    // Create shard 1
    const shard1_data = [_]u8{ 1, 2, 3, 4 };
    const shard1_entries = [_]writer.TensorEntry{
        .{
            .name = "layer1.weight",
            .dtype = .i8,
            .shape = &[_]usize{4},
            .data = &shard1_data,
        },
    };
    const shard1_path = tmp_dir_path ++ "/model-00001-of-00002.safetensors";
    try writer.write(allocator, shard1_path, &shard1_entries);

    // Create shard 2
    const shard2_data = [_]u8{ 5, 6, 7, 8 };
    const shard2_entries = [_]writer.TensorEntry{
        .{
            .name = "layer2.weight",
            .dtype = .i8,
            .shape = &[_]usize{4},
            .data = &shard2_data,
        },
    };
    const shard2_path = tmp_dir_path ++ "/model-00002-of-00002.safetensors";
    try writer.write(allocator, shard2_path, &shard2_entries);

    // Create index file
    const index_json =
        \\{
        \\  "metadata": {"total_size": 8},
        \\  "weight_map": {
        \\    "layer1.weight": "model-00001-of-00002.safetensors",
        \\    "layer2.weight": "model-00002-of-00002.safetensors"
        \\  }
        \\}
    ;
    const index_path = tmp_dir_path ++ "/model.safetensors.index.json";
    var index_file = try std.fs.cwd().createFile(index_path, .{});
    defer index_file.close();
    try index_file.writeAll(index_json);

    // Load sharded safetensors
    var sharded = try ShardedSafeTensors.load(allocator, index_path);
    defer sharded.deinit();

    // Verify tensor count and names
    try std.testing.expectEqual(@as(usize, 2), sharded.tensorCount());
    try std.testing.expect(sharded.hasTensor("layer1.weight"));
    try std.testing.expect(sharded.hasTensor("layer2.weight"));
    try std.testing.expect(!sharded.hasTensor("nonexistent"));

    // Get tensors
    const t1 = try sharded.getTensor("layer1.weight", .i8);
    try std.testing.expectEqual(@as(usize, 4), t1.numel);

    const t2 = try sharded.getTensor("layer2.weight", .i8);
    try std.testing.expectEqual(@as(usize, 4), t2.numel);
}

test "ShardedSafeTensors: getTensor with dtype validation" {
    const allocator = std.testing.allocator;
    const writer = @import("writer.zig");

    const tmp_dir_path = "/tmp/test_sharded_dtype";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    // Create shard with F32 data
    const shard_data = [_]u8{ 0, 0, 128, 63 }; // 1.0 in F32
    const shard_entries = [_]writer.TensorEntry{
        .{
            .name = "weight",
            .dtype = .f32,
            .shape = &[_]usize{1},
            .data = &shard_data,
        },
    };
    const shard_path = tmp_dir_path ++ "/model.safetensors";
    try writer.write(allocator, shard_path, &shard_entries);

    const index_json =
        \\{
        \\  "weight_map": {
        \\    "weight": "model.safetensors"
        \\  }
        \\}
    ;
    const index_path = tmp_dir_path ++ "/model.safetensors.index.json";
    var index_file = try std.fs.cwd().createFile(index_path, .{});
    defer index_file.close();
    try index_file.writeAll(index_json);

    var sharded = try ShardedSafeTensors.load(allocator, index_path);
    defer sharded.deinit();

    // Should succeed with correct dtype
    _ = try sharded.getTensor("weight", .f32);

    // Should succeed with null (no dtype check)
    _ = try sharded.getTensor("weight", null);

    // Should fail with wrong dtype
    try std.testing.expectError(error.UnexpectedDType, sharded.getTensor("weight", .i8));
}

test "ShardedSafeTensors: tensorNames listing" {
    const allocator = std.testing.allocator;
    const writer = @import("writer.zig");

    const tmp_dir_path = "/tmp/test_sharded_names";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    // Create shard
    const shard_data = [_]u8{ 1, 2, 3, 4, 5, 6 };
    const shard_entries = [_]writer.TensorEntry{
        .{ .name = "w1", .dtype = .i8, .shape = &[_]usize{2}, .data = shard_data[0..2] },
        .{ .name = "w2", .dtype = .i8, .shape = &[_]usize{2}, .data = shard_data[2..4] },
        .{ .name = "b1", .dtype = .i8, .shape = &[_]usize{2}, .data = shard_data[4..6] },
    };
    const shard_path = tmp_dir_path ++ "/model.safetensors";
    try writer.write(allocator, shard_path, &shard_entries);

    const index_json =
        \\{
        \\  "weight_map": {
        \\    "w1": "model.safetensors",
        \\    "w2": "model.safetensors",
        \\    "b1": "model.safetensors"
        \\  }
        \\}
    ;
    const index_path = tmp_dir_path ++ "/model.safetensors.index.json";
    var index_file = try std.fs.cwd().createFile(index_path, .{});
    defer index_file.close();
    try index_file.writeAll(index_json);

    var sharded = try ShardedSafeTensors.load(allocator, index_path);
    defer sharded.deinit();

    const names = try sharded.tensorNames(allocator);
    defer allocator.free(names);

    try std.testing.expectEqual(@as(usize, 3), names.len);

    // Check that all expected names are present (order may vary)
    var found_w1 = false;
    var found_w2 = false;
    var found_b1 = false;

    for (names) |name| {
        if (std.mem.eql(u8, name, "w1")) found_w1 = true;
        if (std.mem.eql(u8, name, "w2")) found_w2 = true;
        if (std.mem.eql(u8, name, "b1")) found_b1 = true;
    }

    try std.testing.expect(found_w1);
    try std.testing.expect(found_w2);
    try std.testing.expect(found_b1);
}

test "ShardedSafeTensors: shardCount calculation" {
    const allocator = std.testing.allocator;
    const writer = @import("writer.zig");

    const tmp_dir_path = "/tmp/test_sharded_count";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    // Create 3 shards
    for (0..3) |shard_idx| {
        const shard_data = [_]u8{@intCast(shard_idx)};
        var name_buf: [32]u8 = undefined;
        const tensor_name = try std.fmt.bufPrint(&name_buf, "tensor{d}", .{shard_idx});
        const shard_entries = [_]writer.TensorEntry{
            .{
                .name = tensor_name,
                .dtype = .i8,
                .shape = &[_]usize{1},
                .data = &shard_data,
            },
        };
        var path_buf: [256]u8 = undefined;
        const shard_path = try std.fmt.bufPrint(&path_buf, "{s}/model-{d:0>5}-of-00003.safetensors", .{ tmp_dir_path, shard_idx + 1 });
        try writer.write(allocator, shard_path, &shard_entries);
    }

    const index_json =
        \\{
        \\  "weight_map": {
        \\    "tensor0": "model-00001-of-00003.safetensors",
        \\    "tensor1": "model-00002-of-00003.safetensors",
        \\    "tensor2": "model-00003-of-00003.safetensors"
        \\  }
        \\}
    ;
    const index_path = tmp_dir_path ++ "/model.safetensors.index.json";
    var index_file = try std.fs.cwd().createFile(index_path, .{});
    defer index_file.close();
    try index_file.writeAll(index_json);

    var sharded = try ShardedSafeTensors.load(allocator, index_path);
    defer sharded.deinit();

    try std.testing.expectEqual(@as(usize, 3), sharded.shardCount());
}

test "ShardedSafeTensors: tryGetBytes helper" {
    const allocator = std.testing.allocator;
    const writer = @import("writer.zig");

    const tmp_dir_path = "/tmp/test_sharded_getbytes";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    // Create shard
    const shard_data = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const shard_entries = [_]writer.TensorEntry{
        .{ .name = "model.weight", .dtype = .i8, .shape = &[_]usize{4}, .data = shard_data[0..4] },
        .{ .name = "model.bias", .dtype = .i8, .shape = &[_]usize{4}, .data = shard_data[4..8] },
    };
    const shard_path = tmp_dir_path ++ "/model.safetensors";
    try writer.write(allocator, shard_path, &shard_entries);

    const index_json =
        \\{
        \\  "weight_map": {
        \\    "model.weight": "model.safetensors",
        \\    "model.bias": "model.safetensors"
        \\  }
        \\}
    ;
    const index_path = tmp_dir_path ++ "/model.safetensors.index.json";
    var index_file = try std.fs.cwd().createFile(index_path, .{});
    defer index_file.close();
    try index_file.writeAll(index_json);

    var sharded = try ShardedSafeTensors.load(allocator, index_path);
    defer sharded.deinit();

    // Test successful lookup
    const weight_data = sharded.tryGetBytes("model.", "weight");
    try std.testing.expect(weight_data != null);
    try std.testing.expectEqual(@as(usize, 4), weight_data.?.len);

    const bias_data = sharded.tryGetBytes("model.", "bias");
    try std.testing.expect(bias_data != null);
    try std.testing.expectEqual(@as(usize, 4), bias_data.?.len);

    // Test failed lookup
    const missing_data = sharded.tryGetBytes("model.", "missing");
    try std.testing.expect(missing_data == null);
}

test "ShardedSafeTensors: getTensor for nonexistent tensor" {
    const allocator = std.testing.allocator;
    const writer = @import("writer.zig");

    const tmp_dir_path = "/tmp/test_sharded_notfound";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    const shard_data = [_]u8{1};
    const shard_entries = [_]writer.TensorEntry{
        .{ .name = "exists", .dtype = .i8, .shape = &[_]usize{1}, .data = &shard_data },
    };
    const shard_path = tmp_dir_path ++ "/model.safetensors";
    try writer.write(allocator, shard_path, &shard_entries);

    const index_json =
        \\{
        \\  "weight_map": {
        \\    "exists": "model.safetensors"
        \\  }
        \\}
    ;
    const index_path = tmp_dir_path ++ "/model.safetensors.index.json";
    var index_file = try std.fs.cwd().createFile(index_path, .{});
    defer index_file.close();
    try index_file.writeAll(index_json);

    var sharded = try ShardedSafeTensors.load(allocator, index_path);
    defer sharded.deinit();

    try std.testing.expectError(error.NotFound, sharded.getTensor("nonexistent", null));
}

test "ShardedSafeTensors: fileSize reporting" {
    const allocator = std.testing.allocator;
    const writer = @import("writer.zig");

    const tmp_dir_path = "/tmp/test_sharded_filesize";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    const shard_data = [_]u8{1};
    const shard_entries = [_]writer.TensorEntry{
        .{ .name = "weight", .dtype = .i8, .shape = &[_]usize{1}, .data = &shard_data },
    };
    const shard_path = tmp_dir_path ++ "/model.safetensors";
    try writer.write(allocator, shard_path, &shard_entries);

    const index_json =
        \\{
        \\  "weight_map": {
        \\    "weight": "model.safetensors"
        \\  }
        \\}
    ;
    const index_path = tmp_dir_path ++ "/model.safetensors.index.json";
    var index_file = try std.fs.cwd().createFile(index_path, .{});
    defer index_file.close();
    try index_file.writeAll(index_json);

    var sharded = try ShardedSafeTensors.load(allocator, index_path);
    defer sharded.deinit();

    // fileSize should be 0 before loading any shards
    try std.testing.expectEqual(@as(usize, 0), sharded.fileSize());

    // Load a tensor (which loads the shard)
    _ = try sharded.getTensor("weight", null);

    // Now fileSize should be non-zero
    try std.testing.expect(sharded.fileSize() > 0);
}

test "ShardedSafeTensors: invalid index file - missing weight_map" {
    const allocator = std.testing.allocator;

    const tmp_dir_path = "/tmp/test_sharded_invalid_nomap";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    const index_json = "{}";
    const index_path = tmp_dir_path ++ "/model.safetensors.index.json";
    var index_file = try std.fs.cwd().createFile(index_path, .{});
    defer index_file.close();
    try index_file.writeAll(index_json);

    try std.testing.expectError(ShardedLoadError.MissingWeightMap, ShardedSafeTensors.load(allocator, index_path));
}

test "ShardedSafeTensors: invalid index file - malformed JSON" {
    const allocator = std.testing.allocator;

    const tmp_dir_path = "/tmp/test_sharded_invalid_json";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    const index_json = "not valid json";
    const index_path = tmp_dir_path ++ "/model.safetensors.index.json";
    var index_file = try std.fs.cwd().createFile(index_path, .{});
    defer index_file.close();
    try index_file.writeAll(index_json);

    try std.testing.expectError(ShardedLoadError.InvalidIndexFile, ShardedSafeTensors.load(allocator, index_path));
}

test "ShardedSafeTensors: nonexistent index file" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(ShardedLoadError.FileNotFound, ShardedSafeTensors.load(allocator, "/tmp/nonexistent_index.json"));
}

test "ShardedSafeTensors: shard file not found" {
    const allocator = std.testing.allocator;

    const tmp_dir_path = "/tmp/test_sharded_missing_shard";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    const index_json =
        \\{
        \\  "weight_map": {
        \\    "weight": "nonexistent.safetensors"
        \\  }
        \\}
    ;
    const index_path = tmp_dir_path ++ "/model.safetensors.index.json";
    var index_file = try std.fs.cwd().createFile(index_path, .{});
    defer index_file.close();
    try index_file.writeAll(index_json);

    var sharded = try ShardedSafeTensors.load(allocator, index_path);
    defer sharded.deinit();

    // Should fail when trying to load the missing shard
    try std.testing.expectError(ShardedLoadError.ShardNotFound, sharded.getTensor("weight", null));
}

test "getIndexPath: finds index file in directory" {
    const allocator = std.testing.allocator;

    const tmp_dir_path = "/tmp/test_getindexpath";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    // Create index file
    const index_path = tmp_dir_path ++ "/model.safetensors.index.json";
    var index_file = try std.fs.cwd().createFile(index_path, .{});
    index_file.close();

    const result = try getIndexPath(allocator, tmp_dir_path);
    try std.testing.expect(result != null);
    defer allocator.free(result.?);

    try std.testing.expect(std.mem.endsWith(u8, result.?, "model.safetensors.index.json"));
}

test "getIndexPath: returns null when no index file exists" {
    const allocator = std.testing.allocator;

    const tmp_dir_path = "/tmp/test_getindexpath_nofile";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    const result = try getIndexPath(allocator, tmp_dir_path);
    try std.testing.expect(result == null);
}

test "UnifiedSafeTensors: load sharded via index path" {
    const allocator = std.testing.allocator;
    const writer = @import("writer.zig");

    const tmp_dir_path = "/tmp/test_unified_sharded";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    // Create shard
    const shard_data = [_]u8{ 1, 2, 3, 4 };
    const shard_entries = [_]writer.TensorEntry{
        .{ .name = "weight", .dtype = .i8, .shape = &[_]usize{4}, .data = &shard_data },
    };
    const shard_path = tmp_dir_path ++ "/model.safetensors";
    try writer.write(allocator, shard_path, &shard_entries);

    const index_json =
        \\{
        \\  "weight_map": {
        \\    "weight": "model.safetensors"
        \\  }
        \\}
    ;
    const index_path = tmp_dir_path ++ "/model.safetensors.index.json";
    var index_file = try std.fs.cwd().createFile(index_path, .{});
    defer index_file.close();
    try index_file.writeAll(index_json);

    // Load via explicit index path
    var unified = try UnifiedSafeTensors.load(allocator, index_path);
    defer unified.deinit();

    try std.testing.expect(unified.hasTensor("weight"));
    try std.testing.expectEqual(@as(usize, 1), unified.tensorCount());
}

test "UnifiedSafeTensors: load single file" {
    const allocator = std.testing.allocator;
    const writer = @import("writer.zig");

    const tmp_dir_path = "/tmp/test_unified_single";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    const shard_data = [_]u8{ 1, 2, 3, 4 };
    const shard_entries = [_]writer.TensorEntry{
        .{ .name = "weight", .dtype = .i8, .shape = &[_]usize{4}, .data = &shard_data },
    };
    const model_path = tmp_dir_path ++ "/model.safetensors";
    try writer.write(allocator, model_path, &shard_entries);

    var unified = try UnifiedSafeTensors.load(allocator, model_path);
    defer unified.deinit();

    try std.testing.expect(unified.hasTensor("weight"));
    try std.testing.expectEqual(@as(usize, 1), unified.tensorCount());
}

test "UnifiedSafeTensors: auto-detect sharded in same directory" {
    const allocator = std.testing.allocator;
    const writer = @import("writer.zig");

    const tmp_dir_path = "/tmp/test_unified_autodetect";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    // Create shard file
    const shard_data = [_]u8{ 1, 2, 3, 4 };
    const shard_entries = [_]writer.TensorEntry{
        .{ .name = "weight", .dtype = .i8, .shape = &[_]usize{4}, .data = &shard_data },
    };
    const shard_path = tmp_dir_path ++ "/model-00001-of-00001.safetensors";
    try writer.write(allocator, shard_path, &shard_entries);

    // Create index file
    const index_json =
        \\{
        \\  "weight_map": {
        \\    "weight": "model-00001-of-00001.safetensors"
        \\  }
        \\}
    ;
    const index_path = tmp_dir_path ++ "/model.safetensors.index.json";
    var index_file = try std.fs.cwd().createFile(index_path, .{});
    defer index_file.close();
    try index_file.writeAll(index_json);

    // Load via shard path (should auto-detect index)
    var unified = try UnifiedSafeTensors.load(allocator, shard_path);
    defer unified.deinit();

    try std.testing.expect(unified.hasTensor("weight"));
}

test "UnifiedSafeTensors: getTensor from both modes" {
    const allocator = std.testing.allocator;
    const writer = @import("writer.zig");

    const tmp_dir_path = "/tmp/test_unified_gettensor";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    const test_data = [_]u8{ 0, 0, 128, 63 }; // 1.0 in F32

    // Test single file mode
    {
        const entries = [_]writer.TensorEntry{
            .{ .name = "weight", .dtype = .f32, .shape = &[_]usize{1}, .data = &test_data },
        };
        const single_path = tmp_dir_path ++ "/single.safetensors";
        try writer.write(allocator, single_path, &entries);

        var unified = try UnifiedSafeTensors.load(allocator, single_path);
        defer unified.deinit();

        const t = try unified.getTensor("weight", .f32);
        try std.testing.expectEqual(DType.f32, t.dtype);
    }

    // Test sharded mode
    {
        const shard_path = tmp_dir_path ++ "/sharded.safetensors";
        const entries = [_]writer.TensorEntry{
            .{ .name = "weight", .dtype = .f32, .shape = &[_]usize{1}, .data = &test_data },
        };
        try writer.write(allocator, shard_path, &entries);

        const index_json =
            \\{
            \\  "weight_map": {
            \\    "weight": "sharded.safetensors"
            \\  }
            \\}
        ;
        const index_path = tmp_dir_path ++ "/sharded.safetensors.index.json";
        var index_file = try std.fs.cwd().createFile(index_path, .{});
        defer index_file.close();
        try index_file.writeAll(index_json);

        var unified = try UnifiedSafeTensors.load(allocator, index_path);
        defer unified.deinit();

        const t = try unified.getTensor("weight", .f32);
        try std.testing.expectEqual(DType.f32, t.dtype);
    }
}

test "UnifiedSafeTensors: tryGetBytes from both modes" {
    const allocator = std.testing.allocator;
    const writer = @import("writer.zig");

    const tmp_dir_path = "/tmp/test_unified_getbytes";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    const test_data = [_]u8{ 1, 2, 3, 4 };

    // Test single file mode
    {
        const entries = [_]writer.TensorEntry{
            .{ .name = "model.weight", .dtype = .i8, .shape = &[_]usize{4}, .data = &test_data },
        };
        const single_path = tmp_dir_path ++ "/single2.safetensors";
        try writer.write(allocator, single_path, &entries);

        var unified = try UnifiedSafeTensors.load(allocator, single_path);
        defer unified.deinit();

        const bytes = unified.tryGetBytes("model.", "weight");
        try std.testing.expect(bytes != null);
        try std.testing.expectEqual(@as(usize, 4), bytes.?.len);
    }

    // Test sharded mode
    {
        const shard_path = tmp_dir_path ++ "/sharded2.safetensors";
        const entries = [_]writer.TensorEntry{
            .{ .name = "model.weight", .dtype = .i8, .shape = &[_]usize{4}, .data = &test_data },
        };
        try writer.write(allocator, shard_path, &entries);

        const index_json =
            \\{
            \\  "weight_map": {
            \\    "model.weight": "sharded2.safetensors"
            \\  }
            \\}
        ;
        const index_path = tmp_dir_path ++ "/sharded2.safetensors.index.json";
        var index_file = try std.fs.cwd().createFile(index_path, .{});
        defer index_file.close();
        try index_file.writeAll(index_json);

        var unified = try UnifiedSafeTensors.load(allocator, index_path);
        defer unified.deinit();

        const bytes = unified.tryGetBytes("model.", "weight");
        try std.testing.expect(bytes != null);
        try std.testing.expectEqual(@as(usize, 4), bytes.?.len);
    }
}

// Tests for ShardedSafeTensors.deinit
test "ShardedSafeTensors.deinit: cleans up weight_map and base_dir" {
    const allocator = std.testing.allocator;
    const writer = @import("writer.zig");

    const tmp_dir_path = "/tmp/test_sharded_deinit_basic";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    const shard_data = [_]u8{ 1, 2, 3, 4 };
    const shard_entries = [_]writer.TensorEntry{
        .{ .name = "weight", .dtype = .i8, .shape = &[_]usize{4}, .data = &shard_data },
    };
    const shard_path = tmp_dir_path ++ "/model.safetensors";
    try writer.write(allocator, shard_path, &shard_entries);

    const index_json =
        \\{
        \\  "weight_map": {
        \\    "weight": "model.safetensors"
        \\  }
        \\}
    ;
    const index_path = tmp_dir_path ++ "/model.safetensors.index.json";
    var index_file = try std.fs.cwd().createFile(index_path, .{});
    defer index_file.close();
    try index_file.writeAll(index_json);

    var sharded = try ShardedSafeTensors.load(allocator, index_path);
    try std.testing.expectEqual(@as(usize, 1), sharded.weight_map.count());
    try std.testing.expect(sharded.base_dir.len > 0);

    sharded.deinit();
    // After deinit, the struct is marked undefined, so we can't access fields.
    // The test verifies no memory leaks via the allocator.
}

test "ShardedSafeTensors.deinit: cleans up loaded shards" {
    const allocator = std.testing.allocator;
    const writer = @import("writer.zig");

    const tmp_dir_path = "/tmp/test_sharded_deinit_shards";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    // Create two shards
    const shard1_data = [_]u8{ 1, 2, 3, 4 };
    const shard1_entries = [_]writer.TensorEntry{
        .{ .name = "layer1.weight", .dtype = .i8, .shape = &[_]usize{4}, .data = &shard1_data },
    };
    const shard1_path = tmp_dir_path ++ "/model-00001-of-00002.safetensors";
    try writer.write(allocator, shard1_path, &shard1_entries);

    const shard2_data = [_]u8{ 5, 6, 7, 8 };
    const shard2_entries = [_]writer.TensorEntry{
        .{ .name = "layer2.weight", .dtype = .i8, .shape = &[_]usize{4}, .data = &shard2_data },
    };
    const shard2_path = tmp_dir_path ++ "/model-00002-of-00002.safetensors";
    try writer.write(allocator, shard2_path, &shard2_entries);

    const index_json =
        \\{
        \\  "weight_map": {
        \\    "layer1.weight": "model-00001-of-00002.safetensors",
        \\    "layer2.weight": "model-00002-of-00002.safetensors"
        \\  }
        \\}
    ;
    const index_path = tmp_dir_path ++ "/model.safetensors.index.json";
    var index_file = try std.fs.cwd().createFile(index_path, .{});
    defer index_file.close();
    try index_file.writeAll(index_json);

    var sharded = try ShardedSafeTensors.load(allocator, index_path);

    // Load both shards
    _ = try sharded.getTensor("layer1.weight", null);
    _ = try sharded.getTensor("layer2.weight", null);

    // Verify shards are loaded
    try std.testing.expectEqual(@as(usize, 2), sharded.shards.count());

    sharded.deinit();
    // The test verifies no memory leaks via the allocator.
}

test "ShardedSafeTensors.deinit: handles empty weight_map" {
    const allocator = std.testing.allocator;

    const tmp_dir_path = "/tmp/test_sharded_deinit_empty";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    const index_json =
        \\{
        \\  "weight_map": {}
        \\}
    ;
    const index_path = tmp_dir_path ++ "/model.safetensors.index.json";
    var index_file = try std.fs.cwd().createFile(index_path, .{});
    defer index_file.close();
    try index_file.writeAll(index_json);

    var sharded = try ShardedSafeTensors.load(allocator, index_path);
    try std.testing.expectEqual(@as(usize, 0), sharded.weight_map.count());

    sharded.deinit();
    // The test verifies no memory leaks via the allocator.
}

// Tests for ShardedSafeTensors.hasTensor
test "ShardedSafeTensors.hasTensor: returns true for existing tensor" {
    const allocator = std.testing.allocator;
    const writer = @import("writer.zig");

    const tmp_dir_path = "/tmp/test_sharded_hastensor_exists";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    const shard_data = [_]u8{ 1, 2, 3, 4 };
    const shard_entries = [_]writer.TensorEntry{
        .{ .name = "model.weight", .dtype = .i8, .shape = &[_]usize{4}, .data = &shard_data },
    };
    const shard_path = tmp_dir_path ++ "/model.safetensors";
    try writer.write(allocator, shard_path, &shard_entries);

    const index_json =
        \\{
        \\  "weight_map": {
        \\    "model.weight": "model.safetensors"
        \\  }
        \\}
    ;
    const index_path = tmp_dir_path ++ "/model.safetensors.index.json";
    var index_file = try std.fs.cwd().createFile(index_path, .{});
    defer index_file.close();
    try index_file.writeAll(index_json);

    var sharded = try ShardedSafeTensors.load(allocator, index_path);
    defer sharded.deinit();

    try std.testing.expect(sharded.hasTensor("model.weight"));
}

test "ShardedSafeTensors.hasTensor: returns false for nonexistent tensor" {
    const allocator = std.testing.allocator;
    const writer = @import("writer.zig");

    const tmp_dir_path = "/tmp/test_sharded_hastensor_notexists";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    const shard_data = [_]u8{ 1, 2, 3, 4 };
    const shard_entries = [_]writer.TensorEntry{
        .{ .name = "model.weight", .dtype = .i8, .shape = &[_]usize{4}, .data = &shard_data },
    };
    const shard_path = tmp_dir_path ++ "/model.safetensors";
    try writer.write(allocator, shard_path, &shard_entries);

    const index_json =
        \\{
        \\  "weight_map": {
        \\    "model.weight": "model.safetensors"
        \\  }
        \\}
    ;
    const index_path = tmp_dir_path ++ "/model.safetensors.index.json";
    var index_file = try std.fs.cwd().createFile(index_path, .{});
    defer index_file.close();
    try index_file.writeAll(index_json);

    var sharded = try ShardedSafeTensors.load(allocator, index_path);
    defer sharded.deinit();

    try std.testing.expect(!sharded.hasTensor("nonexistent.tensor"));
    try std.testing.expect(!sharded.hasTensor("model.bias"));
    try std.testing.expect(!sharded.hasTensor(""));
}

test "ShardedSafeTensors.hasTensor: handles multiple tensors" {
    const allocator = std.testing.allocator;
    const writer = @import("writer.zig");

    const tmp_dir_path = "/tmp/test_sharded_hastensor_multiple";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    const shard_data = [_]u8{ 1, 2, 3, 4, 5, 6 };
    const shard_entries = [_]writer.TensorEntry{
        .{ .name = "weight", .dtype = .i8, .shape = &[_]usize{2}, .data = shard_data[0..2] },
        .{ .name = "bias", .dtype = .i8, .shape = &[_]usize{2}, .data = shard_data[2..4] },
        .{ .name = "scale", .dtype = .i8, .shape = &[_]usize{2}, .data = shard_data[4..6] },
    };
    const shard_path = tmp_dir_path ++ "/model.safetensors";
    try writer.write(allocator, shard_path, &shard_entries);

    const index_json =
        \\{
        \\  "weight_map": {
        \\    "weight": "model.safetensors",
        \\    "bias": "model.safetensors",
        \\    "scale": "model.safetensors"
        \\  }
        \\}
    ;
    const index_path = tmp_dir_path ++ "/model.safetensors.index.json";
    var index_file = try std.fs.cwd().createFile(index_path, .{});
    defer index_file.close();
    try index_file.writeAll(index_json);

    var sharded = try ShardedSafeTensors.load(allocator, index_path);
    defer sharded.deinit();

    try std.testing.expect(sharded.hasTensor("weight"));
    try std.testing.expect(sharded.hasTensor("bias"));
    try std.testing.expect(sharded.hasTensor("scale"));
    try std.testing.expect(!sharded.hasTensor("other"));
}

// Tests for ShardedSafeTensors.tensorCount
test "ShardedSafeTensors.tensorCount: returns correct count for single tensor" {
    const allocator = std.testing.allocator;
    const writer = @import("writer.zig");

    const tmp_dir_path = "/tmp/test_sharded_tensorcount_single";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    const shard_data = [_]u8{ 1, 2, 3, 4 };
    const shard_entries = [_]writer.TensorEntry{
        .{ .name = "weight", .dtype = .i8, .shape = &[_]usize{4}, .data = &shard_data },
    };
    const shard_path = tmp_dir_path ++ "/model.safetensors";
    try writer.write(allocator, shard_path, &shard_entries);

    const index_json =
        \\{
        \\  "weight_map": {
        \\    "weight": "model.safetensors"
        \\  }
        \\}
    ;
    const index_path = tmp_dir_path ++ "/model.safetensors.index.json";
    var index_file = try std.fs.cwd().createFile(index_path, .{});
    defer index_file.close();
    try index_file.writeAll(index_json);

    var sharded = try ShardedSafeTensors.load(allocator, index_path);
    defer sharded.deinit();

    try std.testing.expectEqual(@as(usize, 1), sharded.tensorCount());
}

test "ShardedSafeTensors.tensorCount: returns correct count for multiple tensors" {
    const allocator = std.testing.allocator;
    const writer = @import("writer.zig");

    const tmp_dir_path = "/tmp/test_sharded_tensorcount_multiple";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    const shard_data = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const shard_entries = [_]writer.TensorEntry{
        .{ .name = "w1", .dtype = .i8, .shape = &[_]usize{2}, .data = shard_data[0..2] },
        .{ .name = "w2", .dtype = .i8, .shape = &[_]usize{2}, .data = shard_data[2..4] },
        .{ .name = "w3", .dtype = .i8, .shape = &[_]usize{2}, .data = shard_data[4..6] },
        .{ .name = "w4", .dtype = .i8, .shape = &[_]usize{2}, .data = shard_data[6..8] },
    };
    const shard_path = tmp_dir_path ++ "/model.safetensors";
    try writer.write(allocator, shard_path, &shard_entries);

    const index_json =
        \\{
        \\  "weight_map": {
        \\    "w1": "model.safetensors",
        \\    "w2": "model.safetensors",
        \\    "w3": "model.safetensors",
        \\    "w4": "model.safetensors"
        \\  }
        \\}
    ;
    const index_path = tmp_dir_path ++ "/model.safetensors.index.json";
    var index_file = try std.fs.cwd().createFile(index_path, .{});
    defer index_file.close();
    try index_file.writeAll(index_json);

    var sharded = try ShardedSafeTensors.load(allocator, index_path);
    defer sharded.deinit();

    try std.testing.expectEqual(@as(usize, 4), sharded.tensorCount());
}

test "ShardedSafeTensors.tensorCount: returns zero for empty weight_map" {
    const allocator = std.testing.allocator;

    const tmp_dir_path = "/tmp/test_sharded_tensorcount_zero";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    const index_json =
        \\{
        \\  "weight_map": {}
        \\}
    ;
    const index_path = tmp_dir_path ++ "/model.safetensors.index.json";
    var index_file = try std.fs.cwd().createFile(index_path, .{});
    defer index_file.close();
    try index_file.writeAll(index_json);

    var sharded = try ShardedSafeTensors.load(allocator, index_path);
    defer sharded.deinit();

    try std.testing.expectEqual(@as(usize, 0), sharded.tensorCount());
}

test "ShardedSafeTensors.tensorCount: count across multiple shards" {
    const allocator = std.testing.allocator;
    const writer = @import("writer.zig");

    const tmp_dir_path = "/tmp/test_sharded_tensorcount_shards";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    // Create shard 1 with 2 tensors
    const shard1_data = [_]u8{ 1, 2, 3, 4 };
    const shard1_entries = [_]writer.TensorEntry{
        .{ .name = "layer1.weight", .dtype = .i8, .shape = &[_]usize{2}, .data = shard1_data[0..2] },
        .{ .name = "layer1.bias", .dtype = .i8, .shape = &[_]usize{2}, .data = shard1_data[2..4] },
    };
    const shard1_path = tmp_dir_path ++ "/model-00001-of-00002.safetensors";
    try writer.write(allocator, shard1_path, &shard1_entries);

    // Create shard 2 with 3 tensors
    const shard2_data = [_]u8{ 5, 6, 7, 8, 9, 10 };
    const shard2_entries = [_]writer.TensorEntry{
        .{ .name = "layer2.weight", .dtype = .i8, .shape = &[_]usize{2}, .data = shard2_data[0..2] },
        .{ .name = "layer2.bias", .dtype = .i8, .shape = &[_]usize{2}, .data = shard2_data[2..4] },
        .{ .name = "layer2.scale", .dtype = .i8, .shape = &[_]usize{2}, .data = shard2_data[4..6] },
    };
    const shard2_path = tmp_dir_path ++ "/model-00002-of-00002.safetensors";
    try writer.write(allocator, shard2_path, &shard2_entries);

    const index_json =
        \\{
        \\  "weight_map": {
        \\    "layer1.weight": "model-00001-of-00002.safetensors",
        \\    "layer1.bias": "model-00001-of-00002.safetensors",
        \\    "layer2.weight": "model-00002-of-00002.safetensors",
        \\    "layer2.bias": "model-00002-of-00002.safetensors",
        \\    "layer2.scale": "model-00002-of-00002.safetensors"
        \\  }
        \\}
    ;
    const index_path = tmp_dir_path ++ "/model.safetensors.index.json";
    var index_file = try std.fs.cwd().createFile(index_path, .{});
    defer index_file.close();
    try index_file.writeAll(index_json);

    var sharded = try ShardedSafeTensors.load(allocator, index_path);
    defer sharded.deinit();

    // Total: 2 tensors from shard 1 + 3 tensors from shard 2 = 5 tensors
    try std.testing.expectEqual(@as(usize, 5), sharded.tensorCount());
}

// Tests for UnifiedSafeTensors.deinit
test "UnifiedSafeTensors.deinit: single mode cleanup" {
    const allocator = std.testing.allocator;
    const writer = @import("writer.zig");

    const tmp_dir_path = "/tmp/test_unified_deinit_single";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    const shard_data = [_]u8{ 1, 2, 3, 4 };
    const shard_entries = [_]writer.TensorEntry{
        .{ .name = "weight", .dtype = .i8, .shape = &[_]usize{4}, .data = &shard_data },
    };
    const model_path = tmp_dir_path ++ "/model.safetensors";
    try writer.write(allocator, model_path, &shard_entries);

    var unified = try UnifiedSafeTensors.load(allocator, model_path);
    unified.deinit();
    // The test verifies no memory leaks via the allocator.
}

test "UnifiedSafeTensors.deinit: sharded mode cleanup" {
    const allocator = std.testing.allocator;
    const writer = @import("writer.zig");

    const tmp_dir_path = "/tmp/test_unified_deinit_sharded";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    const shard_data = [_]u8{ 1, 2, 3, 4 };
    const shard_entries = [_]writer.TensorEntry{
        .{ .name = "weight", .dtype = .i8, .shape = &[_]usize{4}, .data = &shard_data },
    };
    const shard_path = tmp_dir_path ++ "/model.safetensors";
    try writer.write(allocator, shard_path, &shard_entries);

    const index_json =
        \\{
        \\  "weight_map": {
        \\    "weight": "model.safetensors"
        \\  }
        \\}
    ;
    const index_path = tmp_dir_path ++ "/model.safetensors.index.json";
    var index_file = try std.fs.cwd().createFile(index_path, .{});
    defer index_file.close();
    try index_file.writeAll(index_json);

    var unified = try UnifiedSafeTensors.load(allocator, index_path);
    unified.deinit();
    // The test verifies no memory leaks via the allocator.
}

// Tests for UnifiedSafeTensors.hasTensor
test "UnifiedSafeTensors.hasTensor: single mode returns true for existing" {
    const allocator = std.testing.allocator;
    const writer = @import("writer.zig");

    const tmp_dir_path = "/tmp/test_unified_hastensor_single_true";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    const shard_data = [_]u8{ 1, 2, 3, 4 };
    const shard_entries = [_]writer.TensorEntry{
        .{ .name = "my.tensor", .dtype = .i8, .shape = &[_]usize{4}, .data = &shard_data },
    };
    const model_path = tmp_dir_path ++ "/model.safetensors";
    try writer.write(allocator, model_path, &shard_entries);

    var unified = try UnifiedSafeTensors.load(allocator, model_path);
    defer unified.deinit();

    try std.testing.expect(unified.hasTensor("my.tensor"));
}

test "UnifiedSafeTensors.hasTensor: single mode returns false for nonexistent" {
    const allocator = std.testing.allocator;
    const writer = @import("writer.zig");

    const tmp_dir_path = "/tmp/test_unified_hastensor_single_false";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    const shard_data = [_]u8{ 1, 2, 3, 4 };
    const shard_entries = [_]writer.TensorEntry{
        .{ .name = "my.tensor", .dtype = .i8, .shape = &[_]usize{4}, .data = &shard_data },
    };
    const model_path = tmp_dir_path ++ "/model.safetensors";
    try writer.write(allocator, model_path, &shard_entries);

    var unified = try UnifiedSafeTensors.load(allocator, model_path);
    defer unified.deinit();

    try std.testing.expect(!unified.hasTensor("nonexistent"));
}

test "UnifiedSafeTensors.hasTensor: sharded mode returns true for existing" {
    const allocator = std.testing.allocator;
    const writer = @import("writer.zig");

    const tmp_dir_path = "/tmp/test_unified_hastensor_sharded_true";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    const shard_data = [_]u8{ 1, 2, 3, 4 };
    const shard_entries = [_]writer.TensorEntry{
        .{ .name = "my.tensor", .dtype = .i8, .shape = &[_]usize{4}, .data = &shard_data },
    };
    const shard_path = tmp_dir_path ++ "/model.safetensors";
    try writer.write(allocator, shard_path, &shard_entries);

    const index_json =
        \\{
        \\  "weight_map": {
        \\    "my.tensor": "model.safetensors"
        \\  }
        \\}
    ;
    const index_path = tmp_dir_path ++ "/model.safetensors.index.json";
    var index_file = try std.fs.cwd().createFile(index_path, .{});
    defer index_file.close();
    try index_file.writeAll(index_json);

    var unified = try UnifiedSafeTensors.load(allocator, index_path);
    defer unified.deinit();

    try std.testing.expect(unified.hasTensor("my.tensor"));
}

test "UnifiedSafeTensors.hasTensor: sharded mode returns false for nonexistent" {
    const allocator = std.testing.allocator;
    const writer = @import("writer.zig");

    const tmp_dir_path = "/tmp/test_unified_hastensor_sharded_false";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    const shard_data = [_]u8{ 1, 2, 3, 4 };
    const shard_entries = [_]writer.TensorEntry{
        .{ .name = "my.tensor", .dtype = .i8, .shape = &[_]usize{4}, .data = &shard_data },
    };
    const shard_path = tmp_dir_path ++ "/model.safetensors";
    try writer.write(allocator, shard_path, &shard_entries);

    const index_json =
        \\{
        \\  "weight_map": {
        \\    "my.tensor": "model.safetensors"
        \\  }
        \\}
    ;
    const index_path = tmp_dir_path ++ "/model.safetensors.index.json";
    var index_file = try std.fs.cwd().createFile(index_path, .{});
    defer index_file.close();
    try index_file.writeAll(index_json);

    var unified = try UnifiedSafeTensors.load(allocator, index_path);
    defer unified.deinit();

    try std.testing.expect(!unified.hasTensor("nonexistent"));
}

// Tests for UnifiedSafeTensors.tensorCount
test "UnifiedSafeTensors.tensorCount: single mode returns correct count" {
    const allocator = std.testing.allocator;
    const writer = @import("writer.zig");

    const tmp_dir_path = "/tmp/test_unified_tensorcount_single";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    const shard_data = [_]u8{ 1, 2, 3, 4, 5, 6 };
    const shard_entries = [_]writer.TensorEntry{
        .{ .name = "w1", .dtype = .i8, .shape = &[_]usize{2}, .data = shard_data[0..2] },
        .{ .name = "w2", .dtype = .i8, .shape = &[_]usize{2}, .data = shard_data[2..4] },
        .{ .name = "w3", .dtype = .i8, .shape = &[_]usize{2}, .data = shard_data[4..6] },
    };
    const model_path = tmp_dir_path ++ "/model.safetensors";
    try writer.write(allocator, model_path, &shard_entries);

    var unified = try UnifiedSafeTensors.load(allocator, model_path);
    defer unified.deinit();

    try std.testing.expectEqual(@as(usize, 3), unified.tensorCount());
}

test "UnifiedSafeTensors.tensorCount: sharded mode returns correct count" {
    const allocator = std.testing.allocator;
    const writer = @import("writer.zig");

    const tmp_dir_path = "/tmp/test_unified_tensorcount_sharded";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    const shard_data = [_]u8{ 1, 2, 3, 4, 5, 6 };
    const shard_entries = [_]writer.TensorEntry{
        .{ .name = "w1", .dtype = .i8, .shape = &[_]usize{2}, .data = shard_data[0..2] },
        .{ .name = "w2", .dtype = .i8, .shape = &[_]usize{2}, .data = shard_data[2..4] },
        .{ .name = "w3", .dtype = .i8, .shape = &[_]usize{2}, .data = shard_data[4..6] },
    };
    const shard_path = tmp_dir_path ++ "/model.safetensors";
    try writer.write(allocator, shard_path, &shard_entries);

    const index_json =
        \\{
        \\  "weight_map": {
        \\    "w1": "model.safetensors",
        \\    "w2": "model.safetensors",
        \\    "w3": "model.safetensors"
        \\  }
        \\}
    ;
    const index_path = tmp_dir_path ++ "/model.safetensors.index.json";
    var index_file = try std.fs.cwd().createFile(index_path, .{});
    defer index_file.close();
    try index_file.writeAll(index_json);

    var unified = try UnifiedSafeTensors.load(allocator, index_path);
    defer unified.deinit();

    try std.testing.expectEqual(@as(usize, 3), unified.tensorCount());
}
