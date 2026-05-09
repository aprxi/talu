//! HuggingFace model config fetch + normalization.
//!
//! This module is the single core-side place that fetches model config data for
//! repository-backed model IDs (`org/name`). It returns JSON that combines the
//! raw `config.json` with a stable minimal view used by API callers.

const std = @import("std");
const json = @import("json/root.zig");
const hf = @import("transport/hf.zig");
const talu_cache = @import("repository/talu_cache.zig");

pub const FetchOptions = struct {
    token: ?[]const u8 = null,
    endpoint_url: ?[]const u8 = null,
    revision: ?[]const u8 = null,
    force: bool = false,
    include_size: bool = true,
};

const max_config_bytes: usize = 16 * 1024 * 1024;
const max_cached_envelope_bytes: usize = max_config_bytes + (2 * 1024 * 1024);

fn getTaluHome(allocator: std.mem.Allocator) ![]const u8 {
    return talu_cache.getTaluHome(allocator);
}

fn isModelIdPathByte(byte: u8) bool {
    return std.ascii.isAlphanumeric(byte) or byte == '-' or byte == '_' or byte == '.';
}

fn encodeModelIdForPath(allocator: std.mem.Allocator, model_id: []const u8) ![]u8 {
    var out = std.ArrayListUnmanaged(u8){};
    errdefer out.deinit(allocator);

    for (model_id) |byte| {
        if (byte == '/') {
            try out.appendSlice(allocator, "--");
            continue;
        }
        if (isModelIdPathByte(byte)) {
            try out.append(allocator, byte);
        } else {
            try out.append(allocator, '_');
        }
    }

    return out.toOwnedSlice(allocator);
}

fn getModelConfigCachePath(
    allocator: std.mem.Allocator,
    model_id: []const u8,
    revision: []const u8,
    include_size: bool,
) ![]const u8 {
    const talu_home = try getTaluHome(allocator);
    defer allocator.free(talu_home);

    const model_path = try encodeModelIdForPath(allocator, model_id);
    defer allocator.free(model_path);

    const size_mode = if (include_size) "with_size" else "without_size";
    return std.fs.path.join(allocator, &.{
        talu_home,
        "model_configs",
        model_path,
        revision,
        size_mode,
        "envelope.json",
    });
}

fn tryLoadCachedEnvelope(
    allocator: std.mem.Allocator,
    model_id: []const u8,
    revision: []const u8,
    include_size: bool,
) !?[]u8 {
    const cache_path = getModelConfigCachePath(allocator, model_id, revision, include_size) catch |err| {
        return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => null,
        };
    };
    defer allocator.free(cache_path);

    const cached = std.fs.cwd().readFileAlloc(allocator, cache_path, max_cached_envelope_bytes) catch |err| {
        return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => null,
        };
    };
    errdefer allocator.free(cached);

    const parsed = json.parseValue(allocator, cached, .{
        .max_size_bytes = max_cached_envelope_bytes,
        .max_value_bytes = max_cached_envelope_bytes,
        .max_string_bytes = max_cached_envelope_bytes,
    }) catch |err| {
        return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => null,
        };
    };
    parsed.deinit();

    return cached;
}

fn cacheEnvelope(
    allocator: std.mem.Allocator,
    model_id: []const u8,
    revision: []const u8,
    include_size: bool,
    payload: []const u8,
) !void {
    const cache_path = try getModelConfigCachePath(allocator, model_id, revision, include_size);
    defer allocator.free(cache_path);

    const parent_dir = std.fs.path.dirname(cache_path) orelse return error.InvalidArgument;
    try std.fs.cwd().makePath(parent_dir);

    const tmp_path = try std.fmt.allocPrint(allocator, "{s}.tmp", .{cache_path});
    defer allocator.free(tmp_path);

    var tmp_file = try std.fs.cwd().createFile(tmp_path, .{ .truncate = true });
    defer tmp_file.close();
    errdefer std.fs.cwd().deleteFile(tmp_path) catch {};

    try tmp_file.writeAll(payload);
    try tmp_file.sync();

    std.fs.cwd().rename(tmp_path, cache_path) catch |err| switch (err) {
        error.PathAlreadyExists => {
            std.fs.cwd().deleteFile(cache_path) catch {};
            try std.fs.cwd().rename(tmp_path, cache_path);
        },
        else => return err,
    };
}

fn writeJson(writer: anytype, value: anytype) !void {
    try writer.print("{f}", .{std.json.fmt(value, .{})});
}

/// Fetch HuggingFace `config.json` and return a normalized JSON envelope.
///
/// Caller owns the returned JSON bytes.
pub fn fetchHfModelConfigJson(
    allocator: std.mem.Allocator,
    model_id: []const u8,
    options: FetchOptions,
) ![]u8 {
    if (std.mem.indexOfScalar(u8, model_id, '/') == null) {
        return error.InvalidArgument;
    }

    const revision = options.revision orelse "main";
    // TODO(anthonyp): Support arbitrary revisions once hf.fetchFile supports
    // selecting non-main refs.
    if (!std.mem.eql(u8, revision, "main")) {
        return error.InvalidArgument;
    }

    if (!options.force) {
        if (try tryLoadCachedEnvelope(allocator, model_id, revision, options.include_size)) |cached| {
            return cached;
        }
    }

    const config_path = hf.fetchFile(allocator, model_id, "config.json", .{
        .token = options.token,
        .endpoint_url = options.endpoint_url,
        .force = options.force,
    }) catch |err| return mapDownloadError(err);
    defer allocator.free(config_path);

    const config_bytes = std.fs.cwd().readFileAlloc(allocator, config_path, max_config_bytes) catch |err| {
        return switch (err) {
            error.FileNotFound => error.ModelConfigMissing,
            error.AccessDenied => error.AccessDenied,
            error.OutOfMemory => error.OutOfMemory,
            else => error.HttpError,
        };
    };
    defer allocator.free(config_bytes);

    const parsed_config = json.parseValue(allocator, config_bytes, .{
        .max_size_bytes = max_config_bytes,
        .max_value_bytes = max_config_bytes,
        .max_string_bytes = max_config_bytes,
    }) catch |err| {
        return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.ApiResponseParseError,
        };
    };
    defer parsed_config.deinit();

    const size_bytes = if (options.include_size)
        fetchHfModelSizeBytes(allocator, model_id, options) catch null
    else
        null;

    var out = std.ArrayListUnmanaged(u8){};
    errdefer out.deinit(allocator);
    const writer = out.writer(allocator);

    try writer.writeAll("{");

    try writer.writeAll("\"model\":");
    try writeJson(writer, model_id);

    try writer.writeAll(",\"revision\":");
    try writeJson(writer, revision);

    try writer.writeAll(",\"source\":\"huggingface\"");

    try writer.writeAll(",\"config\":");
    try writeJson(writer, parsed_config.value);

    try writer.writeAll(",\"minimal\":");
    try writeMinimalView(writer, parsed_config.value);

    try writer.writeAll(",\"size_bytes\":");
    if (size_bytes) |v| {
        try writer.print("{d}", .{v});
    } else {
        try writer.writeAll("null");
    }

    try writer.writeAll("}");

    const envelope = try out.toOwnedSlice(allocator);
    cacheEnvelope(allocator, model_id, revision, options.include_size, envelope) catch {};
    return envelope;
}

fn mapDownloadError(err: hf.DownloadError) anyerror {
    return switch (err) {
        error.InvalidModelId => error.InvalidArgument,
        error.ModelNotFound, error.NotFound => error.ModelNotFound,
        error.ConfigNotFound => error.ModelConfigMissing,
        error.Unauthorized => error.AccessDenied,
        error.OutOfMemory => error.OutOfMemory,
        error.RateLimited => error.RateLimited,
        else => error.HttpError,
    };
}

fn fetchHfModelSizeBytes(
    allocator: std.mem.Allocator,
    model_id: []const u8,
    options: FetchOptions,
) !?u64 {
    const results = hf.searchModelsRich(allocator, model_id, .{
        .token = options.token,
        .limit = 20,
        .filter = null,
        .library = null,
        .endpoint_url = options.endpoint_url,
    }) catch |err| return mapDownloadError(err);
    defer {
        for (results) |*entry| entry.deinit(allocator);
        allocator.free(results);
    }

    for (results) |entry| {
        if (!std.mem.eql(u8, entry.model_id, model_id)) continue;
        if (entry.params_total <= 0) return null;
        return @intCast(entry.params_total);
    }

    return null;
}

fn objectField(value: std.json.Value, key: []const u8) ?std.json.Value {
    if (value != .object) return null;
    return value.object.get(key);
}

fn writeStringValueOrNull(writer: anytype, value: ?std.json.Value) !void {
    if (value) |v| {
        if (v == .string) {
            try writeJson(writer, v.string);
            return;
        }
    }
    try writer.writeAll("null");
}

fn writeIntegerValueOrNull(writer: anytype, value: ?std.json.Value) !void {
    if (value) |v| {
        if (v == .integer) {
            try writer.print("{d}", .{v.integer});
            return;
        }
    }
    try writer.writeAll("null");
}

fn writeArchitecturesOrNull(writer: anytype, value: ?std.json.Value) !void {
    if (value) |v| {
        if (v == .array) {
            try writer.writeByte('[');
            var first = true;
            for (v.array.items) |item| {
                if (item != .string) continue;
                if (!first) try writer.writeByte(',');
                first = false;
                try writeJson(writer, item.string);
            }
            try writer.writeByte(']');
            return;
        }
    }
    try writer.writeAll("null");
}

fn writeJsonValueOrNull(writer: anytype, value: ?std.json.Value) !void {
    if (value) |v| {
        try writeJson(writer, v);
    } else {
        try writer.writeAll("null");
    }
}

fn writeMinimalView(writer: anytype, config: std.json.Value) !void {
    const model_type = objectField(config, "model_type");
    const architectures = objectField(config, "architectures");
    const max_position_embeddings = objectField(config, "max_position_embeddings");
    const torch_dtype = objectField(config, "torch_dtype");
    const quantization_config = objectField(config, "quantization_config");
    const vision_config = objectField(config, "vision_config");

    const has_vision_config = blk: {
        const v = vision_config orelse break :blk false;
        if (v == .null) break :blk false;
        break :blk true;
    };

    try writer.writeByte('{');

    try writer.writeAll("\"model_type\":");
    try writeStringValueOrNull(writer, model_type);

    try writer.writeAll(",\"architectures\":");
    try writeArchitecturesOrNull(writer, architectures);

    try writer.writeAll(",\"max_position_embeddings\":");
    try writeIntegerValueOrNull(writer, max_position_embeddings);

    try writer.writeAll(",\"torch_dtype\":");
    try writeStringValueOrNull(writer, torch_dtype);

    try writer.writeAll(",\"quantization_config\":");
    try writeJsonValueOrNull(writer, quantization_config);

    try writer.writeAll(",\"has_vision_config\":");
    try writeJson(writer, has_vision_config);

    try writer.writeByte('}');
}

test "fetchHfModelConfigJson rejects invalid model id" {
    const result = fetchHfModelConfigJson(std.testing.allocator, "invalid-model-id", .{});
    try std.testing.expectError(error.InvalidArgument, result);
}

test "fetchHfModelConfigJson returns cached envelope from TALU_HOME" {
    const allocator = std.testing.allocator;

    const old_talu_home = std.posix.getenv("TALU_HOME");
    defer {
        if (old_talu_home) |previous_value| {
            setEnvVarForTest(allocator, "TALU_HOME", std.mem.sliceTo(previous_value, 0)) catch {};
        } else {
            unsetEnvVarForTest(allocator, "TALU_HOME") catch {};
        }
    }

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    try setEnvVarForTest(allocator, "TALU_HOME", tmp_path);

    const cached_json =
        \\{"model":"org/model","revision":"main","source":"huggingface","config":{},"minimal":{"model_type":null,"architectures":null,"max_position_embeddings":null,"torch_dtype":null,"quantization_config":null,"has_vision_config":false},"size_bytes":null}
    ;
    const cache_path = try getModelConfigCachePath(allocator, "org/model", "main", true);
    defer allocator.free(cache_path);
    const cache_dir = std.fs.path.dirname(cache_path) orelse return error.Unexpected;
    try std.fs.cwd().makePath(cache_dir);

    var file = try std.fs.cwd().createFile(cache_path, .{ .truncate = true });
    defer file.close();
    try file.writeAll(cached_json);

    const result = try fetchHfModelConfigJson(allocator, "org/model", .{
        .include_size = true,
        .force = false,
    });
    defer allocator.free(result);

    try std.testing.expectEqualStrings(cached_json, result);
}

test "writeMinimalView extracts expected fields" {
    const sample =
        \\{
        \\  "model_type": "qwen2_5_vl",
        \\  "architectures": ["Qwen2_5_VLForConditionalGeneration"],
        \\  "max_position_embeddings": 128000,
        \\  "torch_dtype": "bfloat16",
        \\  "quantization_config": {"quant_method":"awq"},
        \\  "vision_config": {"hidden_size": 1280}
        \\}
    ;

    const parsed = try json.parseValue(std.testing.allocator, sample, .{
        .max_size_bytes = 1024,
        .max_value_bytes = 1024,
        .max_string_bytes = 1024,
    });
    defer parsed.deinit();

    var out = std.ArrayListUnmanaged(u8){};
    defer out.deinit(std.testing.allocator);
    try writeMinimalView(out.writer(std.testing.allocator), parsed.value);

    const minimal = try json.parseValue(std.testing.allocator, out.items, .{
        .max_size_bytes = 1024,
        .max_value_bytes = 1024,
        .max_string_bytes = 1024,
    });
    defer minimal.deinit();

    const root = minimal.value;
    try std.testing.expect(root == .object);
    try std.testing.expect(root.object.get("model_type").?.string.len > 0);
    try std.testing.expect(root.object.get("architectures").? == .array);
    try std.testing.expectEqual(@as(i64, 128000), root.object.get("max_position_embeddings").?.integer);
    try std.testing.expectEqualStrings("bfloat16", root.object.get("torch_dtype").?.string);
    try std.testing.expect(root.object.get("quantization_config").? == .object);
    try std.testing.expectEqual(true, root.object.get("has_vision_config").?.bool);
}

test "writeMinimalView handles missing fields" {
    const sample = "{}";
    const parsed = try json.parseValue(std.testing.allocator, sample, .{
        .max_size_bytes = 64,
        .max_value_bytes = 64,
        .max_string_bytes = 64,
    });
    defer parsed.deinit();

    var out = std.ArrayListUnmanaged(u8){};
    defer out.deinit(std.testing.allocator);
    try writeMinimalView(out.writer(std.testing.allocator), parsed.value);

    const minimal = try json.parseValue(std.testing.allocator, out.items, .{
        .max_size_bytes = 256,
        .max_value_bytes = 256,
        .max_string_bytes = 256,
    });
    defer minimal.deinit();

    const root = minimal.value.object;
    try std.testing.expect(root.get("model_type").? == .null);
    try std.testing.expect(root.get("architectures").? == .null);
    try std.testing.expect(root.get("max_position_embeddings").? == .null);
    try std.testing.expect(root.get("torch_dtype").? == .null);
    try std.testing.expect(root.get("quantization_config").? == .null);
    try std.testing.expectEqual(false, root.get("has_vision_config").?.bool);
}

const EnvFns = struct {
    extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
    extern "c" fn unsetenv(name: [*:0]const u8) c_int;
};

fn setEnvVarForTest(allocator: std.mem.Allocator, key: []const u8, value: []const u8) !void {
    const key_z = try allocator.allocSentinel(u8, key.len, 0);
    defer allocator.free(key_z);
    @memcpy(key_z[0..key.len], key);

    const value_z = try allocator.allocSentinel(u8, value.len, 0);
    defer allocator.free(value_z);
    @memcpy(value_z[0..value.len], value);

    if (EnvFns.setenv(key_z.ptr, value_z.ptr, 1) != 0) return error.Unexpected;
}

fn unsetEnvVarForTest(allocator: std.mem.Allocator, key: []const u8) !void {
    const key_z = try allocator.allocSentinel(u8, key.len, 0);
    defer allocator.free(key_z);
    @memcpy(key_z[0..key.len], key);

    _ = EnvFns.unsetenv(key_z.ptr);
}
