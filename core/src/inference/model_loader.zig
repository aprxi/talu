//! Inference model loading and architecture support checks.
//!
//! Keeps runtime model-loading/compatibility policy in the inference layer.
//! This is the only graph-loading surface that inference modules should use.

const std = @import("std");
const json = @import("../io/json/root.zig");
const graph = @import("../graph/root.zig");

pub const LoadedModel = graph.LoadedModel;
pub const LoadOptions = graph.LoadOptions;
pub const weights = graph.loader.weights;
pub const loadModel = graph.loadModel;
pub const loadArchitectureDefinitions = graph.loadArchitectureDefinitions;

/// Result of checking model architecture support.
pub const ArchitectureCheck = struct {
    supported: bool,
    model_type_buf: [64]u8 = undefined,
    model_type_len: usize = 0,
    architecture_buf: [64]u8 = undefined,
    architecture_len: usize = 0,

    pub fn getModelType(self: *const @This()) ?[]const u8 {
        if (self.model_type_len == 0) return null;
        return self.model_type_buf[0..self.model_type_len];
    }

    pub fn getArchitecture(self: *const @This()) ?[]const u8 {
        if (self.architecture_len == 0) return null;
        return self.architecture_buf[0..self.architecture_len];
    }
};

/// Check if a model's architecture is supported without fully loading.
/// Checks against the runtime graph registry.
pub fn checkArchitecture(allocator: std.mem.Allocator, config_path: []const u8) !ArchitectureCheck {
    var arch_check = ArchitectureCheck{ .supported = false };

    const config_bytes = std.fs.cwd().readFileAlloc(allocator, config_path, 256 * 1024) catch {
        // Can't read config - assume supported (might be older format)
        arch_check.supported = true;
        return arch_check;
    };
    defer allocator.free(config_bytes);

    const parsed_config = json.parseValue(allocator, config_bytes, .{ .max_size_bytes = 256 * 1024 }) catch {
        // Can't parse - assume supported
        arch_check.supported = true;
        return arch_check;
    };
    defer parsed_config.deinit();

    const obj = switch (parsed_config.value) {
        .object => |o| o,
        else => {
            arch_check.supported = true;
            return arch_check;
        },
    };

    if (obj.get("model_type")) |v| {
        if (v == .string) {
            const model_type = v.string;
            const len = @min(model_type.len, arch_check.model_type_buf.len);
            @memcpy(arch_check.model_type_buf[0..len], model_type[0..len]);
            arch_check.model_type_len = len;
        }
    }

    if (obj.get("architectures")) |v| {
        if (v == .array and v.array.items.len > 0) {
            const first = v.array.items[0];
            if (first == .string) {
                const architecture_name = first.string;
                const len = @min(architecture_name.len, arch_check.architecture_buf.len);
                @memcpy(arch_check.architecture_buf[0..len], architecture_name[0..len]);
                arch_check.architecture_len = len;
            }
        }
    }

    if (arch_check.getModelType()) |model_type| {
        graph.init(allocator);
        _ = graph.loadArchitectureDefinitions(allocator);
        arch_check.supported = graph.detectFromModelType(model_type) != null;
        return arch_check;
    }

    // No model_type found - assume supported (older models)
    arch_check.supported = true;
    return arch_check;
}

/// Check model architecture from a model directory.
pub fn checkArchitectureFromDir(allocator: std.mem.Allocator, model_dir: []const u8) !ArchitectureCheck {
    const config_path = try std.fs.path.join(allocator, &.{ model_dir, "config.json" });
    defer allocator.free(config_path);
    return checkArchitecture(allocator, config_path);
}

test "ArchitectureCheck.getModelType returns model type when present" {
    var check = ArchitectureCheck{ .supported = true };
    const model_type = "llama";
    @memcpy(check.model_type_buf[0..model_type.len], model_type);
    check.model_type_len = model_type.len;

    const result = check.getModelType();
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("llama", result.?);
}

test "ArchitectureCheck.getModelType returns null when not present" {
    const check = ArchitectureCheck{ .supported = true };
    try std.testing.expect(check.getModelType() == null);
}

test "ArchitectureCheck.getArchitecture returns architecture when present" {
    var check = ArchitectureCheck{ .supported = true };
    const arch = "LlamaForCausalLM";
    @memcpy(check.architecture_buf[0..arch.len], arch);
    check.architecture_len = arch.len;

    const result = check.getArchitecture();
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("LlamaForCausalLM", result.?);
}

test "ArchitectureCheck.getArchitecture returns null when not present" {
    const check = ArchitectureCheck{ .supported = true };
    try std.testing.expect(check.getArchitecture() == null);
}
