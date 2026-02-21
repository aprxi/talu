//! Model change policy checker.
//!
//! Enforces onboarding hygiene for model architecture changes:
//! - model architecture metadata changes must include model tests updates.
//! - model architecture + inference code changes require explicit reason tag.
//!
//! Usage:
//!   zig run core/tests/helpers/model_policy/root.zig -- <changed-file>...
//!
//! Environment:
//!   TALU_CHANGED_FILES: newline-separated changed file paths (used when no args)
//!   TALU_MODEL_INFERENCE_CHANGE_REASON: required when model+inference both change

const std = @import("std");

const PolicyState = struct {
    has_architecture_changes: bool = false,
    has_model_tests_changes: bool = false,
    has_inference_changes: bool = false,
};

fn isArchitectureMetadataPath(path: []const u8) bool {
    if (!std.mem.startsWith(u8, path, "core/src/models/")) return false;
    if (!std.mem.endsWith(u8, path, ".zig")) return false;

    if (std.mem.indexOf(u8, path, "/load/") != null) return false;
    if (std.mem.indexOf(u8, path, "/config/") != null) return false;
    if (std.mem.indexOf(u8, path, "/common/") != null) return false;

    if (std.mem.endsWith(u8, path, "/root.zig")) return false;
    if (std.mem.endsWith(u8, path, "/registry.zig")) return false;
    if (std.mem.endsWith(u8, path, "/runtime_architectures.zig")) return false;
    if (std.mem.endsWith(u8, path, "/layer_ops.zig")) return false;
    if (std.mem.endsWith(u8, path, "/op_types.zig")) return false;
    if (std.mem.endsWith(u8, path, "/loader.zig")) return false;

    return true;
}

fn collectPolicyState(paths: []const []const u8) PolicyState {
    var state = PolicyState{};
    for (paths) |path| {
        if (isArchitectureMetadataPath(path)) state.has_architecture_changes = true;
        if (std.mem.startsWith(u8, path, "core/tests/models/")) state.has_model_tests_changes = true;
        if (std.mem.startsWith(u8, path, "core/src/inference/")) state.has_inference_changes = true;
    }
    return state;
}

fn appendEnvChangedFiles(allocator: std.mem.Allocator, out: *std.ArrayList([]const u8)) !void {
    const raw = std.process.getEnvVarOwned(allocator, "TALU_CHANGED_FILES") catch |err| switch (err) {
        error.EnvironmentVariableNotFound => return,
        else => return err,
    };
    defer allocator.free(raw);

    var it = std.mem.tokenizeScalar(u8, raw, '\n');
    while (it.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (trimmed.len == 0) continue;
        try out.append(allocator, trimmed);
    }
}

pub fn runPolicyCheck(
    writer: anytype,
    changed_paths: []const []const u8,
    reason: ?[]const u8,
) !void {
    const state = collectPolicyState(changed_paths);

    if (!state.has_architecture_changes) {
        try writer.writeAll("model-policy: no architecture metadata changes detected\n");
        return;
    }

    if (!state.has_model_tests_changes) {
        try writer.writeAll("model-policy: architecture metadata changed but core/tests/models/* was not updated\n");
        return error.ModelTestsRequired;
    }

    if (state.has_inference_changes) {
        const trimmed_reason = if (reason) |r| std.mem.trim(u8, r, " \t\r\n") else "";
        if (trimmed_reason.len == 0) {
            try writer.writeAll("model-policy: architecture + inference changed together; set TALU_MODEL_INFERENCE_CHANGE_REASON\n");
            return error.InferenceChangeReasonRequired;
        }
    }

    try writer.writeAll("model-policy: ok\n");
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var changed = std.ArrayList([]const u8){};
    defer changed.deinit(allocator);

    if (args.len > 1) {
        for (args[1..]) |arg| {
            try changed.append(allocator, arg);
        }
    } else {
        try appendEnvChangedFiles(allocator, &changed);
    }

    const reason = std.process.getEnvVarOwned(allocator, "TALU_MODEL_INFERENCE_CHANGE_REASON") catch |err| switch (err) {
        error.EnvironmentVariableNotFound => null,
        else => return err,
    };
    defer if (reason) |r| allocator.free(r);

    if (changed.items.len == 0) {
        std.debug.print("model-policy: no changed files provided; set TALU_CHANGED_FILES or pass paths as args\n", .{});
        return;
    }

    try runPolicyCheck(std.fs.File.stdout().deprecatedWriter(), changed.items, reason);
}

test "collectPolicyState detects architecture changes and tests" {
    const files = [_][]const u8{
        "core/src/models/llama/llama3.zig",
        "core/tests/models/onboarding_contract_test.zig",
    };
    const state = collectPolicyState(&files);
    try std.testing.expect(state.has_architecture_changes);
    try std.testing.expect(state.has_model_tests_changes);
    try std.testing.expect(!state.has_inference_changes);
}

test "runPolicyCheck requires model tests when architecture changed" {
    const files = [_][]const u8{
        "core/src/models/llama/llama3.zig",
    };

    var out = std.ArrayList(u8){};
    defer out.deinit(std.testing.allocator);

    try std.testing.expectError(
        error.ModelTestsRequired,
        runPolicyCheck(out.writer(std.testing.allocator), &files, null),
    );
}

test "runPolicyCheck requires reason for architecture plus inference changes" {
    const files = [_][]const u8{
        "core/src/models/llama/llama3.zig",
        "core/tests/models/onboarding_contract_test.zig",
        "core/src/inference/backend/cpu/engine.zig",
    };

    var out = std.ArrayList(u8){};
    defer out.deinit(std.testing.allocator);

    try std.testing.expectError(
        error.InferenceChangeReasonRequired,
        runPolicyCheck(out.writer(std.testing.allocator), &files, null),
    );
}

test "runPolicyCheck passes with reason for architecture plus inference changes" {
    const files = [_][]const u8{
        "core/src/models/llama/llama3.zig",
        "core/tests/models/onboarding_contract_test.zig",
        "core/src/inference/backend/cpu/engine.zig",
    };

    var out = std.ArrayList(u8){};
    defer out.deinit(std.testing.allocator);

    try runPolicyCheck(out.writer(std.testing.allocator), &files, "needed new runtime kernel contract");
}
