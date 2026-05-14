//! Lightweight policy linter for core source layering rules.
//!
//! Usage:
//!   zig run core/tests/helpers/lint/root.zig -- core/src

const std = @import("std");

fn lineNumberForOffset(source: []const u8, offset: usize) usize {
    return 1 + std.mem.count(u8, source[0..offset], "\n");
}

fn isComputePath(path: []const u8) bool {
    return std.mem.startsWith(u8, path, "core/src/compute/");
}

fn isInferencePath(path: []const u8) bool {
    return std.mem.startsWith(u8, path, "core/src/inference/");
}

fn isInferenceProductionSourcePath(path: []const u8) bool {
    if (!isInferencePath(path)) return false;
    if (!isCoreSourceFile(path)) return false;
    if (std.mem.endsWith(u8, path, "_test.zig")) return false;
    if (std.mem.endsWith(u8, path, "_tests.zig")) return false;
    if (std.mem.indexOf(u8, path, "/testdata/") != null) return false;
    if (std.mem.indexOf(u8, path, "/bridge_tests_") != null) return false;
    if (std.mem.endsWith(u8, path, "/bridge_config_test.cpp")) return false;
    return true;
}

fn isInferenceBackendPath(path: []const u8) bool {
    return std.mem.startsWith(u8, path, "core/src/inference/backend/");
}

fn isInferenceBackendVisionPath(path: []const u8) bool {
    if (!isInferenceBackendPath(path)) return false;
    if (std.mem.indexOf(u8, path, "/vision/") != null) return true;
    return std.mem.endsWith(u8, path, "/vision.zig");
}

fn isModelsPath(path: []const u8) bool {
    return std.mem.startsWith(u8, path, "core/src/models/");
}

fn isResponsesPath(path: []const u8) bool {
    return std.mem.startsWith(u8, path, "core/src/responses/");
}

fn isResponsesInferenceBridgePath(path: []const u8) bool {
    return std.mem.eql(u8, path, "core/src/responses/inference_bridge.zig");
}

fn isCoreSourceFile(path: []const u8) bool {
    return std.mem.endsWith(u8, path, ".zig") or
        std.mem.endsWith(u8, path, ".c") or
        std.mem.endsWith(u8, path, ".cpp") or
        std.mem.endsWith(u8, path, ".h") or
        std.mem.endsWith(u8, path, ".hpp") or
        std.mem.endsWith(u8, path, ".inc");
}

fn isConverterPath(path: []const u8) bool {
    return std.mem.startsWith(u8, path, "core/src/converter/");
}

fn isConverterInferenceBoundaryPath(path: []const u8) bool {
    return std.mem.eql(u8, path, "core/src/converter/calibration_capture.zig");
}

fn isModelsGenericLoaderPath(path: []const u8) bool {
    return std.mem.eql(u8, path, "core/src/models/load/weights.zig") or
        std.mem.eql(u8, path, "core/src/models/load/generic_weights.zig");
}

fn isCpuVisionSelectorPath(path: []const u8) bool {
    return std.mem.eql(u8, path, "core/src/inference/backend/cpu/vision/root.zig");
}

fn importTargetsCompute(target: []const u8) bool {
    return std.mem.indexOf(u8, target, "compute/") != null;
}

fn importTargetsInference(target: []const u8) bool {
    return std.mem.indexOf(u8, target, "inference/") != null or
        std.mem.eql(u8, target, "inference_pkg");
}

fn importTargetsForbiddenComputeDependency(target: []const u8) bool {
    return std.mem.indexOf(u8, target, "inference/") != null or
        std.mem.indexOf(u8, target, "models/") != null or
        std.mem.indexOf(u8, target, "responses/") != null or
        std.mem.indexOf(u8, target, "protocol/") != null or
        std.mem.indexOf(u8, target, "capi/") != null or
        std.mem.indexOf(u8, target, "bindings/") != null or
        std.mem.indexOf(u8, target, "bridge") != null or
        std.mem.indexOf(u8, target, "scheduler") != null or
        std.mem.indexOf(u8, target, "runtime_contract") != null or
        std.mem.eql(u8, target, "inference_pkg") or
        std.mem.eql(u8, target, "models_pkg") or
        std.mem.eql(u8, target, "capi_pkg");
}

fn isComputeRootImport(target: []const u8) bool {
    return std.mem.endsWith(u8, target, "compute/root.zig");
}

fn isOldTopLevelSimdOrQuantImport(target: []const u8) bool {
    if (std.mem.startsWith(u8, target, "../simd/") and !std.mem.startsWith(u8, target, "../simd/arch/")) return true;
    if (std.mem.startsWith(u8, target, "../../simd/") and !std.mem.startsWith(u8, target, "../../simd/arch/")) return true;
    if (std.mem.startsWith(u8, target, "../../../simd/") and !std.mem.startsWith(u8, target, "../../../simd/arch/")) return true;
    if (std.mem.startsWith(u8, target, "../quant/")) return true;
    if (std.mem.startsWith(u8, target, "../../quant/")) return true;
    if (std.mem.startsWith(u8, target, "../../../quant/")) return true;
    if (std.mem.startsWith(u8, target, "compute/simd/")) return true;
    if (std.mem.startsWith(u8, target, "compute/quant/")) return true;
    if (std.mem.indexOf(u8, target, "/compute/simd/") != null) return true;
    if (std.mem.indexOf(u8, target, "/compute/quant/") != null) return true;
    return false;
}

fn extractImportTarget(
    allocator: std.mem.Allocator,
    tree: std.zig.Ast,
    arg_node: std.zig.Ast.Node.Index,
) !?[]u8 {
    return switch (tree.nodeTag(arg_node)) {
        .string_literal => blk: {
            const tok = tree.firstToken(arg_node);
            const raw = tree.tokenSlice(tok);
            break :blk try std.zig.string_literal.parseAlloc(allocator, raw);
        },
        else => null,
    };
}

fn lintSource(allocator: std.mem.Allocator, file_path: []const u8, source: []const u8, emit: bool) !usize {
    const source_z = try allocator.dupeZ(u8, source);
    defer allocator.free(source_z);
    var tree = try std.zig.Ast.parse(allocator, source_z, .zig);
    defer tree.deinit(allocator);

    var violations: usize = 0;

    var builtin_params_buf: [2]std.zig.Ast.Node.Index = undefined;
    for (0..tree.nodes.len) |node_idx_raw| {
        const node: std.zig.Ast.Node.Index = @enumFromInt(node_idx_raw);
        const tag = tree.nodeTag(node);
        switch (tag) {
            .builtin_call,
            .builtin_call_comma,
            .builtin_call_two,
            .builtin_call_two_comma,
            => {},
            else => continue,
        }

        const main_tok = tree.nodeMainToken(node);
        if (!std.mem.eql(u8, tree.tokenSlice(main_tok), "@import")) continue;

        const params = tree.builtinCallParams(&builtin_params_buf, node) orelse continue;
        if (params.len == 0) continue;

        const arg_node = params[0];
        const target_owned = try extractImportTarget(allocator, tree, arg_node);
        defer if (target_owned) |target| allocator.free(target);
        const target = target_owned orelse continue;

        const line = lineNumberForOffset(source, tree.tokenStart(tree.firstToken(arg_node)));
        if (isComputePath(file_path) and importTargetsForbiddenComputeDependency(target)) {
            violations += 1;
            if (emit) {
                std.debug.print("{s}:{d}: forbidden compute dependency import: \"{s}\"\n", .{ file_path, line, target });
            }
        }

        if (isOldTopLevelSimdOrQuantImport(target)) {
            violations += 1;
            if (emit) {
                std.debug.print("{s}:{d}: forbidden legacy compute import path: \"{s}\"\n", .{ file_path, line, target });
            }
        }

        if (isInferencePath(file_path) and importTargetsCompute(target) and !isComputeRootImport(target)) {
            violations += 1;
            if (emit) {
                std.debug.print("{s}:{d}: inference must import compute via compute/root.zig only: \"{s}\"\n", .{ file_path, line, target });
            }
        }

        if (isResponsesPath(file_path) and !isResponsesInferenceBridgePath(file_path) and importTargetsInference(target)) {
            violations += 1;
            if (emit) {
                std.debug.print("{s}:{d}: responses must import inference only via responses/inference_bridge.zig: \"{s}\"\n", .{ file_path, line, target });
            }
        }

        if (isConverterPath(file_path) and !isConverterInferenceBoundaryPath(file_path) and importTargetsInference(target)) {
            violations += 1;
            if (emit) {
                std.debug.print("{s}:{d}: converter must import inference only via converter/calibration_capture.zig: \"{s}\"\n", .{ file_path, line, target });
            }
        }

        if (isModelsPath(file_path) and std.mem.indexOf(u8, target, "inference/") != null) {
            violations += 1;
            if (emit) {
                std.debug.print("{s}:{d}: models must not import inference internals: \"{s}\"\n", .{ file_path, line, target });
            }
        }
    }

    // The CPU vision selector/hydrator must consume model metadata and must not
    // carry hardcoded model tensor naming templates.
    if (isCpuVisionSelectorPath(file_path)) {
        const forbidden = [_][]const u8{
            "model.visual.",
            "vision_tower.vision_model",
            "model.vision_model.",
            "model.multi_modal_projector.",
        };
        for (forbidden) |token| {
            if (std.mem.indexOf(u8, source, token) != null) {
                violations += 1;
                if (emit) {
                    std.debug.print("{s}: forbidden hardcoded vision tensor template token: \"{s}\"\n", .{ file_path, token });
                }
            }
        }
    }

    // Generic loader files must remain metadata-driven and must not reintroduce
    // model-family string heuristics for core shape/feature decisions.
    if (isModelsGenericLoaderPath(file_path)) {
        const forbidden = [_][]const u8{
            "mlp.experts.0.gate_proj.weight",
            "mixer.conv1d.weight",
            "inferMoEFromWeights",
            "maybeForceMambaF32",
        };
        for (forbidden) |token| {
            if (std.mem.indexOf(u8, source, token) != null) {
                violations += 1;
                if (emit) {
                    std.debug.print("{s}: forbidden loader heuristic token: \"{s}\"\n", .{ file_path, token });
                }
            }
        }
    }

    // Non-vision inference backend modules must stay model-family agnostic.
    // Model naming conventions belong to models metadata contracts.
    if (isInferenceBackendPath(file_path) and !isInferenceBackendVisionPath(file_path)) {
        const forbidden = [_][]const u8{
            "\"llama\"",
            "\"llama2\"",
            "\"llama3\"",
            "\"qwen\"",
            "\"granite\"",
            "\"gemma\"",
            "\"mistral\"",
            "\"phi\"",
            "\"gpt_oss\"",
            "\"lfm2\"",
            "\"youtu\"",
        };
        for (forbidden) |token| {
            if (std.mem.indexOf(u8, source, token) != null) {
                violations += 1;
                if (emit) {
                    std.debug.print("{s}: forbidden model-family literal in inference backend: \"{s}\"\n", .{ file_path, token });
                }
            }
        }
    }

    // Inference must route SDPA usage through stable compute namespaces.
    // Transitional aliases in `compute.cpu.*` are forbidden.
    if (isInferencePath(file_path)) {
        const forbidden = [_][]const u8{
            "compute.cpu.linalg_sdpa",
            "compute.cpu.sdpa_decode",
        };
        for (forbidden) |token| {
            if (std.mem.indexOf(u8, source, token) != null) {
                violations += 1;
                if (emit) {
                    std.debug.print("{s}: forbidden transitional compute symbol: \"{s}\"\n", .{ file_path, token });
                }
            }
        }
    }

    return violations;
}

fn collectZigBasenames(
    allocator: std.mem.Allocator,
    dir_path: []const u8,
) !std.StringHashMap(void) {
    var out = std.StringHashMap(void).init(allocator);
    errdefer {
        var it = out.iterator();
        while (it.next()) |entry| allocator.free(entry.key_ptr.*);
        out.deinit();
    }

    var dir = try std.fs.cwd().openDir(dir_path, .{ .iterate = true });
    defer dir.close();
    var it = dir.iterate();
    while (try it.next()) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".zig")) continue;
        const owned = try allocator.dupe(u8, entry.name);
        try out.put(owned, {});
    }
    return out;
}

fn freeNameSet(allocator: std.mem.Allocator, set: *std.StringHashMap(void)) void {
    var it = set.iterator();
    while (it.next()) |entry| allocator.free(entry.key_ptr.*);
    set.deinit();
}

const BackendParityGap = enum {
    metal_executor,
    cpu_executor,
    metal_kernel,
    cpu_kernel,
};

fn stringIn(name: []const u8, allowed: []const []const u8) bool {
    for (allowed) |item| {
        if (std.mem.eql(u8, name, item)) return true;
    }
    return false;
}

fn isAllowedBackendParityGap(gap: BackendParityGap, name: []const u8) bool {
    const metal_executor_gaps = [_][]const u8{
        "block.zig",
        "model.zig",
        "runtime.zig",
    };
    const cpu_executor_gaps = [_][]const u8{
        "runtime_graph.zig",
    };
    const metal_kernel_gaps = [_][]const u8{
        "attention.zig",
        "describe_fmt.zig",
        "embedding.zig",
        "ffn.zig",
        "gated_delta.zig",
        "kv_cache.zig",
        "mamba.zig",
        "mla_attention.zig",
        "moe.zig",
        "norm.zig",
        "per_layer_branch.zig",
        "rope.zig",
        "shortconv.zig",
        "weights.zig",
    };

    // Metal currently owns execution through the runtime graph and MLX bridge,
    // so these named gaps are classified asymmetries rather than silent drift.
    // New CPU/Metal file drift still fails closed until it is classified here.
    return switch (gap) {
        .metal_executor => stringIn(name, &metal_executor_gaps),
        .cpu_executor => stringIn(name, &cpu_executor_gaps),
        .metal_kernel => stringIn(name, &metal_kernel_gaps),
        .cpu_kernel => false,
    };
}

fn sourceHasPubConst(source: []const u8, name: []const u8) bool {
    var buf: [128]u8 = undefined;
    const needle = std.fmt.bufPrint(&buf, "pub const {s}", .{name}) catch return false;
    return std.mem.indexOf(u8, source, needle) != null;
}

fn checkRequiredPubConsts(
    file_path: []const u8,
    source: []const u8,
    required: []const []const u8,
    emit: bool,
) usize {
    var violations: usize = 0;
    for (required) |name| {
        if (!sourceHasPubConst(source, name)) {
            violations += 1;
            if (emit) {
                std.debug.print("{s}: missing required pub const `{s}` for backend symmetry\n", .{ file_path, name });
            }
        }
    }
    return violations;
}

fn lintLoggingPolicy(file_path: []const u8, source: []const u8, emit: bool) usize {
    var violations: usize = 0;
    if (std.mem.endsWith(u8, file_path, ".zig")) {
        if (std.mem.indexOf(u8, source, "std.log.") != null) {
            violations += 1;
            if (emit) {
                std.debug.print("{s}: forbidden std.log usage; use core/src/log.zig\n", .{file_path});
            }
        }
    }

    const forbidden_c_stderr = [_][]const u8{
        "std::fprintf(stderr",
        "std::fflush(stderr",
        "fprintf(stderr",
        "fflush(stderr",
    };
    var found_c_stderr = false;
    for (forbidden_c_stderr) |token| {
        if (std.mem.indexOf(u8, source, token) != null) {
            if (!found_c_stderr) {
                violations += 1;
                found_c_stderr = true;
            }
            if (emit) {
                std.debug.print("{s}: forbidden direct stderr logging token `{s}`; use core logging boundary\n", .{ file_path, token });
            }
        }
    }
    return violations;
}

fn lintInferenceProductionLexicon(file_path: []const u8, source: []const u8, emit: bool) usize {
    if (!isInferenceProductionSourcePath(file_path)) return 0;
    const forbidden = [_][]const u8{ "fallback", "fall back", "falling back", "degraded", "legacy" };
    var violations: usize = 0;
    for (forbidden) |token| {
        var search_start: usize = 0;
        while (indexOfIgnoreCase(source, token, search_start)) |offset| {
            violations += 1;
            if (emit) {
                std.debug.print("{s}:{d}: forbidden inference production token: \"{s}\"\n", .{
                    file_path,
                    lineNumberForOffset(source, offset),
                    token,
                });
            }
            search_start = offset + token.len;
        }
    }
    return violations;
}

fn indexOfIgnoreCase(source: []const u8, token: []const u8, start: usize) ?usize {
    if (token.len == 0 or start >= source.len) return null;
    var offset = start;
    while (offset + token.len <= source.len) : (offset += 1) {
        if (std.ascii.eqlIgnoreCase(source[offset .. offset + token.len], token)) return offset;
    }
    return null;
}

fn lintBackendParity(allocator: std.mem.Allocator, emit: bool) !usize {
    var violations: usize = 0;

    const executor_cpu = "core/src/inference/backend/cpu/executor";
    const executor_metal = "core/src/inference/backend/metal/executor";
    const kernels_cpu = "core/src/inference/backend/cpu/kernels";
    const kernels_metal = "core/src/inference/backend/metal/kernels";

    var exec_cpu_set = try collectZigBasenames(allocator, executor_cpu);
    defer freeNameSet(allocator, &exec_cpu_set);
    var exec_metal_set = try collectZigBasenames(allocator, executor_metal);
    defer freeNameSet(allocator, &exec_metal_set);
    var kern_cpu_set = try collectZigBasenames(allocator, kernels_cpu);
    defer freeNameSet(allocator, &kern_cpu_set);
    var kern_metal_set = try collectZigBasenames(allocator, kernels_metal);
    defer freeNameSet(allocator, &kern_metal_set);

    var it_exec_cpu = exec_cpu_set.iterator();
    while (it_exec_cpu.next()) |entry| {
        if (!exec_metal_set.contains(entry.key_ptr.*)) {
            if (isAllowedBackendParityGap(.metal_executor, entry.key_ptr.*)) continue;
            violations += 1;
            if (emit) {
                std.debug.print("backend parity: missing metal executor file `{s}`\n", .{entry.key_ptr.*});
            }
        }
    }
    var it_exec_metal = exec_metal_set.iterator();
    while (it_exec_metal.next()) |entry| {
        if (!exec_cpu_set.contains(entry.key_ptr.*)) {
            if (isAllowedBackendParityGap(.cpu_executor, entry.key_ptr.*)) continue;
            violations += 1;
            if (emit) {
                std.debug.print("backend parity: missing cpu executor file `{s}`\n", .{entry.key_ptr.*});
            }
        }
    }

    var it_kern_cpu = kern_cpu_set.iterator();
    while (it_kern_cpu.next()) |entry| {
        if (!kern_metal_set.contains(entry.key_ptr.*)) {
            if (isAllowedBackendParityGap(.metal_kernel, entry.key_ptr.*)) continue;
            violations += 1;
            if (emit) {
                std.debug.print("backend parity: missing metal kernel file `{s}`\n", .{entry.key_ptr.*});
            }
        }
    }
    var it_kern_metal = kern_metal_set.iterator();
    while (it_kern_metal.next()) |entry| {
        if (!kern_cpu_set.contains(entry.key_ptr.*)) {
            if (isAllowedBackendParityGap(.cpu_kernel, entry.key_ptr.*)) continue;
            violations += 1;
            if (emit) {
                std.debug.print("backend parity: missing cpu kernel file `{s}`\n", .{entry.key_ptr.*});
            }
        }
    }

    const kernel_root_required = [_][]const u8{
        "support",
        "TransformerBlock",
        "MultiHeadAttention",
        "SwiGLU",
        "RMSNorm",
        "ShortConvKernel",
        "MoEFFN",
        "EmbeddingLookup",
        "KVCache",
        "FusedAttention",
        "RotaryEmbedding",
        "WeightAccess",
    };
    const executor_root_required = [_][]const u8{
        "weights",
        "runtime",
        "model",
        "block",
        "Model",
        "Transformer",
        "Block",
        "TransformerBlock",
        "BlockKind",
        "Attention",
        "RMSNorm",
        "FFNLayer",
        "AttnTemp",
        "AttnCache",
        "ScratchBuffer",
    };

    const kernel_root_cpu = try std.fs.cwd().readFileAlloc(
        allocator,
        "core/src/inference/backend/cpu/kernels/root.zig",
        1024 * 1024,
    );
    defer allocator.free(kernel_root_cpu);
    const kernel_root_metal = try std.fs.cwd().readFileAlloc(
        allocator,
        "core/src/inference/backend/metal/kernels/root.zig",
        1024 * 1024,
    );
    defer allocator.free(kernel_root_metal);
    const executor_root_cpu = try std.fs.cwd().readFileAlloc(
        allocator,
        "core/src/inference/backend/cpu/executor/root.zig",
        1024 * 1024,
    );
    defer allocator.free(executor_root_cpu);
    const executor_root_metal = try std.fs.cwd().readFileAlloc(
        allocator,
        "core/src/inference/backend/metal/executor/root.zig",
        1024 * 1024,
    );
    defer allocator.free(executor_root_metal);

    violations += checkRequiredPubConsts("core/src/inference/backend/cpu/kernels/root.zig", kernel_root_cpu, &kernel_root_required, emit);
    violations += checkRequiredPubConsts("core/src/inference/backend/metal/kernels/root.zig", kernel_root_metal, &kernel_root_required, emit);
    violations += checkRequiredPubConsts("core/src/inference/backend/cpu/executor/root.zig", executor_root_cpu, &executor_root_required, emit);
    violations += checkRequiredPubConsts("core/src/inference/backend/metal/executor/root.zig", executor_root_metal, &executor_root_required, emit);

    return violations;
}

fn lintTree(allocator: std.mem.Allocator, root_path: []const u8) !usize {
    var total_violations: usize = 0;
    var dir = try std.fs.cwd().openDir(root_path, .{ .iterate = true });
    defer dir.close();

    var walker = try dir.walk(allocator);
    defer walker.deinit();

    while (try walker.next()) |entry| {
        if (entry.kind != .file) continue;
        if (!isCoreSourceFile(entry.path)) continue;

        const full_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ root_path, entry.path });
        defer allocator.free(full_path);

        const source = try std.fs.cwd().readFileAlloc(allocator, full_path, 32 * 1024 * 1024);
        defer allocator.free(source);

        total_violations += lintLoggingPolicy(full_path, source, true);
        total_violations += lintInferenceProductionLexicon(full_path, source, true);
        if (std.mem.endsWith(u8, entry.path, ".zig")) {
            total_violations += try lintSource(allocator, full_path, source, true);
        }
    }

    return total_violations;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    const root_path = if (args.len >= 2) args[1] else "core/src";
    var violations = try lintTree(allocator, root_path);
    violations += try lintBackendParity(allocator, true);
    if (violations != 0) {
        std.debug.print("lint: found {d} violation(s)\n", .{violations});
        return error.LintFailed;
    }
}

test "lintSource rejects inference import in compute" {
    const src =
        \\const bad = @import("../../inference/root.zig");
    ;
    try std.testing.expectEqual(@as(usize, 1), try lintSource(std.testing.allocator, "core/src/compute/cpu/foo.zig", src, false));
}

test "lintSource rejects models import in compute" {
    const src =
        \\const bad = @import("../../models/root.zig");
    ;
    try std.testing.expectEqual(@as(usize, 1), try lintSource(std.testing.allocator, "core/src/compute/cpu/bar.zig", src, false));
}

test "lintSource rejects C API import in compute" {
    const src =
        \\const bad = @import("../../capi/root.zig");
    ;
    try std.testing.expectEqual(@as(usize, 1), try lintSource(std.testing.allocator, "core/src/compute/cpu/copy.zig", src, false));
}

test "lintSource rejects runtime contract import in compute" {
    const src =
        \\const bad = @import("../../models/plan/runtime_contract/types.zig");
    ;
    try std.testing.expectEqual(@as(usize, 1), try lintSource(std.testing.allocator, "core/src/compute/cpu/capabilities.zig", src, false));
}

test "lintSource rejects bridge and scheduler imports in compute" {
    const src =
        \\const bad_bridge = @import("../../responses/inference_bridge.zig");
        \\const bad_scheduler = @import("../../inference/scheduler/generic.zig");
    ;
    try std.testing.expectEqual(@as(usize, 2), try lintSource(std.testing.allocator, "core/src/compute/cuda/capabilities.zig", src, false));
}

test "lintSource rejects bindings import in compute" {
    const src =
        \\const bad = @import("../../../bindings/python/root.zig");
    ;
    try std.testing.expectEqual(@as(usize, 1), try lintSource(std.testing.allocator, "core/src/compute/metal/capabilities.zig", src, false));
}

test "lintSource rejects legacy top-level simd import path" {
    const src =
        \\const simd = @import("../simd/root.zig");
    ;
    try std.testing.expectEqual(@as(usize, 1), try lintSource(std.testing.allocator, "core/src/compute/cpu/reduction.zig", src, false));
}

test "lintSource allows new cpu simd arch path" {
    const src =
        \\const simd = @import("simd/arch/root.zig");
    ;
    try std.testing.expectEqual(@as(usize, 0), try lintSource(std.testing.allocator, "core/src/compute/cpu/reduction.zig", src, false));
}

test "lintSource rejects deep compute import from inference" {
    const src =
        \\const bad = @import("../../../compute/cpu/memory.zig");
    ;
    try std.testing.expectEqual(@as(usize, 1), try lintSource(std.testing.allocator, "core/src/inference/backend/cpu/engine.zig", src, false));
}

test "lintSource allows compute root import from inference" {
    const src =
        \\const ok = @import("../../../compute/root.zig");
    ;
    try std.testing.expectEqual(@as(usize, 0), try lintSource(std.testing.allocator, "core/src/inference/backend/cpu/engine.zig", src, false));
}

test "lintLoggingPolicy rejects std.log in Zig source" {
    const src =
        \\pub fn bad() void {
        \\    std.log.warn("bad", .{});
        \\}
    ;
    try std.testing.expectEqual(
        @as(usize, 1),
        lintLoggingPolicy("core/src/inference/bad.zig", src, false),
    );
}

test "lintLoggingPolicy rejects direct C stderr logging" {
    const src =
        \\void bad(void) {
        \\    std::fprintf(stderr, "bad\n");
        \\}
    ;
    try std.testing.expectEqual(
        @as(usize, 1),
        lintLoggingPolicy("core/src/inference/backend/metal/mlx_bridge/bad.inc", src, false),
    );
}

test "lintInferenceProductionLexicon rejects hidden-route tokens" {
    const src =
        \\const a = "Fallback";
        \\const b = "fall back";
        \\const c = "falling back";
        \\const d = "degraded";
        \\const e = "legacy";
    ;
    try std.testing.expectEqual(
        @as(usize, 5),
        lintInferenceProductionLexicon("core/src/inference/backend/cuda/route.zig", src, false),
    );
}

test "lintInferenceProductionLexicon allows test source paths" {
    const src =
        \\const a = "fallback";
        \\const b = "degraded";
        \\const c = "legacy";
    ;
    try std.testing.expectEqual(
        @as(usize, 0),
        lintInferenceProductionLexicon("core/src/inference/backend/cuda/route_test.zig", src, false),
    );
}

test "lintSource rejects models importing inference internals" {
    const src =
        \\const bad = @import("../../inference/backend/topology.zig");
    ;
    try std.testing.expectEqual(
        @as(usize, 1),
        try lintSource(std.testing.allocator, "core/src/models/registry.zig", src, false),
    );
}

test "lintSource rejects responses direct inference import" {
    const src =
        \\const bad = @import("../inference/root.zig");
    ;
    try std.testing.expectEqual(
        @as(usize, 1),
        try lintSource(std.testing.allocator, "core/src/responses/local.zig", src, false),
    );
}

test "lintSource allows responses inference bridge import" {
    const src =
        \\const ok = @import("inference_bridge.zig");
    ;
    try std.testing.expectEqual(
        @as(usize, 0),
        try lintSource(std.testing.allocator, "core/src/responses/local.zig", src, false),
    );
}

test "lintSource rejects converter direct inference import" {
    const src =
        \\const bad = @import("../inference/calibration_capture.zig");
    ;
    try std.testing.expectEqual(
        @as(usize, 1),
        try lintSource(std.testing.allocator, "core/src/converter/mxfp8.zig", src, false),
    );
}

test "lintSource allows converter inference boundary import" {
    const src =
        \\const ok = @import("inference_pkg");
    ;
    try std.testing.expectEqual(
        @as(usize, 0),
        try lintSource(std.testing.allocator, "core/src/converter/calibration_capture.zig", src, false),
    );
}

test "models to inference import count is zero" {
    try std.testing.expectEqual(@as(usize, 0), try lintTree(std.testing.allocator, "core/src/models"));
}

test "lintSource rejects hardcoded vision tensor templates in selector" {
    const src =
        \\const bad = "model.visual.blocks.0.attn.q_proj.weight";
    ;
    try std.testing.expectEqual(
        @as(usize, 1),
        try lintSource(std.testing.allocator, "core/src/inference/backend/cpu/vision/root.zig", src, false),
    );
}

test "lintSource allows metadata-driven selector without hardcoded names" {
    const src =
        \\const candidates = vision_metadata.split_qkv_probe_candidates;
    ;
    try std.testing.expectEqual(
        @as(usize, 0),
        try lintSource(std.testing.allocator, "core/src/inference/backend/cpu/vision/root.zig", src, false),
    );
}

test "lintSource rejects loader heuristic token in generic loader" {
    const src =
        \\const bad = "mlp.experts.0.gate_proj.weight";
    ;
    try std.testing.expectEqual(
        @as(usize, 1),
        try lintSource(std.testing.allocator, "core/src/models/load/weights.zig", src, false),
    );
}

test "lintSource allows metadata-driven loader without forbidden heuristics" {
    const src =
        \\const ok = arch.d_ff_source_weight_ids;
    ;
    try std.testing.expectEqual(
        @as(usize, 0),
        try lintSource(std.testing.allocator, "core/src/models/load/weights.zig", src, false),
    );
}

test "lintSource rejects model-family literal in non-vision inference backend" {
    const src =
        \\const bad = "llama3";
    ;
    try std.testing.expectEqual(
        @as(usize, 1),
        try lintSource(std.testing.allocator, "core/src/inference/backend/cpu/executor/model.zig", src, false),
    );
}

test "lintSource allows model-family literal in vision backend path" {
    const src =
        \\const ok = "llama3";
    ;
    try std.testing.expectEqual(
        @as(usize, 0),
        try lintSource(std.testing.allocator, "core/src/inference/backend/cpu/vision/split_qkv.zig", src, false),
    );
}

test "lintSource rejects transitional compute symbol in inference" {
    const src =
        \\const bad = compute.cpu.linalg_sdpa;
    ;
    try std.testing.expectEqual(
        @as(usize, 1),
        try lintSource(std.testing.allocator, "core/src/inference/backend/cpu/executor/block.zig", src, false),
    );
}

test "isAllowedBackendParityGap allows classified metal executor asymmetry" {
    try std.testing.expect(isAllowedBackendParityGap(.metal_executor, "runtime.zig"));
    try std.testing.expect(!isAllowedBackendParityGap(.metal_executor, "new_runtime.zig"));
}

test "isAllowedBackendParityGap allows classified metal kernel asymmetry" {
    try std.testing.expect(isAllowedBackendParityGap(.metal_kernel, "embedding.zig"));
    try std.testing.expect(!isAllowedBackendParityGap(.metal_kernel, "new_kernel.zig"));
}

test "isAllowedBackendParityGap keeps metal-only kernels strict" {
    try std.testing.expect(!isAllowedBackendParityGap(.cpu_kernel, "runtime_graph.zig"));
}

test "backend cpu/metal parity checks pass" {
    try std.testing.expectEqual(@as(usize, 0), try lintBackendParity(std.testing.allocator, false));
}
