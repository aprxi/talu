//! Architecture Registry
//!
//! Global registry for model architectures. Architectures are registered from:
//! - Static compile-time definitions
//! - Runtime registration via C API (Python @architecture decorator)
//!
//! The registry maps architecture names and model_types to their compute graphs.

const std = @import("std");
const Allocator = std.mem.Allocator;
const log = @import("../log.zig");

const types = @import("types.zig");
const parser = @import("parser.zig");
const compiler = @import("compiler.zig");
const ops_mod = @import("layer_ops.zig");

pub const Architecture = types.Architecture;

// =============================================================================
// Global State
// =============================================================================

var registry: std.StringHashMapUnmanaged(Architecture) = .{}; // Thread-safe: protected by registry_mutex
var registry_allocator: ?Allocator = null; // Thread-safe: protected by registry_mutex
var initialized: bool = false; // Thread-safe: protected by registry_mutex
var registry_mutex: std.Thread.Mutex = .{};

fn freeWeightSpecs(allocator: Allocator, specs: []const types.WeightSpec) void {
    for (specs) |spec| {
        allocator.free(spec.id);
        for (spec.candidates) |cand| allocator.free(cand);
        allocator.free(spec.candidates);
        allocator.free(spec.module_type);
        allocator.free(spec.dtype);
        if (spec.expected_shape) |shape| allocator.free(shape);
        if (spec.transforms.len > 0) allocator.free(spec.transforms);
    }
    allocator.free(specs);
}

/// Free all allocated memory inside an Op struct.
fn freeOp(allocator: Allocator, op: types.Op) void {
    if (op.name) |name| allocator.free(name);
    if (op.activation) |act| allocator.free(act);
    for (op.inputs) |input| {
        switch (input) {
            .tensor => |t| allocator.free(t),
            .scalar => {},
        }
    }
    if (op.inputs.len > 0) allocator.free(op.inputs);
    for (op.outputs) |out| allocator.free(out);
    if (op.outputs.len > 0) allocator.free(op.outputs);
    if (op.shape.len > 0) allocator.free(op.shape);
    if (op.split_sizes.len > 0) allocator.free(op.split_sizes);
}

/// Free a slice of Ops and all their internals.
fn freeOps(allocator: Allocator, ops: []const types.Op) void {
    for (ops) |op| freeOp(allocator, op);
    if (ops.len > 0) allocator.free(ops);
}

/// Free a compiled LayerOp program and all its allocated internals.
fn freeCompiledProgram(allocator: Allocator, program: []const ops_mod.LayerOp) void {
    for (program) |op| {
        switch (op) {
            .linear => |linear| allocator.free(linear.weight_name),
            .add_param => |add_p| allocator.free(add_p.param_name),
            .add_param_scalar => |add_ps| allocator.free(add_ps.param_name),
            .mul_param => |mul_p| allocator.free(mul_p.param_name),
            .reshape => |reshape| if (reshape.shape.len > 0) allocator.free(reshape.shape),
            .split => |split| if (split.split_sizes.len > 0) allocator.free(split.split_sizes),
            else => {},
        }
    }
    allocator.free(program);
}

// =============================================================================
// Initialization
// =============================================================================

/// Initialize the architecture registry.
/// Must be called before any registration or lookup.
/// Safe to call multiple times - will only initialize once.
pub fn init(allocator: Allocator) void {
    registry_mutex.lock();
    defer registry_mutex.unlock();

    if (initialized) return;
    registry_allocator = allocator;
    registry = std.StringHashMapUnmanaged(Architecture){};
    initialized = true;
}

/// Deinitialize and free all registered architectures.
pub fn deinit() void {
    registry_mutex.lock();
    defer registry_mutex.unlock();

    if (!initialized) return;
    if (registry_allocator) |alloc| {
        var iter = registry.iterator();
        while (iter.next()) |entry| {
            alloc.free(entry.key_ptr.*);
            alloc.free(entry.value_ptr.name);
            for (entry.value_ptr.model_types) |model_type| {
                alloc.free(model_type);
            }
            alloc.free(entry.value_ptr.model_types);

            // Free compiled program FIRST (before freeing ops that it references)
            // Note: compiled programs now own their own string copies, so order doesn't matter
            if (entry.value_ptr.compiled_program) |prog| {
                freeCompiledProgram(alloc, prog);
            }

            // Free block ops and their internals
            freeOps(alloc, entry.value_ptr.block_ops);
            freeOps(alloc, entry.value_ptr.pre_block_ops);
            freeOps(alloc, entry.value_ptr.post_block_ops);

            // Free block_variants (heterogeneous models)
            if (entry.value_ptr.block_variants) |variants| {
                for (variants) |*variant| {
                    alloc.free(variant.name);
                    freeOps(alloc, variant.ops);
                    if (variant.weights.len > 0) freeWeightSpecs(alloc, variant.weights);
                    if (variant.compiled_program) |prog| {
                        freeCompiledProgram(alloc, prog);
                    }
                }
                alloc.free(variants);
            }
            // Free layer_map
            if (entry.value_ptr.layer_map) |map| {
                alloc.free(map);
            }
            if (entry.value_ptr.block_weights.len > 0) {
                freeWeightSpecs(alloc, entry.value_ptr.block_weights);
            }
            if (entry.value_ptr.global_weights.len > 0) {
                freeWeightSpecs(alloc, entry.value_ptr.global_weights);
            }
            if (entry.value_ptr.weight_prefixes.len > 0) {
                for (entry.value_ptr.weight_prefixes) |p| alloc.free(p);
                alloc.free(entry.value_ptr.weight_prefixes);
            }
        }
        registry.deinit(alloc);
    }
    initialized = false;
    registry_allocator = null;
}

fn registerLocked(arch: Architecture) !void {
    if (!initialized) return error.RegistryNotInitialized;
    const alloc = registry_allocator orelse return error.RegistryNotInitialized;

    const key = try alloc.dupe(u8, arch.name);
    errdefer alloc.free(key);

    try registry.put(alloc, key, arch);
}

// =============================================================================
// Registration
// =============================================================================

/// Register a custom architecture.
pub fn register(arch: Architecture) !void {
    registry_mutex.lock();
    defer registry_mutex.unlock();

    try registerLocked(arch);
}

/// Load architecture definition from a JSON file.
pub fn loadFromFile(path: []const u8) !void {
    registry_mutex.lock();
    defer registry_mutex.unlock();

    const alloc = registry_allocator orelse return error.RegistryNotInitialized;

    log.trace("graph", "loadFromFile", .{}, @src());

    const file_data = std.fs.cwd().readFileAlloc(alloc, path, 1024 * 1024) catch |err| {
        if (err == error.FileNotFound) {
            log.trace("graph", "Architecture file not found", .{}, @src());
            return;
        }
        return err;
    };
    defer alloc.free(file_data);

    const arch = try parser.parseFromJson(alloc, file_data);
    errdefer {
        alloc.free(arch.name);
        for (arch.model_types) |model_type| alloc.free(model_type);
        alloc.free(arch.model_types);
        alloc.free(arch.block_ops);
    }

    try registerLocked(arch);

    log.debug("graph", "Loaded architecture", .{}, @src());
}

/// Load architecture from a JSON string.
/// Used by embedded graphs and C API.
pub fn loadFromJson(json_str: []const u8) !void {
    registry_mutex.lock();
    defer registry_mutex.unlock();

    const alloc = registry_allocator orelse return error.RegistryNotInitialized;

    const arch = try parser.parseFromJson(alloc, json_str);
    errdefer {
        alloc.free(arch.name);
        for (arch.model_types) |model_type| alloc.free(model_type);
        alloc.free(arch.model_types);
        alloc.free(arch.block_ops);
    }

    try registerLocked(arch);
}

// =============================================================================
// Lookup
// =============================================================================

/// Get a registered architecture by name.
pub fn get(name: []const u8) ?*Architecture {
    registry_mutex.lock();
    defer registry_mutex.unlock();

    if (!initialized) return null;
    return registry.getPtr(name);
}

/// Check if an architecture is registered.
pub fn has(name: []const u8) bool {
    registry_mutex.lock();
    defer registry_mutex.unlock();

    if (!initialized) return false;
    return registry.contains(name);
}

/// Detect architecture from model_type string.
/// Returns null if not found in registry.
pub fn detectFromModelType(model_type: []const u8) ?*Architecture {
    registry_mutex.lock();
    defer registry_mutex.unlock();

    if (!initialized) {
        log.trace("graph", "Registry not initialized", .{}, @src());
        return null;
    }

    log.trace("graph", "Looking for model type", .{}, @src());

    var iter = registry.iterator();
    while (iter.next()) |entry| {
        for (entry.value_ptr.model_types) |model_type_entry| {
            if (std.mem.eql(u8, model_type, model_type_entry)) {
                log.trace("graph", "Found architecture", .{}, @src());
                return entry.value_ptr;
            }
        }
    }
    log.trace("graph", "Model type not found", .{}, @src());
    return null;
}

/// List all registered architecture names.
pub fn listNames(allocator: Allocator) ![]const []const u8 {
    registry_mutex.lock();
    defer registry_mutex.unlock();

    if (!initialized) return &.{};

    var names = std.ArrayListUnmanaged([]const u8){};
    var iter = registry.iterator();
    while (iter.next()) |entry| {
        try names.append(allocator, entry.key_ptr.*);
    }
    return try names.toOwnedSlice(allocator);
}

// =============================================================================
// Compilation
// =============================================================================

/// Ensure an architecture has a compiled block program.
/// Lazily compiles on first access.
pub fn ensureCompiled(arch: *Architecture) ![]const ops_mod.LayerOp {
    registry_mutex.lock();
    defer registry_mutex.unlock();

    if (arch.compiled_program) |prog| {
        return prog;
    }

    const alloc = registry_allocator orelse return error.RegistryNotInitialized;
    const prog = try compiler.compile(alloc, arch.block_ops);
    arch.compiled_program = prog;
    return prog;
}

/// Ensure a compiled program for a specific layer index.
/// For homogeneous models, returns the same program for all layers.
/// For heterogeneous models, returns the appropriate variant's program.
pub fn ensureCompiledForLayer(arch: *Architecture, layer_idx: usize) ![]const ops_mod.LayerOp {
    return ensureCompiledForLayerWithOverride(arch, layer_idx, null);
}

/// Ensure a compiled program for a specific layer index, with optional layer_types override.
/// For heterogeneous models, the override allows different model sizes to specify their
/// own layer arrangements (e.g., where attention vs mamba layers are positioned).
pub fn ensureCompiledForLayerWithOverride(arch: *Architecture, layer_idx: usize, layer_types_override: ?[]const u8) ![]const ops_mod.LayerOp {
    registry_mutex.lock();
    defer registry_mutex.unlock();

    const alloc = registry_allocator orelse return error.RegistryNotInitialized;

    // Check if this is a heterogeneous model
    if (arch.block_variants) |variants| {
        const variant_idx = arch.getVariantIndexWithOverride(layer_idx, layer_types_override);
        if (variant_idx >= variants.len) {
            return error.InvalidVariantIndex;
        }
        var variant = &variants[variant_idx];

        // Lazily compile the variant's program
        if (variant.compiled_program) |prog| {
            return prog;
        }

        const prog = try compiler.compile(alloc, variant.ops);
        variant.compiled_program = prog;
        return prog;
    }

    // Homogeneous model - use the standard compiled_program
    if (arch.compiled_program) |prog| {
        return prog;
    }

    const prog = try compiler.compile(alloc, arch.block_ops);
    arch.compiled_program = prog;
    return prog;
}

/// Get the registry allocator (for use by compiler).
fn getAllocator() ?Allocator {
    registry_mutex.lock();
    defer registry_mutex.unlock();

    return registry_allocator;
}

// =============================================================================
// Unit Tests
// =============================================================================

test "init and deinit registry" {
    const allocator = std.testing.allocator;

    init(allocator);
    defer deinit();

    try std.testing.expect(initialized);
    try std.testing.expect(registry_allocator != null);
}

test "init multiple times is safe" {
    const allocator = std.testing.allocator;

    init(allocator);
    defer deinit();

    // Calling init again should be safe
    init(allocator);
    try std.testing.expect(initialized);
}

test "deinit before init is safe" {
    deinit();
    // Should not crash
}

test "register and get architecture" {
    const allocator = std.testing.allocator;

    init(allocator);
    defer deinit();

    const name = try allocator.dupe(u8, "test_arch");
    errdefer allocator.free(name);
    const model_types = try allocator.alloc([]const u8, 1);
    errdefer allocator.free(model_types);
    const model_type0 = try allocator.dupe(u8, "TestModel");
    errdefer allocator.free(model_type0);
    model_types[0] = model_type0;
    const block_ops = try allocator.alloc(types.Op, 1);
    errdefer allocator.free(block_ops);
    block_ops[0] = types.Op{ .op_type = .norm };

    const arch = Architecture{
        .name = name,
        .model_types = model_types,
        .block_ops = block_ops,
    };

    try register(arch);

    const retrieved = get("test_arch");
    try std.testing.expect(retrieved != null);
    try std.testing.expectEqualStrings("test_arch", retrieved.?.name);
    try std.testing.expectEqual(@as(usize, 1), retrieved.?.model_types.len);
    try std.testing.expectEqualStrings("TestModel", retrieved.?.model_types[0]);
}

test "get non-existent architecture returns null" {
    const allocator = std.testing.allocator;

    init(allocator);
    defer deinit();

    const result = get("does_not_exist");
    try std.testing.expect(result == null);
}

test "get when not initialized returns null" {
    deinit(); // Ensure not initialized

    const result = get("test");
    try std.testing.expect(result == null);
}

test "has returns true for registered architecture" {
    const allocator = std.testing.allocator;

    init(allocator);
    defer deinit();

    const name = try allocator.dupe(u8, "test_arch");
    errdefer allocator.free(name);
    const model_types = try allocator.alloc([]const u8, 1);
    errdefer allocator.free(model_types);
    const model_type0 = try allocator.dupe(u8, "TestModel");
    errdefer allocator.free(model_type0);
    model_types[0] = model_type0;
    const block_ops = try allocator.alloc(types.Op, 0);
    errdefer allocator.free(block_ops);

    const arch = Architecture{
        .name = name,
        .model_types = model_types,
        .block_ops = block_ops,
    };

    try register(arch);

    try std.testing.expect(has("test_arch"));
}

test "has returns false for non-existent architecture" {
    const allocator = std.testing.allocator;

    init(allocator);
    defer deinit();

    try std.testing.expect(!has("does_not_exist"));
}

test "has when not initialized returns false" {
    deinit(); // Ensure not initialized

    try std.testing.expect(!has("test"));
}

test "detectFromModelType finds architecture" {
    const allocator = std.testing.allocator;

    init(allocator);
    defer deinit();

    const name = try allocator.dupe(u8, "test_arch");
    errdefer allocator.free(name);
    const model_types = try allocator.alloc([]const u8, 2);
    errdefer allocator.free(model_types);
    const model_type0 = try allocator.dupe(u8, "type_a");
    errdefer allocator.free(model_type0);
    model_types[0] = model_type0;
    const model_type1 = try allocator.dupe(u8, "type_b");
    errdefer allocator.free(model_type1);
    model_types[1] = model_type1;
    const block_ops = try allocator.alloc(types.Op, 0);
    errdefer allocator.free(block_ops);

    const arch = Architecture{
        .name = name,
        .model_types = model_types,
        .block_ops = block_ops,
    };

    try register(arch);

    const result = detectFromModelType("type_a");
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("test_arch", result.?.name);

    const result2 = detectFromModelType("type_b");
    try std.testing.expect(result2 != null);
    try std.testing.expectEqualStrings("test_arch", result2.?.name);
}

test "detectFromModelType returns null for unknown model type" {
    const allocator = std.testing.allocator;

    init(allocator);
    defer deinit();

    const result = detectFromModelType("unknown_model");
    try std.testing.expect(result == null);
}

test "detectFromModelType when not initialized returns null" {
    deinit(); // Ensure not initialized

    const result = detectFromModelType("test");
    try std.testing.expect(result == null);
}

test "listNames returns all registered architectures" {
    const allocator = std.testing.allocator;

    init(allocator);
    defer deinit();

    // Register first architecture
    {
        const name = try allocator.dupe(u8, "arch1");
        errdefer allocator.free(name);
        const model_types = try allocator.alloc([]const u8, 1);
        errdefer allocator.free(model_types);
        const model_type0 = try allocator.dupe(u8, "Model1");
        errdefer allocator.free(model_type0);
        model_types[0] = model_type0;
        const block_ops = try allocator.alloc(types.Op, 0);
        errdefer allocator.free(block_ops);

        try register(Architecture{
            .name = name,
            .model_types = model_types,
            .block_ops = block_ops,
        });
    }

    // Register second architecture
    {
        const name = try allocator.dupe(u8, "arch2");
        errdefer allocator.free(name);
        const model_types = try allocator.alloc([]const u8, 1);
        errdefer allocator.free(model_types);
        const model_type0 = try allocator.dupe(u8, "Model2");
        errdefer allocator.free(model_type0);
        model_types[0] = model_type0;
        const block_ops = try allocator.alloc(types.Op, 0);
        errdefer allocator.free(block_ops);

        try register(Architecture{
            .name = name,
            .model_types = model_types,
            .block_ops = block_ops,
        });
    }

    const names = try listNames(allocator);
    defer allocator.free(names);

    try std.testing.expectEqual(@as(usize, 2), names.len);

    // Check that both names are in the list (order may vary)
    var found_arch1 = false;
    var found_arch2 = false;
    for (names) |name| {
        if (std.mem.eql(u8, name, "arch1")) found_arch1 = true;
        if (std.mem.eql(u8, name, "arch2")) found_arch2 = true;
    }
    try std.testing.expect(found_arch1);
    try std.testing.expect(found_arch2);
}

test "listNames when not initialized returns empty slice" {
    const allocator = std.testing.allocator;

    deinit(); // Ensure not initialized

    const names = try listNames(allocator);
    defer allocator.free(names);

    try std.testing.expectEqual(@as(usize, 0), names.len);
}

test "ensureCompiled compiles architecture once" {
    const allocator = std.testing.allocator;

    init(allocator);
    defer deinit();

    const name = try allocator.dupe(u8, "test_arch");
    errdefer allocator.free(name);
    const model_types = try allocator.alloc([]const u8, 1);
    errdefer allocator.free(model_types);
    const model_type0 = try allocator.dupe(u8, "TestModel");
    errdefer allocator.free(model_type0);
    model_types[0] = model_type0;
    const block_ops = try allocator.alloc(types.Op, 2);
    errdefer allocator.free(block_ops);
    block_ops[0] = types.Op{ .op_type = .norm };
    block_ops[1] = types.Op{ .op_type = .add };

    const arch = Architecture{
        .name = name,
        .model_types = model_types,
        .block_ops = block_ops,
    };

    try register(arch);

    const arch_ptr = get("test_arch").?;
    try std.testing.expect(arch_ptr.compiled_program == null);

    const program1 = try ensureCompiled(arch_ptr);
    try std.testing.expect(program1.len > 0);
    try std.testing.expect(arch_ptr.compiled_program != null);

    // Calling again should return the same program
    const program2 = try ensureCompiled(arch_ptr);
    try std.testing.expectEqual(program1.ptr, program2.ptr);
}

test "loadFromFile - loadFromJson registers architecture from JSON string" {
    const allocator = std.testing.allocator;

    init(allocator);
    defer deinit();

    const json =
        \\{
        \\  "name": "json_test_arch",
        \\  "model_types": ["JsonTestModel"],
        \\  "block": [
        \\    {"op": "norm"},
        \\    {"op": "add"}
        \\  ]
        \\}
    ;

    try loadFromJson(json);

    try std.testing.expect(has("json_test_arch"));
    const arch = get("json_test_arch").?;
    try std.testing.expectEqualStrings("json_test_arch", arch.name);
    try std.testing.expectEqual(@as(usize, 1), arch.model_types.len);
    try std.testing.expectEqualStrings("JsonTestModel", arch.model_types[0]);
}

test "register when not initialized returns error" {
    deinit(); // Ensure not initialized

    const allocator = std.testing.allocator;
    const name = try allocator.dupe(u8, "test");
    defer allocator.free(name);
    const model_types = try allocator.alloc([]const u8, 0);
    defer allocator.free(model_types);
    const block_ops = try allocator.alloc(types.Op, 0);
    defer allocator.free(block_ops);

    const arch = Architecture{
        .name = name,
        .model_types = model_types,
        .block_ops = block_ops,
    };

    const result = register(arch);
    try std.testing.expectError(error.RegistryNotInitialized, result);
}

test "register multiple architectures with different model types" {
    const allocator = std.testing.allocator;

    init(allocator);
    defer deinit();

    // Register first architecture (with multiple model types mapping to it)
    {
        const name = try allocator.dupe(u8, "arch_a");
        errdefer allocator.free(name);
        const model_types = try allocator.alloc([]const u8, 2);
        errdefer allocator.free(model_types);
        const model_type0 = try allocator.dupe(u8, "type_x");
        errdefer allocator.free(model_type0);
        model_types[0] = model_type0;
        const model_type1 = try allocator.dupe(u8, "type_y");
        errdefer allocator.free(model_type1);
        model_types[1] = model_type1;
        const block_ops = try allocator.alloc(types.Op, 0);
        errdefer allocator.free(block_ops);

        try register(Architecture{
            .name = name,
            .model_types = model_types,
            .block_ops = block_ops,
        });
    }

    // Register second architecture
    {
        const name = try allocator.dupe(u8, "arch_b");
        errdefer allocator.free(name);
        const model_types = try allocator.alloc([]const u8, 1);
        errdefer allocator.free(model_types);
        const model_type0 = try allocator.dupe(u8, "type_z");
        errdefer allocator.free(model_type0);
        model_types[0] = model_type0;
        const block_ops = try allocator.alloc(types.Op, 0);
        errdefer allocator.free(block_ops);

        try register(Architecture{
            .name = name,
            .model_types = model_types,
            .block_ops = block_ops,
        });
    }

    // Verify both are registered
    try std.testing.expect(has("arch_a"));
    try std.testing.expect(has("arch_b"));

    // Verify model type detection works for both
    const arch_a_from_x = detectFromModelType("type_x");
    try std.testing.expect(arch_a_from_x != null);
    try std.testing.expectEqualStrings("arch_a", arch_a_from_x.?.name);

    const arch_a_from_y = detectFromModelType("type_y");
    try std.testing.expect(arch_a_from_y != null);
    try std.testing.expectEqualStrings("arch_a", arch_a_from_y.?.name);

    const arch_b_from_z = detectFromModelType("type_z");
    try std.testing.expect(arch_b_from_z != null);
    try std.testing.expectEqualStrings("arch_b", arch_b_from_z.?.name);
}

test "loadFromFile loads valid JSON architecture" {
    const allocator = std.testing.allocator;

    init(allocator);
    defer deinit();

    // Create a temporary test file
    const test_json =
        \\{
        \\  "name": "test_from_file",
        \\  "model_types": ["TestFileModel", "TestFileModel2"],
        \\  "block": [
        \\    {"op": "norm"},
        \\    {"op": "add"}
        \\  ]
        \\}
    ;

    const tmp_path = "/tmp/test_registry_loadFromFile.json";
    {
        const file = try std.fs.cwd().createFile(tmp_path, .{});
        defer file.close();
        try file.writeAll(test_json);
    }
    defer std.fs.cwd().deleteFile(tmp_path) catch {};

    // Load the architecture
    try loadFromFile(tmp_path);

    // Verify it was registered
    try std.testing.expect(has("test_from_file"));

    const arch = get("test_from_file").?;
    try std.testing.expectEqualStrings("test_from_file", arch.name);
    try std.testing.expectEqual(@as(usize, 2), arch.model_types.len);
    try std.testing.expectEqualStrings("TestFileModel", arch.model_types[0]);
    try std.testing.expectEqualStrings("TestFileModel2", arch.model_types[1]);
    try std.testing.expectEqual(@as(usize, 2), arch.block_ops.len);
}

test "loadFromFile silently returns when file not found" {
    const allocator = std.testing.allocator;

    init(allocator);
    defer deinit();

    // Try to load a non-existent file - should not error
    try loadFromFile("/tmp/does_not_exist_registry_test.json");

    // Registry should still be empty
    const names = try listNames(allocator);
    defer allocator.free(names);
    try std.testing.expectEqual(@as(usize, 0), names.len);
}

test "loadFromFile returns error when not initialized" {
    deinit(); // Ensure not initialized

    const result = loadFromFile("/tmp/any_path.json");
    try std.testing.expectError(error.RegistryNotInitialized, result);
}

test "loadFromFile returns error for invalid JSON" {
    const allocator = std.testing.allocator;

    init(allocator);
    defer deinit();

    // Create a file with invalid JSON
    const invalid_json = "{ this is not valid json }";

    const tmp_path = "/tmp/test_registry_invalid.json";
    {
        const file = try std.fs.cwd().createFile(tmp_path, .{});
        defer file.close();
        try file.writeAll(invalid_json);
    }
    defer std.fs.cwd().deleteFile(tmp_path) catch {};

    // Should return a parse error
    const result = loadFromFile(tmp_path);
    try std.testing.expect(result != error.RegistryNotInitialized);
    try std.testing.expect(result != error.FileNotFound);
    // The specific error depends on the parser implementation
}

test "loadFromFile handles architecture with all fields" {
    const allocator = std.testing.allocator;

    init(allocator);
    defer deinit();

    // Create JSON with pre_block and post_block ops
    const test_json =
        \\{
        \\  "name": "full_test_arch",
        \\  "model_types": ["FullTestModel"],
        \\  "pre_block": [
        \\    {"op": "norm"}
        \\  ],
        \\  "block": [
        \\    {"op": "add"}
        \\  ],
        \\  "post_block": [
        \\    {"op": "norm"}
        \\  ]
        \\}
    ;

    const tmp_path = "/tmp/test_registry_full.json";
    {
        const file = try std.fs.cwd().createFile(tmp_path, .{});
        defer file.close();
        try file.writeAll(test_json);
    }
    defer std.fs.cwd().deleteFile(tmp_path) catch {};

    try loadFromFile(tmp_path);

    const arch = get("full_test_arch").?;
    try std.testing.expectEqualStrings("full_test_arch", arch.name);
    try std.testing.expectEqual(@as(usize, 1), arch.model_types.len);
    try std.testing.expectEqual(@as(usize, 1), arch.block_ops.len);
    try std.testing.expectEqual(@as(usize, 1), arch.pre_block_ops.len);
    try std.testing.expectEqual(@as(usize, 1), arch.post_block_ops.len);
}

test "ensureCompiledForLayer compiles homogeneous architecture" {
    const allocator = std.testing.allocator;

    init(allocator);
    defer deinit();

    const name = try allocator.dupe(u8, "test_arch");
    errdefer allocator.free(name);
    const model_types = try allocator.alloc([]const u8, 1);
    errdefer allocator.free(model_types);
    const model_type0 = try allocator.dupe(u8, "TestModel");
    errdefer allocator.free(model_type0);
    model_types[0] = model_type0;
    const block_ops = try allocator.alloc(types.Op, 2);
    errdefer allocator.free(block_ops);
    block_ops[0] = types.Op{ .op_type = .norm };
    block_ops[1] = types.Op{ .op_type = .add };

    const arch = Architecture{
        .name = name,
        .model_types = model_types,
        .block_ops = block_ops,
    };

    try register(arch);

    const arch_ptr = get("test_arch").?;
    try std.testing.expect(arch_ptr.compiled_program == null);

    // Compile for layer 0 - should create the compiled program
    const program1 = try ensureCompiledForLayer(arch_ptr, 0);
    try std.testing.expect(program1.len > 0);
    try std.testing.expect(arch_ptr.compiled_program != null);

    // Compile for layer 5 - should return same program (homogeneous)
    const program2 = try ensureCompiledForLayer(arch_ptr, 5);
    try std.testing.expectEqual(program1.ptr, program2.ptr);
}

test "ensureCompiledForLayer when not initialized returns error" {
    deinit(); // Ensure not initialized

    const allocator = std.testing.allocator;
    const name = try allocator.dupe(u8, "test");
    defer allocator.free(name);
    const model_types = try allocator.alloc([]const u8, 0);
    defer allocator.free(model_types);
    const block_ops = try allocator.alloc(types.Op, 0);
    defer allocator.free(block_ops);

    var arch = Architecture{
        .name = name,
        .model_types = model_types,
        .block_ops = block_ops,
    };

    const result = ensureCompiledForLayer(&arch, 0);
    try std.testing.expectError(error.RegistryNotInitialized, result);
}

test "ensureCompiledForLayerWithOverride selects variant via override" {
    const allocator = std.testing.allocator;

    init(allocator);
    defer deinit();

    // Build a minimal heterogeneous architecture
    const name = try allocator.dupe(u8, "test_hetero_override");
    errdefer allocator.free(name);
    const model_types = try allocator.alloc([]const u8, 1);
    errdefer allocator.free(model_types);
    const mt0 = try allocator.dupe(u8, "HeteroModel");
    errdefer allocator.free(mt0);
    model_types[0] = mt0;

    // Two variants with different ops
    var variants = try allocator.alloc(types.BlockVariant, 2);
    errdefer allocator.free(variants);

    const ops0 = try allocator.alloc(types.Op, 1);
    errdefer allocator.free(ops0);
    ops0[0] = types.Op{ .op_type = .norm };

    const ops1 = try allocator.alloc(types.Op, 2);
    errdefer allocator.free(ops1);
    ops1[0] = types.Op{ .op_type = .norm };
    ops1[1] = types.Op{ .op_type = .add };

    const vn0 = try allocator.dupe(u8, "mamba");
    errdefer allocator.free(vn0);
    const vn1 = try allocator.dupe(u8, "attention");
    errdefer allocator.free(vn1);
    variants[0] = .{ .name = vn0, .ops = ops0 };
    variants[1] = .{ .name = vn1, .ops = ops1 };

    try register(.{
        .name = name,
        .model_types = model_types,
        .block_variants = variants,
    });
    const arch_ptr = get("test_hetero_override").?;

    // Override says layer 0 → variant 1, layer 1 → variant 0
    const override = &[_]u8{ 1, 0 };

    const prog_layer0 = try ensureCompiledForLayerWithOverride(arch_ptr, 0, override);
    try std.testing.expect(prog_layer0.len > 0);

    const prog_layer1 = try ensureCompiledForLayerWithOverride(arch_ptr, 1, override);
    try std.testing.expect(prog_layer1.len > 0);

    // Different variants produce different compiled programs
    try std.testing.expect(prog_layer0.ptr != prog_layer1.ptr);

    // Same variant returns cached program
    const prog_layer0_again = try ensureCompiledForLayerWithOverride(arch_ptr, 0, override);
    try std.testing.expectEqual(prog_layer0.ptr, prog_layer0_again.ptr);
}

test "ensureCompiledForLayerWithOverride returns InvalidVariantIndex for out-of-bounds" {
    const allocator = std.testing.allocator;

    init(allocator);
    defer deinit();

    const name = try allocator.dupe(u8, "test_hetero_oob");
    errdefer allocator.free(name);
    const model_types = try allocator.alloc([]const u8, 1);
    errdefer allocator.free(model_types);
    const mt0 = try allocator.dupe(u8, "OobModel");
    errdefer allocator.free(mt0);
    model_types[0] = mt0;

    // Only one variant (index 0)
    var variants = try allocator.alloc(types.BlockVariant, 1);
    errdefer allocator.free(variants);
    const ops0 = try allocator.alloc(types.Op, 1);
    errdefer allocator.free(ops0);
    ops0[0] = types.Op{ .op_type = .norm };
    const vn0 = try allocator.dupe(u8, "only");
    errdefer allocator.free(vn0);
    variants[0] = .{ .name = vn0, .ops = ops0 };

    try register(.{
        .name = name,
        .model_types = model_types,
        .block_variants = variants,
    });
    const arch_ptr = get("test_hetero_oob").?;

    // Override points layer 0 to variant index 5, which doesn't exist
    const override = &[_]u8{5};
    const result = ensureCompiledForLayerWithOverride(arch_ptr, 0, override);
    try std.testing.expectError(error.InvalidVariantIndex, result);
}

test "ensureCompiledForLayerWithOverride with null override on homogeneous model" {
    const allocator = std.testing.allocator;

    init(allocator);
    defer deinit();

    const name = try allocator.dupe(u8, "test_homo_override");
    errdefer allocator.free(name);
    const model_types = try allocator.alloc([]const u8, 1);
    errdefer allocator.free(model_types);
    const mt0 = try allocator.dupe(u8, "HomoModel");
    errdefer allocator.free(mt0);
    model_types[0] = mt0;
    const block_ops = try allocator.alloc(types.Op, 1);
    errdefer allocator.free(block_ops);
    block_ops[0] = types.Op{ .op_type = .norm };

    const arch = Architecture{
        .name = name,
        .model_types = model_types,
        .block_ops = block_ops,
    };

    try register(arch);
    const arch_ptr = get("test_homo_override").?;

    // Null override on homogeneous model — should compile and cache normally
    const prog1 = try ensureCompiledForLayerWithOverride(arch_ptr, 0, null);
    try std.testing.expect(prog1.len > 0);

    // Different layer returns same cached program (homogeneous)
    const prog2 = try ensureCompiledForLayerWithOverride(arch_ptr, 3, null);
    try std.testing.expectEqual(prog1.ptr, prog2.ptr);
}

test "loadFromFile can load multiple architectures" {
    const allocator = std.testing.allocator;

    init(allocator);
    defer deinit();

    // Create first architecture file
    const test_json1 =
        \\{
        \\  "name": "multi_test_1",
        \\  "model_types": ["Model1"],
        \\  "block": [{"op": "norm"}]
        \\}
    ;

    const tmp_path1 = "/tmp/test_registry_multi1.json";
    {
        const file = try std.fs.cwd().createFile(tmp_path1, .{});
        defer file.close();
        try file.writeAll(test_json1);
    }
    defer std.fs.cwd().deleteFile(tmp_path1) catch {};

    // Create second architecture file
    const test_json2 =
        \\{
        \\  "name": "multi_test_2",
        \\  "model_types": ["Model2"],
        \\  "block": [{"op": "add"}]
        \\}
    ;

    const tmp_path2 = "/tmp/test_registry_multi2.json";
    {
        const file = try std.fs.cwd().createFile(tmp_path2, .{});
        defer file.close();
        try file.writeAll(test_json2);
    }
    defer std.fs.cwd().deleteFile(tmp_path2) catch {};

    // Load both architectures
    try loadFromFile(tmp_path1);
    try loadFromFile(tmp_path2);

    // Verify both are registered
    try std.testing.expect(has("multi_test_1"));
    try std.testing.expect(has("multi_test_2"));

    const arch1 = get("multi_test_1").?;
    const arch2 = get("multi_test_2").?;
    try std.testing.expectEqualStrings("multi_test_1", arch1.name);
    try std.testing.expectEqualStrings("multi_test_2", arch2.name);
}
