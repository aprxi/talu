//! C API for Architecture Registration
//!
//! Allows Python (or other languages) to register custom architectures at runtime.

const std = @import("std");
const graph = @import("../graph/root.zig");
const ffi = @import("../helpers/ffi.zig");
const capi_error = @import("error.zig");
const error_codes = @import("error_codes.zig");

// Use c_allocator for consistency with other capi modules
const allocator = std.heap.c_allocator;

fn setErr(comptime context: []const u8, err: anyerror) void {
    capi_error.setError(err, "{s}: {s}", .{ context, @errorName(err) });
}

fn setMsg(comptime fmt: []const u8, args: anytype) void {
    capi_error.setError(error.InvalidArgument, fmt, args);
}

/// Initialize the runtime architecture registry.
/// Must be called before any architecture registration.
/// Safe to call multiple times.
pub export fn talu_arch_init() callconv(.c) void {
    graph.init(allocator);
}

/// Deinitialize and free all registered architectures.
pub export fn talu_arch_deinit() callconv(.c) void {
    graph.deinit();
}

/// Register a custom architecture from JSON definition.
///
/// Expected JSON format:
/// ```json
/// {
///   "name": "my_model",
///   "model_types": ["my_model", "my_model_v2"],
///   "block": [
///     {"op": "norm", "name": "input_layernorm"},
///     {"op": "attention", "qk_norm": false},
///     {"op": "add", "scale": 1.0},
///     {"op": "norm", "name": "post_attention_layernorm"},
///     {"op": "mlp", "activation": "silu"},
///     {"op": "add", "scale": 1.0}
///   ]
/// }
/// ```
///
/// Returns 0 on success, negative error code on failure.
pub export fn talu_arch_register(
    json_cstr: [*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const json_text = std.mem.span(json_cstr);

    // Parse JSON into Architecture
    const architecture = graph.parseFromJson(allocator, json_text) catch |err| {
        setErr("arch_register: JSON parse failed", err);
        return @intFromEnum(error_codes.errorToCode(err));
    };

    // Register
    graph.register(architecture) catch |err| {
        setErr("arch_register: registration failed", err);
        return @intFromEnum(error_codes.errorToCode(err));
    };

    return 0;
}

/// Check if an architecture is registered.
/// Returns false if name is null or not found.
pub export fn talu_arch_exists(
    name_cstr: [*:0]const u8,
) callconv(.c) bool {
    // Note: Query function - returns false rather than error for not-found
    const arch_name = std.mem.span(name_cstr);
    return graph.has(arch_name);
}

/// Get the number of registered runtime architectures.
/// Returns 0 on error (check talu_error_message for details).
pub export fn talu_arch_count() callconv(.c) usize {
    capi_error.clearError();
    const names = graph.listNames(allocator) catch |err| {
        setErr("arch_count failed", err);
        return 0;
    };
    defer allocator.free(names);
    return names.len;
}

/// List all registered architectures as JSON array.
/// On success, writes a null-terminated string to out_json.
/// Caller must free the returned string with talu_arch_free_string.
pub export fn talu_arch_list(out_json: *?[*:0]u8) callconv(.c) i32 {
    capi_error.clearError();
    out_json.* = null;

    // Get registered architecture names
    const arch_names = graph.listNames(allocator) catch |err| {
        setErr("arch_list", err);
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer allocator.free(arch_names);

    // Build JSON array of architecture names
    const result = ffi.buildJsonStringArray(allocator, &.{arch_names}) catch {
        setMsg("arch_list: out of memory", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };
    out_json.* = result.ptr;
    return 0;
}

/// Free a string returned by talu_arch_list.
pub export fn talu_arch_free_string(ptr: ?[*:0]u8) callconv(.c) void {
    if (ptr) |arch_cstr| {
        const length = std.mem.len(arch_cstr);
        allocator.free(arch_cstr[0 .. length + 1]);
    }
}

/// Check if a model_type string maps to a runtime-registered architecture.
/// Returns the architecture name if found, null otherwise.
/// Caller must NOT free the returned string (it's owned by the registry).
pub export fn talu_arch_detect(
    model_type_cstr: [*:0]const u8,
    out_name: *?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();
    out_name.* = null;
    const model_type_name = std.mem.span(model_type_cstr);

    if (graph.detectFromModelType(model_type_name)) |registered_arch| {
        // Return pointer to the name stored in registry
        // This is safe because the registry owns the string
        out_name.* = @ptrCast(registered_arch.name.ptr);
        return 0;
    }

    return 0;
}

// =============================================================================
// Tests
// =============================================================================

test "talu_arch_register and talu_arch_detect" {
    // Initialize
    talu_arch_init();
    defer talu_arch_deinit();

    // Register a test architecture
    const arch_json =
        \\{"name": "test_arch", "model_types": ["test_model"], "block": [{"op": "norm"}, {"op": "multihead_attention"}, {"op": "add"}, {"op": "norm"}, {"op": "mlp"}, {"op": "add"}]}
    ;
    const register_rc = talu_arch_register(arch_json);
    try std.testing.expectEqual(@as(i32, 0), register_rc);

    // Check it exists
    try std.testing.expect(talu_arch_exists("test_arch"));
    try std.testing.expect(!talu_arch_exists("nonexistent"));

    // Detect from model_type
    var detected_name: ?[*:0]const u8 = null;
    const detect_status = talu_arch_detect("test_model", &detected_name);
    try std.testing.expectEqual(@as(i32, 0), detect_status);
    try std.testing.expect(detected_name != null);

    // Count
    try std.testing.expectEqual(@as(usize, 1), talu_arch_count());

    // List
    var list_cstr: ?[*:0]u8 = null;
    const list_rc = talu_arch_list(&list_cstr);
    try std.testing.expectEqual(@as(i32, 0), list_rc);
    try std.testing.expect(list_cstr != null);
    defer talu_arch_free_string(list_cstr);

    const list_text = std.mem.span(list_cstr.?);
    try std.testing.expect(std.mem.indexOf(u8, list_text, "\"test_arch\"") != null);
}

test "fuzz talu_arch_register" {
    // Fuzz architecture registration with arbitrary JSON input.
    try std.testing.fuzz({}, struct {
        fn testOne(_: void, input: []const u8) !void {
            const alloc = std.testing.allocator;
            const json_z = try alloc.allocSentinel(u8, input.len, 0);
            defer alloc.free(json_z[0 .. input.len + 1]);
            @memcpy(json_z[0..input.len], input);

            // Reset registry each iteration to avoid unbounded growth.
            talu_arch_deinit();
            talu_arch_init();

            _ = talu_arch_register(json_z.ptr);
            if (capi_error.talu_last_error_code() != 0) {
                try std.testing.expect(capi_error.talu_last_error() != null);
            }
        }
    }.testOne, .{});
}

test "talu_arch_register invalid json maps to invalid_argument" {
    talu_arch_init();
    defer talu_arch_deinit();

    const rc = talu_arch_register("{");
    try std.testing.expect(rc != 0);
    try std.testing.expectEqual(@as(i32, @intFromEnum(error_codes.ErrorCode.invalid_argument)), capi_error.talu_last_error_code());
}
