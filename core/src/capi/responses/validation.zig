//! Chat completions request validation C API.
//!
//! Thread safety: NOT thread-safe. All access must be from a single thread.

const std = @import("std");
const responses_mod = @import("../../responses/root.zig");
const conversation_mod = @import("../../responses/conversation/root.zig");

const allocator = std.heap.c_allocator;

const capi_error = @import("../error.zig");
const error_codes = @import("../error_codes.zig");

const completions_protocol = responses_mod.protocol.chat_completions;

const RequestValidationError = error{
    InvalidArgument,
    OutOfMemory,
};

fn invalidRequest(comptime fmt: []const u8, args: anytype) c_int {
    capi_error.setErrorWithCode(.invalid_argument, fmt, args);
    return @intFromEnum(error_codes.ErrorCode.invalid_argument);
}

fn isValidToolName(name: []const u8) bool {
    if (name.len == 0 or name.len > 64) return false;
    for (name) |ch| {
        if (!std.ascii.isAlphanumeric(ch) and ch != '_' and ch != '-') return false;
    }
    return true;
}

fn parseJsonValueOwned(json_slice: []const u8) RequestValidationError!std.json.Parsed(std.json.Value) {
    return std.json.parseFromSlice(std.json.Value, allocator, json_slice, .{}) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => {
            capi_error.setContext("invalid JSON payload", .{});
            return error.InvalidArgument;
        },
    };
}

fn validateToolsJsonShape(json_slice: []const u8) RequestValidationError!void {
    var parsed = try parseJsonValueOwned(json_slice);
    defer parsed.deinit();

    const tools_array = switch (parsed.value) {
        .array => |arr| arr,
        else => {
            capi_error.setContext("`tools` must be an array", .{});
            return error.InvalidArgument;
        },
    };

    for (tools_array.items, 0..) |tool_value, idx| {
        const tool_obj = switch (tool_value) {
            .object => |obj| obj,
            else => {
                capi_error.setContext("`tools[{d}]` must be an object", .{idx});
                return error.InvalidArgument;
            },
        };

        const type_value = tool_obj.get("type") orelse {
            capi_error.setContext("`tools[{d}].type` is required", .{idx});
            return error.InvalidArgument;
        };
        const type_str = switch (type_value) {
            .string => |s| s,
            else => {
                capi_error.setContext("`tools[{d}].type` must be a string", .{idx});
                return error.InvalidArgument;
            },
        };
        if (!std.mem.eql(u8, type_str, "function")) {
            capi_error.setContext("`tools[{d}].type` must be `function`", .{idx});
            return error.InvalidArgument;
        }

        // Accept both nested and flat function shapes.
        if (tool_obj.get("function")) |function_value| {
            const function_obj = switch (function_value) {
                .object => |obj| obj,
                else => {
                    capi_error.setContext("`tools[{d}].function` must be an object", .{idx});
                    return error.InvalidArgument;
                },
            };
            const name_value = function_obj.get("name") orelse {
                capi_error.setContext("`tools[{d}].function.name` is required", .{idx});
                return error.InvalidArgument;
            };
            const name = switch (name_value) {
                .string => |s| s,
                else => {
                    capi_error.setContext("`tools[{d}].function.name` must be a string", .{idx});
                    return error.InvalidArgument;
                },
            };
            if (!isValidToolName(name)) {
                capi_error.setContext("`tools[{d}].function.name` must match ^[a-zA-Z0-9_-]{{1,64}}$", .{idx});
                return error.InvalidArgument;
            }
            if (function_obj.get("parameters")) |parameters| {
                if (parameters != .object) {
                    capi_error.setContext("`tools[{d}].function.parameters` must be an object", .{idx});
                    return error.InvalidArgument;
                }
            }
        } else {
            const name_value = tool_obj.get("name") orelse {
                capi_error.setContext("`tools[{d}].name` is required", .{idx});
                return error.InvalidArgument;
            };
            const name = switch (name_value) {
                .string => |s| s,
                else => {
                    capi_error.setContext("`tools[{d}].name` must be a string", .{idx});
                    return error.InvalidArgument;
                },
            };
            if (!isValidToolName(name)) {
                capi_error.setContext("`tools[{d}].name` must match ^[a-zA-Z0-9_-]{{1,64}}$", .{idx});
                return error.InvalidArgument;
            }
            if (tool_obj.get("parameters")) |parameters| {
                if (parameters != .object) {
                    capi_error.setContext("`tools[{d}].parameters` must be an object", .{idx});
                    return error.InvalidArgument;
                }
            }
        }
    }
}

fn validateToolChoiceJsonShape(json_slice: []const u8) RequestValidationError!void {
    var parsed = try parseJsonValueOwned(json_slice);
    defer parsed.deinit();

    switch (parsed.value) {
        .string => |choice| {
            if (!std.mem.eql(u8, choice, "none") and
                !std.mem.eql(u8, choice, "auto") and
                !std.mem.eql(u8, choice, "required"))
            {
                capi_error.setContext("`tool_choice` string must be one of: none, auto, required", .{});
                return error.InvalidArgument;
            }
            return;
        },
        .object => |obj| {
            const type_value = obj.get("type") orelse {
                capi_error.setContext("`tool_choice.type` is required", .{});
                return error.InvalidArgument;
            };
            const type_str = switch (type_value) {
                .string => |s| s,
                else => {
                    capi_error.setContext("`tool_choice.type` must be a string", .{});
                    return error.InvalidArgument;
                },
            };
            if (!std.mem.eql(u8, type_str, "function")) {
                capi_error.setContext("`tool_choice.type` must be `function`", .{});
                return error.InvalidArgument;
            }

            if (obj.get("function")) |function_value| {
                const function_obj = switch (function_value) {
                    .object => |f| f,
                    else => {
                        capi_error.setContext("`tool_choice.function` must be an object", .{});
                        return error.InvalidArgument;
                    },
                };
                const name_value = function_obj.get("name") orelse {
                    capi_error.setContext("`tool_choice.function.name` is required", .{});
                    return error.InvalidArgument;
                };
                const name = switch (name_value) {
                    .string => |s| s,
                    else => {
                        capi_error.setContext("`tool_choice.function.name` must be a string", .{});
                        return error.InvalidArgument;
                    },
                };
                if (!isValidToolName(name)) {
                    capi_error.setContext("`tool_choice.function.name` must match ^[a-zA-Z0-9_-]{{1,64}}$", .{});
                    return error.InvalidArgument;
                }
                return;
            }

            if (obj.get("name")) |name_value| {
                const name = switch (name_value) {
                    .string => |s| s,
                    else => {
                        capi_error.setContext("`tool_choice.name` must be a string", .{});
                        return error.InvalidArgument;
                    },
                };
                if (!isValidToolName(name)) {
                    capi_error.setContext("`tool_choice.name` must match ^[a-zA-Z0-9_-]{{1,64}}$", .{});
                    return error.InvalidArgument;
                }
                return;
            }

            capi_error.setContext("`tool_choice` of type `function` requires a function name", .{});
            return error.InvalidArgument;
        },
        else => {
            capi_error.setContext("`tool_choice` must be a string or object", .{});
            return error.InvalidArgument;
        },
    }
}

/// Validate `/v1/chat/completions` request semantics in core.
///
/// This keeps request-contract logic in Zig (core) while Rust stays a thin
/// HTTP boundary layer. Optional scalar fields are passed as:
/// - max_tokens / max_completion_tokens: explicit `has_*` flags
/// - floating fields: `NaN` means omitted
pub export fn talu_completions_validate_request(
    messages_json_ptr: ?[*]const u8,
    messages_json_len: usize,
    has_max_tokens: usize,
    max_tokens: i64,
    has_max_completion_tokens: usize,
    max_completion_tokens: i64,
    temperature: f64,
    top_p: f64,
    presence_penalty: f64,
    frequency_penalty: f64,
    tools_json_ptr: ?[*]const u8,
    tools_json_len: usize,
    tool_choice_json_ptr: ?[*]const u8,
    tool_choice_json_len: usize,
) callconv(.c) c_int {
    capi_error.clearError();

    const messages_ptr = messages_json_ptr orelse {
        return invalidRequest("messages_json_ptr is null", .{});
    };
    const messages_json = messages_ptr[0..messages_json_len];

    const conv = conversation_mod.Conversation.init(allocator) catch {
        capi_error.setErrorWithCode(.out_of_memory, "failed to allocate temporary conversation for validation", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };
    defer conv.deinit();

    completions_protocol.parse(conv, messages_json) catch |err| {
        capi_error.setErrorWithCode(
            .invalid_argument,
            "invalid messages for completions request: {s}",
            .{@errorName(err)},
        );
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    if (conv.len() == 0) {
        return invalidRequest("messages array is empty", .{});
    }

    if (has_max_tokens > 1) {
        return invalidRequest("`has_max_tokens` must be 0 or 1", .{});
    }
    if (has_max_completion_tokens > 1) {
        return invalidRequest("`has_max_completion_tokens` must be 0 or 1", .{});
    }

    if (has_max_tokens == 1 and max_tokens < 1) {
        return invalidRequest("`max_tokens` must be at least 1", .{});
    }
    if (has_max_completion_tokens == 1 and max_completion_tokens < 1) {
        return invalidRequest("`max_completion_tokens` must be at least 1", .{});
    }

    if (!std.math.isNan(temperature) and (temperature < 0.0 or temperature > 2.0)) {
        return invalidRequest("`temperature` must be between 0 and 2", .{});
    }
    if (!std.math.isNan(top_p) and (top_p < 0.0 or top_p > 1.0)) {
        return invalidRequest("`top_p` must be between 0 and 1", .{});
    }
    if (!std.math.isNan(presence_penalty) and (presence_penalty < -2.0 or presence_penalty > 2.0)) {
        return invalidRequest("`presence_penalty` must be between -2 and 2", .{});
    }
    if (!std.math.isNan(frequency_penalty) and (frequency_penalty < -2.0 or frequency_penalty > 2.0)) {
        return invalidRequest("`frequency_penalty` must be between -2 and 2", .{});
    }

    if (tools_json_ptr == null and tools_json_len != 0) {
        return invalidRequest("tools_json_ptr is null but tools_json_len > 0", .{});
    }
    if (tools_json_ptr) |ptr| {
        const tools_json = ptr[0..tools_json_len];
        validateToolsJsonShape(tools_json) catch |err| switch (err) {
            error.InvalidArgument => {
                capi_error.setErrorWithCode(.invalid_argument, "invalid tools for completions request", .{});
                return @intFromEnum(error_codes.ErrorCode.invalid_argument);
            },
            error.OutOfMemory => {
                capi_error.setErrorWithCode(.out_of_memory, "out of memory while validating tools", .{});
                return @intFromEnum(error_codes.ErrorCode.out_of_memory);
            },
        };
    }

    if (tool_choice_json_ptr == null and tool_choice_json_len != 0) {
        return invalidRequest("tool_choice_json_ptr is null but tool_choice_json_len > 0", .{});
    }
    if (tool_choice_json_ptr) |ptr| {
        const tool_choice_json = ptr[0..tool_choice_json_len];
        validateToolChoiceJsonShape(tool_choice_json) catch |err| switch (err) {
            error.InvalidArgument => {
                capi_error.setErrorWithCode(.invalid_argument, "invalid tool_choice for completions request", .{});
                return @intFromEnum(error_codes.ErrorCode.invalid_argument);
            },
            error.OutOfMemory => {
                capi_error.setErrorWithCode(.out_of_memory, "out of memory while validating tool_choice", .{});
                return @intFromEnum(error_codes.ErrorCode.out_of_memory);
            },
        };
    }

    return 0;
}
