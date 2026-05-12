//! Responses, chat, router, config, backend, and validation C API aggregation.
//!
//! This module groups the C API surface that backs chat sessions, OpenAI
//! Responses-style item access, request validation, embeddings, and backend
//! management.

const std = @import("std");

pub const types = @import("types.zig");
pub const conversation = @import("conversation.zig");
pub const items = @import("items.zig");
pub const write = @import("write.zig");
pub const chat = @import("chat.zig");
pub const validation = @import("validation.zig");
pub const engine = @import("engine.zig");

const capi_error = @import("../error.zig");
const error_codes = @import("../error_codes.zig");
const conversation_mod = @import("../../responses/conversation/root.zig");
const Chat = conversation_mod.Chat;

// Shared handles and ABI structs.
pub const ResponsesHandle = types.ResponsesHandle;
pub const ChatHandle = types.ChatHandle;
pub const ChatCreateOptions = types.ChatCreateOptions;
pub const CItem = types.CItem;
pub const CMessageItem = types.CMessageItem;
pub const CFunctionCallItem = types.CFunctionCallItem;
pub const CFunctionCallOutputItem = types.CFunctionCallOutputItem;
pub const CReasoningItem = types.CReasoningItem;
pub const CItemReferenceItem = types.CItemReferenceItem;
pub const CContentPart = types.CContentPart;

// Conversation lifecycle, serialization, cloning, and status.
pub const talu_responses_create = conversation.talu_responses_create;
pub const talu_responses_create_with_session = conversation.talu_responses_create_with_session;
pub const talu_responses_free = conversation.talu_responses_free;
pub const talu_responses_to_responses_json = conversation.talu_responses_to_responses_json;
pub const talu_responses_to_completions_json = conversation.talu_responses_to_completions_json;
pub const talu_responses_clear = conversation.talu_responses_clear;
pub const talu_responses_clear_keeping_system = conversation.talu_responses_clear_keeping_system;
pub const talu_responses_load_completions_json = conversation.talu_responses_load_completions_json;
pub const talu_responses_load_responses_json = conversation.talu_responses_load_responses_json;
pub const talu_responses_clone = conversation.talu_responses_clone;
pub const talu_responses_clone_prefix = conversation.talu_responses_clone_prefix;
pub const talu_responses_truncate_after = conversation.talu_responses_truncate_after;
pub const talu_responses_set_item_status = conversation.talu_responses_set_item_status;

// Item and content accessors.
pub const talu_responses_item_count = items.talu_responses_item_count;
pub const talu_responses_item_type = items.talu_responses_item_type;
pub const talu_responses_get_item = items.talu_responses_get_item;
pub const talu_responses_item_get_generation_json = items.talu_responses_item_get_generation_json;
pub const talu_responses_item_as_message = items.talu_responses_item_as_message;
pub const talu_responses_item_as_function_call = items.talu_responses_item_as_function_call;
pub const talu_responses_item_as_function_call_output = items.talu_responses_item_as_function_call_output;
pub const talu_responses_item_as_reasoning = items.talu_responses_item_as_reasoning;
pub const talu_responses_item_as_item_reference = items.talu_responses_item_as_item_reference;
pub const talu_responses_item_message_content_count = items.talu_responses_item_message_content_count;
pub const talu_responses_item_message_get_content = items.talu_responses_item_message_get_content;
pub const talu_responses_item_reasoning_content_count = items.talu_responses_item_reasoning_content_count;
pub const talu_responses_item_reasoning_get_content = items.talu_responses_item_reasoning_get_content;
pub const talu_responses_item_reasoning_summary_count = items.talu_responses_item_reasoning_summary_count;
pub const talu_responses_item_reasoning_get_summary = items.talu_responses_item_reasoning_get_summary;
pub const talu_responses_item_fco_get_part = items.talu_responses_item_fco_get_part;

// Write operations.
pub const talu_responses_append_message = write.talu_responses_append_message;
pub const talu_responses_append_message_hidden = write.talu_responses_append_message_hidden;
pub const talu_responses_append_function_call = write.talu_responses_append_function_call;
pub const talu_responses_append_function_call_output = write.talu_responses_append_function_call_output;
pub const talu_responses_append_text_content = write.talu_responses_append_text_content;
pub const talu_responses_pop = write.talu_responses_pop;
pub const talu_responses_remove = write.talu_responses_remove;
pub const talu_responses_insert_message = write.talu_responses_insert_message;
pub const talu_responses_insert_message_hidden = write.talu_responses_insert_message_hidden;

// Chat API.
pub const talu_chat_create = chat.talu_chat_create;
pub const talu_chat_create_with_system = chat.talu_chat_create_with_system;
pub const talu_chat_create_with_session = chat.talu_chat_create_with_session;
pub const talu_chat_create_with_system_and_session = chat.talu_chat_create_with_system_and_session;
pub const talu_chat_free = chat.talu_chat_free;
pub const talu_chat_get_conversation = chat.talu_chat_get_conversation;
pub const talu_session_id_new = chat.talu_session_id_new;
pub const talu_chat_set_ttl_ts = chat.talu_chat_set_ttl_ts;
pub const talu_chat_set_system = chat.talu_chat_set_system;
pub const talu_chat_get_system = chat.talu_chat_get_system;
pub const talu_chat_get_prompt_id = chat.talu_chat_get_prompt_id;
pub const talu_chat_set_prompt_id = chat.talu_chat_set_prompt_id;
pub const talu_chat_set_tools = chat.talu_chat_set_tools;
pub const talu_chat_get_tools = chat.talu_chat_get_tools;
pub const talu_chat_set_tool_choice = chat.talu_chat_set_tool_choice;
pub const talu_chat_get_tool_choice = chat.talu_chat_get_tool_choice;
pub const talu_chat_to_json = chat.talu_chat_to_json;
pub const talu_chat_set_messages = chat.talu_chat_set_messages;
pub const talu_chat_load_completions_json = chat.talu_chat_load_completions_json;
pub const talu_chat_count_tokens = chat.talu_chat_count_tokens;
pub const talu_chat_max_context_length = chat.talu_chat_max_context_length;

// Chat completions request validation.
pub const talu_completions_validate_request = validation.talu_completions_validate_request;

// Router/config/backend APIs and ABI types.
pub const CLogitBiasEntry = engine.CLogitBiasEntry;
pub const RouterGenerateConfig = engine.RouterGenerateConfig;
pub const CPoolingStrategy = engine.CPoolingStrategy;
pub const BackendType = engine.BackendType;
pub const BackendUnion = engine.BackendUnion;
pub const TaluModelSpec = engine.TaluModelSpec;
pub const TaluCapabilities = engine.TaluCapabilities;
pub const TaluCanonicalSpec = engine.TaluCanonicalSpec;
pub const TaluInferenceBackend = engine.TaluInferenceBackend;
pub const BackendCreateOptions = engine.BackendCreateOptions;
pub const CModelInfo = engine.CModelInfo;
pub const talu_router_close_all = engine.talu_router_close_all;
pub const talu_router_embedding_dim = engine.talu_router_embedding_dim;
pub const talu_router_embed = engine.talu_router_embed;
pub const talu_router_embedding_free = engine.talu_router_embedding_free;
pub const talu_config_canonicalize = engine.talu_config_canonicalize;
pub const talu_config_get_view = engine.talu_config_get_view;
pub const talu_config_free = engine.talu_config_free;
pub const talu_backend_get_capabilities = engine.talu_backend_get_capabilities;
pub const talu_backend_create_from_canonical = engine.talu_backend_create_from_canonical;
pub const talu_backend_free = engine.talu_backend_free;
pub const talu_backend_synchronize = engine.talu_backend_synchronize;
pub const talu_backend_model_info = engine.talu_backend_model_info;

comptime {
    // Force all moved C exports to be emitted even when callers reach them
    // only through dynamic FFI instead of Zig references.
    _ = &talu_responses_create;
    _ = &talu_responses_create_with_session;
    _ = &talu_responses_free;
    _ = &talu_responses_to_responses_json;
    _ = &talu_responses_to_completions_json;
    _ = &talu_responses_clear;
    _ = &talu_responses_clear_keeping_system;
    _ = &talu_responses_load_completions_json;
    _ = &talu_responses_load_responses_json;
    _ = &talu_responses_clone;
    _ = &talu_responses_clone_prefix;
    _ = &talu_responses_truncate_after;
    _ = &talu_responses_set_item_status;

    _ = &talu_responses_item_count;
    _ = &talu_responses_item_type;
    _ = &talu_responses_get_item;
    _ = &talu_responses_item_get_generation_json;
    _ = &talu_responses_item_as_message;
    _ = &talu_responses_item_as_function_call;
    _ = &talu_responses_item_as_function_call_output;
    _ = &talu_responses_item_as_reasoning;
    _ = &talu_responses_item_as_item_reference;
    _ = &talu_responses_item_message_content_count;
    _ = &talu_responses_item_message_get_content;
    _ = &talu_responses_item_reasoning_content_count;
    _ = &talu_responses_item_reasoning_get_content;
    _ = &talu_responses_item_reasoning_summary_count;
    _ = &talu_responses_item_reasoning_get_summary;
    _ = &talu_responses_item_fco_get_part;

    _ = &talu_responses_append_message;
    _ = &talu_responses_append_message_hidden;
    _ = &talu_responses_append_function_call;
    _ = &talu_responses_append_function_call_output;
    _ = &talu_responses_append_text_content;
    _ = &talu_responses_pop;
    _ = &talu_responses_remove;
    _ = &talu_responses_insert_message;
    _ = &talu_responses_insert_message_hidden;

    _ = &talu_chat_create;
    _ = &talu_chat_create_with_system;
    _ = &talu_chat_create_with_session;
    _ = &talu_chat_create_with_system_and_session;
    _ = &talu_chat_free;
    _ = &talu_chat_get_conversation;
    _ = &talu_session_id_new;
    _ = &talu_chat_set_ttl_ts;
    _ = &talu_chat_set_system;
    _ = &talu_chat_get_system;
    _ = &talu_chat_get_prompt_id;
    _ = &talu_chat_set_prompt_id;
    _ = &talu_chat_set_tools;
    _ = &talu_chat_get_tools;
    _ = &talu_chat_set_tool_choice;
    _ = &talu_chat_get_tool_choice;
    _ = &talu_chat_to_json;
    _ = &talu_chat_set_messages;
    _ = &talu_chat_load_completions_json;
    _ = &talu_chat_count_tokens;
    _ = &talu_chat_max_context_length;

    _ = &talu_completions_validate_request;
    _ = &talu_router_close_all;
    _ = &talu_router_embedding_dim;
    _ = &talu_router_embed;
    _ = &talu_router_embedding_free;
    _ = &talu_config_canonicalize;
    _ = &talu_config_get_view;
    _ = &talu_config_free;
    _ = &talu_backend_get_capabilities;
    _ = &talu_backend_create_from_canonical;
    _ = &talu_backend_free;
    _ = &talu_backend_synchronize;
    _ = &talu_backend_model_info;
}

// =============================================================================
// Tests
// =============================================================================

test "CItem struct size and alignment" {
    // Verify ABI stability
    try std.testing.expectEqual(@as(usize, 56), @sizeOf(CItem));
    try std.testing.expectEqual(@as(usize, 8), @alignOf(CItem));
}

test "CMessageItem struct size and alignment" {
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(CMessageItem));
    try std.testing.expectEqual(@as(usize, 8), @alignOf(CMessageItem));
}

test "CContentPart struct size and alignment" {
    // Size increased from 56 to 72 with addition of quaternary_ptr/len for code_blocks_json
    try std.testing.expectEqual(@as(usize, 72), @sizeOf(CContentPart));
    try std.testing.expectEqual(@as(usize, 8), @alignOf(CContentPart));
}

test "talu_responses_create and free" {
    const handle = talu_responses_create();
    try std.testing.expect(handle != null);
    defer talu_responses_free(handle);

    const count = talu_responses_item_count(handle);
    try std.testing.expectEqual(@as(usize, 0), count);
}

test "talu_responses_item_type returns 255 for invalid index" {
    const handle = talu_responses_create();
    defer talu_responses_free(handle);

    const item_type = talu_responses_item_type(handle, 0);
    try std.testing.expectEqual(@as(u8, 255), item_type);
}

test "talu_responses_get_item returns error for invalid index" {
    const handle = talu_responses_create();
    defer talu_responses_free(handle);

    var item: CItem = undefined;
    const result = talu_responses_get_item(handle, 0, &item);
    try std.testing.expect(result != 0); // Should return error
}

// =============================================================================
// ABI Compatibility: Struct Size Validation
// =============================================================================
//
// ABI sizes are validated at compile time via comptime assertions.
// See core/src/capi/abi.zig for the canonical size definitions.
// Python bindings validate against the same sizes in tests/compat/abi/.

// =============================================================================
// Fuzz Tests
// =============================================================================

test "fuzz talu_responses_load_responses_json" {
    try std.testing.fuzz({}, struct {
        fn testOne(_: void, input: []const u8) !void {
            const handle = talu_responses_create();
            if (handle == null) return;
            defer talu_responses_free(handle);
            const alloc = std.testing.allocator;
            const json_z = try alloc.dupeZ(u8, input);
            defer alloc.free(json_z);
            const rc = talu_responses_load_responses_json(handle, json_z.ptr);
            if (rc != 0) {
                try std.testing.expect(capi_error.talu_last_error_code() != 0);
                try std.testing.expect(capi_error.talu_last_error() != null);
            }
        }
    }.testOne, .{});
}

test "fuzz talu_responses_load_completions_json" {
    try std.testing.fuzz({}, struct {
        fn testOne(_: void, input: []const u8) !void {
            const handle = talu_responses_create();
            if (handle == null) return;
            defer talu_responses_free(handle);
            const alloc = std.testing.allocator;
            const json_z = try alloc.dupeZ(u8, input);
            defer alloc.free(json_z);
            const rc = talu_responses_load_completions_json(handle, json_z.ptr);
            if (rc != 0) {
                try std.testing.expect(capi_error.talu_last_error_code() != 0);
                try std.testing.expect(capi_error.talu_last_error() != null);
            }
        }
    }.testOne, .{});
}

test "fuzz talu_chat_set_messages" {
    try std.testing.fuzz({}, struct {
        fn testOne(_: void, input: []const u8) !void {
            const chat_handle = talu_chat_create(null) orelse return;
            defer talu_chat_free(chat_handle);

            const alloc = std.testing.allocator;
            const json_z = try alloc.dupeZ(u8, input);
            defer alloc.free(json_z);
            const rc = talu_chat_set_messages(chat_handle, json_z.ptr);
            if (rc != 0) {
                try std.testing.expect(capi_error.talu_last_error_code() != 0);
                try std.testing.expect(capi_error.talu_last_error() != null);
            }
        }
    }.testOne, .{});
}

test "talu_responses_load_responses_json invalid json maps to invalid_argument" {
    const handle = talu_responses_create();
    if (handle == null) return error.OutOfMemory;
    defer talu_responses_free(handle);

    const rc = talu_responses_load_responses_json(handle, "{");
    try std.testing.expect(rc != 0);
    try std.testing.expectEqual(@as(i32, @intFromEnum(error_codes.ErrorCode.invalid_argument)), capi_error.talu_last_error_code());
}

test "talu_responses_load_completions_json invalid json maps to invalid_argument" {
    const handle = talu_responses_create();
    if (handle == null) return error.OutOfMemory;
    defer talu_responses_free(handle);

    const rc = talu_responses_load_completions_json(handle, "{");
    try std.testing.expect(rc != 0);
    try std.testing.expectEqual(@as(i32, @intFromEnum(error_codes.ErrorCode.invalid_argument)), capi_error.talu_last_error_code());
}

test "talu_chat_set_messages invalid json maps to invalid_argument" {
    const chat_handle = talu_chat_create(null) orelse return error.OutOfMemory;
    defer talu_chat_free(chat_handle);

    const rc = talu_chat_set_messages(chat_handle, "{");
    try std.testing.expect(rc != 0);
    try std.testing.expectEqual(@as(i32, @intFromEnum(error_codes.ErrorCode.invalid_argument)), capi_error.talu_last_error_code());
}

test "talu_chat_load_completions_json null handle returns invalid_handle" {
    const rc = talu_chat_load_completions_json(null, null, 0);
    try std.testing.expect(rc != 0);
    try std.testing.expectEqual(@as(i32, @intFromEnum(error_codes.ErrorCode.invalid_handle)), capi_error.talu_last_error_code());
}

test "talu_chat_load_completions_json null json_ptr returns invalid_argument" {
    const chat_handle = talu_chat_create(null) orelse return error.OutOfMemory;
    defer talu_chat_free(chat_handle);

    const rc = talu_chat_load_completions_json(chat_handle, null, 0);
    try std.testing.expect(rc != 0);
    try std.testing.expectEqual(@as(i32, @intFromEnum(error_codes.ErrorCode.invalid_argument)), capi_error.talu_last_error_code());
}

test "talu_chat_load_completions_json valid messages loads into conversation" {
    const chat_handle = talu_chat_create(null) orelse return error.OutOfMemory;
    defer talu_chat_free(chat_handle);

    const json = "[{\"role\":\"user\",\"content\":\"Hello\"}]";
    const rc = talu_chat_load_completions_json(chat_handle, json.ptr, json.len);
    try std.testing.expectEqual(@as(i32, 0), rc);
    const chat_state: *Chat = @ptrCast(@alignCast(chat_handle));
    try std.testing.expect(chat_state.len() > 0);
}

test "talu_chat_load_completions_json invalid json returns error" {
    const chat_handle = talu_chat_create(null) orelse return error.OutOfMemory;
    defer talu_chat_free(chat_handle);

    const json = "{not valid";
    const rc = talu_chat_load_completions_json(chat_handle, json.ptr, json.len);
    try std.testing.expect(rc != 0);
}
