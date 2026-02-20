//! Chat Template Rendering
//!
//! Render Jinja2 chat templates with JSON message arrays.
//! Supports multi-turn conversations, tool calls, and custom context.

const std = @import("std");
const io = @import("../io/root.zig");
const template_engine = @import("root.zig");

pub const Error = template_engine.Error || error{InvalidMessages};

/// Render a chat template with a JSON array of messages.
///
/// Supports full multi-turn conversations:
/// - system, user, assistant messages
/// - tool_calls and tool responses
/// - add_generation_prompt controls whether to append assistant prompt
pub fn render(
    allocator: std.mem.Allocator,
    template: []const u8,
    messages_json: []const u8,
    bos_token: []const u8,
    eos_token: []const u8,
    add_generation_prompt: bool,
) Error![]const u8 {
    return renderWithContext(allocator, template, messages_json, bos_token, eos_token, add_generation_prompt, null);
}

/// Render a chat template with messages and optional extra context variables.
///
/// Like render(), but allows injecting additional template variables beyond
/// the standard messages, bos_token, eos_token, and add_generation_prompt.
///
/// Args:
///   allocator: Memory allocator
///   template: Jinja2 template string
///   messages_json: JSON array of messages
///   bos_token: Beginning of sequence token string
///   eos_token: End of sequence token string
///   add_generation_prompt: Whether to add the assistant prompt suffix
///   extra_context_json: Optional JSON object with additional template variables.
///       These are merged into the template context, allowing templates to access
///       custom variables like {{ tools }}, {{ system_prompt }}, {{ date }}, etc.
///       Must be a JSON object (not array).
pub fn renderWithContext(
    allocator: std.mem.Allocator,
    template: []const u8,
    messages_json: []const u8,
    bos_token: []const u8,
    eos_token: []const u8,
    add_generation_prompt: bool,
    extra_context_json: ?[]const u8,
) Error![]const u8 {
    var template_context = template_engine.TemplateParser.init(allocator);
    defer template_context.deinit();

    // Parse messages JSON array
    const parsed_messages = io.json.parseValue(allocator, messages_json, .{
        .max_size_bytes = 50 * 1024 * 1024,
        .max_value_bytes = 50 * 1024 * 1024,
        .max_string_bytes = 50 * 1024 * 1024,
    }) catch |err| {
        return switch (err) {
            error.InputTooLarge => error.InvalidMessages,
            error.InputTooDeep => error.InvalidMessages,
            error.StringTooLong => error.InvalidMessages,
            error.InvalidJson => error.InvalidMessages,
            error.OutOfMemory => error.OutOfMemory,
        };
    };
    defer parsed_messages.deinit();

    // Convert JSON array to template Value array using arena allocator
    // so it gets freed when template_context.deinit() is called
    const arena_alloc = template_context.arenaAllocator();
    const message_values = try jsonToValue(arena_alloc, parsed_messages.value);
    try template_context.set("messages", message_values);
    try template_context.set("add_generation_prompt", .{ .boolean = add_generation_prompt });
    try template_context.set("bos_token", .{ .string = bos_token });
    try template_context.set("eos_token", .{ .string = eos_token });

    // Mark strftime_now as defined (actual function is handled as builtin)
    // This allows templates to check `if strftime_now is defined`
    try template_context.set("strftime_now", .{ .boolean = true });

    // Parse and merge extra context if provided
    if (extra_context_json) |extra_json| {
        const parsed_extra = io.json.parseValue(allocator, extra_json, .{
            .max_size_bytes = 50 * 1024 * 1024,
            .max_value_bytes = 50 * 1024 * 1024,
            .max_string_bytes = 50 * 1024 * 1024,
        }) catch |err| {
            return switch (err) {
                error.InputTooLarge => error.InvalidMessages,
                error.InputTooDeep => error.InvalidMessages,
                error.StringTooLong => error.InvalidMessages,
                error.InvalidJson => error.InvalidMessages,
                error.OutOfMemory => error.OutOfMemory,
            };
        };
        defer parsed_extra.deinit();

        // Extra context must be an object
        if (parsed_extra.value != .object) {
            return error.InvalidMessages;
        }

        // Merge each key-value pair into the template context
        var iter = parsed_extra.value.object.iterator();
        while (iter.next()) |entry| {
            const key = entry.key_ptr.*;
            const value = try jsonToValue(arena_alloc, entry.value_ptr.*);
            try template_context.set(key, value);
        }
    }

    return template_engine.render(allocator, template, &template_context);
}

/// Convert std.json.Value to template_engine.TemplateInput
fn jsonToValue(allocator: std.mem.Allocator, json_value: std.json.Value) Error!template_engine.TemplateInput {
    switch (json_value) {
        .null => return .none,
        .bool => |bool_value| return .{ .boolean = bool_value },
        .integer => |integer_value| return .{ .integer = integer_value },
        .float => |float_value| return .{ .float = float_value },
        .string => |string_value| return .{ .string = string_value },
        .array => |array_value| {
            const value_items = try allocator.alloc(template_engine.TemplateInput, array_value.items.len);
            errdefer allocator.free(value_items);
            for (array_value.items, 0..) |item, idx| {
                value_items[idx] = try jsonToValue(allocator, item);
            }
            return .{ .array = value_items };
        },
        .object => |object_value| {
            var object_map = std.StringHashMapUnmanaged(template_engine.TemplateInput){};
            var object_iter = object_value.iterator();
            while (object_iter.next()) |entry| {
                const mapped_value = try jsonToValue(allocator, entry.value_ptr.*);
                try object_map.put(allocator, entry.key_ptr.*, mapped_value);
            }
            return .{ .map = object_map };
        },
        .number_string => return .none, // Not typically used
    }
}

// ============================================================================
// Tests
// ============================================================================

test "render basic user/assistant messages" {
    const allocator = std.testing.allocator;

    const template = "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}";
    const messages_json =
        \\[
        \\  {"role": "user", "content": "Hello!"},
        \\  {"role": "assistant", "content": "Hi there!"}
        \\]
    ;

    const result = try render(allocator, template, messages_json, "", "", false);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("user: Hello!\nassistant: Hi there!\n", result);
}

test "render with system message" {
    const allocator = std.testing.allocator;

    const template = "{% for message in messages %}[{{ message.role }}] {{ message.content }}\n{% endfor %}";
    const messages_json =
        \\[
        \\  {"role": "system", "content": "You are a helpful assistant."},
        \\  {"role": "user", "content": "What is 2+2?"},
        \\  {"role": "assistant", "content": "4"}
        \\]
    ;

    const result = try render(allocator, template, messages_json, "", "", false);
    defer allocator.free(result);

    try std.testing.expect(std.mem.indexOf(u8, result, "[system] You are a helpful assistant.") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "[user] What is 2+2?") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "[assistant] 4") != null);
}

test "render with add_generation_prompt true" {
    const allocator = std.testing.allocator;

    const template =
        \\{% for message in messages %}{{ message.role }}: {{ message.content }}
        \\{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}
    ;
    const messages_json =
        \\[
        \\  {"role": "user", "content": "Hello"}
        \\]
    ;

    const result = try render(allocator, template, messages_json, "", "", true);
    defer allocator.free(result);

    try std.testing.expect(std.mem.indexOf(u8, result, "user: Hello") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "assistant: ") != null);
}

test "render with add_generation_prompt false" {
    const allocator = std.testing.allocator;

    const template =
        \\{% for message in messages %}{{ message.role }}: {{ message.content }}
        \\{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}
    ;
    const messages_json =
        \\[
        \\  {"role": "user", "content": "Hello"}
        \\]
    ;

    const result = try render(allocator, template, messages_json, "", "", false);
    defer allocator.free(result);

    try std.testing.expect(std.mem.indexOf(u8, result, "user: Hello") != null);
    // Should NOT contain the assistant prompt
    try std.testing.expect(std.mem.lastIndexOf(u8, result, "assistant:") == null);
}

test "render with BOS/EOS tokens" {
    const allocator = std.testing.allocator;

    const template = "{{ bos_token }}{% for message in messages %}{{ message.content }}{{ eos_token }}{% endfor %}";
    const messages_json =
        \\[
        \\  {"role": "user", "content": "test"}
        \\]
    ;

    const result = try render(allocator, template, messages_json, "<s>", "</s>", false);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("<s>test</s>", result);
}

test "render empty messages array" {
    const allocator = std.testing.allocator;

    const template = "{% for message in messages %}{{ message.content }}{% endfor %}empty";
    const messages_json = "[]";

    const result = try render(allocator, template, messages_json, "", "", false);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("empty", result);
}

test "render invalid JSON returns error" {
    const allocator = std.testing.allocator;

    const template = "{% for message in messages %}{{ message.content }}{% endfor %}";
    const invalid_json = "[{invalid}]";

    const result = render(allocator, template, invalid_json, "", "", false);
    try std.testing.expectError(error.InvalidMessages, result);
}

test "render malformed JSON returns error" {
    const allocator = std.testing.allocator;

    const template = "test";
    const malformed_json = "{not an array}";

    const result = render(allocator, template, malformed_json, "", "", false);
    try std.testing.expectError(error.InvalidMessages, result);
}

test "render ChatML-style template" {
    const allocator = std.testing.allocator;

    const template =
        \\{%- for message in messages -%}
        \\<|im_start|>{{ message.role }}
        \\{{ message.content }}<|im_end|>
        \\{% endfor -%}
        \\{%- if add_generation_prompt -%}
        \\<|im_start|>assistant
        \\{% endif -%}
    ;

    const messages_json =
        \\[
        \\  {"role": "system", "content": "You are helpful."},
        \\  {"role": "user", "content": "Hi!"}
        \\]
    ;

    const result = try render(allocator, template, messages_json, "", "", true);
    defer allocator.free(result);

    try std.testing.expect(std.mem.indexOf(u8, result, "<|im_start|>system") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "You are helpful.") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "<|im_start|>user") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "Hi!") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "<|im_start|>assistant") != null);
}

test "render with tool_calls in message" {
    const allocator = std.testing.allocator;

    const template =
        \\{% for message in messages -%}
        \\{{ message.role }}:
        \\{% if message.tool_calls %}[TOOL_CALL]{% else %}{{ message.content }}{% endif %}
        \\{% endfor %}
    ;

    const messages_json =
        \\[
        \\  {"role": "user", "content": "What's the weather?"},
        \\  {"role": "assistant", "tool_calls": [{"name": "get_weather"}]}
        \\]
    ;

    const result = try render(allocator, template, messages_json, "", "", false);
    defer allocator.free(result);

    try std.testing.expect(std.mem.indexOf(u8, result, "user:") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "What's the weather?") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "assistant:") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "[TOOL_CALL]") != null);
}

test "render multi-turn conversation" {
    const allocator = std.testing.allocator;

    const template = "{% for message in messages %}{{ loop.index }}.{{ message.role }}: {{ message.content }}\n{% endfor %}";
    const messages_json =
        \\[
        \\  {"role": "user", "content": "Hi"},
        \\  {"role": "assistant", "content": "Hello"},
        \\  {"role": "user", "content": "How are you?"},
        \\  {"role": "assistant", "content": "I'm well!"}
        \\]
    ;

    const result = try render(allocator, template, messages_json, "", "", false);
    defer allocator.free(result);

    try std.testing.expect(std.mem.indexOf(u8, result, "1.user: Hi") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "2.assistant: Hello") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "3.user: How are you?") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "4.assistant: I'm well!") != null);
}

test "render with conditional system message" {
    const allocator = std.testing.allocator;

    const template =
        \\{% if messages[0].role == 'system' %}SYSTEM: {{ messages[0].content }}
        \\{% endif %}{% for message in messages %}{% if message.role != 'system' %}{{ message.role }}: {{ message.content }}
        \\{% endif %}{% endfor %}
    ;

    const messages_json =
        \\[
        \\  {"role": "system", "content": "Be helpful."},
        \\  {"role": "user", "content": "Hello"}
        \\]
    ;

    const result = try render(allocator, template, messages_json, "", "", false);
    defer allocator.free(result);

    try std.testing.expect(std.mem.indexOf(u8, result, "SYSTEM: Be helpful.") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "user: Hello") != null);
}

test "render with nested message properties" {
    const allocator = std.testing.allocator;

    const template =
        \\{% for message in messages %}{% if message.metadata %}[{{ message.metadata.source }}] {% endif %}{{ message.content }}
        \\{% endfor %}
    ;

    const messages_json =
        \\[
        \\  {"role": "user", "content": "test", "metadata": {"source": "web"}}
        \\]
    ;

    const result = try render(allocator, template, messages_json, "", "", false);
    defer allocator.free(result);

    try std.testing.expect(std.mem.indexOf(u8, result, "[web] test") != null);
}

test "render Llama-style template with BOS/EOS" {
    const allocator = std.testing.allocator;

    const template =
        \\{{ bos_token }}{% for message in messages %}{% if message.role == 'system' %}[INST] <<SYS>>
        \\{{ message.content }}
        \\<</SYS>>
        \\
        \\{% elif message.role == 'user' %}{{ message.content }} [/INST]{% elif message.role == 'assistant' %} {{ message.content }}{{ eos_token }}{% endif %}{% endfor %}
    ;

    const messages_json =
        \\[
        \\  {"role": "system", "content": "You are helpful."},
        \\  {"role": "user", "content": "Hi"},
        \\  {"role": "assistant", "content": "Hello!"}
        \\]
    ;

    const result = try render(allocator, template, messages_json, "<s>", "</s>", false);
    defer allocator.free(result);

    try std.testing.expect(std.mem.indexOf(u8, result, "<s>") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "<<SYS>>") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "You are helpful.") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "[/INST]") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "Hello!") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "</s>") != null);
}

// =============================================================================
// Tests for renderWithContext (extra context injection)
// =============================================================================

test "renderWithContext with extra string variable" {
    const allocator = std.testing.allocator;

    const template = "Date: {{ date }}\n{% for m in messages %}{{ m.content }}\n{% endfor %}";
    const messages_json =
        \\[{"role": "user", "content": "Hi"}]
    ;
    const extra_context = "{\"date\": \"2024-01-15\"}";

    const result = try renderWithContext(allocator, template, messages_json, "", "", false, extra_context);
    defer allocator.free(result);

    try std.testing.expect(std.mem.indexOf(u8, result, "Date: 2024-01-15") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "Hi") != null);
}

test "renderWithContext with extra array variable (tools)" {
    const allocator = std.testing.allocator;

    const template =
        \\{% if tools %}Tools: {% for tool in tools %}{{ tool.name }}{% if not loop.last %}, {% endif %}{% endfor %}
        \\{% endif %}{{ messages[0].content }}
    ;
    const messages_json =
        \\[{"role": "user", "content": "What can you do?"}]
    ;
    const extra_context =
        \\{"tools": [{"name": "search"}, {"name": "calculator"}, {"name": "weather"}]}
    ;

    const result = try renderWithContext(allocator, template, messages_json, "", "", false, extra_context);
    defer allocator.free(result);

    try std.testing.expect(std.mem.indexOf(u8, result, "Tools: search, calculator, weather") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "What can you do?") != null);
}

test "renderWithContext with boolean flag" {
    const allocator = std.testing.allocator;

    const template =
        \\{% if enable_thinking %}<think>
        \\{% endif %}{{ messages[0].content }}{% if enable_thinking %}
        \\</think>{% endif %}
    ;
    const messages_json =
        \\[{"role": "user", "content": "Solve 2+2"}]
    ;
    const extra_context = "{\"enable_thinking\": true}";

    const result = try renderWithContext(allocator, template, messages_json, "", "", false, extra_context);
    defer allocator.free(result);

    try std.testing.expect(std.mem.indexOf(u8, result, "<think>") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "</think>") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "Solve 2+2") != null);
}

test "renderWithContext with null extra context" {
    const allocator = std.testing.allocator;

    const template = "{{ messages[0].content }}";
    const messages_json =
        \\[{"role": "user", "content": "Hello"}]
    ;

    const result = try renderWithContext(allocator, template, messages_json, "", "", false, null);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Hello", result);
}

test "renderWithContext extra context can override standard variables" {
    const allocator = std.testing.allocator;

    // Extra context can override bos_token
    const template = "{{ bos_token }}{{ messages[0].content }}";
    const messages_json =
        \\[{"role": "user", "content": "test"}]
    ;
    const extra_context = "{\"bos_token\": \"<custom_bos>\"}";

    const result = try renderWithContext(allocator, template, messages_json, "<s>", "</s>", false, extra_context);
    defer allocator.free(result);

    // Extra context should override the bos_token parameter
    try std.testing.expectEqualStrings("<custom_bos>test", result);
}

test "renderWithContext with nested object in extra context" {
    const allocator = std.testing.allocator;

    const template = "User: {{ user.name }} ({{ user.role }})\n{{ messages[0].content }}";
    const messages_json =
        \\[{"role": "user", "content": "Hello"}]
    ;
    const extra_context =
        \\{"user": {"name": "Alice", "role": "admin"}}
    ;

    const result = try renderWithContext(allocator, template, messages_json, "", "", false, extra_context);
    defer allocator.free(result);

    try std.testing.expect(std.mem.indexOf(u8, result, "User: Alice (admin)") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "Hello") != null);
}

test "renderWithContext with invalid extra context JSON returns error" {
    const allocator = std.testing.allocator;

    const template = "{{ messages[0].content }}";
    const messages_json =
        \\[{"role": "user", "content": "test"}]
    ;
    const invalid_json = "{not valid json}";

    const result = renderWithContext(allocator, template, messages_json, "", "", false, invalid_json);
    try std.testing.expectError(error.InvalidMessages, result);
}

test "renderWithContext with array extra context returns error" {
    const allocator = std.testing.allocator;

    const template = "{{ messages[0].content }}";
    const messages_json =
        \\[{"role": "user", "content": "test"}]
    ;
    // Extra context must be an object, not an array
    const array_json = "[1, 2, 3]";

    const result = renderWithContext(allocator, template, messages_json, "", "", false, array_json);
    try std.testing.expectError(error.InvalidMessages, result);
}
