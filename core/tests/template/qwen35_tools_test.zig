//! Qwen3.5 Template + Tools Rendering Tests
//!
//! Verifies that the template engine produces valid UTF-8 output when rendering
//! the real Qwen3.5 chat template with tool definitions in the context.
//! Regression test for Utf8ProcFailed during tokenizer normalization.

const std = @import("std");
const main = @import("main");

const renderWithContext = main.template.chat_template.renderWithContext;

test "Qwen3.5 template with tools produces valid UTF-8" {
    const allocator = std.testing.allocator;

    const template_source = @embedFile("qwen35_chat_template.txt");

    const messages_json =
        \\[{"role": "user", "content": "What is the area of a triangle with base 10 and height 5?"}]
    ;

    const extra_context =
        \\{"enable_thinking": true, "add_vision_id": false, "tools": [{"type":"function","function":{"name":"calc_area","description":"Calculate the area of a shape","parameters":{"type":"object","properties":{"base":{"type":"number","description":"Base length"},"height":{"type":"number","description":"Height"}},"required":["base","height"]}}}]}
    ;

    const result = try renderWithContext(
        allocator,
        template_source,
        messages_json,
        "", // bos_token
        "<|endoftext|>", // eos_token
        true, // add_generation_prompt
        extra_context,
    );
    defer allocator.free(result);

    // Must be valid UTF-8.
    try std.testing.expect(std.unicode.utf8ValidateSlice(result));

    // Must not contain null bytes.
    try std.testing.expect(std.mem.indexOfScalar(u8, result, 0) == null);

    // Must contain the tool name.
    try std.testing.expect(std.mem.indexOf(u8, result, "calc_area") != null);

    // Must contain the user message.
    try std.testing.expect(std.mem.indexOf(u8, result, "What is the area") != null);

    // Must contain the generation prompt.
    try std.testing.expect(std.mem.indexOf(u8, result, "<|im_start|>assistant") != null);

    // Must contain the thinking tag (enable_thinking=true).
    try std.testing.expect(std.mem.indexOf(u8, result, "<think>") != null);

    // All bytes should be ASCII (template + tools are pure ASCII).
    for (result) |byte| {
        if (byte > 127) {
            std.debug.print("\nNon-ASCII byte 0x{x:0>2} found in output\n", .{byte});
            try std.testing.expect(false);
        }
    }
}

test "Qwen3.5 template with tools matches Jinja2 structure" {
    const allocator = std.testing.allocator;

    const template_source = @embedFile("qwen35_chat_template.txt");

    const messages_json =
        \\[{"role": "user", "content": "What is the area of a triangle with base 10 and height 5?"}]
    ;

    const extra_context =
        \\{"enable_thinking": true, "add_vision_id": false, "tools": [{"type":"function","function":{"name":"calc_area","description":"Calculate the area of a shape","parameters":{"type":"object","properties":{"base":{"type":"number","description":"Base length"},"height":{"type":"number","description":"Height"}},"required":["base","height"]}}}]}
    ;

    const result = try renderWithContext(
        allocator,
        template_source,
        messages_json,
        "",
        "<|endoftext|>",
        true,
        extra_context,
    );
    defer allocator.free(result);

    // The Qwen3.5 template with tools should produce output matching this structure:
    // <|im_start|>system\n# Tools\n\n...tool definitions...\n<|im_end|>\n
    // <|im_start|>user\n...\n<|im_end|>\n
    // <|im_start|>assistant\n<think>\n
    try std.testing.expect(std.mem.startsWith(u8, result, "<|im_start|>system\n# Tools"));

    // Tools section should contain <tools> tags
    try std.testing.expect(std.mem.indexOf(u8, result, "<tools>") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "</tools>") != null);

    // Should end with assistant generation prompt + think tag
    try std.testing.expect(std.mem.endsWith(u8, result, "<|im_start|>assistant\n<think>\n"));
}
