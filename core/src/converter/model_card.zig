//! Model Card Generation
//!
//! Generates README.md files for converted models with YAML frontmatter
//! for HuggingFace Hub compatibility.

const std = @import("std");
const scheme_mod = @import("scheme.zig");

/// Generate a Model Card (README.md) for a converted model.
///
/// The generated card includes:
/// - YAML frontmatter for HuggingFace Hub indexing
/// - Model description with quantization details
/// - Usage example with talu CLI
/// - Link to original model
pub fn generateModelCard(
    allocator: std.mem.Allocator,
    model_name: []const u8,
    base_model_id: []const u8,
    scheme: scheme_mod.Scheme,
) ![]u8 {
    var buf = std.ArrayListUnmanaged(u8){};
    errdefer buf.deinit(allocator);
    const writer = buf.writer(allocator);

    const scheme_name = scheme.toString();
    const suffix = scheme.toOutputSuffix();
    const bits = scheme.getBits();
    const method = scheme.getMethod();
    const method_name = switch (method) {
        .grouped_affine => "Grouped Affine (MLX)",
        .fp8 => "FP8",
        .mxfp4 => "MXFP4",
        .nvfp4 => "NVFP4",
    };

    // YAML frontmatter
    try writer.writeAll("---\n");
    try writer.print("base_model: {s}\n", .{base_model_id});
    try writer.writeAll("library_name: talu\n");
    try writer.writeAll("pipeline_tag: text-generation\n");
    try writer.writeAll("tags:\n");
    try writer.writeAll("  - talu\n");
    try writer.writeAll("  - quantized\n");

    // Add method-specific tags
    switch (method) {
        .grouped_affine => {
            try writer.writeAll("  - mlx\n");
            try writer.writeAll("  - grouped-affine\n");
        },
        .fp8 => try writer.writeAll("  - fp8\n"),
        .mxfp4 => try writer.writeAll("  - mxfp4\n"),
        .nvfp4 => try writer.writeAll("  - nvfp4\n"),
    }

    try writer.print("  - {d}-bit\n", .{bits});
    try writer.writeAll("---\n\n");

    // Title
    try writer.print("# {s}-{s} (Talu Quantized)\n\n", .{ model_name, suffix });

    // Description
    try writer.print(
        "This model was converted to **{s}** format ({s}, {d}-bit) using [Talu](https://github.com/aprxi/talu).\n\n",
        .{ scheme_name, method_name, bits },
    );

    // Quantization details
    try writer.writeAll("## Quantization Details\n\n");
    try writer.print("| Property | Value |\n", .{});
    try writer.print("|----------|-------|\n", .{});
    try writer.print("| Scheme | `{s}` |\n", .{scheme_name});
    try writer.print("| Method | {s} |\n", .{method_name});
    try writer.print("| Bits | {d} |\n", .{bits});

    const group_size = scheme.getGroupSize();
    if (group_size > 0) {
        try writer.print("| Group Size | {d} |\n", .{group_size});
    }
    try writer.writeAll("\n");

    // Usage
    try writer.writeAll("## Usage\n\n");
    try writer.writeAll("### Python\n\n");
    try writer.writeAll("```python\n");
    try writer.writeAll("import talu\n\n");
    try writer.writeAll("chat = talu.Chat(\"path/to/this/model\")\n");
    try writer.writeAll("response = chat(\"Hello, how are you?\")\n");
    try writer.writeAll("print(response)\n");
    try writer.writeAll("```\n\n");

    try writer.writeAll("### CLI\n\n");
    try writer.writeAll("```bash\n");
    try writer.writeAll("talu generate -m path/to/this/model \"Hello, how are you?\"\n");
    try writer.writeAll("```\n\n");

    // Original model
    try writer.writeAll("## Original Model\n\n");
    try writer.print(
        "This model is a quantized version of [{s}](https://huggingface.co/{s}).\n\n",
        .{ base_model_id, base_model_id },
    );

    // Footer
    try writer.writeAll("---\n\n");
    try writer.writeAll("*Generated with [Talu](https://github.com/aprxi/talu)*\n");

    return buf.toOwnedSlice(allocator);
}

/// Write a Model Card to a file in the output directory.
pub fn writeModelCard(
    allocator: std.mem.Allocator,
    output_dir: []const u8,
    model_name: []const u8,
    base_model_id: []const u8,
    scheme: scheme_mod.Scheme,
) !void {
    const content = try generateModelCard(allocator, model_name, base_model_id, scheme);
    defer allocator.free(content);

    const readme_path = try std.fs.path.join(allocator, &.{ output_dir, "README.md" });
    defer allocator.free(readme_path);

    var file = try std.fs.cwd().createFile(readme_path, .{});
    defer file.close();
    try file.writeAll(content);
}

/// Extract model name from a model path or HuggingFace ID.
/// "org/model-name" -> "model-name"
/// "models--org--model-name" -> "model-name"
/// "/path/to/models--org--model-name/snapshots/abc" -> "model-name"
pub fn extractModelName(path: []const u8) []const u8 {
    // Handle HuggingFace ID format: "org/model"
    if (std.mem.indexOf(u8, path, "/")) |slash_idx| {
        // Check if it's a simple "org/model" format (no more slashes after)
        const after_slash = path[slash_idx + 1 ..];
        if (std.mem.indexOf(u8, after_slash, "/") == null) {
            return after_slash;
        }
    }

    // Handle cache format: "models--org--model" or path containing it
    const basename = std.fs.path.basename(path);

    // Check for "models--" prefix in path components
    var iter = std.mem.splitSequence(u8, path, "/");
    var last_models_component: ?[]const u8 = null;
    while (iter.next()) |component| {
        if (std.mem.startsWith(u8, component, "models--")) {
            last_models_component = component;
        }
    }

    if (last_models_component) |component| {
        // Parse "models--org--model" -> "model"
        var parts = std.mem.splitSequence(u8, component, "--");
        var last_part: []const u8 = component;
        while (parts.next()) |part| {
            last_part = part;
        }
        return last_part;
    }

    return basename;
}

/// Extract base model ID from a model path.
/// "org/model-name" -> "org/model-name"
/// "models--org--model-name" -> "org/model-name"
/// "/path/to/models--org--model-name/snapshots/abc" -> "org/model-name"
pub fn extractBaseModelId(path: []const u8) []const u8 {
    // Handle HuggingFace ID format: "org/model"
    if (std.mem.indexOf(u8, path, "/")) |slash_idx| {
        const after_slash = path[slash_idx + 1 ..];
        if (std.mem.indexOf(u8, after_slash, "/") == null) {
            // Simple "org/model" format
            return path;
        }
    }

    // Handle cache format: find "models--org--model" component
    var iter = std.mem.splitSequence(u8, path, "/");
    while (iter.next()) |component| {
        if (std.mem.startsWith(u8, component, "models--")) {
            // Parse "models--org--model" -> return as-is for now
            // The caller should convert "--" to "/" if needed
            return component;
        }
    }

    return path;
}

// =============================================================================
// Tests
// =============================================================================

test "generateModelCard produces valid YAML frontmatter" {
    const allocator = std.testing.allocator;

    const card = try generateModelCard(
        allocator,
        "model-name",
        "org/model-name",
        .gaf4_64,
    );
    defer allocator.free(card);

    // Check YAML frontmatter
    try std.testing.expect(std.mem.startsWith(u8, card, "---\n"));
    try std.testing.expect(std.mem.indexOf(u8, card, "base_model: org/model-name") != null);
    try std.testing.expect(std.mem.indexOf(u8, card, "library_name: talu") != null);
    try std.testing.expect(std.mem.indexOf(u8, card, "pipeline_tag: text-generation") != null);
    try std.testing.expect(std.mem.indexOf(u8, card, "- mlx") != null);
    try std.testing.expect(std.mem.indexOf(u8, card, "- 4-bit") != null);
}

test "generateModelCard includes scheme details" {
    const allocator = std.testing.allocator;

    const card = try generateModelCard(
        allocator,
        "model-name",
        "org/model-name",
        .gaf4_64,
    );
    defer allocator.free(card);

    // Check MLX-specific tags
    try std.testing.expect(std.mem.indexOf(u8, card, "- mlx") != null);
    try std.testing.expect(std.mem.indexOf(u8, card, "- grouped-affine") != null);
    try std.testing.expect(std.mem.indexOf(u8, card, "| Group Size | 64 |") != null);
}

test "extractModelName handles HuggingFace ID" {
    try std.testing.expectEqualStrings("model-name", extractModelName("org/model-name"));
    try std.testing.expectEqualStrings("model-8b", extractModelName("some-org/model-8b"));
}

test "extractModelName handles cache format" {
    try std.testing.expectEqualStrings("model-name", extractModelName("models--org--model-name"));
    try std.testing.expectEqualStrings(
        "model-name",
        extractModelName("/home/user/.cache/huggingface/hub/models--org--model-name/snapshots/abc123"),
    );
}

test "extractBaseModelId handles HuggingFace ID" {
    try std.testing.expectEqualStrings("org/model-name", extractBaseModelId("org/model-name"));
}

test "extractBaseModelId handles cache format" {
    try std.testing.expectEqualStrings(
        "models--org--model-name",
        extractBaseModelId("/home/user/.cache/huggingface/hub/models--org--model-name/snapshots/abc"),
    );
}

test "writeModelCard: signature verification" {
    // writeModelCard writes to filesystem, so we verify the function signature
    // and test that generateModelCard (which it wraps) works correctly.
    const F = @TypeOf(writeModelCard);
    const info = @typeInfo(F).@"fn";

    // Should take (allocator, output_dir, model_name, base_model_id, scheme)
    try std.testing.expectEqual(@as(usize, 5), info.params.len);

    // Return type should be error union with void
    const return_info = @typeInfo(info.return_type.?);
    try std.testing.expect(return_info == .error_union);
}

test "writeModelCard writes README.md to temp directory" {
    const allocator = std.testing.allocator;

    // Create a temp directory for the test
    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const tmp_path = try tmp_dir.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    // Write the model card
    try writeModelCard(
        allocator,
        tmp_path,
        "TestModel",
        "test/TestModel",
        .gaf4_64,
    );

    // Verify the file was created and has content
    const readme_content = try tmp_dir.dir.readFileAlloc(allocator, "README.md", 1024 * 64);
    defer allocator.free(readme_content);

    // Check expected content
    try std.testing.expect(readme_content.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, readme_content, "---\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, readme_content, "base_model: test/TestModel") != null);
    try std.testing.expect(std.mem.indexOf(u8, readme_content, "TestModel-GAF4") != null);
}
