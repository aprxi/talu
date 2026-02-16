//! Generation Configuration
//!
//! Unified loading of generation config, chat templates, and special tokens.
//! This is the SINGLE source of truth - used by both CLI and C API.

const std = @import("std");
const json = @import("../json/root.zig");
const template = @import("../../template/root.zig");
const responses_mod = @import("../../responses/root.zig");
const log = @import("../../log.zig");

// =============================================================================
// Generation Config
// =============================================================================

/// Generation configuration loaded from model directory.
///
/// Token ID resolution priority (HuggingFace-compatible):
/// 1. tokenizer_config.json bos_token string (stored in bos_token_str, resolved later)
/// 2. generation_config.json / config.json bos_token_id (stored in bos_token_id)
/// 3. Tokenizer vocab heuristics (handled at lookup time)
pub const GenerationConfig = struct {
    temperature: f32 = 1.0,
    top_k: usize = 50,
    top_p: f32 = 1.0,
    // HuggingFace default when field is absent is greedy decoding.
    do_sample: bool = false,
    eos_token_ids: []const u32 = &.{},
    /// BOS token ID from generation_config.json or config.json (numeric).
    /// This is the fallback if bos_token_str is not set.
    bos_token_id: ?u32 = null,
    pad_token_id: ?u32 = null,
    add_bos_token: bool = false,
    /// BOS token string from tokenizer_config.json (e.g., "<|begin_of_text|>").
    /// Must be resolved to an ID via vocab lookup after tokenizer loads.
    /// Takes precedence over bos_token_id when set.
    bos_token_str: ?[]const u8 = null,
    /// True if tokenizer_config.json explicitly set bos_token to null.
    /// This means the model intentionally has no BOS token.
    bos_explicitly_disabled: bool = false,
    /// Maximum sequence length the model supports (from tokenizer_config.json).
    /// Used as default when truncation=True without explicit max_length.
    model_max_length: ?u64 = null,

    pub fn deinit(self: *GenerationConfig, allocator: std.mem.Allocator) void {
        if (self.eos_token_ids.len > 0) {
            allocator.free(self.eos_token_ids);
            self.eos_token_ids = &.{};
        }
        if (self.bos_token_str) |s| {
            allocator.free(s);
            self.bos_token_str = null;
        }
    }
};

/// Load generation config from a model path (directory).
pub fn loadGenerationConfig(allocator: std.mem.Allocator, model_path: []const u8) !GenerationConfig {
    return loadDirectoryGenerationConfig(allocator, model_path);
}

fn loadDirectoryGenerationConfig(allocator: std.mem.Allocator, model_dir: []const u8) !GenerationConfig {
    const config_path = try std.fs.path.join(allocator, &.{ model_dir, "generation_config.json" });
    defer allocator.free(config_path);

    const config_bytes = std.fs.cwd().readFileAlloc(allocator, config_path, 4 * 1024 * 1024) catch |err| {
        if (err == error.FileNotFound or err == error.NotDir) {
            log.warn("load", "No generation_config.json found, using neutral defaults", .{});
            var gen_config = GenerationConfig{};
            // Fall back to config.json for special token ids (some models omit from generation_config)
            try fillTokenIdsFromModelConfig(allocator, model_dir, &gen_config);
            // Load tokenizer_config.json for BOS token info
            loadTokenizerConfig(allocator, model_dir, &gen_config);
            return gen_config;
        }
        return err;
    };
    defer allocator.free(config_bytes);

    const parsed_config = json.parseValue(allocator, config_bytes, .{ .max_size_bytes = 4 * 1024 * 1024 }) catch |err| {
        log.warn("load", "Invalid JSON in generation_config.json", .{ .@"error" = @errorName(err) });
        return .{};
    };
    defer parsed_config.deinit();

    const config_fields = parsed_config.value.object;

    var gen_config = GenerationConfig{
        .temperature = getFloat(f32, config_fields, "temperature", 1.0),
        .top_k = getInt(usize, config_fields, "top_k", 50),
        .top_p = getFloat(f32, config_fields, "top_p", 1.0),
        .do_sample = getBool(config_fields, "do_sample", false),
        .bos_token_id = getOptionalInt(u32, config_fields, "bos_token_id"),
        .pad_token_id = getOptionalInt(u32, config_fields, "pad_token_id"),
        .eos_token_ids = try getIntArray(u32, allocator, config_fields, "eos_token_id"),
    };

    // Fill any missing token ids from config.json (do not override generation_config.json)
    try fillTokenIdsFromModelConfig(allocator, model_dir, &gen_config);
    // Load tokenizer_config.json for BOS token info (string, add_bos_token flag)
    loadTokenizerConfig(allocator, model_dir, &gen_config);

    return gen_config;
}

fn fillTokenIdsFromModelConfig(allocator: std.mem.Allocator, model_dir: []const u8, cfg: *GenerationConfig) !void {
    const config_path = try std.fs.path.join(allocator, &.{ model_dir, "config.json" });
    defer allocator.free(config_path);

    const config_bytes = std.fs.cwd().readFileAlloc(allocator, config_path, 4 * 1024 * 1024) catch return;
    defer allocator.free(config_bytes);

    const parsed_config = json.parseValue(allocator, config_bytes, .{ .max_size_bytes = 4 * 1024 * 1024 }) catch return;
    defer parsed_config.deinit();

    const config_fields = parsed_config.value.object;
    const text_config_fields: ?std.json.ObjectMap = if (config_fields.get("text_config")) |v|
        switch (v) {
            .object => |o| o,
            else => null,
        }
    else
        null;

    if (cfg.eos_token_ids.len == 0) {
        cfg.eos_token_ids = try getIntArray(u32, allocator, config_fields, "eos_token_id");
        if (cfg.eos_token_ids.len == 0) {
            if (text_config_fields) |text_fields| {
                cfg.eos_token_ids = try getIntArray(u32, allocator, text_fields, "eos_token_id");
            }
        }
    }
    if (cfg.bos_token_id == null) {
        cfg.bos_token_id = getOptionalInt(u32, config_fields, "bos_token_id");
        if (cfg.bos_token_id == null) {
            if (text_config_fields) |text_fields| {
                cfg.bos_token_id = getOptionalInt(u32, text_fields, "bos_token_id");
            }
        }
    }
    if (cfg.pad_token_id == null) {
        cfg.pad_token_id = getOptionalInt(u32, config_fields, "pad_token_id");
        if (cfg.pad_token_id == null) {
            if (text_config_fields) |text_fields| {
                cfg.pad_token_id = getOptionalInt(u32, text_fields, "pad_token_id");
            }
        }
    }
    // Fallback context length from config.json; loadTokenizerConfig may override
    // with model_max_length from tokenizer_config.json.
    cfg.model_max_length = getOptionalU64(config_fields, "max_position_embeddings");
    if (cfg.model_max_length == null) {
        if (text_config_fields) |text_fields| {
            cfg.model_max_length = getOptionalU64(text_fields, "max_position_embeddings");
        }
    }
}

/// Load BOS-related config from tokenizer_config.json into GenerationConfig.
/// Reads: add_bos_token, bos_token (string), model_max_length, and detects explicit null.
fn loadTokenizerConfig(allocator: std.mem.Allocator, model_dir: []const u8, cfg: *GenerationConfig) void {
    const config_path = std.fs.path.join(allocator, &.{ model_dir, "tokenizer_config.json" }) catch return;
    defer allocator.free(config_path);

    const config_bytes = std.fs.cwd().readFileAlloc(allocator, config_path, 4 * 1024 * 1024) catch return;
    defer allocator.free(config_bytes);

    const parsed_config = json.parseValue(allocator, config_bytes, .{ .max_size_bytes = 4 * 1024 * 1024 }) catch return;
    defer parsed_config.deinit();

    const obj = parsed_config.value.object;

    // Load add_bos_token flag.
    // HuggingFace behavior: only add BOS if explicitly set to true in tokenizer_config.json.
    // Missing key or null value → don't add BOS (matches HuggingFace default).
    if (obj.get("add_bos_token")) |v| {
        cfg.add_bos_token = switch (v) {
            .bool => |b| b,
            .null => false, // Explicitly null means don't add BOS
            else => false, // Non-boolean/non-null treated as false
        };
        log.debug("load", "Loaded add_bos_token from tokenizer_config.json", .{ .add_bos_token = cfg.add_bos_token, .value_type = @tagName(v) }, @src());
    } else {
        cfg.add_bos_token = false; // Key missing, HuggingFace default is false
        log.debug("load", "add_bos_token not in tokenizer_config.json, defaulting to false", .{}, @src());
    }

    // Load model_max_length (context length for truncation).
    // Only override if tokenizer_config.json has a valid value;
    // config.json's max_position_embeddings may already be set as fallback.
    if (getOptionalU64(obj, "model_max_length")) |max_len| {
        cfg.model_max_length = max_len;
    }

    // Load bos_token string (for vocab lookup after tokenizer loads)
    if (obj.get("bos_token")) |v| {
        switch (v) {
            .string => |s| {
                // Allocate a copy since JSON will be freed
                cfg.bos_token_str = allocator.dupe(u8, s) catch null;
            },
            .null => {
                // Explicitly set to null - model has no BOS token
                cfg.bos_explicitly_disabled = true;
            },
            else => {},
        }
    }
}

// =============================================================================
// Chat Template
// =============================================================================

/// Apply chat template with a JSON array of messages.
/// Supports multi-turn conversations, tool calls, and assistant prefill.
/// Returns allocated string that caller must free.
pub fn applyChatTemplate(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    messages_json: []const u8,
    add_generation_prompt: bool,
) ![]const u8 {
    return applyChatTemplateWithOverrides(allocator, model_path, messages_json, add_generation_prompt, null, null);
}

/// Apply a chat template string directly without loading from model directory.
///
/// This is useful for:
/// - Testing with custom templates
/// - Standalone tokenization without model files
/// - Custom template experimentation
///
/// Args:
///   allocator: Memory allocator
///   template_str: Jinja2-compatible chat template string
///   messages_json: JSON array of chat messages
///   add_generation_prompt: Whether to append assistant prompt
///   bos_token: Beginning-of-sequence token string (can be empty)
///   eos_token: End-of-sequence token string (can be empty)
///
/// Returns allocated string that caller must free.
pub fn applyChatTemplateFromString(
    allocator: std.mem.Allocator,
    template_str: []const u8,
    messages_json: []const u8,
    add_generation_prompt: bool,
    bos_token: []const u8,
    eos_token: []const u8,
) ![]const u8 {
    return template.chat_template.renderWithContext(
        allocator,
        template_str,
        messages_json,
        bos_token,
        eos_token,
        add_generation_prompt,
        null, // extra_context_json
    );
}

/// Apply chat template with optional overrides.
///
/// Like applyChatTemplate(), but allows:
/// - template_override: Use a custom template string instead of model's template
/// - extra_context_json: Inject additional variables into the template context
///
/// This enables:
/// - Using custom templates without modifying model files
/// - Passing tool definitions, system prompts, dates, etc. to templates
///
/// Args:
///   allocator: Memory allocator
///   model_path: Path to model directory (for loading tokenizer_config.json)
///   messages_json: JSON array of chat messages
///   add_generation_prompt: Whether to append assistant prompt
///   template_override: Optional custom template string. If null, uses model's template.
///   extra_context_json: Optional JSON object with additional template variables.
///       These are merged into context, e.g., {"tools": [...], "date": "2024-01-15"}
pub fn applyChatTemplateWithOverrides(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    messages_json: []const u8,
    add_generation_prompt: bool,
    template_override: ?[]const u8,
    extra_context_json: ?[]const u8,
) ![]const u8 {
    return applyDirectoryChatTemplate(allocator, model_path, messages_json, add_generation_prompt, template_override, extra_context_json);
}

fn applyDirectoryChatTemplate(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    messages_json: []const u8,
    add_generation_prompt: bool,
    template_override: ?[]const u8,
    extra_context_json: ?[]const u8,
) ![]const u8 {
    const config_path = try std.fs.path.join(allocator, &.{ model_dir, "tokenizer_config.json" });
    defer allocator.free(config_path);

    const config_bytes = std.fs.cwd().readFileAlloc(allocator, config_path, 4 * 1024 * 1024) catch |err| {
        return err;
    };
    defer allocator.free(config_bytes);

    const parsed_config = json.parseValue(allocator, config_bytes, .{ .max_size_bytes = 4 * 1024 * 1024 }) catch {
        return error.InvalidJson;
    };
    defer parsed_config.deinit();

    // Use template_override if provided, otherwise load from model
    var template_from_file: ?[]const u8 = null;
    const chat_template: []const u8 = if (template_override) |override|
        override
    else blk: {
        if (parsed_config.value.object.get("chat_template")) |template_value| {
            switch (template_value) {
                .string => |s| break :blk s,
                else => {},
            }
        }
        const jinja_path = std.fs.path.join(allocator, &.{ model_dir, "chat_template.jinja" }) catch {
            return error.MissingChatTemplate;
        };
        defer allocator.free(jinja_path);

        template_from_file = std.fs.cwd().readFileAlloc(allocator, jinja_path, 64 * 1024) catch {
            return error.MissingChatTemplate;
        };
        break :blk template_from_file.?;
    };
    defer if (template_from_file) |t| allocator.free(t);

    const bos_token_str = if (parsed_config.value.object.get("bos_token")) |v| switch (v) {
        .string => |s| s,
        else => "",
    } else "";
    const eos_token_str = if (parsed_config.value.object.get("eos_token")) |v| switch (v) {
        .string => |s| s,
        else => "",
    } else "";

    return template.chat_template.renderWithContext(
        allocator,
        chat_template,
        messages_json,
        bos_token_str,
        eos_token_str,
        add_generation_prompt,
        extra_context_json,
    );
}

/// Get the raw chat template source string from a model directory.
/// Returns the template from tokenizer_config.json or chat_template.jinja.
/// Caller must free the returned string.
pub fn getChatTemplateSource(allocator: std.mem.Allocator, model_dir: []const u8) ![]const u8 {
    const config_path = try std.fs.path.join(allocator, &.{ model_dir, "tokenizer_config.json" });
    defer allocator.free(config_path);

    const config_bytes = std.fs.cwd().readFileAlloc(allocator, config_path, 4 * 1024 * 1024) catch |err| {
        return err;
    };
    defer allocator.free(config_bytes);

    const parsed_config = json.parseValue(allocator, config_bytes, .{ .max_size_bytes = 4 * 1024 * 1024 }) catch {
        return error.InvalidJson;
    };
    defer parsed_config.deinit();

    // Try inline chat_template first
    if (parsed_config.value.object.get("chat_template")) |template_value| {
        switch (template_value) {
            .string => |s| {
                // Return a copy that caller can free
                return allocator.dupe(u8, s);
            },
            else => {},
        }
    }

    // Fall back to chat_template.jinja file
    const jinja_path = try std.fs.path.join(allocator, &.{ model_dir, "chat_template.jinja" });
    defer allocator.free(jinja_path);

    return std.fs.cwd().readFileAlloc(allocator, jinja_path, 64 * 1024) catch {
        return error.MissingChatTemplate;
    };
}

// =============================================================================
// JSON Helpers
// =============================================================================

fn getFloat(comptime T: type, obj: std.json.ObjectMap, key: []const u8, default: T) T {
    if (obj.get(key)) |value| {
        return switch (value) {
            .float => |f| @floatCast(f),
            .integer => |int_val| @floatFromInt(int_val),
            else => default,
        };
    }
    return default;
}

fn getInt(comptime T: type, obj: std.json.ObjectMap, key: []const u8, default: T) T {
    if (obj.get(key)) |value| {
        return switch (value) {
            .integer => |int_val| @intCast(int_val),
            else => default,
        };
    }
    return default;
}

fn getOptionalInt(comptime T: type, obj: std.json.ObjectMap, key: []const u8) ?T {
    if (obj.get(key)) |value| {
        return switch (value) {
            .integer => |int_val| if (int_val >= 0) @intCast(int_val) else null,
            else => null,
        };
    }
    return null;
}

/// Get optional u64 from JSON, handling both integer and float (for large values).
/// Some models have model_max_length as a very large float (e.g., 1e30).
fn getOptionalU64(obj: std.json.ObjectMap, key: []const u8) ?u64 {
    if (obj.get(key)) |value| {
        return switch (value) {
            .integer => |int_val| if (int_val >= 0) @intCast(int_val) else null,
            .float => |f| if (f >= 0 and f <= @as(f64, @floatFromInt(std.math.maxInt(u64)))) @intFromFloat(f) else null,
            else => null,
        };
    }
    return null;
}

fn getBool(obj: std.json.ObjectMap, key: []const u8, default: bool) bool {
    if (obj.get(key)) |value| {
        return switch (value) {
            .bool => |b| b,
            else => default,
        };
    }
    return default;
}

fn getIntArray(comptime T: type, allocator: std.mem.Allocator, obj: std.json.ObjectMap, key: []const u8) ![]const T {
    const value = obj.get(key) orelse return &.{};

    switch (value) {
        .integer => |int_val| {
            if (int_val < 0) return &.{};
            const ids = try allocator.alloc(T, 1);
            ids[0] = @intCast(int_val);
            return ids;
        },
        .array => |arr| {
            if (arr.items.len == 0) return &.{};
            var ids = try allocator.alloc(T, arr.items.len);
            var count: usize = 0;
            for (arr.items) |item| {
                if (item == .integer and item.integer >= 0) {
                    ids[count] = @intCast(item.integer);
                    count += 1;
                }
            }
            if (count == 0) {
                allocator.free(ids);
                return &.{};
            }
            return try allocator.realloc(ids, count);
        },
        else => return &.{},
    }
}

// =============================================================================
// EOS Token Helpers
// =============================================================================

/// Check if a token ID is in the EOS list.
pub fn isEosToken(eos_token_ids: []const u32, token: u32) bool {
    for (eos_token_ids) |eos| {
        if (eos == token) return true;
    }
    return false;
}

/// Add an EOS token ID if not already present.
pub fn addEosTokenId(allocator: std.mem.Allocator, cfg: *GenerationConfig, id: u32) !void {
    if (isEosToken(cfg.eos_token_ids, id)) return;
    if (cfg.eos_token_ids.len == 0) {
        const ids = try allocator.alloc(u32, 1);
        ids[0] = id;
        cfg.eos_token_ids = ids;
        return;
    }
    const old = cfg.eos_token_ids;
    const ids = try allocator.alloc(u32, old.len + 1);
    @memcpy(ids[0..old.len], old);
    ids[old.len] = id;
    allocator.free(old);
    cfg.eos_token_ids = ids;
}

/// Tokenizer handle type for token lookups.
pub const TokenizerHandle = @import("../../tokenizer/root.zig").CTokenizer;

/// Add an EOS token from tokenizer vocabulary if not already in the list.
/// Looks up the token text in the tokenizer's vocabulary and adds its ID.
pub fn addEosFromTokenizer(allocator: std.mem.Allocator, tokenizer_handle: *TokenizerHandle, cfg: *GenerationConfig, token_text: []const u8) void {
    if (tokenizer_handle.tokenToId(token_text)) |token_id| {
        const eos_id: u32 = @intCast(token_id);
        // Check if already in list
        if (isEosToken(cfg.eos_token_ids, eos_id)) return;
        // Add to list
        addEosTokenId(allocator, cfg, eos_id) catch return;
    }
}

// =============================================================================
// Tests
// =============================================================================

test "GenerationConfig.deinit frees allocated memory" {
    var config = GenerationConfig{};

    // Allocate EOS tokens
    const eos_ids = try std.testing.allocator.alloc(u32, 3);
    errdefer std.testing.allocator.free(eos_ids);
    eos_ids[0] = 1;
    eos_ids[1] = 2;
    eos_ids[2] = 3;
    config.eos_token_ids = eos_ids;

    // Allocate BOS token string
    config.bos_token_str = try std.testing.allocator.dupe(u8, "<|begin|>");

    // Should not leak
    config.deinit(std.testing.allocator);
}

test "GenerationConfig.deinit handles empty arrays" {
    var config = GenerationConfig{};
    // Should not crash with empty arrays
    config.deinit(std.testing.allocator);
}

test "isEosToken returns true for matching token" {
    const eos_ids = [_]u32{ 1, 2, 3, 128009 };
    try std.testing.expect(isEosToken(&eos_ids, 1));
    try std.testing.expect(isEosToken(&eos_ids, 2));
    try std.testing.expect(isEosToken(&eos_ids, 3));
    try std.testing.expect(isEosToken(&eos_ids, 128009));
}

test "isEosToken returns false for non-matching token" {
    const eos_ids = [_]u32{ 1, 2, 3 };
    try std.testing.expect(!isEosToken(&eos_ids, 0));
    try std.testing.expect(!isEosToken(&eos_ids, 4));
    try std.testing.expect(!isEosToken(&eos_ids, 128009));
}

test "isEosToken handles empty array" {
    const eos_ids = [_]u32{};
    try std.testing.expect(!isEosToken(&eos_ids, 1));
    try std.testing.expect(!isEosToken(&eos_ids, 0));
}

test "addEosTokenId adds token to empty config" {
    var config = GenerationConfig{};
    defer config.deinit(std.testing.allocator);

    try addEosTokenId(std.testing.allocator, &config, 42);

    try std.testing.expectEqual(@as(usize, 1), config.eos_token_ids.len);
    try std.testing.expectEqual(@as(u32, 42), config.eos_token_ids[0]);
}

test "addEosTokenId appends token to existing list" {
    var config = GenerationConfig{};
    defer config.deinit(std.testing.allocator);

    // Start with two tokens
    const initial_ids = try std.testing.allocator.alloc(u32, 2);
    errdefer std.testing.allocator.free(initial_ids);
    initial_ids[0] = 1;
    initial_ids[1] = 2;
    config.eos_token_ids = initial_ids;

    // Add a third
    try addEosTokenId(std.testing.allocator, &config, 3);

    try std.testing.expectEqual(@as(usize, 3), config.eos_token_ids.len);
    try std.testing.expectEqual(@as(u32, 1), config.eos_token_ids[0]);
    try std.testing.expectEqual(@as(u32, 2), config.eos_token_ids[1]);
    try std.testing.expectEqual(@as(u32, 3), config.eos_token_ids[2]);
}

test "addEosTokenId does not add duplicate token" {
    var config = GenerationConfig{};
    defer config.deinit(std.testing.allocator);

    const initial_ids = try std.testing.allocator.alloc(u32, 2);
    errdefer std.testing.allocator.free(initial_ids);
    initial_ids[0] = 1;
    initial_ids[1] = 2;
    config.eos_token_ids = initial_ids;

    // Try to add existing token
    try addEosTokenId(std.testing.allocator, &config, 1);

    // Should still be 2
    try std.testing.expectEqual(@as(usize, 2), config.eos_token_ids.len);
    try std.testing.expectEqual(@as(u32, 1), config.eos_token_ids[0]);
    try std.testing.expectEqual(@as(u32, 2), config.eos_token_ids[1]);
}

test "getFloat extracts float values from JSON" {
    var obj = std.json.ObjectMap.init(std.testing.allocator);
    defer obj.deinit();

    try obj.put("temperature", .{ .float = 0.8 });

    const result = getFloat(f32, obj, "temperature", 1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), result, 0.001);
}

test "getFloat handles integer as float" {
    var obj = std.json.ObjectMap.init(std.testing.allocator);
    defer obj.deinit();

    try obj.put("temperature", .{ .integer = 2 });

    const result = getFloat(f32, obj, "temperature", 1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), result, 0.001);
}

test "getFloat returns default for missing key" {
    var obj = std.json.ObjectMap.init(std.testing.allocator);
    defer obj.deinit();

    const result = getFloat(f32, obj, "missing", 1.5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), result, 0.001);
}

test "getFloat returns default for wrong type" {
    var obj = std.json.ObjectMap.init(std.testing.allocator);
    defer obj.deinit();

    try obj.put("temperature", .{ .string = "not a number" });

    const result = getFloat(f32, obj, "temperature", 1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result, 0.001);
}

test "getInt extracts integer values from JSON" {
    var obj = std.json.ObjectMap.init(std.testing.allocator);
    defer obj.deinit();

    try obj.put("top_k", .{ .integer = 50 });

    const result = getInt(usize, obj, "top_k", 100);
    try std.testing.expectEqual(@as(usize, 50), result);
}

test "getInt returns default for missing key" {
    var obj = std.json.ObjectMap.init(std.testing.allocator);
    defer obj.deinit();

    const result = getInt(usize, obj, "missing", 100);
    try std.testing.expectEqual(@as(usize, 100), result);
}

test "getInt returns default for wrong type" {
    var obj = std.json.ObjectMap.init(std.testing.allocator);
    defer obj.deinit();

    try obj.put("top_k", .{ .string = "fifty" });

    const result = getInt(usize, obj, "top_k", 100);
    try std.testing.expectEqual(@as(usize, 100), result);
}

test "getOptionalInt extracts positive integers" {
    var obj = std.json.ObjectMap.init(std.testing.allocator);
    defer obj.deinit();

    try obj.put("bos_token_id", .{ .integer = 1 });

    const result = getOptionalInt(u32, obj, "bos_token_id");
    try std.testing.expect(result != null);
    try std.testing.expectEqual(@as(u32, 1), result.?);
}

test "getOptionalInt returns null for negative integers" {
    var obj = std.json.ObjectMap.init(std.testing.allocator);
    defer obj.deinit();

    try obj.put("bos_token_id", .{ .integer = -1 });

    const result = getOptionalInt(u32, obj, "bos_token_id");
    try std.testing.expect(result == null);
}

test "getOptionalInt returns null for missing key" {
    var obj = std.json.ObjectMap.init(std.testing.allocator);
    defer obj.deinit();

    const result = getOptionalInt(u32, obj, "missing");
    try std.testing.expect(result == null);
}

test "getBool extracts boolean values from JSON" {
    var obj = std.json.ObjectMap.init(std.testing.allocator);
    defer obj.deinit();

    try obj.put("do_sample", .{ .bool = false });

    const result = getBool(obj, "do_sample", true);
    try std.testing.expectEqual(false, result);
}

test "getBool returns default for missing key" {
    var obj = std.json.ObjectMap.init(std.testing.allocator);
    defer obj.deinit();

    const result = getBool(obj, "missing", true);
    try std.testing.expectEqual(true, result);
}

test "getBool returns default for wrong type" {
    var obj = std.json.ObjectMap.init(std.testing.allocator);
    defer obj.deinit();

    try obj.put("do_sample", .{ .string = "yes" });

    const result = getBool(obj, "do_sample", true);
    try std.testing.expectEqual(true, result);
}

test "getIntArray handles single integer" {
    var obj = std.json.ObjectMap.init(std.testing.allocator);
    defer obj.deinit();

    try obj.put("eos_token_id", .{ .integer = 128009 });

    const result = try getIntArray(u32, std.testing.allocator, obj, "eos_token_id");
    defer if (result.len > 0) std.testing.allocator.free(result);

    try std.testing.expectEqual(@as(usize, 1), result.len);
    try std.testing.expectEqual(@as(u32, 128009), result[0]);
}

test "getIntArray handles array of integers" {
    var obj = std.json.ObjectMap.init(std.testing.allocator);
    defer obj.deinit();

    var arr = std.json.Array.init(std.testing.allocator);
    defer arr.deinit();
    try arr.append(.{ .integer = 1 });
    try arr.append(.{ .integer = 2 });
    try arr.append(.{ .integer = 3 });

    try obj.put("eos_token_id", .{ .array = arr });

    const result = try getIntArray(u32, std.testing.allocator, obj, "eos_token_id");
    defer if (result.len > 0) std.testing.allocator.free(result);

    try std.testing.expectEqual(@as(usize, 3), result.len);
    try std.testing.expectEqual(@as(u32, 1), result[0]);
    try std.testing.expectEqual(@as(u32, 2), result[1]);
    try std.testing.expectEqual(@as(u32, 3), result[2]);
}

test "getIntArray filters out negative integers" {
    var obj = std.json.ObjectMap.init(std.testing.allocator);
    defer obj.deinit();

    var arr = std.json.Array.init(std.testing.allocator);
    defer arr.deinit();
    try arr.append(.{ .integer = 1 });
    try arr.append(.{ .integer = -1 });
    try arr.append(.{ .integer = 3 });

    try obj.put("eos_token_id", .{ .array = arr });

    const result = try getIntArray(u32, std.testing.allocator, obj, "eos_token_id");
    defer if (result.len > 0) std.testing.allocator.free(result);

    try std.testing.expectEqual(@as(usize, 2), result.len);
    try std.testing.expectEqual(@as(u32, 1), result[0]);
    try std.testing.expectEqual(@as(u32, 3), result[1]);
}

test "getIntArray returns empty for missing key" {
    var obj = std.json.ObjectMap.init(std.testing.allocator);
    defer obj.deinit();

    const result = try getIntArray(u32, std.testing.allocator, obj, "missing");
    defer if (result.len > 0) std.testing.allocator.free(result);

    try std.testing.expectEqual(@as(usize, 0), result.len);
}

test "getIntArray returns empty for negative single integer" {
    var obj = std.json.ObjectMap.init(std.testing.allocator);
    defer obj.deinit();

    try obj.put("eos_token_id", .{ .integer = -1 });

    const result = try getIntArray(u32, std.testing.allocator, obj, "eos_token_id");
    defer if (result.len > 0) std.testing.allocator.free(result);

    try std.testing.expectEqual(@as(usize, 0), result.len);
}

test "getIntArray returns empty for empty array" {
    var obj = std.json.ObjectMap.init(std.testing.allocator);
    defer obj.deinit();

    var arr = std.json.Array.init(std.testing.allocator);
    defer arr.deinit();

    try obj.put("eos_token_id", .{ .array = arr });

    const result = try getIntArray(u32, std.testing.allocator, obj, "eos_token_id");
    defer if (result.len > 0) std.testing.allocator.free(result);

    try std.testing.expectEqual(@as(usize, 0), result.len);
}

test "getIntArray returns empty for array with no valid integers" {
    var obj = std.json.ObjectMap.init(std.testing.allocator);
    defer obj.deinit();

    var arr = std.json.Array.init(std.testing.allocator);
    defer arr.deinit();
    try arr.append(.{ .string = "not a number" });
    try arr.append(.{ .bool = true });

    try obj.put("eos_token_id", .{ .array = arr });

    const result = try getIntArray(u32, std.testing.allocator, obj, "eos_token_id");
    defer if (result.len > 0) std.testing.allocator.free(result);

    try std.testing.expectEqual(@as(usize, 0), result.len);
}

test "GenerationConfig default values are sensible" {
    const config = GenerationConfig{};

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), config.temperature, 0.001);
    try std.testing.expectEqual(@as(usize, 50), config.top_k);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), config.top_p, 0.001);
    try std.testing.expectEqual(false, config.do_sample);
    try std.testing.expectEqual(@as(usize, 0), config.eos_token_ids.len);
    try std.testing.expect(config.bos_token_id == null);
    try std.testing.expect(config.pad_token_id == null);
    try std.testing.expectEqual(false, config.add_bos_token);
    try std.testing.expect(config.bos_token_str == null);
    try std.testing.expectEqual(false, config.bos_explicitly_disabled);
}

test "loadGenerationConfig loads from valid generation_config.json" {
    const allocator = std.testing.allocator;

    // Create a temporary directory with generation_config.json
    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const config_json =
        \\{
        \\  "temperature": 0.7,
        \\  "top_k": 40,
        \\  "top_p": 0.9,
        \\  "do_sample": false,
        \\  "bos_token_id": 1,
        \\  "eos_token_id": [2, 3],
        \\  "pad_token_id": 0
        \\}
    ;

    try tmp_dir.dir.writeFile(.{ .sub_path = "generation_config.json", .data = config_json });

    const tmp_path = try tmp_dir.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    var config = try loadGenerationConfig(allocator, tmp_path);
    defer config.deinit(allocator);

    try std.testing.expectApproxEqAbs(@as(f32, 0.7), config.temperature, 0.001);
    try std.testing.expectEqual(@as(usize, 40), config.top_k);
    try std.testing.expectApproxEqAbs(@as(f32, 0.9), config.top_p, 0.001);
    try std.testing.expectEqual(false, config.do_sample);
    try std.testing.expectEqual(@as(u32, 1), config.bos_token_id.?);
    try std.testing.expectEqual(@as(u32, 0), config.pad_token_id.?);
    try std.testing.expectEqual(@as(usize, 2), config.eos_token_ids.len);
    try std.testing.expectEqual(@as(u32, 2), config.eos_token_ids[0]);
    try std.testing.expectEqual(@as(u32, 3), config.eos_token_ids[1]);
}

test "loadGenerationConfig falls back to defaults when file missing" {
    const allocator = std.testing.allocator;

    // Create a temporary directory without generation_config.json
    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const tmp_path = try tmp_dir.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    var config = try loadGenerationConfig(allocator, tmp_path);
    defer config.deinit(allocator);

    // Should have default values
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), config.temperature, 0.001);
    try std.testing.expectEqual(@as(usize, 50), config.top_k);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), config.top_p, 0.001);
    try std.testing.expectEqual(false, config.do_sample);
}

test "loadGenerationConfig merges config.json when generation_config.json missing" {
    const allocator = std.testing.allocator;

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const config_json =
        \\{
        \\  "bos_token_id": 5,
        \\  "eos_token_id": 6,
        \\  "pad_token_id": 7
        \\}
    ;

    try tmp_dir.dir.writeFile(.{ .sub_path = "config.json", .data = config_json });

    const tmp_path = try tmp_dir.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    var config = try loadGenerationConfig(allocator, tmp_path);
    defer config.deinit(allocator);

    // Should get token IDs from config.json
    try std.testing.expectEqual(@as(u32, 5), config.bos_token_id.?);
    try std.testing.expectEqual(@as(u32, 7), config.pad_token_id.?);
    try std.testing.expectEqual(@as(usize, 1), config.eos_token_ids.len);
    try std.testing.expectEqual(@as(u32, 6), config.eos_token_ids[0]);
}

test "loadGenerationConfig reads token IDs from config.json text_config fallback" {
    const allocator = std.testing.allocator;

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const config_json =
        \\{
        \\  "model_type": "multimodal_text_bridge",
        \\  "text_config": {
        \\    "bos_token_id": 151643,
        \\    "eos_token_id": 151645,
        \\    "pad_token_id": 151643,
        \\    "max_position_embeddings": 262144
        \\  }
        \\}
    ;

    try tmp_dir.dir.writeFile(.{ .sub_path = "config.json", .data = config_json });

    const tmp_path = try tmp_dir.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    var config = try loadGenerationConfig(allocator, tmp_path);
    defer config.deinit(allocator);

    try std.testing.expectEqual(@as(u32, 151643), config.bos_token_id.?);
    try std.testing.expectEqual(@as(u32, 151643), config.pad_token_id.?);
    try std.testing.expectEqual(@as(usize, 1), config.eos_token_ids.len);
    try std.testing.expectEqual(@as(u32, 151645), config.eos_token_ids[0]);
    try std.testing.expectEqual(@as(?u64, 262144), config.model_max_length);
}

test "loadGenerationConfig loads bos_token_str from tokenizer_config.json" {
    const allocator = std.testing.allocator;

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const tokenizer_config_json =
        \\{
        \\  "bos_token": "<|begin_of_text|>",
        \\  "add_bos_token": false
        \\}
    ;

    try tmp_dir.dir.writeFile(.{ .sub_path = "tokenizer_config.json", .data = tokenizer_config_json });

    const tmp_path = try tmp_dir.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    var config = try loadGenerationConfig(allocator, tmp_path);
    defer config.deinit(allocator);

    try std.testing.expect(config.bos_token_str != null);
    try std.testing.expectEqualStrings("<|begin_of_text|>", config.bos_token_str.?);
    try std.testing.expectEqual(false, config.add_bos_token);
}

test "loadGenerationConfig detects explicit null bos_token" {
    const allocator = std.testing.allocator;

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const tokenizer_config_json =
        \\{
        \\  "bos_token": null
        \\}
    ;

    try tmp_dir.dir.writeFile(.{ .sub_path = "tokenizer_config.json", .data = tokenizer_config_json });

    const tmp_path = try tmp_dir.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    var config = try loadGenerationConfig(allocator, tmp_path);
    defer config.deinit(allocator);

    try std.testing.expectEqual(true, config.bos_explicitly_disabled);
    try std.testing.expect(config.bos_token_str == null);
}

test "loadGenerationConfig treats null add_bos_token as false" {
    const allocator = std.testing.allocator;

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    // GPT-OSS style: add_bos_token is explicitly null (not missing)
    const tokenizer_config_json =
        \\{
        \\  "add_bos_token": null,
        \\  "bos_token": "<|startoftext|>"
        \\}
    ;

    try tmp_dir.dir.writeFile(.{ .sub_path = "tokenizer_config.json", .data = tokenizer_config_json });

    const tmp_path = try tmp_dir.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    var config = try loadGenerationConfig(allocator, tmp_path);
    defer config.deinit(allocator);

    // Null add_bos_token should be treated as false (HuggingFace behavior)
    try std.testing.expectEqual(false, config.add_bos_token);
    try std.testing.expect(config.bos_token_str != null);
    try std.testing.expectEqualStrings("<|startoftext|>", config.bos_token_str.?);
}

test "loadGenerationConfig defaults missing add_bos_token to false" {
    // GPT-OSS style: add_bos_token key is missing entirely from tokenizer_config.json
    const allocator = std.testing.allocator;

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    // No add_bos_token in tokenizer_config.json at all
    const tokenizer_config_json =
        \\{
        \\  "bos_token": "<|startoftext|>"
        \\}
    ;

    try tmp_dir.dir.writeFile(.{ .sub_path = "tokenizer_config.json", .data = tokenizer_config_json });

    const tmp_path = try tmp_dir.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    var config = try loadGenerationConfig(allocator, tmp_path);
    defer config.deinit(allocator);

    // Missing add_bos_token should default to false (HuggingFace behavior)
    try std.testing.expectEqual(false, config.add_bos_token);
    // bos_token string should still be loaded
    try std.testing.expect(config.bos_token_str != null);
    try std.testing.expectEqualStrings("<|startoftext|>", config.bos_token_str.?);
}

test "loadGenerationConfig respects explicit true add_bos_token" {
    // Some models explicitly set add_bos_token: true
    const allocator = std.testing.allocator;

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const tokenizer_config_json =
        \\{
        \\  "add_bos_token": true,
        \\  "bos_token": "<bos>"
        \\}
    ;

    try tmp_dir.dir.writeFile(.{ .sub_path = "tokenizer_config.json", .data = tokenizer_config_json });

    const tmp_path = try tmp_dir.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    var config = try loadGenerationConfig(allocator, tmp_path);
    defer config.deinit(allocator);

    // Explicit true should be respected
    try std.testing.expectEqual(true, config.add_bos_token);
}

test "loadGenerationConfig loads model_max_length from tokenizer_config.json" {
    const allocator = std.testing.allocator;

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const tokenizer_config_json =
        \\{
        \\  "model_max_length": 32768
        \\}
    ;

    try tmp_dir.dir.writeFile(.{ .sub_path = "tokenizer_config.json", .data = tokenizer_config_json });

    const tmp_path = try tmp_dir.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    var config = try loadGenerationConfig(allocator, tmp_path);
    defer config.deinit(allocator);

    try std.testing.expect(config.model_max_length != null);
    try std.testing.expectEqual(@as(u64, 32768), config.model_max_length.?);
}

test "loadGenerationConfig handles missing model_max_length" {
    const allocator = std.testing.allocator;

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const tokenizer_config_json =
        \\{
        \\  "bos_token": "<s>"
        \\}
    ;

    try tmp_dir.dir.writeFile(.{ .sub_path = "tokenizer_config.json", .data = tokenizer_config_json });

    const tmp_path = try tmp_dir.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    var config = try loadGenerationConfig(allocator, tmp_path);
    defer config.deinit(allocator);

    try std.testing.expect(config.model_max_length == null);
}

test "loadGenerationConfig falls back to max_position_embeddings from config.json" {
    const allocator = std.testing.allocator;

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    // tokenizer_config.json without model_max_length
    const tokenizer_config_json =
        \\{
        \\  "bos_token": "<s>"
        \\}
    ;

    // config.json with max_position_embeddings
    const config_json =
        \\{
        \\  "max_position_embeddings": 128000,
        \\  "bos_token_id": 1
        \\}
    ;

    try tmp_dir.dir.writeFile(.{ .sub_path = "tokenizer_config.json", .data = tokenizer_config_json });
    try tmp_dir.dir.writeFile(.{ .sub_path = "config.json", .data = config_json });

    const tmp_path = try tmp_dir.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    var config = try loadGenerationConfig(allocator, tmp_path);
    defer config.deinit(allocator);

    // Should fall back to max_position_embeddings from config.json
    try std.testing.expect(config.model_max_length != null);
    try std.testing.expectEqual(@as(u64, 128000), config.model_max_length.?);
}

test "loadGenerationConfig model_max_length overrides max_position_embeddings" {
    const allocator = std.testing.allocator;

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    // tokenizer_config.json WITH model_max_length
    const tokenizer_config_json =
        \\{
        \\  "model_max_length": 4096
        \\}
    ;

    // config.json with different max_position_embeddings
    const config_json =
        \\{
        \\  "max_position_embeddings": 128000
        \\}
    ;

    try tmp_dir.dir.writeFile(.{ .sub_path = "tokenizer_config.json", .data = tokenizer_config_json });
    try tmp_dir.dir.writeFile(.{ .sub_path = "config.json", .data = config_json });

    const tmp_path = try tmp_dir.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    var config = try loadGenerationConfig(allocator, tmp_path);
    defer config.deinit(allocator);

    // model_max_length from tokenizer_config.json takes precedence
    try std.testing.expect(config.model_max_length != null);
    try std.testing.expectEqual(@as(u64, 4096), config.model_max_length.?);
}

test "applyChatTemplate renders simple template from tokenizer_config.json" {
    const allocator = std.testing.allocator;

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const tokenizer_config_json =
        \\{
        \\  "chat_template": "{{ messages[0].content }}",
        \\  "bos_token": "<s>",
        \\  "eos_token": "</s>"
        \\}
    ;

    const messages_json =
        \\[{"role": "user", "content": "Hello world"}]
    ;

    try tmp_dir.dir.writeFile(.{ .sub_path = "tokenizer_config.json", .data = tokenizer_config_json });

    const tmp_path = try tmp_dir.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    const result = try applyChatTemplate(allocator, tmp_path, messages_json, false);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Hello world", result);
}

test "applyChatTemplate renders multi-turn conversation" {
    const allocator = std.testing.allocator;

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const tokenizer_config_json =
        \\{
        \\  "chat_template": "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}",
        \\  "bos_token": "",
        \\  "eos_token": ""
        \\}
    ;

    const messages_json =
        \\[
        \\  {"role": "user", "content": "Hi"},
        \\  {"role": "assistant", "content": "Hello!"},
        \\  {"role": "user", "content": "How are you?"}
        \\]
    ;

    try tmp_dir.dir.writeFile(.{ .sub_path = "tokenizer_config.json", .data = tokenizer_config_json });

    const tmp_path = try tmp_dir.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    const result = try applyChatTemplate(allocator, tmp_path, messages_json, false);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("user: Hi\nassistant: Hello!\nuser: How are you?\n", result);
}

test "applyChatTemplate reads from chat_template.jinja file" {
    const allocator = std.testing.allocator;

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const tokenizer_config_json =
        \\{
        \\  "bos_token": "",
        \\  "eos_token": ""
        \\}
    ;

    const template_content = "Jinja: {{ messages[0].content }}";

    try tmp_dir.dir.writeFile(.{ .sub_path = "tokenizer_config.json", .data = tokenizer_config_json });
    try tmp_dir.dir.writeFile(.{ .sub_path = "chat_template.jinja", .data = template_content });

    const messages_json =
        \\[{"role": "user", "content": "Test message"}]
    ;

    const tmp_path = try tmp_dir.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    const result = try applyChatTemplate(allocator, tmp_path, messages_json, false);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Jinja: Test message", result);
}

test "applyChatTemplate returns error when no template found" {
    const allocator = std.testing.allocator;

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const tokenizer_config_json =
        \\{
        \\  "bos_token": "",
        \\  "eos_token": ""
        \\}
    ;

    try tmp_dir.dir.writeFile(.{ .sub_path = "tokenizer_config.json", .data = tokenizer_config_json });

    const messages_json =
        \\[{"role": "user", "content": "Test"}]
    ;

    const tmp_path = try tmp_dir.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    const result = applyChatTemplate(allocator, tmp_path, messages_json, false);
    try std.testing.expectError(error.MissingChatTemplate, result);
}

test "applyChatTemplate returns error when tokenizer_config.json missing" {
    const allocator = std.testing.allocator;

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const messages_json =
        \\[{"role": "user", "content": "Test"}]
    ;

    const tmp_path = try tmp_dir.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    const result = applyChatTemplate(allocator, tmp_path, messages_json, false);
    try std.testing.expectError(error.FileNotFound, result);
}

test "getChatTemplateSource returns template from tokenizer_config.json" {
    const allocator = std.testing.allocator;

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const expected_template = "{% for msg in messages %}{{ msg.content }}{% endfor %}";

    const tokenizer_config_json =
        \\{
        \\  "chat_template": "{% for msg in messages %}{{ msg.content }}{% endfor %}"
        \\}
    ;

    try tmp_dir.dir.writeFile(.{ .sub_path = "tokenizer_config.json", .data = tokenizer_config_json });

    const tmp_path = try tmp_dir.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    const result = try getChatTemplateSource(allocator, tmp_path);
    defer allocator.free(result);

    try std.testing.expectEqualStrings(expected_template, result);
}

test "getChatTemplateSource returns template from chat_template.jinja file" {
    const allocator = std.testing.allocator;

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const expected_template = "{{ bos_token }}{% for msg in messages %}{{ msg.role }}: {{ msg.content }}\n{% endfor %}";

    const tokenizer_config_json =
        \\{
        \\  "bos_token": "<s>"
        \\}
    ;

    try tmp_dir.dir.writeFile(.{ .sub_path = "tokenizer_config.json", .data = tokenizer_config_json });
    try tmp_dir.dir.writeFile(.{ .sub_path = "chat_template.jinja", .data = expected_template });

    const tmp_path = try tmp_dir.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    const result = try getChatTemplateSource(allocator, tmp_path);
    defer allocator.free(result);

    try std.testing.expectEqualStrings(expected_template, result);
}

test "getChatTemplateSource prefers inline template over file" {
    const allocator = std.testing.allocator;

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const inline_template = "inline template";
    const file_template = "file template";

    const tokenizer_config_json =
        \\{
        \\  "chat_template": "inline template"
        \\}
    ;

    try tmp_dir.dir.writeFile(.{ .sub_path = "tokenizer_config.json", .data = tokenizer_config_json });
    try tmp_dir.dir.writeFile(.{ .sub_path = "chat_template.jinja", .data = file_template });

    const tmp_path = try tmp_dir.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    const result = try getChatTemplateSource(allocator, tmp_path);
    defer allocator.free(result);

    try std.testing.expectEqualStrings(inline_template, result);
}

test "getChatTemplateSource returns error when no template found" {
    const allocator = std.testing.allocator;

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const tokenizer_config_json =
        \\{
        \\  "bos_token": "<s>"
        \\}
    ;

    try tmp_dir.dir.writeFile(.{ .sub_path = "tokenizer_config.json", .data = tokenizer_config_json });

    const tmp_path = try tmp_dir.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    const result = getChatTemplateSource(allocator, tmp_path);
    try std.testing.expectError(error.MissingChatTemplate, result);
}

test "getChatTemplateSource returns error when tokenizer_config.json missing" {
    const allocator = std.testing.allocator;

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const tmp_path = try tmp_dir.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    const result = getChatTemplateSource(allocator, tmp_path);
    try std.testing.expectError(error.FileNotFound, result);
}
