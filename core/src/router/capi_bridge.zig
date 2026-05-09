//! Router C API Shared Types
//!
//! Defines C-compatible generation configuration types and converts them into
//! router engine options. The batch C API owns local generation execution.
//!
//! C API files use these helpers to keep exported functions focused on
//! validation, pointer checks, error mapping, and ownership boundaries.

const std = @import("std");
const local = @import("local.zig");
const inference_bridge = @import("inference_bridge.zig");
const vision_types = inference_bridge.vision_types;
const sampler = inference_bridge.sampling;

const LocalEngine = local.LocalEngine;
const GenerateOptions = local.GenerateOptions;

// =============================================================================
// C-Compatible Input Types
// =============================================================================

/// Logit bias entry from C API.
pub const CLogitBiasEntry = extern struct {
    token_id: u32,
    bias: f32,
};

/// Preprocessed vision image payload from C API callers.
///
/// `pixels_ptr` points to a contiguous array of `pixel_count` f32 values
/// (row-major CHW layout expected by inference backends).
pub const CGenerateVisionImage = extern struct {
    pixels_ptr: ?[*]const f32 = null,
    pixel_count: usize = 0,
    width: u32 = 0,
    height: u32 = 0,
    grid_temporal: u32 = 0,
    grid_height: u32 = 0,
    grid_width: u32 = 0,
    token_count: usize = 0,
};

/// Generation configuration from C API.
pub const CGenerateConfig = extern struct {
    max_tokens: usize = 0,
    /// Maximum tokens for the answer/completion only (0 = unlimited).
    max_completion_tokens: usize = 0,
    /// Maximum thinking/reasoning tokens. maxInt = unset (use effort), 0 = no thinking.
    max_reasoning_tokens: usize = std.math.maxInt(usize),
    temperature: f32 = -1.0,
    top_k: usize = 0,
    top_p: f32 = -1.0,
    min_p: f32 = -1.0,
    repetition_penalty: f32 = -1.0,
    presence_penalty: f32 = -1.0,
    frequency_penalty: f32 = -1.0,
    stop_sequences: ?[*]const [*:0]const u8 = null,
    stop_sequence_count: usize = 0,
    logit_bias: ?[*]const CLogitBiasEntry = null,
    logit_bias_count: usize = 0,
    seed: u64 = 0,
    template_override: ?[*:0]const u8 = null,
    extra_context_json: ?[*:0]const u8 = null,
    /// Reasoning effort level: "none", "low", "medium", "high", "xhigh".
    /// Mapped to template variables (e.g. enable_thinking) by the router.
    reasoning_effort: ?[*:0]const u8 = null,
    /// Tool definitions as JSON array. Format: [{"type":"function","function":{...}}]
    tools_json: ?[*:0]const u8 = null,
    /// Tool choice strategy: "auto", "required", "none", or specific function name.
    tool_choice: ?[*:0]const u8 = null,
    /// Optional stop flag for cancellation. When set to true (non-zero), generation stops.
    /// This allows external cancellation (e.g., client disconnect, asyncio.CancelledError)
    /// without waiting for the next callback invocation.
    /// The pointer must remain valid for the duration of generation.
    stop_flag: ?*const std.atomic.Value(bool) = null,
    /// Reserved for compatibility; ignored by local inference.
    extra_body_json: ?[*:0]const u8 = null,

    /// When non-zero, preserve raw model output bytes without reasoning-tag filtering.
    /// Default (0) keeps current behavior (reasoning tags parsed into typed items).
    raw_output: usize = 0,

    /// When non-zero, behave like a standard completions endpoint: no thinking
    /// intervention, no reasoning separation, just raw token generation with
    /// max_tokens as the sole cap.
    completions_mode: usize = 0,

    /// Optional prefill progress callback. Called once per transformer layer
    /// during prefill (not decode). Signature: fn(completed_layers, total_layers, userdata).
    prefill_progress_fn: ?*const fn (usize, usize, ?*anyopaque) callconv(.c) void = null,
    prefill_progress_data: ?*anyopaque = null,

    /// Optional externally preprocessed vision payload.
    /// When present, local image decode/preprocess is skipped.
    external_vision_images: ?[*]const CGenerateVisionImage = null,
    external_vision_image_count: usize = 0,
    external_vision_image_token_id: u32 = 0,
    _external_vision_padding: u32 = 0,
};

// =============================================================================
// Result Types
// =============================================================================

/// C-compatible tool call reference.
pub const CToolCallRef = extern struct {
    /// Index in Conversation.items
    item_index: usize,
    /// Unique identifier for this call (null-terminated).
    call_id: ?[*:0]const u8,
    /// Function name (null-terminated).
    name: ?[*:0]const u8,
    /// Function arguments (null-terminated JSON string).
    arguments: ?[*:0]const u8,
};

/// Finish reason enum values (matches inference/root.zig FinishReason).
pub const CFinishReason = enum(u8) {
    /// Generation stopped due to EOS token.
    eos_token = 0,
    /// Maximum token limit reached.
    length = 1,
    /// A stop sequence was matched.
    stop_sequence = 2,
    /// Model requested tool/function calls.
    tool_calls = 3,
    /// Content was filtered (safety).
    content_filter = 4,
    /// Request was cancelled (e.g., client disconnect, stop flag set).
    cancelled = 5,
};

// =============================================================================
// Internal Implementation
// =============================================================================

/// Built options with allocated resources that need cleanup.
pub const BuiltOptions = struct {
    options: GenerateOptions,
    stop_sequences: [][]u32,
    logit_bias: []sampler.LogitBiasEntry,
    external_vision_input: ?vision_types.PrefillVisionInput,

    pub fn deinit(self: *BuiltOptions, allocator: std.mem.Allocator) void {
        for (self.stop_sequences) |seq| allocator.free(seq);
        if (self.stop_sequences.len > 0) allocator.free(self.stop_sequences);
        if (self.logit_bias.len > 0) allocator.free(self.logit_bias);
        if (self.external_vision_input) |*vision_input| vision_input.deinit(allocator);
    }
};

fn copyExternalVisionInput(
    allocator: std.mem.Allocator,
    cfg: *const CGenerateConfig,
) !?vision_types.PrefillVisionInput {
    if (cfg.external_vision_images == null or cfg.external_vision_image_count == 0) return null;
    if (cfg.external_vision_image_token_id == 0) return error.InvalidArgument;

    const src_images = cfg.external_vision_images.?[0..cfg.external_vision_image_count];
    const images = try allocator.alloc(vision_types.PrefillVisionImage, src_images.len);

    var written: usize = 0;
    errdefer {
        for (images[0..written]) |*img| img.deinit(allocator);
        allocator.free(images);
    }

    for (src_images, 0..) |src_img, i| {
        if (src_img.width == 0 or src_img.height == 0) return error.InvalidArgument;
        if (src_img.token_count == 0) return error.InvalidArgument;
        if (src_img.pixel_count > 0 and src_img.pixels_ptr == null) return error.InvalidArgument;

        const src_pixels: []const f32 = if (src_img.pixel_count > 0)
            src_img.pixels_ptr.?[0..src_img.pixel_count]
        else
            &.{};

        const copied_pixels = try allocator.dupe(f32, src_pixels);
        images[i] = .{
            .pixels = copied_pixels,
            .width = src_img.width,
            .height = src_img.height,
            .grid = .{
                .temporal = src_img.grid_temporal,
                .height = src_img.grid_height,
                .width = src_img.grid_width,
            },
            .token_count = src_img.token_count,
        };
        written += 1;
    }

    return vision_types.PrefillVisionInput{
        .images = images,
        .image_token_id = cfg.external_vision_image_token_id,
    };
}

/// Build GenerateOptions from C config.
pub fn buildOptions(
    allocator: std.mem.Allocator,
    config: ?*const CGenerateConfig,
    engine: *LocalEngine,
) !BuiltOptions {
    var result = BuiltOptions{
        .options = .{},
        .stop_sequences = &.{},
        .logit_bias = &.{},
        .external_vision_input = null,
    };

    const cfg = config orelse return result;

    if (cfg.max_tokens > 0) result.options.max_tokens = cfg.max_tokens;
    if (cfg.max_completion_tokens > 0) result.options.max_completion_tokens = cfg.max_completion_tokens;
    if (cfg.max_reasoning_tokens != std.math.maxInt(usize)) result.options.max_reasoning_tokens = cfg.max_reasoning_tokens;
    if (cfg.temperature >= 0) result.options.temperature = cfg.temperature;
    if (cfg.top_k > 0) result.options.top_k = cfg.top_k;
    if (cfg.top_p >= 0) result.options.top_p = cfg.top_p;
    if (cfg.min_p >= 0) result.options.min_p = cfg.min_p;
    if (cfg.repetition_penalty >= 0) result.options.repetition_penalty = cfg.repetition_penalty;
    if (cfg.presence_penalty >= 0) result.options.presence_penalty = cfg.presence_penalty;
    if (cfg.frequency_penalty >= 0) result.options.frequency_penalty = cfg.frequency_penalty;
    if (cfg.seed != 0) result.options.seed = cfg.seed;

    if (cfg.template_override) |t| result.options.template_override = std.mem.sliceTo(t, 0);
    if (cfg.extra_context_json) |j| result.options.extra_context_json = std.mem.sliceTo(j, 0);
    if (cfg.reasoning_effort) |r| result.options.reasoning_effort = std.mem.sliceTo(r, 0);
    if (cfg.tools_json) |t| result.options.tools_json = std.mem.sliceTo(t, 0);
    if (cfg.tool_choice) |c| result.options.tool_choice = std.mem.sliceTo(c, 0);
    result.options.raw_output = cfg.raw_output != 0;
    result.options.completions_mode = cfg.completions_mode != 0;

    // Pass through stop flag for cancellation support
    result.options.stop_flag = cfg.stop_flag;

    // Pass through prefill progress callback
    result.options.prefill_progress_fn = cfg.prefill_progress_fn;
    result.options.prefill_progress_data = cfg.prefill_progress_data;

    if (try copyExternalVisionInput(allocator, cfg)) |external_vision_input| {
        result.external_vision_input = external_vision_input;
        result.options.external_vision_input = &result.external_vision_input.?;
    }

    // Tokenize stop sequences
    if (cfg.stop_sequences != null and cfg.stop_sequence_count > 0) {
        const seqs = cfg.stop_sequences.?[0..cfg.stop_sequence_count];
        result.stop_sequences = try allocator.alloc([]u32, seqs.len);
        for (seqs, 0..) |seq_ptr, i| {
            result.stop_sequences[i] = try engine.encode(std.mem.sliceTo(seq_ptr, 0));
        }
        result.options.stop_sequences = @as([]const []const u32, result.stop_sequences);
    }

    // Convert logit bias
    if (cfg.logit_bias != null and cfg.logit_bias_count > 0) {
        const entries = cfg.logit_bias.?[0..cfg.logit_bias_count];
        result.logit_bias = try allocator.alloc(sampler.LogitBiasEntry, entries.len);
        for (entries, 0..) |e, i| {
            result.logit_bias[i] = .{ .token_id = e.token_id, .bias = e.bias };
        }
        result.options.logit_bias = result.logit_bias;
    }

    return result;
}

// =============================================================================
// Tests
// =============================================================================

test "CGenerateConfig defaults" {
    // Test Zig default initialization (what Zig code gets)
    const cfg: CGenerateConfig = .{};
    try std.testing.expectEqual(@as(usize, 0), cfg.max_tokens);
    try std.testing.expectEqual(@as(f32, -1.0), cfg.temperature);
    try std.testing.expectEqual(@as(usize, 0), cfg.top_k);
    try std.testing.expectEqual(@as(f32, -1.0), cfg.top_p);
    try std.testing.expectEqual(@as(f32, -1.0), cfg.min_p);
    try std.testing.expectEqual(@as(f32, -1.0), cfg.repetition_penalty);
    try std.testing.expectEqual(@as(f32, -1.0), cfg.presence_penalty);
    try std.testing.expectEqual(@as(f32, -1.0), cfg.frequency_penalty);
    try std.testing.expect(cfg.stop_sequences == null);
    try std.testing.expectEqual(@as(usize, 0), cfg.stop_sequence_count);
    try std.testing.expect(cfg.logit_bias == null);
    try std.testing.expectEqual(@as(usize, 0), cfg.logit_bias_count);
    try std.testing.expectEqual(@as(u64, 0), cfg.seed);
    try std.testing.expect(cfg.template_override == null);
    try std.testing.expect(cfg.extra_context_json == null);
    try std.testing.expect(cfg.extra_body_json == null);
    try std.testing.expectEqual(@as(usize, 0), cfg.raw_output);
    try std.testing.expectEqual(@as(usize, 0), cfg.completions_mode);
    try std.testing.expect(cfg.external_vision_images == null);
    try std.testing.expectEqual(@as(usize, 0), cfg.external_vision_image_count);
    try std.testing.expectEqual(@as(u32, 0), cfg.external_vision_image_token_id);
}

test "buildOptions accepts null config without allocations" {
    const allocator = std.testing.allocator;
    const engine: *LocalEngine = @ptrFromInt(1);

    var built = try buildOptions(allocator, null, engine);
    defer built.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), built.stop_sequences.len);
    try std.testing.expectEqual(@as(usize, 0), built.logit_bias.len);
    try std.testing.expect(built.external_vision_input == null);
}

test "buildOptions preserves completions_mode" {
    const allocator = std.testing.allocator;
    const engine: *LocalEngine = @ptrFromInt(1);
    var cfg = std.mem.zeroes(CGenerateConfig);
    cfg.completions_mode = 1;
    cfg.max_reasoning_tokens = std.math.maxInt(usize);
    cfg.temperature = -1.0;
    cfg.top_p = -1.0;
    cfg.min_p = -1.0;
    cfg.repetition_penalty = -1.0;
    cfg.presence_penalty = -1.0;
    cfg.frequency_penalty = -1.0;

    var built = try buildOptions(allocator, &cfg, engine);
    defer built.deinit(allocator);

    try std.testing.expect(built.options.completions_mode);
}

test "copyExternalVisionInput copies payload" {
    const allocator = std.testing.allocator;
    const pixels = [_]f32{ 0.25, 0.5, 0.75, 1.0 };
    const c_images = [_]CGenerateVisionImage{
        .{
            .pixels_ptr = pixels[0..].ptr,
            .pixel_count = pixels.len,
            .width = 2,
            .height = 2,
            .grid_temporal = 1,
            .grid_height = 1,
            .grid_width = 1,
            .token_count = 3,
        },
    };
    var cfg: CGenerateConfig = .{};
    cfg.external_vision_images = c_images[0..].ptr;
    cfg.external_vision_image_count = c_images.len;
    cfg.external_vision_image_token_id = 1234;

    var copied = (try copyExternalVisionInput(allocator, &cfg)).?;
    defer copied.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), copied.images.len);
    try std.testing.expectEqual(@as(u32, 1234), copied.image_token_id);
    try std.testing.expectEqual(@as(u32, 2), copied.images[0].width);
    try std.testing.expectEqual(@as(usize, 3), copied.images[0].token_count);
    try std.testing.expectEqual(@as(usize, pixels.len), copied.images[0].pixels.len);
    try std.testing.expectEqual(@as(f32, 0.75), copied.images[0].pixels[2]);
}

test "copyExternalVisionInput rejects missing image token id" {
    const allocator = std.testing.allocator;
    const pixels = [_]f32{1.0};
    const c_images = [_]CGenerateVisionImage{
        .{
            .pixels_ptr = pixels[0..].ptr,
            .pixel_count = pixels.len,
            .width = 1,
            .height = 1,
            .grid_temporal = 1,
            .grid_height = 1,
            .grid_width = 1,
            .token_count = 1,
        },
    };
    var cfg: CGenerateConfig = .{};
    cfg.external_vision_images = c_images[0..].ptr;
    cfg.external_vision_image_count = c_images.len;
    cfg.external_vision_image_token_id = 0;

    try std.testing.expectError(error.InvalidArgument, copyExternalVisionInput(allocator, &cfg));
}

test "CLogitBiasEntry struct layout" {
    var entry = std.mem.zeroes(CLogitBiasEntry);
    entry.token_id = 42;
    entry.bias = -100.0;
    try std.testing.expectEqual(@as(u32, 42), entry.token_id);
    try std.testing.expectEqual(@as(f32, -100.0), entry.bias);
}

// -----------------------------------------------------------------------------
// Tool Call Tests
// -----------------------------------------------------------------------------

test "CToolCallRef struct layout" {
    var ref = std.mem.zeroes(CToolCallRef);
    ref.item_index = 42;
    try std.testing.expectEqual(@as(usize, 42), ref.item_index);
    try std.testing.expect(ref.call_id == null);
    try std.testing.expect(ref.name == null);
    try std.testing.expect(ref.arguments == null);
}

test "CFinishReason enum values" {
    try std.testing.expectEqual(@as(u8, 0), @intFromEnum(CFinishReason.eos_token));
    try std.testing.expectEqual(@as(u8, 1), @intFromEnum(CFinishReason.length));
    try std.testing.expectEqual(@as(u8, 2), @intFromEnum(CFinishReason.stop_sequence));
    try std.testing.expectEqual(@as(u8, 3), @intFromEnum(CFinishReason.tool_calls));
    try std.testing.expectEqual(@as(u8, 4), @intFromEnum(CFinishReason.content_filter));
}

test "CGenerateConfig: tools_json and tool_choice fields" {
    const cfg = std.mem.zeroes(CGenerateConfig);
    try std.testing.expect(cfg.tools_json == null);
    try std.testing.expect(cfg.tool_choice == null);

    var cfg2 = std.mem.zeroes(CGenerateConfig);
    cfg2.tools_json = "tools";
    cfg2.tool_choice = "auto";
    try std.testing.expect(cfg2.tools_json != null);
    try std.testing.expectEqualStrings("tools", std.mem.sliceTo(cfg2.tools_json.?, 0));
    try std.testing.expectEqualStrings("auto", std.mem.sliceTo(cfg2.tool_choice.?, 0));
    cfg2.raw_output = 1;
    try std.testing.expectEqual(@as(u8, 1), cfg2.raw_output);
}
