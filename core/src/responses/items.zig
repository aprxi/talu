//! Items - Core data types for the Open Responses architecture.
//!
//! This module defines the Item-Based data model that is a strict superset of
//! the OpenAI Responses API v2.3.0 `ItemParam` specification.
//!
//! # Architecture
//!
//! The data model stores atomic "Items" (Thoughts, Tool Intents, Results, Messages)
//! rather than grouped "Turns". This enables:
//!   - **Responses Projection:** Zero-cost pass-through for modern endpoints
//!   - **Completions Projection:** Folding transform for legacy endpoints
//!
//! # Schema Mapping
//!
//! All enums and structs map 1:1 to the `openapi.json` schemas to ensure
//! lossless serialization. See `env/openapi.json` for the source spec.
//!
//! Thread safety: NOT thread-safe. All access must be from a single thread.

const std = @import("std");

// =============================================================================
// Enums and Discriminators
// =============================================================================

/// The processing status of an item.
/// Maps to Schema: "status" / "FunctionCallItemStatus".
///
/// All item types are state machines that transition through these states:
/// - `in_progress`: Currently being streamed/processed
/// - `waiting`: Waiting on tool or external action (agent UI can render a spinner/CTA)
/// - `completed`: Successfully finished
/// - `incomplete`: Interrupted or truncated (e.g., max tokens reached)
/// - `failed`: Processing failed (e.g., content filter, error)
pub const ItemStatus = enum(u8) {
    in_progress = 0,
    waiting = 1,
    completed = 2,
    incomplete = 3,
    failed = 4,

    pub fn toString(self: ItemStatus) []const u8 {
        return switch (self) {
            .in_progress => "in_progress",
            .waiting => "waiting",
            .completed => "completed",
            .incomplete => "incomplete",
            .failed => "failed",
        };
    }

    pub fn fromString(s: []const u8) ?ItemStatus {
        if (std.mem.eql(u8, s, "in_progress")) return .in_progress;
        if (std.mem.eql(u8, s, "waiting")) return .waiting;
        if (std.mem.eql(u8, s, "completed")) return .completed;
        if (std.mem.eql(u8, s, "incomplete")) return .incomplete;
        if (std.mem.eql(u8, s, "failed")) return .failed;
        return null;
    }
};

/// The discriminator for the Item Variant.
/// Maps to Schema: "type" field on "ItemParam" oneOf.
pub const ItemType = enum(u8) {
    /// A chat message (user, assistant, system, developer).
    /// Schema: "message" (UserMessageItemParam, AssistantMessageItemParam, etc.)
    message = 0,

    /// A function/tool call intent from the assistant.
    /// Schema: "function_call" (FunctionCallItemParam)
    function_call = 1,

    /// The output/result of a function/tool call.
    /// Schema: "function_call_output" (FunctionCallOutputItemParam)
    function_call_output = 2,

    /// Reasoning content (chain-of-thought, o1/o3 models).
    /// Schema: "reasoning" (ReasoningItemParam)
    reasoning = 3,

    /// Reference to a previous item (for context replay).
    /// Schema: "item_reference" (ItemReferenceParam)
    item_reference = 4,

    /// Unknown/unrecognized item type (for forward compatibility).
    unknown = 255,

    pub fn toString(self: ItemType) []const u8 {
        return switch (self) {
            .message => "message",
            .function_call => "function_call",
            .function_call_output => "function_call_output",
            .reasoning => "reasoning",
            .item_reference => "item_reference",
            .unknown => "unknown",
        };
    }

    pub fn fromString(s: []const u8) ItemType {
        if (std.mem.eql(u8, s, "message")) return .message;
        if (std.mem.eql(u8, s, "function_call")) return .function_call;
        if (std.mem.eql(u8, s, "function_call_output")) return .function_call_output;
        if (std.mem.eql(u8, s, "reasoning")) return .reasoning;
        if (std.mem.eql(u8, s, "item_reference")) return .item_reference;
        return .unknown;
    }
};

/// The role of a Message.
/// Maps to Schema: "MessageRole".
///
/// The `unknown` variant (255) enables forward compatibility: if a provider
/// returns a new role (e.g., "critic", "tool"), we can round-trip it without
/// data loss. The original role string is stored in MessageData.raw_role.
pub const MessageRole = enum(u8) {
    system = 0,
    user = 1,
    assistant = 2,
    developer = 3, // Required for o1/o3 models
    unknown = 255, // Forward compatibility

    pub fn toString(self: MessageRole) []const u8 {
        return switch (self) {
            .system => "system",
            .user => "user",
            .assistant => "assistant",
            .developer => "developer",
            .unknown => "unknown",
        };
    }

    /// Parse role from string. Returns the specific role if known,
    /// or `.unknown` for unrecognized roles (enabling forward compatibility).
    /// When `.unknown` is returned, caller should store the raw string in
    /// MessageData.raw_role for round-tripping.
    pub fn fromString(s: []const u8) MessageRole {
        if (std.mem.eql(u8, s, "system")) return .system;
        if (std.mem.eql(u8, s, "user")) return .user;
        if (std.mem.eql(u8, s, "assistant")) return .assistant;
        if (std.mem.eql(u8, s, "developer")) return .developer;
        return .unknown;
    }

    /// Normalize role for Completions projection.
    /// `developer` maps to `system` for backends that don't support it.
    /// `unknown` maps to `user` (safest default).
    pub fn toLegacyRole(self: MessageRole) MessageRole {
        return switch (self) {
            .developer => .system,
            .unknown => .user,
            else => self,
        };
    }

    /// Check if this is a known role.
    pub fn isKnown(self: MessageRole) bool {
        return self != .unknown;
    }
};

/// Content type discriminators.
/// Strictly maps to the mutually exclusive content types in openapi.json.
///
/// NOTE: url_citation is NOT a content type - it's an annotation nested inside
/// output_text. See UrlCitation struct and output_text.annotations_json.
///
/// The `unknown` variant (255) enables forward compatibility: if a provider
/// sends a new content type (e.g., "input_3d_model"), we can round-trip it
/// without data loss instead of crashing or discarding.
pub const ContentType = enum(u8) {
    // --- Inputs (User/System/Developer) ---
    /// Schema: "InputTextContentParam"
    input_text = 0,
    /// Schema: "InputImageContentParamAutoParam"
    input_image = 1,
    /// Extension for audio input
    input_audio = 2,
    /// Schema: "InputVideoContent" (in FunctionCallOutput)
    input_video = 3,
    /// Schema: "InputFileContentParam"
    input_file = 4,

    // --- Outputs (Assistant) ---
    /// Schema: "OutputTextContentParam" (Has logprobs/annotations)
    output_text = 5,
    /// Schema: "RefusalContentParam"
    refusal = 6,

    // --- Reasoning / Generic ---
    /// Schema: "TextContent" (Required inside ReasoningBody)
    text = 7,
    /// Schema: "ReasoningTextContent"
    reasoning_text = 8,
    /// Schema: "ReasoningSummaryContentParam"
    summary_text = 9,

    // --- Forward Compatibility ---
    /// Unknown/unrecognized content type (for forward compatibility).
    /// Allows round-tripping provider-specific content types without data loss.
    unknown = 255,

    pub fn toString(self: ContentType) []const u8 {
        return switch (self) {
            .input_text => "input_text",
            .input_image => "input_image",
            .input_audio => "input_audio",
            .input_video => "input_video",
            .input_file => "input_file",
            .output_text => "output_text",
            .refusal => "refusal",
            .text => "text",
            .reasoning_text => "reasoning_text",
            .summary_text => "summary_text",
            .unknown => "unknown",
        };
    }

    /// Parse content type from string. Returns the specific type if known,
    /// or `.unknown` for unrecognized types (enabling forward compatibility).
    pub fn fromString(s: []const u8) ContentType {
        if (std.mem.eql(u8, s, "input_text")) return .input_text;
        if (std.mem.eql(u8, s, "input_image")) return .input_image;
        if (std.mem.eql(u8, s, "input_audio")) return .input_audio;
        if (std.mem.eql(u8, s, "input_video")) return .input_video;
        if (std.mem.eql(u8, s, "input_file")) return .input_file;
        if (std.mem.eql(u8, s, "output_text")) return .output_text;
        if (std.mem.eql(u8, s, "refusal")) return .refusal;
        if (std.mem.eql(u8, s, "text")) return .text;
        if (std.mem.eql(u8, s, "reasoning_text")) return .reasoning_text;
        if (std.mem.eql(u8, s, "summary_text")) return .summary_text;
        return .unknown;
    }

    /// Check if this content type is an input type.
    pub fn isInput(self: ContentType) bool {
        return switch (self) {
            .input_text, .input_image, .input_audio, .input_video, .input_file => true,
            else => false,
        };
    }

    /// Check if this content type is an output type.
    pub fn isOutput(self: ContentType) bool {
        return switch (self) {
            .output_text, .refusal => true,
            else => false,
        };
    }

    /// Check if this is an unknown/unrecognized content type.
    pub fn isUnknown(self: ContentType) bool {
        return self == .unknown;
    }
};

/// Image detail level for input_image content.
/// Schema: "DetailEnum" / "ImageDetail"
pub const ImageDetail = enum(u8) {
    auto = 0,
    low = 1,
    high = 2,

    pub fn toString(self: ImageDetail) []const u8 {
        return switch (self) {
            .auto => "auto",
            .low => "low",
            .high => "high",
        };
    }

    pub fn fromString(s: []const u8) ?ImageDetail {
        if (std.mem.eql(u8, s, "auto")) return .auto;
        if (std.mem.eql(u8, s, "low")) return .low;
        if (std.mem.eql(u8, s, "high")) return .high;
        return null;
    }
};

// =============================================================================
// Content Structures - Tagged Union for Schema Fidelity
// =============================================================================
//
// ContentPart uses a tagged union (ContentVariant) to ensure each content type
// stores exactly the fields defined in the OpenAPI schema. This prevents the
// "bag-of-optional-fields" anti-pattern that loses type safety.
//
// Schema mapping:
//   - input_text -> InputTextContentParam
//   - input_image -> InputImageContentParamAutoParam
//   - input_file -> InputFileContentParam
//   - output_text -> OutputTextContentParam (contains annotations as UrlCitation[])
//   - refusal -> RefusalContentParam
//   - text -> TextContent
//   - reasoning_text -> ReasoningTextContent
//   - summary_text -> ReasoningSummaryContentParam
//
// NOTE: UrlCitation is NOT a content type - it's nested inside output_text.annotations

/// URL Citation annotation (nested inside output_text, NOT a top-level content type).
/// Schema: UrlCitationParam { type: "url_citation", url, title, start_index, end_index }
///
/// These appear in the `annotations` array of OutputTextContentParam.
/// When serializing output_text, annotations should be emitted as:
///   { "type": "url_citation", "url": "...", "title": "...", "start_index": N, "end_index": M }
pub const UrlCitation = struct {
    start_index: u32,
    end_index: u32,
    url: []const u8,
    title: []const u8,

    pub fn deinit(self: *UrlCitation, allocator: std.mem.Allocator) void {
        if (self.url.len > 0) allocator.free(self.url);
        if (self.title.len > 0) allocator.free(self.title);
    }

    /// Create a UrlCitation with owned copies of url and title.
    pub fn init(allocator: std.mem.Allocator, url: []const u8, title: []const u8, start_index: u32, end_index: u32) !UrlCitation {
        return .{
            .url = try allocator.dupe(u8, url),
            .title = try allocator.dupe(u8, title),
            .start_index = start_index,
            .end_index = end_index,
        };
    }
};

/// Tagged union for content-type-specific data.
/// Each variant contains exactly the fields defined in the OpenAPI schema.
pub const ContentVariant = union(ContentType) {
    /// Schema: InputTextContentParam { type: "input_text", text: string }
    input_text: struct {
        text: std.ArrayListUnmanaged(u8) = .{},

        pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
            self.text.deinit(allocator);
        }

        pub fn getText(self: *const @This()) []const u8 {
            return self.text.items;
        }
    },

    /// Schema: InputImageContentParamAutoParam { type: "input_image", image_url: string, detail?: enum }
    input_image: struct {
        image_url: std.ArrayListUnmanaged(u8) = .{},
        detail: ImageDetail = .auto,

        pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
            self.image_url.deinit(allocator);
        }

        pub fn getUrl(self: *const @This()) []const u8 {
            return self.image_url.items;
        }
    },

    /// Extension for audio input (not in OpenAPI but supported)
    input_audio: struct {
        audio_data: std.ArrayListUnmanaged(u8) = .{},

        pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
            self.audio_data.deinit(allocator);
        }

        pub fn getData(self: *const @This()) []const u8 {
            return self.audio_data.items;
        }
    },

    /// Schema: InputVideoContent { type: "input_video", video_url: string }
    input_video: struct {
        video_url: std.ArrayListUnmanaged(u8) = .{},

        pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
            self.video_url.deinit(allocator);
        }

        pub fn getUrl(self: *const @This()) []const u8 {
            return self.video_url.items;
        }
    },

    /// Schema: InputFileContentParam { type: "input_file", filename?: string, file_data?: string, file_url?: string }
    input_file: struct {
        filename: ?[]const u8 = null,
        file_data: ?std.ArrayListUnmanaged(u8) = null,
        file_url: ?std.ArrayListUnmanaged(u8) = null,

        pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
            if (self.filename) |f| allocator.free(f);
            if (self.file_data) |*d| d.deinit(allocator);
            if (self.file_url) |*u| u.deinit(allocator);
        }

        pub fn getFilename(self: *const @This()) ?[]const u8 {
            return self.filename;
        }

        pub fn getFileData(self: *const @This()) ?[]const u8 {
            if (self.file_data) |d| return d.items;
            return null;
        }

        pub fn getFileUrl(self: *const @This()) ?[]const u8 {
            if (self.file_url) |u| return u.items;
            return null;
        }
    },

    /// Schema: OutputTextContentParam { type: "output_text", text: string, logprobs?: array, annotations?: array }
    output_text: struct {
        text: std.ArrayListUnmanaged(u8) = .{},
        /// JSON-encoded logprobs array, or null.
        logprobs_json: ?[]const u8 = null,
        /// JSON-encoded annotations array (UrlCitationParam[]), or null.
        annotations_json: ?[]const u8 = null,
        /// JSON-encoded code blocks array (detected fenced code blocks), or null.
        /// Set at finalization time by extractCodeBlocks().
        code_blocks_json: ?[]const u8 = null,

        pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
            self.text.deinit(allocator);
            if (self.logprobs_json) |l| allocator.free(l);
            if (self.annotations_json) |a| allocator.free(a);
            if (self.code_blocks_json) |c| allocator.free(c);
        }

        pub fn getText(self: *const @This()) []const u8 {
            return self.text.items;
        }
    },

    /// Schema: RefusalContentParam { type: "refusal", refusal: string }
    refusal: struct {
        refusal: std.ArrayListUnmanaged(u8) = .{},

        pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
            self.refusal.deinit(allocator);
        }

        pub fn getText(self: *const @This()) []const u8 {
            return self.refusal.items;
        }
    },

    /// Schema: TextContent { type: "text", text: string }
    /// Used in ReasoningBody.content and generic text contexts.
    text: struct {
        text: std.ArrayListUnmanaged(u8) = .{},

        pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
            self.text.deinit(allocator);
        }

        pub fn getText(self: *const @This()) []const u8 {
            return self.text.items;
        }
    },

    /// Schema: ReasoningTextContent (thinking content in reasoning)
    reasoning_text: struct {
        text: std.ArrayListUnmanaged(u8) = .{},

        pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
            self.text.deinit(allocator);
        }

        pub fn getText(self: *const @This()) []const u8 {
            return self.text.items;
        }
    },

    /// Schema: ReasoningSummaryContentParam { type: "summary_text", text: string }
    summary_text: struct {
        text: std.ArrayListUnmanaged(u8) = .{},

        pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
            self.text.deinit(allocator);
        }

        pub fn getText(self: *const @This()) []const u8 {
            return self.text.items;
        }
    },

    /// Unknown/unrecognized content type (for forward compatibility).
    /// Stores the raw type string and payload to enable round-tripping without data loss.
    unknown: struct {
        /// The raw type string that was not recognized.
        raw_type: []const u8 = "",
        /// Raw data payload for the unrecognized content.
        raw_data: std.ArrayListUnmanaged(u8) = .{},

        pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
            if (self.raw_type.len > 0) allocator.free(self.raw_type);
            self.raw_data.deinit(allocator);
        }

        pub fn getText(self: *const @This()) []const u8 {
            return self.raw_data.items;
        }
    },

    // REMOVED: url_citation - it's an annotation nested inside output_text, not a content type
    // Use the UrlCitation helper struct and store in output_text.annotations_json

    /// Free all owned memory based on active variant.
    pub fn deinit(self: *ContentVariant, allocator: std.mem.Allocator) void {
        switch (self.*) {
            inline else => |*v| v.deinit(allocator),
        }
    }

    /// Get the content type tag.
    pub fn getType(self: *const ContentVariant) ContentType {
        return std.meta.activeTag(self.*);
    }

    /// Get text content if this variant contains text.
    /// Returns the primary text field for text-like content types.
    pub fn getText(self: *const ContentVariant) []const u8 {
        return switch (self.*) {
            .input_text => |*v| v.getText(),
            .output_text => |*v| v.getText(),
            .text => |*v| v.getText(),
            .reasoning_text => |*v| v.getText(),
            .summary_text => |*v| v.getText(),
            .refusal => |*v| v.getText(),
            .input_image => |*v| v.getUrl(),
            .input_audio => |*v| v.getData(),
            .input_video => |*v| v.getUrl(),
            .input_file => |v| v.getFileData() orelse v.getFileUrl() orelse "",
            .unknown => |*v| v.getText(),
        };
    }
};

/// A single content part within a message or tool output.
///
/// ContentPart wraps ContentVariant to provide a uniform interface while
/// maintaining schema fidelity through the tagged union.
///
/// Thread safety: NOT thread-safe. All access must be from a single thread.
pub const ContentPart = struct {
    /// The content data stored as a tagged union.
    variant: ContentVariant,

    /// Free all owned memory.
    pub fn deinit(self: *ContentPart, allocator: std.mem.Allocator) void {
        self.variant.deinit(allocator);
    }

    /// Get content type discriminator.
    pub fn getContentType(self: *const ContentPart) ContentType {
        return self.variant.getType();
    }

    // Legacy compatibility alias
    pub const content_type = getContentType;

    /// Get data as slice (legacy compatibility).
    /// Returns the primary text/data field from the variant.
    pub fn getData(self: *const ContentPart) []const u8 {
        return self.variant.getText();
    }

    /// Get image detail level (returns .auto for non-image types).
    pub fn getImageDetail(self: *const ContentPart) ImageDetail {
        return switch (self.variant) {
            .input_image => |v| v.detail,
            else => .auto,
        };
    }

    // Legacy compatibility - image_detail field accessor
    pub const image_detail = getImageDetail;

    /// Append data to this part (for streaming).
    /// Only works for text-based content types.
    pub fn appendData(self: *ContentPart, allocator: std.mem.Allocator, bytes: []const u8) !void {
        switch (self.variant) {
            .input_text => |*v| try v.text.appendSlice(allocator, bytes),
            .output_text => |*v| try v.text.appendSlice(allocator, bytes),
            .text => |*v| try v.text.appendSlice(allocator, bytes),
            .reasoning_text => |*v| try v.text.appendSlice(allocator, bytes),
            .summary_text => |*v| try v.text.appendSlice(allocator, bytes),
            .refusal => |*v| try v.refusal.appendSlice(allocator, bytes),
            .input_image => |*v| try v.image_url.appendSlice(allocator, bytes),
            .input_audio => |*v| try v.audio_data.appendSlice(allocator, bytes),
            .input_video => |*v| try v.video_url.appendSlice(allocator, bytes),
            .input_file => |*v| {
                // For files, append to file_data
                if (v.file_data == null) {
                    v.file_data = .{};
                }
                try v.file_data.?.appendSlice(allocator, bytes);
            },
            .unknown => |*v| try v.raw_data.appendSlice(allocator, bytes),
        }
    }

    /// Clear data from this part (for updating content).
    /// Only works for text-based content types.
    pub fn clearData(self: *ContentPart) void {
        switch (self.variant) {
            .input_text => |*v| v.text.clearRetainingCapacity(),
            .output_text => |*v| v.text.clearRetainingCapacity(),
            .text => |*v| v.text.clearRetainingCapacity(),
            .reasoning_text => |*v| v.text.clearRetainingCapacity(),
            .summary_text => |*v| v.text.clearRetainingCapacity(),
            .refusal => |*v| v.refusal.clearRetainingCapacity(),
            .input_image => |*v| v.image_url.clearRetainingCapacity(),
            .input_audio => |*v| v.audio_data.clearRetainingCapacity(),
            .input_video => |*v| v.video_url.clearRetainingCapacity(),
            .input_file => |*v| if (v.file_data) |*fd| fd.clearRetainingCapacity(),
            .unknown => |*v| v.raw_data.clearRetainingCapacity(),
        }
    }

    /// Create a deep copy of this ContentPart.
    pub fn clone(self: *const ContentPart, allocator: std.mem.Allocator) !ContentPart {
        var result: ContentPart = .{
            .variant = undefined,
        };

        result.variant = switch (self.variant) {
            .input_text => |v| .{ .input_text = .{ .text = blk: {
                var list: std.ArrayListUnmanaged(u8) = .{};
                try list.appendSlice(allocator, v.text.items);
                break :blk list;
            } } },
            .output_text => |v| .{ .output_text = .{
                .text = blk: {
                    var list: std.ArrayListUnmanaged(u8) = .{};
                    try list.appendSlice(allocator, v.text.items);
                    break :blk list;
                },
                .logprobs_json = if (v.logprobs_json) |lp| try allocator.dupe(u8, lp) else null,
                .annotations_json = if (v.annotations_json) |an| try allocator.dupe(u8, an) else null,
            } },
            .text => |v| .{ .text = .{ .text = blk: {
                var list: std.ArrayListUnmanaged(u8) = .{};
                try list.appendSlice(allocator, v.text.items);
                break :blk list;
            } } },
            .input_image => |v| .{ .input_image = .{
                .image_url = blk: {
                    var list: std.ArrayListUnmanaged(u8) = .{};
                    try list.appendSlice(allocator, v.image_url.items);
                    break :blk list;
                },
                .detail = v.detail,
            } },
            .input_audio => |v| .{ .input_audio = .{ .audio_data = blk: {
                var list: std.ArrayListUnmanaged(u8) = .{};
                try list.appendSlice(allocator, v.audio_data.items);
                break :blk list;
            } } },
            .input_video => |v| .{ .input_video = .{ .video_url = blk: {
                var list: std.ArrayListUnmanaged(u8) = .{};
                try list.appendSlice(allocator, v.video_url.items);
                break :blk list;
            } } },
            .input_file => |v| .{ .input_file = .{
                .filename = if (v.filename) |fn_| try allocator.dupe(u8, fn_) else null,
                .file_data = if (v.file_data) |fd| blk: {
                    var list: std.ArrayListUnmanaged(u8) = .{};
                    try list.appendSlice(allocator, fd.items);
                    break :blk list;
                } else null,
                .file_url = if (v.file_url) |fu| blk: {
                    var list: std.ArrayListUnmanaged(u8) = .{};
                    try list.appendSlice(allocator, fu.items);
                    break :blk list;
                } else null,
            } },
            .refusal => |v| .{ .refusal = .{ .refusal = blk: {
                var list: std.ArrayListUnmanaged(u8) = .{};
                try list.appendSlice(allocator, v.refusal.items);
                break :blk list;
            } } },
            .reasoning_text => |v| .{ .reasoning_text = .{ .text = blk: {
                var list: std.ArrayListUnmanaged(u8) = .{};
                try list.appendSlice(allocator, v.text.items);
                break :blk list;
            } } },
            .summary_text => |v| .{ .summary_text = .{ .text = blk: {
                var list: std.ArrayListUnmanaged(u8) = .{};
                try list.appendSlice(allocator, v.text.items);
                break :blk list;
            } } },
            .unknown => |v| .{ .unknown = .{
                .raw_type = if (v.raw_type.len > 0) try allocator.dupe(u8, v.raw_type) else "",
                .raw_data = blk: {
                    var list: std.ArrayListUnmanaged(u8) = .{};
                    try list.appendSlice(allocator, v.raw_data.items);
                    break :blk list;
                },
            } },
        };

        return result;
    }

    /// Set filename (only valid for input_file).
    pub fn setFilename(self: *ContentPart, allocator: std.mem.Allocator, name: []const u8) !void {
        switch (self.variant) {
            .input_file => |*v| {
                if (v.filename) |old| allocator.free(old);
                v.filename = try allocator.dupe(u8, name);
            },
            else => return error.InvalidOperation,
        }
    }

    /// Set logprobs JSON (only valid for output_text).
    pub fn setLogprobs(self: *ContentPart, allocator: std.mem.Allocator, json: []const u8) !void {
        switch (self.variant) {
            .output_text => |*v| {
                if (v.logprobs_json) |old| allocator.free(old);
                v.logprobs_json = try allocator.dupe(u8, json);
            },
            else => return error.InvalidOperation,
        }
    }

    /// Set annotations JSON (only valid for output_text).
    pub fn setAnnotations(self: *ContentPart, allocator: std.mem.Allocator, json: []const u8) !void {
        switch (self.variant) {
            .output_text => |*v| {
                if (v.annotations_json) |old| allocator.free(old);
                v.annotations_json = try allocator.dupe(u8, json);
            },
            else => return error.InvalidOperation,
        }
    }

    // =========================================================================
    // Factory Functions - Create ContentPart for each type
    // =========================================================================

    /// Create an input_text content part.
    pub fn initInputText() ContentPart {
        return .{ .variant = .{ .input_text = .{} } };
    }

    /// Create an input_image content part.
    pub fn initInputImage(detail: ImageDetail) ContentPart {
        return .{ .variant = .{ .input_image = .{ .detail = detail } } };
    }

    /// Create an input_audio content part.
    pub fn initInputAudio() ContentPart {
        return .{ .variant = .{ .input_audio = .{} } };
    }

    /// Create an input_video content part.
    pub fn initInputVideo() ContentPart {
        return .{ .variant = .{ .input_video = .{} } };
    }

    /// Create an input_file content part.
    pub fn initInputFile() ContentPart {
        return .{ .variant = .{ .input_file = .{} } };
    }

    /// Create an output_text content part.
    pub fn initOutputText() ContentPart {
        return .{ .variant = .{ .output_text = .{} } };
    }

    /// Create a refusal content part.
    pub fn initRefusal() ContentPart {
        return .{ .variant = .{ .refusal = .{} } };
    }

    /// Create a text content part (generic text).
    pub fn initText() ContentPart {
        return .{ .variant = .{ .text = .{} } };
    }

    /// Create a reasoning_text content part.
    pub fn initReasoningText() ContentPart {
        return .{ .variant = .{ .reasoning_text = .{} } };
    }

    /// Create a summary_text content part.
    pub fn initSummaryText() ContentPart {
        return .{ .variant = .{ .summary_text = .{} } };
    }

    /// Create an unknown content part (for forward compatibility).
    /// Used when receiving unrecognized content types from providers.
    pub fn initUnknown() ContentPart {
        return .{ .variant = .{ .unknown = .{} } };
    }

    /// Create an unknown content part with a raw type string.
    /// Caller must own the raw_type memory or dupe it.
    pub fn initUnknownWithType(allocator: std.mem.Allocator, raw_type: []const u8) !ContentPart {
        return .{ .variant = .{ .unknown = .{
            .raw_type = try allocator.dupe(u8, raw_type),
        } } };
    }

    // REMOVED: initUrlCitation - url_citation is not a content type, it's an annotation
    // nested inside output_text. Use the UrlCitation helper struct instead and store
    // as JSON in output_text.annotations_json.
};

fn cloneContentParts(
    allocator: std.mem.Allocator,
    parts: []const ContentPart,
) !std.ArrayListUnmanaged(ContentPart) {
    var out: std.ArrayListUnmanaged(ContentPart) = .{};
    errdefer freeContentParts(allocator, &out);

    for (parts) |part| {
        const cloned = try part.clone(allocator);
        try out.append(allocator, cloned);
    }

    return out;
}

fn freeContentParts(
    allocator: std.mem.Allocator,
    parts: *std.ArrayListUnmanaged(ContentPart),
) void {
    for (parts.items) |*part| {
        part.deinit(allocator);
    }
    parts.deinit(allocator);
}

// =============================================================================
// Item Payload Variants
// =============================================================================

/// Payload for ItemType.message.
/// Schema: UserMessageItemParam, AssistantMessageItemParam, SystemMessageItemParam, DeveloperMessageItemParam
pub const MessageData = struct {
    role: MessageRole,
    status: ItemStatus = .completed,
    content: std.ArrayListUnmanaged(ContentPart),
    /// Raw role string for forward compatibility when role == .unknown.
    /// Enables round-tripping unknown roles without data loss.
    raw_role: ?[:0]const u8 = null,

    pub fn deinit(self: *MessageData, allocator: std.mem.Allocator) void {
        for (self.content.items) |*part| {
            part.deinit(allocator);
        }
        self.content.deinit(allocator);
        if (self.raw_role) |r| allocator.free(r);
    }

    /// Get the role string for serialization.
    /// Returns raw_role if role is unknown and raw_role is set,
    /// otherwise returns the enum's string representation.
    pub fn getRoleString(self: *const MessageData) []const u8 {
        if (self.role == .unknown) {
            if (self.raw_role) |r| return r;
        }
        return self.role.toString();
    }

    /// Get the number of content parts.
    pub fn partCount(self: *const MessageData) usize {
        return self.content.items.len;
    }

    /// Get a content part by index.
    pub fn getPart(self: *const MessageData, index: usize) ?*const ContentPart {
        if (index >= self.content.items.len) return null;
        return &self.content.items[index];
    }

    /// Get first text content (convenience for simple text messages).
    pub fn getFirstText(self: *const MessageData) []const u8 {
        for (self.content.items) |*part| {
            switch (part.variant) {
                .input_text, .output_text, .text => return part.getData(),
                else => {},
            }
        }
        return "";
    }

    /// Get all text content concatenated.
    /// Caller owns returned memory.
    pub fn getAllText(self: *const MessageData, allocator: std.mem.Allocator) ![]u8 {
        var result: std.ArrayListUnmanaged(u8) = .{};
        errdefer result.deinit(allocator);

        for (self.content.items) |*part| {
            switch (part.variant) {
                .input_text, .output_text, .text => {
                    try result.appendSlice(allocator, part.getData());
                },
                else => {},
            }
        }

        return result.toOwnedSlice(allocator);
    }

    pub fn clone(self: *const MessageData, allocator: std.mem.Allocator) !MessageData {
        var content = try cloneContentParts(allocator, self.content.items);
        errdefer freeContentParts(allocator, &content);

        return MessageData{
            .role = self.role,
            .status = self.status,
            .content = content,
            .raw_role = if (self.raw_role) |r| try allocator.dupeZ(u8, r) else null,
        };
    }
};

/// Payload for ItemType.function_call.
/// Schema: FunctionCallItemParam
pub const FunctionCallData = struct {
    /// The unique ID of the function call (generated by the model).
    call_id: [:0]const u8,
    /// The name of the function to call.
    name: [:0]const u8,
    /// The function arguments as a JSON string.
    arguments: std.ArrayListUnmanaged(u8),
    status: ItemStatus = .completed,

    pub fn deinit(self: *FunctionCallData, allocator: std.mem.Allocator) void {
        allocator.free(self.call_id);
        allocator.free(self.name);
        self.arguments.deinit(allocator);
    }

    /// Get arguments as slice.
    pub fn getArguments(self: *const FunctionCallData) []const u8 {
        return self.arguments.items;
    }

    pub fn clone(self: *const FunctionCallData, allocator: std.mem.Allocator) !FunctionCallData {
        var args: std.ArrayListUnmanaged(u8) = .{};
        errdefer args.deinit(allocator);

        try args.appendSlice(allocator, self.arguments.items);
        return FunctionCallData{
            .call_id = try allocator.dupeZ(u8, self.call_id),
            .name = try allocator.dupeZ(u8, self.name),
            .arguments = args,
            .status = self.status,
        };
    }
};

/// Output union for FunctionCallOutputData.
/// Schema: FunctionCallOutputItemParam.output is oneOf: string | array<content>
pub const FunctionOutputValue = union(enum) {
    /// Simple text output (most common case).
    text: std.ArrayListUnmanaged(u8),

    /// Array of content parts (multimodal output with images, files, etc.).
    parts: std.ArrayListUnmanaged(ContentPart),

    pub fn deinit(self: *FunctionOutputValue, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .text => |*t| t.deinit(allocator),
            .parts => |*p| {
                for (p.items) |*part| {
                    part.deinit(allocator);
                }
                p.deinit(allocator);
            },
        }
    }

    /// Get output as text.
    /// For .text variant, returns the text directly.
    /// For .parts variant, returns first text part's content.
    pub fn getText(self: *const FunctionOutputValue) []const u8 {
        switch (self.*) {
            .text => |t| return t.items,
            .parts => |p| {
                for (p.items) |*part| {
                    switch (part.variant) {
                        .input_text, .text, .output_text => return part.getData(),
                        else => {},
                    }
                }
                return "";
            },
        }
    }

    /// Check if this is a simple text output.
    pub fn isText(self: *const FunctionOutputValue) bool {
        return self.* == .text;
    }

    pub fn clone(self: *const FunctionOutputValue, allocator: std.mem.Allocator) !FunctionOutputValue {
        return switch (self.*) {
            .text => |t| blk: {
                var list: std.ArrayListUnmanaged(u8) = .{};
                errdefer list.deinit(allocator);
                try list.appendSlice(allocator, t.items);
                break :blk FunctionOutputValue{ .text = list };
            },
            .parts => |p| blk: {
                var parts = try cloneContentParts(allocator, p.items);
                errdefer freeContentParts(allocator, &parts);
                break :blk FunctionOutputValue{ .parts = parts };
            },
        };
    }
};

/// Payload for ItemType.function_call_output.
/// Schema: FunctionCallOutputItemParam
pub const FunctionCallOutputData = struct {
    /// The call_id this output is for (matches FunctionCallData.call_id).
    call_id: [:0]const u8,
    /// Output content - can be simple text or array of content parts.
    /// Schema: oneOf: string | array<InputTextContentParam | InputImageContentParam | etc.>
    output: FunctionOutputValue,
    status: ItemStatus = .completed,

    pub fn deinit(self: *FunctionCallOutputData, allocator: std.mem.Allocator) void {
        allocator.free(self.call_id);
        self.output.deinit(allocator);
    }

    /// Get output as text (convenience method).
    pub fn getOutputText(self: *const FunctionCallOutputData) []const u8 {
        return self.output.getText();
    }

    /// Check if output is simple text (vs multimodal parts).
    pub fn isTextOutput(self: *const FunctionCallOutputData) bool {
        return self.output.isText();
    }

    pub fn clone(self: *const FunctionCallOutputData, allocator: std.mem.Allocator) !FunctionCallOutputData {
        return FunctionCallOutputData{
            .call_id = try allocator.dupeZ(u8, self.call_id),
            .output = try self.output.clone(allocator),
            .status = self.status,
        };
    }
};

/// Payload for ItemType.reasoning.
/// Schema: ReasoningItemParam
pub const ReasoningData = struct {
    /// Reasoning content (uses ContentType.text or .reasoning_text).
    content: std.ArrayListUnmanaged(ContentPart),
    /// Summary content (uses ContentType.summary_text).
    summary: std.ArrayListUnmanaged(ContentPart),
    /// Encrypted reasoning content (for rehydration).
    encrypted_content: ?[]const u8 = null,
    /// Processing status of the reasoning item.
    status: ItemStatus = .completed,

    pub fn deinit(self: *ReasoningData, allocator: std.mem.Allocator) void {
        for (self.content.items) |*part| {
            part.deinit(allocator);
        }
        self.content.deinit(allocator);
        for (self.summary.items) |*part| {
            part.deinit(allocator);
        }
        self.summary.deinit(allocator);
        if (self.encrypted_content) |e| allocator.free(e);
    }

    /// Get summary text (concatenated).
    pub fn getSummaryText(self: *const ReasoningData) []const u8 {
        if (self.summary.items.len > 0) {
            return self.summary.items[0].getData();
        }
        return "";
    }

    pub fn clone(self: *const ReasoningData, allocator: std.mem.Allocator) !ReasoningData {
        var content = try cloneContentParts(allocator, self.content.items);
        errdefer freeContentParts(allocator, &content);

        var summary = try cloneContentParts(allocator, self.summary.items);
        errdefer freeContentParts(allocator, &summary);

        return ReasoningData{
            .content = content,
            .summary = summary,
            .encrypted_content = if (self.encrypted_content) |e| try allocator.dupe(u8, e) else null,
            .status = self.status,
        };
    }
};

/// Payload for ItemType.item_reference.
/// Schema: ItemReferenceParam
pub const ItemReferenceData = struct {
    /// The ID of the item to reference (string format like "msg_123").
    id: [:0]const u8,
    /// Processing status of the item reference.
    status: ItemStatus = .completed,

    pub fn deinit(self: *ItemReferenceData, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
    }

    pub fn clone(self: *const ItemReferenceData, allocator: std.mem.Allocator) !ItemReferenceData {
        return ItemReferenceData{
            .id = try allocator.dupeZ(u8, self.id),
            .status = self.status,
        };
    }
};

/// Payload for ItemType.unknown (forward compatibility).
pub const UnknownData = struct {
    /// The raw type string that was not recognized.
    raw_type: []const u8,
    /// Raw JSON payload for the unrecognized item.
    payload: []const u8,

    pub fn deinit(self: *UnknownData, allocator: std.mem.Allocator) void {
        allocator.free(self.raw_type);
        allocator.free(self.payload);
    }

    pub fn clone(self: *const UnknownData, allocator: std.mem.Allocator) !UnknownData {
        return UnknownData{
            .raw_type = try allocator.dupe(u8, self.raw_type),
            .payload = try allocator.dupe(u8, self.payload),
        };
    }
};

/// The polymorphic Item payload.
/// Tagged union matching ItemType discriminator.
pub const ItemVariant = union(ItemType) {
    message: MessageData,
    function_call: FunctionCallData,
    function_call_output: FunctionCallOutputData,
    reasoning: ReasoningData,
    item_reference: ItemReferenceData,
    unknown: UnknownData,

    pub fn deinit(self: *ItemVariant, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .message => |*m| m.deinit(allocator),
            .function_call => |*f| f.deinit(allocator),
            .function_call_output => |*f| f.deinit(allocator),
            .reasoning => |*r| r.deinit(allocator),
            .item_reference => |*i| i.deinit(allocator),
            .unknown => |*u| u.deinit(allocator),
        }
    }

    /// Get the item type discriminator.
    pub fn getType(self: *const ItemVariant) ItemType {
        return std.meta.activeTag(self.*);
    }

    /// Get the item status.
    pub fn getStatus(self: *const ItemVariant) ItemStatus {
        return switch (self.*) {
            .message => |m| m.status,
            .function_call => |f| f.status,
            .function_call_output => |f| f.status,
            .reasoning => |r| r.status,
            .item_reference => |r| r.status,
            .unknown => .completed,
        };
    }

    /// Set the item status. Unknown variants are immutable (always completed).
    pub fn setStatus(self: *ItemVariant, status: ItemStatus) void {
        switch (self.*) {
            .message => |*m| m.status = status,
            .function_call => |*f| f.status = status,
            .function_call_output => |*f| f.status = status,
            .reasoning => |*r| r.status = status,
            .item_reference => |*r| r.status = status,
            .unknown => {},
        }
    }

    pub fn clone(self: *const ItemVariant, allocator: std.mem.Allocator) !ItemVariant {
        return switch (self.*) {
            .message => |*m| .{ .message = try m.clone(allocator) },
            .function_call => |*f| .{ .function_call = try f.clone(allocator) },
            .function_call_output => |*f| .{ .function_call_output = try f.clone(allocator) },
            .reasoning => |*r| .{ .reasoning = try r.clone(allocator) },
            .item_reference => |*i| .{ .item_reference = try i.clone(allocator) },
            .unknown => |*u| .{ .unknown = try u.clone(allocator) },
        };
    }
};

// =============================================================================
// The Item Struct
// =============================================================================

/// An Item in the conversation history.
///
/// This is the atomic unit of the Open Responses architecture.
/// Each Item has:
///   - Session-scoped monotonic ID (O(1) lookups via array index)
///   - Creation timestamp
///   - Polymorphic payload (message, function_call, reasoning, etc.)
///   - Optional developer metadata
///
/// Thread safety: NOT thread-safe. All access must be from a single thread.
pub const Item = struct {
    /// Internal U64 ID for O(1) lookups; monotonic within the session.
    /// This avoids UUID map lookups and keeps items contiguous in memory and storage.
    /// For API responses, this is formatted as a prefixed string (e.g., "msg_123").
    id: u64,

    /// Creation timestamp (Unix milliseconds).
    /// Schema: "created_at"
    created_at_ms: i64,

    /// Expiration timestamp for retention (Unix ms). 0 = no expiry.
    ttl_ts: i64 = 0,

    /// Input token count (prompt tokens).
    input_tokens: u32 = 0,

    /// Output token count (completion tokens).
    output_tokens: u32 = 0,

    /// UI visibility flag.
    /// True means include in LLM context but hide from UI history.
    hidden: bool = false,

    /// Retention flag (true = prioritize in context window).
    pinned: bool = false,

    /// Structured output validation: JSON parsed successfully.
    json_valid: bool = false,

    /// Structured output validation: schema validation passed.
    schema_valid: bool = false,

    /// Structured output validation: output was repaired (e.g., auto-closed braces).
    repaired: bool = false,

    /// Optional parent item reference (edit/regenerate lineage).
    parent_item_id: ?u64 = null,

    /// Optional origin lineage for forked items.
    origin_session_id: ?[]const u8 = null,

    /// Origin item ID for forked items.
    origin_item_id: ?u64 = null,

    /// Finish reason for generation (e.g., "stop", "length").
    finish_reason: ?[:0]const u8 = null,

    /// Prefill time in nanoseconds.
    prefill_ns: u64 = 0,

    /// Generation time in nanoseconds.
    generation_ns: u64 = 0,

    /// The polymorphic payload.
    data: ItemVariant,

    /// Developer-defined metadata (JSON key-value pairs).
    /// Schema: "metadata" (MetadataParam)
    metadata: ?[]const u8 = null,

    /// Generation parameters used to produce this item (assistant messages only).
    /// JSON object containing model, temperature, top_p, top_k, etc.
    /// Null for non-assistant messages.
    generation_json: ?[]const u8 = null,

    /// Free all owned memory.
    pub fn deinit(self: *Item, allocator: std.mem.Allocator) void {
        self.data.deinit(allocator);
        if (self.metadata) |m| allocator.free(m);
        if (self.generation_json) |g| allocator.free(g);
        if (self.origin_session_id) |sid| allocator.free(sid);
    }

    /// Clone an item with origin lineage tracking for fork operations.
    ///
    /// # Lineage Tracking
    ///
    /// Origin fields track where an item came from when forking conversations:
    ///
    ///   - If this item already has origin (was previously forked), preserve it
    ///   - If source_session_id is provided, set origin to (source_session_id, self.id)
    ///   - If source_session_id is null (ephemeral chat), origin remains null
    ///
    /// # Ephemeral Chat Forking
    ///
    /// When forking from an ephemeral chat (no session_id), the cloned items will
    /// have null origin fields. This is intentional:
    ///
    ///   - Ephemeral chats have no persistent identity to reference
    ///   - Creating a fake origin would be semantically incorrect
    ///   - The fork becomes the "original" from the storage perspective
    ///
    /// If lineage tracking is important for your use case, provide a session_id
    /// when creating the source Chat, even if it won't be persisted.
    pub fn cloneWithOrigin(
        self: *const Item,
        allocator: std.mem.Allocator,
        source_session_id: ?[]const u8,
    ) !Item {
        var variant = try self.data.clone(allocator);
        errdefer variant.deinit(allocator);

        var metadata: ?[]const u8 = null;
        errdefer if (metadata) |m| allocator.free(m);
        if (self.metadata) |m| {
            metadata = try allocator.dupe(u8, m);
        }

        var generation_json: ?[]const u8 = null;
        errdefer if (generation_json) |g| allocator.free(g);
        if (self.generation_json) |g| {
            generation_json = try allocator.dupe(u8, g);
        }

        var origin_session_id: ?[]const u8 = null;
        errdefer if (origin_session_id) |sid| allocator.free(sid);
        var origin_item_id: ?u64 = null;

        // Lineage logic: preserve existing origin, or set from source if available.
        // Ephemeral sources (null session_id) produce items with null origin.
        if (self.origin_session_id != null and self.origin_item_id != null) {
            origin_session_id = try allocator.dupe(u8, self.origin_session_id.?);
            origin_item_id = self.origin_item_id;
        } else if (source_session_id) |sid| {
            origin_session_id = try allocator.dupe(u8, sid);
            origin_item_id = self.id;
        }

        return Item{
            .id = self.id,
            .created_at_ms = self.created_at_ms,
            .ttl_ts = self.ttl_ts,
            .input_tokens = self.input_tokens,
            .output_tokens = self.output_tokens,
            .hidden = self.hidden,
            .pinned = self.pinned,
            .json_valid = self.json_valid,
            .schema_valid = self.schema_valid,
            .repaired = self.repaired,
            .parent_item_id = self.parent_item_id,
            .origin_session_id = origin_session_id,
            .origin_item_id = origin_item_id,
            .finish_reason = self.finish_reason,
            .prefill_ns = self.prefill_ns,
            .generation_ns = self.generation_ns,
            .data = variant,
            .metadata = metadata,
            .generation_json = generation_json,
        };
    }

    /// Get the item type.
    pub fn getType(self: *const Item) ItemType {
        return self.data.getType();
    }

    /// Get as MessageData (returns null if not a message).
    pub fn asMessage(self: *const Item) ?*const MessageData {
        return switch (self.data) {
            .message => |*m| m,
            else => null,
        };
    }

    /// Get as mutable MessageData (returns null if not a message).
    pub fn asMessageMut(self: *Item) ?*MessageData {
        return switch (self.data) {
            .message => |*m| m,
            else => null,
        };
    }

    /// Get as FunctionCallData (returns null if not a function_call).
    pub fn asFunctionCall(self: *const Item) ?*const FunctionCallData {
        return switch (self.data) {
            .function_call => |*f| f,
            else => null,
        };
    }

    /// Get as FunctionCallOutputData (returns null if not a function_call_output).
    pub fn asFunctionCallOutput(self: *const Item) ?*const FunctionCallOutputData {
        return switch (self.data) {
            .function_call_output => |*f| f,
            else => null,
        };
    }

    /// Get as ReasoningData (returns null if not reasoning).
    pub fn asReasoning(self: *const Item) ?*const ReasoningData {
        return switch (self.data) {
            .reasoning => |*r| r,
            else => null,
        };
    }

    /// Get as ItemReferenceData (returns null if not an item_reference).
    pub fn asItemReference(self: *const Item) ?*const ItemReferenceData {
        return switch (self.data) {
            .item_reference => |*i| i,
            else => null,
        };
    }

    /// Format the ID as a prefixed string (e.g., "msg_123", "fc_456").
    /// Caller owns returned memory.
    pub fn formatId(self: *const Item, allocator: std.mem.Allocator) ![]u8 {
        const prefix = switch (self.data.getType()) {
            .message => "msg_",
            .function_call => "fc_",
            .function_call_output => "fco_",
            .reasoning => "rs_",
            .item_reference => "ref_",
            .unknown => "unk_",
        };
        return std.fmt.allocPrint(allocator, "{s}{d}", .{ prefix, self.id });
    }
};

// =============================================================================
// Tests
// =============================================================================

test "ItemStatus.toString" {
    try std.testing.expectEqualStrings("in_progress", ItemStatus.in_progress.toString());
    try std.testing.expectEqualStrings("completed", ItemStatus.completed.toString());
    try std.testing.expectEqualStrings("incomplete", ItemStatus.incomplete.toString());
}

test "ItemStatus.fromString" {
    try std.testing.expectEqual(ItemStatus.in_progress, ItemStatus.fromString("in_progress").?);
    try std.testing.expectEqual(ItemStatus.completed, ItemStatus.fromString("completed").?);
    try std.testing.expectEqual(ItemStatus.incomplete, ItemStatus.fromString("incomplete").?);
    try std.testing.expectEqual(@as(?ItemStatus, null), ItemStatus.fromString("invalid"));
}

test "ItemType.toString" {
    try std.testing.expectEqualStrings("message", ItemType.message.toString());
    try std.testing.expectEqualStrings("function_call", ItemType.function_call.toString());
    try std.testing.expectEqualStrings("function_call_output", ItemType.function_call_output.toString());
    try std.testing.expectEqualStrings("reasoning", ItemType.reasoning.toString());
    try std.testing.expectEqualStrings("item_reference", ItemType.item_reference.toString());
}

test "ItemType.fromString" {
    try std.testing.expectEqual(ItemType.message, ItemType.fromString("message"));
    try std.testing.expectEqual(ItemType.function_call, ItemType.fromString("function_call"));
    try std.testing.expectEqual(ItemType.function_call_output, ItemType.fromString("function_call_output"));
    try std.testing.expectEqual(ItemType.reasoning, ItemType.fromString("reasoning"));
    try std.testing.expectEqual(ItemType.item_reference, ItemType.fromString("item_reference"));
    try std.testing.expectEqual(ItemType.unknown, ItemType.fromString("unrecognized_type"));
}

test "MessageRole.toString" {
    try std.testing.expectEqualStrings("system", MessageRole.system.toString());
    try std.testing.expectEqualStrings("user", MessageRole.user.toString());
    try std.testing.expectEqualStrings("assistant", MessageRole.assistant.toString());
    try std.testing.expectEqualStrings("developer", MessageRole.developer.toString());
}

test "MessageRole.fromString" {
    try std.testing.expectEqual(MessageRole.system, MessageRole.fromString("system"));
    try std.testing.expectEqual(MessageRole.user, MessageRole.fromString("user"));
    try std.testing.expectEqual(MessageRole.assistant, MessageRole.fromString("assistant"));
    try std.testing.expectEqual(MessageRole.developer, MessageRole.fromString("developer"));
    // Unknown roles return .unknown for forward compatibility
    try std.testing.expectEqual(MessageRole.unknown, MessageRole.fromString("invalid"));
    try std.testing.expectEqual(MessageRole.unknown, MessageRole.fromString("critic"));
    try std.testing.expectEqual(MessageRole.unknown, MessageRole.fromString("tool"));
}

test "MessageRole.toLegacyRole" {
    try std.testing.expectEqual(MessageRole.system, MessageRole.system.toLegacyRole());
    try std.testing.expectEqual(MessageRole.user, MessageRole.user.toLegacyRole());
    try std.testing.expectEqual(MessageRole.assistant, MessageRole.assistant.toLegacyRole());
    try std.testing.expectEqual(MessageRole.system, MessageRole.developer.toLegacyRole());
    // Unknown roles map to user for legacy (safest default)
    try std.testing.expectEqual(MessageRole.user, MessageRole.unknown.toLegacyRole());
}

test "ContentType.toString" {
    try std.testing.expectEqualStrings("input_text", ContentType.input_text.toString());
    try std.testing.expectEqualStrings("input_image", ContentType.input_image.toString());
    try std.testing.expectEqualStrings("output_text", ContentType.output_text.toString());
    try std.testing.expectEqualStrings("refusal", ContentType.refusal.toString());
    try std.testing.expectEqualStrings("summary_text", ContentType.summary_text.toString());
}

test "ContentType.fromString" {
    try std.testing.expectEqual(ContentType.input_text, ContentType.fromString("input_text"));
    try std.testing.expectEqual(ContentType.input_image, ContentType.fromString("input_image"));
    try std.testing.expectEqual(ContentType.output_text, ContentType.fromString("output_text"));
    try std.testing.expectEqual(ContentType.refusal, ContentType.fromString("refusal"));
    // Unrecognized strings return .unknown for forward compatibility
    try std.testing.expectEqual(ContentType.unknown, ContentType.fromString("invalid"));
    try std.testing.expectEqual(ContentType.unknown, ContentType.fromString("future_3d_model"));
}

test "ContentType.isInput" {
    try std.testing.expect(ContentType.input_text.isInput());
    try std.testing.expect(ContentType.input_image.isInput());
    try std.testing.expect(ContentType.input_file.isInput());
    try std.testing.expect(!ContentType.output_text.isInput());
    try std.testing.expect(!ContentType.refusal.isInput());
}

test "ContentType.isOutput" {
    try std.testing.expect(ContentType.output_text.isOutput());
    try std.testing.expect(ContentType.refusal.isOutput());
    try std.testing.expect(!ContentType.input_text.isOutput());
    try std.testing.expect(!ContentType.input_image.isOutput());
}

test "ImageDetail round trip" {
    const details = [_]ImageDetail{ .auto, .low, .high };
    for (details) |detail| {
        const str = detail.toString();
        const parsed = ImageDetail.fromString(str).?;
        try std.testing.expectEqual(detail, parsed);
    }
}

test "ContentPart data operations" {
    const allocator = std.testing.allocator;
    var part = ContentPart.initInputText();
    defer part.deinit(allocator);

    try part.appendData(allocator, "Hello");
    try std.testing.expectEqualStrings("Hello", part.getData());

    try part.appendData(allocator, " World");
    try std.testing.expectEqualStrings("Hello World", part.getData());
}

test "ContentPart metadata operations" {
    const allocator = std.testing.allocator;
    var part = ContentPart.initOutputText();
    defer part.deinit(allocator);

    // Set logprobs
    try part.setLogprobs(allocator, "[{\"token\":\"hello\"}]");
    try std.testing.expectEqualStrings("[{\"token\":\"hello\"}]", part.variant.output_text.logprobs_json.?);

    // Replace logprobs
    try part.setLogprobs(allocator, "[{\"token\":\"world\"}]");
    try std.testing.expectEqualStrings("[{\"token\":\"world\"}]", part.variant.output_text.logprobs_json.?);

    // Set annotations
    try part.setAnnotations(allocator, "[{\"url\":\"https://example.com\"}]");
    try std.testing.expectEqualStrings("[{\"url\":\"https://example.com\"}]", part.variant.output_text.annotations_json.?);
}

test "ContentPart filename" {
    const allocator = std.testing.allocator;
    var part = ContentPart.initInputFile();
    defer part.deinit(allocator);

    try part.setFilename(allocator, "document.pdf");
    try std.testing.expectEqualStrings("document.pdf", part.variant.input_file.filename.?);

    try part.setFilename(allocator, "renamed.pdf");
    try std.testing.expectEqualStrings("renamed.pdf", part.variant.input_file.filename.?);
}

test "UrlCitation helper struct" {
    const allocator = std.testing.allocator;
    var citation = try UrlCitation.init(allocator, "https://example.com", "Example Title", 10, 50);
    defer citation.deinit(allocator);

    try std.testing.expectEqualStrings("https://example.com", citation.url);
    try std.testing.expectEqualStrings("Example Title", citation.title);
    try std.testing.expectEqual(@as(u32, 10), citation.start_index);
    try std.testing.expectEqual(@as(u32, 50), citation.end_index);
}

test "ContentVariant getText for different types" {
    const allocator = std.testing.allocator;

    // Test input_text
    var input_text_part = ContentPart.initInputText();
    defer input_text_part.deinit(allocator);
    try input_text_part.appendData(allocator, "Hello");
    try std.testing.expectEqualStrings("Hello", input_text_part.getData());

    // Test output_text
    var output_text_part = ContentPart.initOutputText();
    defer output_text_part.deinit(allocator);
    try output_text_part.appendData(allocator, "World");
    try std.testing.expectEqualStrings("World", output_text_part.getData());

    // Test refusal
    var refusal_part = ContentPart.initRefusal();
    defer refusal_part.deinit(allocator);
    try refusal_part.appendData(allocator, "I cannot help with that");
    try std.testing.expectEqualStrings("I cannot help with that", refusal_part.getData());
}

test "MessageData basic operations" {
    const allocator = std.testing.allocator;
    var msg = MessageData{
        .role = .user,
        .status = .completed,
        .content = .{},
    };
    defer msg.deinit(allocator);

    // Add content parts using factory function
    try msg.content.append(allocator, ContentPart.initInputText());
    try msg.content.items[0].appendData(allocator, "Hello");

    try std.testing.expectEqual(@as(usize, 1), msg.partCount());
    try std.testing.expectEqualStrings("Hello", msg.getFirstText());
}

test "MessageData getAllText" {
    const allocator = std.testing.allocator;
    var msg = MessageData{
        .role = .user,
        .status = .completed,
        .content = .{},
    };
    defer msg.deinit(allocator);

    // Add multiple text parts using factory functions
    try msg.content.append(allocator, ContentPart.initInputText());
    try msg.content.items[0].appendData(allocator, "Hello ");

    try msg.content.append(allocator, ContentPart.initInputText());
    try msg.content.items[1].appendData(allocator, "World");

    const all_text = try msg.getAllText(allocator);
    defer allocator.free(all_text);

    try std.testing.expectEqualStrings("Hello World", all_text);
}

test "FunctionCallData basic" {
    const allocator = std.testing.allocator;
    var fc = FunctionCallData{
        .call_id = try allocator.dupeZ(u8, "call_123"),
        .name = try allocator.dupeZ(u8, "get_weather"),
        .arguments = .{},
        .status = .completed,
    };
    defer fc.deinit(allocator);

    try fc.arguments.appendSlice(allocator, "{\"city\":\"NYC\"}");
    try std.testing.expectEqualStrings("{\"city\":\"NYC\"}", fc.getArguments());
}

test "FunctionCallOutputData basic with text" {
    const allocator = std.testing.allocator;
    var fco = FunctionCallOutputData{
        .call_id = try allocator.dupeZ(u8, "call_123"),
        .output = .{ .text = .{} },
        .status = .completed,
    };
    defer fco.deinit(allocator);

    try fco.output.text.appendSlice(allocator, "Sunny, 72F");

    try std.testing.expectEqualStrings("Sunny, 72F", fco.getOutputText());
    try std.testing.expect(fco.isTextOutput());
}

test "FunctionCallOutputData with parts" {
    const allocator = std.testing.allocator;
    var fco = FunctionCallOutputData{
        .call_id = try allocator.dupeZ(u8, "call_456"),
        .output = .{ .parts = .{} },
        .status = .completed,
    };
    defer fco.deinit(allocator);

    // Add a text part
    try fco.output.parts.append(allocator, ContentPart.initInputText());
    try fco.output.parts.items[0].appendData(allocator, "Weather data");

    try std.testing.expectEqualStrings("Weather data", fco.getOutputText());
    try std.testing.expect(!fco.isTextOutput());
}

test "ReasoningData basic" {
    const allocator = std.testing.allocator;
    var rd = ReasoningData{
        .content = .{},
        .summary = .{},
        .encrypted_content = null,
    };
    defer rd.deinit(allocator);

    // Add summary using factory function
    try rd.summary.append(allocator, ContentPart.initSummaryText());
    try rd.summary.items[0].appendData(allocator, "Reasoning summary");

    try std.testing.expectEqualStrings("Reasoning summary", rd.getSummaryText());
}

test "ItemVariant deinit" {
    const allocator = std.testing.allocator;

    // Test message variant
    var msg_variant = ItemVariant{
        .message = MessageData{
            .role = .user,
            .status = .completed,
            .content = .{},
        },
    };
    try msg_variant.message.content.append(allocator, ContentPart.initInputText());
    try msg_variant.message.content.items[0].appendData(allocator, "Test");
    msg_variant.deinit(allocator);

    // Test function_call variant
    var fc_variant = ItemVariant{
        .function_call = FunctionCallData{
            .call_id = try allocator.dupeZ(u8, "call_1"),
            .name = try allocator.dupeZ(u8, "func"),
            .arguments = .{},
            .status = .completed,
        },
    };
    fc_variant.deinit(allocator);
}

test "ItemVariant.setStatus transitions all variant types" {
    // Message variant
    var msg = ItemVariant{ .message = MessageData{ .role = .assistant, .status = .in_progress, .content = .{} } };
    try std.testing.expectEqual(ItemStatus.in_progress, msg.getStatus());
    msg.setStatus(.completed);
    try std.testing.expectEqual(ItemStatus.completed, msg.getStatus());
    msg.setStatus(.failed);
    try std.testing.expectEqual(ItemStatus.failed, msg.getStatus());

    // FunctionCall variant
    var fc = ItemVariant{ .function_call = FunctionCallData{ .call_id = &.{}, .name = &.{}, .arguments = .{}, .status = .in_progress } };
    try std.testing.expectEqual(ItemStatus.in_progress, fc.getStatus());
    fc.setStatus(.completed);
    try std.testing.expectEqual(ItemStatus.completed, fc.getStatus());

    // FunctionCallOutput variant
    var fco = ItemVariant{ .function_call_output = FunctionCallOutputData{ .call_id = &.{}, .output = .{ .text = .{} }, .status = .in_progress } };
    try std.testing.expectEqual(ItemStatus.in_progress, fco.getStatus());
    fco.setStatus(.incomplete);
    try std.testing.expectEqual(ItemStatus.incomplete, fco.getStatus());

    // Reasoning variant
    var r = ItemVariant{ .reasoning = ReasoningData{ .content = .{}, .summary = .{}, .status = .in_progress } };
    try std.testing.expectEqual(ItemStatus.in_progress, r.getStatus());
    r.setStatus(.completed);
    try std.testing.expectEqual(ItemStatus.completed, r.getStatus());

    // ItemReference variant
    var ref = ItemVariant{ .item_reference = ItemReferenceData{ .id = &.{}, .status = .in_progress } };
    try std.testing.expectEqual(ItemStatus.in_progress, ref.getStatus());
    ref.setStatus(.waiting);
    try std.testing.expectEqual(ItemStatus.waiting, ref.getStatus());

    // Unknown variant is immutable (always completed)
    var unk = ItemVariant{ .unknown = UnknownData{ .raw_type = &.{}, .payload = &.{} } };
    try std.testing.expectEqual(ItemStatus.completed, unk.getStatus());
    unk.setStatus(.failed); // no-op
    try std.testing.expectEqual(ItemStatus.completed, unk.getStatus());
}

test "Item formatId" {
    const allocator = std.testing.allocator;

    var item = Item{
        .id = 123,
        .created_at_ms = 1000,
        .data = ItemVariant{
            .message = MessageData{
                .role = .user,
                .status = .completed,
                .content = .{},
            },
        },
        .metadata = null,
    };
    defer item.deinit(allocator);

    const id_str = try item.formatId(allocator);
    defer allocator.free(id_str);

    try std.testing.expectEqualStrings("msg_123", id_str);
}

test "Item type accessors" {
    const allocator = std.testing.allocator;

    var item = Item{
        .id = 1,
        .created_at_ms = 1000,
        .data = ItemVariant{
            .message = MessageData{
                .role = .assistant,
                .status = .completed,
                .content = .{},
            },
        },
        .metadata = null,
    };
    defer item.deinit(allocator);

    try std.testing.expectEqual(ItemType.message, item.getType());
    try std.testing.expect(item.asMessage() != null);
    try std.testing.expect(item.asFunctionCall() == null);
    try std.testing.expect(item.asFunctionCallOutput() == null);
    try std.testing.expect(item.asReasoning() == null);
    try std.testing.expect(item.asItemReference() == null);
}

test "Item with metadata" {
    const allocator = std.testing.allocator;

    var item = Item{
        .id = 1,
        .created_at_ms = 1000,
        .data = ItemVariant{
            .message = MessageData{
                .role = .user,
                .status = .completed,
                .content = .{},
            },
        },
        .metadata = try allocator.dupe(u8, "{\"key\":\"value\"}"),
    };
    defer item.deinit(allocator);

    try std.testing.expectEqualStrings("{\"key\":\"value\"}", item.metadata.?);
}
