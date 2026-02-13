//! Model Bundle - unified representation of model resources
//!
//! A Bundle represents a pre-validated, ready-to-load model with all its
//! required files (config, weights, tokenizer) resolved to concrete sources.
//! This abstracts away the differences between local paths and HF cache format.

const std = @import("std");

/// A validated bundle of model resources ready for loading.
///
/// The Bundle owns all allocated paths and must be deinitialized when done.
/// Use `resolve()` or `fetch()` from the storage module to create a Bundle.
pub const Bundle = struct {
    allocator: std.mem.Allocator,

    /// Base directory containing the model files
    dir: []const u8,

    /// Model configuration source
    config: ConfigSource,

    /// Model weights source
    weights: WeightsSource,

    /// Tokenizer source
    tokenizer: TokenizerSource,

    /// Detected model format
    format: Format,

    /// Model format type
    pub const Format = enum {
        /// MLX-style SafeTensors (4-bit/8-bit quantized)
        mlx,
        /// Standard SafeTensors
        safetensors,
    };

    /// Configuration source - either a file path or inline JSON
    pub const ConfigSource = union(enum) {
        /// Path to config.json
        path: []const u8,
        /// Inline JSON string
        json: []const u8,
    };

    /// Weights source - single file, sharded, or absent
    pub const WeightsSource = union(enum) {
        /// Single weights file (model.safetensors)
        single: []const u8,
        /// Sharded weights (model.safetensors.index.json + shards)
        sharded: ShardedInfo,
        /// No weights available (tokenizer-only resolution)
        none,
    };

    /// Information for sharded weight files
    pub const ShardedInfo = struct {
        /// Path to the index file (model.safetensors.index.json)
        index_path: []const u8,
        /// Directory containing shard files
        shard_dir: []const u8,
    };

    /// Tokenizer source - file path, inline JSON, or none
    pub const TokenizerSource = union(enum) {
        /// Path to tokenizer.json
        path: []const u8,
        /// Inline JSON string
        json: []const u8,
        /// No tokenizer available
        none,
    };

    /// Get config path (for ModelLocation consumers)
    pub fn config_path(self: Bundle) []const u8 {
        return switch (self.config) {
            .path => |p| p,
            .json => |j| j,
        };
    }

    /// Get weights path (for ModelLocation consumers).
    /// Returns null when weights are absent (tokenizer-only resolution).
    pub fn weights_path(self: Bundle) ?[]const u8 {
        return switch (self.weights) {
            .single => |p| p,
            .sharded => |s| s.index_path,
            .none => null,
        };
    }

    /// Get tokenizer path (for ModelLocation consumers)
    /// Returns empty string if no path-based tokenizer.
    pub fn tokenizer_path(self: Bundle) []const u8 {
        return switch (self.tokenizer) {
            .path => |p| p,
            .json, .none => "",
        };
    }

    /// Get tokenizer JSON if inline
    pub fn tokenizer_json(self: Bundle) ?[]const u8 {
        return switch (self.tokenizer) {
            .json => |j| j,
            .path, .none => null,
        };
    }

    /// Check if weights are sharded
    pub fn isSharded(self: Bundle) bool {
        return switch (self.weights) {
            .sharded => true,
            .single, .none => false,
        };
    }

    /// Free all allocated resources
    pub fn deinit(self: *Bundle) void {
        // Free directory
        if (self.dir.len > 0) {
            self.allocator.free(self.dir);
        }

        // Free config source
        switch (self.config) {
            .path => |p| self.allocator.free(p),
            .json => |j| self.allocator.free(j),
        }

        // Free weights source
        switch (self.weights) {
            .single => |p| self.allocator.free(p),
            .sharded => |s| {
                self.allocator.free(s.index_path);
                self.allocator.free(s.shard_dir);
            },
            .none => {},
        }

        // Free tokenizer source
        switch (self.tokenizer) {
            .path => |p| self.allocator.free(p),
            .json => |j| self.allocator.free(j),
            .none => {},
        }

        self.* = undefined;
    }
};

// =============================================================================
// Tests
// =============================================================================

test "Bundle accessors" {
    const allocator = std.testing.allocator;

    var model_bundle = Bundle{
        .allocator = allocator,
        .dir = try allocator.dupe(u8, "/models/test"),
        .config = .{ .path = try allocator.dupe(u8, "/models/test/config.json") },
        .weights = .{ .single = try allocator.dupe(u8, "/models/test/model.safetensors") },
        .tokenizer = .{ .path = try allocator.dupe(u8, "/models/test/tokenizer.json") },
        .format = .safetensors,
    };
    defer model_bundle.deinit();

    try std.testing.expectEqualStrings("/models/test/config.json", model_bundle.config_path());
    try std.testing.expectEqualStrings("/models/test/model.safetensors", model_bundle.weights_path().?);
    try std.testing.expectEqualStrings("/models/test/tokenizer.json", model_bundle.tokenizer_path());
    try std.testing.expect(!model_bundle.isSharded());
}

test "Bundle with sharded weights" {
    const allocator = std.testing.allocator;

    var model_bundle = Bundle{
        .allocator = allocator,
        .dir = try allocator.dupe(u8, "/models/sharded"),
        .config = .{ .path = try allocator.dupe(u8, "/models/sharded/config.json") },
        .weights = .{ .sharded = .{
            .index_path = try allocator.dupe(u8, "/models/sharded/model.safetensors.index.json"),
            .shard_dir = try allocator.dupe(u8, "/models/sharded"),
        } },
        .tokenizer = .{ .none = {} },
        .format = .safetensors,
    };
    defer model_bundle.deinit();

    try std.testing.expect(model_bundle.isSharded());
    try std.testing.expectEqualStrings("", model_bundle.tokenizer_path());
}

test "Bundle config_path returns correct path for path-based config" {
    const allocator = std.testing.allocator;

    var model_bundle = Bundle{
        .allocator = allocator,
        .dir = try allocator.dupe(u8, "/test"),
        .config = .{ .path = try allocator.dupe(u8, "/test/config.json") },
        .weights = .{ .single = try allocator.dupe(u8, "/test/model.safetensors") },
        .tokenizer = .{ .none = {} },
        .format = .safetensors,
    };
    defer model_bundle.deinit();

    try std.testing.expectEqualStrings("/test/config.json", model_bundle.config_path());
}

test "Bundle config_path returns JSON string for inline config" {
    const allocator = std.testing.allocator;

    const json_str = "{\"hidden_size\": 768}";
    var model_bundle = Bundle{
        .allocator = allocator,
        .dir = try allocator.dupe(u8, "/test"),
        .config = .{ .json = try allocator.dupe(u8, json_str) },
        .weights = .{ .single = try allocator.dupe(u8, "/test/model.safetensors") },
        .tokenizer = .{ .none = {} },
        .format = .safetensors,
    };
    defer model_bundle.deinit();

    try std.testing.expectEqualStrings(json_str, model_bundle.config_path());
}

test "Bundle weights_path returns single file path" {
    const allocator = std.testing.allocator;

    var model_bundle = Bundle{
        .allocator = allocator,
        .dir = try allocator.dupe(u8, "/test"),
        .config = .{ .path = try allocator.dupe(u8, "/test/config.json") },
        .weights = .{ .single = try allocator.dupe(u8, "/test/model.safetensors") },
        .tokenizer = .{ .none = {} },
        .format = .safetensors,
    };
    defer model_bundle.deinit();

    try std.testing.expectEqualStrings("/test/model.safetensors", model_bundle.weights_path().?);
}

test "Bundle weights_path returns index path for sharded weights" {
    const allocator = std.testing.allocator;

    var model_bundle = Bundle{
        .allocator = allocator,
        .dir = try allocator.dupe(u8, "/test"),
        .config = .{ .path = try allocator.dupe(u8, "/test/config.json") },
        .weights = .{ .sharded = .{
            .index_path = try allocator.dupe(u8, "/test/model.safetensors.index.json"),
            .shard_dir = try allocator.dupe(u8, "/test"),
        } },
        .tokenizer = .{ .none = {} },
        .format = .safetensors,
    };
    defer model_bundle.deinit();

    try std.testing.expectEqualStrings("/test/model.safetensors.index.json", model_bundle.weights_path().?);
}

test "Bundle tokenizer_path returns path when tokenizer is path-based" {
    const allocator = std.testing.allocator;

    var model_bundle = Bundle{
        .allocator = allocator,
        .dir = try allocator.dupe(u8, "/test"),
        .config = .{ .path = try allocator.dupe(u8, "/test/config.json") },
        .weights = .{ .single = try allocator.dupe(u8, "/test/model.safetensors") },
        .tokenizer = .{ .path = try allocator.dupe(u8, "/test/tokenizer.json") },
        .format = .safetensors,
    };
    defer model_bundle.deinit();

    try std.testing.expectEqualStrings("/test/tokenizer.json", model_bundle.tokenizer_path());
}

test "Bundle tokenizer_path returns empty string for inline tokenizer" {
    const allocator = std.testing.allocator;

    var model_bundle = Bundle{
        .allocator = allocator,
        .dir = try allocator.dupe(u8, "/test"),
        .config = .{ .path = try allocator.dupe(u8, "/test/config.json") },
        .weights = .{ .single = try allocator.dupe(u8, "/test/model.safetensors") },
        .tokenizer = .{ .json = try allocator.dupe(u8, "{\"vocab\": {}}") },
        .format = .safetensors,
    };
    defer model_bundle.deinit();

    try std.testing.expectEqualStrings("", model_bundle.tokenizer_path());
}

test "Bundle tokenizer_path returns empty string when tokenizer is none" {
    const allocator = std.testing.allocator;

    var model_bundle = Bundle{
        .allocator = allocator,
        .dir = try allocator.dupe(u8, "/test"),
        .config = .{ .path = try allocator.dupe(u8, "/test/config.json") },
        .weights = .{ .single = try allocator.dupe(u8, "/test/model.safetensors") },
        .tokenizer = .{ .none = {} },
        .format = .safetensors,
    };
    defer model_bundle.deinit();

    try std.testing.expectEqualStrings("", model_bundle.tokenizer_path());
}

test "Bundle tokenizer_json returns JSON for inline tokenizer" {
    const allocator = std.testing.allocator;

    const json_str = "{\"vocab\": {\"test\": 0}}";
    var model_bundle = Bundle{
        .allocator = allocator,
        .dir = try allocator.dupe(u8, "/test"),
        .config = .{ .path = try allocator.dupe(u8, "/test/config.json") },
        .weights = .{ .single = try allocator.dupe(u8, "/test/model.safetensors") },
        .tokenizer = .{ .json = try allocator.dupe(u8, json_str) },
        .format = .safetensors,
    };
    defer model_bundle.deinit();

    const result = model_bundle.tokenizer_json();
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings(json_str, result.?);
}

test "Bundle tokenizer_json returns null for path-based tokenizer" {
    const allocator = std.testing.allocator;

    var model_bundle = Bundle{
        .allocator = allocator,
        .dir = try allocator.dupe(u8, "/test"),
        .config = .{ .path = try allocator.dupe(u8, "/test/config.json") },
        .weights = .{ .single = try allocator.dupe(u8, "/test/model.safetensors") },
        .tokenizer = .{ .path = try allocator.dupe(u8, "/test/tokenizer.json") },
        .format = .safetensors,
    };
    defer model_bundle.deinit();

    try std.testing.expect(model_bundle.tokenizer_json() == null);
}

test "Bundle tokenizer_json returns null when tokenizer is none" {
    const allocator = std.testing.allocator;

    var model_bundle = Bundle{
        .allocator = allocator,
        .dir = try allocator.dupe(u8, "/test"),
        .config = .{ .path = try allocator.dupe(u8, "/test/config.json") },
        .weights = .{ .single = try allocator.dupe(u8, "/test/model.safetensors") },
        .tokenizer = .{ .none = {} },
        .format = .safetensors,
    };
    defer model_bundle.deinit();

    try std.testing.expect(model_bundle.tokenizer_json() == null);
}

test "Bundle isSharded returns false for single weights file" {
    const allocator = std.testing.allocator;

    var model_bundle = Bundle{
        .allocator = allocator,
        .dir = try allocator.dupe(u8, "/test"),
        .config = .{ .path = try allocator.dupe(u8, "/test/config.json") },
        .weights = .{ .single = try allocator.dupe(u8, "/test/model.safetensors") },
        .tokenizer = .{ .none = {} },
        .format = .safetensors,
    };
    defer model_bundle.deinit();

    try std.testing.expect(!model_bundle.isSharded());
}

test "Bundle isSharded returns true for sharded weights" {
    const allocator = std.testing.allocator;

    var model_bundle = Bundle{
        .allocator = allocator,
        .dir = try allocator.dupe(u8, "/test"),
        .config = .{ .path = try allocator.dupe(u8, "/test/config.json") },
        .weights = .{ .sharded = .{
            .index_path = try allocator.dupe(u8, "/test/model.safetensors.index.json"),
            .shard_dir = try allocator.dupe(u8, "/test"),
        } },
        .tokenizer = .{ .none = {} },
        .format = .safetensors,
    };
    defer model_bundle.deinit();

    try std.testing.expect(model_bundle.isSharded());
}

test "Bundle deinit properly frees all resources" {
    const allocator = std.testing.allocator;

    // Test with all possible resource types
    var model_bundle = Bundle{
        .allocator = allocator,
        .dir = try allocator.dupe(u8, "/test"),
        .config = .{ .path = try allocator.dupe(u8, "/test/config.json") },
        .weights = .{ .sharded = .{
            .index_path = try allocator.dupe(u8, "/test/model.safetensors.index.json"),
            .shard_dir = try allocator.dupe(u8, "/test/shards"),
        } },
        .tokenizer = .{ .path = try allocator.dupe(u8, "/test/tokenizer.json") },
        .format = .mlx,
    };

    // deinit should free all allocations without leaking
    model_bundle.deinit();
}

test "Bundle with MLX format" {
    const allocator = std.testing.allocator;

    var model_bundle = Bundle{
        .allocator = allocator,
        .dir = try allocator.dupe(u8, "/models/my-model-MLX-4bit"),
        .config = .{ .path = try allocator.dupe(u8, "/models/my-model-MLX-4bit/config.json") },
        .weights = .{ .single = try allocator.dupe(u8, "/models/my-model-MLX-4bit/model.safetensors") },
        .tokenizer = .{ .path = try allocator.dupe(u8, "/models/my-model-MLX-4bit/tokenizer.json") },
        .format = .mlx,
    };
    defer model_bundle.deinit();

    try std.testing.expectEqual(Bundle.Format.mlx, model_bundle.format);
}
