//! Conversion Scheme Definitions
//!
//! Unified quantization scheme that encodes all conversion settings.
//! This eliminates invalid parameter combinations and provides a
//! simple, self-documenting API.

const std = @import("std");
const grouped_affine = @import("grouped_affine.zig");
const progress_api = @import("../capi/progress.zig");

// =============================================================================
// Scheme Definitions (Single Source of Truth)
// =============================================================================

/// Definition of a scheme and its aliases.
const SchemeDefinition = struct {
    val: Scheme,
    name: []const u8,
    aliases: []const []const u8,
};

/// Single Source of Truth for Scheme configuration.
/// Each scheme has a canonical name and optional user-friendly aliases.
const DEFS = [_]SchemeDefinition{
    // No quantization
    .{ .val = .f16, .name = "f16", .aliases = &.{ "16bit", "half" } },

    // Grouped Affine (MLX Compatible) - Map generic aliases to group_size=64
    .{ .val = .gaf4_32, .name = "gaf4_32", .aliases = &.{} },
    .{ .val = .gaf4_64, .name = "gaf4_64", .aliases = &.{ "mlx", "mlx4", "gaf4", "4bit", "q4", "int4" } },
    .{ .val = .gaf4_128, .name = "gaf4_128", .aliases = &.{} },
    .{ .val = .gaf8_32, .name = "gaf8_32", .aliases = &.{} },
    .{ .val = .gaf8_64, .name = "gaf8_64", .aliases = &.{ "mlx8", "gaf8", "8bit", "q8", "int8" } },
    .{ .val = .gaf8_128, .name = "gaf8_128", .aliases = &.{} },

    // Hardware Float
    .{ .val = .fp8_e4m3, .name = "fp8_e4m3", .aliases = &.{"fp8"} },
    .{ .val = .fp8_e5m2, .name = "fp8_e5m2", .aliases = &.{} },
    .{ .val = .mxfp4, .name = "mxfp4", .aliases = &.{} },
    .{ .val = .nvfp4, .name = "nvfp4", .aliases = &.{} },
};

// =============================================================================
// Platform and Quantization Level Enums
// =============================================================================

/// Target platform for optimized conversion.
/// Different platforms benefit from different quantization formats.
pub const Platform = enum(u32) {
    cpu = 0, // Generic CPU: use Grouped Affine (gaf4_64, gaf8_64, f16)
    metal = 1, // Apple Silicon: use Grouped Affine (gaf4_64, gaf8_64, f16)
    cuda = 2, // NVIDIA GPU: use Grouped Affine (gaf4_64, gaf8_64, f16)

    /// Parse platform from string (case-insensitive).
    pub fn fromString(s: []const u8) ?Platform {
        if (std.ascii.eqlIgnoreCase(s, "cpu")) return .cpu;
        if (std.ascii.eqlIgnoreCase(s, "metal") or std.ascii.eqlIgnoreCase(s, "mps") or std.ascii.eqlIgnoreCase(s, "apple")) return .metal;
        if (std.ascii.eqlIgnoreCase(s, "cuda") or std.ascii.eqlIgnoreCase(s, "gpu") or std.ascii.eqlIgnoreCase(s, "nvidia")) return .cuda;
        return null;
    }

    /// Get canonical string representation.
    pub fn toString(self: Platform) []const u8 {
        return switch (self) {
            .cpu => "cpu",
            .metal => "metal",
            .cuda => "cuda",
        };
    }
};

/// Quantization level (bit precision).
pub const QuantLevel = enum(u32) {
    q4 = 0, // 4-bit quantization (smallest, faster)
    q8 = 1, // 8-bit quantization (balanced)
    q16 = 2, // 16-bit (no quantization, highest quality)

    /// Parse quant level from string (case-insensitive).
    pub fn fromString(s: []const u8) ?QuantLevel {
        if (std.ascii.eqlIgnoreCase(s, "4bit") or std.ascii.eqlIgnoreCase(s, "q4") or std.ascii.eqlIgnoreCase(s, "int4")) return .q4;
        if (std.ascii.eqlIgnoreCase(s, "8bit") or std.ascii.eqlIgnoreCase(s, "q8") or std.ascii.eqlIgnoreCase(s, "int8")) return .q8;
        if (std.ascii.eqlIgnoreCase(s, "16bit") or std.ascii.eqlIgnoreCase(s, "q16") or std.ascii.eqlIgnoreCase(s, "f16") or std.ascii.eqlIgnoreCase(s, "half")) return .q16;
        return null;
    }

    /// Get canonical string representation.
    pub fn toString(self: QuantLevel) []const u8 {
        return switch (self) {
            .q4 => "4bit",
            .q8 => "8bit",
            .q16 => "16bit",
        };
    }
};

// =============================================================================
// Scheme Enum
// =============================================================================

/// Unified quantization scheme.
///
/// Each scheme encodes all necessary conversion parameters (method, bits, group_size).
/// This eliminates invalid parameter combinations.
///
/// ## No Quantization
/// - f16
///
/// ## Grouped Affine (MLX Compatible)
/// - gaf4_32, gaf4_64, gaf4_128, gaf8_32, gaf8_64, gaf8_128
///
/// ## Hardware Float (Not Yet Implemented)
/// - fp8_e4m3, fp8_e5m2, mxfp4, nvfp4
pub const Scheme = enum(u32) {
    // No quantization
    f16 = 4, // No quantization

    // Grouped Affine (MLX compatible) - values 10-15
    gaf4_32 = 10, // 4-bit, group_size=32 (highest accuracy)
    gaf4_64 = 11, // 4-bit, group_size=64 (balanced, DEFAULT)
    gaf4_128 = 12, // 4-bit, group_size=128 (smallest)
    gaf8_32 = 13, // 8-bit, group_size=32
    gaf8_64 = 14, // 8-bit, group_size=64
    gaf8_128 = 15, // 8-bit, group_size=128

    // Hardware float (not yet implemented) - values 20-23
    fp8_e4m3 = 20, // FP8 E4M3 for inference (H100/vLLM)
    fp8_e5m2 = 21, // FP8 E5M2 for training
    mxfp4 = 22, // OCP Microscaling 4-bit (group_size=32 fixed)
    nvfp4 = 23, // NVIDIA Blackwell 4-bit (group_size=32 fixed)

    /// Get the conversion method for this scheme.
    pub fn getMethod(self: Scheme) Method {
        return switch (self) {
            .f16, .gaf4_32, .gaf4_64, .gaf4_128, .gaf8_32, .gaf8_64, .gaf8_128 => .grouped_affine,
            .fp8_e4m3, .fp8_e5m2 => .fp8,
            .mxfp4 => .mxfp4,
            .nvfp4 => .nvfp4,
        };
    }

    /// Get the bit width for this scheme.
    pub fn getBits(self: Scheme) u8 {
        return switch (self) {
            .gaf4_32, .gaf4_64, .gaf4_128, .mxfp4, .nvfp4 => 4,
            .gaf8_32, .gaf8_64, .gaf8_128, .fp8_e4m3, .fp8_e5m2 => 8,
            .f16 => 16,
        };
    }

    /// Get estimated bits per parameter including overhead.
    /// GAF has scale/bias per group.
    /// Returns bits * 100 to avoid floating point (e.g., 450 = 4.5 bits).
    pub fn getEffectiveBitsX100(self: Scheme) u32 {
        return switch (self) {
            .f16 => 1600, // 16 bits exactly

            // GAF: bits + scale/bias overhead (16-bit scale + 16-bit bias per group)
            // Overhead = 32 bits / group_size per param
            .gaf4_32 => 500, // 4 + 1.0 (32 bits / 32 group)
            .gaf4_64 => 450, // 4 + 0.5 (32 bits / 64 group)
            .gaf4_128 => 425, // 4 + 0.25 (32 bits / 128 group)
            .gaf8_32 => 900, // 8 + 1.0
            .gaf8_64 => 850, // 8 + 0.5
            .gaf8_128 => 825, // 8 + 0.25

            // FP8: 8 bits exactly
            .fp8_e4m3, .fp8_e5m2 => 800,

            // Microscaling: 4 bits + scale overhead
            .mxfp4 => 500, // 4 + 1.0 (group_size=32)
            .nvfp4 => 500, // 4 + 1.0 (group_size=32)
        };
    }

    /// Get the group size for this scheme (0 if not applicable).
    pub fn getGroupSize(self: Scheme) u32 {
        return switch (self) {
            .gaf4_32, .gaf8_32, .mxfp4, .nvfp4 => 32,
            .gaf4_64, .gaf8_64, .f16 => 64, // f16 uses default group_size
            .gaf4_128, .gaf8_128 => 128,
            .fp8_e4m3, .fp8_e5m2 => 0,
        };
    }

    /// Get the output name suffix for this scheme.
    pub fn toOutputSuffix(self: Scheme) []const u8 {
        return switch (self) {
            .f16 => "F16",
            .gaf4_32 => "GAF4-G32",
            .gaf4_64 => "GAF4",
            .gaf4_128 => "GAF4-G128",
            .gaf8_32 => "GAF8-G32",
            .gaf8_64 => "GAF8",
            .gaf8_128 => "GAF8-G128",
            .fp8_e4m3 => "FP8",
            .fp8_e5m2 => "FP8-E5M2",
            .mxfp4 => "MXFP4",
            .nvfp4 => "NVFP4",
        };
    }

    /// Parse scheme from string (case-insensitive).
    /// Checks both canonical names and aliases.
    pub fn fromString(s: []const u8) ?Scheme {
        inline for (DEFS) |def| {
            if (std.ascii.eqlIgnoreCase(s, def.name)) return def.val;
            inline for (def.aliases) |alias| {
                if (std.ascii.eqlIgnoreCase(s, alias)) return def.val;
            }
        }
        return null;
    }

    /// Get canonical string representation.
    pub fn toString(self: Scheme) []const u8 {
        inline for (DEFS) |def| {
            if (self == def.val) return def.name;
        }
        unreachable;
    }

    /// Resolve the optimal scheme for a given platform and quantization level.
    ///
    /// This is the single source of truth for platform/quant -> scheme mapping.
    /// All platforms use Grouped Affine (GAF) quantization.
    ///
    /// | Platform | Quant | Resolved Scheme |
    /// | -------- | ----- | --------------- |
    /// | cpu      | q4    | gaf4_64         |
    /// | cpu      | q8    | gaf8_64         |
    /// | cpu      | q16   | f16             |
    /// | metal    | q4    | gaf4_64         |
    /// | metal    | q8    | gaf8_64         |
    /// | metal    | q16   | f16             |
    /// | cuda     | q4    | gaf4_64         |
    /// | cuda     | q8    | gaf8_64         |
    /// | cuda     | q16   | f16             |
    pub fn resolve(platform: Platform, quant: QuantLevel) Scheme {
        _ = platform; // All platforms use the same schemes now
        return switch (quant) {
            .q4 => .gaf4_64,
            .q8 => .gaf8_64,
            .q16 => .f16,
        };
    }

    /// Generate JSON string of schemes and aliases (comptime).
    fn generateJson() *const [jsonLen()]u8 {
        @setEvalBranchQuota(20000);
        comptime var buf: [4096]u8 = undefined;
        comptime var pos: usize = 0;

        inline for ("{") |c| {
            buf[pos] = c;
            pos += 1;
        }

        inline for (DEFS, 0..) |def, i| {
            if (i > 0) {
                buf[pos] = ',';
                pos += 1;
            }
            buf[pos] = '"';
            pos += 1;
            inline for (def.name) |c| {
                buf[pos] = c;
                pos += 1;
            }
            inline for ("\":[") |c| {
                buf[pos] = c;
                pos += 1;
            }

            inline for (def.aliases, 0..) |alias, j| {
                if (j > 0) {
                    buf[pos] = ',';
                    pos += 1;
                }
                buf[pos] = '"';
                pos += 1;
                inline for (alias) |c| {
                    buf[pos] = c;
                    pos += 1;
                }
                buf[pos] = '"';
                pos += 1;
            }
            buf[pos] = ']';
            pos += 1;
        }
        buf[pos] = '}';
        pos += 1;
        const final = buf[0..pos].*;
        return &final;
    }

    fn jsonLen() usize {
        @setEvalBranchQuota(20000);
        comptime var len: usize = 1; // opening {
        inline for (DEFS, 0..) |def, i| {
            if (i > 0) len += 1; // comma
            len += 1; // opening quote
            len += def.name.len;
            len += 3; // ":[
            inline for (def.aliases, 0..) |alias, j| {
                if (j > 0) len += 1; // comma
                len += 1; // opening quote
                len += alias.len;
                len += 1; // closing quote
            }
            len += 1; // ]
        }
        len += 1; // closing }
        return len;
    }

    /// JSON string of all schemes and aliases for API discovery.
    pub const all_schemes_json: []const u8 = generateJson();

    /// Generate CSV string of scheme names (comptime, legacy).
    fn generateCsv() *const [csvLen()]u8 {
        @setEvalBranchQuota(10000);
        comptime var buf: [2048]u8 = undefined;
        comptime var pos: usize = 0;
        inline for (DEFS, 0..) |def, i| {
            if (i > 0) {
                buf[pos] = ',';
                pos += 1;
            }
            inline for (def.name) |c| {
                buf[pos] = c;
                pos += 1;
            }
        }
        const final = buf[0..pos].*;
        return &final;
    }

    fn csvLen() usize {
        @setEvalBranchQuota(10000);
        comptime var len: usize = 0;
        inline for (DEFS, 0..) |def, i| {
            if (i > 0) len += 1; // comma
            len += def.name.len;
        }
        return len;
    }

    /// Get all available schemes as a comma-separated string (legacy).
    pub const all_schemes_string: []const u8 = generateCsv();
};

/// Conversion method (derived from scheme).
pub const Method = enum {
    grouped_affine, // MLX-style grouped affine
    fp8, // FP8 hardware float
    mxfp4, // OCP Microscaling
    nvfp4, // NVIDIA Blackwell
};

// =============================================================================
// Progress Types (from unified progress API)
// =============================================================================

/// Re-export unified progress types for converter use.
pub const CProgressCallback = progress_api.CProgressCallback;
pub const ProgressUpdate = progress_api.ProgressUpdate;
pub const ProgressAction = progress_api.ProgressAction;
pub const ProgressContext = progress_api.ProgressContext;

// =============================================================================
// Override Rules
// =============================================================================

/// Maximum number of override rules.
pub const MAX_OVERRIDES: usize = 32;

/// Override rule for per-tensor quantization.
/// Pattern uses glob syntax (e.g., "model.layers.*.mlp.experts.*").
pub const OverrideRule = extern struct {
    pattern: ?[*:0]const u8 = null,
    scheme: Scheme = .gaf4_64,
};

// =============================================================================
// Conversion Options and Result
// =============================================================================

/// Conversion options.
pub const ConvertOptions = extern struct {
    /// Explicit scheme selection. Ignored if use_platform_quant is true.
    scheme: Scheme = .gaf4_64,
    force: bool = false,
    offline: bool = false,
    destination: ?[*:0]const u8 = null,
    overrides: [MAX_OVERRIDES]OverrideRule = std.mem.zeroes([MAX_OVERRIDES]OverrideRule),
    num_overrides: u32 = 0,
    /// Maximum shard size in bytes. 0 = no limit (single file).
    /// Used for splitting large models into multiple SafeTensors files.
    max_shard_size: u64 = 0,
    /// If true, estimate conversion without writing files.
    /// Returns JSON with estimation instead of output path.
    dry_run: bool = false,
    /// If true, return a model ID (org/model-suffix) instead of a filesystem path.
    return_model_id: bool = false,
    /// Target platform for automatic scheme resolution.
    platform: Platform = .cpu,
    /// Quantization level for automatic scheme resolution.
    quant: QuantLevel = .q4,
    /// If true, resolve scheme from platform/quant instead of using scheme directly.
    use_platform_quant: bool = false,
    /// Unified progress callback. Receives ProgressUpdate structs for all operations
    /// (download, convert). Binding doesn't need to know what operation is running.
    progress_callback: ?CProgressCallback = null,
    /// User-provided context pointer passed to progress_callback.
    progress_user_data: ?*anyopaque = null,

    /// Get the effective scheme, resolving from platform/quant if needed.
    pub fn getEffectiveScheme(self: ConvertOptions) Scheme {
        if (self.use_platform_quant) {
            return Scheme.resolve(self.platform, self.quant);
        }
        return self.scheme;
    }

    /// Get a ProgressContext for emitting progress updates.
    pub fn progressContext(self: ConvertOptions) ProgressContext {
        return ProgressContext.init(self.progress_callback, self.progress_user_data);
    }
};

/// Result from conversion.
pub const ConvertResult = struct {
    output_path: ?[]const u8 = null,
    err: ?anyerror = null,

    pub fn deinit(self: *ConvertResult, allocator: std.mem.Allocator) void {
        if (self.output_path) |p| allocator.free(p);
    }
};

pub const InvalidOutputPath = error{InvalidOutputPath};

/// Dry run estimation result.
pub const DryRunEstimate = struct {
    total_params: u64,
    estimated_size_bytes: u64,
    shard_count: u32,
    scheme: []const u8,
    bits_per_param: f32,

    /// Format as JSON string.
    pub fn toJson(self: DryRunEstimate, allocator: std.mem.Allocator) ![]u8 {
        return std.fmt.allocPrint(allocator,
            \\{{"total_params":{d},"estimated_size_bytes":{d},"shard_count":{d},"scheme":"{s}","bits_per_param":{d:.2}}}
        , .{
            self.total_params,
            self.estimated_size_bytes,
            self.shard_count,
            self.scheme,
            self.bits_per_param,
        });
    }
};

// =============================================================================
// Dry Run Estimation
// =============================================================================

const safetensors = @import("../io/safetensors/root.zig");
const repository = @import("../io/repository/root.zig");

/// Estimate conversion without writing files.
/// Returns JSON string with estimation results.
pub fn estimateDryRun(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    scheme: Scheme,
    options: ConvertOptions,
) ![]u8 {
    // 1. Resolve model bundle (offline option not currently used)
    var model_bundle = try repository.resolve(allocator, model_path);
    defer model_bundle.deinit();

    // 2. Load SafeTensors to count parameters (supports sharded models)
    var source_tensors = try safetensors.UnifiedSafeTensors.load(allocator, model_bundle.weights_path());
    defer source_tensors.deinit();

    // 3. Count total parameters via tensor names
    const tensor_names = try source_tensors.tensorNames(allocator);
    defer allocator.free(tensor_names);

    var total_params: u64 = 0;
    for (tensor_names) |name| {
        const tensor = source_tensors.getTensor(name, null) catch continue;
        var numel: u64 = 1;
        for (tensor.shape[0..@intCast(tensor.n_dims)]) |dim| {
            numel *= @intCast(dim);
        }
        total_params += numel;
    }

    // 4. Estimate size based on scheme
    const bits_x100 = scheme.getEffectiveBitsX100();
    const bits_per_param: f32 = @as(f32, @floatFromInt(bits_x100)) / 100.0;
    const estimated_size_bytes: u64 = (total_params * bits_x100) / 800; // bits_x100 / 8 / 100

    // 5. Calculate shard count
    const shard_count: u32 = if (options.max_shard_size > 0)
        @intCast(@max(1, (estimated_size_bytes + options.max_shard_size - 1) / options.max_shard_size))
    else
        1;

    // 6. Build result
    const estimate = DryRunEstimate{
        .total_params = total_params,
        .estimated_size_bytes = estimated_size_bytes,
        .shard_count = shard_count,
        .scheme = scheme.toString(),
        .bits_per_param = bits_per_param,
    };

    return estimate.toJson(allocator);
}

// =============================================================================
// Conversion Entry Point
// =============================================================================

/// Convert a model using the specified scheme.
/// Dispatches to the appropriate converter based on the scheme's method.
/// If dry_run is true, returns JSON estimation instead of performing conversion.
pub fn convert(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    output_dir: []const u8,
    options: ConvertOptions,
) ConvertResult {
    // Resolve effective scheme (from platform/quant or explicit)
    const scheme = options.getEffectiveScheme();

    // Handle dry run mode
    if (options.dry_run) {
        const json = estimateDryRun(allocator, model_path, scheme, options) catch |err| {
            return .{ .err = err };
        };
        return .{ .output_path = json };
    }

    const destination: ?[]const u8 = if (options.destination) |d| std.mem.span(d) else null;

    switch (scheme.getMethod()) {
        .grouped_affine => {
            if (options.num_overrides > 0) {
                return .{ .err = error.InvalidArgument };
            }

            const output_path = grouped_affine.convertToGroupedAffine(allocator, model_path, .{
                .quant = .{
                    .bits = scheme.getBits(),
                    .group_size = scheme.getGroupSize(),
                },
                .output_dir = output_dir,
                .destination = destination,
                .output_suffix = scheme.toOutputSuffix(),
                .force = options.force,
                .max_shard_size = options.max_shard_size,
                .progress = options.progressContext(),
            }) catch |err| {
                return .{ .err = err };
            };

            if (options.return_model_id) {
                const model_id = grouped_affine.modelIdFromOutputPath(allocator, output_path) catch {
                    allocator.free(output_path);
                    return .{ .err = error.InvalidOutputPath };
                };
                allocator.free(output_path);
                return .{ .output_path = model_id };
            }

            return .{ .output_path = output_path };
        },
        .fp8, .mxfp4, .nvfp4 => {
            return .{ .err = error.UnsupportedFormat };
        },
    }
}

// =============================================================================
// Tests
// =============================================================================

test "Scheme.getMethod" {
    try std.testing.expectEqual(Method.grouped_affine, Scheme.f16.getMethod());
    try std.testing.expectEqual(Method.grouped_affine, Scheme.gaf4_64.getMethod());
    try std.testing.expectEqual(Method.fp8, Scheme.fp8_e4m3.getMethod());
}

test "Scheme.getBits" {
    try std.testing.expectEqual(@as(u8, 16), Scheme.f16.getBits());
    try std.testing.expectEqual(@as(u8, 4), Scheme.gaf4_64.getBits());
    try std.testing.expectEqual(@as(u8, 8), Scheme.gaf8_64.getBits());
}

test "Scheme.getGroupSize" {
    try std.testing.expectEqual(@as(u32, 32), Scheme.gaf4_32.getGroupSize());
    try std.testing.expectEqual(@as(u32, 64), Scheme.gaf4_64.getGroupSize());
    try std.testing.expectEqual(@as(u32, 128), Scheme.gaf4_128.getGroupSize());
}

test "Scheme.fromString" {
    // Canonical names
    try std.testing.expectEqual(Scheme.gaf4_64, Scheme.fromString("gaf4_64").?);
    try std.testing.expect(Scheme.fromString("invalid") == null);

    // Aliases - now map to GAF
    try std.testing.expectEqual(Scheme.gaf4_64, Scheme.fromString("4bit").?);
    try std.testing.expectEqual(Scheme.gaf4_64, Scheme.fromString("q4").?);
    try std.testing.expectEqual(Scheme.gaf4_64, Scheme.fromString("int4").?);
    try std.testing.expectEqual(Scheme.gaf8_64, Scheme.fromString("8bit").?);
    try std.testing.expectEqual(Scheme.gaf8_64, Scheme.fromString("int8").?);
    try std.testing.expectEqual(Scheme.f16, Scheme.fromString("16bit").?);
    try std.testing.expectEqual(Scheme.f16, Scheme.fromString("half").?);
    try std.testing.expectEqual(Scheme.gaf4_64, Scheme.fromString("mlx").?);
    try std.testing.expectEqual(Scheme.gaf4_64, Scheme.fromString("mlx4").?);
    try std.testing.expectEqual(Scheme.gaf8_64, Scheme.fromString("mlx8").?);
    try std.testing.expectEqual(Scheme.fp8_e4m3, Scheme.fromString("fp8").?);

    // Case insensitive aliases
    try std.testing.expectEqual(Scheme.gaf4_64, Scheme.fromString("4BIT").?);
    try std.testing.expectEqual(Scheme.gaf4_64, Scheme.fromString("MLX").?);
}

test "Scheme.toString" {
    try std.testing.expectEqualStrings("gaf4_64", Scheme.gaf4_64.toString());
    try std.testing.expectEqualStrings("f16", Scheme.f16.toString());
}

test "Scheme.all_schemes_json" {
    const json = Scheme.all_schemes_json;
    // Check it's valid JSON structure
    try std.testing.expect(json[0] == '{');
    try std.testing.expect(json[json.len - 1] == '}');
    // Check it contains expected schemes and aliases
    try std.testing.expect(std.mem.indexOf(u8, json, "\"gaf4_64\":[\"mlx\",\"mlx4\",\"gaf4\",\"4bit\",\"q4\",\"int4\"]") != null);
}

test "Scheme.all_schemes_string" {
    const csv = Scheme.all_schemes_string;
    // Check it contains expected schemes
    try std.testing.expect(std.mem.indexOf(u8, csv, "gaf4_64") != null);
    try std.testing.expect(std.mem.indexOf(u8, csv, "gaf8_64") != null);
}

test "Platform.fromString" {
    // CPU variants
    try std.testing.expectEqual(Platform.cpu, Platform.fromString("cpu").?);
    try std.testing.expectEqual(Platform.cpu, Platform.fromString("CPU").?);

    // Metal variants
    try std.testing.expectEqual(Platform.metal, Platform.fromString("metal").?);
    try std.testing.expectEqual(Platform.metal, Platform.fromString("mps").?);
    try std.testing.expectEqual(Platform.metal, Platform.fromString("apple").?);
    try std.testing.expectEqual(Platform.metal, Platform.fromString("METAL").?);

    // CUDA variants
    try std.testing.expectEqual(Platform.cuda, Platform.fromString("cuda").?);
    try std.testing.expectEqual(Platform.cuda, Platform.fromString("gpu").?);
    try std.testing.expectEqual(Platform.cuda, Platform.fromString("nvidia").?);

    // Invalid
    try std.testing.expect(Platform.fromString("invalid") == null);
}

test "Platform.toString" {
    try std.testing.expectEqualStrings("cpu", Platform.cpu.toString());
    try std.testing.expectEqualStrings("metal", Platform.metal.toString());
    try std.testing.expectEqualStrings("cuda", Platform.cuda.toString());
}

test "QuantLevel.fromString" {
    // 4-bit variants
    try std.testing.expectEqual(QuantLevel.q4, QuantLevel.fromString("4bit").?);
    try std.testing.expectEqual(QuantLevel.q4, QuantLevel.fromString("q4").?);
    try std.testing.expectEqual(QuantLevel.q4, QuantLevel.fromString("int4").?);
    try std.testing.expectEqual(QuantLevel.q4, QuantLevel.fromString("4BIT").?);

    // 8-bit variants
    try std.testing.expectEqual(QuantLevel.q8, QuantLevel.fromString("8bit").?);
    try std.testing.expectEqual(QuantLevel.q8, QuantLevel.fromString("q8").?);
    try std.testing.expectEqual(QuantLevel.q8, QuantLevel.fromString("int8").?);

    // 16-bit variants
    try std.testing.expectEqual(QuantLevel.q16, QuantLevel.fromString("16bit").?);
    try std.testing.expectEqual(QuantLevel.q16, QuantLevel.fromString("q16").?);
    try std.testing.expectEqual(QuantLevel.q16, QuantLevel.fromString("f16").?);
    try std.testing.expectEqual(QuantLevel.q16, QuantLevel.fromString("half").?);

    // Invalid
    try std.testing.expect(QuantLevel.fromString("invalid") == null);
}

test "QuantLevel.toString" {
    try std.testing.expectEqualStrings("4bit", QuantLevel.q4.toString());
    try std.testing.expectEqualStrings("8bit", QuantLevel.q8.toString());
    try std.testing.expectEqualStrings("16bit", QuantLevel.q16.toString());
}

test "Scheme.resolve - all platforms use GAF" {
    // All platforms now use GAF schemes
    try std.testing.expectEqual(Scheme.gaf4_64, Scheme.resolve(.cpu, .q4));
    try std.testing.expectEqual(Scheme.gaf8_64, Scheme.resolve(.cpu, .q8));
    try std.testing.expectEqual(Scheme.f16, Scheme.resolve(.cpu, .q16));

    try std.testing.expectEqual(Scheme.gaf4_64, Scheme.resolve(.metal, .q4));
    try std.testing.expectEqual(Scheme.gaf8_64, Scheme.resolve(.metal, .q8));
    try std.testing.expectEqual(Scheme.f16, Scheme.resolve(.metal, .q16));

    try std.testing.expectEqual(Scheme.gaf4_64, Scheme.resolve(.cuda, .q4));
    try std.testing.expectEqual(Scheme.gaf8_64, Scheme.resolve(.cuda, .q8));
    try std.testing.expectEqual(Scheme.f16, Scheme.resolve(.cuda, .q16));
}

test "ConvertOptions.getEffectiveScheme - explicit scheme" {
    var opts = std.mem.zeroes(ConvertOptions);
    opts.scheme = .gaf8_64;
    opts.use_platform_quant = false;
    try std.testing.expectEqual(Scheme.gaf8_64, opts.getEffectiveScheme());
}

test "ConvertOptions.getEffectiveScheme - platform/quant resolution" {
    var opts = std.mem.zeroes(ConvertOptions);
    opts.use_platform_quant = true;

    // CPU + 4bit -> gaf4_64
    opts.platform = .cpu;
    opts.quant = .q4;
    try std.testing.expectEqual(Scheme.gaf4_64, opts.getEffectiveScheme());

    // Metal + 4bit -> gaf4_64
    opts.platform = .metal;
    opts.quant = .q4;
    try std.testing.expectEqual(Scheme.gaf4_64, opts.getEffectiveScheme());

    // Metal + 8bit -> gaf8_64
    opts.platform = .metal;
    opts.quant = .q8;
    try std.testing.expectEqual(Scheme.gaf8_64, opts.getEffectiveScheme());
}
