//! Conversion Scheme Definitions
//!
//! Unified quantization scheme that encodes all conversion settings.
//! This eliminates invalid parameter combinations and provides a
//! simple, self-documenting API.

const std = @import("std");
const grouped_affine = @import("grouped_affine.zig");
const fp8_converter = @import("fp8.zig");
const mxfp8_converter = @import("mxfp8.zig");
const nvfp4_converter = @import("nvfp4.zig");
const progress_api = @import("../capi/progress.zig");
const json = @import("../io/json/root.zig");
const safetensors = @import("../io/safetensors/root.zig");
const repository = @import("../io/repository/root.zig");

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
    .{ .val = .mxfp8, .name = "mxfp8", .aliases = &.{} },
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

/// Conversion quality profile used by hardware-float converters.
pub const QualityProfile = enum(u32) {
    best,
    good,
    balanced,
    fast,
    custom,

    pub fn fromString(s: []const u8) ?QualityProfile {
        if (std.ascii.eqlIgnoreCase(s, "best") or std.ascii.eqlIgnoreCase(s, "quality")) return .best;
        if (std.ascii.eqlIgnoreCase(s, "good")) return .good;
        if (std.ascii.eqlIgnoreCase(s, "balanced")) return .balanced;
        if (std.ascii.eqlIgnoreCase(s, "fast") or std.ascii.eqlIgnoreCase(s, "quick")) return .fast;
        if (std.ascii.eqlIgnoreCase(s, "custom")) return .custom;
        return null;
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
/// ## Hardware Float
/// - fp8_e4m3, fp8_e5m2, mxfp4, nvfp4, mxfp8
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
    mxfp8 = 24, // OCP Microscaling 8-bit E4M3 + E8M0 scales (group_size=32)

    /// Get the conversion method for this scheme.
    pub fn getMethod(self: Scheme) Method {
        return switch (self) {
            .f16, .gaf4_32, .gaf4_64, .gaf4_128, .gaf8_32, .gaf8_64, .gaf8_128 => .grouped_affine,
            .fp8_e4m3, .fp8_e5m2 => .fp8,
            .mxfp4 => .mxfp4,
            .nvfp4 => .nvfp4,
            .mxfp8 => .mxfp8,
        };
    }

    /// Get the bit width for this scheme.
    pub fn getBits(self: Scheme) u8 {
        return switch (self) {
            .gaf4_32, .gaf4_64, .gaf4_128, .mxfp4, .nvfp4 => 4,
            .gaf8_32, .gaf8_64, .gaf8_128, .fp8_e4m3, .fp8_e5m2, .mxfp8 => 8,
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

            // Microscaling: data bits + scale overhead
            .mxfp4 => 500, // 4 + 1.0 (group_size=32)
            .nvfp4 => 500, // 4 + 1.0 (group_size=32)
            .mxfp8 => 825, // 8 + 0.25 (1 byte scale per 32 elements)
        };
    }

    /// Get the group size for this scheme (0 if not applicable).
    pub fn getGroupSize(self: Scheme) u32 {
        return switch (self) {
            .gaf4_32, .gaf8_32, .mxfp4, .nvfp4, .mxfp8 => 32,
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
            .mxfp8 => "MXFP8",
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
    mxfp8, // OCP Microscaling FP8
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
    /// Calibration profile for MXFP8/NVFP4 conversion.
    calibration_profile: QualityProfile = .best,
    /// Deterministic calibration seed.
    calibration_seed: u64 = 42,
    /// Explicit calibration iteration override.
    /// Zero means use profile default.
    calibration_iters: u32 = 0,
    /// Explicit calibration sample override.
    /// Zero means use profile default.
    calibration_nsamples: u32 = 0,
    /// Explicit calibration sequence length override.
    /// Zero means use profile default.
    calibration_seqlen: u32 = 0,
    /// Explicit calibration batch-size override.
    /// Zero means use profile default.
    calibration_batch_size: u32 = 0,
    /// Explicit calibration block-count override.
    /// Zero means use profile default.
    calibration_nblocks: u32 = 0,
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
pub const supported_quant_contract_version: i64 = 1;

const CalibrationSettings = struct {
    profile: QualityProfile,
    iters: u32,
    nsamples: u32,
    seqlen: u32,
    batch_size: u32,
    nblocks: u32,
    seed: u64,
};

fn defaultCalibrationFor(profile: QualityProfile, scheme: Scheme) CalibrationSettings {
    const effective_profile: QualityProfile = if (profile == .custom) .best else profile;
    return switch (scheme) {
        .mxfp8 => switch (effective_profile) {
            .best => .{ .profile = profile, .iters = 128, .nsamples = 256, .seqlen = 2048, .batch_size = 1, .nblocks = 1, .seed = 42 },
            .good => .{ .profile = profile, .iters = 64, .nsamples = 256, .seqlen = 2048, .batch_size = 1, .nblocks = 1, .seed = 42 },
            .balanced => .{ .profile = profile, .iters = 32, .nsamples = 256, .seqlen = 2048, .batch_size = 1, .nblocks = 1, .seed = 42 },
            .fast => .{ .profile = profile, .iters = 32, .nsamples = 256, .seqlen = 2048, .batch_size = 1, .nblocks = 1, .seed = 42 },
            .custom => unreachable,
        },
        .nvfp4 => switch (effective_profile) {
            .best => .{ .profile = profile, .iters = 500, .nsamples = 256, .seqlen = 2048, .batch_size = 1, .nblocks = 1, .seed = 42 },
            .good => .{ .profile = profile, .iters = 350, .nsamples = 192, .seqlen = 1536, .batch_size = 1, .nblocks = 1, .seed = 42 },
            // Balanced is an intentionally lighter clip+search profile.
            .balanced => .{ .profile = profile, .iters = 16, .nsamples = 128, .seqlen = 1024, .batch_size = 1, .nblocks = 1, .seed = 42 },
            .fast => .{ .profile = profile, .iters = 1, .nsamples = 16, .seqlen = 256, .batch_size = 1, .nblocks = 1, .seed = 42 },
            .custom => unreachable,
        },
        .gaf4_32, .gaf4_64, .gaf4_128, .gaf8_32, .gaf8_64, .gaf8_128 => switch (effective_profile) {
            .best => .{ .profile = profile, .iters = 32, .nsamples = 256, .seqlen = 2048, .batch_size = 1, .nblocks = 1, .seed = 42 },
            .good => .{ .profile = profile, .iters = 16, .nsamples = 128, .seqlen = 1536, .batch_size = 1, .nblocks = 1, .seed = 42 },
            .balanced => .{ .profile = profile, .iters = 8, .nsamples = 64, .seqlen = 1024, .batch_size = 1, .nblocks = 1, .seed = 42 },
            .fast => .{ .profile = profile, .iters = 1, .nsamples = 16, .seqlen = 256, .batch_size = 1, .nblocks = 1, .seed = 42 },
            .custom => unreachable,
        },
        else => .{ .profile = profile, .iters = 1, .nsamples = 16, .seqlen = 256, .batch_size = 1, .nblocks = 1, .seed = 42 },
    };
}

fn resolveCalibrationFromOptions(options: ConvertOptions, scheme: Scheme) CalibrationSettings {
    var settings = defaultCalibrationFor(options.calibration_profile, scheme);
    settings.seed = options.calibration_seed;
    if (options.calibration_iters > 0) settings.iters = options.calibration_iters;
    if (options.calibration_nsamples > 0) settings.nsamples = options.calibration_nsamples;
    if (options.calibration_seqlen > 0) settings.seqlen = options.calibration_seqlen;
    if (options.calibration_batch_size > 0) settings.batch_size = options.calibration_batch_size;
    if (options.calibration_nblocks > 0) settings.nblocks = options.calibration_nblocks;
    return settings;
}

fn parseBoolEnv(value: []const u8) ?bool {
    if (value.len == 0) return null;
    const trimmed = std.mem.trim(u8, value, " \t\r\n");
    if (trimmed.len == 0) return null;
    if (std.ascii.eqlIgnoreCase(trimmed, "1")) return true;
    if (std.ascii.eqlIgnoreCase(trimmed, "true")) return true;
    if (std.ascii.eqlIgnoreCase(trimmed, "yes")) return true;
    if (std.ascii.eqlIgnoreCase(trimmed, "on")) return true;
    if (std.ascii.eqlIgnoreCase(trimmed, "0")) return false;
    if (std.ascii.eqlIgnoreCase(trimmed, "false")) return false;
    if (std.ascii.eqlIgnoreCase(trimmed, "no")) return false;
    if (std.ascii.eqlIgnoreCase(trimmed, "off")) return false;
    return null;
}

fn envFlagEnabled(name: []const u8) bool {
    const value = std.process.getEnvVarOwned(std.heap.page_allocator, name) catch return false;
    defer std.heap.page_allocator.free(value);
    return parseBoolEnv(value) orelse false;
}

fn isMxfp8CalibrationProbeOnly(calibration: CalibrationSettings) bool {
    return calibration.profile == .custom and calibration.iters > 0 and envFlagEnabled("TALU_CONVERT_CALIB_PROBE_ONLY");
}

fn mapJsonParseError(err: anyerror) anyerror {
    return switch (err) {
        error.InputTooLarge, error.InputTooDeep, error.StringTooLong, error.InvalidJson => error.InvalidConfig,
        else => err,
    };
}

fn parseConfigAtPath(allocator: std.mem.Allocator, config_path: []const u8) !std.json.Parsed(std.json.Value) {
    const config_bytes = try std.fs.cwd().readFileAlloc(allocator, config_path, 1024 * 1024);
    defer allocator.free(config_bytes);
    return json.parseValue(allocator, config_bytes, .{
        .max_size_bytes = 1024 * 1024,
        .max_value_bytes = 1024 * 1024,
        .max_string_bytes = 256 * 1024,
    }) catch |err| return mapJsonParseError(err);
}

fn objectField(obj: std.json.ObjectMap, key: []const u8) ?std.json.Value {
    return obj.get(key);
}

fn objectFieldAsObject(obj: std.json.ObjectMap, key: []const u8) ?std.json.ObjectMap {
    const value = objectField(obj, key) orelse return null;
    if (value != .object) return null;
    return value.object;
}

fn objectFieldAsString(obj: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const value = objectField(obj, key) orelse return null;
    if (value != .string) return null;
    return value.string;
}

fn objectFieldAsInt(obj: std.json.ObjectMap, key: []const u8) ?i64 {
    const value = objectField(obj, key) orelse return null;
    return switch (value) {
        .integer => value.integer,
        else => null,
    };
}

fn resolveOutputWeightsPath(allocator: std.mem.Allocator, output_path: []const u8) ![]u8 {
    const single_path = try std.fs.path.join(allocator, &.{ output_path, "model.safetensors" });
    errdefer allocator.free(single_path);
    if (std.fs.cwd().access(single_path, .{})) |_| return single_path else |err| switch (err) {
        error.FileNotFound => {},
        else => return err,
    }

    allocator.free(single_path);
    const index_path = try std.fs.path.join(allocator, &.{ output_path, "model.safetensors.index.json" });
    errdefer allocator.free(index_path);
    if (std.fs.cwd().access(index_path, .{})) |_| return index_path else |err| switch (err) {
        error.FileNotFound => return error.WeightsNotFound,
        else => return err,
    }
}

fn validateMxfp8Config(config_obj: std.json.ObjectMap) !void {
    const qcfg = objectFieldAsObject(config_obj, "quantization_config") orelse return error.InvalidConfig;
    if (!std.mem.eql(u8, objectFieldAsString(qcfg, "quant_method") orelse return error.InvalidConfig, "mxfp8")) {
        return error.InvalidConfig;
    }
    if ((objectFieldAsInt(qcfg, "quant_contract_version") orelse return error.InvalidConfig) != supported_quant_contract_version) {
        return error.InvalidConfig;
    }
    if (!std.mem.eql(u8, objectFieldAsString(qcfg, "fmt") orelse return error.InvalidConfig, "e4m3")) {
        return error.InvalidConfig;
    }
    if (!std.mem.eql(u8, objectFieldAsString(qcfg, "scale_fmt") orelse return error.InvalidConfig, "e8m0")) {
        return error.InvalidConfig;
    }
    if ((objectFieldAsInt(qcfg, "block_size") orelse return error.InvalidConfig) != 32) {
        return error.InvalidConfig;
    }
}

fn validateNvfp4Config(config_obj: std.json.ObjectMap) !void {
    const quant = objectFieldAsObject(config_obj, "quantization") orelse return error.InvalidConfig;
    if ((objectFieldAsInt(quant, "group_size") orelse return error.InvalidConfig) != 32) return error.InvalidConfig;
    if ((objectFieldAsInt(quant, "bits") orelse return error.InvalidConfig) != 4) return error.InvalidConfig;

    const qcfg = objectFieldAsObject(config_obj, "quantization_config") orelse return error.InvalidConfig;
    if (!std.mem.eql(u8, objectFieldAsString(qcfg, "quant_method") orelse return error.InvalidConfig, "nvfp4")) {
        return error.InvalidConfig;
    }
    if ((objectFieldAsInt(qcfg, "quant_contract_version") orelse return error.InvalidConfig) != supported_quant_contract_version) {
        return error.InvalidConfig;
    }
    if ((objectFieldAsInt(qcfg, "group_size") orelse return error.InvalidConfig) != 32) return error.InvalidConfig;
    if ((objectFieldAsInt(qcfg, "bits") orelse return error.InvalidConfig) != 4) return error.InvalidConfig;
}

fn validateMxfp8Weights(allocator: std.mem.Allocator, output_path: []const u8) !void {
    const weights_path = try resolveOutputWeightsPath(allocator, output_path);
    defer allocator.free(weights_path);

    var st = try safetensors.UnifiedSafeTensors.load(allocator, weights_path);
    defer st.deinit();

    const names = try st.tensorNames(allocator);
    defer allocator.free(names);

    var saw_scale = false;
    for (names) |name| {
        if (!std.mem.endsWith(u8, name, ".weight_block_scale")) continue;
        saw_scale = true;

        const scale_tensor = try st.getTensor(name, null);
        if (scale_tensor.n_dims != 2) return error.InvalidConfig;
        if (scale_tensor.dtype != .u8 and scale_tensor.dtype != .i8) return error.InvalidConfig;

        const base = name[0 .. name.len - ".weight_block_scale".len];
        const weight_name = try std.fmt.allocPrint(allocator, "{s}.weight", .{base});
        defer allocator.free(weight_name);
        if (!st.hasTensor(weight_name)) return error.InvalidConfig;

        const weight_tensor = try st.getTensor(weight_name, null);
        if (weight_tensor.dtype != .f8_e4m3) return error.InvalidConfig;
        if (weight_tensor.n_dims != 2) return error.InvalidConfig;

        const rows: i64 = weight_tensor.shape[0];
        const cols: i64 = weight_tensor.shape[1];
        const scale_rows: i64 = scale_tensor.shape[0];
        const scale_cols: i64 = scale_tensor.shape[1];
        if (rows <= 0 or cols <= 0) return error.InvalidConfig;
        if (rows != scale_rows) return error.InvalidConfig;
        const cols_usize = std.math.cast(usize, cols) orelse return error.InvalidConfig;
        const expected_scale_cols: i64 = @intCast((cols_usize + 31) / 32);
        if (scale_cols != expected_scale_cols) return error.InvalidConfig;
    }

    if (!saw_scale) return error.InvalidConfig;
}

fn validateNvfp4Weights(allocator: std.mem.Allocator, output_path: []const u8) !void {
    const weights_path = try resolveOutputWeightsPath(allocator, output_path);
    defer allocator.free(weights_path);

    var st = try safetensors.UnifiedSafeTensors.load(allocator, weights_path);
    defer st.deinit();

    const names = try st.tensorNames(allocator);
    defer allocator.free(names);

    var saw_nvfp4_weight = false;
    for (names) |name| {
        const weight_tensor = st.getTensor(name, null) catch continue;
        if (weight_tensor.dtype != .grouped_affine_u4) continue;
        saw_nvfp4_weight = true;
        if (!std.mem.endsWith(u8, name, ".weight")) continue;
        if (weight_tensor.n_dims != 2) return error.InvalidConfig;
        if (weight_tensor.shape[0] <= 0 or weight_tensor.shape[1] <= 0) return error.InvalidConfig;

        const base = name[0 .. name.len - ".weight".len];
        const scales_name = try std.fmt.allocPrint(allocator, "{s}.scales", .{base});
        defer allocator.free(scales_name);
        const biases_name = try std.fmt.allocPrint(allocator, "{s}.biases", .{base});
        defer allocator.free(biases_name);
        if (!st.hasTensor(scales_name) or !st.hasTensor(biases_name)) return error.InvalidConfig;

        const scales = try st.getTensor(scales_name, null);
        const biases = try st.getTensor(biases_name, null);
        if (scales.n_dims != 2 or biases.n_dims != 2) return error.InvalidConfig;
        if ((scales.dtype != .bf16 and scales.dtype != .f16) or (biases.dtype != .bf16 and biases.dtype != .f16)) {
            return error.InvalidConfig;
        }
        if (scales.shape[0] != weight_tensor.shape[0] or biases.shape[0] != weight_tensor.shape[0]) return error.InvalidConfig;

        const packed_cols: i64 = weight_tensor.shape[1];
        if (@mod(packed_cols, 4) != 0) return error.InvalidConfig;
        const expected_group_cols: i64 = @divTrunc(packed_cols, 4);
        if (scales.shape[1] != expected_group_cols or biases.shape[1] != expected_group_cols) return error.InvalidConfig;
    }

    if (!saw_nvfp4_weight) return error.InvalidConfig;
}

fn validateCanonicalOutput(allocator: std.mem.Allocator, output_path: []const u8, scheme: Scheme) !void {
    if (scheme != .mxfp8 and scheme != .nvfp4) return;

    const config_path = try std.fs.path.join(allocator, &.{ output_path, "config.json" });
    defer allocator.free(config_path);
    var parsed = try parseConfigAtPath(allocator, config_path);
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidConfig;
    const config_obj = parsed.value.object;

    switch (scheme) {
        .mxfp8 => {
            try validateMxfp8Config(config_obj);
            try validateMxfp8Weights(allocator, output_path);
        },
        .nvfp4 => {
            try validateNvfp4Config(config_obj);
            try validateNvfp4Weights(allocator, output_path);
        },
        else => unreachable,
    }
}

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

/// Estimate conversion without writing files.
/// Returns JSON string with estimation results.
pub fn estimateDryRun(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    scheme: Scheme,
    options: ConvertOptions,
) ![]u8 {
    // 1. Resolve model bundle (offline option not currently used)
    var model_bundle = try repository.resolve(allocator, model_path, .{});
    defer model_bundle.deinit();

    // 2. Load SafeTensors to count parameters (supports sharded models)
    var source_tensors = try safetensors.UnifiedSafeTensors.load(allocator, model_bundle.weights_path() orelse return error.WeightsNotFound);
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
    const calibration = resolveCalibrationFromOptions(options, scheme);

    // Handle dry run mode
    if (options.dry_run) {
        const estimate_json = estimateDryRun(allocator, model_path, scheme, options) catch |err| {
            return .{ .err = err };
        };
        return .{ .output_path = estimate_json };
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
                .profile = calibration.profile,
                .calib_iters = calibration.iters,
                .calib_nsamples = calibration.nsamples,
                .calib_seqlen = calibration.seqlen,
                .calib_batch_size = calibration.batch_size,
                .calib_nblocks = calibration.nblocks,
                .calib_seed = calibration.seed,
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
        .fp8 => {
            const output_path = fp8_converter.convertToFp8(allocator, model_path, .{
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
                const model_id = fp8_converter.modelIdFromOutputPath(allocator, output_path) catch {
                    allocator.free(output_path);
                    return .{ .err = error.InvalidOutputPath };
                };
                allocator.free(output_path);
                return .{ .output_path = model_id };
            }

            return .{ .output_path = output_path };
        },
        .mxfp8 => {
            const probe_only = isMxfp8CalibrationProbeOnly(calibration);
            const output_path = mxfp8_converter.convertToMxfp8(allocator, model_path, .{
                .output_dir = output_dir,
                .destination = destination,
                .output_suffix = scheme.toOutputSuffix(),
                .force = options.force,
                .max_shard_size = options.max_shard_size,
                .progress = options.progressContext(),
                .profile = calibration.profile,
                .calib_iters = calibration.iters,
                .calib_nsamples = calibration.nsamples,
                .calib_seqlen = calibration.seqlen,
                .calib_batch_size = calibration.batch_size,
                .calib_nblocks = calibration.nblocks,
                .calib_seed = calibration.seed,
            }) catch |err| {
                return .{ .err = err };
            };

            if (!probe_only) {
                validateCanonicalOutput(allocator, output_path, .mxfp8) catch |err| {
                    allocator.free(output_path);
                    return .{ .err = err };
                };
            }

            if (options.return_model_id) {
                const model_id = mxfp8_converter.modelIdFromOutputPath(allocator, output_path) catch {
                    allocator.free(output_path);
                    return .{ .err = error.InvalidOutputPath };
                };
                allocator.free(output_path);
                return .{ .output_path = model_id };
            }

            return .{ .output_path = output_path };
        },
        .nvfp4 => {
            if (options.num_overrides > 0) {
                return .{ .err = error.InvalidArgument };
            }

            const output_path = nvfp4_converter.convertToNvfp4(allocator, model_path, .{
                .output_dir = output_dir,
                .destination = destination,
                .output_suffix = scheme.toOutputSuffix(),
                .force = options.force,
                .max_shard_size = options.max_shard_size,
                .progress = options.progressContext(),
                .profile = calibration.profile,
                .calib_iters = calibration.iters,
                .calib_nsamples = calibration.nsamples,
                .calib_seqlen = calibration.seqlen,
                .calib_batch_size = calibration.batch_size,
                .calib_nblocks = calibration.nblocks,
                .calib_seed = calibration.seed,
            }) catch |err| {
                return .{ .err = err };
            };

            validateCanonicalOutput(allocator, output_path, .nvfp4) catch |err| {
                allocator.free(output_path);
                return .{ .err = err };
            };

            if (options.return_model_id) {
                const model_id = nvfp4_converter.modelIdFromOutputPath(allocator, output_path) catch {
                    allocator.free(output_path);
                    return .{ .err = error.InvalidOutputPath };
                };
                allocator.free(output_path);
                return .{ .output_path = model_id };
            }

            return .{ .output_path = output_path };
        },
        .mxfp4 => {
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
    const scheme_json = Scheme.all_schemes_json;
    // Check it's valid JSON structure
    try std.testing.expect(scheme_json[0] == '{');
    try std.testing.expect(scheme_json[scheme_json.len - 1] == '}');
    // Check it contains expected schemes and aliases
    try std.testing.expect(std.mem.indexOf(u8, scheme_json, "\"gaf4_64\":[\"mlx\",\"mlx4\",\"gaf4\",\"4bit\",\"q4\",\"int4\"]") != null);
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

test "QualityProfile.fromString" {
    try std.testing.expectEqual(QualityProfile.best, QualityProfile.fromString("best").?);
    try std.testing.expectEqual(QualityProfile.best, QualityProfile.fromString("QUALITY").?);
    try std.testing.expectEqual(QualityProfile.good, QualityProfile.fromString("good").?);
    try std.testing.expectEqual(QualityProfile.balanced, QualityProfile.fromString("balanced").?);
    try std.testing.expectEqual(QualityProfile.fast, QualityProfile.fromString("fast").?);
    try std.testing.expectEqual(QualityProfile.fast, QualityProfile.fromString("quick").?);
    try std.testing.expectEqual(QualityProfile.custom, QualityProfile.fromString("custom").?);
    try std.testing.expect(QualityProfile.fromString("invalid") == null);
}

test "defaultCalibrationFor mxfp8 profiles" {
    const best = defaultCalibrationFor(.best, .mxfp8);
    try std.testing.expectEqual(@as(u32, 128), best.iters);
    try std.testing.expectEqual(@as(u32, 256), best.nsamples);
    try std.testing.expectEqual(@as(u32, 2048), best.seqlen);
    try std.testing.expectEqual(@as(u32, 1), best.batch_size);
    try std.testing.expectEqual(@as(u32, 1), best.nblocks);

    const good = defaultCalibrationFor(.good, .mxfp8);
    try std.testing.expectEqual(@as(u32, 64), good.iters);
    try std.testing.expectEqual(@as(u32, 256), good.nsamples);
    try std.testing.expectEqual(@as(u32, 2048), good.seqlen);

    const balanced = defaultCalibrationFor(.balanced, .mxfp8);
    try std.testing.expectEqual(@as(u32, 32), balanced.iters);
    try std.testing.expectEqual(@as(u32, 256), balanced.nsamples);
    try std.testing.expectEqual(@as(u32, 2048), balanced.seqlen);

    const fast = defaultCalibrationFor(.fast, .mxfp8);
    try std.testing.expectEqual(@as(u32, 32), fast.iters);
    try std.testing.expectEqual(@as(u32, 256), fast.nsamples);
    try std.testing.expectEqual(@as(u32, 2048), fast.seqlen);
}

test "defaultCalibrationFor nvfp4 profiles" {
    const best = defaultCalibrationFor(.best, .nvfp4);
    try std.testing.expectEqual(@as(u32, 500), best.iters);
    try std.testing.expectEqual(@as(u32, 256), best.nsamples);
    try std.testing.expectEqual(@as(u32, 2048), best.seqlen);
    try std.testing.expectEqual(@as(u32, 1), best.batch_size);
    try std.testing.expectEqual(@as(u32, 1), best.nblocks);

    const good = defaultCalibrationFor(.good, .nvfp4);
    try std.testing.expectEqual(@as(u32, 350), good.iters);
    try std.testing.expectEqual(@as(u32, 192), good.nsamples);
    try std.testing.expectEqual(@as(u32, 1536), good.seqlen);

    const balanced = defaultCalibrationFor(.balanced, .nvfp4);
    try std.testing.expectEqual(@as(u32, 16), balanced.iters);
    try std.testing.expectEqual(@as(u32, 128), balanced.nsamples);
    try std.testing.expectEqual(@as(u32, 1024), balanced.seqlen);

    const fast = defaultCalibrationFor(.fast, .nvfp4);
    try std.testing.expectEqual(@as(u32, 1), fast.iters);
    try std.testing.expectEqual(@as(u32, 16), fast.nsamples);
    try std.testing.expectEqual(@as(u32, 256), fast.seqlen);
}

test "defaultCalibrationFor grouped-affine profiles" {
    const best = defaultCalibrationFor(.best, .gaf8_64);
    try std.testing.expectEqual(@as(u32, 32), best.iters);
    try std.testing.expectEqual(@as(u32, 256), best.nsamples);
    try std.testing.expectEqual(@as(u32, 2048), best.seqlen);

    const good = defaultCalibrationFor(.good, .gaf8_64);
    try std.testing.expectEqual(@as(u32, 16), good.iters);
    try std.testing.expectEqual(@as(u32, 128), good.nsamples);
    try std.testing.expectEqual(@as(u32, 1536), good.seqlen);

    const balanced = defaultCalibrationFor(.balanced, .gaf8_64);
    try std.testing.expectEqual(@as(u32, 8), balanced.iters);
    try std.testing.expectEqual(@as(u32, 64), balanced.nsamples);
    try std.testing.expectEqual(@as(u32, 1024), balanced.seqlen);

    const fast = defaultCalibrationFor(.fast, .gaf8_64);
    try std.testing.expectEqual(@as(u32, 1), fast.iters);
    try std.testing.expectEqual(@as(u32, 16), fast.nsamples);
    try std.testing.expectEqual(@as(u32, 256), fast.seqlen);
}

test "defaultCalibrationFor custom uses best defaults" {
    const mxfp8_custom = defaultCalibrationFor(.custom, .mxfp8);
    try std.testing.expectEqual(QualityProfile.custom, mxfp8_custom.profile);
    try std.testing.expectEqual(@as(u32, 128), mxfp8_custom.iters);
    try std.testing.expectEqual(@as(u32, 256), mxfp8_custom.nsamples);
    try std.testing.expectEqual(@as(u32, 2048), mxfp8_custom.seqlen);

    const nvfp4_custom = defaultCalibrationFor(.custom, .nvfp4);
    try std.testing.expectEqual(QualityProfile.custom, nvfp4_custom.profile);
    try std.testing.expectEqual(@as(u32, 500), nvfp4_custom.iters);
    try std.testing.expectEqual(@as(u32, 256), nvfp4_custom.nsamples);
    try std.testing.expectEqual(@as(u32, 2048), nvfp4_custom.seqlen);
}

test "resolveCalibrationFromOptions uses profile defaults and seed" {
    const options = ConvertOptions{
        .scheme = .mxfp8,
        .calibration_profile = .good,
        .calibration_seed = 1234,
    };
    const resolved = resolveCalibrationFromOptions(options, .mxfp8);
    try std.testing.expectEqual(QualityProfile.good, resolved.profile);
    try std.testing.expectEqual(@as(u32, 64), resolved.iters);
    try std.testing.expectEqual(@as(u32, 256), resolved.nsamples);
    try std.testing.expectEqual(@as(u32, 2048), resolved.seqlen);
    try std.testing.expectEqual(@as(u64, 1234), resolved.seed);
}

test "resolveCalibrationFromOptions honors explicit overrides" {
    const options = ConvertOptions{
        .scheme = .nvfp4,
        .calibration_profile = .custom,
        .calibration_seed = 9876,
        .calibration_iters = 12,
        .calibration_nsamples = 34,
        .calibration_seqlen = 56,
        .calibration_batch_size = 2,
        .calibration_nblocks = 3,
    };
    const resolved = resolveCalibrationFromOptions(options, .nvfp4);
    try std.testing.expectEqual(QualityProfile.custom, resolved.profile);
    try std.testing.expectEqual(@as(u32, 12), resolved.iters);
    try std.testing.expectEqual(@as(u32, 34), resolved.nsamples);
    try std.testing.expectEqual(@as(u32, 56), resolved.seqlen);
    try std.testing.expectEqual(@as(u32, 2), resolved.batch_size);
    try std.testing.expectEqual(@as(u32, 3), resolved.nblocks);
    try std.testing.expectEqual(@as(u64, 9876), resolved.seed);
}

test "rewriteNvfp4Config writes canonical quant contract fields" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const dir_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(dir_path);
    const config_path = try std.fs.path.join(allocator, &.{ dir_path, "config.json" });
    defer allocator.free(config_path);
    {
        var file = try std.fs.cwd().createFile(config_path, .{});
        defer file.close();
        try file.writeAll("{\"model_type\":\"qwen\",\"hidden_size\":1024,\"quantization\":{\"group_size\":64,\"bits\":4}}");
    }

    try nvfp4_converter.rewriteConfigToCanonical(allocator, dir_path);

    var parsed = try parseConfigAtPath(allocator, config_path);
    defer parsed.deinit();
    try std.testing.expect(parsed.value == .object);
    const obj = parsed.value.object;
    const quant = objectFieldAsObject(obj, "quantization").?;
    try std.testing.expectEqual(@as(i64, 32), objectFieldAsInt(quant, "group_size").?);
    try std.testing.expectEqual(@as(i64, 4), objectFieldAsInt(quant, "bits").?);

    const qcfg = objectFieldAsObject(obj, "quantization_config").?;
    try std.testing.expectEqualStrings("nvfp4", objectFieldAsString(qcfg, "quant_method").?);
    try std.testing.expectEqual(@as(i64, supported_quant_contract_version), objectFieldAsInt(qcfg, "quant_contract_version").?);
}

test "validateCanonicalOutput accepts canonical mxfp8 artifact" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const dir_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(dir_path);

    const config_path = try std.fs.path.join(allocator, &.{ dir_path, "config.json" });
    defer allocator.free(config_path);
    {
        var file = try std.fs.cwd().createFile(config_path, .{});
        defer file.close();
        try file.writeAll("{\"quantization_config\":{\"quant_method\":\"mxfp8\",\"quant_type\":\"mxfp8\",\"fmt\":\"e4m3\",\"scale_fmt\":\"e8m0\",\"block_size\":32,\"quant_contract_version\":1}}");
    }

    var builder = safetensors.Builder.init(allocator);
    defer builder.deinit();
    var weight: [32]u8 = [_]u8{1} ** 32;
    var scale: [1]u8 = [_]u8{127};
    try builder.addTensor("layer.weight", .f8_e4m3, &[_]usize{ 1, 32 }, &weight);
    try builder.addTensor("layer.weight_block_scale", .u8, &[_]usize{ 1, 1 }, &scale);
    try builder.save(dir_path, "model.safetensors");

    try validateCanonicalOutput(allocator, dir_path, .mxfp8);
}

test "validateCanonicalOutput accepts canonical nvfp4 artifact" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const dir_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(dir_path);

    const config_path = try std.fs.path.join(allocator, &.{ dir_path, "config.json" });
    defer allocator.free(config_path);
    {
        var file = try std.fs.cwd().createFile(config_path, .{});
        defer file.close();
        try file.writeAll("{\"quantization\":{\"group_size\":32,\"bits\":4},\"quantization_config\":{\"quant_method\":\"nvfp4\",\"quant_type\":\"nvfp4\",\"bits\":4,\"group_size\":32,\"quant_contract_version\":1}}");
    }

    var builder = safetensors.Builder.init(allocator);
    defer builder.deinit();
    var qweight: [16]u8 = [_]u8{0} ** 16;
    var scales: [2]u8 = .{ 0x00, 0x3C };
    var biases: [2]u8 = .{ 0x00, 0x00 };
    try builder.addTensor("layer.weight", .grouped_affine_u4, &[_]usize{ 1, 4 }, &qweight);
    try builder.addTensor("layer.scales", .bf16, &[_]usize{ 1, 1 }, &scales);
    try builder.addTensor("layer.biases", .bf16, &[_]usize{ 1, 1 }, &biases);
    try builder.save(dir_path, "model.safetensors");

    try validateCanonicalOutput(allocator, dir_path, .nvfp4);
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
