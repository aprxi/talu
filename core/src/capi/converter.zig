//! C API for Model Conversion and Inspection
//!
//! Exposes model conversion and introspection functionality to Python/C clients.
//! Maps to Python's talu/converter/ module.
//!
//! ## Scheme-Based API
//!
//! The converter uses a single `scheme` parameter that encodes all conversion
//! settings. This eliminates invalid parameter combinations and provides a
//! simple, self-documenting API.

const std = @import("std");
const scheme_mod = @import("../converter/scheme.zig");
const io = @import("../io/root.zig");
const graph_config = @import("../graph/config/root.zig");
const capi_error = @import("error.zig");
const error_codes = @import("error_codes.zig");
const xray = @import("../xray/root.zig");
const execution_plan = xray.execution_plan;

// Import session for path resolution and tokenizer for memory utils
const session_mod = @import("session.zig");
const tokenizer_mod = @import("tokenizer.zig");
const talu_resolve_model_path = session_mod.talu_resolve_model_path;
const talu_text_free = tokenizer_mod.talu_text_free;

// =============================================================================
// Re-exported Types (from scheme module)
// =============================================================================

/// Unified quantization scheme.
/// See scheme_mod.Scheme for full documentation.
pub const Scheme = scheme_mod.Scheme;

/// Conversion method (derived from scheme).
pub const Method = scheme_mod.Method;

/// Maximum number of override rules.
pub const MAX_OVERRIDES = scheme_mod.MAX_OVERRIDES;

/// Override rule for per-tensor quantization.
pub const OverrideRule = scheme_mod.OverrideRule;

/// Conversion options.
pub const ConvertOptions = scheme_mod.ConvertOptions;

/// Progress callback for conversion operations.
/// Parameters: (current, total, message, user_data)
pub const CProgressCallback = scheme_mod.CProgressCallback;

// =============================================================================
// C API Types
// =============================================================================

/// Result from conversion (C-compatible with null-terminated strings).
pub const ConvertResult = extern struct {
    output_path: ?[*:0]const u8 = null,
    error_msg: ?[*:0]const u8 = null,
    success: bool = false,
};

/// Quantization method enum (matches tensor.QuantMethod)
pub const QuantMethodEnum = enum(i32) {
    none = 0,
    gaffine = 1,
    mxfp4 = 2,
    native = 3,
};

/// Model information returned by describe.
pub const ModelInfo = extern struct {
    // Core architecture
    vocab_size: i32,
    hidden_size: i32,
    num_layers: i32,
    num_heads: i32,
    num_kv_heads: i32,
    intermediate_size: i32,
    max_seq_len: i32,
    head_dim: i32,

    // RoPE parameters
    rope_theta: f32,
    norm_eps: f32,

    // Quantization
    quant_bits: i32,
    quant_group_size: i32,
    quant_method: QuantMethodEnum,

    // Architecture info (null-terminated strings, caller must free)
    model_type: ?[*:0]u8,
    architecture: ?[*:0]u8,

    // Flags
    tie_word_embeddings: bool,
    use_gelu: bool,

    // MoE
    num_experts: i32,
    experts_per_token: i32,

    // Error (null on success)
    error_msg: ?[*:0]const u8,
};

const DescribeError = error{
    OutOfMemory,
    ReadConfigFailed,
    InvalidJson,
    MissingField,
    InvalidValue,
    LoadConfigFailed,
};

// =============================================================================
// Internal State
// =============================================================================

const allocator = std.heap.c_allocator;

// =============================================================================
// Conversion API
// =============================================================================

/// Creates a conversion error result with both capi_error and result.error_msg set.
fn convertError(err: anyerror, comptime fmt: []const u8, args: anytype) ConvertResult {
    capi_error.setError(err, fmt, args);
    return .{ .error_msg = convertErrorToString(err) };
}

/// Creates a conversion error result for a static message.
fn convertArgError(msg: []const u8) ConvertResult {
    capi_error.setError(error.InvalidArgument, "{s}", .{msg});
    return .{ .error_msg = allocator.dupeZ(u8, msg) catch null };
}

/// Convert a model to a quantized format.
pub export fn talu_convert(
    model_path: ?[*:0]const u8,
    output_dir: ?[*:0]const u8,
    options: ?*const ConvertOptions,
) callconv(.c) ConvertResult {
    capi_error.clearError();
    const model_path_cstr = model_path orelse return convertArgError("model_path is required");
    const output_dir_cstr = output_dir orelse return convertArgError("output_dir is required");
    const convert_options = options orelse &ConvertOptions{};

    const resolved_path = io.repository.resolveModelPath(
        allocator,
        std.mem.span(model_path_cstr),
        .{ .offline = convert_options.offline, .progress = convert_options.progressContext() },
    ) catch |err| return convertError(err, "Model path resolution failed: {s}", .{@errorName(err)});
    defer allocator.free(resolved_path);

    const result = scheme_mod.convert(allocator, resolved_path, std.mem.span(output_dir_cstr), convert_options.*);
    if (result.err) |err| return convertError(err, "Convert failed: {s}", .{@errorName(err)});

    const output_path = result.output_path orelse return convertArgError("No output path returned");
    const output_path_cstr = allocator.dupeZ(u8, output_path) catch {
        allocator.free(output_path);
        return convertArgError("Out of memory");
    };
    allocator.free(output_path);
    return .{ .success = true, .output_path = output_path_cstr.ptr };
}

/// Free a string returned by talu_convert.
pub export fn talu_convert_free_string(string_ptr: ?[*:0]const u8) callconv(.c) void {
    if (string_ptr) |ptr| {
        const length = std.mem.len(ptr);
        allocator.free(ptr[0 .. length + 1]);
    }
}


/// Get all available schemes as a JSON string: {"canonical": ["alias1", ...], ...}.
///
/// Caller owns the returned string. Call talu_convert_free_schemes() when done.
pub export fn talu_convert_schemes(out_schemes: *?[*:0]const u8) callconv(.c) i32 {
    capi_error.clearError();
    out_schemes.* = null;
    const result = allocator.dupeZ(u8, Scheme.all_schemes_json) catch {
        capi_error.setError(error.OutOfMemory, "Schemes allocation failed", .{});
        return @intFromEnum(error_codes.errorToCode(error.OutOfMemory));
    };
    out_schemes.* = result.ptr;
    return 0;
}

/// Parse a scheme string (canonical or alias) into its enum integer.
/// Returns -1 if the scheme name is invalid.
pub export fn talu_convert_parse_scheme(name: [*:0]const u8) callconv(.c) c_int {
    const s = std.mem.span(name);
    if (scheme_mod.Scheme.fromString(s)) |scheme| {
        return @intCast(@intFromEnum(scheme));
    }
    return -1;
}

// =============================================================================
// Model Inspection API
// =============================================================================

/// Get model information from a model directory.
/// Caller must free model_type and architecture strings with talu_model_info_free.
pub export fn talu_describe(model_path: [*:0]const u8) callconv(.c) ModelInfo {
    var resolved_path_ptr: ?[*:0]u8 = null;
    if (talu_resolve_model_path(model_path, &resolved_path_ptr) != 0) {
        return describeErrorResult("Failed to resolve model path");
    }
    if (resolved_path_ptr == null) {
        return describeErrorResult("Failed to resolve model path");
    }
    const resolved_path = std.mem.span(resolved_path_ptr.?);
    defer talu_text_free(resolved_path_ptr);

    const info = describeFromResolvedPath(resolved_path) catch |err| {
        return switch (err) {
            error.OutOfMemory => describeErrorResult("OutOfMemory"),
            error.ReadConfigFailed => describeErrorResult("Failed to read config.json"),
            error.InvalidJson => describeErrorResult("Invalid JSON in config.json"),
            error.MissingField => describeErrorResult("Missing required field in config.json"),
            error.InvalidValue => describeErrorResult("Invalid value in config.json"),
            error.LoadConfigFailed => describeErrorResult("Failed to load config"),
        };
    };
    return info;
}

/// Free a C string allocated by this module.
fn freeCString(cstr: [*:0]const u8) void {
    const slice = std.mem.sliceTo(cstr, 0);
    allocator.free(slice.ptr[0 .. slice.len + 1]);
}

/// Free model info strings (model_type and architecture).
pub export fn talu_model_info_free(info: *ModelInfo) callconv(.c) void {
    if (info.model_type) |s| {
        freeCString(s);
        info.model_type = null;
    }
    if (info.architecture) |s| {
        freeCString(s);
        info.architecture = null;
    }
}

// =============================================================================
// Error Helpers
// =============================================================================

// Error messages are kept generic - bindings add their own context-specific hints
// (e.g., CLI adds "--force", Python adds "force=True")
fn convertErrorToString(err: anyerror) ?[*:0]const u8 {
    const msg: []const u8 = switch (err) {
        error.FileNotFound, error.WeightsNotFound, error.ModelNotFound => "Model not found",
        error.ModelNotCached => "Model not found in cache",
        error.InvalidModelId => "Invalid model ID format",
        error.AlreadyQuantized => "Model is already quantized",
        error.OutputExists => "Output directory already exists",
        error.OutOfMemory => "Out of memory",
        error.AccessDenied => "Access denied",
        error.UnsupportedFormat => "Scheme not yet implemented",
        error.InvalidArgument => "Invalid argument",
        error.InvalidFile => "Invalid or corrupted model file",
        error.ResourceExhausted => "Not enough memory to load model",
        error.LoadConfigFailed => "Failed to load model config",
        error.ConfigNotFound => "Model config not found",
        error.TokenizerNotFound => "Tokenizer not found",
        else => {
            // Include the error name for debugging unhandled cases
            const err_name = @errorName(err);
            var buf: [128]u8 = undefined;
            const formatted = std.fmt.bufPrintZ(&buf, "Conversion failed: {s}", .{err_name}) catch "Conversion failed";
            const result = allocator.dupeZ(u8, std.mem.sliceTo(formatted, 0)) catch return null;
            return result.ptr;
        },
    };
    const result = allocator.dupeZ(u8, msg) catch return null;
    return result.ptr;
}

fn describeErrorResult(msg: [*:0]const u8) ModelInfo {
    var info = std.mem.zeroes(ModelInfo);
    info.quant_method = .none;
    info.error_msg = msg;
    return info;
}

fn describeFromResolvedPath(resolved_path: []const u8) DescribeError!ModelInfo {
    // Delegate to graph.config.ModelDescription which handles all the logic
    const desc = graph_config.ModelDescription.fromDir(allocator, resolved_path) catch |err| {
        return switch (err) {
            error.InvalidJson => error.InvalidJson,
            error.MissingField => error.MissingField,
            error.InvalidValue => error.InvalidValue,
            error.OutOfMemory => error.OutOfMemory,
            else => error.LoadConfigFailed,
        };
    };

    // Convert to C API struct (compatible layout, just need to convert pointers)
    var info = std.mem.zeroes(ModelInfo);
    info.vocab_size = desc.vocab_size;
    info.hidden_size = desc.hidden_size;
    info.num_layers = desc.num_layers;
    info.num_heads = desc.num_heads;
    info.num_kv_heads = desc.num_kv_heads;
    info.intermediate_size = desc.intermediate_size;
    info.max_seq_len = desc.max_seq_len;
    info.head_dim = desc.head_dim;
    info.rope_theta = desc.rope_theta;
    info.norm_eps = desc.norm_eps;
    info.quant_bits = desc.quant_bits;
    info.quant_group_size = desc.quant_group_size;
    info.quant_method = @enumFromInt(@intFromEnum(desc.quant_method));
    info.model_type = if (desc.model_type) |mt| mt.ptr else null;
    info.architecture = if (desc.architecture) |arch| arch.ptr else null;
    info.tie_word_embeddings = desc.tie_word_embeddings;
    info.use_gelu = desc.use_gelu;
    info.num_experts = desc.num_experts;
    info.experts_per_token = desc.experts_per_token;
    return info;
}

// =============================================================================
// Execution Plan API
// =============================================================================

/// Execution plan returned by talu_execution_plan.
/// Shows which kernels will be used for a model based on config.json.
pub const ExecutionPlanInfo = extern struct {
    // Kernel selections (null-terminated strings)
    matmul_kernel: [*:0]const u8,
    attention_type: [*:0]const u8,
    ffn_type: [*:0]const u8,

    // Dimensions (for reference)
    num_layers: i32,
    hidden_size: i32,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,

    // MoE
    num_experts: i32,
    experts_per_token: i32,

    // Quantization
    quant_bits: i32,
    quant_group_size: i32,

    // Flags
    uses_gqa: bool,
    uses_moe: bool,
    uses_quantization: bool,
    uses_gelu: bool,
    is_supported: bool, // Whether model type is supported by talu

    // Error (null on success)
    error_msg: ?[*:0]const u8,
};

/// Get execution plan from ModelInfo.
/// This shows which kernels will be used without loading the model.
pub export fn talu_execution_plan(info: *const ModelInfo) callconv(.c) ExecutionPlanInfo {
    if (info.error_msg != null) {
        return executionPlanError("Input ModelInfo contains error");
    }

    const quant_method: execution_plan.QuantMethod = switch (info.quant_method) {
        .none, .native => .none,
        .gaffine => .gaffine,
        .mxfp4 => .mxfp4,
    };
    const config = execution_plan.configFromDescribe(.{
        .model_type = if (info.model_type) |mt| std.mem.sliceTo(mt, 0) else null,
        .vocab_size = info.vocab_size,
        .hidden_size = info.hidden_size,
        .num_layers = info.num_layers,
        .num_heads = info.num_heads,
        .num_kv_heads = info.num_kv_heads,
        .intermediate_size = info.intermediate_size,
        .head_dim = info.head_dim,
        .quant_bits = info.quant_bits,
        .quant_group_size = info.quant_group_size,
        .quant_method = quant_method,
        .use_gelu = info.use_gelu,
        .tie_word_embeddings = info.tie_word_embeddings,
        .num_experts = info.num_experts,
        .experts_per_token = info.experts_per_token,
    });
    const plan = execution_plan.analyze(config);

    return planToInfo(plan);
}

fn planToInfo(plan: execution_plan.ExecutionPlan) ExecutionPlanInfo {
    var info = std.mem.zeroes(ExecutionPlanInfo);
    info.matmul_kernel = plan.matmul_kernel.name().ptr;
    info.attention_type = plan.attention_type.name().ptr;
    info.ffn_type = plan.ffn_type.name().ptr;
    info.num_layers = @intCast(plan.num_layers);
    info.hidden_size = @intCast(plan.hidden_size);
    info.num_heads = @intCast(plan.num_heads);
    info.num_kv_heads = @intCast(plan.num_kv_heads);
    info.head_dim = @intCast(plan.head_dim);
    info.num_experts = @intCast(plan.num_experts);
    info.experts_per_token = @intCast(plan.experts_per_token);
    info.quant_bits = plan.quant_bits;
    info.quant_group_size = plan.quant_group_size;
    info.uses_gqa = plan.uses_gqa;
    info.uses_moe = plan.uses_moe;
    info.uses_quantization = plan.uses_quantization;
    info.uses_gelu = plan.uses_gelu;
    info.is_supported = plan.is_supported;
    return info;
}

fn executionPlanError(msg: [*:0]const u8) ExecutionPlanInfo {
    var info = std.mem.zeroes(ExecutionPlanInfo);
    info.matmul_kernel = "unknown";
    info.attention_type = "unknown";
    info.ffn_type = "unknown";
    info.is_supported = false;
    info.error_msg = msg;
    return info;
}

// =============================================================================
// Fuzz Tests
// =============================================================================

test "fuzz talu_convert_parse_scheme with random strings" {
    // Fuzz the scheme parser with arbitrary byte sequences.
    // The function should never crash, only return -1 for invalid schemes.
    try std.testing.fuzz({}, struct {
        fn testOne(_: void, input: []const u8) !void {
            // Need null-terminated string for C API
            const scheme_z = std.testing.allocator.allocSentinel(u8, input.len, 0) catch return;
            defer std.testing.allocator.free(scheme_z[0 .. input.len + 1]);
            @memcpy(scheme_z[0..input.len], input);

            const result = talu_convert_parse_scheme(scheme_z.ptr);

            // Result should be either -1 (invalid) or a valid enum value
            if (result >= 0) {
                const scheme = try std.meta.intToEnum(Scheme, @as(u8, @intCast(result)));
                const parsed = Scheme.fromString(std.mem.span(scheme_z.ptr));
                try std.testing.expect(parsed != null);
                try std.testing.expectEqual(scheme, parsed.?);
            } else {
                try std.testing.expectEqual(@as(c_int, -1), result);
            }
        }
    }.testOne, .{});
}
