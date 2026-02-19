//! Inference - sampling, scheduling, loading, and backend execution.
//!
//! This module is the inference boundary used by router/bindings:
//! - `types` - generation request/result types
//! - `sampling` - token sampling policies
//! - `scheduler` - continuous batching runtime
//! - `model_loader` - model loading/architecture checks
//! - `backend` - CPU/Metal inference backends

const std = @import("std");
const json = @import("../io/json/root.zig");
const models = @import("../models/root.zig");

pub const sampling = @import("backend/cpu/sampling.zig");
pub const scheduler = @import("backend/cpu/scheduler.zig");
pub const config = @import("config/root.zig");

/// Generation request/response types.
pub const types = struct {
    /// Callback function type for streaming token output.
    /// Called with each newly generated token ID and optional user data.
    pub const TokenCallback = *const fn (token_id: u32, user_data: ?*anyopaque) void;

    /// Request configuration for single generation runs.
    pub const InferenceConfig = struct {
        max_new_tokens: usize = 32,
        sampling: sampling.SamplingConfig = .{},
        eos_token_ids: []const u32 = &.{},
        /// BOS token to prepend to input (from model config)
        bos_token_id: ?u32 = null,
        /// Optional callback for streaming output. Called after each token is sampled.
        token_callback: ?types.TokenCallback = null,
        /// User data passed to the token callback.
        callback_data: ?*anyopaque = null,
        /// Stop sequences (already tokenized). Generation stops when any sequence matches.
        /// Each inner slice is a tokenized stop sequence.
        stop_sequences: []const []const u32 = &.{},
        /// Optional stop flag for cancellation. When set to true, generation stops.
        stop_flag: ?*const std.atomic.Value(bool) = null,
    };

    /// Reason why generation stopped.
    pub const FinishReason = enum(u8) {
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

        /// Convert to C-compatible integer for C-API.
        pub fn toInt(self: types.FinishReason) u8 {
            return @intFromEnum(self);
        }
    };

    /// Full state returned by low-level `run()` APIs.
    pub const InferenceState = struct {
        tokens: []u32,
        final_logits: []f32,
        prompt_len: usize,
        generated_len: usize,
        prefill_ns: u64,
        decode_ns: u64,
        finish_reason: types.FinishReason = .eos_token,
    };
};

/// Boundary module: inference code should load models through this facade.
pub const model_loader = struct {
    // Hard-switch guardrail: inference routes model topology through models root.
    pub const LoadedModel = models.LoadedModel;
    pub const LoadOptions = models.LoadOptions;
    pub const weights = models.weights;
    pub const loadModel = models.loadModel;
    pub const loadArchitectureDefinitions = models.loadArchitectureDefinitions;
    comptime {
        if (@TypeOf(loadModel) != @TypeOf(models.loadModel)) {
            @compileError("inference.model_loader must source loadModel from models.root");
        }
    }

    /// Result of checking model architecture support.
    pub const ArchitectureCheck = struct {
        supported: bool,
        model_type_buf: [64]u8 = undefined,
        model_type_len: usize = 0,
        architecture_buf: [64]u8 = undefined,
        architecture_len: usize = 0,

        pub fn getModelType(self: *const @This()) ?[]const u8 {
            if (self.model_type_len == 0) return null;
            return self.model_type_buf[0..self.model_type_len];
        }

        pub fn getArchitecture(self: *const @This()) ?[]const u8 {
            if (self.architecture_len == 0) return null;
            return self.architecture_buf[0..self.architecture_len];
        }
    };

    /// Check if a model's architecture is supported without fully loading.
    /// Checks against static model registry metadata.
    pub fn checkArchitecture(allocator: std.mem.Allocator, config_path: []const u8) !ArchitectureCheck {
        var arch_check = ArchitectureCheck{ .supported = false };

        const config_bytes = std.fs.cwd().readFileAlloc(allocator, config_path, 256 * 1024) catch {
            // Can't read config - assume supported (might be older format)
            arch_check.supported = true;
            return arch_check;
        };
        defer allocator.free(config_bytes);

        const parsed_config = json.parseValue(allocator, config_bytes, .{ .max_size_bytes = 256 * 1024 }) catch {
            // Can't parse - assume supported
            arch_check.supported = true;
            return arch_check;
        };
        defer parsed_config.deinit();

        const obj = switch (parsed_config.value) {
            .object => |o| o,
            else => {
                arch_check.supported = true;
                return arch_check;
            },
        };

        if (obj.get("model_type")) |v| {
            if (v == .string) {
                const model_type = v.string;
                const len = @min(model_type.len, arch_check.model_type_buf.len);
                @memcpy(arch_check.model_type_buf[0..len], model_type[0..len]);
                arch_check.model_type_len = len;
            }
        }

        if (obj.get("architectures")) |v| {
            if (v == .array and v.array.items.len > 0) {
                const first = v.array.items[0];
                if (first == .string) {
                    const architecture_name = first.string;
                    const len = @min(architecture_name.len, arch_check.architecture_buf.len);
                    @memcpy(arch_check.architecture_buf[0..len], architecture_name[0..len]);
                    arch_check.architecture_len = len;
                }
            }
        }

        if (arch_check.getModelType()) |model_type| {
            arch_check.supported = models.isSupportedModelType(model_type);
            if (arch_check.supported and arch_check.architecture_len == 0) {
                if (models.detectByModelType(model_type)) |entry| {
                    const len = @min(entry.id.len, arch_check.architecture_buf.len);
                    @memcpy(arch_check.architecture_buf[0..len], entry.id[0..len]);
                    arch_check.architecture_len = len;
                }
            }
            return arch_check;
        }

        // No model_type found - assume supported (older models)
        arch_check.supported = true;
        return arch_check;
    }

    /// Check model architecture from a model directory.
    pub fn checkArchitectureFromDir(allocator: std.mem.Allocator, model_dir: []const u8) !ArchitectureCheck {
        const config_path = try std.fs.path.join(allocator, &.{ model_dir, "config.json" });
        defer allocator.free(config_path);
        return checkArchitecture(allocator, config_path);
    }
};

/// Pooling strategy for embedding extraction.
pub const PoolingStrategy = @import("backend/contract.zig").PoolingStrategy;
pub const pooling = struct {
    pub const PoolingStrategy = @import("backend/contract.zig").PoolingStrategy;
};

// Re-export common generation types
pub const InferenceConfig = types.InferenceConfig;
pub const InferenceState = types.InferenceState;
pub const TokenCallback = types.TokenCallback;
pub const FinishReason = types.FinishReason;

pub const Sampler = sampling.Sampler;
pub const SamplingConfig = sampling.SamplingConfig;
pub const SamplingStrategy = sampling.SamplingStrategy;

pub const Scheduler = scheduler.Scheduler;
pub const SchedulerConfig = scheduler.SchedulerConfig;
pub const RequestState = scheduler.RequestState;
pub const TokenEvent = scheduler.TokenEvent;
pub const Request = scheduler.Request;
pub const SchedulerFinishReason = scheduler.FinishReason;
pub const GenerateSyncResult = scheduler.Scheduler.GenerateSyncResult;
pub const SchedulerSubmitOptions = scheduler.Scheduler.SubmitOptions;

// Re-export sampling behavioral types so check_coverage.sh --integration can verify test coverage
pub const SamplingWorkspace = sampling.Workspace;

// =============================================================================
// Internal API (for core/src/ only)
// =============================================================================

/// Executor - LayerOp bytecode execution
pub const executor = @import("backend/cpu/executor/root.zig");

/// Backend implementations
pub const backend = struct {
    pub const cpu = @import("backend/cpu/root.zig");
    pub const metal = @import("backend/metal/root.zig");
    pub const cpu_executor = @import("backend/cpu/executor/root.zig");

    /// CPU block kernels - weight types and block building
    pub const block_kernels = cpu_executor.weights;

    /// CPU kernel implementations
    pub const kernels = struct {
        pub const moe = @import("backend/cpu/kernels/moe.zig");
        pub const attention = @import("backend/cpu/kernels/attention.zig");
        pub const ffn = @import("backend/cpu/kernels/ffn.zig");
        pub const kv_cache = @import("backend/cpu/kernels/kv_cache.zig");
    };

    // Re-export inference behavioral types so check_coverage.sh --integration can verify test coverage
    pub const FusedCpuBackend = cpu.FusedCpuBackend;
    pub const MultiHeadAttention = block_kernels.MultiHeadAttention;
    pub const SwiGLU = block_kernels.SwiGLU;
    pub const FfnLayer = block_kernels.FfnLayer;
    pub const AttnTemp = block_kernels.AttnTemp;
    pub const AttnCache = block_kernels.AttnCache;
    pub const FfnScratch = block_kernels.FfnScratch;
    pub const ScratchBuffer = block_kernels.ScratchBuffer;
    pub const TransformerBlock = block_kernels.TransformerBlock;
};

test "ArchitectureCheck.getModelType returns model type when present" {
    var check = model_loader.ArchitectureCheck{ .supported = true };
    const model_type = "llama";
    @memcpy(check.model_type_buf[0..model_type.len], model_type);
    check.model_type_len = model_type.len;

    const result = check.getModelType();
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("llama", result.?);
}

test "ArchitectureCheck.getModelType returns null when not present" {
    const check = model_loader.ArchitectureCheck{ .supported = true };
    try std.testing.expect(check.getModelType() == null);
}

test "ArchitectureCheck.getArchitecture returns architecture when present" {
    var check = model_loader.ArchitectureCheck{ .supported = true };
    const arch = "LlamaForCausalLM";
    @memcpy(check.architecture_buf[0..arch.len], arch);
    check.architecture_len = arch.len;

    const result = check.getArchitecture();
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("LlamaForCausalLM", result.?);
}

test "ArchitectureCheck.getArchitecture returns null when not present" {
    const check = model_loader.ArchitectureCheck{ .supported = true };
    try std.testing.expect(check.getArchitecture() == null);
}
