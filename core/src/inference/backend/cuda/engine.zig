//! CUDA backend engine (Phase 1 stub).
//!
//! This implements the backend contract while returning explicit typed errors
//! for unimplemented execution methods.

const std = @import("std");
const models = @import("../../../models/root.zig");
const contract = @import("../contract.zig");
const cpu_vision = @import("../cpu/vision/root.zig");
const compute = @import("../../../compute/root.zig");
const log = @import("../../../log.zig");

const LoadedModel = models.LoadedModel;

pub const CudaBackend = struct {
    pub const capabilities: contract.Capabilities = .{
        .vision_prefill = true,
        .decode_batch = true,
        .decode_streaming = false,
        .embedding = false,
        .warmup = false,
    };

    pub const PrefillVisionInput = cpu_vision.PrefillVisionInput;

    allocator: std.mem.Allocator,
    loaded: *LoadedModel,
    device: compute.cuda.Device,
    d_model: usize,
    vocab_size: usize,
    max_batch_size: usize = 1,
    slot_in_use: bool = false,
    slot_position: usize = 0,

    pub fn init(allocator: std.mem.Allocator, loaded: *LoadedModel) !CudaBackend {
        var device = try compute.cuda.Device.init();
        errdefer device.deinit();

        log.info("inference", "CUDA device ready", .{ .name = device.name() });

        return .{
            .allocator = allocator,
            .loaded = loaded,
            .device = device,
            .d_model = @intCast(loaded.config.d_model),
            .vocab_size = @intCast(loaded.config.vocab_size),
        };
    }

    pub fn deinit(self: *CudaBackend) void {
        self.device.deinit();
        self.* = undefined;
    }

    pub fn maxBatchSize(self: *const CudaBackend) usize {
        return self.max_batch_size;
    }

    pub fn vocabSize(self: *const CudaBackend) usize {
        return self.vocab_size;
    }

    pub fn prefill(self: *CudaBackend, tokens: []const u32, logits_out: []f32) !void {
        _ = self;
        _ = tokens;
        _ = logits_out;
        return error.CudaNotImplemented;
    }

    pub fn decode(self: *CudaBackend, token: u32, position: usize, logits_out: []f32) !void {
        _ = self;
        _ = token;
        _ = position;
        _ = logits_out;
        return error.CudaNotImplemented;
    }

    pub fn allocSlot(self: *CudaBackend) ?usize {
        if (self.slot_in_use) return null;
        self.slot_in_use = true;
        self.slot_position = 0;
        return 0;
    }

    pub fn freeSlot(self: *CudaBackend, slot_index: usize) void {
        if (slot_index != 0) return;
        self.slot_in_use = false;
        self.slot_position = 0;
    }

    pub fn resetSlot(self: *CudaBackend, slot_index: usize) void {
        if (slot_index != 0) return;
        self.slot_position = 0;
    }

    pub fn getPosition(self: *const CudaBackend, slot_index: usize) usize {
        if (slot_index != 0) return 0;
        return self.slot_position;
    }

    pub fn prefillSlot(
        self: *CudaBackend,
        slot_index: usize,
        tokens: []const u32,
        logits_out: []f32,
    ) !void {
        _ = logits_out;
        if (!self.slot_in_use or slot_index != 0) return error.InvalidArgument;
        self.slot_position = tokens.len;
        return error.CudaNotImplemented;
    }

    pub fn prefillSlotWithVision(
        self: *CudaBackend,
        slot_index: usize,
        tokens: []const u32,
        vision_input: ?*const PrefillVisionInput,
        logits_out: []f32,
    ) !void {
        _ = vision_input;
        return self.prefillSlot(slot_index, tokens, logits_out);
    }

    pub fn decodeBatch(
        self: *CudaBackend,
        requests: []const contract.DecodeRequest,
        results: []contract.DecodeResult,
    ) !void {
        _ = self;
        _ = requests;
        _ = results;
        return error.CudaNotImplemented;
    }
};

test "allocSlot allows only a single slot in stub backend" {
    if (compute.cuda.probeRuntime() != .available) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var loaded: LoadedModel = undefined;
    loaded.config = std.mem.zeroes(@TypeOf(loaded.config));
    loaded.config.vocab_size = 32;
    loaded.config.d_model = 16;

    var backend = try CudaBackend.init(allocator, &loaded);
    defer backend.deinit();

    try std.testing.expectEqual(@as(?usize, 0), backend.allocSlot());
    try std.testing.expectEqual(@as(?usize, null), backend.allocSlot());
}

test "prefill returns typed not-implemented error" {
    if (compute.cuda.probeRuntime() != .available) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var loaded: LoadedModel = undefined;
    loaded.config = std.mem.zeroes(@TypeOf(loaded.config));
    loaded.config.vocab_size = 8;
    loaded.config.d_model = 4;

    var backend = try CudaBackend.init(allocator, &loaded);
    defer backend.deinit();

    var logits: [8]f32 = undefined;
    try std.testing.expectError(error.CudaNotImplemented, backend.prefill(&.{ 1, 2 }, &logits));
}
