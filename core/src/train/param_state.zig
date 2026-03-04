//! Trainable parameter state: bundles parameter data, gradient, and optimizer state.
//!
//! Each TrainableParam represents one tensor that participates in training
//! (either a LoRA adapter matrix or a full weight).

const std = @import("std");
const grad_mod = @import("grad.zig");
const optimizer_mod = @import("optimizer.zig");

const GradTensor = grad_mod.GradTensor;
const OptimizerState = optimizer_mod.ParamState;
const Allocator = std.mem.Allocator;

/// A single trainable parameter with its gradient and optimizer state.
pub const TrainableParam = struct {
    /// Identifier for this parameter (e.g., "layer.0.self_attn.q_proj.lora_A").
    id: []const u8,
    /// Mutable view of the parameter data (f32).
    data: []f32,
    /// Gradient storage.
    grad: GradTensor,
    /// AdamW optimizer state (moments).
    opt_state: OptimizerState,

    pub fn init(allocator: Allocator, id: []const u8, data: []f32) !TrainableParam {
        const n = data.len;
        var g = try GradTensor.init(allocator, &.{n});
        errdefer g.deinit();
        var opt = try OptimizerState.init(allocator, n);
        errdefer opt.deinit();
        return .{
            .id = id,
            .data = data,
            .grad = g,
            .opt_state = opt,
        };
    }

    pub fn deinit(self: *TrainableParam) void {
        self.grad.deinit();
        self.opt_state.deinit();
        self.* = undefined;
    }
};

/// Collection of trainable parameters.
pub const TrainableParams = struct {
    params: std.ArrayListUnmanaged(TrainableParam),
    allocator: Allocator,

    pub fn init(allocator: Allocator) TrainableParams {
        return .{ .params = .{}, .allocator = allocator };
    }

    pub fn addParam(self: *TrainableParams, param: TrainableParam) !void {
        try self.params.append(self.allocator, param);
    }

    /// Zero all gradients. Called at the start of each training step.
    pub fn zeroAllGrads(self: *TrainableParams) void {
        for (self.params.items) |*p| {
            p.grad.zero();
        }
    }

    /// Find a parameter by ID.
    pub fn getParam(self: *const TrainableParams, id: []const u8) ?*TrainableParam {
        for (self.params.items) |*p| {
            if (std.mem.eql(u8, p.id, id)) return p;
        }
        return null;
    }

    /// Total number of trainable scalar parameters.
    pub fn totalParamCount(self: *const TrainableParams) usize {
        var total: usize = 0;
        for (self.params.items) |*p| {
            total += p.data.len;
        }
        return total;
    }

    /// Number of parameter groups.
    pub fn count(self: *const TrainableParams) usize {
        return self.params.items.len;
    }

    pub fn deinit(self: *TrainableParams) void {
        for (self.params.items) |*p| {
            p.deinit();
        }
        self.params.deinit(self.allocator);
        self.* = undefined;
    }
};

// =============================================================================
// Tests
// =============================================================================

test "TrainableParam init and deinit" {
    const allocator = std.testing.allocator;
    var data = [_]f32{ 1.0, 2.0, 3.0 };

    var param = try TrainableParam.init(allocator, "test", &data);
    defer param.deinit();

    try std.testing.expectEqual(@as(usize, 3), param.data.len);
    try std.testing.expectEqual(@as(usize, 3), param.grad.numElements());
}

test "TrainableParams collection" {
    const allocator = std.testing.allocator;
    var params = TrainableParams.init(allocator);
    defer params.deinit();

    var data1 = [_]f32{ 1, 2, 3 };
    var data2 = [_]f32{ 4, 5 };

    var p1 = try TrainableParam.init(allocator, "w1", &data1);
    errdefer p1.deinit();
    try params.addParam(p1);

    var p2 = try TrainableParam.init(allocator, "w2", &data2);
    errdefer p2.deinit();
    try params.addParam(p2);

    try std.testing.expectEqual(@as(usize, 2), params.count());
    try std.testing.expectEqual(@as(usize, 5), params.totalParamCount());
}

test "TrainableParams zeroAllGrads" {
    const allocator = std.testing.allocator;
    var params = TrainableParams.init(allocator);
    defer params.deinit();

    var data = [_]f32{ 1, 2 };
    var p = try TrainableParam.init(allocator, "w", &data);
    errdefer p.deinit();

    // Write to gradient
    const g = p.grad.asSliceMut();
    g[0] = 5.0;
    g[1] = 10.0;

    try params.addParam(p);
    params.zeroAllGrads();

    // Should be zeroed
    const found = params.getParam("w").?;
    for (found.grad.asSlice()) |v| {
        try std.testing.expectEqual(@as(f32, 0.0), v);
    }
}

test "TrainableParams getParam lookup" {
    const allocator = std.testing.allocator;
    var params = TrainableParams.init(allocator);
    defer params.deinit();

    var data = [_]f32{42.0};
    var p = try TrainableParam.init(allocator, "my_param", &data);
    errdefer p.deinit();
    try params.addParam(p);

    try std.testing.expect(params.getParam("my_param") != null);
    try std.testing.expect(params.getParam("other") == null);
}
