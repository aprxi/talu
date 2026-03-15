//! Model-owned sampling preset metadata.
//!
//! Defines recommended sampling parameters for common use cases.
//! Each model architecture registers its presets via the `Architecture`
//! struct. Consumers look up presets through `registry.samplingPresetsByName()`.
//!
//! Categories follow the Qwen3.5 recommendation structure but are
//! model-agnostic — every architecture can override the defaults.

const std = @import("std");

pub const Category = enum {
    /// Thinking mode for general tasks.
    general,
    /// Thinking mode for precise coding tasks.
    coding,
    /// Non-thinking (instruct) mode for general tasks.
    instruct,
    /// Fully reproducible, greedy-like sampling.
    deterministic,
};

pub const Params = struct {
    temperature: f32,
    top_p: f32,
    top_k: usize,
    presence_penalty: f32,
};

pub const SamplingPresets = struct {
    general: Params,
    coding: Params,
    instruct: Params,
    deterministic: Params,

    pub fn get(self: *const SamplingPresets, category: Category) Params {
        return switch (category) {
            .general => self.general,
            .coding => self.coding,
            .instruct => self.instruct,
            .deterministic => self.deterministic,
        };
    }
};

/// Default presets (Qwen3.5-recommended values). Models can override
/// by defining their own `const` and registering it in `Architecture`.
pub const default: SamplingPresets = .{
    .general = .{ .temperature = 1.0, .top_p = 0.95, .top_k = 20, .presence_penalty = 1.5 },
    .coding = .{ .temperature = 0.6, .top_p = 0.95, .top_k = 20, .presence_penalty = 0.0 },
    .instruct = .{ .temperature = 0.7, .top_p = 0.8, .top_k = 20, .presence_penalty = 1.5 },
    .deterministic = .{ .temperature = 0.0, .top_p = 1.0, .top_k = 1, .presence_penalty = 0.0 },
};

test "default presets have expected values" {
    const general = default.get(.general);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), general.temperature, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.95), general.top_p, 0.001);
    try std.testing.expectEqual(@as(usize, 20), general.top_k);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), general.presence_penalty, 0.001);

    const coding = default.get(.coding);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), coding.temperature, 0.001);

    const deterministic = default.get(.deterministic);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), deterministic.temperature, 0.001);
    try std.testing.expectEqual(@as(usize, 1), deterministic.top_k);
}
