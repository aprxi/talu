//! Inference sampling policy module.
//!
//! Sampling request configuration and host-side sampler runtime are
//! backend-neutral inference policy.

const std = @import("std");

pub const contracts = @import("contracts.zig");
pub const policy = @import("policy.zig");
pub const runtime = @import("runtime.zig");

pub const SamplingConfig = contracts.SamplingConfig;
pub const SamplingStrategy = contracts.SamplingStrategy;
pub const LogitBiasEntry = contracts.LogitBiasEntry;
pub const Sampler = runtime.Sampler;
pub const Workspace = runtime.Workspace;
pub const sample = runtime.sample;

test "inference sampling facade preserves neutral config and concrete sampler exports" {
    try std.testing.expect(SamplingConfig == contracts.SamplingConfig);
    try std.testing.expect(LogitBiasEntry == contracts.LogitBiasEntry);
    try std.testing.expect(@hasDecl(@This(), "policy"));
    try std.testing.expect(@hasDecl(@This(), "Sampler"));
    try std.testing.expect(@hasDecl(@This(), "Workspace"));
    try std.testing.expect(@hasDecl(@This(), "sample"));

    var sampler_state = try Sampler.init(std.testing.allocator, 1, 4);
    defer sampler_state.deinit();
    const logits = [_]f32{ 0.0, 1.0 };
    try std.testing.expectEqual(@as(usize, 1), try sample(&sampler_state, &logits, .{}));
}
