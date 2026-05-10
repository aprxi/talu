//! Inference sampling policy module.
//!
//! Sampling request configuration is backend-neutral. The concrete production
//! sampler implementation is currently shared from the CPU backend module.

const std = @import("std");

pub const contracts = @import("sampling/contracts.zig");
const cpu_sampling = @import("backend/cpu/sampling.zig");

pub const Sampler = cpu_sampling.Sampler;
pub const SamplingConfig = contracts.SamplingConfig;
pub const SamplingStrategy = contracts.SamplingStrategy;
pub const Workspace = cpu_sampling.Workspace;
pub const LogitBiasEntry = contracts.LogitBiasEntry;
pub const sample = cpu_sampling.sample;

test "sampling facade preserves neutral config and concrete sampler exports" {
    try std.testing.expect(SamplingConfig == contracts.SamplingConfig);
    try std.testing.expect(LogitBiasEntry == contracts.LogitBiasEntry);
    try std.testing.expect(@hasDecl(@This(), "Sampler"));
    try std.testing.expect(@hasDecl(@This(), "Workspace"));
    try std.testing.expect(@hasDecl(@This(), "sample"));
}
