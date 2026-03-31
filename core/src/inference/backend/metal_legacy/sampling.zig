//! Metal backend sampling surface.
//!
//! Deliberately aliases CPU sampling exports for now to keep backend module
//! contracts explicit and symmetric while behavior remains shared.

const cpu_sampling = @import("../cpu/sampling.zig");

pub const SamplingStrategy = cpu_sampling.SamplingStrategy;
pub const LogitBiasEntry = cpu_sampling.LogitBiasEntry;
pub const SamplingConfig = cpu_sampling.SamplingConfig;
pub const Workspace = cpu_sampling.Workspace;
pub const Sampler = cpu_sampling.Sampler;
