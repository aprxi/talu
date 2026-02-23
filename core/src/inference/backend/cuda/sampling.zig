//! CUDA backend sampling module.
//!
//! Sampling is host-side inference orchestration logic shared across backends.
//! CUDA backend reuses the production implementation while CUDA compute remains
//! in backend engine + compute layers.

const shared_sampling = @import("../cpu/sampling.zig");

pub const SamplingStrategy = shared_sampling.SamplingStrategy;
pub const LogitBiasEntry = shared_sampling.LogitBiasEntry;
pub const SamplingConfig = shared_sampling.SamplingConfig;
pub const Workspace = shared_sampling.Workspace;
pub const Sampler = shared_sampling.Sampler;
pub const sample = shared_sampling.sample;
