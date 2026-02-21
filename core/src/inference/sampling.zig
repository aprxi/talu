//! Inference sampling policy module.
//!
//! Implementation currently lives in CPU backend path; this wrapper keeps the
//! public inference surface backend-neutral by module path.

const cpu_sampling = @import("backend/cpu/sampling.zig");

pub const Sampler = cpu_sampling.Sampler;
pub const SamplingConfig = cpu_sampling.SamplingConfig;
pub const SamplingStrategy = cpu_sampling.SamplingStrategy;
pub const Workspace = cpu_sampling.Workspace;
pub const LogitBiasEntry = cpu_sampling.LogitBiasEntry;
pub const sample = cpu_sampling.sample;

