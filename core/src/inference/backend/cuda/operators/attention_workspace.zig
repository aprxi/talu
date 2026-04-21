//! Attention workspace facade for the CUDA inference backend.

const workspace = @import("attention/workspace.zig");

pub const ensureAttnScoresWorkspace = workspace.ensureAttnScoresWorkspace;
pub const ensureAttnU16Workspace = workspace.ensureAttnU16Workspace;
