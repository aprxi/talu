//! Attention operator family root.

pub const prefill = @import("prefill.zig");
pub const decode = @import("decode.zig");
pub const workspace = @import("workspace.zig");

pub const runAttentionMixerStep = prefill.runAttentionMixerStep;
pub const runAttentionMixerPrefillBatchedNoQueryGate = prefill.runAttentionMixerPrefillBatchedNoQueryGate;
pub const runAttentionMixerPrefillBatchedWithQueryGate = prefill.runAttentionMixerPrefillBatchedWithQueryGate;
pub const runBatchedDecodeAttentionMixer = decode.runBatchedDecodeAttentionMixer;
pub const attentionFallbackUsesCache = decode.attentionFallbackUsesCache;
pub const ensureAttnScoresWorkspace = workspace.ensureAttnScoresWorkspace;
pub const ensureAttnU16Workspace = workspace.ensureAttnU16Workspace;
