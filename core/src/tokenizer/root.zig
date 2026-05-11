//! Tokenizer subsystem.
//!
//! This module owns the full text->ids->text lifecycle:
//! - model backends (`bpe`, `unigram`, `wordpiece`)
//! - schema + JSON loader
//! - high-level API

const api_mod = @import("api.zig");
const offsets_mod = @import("offsets.zig");
const batch_mod = @import("batch.zig");
const handle_mod = @import("handle.zig");
const c_types_mod = @import("c_types.zig");

// =============================================================================
// Public API
// =============================================================================

/// High-level tokenizer interface.
pub const Tokenizer = api_mod.Tokenizer;
pub const TokenizerError = api_mod.TokenizerError;
pub const StreamingDecoder = api_mod.StreamingDecoder;

/// Tokenizer with model context (for C API).
pub const TokenizerHandle = handle_mod.TokenizerHandle;

/// Vocabulary access.
pub const VocabEntry = api_mod.VocabEntry;
pub const VocabResult = api_mod.VocabResult;
pub const getVocab = api_mod.getVocab;

/// Tokenize to bytes.
pub const TokenizeBytesResult = api_mod.TokenizeBytesResult;
pub const tokenizeToBytes = api_mod.tokenizeToBytes;

/// Token offset computation and encoding.
pub const offsets = offsets_mod;
pub const encode = offsets_mod.encode;

/// Batch operations.
pub const batch = batch_mod;

// =============================================================================
// Internal API
// =============================================================================

pub const pipeline = @import("pipeline.zig");
// Re-export behavioral types so check_coverage.sh --integration can verify test coverage
pub const CTokenizer = c_types_mod.Tokenizer;

// Re-export C-compatible vocabulary types for FFI
pub const VocabEntryC = c_types_mod.VocabEntry;
pub const VocabListC = c_types_mod.VocabListC;
