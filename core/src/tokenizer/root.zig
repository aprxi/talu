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
pub const TokenOffset = offsets_mod.TokenOffset;
pub const computeOffsetsFromEncoding = offsets_mod.computeOffsetsFromEncoding;
pub const Encoding = offsets_mod.Encoding;
pub const encode = offsets_mod.encode;

/// Batch operations.
pub const batch = batch_mod;
pub const BatchEncodeResult = batch_mod.BatchEncodeResult;
pub const BatchEncodeContext = batch_mod.BatchEncodeContext;
pub const PaddedTensorOptions = batch_mod.PaddedTensorOptions;
pub const PaddedTensorResult = batch_mod.PaddedTensorResult;
pub const PaddingSide = batch_mod.PaddingSide;
pub const encodeBatch = batch_mod.encodeBatch;
pub const batchToPaddedTensor = batch_mod.batchToPaddedTensor;
pub const batchEncodeWorker = batch_mod.batchEncodeWorker;

// =============================================================================
// Internal API
// =============================================================================

pub const pipeline = @import("pipeline.zig");
pub const loader = @import("loader.zig");
pub const schema = @import("schema.zig");
pub const decoders = @import("decoders.zig");
pub const c_types = @import("c_types.zig");

// Tokenizer model backends.
pub const bpe = @import("bpe.zig");
pub const unigram = @import("unigram.zig");
pub const wordpiece = @import("wordpiece.zig");

// Re-export behavioral types so check_coverage.sh --integration can verify test coverage
pub const CTokenizer = c_types.Tokenizer;
pub const DecodeOptions = c_types.DecodeOptions;
pub const TokenizerEncoding = c_types.TokenizerEncoding;

// Re-export C-compatible vocabulary types for FFI
pub const VocabEntryC = c_types.VocabEntry;
pub const VocabListC = c_types.VocabListC;
