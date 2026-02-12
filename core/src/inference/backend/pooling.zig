//! Pooling strategy for embedding extraction.

/// Pooling strategy for embedding extraction.
pub const PoolingStrategy = enum(u8) {
    /// Use last token's hidden state (default for decoder models).
    last = 0,
    /// Average all token hidden states.
    mean = 1,
    /// Use first token (CLS token for BERT-style models).
    first = 2,
};
