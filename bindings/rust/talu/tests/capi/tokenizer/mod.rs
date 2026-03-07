//! Tokenizer CAPI tests.

pub mod common;

mod added_tokens;
mod batch;
mod bpe_merges;
mod byte_fallback;
mod byte_level;
mod decode;
mod edge_cases;
mod encode;
#[cfg(feature = "fixtures")]
mod fixtures;
mod gpt2;
mod json_loading;
mod lifecycle;
mod memory;
mod offsets;
mod pipeline;
mod property;
mod stress;
mod tokenize;
mod unicode;
mod unigram;
mod vocab;
mod wordpiece;
