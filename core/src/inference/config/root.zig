//! Inference configuration surface.
//!
//! Owns generation-time configuration parsing used by router/session/tokenizer.

const generation = @import("generation.zig");

pub const GenerationConfig = generation.GenerationConfig;
pub const loadGenerationConfig = generation.loadGenerationConfig;
pub const applyChatTemplate = generation.applyChatTemplate;
pub const getChatTemplateSource = generation.getChatTemplateSource;
pub const isEosToken = generation.isEosToken;
pub const addEosTokenId = generation.addEosTokenId;
