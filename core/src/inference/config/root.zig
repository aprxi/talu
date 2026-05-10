//! Inference configuration surface.
//!
//! Re-exports model generation defaults used by local inference setup, and owns
//! inference-specific preprocessor config parsing.

const generation = @import("../../models/config/generation.zig");

pub const GenerationConfig = generation.GenerationConfig;
pub const loadGenerationConfig = generation.loadGenerationConfig;
pub const applyChatTemplate = generation.applyChatTemplate;
pub const getChatTemplateSource = generation.getChatTemplateSource;
pub const isEosToken = generation.isEosToken;
pub const addEosTokenId = generation.addEosTokenId;

const preprocessor = @import("preprocessor.zig");
pub const PreprocessorConfig = preprocessor.PreprocessorConfig;
pub const loadPreprocessorConfig = preprocessor.loadPreprocessorConfig;
