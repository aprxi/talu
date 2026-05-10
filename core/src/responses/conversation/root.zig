//! Conversation Module - item-based chat and response state.
//!
//! This module provides the core runtime data types used by local generation:
//!   - Item: Atomic unit (message, function_call, reasoning, etc.)
//!   - Conversation: Container for Items with serialization
//!   - Chat: Lightweight wrapper with sampling parameters

pub const chat = @import("chat.zig");
pub const reasoning_parser = @import("reasoning_parser.zig");
pub const session_id = @import("session_id.zig");

// Open Responses architecture modules
pub const items = @import("items.zig");
pub const conversation = @import("conversation.zig");

// Re-export main types
pub const Conversation = conversation.Conversation;
pub const Item = items.Item;
pub const ItemType = items.ItemType;
pub const ItemStatus = items.ItemStatus;
pub const ItemVariant = items.ItemVariant;
pub const MessageData = items.MessageData;
pub const MessageRole = items.MessageRole;
pub const FunctionCallData = items.FunctionCallData;
pub const FunctionCallOutputData = items.FunctionCallOutputData;
pub const FunctionOutputValue = items.FunctionOutputValue;
pub const ReasoningData = items.ReasoningData;
pub const ItemReferenceData = items.ItemReferenceData;
pub const ContentType = items.ContentType;
pub const ContentPart = items.ContentPart;
pub const ContentVariant = items.ContentVariant;
pub const ImageDetail = items.ImageDetail;
pub const UrlCitation = items.UrlCitation;

// Reasoning parser
pub const ReasoningParser = reasoning_parser.ReasoningParser;
pub const ReasoningFormat = reasoning_parser.ReasoningFormat;
pub const generateSessionId = session_id.generateSessionId;

// Serialization types
pub const SerializationDirection = conversation.SerializationDirection;
pub const ResponsesSerializationOptions = conversation.ResponsesSerializationOptions;

// Re-export chat types (canonical location)
pub const Chat = chat.Chat;
pub const ResolutionConfig = chat.ResolutionConfig;
