//! Protocol Module - Format adapters for external providers.
//!
//! This module contains adapters that convert between talu's internal
//! Item-based Conversation format and external API formats.
//!
//! # Architecture
//!
//! ```
//! core/src/messages/         # Pure Item-based data model
//!     └── Conversation       # Source of truth for conversation history
//!
//! core/src/router/protocol/  # Format conversion for providers
//!     ├── completions.zig    # Items → OpenAI Chat Completions format
//!     └── (responses.zig)    # Items → Responses format (if needed externally)
//! ```
//!
//! # Design Principle
//!
//! The messages module should have NO knowledge of external formats.
//! All format conversion logic lives here in the protocol module.
//!
//! This separation ensures:
//!   1. Single source of truth (Conversation)
//!   2. No dual code paths for storage/serialization
//!   3. Easy addition of new provider formats without touching core data model

pub const completions = @import("completions.zig");
pub const responses = @import("responses.zig");

// Re-export main functions for convenience
pub const serializeCompletions = completions.serialize;
pub const CompletionsOptions = completions.Options;
pub const parseResponsesInput = responses.parse;
