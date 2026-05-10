//! Protocol Module - Format adapters for public API projections.
//!
//! This module contains adapters that convert between talu's internal
//! Item-based Conversation format and HTTP API formats.
//!
//! # Architecture
//!
//! ```
//! core/src/responses/conversation/
//!     # Pure item-based conversation model
//!     └── Conversation       # Source of truth for conversation history
//!
//! core/src/responses/protocol/  # Format conversion for API projections
//!     ├── chat_completions.zig  # Items → OpenAI Chat Completions format
//!     └── openai_responses.zig  # Items → OpenAI Responses format
//! ```
//!
//! # Design Principle
//!
//! The conversation module should have NO knowledge of HTTP projection formats.
//! All format conversion logic lives here in the protocol module.
//!
//! This separation ensures:
//!   1. Single source of truth (Conversation)
//!   2. No dual code paths for projection/serialization
//!   3. Adding/adjusting API projections without touching core data model

pub const chat_completions = @import("chat_completions.zig");
pub const openai_responses = @import("openai_responses.zig");
