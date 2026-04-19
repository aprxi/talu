//! Protocol Module - Format adapters for public API projections.
//!
//! This module contains adapters that convert between talu's internal
//! Item-based Conversation format and HTTP API formats.
//!
//! # Architecture
//!
//! ```
//! core/src/responses/        # Pure Item-based data model
//!     └── Conversation       # Source of truth for conversation history
//!
//! core/src/router/protocol/  # Format conversion for API projections
//!     ├── completions.zig    # Items → OpenAI Chat Completions format
//!     └── responses.zig      # Items → Responses format
//! ```
//!
//! # Design Principle
//!
//! The responses module should have NO knowledge of HTTP projection formats.
//! All format conversion logic lives here in the protocol module.
//!
//! This separation ensures:
//!   1. Single source of truth (Conversation)
//!   2. No dual code paths for projection/serialization
//!   3. Adding/adjusting API projections without touching core data model

pub const completions = @import("completions.zig");
pub const responses = @import("responses.zig");
