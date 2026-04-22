//! Agent runtime primitives retained by the slim server build.
//!
//! This repo keeps the workspace/runtime tools and policy surface used by the
//! `/v1/agent/**` endpoints, but does not keep the higher-level agent loop or
//! backend-driven orchestration.

pub const tool = @import("tool.zig");
pub const tools = @import("tools/root.zig");
pub const fs = @import("fs/root.zig");
pub const shell = @import("shell/root.zig");
pub const process = @import("process/root.zig");
pub const policy = @import("policy/root.zig");
pub const sandbox = @import("sandbox/root.zig");

// Re-export primary types at module level

// Tool types
pub const Tool = tool.Tool;
pub const ToolResult = tool.ToolResult;
pub const ToolRegistry = tool.ToolRegistry;
pub const ToolRegistryError = tool.ToolRegistryError;
