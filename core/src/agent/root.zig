//! Agent - Agentic framework with stateful agents and inter-agent messaging.
//!
//! Provides the complete agent framework:
//!   - `tool.zig`: Tool interface (vtable) + ToolRegistry
//!   - `context.zig`: System prompt builder
//!   - `loop.zig`: Stateless generate → execute → repeat loop
//!   - `agent.zig`: Stateful Agent with compaction, retry, inbox drain
//!   - `bus.zig`: Transport-agnostic inter-agent MessageBus
//!
//! Two levels of usage:
//!   1. **Low-level** (`loop.run()`): Caller manages everything via callbacks.
//!   2. **High-level** (`Agent.prompt()`): Agent manages compaction, retry,
//!      inbox drain internally. Callers interact via prompt/heartbeat/abort.

pub const tool = @import("tool.zig");
pub const context = @import("context.zig");
pub const loop = @import("loop.zig");
pub const agent = @import("agent.zig");
pub const bus = @import("bus.zig");
pub const compaction = @import("compaction.zig");

// Re-export primary types at module level

// Tool types
pub const Tool = tool.Tool;
pub const ToolResult = tool.ToolResult;
pub const ToolRegistry = tool.ToolRegistry;
pub const ToolRegistryError = tool.ToolRegistryError;

// Context builder
pub const ContextConfig = context.ContextConfig;
pub const buildSystemPrompt = context.buildSystemPrompt;

// Low-level loop types
pub const AgentLoopConfig = loop.AgentLoopConfig;
pub const AgentLoopResult = loop.AgentLoopResult;
pub const LoopStopReason = loop.LoopStopReason;
pub const AgentEvent = loop.AgentEvent;
pub const AgentEventType = loop.AgentEventType;
pub const OnTokenFn = loop.OnTokenFn;
pub const OnEventFn = loop.OnEventFn;
pub const OnContextFn = loop.OnContextFn;
pub const ContextInfo = loop.ContextInfo;
pub const ToolConfirmation = loop.ToolConfirmation;
pub const OnBeforeToolFn = loop.OnBeforeToolFn;
pub const OnAfterToolFn = loop.OnAfterToolFn;
pub const OnSessionFn = loop.OnSessionFn;
pub const SessionEvent = loop.SessionEvent;
pub const SessionEventType = loop.SessionEventType;
pub const run = loop.run;

// High-level agent types
pub const Agent = agent.Agent;
pub const AgentConfig = agent.AgentConfig;
pub const AgentError = agent.AgentError;
pub const OnContextInjectFn = agent.OnContextInjectFn;
pub const OnEmbedFn = agent.OnEmbedFn;
pub const OnResolveDocFn = agent.OnResolveDocFn;
pub const RagConfig = agent.RagConfig;

// Compaction
pub const Turn = compaction.Turn;
pub const identifyTurns = compaction.identifyTurns;
pub const compactTurns = compaction.compactTurns;
pub const truncateOversizedItem = compaction.truncateOversizedItem;

// Inter-agent messaging
pub const MessageBus = bus.MessageBus;
pub const BusError = bus.BusError;
pub const Message = bus.Message;
pub const OnMessageNotifyFn = bus.OnMessageNotifyFn;
pub const PeerTransport = bus.PeerTransport;
pub const PeerInfo = bus.PeerInfo;
