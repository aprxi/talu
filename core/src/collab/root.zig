//! Collaboration domain root.
//!
//! This module centralizes collaboration-oriented storage semantics on top of
//! DB primitives. It intentionally keeps transport concerns out of core.

pub const types = @import("types.zig");
pub const store = @import("store.zig");
pub const resource_store = @import("resource_store.zig");
pub const crdt = @import("crdt/root.zig");

pub const SessionStore = store.SessionStore;
pub const ResourceStore = resource_store.ResourceStore;
pub const OperationEnvelope = types.OperationEnvelope;
pub const StorageLane = types.StorageLane;
pub const TextCrdt = crdt.text_engine.TextCrdt;
pub const LamportClock = crdt.clock.LamportClock;
