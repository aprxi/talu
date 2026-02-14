//! Blob subsystem for content-addressable storage and offline GC.

pub const store = @import("store.zig");
pub const gc = @import("gc.zig");

pub const BlobStore = store.BlobStore;
pub const BlobRef = store.BlobRef;
pub const SweepStats = gc.SweepStats;
pub const SweepOptions = gc.SweepOptions;
