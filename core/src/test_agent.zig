//! Test root for agent submodules that do not import router/inference.
//! agent.zig, loop.zig, capi_bridge.zig are excluded (router → inference).

comptime {
    _ = @import("agent/tool.zig");
    _ = @import("agent/context.zig");
    _ = @import("agent/bus.zig");
    _ = @import("agent/compaction.zig");
    _ = @import("agent/tools/root.zig");
    _ = @import("agent/memory/root.zig");
    _ = @import("agent/fs/root.zig");
    _ = @import("agent/shell/root.zig");
}
