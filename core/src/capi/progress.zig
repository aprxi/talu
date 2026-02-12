//! Unified Progress API
//!
//! Core-owned progress reporting for all long-running operations.
//! Bindings receive structured updates and render them however they choose.
//!
//! ## Design Principles
//!
//! 1. **Core owns state**: Core decides what operations are active, their progress,
//!    labels, and when they complete. Bindings don't need to know what operation
//!    is running (download vs convert) - they just render what they're told.
//!
//! 2. **Line-based model**: Progress is reported as lines that can be added,
//!    updated, or completed. This maps naturally to progress bars (indicatif)
//!    or text output (terminals, Jupyter).
//!
//! 3. **Single callback**: One callback signature handles all operations.
//!    The callback receives a ProgressUpdate struct with all information needed
//!    to render the current state.
//!
//! ## Example Flow
//!
//! ```
//! talu_convert("org/model-name", options_with_progress_callback)
//!
//! Callback receives:
//!   add(line=0, label="Downloading", message="config.json", total=5)
//!   update(line=0, current=1, message="model.safetensors")
//!   update(line=0, current=2, message="tokenizer.json")
//!   ...
//!   complete(line=0)
//!   add(line=0, label="Converting", message="embed_tokens", total=311)
//!   update(line=0, current=1, message="model.layers.0.self_attn.q_proj")
//!   ...
//!   complete(line=0)
//! ```

const std = @import("std");

// =============================================================================
// Progress Update Types
// =============================================================================

/// Action to perform on a progress line.
pub const ProgressAction = enum(u8) {
    /// Add a new progress line. If line_id already exists, resets it.
    add = 0,
    /// Update an existing line's progress/message.
    update = 1,
    /// Mark line as complete and remove it from display.
    complete = 2,
};

/// Progress update sent to bindings.
/// Contains all information needed to render progress state.
pub const ProgressUpdate = extern struct {
    /// Which line to update (0-based integer key). Bindings maintain a map of line_id → UI element.
    line_id: u8 = 0,

    /// What action to perform on this line.
    action: ProgressAction = .update,

    /// Current progress value (items completed, bytes downloaded, etc.).
    current: u64 = 0,

    /// Total expected value. 0 means indeterminate (spinner instead of bar).
    total: u64 = 0,

    /// Label for this progress line (e.g., "Downloading", "Converting").
    /// Null-terminated string, may be null for updates that don't change label.
    label: ?[*:0]const u8 = null,

    /// Context message (e.g., filename, tensor name).
    /// Null-terminated string, may be null.
    message: ?[*:0]const u8 = null,

    /// Unit for display (e.g., "files", "tensors", "bytes").
    /// Null-terminated string, may be null (binding chooses default).
    unit: ?[*:0]const u8 = null,
};

/// Progress callback signature.
/// Bindings implement this to receive progress updates from core.
///
/// Parameters:
/// - update: Structured progress update with all rendering information
/// - user_data: User-provided context pointer (passed back unchanged)
pub const CProgressCallback = *const fn (
    update: *const ProgressUpdate,
    user_data: ?*anyopaque,
) callconv(.c) void;

// =============================================================================
// Progress Context (Internal)
// =============================================================================

/// Internal progress context passed through the call chain.
/// Wraps the user callback and provides helper methods for emitting updates.
pub const ProgressContext = struct {
    callback: ?CProgressCallback,
    user_data: ?*anyopaque,

    /// Create a no-op context (when callback is null).
    pub const NONE: ProgressContext = .{ .callback = null, .user_data = null };

    /// Create a context from callback and user_data.
    pub fn init(callback: ?CProgressCallback, user_data: ?*anyopaque) ProgressContext {
        return .{ .callback = callback, .user_data = user_data };
    }

    /// Check if this context has an active callback.
    pub fn isActive(self: ProgressContext) bool {
        return self.callback != null;
    }

    /// Emit a progress update.
    pub fn emit(self: ProgressContext, update: ProgressUpdate) void {
        if (self.callback) |cb| {
            cb(&update, self.user_data);
        }
    }

    /// Helper: Add a new progress line.
    pub fn addLine(
        self: ProgressContext,
        line_id: u8,
        label: ?[*:0]const u8,
        total: u64,
        message: ?[*:0]const u8,
        unit: ?[*:0]const u8,
    ) void {
        self.emit(.{
            .line_id = line_id,
            .action = .add,
            .current = 0,
            .total = total,
            .label = label,
            .message = message,
            .unit = unit,
        });
    }

    /// Helper: Update progress on a line.
    pub fn updateLine(
        self: ProgressContext,
        line_id: u8,
        current: u64,
        message: ?[*:0]const u8,
    ) void {
        self.emit(.{
            .line_id = line_id,
            .action = .update,
            .current = current,
            .total = 0, // Binding uses previously set total
            .label = null, // Keep existing label
            .message = message,
            .unit = null, // Keep existing unit
        });
    }

    /// Helper: Mark a line as complete.
    pub fn completeLine(self: ProgressContext, line_id: u8) void {
        self.emit(.{
            .line_id = line_id,
            .action = .complete,
            .current = 0,
            .total = 0,
            .label = null,
            .message = null,
            .unit = null,
        });
    }
};

// =============================================================================
// Tests
// =============================================================================

test "ProgressUpdate struct is C-compatible" {
    // Verify the struct can be used across FFI boundary
    var update = std.mem.zeroes(ProgressUpdate);
    update.line_id = 0;
    update.action = .add;
    update.current = 100;
    update.total = 1000;
    update.label = "Testing";
    update.message = "test.txt";
    update.unit = "files";

    try std.testing.expectEqual(@as(u8, 0), update.line_id);
    try std.testing.expectEqual(ProgressAction.add, update.action);
    try std.testing.expectEqual(@as(u64, 100), update.current);
    try std.testing.expectEqual(@as(u64, 1000), update.total);
}

test "ProgressContext emit calls callback" {
    const State = struct {
        var called: bool = false;
        var last_action: ProgressAction = .update;
        var last_current: u64 = 0;

        fn callback(update: *const ProgressUpdate, _: ?*anyopaque) callconv(.c) void {
            called = true;
            last_action = update.action;
            last_current = update.current;
        }
    };

    State.called = false;
    const ctx = ProgressContext.init(State.callback, null);

    ctx.emit(.{ .action = .add, .current = 42 });

    try std.testing.expect(State.called);
    try std.testing.expectEqual(ProgressAction.add, State.last_action);
    try std.testing.expectEqual(@as(u64, 42), State.last_current);
}

test "ProgressContext.NONE does not crash" {
    // NONE context should be safe to use (no-op)
    const ctx = ProgressContext.NONE;
    try std.testing.expect(!ctx.isActive());

    // These should be no-ops, not crash
    ctx.emit(.{ .action = .update, .current = 100 });
    ctx.addLine(0, "Test", 100, null, null);
    ctx.updateLine(0, 50, "halfway");
    ctx.completeLine(0);
}
