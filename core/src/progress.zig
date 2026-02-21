//! Core progress reporting contract for long-running operations.
//!
//! This module is core-internal and backend-agnostic. Boundary layers (C-API,
//! router, models, converter) may adapt their own callback surfaces to this.

const std = @import("std");

pub const ProgressAction = enum(u8) {
    add = 0,
    update = 1,
    complete = 2,
};

pub const ProgressUpdate = extern struct {
    line_id: u8 = 0,
    action: ProgressAction = .update,
    current: u64 = 0,
    total: u64 = 0,
    label: ?[*:0]const u8 = null,
    message: ?[*:0]const u8 = null,
    unit: ?[*:0]const u8 = null,
};

pub const Callback = *const fn (
    update: *const ProgressUpdate,
    user_data: ?*anyopaque,
) callconv(.c) void;

pub const Context = struct {
    callback: ?Callback,
    user_data: ?*anyopaque,

    pub const NONE: Context = .{ .callback = null, .user_data = null };

    pub fn init(callback: ?Callback, user_data: ?*anyopaque) Context {
        return .{ .callback = callback, .user_data = user_data };
    }

    pub fn isActive(self: Context) bool {
        return self.callback != null;
    }

    pub fn emit(self: Context, update: ProgressUpdate) void {
        if (self.callback) |cb| cb(&update, self.user_data);
    }

    pub fn addLine(
        self: Context,
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

    pub fn updateLine(self: Context, line_id: u8, current: u64, message: ?[*:0]const u8) void {
        self.emit(.{
            .line_id = line_id,
            .action = .update,
            .current = current,
            .total = 0,
            .label = null,
            .message = message,
            .unit = null,
        });
    }

    pub fn completeLine(self: Context, line_id: u8) void {
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

test "Context emit calls callback" {
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
    const ctx = Context.init(State.callback, null);
    ctx.emit(.{ .action = .add, .current = 42 });

    try std.testing.expect(State.called);
    try std.testing.expectEqual(ProgressAction.add, State.last_action);
    try std.testing.expectEqual(@as(u64, 42), State.last_current);
}

