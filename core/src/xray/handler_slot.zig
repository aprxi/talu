const std = @import("std");

/// Synchronizes enable/disable against in-flight handler callbacks.
///
/// Atomic pointer swaps are not sufficient here: a backend thread can load the
/// handler function and enter the callback while teardown concurrently disables
/// capture and destroys the target object. This slot serializes both sides so
/// disable waits until any in-flight callback has finished using the pointer.
pub fn HandlerSlot(comptime T: type) type {
    return struct {
        const Self = @This();

        mutex: std.Thread.Mutex = .{},
        ptr: ?*T = null,

        pub const Locked = struct {
            slot: *Self,
            ptr: ?*T,

            pub fn release(self: *Locked) void {
                self.slot.mutex.unlock();
            }
        };

        pub fn set(self: *Self, ptr: ?*T) void {
            self.mutex.lock();
            defer self.mutex.unlock();
            self.ptr = ptr;
        }

        pub fn acquire(self: *Self) Locked {
            self.mutex.lock();
            return .{
                .slot = self,
                .ptr = self.ptr,
            };
        }

        pub fn isEnabled(self: *Self) bool {
            self.mutex.lock();
            defer self.mutex.unlock();
            return self.ptr != null;
        }
    };
}

test "HandlerSlot disable waits for in-flight callback" {
    const Dummy = struct {
        mu: std.Thread.Mutex = .{},
        cv: std.Thread.Condition = .{},
        entered: bool = false,
        release_handler: bool = false,
        disabled: bool = false,
    };
    const Slot = HandlerSlot(Dummy);
    const HandlerCtx = struct {
        slot: *Slot,

        fn run(ctx: *@This()) void {
            var locked = ctx.slot.acquire();
            defer locked.release();
            if (locked.ptr) |dummy| {
                dummy.mu.lock();
                dummy.entered = true;
                dummy.cv.signal();
                while (!dummy.release_handler) {
                    dummy.cv.wait(&dummy.mu);
                }
                dummy.mu.unlock();
            }
        }
    };
    const DisableCtx = struct {
        slot: *Slot,
        dummy: *Dummy,

        fn run(ctx: *@This()) void {
            ctx.slot.set(null);
            ctx.dummy.mu.lock();
            ctx.dummy.disabled = true;
            ctx.dummy.cv.signal();
            ctx.dummy.mu.unlock();
        }
    };

    var slot = Slot{};
    var dummy = Dummy{};
    slot.set(&dummy);

    var handler_ctx = HandlerCtx{ .slot = &slot };
    const handler_thread = try std.Thread.spawn(.{}, HandlerCtx.run, .{&handler_ctx});

    dummy.mu.lock();
    while (!dummy.entered) {
        dummy.cv.wait(&dummy.mu);
    }
    dummy.mu.unlock();

    var disable_ctx = DisableCtx{
        .slot = &slot,
        .dummy = &dummy,
    };
    const disable_thread = try std.Thread.spawn(.{}, DisableCtx.run, .{&disable_ctx});

    dummy.mu.lock();
    try std.testing.expect(!dummy.disabled);
    dummy.release_handler = true;
    dummy.cv.signal();
    while (!dummy.disabled) {
        dummy.cv.wait(&dummy.mu);
    }
    dummy.mu.unlock();

    handler_thread.join();
    disable_thread.join();

    try std.testing.expect(!slot.isEnabled());
}
