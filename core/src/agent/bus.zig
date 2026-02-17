//! MessageBus - Transport-agnostic inter-agent communication.
//!
//! Agents send messages by ID. The bus routes messages through the fastest
//! available transport:
//!
//!   - **In-process**: Direct mailbox enqueue (no I/O, no serialization).
//!   - **HTTP**: POST via libcurl to a remote agent's endpoint.
//!
//! Each agent has a local mailbox (FIFO queue). Remote agents are registered
//! as peers with a URL and transport type. `send()` checks local first, then
//! falls back to the peer registry.
//!
//! HTTP receiving is caller-side: the host's HTTP server calls `deliver()`
//! when a remote message arrives. The bus does not run its own HTTP server.
//!
//! # Thread Safety
//!
//! Fully thread-safe. All mailbox and peer operations are protected by a mutex.
//! The bus is designed to be shared across agents in different threads.

const std = @import("std");
const Allocator = std.mem.Allocator;

const c = @cImport({
    @cInclude("curl/curl.h");
});

// =============================================================================
// Message types
// =============================================================================

/// A message in transit between agents.
///
/// All fields are owned by the bus and freed via `freeMessage()`.
pub const Message = struct {
    from: []const u8,
    to: []const u8,
    payload: []const u8,
    timestamp_ms: i64,
};

/// Transport type for a remote peer.
pub const PeerTransport = enum(u8) {
    /// POST via libcurl to peer URL.
    http = 0,
};

/// Remote peer registration info.
pub const PeerInfo = struct {
    url: []const u8,
    transport: PeerTransport,
};

/// Callback invoked when a message is enqueued into an agent's mailbox.
///
/// Called **outside** the bus mutex to avoid deadlocks with agent-side locking.
/// Callers should not perform heavy work inside this callback; use it to
/// signal a condition variable or set a flag.
pub const OnMessageNotifyFn = *const fn (
    agent_id: [*:0]const u8,
    pending_count: usize,
    user_data: ?*anyopaque,
) callconv(.c) void;

// =============================================================================
// MessageQueue - FIFO queue for a single mailbox
// =============================================================================

/// Simple FIFO message queue backed by an ArrayList.
/// Not independently thread-safe — protected by the bus mutex.
const MessageQueue = struct {
    messages: std.ArrayListUnmanaged(Message),

    fn init() MessageQueue {
        return .{ .messages = .{} };
    }

    fn deinit(self: *MessageQueue, allocator: Allocator) void {
        for (self.messages.items) |msg| {
            freeMessageFields(allocator, msg);
        }
        self.messages.deinit(allocator);
    }

    fn enqueue(self: *MessageQueue, allocator: Allocator, msg: Message) !void {
        try self.messages.append(allocator, msg);
    }

    /// Dequeue the oldest message. Returns null if empty.
    fn dequeue(self: *MessageQueue) ?Message {
        if (self.messages.items.len == 0) return null;
        return self.messages.orderedRemove(0);
    }

    fn count(self: *const MessageQueue) usize {
        return self.messages.items.len;
    }
};

/// Mailbox wraps a MessageQueue with an optional notification callback.
const Mailbox = struct {
    queue: MessageQueue,
    on_notify: ?OnMessageNotifyFn = null,
    on_notify_data: ?*anyopaque = null,

    fn init() Mailbox {
        return .{ .queue = MessageQueue.init() };
    }

    fn deinit(self: *Mailbox, allocator: Allocator) void {
        self.queue.deinit(allocator);
    }
};

/// Free the owned fields of a Message (but not the Message struct itself).
fn freeMessageFields(allocator: Allocator, msg: Message) void {
    allocator.free(msg.from);
    allocator.free(msg.to);
    allocator.free(msg.payload);
}

// =============================================================================
// MessageBus
// =============================================================================

pub const BusError = error{
    AgentAlreadyRegistered,
    AgentNotFound,
    PeerAlreadyRegistered,
    PeerNotFound,
    SendFailed,
    OutOfMemory,
};

/// Transport-agnostic message bus for inter-agent communication.
///
/// Manages local mailboxes for in-process agents and a peer registry for
/// remote agents reachable via HTTP. Shared across threads via mutex.
pub const MessageBus = struct {
    allocator: Allocator,
    mutex: std.Thread.Mutex,

    /// Local mailboxes: agent_id -> mailbox (queue + notification callback).
    mailboxes: std.StringHashMapUnmanaged(Mailbox),

    /// Remote peer registry: agent_id -> URL + transport.
    peers: std.StringHashMapUnmanaged(PeerInfo),

    /// Create a new message bus.
    pub fn init(allocator: Allocator) MessageBus {
        return .{
            .allocator = allocator,
            .mutex = .{},
            .mailboxes = .{},
            .peers = .{},
        };
    }

    /// Shut down the bus and free all resources.
    ///
    /// Frees all queued messages, mailbox keys, peer keys, and peer URLs.
    pub fn deinit(self: *MessageBus) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Free all mailboxes and their keys
        var mb_it = self.mailboxes.iterator();
        while (mb_it.next()) |entry| {
            entry.value_ptr.queue.deinit(self.allocator);
            self.allocator.free(entry.key_ptr.*);
        }
        self.mailboxes.deinit(self.allocator);

        // Free all peers and their keys
        var peer_it = self.peers.iterator();
        while (peer_it.next()) |entry| {
            self.allocator.free(entry.value_ptr.url);
            self.allocator.free(entry.key_ptr.*);
        }
        self.peers.deinit(self.allocator);
    }

    // =========================================================================
    // Local agent registration
    // =========================================================================

    /// Register a local agent, creating its mailbox.
    /// Returns error if an agent with this ID is already registered.
    pub fn register(self: *MessageBus, agent_id: []const u8) BusError!void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const gop = self.mailboxes.getOrPut(self.allocator, agent_id) catch
            return BusError.OutOfMemory;

        if (gop.found_existing) {
            return BusError.AgentAlreadyRegistered;
        }

        // Own the key
        gop.key_ptr.* = self.allocator.dupe(u8, agent_id) catch
            return BusError.OutOfMemory;
        gop.value_ptr.* = Mailbox.init();
    }

    /// Unregister a local agent and free its mailbox.
    /// Silently ignores unknown agent IDs.
    pub fn unregister(self: *MessageBus, agent_id: []const u8) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const entry = self.mailboxes.fetchRemove(agent_id) orelse return;
        var mailbox = entry.value;
        mailbox.queue.deinit(self.allocator);
        self.allocator.free(entry.key);
    }

    // =========================================================================
    // Remote peer registration
    // =========================================================================

    /// Register a remote peer agent reachable at the given URL.
    pub fn addPeer(
        self: *MessageBus,
        agent_id: []const u8,
        url: []const u8,
        transport: PeerTransport,
    ) BusError!void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const gop = self.peers.getOrPut(self.allocator, agent_id) catch
            return BusError.OutOfMemory;

        if (gop.found_existing) {
            return BusError.PeerAlreadyRegistered;
        }

        gop.key_ptr.* = self.allocator.dupe(u8, agent_id) catch
            return BusError.OutOfMemory;
        gop.value_ptr.* = .{
            .url = self.allocator.dupe(u8, url) catch
                return BusError.OutOfMemory,
            .transport = transport,
        };
    }

    /// Remove a remote peer. Silently ignores unknown peer IDs.
    pub fn removePeer(self: *MessageBus, agent_id: []const u8) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const entry = self.peers.fetchRemove(agent_id) orelse return;
        self.allocator.free(entry.value.url);
        self.allocator.free(entry.key);
    }

    // =========================================================================
    // Notification
    // =========================================================================

    /// Set or clear the notification callback for a local agent.
    ///
    /// The callback fires after a message is enqueued into the agent's
    /// mailbox (via send/deliver/broadcast). It is called **outside** the
    /// bus mutex to avoid deadlocks with caller-side locking.
    ///
    /// Pass null for on_notify to clear the callback.
    pub fn setNotify(
        self: *MessageBus,
        agent_id: []const u8,
        on_notify: ?OnMessageNotifyFn,
        on_notify_data: ?*anyopaque,
    ) BusError!void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const mailbox = self.mailboxes.getPtr(agent_id) orelse
            return BusError.AgentNotFound;

        mailbox.on_notify = on_notify;
        mailbox.on_notify_data = on_notify_data;
    }

    // =========================================================================
    // Messaging
    // =========================================================================

    /// Send a message from one agent to another.
    ///
    /// Routing order:
    ///   1. If `to` has a local mailbox -> direct enqueue (fast, no I/O).
    ///   2. If `to` is a registered peer -> HTTP POST to peer URL.
    ///   3. If `to == "*"` -> broadcast to all local + remote.
    ///   4. Otherwise -> BusError.AgentNotFound.
    pub fn send(
        self: *MessageBus,
        from: []const u8,
        to: []const u8,
        payload: []const u8,
    ) BusError!void {
        // Broadcast case
        if (std.mem.eql(u8, to, "*")) {
            return self.broadcast(from, payload);
        }

        // Try local mailbox first
        var notify_fn: ?OnMessageNotifyFn = null;
        var notify_data: ?*anyopaque = null;
        var notify_count: usize = 0;
        var found_local = false;
        {
            self.mutex.lock();
            defer self.mutex.unlock();

            if (self.mailboxes.getPtr(to)) |mailbox| {
                found_local = true;
                const msg = self.createMessage(from, to, payload) catch
                    return BusError.OutOfMemory;
                mailbox.queue.enqueue(self.allocator, msg) catch
                    return BusError.OutOfMemory;

                // Capture notification info under lock
                if (mailbox.on_notify) |nfn| {
                    notify_fn = nfn;
                    notify_data = mailbox.on_notify_data;
                    notify_count = mailbox.queue.count();
                }
            }
        }

        if (found_local) {
            // Fire notification outside lock
            if (notify_fn) |nfn| {
                self.fireNotify(nfn, to, notify_count, notify_data);
            }
            return;
        }

        // Try remote peer
        const peer_info = blk: {
            self.mutex.lock();
            defer self.mutex.unlock();
            break :blk self.peers.get(to);
        };

        if (peer_info) |peer| {
            return self.sendRemote(from, to, payload, peer);
        }

        return BusError.AgentNotFound;
    }

    /// Receive the next message for a local agent.
    /// Returns null if the mailbox is empty.
    ///
    /// Caller must call `freeMessage()` when done with the message.
    pub fn receive(self: *MessageBus, agent_id: []const u8) ?Message {
        self.mutex.lock();
        defer self.mutex.unlock();

        const mailbox = self.mailboxes.getPtr(agent_id) orelse return null;
        return mailbox.queue.dequeue();
    }

    /// Deliver an incoming message into a local mailbox.
    ///
    /// Called by the host's HTTP server (or other transport receiver) when
    /// a remote message arrives. Enqueues into the recipient's mailbox.
    pub fn deliver(
        self: *MessageBus,
        from: []const u8,
        to: []const u8,
        payload: []const u8,
    ) BusError!void {
        var notify_fn: ?OnMessageNotifyFn = null;
        var notify_data: ?*anyopaque = null;
        var notify_count: usize = 0;
        {
            self.mutex.lock();
            defer self.mutex.unlock();

            const mailbox = self.mailboxes.getPtr(to) orelse
                return BusError.AgentNotFound;

            const msg = self.createMessage(from, to, payload) catch
                return BusError.OutOfMemory;
            mailbox.queue.enqueue(self.allocator, msg) catch
                return BusError.OutOfMemory;

            if (mailbox.on_notify) |nfn| {
                notify_fn = nfn;
                notify_data = mailbox.on_notify_data;
                notify_count = mailbox.queue.count();
            }
        }

        // Fire notification outside lock
        if (notify_fn) |nfn| {
            self.fireNotify(nfn, to, notify_count, notify_data);
        }
    }

    /// Broadcast a message to all local mailboxes and remote peers.
    ///
    /// The sender is excluded from receiving its own broadcast.
    pub fn broadcast(
        self: *MessageBus,
        from: []const u8,
        payload: []const u8,
    ) BusError!void {
        // Collect local recipients + notifications under lock
        const NotifyEntry = struct {
            notify_fn: OnMessageNotifyFn,
            agent_id: []const u8,
            pending_count: usize,
            data: ?*anyopaque,
        };
        var notifications = std.ArrayListUnmanaged(NotifyEntry){};
        defer notifications.deinit(self.allocator);
        {
            self.mutex.lock();
            defer self.mutex.unlock();

            var mb_it = self.mailboxes.iterator();
            while (mb_it.next()) |entry| {
                // Skip sender
                if (std.mem.eql(u8, entry.key_ptr.*, from)) continue;

                const msg = self.createMessage(from, entry.key_ptr.*, payload) catch
                    return BusError.OutOfMemory;
                entry.value_ptr.queue.enqueue(self.allocator, msg) catch
                    return BusError.OutOfMemory;

                if (entry.value_ptr.on_notify) |nfn| {
                    notifications.append(self.allocator, .{
                        .notify_fn = nfn,
                        .agent_id = entry.key_ptr.*,
                        .pending_count = entry.value_ptr.queue.count(),
                        .data = entry.value_ptr.on_notify_data,
                    }) catch {};
                }
            }
        }

        // Fire notifications outside lock
        for (notifications.items) |entry| {
            self.fireNotify(entry.notify_fn, entry.agent_id, entry.pending_count, entry.data);
        }

        // Collect remote peers to send to (outside lock to avoid holding
        // mutex during I/O)
        const PeerEntry = struct { id: []const u8, info: PeerInfo };
        var remote_targets = std.ArrayListUnmanaged(PeerEntry){};
        defer remote_targets.deinit(self.allocator);

        {
            self.mutex.lock();
            defer self.mutex.unlock();

            var peer_it = self.peers.iterator();
            while (peer_it.next()) |entry| {
                if (std.mem.eql(u8, entry.key_ptr.*, from)) continue;
                remote_targets.append(self.allocator, .{
                    .id = entry.key_ptr.*,
                    .info = entry.value_ptr.*,
                }) catch return BusError.OutOfMemory;
            }
        }

        // Send to remote peers outside lock
        for (remote_targets.items) |target| {
            self.sendRemote(from, target.id, payload, target.info) catch {};
        }
    }

    /// Free a message returned by `receive()`.
    pub fn freeMessage(self: *MessageBus, msg: Message) void {
        freeMessageFields(self.allocator, msg);
    }

    /// Number of messages waiting in an agent's mailbox.
    /// Returns 0 for unknown agents.
    pub fn pendingCount(self: *MessageBus, agent_id: []const u8) usize {
        self.mutex.lock();
        defer self.mutex.unlock();

        const mailbox = self.mailboxes.getPtr(agent_id) orelse return 0;
        return mailbox.queue.count();
    }

    // =========================================================================
    // Internal helpers
    // =========================================================================

    /// Fire a notification callback with a null-terminated agent_id.
    ///
    /// Must be called **outside** the bus mutex.
    fn fireNotify(
        self: *MessageBus,
        notify_fn: OnMessageNotifyFn,
        agent_id: []const u8,
        pending_count: usize,
        notify_data: ?*anyopaque,
    ) void {
        const id_z = self.allocator.dupeZ(u8, agent_id) catch return;
        defer self.allocator.free(id_z);
        notify_fn(id_z.ptr, pending_count, notify_data);
    }

    /// Create a Message with owned copies of all fields.
    /// Caller must hold mutex or ensure no concurrent access to allocator.
    fn createMessage(
        self: *MessageBus,
        from: []const u8,
        to: []const u8,
        payload: []const u8,
    ) Allocator.Error!Message {
        return .{
            .from = try self.allocator.dupe(u8, from),
            .to = try self.allocator.dupe(u8, to),
            .payload = try self.allocator.dupe(u8, payload),
            .timestamp_ms = std.time.milliTimestamp(),
        };
    }

    /// Send a message to a remote peer via its configured transport.
    fn sendRemote(
        self: *MessageBus,
        from: []const u8,
        to: []const u8,
        payload: []const u8,
        peer: PeerInfo,
    ) BusError!void {
        switch (peer.transport) {
            .http => return self.sendHttp(from, to, payload, peer.url),
        }
    }

    /// Send a message via HTTP POST to a remote peer.
    ///
    /// POST body: {"from":"<agent_id>","to":"<agent_id>","payload":"<content>"}
    /// Content-Type: application/json
    fn sendHttp(
        self: *MessageBus,
        from: []const u8,
        to: []const u8,
        payload: []const u8,
        url: []const u8,
    ) BusError!void {
        // Build JSON body
        const body = std.fmt.allocPrint(
            self.allocator,
            "{{\"from\":\"{s}\",\"to\":\"{s}\",\"payload\":\"{s}\"}}",
            .{ from, to, payload },
        ) catch return BusError.OutOfMemory;
        defer self.allocator.free(body);

        // Null-terminate URL for curl
        const url_z = self.allocator.dupeZ(u8, url) catch
            return BusError.OutOfMemory;
        defer self.allocator.free(url_z);

        // Initialize curl handle
        const curl_handle = c.curl_easy_init() orelse
            return BusError.SendFailed;
        defer c.curl_easy_cleanup(curl_handle);

        // Set URL
        if (c.curl_easy_setopt(curl_handle, c.CURLOPT_URL, url_z.ptr) != c.CURLE_OK)
            return BusError.SendFailed;

        // POST method
        if (c.curl_easy_setopt(curl_handle, c.CURLOPT_POST, @as(c_long, 1)) != c.CURLE_OK)
            return BusError.SendFailed;

        // POST body
        if (c.curl_easy_setopt(curl_handle, c.CURLOPT_POSTFIELDS, body.ptr) != c.CURLE_OK)
            return BusError.SendFailed;
        if (c.curl_easy_setopt(curl_handle, c.CURLOPT_POSTFIELDSIZE, @as(c_long, @intCast(body.len))) != c.CURLE_OK)
            return BusError.SendFailed;

        // Headers
        var headers: ?*c.struct_curl_slist = null;
        headers = c.curl_slist_append(headers, "Content-Type: application/json");
        defer if (headers) |h| c.curl_slist_free_all(h);

        if (c.curl_easy_setopt(curl_handle, c.CURLOPT_HTTPHEADER, headers) != c.CURLE_OK)
            return BusError.SendFailed;

        // Timeouts: 5s connect, 10s total
        if (c.curl_easy_setopt(curl_handle, c.CURLOPT_CONNECTTIMEOUT_MS, @as(c_long, 5000)) != c.CURLE_OK)
            return BusError.SendFailed;
        if (c.curl_easy_setopt(curl_handle, c.CURLOPT_TIMEOUT_MS, @as(c_long, 10000)) != c.CURLE_OK)
            return BusError.SendFailed;

        // Discard response body
        if (c.curl_easy_setopt(curl_handle, c.CURLOPT_WRITEFUNCTION, @as(?*const anyopaque, @ptrCast(&discardWriteCallback))) != c.CURLE_OK)
            return BusError.SendFailed;

        // Perform
        if (c.curl_easy_perform(curl_handle) != c.CURLE_OK)
            return BusError.SendFailed;

        // Check HTTP status
        var status_code: c_long = 0;
        _ = c.curl_easy_getinfo(curl_handle, c.CURLINFO_RESPONSE_CODE, &status_code);
        if (status_code >= 400)
            return BusError.SendFailed;
    }

    /// Curl write callback that discards all response data.
    fn discardWriteCallback(
        _: [*c]u8,
        size: usize,
        nmemb: usize,
        _: ?*anyopaque,
    ) callconv(.c) usize {
        return size * nmemb;
    }
};

// =============================================================================
// Tests
// =============================================================================

test "MessageBus init and deinit" {
    const allocator = std.testing.allocator;
    var bus = MessageBus.init(allocator);
    bus.deinit();
}

test "MessageBus register creates mailbox" {
    const allocator = std.testing.allocator;
    var bus = MessageBus.init(allocator);
    defer bus.deinit();

    try bus.register("agent-a");
    try std.testing.expectEqual(@as(usize, 0), bus.pendingCount("agent-a"));
}

test "MessageBus register rejects duplicate" {
    const allocator = std.testing.allocator;
    var bus = MessageBus.init(allocator);
    defer bus.deinit();

    try bus.register("agent-a");
    try std.testing.expectError(BusError.AgentAlreadyRegistered, bus.register("agent-a"));
}

test "MessageBus unregister removes mailbox" {
    const allocator = std.testing.allocator;
    var bus = MessageBus.init(allocator);
    defer bus.deinit();

    try bus.register("agent-a");
    bus.unregister("agent-a");
    // Pending count returns 0 for unknown agents
    try std.testing.expectEqual(@as(usize, 0), bus.pendingCount("agent-a"));
}

test "MessageBus unregister unknown is safe" {
    const allocator = std.testing.allocator;
    var bus = MessageBus.init(allocator);
    defer bus.deinit();

    bus.unregister("nonexistent");
}

test "MessageBus send and receive local message" {
    const allocator = std.testing.allocator;
    var bus = MessageBus.init(allocator);
    defer bus.deinit();

    try bus.register("alice");
    try bus.register("bob");

    try bus.send("alice", "bob", "hello bob");
    try std.testing.expectEqual(@as(usize, 1), bus.pendingCount("bob"));
    try std.testing.expectEqual(@as(usize, 0), bus.pendingCount("alice"));

    const msg = bus.receive("bob") orelse return error.TestUnexpectedResult;
    defer bus.freeMessage(msg);

    try std.testing.expectEqualStrings("alice", msg.from);
    try std.testing.expectEqualStrings("bob", msg.to);
    try std.testing.expectEqualStrings("hello bob", msg.payload);
    try std.testing.expect(msg.timestamp_ms > 0);

    // Mailbox is now empty
    try std.testing.expectEqual(@as(usize, 0), bus.pendingCount("bob"));
    try std.testing.expect(bus.receive("bob") == null);
}

test "MessageBus send to unknown agent returns error" {
    const allocator = std.testing.allocator;
    var bus = MessageBus.init(allocator);
    defer bus.deinit();

    try bus.register("alice");
    try std.testing.expectError(BusError.AgentNotFound, bus.send("alice", "unknown", "hello"));
}

test "MessageBus receive from unknown agent returns null" {
    const allocator = std.testing.allocator;
    var bus = MessageBus.init(allocator);
    defer bus.deinit();

    try std.testing.expect(bus.receive("nonexistent") == null);
}

test "MessageBus deliver enqueues into local mailbox" {
    const allocator = std.testing.allocator;
    var bus = MessageBus.init(allocator);
    defer bus.deinit();

    try bus.register("bob");
    try bus.deliver("remote-alice", "bob", "hello from remote");

    try std.testing.expectEqual(@as(usize, 1), bus.pendingCount("bob"));
    const msg = bus.receive("bob") orelse return error.TestUnexpectedResult;
    defer bus.freeMessage(msg);

    try std.testing.expectEqualStrings("remote-alice", msg.from);
    try std.testing.expectEqualStrings("hello from remote", msg.payload);
}

test "MessageBus deliver to unknown agent returns error" {
    const allocator = std.testing.allocator;
    var bus = MessageBus.init(allocator);
    defer bus.deinit();

    try std.testing.expectError(BusError.AgentNotFound, bus.deliver("a", "unknown", "hi"));
}

test "MessageBus broadcast delivers to all except sender" {
    const allocator = std.testing.allocator;
    var bus = MessageBus.init(allocator);
    defer bus.deinit();

    try bus.register("alice");
    try bus.register("bob");
    try bus.register("carol");

    try bus.send("alice", "*", "announcement");

    // alice should NOT receive her own broadcast
    try std.testing.expectEqual(@as(usize, 0), bus.pendingCount("alice"));
    // bob and carol should each have 1 message
    try std.testing.expectEqual(@as(usize, 1), bus.pendingCount("bob"));
    try std.testing.expectEqual(@as(usize, 1), bus.pendingCount("carol"));

    const bob_msg = bus.receive("bob") orelse return error.TestUnexpectedResult;
    defer bus.freeMessage(bob_msg);
    try std.testing.expectEqualStrings("announcement", bob_msg.payload);

    const carol_msg = bus.receive("carol") orelse return error.TestUnexpectedResult;
    defer bus.freeMessage(carol_msg);
    try std.testing.expectEqualStrings("announcement", carol_msg.payload);
}

test "MessageBus FIFO ordering" {
    const allocator = std.testing.allocator;
    var bus = MessageBus.init(allocator);
    defer bus.deinit();

    try bus.register("a");
    try bus.register("b");

    try bus.send("a", "b", "first");
    try bus.send("a", "b", "second");
    try bus.send("a", "b", "third");
    try std.testing.expectEqual(@as(usize, 3), bus.pendingCount("b"));

    const m1 = bus.receive("b") orelse return error.TestUnexpectedResult;
    defer bus.freeMessage(m1);
    try std.testing.expectEqualStrings("first", m1.payload);

    const m2 = bus.receive("b") orelse return error.TestUnexpectedResult;
    defer bus.freeMessage(m2);
    try std.testing.expectEqualStrings("second", m2.payload);

    const m3 = bus.receive("b") orelse return error.TestUnexpectedResult;
    defer bus.freeMessage(m3);
    try std.testing.expectEqualStrings("third", m3.payload);
}

test "MessageBus addPeer and removePeer" {
    const allocator = std.testing.allocator;
    var bus = MessageBus.init(allocator);
    defer bus.deinit();

    try bus.addPeer("remote-agent", "http://remote:8080/v1/agents/remote-agent/message", .http);
    try std.testing.expectError(
        BusError.PeerAlreadyRegistered,
        bus.addPeer("remote-agent", "http://other:9090", .http),
    );

    bus.removePeer("remote-agent");
    bus.removePeer("nonexistent"); // safe
}

test "MessageBus pendingCount returns 0 for unknown agent" {
    const allocator = std.testing.allocator;
    var bus = MessageBus.init(allocator);
    defer bus.deinit();

    try std.testing.expectEqual(@as(usize, 0), bus.pendingCount("ghost"));
}

test "MessageBus unregister frees queued messages" {
    const allocator = std.testing.allocator;
    var bus = MessageBus.init(allocator);
    defer bus.deinit();

    try bus.register("a");
    try bus.register("b");
    try bus.send("a", "b", "will be freed on unregister");
    try bus.send("a", "b", "also freed");
    bus.unregister("b");
    // testing.allocator will catch leaks if messages weren't freed
}

test "PeerTransport enum values" {
    try std.testing.expectEqual(@as(u8, 0), @intFromEnum(PeerTransport.http));
}

test "MessageBus setNotify fires on send" {
    const allocator = std.testing.allocator;
    var bus = MessageBus.init(allocator);
    defer bus.deinit();

    const Ctx = struct {
        var call_count: usize = 0;
        var last_pending: usize = 0;

        fn notify(_: [*:0]const u8, pending_count: usize, _: ?*anyopaque) callconv(.c) void {
            call_count += 1;
            last_pending = pending_count;
        }
    };

    try bus.register("alice");
    try bus.register("bob");
    try bus.setNotify("bob", Ctx.notify, null);

    Ctx.call_count = 0;
    try bus.send("alice", "bob", "hello");
    try std.testing.expectEqual(@as(usize, 1), Ctx.call_count);
    try std.testing.expectEqual(@as(usize, 1), Ctx.last_pending);

    // Second message increases pending count
    try bus.send("alice", "bob", "again");
    try std.testing.expectEqual(@as(usize, 2), Ctx.call_count);
    try std.testing.expectEqual(@as(usize, 2), Ctx.last_pending);
}

test "MessageBus setNotify fires on deliver" {
    const allocator = std.testing.allocator;
    var bus = MessageBus.init(allocator);
    defer bus.deinit();

    const Ctx = struct {
        var call_count: usize = 0;

        fn notify(_: [*:0]const u8, _: usize, _: ?*anyopaque) callconv(.c) void {
            call_count += 1;
        }
    };

    try bus.register("bob");
    try bus.setNotify("bob", Ctx.notify, null);

    Ctx.call_count = 0;
    try bus.deliver("remote-alice", "bob", "hello from remote");
    try std.testing.expectEqual(@as(usize, 1), Ctx.call_count);
}

test "MessageBus setNotify fires on broadcast" {
    const allocator = std.testing.allocator;
    var bus = MessageBus.init(allocator);
    defer bus.deinit();

    const Ctx = struct {
        var call_count: usize = 0;

        fn notify(_: [*:0]const u8, _: usize, _: ?*anyopaque) callconv(.c) void {
            call_count += 1;
        }
    };

    try bus.register("alice");
    try bus.register("bob");
    try bus.register("carol");
    try bus.setNotify("bob", Ctx.notify, null);
    try bus.setNotify("carol", Ctx.notify, null);

    Ctx.call_count = 0;
    try bus.send("alice", "*", "broadcast");
    // Both bob and carol should be notified (alice is sender, excluded)
    try std.testing.expectEqual(@as(usize, 2), Ctx.call_count);
}

test "MessageBus setNotify for unknown agent returns error" {
    const allocator = std.testing.allocator;
    var bus = MessageBus.init(allocator);
    defer bus.deinit();

    try std.testing.expectError(BusError.AgentNotFound, bus.setNotify("ghost", struct {
        fn noop(_: [*:0]const u8, _: usize, _: ?*anyopaque) callconv(.c) void {}
    }.noop, null));
}

test "MessageBus setNotify clear with null" {
    const allocator = std.testing.allocator;
    var bus = MessageBus.init(allocator);
    defer bus.deinit();

    const Ctx = struct {
        var call_count: usize = 0;

        fn notify(_: [*:0]const u8, _: usize, _: ?*anyopaque) callconv(.c) void {
            call_count += 1;
        }
    };

    try bus.register("alice");
    try bus.register("bob");
    try bus.setNotify("bob", Ctx.notify, null);

    Ctx.call_count = 0;
    try bus.send("alice", "bob", "first");
    try std.testing.expectEqual(@as(usize, 1), Ctx.call_count);

    // Clear notification
    try bus.setNotify("bob", null, null);
    try bus.send("alice", "bob", "second");
    // Should NOT trigger notification
    try std.testing.expectEqual(@as(usize, 1), Ctx.call_count);
}

test "MessageBus no notification without setNotify" {
    const allocator = std.testing.allocator;
    var bus = MessageBus.init(allocator);
    defer bus.deinit();

    try bus.register("alice");
    try bus.register("bob");

    // send without notification set — should not crash
    try bus.send("alice", "bob", "hello");
    try std.testing.expectEqual(@as(usize, 1), bus.pendingCount("bob"));
}
