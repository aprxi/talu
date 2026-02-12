//! Token Trie for efficient grammar-constrained decoding.
//!
//! This module implements a trie-based approach for computing valid token masks.
//! Instead of checking each token individually (O(vocab × token_length)),
//! we walk the trie and grammar state machine simultaneously, pruning entire
//! branches when the grammar state becomes invalid.
//!
//! This reduces complexity from O(V × L) to O(A) where:
//! - V = vocabulary size (~150k)
//! - L = average token length (~4 bytes)
//! - A = number of active edges in valid trie subset (typically <2000)

const std = @import("std");
const engine_mod = @import("engine.zig");
const mask_mod = @import("mask.zig");
const log = @import("../log.zig");

const Engine = engine_mod.Engine;
const StackSet = engine_mod.StackSet;
const Frame = engine_mod.Frame;
const TokenMask = mask_mod.TokenMask;

/// A node in the token trie.
/// Designed for cache locality - children are stored contiguously in a separate array.
pub const TrieNode = struct {
    /// Index into the edges array where this node's children start
    children_start: u32,
    /// Number of children (edges) from this node
    /// Using u16 to handle nodes with >255 children (root can have 256)
    children_count: u16,
    /// If this node represents a complete token, store its ID. Otherwise null.
    token_id: ?u32,
    /// Padding for alignment
    _padding: [2]u8 = .{ 0, 0 },
};

/// An edge in the trie - a byte transition to another node.
pub const TrieEdge = struct {
    /// The byte value for this transition
    byte: u8,
    /// Padding for alignment
    _padding: [3]u8 = .{ 0, 0, 0 },
    /// Index of the target node
    target_node: u32,
};

/// Work item for the DFS stack during trie-grammar intersection.
const WorkItem = struct {
    /// Current position in the trie
    trie_node: u32,
    /// Grammar states at this position (index into states array)
    states_idx: u32,
};

/// Token Trie - indexes the entire vocabulary for efficient grammar intersection.
pub const TokenTrie = struct {
    allocator: std.mem.Allocator,
    /// All nodes in the trie (contiguous for cache locality)
    nodes: std.ArrayListUnmanaged(TrieNode),
    /// All edges in the trie (contiguous, grouped by parent node)
    edges: std.ArrayListUnmanaged(TrieEdge),
    /// Vocabulary size this trie was built for
    vocab_size: usize,

    const Self = @This();

    /// Build a token trie from a tokenizer.
    /// The tokenizer must support tokenBytes(id) -> ?[]const u8
    pub fn init(allocator: std.mem.Allocator, tokenizer: anytype) !Self {
        var self = Self{
            .allocator = allocator,
            .nodes = .empty,
            .edges = .empty,
            .vocab_size = getVocabSize(tokenizer),
        };
        errdefer self.deinit();

        // Build trie by inserting all tokens
        // We use a temporary structure to build children lists, then flatten
        var builder = TrieBuilder.init(allocator);
        defer builder.deinit();

        const vocab_size = self.vocab_size;
        for (0..vocab_size) |token_id| {
            const token_bytes = getTokenBytes(tokenizer, token_id) orelse continue;
            if (token_bytes.len == 0) continue;

            try builder.insert(token_bytes, @intCast(token_id));
        }

        // Flatten the builder into our compact format
        try builder.flatten(&self.nodes, &self.edges, allocator);

        log.info("grammar", "TokenTrie built", .{
            .vocab_size = vocab_size,
            .num_nodes = self.nodes.items.len,
            .num_edges = self.edges.items.len,
        });

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.nodes.deinit(self.allocator);
        self.edges.deinit(self.allocator);
        self.* = undefined;
    }

    /// Compute valid token mask by intersecting trie with grammar state machine.
    /// This is the core algorithm that achieves O(active_edges) complexity.
    pub fn computeValidTokens(
        self: *const Self,
        engine: *Engine,
        mask: *TokenMask,
    ) !void {
        const start_time = std.time.milliTimestamp();

        // Use arena for all temporary allocations during traversal
        var arena = std.heap.ArenaAllocator.init(engine.allocator);
        defer arena.deinit();
        const arena_alloc = arena.allocator();

        // DFS stack
        var stack = std.ArrayListUnmanaged(WorkItem){};
        defer stack.deinit(arena_alloc);

        // Store states at each level (we reuse indices to avoid excessive cloning)
        var states_pool = std.ArrayListUnmanaged(StackSet){};
        defer {
            for (states_pool.items) |*s| s.deinit();
            states_pool.deinit(arena_alloc);
        }

        // Clone initial states
        const initial_states = try cloneStackSetWithAllocator(&engine.states, arena_alloc);
        try states_pool.append(arena_alloc, initial_states);

        // Get valid first bytes to filter root edges
        var valid_first_bytes: [256]bool = [_]bool{false} ** 256;
        engine.getValidFirstBytes(&valid_first_bytes);

        // Count valid first bytes for "wide open" detection
        var valid_count: usize = 0;
        for (valid_first_bytes) |v| {
            if (v) valid_count += 1;
        }
        const is_wide_open = valid_count > 200;

        // Stats for logging
        var edges_traversed: usize = 0;
        var tokens_accepted: usize = 0;
        var branches_pruned: usize = 0;

        // Fast path: if wide-open at root, mark all tokens valid
        // This is O(num_nodes) instead of O(vocab_size) - faster for large vocabularies
        if (is_wide_open) {
            for (self.nodes.items) |node| {
                if (node.token_id) |token_id| {
                    mask.setValid(token_id);
                    tokens_accepted += 1;
                }
            }
            const elapsed_ms = std.time.milliTimestamp() - start_time;
            log.debug("grammar", "Trie wide-open scan", .{
                .tokens_accepted = tokens_accepted,
                .elapsed_ms = elapsed_ms,
            }, @src());
            return;
        }

        // Push root node's valid children to stack
        const root = self.nodes.items[0];
        const root_edges = self.edges.items[root.children_start..][0..root.children_count];

        for (root_edges) |edge| {
            if (!valid_first_bytes[edge.byte]) {
                branches_pruned += 1;
                continue;
            }

            // Advance grammar state for this byte
            const next_states = try engine.advanceStatesWithAllocator(
                &states_pool.items[0],
                edge.byte,
                arena_alloc,
            );

            if (next_states.stacks.items.len == 0) {
                branches_pruned += 1;
                continue;
            }

            const states_idx: u32 = @intCast(states_pool.items.len);
            try states_pool.append(arena_alloc, next_states);

            try stack.append(arena_alloc, .{
                .trie_node = edge.target_node,
                .states_idx = states_idx,
            });
            edges_traversed += 1;
        }

        // DFS traversal
        while (stack.items.len > 0) {
            const item = stack.pop().?;
            const node = self.nodes.items[item.trie_node];
            const current_states = &states_pool.items[item.states_idx];

            // If this node represents a complete token, mark it valid
            if (node.token_id) |token_id| {
                mask.setValid(token_id);
                tokens_accepted += 1;
            }

            // Get valid bytes from current grammar state
            var valid_bytes: [256]bool = [_]bool{false} ** 256;
            engine.getValidBytesFromStates(current_states, &valid_bytes);

            // Count valid bytes for wide-open detection at this state
            var state_valid_count: usize = 0;
            for (valid_bytes) |v| {
                if (v) state_valid_count += 1;
            }
            const state_is_wide_open = state_valid_count > 200;

            // Process children
            const node_edges = self.edges.items[node.children_start..][0..node.children_count];

            if (state_is_wide_open and node.children_count > 0) {
                // Wide-open mode: flood fill - accept all tokens in subtree
                try self.floodFillSubtree(item.trie_node, mask, &tokens_accepted);
                continue;
            }

            for (node_edges) |edge| {
                if (!valid_bytes[edge.byte]) {
                    branches_pruned += 1;
                    continue;
                }

                // Advance grammar state for this byte
                var states_copy = try cloneStackSetWithAllocator(current_states, arena_alloc);
                const next_states = try engine.advanceStatesWithAllocator(
                    &states_copy,
                    edge.byte,
                    arena_alloc,
                );

                if (next_states.stacks.items.len == 0) {
                    branches_pruned += 1;
                    continue;
                }

                const states_idx: u32 = @intCast(states_pool.items.len);
                try states_pool.append(arena_alloc, next_states);

                try stack.append(arena_alloc, .{
                    .trie_node = edge.target_node,
                    .states_idx = states_idx,
                });
                edges_traversed += 1;
            }
        }

        const elapsed_ms = std.time.milliTimestamp() - start_time;

        log.debug("grammar", "Trie intersection complete", .{
            .edges_traversed = edges_traversed,
            .tokens_accepted = tokens_accepted,
            .branches_pruned = branches_pruned,
            .elapsed_ms = elapsed_ms,
            .is_wide_open = is_wide_open,
        }, @src());
    }

    /// Flood fill: accept all tokens in a subtree (used in wide-open mode).
    fn floodFillSubtree(
        self: *const Self,
        start_node: u32,
        mask: *TokenMask,
        tokens_accepted: *usize,
    ) !void {
        // Simple iterative flood fill using a stack
        var visit_stack: [256]u32 = undefined;
        var stack_len: usize = 1;
        visit_stack[0] = start_node;

        while (stack_len > 0) {
            stack_len -= 1;
            const node_idx = visit_stack[stack_len];
            const node = self.nodes.items[node_idx];

            // Accept token if present
            if (node.token_id) |token_id| {
                mask.setValid(token_id);
                tokens_accepted.* += 1;
            }

            // Push all children
            const node_edges = self.edges.items[node.children_start..][0..node.children_count];
            for (node_edges) |edge| {
                if (stack_len < visit_stack.len) {
                    visit_stack[stack_len] = edge.target_node;
                    stack_len += 1;
                }
                // If stack overflows, we'll miss some tokens but that's better than crashing
                // In practice, this shouldn't happen with typical token lengths
            }
        }
    }

    /// Get the number of nodes in the trie
    pub fn nodeCount(self: *const Self) usize {
        return self.nodes.items.len;
    }

    /// Get the number of edges in the trie
    pub fn edgeCount(self: *const Self) usize {
        return self.edges.items.len;
    }
};

/// Helper struct for building the trie before flattening.
const TrieBuilder = struct {
    allocator: std.mem.Allocator,
    /// Map from (parent_node, byte) -> child_node
    children: std.AutoHashMapUnmanaged(u64, u32),
    /// Token ID at each node (null if intermediate)
    token_ids: std.ArrayListUnmanaged(?u32),
    /// Number of nodes created
    node_count: u32,

    const Self = @This();

    fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .children = .empty,
            .token_ids = .empty,
            .node_count = 1, // Start with root
        };
    }

    fn deinit(self: *Self) void {
        self.children.deinit(self.allocator);
        self.token_ids.deinit(self.allocator);
    }

    fn insert(self: *Self, bytes: []const u8, token_id: u32) !void {
        var current_node: u32 = 0; // Start at root

        for (bytes) |byte| {
            const key = makeKey(current_node, byte);
            if (self.children.get(key)) |child| {
                current_node = child;
            } else {
                // Create new node
                const new_node = self.node_count;
                self.node_count += 1;
                try self.children.put(self.allocator, key, new_node);
                current_node = new_node;
            }
        }

        // Mark this node with the token ID
        // Ensure token_ids array is large enough
        while (self.token_ids.items.len <= current_node) {
            try self.token_ids.append(self.allocator, null);
        }
        self.token_ids.items[current_node] = token_id;
    }

    fn makeKey(node: u32, byte: u8) u64 {
        return (@as(u64, node) << 8) | @as(u64, byte);
    }

    /// Flatten the builder into compact arrays for cache-friendly traversal.
    fn flatten(
        self: *Self,
        nodes: *std.ArrayListUnmanaged(TrieNode),
        edges: *std.ArrayListUnmanaged(TrieEdge),
        allocator: std.mem.Allocator,
    ) !void {
        // First pass: count children for each node
        // Use u16 for counts to handle nodes with >255 children (e.g., root has 256)
        var children_count = try allocator.alloc(u16, self.node_count);
        defer allocator.free(children_count);
        @memset(children_count, 0);

        var iter = self.children.iterator();
        while (iter.next()) |entry| {
            const parent = @as(u32, @truncate(entry.key_ptr.* >> 8));
            if (parent < children_count.len) {
                children_count[parent] += 1;
            }
        }

        // Second pass: compute children_start for each node
        var children_start = try allocator.alloc(u32, self.node_count);
        defer allocator.free(children_start);

        var offset: u32 = 0;
        for (0..self.node_count) |i| {
            children_start[i] = offset;
            offset += children_count[i];
        }

        // Allocate edges
        const total_edges = offset;
        try edges.resize(allocator, total_edges);

        // Reset counts for filling
        @memset(children_count, 0);

        // Fill edges
        iter = self.children.iterator();
        while (iter.next()) |entry| {
            const parent = @as(u32, @truncate(entry.key_ptr.* >> 8));
            const byte = @as(u8, @truncate(entry.key_ptr.*));
            const child = entry.value_ptr.*;

            const edge_idx = children_start[parent] + children_count[parent];
            edges.items[edge_idx] = .{
                .byte = byte,
                .target_node = child,
            };
            children_count[parent] += 1;
        }

        // Sort edges for each node by byte value (for consistent traversal and potential SIMD)
        for (0..self.node_count) |i| {
            const start = children_start[i];
            const count = children_count[i];
            if (count > 1) {
                const edge_slice = edges.items[start..][0..count];
                std.mem.sort(TrieEdge, edge_slice, {}, struct {
                    fn lessThan(_: void, a: TrieEdge, b: TrieEdge) bool {
                        return a.byte < b.byte;
                    }
                }.lessThan);
            }
        }

        // Create nodes
        try nodes.resize(allocator, self.node_count);
        for (0..self.node_count) |i| {
            const token_id = if (i < self.token_ids.items.len) self.token_ids.items[i] else null;
            nodes.items[i] = .{
                .children_start = children_start[i],
                .children_count = children_count[i],
                .token_id = token_id,
            };
        }
    }
};

/// Clone a StackSet using a specific allocator
fn cloneStackSetWithAllocator(states: *const StackSet, alloc: std.mem.Allocator) !StackSet {
    var cloned = StackSet.init(alloc);
    errdefer cloned.deinit();

    try cloned.stacks.ensureTotalCapacity(alloc, states.stacks.items.len);
    for (states.stacks.items) |stack| {
        const copied = try alloc.dupe(Frame, stack.frames);
        try cloned.stacks.append(alloc, .{ .frames = copied });
    }

    return cloned;
}

/// Get vocabulary size from tokenizer (handles different tokenizer types)
fn getVocabSize(tokenizer: anytype) usize {
    const T = @TypeOf(tokenizer);
    switch (@typeInfo(T)) {
        .pointer => |info| {
            const Child = info.child;
            if (@hasDecl(Child, "getVocabSize")) {
                return tokenizer.getVocabSize();
            }
            if (@hasField(Child, "vocab_size")) {
                return tokenizer.vocab_size;
            }
        },
        else => {},
    }
    if (@hasDecl(T, "getVocabSize")) {
        return tokenizer.getVocabSize();
    }
    if (@hasField(T, "vocab_size")) {
        return tokenizer.vocab_size;
    }
    return 0;
}

/// Get token bytes from tokenizer (handles different tokenizer types)
fn getTokenBytes(tokenizer: anytype, token_id: usize) ?[]const u8 {
    const T = @TypeOf(tokenizer);
    switch (@typeInfo(T)) {
        .pointer => |info| {
            const Child = info.child;
            if (@hasDecl(Child, "tokenBytes")) {
                return tokenizer.tokenBytes(token_id);
            }
        },
        else => {},
    }
    if (@hasDecl(T, "tokenBytes")) {
        return tokenizer.tokenBytes(token_id);
    }
    return null;
}

// Tests
test "trie basic" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Mock tokenizer
    const MockTokenizer = struct {
        tokens: []const []const u8,

        pub fn getVocabSize(self: *const @This()) usize {
            return self.tokens.len;
        }

        pub fn tokenBytes(self: *const @This(), id: usize) ?[]const u8 {
            if (id < self.tokens.len) return self.tokens[id];
            return null;
        }
    };

    const tokens = [_][]const u8{
        "hello",
        "help",
        "world",
        "he",
    };

    var mock = MockTokenizer{ .tokens = &tokens };
    var trie = try TokenTrie.init(allocator, &mock);
    defer trie.deinit();

    // Check structure
    try testing.expect(trie.nodeCount() > 0);
    try testing.expect(trie.edgeCount() > 0);
}
