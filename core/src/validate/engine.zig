//! Grammar engine - state machine for constrained generation.
//!
//! Tracks the current parsing state and determines which tokens
//! can validly continue the generation. Uses trie-based intersection
//! for O(active_edges) complexity instead of O(vocab × token_length).

const std = @import("std");
const ast = @import("ast.zig");
const mask_mod = @import("mask.zig");
const cache = @import("cache.zig");
const log = @import("../log.zig");
const trie_mod = @import("trie.zig");

const Grammar = ast.Grammar;
const Rule = ast.Rule;
const RuleId = ast.RuleId;
const TokenMask = mask_mod.TokenMask;
const GlobalMaskCache = cache.GlobalMaskCache;
const TokenTrie = trie_mod.TokenTrie;

pub const FrameKind = enum {
    rule,
    repeat,
};

pub const Frame = struct {
    kind: FrameKind,
    rule_id: RuleId,
    position: usize,
};

pub const Stack = struct {
    frames: []Frame,
};

pub const StackSet = struct {
    allocator: std.mem.Allocator,
    stacks: std.ArrayList(Stack),

    pub fn init(allocator: std.mem.Allocator) StackSet {
        return .{
            .allocator = allocator,
            .stacks = .empty,
        };
    }

    pub fn deinit(self: *StackSet) void {
        for (self.stacks.items) |stack| {
            if (stack.frames.len > 0) {
                self.allocator.free(stack.frames);
            }
        }
        self.stacks.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn append(self: *StackSet, frames: []Frame) !void {
        try self.stacks.append(self.allocator, .{ .frames = frames });
    }

    pub fn clone(self: *const StackSet) !StackSet {
        var cloned = StackSet.init(self.allocator);
        errdefer cloned.deinit();

        try cloned.stacks.ensureTotalCapacity(self.allocator, self.stacks.items.len);
        for (self.stacks.items) |stack| {
            const copied = try self.allocator.dupe(Frame, stack.frames);
            try cloned.stacks.append(self.allocator, .{ .frames = copied });
        }

        return cloned;
    }
};

pub const Snapshot = struct {
    states: StackSet,

    pub fn deinit(self: *Snapshot) void {
        self.states.deinit();
        self.* = undefined;
    }
};

pub const Engine = struct {
    allocator: std.mem.Allocator,
    grammar: *const Grammar,
    states: StackSet,
    /// Key for this grammar in the global cache (based on grammar pointer address)
    grammar_key: cache.CacheKey,
    /// Optional token trie for O(active_edges) token validation (built lazily)
    trie: ?TokenTrie = null,

    pub fn init(allocator: std.mem.Allocator, grammar: *const Grammar) !Engine {
        var engine = Engine{
            .allocator = allocator,
            .grammar = grammar,
            .states = StackSet.init(allocator),
            // Use grammar pointer as key - same grammar object = same masks
            .grammar_key = @intFromPtr(grammar),
            .trie = null,
        };
        errdefer engine.states.deinit();

        try engine.reset();
        return engine;
    }

    pub fn deinit(self: *Engine) void {
        if (self.trie) |*t| t.deinit();
        self.states.deinit();
        self.* = undefined;
    }

    pub fn reset(self: *Engine) !void {
        self.states.deinit();
        self.states = StackSet.init(self.allocator);

        const start_frames = try self.allocator.alloc(Frame, 1);
        start_frames[0] = .{ .kind = .rule, .rule_id = self.grammar.root_rule, .position = 0 };

        const expanded = try self.expandFromStack(start_frames);
        self.states.deinit();
        self.states = expanded;
    }

    pub fn isComplete(self: *const Engine) bool {
        for (self.states.stacks.items) |stack| {
            if (stack.frames.len == 0) return true;
        }
        return false;
    }

    /// Check if the current grammar state has a deterministic continuation.
    /// Returns the literal bytes that MUST be output next, or null if non-deterministic.
    /// This enables "Jump Forward" optimization - skipping model inference for known sequences.
    pub fn getDeterministicContinuation(self: *const Engine) ?[]const u8 {
        // Fast path: if we have more than 1 state, it's unlikely to be deterministic
        // (multiple alternatives or completions)
        const num_stacks = self.states.stacks.items.len;
        if (num_stacks == 0) return null;
        if (num_stacks > 2) return null; // Too many alternatives

        // For Jump Forward to work, ALL active states must require the SAME literal prefix.
        // If any state allows different bytes, we can't jump forward.

        var required_literal: ?[]const u8 = null;
        var required_pos: usize = 0;
        var num_states: usize = 0;
        var num_literals: usize = 0;

        for (self.states.stacks.items) |stack| {
            if (stack.frames.len == 0) continue; // Complete state - doesn't constrain

            const top = stack.frames[stack.frames.len - 1];
            if (top.kind != .rule) continue;

            num_states += 1;
            const rule = self.grammar.getRule(top.rule_id) orelse continue;

            switch (rule) {
                .literal => |lit| {
                    num_literals += 1;
                    const remaining = lit[top.position..];
                    if (remaining.len == 0) continue; // Literal complete, will expand

                    if (required_literal) |req| {
                        // Check if this literal has the same prefix as the required one
                        const req_remaining = req[required_pos..];
                        if (remaining.len < req_remaining.len) {
                            // This literal is shorter - check it's a prefix
                            if (!std.mem.startsWith(u8, req_remaining, remaining)) {
                                return null; // Conflict
                            }
                            // Use the shorter one
                            required_literal = lit;
                            required_pos = top.position;
                        } else {
                            // Required is shorter - check it's a prefix of this
                            if (!std.mem.startsWith(u8, remaining, req_remaining)) {
                                return null; // Conflict
                            }
                            // Keep the shorter required_literal
                        }
                    } else {
                        required_literal = lit;
                        required_pos = top.position;
                    }
                },
                .char, .char_range => {
                    // Single byte alternatives - not deterministic unless all states agree
                    return null;
                },
                else => {
                    // Sequences, alternatives, etc. need expansion first
                    return null;
                },
            }
        }

        // Only return deterministic if ALL states are on the same literal
        if (required_literal) |lit| {
            const remaining = lit[required_pos..];
            // Only report as deterministic if we have exactly one active state path
            // or if all states agree on the literal
            if (remaining.len > 0 and num_literals == num_states and num_states > 0) {
                log.debug("grammar", "deterministic continuation found", .{
                    .literal_len = remaining.len,
                    .num_states = num_states,
                }, @src());
                return remaining;
            }
        }

        return null;
    }

    pub fn snapshot(self: *const Engine) !Snapshot {
        return .{ .states = try self.states.clone() };
    }

    pub fn restore(self: *Engine, snap: *Snapshot) void {
        self.states.deinit();
        self.states = snap.states;
        snap.* = undefined;
    }

    /// Eagerly precompute token masks for reachable grammar states.
    /// This explores the grammar by simulating byte transitions and caches masks.
    /// If the initial state is already cached, skip exploration entirely.
    pub fn eagerPrecompute(self: *Engine, tokenizer: anytype) !void {
        const vocab_size = tokenizerVocabSize(tokenizer);
        const global_cache = cache.getGlobalMaskCache(self.allocator);

        // Check if the initial state is already cached - if so, skip exploration
        const initial_hash = self.computeStateHash();
        const initial_key = GlobalMaskCache.computeMaskKey(self.grammar_key, initial_hash, vocab_size);
        if (global_cache.get(initial_key, vocab_size) != null) {
            // Cache is warm, no need to precompute
            return;
        }

        const start_time = std.time.milliTimestamp();

        // Use a set to track visited state hashes
        var visited = std.AutoHashMap(u64, void).init(self.allocator);
        defer visited.deinit();

        // Queue of states to explore
        var queue: std.ArrayList(StackSet) = .empty;
        defer {
            for (queue.items) |*s| s.deinit();
            queue.deinit(self.allocator);
        }

        // Start with current state
        const initial_clone = try self.states.clone();
        try queue.append(self.allocator, initial_clone);

        var states_computed: usize = 0;
        const max_states = 32; // Limit exploration to avoid exponential blowup

        while (queue.items.len > 0 and states_computed < max_states) {
            var current = queue.pop().?;
            defer current.deinit();

            // Compute hash for this state
            const old_states = self.states;
            self.states = current;
            const state_hash = self.computeStateHash();
            self.states = old_states;

            // Skip if already visited
            if (visited.contains(state_hash)) continue;
            try visited.put(state_hash, {});

            // Compute and cache the mask for this state
            const mask_key = GlobalMaskCache.computeMaskKey(self.grammar_key, state_hash, vocab_size);
            if (global_cache.get(mask_key, vocab_size) == null) {
                // Temporarily set states to compute mask
                const saved_states = self.states;
                self.states = try current.clone();

                // This will compute and cache the mask
                var mask = try self.getValidTokens(tokenizer);
                mask.deinit();
                self.states.deinit();
                self.states = saved_states;
                states_computed += 1;
            }

            // Only explore further if we computed this state (not from cache)
            // This avoids wasting time on exploration when cache is already populated
            if (states_computed == 0) continue;

            // Explore transitions: for each valid first byte, advance and queue new state
            var valid_bytes: [256]bool = [_]bool{false} ** 256;
            // Temporarily use current states
            const saved = self.states;
            self.states = try current.clone();
            self.getValidFirstBytes(&valid_bytes);

            // Only explore a limited number of transitions to avoid blowup
            var transitions_explored: usize = 0;
            const max_transitions = 8;

            for (valid_bytes, 0..) |is_valid, byte| {
                if (!is_valid) continue;
                if (transitions_explored >= max_transitions) break;

                // Try advancing by this byte
                var next_states = self.advanceStates(&self.states, @intCast(byte)) catch continue;

                if (next_states.stacks.items.len > 0) {
                    // Check if this is a new state
                    const old2 = self.states;
                    self.states = next_states;
                    const next_hash = self.computeStateHash();
                    self.states = old2;

                    if (!visited.contains(next_hash)) {
                        try queue.append(self.allocator, next_states);
                        transitions_explored += 1;
                    } else {
                        next_states.deinit();
                    }
                } else {
                    next_states.deinit();
                }
            }

            self.states.deinit();
            self.states = saved;
        }

        const elapsed_ms = std.time.milliTimestamp() - start_time;
        log.debug("grammar", "eagerPrecompute complete", .{
            .states_computed = states_computed,
            .elapsed_ms = elapsed_ms,
        }, @src());
    }

    /// Compute a hash of the current grammar state for caching.
    /// We hash a NORMALIZED view of the state: just the (rule_id, position) pairs
    /// sorted and deduplicated. This ensures that equivalent states (same rules at
    /// same positions, but different stack depths) produce the same hash.
    fn computeStateHash(self: *const Engine) u64 {
        var hash: u64 = 0;
        const prime: u64 = 0x9e3779b97f4a7c15;

        // Collect unique (rule_id, position) pairs from top frames
        var seen: [64]u64 = undefined; // Safe: only seen[0..seen_count] is read
        var seen_count: usize = 0;

        for (self.states.stacks.items) |stack| {
            if (stack.frames.len == 0) {
                // Empty stack means complete - use a special marker
                const marker: u64 = std.math.maxInt(u64);
                var found = false;
                for (seen[0..seen_count]) |s| {
                    if (s == marker) {
                        found = true;
                        break;
                    }
                }
                if (!found and seen_count < 64) {
                    seen[seen_count] = marker;
                    seen_count += 1;
                }
                continue;
            }

            const top = stack.frames[stack.frames.len - 1];
            if (top.kind != .rule) continue;

            // Combine rule_id and position into a single u64
            const key = (@as(u64, top.rule_id) << 32) | @as(u64, top.position);

            // Check if already seen (dedup)
            var found = false;
            for (seen[0..seen_count]) |s| {
                if (s == key) {
                    found = true;
                    break;
                }
            }
            if (!found and seen_count < 64) {
                seen[seen_count] = key;
                seen_count += 1;
            }
        }

        // Sort for order-independent hashing
        if (seen_count > 1) {
            std.mem.sort(u64, seen[0..seen_count], {}, std.sort.asc(u64));
        }

        // Hash the sorted, deduplicated keys
        for (seen[0..seen_count]) |key| {
            hash ^= key *% prime;
            hash = (hash << 17) | (hash >> 47); // Rotate
        }

        return hash;
    }

    pub fn getValidTokens(self: *Engine, tokenizer: anytype) !TokenMask {
        const start_time = std.time.milliTimestamp();
        const vocab_size = tokenizerVocabSize(tokenizer);

        // Check GLOBAL cache first (shared across all Engine instances)
        const state_hash = self.computeStateHash();
        const global_cache = cache.getGlobalMaskCache(self.allocator);
        const mask_key = GlobalMaskCache.computeMaskKey(self.grammar_key, state_hash, vocab_size);

        if (global_cache.get(mask_key, vocab_size)) |cached_data| {
            // Return a copy of the cached mask
            const mask = try TokenMask.init(self.allocator, vocab_size);
            @memcpy(mask.bits, cached_data);

            const elapsed_ms = std.time.milliTimestamp() - start_time;
            log.debug("grammar", "getValidTokens global cache hit", .{
                .state_hash = state_hash,
                .elapsed_ms = elapsed_ms,
            }, @src());

            return mask;
        }

        var mask = try TokenMask.init(self.allocator, vocab_size);

        // Use trie-based intersection for O(active_edges) complexity
        // Build trie lazily on first call
        if (self.trie == null) {
            self.trie = try TokenTrie.init(self.allocator, tokenizer);
        }

        const trie = &self.trie.?;
        try trie.computeValidTokens(self, &mask);

        const elapsed_ms = std.time.milliTimestamp() - start_time;
        log.debug("grammar", "getValidTokens trie complete", .{
            .elapsed_ms = elapsed_ms,
            .vocab_size = vocab_size,
        }, @src());

        // Cache the computed mask
        try global_cache.put(mask_key, mask.bits, vocab_size);
        return mask;
    }

    pub fn getValidFirstBytes(self: *const Engine, valid: *[256]bool) void {
        self.getValidBytesFromStates(&self.states, valid);
    }

    /// Get valid bytes from a given state set
    pub fn getValidBytesFromStates(self: *const Engine, states: *const StackSet, valid: *[256]bool) void {
        for (states.stacks.items) |stack| {
            if (stack.frames.len == 0) continue;
            const top = stack.frames[stack.frames.len - 1];
            if (top.kind != .rule) continue;

            const rule = self.grammar.getRule(top.rule_id) orelse continue;
            switch (rule) {
                .char => |c| valid[c] = true,
                .char_range => |r| {
                    var c: usize = r.start;
                    while (c <= r.end) : (c += 1) {
                        valid[c] = true;
                    }
                },
                .literal => |lit| {
                    if (top.position < lit.len) {
                        valid[lit[top.position]] = true;
                    }
                },
                else => {},
            }
        }
    }

    pub fn canAccept(self: *Engine, text: []const u8) !bool {
        // Use arena allocator for temporary allocations - much faster than individual allocs
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const arena_alloc = arena.allocator();

        var states = try self.cloneStatesWithAllocator(arena_alloc);

        for (text) |byte| {
            const next = try self.advanceStatesWithAllocator(&states, byte, arena_alloc);
            // No need to free - arena handles it
            states = next;
            if (states.stacks.items.len == 0) return false;
        }

        return states.stacks.items.len > 0;
    }

    /// Clone states using a specific allocator (for arena allocation)
    fn cloneStatesWithAllocator(self: *const Engine, alloc: std.mem.Allocator) !StackSet {
        var cloned = StackSet.init(alloc);
        errdefer cloned.deinit();

        try cloned.stacks.ensureTotalCapacity(alloc, self.states.stacks.items.len);
        for (self.states.stacks.items) |stack| {
            const copied = try alloc.dupe(Frame, stack.frames);
            try cloned.stacks.append(alloc, .{ .frames = copied });
        }

        return cloned;
    }

    /// Advance states using a specific allocator (for arena allocation)
    pub fn advanceStatesWithAllocator(self: *Engine, states: *StackSet, byte: u8, alloc: std.mem.Allocator) !StackSet {
        var next_states = StackSet.init(alloc);
        errdefer next_states.deinit();

        var seen = std.AutoHashMap(u64, void).init(alloc);
        defer seen.deinit();

        for (states.stacks.items) |stack| {
            if (stack.frames.len == 0) continue;
            const top = stack.frames[stack.frames.len - 1];
            if (top.kind != .rule) continue;

            const rule = self.grammar.getRule(top.rule_id) orelse continue;
            switch (rule) {
                .char => |c| {
                    if (byte == c) {
                        const popped = try copyWithoutTopAlloc(alloc, stack.frames);
                        try self.addExpandedStackWithAlloc(&next_states, &seen, popped, alloc);
                    }
                },
                .char_range => |r| {
                    if (byte >= r.start and byte <= r.end) {
                        const popped = try copyWithoutTopAlloc(alloc, stack.frames);
                        try self.addExpandedStackWithAlloc(&next_states, &seen, popped, alloc);
                    }
                },
                .literal => |lit| {
                    if (top.position < lit.len and byte == lit[top.position]) {
                        if (top.position + 1 < lit.len) {
                            const updated = try updateTopPositionAlloc(alloc, stack.frames, top.position + 1);
                            try addStackUniqueAlloc(&next_states, &seen, updated, alloc);
                        } else {
                            const popped = try copyWithoutTopAlloc(alloc, stack.frames);
                            try self.addExpandedStackWithAlloc(&next_states, &seen, popped, alloc);
                        }
                    }
                },
                else => {},
            }
        }

        return next_states;
    }

    /// Advance states reusing an existing seen hashmap (avoids hashmap allocation per byte)
    fn advanceStatesReusingSeen(
        self: *Engine,
        states: *StackSet,
        byte: u8,
        alloc: std.mem.Allocator,
        seen: *std.AutoHashMap(u64, void),
    ) !StackSet {
        var next_states = StackSet.init(alloc);
        errdefer next_states.deinit();

        for (states.stacks.items) |stack| {
            if (stack.frames.len == 0) continue;
            const top = stack.frames[stack.frames.len - 1];
            if (top.kind != .rule) continue;

            const rule = self.grammar.getRule(top.rule_id) orelse continue;
            switch (rule) {
                .char => |c| {
                    if (byte == c) {
                        const popped = try copyWithoutTopAlloc(alloc, stack.frames);
                        try self.addExpandedStackWithAlloc(&next_states, seen, popped, alloc);
                    }
                },
                .char_range => |r| {
                    if (byte >= r.start and byte <= r.end) {
                        const popped = try copyWithoutTopAlloc(alloc, stack.frames);
                        try self.addExpandedStackWithAlloc(&next_states, seen, popped, alloc);
                    }
                },
                .literal => |lit| {
                    if (top.position < lit.len and byte == lit[top.position]) {
                        if (top.position + 1 < lit.len) {
                            const updated = try updateTopPositionAlloc(alloc, stack.frames, top.position + 1);
                            try addStackUniqueAlloc(&next_states, seen, updated, alloc);
                        } else {
                            const popped = try copyWithoutTopAlloc(alloc, stack.frames);
                            try self.addExpandedStackWithAlloc(&next_states, seen, popped, alloc);
                        }
                    }
                },
                else => {},
            }
        }

        return next_states;
    }

    /// Final byte advance - allocates result with main allocator but uses arena for expansion
    fn advanceStatesFinal(
        self: *Engine,
        states: *StackSet,
        byte: u8,
        seen: *std.AutoHashMap(u64, void),
    ) !StackSet {
        var next_states = StackSet.init(self.allocator);
        errdefer next_states.deinit();

        // Use a local arena for expansion operations (internal allocations)
        var expand_arena = std.heap.ArenaAllocator.init(self.allocator);
        defer expand_arena.deinit();
        const expand_alloc = expand_arena.allocator();

        for (states.stacks.items) |stack| {
            if (stack.frames.len == 0) continue;
            const top = stack.frames[stack.frames.len - 1];
            if (top.kind != .rule) continue;

            const rule = self.grammar.getRule(top.rule_id) orelse continue;
            switch (rule) {
                .char => |c| {
                    if (byte == c) {
                        // Use arena for expansion, but copy result to main allocator
                        const popped = try copyWithoutTopAlloc(expand_alloc, stack.frames);
                        try self.addExpandedStackFinal(&next_states, seen, popped, expand_alloc);
                    }
                },
                .char_range => |r| {
                    if (byte >= r.start and byte <= r.end) {
                        const popped = try copyWithoutTopAlloc(expand_alloc, stack.frames);
                        try self.addExpandedStackFinal(&next_states, seen, popped, expand_alloc);
                    }
                },
                .literal => |lit| {
                    if (top.position < lit.len and byte == lit[top.position]) {
                        if (top.position + 1 < lit.len) {
                            // Literal position update - allocate with main allocator (result)
                            const updated = try updateTopPosition(self.allocator, stack.frames, top.position + 1);
                            try self.addStackUnique(&next_states, seen, updated);
                        } else {
                            const popped = try copyWithoutTopAlloc(expand_alloc, stack.frames);
                            try self.addExpandedStackFinal(&next_states, seen, popped, expand_alloc);
                        }
                    }
                },
                else => {},
            }
        }

        return next_states;
    }

    /// Add expanded stack - uses arena for expansion but copies results to main allocator
    fn addExpandedStackFinal(
        self: *Engine,
        next_states: *StackSet,
        seen: *std.AutoHashMap(u64, void),
        frames: []Frame,
        expand_alloc: std.mem.Allocator,
    ) !void {
        const expanded = try self.expandFromStackWithAlloc(frames, expand_alloc);
        // Don't deinit - arena handles it

        for (expanded.stacks.items) |stack| {
            // Copy to main allocator for the final result
            const copied = try self.allocator.dupe(Frame, stack.frames);
            try self.addStackUnique(next_states, seen, copied);
        }
    }

    fn addExpandedStackWithAlloc(
        self: *Engine,
        next_states: *StackSet,
        seen: *std.AutoHashMap(u64, void),
        frames: []Frame,
        alloc: std.mem.Allocator,
    ) !void {
        const expanded = try self.expandFromStackWithAlloc(frames, alloc);
        // Don't deinit - arena handles it

        for (expanded.stacks.items) |stack| {
            const copied = try alloc.dupe(Frame, stack.frames); // lint:ignore errdefer-alloc - arena allocator, freed atomically on deinit
            try addStackUniqueAlloc(next_states, seen, copied, alloc);
        }
    }

    fn expandFromStackWithAlloc(self: *Engine, frames: []Frame, alloc: std.mem.Allocator) !StackSet {
        var queue = std.ArrayList(Stack).empty;
        defer queue.deinit(alloc);

        var result = StackSet.init(alloc);
        errdefer result.deinit();

        var visited = std.AutoHashMap(u64, void).init(alloc);
        defer visited.deinit();

        try queue.append(alloc, .{ .frames = frames });

        while (queue.pop()) |stack| {
            const hash = hashStack(stack.frames);
            if (visited.contains(hash)) continue;
            try visited.put(hash, {});

            if (stack.frames.len == 0) {
                try result.append(stack.frames);
                continue;
            }

            const top = stack.frames[stack.frames.len - 1];

            if (top.kind == .repeat) {
                try self.expandRepeatWithAlloc(stack.frames, top.rule_id, &queue, alloc);
                continue;
            }

            const rule = self.grammar.getRule(top.rule_id) orelse continue;

            switch (rule) {
                .char, .char_range => {
                    const copied = try alloc.dupe(Frame, stack.frames);
                    try result.append(copied);
                },
                .literal => |lit| {
                    if (top.position >= lit.len) {
                        const popped = try copyWithoutTopAlloc(alloc, stack.frames);
                        try queue.append(alloc, .{ .frames = popped });
                    } else {
                        const copied = try alloc.dupe(Frame, stack.frames);
                        try result.append(copied);
                    }
                },
                .sequence => |seq| {
                    const expanded = try expandSequenceAlloc(alloc, stack.frames, seq);
                    try queue.append(alloc, .{ .frames = expanded });
                },
                .alternatives => |alts| {
                    for (alts) |alt| {
                        const expanded = try popAndPushAlloc(alloc, stack.frames, .{ .kind = .rule, .rule_id = alt, .position = 0 });
                        try queue.append(alloc, .{ .frames = expanded });
                    }
                },
                .optional => |inner| {
                    const skipped = try copyWithoutTopAlloc(alloc, stack.frames);
                    try queue.append(alloc, .{ .frames = skipped });

                    const taken = try popAndPushAlloc(alloc, stack.frames, .{ .kind = .rule, .rule_id = inner, .position = 0 });
                    try queue.append(alloc, .{ .frames = taken });
                },
                .star => |inner| {
                    const skipped = try copyWithoutTopAlloc(alloc, stack.frames);
                    try queue.append(alloc, .{ .frames = skipped });

                    const taken = try popAndPushTwoAlloc(
                        alloc,
                        stack.frames,
                        .{ .kind = .rule, .rule_id = inner, .position = 0 },
                        .{ .kind = .rule, .rule_id = top.rule_id, .position = 0 },
                    );
                    try queue.append(alloc, .{ .frames = taken });
                },
                .plus => |inner| {
                    const expanded = try popAndPushTwoAlloc(
                        alloc,
                        stack.frames,
                        .{ .kind = .rule, .rule_id = inner, .position = 0 },
                        .{ .kind = .repeat, .rule_id = inner, .position = 0 },
                    );
                    try queue.append(alloc, .{ .frames = expanded });
                },
                .reference => |ref| {
                    const expanded = try popAndPushAlloc(alloc, stack.frames, .{ .kind = .rule, .rule_id = ref, .position = 0 });
                    try queue.append(alloc, .{ .frames = expanded });
                },
                .end => {
                    const popped = try copyWithoutTopAlloc(alloc, stack.frames);
                    try queue.append(alloc, .{ .frames = popped });
                },
            }
        }

        return result;
    }

    fn expandRepeatWithAlloc(
        _: *Engine,
        frames: []Frame,
        inner: RuleId,
        queue: *std.ArrayList(Stack),
        alloc: std.mem.Allocator,
    ) !void {
        const skipped = try copyWithoutTopAlloc(alloc, frames);
        try queue.append(alloc, .{ .frames = skipped });

        const taken = try popAndPushTwoAlloc(
            alloc,
            frames,
            .{ .kind = .rule, .rule_id = inner, .position = 0 },
            .{ .kind = .repeat, .rule_id = inner, .position = 0 },
        );
        try queue.append(alloc, .{ .frames = taken });
    }

    pub fn advance(self: *Engine, token_text: []const u8) !void {
        if (token_text.len == 0) return;

        // Use arena allocator for ALL intermediate allocations
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const arena_alloc = arena.allocator();

        // Reuse a single hashmap across all bytes (clear between uses)
        var seen = std.AutoHashMap(u64, void).init(arena_alloc);

        // Clone initial states to arena
        var current = try self.cloneStatesWithAllocator(arena_alloc);

        // Process all bytes except the last one using arena
        for (token_text[0 .. token_text.len - 1]) |byte| {
            seen.clearRetainingCapacity();
            current = try self.advanceStatesReusingSeen(&current, byte, arena_alloc, &seen);
        }

        // Final byte: allocate result with real allocator, but still reuse seen map
        seen.clearRetainingCapacity();
        const final = try self.advanceStatesFinal(&current, token_text[token_text.len - 1], &seen);

        self.states.deinit();
        self.states = final;
    }

    fn advanceStates(self: *Engine, states: *StackSet, byte: u8) !StackSet {
        var next_states = StackSet.init(self.allocator);
        errdefer next_states.deinit();

        var seen = std.AutoHashMap(u64, void).init(self.allocator);
        defer seen.deinit();

        for (states.stacks.items) |stack| {
            if (stack.frames.len == 0) continue;
            const top = stack.frames[stack.frames.len - 1];
            if (top.kind != .rule) continue;

            const rule = self.grammar.getRule(top.rule_id) orelse continue;
            switch (rule) {
                .char => |c| {
                    if (byte == c) {
                        const popped = try copyWithoutTop(self.allocator, stack.frames);
                        try self.addExpandedStack(&next_states, &seen, popped);
                    }
                },
                .char_range => |r| {
                    if (byte >= r.start and byte <= r.end) {
                        const popped = try copyWithoutTop(self.allocator, stack.frames);
                        try self.addExpandedStack(&next_states, &seen, popped);
                    }
                },
                .literal => |lit| {
                    if (top.position < lit.len and byte == lit[top.position]) {
                        if (top.position + 1 < lit.len) {
                            const updated = try updateTopPosition(self.allocator, stack.frames, top.position + 1);
                            try self.addStackUnique(&next_states, &seen, updated);
                        } else {
                            const popped = try copyWithoutTop(self.allocator, stack.frames);
                            try self.addExpandedStack(&next_states, &seen, popped);
                        }
                    }
                },
                else => {},
            }
        }

        return next_states;
    }

    fn addExpandedStack(
        self: *Engine,
        next_states: *StackSet,
        seen: *std.AutoHashMap(u64, void),
        frames: []Frame,
    ) !void {
        var expanded = try self.expandFromStack(frames);
        defer expanded.deinit();

        for (expanded.stacks.items) |stack| {
            const copied = try self.allocator.dupe(Frame, stack.frames);
            try self.addStackUnique(next_states, seen, copied);
        }
    }

    fn addStackUnique(
        self: *Engine,
        next_states: *StackSet,
        seen: *std.AutoHashMap(u64, void),
        frames: []Frame,
    ) !void {
        const hash = hashStack(frames);
        if (seen.contains(hash)) {
            if (frames.len > 0) self.allocator.free(frames);
            return;
        }
        try seen.put(hash, {});
        try next_states.append(frames);
    }

    fn expandFromStack(self: *Engine, frames: []Frame) !StackSet {
        var queue = std.ArrayList(Stack).empty;
        defer queue.deinit(self.allocator);

        var result = StackSet.init(self.allocator);
        errdefer result.deinit();

        var visited = std.AutoHashMap(u64, void).init(self.allocator);
        defer visited.deinit();

        try queue.append(self.allocator, .{ .frames = frames });

        while (queue.pop()) |stack| {
            const hash = hashStack(stack.frames);
            if (visited.contains(hash)) {
                if (stack.frames.len > 0) self.allocator.free(stack.frames);
                continue;
            }
            try visited.put(hash, {});

            if (stack.frames.len == 0) {
                try result.append(stack.frames);
                continue;
            }

            const top = stack.frames[stack.frames.len - 1];

            if (top.kind == .repeat) {
                try self.expandRepeat(stack.frames, top.rule_id, &queue);
                continue;
            }

            const rule = self.grammar.getRule(top.rule_id) orelse {
                if (stack.frames.len > 0) self.allocator.free(stack.frames);
                continue;
            };

            switch (rule) {
                .char, .char_range => {
                    const copied = try self.allocator.dupe(Frame, stack.frames);
                    try result.append(copied);
                    if (stack.frames.len > 0) {
                        self.allocator.free(stack.frames);
                    }
                },
                .literal => |lit| {
                    if (top.position >= lit.len) {
                        const popped = try copyWithoutTop(self.allocator, stack.frames);
                        try queue.append(self.allocator, .{ .frames = popped });
                        self.allocator.free(stack.frames);
                    } else {
                        const copied = try self.allocator.dupe(Frame, stack.frames);
                        try result.append(copied);
                        if (stack.frames.len > 0) {
                            self.allocator.free(stack.frames);
                        }
                    }
                },
                .sequence => |seq| {
                    const expanded = try expandSequence(self.allocator, stack.frames, seq);
                    try queue.append(self.allocator, .{ .frames = expanded });
                    self.allocator.free(stack.frames);
                },
                .alternatives => |alts| {
                    for (alts) |alt| {
                        const expanded = try popAndPush(self.allocator, stack.frames, .{ .kind = .rule, .rule_id = alt, .position = 0 });
                        try queue.append(self.allocator, .{ .frames = expanded });
                    }
                    self.allocator.free(stack.frames);
                },
                .optional => |inner| {
                    const skipped = try copyWithoutTop(self.allocator, stack.frames);
                    try queue.append(self.allocator, .{ .frames = skipped });

                    const taken = try popAndPush(self.allocator, stack.frames, .{ .kind = .rule, .rule_id = inner, .position = 0 });
                    try queue.append(self.allocator, .{ .frames = taken });

                    self.allocator.free(stack.frames);
                },
                .star => |inner| {
                    const skipped = try copyWithoutTop(self.allocator, stack.frames);
                    try queue.append(self.allocator, .{ .frames = skipped });

                    const taken = try popAndPushTwo(
                        self.allocator,
                        stack.frames,
                        .{ .kind = .rule, .rule_id = inner, .position = 0 },
                        .{ .kind = .rule, .rule_id = top.rule_id, .position = 0 },
                    );
                    try queue.append(self.allocator, .{ .frames = taken });

                    self.allocator.free(stack.frames);
                },
                .plus => |inner| {
                    const expanded = try popAndPushTwo(
                        self.allocator,
                        stack.frames,
                        .{ .kind = .rule, .rule_id = inner, .position = 0 },
                        .{ .kind = .repeat, .rule_id = inner, .position = 0 },
                    );
                    try queue.append(self.allocator, .{ .frames = expanded });
                    self.allocator.free(stack.frames);
                },
                .reference => |ref| {
                    const expanded = try popAndPush(self.allocator, stack.frames, .{ .kind = .rule, .rule_id = ref, .position = 0 });
                    try queue.append(self.allocator, .{ .frames = expanded });
                    self.allocator.free(stack.frames);
                },
                .end => {
                    const popped = try copyWithoutTop(self.allocator, stack.frames);
                    try queue.append(self.allocator, .{ .frames = popped });
                    self.allocator.free(stack.frames);
                },
            }
        }

        return result;
    }

    fn expandRepeat(
        self: *Engine,
        frames: []Frame,
        inner: RuleId,
        queue: *std.ArrayList(Stack),
    ) !void {
        const skipped = try copyWithoutTop(self.allocator, frames);
        try queue.append(self.allocator, .{ .frames = skipped });

        const taken = try popAndPushTwo(
            self.allocator,
            frames,
            .{ .kind = .rule, .rule_id = inner, .position = 0 },
            .{ .kind = .repeat, .rule_id = inner, .position = 0 },
        );
        try queue.append(self.allocator, .{ .frames = taken });

        self.allocator.free(frames);
    }
};

fn tokenizerVocabSize(tokenizer: anytype) usize {
    const T = @TypeOf(tokenizer);
    switch (@typeInfo(T)) {
        .pointer => |info| {
            const Child = info.child;
            if (@hasDecl(Child, "getVocabSize")) {
                return tokenizer.getVocabSize();
            }
            if (@hasField(Child, "tokenizer_handle")) {
                return tokenizer.tokenizer_handle.getVocabSize();
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
    if (@hasField(T, "tokenizer_handle")) {
        return tokenizer.tokenizer_handle.getVocabSize();
    }
    if (@hasField(T, "vocab_size")) {
        return tokenizer.vocab_size;
    }

    return 0;
}

fn hashStack(frames: []const Frame) u64 {
    var hasher = std.hash.Wyhash.init(0);
    for (frames) |frame| {
        hasher.update(std.mem.asBytes(&frame.kind));
        hasher.update(std.mem.asBytes(&frame.rule_id));
        hasher.update(std.mem.asBytes(&frame.position));
    }
    return hasher.final();
}

fn copyWithoutTop(allocator: std.mem.Allocator, frames: []const Frame) ![]Frame {
    if (frames.len == 0) return &[_]Frame{};
    return allocator.dupe(Frame, frames[0 .. frames.len - 1]);
}

fn updateTopPosition(allocator: std.mem.Allocator, frames: []const Frame, pos: usize) ![]Frame {
    const copy = try allocator.dupe(Frame, frames);
    copy[copy.len - 1].position = pos;
    return copy;
}

fn popAndPush(allocator: std.mem.Allocator, frames: []const Frame, frame: Frame) ![]Frame {
    const base_len = if (frames.len == 0) 0 else frames.len - 1;
    var new_frames = try allocator.alloc(Frame, base_len + 1);
    if (base_len > 0) {
        @memcpy(new_frames[0..base_len], frames[0..base_len]);
    }
    new_frames[base_len] = frame;
    return new_frames;
}

fn popAndPushTwo(allocator: std.mem.Allocator, frames: []const Frame, first: Frame, second: Frame) ![]Frame {
    const base_len = if (frames.len == 0) 0 else frames.len - 1;
    var new_frames = try allocator.alloc(Frame, base_len + 2);
    if (base_len > 0) {
        @memcpy(new_frames[0..base_len], frames[0..base_len]);
    }
    new_frames[base_len] = second;
    new_frames[base_len + 1] = first;
    return new_frames;
}

fn expandSequence(allocator: std.mem.Allocator, frames: []const Frame, seq: []const RuleId) ![]Frame {
    const base_len = if (frames.len == 0) 0 else frames.len - 1;
    var new_frames = try allocator.alloc(Frame, base_len + seq.len);
    if (base_len > 0) {
        @memcpy(new_frames[0..base_len], frames[0..base_len]);
    }
    var idx: usize = 0;
    while (idx < seq.len) : (idx += 1) {
        const rule_id = seq[seq.len - 1 - idx];
        new_frames[base_len + idx] = .{ .kind = .rule, .rule_id = rule_id, .position = 0 };
    }
    return new_frames;
}

// Allocator-parameterized versions for arena allocation (no freeing needed)
fn copyWithoutTopAlloc(alloc: std.mem.Allocator, frames: []const Frame) ![]Frame {
    if (frames.len == 0) return &[_]Frame{};
    return alloc.dupe(Frame, frames[0 .. frames.len - 1]);
}

fn updateTopPositionAlloc(alloc: std.mem.Allocator, frames: []const Frame, pos: usize) ![]Frame {
    const copy = try alloc.dupe(Frame, frames);
    copy[copy.len - 1].position = pos;
    return copy;
}

fn popAndPushAlloc(alloc: std.mem.Allocator, frames: []const Frame, frame: Frame) ![]Frame {
    const base_len = if (frames.len == 0) 0 else frames.len - 1;
    var new_frames = try alloc.alloc(Frame, base_len + 1);
    if (base_len > 0) {
        @memcpy(new_frames[0..base_len], frames[0..base_len]);
    }
    new_frames[base_len] = frame;
    return new_frames;
}

fn popAndPushTwoAlloc(alloc: std.mem.Allocator, frames: []const Frame, first: Frame, second: Frame) ![]Frame {
    const base_len = if (frames.len == 0) 0 else frames.len - 1;
    var new_frames = try alloc.alloc(Frame, base_len + 2);
    if (base_len > 0) {
        @memcpy(new_frames[0..base_len], frames[0..base_len]);
    }
    new_frames[base_len] = second;
    new_frames[base_len + 1] = first;
    return new_frames;
}

fn expandSequenceAlloc(alloc: std.mem.Allocator, frames: []const Frame, seq: []const RuleId) ![]Frame {
    const base_len = if (frames.len == 0) 0 else frames.len - 1;
    var new_frames = try alloc.alloc(Frame, base_len + seq.len);
    if (base_len > 0) {
        @memcpy(new_frames[0..base_len], frames[0..base_len]);
    }
    var idx: usize = 0;
    while (idx < seq.len) : (idx += 1) {
        const rule_id = seq[seq.len - 1 - idx];
        new_frames[base_len + idx] = .{ .kind = .rule, .rule_id = rule_id, .position = 0 };
    }
    return new_frames;
}

fn addStackUniqueAlloc(
    next_states: *StackSet,
    seen: *std.AutoHashMap(u64, void),
    frames: []Frame,
    alloc: std.mem.Allocator,
) !void {
    const hash = hashStack(frames);
    if (seen.contains(hash)) return; // No free needed with arena
    try seen.put(hash, {});
    try next_states.stacks.append(alloc, .{ .frames = frames });
}

// CRITICAL TESTS: Whitespace handling with composite tokens

test "whitespace composite token handling" {
    const allocator = std.testing.allocator;

    var grammar = ast.Grammar.init(allocator);
    defer grammar.deinit();

    grammar.ws_rule = try ast.JsonRules.whitespace(&grammar);
    const key_literal = try grammar.addRule(.{ .literal = "key" });
    grammar.root_rule = try grammar.addRule(.{
        .sequence = try ast.dupeRuleIds(grammar.allocator(), &[_]RuleId{ grammar.ws_rule, key_literal }),
    });

    var engine = try Engine.init(allocator, &grammar);
    defer engine.deinit();

    const composite_token = " key";
    const is_valid = try engine.canAccept(composite_token);

    try std.testing.expect(is_valid);
}

test "whitespace optional before literal" {
    const allocator = std.testing.allocator;

    var grammar = ast.Grammar.init(allocator);
    defer grammar.deinit();

    grammar.ws_rule = try ast.JsonRules.whitespace(&grammar);
    const foo_literal = try grammar.addRule(.{ .literal = "foo" });
    const opt_ws = try grammar.addRule(.{ .optional = grammar.ws_rule });
    grammar.root_rule = try grammar.addRule(.{
        .sequence = try ast.dupeRuleIds(grammar.allocator(), &[_]RuleId{ opt_ws, foo_literal }),
    });

    var engine = try Engine.init(allocator, &grammar);
    defer engine.deinit();

    try std.testing.expect(try engine.canAccept("foo"));

    try engine.reset();
    try std.testing.expect(try engine.canAccept(" foo"));

    try engine.reset();
    try std.testing.expect(try engine.canAccept("  foo"));
}

test "minified json mode (no whitespace)" {
    const allocator = std.testing.allocator;

    var grammar = ast.Grammar.init(allocator);
    defer grammar.deinit();

    grammar.ws_rule = try ast.JsonRules.whitespace(&grammar);
    const key_literal = try grammar.addRule(.{ .literal = "\"key\"" });
    const value_literal = try grammar.addRule(.{ .literal = "\"value\"" });
    const colon = try grammar.addRule(.{ .char = ':' });
    const open = try grammar.addRule(.{ .char = '{' });
    const close = try grammar.addRule(.{ .char = '}' });

    grammar.root_rule = try grammar.addRule(.{
        .sequence = try ast.dupeRuleIds(grammar.allocator(), &[_]RuleId{ open, key_literal, colon, value_literal, close }),
    });

    var engine = try Engine.init(allocator, &grammar);
    defer engine.deinit();

    try std.testing.expect(try engine.canAccept("{\"key\":\"value\"}"));
}
