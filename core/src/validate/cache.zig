//! Grammar cache - LRU cache for compiled grammars and token masks.

const std = @import("std");
const ast = @import("ast.zig");
const schema_mod = @import("schema.zig");

const Grammar = ast.Grammar;

pub const CacheKey = u64;

pub fn computeKey(schema_json: []const u8, config: schema_mod.CompilerConfig) CacheKey {
    var hasher = std.hash.Wyhash.init(0);
    hasher.update(schema_json);
    hasher.update(std.mem.asBytes(&config.max_exact_span));
    hasher.update(std.mem.asBytes(&config.max_exact_value));
    hasher.update(std.mem.asBytes(&config.max_depth));
    return hasher.final();
}

// =============================================================================
// Global Token Mask Cache
// =============================================================================
// Caches token masks indexed by (grammar_key XOR state_hash XOR vocab_size).
// This allows different Engine instances using the same grammar to share masks.

const MaskCacheEntry = struct {
    mask_data: []u64,
    vocab_size: usize,
};

pub const GlobalMaskCache = struct {
    allocator: std.mem.Allocator,
    entries: std.AutoHashMap(u64, MaskCacheEntry),
    max_entries: usize,
    mutex: std.Thread.Mutex,
    hits: usize = 0,
    misses: usize = 0,

    pub fn init(allocator: std.mem.Allocator, max_entries: usize) GlobalMaskCache {
        return .{
            .allocator = allocator,
            .entries = std.AutoHashMap(u64, MaskCacheEntry).init(allocator),
            .max_entries = max_entries,
            .mutex = .{},
        };
    }

    pub fn deinit(self: *GlobalMaskCache) void {
        var iter = self.entries.valueIterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.mask_data);
        }
        self.entries.deinit();
    }

    pub fn computeMaskKey(grammar_key: CacheKey, state_hash: u64, vocab_size: usize) u64 {
        const prime: u64 = 0x9e3779b97f4a7c15;
        return grammar_key ^ (state_hash *% prime) ^ (@as(u64, vocab_size) *% (prime >> 1));
    }

    pub fn get(self: *GlobalMaskCache, key: u64, vocab_size: usize) ?[]const u64 {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.entries.get(key)) |entry| {
            if (entry.vocab_size == vocab_size) {
                self.hits += 1;
                return entry.mask_data;
            }
        }
        self.misses += 1;
        return null;
    }

    pub fn put(self: *GlobalMaskCache, key: u64, mask_data: []const u64, vocab_size: usize) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Check if already exists
        if (self.entries.contains(key)) return;

        // Evict if at capacity
        if (self.entries.count() >= self.max_entries) {
            // Simple eviction: remove first entry found
            var iter = self.entries.keyIterator();
            if (iter.next()) |first_key| {
                if (self.entries.fetchRemove(first_key.*)) |removed| {
                    self.allocator.free(removed.value.mask_data);
                }
            }
        }

        // Store a copy
        const copied = try self.allocator.dupe(u64, mask_data);
        try self.entries.put(key, .{
            .mask_data = copied,
            .vocab_size = vocab_size,
        });
    }
};

// Thread-safe: protected by global_mask_cache_mutex
var global_mask_cache: ?GlobalMaskCache = null;
var global_mask_cache_mutex: std.Thread.Mutex = .{};

/// Get the global mask cache. Uses page allocator to avoid test allocator leak issues.
/// The allocator parameter is ignored - global caches use their own allocator.
pub fn getGlobalMaskCache(_: std.mem.Allocator) *GlobalMaskCache {
    global_mask_cache_mutex.lock();
    defer global_mask_cache_mutex.unlock();

    if (global_mask_cache == null) {
        // Use page allocator for global cache to avoid test allocator leak detection
        global_mask_cache = GlobalMaskCache.init(std.heap.page_allocator, 256);
    }
    return &global_mask_cache.?;
}

const CacheEntry = struct {
    grammar: Grammar,
    last_access: i64,
};

pub const GrammarCache = struct {
    allocator: std.mem.Allocator,
    entries: std.AutoHashMap(CacheKey, CacheEntry),
    max_entries: usize,
    mutex: std.Thread.Mutex,

    pub fn init(allocator: std.mem.Allocator, max_entries: usize) GrammarCache {
        return .{
            .allocator = allocator,
            .entries = std.AutoHashMap(CacheKey, CacheEntry).init(allocator),
            .max_entries = max_entries,
            .mutex = .{},
        };
    }

    pub fn deinit(self: *GrammarCache) void {
        var iter = self.entries.valueIterator();
        while (iter.next()) |entry| {
            entry.grammar.deinit();
        }
        self.entries.deinit();
    }

    pub fn getOrCompile(
        self: *GrammarCache,
        schema_json: []const u8,
        config: schema_mod.CompilerConfig,
    ) !*const Grammar {
        const key = computeKey(schema_json, config);

        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.entries.getPtr(key)) |entry| {
            entry.last_access = std.time.milliTimestamp();
            return &entry.grammar;
        }

        if (self.entries.count() >= self.max_entries) {
            self.evictLRU();
        }

        const grammar = try schema_mod.compile(self.allocator, schema_json, config);

        try self.entries.put(key, .{
            .grammar = grammar,
            .last_access = std.time.milliTimestamp(),
        });

        return &self.entries.getPtr(key).?.grammar;
    }

    fn evictLRU(self: *GrammarCache) void {
        var oldest_key: ?CacheKey = null;
        var oldest_time: i64 = std.math.maxInt(i64);

        var iter = self.entries.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.last_access < oldest_time) {
                oldest_time = entry.value_ptr.last_access;
                oldest_key = entry.key_ptr.*;
            }
        }

        if (oldest_key) |key| {
            if (self.entries.fetchRemove(key)) |removed| {
                var removed_value = removed.value;
                removed_value.grammar.deinit();
            }
        }
    }

    pub fn clear(self: *GrammarCache) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        var iter = self.entries.valueIterator();
        while (iter.next()) |entry| {
            entry.grammar.deinit();
        }
        self.entries.clearRetainingCapacity();
    }
};

// Thread-safe: protected by global_cache_mutex
var global_cache: ?GrammarCache = null;
var global_cache_mutex: std.Thread.Mutex = .{};

/// Get the global grammar cache. Uses page allocator to avoid test allocator leak issues.
/// The allocator parameter is ignored - global caches use their own allocator.
pub fn getGlobalCache(_: std.mem.Allocator) *GrammarCache {
    global_cache_mutex.lock();
    defer global_cache_mutex.unlock();

    if (global_cache == null) {
        // Use page allocator for global cache to avoid test allocator leak detection
        global_cache = GrammarCache.init(std.heap.page_allocator, 64);
    }
    return &global_cache.?;
}

/// Cleanup global caches - for use in tests to avoid leak detection failures.
/// In production, these caches live for the lifetime of the process.
pub fn cleanupGlobalCaches() void {
    {
        global_mask_cache_mutex.lock();
        defer global_mask_cache_mutex.unlock();
        if (global_mask_cache) |*c| {
            c.deinit();
            global_mask_cache = null;
        }
    }
    {
        global_cache_mutex.lock();
        defer global_cache_mutex.unlock();
        if (global_cache) |*c| {
            c.deinit();
            global_cache = null;
        }
    }
}

test "cache hit returns same grammar" {
    const allocator = std.testing.allocator;
    var cache = GrammarCache.init(allocator, 10);
    defer cache.deinit();

    const schema = \\{"type": "object", "properties": {"name": {"type": "string"}}}
    ;

    const g1 = try cache.getOrCompile(schema, .{});
    const g2 = try cache.getOrCompile(schema, .{});

    try std.testing.expectEqual(g1, g2);
}
