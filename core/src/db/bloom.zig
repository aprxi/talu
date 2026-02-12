//! Bloom filter implementation for fast negative lookups.
//!
//! Bloom filters enable O(1) "definitely not here" checks by maintaining
//! a probabilistic bit set. A positive result may be a false positive,
//! but a negative result is always correct.
//!
//! Usage:
//! - Build: call add() for each item during block flush
//! - Query: call mayContain() before scanning block
//!
//! Parameters tuned for document storage:
//! - ~1KB per 1000 items
//! - ~1% false positive rate
//! - 7 hash functions (optimal for this configuration)

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Bloom filter with configurable size.
pub const BloomFilter = struct {
    bits: []u8,
    num_hashes: u8,
    allocator: Allocator,

    /// Create a new bloom filter sized for the expected item count.
    /// Uses ~10 bits per item for ~1% false positive rate.
    pub fn init(allocator: Allocator, expected_items: usize) !BloomFilter {
        // Calculate optimal size: m = -n * ln(p) / (ln(2))^2
        // For p = 0.01 (1%): m ≈ 9.6n bits
        // Round up to bytes, minimum 64 bytes
        const bits_needed = @max(expected_items * 10, 512);
        const bytes_needed = (bits_needed + 7) / 8;
        const size = @max(bytes_needed, 64);

        const bits = try allocator.alloc(u8, size);
        @memset(bits, 0);

        return .{
            .bits = bits,
            .num_hashes = 7, // Optimal for 10 bits/item
            .allocator = allocator,
        };
    }

    /// Create a bloom filter from existing bytes (for deserialization).
    pub fn initFromBytes(allocator: Allocator, data: []const u8) !BloomFilter {
        const bits = try allocator.dupe(u8, data);
        return .{
            .bits = bits,
            .num_hashes = 7,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BloomFilter) void {
        self.allocator.free(self.bits);
    }

    /// Add an item to the bloom filter.
    pub fn add(self: *BloomFilter, item: u64) void {
        const bit_count = self.bits.len * 8;
        var hash = item;

        for (0..self.num_hashes) |i| {
            // Double hashing: h(i) = h1 + i * h2
            hash = mixHash(hash, @intCast(i));
            const bit_index = hash % bit_count;
            const byte_index = bit_index / 8;
            const bit_offset: u3 = @intCast(bit_index % 8);
            self.bits[byte_index] |= @as(u8, 1) << bit_offset;
        }
    }

    /// Check if an item may be in the bloom filter.
    /// Returns false if definitely not present, true if possibly present.
    pub fn mayContain(self: *const BloomFilter, item: u64) bool {
        const bit_count = self.bits.len * 8;
        var hash = item;

        for (0..self.num_hashes) |i| {
            hash = mixHash(hash, @intCast(i));
            const bit_index = hash % bit_count;
            const byte_index = bit_index / 8;
            const bit_offset: u3 = @intCast(bit_index % 8);
            if ((self.bits[byte_index] & (@as(u8, 1) << bit_offset)) == 0) {
                return false; // Definitely not present
            }
        }
        return true; // Possibly present
    }

    /// Get the raw bytes for serialization.
    pub fn toBytes(self: *const BloomFilter) []const u8 {
        return self.bits;
    }

    /// Get the number of bits set (for statistics).
    pub fn popCount(self: *const BloomFilter) usize {
        var count: usize = 0;
        for (self.bits) |byte| {
            count += @popCount(byte);
        }
        return count;
    }

    /// Estimate false positive rate based on current fill.
    pub fn estimateFalsePositiveRate(self: *const BloomFilter) f64 {
        const total_bits: f64 = @floatFromInt(self.bits.len * 8);
        const set_bits: f64 = @floatFromInt(self.popCount());
        const fill_ratio = set_bits / total_bits;
        // FPR ≈ fill_ratio^k
        return std.math.pow(f64, fill_ratio, @floatFromInt(self.num_hashes));
    }
};

/// Mix hash with seed using wyhash-style mixing.
fn mixHash(hash: u64, seed: u64) u64 {
    var h = hash;
    h ^= seed *% 0x9e3779b97f4a7c15;
    h ^= h >> 30;
    h *%= 0xbf58476d1ce4e5b9;
    h ^= h >> 27;
    h *%= 0x94d049bb133111eb;
    h ^= h >> 31;
    return h;
}

/// Compute hash for a string (uses wyhash).
pub fn hashString(s: []const u8) u64 {
    return std.hash.Wyhash.hash(0, s);
}

// =============================================================================
// Block-Level Bloom Filter Cache
// =============================================================================

/// Cache of bloom filters per block, keyed by (file_path_hash, block_offset).
/// Used by adapters to skip blocks during point lookups.
pub const BloomCache = struct {
    filters: std.AutoHashMap(BlockKey, BloomFilter),
    allocator: Allocator,

    const BlockKey = struct {
        file_hash: u64,
        block_offset: u64,
    };

    pub fn init(allocator: Allocator) BloomCache {
        return .{
            .filters = std.AutoHashMap(BlockKey, BloomFilter).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BloomCache) void {
        var it = self.filters.valueIterator();
        while (it.next()) |filter| {
            var f = filter.*;
            f.deinit();
        }
        self.filters.deinit();
    }

    /// Get or create a bloom filter for a block.
    pub fn getOrCreate(
        self: *BloomCache,
        file_path: []const u8,
        block_offset: u64,
        expected_items: usize,
    ) !*BloomFilter {
        const key = BlockKey{
            .file_hash = hashString(file_path),
            .block_offset = block_offset,
        };

        const gop = try self.filters.getOrPut(key);
        if (!gop.found_existing) {
            gop.value_ptr.* = try BloomFilter.init(self.allocator, expected_items);
        }
        return gop.value_ptr;
    }

    /// Check if an item may be in a specific block.
    pub fn mayContainInBlock(
        self: *const BloomCache,
        file_path: []const u8,
        block_offset: u64,
        item_hash: u64,
    ) ?bool {
        const key = BlockKey{
            .file_hash = hashString(file_path),
            .block_offset = block_offset,
        };

        if (self.filters.get(key)) |filter| {
            return filter.mayContain(item_hash);
        }
        return null; // No bloom filter cached for this block
    }
};

// =============================================================================
// Tests
// =============================================================================

test "BloomFilter basic operations" {
    const allocator = std.testing.allocator;

    var bloom = try BloomFilter.init(allocator, 100);
    defer bloom.deinit();

    // Add some items
    bloom.add(12345);
    bloom.add(67890);
    bloom.add(11111);

    // Items added should be found
    try std.testing.expect(bloom.mayContain(12345));
    try std.testing.expect(bloom.mayContain(67890));
    try std.testing.expect(bloom.mayContain(11111));

    // Item not added should (usually) not be found
    // Note: There's a small chance of false positive
    var false_positives: usize = 0;
    for (0..1000) |i| {
        const test_val = @as(u64, i) + 100000;
        if (bloom.mayContain(test_val)) {
            false_positives += 1;
        }
    }
    // Expect < 5% false positives (generous margin for small filter)
    try std.testing.expect(false_positives < 50);
}

test "BloomFilter serialization round-trip" {
    const allocator = std.testing.allocator;

    var bloom1 = try BloomFilter.init(allocator, 100);
    defer bloom1.deinit();

    bloom1.add(12345);
    bloom1.add(67890);

    const bytes = bloom1.toBytes();
    var bloom2 = try BloomFilter.initFromBytes(allocator, bytes);
    defer bloom2.deinit();

    try std.testing.expect(bloom2.mayContain(12345));
    try std.testing.expect(bloom2.mayContain(67890));
}

test "BloomFilter sizing" {
    const allocator = std.testing.allocator;

    // Small filter
    var small = try BloomFilter.init(allocator, 10);
    defer small.deinit();
    try std.testing.expect(small.bits.len >= 64); // Minimum size

    // Large filter
    var large = try BloomFilter.init(allocator, 10000);
    defer large.deinit();
    try std.testing.expect(large.bits.len >= 12500); // ~10 bits per item
}

test "BloomCache basic operations" {
    const allocator = std.testing.allocator;

    var cache = BloomCache.init(allocator);
    defer cache.deinit();

    // Get or create bloom filter
    const filter = try cache.getOrCreate("/path/to/file.talu", 0x1000, 100);
    filter.add(hashString("doc-123"));
    filter.add(hashString("doc-456"));

    // Check via cache
    const result1 = cache.mayContainInBlock("/path/to/file.talu", 0x1000, hashString("doc-123"));
    try std.testing.expect(result1 != null);
    try std.testing.expect(result1.?);

    // Check non-existent block
    const result2 = cache.mayContainInBlock("/other/path.talu", 0x2000, hashString("doc-123"));
    try std.testing.expect(result2 == null);
}

test "hashString is deterministic" {
    const h1 = hashString("test-document");
    const h2 = hashString("test-document");
    const h3 = hashString("different-document");

    try std.testing.expectEqual(h1, h2);
    try std.testing.expect(h1 != h3);
}
