//! Global cache for compiled tree-sitter highlight queries.
//!
//! Highlight queries are compile-time constants (@embedFile .scm patterns).
//! Compiling them into TSQuery objects is expensive (S-expression parsing +
//! grammar validation). Since there are only 11 languages and queries never
//! change, we cache them permanently in a global array.
//!
//! Thread safety: Mutex protects lazy initialization. After initialization,
//! cached Query objects are immutable and safe to share across threads.

const std = @import("std");
const Language = @import("language.zig").Language;
const query_mod = @import("query.zig");
const Query = query_mod.Query;
const QueryError = query_mod.QueryError;

const NUM_LANGUAGES = @typeInfo(Language).@"enum".fields.len;

var cached_queries: [NUM_LANGUAGES]?Query = .{null} ** NUM_LANGUAGES;
var cache_mutex: std.Thread.Mutex = .{};

/// Get a cached compiled highlight query for the given language.
///
/// On first call per language, compiles and caches the query. Subsequent
/// calls return the cached query instantly. The returned pointer is owned
/// by the cache — callers must NOT call deinit on it.
pub fn getHighlightQuery(language: Language) QueryError!*const Query {
    const idx = @intFromEnum(language);

    cache_mutex.lock();
    defer cache_mutex.unlock();

    if (cached_queries[idx]) |*q| {
        return q;
    }

    const query_source = language.highlightQuery();
    cached_queries[idx] = try Query.init(language, query_source);
    return &cached_queries[idx].?;
}

/// Release all cached queries. For use in test cleanup or process shutdown.
pub fn cleanupQueryCache() void {
    cache_mutex.lock();
    defer cache_mutex.unlock();

    for (&cached_queries) |*slot| {
        if (slot.*) |*q| {
            q.deinit();
            slot.* = null;
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

test "getHighlightQuery returns same pointer on repeated calls" {
    const q1 = try getHighlightQuery(.python);
    const q2 = try getHighlightQuery(.python);
    try std.testing.expectEqual(@intFromPtr(q1), @intFromPtr(q2));
    cleanupQueryCache();
}

test "getHighlightQuery works for all languages" {
    inline for (@typeInfo(Language).@"enum".fields) |field| {
        const lang: Language = @enumFromInt(field.value);
        const q = try getHighlightQuery(lang);
        try std.testing.expect(q.captureCount() > 0);
    }
    cleanupQueryCache();
}

test "cleanupQueryCache allows re-initialization" {
    const q1 = try getHighlightQuery(.javascript);
    const count1 = q1.captureCount();
    cleanupQueryCache();

    // After cleanup, next call re-compiles
    const q2 = try getHighlightQuery(.javascript);
    try std.testing.expectEqual(count1, q2.captureCount());
    cleanupQueryCache();
}
