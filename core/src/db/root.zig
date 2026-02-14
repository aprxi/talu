//! TaluDB - append-only columnar storage engine.
//!
//! This is the single entry point for the db module. All external code should
//! import from here.
//!
//! ## Public API
//!
//! - `types` - On-disk ABI format definitions (block headers, column descriptors, enums)
//! - `checksum` - CRC32C hashing
//! - `lock` - File locking primitives
//! - `block_writer` - Block serialization (header + columns + arena + directory)
//! - `block_reader` - Block deserialization with jump reads
//! - `wal` - Write-ahead log for crash recovery
//! - `writer` - Namespace write path (WAL + in-memory buffer + block flush)
//! - `reader` - Namespace read path (manifest + current file)
//! - `manifest` - Sealed segment index (JSON)
//! - `blob` - Blob subsystem (content-addressable store + offline GC)
//! - `vector` - Vector search domain (storage, search, indexing)
//! - `table` - Table storage domain (chat, future structured storage)

// =============================================================================
// Public API
// =============================================================================

/// On-disk ABI format definitions.
pub const types = @import("types.zig");

/// CRC32C hashing.
pub const checksum = @import("checksum.zig");

/// Bloom filters for fast negative lookups.
pub const bloom = @import("bloom.zig");

/// File locking primitives.
pub const lock = @import("lock.zig");

/// Block serialization.
pub const block_writer = @import("block_writer.zig");

/// Block deserialization with jump reads.
pub const block_reader = @import("block_reader.zig");

/// Write-ahead log for crash recovery.
pub const wal = @import("wal.zig");

/// Namespace write path (WAL + buffer + block flush).
pub const writer = @import("writer.zig");

/// Sealed segment manifest.
pub const manifest = @import("manifest.zig");

/// Blob subsystem (content-addressable store + offline GC).
pub const blob = @import("blob/root.zig");

/// Namespace read path (manifest + current file).
pub const reader = @import("reader.zig");

/// Vector search domain (storage, search, indexing).
pub const vector = @import("vector/root.zig");

/// Table storage domain (session-scoped item persistence).
pub const table = @import("table/root.zig");

// Re-export behavioral types for cross-module use
pub const BlockBuilder = block_writer.BlockBuilder;
pub const BlockReader = block_reader.BlockReader;
pub const WalWriter = wal.WalWriter;
pub const WalIterator = wal.WalIterator;
pub const Writer = writer.Writer;
pub const Durability = writer.Durability;
pub const Manifest = manifest.Manifest;
pub const BlobStore = blob.BlobStore;
pub const BlobRef = blob.BlobRef;
pub const BlobReadStream = blob.BlobReadStream;
pub const BlobSweepStats = blob.SweepStats;
pub const BlobSweepOptions = blob.SweepOptions;
pub const Reader = reader.Reader;
pub const TableAdapter = table.sessions.TableAdapter;
pub const VectorAdapter = vector.store.VectorAdapter;
pub const BloomFilter = bloom.BloomFilter;
pub const BloomCache = bloom.BloomCache;
