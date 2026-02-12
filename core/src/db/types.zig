//! ABI-stable binary layouts for StoreFS blocks.
//!
//! These structs define the on-disk format and must remain layout-stable.

const std = @import("std");

pub const MagicValues = struct {
    pub const BLOCK: u32 = 0x554C4154; // "TALU"
    pub const FOOTER: u32 = 0x4F4F4654; // "TFOO"
    pub const WAL: u32 = 0x314C4157; // "WAL1"
    pub const WAL_BATCH: u32 = 0x424C4157; // "WALB"
};

pub const BlockFlags = packed struct(u16) {
    has_ts_range: bool,
    has_arena: bool,
    has_primary_vector_hints: bool,
    blob_encoding_zstd: bool,
    _reserved: u12 = 0,
};

pub const BlockHeader = extern struct {
    magic: u32,
    version: u16,
    header_len: u16,
    flags: u16, // Use @bitCast from BlockFlags
    schema_id: u16,
    row_count: u32,
    block_len: u32,
    crc32c: u32,
    coldir_off: u32,
    coldir_len: u32,
    arena_off: u32,
    arena_len: u32,
    min_ts: i64,
    max_ts: i64,
    primary_centroid_id: u32,
    primary_vec_dims: u16,
    primary_vec_type: u8,
    _rsvd: u8,
};

pub const FooterTrailer = extern struct {
    magic: u32,
    version: u16,
    flags: u16,
    footer_len: u32,
    footer_crc32c: u32,
    segment_crc32c: u32,
    reserved: [12]u8,
};

/// Footer block index entry for O(1) block discovery in sealed segments.
/// Layout: 16 bytes, packed for efficient storage in footer payload.
pub const FooterBlockEntry = extern struct {
    /// Absolute byte offset of block within the segment file.
    block_off: u64,
    /// Total length of the block in bytes.
    block_len: u32,
    /// Schema ID of this block (for filtering without header reads).
    schema_id: u16,
    /// Reserved for future use (alignment padding).
    _reserved: u16 = 0,
};

pub const ColumnShape = enum(u8) { SCALAR = 1, VECTOR = 2, VARBYTES = 3 };

pub const PhysicalType = enum(u8) {
    U8 = 0,
    U16 = 1,
    U32 = 2,
    U64 = 3,
    I8 = 4,
    I16 = 5,
    I32 = 6,
    I64 = 7,
    F32 = 8,
    F16 = 9,
    BF16 = 10,
    F64 = 11,
    BINARY = 20,
};

pub const Encoding = enum(u8) { PLAIN = 0, RAW = 10, ZSTD = 11, KVBUF = 12 };

pub const ColumnDesc = extern struct {
    column_id: u32,
    shape: u8, // ColumnShape
    phys_type: u8, // PhysicalType
    encoding: u8, // Encoding
    dims: u16,
    _rsvd: u16,
    data_off: u32,
    data_len: u32,
    offsets_off: u32,
    lengths_off: u32,
    _pad: [4]u8 = [_]u8{0} ** 4,
};

comptime {
    std.debug.assert(@sizeOf(BlockHeader) == 64);
    std.debug.assert(@sizeOf(FooterTrailer) == 32);
    std.debug.assert(@sizeOf(ColumnDesc) == 32);
    std.debug.assert(@sizeOf(FooterBlockEntry) == 16);
}
