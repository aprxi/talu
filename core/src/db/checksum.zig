//! CRC32C (Castagnoli) checksum utilities for StoreFS.
//!
//! Uses the iSCSI polynomial (0x1EDC6F41) for CRC32C.

const std = @import("std");

/// Computes a CRC32C checksum over the provided bytes.
pub fn crc32c(bytes: []const u8) u32 {
    var crc = std.hash.crc.Crc32Iscsi.init();
    crc.update(bytes);
    return crc.final();
}

test "crc32c matches known vector" {
    const input = "123456789";
    try std.testing.expectEqual(@as(u32, 0xE3069283), crc32c(input));
}
