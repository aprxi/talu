//! ABI Compatibility: Compile-Time Struct Size Validation
//!
//! This module validates that C API struct sizes match expected values at compile time.
//! When struct layouts change, compilation fails with a clear error message.
//!
//! To update expected sizes after intentional struct changes:
//! 1. Run `zig build` - it will show the new actual size in the error
//! 2. Update the expected size here
//! 3. Update bindings/python/talu/_abi.py with the same value
//!
//! This ensures Zig core and Python bindings stay in sync.
//!
//! Supported architectures: x86_64, aarch64 (arm64)
//! Both use LP64 data model with 8-byte pointers, so sizes are identical.

const std = @import("std");
const capi_boundary = @import("../responses/capi_boundary.zig");
const responses = @import("responses/root.zig");
const capi_types = @import("types.zig");

// Ensure 64-bit architecture (x86_64 or aarch64)
comptime {
    if (@sizeOf(usize) != 8) {
        @compileError("ABI validation only supports 64-bit architectures (x86_64, aarch64)");
    }
}

// =============================================================================
// Expected Sizes (64-bit LP64: x86_64, aarch64/arm64)
// =============================================================================
// Both x86_64 and aarch64 use 8-byte pointers with the same struct layouts.
// Update these when struct layouts change intentionally.
// Python bindings/python/talu/_abi.py must have identical values.

pub const EXPECTED_SIZES = struct {
    // Router/generation structs
    pub const RouterGenerateConfig = 152; // Added external vision payload pointers.
    pub const CToolCallRef = 24;
    pub const CLogitBiasEntry = 8;
    pub const CGenerateVisionImage = 48;

    // Conversation/Items structs (from responses/types.zig)
    pub const CItem = 24; // Fixed: was 40
    pub const CMessageItem = 24; // Fixed: was 40
    pub const CFunctionCallItem = 32; // Fixed: was 48
    pub const CFunctionCallOutputItem = 40; // Fixed: was 24
    pub const CReasoningItem = 32; // Fixed: was 24
    pub const CItemReferenceItem = 8; // Fixed: was 16
    pub const CContentPart = 56; // Fixed: was 40

    // Spec structs (from types.zig)
    pub const TaluModelSpec = 88;
    pub const TaluCapabilities = 48;

    // Backend info struct (from responses/engine.zig)
    pub const CModelInfo = 56;
};

// =============================================================================
// Compile-Time Assertions
// =============================================================================

fn assertSize(comptime name: []const u8, comptime T: type, comptime expected: usize) void {
    const actual = @sizeOf(T);
    if (actual != expected) {
        @compileError(std.fmt.comptimePrint(
            "ABI size mismatch for {s}: expected {d}, got {d}. " ++
                "Update core/src/capi/abi.zig and bindings/python/talu/_abi.py",
            .{ name, expected, actual },
        ));
    }
}

comptime {
    // Router/generation structs
    assertSize("RouterGenerateConfig", capi_boundary.CGenerateConfig, EXPECTED_SIZES.RouterGenerateConfig);
    assertSize("CToolCallRef", capi_boundary.CToolCallRef, EXPECTED_SIZES.CToolCallRef);
    assertSize("CLogitBiasEntry", capi_boundary.CLogitBiasEntry, EXPECTED_SIZES.CLogitBiasEntry);
    assertSize("CGenerateVisionImage", capi_boundary.CGenerateVisionImage, EXPECTED_SIZES.CGenerateVisionImage);

    // Conversation/Items structs
    assertSize("CItem", responses.CItem, EXPECTED_SIZES.CItem);
    assertSize("CMessageItem", responses.CMessageItem, EXPECTED_SIZES.CMessageItem);
    assertSize("CFunctionCallItem", responses.CFunctionCallItem, EXPECTED_SIZES.CFunctionCallItem);
    assertSize("CFunctionCallOutputItem", responses.CFunctionCallOutputItem, EXPECTED_SIZES.CFunctionCallOutputItem);
    assertSize("CReasoningItem", responses.CReasoningItem, EXPECTED_SIZES.CReasoningItem);
    assertSize("CItemReferenceItem", responses.CItemReferenceItem, EXPECTED_SIZES.CItemReferenceItem);
    assertSize("CContentPart", responses.CContentPart, EXPECTED_SIZES.CContentPart);

    // Spec structs
    assertSize("TaluModelSpec", capi_types.TaluModelSpec, EXPECTED_SIZES.TaluModelSpec);
    assertSize("TaluCapabilities", capi_types.TaluCapabilities, EXPECTED_SIZES.TaluCapabilities);

    // Backend info struct
    assertSize("CModelInfo", responses.CModelInfo, EXPECTED_SIZES.CModelInfo);
}

test "abi sizes are validated at comptime" {
    // This test exists to ensure abi.zig is compiled during `zig build test`.
    // The actual validation happens at compile time via comptime blocks above.
}
