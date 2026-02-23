//! Explicit Metal primitive capability map.

pub const PrimitiveCapabilities = struct {
    linalg: bool = true,
    normalization: bool = true,
    activation: bool = true,
    softmax: bool = true,
    layout: bool = true,
    memory: bool = true,
    indexing: bool = true,
    quant_decode: bool = true,
    state_space: bool = true,
};

pub const support: PrimitiveCapabilities = .{};

test "metal primitive capabilities advertise state_space support" {
    try @import("std").testing.expect(support.state_space);
}
