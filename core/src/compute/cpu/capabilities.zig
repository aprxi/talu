//! Explicit CPU primitive capability map.

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
