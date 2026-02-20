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
    state_space: bool = false,
};

pub const support: PrimitiveCapabilities = .{};
