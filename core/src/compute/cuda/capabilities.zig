//! Explicit CUDA primitive capability map (stub backend).

pub const PrimitiveCapabilities = struct {
    linalg: bool = true,
    normalization: bool = false,
    activation: bool = false,
    softmax: bool = false,
    layout: bool = false,
    memory: bool = true,
    indexing: bool = false,
    quant_decode: bool = false,
    state_space: bool = false,
};

pub const support: PrimitiveCapabilities = .{};
