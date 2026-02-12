//! Fast math operations with SIMD acceleration.
//!
//! Provides optimized implementations of activation functions (SiLU, GELU, ReLU),
//! normalization (RMSNorm, LayerNorm), softmax, and RoPE position encoding.

pub const simd = @import("../../simd/root.zig");

const _fast_math = @import("fast_math.zig");
const _activations = @import("activations.zig");
const _elementwise = @import("elementwise.zig");
const _softmax = @import("softmax.zig");
const _normalization = @import("normalization.zig");
const _rope = @import("rope.zig");

// Fast math
pub const fastExp = _fast_math.fastExp;
pub const fastExpScalar = _fast_math.fastExpScalar;

// Activations
pub const siluScalarReference = _activations.siluScalarReference;
pub const siluContiguous = _activations.siluContiguous;
pub const geluContiguous = _activations.geluContiguous;
pub const reluContiguous = _activations.reluContiguous;
pub const sigmoidContiguous = _activations.sigmoidContiguous;
pub const tanhContiguous = _activations.tanhContiguous;

// Element-wise ops
pub const addContiguous = _elementwise.addContiguous;
pub const subContiguous = _elementwise.subContiguous;
pub const mulContiguous = _elementwise.mulContiguous;
pub const divContiguous = _elementwise.divContiguous;

// Softmax
pub const softmaxContiguous = _softmax.softmaxContiguous;
pub const softmaxMaskedInPlaceWithMax = _softmax.softmaxMaskedInPlaceWithMax;

// Normalization
pub const rmsnormContiguous = _normalization.rmsnormContiguous;
pub const layerNormContiguous = _normalization.layerNormContiguous;
pub const rmsnormInPlaceWeightTensor = _normalization.rmsnormInPlaceWeightTensor;

// RoPE
pub const ropeFillCosSinCombinedStrided = _rope.ropeFillCosSinCombinedStrided;
pub const applyRopeRotationStrided = _rope.applyRopeRotationStrided;
pub const RoPE = _rope.RoPE;

// Reference test implementations
test {
    _ = _fast_math;
    _ = _activations;
    _ = _elementwise;
    _ = _softmax;
    _ = _normalization;
    _ = _rope;
}
