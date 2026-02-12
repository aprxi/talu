//! Core ops entrypoint
//!
//! This file is the "table of contents" for `src/compute/ops/`.
//! Keep it small and ordered so readers can discover where functionality lives.

// === Stride-aware ops for C API (TensorView-based) ===
pub const tensor_view = @import("tensor_view.zig");
pub const activation = @import("activation.zig");
pub const norm = @import("norm_primitives.zig");
pub const shape = @import("shape.zig");
pub const attention = @import("attn_primitives.zig");
pub const creation = @import("creation.zig");
pub const reduce = @import("reduce_primitives.zig");
pub const dtype_convert = @import("dtype_convert.zig");
pub const padding = @import("padding.zig");
pub const index_ops = @import("index_ops.zig");
pub const kv_cache = @import("kv_cache.zig");
pub const dot_product = @import("dot_product.zig");

// === Quantized ops ===
pub const mxfp4 = @import("mxfp4.zig");
pub const quant = @import("../quant/root.zig");
pub const grouped_affine_quant = quant.grouped_affine;

// === Legacy internal ops (Tensor-based, for graph executor) ===
pub const matmul = @import("matmul_primitives.zig");
pub const matmul_prefill = @import("matmul_prefill.zig");

pub const math = @import("math_primitives/root.zig");

// === SIMD kernels ===
pub const simd = @import("simd/root.zig");
