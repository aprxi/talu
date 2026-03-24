// =============================================================================
// Metal Matrix Multiplication - C API
// =============================================================================

#ifndef TALU_METAL_MATMUL_H
#define TALU_METAL_MATMUL_H

#include <stdint.h>
#include <stdbool.h>
#include "device.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Matrix multiplication using Metal Performance Shaders: C = A @ B.
/// A: [m x k], B: [k x n], C: [m x n].
/// All matrices are row-major f32.
bool metal_matmul_f32(
    MetalDevice* device,
    const float* a, size_t m, size_t k,
    const float* b, size_t n,
    float* c
);

/// Matrix multiplication with transposed B and scaling: C = alpha * A @ B^T.
/// A: [m x k], B: [n x k] (stored row-major, will be transposed), C: [m x n].
/// Optimized for attention: Q @ K^T where Q=[queries, head_dim], K=[history, head_dim].
bool metal_matmul_f32_transB_scaled(
    MetalDevice* device,
    const float* a, size_t m, size_t k,
    const float* b, size_t n,
    float* c,
    float alpha
);

/// INT8 K matmul with on-the-fly dequant: C = alpha * A @ dequant(B)^T.
/// A: [m x k] f32 (queries)
/// B: [n x k] i8 (keys)
/// B_scales: [n] f32 (per-row scales for K)
/// C: [m x n] f32 (output scores)
/// Dequants B to f32 on GPU, then computes matmul.
bool metal_matmul_f32_i8_transB_scaled(
    MetalDevice* device,
    const float* a, size_t m, size_t k,
    const int8_t* b, size_t n,
    const float* b_scales,
    float* c,
    float alpha
);

#ifdef __cplusplus
}
#endif

#endif // TALU_METAL_MATMUL_H
