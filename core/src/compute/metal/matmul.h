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

/// Matrix multiplication for grouped-affine u4 quantized weights.
/// A: [m x k] f32, B: [k x n] grouped_affine_u4 (packed nibbles with BF16 scales/biases), C: [m x n] f32.
/// group_size: number of elements per quantization group (typically 32 or 64).
bool metal_matmul_mlx4bit(
    MetalDevice* device,
    const float* a, size_t m, size_t k,
    const uint8_t* b_data,
    const uint16_t* b_scales,
    const uint16_t* b_biases,
    size_t n,
    size_t group_size,
    float* c
);

#ifdef __cplusplus
}
#endif

#endif // TALU_METAL_MATMUL_H
