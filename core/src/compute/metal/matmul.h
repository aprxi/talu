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

#ifdef __cplusplus
}
#endif

#endif // TALU_METAL_MATMUL_H
