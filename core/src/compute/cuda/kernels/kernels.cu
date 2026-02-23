#include <cuda_fp16.h>
#include <stdint.h>

// Aggregation unit for CUDA kernel symbols compiled into kernels.fatbin.
// Keep this file as the single nvcc entrypoint; place actual kernels in ops/*.cu.
#include "ops/elementwise_ops.cu"
#include "ops/norm_rope_ops.cu"
#include "ops/attention_ops.cu"
#include "ops/reduction_ops.cu"
#include "ops/common_decode.cuh"
#include "ops/matvec_ops.cu"
#include "ops/quant_ops.cu"
#include "ops/shortconv_ops.cu"
