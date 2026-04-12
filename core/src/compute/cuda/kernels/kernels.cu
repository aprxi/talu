#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <stdint.h>

// Aggregation unit for CUDA kernel symbols compiled into kernels.fatbin.
// Keep this file as the single nvcc entrypoint; place actual kernels in ops/*.cu.
#include "ops/elementwise_ops.cu"
#include "ops/norm_rope_ops.cu"
#include "ops/attention_ops.cu"
#include "ops/gated_attention_ops.cu"
#include "ops/reduction_ops.cu"
#include "ops/common_decode.cuh"
#include "ops/matvec_ops.cu"
#include "ops/quant_common.cuh"
#include "ops/quant_gaffine_u4_ops.cu"
#include "ops/quant_gaffine_u8_ops.cu"
#include "ops/quant_i8_ops.cu"
#include "ops/quant_calibration_ops.cu"
#include "ops/quant_fp8_ops.cu"
#include "ops/quant_mxfp8_ops.cu"
#include "ops/quant_nvfp4_ops.cu"
#include "ops/shortconv_ops.cu"
#include "ops/gated_delta_conv_ops.cu"
#include "ops/gated_delta_norm_ops.cu"
#include "ops/gated_delta_ops.cu"
#include "ops/flash_attention_ops.cu"
