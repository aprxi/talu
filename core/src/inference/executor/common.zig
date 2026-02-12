//! Executor common imports - kernel access for block execution.

const tensor_mod = @import("../../tensor.zig");
const compute = @import("../../compute/root.zig");
const matmul_mod = compute.ops.matmul;
const dtype_mod = @import("../../dtype.zig");
const inspect = @import("../../xray/root.zig");
pub const kernel_info = inspect.kernel_info;
pub const perf_estimate = inspect.perf_estimate;

// Inference kernels (CPU today)
pub const attn_kernel = @import("../backend/cpu/kernels/attention.zig");
pub const mla_kernel = @import("../backend/cpu/kernels/mla_attention.zig");
pub const ffn_kernel = @import("../backend/cpu/kernels/ffn.zig");
pub const moe_kernel = @import("../backend/cpu/kernels/moe.zig");
pub const norm_kernel = @import("../backend/cpu/kernels/norm.zig");
pub const rope_kernel = @import("../backend/cpu/kernels/rope.zig");
pub const embedding_kernel = @import("../backend/cpu/kernels/embedding.zig");

const cpu_blocks = @import("../backend/cpu/block_kernels.zig");
const cpu_scratch = @import("../backend/cpu/scratch.zig");
pub const forward = cpu_blocks; // Back-compat for existing imports.
pub const block_kernels = cpu_blocks; // Alias for clarity

// Common core types
pub const Tensor = tensor_mod.Tensor;
pub const OwnedTensor = tensor_mod.OwnedTensor;
pub const DType = dtype_mod.DType;
pub const matmul = matmul_mod;
pub const MatmulFn = matmul_mod.MatmulFn;
pub const KernelOp = kernel_info.KernelOp;

// Kernel struct types
pub const Attention = attn_kernel.MultiHeadAttention;
pub const MLAttention = mla_kernel.MLAttention;
pub const RMSNorm = norm_kernel.RMSNorm;
pub const AttnTemp = attn_kernel.AttnTemp;
pub const AttnCache = attn_kernel.AttnCache;
pub const MLACache = mla_kernel.MLACache;
pub const MLATemp = mla_kernel.MLATemp;
pub const RoPE = rope_kernel.RoPE;
pub const FfnScratch = ffn_kernel.FfnScratch;
pub const MoeScratch = moe_kernel.MoEScratch;
pub const ScratchBuffer = cpu_scratch.ScratchBuffer;
pub const FFNLayer = cpu_blocks.FfnLayer;
pub const TransformerBlock = cpu_blocks.TransformerBlock;

// Forward helpers
pub const addIntoScaled = cpu_blocks.addIntoScaled;
pub const copyTensor = cpu_blocks.copyTensor;
