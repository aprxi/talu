//! CPU Kernel Wrapper
//!
//! Provides a generic kernel interface for executor dispatch, using
//! stable pointers to concrete kernel implementations.
//!
//! ## Batched Execution
//!
//! The `forwardBatched()` method uses slot-indexed scratch accessors
//! (`getFfnScratch(slot_index)`, `getMoeScratch(slot_index)`) to support
//! future parallel batched inference.
//!
//! Current limitation: continuous batching executes slots sequentially.
//! The accessors return shared scratch safely in this mode. When adding
//! parallelism, only the accessor internals change to return per-slot
//! disjoint regions.

const compute = @import("../../../compute/root.zig");
const matmul = compute.ops.matmul;
const graph_runtime = @import("../../graph_runtime/root.zig");
const OpType = graph_runtime.OpType;

const tensor = @import("../../../tensor.zig");
const Tensor = tensor.Tensor;

const attn = @import("kernels/attention.zig");
const ffn = @import("kernels/ffn.zig");
const moe = @import("kernels/moe.zig");
const norm = @import("kernels/norm.zig");
const mamba = @import("kernels/mamba.zig");
const shortconv = @import("kernels/shortconv.zig");
const mla = @import("kernels/mla_attention.zig");
const kv_cache = @import("kernels/kv_cache.zig");
const scratch_mod = @import("scratch.zig");

const AttnCache = attn.AttnCache;
const BatchedKVCache = kv_cache.BatchedKVCache;
const ScratchBuffer = scratch_mod.ScratchBuffer;
const MambaState = scratch_mod.MambaState;
const MambaScratch = scratch_mod.MambaScratch;
const ShortConvState = shortconv.ShortConvState;
const ShortConvScratch = shortconv.ShortConvScratch;

/// Runtime resources needed by kernels during execution.
pub const KernelContext = struct {
    scratch: *ScratchBuffer,
    matmul_scratch: *matmul.MatmulScratch,
    // Single-sequence attention cache (null in batched path)
    attn_cache: ?*AttnCache = null,
    // MLA attention resources (null for non-MLA blocks)
    mla_cache: ?*mla.MLACache = null,
    mla_scratch: ?*mla.MLATemp = null,
    // Mamba resources (null for non-Mamba blocks)
    mamba_state: ?*MambaState = null,
    mamba_scratch: ?*MambaScratch = null,
    // ShortConv resources (null for non-ShortConv blocks)
    shortconv_state: ?*ShortConvState = null,
    shortconv_scratch: ?*ShortConvScratch = null,
    // Cache usage flag
    use_cache: bool,
};

/// CPU kernel dispatch union.
/// Holds pointers to concrete kernel implementations.
pub const CpuKernel = union(enum) {
    attention: *const attn.MultiHeadAttention,
    mla_attention: *const mla.MLAttention,
    mamba: *const mamba.MambaKernel,
    shortconv: *const shortconv.ShortConvKernel,
    swiglu: *const ffn.SwiGLU,
    moe: *const moe.MoEFFN,
    norm: *const norm.NormKernel,

    /// Returns the operation type for debug validation.
    /// Used to verify graph compiler emits kernels in same order as block builds.
    pub fn getOpType(self: CpuKernel) OpType {
        return switch (self) {
            .attention, .mla_attention => .multihead_attention,
            .mamba => .mamba_mixer,
            .shortconv => .shortconv,
            .swiglu => .mlp,
            .moe => .moe,
            .norm => .norm,
        };
    }

    pub fn forward(self: CpuKernel, input: *const Tensor, output: *Tensor, ctx: KernelContext) !void {
        switch (self) {
            .attention => |k| try k.forward(input, output, ctx.attn_cache.?, &ctx.scratch.attn_scratch, ctx.matmul_scratch, ctx.use_cache),
            .mla_attention => |k| try k.forward(input, output, ctx.mla_cache.?, ctx.mla_scratch.?, ctx.matmul_scratch, ctx.use_cache),
            .mamba => |k| try k.forward(input, output, ctx.mamba_state.?, ctx.mamba_scratch.?, ctx.matmul_scratch),
            .shortconv => |k| try k.forward(input, output, ctx.shortconv_state.?, ctx.shortconv_scratch.?, ctx.matmul_scratch),
            .swiglu => |k| try k.forward(input, output, &ctx.scratch.ffn_scratch, ctx.matmul_scratch),
            .moe => |k| try k.forward(input, output, &ctx.scratch.moe_scratch, ctx.matmul_scratch),
            .norm => |k| k.forward(input, output),
        }
    }

    pub fn forwardBatched(
        self: CpuKernel,
        input: *const Tensor,
        output: *Tensor,
        ctx: KernelContext,
        batched_cache: *BatchedKVCache,
        slot_index: usize,
    ) !void {
        switch (self) {
            .attention => |k| try k.forwardWithBatchedCache(input, output, batched_cache, slot_index, &ctx.scratch.attn_scratch, ctx.matmul_scratch, ctx.use_cache),
            // MLA batched forward not yet implemented - fall back to single-sequence path
            .mla_attention => |k| try k.forward(input, output, ctx.mla_cache.?, ctx.mla_scratch.?, ctx.matmul_scratch, ctx.use_cache),
            .mamba => |k| try k.forward(input, output, ctx.mamba_state.?, ctx.mamba_scratch.?, ctx.matmul_scratch),
            .shortconv => |k| try k.forward(input, output, ctx.shortconv_state.?, ctx.shortconv_scratch.?, ctx.matmul_scratch),
            .swiglu => |k| try k.forward(input, output, ctx.scratch.getFfnScratch(slot_index), ctx.matmul_scratch),
            .moe => |k| try k.forward(input, output, ctx.scratch.getMoeScratch(slot_index), ctx.matmul_scratch),
            .norm => |k| k.forward(input, output),
        }
    }
};
