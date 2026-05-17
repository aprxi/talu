//! Concrete inference backend namespace.
//!
//! This module exposes CPU/Metal/CUDA backend implementations and shared backend
//! contracts. Scheduler-facing execution-target selection lives in `../scheduler/`.

const builtin = @import("builtin");
const build_options = @import("build_options");

pub const contract = @import("contract.zig");
pub const cpu = @import("cpu/root.zig");

pub const has_metal = build_options.enable_metal and builtin.os.tag == .macos;
pub const has_cuda = build_options.enable_cuda and (builtin.os.tag == .linux or builtin.os.tag == .windows);

pub const metal = if (has_metal) @import("metal/root.zig") else struct {
    pub const BackendType = void;
};

pub const cuda = if (has_cuda) @import("cuda/root.zig") else struct {
    pub const BackendType = void;
};

pub const PoolingStrategy = contract.PoolingStrategy;

// CPU backend behavioral and weight types retained as concrete backend exports
// for existing inference tests/tooling.
pub const FusedCpuBackend = cpu.FusedCpuBackend;
pub const kernels = cpu.kernels;
pub const block_kernels = cpu.executor.weights;
pub const TransformerBlock = kernels.TransformerBlock;
pub const ScratchBuffer = kernels.ScratchBuffer;
pub const AttnTemp = kernels.AttnTemp;
pub const AttnCache = kernels.AttnCache;
pub const FfnScratch = kernels.FfnScratch;
pub const MultiHeadAttention = kernels.MultiHeadAttention;
pub const SwiGLU = kernels.SwiGLU;
pub const FfnLayer = block_kernels.FfnLayer;
pub const RMSNorm = kernels.RMSNorm;

comptime {
    contract.assertBackendModuleLayout(cpu, "cpu");
    contract.assertInterfaceModuleLayout(cpu.interface, "cpu");
    contract.assertVisionModuleLayout(cpu.vision, "cpu");
    contract.assertExecutorModuleLayout(cpu.executor, "cpu");
    contract.assertExecutorSymbolLayout(cpu.executor, "cpu");
    contract.assertKernelModuleLayout(cpu.kernels, "cpu");
    contract.assertKernelSupportMap(cpu.kernels, "cpu");
    contract.assertKernelSymbolLayout(cpu.kernels, "cpu");
    contract.assertUnsupportedKernelPolicy(cpu.kernels, "cpu");
    contract.assertSchedulerModuleLayout(cpu.scheduler, "cpu");
    contract.assertSamplingModuleLayout(cpu.sampling, "cpu");
    contract.assertBackendType(cpu.BackendType);
    if (has_metal) {
        contract.assertBackendModuleLayout(metal, "metal");
        contract.assertInterfaceModuleLayout(metal.interface, "metal");
        contract.assertVisionModuleLayout(metal.vision, "metal");
        contract.assertExecutorModuleLayout(metal.executor, "metal");
        contract.assertExecutorSymbolLayout(metal.executor, "metal");
        contract.assertKernelModuleLayout(metal.kernels, "metal");
        contract.assertKernelSupportMap(metal.kernels, "metal");
        contract.assertKernelSymbolLayout(metal.kernels, "metal");
        contract.assertUnsupportedKernelPolicy(metal.kernels, "metal");
        contract.assertSchedulerModuleLayout(metal.scheduler, "metal");
        contract.assertSamplingModuleLayout(metal.sampling, "metal");
        contract.assertBackendType(metal.BackendType);
    }
    if (has_cuda) {
        contract.assertBackendModuleLayout(cuda, "cuda");
        contract.assertInterfaceModuleLayout(cuda.interface, "cuda");
        contract.assertVisionModuleLayout(cuda.vision, "cuda");
        contract.assertExecutorModuleLayout(cuda.executor, "cuda");
        contract.assertExecutorSymbolLayout(cuda.executor, "cuda");
        contract.assertKernelModuleLayout(cuda.kernels, "cuda");
        contract.assertKernelSupportMap(cuda.kernels, "cuda");
        contract.assertKernelSymbolLayout(cuda.kernels, "cuda");
        contract.assertUnsupportedKernelPolicy(cuda.kernels, "cuda");
        contract.assertSchedulerModuleLayout(cuda.scheduler, "cuda");
        contract.assertSamplingModuleLayout(cuda.sampling, "cuda");
        contract.assertBackendType(cuda.BackendType);
    }
}
