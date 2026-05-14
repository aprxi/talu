//! Shared CUDA inference adapters for generic fused linear compute execution.

const builtin = @import("builtin");
const compute = @import("compute_pkg");

const engine_types = @import("../runtime/root.zig");
const Nvfp4RouteKind = engine_types.Nvfp4RouteKind;

pub const linear = compute.cuda.linear.fused;

fn recordNvfp4Route(self: anytype, comptime kind: Nvfp4RouteKind) void {
    const SelfType = @TypeOf(self.*);
    if (comptime @hasField(SelfType, "nvfp4_route_counters")) {
        self.nvfp4_route_counters.record(kind);
    }
}

fn shouldAvoidWindowsPreSm89Nvfp4Fused(self: anytype) bool {
    if (builtin.os.tag != .windows) return false;
    const capability = self.device.computeCapability() catch return true;
    return capability.major < 8 or (capability.major == 8 and capability.minor < 9);
}

pub fn makeCapabilities(self: anytype) linear.CapabilityFlags {
    return .{
        .i8_blas_supported = self.i8_blas_supported,
        .gaffine_u4_tile8_enabled = self.gaffine_u4_tile8_enabled,
        .nvfp4_pair_multi_row_supported = self.nvfp4_sequence_fused_gate_up_supported,
        .nvfp4_triple_multi_row_supported = self.nvfp4_sequence_fused_qkv_supported,
        .nvfp4_custom_supported = !shouldAvoidWindowsPreSm89Nvfp4Fused(self),
    };
}

pub fn makeContext(
    self: anytype,
    diagnostics: *linear.Diagnostics,
    capabilities: *linear.CapabilityFlags,
) linear.FusedContext {
    const blas_lt = if (self.blas_lt) |*handle| handle else null;
    return .{
        .device = &self.device,
        .arg_pack = &self.kernel_arg_pack,
        .blas = &self.blas,
        .blas_lt = blas_lt,
        .workspace = .{
            .activation_scratch = self.runtime_buffers.activation_u16_dev,
            .auxiliary_scratch = self.runtime_buffers.dequant_f16_dev,
        },
        .capabilities = capabilities,
        .diagnostics = diagnostics,
        .dense_u16_pair_f16_function = self.matvec_gate_up_f16_function,
        .dense_u16_pair_bf16_function = self.matvec_gate_up_bf16_function,
        .dense_u16_pair_silu_f16_function = self.matvec_gate_up_silu_f16_function,
        .dense_u16_pair_silu_bf16_function = self.matvec_gate_up_silu_bf16_function,
        .dense_u16_triple_f16_function = self.matvec_qkv_f16_function,
        .dense_u16_triple_bf16_function = self.matvec_qkv_bf16_function,
        .gaffine_u4_pair_silu_function = self.gaffine_u4_matvec_gate_up_silu_function,
        .gaffine_u4_pair_silu_tile8_function = self.gaffine_u4_matvec_gate_up_silu_tile8_function,
        .gaffine_u4_triple_function = self.gaffine_u4_matvec_qkv_function,
        .gaffine_u4_triple_tile8_function = self.gaffine_u4_matvec_qkv_tile8_function,
        .gaffine_u8_pair_function = self.gaffine_u8_matvec_gate_up_function,
        .gaffine_u8_pair_silu_function = self.gaffine_u8_matvec_gate_up_silu_function,
        .gaffine_u8_triple_function = self.gaffine_u8_matvec_qkv_function,
        .fp8_pair_function = self.fp8_matvec_gate_up_function,
        .fp8_pair_tile8_function = self.fp8_matvec_gate_up_tile8_function,
        .fp8_pair_silu_function = self.fp8_matvec_gate_up_silu_function,
        .fp8_pair_silu_tile8_function = self.fp8_matvec_gate_up_silu_tile8_function,
        .mxfp8_pair_function = self.mxfp8_matvec_gate_up_function,
        .mxfp8_pair_tile8_function = self.mxfp8_matvec_gate_up_tile8_function,
        .mxfp8_pair_silu_function = self.mxfp8_matvec_gate_up_silu_function,
        .mxfp8_pair_silu_tile8_function = self.mxfp8_matvec_gate_up_silu_tile8_function,
        .nvfp4_pair_function = self.nvfp4_matvec_gate_up_function,
        .nvfp4_pair_tile8_function = self.nvfp4_matvec_gate_up_tile8_function,
        .nvfp4_pair_silu_function = self.nvfp4_matvec_gate_up_silu_function,
        .nvfp4_pair_silu_tile8_function = self.nvfp4_matvec_gate_up_silu_tile8_function,
        .nvfp4_pair_gelu_function = self.nvfp4_matvec_gate_up_gelu_function,
        .nvfp4_pair_gelu_tile8_function = self.nvfp4_matvec_gate_up_gelu_tile8_function,
        .nvfp4_triple_function = self.nvfp4_matvec_qkv_function,
        .nvfp4_triple_tile8_function = self.nvfp4_matvec_qkv_tile8_function,
        .quantize_f32_to_nvfp4_function = self.quantize_f32_to_nvfp4_function,
        .quantize_f32_to_i8_simple_function = self.quantize_f32_to_i8_simple_function,
        .dequant_i32_scales_split3_function = self.dequant_i32_scales_split3_function,
    };
}

pub fn syncCapabilityFlags(self: anytype, capabilities: *const linear.CapabilityFlags) void {
    self.i8_blas_supported = capabilities.i8_blas_supported;
}

pub fn emitDiagnostics(self: anytype, diagnostics: *const linear.Diagnostics) void {
    if (diagnostics.nvfp4_route) |route| {
        switch (route) {
            .pair_custom_kernel => recordNvfp4Route(self, .fused_gate_up_custom),
            .pair_native_cublaslt => recordNvfp4Route(self, .fused_gate_up_native_cublaslt),
            .triple_custom_kernel => recordNvfp4Route(self, .fused_qkv_custom),
            .triple_native_cublaslt => recordNvfp4Route(self, .fused_qkv_native_cublaslt),
        }
    }
}
