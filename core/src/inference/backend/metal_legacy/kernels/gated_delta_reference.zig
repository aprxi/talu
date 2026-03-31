//! Metal gated-delta reference path.
//!
//! This file intentionally keeps a separate, explicit execution path that can
//! be toggled at runtime for isolation/debugging:
//! `TALU_METAL_GDELTA_REFERENCE=1`
//! or globally with `TALU_METAL_ATTN_REFERENCE=1`.

const std = @import("std");
const mlx_fused = @import("../mlx/ffi.zig");

const ArrayHandle = mlx_fused.ArrayHandle;

pub fn enabled() bool {
    const raw = std.posix.getenv("TALU_METAL_GDELTA_REFERENCE") orelse std.posix.getenv("TALU_METAL_ATTN_REFERENCE") orelse return false;
    const value = std.mem.sliceTo(raw, 0);
    if (value.len == 0) return false;
    if (std.ascii.eqlIgnoreCase(value, "0")) return false;
    if (std.ascii.eqlIgnoreCase(value, "false")) return false;
    if (std.ascii.eqlIgnoreCase(value, "off")) return false;
    if (std.ascii.eqlIgnoreCase(value, "no")) return false;
    return true;
}

pub const CaptureTargets = struct {
    in_proj: *ArrayHandle,
    conv: *ArrayHandle,
    ssm: *ArrayHandle,
    norm: *ArrayHandle,
    state_conv: *ArrayHandle,
    state_ssm: *ArrayHandle,
};

pub fn forwardBf16(
    input_tensor: ArrayHandle,
    output_tensor: *ArrayHandle,
    in_proj_bf16: ArrayHandle,
    conv_weight: ArrayHandle,
    conv_bias: ?ArrayHandle,
    a_log: ArrayHandle,
    dt_bias: ?ArrayHandle,
    norm_weight: ?ArrayHandle,
    out_proj_bf16: ArrayHandle,
    cache_handle: ?*anyopaque,
    layer_idx: usize,
    d_conv: usize,
    n_heads: usize,
    n_key_heads: usize,
    d_head: usize,
    capture_enabled: bool,
    capture: CaptureTargets,
) void {
    if (capture_enabled) {
        output_tensor.* = mlx_fused.mlx_lazy_gated_delta_mixer_bf16_capture(
            input_tensor,
            in_proj_bf16,
            conv_weight,
            if (conv_bias) |bias| bias else null,
            a_log,
            if (dt_bias) |bias| bias else null,
            if (norm_weight) |weight| weight else null,
            out_proj_bf16,
            cache_handle,
            layer_idx,
            d_conv,
            n_heads,
            n_key_heads,
            d_head,
            capture.in_proj,
            capture.conv,
            capture.ssm,
            capture.norm,
            capture.state_conv,
            capture.state_ssm,
        );
        return;
    }
    output_tensor.* = mlx_fused.mlx_lazy_gated_delta_mixer_bf16(
        input_tensor,
        in_proj_bf16,
        conv_weight,
        if (conv_bias) |bias| bias else null,
        a_log,
        if (dt_bias) |bias| bias else null,
        if (norm_weight) |weight| weight else null,
        out_proj_bf16,
        cache_handle,
        layer_idx,
        d_conv,
        n_heads,
        n_key_heads,
        d_head,
    );
}

pub fn forwardQuantized(
    input_tensor: ArrayHandle,
    output_tensor: *ArrayHandle,
    in_weights: ArrayHandle,
    in_scales: ArrayHandle,
    in_biases: ArrayHandle,
    conv_weight: ArrayHandle,
    conv_bias: ?ArrayHandle,
    a_log: ArrayHandle,
    dt_bias: ?ArrayHandle,
    norm_weight: ?ArrayHandle,
    out_weights: ArrayHandle,
    out_scales: ArrayHandle,
    out_biases: ArrayHandle,
    group_size: usize,
    bits: usize,
    cache_handle: ?*anyopaque,
    layer_idx: usize,
    d_conv: usize,
    n_heads: usize,
    n_key_heads: usize,
    d_head: usize,
    capture_enabled: bool,
    capture: CaptureTargets,
) void {
    if (capture_enabled) {
        output_tensor.* = mlx_fused.mlx_lazy_gated_delta_mixer_quantized_capture(
            input_tensor,
            in_weights,
            in_scales,
            in_biases,
            conv_weight,
            if (conv_bias) |bias| bias else null,
            a_log,
            if (dt_bias) |bias| bias else null,
            if (norm_weight) |weight| weight else null,
            out_weights,
            out_scales,
            out_biases,
            group_size,
            bits,
            cache_handle,
            layer_idx,
            d_conv,
            n_heads,
            n_key_heads,
            d_head,
            capture.in_proj,
            capture.conv,
            capture.ssm,
            capture.norm,
            capture.state_conv,
            capture.state_ssm,
        );
        return;
    }
    output_tensor.* = mlx_fused.mlx_lazy_gated_delta_mixer_quantized(
        input_tensor,
        in_weights,
        in_scales,
        in_biases,
        conv_weight,
        if (conv_bias) |bias| bias else null,
        a_log,
        if (dt_bias) |bias| bias else null,
        if (norm_weight) |weight| weight else null,
        out_weights,
        out_scales,
        out_biases,
        group_size,
        bits,
        cache_handle,
        layer_idx,
        d_conv,
        n_heads,
        n_key_heads,
        d_head,
    );
}

test "enabled parses truthy/falsy env values" {
    // Falsy by default in test process.
    try std.testing.expect(!enabled());
}
