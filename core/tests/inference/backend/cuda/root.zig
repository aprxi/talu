//! Integration tests for the inference/backend/cuda module.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const main = @import("main");

const backend = main.inference.backend;
const has_cuda = build_options.enable_cuda and (builtin.os.tag == .linux or builtin.os.tag == .windows);

test "cuda backend module export is available" {
    _ = backend.cuda;
}

test "cuda backend type is available when enabled" {
    if (comptime !has_cuda) return;
    _ = backend.cuda.BackendType;
}

test {
    _ = @import("gated_attention_compact_q_test.zig");
    _ = @import("gated_attention_output_gate_test.zig");
    _ = @import("gated_delta_conv_test.zig");
    _ = @import("gated_delta_qk_norm_test.zig");
    _ = @import("gated_delta_ssm_test.zig");
    std.testing.refAllDecls(@This());
}
