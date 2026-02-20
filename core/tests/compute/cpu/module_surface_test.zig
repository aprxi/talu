const std = @import("std");
const compute = @import("main").compute;

test "compute.cpu exposes primitive-first modules" {
    _ = compute.cpu.common;
    _ = compute.cpu.activation;
    _ = compute.cpu.normalization;
    _ = compute.cpu.rowwise;
    _ = compute.cpu.layout.transform;
    _ = compute.cpu.memory.copy;
    _ = compute.cpu.memory.gather;
    _ = compute.cpu.quant_decode;
    _ = compute.cpu.cache.layout;
    _ = compute.cpu.cache.store;
    _ = compute.cpu.rotary;
    _ = compute.cpu.conv1d_depthwise;
    _ = compute.cpu.linalg.matvec;
    _ = compute.cpu.topk;
    _ = compute.cpu.reduction;
    _ = compute.cpu.softmax;
    _ = compute.cpu.sdpa_decode;
}

test "compute.cpu transitional modules remain available during migration" {
    _ = compute.cpu.linalg.matmul;
    _ = compute.cpu.normalization;
    _ = compute.cpu.attn_primitives;
    _ = compute.cpu.math_primitives;
}
