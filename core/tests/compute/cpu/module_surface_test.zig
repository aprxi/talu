const std = @import("std");
const compute = @import("main").compute;

test "compute.cpu exposes primitive-first modules" {
    _ = compute.cpu.common;
    _ = compute.cpu.activation;
    _ = compute.cpu.normalization;
    _ = compute.cpu.rowwise;
    _ = compute.cpu.layout_transform;
    _ = compute.cpu.tensor_copy;
    _ = compute.cpu.tensor_gather;
    _ = compute.cpu.quant_decode;
    _ = compute.cpu.cache_layout;
    _ = compute.cpu.cache_store;
    _ = compute.cpu.rotary;
    _ = compute.cpu.conv1d_depthwise;
    _ = compute.cpu.matvec;
    _ = compute.cpu.topk;
    _ = compute.cpu.reduction;
    _ = compute.cpu.softmax;
    _ = compute.cpu.sdpa_decode;
}

test "compute.cpu transitional modules remain available during migration" {
    _ = compute.cpu.matmul;
    _ = compute.cpu.norm;
    _ = compute.cpu.attention;
    _ = compute.cpu.math;
}
