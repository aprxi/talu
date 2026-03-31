//! Batched top-k kernel wrapper over row-major logits [rows, vocab].
//!
//! Uses a two-phase parallel algorithm:
//!   Phase 1: Split each row across CHUNKS blocks (grid_x = rows * chunks).
//!            Each block finds top-k from its chunk.
//!   Phase 2: Merge per-chunk candidates into final top-k (grid_x = rows).
//!
//! This keeps all SMs busy (rows * CHUNKS blocks) instead of only `rows`
//! blocks, giving ~20-30x speedup for small batch sizes on large GPUs.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");
const registry_mod = @import("registry.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;

pub const phase1_symbol: [:0]const u8 = "talu_topk_rows_phase1";
pub const phase2_symbol: [:0]const u8 = "talu_topk_rows_phase2";
pub const phase1_op_name: []const u8 = "topk_rows_phase1";
pub const phase2_op_name: []const u8 = "topk_rows_phase2";

/// Number of chunks to split each row into for phase 1 parallelism.
pub const CHUNKS: u32 = 20;

pub fn runTwoPhase(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    phase1_fn: module_mod.Function,
    phase2_fn: module_mod.Function,
    values_out: *device_mod.Buffer,
    ids_out: *device_mod.Buffer,
    logits: *const device_mod.Buffer,
    scratch_vals: *device_mod.Buffer,
    scratch_ids: *device_mod.Buffer,
    rows: u32,
    vocab: u32,
    row_stride: u32,
    k: u32,
) !void {
    if (rows == 0 or vocab == 0 or row_stride == 0 or k == 0) return error.InvalidArgument;
    if (k > row_stride) return error.InvalidArgument;

    const chunks: u32 = CHUNKS;
    const block_x: u32 = 256;

    // Phase 1: per-chunk top-k extraction.
    arg_pack.reset();
    try arg_pack.appendBufferPtr(scratch_vals);
    try arg_pack.appendBufferPtr(scratch_ids);
    try arg_pack.appendBufferPtr(logits);
    try arg_pack.appendScalar(u32, rows);
    try arg_pack.appendScalar(u32, vocab);
    try arg_pack.appendScalar(u32, chunks);
    try arg_pack.appendScalar(u32, k);

    try launch_mod.launchWithFamily(device, phase1_fn, .{
        .grid_x = rows * chunks,
        .block_x = block_x,
    }, arg_pack, .pointwise);

    // Phase 2: merge chunk candidates into final top-k.
    arg_pack.reset();
    try arg_pack.appendBufferPtr(values_out);
    try arg_pack.appendBufferPtr(ids_out);
    try arg_pack.appendBufferPtr(scratch_vals);
    try arg_pack.appendBufferPtr(scratch_ids);
    try arg_pack.appendScalar(u32, rows);
    try arg_pack.appendScalar(u32, chunks);
    try arg_pack.appendScalar(u32, k);
    try arg_pack.appendScalar(u32, row_stride);

    try launch_mod.launchWithFamily(device, phase2_fn, .{
        .grid_x = rows,
        .block_x = block_x,
    }, arg_pack, .pointwise);
}

/// Scratch buffer size in bytes for phase 1 intermediate results.
pub fn scratchBytes(max_rows: u32, k: u32) usize {
    const entries = @as(usize, max_rows) * @as(usize, CHUNKS) * @as(usize, k);
    return entries * @sizeOf(f32); // same count for ids (u32 = same size)
}
