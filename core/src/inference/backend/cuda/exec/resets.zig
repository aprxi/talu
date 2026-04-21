//! Forward pass computation functions.
//!
//! Contains the main forward-pass entry points (single-token decode, batched
//! decode, prefill), KV capacity management, and recurrent state resets.
//! Functions use `self: anytype` to avoid circular imports with engine.zig.

const std = @import("std");
const compute = @import("compute_pkg");
const tensor = @import("tensor_pkg");
const log = @import("log_pkg");
const trace = @import("xray_pkg").trace;
const staged_orchestrator = @import("../../staged_orchestrator.zig");
const per_layer_branch_feature = @import("../per_layer_branch.zig");

// --- Shared types from engine_types.zig ---
const engine_types = @import("../runtime/_types_impl.zig");
const BatchDecodeInfo = engine_types.BatchDecodeInfo;
const KvCacheDtype = engine_types.KvCacheDtype;
const enable_dispatch_observability = engine_types.enable_dispatch_observability;
const enable_device_embedding_lookup = engine_types.enable_device_embedding_lookup;
const AttentionKernelSet = engine_types.AttentionKernelSet;

// --- Compute ops from engine_ops.zig ---
const engine_ops = @import("../operators/root.zig");

// --- Utilities from engine_weights.zig ---
const engine_weights = @import("../weights/root.zig");
const bufferSlice = engine_weights.bufferSlice;
const populatePrefillHiddenFromTokens = engine_weights.populatePrefillHiddenFromTokens;
const tryPopulateHiddenFromToken = engine_weights.tryPopulateHiddenFromToken;

const saturatingU64FromU128 = engine_types.saturatingU64FromU128;
const logicalF32RowSlice = engine_types.logicalF32RowSlice;

fn topologyModeTag(self: anytype) ?[]const u8 {
    const SelfType = @TypeOf(self.*);
    if (comptime !@hasField(SelfType, "topology_mode")) return null;
    return @tagName(self.topology_mode);
}

fn topologyModeIs(self: anytype, comptime expected: []const u8) bool {
    const tag = topologyModeTag(self) orelse return false;
    return std.mem.eql(u8, tag, expected);
}

/// Resolve staged prefill chunk rows for a specific request length.
/// Keeps explicit env override behavior unchanged.
pub fn resetShortConvStates(self: anytype) !void {
    for (self.block_runtime.blocks) |*layer| {
        const block = layer.shortconv_binding orelse continue;
        const elems = std.math.divExact(usize, block.conv_state.size, @sizeOf(f32)) catch return error.InvalidArgument;
        const zeros = try self.allocator.alloc(f32, elems);
        defer self.allocator.free(zeros);
        @memset(zeros, 0.0);
        try block.conv_state.upload(&self.device, std.mem.sliceAsBytes(zeros));
    }
}

pub fn resetGatedDeltaStates(self: anytype) void {
    for (self.block_runtime.blocks) |*layer| {
        const block = layer.gated_delta_binding orelse continue;
        block.state.reset();
        block.conv_ring_head = 0;
        const conv_elems = std.math.divExact(usize, block.conv_state_dev.size, @sizeOf(f32)) catch continue;
        const conv_zeros = self.allocator.alloc(f32, conv_elems) catch continue;
        defer self.allocator.free(conv_zeros);
        @memset(conv_zeros, 0.0);
        block.conv_state_dev.upload(&self.device, std.mem.sliceAsBytes(conv_zeros)) catch {};
        const BlockType = @TypeOf(block.*);
        if (comptime @hasDecl(BlockType, "ssmStateDataBytes") and
            @hasDecl(BlockType, "ssmStateScalesCount") and
            @hasField(BlockType, "ssm_state_format") and
            @hasField(BlockType, "ssm_state_scales_offset"))
        {
            const ssm_data_bytes = block.ssmStateDataBytes() catch continue;
            switch (block.ssm_state_format) {
                .f32 => {
                    const ssm_elems = std.math.divExact(usize, ssm_data_bytes, @sizeOf(f32)) catch continue;
                    const zeros = self.allocator.alloc(f32, ssm_elems) catch continue;
                    defer self.allocator.free(zeros);
                    @memset(zeros, 0.0);
                    block.ssm_state_dev.upload(&self.device, std.mem.sliceAsBytes(zeros)) catch {};
                },
                .i8_per_column_scale => {
                    const zeros_i8 = self.allocator.alloc(i8, ssm_data_bytes) catch continue;
                    defer self.allocator.free(zeros_i8);
                    @memset(zeros_i8, 0);
                    var ssm_i8_dev = bufferSlice(&block.ssm_state_dev, 0, ssm_data_bytes) catch continue;
                    ssm_i8_dev.upload(&self.device, std.mem.sliceAsBytes(zeros_i8)) catch {};

                    const scale_count = block.ssmStateScalesCount();
                    if (scale_count > 0) {
                        const scale_bytes = std.math.mul(usize, scale_count, @sizeOf(f32)) catch continue;
                        const scales = self.allocator.alloc(f32, scale_count) catch continue;
                        defer self.allocator.free(scales);
                        @memset(scales, 1.0);
                        var scales_dev = bufferSlice(&block.ssm_state_dev, @as(usize, block.ssm_state_scales_offset), scale_bytes) catch continue;
                        scales_dev.upload(&self.device, std.mem.sliceAsBytes(scales)) catch {};
                    }
                },
            }
        } else {
            const ssm_elems = std.math.divExact(usize, block.ssm_state_dev.size, @sizeOf(f32)) catch continue;
            const zeros = self.allocator.alloc(f32, ssm_elems) catch continue;
            defer self.allocator.free(zeros);
            @memset(zeros, 0.0);
            block.ssm_state_dev.upload(&self.device, std.mem.sliceAsBytes(zeros)) catch {};
        }
    }
}

pub fn resetAttentionCpuStates(self: anytype) void {
    for (self.block_runtime.blocks) |*layer| {
        const block = layer.attention_binding orelse continue;
        if (block.cpu_cache) |*cache| cache.resetCache();
    }
}

pub fn ensureGatedDeltaHostStageCapacity(self: anytype, elements: usize) !void {
    if (elements == 0) return error.InvalidArgument;
    if (self.gated_delta_stage_input_host.len < elements) {
        if (self.gated_delta_stage_input_host.len > 0) self.allocator.free(self.gated_delta_stage_input_host);
        self.gated_delta_stage_input_host = try self.allocator.alloc(f32, elements);
    }
    if (self.gated_delta_stage_mid_host.len < elements) {
        if (self.gated_delta_stage_mid_host.len > 0) self.allocator.free(self.gated_delta_stage_mid_host);
        self.gated_delta_stage_mid_host = try self.allocator.alloc(f32, elements);
    }
    if (self.gated_delta_stage_output_host.len < elements) {
        if (self.gated_delta_stage_output_host.len > 0) self.allocator.free(self.gated_delta_stage_output_host);
        self.gated_delta_stage_output_host = try self.allocator.alloc(f32, elements);
    }
}
