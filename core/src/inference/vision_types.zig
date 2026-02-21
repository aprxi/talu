//! Shared vision payload types across inference backends.
//!
//! This module defines backend-agnostic image/prefill data structures.

const std = @import("std");
const image_mod = @import("../image/root.zig");

pub const PrefillVisionImage = struct {
    pixels: []f32,
    width: u32,
    height: u32,
    grid: image_mod.VisionGrid,
    token_count: usize,

    pub fn deinit(self: *PrefillVisionImage, allocator: std.mem.Allocator) void {
        if (self.pixels.len > 0) allocator.free(self.pixels);
        self.* = .{
            .pixels = &.{},
            .width = 0,
            .height = 0,
            .grid = .{ .temporal = 0, .height = 0, .width = 0 },
            .token_count = 0,
        };
    }
};

pub const PrefillVisionInput = struct {
    images: []PrefillVisionImage,
    image_token_id: u32,

    pub fn deinit(self: *PrefillVisionInput, allocator: std.mem.Allocator) void {
        for (self.images) |*img| img.deinit(allocator);
        if (self.images.len > 0) allocator.free(self.images);
        self.* = .{ .images = &.{}, .image_token_id = 0 };
    }
};

pub const EncodedVisionOutput = struct {
    merged_embeddings: []f32,
    deepstack_layer_embeddings: []const []const f32,

    pub fn deinit(self: *EncodedVisionOutput, allocator: std.mem.Allocator) void {
        if (self.merged_embeddings.len > 0) allocator.free(self.merged_embeddings);
        if (self.deepstack_layer_embeddings.len > 0) {
            for (self.deepstack_layer_embeddings) |layer_embed| {
                if (layer_embed.len > 0) allocator.free(layer_embed);
            }
            allocator.free(self.deepstack_layer_embeddings);
        }
        self.* = .{
            .merged_embeddings = &.{},
            .deepstack_layer_embeddings = &.{},
        };
    }
};

