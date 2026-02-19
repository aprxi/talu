//! Static model weight manifest primitives.
//!
//! These are intentionally model-agnostic contract types used by model-version
//! files to describe required/optional tensors and loading transforms.

pub const TensorLayout = enum {
    none,
    linear,
    embedding,
    conv1d_depthwise,
    gaffine,
};

pub const Transform = enum {
    dtype_f32,
    transpose,
    fuse_qkv,
    fuse_gate_up,
};

pub const FieldSpec = struct {
    id: []const u8,
    required: bool = true,
    candidates: []const []const u8,
    layout: TensorLayout = .none,
    transforms: []const Transform = &.{},
};

pub const WeightManifest = struct {
    global: []const FieldSpec = &.{},
    per_layer: []const FieldSpec = &.{},
    weight_prefixes: []const []const u8 = &.{},
};
