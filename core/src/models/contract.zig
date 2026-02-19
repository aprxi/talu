//! Static model contract.
//!
//! This module defines the runtime contract that `core/src/models/*` will own.
//! Phase 1 introduces the types; execution wiring is migrated in later phases.

const common_types = @import("common/types.zig");
const manifest_mod = @import("common/weight_manifest.zig");
const mem_mod = @import("common/memory_requirements.zig");

pub const ModelDescriptor = common_types.ModelDescriptor;
pub const WeightManifest = manifest_mod.WeightManifest;
pub const FieldSpec = manifest_mod.FieldSpec;
pub const TensorLayout = manifest_mod.TensorLayout;
pub const Transform = manifest_mod.Transform;
pub const MemoryRequirements = mem_mod.MemoryRequirements;
pub const KVCacheRequirements = mem_mod.KVCacheRequirements;
pub const ScratchRequirements = mem_mod.ScratchRequirements;

pub const BackendKind = enum {
    cpu,
    metal,
    cuda,
};

/// Placeholder runtime handle union for phased migration.
/// Concrete topology runtime entry points are added in subsequent phases.
pub const ModelRuntime = union(enum) {
    none: void,
    cpu: *anyopaque,
    metal: *anyopaque,
    cuda: *anyopaque,
};
