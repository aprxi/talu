//! Shared model-system descriptor types.

pub const ModelDescriptor = struct {
    id: []const u8,
    family: []const u8,
    version: []const u8,
    model_types: []const []const u8,
};
