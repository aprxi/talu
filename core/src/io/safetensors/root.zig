//! SafeTensors binary format support
//!
//! SafeTensors is a simple, safe tensor storage format.
//! It stores tensors in a flat binary file with a JSON header containing metadata.

pub const reader = @import("reader.zig");
pub const writer = @import("writer.zig");
pub const sharded = @import("sharded.zig");

// Re-export commonly used types
pub const SafeTensors = reader.SafeTensors;
pub const ShardedSafeTensors = sharded.ShardedSafeTensors;
pub const UnifiedSafeTensors = sharded.UnifiedSafeTensors;
pub const LoadError = reader.LoadError;
pub const tryGetBytes = reader.tryGetBytes;
pub const isShardedModel = sharded.isShardedModel;

pub const TensorEntry = writer.TensorEntry;
pub const WriteError = writer.WriteError;
pub const Builder = writer.Builder;
pub const write = writer.write;
pub const writeToFile = writer.writeToFile;
pub const writeWithMetadata = writer.writeWithMetadata;
pub const writeToFileWithMetadata = writer.writeToFileWithMetadata;
