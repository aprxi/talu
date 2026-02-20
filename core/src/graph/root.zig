//! Graph Subsystem
//!
//! Parses and compiles compute graphs from Python model definitions.
//!
//! ## Pipeline
//!
//! ```
//! JSON graph definitions (runtime/custom registration path)
//!     ↓ parse (parser.zig)
//! Op[] (intermediate representation)
//!     ↓ compile (compiler.zig)
//! LayerOp[] (executable bytecode)
//!     ↓ execute (model/block.zig)
//! SIMD Kernels
//! ```
//!
//! ## Usage
//!
//! ```zig
//! const graph = @import("graph/root.zig");
//!
//! // Initialize registry
//! graph.init(allocator);
//!
//! // Load architecture from JSON string
//! try graph.loadFromJson(json_string);
//!
//! // Get architecture by model_type
//! if (graph.detectFromModelType("model_type")) |arch| {
//!     const program = try graph.ensureCompiled(arch);
//!     // Execute program...
//! }
//! ```

const std = @import("std");

// Re-export types
pub const types = @import("types.zig");
pub const Op = types.Op;
pub const OpType = types.OpType;
pub const OpInput = types.OpInput;
pub const Architecture = types.Architecture;

// Re-export registry functions
const registry = @import("registry.zig");
pub const init = registry.init;
pub const deinit = registry.deinit;
pub const register = registry.register;
pub const loadFromFile = registry.loadFromFile;
pub const loadFromJson = registry.loadFromJson;
pub const get = registry.get;
pub const has = registry.has;
pub const detectFromModelType = registry.detectFromModelType;
pub const listNames = registry.listNames;
pub const ensureCompiled = registry.ensureCompiled;
pub const ensureCompiledForLayer = registry.ensureCompiledForLayer;
pub const ensureCompiledForLayerWithOverride = registry.ensureCompiledForLayerWithOverride;
pub const getAllocator = registry.getAllocator;

// Re-export parser for direct use
pub const parser = @import("parser.zig");
pub const parseFromJson = parser.parseFromJson;

// Re-export compiler for direct use
pub const compiler = @import("compiler.zig");
pub const compile = compiler.compile;

// Re-export layer ops (bytecode format)
pub const layer_ops = @import("layer_ops.zig");
pub const LayerOp = layer_ops.LayerOp;
pub const BufferId = layer_ops.BufferId;
pub const ResidualScale = layer_ops.ResidualScale;

// Re-export graph-owned model config parsing.
pub const config = @import("config/root.zig");

// Re-export model loading from graph/loader.
pub const loader = @import("loader/root.zig");
pub const LoadedModel = loader.LoadedModel;
pub const LoadOptions = loader.LoadOptions;
pub const loadModel = loader.loadModel;
pub const loadArchitectureDefinitions = loader.loadArchitectureDefinitions;
pub const validateLoadedModel = loader.validateLoadedModel;
