//! Metal model-runtime API boundary.
//!
//! This module is the inference-owned surface for fused/dense model lifecycle
//! and decode entrypoints. Low-level array primitives stay in `compute.metal.graph`.

const compute = @import("../../../compute/root.zig");
const graph = compute.metal.graph;

pub const FusedModelHandle = graph.FusedModelHandle;
pub const DenseModelHandle = graph.DenseModelHandle;

pub const mlx_fused_model_create = graph.mlx_fused_model_create;
pub const mlx_fused_model_set_embeddings = graph.mlx_fused_model_set_embeddings;
pub const mlx_fused_model_set_final = graph.mlx_fused_model_set_final;
pub const mlx_fused_model_set_rope_freqs = graph.mlx_fused_model_set_rope_freqs;
pub const mlx_fused_model_set_arch_config = graph.mlx_fused_model_set_arch_config;
pub const mlx_fused_model_set_scaling_config = graph.mlx_fused_model_set_scaling_config;
pub const mlx_fused_model_set_topology = graph.mlx_fused_model_set_topology;
pub const mlx_fused_model_set_layer = graph.mlx_fused_model_set_layer;
pub const mlx_fused_model_optimize = graph.mlx_fused_model_optimize;
pub const mlx_fused_model_free = graph.mlx_fused_model_free;

pub const mlx_dense_model_create = graph.mlx_dense_model_create;
pub const mlx_dense_model_set_embeddings = graph.mlx_dense_model_set_embeddings;
pub const mlx_dense_model_set_final = graph.mlx_dense_model_set_final;
pub const mlx_dense_model_set_topology = graph.mlx_dense_model_set_topology;
pub const mlx_dense_model_set_layer = graph.mlx_dense_model_set_layer;
pub const mlx_dense_model_free = graph.mlx_dense_model_free;

pub const mlx_fused_decode_step_logits = graph.mlx_fused_decode_step_logits;
pub const mlx_dense_decode_step_logits = graph.mlx_dense_decode_step_logits;
pub const mlx_fused_decode_batch = graph.mlx_fused_decode_batch;
pub const mlx_dense_decode_batch = graph.mlx_dense_decode_batch;

pub const mlx_pipeline_prime = graph.mlx_pipeline_prime;
pub const mlx_pipeline_step = graph.mlx_pipeline_step;
pub const mlx_pipeline_flush = graph.mlx_pipeline_flush;

pub const mlx_dense_pipeline_prime = graph.mlx_dense_pipeline_prime;
pub const mlx_dense_pipeline_step = graph.mlx_dense_pipeline_step;
pub const mlx_dense_pipeline_flush = graph.mlx_dense_pipeline_flush;

