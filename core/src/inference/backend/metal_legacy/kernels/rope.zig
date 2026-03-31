//! Metal RoPE kernel surface.

pub const supported = true;

const compute = @import("../../../../compute/root.zig");

const mlx_graph = compute.metal.graph;
const ArrayHandle = mlx_graph.ArrayHandle;

pub const RotaryEmbedding = struct {
    /// Canonical kernel-call contract for backend parity checks.
    pub const ForwardParams = struct {
        input_vector: ArrayHandle,
        output_vector: *ArrayHandle,
        position: usize,
    };

    head_dim: usize,
    rope_theta: f32,

    pub fn forward(
        self: *const RotaryEmbedding,
        input_vector: ArrayHandle,
        output_vector: *ArrayHandle,
        position: usize,
    ) void {
        output_vector.* = mlx_graph.mlx_lazy_rope(input_vector, self.head_dim, position, self.rope_theta);
    }
};

test {
    _ = RotaryEmbedding;
}
