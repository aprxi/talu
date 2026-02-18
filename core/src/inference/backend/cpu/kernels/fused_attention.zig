//! Transitional fused-attention compatibility wrapper.
//!
//! Low-level layout work lives in `compute/cpu/layout_transform.zig`.

const compute = @import("../../../../compute/root.zig");
const attention = @import("attention.zig");

const layout = compute.cpu.layout_transform;

pub const QkvViews = layout.QkvViews;
pub const projectQkv = layout.projectQkv;
pub const FusedAttention = attention.MultiHeadAttention;

test {
    _ = QkvViews;
    _ = projectQkv;
    _ = FusedAttention;
}
