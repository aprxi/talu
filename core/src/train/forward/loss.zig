//! Cross-entropy loss computation for training forward pass.
//!
//! Re-exports from the existing loss module. The loss.zig at the train/ level
//! owns the implementation; this module provides a convenient forward/ namespace.

const loss_mod = @import("../loss.zig");

pub const crossEntropyLoss = loss_mod.crossEntropyLoss;
