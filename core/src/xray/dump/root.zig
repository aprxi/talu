//! Tensor Dump Module
//!
//! Dev-only tooling for capturing full tensors during inference to NPZ files.
//! Used for debugging new model integrations by comparing against PyTorch reference.
//!
//! This is NOT the lightweight emit system - it captures FULL tensor data and
//! goes deep into kernel paths. Only compiled into the talu-dump binary.
//!
//! Usage:
//!   zig build dump -Drelease
//!   ./zig-out/bin/talu-dump --model path/to/model --prompt "Hello" -o /tmp/talu.npz

pub const capture = @import("capture.zig");
pub const npz = @import("npz.zig");

pub const Capture = capture.Capture;
pub const NpzWriter = npz.NpzWriter;
