//! Quantization utilities
//!
//! This module provides quantization helpers for GAF (Grouped Affine) format.

pub const grouped_affine = @import("grouped_affine_quant.zig");

// Re-export commonly used items
pub const scaleBiasToF32 = grouped_affine.scaleBiasToF32;
pub const extractNibbles = grouped_affine.extractNibbles;
pub const extract32NibblesToFloat = grouped_affine.extract32NibblesToFloat;
pub const extractBytes = grouped_affine.extractBytes;
