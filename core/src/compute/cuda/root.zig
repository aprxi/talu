//! CUDA GPU compute primitives.
//!
//! Current phase includes runtime probing plus device/buffer lifecycle
//! primitives. Math kernels and model execution remain unimplemented.

pub const capabilities = @import("capabilities.zig");
pub const device = @import("device.zig");

pub const Device = device.Device;
pub const Buffer = device.Buffer;
pub const Probe = device.Probe;
pub const probeRuntime = device.probeRuntime;

test {
    _ = capabilities;
    _ = device;
}
