//! Package root for named `compute_pkg` imports.

const compute = @import("compute/root.zig");

pub const device = compute.device;
pub const parallel = compute.parallel;
pub const dlpack = compute.dlpack;
pub const mmap_policy = compute.mmap_policy;
pub const metal = compute.metal;
pub const cuda = compute.cuda;
pub const cpu = compute.cpu;

pub const Device = compute.Device;
pub const DeviceType = compute.DeviceType;
pub const ThreadPool = compute.ThreadPool;
pub const DLDataType = compute.DLDataType;
pub const DLDevice = compute.DLDevice;
pub const DLTensor = compute.DLTensor;
pub const DLManagedTensor = compute.DLManagedTensor;
pub const MatmulScratch = compute.MatmulScratch;
pub const TensorView = compute.TensorView;
