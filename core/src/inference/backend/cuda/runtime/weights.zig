//! CUDA runtime device tensor and weight handle types.

const std = @import("std");
const compute = @import("compute_pkg");
const tensor = @import("compute_pkg").tensor;

const Tensor = tensor.Tensor;
const config = @import("config.zig");
const DenseU16Dtype = config.DenseU16Dtype;
const EmbeddingLookupKind = config.EmbeddingLookupKind;

pub const DeviceTensor = struct {
    rows: usize,
    cols: usize,
    buffer: compute.cuda.Buffer,

    pub fn deinit(self: *DeviceTensor, device: *compute.cuda.Device) void {
        self.buffer.deinit(device);
    }

    pub fn byteSize(self: *const DeviceTensor) usize {
        return self.buffer.size;
    }
};

pub const missing_device_tensor: DeviceTensor = std.mem.zeroes(DeviceTensor);
pub const missing_host_tensor: Tensor = std.mem.zeroes(Tensor);

pub const EmbeddingLookup = struct {
    kind: EmbeddingLookupKind,
    dim0: u32,
    dim1: u32,
    hidden_dim: u32,
    layout_tag: u32,
    group_size: u32 = 0,
    scales_dtype_tag: u32 = 0,
    scales: ?compute.cuda.Buffer = null,
    biases: ?compute.cuda.Buffer = null,
    multiplier: f32,
    buffer: compute.cuda.Buffer,

    pub fn deinit(self: *EmbeddingLookup, device: *compute.cuda.Device) void {
        if (self.biases) |*buf| buf.deinit(device);
        if (self.scales) |*buf| buf.deinit(device);
        self.buffer.deinit(device);
    }

    pub fn byteSize(self: *const EmbeddingLookup) usize {
        return self.buffer.size +
            (if (self.scales) |buf| buf.size else 0) +
            (if (self.biases) |buf| buf.size else 0);
    }
};

pub const GaffineU4LinearWeight = struct {
    rows: usize,
    cols: usize,
    packed_data: compute.cuda.Buffer,
    scales: compute.cuda.Buffer,
    biases: compute.cuda.Buffer,
    group_size: u32,
    scales_dtype_tag: u32,
    dequant_f16_cache: compute.cuda.Buffer = .{ .pointer = 0, .size = 0 },
    dequant_i8_cache: compute.cuda.Buffer = .{ .pointer = 0, .size = 0 },
    mean_scale_cache: compute.cuda.Buffer = .{ .pointer = 0, .size = 0 },

    pub fn deinit(self: *GaffineU4LinearWeight, device: *compute.cuda.Device) void {
        if (self.mean_scale_cache.pointer != 0) self.mean_scale_cache.deinit(device);
        if (self.dequant_i8_cache.pointer != 0) self.dequant_i8_cache.deinit(device);
        if (self.dequant_f16_cache.pointer != 0) self.dequant_f16_cache.deinit(device);
        self.biases.deinit(device);
        self.scales.deinit(device);
        self.packed_data.deinit(device);
    }

    pub fn byteSize(self: *const GaffineU4LinearWeight) usize {
        return self.packed_data.size + self.scales.size + self.biases.size +
            self.dequant_f16_cache.size + self.dequant_i8_cache.size + self.mean_scale_cache.size;
    }
};

pub const GaffineU8LinearWeight = GaffineU4LinearWeight;

pub const U16LinearWeight = struct {
    rows: usize,
    cols: usize,
    buffer: compute.cuda.Buffer,
    dtype: DenseU16Dtype,

    pub fn deinit(self: *U16LinearWeight, device: *compute.cuda.Device) void {
        self.buffer.deinit(device);
    }

    pub fn byteSize(self: *const U16LinearWeight) usize {
        return self.buffer.size;
    }
};

pub const Fp8LinearWeight = struct {
    rows: usize,
    cols: usize,
    buffer: compute.cuda.Buffer,
    /// GPU buffer holding BF16 per-block scales [scale_rows × scale_cols]
    scales_buffer: compute.cuda.Buffer,
    scale_rows: u32,
    scale_cols: u32,
    block_size: u32,
    /// Per-tensor scale (used only when scale_rows == 1 && scale_cols == 1)
    weight_scale_inv: f32,

    pub fn deinit(self: *Fp8LinearWeight, device: *compute.cuda.Device) void {
        if (self.scales_buffer.size > 0) self.scales_buffer.deinit(device);
        self.buffer.deinit(device);
    }

    pub fn byteSize(self: *const Fp8LinearWeight) usize {
        return self.buffer.size + self.scales_buffer.size;
    }
};

pub const Mxfp8LinearWeight = struct {
    rows: usize,
    cols: usize,
    /// GPU buffer holding E4M3 weight bytes [rows × cols]
    buffer: compute.cuda.Buffer,
    /// GPU buffer holding UE8M0 block-32 scales in cuBLASLt interleaved layout.
    /// Padded to 128-tile boundaries: [padded_outer × padded_sf_k] bytes.
    scales_buffer: compute.cuda.Buffer,
    /// GPU buffer holding UE8M0 block-32 scales in simple row-major layout
    /// [cols × scale_cols] for the GEMV kernel path. Same data, different layout.
    scales_raw_buffer: compute.cuda.Buffer,
    scale_cols: u32,

    pub fn deinit(self: *Mxfp8LinearWeight, device: *compute.cuda.Device) void {
        if (self.scales_raw_buffer.size > 0) self.scales_raw_buffer.deinit(device);
        if (self.scales_buffer.size > 0) self.scales_buffer.deinit(device);
        self.buffer.deinit(device);
    }

    pub fn byteSize(self: *const Mxfp8LinearWeight) usize {
        return self.buffer.size + self.scales_buffer.size + self.scales_raw_buffer.size;
    }

    /// Compute cuBLASLt-required scale tensor size for VEC32_UE8M0 block scaling.
    /// inner = contraction dimension (K), outer = non-contraction dimension (M or N).
    /// Returns the total number of UE8M0 scale bytes needed (padded to 128-tile boundaries).
    pub fn cublasLtScaleTensorSize(inner: usize, outer: usize) usize {
        const block_rows: usize = 128; // inner dimension tile
        const block_cols: usize = 128; // outer dimension tile
        const s_rows = roundoff(inner, block_rows) / 32;
        const s_cols = roundoff(outer, block_cols);
        return s_rows * s_cols;
    }

    pub fn roundoff(x: usize, granul: usize) usize {
        return granul * ((x + (granul - 1)) / granul);
    }
};

pub const Nvfp4LinearWeight = struct {
    rows: usize,
    cols: usize,
    /// Packed FP4 bytes: [out_dim × packed_in] (2 FP4 values per byte).
    buffer: compute.cuda.Buffer,
    /// FP8 E4M3 scales in row-major layout [out_dim × scale_cols].
    scales_buffer: compute.cuda.Buffer,
    /// FP8 UE4M3 block-16 scales in cuBLASLt interleaved layout.
    /// Padded to 128-tile boundaries: [padded_outer × padded_sf_k] bytes.
    scales_lt_buffer: compute.cuda.Buffer,
    packed_cols: u32,
    scale_cols: u32,
    group_size: u32,
    weight_global_scale: f32,
    /// Optional persistent INT8 cache for fast decode kernels.
    /// Built once at model init via NVFP4->I8 conversion.
    dequant_i8_cache: compute.cuda.Buffer = .{ .pointer = 0, .size = 0 },
    /// Per-output-row scales for dequant_i8_cache.
    mean_scale_cache: compute.cuda.Buffer = .{ .pointer = 0, .size = 0 },

    pub fn deinit(self: *Nvfp4LinearWeight, device: *compute.cuda.Device) void {
        if (self.mean_scale_cache.pointer != 0) self.mean_scale_cache.deinit(device);
        if (self.dequant_i8_cache.pointer != 0) self.dequant_i8_cache.deinit(device);
        if (self.scales_lt_buffer.size > 0) self.scales_lt_buffer.deinit(device);
        if (self.scales_buffer.size > 0) self.scales_buffer.deinit(device);
        self.buffer.deinit(device);
    }

    pub fn byteSize(self: *const Nvfp4LinearWeight) usize {
        return self.buffer.size +
            self.scales_buffer.size +
            self.scales_lt_buffer.size +
            self.dequant_i8_cache.size +
            self.mean_scale_cache.size;
    }

    /// Compute cuBLASLt-required scale tensor size for VEC16_UE4M3 block scaling.
    /// inner = contraction dimension (K), outer = non-contraction dimension (M or N).
    /// Returns total UE4M3 scale bytes padded to cuBLASLt tile boundaries.
    pub fn cublasLtScaleTensorSize(inner: usize, outer: usize) usize {
        const sf_k = roundoff((inner + 15) / 16, 4);
        return roundoff(outer, 128) * sf_k;
    }

    pub fn roundoff(x: usize, granul: usize) usize {
        return granul * ((x + (granul - 1)) / granul);
    }
};

pub const LinearWeight = union(enum) {
    dense_f32: DeviceTensor,
    dense_u16: U16LinearWeight,
    gaffine_u4: GaffineU4LinearWeight,
    gaffine_u8: GaffineU8LinearWeight,
    fp8: Fp8LinearWeight,
    mxfp8: Mxfp8LinearWeight,
    nvfp4: Nvfp4LinearWeight,

    pub fn deinit(self: *LinearWeight, device: *compute.cuda.Device) void {
        switch (self.*) {
            .dense_f32 => |*w| w.deinit(device),
            .dense_u16 => |*w| w.deinit(device),
            .gaffine_u4 => |*w| w.deinit(device),
            .gaffine_u8 => |*w| w.deinit(device),
            .fp8 => |*w| w.deinit(device),
            .mxfp8 => |*w| w.deinit(device),
            .nvfp4 => |*w| w.deinit(device),
        }
    }

    pub fn rows(self: *const LinearWeight) usize {
        return switch (self.*) {
            .dense_f32 => |w| w.rows,
            .dense_u16 => |w| w.rows,
            .gaffine_u4 => |w| w.rows,
            .gaffine_u8 => |w| w.rows,
            .fp8 => |w| w.rows,
            .mxfp8 => |w| w.rows,
            .nvfp4 => |w| w.rows,
        };
    }

    pub fn cols(self: *const LinearWeight) usize {
        return switch (self.*) {
            .dense_f32 => |w| w.cols,
            .dense_u16 => |w| w.cols,
            .gaffine_u4 => |w| w.cols,
            .gaffine_u8 => |w| w.cols,
            .fp8 => |w| w.cols,
            .mxfp8 => |w| w.cols,
            .nvfp4 => |w| w.cols,
        };
    }

    pub fn byteSize(self: *const LinearWeight) usize {
        return switch (self.*) {
            .dense_f32 => |w| w.buffer.size,
            .dense_u16 => |w| w.byteSize(),
            .gaffine_u4 => |w| w.byteSize(),
            .gaffine_u8 => |w| w.byteSize(),
            .fp8 => |w| w.byteSize(),
            .mxfp8 => |w| w.byteSize(),
            .nvfp4 => |w| w.byteSize(),
        };
    }
};
