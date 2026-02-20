//! Unified Tensor type for the entire codebase.
//!
//! This is THE tensor type - used everywhere from model loading to C API.
//! - DLPack-compatible for Python/C interop
//! - Stride-aware (with contiguity assertions)
//! - Supports up to 8 dimensions
//! - Supports quantized dtypes (MLX, GGML, MXFP4)
//!
//! Design: All tensors MUST be contiguous. Non-contiguous tensors from Python
//! will panic immediately. Call .contiguous() in Python before passing to talu.

const std = @import("std");
const builtin = @import("builtin");
const dtype_mod = @import("dtype.zig");

const c = @cImport({
    @cInclude("stdlib.h");
});

// Import dtype types used by Tensor
pub const DType = dtype_mod.DType;
pub const GroupedAffineMeta = dtype_mod.GroupedAffineMeta;

/// Memory alignment constants
pub const mem = struct {
    pub const huge_page_size: usize = 2 * 1024 * 1024;
    pub const cache_line: usize = 64;
    pub const simd_alignment: usize = 32; // AVX2 alignment
};

/// Maximum number of dimensions supported
pub const MAX_NDIM: usize = 8;

// =============================================================================
// DLPack Protocol Types
// =============================================================================

/// DLPack device type codes
pub const DLDeviceType = enum(i32) {
    kDLCPU = 1,
    kDLCUDA = 2,
    kDLCUDAHost = 3,
    kDLOpenCL = 4,
    kDLVulkan = 7,
    kDLMetal = 8,
    kDLVPI = 9,
    kDLROCM = 10,
    kDLROCMHost = 11,
    kDLExtDev = 12,
    kDLCUDAManaged = 13,
    kDLOneAPI = 14,
    kDLWebGPU = 15,
    kDLHexagon = 16,
};

/// DLPack data type codes
pub const DLDataTypeCode = enum(u8) {
    kDLInt = 0,
    kDLUInt = 1,
    kDLFloat = 2,
    kDLBfloat = 4,
    kDLComplex = 5,
    kDLBool = 6,
};

/// Device descriptor
pub const Device = extern struct {
    device_type: DLDeviceType,
    device_id: i32,

    pub fn cpu() Device {
        var d = std.mem.zeroes(Device);
        d.device_type = .kDLCPU;
        d.device_id = 0;
        return d;
    }

    pub fn metal(device_id: i32) Device {
        var d = std.mem.zeroes(Device);
        d.device_type = .kDLMetal;
        d.device_id = device_id;
        return d;
    }

    pub fn cuda(device_id: i32) Device {
        var d = std.mem.zeroes(Device);
        d.device_type = .kDLCUDA;
        d.device_id = device_id;
        return d;
    }
};

/// DLDataType for DLPack protocol
pub const DLDataType = extern struct {
    code: DLDataTypeCode,
    bits: u8,
    lanes: u16,

    pub fn float32() DLDataType {
        var t = std.mem.zeroes(DLDataType);
        t.code = .kDLFloat;
        t.bits = 32;
        t.lanes = 1;
        return t;
    }

    fn make(code: DLDataTypeCode, bits: u8, lanes: u16) DLDataType {
        var t = std.mem.zeroes(DLDataType);
        t.code = code;
        t.bits = bits;
        t.lanes = lanes;
        return t;
    }

    pub fn fromDType(dt: DType) DLDataType {
        return switch (dt) {
            .f32 => make(.kDLFloat, 32, 1),
            .f64 => make(.kDLFloat, 64, 1),
            .f16 => make(.kDLFloat, 16, 1),
            .bf16 => make(.kDLBfloat, 16, 1),
            .i8 => make(.kDLInt, 8, 1),
            .i16 => make(.kDLInt, 16, 1),
            .i32 => make(.kDLInt, 32, 1),
            .i64 => make(.kDLInt, 64, 1),
            .u8 => make(.kDLUInt, 8, 1),
            .u16 => make(.kDLUInt, 16, 1),
            .u32 => make(.kDLUInt, 32, 1),
            .u64 => make(.kDLUInt, 64, 1),
            // Quantized types appear as u8 arrays
            .grouped_affine_u4, .grouped_affine_u8, .mxfp4, .f8_e4m3 => make(.kDLUInt, 8, 1),
        };
    }
};

/// DLTensor - the core DLPack tensor descriptor
pub const DLTensor = extern struct {
    data: ?*anyopaque,
    device: Device,
    ndim: i32,
    dtype: DLDataType,
    shape: [*]i64,
    strides: ?[*]i64,
    byte_offset: u64,
};

/// Deleter function type for DLManagedTensor
pub const DLManagedTensorDeleter = *const fn (*DLManagedTensor) callconv(.c) void;

/// DLManagedTensor - tensor with lifecycle management
pub const DLManagedTensor = extern struct {
    dl_tensor: DLTensor,
    manager_ctx: ?*anyopaque,
    deleter: ?DLManagedTensorDeleter,
};

// =============================================================================
// Unified Tensor Type
// =============================================================================

/// Unified tensor type - THE tensor for the entire codebase.
/// DLPack-compatible, stride-aware, supports quantized types.
pub const Tensor = struct {
    /// Data type (full, including quantized)
    dtype: DType,
    /// Number of dimensions (i32 to match DLPack conventions)
    n_dims: i32,
    /// Shape array (8 dimensions, i64 to match DLPack conventions)
    shape: [MAX_NDIM]i64,
    /// Pointer to the raw data
    data_ptr: ?[*]u8,
    /// Total byte size of data
    data_size: usize,
    /// Strides in elements (not bytes)
    strides: [MAX_NDIM]i64 = .{ 0, 0, 0, 0, 0, 0, 0, 0 },
    /// Device location
    device: Device = Device.cpu(),
    /// Total number of elements
    numel: usize = 0,
    /// Whether this tensor owns its data
    owns_data: bool = false,
    /// Grouped-affine quantization metadata (optional)
    gaffine: ?GroupedAffineMeta = null,

    const Self = @This();

    // =========================================================================
    // Creation
    // =========================================================================

    /// Create a tensor with allocated memory (uses libc malloc for FFI compat)
    pub fn init(allocator: std.mem.Allocator, shape_slice: []const i64, dtype: DType, device: Device) !*Self {
        var tensor = try allocator.create(Self);
        errdefer allocator.destroy(tensor);

        var numel: usize = 1;
        for (shape_slice, 0..) |dim, dim_idx| {
            tensor.shape[dim_idx] = dim;
            numel *= @intCast(dim);
        }
        tensor.n_dims = @intCast(shape_slice.len);
        tensor.numel = numel;
        tensor.dtype = dtype;
        tensor.device = device;
        tensor.owns_data = true;
        tensor.gaffine = null;

        // Calculate strides (row-major / C-contiguous)
        var stride: i64 = 1;
        var dim_idx: usize = shape_slice.len;
        while (dim_idx > 0) {
            dim_idx -= 1;
            tensor.strides[dim_idx] = stride;
            stride *= shape_slice[dim_idx];
        }

        // Zero out unused slots
        for (shape_slice.len..MAX_NDIM) |slot_idx| {
            tensor.shape[slot_idx] = 0;
            tensor.strides[slot_idx] = 0;
        }

        // Allocate data
        const elem_size = dtype.elementSize();
        const byte_size = numel * elem_size;
        const raw_ptr = c.malloc(byte_size) orelse return error.OutOfMemory;
        tensor.data_ptr = @ptrCast(raw_ptr);
        tensor.data_size = byte_size;

        return tensor;
    }

    /// Create a non-owning view from existing data.
    ///
    /// For standard dtypes (f32, f16, etc.), data_size can be null and will be
    /// computed from shape. For quantized dtypes (mxfp4, q4_k, etc.), data_size
    /// must be provided since the physical storage size differs from logical shape.
    pub fn view(data_ptr: [*]u8, shape_slice: []const usize, dtype: DType, data_size: ?usize) Self {
        var tensor: Self = undefined;
        tensor.data_ptr = data_ptr;
        tensor.dtype = dtype;
        tensor.device = Device.cpu();
        tensor.owns_data = false;
        tensor.gaffine = null;
        tensor.n_dims = @intCast(shape_slice.len);

        var numel: usize = 1;
        for (shape_slice, 0..) |dim, dim_idx| {
            tensor.shape[dim_idx] = @intCast(dim);
            numel *= dim;
        }
        tensor.numel = numel;

        // Compute data_size: use explicit value if provided, otherwise compute from shape
        const computed_size = numel * dtype.elementSize();
        if (data_size) |size| {
            tensor.data_size = size;
        } else if (dtype.isQuantized()) {
            @panic("data_size required for quantized dtype - cannot compute from shape");
        } else {
            tensor.data_size = computed_size;
        }

        // Compute contiguous strides
        var stride: i64 = 1;
        var dim_idx: usize = shape_slice.len;
        while (dim_idx > 0) {
            dim_idx -= 1;
            tensor.strides[dim_idx] = stride;
            stride *= @intCast(shape_slice[dim_idx]);
        }

        for (shape_slice.len..MAX_NDIM) |slot_idx| {
            tensor.shape[slot_idx] = 0;
            tensor.strides[slot_idx] = 0;
        }

        return tensor;
    }

    /// Free the tensor
    pub fn deinit(self: *Self, alloc: std.mem.Allocator) void {
        if (self.owns_data) {
            if (self.data_ptr) |ptr| {
                c.free(ptr);
            }
        }
        alloc.destroy(self);
    }

    // =========================================================================
    // Data access
    // =========================================================================

    /// Get raw data as byte slice.
    pub fn data(self: *const Self) []u8 {
        if (self.data_ptr) |ptr| {
            return ptr[0..self.data_size];
        }
        return &[_]u8{};
    }

    /// Get first 4 dimensions as usize array (for APIs expecting fixed-size shape).
    pub fn shapeAsUsize(self: *const Self) [4]usize {
        return .{
            @intCast(self.shape[0]),
            @intCast(self.shape[1]),
            @intCast(self.shape[2]),
            @intCast(self.shape[3]),
        };
    }

    pub fn asSlice(self: *const Self, comptime T: type) []T {
        if (self.data_ptr) |ptr| {
            const typed: [*]T = @ptrCast(@alignCast(ptr));
            return typed[0..self.numel];
        }
        return &[_]T{};
    }

    pub fn asSliceMut(self: *Self, comptime T: type) []T {
        if (self.data_ptr) |ptr| {
            const typed: [*]T = @ptrCast(@alignCast(ptr));
            return typed[0..self.numel];
        }
        return &[_]T{};
    }

    pub fn asSliceUnaligned(self: *const Self, comptime T: type) []align(1) T {
        if (self.data_ptr) |ptr| {
            return @as([*]align(1) T, @ptrCast(ptr))[0..self.numel];
        }
        return @as([*]align(1) T, undefined)[0..0];
    }

    pub inline fn rowPtr(self: *const Self, comptime T: type, row: usize) [*]T {
        const cols: usize = @intCast(self.shape[1]);
        if (self.data_ptr) |ptr| {
            const aligned: [*]align(@alignOf(T)) u8 = @alignCast(ptr);
            return @as([*]T, @ptrCast(aligned)) + row * cols;
        }
        return undefined;
    }

    // =========================================================================
    // Contiguity
    // =========================================================================

    pub fn assertContiguous(self: *const Self) void {
        if (!self.isContiguous()) {
            @panic("Non-contiguous tensor not supported. Call .contiguous() in Python first.");
        }
    }

    pub fn isContiguous(self: *const Self) bool {
        if (@as(usize, @intCast(self.n_dims)) == 0) return true;

        var expected_stride: i64 = 1;
        var dim_idx: usize = @as(usize, @intCast(self.n_dims));
        while (dim_idx > 0) {
            dim_idx -= 1;
            if (self.strides[dim_idx] != 0 and self.strides[dim_idx] != expected_stride) return false;
            expected_stride *= self.shape[dim_idx];
        }
        return true;
    }

    // =========================================================================
    // DType conversion (for FFI boundaries)
    // =========================================================================

    /// Get FFI-compatible dtype. Quantized types return .u8.
    pub fn simpleDType(self: *const Self) DType {
        return switch (self.dtype) {
            // Standard types pass through
            .f32, .f64, .f16, .bf16, .i8, .i16, .i32, .i64, .u8, .u16, .u32, .u64 => self.dtype,
            // Quantized types appear as u8 externally
            .f8_e4m3, .grouped_affine_u4, .grouped_affine_u8, .mxfp4 => .u8,
        };
    }

    // =========================================================================
    // Device checks
    // =========================================================================

    pub fn isCPU(self: *const Self) bool {
        return self.device.device_type == .kDLCPU;
    }

    // =========================================================================
    // Convenience view constructors (f32 tensors from existing data)
    // =========================================================================

    /// Create a 2D f32 tensor view from byte slice.
    pub fn view2D(data_slice: []u8, rows: usize, cols: usize) Self {
        var tensor: Self = undefined;
        tensor.data_ptr = data_slice.ptr;
        tensor.dtype = .f32;
        tensor.device = Device.cpu();
        tensor.owns_data = false;
        tensor.gaffine = null;
        tensor.n_dims = 2;
        tensor.shape = .{ @intCast(rows), @intCast(cols), 0, 0, 0, 0, 0, 0 };
        tensor.numel = rows * cols;
        tensor.data_size = data_slice.len;
        tensor.strides = .{ @intCast(cols), 1, 0, 0, 0, 0, 0, 0 };
        return tensor;
    }

    pub fn view3D(data_slice: []u8, rows: usize, cols: usize) Self {
        var tensor: Self = undefined;
        tensor.data_ptr = data_slice.ptr;
        tensor.dtype = .f32;
        tensor.device = Device.cpu();
        tensor.owns_data = false;
        tensor.gaffine = null;
        tensor.n_dims = 3;
        tensor.shape = .{ 1, @intCast(rows), @intCast(cols), 0, 0, 0, 0, 0 };
        tensor.numel = rows * cols;
        tensor.data_size = data_slice.len;
        tensor.strides = .{ @intCast(rows * cols), @intCast(cols), 1, 0, 0, 0, 0, 0 };
        return tensor;
    }

    pub fn view2DSlice(data_slice: []f32, rows: usize, cols: usize) Self {
        const bytes = std.mem.sliceAsBytes(data_slice[0 .. rows * cols]);
        return view2D(@constCast(bytes), rows, cols);
    }

    pub fn view3DSlice(data_slice: []f32, rows: usize, cols: usize) Self {
        const bytes = std.mem.sliceAsBytes(data_slice[0 .. rows * cols]);
        return view3D(@constCast(bytes), rows, cols);
    }

    // =========================================================================
    // DLPack export
    // =========================================================================

    pub fn toDLPack(self: *Self, allocator: std.mem.Allocator) !*DLManagedTensor {
        const managed = try allocator.create(DLManagedTensor);
        managed.* = std.mem.zeroes(DLManagedTensor);
        managed.dl_tensor.data = self.data_ptr;
        managed.dl_tensor.device = self.device;
        managed.dl_tensor.ndim = @intCast(@as(usize, @intCast(self.n_dims)));
        managed.dl_tensor.dtype = DLDataType.fromDType(self.simpleDType());
        managed.dl_tensor.shape = &self.shape;
        managed.dl_tensor.strides = &self.strides;
        managed.dl_tensor.byte_offset = 0;
        managed.manager_ctx = self;
        managed.deleter = &dlpackDeleter;
        return managed;
    }
};

/// DLDevice alias
pub const DLDevice = Device;

/// DLPack deleter callback
fn dlpackDeleter(managed: *DLManagedTensor) callconv(.c) void {
    if (managed.manager_ctx) |ctx| {
        const tensor: *Tensor = @ptrCast(@alignCast(ctx));
        tensor.deinit(std.heap.c_allocator);
    }
    std.heap.c_allocator.destroy(managed);
}

// =============================================================================
// OwnedTensor - Stack-allocated owning tensor with aligned memory
// =============================================================================

/// Owning tensor with SIMD-aligned memory (uses Zig allocator, not libc)
pub const OwnedTensor = struct {
    allocator: std.mem.Allocator,
    dtype: DType,
    n_dims: i32,
    shape: [4]usize,
    data: []align(mem.simd_alignment) u8,
    data_size: usize,
    gaffine: ?GroupedAffineMeta = null,

    pub fn init(allocator: std.mem.Allocator, dtype: DType, shape: []const usize) !OwnedTensor {
        var fixed_shape: [4]usize = .{0} ** 4;
        if (shape.len > fixed_shape.len) return error.ShapeTooLarge;
        std.mem.copyForwards(usize, fixed_shape[0..shape.len], shape);

        const elem_size: usize = dtype.elementSize();
        var n: usize = 1;
        for (shape) |s| n *= s;
        const total = elem_size * n;

        const data_buf = try allocator.alignedAlloc(u8, .@"32", total);
        @memset(data_buf, 0);

        return .{
            .allocator = allocator,
            .dtype = dtype,
            .n_dims = @intCast(shape.len),
            .shape = fixed_shape,
            .data = data_buf,
            .data_size = total,
        };
    }

    pub fn deinit(self: *OwnedTensor) void {
        self.allocator.free(self.data);
        self.* = undefined;
    }

    pub fn numElements(self: OwnedTensor) usize {
        var total: usize = 1;
        var dim_idx: usize = 0;
        while (dim_idx < self.n_dims) : (dim_idx += 1) {
            total *= self.shape[dim_idx];
        }
        return total;
    }

    pub fn asSlice(self: OwnedTensor, comptime T: type) []T {
        const aligned: [*]align(@alignOf(T)) u8 = @alignCast(self.data.ptr);
        return @as([*]T, @ptrCast(aligned))[0 .. self.data.len / @sizeOf(T)];
    }

    /// Convert to Tensor view
    pub fn toTensor(self: *const OwnedTensor) Tensor {
        var tensor: Tensor = undefined;
        tensor.data_ptr = self.data.ptr;
        tensor.dtype = self.dtype;
        tensor.device = Device.cpu();
        tensor.owns_data = false;
        tensor.gaffine = self.gaffine;
        tensor.n_dims = @intCast(self.n_dims);
        tensor.data_size = self.data_size;

        var numel: usize = 1;
        for (0..@as(usize, @intCast(self.n_dims))) |dim_idx| {
            tensor.shape[dim_idx] = @intCast(self.shape[dim_idx]);
            numel *= self.shape[dim_idx];
        }
        tensor.numel = numel;

        // Compute strides
        var stride: i64 = 1;
        var dim_idx: usize = @intCast(self.n_dims);
        while (dim_idx > 0) {
            dim_idx -= 1;
            tensor.strides[dim_idx] = stride;
            stride *= @intCast(self.shape[dim_idx]);
        }

        for (@as(usize, @intCast(self.n_dims))..MAX_NDIM) |pad_idx| {
            tensor.shape[pad_idx] = 0;
            tensor.strides[pad_idx] = 0;
        }

        return tensor;
    }

    /// Convenience alias for toTensor().
    pub fn view(self: *const OwnedTensor) Tensor {
        return self.toTensor();
    }
};

// =============================================================================
// Model Configuration Types
// =============================================================================

pub const QuantMethod = enum {
    none,
    gaffine,
    mxfp4,
    native, // reserved, not currently used
};

pub const RopeScaling = struct {
    rope_type: enum { none, llama3, linear, yarn } = .none,
    factor: f32 = 1.0,
    low_freq_factor: f32 = 1.0,
    high_freq_factor: f32 = 4.0,
    // YaRN parameters (defaults match reference implementations)
    beta_slow: f32 = 1.0,
    beta_fast: f32 = 32.0,
    attention_factor: f32 = 0.0,
    mscale: f32 = 0.0,
    mscale_all_dim: f32 = 0.0,
    truncate: bool = true,
    original_max_position_embeddings: i32 = 8192,
    /// Optional multimodal RoPE section sizes (model-defined).
    mrope_section: [3]u32 = .{ 0, 0, 0 },
    mrope_interleaved: bool = false,
};

pub const ModelArch = enum {
    custom,
};

pub const ModelRuntime = struct {
    /// Canonical architecture id resolved at load time (e.g. "llama3", "granite_hybrid").
    architecture_id: ?[]const u8 = null,
    /// Architecture capability flags copied from static model metadata.
    has_moe: bool = false,
    has_mamba: bool = false,
    has_shortconv: bool = false,
    has_mla: bool = false,
    weight_offset: f32 = 0.0,
    qk_norm_weight_offset: f32 = 0.0,
    explicit_qk_norm_ops: bool = false,
    use_swiglu_variant: bool = false, // SwiGLU variant: alpha=1.702, clipping, (up+1) formulation
    use_transposed_mxfp4: bool = false,
};

pub const ModelConfig = struct {
    vocab_size: i32,
    d_model: i32,
    n_layers: i32,
    n_heads: i32,
    n_kv_groups: i32,
    d_ff: i32,
    max_seq_len: i32,
    head_dim: i32,
    rope_dim: i32 = 0,
    rope_theta: f32,
    norm_eps: f32,
    gaffine_group_size: i32,
    gaffine_bits: i32 = 4,
    tie_word_embeddings: bool = true,
    num_experts: i32 = 0,
    experts_per_token: i32 = 0,
    attention_bias: bool = false,
    quant_method: QuantMethod = .none,
    rope_scaling: RopeScaling = .{},
    model_arch: ModelArch = .custom,
    use_gelu: bool = false,
    use_qk_norm: bool = false,
    query_pre_attn_scalar: f32 = 0,
    rope_local_theta: f32 = 0,
    sliding_window: i32 = 0,
    sliding_window_pattern: i32 = 0,
    embedding_multiplier: f32 = 1.0,
    attention_multiplier: f32 = 0,
    residual_multiplier: f32 = 1.0,
    logits_scaling: f32 = 1.0,
    bos_token_id: ?i32 = null,
    // Mamba/SSM config (for heterogeneous models like Granite Hybrid)
    mamba_d_state: i32 = 0, // SSM state dimension (e.g., 128)
    mamba_d_conv: i32 = 0, // Convolution kernel size (e.g., 4)
    mamba_n_heads: i32 = 0, // Number of SSM heads
    mamba_d_head: i32 = 0, // Head dimension for Mamba
    mamba_n_groups: i32 = 1, // Groups for B/C projection
    mamba_expand: i32 = 2, // Expansion factor (d_inner = d_model * expand)
    // ShortConv config (for heterogeneous models)
    shortconv_d_conv: i32 = 0, // Convolution kernel size (L_cache, e.g., 3)
    shortconv_conv_dim: i32 = 0, // Intermediate dimension
    shortconv_conv_dim_out: i32 = 0, // Output dimension (usually = d_model)
    shortconv_has_bias: bool = false, // Whether conv has bias
    // Vision encoder config (for multimodal models)
    vision_hidden_size: i32 = 0,
    vision_depth: i32 = 0,
    vision_num_heads: i32 = 0,
    vision_intermediate_size: i32 = 0,
    projector_hidden_size: i32 = 0,
    vision_out_hidden_size: i32 = 0,
    vision_patch_size: i32 = 0,
    vision_spatial_merge_size: i32 = 0,
    vision_temporal_patch_size: i32 = 0,
    vision_num_position_embeddings: i32 = 0,
    vision_max_num_patches: i32 = 0,
    // Vision special token IDs (0 means "unset")
    image_token_id: i32 = 0,
    vision_start_token_id: i32 = 0,
    vision_end_token_id: i32 = 0,
    // Optional vision probe layer indexes (for model-specific deepstack-style injection).
    vision_probe_layer_count: u8 = 0,
    vision_probe_layers: [8]u16 = [_]u16{0} ** 8,
    /// Whether Flash Attention is compatible with this model's head_dim.
    flash_attn_compatible: bool = false,

    /// Layer types for heterogeneous models (e.g., [0, 0, 1, 0, ...] for mamba=0, attention=1).
    /// Parsed from config.json's `layer_types` array and mapped to variant indices.
    /// When present, this overrides the graph's hardcoded layer_map.
    layer_types: ?[]const u8 = null,

    pub fn initFlashAttnCompat(self: *ModelConfig) void {
        self.flash_attn_compatible = switch (self.head_dim) {
            64, 128 => true,
            else => false,
        };
    }
};

// =============================================================================
// Utility Functions
// =============================================================================

fn allocAlignedBytes(allocator: std.mem.Allocator, len: usize, comptime alignment: u29) ![]align(alignment) u8 {
    const result = try allocator.alignedAlloc(u8, alignment, len);
    if (builtin.os.tag == .linux and len >= mem.huge_page_size) {
        _ = std.os.linux.madvise(@ptrCast(result.ptr), result.len, 14);
    }
    return result;
}

fn freeAlignedBytes(allocator: std.mem.Allocator, buffer: []u8) void {
    allocator.free(buffer);
}

pub fn isContiguous(tensor: *const Tensor) bool {
    return tensor.isContiguous();
}

fn isContiguousOwned(tensor: *const OwnedTensor) bool {
    _ = tensor;
    return true;
}

// =============================================================================
// Unit Tests
// =============================================================================

const testing = std.testing;

// Device type constants tests
test "Device.cpu creates CPU device" {
    const device = Device.cpu();
    try testing.expectEqual(DLDeviceType.kDLCPU, device.device_type);
    try testing.expectEqual(@as(i32, 0), device.device_id);
}

test "Device.metal creates Metal device" {
    const device = Device.metal(3);
    try testing.expectEqual(DLDeviceType.kDLMetal, device.device_type);
    try testing.expectEqual(@as(i32, 3), device.device_id);
}

test "Device.cuda creates CUDA device" {
    const device = Device.cuda(2);
    try testing.expectEqual(DLDeviceType.kDLCUDA, device.device_type);
    try testing.expectEqual(@as(i32, 2), device.device_id);
}

// DLDataType helpers tests
test "DLDataType.float32 creates correct descriptor" {
    const dt = DLDataType.float32();
    try testing.expectEqual(DLDataTypeCode.kDLFloat, dt.code);
    try testing.expectEqual(@as(u8, 32), dt.bits);
    try testing.expectEqual(@as(u16, 1), dt.lanes);
}

test "DLDataType.fromDType converts standard types correctly" {
    // Float types
    var dt = DLDataType.fromDType(.f32);
    try testing.expectEqual(DLDataTypeCode.kDLFloat, dt.code);
    try testing.expectEqual(@as(u8, 32), dt.bits);

    dt = DLDataType.fromDType(.f16);
    try testing.expectEqual(DLDataTypeCode.kDLFloat, dt.code);
    try testing.expectEqual(@as(u8, 16), dt.bits);

    dt = DLDataType.fromDType(.bf16);
    try testing.expectEqual(DLDataTypeCode.kDLBfloat, dt.code);
    try testing.expectEqual(@as(u8, 16), dt.bits);

    // Integer types
    dt = DLDataType.fromDType(.i32);
    try testing.expectEqual(DLDataTypeCode.kDLInt, dt.code);
    try testing.expectEqual(@as(u8, 32), dt.bits);

    dt = DLDataType.fromDType(.u8);
    try testing.expectEqual(DLDataTypeCode.kDLUInt, dt.code);
    try testing.expectEqual(@as(u8, 8), dt.bits);

    // Quantized types appear as u8
    dt = DLDataType.fromDType(.mxfp4);
    try testing.expectEqual(DLDataTypeCode.kDLUInt, dt.code);
    try testing.expectEqual(@as(u8, 8), dt.bits);
}

// Tensor lifecycle tests
test "Tensor.init creates tensor with correct shape and strides" {
    const shape = [_]i64{ 3, 4 };
    const tensor = try Tensor.init(testing.allocator, &shape, .f32, Device.cpu());
    defer tensor.deinit(testing.allocator);

    try testing.expectEqual(@as(i32, 2), tensor.n_dims);
    try testing.expectEqual(@as(i64, 3), tensor.shape[0]);
    try testing.expectEqual(@as(i64, 4), tensor.shape[1]);
    try testing.expectEqual(@as(usize, 12), tensor.numel);
    try testing.expectEqual(DType.f32, tensor.dtype);
    try testing.expect(tensor.owns_data);
    try testing.expectEqual(@as(i64, 4), tensor.strides[0]);
    try testing.expectEqual(@as(i64, 1), tensor.strides[1]);
}

test "Tensor.init handles 1D tensors" {
    const shape = [_]i64{10};
    const tensor = try Tensor.init(testing.allocator, &shape, .f16, Device.cpu());
    defer tensor.deinit(testing.allocator);

    try testing.expectEqual(@as(i32, 1), tensor.n_dims);
    try testing.expectEqual(@as(i64, 10), tensor.shape[0]);
    try testing.expectEqual(@as(usize, 10), tensor.numel);
    try testing.expectEqual(@as(i64, 1), tensor.strides[0]);
}

test "Tensor.init handles 3D tensors" {
    const shape = [_]i64{ 2, 3, 4 };
    const tensor = try Tensor.init(testing.allocator, &shape, .f32, Device.cpu());
    defer tensor.deinit(testing.allocator);

    try testing.expectEqual(@as(i32, 3), tensor.n_dims);
    try testing.expectEqual(@as(i64, 2), tensor.shape[0]);
    try testing.expectEqual(@as(i64, 3), tensor.shape[1]);
    try testing.expectEqual(@as(i64, 4), tensor.shape[2]);
    try testing.expectEqual(@as(usize, 24), tensor.numel);
    try testing.expectEqual(@as(i64, 12), tensor.strides[0]);
    try testing.expectEqual(@as(i64, 4), tensor.strides[1]);
    try testing.expectEqual(@as(i64, 1), tensor.strides[2]);
}

test "Tensor.view creates non-owning view" {
    var data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const bytes = std.mem.sliceAsBytes(data[0..]);
    const shape = [_]usize{ 2, 3 };

    const tensor = Tensor.view(@constCast(bytes.ptr), &shape, .f32, null);

    try testing.expectEqual(@as(i32, 2), tensor.n_dims);
    try testing.expectEqual(@as(i64, 2), tensor.shape[0]);
    try testing.expectEqual(@as(i64, 3), tensor.shape[1]);
    try testing.expectEqual(@as(usize, 6), tensor.numel);
    try testing.expect(!tensor.owns_data);
    try testing.expectEqual(@as(i64, 3), tensor.strides[0]);
    try testing.expectEqual(@as(i64, 1), tensor.strides[1]);
}

test "Tensor.view computes data_size for non-quantized types" {
    var data = [_]f32{ 1, 2, 3, 4 };
    const bytes = std.mem.sliceAsBytes(data[0..]);
    const shape = [_]usize{ 2, 2 };

    const tensor = Tensor.view(@constCast(bytes.ptr), &shape, .f32, null);
    try testing.expectEqual(@as(usize, 16), tensor.data_size); // 4 elements * 4 bytes
}

test "Tensor.view uses explicit data_size when provided" {
    var data = [_]u8{0} ** 100;
    const shape = [_]usize{ 10, 10 };

    const tensor = Tensor.view(&data, &shape, .f32, 100);
    try testing.expectEqual(@as(usize, 100), tensor.data_size);
}

test "Tensor.deinit frees owned memory" {
    const shape = [_]i64{ 2, 2 };
    const tensor = try Tensor.init(testing.allocator, &shape, .f32, Device.cpu());
    // Just verify it doesn't crash
    tensor.deinit(testing.allocator);
}

// Data accessor tests
test "Tensor.data returns byte slice" {
    const shape = [_]i64{ 2, 2 };
    const tensor = try Tensor.init(testing.allocator, &shape, .f32, Device.cpu());
    defer tensor.deinit(testing.allocator);

    const data_slice = tensor.data();
    try testing.expectEqual(@as(usize, 16), data_slice.len); // 4 floats * 4 bytes
}

test "Tensor.shapeAsUsize converts to usize array" {
    const shape = [_]i64{ 2, 3, 4, 5 };
    const tensor = try Tensor.init(testing.allocator, &shape, .f32, Device.cpu());
    defer tensor.deinit(testing.allocator);

    const usize_shape = tensor.shapeAsUsize();
    try testing.expectEqual(@as(usize, 2), usize_shape[0]);
    try testing.expectEqual(@as(usize, 3), usize_shape[1]);
    try testing.expectEqual(@as(usize, 4), usize_shape[2]);
    try testing.expectEqual(@as(usize, 5), usize_shape[3]);
}

test "Tensor.asSlice returns typed slice" {
    const shape = [_]i64{4};
    const tensor = try Tensor.init(testing.allocator, &shape, .f32, Device.cpu());
    defer tensor.deinit(testing.allocator);

    const slice = tensor.asSlice(f32);
    try testing.expectEqual(@as(usize, 4), slice.len);

    // Write and read back
    slice[0] = 1.5;
    slice[1] = 2.5;
    slice[2] = 3.5;
    slice[3] = 4.5;

    try testing.expectEqual(@as(f32, 1.5), slice[0]);
    try testing.expectEqual(@as(f32, 2.5), slice[1]);
    try testing.expectEqual(@as(f32, 3.5), slice[2]);
    try testing.expectEqual(@as(f32, 4.5), slice[3]);
}

test "Tensor.asSliceMut returns mutable typed slice" {
    const shape = [_]i64{3};
    var tensor = try Tensor.init(testing.allocator, &shape, .i32, Device.cpu());
    defer tensor.deinit(testing.allocator);

    const slice = tensor.asSliceMut(i32);
    try testing.expectEqual(@as(usize, 3), slice.len);

    slice[0] = 10;
    slice[1] = 20;
    slice[2] = 30;

    const const_slice = tensor.asSlice(i32);
    try testing.expectEqual(@as(i32, 10), const_slice[0]);
    try testing.expectEqual(@as(i32, 20), const_slice[1]);
    try testing.expectEqual(@as(i32, 30), const_slice[2]);
}

test "Tensor.asSliceUnaligned handles unaligned access" {
    var data = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const shape = [_]usize{8};

    const tensor = Tensor.view(&data, &shape, .u8, null);
    const slice = tensor.asSliceUnaligned(u8);
    try testing.expectEqual(@as(usize, 8), slice.len);
    try testing.expectEqual(@as(u8, 1), slice[0]);
    try testing.expectEqual(@as(u8, 8), slice[7]);
}

test "Tensor.rowPtr returns pointer to row" {
    const shape = [_]i64{ 3, 4 };
    const tensor = try Tensor.init(testing.allocator, &shape, .f32, Device.cpu());
    defer tensor.deinit(testing.allocator);

    const slice = tensor.asSliceMut(f32);
    for (slice, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }

    const row1 = tensor.rowPtr(f32, 1);
    try testing.expectEqual(@as(f32, 4.0), row1[0]);
    try testing.expectEqual(@as(f32, 5.0), row1[1]);
    try testing.expectEqual(@as(f32, 6.0), row1[2]);
    try testing.expectEqual(@as(f32, 7.0), row1[3]);

    const row2 = tensor.rowPtr(f32, 2);
    try testing.expectEqual(@as(f32, 8.0), row2[0]);
    try testing.expectEqual(@as(f32, 9.0), row2[1]);
}

// Contiguity tests
test "Tensor.isContiguous returns true for contiguous tensor" {
    const shape = [_]i64{ 2, 3, 4 };
    const tensor = try Tensor.init(testing.allocator, &shape, .f32, Device.cpu());
    defer tensor.deinit(testing.allocator);

    try testing.expect(tensor.isContiguous());
}

test "Tensor.isContiguous handles empty tensor" {
    var tensor = Tensor.view(&[_]u8{}, &[_]usize{}, .f32, 0);
    try testing.expect(tensor.isContiguous());
}

test "Tensor.assertContiguous succeeds on contiguous tensor" {
    const shape = [_]i64{ 2, 2 };
    const tensor = try Tensor.init(testing.allocator, &shape, .f32, Device.cpu());
    defer tensor.deinit(testing.allocator);

    // Should not panic
    tensor.assertContiguous();
}

// DType conversion tests
test "Tensor.simpleDType passes through standard types" {
    const shape = [_]i64{4};

    var tensor = try Tensor.init(testing.allocator, &shape, .f32, Device.cpu());
    try testing.expectEqual(DType.f32, tensor.simpleDType());
    tensor.deinit(testing.allocator);

    tensor = try Tensor.init(testing.allocator, &shape, .f16, Device.cpu());
    try testing.expectEqual(DType.f16, tensor.simpleDType());
    tensor.deinit(testing.allocator);

    tensor = try Tensor.init(testing.allocator, &shape, .i32, Device.cpu());
    try testing.expectEqual(DType.i32, tensor.simpleDType());
    tensor.deinit(testing.allocator);

    tensor = try Tensor.init(testing.allocator, &shape, .u8, Device.cpu());
    try testing.expectEqual(DType.u8, tensor.simpleDType());
    tensor.deinit(testing.allocator);
}

test "Tensor.simpleDType converts quantized types to u8" {
    var data = [_]u8{0} ** 32;
    const shape = [_]usize{8};

    var tensor = Tensor.view(&data, &shape, .mxfp4, 32);
    try testing.expectEqual(DType.u8, tensor.simpleDType());

    tensor = Tensor.view(&data, &shape, .grouped_affine_u4, 32);
    try testing.expectEqual(DType.u8, tensor.simpleDType());
}

// Device check tests
test "Tensor.isCPU returns true for CPU tensors" {
    const shape = [_]i64{4};
    const tensor = try Tensor.init(testing.allocator, &shape, .f32, Device.cpu());
    defer tensor.deinit(testing.allocator);

    try testing.expect(tensor.isCPU());
}

test "Tensor.isCPU returns false for Metal tensors" {
    var data = [_]f32{ 1, 2, 3, 4 };
    const bytes = std.mem.sliceAsBytes(data[0..]);
    const shape = [_]usize{4};

    var tensor = Tensor.view(@constCast(bytes.ptr), &shape, .f32, null);
    tensor.device = Device.metal(0);

    try testing.expect(!tensor.isCPU());
}

test "Tensor.isCPU returns false for CUDA tensors" {
    var data = [_]f32{ 1, 2, 3, 4 };
    const bytes = std.mem.sliceAsBytes(data[0..]);
    const shape = [_]usize{4};

    var tensor = Tensor.view(@constCast(bytes.ptr), &shape, .f32, null);
    tensor.device = Device.cuda(0);

    try testing.expect(!tensor.isCPU());
}

// View constructor tests
test "Tensor.view2D creates 2D f32 view" {
    var data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const bytes = std.mem.sliceAsBytes(data[0..]);

    const tensor = Tensor.view2D(@constCast(bytes), 2, 3);

    try testing.expectEqual(@as(i32, 2), tensor.n_dims);
    try testing.expectEqual(@as(i64, 2), tensor.shape[0]);
    try testing.expectEqual(@as(i64, 3), tensor.shape[1]);
    try testing.expectEqual(@as(usize, 6), tensor.numel);
    try testing.expectEqual(DType.f32, tensor.dtype);
    try testing.expectEqual(@as(i64, 3), tensor.strides[0]);
    try testing.expectEqual(@as(i64, 1), tensor.strides[1]);
    try testing.expect(!tensor.owns_data);
    try testing.expect(tensor.isCPU());
}

test "Tensor.view3D creates 3D f32 view" {
    var data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const bytes = std.mem.sliceAsBytes(data[0..]);

    const tensor = Tensor.view3D(@constCast(bytes), 2, 3);

    try testing.expectEqual(@as(i32, 3), tensor.n_dims);
    try testing.expectEqual(@as(i64, 1), tensor.shape[0]);
    try testing.expectEqual(@as(i64, 2), tensor.shape[1]);
    try testing.expectEqual(@as(i64, 3), tensor.shape[2]);
    try testing.expectEqual(@as(usize, 6), tensor.numel);
    try testing.expectEqual(DType.f32, tensor.dtype);
    try testing.expectEqual(@as(i64, 6), tensor.strides[0]);
    try testing.expectEqual(@as(i64, 3), tensor.strides[1]);
    try testing.expectEqual(@as(i64, 1), tensor.strides[2]);
}

test "Tensor.view2DSlice creates 2D view from f32 slice" {
    var data = [_]f32{ 1, 2, 3, 4 };
    const tensor = Tensor.view2DSlice(&data, 2, 2);

    try testing.expectEqual(@as(i32, 2), tensor.n_dims);
    try testing.expectEqual(@as(i64, 2), tensor.shape[0]);
    try testing.expectEqual(@as(i64, 2), tensor.shape[1]);
    try testing.expectEqual(DType.f32, tensor.dtype);

    const slice = tensor.asSlice(f32);
    try testing.expectEqual(@as(f32, 1), slice[0]);
    try testing.expectEqual(@as(f32, 4), slice[3]);
}

test "Tensor.view3DSlice creates 3D view from f32 slice" {
    var data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const tensor = Tensor.view3DSlice(&data, 2, 3);

    try testing.expectEqual(@as(i32, 3), tensor.n_dims);
    try testing.expectEqual(@as(i64, 1), tensor.shape[0]);
    try testing.expectEqual(@as(i64, 2), tensor.shape[1]);
    try testing.expectEqual(@as(i64, 3), tensor.shape[2]);
    try testing.expectEqual(DType.f32, tensor.dtype);

    const slice = tensor.asSlice(f32);
    try testing.expectEqual(@as(f32, 1), slice[0]);
    try testing.expectEqual(@as(f32, 6), slice[5]);
}

// DLPack export test
test "Tensor.toDLPack creates DLManagedTensor" {
    const shape = [_]i64{ 2, 3 };
    var tensor = try Tensor.init(testing.allocator, &shape, .f32, Device.cpu());
    defer tensor.deinit(testing.allocator);

    const managed = try tensor.toDLPack(testing.allocator);
    defer testing.allocator.destroy(managed);

    try testing.expectEqual(@as(i32, 2), managed.dl_tensor.ndim);
    try testing.expectEqual(@as(i64, 2), managed.dl_tensor.shape[0]);
    try testing.expectEqual(@as(i64, 3), managed.dl_tensor.shape[1]);
    try testing.expectEqual(DLDeviceType.kDLCPU, managed.dl_tensor.device.device_type);
    try testing.expect(managed.deleter != null);
}

// OwnedTensor tests
test "OwnedTensor.init creates aligned tensor" {
    const shape = [_]usize{ 2, 3 };
    var owned = try OwnedTensor.init(testing.allocator, .f32, &shape);
    defer owned.deinit();

    try testing.expectEqual(@as(i32, 2), owned.n_dims);
    try testing.expectEqual(@as(usize, 2), owned.shape[0]);
    try testing.expectEqual(@as(usize, 3), owned.shape[1]);
    try testing.expectEqual(DType.f32, owned.dtype);
    try testing.expectEqual(@as(usize, 24), owned.data_size); // 6 * 4 bytes
}

test "OwnedTensor.numElements calculates correct count" {
    const shape = [_]usize{ 2, 3, 4 };
    var owned = try OwnedTensor.init(testing.allocator, .f32, &shape);
    defer owned.deinit();

    try testing.expectEqual(@as(usize, 24), owned.numElements());
}

test "OwnedTensor.asSlice returns typed slice" {
    const shape = [_]usize{4};
    var owned = try OwnedTensor.init(testing.allocator, .f32, &shape);
    defer owned.deinit();

    const slice = owned.asSlice(f32);
    try testing.expectEqual(@as(usize, 4), slice.len);

    slice[0] = 1.0;
    slice[1] = 2.0;
    try testing.expectEqual(@as(f32, 1.0), slice[0]);
    try testing.expectEqual(@as(f32, 2.0), slice[1]);
}

test "OwnedTensor.toTensor creates Tensor view" {
    const shape = [_]usize{ 2, 3 };
    var owned = try OwnedTensor.init(testing.allocator, .f32, &shape);
    defer owned.deinit();

    const tensor = owned.toTensor();

    try testing.expectEqual(@as(i32, 2), tensor.n_dims);
    try testing.expectEqual(@as(i64, 2), tensor.shape[0]);
    try testing.expectEqual(@as(i64, 3), tensor.shape[1]);
    try testing.expectEqual(@as(usize, 6), tensor.numel);
    try testing.expectEqual(DType.f32, tensor.dtype);
    try testing.expect(!tensor.owns_data);
    try testing.expect(tensor.isContiguous());
}

test "OwnedTensor.view is alias for toTensor" {
    const shape = [_]usize{ 2, 2 };
    var owned = try OwnedTensor.init(testing.allocator, .f32, &shape);
    defer owned.deinit();

    const tensor1 = owned.toTensor();
    const tensor2 = owned.view();

    try testing.expectEqual(tensor1.n_dims, tensor2.n_dims);
    try testing.expectEqual(tensor1.numel, tensor2.numel);
    try testing.expectEqual(tensor1.dtype, tensor2.dtype);
}

test "OwnedTensor.isContiguous always returns true" {
    const shape = [_]usize{ 3, 4 };
    var owned = try OwnedTensor.init(testing.allocator, .f32, &shape);
    defer owned.deinit();

    const tensor = owned.toTensor();
    try testing.expect(tensor.isContiguous());
}

// Edge case tests
test "init empty dimensions zeroed" {
    const shape = [_]i64{ 2, 3 };
    const tensor = try Tensor.init(testing.allocator, &shape, .f32, Device.cpu());
    defer tensor.deinit(testing.allocator);

    // Dimensions 2-7 should be zero
    for (2..MAX_NDIM) |i| {
        try testing.expectEqual(@as(i64, 0), tensor.shape[i]);
        try testing.expectEqual(@as(i64, 0), tensor.strides[i]);
    }
}

test "init stride calculation" {
    // 1D: [5] -> strides [1]
    const shape_1d = [_]i64{5};
    var tensor = try Tensor.init(testing.allocator, &shape_1d, .f32, Device.cpu());
    try testing.expectEqual(@as(i64, 1), tensor.strides[0]);
    tensor.deinit(testing.allocator);

    // 2D: [3, 4] -> strides [4, 1]
    const shape_2d = [_]i64{ 3, 4 };
    tensor = try Tensor.init(testing.allocator, &shape_2d, .f32, Device.cpu());
    try testing.expectEqual(@as(i64, 4), tensor.strides[0]);
    try testing.expectEqual(@as(i64, 1), tensor.strides[1]);
    tensor.deinit(testing.allocator);

    // 3D: [2, 3, 4] -> strides [12, 4, 1]
    const shape_3d = [_]i64{ 2, 3, 4 };
    tensor = try Tensor.init(testing.allocator, &shape_3d, .f32, Device.cpu());
    try testing.expectEqual(@as(i64, 12), tensor.strides[0]);
    try testing.expectEqual(@as(i64, 4), tensor.strides[1]);
    try testing.expectEqual(@as(i64, 1), tensor.strides[2]);
    tensor.deinit(testing.allocator);

    // 4D: [2, 3, 4, 5] -> strides [60, 20, 5, 1]
    const shape_4d = [_]i64{ 2, 3, 4, 5 };
    tensor = try Tensor.init(testing.allocator, &shape_4d, .f32, Device.cpu());
    try testing.expectEqual(@as(i64, 60), tensor.strides[0]);
    try testing.expectEqual(@as(i64, 20), tensor.strides[1]);
    try testing.expectEqual(@as(i64, 5), tensor.strides[2]);
    try testing.expectEqual(@as(i64, 1), tensor.strides[3]);
    tensor.deinit(testing.allocator);
}

test "init different dtypes" {
    const shape = [_]i64{4};

    // f32: 4 elements * 4 bytes = 16 bytes
    var tensor = try Tensor.init(testing.allocator, &shape, .f32, Device.cpu());
    try testing.expectEqual(@as(usize, 16), tensor.data_size);
    tensor.deinit(testing.allocator);

    // f16: 4 elements * 2 bytes = 8 bytes
    tensor = try Tensor.init(testing.allocator, &shape, .f16, Device.cpu());
    try testing.expectEqual(@as(usize, 8), tensor.data_size);
    tensor.deinit(testing.allocator);

    // u8: 4 elements * 1 byte = 4 bytes
    tensor = try Tensor.init(testing.allocator, &shape, .u8, Device.cpu());
    try testing.expectEqual(@as(usize, 4), tensor.data_size);
    tensor.deinit(testing.allocator);

    // i64: 4 elements * 8 bytes = 32 bytes
    tensor = try Tensor.init(testing.allocator, &shape, .i64, Device.cpu());
    try testing.expectEqual(@as(usize, 32), tensor.data_size);
    tensor.deinit(testing.allocator);
}

test "Tensor.data returns empty slice for null pointer" {
    var tensor: Tensor = undefined;
    tensor.data_ptr = null;
    tensor.data_size = 0;

    const slice = tensor.data();
    try testing.expectEqual(@as(usize, 0), slice.len);
}

test "Tensor.asSlice returns empty slice for null pointer" {
    var tensor: Tensor = undefined;
    tensor.data_ptr = null;
    tensor.numel = 0;

    const slice = tensor.asSlice(f32);
    try testing.expectEqual(@as(usize, 0), slice.len);
}

test "module-level isContiguous function" {
    const shape = [_]i64{ 2, 3 };
    const tensor = try Tensor.init(testing.allocator, &shape, .f32, Device.cpu());
    defer tensor.deinit(testing.allocator);

    try testing.expect(isContiguous(tensor));
}
