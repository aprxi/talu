//! CUDA driver-backed device and buffer lifecycle.
//!
//! This module intentionally uses dynamic symbol loading (`libcuda`) so the
//! binary can run on non-CUDA hosts without link-time CUDA dependencies.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");

const cuda_success: c_int = 0;
const cuda_error_out_of_memory: c_int = 2;
const cuda_error_invalid_value: c_int = 1;
const cuda_error_deinitialized: c_int = 4;
const cuda_error_not_initialized: c_int = 3;
const cuda_error_invalid_context: c_int = 201;
const cuda_error_context_is_destroyed: c_int = 709;

const CuInitFn = *const fn (u32) callconv(.c) c_int;
const CuDeviceGetCountFn = *const fn (*c_int) callconv(.c) c_int;
const CuDeviceGetFn = *const fn (*c_int, c_int) callconv(.c) c_int;
const CuCtxCreateFn = *const fn (*?*anyopaque, u32, c_int) callconv(.c) c_int;
const CuCtxDestroyFn = *const fn (?*anyopaque) callconv(.c) c_int;
const CuCtxSetCurrentFn = *const fn (?*anyopaque) callconv(.c) c_int;
const CuCtxSynchronizeFn = *const fn () callconv(.c) c_int;
const CuMemAllocFn = *const fn (*u64, usize) callconv(.c) c_int;
const CuMemFreeFn = *const fn (u64) callconv(.c) c_int;
const CuMemcpyHtoDFn = *const fn (u64, *const anyopaque, usize) callconv(.c) c_int;
const CuMemcpyDtoHFn = *const fn (*anyopaque, u64, usize) callconv(.c) c_int;
const CuDeviceGetNameFn = *const fn ([*]u8, c_int, c_int) callconv(.c) c_int;
const CuDeviceTotalMemFn = *const fn (*usize, c_int) callconv(.c) c_int;
const CuDeviceGetAttributeFn = *const fn (*c_int, c_int, c_int) callconv(.c) c_int;
const CuModuleLoadDataFn = *const fn (*?*anyopaque, *const anyopaque) callconv(.c) c_int;
const CuModuleGetFunctionFn = *const fn (*?*anyopaque, ?*anyopaque, [*:0]const u8) callconv(.c) c_int;
const CuModuleUnloadFn = *const fn (?*anyopaque) callconv(.c) c_int;
const CuLaunchKernelFn = *const fn (
    ?*anyopaque,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    ?*anyopaque,
    ?*anyopaque,
    ?*anyopaque,
) callconv(.c) c_int;

pub const ModuleHandle = *anyopaque;
pub const FunctionHandle = *anyopaque;

pub const Probe = enum {
    disabled,
    available,
    driver_not_found,
    symbols_missing,
    init_failed,
    no_devices,
};

const DriverApi = struct {
    cu_init: CuInitFn,
    cu_device_get_count: CuDeviceGetCountFn,
    cu_device_get: CuDeviceGetFn,
    cu_ctx_create: CuCtxCreateFn,
    cu_ctx_destroy: CuCtxDestroyFn,
    cu_ctx_set_current: CuCtxSetCurrentFn,
    cu_ctx_synchronize: CuCtxSynchronizeFn,
    cu_mem_alloc: CuMemAllocFn,
    cu_mem_free: CuMemFreeFn,
    cu_memcpy_htod: CuMemcpyHtoDFn,
    cu_memcpy_dtoh: CuMemcpyDtoHFn,
    cu_device_get_name: CuDeviceGetNameFn,
    cu_device_total_mem: ?CuDeviceTotalMemFn,
    cu_device_get_attribute: ?CuDeviceGetAttributeFn,
    cu_module_load_data: ?CuModuleLoadDataFn,
    cu_module_get_function: ?CuModuleGetFunctionFn,
    cu_module_unload: ?CuModuleUnloadFn,
    cu_launch_kernel: ?CuLaunchKernelFn,
};

const cu_device_attribute_compute_capability_major: c_int = 75;
const cu_device_attribute_compute_capability_minor: c_int = 76;

pub const ComputeCapability = struct {
    major: u32,
    minor: u32,
};

pub fn isRuntimeSupported() bool {
    return build_options.enable_cuda and (builtin.os.tag == .linux or builtin.os.tag == .windows);
}

pub fn probeRuntime() Probe {
    if (!isRuntimeSupported()) return .disabled;

    var lib = openDriverLibrary() catch return .driver_not_found;
    defer lib.close();

    const cu_init = lookupRequired(CuInitFn, &lib, "cuInit") catch return .symbols_missing;
    const cu_device_get_count = lookupRequired(CuDeviceGetCountFn, &lib, "cuDeviceGetCount") catch return .symbols_missing;

    if (cu_init(0) != cuda_success) return .init_failed;

    var device_count: c_int = 0;
    if (cu_device_get_count(&device_count) != cuda_success) return .init_failed;
    if (device_count <= 0) return .no_devices;
    return .available;
}

pub const Device = struct {
    lib: std.DynLib,
    api: DriverApi,
    context: ?*anyopaque,
    device_index: c_int,
    name_buffer: [128]u8,

    pub fn init() !Device {
        if (!isRuntimeSupported()) return error.CudaNotEnabled;

        var lib = try openDriverLibrary();
        errdefer lib.close();

        const api = try loadDriverApi(&lib);
        if (api.cu_init(0) != cuda_success) return error.CudaInitFailed;

        var device_count: c_int = 0;
        if (api.cu_device_get_count(&device_count) != cuda_success) return error.CudaInitFailed;
        if (device_count <= 0) return error.CudaNoDevices;

        var device_index: c_int = 0;
        if (api.cu_device_get(&device_index, 0) != cuda_success) return error.CudaInitFailed;

        var context: ?*anyopaque = null;
        if (api.cu_ctx_create(&context, 0, device_index) != cuda_success or context == null) {
            return error.CudaContextCreateFailed;
        }
        errdefer _ = api.cu_ctx_destroy(context);

        var name_buffer = [_]u8{0} ** 128;
        const name_status = api.cu_device_get_name(name_buffer[0..].ptr, @intCast(name_buffer.len), device_index);
        if (name_status != cuda_success or name_buffer[0] == 0) {
            const fallback_name = "cuda:0";
            @memcpy(name_buffer[0..fallback_name.len], fallback_name);
            name_buffer[fallback_name.len] = 0;
        }

        return .{
            .lib = lib,
            .api = api,
            .context = context,
            .device_index = device_index,
            .name_buffer = name_buffer,
        };
    }

    pub fn deinit(self: *Device) void {
        if (self.context) |ctx| {
            _ = self.api.cu_ctx_destroy(ctx);
            self.context = null;
        }
        self.lib.close();
    }

    pub fn name(self: *const Device) []const u8 {
        const end = std.mem.indexOfScalar(u8, self.name_buffer[0..], 0) orelse self.name_buffer.len;
        return self.name_buffer[0..end];
    }

    pub fn synchronize(self: *Device) !void {
        try self.makeCurrent();
        if (self.api.cu_ctx_synchronize() != cuda_success) return error.CudaSynchronizeFailed;
    }

    pub fn totalMemory(self: *Device) !usize {
        const cu_device_total_mem = self.api.cu_device_total_mem orelse return error.CudaQueryUnavailable;
        var total: usize = 0;
        if (cu_device_total_mem(&total, self.device_index) != cuda_success) return error.CudaQueryFailed;
        return total;
    }

    pub fn computeCapability(self: *Device) !ComputeCapability {
        const cu_device_get_attribute = self.api.cu_device_get_attribute orelse return error.CudaQueryUnavailable;

        var major_raw: c_int = 0;
        if (cu_device_get_attribute(&major_raw, cu_device_attribute_compute_capability_major, self.device_index) != cuda_success) {
            return error.CudaQueryFailed;
        }
        var minor_raw: c_int = 0;
        if (cu_device_get_attribute(&minor_raw, cu_device_attribute_compute_capability_minor, self.device_index) != cuda_success) {
            return error.CudaQueryFailed;
        }
        if (major_raw < 0 or minor_raw < 0) return error.CudaQueryFailed;

        return .{
            .major = @intCast(major_raw),
            .minor = @intCast(minor_raw),
        };
    }

    pub fn allocBuffer(self: *Device, buffer_size: usize) !Buffer {
        try self.makeCurrent();

        var pointer: u64 = 0;
        const rc = self.api.cu_mem_alloc(&pointer, buffer_size);
        if (rc == cuda_error_out_of_memory) return error.OutOfMemory;
        if (rc == cuda_error_invalid_value) return error.InvalidArgument;
        if (rc == cuda_error_deinitialized or rc == cuda_error_not_initialized) return error.CudaInitFailed;
        if (rc == cuda_error_invalid_context or rc == cuda_error_context_is_destroyed) return error.CudaContextLost;
        if (rc != cuda_success) return error.CudaAllocFailed;
        return .{ .pointer = pointer, .size = buffer_size };
    }

    pub fn makeCurrent(self: *Device) !void {
        if (self.context == null) return error.CudaContextLost;
        if (self.api.cu_ctx_set_current(self.context) != cuda_success) return error.CudaContextLost;
    }

    pub fn supportsModuleLaunch(self: *const Device) bool {
        return self.api.cu_module_load_data != null and
            self.api.cu_module_get_function != null and
            self.api.cu_module_unload != null and
            self.api.cu_launch_kernel != null;
    }

    pub fn moduleLoadData(self: *Device, image: *const anyopaque) !ModuleHandle {
        const cu_module_load_data = self.api.cu_module_load_data orelse return error.CudaModuleApiUnavailable;
        try self.makeCurrent();

        var module_handle: ?*anyopaque = null;
        if (cu_module_load_data(&module_handle, image) != cuda_success or module_handle == null) {
            return error.CudaModuleLoadFailed;
        }
        return module_handle.?;
    }

    pub fn moduleGetFunction(self: *Device, module: ModuleHandle, symbol: [:0]const u8) !FunctionHandle {
        const cu_module_get_function = self.api.cu_module_get_function orelse return error.CudaModuleApiUnavailable;
        try self.makeCurrent();

        var function_handle: ?*anyopaque = null;
        if (cu_module_get_function(&function_handle, module, symbol.ptr) != cuda_success or function_handle == null) {
            return error.CudaFunctionLookupFailed;
        }
        return function_handle.?;
    }

    pub fn moduleUnload(self: *Device, module: ModuleHandle) void {
        const cu_module_unload = self.api.cu_module_unload orelse return;
        self.makeCurrent() catch return;
        _ = cu_module_unload(module);
    }

    pub fn launchKernel(
        self: *Device,
        function: FunctionHandle,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        block_x: u32,
        block_y: u32,
        block_z: u32,
        shared_mem_bytes: u32,
        kernel_params: ?[*]const ?*anyopaque,
    ) !void {
        const cu_launch_kernel = self.api.cu_launch_kernel orelse return error.CudaModuleApiUnavailable;
        try self.makeCurrent();

        const params_ptr: ?*anyopaque = if (kernel_params) |params|
            @ptrCast(@constCast(params))
        else
            null;

        if (cu_launch_kernel(
            function,
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z,
            shared_mem_bytes,
            null, // stream
            params_ptr,
            null, // extra
        ) != cuda_success) {
            return error.CudaKernelLaunchFailed;
        }
    }
};

pub const Buffer = struct {
    pointer: u64,
    size: usize,

    pub fn deinit(self: *Buffer, device: *Device) void {
        if (self.pointer == 0) return;
        device.makeCurrent() catch return;
        _ = device.api.cu_mem_free(self.pointer);
        self.pointer = 0;
        self.size = 0;
    }

    pub fn upload(self: *const Buffer, device: *Device, data: []const u8) !void {
        if (data.len > self.size) return error.InvalidArgument;
        if (data.len == 0) return;

        try device.makeCurrent();
        if (device.api.cu_memcpy_htod(self.pointer, @ptrCast(data.ptr), data.len) != cuda_success) {
            return error.CudaCopyFailed;
        }
    }

    pub fn download(self: *const Buffer, device: *Device, data: []u8) !void {
        if (data.len > self.size) return error.InvalidArgument;
        if (data.len == 0) return;

        try device.makeCurrent();
        if (device.api.cu_memcpy_dtoh(@ptrCast(data.ptr), self.pointer, data.len) != cuda_success) {
            return error.CudaCopyFailed;
        }
    }
};

fn openDriverLibrary() !std.DynLib {
    const names: []const []const u8 = switch (builtin.os.tag) {
        .linux => &.{ "libcuda.so.1", "libcuda.so" },
        .windows => &.{"nvcuda.dll"},
        else => &.{},
    };

    for (names) |name| {
        if (std.DynLib.open(name)) |lib| return lib else |_| {}
    }
    return error.CudaDriverNotFound;
}

fn lookupRequired(comptime T: type, lib: *std.DynLib, symbol: [:0]const u8) !T {
    return lib.lookup(T, symbol) orelse error.CudaSymbolMissing;
}

fn lookupOptional(comptime T: type, lib: *std.DynLib, symbol: [:0]const u8) ?T {
    return lib.lookup(T, symbol);
}

fn lookupRequiredAny(comptime T: type, lib: *std.DynLib, symbols: []const [:0]const u8) !T {
    for (symbols) |symbol| {
        if (lib.lookup(T, symbol)) |fn_ptr| return fn_ptr;
    }
    return error.CudaSymbolMissing;
}

fn lookupOptionalAny(comptime T: type, lib: *std.DynLib, symbols: []const [:0]const u8) ?T {
    for (symbols) |symbol| {
        if (lib.lookup(T, symbol)) |fn_ptr| return fn_ptr;
    }
    return null;
}

fn loadDriverApi(lib: *std.DynLib) !DriverApi {
    return .{
        .cu_init = try lookupRequired(CuInitFn, lib, "cuInit"),
        .cu_device_get_count = try lookupRequired(CuDeviceGetCountFn, lib, "cuDeviceGetCount"),
        .cu_device_get = try lookupRequired(CuDeviceGetFn, lib, "cuDeviceGet"),
        .cu_ctx_create = try lookupRequiredAny(CuCtxCreateFn, lib, &.{ "cuCtxCreate_v2", "cuCtxCreate" }),
        .cu_ctx_destroy = try lookupRequiredAny(CuCtxDestroyFn, lib, &.{ "cuCtxDestroy_v2", "cuCtxDestroy" }),
        .cu_ctx_set_current = try lookupRequired(CuCtxSetCurrentFn, lib, "cuCtxSetCurrent"),
        .cu_ctx_synchronize = try lookupRequired(CuCtxSynchronizeFn, lib, "cuCtxSynchronize"),
        .cu_mem_alloc = try lookupRequiredAny(CuMemAllocFn, lib, &.{ "cuMemAlloc_v2", "cuMemAlloc" }),
        .cu_mem_free = try lookupRequiredAny(CuMemFreeFn, lib, &.{ "cuMemFree_v2", "cuMemFree" }),
        .cu_memcpy_htod = try lookupRequiredAny(CuMemcpyHtoDFn, lib, &.{ "cuMemcpyHtoD_v2", "cuMemcpyHtoD" }),
        .cu_memcpy_dtoh = try lookupRequiredAny(CuMemcpyDtoHFn, lib, &.{ "cuMemcpyDtoH_v2", "cuMemcpyDtoH" }),
        .cu_device_get_name = try lookupRequired(CuDeviceGetNameFn, lib, "cuDeviceGetName"),
        .cu_device_total_mem = lookupOptionalAny(CuDeviceTotalMemFn, lib, &.{ "cuDeviceTotalMem_v2", "cuDeviceTotalMem" }),
        .cu_device_get_attribute = lookupOptional(CuDeviceGetAttributeFn, lib, "cuDeviceGetAttribute"),
        .cu_module_load_data = lookupOptional(CuModuleLoadDataFn, lib, "cuModuleLoadData"),
        .cu_module_get_function = lookupOptional(CuModuleGetFunctionFn, lib, "cuModuleGetFunction"),
        .cu_module_unload = lookupOptional(CuModuleUnloadFn, lib, "cuModuleUnload"),
        .cu_launch_kernel = lookupOptional(CuLaunchKernelFn, lib, "cuLaunchKernel"),
    };
}

test "probeRuntime returns disabled when isRuntimeSupported is false" {
    if (isRuntimeSupported()) return;
    try std.testing.expectEqual(Probe.disabled, probeRuntime());
}

test "probeRuntime returns stable value across repeated calls" {
    if (!isRuntimeSupported()) return;
    const first = probeRuntime();
    const second = probeRuntime();
    try std.testing.expectEqual(first, second);
}

test "Device.init Device.deinit Device.name work when probeRuntime is available" {
    if (probeRuntime() != .available) return error.SkipZigTest;

    var device = try Device.init();
    defer device.deinit();

    try std.testing.expect(device.name().len > 0);
}

test "Device.allocBuffer Buffer.upload Buffer.download Buffer.deinit roundtrip" {
    if (probeRuntime() != .available) return error.SkipZigTest;

    var device = try Device.init();
    defer device.deinit();

    var buffer = try device.allocBuffer(32);
    defer buffer.deinit(&device);

    const input = [_]u8{
        0,  1,  2,  3,  4,  5,  6,  7,
        8,  9,  10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31,
    };
    try buffer.upload(&device, &input);
    try device.synchronize();

    var output: [32]u8 = undefined;
    try buffer.download(&device, &output);
    try std.testing.expectEqualSlices(u8, &input, &output);
}

test "Device.totalMemory reports non-zero bytes when available" {
    if (probeRuntime() != .available) return error.SkipZigTest;

    var device = try Device.init();
    defer device.deinit();

    const bytes = device.totalMemory() catch |err| {
        try std.testing.expect(err == error.CudaQueryUnavailable);
        return;
    };
    try std.testing.expect(bytes > 0);
}

test "Device.supportsModuleLaunch reports availability after Device.init" {
    if (probeRuntime() != .available) return error.SkipZigTest;

    var device = try Device.init();
    defer device.deinit();

    _ = device.supportsModuleLaunch();
}

test "Device.computeCapability returns non-zero major when available" {
    if (probeRuntime() != .available) return error.SkipZigTest;

    var device = try Device.init();
    defer device.deinit();

    const capability = device.computeCapability() catch |err| {
        try std.testing.expect(err == error.CudaQueryUnavailable);
        return;
    };
    try std.testing.expect(capability.major > 0);
}
