//! CUDA driver-backed device and buffer lifecycle.
//!
//! This module intentionally uses dynamic symbol loading (`libcuda`) so the
//! binary can run on non-CUDA hosts without link-time CUDA dependencies.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const log = @import("../../log.zig");

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
const CuMemGetInfoFn = *const fn (*usize, *usize) callconv(.c) c_int;
const CuMemcpyHtoDFn = *const fn (u64, *const anyopaque, usize) callconv(.c) c_int;
const CuMemcpyDtoHFn = *const fn (*anyopaque, u64, usize) callconv(.c) c_int;
const CuMemcpyDtoDFn = *const fn (u64, u64, usize) callconv(.c) c_int;
const CuDeviceGetNameFn = *const fn ([*]u8, c_int, c_int) callconv(.c) c_int;
const CuDeviceTotalMemFn = *const fn (*usize, c_int) callconv(.c) c_int;
const CuDeviceGetAttributeFn = *const fn (*c_int, c_int, c_int) callconv(.c) c_int;
const CuModuleLoadDataFn = *const fn (*?*anyopaque, *const anyopaque) callconv(.c) c_int;
const CuModuleGetFunctionFn = *const fn (*?*anyopaque, ?*anyopaque, [*:0]const u8) callconv(.c) c_int;
const CuModuleUnloadFn = *const fn (?*anyopaque) callconv(.c) c_int;
const CuStreamCreateFn = *const fn (*?*anyopaque, u32) callconv(.c) c_int;
const CuStreamDestroyFn = *const fn (?*anyopaque) callconv(.c) c_int;
const CuStreamSynchronizeFn = *const fn (?*anyopaque) callconv(.c) c_int;
const CuStreamBeginCaptureFn = *const fn (?*anyopaque, c_int) callconv(.c) c_int;
const CuStreamEndCaptureFn = *const fn (?*anyopaque, *?*anyopaque) callconv(.c) c_int;
const CuGraphInstantiateWithFlagsFn = *const fn (*?*anyopaque, ?*anyopaque, u64) callconv(.c) c_int;
const CuGraphDestroyFn = *const fn (?*anyopaque) callconv(.c) c_int;
const CuGraphExecDestroyFn = *const fn (?*anyopaque) callconv(.c) c_int;
const CuGraphExecUpdateFn = *const fn (?*anyopaque, ?*anyopaque, *?*anyopaque, *c_int) callconv(.c) c_int;
const CuGraphLaunchFn = *const fn (?*anyopaque, ?*anyopaque) callconv(.c) c_int;
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

const CuDeviceCanAccessPeerFn = *const fn (*c_int, c_int, c_int) callconv(.c) c_int;
const CuCtxEnablePeerAccessFn = *const fn (?*anyopaque, u32) callconv(.c) c_int;
const CuMemcpyPeerAsyncFn = *const fn (u64, ?*anyopaque, u64, ?*anyopaque, usize, ?*anyopaque) callconv(.c) c_int;

pub const ModuleHandle = *anyopaque;
pub const FunctionHandle = *anyopaque;
pub const StreamHandle = *anyopaque;
pub const GraphHandle = *anyopaque;
pub const GraphExecHandle = *anyopaque;

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
    cu_mem_get_info: ?CuMemGetInfoFn,
    cu_memcpy_htod: CuMemcpyHtoDFn,
    cu_memcpy_dtoh: CuMemcpyDtoHFn,
    cu_memcpy_dtod: CuMemcpyDtoDFn,
    cu_device_get_name: CuDeviceGetNameFn,
    cu_device_total_mem: ?CuDeviceTotalMemFn,
    cu_device_get_attribute: ?CuDeviceGetAttributeFn,
    cu_module_load_data: ?CuModuleLoadDataFn,
    cu_module_get_function: ?CuModuleGetFunctionFn,
    cu_module_unload: ?CuModuleUnloadFn,
    cu_stream_create: ?CuStreamCreateFn,
    cu_stream_destroy: ?CuStreamDestroyFn,
    cu_stream_synchronize: ?CuStreamSynchronizeFn,
    cu_stream_begin_capture: ?CuStreamBeginCaptureFn,
    cu_stream_end_capture: ?CuStreamEndCaptureFn,
    cu_graph_instantiate_with_flags: ?CuGraphInstantiateWithFlagsFn,
    cu_graph_destroy: ?CuGraphDestroyFn,
    cu_graph_exec_destroy: ?CuGraphExecDestroyFn,
    cu_graph_exec_update: ?CuGraphExecUpdateFn,
    cu_graph_launch: ?CuGraphLaunchFn,
    cu_launch_kernel: ?CuLaunchKernelFn,
    cu_device_can_access_peer: ?CuDeviceCanAccessPeerFn,
    cu_ctx_enable_peer_access: ?CuCtxEnablePeerAccessFn,
    cu_memcpy_peer_async: ?CuMemcpyPeerAsyncFn,
};

const cu_device_attribute_compute_capability_major: c_int = 75;
const cu_device_attribute_compute_capability_minor: c_int = 76;
const cu_stream_capture_mode_global: c_int = 0;

pub const LaunchFamily = enum(u8) {
    other = 0,
    matvec = 1,
    matvec_qkv = 2,
    matvec_gate_up_silu = 3,
    matmul = 4,
    attention = 5,
    gated_delta = 6,
    norm = 7,
    rope = 8,
    kv_write = 9,
    copy_cast = 10,
    embedding = 11,
    pointwise = 12,
};

const launch_family_names = [_][]const u8{
    "other",
    "matvec",
    "matvec_qkv",
    "matvec_gate_up_silu",
    "matmul",
    "attention",
    "gated_delta",
    "norm",
    "rope",
    "kv_write",
    "copy_cast",
    "embedding",
    "pointwise",
};
const launch_family_count = launch_family_names.len;

threadlocal var tls_current_context: ?*anyopaque = null;
var stats_launch_calls = std.atomic.Value(u64).init(0);
var stats_launch_ns = std.atomic.Value(u64).init(0);
var stats_launch_calls_prefill = std.atomic.Value(u64).init(0);
var stats_launch_ns_prefill = std.atomic.Value(u64).init(0);
var stats_launch_calls_decode = std.atomic.Value(u64).init(0);
var stats_launch_ns_decode = std.atomic.Value(u64).init(0);
var stats_make_current_calls = std.atomic.Value(u64).init(0);
var stats_make_current_fastpath_hits = std.atomic.Value(u64).init(0);
var stats_set_current_calls = std.atomic.Value(u64).init(0);
var stats_set_current_ns = std.atomic.Value(u64).init(0);
var stats_family_calls = [_]std.atomic.Value(u64){std.atomic.Value(u64).init(0)} ** launch_family_count;
var stats_family_ns = [_]std.atomic.Value(u64){std.atomic.Value(u64).init(0)} ** launch_family_count;
var stats_family_calls_prefill = [_]std.atomic.Value(u64){std.atomic.Value(u64).init(0)} ** launch_family_count;
var stats_family_ns_prefill = [_]std.atomic.Value(u64){std.atomic.Value(u64).init(0)} ** launch_family_count;
var stats_family_calls_decode = [_]std.atomic.Value(u64){std.atomic.Value(u64).init(0)} ** launch_family_count;
var stats_family_ns_decode = [_]std.atomic.Value(u64){std.atomic.Value(u64).init(0)} ** launch_family_count;

pub const ComputeCapability = struct {
    major: u32,
    minor: u32,
};

pub const LaunchPhase = enum(u8) {
    none = 0,
    prefill = 1,
    decode = 2,
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
    ordinal_index: usize,
    name_buffer: [128]u8,
    launch_stream: ?StreamHandle,
    launch_stats_enabled: bool,
    launch_phase: LaunchPhase,
    launch_family: LaunchFamily,

    pub fn init() !Device {
        return initAt(0);
    }

    /// Initialize a CUDA device+context for a specific GPU ordinal.
    /// Validates that the ordinal is within the range of available devices.
    pub fn initAt(device_ordinal: usize) !Device {
        if (!isRuntimeSupported()) return error.CudaNotEnabled;

        var lib = try openDriverLibrary();
        errdefer lib.close();

        const api = try loadDriverApi(&lib);
        if (api.cu_init(0) != cuda_success) return error.CudaInitFailed;

        var device_count: c_int = 0;
        if (api.cu_device_get_count(&device_count) != cuda_success) return error.CudaInitFailed;
        if (device_count <= 0) return error.CudaNoDevices;

        const ordinal_c: c_int = std.math.cast(c_int, device_ordinal) orelse return error.CudaInvalidDevice;
        if (ordinal_c >= device_count) return error.CudaInvalidDevice;

        var device_index: c_int = 0;
        if (api.cu_device_get(&device_index, ordinal_c) != cuda_success) return error.CudaInitFailed;

        var context: ?*anyopaque = null;
        if (api.cu_ctx_create(&context, 0, device_index) != cuda_success or context == null) {
            return error.CudaContextCreateFailed;
        }
        errdefer _ = api.cu_ctx_destroy(context);

        var name_buffer = [_]u8{0} ** 128;
        const name_status = api.cu_device_get_name(name_buffer[0..].ptr, @intCast(name_buffer.len), device_index);
        if (name_status != cuda_success or name_buffer[0] == 0) {
            var fallback_buf: [16]u8 = undefined;
            const fallback_name = std.fmt.bufPrint(&fallback_buf, "cuda:{d}", .{device_ordinal}) catch "cuda:?";
            @memcpy(name_buffer[0..fallback_name.len], fallback_name);
            name_buffer[fallback_name.len] = 0;
        }

        return .{
            .lib = lib,
            .api = api,
            .context = context,
            .device_index = device_index,
            .ordinal_index = device_ordinal,
            .name_buffer = name_buffer,
            .launch_stream = null,
            .launch_stats_enabled = launchStatsEnabledForCurrentLogLevel(),
            .launch_phase = .none,
            .launch_family = .other,
        };
    }

    pub fn ordinal(self: *const Device) usize {
        return self.ordinal_index;
    }

    /// Query the number of available CUDA devices.
    /// Requires a working CUDA driver; opens and closes its own library handle.
    pub fn deviceCount() !usize {
        if (!isRuntimeSupported()) return error.CudaNotEnabled;

        var lib = try openDriverLibrary();
        defer lib.close();

        const cu_init_fn = try lookupRequired(CuInitFn, &lib, "cuInit");
        const cu_device_get_count_fn = try lookupRequired(CuDeviceGetCountFn, &lib, "cuDeviceGetCount");

        if (cu_init_fn(0) != cuda_success) return error.CudaInitFailed;

        var count: c_int = 0;
        if (cu_device_get_count_fn(&count) != cuda_success) return error.CudaInitFailed;
        if (count < 0) return error.CudaInitFailed;
        return @intCast(count);
    }

    /// Query total memory for each visible CUDA device without creating contexts.
    /// Opens its own library handle; no CUDA contexts are created or destroyed.
    /// Caller owns the returned slice.
    pub fn deviceTotalMemories(allocator: std.mem.Allocator) ![]usize {
        if (!isRuntimeSupported()) return error.CudaNotEnabled;

        var lib = try openDriverLibrary();
        defer lib.close();

        const cu_init_fn = try lookupRequired(CuInitFn, &lib, "cuInit");
        const cu_device_get_count_fn = try lookupRequired(CuDeviceGetCountFn, &lib, "cuDeviceGetCount");
        const cu_device_get_fn = try lookupRequired(CuDeviceGetFn, &lib, "cuDeviceGet");
        const cu_device_total_mem_fn = lookupOptionalAny(CuDeviceTotalMemFn, &lib, &.{ "cuDeviceTotalMem_v2", "cuDeviceTotalMem" }) orelse
            return error.CudaQueryUnavailable;

        if (cu_init_fn(0) != cuda_success) return error.CudaInitFailed;

        var count: c_int = 0;
        if (cu_device_get_count_fn(&count) != cuda_success) return error.CudaInitFailed;
        if (count <= 0) return error.CudaNoDevices;

        const n: usize = @intCast(count);
        const mems = try allocator.alloc(usize, n);
        errdefer allocator.free(mems);

        for (0..n) |i| {
            var dev_handle: c_int = 0;
            if (cu_device_get_fn(&dev_handle, @intCast(i)) != cuda_success) {
                mems[i] = 0;
                continue;
            }
            var total: usize = 0;
            if (cu_device_total_mem_fn(&total, dev_handle) != cuda_success) {
                mems[i] = 0;
                continue;
            }
            mems[i] = total;
        }
        return mems;
    }

    pub fn deinit(self: *Device) void {
        self.logLaunchStatsSnapshot("deinit");
        if (self.context) |ctx| {
            if (tls_current_context == ctx) tls_current_context = null;
            _ = self.api.cu_ctx_destroy(ctx);
            self.context = null;
        }
        self.lib.close();
    }

    pub fn launchStatsEnabled(self: *const Device) bool {
        return self.launch_stats_enabled;
    }

    pub fn setLaunchPhase(self: *Device, phase: LaunchPhase) LaunchPhase {
        const previous = self.launch_phase;
        self.launch_phase = phase;
        return previous;
    }

    pub fn setLaunchFamily(self: *Device, family: LaunchFamily) LaunchFamily {
        const previous = self.launch_family;
        self.launch_family = family;
        return previous;
    }

    pub fn logLaunchStatsSnapshot(self: *const Device, phase: []const u8) void {
        if (!self.launch_stats_enabled) return;
        const launch_calls = stats_launch_calls.load(.monotonic);
        const launch_ns = stats_launch_ns.load(.monotonic);
        const launch_calls_prefill = stats_launch_calls_prefill.load(.monotonic);
        const launch_ns_prefill = stats_launch_ns_prefill.load(.monotonic);
        const launch_calls_decode = stats_launch_calls_decode.load(.monotonic);
        const launch_ns_decode = stats_launch_ns_decode.load(.monotonic);
        const make_current_calls = stats_make_current_calls.load(.monotonic);
        const make_current_fastpath_hits = stats_make_current_fastpath_hits.load(.monotonic);
        const set_current_calls = stats_set_current_calls.load(.monotonic);
        const set_current_ns = stats_set_current_ns.load(.monotonic);
        const avg_launch_us: f64 = if (launch_calls == 0)
            0.0
        else
            @as(f64, @floatFromInt(launch_ns)) / @as(f64, @floatFromInt(launch_calls)) / 1000.0;
        const avg_set_current_us: f64 = if (set_current_calls == 0)
            0.0
        else
            @as(f64, @floatFromInt(set_current_ns)) / @as(f64, @floatFromInt(set_current_calls)) / 1000.0;
        var body_buf: [192]u8 = undefined;
        const body = std.fmt.bufPrint(
            &body_buf,
            "CUDA launch stats {s}: launches={d} launch_ms={d:.3} launch_us={d:.3} make_current={d} fastpath={d} set_current={d} set_current_ms={d:.3} set_current_us={d:.3}",
            .{
                phase,
                launch_calls,
                @as(f64, @floatFromInt(launch_ns)) / 1_000_000.0,
                avg_launch_us,
                make_current_calls,
                make_current_fastpath_hits,
                set_current_calls,
                @as(f64, @floatFromInt(set_current_ns)) / 1_000_000.0,
                avg_set_current_us,
            },
        ) catch "CUDA launch stats";
        log.info("inference", body, .{});
        var phase_body_buf: [192]u8 = undefined;
        const phase_body = std.fmt.bufPrint(
            &phase_body_buf,
            "CUDA launch phase stats {s}: prefill_launches={d} prefill_ms={d:.3} decode_launches={d} decode_ms={d:.3}",
            .{
                phase,
                launch_calls_prefill,
                @as(f64, @floatFromInt(launch_ns_prefill)) / 1_000_000.0,
                launch_calls_decode,
                @as(f64, @floatFromInt(launch_ns_decode)) / 1_000_000.0,
            },
        ) catch "CUDA launch phase stats";
        log.info("inference", phase_body, .{});
        self.logLaunchFamilyStats(phase);
    }

    fn logLaunchFamilyStats(self: *const Device, phase: []const u8) void {
        _ = self;
        var idx: usize = 0;
        while (idx < launch_family_count) : (idx += 1) {
            const calls = stats_family_calls[idx].load(.monotonic);
            if (calls == 0) continue;
            const ns_total = stats_family_ns[idx].load(.monotonic);
            const calls_prefill = stats_family_calls_prefill[idx].load(.monotonic);
            const ns_prefill = stats_family_ns_prefill[idx].load(.monotonic);
            const calls_decode = stats_family_calls_decode[idx].load(.monotonic);
            const ns_decode = stats_family_ns_decode[idx].load(.monotonic);
            const avg_us: f64 = @as(f64, @floatFromInt(ns_total)) / @as(f64, @floatFromInt(calls)) / 1000.0;
            var family_buf: [224]u8 = undefined;
            const family_body = std.fmt.bufPrint(
                &family_buf,
                "CUDA launch family stats {s}: family={s} launches={d} launch_ms={d:.3} launch_us={d:.3} prefill_launches={d} prefill_ms={d:.3} decode_launches={d} decode_ms={d:.3}",
                .{
                    phase,
                    launch_family_names[idx],
                    calls,
                    @as(f64, @floatFromInt(ns_total)) / 1_000_000.0,
                    avg_us,
                    calls_prefill,
                    @as(f64, @floatFromInt(ns_prefill)) / 1_000_000.0,
                    calls_decode,
                    @as(f64, @floatFromInt(ns_decode)) / 1_000_000.0,
                },
            ) catch "CUDA launch family stats";
            log.info("inference", family_body, .{});
        }
    }

    pub fn name(self: *const Device) []const u8 {
        const end = std.mem.indexOfScalar(u8, self.name_buffer[0..], 0) orelse self.name_buffer.len;
        return self.name_buffer[0..end];
    }

    pub fn setLaunchStream(self: *Device, stream: ?StreamHandle) void {
        self.launch_stream = stream;
    }

    pub fn getLaunchStream(self: *const Device) ?StreamHandle {
        return self.launch_stream;
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

    pub fn memoryInfo(self: *Device) !struct { free: usize, total: usize } {
        const cu_mem_get_info = self.api.cu_mem_get_info orelse return error.CudaQueryUnavailable;
        try self.makeCurrent();
        var free: usize = 0;
        var total: usize = 0;
        if (cu_mem_get_info(&free, &total) != cuda_success) return error.CudaQueryFailed;
        return .{ .free = free, .total = total };
    }

    pub fn usedMemory(self: *Device) !usize {
        const info = try self.memoryInfo();
        if (info.total >= info.free) return info.total - info.free;
        return 0;
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
        if (self.launch_stats_enabled) _ = stats_make_current_calls.fetchAdd(1, .monotonic);
        if (self.context == null) return error.CudaContextLost;
        if (tls_current_context == self.context) {
            if (self.launch_stats_enabled) _ = stats_make_current_fastpath_hits.fetchAdd(1, .monotonic);
            return;
        }
        const set_start_ns: u64 = if (self.launch_stats_enabled) monotonicNowNs() else 0;
        if (self.api.cu_ctx_set_current(self.context) != cuda_success) {
            tls_current_context = null;
            return error.CudaContextLost;
        }
        tls_current_context = self.context;
        if (self.launch_stats_enabled) {
            _ = stats_set_current_calls.fetchAdd(1, .monotonic);
            _ = stats_set_current_ns.fetchAdd(monotonicNowNs() - set_start_ns, .monotonic);
        }
    }

    pub fn supportsModuleLaunch(self: *const Device) bool {
        return self.api.cu_module_load_data != null and
            self.api.cu_module_get_function != null and
            self.api.cu_module_unload != null and
            self.api.cu_launch_kernel != null;
    }

    pub fn supportsStreams(self: *const Device) bool {
        return self.api.cu_stream_create != null and
            self.api.cu_stream_destroy != null and
            self.api.cu_stream_synchronize != null;
    }

    pub fn supportsGraphLaunch(self: *const Device) bool {
        return self.api.cu_stream_begin_capture != null and
            self.api.cu_stream_end_capture != null and
            self.api.cu_graph_instantiate_with_flags != null and
            self.api.cu_graph_destroy != null and
            self.api.cu_graph_exec_destroy != null and
            self.api.cu_graph_exec_update != null and
            self.api.cu_graph_launch != null;
    }

    pub fn streamBeginCapture(self: *Device, stream: StreamHandle) !void {
        const cu_stream_begin_capture = self.api.cu_stream_begin_capture orelse return error.CudaGraphApiUnavailable;
        try self.makeCurrent();
        if (cu_stream_begin_capture(stream, cu_stream_capture_mode_global) != cuda_success) return error.CudaGraphCaptureFailed;
    }

    pub fn streamEndCapture(self: *Device, stream: StreamHandle) !GraphHandle {
        const cu_stream_end_capture = self.api.cu_stream_end_capture orelse return error.CudaGraphApiUnavailable;
        try self.makeCurrent();

        var graph: ?*anyopaque = null;
        if (cu_stream_end_capture(stream, &graph) != cuda_success or graph == null) return error.CudaGraphCaptureFailed;
        return graph.?;
    }

    pub fn createStream(self: *Device) !StreamHandle {
        const cu_stream_create = self.api.cu_stream_create orelse return error.CudaStreamApiUnavailable;
        try self.makeCurrent();

        var stream: ?*anyopaque = null;
        if (cu_stream_create(&stream, 0) != cuda_success or stream == null) {
            return error.CudaStreamCreateFailed;
        }
        return stream.?;
    }

    pub fn destroyStream(self: *Device, stream: StreamHandle) void {
        const cu_stream_destroy = self.api.cu_stream_destroy orelse return;
        self.makeCurrent() catch return;
        _ = cu_stream_destroy(stream);
    }

    pub fn synchronizeStream(self: *Device, stream: StreamHandle) !void {
        const cu_stream_synchronize = self.api.cu_stream_synchronize orelse return error.CudaStreamApiUnavailable;
        try self.makeCurrent();
        if (cu_stream_synchronize(stream) != cuda_success) return error.CudaSynchronizeFailed;
    }

    pub fn graphInstantiate(self: *Device, graph: GraphHandle) !GraphExecHandle {
        const cu_graph_instantiate_with_flags = self.api.cu_graph_instantiate_with_flags orelse return error.CudaGraphApiUnavailable;
        try self.makeCurrent();

        var exec: ?*anyopaque = null;
        if (cu_graph_instantiate_with_flags(&exec, graph, 0) != cuda_success or exec == null) {
            return error.CudaGraphInstantiateFailed;
        }
        return exec.?;
    }

    pub fn graphDestroy(self: *Device, graph: GraphHandle) void {
        const cu_graph_destroy = self.api.cu_graph_destroy orelse return;
        self.makeCurrent() catch return;
        _ = cu_graph_destroy(graph);
    }

    pub fn graphExecDestroy(self: *Device, exec: GraphExecHandle) void {
        const cu_graph_exec_destroy = self.api.cu_graph_exec_destroy orelse return;
        self.makeCurrent() catch return;
        _ = cu_graph_exec_destroy(exec);
    }

    pub fn graphExecUpdate(self: *Device, exec: GraphExecHandle, graph: GraphHandle) !void {
        const cu_graph_exec_update = self.api.cu_graph_exec_update orelse return error.CudaGraphApiUnavailable;
        try self.makeCurrent();

        var error_node: ?*anyopaque = null;
        var update_result: c_int = 0;
        if (cu_graph_exec_update(exec, graph, &error_node, &update_result) != cuda_success) {
            return error.CudaGraphUpdateFailed;
        }
        if (update_result != 0) return error.CudaGraphUpdateFailed;
    }

    pub fn graphLaunch(self: *Device, exec: GraphExecHandle, stream: ?StreamHandle) !void {
        const cu_graph_launch = self.api.cu_graph_launch orelse return error.CudaGraphApiUnavailable;
        try self.makeCurrent();
        if (cu_graph_launch(exec, if (stream) |s| s else null) != cuda_success) {
            return error.CudaGraphLaunchFailed;
        }
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
        return self.launchKernelOnStream(
            function,
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z,
            shared_mem_bytes,
            self.launch_stream,
            kernel_params,
        );
    }

    pub fn launchKernelOnStream(
        self: *Device,
        function: FunctionHandle,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        block_x: u32,
        block_y: u32,
        block_z: u32,
        shared_mem_bytes: u32,
        stream: ?StreamHandle,
        kernel_params: ?[*]const ?*anyopaque,
    ) !void {
        const cu_launch_kernel = self.api.cu_launch_kernel orelse return error.CudaModuleApiUnavailable;
        try self.makeCurrent();
        const launch_start_ns: u64 = if (self.launch_stats_enabled) monotonicNowNs() else 0;

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
            if (stream) |s| s else null,
            params_ptr,
            null, // extra
        ) != cuda_success) {
            return error.CudaKernelLaunchFailed;
        }
        if (self.launch_stats_enabled) {
            const launch_elapsed_ns = monotonicNowNs() - launch_start_ns;
            const family_idx: usize = @intFromEnum(self.launch_family);
            _ = stats_launch_calls.fetchAdd(1, .monotonic);
            _ = stats_launch_ns.fetchAdd(launch_elapsed_ns, .monotonic);
            _ = stats_family_calls[family_idx].fetchAdd(1, .monotonic);
            _ = stats_family_ns[family_idx].fetchAdd(launch_elapsed_ns, .monotonic);
            switch (self.launch_phase) {
                .prefill => {
                    _ = stats_launch_calls_prefill.fetchAdd(1, .monotonic);
                    _ = stats_launch_ns_prefill.fetchAdd(launch_elapsed_ns, .monotonic);
                    _ = stats_family_calls_prefill[family_idx].fetchAdd(1, .monotonic);
                    _ = stats_family_ns_prefill[family_idx].fetchAdd(launch_elapsed_ns, .monotonic);
                },
                .decode => {
                    _ = stats_launch_calls_decode.fetchAdd(1, .monotonic);
                    _ = stats_launch_ns_decode.fetchAdd(launch_elapsed_ns, .monotonic);
                    _ = stats_family_calls_decode[family_idx].fetchAdd(1, .monotonic);
                    _ = stats_family_ns_decode[family_idx].fetchAdd(launch_elapsed_ns, .monotonic);
                },
                .none => {},
            }
        }
    }

    /// Probe whether this device can directly access memory on a peer device via P2P.
    pub fn canAccessPeer(self: *Device, peer: *Device) bool {
        const cu_fn = self.api.cu_device_can_access_peer orelse return false;
        var can_access: c_int = 0;
        if (cu_fn(&can_access, self.device_index, peer.device_index) != cuda_success) return false;
        return can_access != 0;
    }

    /// Enable P2P memory access from this device's context to a peer device's context.
    /// Both devices must support P2P (check with canAccessPeer first).
    pub fn enablePeerAccess(self: *Device, peer: *Device) !void {
        const cu_fn = self.api.cu_ctx_enable_peer_access orelse return error.CudaPeerAccessUnavailable;
        try self.makeCurrent();
        const rc = cu_fn(peer.context, 0);
        // CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED (704) is harmless.
        if (rc != cuda_success and rc != 704) return error.CudaPeerAccessFailed;
    }

    /// Async copy between two device contexts. The copy is enqueued on the given stream
    /// (which must belong to the source context).
    pub fn memcpyPeerAsync(
        self: *Device,
        dst_ptr: u64,
        dst_context: ?*anyopaque,
        src_ptr: u64,
        src_context: ?*anyopaque,
        byte_count: usize,
        stream: ?StreamHandle,
    ) !void {
        const cu_fn = self.api.cu_memcpy_peer_async orelse return error.CudaPeerAccessUnavailable;
        try self.makeCurrent();
        if (cu_fn(dst_ptr, dst_context, src_ptr, src_context, byte_count, if (stream) |s| s else null) != cuda_success) {
            return error.CudaCopyFailed;
        }
    }
};

fn launchStatsEnabledForCurrentLogLevel() bool {
    return @intFromEnum(log.getLogLevel()) <= @intFromEnum(log.Level.info);
}

fn monotonicNowNs() u64 {
    return @as(u64, @intCast(std.time.nanoTimestamp()));
}

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

    pub fn copyFrom(self: *const Buffer, device: *Device, src: *const Buffer, byte_count: usize) !void {
        if (byte_count > self.size or byte_count > src.size) return error.InvalidArgument;
        if (byte_count == 0) return;
        try device.makeCurrent();
        if (device.api.cu_memcpy_dtod(self.pointer, src.pointer, byte_count) != cuda_success) {
            return error.CudaCopyFailed;
        }
    }

    pub fn download(self: *const Buffer, device: *Device, data: []u8) !void {
        if (data.len > self.size) return error.InvalidArgument;
        if (data.len == 0) return;

        try device.makeCurrent();
        if (device.launch_stream) |stream| {
            try device.synchronizeStream(stream);
        }
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
        .cu_mem_get_info = lookupOptionalAny(CuMemGetInfoFn, lib, &.{ "cuMemGetInfo_v2", "cuMemGetInfo" }),
        .cu_memcpy_htod = try lookupRequiredAny(CuMemcpyHtoDFn, lib, &.{ "cuMemcpyHtoD_v2", "cuMemcpyHtoD" }),
        .cu_memcpy_dtoh = try lookupRequiredAny(CuMemcpyDtoHFn, lib, &.{ "cuMemcpyDtoH_v2", "cuMemcpyDtoH" }),
        .cu_memcpy_dtod = try lookupRequiredAny(CuMemcpyDtoDFn, lib, &.{ "cuMemcpyDtoD_v2", "cuMemcpyDtoD" }),
        .cu_device_get_name = try lookupRequired(CuDeviceGetNameFn, lib, "cuDeviceGetName"),
        .cu_device_total_mem = lookupOptionalAny(CuDeviceTotalMemFn, lib, &.{ "cuDeviceTotalMem_v2", "cuDeviceTotalMem" }),
        .cu_device_get_attribute = lookupOptional(CuDeviceGetAttributeFn, lib, "cuDeviceGetAttribute"),
        .cu_module_load_data = lookupOptional(CuModuleLoadDataFn, lib, "cuModuleLoadData"),
        .cu_module_get_function = lookupOptional(CuModuleGetFunctionFn, lib, "cuModuleGetFunction"),
        .cu_module_unload = lookupOptional(CuModuleUnloadFn, lib, "cuModuleUnload"),
        .cu_stream_create = lookupOptional(CuStreamCreateFn, lib, "cuStreamCreate"),
        .cu_stream_destroy = lookupOptionalAny(CuStreamDestroyFn, lib, &.{ "cuStreamDestroy_v2", "cuStreamDestroy" }),
        .cu_stream_synchronize = lookupOptional(CuStreamSynchronizeFn, lib, "cuStreamSynchronize"),
        .cu_stream_begin_capture = lookupOptional(CuStreamBeginCaptureFn, lib, "cuStreamBeginCapture"),
        .cu_stream_end_capture = lookupOptional(CuStreamEndCaptureFn, lib, "cuStreamEndCapture"),
        .cu_graph_instantiate_with_flags = lookupOptional(CuGraphInstantiateWithFlagsFn, lib, "cuGraphInstantiateWithFlags"),
        .cu_graph_destroy = lookupOptional(CuGraphDestroyFn, lib, "cuGraphDestroy"),
        .cu_graph_exec_destroy = lookupOptional(CuGraphExecDestroyFn, lib, "cuGraphExecDestroy"),
        .cu_graph_exec_update = lookupOptional(CuGraphExecUpdateFn, lib, "cuGraphExecUpdate"),
        .cu_graph_launch = lookupOptional(CuGraphLaunchFn, lib, "cuGraphLaunch"),
        .cu_launch_kernel = lookupOptional(CuLaunchKernelFn, lib, "cuLaunchKernel"),
        .cu_device_can_access_peer = lookupOptional(CuDeviceCanAccessPeerFn, lib, "cuDeviceCanAccessPeer"),
        .cu_ctx_enable_peer_access = lookupOptional(CuCtxEnablePeerAccessFn, lib, "cuCtxEnablePeerAccess"),
        .cu_memcpy_peer_async = lookupOptionalAny(CuMemcpyPeerAsyncFn, lib, &.{ "cuMemcpyPeerAsync_v2", "cuMemcpyPeerAsync" }),
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

test "Device.usedMemory is bounded by Device.totalMemory when available" {
    if (probeRuntime() != .available) return error.SkipZigTest;

    var device = try Device.init();
    defer device.deinit();

    const total = device.totalMemory() catch |err| {
        try std.testing.expect(err == error.CudaQueryUnavailable);
        return;
    };
    const used = device.usedMemory() catch |err| {
        try std.testing.expect(err == error.CudaQueryUnavailable or err == error.CudaQueryFailed);
        return;
    };
    try std.testing.expect(used <= total);
}

test "Device.supportsModuleLaunch reports availability after Device.init" {
    if (probeRuntime() != .available) return error.SkipZigTest;

    var device = try Device.init();
    defer device.deinit();

    _ = device.supportsModuleLaunch();
}

test "Device graph/stream support probes are callable" {
    if (probeRuntime() != .available) return error.SkipZigTest;

    var device = try Device.init();
    defer device.deinit();

    _ = device.supportsStreams();
    _ = device.supportsGraphLaunch();
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

test "Device.initAt returns CudaInvalidDevice for out-of-range ordinal" {
    if (probeRuntime() != .available) return error.SkipZigTest;

    const count = try Device.deviceCount();
    const result = Device.initAt(count);
    try std.testing.expectError(error.CudaInvalidDevice, result);
}

test "Device.ordinal matches initAt argument" {
    if (probeRuntime() != .available) return error.SkipZigTest;

    var device = try Device.initAt(0);
    defer device.deinit();

    try std.testing.expectEqual(@as(usize, 0), device.ordinal());
}

test "Device.deviceCount returns at least one when runtime is available" {
    if (probeRuntime() != .available) return error.SkipZigTest;

    const count = try Device.deviceCount();
    try std.testing.expect(count >= 1);
}

test "Device.canAccessPeer returns without error for same device" {
    if (probeRuntime() != .available) return error.SkipZigTest;

    var device = try Device.initAt(0);
    defer device.deinit();

    // P2P to self may or may not be supported; just verify no crash.
    _ = device.canAccessPeer(&device);
}

test "Device.initAt creates independent contexts for different ordinals" {
    if (probeRuntime() != .available) return error.SkipZigTest;

    const count = try Device.deviceCount();
    if (count < 2) return error.SkipZigTest;

    var dev0 = try Device.initAt(0);
    defer dev0.deinit();

    var dev1 = try Device.initAt(1);
    defer dev1.deinit();

    try std.testing.expectEqual(@as(usize, 0), dev0.ordinal());
    try std.testing.expectEqual(@as(usize, 1), dev1.ordinal());
    try std.testing.expect(dev0.context != dev1.context);
}
