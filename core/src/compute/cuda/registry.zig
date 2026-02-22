//! Kernel registry and source selection for CUDA modules.

const std = @import("std");
const device_mod = @import("device.zig");
const module_mod = @import("module.zig");
const manifest_mod = @import("manifest.zig");

pub const KernelSource = enum {
    embedded_ptx,
    sideload_cubin,
};

pub const ResolvedFunction = struct {
    source: KernelSource,
    function: module_mod.Function,
};

pub const Registry = struct {
    allocator: std.mem.Allocator,
    device: *device_mod.Device,
    embedded_module: ?module_mod.Module = null,
    sideload_module: ?module_mod.Module = null,
    sideload_manifest: ?manifest_mod.ParsedManifest = null,

    pub fn init(allocator: std.mem.Allocator, device: *device_mod.Device) Registry {
        return .{
            .allocator = allocator,
            .device = device,
        };
    }

    pub fn deinit(self: *Registry) void {
        if (self.sideload_module) |*module| module.deinit(self.device);
        if (self.embedded_module) |*module| module.deinit(self.device);
        if (self.sideload_manifest) |*manifest| manifest.deinit();
        self.* = undefined;
    }

    pub fn loadEmbeddedModule(self: *Registry, module_bytes: []const u8) !void {
        if (self.embedded_module) |*module| {
            module.deinit(self.device);
            self.embedded_module = null;
        }
        self.embedded_module = try module_mod.Module.load(self.device, module_bytes);
    }

    pub fn loadSideloadModule(
        self: *Registry,
        manifest_bytes: []const u8,
        module_bytes: []const u8,
    ) !void {
        if (self.sideload_module) |*module| {
            module.deinit(self.device);
            self.sideload_module = null;
        }
        if (self.sideload_manifest) |*manifest| {
            manifest.deinit();
            self.sideload_manifest = null;
        }

        self.sideload_manifest = try manifest_mod.parse(self.allocator, manifest_bytes);
        self.sideload_module = try module_mod.Module.load(self.device, module_bytes);
    }

    pub fn resolveFunction(
        self: *Registry,
        op_name: []const u8,
        embedded_symbol: [:0]const u8,
    ) !ResolvedFunction {
        if (self.sideload_manifest != null and self.sideload_module != null) {
            const manifest = &self.sideload_manifest.?.manifest;
            if (manifest.findSymbol(op_name)) |symbol| {
                const symbol_z = try self.allocator.dupeZ(u8, symbol);
                defer self.allocator.free(symbol_z);

                const function = try self.sideload_module.?.getFunction(self.device, symbol_z);
                return .{
                    .source = .sideload_cubin,
                    .function = function,
                };
            }
        }

        if (self.embedded_module) |module| {
            return .{
                .source = .embedded_ptx,
                .function = try module.getFunction(self.device, embedded_symbol),
            };
        }
        return error.CudaKernelUnavailable;
    }
};

test "Registry.init sets empty module slots" {
    var fake_device: device_mod.Device = undefined;
    const registry = Registry.init(std.testing.allocator, &fake_device);

    try std.testing.expect(registry.embedded_module == null);
    try std.testing.expect(registry.sideload_module == null);
    try std.testing.expect(registry.sideload_manifest == null);
}
