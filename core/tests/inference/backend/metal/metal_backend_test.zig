//! Integration tests for inference.backend.metal.MetalBackend
//!
//! Note: MetalBackend is only available on macOS with Metal support.

const std = @import("std");
const builtin = @import("builtin");
const main = @import("main");

const has_metal = builtin.os.tag == .macos;
const metal = if (has_metal) main.inference.backend.metal else void;
const backend_scheduler = main.inference.scheduler;
const runtime_contract = main.inference.runtime_contract;
const weights = if (has_metal) main.inference.backend.metal.executor.weights else void;
const xray = main.xray;
const sampling = main.inference.backend.metal.sampling;

extern fn mlx_test_grouped_affine_moe_gpu_path() c_int;
extern fn mlx_test_depthwise_conv_decode_step() c_int;
extern fn mlx_test_single_query_attention_matches_sdpa() c_int;
extern fn mlx_test_kv_cache_reserve_preserves_prefix() c_int;
extern fn mlx_test_shared_expert_gate_up_fusion() c_int;
extern fn mlx_test_dense_mlp_gate_up_fusion() c_int;
extern fn mlx_test_full_attention_qkv_fusion() c_int;
extern fn mlx_test_topk_candidate_extraction_multi() c_int;
extern fn mlx_last_error() [*:0]const u8;

fn pathExists(path: []const u8) bool {
    if (std.fs.path.isAbsolute(path)) {
        std.fs.accessAbsolute(path, .{}) catch return false;
        return true;
    }
    std.fs.cwd().access(path, .{}) catch return false;
    return true;
}

fn resolveMlxMetallibPath() ?[]const u8 {
    const env = std.posix.getenv("MLX_METALLIB");
    if (env) |value| {
        const raw = std.mem.sliceTo(value, 0);
        if (raw.len > 0 and pathExists(raw)) return raw;
    }

    const candidates = [_][]const u8{
        "zig-out/bin/mlx.metallib",
        "zig-out/lib/mlx.metallib",
        "deps/mlx/lib/mlx.metallib",
        "deps/mlx-src/build/mlx/backend/metal/kernels/mlx.metallib",
        "/opt/homebrew/bin/mlx.metallib",
        "/usr/local/bin/mlx.metallib",
    };
    for (candidates) |candidate| {
        if (pathExists(candidate)) return candidate;
    }
    return null;
}

fn canRunMetalRuntime() bool {
    if (comptime !has_metal) return false;
    if (!metal.isAvailable()) return false;
    const metallib = resolveMlxMetallibPath() orelse return false;

    const allocator = std.testing.allocator;
    const exe_dir = std.fs.selfExeDirPathAlloc(allocator) catch return false;
    defer allocator.free(exe_dir);
    const colocated = std.fs.path.join(allocator, &.{ exe_dir, "mlx.metallib" }) catch return false;
    defer allocator.free(colocated);

    if (!pathExists(colocated)) {
        const metallib_abs = if (std.fs.path.isAbsolute(metallib))
            metallib
        else
            (std.fs.cwd().realpathAlloc(allocator, metallib) catch return false);
        defer if (!std.fs.path.isAbsolute(metallib)) allocator.free(metallib_abs);

        std.fs.copyFileAbsolute(metallib_abs, colocated, .{}) catch return false;
    }
    return true;
}

const BoundSlotState = struct {
    allocator: std.mem.Allocator,
    backend: *metal.MetalBackend,
    slot_index: usize,
    buffers: [runtime_contract.max_state_descriptors][]align(64) u8 = undefined,
    count: usize = 0,

    fn init(
        allocator: std.mem.Allocator,
        backend: *metal.MetalBackend,
        slot_index: usize,
    ) !BoundSlotState {
        var bound = BoundSlotState{
            .allocator = allocator,
            .backend = backend,
            .slot_index = slot_index,
        };
        errdefer bound.deinit();

        const descriptors = backend.stateDescriptors();
        if (descriptors.len == 0) return bound;

        var handles: [runtime_contract.max_state_descriptors]runtime_contract.StateBlockHandle = undefined;
        for (descriptors, 0..) |descriptor, idx| {
            if (descriptor.align_bytes > 64) return error.InvalidStateDescriptorBinding;
            const bytes_u64 = @max(descriptor.size_bytes, runtime_contract.builtin_state_block_bytes);
            const bytes: usize = @intCast(bytes_u64);
            const storage = try allocator.alignedAlloc(u8, .@"64", bytes);
            @memset(storage, 0);
            bound.buffers[idx] = storage;
            bound.count += 1;

            handles[idx] = .{
                .id = descriptor.id,
                .ptr = storage.ptr,
                .size = bytes_u64,
                .align_bytes = 64,
            };
        }

        try backend.bindSlotStateBlocks(slot_index, handles[0..descriptors.len]);
        return bound;
    }

    fn deinit(self: *BoundSlotState) void {
        self.backend.unbindSlotStateBlocks(self.slot_index);
        for (self.buffers[0..self.count]) |buf| self.allocator.free(buf);
        self.count = 0;
    }
};

test "metal backend init/prefill/decode lifecycle is stable across repeated iterations" {
    if (!canRunMetalRuntime()) return;

    const allocator = std.testing.allocator;
    const iterations: usize = 24;

    for (0..iterations) |_| {
        const loaded = try weights.createTestLoadedModel(allocator);
        defer weights.destroyTestLoadedModel(allocator, loaded);
        loaded.runtime.architecture_id = "qwen3";

        var backend = try metal.MetalBackend.init(allocator, loaded, .{});
        defer backend.deinit();

        var bound_state = try BoundSlotState.init(allocator, &backend, 0);
        defer bound_state.deinit();

        const vocab_size = backend.vocabSize();
        const logits = try allocator.alloc(f32, vocab_size);
        defer allocator.free(logits);

        try backend.prefill(&[_]u32{ 1, 2, 3 }, logits);
        const pos = backend.getPosition(0);
        try std.testing.expectEqual(@as(usize, 3), pos);
        try backend.decode(4, pos, logits);
    }
}

test "metal backend slot allocate/free churn is stable" {
    if (!canRunMetalRuntime()) return;

    const allocator = std.testing.allocator;
    const loaded = try weights.createTestLoadedModel(allocator);
    defer weights.destroyTestLoadedModel(allocator, loaded);
    loaded.runtime.architecture_id = "qwen3";

    var backend = try metal.MetalBackend.init(allocator, loaded, .{});
    defer backend.deinit();

    // Repeated slot churn exercises reset/free paths that are easy to get
    // wrong and can cause sporadic heap corruption.
    for (0..64) |_| {
        const slot = backend.allocSlot() orelse return error.TestUnexpectedResult;
        backend.freeSlot(slot);
    }
}

test "metal backend xray record+verify capture path is stable under repeated runs" {
    if (!canRunMetalRuntime()) return;

    const allocator = std.testing.allocator;
    const iterations: usize = 20;

    for (0..iterations) |_| {
        const loaded = try weights.createTestLoadedModel(allocator);
        defer weights.destroyTestLoadedModel(allocator, loaded);
        loaded.runtime.architecture_id = "qwen3";

        var backend = try metal.MetalBackend.init(allocator, loaded, .{});
        defer backend.deinit();

        var bound_state = try BoundSlotState.init(allocator, &backend, 0);
        defer bound_state.deinit();

        const vocab_size = backend.vocabSize();
        const logits = try allocator.alloc(f32, vocab_size);
        defer allocator.free(logits);

        var recorder = try xray.ReferenceRecorder.init(allocator, "metal-test", 42, 1.0, 8);
        defer recorder.deinit();
        {
            var record_capture = xray.VerifyCapture.initRecording(allocator, &recorder);
            defer record_capture.deinit();

            xray.enableVerifyCapture(&record_capture);
            defer xray.disableVerifyCapture();

            try backend.prefill(&[_]u32{ 1, 2, 3, 4 }, logits);
            try backend.decode(5, backend.getPosition(0), logits);
            try backend.decode(6, backend.getPosition(0), logits);
        }

        var ref_data = try recorder.finalize();
        defer ref_data.deinit();

        var verifier = xray.ReferenceVerifier.init(allocator, &ref_data, 1e-3, 1e-3);
        defer verifier.deinit();
        {
            var verify_capture = xray.VerifyCapture.initVerification(allocator, &verifier, null);
            defer verify_capture.deinit();

            xray.enableVerifyCapture(&verify_capture);
            defer xray.disableVerifyCapture();

            try backend.prefill(&[_]u32{ 1, 2, 3, 4 }, logits);
            try backend.decode(5, backend.getPosition(0), logits);
            try backend.decode(6, backend.getPosition(0), logits);
        }
        try verifier.finish();
    }
}

test "metal bridge grouped-affine moe gpu path matches reference implementation" {
    if (!canRunMetalRuntime()) return;

    const status = mlx_test_grouped_affine_moe_gpu_path();
    if (status != 1) {
        std.debug.print("mlx grouped-affine moe self-test failed: {s}\n", .{std.mem.span(mlx_last_error())});
    }
    try std.testing.expectEqual(@as(c_int, 1), status);
}

test "metal bridge depthwise conv decode step matches conv1d" {
    if (!canRunMetalRuntime()) return;

    const status = mlx_test_depthwise_conv_decode_step();
    if (status != 1) {
        std.debug.print("mlx depthwise conv decode self-test failed: {s}\n", .{std.mem.span(mlx_last_error())});
    }
    try std.testing.expectEqual(@as(c_int, 1), status);
}

test "metal bridge single-query attention matches sdpa" {
    if (!canRunMetalRuntime()) return;

    const status = mlx_test_single_query_attention_matches_sdpa();
    if (status != 1) {
        std.debug.print("mlx single-query attention self-test failed: {s}\n", .{std.mem.span(mlx_last_error())});
    }
    try std.testing.expectEqual(@as(c_int, 1), status);
}

test "metal bridge kv cache reserve preserves prefix" {
    if (!canRunMetalRuntime()) return;

    const status = mlx_test_kv_cache_reserve_preserves_prefix();
    if (status != 1) {
        std.debug.print("mlx kv cache reserve self-test failed: {s}\n", .{std.mem.span(mlx_last_error())});
    }
    try std.testing.expectEqual(@as(c_int, 1), status);
}

test "metal bridge shared expert gate-up fusion matches split projections" {
    if (!canRunMetalRuntime()) return;

    const status = mlx_test_shared_expert_gate_up_fusion();
    if (status != 1) {
        std.debug.print("mlx shared expert gate-up self-test failed: {s}\n", .{std.mem.span(mlx_last_error())});
    }
    try std.testing.expectEqual(@as(c_int, 1), status);
}

test "metal bridge dense mlp gate-up fusion matches split projections" {
    if (!canRunMetalRuntime()) return;

    const status = mlx_test_dense_mlp_gate_up_fusion();
    if (status != 1) {
        std.debug.print("mlx dense mlp gate-up self-test failed: {s}\n", .{std.mem.span(mlx_last_error())});
    }
    try std.testing.expectEqual(@as(c_int, 1), status);
}

test "metal bridge full attention qkv fusion matches split projections" {
    if (!canRunMetalRuntime()) return;

    const status = mlx_test_full_attention_qkv_fusion();
    if (status != 1) {
        std.debug.print("mlx full attention qkv fusion self-test failed: {s}\n", .{std.mem.span(mlx_last_error())});
    }
    try std.testing.expectEqual(@as(c_int, 1), status);
}

test "metal bridge top-k candidate extraction supports top_k greater than one" {
    if (!canRunMetalRuntime()) return;

    const status = mlx_test_topk_candidate_extraction_multi();
    if (status != 1) {
        std.debug.print("mlx top-k extraction self-test failed: {s}\n", .{std.mem.span(mlx_last_error())});
    }
    try std.testing.expectEqual(@as(c_int, 1), status);
}

test "metal backend remains stable across repeated scheduler lifecycles on one backend" {
    if (!canRunMetalRuntime()) return;

    const allocator = std.testing.allocator;
    const Scheduler = backend_scheduler.GenericScheduler(metal.MetalBackend);

    const loaded = try weights.createTestLoadedModel(allocator);
    defer weights.destroyTestLoadedModel(allocator, loaded);
    loaded.runtime.architecture_id = "qwen3";

    var backend = try metal.MetalBackend.init(allocator, loaded, .{});
    defer backend.deinit();

    const prompt = [_]u32{ 1, 2, 3, 4 };
    const iterations: usize = 8;

    for (0..iterations) |_| {
        var scheduler = try Scheduler.init(allocator, &backend, .{
            .state_descriptors = backend.stateDescriptors(),
        });
        defer scheduler.deinit();

        var result = try scheduler.generateSync(prompt[0..], 2, null);
        defer result.deinit(allocator);

        try std.testing.expect(result.tokens.len > 0);
    }
}

test "metal backend explicit worker-thread teardown is stable across repeated runs" {
    if (!canRunMetalRuntime()) return;

    const allocator = std.testing.allocator;
    const Scheduler = backend_scheduler.GenericScheduler(metal.MetalBackend);

    const loaded = try weights.createTestLoadedModel(allocator);
    defer weights.destroyTestLoadedModel(allocator, loaded);
    loaded.runtime.architecture_id = "qwen3";

    var backend = try metal.MetalBackend.init(allocator, loaded, .{});
    defer backend.deinit();

    const prompt = [_]u32{ 1, 2, 3, 4 };
    const iterations: usize = 8;

    const WorkerCtx = struct {
        allocator: std.mem.Allocator,
        backend: *metal.MetalBackend,
        prompt: []const u32,
        runError: ?anyerror = null,

        fn run(self: *@This()) void {
            defer self.backend.teardownExecutionThreadState();
            self.runOnce() catch |err| {
                self.runError = err;
            };
        }

        fn runOnce(self: *@This()) !void {
            var scheduler = try Scheduler.init(self.allocator, self.backend, .{
                .state_descriptors = self.backend.stateDescriptors(),
            });
            defer scheduler.deinit();

            var result = try scheduler.generateSync(self.prompt, 2, null);
            defer result.deinit(self.allocator);

            try std.testing.expect(result.tokens.len > 0);
        }
    };

    for (0..iterations) |_| {
        var ctx = WorkerCtx{
            .allocator = allocator,
            .backend = &backend,
            .prompt = prompt[0..],
        };
        const thread = try std.Thread.spawn(.{}, WorkerCtx.run, .{&ctx});
        thread.join();
        if (ctx.runError) |err| return err;
    }
}

test "metal backend scheduler top-k route remains stable across repeated runs" {
    if (!canRunMetalRuntime()) return;

    const allocator = std.testing.allocator;
    const Scheduler = backend_scheduler.GenericScheduler(metal.MetalBackend);

    const loaded = try weights.createTestLoadedModel(allocator);
    defer weights.destroyTestLoadedModel(allocator, loaded);
    // The lightweight test fixture weights are qwen3-shaped. Forcing qwen3_5
    // metadata here hits deterministic projection-shape mismatch in fused dense
    // kernels before decode routing logic is exercised.
    loaded.runtime.architecture_id = "qwen3";

    var backend = try metal.MetalBackend.init(allocator, loaded, .{});
    defer backend.deinit();

    const prompt = [_]u32{ 1, 2, 3, 4 };
    const iterations: usize = 16;

    const sampling_config = sampling.SamplingConfig{
        .strategy = .top_k,
        .temperature = 0.8,
        .top_k = 10,
        .top_p = 1.0,
    };

    for (0..iterations) |_| {
        var scheduler = try Scheduler.init(allocator, &backend, .{
            .state_descriptors = backend.stateDescriptors(),
            .default_sampling = sampling_config,
        });
        defer scheduler.deinit();

        var result = try scheduler.generateSync(prompt[0..], 8, .{
            .sampling = sampling_config,
        });
        defer result.deinit(allocator);

        try std.testing.expect(result.tokens.len > 0);
    }
}
