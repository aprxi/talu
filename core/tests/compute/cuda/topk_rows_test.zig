//! Integration tests for CUDA batched top-k row extraction.

const std = @import("std");
const main = @import("main");
const cuda = main.core.compute.cuda;

const test_rows: usize = 3;
const test_vocab: usize = 67;
const test_k: usize = 16;

fn cpuTopKRow(logits: []const f32, out_ids: *[test_k]u32, out_vals: *[test_k]f32) void {
    std.debug.assert(logits.len == test_vocab);

    var sorted_ids: [test_vocab]u32 = undefined;
    for (0..test_vocab) |idx| {
        sorted_ids[idx] = @intCast(idx);
    }

    const SortCtx = struct {
        logits: []const f32,
    };
    const sortLess = struct {
        fn call(ctx: SortCtx, lhs: u32, rhs: u32) bool {
            const lhs_idx: usize = @intCast(lhs);
            const rhs_idx: usize = @intCast(rhs);
            const lhs_value = ctx.logits[lhs_idx];
            const rhs_value = ctx.logits[rhs_idx];
            return (lhs_value > rhs_value) or (lhs_value == rhs_value and lhs < rhs);
        }
    }.call;

    std.sort.heap(u32, sorted_ids[0..], SortCtx{ .logits = logits }, sortLess);

    for (0..test_k) |pick| {
        const id = sorted_ids[pick];
        const id_idx: usize = @intCast(id);
        out_ids[pick] = id;
        out_vals[pick] = logits[id_idx];
    }
}

test "topk_rows_f32.runTwoPhase matches CPU reference and preserves unique ordering" {
    if (cuda.probeRuntime() != .available) return error.SkipZigTest;

    var device = cuda.Device.init() catch |err| {
        if (err == error.CudaInitFailed or err == error.CudaNoDevices) return error.SkipZigTest;
        return err;
    };
    defer device.deinit();

    if (!device.supportsModuleLaunch()) return error.SkipZigTest;

    var registry = cuda.Registry.init(std.testing.allocator, &device);
    defer registry.deinit();

    try registry.loadEmbeddedModule(cuda.topk_rows_f32.embedded_module);
    const phase1 = try registry.resolveFunction(
        cuda.topk_rows_f32.phase1_op_name,
        cuda.topk_rows_f32.phase1_symbol,
    );
    const phase2 = try registry.resolveFunction(
        cuda.topk_rows_f32.phase2_op_name,
        cuda.topk_rows_f32.phase2_symbol,
    );
    try std.testing.expectEqual(cuda.registry.KernelSource.embedded_module, phase1.source);
    try std.testing.expectEqual(cuda.registry.KernelSource.embedded_module, phase2.source);

    var logits_host = [_]f32{0.0} ** (test_rows * test_vocab);
    for (0..test_rows) |row| {
        for (0..test_vocab) |col| {
            const raw: u32 = @intCast((row * 53 + col * 29 + col / 3) % 211);
            var value = @as(f32, @floatFromInt(raw)) * 0.013 - 1.3;
            if (col % 7 == 0 or col % 11 == 0) value = 0.75; // inject tie-rich plateaus
            if (col == ((13 + row * 5) % test_vocab) or col == ((31 + row * 9) % test_vocab)) {
                value = 1.9; // deterministic equal maxima, tie-broken by lower id
            }
            logits_host[row * test_vocab + col] = value;
        }
    }

    var out_vals_host = [_]f32{0.0} ** (test_rows * test_k);
    var out_ids_host = [_]u32{0} ** (test_rows * test_k);

    var logits_dev = try device.allocBuffer(logits_host.len * @sizeOf(f32));
    defer logits_dev.deinit(&device);
    var out_vals_dev = try device.allocBuffer(out_vals_host.len * @sizeOf(f32));
    defer out_vals_dev.deinit(&device);
    var out_ids_dev = try device.allocBuffer(out_ids_host.len * @sizeOf(u32));
    defer out_ids_dev.deinit(&device);

    const scratch_bytes = cuda.topk_rows_f32.scratchBytes(@intCast(test_rows), @intCast(test_k));
    var scratch_vals_dev = try device.allocBuffer(scratch_bytes);
    defer scratch_vals_dev.deinit(&device);
    var scratch_ids_dev = try device.allocBuffer(scratch_bytes);
    defer scratch_ids_dev.deinit(&device);

    try logits_dev.upload(&device, std.mem.sliceAsBytes(logits_host[0..]));

    var arg_pack = cuda.ArgPack.init(std.testing.allocator);
    defer arg_pack.deinit();

    try cuda.topk_rows_f32.runTwoPhase(
        &arg_pack,
        &device,
        phase1.function,
        phase2.function,
        &out_vals_dev,
        &out_ids_dev,
        &logits_dev,
        &scratch_vals_dev,
        &scratch_ids_dev,
        @intCast(test_rows),
        @intCast(test_vocab),
        @intCast(test_k),
        @intCast(test_k),
    );
    try device.synchronize();

    try out_vals_dev.download(&device, std.mem.sliceAsBytes(out_vals_host[0..]));
    try out_ids_dev.download(&device, std.mem.sliceAsBytes(out_ids_host[0..]));

    for (0..test_rows) |row| {
        const row_start = row * test_vocab;
        const row_end = row_start + test_vocab;
        const logits_row = logits_host[row_start..row_end];

        var expected_ids: [test_k]u32 = undefined;
        var expected_vals: [test_k]f32 = undefined;
        cpuTopKRow(logits_row, &expected_ids, &expected_vals);

        var seen_ids = [_]bool{false} ** test_vocab;
        for (0..test_k) |pick| {
            const out_idx = row * test_k + pick;
            const got_id = out_ids_host[out_idx];
            const got_id_idx: usize = @intCast(got_id);
            const got_val = out_vals_host[out_idx];

            try std.testing.expect(got_id_idx < test_vocab);
            try std.testing.expect(!seen_ids[got_id_idx]);
            seen_ids[got_id_idx] = true;

            if (pick > 0) {
                const prev_idx = row * test_k + (pick - 1);
                const prev_val = out_vals_host[prev_idx];
                const prev_id = out_ids_host[prev_idx];
                try std.testing.expect(prev_val > got_val or (prev_val == got_val and prev_id < got_id));
            }

            try std.testing.expectEqual(expected_ids[pick], got_id);
            try std.testing.expectApproxEqAbs(expected_vals[pick], got_val, 0.0001);
        }
    }
}
