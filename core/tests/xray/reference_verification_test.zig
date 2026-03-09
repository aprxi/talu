//! End-to-end test for reference recording and verification system

const std = @import("std");
const testing = std.testing;
const xray = @import("../../src/xray/root.zig");

/// Mock sampler that uses teacher forcing
const MockSampler = struct {
    tokens: []const u32,
    index: usize,

    fn sampleNext(self: *MockSampler) ?u32 {
        // Check teacher forcing first
        if (xray.getNextForcedToken()) |forced| {
            return forced;
        }

        // Normal sampling (mock)
        if (self.index >= self.tokens.len) return null;
        const token = self.tokens[self.index];
        self.index += 1;
        return token;
    }
};

/// Mock inference that emits trace points
fn runMockInference(token: u32, add_noise: bool) void {
    // Simulate layer execution with trace emissions
    const layers = 3;
    var layer: u16 = 0;
    while (layer < layers) : (layer += 1) {
        // Simulate attention output
        var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };

        // Add noise if requested (to simulate divergence)
        if (add_noise and layer == 1) {
            data[4] += 10.0; // Significant change
        }

        xray.trace.emit(
            .attn_out,
            layer,
            token,
            token,
            @ptrCast(&data),
            .f32,
            .{ 8, 0, 0, 0 },
            1,
            "mock_attention",
        );

        // Simulate FFN output
        var ffn_data = [_]f32{ 0.1, 0.2, 0.3, 0.4 };
        if (add_noise and layer == 1) {
            ffn_data[2] += 5.0;
        }

        xray.trace.emit(
            .ffn_down,
            layer,
            token,
            token,
            @ptrCast(&ffn_data),
            .f32,
            .{ 4, 0, 0, 0 },
            1,
            "mock_ffn",
        );
    }

    // Final logits
    var logits = [_]f32{ 0.5, 1.5, 2.5, 3.5 };
    xray.trace.emitFinal(
        .lm_head,
        token,
        token,
        @ptrCast(&logits),
        .f32,
        .{ 4, 0, 0, 0 },
        1,
        "mock_lm_head",
    );
}

test "reference recording and verification - success case" {
    const allocator = testing.allocator;

    // ========================================================================
    // PHASE 1: Recording (simulating CPU backend)
    // ========================================================================
    std.debug.print("\n=== Phase 1: Recording Reference ===\n", .{});

    // Create reference recorder
    var recorder = try xray.ReferenceRecorder.init(
        allocator,
        "mock_model",
        42, // seed
        1.0, // temperature
        3, // max_tokens
    );
    defer recorder.deinit();

    // Create verify capture in recording mode
    var record_capture = xray.VerifyCapture.initRecording(allocator, &recorder);
    defer record_capture.deinit();

    // Enable capture
    xray.enableVerifyCapture(&record_capture);

    // Simulate generation of 3 tokens
    const mock_tokens = [_]u32{ 100, 200, 300 };
    for (mock_tokens, 0..) |token, i| {
        std.debug.print("Recording token {d}/{d}: {d}\n", .{ i + 1, mock_tokens.len, token });

        // Run inference (no noise)
        runMockInference(token, false);

        // Record token
        try recorder.recordToken(token);
        recorder.nextToken();
    }

    xray.disableVerifyCapture();

    // Finalize and save reference
    var reference = try recorder.finalize();
    defer reference.deinit();

    std.debug.print("Recorded {d} tokens, {d} stats records\n", .{
        reference.token_transcript.len,
        reference.stats_records.len,
    });

    try testing.expectEqual(@as(usize, 3), reference.token_transcript.len);
    try testing.expect(reference.stats_records.len > 0);

    // ========================================================================
    // PHASE 2: Verification (simulating Metal/CUDA backend) - MATCHING
    // ========================================================================
    std.debug.print("\n=== Phase 2: Verification (no noise) ===\n", .{});

    // Create verifier
    var verifier = xray.ReferenceVerifier.init(allocator, &reference, 1e-3);

    // Create verify capture in verification mode
    var verify_capture = xray.VerifyCapture.initVerification(allocator, &verifier, null);
    defer verify_capture.deinit();

    // Enable capture
    xray.enableVerifyCapture(&verify_capture);

    // Enable teacher forcing
    const TokenProvider = struct {
        ver: *xray.ReferenceVerifier,
        fn getNext(ctx: ?*anyopaque) ?u32 {
            const self: *@This() = @ptrCast(@alignCast(ctx.?));
            return self.ver.getNextToken();
        }
    };
    var token_provider = TokenProvider{ .ver = &verifier };
    xray.enableTeacherForcing(&TokenProvider.getNext, &token_provider);

    // Create mock sampler
    var sampler = MockSampler{ .tokens = &mock_tokens, .index = 0 };

    // Verify generation (should match exactly)
    var verified_count: usize = 0;
    while (sampler.sampleNext()) |token| {
        std.debug.print("Verifying token {d}/{d}: {d}\n", .{
            verified_count + 1,
            mock_tokens.len,
            token,
        });

        // Run inference (no noise - should match reference)
        runMockInference(token, false);

        verifier.nextToken();
        verified_count += 1;
    }

    xray.disableTeacherForcing();
    xray.disableVerifyCapture();

    // Check that verification succeeded
    try testing.expect(!verifier.has_diverged);
    try testing.expectEqual(@as(usize, 3), verified_count);

    std.debug.print("✓ Verification passed: {d} tokens verified\n", .{verified_count});
}

test "reference verification - divergence detection" {
    const allocator = testing.allocator;

    // ========================================================================
    // PHASE 1: Recording
    // ========================================================================
    var recorder = try xray.ReferenceRecorder.init(allocator, "mock_model", 42, 1.0, 2);
    defer recorder.deinit();

    var record_capture = xray.VerifyCapture.initRecording(allocator, &recorder);
    defer record_capture.deinit();

    xray.enableVerifyCapture(&record_capture);

    const mock_tokens = [_]u32{ 100, 200 };
    for (mock_tokens) |token| {
        runMockInference(token, false);
        try recorder.recordToken(token);
        recorder.nextToken();
    }

    xray.disableVerifyCapture();

    var reference = try recorder.finalize();
    defer reference.deinit();

    // ========================================================================
    // PHASE 2: Verification with Noise (should diverge)
    // ========================================================================
    std.debug.print("\n=== Testing Divergence Detection ===\n", .{});

    var verifier = xray.ReferenceVerifier.init(allocator, &reference, 1e-3);

    var verify_capture = xray.VerifyCapture.initVerification(allocator, &verifier, null);
    defer verify_capture.deinit();

    xray.enableVerifyCapture(&verify_capture);

    const TokenProvider = struct {
        ver: *xray.ReferenceVerifier,
        fn getNext(ctx: ?*anyopaque) ?u32 {
            const self: *@This() = @ptrCast(@alignCast(ctx.?));
            return self.ver.getNextToken();
        }
    };
    var token_provider = TokenProvider{ .ver = &verifier };
    xray.enableTeacherForcing(&TokenProvider.getNext, &token_provider);

    var sampler = MockSampler{ .tokens = &mock_tokens, .index = 0 };

    // Run with noise - should detect divergence
    while (sampler.sampleNext()) |token| {
        std.debug.print("Verifying token (with noise): {d}\n", .{token});

        // Run inference WITH NOISE
        runMockInference(token, true);

        verifier.nextToken();
    }

    xray.disableTeacherForcing();
    xray.disableVerifyCapture();

    // Should have detected divergence
    try testing.expect(verifier.has_diverged);
    try testing.expect(verifier.divergence_point != null);

    if (verifier.divergence_point) |div| {
        std.debug.print("✓ Divergence detected at layer {d}, point {s}\n", .{
            div.layer,
            div.point.name(),
        });
        std.debug.print("  Expected RMS: {d:.6}\n", .{div.expected.rms()});
        std.debug.print("  Actual RMS: {d:.6}\n", .{div.actual.rms()});
    }
}

test "JSON serialization round-trip" {
    const allocator = testing.allocator;

    // Create reference
    const stats = xray.TensorStats{
        .count = 100,
        .min = -1.0,
        .max = 1.0,
        .sum = 5.0,
        .sum_sq = 50.0,
        .nan_count = 0,
        .inf_count = 0,
    };

    const original = xray.ReferenceData{
        .model_name = "test_model",
        .seed = 12345,
        .temperature = 0.8,
        .max_tokens = 5,
        .token_transcript = &[_]u32{ 1, 2, 3, 4, 5 },
        .stats_records = &[_]xray.StatsRecord{
            .{
                .token_idx = 0,
                .layer = 0,
                .point = .attn_out,
                .position = 0,
                .stats = stats,
            },
            .{
                .token_idx = 1,
                .layer = 0,
                .point = .ffn_down,
                .position = 1,
                .stats = stats,
            },
        },
        .allocator = allocator,
    };

    // Serialize to string
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();

    try xray.reference.JsonFormat.serialize(&original, buffer.writer());
    const json_str = buffer.items;

    std.debug.print("\nSerialized JSON ({d} bytes):\n{s}\n", .{ json_str.len, json_str[0..@min(json_str.len, 200)] });

    // Deserialize
    var loaded = try xray.reference.JsonFormat.deserialize(allocator, json_str);
    defer loaded.deinit();

    // Verify round-trip
    try testing.expectEqualStrings("test_model", loaded.model_name);
    try testing.expectEqual(@as(u64, 12345), loaded.seed);
    try testing.expectEqual(@as(f32, 0.8), loaded.temperature);
    try testing.expectEqual(@as(u32, 5), loaded.max_tokens);
    try testing.expectEqual(@as(usize, 5), loaded.token_transcript.len);
    try testing.expectEqual(@as(usize, 2), loaded.stats_records.len);

    // Verify tokens
    for (original.token_transcript, loaded.token_transcript) |orig, load| {
        try testing.expectEqual(orig, load);
    }

    // Verify stats
    try testing.expectEqual(original.stats_records[0].point, loaded.stats_records[0].point);
    try testing.expectApproxEqAbs(
        original.stats_records[0].stats.rms(),
        loaded.stats_records[0].stats.rms(),
        0.001,
    );

    std.debug.print("✓ Round-trip successful\n", .{});
}
