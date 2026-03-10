//! Verification System Integration
//!
//! Integrates reference recording/verification with the trace capture system.
//! Handles:
//! - Recording mode: Captures stats and feeds them to ReferenceRecorder
//! - Verification mode: Checks emissions against reference, triggers panic dump on divergence

const std = @import("std");
const trace = @import("trace.zig");
const capture_mod = @import("capture.zig");
const stats_mod = @import("stats.zig");
const reference_mod = @import("reference.zig");

const TraceEmission = trace.TraceEmission;
const TensorStats = stats_mod.TensorStats;
const TraceCapture = capture_mod.TraceCapture;
const TraceCaptureConfig = capture_mod.TraceCaptureConfig;
const TraceCaptureMode = capture_mod.TraceCaptureMode;
const ReferenceRecorder = reference_mod.ReferenceRecorder;
const ReferenceVerifier = reference_mod.ReferenceVerifier;

/// Mode for verification capture
pub const VerifyMode = enum {
    /// Recording mode: generate normally and capture stats
    record,
    /// Verification mode: force tokens and check stats
    verify,
};

/// Enhanced capture that integrates with reference system
pub const VerifyCapture = struct {
    allocator: std.mem.Allocator,

    /// Base capture for stats collection
    capture: TraceCapture,

    /// Mode
    mode: VerifyMode,

    /// Recording state (when mode == .record)
    recorder: ?*ReferenceRecorder,

    /// Verification state (when mode == .verify)
    verifier: ?*ReferenceVerifier,

    /// Panic dump path (for full tensor dumps on divergence)
    panic_dump_dir: ?[]const u8,

    /// Has a panic dump been triggered?
    panic_triggered: bool,

    fn verificationPointSet() capture_mod.TracePointSet {
        return .{
            .final_norm = true,
            .lm_head = true,
            .logits_scaled = true,
            .logits_ready = true,
            .token_select = true,
        };
    }

    pub fn initRecording(
        allocator: std.mem.Allocator,
        recorder: *ReferenceRecorder,
    ) VerifyCapture {
        const config = TraceCaptureConfig{
            .mode = .stats,
            .points = verificationPointSet(),
        };

        return .{
            .allocator = allocator,
            .capture = TraceCapture.init(allocator, config),
            .mode = .record,
            .recorder = recorder,
            .verifier = null,
            .panic_dump_dir = null,
            .panic_triggered = false,
        };
    }

    pub fn initVerification(
        allocator: std.mem.Allocator,
        verifier: *ReferenceVerifier,
        panic_dump_dir: ?[]const u8,
    ) VerifyCapture {
        const config = TraceCaptureConfig{
            .mode = .stats,
            .points = verificationPointSet(),
        };

        return .{
            .allocator = allocator,
            .capture = TraceCapture.init(allocator, config),
            .mode = .verify,
            .recorder = null,
            .verifier = verifier,
            .panic_dump_dir = panic_dump_dir,
            .panic_triggered = false,
        };
    }

    pub fn deinit(self: *VerifyCapture) void {
        self.capture.deinit();
    }

    fn isTranscriptTokenSelect(emission: TraceEmission) bool {
        return emission.point == .token_select and
            emission.tensor.dtype == .u32 and
            emission.tensor.elementCount() == 1;
    }

    /// Handle a trace emission
    pub fn handleEmission(self: *VerifyCapture, emission: TraceEmission) void {
        const is_token_select = emission.point == .token_select;
        const is_transcript_token_select = isTranscriptTokenSelect(emission);

        // token_select is a control signal: update transcript/token index from
        // scalar u32 events only, then skip stats comparison for this point.
        switch (self.mode) {
            .record => {
                if (self.recorder) |rec| {
                    if (is_transcript_token_select) {
                        const token_id_ptr: *const u32 = @ptrCast(@alignCast(emission.tensor.ptr));
                        rec.recordToken(token_id_ptr.*) catch |err| {
                            std.log.err("Failed to record generated token transcript: {}", .{err});
                        };
                        rec.nextToken();
                    }
                }
            },
            .verify => {
                if (self.verifier) |ver| {
                    if (is_transcript_token_select) {
                        const token_id_ptr: *const u32 = @ptrCast(@alignCast(emission.tensor.ptr));
                        ver.checkToken(token_id_ptr.*) catch |err| {
                            std.log.err("DIVERGENCE DETECTED: {}", .{err});
                            if (ver.divergence_point) |div| {
                                const msg_len = std.mem.indexOfScalar(u8, &div.message, 0) orelse div.message.len;
                                std.log.err("{s}", .{div.message[0..msg_len]});
                            }
                            return;
                        };
                        ver.nextToken();
                    }
                }
            },
        }
        if (is_token_select) return;
        if (!self.capture.config.points.contains(emission.point)) return;

        // Xray emitters must only publish host-accessible tensors. This keeps
        // verification backend-agnostic and avoids duplicating device-specific
        // stats paths here.
        const tensor_stats = stats_mod.compute(emission.tensor);

        switch (self.mode) {
            .record => {
                // Recording mode: feed stats to recorder
                if (self.recorder) |rec| {
                    rec.recordEmission(emission, tensor_stats) catch |err| {
                        std.log.err("Failed to record emission: {}", .{err});
                    };
                }
            },
            .verify => {
                // Verification mode: check against reference
                if (self.verifier) |ver| {
                    ver.checkEmission(emission, tensor_stats) catch |err| {
                        // Divergence detected!
                        std.log.err("DIVERGENCE DETECTED: {}", .{err});

                        if (ver.divergence_point) |div| {
                            const msg_len = std.mem.indexOfScalar(u8, &div.message, 0) orelse div.message.len;
                            std.log.err("{s}", .{div.message[0..msg_len]});

                            // Trigger panic dump if not already done
                            if (!self.panic_triggered and self.panic_dump_dir != null) {
                                self.triggerPanicDump(emission, tensor_stats, &div) catch |dump_err| {
                                    std.log.err("Failed to write panic dump: {}", .{dump_err});
                                };
                                self.panic_triggered = true;
                            }
                        }
                    };
                }
            },
        }
    }

    fn triggerPanicDump(
        self: *VerifyCapture,
        emission: TraceEmission,
        actual_stats: TensorStats,
        divergence: *const ReferenceVerifier.DivergenceInfo,
    ) !void {
        const dump_dir = self.panic_dump_dir orelse return;

        // Switch to full capture mode temporarily to capture the failing tensor
        const old_mode = self.capture.config.mode;
        self.capture.config.mode = .full;
        defer self.capture.config.mode = old_mode;

        // Capture the full tensor
        self.capture.handleEmission(emission);

        // Build filename
        var filename_buf: [256]u8 = undefined;
        const filename = try std.fmt.bufPrint(&filename_buf, "divergence_token{d}_layer{d}_{s}.npz", .{ divergence.token_idx, divergence.layer, divergence.point.name() });

        const full_path = try std.fs.path.join(self.allocator, &[_][]const u8{ dump_dir, filename });
        defer self.allocator.free(full_path);

        // Write NPZ with full tensor data
        // TODO: Integrate with npz writer
        std.log.warn("Would write panic dump to: {s}", .{full_path});
        std.log.warn("Expected RMS: {d:.6}, Actual RMS: {d:.6}", .{
            divergence.expected.rms(),
            actual_stats.rms(),
        });
    }
};

/// Global verify capture for handler integration
var global_verify_capture: ?*VerifyCapture = null;

fn globalVerifyHandler(emission: TraceEmission) void {
    if (global_verify_capture) |cap| {
        cap.handleEmission(emission);
    }
}

pub fn enableVerifyCapture(cap: *VerifyCapture) void {
    global_verify_capture = cap;
    trace.setHandler(&globalVerifyHandler);
}

pub fn disableVerifyCapture() void {
    global_verify_capture = null;
    trace.setHandler(null);
}

// ============================================================================
// Tests
// ============================================================================

test "VerifyCapture recording mode" {
    const allocator = std.testing.allocator;

    var recorder = try ReferenceRecorder.init(allocator, "test_model", 42, 1.0, 10);
    defer recorder.deinit();

    var verify_cap = VerifyCapture.initRecording(allocator, &recorder);
    defer verify_cap.deinit();

    // Simulate emission
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const emission = TraceEmission{
        .point = .lm_head,
        .layer = trace.TraceEmission.NO_LAYER,
        .token = 0,
        .position = 0,
        .backend = .cpu,
        .tensor = .{
            .ptr = @ptrCast(&data),
            .dtype = .f32,
            .shape = .{ 4, 0, 0, 0 },
            .ndim = 1,
        },
        .timestamp_ns = 0,
        .kernel_name = std.mem.zeroes([48]u8),
    };

    verify_cap.handleEmission(emission);

    // Check that recorder captured the stats
    try std.testing.expectEqual(@as(usize, 1), recorder.stats_records.items.len);
}

test "VerifyCapture verification mode detects divergence" {
    const allocator = std.testing.allocator;

    // Create reference with golden stats
    const golden_stats = TensorStats{
        .count = 4,
        .min = 1.0,
        .max = 4.0,
        .sum = 10.0,
        .sum_sq = 30.0,
        .nan_count = 0,
        .inf_count = 0,
    };

    const ref = reference_mod.ReferenceData{
        .model_name = "test",
        .seed = 42,
        .temperature = 1.0,
        .max_tokens = 1,
        .token_transcript = &[_]u32{100},
        .stats_records = &[_]reference_mod.StatsRecord{.{
            .token_idx = 0,
            .layer = trace.TraceEmission.NO_LAYER,
            .point = .lm_head,
            .position = 0,
            .stats = golden_stats,
        }},
        .allocator = allocator,
    };

    var verifier = ReferenceVerifier.init(allocator, &ref, 1e-3);

    var verify_cap = VerifyCapture.initVerification(allocator, &verifier, null);
    defer verify_cap.deinit();

    // Emit with different stats - should trigger divergence detection
    const data = [_]f32{ 1.0, 2.0, 3.0, 10.0 }; // Different values
    const emission = TraceEmission{
        .point = .lm_head,
        .layer = trace.TraceEmission.NO_LAYER,
        .token = 0,
        .position = 0,
        .backend = .cpu,
        .tensor = .{
            .ptr = @ptrCast(&data),
            .dtype = .f32,
            .shape = .{ 4, 0, 0, 0 },
            .ndim = 1,
        },
        .timestamp_ns = 0,
        .kernel_name = std.mem.zeroes([48]u8),
    };

    verify_cap.handleEmission(emission);

    // Verifier should have detected divergence
    try std.testing.expect(verifier.has_diverged);
    try std.testing.expect(verifier.divergence_point != null);
}

test "VerifyCapture verification mode compares host tensors from non-CPU backends" {
    const allocator = std.testing.allocator;

    const golden_stats = TensorStats{
        .count = 4,
        .min = 1.0,
        .max = 4.0,
        .sum = 10.0,
        .sum_sq = 30.0,
        .nan_count = 0,
        .inf_count = 0,
    };

    const ref = reference_mod.ReferenceData{
        .model_name = "test",
        .seed = 42,
        .temperature = 1.0,
        .max_tokens = 1,
        .token_transcript = &[_]u32{100},
        .stats_records = &[_]reference_mod.StatsRecord{.{
            .token_idx = 0,
            .layer = trace.TraceEmission.NO_LAYER,
            .point = .lm_head,
            .position = 0,
            .stats = golden_stats,
        }},
        .allocator = allocator,
    };

    var verifier = ReferenceVerifier.init(allocator, &ref, 1e-3);
    var verify_cap = VerifyCapture.initVerification(allocator, &verifier, null);
    defer verify_cap.deinit();

    const data = [_]f32{ 1.0, 2.0, 3.0, 10.0 };
    const emission = TraceEmission{
        .point = .lm_head,
        .layer = trace.TraceEmission.NO_LAYER,
        .token = 0,
        .position = 0,
        .backend = .cuda,
        .tensor = .{
            .ptr = @ptrCast(&data),
            .dtype = .f32,
            .shape = .{ 4, 0, 0, 0 },
            .ndim = 1,
        },
        .timestamp_ns = 0,
        .kernel_name = std.mem.zeroes([48]u8),
    };

    verify_cap.handleEmission(emission);

    try std.testing.expect(verifier.has_diverged);
    try std.testing.expect(verifier.divergence_point != null);
}

test "VerifyCapture ignores non-verification trace points" {
    const allocator = std.testing.allocator;

    const golden_stats = TensorStats{
        .count = 4,
        .min = 1.0,
        .max = 4.0,
        .sum = 10.0,
        .sum_sq = 30.0,
        .nan_count = 0,
        .inf_count = 0,
    };

    const ref = reference_mod.ReferenceData{
        .model_name = "test",
        .seed = 42,
        .temperature = 1.0,
        .max_tokens = 1,
        .token_transcript = &[_]u32{100},
        .stats_records = &[_]reference_mod.StatsRecord{.{
            .token_idx = 0,
            .layer = trace.TraceEmission.NO_LAYER,
            .point = .lm_head,
            .position = 0,
            .stats = golden_stats,
        }},
        .allocator = allocator,
    };

    var verifier = ReferenceVerifier.init(allocator, &ref, 1e-3);
    var verify_cap = VerifyCapture.initVerification(allocator, &verifier, null);
    defer verify_cap.deinit();

    const data = [_]f32{ 1.0, 2.0, 3.0, 10.0 };
    const emission = TraceEmission{
        .point = .layer_ffn_norm,
        .layer = 0,
        .token = 0,
        .position = 14,
        .backend = .cpu,
        .tensor = .{
            .ptr = @ptrCast(&data),
            .dtype = .f32,
            .shape = .{ 4, 0, 0, 0 },
            .ndim = 1,
        },
        .timestamp_ns = 0,
        .kernel_name = std.mem.zeroes([48]u8),
    };

    verify_cap.handleEmission(emission);

    try std.testing.expect(!verifier.has_diverged);
}

test "VerifyCapture token_select u32 advances verifier and skips stats comparison" {
    const allocator = std.testing.allocator;

    const ref = reference_mod.ReferenceData{
        .model_name = "test",
        .seed = 42,
        .temperature = 1.0,
        .max_tokens = 1,
        .token_transcript = &[_]u32{123},
        .stats_records = &.{},
        .allocator = allocator,
    };

    var verifier = ReferenceVerifier.init(allocator, &ref, 1e-3);
    var verify_cap = VerifyCapture.initVerification(allocator, &verifier, null);
    defer verify_cap.deinit();

    const token_id: u32 = 123;
    const emission = TraceEmission{
        .point = .token_select,
        .layer = trace.TraceEmission.NO_LAYER,
        .token = 0,
        .position = 14,
        .backend = .metal,
        .tensor = .{
            .ptr = @ptrCast(std.mem.asBytes(&token_id).ptr),
            .dtype = .u32,
            .shape = .{ 1, 0, 0, 0 },
            .ndim = 1,
        },
        .timestamp_ns = 0,
        .kernel_name = std.mem.zeroes([48]u8),
    };

    verify_cap.handleEmission(emission);

    try std.testing.expectEqual(@as(u32, 1), verifier.token_idx);
    try std.testing.expect(!verifier.has_diverged);
}

test "VerifyCapture token_select f32 does not advance verifier" {
    const allocator = std.testing.allocator;

    const ref = reference_mod.ReferenceData{
        .model_name = "test",
        .seed = 42,
        .temperature = 1.0,
        .max_tokens = 1,
        .token_transcript = &[_]u32{123},
        .stats_records = &.{},
        .allocator = allocator,
    };

    var verifier = ReferenceVerifier.init(allocator, &ref, 1e-3);
    var verify_cap = VerifyCapture.initVerification(allocator, &verifier, null);
    defer verify_cap.deinit();

    const token_id_f32 = [_]f32{123};
    const emission = TraceEmission{
        .point = .token_select,
        .layer = trace.TraceEmission.NO_LAYER,
        .token = 0,
        .position = 14,
        .backend = .cpu,
        .tensor = .{
            .ptr = @ptrCast(token_id_f32[0..].ptr),
            .dtype = .f32,
            .shape = .{ 1, 0, 0, 0 },
            .ndim = 1,
        },
        .timestamp_ns = 0,
        .kernel_name = std.mem.zeroes([48]u8),
    };

    verify_cap.handleEmission(emission);

    try std.testing.expectEqual(@as(u32, 0), verifier.token_idx);
    try std.testing.expect(!verifier.has_diverged);
}

test "VerifyCapture recording token_select u32 updates transcript without stats record" {
    const allocator = std.testing.allocator;

    var recorder = try ReferenceRecorder.init(allocator, "test_model", 42, 1.0, 10);
    defer recorder.deinit();

    var verify_cap = VerifyCapture.initRecording(allocator, &recorder);
    defer verify_cap.deinit();

    const token_id: u32 = 321;
    const emission = TraceEmission{
        .point = .token_select,
        .layer = trace.TraceEmission.NO_LAYER,
        .token = 0,
        .position = 0,
        .backend = .metal,
        .tensor = .{
            .ptr = @ptrCast(std.mem.asBytes(&token_id).ptr),
            .dtype = .u32,
            .shape = .{ 1, 0, 0, 0 },
            .ndim = 1,
        },
        .timestamp_ns = 0,
        .kernel_name = std.mem.zeroes([48]u8),
    };

    verify_cap.handleEmission(emission);

    try std.testing.expectEqual(@as(usize, 1), recorder.token_transcript.items.len);
    try std.testing.expectEqual(@as(u32, 321), recorder.token_transcript.items[0]);
    try std.testing.expectEqual(@as(u32, 1), recorder.current_token_idx);
    try std.testing.expectEqual(@as(usize, 0), recorder.stats_records.items.len);
}
