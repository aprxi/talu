//! Verification System Integration
//!
//! Integrates reference recording/verification with the trace capture system.
//! Handles:
//! - Recording mode: Captures stats and feeds them to ReferenceRecorder
//! - Verification mode: Checks emissions against reference, triggers panic dump on divergence

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const trace = @import("trace.zig");
const capture_mod = @import("capture.zig");
const stats_mod = @import("stats.zig");
const reference_mod = @import("reference.zig");
const teacher_forcing = @import("teacher_forcing.zig");
const dump_capture_mod = @import("dump/capture.zig");
const dump_npz_mod = @import("dump/npz.zig");
const core_dtype = @import("dtype_pkg");
const handler_slot_mod = @import("handler_slot.zig");
const xray_bridge_enabled: bool = if (@hasDecl(build_options, "xray_bridge")) build_options.xray_bridge else true;

const TraceEmission = trace.TraceEmission;
const TensorStats = stats_mod.TensorStats;
const TraceCapture = capture_mod.TraceCapture;
const TraceCaptureConfig = capture_mod.TraceCaptureConfig;
const ReferenceRecorder = reference_mod.ReferenceRecorder;
const ReferenceVerifier = reference_mod.ReferenceVerifier;

var ignore_token_parity_enabled: std.atomic.Value(bool) = std.atomic.Value(bool).init(false);
var token_only_enabled: std.atomic.Value(bool) = std.atomic.Value(bool).init(false);
var verification_full_capture_enabled: std.atomic.Value(bool) = std.atomic.Value(bool).init(false);
var verification_point_mask_override: std.atomic.Value(u64) = std.atomic.Value(u64).init(0);
var verification_exact_filter_enabled: std.atomic.Value(bool) = std.atomic.Value(bool).init(false);
var verification_exact_filter_point: std.atomic.Value(u8) = std.atomic.Value(u8).init(0);
var verification_exact_filter_layer: std.atomic.Value(u16) = std.atomic.Value(u16).init(0);
var verification_exact_filter_position: std.atomic.Value(u32) = std.atomic.Value(u32).init(0);
var stop_requested: std.atomic.Value(bool) = std.atomic.Value(bool).init(false);

/// XRAY VERIFY CONTROL:
/// These controls are observability/transcript-only switches.
/// They MUST NOT alter backend route selection, fusion policy, or kernel math.
pub fn setIgnoreTokenParityOverride(enabled: bool) void {
    ignore_token_parity_enabled.store(enabled, .release);
}

pub fn clearIgnoreTokenParityOverride() void {
    ignore_token_parity_enabled.store(false, .release);
}

pub fn setTokenOnlyOverride(enabled: bool) void {
    token_only_enabled.store(enabled, .release);
}

pub fn clearTokenOnlyOverride() void {
    token_only_enabled.store(false, .release);
}

pub fn setVerificationFullCaptureOverride(enabled: bool) void {
    verification_full_capture_enabled.store(enabled, .release);
}

pub fn clearVerificationFullCaptureOverride() void {
    verification_full_capture_enabled.store(false, .release);
}

pub fn setVerificationPointMaskOverride(mask: u64) void {
    verification_point_mask_override.store(mask, .release);
}

pub fn clearVerificationPointMaskOverride() void {
    verification_point_mask_override.store(0, .release);
}

pub fn setVerificationExactEmissionOverride(
    point: trace.TracePoint,
    layer: u16,
    position: u32,
) void {
    verification_exact_filter_point.store(@intFromEnum(point), .release);
    verification_exact_filter_layer.store(layer, .release);
    verification_exact_filter_position.store(position, .release);
    verification_exact_filter_enabled.store(true, .release);
}

pub fn clearVerificationExactEmissionOverride() void {
    verification_exact_filter_enabled.store(false, .release);
}

pub fn isStopRequested() bool {
    return stop_requested.load(.acquire);
}

/// Mode for verification capture.
///
/// XRAY VERIFY CONTRACT:
/// - Verify is capture/compare orchestration only.
/// - Verify controls transcript handling (teacher forcing / token parity checks).
/// - Verify MUST NOT control backend route selection, fusion policy, or kernel math.
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

    /// Full-tensor recording sidecar.
    recording_full_capture: ?dump_capture_mod.Capture,
    recording_full_index: usize,

    /// Panic dump path (for full tensor dumps on divergence)
    panic_dump_dir: ?[]const u8,

    /// Has a panic dump been triggered?
    panic_triggered: bool,

    fn defaultVerificationPointSet() capture_mod.TracePointSet {
        // Verify only points that every backend can emit as real host-readable
        // tensors on the production execution path. Marker-only points and
        // backend-private internals do not belong in the default parity contract.
        return .{
            .embed = true,
            .layer_input = true,
            .layer_attn_norm = true,
            .attn_q = true,
            .attn_k = true,
            .attn_q_proj_raw = true,
            .attn_k_proj_raw = true,
            .attn_q_norm = true,
            .attn_k_norm = true,
            .attn_q_rope = true,
            .attn_k_rope = true,
            .attn_qk = true,
            .attn_weights = true,
            .attn_out = true,
            .gdelta_in_proj = true,
            .gdelta_conv = true,
            .gdelta_ssm = true,
            .gdelta_norm = true,
            .block_out = true,
            .final_norm = true,
            .lm_head = true,
            .token_select = true,
            .gdelta_out = true,
            .gdelta_state_conv = true,
            .gdelta_state_ssm = true,
        };
    }

    fn configuredVerificationPointSet() capture_mod.TracePointSet {
        const override_mask = verification_point_mask_override.load(.acquire);
        if (override_mask != 0) {
            return capture_mod.TracePointSet.fromBuiltinMask(override_mask);
        }
        return defaultVerificationPointSet();
    }

    fn verificationFullCaptureEnabled() bool {
        return verification_full_capture_enabled.load(.acquire);
    }

    fn configuredExactEmissionFilter() ?trace.ExactEmissionFilter {
        if (!verification_exact_filter_enabled.load(.acquire)) return null;
        return .{
            .point = @enumFromInt(verification_exact_filter_point.load(.acquire)),
            .layer = verification_exact_filter_layer.load(.acquire),
            .position = verification_exact_filter_position.load(.acquire),
        };
    }

    fn effectivePointSet(self: *const VerifyCapture) capture_mod.TracePointSet {
        if (self.mode == .verify and tokenOnlyVerificationEnabled() and !teacher_forcing.isEnabled()) {
            return .{ .token_select = true };
        }
        return self.capture.config.points;
    }

    pub fn initRecording(
        allocator: std.mem.Allocator,
        recorder: *ReferenceRecorder,
    ) VerifyCapture {
        const config = TraceCaptureConfig{
            .mode = .stats,
            .points = configuredVerificationPointSet(),
            .allow_non_cpu_host_data = true,
        };

        return .{
            .allocator = allocator,
            .capture = TraceCapture.init(allocator, config),
            .mode = .record,
            .recorder = recorder,
            .verifier = null,
            .recording_full_capture = blk: {
                var capture = dump_capture_mod.Capture.init(allocator);
                capture.enable();
                break :blk capture;
            },
            .recording_full_index = 0,
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
            .points = configuredVerificationPointSet(),
            .allow_non_cpu_host_data = true,
        };

        return .{
            .allocator = allocator,
            .capture = TraceCapture.init(allocator, config),
            .mode = .verify,
            .recorder = null,
            .verifier = verifier,
            .recording_full_capture = if (verificationFullCaptureEnabled()) blk: {
                var capture = dump_capture_mod.Capture.init(allocator);
                capture.enable();
                break :blk capture;
            } else null,
            .recording_full_index = 0,
            .panic_dump_dir = panic_dump_dir,
            .panic_triggered = false,
        };
    }

    pub fn deinit(self: *VerifyCapture) void {
        if (self.recording_full_capture) |*recording_full_capture| {
            recording_full_capture.disable();
            recording_full_capture.deinit();
        }
        self.capture.deinit();
    }

    /// Persist full host-readable tensor values captured for this verify capture to NPZ.
    /// In record mode this is the CPU golden sidecar. In verify mode this is the
    /// backend candidate sidecar (typically phase-2 teacher-forced run).
    pub fn saveFullNpz(self: *VerifyCapture, path: []const u8) !void {
        const full_capture = &(self.recording_full_capture orelse return error.InvalidMode);
        var writer = dump_npz_mod.NpzWriter.init(self.allocator);
        defer writer.deinit();
        try writer.addAll(full_capture);
        try writer.write(path);
    }

    fn isTranscriptTokenSelect(emission: TraceEmission) bool {
        return emission.point == .token_select and
            emission.tensor.dtype == .u32 and
            emission.tensor.elementCount() == 1;
    }

    fn ignoreTokenParity() bool {
        // Transcript policy only. This flag is permitted because it changes
        // verifier token-check behavior, not backend compute behavior.
        return ignore_token_parity_enabled.load(.acquire);
    }

    fn tokenOnlyVerificationEnabled() bool {
        // Capture policy only. This must never be used as a backend execution
        // switch (route/fusion/kernel selection).
        return token_only_enabled.load(.acquire);
    }

    fn kernelNameSlice(name: [48]u8) []const u8 {
        const len = std.mem.indexOfScalar(u8, &name, 0) orelse name.len;
        return name[0..len];
    }

    fn isHostReadableEmission(emission: TraceEmission) bool {
        if (emission.backend == .cpu) return true;
        const kernel_name = kernelNameSlice(emission.kernel_name);
        return std.mem.endsWith(u8, kernel_name, "_host");
    }

    fn logVerificationDivergence(verifier: *const ReferenceVerifier, err: anyerror) void {
        if (std.posix.getenv("TALU_PARITY_QUIET")) |raw| {
            const enabled = !(raw.len == 0 or (raw.len == 1 and raw[0] == '0'));
            if (enabled) return;
        }
        if (comptime builtin.is_test) {
            std.log.warn("DIVERGENCE DETECTED: {}", .{err});
        } else {
            std.log.err("DIVERGENCE DETECTED: {}", .{err});
        }
        if (verifier.divergence_point) |div| {
            const msg_len = std.mem.indexOfScalar(u8, &div.message, 0) orelse div.message.len;
            if (comptime builtin.is_test) {
                std.log.warn("{s}", .{div.message[0..msg_len]});
            } else {
                std.log.err("{s}", .{div.message[0..msg_len]});
            }
        }
    }

    /// Handle a trace emission
    pub fn handleEmission(self: *VerifyCapture, emission: TraceEmission) void {
        if (self.mode == .verify) {
            if (self.verifier) |ver| {
                if (ver.has_diverged) return;
            }
        }

        const is_token_select = emission.point == .token_select;
        const is_transcript_token_select = isTranscriptTokenSelect(emission);

        self.recordFullEmission(emission);

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
                        // In localization passes, token_select is used only to advance token
                        // index so numeric checkpoints can still be compared after free-run
                        // token mismatch.
                        if (!ignoreTokenParity() and !teacher_forcing.isEnabled()) {
                            ver.checkToken(token_id_ptr.*) catch |err| {
                                stop_requested.store(true, .release);
                                logVerificationDivergence(ver, err);
                                return;
                            };
                        }
                        ver.nextToken();
                    }
                }
            },
        }
        if (is_token_select) return;
        if (!self.capture.config.points.contains(emission.point)) return;
        if (self.mode == .verify and tokenOnlyVerificationEnabled() and !teacher_forcing.isEnabled()) return;
        if (!isHostReadableEmission(emission)) return;

        // Xray emitters must only publish host-accessible tensors. This keeps
        // verification backend-agnostic and avoids duplicating device-specific
        // stats paths here.
        const tensor_stats = stats_mod.compute(emission.tensor);
        if (emission.point == .attn_q_norm or emission.point == .attn_k_norm or emission.point == .attn_q_rope or emission.point == .attn_k_rope or emission.point == .attn_qk) {
            std.log.err("XRAY probe point={s} layer={} pos={} token={}", .{
                emission.point.name(),
                emission.layer,
                emission.position,
                emission.token,
            });
        }

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
                        stop_requested.store(true, .release);
                        logVerificationDivergence(ver, err);

                        if (ver.divergence_point) |div| {
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
        const quiet = blk: {
            if (std.posix.getenv("TALU_PARITY_QUIET")) |raw| {
                break :blk !(raw.len == 0 or (raw.len == 1 and raw[0] == '0'));
            }
            break :blk false;
        };
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
        if (!quiet) {
            std.log.warn("Would write panic dump to: {s}", .{full_path});
            std.log.warn("Expected RMS: {d:.6}, Actual RMS: {d:.6}", .{
                divergence.expected.rms(),
                actual_stats.rms(),
            });
        }
    }

    fn recordFullEmission(self: *VerifyCapture, emission: TraceEmission) void {
        const full_capture = &(self.recording_full_capture orelse return);
        if (!self.capture.config.points.contains(emission.point)) return;
        if (self.mode == .verify and tokenOnlyVerificationEnabled() and !teacher_forcing.isEnabled()) {
            return;
        }
        if (!isHostReadableEmission(emission)) return;
        const dtype = switch (emission.tensor.dtype) {
            .f32 => core_dtype.DType.f32,
            .f64 => core_dtype.DType.f64,
            .i32 => core_dtype.DType.i32,
            .i64 => core_dtype.DType.i64,
            .f16 => core_dtype.DType.f16,
            .bf16 => core_dtype.DType.bf16,
            .i8 => core_dtype.DType.i8,
            .i16 => core_dtype.DType.i16,
            .u8 => core_dtype.DType.u8,
            .u16 => core_dtype.DType.u16,
            .u32 => core_dtype.DType.u32,
            .u64 => core_dtype.DType.u64,
            .grouped_affine_u4 => core_dtype.DType.grouped_affine_u4,
            .grouped_affine_u8 => core_dtype.DType.grouped_affine_u8,
        };
        var shape_usize: [4]usize = .{ 0, 0, 0, 0 };
        for (0..emission.tensor.ndim) |dim| {
            shape_usize[dim] = @intCast(emission.tensor.shape[dim]);
        }

        const layer_label: []const u8 = if (emission.layer == trace.TraceEmission.NO_LAYER)
            "global"
        else
            "layer";
        const layer_idx: u16 = if (emission.layer == trace.TraceEmission.NO_LAYER)
            0
        else
            emission.layer;
        const logical_token_idx: u32 = switch (self.mode) {
            .record => if (self.recorder) |rec| rec.current_token_idx else emission.token,
            .verify => if (self.verifier) |ver| ver.token_idx else emission.token,
        };

        var name_buf: [192]u8 = undefined;
        const tensor_name = std.fmt.bufPrint(
            &name_buf,
            "{d}_tok{d}_pos{d}_{s}_{d}_{s}",
            .{
                self.recording_full_index,
                logical_token_idx,
                emission.position,
                layer_label,
                layer_idx,
                emission.point.name(),
            },
        ) catch |err| {
            std.log.warn("failed to build full tensor sidecar name: {}", .{err});
            return;
        };

        full_capture.record(
            tensor_name,
            emission.tensor.ptr,
            dtype,
            shape_usize,
            emission.tensor.ndim,
        ) catch |err| {
            std.log.warn("failed to capture full tensor sidecar emission: {}", .{err});
        };
        self.recording_full_index += 1;
    }
};

/// Global verify capture for handler integration.
///
/// Verification can emit from backend-managed worker threads while Rust/CLI
/// enables/disables capture around a generation run. Disable must therefore
/// wait for any in-flight callback before verifier/capture teardown proceeds.
var global_verify_capture_slot: handler_slot_mod.HandlerSlot(VerifyCapture) = .{};

fn globalVerifyHandler(emission: TraceEmission) void {
    var locked = global_verify_capture_slot.acquire();
    defer locked.release();
    if (locked.ptr) |cap| {
        cap.handleEmission(emission);
    }
}

pub fn enableVerifyCapture(cap: *VerifyCapture) void {
    global_verify_capture_slot.set(cap);
    stop_requested.store(false, .release);
    trace.setActiveBuiltInPointMask(cap.effectivePointSet().builtinMask());
    trace.setActiveExactEmissionFilter(VerifyCapture.configuredExactEmissionFilter());
    trace.setHandler(&globalVerifyHandler);
}

pub fn disableVerifyCapture() void {
    global_verify_capture_slot.set(null);
    stop_requested.store(false, .release);
    trace.setHandler(null);
    trace.setActiveExactEmissionFilter(null);
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

test "VerifyCapture recording mode writes full tensor NPZ sidecar" {
    const allocator = std.testing.allocator;

    var recorder = try ReferenceRecorder.init(allocator, "test_model", 42, 1.0, 10);
    defer recorder.deinit();

    var verify_cap = VerifyCapture.initRecording(allocator, &recorder);
    defer verify_cap.deinit();

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

    const full_capture = &(verify_cap.recording_full_capture orelse unreachable);
    try std.testing.expectEqual(@as(usize, 1), full_capture.tensors.items.len);

    const path = try std.fs.path.join(allocator, &[_][]const u8{
        "/tmp",
        "talu_verify_capture_recording_full_sidecar_test.npz",
    });
    defer allocator.free(path);
    std.fs.deleteFileAbsolute(path) catch {};
    defer std.fs.deleteFileAbsolute(path) catch {};

    try verify_cap.saveFullNpz(path);

    const file = try std.fs.openFileAbsolute(path, .{});
    defer file.close();
    const stat = try file.stat();
    try std.testing.expect(stat.size > 0);
}

test "VerifyCapture recording mode honors point-mask override" {
    const allocator = std.testing.allocator;
    const point_set = capture_mod.TracePointSet{ .token_select = true };

    setVerificationPointMaskOverride(point_set.builtinMask());
    defer clearVerificationPointMaskOverride();

    var recorder = try ReferenceRecorder.init(allocator, "test_model", 42, 1.0, 10);
    defer recorder.deinit();

    var verify_cap = VerifyCapture.initRecording(allocator, &recorder);
    defer verify_cap.deinit();

    try std.testing.expect(verify_cap.capture.config.points.token_select);
    try std.testing.expect(!verify_cap.capture.config.points.lm_head);
    try std.testing.expect(!verify_cap.capture.config.points.gdelta_in_proj);
}

test "VerifyCapture skips non-host Metal emissions" {
    const allocator = std.testing.allocator;

    var recorder = try ReferenceRecorder.init(allocator, "test_model", 42, 1.0, 10);
    defer recorder.deinit();

    var verify_cap = VerifyCapture.initRecording(allocator, &recorder);
    defer verify_cap.deinit();

    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var kernel_name = std.mem.zeroes([48]u8);
    @memcpy(kernel_name[0.."metal_test_kernel".len], "metal_test_kernel");
    const emission = TraceEmission{
        .point = .lm_head,
        .layer = trace.TraceEmission.NO_LAYER,
        .token = 0,
        .position = 0,
        .backend = .metal,
        .tensor = .{
            .ptr = @ptrCast(&data),
            .dtype = .f32,
            .shape = .{ 4, 0, 0, 0 },
            .ndim = 1,
        },
        .timestamp_ns = 0,
        .kernel_name = kernel_name,
    };

    verify_cap.handleEmission(emission);

    try std.testing.expectEqual(@as(usize, 0), recorder.stats_records.items.len);
    const full_capture = &(verify_cap.recording_full_capture orelse unreachable);
    try std.testing.expectEqual(@as(usize, 0), full_capture.tensors.items.len);
}

test "VerifyCapture accepts host-tagged Metal emissions" {
    const allocator = std.testing.allocator;

    var recorder = try ReferenceRecorder.init(allocator, "test_model", 42, 1.0, 10);
    defer recorder.deinit();

    var verify_cap = VerifyCapture.initRecording(allocator, &recorder);
    defer verify_cap.deinit();

    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var kernel_name = std.mem.zeroes([48]u8);
    @memcpy(kernel_name[0.."metal_test_kernel_host".len], "metal_test_kernel_host");
    const emission = TraceEmission{
        .point = .lm_head,
        .layer = trace.TraceEmission.NO_LAYER,
        .token = 0,
        .position = 0,
        .backend = .metal,
        .tensor = .{
            .ptr = @ptrCast(&data),
            .dtype = .f32,
            .shape = .{ 4, 0, 0, 0 },
            .ndim = 1,
        },
        .timestamp_ns = 0,
        .kernel_name = kernel_name,
    };

    verify_cap.handleEmission(emission);

    try std.testing.expectEqual(@as(usize, 1), recorder.stats_records.items.len);
    const full_capture = &(verify_cap.recording_full_capture orelse unreachable);
    try std.testing.expectEqual(@as(usize, 1), full_capture.tensors.items.len);
}

test "VerifyCapture full sidecar names use logical token index progression" {
    const allocator = std.testing.allocator;

    var recorder = try ReferenceRecorder.init(allocator, "test_model", 42, 1.0, 10);
    defer recorder.deinit();

    var verify_cap = VerifyCapture.initRecording(allocator, &recorder);
    defer verify_cap.deinit();

    const selected_token: u32 = 123;
    const token_select = TraceEmission{
        .point = .token_select,
        .layer = trace.TraceEmission.NO_LAYER,
        .token = 0, // backend slot token, should not drive sidecar token naming
        .position = 10,
        .backend = .cpu,
        .tensor = .{
            .ptr = @ptrCast(&selected_token),
            .dtype = .u32,
            .shape = .{ 1, 0, 0, 0 },
            .ndim = 1,
        },
        .timestamp_ns = 0,
        .kernel_name = std.mem.zeroes([48]u8),
    };
    verify_cap.handleEmission(token_select);

    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const layer_emission = TraceEmission{
        .point = .block_out,
        .layer = 0,
        .token = 0, // still backend slot token
        .position = 11,
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
    verify_cap.handleEmission(layer_emission);

    const full_capture = &(verify_cap.recording_full_capture orelse unreachable);
    try std.testing.expectEqual(@as(usize, 2), full_capture.tensors.items.len);
    try std.testing.expect(std.mem.indexOf(u8, full_capture.tensors.items[0].name, "_tok0_") != null);
    try std.testing.expect(std.mem.indexOf(u8, full_capture.tensors.items[1].name, "_tok1_") != null);
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

    var verifier = ReferenceVerifier.init(allocator, &ref, 1e-3, 1e-3);

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

    var verifier = ReferenceVerifier.init(allocator, &ref, 1e-3, 1e-3);
    var verify_cap = VerifyCapture.initVerification(allocator, &verifier, null);
    defer verify_cap.deinit();

    const data = [_]f32{ 1.0, 2.0, 3.0, 10.0 };
    var kernel_name = std.mem.zeroes([48]u8);
    @memcpy(kernel_name[0.."cuda_lm_head_host".len], "cuda_lm_head_host");
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
        .kernel_name = kernel_name,
    };

    verify_cap.handleEmission(emission);

    try std.testing.expect(verifier.has_diverged);
    try std.testing.expect(verifier.divergence_point != null);
}

test "VerifyCapture verification mode disables full sidecar capture by default" {
    const allocator = std.testing.allocator;

    const ref = reference_mod.ReferenceData{
        .model_name = "test",
        .seed = 42,
        .temperature = 1.0,
        .max_tokens = 1,
        .token_transcript = &[_]u32{100},
        .stats_records = &.{},
        .allocator = allocator,
    };

    var verifier = ReferenceVerifier.init(allocator, &ref, 1e-3, 1e-3);
    var verify_cap = VerifyCapture.initVerification(allocator, &verifier, null);
    defer verify_cap.deinit();

    try std.testing.expect(verify_cap.recording_full_capture == null);
}

test "VerifyCapture verification mode enables full sidecar capture only when requested" {
    const allocator = std.testing.allocator;

    const ref = reference_mod.ReferenceData{
        .model_name = "test",
        .seed = 42,
        .temperature = 1.0,
        .max_tokens = 1,
        .token_transcript = &[_]u32{100},
        .stats_records = &.{},
        .allocator = allocator,
    };

    setVerificationFullCaptureOverride(true);
    defer clearVerificationFullCaptureOverride();

    var verifier = ReferenceVerifier.init(allocator, &ref, 1e-3, 1e-3);
    var verify_cap = VerifyCapture.initVerification(allocator, &verifier, null);
    defer verify_cap.deinit();

    try std.testing.expect(verify_cap.recording_full_capture != null);
}

test "VerifyCapture token-only phase narrows effective point set to token_select" {
    const allocator = std.testing.allocator;

    const ref = reference_mod.ReferenceData{
        .model_name = "test",
        .seed = 42,
        .temperature = 1.0,
        .max_tokens = 1,
        .token_transcript = &[_]u32{100},
        .stats_records = &.{},
        .allocator = allocator,
    };

    setTokenOnlyOverride(true);
    defer clearTokenOnlyOverride();

    var verifier = ReferenceVerifier.init(allocator, &ref, 1e-3, 1e-3);
    var verify_cap = VerifyCapture.initVerification(allocator, &verifier, null);
    defer verify_cap.deinit();

    const points = verify_cap.effectivePointSet();
    try std.testing.expect(points.contains(.token_select));
    try std.testing.expect(!points.contains(.layer_attn_norm));
    try std.testing.expect(!points.contains(.gdelta_out));
    try std.testing.expect(!points.contains(.lm_head));
}

test "VerifyCapture ignores custom trace points outside verification set" {
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

    var verifier = ReferenceVerifier.init(allocator, &ref, 1e-3, 1e-3);
    var verify_cap = VerifyCapture.initVerification(allocator, &verifier, null);
    defer verify_cap.deinit();

    const data = [_]f32{ 1.0, 2.0, 3.0, 10.0 };
    const emission = TraceEmission{
        .point = @enumFromInt(200),
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

    var verifier = ReferenceVerifier.init(allocator, &ref, 1e-3, 1e-3);
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

test "VerifyCapture token_select u32 ignores token parity when teacher forcing is enabled" {
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

    var verifier = ReferenceVerifier.init(allocator, &ref, 1e-3, 1e-3);
    var verify_cap = VerifyCapture.initVerification(allocator, &verifier, null);
    defer verify_cap.deinit();

    const Provider = struct {
        fn getNext(_: ?*anyopaque) ?u32 {
            return 123;
        }
    };
    teacher_forcing.enable(&Provider.getNext, null);
    defer teacher_forcing.disable();

    const mismatched_token_id: u32 = 999;
    const emission = TraceEmission{
        .point = .token_select,
        .layer = trace.TraceEmission.NO_LAYER,
        .token = 0,
        .position = 14,
        .backend = .metal,
        .tensor = .{
            .ptr = @ptrCast(std.mem.asBytes(&mismatched_token_id).ptr),
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

    var verifier = ReferenceVerifier.init(allocator, &ref, 1e-3, 1e-3);
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

test "VerifyCapture verification point set is complete" {
    const points = VerifyCapture.defaultVerificationPointSet();
    try std.testing.expect(points.contains(.embed));
    try std.testing.expect(points.contains(.layer_input));
    try std.testing.expect(points.contains(.layer_attn_norm));
    try std.testing.expect(points.contains(.attn_out));
    try std.testing.expect(points.contains(.block_out));
    try std.testing.expect(points.contains(.final_norm));
    try std.testing.expect(points.contains(.lm_head));
    try std.testing.expect(!points.contains(.logits_ready));
    try std.testing.expect(points.contains(.token_select));
    try std.testing.expect(points.contains(.gdelta_in_proj));
    try std.testing.expect(points.contains(.gdelta_conv));
    try std.testing.expect(points.contains(.gdelta_ssm));
    try std.testing.expect(points.contains(.gdelta_norm));
    try std.testing.expect(points.contains(.gdelta_out));
    try std.testing.expect(!points.contains(.layer_ffn_norm));
    try std.testing.expect(!points.contains(.ffn_act));
    try std.testing.expect(!points.contains(.ffn_act_map));
    try std.testing.expect(!points.contains(.ffn_act_mix));
    try std.testing.expect(!points.contains(.conv_out_proj));
    try std.testing.expect(!points.contains(.mamba_out));
    try std.testing.expect(!points.contains(.logits_scaled));
}

test "VerifyCapture configured verification point set honors override" {
    setVerificationPointMaskOverride((@as(u64, 1) << @intFromEnum(trace.TracePoint.lm_head)));
    defer clearVerificationPointMaskOverride();

    const points = VerifyCapture.configuredVerificationPointSet();
    try std.testing.expect(points.contains(.lm_head));
    try std.testing.expect(!points.contains(.gdelta_out));
    try std.testing.expect(!points.contains(.layer_attn_norm));
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

test "enableVerifyCapture routes emissions and disableVerifyCapture stops them" {
    const allocator = std.testing.allocator;

    var recorder = try ReferenceRecorder.init(allocator, "test_model", 42, 1.0, 10);
    defer recorder.deinit();

    var verify_cap = VerifyCapture.initRecording(allocator, &recorder);
    defer verify_cap.deinit();

    enableVerifyCapture(&verify_cap);
    defer disableVerifyCapture();

    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const emission = TraceEmission{
        .point = .lm_head,
        .layer = trace.TraceEmission.NO_LAYER,
        .token = 0,
        .position = 0,
        .backend = .cpu,
        .tensor = .{
            .ptr = @ptrCast(data[0..].ptr),
            .dtype = .f32,
            .shape = .{ 4, 0, 0, 0 },
            .ndim = 1,
        },
        .timestamp_ns = 0,
        .kernel_name = std.mem.zeroes([48]u8),
    };

    trace.emitFinal(
        .lm_head,
        0,
        0,
        emission.tensor.ptr,
        .f32,
        .{ 4, 0, 0, 0 },
        1,
        "unit_test_host",
    );
    if (xray_bridge_enabled) {
        try std.testing.expectEqual(@as(usize, 1), recorder.stats_records.items.len);
    } else {
        try std.testing.expectEqual(@as(usize, 0), recorder.stats_records.items.len);
    }

    disableVerifyCapture();
    trace.emitFinal(
        .lm_head,
        0,
        0,
        emission.tensor.ptr,
        .f32,
        .{ 4, 0, 0, 0 },
        1,
        "unit_test_host",
    );
    if (xray_bridge_enabled) {
        try std.testing.expectEqual(@as(usize, 1), recorder.stats_records.items.len);
    } else {
        try std.testing.expectEqual(@as(usize, 0), recorder.stats_records.items.len);
    }
}
