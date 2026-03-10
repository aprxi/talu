//! Reference File System for Cross-Backend Verification
//!
//! Records lightweight statistical fingerprints and token history during generation,
//! then verifies subsequent runs match exactly. Enables catching numerical divergence
//! between backends (CPU vs Metal vs CUDA) without massive disk I/O.
//!
//! Architecture:
//! 1. Recording Phase: Generate tokens normally, capture stats at every TracePoint
//! 2. Verification Phase: Force exact token replay (teacher forcing), compare stats on-the-fly
//! 3. Panic Dump: If divergence detected, switch to full tensor capture for that point

const std = @import("std");
const stats_mod = @import("stats.zig");
const trace = @import("trace.zig");
const capture_mod = @import("capture.zig");

const TensorStats = stats_mod.TensorStats;
const TracePoint = trace.TracePoint;

/// Statistical fingerprint for a single tensor at a specific execution point
pub const StatsRecord = struct {
    /// Token index in generation sequence
    token_idx: u32,
    /// Layer index (or NO_LAYER for non-layer points)
    layer: u16,
    /// Which trace point
    point: TracePoint,
    /// Position in sequence (for KV cache tracking)
    position: u32,
    /// The golden statistics
    stats: TensorStats,

    pub fn matches(
        self: StatsRecord,
        token_idx: u32,
        layer: u16,
        point: TracePoint,
        position: u32,
    ) bool {
        return self.token_idx == token_idx and
            self.layer == layer and
            self.point == point and
            self.position == position;
    }
};

/// Complete reference data for a generation run
pub const ReferenceData = struct {
    /// Metadata
    model_name: []const u8,
    seed: u64,
    temperature: f32,
    max_tokens: u32,

    /// The exact token sequence generated (prompt + generated tokens)
    token_transcript: []const u32,

    /// Statistical fingerprints for every trace point
    /// Sorted by (token_idx, layer, point) for binary search
    stats_records: []const StatsRecord,

    /// Allocator used for owned slices
    allocator: std.mem.Allocator,

    pub fn deinit(self: *ReferenceData) void {
        self.allocator.free(self.model_name);
        self.allocator.free(self.token_transcript);
        self.allocator.free(self.stats_records);
    }

    /// Find the golden stats for a given execution point
    pub fn findStats(
        self: *const ReferenceData,
        token_idx: u32,
        layer: u16,
        point: TracePoint,
        position: u32,
    ) ?*const StatsRecord {
        // Linear search for now (fast enough for verification)
        for (self.stats_records) |*record| {
            if (record.matches(token_idx, layer, point, position)) {
                return record;
            }
        }
        return null;
    }
};

/// Builder for recording a reference during generation
pub const ReferenceRecorder = struct {
    allocator: std.mem.Allocator,

    /// Generation metadata
    model_name: []const u8,
    seed: u64,
    temperature: f32,
    max_tokens: u32,

    /// Token history (grows as generation proceeds)
    token_transcript: std.ArrayList(u32),

    /// Stats accumulator
    stats_records: std.ArrayList(StatsRecord),

    /// Current token being generated (for tracking)
    current_token_idx: u32,

    pub fn init(
        allocator: std.mem.Allocator,
        model_name: []const u8,
        seed: u64,
        temperature: f32,
        max_tokens: u32,
    ) !ReferenceRecorder {
        return .{
            .allocator = allocator,
            .model_name = try allocator.dupe(u8, model_name),
            .seed = seed,
            .temperature = temperature,
            .max_tokens = max_tokens,
            .token_transcript = std.ArrayList(u32){},
            .stats_records = std.ArrayList(StatsRecord){},
            .current_token_idx = 0,
        };
    }

    pub fn deinit(self: *ReferenceRecorder) void {
        self.allocator.free(self.model_name);
        self.token_transcript.deinit(self.allocator);
        self.stats_records.deinit(self.allocator);
    }

    /// Record a sampled token (called by router after sampling)
    pub fn recordToken(self: *ReferenceRecorder, token_id: u32) !void {
        try self.token_transcript.append(self.allocator, token_id);
    }

    /// Advance to next token in sequence
    pub fn nextToken(self: *ReferenceRecorder) void {
        self.current_token_idx += 1;
    }

    /// Record stats from a trace emission
    pub fn recordEmission(
        self: *ReferenceRecorder,
        emission: trace.TraceEmission,
        stats: TensorStats,
    ) !void {
        try self.stats_records.append(self.allocator, .{
            .token_idx = self.current_token_idx,
            .layer = emission.layer,
            .point = emission.point,
            .position = emission.position,
            .stats = stats,
        });
    }

    /// Finalize and return reference data
    pub fn finalize(self: *ReferenceRecorder) !ReferenceData {
        return .{
            .model_name = try self.allocator.dupe(u8, self.model_name),
            .seed = self.seed,
            .temperature = self.temperature,
            .max_tokens = self.max_tokens,
            .token_transcript = try self.token_transcript.toOwnedSlice(self.allocator),
            .stats_records = try self.stats_records.toOwnedSlice(self.allocator),
            .allocator = self.allocator,
        };
    }
};

/// Verifier for checking generation against reference
pub const ReferenceVerifier = struct {
    allocator: std.mem.Allocator,
    reference: *const ReferenceData,

    /// Tolerance for floating-point comparison (epsilon)
    tolerance: f32,

    /// Current position in token transcript (for teacher forcing)
    token_idx: u32,

    /// Next reference stats record expected from verification.
    expected_record_idx: usize,

    /// Divergence detection
    has_diverged: bool,
    divergence_point: ?DivergenceInfo,

    pub const DivergenceInfo = struct {
        token_idx: u32,
        layer: u16,
        point: TracePoint,
        position: u32,
        expected: TensorStats,
        actual: TensorStats,
        message: [256]u8,
    };

    pub fn init(
        allocator: std.mem.Allocator,
        reference: *const ReferenceData,
        tolerance: f32,
    ) ReferenceVerifier {
        return .{
            .allocator = allocator,
            .reference = reference,
            .tolerance = tolerance,
            .token_idx = 0,
            .expected_record_idx = 0,
            .has_diverged = false,
            .divergence_point = null,
        };
    }

    pub fn deinit(self: *ReferenceVerifier) void {
        _ = self;
    }

    /// Get the next token to force (for teacher forcing)
    pub fn getNextToken(self: *const ReferenceVerifier) ?u32 {
        if (self.token_idx >= self.reference.token_transcript.len) {
            return null;
        }
        return self.reference.token_transcript[self.token_idx];
    }

    /// Advance to next token
    pub fn nextToken(self: *ReferenceVerifier) void {
        self.token_idx += 1;
    }

    /// Check that a sampled token matches the reference transcript at current index.
    pub fn checkToken(self: *ReferenceVerifier, actual_token_id: u32) !void {
        if (self.has_diverged) return;

        if (self.token_idx >= self.reference.token_transcript.len) {
            self.has_diverged = true;
            var msg_buf: [256]u8 = undefined;
            const msg = try std.fmt.bufPrint(
                &msg_buf,
                "Extra generated token at token={d}: actual={d}, no reference token available",
                .{ self.token_idx, actual_token_id },
            );
            self.divergence_point = .{
                .token_idx = self.token_idx,
                .layer = trace.TraceEmission.NO_LAYER,
                .point = .token_select,
                .position = self.token_idx,
                .expected = TensorStats.EMPTY,
                .actual = TensorStats.EMPTY,
                .message = std.mem.zeroes([256]u8),
            };
            @memcpy(self.divergence_point.?.message[0..msg.len], msg);
            return error.TokenDivergence;
        }

        const expected_token_id = self.reference.token_transcript[self.token_idx];
        if (expected_token_id == actual_token_id) return;

        self.has_diverged = true;
        var msg_buf: [256]u8 = undefined;
        const msg = try std.fmt.bufPrint(
            &msg_buf,
            "Token divergence at token={d}: expected={d} actual={d}",
            .{ self.token_idx, expected_token_id, actual_token_id },
        );
        self.divergence_point = .{
            .token_idx = self.token_idx,
            .layer = trace.TraceEmission.NO_LAYER,
            .point = .token_select,
            .position = self.token_idx,
            .expected = TensorStats.EMPTY,
            .actual = TensorStats.EMPTY,
            .message = std.mem.zeroes([256]u8),
        };
        @memcpy(self.divergence_point.?.message[0..msg.len], msg);
        return error.TokenDivergence;
    }

    /// Check stats from an emission against reference
    pub fn checkEmission(
        self: *ReferenceVerifier,
        emission: trace.TraceEmission,
        actual_stats: TensorStats,
    ) !void {
        // Skip if already diverged
        if (self.has_diverged) return;

        if (self.expected_record_idx >= self.reference.stats_records.len) return;

        const golden = &self.reference.stats_records[self.expected_record_idx];
        if (!golden.matches(
            self.token_idx,
            emission.layer,
            emission.point,
            emission.position,
        )) {
            // Backends may emit additional xray points or a finer-grained view.
            // Ignore unmatched emissions here, then enforce completeness at the end.
            return;
        }

        // Compare stats
        if (!self.statsMatch(golden.stats, actual_stats)) {
            self.has_diverged = true;
            var msg_buf: [256]u8 = undefined;
            const msg = try std.fmt.bufPrint(&msg_buf, "Stats divergence at token={d} layer={d} point={s}: RMS expected={d:.6} actual={d:.6}", .{ self.token_idx, emission.layer, emission.point.name(), golden.stats.rms(), actual_stats.rms() });
            self.divergence_point = .{
                .token_idx = self.token_idx,
                .layer = emission.layer,
                .point = emission.point,
                .position = emission.position,
                .expected = golden.stats,
                .actual = actual_stats,
                .message = std.mem.zeroes([256]u8),
            };
            @memcpy(self.divergence_point.?.message[0..msg.len], msg);
            return error.StatsDivergence;
        }

        self.expected_record_idx += 1;
    }

    pub fn finish(self: *ReferenceVerifier) !void {
        if (self.has_diverged) {
            if (self.divergence_point) |div| {
                if (div.point == .token_select) return error.TokenDivergence;
            }
            return error.StatsDivergence;
        }
        if (self.expected_record_idx >= self.reference.stats_records.len) return;

        const missing = self.reference.stats_records[self.expected_record_idx];
        self.has_diverged = true;
        var msg_buf: [256]u8 = undefined;
        const msg = try std.fmt.bufPrint(
            &msg_buf,
            "Missing expected stats for token={d} layer={d} point={s} pos={d}",
            .{ missing.token_idx, missing.layer, missing.point.name(), missing.position },
        );
        self.divergence_point = .{
            .token_idx = missing.token_idx,
            .layer = missing.layer,
            .point = missing.point,
            .position = missing.position,
            .expected = missing.stats,
            .actual = TensorStats.EMPTY,
            .message = std.mem.zeroes([256]u8),
        };
        @memcpy(self.divergence_point.?.message[0..msg.len], msg);
        return error.ReferenceMissing;
    }

    fn statsMatch(self: *const ReferenceVerifier, expected: TensorStats, actual: TensorStats) bool {
        // Check for anomaly mismatch
        if (expected.nan_count != actual.nan_count) return false;
        if (expected.inf_count != actual.inf_count) return false;

        // Check element count
        if (expected.count != actual.count) return false;

        // Check RMS difference
        const rms_diff = @abs(expected.rms() - actual.rms());
        const rms_rel = if (expected.rms() > 0) rms_diff / expected.rms() else rms_diff;
        if (rms_rel > self.tolerance) return false;

        // Check L2 norm difference
        const l2_diff = @abs(expected.l2Norm() - actual.l2Norm());
        const l2_rel = if (expected.l2Norm() > 0) l2_diff / expected.l2Norm() else l2_diff;
        if (l2_rel > self.tolerance) return false;

        // Check mean difference
        const mean_diff = @abs(expected.mean() - actual.mean());
        if (mean_diff > self.tolerance) return false;

        return true;
    }
};

/// JSON serialization format
/// Format: { "metadata": {...}, "tokens": [...], "stats": [...] }
pub const JsonFormat = struct {
    pub fn serialize(ref: *const ReferenceData, writer: anytype) !void {
        try writer.writeAll("{\"metadata\":{");
        try writer.print("\"model_name\":\"{s}\",", .{ref.model_name});
        try writer.print("\"seed\":{d},", .{ref.seed});
        try writer.print("\"temperature\":{d},", .{ref.temperature});
        try writer.print("\"max_tokens\":{d}", .{ref.max_tokens});
        try writer.writeAll("},\"tokens\":[");

        // Write token transcript
        for (ref.token_transcript, 0..) |token, i| {
            if (i > 0) try writer.writeAll(",");
            try writer.print("{d}", .{token});
        }

        try writer.writeAll("],\"stats\":[");

        // Write stats records
        for (ref.stats_records, 0..) |record, i| {
            if (i > 0) try writer.writeAll(",");
            try writer.writeAll("{");
            try writer.print("\"token_idx\":{d},", .{record.token_idx});
            try writer.print("\"layer\":{d},", .{record.layer});
            try writer.print("\"point\":\"{s}\",", .{record.point.name()});
            try writer.print("\"position\":{d},", .{record.position});
            try writer.writeAll("\"stats\":{");
            try writer.print("\"count\":{d},", .{record.stats.count});
            // Handle special float values (inf, -inf, nan) as JSON null
            try writer.writeAll("\"min\":");
            if (std.math.isNan(record.stats.min) or std.math.isInf(record.stats.min)) {
                try writer.writeAll("null");
            } else {
                try writer.print("{d}", .{record.stats.min});
            }
            try writer.writeAll(",\"max\":");
            if (std.math.isNan(record.stats.max) or std.math.isInf(record.stats.max)) {
                try writer.writeAll("null");
            } else {
                try writer.print("{d}", .{record.stats.max});
            }
            try writer.writeAll(",\"sum\":");
            if (std.math.isNan(record.stats.sum) or std.math.isInf(record.stats.sum)) {
                try writer.writeAll("null");
            } else {
                try writer.print("{d}", .{record.stats.sum});
            }
            try writer.writeAll(",\"sum_sq\":");
            if (std.math.isNan(record.stats.sum_sq) or std.math.isInf(record.stats.sum_sq)) {
                try writer.writeAll("null");
            } else {
                try writer.print("{d}", .{record.stats.sum_sq});
            }
            try writer.print(",\"nan_count\":{d},", .{record.stats.nan_count});
            try writer.print("\"inf_count\":{d}", .{record.stats.inf_count});
            try writer.writeAll("}}");
        }

        try writer.writeAll("]}");
    }

    pub fn writeToFile(ref: *const ReferenceData, path: []const u8) !void {
        // Serialize to memory buffer (references are typically < 10MB)
        var buffer = try std.ArrayList(u8).initCapacity(ref.allocator, 1024 * 1024); // 1MB initial
        defer buffer.deinit(ref.allocator);

        try serialize(ref, buffer.writer(ref.allocator));

        // Write to file atomically
        try std.fs.cwd().writeFile(.{ .sub_path = path, .data = buffer.items });
    }

    pub fn deserialize(allocator: std.mem.Allocator, json_str: []const u8) !ReferenceData {
        const parsed = try std.json.parseFromSlice(
            JsonReferenceData,
            allocator,
            json_str,
            .{ .allocate = .alloc_always },
        );
        defer parsed.deinit();

        const value = parsed.value;

        // Convert stats records
        var stats_records = std.ArrayList(StatsRecord){};
        errdefer stats_records.deinit(allocator);

        for (value.stats) |json_record| {
            const point = parseTracePoint(json_record.point) orelse continue;
            try stats_records.append(allocator, .{
                .token_idx = json_record.token_idx,
                .layer = json_record.layer,
                .point = point,
                .position = json_record.position,
                .stats = .{
                    .count = json_record.stats.count,
                    // Restore infinity/nan from null
                    .min = json_record.stats.min orelse std.math.inf(f32),
                    .max = json_record.stats.max orelse -std.math.inf(f32),
                    .sum = json_record.stats.sum orelse 0.0,
                    .sum_sq = json_record.stats.sum_sq orelse 0.0,
                    .nan_count = json_record.stats.nan_count,
                    .inf_count = json_record.stats.inf_count,
                },
            });
        }

        return .{
            .model_name = try allocator.dupe(u8, value.metadata.model_name),
            .seed = value.metadata.seed,
            .temperature = value.metadata.temperature,
            .max_tokens = value.metadata.max_tokens,
            .token_transcript = try allocator.dupe(u32, value.tokens),
            .stats_records = try stats_records.toOwnedSlice(allocator),
            .allocator = allocator,
        };
    }

    pub fn readFromFile(allocator: std.mem.Allocator, path: []const u8) !ReferenceData {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const content = try file.readToEndAlloc(allocator, 100 * 1024 * 1024); // 100MB max
        defer allocator.free(content);

        return try deserialize(allocator, content);
    }

    // Helper structs for JSON parsing
    const JsonReferenceData = struct {
        metadata: struct {
            model_name: []const u8,
            seed: u64,
            temperature: f32,
            max_tokens: u32,
        },
        tokens: []const u32,
        stats: []const struct {
            token_idx: u32,
            layer: u16,
            point: []const u8,
            position: u32,
            stats: struct {
                count: u64,
                min: ?f32, // Nullable to handle inf/nan
                max: ?f32,
                sum: ?f64,
                sum_sq: ?f64,
                nan_count: u32,
                inf_count: u32,
            },
        },
    };

    fn parseTracePoint(name: []const u8) ?TracePoint {
        // Map string names back to TracePoint enum
        const pairs = .{
            .{ "embed_tokens", TracePoint.embed },
            .{ "embed_pos", TracePoint.embed_pos },
            .{ "layer_input", TracePoint.layer_input },
            .{ "layer_attn_norm", TracePoint.layer_attn_norm },
            .{ "attn.q", TracePoint.attn_q },
            .{ "attn.k", TracePoint.attn_k },
            .{ "attn.v", TracePoint.attn_v },
            .{ "attn.qk", TracePoint.attn_qk },
            .{ "attn.weights", TracePoint.attn_weights },
            .{ "attn.out", TracePoint.attn_out },
            .{ "layer_ffn_norm", TracePoint.layer_ffn_norm },
            .{ "ffn.gate", TracePoint.ffn_gate },
            .{ "ffn.up", TracePoint.ffn_up },
            .{ "ffn.act", TracePoint.ffn_act },
            .{ "ffn.down", TracePoint.ffn_down },
            .{ "block.out", TracePoint.block_out },
            .{ "mamba.out", TracePoint.mamba_out },
            .{ "conv.in_proj", TracePoint.conv_in_proj },
            .{ "conv.conv", TracePoint.conv_conv },
            .{ "conv.out_proj", TracePoint.conv_out_proj },
            .{ "final_norm", TracePoint.final_norm },
            .{ "lm_head", TracePoint.lm_head },
            .{ "logits_scaled", TracePoint.logits_scaled },
            .{ "logits_ready", TracePoint.logits_ready },
            .{ "token_select", TracePoint.token_select },
            .{ "ffn.act.map", TracePoint.ffn_act_map },
            .{ "ffn.act.mix", TracePoint.ffn_act_mix },
            .{ "gdelta.in_proj", TracePoint.gdelta_in_proj },
            .{ "gdelta.conv", TracePoint.gdelta_conv },
            .{ "gdelta.ssm", TracePoint.gdelta_ssm },
            .{ "gdelta.norm", TracePoint.gdelta_norm },
            .{ "gdelta.out", TracePoint.gdelta_out },
        };

        inline for (pairs) |pair| {
            if (std.mem.eql(u8, name, pair[0])) {
                return pair[1];
            }
        }
        return null;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "ReferenceRecorder basic flow" {
    const allocator = std.testing.allocator;

    var recorder = try ReferenceRecorder.init(allocator, "test_model", 42, 1.0, 10);
    defer recorder.deinit();

    // Record some tokens
    try recorder.recordToken(1);
    try recorder.recordToken(2);
    try recorder.recordToken(3);

    // Record some stats
    const emission = trace.TraceEmission{
        .point = .lm_head,
        .layer = trace.TraceEmission.NO_LAYER,
        .token = 0,
        .position = 0,
        .backend = .cpu,
        .tensor = .{
            .ptr = @ptrFromInt(0x1000),
            .dtype = .f32,
            .shape = .{ 4, 0, 0, 0 },
            .ndim = 1,
        },
        .timestamp_ns = 0,
        .kernel_name = std.mem.zeroes([48]u8),
    };

    const stats = TensorStats{
        .count = 4,
        .min = 1.0,
        .max = 4.0,
        .sum = 10.0,
        .sum_sq = 30.0,
        .nan_count = 0,
        .inf_count = 0,
    };

    try recorder.recordEmission(emission, stats);

    // Finalize
    var ref = try recorder.finalize();
    defer ref.deinit();

    try std.testing.expectEqual(@as(usize, 3), ref.token_transcript.len);
    try std.testing.expectEqual(@as(u32, 1), ref.token_transcript[0]);
    try std.testing.expectEqual(@as(usize, 1), ref.stats_records.len);
}

test "ReferenceVerifier matches good stats" {
    const allocator = std.testing.allocator;

    // Create reference
    const golden_stats = TensorStats{
        .count = 4,
        .min = 1.0,
        .max = 4.0,
        .sum = 10.0,
        .sum_sq = 30.0,
        .nan_count = 0,
        .inf_count = 0,
    };

    const ref = ReferenceData{
        .model_name = "test",
        .seed = 42,
        .temperature = 1.0,
        .max_tokens = 1,
        .token_transcript = &[_]u32{100},
        .stats_records = &[_]StatsRecord{.{
            .token_idx = 0,
            .layer = trace.TraceEmission.NO_LAYER,
            .point = .lm_head,
            .position = 0,
            .stats = golden_stats,
        }},
        .allocator = allocator,
    };

    var verifier = ReferenceVerifier.init(allocator, &ref, 1e-3);
    defer verifier.deinit();

    const emission = trace.TraceEmission{
        .point = .lm_head,
        .layer = trace.TraceEmission.NO_LAYER,
        .token = 0,
        .position = 0,
        .backend = .cpu,
        .tensor = .{
            .ptr = @ptrFromInt(0x1000),
            .dtype = .f32,
            .shape = .{ 4, 0, 0, 0 },
            .ndim = 1,
        },
        .timestamp_ns = 0,
        .kernel_name = std.mem.zeroes([48]u8),
    };

    // Should pass with identical stats
    try verifier.checkEmission(emission, golden_stats);
    try std.testing.expect(!verifier.has_diverged);
}

test "ReferenceVerifier detects divergence" {
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

    const ref = ReferenceData{
        .model_name = "test",
        .seed = 42,
        .temperature = 1.0,
        .max_tokens = 1,
        .token_transcript = &[_]u32{100},
        .stats_records = &[_]StatsRecord{.{
            .token_idx = 0,
            .layer = trace.TraceEmission.NO_LAYER,
            .point = .lm_head,
            .position = 0,
            .stats = golden_stats,
        }},
        .allocator = allocator,
    };

    var verifier = ReferenceVerifier.init(allocator, &ref, 1e-3);
    defer verifier.deinit();

    const emission = trace.TraceEmission{
        .point = .lm_head,
        .layer = trace.TraceEmission.NO_LAYER,
        .token = 0,
        .position = 0,
        .backend = .cpu,
        .tensor = .{
            .ptr = @ptrFromInt(0x1000),
            .dtype = .f32,
            .shape = .{ 4, 0, 0, 0 },
            .ndim = 1,
        },
        .timestamp_ns = 0,
        .kernel_name = std.mem.zeroes([48]u8),
    };

    // Different stats - should diverge
    const bad_stats = TensorStats{
        .count = 4,
        .min = 1.0,
        .max = 4.0,
        .sum = 10.0,
        .sum_sq = 50.0, // Different!
        .nan_count = 0,
        .inf_count = 0,
    };

    const result = verifier.checkEmission(emission, bad_stats);
    try std.testing.expectError(error.StatsDivergence, result);
    try std.testing.expect(verifier.has_diverged);
    try std.testing.expect(verifier.divergence_point != null);
}

test "ReferenceVerifier detects token divergence" {
    const allocator = std.testing.allocator;

    const ref = ReferenceData{
        .model_name = "test",
        .seed = 42,
        .temperature = 1.0,
        .max_tokens = 2,
        .token_transcript = &[_]u32{ 100, 200 },
        .stats_records = &.{},
        .allocator = allocator,
    };

    var verifier = ReferenceVerifier.init(allocator, &ref, 1e-3);

    try verifier.checkToken(100);
    verifier.nextToken();

    const result = verifier.checkToken(201);
    try std.testing.expectError(error.TokenDivergence, result);
    try std.testing.expect(verifier.has_diverged);
    try std.testing.expect(verifier.divergence_point != null);
    try std.testing.expectError(error.TokenDivergence, verifier.finish());
}

test "ReferenceVerifier ignores extra emissions but finish enforces completeness" {
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

    const ref = ReferenceData{
        .model_name = "test",
        .seed = 42,
        .temperature = 1.0,
        .max_tokens = 1,
        .token_transcript = &[_]u32{100},
        .stats_records = &[_]StatsRecord{
            .{
                .token_idx = 0,
                .layer = 7,
                .point = .layer_ffn_norm,
                .position = 14,
                .stats = golden_stats,
            },
        },
        .allocator = allocator,
    };

    var verifier = ReferenceVerifier.init(allocator, &ref, 1e-3);
    defer verifier.deinit();

    const extra_emission = trace.TraceEmission{
        .point = .layer_ffn_norm,
        .layer = 0,
        .token = 0,
        .position = 0,
        .backend = .cuda,
        .tensor = .{
            .ptr = @ptrFromInt(0x1000),
            .dtype = .f32,
            .shape = .{ 4, 0, 0, 0 },
            .ndim = 1,
        },
        .timestamp_ns = 0,
        .kernel_name = std.mem.zeroes([48]u8),
    };

    try verifier.checkEmission(extra_emission, golden_stats);
    try std.testing.expect(!verifier.has_diverged);

    const result = verifier.finish();
    try std.testing.expectError(error.ReferenceMissing, result);
    try std.testing.expect(verifier.has_diverged);
    try std.testing.expect(verifier.divergence_point != null);
}
