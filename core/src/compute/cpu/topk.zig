//! Top-k selection primitives used by CPU kernels.

const std = @import("std");

/// Maximum number of experts supported by current static bitset implementation.
pub const MAX_EXPERTS: usize = 256;

/// Select top-k indices and return normalized softmax weights over selected logits.
pub fn selectTopKNormalized(
    logits: []const f32,
    top_k: usize,
    indices: []u32,
    weights: []f32,
) error{InvalidMoEConfig}!void {
    const num_experts = logits.len;

    if (top_k == 0) return error.InvalidMoEConfig;
    if (top_k > num_experts) return error.InvalidMoEConfig;
    if (num_experts > MAX_EXPERTS) return error.InvalidMoEConfig;
    if (indices.len < top_k or weights.len < top_k) return error.InvalidMoEConfig;

    var used_mask = std.StaticBitSet(MAX_EXPERTS).initEmpty();

    for (0..top_k) |rank_idx| {
        var best_expert_index: usize = 0;
        var best_logit: f32 = -std.math.inf(f32);
        var found_finite = false;

        for (0..num_experts) |expert_index| {
            if (used_mask.isSet(expert_index)) continue;
            const logit = logits[expert_index];
            if (!std.math.isFinite(logit)) continue;
            if (!found_finite or logit > best_logit) {
                found_finite = true;
                best_logit = logit;
                best_expert_index = expert_index;
            }
        }

        if (!found_finite) {
            for (0..num_experts) |expert_index| {
                if (!used_mask.isSet(expert_index)) {
                    best_expert_index = expert_index;
                    break;
                }
            }
            best_logit = -std.math.inf(f32);
        }

        indices[rank_idx] = @intCast(best_expert_index);
        weights[rank_idx] = best_logit;
        used_mask.set(best_expert_index);
    }

    var max_logit: f32 = -std.math.inf(f32);
    for (weights[0..top_k]) |w| {
        if (std.math.isFinite(w) and w > max_logit) max_logit = w;
    }

    if (!std.math.isFinite(max_logit)) {
        const uniform = 1.0 / @as(f32, @floatFromInt(top_k));
        for (0..top_k) |idx| weights[idx] = uniform;
        return;
    }

    var sum_exp: f32 = 0.0;
    for (0..top_k) |idx| {
        const logit = weights[idx];
        weights[idx] = if (std.math.isFinite(logit)) @exp(logit - max_logit) else 0.0;
        sum_exp += weights[idx];
    }

    if (!std.math.isFinite(sum_exp) or sum_exp <= 0.0) {
        const uniform = 1.0 / @as(f32, @floatFromInt(top_k));
        for (0..top_k) |idx| weights[idx] = uniform;
        return;
    }

    for (0..top_k) |idx| {
        weights[idx] /= sum_exp;
    }
}

/// Comparator helper for descending `.value`.
pub fn byValueDesc(_: void, a: anytype, b: @TypeOf(a)) bool {
    return a.value > b.value;
}

/// Hoare partition (descending by `.value`) for entry slices.
pub inline fn partition(entries: anytype, left_bound: usize, right_bound: usize) usize {
    const middle = left_bound + (right_bound - left_bound) / 2;
    const left_value = entries[left_bound].value;
    const middle_value = entries[middle].value;
    const right_value = entries[right_bound].value;
    const pivot_index = if ((left_value >= middle_value) == (middle_value >= right_value))
        middle
    else if ((left_value >= middle_value) == (right_value >= left_value))
        left_bound
    else
        right_bound;
    const pivot_value = entries[pivot_index].value;

    var left_index = left_bound;
    var right_index = right_bound;

    while (true) {
        while (entries[left_index].value > pivot_value) left_index += 1;
        while (entries[right_index].value < pivot_value) right_index -= 1;
        if (left_index >= right_index) return right_index;
        const swap_value = entries[left_index];
        entries[left_index] = entries[right_index];
        entries[right_index] = swap_value;
        left_index += 1;
        right_index -= 1;
    }
}

/// Partial top-k selection for entry slices with `.value` fields.
pub fn quickSelectTopK(entries: anytype, top_k: usize) void {
    if (entries.len <= 1 or top_k == 0) return;
    const target_count = @min(top_k, entries.len);

    var left_index: usize = 0;
    var right_index: usize = entries.len - 1;
    while (left_index < right_index) {
        const pivot_index = partition(entries, left_index, right_index);
        if (pivot_index + 1 >= target_count) {
            right_index = pivot_index;
        } else {
            left_index = pivot_index + 1;
        }
    }
}

test "selectTopKNormalized picks top logits and normalizes" {
    const logits = [_]f32{ 0.1, 1.0, 0.4, 2.0 };
    var indices = [_]u32{ 0, 0 };
    var weights = [_]f32{ 0, 0 };

    try selectTopKNormalized(&logits, 2, &indices, &weights);

    try std.testing.expectEqual(@as(u32, 3), indices[0]);
    try std.testing.expectEqual(@as(u32, 1), indices[1]);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), weights[0] + weights[1], 1e-6);
}
