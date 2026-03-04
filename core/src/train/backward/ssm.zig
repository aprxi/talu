//! Mamba2 SSM backward pass (reverse scan through time).
//!
//! Forward recurrence per head:
//!   h[t] = dA * h[t-1] + dt * B * x[t]
//!   y[t] = C · h[t] + D * x[t]
//!
//! where dA = exp(A_log * dt), A_log is the log-space diagonal decay.
//!
//! Backward is a reverse scan:
//!   d_h[t] = C[t] * d_y[t] + dA[t+1] * d_h[t+1]
//!
//! Then gradients for B, C, D, dt, x follow from chain rule.

const std = @import("std");

/// SSM backward pass for one head across a sequence.
///
/// Computes gradients for all SSM parameters given saved forward activations.
///
/// Params:
///   grad_x:         [seq_len * d_head]   — overwritten: gradient w.r.t. input x
///   grad_B:         [seq_len * d_state]  — accumulated: gradient w.r.t. B per step
///   grad_C:         [seq_len * d_state]  — accumulated: gradient w.r.t. C per step
///   grad_D:         [1]                  — accumulated: gradient w.r.t. skip connection
///   grad_dt:        [seq_len]            — accumulated: gradient w.r.t. dt per step
///   grad_A_log:     [1]                  — accumulated: gradient w.r.t. log A
///   grad_output:    [seq_len * d_head]   — gradient from upstream
///   x:              [seq_len * d_head]   — saved input
///   B:              [seq_len * d_state]  — saved B values
///   C:              [seq_len * d_state]  — saved C values
///   D_val:          scalar               — skip connection value
///   A_log_val:      scalar               — log-space diagonal A
///   dt:             [seq_len]            — saved delta-time values (after softplus)
///   states:         [seq_len * d_head * d_state] — saved SSM states h[t]
///   d_head:         head dimension
///   d_state:        state dimension
///   seq_len:        sequence length
pub fn ssmBackward(
    grad_x: []f32,
    grad_B: []f32,
    grad_C: []f32,
    grad_D: []f32,
    grad_dt: []f32,
    grad_A_log: []f32,
    grad_output: []const f32,
    x: []const f32,
    B: []const f32,
    C: []const f32,
    D_val: f32,
    A_log_val: f32,
    dt: []const f32,
    states: []const f32,
    d_head: usize,
    d_state: usize,
    seq_len: usize,
) void {
    std.debug.assert(grad_x.len == seq_len * d_head);
    std.debug.assert(grad_B.len == seq_len * d_state);
    std.debug.assert(grad_C.len == seq_len * d_state);
    std.debug.assert(grad_D.len >= 1);
    std.debug.assert(grad_dt.len == seq_len);
    std.debug.assert(grad_A_log.len >= 1);
    std.debug.assert(grad_output.len == seq_len * d_head);
    std.debug.assert(x.len == seq_len * d_head);
    std.debug.assert(B.len == seq_len * d_state);
    std.debug.assert(C.len == seq_len * d_state);
    std.debug.assert(dt.len == seq_len);
    std.debug.assert(states.len == seq_len * d_head * d_state);

    const a_val = -@exp(A_log_val);

    // Hidden state gradient buffer: d_h[d_head, d_state]
    var d_h_buf: [4096]f32 = undefined;
    const d_h = d_h_buf[0 .. d_head * d_state];
    @memset(d_h, 0);

    // Reverse scan
    var t_rev: usize = 0;
    while (t_rev < seq_len) : (t_rev += 1) {
        const t = seq_len - 1 - t_rev;
        const dt_t = dt[t];
        const dA_t = @exp(a_val * dt_t);

        const dy = grad_output[t * d_head ..][0..d_head];
        const x_t = x[t * d_head ..][0..d_head];
        const B_t = B[t * d_state ..][0..d_state];
        const C_t = C[t * d_state ..][0..d_state];
        const h_t = states[t * d_head * d_state ..][0 .. d_head * d_state];

        // y[t,d] = sum_s(C[t,s] * h[t,d,s]) + D * x[t,d]
        // d_h[d,s] += C[t,s] * dy[d]  (from output)
        for (0..d_head) |d| {
            for (0..d_state) |s| {
                d_h[d * d_state + s] += C_t[s] * dy[d];
            }
        }

        // d_C[t,s] += sum_d(dy[d] * h[t,d,s])
        for (0..d_state) |s| {
            var sum_val: f32 = 0.0;
            for (0..d_head) |d| {
                sum_val += dy[d] * h_t[d * d_state + s];
            }
            grad_C[t * d_state + s] += sum_val;
        }

        // d_D += sum_d(dy[d] * x[t,d])
        var d_D_sum: f32 = 0.0;
        for (0..d_head) |d| {
            d_D_sum += dy[d] * x_t[d];
        }
        grad_D[0] += d_D_sum;

        // Gradient w.r.t. x from skip: d_x[t,d] = D * dy[d]
        const dx_t = grad_x[t * d_head ..][0..d_head];
        for (0..d_head) |d| {
            dx_t[d] = D_val * dy[d];
        }

        // h[t] = dA * h[t-1] + dt * B * x
        // d_x[t,d] += sum_s(d_h[d,s] * dt * B[t,s])
        for (0..d_head) |d| {
            for (0..d_state) |s| {
                dx_t[d] += d_h[d * d_state + s] * dt_t * B_t[s];
            }
        }

        // d_B[t,s] += sum_d(d_h[d,s] * dt * x[t,d])
        for (0..d_state) |s| {
            var sum_val: f32 = 0.0;
            for (0..d_head) |d| {
                sum_val += d_h[d * d_state + s] * dt_t * x_t[d];
            }
            grad_B[t * d_state + s] += sum_val;
        }

        // d_dt[t] += sum_{d,s}(d_h[d,s] * B[t,s] * x[t,d])
        //          + sum_{d,s}(d_h[d,s] * a_val * h[t-1,d,s])  [from dA gradient]
        var d_dt_val: f32 = 0.0;
        for (0..d_head) |d| {
            for (0..d_state) |s| {
                d_dt_val += d_h[d * d_state + s] * B_t[s] * x_t[d];
            }
        }
        // d_A_log contribution through dA = exp(a_val * dt)
        if (t > 0) {
            const h_prev = states[(t - 1) * d_head * d_state ..][0 .. d_head * d_state];
            var dA_dt_contrib: f32 = 0.0;
            for (0..d_head) |d| {
                for (0..d_state) |s| {
                    dA_dt_contrib += d_h[d * d_state + s] * h_prev[d * d_state + s];
                }
            }
            d_dt_val += dA_dt_contrib * a_val * dA_t;

            // d_A_log += dA_dt_contrib * dt * dA * (-exp(A_log))
            grad_A_log[0] += dA_dt_contrib * dt_t * dA_t * a_val;
        }
        grad_dt[t] += d_dt_val;

        // Propagate d_h backward: d_h_prev = dA * d_h_current
        for (d_h) |*v| {
            v.* *= dA_t;
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

test "ssmBackward skip connection gradient" {
    // D=2, trivial: y = D*x, so d_D = sum(dy * x), d_x = D * dy
    const seq_len: usize = 2;
    const d_head: usize = 1;
    const d_state: usize = 1;

    var grad_x = [_]f32{ 0, 0 };
    var grad_B = [_]f32{ 0, 0 };
    var grad_C = [_]f32{ 0, 0 };
    var grad_D = [_]f32{0};
    var grad_dt = [_]f32{ 0, 0 };
    var grad_A_log = [_]f32{0};

    const grad_output = [_]f32{ 1.0, 1.0 };
    const x_val = [_]f32{ 3.0, 4.0 };
    const B_val = [_]f32{ 0, 0 };
    const C_val = [_]f32{ 0, 0 };
    const dt_val = [_]f32{ 0, 0 };
    const states_val = [_]f32{ 0, 0 };

    ssmBackward(
        &grad_x, &grad_B, &grad_C, &grad_D, &grad_dt, &grad_A_log,
        &grad_output, &x_val, &B_val, &C_val,
        2.0, -1.0, // D=2, A_log=-1
        &dt_val, &states_val,
        d_head, d_state, seq_len,
    );

    // d_D = sum(dy * x) = 1*3 + 1*4 = 7
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), grad_D[0], 1e-5);

    // d_x = D * dy = 2 * [1, 1] = [2, 2]
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), grad_x[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), grad_x[1], 1e-5);
}

test "ssmBackward B gradient accumulates" {
    const seq_len: usize = 1;
    const d_head: usize = 1;
    const d_state: usize = 2;

    var grad_x = [_]f32{0};
    var grad_B = [_]f32{ 0, 0 };
    var grad_C = [_]f32{ 0, 0 };
    var grad_D = [_]f32{0};
    var grad_dt = [_]f32{0};
    var grad_A_log = [_]f32{0};

    const grad_output = [_]f32{1.0};
    const x_val = [_]f32{2.0};
    const B_val = [_]f32{ 1.0, 0.5 };
    const C_val = [_]f32{ 1.0, 1.0 };
    const dt_val = [_]f32{1.0};
    // h = dt * B * x = 1 * [1, 0.5] * 2 = [2, 1]
    const states_val = [_]f32{ 2.0, 1.0 };

    ssmBackward(
        &grad_x, &grad_B, &grad_C, &grad_D, &grad_dt, &grad_A_log,
        &grad_output, &x_val, &B_val, &C_val,
        0.0, -1.0,
        &dt_val, &states_val,
        d_head, d_state, seq_len,
    );

    // grad_B should be non-zero (from d_h propagation)
    var any_nonzero = false;
    for (grad_B) |v| {
        if (@abs(v) > 1e-6) {
            any_nonzero = true;
            break;
        }
    }
    try std.testing.expect(any_nonzero);
}
