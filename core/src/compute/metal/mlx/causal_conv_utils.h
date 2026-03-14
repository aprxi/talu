// MLX Bridge - shared causal-conv sequence utilities.
//
// This helper keeps dense and fused paths on one vectorized convolution
// implementation: no token-by-token graph construction in hot paths.

#pragma once

#include "compute_common.h"

inline array causal_conv_run_sequence(
    array* conv_state,
    const array& bx,
    const array& conv_kernel_broadcast,
    int seq_len,
    int d_conv_i,
    int conv_dim_i
) {
    if (conv_state == nullptr) {
        throw std::invalid_argument("[causal_conv] null conv_state");
    }
    if (seq_len <= 0) {
        throw std::invalid_argument("[causal_conv] seq_len must be > 0");
    }

    // Decode fast path (seq_len == 1): avoid a full conv1d launch and compute
    // the same depthwise update with direct tensor ops.
    if (seq_len == 1) {
        const array shifted = slice(*conv_state, {0, 1, 0}, {1, d_conv_i, conv_dim_i});
        const array next_state = concatenate({shifted, bx}, 1);
        const array conv_token = sum(next_state * conv_kernel_broadcast, 1);
        *conv_state = next_state;
        return reshape(conv_token, {1, 1, conv_dim_i});
    }

    // Build [history, new_tokens] and run grouped depthwise conv1d in one op.
    const array history = concatenate({*conv_state, bx}, 1);
    const array weight = transpose(conv_kernel_broadcast, {2, 1, 0}); // [conv_dim, d_conv, 1]
    const array conv_full = conv1d(history, weight, 1, 0, 1, conv_dim_i);

    // Windows aligned with each new token.
    const array conv_seq = slice(conv_full, {0, 1, 0}, {1, seq_len + 1, conv_dim_i});

    // Persist rolling history for next call.
    *conv_state = slice(history, {0, seq_len, 0}, {1, seq_len + d_conv_i, conv_dim_i});
    return conv_seq;
}

// Stepwise reference recurrence retained for deep parity debugging.
inline array causal_conv_run_sequence_stepwise(
    array* conv_state,
    const array& bx,
    const array& conv_kernel_broadcast,
    int seq_len,
    int d_conv_i,
    int conv_dim_i
) {
    if (conv_state == nullptr) {
        throw std::invalid_argument("[causal_conv] null conv_state");
    }
    if (seq_len <= 0) {
        throw std::invalid_argument("[causal_conv] seq_len must be > 0");
    }

    array state = *conv_state;
    std::vector<array> outputs;
    outputs.reserve(static_cast<size_t>(seq_len));

    for (int token_idx = 0; token_idx < seq_len; ++token_idx) {
        const array bx_token = slice(bx, {0, token_idx, 0}, {1, token_idx + 1, conv_dim_i});
        const array shifted = slice(state, {0, 1, 0}, {1, d_conv_i, conv_dim_i});
        state = concatenate({shifted, bx_token}, 1);
        const array conv_token = sum(state * conv_kernel_broadcast, 1);
        outputs.emplace_back(reshape(conv_token, {1, 1, conv_dim_i}));
    }

    *conv_state = state;
    if (outputs.size() == 1) {
        return outputs.front();
    }
    return concatenate(outputs, 1);
}
