// MLX Bridge - shared ShortConv sequence utilities.
//
// This helper keeps dense and fused paths on one vectorized convolution
// implementation: no token-by-token graph construction in hot paths.

#pragma once

#include "compute_common.h"

inline array shortconv_run_sequence_conv(
    array* conv_state,
    const array& bx,
    const array& conv_kernel_broadcast,
    int seq_len,
    int d_conv_i,
    int conv_dim_i
) {
    if (conv_state == nullptr) {
        throw std::invalid_argument("[shortconv] null conv_state");
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
