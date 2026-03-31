// Non-fused attention reference path (correctness-first).
//
// This is intentionally explicit: separate Q/K/V projections and explicit
// staged math to mirror CPU control flow more closely.
#include "attention_utils.h"
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static inline float cpu_like_fast_exp_scalar(float x) {
    constexpr float log2e = 1.0f / 0.6931471805599453f;
    constexpr float ln2_hi = 0.693359375f;
    constexpr float ln2_lo = -2.12194440e-4f;
    constexpr float exp_hi = 88.3762626647949f;
    constexpr float exp_lo = -88.3762626647949f;
    constexpr float p0 = 1.9875691500e-4f;
    constexpr float p1 = 1.3981999507e-3f;
    constexpr float p2 = 8.3334519073e-3f;
    constexpr float p3 = 4.1665795894e-2f;
    constexpr float p4 = 1.6666665459e-1f;
    constexpr float p5 = 5.0000001201e-1f;

    float x_clamped = std::max(std::min(x, exp_hi), exp_lo);
    const float fx = std::floor(x_clamped * log2e + 0.5f);
    const int32_t fxi = static_cast<int32_t>(fx);
    x_clamped = x_clamped - fx * ln2_hi - fx * ln2_lo;

    float y = p0;
    y = y * x_clamped + p1;
    y = y * x_clamped + p2;
    y = y * x_clamped + p3;
    y = y * x_clamped + p4;
    y = y * x_clamped + p5;
    y = y * x_clamped * x_clamped + x_clamped + 1.0f;

    const uint32_t exponent_bits = static_cast<uint32_t>((fxi + 127) << 23);
    float pow2n = 0.0f;
    std::memcpy(&pow2n, &exponent_bits, sizeof(float));
    return y * pow2n;
}

static inline array owned_f32_array(const std::vector<float>& values, const Shape& shape) {
    static constexpr size_t kAlignment = 16 * 1024;
    const size_t element_count = std::max<size_t>(values.size(), 1);
    const size_t byte_count = element_count * sizeof(float);
    void* aligned_ptr = nullptr;
    if (posix_memalign(&aligned_ptr, kAlignment, byte_count) != 0 || aligned_ptr == nullptr) {
        throw std::bad_alloc();
    }
    auto* copied = static_cast<float*>(aligned_ptr);
    if (!values.empty()) {
        std::memcpy(copied, values.data(), values.size() * sizeof(float));
    } else {
        copied[0] = 0.0f;
    }
    auto owner = std::shared_ptr<void>(aligned_ptr, [](void* ptr) {
        std::free(ptr);
    });
    auto deleter = [owner](void*) {
        // Ownership retained by closure capture.
    };
    return array(copied, shape, float32, deleter);
}

static inline float attention_truncate_to_bf16_f32(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(uint32_t));
    const uint32_t truncated = bits & 0xFFFF0000u;
    float out = 0.0f;
    std::memcpy(&out, &truncated, sizeof(float));
    return out;
}

static inline float cpu_like_dot_row(
    const float* a,
    const float* b,
    int len
) {
    // Scalar fallback keeps the same 4x-unrolled grouping as CPU dotRow.
    float sum0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float sum1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float sum2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float sum3[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int idx = 0;

    for (; idx + 16 <= len; idx += 16) {
        for (int lane = 0; lane < 4; ++lane) {
            sum0[lane] = std::fma(a[idx + lane], b[idx + lane], sum0[lane]);
            sum1[lane] = std::fma(a[idx + 4 + lane], b[idx + 4 + lane], sum1[lane]);
            sum2[lane] = std::fma(a[idx + 8 + lane], b[idx + 8 + lane], sum2[lane]);
            sum3[lane] = std::fma(a[idx + 12 + lane], b[idx + 12 + lane], sum3[lane]);
        }
    }

    for (; idx + 4 <= len; idx += 4) {
        for (int lane = 0; lane < 4; ++lane) {
            sum0[lane] = std::fma(a[idx + lane], b[idx + lane], sum0[lane]);
        }
    }

    float sum = 0.0f;
    for (int lane = 0; lane < 4; ++lane) {
        sum += sum0[lane] + sum1[lane] + sum2[lane] + sum3[lane];
    }
    for (; idx < len; ++idx) {
        sum = std::fma(a[idx], b[idx], sum);
    }
    return sum;
}

static inline float cpu_like_bf16_dot_row(
    const float* a,
    const float* b,
    int len
) {
    auto dot8_into_lanes = [](float lanes[4], const float* lhs, const float* rhs) {
        const float a0 = attention_truncate_to_bf16_f32(lhs[0]);
        const float a1 = attention_truncate_to_bf16_f32(lhs[1]);
        const float a2 = attention_truncate_to_bf16_f32(lhs[2]);
        const float a3 = attention_truncate_to_bf16_f32(lhs[3]);
        const float a4 = attention_truncate_to_bf16_f32(lhs[4]);
        const float a5 = attention_truncate_to_bf16_f32(lhs[5]);
        const float a6 = attention_truncate_to_bf16_f32(lhs[6]);
        const float a7 = attention_truncate_to_bf16_f32(lhs[7]);
        const float b0 = attention_truncate_to_bf16_f32(rhs[0]);
        const float b1 = attention_truncate_to_bf16_f32(rhs[1]);
        const float b2 = attention_truncate_to_bf16_f32(rhs[2]);
        const float b3 = attention_truncate_to_bf16_f32(rhs[3]);
        const float b4 = attention_truncate_to_bf16_f32(rhs[4]);
        const float b5 = attention_truncate_to_bf16_f32(rhs[5]);
        const float b6 = attention_truncate_to_bf16_f32(rhs[6]);
        const float b7 = attention_truncate_to_bf16_f32(rhs[7]);

        lanes[0] = std::fma(a0, b0, lanes[0]);
        lanes[0] = std::fma(a1, b1, lanes[0]);
        lanes[1] = std::fma(a2, b2, lanes[1]);
        lanes[1] = std::fma(a3, b3, lanes[1]);
        lanes[2] = std::fma(a4, b4, lanes[2]);
        lanes[2] = std::fma(a5, b5, lanes[2]);
        lanes[3] = std::fma(a6, b6, lanes[3]);
        lanes[3] = std::fma(a7, b7, lanes[3]);
    };

    float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc3[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int idx = 0;

    while (idx + 32 <= len) {
        dot8_into_lanes(acc0, a + idx + 0, b + idx + 0);
        dot8_into_lanes(acc1, a + idx + 8, b + idx + 8);
        dot8_into_lanes(acc2, a + idx + 16, b + idx + 16);
        dot8_into_lanes(acc3, a + idx + 24, b + idx + 24);
        idx += 32;
    }
    while (idx + 8 <= len) {
        dot8_into_lanes(acc0, a + idx, b + idx);
        idx += 8;
    }

    const float lane0 = (acc0[0] + acc1[0]) + (acc2[0] + acc3[0]);
    const float lane1 = (acc0[1] + acc1[1]) + (acc2[1] + acc3[1]);
    const float lane2 = (acc0[2] + acc1[2]) + (acc2[2] + acc3[2]);
    const float lane3 = (acc0[3] + acc1[3]) + (acc2[3] + acc3[3]);
    float sum = (lane0 + lane1) + (lane2 + lane3);
    for (; idx < len; ++idx) {
        sum = std::fma(
            attention_truncate_to_bf16_f32(a[idx]),
            attention_truncate_to_bf16_f32(b[idx]),
            sum
        );
    }
    return sum;
}

static inline array cpu_like_softmax_scores(
    const array& scores,
    int batch_size,
    int n_heads,
    int q_len,
    int k_len,
    const void* attn_sinks
) {
    array scores_f32 = (scores.dtype() == float32) ? scores : astype(scores, float32);
    array flat_scores = reshape(scores_f32, {batch_size * n_heads * q_len * k_len});
    eval(flat_scores);
    const float* scores_ptr = flat_scores.data<float>();

    std::vector<float> out(static_cast<size_t>(batch_size) * static_cast<size_t>(n_heads) * static_cast<size_t>(q_len) * static_cast<size_t>(k_len));
    std::memcpy(out.data(), scores_ptr, out.size() * sizeof(float));

    std::optional<std::vector<float>> sinks_host = std::nullopt;
    if (attn_sinks != nullptr) {
        const auto& sinks = *static_cast<const array*>(attn_sinks);
        array sinks_f32 = (sinks.dtype() == float32) ? sinks : astype(sinks, float32);
        array flat_sinks = reshape(sinks_f32, {n_heads});
        eval(flat_sinks);
        const float* sinks_ptr = flat_sinks.data<float>();
        sinks_host = std::vector<float>(sinks_ptr, sinks_ptr + n_heads);
    }

    const float cutoff = -std::numeric_limits<float>::infinity() + 1.0f;
    for (int batch = 0; batch < batch_size; ++batch) {
        for (int head = 0; head < n_heads; ++head) {
            const bool has_sink = sinks_host.has_value();
            const float sink_logit = has_sink ? sinks_host.value()[static_cast<size_t>(head)] : 0.0f;
            for (int q_idx = 0; q_idx < q_len; ++q_idx) {
                const size_t row_base =
                    ((static_cast<size_t>(batch) * static_cast<size_t>(n_heads) + static_cast<size_t>(head)) * static_cast<size_t>(q_len) + static_cast<size_t>(q_idx)) *
                    static_cast<size_t>(k_len);
                float maxv = -std::numeric_limits<float>::infinity();
                for (int k_idx = 0; k_idx < k_len; ++k_idx) {
                    const float score = out[row_base + static_cast<size_t>(k_idx)];
                    if (score <= cutoff) continue;
                    maxv = std::max(maxv, score);
                }
                if (has_sink) {
                    maxv = std::max(maxv, sink_logit);
                }
                float sum = 0.0f;
                if (has_sink) {
                    sum += cpu_like_fast_exp_scalar(sink_logit - maxv);
                }
                for (int k_idx = 0; k_idx < k_len; ++k_idx) {
                    const size_t idx = row_base + static_cast<size_t>(k_idx);
                    const float score = out[idx];
                    if (score <= cutoff) {
                        out[idx] = 0.0f;
                        continue;
                    }
                    const float exp_score = cpu_like_fast_exp_scalar(score - maxv);
                    out[idx] = exp_score;
                    sum += exp_score;
                }
                if (sum == 0.0f) {
                    for (int k_idx = 0; k_idx < k_len; ++k_idx) {
                        out[row_base + static_cast<size_t>(k_idx)] = 0.0f;
                    }
                } else {
                    const float inv_sum = 1.0f / sum;
                    for (int k_idx = 0; k_idx < k_len; ++k_idx) {
                        out[row_base + static_cast<size_t>(k_idx)] *= inv_sum;
                    }
                }
            }
        }
    }

    return owned_f32_array(out, Shape{batch_size, n_heads, q_len, k_len});
}

static inline array cpu_like_attention_scores(
    const array& q,
    const array& k_expanded,
    int batch_size,
    int n_heads,
    int q_len,
    int k_len,
    int head_dim,
    float scale_value,
    bool is_prefill,
    int sliding_window
) {
    array q_f32 = (q.dtype() == float32) ? q : astype(q, float32);
    array k_f32 = (k_expanded.dtype() == float32) ? k_expanded : astype(k_expanded, float32);
    array q_flat = reshape(q_f32, {batch_size * n_heads * q_len * head_dim});
    array k_flat = reshape(k_f32, {batch_size * n_heads * k_len * head_dim});
    eval(q_flat, k_flat);
    const float* q_ptr = q_flat.data<float>();
    const float* k_ptr = k_flat.data<float>();

    std::vector<float> scores_host(
        static_cast<size_t>(batch_size) * static_cast<size_t>(n_heads) *
        static_cast<size_t>(q_len) * static_cast<size_t>(k_len)
    );
    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (int batch = 0; batch < batch_size; ++batch) {
        for (int head = 0; head < n_heads; ++head) {
            for (int q_idx = 0; q_idx < q_len; ++q_idx) {
                const int window_start = is_prefill
                    ? ((sliding_window > 0 && (q_idx + 1) > sliding_window) ? (q_idx + 1 - sliding_window) : 0)
                    : ((sliding_window > 0 && k_len > sliding_window) ? (k_len - sliding_window) : 0);
                for (int k_idx = 0; k_idx < k_len; ++k_idx) {
                    const size_t dst =
                        (((static_cast<size_t>(batch) * static_cast<size_t>(n_heads) + static_cast<size_t>(head)) * static_cast<size_t>(q_len) + static_cast<size_t>(q_idx)) *
                         static_cast<size_t>(k_len)) +
                        static_cast<size_t>(k_idx);
                    if ((is_prefill && k_idx > q_idx) || k_idx < window_start) {
                        scores_host[dst] = neg_inf;
                        continue;
                    }
                    const size_t q_base =
                        (((static_cast<size_t>(batch) * static_cast<size_t>(n_heads) + static_cast<size_t>(head)) * static_cast<size_t>(q_len) + static_cast<size_t>(q_idx)) *
                         static_cast<size_t>(head_dim));
                    const size_t k_base =
                        (((static_cast<size_t>(batch) * static_cast<size_t>(n_heads) + static_cast<size_t>(head)) * static_cast<size_t>(k_len) + static_cast<size_t>(k_idx)) *
                         static_cast<size_t>(head_dim));
                    const float dot = cpu_like_dot_row(
                        q_ptr + q_base,
                        k_ptr + k_base,
                        head_dim
                    );
                    scores_host[dst] = dot * scale_value;
                }
            }
        }
    }
    return owned_f32_array(scores_host, Shape{batch_size, n_heads, q_len, k_len});
}

static inline array cpu_like_trace_raw_qk_last_row(
    const array& q,
    const array& k_for_attn,
    int batch_size,
    int n_heads,
    int n_kv_heads,
    int q_len,
    int k_len,
    int head_dim,
    float scale_value,
    bool is_prefill,
    int sliding_window
) {
    array q_f32 = (q.dtype() == float32) ? q : astype(q, float32);
    array k_f32 = (k_for_attn.dtype() == float32) ? k_for_attn : astype(k_for_attn, float32);
    array q_flat = reshape(q_f32, {batch_size * n_heads * q_len * head_dim});
    array k_flat = reshape(k_f32, {batch_size * n_kv_heads * k_len * head_dim});
    eval(q_flat, k_flat);
    const float* q_ptr = q_flat.data<float>();
    const float* k_ptr = k_flat.data<float>();

    const int q_idx = q_len - 1;
    const int window_start = is_prefill
        ? ((sliding_window > 0 && (q_idx + 1) > sliding_window) ? (q_idx + 1 - sliding_window) : 0)
        : ((sliding_window > 0 && k_len > sliding_window) ? (k_len - sliding_window) : 0);
    const int heads_per_kv_group = n_heads / n_kv_heads;
    std::vector<float> trace_scores_host(static_cast<size_t>(n_heads) * static_cast<size_t>(k_len));
    const float neg_inf = -std::numeric_limits<float>::infinity();
    const int batch = 0;

    for (int head = 0; head < n_heads; ++head) {
        const int kv_head = head / heads_per_kv_group;
        const size_t q_base =
            (((static_cast<size_t>(batch) * static_cast<size_t>(n_heads) + static_cast<size_t>(head)) * static_cast<size_t>(q_len) + static_cast<size_t>(q_idx)) *
             static_cast<size_t>(head_dim));
        for (int k_idx = 0; k_idx < k_len; ++k_idx) {
            const size_t dst = static_cast<size_t>(head) * static_cast<size_t>(k_len) + static_cast<size_t>(k_idx);
            if ((is_prefill && k_idx > q_idx) || k_idx < window_start) {
                trace_scores_host[dst] = neg_inf;
                continue;
            }
            const size_t k_base =
                (((static_cast<size_t>(batch) * static_cast<size_t>(n_kv_heads) + static_cast<size_t>(kv_head)) * static_cast<size_t>(k_len) + static_cast<size_t>(k_idx)) *
                 static_cast<size_t>(head_dim));
            trace_scores_host[dst] = cpu_like_dot_row(q_ptr + q_base, k_ptr + k_base, head_dim) * scale_value;
        }
    }

    return owned_f32_array(trace_scores_host, Shape{n_heads, k_len});
}

static inline array cpu_like_linear_lastdim(
    const array& input,
    const array& rhs
) {
    if (input.ndim() != 3 || rhs.ndim() != 2) {
        throw std::invalid_argument("[attention_reference] cpu_like_linear_lastdim expects [B,S,I] x [I,O]|[O,I]");
    }
    const int batch = input.shape(0);
    const int seq_len = input.shape(1);
    const int in_dim = input.shape(2);
    const int rhs0 = rhs.shape(0);
    const int rhs1 = rhs.shape(1);
    const bool rhs_is_io = (rhs0 == in_dim);
    const bool rhs_is_oi = (rhs1 == in_dim);
    if (!rhs_is_io && !rhs_is_oi) {
        throw std::invalid_argument("[attention_reference] linear rhs/input shape mismatch");
    }
    const int out_dim = rhs_is_io ? rhs1 : rhs0;
    const bool emulate_bf16_rhs = rhs.dtype() == bfloat16;

    array input_f32 = (input.dtype() == float32) ? input : astype(input, float32);
    array rhs_f32 = (rhs.dtype() == float32) ? rhs : astype(rhs, float32);
    array input_flat = reshape(input_f32, {batch * seq_len * in_dim});
    array rhs_flat = reshape(rhs_f32, {rhs0 * rhs1});
    eval(input_flat, rhs_flat);
    std::vector<float> input_vec(static_cast<size_t>(batch) * static_cast<size_t>(seq_len) * static_cast<size_t>(in_dim));
    std::vector<float> rhs_vec(static_cast<size_t>(rhs0) * static_cast<size_t>(rhs1));
    std::memcpy(input_vec.data(), input_flat.data<float>(), input_vec.size() * sizeof(float));
    std::memcpy(rhs_vec.data(), rhs_flat.data<float>(), rhs_vec.size() * sizeof(float));
    const float* input_ptr = input_vec.data();
    const float* rhs_ptr = rhs_vec.data();
    std::vector<float> rhs_rows_io;
    const float* rhs_io_ptr = nullptr;
    if (rhs_is_io) {
        rhs_rows_io.resize(static_cast<size_t>(out_dim) * static_cast<size_t>(in_dim));
        for (int o = 0; o < out_dim; ++o) {
            float* dst_row = rhs_rows_io.data() + static_cast<size_t>(o) * static_cast<size_t>(in_dim);
            for (int i = 0; i < in_dim; ++i) {
                dst_row[static_cast<size_t>(i)] =
                    rhs_ptr[static_cast<size_t>(i) * static_cast<size_t>(rhs1) + static_cast<size_t>(o)];
            }
        }
        rhs_io_ptr = rhs_rows_io.data();
    }

    std::vector<float> out(static_cast<size_t>(batch) * static_cast<size_t>(seq_len) * static_cast<size_t>(out_dim), 0.0f);
    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            const size_t in_base = (static_cast<size_t>(b) * static_cast<size_t>(seq_len) + static_cast<size_t>(s)) * static_cast<size_t>(in_dim);
            const size_t out_base = (static_cast<size_t>(b) * static_cast<size_t>(seq_len) + static_cast<size_t>(s)) * static_cast<size_t>(out_dim);
            for (int o = 0; o < out_dim; ++o) {
                float acc = 0.0f;
                if (rhs_is_oi) {
                    const float* rhs_row = rhs_ptr + static_cast<size_t>(o) * static_cast<size_t>(rhs1);
                    acc = emulate_bf16_rhs
                        ? cpu_like_bf16_dot_row(input_ptr + in_base, rhs_row, in_dim)
                        : cpu_like_dot_row(input_ptr + in_base, rhs_row, in_dim);
                } else {
                    const float* rhs_row = rhs_io_ptr + static_cast<size_t>(o) * static_cast<size_t>(in_dim);
                    acc = emulate_bf16_rhs
                        ? cpu_like_bf16_dot_row(input_ptr + in_base, rhs_row, in_dim)
                        : cpu_like_dot_row(input_ptr + in_base, rhs_row, in_dim);
                }
                out[out_base + static_cast<size_t>(o)] = acc;
            }
        }
    }
    return owned_f32_array(out, Shape{batch, seq_len, out_dim});
}

static inline array apply_query_output_gate_cpu_like(
    const array& attention_out,
    const std::optional<array>& query_gate_values
) {
    if (!query_gate_values.has_value()) {
        return attention_out;
    }

    array attention_f32 = (attention_out.dtype() == float32) ? attention_out : astype(attention_out, float32);
    array gates_f32 = (query_gate_values.value().dtype() == float32) ? query_gate_values.value() : astype(query_gate_values.value(), float32);
    array flat_attention = reshape(attention_f32, {static_cast<int>(attention_f32.size())});
    array flat_gates = reshape(gates_f32, {static_cast<int>(gates_f32.size())});
    eval(flat_attention, flat_gates);
    const float* attn_ptr = flat_attention.data<float>();
    const float* gate_ptr = flat_gates.data<float>();
    const size_t count = static_cast<size_t>(attention_f32.size());
    std::vector<float> gated(count);
    for (size_t idx = 0; idx < count; ++idx) {
        const float sig = 1.0f / (1.0f + cpu_like_fast_exp_scalar(-gate_ptr[idx]));
        gated[idx] = attn_ptr[idx] * sig;
    }
    array gated_arr = [&]() -> array {
        if (attention_f32.ndim() == 4) {
            return owned_f32_array(
                gated,
                Shape{
                    attention_f32.shape(0),
                    attention_f32.shape(1),
                    attention_f32.shape(2),
                    attention_f32.shape(3),
                }
            );
        }
        if (attention_f32.ndim() == 3) {
            return owned_f32_array(
                gated,
                Shape{
                    attention_f32.shape(0),
                    attention_f32.shape(1),
                    attention_f32.shape(2),
                }
            );
        }
        if (attention_f32.ndim() == 2) {
            return owned_f32_array(
                gated,
                Shape{
                    attention_f32.shape(0),
                    attention_f32.shape(1),
                }
            );
        }
        if (attention_f32.ndim() == 1) {
            return owned_f32_array(gated, Shape{attention_f32.shape(0)});
        }
        throw std::invalid_argument("[attention_reference] unsupported rank for gated attention output");
    }();
    if (attention_out.dtype() != float32) {
        gated_arr = astype(gated_arr, attention_out.dtype());
    }
    return gated_arr;
}


static inline void* mlx_lazy_attention_reference_bf16_impl(
    const void* input,
    const void* q_w, const void* k_w, const void* v_w, const void* o_w,
    const void* q_norm_w,
    const void* k_norm_w,
    const void* q_bias,
    const void* k_bias,
    const void* v_bias,
    const void* o_bias,
    const void* attn_sinks,
    void* cache_ptr, size_t layer_idx,
    size_t n_heads, size_t n_kv_heads, size_t head_dim,
    size_t rope_dim,
    size_t pos_offset, float rope_theta,
    bool rope_interleaved,
    const void* runtime_rope_cos, const void* runtime_rope_sin, size_t runtime_rope_dim,
    float rms_eps,
    float query_pre_attn_scalar,
    float attention_multiplier,
    size_t sliding_window,
    bool query_gate,
    void** out_attn_q_proj_raw,
    void** out_attn_k_proj_raw,
    void** out_attn_q_norm,
    void** out_attn_k_norm,
    void** out_attn_q_rope,
    void** out_attn_k_rope,
    void** out_attn_qk,
    void** out_attn_weights,
    void** out_attn_q,
    void** out_attn_k
) {
    mlx_count_op();
    const auto& x = *static_cast<const array*>(input);
    const int batch_size = x.shape(0);
    const int seq_len = x.shape(1);
    const int hidden_dim = x.shape(2);
    const int n_heads_int = static_cast<int>(n_heads);
    const int n_kv_heads_int = static_cast<int>(n_kv_heads);
    const int head_dim_int = static_cast<int>(head_dim);
    const int rope_dim_int = (rope_dim > 0) ? static_cast<int>(rope_dim) : head_dim_int;
    const int sliding_window_int = static_cast<int>(sliding_window);
    const int q_out = n_heads_int * head_dim_int;
    const int q_proj_out = query_gate ? (q_out * 2) : q_out;
    const int kv_out = n_kv_heads_int * head_dim_int;
    const float scale_value = resolve_attention_scale(query_pre_attn_scalar, attention_multiplier, head_dim_int);
    const array x_f32 = (x.dtype() == float32) ? x : astype(x, float32);

    // Separate projections (no fused qkv weight path).
    // Reference path: consume raw model tensors directly and let the CPU-like
    // linear helper infer [I,O] vs [O,I] orientation from shape.
    array q_w_ref = *static_cast<const array*>(q_w);
    array k_w_ref = *static_cast<const array*>(k_w);
    array v_w_ref = *static_cast<const array*>(v_w);
    array o_w_ref = *static_cast<const array*>(o_w);

    array q = cpu_like_linear_lastdim(x_f32, q_w_ref);
    array k = cpu_like_linear_lastdim(x_f32, k_w_ref);
    array v = cpu_like_linear_lastdim(x_f32, v_w_ref);
    maybe_add_linear_bias(&q, q_bias);
    maybe_add_linear_bias(&k, k_bias);
    maybe_add_linear_bias(&v, v_bias);
    if (out_attn_q_proj_raw != nullptr) {
        if (batch_size > 0 && seq_len > 0 && q_proj_out > 0) {
            array q_proj_raw_f32 = (q.dtype() == float32) ? q : astype(q, float32);
            array q_proj_raw_flat = reshape(q_proj_raw_f32, {batch_size * seq_len * q_proj_out});
            eval(q_proj_raw_flat);
            const float* q_proj_raw_ptr = q_proj_raw_flat.data<float>();
            std::vector<float> trace_q_proj_raw_host(
                static_cast<size_t>(seq_len) * static_cast<size_t>(q_proj_out)
            );
            const int batch_idx = 0;
            for (int s = 0; s < seq_len; ++s) {
                for (int d = 0; d < q_proj_out; ++d) {
                    const size_t src =
                        ((static_cast<size_t>(batch_idx) * static_cast<size_t>(seq_len) + static_cast<size_t>(s)) * static_cast<size_t>(q_proj_out)) +
                        static_cast<size_t>(d);
                    const size_t dst =
                        static_cast<size_t>(s) * static_cast<size_t>(q_proj_out) +
                        static_cast<size_t>(d);
                    trace_q_proj_raw_host[dst] = q_proj_raw_ptr[src];
                }
            }
            array trace_q_proj_raw = owned_f32_array(trace_q_proj_raw_host, Shape{1, seq_len, q_proj_out});
            *out_attn_q_proj_raw = pool_array(std::move(trace_q_proj_raw));
        } else {
            *out_attn_q_proj_raw = nullptr;
        }
    }
    if (out_attn_k_proj_raw != nullptr) {
        if (batch_size > 0 && seq_len > 0 && kv_out > 0) {
            array k_proj_raw_f32 = (k.dtype() == float32) ? k : astype(k, float32);
            array k_proj_raw_flat = reshape(k_proj_raw_f32, {batch_size * seq_len * kv_out});
            eval(k_proj_raw_flat);
            const float* k_proj_raw_ptr = k_proj_raw_flat.data<float>();
            std::vector<float> trace_k_proj_raw_host(
                static_cast<size_t>(seq_len) * static_cast<size_t>(kv_out)
            );
            const int batch_idx = 0;
            for (int s = 0; s < seq_len; ++s) {
                for (int d = 0; d < kv_out; ++d) {
                    const size_t src =
                        ((static_cast<size_t>(batch_idx) * static_cast<size_t>(seq_len) + static_cast<size_t>(s)) * static_cast<size_t>(kv_out)) +
                        static_cast<size_t>(d);
                    const size_t dst =
                        static_cast<size_t>(s) * static_cast<size_t>(kv_out) +
                        static_cast<size_t>(d);
                    trace_k_proj_raw_host[dst] = k_proj_raw_ptr[src];
                }
            }
            array trace_k_proj_raw = owned_f32_array(trace_k_proj_raw_host, Shape{1, seq_len, kv_out});
            *out_attn_k_proj_raw = pool_array(std::move(trace_k_proj_raw));
        } else {
            *out_attn_k_proj_raw = nullptr;
        }
    }

    std::optional<array> query_gate_values = std::nullopt;
    if (query_gate) {
        // Mirror CPU compactQueryProjection/applyOutputGateInPlace layout exactly:
        // per head lane is packed as [q(head_dim) | gate(head_dim)].
        array q_packed_f32 = (q.dtype() == float32) ? q : astype(q, float32);
        array q_packed_flat = reshape(q_packed_f32, {batch_size * seq_len * q_proj_out});
        eval(q_packed_flat);
        const float* q_packed_ptr = q_packed_flat.data<float>();
        std::vector<float> q_values_host(static_cast<size_t>(batch_size) * static_cast<size_t>(seq_len) * static_cast<size_t>(q_out), 0.0f);
        std::vector<float> q_gate_host(static_cast<size_t>(batch_size) * static_cast<size_t>(seq_len) * static_cast<size_t>(q_out), 0.0f);

        for (int b = 0; b < batch_size; ++b) {
            for (int s = 0; s < seq_len; ++s) {
                const size_t packed_row_base =
                    (static_cast<size_t>(b) * static_cast<size_t>(seq_len) + static_cast<size_t>(s)) * static_cast<size_t>(q_proj_out);
                const size_t q_row_base =
                    (static_cast<size_t>(b) * static_cast<size_t>(seq_len) + static_cast<size_t>(s)) * static_cast<size_t>(q_out);
                for (int h = 0; h < n_heads_int; ++h) {
                    const size_t src_head_base = packed_row_base + static_cast<size_t>(h) * static_cast<size_t>(2 * head_dim_int);
                    const size_t dst_head_base = q_row_base + static_cast<size_t>(h) * static_cast<size_t>(head_dim_int);
                    for (int d = 0; d < head_dim_int; ++d) {
                        q_values_host[dst_head_base + static_cast<size_t>(d)] =
                            q_packed_ptr[src_head_base + static_cast<size_t>(d)];
                        q_gate_host[dst_head_base + static_cast<size_t>(d)] =
                            q_packed_ptr[src_head_base + static_cast<size_t>(head_dim_int + d)];
                    }
                }
            }
        }

        q = owned_f32_array(q_values_host, Shape{batch_size, seq_len, n_heads_int, head_dim_int});
        query_gate_values = owned_f32_array(q_gate_host, Shape{batch_size, seq_len, n_heads_int, head_dim_int});
    } else {
        q = reshape(q, {batch_size, seq_len, n_heads_int, head_dim_int});
    }
    k = reshape(k, {batch_size, seq_len, n_kv_heads_int, head_dim_int});
    v = reshape(v, {batch_size, seq_len, n_kv_heads_int, head_dim_int});

    q = transpose(q, {0, 2, 1, 3});
    k = transpose(k, {0, 2, 1, 3});
    v = transpose(v, {0, 2, 1, 3});
    if (query_gate_values.has_value()) {
        query_gate_values = transpose(query_gate_values.value(), {0, 2, 1, 3});
    }

    maybe_apply_qk_norm(&q, &k, q_norm_w, k_norm_w, rms_eps);
    if (out_attn_q_norm != nullptr) {
        const int q_len = q.shape(2);
        if (batch_size > 0 and q_len > 0 and n_heads_int > 0 and head_dim_int > 0) {
            array q_norm_f32 = (q.dtype() == float32) ? q : astype(q, float32);
            array q_norm_flat = reshape(q_norm_f32, {batch_size * n_heads_int * q_len * head_dim_int});
            eval(q_norm_flat);
            const float* q_norm_ptr = q_norm_flat.data<float>();
            std::vector<float> trace_q_norm_host(
                static_cast<size_t>(q_len) * static_cast<size_t>(n_heads_int) * static_cast<size_t>(head_dim_int)
            );
            const int batch_idx = 0;
            for (int s = 0; s < q_len; ++s) {
                for (int h = 0; h < n_heads_int; ++h) {
                    for (int d = 0; d < head_dim_int; ++d) {
                        const size_t src =
                            (((static_cast<size_t>(batch_idx) * static_cast<size_t>(n_heads_int) + static_cast<size_t>(h)) * static_cast<size_t>(q_len) + static_cast<size_t>(s)) *
                             static_cast<size_t>(head_dim_int)) +
                            static_cast<size_t>(d);
                        const size_t dst =
                            (static_cast<size_t>(s) * static_cast<size_t>(n_heads_int) + static_cast<size_t>(h)) * static_cast<size_t>(head_dim_int) +
                            static_cast<size_t>(d);
                        trace_q_norm_host[dst] = q_norm_ptr[src];
                    }
                }
            }
            array trace_q_norm = owned_f32_array(trace_q_norm_host, Shape{1, q_len, n_heads_int * head_dim_int});
            *out_attn_q_norm = pool_array(std::move(trace_q_norm));
        } else {
            *out_attn_q_norm = nullptr;
        }
    }
    if (out_attn_k_norm != nullptr) {
        const int k_len = k.shape(2);
        if (batch_size > 0 and k_len > 0 and n_kv_heads_int > 0 and head_dim_int > 0) {
            array k_norm_f32 = (k.dtype() == float32) ? k : astype(k, float32);
            array k_norm_flat = reshape(k_norm_f32, {batch_size * n_kv_heads_int * k_len * head_dim_int});
            eval(k_norm_flat);
            const float* k_norm_ptr = k_norm_flat.data<float>();
            std::vector<float> trace_k_norm_host(
                static_cast<size_t>(k_len) * static_cast<size_t>(n_kv_heads_int) * static_cast<size_t>(head_dim_int)
            );
            const int batch_idx = 0;
            for (int s = 0; s < k_len; ++s) {
                for (int h = 0; h < n_kv_heads_int; ++h) {
                    for (int d = 0; d < head_dim_int; ++d) {
                        const size_t src =
                            (((static_cast<size_t>(batch_idx) * static_cast<size_t>(n_kv_heads_int) + static_cast<size_t>(h)) * static_cast<size_t>(k_len) + static_cast<size_t>(s)) *
                             static_cast<size_t>(head_dim_int)) +
                            static_cast<size_t>(d);
                        const size_t dst =
                            (static_cast<size_t>(s) * static_cast<size_t>(n_kv_heads_int) + static_cast<size_t>(h)) * static_cast<size_t>(head_dim_int) +
                            static_cast<size_t>(d);
                        trace_k_norm_host[dst] = k_norm_ptr[src];
                    }
                }
            }
            array trace_k_norm = owned_f32_array(trace_k_norm_host, Shape{1, k_len, n_kv_heads_int * head_dim_int});
            *out_attn_k_norm = pool_array(std::move(trace_k_norm));
        } else {
            *out_attn_k_norm = nullptr;
        }
    }
    apply_attention_rope(
        &q,
        &k,
        seq_len,
        head_dim_int,
        rope_dim_int,
        pos_offset,
        rope_theta,
        rope_interleaved,
        runtime_rope_cos,
        runtime_rope_sin,
        runtime_rope_dim
    );
    if (out_attn_q_rope != nullptr) {
        const int q_len = q.shape(2);
        if (batch_size > 0 and q_len > 0 and n_heads_int > 0 and head_dim_int > 0) {
            array q_rope_f32 = (q.dtype() == float32) ? q : astype(q, float32);
            array q_rope_flat = reshape(q_rope_f32, {batch_size * n_heads_int * q_len * head_dim_int});
            eval(q_rope_flat);
            const float* q_rope_ptr = q_rope_flat.data<float>();
            std::vector<float> trace_q_rope_host(
                static_cast<size_t>(q_len) * static_cast<size_t>(n_heads_int) * static_cast<size_t>(head_dim_int)
            );
            const int batch_idx = 0;
            for (int s = 0; s < q_len; ++s) {
                for (int h = 0; h < n_heads_int; ++h) {
                    for (int d = 0; d < head_dim_int; ++d) {
                        const size_t src =
                            (((static_cast<size_t>(batch_idx) * static_cast<size_t>(n_heads_int) + static_cast<size_t>(h)) * static_cast<size_t>(q_len) + static_cast<size_t>(s)) *
                             static_cast<size_t>(head_dim_int)) +
                            static_cast<size_t>(d);
                        const size_t dst =
                            (static_cast<size_t>(s) * static_cast<size_t>(n_heads_int) + static_cast<size_t>(h)) * static_cast<size_t>(head_dim_int) +
                            static_cast<size_t>(d);
                        trace_q_rope_host[dst] = q_rope_ptr[src];
                    }
                }
            }
            array trace_q_rope = owned_f32_array(trace_q_rope_host, Shape{1, q_len, n_heads_int * head_dim_int});
            *out_attn_q_rope = pool_array(std::move(trace_q_rope));
        } else {
            *out_attn_q_rope = nullptr;
        }
    }
    if (out_attn_k_rope != nullptr) {
        const int k_len = k.shape(2);
        if (batch_size > 0 and k_len > 0 and n_kv_heads_int > 0 and head_dim_int > 0) {
            array k_rope_f32 = (k.dtype() == float32) ? k : astype(k, float32);
            array k_rope_flat = reshape(k_rope_f32, {batch_size * n_kv_heads_int * k_len * head_dim_int});
            eval(k_rope_flat);
            const float* k_rope_ptr = k_rope_flat.data<float>();
            std::vector<float> trace_k_rope_host(
                static_cast<size_t>(k_len) * static_cast<size_t>(n_kv_heads_int) * static_cast<size_t>(head_dim_int)
            );
            const int batch_idx = 0;
            for (int s = 0; s < k_len; ++s) {
                for (int h = 0; h < n_kv_heads_int; ++h) {
                    for (int d = 0; d < head_dim_int; ++d) {
                        const size_t src =
                            (((static_cast<size_t>(batch_idx) * static_cast<size_t>(n_kv_heads_int) + static_cast<size_t>(h)) * static_cast<size_t>(k_len) + static_cast<size_t>(s)) *
                             static_cast<size_t>(head_dim_int)) +
                            static_cast<size_t>(d);
                        const size_t dst =
                            (static_cast<size_t>(s) * static_cast<size_t>(n_kv_heads_int) + static_cast<size_t>(h)) * static_cast<size_t>(head_dim_int) +
                            static_cast<size_t>(d);
                        trace_k_rope_host[dst] = k_rope_ptr[src];
                    }
                }
            }
            array trace_k_rope = owned_f32_array(trace_k_rope_host, Shape{1, k_len, n_kv_heads_int * head_dim_int});
            *out_attn_k_rope = pool_array(std::move(trace_k_rope));
        } else {
            *out_attn_k_rope = nullptr;
        }
    }
    if (out_attn_q != nullptr) {
        const int q_len = q.shape(2);
        if (batch_size > 0 and q_len > 0 and n_heads_int > 0 and head_dim_int > 0) {
            array q_f32 = (q.dtype() == float32) ? q : astype(q, float32);
            array q_flat = reshape(q_f32, {batch_size * n_heads_int * q_len * head_dim_int});
            eval(q_flat);
            const float* q_ptr = q_flat.data<float>();
            std::vector<float> trace_q_host(
                static_cast<size_t>(q_len) * static_cast<size_t>(n_heads_int) * static_cast<size_t>(head_dim_int)
            );
            const int batch_idx = 0;
            for (int s = 0; s < q_len; ++s) {
                for (int h = 0; h < n_heads_int; ++h) {
                    for (int d = 0; d < head_dim_int; ++d) {
                        const size_t src =
                            (((static_cast<size_t>(batch_idx) * static_cast<size_t>(n_heads_int) + static_cast<size_t>(h)) * static_cast<size_t>(q_len) + static_cast<size_t>(s)) *
                             static_cast<size_t>(head_dim_int)) +
                            static_cast<size_t>(d);
                        const size_t dst =
                            (static_cast<size_t>(s) * static_cast<size_t>(n_heads_int) + static_cast<size_t>(h)) * static_cast<size_t>(head_dim_int) +
                            static_cast<size_t>(d);
                        trace_q_host[dst] = q_ptr[src];
                    }
                }
            }
            array trace_q = owned_f32_array(trace_q_host, Shape{1, q_len, n_heads_int * head_dim_int});
            *out_attn_q = pool_array(std::move(trace_q));
        } else {
            *out_attn_q = nullptr;
        }
    }
    if (out_attn_k != nullptr) {
        const int k_len = k.shape(2);
        if (batch_size > 0 and k_len > 0 and n_kv_heads_int > 0 and head_dim_int > 0) {
            array k_f32 = (k.dtype() == float32) ? k : astype(k, float32);
            array k_flat = reshape(k_f32, {batch_size * n_kv_heads_int * k_len * head_dim_int});
            eval(k_flat);
            const float* k_ptr = k_flat.data<float>();
            std::vector<float> trace_k_host(
                static_cast<size_t>(k_len) * static_cast<size_t>(n_kv_heads_int) * static_cast<size_t>(head_dim_int)
            );
            const int batch_idx = 0;
            for (int s = 0; s < k_len; ++s) {
                for (int h = 0; h < n_kv_heads_int; ++h) {
                    for (int d = 0; d < head_dim_int; ++d) {
                        const size_t src =
                            (((static_cast<size_t>(batch_idx) * static_cast<size_t>(n_kv_heads_int) + static_cast<size_t>(h)) * static_cast<size_t>(k_len) + static_cast<size_t>(s)) *
                             static_cast<size_t>(head_dim_int)) +
                            static_cast<size_t>(d);
                        const size_t dst =
                            (static_cast<size_t>(s) * static_cast<size_t>(n_kv_heads_int) + static_cast<size_t>(h)) * static_cast<size_t>(head_dim_int) +
                            static_cast<size_t>(d);
                        trace_k_host[dst] = k_ptr[src];
                    }
                }
            }
            array trace_k = owned_f32_array(trace_k_host, Shape{1, k_len, n_kv_heads_int * head_dim_int});
            *out_attn_k = pool_array(std::move(trace_k));
        } else {
            *out_attn_k = nullptr;
        }
    }

    array k_for_attn = k;
    array v_for_attn = v;
    bool is_prefill = true;
    resolve_attention_cache(
        cache_ptr,
        layer_idx,
        &k,
        &v,
        &k_for_attn,
        &v_for_attn,
        &is_prefill
    );

    // Explicit score/softmax/value reduction for both prefill and decode.
    // This keeps reference behavior independent from fused SDPA kernel internals.
    array k_expanded = k_for_attn;
    array v_expanded = v_for_attn;
    gqa_expand_attention_kv(q, k_for_attn, v_for_attn, &k_expanded, &v_expanded);
    array scores = cpu_like_attention_scores(
        q,
        k_expanded,
        batch_size,
        n_heads_int,
        q.shape(2),
        k_expanded.shape(2),
        head_dim_int,
        scale_value,
        is_prefill,
        sliding_window_int
    );
    const int q_len = scores.shape(2);
    const int k_len = scores.shape(3);
    if (out_attn_qk != nullptr) {
        if (q_len > 0 and k_len > 0 and n_heads_int > 0 and n_kv_heads_int > 0) {
            array trace_scores = cpu_like_trace_raw_qk_last_row(
                q,
                k_for_attn,
                batch_size,
                n_heads_int,
                n_kv_heads_int,
                q_len,
                k_len,
                head_dim_int,
                scale_value,
                is_prefill,
                sliding_window_int
            );
            *out_attn_qk = pool_array(std::move(trace_scores));
        } else {
            *out_attn_qk = nullptr;
        }
    }
    const array weights = cpu_like_softmax_scores(scores, batch_size, n_heads_int, q_len, k_len, attn_sinks);
    if (out_attn_weights != nullptr) {
        if (q_len > 0 and k_len > 0 and n_heads_int > 0) {
            // Emit CPU-contract-compatible attn.weights checkpoint:
            // last query row for batch=0, all heads -> shape [n_heads, k_len].
            array weights_f32 = (weights.dtype() == float32) ? weights : astype(weights, float32);
            array flat_weights = reshape(weights_f32, {batch_size * n_heads_int * q_len * k_len});
            eval(flat_weights);
            const float* weights_ptr = flat_weights.data<float>();

            std::vector<float> trace_host(static_cast<size_t>(n_heads_int) * static_cast<size_t>(k_len));
            const int batch_idx = 0;
            const int q_idx = q_len - 1;
            for (int head = 0; head < n_heads_int; ++head) {
                for (int k_idx = 0; k_idx < k_len; ++k_idx) {
                    const size_t src =
                        (((static_cast<size_t>(batch_idx) * static_cast<size_t>(n_heads_int) + static_cast<size_t>(head)) * static_cast<size_t>(q_len) + static_cast<size_t>(q_idx)) *
                         static_cast<size_t>(k_len)) +
                        static_cast<size_t>(k_idx);
                    const size_t dst = static_cast<size_t>(head) * static_cast<size_t>(k_len) + static_cast<size_t>(k_idx);
                    trace_host[dst] = weights_ptr[src];
                }
            }
            array trace_weights = owned_f32_array(trace_host, Shape{n_heads_int, k_len});
            *out_attn_weights = pool_array(std::move(trace_weights));
        } else {
            *out_attn_weights = nullptr;
        }
    }
    array attn_out = matmul(weights, v_expanded);
    attn_out = apply_query_output_gate_cpu_like(attn_out, query_gate_values);
    attn_out = reshape_attention_output(attn_out, batch_size, seq_len, q_out);

    auto out = cpu_like_linear_lastdim(attn_out, o_w_ref);
    if (o_bias != nullptr) {
        const auto& bias = *static_cast<const array*>(o_bias);
        out = out + cast_bias_cached(o_bias, bias, out.dtype());
    }
    return pool_array(std::move(out));
}

void* mlx_lazy_attention_reference_bf16(
    const void* input,
    const void* q_w, const void* k_w, const void* v_w, const void* o_w,
    const void* q_norm_w,
    const void* k_norm_w,
    const void* q_bias,
    const void* k_bias,
    const void* v_bias,
    const void* o_bias,
    const void* attn_sinks,
    void* cache_ptr, size_t layer_idx,
    size_t n_heads, size_t n_kv_heads, size_t head_dim,
    size_t rope_dim,
    size_t pos_offset, float rope_theta,
    bool rope_interleaved,
    const void* runtime_rope_cos, const void* runtime_rope_sin, size_t runtime_rope_dim,
    float rms_eps,
    float query_pre_attn_scalar,
    float attention_multiplier,
    size_t sliding_window,
    bool query_gate
) {
    return mlx_lazy_attention_reference_bf16_impl(
        input,
        q_w, k_w, v_w, o_w,
        q_norm_w,
        k_norm_w,
        q_bias,
        k_bias,
        v_bias,
        o_bias,
        attn_sinks,
        cache_ptr, layer_idx,
        n_heads, n_kv_heads, head_dim, rope_dim,
        pos_offset, rope_theta,
        rope_interleaved,
        runtime_rope_cos, runtime_rope_sin, runtime_rope_dim,
        rms_eps,
        query_pre_attn_scalar,
        attention_multiplier,
        sliding_window,
        query_gate,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr
    );
}

void* mlx_lazy_attention_reference_bf16_capture(
    const void* input,
    const void* q_w, const void* k_w, const void* v_w, const void* o_w,
    const void* q_norm_w,
    const void* k_norm_w,
    const void* q_bias,
    const void* k_bias,
    const void* v_bias,
    const void* o_bias,
    const void* attn_sinks,
    void* cache_ptr, size_t layer_idx,
    size_t n_heads, size_t n_kv_heads, size_t head_dim,
    size_t rope_dim,
    size_t pos_offset, float rope_theta,
    bool rope_interleaved,
    const void* runtime_rope_cos, const void* runtime_rope_sin, size_t runtime_rope_dim,
    float rms_eps,
    float query_pre_attn_scalar,
    float attention_multiplier,
    size_t sliding_window,
    bool query_gate,
    void** out_attn_q_proj_raw,
    void** out_attn_k_proj_raw,
    void** out_attn_q_norm,
    void** out_attn_k_norm,
    void** out_attn_q_rope,
    void** out_attn_k_rope,
    void** out_attn_qk,
    void** out_attn_weights,
    void** out_attn_q,
    void** out_attn_k
) {
    return mlx_lazy_attention_reference_bf16_impl(
        input,
        q_w, k_w, v_w, o_w,
        q_norm_w,
        k_norm_w,
        q_bias,
        k_bias,
        v_bias,
        o_bias,
        attn_sinks,
        cache_ptr, layer_idx,
        n_heads, n_kv_heads, head_dim, rope_dim,
        pos_offset, rope_theta,
        rope_interleaved,
        runtime_rope_cos, runtime_rope_sin, runtime_rope_dim,
        rms_eps,
        query_pre_attn_scalar,
        attention_multiplier,
        sliding_window,
        query_gate,
        out_attn_q_proj_raw,
        out_attn_k_proj_raw,
        out_attn_q_norm,
        out_attn_k_norm,
        out_attn_q_rope,
        out_attn_k_rope,
        out_attn_qk,
        out_attn_weights,
        out_attn_q,
        out_attn_k
    );
}
