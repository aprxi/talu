#include <immintrin.h>
#include <stdint.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static inline float hsum256_ps(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    vlow = _mm_hadd_ps(vlow, vlow);
    vlow = _mm_hadd_ps(vlow, vlow);
    return _mm_cvtss_f32(vlow);
}

static float sigmoid_lut[4096];
static int sigmoid_lut_init = 0;

static inline void init_sigmoid_lut(void) {
    if (sigmoid_lut_init) {
        return;
    }
#pragma omp critical
    {
        if (!sigmoid_lut_init) {
            const float min_x = -12.0f;
            const float max_x = 12.0f;
            const float step = (max_x - min_x) / 4095.0f;
            for (int i = 0; i < 4096; ++i) {
                float x = min_x + step * (float)i;
                sigmoid_lut[i] = 1.0f / (1.0f + expf(-x));
            }
            sigmoid_lut_init = 1;
        }
    }
}

static inline float fast_sigmoid(float x) {
    const float min_x = -12.0f;
    const float max_x = 12.0f;
    if (x <= min_x) {
        return 0.0f;
    }
    if (x >= max_x) {
        return 1.0f;
    }
    const float scale = 4095.0f / (max_x - min_x);
    int idx = (int)((x - min_x) * scale);
    return sigmoid_lut[idx];
}

static inline float dot_block32_mxfp4(const float *x, const uint8_t *w, float scale) {
    const __m128i packed = _mm_loadu_si128((const __m128i *)w);
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i low = _mm_and_si128(packed, mask);
    const __m128i high = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
    const __m128i lo_inter = _mm_unpacklo_epi8(low, high);
    const __m128i hi_inter = _mm_unpackhi_epi8(low, high);
    const __m256i indices = _mm256_set_m128i(hi_inter, lo_inter);

    const __m256i lut = _mm256_setr_epi8(
        0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
        0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m256i vals_i8 = _mm256_shuffle_epi8(lut, indices);

    const __m128i lo128 = _mm256_castsi256_si128(vals_i8);
    const __m128i hi128 = _mm256_extracti128_si256(vals_i8, 1);
    const __m256i lo_i16 = _mm256_cvtepi8_epi16(lo128);
    const __m256i hi_i16 = _mm256_cvtepi8_epi16(hi128);

    const __m128i lo_i16_low = _mm256_castsi256_si128(lo_i16);
    const __m128i lo_i16_high = _mm256_extracti128_si256(lo_i16, 1);
    const __m128i hi_i16_low = _mm256_castsi256_si128(hi_i16);
    const __m128i hi_i16_high = _mm256_extracti128_si256(hi_i16, 1);

    __m256 v0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(lo_i16_low));
    __m256 v1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(lo_i16_high));
    __m256 v2 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(hi_i16_low));
    __m256 v3 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(hi_i16_high));

    const __m256 scale_vec = _mm256_set1_ps(scale * 0.5f);
    v0 = _mm256_mul_ps(v0, scale_vec);
    v1 = _mm256_mul_ps(v1, scale_vec);
    v2 = _mm256_mul_ps(v2, scale_vec);
    v3 = _mm256_mul_ps(v3, scale_vec);

    const __m256 x0 = _mm256_loadu_ps(x);
    const __m256 x1 = _mm256_loadu_ps(x + 8);
    const __m256 x2 = _mm256_loadu_ps(x + 16);
    const __m256 x3 = _mm256_loadu_ps(x + 24);

    __m256 acc = _mm256_mul_ps(x0, v0);
    acc = _mm256_add_ps(acc, _mm256_mul_ps(x1, v1));
    acc = _mm256_add_ps(acc, _mm256_mul_ps(x2, v2));
    acc = _mm256_add_ps(acc, _mm256_mul_ps(x3, v3));
    return hsum256_ps(acc);
}

void mxfp4_matmul(const float *input,
                  const uint8_t *weights,
                  const float *scales,
                  float *output,
                  int batch,
                  int in_features,
                  int out_features,
                  int blocks_per_feature) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < batch; ++b) {
        for (int out = 0; out < out_features; ++out) {
            const float *x = input + (size_t)b * (size_t)in_features;
            float sum = 0.0f;
            const uint8_t *w = weights + ((size_t)out * (size_t)blocks_per_feature * 16);
            const float *s = scales + ((size_t)out * (size_t)blocks_per_feature);
            for (int block = 0; block < blocks_per_feature; ++block) {
                sum += dot_block32_mxfp4(x + block * 32, w + block * 16, s[block]);
            }
            output[(size_t)b * (size_t)out_features + (size_t)out] = sum;
        }
    }
}

void mxfp4_matmul_expert_list(const float *input,
                             const uint8_t *weights,
                             const float *scales,
                             const int32_t *expert_indices,
                             float *output,
                             int batch,
                             int in_features,
                             int out_features,
                             int blocks_per_feature,
                             int top_k,
                             int expert_stride_blocks,
                             int expert_stride_scales) {
    const int work_items = batch * top_k;
#pragma omp parallel for if(work_items >= 8) collapse(2) schedule(static)
    for (int b = 0; b < batch; ++b) {
        for (int k = 0; k < top_k; ++k) {
            const int32_t expert = expert_indices[(size_t)b * (size_t)top_k + (size_t)k];
            const uint8_t *w_base = weights + (size_t)expert * (size_t)expert_stride_blocks;
            const float *s_base = scales + (size_t)expert * (size_t)expert_stride_scales;
            const float *x = input + (size_t)b * (size_t)in_features;
            float *y = output + ((size_t)b * (size_t)top_k + (size_t)k) * (size_t)out_features;
            for (int out = 0; out < out_features; ++out) {
                float sum = 0.0f;
                const uint8_t *w = w_base + (size_t)out * (size_t)blocks_per_feature * 16;
                const float *s = s_base + (size_t)out * (size_t)blocks_per_feature;
                for (int block = 0; block < blocks_per_feature; ++block) {
                    sum += dot_block32_mxfp4(x + block * 32, w + block * 16, s[block]);
                }
                y[out] = sum;
            }
        }
    }
}

void mxfp4_moe_fused(const float *input,
                     const uint8_t *gate_up_blocks,
                     const float *gate_up_scales,
                     const float *gate_up_bias,
                     const uint8_t *down_blocks,
                     const float *down_scales,
                     const float *down_bias,
                     const int32_t *expert_indices,
                     const float *routing_weights,
                     float *output,
                     int batch,
                     int in_features,
                     int expert_dim,
                     int hidden_size,
                     int blocks_per_feature_gate,
                     int blocks_per_feature_down,
                     int top_k,
                     int expert_stride_gate_blocks,
                     int expert_stride_gate_scales,
                     int expert_stride_down_blocks,
                     int expert_stride_down_scales) {
    const int gate_up_out = expert_dim * 2;
    const float alpha = 1.702f;
    const float limit = 7.0f;

    init_sigmoid_lut();

#pragma omp parallel for schedule(static)
    for (int b = 0; b < batch; ++b) {
        const float *x = input + (size_t)b * (size_t)in_features;
        float *y = output + (size_t)b * (size_t)hidden_size;
        for (int out = 0; out < hidden_size; ++out) {
            y[out] = 0.0f;
        }

        for (int k = 0; k < top_k; ++k) {
            const int32_t expert = expert_indices[(size_t)b * (size_t)top_k + (size_t)k];
            const uint8_t *gate_w_base = gate_up_blocks + (size_t)expert * (size_t)expert_stride_gate_blocks;
            const float *gate_s_base = gate_up_scales + (size_t)expert * (size_t)expert_stride_gate_scales;
            const float *gate_bias = gate_up_bias + (size_t)expert * (size_t)gate_up_out;

            float gate_up_buf[gate_up_out];
            for (int out = 0; out < gate_up_out; ++out) {
                float sum = 0.0f;
                const uint8_t *w = gate_w_base + (size_t)out * (size_t)blocks_per_feature_gate * 16;
                const float *s = gate_s_base + (size_t)out * (size_t)blocks_per_feature_gate;
                for (int block = 0; block < blocks_per_feature_gate; ++block) {
                    sum += dot_block32_mxfp4(x + block * 32, w + block * 16, s[block]);
                }
                gate_up_buf[out] = sum;
            }

            for (int i = 0; i < expert_dim; ++i) {
                float gate = gate_up_buf[2 * i] + gate_bias[2 * i];
                float up = gate_up_buf[2 * i + 1] + gate_bias[2 * i + 1];
                if (gate > limit) {
                    gate = limit;
                }
                if (up > limit) {
                    up = limit;
                } else if (up < -limit) {
                    up = -limit;
                }
                float gated = gate * fast_sigmoid(gate * alpha);
                gate_up_buf[i] = (up + 1.0f) * gated;
            }

            const uint8_t *down_w_base = down_blocks + (size_t)expert * (size_t)expert_stride_down_blocks;
            const float *down_s_base = down_scales + (size_t)expert * (size_t)expert_stride_down_scales;
            const float *down_b = down_bias + (size_t)expert * (size_t)hidden_size;
            const float weight = routing_weights[(size_t)b * (size_t)top_k + (size_t)k];

            for (int out = 0; out < hidden_size; ++out) {
                float sum = 0.0f;
                const uint8_t *w = down_w_base + (size_t)out * (size_t)blocks_per_feature_down * 16;
                const float *s = down_s_base + (size_t)out * (size_t)blocks_per_feature_down;
                for (int block = 0; block < blocks_per_feature_down; ++block) {
                    sum += dot_block32_mxfp4(gate_up_buf + block * 32, w + block * 16, s[block]);
                }
                y[out] += weight * (sum + down_b[out]);
            }
        }
    }
}
