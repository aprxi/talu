// Gated-delta reference recurrence path (correctness-first, not optimized).
//
// This path is intentionally explicit and separate from the custom fused
// recurrence kernel. It is toggled at runtime by:
//   TALU_METAL_GDELTA_REFERENCE=1
// or globally by:
//   TALU_METAL_ATTN_REFERENCE=1

static inline bool gated_delta_reference_enabled() {
    static thread_local int cached = -1;
    if (cached >= 0) {
        return cached != 0;
    }
    const char* raw = std::getenv("TALU_METAL_GDELTA_REFERENCE");
    if (raw == nullptr || raw[0] == '\0') {
        raw = std::getenv("TALU_METAL_ATTN_REFERENCE");
    }
    if (raw == nullptr || raw[0] == '\0') {
        cached = 0;
        return false;
    }
    if (std::strcmp(raw, "0") == 0 ||
        std::strcmp(raw, "false") == 0 ||
        std::strcmp(raw, "FALSE") == 0 ||
        std::strcmp(raw, "off") == 0 ||
        std::strcmp(raw, "OFF") == 0 ||
        std::strcmp(raw, "no") == 0 ||
        std::strcmp(raw, "NO") == 0) {
        cached = 0;
        return false;
    }
    cached = 1;
    return true;
}

static inline void gated_delta_reference_emit_once() {
    static thread_local bool emitted = false;
    if (emitted) return;
    emitted = true;
    std::fprintf(stderr, "GDELTA_REFERENCE: enabled\n");
}

struct GatedDeltaReferenceStep {
    array ssm;
    array norm;
    array next_state;
};

static inline float gated_delta_fast_exp_scalar(float x) {
    constexpr float log2e = 1.4426950408889634f;
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
    const int fxi = static_cast<int>(fx);
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

static inline float gated_delta_fast_sigmoid_scalar(float x) {
    return 1.0f / (1.0f + gated_delta_fast_exp_scalar(-x));
}

static inline float gated_delta_exact_silu_scalar(float x) {
    return x / (1.0f + std::exp(-x));
}

static inline float gated_delta_fast_softplus_scalar(float x) {
    if (x > 20.0f) return x;
    return std::log(1.0f + gated_delta_fast_exp_scalar(x));
}

static inline std::vector<float> to_host_f32_vector(const array& src) {
    const array casted = (src.dtype() == float32) ? src : astype(src, float32);
    eval(casted);
    const size_t n = casted.size();
    std::vector<float> out(n);
    if (n > 0) {
        std::memcpy(out.data(), casted.data<float>(), n * sizeof(float));
    }
    return out;
}

static inline float state_space_cpu_like_dot_row(
    const float* a,
    const float* b,
    int len
) {
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    int idx = 0;

    while (idx + 16 <= len) {
        sum0 = std::fma(a[idx + 0], b[idx + 0], sum0);
        sum1 = std::fma(a[idx + 4], b[idx + 4], sum1);
        sum2 = std::fma(a[idx + 8], b[idx + 8], sum2);
        sum3 = std::fma(a[idx + 12], b[idx + 12], sum3);
        sum0 = std::fma(a[idx + 1], b[idx + 1], sum0);
        sum1 = std::fma(a[idx + 5], b[idx + 5], sum1);
        sum2 = std::fma(a[idx + 9], b[idx + 9], sum2);
        sum3 = std::fma(a[idx + 13], b[idx + 13], sum3);
        sum0 = std::fma(a[idx + 2], b[idx + 2], sum0);
        sum1 = std::fma(a[idx + 6], b[idx + 6], sum1);
        sum2 = std::fma(a[idx + 10], b[idx + 10], sum2);
        sum3 = std::fma(a[idx + 14], b[idx + 14], sum3);
        sum0 = std::fma(a[idx + 3], b[idx + 3], sum0);
        sum1 = std::fma(a[idx + 7], b[idx + 7], sum1);
        sum2 = std::fma(a[idx + 11], b[idx + 11], sum2);
        sum3 = std::fma(a[idx + 15], b[idx + 15], sum3);
        idx += 16;
    }

    while (idx + 4 <= len) {
        sum0 = std::fma(a[idx + 0], b[idx + 0], sum0);
        sum1 = std::fma(a[idx + 1], b[idx + 1], sum1);
        sum2 = std::fma(a[idx + 2], b[idx + 2], sum2);
        sum3 = std::fma(a[idx + 3], b[idx + 3], sum3);
        idx += 4;
    }

    float sum = (sum0 + sum1) + (sum2 + sum3);
    for (; idx < len; ++idx) {
        sum = std::fma(a[idx], b[idx], sum);
    }
    return sum;
}

static inline float truncate_to_bf16_f32(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(uint32_t));
    const uint32_t truncated = bits & 0xFFFF0000u;
    float out = 0.0f;
    std::memcpy(&out, &truncated, sizeof(float));
    return out;
}


static inline array owned_array_from_f32_vector(std::vector<float>&& values, const Shape& shape) {
    static constexpr size_t kAlignment = 16 * 1024;
    const size_t byte_count = std::max(values.size() * sizeof(float), sizeof(float));
    void* aligned_ptr = nullptr;
    if (posix_memalign(&aligned_ptr, kAlignment, byte_count) != 0 || aligned_ptr == nullptr) {
        throw std::bad_alloc();
    }
    if (!values.empty()) {
        std::memcpy(aligned_ptr, values.data(), values.size() * sizeof(float));
    }
    auto owner = std::shared_ptr<void>(aligned_ptr, [](void* ptr) {
        std::free(ptr);
    });
    auto deleter = [owner](void*) {};
    return array(static_cast<float*>(aligned_ptr), shape, float32, deleter);
}

static inline float state_space_cpu_like_bf16_dot_row(
    const float* a,
    const float* b,
    int len
) {
    auto dot8_into_lanes = [](float lanes[4], const float* lhs, const float* rhs) {
        const float a0 = truncate_to_bf16_f32(lhs[0]);
        const float a1 = truncate_to_bf16_f32(lhs[1]);
        const float a2 = truncate_to_bf16_f32(lhs[2]);
        const float a3 = truncate_to_bf16_f32(lhs[3]);
        const float a4 = truncate_to_bf16_f32(lhs[4]);
        const float a5 = truncate_to_bf16_f32(lhs[5]);
        const float a6 = truncate_to_bf16_f32(lhs[6]);
        const float a7 = truncate_to_bf16_f32(lhs[7]);
        const float b0 = truncate_to_bf16_f32(rhs[0]);
        const float b1 = truncate_to_bf16_f32(rhs[1]);
        const float b2 = truncate_to_bf16_f32(rhs[2]);
        const float b3 = truncate_to_bf16_f32(rhs[3]);
        const float b4 = truncate_to_bf16_f32(rhs[4]);
        const float b5 = truncate_to_bf16_f32(rhs[5]);
        const float b6 = truncate_to_bf16_f32(rhs[6]);
        const float b7 = truncate_to_bf16_f32(rhs[7]);

        lanes[0] = std::fma(a0, b0, lanes[0]);
        lanes[0] = std::fma(a1, b1, lanes[0]);
        lanes[1] = std::fma(a2, b2, lanes[1]);
        lanes[1] = std::fma(a3, b3, lanes[1]);
        lanes[2] = std::fma(a4, b4, lanes[2]);
        lanes[2] = std::fma(a5, b5, lanes[2]);
        lanes[3] = std::fma(a6, b6, lanes[3]);
        lanes[3] = std::fma(a7, b7, lanes[3]);
    };

    float acc0[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float acc1[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float acc2[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float acc3[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    int idx = 0;

    // Match CPU's 32-value unrolled BFDOT structure:
    // four independent 8-element lane accumulators, then a single final reduce.
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
            truncate_to_bf16_f32(a[idx]),
            truncate_to_bf16_f32(b[idx]),
            sum
        );
    }
    return sum;
}

static inline float state_space_cpu_like_bf16_dot_row_single_row(
    const float* a,
    const float* b,
    int len
) {
    auto dot8_into_lanes = [](float lanes[4], const float* lhs, const float* rhs) {
        const float a0 = truncate_to_bf16_f32(lhs[0]);
        const float a1 = truncate_to_bf16_f32(lhs[1]);
        const float a2 = truncate_to_bf16_f32(lhs[2]);
        const float a3 = truncate_to_bf16_f32(lhs[3]);
        const float a4 = truncate_to_bf16_f32(lhs[4]);
        const float a5 = truncate_to_bf16_f32(lhs[5]);
        const float a6 = truncate_to_bf16_f32(lhs[6]);
        const float a7 = truncate_to_bf16_f32(lhs[7]);
        const float b0 = truncate_to_bf16_f32(rhs[0]);
        const float b1 = truncate_to_bf16_f32(rhs[1]);
        const float b2 = truncate_to_bf16_f32(rhs[2]);
        const float b3 = truncate_to_bf16_f32(rhs[3]);
        const float b4 = truncate_to_bf16_f32(rhs[4]);
        const float b5 = truncate_to_bf16_f32(rhs[5]);
        const float b6 = truncate_to_bf16_f32(rhs[6]);
        const float b7 = truncate_to_bf16_f32(rhs[7]);

        lanes[0] = std::fma(a0, b0, lanes[0]);
        lanes[0] = std::fma(a1, b1, lanes[0]);
        lanes[1] = std::fma(a2, b2, lanes[1]);
        lanes[1] = std::fma(a3, b3, lanes[1]);
        lanes[2] = std::fma(a4, b4, lanes[2]);
        lanes[2] = std::fma(a5, b5, lanes[2]);
        lanes[3] = std::fma(a6, b6, lanes[3]);
        lanes[3] = std::fma(a7, b7, lanes[3]);
    };

    float acc0[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float acc1[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float acc2[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float acc3[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
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
            truncate_to_bf16_f32(a[idx]),
            truncate_to_bf16_f32(b[idx]),
            sum
        );
    }
    return sum;
}

static inline float state_space_cpu_like_bf16_dot_row_strided(
    const float* a,
    const float* b,
    int len,
    int b_stride
) {
    auto dot8_into_lanes = [&](float lanes[4], int base) {
        const float a0 = truncate_to_bf16_f32(a[base + 0]);
        const float a1 = truncate_to_bf16_f32(a[base + 1]);
        const float a2 = truncate_to_bf16_f32(a[base + 2]);
        const float a3 = truncate_to_bf16_f32(a[base + 3]);
        const float a4 = truncate_to_bf16_f32(a[base + 4]);
        const float a5 = truncate_to_bf16_f32(a[base + 5]);
        const float a6 = truncate_to_bf16_f32(a[base + 6]);
        const float a7 = truncate_to_bf16_f32(a[base + 7]);
        const float b0 = truncate_to_bf16_f32(b[static_cast<size_t>(base + 0) * static_cast<size_t>(b_stride)]);
        const float b1 = truncate_to_bf16_f32(b[static_cast<size_t>(base + 1) * static_cast<size_t>(b_stride)]);
        const float b2 = truncate_to_bf16_f32(b[static_cast<size_t>(base + 2) * static_cast<size_t>(b_stride)]);
        const float b3 = truncate_to_bf16_f32(b[static_cast<size_t>(base + 3) * static_cast<size_t>(b_stride)]);
        const float b4 = truncate_to_bf16_f32(b[static_cast<size_t>(base + 4) * static_cast<size_t>(b_stride)]);
        const float b5 = truncate_to_bf16_f32(b[static_cast<size_t>(base + 5) * static_cast<size_t>(b_stride)]);
        const float b6 = truncate_to_bf16_f32(b[static_cast<size_t>(base + 6) * static_cast<size_t>(b_stride)]);
        const float b7 = truncate_to_bf16_f32(b[static_cast<size_t>(base + 7) * static_cast<size_t>(b_stride)]);

        lanes[0] = std::fma(a0, b0, lanes[0]);
        lanes[0] = std::fma(a1, b1, lanes[0]);
        lanes[1] = std::fma(a2, b2, lanes[1]);
        lanes[1] = std::fma(a3, b3, lanes[1]);
        lanes[2] = std::fma(a4, b4, lanes[2]);
        lanes[2] = std::fma(a5, b5, lanes[2]);
        lanes[3] = std::fma(a6, b6, lanes[3]);
        lanes[3] = std::fma(a7, b7, lanes[3]);
    };

    float acc0[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float acc1[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float acc2[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float acc3[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    int idx = 0;

    while (idx + 32 <= len) {
        dot8_into_lanes(acc0, idx + 0);
        dot8_into_lanes(acc1, idx + 8);
        dot8_into_lanes(acc2, idx + 16);
        dot8_into_lanes(acc3, idx + 24);
        idx += 32;
    }
    while (idx + 8 <= len) {
        dot8_into_lanes(acc0, idx);
        idx += 8;
    }

    float combined[4] = {
        acc0[0] + acc1[0],
        acc0[1] + acc1[1],
        acc0[2] + acc1[2],
        acc0[3] + acc1[3],
    };
    combined[0] += acc2[0];
    combined[1] += acc2[1];
    combined[2] += acc2[2];
    combined[3] += acc2[3];
    combined[0] += acc3[0];
    combined[1] += acc3[1];
    combined[2] += acc3[2];
    combined[3] += acc3[3];
    float sum = combined[0];
    sum += combined[1];
    sum += combined[2];
    sum += combined[3];
    for (; idx < len; ++idx) {
        sum = std::fma(
            truncate_to_bf16_f32(a[idx]),
            truncate_to_bf16_f32(b[static_cast<size_t>(idx) * static_cast<size_t>(b_stride)]),
            sum
        );
    }
    return sum;
}

static inline float state_space_cpu_like_bf16_dot_row_strided_single_row(
    const float* a,
    const float* b,
    int len,
    int b_stride
) {
    auto dot8_into_lanes = [&](float lanes[4], int base) {
        const float a0 = truncate_to_bf16_f32(a[base + 0]);
        const float a1 = truncate_to_bf16_f32(a[base + 1]);
        const float a2 = truncate_to_bf16_f32(a[base + 2]);
        const float a3 = truncate_to_bf16_f32(a[base + 3]);
        const float a4 = truncate_to_bf16_f32(a[base + 4]);
        const float a5 = truncate_to_bf16_f32(a[base + 5]);
        const float a6 = truncate_to_bf16_f32(a[base + 6]);
        const float a7 = truncate_to_bf16_f32(a[base + 7]);
        const float b0 = truncate_to_bf16_f32(b[static_cast<size_t>(base + 0) * static_cast<size_t>(b_stride)]);
        const float b1 = truncate_to_bf16_f32(b[static_cast<size_t>(base + 1) * static_cast<size_t>(b_stride)]);
        const float b2 = truncate_to_bf16_f32(b[static_cast<size_t>(base + 2) * static_cast<size_t>(b_stride)]);
        const float b3 = truncate_to_bf16_f32(b[static_cast<size_t>(base + 3) * static_cast<size_t>(b_stride)]);
        const float b4 = truncate_to_bf16_f32(b[static_cast<size_t>(base + 4) * static_cast<size_t>(b_stride)]);
        const float b5 = truncate_to_bf16_f32(b[static_cast<size_t>(base + 5) * static_cast<size_t>(b_stride)]);
        const float b6 = truncate_to_bf16_f32(b[static_cast<size_t>(base + 6) * static_cast<size_t>(b_stride)]);
        const float b7 = truncate_to_bf16_f32(b[static_cast<size_t>(base + 7) * static_cast<size_t>(b_stride)]);

        lanes[0] = std::fma(a0, b0, lanes[0]);
        lanes[0] = std::fma(a1, b1, lanes[0]);
        lanes[1] = std::fma(a2, b2, lanes[1]);
        lanes[1] = std::fma(a3, b3, lanes[1]);
        lanes[2] = std::fma(a4, b4, lanes[2]);
        lanes[2] = std::fma(a5, b5, lanes[2]);
        lanes[3] = std::fma(a6, b6, lanes[3]);
        lanes[3] = std::fma(a7, b7, lanes[3]);
    };

    float acc0[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float acc1[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float acc2[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float acc3[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    int idx = 0;

    while (idx + 32 <= len) {
        dot8_into_lanes(acc0, idx + 0);
        dot8_into_lanes(acc1, idx + 8);
        dot8_into_lanes(acc2, idx + 16);
        dot8_into_lanes(acc3, idx + 24);
        idx += 32;
    }
    while (idx + 8 <= len) {
        dot8_into_lanes(acc0, idx);
        idx += 8;
    }

    float combined[4] = {
        acc0[0] + acc1[0],
        acc0[1] + acc1[1],
        acc0[2] + acc1[2],
        acc0[3] + acc1[3],
    };
    combined[0] += acc2[0];
    combined[1] += acc2[1];
    combined[2] += acc2[2];
    combined[3] += acc2[3];
    combined[0] += acc3[0];
    combined[1] += acc3[1];
    combined[2] += acc3[2];
    combined[3] += acc3[3];
    float sum = combined[0];
    sum += combined[1];
    sum += combined[2];
    sum += combined[3];
    for (; idx < len; ++idx) {
        sum = std::fma(
            truncate_to_bf16_f32(a[idx]),
            truncate_to_bf16_f32(b[static_cast<size_t>(idx) * static_cast<size_t>(b_stride)]),
            sum
        );
    }
    return sum;
}

static inline array state_space_host_matmul_lastdim_reference(
    const array& input,
    const array& rhs
) {
    if (rhs.ndim() != 2) {
        throw std::invalid_argument("[state_space_reference] matmul rhs must be rank-2");
    }
    const array input_f32 = (input.dtype() == float32) ? input : astype(input, float32);
    const array rhs_f32 = (rhs.dtype() == float32) ? rhs : astype(rhs, float32);

    if (input_f32.ndim() < 2) {
        throw std::invalid_argument("[state_space_reference] matmul input rank must be >=2");
    }

    const int in_features = input_f32.shape(input_f32.ndim() - 1);
    if (rhs_f32.shape(0) != in_features) {
        throw std::invalid_argument("[state_space_reference] matmul rhs/input feature mismatch");
    }

    int rows = 1;
    for (int axis = 0; axis < input_f32.ndim() - 1; ++axis) {
        rows *= input_f32.shape(axis);
    }
    const int out_features = rhs_f32.shape(1);
    const bool emulate_bf16_rhs = rhs.dtype() == bfloat16;
    const bool emulate_bf16_input = input.dtype() == bfloat16 || emulate_bf16_rhs;

    std::vector<float> input_vec = to_host_f32_vector(input_f32);
    std::vector<float> rhs_vec = to_host_f32_vector(rhs_f32);
    if (emulate_bf16_input) {
        for (float& v : input_vec) {
            v = truncate_to_bf16_f32(v);
        }
    }
    if (emulate_bf16_rhs) {
        for (float& v : rhs_vec) {
            v = truncate_to_bf16_f32(v);
        }
    }
    std::vector<float> out_vec(static_cast<size_t>(rows) * static_cast<size_t>(out_features), 0.0f);

    for (int row = 0; row < rows; ++row) {
        const float* input_row = input_vec.data() + static_cast<size_t>(row) * static_cast<size_t>(in_features);
        for (int out_idx = 0; out_idx < out_features; ++out_idx) {
            const float* rhs_col = rhs_vec.data() + static_cast<size_t>(out_idx);
            if (emulate_bf16_input && emulate_bf16_rhs && (in_features % 8) == 0) {
                out_vec[static_cast<size_t>(row) * static_cast<size_t>(out_features) + static_cast<size_t>(out_idx)] =
                    (rows == 1)
                    ? state_space_cpu_like_bf16_dot_row_strided_single_row(input_row, rhs_col, in_features, out_features)
                    : state_space_cpu_like_bf16_dot_row_strided(input_row, rhs_col, in_features, out_features);
                continue;
            }
            float sum0 = 0.0f;
            float sum1 = 0.0f;
            float sum2 = 0.0f;
            float sum3 = 0.0f;
            int idx = 0;
            for (; idx + 4 <= in_features; idx += 4) {
                sum0 = std::fma(input_row[idx + 0], rhs_col[static_cast<size_t>(idx + 0) * static_cast<size_t>(out_features)], sum0);
                sum1 = std::fma(input_row[idx + 1], rhs_col[static_cast<size_t>(idx + 1) * static_cast<size_t>(out_features)], sum1);
                sum2 = std::fma(input_row[idx + 2], rhs_col[static_cast<size_t>(idx + 2) * static_cast<size_t>(out_features)], sum2);
                sum3 = std::fma(input_row[idx + 3], rhs_col[static_cast<size_t>(idx + 3) * static_cast<size_t>(out_features)], sum3);
            }
            float sum = (sum0 + sum1) + (sum2 + sum3);
            for (; idx < in_features; ++idx) {
                sum = std::fma(input_row[idx], rhs_col[static_cast<size_t>(idx) * static_cast<size_t>(out_features)], sum);
            }
            out_vec[static_cast<size_t>(row) * static_cast<size_t>(out_features) + static_cast<size_t>(out_idx)] = sum;
        }
    }

    Shape out_shape = input_f32.shape();
    out_shape.back() = out_features;
    return owned_array_from_f32_vector(std::move(out_vec), out_shape);
}

static inline array state_space_host_linear_reference(
    const array& input,
    const array& weight
) {
    if (weight.ndim() != 2) {
        throw std::invalid_argument("[state_space_reference] expected 2D weight");
    }
    if (input.ndim() < 2) {
        throw std::invalid_argument("[state_space_reference] linear input rank must be >=2");
    }

    const int in_features = input.shape(input.ndim() - 1);
    const int rows = weight.shape(0);
    const int cols = weight.shape(1);
    const bool direct = rows == in_features;
    const bool transposed = cols == in_features;

    if (direct && !transposed) {
        return state_space_host_matmul_lastdim_reference(input, weight);
    }
    if (transposed && !direct) {
        if (input.ndim() >= 2) {
            int input_rows = 1;
            for (int axis = 0; axis + 1 < input.ndim(); ++axis) {
                input_rows *= input.shape(axis);
            }
            if (input_rows == 1 && weight.dtype() == bfloat16) {
                const int weight_rows = weight.shape(0);
                const int weight_cols = weight.shape(1);
                const array input_f32 = (input.dtype() == float32) ? input : astype(input, float32);
                const array weight_f32 = astype(weight, float32);
                array flat_input = reshape(input_f32, {in_features});
                array flat_weight = reshape(weight_f32, {weight_rows * weight_cols});
                eval(flat_input, flat_weight);

                std::vector<float> input_vec = to_host_f32_vector(flat_input);
                std::vector<float> weight_vec = to_host_f32_vector(flat_weight);
                for (float& v : input_vec) v = truncate_to_bf16_f32(v);
                for (float& v : weight_vec) v = truncate_to_bf16_f32(v);

                std::vector<float> out(static_cast<size_t>(weight_rows), 0.0f);
                const float* input_row = input_vec.data();
                for (int o = 0; o < weight_rows; ++o) {
                    const float* weight_row = weight_vec.data() + static_cast<size_t>(o) * static_cast<size_t>(in_features);
                    out[static_cast<size_t>(o)] = (in_features % 8) == 0
                        ? state_space_cpu_like_bf16_dot_row_single_row(input_row, weight_row, in_features)
                        : state_space_cpu_like_dot_row(input_row, weight_row, in_features);
                }

                Shape out_shape = input.shape();
                out_shape.back() = weight_rows;
                return owned_array_from_f32_vector(std::move(out), out_shape);
            }
        }
        return state_space_host_matmul_lastdim_reference(input, contiguous(transpose(weight)));
    }
    if (direct && transposed) {
        // Match CPU Tensor-matmul behavior for square weights: treat the stored
        // tensor as already aligned with the input feature axis.
        return state_space_host_matmul_lastdim_reference(input, weight);
    }
    throw std::invalid_argument("[state_space_reference] weight orientation incompatible with input");
}

static inline array gated_delta_apply_fast_silu_reference(const array& src) {
    std::vector<float> values = to_host_f32_vector(src);
    for (float& v : values) {
        v = gated_delta_exact_silu_scalar(v);
    }
    if (src.ndim() == 3) {
        return owned_array_from_f32_vector(
            std::move(values),
            {src.shape(0), src.shape(1), src.shape(2)}
        );
    }
    if (src.ndim() == 2) {
        return owned_array_from_f32_vector(
            std::move(values),
            {src.shape(0), src.shape(1)}
        );
    }
    if (src.ndim() == 1) {
        return owned_array_from_f32_vector(std::move(values), {src.shape(0)});
    }
    throw std::invalid_argument("[gated_delta_reference] unsupported rank for fast-silu");
}

static inline GatedDeltaReferenceStep gated_delta_recurrence_step_reference(
    const array& state,
    const array& query,
    const array& key,
    const array& value,
    const array& z,
    const array& beta_raw,
    const array& a_raw,
    const array& a_log_exp_row,
    const array& dt_bias_heads,
    const array& norm_weight_heads,
    int n_heads,
    int d_head
) {
    gated_delta_reference_emit_once();
    constexpr float eps = 1.0e-6f;
    const float query_scale = 1.0f / std::sqrt(static_cast<float>(d_head));
    const int state_size = n_heads * d_head * d_head;
    const int head_size = n_heads * d_head;

    std::vector<float> state_vec = to_host_f32_vector(state);
    std::vector<float> query_vec = to_host_f32_vector(query);
    std::vector<float> key_vec = to_host_f32_vector(key);
    std::vector<float> value_vec = to_host_f32_vector(value);
    std::vector<float> z_vec = to_host_f32_vector(z);
    std::vector<float> beta_raw_vec = to_host_f32_vector(beta_raw);
    std::vector<float> a_raw_vec = to_host_f32_vector(a_raw);
    std::vector<float> a_log_vec = to_host_f32_vector(a_log_exp_row);
    std::vector<float> dt_bias_vec = to_host_f32_vector(dt_bias_heads);
    std::vector<float> norm_weight_vec = to_host_f32_vector(norm_weight_heads);

    if (static_cast<int>(state_vec.size()) != state_size ||
        static_cast<int>(query_vec.size()) != head_size ||
        static_cast<int>(key_vec.size()) != head_size ||
        static_cast<int>(value_vec.size()) != head_size ||
        static_cast<int>(z_vec.size()) != head_size ||
        static_cast<int>(beta_raw_vec.size()) != n_heads ||
        static_cast<int>(a_raw_vec.size()) != n_heads ||
        static_cast<int>(a_log_vec.size()) != n_heads ||
        static_cast<int>(dt_bias_vec.size()) != n_heads ||
        static_cast<int>(norm_weight_vec.size()) != head_size) {
        throw std::invalid_argument("[gated_delta_reference] unexpected tensor shape in host recurrence");
    }

    std::vector<float> ssm_out_vec(static_cast<size_t>(head_size), 0.0f);
    std::vector<float> norm_vec(static_cast<size_t>(head_size), 0.0f);
    std::vector<float> q_norm_vec(static_cast<size_t>(head_size), 0.0f);
    std::vector<float> k_norm_vec(static_cast<size_t>(head_size), 0.0f);

    for (int h = 0; h < n_heads; ++h) {
        const int h_base = h * d_head;
        const int s_base = h * d_head * d_head;

        const float* query_head = query_vec.data() + static_cast<size_t>(h_base);
        const float* key_head = key_vec.data() + static_cast<size_t>(h_base);
        float q_norm_sq = state_space_cpu_like_dot_row(query_head, query_head, d_head);
        float k_norm_sq = state_space_cpu_like_dot_row(key_head, key_head, d_head);
        const float q_inv = 1.0f / std::sqrt(q_norm_sq + eps);
        const float k_inv = 1.0f / std::sqrt(k_norm_sq + eps);
        for (int i = 0; i < d_head; ++i) {
            q_norm_vec[static_cast<size_t>(h_base + i)] = query_vec[static_cast<size_t>(h_base + i)] * q_inv * query_scale;
            k_norm_vec[static_cast<size_t>(h_base + i)] = key_vec[static_cast<size_t>(h_base + i)] * k_inv;
        }

        const float beta = gated_delta_fast_sigmoid_scalar(beta_raw_vec[static_cast<size_t>(h)]);
        const float dt_term = dt_bias_vec[static_cast<size_t>(h)];
        const float a_log = a_log_vec[static_cast<size_t>(h)];
        const float g = -gated_delta_fast_exp_scalar(a_log) * gated_delta_fast_softplus_scalar(a_raw_vec[static_cast<size_t>(h)] + dt_term);
        const float g_exp = gated_delta_fast_exp_scalar(g);

        const float* q_norm_head = q_norm_vec.data() + static_cast<size_t>(h_base);
        const float* k_norm_head = k_norm_vec.data() + static_cast<size_t>(h_base);
        const float qk_dot = state_space_cpu_like_dot_row(q_norm_head, k_norm_head, d_head);

        std::vector<float> kv_mem(static_cast<size_t>(d_head), 0.0f);
        std::vector<float> ssm_head(static_cast<size_t>(d_head), 0.0f);
        for (int k_idx = 0; k_idx < d_head; ++k_idx) {
            const float key_weight = k_norm_vec[static_cast<size_t>(h_base + k_idx)];
            const float proj_weight = q_norm_vec[static_cast<size_t>(h_base + k_idx)];
            const int row_base = s_base + k_idx * d_head;
            for (int v_idx = 0; v_idx < d_head; ++v_idx) {
                const int state_idx = row_base + v_idx;
                const float scaled = state_vec[static_cast<size_t>(state_idx)] * g_exp;
                state_vec[static_cast<size_t>(state_idx)] = scaled;
                kv_mem[static_cast<size_t>(v_idx)] += scaled * key_weight;
                ssm_head[static_cast<size_t>(v_idx)] += scaled * proj_weight;
            }
        }

        for (int v_idx = 0; v_idx < d_head; ++v_idx) {
            const float vv = value_vec[static_cast<size_t>(h_base + v_idx)];
            kv_mem[static_cast<size_t>(v_idx)] = (vv - kv_mem[static_cast<size_t>(v_idx)]) * beta;
            ssm_head[static_cast<size_t>(v_idx)] += kv_mem[static_cast<size_t>(v_idx)] * qk_dot;
        }
        for (int k_idx = 0; k_idx < d_head; ++k_idx) {
            const float key_weight = k_norm_vec[static_cast<size_t>(h_base + k_idx)];
            const int row_base = s_base + k_idx * d_head;
            for (int v_idx = 0; v_idx < d_head; ++v_idx) {
                state_vec[static_cast<size_t>(row_base + v_idx)] += kv_mem[static_cast<size_t>(v_idx)] * key_weight;
            }
        }

        for (int v_idx = 0; v_idx < d_head; ++v_idx) {
            const float ov = ssm_head[static_cast<size_t>(v_idx)];
            ssm_out_vec[static_cast<size_t>(h_base + v_idx)] = ov;
        }
        const float mean_sq =
            state_space_cpu_like_dot_row(ssm_head.data(), ssm_head.data(), d_head) /
            static_cast<float>(d_head);
        const float inv_rms = 1.0f / std::sqrt(mean_sq + eps);
        for (int v_idx = 0; v_idx < d_head; ++v_idx) {
            const float weight = norm_weight_vec[static_cast<size_t>(h_base + v_idx)];
            const float gate = z_vec[static_cast<size_t>(h_base + v_idx)];
            const float silu = gated_delta_exact_silu_scalar(gate);
            norm_vec[static_cast<size_t>(h_base + v_idx)] = ssm_head[static_cast<size_t>(v_idx)] * inv_rms * weight * silu;
        }
    }

    return {
        .ssm = owned_array_from_f32_vector(std::move(ssm_out_vec), {n_heads, d_head}),
        .norm = owned_array_from_f32_vector(std::move(norm_vec), {n_heads, d_head}),
        .next_state = owned_array_from_f32_vector(std::move(state_vec), {n_heads, d_head, d_head}),
    };
}
