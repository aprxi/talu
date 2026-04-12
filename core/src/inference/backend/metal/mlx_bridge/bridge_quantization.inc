// bridge_quantization.inc — Weight transformation and quantization format support:
// FP8, MXFP8, NVFP4, grouped-affine. Dequantization, transcoding, weight capture.

array to_rhs(const array& weight_2d, const std::string& name) {
    if (weight_2d.ndim() != 2) {
        throw std::runtime_error("expected rank-2 weight for " + name);
    }
    // MLX-LM nn.Linear uses [out, in] weights. Convert once to [in, out] RHS.
    // Materialize a contiguous RHS tensor once at load-time so decode matmuls
    // do not pay strided-view penalties on every token step.
    return stop_gradient(copy(transpose(weight_2d, {1, 0})));
}

size_t grouped_affine_dequantized_weight_bytes(
    const array& q_w,
    int quant_bits,
    Dtype dtype
) {
    if (q_w.ndim() != 2 || quant_bits <= 0) return 0;
    const size_t out_dim = static_cast<size_t>(q_w.shape(0));
    const size_t packed_cols = static_cast<size_t>(q_w.shape(1));
    const size_t in_dim = packed_cols * 32ULL / static_cast<size_t>(quant_bits);
    return out_dim * in_dim * static_cast<size_t>(size_of(dtype));
}

array build_grouped_affine_prefill_rhs(
    const std::string& name,
    const array& q_w,
    const array& q_scales,
    const array& q_biases,
    int quant_group_size,
    int quant_bits,
    Dtype cache_dtype = bfloat16
) {
    if (q_w.ndim() != 2 || q_scales.ndim() != 2 || q_biases.ndim() != 2) {
        return array(0.0f);
    }
    if (q_w.size() <= 1 || q_scales.size() <= 1 || q_biases.size() <= 1) {
        return array(0.0f);
    }
    if (quant_group_size <= 0 || (quant_bits != 4 && quant_bits != 8)) {
        return array(0.0f);
    }

    const array dense = dequantize(
        q_w,
        q_scales,
        std::optional<array>(q_biases),
        quant_group_size,
        quant_bits,
        "affine",
        std::nullopt,
        std::optional<Dtype>(cache_dtype)
    );
    return to_rhs(dense, name + ".prefill_cache");
}

array expand_block_scales(const array& scales_2d_f32, int rows, int cols, const std::string& name) {
    if (scales_2d_f32.ndim() != 2) {
        throw std::runtime_error("expected rank-2 FP8 scale tensor for " + name);
    }
    const int scale_rows = scales_2d_f32.shape(0);
    const int scale_cols = scales_2d_f32.shape(1);
    if (scale_rows <= 0 || scale_cols <= 0) {
        throw std::runtime_error("invalid FP8 scale shape for " + name);
    }

    const int block_row = (rows + scale_rows - 1) / scale_rows;
    const int block_col = (cols + scale_cols - 1) / scale_cols;
    array expanded = repeat(scales_2d_f32, block_row, 0);
    expanded = repeat(expanded, block_col, 1);
    return slice(expanded, {0, 0}, {rows, cols});
}

array maybe_dequantize_fp8_weight(
    const std::unordered_map<std::string, array>& tensors,
    const std::string& weight_key,
    const array& weight_2d
) {
    if (!ends_with(weight_key, ".weight")) {
        return weight_2d;
    }
    if (weight_2d.ndim() != 2) {
        throw std::runtime_error("expected rank-2 weight for " + weight_key);
    }

    const std::string base = weight_key.substr(0, weight_key.size() - std::string(".weight").size());
    const std::string scale_inv_key = base + ".weight_scale_inv";
    const std::string block_scale_key = base + ".weight_block_scale";
    const std::string weight_scale_key = base + ".weight_scale";
    const bool has_nvfp4_packed = tensors.find(base + ".weight_packed") != tensors.end();
    const bool has_grouped_affine_scales = tensors.find(base + ".scales") != tensors.end();
    const bool has_grouped_affine_biases =
        tensors.find(base + ".biases") != tensors.end() ||
        tensors.find(base + ".weight_bias") != tensors.end();

    auto scale_inv_it = tensors.find(scale_inv_key);
    auto block_scale_it = tensors.find(block_scale_key);
    if (block_scale_it == tensors.end() &&
        !has_nvfp4_packed &&
        !(has_grouped_affine_scales && has_grouped_affine_biases)) {
        block_scale_it = tensors.find(weight_scale_key);
    }
    if (scale_inv_it == tensors.end() && block_scale_it == tensors.end()) {
        return weight_2d;
    }

    const int rows = weight_2d.shape(0);
    const int cols = weight_2d.shape(1);
    array scales_f32 = array(1.0f);

    if (block_scale_it != tensors.end()) {
        const array& block_scales = block_scale_it->second;
        if (block_scales.ndim() != 2) {
            throw std::runtime_error("mxfp8 block scales must be rank-2 for " + weight_key);
        }
        if (block_scales.shape(0) != rows) {
            throw std::runtime_error("mxfp8 block scales row mismatch for " + weight_key);
        }
        const int expected_scale_cols = (cols + 31) / 32;
        if (block_scales.shape(1) != expected_scale_cols) {
            throw std::runtime_error(
                "mxfp8 block scales col mismatch for " + weight_key +
                " expected=" + std::to_string(expected_scale_cols) +
                " got=" + std::to_string(block_scales.shape(1))
            );
        }

        if (block_scales.dtype() == uint8 || block_scales.dtype() == int8) {
            const array block_scales_u8 = astype(block_scales, uint8);
            const int flat_size = static_cast<int>(block_scales_u8.size());
            const array flat_u8 = reshape(block_scales_u8, {flat_size});
            eval(flat_u8);

            const uint8_t* scale_bytes = flat_u8.data<uint8_t>();
            std::vector<float> decoded(static_cast<size_t>(flat_size));
            for (int idx = 0; idx < flat_size; ++idx) {
                const uint32_t exp_bits = static_cast<uint32_t>(scale_bytes[idx]) << 23;
                float scale = 0.0f;
                std::memcpy(&scale, &exp_bits, sizeof(float));
                decoded[static_cast<size_t>(idx)] = scale;
            }
            scales_f32 = array(decoded.begin(), Shape{block_scales.shape(0), block_scales.shape(1)}, float32);
        } else {
            scales_f32 = astype(block_scales, float32);
        }
    } else {
        const array& scale_inv = scale_inv_it->second;
        if (scale_inv.ndim() != 2) {
            throw std::runtime_error("fp8 scale_inv must be rank-2 for " + weight_key);
        }
        scales_f32 = astype(scale_inv, float32);
    }

    const array expanded_scales = expand_block_scales(scales_f32, rows, cols, weight_key);
    array weight_as_f32 = array(0.0f);
    if (weight_2d.dtype() == uint8) {
        weight_as_f32 = from_fp8(weight_2d, float32);
    } else {
        weight_as_f32 = astype(weight_2d, float32);
    }
    const array dequant_f32 = weight_as_f32 * expanded_scales;
    array dequant_bf16 = astype(dequant_f32, bfloat16);
    dequant_bf16 = stop_gradient(copy(dequant_bf16));
    // Optional eager materialization for debugging/benchmark parity. Disabled
    // by default to reduce init-time memory pressure on larger checkpoints.
    if (const char* eager = std::getenv("TALU_METAL_EAGER_DEQUANT")) {
        if (std::string(eager) == "1") {
            eval(dequant_bf16);
            synchronize();
        }
    }
    return dequant_bf16;
}

float fp4_e2m1_nibble_to_f32(uint8_t nibble) {
    static constexpr float codebook[16] = {
        0.0f, 0.5f, 1.0f, 1.5f,
        2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f,
        -2.0f, -3.0f, -4.0f, -6.0f,
    };
    return codebook[nibble & 0x0F];
}

bool has_nvfp4_metadata_for_key(
    const std::unordered_map<std::string, array>& tensors,
    const std::string& weight_key
) {
    if (!ends_with(weight_key, ".weight")) return false;
    const std::string base = weight_key.substr(0, weight_key.size() - std::string(".weight").size());
    return tensors.find(base + ".weight_packed") != tensors.end() &&
        tensors.find(base + ".weight_scale") != tensors.end();
}

const array& nvfp4_unit_global_scale() {
    static thread_local std::optional<array> scale;
    if (!scale.has_value()) {
        scale = stop_gradient(copy(array(1.0f, float32)));
    }
    return *scale;
}

array dequantize_nvfp4_weight(
    const std::unordered_map<std::string, array>& tensors,
    const std::string& weight_key
) {
    const std::string base = weight_key.substr(0, weight_key.size() - std::string(".weight").size());
    const array packed_u8 = astype(require_tensor(tensors, base + ".weight_packed"), uint8);
    const array scales_f32 = astype(require_tensor(tensors, base + ".weight_scale"), float32);
    eval(packed_u8);
    eval(scales_f32);

    if (packed_u8.ndim() != 2 || scales_f32.ndim() != 2) {
        throw std::runtime_error("nvfp4 packed/scales must be rank-2 for " + weight_key);
    }

    const int rows = packed_u8.shape(0);
    const int packed_cols = packed_u8.shape(1);
    const int cols = packed_cols * 2;
    if (scales_f32.shape(0) != rows) {
        throw std::runtime_error("nvfp4 scales row mismatch for " + weight_key);
    }
    const int scale_cols = scales_f32.shape(1);
    if (scale_cols <= 0 || cols % scale_cols != 0) {
        throw std::runtime_error("invalid nvfp4 scales shape for " + weight_key);
    }
    const int group_size = cols / scale_cols;

    float global_scale = 1.0f;
    if (has_tensor(tensors, base + ".weight_global_scale")) {
        const array global_f32 = astype(require_tensor(tensors, base + ".weight_global_scale"), float32);
        eval(global_f32);
        if (global_f32.size() > 0) global_scale = global_f32.item<float>();
    }
    if (global_scale == 0.0f) {
        throw std::runtime_error("invalid nvfp4 global scale for " + weight_key);
    }

    const uint8_t* packed_ptr = packed_u8.data<uint8_t>();
    const float* scale_ptr = scales_f32.data<float>();
    std::vector<float> decoded(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    for (int r = 0; r < rows; ++r) {
        const size_t packed_row_off = static_cast<size_t>(r) * static_cast<size_t>(packed_cols);
        const size_t scale_row_off = static_cast<size_t>(r) * static_cast<size_t>(scale_cols);
        const size_t out_row_off = static_cast<size_t>(r) * static_cast<size_t>(cols);
        for (int c = 0; c < cols; ++c) {
            const uint8_t packed = packed_ptr[packed_row_off + static_cast<size_t>(c / 2)];
            const uint8_t nibble = (c & 1) == 0 ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
            const float local_scale = scale_ptr[scale_row_off + static_cast<size_t>(c / group_size)];
            decoded[out_row_off + static_cast<size_t>(c)] = fp4_e2m1_nibble_to_f32(nibble) * (local_scale / global_scale);
        }
    }

    array dequant_bf16 = astype(array(decoded.begin(), Shape{rows, cols}, float32), bfloat16);
    dequant_bf16 = stop_gradient(copy(dequant_bf16));
    return dequant_bf16;
}

bool transcode_nvfp4_to_affine_qmm(
    const std::unordered_map<std::string, array>& tensors,
    const std::string& weight_key,
    const array& lhs_weight,
    int target_group_size,
    array* out_q_w,
    array* out_q_scales,
    array* out_q_biases
) {
    if (!out_q_w || !out_q_scales || !out_q_biases) return false;
    if (target_group_size <= 0 || !has_nvfp4_metadata_for_key(tensors, weight_key)) return false;

    const array dense_bf16 = dequantize_nvfp4_weight(tensors, weight_key);
    const array dense_f32 = astype(dense_bf16, float32);
    eval(dense_f32);
    if (dense_f32.ndim() != 2 || lhs_weight.ndim() != 2) return false;

    const int rows = dense_f32.shape(0);
    const int cols = dense_f32.shape(1);
    const int packed_cols = cols / 8;
    if (rows <= 0 || cols <= 0 || (cols % target_group_size) != 0 || (cols % 8) != 0) return false;
    const bool lhs_matches_unpacked =
        lhs_weight.shape(0) == rows && lhs_weight.shape(1) == cols;
    const bool lhs_matches_packed =
        lhs_weight.shape(0) == rows && lhs_weight.shape(1) == packed_cols;
    if (!lhs_matches_unpacked && !lhs_matches_packed) {
        // Synthetic fused decode weights (for example linear-attention qkvz/ba)
        // must quantize the full concatenated tensor, not a single source slice.
        return false;
    }

    const int groups = cols / target_group_size;
    const float* src = dense_f32.data<float>();

    std::vector<uint32_t> packed(static_cast<size_t>(rows) * static_cast<size_t>(packed_cols), 0);
    std::vector<float> scales(static_cast<size_t>(rows) * static_cast<size_t>(groups), 0.0f);
    std::vector<float> biases(static_cast<size_t>(rows) * static_cast<size_t>(groups), 0.0f);

    for (int r = 0; r < rows; ++r) {
        const size_t row_off = static_cast<size_t>(r) * static_cast<size_t>(cols);
        const size_t packed_row_off = static_cast<size_t>(r) * static_cast<size_t>(packed_cols);
        const size_t group_row_off = static_cast<size_t>(r) * static_cast<size_t>(groups);
        for (int g = 0; g < groups; ++g) {
            const int start = g * target_group_size;
            float min_v = src[row_off + static_cast<size_t>(start)];
            float max_v = min_v;
            for (int i = 1; i < target_group_size; ++i) {
                const float v = src[row_off + static_cast<size_t>(start + i)];
                min_v = std::min(min_v, v);
                max_v = std::max(max_v, v);
            }

            float scale = (max_v - min_v) / 15.0f;
            float bias = min_v;
            if (!(scale > 0.0f) || !std::isfinite(scale)) {
                scale = 1.0f;
                bias = min_v;
            }
            scales[group_row_off + static_cast<size_t>(g)] = scale;
            biases[group_row_off + static_cast<size_t>(g)] = bias;

            for (int i = 0; i < target_group_size; ++i) {
                const int c = start + i;
                const float v = src[row_off + static_cast<size_t>(c)];
                int q = static_cast<int>(std::lrint((v - bias) / scale));
                q = std::max(0, std::min(15, q));
                const size_t packed_idx = packed_row_off + static_cast<size_t>(c / 8);
                const int shift = (c % 8) * 4;
                packed[packed_idx] |= (static_cast<uint32_t>(q) << shift);
            }
        }
    }

    *out_q_w = stop_gradient(copy(array(packed.begin(), Shape{rows, packed_cols}, uint32)));
    *out_q_scales = stop_gradient(copy(astype(array(scales.begin(), Shape{rows, groups}, float32), bfloat16)));
    *out_q_biases = stop_gradient(copy(astype(array(biases.begin(), Shape{rows, groups}, float32), bfloat16)));
    eval(*out_q_w);
    eval(*out_q_scales);
    eval(*out_q_biases);
    return true;
}

bool capture_native_nvfp4_quant(
    const std::unordered_map<std::string, array>& tensors,
    const std::string& weight_key,
    array* out_q_w,
    array* out_q_scales,
    array* out_q_biases
) {
    if (!out_q_w || !out_q_scales || !out_q_biases) return false;
    if (!has_nvfp4_metadata_for_key(tensors, weight_key)) return false;
    if (!ends_with(weight_key, ".weight")) return false;

    const std::string base = weight_key.substr(0, weight_key.size() - std::string(".weight").size());
    const array packed_src = astype(require_tensor(tensors, base + ".weight_packed"), uint8);
    const array scale_src = view(require_tensor(tensors, base + ".weight_scale"), uint8);
    eval(packed_src);
    eval(scale_src);

    if (packed_src.ndim() != 2 || scale_src.ndim() != 2) return false;
    const int rows = packed_src.shape(0);
    const int packed_byte_cols = packed_src.shape(1);
    const int cols = packed_byte_cols * 2;
    if (packed_byte_cols <= 0 || (packed_byte_cols % 4) != 0) return false;
    if ((cols % 16) != 0) return false;
    if (scale_src.shape(0) != rows || scale_src.shape(1) != (cols / 16)) return false;

    array global_scale = array(1.0f, float32);
    if (has_tensor(tensors, base + ".weight_global_scale")) {
        global_scale = astype(require_tensor(tensors, base + ".weight_global_scale"), float32);
    }

    *out_q_w = stop_gradient(copy(reshape(view(packed_src, uint32), {rows, packed_byte_cols / 4})));
    *out_q_scales = stop_gradient(copy(scale_src));
    *out_q_biases = stop_gradient(copy(global_scale));
    return true;
}

void materialize_nvfp4_affine_execution_tensors(
    std::unordered_map<std::string, array>* tensors,
    int target_group_size
) {
    if (!tensors || target_group_size <= 0) return;

    std::vector<std::string> packed_keys;
    packed_keys.reserve(tensors->size());
    for (const auto& entry : *tensors) {
        if (ends_with(entry.first, ".weight_packed")) {
            packed_keys.push_back(entry.first);
        }
    }

    for (const std::string& packed_key : packed_keys) {
        const std::string base = packed_key.substr(0, packed_key.size() - std::string(".weight_packed").size());
        const std::string weight_key = base + ".weight";
        if (tensors->find(weight_key) != tensors->end()) continue;

        auto packed_it = tensors->find(packed_key);
        if (packed_it == tensors->end()) continue;

        array q_w = array(0.0f);
        array q_scales = array(0.0f);
        array q_biases = array(0.0f);
        if (!transcode_nvfp4_to_affine_qmm(
                *tensors,
                weight_key,
                packed_it->second,
                target_group_size,
                &q_w,
                &q_scales,
                &q_biases))
        {
            continue;
        }

        tensors->insert_or_assign(weight_key, q_w);
        tensors->insert_or_assign(base + ".scales", q_scales);
        tensors->insert_or_assign(base + ".biases", q_biases);
    }
}

array maybe_dequantize_grouped_affine_weight(
    const std::unordered_map<std::string, array>& tensors,
    const BridgeModelConfig& cfg,
    const std::string& weight_key,
    const array& weight_2d
) {
    if (!ends_with(weight_key, ".weight")) {
        return weight_2d;
    }
    if (weight_2d.ndim() != 2) {
        throw std::runtime_error("expected rank-2 weight for " + weight_key);
    }

    const std::string base = weight_key.substr(0, weight_key.size() - std::string(".weight").size());
    const std::string scales_key = base + ".scales";
    const std::string biases_key = base + ".biases";
    const std::string weight_scale_key = base + ".weight_scale";
    const std::string weight_bias_key = base + ".weight_bias";

    auto scales_it = tensors.find(scales_key);
    if (scales_it == tensors.end()) scales_it = tensors.find(weight_scale_key);
    auto biases_it = tensors.find(biases_key);
    if (biases_it == tensors.end()) biases_it = tensors.find(weight_bias_key);
    if (scales_it == tensors.end() || biases_it == tensors.end()) {
        return weight_2d;
    }

    if (cfg.quant_group_size <= 0) {
        throw std::runtime_error("grouped-affine metadata found but quantization.group_size is missing/invalid");
    }

    const int rows = weight_2d.shape(0);
    const int packed_cols = weight_2d.shape(1);

    const array scales_f32 = astype(scales_it->second, float32);
    const array biases_f32 = astype(biases_it->second, float32);
    if (scales_f32.ndim() != 2 || biases_f32.ndim() != 2) {
        throw std::runtime_error("grouped-affine scales/biases must be rank-2 for " + weight_key);
    }
    if (scales_f32.shape(0) != rows || biases_f32.shape(0) != rows) {
        throw std::runtime_error("grouped-affine scales/biases row mismatch for " + weight_key);
    }
    if (scales_f32.shape(1) != biases_f32.shape(1)) {
        throw std::runtime_error("grouped-affine scales/biases group mismatch for " + weight_key);
    }

    // Auto-detect bits from scales shape. Mixed-precision models may use
    // different bits for embedding vs linear weights (e.g. U8 embed + U4 linear).
    const int scale_cols = scales_f32.shape(1);
    const int gs = cfg.quant_group_size;
    int bits = cfg.quant_bits;
    if (bits != 4 && bits != 8) bits = 4;
    // Verify the config hint matches; if not, try the other interpretation.
    const int cols_from_hint = packed_cols * (bits == 4 ? 8 : 4);
    const int expected_groups_hint = (cols_from_hint + gs - 1) / gs;
    if (expected_groups_hint != scale_cols) {
        const int alt_bits = bits == 4 ? 8 : 4;
        const int cols_alt = packed_cols * (alt_bits == 4 ? 8 : 4);
        const int expected_groups_alt = (cols_alt + gs - 1) / gs;
        if (expected_groups_alt == scale_cols) {
            bits = alt_bits;
        } else {
            throw std::runtime_error(
                "grouped-affine scales cols mismatch for " + weight_key +
                " (packed_cols=" + std::to_string(packed_cols) +
                ", scale_cols=" + std::to_string(scale_cols) +
                ", group_size=" + std::to_string(gs) + ")"
            );
        }
    }
    const int values_per_word = bits == 4 ? 8 : 4;
    const uint32_t mask = bits == 4 ? 0xF : 0xFF;
    const int cols = packed_cols * values_per_word;

    // Use MLX's GPU-side dequantize — avoids host-side std::vector<float>
    // balloon that the old CPU loop created (32x expansion for 4-bit).
    const array packed_u32 = astype(weight_2d, uint32);
    array dequant = mlx::core::dequantize(
        packed_u32,
        scales_it->second,
        std::optional<array>(biases_it->second),
        cfg.quant_group_size,
        bits,
        "affine",
        std::nullopt,
        bfloat16
    );
    return stop_gradient(dequant);
}

bool has_grouped_affine_metadata_for_key(
    const std::unordered_map<std::string, array>& tensors,
    const std::string& weight_key
) {
    if (!ends_with(weight_key, ".weight")) return false;
    const std::string base = weight_key.substr(0, weight_key.size() - std::string(".weight").size());
    const bool has_scales =
        tensors.find(base + ".scales") != tensors.end() ||
        tensors.find(base + ".weight_scale") != tensors.end();
    const bool has_biases =
        tensors.find(base + ".biases") != tensors.end() ||
        tensors.find(base + ".weight_bias") != tensors.end();
    return has_scales && has_biases;
}

bool maybe_capture_grouped_affine_quant(
    const std::unordered_map<std::string, array>& tensors,
    const BridgeModelConfig& cfg,
    const std::string& weight_key,
    const array& lhs_weight,
    array* out_q_w,
    array* out_q_scales,
    array* out_q_biases,
    int* out_detected_bits = nullptr
) {
    if (!has_grouped_affine_metadata_for_key(tensors, weight_key)) return false;
    if (!ends_with(weight_key, ".weight")) return false;
    if (lhs_weight.ndim() != 2) return false;
    if (cfg.quant_bits != 4 && cfg.quant_bits != 8) return false;
    if (cfg.quant_group_size <= 0) return false;

    const std::string base = weight_key.substr(0, weight_key.size() - std::string(".weight").size());
    auto raw_it = tensors.find(weight_key);
    if (raw_it == tensors.end()) return false;

    auto scales_it = tensors.find(base + ".scales");
    if (scales_it == tensors.end()) scales_it = tensors.find(base + ".weight_scale");
    auto biases_it = tensors.find(base + ".biases");
    if (biases_it == tensors.end()) biases_it = tensors.find(base + ".weight_bias");
    if (scales_it == tensors.end() || biases_it == tensors.end()) return false;

    const array& raw_weight = raw_it->second;
    if (raw_weight.ndim() != 2) return false;
    const int rows = raw_weight.shape(0);
    const int packed_cols = raw_weight.shape(1);

    const array& scales_src = scales_it->second;
    const array& biases_src = biases_it->second;
    const Dtype scales_dt = scales_src.dtype();
    const Dtype biases_dt = biases_src.dtype();
    const bool scales_float = scales_dt == float16 || scales_dt == bfloat16 || scales_dt == float32 || scales_dt == float64;
    const bool biases_float = biases_dt == float16 || biases_dt == bfloat16 || biases_dt == float32 || biases_dt == float64;
    if (!scales_float || !biases_float) return false;
    if (scales_src.ndim() != 2 || biases_src.ndim() != 2) return false;
    if (scales_src.shape(0) != rows || biases_src.shape(0) != rows) return false;
    if (scales_src.shape(1) != biases_src.shape(1)) return false;

    // Auto-detect bits from scales shape for mixed-precision models.
    const int gs = cfg.quant_group_size;
    const int scale_cols = scales_src.shape(1);
    int bits = cfg.quant_bits;
    if (bits != 4 && bits != 8) bits = 4;
    int cols = packed_cols * (bits == 4 ? 8 : 4);
    int expected_groups = (cols + gs - 1) / gs;
    if (expected_groups != scale_cols) {
        const int alt_bits = bits == 4 ? 8 : 4;
        const int alt_cols = packed_cols * (alt_bits == 4 ? 8 : 4);
        const int alt_groups = (alt_cols + gs - 1) / gs;
        if (alt_groups == scale_cols) {
            bits = alt_bits;
            cols = alt_cols;
            expected_groups = alt_groups;
        } else {
            return false;
        }
    }

    const int values_per_word = bits == 4 ? 8 : 4;
    const bool lhs_matches_unpacked =
        lhs_weight.shape(0) == rows && lhs_weight.shape(1) == cols;
    const bool lhs_matches_packed =
        lhs_weight.shape(0) == rows && lhs_weight.shape(1) == packed_cols;
    if (!lhs_matches_unpacked && !lhs_matches_packed) {
        return false;
    }

    *out_q_w = astype(raw_weight, uint32);
    *out_q_scales = scales_src;
    *out_q_biases = biases_src;
    if (out_detected_bits) *out_detected_bits = bits;
    return true;
}

array load_linear_weight(
    const std::unordered_map<std::string, array>& tensors,
    const BridgeModelConfig& cfg,
    const std::string& weight_key
) {
    const array& raw_weight = require_tensor(tensors, weight_key);
    const array grouped = maybe_dequantize_grouped_affine_weight(tensors, cfg, weight_key, raw_weight);
    array out = array(0.0f);
    if (grouped.size() != raw_weight.size() || grouped.dtype() != raw_weight.dtype() || grouped.ndim() != raw_weight.ndim()) {
        out = maybe_dequantize_fp8_weight(tensors, weight_key, grouped);
    } else if (has_nvfp4_metadata_for_key(tensors, weight_key)) {
        out = dequantize_nvfp4_weight(tensors, weight_key);
    } else {
        out = maybe_dequantize_fp8_weight(tensors, weight_key, grouped);
    }

    return out;
}

std::pair<array, array> split_fused_moe_gate_up_projection(
    const array& fused,
    const std::string& name
) {
    if (fused.ndim() != 3) {
        throw std::runtime_error("MoE fused projection must be rank-3: " + name);
    }
    const int num_experts = fused.shape(0);
    const int fused_rows = fused.shape(1);
    if (fused_rows % 2 != 0) {
        throw std::runtime_error("MoE fused projection row count must be even: " + name);
    }
    const int rows = fused_rows / 2;
    const array gate = slice(
        fused,
        {0, 0, 0},
        {num_experts, rows, fused.shape(2)}
    );
    const array up = slice(
        fused,
        {0, rows, 0},
        {num_experts, fused_rows, fused.shape(2)}
    );
    return {gate, up};
}

void maybe_quantize_mxfp8_matrix(
    const std::unordered_map<std::string, array>& tensors,
    const BridgeModelConfig& cfg,
    const std::string& weight_key,
    const array& lhs_weight,
    bool enabled,
    array* out_q_w,
    array* out_q_scales,
    array* out_q_biases,
    bool* out_has_q,
    int* out_detected_bits = nullptr
) {
    if (!enabled) {
        *out_has_q = false;
        return;
    }
    if (out_q_biases) {
        *out_q_biases = array(0.0f);
    }
    g_qmm_attempts += 1;
    // Fast path for native MXFP8 checkpoints: reuse packed FP8 payload and
    // block scales directly for decode-time quantized matmul.
    if (ends_with(weight_key, ".weight")) {
        const std::string base = weight_key.substr(0, weight_key.size() - std::string(".weight").size());
        const std::string block_scale_key = base + ".weight_block_scale";
        const std::string weight_scale_key = base + ".weight_scale";
        auto raw_it = tensors.find(weight_key);
        auto scales_it = tensors.find(block_scale_key);
        if (scales_it == tensors.end()) {
            scales_it = tensors.find(weight_scale_key);
        }
        if (raw_it != tensors.end() && scales_it != tensors.end()) {
            const array& raw_weight = raw_it->second;
            const array& raw_scales = scales_it->second;
            if (raw_weight.ndim() == 2 && raw_scales.ndim() == 2) {
                const int rows = raw_weight.shape(0);
                const int cols = raw_weight.shape(1);
                const int expected_scale_cols = (cols + 31) / 32;
                if (raw_scales.shape(0) == rows && raw_scales.shape(1) == expected_scale_cols && cols % 4 == 0) {
                    try {
                        const array packed = reshape(view(raw_weight, uint32), {rows, cols / 4});
                        *out_q_w = packed;
                        *out_q_scales = astype(raw_scales, uint8);
                        *out_has_q = true;
                        g_qmm_success += 1;
                        return;
                    } catch (const std::exception&) {
                        // Fall through to generic quantize path.
                    }
                }
            }
        }
    }

    if (out_q_biases != nullptr &&
        has_nvfp4_metadata_for_key(tensors, weight_key) &&
        !has_grouped_affine_metadata_for_key(tensors, weight_key))
    {
        if (capture_native_nvfp4_quant(
                tensors,
                weight_key,
                out_q_w,
                out_q_scales,
                out_q_biases))
        {
            *out_has_q = true;
            g_qmm_success += 1;
            return;
        }
    }

    const bool enforce_grouped_affine_lossless = env_truthy("TALU_METAL_GAFFINE_LOSSLESS", true);
    if (enforce_grouped_affine_lossless && out_q_biases != nullptr) {
        if (maybe_capture_grouped_affine_quant(
                tensors,
                cfg,
                weight_key,
                lhs_weight,
                out_q_w,
                out_q_scales,
                out_q_biases,
                out_detected_bits))
        {
            *out_has_q = true;
            g_qmm_success += 1;
            return;
        }
        // If grouped-affine metadata exists but cannot be mapped directly
        // (for example synthetic fused tensors), avoid lossy re-quantization.
        if (has_grouped_affine_metadata_for_key(tensors, weight_key)) {
            *out_has_q = false;
            return;
        }
    }

    array quant_input = lhs_weight;
    const Dtype dt = quant_input.dtype();
    const bool is_float_input = (dt == float16 || dt == bfloat16 || dt == float32 || dt == float64);
    if (!is_float_input && ends_with(weight_key, ".weight")) {
        // Some converted checkpoints keep FP8 payload tensors as uint8 and
        // expose scales via metadata. Dequantize first when needed so MLX
        // quantize() receives a real floating tensor.
        if (has_nvfp4_metadata_for_key(tensors, weight_key)) {
            quant_input = dequantize_nvfp4_weight(tensors, weight_key);
        } else {
            quant_input = maybe_dequantize_grouped_affine_weight(tensors, cfg, weight_key, quant_input);
            quant_input = maybe_dequantize_fp8_weight(tensors, weight_key, quant_input);
        }
    }
    const Dtype qdt = quant_input.dtype();
    const bool quant_input_is_float = (qdt == float16 || qdt == bfloat16 || qdt == float32 || qdt == float64);
    if (!quant_input_is_float) {
        *out_has_q = false;
        return;
    }

    std::vector<array> q;
    try {
        q = quantize(
            quant_input,
            std::nullopt,
            std::nullopt,
            "mxfp8"
        );
    } catch (const std::exception&) {
        *out_has_q = false;
        return;
    }
    if (q.size() != 2) {
        throw std::runtime_error("mlx quantize(mxfp8) returned unexpected output count");
    }
    *out_q_w = q[0];
    *out_q_scales = q[1];
    *out_has_q = true;
    g_qmm_success += 1;
}
