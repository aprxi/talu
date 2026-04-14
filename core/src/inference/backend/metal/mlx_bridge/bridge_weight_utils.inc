// bridge_weight_utils.inc — Greedy decode helper, file I/O, weight file discovery,
// tensor lookup helpers, model prefix detection.

array next_token_greedy(const array& logits_last) {
    return astype(argmax(logits_last, -1), int32);
}

bool ends_with(const std::string& s, const std::string& suffix) {
    if (suffix.size() > s.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open file: " + path.string());
    }
    std::string text;
    in.seekg(0, std::ios::end);
    text.resize(static_cast<size_t>(in.tellg()));
    in.seekg(0, std::ios::beg);
    in.read(text.data(), static_cast<std::streamsize>(text.size()));
    if (!in.good() && !in.eof()) {
        throw std::runtime_error("failed to read file: " + path.string());
    }
    return text;
}

bool model_has_talu_meta_scheme(const std::string& model_path, const char* scheme_name) {
    if (scheme_name == nullptr || std::strlen(scheme_name) == 0) return false;
    const std::filesystem::path meta_path = std::filesystem::path(model_path) / "talu_meta.json";
    if (!std::filesystem::exists(meta_path)) return false;
    std::string text;
    try {
        text = read_file(meta_path);
    } catch (...) {
        return false;
    }

    static const std::regex scheme_re("\\\"scheme\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");
    std::smatch match;
    if (!std::regex_search(text, match, scheme_re) || match.size() < 2) return false;

    std::string actual = match[1].str();
    std::string expected = scheme_name;
    std::transform(actual.begin(), actual.end(), actual.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    std::transform(expected.begin(), expected.end(), expected.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return actual == expected;
}

std::vector<std::string> discover_weight_files(const std::string& model_path) {
    namespace fs = std::filesystem;
    const fs::path model_dir(model_path);

    std::unordered_set<std::string> unique;

    const fs::path direct = model_dir / "model.safetensors";
    if (fs::exists(direct)) unique.insert("model.safetensors");

    const fs::path index_file = model_dir / "model.safetensors.index.json";
    if (fs::exists(index_file)) {
        const std::string json = read_file(index_file);
        static const std::regex re("\\\"([^\\\"]+\\.safetensors)\\\"");
        for (std::sregex_iterator it(json.begin(), json.end(), re), end; it != end; ++it) {
            const std::string rel = (*it)[1].str();
            const fs::path abs = model_dir / rel;
            if (fs::exists(abs)) unique.insert(rel);
        }
    }

    if (unique.empty()) {
        for (const auto& entry : fs::directory_iterator(model_dir)) {
            if (!entry.is_regular_file()) continue;
            const std::string name = entry.path().filename().string();
            if (name.size() >= 12 && name.substr(name.size() - 12) == ".safetensors") {
                unique.insert(name);
            }
        }
    }

    std::vector<std::string> files(unique.begin(), unique.end());
    std::sort(files.begin(), files.end());
    if (files.empty()) {
        throw std::runtime_error("no safetensors weights found in model directory: " + model_path);
    }
    return files;
}

std::unordered_map<std::string, array> load_weight_tensors(const std::string& model_path) {
    const std::vector<std::string> weight_files = discover_weight_files(model_path);

    std::unordered_map<std::string, array> tensors;
    for (const std::string& rel : weight_files) {
        const std::string abs = (std::filesystem::path(model_path) / rel).string();
        auto loaded = load_safetensors(abs);
        for (auto& kv : loaded.first) {
            auto it = tensors.find(kv.first);
            if (it == tensors.end()) {
                tensors.emplace(kv.first, std::move(kv.second));
            } else {
                it->second = std::move(kv.second);
            }
        }
    }

    if (tensors.empty()) throw std::runtime_error("no tensors were loaded from safetensors files");
    return tensors;
}

std::vector<int> resolve_shared_kv_source_layers(const BridgeModelConfig& cfg) {
    const int n_layers = cfg.num_hidden_layers;
    std::vector<int> sources(static_cast<size_t>(std::max(0, n_layers)), -1);
    if (cfg.num_kv_shared_layers <= 0 || n_layers <= 0) return sources;
    if (static_cast<int>(cfg.layer_types.size()) != n_layers) return sources;

    const int shared_count = std::min(cfg.num_kv_shared_layers, n_layers);
    if (shared_count <= 0 || shared_count >= n_layers) return sources;
    const int first_shared_layer = n_layers - shared_count;
    if (first_shared_layer <= 0) return sources;

    for (int layer_idx = first_shared_layer; layer_idx < n_layers; ++layer_idx) {
        const std::string& target_layer_type = cfg.layer_types[static_cast<size_t>(layer_idx)];
        for (int src = first_shared_layer - 1; src >= 0; --src) {
            if (cfg.layer_types[static_cast<size_t>(src)] == target_layer_type) {
                sources[static_cast<size_t>(layer_idx)] = src;
                break;
            }
        }
    }
    return sources;
}

void sanitize_model_tensors(
    std::unordered_map<std::string, array>& tensors,
    bool tie_word_embeddings,
    bool allow_norm_shift
) {
    bool has_mtp_weights = false;
    bool has_unsanitized_conv1d = false;

    std::unordered_map<std::string, array> filtered;
    filtered.reserve(tensors.size());

    for (auto& kv : tensors) {
        const std::string& k = kv.first;
        if (k.find("mtp.") != std::string::npos) {
            has_mtp_weights = true;
            continue;
        }
        if (k.rfind("vision_tower", 0) == 0 ||
            k.rfind("model.visual", 0) == 0 ||
            k.rfind("model.vision_tower", 0) == 0 ||
            k.rfind("model.multi_modal_projector", 0) == 0) {
            continue;
        }
        if ((k.find("conv1d.weight") != std::string::npos || k.find("conv.conv.weight") != std::string::npos) &&
            kv.second.ndim() == 3 && kv.second.shape(2) != 1) {
            has_unsanitized_conv1d = true;
        }
        filtered.emplace(k, kv.second);
    }

    if (tie_word_embeddings) {
        filtered.erase("lm_head.weight");
        filtered.erase("model.language_model.lm_head.weight");
        filtered.erase("model.lm_head.weight");
    }

    const bool should_shift_norm = allow_norm_shift && (has_mtp_weights || has_unsanitized_conv1d);

    const std::vector<std::string> norm_suffixes = {
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        "model.language_model.norm.weight",
        "model.norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    };

    for (auto& kv : filtered) {
        const std::string& key = kv.first;
        array v = kv.second;

        if ((key.find("conv1d.weight") != std::string::npos || key.find("conv.conv.weight") != std::string::npos) &&
            v.ndim() == 3 && v.shape(2) != 1) {
            // model.sanitize: v = v.moveaxis(2, 1)
            v = transpose(v, {0, 2, 1});
        }

        if (should_shift_norm && v.ndim() == 1) {
            bool match = false;
            for (const auto& sfx : norm_suffixes) {
                if (ends_with(key, sfx)) {
                    match = true;
                    break;
                }
            }
            if (match) v = v + array(1.0f, v.dtype());
        }

        kv.second = v;
    }

    tensors = std::move(filtered);
}

const array& require_tensor(const std::unordered_map<std::string, array>& tensors, const std::string& key) {
    auto it = tensors.find(key);
    if (it == tensors.end() && ends_with(key, ".weight")) {
        const std::string packed_key = key.substr(0, key.size() - std::string(".weight").size()) + ".weight_packed";
        it = tensors.find(packed_key);
    }
    if (it == tensors.end()) throw std::runtime_error("missing required tensor: " + key);
    return it->second;
}

const array& require_any_tensor(
    const std::unordered_map<std::string, array>& tensors,
    const std::vector<std::string>& keys,
    const std::string& context
) {
    for (const auto& key : keys) {
        auto it = tensors.find(key);
        if (it != tensors.end()) return it->second;
    }
    throw std::runtime_error("missing required tensor for " + context);
}

bool has_tensor(const std::unordered_map<std::string, array>& tensors, const std::string& key) {
    if (tensors.find(key) != tensors.end()) return true;
    if (ends_with(key, ".weight")) {
        const std::string packed_key = key.substr(0, key.size() - std::string(".weight").size()) + ".weight_packed";
        return tensors.find(packed_key) != tensors.end();
    }
    return false;
}

std::string detect_text_prefix(const std::unordered_map<std::string, array>& tensors) {
    if (has_tensor(tensors, "model.language_model.embed_tokens.weight")) {
        return "model.language_model.";
    }
    if (has_tensor(tensors, "model.embed_tokens.weight")) {
        return "model.";
    }

    // Fallback for converted checkpoints that wrap text tensors under repeated
    // namespaces (for example model.language_model.language_model.*).
    const std::string embed_suffix = "embed_tokens.weight";
    std::string best_prefix;
    size_t embed_candidate_count = 0;
    std::string first_embed_key;
    for (const auto& kv : tensors) {
        const std::string& key = kv.first;
        if (!ends_with(key, embed_suffix)) continue;
        embed_candidate_count += 1;
        if (first_embed_key.empty()) first_embed_key = key;
        const std::string prefix = key.substr(0, key.size() - embed_suffix.size());
        if (prefix.empty()) continue;
        if (!has_tensor(tensors, prefix + "layers.0.input_layernorm.weight")) continue;
        if (prefix.size() > best_prefix.size()) {
            best_prefix = prefix;
        }
    }
    if (!best_prefix.empty()) {
        return best_prefix;
    }

    throw std::runtime_error(
        "missing embed_tokens.weight for known text prefixes"
        " (embed_candidates=" + std::to_string(embed_candidate_count) +
        (first_embed_key.empty() ? "" : ", first_embed_key=" + first_embed_key) +
        ")"
    );
}
