// Shared attention helpers for dense and quantized model paths.

#pragma once

#include "compute_common.h"
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

static constexpr size_t kGqaIndexCacheMaxEntries = 64;

static inline uint64_t gqa_index_cache_key(int q_heads, int kv_heads) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(q_heads)) << 32) |
           static_cast<uint64_t>(static_cast<uint32_t>(kv_heads));
}

static inline std::unordered_map<uint64_t, array>& gqa_index_cache_store() {
    static thread_local std::unordered_map<uint64_t, array> gqa_index_cache;
    return gqa_index_cache;
}

static inline void gqa_index_cache_clear() {
    gqa_index_cache_store().clear();
}

static inline size_t gqa_index_cache_size() {
    return gqa_index_cache_store().size();
}

static inline size_t gqa_index_cache_max_entries() {
    return kGqaIndexCacheMaxEntries;
}

static inline array gqa_cached_gather_indices(int q_heads, int kv_heads) {
    if (q_heads <= 0 || kv_heads <= 0 || (q_heads % kv_heads) != 0) {
        throw std::invalid_argument("invalid GQA head layout");
    }

    auto& gqa_index_cache = gqa_index_cache_store();
    const uint64_t key = gqa_index_cache_key(q_heads, kv_heads);
    auto it = gqa_index_cache.find(key);
    if (it != gqa_index_cache.end()) {
        return it->second;
    }
    if (gqa_index_cache.size() >= kGqaIndexCacheMaxEntries) {
        gqa_index_cache.clear();
    }

    const int heads_per_kv = q_heads / kv_heads;
    std::vector<int32_t> gather_idx(static_cast<size_t>(q_heads));
    for (int head_idx = 0; head_idx < q_heads; head_idx++) {
        gather_idx[static_cast<size_t>(head_idx)] = static_cast<int32_t>(head_idx / heads_per_kv);
    }
    array idx = array(gather_idx.data(), {q_heads}, int32);
    auto [inserted_it, _] = gqa_index_cache.emplace(key, idx);
    return inserted_it->second;
}

static inline void gqa_expand_attention_kv(
    const array& q,
    const array& k_in,
    const array& v_in,
    array* k_out,
    array* v_out
) {
    if (k_out == nullptr || v_out == nullptr) {
        throw std::invalid_argument("null gqa output");
    }

    const int q_heads = q.shape(1);
    const int kv_heads = k_in.shape(1);
    if (q_heads == kv_heads) {
        *k_out = k_in;
        *v_out = v_in;
        return;
    }
    if (v_in.shape(1) != kv_heads) {
        throw std::invalid_argument("invalid GQA head layout");
    }

    array idx = gqa_cached_gather_indices(q_heads, kv_heads);
    *k_out = take(k_in, idx, 1);
    *v_out = take(v_in, idx, 1);
}
