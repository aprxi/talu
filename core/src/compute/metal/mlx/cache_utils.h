// Shared cache growth helpers for MLX compute paths.

#pragma once

#include "model_state.h"
#include <string>

static inline int mlx_round_up_step(size_t value, int step) {
    const size_t s = static_cast<size_t>(step);
    return static_cast<int>(((value + s - 1) / s) * s);
}

static inline int mlx_next_cache_capacity(
    const CacheLayer& layer,
    size_t required,
    int current_capacity,
    const char* scope_tag
) {
    if (required == 0) return current_capacity;

    const size_t current = current_capacity > 0 ? static_cast<size_t>(current_capacity) : 0;

    if (layer.max_seq_len > 0) {
        const size_t max_cap = layer.max_seq_len;
        if (required > max_cap) {
            throw std::invalid_argument(std::string(scope_tag) + " kv cache capacity exceeded");
        }
        if (current == 0) return static_cast<int>(max_cap);
        if (required > current) return static_cast<int>(max_cap);
        return static_cast<int>(current);
    }

    const size_t step = static_cast<size_t>(layer.step);
    size_t capacity = current;
    if (capacity == 0) {
        const size_t base = std::max(required, step);
        capacity = static_cast<size_t>(mlx_round_up_step(base, layer.step));
    }
    while (capacity < required) {
        const size_t doubled = capacity * 2;
        if (doubled <= capacity) {
            throw std::invalid_argument(std::string(scope_tag) + " kv cache capacity exceeded");
        }
        capacity = doubled;
    }
    return static_cast<int>(capacity);
}
