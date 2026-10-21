#pragma once

#include <cstddef>
#include <cstdint>

namespace retbit {

struct RetentionShape {
    std::size_t batch;
    std::size_t heads;
    std::size_t sequence;
    std::size_t head_dim;
};

// Layouts:
// q/k/v/output: [batch, heads, sequence, head_dim]
// state:        [batch, heads, head_dim, head_dim]
// gamma:        [heads]
//
// Implements the exact recurrent form of the causal parallel retention used by
// src.model.MultiScaleRetention:
//   S_t = gamma * S_(t-1) + outer(k_t, v_t)
//   y_t = q_t * S_t / sqrt(head_dim)
// Keeping state between calls enables token-by-token decoding without a KV cache.
void recurrent_retention(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    float* state,
    const float* gamma,
    RetentionShape shape,
    bool reset_state = false
);

// Reference CPU operator for the project's BitLinear inference representation.
// input: [rows, in_features], ternary_weight: [out_features, in_features].
// The converter will store ternary values {-1, 0, 1} plus the tensor scale.
void ternary_linear(
    const float* input,
    const std::int8_t* ternary_weight,
    float weight_scale,
    float* output,
    std::size_t rows,
    std::size_t in_features,
    std::size_t out_features
);

}  // namespace retbit

