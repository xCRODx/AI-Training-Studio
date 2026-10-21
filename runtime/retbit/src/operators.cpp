#include "retbit/operators.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace retbit {
namespace {

std::size_t token_offset(
    const RetentionShape& shape,
    std::size_t batch,
    std::size_t head,
    std::size_t token
) {
    return (((batch * shape.heads) + head) * shape.sequence + token) * shape.head_dim;
}

std::size_t state_offset(
    const RetentionShape& shape,
    std::size_t batch,
    std::size_t head
) {
    return ((batch * shape.heads) + head) * shape.head_dim * shape.head_dim;
}

}  // namespace

void recurrent_retention(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    float* state,
    const float* gamma,
    RetentionShape shape,
    bool reset_state
) {
    if (!q || !k || !v || !output || !state || !gamma) {
        throw std::invalid_argument("retbit: retention received a null buffer");
    }
    if (shape.batch == 0 || shape.heads == 0 || shape.sequence == 0 || shape.head_dim == 0) {
        throw std::invalid_argument("retbit: retention dimensions must be positive");
    }

    const std::size_t matrix_size = shape.head_dim * shape.head_dim;
    if (reset_state) {
        std::fill(state, state + shape.batch * shape.heads * matrix_size, 0.0F);
    }

    const float inverse_sqrt_dim = 1.0F / std::sqrt(static_cast<float>(shape.head_dim));
    for (std::size_t batch = 0; batch < shape.batch; ++batch) {
        for (std::size_t head = 0; head < shape.heads; ++head) {
            float* head_state = state + state_offset(shape, batch, head);
            const float decay = gamma[head];
            for (std::size_t token = 0; token < shape.sequence; ++token) {
                const std::size_t offset = token_offset(shape, batch, head, token);
                const float* q_token = q + offset;
                const float* k_token = k + offset;
                const float* v_token = v + offset;
                float* out_token = output + offset;

                for (std::size_t row = 0; row < shape.head_dim; ++row) {
                    for (std::size_t column = 0; column < shape.head_dim; ++column) {
                        const std::size_t index = row * shape.head_dim + column;
                        head_state[index] = decay * head_state[index]
                            + k_token[row] * v_token[column];
                    }
                }

                for (std::size_t column = 0; column < shape.head_dim; ++column) {
                    float sum = 0.0F;
                    for (std::size_t row = 0; row < shape.head_dim; ++row) {
                        sum += q_token[row] * head_state[row * shape.head_dim + column];
                    }
                    out_token[column] = sum * inverse_sqrt_dim;
                }
            }
        }
    }
}

void ternary_linear(
    const float* input,
    const std::int8_t* ternary_weight,
    float weight_scale,
    float* output,
    std::size_t rows,
    std::size_t in_features,
    std::size_t out_features
) {
    if (!input || !ternary_weight || !output) {
        throw std::invalid_argument("retbit: ternary linear received a null buffer");
    }
    if (rows == 0 || in_features == 0 || out_features == 0 || weight_scale < 0.0F) {
        throw std::invalid_argument("retbit: invalid ternary linear dimensions or scale");
    }

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t output_feature = 0; output_feature < out_features; ++output_feature) {
            float sum = 0.0F;
            const std::int8_t* weight_row = ternary_weight + output_feature * in_features;
            const float* input_row = input + row * in_features;
            for (std::size_t input_feature = 0; input_feature < in_features; ++input_feature) {
                const std::int8_t weight = weight_row[input_feature];
                if (weight < -1 || weight > 1) {
                    throw std::invalid_argument("retbit: a ternary weight is outside {-1, 0, 1}");
                }
                sum += input_row[input_feature] * static_cast<float>(weight);
            }
            output[row * out_features + output_feature] = sum * weight_scale;
        }
    }
}

}  // namespace retbit

