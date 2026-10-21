#include "retbit/operators.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

bool close(float actual, float expected, float tolerance = 1.0e-5F) {
    return std::abs(actual - expected) <= tolerance;
}

int test_ternary_linear() {
    const std::vector<float> input {1.0F, 2.0F, -1.0F, 0.5F, -2.0F, 3.0F};
    const std::vector<std::int8_t> weights {1, 0, -1, -1, 1, 1};
    std::vector<float> output(4, 0.0F);
    retbit::ternary_linear(input.data(), weights.data(), 0.5F, output.data(), 2, 3, 2);
    const std::vector<float> expected {1.0F, 0.0F, -1.25F, 0.25F};
    for (std::size_t index = 0; index < output.size(); ++index) {
        if (!close(output[index], expected[index])) {
            std::cerr << "ternary_linear mismatch at " << index << ": "
                      << output[index] << " != " << expected[index] << '\n';
            return 1;
        }
    }
    return 0;
}

int test_retention_matches_parallel_reference() {
    const retbit::RetentionShape shape {1, 2, 4, 3};
    const std::size_t count = shape.batch * shape.heads * shape.sequence * shape.head_dim;
    std::vector<float> q(count);
    std::vector<float> k(count);
    std::vector<float> v(count);
    for (std::size_t index = 0; index < count; ++index) {
        q[index] = static_cast<float>(static_cast<int>(index % 7) - 3) * 0.07F;
        k[index] = static_cast<float>(static_cast<int>(index % 5) - 2) * 0.09F;
        v[index] = static_cast<float>(static_cast<int>(index % 11) - 5) * 0.04F;
    }
    const std::vector<float> gamma {0.75F, 0.9F};
    std::vector<float> recurrent(count, 0.0F);
    std::vector<float> state(shape.batch * shape.heads * shape.head_dim * shape.head_dim, 0.0F);
    retbit::recurrent_retention(
        q.data(), k.data(), v.data(), recurrent.data(), state.data(), gamma.data(), shape, true
    );

    std::vector<float> parallel(count, 0.0F);
    const float scale = 1.0F / std::sqrt(static_cast<float>(shape.head_dim));
    for (std::size_t head = 0; head < shape.heads; ++head) {
        for (std::size_t target = 0; target < shape.sequence; ++target) {
            for (std::size_t source = 0; source <= target; ++source) {
                const std::size_t q_offset = (head * shape.sequence + target) * shape.head_dim;
                const std::size_t kv_offset = (head * shape.sequence + source) * shape.head_dim;
                float score = 0.0F;
                for (std::size_t dim = 0; dim < shape.head_dim; ++dim) {
                    score += q[q_offset + dim] * k[kv_offset + dim];
                }
                score *= scale * std::pow(gamma[head], static_cast<float>(target - source));
                for (std::size_t dim = 0; dim < shape.head_dim; ++dim) {
                    parallel[q_offset + dim] += score * v[kv_offset + dim];
                }
            }
        }
    }

    for (std::size_t index = 0; index < count; ++index) {
        if (!close(recurrent[index], parallel[index], 2.0e-5F)) {
            std::cerr << "retention mismatch at " << index << ": "
                      << recurrent[index] << " != " << parallel[index] << '\n';
            return 1;
        }
    }
    return 0;
}

int test_chunked_retention_keeps_state() {
    const retbit::RetentionShape full_shape {1, 1, 2, 2};
    const std::vector<float> q {1.0F, 0.5F, -0.25F, 0.75F};
    const std::vector<float> k {0.4F, -0.2F, 0.1F, 0.3F};
    const std::vector<float> v {0.8F, 0.6F, -0.5F, 0.2F};
    const float gamma = 0.85F;
    std::vector<float> full_output(4, 0.0F);
    std::vector<float> full_state(4, 0.0F);
    retbit::recurrent_retention(
        q.data(), k.data(), v.data(), full_output.data(), full_state.data(), &gamma, full_shape, true
    );

    const retbit::RetentionShape token_shape {1, 1, 1, 2};
    std::vector<float> chunked_output(4, 0.0F);
    std::vector<float> chunked_state(4, 0.0F);
    retbit::recurrent_retention(
        q.data(), k.data(), v.data(), chunked_output.data(), chunked_state.data(), &gamma, token_shape, true
    );
    retbit::recurrent_retention(
        q.data() + 2, k.data() + 2, v.data() + 2, chunked_output.data() + 2,
        chunked_state.data(), &gamma, token_shape, false
    );

    for (std::size_t index = 0; index < full_output.size(); ++index) {
        if (!close(full_output[index], chunked_output[index])) {
            std::cerr << "chunked retention mismatch at " << index << '\n';
            return 1;
        }
    }
    return 0;
}

}  // namespace

int main() {
    if (test_ternary_linear() != 0) return 1;
    if (test_retention_matches_parallel_reference() != 0) return 1;
    if (test_chunked_retention_keeps_state() != 0) return 1;
    std::cout << "retbit core: all operator tests passed\n";
    return 0;
}
