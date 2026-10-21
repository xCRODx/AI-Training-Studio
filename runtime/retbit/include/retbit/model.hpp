#pragma once

#include "retbit/gguf.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace retbit {

class WorkerPool;

struct GenerationOptions {
    std::size_t max_new_tokens = 32;
    float temperature = 0.7F;
    float top_p = 0.9F;
    std::size_t top_k = 50;
    float repetition_penalty = 1.05F;
    std::uint32_t seed = 42;
    std::uint32_t eos_token_id = 1;
};

class RetBitModel {
public:
    explicit RetBitModel(const std::string& model_path, std::size_t threads = 0);
    ~RetBitModel();

    void reset();
    const std::vector<float>& forward_token(std::uint32_t token_id);
    std::vector<std::uint32_t> generate(
        const std::vector<std::uint32_t>& prompt,
        const GenerationOptions& options,
        const std::function<void(std::uint32_t)>& on_token = {}
    );

    std::uint32_t vocab_size() const noexcept { return vocab_size_; }
    std::uint32_t layer_count() const noexcept { return layer_count_; }
    std::uint32_t embedding_length() const noexcept { return embedding_length_; }

private:
    struct Linear {
        const TensorView* weight;
        float scale;
        std::size_t input_size;
        std::size_t output_size;
    };

    struct Layer {
        const TensorView* attn_norm;
        const TensorView* ffn_norm;
        const TensorView* gamma;
        Linear q;
        Linear k;
        Linear v;
        Linear retention_output;
        Linear ffn_up;
        Linear ffn_down;
        std::vector<float> retention_state;
    };

    GGUFFile file_;
    std::uint32_t vocab_size_;
    std::uint32_t layer_count_;
    std::uint32_t embedding_length_;
    std::uint32_t head_count_;
    std::uint32_t head_dim_;
    std::uint32_t ffn_length_;
    float rms_epsilon_;
    const TensorView* embeddings_;
    const TensorView* output_norm_;
    std::vector<Layer> layers_;
    std::vector<float> logits_;
    std::unique_ptr<WorkerPool> workers_;

    Linear linear(const std::string& prefix, std::size_t input_size, std::size_t output_size) const;
    void apply_linear(const Linear& linear, const std::vector<float>& input, std::vector<float>& output) const;
    void rms_norm(const std::vector<float>& input, const TensorView& weight, std::vector<float>& output) const;
    std::uint32_t sample(
        const std::vector<float>& logits,
        const std::vector<std::uint32_t>& history,
        const GenerationOptions& options,
        std::mt19937& generator
    ) const;
};

}  // namespace retbit
