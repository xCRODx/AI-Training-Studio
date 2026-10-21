#include "retbit/model.hpp"
#include "retbit/operators.hpp"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <numeric>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace retbit {

namespace {

float horizontal_sum(__m256 value) {
    const __m128 low = _mm256_castps256_ps128(value);
    const __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

const std::array<std::uint32_t, 256>& ternary_decode_table() {
    static const auto table = [] {
        std::array<std::uint32_t, 256> values {};
        for (std::size_t packed = 0; packed < values.size(); ++packed) {
            std::uint32_t decoded = 0;
            for (std::size_t index = 0; index < 4; ++index) {
                const auto code = static_cast<std::uint8_t>((packed >> (2 * index)) & 0x03U);
                const auto weight = static_cast<std::int8_t>(static_cast<int>(code) - 1);
                decoded |= static_cast<std::uint32_t>(static_cast<std::uint8_t>(weight)) << (8 * index);
            }
            values[packed] = decoded;
        }
        return values;
    }();
    return table;
}

float ternary_dot_avx2(const float* input, const std::uint8_t* packed, std::size_t count) {
    const auto& lookup = ternary_decode_table();
    __m256 accumulator = _mm256_setzero_ps();
    std::size_t index = 0;
    std::size_t packed_index = 0;
    for (; index + 32 <= count; index += 32, packed_index += 8) {
        const __m256i decoded = _mm256_setr_epi32(
            static_cast<int>(lookup[packed[packed_index]]),
            static_cast<int>(lookup[packed[packed_index + 1]]),
            static_cast<int>(lookup[packed[packed_index + 2]]),
            static_cast<int>(lookup[packed[packed_index + 3]]),
            static_cast<int>(lookup[packed[packed_index + 4]]),
            static_cast<int>(lookup[packed[packed_index + 5]]),
            static_cast<int>(lookup[packed[packed_index + 6]]),
            static_cast<int>(lookup[packed[packed_index + 7]])
        );
        const __m128i lower = _mm256_castsi256_si128(decoded);
        const __m128i upper = _mm256_extracti128_si256(decoded, 1);
        const __m256 weights0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lower));
        const __m256 weights1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lower, 8)));
        const __m256 weights2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(upper));
        const __m256 weights3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(upper, 8)));
        accumulator = _mm256_fmadd_ps(_mm256_loadu_ps(input + index), weights0, accumulator);
        accumulator = _mm256_fmadd_ps(_mm256_loadu_ps(input + index + 8), weights1, accumulator);
        accumulator = _mm256_fmadd_ps(_mm256_loadu_ps(input + index + 16), weights2, accumulator);
        accumulator = _mm256_fmadd_ps(_mm256_loadu_ps(input + index + 24), weights3, accumulator);
    }
    float sum = horizontal_sum(accumulator);
    for (; index < count; ++index) {
        const std::uint8_t code = (packed[index / 4] >> (2 * (index % 4))) & 0x03U;
        sum += input[index] * static_cast<float>(static_cast<int>(code) - 1);
    }
    return sum;
}

float bf16_dot_avx2(const float* input, const std::uint16_t* weights, std::size_t count) {
    __m256 accumulator = _mm256_setzero_ps();
    std::size_t index = 0;
    for (; index + 8 <= count; index += 8) {
        const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + index));
        const __m256i widened = _mm256_slli_epi32(_mm256_cvtepu16_epi32(packed), 16);
        const __m256 values = _mm256_castsi256_ps(widened);
        accumulator = _mm256_fmadd_ps(_mm256_loadu_ps(input + index), values, accumulator);
    }
    float sum = horizontal_sum(accumulator);
    for (; index < count; ++index) {
        const std::uint32_t bits = static_cast<std::uint32_t>(weights[index]) << 16U;
        float value = 0.0F;
        std::memcpy(&value, &bits, sizeof(value));
        sum += input[index] * value;
    }
    return sum;
}

void bf16_to_float_avx2(const std::uint16_t* input, float* output, std::size_t count) {
    std::size_t index = 0;
    for (; index + 8 <= count; index += 8) {
        const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + index));
        const __m256i widened = _mm256_slli_epi32(_mm256_cvtepu16_epi32(packed), 16);
        _mm256_storeu_ps(output + index, _mm256_castsi256_ps(widened));
    }
    for (; index < count; ++index) {
        const std::uint32_t bits = static_cast<std::uint32_t>(input[index]) << 16U;
        std::memcpy(output + index, &bits, sizeof(float));
    }
}

}  // namespace

class WorkerPool {
public:
    explicit WorkerPool(std::size_t total_threads)
        : total_threads_(total_threads == 0
            ? std::max<std::size_t>(1, std::thread::hardware_concurrency())
            : std::max<std::size_t>(1, total_threads)) {
        for (std::size_t index = 1; index < total_threads_; ++index) {
            threads_.emplace_back([this, index] { worker_loop(index); });
        }
    }

    ~WorkerPool() {
        {
            std::lock_guard<std::mutex> guard(mutex_);
            stopping_ = true;
            ++epoch_;
        }
        work_ready_.notify_all();
        for (auto& thread : threads_) thread.join();
    }

    void run(std::size_t count, const std::function<void(std::size_t)>& task) {
        if (threads_.empty() || count < 2) {
            for (std::size_t index = 0; index < count; ++index) task(index);
            return;
        }
        {
            std::lock_guard<std::mutex> guard(mutex_);
            task_ = task;
            count_ = count;
            completed_ = 0;
            ++epoch_;
        }
        work_ready_.notify_all();
        execute_current(0);
        std::unique_lock<std::mutex> lock(mutex_);
        work_done_.wait(lock, [this] { return completed_ == threads_.size(); });
    }

private:
    void execute_current(std::size_t worker_index) {
        const std::size_t begin = count_ * worker_index / total_threads_;
        const std::size_t end = count_ * (worker_index + 1) / total_threads_;
        for (std::size_t index = begin; index < end; ++index) task_(index);
    }

    void worker_loop(std::size_t worker_index) {
        std::size_t observed_epoch = 0;
        while (true) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                work_ready_.wait(lock, [this, observed_epoch] { return stopping_ || epoch_ != observed_epoch; });
                if (stopping_) return;
                observed_epoch = epoch_;
            }
            execute_current(worker_index);
            {
                std::lock_guard<std::mutex> guard(mutex_);
                ++completed_;
                if (completed_ == threads_.size()) work_done_.notify_one();
            }
        }
    }

    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable work_ready_;
    std::condition_variable work_done_;
    std::function<void(std::size_t)> task_;
    std::size_t total_threads_ = 1;
    std::size_t count_ = 0;
    std::size_t completed_ = 0;
    std::size_t epoch_ = 0;
    bool stopping_ = false;
};

RetBitModel::RetBitModel(const std::string& model_path, std::size_t threads)
    : file_(model_path),
      vocab_size_(file_.u32("retbit.vocab_size")),
      layer_count_(file_.u32("retbit.block_count")),
      embedding_length_(file_.u32("retbit.embedding_length")),
      head_count_(file_.u32("retbit.head_count")),
      head_dim_(file_.u32("retbit.head_dim")),
      ffn_length_(file_.u32("retbit.feed_forward_length")),
      rms_epsilon_(file_.f32("retbit.rms_norm_epsilon")),
      embeddings_(&file_.tensor("token_embd.weight")),
      output_norm_(&file_.tensor("output_norm.weight")),
      logits_(vocab_size_, 0.0F),
      workers_(std::make_unique<WorkerPool>(threads)) {
    if (file_.string("general.architecture") != "retbit") throw std::runtime_error("GGUF is not RetBit");
    if (head_count_ * head_dim_ != embedding_length_) throw std::runtime_error("Invalid RetBit head dimensions");
    layers_.reserve(layer_count_);
    for (std::uint32_t index = 0; index < layer_count_; ++index) {
        const std::string prefix = "blk." + std::to_string(index) + ".";
        layers_.push_back(Layer{
            &file_.tensor(prefix + "attn_norm.weight"),
            &file_.tensor(prefix + "ffn_norm.weight"),
            &file_.tensor(prefix + "retention_gamma"),
            linear(prefix + "retention_q", embedding_length_, embedding_length_),
            linear(prefix + "retention_k", embedding_length_, embedding_length_),
            linear(prefix + "retention_v", embedding_length_, embedding_length_),
            linear(prefix + "retention_output", embedding_length_, embedding_length_),
            linear(prefix + "ffn_up", embedding_length_, ffn_length_),
            linear(prefix + "ffn_down", ffn_length_, embedding_length_),
            std::vector<float>(head_count_ * head_dim_ * head_dim_, 0.0F),
        });
    }
}

RetBitModel::~RetBitModel() = default;

RetBitModel::Linear RetBitModel::linear(const std::string& prefix, std::size_t input_size, std::size_t output_size) const {
    const auto& weight = file_.tensor(prefix + ".weight");
    const auto& scale = file_.tensor(prefix + ".scale");
    const std::size_t expected_bytes = (input_size * output_size + 3) / 4;
    if (weight.type != GGMLType::I8 || weight.element_count() != expected_bytes) {
        throw std::runtime_error("Invalid ternary matrix: " + prefix);
    }
    return Linear{&weight, scale.value(0), input_size, output_size};
}

void RetBitModel::reset() {
    for (auto& layer : layers_) std::fill(layer.retention_state.begin(), layer.retention_state.end(), 0.0F);
}

void RetBitModel::apply_linear(const Linear& linear, const std::vector<float>& input, std::vector<float>& output) const {
    if (input.size() != linear.input_size) throw std::runtime_error("RetBit linear input shape mismatch");
    output.assign(linear.output_size, 0.0F);
    const auto* packed = reinterpret_cast<const std::uint8_t*>(linear.weight->i8_data());
    if (linear.input_size % 4 != 0) throw std::runtime_error("RetBit ternary rows must be byte aligned");
    const std::size_t row_bytes = linear.input_size / 4;
    workers_->run(linear.output_size, [&](std::size_t row) {
        output[row] = ternary_dot_avx2(input.data(), packed + row * row_bytes, linear.input_size) * linear.scale;
    });
}

void RetBitModel::rms_norm(const std::vector<float>& input, const TensorView& weight, std::vector<float>& output) const {
    float squares = 0.0F;
    for (const float value : input) squares += value * value;
    const float inverse_rms = 1.0F / std::sqrt(squares / static_cast<float>(input.size()) + rms_epsilon_);
    output.resize(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) output[index] = input[index] * inverse_rms * weight.value(index);
}

const std::vector<float>& RetBitModel::forward_token(std::uint32_t token_id) {
    if (token_id >= vocab_size_) throw std::out_of_range("Token outside RetBit vocabulary");
    std::vector<float> hidden(embedding_length_);
    const std::size_t embedding_offset = static_cast<std::size_t>(token_id) * embedding_length_;
    if (embeddings_->type == GGMLType::BF16) {
        const auto* source = reinterpret_cast<const std::uint16_t*>(embeddings_->data) + embedding_offset;
        bf16_to_float_avx2(source, hidden.data(), embedding_length_);
    } else {
        for (std::size_t index = 0; index < embedding_length_; ++index) hidden[index] = embeddings_->value(embedding_offset + index);
    }

    std::vector<float> normalized, q, k, v, retained(embedding_length_), projection, expanded, contracted;
    std::vector<float> gamma(head_count_);
    const RetentionShape shape {1, head_count_, 1, head_dim_};
    for (auto& layer : layers_) {
        rms_norm(hidden, *layer.attn_norm, normalized);
        apply_linear(layer.q, normalized, q);
        apply_linear(layer.k, normalized, k);
        apply_linear(layer.v, normalized, v);
        for (std::size_t head = 0; head < head_count_; ++head) gamma[head] = layer.gamma->value(head);
        recurrent_retention(q.data(), k.data(), v.data(), retained.data(), layer.retention_state.data(), gamma.data(), shape, false);
        apply_linear(layer.retention_output, retained, projection);
        for (std::size_t index = 0; index < hidden.size(); ++index) hidden[index] += projection[index];

        rms_norm(hidden, *layer.ffn_norm, normalized);
        apply_linear(layer.ffn_up, normalized, expanded);
        for (float& value : expanded) value = value / (1.0F + std::exp(-value));
        apply_linear(layer.ffn_down, expanded, contracted);
        for (std::size_t index = 0; index < hidden.size(); ++index) hidden[index] += contracted[index];
    }

    rms_norm(hidden, *output_norm_, normalized);
    if (embeddings_->type == GGMLType::BF16) {
        const auto* embedding_data = reinterpret_cast<const std::uint16_t*>(embeddings_->data);
        workers_->run(vocab_size_, [&](std::size_t token) {
            logits_[token] = bf16_dot_avx2(
                normalized.data(),
                embedding_data + token * embedding_length_,
                embedding_length_
            );
        });
    } else {
        workers_->run(vocab_size_, [&](std::size_t token) {
            float sum = 0.0F;
            const std::size_t offset = token * embedding_length_;
            for (std::size_t index = 0; index < embedding_length_; ++index) sum += normalized[index] * embeddings_->value(offset + index);
            logits_[token] = sum;
        });
    }
    return logits_;
}

std::uint32_t RetBitModel::sample(
    const std::vector<float>& logits,
    const std::vector<std::uint32_t>& history,
    const GenerationOptions& options,
    std::mt19937& generator
) const {
    std::vector<float> adjusted = logits;
    if (options.repetition_penalty != 1.0F) {
        for (const auto token : history) {
            if (token < adjusted.size()) adjusted[token] = adjusted[token] < 0.0F
                ? adjusted[token] * options.repetition_penalty
                : adjusted[token] / options.repetition_penalty;
        }
    }
    if (options.temperature <= 0.0F) {
        return static_cast<std::uint32_t>(std::distance(adjusted.begin(), std::max_element(adjusted.begin(), adjusted.end())));
    }
    const float maximum = *std::max_element(adjusted.begin(), adjusted.end());
    std::vector<std::pair<float, std::uint32_t>> probabilities;
    probabilities.reserve(adjusted.size());
    float total = 0.0F;
    for (std::uint32_t token = 0; token < adjusted.size(); ++token) {
        const float probability = std::exp((adjusted[token] - maximum) / options.temperature);
        probabilities.emplace_back(probability, token);
        total += probability;
    }
    const auto descending = [](const auto& left, const auto& right) { return left.first > right.first; };
    if (options.top_k > 0 && options.top_k < probabilities.size()) {
        const auto boundary = probabilities.begin() + static_cast<std::ptrdiff_t>(options.top_k);
        std::nth_element(probabilities.begin(), boundary, probabilities.end(), descending);
        probabilities.resize(options.top_k);
    }
    std::sort(probabilities.begin(), probabilities.end(), descending);
    total = 0.0F;
    for (const auto& item : probabilities) total += item.first;
    float cumulative = 0.0F;
    std::size_t keep = 0;
    const float threshold = std::clamp(options.top_p, 0.0001F, 1.0F) * total;
    while (keep < probabilities.size() && (cumulative < threshold || keep == 0)) cumulative += probabilities[keep++].first;
    std::uniform_real_distribution<float> distribution(0.0F, cumulative);
    float draw = distribution(generator);
    for (std::size_t index = 0; index < keep; ++index) {
        draw -= probabilities[index].first;
        if (draw <= 0.0F) return probabilities[index].second;
    }
    return probabilities[keep - 1].second;
}

std::vector<std::uint32_t> RetBitModel::generate(
    const std::vector<std::uint32_t>& prompt,
    const GenerationOptions& options,
    const std::function<void(std::uint32_t)>& on_token
) {
    if (prompt.empty()) throw std::invalid_argument("RetBit prompt tokens cannot be empty");
    reset();
    const std::vector<float>* current_logits = nullptr;
    std::vector<std::uint32_t> history = prompt;
    for (const auto token : prompt) current_logits = &forward_token(token);
    std::vector<std::uint32_t> generated;
    std::mt19937 generator(options.seed);
    for (std::size_t index = 0; index < options.max_new_tokens; ++index) {
        const auto token = sample(*current_logits, history, options, generator);
        if (token == options.eos_token_id) break;
        generated.push_back(token);
        if (on_token) on_token(token);
        history.push_back(token);
        if (index + 1 < options.max_new_tokens) current_logits = &forward_token(token);
    }
    return generated;
}

}  // namespace retbit
