#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cfloat>

namespace {

constexpr int kWarpsPerBlock = 8;
constexpr int kSamplerThreads = 256;
constexpr int kMaximumTopK = 256;

__global__ void ternary_linear_bf16_kernel(
    const __nv_bfloat16* input,
    const std::uint8_t* packed_weight,
    float scale,
    __nv_bfloat16* output,
    int rows,
    int input_size,
    int output_size
) {
    const int warp = threadIdx.x / warpSize;
    const int lane = threadIdx.x % warpSize;
    const int output_row = blockIdx.x * kWarpsPerBlock + warp;
    const int batch_row = blockIdx.y;
    if (batch_row >= rows || output_row >= output_size) return;

    const __nv_bfloat16* input_row = input + static_cast<std::size_t>(batch_row) * input_size;
    const int packed_columns = input_size / 4;
    const std::uint8_t* weight_row = packed_weight
        + static_cast<std::size_t>(output_row) * packed_columns;
    float sum = 0.0F;
    for (int group = lane; group < packed_columns; group += warpSize) {
        const std::uint8_t packed = weight_row[group];
        const int base = group * 4;
#pragma unroll
        for (int index = 0; index < 4; ++index) {
            const int weight = static_cast<int>((packed >> (2 * index)) & 0x03U) - 1;
            sum = fmaf(__bfloat162float(input_row[base + index]), static_cast<float>(weight), sum);
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFFU, sum, offset);
    }
    if (lane == 0) {
        output[static_cast<std::size_t>(batch_row) * output_size + output_row]
            = __float2bfloat16(sum * scale);
    }
}

__global__ void sample_topk_bf16_kernel(
    const __nv_bfloat16* logits,
    float* workspace,
    std::uint8_t* repetition_mask,
    float repetition_penalty,
    float temperature,
    float top_p,
    int top_k,
    float uniform,
    std::int64_t* output,
    int vocabulary_size
) {
    __shared__ float candidate_values[kSamplerThreads];
    __shared__ int candidate_ids[kSamplerThreads];
    __shared__ float top_values[kMaximumTopK];
    __shared__ int top_ids[kMaximumTopK];

    const int thread = threadIdx.x;
    for (int token = thread; token < vocabulary_size; token += blockDim.x) {
        float value = __bfloat162float(logits[token]);
        if (repetition_mask[token]) {
            value = value < 0.0F ? value * repetition_penalty : value / repetition_penalty;
        }
        workspace[token] = value / temperature;
    }
    __syncthreads();

    for (int rank = 0; rank < top_k; ++rank) {
        float best_value = -FLT_MAX;
        int best_id = -1;
        for (int token = thread; token < vocabulary_size; token += blockDim.x) {
            const float value = workspace[token];
            if (value > best_value) {
                best_value = value;
                best_id = token;
            }
        }
        candidate_values[thread] = best_value;
        candidate_ids[thread] = best_id;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (thread < stride && candidate_values[thread + stride] > candidate_values[thread]) {
                candidate_values[thread] = candidate_values[thread + stride];
                candidate_ids[thread] = candidate_ids[thread + stride];
            }
            __syncthreads();
        }
        if (thread == 0) {
            top_values[rank] = candidate_values[0];
            top_ids[rank] = candidate_ids[0];
            workspace[candidate_ids[0]] = -FLT_MAX;
        }
        __syncthreads();
    }

    if (thread == 0) {
        const float maximum = top_values[0];
        float total = 0.0F;
        for (int rank = 0; rank < top_k; ++rank) total += expf(top_values[rank] - maximum);
        const float nucleus = fminf(fmaxf(top_p, 0.0001F), 1.0F) * total;
        float retained = 0.0F;
        int keep = 0;
        while (keep < top_k && (retained < nucleus || keep == 0)) {
            retained += expf(top_values[keep] - maximum);
            ++keep;
        }
        float draw = fminf(fmaxf(uniform, 0.0F), 0.99999994F) * retained;
        int selected = top_ids[keep - 1];
        for (int rank = 0; rank < keep; ++rank) {
            draw -= expf(top_values[rank] - maximum);
            if (draw <= 0.0F) {
                selected = top_ids[rank];
                break;
            }
        }
        output[0] = static_cast<std::int64_t>(selected);
        repetition_mask[selected] = 1;
    }
}

}  // namespace

extern "C" __declspec(dllexport) int retbit_ternary_linear_bf16(
    const void* input,
    const void* packed_weight,
    float scale,
    void* output,
    int rows,
    int input_size,
    int output_size,
    void* stream
) {
    if (!input || !packed_weight || !output || input_size <= 0 || input_size % 4 != 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const dim3 block(kWarpsPerBlock * 32);
    const dim3 grid((output_size + kWarpsPerBlock - 1) / kWarpsPerBlock, rows);
    ternary_linear_bf16_kernel<<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<const std::uint8_t*>(packed_weight),
        scale,
        static_cast<__nv_bfloat16*>(output),
        rows,
        input_size,
        output_size
    );
    return static_cast<int>(cudaPeekAtLastError());
}

extern "C" __declspec(dllexport) const char* retbit_cuda_error_string(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}

extern "C" __declspec(dllexport) int retbit_sample_topk_bf16(
    const void* logits,
    void* workspace,
    void* repetition_mask,
    float repetition_penalty,
    float temperature,
    float top_p,
    int top_k,
    float uniform,
    void* output,
    int vocabulary_size,
    void* stream
) {
    if (!logits || !workspace || !repetition_mask || !output || vocabulary_size <= 0
        || top_k <= 0 || top_k > kMaximumTopK || top_k > vocabulary_size) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    sample_topk_bf16_kernel<<<1, kSamplerThreads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(logits),
        static_cast<float*>(workspace),
        static_cast<std::uint8_t*>(repetition_mask),
        repetition_penalty,
        fmaxf(temperature, 1e-5F),
        top_p,
        top_k,
        uniform,
        static_cast<std::int64_t*>(output),
        vocabulary_size
    );
    return static_cast<int>(cudaPeekAtLastError());
}
