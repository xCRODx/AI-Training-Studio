#include "retbit/gguf.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace retbit {
namespace {

class Reader {
public:
    explicit Reader(const std::vector<std::uint8_t>& bytes) : bytes_(bytes), cursor_(0) {}

    template <typename T>
    T read() {
        require(sizeof(T));
        T value {};
        std::memcpy(&value, bytes_.data() + cursor_, sizeof(T));
        cursor_ += sizeof(T);
        return value;
    }

    std::string string() {
        const auto length = read<std::uint64_t>();
        if (length > std::numeric_limits<std::size_t>::max()) throw std::runtime_error("GGUF string too large");
        require(static_cast<std::size_t>(length));
        std::string result(reinterpret_cast<const char*>(bytes_.data() + cursor_), static_cast<std::size_t>(length));
        cursor_ += static_cast<std::size_t>(length);
        return result;
    }

    void skip_value(std::uint32_t type) {
        switch (type) {
            case 0: case 1: case 7: require(1); cursor_ += 1; break;
            case 2: case 3: require(2); cursor_ += 2; break;
            case 4: case 5: case 6: require(4); cursor_ += 4; break;
            case 8: (void) string(); break;
            case 9: {
                const auto element_type = read<std::uint32_t>();
                const auto count = read<std::uint64_t>();
                for (std::uint64_t index = 0; index < count; ++index) skip_value(element_type);
                break;
            }
            case 10: case 11: case 12: require(8); cursor_ += 8; break;
            default: throw std::runtime_error("Unsupported GGUF metadata type: " + std::to_string(type));
        }
    }

    std::size_t position() const noexcept { return cursor_; }

private:
    void require(std::size_t count) const {
        if (cursor_ > bytes_.size() || count > bytes_.size() - cursor_) {
            throw std::runtime_error("Unexpected end of GGUF file");
        }
    }
    const std::vector<std::uint8_t>& bytes_;
    std::size_t cursor_;
};

struct TensorInfo {
    std::string name;
    std::vector<std::uint64_t> dimensions;
    GGMLType type;
    std::uint64_t offset;
};

std::size_t type_size(GGMLType type) {
    switch (type) {
        case GGMLType::F32: return 4;
        case GGMLType::F16: return 2;
        case GGMLType::I8: return 1;
        case GGMLType::BF16: return 2;
    }
    throw std::runtime_error("Unsupported RetBit GGML tensor type");
}

std::size_t aligned(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

}  // namespace

float fp16_to_fp32(std::uint16_t value) {
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000U) << 16U;
    std::uint32_t exponent = (value >> 10U) & 0x1FU;
    std::uint32_t mantissa = value & 0x03FFU;
    std::uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x0400U) == 0) { mantissa <<= 1U; --exponent; }
            mantissa &= 0x03FFU;
            bits = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7F800000U | (mantissa << 13U);
    } else {
        bits = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
    }
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::size_t TensorView::element_count() const {
    std::size_t count = 1;
    for (const auto dimension : dimensions) count *= static_cast<std::size_t>(dimension);
    return count;
}

float TensorView::value(std::size_t index) const {
    if (index >= element_count()) throw std::out_of_range("Tensor index outside " + name);
    if (type == GGMLType::F32) {
        float result;
        std::memcpy(&result, data + index * sizeof(float), sizeof(float));
        return result;
    }
    if (type == GGMLType::F16) {
        std::uint16_t half;
        std::memcpy(&half, data + index * sizeof(std::uint16_t), sizeof(half));
        return fp16_to_fp32(half);
    }
    if (type == GGMLType::BF16) {
        std::uint16_t high;
        std::memcpy(&high, data + index * sizeof(std::uint16_t), sizeof(high));
        const std::uint32_t bits = static_cast<std::uint32_t>(high) << 16U;
        float result;
        std::memcpy(&result, &bits, sizeof(result));
        return result;
    }
    if (type == GGMLType::I8) return static_cast<float>(reinterpret_cast<const std::int8_t*>(data)[index]);
    throw std::runtime_error("Unsupported value conversion for " + name);
}

const std::int8_t* TensorView::i8_data() const {
    if (type != GGMLType::I8) throw std::runtime_error(name + " is not I8");
    return reinterpret_cast<const std::int8_t*>(data);
}

GGUFFile::GGUFFile(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("Unable to open GGUF: " + path);
    const auto size = input.tellg();
    if (size <= 0) throw std::runtime_error("Empty GGUF: " + path);
    storage_.resize(static_cast<std::size_t>(size));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(storage_.data()), size);
    if (!input) throw std::runtime_error("Unable to read GGUF: " + path);

    Reader reader(storage_);
    if (reader.read<std::uint32_t>() != 0x46554747U) throw std::runtime_error("Invalid GGUF magic");
    const auto version = reader.read<std::uint32_t>();
    if (version != 3) throw std::runtime_error("Only GGUF v3 is supported");
    const auto tensor_count = reader.read<std::uint64_t>();
    const auto metadata_count = reader.read<std::uint64_t>();
    std::size_t alignment = 32;

    for (std::uint64_t index = 0; index < metadata_count; ++index) {
        const std::string key = reader.string();
        const auto type = reader.read<std::uint32_t>();
        if (type == 8) {
            strings_[key] = reader.string();
        } else if (type == 4) {
            const auto value = reader.read<std::uint32_t>(); integers_[key] = value;
            if (key == "general.alignment") alignment = value;
        } else if (type == 10) {
            integers_[key] = reader.read<std::uint64_t>();
        } else if (type == 6) {
            numbers_[key] = reader.read<float>();
        } else if (type == 12) {
            numbers_[key] = reader.read<double>();
        } else {
            reader.skip_value(type);
        }
    }

    std::vector<TensorInfo> infos;
    infos.reserve(static_cast<std::size_t>(tensor_count));
    for (std::uint64_t index = 0; index < tensor_count; ++index) {
        TensorInfo info;
        info.name = reader.string();
        const auto dimensions = reader.read<std::uint32_t>();
        info.dimensions.reserve(dimensions);
        for (std::uint32_t dimension = 0; dimension < dimensions; ++dimension) {
            info.dimensions.push_back(reader.read<std::uint64_t>());
        }
        info.type = static_cast<GGMLType>(reader.read<std::uint32_t>());
        info.offset = reader.read<std::uint64_t>();
        infos.push_back(std::move(info));
    }
    const std::size_t data_start = aligned(reader.position(), alignment);
    for (const auto& info : infos) {
        std::size_t count = 1;
        for (const auto dimension : info.dimensions) count *= static_cast<std::size_t>(dimension);
        const std::size_t bytes = count * type_size(info.type);
        const std::size_t offset = data_start + static_cast<std::size_t>(info.offset);
        if (offset > storage_.size() || bytes > storage_.size() - offset) {
            throw std::runtime_error("Tensor outside GGUF: " + info.name);
        }
        tensors_.emplace(info.name, TensorView{info.name, info.dimensions, info.type, storage_.data() + offset, bytes});
    }
}

const TensorView& GGUFFile::tensor(const std::string& name) const {
    const auto found = tensors_.find(name);
    if (found == tensors_.end()) throw std::runtime_error("Missing GGUF tensor: " + name);
    return found->second;
}

std::uint32_t GGUFFile::u32(const std::string& key) const {
    const auto found = integers_.find(key);
    if (found == integers_.end()) throw std::runtime_error("Missing GGUF integer: " + key);
    return static_cast<std::uint32_t>(found->second);
}

float GGUFFile::f32(const std::string& key) const {
    const auto found = numbers_.find(key);
    if (found == numbers_.end()) throw std::runtime_error("Missing GGUF number: " + key);
    return static_cast<float>(found->second);
}

const std::string& GGUFFile::string(const std::string& key) const {
    const auto found = strings_.find(key);
    if (found == strings_.end()) throw std::runtime_error("Missing GGUF string: " + key);
    return found->second;
}

}  // namespace retbit
