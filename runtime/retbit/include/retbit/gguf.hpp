#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace retbit {

enum class GGMLType : std::uint32_t {
    F32 = 0,
    F16 = 1,
    I8 = 24,
    BF16 = 30,
};

struct TensorView {
    std::string name;
    std::vector<std::uint64_t> dimensions;
    GGMLType type;
    const std::uint8_t* data;
    std::size_t bytes;

    std::size_t element_count() const;
    float value(std::size_t index) const;
    const std::int8_t* i8_data() const;
};

class GGUFFile {
public:
    explicit GGUFFile(const std::string& path);

    const TensorView& tensor(const std::string& name) const;
    std::uint32_t u32(const std::string& key) const;
    float f32(const std::string& key) const;
    const std::string& string(const std::string& key) const;
    std::size_t tensor_count() const noexcept { return tensors_.size(); }

private:
    std::vector<std::uint8_t> storage_;
    std::unordered_map<std::string, TensorView> tensors_;
    std::unordered_map<std::string, std::string> strings_;
    std::unordered_map<std::string, std::uint64_t> integers_;
    std::unordered_map<std::string, double> numbers_;
};

float fp16_to_fp32(std::uint16_t value);

}  // namespace retbit
