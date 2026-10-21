#include "retbit/model.hpp"

#include <cstdlib>
#include <chrono>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::uint32_t> parse_tokens(const std::string& value) {
    std::vector<std::uint32_t> tokens;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) if (!item.empty()) tokens.push_back(static_cast<std::uint32_t>(std::stoul(item)));
    return tokens;
}

std::vector<std::string> split(const std::string& value, char delimiter) {
    std::vector<std::string> parts;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, delimiter)) parts.push_back(item);
    return parts;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string model_path;
        std::string token_text;
        bool validate_only = false;
        bool server_mode = false;
        std::size_t threads = 0;
        retbit::GenerationOptions options;
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            auto next = [&]() -> std::string {
                if (++index >= argc) throw std::invalid_argument("Missing value for " + argument);
                return argv[index];
            };
            if (argument == "--model") model_path = next();
            else if (argument == "--tokens") token_text = next();
            else if (argument == "--max-new-tokens") options.max_new_tokens = std::stoul(next());
            else if (argument == "--temperature") options.temperature = std::stof(next());
            else if (argument == "--top-p") options.top_p = std::stof(next());
            else if (argument == "--top-k") options.top_k = std::stoul(next());
            else if (argument == "--repetition-penalty") options.repetition_penalty = std::stof(next());
            else if (argument == "--seed") options.seed = static_cast<std::uint32_t>(std::stoul(next()));
            else if (argument == "--eos-token-id") options.eos_token_id = static_cast<std::uint32_t>(std::stoul(next()));
            else if (argument == "--validate") validate_only = true;
            else if (argument == "--server") server_mode = true;
            else if (argument == "--threads") threads = std::stoul(next());
            else throw std::invalid_argument("Unknown argument: " + argument);
        }
        if (model_path.empty()) throw std::invalid_argument("--model is required");
        retbit::RetBitModel model(model_path, threads);
        if (validate_only) {
            std::cout << "RETBit GGUF OK layers=" << model.layer_count()
                      << " embedding=" << model.embedding_length()
                      << " vocab=" << model.vocab_size() << '\n';
            return 0;
        }
        if (server_mode) {
            std::cout << "READY:" << model.layer_count() << ':' << model.embedding_length() << ':' << model.vocab_size() << '\n' << std::flush;
            std::string request;
            while (std::getline(std::cin, request)) {
                if (request == "QUIT") break;
                try {
                    const auto fields = split(request, '|');
                    if (fields.size() != 8) throw std::invalid_argument("server request requires 8 fields");
                    const auto prompt = parse_tokens(fields[0]);
                    retbit::GenerationOptions request_options;
                    request_options.max_new_tokens = std::stoul(fields[1]);
                    request_options.temperature = std::stof(fields[2]);
                    request_options.top_p = std::stof(fields[3]);
                    request_options.top_k = std::stoul(fields[4]);
                    request_options.repetition_penalty = std::stof(fields[5]);
                    request_options.seed = static_cast<std::uint32_t>(std::stoul(fields[6]));
                    request_options.eos_token_id = static_cast<std::uint32_t>(std::stoul(fields[7]));
                    const auto started = std::chrono::steady_clock::now();
                    const auto generated = model.generate(prompt, request_options, [](std::uint32_t token) {
                        std::cout << "TOKEN:" << token << '\n' << std::flush;
                    });
                    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started).count();
                    std::cout << "DONE:" << generated.size() << ':' << elapsed << '\n' << std::flush;
                } catch (const std::exception& error) {
                    std::cout << "ERROR:" << error.what() << '\n' << std::flush;
                }
            }
            return 0;
        }
        const auto prompt = parse_tokens(token_text);
        const auto generated = model.generate(prompt, options);
        std::cout << "TOKENS:";
        for (const auto token : generated) std::cout << ' ' << token;
        std::cout << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "retbit_cli: " << error.what() << '\n';
        return 1;
    }
}
