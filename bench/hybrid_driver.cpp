#include "simd_fast.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <wheel30|wheel210> <count>\n";
        return 1;
    }

    const std::string mode = argv[1];
    const size_t count = static_cast<size_t>(std::strtoull(argv[2], nullptr, 10));

    std::vector<uint64_t> numbers(count);
    const size_t bytes = count * sizeof(uint64_t);
    if (!std::cin.read(reinterpret_cast<char*>(numbers.data()), bytes)) {
        std::cerr << "Failed to read " << bytes << " bytes from stdin\n";
        return 2;
    }

    std::vector<uint8_t> bitmap((count + 7) / 8);

    if (mode == "wheel30") {
        neon_wheel::filter_stream_u64_wheel_bitmap(numbers.data(), bitmap.data(), count);
    } else if (mode == "wheel210") {
        neon_wheel210_efficient::filter_stream_u64_wheel210_efficient_bitmap(
            numbers.data(), bitmap.data(), count);
    } else {
        std::cerr << "Unknown mode '" << mode << "'\n";
        return 3;
    }

    for (size_t i = 0; i < count; ++i) {
        if (bitmap[i >> 3] & (1u << (i & 7))) {
            std::cout << numbers[i] << '\n';
        }
    }

    return 0;
}
