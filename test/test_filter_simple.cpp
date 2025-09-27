#include "src/simd_fast.hpp"
#include <iostream>
#include <vector>
#include <cstring>

int main() {
    // Test just a few numbers
    std::vector<uint64_t> numbers = {2, 3, 4, 5};

    // Allocate bitmap
    size_t bitmap_size = (numbers.size() + 7) / 8;
    std::vector<uint8_t> bitmap(bitmap_size, 0);

    // Run filter
    neon_wheel::filter_stream_u64_wheel_bitmap(numbers.data(), bitmap.data(), numbers.size());

    // Check results
    for (size_t i = 0; i < numbers.size(); i++) {
        bool passed = (bitmap[i/8] >> (i%8)) & 1;
        std::cout << numbers[i] << ": " << (passed ? "PASS" : "FAIL") << "\n";
    }

    // Show raw bitmap
    std::cout << "Raw bitmap bytes: ";
    for (auto b : bitmap) {
        std::cout << std::hex << (int)b << " ";
    }
    std::cout << "\n";

    return 0;
}