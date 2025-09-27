#include "src/simd_fast.hpp"
#include <iostream>
#include <vector>
#include <cstring>
#include <bitset>

int main() {
    // Test exactly 16 numbers (fits one SIMD block)
    std::vector<uint64_t> numbers(16);
    for (int i = 0; i < 16; i++) {
        numbers[i] = i + 2; // 2, 3, 4, 5, ..., 17
    }

    // Allocate bitmap
    size_t bitmap_size = (numbers.size() + 7) / 8;
    std::vector<uint8_t> bitmap(bitmap_size, 0);

    // Run filter
    neon_wheel::filter_stream_u64_wheel_bitmap(numbers.data(), bitmap.data(), numbers.size());

    // Check results
    std::cout << "Testing numbers 2-17:\n";
    for (size_t i = 0; i < numbers.size(); i++) {
        bool passed = (bitmap[i/8] >> (i%8)) & 1;

        // Is it actually prime?
        uint32_t n = numbers[i];
        bool actually_prime = false;
        if (n == 2 || n == 3 || n == 5 || n == 7 || n == 11 || n == 13 || n == 17) {
            actually_prime = true;
        }

        std::cout << n << ": " << (passed ? "PASS" : "FAIL")
                  << " (should be " << (actually_prime ? "PASS" : "FAIL") << ")"
                  << (passed == actually_prime ? " ✓" : " ✗") << "\n";
    }

    // Show bitmap as binary
    std::cout << "\nBitmap (binary): ";
    for (int byte_idx = 0; byte_idx < bitmap_size; byte_idx++) {
        std::bitset<8> bits(bitmap[byte_idx]);
        std::cout << bits << " ";
    }
    std::cout << "\n";

    return 0;
}