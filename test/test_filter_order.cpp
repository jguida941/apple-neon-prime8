#include "src/simd_fast.hpp"
#include <iostream>
#include <vector>
#include <cstring>

int main() {
    // Test primes separately
    std::vector<uint64_t> primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53};

    // Allocate bitmap
    size_t bitmap_size = (primes.size() + 7) / 8;
    std::vector<uint8_t> bitmap(bitmap_size, 0);

    // Run filter
    neon_wheel::filter_stream_u64_wheel_bitmap(primes.data(), bitmap.data(), primes.size());

    // Check results
    std::cout << "Testing primes (should all pass = 1):\n";
    for (size_t i = 0; i < primes.size(); i++) {
        bool passed = (bitmap[i/8] >> (i%8)) & 1;
        std::cout << primes[i] << ": " << (passed ? "PASS" : "FAIL") << "\n";
    }

    // Now test composites
    std::vector<uint64_t> composites = {4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25, 26};
    bitmap_size = (composites.size() + 7) / 8;
    bitmap.resize(bitmap_size);
    std::fill(bitmap.begin(), bitmap.end(), 0);

    neon_wheel::filter_stream_u64_wheel_bitmap(composites.data(), bitmap.data(), composites.size());

    std::cout << "\nTesting composites (should all fail = 0):\n";
    for (size_t i = 0; i < composites.size(); i++) {
        bool passed = (bitmap[i/8] >> (i%8)) & 1;
        std::cout << composites[i] << ": " << (passed ? "PASS (wrong)" : "FAIL (correct)") << "\n";
    }

    return 0;
}