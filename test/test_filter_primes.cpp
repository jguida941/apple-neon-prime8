#include "src/simd_fast.hpp"
#include <iostream>
#include <vector>
#include <cstring>

int main() {
    // Test known small primes
    std::vector<uint64_t> test_primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53};

    // Also test some composites
    std::vector<uint64_t> test_composites = {4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25, 26};

    // Combine into one test
    std::vector<uint64_t> all_numbers;
    all_numbers.insert(all_numbers.end(), test_primes.begin(), test_primes.end());
    all_numbers.insert(all_numbers.end(), test_composites.begin(), test_composites.end());

    // Allocate bitmap
    size_t bitmap_size = (all_numbers.size() + 7) / 8;
    std::vector<uint8_t> bitmap(bitmap_size, 0);

    // Run filter
    neon_wheel::filter_stream_u64_wheel_bitmap(all_numbers.data(), bitmap.data(), all_numbers.size());

    // Check results
    std::cout << "Testing known primes (should all pass = 1):\n";
    for (size_t i = 0; i < test_primes.size(); i++) {
        bool passed = (bitmap[i/8] >> (i%8)) & 1;
        std::cout << test_primes[i] << ": " << (passed ? "PASS" : "FAIL") << "\n";
        if (!passed && test_primes[i] > 53) {  // Our filter only checks up to 53
            std::cout << "  (Expected - beyond our prime list)\n";
        }
    }

    std::cout << "\nTesting known composites (should all fail = 0):\n";
    size_t offset = test_primes.size();
    for (size_t i = 0; i < test_composites.size(); i++) {
        size_t idx = offset + i;
        bool passed = (bitmap[idx/8] >> (idx%8)) & 1;
        std::cout << test_composites[i] << ": " << (passed ? "PASS (false positive)" : "FAIL (correct)") << "\n";
    }

    return 0;
}