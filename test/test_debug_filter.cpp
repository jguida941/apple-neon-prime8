#include "src/simd_fast.hpp"
#include <iostream>
#include <arm_neon.h>

// Debug version of filter16 that prints intermediate values
uint16_t debug_filter16(const uint64_t* numbers) {
    // Load 16 numbers
    uint32_t n[16];
    for (int i = 0; i < 16; i++) {
        n[i] = numbers[i];
        std::cout << "n[" << i << "] = " << n[i] << "\n";
    }

    // Check each one manually
    uint16_t result = 0;
    for (int i = 0; i < 16; i++) {
        bool pass = false;

        // Special cases
        if (n[i] == 2 || n[i] == 3 || n[i] == 5) {
            pass = true;
            std::cout << "  " << n[i] << " is special prime\n";
        } else {
            // Check wheel-30
            uint32_t r30 = n[i] % 30;
            bool wheel_pass = (r30 == 1 || r30 == 7 || r30 == 11 || r30 == 13 ||
                              r30 == 17 || r30 == 19 || r30 == 23 || r30 == 29);

            if (!wheel_pass) {
                std::cout << "  " << n[i] << " fails wheel (r=" << r30 << ")\n";
                continue;
            }

            // Check small primes
            pass = true;
            uint32_t primes[] = {7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53};
            for (uint32_t p : primes) {
                if (n[i] != p && n[i] % p == 0) {
                    std::cout << "  " << n[i] << " divisible by " << p << "\n";
                    pass = false;
                    break;
                }
            }

            if (pass) {
                std::cout << "  " << n[i] << " passes all tests\n";
            }
        }

        if (pass) {
            result |= (1u << i);
        }
    }

    return result;
}

int main() {
    std::vector<uint64_t> numbers = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};

    uint16_t manual_result = debug_filter16(numbers.data());

    std::cout << "\nManual bitmap: ";
    for (int i = 0; i < 16; i++) {
        std::cout << ((manual_result >> i) & 1);
    }
    std::cout << " (0x" << std::hex << manual_result << ")\n";

    // Now test the real function
    std::vector<uint8_t> bitmap(2, 0);
    neon_wheel::filter_stream_u64_wheel_bitmap(numbers.data(), bitmap.data(), 16);

    std::cout << "SIMD bitmap:   ";
    for (int i = 0; i < 16; i++) {
        std::cout << ((bitmap[i/8] >> (i%8)) & 1);
    }
    uint16_t simd_result = bitmap[0] | (bitmap[1] << 8);
    std::cout << " (0x" << std::hex << simd_result << ")\n";

    return 0;
}