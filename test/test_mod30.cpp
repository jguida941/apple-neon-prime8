#include <iostream>
#include <cstdint>

int main() {
    uint32_t MU30 = 143165576u; // floor(2^32 / 30)

    uint32_t test_values[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53};

    for (uint32_t n : test_values) {
        // Compute n % 30 using Barrett
        uint64_t q = ((uint64_t)n * MU30) >> 32;
        uint32_t r = n - q * 30;
        if (r >= 30) r -= 30;

        // Also compute direct modulo
        uint32_t direct = n % 30;

        std::cout << n << ": Barrett gives " << r << ", direct gives " << direct;

        // Check if it's a coprime residue (1,7,11,13,17,19,23,29)
        bool is_coprime = (r == 1 || r == 7 || r == 11 || r == 13 ||
                          r == 17 || r == 19 || r == 23 || r == 29);

        // Special case for 2, 3, 5
        bool special = (n == 2 || n == 3 || n == 5);

        std::cout << " -> " << (is_coprime || special ? "PASS" : "FAIL") << "\n";
    }

    return 0;
}