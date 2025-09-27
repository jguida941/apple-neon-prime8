#include "src/simd_fast.hpp"
#include <iostream>
#include <vector>
#include <cstring>
#include <iomanip>

// Simple prime checking for validation
bool is_probable_prime(uint64_t n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;

    // Check small primes
    const uint32_t small_primes[] = {2,3,5,7,11,13,17,19};
    const uint32_t ext_primes[] = {23,29,31,37,41,43,47,53};

    for (auto p : small_primes) {
        if (n == p) return true;
        if (n % p == 0) return false;
    }
    for (auto p : ext_primes) {
        if (n == p) return true;
        if (n % p == 0) return false;
    }

    return true;
}

void test_wheel_bitmap() {
    std::cout << "Testing wheel bitmap implementation...\n";

    // Test 1: Small aligned batch (32 numbers)
    {
        std::vector<uint64_t> numbers(32);
        for (size_t i = 0; i < 32; i++) {
            numbers[i] = i + 100;
        }

        std::vector<uint8_t> bitmap((32 + 7) / 8, 0);
        neon_wheel::filter_stream_u64_wheel_bitmap(numbers.data(), bitmap.data(), 32);

        std::cout << "Test 1 (32 aligned): ";
        int correct = 0, total = 0;
        for (size_t i = 0; i < 32; i++) {
            bool simd_result = (bitmap[i/8] >> (i%8)) & 1;
            bool expected = is_probable_prime(numbers[i]);
            if (simd_result == expected) correct++;
            total++;
        }
        std::cout << correct << "/" << total << " correct\n";
    }

    // Test 2: Unaligned batch (17 numbers) - tests tail handling
    {
        std::vector<uint64_t> numbers(17);
        for (size_t i = 0; i < 17; i++) {
            numbers[i] = i * 2 + 1; // odd numbers
        }

        std::vector<uint8_t> bitmap((17 + 7) / 8, 0xFF); // Pre-fill to catch overwrites
        neon_wheel::filter_stream_u64_wheel_bitmap(numbers.data(), bitmap.data(), 17);

        std::cout << "Test 2 (17 unaligned): ";
        int correct = 0, total = 0;
        for (size_t i = 0; i < 17; i++) {
            bool simd_result = (bitmap[i/8] >> (i%8)) & 1;
            bool expected = is_probable_prime(numbers[i]);
            if (simd_result == expected) correct++;
            total++;
        }
        std::cout << correct << "/" << total << " correct\n";
    }

    // Test 3: Very small batch (3 numbers) - tests final tail
    {
        std::vector<uint64_t> numbers = {7, 8, 11};
        std::vector<uint8_t> bitmap(1, 0);
        neon_wheel::filter_stream_u64_wheel_bitmap(numbers.data(), bitmap.data(), 3);

        std::cout << "Test 3 (3 numbers): ";
        bool b0 = (bitmap[0] >> 0) & 1; // 7 is prime
        bool b1 = (bitmap[0] >> 1) & 1; // 8 is not prime
        bool b2 = (bitmap[0] >> 2) & 1; // 11 is prime

        if (b0 && !b1 && b2) {
            std::cout << "PASS\n";
        } else {
            std::cout << "FAIL (got " << b0 << "," << b1 << "," << b2 << ")\n";
        }
    }

    // Test 4: Boundary test (8, 16, 24 numbers)
    for (size_t count : {8, 16, 24}) {
        std::vector<uint64_t> numbers(count);
        for (size_t i = 0; i < count; i++) {
            numbers[i] = i + 1;
        }

        std::vector<uint8_t> bitmap((count + 7) / 8, 0);
        neon_wheel::filter_stream_u64_wheel_bitmap(numbers.data(), bitmap.data(), count);

        std::cout << "Test 4 (boundary " << count << "): ";
        int correct = 0, total = 0;
        for (size_t i = 0; i < count; i++) {
            bool simd_result = (bitmap[i/8] >> (i%8)) & 1;
            bool expected = is_probable_prime(numbers[i]);
            if (simd_result == expected) correct++;
            total++;
        }
        std::cout << correct << "/" << total << " correct\n";
    }
}

void test_ultra_bitmap() {
    std::cout << "\nTesting ultra implementation...\n";

    // Test with various sizes
    for (size_t count : {16, 32, 33, 47}) {
        std::vector<uint64_t> numbers(count);
        for (size_t i = 0; i < count; i++) {
            numbers[i] = i + 50;
        }

        std::vector<uint8_t> out(count, 0xFF);
        neon_ultra::filter_stream_u64_barrett16_ultra(numbers.data(), out.data(), count);

        std::cout << "Test (size " << count << "): ";
        int correct = 0, total = 0;
        for (size_t i = 0; i < count; i++) {
            bool simd_result = out[i];
            bool expected = is_probable_prime(numbers[i]);
            if (simd_result == expected) correct++;
            total++;
        }
        std::cout << correct << "/" << total << " correct\n";
    }
}

int main() {
    std::cout << "=== SIMD Prime Filter Fix Validation ===\n\n";

    test_wheel_bitmap();
    test_ultra_bitmap();

    std::cout << "\n=== Tests Complete ===\n";

    return 0;
}