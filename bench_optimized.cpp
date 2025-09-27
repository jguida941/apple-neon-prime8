#include "src/simd_fast.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>

// Forward declare the optimized version
namespace neon_optimized {
    void filter_stream_u64_wheel_optimized(const uint64_t* __restrict numbers,
                                          uint8_t*       __restrict bitmap,
                                          size_t count);
}

using namespace std::chrono;

int main() {
    std::cout << "=== Comparing Fixed vs Optimized Implementations ===\n\n";

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> dist32(0, 0xffffffffull);

    for (size_t size : {1024, 8192, 65536}) {
        std::vector<uint64_t> numbers(size);
        for (auto& n : numbers) n = dist32(rng);

        std::vector<uint8_t> bitmap((size + 7) / 8);

        // Benchmark FIXED version
        {
            // Warmup
            for (int i = 0; i < 10; i++) {
                neon_wheel::filter_stream_u64_wheel_bitmap(numbers.data(), bitmap.data(), size);
            }

            const int iterations = 1000;
            auto start = high_resolution_clock::now();
            for (int i = 0; i < iterations; i++) {
                neon_wheel::filter_stream_u64_wheel_bitmap(numbers.data(), bitmap.data(), size);
            }
            auto end = high_resolution_clock::now();

            auto duration = duration_cast<microseconds>(end - start).count();
            double throughput = (double(size * iterations) / duration) * 1e6 / 1e9;

            std::cout << "FIXED   - Size " << std::setw(6) << size << ": "
                      << std::fixed << std::setprecision(3)
                      << throughput << " Gnum/s\n";
        }

        // Benchmark OPTIMIZED version
        {
            // Warmup
            for (int i = 0; i < 10; i++) {
                neon_optimized::filter_stream_u64_wheel_optimized(numbers.data(), bitmap.data(), size);
            }

            const int iterations = 1000;
            auto start = high_resolution_clock::now();
            for (int i = 0; i < iterations; i++) {
                neon_optimized::filter_stream_u64_wheel_optimized(numbers.data(), bitmap.data(), size);
            }
            auto end = high_resolution_clock::now();

            auto duration = duration_cast<microseconds>(end - start).count();
            double throughput = (double(size * iterations) / duration) * 1e6 / 1e9;

            std::cout << "OPTIMIZED - Size " << std::setw(6) << size << ": "
                      << std::fixed << std::setprecision(3)
                      << throughput << " Gnum/s\n";
        }
        std::cout << "\n";
    }

    return 0;
}