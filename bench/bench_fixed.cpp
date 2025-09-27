#include "src/simd_fast.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>

using namespace std::chrono;

void benchmark_wheel() {
    std::cout << "Benchmarking Wheel-30 Bitmap Implementation (Fixed)\n";
    std::cout << "===================================================\n";

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> dist32(0, 0xffffffffull);

    for (size_t size : {1024, 8192, 65536}) {
        std::vector<uint64_t> numbers(size);
        for (auto& n : numbers) n = dist32(rng);

        std::vector<uint8_t> bitmap((size + 7) / 8);

        // Warmup
        for (int i = 0; i < 10; i++) {
            neon_wheel::filter_stream_u64_wheel_bitmap(numbers.data(), bitmap.data(), size);
        }

        // Benchmark
        const int iterations = 1000;
        auto start = high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            neon_wheel::filter_stream_u64_wheel_bitmap(numbers.data(), bitmap.data(), size);
        }
        auto end = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(end - start).count();
        double throughput = (double(size * iterations) / duration) * 1e6 / 1e9;

        std::cout << "Size " << std::setw(6) << size << ": "
                  << std::fixed << std::setprecision(2)
                  << throughput << " Gnum/s\n";
    }
}

void benchmark_ultra() {
    std::cout << "\nBenchmarking Ultra Implementation (Fixed)\n";
    std::cout << "==========================================\n";

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> dist32(0, 0xffffffffull);

    for (size_t size : {1024, 8192, 65536}) {
        std::vector<uint64_t> numbers(size);
        for (auto& n : numbers) n = dist32(rng);

        std::vector<uint8_t> out(size);

        // Warmup
        for (int i = 0; i < 10; i++) {
            neon_ultra::filter_stream_u64_barrett16_ultra(numbers.data(), out.data(), size);
        }

        // Benchmark
        const int iterations = 1000;
        auto start = high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            neon_ultra::filter_stream_u64_barrett16_ultra(numbers.data(), out.data(), size);
        }
        auto end = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(end - start).count();
        double throughput = (double(size * iterations) / duration) * 1e6 / 1e9;

        std::cout << "Size " << std::setw(6) << size << ": "
                  << std::fixed << std::setprecision(2)
                  << throughput << " Gnum/s\n";
    }
}

int main() {
    std::cout << "=== Fixed SIMD Prime Filter Performance ===\n\n";

    benchmark_wheel();
    benchmark_ultra();

    std::cout << "\n=== Benchmarks Complete (No Crashes!) ===\n";
    return 0;
}