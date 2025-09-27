#include "../src/simd_fast.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>

using namespace std::chrono;

// Naive scalar for baseline
void filter_scalar_naive(const uint64_t* numbers, uint8_t* out, size_t count) {
    for (size_t i = 0; i < count; i++) {
        uint64_t n = numbers[i];
        if (n <= 1) { out[i] = 0; continue; }
        if (n <= 3) { out[i] = 1; continue; }
        if (n % 2 == 0 || n % 3 == 0) { out[i] = 0; continue; }

        bool is_prime = true;
        uint64_t limit = std::sqrt(n);
        for (uint64_t d = 5; d <= limit; d += 6) {
            if (n % d == 0 || n % (d + 2) == 0) {
                is_prime = false;
                break;
            }
        }
        out[i] = is_prime ? 1 : 0;
    }
}

template<typename Func>
double benchmark_throughput(Func func, const std::vector<uint64_t>& numbers,
                           std::vector<uint8_t>& output, int iterations) {
    // Warmup
    for (int i = 0; i < 5; i++) {
        func(numbers.data(), output.data(), numbers.size());
    }

    // Benchmark
    auto start = high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        func(numbers.data(), output.data(), numbers.size());
    }
    auto end = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(end - start).count();
    return (double(numbers.size() * iterations) / duration) * 1e6 / 1e9;
}

int main() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "        APPLE SILICON NEON PRIME FILTER - FULL COMPARISON\n";
    std::cout << "                  M-series Single Core Performance\n";
    std::cout << std::string(80, '=') << "\n\n";

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> dist32(1, 0xffffffffu);

    // Test multiple sizes
    for (size_t size : {1024, 16384, 65536}) {
        std::cout << "DATASET: " << size << " numbers (32-bit random)\n";
        std::cout << std::string(60, '-') << "\n";

        // Generate test data
        std::vector<uint64_t> numbers(size);
        for (auto& n : numbers) n = dist32(rng);

        // Buffers
        std::vector<uint8_t> output(size);
        std::vector<uint8_t> bitmap((size + 7) / 8);

        // Results table
        std::cout << std::left << std::setw(30) << "Method"
                  << std::right << std::setw(12) << "Throughput"
                  << std::setw(10) << "Speedup"
                  << std::setw(12) << "Latency\n";
        std::cout << std::string(64, '-') << "\n";

        // Benchmark each method
        double baseline = benchmark_throughput(filter_scalar_naive, numbers, output, 10);
        std::cout << std::left << std::setw(30) << "Scalar (naive)"
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(9) << baseline << " Gn/s"
                  << std::setw(8) << "1.0x"
                  << std::setw(9) << 1000.0/baseline << " ns\n";

        double ultra = benchmark_throughput(neon_ultra::filter_stream_u64_barrett16_ultra,
                                           numbers, output, 100);
        std::cout << std::left << std::setw(30) << "SIMD Ultra Barrett-16"
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(9) << ultra << " Gn/s"
                  << std::setw(8) << std::setprecision(1) << ultra/baseline << "x"
                  << std::setw(9) << std::setprecision(2) << 1000.0/ultra << " ns\n";

        double wheel30 = benchmark_throughput(neon_wheel::filter_stream_u64_wheel_bitmap,
                                             numbers, bitmap, 1000);
        std::cout << std::left << std::setw(30) << "SIMD Wheel-30 (73% elim)"
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(9) << wheel30 << " Gn/s"
                  << std::setw(8) << std::setprecision(1) << wheel30/baseline << "x"
                  << std::setw(9) << std::setprecision(2) << 1000.0/wheel30 << " ns\n";

        double wheel210 = benchmark_throughput(neon_wheel210::filter_stream_u64_wheel210_bitmap,
                                               numbers, bitmap, 1000);
        std::cout << std::left << std::setw(30) << "SIMD Wheel-210 (77% elim)"
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(9) << wheel210 << " Gn/s"
                  << std::setw(8) << std::setprecision(1) << wheel210/baseline << "x"
                  << std::setw(9) << std::setprecision(2) << 1000.0/wheel210 << " ns\n";

        std::cout << "\n";

        // Performance gain summary
        std::cout << "PERFORMANCE GAINS:\n";
        std::cout << "  Wheel-210 vs Wheel-30: +"
                  << std::fixed << std::setprecision(1)
                  << (wheel210 - wheel30) / wheel30 * 100 << "%\n";
        std::cout << "  Wheel-210 vs Ultra:   +"
                  << (wheel210 - ultra) / ultra * 100 << "%\n";
        std::cout << "  Wheel-210 vs Scalar:  "
                  << std::setprecision(0) << wheel210/baseline << "x faster\n";

        std::cout << "\n";
    }

    // Theoretical analysis
    std::cout << std::string(80, '=') << "\n";
    std::cout << "THEORETICAL ANALYSIS\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << "Wheel-30  (2×3×5):   Eliminates 22/30  = 73.3% before Barrett\n";
    std::cout << "Wheel-210 (2×3×5×7): Eliminates 162/210 = 77.1% before Barrett\n";
    std::cout << "Improvement: 77.1% - 73.3% = 3.8% more elimination\n";
    std::cout << "Work reduction: (1-0.771)/(1-0.733) = 85.8% of Wheel-30's work\n";
    std::cout << std::string(80, '=') << "\n\n";

    return 0;
}