#include "src/simd_fast.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace std::chrono;

int main() {
    std::cout << "\n=== Testing Efficient Wheel-210 (Wheel-30 + mod 7) ===\n\n";

    // Test data
    std::vector<uint64_t> numbers(65536);
    for (size_t i = 0; i < numbers.size(); i++) {
        numbers[i] = i + 1000000;
    }

    std::vector<uint8_t> bitmap((numbers.size() + 7) / 8);

    // Warmup
    for (int i = 0; i < 10; i++) {
        neon_wheel210_efficient::filter_stream_u64_wheel210_efficient_bitmap(
            numbers.data(), bitmap.data(), numbers.size());
    }

    // Benchmark
    const int iterations = 1000;
    auto start = high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        neon_wheel210_efficient::filter_stream_u64_wheel210_efficient_bitmap(
            numbers.data(), bitmap.data(), numbers.size());
    }
    auto end = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(end - start).count();
    double throughput = (double(numbers.size() * iterations) / duration) * 1e6 / 1e9;

    std::cout << "Efficient Wheel-210 Performance:\n";
    std::cout << "  Throughput: " << std::fixed << std::setprecision(3)
              << throughput << " Gnum/s\n";
    std::cout << "  Latency: " << std::setprecision(2)
              << 1000.0/throughput << " ns/number\n";

    // Compare with Wheel-30
    auto start2 = high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        neon_wheel::filter_stream_u64_wheel_bitmap(
            numbers.data(), bitmap.data(), numbers.size());
    }
    auto end2 = high_resolution_clock::now();

    auto duration2 = duration_cast<microseconds>(end2 - start2).count();
    double throughput2 = (double(numbers.size() * iterations) / duration2) * 1e6 / 1e9;

    std::cout << "\nWheel-30 Performance:\n";
    std::cout << "  Throughput: " << std::fixed << std::setprecision(3)
              << throughput2 << " Gnum/s\n";
    std::cout << "  Latency: " << std::setprecision(2)
              << 1000.0/throughput2 << " ns/number\n";

    std::cout << "\nImprovement: " << std::setprecision(1)
              << ((throughput - throughput2) / throughput2 * 100)
              << "% (expected: ~5-10% from 3.8% more elimination)\n";

    return 0;
}