#include "../src/simd_fast.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>

using namespace std::chrono;

// Naive scalar implementation for baseline
void filter_scalar_naive(const uint64_t* numbers, uint8_t* out, size_t count) {
    for (size_t i = 0; i < count; i++) {
        uint64_t n = numbers[i];
        if (n <= 1) {
            out[i] = 0;
            continue;
        }
        if (n <= 3) {
            out[i] = 1;
            continue;
        }
        if (n % 2 == 0 || n % 3 == 0) {
            out[i] = 0;
            continue;
        }

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

// Scalar with Barrett reduction (no SIMD)
void filter_scalar_barrett(const uint64_t* numbers, uint8_t* out, size_t count) {
    const uint32_t primes[] = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53};
    const uint32_t mu[] = {
        2147483649u, 2863311531u, 3435973837u, 3067833783u,
        3123612579u, 2987803337u, 4042322161u, 3407155659u,
        3303820997u, 2957159389u, 2756048309u, 2480700261u,
        3665941669u, 3193865225u, 2938661835u, 2695319797u
    };

    for (size_t i = 0; i < count; i++) {
        uint64_t n = numbers[i];
        if (n > 0xffffffffu) {
            out[i] = 0;
            continue;
        }

        uint32_t n32 = (uint32_t)n;
        bool survive = true;

        for (int j = 0; j < 16; j++) {
            if (n32 == primes[j]) continue;
            uint64_t q = (uint64_t)n32 * mu[j] >> 32;
            uint32_t r = n32 - q * primes[j];
            if (r >= primes[j]) r -= primes[j];
            if (r == 0) {
                survive = false;
                break;
            }
        }
        out[i] = survive ? 1 : 0;
    }
}

// Simple Sieve of Eratosthenes for comparison
std::vector<bool> sieve_of_eratosthenes(uint64_t max_val) {
    std::vector<bool> is_prime(max_val + 1, true);
    is_prime[0] = is_prime[1] = false;

    for (uint64_t i = 2; i * i <= max_val; i++) {
        if (is_prime[i]) {
            for (uint64_t j = i * i; j <= max_val; j += i) {
                is_prime[j] = false;
            }
        }
    }
    return is_prime;
}

void benchmark_method(const std::string& name,
                     void (*filter_func)(const uint64_t*, uint8_t*, size_t),
                     const std::vector<uint64_t>& numbers,
                     std::vector<uint8_t>& output,
                     int iterations = 100) {

    // Warmup
    for (int i = 0; i < 5; i++) {
        filter_func(numbers.data(), output.data(), numbers.size());
    }

    // Benchmark
    auto start = high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        filter_func(numbers.data(), output.data(), numbers.size());
    }
    auto end = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(end - start).count();
    double throughput = (double(numbers.size() * iterations) / duration) * 1e6 / 1e9;
    double ns_per_num = (double(duration) * 1000.0) / (numbers.size() * iterations);

    std::cout << std::left << std::setw(25) << name
              << std::right << std::fixed << std::setprecision(3)
              << std::setw(8) << throughput << " Gnum/s"
              << std::setw(10) << ns_per_num << " ns/num"
              << std::endl;
}

void benchmark_bitmap(const std::string& name,
                     void (*filter_func)(const uint64_t*, uint8_t*, size_t),
                     const std::vector<uint64_t>& numbers,
                     std::vector<uint8_t>& bitmap,
                     int iterations = 100) {

    // Warmup
    for (int i = 0; i < 5; i++) {
        filter_func(numbers.data(), bitmap.data(), numbers.size());
    }

    // Benchmark
    auto start = high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        filter_func(numbers.data(), bitmap.data(), numbers.size());
    }
    auto end = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(end - start).count();
    double throughput = (double(numbers.size() * iterations) / duration) * 1e6 / 1e9;
    double ns_per_num = (double(duration) * 1000.0) / (numbers.size() * iterations);

    std::cout << std::left << std::setw(25) << name
              << std::right << std::fixed << std::setprecision(3)
              << std::setw(8) << throughput << " Gnum/s"
              << std::setw(10) << ns_per_num << " ns/num"
              << std::endl;
}

int main() {
    std::cout << "\n================================================\n";
    std::cout << "   SIMD Prime Filter - Performance Comparison\n";
    std::cout << "   Apple Silicon (M-series) - Single Core\n";
    std::cout << "================================================\n\n";

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> dist32(1, 0xffffffffu);

    // Test different input sizes
    for (size_t size : {1024, 16384, 65536}) {
        std::cout << "Dataset Size: " << size << " numbers\n";
        std::cout << "------------------------------------------------\n";

        // Generate test data
        std::vector<uint64_t> numbers(size);
        for (auto& n : numbers) n = dist32(rng);

        // Output buffers
        std::vector<uint8_t> output(size);
        std::vector<uint8_t> bitmap((size + 7) / 8);

        // Run benchmarks
        std::cout << "Method                   Throughput    Latency\n";
        std::cout << "------------------------------------------------\n";

        // Baseline methods
        benchmark_method("Scalar (naive modulo)", filter_scalar_naive, numbers, output, 10);
        benchmark_method("Scalar (Barrett)", filter_scalar_barrett, numbers, output, 100);

        // SIMD methods - byte output
        benchmark_method("SIMD Ultra (16-wide)",
                        neon_ultra::filter_stream_u64_barrett16_ultra,
                        numbers, output, 100);

        // SIMD methods - bitmap output (fastest)
        benchmark_bitmap("SIMD Wheel-30 + Bitmap",
                        neon_wheel::filter_stream_u64_wheel_bitmap,
                        numbers, bitmap, 1000);

        // Calculate speedups
        std::cout << "\n";

        // Quick speedup calculation (re-run for accurate measurement)
        auto start_naive = high_resolution_clock::now();
        filter_scalar_naive(numbers.data(), output.data(), numbers.size());
        auto end_naive = high_resolution_clock::now();
        double naive_time = duration_cast<microseconds>(end_naive - start_naive).count();

        auto start_simd = high_resolution_clock::now();
        neon_wheel::filter_stream_u64_wheel_bitmap(numbers.data(), bitmap.data(), numbers.size());
        auto end_simd = high_resolution_clock::now();
        double simd_time = duration_cast<microseconds>(end_simd - start_simd).count();

        double speedup = naive_time / simd_time;

        std::cout << "Speedup vs Scalar: " << std::fixed << std::setprecision(1)
                  << speedup << "x faster\n";
        std::cout << "\n";
    }

    std::cout << "================================================\n";
    std::cout << "Key Insights:\n";
    std::cout << "------------------------------------------------\n";
    std::cout << "- Wheel-30 prefilter eliminates 73% of work\n";
    std::cout << "- Barrett reduction avoids expensive division\n";
    std::cout << "- SIMD processes 16 numbers in parallel\n";
    std::cout << "- Bitmap output uses 8x less memory\n";
    std::cout << "================================================\n\n";

    return 0;
}