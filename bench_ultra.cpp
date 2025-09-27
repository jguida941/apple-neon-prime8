#include <cstdio>
#include <vector>
#include <random>
#include <chrono>
#include <cstring>
#include "src/simd_fast.hpp"

// Forward declarations for ultra versions
namespace neon_ultra {
  void filter_stream_u64_barrett16_ultra(const uint64_t* __restrict numbers,
                                         uint8_t*       __restrict out,
                                         size_t count);
}

using bench_clock = std::chrono::high_resolution_clock;

static void benchmark(const char* name,
                     void(*fn)(const uint64_t*, uint8_t*, size_t),
                     const std::vector<uint64_t>& data,
                     int warmup = 10, int runs = 100) {
  std::vector<uint8_t> out(data.size());

  // Warmup
  for (int i = 0; i < warmup; ++i) {
    fn(data.data(), out.data(), data.size());
  }

  // Timed runs
  auto t0 = bench_clock::now();
  for (int i = 0; i < runs; ++i) {
    fn(data.data(), out.data(), data.size());
  }
  auto t1 = bench_clock::now();

  double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  double ms_per_run = total_ms / runs;
  double throughput = (data.size() / 1e6) / (ms_per_run / 1000.0);

  // Hash for validation
  uint64_t h = 0xcbf29ce484222325ULL;
  for (auto v : out) { h ^= v; h *= 0x100000001b3ULL; }

  std::printf("%-20s: %.3f ms/run, %.1f Mnums/s, hash=%016llx\n",
              name, ms_per_run, throughput, (unsigned long long)h);
}

int main() {
  std::printf("=== Ultra-Optimized SIMD Benchmark ===\n\n");

  // Test different sizes to see where each strategy wins
  const size_t sizes[] = {1000, 10000, 100000, 1000000, 10000000};

  for (size_t N : sizes) {
    std::printf("--- N = %zu ---\n", N);

    // Generate test data
    std::vector<uint64_t> data(N);
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> u32(0, 0xffffffffu);
    for (auto& x : data) x = u32(rng);

    // Ensure alignment for best performance
    std::vector<uint64_t> aligned_data(N + 8);
    uint64_t* aligned_ptr = (uint64_t*)(((uintptr_t)aligned_data.data() + 63) & ~63);
    std::memcpy(aligned_ptr, data.data(), N * sizeof(uint64_t));

    // Run benchmarks
    benchmark("Original", neon_fast::filter_stream_u64_barrett16, data);
    benchmark("Ultra (16-wide)", neon_ultra::filter_stream_u64_barrett16_ultra, data);

    // Also test with aligned data
    std::vector<uint64_t> aligned_vec(aligned_ptr, aligned_ptr + N);
    benchmark("Original (aligned)", neon_fast::filter_stream_u64_barrett16, aligned_vec);
    benchmark("Ultra (aligned)", neon_ultra::filter_stream_u64_barrett16_ultra, aligned_vec);

    std::printf("\n");
  }

  // Test specific patterns
  std::printf("--- Pattern Tests (N=1M) ---\n");
  const size_t N = 1000000;

  // All 32-bit
  {
    std::vector<uint64_t> data(N);
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> u32(0, 0xffffffffu);
    for (auto& x : data) x = u32(rng);
    std::printf("All 32-bit:\n");
    benchmark("  Original", neon_fast::filter_stream_u64_barrett16, data);
    benchmark("  Ultra", neon_ultra::filter_stream_u64_barrett16_ultra, data);
  }

  // Sequential (cache-friendly)
  {
    std::vector<uint64_t> data(N);
    for (size_t i = 0; i < N; ++i) data[i] = i + 1;
    std::printf("Sequential:\n");
    benchmark("  Original", neon_fast::filter_stream_u64_barrett16, data);
    benchmark("  Ultra", neon_ultra::filter_stream_u64_barrett16_ultra, data);
  }

  // Powers of 2 (branch predictor friendly)
  {
    std::vector<uint64_t> data(N);
    for (size_t i = 0; i < N; ++i) data[i] = 1ull << (i % 30);
    std::printf("Powers of 2:\n");
    benchmark("  Original", neon_fast::filter_stream_u64_barrett16, data);
    benchmark("  Ultra", neon_ultra::filter_stream_u64_barrett16_ultra, data);
  }

  return 0;
}