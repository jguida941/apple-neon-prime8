#include <cstdio>
#include <vector>
#include <random>
#include <chrono>
#include <cstring>
#include "src/simd_fast.hpp"

// Forward declarations
namespace neon_ultra {
  void filter_stream_u64_barrett16_ultra(const uint64_t* __restrict, uint8_t* __restrict, size_t);
}
namespace neon_final {
  void filter_stream_u64_barrett16_final(const uint64_t* __restrict, uint8_t* __restrict, size_t);
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

  std::printf("%-25s: %.3f ms, %.1f Mnums/s, hash=%016llx\n",
              name, ms_per_run, throughput, (unsigned long long)h);
}

int main() {
  std::printf("=== Final Optimizations Benchmark ===\n\n");

  // Test with different patterns
  const size_t N = 10'000'000;

  std::printf("--- Random 32-bit (N=10M) ---\n");
  {
    std::vector<uint64_t> data(N);
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> u32(0, 0xffffffffu);
    for (auto& x : data) x = u32(rng);

    benchmark("Original (236 Mnums/s)", neon_fast::filter_stream_u64_barrett16, data);
    benchmark("Ultra (249 Mnums/s)", neon_ultra::filter_stream_u64_barrett16_ultra, data);
    benchmark("Final (all opts)", neon_final::filter_stream_u64_barrett16_final, data);
  }

  std::printf("\n--- Composite-heavy (all even) ---\n");
  {
    std::vector<uint64_t> data(N);
    for (size_t i = 0; i < N; ++i) data[i] = 2 * (i + 1);

    benchmark("Original", neon_fast::filter_stream_u64_barrett16, data);
    benchmark("Ultra", neon_ultra::filter_stream_u64_barrett16_ultra, data);
    benchmark("Final (early-out)", neon_final::filter_stream_u64_barrett16_final, data);
  }

  std::printf("\n--- Mixed (80%% composite) ---\n");
  {
    std::vector<uint64_t> data(N);
    std::mt19937_64 rng(42);
    for (size_t i = 0; i < N; ++i) {
      if (i % 5 == 0) {
        data[i] = 59 + (i % 100) * 2; // Likely prime
      } else {
        data[i] = 6 * (i + 1); // Divisible by 2 and 3
      }
    }

    benchmark("Original", neon_fast::filter_stream_u64_barrett16, data);
    benchmark("Ultra", neon_ultra::filter_stream_u64_barrett16_ultra, data);
    benchmark("Final (wheel+early)", neon_final::filter_stream_u64_barrett16_final, data);
  }

  std::printf("\n--- Large primes only ---\n");
  {
    std::vector<uint64_t> data(N);
    const uint64_t primes[] = {59, 61, 67, 71, 73, 79, 83, 89, 97, 101,
                               103, 107, 109, 113, 127, 131, 137, 139, 149, 151};
    for (size_t i = 0; i < N; ++i) {
      data[i] = primes[i % 20];
    }

    benchmark("Original", neon_fast::filter_stream_u64_barrett16, data);
    benchmark("Ultra", neon_ultra::filter_stream_u64_barrett16_ultra, data);
    benchmark("Final (worst case)", neon_final::filter_stream_u64_barrett16_final, data);
  }

  std::printf("\n--- Small dataset (N=10K) ---\n");
  {
    const size_t small_N = 10000;
    std::vector<uint64_t> data(small_N);
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> u32(0, 0xffffffffu);
    for (auto& x : data) x = u32(rng);

    benchmark("Original", neon_fast::filter_stream_u64_barrett16, data, 100, 1000);
    benchmark("Ultra", neon_ultra::filter_stream_u64_barrett16_ultra, data, 100, 1000);
    benchmark("Final", neon_final::filter_stream_u64_barrett16_final, data, 100, 1000);
  }

  std::printf("\n=== Summary ===\n");
  std::printf("Original: ~236 Mnums/s baseline\n");
  std::printf("Ultra: ~249 Mnums/s (16-wide processing)\n");
  std::printf("Final: Target 250+ Mnums/s with early-out optimizations\n");

  return 0;
}