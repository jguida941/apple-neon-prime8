#include <cstdio>
#include <vector>
#include <random>
#include <chrono>
#include "src/simd_fast.hpp"

namespace neon_ultra {
  void filter_stream_u64_barrett16_ultra(const uint64_t* __restrict, uint8_t* __restrict, size_t);
}

using bench_clock = std::chrono::high_resolution_clock;

static void benchmark(const char* name,
                     void(*fn)(const uint64_t*, uint8_t*, size_t),
                     const std::vector<uint64_t>& data) {
  std::vector<uint8_t> out(data.size());

  // Warmup
  for (int i = 0; i < 10; ++i) {
    fn(data.data(), out.data(), data.size());
  }

  // Timed runs
  const int runs = 100;
  auto t0 = bench_clock::now();
  for (int i = 0; i < runs; ++i) {
    fn(data.data(), out.data(), data.size());
  }
  auto t1 = bench_clock::now();

  double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  double ms_per_run = total_ms / runs;
  double throughput = (data.size() / 1e6) / (ms_per_run / 1000.0);

  std::printf("%-20s: %.3f ms, %.1f Mnums/s\n", name, ms_per_run, throughput);
}

int main() {
  std::printf("=== Working Versions Performance ===\n\n");

  const size_t N = 10'000'000;
  std::vector<uint64_t> data(N);
  std::mt19937_64 rng(42);
  std::uniform_int_distribution<uint64_t> u32(0, 0xffffffffu);
  for (auto& x : data) x = u32(rng);

  std::printf("Random 32-bit (N=10M):\n");
  benchmark("Original", neon_fast::filter_stream_u64_barrett16, data);
  benchmark("Ultra (16-wide)", neon_ultra::filter_stream_u64_barrett16_ultra, data);

  std::printf("\nComposite-heavy (all even):\n");
  for (size_t i = 0; i < N; ++i) data[i] = 2 * (i + 1);
  benchmark("Original", neon_fast::filter_stream_u64_barrett16, data);
  benchmark("Ultra (16-wide)", neon_ultra::filter_stream_u64_barrett16_ultra, data);

  std::printf("\nLarge primes:\n");
  const uint64_t primes[] = {59, 61, 67, 71, 73, 79, 83, 89, 97, 101};
  for (size_t i = 0; i < N; ++i) data[i] = primes[i % 10];
  benchmark("Original", neon_fast::filter_stream_u64_barrett16, data);
  benchmark("Ultra (16-wide)", neon_ultra::filter_stream_u64_barrett16_ultra, data);

  return 0;
}