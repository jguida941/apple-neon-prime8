#include <cstdio>
#include <vector>
#include <random>
#include <chrono>
#include "src/simd_fast.hpp"

namespace neon_ultra {
  void filter_stream_u64_barrett16_ultra(const uint64_t* __restrict, uint8_t* __restrict, size_t);
}
namespace neon_wheel {
  void filter_stream_u64_wheel(const uint64_t* __restrict, uint8_t* __restrict, size_t);
  void filter_stream_u64_wheel_bitmap(const uint64_t* __restrict, uint8_t* __restrict, size_t);
}

using bench_clock = std::chrono::high_resolution_clock;

static void benchmark(const char* name,
                     void(*fn)(const uint64_t*, uint8_t*, size_t),
                     const std::vector<uint64_t>& data,
                     bool is_bitmap = false) {
  size_t out_size = is_bitmap ? (data.size() + 7) / 8 : data.size();
  std::vector<uint8_t> out(out_size);

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

  // Count survivors (for validation)
  int survivors = 0;
  if (is_bitmap) {
    for (size_t i = 0; i < data.size() && i/8 < out.size(); ++i) {
      if ((out[i/8] >> (i%8)) & 1) survivors++;
    }
  } else {
    for (auto v : out) survivors += v;
  }

  std::printf("%-25s: %.3f ms, %.1f Mnums/s, survivors=%d\n",
              name, ms_per_run, throughput, survivors);
}

int main() {
  std::printf("=== Wheel Optimization Benchmark ===\n\n");

  const size_t N = 10'000'000;

  // Test 1: Random 32-bit
  std::printf("Random 32-bit (N=10M):\n");
  {
    std::vector<uint64_t> data(N);
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> u32(0, 0xffffffffu);
    for (auto& x : data) x = u32(rng);

    benchmark("Original (236)", neon_fast::filter_stream_u64_barrett16, data);
    benchmark("Ultra (249)", neon_ultra::filter_stream_u64_barrett16_ultra, data);
    benchmark("Wheel+Barrett", neon_wheel::filter_stream_u64_wheel, data);
    benchmark("Wheel+Bitmap", neon_wheel::filter_stream_u64_wheel_bitmap, data, true);
  }

  // Test 2: Composite-heavy (should benefit most from wheel)
  std::printf("\nComposite-heavy (multiples of 6):\n");
  {
    std::vector<uint64_t> data(N);
    for (size_t i = 0; i < N; ++i) {
      data[i] = 6 * (i + 1); // All divisible by 2 and 3
    }

    benchmark("Original", neon_fast::filter_stream_u64_barrett16, data);
    benchmark("Ultra", neon_ultra::filter_stream_u64_barrett16_ultra, data);
    benchmark("Wheel+Barrett", neon_wheel::filter_stream_u64_wheel, data);
    benchmark("Wheel+Bitmap", neon_wheel::filter_stream_u64_wheel_bitmap, data, true);
  }

  // Test 3: Mixed pattern (80% composite)
  std::printf("\nMixed (80%% composite by 2,3,5):\n");
  {
    std::vector<uint64_t> data(N);
    std::mt19937_64 rng(42);
    for (size_t i = 0; i < N; ++i) {
      if (i % 5 < 4) {
        // 80% are multiples of 2, 3, or 5
        int div = (i % 3) + 2;
        if (div == 4) div = 5;
        data[i] = div * (rng() % 100000);
      } else {
        // 20% are potential primes
        data[i] = 30 * (rng() % 100000) + 1; // Residue 1 mod 30
      }
    }

    benchmark("Original", neon_fast::filter_stream_u64_barrett16, data);
    benchmark("Ultra", neon_ultra::filter_stream_u64_barrett16_ultra, data);
    benchmark("Wheel+Barrett", neon_wheel::filter_stream_u64_wheel, data);
    benchmark("Wheel+Bitmap", neon_wheel::filter_stream_u64_wheel_bitmap, data, true);
  }

  // Test 4: Large primes (worst case for wheel)
  std::printf("\nLarge primes only:\n");
  {
    std::vector<uint64_t> data(N);
    const uint64_t primes[] = {59, 61, 67, 71, 73, 79, 83, 89, 97, 101};
    for (size_t i = 0; i < N; ++i) {
      data[i] = primes[i % 10];
    }

    benchmark("Original", neon_fast::filter_stream_u64_barrett16, data);
    benchmark("Ultra", neon_ultra::filter_stream_u64_barrett16_ultra, data);
    benchmark("Wheel+Barrett", neon_wheel::filter_stream_u64_wheel, data);
    benchmark("Wheel+Bitmap", neon_wheel::filter_stream_u64_wheel_bitmap, data, true);
  }

  std::printf("\n=== Summary ===\n");
  std::printf("Wheel optimization most effective on composite-heavy data\n");
  std::printf("Bitmap output reduces memory bandwidth by 8×\n");

  return 0;
}