#include <cstdio>
#include <vector>
#include <random>
#include <chrono>
#include <cstring>
#include "src/simd_fast.hpp"

using bench_clock = std::chrono::high_resolution_clock;

static void benchmark_pattern(const char* name, const std::vector<uint64_t>& data) {
  const int RUNS = 100;
  double byte_time = 0.0, bitmap_time = 0.0;

  std::vector<uint8_t> byte_out(data.size());
  std::vector<uint8_t> bitmap_out((data.size() + 7) / 8);

  // Warm-up
  for (int i = 0; i < 10; ++i) {
    neon_fast::filter_stream_u64_barrett16(data.data(), byte_out.data(), data.size());
    neon_fast::filter_stream_u64_barrett16_bitmap(data.data(), bitmap_out.data(), data.size());
  }

  // Byte output benchmark
  auto t0 = bench_clock::now();
  for (int i = 0; i < RUNS; ++i) {
    neon_fast::filter_stream_u64_barrett16(data.data(), byte_out.data(), data.size());
  }
  auto t1 = bench_clock::now();
  byte_time = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

  // Bitmap output benchmark
  t0 = bench_clock::now();
  for (int i = 0; i < RUNS; ++i) {
    neon_fast::filter_stream_u64_barrett16_bitmap(data.data(), bitmap_out.data(), data.size());
  }
  t1 = bench_clock::now();
  bitmap_time = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

  // Verify correctness
  bool correct = true;
  for (size_t i = 0; i < data.size(); ++i) {
    uint8_t bit = (bitmap_out[i / 8] >> (i % 8)) & 1;
    if (byte_out[i] != bit) {
      correct = false;
      break;
    }
  }

  // Count survivors
  int survivors = 0;
  for (auto b : byte_out) survivors += b;

  double byte_thr = (data.size() / 1e6) / (byte_time / 1000.0);
  double bitmap_thr = (data.size() / 1e6) / (bitmap_time / 1000.0);

  std::printf("%-20s: byte=%.2fms (%.1f M/s), bitmap=%.2fms (%.1f M/s), speedup=%.2fx, survivors=%d/%zu %s\n",
              name, byte_time, byte_thr, bitmap_time, bitmap_thr,
              byte_time / bitmap_time, survivors, data.size(),
              correct ? "✓" : "✗ MISMATCH");
}

int main() {
  const size_t N = 1'000'000;

  std::printf("=== SIMD Prime Filter Comprehensive Test ===\n");
  std::printf("Testing %zu numbers per pattern\n\n", N);

  // Pattern 1: Random 32-bit values
  {
    std::vector<uint64_t> data(N);
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> u32(0, 0xffffffffu);
    for (auto& x : data) x = u32(rng);
    benchmark_pattern("Random 32-bit", data);
  }

  // Pattern 2: Mix of 32-bit and 64-bit
  {
    std::vector<uint64_t> data(N);
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> u32(0, 0xffffffffu);
    std::uniform_int_distribution<uint64_t> u64(0x100000000ull, 0xffffffffffffffffull);
    for (size_t i = 0; i < N; ++i) {
      data[i] = (i % 10 < 8) ? u32(rng) : u64(rng);  // 80% 32-bit, 20% 64-bit
    }
    benchmark_pattern("Mixed 80/20", data);
  }

  // Pattern 3: Small primes (worst case - all survive)
  {
    std::vector<uint64_t> data(N);
    const uint64_t primes[] = {59, 61, 67, 71, 73, 79, 83, 89, 97, 101};
    for (size_t i = 0; i < N; ++i) {
      data[i] = primes[i % 10];
    }
    benchmark_pattern("Large primes", data);
  }

  // Pattern 4: Composites (best case - none survive)
  {
    std::vector<uint64_t> data(N);
    for (size_t i = 0; i < N; ++i) {
      data[i] = (i + 1) * 6;  // All divisible by 2 and 3
    }
    benchmark_pattern("All composites", data);
  }

  // Pattern 5: Sequential numbers
  {
    std::vector<uint64_t> data(N);
    for (size_t i = 0; i < N; ++i) {
      data[i] = i + 1;
    }
    benchmark_pattern("Sequential 1..N", data);
  }

  // Pattern 6: Powers of 2
  {
    std::vector<uint64_t> data(N);
    for (size_t i = 0; i < N; ++i) {
      data[i] = 1ull << (i % 32);
    }
    benchmark_pattern("Powers of 2", data);
  }

  // Test various sizes for scalability
  std::printf("\n=== Scalability Test ===\n");
  std::mt19937_64 rng(42);
  std::uniform_int_distribution<uint64_t> u32(0, 0xffffffffu);

  for (size_t size : {8, 16, 32, 64, 128, 256, 512, 1024, 10000, 100000, 1000000}) {
    std::vector<uint64_t> data(size);
    for (auto& x : data) x = u32(rng);

    char name[32];
    std::snprintf(name, sizeof(name), "Size %zu", size);
    benchmark_pattern(name, data);
  }

  return 0;
}