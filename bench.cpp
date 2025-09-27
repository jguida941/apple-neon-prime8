#include <cstdio>
#include <vector>
#include <random>
#include <chrono>
#include <cstring>
#include "simd_fast.hpp"

using bench_clock = std::chrono::high_resolution_clock;

static void run(const char* name,
                void(*fn)(const uint64_t*, uint8_t*, size_t),
                std::vector<uint64_t>& data)
{
  std::vector<uint8_t> out(data.size());
  auto t0 = bench_clock::now();
  fn(data.data(), out.data(), data.size());
  auto t1 = bench_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  // simple rolling hash so we can sanity check output stability
  uint64_t h = 0xcbf29ce484222325ULL;
  for (auto v : out) { h ^= v; h *= 0x100000001b3ULL; }
  std::printf("%-8s n=%zu time=%.3f ms thr=%.2f Mnums/s hash=%016llx\n",
              name, data.size(), ms, (data.size()/1e6)/(ms/1000.0), (unsigned long long)h);
}

static void run_bitmap(const char* name,
                       void(*fn)(const uint64_t*, uint8_t*, size_t),
                       std::vector<uint64_t>& data)
{
  std::vector<uint8_t> out((data.size() + 7) / 8);
  auto t0 = bench_clock::now();
  fn(data.data(), out.data(), data.size());
  auto t1 = bench_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  // simple rolling hash so we can sanity check output stability
  uint64_t h = 0xcbf29ce484222325ULL;
  for (auto v : out) { h ^= v; h *= 0x100000001b3ULL; }
  std::printf("%-8s n=%zu time=%.3f ms thr=%.2f Mnums/s hash=%016llx\n",
              name, data.size(), ms, (data.size()/1e6)/(ms/1000.0), (unsigned long long)h);
}

static bool verify_correctness(const uint64_t* numbers, size_t count) {
  std::vector<uint8_t> byte_out(count);
  std::vector<uint8_t> bitmap_out((count + 7) / 8);

  neon_fast::filter_stream_u64_barrett16(numbers, byte_out.data(), count);
  neon_fast::filter_stream_u64_barrett16_bitmap(numbers, bitmap_out.data(), count);

  for (size_t i = 0; i < count; ++i) {
    uint8_t byte_val = byte_out[i];
    uint8_t bit_val = (bitmap_out[i / 8] >> (i % 8)) & 1;
    if (byte_val != bit_val) {
      std::printf("Mismatch at index %zu: byte=%u, bit=%u, value=%llu\n",
                  i, byte_val, bit_val, (unsigned long long)numbers[i]);
      return false;
    }
  }
  return true;
}

static void test_tails(void(*fn)(const uint64_t*, uint8_t*, size_t)) {
  std::mt19937_64 rng(123);
  std::uniform_int_distribution<uint64_t> u32(0, 0xffffffffu);

  for (size_t tail = 1; tail <= 15; ++tail) {
    std::vector<uint64_t> tiny(tail);
    for (auto& x : tiny) x = u32(rng);

    std::vector<uint8_t> out(tail);
    fn(tiny.data(), out.data(), tail);

    uint64_t h = 0xcbf29ce484222325ULL;
    for (auto v : out) { h ^= v; h *= 0x100000001b3ULL; }
    std::printf("tail-%02zu: hash=%016llx\n", tail, (unsigned long long)h);
  }
}

int main(int argc, char** argv) {
  const size_t N = 10'000'000;
  std::mt19937_64 rng(42);
  std::uniform_int_distribution<uint64_t> u32(0, 0xffffffffu);

  std::vector<uint64_t> data(N);
  for (auto& x : data) x = u32(rng);

  std::printf("=== Correctness verification ===\n");
  if (verify_correctness(data.data(), 1000)) {
    std::printf("✓ Byte and bitmap outputs match (first 1000 elements)\n");
  } else {
    std::printf("✗ MISMATCH detected between byte and bitmap outputs\n");
    return 1;
  }

  std::printf("\n=== Performance benchmarks ===\n");
  run("simd8", neon_fast::filter_stream_u64_barrett16, data);
  run_bitmap("bitmap", neon_fast::filter_stream_u64_barrett16_bitmap, data);

  std::printf("\n=== Tail handling (<8 elements) ===\n");
  test_tails(neon_fast::filter_stream_u64_barrett16);

  return 0;
}
