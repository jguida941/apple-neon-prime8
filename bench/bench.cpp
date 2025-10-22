#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
#include <arm_neon.h>
#include "simd_fast.hpp"
#include "primes_tables.hpp"

using bench_clock = std::chrono::high_resolution_clock;

namespace {

constexpr uint64_t kFnvBasis = 0xcbf29ce484222325ULL;
constexpr uint64_t kFnvPrime = 0x100000001b3ULL;

uint64_t hash_bytes(const std::vector<uint8_t>& data) {
  uint64_t h = kFnvBasis;
  for (uint8_t b : data) {
    h ^= b;
    h *= kFnvPrime;
  }
  return h;
}

uint8_t scalar_ref(uint64_t v) {
  if (v > 0xffffffffu) return 0;
  uint32_t n = static_cast<uint32_t>(v);
  for (int i = 0; i < 8; ++i) {
    if (n != SMALL_PRIMES[i]) {
      uint64_t q = (uint64_t)n * SMALL_MU[i] >> 32;
      uint32_t r = n - static_cast<uint32_t>(q) * SMALL_PRIMES[i];
      if (r >= SMALL_PRIMES[i]) r -= SMALL_PRIMES[i];
      if (r == 0) return 0;
    }
    if (n != EXT_PRIMES[i]) {
      uint64_t q = (uint64_t)n * EXT_MU[i] >> 32;
      uint32_t r = n - static_cast<uint32_t>(q) * EXT_PRIMES[i];
      if (r >= EXT_PRIMES[i]) r -= EXT_PRIMES[i];
      if (r == 0) return 0;
    }
  }
  return 1;
}

double run_scalar(const char* label, const std::vector<uint64_t>& numbers) {
  std::vector<uint8_t> out(numbers.size());
  auto t0 = bench_clock::now();
  for (size_t i = 0; i < numbers.size(); ++i) out[i] = scalar_ref(numbers[i]);
  auto t1 = bench_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  const uint64_t h = hash_bytes(out);
  std::printf("%-14s n=%zu time=%9.3f ms thr=%7.2f Mnums/s hash=%016llx\n",
              label, numbers.size(), ms,
              (numbers.size() / 1e6) / (ms / 1000.0),
              static_cast<unsigned long long>(h));
  return ms;
}

double run_bytes(const char* label,
                 void (*fn)(const uint64_t*, uint8_t*, size_t),
                 const std::vector<uint64_t>& numbers) {
  std::vector<uint8_t> out(numbers.size());
  auto t0 = bench_clock::now();
  fn(numbers.data(), out.data(), numbers.size());
  auto t1 = bench_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  const uint64_t h = hash_bytes(out);
  std::printf("%-14s n=%zu time=%9.3f ms thr=%7.2f Mnums/s hash=%016llx\n",
              label, numbers.size(), ms,
              (numbers.size() / 1e6) / (ms / 1000.0),
              static_cast<unsigned long long>(h));
  return ms;
}

double run_bitmap(const char* label,
                  void (*fn)(const uint64_t*, uint8_t*, size_t),
                  const std::vector<uint64_t>& numbers) {
  std::vector<uint8_t> out((numbers.size() + 7) / 8);
  auto t0 = bench_clock::now();
  fn(numbers.data(), out.data(), numbers.size());
  auto t1 = bench_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  const uint64_t h = hash_bytes(out);
  std::printf("%-14s n=%zu time=%9.3f ms thr=%7.2f Mnums/s hash=%016llx\n",
              label, numbers.size(), ms,
              (numbers.size() / 1e6) / (ms / 1000.0),
              static_cast<unsigned long long>(h));
  return ms;
}

bool verify_consistency(const std::vector<uint64_t>& numbers) {
  const size_t n = numbers.size();
  std::vector<uint8_t> bytes(n);
  std::vector<uint8_t> bitmap((n + 7) / 8);
  std::vector<uint8_t> wheel210((n + 7) / 8);

  neon_fast::filter_stream_u64_barrett16(numbers.data(), bytes.data(), n);
  neon_fast::filter_stream_u64_barrett16_bitmap(numbers.data(), bitmap.data(), n);
  neon_wheel210_efficient::filter_stream_u64_wheel210_efficient_bitmap(
      numbers.data(), wheel210.data(), n);

  for (size_t i = 0; i < n; ++i) {
    uint8_t byte_val = bytes[i];
    uint8_t bitmap_val = (bitmap[i >> 3] >> (i & 7)) & 1u;
    uint8_t wheel_val = (wheel210[i >> 3] >> (i & 7)) & 1u;
    if (byte_val != bitmap_val || (wheel_val && !byte_val)) {
      std::printf("Consistency failure at idx=%zu value=%llu byte=%u bitmap=%u wheel=%u\n",
                  i, static_cast<unsigned long long>(numbers[i]),
                  static_cast<unsigned>(byte_val),
                  static_cast<unsigned>(bitmap_val),
                  static_cast<unsigned>(wheel_val));
      return false;
    }
  }
  return true;
}

std::vector<uint64_t> make_uniform_dataset(size_t n, std::mt19937_64 rng) {
  std::uniform_int_distribution<uint64_t> dist(0, 0xffffffffu);
  std::vector<uint64_t> data(n);
  for (auto& v : data) v = dist(rng);
  return data;
}

std::vector<uint64_t> make_mixed_dataset(size_t n, std::mt19937_64 rng) {
  std::uniform_int_distribution<uint64_t> dist32(0, 0xffffffffu);
  std::uniform_int_distribution<uint64_t> dist48(0, (1ull << 48) - 1);
  std::vector<uint64_t> data(n);
  for (size_t i = 0; i < n; ++i) {
    if (i % 5 == 0) {
      data[i] = 0x100000000ull + (dist48(rng) & 0xffffu);
    } else if (i % 11 == 0) {
      data[i] = (static_cast<uint64_t>(i) << 32) | 0xabcdefu;
    } else {
      data[i] = dist32(rng);
    }
  }
  return data;
}

void test_tails(void (*fn)(const uint64_t*, uint8_t*, size_t)) {
  std::mt19937_64 rng(321);
  std::uniform_int_distribution<uint64_t> dist32(0, 0xffffffffu);
  for (size_t tail = 1; tail <= 15; ++tail) {
    std::vector<uint64_t> values(tail);
    for (auto& v : values) v = dist32(rng);
    std::vector<uint8_t> out(tail);
    fn(values.data(), out.data(), tail);
    std::printf("tail-%02zu: hash=%016llx\n",
                tail, static_cast<unsigned long long>(hash_bytes(out)));
  }
}

void benchmark_suite(const char* label, const std::vector<uint64_t>& data) {
  std::printf("\n=== Performance (%s) ===\n", label);
  const double scalar_ms = run_scalar("scalar-ref", data);
  const double simd_bytes_ms =
      run_bytes("simd8-bytes", neon_fast::filter_stream_u64_barrett16, data);
  const double simd_bitmap_ms =
      run_bitmap("simd8-bitmap", neon_fast::filter_stream_u64_barrett16_bitmap, data);
  const double wheel_ms =
      run_bitmap("wheel210-bm", neon_wheel210_efficient::filter_stream_u64_wheel210_efficient_bitmap, data);
  std::printf("   speedups vs scalar: bytes %.2fx  bitmap %.2fx  wheel210 %.2fx\n",
              scalar_ms / simd_bytes_ms,
              scalar_ms / simd_bitmap_ms,
              scalar_ms / wheel_ms);
}

} // namespace

int main(int argc, char** argv) {
  size_t N = 10'000'000;
  uint64_t seed = 42;
  if (argc > 1) N = std::strtoull(argv[1], nullptr, 10);
  if (argc > 2) seed = std::strtoull(argv[2], nullptr, 10);

  auto uniform = make_uniform_dataset(N, std::mt19937_64(seed));
  auto mixed = make_mixed_dataset(
      N, std::mt19937_64(seed ^ 0x9e3779b97f4a7c15ULL));

  std::printf("Dataset size: %zu numbers (seed=%llu)\n",
              N, static_cast<unsigned long long>(seed));

  std::printf("=== Correctness verification ===\n");
  if (!verify_consistency(uniform)) {
    std::puts("✗ Consistency failure on uniform dataset");
    return 1;
  }
  if (!verify_consistency(mixed)) {
    std::puts("✗ Consistency failure on mixed dataset");
    return 1;
  }
  std::puts("✓ Byte/bitmap outputs consistent on both datasets");

  benchmark_suite("uniform 32-bit random", uniform);
  benchmark_suite("mixed 32/64-bit", mixed);

  std::printf("\n=== Tail handling (<8 elements) ===\n");
  test_tails(neon_fast::filter_stream_u64_barrett16);
  return 0;
}
