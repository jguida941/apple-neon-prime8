#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
#include <arm_neon.h>
#include "simd_fast.hpp"
#include "primes_tables.hpp"

namespace {

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

bool verify_filter8_random(uint64_t high_max, int iters, std::mt19937_64& rng) {
  std::uniform_int_distribution<uint64_t> dist(0, high_max);
  std::vector<uint64_t> input(8);
  std::vector<uint8_t> out(8), ref(8);
  for (int it = 0; it < iters; ++it) {
    for (int i = 0; i < 8; ++i) input[i] = dist(rng);
    neon_fast::filter8_u64_barrett16(input.data(), out.data());
    for (int i = 0; i < 8; ++i) ref[i] = scalar_ref(input[i]);
    for (int i = 0; i < 8; ++i) {
      if (out[i] != ref[i]) {
        std::printf("filter8 mismatch (iter=%d lane=%d value=%llu) got=%u expected=%u\n",
                    it, i, static_cast<unsigned long long>(input[i]),
                    static_cast<unsigned>(out[i]), static_cast<unsigned>(ref[i]));
        return false;
      }
    }
  }
  return true;
}

uint8_t bitmap_get(const std::vector<uint8_t>& bitmap, size_t idx) {
  return (bitmap[idx >> 3] >> (idx & 7)) & 1u;
}

bool verify_stream_functions(const std::vector<uint64_t>& values) {
  const size_t n = values.size();
  std::vector<uint8_t> ref(n);
  for (size_t i = 0; i < n; ++i) ref[i] = scalar_ref(values[i]);

  std::vector<uint8_t> byte_out(n), bitmap_out((n + 7) / 8),
      wheel_bitmap((n + 7) / 8);

  neon_fast::filter_stream_u64_barrett16(values.data(), byte_out.data(), n);
  neon_fast::filter_stream_u64_barrett16_bitmap(values.data(), bitmap_out.data(), n);
  neon_wheel210_efficient::filter_stream_u64_wheel210_efficient_bitmap(
      values.data(), wheel_bitmap.data(), n);

  for (size_t i = 0; i < n; ++i) {
    const uint8_t expected = ref[i];
    const uint8_t byte_val = byte_out[i];
    const uint8_t bit_val = bitmap_get(bitmap_out, i);
    const uint8_t wheel_val = bitmap_get(wheel_bitmap, i);

    if (byte_val != expected || bit_val != expected || (wheel_val && !expected)) {
      std::printf("stream mismatch idx=%zu value=%llu expected=%u byte=%u bit=%u wheel210=%u\n",
                  i, static_cast<unsigned long long>(values[i]),
                  static_cast<unsigned>(expected),
                  static_cast<unsigned>(byte_val),
                  static_cast<unsigned>(bit_val),
                  static_cast<unsigned>(wheel_val));
      return false;
    }
  }
  return true;
}

std::vector<uint64_t> make_mixed_values(size_t n, std::mt19937_64& rng) {
  std::vector<uint64_t> values(n);
  std::uniform_int_distribution<uint64_t> small(0, 0xffffffffu);
  std::uniform_int_distribution<uint64_t> large64(0, (1ull << 48) - 1);
  for (size_t i = 0; i < n; ++i) {
    if (i % 4 == 3) {
      values[i] = 0xffffffffull + 1ull + (large64(rng) & 0xffff);
    } else if (i % 5 == 0) {
      values[i] = (static_cast<uint64_t>(i) << 32) | 0xfedcba98ull;
    } else {
      values[i] = small(rng);
    }
  }
  return values;
}

std::vector<uint32_t> sieve_primes(uint32_t limit) {
  std::vector<uint8_t> sieve(limit + 1, 1);
  if (!sieve.empty()) sieve[0] = 0;
  if (limit >= 1) sieve[1] = 0;
  for (uint32_t p = 2; p * p <= limit; ++p) {
    if (!sieve[p]) continue;
    for (uint32_t q = p * p; q <= limit; q += p) sieve[q] = 0;
  }
  std::vector<uint32_t> primes;
  for (uint32_t i = 2; i <= limit; ++i) {
    if (sieve[i]) primes.push_back(i);
  }
  return primes;
}

bool verify_primes_survive(uint32_t limit) {
  auto primes = sieve_primes(limit);
  std::vector<uint64_t> values(primes.begin(), primes.end());
  return verify_stream_functions(values);
}

bool verify_small_prime_multiples(uint32_t limit) {
  std::vector<uint64_t> values;
  values.reserve(limit / 2);
  for (uint32_t p : SMALL_PRIMES) {
    for (uint32_t n = p * 2; n <= limit; n += p) values.push_back(n);
  }
  for (uint32_t p : EXT_PRIMES) {
    for (uint32_t n = p * 2; n <= limit; n += p) values.push_back(n);
  }
  return verify_stream_functions(values);
}

bool verify_high32_elimination(size_t count) {
  std::vector<uint64_t> values(count);
  uint64_t base = 0x1'0000'0000ull;
  for (size_t i = 0; i < count; ++i) {
    values[i] = base + static_cast<uint64_t>(i) * 1021ull;
  }
  return verify_stream_functions(values);
}

bool verify_known_patterns() {
  std::vector<uint64_t> values;
  // Exact small numbers, including primes and composites, plus >32-bit cases.
  for (uint64_t v = 0; v <= 127; ++v) values.push_back(v);
  std::array<uint64_t, 10> extras = {
      0xffffffffull, 0x100000000ull, 0x100000001ull, 0xfffffffbull,
      0xfffffffdull, 0x7fffffffull, 0x80000000ull, 0xffffffffull - 1,
      4294967291ull, 4294967295ull};
  values.insert(values.end(), extras.begin(), extras.end());
  return verify_stream_functions(values);
}

bool verify_random_batches(std::mt19937_64& rng) {
  // Random 32-bit range batches
  for (int rep = 0; rep < 5; ++rep) {
    auto mixed = make_mixed_values(8192, rng);
    if (!verify_stream_functions(mixed)) return false;
  }
  return true;
}

bool verify_tails(std::mt19937_64& rng) {
  for (size_t len = 0; len <= 31; ++len) {
    auto values = make_mixed_values(len, rng);
    if (!verify_stream_functions(values)) {
      std::printf("tail verification failed for length=%zu\n", len);
      return false;
    }
  }
  return true;
}

} // namespace

int main() {
  std::mt19937_64 rng(12345);

  if (!verify_filter8_random(0xffffffffu, 20000, rng)) return 1;
  if (!verify_filter8_random((1ull << 48) - 1, 20000, rng)) return 1;

  if (!verify_known_patterns()) return 1;
  if (!verify_primes_survive(1'000'000)) return 1;
  if (!verify_small_prime_multiples(1'000'000)) return 1;
  if (!verify_high32_elimination(16'384)) return 1;
  if (!verify_random_batches(rng)) return 1;
  if (!verify_tails(rng)) return 1;

  std::puts("OK");
  return 0;
}
