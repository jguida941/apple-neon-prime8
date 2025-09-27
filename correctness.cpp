#include <cstdio>
#include <random>
#include <vector>
#include <arm_neon.h>
#include "simd_fast.hpp"
#include "primes_tables.hpp"

static uint8_t scalar_ref(uint64_t v) {
  if (v > 0xffffffffu) return 0;
  uint32_t n = (uint32_t)v;
  for (int i=0;i<8;++i) {
    if (n != SMALL_PRIMES[i]) {
      uint64_t q = (uint64_t)n * SMALL_MU[i] >> 32;
      uint32_t r = n - (uint32_t)q * SMALL_PRIMES[i];
      if (r >= SMALL_PRIMES[i]) r -= SMALL_PRIMES[i];
      if (r == 0) return 0;
    }
    if (n != EXT_PRIMES[i]) {
      uint64_t q = (uint64_t)n * EXT_MU[i] >> 32;
      uint32_t r = n - (uint32_t)q * EXT_PRIMES[i];
      if (r >= EXT_PRIMES[i]) r -= EXT_PRIMES[i];
      if (r == 0) return 0;
    }
  }
  return 1;
}

int main() {
  std::mt19937_64 rng(12345);
  std::uniform_int_distribution<uint64_t> u32(0, 0xffffffffu);

  const int ITERS = 20000; // 160k lanes
  std::vector<uint64_t> input(8);
  std::vector<uint8_t>  out(8), ref(8);

  for (int it=0; it<ITERS; ++it) {
    for (int i=0;i<8;++i) input[i] = u32(rng);
    neon_fast::filter8_u64_barrett16(input.data(), out.data());
    for (int i=0;i<8;++i) ref[i] = scalar_ref(input[i]);
    for (int i=0;i<8;++i) {
      if (out[i] != ref[i]) {
        std::printf("Mismatch lane=%d n=%u out=%u ref=%u\n",
                    i, (unsigned)input[i], (unsigned)out[i], (unsigned)ref[i]);
        return 1;
      }
    }
  }
  std::puts("OK");
  return 0;
}