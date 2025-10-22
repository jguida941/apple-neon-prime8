// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Justin Guida
#include "simd_fast.hpp"
#include "primes_tables.hpp"
#include <arm_neon.h>
#include <cstdint>
#include <cstddef>

namespace neon_fast {

// --- helpers: register-only u64→u32 loads, lane enable, barrett dual ---
__attribute__((always_inline)) inline
uint32x4_t load_u32x4_from_u64(const uint64_t* p) {
  // load 4×u64 as two 128b regs, then narrow (low 32b of each)
  uint64x2_t a = vld1q_u64(p + 0);
  uint64x2_t b = vld1q_u64(p + 2);
  return vcombine_u32(vmovn_u64(a), vmovn_u64(b));
}


__attribute__((always_inline)) inline
void barrett_modq_u32_dual(uint32x4_t n1, uint32x4_t n2,
                          uint32x4_t mu, uint32x4_t p,
                          uint32x4_t& r1, uint32x4_t& r2) {
  uint64x2_t lo1 = vmull_u32(vget_low_u32(n1), vget_low_u32(mu));
  uint64x2_t hi1 = vmull_u32(vget_high_u32(n1), vget_high_u32(mu));
  uint64x2_t lo2 = vmull_u32(vget_low_u32(n2), vget_low_u32(mu));
  uint64x2_t hi2 = vmull_u32(vget_high_u32(n2), vget_high_u32(mu));

  uint32x2_t ql1 = vshrn_n_u64(lo1, 32);
  uint32x2_t qh1 = vshrn_n_u64(hi1, 32);
  uint32x2_t ql2 = vshrn_n_u64(lo2, 32);
  uint32x2_t qh2 = vshrn_n_u64(hi2, 32);
  uint32x4_t q1 = vcombine_u32(ql1, qh1);
  uint32x4_t q2 = vcombine_u32(ql2, qh2);

  uint32x4_t qp1 = vmulq_u32(q1, p);
  uint32x4_t qp2 = vmulq_u32(q2, p);
  r1 = vsubq_u32(n1, qp1);
  r2 = vsubq_u32(n2, qp2);
  r1 = vsubq_u32(r1, vandq_u32(vcgeq_u32(r1, p), p));
  r2 = vsubq_u32(r2, vandq_u32(vcgeq_u32(r2, p), p));
}

// build composite mask over 16 primes for two vectors
__attribute__((always_inline)) inline
void divisible_mask_dual16(uint32x4_t n1, uint32x4_t n2,
                           uint32x4_t& m1, uint32x4_t& m2) {
  const uint32x4_t zero = vdupq_n_u32(0);
  m1 = vdupq_n_u32(0);
  m2 = vdupq_n_u32(0);

  // Interleave small and ext primes for better dual-issue
  // This pattern helps M-series dual pipelines
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    // Small prime
    {
      const uint32x4_t p  = vdupq_n_u32(SMALL_PRIMES[i]);
      const uint32x4_t mu = vdupq_n_u32(SMALL_MU[i]);
      uint32x4_t r1, r2;
      barrett_modq_u32_dual(n1, n2, mu, p, r1, r2);
      uint32x4_t d1 = vceqq_u32(r1, zero);
      uint32x4_t d2 = vceqq_u32(r2, zero);
      d1 = vandq_u32(d1, vmvnq_u32(vceqq_u32(n1, p)));
      d2 = vandq_u32(d2, vmvnq_u32(vceqq_u32(n2, p)));
      m1 = vorrq_u32(m1, d1);
      m2 = vorrq_u32(m2, d2);
    }
    // Ext prime
    {
      const uint32x4_t p  = vdupq_n_u32(EXT_PRIMES[i]);
      const uint32x4_t mu = vdupq_n_u32(EXT_MU[i]);
      uint32x4_t r1, r2;
      barrett_modq_u32_dual(n1, n2, mu, p, r1, r2);
      uint32x4_t d1 = vceqq_u32(r1, zero);
      uint32x4_t d2 = vceqq_u32(r2, zero);
      d1 = vandq_u32(d1, vmvnq_u32(vceqq_u32(n1, p)));
      d2 = vandq_u32(d2, vmvnq_u32(vceqq_u32(n2, p)));
      m1 = vorrq_u32(m1, d1);
      m2 = vorrq_u32(m2, d2);
    }
  }
}
// Convert 0xFFFFFFFF/0x00000000 to byte lanes (1/0) and store with one vst1_u8.
__attribute__((always_inline)) inline
uint8x8_t bytes_from_u32_mask(uint32x4_t sv1, uint32x4_t sv2) {
  uint16x4_t s1_16 = vmovn_u32(sv1);
  uint16x4_t s2_16 = vmovn_u32(sv2);
  uint8x8_t  s8    = vmovn_u16(vcombine_u16(s1_16, s2_16)); // 0xFF/0x00
  return vshr_n_u8(s8, 7); // 1/0 per lane
}

// --- replace the whole function ---
__attribute__((used)) __attribute__((always_inline)) inline
void filter8_u64_barrett16(const uint64_t* __restrict ptr,
                           uint8_t*       __restrict out)
{
  // Load 8×u64 as four 128b regs (kept in registers for reuse)
  uint64x2_t a0 = vld1q_u64(ptr + 0);
  uint64x2_t a1 = vld1q_u64(ptr + 2);
  uint64x2_t a2 = vld1q_u64(ptr + 4);
  uint64x2_t a3 = vld1q_u64(ptr + 6);

  // One-time lane-width check for lanes 0..7 (high32 == 0 ⇒ fits u32)
  uint64x2_t h0 = vshrq_n_u64(a0, 32);
  uint64x2_t h1 = vshrq_n_u64(a1, 32);
  uint64x2_t h2 = vshrq_n_u64(a2, 32);
  uint64x2_t h3 = vshrq_n_u64(a3, 32);
  uint64x2_t any01 = vorrq_u64(h0, h1);
  uint64x2_t any23 = vorrq_u64(h2, h3);
  uint64x2_t any   = vorrq_u64(any01, any23);
  const bool all32 = ((vgetq_lane_u64(any,0) | vgetq_lane_u64(any,1)) == 0ULL);

  // Form n1 = lanes 0..3, n2 = lanes 4..7 (register-only narrowing)
  uint32x4_t n1 = vcombine_u32(vmovn_u64(a0), vmovn_u64(a1));
  uint32x4_t n2 = vcombine_u32(vmovn_u64(a2), vmovn_u64(a3));

  // Barrett filters across 16 primes (dual 4-lane)
  uint32x4_t m1, m2;
  divisible_mask_dual16(n1, n2, m1, m2); // 0xFFFFFFFF where composite (and n!=p handled inside)

  const uint32x4_t zero = vdupq_n_u32(0);
  uint32x4_t sv1 = vceqq_u32(m1, zero);
  uint32x4_t sv2 = vceqq_u32(m2, zero);

  if (!all32) {
    // Per-lane enable only when needed
    uint32x4_t en_lo = vceqq_u32(vcombine_u32(vmovn_u64(h0), vmovn_u64(h1)), vdupq_n_u32(0));
    uint32x4_t en_hi = vceqq_u32(vcombine_u32(vmovn_u64(h2), vmovn_u64(h3)), vdupq_n_u32(0));
    sv1 = vandq_u32(sv1, en_lo);
    sv2 = vandq_u32(sv2, en_hi);
  }

  // Single 8-byte store (1/0 per lane as bytes)
  vst1_u8(out, bytes_from_u32_mask(sv1, sv2));
}

// tiny scalar tail for <8 elements, matching the same semantics
__attribute__((always_inline)) inline
uint8_t scalar_survive_barrett16(uint64_t v) {
  if (v > 0xffffffffu) return 0; // disabled in our fastpath contract → count as 0 here
  uint32_t n = (uint32_t)v;
  #pragma clang loop unroll_count(2)
  for (int i=0;i<8;++i) {
    if (n != SMALL_PRIMES[i]) {
      const uint64_t q = (uint64_t)n * SMALL_MU[i] >> 32;
      uint32_t r = n - (uint32_t)q * SMALL_PRIMES[i];
      if (r >= SMALL_PRIMES[i]) r -= SMALL_PRIMES[i];
      if (r == 0) return 0;
    }
    if (n != EXT_PRIMES[i]) {
      const uint64_t q = (uint64_t)n * EXT_MU[i] >> 32;
      uint32_t r = n - (uint32_t)q * EXT_PRIMES[i];
      if (r >= EXT_PRIMES[i]) r -= EXT_PRIMES[i];
      if (r == 0) return 0;
    }
  }
  return 1;
}

void filter_stream_u64_barrett16(const uint64_t* __restrict numbers,
                                 uint8_t*       __restrict out,
                                 size_t count) {
  size_t i = 0;
  // Process in batches of 16 for better pipelining
  for (; i + 16 <= count; i += 16) {
    __builtin_prefetch(numbers + i + 32, 0, 1);  // read, low temporal
    filter8_u64_barrett16(numbers + i, out + i);
    filter8_u64_barrett16(numbers + i + 8, out + i + 8);
  }
  // Process remaining 8
  for (; i + 8 <= count; i += 8) {
    filter8_u64_barrett16(numbers + i, out + i);
  }
  // tail
  for (; i < count; ++i) out[i] = scalar_survive_barrett16(numbers[i]);
}


__attribute__((always_inline)) inline
uint8_t bitpack_from_u32_mask(uint32x4_t sv1, uint32x4_t sv2) {
  // sv* are 0xFFFFFFFF for "survive", 0x00000000 otherwise.
  // Use SIMD to efficiently pack bits
  uint16x4_t s1_16 = vmovn_u32(sv1);  // 0xFFFF or 0x0000 per lane
  uint16x4_t s2_16 = vmovn_u32(sv2);
  uint8x8_t s8 = vmovn_u16(vcombine_u16(s1_16, s2_16));  // 0xFF or 0x00

  // Now s8 contains 8 bytes, each 0xFF or 0x00
  // Convert to bit positions and accumulate
  const uint8x8_t weights = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
  uint8x8_t bits = vand_u8(s8, weights);  // Mask each byte with its bit position

  // Sum all bytes to get final packed byte
  return vaddv_u8(bits);
}

void filter_stream_u64_barrett16_bitmap(const uint64_t* __restrict numbers,
                                        uint8_t*       __restrict bitmap,
                                        size_t count)
{
  size_t i = 0, byte = 0;
  // Process in batches of 16 for better pipelining
  for (; i + 16 <= count; i += 16, byte += 2) {
    __builtin_prefetch(numbers + i + 32, 0, 1);  // read, low temporal

    // Load four regs
    uint64x2_t a0 = vld1q_u64(numbers + i + 0);
    uint64x2_t a1 = vld1q_u64(numbers + i + 2);
    uint64x2_t a2 = vld1q_u64(numbers + i + 4);
    uint64x2_t a3 = vld1q_u64(numbers + i + 6);

    uint64x2_t h0 = vshrq_n_u64(a0, 32);
    uint64x2_t h1 = vshrq_n_u64(a1, 32);
    uint64x2_t h2 = vshrq_n_u64(a2, 32);
    uint64x2_t h3 = vshrq_n_u64(a3, 32);

    uint64x2_t any01 = vorrq_u64(h0, h1);
    uint64x2_t any23 = vorrq_u64(h2, h3);
    uint64x2_t any   = vorrq_u64(any01, any23);
    const bool all32 = ((vgetq_lane_u64(any,0) | vgetq_lane_u64(any,1)) == 0ULL);

    uint32x4_t n1 = vcombine_u32(vmovn_u64(a0), vmovn_u64(a1));
    uint32x4_t n2 = vcombine_u32(vmovn_u64(a2), vmovn_u64(a3));

    uint32x4_t m1, m2;
    divisible_mask_dual16(n1, n2, m1, m2);

    const uint32x4_t zero = vdupq_n_u32(0);
    uint32x4_t sv1 = vceqq_u32(m1, zero);
    uint32x4_t sv2 = vceqq_u32(m2, zero);

    if (!all32) {
      uint32x4_t en_lo = vceqq_u32(vcombine_u32(vmovn_u64(h0), vmovn_u64(h1)), vdupq_n_u32(0));
      uint32x4_t en_hi = vceqq_u32(vcombine_u32(vmovn_u64(h2), vmovn_u64(h3)), vdupq_n_u32(0));
      sv1 = vandq_u32(sv1, en_lo);
      sv2 = vandq_u32(sv2, en_hi);
    }

    bitmap[byte] = bitpack_from_u32_mask(sv1, sv2);

    // Process second batch of 8
    a0 = vld1q_u64(numbers + i + 8);
    a1 = vld1q_u64(numbers + i + 10);
    a2 = vld1q_u64(numbers + i + 12);
    a3 = vld1q_u64(numbers + i + 14);

    h0 = vshrq_n_u64(a0, 32);
    h1 = vshrq_n_u64(a1, 32);
    h2 = vshrq_n_u64(a2, 32);
    h3 = vshrq_n_u64(a3, 32);

    any01 = vorrq_u64(h0, h1);
    any23 = vorrq_u64(h2, h3);
    any   = vorrq_u64(any01, any23);
    const bool all32_2 = ((vgetq_lane_u64(any,0) | vgetq_lane_u64(any,1)) == 0ULL);

    n1 = vcombine_u32(vmovn_u64(a0), vmovn_u64(a1));
    n2 = vcombine_u32(vmovn_u64(a2), vmovn_u64(a3));

    divisible_mask_dual16(n1, n2, m1, m2);

    sv1 = vceqq_u32(m1, zero);
    sv2 = vceqq_u32(m2, zero);

    if (!all32_2) {
      uint32x4_t en_lo = vceqq_u32(vcombine_u32(vmovn_u64(h0), vmovn_u64(h1)), vdupq_n_u32(0));
      uint32x4_t en_hi = vceqq_u32(vcombine_u32(vmovn_u64(h2), vmovn_u64(h3)), vdupq_n_u32(0));
      sv1 = vandq_u32(sv1, en_lo);
      sv2 = vandq_u32(sv2, en_hi);
    }

    bitmap[byte + 1] = bitpack_from_u32_mask(sv1, sv2);
  }

  // Process remaining 8
  for (; i + 8 <= count; i += 8, ++byte) {
    uint64x2_t a0 = vld1q_u64(numbers + i + 0);
    uint64x2_t a1 = vld1q_u64(numbers + i + 2);
    uint64x2_t a2 = vld1q_u64(numbers + i + 4);
    uint64x2_t a3 = vld1q_u64(numbers + i + 6);

    uint64x2_t h0 = vshrq_n_u64(a0, 32);
    uint64x2_t h1 = vshrq_n_u64(a1, 32);
    uint64x2_t h2 = vshrq_n_u64(a2, 32);
    uint64x2_t h3 = vshrq_n_u64(a3, 32);

    uint64x2_t any01 = vorrq_u64(h0, h1);
    uint64x2_t any23 = vorrq_u64(h2, h3);
    uint64x2_t any   = vorrq_u64(any01, any23);
    const bool all32 = ((vgetq_lane_u64(any,0) | vgetq_lane_u64(any,1)) == 0ULL);

    uint32x4_t n1 = vcombine_u32(vmovn_u64(a0), vmovn_u64(a1));
    uint32x4_t n2 = vcombine_u32(vmovn_u64(a2), vmovn_u64(a3));

    uint32x4_t m1, m2;
    divisible_mask_dual16(n1, n2, m1, m2);

    const uint32x4_t zero = vdupq_n_u32(0);
    uint32x4_t sv1 = vceqq_u32(m1, zero);
    uint32x4_t sv2 = vceqq_u32(m2, zero);

    if (!all32) {
      uint32x4_t en_lo = vceqq_u32(vcombine_u32(vmovn_u64(h0), vmovn_u64(h1)), vdupq_n_u32(0));
      uint32x4_t en_hi = vceqq_u32(vcombine_u32(vmovn_u64(h2), vmovn_u64(h3)), vdupq_n_u32(0));
      sv1 = vandq_u32(sv1, en_lo);
      sv2 = vandq_u32(sv2, en_hi);
    }

    bitmap[byte] = bitpack_from_u32_mask(sv1, sv2);
  }

  // Tail (<8): same semantics as scalar_survive_barrett16
  if (i < count) {
    uint8_t last = 0;
    for (unsigned b = 0; i < count; ++i, ++b) {
      last |= (scalar_survive_barrett16(numbers[i]) & 1u) << b;
    }
    bitmap[byte] = last;
  }
}
} // namespace neon_fast
