// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Justin Guida
#include "simd_fast.hpp"
#include "primes_tables.hpp"
#include <arm_neon.h>
#include <cstdint>
#include <cstddef>
#include <cstring>

namespace neon_ultra {

// === SIMD bitpack helper for ARM NEON ===
__attribute__((always_inline)) inline
uint8_t movemask8_from_u32(uint32x4_t sv1, uint32x4_t sv2) {
  uint16x4_t s1 = vmovn_u32(sv1);
  uint16x4_t s2 = vmovn_u32(sv2);
  uint8x8_t  b  = vmovn_u16(vcombine_u16(s1, s2)); // 0xFF/0x00 per lane
  // turn bytes into bits and horizontally sum
  const uint8x8_t w = {1,2,4,8,16,32,64,128};
  uint8x8_t t = vand_u8(vshr_n_u8(b, 7), w);
  t = vpadd_u8(t, t); t = vpadd_u8(t, t); t = vpadd_u8(t, t);
  return vget_lane_u8(t, 0);
}

// === Strategy 1: Process 32 numbers at once (4x unroll) ===
__attribute__((always_inline)) inline
void barrett_modq_u32_quad(uint32x4_t n1, uint32x4_t n2, uint32x4_t n3, uint32x4_t n4,
                           uint32x4_t mu, uint32x4_t p,
                           uint32x4_t& r1, uint32x4_t& r2, uint32x4_t& r3, uint32x4_t& r4) {
  // Process 4 vectors in parallel to maximize ILP
  uint64x2_t lo1 = vmull_u32(vget_low_u32(n1), vget_low_u32(mu));
  uint64x2_t hi1 = vmull_u32(vget_high_u32(n1), vget_high_u32(mu));
  uint64x2_t lo2 = vmull_u32(vget_low_u32(n2), vget_low_u32(mu));
  uint64x2_t hi2 = vmull_u32(vget_high_u32(n2), vget_high_u32(mu));
  uint64x2_t lo3 = vmull_u32(vget_low_u32(n3), vget_low_u32(mu));
  uint64x2_t hi3 = vmull_u32(vget_high_u32(n3), vget_high_u32(mu));
  uint64x2_t lo4 = vmull_u32(vget_low_u32(n4), vget_low_u32(mu));
  uint64x2_t hi4 = vmull_u32(vget_high_u32(n4), vget_high_u32(mu));

  uint32x4_t q1 = vcombine_u32(vshrn_n_u64(lo1, 32), vshrn_n_u64(hi1, 32));
  uint32x4_t q2 = vcombine_u32(vshrn_n_u64(lo2, 32), vshrn_n_u64(hi2, 32));
  uint32x4_t q3 = vcombine_u32(vshrn_n_u64(lo3, 32), vshrn_n_u64(hi3, 32));
  uint32x4_t q4 = vcombine_u32(vshrn_n_u64(lo4, 32), vshrn_n_u64(hi4, 32));

  uint32x4_t qp1 = vmulq_u32(q1, p);
  uint32x4_t qp2 = vmulq_u32(q2, p);
  uint32x4_t qp3 = vmulq_u32(q3, p);
  uint32x4_t qp4 = vmulq_u32(q4, p);

  r1 = vsubq_u32(n1, qp1);
  r2 = vsubq_u32(n2, qp2);
  r3 = vsubq_u32(n3, qp3);
  r4 = vsubq_u32(n4, qp4);

  // Conditional subtraction
  r1 = vsubq_u32(r1, vandq_u32(vcgeq_u32(r1, p), p));
  r2 = vsubq_u32(r2, vandq_u32(vcgeq_u32(r2, p), p));
  r3 = vsubq_u32(r3, vandq_u32(vcgeq_u32(r3, p), p));
  r4 = vsubq_u32(r4, vandq_u32(vcgeq_u32(r4, p), p));
}

// === Strategy 2: Precomputed constant vectors ===
struct PrimeConstants {
  uint32x4_t p[16];
  uint32x4_t mu[16];

  PrimeConstants() {
    for (int i = 0; i < 8; ++i) {
      p[i] = vdupq_n_u32(SMALL_PRIMES[i]);
      mu[i] = vdupq_n_u32(SMALL_MU[i]);
      p[i+8] = vdupq_n_u32(EXT_PRIMES[i]);
      mu[i+8] = vdupq_n_u32(EXT_MU[i]);
    }
  }
};

static const PrimeConstants PRIME_CONSTS;

// === Strategy 3: 16-number processing with better memory pattern ===
__attribute__((flatten, always_inline)) inline
void filter16_u64_barrett16_ultra(const uint64_t* __restrict ptr,
                                  uint8_t*       __restrict out) {
  // Load 16 numbers as 8 NEON registers
  uint64x2_t a0 = vld1q_u64(ptr + 0);
  uint64x2_t a1 = vld1q_u64(ptr + 2);
  uint64x2_t a2 = vld1q_u64(ptr + 4);
  uint64x2_t a3 = vld1q_u64(ptr + 6);
  uint64x2_t a4 = vld1q_u64(ptr + 8);
  uint64x2_t a5 = vld1q_u64(ptr + 10);
  uint64x2_t a6 = vld1q_u64(ptr + 12);
  uint64x2_t a7 = vld1q_u64(ptr + 14);

  // Check if all fit in 32 bits
  uint64x2_t any01 = vorrq_u64(vshrq_n_u64(a0, 32), vshrq_n_u64(a1, 32));
  uint64x2_t any23 = vorrq_u64(vshrq_n_u64(a2, 32), vshrq_n_u64(a3, 32));
  uint64x2_t any45 = vorrq_u64(vshrq_n_u64(a4, 32), vshrq_n_u64(a5, 32));
  uint64x2_t any67 = vorrq_u64(vshrq_n_u64(a6, 32), vshrq_n_u64(a7, 32));
  uint64x2_t any_lo = vorrq_u64(any01, any23);
  uint64x2_t any_hi = vorrq_u64(any45, any67);
  uint64x2_t any = vorrq_u64(any_lo, any_hi);

  const bool all32 = ((vgetq_lane_u64(any,0) | vgetq_lane_u64(any,1)) == 0ULL);

  // Narrow to 32-bit
  uint32x4_t n1 = vcombine_u32(vmovn_u64(a0), vmovn_u64(a1));
  uint32x4_t n2 = vcombine_u32(vmovn_u64(a2), vmovn_u64(a3));
  uint32x4_t n3 = vcombine_u32(vmovn_u64(a4), vmovn_u64(a5));
  uint32x4_t n4 = vcombine_u32(vmovn_u64(a6), vmovn_u64(a7));

  // Process all 16 primes
  const uint32x4_t zero = vdupq_n_u32(0);
  uint32x4_t m1 = zero, m2 = zero, m3 = zero, m4 = zero;

  // Unroll with precomputed constants
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    uint32x4_t r1, r2, r3, r4;
    barrett_modq_u32_quad(n1, n2, n3, n4, PRIME_CONSTS.mu[i], PRIME_CONSTS.p[i],
                         r1, r2, r3, r4);

    uint32x4_t d1 = vceqq_u32(r1, zero);
    uint32x4_t d2 = vceqq_u32(r2, zero);
    uint32x4_t d3 = vceqq_u32(r3, zero);
    uint32x4_t d4 = vceqq_u32(r4, zero);

    // Exclude n==p
    d1 = vandq_u32(d1, vmvnq_u32(vceqq_u32(n1, PRIME_CONSTS.p[i])));
    d2 = vandq_u32(d2, vmvnq_u32(vceqq_u32(n2, PRIME_CONSTS.p[i])));
    d3 = vandq_u32(d3, vmvnq_u32(vceqq_u32(n3, PRIME_CONSTS.p[i])));
    d4 = vandq_u32(d4, vmvnq_u32(vceqq_u32(n4, PRIME_CONSTS.p[i])));

    m1 = vorrq_u32(m1, d1);
    m2 = vorrq_u32(m2, d2);
    m3 = vorrq_u32(m3, d3);
    m4 = vorrq_u32(m4, d4);
  }

  // Invert masks
  uint32x4_t sv1 = vceqq_u32(m1, zero);
  uint32x4_t sv2 = vceqq_u32(m2, zero);
  uint32x4_t sv3 = vceqq_u32(m3, zero);
  uint32x4_t sv4 = vceqq_u32(m4, zero);

  if (!all32) {
    // Apply lane enables
    uint32x4_t en1 = vceqq_u32(vcombine_u32(vmovn_u64(vshrq_n_u64(a0, 32)),
                                             vmovn_u64(vshrq_n_u64(a1, 32))), zero);
    uint32x4_t en2 = vceqq_u32(vcombine_u32(vmovn_u64(vshrq_n_u64(a2, 32)),
                                             vmovn_u64(vshrq_n_u64(a3, 32))), zero);
    uint32x4_t en3 = vceqq_u32(vcombine_u32(vmovn_u64(vshrq_n_u64(a4, 32)),
                                             vmovn_u64(vshrq_n_u64(a5, 32))), zero);
    uint32x4_t en4 = vceqq_u32(vcombine_u32(vmovn_u64(vshrq_n_u64(a6, 32)),
                                             vmovn_u64(vshrq_n_u64(a7, 32))), zero);
    sv1 = vandq_u32(sv1, en1);
    sv2 = vandq_u32(sv2, en2);
    sv3 = vandq_u32(sv3, en3);
    sv4 = vandq_u32(sv4, en4);
  }

  // Convert to bytes and store
  uint16x4_t s1_16 = vmovn_u32(sv1);
  uint16x4_t s2_16 = vmovn_u32(sv2);
  uint16x4_t s3_16 = vmovn_u32(sv3);
  uint16x4_t s4_16 = vmovn_u32(sv4);

  uint8x8_t s12 = vmovn_u16(vcombine_u16(s1_16, s2_16));
  uint8x8_t s34 = vmovn_u16(vcombine_u16(s3_16, s4_16));

  uint8x16_t result = vcombine_u8(vshr_n_u8(s12, 7), vshr_n_u8(s34, 7));
  vst1q_u8(out, result);
}

// === Strategy 4: Aligned memory access with hints ===
void filter_stream_u64_barrett16_ultra(const uint64_t* __restrict numbers,
                                       uint8_t*       __restrict out,
                                       size_t count) {
  // Assume aligned for better loads
  numbers = (const uint64_t*)__builtin_assume_aligned(numbers, 16);
  out = (uint8_t*)__builtin_assume_aligned(out, 16);

  size_t i = 0;

  // Process 32 at a time for maximum throughput
  for (; i + 32 <= count; i += 32) {
    __builtin_prefetch(numbers + i + 64, 0, 1);
    __builtin_prefetch(numbers + i + 80, 0, 1);

    filter16_u64_barrett16_ultra(numbers + i, out + i);
    filter16_u64_barrett16_ultra(numbers + i + 16, out + i + 16);
  }

  // Process 16 at a time
  for (; i + 16 <= count; i += 16) {
    filter16_u64_barrett16_ultra(numbers + i, out + i);
  }

  // Fall back to regular 8-at-a-time for remainder
  for (; i + 8 <= count; i += 8) {
    // Inline the 8-element version here
    uint64x2_t a0 = vld1q_u64(numbers + i + 0);
    uint64x2_t a1 = vld1q_u64(numbers + i + 2);
    uint64x2_t a2 = vld1q_u64(numbers + i + 4);
    uint64x2_t a3 = vld1q_u64(numbers + i + 6);

    uint32x4_t n1 = vcombine_u32(vmovn_u64(a0), vmovn_u64(a1));
    uint32x4_t n2 = vcombine_u32(vmovn_u64(a2), vmovn_u64(a3));

    const uint32x4_t zero = vdupq_n_u32(0);
    uint32x4_t m1 = zero, m2 = zero;

    for (int j = 0; j < 16; ++j) {
      uint32x4_t r1, r2;
      // Simple dual barrett
      uint64x2_t lo1 = vmull_u32(vget_low_u32(n1), vget_low_u32(PRIME_CONSTS.mu[j]));
      uint64x2_t hi1 = vmull_u32(vget_high_u32(n1), vget_high_u32(PRIME_CONSTS.mu[j]));
      uint64x2_t lo2 = vmull_u32(vget_low_u32(n2), vget_low_u32(PRIME_CONSTS.mu[j]));
      uint64x2_t hi2 = vmull_u32(vget_high_u32(n2), vget_high_u32(PRIME_CONSTS.mu[j]));

      uint32x4_t q1 = vcombine_u32(vshrn_n_u64(lo1, 32), vshrn_n_u64(hi1, 32));
      uint32x4_t q2 = vcombine_u32(vshrn_n_u64(lo2, 32), vshrn_n_u64(hi2, 32));

      r1 = vsubq_u32(n1, vmulq_u32(q1, PRIME_CONSTS.p[j]));
      r2 = vsubq_u32(n2, vmulq_u32(q2, PRIME_CONSTS.p[j]));

      r1 = vsubq_u32(r1, vandq_u32(vcgeq_u32(r1, PRIME_CONSTS.p[j]), PRIME_CONSTS.p[j]));
      r2 = vsubq_u32(r2, vandq_u32(vcgeq_u32(r2, PRIME_CONSTS.p[j]), PRIME_CONSTS.p[j]));

      uint32x4_t d1 = vceqq_u32(r1, zero);
      uint32x4_t d2 = vceqq_u32(r2, zero);

      d1 = vandq_u32(d1, vmvnq_u32(vceqq_u32(n1, PRIME_CONSTS.p[j])));
      d2 = vandq_u32(d2, vmvnq_u32(vceqq_u32(n2, PRIME_CONSTS.p[j])));

      m1 = vorrq_u32(m1, d1);
      m2 = vorrq_u32(m2, d2);
    }

    uint32x4_t sv1 = vceqq_u32(m1, zero);
    uint32x4_t sv2 = vceqq_u32(m2, zero);

    // Check for 64-bit values
    uint64x2_t h0 = vshrq_n_u64(a0, 32);
    uint64x2_t h1 = vshrq_n_u64(a1, 32);
    uint64x2_t h2 = vshrq_n_u64(a2, 32);
    uint64x2_t h3 = vshrq_n_u64(a3, 32);
    uint64x2_t any = vorrq_u64(vorrq_u64(h0, h1), vorrq_u64(h2, h3));
    if ((vgetq_lane_u64(any,0) | vgetq_lane_u64(any,1)) != 0ULL) {
      uint32x4_t en_lo = vceqq_u32(vcombine_u32(vmovn_u64(h0), vmovn_u64(h1)), zero);
      uint32x4_t en_hi = vceqq_u32(vcombine_u32(vmovn_u64(h2), vmovn_u64(h3)), zero);
      sv1 = vandq_u32(sv1, en_lo);
      sv2 = vandq_u32(sv2, en_hi);
    }

    // Convert to bytes
    uint16x4_t s1_16 = vmovn_u32(sv1);
    uint16x4_t s2_16 = vmovn_u32(sv2);
    uint8x8_t s8 = vmovn_u16(vcombine_u16(s1_16, s2_16));
    vst1_u8(out + i, vshr_n_u8(s8, 7));
  }

  // Scalar tail
  for (; i < count; ++i) {
    uint64_t v = numbers[i];
    if (v > 0xffffffffu) {
      out[i] = 0;
      continue;
    }
    uint32_t n = (uint32_t)v;
    uint8_t survive = 1;
    for (int j = 0; j < 8; ++j) {
      if (n != SMALL_PRIMES[j]) {
        uint64_t q = (uint64_t)n * SMALL_MU[j] >> 32;
        uint32_t r = n - (uint32_t)q * SMALL_PRIMES[j];
        if (r >= SMALL_PRIMES[j]) r -= SMALL_PRIMES[j];
        if (r == 0) { survive = 0; break; }
      }
      if (n != EXT_PRIMES[j]) {
        uint64_t q = (uint64_t)n * EXT_MU[j] >> 32;
        uint32_t r = n - (uint32_t)q * EXT_PRIMES[j];
        if (r >= EXT_PRIMES[j]) r -= EXT_PRIMES[j];
        if (r == 0) { survive = 0; break; }
      }
    }
    out[i] = survive;
  }
}

// === Strategy 5: Bitmap with 2-byte packing ===
__attribute__((always_inline)) inline
uint16_t bitpack16_from_u32_masks(uint32x4_t sv1, uint32x4_t sv2,
                                  uint32x4_t sv3, uint32x4_t sv4) {
  // Use SIMD to pack 16 lanes into 16 bits efficiently
  const uint8_t lo = movemask8_from_u32(sv1, sv2);
  const uint8_t hi = movemask8_from_u32(sv3, sv4);
  return (uint16_t)lo | ((uint16_t)hi << 8);
}

} // namespace neon_ultra
