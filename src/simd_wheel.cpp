// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Justin Guida
#include "simd_fast.hpp"
#include "primes_tables.hpp"
#include <arm_neon.h>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>

namespace neon_wheel {

// === SIMD bitpack helper for ARM NEON ===
__attribute__((always_inline)) inline
uint8_t movemask8_from_u32(uint32x4_t sv1, uint32x4_t sv2) {
  uint16x4_t s1 = vmovn_u32(sv1);
  uint16x4_t s2 = vmovn_u32(sv2);
  uint8x8_t  b  = vmovn_u16(vcombine_u16(s1, s2)); // 0xFF/0x00 per lane

  // Extract each byte's MSB and build mask
  uint8_t mask = 0;
  mask |= (vget_lane_u8(b, 0) & 0x80) ? 0x01 : 0;
  mask |= (vget_lane_u8(b, 1) & 0x80) ? 0x02 : 0;
  mask |= (vget_lane_u8(b, 2) & 0x80) ? 0x04 : 0;
  mask |= (vget_lane_u8(b, 3) & 0x80) ? 0x08 : 0;
  mask |= (vget_lane_u8(b, 4) & 0x80) ? 0x10 : 0;
  mask |= (vget_lane_u8(b, 5) & 0x80) ? 0x20 : 0;
  mask |= (vget_lane_u8(b, 6) & 0x80) ? 0x40 : 0;
  mask |= (vget_lane_u8(b, 7) & 0x80) ? 0x80 : 0;
  return mask;
}

__attribute__((always_inline)) inline
uint16_t bitpack16_from_u32_masks(uint32x4_t sv1, uint32x4_t sv2,
                                  uint32x4_t sv3, uint32x4_t sv4) {
  const uint8_t lo = movemask8_from_u32(sv1, sv2);
  const uint8_t hi = movemask8_from_u32(sv3, sv4);
  return (uint16_t)lo | ((uint16_t)hi << 8);
}

// === Wheel-30 (2×3×5) prefilter tables ===
// Only 8 residues mod 30 can be prime: {1,7,11,13,17,19,23,29}
// This eliminates 22/30 = 73.3% of numbers before Barrett
static const uint8_t WHEEL30_COPRIME[30] = {
  0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,1
};

// Barrett constant for mod 30
static const uint32_t MU30 = 143165576u; // floor(2^32 / 30)

// === Quad Barrett reduction (from Ultra) ===
__attribute__((always_inline)) inline
void barrett_modq_u32_quad(uint32x4_t n1, uint32x4_t n2, uint32x4_t n3, uint32x4_t n4,
                           uint32x4_t mu, uint32x4_t p,
                           uint32x4_t& r1, uint32x4_t& r2, uint32x4_t& r3, uint32x4_t& r4) {
  // Multiply all 4 vectors with mu
  uint64x2_t lo1 = vmull_u32(vget_low_u32(n1), vget_low_u32(mu));
  uint64x2_t hi1 = vmull_u32(vget_high_u32(n1), vget_high_u32(mu));
  uint64x2_t lo2 = vmull_u32(vget_low_u32(n2), vget_low_u32(mu));
  uint64x2_t hi2 = vmull_u32(vget_high_u32(n2), vget_high_u32(mu));
  uint64x2_t lo3 = vmull_u32(vget_low_u32(n3), vget_low_u32(mu));
  uint64x2_t hi3 = vmull_u32(vget_high_u32(n3), vget_high_u32(mu));
  uint64x2_t lo4 = vmull_u32(vget_low_u32(n4), vget_low_u32(mu));
  uint64x2_t hi4 = vmull_u32(vget_high_u32(n4), vget_high_u32(mu));

  // Extract quotients
  uint32x4_t q1 = vcombine_u32(vshrn_n_u64(lo1, 32), vshrn_n_u64(hi1, 32));
  uint32x4_t q2 = vcombine_u32(vshrn_n_u64(lo2, 32), vshrn_n_u64(hi2, 32));
  uint32x4_t q3 = vcombine_u32(vshrn_n_u64(lo3, 32), vshrn_n_u64(hi3, 32));
  uint32x4_t q4 = vcombine_u32(vshrn_n_u64(lo4, 32), vshrn_n_u64(hi4, 32));

  // Compute remainders
  r1 = vsubq_u32(n1, vmulq_u32(q1, p));
  r2 = vsubq_u32(n2, vmulq_u32(q2, p));
  r3 = vsubq_u32(n3, vmulq_u32(q3, p));
  r4 = vsubq_u32(n4, vmulq_u32(q4, p));

  // Conditional subtraction for exact modulo
  r1 = vsubq_u32(r1, vandq_u32(vcgeq_u32(r1, p), p));
  r2 = vsubq_u32(r2, vandq_u32(vcgeq_u32(r2, p), p));
  r3 = vsubq_u32(r3, vandq_u32(vcgeq_u32(r3, p), p));
  r4 = vsubq_u32(r4, vandq_u32(vcgeq_u32(r4, p), p));
}

// === Wheel-30 prefilter for 16 lanes ===
__attribute__((always_inline)) inline
uint32x4_t wheel30_mask(uint32x4_t n) {
  // Compute n % 30 using Barrett
  const uint32x4_t thirty = vdupq_n_u32(30);
  const uint32x4_t mu30 = vdupq_n_u32(MU30);

  uint64x2_t lo = vmull_u32(vget_low_u32(n), vget_low_u32(mu30));
  uint64x2_t hi = vmull_u32(vget_high_u32(n), vget_high_u32(mu30));
  uint32x4_t q = vcombine_u32(vshrn_n_u64(lo, 32), vshrn_n_u64(hi, 32));
  uint32x4_t r = vsubq_u32(n, vmulq_u32(q, thirty));
  r = vsubq_u32(r, vandq_u32(vcgeq_u32(r, thirty), thirty)); // if (r>=30) r-=30

  // Check coprime residues: 1,7,11,13,17,19,23,29
  const uint32x4_t r1  = vdupq_n_u32(1);
  const uint32x4_t r7  = vdupq_n_u32(7);
  const uint32x4_t r11 = vdupq_n_u32(11);
  const uint32x4_t r13 = vdupq_n_u32(13);
  const uint32x4_t r17 = vdupq_n_u32(17);
  const uint32x4_t r19 = vdupq_n_u32(19);
  const uint32x4_t r23 = vdupq_n_u32(23);
  const uint32x4_t r29 = vdupq_n_u32(29);

  uint32x4_t mask = vceqq_u32(r, r1);
  mask = vorrq_u32(mask, vceqq_u32(r, r7));
  mask = vorrq_u32(mask, vceqq_u32(r, r11));
  mask = vorrq_u32(mask, vceqq_u32(r, r13));
  mask = vorrq_u32(mask, vceqq_u32(r, r17));
  mask = vorrq_u32(mask, vceqq_u32(r, r19));
  mask = vorrq_u32(mask, vceqq_u32(r, r23));
  mask = vorrq_u32(mask, vceqq_u32(r, r29));

  return mask; // 0xFFFFFFFF if possibly prime, 0 if definitely composite
}

// === Process 16 numbers with wheel prefilter + quad Barrett ===
__attribute__((always_inline, flatten)) inline
uint16_t filter16_u64_wheel_bitmap(const uint64_t* __restrict ptr) {
  // Load 16×u64 as 8 NEON registers (aligned if possible)
  ptr = (const uint64_t*)__builtin_assume_aligned(ptr, 16);

  uint64x2_t a0 = vld1q_u64(ptr + 0);
  uint64x2_t a1 = vld1q_u64(ptr + 2);
  uint64x2_t a2 = vld1q_u64(ptr + 4);
  uint64x2_t a3 = vld1q_u64(ptr + 6);
  uint64x2_t a4 = vld1q_u64(ptr + 8);
  uint64x2_t a5 = vld1q_u64(ptr + 10);
  uint64x2_t a6 = vld1q_u64(ptr + 12);
  uint64x2_t a7 = vld1q_u64(ptr + 14);

  // Check if all fit in 32 bits
  uint64x2_t h01 = vorrq_u64(vshrq_n_u64(a0, 32), vshrq_n_u64(a1, 32));
  uint64x2_t h23 = vorrq_u64(vshrq_n_u64(a2, 32), vshrq_n_u64(a3, 32));
  uint64x2_t h45 = vorrq_u64(vshrq_n_u64(a4, 32), vshrq_n_u64(a5, 32));
  uint64x2_t h67 = vorrq_u64(vshrq_n_u64(a6, 32), vshrq_n_u64(a7, 32));
  uint64x2_t any = vorrq_u64(vorrq_u64(h01, h23), vorrq_u64(h45, h67));
  const bool all32 = ((vgetq_lane_u64(any,0) | vgetq_lane_u64(any,1)) == 0ULL);

  // Narrow to 32-bit (4 vectors of 4 lanes each)
  uint32x4_t n1 = vcombine_u32(vmovn_u64(a0), vmovn_u64(a1));
  uint32x4_t n2 = vcombine_u32(vmovn_u64(a2), vmovn_u64(a3));
  uint32x4_t n3 = vcombine_u32(vmovn_u64(a4), vmovn_u64(a5));
  uint32x4_t n4 = vcombine_u32(vmovn_u64(a6), vmovn_u64(a7));

  // Apply wheel-30 prefilter
  uint32x4_t wheel1 = wheel30_mask(n1);
  uint32x4_t wheel2 = wheel30_mask(n2);
  uint32x4_t wheel3 = wheel30_mask(n3);
  uint32x4_t wheel4 = wheel30_mask(n4);

  // Special case: primes 2, 3, 5 must always pass
  const uint32x4_t two = vdupq_n_u32(2);
  const uint32x4_t three = vdupq_n_u32(3);
  const uint32x4_t five = vdupq_n_u32(5);
  wheel1 = vorrq_u32(wheel1, vceqq_u32(n1, two));
  wheel1 = vorrq_u32(wheel1, vceqq_u32(n1, three));
  wheel1 = vorrq_u32(wheel1, vceqq_u32(n1, five));
  wheel2 = vorrq_u32(wheel2, vceqq_u32(n2, two));
  wheel2 = vorrq_u32(wheel2, vceqq_u32(n2, three));
  wheel2 = vorrq_u32(wheel2, vceqq_u32(n2, five));
  wheel3 = vorrq_u32(wheel3, vceqq_u32(n3, two));
  wheel3 = vorrq_u32(wheel3, vceqq_u32(n3, three));
  wheel3 = vorrq_u32(wheel3, vceqq_u32(n3, five));
  wheel4 = vorrq_u32(wheel4, vceqq_u32(n4, two));
  wheel4 = vorrq_u32(wheel4, vceqq_u32(n4, three));
  wheel4 = vorrq_u32(wheel4, vceqq_u32(n4, five));

  // If all lanes fail wheel test, return 0 (all composite)
  if (!all32) {
    // Need to apply 64-bit masks
    uint32x4_t en1 = vceqq_u32(vcombine_u32(vmovn_u64(vshrq_n_u64(a0,32)),
                                              vmovn_u64(vshrq_n_u64(a1,32))), vdupq_n_u32(0));
    uint32x4_t en2 = vceqq_u32(vcombine_u32(vmovn_u64(vshrq_n_u64(a2,32)),
                                              vmovn_u64(vshrq_n_u64(a3,32))), vdupq_n_u32(0));
    uint32x4_t en3 = vceqq_u32(vcombine_u32(vmovn_u64(vshrq_n_u64(a4,32)),
                                              vmovn_u64(vshrq_n_u64(a5,32))), vdupq_n_u32(0));
    uint32x4_t en4 = vceqq_u32(vcombine_u32(vmovn_u64(vshrq_n_u64(a6,32)),
                                              vmovn_u64(vshrq_n_u64(a7,32))), vdupq_n_u32(0));
    wheel1 = vandq_u32(wheel1, en1);
    wheel2 = vandq_u32(wheel2, en2);
    wheel3 = vandq_u32(wheel3, en3);
    wheel4 = vandq_u32(wheel4, en4);
  }

  // Quick check: if no lanes pass wheel (after special cases), all are composite
  if ((vmaxvq_u32(wheel1) | vmaxvq_u32(wheel2) |
       vmaxvq_u32(wheel3) | vmaxvq_u32(wheel4)) == 0) {
    return 0;
  }

  // Full Barrett reduction for lanes that passed wheel
  const uint32x4_t zero = vdupq_n_u32(0);
  uint32x4_t m1 = zero, m2 = zero, m3 = zero, m4 = zero;

  // Test against remaining primes (skip 2,3,5 since wheel handled them)
  // Start from prime 7 (index 3)
  for (int i = 3; i < 8; ++i) {
    const uint32x4_t p = vdupq_n_u32(SMALL_PRIMES[i]);
    const uint32x4_t mu = vdupq_n_u32(SMALL_MU[i]);

    uint32x4_t r1, r2, r3, r4;
    barrett_modq_u32_quad(n1, n2, n3, n4, mu, p, r1, r2, r3, r4);

    // Mark composite lanes (r==0 and n!=p)
    uint32x4_t d1 = vandq_u32(vceqq_u32(r1, zero), vmvnq_u32(vceqq_u32(n1, p)));
    uint32x4_t d2 = vandq_u32(vceqq_u32(r2, zero), vmvnq_u32(vceqq_u32(n2, p)));
    uint32x4_t d3 = vandq_u32(vceqq_u32(r3, zero), vmvnq_u32(vceqq_u32(n3, p)));
    uint32x4_t d4 = vandq_u32(vceqq_u32(r4, zero), vmvnq_u32(vceqq_u32(n4, p)));

    // Only test lanes that passed wheel
    d1 = vandq_u32(d1, wheel1);
    d2 = vandq_u32(d2, wheel2);
    d3 = vandq_u32(d3, wheel3);
    d4 = vandq_u32(d4, wheel4);

    m1 = vorrq_u32(m1, d1);
    m2 = vorrq_u32(m2, d2);
    m3 = vorrq_u32(m3, d3);
    m4 = vorrq_u32(m4, d4);
  }

  // Continue with extended primes
  for (int i = 0; i < 8; ++i) {
    const uint32x4_t p = vdupq_n_u32(EXT_PRIMES[i]);
    const uint32x4_t mu = vdupq_n_u32(EXT_MU[i]);

    uint32x4_t r1, r2, r3, r4;
    barrett_modq_u32_quad(n1, n2, n3, n4, mu, p, r1, r2, r3, r4);

    uint32x4_t d1 = vandq_u32(vceqq_u32(r1, zero), vmvnq_u32(vceqq_u32(n1, p)));
    uint32x4_t d2 = vandq_u32(vceqq_u32(r2, zero), vmvnq_u32(vceqq_u32(n2, p)));
    uint32x4_t d3 = vandq_u32(vceqq_u32(r3, zero), vmvnq_u32(vceqq_u32(n3, p)));
    uint32x4_t d4 = vandq_u32(vceqq_u32(r4, zero), vmvnq_u32(vceqq_u32(n4, p)));

    d1 = vandq_u32(d1, wheel1);
    d2 = vandq_u32(d2, wheel2);
    d3 = vandq_u32(d3, wheel3);
    d4 = vandq_u32(d4, wheel4);

    m1 = vorrq_u32(m1, d1);
    m2 = vorrq_u32(m2, d2);
    m3 = vorrq_u32(m3, d3);
    m4 = vorrq_u32(m4, d4);
  }

  // Survivors = passed wheel AND not marked composite
  uint32x4_t sv1 = vandq_u32(wheel1, vceqq_u32(m1, zero));
  uint32x4_t sv2 = vandq_u32(wheel2, vceqq_u32(m2, zero));
  uint32x4_t sv3 = vandq_u32(wheel3, vceqq_u32(m3, zero));
  uint32x4_t sv4 = vandq_u32(wheel4, vceqq_u32(m4, zero));

  // Pack 16 lanes into 16-bit bitmap using SIMD
  return bitpack16_from_u32_masks(sv1, sv2, sv3, sv4);
}

// === Main streaming function with wheel+bitmap ===
void filter_stream_u64_wheel_bitmap(const uint64_t* __restrict numbers,
                                    uint8_t*       __restrict bitmap,
                                    size_t count) {
  size_t i = 0;

  // Process 32 at a time (2×16) for better throughput
  for (; i + 32 <= count; i += 32) {
    __builtin_prefetch(numbers + i + 64, 0, 1);  // 2 cache lines ahead

    uint16_t b0 = filter16_u64_wheel_bitmap(numbers + i);
    uint16_t b1 = filter16_u64_wheel_bitmap(numbers + i + 16);
    uint32_t packed = b0 | (uint32_t(b1) << 16);

    // Fix: Use memcpy to avoid aliasing issues
    std::memcpy(bitmap + (i >> 3), &packed, 4);
  }

  // Process remaining 16
  if (i + 16 <= count) {
    uint16_t bits = filter16_u64_wheel_bitmap(numbers + i);
    std::memcpy(bitmap + (i >> 3), &bits, 2);
    i += 16;
  }

  // Process remaining 8
  if (i + 8 <= count) {
    const size_t base = i;
    uint8_t byte = 0;
    for (int bit = 0; bit < 8 && i < count; ++bit, ++i) {
      uint64_t n = numbers[i];
      if (n > 0xffffffffu) continue;

      // Quick wheel check
      uint32_t n32 = (uint32_t)n;
      if (n32 == 2 || n32 == 3 || n32 == 5) {
        byte |= 1 << bit;
        continue;
      }
      if (n32 % 2 == 0) continue;
      if (n32 % 3 == 0) continue;
      if (n32 % 5 == 0) continue;

      // Full Barrett check
      bool survive = true;
      for (int k = 3; k < 8; ++k) {
        if (n32 != SMALL_PRIMES[k]) {
          uint64_t q = (uint64_t)n32 * SMALL_MU[k] >> 32;
          uint32_t r = n32 - (uint32_t)q * SMALL_PRIMES[k];
          if (r >= SMALL_PRIMES[k]) r -= SMALL_PRIMES[k];
          if (r == 0) { survive = false; break; }
        }
      }

      if (survive) {
        for (int k = 0; k < 8; ++k) {
          if (n32 != EXT_PRIMES[k]) {
            uint64_t q = (uint64_t)n32 * EXT_MU[k] >> 32;
            uint32_t r = n32 - (uint32_t)q * EXT_PRIMES[k];
            if (r >= EXT_PRIMES[k]) r -= EXT_PRIMES[k];
            if (r == 0) { survive = false; break; }
          }
        }
      }

      if (survive) byte |= 1 << bit;
    }
    bitmap[base >> 3] = byte;
  }

  // Final tail bits
  if (i < count) {
    const size_t base = i;
    uint8_t last = 0;
    for (unsigned bit = 0; i < count; ++bit, ++i) {
      uint64_t n = numbers[i];
      if (n > 0xffffffffu) continue;

      uint32_t n32 = (uint32_t)n;
      if (n32 == 2 || n32 == 3 || n32 == 5) {
        last |= (1u << bit);
        continue;
      }
      if (n32 % 2 == 0) continue;
      if (n32 % 3 == 0) continue;
      if (n32 % 5 == 0) continue;

      bool survive = true;
      for (int k = 3; k < 8; ++k) {
        if (n32 != SMALL_PRIMES[k]) {
          uint64_t q = (uint64_t)n32 * SMALL_MU[k] >> 32;
          uint32_t r = n32 - (uint32_t)q * SMALL_PRIMES[k];
          if (r >= SMALL_PRIMES[k]) r -= SMALL_PRIMES[k];
          if (r == 0) { survive = false; break; }
        }
      }

      if (survive) {
        for (int k = 0; k < 8; ++k) {
          if (n32 != EXT_PRIMES[k]) {
            uint64_t q = (uint64_t)n32 * EXT_MU[k] >> 32;
            uint32_t r = n32 - (uint32_t)q * EXT_PRIMES[k];
            if (r >= EXT_PRIMES[k]) r -= EXT_PRIMES[k];
            if (r == 0) { survive = false; break; }
          }
        }
      }

      if (survive) last |= (1u << bit);
    }
    bitmap[base >> 3] = last;
  }
}

// === Byte output version (for compatibility) ===
void filter_stream_u64_wheel(const uint64_t* __restrict numbers,
                             uint8_t*       __restrict out,
                             size_t count) {
  // Use bitmap internally, then expand
  size_t bitmap_size = (count + 7) / 8;
  std::vector<uint8_t> bitmap(bitmap_size);

  filter_stream_u64_wheel_bitmap(numbers, bitmap.data(), count);

  // Expand bitmap to bytes
  for (size_t i = 0; i < count; ++i) {
    out[i] = (bitmap[i/8] >> (i%8)) & 1;
  }
}

} // namespace neon_wheel
