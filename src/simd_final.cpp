#include "simd_fast.hpp"
#include "primes_tables.hpp"
#include <arm_neon.h>
#include <cstdint>
#include <cstddef>

namespace neon_final {

// === Strategy 1: Hoisted constant vectors (initialized once) ===
struct alignas(64) PrimeVectors {
  uint32x4_t p[16];
  uint32x4_t mu[16];

  PrimeVectors() {
    for (int i = 0; i < 8; ++i) {
      p[i]  = vdupq_n_u32(SMALL_PRIMES[i]);
      mu[i] = vdupq_n_u32(SMALL_MU[i]);
      p[i+8]  = vdupq_n_u32(EXT_PRIMES[i]);
      mu[i+8] = vdupq_n_u32(EXT_MU[i]);
    }
  }
};

static const PrimeVectors PVEC;

// === Strategy 2: Barrett with early-out ===
__attribute__((always_inline)) inline
void barrett_modq_u32_dual(uint32x4_t n1, uint32x4_t n2,
                          uint32x4_t mu, uint32x4_t p,
                          uint32x4_t& r1, uint32x4_t& r2) {
  uint64x2_t lo1 = vmull_u32(vget_low_u32(n1), vget_low_u32(mu));
  uint64x2_t hi1 = vmull_u32(vget_high_u32(n1), vget_high_u32(mu));
  uint64x2_t lo2 = vmull_u32(vget_low_u32(n2), vget_low_u32(mu));
  uint64x2_t hi2 = vmull_u32(vget_high_u32(n2), vget_high_u32(mu));

  uint32x4_t q1 = vcombine_u32(vshrn_n_u64(lo1, 32), vshrn_n_u64(hi1, 32));
  uint32x4_t q2 = vcombine_u32(vshrn_n_u64(lo2, 32), vshrn_n_u64(hi2, 32));

  uint32x4_t qp1 = vmulq_u32(q1, p);
  uint32x4_t qp2 = vmulq_u32(q2, p);

  r1 = vsubq_u32(n1, qp1);
  r2 = vsubq_u32(n2, qp2);

  r1 = vsubq_u32(r1, vandq_u32(vcgeq_u32(r1, p), p));
  r2 = vsubq_u32(r2, vandq_u32(vcgeq_u32(r2, p), p));
}

// === Strategy 3: Pack-level early-out divisibility check ===
__attribute__((always_inline)) inline
void divisible_mask_dual16_earlyout(uint32x4_t n1, uint32x4_t n2,
                                    uint32x4_t& m1, uint32x4_t& m2) {
  const uint32x4_t zero = vdupq_n_u32(0);
  const uint32x4_t all_ones = vdupq_n_u32(0xFFFFFFFF);

  m1 = zero;
  m2 = zero;

  // Track which lanes are still "alive" (not yet marked composite)
  uint32x4_t alive1 = all_ones;
  uint32x4_t alive2 = all_ones;

  // Check small primes first (most likely to eliminate composites)
  // Process in pairs and check for early-out every 2 primes

  // Primes 2, 3
  for (int i = 0; i < 2; ++i) {
    uint32x4_t r1, r2;
    barrett_modq_u32_dual(n1, n2, PVEC.mu[i], PVEC.p[i], r1, r2);

    uint32x4_t d1 = vandq_u32(vceqq_u32(r1, zero), vmvnq_u32(vceqq_u32(n1, PVEC.p[i])));
    uint32x4_t d2 = vandq_u32(vceqq_u32(r2, zero), vmvnq_u32(vceqq_u32(n2, PVEC.p[i])));

    d1 = vandq_u32(d1, alive1);
    d2 = vandq_u32(d2, alive2);

    m1 = vorrq_u32(m1, d1);
    m2 = vorrq_u32(m2, d2);

    alive1 = vandq_u32(alive1, vmvnq_u32(d1));
    alive2 = vandq_u32(alive2, vmvnq_u32(d2));
  }

  // Check if all lanes are dead
  if ((vmaxvq_u32(alive1) | vmaxvq_u32(alive2)) == 0) return;

  // Primes 5, 7
  for (int i = 2; i < 4; ++i) {
    uint32x4_t r1, r2;
    barrett_modq_u32_dual(n1, n2, PVEC.mu[i], PVEC.p[i], r1, r2);

    uint32x4_t d1 = vandq_u32(vceqq_u32(r1, zero), vmvnq_u32(vceqq_u32(n1, PVEC.p[i])));
    uint32x4_t d2 = vandq_u32(vceqq_u32(r2, zero), vmvnq_u32(vceqq_u32(n2, PVEC.p[i])));

    d1 = vandq_u32(d1, alive1);
    d2 = vandq_u32(d2, alive2);

    m1 = vorrq_u32(m1, d1);
    m2 = vorrq_u32(m2, d2);

    alive1 = vandq_u32(alive1, vmvnq_u32(d1));
    alive2 = vandq_u32(alive2, vmvnq_u32(d2));
  }

  if ((vmaxvq_u32(alive1) | vmaxvq_u32(alive2)) == 0) return;

  // Continue with remaining primes
  for (int i = 4; i < 16; ++i) {
    if (i == 8 || i == 12) {  // Check every 4 primes
      if ((vmaxvq_u32(alive1) | vmaxvq_u32(alive2)) == 0) return;
    }

    uint32x4_t r1, r2;
    barrett_modq_u32_dual(n1, n2, PVEC.mu[i], PVEC.p[i], r1, r2);

    uint32x4_t d1 = vandq_u32(vceqq_u32(r1, zero), vmvnq_u32(vceqq_u32(n1, PVEC.p[i])));
    uint32x4_t d2 = vandq_u32(vceqq_u32(r2, zero), vmvnq_u32(vceqq_u32(n2, PVEC.p[i])));

    d1 = vandq_u32(d1, alive1);
    d2 = vandq_u32(d2, alive2);

    m1 = vorrq_u32(m1, d1);
    m2 = vorrq_u32(m2, d2);

    alive1 = vandq_u32(alive1, vmvnq_u32(d1));
    alive2 = vandq_u32(alive2, vmvnq_u32(d2));
  }
}

// === Strategy 4: Wheel prefilter (mod 30 = 2×3×5) ===
__attribute__((always_inline)) inline
bool wheel30_prefilter(uint32x4_t n1, uint32x4_t n2) {
  // Residues coprime to 30: {1,7,11,13,17,19,23,29}
  // Quick check: if all n%30 are in {0,2,3,4,5,6,8,9,10,12,14,15,16,18,20,21,22,24,25,26,27,28}
  // then entire pack is composite

  const uint32x4_t thirty = vdupq_n_u32(30);
  const uint32x4_t mu30 = vdupq_n_u32(2863311531u); // ceil(2^32/30)

  // Compute n % 30 using Barrett
  uint64x2_t lo1 = vmull_u32(vget_low_u32(n1), vget_low_u32(mu30));
  uint64x2_t hi1 = vmull_u32(vget_high_u32(n1), vget_high_u32(mu30));
  uint64x2_t lo2 = vmull_u32(vget_low_u32(n2), vget_low_u32(mu30));
  uint64x2_t hi2 = vmull_u32(vget_high_u32(n2), vget_high_u32(mu30));

  uint32x4_t q1 = vcombine_u32(vshrn_n_u64(lo1, 32), vshrn_n_u64(hi1, 32));
  uint32x4_t q2 = vcombine_u32(vshrn_n_u64(lo2, 32), vshrn_n_u64(hi2, 32));

  uint32x4_t r1 = vsubq_u32(n1, vmulq_u32(q1, thirty));
  uint32x4_t r2 = vsubq_u32(n2, vmulq_u32(q2, thirty));

  // Check if any residue is coprime to 30
  // Coprime residues: 1,7,11,13,17,19,23,29
  const uint32x4_t one = vdupq_n_u32(1);
  const uint32x4_t seven = vdupq_n_u32(7);
  const uint32x4_t eleven = vdupq_n_u32(11);
  const uint32x4_t thirteen = vdupq_n_u32(13);

  uint32x4_t coprime1 = vceqq_u32(r1, one);
  coprime1 = vorrq_u32(coprime1, vceqq_u32(r1, seven));
  coprime1 = vorrq_u32(coprime1, vceqq_u32(r1, eleven));
  coprime1 = vorrq_u32(coprime1, vceqq_u32(r1, thirteen));

  uint32x4_t coprime2 = vceqq_u32(r2, one);
  coprime2 = vorrq_u32(coprime2, vceqq_u32(r2, seven));
  coprime2 = vorrq_u32(coprime2, vceqq_u32(r2, eleven));
  coprime2 = vorrq_u32(coprime2, vceqq_u32(r2, thirteen));

  // Continue checking other coprime residues
  const uint32x4_t seventeen = vdupq_n_u32(17);
  const uint32x4_t nineteen = vdupq_n_u32(19);
  const uint32x4_t twentythree = vdupq_n_u32(23);
  const uint32x4_t twentynine = vdupq_n_u32(29);

  coprime1 = vorrq_u32(coprime1, vceqq_u32(r1, seventeen));
  coprime1 = vorrq_u32(coprime1, vceqq_u32(r1, nineteen));
  coprime1 = vorrq_u32(coprime1, vceqq_u32(r1, twentythree));
  coprime1 = vorrq_u32(coprime1, vceqq_u32(r1, twentynine));

  coprime2 = vorrq_u32(coprime2, vceqq_u32(r2, seventeen));
  coprime2 = vorrq_u32(coprime2, vceqq_u32(r2, nineteen));
  coprime2 = vorrq_u32(coprime2, vceqq_u32(r2, twentythree));
  coprime2 = vorrq_u32(coprime2, vceqq_u32(r2, twentynine));

  // Return true if ANY lane might be prime (has coprime residue)
  return (vmaxvq_u32(coprime1) | vmaxvq_u32(coprime2)) != 0;
}

// === Main filter with all optimizations ===
__attribute__((always_inline)) inline
void filter8_u64_barrett16_final(const uint64_t* __restrict ptr,
                                 uint8_t*       __restrict out) {
  // Load 8×u64
  uint64x2_t a0 = vld1q_u64(ptr + 0);
  uint64x2_t a1 = vld1q_u64(ptr + 2);
  uint64x2_t a2 = vld1q_u64(ptr + 4);
  uint64x2_t a3 = vld1q_u64(ptr + 6);

  // Check 32-bit fit
  uint64x2_t h0 = vshrq_n_u64(a0, 32);
  uint64x2_t h1 = vshrq_n_u64(a1, 32);
  uint64x2_t h2 = vshrq_n_u64(a2, 32);
  uint64x2_t h3 = vshrq_n_u64(a3, 32);
  uint64x2_t any = vorrq_u64(vorrq_u64(h0, h1), vorrq_u64(h2, h3));
  const bool all32 = ((vgetq_lane_u64(any,0) | vgetq_lane_u64(any,1)) == 0ULL);

  // Narrow to 32-bit
  uint32x4_t n1 = vcombine_u32(vmovn_u64(a0), vmovn_u64(a1));
  uint32x4_t n2 = vcombine_u32(vmovn_u64(a2), vmovn_u64(a3));

  // Wheel prefilter - skip full Barrett if all composite by mod 30
  if (all32 && !wheel30_prefilter(n1, n2)) {
    // All lanes divisible by 2, 3, or 5
    vst1_u8(out, vdup_n_u8(0));
    return;
  }

  // Full Barrett with early-out
  uint32x4_t m1, m2;
  divisible_mask_dual16_earlyout(n1, n2, m1, m2);

  // Invert masks (0 = composite, 0xFFFFFFFF = survives)
  const uint32x4_t zero = vdupq_n_u32(0);
  uint32x4_t sv1 = vceqq_u32(m1, zero);
  uint32x4_t sv2 = vceqq_u32(m2, zero);

  // Apply lane enables for 64-bit values
  if (!all32) {
    uint32x4_t en_lo = vceqq_u32(vcombine_u32(vmovn_u64(h0), vmovn_u64(h1)), zero);
    uint32x4_t en_hi = vceqq_u32(vcombine_u32(vmovn_u64(h2), vmovn_u64(h3)), zero);
    sv1 = vandq_u32(sv1, en_lo);
    sv2 = vandq_u32(sv2, en_hi);
  }

  // Convert to bytes and store
  uint16x4_t s1_16 = vmovn_u32(sv1);
  uint16x4_t s2_16 = vmovn_u32(sv2);
  uint8x8_t s8 = vmovn_u16(vcombine_u16(s1_16, s2_16));
  vst1_u8(out, vshr_n_u8(s8, 7));
}

// === Strategy 5: Software pipelined 32-element processing ===
void filter32_u64_barrett16_pipelined(const uint64_t* __restrict ptr,
                                      uint8_t*       __restrict out) {
  // Process 32 elements with software pipelining
  // Load next batch while processing current batch

  // Load first 8
  uint64x2_t a0 = vld1q_u64(ptr + 0);
  uint64x2_t a1 = vld1q_u64(ptr + 2);
  uint64x2_t a2 = vld1q_u64(ptr + 4);
  uint64x2_t a3 = vld1q_u64(ptr + 6);

  // Start processing first 8
  uint32x4_t n1_a = vcombine_u32(vmovn_u64(a0), vmovn_u64(a1));
  uint32x4_t n2_a = vcombine_u32(vmovn_u64(a2), vmovn_u64(a3));

  // Load second 8 while first is processing
  uint64x2_t b0 = vld1q_u64(ptr + 8);
  uint64x2_t b1 = vld1q_u64(ptr + 10);
  uint64x2_t b2 = vld1q_u64(ptr + 12);
  uint64x2_t b3 = vld1q_u64(ptr + 14);

  // Process first 8
  uint32x4_t m1_a, m2_a;
  divisible_mask_dual16_earlyout(n1_a, n2_a, m1_a, m2_a);

  // Start second 8
  uint32x4_t n1_b = vcombine_u32(vmovn_u64(b0), vmovn_u64(b1));
  uint32x4_t n2_b = vcombine_u32(vmovn_u64(b2), vmovn_u64(b3));

  // Load third 8
  uint64x2_t c0 = vld1q_u64(ptr + 16);
  uint64x2_t c1 = vld1q_u64(ptr + 18);
  uint64x2_t c2 = vld1q_u64(ptr + 20);
  uint64x2_t c3 = vld1q_u64(ptr + 22);

  // Process second 8
  uint32x4_t m1_b, m2_b;
  divisible_mask_dual16_earlyout(n1_b, n2_b, m1_b, m2_b);

  // Start third 8
  uint32x4_t n1_c = vcombine_u32(vmovn_u64(c0), vmovn_u64(c1));
  uint32x4_t n2_c = vcombine_u32(vmovn_u64(c2), vmovn_u64(c3));

  // Load fourth 8
  uint64x2_t d0 = vld1q_u64(ptr + 24);
  uint64x2_t d1 = vld1q_u64(ptr + 26);
  uint64x2_t d2 = vld1q_u64(ptr + 28);
  uint64x2_t d3 = vld1q_u64(ptr + 30);

  // Process third 8
  uint32x4_t m1_c, m2_c;
  divisible_mask_dual16_earlyout(n1_c, n2_c, m1_c, m2_c);

  // Process fourth 8
  uint32x4_t n1_d = vcombine_u32(vmovn_u64(d0), vmovn_u64(d1));
  uint32x4_t n2_d = vcombine_u32(vmovn_u64(d2), vmovn_u64(d3));
  uint32x4_t m1_d, m2_d;
  divisible_mask_dual16_earlyout(n1_d, n2_d, m1_d, m2_d);

  // Convert all results to bytes and store
  const uint32x4_t zero = vdupq_n_u32(0);

  // First 8
  {
    uint32x4_t sv1 = vceqq_u32(m1_a, zero);
    uint32x4_t sv2 = vceqq_u32(m2_a, zero);
    uint64x2_t h0 = vshrq_n_u64(a0, 32);
    uint64x2_t h1 = vshrq_n_u64(a1, 32);
    uint64x2_t h2 = vshrq_n_u64(a2, 32);
    uint64x2_t h3 = vshrq_n_u64(a3, 32);
    uint64x2_t any = vorrq_u64(vorrq_u64(h0, h1), vorrq_u64(h2, h3));
    if ((vgetq_lane_u64(any,0) | vgetq_lane_u64(any,1)) != 0) {
      uint32x4_t en_lo = vceqq_u32(vcombine_u32(vmovn_u64(h0), vmovn_u64(h1)), zero);
      uint32x4_t en_hi = vceqq_u32(vcombine_u32(vmovn_u64(h2), vmovn_u64(h3)), zero);
      sv1 = vandq_u32(sv1, en_lo);
      sv2 = vandq_u32(sv2, en_hi);
    }
    uint8x8_t s8 = vmovn_u16(vcombine_u16(vmovn_u32(sv1), vmovn_u32(sv2)));
    vst1_u8(out + 0, vshr_n_u8(s8, 7));
  }

  // Second 8
  {
    uint32x4_t sv1 = vceqq_u32(m1_b, zero);
    uint32x4_t sv2 = vceqq_u32(m2_b, zero);
    uint64x2_t h0 = vshrq_n_u64(b0, 32);
    uint64x2_t h1 = vshrq_n_u64(b1, 32);
    uint64x2_t h2 = vshrq_n_u64(b2, 32);
    uint64x2_t h3 = vshrq_n_u64(b3, 32);
    uint64x2_t any = vorrq_u64(vorrq_u64(h0, h1), vorrq_u64(h2, h3));
    if ((vgetq_lane_u64(any,0) | vgetq_lane_u64(any,1)) != 0) {
      uint32x4_t en_lo = vceqq_u32(vcombine_u32(vmovn_u64(h0), vmovn_u64(h1)), zero);
      uint32x4_t en_hi = vceqq_u32(vcombine_u32(vmovn_u64(h2), vmovn_u64(h3)), zero);
      sv1 = vandq_u32(sv1, en_lo);
      sv2 = vandq_u32(sv2, en_hi);
    }
    uint8x8_t s8 = vmovn_u16(vcombine_u16(vmovn_u32(sv1), vmovn_u32(sv2)));
    vst1_u8(out + 8, vshr_n_u8(s8, 7));
  }

  // Third and fourth 8 similar...
  {
    uint32x4_t sv1 = vceqq_u32(m1_c, zero);
    uint32x4_t sv2 = vceqq_u32(m2_c, zero);
    uint64x2_t h0 = vshrq_n_u64(c0, 32);
    uint64x2_t h1 = vshrq_n_u64(c1, 32);
    uint64x2_t h2 = vshrq_n_u64(c2, 32);
    uint64x2_t h3 = vshrq_n_u64(c3, 32);
    uint64x2_t any = vorrq_u64(vorrq_u64(h0, h1), vorrq_u64(h2, h3));
    if ((vgetq_lane_u64(any,0) | vgetq_lane_u64(any,1)) != 0) {
      uint32x4_t en_lo = vceqq_u32(vcombine_u32(vmovn_u64(h0), vmovn_u64(h1)), zero);
      uint32x4_t en_hi = vceqq_u32(vcombine_u32(vmovn_u64(h2), vmovn_u64(h3)), zero);
      sv1 = vandq_u32(sv1, en_lo);
      sv2 = vandq_u32(sv2, en_hi);
    }
    uint8x8_t s8 = vmovn_u16(vcombine_u16(vmovn_u32(sv1), vmovn_u32(sv2)));
    vst1_u8(out + 16, vshr_n_u8(s8, 7));
  }

  {
    uint32x4_t sv1 = vceqq_u32(m1_d, zero);
    uint32x4_t sv2 = vceqq_u32(m2_d, zero);
    uint64x2_t h0 = vshrq_n_u64(d0, 32);
    uint64x2_t h1 = vshrq_n_u64(d1, 32);
    uint64x2_t h2 = vshrq_n_u64(d2, 32);
    uint64x2_t h3 = vshrq_n_u64(d3, 32);
    uint64x2_t any = vorrq_u64(vorrq_u64(h0, h1), vorrq_u64(h2, h3));
    if ((vgetq_lane_u64(any,0) | vgetq_lane_u64(any,1)) != 0) {
      uint32x4_t en_lo = vceqq_u32(vcombine_u32(vmovn_u64(h0), vmovn_u64(h1)), zero);
      uint32x4_t en_hi = vceqq_u32(vcombine_u32(vmovn_u64(h2), vmovn_u64(h3)), zero);
      sv1 = vandq_u32(sv1, en_lo);
      sv2 = vandq_u32(sv2, en_hi);
    }
    uint8x8_t s8 = vmovn_u16(vcombine_u16(vmovn_u32(sv1), vmovn_u32(sv2)));
    vst1_u8(out + 24, vshr_n_u8(s8, 7));
  }
}

// === Main streaming function with all optimizations ===
void filter_stream_u64_barrett16_final(const uint64_t* __restrict numbers,
                                       uint8_t*       __restrict out,
                                       size_t count) {
  // Alignment hints
  numbers = (const uint64_t*)__builtin_assume_aligned(numbers, 16);
  out = (uint8_t*)__builtin_assume_aligned(out, 8);

  size_t i = 0;

  // Process 32 at a time with software pipelining
  for (; i + 32 <= count; i += 32) {
    __builtin_prefetch(numbers + i + 64, 0, 1);  // Prefetch 2 cache lines ahead
    __builtin_prefetch(numbers + i + 72, 0, 1);
    filter32_u64_barrett16_pipelined(numbers + i, out + i);
  }

  // Process remaining 8 at a time
  for (; i + 8 <= count; i += 8) {
    filter8_u64_barrett16_final(numbers + i, out + i);
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

    // Unrolled scalar checks
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (n != SMALL_PRIMES[j]) {
        uint64_t q = (uint64_t)n * SMALL_MU[j] >> 32;
        uint32_t r = n - (uint32_t)q * SMALL_PRIMES[j];
        if (r >= SMALL_PRIMES[j]) r -= SMALL_PRIMES[j];
        if (r == 0) { survive = 0; break; }
      }
    }

    if (survive) {
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (n != EXT_PRIMES[j]) {
          uint64_t q = (uint64_t)n * EXT_MU[j] >> 32;
          uint32_t r = n - (uint32_t)q * EXT_PRIMES[j];
          if (r >= EXT_PRIMES[j]) r -= EXT_PRIMES[j];
          if (r == 0) { survive = 0; break; }
        }
      }
    }

    out[i] = survive;
  }
}

} // namespace neon_final