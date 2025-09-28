// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Justin Guida
#include "simd_fast.hpp"
#include "primes_tables.hpp"
#include <arm_neon.h>
#include <cstdint>
#include <cstddef>
#include <cstring>

namespace neon_wheel210_efficient {

// === Efficient Wheel-210 = Wheel-30 + mod 7 check ===
// This avoids 48 residue checks and uses only one extra Barrett reduction

// Barrett constants (FIXED values)
static const uint32_t MU30 = 143165577u;   // ceil(2^32/30)
static const uint32_t MU7  = 613566757u;   // ceil(2^32/7)

// === SIMD bitpack helpers ===
__attribute__((always_inline)) inline
uint8_t movemask8_from_u32(uint32x4_t sv1, uint32x4_t sv2) {
    uint16x4_t s1 = vmovn_u32(sv1);
    uint16x4_t s2 = vmovn_u32(sv2);
    uint8x8_t b = vmovn_u16(vcombine_u16(s1, s2));
    const uint8x8_t w = {1,2,4,8,16,32,64,128};
    uint8x8_t t = vand_u8(vshr_n_u8(b, 7), w);
    t = vpadd_u8(t, t); t = vpadd_u8(t, t); t = vpadd_u8(t, t);
    return vget_lane_u8(t, 0);
}

__attribute__((always_inline)) inline
uint16_t bitpack16_from_u32_masks(uint32x4_t sv1, uint32x4_t sv2,
                                  uint32x4_t sv3, uint32x4_t sv4) {
    const uint8_t lo = movemask8_from_u32(sv1, sv2);
    const uint8_t hi = movemask8_from_u32(sv3, sv4);
    return (uint16_t)lo | ((uint16_t)hi << 8);
}

// === Barrett modulo reduction ===
__attribute__((always_inline)) inline
uint32x4_t barrett_mod_u32(uint32x4_t n, uint32x4_t mu, uint32x4_t p) {
    uint64x2_t lo = vmull_u32(vget_low_u32(n),  vget_low_u32(mu));
    uint64x2_t hi = vmull_u32(vget_high_u32(n), vget_high_u32(mu));
    uint32x4_t q = vcombine_u32(vshrn_n_u64(lo, 32), vshrn_n_u64(hi, 32));
    uint32x4_t r = vsubq_u32(n, vmulq_u32(q, p));
    // One corrective subtract
    return vsubq_u32(r, vandq_u32(vcgeq_u32(r, p), p));
}

// === Wheel-30 prefilter (vectorized) ===
__attribute__((always_inline)) inline
uint32x4_t wheel30_pass(uint32x4_t n) {
    const uint32x4_t thirty = vdupq_n_u32(30);
    const uint32x4_t mu30 = vdupq_n_u32(MU30);
    uint32x4_t r = barrett_mod_u32(n, mu30, thirty);

    // Keep residues {1,7,11,13,17,19,23,29}
    uint32x4_t m = vceqq_u32(r, vdupq_n_u32(1));
    m = vorrq_u32(m, vceqq_u32(r, vdupq_n_u32(7)));
    m = vorrq_u32(m, vceqq_u32(r, vdupq_n_u32(11)));
    m = vorrq_u32(m, vceqq_u32(r, vdupq_n_u32(13)));
    m = vorrq_u32(m, vceqq_u32(r, vdupq_n_u32(17)));
    m = vorrq_u32(m, vceqq_u32(r, vdupq_n_u32(19)));
    m = vorrq_u32(m, vceqq_u32(r, vdupq_n_u32(23)));
    m = vorrq_u32(m, vceqq_u32(r, vdupq_n_u32(29)));
    return m; // 0xFFFFFFFF = passed wheel30
}

// === Wheel-210 = Wheel-30 AND (n % 7 != 0 or n == 7) ===
__attribute__((always_inline)) inline
uint32x4_t wheel210_pass(uint32x4_t n, uint32x4_t mask_w30) {
    const uint32x4_t seven = vdupq_n_u32(7);
    const uint32x4_t mu7 = vdupq_n_u32(MU7);
    uint32x4_t r7 = barrett_mod_u32(n, mu7, seven);

    // Candidate if (r7 != 0) OR (n == 7)
    uint32x4_t ok7 = vorrq_u32(vmvnq_u32(vceqq_u32(r7, vdupq_n_u32(0))),
                                vceqq_u32(n, seven));
    return vandq_u32(mask_w30, ok7);
}

// === Quad Barrett reduction ===
__attribute__((always_inline)) inline
void barrett_modq_u32_quad(uint32x4_t n1, uint32x4_t n2, uint32x4_t n3, uint32x4_t n4,
                           uint32x4_t mu, uint32x4_t p,
                           uint32x4_t& r1, uint32x4_t& r2, uint32x4_t& r3, uint32x4_t& r4) {
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

    r1 = vsubq_u32(n1, vmulq_u32(q1, p));
    r2 = vsubq_u32(n2, vmulq_u32(q2, p));
    r3 = vsubq_u32(n3, vmulq_u32(q3, p));
    r4 = vsubq_u32(n4, vmulq_u32(q4, p));

    r1 = vsubq_u32(r1, vandq_u32(vcgeq_u32(r1, p), p));
    r2 = vsubq_u32(r2, vandq_u32(vcgeq_u32(r2, p), p));
    r3 = vsubq_u32(r3, vandq_u32(vcgeq_u32(r3, p), p));
    r4 = vsubq_u32(r4, vandq_u32(vcgeq_u32(r4, p), p));
}

// === Process 16 numbers with efficient Wheel-210 ===
__attribute__((always_inline, flatten)) inline
uint16_t filter16_u64_wheel210_efficient(const uint64_t* __restrict ptr) {
    ptr = (const uint64_t*)__builtin_assume_aligned(ptr, 16);

    // Load 16 numbers
    uint64x2_t a0 = vld1q_u64(ptr + 0);
    uint64x2_t a1 = vld1q_u64(ptr + 2);
    uint64x2_t a2 = vld1q_u64(ptr + 4);
    uint64x2_t a3 = vld1q_u64(ptr + 6);
    uint64x2_t a4 = vld1q_u64(ptr + 8);
    uint64x2_t a5 = vld1q_u64(ptr + 10);
    uint64x2_t a6 = vld1q_u64(ptr + 12);
    uint64x2_t a7 = vld1q_u64(ptr + 14);

    // Narrow to 32-bit
    uint32x4_t n1 = vcombine_u32(vmovn_u64(a0), vmovn_u64(a1));
    uint32x4_t n2 = vcombine_u32(vmovn_u64(a2), vmovn_u64(a3));
    uint32x4_t n3 = vcombine_u32(vmovn_u64(a4), vmovn_u64(a5));
    uint32x4_t n4 = vcombine_u32(vmovn_u64(a6), vmovn_u64(a7));

    // Check if all fit in 32 bits
    uint64x2_t h01 = vorrq_u64(vshrq_n_u64(a0, 32), vshrq_n_u64(a1, 32));
    uint64x2_t h23 = vorrq_u64(vshrq_n_u64(a2, 32), vshrq_n_u64(a3, 32));
    uint64x2_t h45 = vorrq_u64(vshrq_n_u64(a4, 32), vshrq_n_u64(a5, 32));
    uint64x2_t h67 = vorrq_u64(vshrq_n_u64(a6, 32), vshrq_n_u64(a7, 32));
    uint64x2_t any = vorrq_u64(vorrq_u64(h01, h23), vorrq_u64(h45, h67));
    const bool all32 = ((vgetq_lane_u64(any,0) | vgetq_lane_u64(any,1)) == 0ULL);

    // Efficient Wheel-210: first Wheel-30, then mod 7 check
    uint32x4_t w30_1 = wheel30_pass(n1);
    uint32x4_t w30_2 = wheel30_pass(n2);
    uint32x4_t w30_3 = wheel30_pass(n3);
    uint32x4_t w30_4 = wheel30_pass(n4);

    uint32x4_t wheel1 = wheel210_pass(n1, w30_1);
    uint32x4_t wheel2 = wheel210_pass(n2, w30_2);
    uint32x4_t wheel3 = wheel210_pass(n3, w30_3);
    uint32x4_t wheel4 = wheel210_pass(n4, w30_4);

    if (!all32) {
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

    // Quick exit if all fail wheel test
    if ((vmaxvq_u32(wheel1) | vmaxvq_u32(wheel2) |
         vmaxvq_u32(wheel3) | vmaxvq_u32(wheel4)) == 0) {
        return 0;
    }

    // Full Barrett reduction for survivors (skip 2,3,5,7 - handled by wheel)
    const uint32x4_t zero = vdupq_n_u32(0);
    uint32x4_t m1 = zero, m2 = zero, m3 = zero, m4 = zero;

    // Start from prime 11 (index 4)
    for (int i = 4; i < 8; ++i) {
        const uint32x4_t p = vdupq_n_u32(SMALL_PRIMES[i]);
        const uint32x4_t mu = vdupq_n_u32(SMALL_MU[i]);

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

    // Survivors
    uint32x4_t sv1 = vandq_u32(wheel1, vceqq_u32(m1, zero));
    uint32x4_t sv2 = vandq_u32(wheel2, vceqq_u32(m2, zero));
    uint32x4_t sv3 = vandq_u32(wheel3, vceqq_u32(m3, zero));
    uint32x4_t sv4 = vandq_u32(wheel4, vceqq_u32(m4, zero));

    return bitpack16_from_u32_masks(sv1, sv2, sv3, sv4);
}

// === Main streaming function ===
void filter_stream_u64_wheel210_efficient_bitmap(const uint64_t* __restrict numbers,
                                                 uint8_t*       __restrict bitmap,
                                                 size_t count) {
    bitmap = (uint8_t*)__builtin_assume_aligned(bitmap, 4);

    size_t i = 0;

    // Process 32 at a time
    for (; i + 32 <= count; i += 32) {
        __builtin_prefetch(numbers + i + 32, 0, 1);

        uint16_t b0 = filter16_u64_wheel210_efficient(numbers + i);
        uint16_t b1 = filter16_u64_wheel210_efficient(numbers + i + 16);
        uint32_t word = b0 | (uint32_t(b1) << 16);

        const size_t byte_off = i >> 3;
        std::memcpy(bitmap + byte_off, &word, sizeof(word));
    }

    // Process remaining 16
    if (i + 16 <= count) {
        uint16_t b = filter16_u64_wheel210_efficient(numbers + i);
        const size_t byte_off = i >> 3;
        std::memcpy(bitmap + byte_off, &b, sizeof(b));
        i += 16;
    }

    // Scalar tail
    for (; i < count; ++i) {
        uint64_t n = numbers[i];
        bool survive = false;

        if (n <= 0xffffffffu) {
            uint32_t n32 = (uint32_t)n;

            // Wheel-30 check
            uint32_t r30 = n32 - (uint64_t(n32) * MU30 >> 32) * 30;
            if (r30 >= 30) r30 -= 30;

            bool wheel30 = (r30 == 1) || (r30 == 7) || (r30 == 11) || (r30 == 13) ||
                           (r30 == 17) || (r30 == 19) || (r30 == 23) || (r30 == 29);

            if (wheel30) {
                // Additional mod 7 check for Wheel-210
                uint32_t r7 = n32 - (uint64_t(n32) * MU7 >> 32) * 7;
                if (r7 >= 7) r7 -= 7;
                bool wheel210 = (r7 != 0) || (n32 == 7);

                if (wheel210) {
                    // Full Barrett check (skip 2,3,5,7)
                    survive = true;
                    for (int k = 4; k < 8 && survive; ++k) {
                        if (n32 != SMALL_PRIMES[k]) {
                            uint64_t q = (uint64_t)n32 * SMALL_MU[k] >> 32;
                            uint32_t r = n32 - (uint32_t)q * SMALL_PRIMES[k];
                            if (r >= SMALL_PRIMES[k]) r -= SMALL_PRIMES[k];
                            if (r == 0) survive = false;
                        }
                    }

                    for (int k = 0; k < 8 && survive; ++k) {
                        if (n32 != EXT_PRIMES[k]) {
                            uint64_t q = (uint64_t)n32 * EXT_MU[k] >> 32;
                            uint32_t r = n32 - (uint32_t)q * EXT_PRIMES[k];
                            if (r >= EXT_PRIMES[k]) r -= EXT_PRIMES[k];
                            if (r == 0) survive = false;
                        }
                    }
                }
            }
        }

        // Set bit
        size_t byte_idx = i >> 3;
        size_t bit_idx = i & 7;
        if (survive) {
            bitmap[byte_idx] |= (1u << bit_idx);
        } else {
            bitmap[byte_idx] &= ~(1u << bit_idx);
        }
    }
}

} // namespace neon_wheel210_efficient
