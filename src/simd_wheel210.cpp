// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Justin Guida
#include "simd_fast.hpp"
#include "primes_tables.hpp"
#include <arm_neon.h>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>

namespace neon_wheel210 {

// === Wheel-210 (2×3×5×7) prefilter ===
// Only 48 residues mod 210 can be prime - 77.1% elimination!
// This is a 4.3% improvement over Wheel-30's 73.3% elimination

// The 48 coprime residues mod 210
static const uint8_t WHEEL210_RESIDUES[48] = {
    1, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67,
    71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 121, 127, 131,
    137, 139, 143, 149, 151, 157, 163, 167, 169, 173, 179, 181, 187,
    191, 193, 197, 199, 209
};

// Precomputed lookup table: is residue coprime to 210?
static uint8_t WHEEL210_COPRIME[210] = {0};

// Initialize lookup table
__attribute__((constructor))
static void init_wheel210() {
    for (int i = 0; i < 48; i++) {
        WHEEL210_COPRIME[WHEEL210_RESIDUES[i]] = 1;
    }
}

// Barrett constant for mod 210
static const uint32_t MU210 = 20456360u; // ceil(2^32/210)

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

// === Wheel-210 prefilter for 16 lanes ===
__attribute__((always_inline)) inline
uint32x4_t wheel210_mask(uint32x4_t n) {
    const uint32x4_t two_ten = vdupq_n_u32(210);
    const uint32x4_t mu = vdupq_n_u32(MU210);

    // Compute n % 210 using Barrett
    uint64x2_t lo = vmull_u32(vget_low_u32(n), vget_low_u32(mu));
    uint64x2_t hi = vmull_u32(vget_high_u32(n), vget_high_u32(mu));
    uint32x4_t q = vcombine_u32(vshrn_n_u64(lo, 32), vshrn_n_u64(hi, 32));
    uint32x4_t r = vsubq_u32(n, vmulq_u32(q, two_ten));

    // For small mod 210, we can check all 48 residues efficiently
    // by grouping them into ranges and using SIMD comparisons
    uint32x4_t mask = vdupq_n_u32(0);

    // Check ranges of residues (optimized for SIMD)
    // Group 1: 1-47
    for (int i = 0; i < 12; i++) {
        uint32x4_t res = vdupq_n_u32(WHEEL210_RESIDUES[i]);
        mask = vorrq_u32(mask, vceqq_u32(r, res));
    }

    // Group 2: 53-109
    for (int i = 12; i < 26; i++) {
        uint32x4_t res = vdupq_n_u32(WHEEL210_RESIDUES[i]);
        mask = vorrq_u32(mask, vceqq_u32(r, res));
    }

    // Group 3: 113-169
    for (int i = 26; i < 38; i++) {
        uint32x4_t res = vdupq_n_u32(WHEEL210_RESIDUES[i]);
        mask = vorrq_u32(mask, vceqq_u32(r, res));
    }

    // Group 4: 173-209
    for (int i = 38; i < 48; i++) {
        uint32x4_t res = vdupq_n_u32(WHEEL210_RESIDUES[i]);
        mask = vorrq_u32(mask, vceqq_u32(r, res));
    }

    return mask; // 0xFFFFFFFF if possibly prime, 0 if definitely composite
}

// === Process 16 numbers with Wheel-210 + Barrett ===
__attribute__((always_inline, flatten)) inline
uint16_t filter16_u64_wheel210_bitmap(const uint64_t* __restrict ptr) {
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

    // Check if all fit in 32 bits
    uint64x2_t h01 = vorrq_u64(vshrq_n_u64(a0, 32), vshrq_n_u64(a1, 32));
    uint64x2_t h23 = vorrq_u64(vshrq_n_u64(a2, 32), vshrq_n_u64(a3, 32));
    uint64x2_t h45 = vorrq_u64(vshrq_n_u64(a4, 32), vshrq_n_u64(a5, 32));
    uint64x2_t h67 = vorrq_u64(vshrq_n_u64(a6, 32), vshrq_n_u64(a7, 32));
    uint64x2_t any = vorrq_u64(vorrq_u64(h01, h23), vorrq_u64(h45, h67));
    const bool all32 = ((vgetq_lane_u64(any,0) | vgetq_lane_u64(any,1)) == 0ULL);

    // Narrow to 32-bit
    uint32x4_t n1 = vcombine_u32(vmovn_u64(a0), vmovn_u64(a1));
    uint32x4_t n2 = vcombine_u32(vmovn_u64(a2), vmovn_u64(a3));
    uint32x4_t n3 = vcombine_u32(vmovn_u64(a4), vmovn_u64(a5));
    uint32x4_t n4 = vcombine_u32(vmovn_u64(a6), vmovn_u64(a7));

    // Apply Wheel-210 prefilter (77.1% elimination!)
    uint32x4_t wheel1 = wheel210_mask(n1);
    uint32x4_t wheel2 = wheel210_mask(n2);
    uint32x4_t wheel3 = wheel210_mask(n3);
    uint32x4_t wheel4 = wheel210_mask(n4);

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

    // Full Barrett reduction for survivors
    const uint32x4_t zero = vdupq_n_u32(0);
    uint32x4_t m1 = zero, m2 = zero, m3 = zero, m4 = zero;

    // Skip 2,3,5,7 since wheel handled them - start from prime 11 (index 4)
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

// === Main streaming function with Wheel-210 ===
void filter_stream_u64_wheel210_bitmap(const uint64_t* __restrict numbers,
                                       uint8_t*       __restrict bitmap,
                                       size_t count) {
    bitmap = (uint8_t*)__builtin_assume_aligned(bitmap, 4);

    size_t i = 0;

    // Process 32 at a time
    for (; i + 32 <= count; i += 32) {
        __builtin_prefetch(numbers + i + 32, 0, 1);

        uint16_t b0 = filter16_u64_wheel210_bitmap(numbers + i);
        uint16_t b1 = filter16_u64_wheel210_bitmap(numbers + i + 16);
        uint32_t word = b0 | (uint32_t(b1) << 16);

        const size_t byte_off = i >> 3;
        std::memcpy(bitmap + byte_off, &word, sizeof(word));
    }

    // Process remaining 16
    if (i + 16 <= count) {
        uint16_t b = filter16_u64_wheel210_bitmap(numbers + i);
        const size_t byte_off = i >> 3;
        std::memcpy(bitmap + byte_off, &b, sizeof(b));
        i += 16;
    }

    // Scalar tail with fast modulo
    if (i + 8 <= count) {
        const size_t base = i;
        uint8_t byte = 0;
        for (int bit = 0; bit < 8 && i < count; ++bit, ++i) {
            uint64_t n = numbers[i];
            if (n > 0xffffffffu) continue;

            uint32_t n32 = (uint32_t)n;

            // Quick Wheel-210 check
            uint32_t r210 = n32 - (uint64_t(n32) * MU210 >> 32) * 210;
            if (r210 >= 210) r210 -= 210;
            if (!WHEEL210_COPRIME[r210]) continue;

            // Full Barrett check (skip 2,3,5,7)
            bool survive = true;
            for (int k = 4; k < 8; ++k) {
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

    // Final tail
    if (i < count) {
        const size_t base = i;
        uint8_t last = 0;
        unsigned mask = 0;
        for (unsigned bit = 0; i < count; ++bit, ++i) {
            uint64_t n = numbers[i];
            if (n > 0xffffffffu) continue;

            uint32_t n32 = (uint32_t)n;
            uint32_t r210 = n32 - (uint64_t(n32) * MU210 >> 32) * 210;
            if (r210 >= 210) r210 -= 210;
            if (!WHEEL210_COPRIME[r210]) continue;

            bool survive = true;
            for (int k = 4; k < 8; ++k) {
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

            if (survive) last |= 1 << bit;
            mask |= (1u << bit);
        }
        const size_t byte_off = base >> 3;
        bitmap[byte_off] = (bitmap[byte_off] & ~uint8_t(mask)) | (last & uint8_t(mask));
    }
}

} // namespace neon_wheel210
