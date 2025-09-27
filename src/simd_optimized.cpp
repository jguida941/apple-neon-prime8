#include "simd_fast.hpp"
#include "primes_tables.hpp"
#include <arm_neon.h>
#include <cstdint>
#include <cstddef>
#include <cstring>

namespace neon_optimized {

// === Fast modulo helpers for tail processing ===
__attribute__((always_inline)) inline
uint32_t fast_mod3(uint32_t n) {
    const uint32_t mu3 = 0xAAAAAAABu;  // ceil(2^33/3) >> 1
    uint32_t q = (uint64_t(n) * mu3) >> 33;
    return n - q * 3;
}

__attribute__((always_inline)) inline
uint32_t fast_mod5(uint32_t n) {
    const uint32_t mu5 = 0xCCCCCCCDu;  // ceil(2^34/5) >> 2
    uint32_t q = (uint64_t(n) * mu5) >> 34;
    return n - q * 5;
}

// === SIMD bitpack helper ===
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

// === Interleaved prime constants for better pipeline usage ===
struct InterleavedPrimeConstants {
    uint32x4_t p[16];
    uint32x4_t mu[16];

    InterleavedPrimeConstants() {
        // Interleave small and extended primes for better dual-issue
        for (int i = 0; i < 8; ++i) {
            p[i*2] = vdupq_n_u32(SMALL_PRIMES[i]);
            mu[i*2] = vdupq_n_u32(SMALL_MU[i]);
            p[i*2+1] = vdupq_n_u32(EXT_PRIMES[i]);
            mu[i*2+1] = vdupq_n_u32(EXT_MU[i]);
        }
    }
};

static const InterleavedPrimeConstants INTERLEAVED_PRIMES;

// === Quad Barrett with interleaved scheduling ===
__attribute__((always_inline)) inline
void barrett_modq_u32_quad_interleaved(
    uint32x4_t n1, uint32x4_t n2, uint32x4_t n3, uint32x4_t n4,
    uint32x4_t mu, uint32x4_t p,
    uint32x4_t& r1, uint32x4_t& r2, uint32x4_t& r3, uint32x4_t& r4) {

    // Start all multiplications early (ILP)
    uint64x2_t lo1 = vmull_u32(vget_low_u32(n1), vget_low_u32(mu));
    uint64x2_t lo2 = vmull_u32(vget_low_u32(n2), vget_low_u32(mu));
    uint64x2_t hi1 = vmull_u32(vget_high_u32(n1), vget_high_u32(mu));
    uint64x2_t hi2 = vmull_u32(vget_high_u32(n2), vget_high_u32(mu));

    uint64x2_t lo3 = vmull_u32(vget_low_u32(n3), vget_low_u32(mu));
    uint64x2_t lo4 = vmull_u32(vget_low_u32(n4), vget_low_u32(mu));
    uint64x2_t hi3 = vmull_u32(vget_high_u32(n3), vget_high_u32(mu));
    uint64x2_t hi4 = vmull_u32(vget_high_u32(n4), vget_high_u32(mu));

    // Extract quotients
    uint32x4_t q1 = vcombine_u32(vshrn_n_u64(lo1, 32), vshrn_n_u64(hi1, 32));
    uint32x4_t q2 = vcombine_u32(vshrn_n_u64(lo2, 32), vshrn_n_u64(hi2, 32));
    uint32x4_t q3 = vcombine_u32(vshrn_n_u64(lo3, 32), vshrn_n_u64(hi3, 32));
    uint32x4_t q4 = vcombine_u32(vshrn_n_u64(lo4, 32), vshrn_n_u64(hi4, 32));

    // Compute products
    uint32x4_t qp1 = vmulq_u32(q1, p);
    uint32x4_t qp2 = vmulq_u32(q2, p);
    uint32x4_t qp3 = vmulq_u32(q3, p);
    uint32x4_t qp4 = vmulq_u32(q4, p);

    // Compute remainders
    r1 = vsubq_u32(n1, qp1);
    r2 = vsubq_u32(n2, qp2);
    r3 = vsubq_u32(n3, qp3);
    r4 = vsubq_u32(n4, qp4);

    // Conditional correction
    r1 = vsubq_u32(r1, vandq_u32(vcgeq_u32(r1, p), p));
    r2 = vsubq_u32(r2, vandq_u32(vcgeq_u32(r2, p), p));
    r3 = vsubq_u32(r3, vandq_u32(vcgeq_u32(r3, p), p));
    r4 = vsubq_u32(r4, vandq_u32(vcgeq_u32(r4, p), p));
}

// === Optimized wheel-30 kernel ===
__attribute__((always_inline, flatten)) inline
uint16_t filter16_u64_wheel_optimized(const uint64_t* __restrict ptr) {
    // Assume aligned for better loads
    ptr = (const uint64_t*)__builtin_assume_aligned(ptr, 16);

    // Load with prefetch hint
    uint64x2_t a0 = vld1q_u64(ptr + 0);
    uint64x2_t a1 = vld1q_u64(ptr + 2);
    uint64x2_t a2 = vld1q_u64(ptr + 4);
    uint64x2_t a3 = vld1q_u64(ptr + 6);
    uint64x2_t a4 = vld1q_u64(ptr + 8);
    uint64x2_t a5 = vld1q_u64(ptr + 10);
    uint64x2_t a6 = vld1q_u64(ptr + 12);
    uint64x2_t a7 = vld1q_u64(ptr + 14);

    // Fast 32-bit check
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

    // Wheel-30 prefilter with Barrett MU30
    const uint32x4_t thirty = vdupq_n_u32(30);
    const uint32x4_t mu30 = vdupq_n_u32(2863311531u);

    // Apply wheel mask (optimized)
    auto apply_wheel = [&](uint32x4_t n) -> uint32x4_t {
        uint64x2_t lo = vmull_u32(vget_low_u32(n), vget_low_u32(mu30));
        uint64x2_t hi = vmull_u32(vget_high_u32(n), vget_high_u32(mu30));
        uint32x4_t q = vcombine_u32(vshrn_n_u64(lo, 32), vshrn_n_u64(hi, 32));
        uint32x4_t r = vsubq_u32(n, vmulq_u32(q, thirty));

        // Check coprime residues
        const uint32x4_t r1 = vdupq_n_u32(1);
        const uint32x4_t r7 = vdupq_n_u32(7);
        const uint32x4_t r11 = vdupq_n_u32(11);
        const uint32x4_t r13 = vdupq_n_u32(13);

        uint32x4_t mask = vorrq_u32(vceqq_u32(r, r1), vceqq_u32(r, r7));
        mask = vorrq_u32(mask, vceqq_u32(r, r11));
        mask = vorrq_u32(mask, vceqq_u32(r, r13));

        const uint32x4_t r17 = vdupq_n_u32(17);
        const uint32x4_t r19 = vdupq_n_u32(19);
        const uint32x4_t r23 = vdupq_n_u32(23);
        const uint32x4_t r29 = vdupq_n_u32(29);

        mask = vorrq_u32(mask, vceqq_u32(r, r17));
        mask = vorrq_u32(mask, vceqq_u32(r, r19));
        mask = vorrq_u32(mask, vceqq_u32(r, r23));
        mask = vorrq_u32(mask, vceqq_u32(r, r29));

        return mask;
    };

    uint32x4_t wheel1 = apply_wheel(n1);
    uint32x4_t wheel2 = apply_wheel(n2);
    uint32x4_t wheel3 = apply_wheel(n3);
    uint32x4_t wheel4 = apply_wheel(n4);

    if (!all32) {
        // Apply 64-bit masks
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

    // Fast exit if all fail wheel test
    if ((vmaxvq_u32(wheel1) | vmaxvq_u32(wheel2) |
         vmaxvq_u32(wheel3) | vmaxvq_u32(wheel4)) == 0) {
        return 0;
    }

    // Full Barrett with interleaved primes
    const uint32x4_t zero = vdupq_n_u32(0);
    uint32x4_t m1 = zero, m2 = zero, m3 = zero, m4 = zero;

    // Skip 2,3,5 since wheel handled them - start from index 3*2=6
    #pragma unroll 4
    for (int i = 6; i < 16; ++i) {
        uint32x4_t r1, r2, r3, r4;
        barrett_modq_u32_quad_interleaved(n1, n2, n3, n4,
                                          INTERLEAVED_PRIMES.mu[i],
                                          INTERLEAVED_PRIMES.p[i],
                                          r1, r2, r3, r4);

        // Mark composite lanes
        uint32x4_t d1 = vandq_u32(vceqq_u32(r1, zero), vmvnq_u32(vceqq_u32(n1, INTERLEAVED_PRIMES.p[i])));
        uint32x4_t d2 = vandq_u32(vceqq_u32(r2, zero), vmvnq_u32(vceqq_u32(n2, INTERLEAVED_PRIMES.p[i])));
        uint32x4_t d3 = vandq_u32(vceqq_u32(r3, zero), vmvnq_u32(vceqq_u32(n3, INTERLEAVED_PRIMES.p[i])));
        uint32x4_t d4 = vandq_u32(vceqq_u32(r4, zero), vmvnq_u32(vceqq_u32(n4, INTERLEAVED_PRIMES.p[i])));

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

    // Survivors
    uint32x4_t sv1 = vandq_u32(wheel1, vceqq_u32(m1, zero));
    uint32x4_t sv2 = vandq_u32(wheel2, vceqq_u32(m2, zero));
    uint32x4_t sv3 = vandq_u32(wheel3, vceqq_u32(m3, zero));
    uint32x4_t sv4 = vandq_u32(wheel4, vceqq_u32(m4, zero));

    // Pack to bitmap using SIMD
    return bitpack16_from_u32_masks(sv1, sv2, sv3, sv4);
}

// === Main optimized streaming function ===
void filter_stream_u64_wheel_optimized(const uint64_t* __restrict numbers,
                                       uint8_t*       __restrict bitmap,
                                       size_t count) {
    // Assume bitmap is 4-byte aligned for better stores
    bitmap = (uint8_t*)__builtin_assume_aligned(bitmap, 4);

    size_t i = 0;

    // Process 32 at a time with shorter prefetch distance
    for (; i + 32 <= count; i += 32) {
        __builtin_prefetch(numbers + i + 32, 0, 1);  // Shorter distance for wheel

        uint16_t b0 = filter16_u64_wheel_optimized(numbers + i);
        uint16_t b1 = filter16_u64_wheel_optimized(numbers + i + 16);
        uint32_t word = b0 | (uint32_t(b1) << 16);

        const size_t byte_off = i >> 3;
        std::memcpy(bitmap + byte_off, &word, sizeof(word));
    }

    // Process remaining 16
    if (i + 16 <= count) {
        uint16_t b = filter16_u64_wheel_optimized(numbers + i);
        const size_t byte_off = i >> 3;
        std::memcpy(bitmap + byte_off, &b, sizeof(b));
        i += 16;
    }

    // Optimized scalar tail with fast modulo
    if (i + 8 <= count) {
        const size_t base = i;
        uint8_t byte = 0;
        for (int bit = 0; bit < 8 && i < count; ++bit, ++i) {
            uint64_t n = numbers[i];
            if (n > 0xffffffffu) continue;

            uint32_t n32 = (uint32_t)n;

            // Fast wheel check using Barrett
            if ((n32 & 1) == 0 && n32 != 2) continue;
            if (fast_mod3(n32) == 0 && n32 != 3) continue;
            if (fast_mod5(n32) == 0 && n32 != 5) continue;

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

    // Final tail
    if (i < count) {
        const size_t base = i;
        uint8_t last = 0;
        unsigned mask = 0;
        for (unsigned bit = 0; i < count; ++bit, ++i) {
            uint64_t n = numbers[i];
            if (n > 0xffffffffu) continue;

            uint32_t n32 = (uint32_t)n;
            if ((n32 & 1) == 0 && n32 != 2) continue;
            if (fast_mod3(n32) == 0 && n32 != 3) continue;
            if (fast_mod5(n32) == 0 && n32 != 5) continue;

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

            if (survive) last |= 1 << bit;
            mask |= (1u << bit);
        }
        const size_t byte_off = base >> 3;
        bitmap[byte_off] = (bitmap[byte_off] & ~uint8_t(mask)) | (last & uint8_t(mask));
    }
}

} // namespace neon_optimized