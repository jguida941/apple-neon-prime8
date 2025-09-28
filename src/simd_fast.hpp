// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Justin Guida
#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

namespace neon_fast {

void filter8_u64_barrett16(const uint64_t* __restrict ptr,
                           uint8_t*       __restrict out);

// Stream interface: processes count numbers in batches of 8.
// Tail (<8) is handled by a scalar cleanup that matches the same semantics.
void filter_stream_u64_barrett16(const uint64_t* __restrict numbers,
                                 uint8_t*       __restrict out,
                                 size_t count);

// 1-bit per lane streaming API
void filter_stream_u64_barrett16_bitmap(const uint64_t* __restrict numbers,
                                        uint8_t*       __restrict bitmap,
                                        size_t count);

} // namespace neon_fast

namespace neon_wheel {

// Wheel-30 prefiltered bitmap streaming API
void filter_stream_u64_wheel_bitmap(const uint64_t* __restrict numbers,
                                    uint8_t*       __restrict bitmap,
                                    size_t count);

// Byte output version
void filter_stream_u64_wheel(const uint64_t* __restrict numbers,
                             uint8_t*       __restrict out,
                             size_t count);

} // namespace neon_wheel

namespace neon_ultra {

// Ultra-optimized streaming API
void filter_stream_u64_barrett16_ultra(const uint64_t* __restrict numbers,
                                       uint8_t*       __restrict out,
                                       size_t count);

} // namespace neon_ultra

namespace neon_wheel210 {

// Wheel-210 (2×3×5×7) - 77.1% elimination
void filter_stream_u64_wheel210_bitmap(const uint64_t* __restrict numbers,
                                       uint8_t*       __restrict bitmap,
                                       size_t count);

} // namespace neon_wheel210

namespace neon_wheel210_efficient {

// Efficient Wheel-210 (Wheel-30 + mod 7) - 77.1% elimination
void filter_stream_u64_wheel210_efficient_bitmap(const uint64_t* __restrict numbers,
                                                 uint8_t*       __restrict bitmap,
                                                 size_t count);

} // namespace neon_wheel210_efficient
