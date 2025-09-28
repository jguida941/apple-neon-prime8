// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Justin Guida
#pragma once
#include <cstdint>

// These are defined as static constexpr to be available in any file that includes this header.
alignas(16) static constexpr uint32_t SMALL_PRIMES[8] = {2,3,5,7,11,13,17,19};
alignas(16) static constexpr uint32_t EXT_PRIMES[8]   = {23,29,31,37,41,43,47,53};
alignas(16) static constexpr uint32_t SMALL_MU[8] = {
  2147483648u, 1431655765u,  858993459u,  613566756u,
   390451572u,  330382099u,  252645135u,  226050910u
};
alignas(16) static constexpr uint32_t EXT_MU[8] = {
  186737708u, 148102320u, 138547332u, 116080197u,
  104755299u,  99882960u,  91382282u,   81037118u
};
