#include <arm_neon.h>
#include <iostream>
#include <cstdint>

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

uint16_t bitpack16_from_u32_masks(uint32x4_t sv1, uint32x4_t sv2,
                                  uint32x4_t sv3, uint32x4_t sv4) {
  const uint8_t lo = movemask8_from_u32(sv1, sv2);
  const uint8_t hi = movemask8_from_u32(sv3, sv4);
  return (uint16_t)lo | ((uint16_t)hi << 8);
}

int main() {
    // Test with known mask pattern
    // Create masks: 1,1,0,1,0,1,0,0,0,1,0,1,0,0,0,1 = 0x8a2b
    uint32_t data[16] = {
        0xFFFFFFFF, 0xFFFFFFFF, 0, 0xFFFFFFFF,  // 1101
        0, 0xFFFFFFFF, 0, 0,                    // 0100
        0, 0xFFFFFFFF, 0, 0xFFFFFFFF,           // 0101
        0, 0, 0, 0xFFFFFFFF                     // 0001
    };

    uint32x4_t sv1 = vld1q_u32(data + 0);
    uint32x4_t sv2 = vld1q_u32(data + 4);
    uint32x4_t sv3 = vld1q_u32(data + 8);
    uint32x4_t sv4 = vld1q_u32(data + 12);

    uint16_t result = bitpack16_from_u32_masks(sv1, sv2, sv3, sv4);

    std::cout << "Test pattern: ";
    for (int i = 0; i < 16; i++) {
        std::cout << (data[i] ? "1" : "0");
    }
    std::cout << "\n";

    std::cout << "Packed result: 0x" << std::hex << result << "\n";
    std::cout << "Expected:      0x8a2b\n";
    std::cout << "Match: " << (result == 0x8a2b ? "YES" : "NO") << "\n";

    // Show as binary
    std::cout << "Binary: ";
    for (int i = 0; i < 16; i++) {
        std::cout << ((result >> i) & 1);
    }
    std::cout << "\n";

    return 0;
}