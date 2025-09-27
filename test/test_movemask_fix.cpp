#include <arm_neon.h>
#include <iostream>
#include <cstdint>
#include <bitset>

uint8_t movemask8_from_u32_correct(uint32x4_t sv1, uint32x4_t sv2) {
  // The issue is in how we're extracting the bits
  // We want bit 0 from lane 0, bit 1 from lane 1, etc.

  // First narrow to get 0xFFFF or 0x0000 in u16
  uint16x4_t s1 = vmovn_u32(sv1);
  uint16x4_t s2 = vmovn_u32(sv2);

  // Combine and narrow to get 0xFF or 0x00 in u8
  uint8x8_t b = vmovn_u16(vcombine_u16(s1, s2));

  // Extract each byte and build the mask manually
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

int main() {
    // Test: 1101 0100
    uint32_t data[8] = {
        0xFFFFFFFF, 0xFFFFFFFF, 0, 0xFFFFFFFF,  // 1101
        0, 0xFFFFFFFF, 0, 0                     // 0100
    };

    uint32x4_t sv1 = vld1q_u32(data + 0);
    uint32x4_t sv2 = vld1q_u32(data + 4);

    std::cout << "Testing movemask for pattern: 11010100\n";
    std::cout << "Expected result: 0x2b (00101011)\n";

    uint8_t result = movemask8_from_u32_correct(sv1, sv2);

    std::cout << "Got result: 0x" << std::hex << (int)result << "\n";
    std::bitset<8> bits(result);
    std::cout << "Binary: " << bits << "\n";

    return 0;
}