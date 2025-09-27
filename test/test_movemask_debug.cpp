#include <arm_neon.h>
#include <iostream>
#include <cstdint>
#include <bitset>

uint8_t movemask8_from_u32_debug(uint32x4_t sv1, uint32x4_t sv2) {
  std::cout << "Input sv1: ";
  for (int i = 0; i < 4; i++) {
    std::cout << std::hex << vgetq_lane_u32(sv1, i) << " ";
  }
  std::cout << "\nInput sv2: ";
  for (int i = 0; i < 4; i++) {
    std::cout << std::hex << vgetq_lane_u32(sv2, i) << " ";
  }
  std::cout << "\n";

  uint16x4_t s1 = vmovn_u32(sv1);
  uint16x4_t s2 = vmovn_u32(sv2);

  std::cout << "After narrow to u16, s1: ";
  for (int i = 0; i < 4; i++) {
    std::cout << std::hex << vget_lane_u16(s1, i) << " ";
  }
  std::cout << "\n";

  uint8x8_t b = vmovn_u16(vcombine_u16(s1, s2));

  std::cout << "After narrow to u8, b: ";
  for (int i = 0; i < 8; i++) {
    std::cout << std::hex << (int)vget_lane_u8(b, i) << " ";
  }
  std::cout << "\n";

  // Problem: vmovn will give 0xFF for non-zero, not necessarily 0xFF
  // We need to ensure we have 0xFF or 0x00
  // Actually vmovn_u32 with 0xFFFFFFFF gives 0xFFFF
  // vmovn_u16 with 0xFFFF gives 0xFF, with 0x0000 gives 0x00

  const uint8x8_t w = {1,2,4,8,16,32,64,128};
  uint8x8_t shifted = vshr_n_u8(b, 7); // Get MSB of each byte

  std::cout << "After shift>>7: ";
  for (int i = 0; i < 8; i++) {
    std::cout << (int)vget_lane_u8(shifted, i) << " ";
  }
  std::cout << "\n";

  uint8x8_t t = vand_u8(shifted, w);

  std::cout << "After AND with weights: ";
  for (int i = 0; i < 8; i++) {
    std::cout << (int)vget_lane_u8(t, i) << " ";
  }
  std::cout << "\n";

  t = vpadd_u8(t, t);
  std::cout << "After padd 1: ";
  for (int i = 0; i < 8; i++) {
    std::cout << (int)vget_lane_u8(t, i) << " ";
  }
  std::cout << "\n";

  t = vpadd_u8(t, t);
  std::cout << "After padd 2: ";
  for (int i = 0; i < 8; i++) {
    std::cout << (int)vget_lane_u8(t, i) << " ";
  }
  std::cout << "\n";

  t = vpadd_u8(t, t);
  std::cout << "After padd 3: ";
  for (int i = 0; i < 8; i++) {
    std::cout << (int)vget_lane_u8(t, i) << " ";
  }
  std::cout << "\n";

  uint8_t result = vget_lane_u8(t, 0);
  std::cout << "Final result: " << (int)result << " = ";
  std::bitset<8> bits(result);
  std::cout << bits << "\n";

  return result;
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
    std::cout << "Expected result: 0x2b (00101011 reversed)\n\n";

    uint8_t result = movemask8_from_u32_debug(sv1, sv2);

    return 0;
}