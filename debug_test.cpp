#include <cstdio>
#include <cstdint>
#include <vector>
#include "src/simd_fast.hpp"

int main() {
  std::vector<uint64_t> test_nums = {
    2, 3, 4, 5, 1295391061, 7, 8, 9,  // First 8
    10, 11, 12, 13, 14, 15, 16, 17    // Second 8
  };

  std::vector<uint8_t> byte_out(16);
  std::vector<uint8_t> bitmap_out(2);

  neon_fast::filter_stream_u64_barrett16(test_nums.data(), byte_out.data(), 16);
  neon_fast::filter_stream_u64_barrett16_bitmap(test_nums.data(), bitmap_out.data(), 16);

  printf("Test values and results:\n");
  for (size_t i = 0; i < 16; ++i) {
    uint8_t bit = (bitmap_out[i / 8] >> (i % 8)) & 1;
    printf("  [%2zu] %12llu: byte=%u, bit=%u %s\n",
           i, (unsigned long long)test_nums[i],
           byte_out[i], bit,
           byte_out[i] == bit ? "OK" : "MISMATCH");
  }

  printf("\nBitmap bytes: 0x%02x 0x%02x\n", bitmap_out[0], bitmap_out[1]);

  return 0;
}