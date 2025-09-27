#include <cstdio>
#include <vector>
#include "src/simd_fast.hpp"

int main() {
  printf("Testing basic functionality...\n");

  std::vector<uint64_t> test = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
  std::vector<uint8_t> out(16);

  printf("Running original version...\n");
  neon_fast::filter_stream_u64_barrett16(test.data(), out.data(), 16);

  printf("Results: ");
  for (int i = 0; i < 16; ++i) {
    printf("%d ", out[i]);
  }
  printf("\n");

  printf("Done!\n");
  return 0;
}