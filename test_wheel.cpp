#include <cstdio>
#include <vector>

namespace neon_wheel {
  void filter_stream_u64_wheel(const uint64_t* __restrict, uint8_t* __restrict, size_t);
}

int main() {
  // Test with known primes and composites
  std::vector<uint64_t> test = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    59, 61, 67, 71, 73, 79, 83, 89, 97, 101 // Large primes
  };

  std::vector<uint8_t> out(30);
  neon_wheel::filter_stream_u64_wheel(test.data(), out.data(), 30);

  printf("Testing wheel filter:\n");
  for (size_t i = 0; i < 30; ++i) {
    printf("%llu -> %d\n", (unsigned long long)test[i], out[i]);
  }

  int correct_primes = 0;
  // Expected primes: 2,3,5,7,11,13,17,19,59,61,67,71,73,79,83,89,97,101
  if (out[1]) correct_primes++; // 2
  if (out[2]) correct_primes++; // 3
  if (out[4]) correct_primes++; // 5
  if (out[6]) correct_primes++; // 7
  if (out[10]) correct_primes++; // 11
  if (out[12]) correct_primes++; // 13
  if (out[16]) correct_primes++; // 17
  if (out[18]) correct_primes++; // 19

  printf("\nFound %d primes in first 20\n", correct_primes);

  return 0;
}