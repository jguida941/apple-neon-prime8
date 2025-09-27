#include <cstdio>
#include <vector>
#include <chrono>

namespace neon_ultra {
  void filter_stream_u64_barrett16_ultra(const uint64_t* __restrict, uint8_t* __restrict, size_t);
}

int main() {
  printf("Testing ultra version...\n");

  const size_t N = 1000000;
  std::vector<uint64_t> test(N);
  for (size_t i = 0; i < N; ++i) test[i] = i + 1;

  std::vector<uint8_t> out(N);

  auto t0 = std::chrono::high_resolution_clock::now();
  neon_ultra::filter_stream_u64_barrett16_ultra(test.data(), out.data(), N);
  auto t1 = std::chrono::high_resolution_clock::now();

  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  double throughput = (N / 1e6) / (ms / 1000.0);

  printf("Ultra: %.3f ms, %.1f Mnums/s\n", ms, throughput);

  // Sample check
  int survivors = 0;
  for (size_t i = 0; i < 100; ++i) {
    if (out[i]) survivors++;
  }
  printf("First 100: %d survivors\n", survivors);

  return 0;
}