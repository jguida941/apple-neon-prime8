#include "simd_fast.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

struct Result {
    double seconds{};
    size_t survivors{};
};

Result run_scalar(const std::vector<uint64_t>& numbers) {
    auto start = Clock::now();
    size_t survivors = 0;
    for (uint32_t n : numbers) {
        if (n <= 1) continue;
        bool ok = true;
        for (uint32_t p : {2u,3u,5u,7u,11u,13u,17u,19u,23u,29u,31u,37u,41u,43u,47u,53u}) {
            if (n == p) { ok = true; break; }
            if (n % p == 0) { ok = false; break; }
        }
        if (ok) ++survivors;
    }
    auto stop = Clock::now();
    double secs = std::chrono::duration<double>(stop - start).count();
    return {secs, survivors};
}

Result run_wheel30(const std::vector<uint64_t>& numbers) {
    const size_t n = numbers.size();
    std::vector<uint8_t> bitmap((n + 7) / 8);
    auto start = Clock::now();
    neon_wheel::filter_stream_u64_wheel_bitmap(numbers.data(), bitmap.data(), n);
    auto stop = Clock::now();
    size_t survivors = 0;
    for (size_t i = 0; i < n; ++i) {
        if (bitmap[i >> 3] & (1u << (i & 7))) ++survivors;
    }
    double secs = std::chrono::duration<double>(stop - start).count();
    return {secs, survivors};
}

int main(int argc, char** argv) {
    size_t count = 10'000'000;
    if (argc > 1) count = std::strtoull(argv[1], nullptr, 10);

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint32_t> dist(0, 0xffffffffu);
    std::vector<uint64_t> numbers(count);
    for (auto& v : numbers) v = dist(rng);

    auto scalar = run_scalar(numbers);
    auto w30    = run_wheel30(numbers);

    std::printf("Dataset size: %zu\n", count);
    std::printf("Scalar Barrett:   %.3f ms, throughput %.2f Mnums/s, survivors %zu\n",
                scalar.seconds * 1e3,
                (count / 1e6) / scalar.seconds,
                scalar.survivors);
    std::printf("Wheel-30 bitmap:  %.3f ms, throughput %.2f Mnums/s, survivors %zu\n",
                w30.seconds * 1e3,
                (count / 1e6) / w30.seconds,
                w30.survivors);
    std::puts("Wheel-210 bitmap: [disabled pending kernel fix]");

    return 0;
}
