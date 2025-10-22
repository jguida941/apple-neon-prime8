#include "simd_fast.hpp"
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr std::array<uint32_t, 16> kSmallPrimes = {
    2u, 3u, 5u, 7u, 11u, 13u, 17u, 19u, 23u, 29u, 31u, 37u, 41u, 43u, 47u, 53u
};

bool scalar_survives(uint32_t n) {
    if (n <= 1) return false;
    for (uint32_t p : kSmallPrimes) {
        if (n == p) return true;
        if (n % p == 0) return false;
    }
    return true;
}

std::vector<uint32_t> scalar_filter(const std::vector<uint64_t>& nums) {
    std::vector<uint32_t> survivors;
    survivors.reserve(nums.size());
    for (uint64_t v : nums) {
        uint32_t n = static_cast<uint32_t>(v);
        if (scalar_survives(n)) survivors.push_back(n);
    }
    return survivors;
}

std::vector<uint32_t> simd_filter_w30(const std::vector<uint64_t>& nums) {
    const size_t n = nums.size();
    std::vector<uint8_t> bitmap((n + 7) / 8);
    neon_wheel::filter_stream_u64_wheel_bitmap(nums.data(), bitmap.data(), n);
    std::vector<uint32_t> survivors;
    survivors.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (bitmap[i >> 3] & (1u << (i & 7))) survivors.push_back(static_cast<uint32_t>(nums[i]));
    }
    return survivors;
}

std::vector<uint32_t> simd_filter_w210(const std::vector<uint64_t>& nums) {
    const size_t n = nums.size();
    std::vector<uint8_t> bitmap((n + 7) / 8);
    neon_wheel210_efficient::filter_stream_u64_wheel210_efficient_bitmap(
        nums.data(), bitmap.data(), n);
    std::vector<uint32_t> survivors;
    survivors.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (bitmap[i >> 3] & (1u << (i & 7))) survivors.push_back(static_cast<uint32_t>(nums[i]));
    }
    return survivors;
}

struct Mismatch {
    size_t index;
    uint32_t value;
};

std::vector<Mismatch> diff(const std::vector<uint32_t>& reference,
                           const std::vector<uint32_t>& observed) {
    std::vector<Mismatch> mismatches;
    std::vector<uint32_t> ref = reference;
    std::vector<uint32_t> obs = observed;
    std::sort(ref.begin(), ref.end());
    std::sort(obs.begin(), obs.end());
    size_t i = 0, j = 0;
    while (i < ref.size() || j < obs.size()) {
        uint32_t rv = (i < ref.size()) ? ref[i] : UINT32_MAX;
        uint32_t ov = (j < obs.size()) ? obs[j] : UINT32_MAX;
        if (rv == ov) {
            ++i; ++j;
        } else if (rv < ov) {
            mismatches.push_back({i, rv});
            ++i;
        } else {
            mismatches.push_back({j, ov});
            ++j;
        }
    }
    return mismatches;
}

}  // namespace

int main(int argc, char** argv) {
    size_t count = 1'000'000;
    if (argc > 1) count = std::strtoull(argv[1], nullptr, 10);

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> dist(0, 0xffffffffu);

    std::vector<uint64_t> nums(count);
    for (auto& v : nums) v = dist(rng);

    auto scalar = scalar_filter(nums);
    auto simd30 = simd_filter_w30(nums);
    auto simd210 = simd_filter_w210(nums);

    auto mm30 = diff(scalar, simd30);
    if (!mm30.empty()) {
        std::fprintf(stderr, "wheel-30 mismatch count: %zu\n", mm30.size());
        return 1;
    }

    auto mm210 = diff(scalar, simd210);
    if (!mm210.empty()) {
        std::fprintf(stderr, "wheel-210 mismatch count: %zu\n", mm210.size());
        size_t report = std::min<size_t>(mm210.size(), 20);
        std::fprintf(stderr, "First mismatches:\n");
        for (size_t i = 0; i < report; ++i) {
            std::fprintf(stderr, "  value=%u\n", mm210[i].value);
        }
        return 2;
    }

    std::printf("wheel-30 and wheel-210 match scalar for %zu numbers\n", count);
    return 0;
}
