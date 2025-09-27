#include "src/simd_fast.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cstring>
#include <arm_neon.h>

using namespace std::chrono;

// Block sieving with SIMD - process one prime at a time across the whole block
// This is cache-friendly: each pass touches each cache line once
class BlockSieveSIMD {
private:
    static const size_t BLOCK_SIZE = 65536; // L2 cache friendly
    static constexpr uint32_t PRIMES[16] = {
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53
    };

public:
    // Sieve a block using SIMD - one prime at a time
    static void sieve_block_simd(const uint64_t* numbers, uint8_t* bitmap,
                                  size_t start, size_t end, uint32_t prime) {
        const uint32x4_t p = vdupq_n_u32(prime);
        const uint32x4_t zero = vdupq_n_u32(0);

        // Barrett constant for this prime
        const uint64_t mu = (0x100000000ULL + prime - 1) / prime;
        const uint32x4_t mu_vec = vdupq_n_u32(mu);

        for (size_t i = start; i < end; i += 16) {
            if (i + 16 > end) break; // Handle in scalar tail

            // Load 16 numbers
            uint32_t nums[16];
            for (int j = 0; j < 16; j++) {
                nums[j] = (numbers[i+j] <= 0xFFFFFFFF) ? numbers[i+j] : 0;
            }

            // Process in 4 vectors
            uint32x4_t n1 = vld1q_u32(nums + 0);
            uint32x4_t n2 = vld1q_u32(nums + 4);
            uint32x4_t n3 = vld1q_u32(nums + 8);
            uint32x4_t n4 = vld1q_u32(nums + 12);

            // Compute n % prime using Barrett
            uint64x2_t lo1 = vmull_u32(vget_low_u32(n1), vget_low_u32(mu_vec));
            uint64x2_t hi1 = vmull_u32(vget_high_u32(n1), vget_high_u32(mu_vec));
            uint32x4_t q1 = vcombine_u32(vshrn_n_u64(lo1, 32), vshrn_n_u64(hi1, 32));
            uint32x4_t r1 = vsubq_u32(n1, vmulq_u32(q1, p));

            uint64x2_t lo2 = vmull_u32(vget_low_u32(n2), vget_low_u32(mu_vec));
            uint64x2_t hi2 = vmull_u32(vget_high_u32(n2), vget_high_u32(mu_vec));
            uint32x4_t q2 = vcombine_u32(vshrn_n_u64(lo2, 32), vshrn_n_u64(hi2, 32));
            uint32x4_t r2 = vsubq_u32(n2, vmulq_u32(q2, p));

            uint64x2_t lo3 = vmull_u32(vget_low_u32(n3), vget_low_u32(mu_vec));
            uint64x2_t hi3 = vmull_u32(vget_high_u32(n3), vget_high_u32(mu_vec));
            uint32x4_t q3 = vcombine_u32(vshrn_n_u64(lo3, 32), vshrn_n_u64(hi3, 32));
            uint32x4_t r3 = vsubq_u32(n3, vmulq_u32(q3, p));

            uint64x2_t lo4 = vmull_u32(vget_low_u32(n4), vget_low_u32(mu_vec));
            uint64x2_t hi4 = vmull_u32(vget_high_u32(n4), vget_high_u32(mu_vec));
            uint32x4_t q4 = vcombine_u32(vshrn_n_u64(lo4, 32), vshrn_n_u64(hi4, 32));
            uint32x4_t r4 = vsubq_u32(n4, vmulq_u32(q4, p));

            // Check divisibility (r == 0 and n != p)
            uint32x4_t div1 = vandq_u32(vceqq_u32(r1, zero), vmvnq_u32(vceqq_u32(n1, p)));
            uint32x4_t div2 = vandq_u32(vceqq_u32(r2, zero), vmvnq_u32(vceqq_u32(n2, p)));
            uint32x4_t div3 = vandq_u32(vceqq_u32(r3, zero), vmvnq_u32(vceqq_u32(n3, p)));
            uint32x4_t div4 = vandq_u32(vceqq_u32(r4, zero), vmvnq_u32(vceqq_u32(n4, p)));

            // Clear bits for divisible numbers - extract manually with constant indices
            uint16_t clear_mask = 0;
            if (vgetq_lane_u32(div1, 0)) clear_mask |= (1 << 0);
            if (vgetq_lane_u32(div1, 1)) clear_mask |= (1 << 1);
            if (vgetq_lane_u32(div1, 2)) clear_mask |= (1 << 2);
            if (vgetq_lane_u32(div1, 3)) clear_mask |= (1 << 3);
            if (vgetq_lane_u32(div2, 0)) clear_mask |= (1 << 4);
            if (vgetq_lane_u32(div2, 1)) clear_mask |= (1 << 5);
            if (vgetq_lane_u32(div2, 2)) clear_mask |= (1 << 6);
            if (vgetq_lane_u32(div2, 3)) clear_mask |= (1 << 7);
            if (vgetq_lane_u32(div3, 0)) clear_mask |= (1 << 8);
            if (vgetq_lane_u32(div3, 1)) clear_mask |= (1 << 9);
            if (vgetq_lane_u32(div3, 2)) clear_mask |= (1 << 10);
            if (vgetq_lane_u32(div3, 3)) clear_mask |= (1 << 11);
            if (vgetq_lane_u32(div4, 0)) clear_mask |= (1 << 12);
            if (vgetq_lane_u32(div4, 1)) clear_mask |= (1 << 13);
            if (vgetq_lane_u32(div4, 2)) clear_mask |= (1 << 14);
            if (vgetq_lane_u32(div4, 3)) clear_mask |= (1 << 15);

            // Apply to bitmap
            size_t byte_idx = i / 8;
            uint16_t current;
            std::memcpy(&current, bitmap + byte_idx, sizeof(current));
            current &= ~clear_mask;
            std::memcpy(bitmap + byte_idx, &current, sizeof(current));
        }

        // Handle tail
        for (size_t i = (end & ~15); i < end; i++) {
            if (numbers[i] <= 0xFFFFFFFF) {
                uint32_t n = numbers[i];
                if (n != prime && n % prime == 0) {
                    bitmap[i/8] &= ~(1 << (i%8));
                }
            }
        }
    }

    // Full block sieve with all primes
    static void sieve_block(const uint64_t* numbers, uint8_t* bitmap,
                            size_t count, int num_primes = 16) {
        // Initialize bitmap to all 1s (all potentially prime)
        std::memset(bitmap, 0xFF, (count + 7) / 8);

        // Special handling for 2, 3, 5
        for (size_t i = 0; i < count; i++) {
            if (numbers[i] <= 0xFFFFFFFF) {
                uint32_t n = numbers[i];
                if (n != 2 && n % 2 == 0) {
                    bitmap[i/8] &= ~(1 << (i%8));
                } else if (n != 3 && n % 3 == 0) {
                    bitmap[i/8] &= ~(1 << (i%8));
                } else if (n != 5 && n % 5 == 0) {
                    bitmap[i/8] &= ~(1 << (i%8));
                }
            }
        }

        // Process blocks for remaining primes
        for (size_t block_start = 0; block_start < count; block_start += BLOCK_SIZE) {
            size_t block_end = std::min(block_start + BLOCK_SIZE, count);

            // One pass per prime (cache-friendly)
            for (int p_idx = 3; p_idx < num_primes; p_idx++) {
                sieve_block_simd(numbers, bitmap, block_start, block_end, PRIMES[p_idx]);
            }
        }
    }
};

// Miller-Rabin implementation (same as before)
bool miller_rabin_32(uint32_t n) {
    if (n < 2) return false;
    if (n == 2 || n == 3) return true;
    if ((n & 1) == 0) return false;

    uint32_t d = n - 1;
    int r = 0;
    while ((d & 1) == 0) {
        d >>= 1;
        r++;
    }

    const uint32_t witnesses[] = {2, 7, 61};
    for (uint32_t a : witnesses) {
        if (a >= n) continue;

        uint64_t x = 1;
        uint64_t base = a;
        uint32_t exp = d;

        while (exp > 0) {
            if (exp & 1) {
                x = (x * base) % n;
            }
            base = (base * base) % n;
            exp >>= 1;
        }

        if (x == 1 || x == n - 1) continue;

        bool composite = true;
        for (int i = 0; i < r - 1; i++) {
            x = (x * x) % n;
            if (x == n - 1) {
                composite = false;
                break;
            }
        }

        if (composite) return false;
    }

    return true;
}

struct PipelineStats {
    size_t total_numbers;
    size_t survivors;
    size_t confirmed_primes;
    double ms_filter;
    double ms_mr;
    double ms_total;
};

// Compare different sieving methods
void benchmark_methods(const std::vector<uint64_t>& numbers) {
    const size_t count = numbers.size();

    // Method 1: Original SIMD wheel
    {
        std::cout << "Method 1: Original SIMD Wheel-30\n";

        auto start = high_resolution_clock::now();
        std::vector<uint8_t> bitmap((count + 7) / 8);

        auto filter_start = high_resolution_clock::now();
        neon_wheel::filter_stream_u64_wheel_bitmap(numbers.data(), bitmap.data(), count);
        auto filter_end = high_resolution_clock::now();

        // Count survivors and run MR
        size_t survivors = 0, primes = 0;
        auto mr_start = high_resolution_clock::now();
        for (size_t i = 0; i < count; i++) {
            if ((bitmap[i/8] >> (i%8)) & 1) {
                survivors++;
                if (numbers[i] <= 0xFFFFFFFF && miller_rabin_32(numbers[i])) {
                    primes++;
                }
            }
        }
        auto end = high_resolution_clock::now();

        double filter_ms = duration<double, std::milli>(filter_end - filter_start).count();
        double mr_ms = duration<double, std::milli>(end - mr_start).count();
        double total_ms = duration<double, std::milli>(end - start).count();

        std::cout << "  Filter:    " << std::fixed << std::setprecision(3)
                  << filter_ms << " ms (" << count/filter_ms/1000 << " M/s)\n";
        std::cout << "  MR:        " << mr_ms << " ms\n";
        std::cout << "  Total:     " << total_ms << " ms\n";
        std::cout << "  Survivors: " << survivors << " ("
                  << 100.0*survivors/count << "%)\n";
        std::cout << "  Primes:    " << primes << "\n\n";
    }

    // Method 2: Block sieve
    {
        std::cout << "Method 2: Block Sieve (cache-friendly)\n";

        auto start = high_resolution_clock::now();
        std::vector<uint8_t> bitmap((count + 7) / 8);

        auto filter_start = high_resolution_clock::now();
        BlockSieveSIMD::sieve_block(numbers.data(), bitmap.data(), count);
        auto filter_end = high_resolution_clock::now();

        // Count survivors and run MR
        size_t survivors = 0, primes = 0;
        auto mr_start = high_resolution_clock::now();
        for (size_t i = 0; i < count; i++) {
            if ((bitmap[i/8] >> (i%8)) & 1) {
                survivors++;
                if (numbers[i] <= 0xFFFFFFFF && miller_rabin_32(numbers[i])) {
                    primes++;
                }
            }
        }
        auto end = high_resolution_clock::now();

        double filter_ms = duration<double, std::milli>(filter_end - filter_start).count();
        double mr_ms = duration<double, std::milli>(end - mr_start).count();
        double total_ms = duration<double, std::milli>(end - start).count();

        std::cout << "  Filter:    " << std::fixed << std::setprecision(3)
                  << filter_ms << " ms (" << count/filter_ms/1000 << " M/s)\n";
        std::cout << "  MR:        " << mr_ms << " ms\n";
        std::cout << "  Total:     " << total_ms << " ms\n";
        std::cout << "  Survivors: " << survivors << " ("
                  << 100.0*survivors/count << "%)\n";
        std::cout << "  Primes:    " << primes << "\n\n";
    }

    // Method 3: Compacted index list
    {
        std::cout << "Method 3: Bitmap → Index List (better cache)\n";

        auto start = high_resolution_clock::now();
        std::vector<uint8_t> bitmap((count + 7) / 8);

        auto filter_start = high_resolution_clock::now();
        neon_wheel::filter_stream_u64_wheel_bitmap(numbers.data(), bitmap.data(), count);

        // Convert to index list
        std::vector<uint32_t> survivor_list;
        survivor_list.reserve(count / 4);
        for (size_t i = 0; i < count; i++) {
            if ((bitmap[i/8] >> (i%8)) & 1 && numbers[i] <= 0xFFFFFFFF) {
                survivor_list.push_back(numbers[i]);
            }
        }
        auto filter_end = high_resolution_clock::now();

        // Run MR on compacted list (better cache behavior)
        size_t primes = 0;
        auto mr_start = high_resolution_clock::now();
        for (uint32_t n : survivor_list) {
            if (miller_rabin_32(n)) {
                primes++;
            }
        }
        auto end = high_resolution_clock::now();

        double filter_ms = duration<double, std::milli>(filter_end - filter_start).count();
        double mr_ms = duration<double, std::milli>(end - mr_start).count();
        double total_ms = duration<double, std::milli>(end - start).count();

        std::cout << "  Filter:    " << std::fixed << std::setprecision(3)
                  << filter_ms << " ms (" << count/filter_ms/1000 << " M/s)\n";
        std::cout << "  MR:        " << mr_ms << " ms\n";
        std::cout << "  Total:     " << total_ms << " ms\n";
        std::cout << "  Survivors: " << survivor_list.size() << " ("
                  << 100.0*survivor_list.size()/count << "%)\n";
        std::cout << "  Primes:    " << primes << "\n\n";
    }
}

int main() {
    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "                  BLOCK SIEVE vs ORIGINAL COMPARISON\n";
    std::cout << "================================================================================\n\n";

    // Test with different datasets
    std::vector<std::pair<std::string, std::vector<uint64_t>>> datasets;

    // Random data
    {
        std::mt19937_64 rng(42);
        std::uniform_int_distribution<uint64_t> dist(1, 0xFFFFFFFF);
        std::vector<uint64_t> random_data(1000000);
        for (auto& n : random_data) n = dist(rng);
        datasets.push_back({"Random 32-bit (1M)", std::move(random_data)});
    }

    // Sequential
    {
        std::vector<uint64_t> seq_data(1000000);
        for (size_t i = 0; i < seq_data.size(); i++) {
            seq_data[i] = i + 1000000;
        }
        datasets.push_back({"Sequential (1M)", std::move(seq_data)});
    }

    for (const auto& [name, data] : datasets) {
        std::cout << "DATASET: " << name << "\n";
        std::cout << std::string(70, '-') << "\n";

        // Warm up caches
        for (int i = 0; i < 3; i++) {
            std::vector<uint8_t> tmp((data.size() + 7) / 8);
            neon_wheel::filter_stream_u64_wheel_bitmap(data.data(), tmp.data(), data.size());
        }

        benchmark_methods(data);

        std::cout << std::string(70, '=') << "\n\n";
    }

    return 0;
}