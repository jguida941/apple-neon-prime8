#include "src/simd_fast.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cstring>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>

using namespace std::chrono;

// Deterministic Miller-Rabin for 32-bit numbers
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

// Convert bitmap to index list for better cache behavior
std::vector<uint32_t> bitmap_to_indices(const uint8_t* bitmap, const uint64_t* numbers, size_t count) {
    std::vector<uint32_t> survivors;
    survivors.reserve(count / 4); // Expect ~26.7% survival worst case

    for (size_t i = 0; i < count; i++) {
        if ((bitmap[i/8] >> (i%8)) & 1) {
            if (numbers[i] <= 0xFFFFFFFF) {
                survivors.push_back(numbers[i]);
            }
        }
    }

    return survivors;
}

// Adaptive filter - samples first to determine optimal depth
struct AdaptiveConfig {
    int prime_depth;  // How many primes to check (3-16)
    bool use_wheel;   // Whether to use wheel-30
};

AdaptiveConfig determine_config(const uint64_t* numbers, size_t sample_size = 1000) {
    // Quick sample to estimate composite density
    size_t even_count = 0;
    size_t small_composite = 0;

    for (size_t i = 0; i < sample_size && i < 1000; i++) {
        uint32_t n = numbers[i];
        if (n <= 0xFFFFFFFF) {
            if (n % 2 == 0) even_count++;
            if (n % 3 == 0 || n % 5 == 0) small_composite++;
        }
    }

    double even_ratio = (double)even_count / sample_size;
    double composite_ratio = (double)small_composite / sample_size;

    AdaptiveConfig config;

    // Heavy composite (>80% even) -> minimal filtering
    if (even_ratio > 0.8) {
        config.prime_depth = 5;   // Just check 2,3,5,7,11
        config.use_wheel = false; // Skip wheel overhead
    }
    // Moderate composite -> standard filtering
    else if (composite_ratio > 0.5) {
        config.prime_depth = 8;   // Check up to prime #8
        config.use_wheel = true;
    }
    // Random/prime-heavy -> aggressive filtering
    else {
        config.prime_depth = 16;  // Check all 16 primes
        config.use_wheel = true;
    }

    return config;
}

// Block sieving implementation for cache efficiency
void block_sieve(const uint64_t* numbers, uint8_t* bitmap, size_t count,
                 const uint32_t* primes, int prime_count) {
    const size_t BLOCK_SIZE = 65536; // L2 cache friendly

    for (size_t block_start = 0; block_start < count; block_start += BLOCK_SIZE) {
        size_t block_end = std::min(block_start + BLOCK_SIZE, count);

        // For each prime, mark all multiples in this block
        for (int p_idx = 0; p_idx < prime_count; p_idx++) {
            uint32_t p = primes[p_idx];

            for (size_t i = block_start; i < block_end; i++) {
                if (numbers[i] <= 0xFFFFFFFF) {
                    uint32_t n = numbers[i];
                    if (n != p && n % p == 0) {
                        bitmap[i/8] &= ~(1 << (i%8));
                    }
                }
            }
        }
    }
}

// Producer-consumer threading model
struct WorkItem {
    std::vector<uint32_t> numbers;
    size_t batch_id;
};

class PipelineThreaded {
private:
    std::queue<WorkItem> work_queue;
    std::mutex queue_mutex;
    std::condition_variable cv;
    std::atomic<bool> done{false};
    std::atomic<size_t> primes_found{0};

public:
    void producer(const uint64_t* numbers, size_t count) {
        const size_t BATCH_SIZE = 65536;
        size_t batch_id = 0;

        for (size_t i = 0; i < count; i += BATCH_SIZE) {
            size_t batch_end = std::min(i + BATCH_SIZE, count);

            // Run SIMD filter on this batch
            size_t batch_size = batch_end - i;
            std::vector<uint8_t> bitmap((batch_size + 7) / 8);
            neon_wheel::filter_stream_u64_wheel_bitmap(
                numbers + i, bitmap.data(), batch_size);

            // Convert bitmap to index list
            WorkItem item;
            item.numbers = bitmap_to_indices(bitmap.data(), numbers + i, batch_size);
            item.batch_id = batch_id++;

            // Queue work for consumers
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                work_queue.push(std::move(item));
            }
            cv.notify_one();
        }

        done = true;
        cv.notify_all();
    }

    void consumer() {
        while (true) {
            WorkItem item;

            // Get work from queue
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                cv.wait(lock, [this] { return !work_queue.empty() || done; });

                if (work_queue.empty() && done) break;

                item = std::move(work_queue.front());
                work_queue.pop();
            }

            // Process survivors with Miller-Rabin
            size_t local_primes = 0;
            for (uint32_t n : item.numbers) {
                if (miller_rabin_32(n)) {
                    local_primes++;
                }
            }

            primes_found += local_primes;
        }
    }

    size_t run(const uint64_t* numbers, size_t count, int num_threads = 4) {
        primes_found = 0;
        done = false;

        // Start producer thread
        std::thread prod(&PipelineThreaded::producer, this, numbers, count);

        // Start consumer threads
        std::vector<std::thread> consumers;
        for (int i = 0; i < num_threads; i++) {
            consumers.emplace_back(&PipelineThreaded::consumer, this);
        }

        // Wait for completion
        prod.join();
        for (auto& t : consumers) {
            t.join();
        }

        return primes_found;
    }
};

struct PipelineStats {
    size_t total_numbers;
    size_t filtered_out;
    size_t survivors;
    size_t mr_calls;
    size_t confirmed_primes;
    double ms_filter;
    double ms_mr;
    double ms_total;
    AdaptiveConfig config;
};

// Adaptive pipeline with all optimizations
PipelineStats pipeline_adaptive(const std::vector<uint64_t>& numbers) {
    PipelineStats stats{};
    stats.total_numbers = numbers.size();

    auto start = high_resolution_clock::now();

    // Step 1: Determine optimal configuration
    stats.config = determine_config(numbers.data());

    // Step 2: SIMD filter with adaptive depth
    auto filter_start = high_resolution_clock::now();

    size_t bitmap_size = (numbers.size() + 7) / 8;
    std::vector<uint8_t> bitmap(bitmap_size, 0xFF); // Start with all set

    if (stats.config.use_wheel) {
        neon_wheel::filter_stream_u64_wheel_bitmap(
            numbers.data(), bitmap.data(), numbers.size());
    } else {
        // Simple divisibility check for composite-heavy
        for (size_t i = 0; i < numbers.size(); i++) {
            if (numbers[i] <= 0xFFFFFFFF) {
                uint32_t n = numbers[i];
                if (n % 2 == 0 && n != 2) {
                    bitmap[i/8] &= ~(1 << (i%8));
                }
            }
        }
    }

    auto filter_end = high_resolution_clock::now();
    stats.ms_filter = duration<double, std::milli>(filter_end - filter_start).count();

    // Step 3: Convert bitmap to index list (better cache behavior)
    std::vector<uint32_t> survivors = bitmap_to_indices(
        bitmap.data(), numbers.data(), numbers.size());
    stats.survivors = survivors.size();
    stats.filtered_out = numbers.size() - survivors.size();

    // Step 4: Miller-Rabin on compacted survivors
    auto mr_start = high_resolution_clock::now();

    for (uint32_t n : survivors) {
        stats.mr_calls++;
        if (miller_rabin_32(n)) {
            stats.confirmed_primes++;
        }
    }

    auto mr_end = high_resolution_clock::now();
    stats.ms_mr = duration<double, std::milli>(mr_end - mr_start).count();

    auto end = high_resolution_clock::now();
    stats.ms_total = duration<double, std::milli>(end - start).count();

    return stats;
}

void print_stats(const std::string& name, const PipelineStats& s) {
    double throughput_total = s.total_numbers / s.ms_total / 1000.0;
    double survival_rate = 100.0 * s.survivors / s.total_numbers;
    double prime_rate = 100.0 * s.confirmed_primes / s.total_numbers;

    std::cout << name << ":\n";
    std::cout << "  Config:          depth=" << s.config.prime_depth
              << ", wheel=" << (s.config.use_wheel ? "yes" : "no") << "\n";
    std::cout << "  Total time:      " << std::fixed << std::setprecision(3)
              << s.ms_total << " ms (" << throughput_total << " M/s)\n";
    std::cout << "  Filter time:     " << s.ms_filter << " ms\n";
    std::cout << "  MR time:         " << s.ms_mr << " ms\n";
    std::cout << "  Survival rate:   " << survival_rate << "%\n";
    std::cout << "  MR calls:        " << s.mr_calls << "\n";
    std::cout << "  Confirmed primes: " << s.confirmed_primes
              << " (" << prime_rate << "%)\n";
}

int main() {
    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "         ADAPTIVE PIPELINE WITH OPTIMIZATIONS\n";
    std::cout << "================================================================================\n\n";

    std::vector<std::pair<std::string, std::vector<uint64_t>>> datasets;

    // 1. Random 32-bit
    {
        std::mt19937_64 rng(42);
        std::uniform_int_distribution<uint64_t> dist(1, 0xFFFFFFFF);
        std::vector<uint64_t> random_data(1000000);
        for (auto& n : random_data) n = dist(rng);
        datasets.push_back({"Random 32-bit (1M)", std::move(random_data)});
    }

    // 2. Composite-heavy
    {
        std::vector<uint64_t> comp_data(1000000);
        for (size_t i = 0; i < comp_data.size(); i++) {
            comp_data[i] = (i + 1) * 2;
            if (i % 10 == 0) comp_data[i] = (i + 1) * 2 + 1;
        }
        datasets.push_back({"Composite-heavy (1M)", std::move(comp_data)});
    }

    // 3. Prime-rich (odd numbers)
    {
        std::vector<uint64_t> prime_rich(100000);
        for (size_t i = 0; i < prime_rich.size(); i++) {
            prime_rich[i] = i * 2 + 1;
        }
        datasets.push_back({"Prime-rich odds (100K)", std::move(prime_rich)});
    }

    // Run benchmarks
    for (const auto& [name, data] : datasets) {
        std::cout << "DATASET: " << name << "\n";
        std::cout << std::string(70, '-') << "\n";

        // Warm up
        for (int i = 0; i < 3; i++) {
            pipeline_adaptive(data);
        }

        // Benchmark adaptive pipeline
        auto stats = pipeline_adaptive(data);
        print_stats("Adaptive Pipeline", stats);

        // Test threaded version
        std::cout << "\nThreaded Pipeline (4 threads):\n";
        auto threaded_start = high_resolution_clock::now();

        PipelineThreaded threaded;
        size_t threaded_primes = threaded.run(data.data(), data.size(), 4);

        auto threaded_end = high_resolution_clock::now();
        double threaded_ms = duration<double, std::milli>(threaded_end - threaded_start).count();
        double threaded_throughput = data.size() / threaded_ms / 1000.0;

        std::cout << "  Total time:      " << std::fixed << std::setprecision(3)
                  << threaded_ms << " ms (" << threaded_throughput << " M/s)\n";
        std::cout << "  Confirmed primes: " << threaded_primes << "\n";

        // Calculate speedup vs single-threaded adaptive
        std::cout << "\nSPEEDUP (threaded vs adaptive): " << std::setprecision(2)
                  << stats.ms_total / threaded_ms << "x\n";

        std::cout << "\n" << std::string(70, '=') << "\n\n";
    }

    return 0;
}