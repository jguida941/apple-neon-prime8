# Apple Silicon NEON Prime Filter - Ultra Performance Edition 🚀

High-performance SIMD prime filtering using ARM NEON intrinsics, optimized for Apple Silicon (M1/M2/M3). Achieves **1.35+ billion numbers/second** throughput using wheel factorization and Barrett reduction.

## 🎯 Performance Results

| Implementation | Throughput | Use Case |
|---------------|------------|----------|
| **Wheel-30 + Bitmap** | **1.35-1.36 Gnum/s** | ⚡ Fastest - Composite-heavy workloads |
| Ultra Barrett-16 | 0.25 Gnum/s | Uniform/random distributions |
| Standard Barrett | 0.15-0.20 Gnum/s | Baseline implementation |

## 🏗️ Architecture

### Three Optimization Levels

1. **Wheel-30 Prefilter** (`simd_wheel.cpp`) - **FASTEST**
   - Eliminates 73% of candidates using wheel factorization (2×3×5=30)
   - Only tests 8 residues: {1,7,11,13,17,19,23,29} mod 30
   - Bitmap output (1 bit per number) for memory efficiency
   - Best for composite-heavy inputs

2. **Ultra Barrett** (`simd_ultra_fast.cpp`)
   - 16-wide SIMD processing with quad-Barrett reduction
   - Interleaved prime constants for pipeline optimization
   - Steady performance regardless of input distribution

3. **Optimized Wheel** (`simd_optimized.cpp`)
   - All optimizations combined
   - Fast modulo for scalar tails
   - Tuned prefetch distances
   - Alignment hints for better codegen

## 📦 Building

### Requirements
- Apple Silicon Mac (M1/M2/M3)
- Clang 12+ or GCC 11+
- C++17 support

### Quick Build
```bash
# Clone the repository
git clone https://github.com/yourusername/apple-neon-prime8.git
cd apple-neon-prime8

# Build all implementations
clang++ -std=c++17 -O3 -march=native -I. \
    bench_optimized.cpp \
    src/simd_wheel.cpp \
    src/simd_ultra_fast.cpp \
    src/simd_optimized.cpp \
    -o prime_filter

# Run benchmarks
./prime_filter
```

## 🚀 Usage

### Fastest Path - Wheel-30 with Bitmap Output
```cpp
#include "src/simd_fast.hpp"

// Process 1M numbers
std::vector<uint64_t> numbers(1000000);
// ... fill with your data ...

// Allocate bitmap (1 bit per number)
std::vector<uint8_t> bitmap((1000000 + 7) / 8);

// Filter at 1.35+ Gnum/s\!
neon_wheel::filter_stream_u64_wheel_bitmap(
    numbers.data(),
    bitmap.data(),
    1000000
);

// Check results
for (size_t i = 0; i < 1000000; i++) {
    bool is_probable_prime = (bitmap[i/8] >> (i%8)) & 1;
    // ...
}
```
