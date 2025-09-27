# Apple NEON Prime8 Project Structure

## Directory Tree & File Descriptions

```
apple-neon-prime8/
├── src/                         # Core implementation files
│   ├── primes_tables.hpp        # Precomputed prime tables and constants
│   ├── simd_fast.cpp           # Fast SIMD prime filtering implementation
│   ├── simd_fast.hpp           # Fast SIMD headers and interfaces
│   ├── simd_final.cpp          # Final optimized SIMD implementation
│   ├── simd_optimized.cpp      # Optimized SIMD prime filtering
│   ├── simd_ultra_fast.cpp     # Ultra-fast variant with aggressive opts
│   ├── simd_wheel.cpp          # Wheel-30 factorization implementation
│   ├── simd_wheel210.cpp       # Wheel-210 factorization implementation
│   └── simd_wheel210_efficient.cpp # Efficient wheel-210 variant
│
├── bench/                       # Benchmark implementations
│   ├── bench.cpp               # Main benchmark harness
│   ├── bench_all.cpp           # Comprehensive benchmark suite
│   ├── bench_block_sieve.cpp   # Block sieve algorithm benchmark
│   ├── bench_comparison.cpp    # Method comparison benchmarks
│   ├── bench_final.cpp         # Final implementation benchmark
│   ├── bench_final_complete.cpp # Complete final benchmark suite
│   ├── bench_fixed.cpp         # Fixed-size benchmark tests
│   ├── bench_optimized.cpp     # Optimized version benchmarks
│   ├── bench_pipeline.cpp      # Pipeline architecture benchmark
│   ├── bench_pipeline_adaptive.cpp # Adaptive pipeline benchmark
│   ├── bench_ultra.cpp         # Ultra-fast implementation benchmark
│   ├── bench_wheel.cpp         # Wheel factorization benchmarks
│   ├── bench_working.cpp       # Working/experimental benchmarks
│   ├── bench_gmpy2.py          # Python GMP2 comparison
│   ├── bench_hybrid.py         # Hybrid Python/C++ benchmark
│   └── bench_python.py         # Pure Python baseline benchmark
│
├── test/                        # Test suite
│   ├── comprehensive_test.cpp  # Full test suite runner
│   ├── correctness.cpp         # Correctness validation tests
│   ├── debug_test.cpp          # Debug and diagnostic tests
│   ├── test_bitmap16.cpp       # 16-bit bitmap operations test
│   ├── test_bitpack.cpp        # Bit packing/unpacking tests
│   ├── test_debug_filter.cpp   # Filter debugging tests
│   ├── test_filter_order.cpp   # Filter ordering tests
│   ├── test_filter_primes.cpp  # Prime filter correctness tests
│   ├── test_filter_simple.cpp  # Simple filter tests
│   ├── test_fixes.cpp          # Bug fix regression tests
│   ├── test_mod30.cpp          # Modulo-30 wheel tests
│   ├── test_movemask_debug.cpp # NEON movemask debug tests
│   ├── test_movemask_fix.cpp   # Movemask fix validation
│   ├── test_simple.cpp         # Simple functionality tests
│   ├── test_ultra.cpp          # Ultra-fast implementation tests
│   ├── test_wheel.cpp          # Wheel factorization tests
│   └── test_wheel210.cpp       # Wheel-210 specific tests
│
├── tests/                       # Additional test resources
├── build/                       # Build artifacts
├── cmake-build-debug/           # CMake debug build
├── bootstrap.sh                 # Build bootstrap script
└── comprehensive_test           # Main test executable
```

## Core Components

### 1. SIMD Implementations (src/)
- **Wheel Factorization**: Multiple wheel sizes (30, 210) for initial filtering
- **SIMD Processing**: ARM NEON vectorized prime checking
- **Optimization Levels**: Fast → Optimized → Ultra → Final implementations

### 2. Benchmarks (bench/)
- **Method Comparisons**: Direct comparison of different algorithms
- **Pipeline Tests**: Producer-consumer and adaptive pipeline architectures
- **Language Comparisons**: Python baselines vs C++ SIMD implementations

### 3. Tests (test/)
- **Correctness**: Validates prime generation accuracy
- **Performance**: Ensures optimizations don't break functionality
- **Debug Tools**: Specialized tests for NEON operations and bit manipulation

## Performance Results Summary

### Method Comparison (10M range, random pattern)
1. **Method 1 (Wheel-30)**: filter ≈ 5.7ms, total 42.98ms - Solid baseline
2. **Method 2 (Block Sieve)**: filter ≈ 11.8ms, total 48.48ms - Needs optimization
3. **Method 3 (Bitmap→Index)**: filter ≈ 7.4ms, total 42.96ms - Best MR performance

### Key Findings
- Index-list handoff (Method 3) improves Miller-Rabin by reducing working set
- Block sieve needs cache-aware tuning and mark-by-step approach
- Sequential access patterns significantly faster than random (≈20% improvement)

## Optimization Strategy

### Immediate Actions
1. **Keep bitmap→index handoff** as default for Miller-Rabin
2. **Rework block sieve** with:
   - 64-128 KiB blocks for cache efficiency
   - Mark-by-step instead of per-number checks
   - Bitset operations (8× less memory traffic)
   - NEON vectorized marking for common strides

### Advanced Optimizations
1. **Adaptive filtering depth** based on survival rate sampling
2. **Pipeline parallelism**: Filter on one thread, MR on 2-4 threads
3. **Cache optimization**: Aligned arrays and prefetching
4. **Dynamic prime selection**: Adjust small prime count based on survival rate

### Block Sieve Improvements Needed
- Use bitsets instead of byte arrays
- Implement stride-based marking (avoid per-element Barrett)
- Precompute first multiple per block
- Consider prime-pair interleaving for ILP
- Integrate with wheel factorization (skip 73% immediately)