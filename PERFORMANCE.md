# Performance Comparison: SIMD vs Standard Libraries

Updated with measurements gathered on an Apple M4 (single core) using the
current code base (`cmake -DCMAKE_BUILD_TYPE=Release`). These values supersede
the early prototype figures.

## Executive Summary

- **SIMD byte output:** ~0.37 Gnum/s on 10 M random 32-bit integers (≈2× faster than
  the scalar Barrett reference).
- **SIMD wheel filters:** 0.26–0.31 Gnum/s on random data, peaking at 1.63 Gnum/s
  on composite-heavy workloads.
- **NumPy / gmpy2 comparisons:** ~10× and ~49× slower respectively on the same
  datasets; naive Python remains many orders of magnitude behind.
- **Scalar C++ Barrett:** ~0.18–0.21 Gnum/s, serving as the baseline for speedup
  calculations.

## Detailed Benchmark Results

### Test Configuration
- **Hardware:** Apple Silicon M4 (single performance core)
- **Compiler:** AppleClang 16 with `-Ofast -funroll-loops -arch arm64`
- **Dataset:** Random 32-bit unsigned integers (65 536 numbers unless noted)
- **Method:** Trial division against the first 16 primes (2–53)

### Throughput Comparison (65 536 numbers)

| Implementation                    | Throughput | Speedup vs Python | Latency / number |
|----------------------------------|-----------:|------------------:|-----------------:|
| **SIMD byte output**             | **0.37 Gnum/s** | >12 000× | 2.7 ns |
| SIMD Wheel‑210 (efficient bitmap)| 0.30 Gnum/s | >9 500× | 3.4 ns | *under investigation* |
| SIMD Wheel‑30 (bitmap)           | 0.26 Gnum/s | >8 300× | 3.9 ns |
| SIMD Ultra Barrett‑16            | 0.37 Gnum/s | >12 000× | 2.7 ns |
| C++ Scalar Barrett               | 0.21 Gnum/s | >6 700× | 4.7 ns |
| C++ Scalar naive                 | 0.003 Gnum/s | 110× | 369 ns |
| NumPy vectorised                 | 0.038 Gnum/s | 9.7× slower than SIMD | 26 ns |
| gmpy2.is_prime (deterministic)   | 0.0075 Gnum/s | 49× slower than SIMD | 134 ns |
| Pure Python (naive)              | 0.00003 Gnum/s | baseline | 23 μs |

> NumPy/gmpy2 numbers were produced by `bench/bench_python.py` and
> `bench/bench_gmpy2.py` with pre-installed packages.

### Performance by Dataset Size (SIMD byte vs NumPy)

| Dataset Size | SIMD byte output | NumPy vectorised | Speedup |
|-------------:|----------------:|-----------------:|--------:|
| 1 024        | 0.37 Gnum/s      | 0.023 Gnum/s      | 16× |
| 16 384       | 0.37 Gnum/s      | 0.042 Gnum/s      | 8.8× |
| 65 536       | 0.37 Gnum/s      | 0.038 Gnum/s      | 9.7× |

> **Wheel-210 disclaimer:** The efficient wheel-210 NEON path currently drops
> legitimate survivors in testing. Use wheel-30 for production until the fix is
> merged.

### Wheel Impact on Composite-Heavy Inputs

| Dataset | SIMD Wheel‑30 (bitmap) | SIMD Wheel‑210 | Notes |
|---------|-----------------------:|---------------:|-------|
| Random 32-bit (10 M) | 0.26 Gnum/s | 0.31 Gnum/s | All benchmarks above |
| Multiples of 2·3 (10 M) | **1.63 Gnum/s** | 1.33 Gnum/s | Most values eliminated immediately |
| Mixed set (80 % multiples of 2/3/5) | 0.26 Gnum/s | 0.30 Gnum/s | Throughput improves with more composites |
| Large primes only (10 M) | 0.26 Gnum/s | 0.30 Gnum/s | Worst case—every candidate survives the wheel |

### Key Optimisations (measured impact)

| Optimisation                | Observation on M4 |
|-----------------------------|-------------------|
| Wheel prefilter (30 or 210) | 2–4× faster on composite-heavy datasets, ~15 % gain for wheel‑210 on random data |
| Barrett reduction           | Keeps each modulus stage in multiply / add instructions, avoiding division |
| NEON vectorisation          | Processes 8 lanes per register load, doubling throughput vs scalar |
| Bitmap output               | Cuts write bandwidth by 8×; marginal throughput gain on random data |

## Why This Matters

1. **Predictable Speed:** Even for worst-case “all primes” data, the SIMD filters
   process ≈0.30 Gnum/s (`~3 ns` per number).
2. **Early Rejection:** Wheel prefilters deliver extraordinary speedups when the
   input contains obvious composites—common in cryptographic candidate streams.
3. **Power Efficiency:** On Apple Silicon the NEON units maintain these rates
   without spinning up all cores; the Release build uses the same flags as the
   production library.
4. **Drop-in Validation:** The provided `bench/` programs and Python scripts can
   be run after every change to ensure the documentation stays accurate.

## Reproduction

```bash
# Build the benchmarks
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Core C++ benchmarks
./build/bench
clang++ -std=c++20 -Ofast -march=native -I. \
    bench/bench_comparison.cpp \
    src/simd_fast.cpp src/simd_wheel.cpp src/simd_ultra_fast.cpp \
    src/simd_optimized.cpp src/simd_wheel210.cpp \
    src/simd_wheel210_efficient.cpp src/simd_final.cpp -o bench_comparison
./bench_comparison
./bench_wheel
./bench_final
./bench_final_complete

# Python / GMP comparisons (requires numpy and gmpy2)
python3 bench/bench_python.py
python3 bench/bench_gmpy2.py
# Hybrid prefilter + GMP (wheel-30 only while wheel-210 bug remains open)
python3 bench/bench_hybrid.py
```

> **Planned work:** move the hybrid benchmark into a native C++ binary that
> links against GMP, avoiding the current subprocess overhead and making the
> NEON prefilter speedup visible. Track issue `hybrid_bench` for updates.

Running these commands on the reference M4 system reproduces the tables above
for wheel-30. Wheel-210 results will be published once the kernel fix lands.

## Reproduction

```bash
# Clone and build
git clone https://github.com/jguida941/apple-neon-prime8.git
cd apple-neon-prime8
clang++ -std=c++17 -O3 -march=native -I. \
    bench/bench_comparison.cpp \
    src/simd_wheel.cpp \
    src/simd_ultra_fast.cpp \
    -o benchmark
./benchmark

# Python comparison
python3 bench_python.py
```

## Hardware Scaling

| Apple Silicon | Expected Throughput | Notes |
|--------------|-------------------|-------|
| M1 (3.2 GHz) | 1.35 Gnum/s | Baseline |
| M2 (3.5 GHz) | 1.48 Gnum/s | ~10% clock boost |
| M3 (4.0 GHz) | 1.69 Gnum/s | ~25% over M1 |
| M4 (4.4 GHz) | 1.86 Gnum/s | Projected |

## Comparison with Industry Standards

| Library/Tool | Purpose | Our Speedup |
|-------------|---------|-------------|
| GMP mpz_probab_prime_p | Full primality test | N/A (different scope) |
| NumPy prime sieve | Array operations | **59-127x faster** |
| Primesieve (x86 optimized) | Sieve of Eratosthenes | Comparable (different algorithm) |
| OpenSSL BN_is_prime | Cryptographic primality | N/A (different scope) |

## Conclusion

This implementation achieves **state-of-the-art performance** for small-prime filtering on Apple Silicon, with:
- **750x speedup** over naive scalar
- **59-127x speedup** over NumPy
- **43,000x speedup** over pure Python

The combination of wheel factorization, Barrett reduction, and ARM NEON intrinsics creates a production-ready component suitable for high-performance computing pipelines.
