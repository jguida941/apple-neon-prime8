#!/usr/bin/env python3
"""
Benchmark gmpy2 (GMP) prime testing
"""

import numpy as np
import time
import sys

try:
    import gmpy2
    HAS_GMPY2 = True
except ImportError:
    print("Installing gmpy2...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gmpy2"])
    import gmpy2
    HAS_GMPY2 = True

def benchmark_gmpy2_isprime(numbers, iterations=10):
    """Benchmark gmpy2.is_prime() - full primality test"""
    # Warmup
    for _ in range(2):
        [gmpy2.is_prime(int(n)) for n in numbers]

    start = time.perf_counter()
    for _ in range(iterations):
        result = [gmpy2.is_prime(int(n)) for n in numbers]
    end = time.perf_counter()

    duration = end - start
    throughput = (len(numbers) * iterations) / duration / 1e9
    latency = duration / (len(numbers) * iterations) * 1e9

    return throughput, latency

def benchmark_gmpy2_probab_prime(numbers, iterations=10):
    """Benchmark gmpy2.is_probab_prime() - Miller-Rabin test"""
    # Warmup
    for _ in range(2):
        [gmpy2.is_probab_prime(int(n)) for n in numbers]

    start = time.perf_counter()
    for _ in range(iterations):
        result = [gmpy2.is_probab_prime(int(n)) for n in numbers]
    end = time.perf_counter()

    duration = end - start
    throughput = (len(numbers) * iterations) / duration / 1e9
    latency = duration / (len(numbers) * iterations) * 1e9

    return throughput, latency

def benchmark_gmpy2_next_prime(numbers, iterations=10):
    """Benchmark gmpy2.next_prime() for comparison"""
    # Just find next prime after first number (for reference)
    n = int(numbers[0])

    start = time.perf_counter()
    for _ in range(iterations * len(numbers)):
        p = gmpy2.next_prime(n)
        n = p
    end = time.perf_counter()

    duration = end - start
    throughput = (len(numbers) * iterations) / duration / 1e9
    latency = duration / (len(numbers) * iterations) * 1e9

    return throughput, latency

def main():
    print("\n" + "="*70)
    print("     GMP/GMPY2 PRIME TESTING BENCHMARK")
    print("     Comparing against SIMD Wheel-30: 1.35 Gnum/s")
    print("="*70 + "\n")

    # Test sizes
    sizes = [1024, 16384, 65536]

    for size in sizes:
        print(f"DATASET: {size} numbers (32-bit random)")
        print("-"*60)

        # Generate random 32-bit numbers
        np.random.seed(42)
        numbers = np.random.randint(1, 2**32, size, dtype=np.uint64)

        # Benchmark gmpy2.is_prime (deterministic)
        isprime_tp, isprime_lat = benchmark_gmpy2_isprime(numbers,
                                                          iterations=10 if size <= 16384 else 2)

        # Benchmark gmpy2.is_probab_prime (Miller-Rabin)
        probab_tp, probab_lat = benchmark_gmpy2_probab_prime(numbers,
                                                             iterations=10 if size <= 16384 else 2)

        # Print results
        print(f"{'Method':<35} {'Throughput':>12} {'Latency':>10}")
        print("-"*60)
        print(f"{'gmpy2.is_prime (deterministic)':<35} {isprime_tp:>9.6f} Gn/s {isprime_lat:>8.0f} ns")
        print(f"{'gmpy2.is_probab_prime (M-R)':<35} {probab_tp:>9.6f} Gn/s {probab_lat:>8.0f} ns")

        print("\nOUR SIMD IMPLEMENTATIONS:")
        print(f"{'SIMD Wheel-30 (small primes only)':<35} {'1.350':>9} Gn/s {'0.74':>6} ns")
        print(f"{'SIMD Ultra Barrett-16':<35} {'0.250':>9} Gn/s {'4.00':>6} ns")

        print("\nSPEEDUP COMPARISON:")
        print(f"  SIMD Wheel-30 vs gmpy2.is_prime:        {1.35/isprime_tp:>6.1f}x faster")
        print(f"  SIMD Wheel-30 vs gmpy2.is_probab_prime: {1.35/probab_tp:>6.1f}x faster")

        print("\nIMPORTANT NOTE:")
        print("  - gmpy2 provides FULL primality testing (deterministic or probabilistic)")
        print("  - Our SIMD is a PREFILTER (trial division against 16 primes only)")
        print("  - Different use cases: gmpy2 for correctness, SIMD for speed filtering")
        print()

if __name__ == "__main__":
    main()