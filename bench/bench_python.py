#!/usr/bin/env python3
"""
Benchmark NumPy and standard Python prime filtering
for comparison with our SIMD implementation
"""

import numpy as np
import time
import sys

def naive_is_prime(n):
    """Naive Python prime check"""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def numpy_prime_check(numbers):
    """NumPy vectorized prime checking (still using modulo)"""
    result = np.ones(len(numbers), dtype=bool)

    # Handle special cases
    result[numbers <= 1] = False
    result[numbers == 2] = True
    result[numbers == 3] = True

    # Check divisibility by 2 and 3
    result[numbers % 2 == 0] = False
    result[numbers % 3 == 0] = False
    result[numbers == 2] = True  # 2 is prime
    result[numbers == 3] = True  # 3 is prime

    # Check other potential divisors
    for p in [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]:
        result[numbers % p == 0] = False
        result[numbers == p] = True

    return result

def sieve_of_eratosthenes(max_val):
    """Classic sieve for comparison"""
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(max_val**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False

    return is_prime

def benchmark_method(name, func, numbers, iterations=10):
    """Benchmark a prime filtering method"""
    # Warmup
    for _ in range(2):
        func(numbers)

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        result = func(numbers)
    end = time.perf_counter()

    duration = end - start
    throughput = (len(numbers) * iterations) / duration / 1e9
    latency = duration / (len(numbers) * iterations) * 1e9

    return throughput, latency

def main():
    print("\n" + "="*60)
    print("     PYTHON/NUMPY PRIME FILTER BENCHMARK")
    print("="*60 + "\n")

    # Test sizes
    sizes = [1024, 16384, 65536]

    for size in sizes:
        print(f"DATASET: {size} numbers (32-bit random)")
        print("-"*50)

        # Generate random 32-bit numbers
        np.random.seed(42)
        numbers = np.random.randint(1, 2**32, size, dtype=np.uint64)

        # Python list for native Python
        numbers_list = numbers.tolist()

        # Benchmark Python naive
        python_func = lambda nums: [naive_is_prime(n) for n in nums]
        py_throughput, py_latency = benchmark_method(
            "Python (naive)", python_func, numbers_list, iterations=1
        )

        # Benchmark NumPy
        np_throughput, np_latency = benchmark_method(
            "NumPy vectorized", numpy_prime_check, numbers, iterations=10
        )

        # Benchmark Sieve (if numbers fit)
        max_val = int(numbers.max())
        if max_val < 10_000_000:  # Only for smaller ranges
            sieve_func = lambda nums: sieve_of_eratosthenes(max_val)
            sieve_throughput, sieve_latency = benchmark_method(
                "Sieve of Eratosthenes", sieve_func, numbers, iterations=10
            )
        else:
            sieve_throughput, sieve_latency = 0, float('inf')

        # Print results
        print(f"{'Method':<25} {'Throughput':>12} {'Latency':>12}")
        print("-"*50)
        print(f"{'Python (naive)':<25} {py_throughput:>9.4f} Gn/s {py_latency:>9.0f} ns")
        print(f"{'NumPy vectorized':<25} {np_throughput:>9.4f} Gn/s {np_latency:>9.0f} ns")
        if sieve_throughput > 0:
            print(f"{'Sieve (pre-computed)':<25} {sieve_throughput:>9.4f} Gn/s {sieve_latency:>9.0f} ns")

        print("\nCOMPARISON WITH SIMD:")
        print("  SIMD Wheel-30:  1.35 Gnum/s (0.74 ns/num)")
        print(f"  NumPy speedup:  {1.35/np_throughput:.1f}x faster than NumPy")
        print(f"  Python speedup: {1.35/py_throughput:.1f}x faster than Python\n")
        print()

if __name__ == "__main__":
    main()