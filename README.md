# Apple NEON Prime Prefilter

This repository contains SIMD-accelerated 32-bit prime prefilters for Apple
Silicon (ARM64) using NEON intrinsics. The fast paths implement wheel-30 and
wheel-210 filters followed by Barrett reductions against the first 16 primes.
Release builds are tuned for Apple M-series CPUs (`-Ofast -funroll-loops -arch
arm64`).

## Project Status (March 2025)

- ✅ **Wheel-30**: Production-ready. SIMD byte and bitmap variants match the
  scalar reference and deliver ~0.37 Gnum/s on random data (M4).
- ⚠️ **Wheel-210 (efficient)**: Known bug. The current NEON kernel drops
  legitimate survivors; keep using wheel-30 until the fix lands.
- ⚠️ **Hybrid (SIMD + GMP) benchmark**: The existing Python harness is
  subprocess-bound. A native C++ benchmark that runs NEON + GMP in-process is
  being developed (`hybrid_bench` task).

Running `./build/test_wheel210` currently reports mismatches (expected while the
wheel-210 fix is in progress).

The latest performance tables live in `BENCHMARK_RESULTS.md` and
`PERFORMANCE.md`, which now include notes about the wheel-210 investigation and
hybrid HTTP.

## Building

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Key binaries:

- `build/bench` – NEON vs scalar benchmarks (uniform + mixed datasets)
- `build/correctness` – exhaustive stress tests against scalar reference
- `build/test_wheel210` – unit test comparing wheel-30 vs wheel-210 vs scalar
- `build/demo` – quick throughput demo (scalar vs wheel-30 vs wheel-210)
- `bench/bench_comparison`, `bench_wheel`, `bench_final_complete` – additional
  standalone benchmarks

## Reproducing Python Comparisons

Ensure `numpy` and `gmpy2` are installed, then run:

```bash
python3 bench/bench_python.py
python3 bench/bench_gmpy2.py
# wheel-30 only (wheel-210 disabled until fixed)
python3 bench/bench_hybrid.py
```

## Roadmap

1. **Fix wheel-210 NEON kernel** (`simd_wheel210_efficient.cpp`)
   - Add unit tests that compare wheel-210 SIMD vs scalar survivors
   - Audit the mod-7 stage and lane masking logic
2. **Introduce `hybrid_bench` C++ target**
   - Link against GMP and run NEON + GMP verification in-process
   - Expose the throughput numbers in documentation
3. Optional: Provide Pybind/Cython bindings once the C++ pipeline is stable

## License

Apache License 2.0 — see `LICENSE` for details.

## Demos & Diagnostics

- Run the SIMD demo (scalar vs wheel-30). Wheel-210 output is currently
  disabled pending the kernel fix:
  ```bash
  ./build/demo 10000000
  ```
  The demo executes a single pass that counts survivors, so the reported
  throughput is lower (~0.25 Gnum/s on random data) than the sustained rate in
  the full benchmarks.
- For headline numbers run the release benchmark harness instead:
  ```bash
  ./build/bench
  ./bench/bench_comparison
  ```
  These print the ~0.37 Gnum/s byte-path and ~0.26 Gnum/s wheel-30 throughput
  captured in `BENCHMARK_RESULTS.md`.
- Run wheel-30/wheel-210 vs scalar unit test (expected to fail for wheel-210
  until the bug is fixed):
  ```bash
  ./build/test_wheel210 100000
  ```
  The program prints the first few mismatching values when wheel-210 diverges.
- A helper driver (`bench/hybrid_driver.cpp`) exists for the Python hybrid
  benchmark; it currently emits wheel-30 survivors and will be replaced once the
  in-process C++ hybrid benchmark is ready.
