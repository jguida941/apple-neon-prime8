#!/bin/bash

echo "=== Building with Profile-Guided Optimization (PGO) ==="

# Step 1: Build with instrumentation
echo "Step 1: Building instrumented binary..."
clang++ -O3 -march=native -std=c++17 -fprofile-generate=./pgo_data \
  -I. -o bench_pgo_gen bench_ultra.cpp src/simd_fast.cpp src/simd_ultra_fast.cpp

# Step 2: Run to generate profile data
echo "Step 2: Generating profile data..."
mkdir -p pgo_data
./bench_pgo_gen > /dev/null 2>&1

# Step 3: Build with profile data
echo "Step 3: Building optimized binary with PGO..."
clang++ -O3 -march=native -std=c++17 -fprofile-use=./pgo_data \
  -I. -o bench_pgo bench_ultra.cpp src/simd_fast.cpp src/simd_ultra_fast.cpp

# Step 4: Build with LTO + PGO
echo "Step 4: Building with LTO + PGO..."
clang++ -O3 -march=native -std=c++17 -flto=thin -fprofile-use=./pgo_data \
  -I. -o bench_lto_pgo bench_ultra.cpp src/simd_fast.cpp src/simd_ultra_fast.cpp

echo "=== Running benchmarks ==="

echo -e "\n--- Regular O3 build ---"
./bench_ultra 2>&1 | grep -A2 "N = 10000000"

echo -e "\n--- PGO build ---"
./bench_pgo 2>&1 | grep -A2 "N = 10000000"

echo -e "\n--- LTO+PGO build ---"
./bench_lto_pgo 2>&1 | grep -A2 "N = 10000000"