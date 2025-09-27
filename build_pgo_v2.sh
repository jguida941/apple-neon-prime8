#!/usr/bin/env bash
set -euo pipefail

# ---- Config ----
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SRC_DIR/build_pgo"
BIN_INSTR="$BUILD_DIR/bench_instr"
BIN_OPT="$BUILD_DIR/bench_pgo"
PROFDIR="$BUILD_DIR/profiles"
PROFDATA="$BUILD_DIR/code.profdata"

# Use system clang
CXX="clang++"
PROFTOOL="llvm-profdata"

echo "[*] Using CXX=${CXX}"
echo "[*] Using PROFTOOL=${PROFTOOL}"

# Clean and recreate directories
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR" "$PROFDIR"

# Common sources
SRCS=(
  "$SRC_DIR/bench_ultra.cpp"
  "$SRC_DIR/src/simd_fast.cpp"
  "$SRC_DIR/src/simd_ultra_fast.cpp"
)
INCLUDES=(-I"$SRC_DIR" -I"$SRC_DIR/src")

# ---- 1) Build instrumented binary ----
echo "[*] Building instrumented binary…"
"$CXX" -O3 -std=c++17 -march=native \
  -fno-exceptions -fno-rtti -fstrict-aliasing -funroll-loops \
  -fprofile-instr-generate="$PROFDIR/bench-%p-%m.profraw" \
  "${INCLUDES[@]}" \
  -o "$BIN_INSTR" "${SRCS[@]}"

# ---- 2) Train profiles ----
echo "[*] Running training workloads…"

# Run the instrumented binary to generate profiles
"$BIN_INSTR" > /dev/null 2>&1 || true

# ---- 3) Merge profiles ----
echo "[*] Merging profiles…"
"$PROFTOOL" merge -output="$PROFDATA" "$PROFDIR"/*.profraw

# ---- 4) Build optimized with profile use + ThinLTO ----
echo "[*] Building PGO+ThinLTO optimized binary…"
"$CXX" -O3 -std=c++17 -march=native -flto=thin \
  -fno-exceptions -fno-rtti -fstrict-aliasing -funroll-loops \
  -fprofile-instr-use="$PROFDATA" \
  "${INCLUDES[@]}" \
  -o "$BIN_OPT" "${SRCS[@]}"

# ---- 5) Benchmark comparison ----
echo ""
echo "=== Performance Comparison (10M elements) ==="
echo ""

echo "[Regular O3 build]"
"$SRC_DIR/bench_ultra" 2>&1 | grep -A4 "N = 10000000" | head -5

echo ""
echo "[PGO+LTO optimized build]"
"$BIN_OPT" 2>&1 | grep -A4 "N = 10000000" | head -5

echo ""
echo "=== Summary ==="
echo "Regular build results above ↑"
echo "PGO+LTO build results above ↑"