cmake_minimum_required(VERSION 3.20)
project(apple_neon_prime8 CXX)
set(CMAKE_CXX_STANDARD 20)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(APPLE)
  add_compile_options(-Ofast -fstrict-aliasing -funroll-loops -arch arm64 -fno-lto)
  # If your toolchain supports it, enable:
  # add_compile_options(-mcpu=apple-m4)
endif()

add_library(prime8 STATIC
  src/simd_fast.cpp
  src/simd_optimized.cpp
  src/simd_ultra_fast.cpp
  src/simd_wheel.cpp
  src/simd_wheel210.cpp
  src/simd_wheel210_efficient.cpp
  src/simd_final.cpp
)
target_include_directories(prime8 PUBLIC src)

add_executable(bench bench/bench.cpp)
target_link_libraries(bench PRIVATE prime8)

add_executable(correctness test/correctness.cpp)
target_link_libraries(correctness PRIVATE prime8)

add_executable(test_wheel210
  test/test_wheel210.cpp
)
target_link_libraries(test_wheel210 PRIVATE prime8)

add_executable(demo bench/demo.cpp)
target_link_libraries(demo PRIVATE prime8)
