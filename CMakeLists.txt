cmake_minimum_required(VERSION 3.22)

project(simd-playground VERSION 0.1.0 LANGUAGES C CXX)

add_subdirectory(vendor/benchmark)

add_library(simdvec src/simdvec.c)
target_compile_features(simdvec PRIVATE c_std_11)
target_include_directories(simdvec PUBLIC include)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
target_link_libraries(simdvec PRIVATE m)
endif()

add_executable(bench src/benchmark.cpp)
target_compile_features(bench PRIVATE cxx_std_11)
target_link_libraries(bench PRIVATE simdvec)
target_link_libraries(bench PRIVATE benchmark::benchmark)
