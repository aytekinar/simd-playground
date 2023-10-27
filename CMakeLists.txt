cmake_minimum_required(VERSION 3.22)

project(simd-playground VERSION 0.1.0 LANGUAGES C CXX)

add_subdirectory(vendor/benchmark)

add_library(simdvec src/simdvec.c)
target_compile_features(simdvec PRIVATE c_std_11)
target_include_directories(simdvec PUBLIC include)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
target_link_libraries(simdvec PRIVATE m)
endif()

add_library(simdvecnoattr src/simdvec-attribute.c)
target_compile_definitions(simdvecnoattr PRIVATE NO_ATTRIBUTE)
target_compile_features(simdvecnoattr PRIVATE c_std_11)
target_include_directories(simdvecnoattr PUBLIC include)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
target_link_libraries(simdvecnoattr PRIVATE m)
endif()

add_library(simdvecattr src/simdvec-attribute.c)
target_compile_features(simdvecattr PRIVATE c_std_11)
target_include_directories(simdvecattr PUBLIC include)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
target_link_libraries(simdvecattr PRIVATE m)
endif()

add_executable(bench src/benchmark.cpp)
target_compile_features(bench PRIVATE cxx_std_11)
target_link_libraries(bench PRIVATE simdvec)
target_link_libraries(bench PRIVATE benchmark::benchmark)

add_executable(bench-noattr src/benchmark-attribute.cpp)
target_compile_features(bench-noattr PRIVATE cxx_std_11)
target_link_libraries(bench-noattr PRIVATE simdvecnoattr)
target_link_libraries(bench-noattr PRIVATE benchmark::benchmark)

add_executable(bench-attr src/benchmark-attribute.cpp)
target_compile_features(bench-attr PRIVATE cxx_std_11)
target_link_libraries(bench-attr PRIVATE simdvecattr)
target_link_libraries(bench-attr PRIVATE benchmark::benchmark)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  find_package(GTest REQUIRED)
  enable_testing()
  add_subdirectory(tests)
endif()
