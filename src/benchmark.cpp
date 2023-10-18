#include <iostream>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>

#include "simdvec.h"

static void BM_add(benchmark::State &state) {
  const auto func_id = state.range(0);
  const auto func = func_id == 2 && is_sse_supported()       ? add_sse
                    : func_id == 3 && is_avx_supported()     ? add_avx
                    : func_id == 4 && is_avx512f_supported() ? add_avx512f
                                                             : add;

  const auto dim = state.range(1);
  std::mt19937 gen(42);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> x(dim), y(dim), z(dim);
  std::generate(x.begin(), x.end(), [&]() { return dist(gen); });
  std::generate(y.begin(), y.end(), [&]() { return dist(gen); });

  for (auto _ : state) {
    func(dim, x.data(), y.data(), z.data());
    benchmark::ClobberMemory();
  }
}

BENCHMARK(BM_add)->ArgsProduct({
    benchmark::CreateDenseRange(1, 4, 1),
    benchmark::CreateRange(1 << 10, 16 * (1 << 10), 2),
});

static void BM_sub(benchmark::State &state) {
  const auto func_id = state.range(0);
  const auto func = func_id == 2 && is_sse_supported()       ? sub_sse
                    : func_id == 3 && is_avx_supported()     ? sub_avx
                    : func_id == 4 && is_avx512f_supported() ? sub_avx512f
                                                             : sub;

  const auto dim = state.range(1);
  std::mt19937 gen(42);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> x(dim), y(dim), z(dim);
  std::generate(x.begin(), x.end(), [&]() { return dist(gen); });
  std::generate(y.begin(), y.end(), [&]() { return dist(gen); });

  for (auto _ : state) {
    func(dim, x.data(), y.data(), z.data());
    benchmark::ClobberMemory();
  }
}

BENCHMARK(BM_sub)->ArgsProduct({
    benchmark::CreateDenseRange(1, 4, 1),
    benchmark::CreateRange(1 << 10, 16 * (1 << 10), 2),
});

static void BM_dot_product(benchmark::State &state) {
  const auto func_id = state.range(0);
  const auto func = func_id == 2 && is_sse_supported()   ? dot_product_sse
                    : func_id == 3 && is_avx_supported() ? dot_product_avx
                    : func_id == 4 && is_avx512f_supported()
                        ? dot_product_avx512f
                        : dot_product;

  const auto dim = state.range(1);
  std::mt19937 gen(42);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> x(dim), y(dim);
  std::generate(x.begin(), x.end(), [&]() { return dist(gen); });
  std::generate(y.begin(), y.end(), [&]() { return dist(gen); });

  for (auto _ : state) {
    benchmark::DoNotOptimize(func(dim, x.data(), y.data()));
    benchmark::ClobberMemory();
  }
}

BENCHMARK(BM_dot_product)
    ->ArgsProduct({
        benchmark::CreateDenseRange(1, 4, 1),
        benchmark::CreateRange(1 << 10, 16 * (1 << 10), 2),
    });

static void BM_cosine_distance(benchmark::State &state) {
  const auto func_id = state.range(0);
  const auto func = func_id == 2 && is_sse_supported()   ? cosine_similarity_sse
                    : func_id == 3 && is_avx_supported() ? cosine_similarity_avx
                    : func_id == 4 && is_avx512f_supported()
                        ? cosine_similarity_avx512f
                        : cosine_similarity;

  const auto dim = state.range(1);
  std::mt19937 gen(42);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> x(dim), y(dim);
  std::generate(x.begin(), x.end(), [&]() { return dist(gen); });
  std::generate(y.begin(), y.end(), [&]() { return dist(gen); });

  for (auto _ : state) {
    benchmark::DoNotOptimize(func(dim, x.data(), y.data()));
    benchmark::ClobberMemory();
  }
}

BENCHMARK(BM_cosine_distance)
    ->ArgsProduct({
        benchmark::CreateDenseRange(1, 4, 1),
        benchmark::CreateRange(1 << 10, 16 * (1 << 10), 2),
    });

static void BM_l1_distance(benchmark::State &state) {
  const auto func_id = state.range(0);
  const auto func = func_id == 2 && is_sse_supported()   ? l1_distance_sse
                    : func_id == 3 && is_avx_supported() ? l1_distance_avx
                    : func_id == 4 && is_avx512f_supported()
                        ? l1_distance_avx512f
                        : l1_distance;

  const auto dim = state.range(1);
  std::mt19937 gen(42);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> x(dim), y(dim);
  std::generate(x.begin(), x.end(), [&]() { return dist(gen); });
  std::generate(y.begin(), y.end(), [&]() { return dist(gen); });

  for (auto _ : state) {
    benchmark::DoNotOptimize(func(dim, x.data(), y.data()));
    benchmark::ClobberMemory();
  }
}

BENCHMARK(BM_l1_distance)
    ->ArgsProduct({
        benchmark::CreateDenseRange(1, 4, 1),
        benchmark::CreateRange(1 << 10, 16 * (1 << 10), 2),
    });

static void BM_l1_norm(benchmark::State &state) {
  const auto func_id = state.range(0);
  const auto func = func_id == 2 && is_sse_supported()       ? l1_norm_sse
                    : func_id == 3 && is_avx_supported()     ? l1_norm_avx
                    : func_id == 4 && is_avx512f_supported() ? l1_norm_avx512f
                                                             : l1_norm;

  const auto dim = state.range(1);
  std::mt19937 gen(42);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> x(dim);
  std::generate(x.begin(), x.end(), [&]() { return dist(gen); });

  for (auto _ : state) {
    benchmark::DoNotOptimize(func(dim, x.data()));
    benchmark::ClobberMemory();
  }
}

BENCHMARK(BM_l1_norm)
    ->ArgsProduct({
        benchmark::CreateDenseRange(1, 4, 1),
        benchmark::CreateRange(1 << 10, 16 * (1 << 10), 2),
    });

static void BM_l2_distance(benchmark::State &state) {
  const auto func_id = state.range(0);
  const auto func = func_id == 2 && is_sse_supported()   ? l2_distance_sse
                    : func_id == 3 && is_avx_supported() ? l2_distance_avx
                    : func_id == 4 && is_avx512f_supported()
                        ? l2_distance_avx512f
                        : l2_distance;

  const auto dim = state.range(1);
  std::mt19937 gen(42);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> x(dim), y(dim);
  std::generate(x.begin(), x.end(), [&]() { return dist(gen); });
  std::generate(y.begin(), y.end(), [&]() { return dist(gen); });

  for (auto _ : state) {
    benchmark::DoNotOptimize(func(dim, x.data(), y.data()));
    benchmark::ClobberMemory();
  }
}

BENCHMARK(BM_l2_distance)
    ->ArgsProduct({
        benchmark::CreateDenseRange(1, 4, 1),
        benchmark::CreateRange(1 << 10, 16 * (1 << 10), 2),
    });

static void BM_l2_norm(benchmark::State &state) {
  const auto func_id = state.range(0);
  const auto func = func_id == 2 && is_sse_supported()       ? l2_norm_sse
                    : func_id == 3 && is_avx_supported()     ? l2_norm_avx
                    : func_id == 4 && is_avx512f_supported() ? l2_norm_avx512f
                                                             : l2_norm;

  const auto dim = state.range(1);
  std::mt19937 gen(42);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> x(dim);
  std::generate(x.begin(), x.end(), [&]() { return dist(gen); });

  for (auto _ : state) {
    benchmark::DoNotOptimize(func(dim, x.data()));
    benchmark::ClobberMemory();
  }
}

BENCHMARK(BM_l2_norm)
    ->ArgsProduct({
        benchmark::CreateDenseRange(1, 4, 1),
        benchmark::CreateRange(1 << 10, 16 * (1 << 10), 2),
    });

BENCHMARK_MAIN();
